#!/usr/bin/env python3
"""Stage exact row shards for DS41F dense TP2/TP4 and shared TP12.

Only bounded source ranges are copied. The original layout is never modified.
Metadata binds tensor ranges and checkpoint header hashes to the TP topology.
The shared expert can use a communicator independent of dense attention, so a
TP4 attention layout may carry TP12 shared-expert shards in the same rank
directory.
"""
import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path
from stage_backbone import inventory
from stage_dense import copy_tensor, dense_owner


def row_shard(item, first, rows):
    result = dict(item)
    shape = item['shape']
    if len(shape) != 2 or first < 0 or rows < 1 or first + rows > shape[0]:
        raise ValueError('invalid row range')
    stride = item['bytes'] // shape[0]
    result.update(shape=[rows, shape[1]], source_offset=item['source_offset'] + first * stride,
                  bytes=rows * stride, first_row=first, global_shape=shape)
    return result


def _range(rows, rank, degree, alignment=1):
    """Return an uneven-safe row range for a rank in a TP group."""
    if alignment < 1 or rows % alignment:
        raise ValueError('unaligned TP tensor')
    blocks = rows // alignment
    first = (blocks * rank // degree) * alignment
    end = (blocks * (rank + 1) // degree) * alignment
    return first, end


def _is_attention_shard(name):
    return any(part in name for part in
               ('.attn.wq_b.', '.attn.wo_a.', '.attn.wo_b.'))


def layout(model, rank, tp, shared_tp=None, attention_tp12=False):
    if tp not in (2, 4) or not 0 <= rank < 12:
        raise ValueError('requires rank 0..11 and dense TP2/TP4')
    if shared_tp is None:
        shared_tp = tp
    if shared_tp not in (2, 4, 12):
        raise ValueError('shared TP must be 2,4,12')
    items = [dict(x, reuse=True) for x in inventory(model, rank) if x['kind'] == 'expert']
    dense_inventory = list(inventory(model, 0))
    dense_by_name = {x['name']: x for x in dense_inventory}
    for original in dense_inventory:
        if original['kind'] != 'dense':
            continue
        name, owner = original['name'], dense_owner(original['name'])
        if name == 'head.weight':
            blocks = original['shape'][0] // 32
            first = (blocks * rank // 12) * 32
            end = (blocks * (rank + 1) // 12) * 32
            items.append(row_shard(original, first, end - first))
            continue
        is_shared = '.ffn.shared_experts.' in name
        is_attention = _is_attention_shard(name)
        sharded = is_attention or is_shared
        if sharded:
            # A TP12 shared communicator spans all ranks. Smaller groups keep
            # the existing owner-group placement used by dense attention.
            if is_attention and attention_tp12:
                # Q/WO-A operate on eight 8-head groups. Four ranks stay idle
                # for these matrices because 64 heads cannot be split evenly
                # over twelve ranks. WO-B is row-sharded over all twelve ranks.
                degree = 8 if ('.attn.wq_b.' in name or '.attn.wo_a.' in name) else 12
                member = rank < 8 if degree == 8 else True
            else:
                degree = shared_tp if is_shared else tp
                member = True if degree == 12 else rank // degree == owner // degree
            if member:
                rows = original['shape'][0]
                if not is_shared and not (attention_tp12 and is_attention) and rows % degree:
                    raise ValueError('unaligned TP tensor ' + name)
                alignment = 32 if ((is_shared or (attention_tp12 and is_attention and
                                                  '.attn.wo_b.' in name)) and
                                   original['dtype'] == 'F8_E4M3' and name.endswith('.weight')) else 1
                if (is_shared and original['dtype'] == 'F8_E8M0' and
                        name.endswith('.scale')):
                    # Scale rows describe 32-row FP8 output groups. Split
                    # them by the same group boundaries as the corresponding
                    # weight, otherwise an uneven W2 shard would fail the
                    # local weight/scale geometry check (14 vs 13 groups).
                    weight = dense_by_name.get(name[:-6] + '.weight')
                    if weight is None or weight['shape'][0] % 32:
                        raise ValueError('missing shared FP8 weight for ' + name)
                    wfirst, wend = _range(weight['shape'][0], rank % degree, degree, 32)
                    first, end = wfirst // 32, wend // 32
                    if end > rows:
                        raise ValueError('shared scale range exceeds tensor ' + name)
                elif (attention_tp12 and is_attention and original['dtype'] == 'F8_E8M0' and
                      name.endswith('.scale') and '.attn.wo_b.' in name):
                    weight = dense_by_name.get(name[:-6] + '.weight')
                    if weight is None or weight['shape'][0] % 32:
                        raise ValueError('missing attention FP8 weight for ' + name)
                    wfirst, wend = _range(weight['shape'][0], rank, degree, 32)
                    first, end = wfirst // 32, wend // 32
                    if end > rows:
                        raise ValueError('attention scale range exceeds tensor ' + name)
                else:
                    first, end = _range(rows, rank % degree, degree, alignment)
                items.append(row_shard(original, first, end - first))
        elif name.endswith('.attn.attn_sink') and attention_tp12 and rank < 8:
            items.append(dict(original, reuse=rank == owner))
        elif name.endswith('.attn.attn_sink') and rank // tp == owner // tp:
            items.append(dict(original, reuse=rank == owner))
        elif rank == owner:
            items.append(dict(original, reuse=True))
    return sorted(items, key=lambda x: x['name'])


def header_hashes(model):
    hashes = {}
    for path in sorted(model.glob('*.safetensors')):
        with path.open('rb') as f:
            prefix = f.read(8)
            n = struct.unpack('<Q', prefix)[0]
            hashes[path.name] = hashlib.sha256(prefix + f.read(n)).hexdigest()
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--original', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--shared-tp', type=int, choices=(2, 4, 12), default=None,
                    help='independent shared-expert degree (default: dense TP)')
    ap.add_argument('--attention-tp12', action='store_true',
                    help='stage attention Q/WO-A on eight head ranks and WO-B on all twelve')
    ap.add_argument('--tp', type=int, choices=(2, 4), required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if not 0 <= args.rank < 12:
        ap.error('rank must be 0..11')
    items = layout(args.model, args.rank, args.tp, args.shared_tp, args.attention_tp12)
    shared_tp = args.shared_tp if args.shared_tp is not None else args.tp
    split_manifest = shared_tp != args.tp or args.attention_tp12
    metadata = dict(version=1, tp=args.tp, shared_tp=shared_tp,
                    rank=args.rank, ranks=12,
                    header_sha256=header_hashes(args.model), items=items)
    if args.attention_tp12:
        metadata['attention_tp'] = 8
    print('TP_STAGE_PLAN tp={} rank={} tensors={} resident_bytes={} copied_bytes={}'.format(
        args.tp, args.rank, len(items), sum(x['bytes'] for x in items),
        sum(x['bytes'] for x in items if not x.get('reuse'))), flush=True)
    if args.dry_run:
        return
    dst = args.destination
    dst.mkdir(parents=True, exist_ok=True)
    manifest = dst / 'tp-manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError('existing TP manifest differs')
    else:
        with manifest.open('x') as f:
            json.dump(metadata, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
    for item in items:
        if item.get('reuse'):
            source = args.original / (item['name'] + '.bin')
            if source.stat().st_size != item['bytes']:
                raise ValueError('original tensor size mismatch')
            target = dst / source.name
            if not target.exists():
                target.symlink_to(source)
        else:
            copy_tensor(item, dst)
    # Reuse cold Engram payload and metadata without duplicating disk/HBM.
    for name in ['engram_meta.bin'] + ['layers.{}.engram.embed.{}.bin'.format(layer, part)
                                     for layer in (1, 14) for part in ('weight', 'scale')]:
        source = args.original / name
        if not source.is_file():
            raise ValueError('missing Engram source ' + str(source))
        if not (dst / name).exists():
            (dst / name).symlink_to(source)
    with (dst / 'weights.index.partial').open('w') as f:
        for item in items:
            shape = item['shape'] if len(item['shape']) == 2 else [1, item['shape'][0]]
            f.write('{} {} {} {} {}\n'.format(item['name'], item['dtype'], shape[0], shape[1], item['bytes']))
        f.flush(); os.fsync(f.fileno())
    dense_items = items if not split_manifest else [x for x in items
                                                     if '.ffn.shared_experts.' not in x['name'] and
                                                     not (args.attention_tp12 and
                                                          (_is_attention_shard(x['name']) or
                                                           x['name'].endswith('.attn.attn_sink')))]
    with (dst / 'weights.tp.partial').open('w') as f:
        f.write('DS41FTP 1 {} {} 12\n'.format(args.tp, args.rank))
        for item in dense_items:
            if 'global_shape' in item:
                f.write('{} {} {} {} {}\n'.format(item['name'], item['global_shape'][0],
                    item['global_shape'][1], item['first_row'], item['shape'][0]))
        f.flush(); os.fsync(f.fileno())
    os.replace(str(dst / 'weights.tp.partial'), str(dst / 'weights.tp'))
    if split_manifest:
        shared_items = [x for x in items if '.ffn.shared_experts.' in x['name']]
        with (dst / 'weights.shared.tp.partial').open('w') as f:
            f.write('DS41FSH 1 {} {} 12\n'.format(shared_tp, args.rank))
            for item in shared_items:
                if 'global_shape' in item:
                    f.write('{} {} {} {} {}\n'.format(item['name'], item['global_shape'][0],
                        item['global_shape'][1], item['first_row'], item['shape'][0]))
            f.flush(); os.fsync(f.fileno())
        os.replace(str(dst / 'weights.shared.tp.partial'), str(dst / 'weights.shared.tp'))
    if args.attention_tp12:
        attention_items = [x for x in items if _is_attention_shard(x['name']) or
                           x['name'].endswith('.attn.attn_sink')]
        with (dst / 'weights.attention.tp.partial').open('w') as f:
            f.write('DS41FA 1 8 {} 12\n'.format(args.rank))
            for item in attention_items:
                if 'global_shape' in item:
                    f.write('{} {} {} {} {}\n'.format(item['name'], item['global_shape'][0],
                            item['global_shape'][1], item['first_row'], item['shape'][0]))
            f.flush(); os.fsync(f.fileno())
        os.replace(str(dst / 'weights.attention.tp.partial'), str(dst / 'weights.attention.tp'))
    else:
        # Do not let a previous split staging run change the meaning of the
        # legacy weights.tp file on a resumed destination.
        stale = dst / 'weights.shared.tp'
        if stale.exists():
            stale.unlink()
        stale_attention = dst / 'weights.attention.tp'
        if stale_attention.exists():
            stale_attention.unlink()
    os.replace(str(dst / 'weights.index.partial'), str(dst / 'weights.index'))
    print('TP_STAGE PASS tp={} rank={}'.format(args.tp, args.rank), flush=True)


if __name__ == '__main__':
    main()
