#!/usr/bin/env python3
"""Stage exact row shards for DS41F dense TP2/TP4, reusing local EP/Engram.

Only bounded source ranges are copied. The original layout is never modified.
Metadata binds tensor ranges and checkpoint header hashes to the TP topology.
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


def layout(model, rank, tp):
    if tp not in (2, 4) or not 0 <= rank < 12:
        raise ValueError('requires rank 0..11 and TP2/TP4')
    items = [dict(x, reuse=True) for x in inventory(model, rank) if x['kind'] == 'expert']
    for original in inventory(model, 0):
        if original['kind'] != 'dense':
            continue
        name, owner = original['name'], dense_owner(original['name'])
        if name == 'head.weight':
            blocks = original['shape'][0] // 32
            first = (blocks * rank // 12) * 32
            end = (blocks * (rank + 1) // 12) * 32
            items.append(row_shard(original, first, end - first))
            continue
        sharded = any(part in name for part in
                      ('.attn.wq_b.', '.attn.wo_a.', '.attn.wo_b.', '.ffn.shared_experts.'))
        if sharded:
            if rank // tp == owner // tp:
                rows = original['shape'][0]
                if rows % tp:
                    raise ValueError('unaligned TP tensor ' + name)
                local = rows // tp
                items.append(row_shard(original, (rank % tp) * local, local))
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
    ap.add_argument('--tp', type=int, choices=(2, 4), required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if not 0 <= args.rank < 12:
        ap.error('rank must be 0..11')
    items = layout(args.model, args.rank, args.tp)
    metadata = dict(version=1, tp=args.tp, rank=args.rank, ranks=12,
                    header_sha256=header_hashes(args.model), items=items)
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
    with (dst / 'weights.tp.partial').open('w') as f:
        f.write('DS41FTP 1 {} {} 12\n'.format(args.tp, args.rank))
        for item in items:
            if 'global_shape' in item:
                f.write('{} {} {} {} {}\n'.format(item['name'], item['global_shape'][0],
                    item['global_shape'][1], item['first_row'], item['shape'][0]))
        f.flush(); os.fsync(f.fileno())
    os.replace(str(dst / 'weights.tp.partial'), str(dst / 'weights.tp'))
    os.replace(str(dst / 'weights.index.partial'), str(dst / 'weights.index'))
    print('TP_STAGE PASS tp={} rank={}'.format(args.tp, args.rank), flush=True)


if __name__ == '__main__':
    main()
