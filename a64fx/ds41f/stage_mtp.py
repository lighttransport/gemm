#!/usr/bin/env python3
"""Bounded exact-byte DSpark staging, alongside the existing backbone layout.

Three stage owners are 0/4/8. TP4 shards attention/shared projections and the
main projection. Draft experts use expert % 12; Markov embedding/head rows use
the backbone vocabulary partition. Backbone embedding/head are not copied.
"""
import argparse
import json
import os
import re
import struct
from pathlib import Path
from stage_dense import copy_tensor
from stage_tp import row_shard, header_hashes


def inventory_mtp(model):
    items = []
    for path in sorted(model.glob('*.safetensors')):
        with path.open('rb') as file:
            length = struct.unpack('<Q', file.read(8))[0]
            header = json.loads(file.read(length))
        for name, desc in header.items():
            if not name.startswith('mtp.'):
                continue
            if not re.match(r'mtp\.[012]\.', name):
                raise ValueError('unsupported DSpark stage ' + name)
            first, end = desc['data_offsets']
            items.append(dict(name=name, dtype=desc['dtype'], shape=desc['shape'],
                              source_shape=desc['shape'], source=str(path),
                              source_offset=8 + length + first, bytes=end-first,
                              first_row=0, kind='mtp'))
    if {int(x['name'].split('.')[1]) for x in items} != {0, 1, 2}:
        raise ValueError('expected all three DSpark stages')
    return sorted(items, key=lambda x: x['name'])


def layout(items, rank):
    if not 0 <= rank < 12:
        raise ValueError('rank must be 0..11')
    result = []
    for item in items:
        name = item['name']
        owner = int(name.split('.')[1]) * 4
        expert = re.search(r'\.ffn\.experts\.(\d+)\.', name)
        if expert:
            ident = int(expert.group(1))
            if not 0 <= ident < 128:
                raise ValueError('draft expert ID must be 0..127')
            if ident % 12 == rank:
                result.append(item)
        elif '.markov_head.' in name:
            rows = item['shape'][0]
            if len(item['shape']) != 2 or rows % 32:
                raise ValueError('unaligned Markov vocabulary tensor')
            first = (rows // 32 * rank // 12) * 32
            end = (rows // 32 * (rank + 1) // 12) * 32
            result.append(row_shard(item, first, end-first))
        elif any(part in name for part in ('.attn.wq_b.', '.attn.wo_a.', '.attn.wo_b.',
                                          '.ffn.shared_experts.', '.main_proj.')):
            if rank // 4 == owner // 4:
                rows = item['shape'][0]
                if rows % 4:
                    raise ValueError('unaligned TP4 tensor ' + name)
                result.append(row_shard(item, rank % 4 * (rows // 4), rows // 4))
        elif name.endswith('.attn.attn_sink'):
            if rank // 4 == owner // 4:
                result.append(item)
        elif rank == owner:
            result.append(item)
    return sorted(result, key=lambda x: x['name'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if not 0 <= args.rank < 12:
        ap.error('rank must be 0..11')
    originals = inventory_mtp(args.model)
    items = layout(originals, args.rank)
    metadata = dict(version=1, tp=4, ranks=12, rank=args.rank, stage_owners=[0, 4, 8],
                    draft_block_size=5, noise_token_id=128799, target_layers=[37, 38, 39],
                    markov_rank=256, routed_experts=128, activated_experts=3,
                    source_bytes=sum(x['bytes'] for x in originals),
                    header_sha256=header_hashes(args.model), items=items)
    print('MTP_STAGE_PLAN rank={} tensors={} resident_bytes={}'.format(
        args.rank, len(items), sum(x['bytes'] for x in items)), flush=True)
    if args.dry_run:
        return
    dst = args.destination
    dst.mkdir(parents=True, exist_ok=True)
    manifest = dst / 'mtp-manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError('existing MTP manifest differs')
    else:
        with manifest.open('x') as file:
            json.dump(metadata, file, sort_keys=True)
            file.flush(); os.fsync(file.fileno())
    for item in items:
        copy_tensor(item, dst)
    with (dst / 'weights.index.partial').open('w') as file:
        for item in items:
            shape = item['shape'] if len(item['shape']) == 2 else [1, item['shape'][0]]
            file.write('{} {} {} {} {}\n'.format(item['name'], item['dtype'], shape[0], shape[1], item['bytes']))
        file.flush(); os.fsync(file.fileno())
    with (dst / 'weights.tp.partial').open('w') as file:
        file.write('DS41FTP 1 4 {} 12\n'.format(args.rank))
        for item in items:
            if 'global_shape' in item:
                file.write('{} {} {} {} {}\n'.format(item['name'], item['global_shape'][0],
                    item['global_shape'][1], item['first_row'], item['shape'][0]))
        file.flush(); os.fsync(file.fileno())
    os.replace(str(dst / 'weights.tp.partial'), str(dst / 'weights.tp'))
    os.replace(str(dst / 'weights.index.partial'), str(dst / 'weights.index'))
    print('MTP_STAGE PASS rank={}'.format(args.rank), flush=True)


if __name__ == '__main__':
    main()
