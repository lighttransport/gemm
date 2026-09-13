#!/usr/bin/env python3
"""Bounded, resumable exact-byte staging; no model dequantization.

Expert ownership is expert % 12. Engram rows use contiguous ceil-div shards.
Dense tensors have one canonical disk copy on rank 0; runtime distribution is
separate. Manifest offsets preserve original safetensors tensor contracts.
"""
import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path


def inventory(model, rank):
    for path in sorted(model.glob('*.safetensors')):
        with path.open('rb') as src:
            header_size = struct.unpack('<Q', src.read(8))[0]
            header = json.loads(src.read(header_size))
        for name, desc in sorted(header.items()):
            if name == '__metadata__':
                continue
            if name.startswith(('mtp.', 'vision', 'aligner.', 'image_')):
                continue
            start, end = desc['data_offsets']
            first = 0
            shape = list(desc['shape'])
            expert = re.search(r'\.ffn\.experts\.(\d+)\.', name)
            if expert:
                if int(expert.group(1)) % 12 != rank:
                    continue
                kind = 'expert'
            elif '.engram.embed.' in name:
                kind = 'engram'
                stride = (end - start) // shape[0]
                per = (shape[0] + 11) // 12
                first = min(rank * per, shape[0])
                count = min(per, shape[0] - first)
                start += first * stride
                end = start + count * stride
                shape[0] = count
            else:
                if rank != 0:
                    continue
                kind = 'dense'
            yield dict(name=name, dtype=desc['dtype'], shape=shape,
                       source_shape=desc['shape'], first_row=first,
                       source=str(path), source_offset=8+header_size+start,
                       bytes=end-start, kind=kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if not 0 <= args.rank < 12:
        ap.error('rank must be 0..11')
    items = list(inventory(args.model, args.rank))
    total = sum(x['bytes'] for x in items)
    print('STAGE rank={} tensors={} bytes={}'.format(args.rank,len(items),total), flush=True)
    if args.dry_run:
        return
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = args.destination / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['rank'] != args.rank or old['items'] != items:
            raise RuntimeError('existing staging manifest differs')
    else:
        with manifest.open('x') as f:
            json.dump(dict(rank=args.rank, ranks=12, items=items), f)
    done_bytes = 0
    for i, item in enumerate(items):
        target = args.destination / (item['name'] + '.bin')
        stamp = args.destination / (item['name'] + '.sha256')
        if target.exists() and stamp.exists() and target.stat().st_size == item['bytes']:
            done_bytes += item['bytes']
            continue
        partial = target.with_suffix('.partial')
        digest = hashlib.sha256()
        with open(item['source'], 'rb', buffering=0) as src, partial.open('wb', buffering=0) as dst:
            src.seek(item['source_offset'])
            left = item['bytes']
            while left:
                block = src.read(min(left, 8*1024*1024))
                if not block:
                    raise IOError('short source tensor')
                view = memoryview(block)
                while view:
                    n = dst.write(view)
                    if not n:
                        raise IOError('short destination write')
                    view = view[n:]
                digest.update(block)
                left -= len(block)
                os.fsync(dst.fileno())
                os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(src.fileno(), item['source_offset'] + item['bytes'] - left - len(block), len(block), os.POSIX_FADV_DONTNEED)
        os.replace(str(partial), str(target))
        with stamp.open('w') as f:
            f.write(digest.hexdigest()+'\n')
            f.flush(); os.fsync(f.fileno())
        done_bytes += item['bytes']
        if i % 100 == 0 or item['kind'] == 'engram':
            print('PROGRESS rank={} bytes={}/{} tensor={}'.format(args.rank,done_bytes,total,item['name']), flush=True)
    print('STAGE PASS rank={} bytes={}'.format(args.rank,total), flush=True)


if __name__ == '__main__':
    main()
