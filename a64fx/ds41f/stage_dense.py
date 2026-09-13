#!/usr/bin/env python3
"""Materialize dense layer ownership and a rank-local resident tensor index.

Existing canonical rank-zero tensors are reused. Other owners read only their
dense tensors from shared storage, never a second copy of the expert/Engram
payload. All copies are bounded and source-cache evicted.
"""
import argparse
import hashlib
import os
import re
from pathlib import Path
from stage_backbone import inventory


def dense_owner(name):
    match = re.match(r'layers\.(\d+)\.', name)
    return int(match.group(1)) % 12 if match else (
        11 if name.startswith(('head.', 'norm.')) else 0)


def copy_tensor(item, destination):
    target = destination / (item['name']+'.bin')
    stamp = destination / (item['name']+'.sha256')
    if target.exists() and stamp.exists() and target.stat().st_size == item['bytes']:
        return
    digest = hashlib.sha256()
    partial = destination / (item['name']+'.dense-partial')
    with open(item['source'], 'rb', buffering=0) as src, partial.open('wb', buffering=0) as dst:
        src.seek(item['source_offset'])
        left = item['bytes']
        while left:
            block = src.read(min(left, 8*1024*1024))
            if not block: raise IOError('short source')
            view = memoryview(block)
            while view:
                written = dst.write(view)
                if not written: raise IOError('short destination write')
                view = view[written:]
            digest.update(block)
            left -= len(block)
            os.fsync(dst.fileno())
            os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(src.fileno(), src.tell()-len(block), len(block), os.POSIX_FADV_DONTNEED)
    os.replace(str(partial), str(target))
    with stamp.open('w') as f:
        f.write(digest.hexdigest()+'\n'); f.flush(); os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--rank', type=int, required=True)
    args = ap.parse_args()
    if not 0 <= args.rank < 12: ap.error('rank must be 0..11')
    args.destination.mkdir(parents=True, exist_ok=True)
    experts = [x for x in inventory(args.model, args.rank) if x['kind']=='expert']
    dense = [x for x in inventory(args.model, 0)
             if x['kind']=='dense' and dense_owner(x['name'])==args.rank]
    for item in dense:
        copy_tensor(item, args.destination)
    items = sorted(experts+dense, key=lambda x: x['name'])
    # Publish only after every indexed file exists with the expected size.
    for item in items:
        target = args.destination / (item['name']+'.bin')
        if target.stat().st_size != item['bytes']: raise IOError('incomplete tensor '+str(target))
        if len(item['shape']) not in (1,2): raise ValueError('unsupported tensor rank')
    partial = args.destination/'weights.index.partial'
    with partial.open('w') as f:
        for item in items:
            shape = item['shape'] if len(item['shape'])==2 else [1,item['shape'][0]]
            f.write('{} {} {} {} {}\n'.format(item['name'],item['dtype'],shape[0],shape[1],item['bytes']))
        f.flush(); os.fsync(f.fileno())
    os.replace(str(partial),str(args.destination/'weights.index'))
    print('DENSE_STAGE PASS rank={} tensors={} resident_bytes={}'.format(
        args.rank,len(items),sum(x['bytes'] for x in items)),flush=True)


if __name__ == '__main__': main()
