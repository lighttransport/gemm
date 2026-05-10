#!/usr/bin/env python3
"""Convert PyT shape_dec subdivision dumps to test_hip_tex_dec cache format.

Reads `14_shape_sub{0..3}.coords.npy` (each [N,4] int32 in (b,x,y,z) order at
scales 32, 64, 128, 256). Writes per-stage guide caches:

    cache_scale{16,8,4}_c2s_idx.npy       int64 [N_fine]   parent index
    cache_scale{16,8,4}_c2s_subidx.npy    int32 [N_fine]   slot 0..7
    cache_scale{16,8,4}_c2s_x_coords.npy  int32 [N_fine,4] fine coords (b,x,y,z)

Slot encoding follows trellis2 SparseSpatial2Channel:
    subidx = x_bit + y_bit*2 + z_bit*4
where (x_bit,y_bit,z_bit) = fine_coord[1:] % 2.
"""
import argparse, os, numpy as np

# stage_idx -> (parent_sub, fine_sub, cache_scale)
STAGES = [(0, 1, 16), (1, 2, 8), (2, 3, 4)]


def build_cache(parent_coords, fine_coords):
    # Hash parent coords (b,x,y,z) -> index
    pmap = {tuple(int(v) for v in c): i for i, c in enumerate(parent_coords)}
    N = fine_coords.shape[0]
    idx = np.empty(N, dtype=np.int64)
    subidx = np.empty(N, dtype=np.int32)
    miss = 0
    for k, fc in enumerate(fine_coords):
        b, x, y, z = int(fc[0]), int(fc[1]), int(fc[2]), int(fc[3])
        key = (b, x >> 1, y >> 1, z >> 1)
        p = pmap.get(key, -1)
        if p < 0:
            miss += 1
            p = 0
        idx[k] = p
        subidx[k] = (x & 1) | ((y & 1) << 1) | ((z & 1) << 2)
    return idx, subidx, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dumps', required=True, help='dir with 14_shape_sub{0..3}.coords.npy')
    ap.add_argument('--out', required=True, help='output cache dir')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    subs = [np.load(os.path.join(a.dumps, f'14_shape_sub{i}.coords.npy')) for i in range(4)]
    for i, s in enumerate(subs):
        print(f'sub{i}: {s.shape}, ranges {s.min(0)} .. {s.max(0)}')

    for parent_i, fine_i, scale in STAGES:
        pc, fc = subs[parent_i], subs[fine_i]
        idx, subidx, miss = build_cache(pc, fc)
        if miss:
            print(f'  WARN stage {parent_i}->{fine_i}: {miss} fine coords had no parent')
        np.save(os.path.join(a.out, f'cache_scale{scale}_c2s_idx.npy'), idx)
        np.save(os.path.join(a.out, f'cache_scale{scale}_c2s_subidx.npy'), subidx)
        np.save(os.path.join(a.out, f'cache_scale{scale}_c2s_x_coords.npy'), fc.astype(np.int32, copy=False))
        print(f'  cache_scale{scale}: N_fine={fc.shape[0]} (parent N={pc.shape[0]})')


if __name__ == '__main__':
    main()
