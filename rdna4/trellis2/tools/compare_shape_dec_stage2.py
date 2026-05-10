#!/usr/bin/env python3
"""Compare HIP vs CPU shape_dec post-stage-2-C2S features (Milestone A).

Both dumps must come from running the same weights on the same SLAT input
with the same converted PyT subdivision cache, so coord ordering is
identical (driven by cache_scale4_c2s_x_coords.npy). Reports rel-L2,
max-abs-diff, cosine, and per-channel breakdown.
"""
import argparse, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hip', required=True)
    ap.add_argument('--cpu', required=True)
    a = ap.parse_args()
    h = np.load(a.hip).astype(np.float64)
    c = np.load(a.cpu).astype(np.float64)
    print(f'HIP: {h.shape} {h.dtype}, mean_abs={np.abs(h).mean():.4f} rms={np.sqrt((h*h).mean()):.4f}')
    print(f'CPU: {c.shape} {c.dtype}, mean_abs={np.abs(c).mean():.4f} rms={np.sqrt((c*c).mean()):.4f}')
    if h.shape != c.shape:
        print('SHAPE MISMATCH')
        return
    d = h - c
    rel = np.sqrt((d*d).sum() / max((c*c).sum(), 1e-30))
    mx = np.abs(d).max()
    cos = (h*c).sum() / max(np.sqrt((h*h).sum() * (c*c).sum()), 1e-30)
    print(f'\nrel_L2={rel:.4e}  max_abs_diff={mx:.4e}  cosine={cos:.6f}')

    # Per-channel
    Cn = h.shape[1]
    print('\nch     rel_L2     max_diff   cosine')
    for c_i in range(min(Cn, 16)):
        hi, ci = h[:, c_i], c[:, c_i]
        rl = np.sqrt(((hi-ci)**2).sum() / max((ci*ci).sum(), 1e-30))
        mxc = np.abs(hi-ci).max()
        cs = (hi*ci).sum() / max(np.sqrt((hi*hi).sum() * (ci*ci).sum()), 1e-30)
        print(f'{c_i:3d}   {rl:.3e}   {mxc:.3e}   {cs:.6f}')

if __name__ == '__main__':
    main()
