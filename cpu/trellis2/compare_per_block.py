#!/usr/bin/env python3
"""Layer-by-layer comparison of HIP DiT vs PyTorch reference.

Reads ref_block{00..29}_hidden.npy + hip_block{00..29}_hidden.npy from a
directory and reports per-block max_abs, rel_l2, and corr.
"""
import sys, os, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default='.')
ap.add_argument('--max-bad', type=int, default=3, help='print this many top-diff rows per block')
args = ap.parse_args()

print(f"{'blk':>3} {'shape':>14} {'ref_std':>10} {'hip_std':>10} "
      f"{'max_abs':>10} {'rel_l2':>10} {'corr':>10}  pass")
print('-' * 90)

worst = {'max_abs': (0, -1), 'rel_l2': (0, -1), 'corr': (1, -1)}
for bi in range(30):
    rp = os.path.join(args.dir, f'ref_block{bi:02d}_hidden.npy')
    hp = os.path.join(args.dir, f'hip_block{bi:02d}_hidden.npy')
    if not os.path.exists(rp) or not os.path.exists(hp):
        print(f"{bi:>3}  missing")
        continue
    R = np.load(rp).astype(np.float64)
    H = np.load(hp).astype(np.float64)
    # ref [4096, 1536], hip [4096, 1536] should match shape
    if R.shape != H.shape:
        # ref might be [N_TOK, DIT_DIM] while hip is flat — try reshape
        H = H.reshape(R.shape)
    diff = np.abs(R - H)
    max_abs = diff.max()
    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(R) + 1e-30)
    dot = float((R * H).sum())
    corr = dot / (np.linalg.norm(R) * np.linalg.norm(H) + 1e-30)
    # BF16 across 30 layers: max_abs drifts proportional to magnitude;
    # rel_l2 + corr are the meaningful metrics.
    p = (rel_l2 < 0.05 and corr > 0.999)
    print(f"{bi:>3} {str(R.shape):>14} {R.std():>10.4f} {H.std():>10.4f} "
          f"{max_abs:>10.4f} {rel_l2:>10.6f} {corr:>10.6f}  {'PASS' if p else 'FAIL'}")
    if max_abs > worst['max_abs'][0]: worst['max_abs'] = (max_abs, bi)
    if rel_l2  > worst['rel_l2'][0]:  worst['rel_l2']  = (rel_l2, bi)
    if corr    < worst['corr'][0]:    worst['corr']    = (corr, bi)

print()
print(f"worst max_abs: {worst['max_abs'][0]:.4f} @ block {worst['max_abs'][1]}")
print(f"worst rel_l2:  {worst['rel_l2'][0]:.6f} @ block {worst['rel_l2'][1]}")
print(f"worst corr:    {worst['corr'][0]:.6f} @ block {worst['corr'][1]}")
