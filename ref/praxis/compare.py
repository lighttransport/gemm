#!/usr/bin/env python3
"""Compare .npy dumps: python compare.py --ref-dir R --ours-dir O  (or --ref/--ours).
Pass: rel-L2 <= 0.02 and cosine >= 0.9999 (override with --tol)."""
import argparse, glob, os, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--ref"); ap.add_argument("--ours")
ap.add_argument("--ref-dir"); ap.add_argument("--ours-dir")
ap.add_argument("--tol", type=float, default=0.02)
ap.add_argument("--cos", type=float, default=0.9999)
ap.add_argument("--skip-missing", action="store_true", help="do not list reference files the runner does not produce")
ap.add_argument("--topk-agree", type=float, default=None,
                help="for *_top_idx dumps: pass if this fraction of tokens routes to the identical expert set (bf16 runs flip near-ties)")
a = ap.parse_args()
pairs = [(a.ref, a.ours)] if a.ref else [(f, os.path.join(a.ours_dir, os.path.basename(f))) for f in sorted(glob.glob(a.ref_dir + "/*.npy"))]
bad = 0
for r, o in pairs:
    if not os.path.exists(o):
        if not a.skip_missing: print(f"MISSING {o}")
        continue
    if a.topk_agree is not None and r.endswith("_top_idx.npy"):
        xr, yr = np.load(r), np.load(o)
        agree = float(np.mean(np.all(np.sort(xr, axis=-1) == np.sort(yr, axis=-1), axis=-1)))
        ok = agree >= a.topk_agree
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {os.path.basename(r):20s} topk token agreement={agree:.4f}")
        continue
    x, y = np.load(r).astype(np.float64).ravel(), np.load(o).astype(np.float64).ravel()
    rel = np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-30)
    cos = x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)
    ok = rel <= a.tol and cos >= a.cos
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} {os.path.basename(r):20s} rel_l2={rel:.3e} cos={cos:.6f} maxabs={np.abs(x-y).max():.3e}")
raise SystemExit(1 if bad else 0)
