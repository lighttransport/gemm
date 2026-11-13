#!/usr/bin/env python3
"""Compare praxis_e2e outputs (--ours) against gen_e2e_ref.py (--ref) stage by stage: rel-L2 / cos / max-abs; fails above --tol on the final outputs."""
import argparse, sys, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="tmp/praxis_ref/e2e"); ap.add_argument("--ours", default="tmp/praxis_out/e2e")
ap.add_argument("--tol", type=float, default=0.05)
a = ap.parse_args()
def ld(d, n): return np.load(f"{d}/{n}.npy").astype(np.float64)
bad = 0
def cmp(name, x, y, tol=None):
    global bad
    x, y = x.reshape(-1), y.reshape(-1)
    rel = np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-30); cos = x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)
    ok = tol is None or rel <= tol
    print(f"{'PASS' if ok else 'FAIL'} {name:28s} rel_l2={rel:.3e} cos={cos:.6f} maxabs={np.abs(x - y).max():.3e}" + ("" if tol is None else f" (tol {tol})"))
    bad += not ok
# canvas: uint8 pixel agreement of the tile canvas (ours is [3][T][h][2w], the ref [T][3][h][2w])
rc = np.load(f"{a.ref}/canvas.npy").astype(np.int32).transpose(1, 0, 2, 3); oc = np.load(f"{a.ours}/canvas_chtw.npy").astype(np.int32)
print(f"INFO canvas pixels: {np.mean(rc == oc) * 100:.3f}% identical, max |diff| {np.abs(rc - oc).max()}")
cmp("text (Qwen3.5 [-3], cond)", ld(a.ref, "text"), ld(a.ours, "text"), 0.08)
cmp("policy latent (VAE encode)", ld(a.ref, "latent"), ld(a.ours, "latent"), 0.08)
cmp("pred_video (last chunk)", ld(a.ref, "pred_video"), ld(a.ours, "pred_video"), a.tol)
cmp("pred_action tokens", ld(a.ref, "pred_action"), ld(a.ours, "pred_action"), a.tol)
cmp("action_rows (normalised)", ld(a.ref, "action_rows"), ld(a.ours, "action_rows"), a.tol)
try: cmp("decoded frames (last 24)", ld(a.ref, "decoded"), ld(a.ours, "decoded"), 0.2)
except FileNotFoundError: pass
print("FAILED" if bad else "ALL OK"); sys.exit(1 if bad else 0)
