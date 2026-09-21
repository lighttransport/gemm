#!/usr/bin/env python3
"""Compare qimg21 runner/reference image fixtures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--runner-dir", required=True)
    args = ap.parse_args()
    ref_dir, run_dir = Path(args.reference_dir), Path(args.runner_dir)
    ref_path, run_path = ref_dir / "reference_rgba.npy", run_dir / "runner.png"
    if not ref_path.exists() or not run_path.exists():
        raise SystemExit("expected reference_rgba.npy and runner.png")
    ref = np.load(ref_path).astype(np.float64)
    got = np.asarray(Image.open(run_path).convert("RGBA")).astype(np.float64)
    if ref.shape != got.shape:
        raise SystemExit(f"shape mismatch: {ref.shape} vs {got.shape}")
    diff = got - ref
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    peak = 255.0
    psnr = 20 * math.log10(peak / max(rmse, 1e-12))
    cos = float(np.dot(ref.ravel(), got.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
    print(f"shape={ref.shape} mae={mae:.5f} rmse={rmse:.5f} psnr={psnr:.3f}dB cosine={cos:.9f}")
    for name in ("prompt_embeds.npy", "prompt_mask.npy", "image_pad_mask.npy"):
        rp, gp = ref_dir / name, run_dir / name
        if rp.exists() and gp.exists():
            r, g = np.load(rp).astype(np.float64).ravel(), np.load(gp).astype(np.float64).ravel()
            n = min(r.size, g.size)
            c = float(np.dot(r[:n], g[:n]) / (np.linalg.norm(r[:n]) * np.linalg.norm(g[:n]) + 1e-12))
            e = float(np.linalg.norm(r[:n] - g[:n]) / (np.linalg.norm(r[:n]) + 1e-12))
            print(f"{name}: cosine={c:.9f} rel_l2={e:.8g}")
    ref_steps = sorted(ref_dir.glob("step_*.npy"))
    for rp in ref_steps:
        gp = run_dir / rp.name
        if not gp.exists():
            print(f"missing runner fixture: {gp}")
            continue
        r, g = np.load(rp).astype(np.float64).ravel(), np.load(gp).astype(np.float64).ravel()
        n = min(r.size, g.size)
        c = float(np.dot(r[:n], g[:n]) / (np.linalg.norm(r[:n]) * np.linalg.norm(g[:n]) + 1e-12))
        e = float(np.linalg.norm(r[:n] - g[:n]) / (np.linalg.norm(r[:n]) + 1e-12))
        print(f"{rp.name}: cosine={c:.9f} rel_l2={e:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
