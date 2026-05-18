#!/usr/bin/env python3
"""Compare Hy3D PyTorch trace tensors against HIP replay dumps."""

import argparse
import json
import os

import numpy as np


def _load_first_batch(path):
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32, copy=False)


def _metrics(a, b):
    d = a - b
    denom = np.linalg.norm(b.reshape(-1))
    return {
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "rel_l2": float(np.linalg.norm(d.reshape(-1)) / denom) if denom else 0.0,
        "a_std": float(np.std(a)),
        "b_std": float(np.std(b)),
    }


def _ref_latent_path(ref_dir, step):
    if step == 1:
        return os.path.join(ref_dir, "04_dit_latents_step0.npy")
    return os.path.join(ref_dir, f"05_dit_input_x_{step - 1:03d}.npy")


def _print_table(title, rows):
    print(title)
    print("step  max_abs      mean_abs     rel_l2      hip_std      ref_std")
    for step, m in rows:
        print(f"{step:4d}  {m['max_abs']:10.6g}  {m['mean_abs']:10.6g}  "
              f"{m['rel_l2']:10.6g}  {m['a_std']:10.6g}  {m['b_std']:10.6g}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ref-dir", default="/tmp/hy3d_ref_trace")
    parser.add_argument("--hip-latent-prefix", default="/tmp/hy3d_hip_latent")
    parser.add_argument("--hip-velocity-prefix", default="/tmp/hy3d_hip_velocity")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--json", default=None, help="optional output JSON path")
    args = parser.parse_args()

    out = {"latents": [], "velocities": []}
    latent_rows = []
    velocity_rows = []

    for step in range(1, args.steps + 1):
        hip = _load_first_batch(f"{args.hip_latent_prefix}_{step:03d}.npy")
        ref = _load_first_batch(_ref_latent_path(args.ref_dir, step))
        m = _metrics(hip, ref)
        latent_rows.append((step, m))
        out["latents"].append({"step": step, **m})

    for step in range(1, args.steps + 1):
        hip = _load_first_batch(f"{args.hip_velocity_prefix}_{step:03d}.npy")
        pred = np.load(os.path.join(args.ref_dir, f"06_dit_output_{step - 1:03d}.npy"))
        if pred.ndim != 3 or pred.shape[0] < 2:
            raise ValueError("expected CFG PyTorch output with shape [2, 4096, 64]")
        cfg = pred[1].astype(np.float32) + args.guidance * (
            pred[0].astype(np.float32) - pred[1].astype(np.float32))
        m = _metrics(hip, cfg)
        velocity_rows.append((step, m))
        out["velocities"].append({"step": step, **m})

    _print_table("Latents", latent_rows)
    print()
    _print_table("Velocities vs PyTorch CFG", velocity_rows)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
