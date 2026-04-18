"""Compare selected DiT trajectory latent checkpoints and velocities.

Reference files are expected from run_full_pipeline.py --trace-dir:
  05_dit_input_x_<step0:03d>.npy
  06_dit_output_<step0:03d>.npy

CUDA files are expected from test_cuda_hy3d latent dumps:
  <prefix>_<step1:03d>.npy
where step1 is 1-based diffusion step index.
"""
import argparse
import os
import sys

import numpy as np


def compare_one(step1, ref_dir, cuda_dir, cuda_prefix, cuda_kind, max_abs, mean_abs):
    step0 = step1 - 1
    if cuda_kind == "latent":
        ref_name = f"05_dit_input_x_{step0:03d}.npy"
    elif cuda_kind == "velocity":
        ref_name = f"06_dit_output_{step0:03d}.npy"
    else:
        raise ValueError(f"unknown cuda_kind={cuda_kind}")
    cuda_name = f"{cuda_prefix}_{step1:03d}.npy"
    ref_path = os.path.join(ref_dir, ref_name)
    cuda_path = os.path.join(cuda_dir, cuda_name)
    if not os.path.exists(ref_path):
        print(f"  step {step1}: MISSING ref {ref_name}")
        return False
    if not os.path.exists(cuda_path):
        print(f"  step {step1}: MISSING cuda {cuda_name}")
        return False

    ref = np.load(ref_path)
    test = np.load(cuda_path)
    if ref.ndim == 3 and test.ndim == 2 and ref.shape[1:] == test.shape and ref.shape[0] >= 1:
        # PyTorch trace uses CFG-batched latents [2, N, C]; CUDA dump is [N, C].
        # Compare against the conditional branch (index 0).
        ref = ref[0]
    if ref.shape != test.shape:
        print(f"  step {step1}: SHAPE MISMATCH ref={ref.shape} cuda={test.shape}")
        return False

    diff = np.abs(ref - test)
    dmax = float(diff.max())
    dmean = float(diff.mean())
    ok = dmax <= max_abs and dmean <= mean_abs
    print(
        f"  step {step1}: {'OK' if ok else 'FAIL'} "
        f"max={dmax:.2e} mean={dmean:.2e} shape={ref.shape}"
    )
    if not ok:
        idx = np.unravel_index(int(diff.argmax()), diff.shape)
        print(f"    worst@{idx}: ref={float(ref[idx]):.6f} cuda={float(test[idx]):.6f}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref_trace_dir")
    p.add_argument("cuda_latent_dir")
    p.add_argument("--steps", default="1,15,30", help="comma-separated 1-based steps")
    p.add_argument("--cuda-prefix", default="dit_latent_step")
    p.add_argument("--kind", choices=["latent", "velocity"], default="latent")
    p.add_argument("--max-abs", type=float, default=0.03)
    p.add_argument("--mean-abs", type=float, default=0.001)
    args = p.parse_args()

    steps = []
    for tok in args.steps.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0:
            print(f"Invalid step '{tok}' (must be >= 1)")
            return 1
        steps.append(v)
    steps = sorted(set(steps))

    ok = 0
    fail = 0
    for s in steps:
        if compare_one(
            s,
            args.ref_trace_dir,
            args.cuda_latent_dir,
            args.cuda_prefix,
            args.kind,
            args.max_abs,
            args.mean_abs,
        ):
            ok += 1
        else:
            fail += 1

    print(f"\n{ok} OK, {fail} FAIL / {len(steps)} steps")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
