#!/usr/bin/env python3
"""
Compare .npy outputs: reference (PyTorch) vs our C implementation.

Usage:
    uv run python compare.py --ref output/vae_02_dec_conv1.npy --ours ../../cpu/qwen_image/vae_02_dec_conv1.npy
    uv run python compare.py --ref-dir output/ --ours-dir ../../cpu/qwen_image/  # compare all matching files
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def compare_one(ref_path, ours_path, label=""):
    ref = np.load(ref_path).astype(np.float32).flatten()
    ours = np.load(ours_path).astype(np.float32).flatten()

    if ref.size != ours.size:
        min_n = min(ref.size, ours.size)
        print(f"  {label}: SIZE MISMATCH ref={ref.size} ours={ours.size}, "
              f"comparing first {min_n}")
        ref = ref[:min_n]
        ours = ours[:min_n]

    diff = np.abs(ref - ours)
    max_diff = diff.max()
    mean_diff = diff.mean()
    p99 = np.percentile(diff, 99)

    # Correlation
    if ref.size > 10 and np.std(ref) > 1e-10 and np.std(ours) > 1e-10:
        corr = np.corrcoef(ref, ours)[0, 1]
    else:
        corr = float('nan')

    # Relative error (avoid div by zero)
    denom = np.maximum(np.abs(ref), 1e-8)
    rel_err = (diff / denom).mean()

    status = "PASS" if (corr > 0.99 or max_diff < 0.01) else \
             "MARGINAL" if (corr > 0.95 or max_diff < 0.1) else "FAIL"

    print(f"  {label:40s}  max={max_diff:10.6f}  mean={mean_diff:10.6f}  "
          f"p99={p99:10.6f}  corr={corr:.6f}  rel={rel_err:.6f}  [{status}]")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', help='Reference .npy file')
    parser.add_argument('--ours', help='Our .npy file')
    parser.add_argument('--ref-dir', help='Directory of reference .npy files')
    parser.add_argument('--ours-dir', help='Directory of our .npy files')
    args = parser.parse_args()

    if args.ref and args.ours:
        compare_one(args.ref, args.ours, Path(args.ref).stem)
    elif args.ref_dir and args.ours_dir:
        ref_dir = Path(args.ref_dir)
        ours_dir = Path(args.ours_dir)
        ref_files = sorted(ref_dir.glob("*.npy"))
        if not ref_files:
            print(f"No .npy files in {ref_dir}")
            sys.exit(1)

        results = {"PASS": 0, "MARGINAL": 0, "FAIL": 0}
        for rf in ref_files:
            of = ours_dir / rf.name
            if of.exists():
                s = compare_one(rf, of, rf.stem)
                results[s] += 1
            else:
                print(f"  {rf.stem:40s}  MISSING in {ours_dir}")

        print(f"\nSummary: {results['PASS']} PASS, "
              f"{results['MARGINAL']} MARGINAL, {results['FAIL']} FAIL")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
