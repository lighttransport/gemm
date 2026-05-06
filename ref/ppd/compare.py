#!/usr/bin/env python3
"""Compare two PPD depth .npy files."""

from __future__ import annotations

import argparse
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--mae-thresh", type=float, default=0.10)
    ap.add_argument("--pearson-thresh", type=float, default=0.94)
    args = ap.parse_args()

    ref = np.load(args.reference).astype(np.float64)
    cand = np.load(args.candidate).astype(np.float64)
    if ref.shape != cand.shape:
        raise SystemExit(f"shape mismatch: ref={ref.shape} candidate={cand.shape}")

    diff = cand - ref
    mae = float(np.mean(np.abs(diff)))
    max_ae = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref0 = ref.reshape(-1) - float(np.mean(ref))
    cand0 = cand.reshape(-1) - float(np.mean(cand))
    denom = float(np.linalg.norm(ref0) * np.linalg.norm(cand0))
    pearson = float(np.dot(ref0, cand0) / denom) if denom > 0.0 else 0.0
    passed = mae <= args.mae_thresh and pearson >= args.pearson_thresh

    print(f"shape: {ref.shape}")
    print(f"ref range: [{ref.min():.6f}, {ref.max():.6f}], mean={ref.mean():.6f}")
    print(f"candidate range: [{cand.min():.6f}, {cand.max():.6f}], mean={cand.mean():.6f}")
    print(f"pearson: {pearson:.6f}")
    print(f"mae: {mae:.6f}")
    print(f"rmse: {rmse:.6f}")
    print(f"max_ae: {max_ae:.6f}")
    print("status:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
