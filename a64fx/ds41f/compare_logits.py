#!/usr/bin/env python3
"""Compare bounded runner dumps with independently computed FP32 logits.

This checks CPU numerical agreement, not official GPU-model parity.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", required=True)
    parser.add_argument("--actual-prefix", required=True)
    parser.add_argument("--positions", type=int, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args()
    if args.positions < 1 or not -1 <= args.min_cosine <= 1:
        parser.error("positive positions and cosine in [-1, 1] required")
    passed = True
    for pos in range(args.positions):
        arrays = []
        for prefix in (args.reference_prefix, args.actual_prefix):
            path = Path(f"{prefix}.pos{pos}.bin")
            if path.stat().st_size != 129280 * 4:
                raise ValueError(f"wrong logit geometry: {path}")
            array = np.fromfile(path, dtype="<f4").astype(np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite logits: {path}")
            arrays.append(array)
        reference, actual = arrays
        denom = np.linalg.norm(reference) * np.linalg.norm(actual)
        cosine = float(reference @ actual / denom) if denom else float("nan")
        error = actual - reference
        expected, observed = int(reference.argmax()), int(actual.argmax())
        ok = expected == observed and cosine >= args.min_cosine
        passed &= ok
        print(f"LOGIT_COMPARE {'PASS' if ok else 'FAIL'} pos={pos} "
              f"reference={expected} actual={observed} cosine={cosine:.9f} "
              f"rms={np.sqrt(np.mean(error**2)):.7g} "
              f"max_abs={np.max(np.abs(error)):.7g}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
