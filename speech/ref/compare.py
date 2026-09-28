#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare runner .npy fixtures against the PyTorch reference.

For every <name>.npy present in both directories (or those given with --names),
prints cosine, relative L2 and max-abs error. Float arrays gate on
--min-cos; integer arrays (codes, ids) must match exactly up to the shorter
length and report the first mismatch. Waveforms additionally report SNR (dB).
Exit status is non-zero when any gate fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--runner-dir", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--min-cos", type=float, default=0.9999)
    ap.add_argument("--min-snr", type=float, default=40.0)
    ap.add_argument("--allow-int-prefix", action="store_true",
                    help="integer arrays may differ in length; compare the common prefix")
    args = ap.parse_args()
    ref, run = Path(args.reference_dir), Path(args.runner_dir)
    names = args.names or sorted(p.stem for p in run.glob("*.npy") if (ref / p.name).exists())
    ok = True
    for name in names:
        a, b = np.load(ref / f"{name}.npy"), np.load(run / f"{name}.npy")
        if a.dtype.kind in "iu":
            n = min(a.shape[0], b.shape[0])
            if a.shape != b.shape and not args.allow_int_prefix:
                print(f"{name}: FAIL shape {a.shape} vs {b.shape}")
                ok = False
                continue
            diff = np.nonzero((a[:n] != b[:n]).reshape(n, -1).any(axis=1))[0]
            first = int(diff[0]) if diff.size else -1
            status = "ok" if first < 0 and (a.shape == b.shape or args.allow_int_prefix) else "FAIL"
            ok &= status == "ok"
            print(f"{name}: {status} int len ref={a.shape[0]} run={b.shape[0]} first_mismatch={first}")
            continue
        if a.shape != b.shape:
            if a.size == b.size:
                b = b.reshape(a.shape)
            else:
                print(f"{name}: FAIL shape {a.shape} vs {b.shape}")
                ok = False
                continue
        x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        cos = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))
        rel = float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30))
        mx = float(np.max(np.abs(x - y)))
        line = f"{name}: cos={cos:.9f} rel_l2={rel:.3e} max_abs={mx:.3e}"
        good = cos >= args.min_cos
        if name.startswith("wav"):
            snr = 10 * np.log10(max(np.sum(x * x), 1e-30) / max(np.sum((x - y) ** 2), 1e-30))
            line += f" snr={snr:.1f}dB"
            good &= snr >= args.min_snr
        print(("ok   " if good else "FAIL ") + line)
        ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
