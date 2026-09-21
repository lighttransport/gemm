#!/usr/bin/env python3
"""Compare matched block replays; never treat injected states as model acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare import _cosine_error, NONQUANTIZED_COSINE_THRESHOLD


def compare_stages(reference, candidate, block):
    results = {}
    # img_norm2 hooks capture unmodulated normalization, whereas the native
    # mod_ln2 dump includes modulation: these are deliberately not compared.
    for name in ("attn_raw", "attn_out", "mlp_gate", "mlp_proj", "mlp_out", "block"):
        ref_name = f"block_{block:02d}" if name == "block" else f"b{block}_{name}"
        native_name = f"block_{block:02d}" if name == "block" else name
        a, b = [np.load(path, allow_pickle=False) for path in
                (reference / f"{ref_name}.npy", candidate / f"{native_name}.npy")]
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if b.ndim == 3 and b.shape[0] == 1:
            b = b[0]
        if a.dtype != np.float32 or b.dtype != np.float32 or a.ndim != 2 or a.shape != b.shape:
            raise ValueError(f"{name}: incompatible stage shapes")
        cosine, relative_l2 = _cosine_error(a, b)
        results[name] = dict(cosine=cosine, relative_l2=relative_l2,
                             bit_exact=bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
                             equal_fraction=float(np.mean(a == b)))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--block", required=True, type=int, choices=range(32))
    ap.add_argument("--require-exact", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    stages = compare_stages(args.reference, args.candidate, args.block)
    passed = all(x["bit_exact"] if args.require_exact else
                 x["cosine"] >= NONQUANTIZED_COSINE_THRESHOLD for x in stages.values())
    result = dict(diagnostic_only=True, full_model_acceptance=False,
                  require_exact=args.require_exact, passed=passed, stages=stages)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
