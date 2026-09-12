#!/usr/bin/env python3
"""Compare bounded logit dumps only where both runs used identical tokens."""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def inputs(directory):
    entries = re.findall(r"TOKEN pos=(\d+) input=(\d+) next=\d+",
                         (directory / "inference.rank00.log").read_text())
    result = {int(pos): int(token) for pos, token in entries}
    if len(result) != len(entries):
        raise ValueError(f"duplicate token positions in {directory}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-relative-rms", type=float, default=0.01)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if (args.start < 0 or args.count < 1 or not 0 <= args.min_cosine <= 1
            or not 0 <= args.max_relative_rms < float("inf")):
        parser.error("invalid range or numerical gates")
    reference_inputs, actual_inputs = inputs(args.reference), inputs(args.actual)
    for pos in range(args.start + args.count):
        if (pos not in reference_inputs or pos not in actual_inputs
                or reference_inputs[pos] != actual_inputs[pos]):
            raise ValueError(f"missing/different input token at position {pos}; use fixed-token replay")
    rows = []
    for pos in range(args.start, args.start + args.count):
        vectors = [np.fromfile(directory / f"logits.pos{pos}.bin", dtype="<f4")
                   for directory in (args.reference, args.actual)]
        if any(x.size != 129280 or not np.isfinite(x).all() for x in vectors):
            raise ValueError(f"invalid logit dump at position {pos}")
        exact = bool(np.array_equal(vectors[0].view("<u4"), vectors[1].view("<u4")))
        a, b = (x.astype(np.float64) for x in vectors)
        aa, bb = np.linalg.norm(a), np.linalg.norm(b)
        if aa == 0 or bb == 0:
            raise ValueError(f"zero logit vector at position {pos}")
        cosine = float(np.clip(np.dot(a, b) / aa / bb, -1, 1))
        relative = float(np.linalg.norm(a - b) / aa)
        same_argmax = bool(a.argmax() == b.argmax())
        passed = (same_argmax and cosine >= args.min_cosine
                  and relative <= args.max_relative_rms and (exact or not args.require_exact))
        rows.append(dict(position=pos, cosine=cosine, relative_rms=relative,
                         max_absolute=float(np.max(np.abs(a - b))),
                         argmax_same=same_argmax, bit_exact=exact, passed=passed))
    result = dict(reference=str(args.reference), actual=str(args.actual),
                  min_cosine=args.min_cosine, max_relative_rms=args.max_relative_rms,
                  require_exact=args.require_exact, passed=all(row["passed"] for row in rows),
                  positions=rows)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.write_text(encoded)
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
