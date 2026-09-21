#!/usr/bin/env python3
"""Compare replayed attention without hiding target error in a long prefix."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare import NONQUANTIZED_COSINE_THRESHOLD, _cosine_error


def metrics(reference, candidate, target_tokens):
    if (reference.ndim != 2 or candidate.shape != reference.shape or
            not 0 < target_tokens <= len(reference)):
        raise ValueError("invalid attention shapes or target token count")
    prefix = len(reference) - target_tokens
    result = {"threshold": NONQUANTIZED_COSINE_THRESHOLD, "regions": {}}
    for name, start, end in (("all", 0, len(reference)), ("prefix", 0, prefix),
                             ("target", prefix, len(reference))):
        if start == end:
            continue
        a, b = reference[start:end], candidate[start:end]
        cosine, relative_l2 = _cosine_error(a, b)
        result["regions"][name] = {"cosine": cosine, "relative_l2": relative_l2,
                                   "equal_fraction": float(np.mean(a == b)),
                                   "passed": cosine >= NONQUANTIZED_COSINE_THRESHOLD}
    result["passed"] = all(x["passed"] for x in result["regions"].values())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--target-tokens", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    result = metrics(np.load(args.reference, allow_pickle=False),
                     np.load(args.candidate, allow_pickle=False), args.target_tokens)
    result.update(reference=str(args.reference.resolve()), candidate=str(args.candidate.resolve()))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
