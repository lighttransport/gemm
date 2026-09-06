#!/usr/bin/env python3
"""Gate fast FP8 generations against fp8-exact or BF16 JSONL references.

Each JSONL record needs ``id`` and either ``ids`` or ``text``.  The command exits
non-zero when exact-token agreement or normalized text similarity falls below
the requested floor, making it suitable for a benchmark/run script gate.
"""

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {str(row["id"]): row for row in rows}


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def compare(reference, candidate):
    keys = sorted(set(reference) & set(candidate))
    if not keys:
        raise ValueError("reference and candidate have no common ids")
    token_equal = 0
    similarities = []
    details = []
    for key in keys:
        ref, got = reference[key], candidate[key]
        rids, gids = ref.get("ids"), got.get("ids")
        exact = rids is not None and gids is not None and rids == gids
        token_equal += int(exact)
        rt, gt = str(ref.get("text", "")), str(got.get("text", ""))
        denom = max(len(rt), len(gt), 1)
        similarity = 1.0 - float(edit_distance(rt, gt)) / denom
        similarities.append(similarity)
        details.append({"id": key, "token_exact": exact,
                        "text_similarity": round(similarity, 6)})
    return {"cases": len(keys),
            "token_exact_rate": float(token_equal) / len(keys),
            "mean_text_similarity": sum(similarities) / len(similarities),
            "details": details}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--min-token-exact", type=float, default=0.80)
    p.add_argument("--min-text-similarity", type=float, default=0.95)
    args = p.parse_args(argv)
    try:
        result = compare(load(args.reference), load(args.candidate))
    except (OSError, ValueError, KeyError) as e:
        p.error(str(e))
    passed = result["token_exact_rate"] >= args.min_token_exact and \
        result["mean_text_similarity"] >= args.min_text_similarity
    result["passed"] = passed
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
