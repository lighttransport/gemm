#!/usr/bin/env python3
"""Degeneration metrics for a generated id stream.

Eyeballing the first few hundred characters does not show whether a long
generation collapsed later on, which is exactly the failure mode that matters for
long output. Reports, over the whole stream and over its last quarter:

  distinct-n   fraction of n-grams that are unique (low => repetitive)
  max-run      longest immediately repeated block (a hard loop shows as a big run)
  top-4gram    most frequent 4-gram and its count

Usage: repetition.py gen.ids [gen2.ids ...]
"""
import sys
from collections import Counter


def distinct_n(ids, n):
    if len(ids) < n:
        return 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def max_repeat_run(ids):
    """Longest L such that some block of length L repeats back-to-back."""
    best = 0
    n = len(ids)
    for L in range(1, min(64, n // 2) + 1):
        run = 0
        for i in range(n - L):
            if ids[i] == ids[i + L]:
                run += 1
                if run >= L:          # a full period matched
                    best = max(best, L)
            else:
                run = 0
    return best


def report(path):
    ids = [int(x) for x in open(path).read().split()]
    tail = ids[len(ids) * 3 // 4:]
    print(f"{path}: {len(ids)} tokens")
    for label, seq in (("all ", ids), ("last", tail)):
        d1, d2, d3 = distinct_n(seq, 1), distinct_n(seq, 2), distinct_n(seq, 3)
        c = Counter(tuple(seq[i:i + 4]) for i in range(max(0, len(seq) - 3)))
        top, cnt = c.most_common(1)[0] if c else ((), 0)
        print(f"  {label} n={len(seq):5d}  distinct-1={d1:.3f} distinct-2={d2:.3f} "
              f"distinct-3={d3:.3f}  max-run={max_repeat_run(seq):3d}  top-4gram x{cnt}")
    # a healthy sample sits around distinct-3 > 0.9; a hard loop drives it toward 0
    d3all = distinct_n(ids, 3)
    print(f"  => {'OK' if d3all > 0.85 else 'DEGENERATE'} (distinct-3 = {d3all:.3f})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
