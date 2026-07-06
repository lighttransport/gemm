#!/usr/bin/env python3
"""Compare greedy token outputs captured by bench_kv_accuracy.sh across two KV configs.
Reports per-prompt top-1 agreement and first-divergence depth. Usage: bench_kv_accuracy_diff.py bf16 int8"""
import sys, glob, os

a, b = sys.argv[1], sys.argv[2]
da, db = f"/tmp/kvacc_{a}", f"/tmp/kvacc_{b}"
files = sorted(os.path.basename(f) for f in glob.glob(f"{da}/p*.txt"))
tot_match = tot = 0
print(f"KV accuracy: {a} (reference) vs {b}")
print(f"{'prompt':8} {'len':>4} {'agree':>7} {'first_div':>9}")
for f in files:
    ta = open(f"{da}/{f}").read().split()
    tb = open(f"{db}/{f}").read().split()
    n = min(len(ta), len(tb))
    match = sum(1 for i in range(n) if ta[i] == tb[i])
    div = next((i for i in range(n) if ta[i] != tb[i]), None)
    tot_match += match; tot += n
    print(f"{f:8} {n:4d} {match/n*100:6.1f}% {('none' if div is None else div):>9}")
print(f"{'OVERALL':8} {tot:4d} {tot_match/tot*100:6.1f}%  (top-1 greedy token agreement over {len(files)} prompts)")
