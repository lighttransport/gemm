#!/usr/bin/env python3
"""Compare generated (position, token id) traces between runner/q38d logs.
usage: compare_tokens.py REF.log CAND.log"""
import re, sys

PAT = re.compile(r'(?:bench-token trial=0 warmup=\d+ |q38d: token )n=(\d+) pos=(\d+) id=(\d+) logit=(\S+)')

def load(path):
    out = []
    with open(path, errors='replace') as f:
        for m in PAT.finditer(f.read()):
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), float.fromhex(m.group(4))))
    return out

def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    if not a or not b:
        print(f"MISSING ref={len(a)} cand={len(b)}"); return 2
    n = min(len(a), len(b))
    first = None
    maxdl = 0.0
    for i in range(n):
        if (a[i][1], a[i][2]) != (b[i][1], b[i][2]):
            first = i; break
        maxdl = max(maxdl, abs(a[i][3] - b[i][3]))
    same = n if first is None else first
    print(f"ref={len(a)} cand={len(b)} matched_prefix={same}/{n} "
          f"first_divergence={'none' if first is None else f'n={a[first][0]} pos={a[first][1]} ref_id={a[first][2]} cand_id={b[first][2]}'} "
          f"max_abs_logit_diff_on_prefix={maxdl:.4g}")
    return 0 if first is None and len(a) == len(b) else 1

if __name__ == '__main__':
    sys.exit(main())
