#!/usr/bin/env python3
"""Summarize comparable DS4F-0731 rank reports and choose the runtime A/B."""

import argparse
import re
from pathlib import Path


def values(root: Path, phase: str):
    pat = re.compile(rf"^{phase}:\s+\d+ tok\s+[-+0-9.eE]+ ms/tok\s+([-+0-9.eE]+) tok/s")
    out = []
    for p in sorted(root.glob("ds4f_ep_perf_rank*.txt")):
        for line in p.read_text(errors="replace").splitlines():
            m = pat.search(line)
            if m:
                out.append(float(m.group(1)))
                break
    return out


def one(root: Path):
    result = {}
    for phase in ("prefill", "decode"):
        v = values(root, phase)
        if v:
            result[f"{phase}_min"] = min(v)
            result[f"{phase}_max"] = max(v)
            result[f"{phase}_mean"] = sum(v) / len(v)
            result[f"{phase}_ranks"] = len(v)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("polished", type=Path)
    args = ap.parse_args()
    b, p = one(args.baseline), one(args.polished)
    print(f"BASELINE={args.baseline}")
    print(f"POLISHED={args.polished}")
    for name, data in (("baseline", b), ("polished", p)):
        for phase in ("prefill", "decode"):
            if f"{phase}_mean" in data:
                print(f"{name}_{phase}_tok_s="
                      f"{data[f'{phase}_mean']:.3f} "
                      f"range={data[f'{phase}_min']:.3f}-{data[f'{phase}_max']:.3f} "
                      f"ranks={int(data[f'{phase}_ranks'])}")
            else:
                print(f"{name}_{phase}_tok_s=missing")

    improvements = []
    for phase in ("prefill", "decode"):
        if f"{phase}_mean" in b and f"{phase}_mean" in p and b[f"{phase}_mean"] > 0:
            delta = 100.0 * (p[f"{phase}_mean"] / b[f"{phase}_mean"] - 1.0)
            improvements.append(delta)
            print(f"{phase}_delta_pct={delta:.2f}")
    if not improvements:
        print("DECISION=INCONCLUSIVE missing comparable metrics")
        return 2
    best = max(improvements)
    decision = "KEEP_POLISH" if best >= 3.0 else "KEEP_HARDENING_DISABLE_PERF_LEVER"
    print(f"BEST_DELTA_PCT={best:.2f}")
    print(f"DECISION={decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
