#!/usr/bin/env python3
"""Compare completed TF_DUMP_TOKENS traces (selected logits, not full logits)."""

import argparse
import math
import re
from pathlib import Path


TOKEN = re.compile(r"qwen38: token n=(\d+) pos=(\d+) id=(\d+) logit=(\S+)")
BENCH = re.compile(r"qwen38: bench-token trial=(\d+) warmup=([01]) n=(\d+) pos=(\d+) id=(\d+) logit=(\S+)")
DECODE = re.compile(r"qwen38: decode=(\d+) tokens")


def read_trace(path, count):
    text = Path(path).read_text()
    rows = [(int(n), int(pos), int(token), float.fromhex(logit))
            for n, pos, token, logit in TOKEN.findall(text)]
    if len(rows) != count or [row[0] for row in rows] != list(range(count)):
        raise ValueError(f"{path}: expected {count} ordered token records")
    if DECODE.findall(text) != [str(count)]:
        raise ValueError(f"{path}: missing or ambiguous completed decode summary")
    if not all(math.isfinite(row[3]) for row in rows):
        raise ValueError(f"{path}: nonfinite selected logit")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--max-logit-error", type=float, required=True,
                        help="absolute tolerance; use 0 for repeatability")
    parser.add_argument("--benchmark-trials", type=int, default=0,
                        help="compare every recorded benchmark trial, including warmup")
    args = parser.parse_args()
    if (args.tokens < 1 or args.benchmark_trials < 0 or not math.isfinite(args.max_logit_error)
            or args.max_logit_error < 0):
        parser.error("tokens must be positive and tolerance finite/nonnegative")
    try:
        ref = read_trace(args.reference, args.tokens)
        if args.benchmark_trials:
            text = Path(args.candidate).read_text()
            records = BENCH.findall(text)
            if len(records) != args.benchmark_trials * args.tokens:
                raise ValueError("incomplete benchmark token records")
            candidates = []
            for trial in range(args.benchmark_trials):
                rows = [(int(n), int(pos), int(token), float.fromhex(logit))
                        for t, warm, n, pos, token, logit in records if int(t) == trial]
                if len(rows) != args.tokens or [r[0] for r in rows] != list(range(args.tokens)):
                    raise ValueError(f"missing or unordered benchmark trial {trial}")
                if not all(math.isfinite(r[3]) for r in rows):
                    raise ValueError("nonfinite benchmark logit")
                candidates.append(rows)
            if not re.search(r"^pp\+tg,\d+," + str(args.tokens) + r",\d+,\d+,", text, re.M):
                raise ValueError("missing completed benchmark CSV summary")
        else:
            candidates = [read_trace(args.candidate, args.tokens)]
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    errors = []
    for trial, got in enumerate(candidates):
        for expected, actual in zip(ref, got):
            if expected[:3] != actual[:3]:
                print(f"FAIL: trial={trial} token/position mismatch: {expected[:3]} != {actual[:3]}")
                return 1
        errors.extend(abs(expected[3] - actual[3]) for expected, actual in zip(ref, got))
    maximum = max(errors)
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    passed = maximum <= args.max_logit_error
    print(f"{'PASS' if passed else 'FAIL'}: tokens={args.tokens}/{args.tokens} "
          f"trials={len(candidates)} selected_logit_max_abs={maximum:.9g} rms={rms:.9g} "
          f"tolerance={args.max_logit_error:.9g}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
