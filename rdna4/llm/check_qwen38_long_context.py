#!/usr/bin/env python3
"""Validate deterministic Q8/Q8 long-context benchmark output."""
import argparse
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise SystemExit(f"long-context gate FAIL: {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--depth", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefill-floor", type=float, default=400.0)
    parser.add_argument("--decode-floor", type=float, default=32.0)
    parser.add_argument("--label", default="target")
    args = parser.parse_args()
    text = args.log.read_text(errors="replace")
    require("Result: PASS" in text, "runner did not report PASS")
    depth = re.search(
        rf"Depth prefill: {args.depth} random tokens in .*? -> ([0-9.]+) tok/s .*?seed=1",
        text)
    require(depth, f"missing randomized depth={args.depth} prefill")
    prefill = float(depth.group(1))
    decode = [float(value) for value in
              re.findall(r"^Decode:.*? -> ([0-9.]+) tok/s", text, re.MULTILINE)]
    hashes = re.findall(r"sequence hash=([0-9a-f]+)", text)
    require(len(decode) == args.repeats,
            f"expected {args.repeats} decode measurements, got {len(decode)}")
    require(len(hashes) == args.repeats,
            f"expected {args.repeats} hashes, got {len(hashes)}")
    require(len(set(hashes)) == 1, f"nondeterministic hashes: {hashes}")
    require(prefill >= args.prefill_floor,
            f"prefill {prefill:.2f} < {args.prefill_floor:.2f} tok/s")
    require(min(decode) >= args.decode_floor,
            f"decode {min(decode):.2f} < {args.decode_floor:.2f} tok/s")
    print(f"{args.label}: prefill={prefill:.2f} tok/s "
          f"decode_min={min(decode):.2f} tok/s hash={hashes[0]} PASS")


if __name__ == "__main__":
    main()
