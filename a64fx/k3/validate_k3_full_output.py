#!/usr/bin/env python3
"""Validate a K3 full-runner output outside the inference job.

The runner deliberately writes raw token IDs.  With --expected-ids this is an
exact greedy-token comparison; without it the tool still checks the output
format, token count, and agreement of all rank sidecars.
"""
from __future__ import print_function

import argparse
import glob
import re
import sys


def read_ids(path):
    values = []
    with open(path) as f:
        for line in f:
            for item in line.replace(",", " ").split():
                if item.startswith("#"):
                    break
                values.append(int(item))
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--expected-ids")
    ap.add_argument("--nodes", type=int, default=96)
    ap.add_argument("--mode", choices=("full96", "layer12", "synthetic12"))
    args = ap.parse_args()
    with open(args.output) as f:
        lines = [line.rstrip("\n") for line in f]
    if not lines or not (lines[0].startswith("K3FULLV1 ") or
                         lines[0].startswith("K3FULLV2 ")):
        raise SystemExit("invalid K3FULL header")
    if "status=PASS" not in lines[0]:
        raise SystemExit("runner did not report PASS")
    mode_match = re.search(r"(?:^| )mode=([^ ]+)", lines[0])
    if args.mode and (not mode_match or mode_match.group(1) != args.mode):
        raise SystemExit("runner mode does not match --mode")
    m = re.search(r"generated_tokens=(\d+)", lines[0])
    if not m:
        raise SystemExit("generated_tokens is missing")
    expected_count = int(m.group(1))
    id_lines = [line for line in lines if line.startswith("generated_ids:")]
    if len(id_lines) != 1:
        raise SystemExit("generated_ids line is missing or duplicated")
    ids = [int(x) for x in id_lines[0].split(":", 1)[1].split()]
    if len(ids) != expected_count:
        raise SystemExit("generated ID count %d != header count %d" %
                         (len(ids), expected_count))
    if any(x < 0 or x >= 163840 for x in ids):
        raise SystemExit("generated token is outside the K3 vocabulary")
    if args.expected_ids:
        expected = read_ids(args.expected_ids)
        if ids != expected:
            first = next((i for i, pair in enumerate(zip(ids, expected))
                          if pair[0] != pair[1]), min(len(ids), len(expected)))
            raise SystemExit("greedy output mismatch at index %d: got=%s expected=%s" %
                             (first, ids[first:first + 1], expected[first:first + 1]))
    sidecars = sorted(glob.glob(args.output + ".rank[0-9][0-9][0-9]"))
    if len(sidecars) != args.nodes:
        raise SystemExit("rank sidecars: found %d, expected %d" %
                         (len(sidecars), args.nodes))
    records = []
    for path in sidecars:
        with open(path) as f:
            line = f.readline().strip()
        match = re.search(r"generated=(\d+) hash=([0-9a-f]+) final=(-?\d+)"
                          r"(?: route_hash=([0-9a-f]+) collectives=(\d+)"
                          r" hidden_hash=([0-9a-f]+))?", line)
        if not match:
            raise SystemExit("malformed rank sidecar %s" % path)
        records.append((int(match.group(1)), match.group(2), int(match.group(3)),
                        match.group(4) or "", int(match.group(5) or -1),
                        match.group(6) or ""))
    if len(set(records)) != 1:
        raise SystemExit("rank sidecars disagree: %s" % records[:4])
    print("K3_FULL_OUTPUT PASS mode=%s tokens=%d ranks=%d exact_reference=%s hash=%s" %
          (mode_match.group(1) if mode_match else "legacy", len(ids),
           len(sidecars), bool(args.expected_ids), records[0][1]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("validate_k3_full_output: %s" % exc, file=sys.stderr)
        sys.exit(2)
