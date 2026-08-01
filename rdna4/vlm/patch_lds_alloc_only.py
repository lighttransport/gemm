#!/usr/bin/env python3
"""Bisection sanity check: only bump .amdhsa_group_segment_fixed_size to 40960
without changing any addresses. If this still produces correct output at
~142 TFLOP/s, the build pipeline is fine and the lds160 bug must be in
stride-related changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


LINE_SUBS = [
    ("\t\t.amdhsa_group_segment_fixed_size 36864",
     "\t\t.amdhsa_group_segment_fixed_size 40960"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--out-file", required=True)
    args = ap.parse_args()

    src = Path(args.in_file).read_text()
    out = src
    for old, new in LINE_SUBS:
        if out.count(old) != 1:
            print(f"missing or ambiguous: {old.strip()!r}", file=sys.stderr)
            return 1
        out = out.replace(old, new, 1)
    Path(args.out_file).write_text(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
