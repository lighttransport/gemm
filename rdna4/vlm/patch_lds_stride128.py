#!/usr/bin/env python3
"""Bisection variant of patch_lds_stride.py: try 144 -> 128 (Tensile default)
instead of 144 -> 160. If 128 also fails cosine, the patch-as-stride-rewrite
approach is structurally infeasible and we move to the libhipblaslt-bridge
production path (path A in the slice-2 strategy).

Stride 128 b128 lanes (no padding):
  A-buf  = 64 rows * 128 = 8192 bytes  (was 9216)
  B-base = 2 * 8192 = 16384 = 0x4000   (was 0x4800)
  B-base+768 = 0x4300                  (was 0x4b00)
  s9 = s6 * 4*stride = s6 * 0x200      (was 0x240)
  s8 = s6 * A-buf    = s6 * 0x2000     (was 0x2400)
  total LDS = 4 * 8192 = 32768         (was 36864)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


LINE_SUBS: list[tuple[str, str]] = [
    ("\tv_mul_u32_u24_e32 v172, 0x90, v172",
     "\tv_mul_u32_u24_e32 v172, 0x80, v172"),

    ("\tv_or_b32_e32 v169, 0x4800, v166",
     "\tv_or_b32_e32 v169, 0x4000, v166"),
    ("\tv_lshl_or_b32 v170, v170, 4, 0x4800",
     "\tv_lshl_or_b32 v170, v170, 4, 0x4000"),
    ("\tv_or_b32_e32 v171, 0x4b00, v166",
     "\tv_or_b32_e32 v171, 0x4300, v166"),

    ("\ts_mul_i32 s9, s6, 0x240",
     "\ts_mul_i32 s9, s6, 0x200"),
    ("\ts_mul_i32 s8, s6, 0x2400",
     "\ts_mul_i32 s8, s6, 0x2000"),

    # Prologue ds_store A bank (16/32/48 row * stride).
    ("\tds_store_b128 v166, v[133:136] offset:2304",
     "\tds_store_b128 v166, v[133:136] offset:2048"),
    ("\tds_store_b128 v166, v[137:140] offset:4608",
     "\tds_store_b128 v166, v[137:140] offset:4096"),
    ("\tds_store_b128 v166, v[141:144] offset:6912",
     "\tds_store_b128 v166, v[141:144] offset:6144"),

    # Prologue ds_store B bank (128/144/160/176 row * stride).
    ("\tds_store_b128 v166, v[145:148] offset:18432",
     "\tds_store_b128 v166, v[145:148] offset:16384"),
    ("\tds_store_b128 v166, v[149:152] offset:20736",
     "\tds_store_b128 v166, v[149:152] offset:18432"),
    ("\tds_store_b128 v166, v[153:156] offset:23040",
     "\tds_store_b128 v166, v[153:156] offset:20480"),
    ("\tds_store_b128 v166, v[157:160] offset:25344",
     "\tds_store_b128 v166, v[157:160] offset:22528"),

    # bb.5 ds_stores via v174 (A bank).
    ("\tds_store_b128 v174, v[133:136] offset:2304",
     "\tds_store_b128 v174, v[133:136] offset:2048"),
    ("\tds_store_b128 v174, v[141:144] offset:4608",
     "\tds_store_b128 v174, v[141:144] offset:4096"),
    ("\tds_store_b128 v174, v[137:140] offset:6912",
     "\tds_store_b128 v174, v[137:140] offset:6144"),

    # bb.5 ds_stores via v175 (B bank).
    ("\tds_store_b128 v175, v[153:156] offset:2304",
     "\tds_store_b128 v175, v[153:156] offset:2048"),
    ("\tds_store_b128 v175, v[149:152] offset:4608",
     "\tds_store_b128 v175, v[149:152] offset:4096"),
    ("\tds_store_b128 v175, v[145:148] offset:6912",
     "\tds_store_b128 v175, v[145:148] offset:6144"),

    # Mainloop bb.4 ds_loads (row-2 jump).
    ("\tds_load_b128 v[206:209], v218 offset:4608",
     "\tds_load_b128 v[206:209], v218 offset:4096"),
    ("\tds_load_b128 v[210:213], v218 offset:4864",
     "\tds_load_b128 v[210:213], v218 offset:4352"),
    ("\tds_load_b128 v[214:217], v218 offset:5120",
     "\tds_load_b128 v[214:217], v218 offset:4608"),
    ("\tds_load_b128 v[218:221], v218 offset:5376",
     "\tds_load_b128 v[218:221], v218 offset:4864"),
    ("\tds_load_b128 v[222:225], v230 offset:4608",
     "\tds_load_b128 v[222:225], v230 offset:4096"),
    ("\tds_load_b128 v[226:229], v230 offset:4864",
     "\tds_load_b128 v[226:229], v230 offset:4352"),
    ("\tds_load_b128 v[230:233], v230 offset:5120",
     "\tds_load_b128 v[230:233], v230 offset:4608"),
    ("\tds_load_b128 v[234:237], v234 offset:4608",
     "\tds_load_b128 v[234:237], v234 offset:4096"),

    # LDS allocation: 36864 -> 32768.
    ("\t\t.amdhsa_group_segment_fixed_size 36864",
     "\t\t.amdhsa_group_segment_fixed_size 32768"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--out-file", required=True)
    args = ap.parse_args()

    src = Path(args.in_file).read_text()
    out = src
    miss = []
    for old, new in LINE_SUBS:
        if old not in out:
            miss.append(old.strip())
            continue
        if out.count(old) != 1:
            print(
                f"patch_lds_stride128: ambiguous match ({out.count(old)}x): "
                f"{old.strip()!r}",
                file=sys.stderr,
            )
            return 2
        out = out.replace(old, new, 1)
    if miss:
        print("patch_lds_stride128: missing sites:", file=sys.stderr)
        for m in miss:
            print("  -", m, file=sys.stderr)
        return 1

    Path(args.out_file).write_text(out)
    print(f"patch_lds_stride128: wrote {args.out_file} ({len(LINE_SUBS)} sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
