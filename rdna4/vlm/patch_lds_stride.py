#!/usr/bin/env python3
"""Patch the barriersig_early baseline to use LDS row stride 160 b128 lanes
instead of 144, matching hipBLASLt's MT128x128x32 PGR2 kernel layout.

Item 1 of the post-DTVA structural rewrite (see rdna4_gemm_optimization_log.md
"## RDNA4 schedule sensitivity — five negative results converge"):

  ours        : LDS stride 144 b128  (row=2304B, A-buf=9216B, B-base=18432B)
  hipBLASLt   : LDS stride 160 b128  (row=2560B, A-buf=10240B, B-base=20480B)
  total LDS   : 36864B -> 40960B (still well under RDNA4 64 KiB per WGP cap)

Sites changed (all match exact instruction text; no line-number coupling):

  v_or_b32_e32 v169, 0x4800, v166                  -> 0x5000
  v_lshl_or_b32 v170, v170, 4, 0x4800              -> 0x5000
  v_or_b32_e32 v171, 0x4b00, v166                  -> 0x5300
  s_mul_i32 s9, s6, 0x240                          -> 0x280
  s_mul_i32 s8, s6, 0x2400                         -> 0x2800

  ds_store_b128 ... offset:2304/4608/6912          -> 2560/5120/7680
  ds_store_b128 ... offset:18432/20736/23040/25344 -> 20480/23040/25600/28160
  ds_load_b128  ... offset:4608/4864/5120/5376     -> 5120/5376/5632/5888

  .amdhsa_group_segment_fixed_size 36864           -> 40960

Sub-row offsets 256/512/768 in ds_load are within-row (lane-stride-independent)
and stay unchanged. Likewise the 0x2400 / 0x4800 immediates that appear in
v_mad_co_i64_i32 with v1/v2/v0 (input matrix row-stride 4608*2 bytes, output
matrix row-stride 4608*4 bytes) are unrelated to LDS layout and stay unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Whole-line replacements (exact match required).
LINE_SUBS: list[tuple[str, str]] = [
    # B-bank base bakes 0x4800 (=18432) into v169/v170 with sub-offsets.
    # Per-thread row-base multiply: v172 = (lane-row-index) * stride.
    ("\tv_mul_u32_u24_e32 v172, 0x90, v172",
     "\tv_mul_u32_u24_e32 v172, 0xa0, v172"),

    ("\tv_or_b32_e32 v169, 0x4800, v166",
     "\tv_or_b32_e32 v169, 0x5000, v166"),
    ("\tv_lshl_or_b32 v170, v170, 4, 0x4800",
     "\tv_lshl_or_b32 v170, v170, 4, 0x5000"),
    # v171 = B-base + intra-row sub-offset 768 = 0x4b00.
    # New: 0x5000 + 768 = 0x5300.
    ("\tv_or_b32_e32 v171, 0x4b00, v166",
     "\tv_or_b32_e32 v171, 0x5300, v166"),

    # LDS double-buffer toggle: lane units (s9 = s6 * 4*stride).
    ("\ts_mul_i32 s9, s6, 0x240",
     "\ts_mul_i32 s9, s6, 0x280"),
    # Byte units (s8 = s6 * 2*A-buf-bytes, used in bb.5 v174/v175 setup).
    ("\ts_mul_i32 s8, s6, 0x2400",
     "\ts_mul_i32 s8, s6, 0x2800"),

    # Prologue ds_store row-stride for A bank (offsets 2304/4608/6912).
    ("\tds_store_b128 v166, v[133:136] offset:2304",
     "\tds_store_b128 v166, v[133:136] offset:2560"),
    ("\tds_store_b128 v166, v[137:140] offset:4608",
     "\tds_store_b128 v166, v[137:140] offset:5120"),
    ("\tds_store_b128 v166, v[141:144] offset:6912",
     "\tds_store_b128 v166, v[141:144] offset:7680"),

    # Prologue ds_store for B bank: 18432/20736/23040/25344
    # -> 20480/23040/25600/28160.  (Note: 20736 -> 23040 collides numerically
    # with old 23040 stride-144 value; that's why this is a single-shot
    # textual sub on the EXACT instruction text, not a global rename.)
    ("\tds_store_b128 v166, v[145:148] offset:18432",
     "\tds_store_b128 v166, v[145:148] offset:20480"),
    ("\tds_store_b128 v166, v[149:152] offset:20736",
     "\tds_store_b128 v166, v[149:152] offset:23040"),
    ("\tds_store_b128 v166, v[153:156] offset:23040",
     "\tds_store_b128 v166, v[153:156] offset:25600"),
    ("\tds_store_b128 v166, v[157:160] offset:25344",
     "\tds_store_b128 v166, v[157:160] offset:28160"),

    # bb.5 ds_stores to A bank via v174 (offsets 2304/4608/6912).
    ("\tds_store_b128 v174, v[133:136] offset:2304",
     "\tds_store_b128 v174, v[133:136] offset:2560"),
    ("\tds_store_b128 v174, v[141:144] offset:4608",
     "\tds_store_b128 v174, v[141:144] offset:5120"),
    ("\tds_store_b128 v174, v[137:140] offset:6912",
     "\tds_store_b128 v174, v[137:140] offset:7680"),

    # bb.5 ds_stores to B bank via v175 (which already has 0x5000 baked).
    # Stride-relative offsets 2304/4608/6912 within the B buffer -> 2560/5120/7680.
    ("\tds_store_b128 v175, v[153:156] offset:2304",
     "\tds_store_b128 v175, v[153:156] offset:2560"),
    ("\tds_store_b128 v175, v[149:152] offset:4608",
     "\tds_store_b128 v175, v[149:152] offset:5120"),
    ("\tds_store_b128 v175, v[145:148] offset:6912",
     "\tds_store_b128 v175, v[145:148] offset:7680"),

    # Mainloop bb.4 ds_loads — row-2 jump (4608/4864/5120/5376 -> 5120/5376/5632/5888).
    ("\tds_load_b128 v[206:209], v218 offset:4608",
     "\tds_load_b128 v[206:209], v218 offset:5120"),
    ("\tds_load_b128 v[210:213], v218 offset:4864",
     "\tds_load_b128 v[210:213], v218 offset:5376"),
    ("\tds_load_b128 v[214:217], v218 offset:5120",
     "\tds_load_b128 v[214:217], v218 offset:5632"),
    ("\tds_load_b128 v[218:221], v218 offset:5376",
     "\tds_load_b128 v[218:221], v218 offset:5888"),
    ("\tds_load_b128 v[222:225], v230 offset:4608",
     "\tds_load_b128 v[222:225], v230 offset:5120"),
    ("\tds_load_b128 v[226:229], v230 offset:4864",
     "\tds_load_b128 v[226:229], v230 offset:5376"),
    ("\tds_load_b128 v[230:233], v230 offset:5120",
     "\tds_load_b128 v[230:233], v230 offset:5632"),
    ("\tds_load_b128 v[234:237], v234 offset:4608",
     "\tds_load_b128 v[234:237], v234 offset:5120"),

    # LDS allocation in .amdhsa_kernel block.
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
    miss = []
    for old, new in LINE_SUBS:
        if old not in out:
            miss.append(old.strip())
            continue
        # Unique-occurrence assertion: every site must match exactly once.
        if out.count(old) != 1:
            print(
                f"patch_lds_stride: ambiguous match ({out.count(old)}x): "
                f"{old.strip()!r}",
                file=sys.stderr,
            )
            return 2
        out = out.replace(old, new, 1)
    if miss:
        print("patch_lds_stride: missing expected sites:", file=sys.stderr)
        for m in miss:
            print("  -", m, file=sys.stderr)
        return 1

    Path(args.out_file).write_text(out)
    print(f"patch_lds_stride: wrote {args.out_file} ({len(LINE_SUBS)} sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
