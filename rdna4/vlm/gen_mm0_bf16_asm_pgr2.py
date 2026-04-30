#!/usr/bin/env python3
"""
RDNA4 mm0 BF16 GEMM PGR2 generator — Track B home file.

Status: REAL EMITTER (initial slice). Reads
`generated/mm0_bf16_asm_barriersig_early.s` and emits
`generated/mm0_bf16_asm_pgr2.s` after applying the buffer-descriptor +
buffer_load swap. This first slice is **perf-neutral** vs the
`pgr2-bufload-bse` patch variant (~141.6 TFLOP/s, 72.6% peak) but lives
in the Track B home file so subsequent slices for structural items 1-4
land here as plain function additions instead of further patch variants
on the hipcc-compiled output.

Structural slices to land here next (each in its own emit_slice_*):

  slice 1 (DONE)  : buffer-descriptor prologue + buffer_load swap
                    (delegates to patch_pgr2_bufload).
  slice 2 (DONE)  : LDS row stride 144 -> 128 (Tensile default, no
                    padding). Coordinated rewrite of v172 multiplier
                    + B-base immediates (0x4800/0x4b00 -> 0x4000/0x4300)
                    + s8/s9 toggle scalars + prologue/bb.4/bb.5 offsets
                    + .amdhsa_group_segment_fixed_size 36864 -> 32768.
                    Prior 144 -> 160 attempt failed cosine=0.998 (the
                    +padding direction had a subtle offset miscalc);
                    144 -> 128 (-padding, matches Tensile) passes
                    cleanly.
  slice 3 (TODO)  : per-load distinct soffset SGPRs. 4 SGPRs hold
                    s0+{64,80,96,112}; 8 in-loop buffer_loads use
                    `sX offen` (no immediate offset). Saves the
                    immediate-decode bytes per load.
  slice 4 (TODO)  : W-tile partition rewrite 4 -> 8 b128/thread.
                    Halves WG dim on W axis; doubles per-thread W
                    fragments. Touches kernarg-derived address
                    arithmetic (v163/v164) and bb.4 ds_load patterns.
  slice 5 (TODO)  : VGPR pressure reduction to <=192. Reuses
                    accumulator regs across MIWaveTile iterations.
                    Unlocks 2 WGs/WGP from 1.

Once slices 2-5 are in, the SCHED slice (PGR2 distribution at slots
[0,2,4,6,8,10,12,14] for loads + [14..21] for ds_stores + barrier
slot 24) can be retried — the prior `pgr2-distribute-bse` patch
regressed on RDNA4 (137 TFLOP/s) because slices 2-5 weren't
co-applied.

The 4-pointer ABI is preserved so callers (`bench_vlm_gemm`,
`hip_vision_encoder.c`, etc.) are unaffected:

    void gemm_mm0_bf16_asm(float* Y, const bf16* W, const bf16* X,
                           const float* bias)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


# ---- Constants (verified from hipBLASLt 73624 disasm + Tensile SIA3) ----

SHAPE = dict(M=1024, N=4608, K=4608)
MT = (128, 128)
MI = (16, 16, 16)
DEPTH_U = 32
LOOP_ITERS = DEPTH_U // MI[2]                          # 2
NUM_MFMA_PER_ITER = 2 * 2 * 4                          # 16
TOTAL_WMMAS_PER_KSTAGE = NUM_MFMA_PER_ITER * LOOP_ITERS  # 32

SYNC_PLR_MFMA_IDX = 24
LW_END_MFMA_IDX = SYNC_PLR_MFMA_IDX - 3                # 21

# Buffer-resource flag word (RDNA4, verified from
# /tmp/hipblaslt_bf16_best_73823.s lines 654, 725:
# `s_mov_b32 s51, 0x30020000`).
SRD_FLAG_RDNA4 = 0x30020000

X_BYTES = SHAPE["M"] * SHAPE["K"] * 2                  # 0x00900000
W_BYTES = SHAPE["N"] * SHAPE["K"] * 2                  # 0x02880000

# PGR2 distribution map (used by future slice once 2-5 land).
GLOBAL_LOAD_SLOTS = [0, 2, 4, 6, 8, 10, 12, 14]
LOCAL_WRITE_SLOTS = [14, 15, 16, 17, 18, 19, 20, 21]
BARRIER_SLOT = 24


# ---- Slice 2 substitution table: LDS stride 144 -> 128 ----
#
# All sites are exact-text; each must occur exactly once. Matches the
# verified `patch_lds_stride128.py` table (28 sites). Inlined here so the
# pgr2 emitter owns the transform end-to-end.

LDS_STRIDE128_SUBS: list[tuple[str, str]] = [
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
    ("\tds_store_b128 v166, v[133:136] offset:2304",
     "\tds_store_b128 v166, v[133:136] offset:2048"),
    ("\tds_store_b128 v166, v[137:140] offset:4608",
     "\tds_store_b128 v166, v[137:140] offset:4096"),
    ("\tds_store_b128 v166, v[141:144] offset:6912",
     "\tds_store_b128 v166, v[141:144] offset:6144"),
    ("\tds_store_b128 v166, v[145:148] offset:18432",
     "\tds_store_b128 v166, v[145:148] offset:16384"),
    ("\tds_store_b128 v166, v[149:152] offset:20736",
     "\tds_store_b128 v166, v[149:152] offset:18432"),
    ("\tds_store_b128 v166, v[153:156] offset:23040",
     "\tds_store_b128 v166, v[153:156] offset:20480"),
    ("\tds_store_b128 v166, v[157:160] offset:25344",
     "\tds_store_b128 v166, v[157:160] offset:22528"),
    ("\tds_store_b128 v174, v[133:136] offset:2304",
     "\tds_store_b128 v174, v[133:136] offset:2048"),
    ("\tds_store_b128 v174, v[141:144] offset:4608",
     "\tds_store_b128 v174, v[141:144] offset:4096"),
    ("\tds_store_b128 v174, v[137:140] offset:6912",
     "\tds_store_b128 v174, v[137:140] offset:6144"),
    ("\tds_store_b128 v175, v[153:156] offset:2304",
     "\tds_store_b128 v175, v[153:156] offset:2048"),
    ("\tds_store_b128 v175, v[149:152] offset:4608",
     "\tds_store_b128 v175, v[149:152] offset:4096"),
    ("\tds_store_b128 v175, v[145:148] offset:6912",
     "\tds_store_b128 v175, v[145:148] offset:6144"),
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
    ("\t\t.amdhsa_group_segment_fixed_size 36864",
     "\t\t.amdhsa_group_segment_fixed_size 32768"),
]


def emit_slice_2_lds128(s: str) -> str:
    """Apply LDS stride 144 -> 128 rewrite. Each site must match exactly once."""
    out = s
    for old, new in LDS_STRIDE128_SUBS:
        if old not in out:
            raise RuntimeError(
                f"emit_slice_2_lds128: missing site: {old.strip()!r}"
            )
        if out.count(old) != 1:
            raise RuntimeError(
                f"emit_slice_2_lds128: ambiguous match ({out.count(old)}x): "
                f"{old.strip()!r}"
            )
        out = out.replace(old, new, 1)
    return out


# ---- Slice 1: buffer-descriptor prologue + buffer_load swap ----

def _load_patch_module(repo_root: Path):
    """Load patch_mm0_asm_schedule.py as a module (it's a sibling, not a
    package). Returns the module object; callers grab the patch fns from
    it. Done lazily so importing this generator for `--dry-run` doesn't
    require the patch script to be importable."""
    patch_path = repo_root / "patch_mm0_asm_schedule.py"
    if not patch_path.is_file():
        raise FileNotFoundError(
            f"missing sibling patch script: {patch_path}\n"
            "  the gen_mm0_bf16_asm_pgr2 emitter delegates the buffer_load\n"
            "  swap to patch_mm0_asm_schedule.patch_pgr2_bufload until\n"
            "  slice 1's logic is inlined."
        )
    spec = importlib.util.spec_from_file_location("patch_mm0_asm_schedule", patch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build module spec for {patch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit_slice_1_bufload(baseline_s: str, *, repo_root: Path) -> str:
    """Apply buffer-descriptor + buffer_load swap to the
    `mm0_bf16_asm_barriersig_early.s` baseline.

    Delegates to `patch_pgr2_bufload` in patch_mm0_asm_schedule.py. The
    delegation is intentional: that function is verified (its
    `pgr2-bufload-bse` build passes correctness at 141.6 TFLOP/s), so
    re-implementing its byte-exact transform here would be redundant and
    add divergence risk. When slice 2 (LDS stride 160) lands, it will be
    inlined — at that point the dependent edits make a single coherent
    table cleaner than two patch passes.
    """
    pmod = _load_patch_module(repo_root)
    return pmod.patch_pgr2_bufload(baseline_s)


# ---- Driver ----

def usage_summary() -> str:
    return (
        "PGR2 generator for RDNA4 mm0 BF16 GEMM (Track B home file).\n"
        f"  shape: M={SHAPE['M']} N={SHAPE['N']} K={SHAPE['K']}\n"
        f"  schedule plan: GR slots {GLOBAL_LOAD_SLOTS}, "
        f"LW slots {LOCAL_WRITE_SLOTS}, barrier @ {BARRIER_SLOT}\n"
        f"  SRDs: X NumRecords=0x{X_BYTES:08X}, "
        f"W NumRecords=0x{W_BYTES:08X}, flag 0x{SRD_FLAG_RDNA4:08X}\n"
        "  current slice: 1 (buffer_load swap, perf-neutral baseline)\n"
        "  output: <out-dir>/<basename>.s (consumed by Makefile to produce .co)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--in-file",
        default="generated/mm0_bf16_asm_barriersig_early.s",
        help="Input baseline assembly (default: generated/mm0_bf16_asm_barriersig_early.s)",
    )
    parser.add_argument(
        "--out-file",
        default="generated/mm0_bf16_asm_pgr2.s",
        help="Output assembly path (default: generated/mm0_bf16_asm_pgr2.s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the design summary and exit without emitting",
    )
    parser.add_argument(
        "--with-sched",
        action="store_true",
        help="Also apply PGR2 distribution (8 buffer_loads -> WMMA stream)",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(usage_summary())
        return 0

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    repo_root = Path(__file__).resolve().parent

    if not in_path.is_file():
        print(
            f"gen_mm0_bf16_asm_pgr2: missing input baseline: {in_path}\n"
            "  build it first with `make generated/mm0_bf16_asm_barriersig_early.s`",
            file=sys.stderr,
        )
        return 1

    baseline = in_path.read_text()

    out = emit_slice_1_bufload(baseline, repo_root=repo_root)
    out = emit_slice_2_lds128(out)
    if args.with_sched:
        pmod = _load_patch_module(repo_root)
        out = pmod.patch_pgr2_distribute(out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out)
    sched_tag = "+sched" if args.with_sched else ""
    print(f"gen_mm0_bf16_asm_pgr2: wrote {out_path} (slices 1+2{sched_tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
