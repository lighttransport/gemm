#!/usr/bin/env python3
"""Patch mm0_bf16_directa_pf.s to eliminate the 32 v_dual_mov_b32 swap copies
at top of bb.2 by retargeting global_load destinations directly to a-slots.

Compiler emits per-iter: load X K+1 → na-slots v[129..160], then top-of-loop
copies na→a (v[177..208]) via 32 v_dual_mov_b32. We retarget loads to land in
a-slots directly. The async global_load doesn't overwrite a-slots until completion
(waited at top of next iter), so iter K's WMMAs still see K-th data — but ONLY
if the load issue point is AFTER bb.4's WMMAs. So we move the X prefetch from
bb.3 → bb.5 (after barrier_wait, just before s_branch).

Saves: 32 cycles/iter × 144 iters = ~4600 cycles/CTA (~19% of busy time).
Risks: if RDNA4 retires global_load writes mid-WMMA-stream, would corrupt
operands. Per AMDGPU semantics, writebacks queue until s_wait_loadcnt; so
WMMAs in bb.4 (no intervening loadcnt) read the snapshot pre-overwrite.

Mapping (from existing dual_mov pairs at pf.s lines 161-182):
  na-slot v[129:132] (frag0_h0) → a-slot v[205:208]
  na-slot v[133:136] (frag0_h1) → a-slot v[189:192]
  na-slot v[137:140] (frag1_h0) → a-slot v[201:204]
  na-slot v[141:144] (frag1_h1) → a-slot v[185:188]
  na-slot v[145:148] (frag2_h0) → a-slot v[197:200]
  na-slot v[149:152] (frag2_h1) → a-slot v[181:184]
  na-slot v[153:156] (frag3_h0) → a-slot v[193:196]
  na-slot v[157:160] (frag3_h1) → a-slot v[177:180]
"""
from __future__ import annotations
import argparse
from pathlib import Path


def patch(src: str) -> str:
    # ---------- 1) Retarget prologue X loads (lines 130-137 in input). ----------
    prologue_old = """\
\ts_clause 0x7
\tglobal_load_b128 v[129:132], v[211:212], off
\tglobal_load_b128 v[133:136], v[211:212], off offset:32
\tglobal_load_b128 v[137:140], v[213:214], off
\tglobal_load_b128 v[141:144], v[213:214], off offset:32
\tglobal_load_b128 v[145:148], v[215:216], off
\tglobal_load_b128 v[149:152], v[215:216], off offset:32
\tglobal_load_b128 v[153:156], v[217:218], off
\tglobal_load_b128 v[157:160], v[217:218], off offset:32
"""
    prologue_new = """\
\ts_clause 0x7
\tglobal_load_b128 v[205:208], v[211:212], off
\tglobal_load_b128 v[189:192], v[211:212], off offset:32
\tglobal_load_b128 v[201:204], v[213:214], off
\tglobal_load_b128 v[185:188], v[213:214], off offset:32
\tglobal_load_b128 v[197:200], v[215:216], off
\tglobal_load_b128 v[181:184], v[215:216], off offset:32
\tglobal_load_b128 v[193:196], v[217:218], off
\tglobal_load_b128 v[177:180], v[217:218], off offset:32
"""
    if prologue_old not in src:
        raise RuntimeError("could not find prologue X-load block")
    src = src.replace(prologue_old, prologue_new, 1)

    # ---------- 2) Remove dual_mov block + staggered waits at bb.2 top. ----------
    # Find anchor: the "s_wait_loadcnt 0x6\n" that begins the dual_mov sequence,
    # and the line "v_dual_mov_b32 v206, v130 :: v_dual_mov_b32 v205, v129\n"
    # that is the last copy before the loop test.
    bb2_old = """\
\ts_wait_loadcnt 0x6
\tv_dual_mov_b32 v192, v136 :: v_dual_mov_b32 v191, v135
\ts_wait_loadcnt 0x5
\tv_dual_mov_b32 v204, v140 :: v_dual_mov_b32 v203, v139
\ts_wait_loadcnt 0x4
\tv_dual_mov_b32 v188, v144 :: v_dual_mov_b32 v187, v143
\ts_wait_loadcnt 0x3
\tv_dual_mov_b32 v200, v148 :: v_dual_mov_b32 v199, v147
\ts_wait_loadcnt 0x2
\tv_dual_mov_b32 v184, v152 :: v_dual_mov_b32 v183, v151
\ts_wait_loadcnt 0x1
\tv_dual_mov_b32 v196, v156 :: v_dual_mov_b32 v195, v155
\ts_wait_loadcnt 0x0
\tv_dual_mov_b32 v180, v160 :: v_dual_mov_b32 v179, v159
\tv_dual_mov_b32 v208, v132 :: v_dual_mov_b32 v207, v131
\tv_dual_mov_b32 v190, v134 :: v_dual_mov_b32 v189, v133
\tv_dual_mov_b32 v202, v138 :: v_dual_mov_b32 v201, v137
\tv_dual_mov_b32 v186, v142 :: v_dual_mov_b32 v185, v141
\tv_dual_mov_b32 v198, v146 :: v_dual_mov_b32 v197, v145
\tv_dual_mov_b32 v182, v150 :: v_dual_mov_b32 v181, v149
\tv_dual_mov_b32 v194, v154 :: v_dual_mov_b32 v193, v153
\tv_dual_mov_b32 v178, v158 :: v_dual_mov_b32 v177, v157
\tv_dual_mov_b32 v206, v130 :: v_dual_mov_b32 v205, v129
"""
    bb2_new = """\
"""
    if bb2_old not in src:
        raise RuntimeError("could not find bb.2 dual_mov swap block")
    src = src.replace(bb2_old, bb2_new, 1)

    # ---------- 3) Drop bb.3 X address computes + 8 X loads. ----------
    # bb.3 currently: 1 W base compute (v[129:130] = v[209:210]+s0), 4 X frag
    # base computes (v[133:134], v[141:142], v[149:150], v[157:158]), 4 W loads,
    # 8 X loads. Drop the X portion.
    bb3_x_addr_old = """\
\tv_add_co_u32 v133, vcc_lo, v211, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v134, null, s1, v212, vcc_lo
\tv_add_co_u32 v141, vcc_lo, v213, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v142, null, s1, v214, vcc_lo
\tv_add_co_u32 v149, vcc_lo, v215, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v150, null, s1, v216, vcc_lo
\tv_add_co_u32 v157, vcc_lo, v217, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v158, null, s1, v218, vcc_lo
"""
    if bb3_x_addr_old not in src:
        raise RuntimeError("could not find bb.3 X frag base computes")
    src = src.replace(bb3_x_addr_old, "", 1)

    bb3_x_loads_old = """\
\ts_clause 0x7
\tglobal_load_b128 v[129:132], v[133:134], off offset:64
\tglobal_load_b128 v[133:136], v[133:134], off offset:96
\tglobal_load_b128 v[137:140], v[141:142], off offset:64
\tglobal_load_b128 v[141:144], v[141:142], off offset:96
\tglobal_load_b128 v[145:148], v[149:150], off offset:64
\tglobal_load_b128 v[149:152], v[149:150], off offset:96
\tglobal_load_b128 v[153:156], v[157:158], off offset:64
\tglobal_load_b128 v[157:160], v[157:158], off offset:96
"""
    if bb3_x_loads_old not in src:
        raise RuntimeError("could not find bb.3 X global_loads")
    src = src.replace(bb3_x_loads_old, "", 1)

    # ---------- 3.5) Move wait_loadcnt for X-completion to bb.4 head, before
    #                 the WMMAs but after B ds_load issuance. The B ds_loads
    #                 (lines 226-231) don't depend on v[177..208], so they can
    #                 issue while X loads still in flight. Add wait_loadcnt 0x4
    #                 (≤ 4 outstanding = 8 X done, 4 W still pending) right
    #                 before the first WMMA. ----------
    # ---------- 3.5) Place strict s_wait_loadcnt 0x0 at bb.4 head before WMMAs.
    #                 NOTE: Tested staggered FIFO waits (0x7/0x5/0x4) and bulk
    #                 0x4 — both produce wrong results. AMDGPU loadcnt is NOT
    #                 strict-FIFO across the bb.5-tail X loads vs bb.3 W loads
    #                 (which are separated by barriers + branches). LLVM's
    #                 staggered pattern works in original pf.s only because
    #                 all 12 loads are issued in same bb (bb.3). Here, must
    #                 drain ALL outstanding before reading any a-slot.
    bb4_wait_old = """\
\tds_load_b128 v[242:245], v246
\tds_load_b128 v[246:249], v246 offset:4096
\ts_wait_dscnt 0x5
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[197:200], v[226:229], v[57:64]
"""
    bb4_wait_new = """\
\tds_load_b128 v[242:245], v246
\tds_load_b128 v[246:249], v246 offset:4096
\ts_wait_loadcnt 0x0
\ts_wait_dscnt 0x5
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[197:200], v[226:229], v[57:64]
"""
    if bb4_wait_old not in src:
        raise RuntimeError("could not find bb.4 first-WMMA anchor for loadcnt insertion")
    src = src.replace(bb4_wait_old, bb4_wait_new, 1)

    # ---------- 3.6) Fix bb.5 wait_loadcnt values. Original 0xb/0xa/0x9/0x8
    #                 assumed 12 outstanding (4 W + 8 X) at start of bb.5
    #                 with W issued FIRST in bb.3 (so FIFO completion = W
    #                 done first). With my patch, bb.3 issues only 4 W; the
    #                 8 X are from PREVIOUS iter's bb.5 (issued chronologically
    #                 BEFORE this bb.3's W). FIFO drains X first.
    #
    #                 Now after bb.4 head's wait_loadcnt 0x4, we know all 8
    #                 X are done; 4 W still pending. Update bb.5 waits to
    #                 0x3/0x2/0x1/0x0.
    bb5_waits_old = """\
\ts_wait_loadcnt 0xb
\tds_store_b128 v177, v[173:176]
\ts_wait_loadcnt 0xa
\tds_store_b128 v177, v[169:172] offset:2048
\ts_wait_loadcnt 0x9
\tds_store_b128 v177, v[165:168] offset:4096
\ts_wait_loadcnt 0x8
\tds_store_b128 v177, v[161:164] offset:6144
"""
    bb5_waits_new = """\
\ts_wait_loadcnt 0x3
\tds_store_b128 v177, v[173:176]
\ts_wait_loadcnt 0x2
\tds_store_b128 v177, v[169:172] offset:2048
\ts_wait_loadcnt 0x1
\tds_store_b128 v177, v[165:168] offset:4096
\ts_wait_loadcnt 0x0
\tds_store_b128 v177, v[161:164] offset:6144
"""
    if bb5_waits_old not in src:
        raise RuntimeError("could not find bb.5 wait_loadcnt+ds_store sequence")
    src = src.replace(bb5_waits_old, bb5_waits_new, 1)

    # ---------- 4) Add X prefetch to bb.5 after barrier_wait. ----------
    # Place after `s_barrier_wait 0xffff\n;;#ASMEND\n`, before `s_branch .LBB0_1`.
    # s[0:1] has just been bumped (line 277 of original) to K+1's byte offset,
    # so v[211:212]+s[0:1] etc. give K+1 base address. offset:0 = K+1's K-half-0,
    # offset:32 = K+1's K-half-1.
    bb5_insert_anchor = """\
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
\ts_branch .LBB0_1
"""
    bb5_insert_new = """\
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
\tv_add_co_u32 v133, vcc_lo, v211, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v134, null, s1, v212, vcc_lo
\tv_add_co_u32 v141, vcc_lo, v213, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v142, null, s1, v214, vcc_lo
\tv_add_co_u32 v149, vcc_lo, v215, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v150, null, s1, v216, vcc_lo
\tv_add_co_u32 v157, vcc_lo, v217, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v158, null, s1, v218, vcc_lo
\ts_clause 0x7
\tglobal_load_b128 v[205:208], v[133:134], off
\tglobal_load_b128 v[189:192], v[133:134], off offset:32
\tglobal_load_b128 v[201:204], v[141:142], off
\tglobal_load_b128 v[185:188], v[141:142], off offset:32
\tglobal_load_b128 v[197:200], v[149:150], off
\tglobal_load_b128 v[181:184], v[149:150], off offset:32
\tglobal_load_b128 v[193:196], v[157:158], off
\tglobal_load_b128 v[177:180], v[157:158], off offset:32
\ts_branch .LBB0_1
"""
    if bb5_insert_anchor not in src:
        raise RuntimeError("could not find bb.5 barrier_wait+s_branch tail")
    src = src.replace(bb5_insert_anchor, bb5_insert_new, 1)

    return src


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", default="rdna4/vlm/generated/mm0_bf16_directa_pf.s")
    parser.add_argument("--out-file", default="rdna4/vlm/generated/mm0_bf16_directa_pf_noswap.s")
    args = parser.parse_args()
    src = Path(args.in_file).read_text(encoding="utf-8")
    out = patch(src)
    Path(args.out_file).write_text(out, encoding="utf-8")
    print(args.out_file)


if __name__ == "__main__":
    main()
