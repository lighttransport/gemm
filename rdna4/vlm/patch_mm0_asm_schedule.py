#!/usr/bin/env python3
"""
Patch generated RDNA4 mm0 BF16 AMDGPU assembly with schedule variants.

Input is the compiler-generated `.s` from gen_mm0_bf16_asm.py/hipcc.  Output is
still normal AMDGPU assembly with the same symbol and metadata.  The point is to
mutate a tightly bounded mainloop window while preserving ABI and register
allocation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


HALFSPLIT_BLOCK = """\
\tds_load_b128 v[174:177], v218
\tds_load_b128 v[178:181], v230
\tds_load_b128 v[182:185], v230 offset:256
\tds_load_b128 v[186:189], v230 offset:512
\tds_load_b128 v[190:193], v234
\tds_load_b128 v[194:197], v218 offset:256
\tds_load_b128 v[198:201], v218 offset:512
\tds_load_b128 v[202:205], v218 offset:768
\ts_wait_dscnt 0x6
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]
\ts_wait_dscnt 0x5
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]
\ts_wait_dscnt 0x4
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]
\ts_wait_dscnt 0x3
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]
\ts_wait_dscnt 0x2
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[194:197], v[178:181], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[194:197], v[182:185], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[194:197], v[186:189], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[194:197], v[190:193], v[65:72]
\ts_wait_dscnt 0x1
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[198:201], v[178:181], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[198:201], v[182:185], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[198:201], v[186:189], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[198:201], v[190:193], v[33:40]
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[202:205], v[178:181], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[202:205], v[182:185], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[202:205], v[186:189], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[202:205], v[190:193], v[1:8]
\tds_load_b128 v[206:209], v218 offset:4096
\tds_load_b128 v[222:225], v230 offset:4096
\tds_load_b128 v[226:229], v230 offset:4352
\tds_load_b128 v[230:233], v230 offset:4608
\tds_load_b128 v[234:237], v234 offset:4096
\tds_load_b128 v[210:213], v218 offset:4352
\tds_load_b128 v[214:217], v218 offset:4608
\tds_load_b128 v[218:221], v218 offset:4864
\ts_wait_dscnt 0x6
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
\ts_wait_dscnt 0x5
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
\ts_wait_dscnt 0x4
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
\ts_wait_dscnt 0x3
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
\ts_wait_dscnt 0x2
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[210:213], v[222:225], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[210:213], v[226:229], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[210:213], v[230:233], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[210:213], v[234:237], v[65:72]
\ts_wait_dscnt 0x1
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[214:217], v[222:225], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[214:217], v[226:229], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[214:217], v[230:233], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[214:217], v[234:237], v[33:40]
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[218:221], v[222:225], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[218:221], v[226:229], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[218:221], v[230:233], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
"""

STORE_EARLY20_BLOCK = """\
\tds_load_b128 v[174:177], v218
\tds_load_b128 v[178:181], v230
\tds_load_b128 v[182:185], v230 offset:256
\tds_load_b128 v[186:189], v230 offset:512
\tds_load_b128 v[190:193], v234
\tds_load_b128 v[194:197], v218 offset:256
\tds_load_b128 v[198:201], v218 offset:512
\tds_load_b128 v[202:205], v218 offset:768
\tds_load_b128 v[206:209], v218 offset:4096
\tds_load_b128 v[210:213], v218 offset:4352
\tds_load_b128 v[214:217], v218 offset:4608
\tds_load_b128 v[218:221], v218 offset:4864
\tds_load_b128 v[222:225], v230 offset:4096
\tds_load_b128 v[226:229], v230 offset:4352
\tds_load_b128 v[230:233], v230 offset:4608
\tds_load_b128 v[234:237], v234 offset:4096
\ts_wait_dscnt 0xe
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]
\ts_wait_dscnt 0xd
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]
\ts_wait_dscnt 0xc
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]
\ts_wait_dscnt 0xb
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]
\ts_wait_dscnt 0xa
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[194:197], v[178:181], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[194:197], v[182:185], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[194:197], v[186:189], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[194:197], v[190:193], v[65:72]
\ts_wait_dscnt 0x9
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[198:201], v[178:181], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[198:201], v[182:185], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[198:201], v[186:189], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[198:201], v[190:193], v[33:40]
\ts_wait_dscnt 0x8
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[202:205], v[178:181], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[202:205], v[182:185], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[202:205], v[186:189], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[202:205], v[190:193], v[1:8]
\ts_wait_dscnt 0x3
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
\ts_wait_dscnt 0x2
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
\ts_wait_dscnt 0x1
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
\ts_wait_alu 0xfffe
\ts_cbranch_vccnz .Lmm0_storeearly20_skip_stores
\ts_xor_b32 s6, s6, 1
\ts_add_nc_u64 s[0:1], s[0:1], 64
\ts_wait_alu 0xfffe
\ts_lshl_b32 s8, s6, 13
\ts_add_co_i32 s7, s7, 32
\ts_wait_alu 0xfffe
\tv_or_b32_e32 v174, s8, v166
\tv_add_nc_u32_e32 v175, s8, v169
\ts_mov_b32 s8, 0
\ts_wait_loadcnt 0x7
\tds_store_b128 v174, v[141:144]
\ts_wait_loadcnt 0x6
\tds_store_b128 v174, v[137:140] offset:2048
\ts_wait_loadcnt 0x5
\tds_store_b128 v174, v[133:136] offset:4096
\ts_wait_loadcnt 0x4
\tds_store_b128 v174, v[129:132] offset:6144
\ts_wait_loadcnt 0x3
\tds_store_b128 v175, v[157:160]
\ts_wait_loadcnt 0x2
\tds_store_b128 v175, v[153:156] offset:2048
\ts_wait_loadcnt 0x1
\tds_store_b128 v175, v[149:152] offset:4096
\ts_wait_loadcnt 0x0
\tds_store_b128 v175, v[145:148] offset:6144
.Lmm0_storeearly20_skip_stores:
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[210:213], v[222:225], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[210:213], v[226:229], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[210:213], v[230:233], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[210:213], v[234:237], v[65:72]
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[214:217], v[222:225], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[214:217], v[226:229], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[214:217], v[230:233], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[214:217], v[234:237], v[33:40]
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[218:221], v[222:225], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[218:221], v[226:229], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[218:221], v[230:233], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
\ts_wait_alu 0xfffe
\ts_cbranch_vccnz .LBB0_1
\t;;#ASMSTART
\ts_barrier_signal -1
\t;;#ASMEND
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
\ts_branch .LBB0_1
"""


def patch_halfsplit(src: str) -> str:
    lines = src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line == "\tds_load_b128 v[174:177], v218":
            start = i
            break
    if start is None:
        raise RuntimeError("could not find mainloop ds_load start")

    end = None
    for i in range(start, len(lines)):
        if lines[i] == "\ts_wait_alu 0xfffe" and i > start:
            end = i
            break
    if end is None:
        raise RuntimeError("could not find mainloop schedule end")

    replacement = HALFSPLIT_BLOCK.rstrip("\n").splitlines()
    return "\n".join(lines[:start] + replacement + lines[end:]) + "\n"


def patch_storeearly20(src: str) -> str:
    lines = src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line == "\tds_load_b128 v[174:177], v218":
            start = i
            break
    if start is None:
        raise RuntimeError("could not find mainloop ds_load start")

    end = None
    for i in range(start, len(lines)):
        if lines[i] == ".LBB0_6:":
            end = i
            break
    if end is None:
        raise RuntimeError("could not find post-loop block")

    replacement = STORE_EARLY20_BLOCK.rstrip("\n").splitlines()
    return "\n".join(lines[:start] + replacement + lines[end:]) + "\n"


def patch_bfirst2(src: str) -> str:
    old_loads = """\
\tds_load_b128 v[206:209], v218 offset:4096
\tds_load_b128 v[210:213], v218 offset:4352
\tds_load_b128 v[214:217], v218 offset:4608
\tds_load_b128 v[218:221], v218 offset:4864
\tds_load_b128 v[222:225], v230 offset:4096
\tds_load_b128 v[226:229], v230 offset:4352
\tds_load_b128 v[230:233], v230 offset:4608
\tds_load_b128 v[234:237], v234 offset:4096
"""
    new_loads = """\
\tds_load_b128 v[206:209], v218 offset:4096
\tds_load_b128 v[222:225], v230 offset:4096
\tds_load_b128 v[226:229], v230 offset:4352
\tds_load_b128 v[230:233], v230 offset:4608
\tds_load_b128 v[234:237], v234 offset:4096
\tds_load_b128 v[210:213], v218 offset:4352
\tds_load_b128 v[214:217], v218 offset:4608
\tds_load_b128 v[218:221], v218 offset:4864
"""
    old_wmma = """\
\ts_wait_dscnt 0x3
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
\ts_wait_dscnt 0x2
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
\ts_wait_dscnt 0x1
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[210:213], v[222:225], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[210:213], v[226:229], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[210:213], v[230:233], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[210:213], v[234:237], v[65:72]
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[214:217], v[222:225], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[214:217], v[226:229], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[214:217], v[230:233], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[214:217], v[234:237], v[33:40]
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[218:221], v[222:225], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[218:221], v[226:229], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[218:221], v[230:233], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
"""
    new_wmma = """\
\ts_wait_dscnt 0x6
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
\ts_wait_dscnt 0x5
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
\ts_wait_dscnt 0x4
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
\ts_wait_dscnt 0x3
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
\ts_wait_dscnt 0x2
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[210:213], v[222:225], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[210:213], v[226:229], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[210:213], v[230:233], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[210:213], v[234:237], v[65:72]
\ts_wait_dscnt 0x1
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[214:217], v[222:225], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[214:217], v[226:229], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[214:217], v[230:233], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[214:217], v[234:237], v[33:40]
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[218:221], v[222:225], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[218:221], v[226:229], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[218:221], v[230:233], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
"""
    if old_loads not in src:
        raise RuntimeError("could not find second-half LDS load order")
    out = src.replace(old_loads, new_loads, 1)
    if old_wmma not in out:
        raise RuntimeError("could not find second-half WMMA wait block")
    return out.replace(old_wmma, new_wmma, 1)


PGR1MID_BLOCK = """\
\tds_load_b128 v[174:177], v218
\tds_load_b128 v[178:181], v230
\tds_load_b128 v[182:185], v230 offset:256
\tds_load_b128 v[186:189], v230 offset:512
\tds_load_b128 v[190:193], v234
\tds_load_b128 v[194:197], v218 offset:256
\tds_load_b128 v[198:201], v218 offset:512
\tds_load_b128 v[202:205], v218 offset:768
\tds_load_b128 v[206:209], v218 offset:4608
\tds_load_b128 v[210:213], v218 offset:4864
\tds_load_b128 v[214:217], v218 offset:5120
\tds_load_b128 v[218:221], v218 offset:5376
\tds_load_b128 v[222:225], v230 offset:4608
\tds_load_b128 v[226:229], v230 offset:4864
\tds_load_b128 v[230:233], v230 offset:5120
\tds_load_b128 v[234:237], v234 offset:4608
\ts_wait_dscnt 0xe
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]
\ts_wait_dscnt 0xd
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]
\ts_wait_dscnt 0xc
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]
\ts_wait_dscnt 0xb
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]
\ts_wait_dscnt 0xa
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[194:197], v[178:181], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[194:197], v[182:185], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[194:197], v[186:189], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[194:197], v[190:193], v[65:72]
\ts_wait_dscnt 0x9
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[198:201], v[178:181], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[198:201], v[182:185], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[198:201], v[186:189], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[198:201], v[190:193], v[33:40]
\ts_wait_dscnt 0x8
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[202:205], v[178:181], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[202:205], v[182:185], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[202:205], v[186:189], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[202:205], v[190:193], v[1:8]
\ts_wait_dscnt 0x3
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
\ts_wait_dscnt 0x2
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
\ts_wait_dscnt 0x1
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
\ts_wait_alu 0xfffe
\ts_cbranch_vccnz .Lmm0_pgr1mid_last_iter
\ts_xor_b32 s6, s6, 1
\ts_add_nc_u64 s[0:1], s[0:1], 64
\ts_wait_alu 0xfffe
\ts_mul_i32 s8, s6, 0x2400
\ts_add_co_i32 s7, s7, 32
\ts_wait_alu 0xfffe
\tv_add_nc_u32_e32 v174, s8, v166
\tv_add_nc_u32_e32 v175, s8, v169
\ts_mov_b32 s8, 0
\ts_wait_loadcnt 0x7
\tds_store_b128 v174, v[129:132]
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[210:213], v[222:225], v[89:96]
\ts_wait_loadcnt 0x6
\tds_store_b128 v174, v[133:136] offset:2304
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[210:213], v[226:229], v[81:88]
\ts_wait_loadcnt 0x5
\tds_store_b128 v174, v[141:144] offset:4608
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[210:213], v[230:233], v[73:80]
\ts_wait_loadcnt 0x4
\tds_store_b128 v174, v[137:140] offset:6912
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[210:213], v[234:237], v[65:72]
\ts_wait_loadcnt 0x3
\tds_store_b128 v175, v[157:160]
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[214:217], v[222:225], v[57:64]
\ts_wait_loadcnt 0x2
\tds_store_b128 v175, v[153:156] offset:2304
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[214:217], v[226:229], v[49:56]
\ts_wait_loadcnt 0x1
\tds_store_b128 v175, v[149:152] offset:4608
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[214:217], v[230:233], v[41:48]
\ts_wait_loadcnt 0x0
\tds_store_b128 v175, v[145:148] offset:6912
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[214:217], v[234:237], v[33:40]
\ts_wait_dscnt 0x0
\ts_barrier_signal -1
\ts_barrier_wait 0xffff
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[218:221], v[222:225], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[218:221], v[226:229], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[218:221], v[230:233], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
\ts_branch .LBB0_1
.Lmm0_pgr1mid_last_iter:
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[210:213], v[222:225], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[210:213], v[226:229], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[210:213], v[230:233], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[210:213], v[234:237], v[65:72]
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[214:217], v[222:225], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[214:217], v[226:229], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[214:217], v[230:233], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[214:217], v[234:237], v[33:40]
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[218:221], v[222:225], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[218:221], v[226:229], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[218:221], v[230:233], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
\ts_branch .LBB0_1
"""


def patch_pgr1mid(src: str) -> str:
    lines = src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line == "\tds_load_b128 v[174:177], v218":
            start = i
            break
    if start is None:
        raise RuntimeError("could not find mainloop ds_load start")

    end = None
    for i in range(start, len(lines)):
        if lines[i] == ".LBB0_6:":
            end = i
            break
    if end is None:
        raise RuntimeError("could not find post-loop block")

    replacement = PGR1MID_BLOCK.rstrip("\n").splitlines()
    return "\n".join(lines[:start] + replacement + lines[end:]) + "\n"


def patch_glinline(src: str) -> str:
    """Move 8 global_load_b128 (X+W) from bb.3 to be 1:1 interleaved with the
    first 8 first-half WMMAs in .LBB0_4.  Keeps the v_add_co address setup in
    bb.3 (the loads need v137/v138 and v145/v146 as base) but lets the loads
    overlap with the first WMMA group in flight."""
    old_loads = """\
\ts_clause 0x3
\tglobal_load_b128 v[129:132], v[137:138], off offset:64
\tglobal_load_b128 v[133:136], v[137:138], off offset:80
\tglobal_load_b128 v[141:144], v[137:138], off offset:96
\tglobal_load_b128 v[137:140], v[137:138], off offset:112
\ts_clause 0x3
\tglobal_load_b128 v[157:160], v[145:146], off offset:64
\tglobal_load_b128 v[153:156], v[145:146], off offset:80
\tglobal_load_b128 v[149:152], v[145:146], off offset:96
\tglobal_load_b128 v[145:148], v[145:146], off offset:112
"""
    if old_loads not in src:
        raise RuntimeError("could not find bb.3 global_load block")
    src = src.replace(old_loads, "", 1)

    # Inject one global_load_b128 after each of the first 8 first-half WMMAs.
    # Keep the within-group order so v[137:140] is the LAST X-load (clobbers
    # v137/v138 base) and v[145:148] is the LAST W-load (clobbers v145/v146).
    targets = [
        ("\tv_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]\n",
         "\tv_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]\n"
         "\tglobal_load_b128 v[129:132], v[137:138], off offset:64\n"),
        ("\tv_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]\n",
         "\tv_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]\n"
         "\tglobal_load_b128 v[133:136], v[137:138], off offset:80\n"),
        ("\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]\n",
         "\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]\n"
         "\tglobal_load_b128 v[141:144], v[137:138], off offset:96\n"),
        ("\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]\n",
         "\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]\n"
         "\tglobal_load_b128 v[157:160], v[145:146], off offset:64\n"),
        ("\tv_wmma_f32_16x16x16_bf16 v[89:96], v[194:197], v[178:181], v[89:96]\n",
         "\tv_wmma_f32_16x16x16_bf16 v[89:96], v[194:197], v[178:181], v[89:96]\n"
         "\tglobal_load_b128 v[153:156], v[145:146], off offset:80\n"),
        ("\tv_wmma_f32_16x16x16_bf16 v[81:88], v[194:197], v[182:185], v[81:88]\n",
         "\tv_wmma_f32_16x16x16_bf16 v[81:88], v[194:197], v[182:185], v[81:88]\n"
         "\tglobal_load_b128 v[149:152], v[145:146], off offset:96\n"),
        ("\tv_wmma_f32_16x16x16_bf16 v[73:80], v[194:197], v[186:189], v[73:80]\n",
         "\tv_wmma_f32_16x16x16_bf16 v[73:80], v[194:197], v[186:189], v[73:80]\n"
         "\tglobal_load_b128 v[137:140], v[137:138], off offset:112\n"),
        ("\tv_wmma_f32_16x16x16_bf16 v[65:72], v[194:197], v[190:193], v[65:72]\n",
         "\tv_wmma_f32_16x16x16_bf16 v[65:72], v[194:197], v[190:193], v[65:72]\n"
         "\tglobal_load_b128 v[145:148], v[145:146], off offset:112\n"),
    ]
    for old, new in targets:
        if src.count(old) < 1:
            raise RuntimeError(f"could not find WMMA target: {old.strip()}")
        src = src.replace(old, new, 1)
    return src


def patch_topnowait(src: str) -> str:
    old = """\
; %bb.3:                                ;   in Loop: Header=BB0_2 Depth=1
\ts_wait_loadcnt 0x4
\tv_add_co_u32 v137, vcc_lo, v161, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v138, null, s1, v162, vcc_lo
\ts_wait_loadcnt 0x0
\tv_add_co_u32 v145, vcc_lo, v163, s0
"""
    new = """\
; %bb.3:                                ;   in Loop: Header=BB0_2 Depth=1
\tv_add_co_u32 v137, vcc_lo, v161, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v138, null, s1, v162, vcc_lo
\tv_add_co_u32 v145, vcc_lo, v163, s0
"""
    if old not in src:
        raise RuntimeError("could not find top-of-loop load wait block")
    return src.replace(old, new, 1)


DSMOVE_PROLOGUE_INSERT = """\
\ts_mov_b32 s9, 0
\ts_wait_alu 0xfffe
\tv_add_lshl_u32 v174, v172, s9, 4
\tv_add_nc_u32_e32 v218, v173, v174
\tv_add_nc_u32_e32 v230, v170, v174
\tv_add_nc_u32_e32 v234, v171, v174
\tds_load_b128 v[174:177], v218
\tds_load_b128 v[178:181], v230
\tds_load_b128 v[182:185], v230 offset:256
\tds_load_b128 v[186:189], v230 offset:512
\tds_load_b128 v[190:193], v234
\tds_load_b128 v[194:197], v218 offset:256
\tds_load_b128 v[198:201], v218 offset:512
\tds_load_b128 v[202:205], v218 offset:768
\tds_load_b128 v[206:209], v218 offset:4608
\tds_load_b128 v[210:213], v218 offset:4864
\tds_load_b128 v[214:217], v218 offset:5120
\tds_load_b128 v[218:221], v218 offset:5376
\tds_load_b128 v[222:225], v230 offset:4608
\tds_load_b128 v[226:229], v230 offset:4864
\tds_load_b128 v[230:233], v230 offset:5120
\tds_load_b128 v[234:237], v234 offset:4608
"""

DSMOVE_BB4_BLOCK = """\
.LBB0_4:                                ;   in Loop: Header=BB0_2 Depth=1
\ts_wait_alu 0xfffe
\ts_and_not1_b32 vcc_lo, exec_lo, s8
\ts_mov_b32 s8, -1
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[194:197], v[178:181], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[194:197], v[182:185], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[194:197], v[186:189], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[194:197], v[190:193], v[65:72]
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[198:201], v[178:181], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[198:201], v[182:185], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[198:201], v[186:189], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[198:201], v[190:193], v[33:40]
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[202:205], v[178:181], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[202:205], v[182:185], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[202:205], v[186:189], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[202:205], v[190:193], v[1:8]
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[210:213], v[222:225], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[210:213], v[226:229], v[81:88]
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[210:213], v[230:233], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[210:213], v[234:237], v[65:72]
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[214:217], v[222:225], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[214:217], v[226:229], v[49:56]
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[214:217], v[230:233], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[214:217], v[234:237], v[33:40]
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[218:221], v[222:225], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[218:221], v[226:229], v[17:24]
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[218:221], v[230:233], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
\ts_wait_alu 0xfffe
\ts_cbranch_vccnz .LBB0_1
"""

DSMOVE_BB5_TAIL_INSERT = """\
\ts_mul_i32 s9, s6, 0x240
\ts_wait_alu 0xfffe
\tv_add_lshl_u32 v174, v172, s9, 4
\tv_add_nc_u32_e32 v218, v173, v174
\tv_add_nc_u32_e32 v230, v170, v174
\tv_add_nc_u32_e32 v234, v171, v174
\tds_load_b128 v[174:177], v218
\tds_load_b128 v[178:181], v230
\tds_load_b128 v[182:185], v230 offset:256
\tds_load_b128 v[186:189], v230 offset:512
\tds_load_b128 v[190:193], v234
\tds_load_b128 v[194:197], v218 offset:256
\tds_load_b128 v[198:201], v218 offset:512
\tds_load_b128 v[202:205], v218 offset:768
\tds_load_b128 v[206:209], v218 offset:4608
\tds_load_b128 v[210:213], v218 offset:4864
\tds_load_b128 v[214:217], v218 offset:5120
\tds_load_b128 v[218:221], v218 offset:5376
\tds_load_b128 v[222:225], v230 offset:4608
\tds_load_b128 v[226:229], v230 offset:4864
\tds_load_b128 v[230:233], v230 offset:5120
\tds_load_b128 v[234:237], v234 offset:4608
"""


def patch_dsmove(src: str) -> str:
    """Hoist the 16 ds_load_b128 from top of bb.4 into the end of bb.5
    (after barrier_wait) and into the prologue (after the initial barrier,
    before s_branch .LBB0_2).  Replaces the per-WMMA s_wait_dscnt countdown
    with a single s_wait_dscnt 0x0 at top of bb.4.  LDS load latency now
    overlaps with the loop branch / scc resolve / bb.3 global_load issue."""

    # 1. Prologue: inject ds_loads after the prologue barrier_wait,
    #    before the implicit-def block / s_branch .LBB0_2.
    prologue_marker = "\ts_branch .LBB0_2\n"
    if src.count(prologue_marker) != 1:
        raise RuntimeError("expected exactly one s_branch .LBB0_2 in prologue")
    src = src.replace(prologue_marker, DSMOVE_PROLOGUE_INSERT + prologue_marker, 1)

    # 2. bb.4: replace the entire ds_load + 32-WMMA + cbranch block with the
    #    single-wait variant.  Match from `.LBB0_4:` to `s_cbranch_vccnz .LBB0_1`.
    lines = src.splitlines()
    bb4_start = None
    for i, line in enumerate(lines):
        if line == ".LBB0_4:                                ;   in Loop: Header=BB0_2 Depth=1":
            bb4_start = i
            break
    if bb4_start is None:
        raise RuntimeError("could not find .LBB0_4 label")
    bb4_end = None
    for i in range(bb4_start + 1, len(lines)):
        if lines[i] == "\ts_cbranch_vccnz .LBB0_1":
            bb4_end = i + 1
            break
    if bb4_end is None:
        raise RuntimeError("could not find bb.4 cbranch_vccnz")
    new_block = DSMOVE_BB4_BLOCK.rstrip("\n").splitlines()
    src = "\n".join(lines[:bb4_start] + new_block + lines[bb4_end:]) + "\n"

    # 3. bb.5: inject ds_loads after the barrier_wait ASMEND but before
    #    the s_branch .LBB0_1.  bb.5's barrier_wait is the second occurrence
    #    in the file (first is prologue's).
    bb5_marker = "\t;;#ASMEND\n\ts_branch .LBB0_1\n"
    if bb5_marker not in src:
        raise RuntimeError("could not find bb.5 barrier+branch tail")
    src = src.replace(
        bb5_marker,
        "\t;;#ASMEND\n" + DSMOVE_BB5_TAIL_INSERT + "\ts_branch .LBB0_1\n",
        1,
    )
    return src


def patch_barriersig_early(src: str) -> str:
    """Move s_barrier_signal from bb.5 (after stores) to end of bb.4
    (before cbranch_vccnz .LBB0_1).  Signal runs on every iteration —
    on the last iter, the unmatched signal is harmless (kernel ends
    after epilogue, so the bumped barrier counter doesn't matter).

    Theory: the signal-to-wait window now includes the entire bb.5
    store sequence (~10 cycles), giving cross-wave sync more slack.
    By the time bb.5's barrier_wait fires, all 4 waves have already
    signaled, so the wait completes in ~1 cycle instead of waiting
    for a slow wave to catch up."""

    # 1. End of bb.4: insert s_barrier_signal -1 right before s_cbranch_vccnz.
    bb4_old = """\
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
\ts_wait_alu 0xfffe
\ts_cbranch_vccnz .LBB0_1
"""
    bb4_new = """\
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
\t;;#ASMSTART
\ts_barrier_signal -1
\t;;#ASMEND
\ts_wait_alu 0xfffe
\ts_cbranch_vccnz .LBB0_1
"""
    if bb4_old not in src:
        raise RuntimeError("could not find bb.4 final WMMA + cbranch")
    src = src.replace(bb4_old, bb4_new, 1)

    # 2. Remove s_barrier_signal from bb.5 (keep barrier_wait).
    bb5_old = """\
\t;;#ASMSTART
\ts_barrier_signal -1
\t;;#ASMEND
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
\ts_branch .LBB0_1
"""
    bb5_new = """\
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
\ts_branch .LBB0_1
"""
    if bb5_old not in src:
        raise RuntimeError("could not find bb.5 barrier signal+wait pair")
    src = src.replace(bb5_old, bb5_new, 1)
    return src


def patch_bb5setup_hoist(src: str) -> str:
    """Hoist 9 bb.5 setup instructions (s_xor s6 toggle, s_add s[0:1] global
    address bump, s_mul/s_add/v_add LDS write address compute) from the head
    of bb.5 to the tail of bb.4 — placed right before s_cbranch_vccnz so they
    overlap with WMMA pipeline drain.

    On last iter (cbranch_vccnz falls through to bb.5/exit pre-summary):
    actually the cbranch_vccnz BRANCHES BACK to .LBB0_1 on continuing iter
    and falls through to bb.5 on last (vccz) — wait, summary says reverse:
    bb.4 falls through to bb.5 on continuing iter, cbranch jumps elsewhere
    on last. Either way: hoisted s_xor/s_add adjust state for the NEXT bb.5;
    on last iter that state is unused (post-loop epilogue does not read s6).
    """
    setup_block = """\
\ts_xor_b32 s6, s6, 1
\ts_add_nc_u64 s[0:1], s[0:1], 64
\ts_wait_alu 0xfffe
\ts_mul_i32 s9, s6, 0x2400
\ts_add_co_i32 s7, s7, 32
\ts_wait_alu 0xfffe
\tv_add_nc_u32_e32 v174, s9, v166
\tv_add_nc_u32_e32 v175, s9, v169
\ts_wait_alu 0xfffe
"""
    # Insert before s_cbranch_vccnz .LBB0_1 (works whether or not bse already
    # inserted a s_barrier_signal between the last WMMA and the cbranch).
    cb_marker = "\ts_cbranch_vccnz .LBB0_1\n"
    idx = src.find(cb_marker)
    if idx < 0:
        raise RuntimeError("could not find s_cbranch_vccnz .LBB0_1")
    # walk back to start of the s_wait_alu just before the cbranch, insert
    # before it.
    wait_marker = "\ts_wait_alu 0xfffe\n"
    wait_idx = src.rfind(wait_marker, 0, idx)
    if wait_idx < 0:
        raise RuntimeError("could not find s_wait_alu before cbranch")
    src = src[:wait_idx] + setup_block + src[wait_idx:]

    bb5_old = """\
; %bb.5:                                ;   in Loop: Header=BB0_2 Depth=1
\ts_xor_b32 s6, s6, 1
\ts_add_nc_u64 s[0:1], s[0:1], 64
\ts_wait_alu 0xfffe
\ts_mul_i32 s8, s6, 0x2400
\ts_add_co_i32 s7, s7, 32
\ts_wait_alu 0xfffe
\tv_add_nc_u32_e32 v174, s8, v166
\tv_add_nc_u32_e32 v175, s8, v169
\ts_mov_b32 s8, 0
\ts_wait_loadcnt 0x7
"""
    bb5_new = """\
; %bb.5:                                ;   in Loop: Header=BB0_2 Depth=1
\ts_mov_b32 s8, 0
\ts_wait_loadcnt 0x7
"""
    if bb5_old not in src:
        raise RuntimeError("could not find bb.5 setup block")
    src = src.replace(bb5_old, bb5_new, 1)
    return src


def patch_dscnt_collapse(src: str) -> str:
    """Replace per-WMMA s_wait_dscnt countdown in bb.4 with grouped waits.

    Original:
      s_wait_dscnt 0xe; wmma1
      s_wait_dscnt 0xd; wmma2
      s_wait_dscnt 0xc; wmma3
      s_wait_dscnt 0xb; wmma4
      s_wait_dscnt 0xa; wmma5..wmma8
      ...
    Collapsed (first-half):
      s_wait_dscnt 0xb; wmma1..wmma4
      s_wait_dscnt 0xa; wmma5..wmma8
      ...

    Theory: HW scoreboard handles WMMA-vs-pending-load operand dependence.
    The 4 individual wait_dscnt before wmma1..4 are redundant — only the
    LAST one (wait_dscnt 0xb, ensuring all 4 fragments loaded) is needed.
    Saves 3 wait_dscnt × ~1 cycle each = ~3 cycles per iter."""

    # First-half: collapse 4 individual waits before wmma 1..4 to one wait.
    fh_old = """\
\ts_wait_dscnt 0xe
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]
\ts_wait_dscnt 0xd
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]
\ts_wait_dscnt 0xc
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]
\ts_wait_dscnt 0xb
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]
"""
    fh_new = """\
\ts_wait_dscnt 0xb
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]
"""
    if fh_old not in src:
        raise RuntimeError("could not find first-half wait/wmma block")
    src = src.replace(fh_old, fh_new, 1)

    # Second-half: collapse 3 individual waits before wmma 17..19 to one wait.
    sh_old = """\
\ts_wait_dscnt 0x3
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
\ts_wait_dscnt 0x2
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
\ts_wait_dscnt 0x1
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
"""
    sh_new = """\
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
"""
    if sh_old not in src:
        raise RuntimeError("could not find second-half wait/wmma block")
    src = src.replace(sh_old, sh_new, 1)
    return src


def patch_storewait0(src: str) -> str:
    """Collapse per-store s_wait_loadcnt 0x7→0x0 countdown in bb.5 into a
    single s_wait_loadcnt 0x0 followed by 8 back-to-back ds_store_b128.
    Total stall is bounded by the longest load (same as countdown), but
    we save 7 wait-instruction overheads per iteration."""
    old = """\
\ts_wait_loadcnt 0x7
\tds_store_b128 v174, v[129:132]
\ts_wait_loadcnt 0x6
\tds_store_b128 v174, v[133:136] offset:2304
\ts_wait_loadcnt 0x5
\tds_store_b128 v174, v[141:144] offset:4608
\ts_wait_loadcnt 0x4
\tds_store_b128 v174, v[137:140] offset:6912
\ts_wait_loadcnt 0x3
\tds_store_b128 v175, v[157:160]
\ts_wait_loadcnt 0x2
\tds_store_b128 v175, v[153:156] offset:2304
\ts_wait_loadcnt 0x1
\tds_store_b128 v175, v[149:152] offset:4608
\ts_wait_loadcnt 0x0
\tds_store_b128 v175, v[145:148] offset:6912
"""
    new = """\
\ts_wait_loadcnt 0x0
\tds_store_b128 v174, v[129:132]
\tds_store_b128 v174, v[133:136] offset:2304
\tds_store_b128 v174, v[141:144] offset:4608
\tds_store_b128 v174, v[137:140] offset:6912
\tds_store_b128 v175, v[157:160]
\tds_store_b128 v175, v[153:156] offset:2304
\tds_store_b128 v175, v[149:152] offset:4608
\tds_store_b128 v175, v[145:148] offset:6912
"""
    if old not in src:
        raise RuntimeError("could not find end-of-loop store wait block")
    return src.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=[
            "halfsplit",
            "storeearly20",
            "bfirst2",
            "pgr1mid",
            "glinline",
            "topnowait",
            "storewait0",
            "topnowait-storewait0",
            "dsmove",
            "barriersig-early",
            "topnowait-barriersig-early",
            "topnowait-storewait0-barriersig-early",
            "bb5setup-hoist",
            "bb5setup-hoist-bse",
            "dscnt-collapse",
            "dscnt-collapse-bse",
        ],
        default="halfsplit",
    )
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    src = Path(args.input).read_text(encoding="utf-8")
    if args.variant == "halfsplit":
        out = patch_halfsplit(src)
    elif args.variant == "storeearly20":
        out = patch_storeearly20(src)
    elif args.variant == "bfirst2":
        out = patch_bfirst2(src)
    elif args.variant == "pgr1mid":
        out = patch_pgr1mid(src)
    elif args.variant == "glinline":
        out = patch_glinline(src)
    elif args.variant == "topnowait":
        out = patch_topnowait(src)
    elif args.variant == "storewait0":
        out = patch_storewait0(src)
    elif args.variant == "topnowait-storewait0":
        out = patch_storewait0(patch_topnowait(src))
    elif args.variant == "dsmove":
        out = patch_dsmove(src)
    elif args.variant == "barriersig-early":
        out = patch_barriersig_early(src)
    elif args.variant == "topnowait-barriersig-early":
        out = patch_barriersig_early(patch_topnowait(src))
    elif args.variant == "topnowait-storewait0-barriersig-early":
        out = patch_barriersig_early(patch_storewait0(patch_topnowait(src)))
    elif args.variant == "bb5setup-hoist":
        out = patch_bb5setup_hoist(src)
    elif args.variant == "bb5setup-hoist-bse":
        out = patch_bb5setup_hoist(patch_barriersig_early(src))
    elif args.variant == "dscnt-collapse":
        out = patch_dscnt_collapse(src)
    elif args.variant == "dscnt-collapse-bse":
        out = patch_barriersig_early(patch_dscnt_collapse(src))
    else:
        raise AssertionError(args.variant)
    Path(args.output).write_text(out, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
