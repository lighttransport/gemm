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


def patch_storewait0(src: str) -> str:
    old = """\
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
"""
    new = """\
\ts_wait_loadcnt 0x0
\tds_store_b128 v174, v[141:144]
\tds_store_b128 v174, v[137:140] offset:2048
\tds_store_b128 v174, v[133:136] offset:4096
\tds_store_b128 v174, v[129:132] offset:6144
\tds_store_b128 v175, v[157:160]
\tds_store_b128 v175, v[153:156] offset:2048
\tds_store_b128 v175, v[149:152] offset:4096
\tds_store_b128 v175, v[145:148] offset:6144
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
    else:
        raise AssertionError(args.variant)
    Path(args.output).write_text(out, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
