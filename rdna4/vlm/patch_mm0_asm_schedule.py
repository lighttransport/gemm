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


def patch_pgr2_lite(src: str) -> str:
    """PGR2-lite: pre-issue the first 2 ds_load_b128 (v[174:177] X0 fragment
    and v[178:181] W0 fragment) at the END of bb.5 (after barrier_wait, before
    s_branch) and at the END of the prologue. Remove them from bb.4 head so
    the only consumer of those VGPRs is the pre-loaded copy.

    Theory: in baseline, WMMA1 in bb.4 stalls ~10 cycles for v[174:181] to
    complete (s_wait_dscnt 0xe drops as the first 2 of 16 ds_loads finish).
    Pre-issuing those 2 loads in bb.5 tail gives them ~5+ cycles head start
    over bb.4 entry, so by the time WMMA1 issues, operands may already be
    ready. Net expected save: ~5 cycles/iter × 144 iters = ~720 cycles (~3%).

    Address computation in bb.5: s6 has been toggled (points to slot just
    written), so s8 = s6 * 0x2400 = bb.4-style buffer offset. Read base is
    v172*16 + s8 (same as bb.4 head's `v_add_lshl_u32 v174, v172, s9, 4`
    where s9 = s6 * 0x240). We reuse v218 as scratch for the read base
    since bb.5 has already finished using it as an LDS read address.
    """

    pre_block_addr = """\
\ts_mul_i32 s9, s6, 0x240
\ts_wait_alu 0xfffe
\tv_add_lshl_u32 v218, v172, s9, 4
\tv_add_nc_u32_e32 v230, v170, v218
\tv_add_nc_u32_e32 v218, v173, v218
"""
    pre_block_loads = """\
\tds_load_b128 v[174:177], v218
\tds_load_b128 v[178:181], v230
"""
    # In prologue, where we don't have a separate barrier_signal to insert
    # addr-setup before, we keep the full bundle together.
    pre_block = pre_block_addr + pre_block_loads

    # 1. Prologue: insert pre-loads before `s_branch .LBB0_2` at end of prologue.
    prologue_old = """\
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
                                        ; implicit-def: $vgpr129_vgpr130_vgpr131_vgpr132
                                        ; implicit-def: $vgpr133_vgpr134_vgpr135_vgpr136
                                        ; implicit-def: $vgpr145_vgpr146_vgpr147_vgpr148
                                        ; implicit-def: $vgpr149_vgpr150_vgpr151_vgpr152
                                        ; implicit-def: $vgpr153_vgpr154_vgpr155_vgpr156
                                        ; implicit-def: $vgpr157_vgpr158_vgpr159_vgpr160
                                        ; implicit-def: $vgpr137_vgpr138_vgpr139_vgpr140
                                        ; implicit-def: $vgpr141_vgpr142_vgpr143_vgpr144
\ts_branch .LBB0_2
"""
    prologue_new = """\
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
                                        ; implicit-def: $vgpr129_vgpr130_vgpr131_vgpr132
                                        ; implicit-def: $vgpr133_vgpr134_vgpr135_vgpr136
                                        ; implicit-def: $vgpr145_vgpr146_vgpr147_vgpr148
                                        ; implicit-def: $vgpr149_vgpr150_vgpr151_vgpr152
                                        ; implicit-def: $vgpr153_vgpr154_vgpr155_vgpr156
                                        ; implicit-def: $vgpr157_vgpr158_vgpr159_vgpr160
                                        ; implicit-def: $vgpr137_vgpr138_vgpr139_vgpr140
                                        ; implicit-def: $vgpr141_vgpr142_vgpr143_vgpr144
""" + pre_block + "\ts_branch .LBB0_2\n"

    if prologue_old not in src:
        raise RuntimeError("could not find prologue end (barrier_wait + implicit-defs + branch)")
    src = src.replace(prologue_old, prologue_new, 1)

    # 2. bb.5 tail: split the pre_block. addr-setup goes BEFORE barrier_signal
    #    (so it overlaps with cross-wave sync handshake), ds_loads go AFTER
    #    barrier_wait (minimum critical-path addition: 2 ds_load issues).
    #    Two cases: raw bb.5 (signal+wait) or bse-modified (wait only).
    bb5_raw_old = """\
\t;;#ASMSTART
\ts_barrier_signal -1
\t;;#ASMEND
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
\ts_branch .LBB0_1
"""
    bb5_raw_new = pre_block_addr + """\
\t;;#ASMSTART
\ts_barrier_signal -1
\t;;#ASMEND
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
""" + pre_block_loads + "\ts_branch .LBB0_1\n"

    bb5_bse_old = """\
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
\ts_branch .LBB0_1
"""
    bb5_bse_new = pre_block_addr + """\
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
""" + pre_block_loads + "\ts_branch .LBB0_1\n"

    if bb5_raw_old in src:
        src = src.replace(bb5_raw_old, bb5_raw_new, 1)
    elif bb5_bse_old in src:
        src = src.replace(bb5_bse_old, bb5_bse_new, 1)
    else:
        raise RuntimeError("could not find bb.5 tail (raw or bse-modified)")

    # 3a. Rewrite bb.4 head to use v218 as the address scratch instead of v174,
    # so the pre-loaded v[174:177] data is not clobbered. Original sequence:
    #   v_add_lshl_u32 v174, v172, s9, 4   ; v174 = scratch addr
    #   s_mov_b32 s8, -1
    #   s_delay_alu ...
    #   v_add_nc_u32_e32 v218, v173, v174  ; v218 = X-row read base
    #   v_add_nc_u32_e32 v230, v170, v174  ; v230 = W-row read base 0
    #   v_add_nc_u32_e32 v234, v171, v174  ; v234 = W-row read base 1
    # New: route through v218 directly so v174 stays untouched.
    bb4_addr_old = """\
\tv_add_lshl_u32 v174, v172, s9, 4
\ts_mov_b32 s8, -1
\ts_delay_alu instid0(VALU_DEP_1)
\tv_add_nc_u32_e32 v218, v173, v174
\tv_add_nc_u32_e32 v230, v170, v174
\tv_add_nc_u32_e32 v234, v171, v174
"""
    bb4_addr_new = """\
\tv_add_lshl_u32 v218, v172, s9, 4
\ts_mov_b32 s8, -1
\ts_delay_alu instid0(VALU_DEP_1)
\tv_add_nc_u32_e32 v230, v170, v218
\tv_add_nc_u32_e32 v234, v171, v218
\tv_add_nc_u32_e32 v218, v173, v218
"""
    if bb4_addr_old not in src:
        raise RuntimeError("could not find bb.4 address-setup block")
    src = src.replace(bb4_addr_old, bb4_addr_new, 1)

    # 3b. Remove the first 2 ds_loads (v[174:177] and v[178:181]) from bb.4 head.
    bb4_head_old = """\
\tds_load_b128 v[174:177], v218
\tds_load_b128 v[178:181], v230
\tds_load_b128 v[182:185], v230 offset:256
"""
    bb4_head_new = """\
\tds_load_b128 v[182:185], v230 offset:256
"""
    if bb4_head_old not in src:
        raise RuntimeError("could not find bb.4 first 3 ds_loads")
    src = src.replace(bb4_head_old, bb4_head_new, 1)

    # Wait counts unchanged: total outstanding ds_loads at bb.4 start is still
    # 16 (2 pre-loads from bb.5 tail + 14 from bb.4 head). FIFO completion
    # means `s_wait_dscnt 0xe` still drops when pre-loads (oldest) finish, so
    # WMMA1 still gates correctly on v[174:181]. If pre-loads completed before
    # bb.4 entry, the early dscnt waits become no-ops — exactly what we want.
    return src


def patch_storefuse(src: str) -> str:
    """Fuse bb.5 ds_stores+setup into bb.4 tail, leaving bb.5 with only
    barrier_wait + branch.

    Layout after patch:
      bb.4: ds_loads, 32 WMMAs, [setup: s_xor s6, s_mul s8, v_add v174/v175],
            s_wait_loadcnt 0x0, 8 ds_stores back-to-back, s_barrier_signal -1,
            s_cbranch_vccnz .LBB0_1
      bb.5: s_mov s8 0, s_add s[0:1] 64, s_barrier_wait 0xffff, s_branch .LBB0_1

    Theory: bb.5 setup (s_xor, s_mul, s_add, v_add ×2) is currently serial
    after the cbranch — moving it into bb.4 lets it overlap with WMMA pipeline
    drain. Saves ~5 cycles/iter × 144 iters = ~720 cycles. The combined block
    also makes the early-signal pattern safer (signal still after all stores
    in bb.4, with the ~3-cycle cbranch+branch gap before barrier_wait acting
    as the cross-wave slack window).

    Note: store address compute uses scratch sgpr s9 (not s8) to preserve
    s8=-1 for next iter's bb.4 vcc setup."""

    # 1. Append store sequence to end of bb.4 (after last WMMA, before
    #    s_cbranch_vccnz). Use s9 as temp scale reg.
    bb4_old = """\
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
\ts_wait_alu 0xfffe
\ts_cbranch_vccnz .LBB0_1
"""
    bb4_new = """\
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[218:221], v[234:237], v[1:8]
\ts_xor_b32 s6, s6, 1
\ts_wait_alu 0xfffe
\ts_mul_i32 s9, s6, 0x2400
\ts_wait_alu 0xfffe
\tv_add_nc_u32_e32 v174, s9, v166
\tv_add_nc_u32_e32 v175, s9, v169
\ts_wait_loadcnt 0x0
\tds_store_b128 v174, v[129:132]
\tds_store_b128 v174, v[133:136] offset:2304
\tds_store_b128 v174, v[141:144] offset:4608
\tds_store_b128 v174, v[137:140] offset:6912
\tds_store_b128 v175, v[157:160]
\tds_store_b128 v175, v[153:156] offset:2304
\tds_store_b128 v175, v[149:152] offset:4608
\tds_store_b128 v175, v[145:148] offset:6912
\t;;#ASMSTART
\ts_barrier_signal -1
\t;;#ASMEND
\ts_wait_alu 0xfffe
\ts_cbranch_vccnz .LBB0_1
"""
    if bb4_old not in src:
        raise RuntimeError("could not find bb.4 final WMMA + cbranch")
    src = src.replace(bb4_old, bb4_new, 1)

    # 2. Replace bb.5 with: s_mov s8 0, s_add s[0:1] 64, s_add_co_i32 s7 32,
    #    barrier_wait, s_branch. (s_xor s6, s_mul/v_add for store addr now
    #    in bb.4; the rest still needed.)
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
\t;;#ASMSTART
\ts_barrier_signal -1
\t;;#ASMEND
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
\ts_branch .LBB0_1
"""
    bb5_new = """\
; %bb.5:                                ;   in Loop: Header=BB0_2 Depth=1
\ts_add_nc_u64 s[0:1], s[0:1], 64
\ts_add_co_i32 s7, s7, 32
\ts_mov_b32 s8, 0
\t;;#ASMSTART
\ts_barrier_wait 0xffff
\t;;#ASMEND
\ts_branch .LBB0_1
"""
    if bb5_old not in src:
        raise RuntimeError("could not find bb.5 store sequence")
    src = src.replace(bb5_old, bb5_new, 1)
    return src


def patch_pgr2_bufload(src: str) -> str:
    """Replace 16 global_load_b128 (8 prologue + 8 in-loop bb.3) with
    buffer_load_b128 using bounded SRDs. This is the prerequisite step for
    PGR2 distributed scheduling — it owns the loads' OOB behavior so the
    scc-guard around bb.3 can be relaxed and loads can be relocated into
    the WMMA stream.

    SRD layout (RDNA4, verified from /tmp/hipblaslt_bf16_best_73823.s
    lines 654, 725):
        s[20:21] = X base (lo,hi)   -- copied from kernarg s[8:9]
        s[22]    = X NumRecords     -- M*K*2 = 9437184 (0x00900000)
        s[23]    = 0x30020000       -- RDNA4 SRD flag word
        s[24:25] = W base (lo,hi)   -- copied from kernarg s[6:7]
        s[26]    = W NumRecords     -- N*K*2 = 42467328 (0x02880000)
        s[27]    = 0x30020000

    Per-lane offset: v161 = v1*0x2400 (X), v163 = v2*0x2400 (W). The
    existing v[161:162] and v[163:164] held 64-bit absolute addresses
    via v_mad_co_i64_i32; the buffer_load path needs only the 32-bit
    per-lane row offset, since the SRD's 64-bit base provides the rest.

    The K-iter byte offset uses the existing s0 SGPR (lower 32 bits of
    s[0:1] which advances by 64 per iter). Max iter offset = 64*144 =
    9216 = 0x2400, well within 32-bit soffset range.

    Bumps `.amdhsa_next_free_sgpr` from 12 to 28 to claim s[20:27].
    """

    # 1. Replace prologue addressing + 8 global_loads (X+W, K=0 fragment).
    old_prologue = """\
\tv_mad_co_i64_i32 v[161:162], null, 0x2400, v1, s[8:9]
\tv_mad_co_i64_i32 v[163:164], null, 0x2400, v2, s[6:7]
\tv_dual_mov_b32 v125, v121 :: v_dual_mov_b32 v126, v121
\tv_dual_mov_b32 v127, v121 :: v_dual_mov_b32 v128, v121
\ts_clause 0x3
\tglobal_load_b128 v[129:132], v[161:162], off
\tglobal_load_b128 v[133:136], v[161:162], off offset:16
\tglobal_load_b128 v[137:140], v[161:162], off offset:32
\tglobal_load_b128 v[141:144], v[161:162], off offset:48
\ts_clause 0x3
\tglobal_load_b128 v[145:148], v[163:164], off
\tglobal_load_b128 v[149:152], v[163:164], off offset:16
\tglobal_load_b128 v[153:156], v[163:164], off offset:32
\tglobal_load_b128 v[157:160], v[163:164], off offset:48
"""
    new_prologue = """\
\tv_mul_lo_u32 v161, 0x2400, v1
\tv_mul_lo_u32 v163, 0x2400, v2
\ts_mov_b32 s20, s8
\ts_mov_b32 s21, s9
\ts_mov_b32 s22, 0x00900000
\ts_mov_b32 s23, 0x30020000
\ts_mov_b32 s24, s6
\ts_mov_b32 s25, s7
\ts_mov_b32 s26, 0x02880000
\ts_mov_b32 s27, 0x30020000
\tv_dual_mov_b32 v125, v121 :: v_dual_mov_b32 v126, v121
\tv_dual_mov_b32 v127, v121 :: v_dual_mov_b32 v128, v121
\ts_clause 0x3
\tbuffer_load_b128 v[129:132], v161, s[20:23], null offen
\tbuffer_load_b128 v[133:136], v161, s[20:23], null offen offset:16
\tbuffer_load_b128 v[137:140], v161, s[20:23], null offen offset:32
\tbuffer_load_b128 v[141:144], v161, s[20:23], null offen offset:48
\ts_clause 0x3
\tbuffer_load_b128 v[145:148], v163, s[24:27], null offen
\tbuffer_load_b128 v[149:152], v163, s[24:27], null offen offset:16
\tbuffer_load_b128 v[153:156], v163, s[24:27], null offen offset:32
\tbuffer_load_b128 v[157:160], v163, s[24:27], null offen offset:48
"""
    if old_prologue not in src:
        raise RuntimeError(
            "patch_pgr2_bufload: could not find prologue addressing block"
        )
    src = src.replace(old_prologue, new_prologue, 1)

    # 2. Replace bb.3 in-loop loads. The original block does 64-bit address
    #    arithmetic to advance by s[0:1]; under buffer_load we just pass s0
    #    as the soffset. v[137:138] / v[145:146] are no longer needed as
    #    address registers, but v137-140 / v145-148 ARE still destination
    #    registers, so we keep their reuse pattern intact.
    old_bb3 = """\
\ts_wait_loadcnt 0x4
\tv_add_co_u32 v137, vcc_lo, v161, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v138, null, s1, v162, vcc_lo
\ts_wait_loadcnt 0x0
\tv_add_co_u32 v145, vcc_lo, v163, s0
\ts_wait_alu 0xfffd
\tv_add_co_ci_u32_e64 v146, null, s1, v164, vcc_lo
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
    new_bb3 = """\
\ts_wait_loadcnt 0x0
\ts_clause 0x3
\tbuffer_load_b128 v[129:132], v161, s[20:23], s0 offen offset:64
\tbuffer_load_b128 v[133:136], v161, s[20:23], s0 offen offset:80
\tbuffer_load_b128 v[141:144], v161, s[20:23], s0 offen offset:96
\tbuffer_load_b128 v[137:140], v161, s[20:23], s0 offen offset:112
\ts_clause 0x3
\tbuffer_load_b128 v[157:160], v163, s[24:27], s0 offen offset:64
\tbuffer_load_b128 v[153:156], v163, s[24:27], s0 offen offset:80
\tbuffer_load_b128 v[149:152], v163, s[24:27], s0 offen offset:96
\tbuffer_load_b128 v[145:148], v163, s[24:27], s0 offen offset:112
"""
    if old_bb3 not in src:
        raise RuntimeError(
            "patch_pgr2_bufload: could not find bb.3 in-loop load block"
        )
    src = src.replace(old_bb3, new_bb3, 1)

    # 3. Bump amdhsa_next_free_sgpr from 12 to 28 to claim s[20:27] for SRDs.
    old_sgpr = "\t\t.amdhsa_next_free_sgpr 12\n"
    new_sgpr = "\t\t.amdhsa_next_free_sgpr 28\n"
    if old_sgpr not in src:
        raise RuntimeError(
            "patch_pgr2_bufload: could not find .amdhsa_next_free_sgpr 12"
        )
    src = src.replace(old_sgpr, new_sgpr, 1)

    return src


def patch_pgr2_drop_guard(src: str) -> str:
    """Drop the s_cbranch_scc1 .LBB0_4 last-iter guard around bb.3.

    Safe ONLY after pgr2-bufload because the buffer_load_b128 SRD has
    OOB-mask enabled (Srd127_96 = 0x30020000). On the last-iter pass, the
    bb.3 loads issue with s0 = K-tile-count * 64 = 9216, addressing K=144
    which is the matrix boundary. Within X (9.4 MB) and W (42 MB) buffer
    bounds the reads are not strictly OOB but they bleed into the next
    row's K=0; the resulting wrong data lands in registers, gets stored
    to LDS in bb.5, but no subsequent iter consumes it (loop exits via
    .LBB0_6 instead). Net effect: a few wasted load issue cycles on the
    final iter, removed branch overhead on every iter.

    With this guard removed, bb.3's loads can later be physically merged
    into bb.4 to enable per-WMMA distributed scheduling.
    """
    old = """\
\ts_cmp_gt_u32 s7, 0x11df
\ts_cbranch_scc1 .LBB0_4
"""
    new = """\
\ts_cmp_gt_u32 s7, 0x11df
"""
    if old not in src:
        raise RuntimeError(
            "patch_pgr2_drop_guard: could not find scc-guard before bb.3"
        )
    return src.replace(old, new, 1)


def patch_pgr2_distribute(src: str) -> str:
    """Move the 8 in-loop buffer_load_b128 from bb.3 INTO the bb.4 WMMA
    stream at slots [0,2,4,6,8,10,12,14] — one load every 2 WMMAs across
    the first 16 WMMAs. This is the actual PGR2 perf lever; pgr2-bufload
    alone is perf-neutral infrastructure.

    Issue order is preserved (X before W, same intra-group order) so the
    bb.5 s_wait_loadcnt 0x7→0x0 countdown still matches each load's
    destination register.

    Requires patch_pgr2_bufload to have run first (bb.3 must contain
    buffer_load_b128 not global_load_b128). Composes cleanly with
    patch_pgr2_drop_guard and patch_barriersig_early.
    """

    # 1. Strip the 8 buffer_loads + 2 s_clauses from bb.3, keeping the
    #    s_wait_loadcnt 0x0 (it now serves only as a fence between the
    #    previous iter's WMMA-stream loads completing and this iter's
    #    issue — harmless; the bb.5 countdown of the prior iter already
    #    drained loadcnt to 0 before the back-edge).
    old_bb3 = """\
\ts_wait_loadcnt 0x0
\ts_clause 0x3
\tbuffer_load_b128 v[129:132], v161, s[20:23], s0 offen offset:64
\tbuffer_load_b128 v[133:136], v161, s[20:23], s0 offen offset:80
\tbuffer_load_b128 v[141:144], v161, s[20:23], s0 offen offset:96
\tbuffer_load_b128 v[137:140], v161, s[20:23], s0 offen offset:112
\ts_clause 0x3
\tbuffer_load_b128 v[157:160], v163, s[24:27], s0 offen offset:64
\tbuffer_load_b128 v[153:156], v163, s[24:27], s0 offen offset:80
\tbuffer_load_b128 v[149:152], v163, s[24:27], s0 offen offset:96
\tbuffer_load_b128 v[145:148], v163, s[24:27], s0 offen offset:112
"""
    new_bb3 = "\ts_wait_loadcnt 0x0\n"
    if old_bb3 not in src:
        raise RuntimeError(
            "patch_pgr2_distribute: could not find bb.3 buffer_load block "
            "(must run after patch_pgr2_bufload)"
        )
    src = src.replace(old_bb3, new_bb3, 1)

    # 2. Inject 8 buffer_load_b128 lines into bb.4 WMMA stream at slots
    #    [0,2,4,6,8,10,12,14]. We match the entire 16-WMMA first-half
    #    block as a single contiguous string so insertion points are
    #    unambiguous and the order of WMMAs vs. loads is fully pinned.
    old_wmma_first_half = """\
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
"""
    new_wmma_first_half = """\
\ts_wait_dscnt 0xe
\tbuffer_load_b128 v[129:132], v161, s[20:23], s0 offen offset:64
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[174:177], v[178:181], v[121:128]
\ts_wait_dscnt 0xd
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[174:177], v[182:185], v[113:120]
\ts_wait_dscnt 0xc
\tbuffer_load_b128 v[133:136], v161, s[20:23], s0 offen offset:80
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]
\ts_wait_dscnt 0xb
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]
\ts_wait_dscnt 0xa
\tbuffer_load_b128 v[141:144], v161, s[20:23], s0 offen offset:96
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[194:197], v[178:181], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[194:197], v[182:185], v[81:88]
\tbuffer_load_b128 v[137:140], v161, s[20:23], s0 offen offset:112
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[194:197], v[186:189], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[194:197], v[190:193], v[65:72]
\ts_wait_dscnt 0x9
\tbuffer_load_b128 v[157:160], v163, s[24:27], s0 offen offset:64
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[198:201], v[178:181], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[198:201], v[182:185], v[49:56]
\tbuffer_load_b128 v[153:156], v163, s[24:27], s0 offen offset:80
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[198:201], v[186:189], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[198:201], v[190:193], v[33:40]
\ts_wait_dscnt 0x8
\tbuffer_load_b128 v[149:152], v163, s[24:27], s0 offen offset:96
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[202:205], v[178:181], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[202:205], v[182:185], v[17:24]
\tbuffer_load_b128 v[145:148], v163, s[24:27], s0 offen offset:112
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[202:205], v[186:189], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[202:205], v[190:193], v[1:8]
"""
    if old_wmma_first_half not in src:
        raise RuntimeError(
            "patch_pgr2_distribute: could not find bb.4 first-half WMMA block"
        )
    src = src.replace(old_wmma_first_half, new_wmma_first_half, 1)

    return src


def patch_dsload_interleave(src: str) -> str:
    """Interleave second-half ds_loads with first-half WMMAs.

    Baseline bb.4: 16 ds_loads issued back-to-back, then 32 WMMAs with
    s_wait_dscnt countdown. The second-half loads (regs 206-237, used by
    WMMAs 17-32) sit idle for ~32 cycles before they're consumed. The
    first-half loads (regs 174-205) are needed by WMMAs 1-16 and gate
    the start of computation.

    This patch issues only the FIRST 8 ds_loads upfront (just enough to
    feed WMMAs 1-16), then fires the second-half 8 ds_loads INTO the
    WMMA stream after WMMAs 2, 4, 6, 8, 10, 12, 14, 16 (one per even
    WMMA in the first half). By the time WMMA 17 needs them, they're
    long since complete — eliminating the s_wait_dscnt 0x3/0x2/0x1/0x0
    drains at the first→second-half boundary.

    The dscnt counter dance:
      - Start of bb.4: 8 loads in flight (was 16). dscnt = 8.
      - After 8 WMMAs + 4 mid-stream loads: dscnt floats around 4-8.
      - After 16 WMMAs + 8 mid-stream loads: 16 loads issued total,
        first 8 long done (~30+ cycles ago), last 8 done by issue time
        + ~30 cycles (which is 8 WMMA pairs ago).
      - WMMA 17 with NO s_wait_dscnt — all 16 done.

    Wait values are recomputed: the original 0xe→0x8 sequence (waiting
    for first-half completions among 16 in-flight) becomes 0x6→0x0
    (waiting for first-half completions among only the first-half 8 in
    flight, before second-half issue starts). The boundary 0x3→0x0
    drain is removed entirely.
    """
    # Match the entire bb.4 block from the 16 ds_loads through WMMA 20
    # so the rewrite is unambiguous.
    old = """\
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
"""
    # New schedule: 8 first-half ds_loads upfront. WMMAs 1-16 each
    # interleave one second-half ds_load on the EVEN slots 2,4,6,8,10,12,14,16
    # (one ds_load per pair of WMMAs). Wait counters now refer to a window
    # where loads are continuously being issued, so we use saturating waits
    # (drain-to-N where N = remaining) at the first WMMAs and a final 0x0
    # drain before WMMA 17. We rely on the LDS pipeline keeping outstanding
    # ops bounded.
    new = """\
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
\tds_load_b128 v[206:209], v218 offset:4608
\ts_wait_dscnt 0x5
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[174:177], v[186:189], v[105:112]
\ts_wait_dscnt 0x4
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[174:177], v[190:193], v[97:104]
\tds_load_b128 v[210:213], v218 offset:4864
\ts_wait_dscnt 0x4
\tv_wmma_f32_16x16x16_bf16 v[89:96], v[194:197], v[178:181], v[89:96]
\tv_wmma_f32_16x16x16_bf16 v[81:88], v[194:197], v[182:185], v[81:88]
\tds_load_b128 v[214:217], v218 offset:5120
\tv_wmma_f32_16x16x16_bf16 v[73:80], v[194:197], v[186:189], v[73:80]
\tv_wmma_f32_16x16x16_bf16 v[65:72], v[194:197], v[190:193], v[65:72]
\tds_load_b128 v[218:221], v218 offset:5376
\ts_wait_dscnt 0x4
\tv_wmma_f32_16x16x16_bf16 v[57:64], v[198:201], v[178:181], v[57:64]
\tv_wmma_f32_16x16x16_bf16 v[49:56], v[198:201], v[182:185], v[49:56]
\tds_load_b128 v[222:225], v230 offset:4608
\tv_wmma_f32_16x16x16_bf16 v[41:48], v[198:201], v[186:189], v[41:48]
\tv_wmma_f32_16x16x16_bf16 v[33:40], v[198:201], v[190:193], v[33:40]
\tds_load_b128 v[226:229], v230 offset:4864
\ts_wait_dscnt 0x4
\tv_wmma_f32_16x16x16_bf16 v[25:32], v[202:205], v[178:181], v[25:32]
\tv_wmma_f32_16x16x16_bf16 v[17:24], v[202:205], v[182:185], v[17:24]
\tds_load_b128 v[230:233], v230 offset:5120
\tv_wmma_f32_16x16x16_bf16 v[9:16], v[202:205], v[186:189], v[9:16]
\tv_wmma_f32_16x16x16_bf16 v[1:8], v[202:205], v[190:193], v[1:8]
\tds_load_b128 v[234:237], v234 offset:4608
\ts_wait_dscnt 0x3
\tv_wmma_f32_16x16x16_bf16 v[121:128], v[206:209], v[222:225], v[121:128]
\ts_wait_dscnt 0x2
\tv_wmma_f32_16x16x16_bf16 v[113:120], v[206:209], v[226:229], v[113:120]
\ts_wait_dscnt 0x1
\tv_wmma_f32_16x16x16_bf16 v[105:112], v[206:209], v[230:233], v[105:112]
\ts_wait_dscnt 0x0
\tv_wmma_f32_16x16x16_bf16 v[97:104], v[206:209], v[234:237], v[97:104]
"""
    if old not in src:
        raise RuntimeError(
            "patch_dsload_interleave: could not find bb.4 ds_load+WMMA block"
        )
    return src.replace(old, new, 1)


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
            "storefuse",
            "pgr2-lite",
            "pgr2-lite-bse",
            "pgr2-bufload",
            "pgr2-bufload-bse",
            "pgr2-bufload-noguard",
            "pgr2-bufload-noguard-bse",
            "pgr2-distribute",
            "pgr2-distribute-bse",
            "dsload-interleave",
            "dsload-interleave-bse",
            "dsload-interleave-bufload-bse",
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
    elif args.variant == "storefuse":
        out = patch_storefuse(src)
    elif args.variant == "pgr2-lite":
        out = patch_pgr2_lite(src)
    elif args.variant == "pgr2-lite-bse":
        out = patch_pgr2_lite(patch_barriersig_early(src))
    elif args.variant == "pgr2-bufload":
        out = patch_pgr2_bufload(src)
    elif args.variant == "pgr2-bufload-bse":
        out = patch_pgr2_bufload(patch_barriersig_early(src))
    elif args.variant == "pgr2-bufload-noguard":
        out = patch_pgr2_drop_guard(patch_pgr2_bufload(src))
    elif args.variant == "pgr2-bufload-noguard-bse":
        out = patch_pgr2_drop_guard(patch_pgr2_bufload(patch_barriersig_early(src)))
    elif args.variant == "pgr2-distribute":
        out = patch_pgr2_distribute(patch_pgr2_drop_guard(patch_pgr2_bufload(src)))
    elif args.variant == "pgr2-distribute-bse":
        out = patch_pgr2_distribute(
            patch_pgr2_drop_guard(patch_pgr2_bufload(patch_barriersig_early(src)))
        )
    elif args.variant == "dsload-interleave":
        out = patch_dsload_interleave(src)
    elif args.variant == "dsload-interleave-bse":
        out = patch_dsload_interleave(patch_barriersig_early(src))
    elif args.variant == "dsload-interleave-bufload-bse":
        out = patch_dsload_interleave(
            patch_pgr2_drop_guard(patch_pgr2_bufload(patch_barriersig_early(src)))
        )
    else:
        raise AssertionError(args.variant)
    Path(args.output).write_text(out, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
