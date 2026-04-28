/*
 * RDNA4 GGML dequantization benchmark.
 *
 * Standalone dequant kernels verify exact GGML block decoding against the CPU
 * implementation in common/ggml_dequant.h. Fused GEMM kernels stage dequanted
 * tiles into BF16/FP8/INT8 fragments and exercise gfx12 WMMA instructions.
 */

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../rocew.h"

#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

typedef struct {
    const char *name;
    uint32_t type;
    int block;
    int bytes;
} quant_type_info;

static const quant_type_info g_types[] = {
    {"q2_k",  GGML_TYPE_Q2_K,  256,  84},
    {"q3_k",  GGML_TYPE_Q3_K,  256, 110},
    {"q4_k",  GGML_TYPE_Q4_K,  256, 144},
    {"q5_k",  GGML_TYPE_Q5_K,  256, 176},
    {"q6_k",  GGML_TYPE_Q6_K,  256, 210},
    {"iq2_xxs", GGML_TYPE_IQ2_XXS, 256,  66},
    {"iq2_xs",  GGML_TYPE_IQ2_XS,  256,  74},
    {"iq2_s",   GGML_TYPE_IQ2_S,   256,  82},
    {"iq3_xxs", GGML_TYPE_IQ3_XXS, 256,  98},
    {"iq3_s",   GGML_TYPE_IQ3_S,   256, 110},
    {"iq1_s",   GGML_TYPE_IQ1_S,   256,  50},
    {"iq1_m",   GGML_TYPE_IQ1_M,   256,  56},
    {"iq4_xs",  GGML_TYPE_IQ4_XS,  256, 136},
    {"tq1_0",   GGML_TYPE_TQ1_0,   256,  54},
    {"tq2_0",   GGML_TYPE_TQ2_0,   256,  66},
    {"q4_0",  GGML_TYPE_Q4_0,   32,  18},
    {"q4_1",  GGML_TYPE_Q4_1,   32,  20},
    {"q5_0",  GGML_TYPE_Q5_0,   32,  22},
    {"q5_1",  GGML_TYPE_Q5_1,   32,  24},
    {"q8_0",  GGML_TYPE_Q8_0,   32,  34},
    {"iq4_nl", GGML_TYPE_IQ4_NL,  32,  18},
    {"q1_0",  GGML_TYPE_Q1_0,  128,  18},
    {"mxfp4", GGML_TYPE_MXFP4,  32,  17},
    {"nvfp4", GGML_TYPE_NVFP4,  64,  36},
};

enum {
    MODE_DEQUANT = 1u << 0,
    MODE_BF16    = 1u << 1,
    MODE_INT8    = 1u << 2,
    MODE_FP8     = 1u << 3,
    MODE_QUANT   = 1u << 4,
    MODE_Q2_FP8  = 1u << 5,
    MODE_Q2_LLAMA = 1u << 6,
    MODE_ALL     = MODE_DEQUANT | MODE_BF16 | MODE_INT8 | MODE_FP8 | MODE_QUANT | MODE_Q2_FP8 | MODE_Q2_LLAMA,
};

typedef struct {
    int rows;
    int cols;
    int tokens[16];
    int ntokens;
    int iters;
    int verify;
    int bench;
    int verbose;
    int device;
    uint32_t modes;
    const quant_type_info *only_type;
} options;

static const char *kernel_src =
"typedef unsigned char uchar;\n"
"typedef unsigned short ushort;\n"
"typedef unsigned int uint;\n"
"typedef unsigned long long uint64_t;\n"
"typedef unsigned int uint32_t;\n"
"typedef unsigned short uint16_t;\n"
"typedef unsigned char uint8_t;\n"
"typedef signed char int8_t;\n"
"typedef short bf16x8 __attribute__((ext_vector_type(8)));\n"
"typedef int int32x8 __attribute__((ext_vector_type(8)));\n"
"typedef uint uint32x2 __attribute__((ext_vector_type(2)));\n"
"typedef int int32x2 __attribute__((ext_vector_type(2)));\n"
"typedef float float8 __attribute__((ext_vector_type(8)));\n"
"\n"
"__device__ __forceinline__ float h2f(ushort h) {\n"
"    __half hv; *((ushort*)&hv) = h; return __half2float(hv);\n"
"}\n"
"__device__ __forceinline__ ushort f32_to_f16_bits(float v) {\n"
"    __half hv = __float2half(v); return *((ushort*)&hv);\n"
"}\n"
"__device__ __forceinline__ uint u32le(const uchar *p) {\n"
"    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);\n"
"}\n"
"__device__ __forceinline__ ushort u16le(const uchar *p) {\n"
"    return (ushort)((uint)p[0] | ((uint)p[1] << 8));\n"
"}\n"
"__device__ __forceinline__ float e8m0_to_f32_half(uchar x) {\n"
"    uint bits = x < 2 ? (0x00200000u << x) : ((uint)(x - 1) << 23); float v; memcpy(&v, &bits, 4); return v;\n"
"}\n"
"__device__ __forceinline__ float ue4m3_to_f32(uchar x) {\n"
"    if (x == 0 || x == 0x7f) return 0.0f; int exp = (x >> 3) & 15, man = x & 7;\n"
"    float raw = exp == 0 ? ldexpf((float)man, -9) : ldexpf(1.0f + (float)man / 8.0f, exp - 7); return raw * 0.5f;\n"
"}\n"
"__device__ __forceinline__ void get_scale_min_k4(int j, const uchar *q, uchar *d, uchar *m) {\n"
"    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }\n"
"    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }\n"
"}\n"
"__device__ __forceinline__ signed char q3_scale(const uchar *scales, int is) {\n"
"    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;\n"
"    uint a0 = u32le(scales + 0), a1 = u32le(scales + 4), a2 = u32le(scales + 8), a3 = 0;\n"
"    uint tmp = a2;\n"
"    a2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);\n"
"    a3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);\n"
"    a0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);\n"
"    a1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);\n"
"    uint v = is < 4 ? a0 : (is < 8 ? a1 : (is < 12 ? a2 : a3));\n"
"    return (signed char)((v >> (8 * (is & 3))) & 0xff);\n"
"}\n"
"__device__ __forceinline__ float iq_grid_u8(const uint64_t *tab, int idx, int j) {\n"
"    return (float)((tab[idx] >> (8 * j)) & 255ull);\n"
"}\n"
"__device__ __forceinline__ float iq_grid_i8(const uint64_t *tab, int idx, int j) {\n"
"    return (float)((signed char)((tab[idx] >> (8 * j)) & 255ull));\n"
"}\n"
"__device__ __forceinline__ int iq_grid_u4(const uint32_t *tab, int idx, int j) {\n"
"    return (int)((tab[idx] >> (8 * j)) & 255u);\n"
"}\n"
"__device__ __forceinline__ ushort iq1m_scale_u16(const uchar *scales) {\n"
"    ushort sc0 = u16le(scales + 0), sc1 = u16le(scales + 2), sc2 = u16le(scales + 4), sc3 = u16le(scales + 6);\n"
"    return (ushort)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));\n"
"}\n"
"__device__ __forceinline__ float dequant_q2_k_elem(const uchar *base, int row_bytes, int row, int k) {\n"
"    const uchar *r = base + (long)row * row_bytes; int b = k >> 8, kk = k & 255; const uchar *p = r + b * 84;\n"
"    float d = h2f(u16le(p + 80)); float mn = h2f(u16le(p + 82)); int half = kk >> 7; int rr = kk & 127;\n"
"    int group = rr >> 4; int shift = (group >> 1) * 2; int l = (rr & 15) + ((group & 1) ? 16 : 0);\n"
"    uchar sc = p[half * 8 + group]; int q = (p[16 + half * 32 + l] >> shift) & 3;\n"
"    return d * (float)(sc & 15) * (float)q - mn * (float)(sc >> 4);\n"
"}\n"
"__device__ __forceinline__ float dequant_elem(const uchar *base, int type, int row_bytes, int row, int k) {\n"
"    const uchar *r = base + (long)row * row_bytes;\n"
"    if (type == 8) {\n"
"        int b = k >> 5, j = k & 31; const uchar *p = r + b * 34;\n"
"        return h2f(u16le(p)) * (float)((signed char)p[2 + j]);\n"
"    }\n"
"    if (type == 41) {\n"
"        int b = k >> 7, j = k & 127; const uchar *p = r + b * 18; float d = h2f(u16le(p)); int bit = (p[2 + (j >> 3)] >> (j & 7)) & 1;\n"
"        return bit ? d : -d;\n"
"    }\n"
"    if (type == 39) {\n"
"        int b = k >> 5, j = k & 31; const uchar *p = r + b * 17; float d = e8m0_to_f32_half(p[0]); int q = p[1 + (j & 15)]; q = (j < 16) ? (q & 15) : (q >> 4);\n"
"        return d * (float)kvalues_mxfp4[q];\n"
"    }\n"
"    if (type == 40) {\n"
"        int b = k >> 6, kk = k & 63; const uchar *p = r + b * 36; int sub = kk >> 4, j = kk & 15; float d = ue4m3_to_f32(p[sub]); int q = p[4 + sub * 8 + (j & 7)]; q = (j < 8) ? (q & 15) : (q >> 4);\n"
"        return d * (float)kvalues_mxfp4[q];\n"
"    }\n"
"    if (type == 2) {\n"
"        int b = k >> 5, j = k & 31; const uchar *p = r + b * 18; float d = h2f(u16le(p));\n"
"        int q = p[2 + (j & 15)]; int v = ((j < 16 ? (q & 15) : (q >> 4)) - 8);\n"
"        return d * (float)v;\n"
"    }\n"
"    if (type == 3) {\n"
"        int b = k >> 5, j = k & 31; const uchar *p = r + b * 20; float d = h2f(u16le(p)); float m = h2f(u16le(p + 2));\n"
"        int q = p[4 + (j & 15)]; int v = (j < 16 ? (q & 15) : (q >> 4));\n"
"        return d * (float)v + m;\n"
"    }\n"
"    if (type == 6) {\n"
"        int b = k >> 5, j = k & 31; const uchar *p = r + b * 22; float d = h2f(u16le(p)); uint qh = u32le(p + 2);\n"
"        int q = p[6 + (j & 15)]; int hi = j < 16 ? (((qh >> j) << 4) & 0x10) : ((qh >> (j - 4)) & 0x10);\n"
"        int v = ((j < 16 ? (q & 15) : (q >> 4)) | hi) - 16;\n"
"        return d * (float)v;\n"
"    }\n"
"    if (type == 7) {\n"
"        int b = k >> 5, j = k & 31; const uchar *p = r + b * 24; float d = h2f(u16le(p)); float m = h2f(u16le(p + 2)); uint qh = u32le(p + 4);\n"
"        int q = p[8 + (j & 15)]; int hi = j < 16 ? (((qh >> j) << 4) & 0x10) : ((qh >> (j - 4)) & 0x10);\n"
"        int v = (j < 16 ? (q & 15) : (q >> 4)) | hi;\n"
"        return d * (float)v + m;\n"
"    }\n"
"    if (type == 10) {\n"
"        return dequant_q2_k_elem(base, row_bytes, row, k);\n"
"    }\n"
"    if (type == 11) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 110;\n"
"        float d = h2f(u16le(p + 108)); int half = kk >> 7; int rr = kk & 127; int group = rr >> 4;\n"
"        int shift = (group >> 1) * 2; int l = (rr & 15) + ((group & 1) ? 16 : 0); int is = half * 8 + group;\n"
"        uchar mbit = (uchar)(1u << (is >> 1)); int q = (p[32 + half * 32 + l] >> shift) & 3;\n"
"        int high = (p[l] & mbit) ? 0 : 4;\n"
"        return d * (float)(q3_scale(p + 96, is) - 32) * (float)(q - high);\n"
"    }\n"
"    if (type == 12) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 144;\n"
"        float d = h2f(u16le(p)); float mn = h2f(u16le(p + 2)); int group = kk >> 5; int l = kk & 31; uchar sc, mv;\n"
"        get_scale_min_k4(group, p + 4, &sc, &mv); int q = p[16 + (group >> 1) * 32 + l]; q = (group & 1) ? (q >> 4) : (q & 15);\n"
"        return d * (float)sc * (float)q - mn * (float)mv;\n"
"    }\n"
"    if (type == 13) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 176;\n"
"        float d = h2f(u16le(p)); float mn = h2f(u16le(p + 2)); int group = kk >> 5; int l = kk & 31; uchar sc, mv;\n"
"        get_scale_min_k4(group, p + 4, &sc, &mv); int ql = p[48 + (group >> 1) * 32 + l]; int qh = p[16 + l];\n"
"        int q = ((group & 1) ? (ql >> 4) : (ql & 15)) + ((qh & (1 << group)) ? 16 : 0);\n"
"        return d * (float)sc * (float)q - mn * (float)mv;\n"
"    }\n"
"    if (type == 14) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 210;\n"
"        float d = h2f(u16le(p + 208)); int half = kk >> 7; int rr = kk & 127; int seg = rr >> 5; int l = rr & 31;\n"
"        const uchar *ql = p + half * 64; const uchar *qh = p + 128 + half * 32; const signed char *sc = (const signed char *)(p + 192 + half * 8);\n"
"        int is = l >> 4; int q, sidx;\n"
"        if (seg == 0) { q = (ql[l] & 15) | (((qh[l] >> 0) & 3) << 4); sidx = is + 0; }\n"
"        else if (seg == 1) { q = (ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4); sidx = is + 2; }\n"
"        else if (seg == 2) { q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); sidx = is + 4; }\n"
"        else { q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); sidx = is + 6; }\n"
"        return d * (float)sc[sidx] * (float)(q - 32);\n"
"    }\n"
"    if (type == 16) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 66; float d = h2f(u16le(p));\n"
"        int ib = kk >> 5, in = kk & 31, l = in >> 3, j = in & 7; uint aux0 = u32le(p + 2 + 8*ib), aux1 = u32le(p + 2 + 8*ib + 4);\n"
"        float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f; int gidx = (aux0 >> (8*l)) & 255; uchar signs = ksigns_iq2xs[(aux1 >> (7*l)) & 127];\n"
"        return db * iq_grid_u8(iq2xxs_grid, gidx, j) * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);\n"
"    }\n"
"    if (type == 17) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 74; float d = h2f(u16le(p));\n"
"        int ib = kk >> 5, in = kk & 31, l = in >> 3, j = in & 7; ushort qv = u16le(p + 2 + 2*(4*ib + l));\n"
"        float db = d * (0.5f + (float)((l < 2) ? (p[66 + ib] & 15) : (p[66 + ib] >> 4))) * 0.25f; uchar signs = ksigns_iq2xs[qv >> 9];\n"
"        return db * iq_grid_u8(iq2xs_grid, qv & 511, j) * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);\n"
"    }\n"
"    if (type == 18) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 98; float d = h2f(u16le(p));\n"
"        int ib = kk >> 5, in = kk & 31, l = in >> 3, j = in & 7; uint aux = u32le(p + 2 + 64 + 4*ib);\n"
"        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f; uchar signs = ksigns_iq2xs[(aux >> (7*l)) & 127]; int gidx = p[2 + 8*ib + 2*l + (j >> 2)];\n"
"        return db * (float)iq_grid_u4(iq3xxs_grid, gidx, j & 3) * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);\n"
"    }\n"
"    if (type == 19) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 50; float d = h2f(u16le(p));\n"
"        int ib = kk >> 5, in = kk & 31, l = in >> 3, j = in & 7; ushort qh = u16le(p + 34 + 2*ib);\n"
"        float dl = d * (float)(2*((qh >> 12) & 7) + 1); float delta = (qh & 0x8000) ? -0.125f : 0.125f; int gidx = p[2 + 4*ib + l] | (((qh >> (3*l)) & 7) << 8);\n"
"        return dl * (iq_grid_i8(iq1s_grid, gidx, j) + delta);\n"
"    }\n"
"    if (type == 20) {\n"
"        int b = k >> 5, j = k & 31; const uchar *p = r + b * 18; float d = h2f(u16le(p)); int q = p[2 + (j & 15)]; q = (j < 16) ? (q & 15) : (q >> 4);\n"
"        return d * (float)kvalues_iq4nl[q];\n"
"    }\n"
"    if (type == 21) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 110; float d = h2f(u16le(p));\n"
"        int ib = kk >> 5, pair = ib >> 1, odd = ib & 1, in = kk & 31, l = in >> 3, j = in & 7; uchar scale = p[106 + pair];\n"
"        float db = d * (float)(1 + 2*(odd ? (scale >> 4) : (scale & 15))); uchar qhb = p[66 + 2*pair + odd]; const uchar *qs = p + 2 + 16*pair + 8*odd; const uchar *signs = p + 74 + 8*pair + 4*odd;\n"
"        int gidx = qs[2*l + (j >> 2)] | ((qhb << ((j >> 2) ? (7 - 2*l) : (8 - 2*l))) & 256);\n"
"        return db * (float)iq_grid_u4(iq3s_grid, gidx, j & 3) * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);\n"
"    }\n"
"    if (type == 22) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 82; float d = h2f(u16le(p));\n"
"        int ib = kk >> 5, in = kk & 31, l = in >> 3, j = in & 7; float db = d * (0.5f + (float)((l < 2) ? (p[74 + ib] & 15) : (p[74 + ib] >> 4))) * 0.25f;\n"
"        int gidx = p[2 + 4*ib + l] | ((p[66 + ib] << (8 - 2*l)) & 0x300); uchar signs = p[34 + 4*ib + l];\n"
"        return db * iq_grid_u8(iq2s_grid, gidx, j) * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);\n"
"    }\n"
"    if (type == 23) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 136; float d = h2f(u16le(p)); int ib = kk >> 5, j = kk & 31;\n"
"        int ls = ((p[4 + ib/2] >> (4*(ib&1))) & 15) | (((u16le(p + 2) >> (2*ib)) & 3) << 4); float dl = d * (float)(ls - 32); int q = p[8 + 16*ib + (j & 15)]; q = (j < 16) ? (q & 15) : (q >> 4);\n"
"        return dl * (float)kvalues_iq4nl[q];\n"
"    }\n"
"    if (type == 29) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 56; int ib = kk >> 5, in = kk & 31, l = in >> 3, j = in & 7;\n"
"        float d = h2f(iq1m_scale_u16(p + 48)); ushort sc = u16le(p + 48 + 2*(ib/2)); float dl = d * (float)(2*((sc >> (6*(ib&1) + (l < 2 ? 0 : 3))) & 7) + 1);\n"
"        uchar qh0 = p[32 + 2*ib], qh1 = p[33 + 2*ib]; int gidx; float delta;\n"
"        if (l == 0) { gidx = p[4*ib + 0] | ((qh0 << 8) & 0x700); delta = (qh0 & 0x08) ? -0.125f : 0.125f; }\n"
"        else if (l == 1) { gidx = p[4*ib + 1] | ((qh0 << 4) & 0x700); delta = (qh0 & 0x80) ? -0.125f : 0.125f; }\n"
"        else if (l == 2) { gidx = p[4*ib + 2] | ((qh1 << 8) & 0x700); delta = (qh1 & 0x08) ? -0.125f : 0.125f; }\n"
"        else { gidx = p[4*ib + 3] | ((qh1 << 4) & 0x700); delta = (qh1 & 0x80) ? -0.125f : 0.125f; }\n"
"        return dl * (iq_grid_i8(iq1s_grid, gidx, j) + delta);\n"
"    }\n"
"    if (type == 34) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 54; float d = h2f(u16le(p + 52)); const uchar pow3[5] = {1,3,9,27,81}; int n, m, off; uchar qb;\n"
"        if (kk < 160) { n = kk >> 5; m = kk & 31; qb = p[m]; }\n"
"        else if (kk < 240) { off = kk - 160; n = off / 16; m = off & 15; qb = p[32 + m]; }\n"
"        else { off = kk - 240; n = off >> 2; m = off & 3; qb = p[48 + m]; }\n"
"        uchar q = (uchar)(qb * pow3[n]); int xi = ((ushort)q * 3) >> 8; return (float)(xi - 1) * d;\n"
"    }\n"
"    if (type == 35) {\n"
"        int b = k >> 8, kk = k & 255; const uchar *p = r + b * 66; float d = h2f(u16le(p + 64)); int half = kk >> 7, rem = kk & 127, l = rem >> 5, m = rem & 31;\n"
"        int q = (p[half * 32 + m] >> (2*l)) & 3; return (float)(q - 1) * d;\n"
"    }\n"
"    return 0.0f;\n"
"}\n"
"__device__ __forceinline__ short f32_to_bf16(float v) { uint bits; memcpy(&bits, &v, 4); return (short)(bits >> 16); }\n"
"__device__ __forceinline__ signed char f32_to_i8(float v) { float s = v * 16.0f; int q = (int)(s + (s >= 0.0f ? 0.5f : -0.5f)); if (q > 127) q = 127; if (q < -127) q = -127; return (signed char)q; }\n"
"__device__ __forceinline__ uchar f32_to_fp8(float f) {\n"
"    if (f != f) return 0x7f; if (f == 0.0f) return 0; uint bits; memcpy(&bits, &f, 4);\n"
"    uint sign = (bits >> 31) & 1; int exp = (int)((bits >> 23) & 255) - 127 + 7; uint mant = (bits >> 20) & 7;\n"
"    if (exp >= 15) { exp = 15; mant = 6; } if (exp <= 0) return (uchar)(sign << 7);\n"
"    return (uchar)((sign << 7) | ((exp & 15) << 3) | (mant & 7));\n"
"}\n"
"__device__ __forceinline__ uint pack4_i8(signed char a, signed char b, signed char c, signed char d) {\n"
"    return (uint)(uchar)a | ((uint)(uchar)b << 8) | ((uint)(uchar)c << 16) | ((uint)(uchar)d << 24);\n"
"}\n"
"__device__ __forceinline__ uint pack4_u8(uchar a, uchar b, uchar c, uchar d) {\n"
"    return (uint)a | ((uint)b << 8) | ((uint)c << 16) | ((uint)d << 24);\n"
"}\n"
"__device__ __forceinline__ signed char round_clamp_i8(float v) {\n"
"    int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));\n"
"    if (q > 127) q = 127; if (q < -127) q = -127; return (signed char)q;\n"
"}\n"
"\n"
"extern \"C\" {\n"
"__global__ void dequant_f32_kernel(const uchar *W, float *Y, int type, int rows, int cols, int row_bytes) {\n"
"    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; long n = (long)rows * cols;\n"
"    if (idx >= n) return; int row = (int)(idx / cols); int k = (int)(idx - (long)row * cols);\n"
"    Y[idx] = dequant_elem(W, type, row_bytes, row, k);\n"
"}\n"
"\n"
"__global__ void quant_q8_0_kernel(const float *X, uchar *Q, int rows, int cols, int row_bytes) {\n"
"    long ib = (long)blockIdx.x * blockDim.x + threadIdx.x; int blocks_per_row = cols >> 5; long nblocks = (long)rows * blocks_per_row;\n"
"    if (ib >= nblocks) return; int row = (int)(ib / blocks_per_row); int b = (int)(ib - (long)row * blocks_per_row);\n"
"    const float *src = X + (long)row * cols + b * 32; uchar *dst = Q + (long)row * row_bytes + b * 34;\n"
"    float amax = 0.0f; for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(src[j]));\n"
"    float d = amax * (1.0f / 127.0f); float id = d > 0.0f ? 1.0f / d : 0.0f; ushort dh = f32_to_f16_bits(d);\n"
"    dst[0] = (uchar)(dh & 255); dst[1] = (uchar)(dh >> 8);\n"
"    for (int j = 0; j < 32; j++) dst[2 + j] = (uchar)round_clamp_i8(src[j] * id);\n"
"}\n"
"\n"
"__global__ void quant_q8_1_kernel(const float *X, uchar *Q, int rows, int cols, int row_bytes) {\n"
"    long ib = (long)blockIdx.x * blockDim.x + threadIdx.x; int blocks_per_row = cols >> 5; long nblocks = (long)rows * blocks_per_row;\n"
"    if (ib >= nblocks) return; int row = (int)(ib / blocks_per_row); int b = (int)(ib - (long)row * blocks_per_row);\n"
"    const float *src = X + (long)row * cols + b * 32; uchar *dst = Q + (long)row * row_bytes + b * 36;\n"
"    float amax = 0.0f, sum = 0.0f; for (int j = 0; j < 32; j++) { float v = src[j]; amax = fmaxf(amax, fabsf(v)); sum += v; }\n"
"    float d = amax * (1.0f / 127.0f); float id = d > 0.0f ? 1.0f / d : 0.0f; ushort dh = f32_to_f16_bits(d); ushort sh = f32_to_f16_bits(sum * d);\n"
"    dst[0] = (uchar)(dh & 255); dst[1] = (uchar)(dh >> 8); dst[2] = (uchar)(sh & 255); dst[3] = (uchar)(sh >> 8);\n"
"    for (int j = 0; j < 32; j++) dst[4 + j] = (uchar)round_clamp_i8(src[j] * id);\n"
"}\n"
"\n"
"__device__ __forceinline__ float dot_q2_k_q8_1_block(const uchar *w, const uchar *xq8) {\n"
"    const float d = h2f(u16le(w + 80)); const float dmin = h2f(u16le(w + 82)); float acc = 0.0f;\n"
"    for (int half = 0; half < 2; half++) {\n"
"        const uchar *qs = w + 16 + half * 32;\n"
"        for (int pair = 0; pair < 4; pair++) {\n"
"            int shift = pair * 2; uchar sc0 = w[half * 8 + pair * 2 + 0]; uchar sc1 = w[half * 8 + pair * 2 + 1];\n"
"            const uchar *xb = xq8 + (half * 4 + pair) * 36; float dx = h2f(u16le(xb)); const signed char *xv = (const signed char *)(xb + 4);\n"
"            int dot0 = 0, dot1 = 0, sum0 = 0, sum1 = 0;\n"
"            for (int j = 0; j < 16; j++) { int x0 = (int)xv[j]; int x1 = (int)xv[16 + j]; dot0 += (int)((qs[j] >> shift) & 3) * x0; dot1 += (int)((qs[16 + j] >> shift) & 3) * x1; sum0 += x0; sum1 += x1; }\n"
"            acc += d * dx * ((float)(sc0 & 15) * (float)dot0 + (float)(sc1 & 15) * (float)dot1);\n"
"            acc -= dmin * dx * ((float)(sc0 >> 4) * (float)sum0 + (float)(sc1 >> 4) * (float)sum1);\n"
"        }\n"
"    }\n"
"    return acc;\n"
"}\n"
"\n"
"__global__ void gemm_q2k_q8_1_llama(float *Y, const uchar *W, const uchar *XQ, int n_out, int n_in, int n_tok, int row_bytes, int x_row_bytes) {\n"
"    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; long total = (long)n_tok * n_out; if (idx >= total) return;\n"
"    int row = (int)(idx % n_out); int tok = (int)(idx / n_out); int nb = n_in >> 8; float acc = 0.0f;\n"
"    const uchar *wr = W + (long)row * row_bytes; const uchar *xr = XQ + (long)tok * x_row_bytes;\n"
"    for (int b = 0; b < nb; b++) acc += dot_q2_k_q8_1_block(wr + b * 84, xr + b * 8 * 36);\n"
"    Y[idx] = acc;\n"
"}\n"
"\n"
"#if defined(__gfx1200__) || defined(__gfx1201__)\n"
"__global__ void gemm_dequant_bf16_wmma(float *Y, const uchar *W, const float *X, int type, int n_out, int n_in, int n_tok, int row_bytes) {\n"
"    int tid = threadIdx.x, wave_id = tid >> 5, lane = tid & 31, wM = wave_id & 1, wN = wave_id >> 1;\n"
"    int half = lane >> 4, idx = lane & 15, k_off = half * 8, cta_m0 = blockIdx.y * 128, cta_n0 = blockIdx.x * 128;\n"
"    __shared__ short smA[128*16]; __shared__ short smB[128*16];\n"
"    float8 z = {0,0,0,0,0,0,0,0}; float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        for (int it = 0; it < 8; it++) { int e = tid * 8 + it; int er = e >> 4, ek = e & 15; int row = cta_m0 + er, kp = k + ek; float xv = (row < n_tok && kp < n_in) ? X[(long)row * n_in + kp] : 0.0f; smA[e] = f32_to_bf16(xv); }\n"
"        for (int it = 0; it < 8; it++) { int e = tid * 8 + it; int er = e >> 4, ek = e & 15; int col = cta_n0 + er, kp = k + ek; float wv = (col < n_out && kp < n_in) ? dequant_elem(W, type, row_bytes, col, kp) : 0.0f; smB[e] = f32_to_bf16(wv); }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64, b_base = wN * 32; bf16x8 a0,a1,a2,a3,b0,b1;\n"
"        for (int i = 0; i < 8; i++) { a0[i]=smA[(a_base+0+idx)*16+k_off+i]; a1[i]=smA[(a_base+16+idx)*16+k_off+i]; a2[i]=smA[(a_base+32+idx)*16+k_off+i]; a3[i]=smA[(a_base+48+idx)*16+k_off+i]; b0[i]=smB[(b_base+0+idx)*16+k_off+i]; b1[i]=smB[(b_base+16+idx)*16+k_off+i]; }\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00); cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"        cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10); cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"        cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20); cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"        cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30); cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64, wave_n0 = cta_n0 + wN * 32; float8 *accs[8]={&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31}; int ms[8]={0,0,16,16,32,32,48,48}; int ns[8]={0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) { int col = wave_n0 + ns[t] + idx; if (col >= n_out) continue; float8 acc = *accs[t]; for (int i = 0; i < 8; i++) { int row = wave_m0 + ms[t] + half * 8 + i; if (row < n_tok) Y[(long)row * n_out + col] = acc[i]; } }\n"
"}\n"
"\n"
"__global__ void gemm_dequant_fp8_wmma(float *Y, const uchar *W, const float *X, int type, int n_out, int n_in, int n_tok, int row_bytes) {\n"
"    int tid = threadIdx.x, wave_id = tid >> 5, lane = tid & 31, wM = wave_id & 1, wN = wave_id >> 1;\n"
"    int half = lane >> 4, idx = lane & 15, k_off = half * 8, cta_m0 = blockIdx.y * 128, cta_n0 = blockIdx.x * 128;\n"
"    __shared__ uchar smA[128*16]; __shared__ uchar smB[128*16];\n"
"    float8 z = {0,0,0,0,0,0,0,0}; float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        for (int it = 0; it < 8; it++) { int e = tid * 8 + it; int er = e >> 4, ek = e & 15; int row = cta_m0 + er, kp = k + ek; float xv = (row < n_tok && kp < n_in) ? X[(long)row * n_in + kp] : 0.0f; smA[e] = f32_to_fp8(xv); }\n"
"        for (int it = 0; it < 8; it++) { int e = tid * 8 + it; int er = e >> 4, ek = e & 15; int col = cta_n0 + er, kp = k + ek; float wv = (col < n_out && kp < n_in) ? dequant_elem(W, type, row_bytes, col, kp) : 0.0f; smB[e] = f32_to_fp8(wv); }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64, b_base = wN * 32; int32x2 a0,a1,a2,a3,b0,b1;\n"
"        a0[0]=pack4_u8(smA[(a_base+0+idx)*16+k_off+0],smA[(a_base+0+idx)*16+k_off+1],smA[(a_base+0+idx)*16+k_off+2],smA[(a_base+0+idx)*16+k_off+3]); a0[1]=pack4_u8(smA[(a_base+0+idx)*16+k_off+4],smA[(a_base+0+idx)*16+k_off+5],smA[(a_base+0+idx)*16+k_off+6],smA[(a_base+0+idx)*16+k_off+7]);\n"
"        a1[0]=pack4_u8(smA[(a_base+16+idx)*16+k_off+0],smA[(a_base+16+idx)*16+k_off+1],smA[(a_base+16+idx)*16+k_off+2],smA[(a_base+16+idx)*16+k_off+3]); a1[1]=pack4_u8(smA[(a_base+16+idx)*16+k_off+4],smA[(a_base+16+idx)*16+k_off+5],smA[(a_base+16+idx)*16+k_off+6],smA[(a_base+16+idx)*16+k_off+7]);\n"
"        a2[0]=pack4_u8(smA[(a_base+32+idx)*16+k_off+0],smA[(a_base+32+idx)*16+k_off+1],smA[(a_base+32+idx)*16+k_off+2],smA[(a_base+32+idx)*16+k_off+3]); a2[1]=pack4_u8(smA[(a_base+32+idx)*16+k_off+4],smA[(a_base+32+idx)*16+k_off+5],smA[(a_base+32+idx)*16+k_off+6],smA[(a_base+32+idx)*16+k_off+7]);\n"
"        a3[0]=pack4_u8(smA[(a_base+48+idx)*16+k_off+0],smA[(a_base+48+idx)*16+k_off+1],smA[(a_base+48+idx)*16+k_off+2],smA[(a_base+48+idx)*16+k_off+3]); a3[1]=pack4_u8(smA[(a_base+48+idx)*16+k_off+4],smA[(a_base+48+idx)*16+k_off+5],smA[(a_base+48+idx)*16+k_off+6],smA[(a_base+48+idx)*16+k_off+7]);\n"
"        b0[0]=pack4_u8(smB[(b_base+0+idx)*16+k_off+0],smB[(b_base+0+idx)*16+k_off+1],smB[(b_base+0+idx)*16+k_off+2],smB[(b_base+0+idx)*16+k_off+3]); b0[1]=pack4_u8(smB[(b_base+0+idx)*16+k_off+4],smB[(b_base+0+idx)*16+k_off+5],smB[(b_base+0+idx)*16+k_off+6],smB[(b_base+0+idx)*16+k_off+7]);\n"
"        b1[0]=pack4_u8(smB[(b_base+16+idx)*16+k_off+0],smB[(b_base+16+idx)*16+k_off+1],smB[(b_base+16+idx)*16+k_off+2],smB[(b_base+16+idx)*16+k_off+3]); b1[1]=pack4_u8(smB[(b_base+16+idx)*16+k_off+4],smB[(b_base+16+idx)*16+k_off+5],smB[(b_base+16+idx)*16+k_off+6],smB[(b_base+16+idx)*16+k_off+7]);\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0,b0,cv00); cv01=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0,b1,cv01); cv10=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1,b0,cv10); cv11=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1,b1,cv11);\n"
"        cv20=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a2,b0,cv20); cv21=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a2,b1,cv21); cv30=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a3,b0,cv30); cv31=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a3,b1,cv31);\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64, wave_n0 = cta_n0 + wN * 32; float8 *accs[8]={&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31}; int ms[8]={0,0,16,16,32,32,48,48}; int ns[8]={0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) { int col = wave_n0 + ns[t] + idx; if (col >= n_out) continue; float8 acc = *accs[t]; for (int i = 0; i < 8; i++) { int row = wave_m0 + ms[t] + half * 8 + i; if (row < n_tok) Y[(long)row * n_out + col] = acc[i]; } }\n"
"}\n"
"\n"
"__global__ void gemm_q2k_fp8_wmma(float *Y, const uchar *W, const float *X, int n_out, int n_in, int n_tok, int row_bytes) {\n"
"    int tid = threadIdx.x, wave_id = tid >> 5, lane = tid & 31, wM = wave_id & 1, wN = wave_id >> 1;\n"
"    int half = lane >> 4, idx = lane & 15, k_off = half * 8, cta_m0 = blockIdx.y * 128, cta_n0 = blockIdx.x * 128;\n"
"    __shared__ uchar smA[128*16]; __shared__ uchar smB[128*16];\n"
"    float8 z = {0,0,0,0,0,0,0,0}; float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        for (int it = 0; it < 8; it++) { int e = tid * 8 + it; int er = e >> 4, ek = e & 15; int row = cta_m0 + er, kp = k + ek; float xv = (row < n_tok && kp < n_in) ? X[(long)row * n_in + kp] : 0.0f; smA[e] = f32_to_fp8(xv); }\n"
"        for (int it = 0; it < 8; it++) { int e = tid * 8 + it; int er = e >> 4, ek = e & 15; int col = cta_n0 + er, kp = k + ek; float wv = (col < n_out && kp < n_in) ? dequant_q2_k_elem(W, row_bytes, col, kp) : 0.0f; smB[e] = f32_to_fp8(wv); }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64, b_base = wN * 32; int32x2 a0,a1,a2,a3,b0,b1;\n"
"        a0[0]=pack4_u8(smA[(a_base+0+idx)*16+k_off+0],smA[(a_base+0+idx)*16+k_off+1],smA[(a_base+0+idx)*16+k_off+2],smA[(a_base+0+idx)*16+k_off+3]); a0[1]=pack4_u8(smA[(a_base+0+idx)*16+k_off+4],smA[(a_base+0+idx)*16+k_off+5],smA[(a_base+0+idx)*16+k_off+6],smA[(a_base+0+idx)*16+k_off+7]);\n"
"        a1[0]=pack4_u8(smA[(a_base+16+idx)*16+k_off+0],smA[(a_base+16+idx)*16+k_off+1],smA[(a_base+16+idx)*16+k_off+2],smA[(a_base+16+idx)*16+k_off+3]); a1[1]=pack4_u8(smA[(a_base+16+idx)*16+k_off+4],smA[(a_base+16+idx)*16+k_off+5],smA[(a_base+16+idx)*16+k_off+6],smA[(a_base+16+idx)*16+k_off+7]);\n"
"        a2[0]=pack4_u8(smA[(a_base+32+idx)*16+k_off+0],smA[(a_base+32+idx)*16+k_off+1],smA[(a_base+32+idx)*16+k_off+2],smA[(a_base+32+idx)*16+k_off+3]); a2[1]=pack4_u8(smA[(a_base+32+idx)*16+k_off+4],smA[(a_base+32+idx)*16+k_off+5],smA[(a_base+32+idx)*16+k_off+6],smA[(a_base+32+idx)*16+k_off+7]);\n"
"        a3[0]=pack4_u8(smA[(a_base+48+idx)*16+k_off+0],smA[(a_base+48+idx)*16+k_off+1],smA[(a_base+48+idx)*16+k_off+2],smA[(a_base+48+idx)*16+k_off+3]); a3[1]=pack4_u8(smA[(a_base+48+idx)*16+k_off+4],smA[(a_base+48+idx)*16+k_off+5],smA[(a_base+48+idx)*16+k_off+6],smA[(a_base+48+idx)*16+k_off+7]);\n"
"        b0[0]=pack4_u8(smB[(b_base+0+idx)*16+k_off+0],smB[(b_base+0+idx)*16+k_off+1],smB[(b_base+0+idx)*16+k_off+2],smB[(b_base+0+idx)*16+k_off+3]); b0[1]=pack4_u8(smB[(b_base+0+idx)*16+k_off+4],smB[(b_base+0+idx)*16+k_off+5],smB[(b_base+0+idx)*16+k_off+6],smB[(b_base+0+idx)*16+k_off+7]);\n"
"        b1[0]=pack4_u8(smB[(b_base+16+idx)*16+k_off+0],smB[(b_base+16+idx)*16+k_off+1],smB[(b_base+16+idx)*16+k_off+2],smB[(b_base+16+idx)*16+k_off+3]); b1[1]=pack4_u8(smB[(b_base+16+idx)*16+k_off+4],smB[(b_base+16+idx)*16+k_off+5],smB[(b_base+16+idx)*16+k_off+6],smB[(b_base+16+idx)*16+k_off+7]);\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0,b0,cv00); cv01=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0,b1,cv01); cv10=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1,b0,cv10); cv11=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1,b1,cv11);\n"
"        cv20=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a2,b0,cv20); cv21=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a2,b1,cv21); cv30=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a3,b0,cv30); cv31=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a3,b1,cv31);\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64, wave_n0 = cta_n0 + wN * 32; float8 *accs[8]={&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31}; int ms[8]={0,0,16,16,32,32,48,48}; int ns[8]={0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) { int col = wave_n0 + ns[t] + idx; if (col >= n_out) continue; float8 acc = *accs[t]; for (int i = 0; i < 8; i++) { int row = wave_m0 + ms[t] + half * 8 + i; if (row < n_tok) Y[(long)row * n_out + col] = acc[i]; } }\n"
"}\n"
"\n"
"__global__ void gemm_dequant_int8_wmma(float *Y, const uchar *W, const float *X, int type, int n_out, int n_in, int n_tok, int row_bytes) {\n"
"    int tid = threadIdx.x, wave_id = tid >> 5, lane = tid & 31, wM = wave_id & 1, wN = wave_id >> 1;\n"
"    int half = lane >> 4, idx = lane & 15, k_off = half * 8, cta_m0 = blockIdx.y * 128, cta_n0 = blockIdx.x * 128;\n"
"    __shared__ signed char smA[128*16]; __shared__ signed char smB[128*16];\n"
"    int32x8 z = {0,0,0,0,0,0,0,0}; int32x8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        for (int it = 0; it < 8; it++) { int e = tid * 8 + it; int er = e >> 4, ek = e & 15; int row = cta_m0 + er, kp = k + ek; float xv = (row < n_tok && kp < n_in) ? X[(long)row * n_in + kp] : 0.0f; smA[e] = f32_to_i8(xv); }\n"
"        for (int it = 0; it < 8; it++) { int e = tid * 8 + it; int er = e >> 4, ek = e & 15; int col = cta_n0 + er, kp = k + ek; float wv = (col < n_out && kp < n_in) ? dequant_elem(W, type, row_bytes, col, kp) : 0.0f; smB[e] = f32_to_i8(wv); }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64, b_base = wN * 32; uint32x2 a0,a1,a2,a3,b0,b1;\n"
"        a0[0]=pack4_i8(smA[(a_base+0+idx)*16+k_off+0],smA[(a_base+0+idx)*16+k_off+1],smA[(a_base+0+idx)*16+k_off+2],smA[(a_base+0+idx)*16+k_off+3]); a0[1]=pack4_i8(smA[(a_base+0+idx)*16+k_off+4],smA[(a_base+0+idx)*16+k_off+5],smA[(a_base+0+idx)*16+k_off+6],smA[(a_base+0+idx)*16+k_off+7]);\n"
"        a1[0]=pack4_i8(smA[(a_base+16+idx)*16+k_off+0],smA[(a_base+16+idx)*16+k_off+1],smA[(a_base+16+idx)*16+k_off+2],smA[(a_base+16+idx)*16+k_off+3]); a1[1]=pack4_i8(smA[(a_base+16+idx)*16+k_off+4],smA[(a_base+16+idx)*16+k_off+5],smA[(a_base+16+idx)*16+k_off+6],smA[(a_base+16+idx)*16+k_off+7]);\n"
"        a2[0]=pack4_i8(smA[(a_base+32+idx)*16+k_off+0],smA[(a_base+32+idx)*16+k_off+1],smA[(a_base+32+idx)*16+k_off+2],smA[(a_base+32+idx)*16+k_off+3]); a2[1]=pack4_i8(smA[(a_base+32+idx)*16+k_off+4],smA[(a_base+32+idx)*16+k_off+5],smA[(a_base+32+idx)*16+k_off+6],smA[(a_base+32+idx)*16+k_off+7]);\n"
"        a3[0]=pack4_i8(smA[(a_base+48+idx)*16+k_off+0],smA[(a_base+48+idx)*16+k_off+1],smA[(a_base+48+idx)*16+k_off+2],smA[(a_base+48+idx)*16+k_off+3]); a3[1]=pack4_i8(smA[(a_base+48+idx)*16+k_off+4],smA[(a_base+48+idx)*16+k_off+5],smA[(a_base+48+idx)*16+k_off+6],smA[(a_base+48+idx)*16+k_off+7]);\n"
"        b0[0]=pack4_i8(smB[(b_base+0+idx)*16+k_off+0],smB[(b_base+0+idx)*16+k_off+1],smB[(b_base+0+idx)*16+k_off+2],smB[(b_base+0+idx)*16+k_off+3]); b0[1]=pack4_i8(smB[(b_base+0+idx)*16+k_off+4],smB[(b_base+0+idx)*16+k_off+5],smB[(b_base+0+idx)*16+k_off+6],smB[(b_base+0+idx)*16+k_off+7]);\n"
"        b1[0]=pack4_i8(smB[(b_base+16+idx)*16+k_off+0],smB[(b_base+16+idx)*16+k_off+1],smB[(b_base+16+idx)*16+k_off+2],smB[(b_base+16+idx)*16+k_off+3]); b1[1]=pack4_i8(smB[(b_base+16+idx)*16+k_off+4],smB[(b_base+16+idx)*16+k_off+5],smB[(b_base+16+idx)*16+k_off+6],smB[(b_base+16+idx)*16+k_off+7]);\n"
"        cv00=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a0,true,b0,cv00,false); cv01=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a0,true,b1,cv01,false); cv10=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a1,true,b0,cv10,false); cv11=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a1,true,b1,cv11,false);\n"
"        cv20=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a2,true,b0,cv20,false); cv21=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a2,true,b1,cv21,false); cv30=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a3,true,b0,cv30,false); cv31=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a3,true,b1,cv31,false);\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64, wave_n0 = cta_n0 + wN * 32; int32x8 *accs[8]={&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31}; int ms[8]={0,0,16,16,32,32,48,48}; int ns[8]={0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) { int col = wave_n0 + ns[t] + idx; if (col >= n_out) continue; int32x8 acc = *accs[t]; for (int i = 0; i < 8; i++) { int row = wave_m0 + ms[t] + half * 8 + i; if (row < n_tok) Y[(long)row * n_out + col] = (float)acc[i] * (1.0f / 256.0f); } }\n"
"}\n"
"#endif\n"
"}\n";

static void die_hip(hipError_t err, const char *expr, const char *file, int line) {
    if (err == hipSuccess) return;
    const char *s = "?";
    if (hipGetErrorString) hipGetErrorString(err, &s);
    fprintf(stderr, "HIP error %s:%d: %s failed: %s (%d)\n", file, line, expr, s, err);
    exit(1);
}

#define CHECK_HIP(expr) die_hip((expr), #expr, __FILE__, __LINE__)

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} strbuf;

static void sb_append(strbuf *sb, const char *s, size_t n) {
    if (sb->len + n + 1 > sb->cap) {
        size_t nc = sb->cap ? sb->cap * 2 : 4096;
        while (nc < sb->len + n + 1) nc *= 2;
        char *p = (char *)realloc(sb->data, nc);
        if (!p) {
            fprintf(stderr, "out of memory while building HIP source\n");
            exit(1);
        }
        sb->data = p;
        sb->cap = nc;
    }
    memcpy(sb->data + sb->len, s, n);
    sb->len += n;
    sb->data[sb->len] = '\0';
}

static char *read_text_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long sz = ftell(fp);
    if (sz < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t nr = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[nr] = '\0';
    if (out_size) *out_size = nr;
    return buf;
}

static void append_transformed_table(strbuf *sb, const char *common_src, const char *name) {
    const char *name_pos = strstr(common_src, name);
    if (!name_pos) {
        fprintf(stderr, "missing lookup table %s in common/ggml_dequant.h\n", name);
        exit(1);
    }
    const char *start = name_pos;
    while (start > common_src && strncmp(start, "static const ", 13) != 0) start--;
    if (strncmp(start, "static const ", 13) != 0) {
        fprintf(stderr, "bad lookup table declaration for %s\n", name);
        exit(1);
    }
    const char *end = strstr(name_pos, "};");
    if (!end) {
        fprintf(stderr, "bad lookup table terminator for %s\n", name);
        exit(1);
    }
    end += 2;
    const char prefix[] = "__device__ static const ";
    sb_append(sb, prefix, sizeof(prefix) - 1);
    sb_append(sb, start + 13, (size_t)(end - (start + 13)));
    sb_append(sb, "\n", 1);
}

static char *build_kernel_source(void) {
    static const char *tables[] = {
        "iq2xxs_grid",
        "ksigns_iq2xs",
        "kmask_iq2xs",
        "kvalues_iq4nl",
        "kvalues_mxfp4",
        "iq3xxs_grid",
        "iq3s_grid",
        "iq2xs_grid",
        "iq2s_grid",
        "iq1s_grid",
    };
    const char marker[] = "__device__ __forceinline__ float h2f";
    const char *insert = strstr(kernel_src, marker);
    if (!insert) {
        fprintf(stderr, "internal error: HIP source insertion marker not found\n");
        exit(1);
    }

    size_t common_size = 0;
    char *common_src = read_text_file("../../common/ggml_dequant.h", &common_size);
    if (!common_src) common_src = read_text_file("common/ggml_dequant.h", &common_size);
    if (!common_src) {
        fprintf(stderr, "could not read common/ggml_dequant.h for IQ lookup tables\n");
        exit(1);
    }

    strbuf sb = {0};
    sb_append(&sb, kernel_src, (size_t)(insert - kernel_src));
    for (size_t i = 0; i < sizeof(tables) / sizeof(tables[0]); i++) {
        append_transformed_table(&sb, common_src, tables[i]);
    }
    sb_append(&sb, insert, strlen(insert));
    free(common_src);
    return sb.data;
}

static uint32_t lcg_next(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static void put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 255u);
    p[1] = (uint8_t)(v >> 8);
}

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        uint32_t round = (mant >> (shift - 1)) & 1u;
        uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        half_mant += round && (sticky || (half_mant & 1u));
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t round = (mant >> 12) & 1u;
    uint32_t sticky = mant & 0xfffu;
    half += round && (sticky || (half & 1u));
    return (uint16_t)half;
}

static int round_clamp_i8_cpu(float v) {
    int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return q;
}

static void quantize_row_q8_0_cpu(const float *src, uint8_t *dst, int n) {
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        const float *x = src + b * 32;
        uint8_t *q = dst + b * 34;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[j]));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        put_u16(q, fp32_to_fp16(d));
        for (int j = 0; j < 32; j++) q[2 + j] = (uint8_t)(int8_t)round_clamp_i8_cpu(x[j] * id);
    }
}

static uint16_t scale_half_for_type(uint32_t type) {
    if (type == GGML_TYPE_Q8_0) return 0x2800u; /* 0.03125 */
    return 0x2c00u;                             /* 0.0625 */
}

static void fill_quant(uint8_t *data, const quant_type_info *qt, int rows, int cols) {
    uint32_t seed = 0x1234abcdU ^ qt->type;
    size_t row_bytes = dequant_row_size(qt->type, cols);
    for (int r = 0; r < rows; r++) {
        uint8_t *row = data + (size_t)r * row_bytes;
        for (size_t i = 0; i < row_bytes; i++) row[i] = (uint8_t)(lcg_next(&seed) >> 24);
        int nb = cols / qt->block;
        for (int b = 0; b < nb; b++) {
            uint8_t *p = row + (size_t)b * qt->bytes;
            switch (qt->type) {
                case GGML_TYPE_Q4_0:
                    put_u16(p + 0, scale_half_for_type(qt->type));
                    break;
                case GGML_TYPE_Q4_1:
                    put_u16(p + 0, 0x2800u); put_u16(p + 2, 0x0000u);
                    break;
                case GGML_TYPE_Q5_0:
                    put_u16(p + 0, 0x2800u);
                    break;
                case GGML_TYPE_Q5_1:
                    put_u16(p + 0, 0x2800u); put_u16(p + 2, 0x0000u);
                    break;
                case GGML_TYPE_Q8_0:
                    put_u16(p + 0, scale_half_for_type(qt->type));
                    break;
                case GGML_TYPE_Q1_0:
                    put_u16(p + 0, 0x2c00u);
                    break;
                case GGML_TYPE_MXFP4:
                    p[0] = 124u;
                    break;
                case GGML_TYPE_NVFP4:
                    p[0] = 0x20u; p[1] = 0x20u; p[2] = 0x20u; p[3] = 0x20u;
                    break;
                case GGML_TYPE_Q2_K:
                    put_u16(p + 80, 0x2c00u); put_u16(p + 82, 0x0000u);
                    for (int i = 0; i < 16; i++) p[i] = (uint8_t)(1 + (p[i] & 3));
                    break;
                case GGML_TYPE_Q3_K:
                    put_u16(p + 108, 0x2800u);
                    for (int i = 96; i < 108; i++) p[i] = (uint8_t)(0x20u | (p[i] & 7u));
                    break;
                case GGML_TYPE_Q4_K:
                    put_u16(p + 0, 0x2800u); put_u16(p + 2, 0x0000u);
                    for (int i = 4; i < 16; i++) p[i] &= 0x1f;
                    break;
                case GGML_TYPE_Q5_K:
                    put_u16(p + 0, 0x2400u); put_u16(p + 2, 0x0000u);
                    for (int i = 4; i < 16; i++) p[i] &= 0x1f;
                    break;
                case GGML_TYPE_Q6_K:
                    put_u16(p + 208, 0x2400u);
                    for (int i = 192; i < 208; i++) p[i] = (uint8_t)((int8_t)((p[i] & 15) - 8));
                    break;
                case GGML_TYPE_IQ2_XXS:
                    put_u16(p + 0, 0x2c00u);
                    break;
                case GGML_TYPE_IQ2_XS:
                    put_u16(p + 0, 0x2c00u);
                    for (int i = 66; i < 74; i++) p[i] = 0x11u;
                    break;
                case GGML_TYPE_IQ2_S:
                    put_u16(p + 0, 0x2c00u);
                    for (int i = 74; i < 82; i++) p[i] = 0x11u;
                    break;
                case GGML_TYPE_IQ3_XXS:
                    put_u16(p + 0, 0x2800u);
                    break;
                case GGML_TYPE_IQ3_S:
                    put_u16(p + 0, 0x2800u);
                    for (int i = 106; i < 110; i++) p[i] = 0x11u;
                    break;
                case GGML_TYPE_IQ1_S:
                    put_u16(p + 0, 0x2c00u);
                    break;
                case GGML_TYPE_IQ1_M:
                    put_u16(p + 48, 0x0924u);
                    put_u16(p + 50, 0xc924u);
                    put_u16(p + 52, 0x2924u);
                    put_u16(p + 54, 0x0924u);
                    break;
                case GGML_TYPE_IQ4_XS:
                    put_u16(p + 0, 0x2400u);
                    put_u16(p + 2, 0xaaaau);
                    for (int i = 4; i < 8; i++) p[i] = 0x11u;
                    break;
                case GGML_TYPE_IQ4_NL:
                    put_u16(p + 0, 0x2400u);
                    break;
                case GGML_TYPE_TQ1_0:
                    put_u16(p + 52, 0x2c00u);
                    break;
                case GGML_TYPE_TQ2_0:
                    put_u16(p + 64, 0x2c00u);
                    break;
                default:
                    break;
            }
        }
    }
}

static void fill_x(float *x, int tokens, int cols) {
    for (int t = 0; t < tokens; t++) {
        for (int k = 0; k < cols; k++) {
            int v = ((t * 131 + k * 17) & 255) - 128;
            x[(size_t)t * cols + k] = (float)v / 128.0f;
        }
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static float elapsed_kernel_ms(hipFunction_t fn, unsigned gx, unsigned gy, unsigned bx,
                               void **args, int iters) {
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipEventRecord(start, NULL));
    for (int i = 0; i < iters; i++) {
        CHECK_HIP(hipModuleLaunchKernel(fn, gx, gy, 1, bx, 1, 1, 0, NULL, args, NULL));
    }
    CHECK_HIP(hipEventRecord(stop, NULL));
    CHECK_HIP(hipEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    return ms / (float)iters;
}

static void cpu_dequant_all(const uint8_t *src, float *dst, const quant_type_info *qt, int rows, int cols) {
    size_t rb = dequant_row_size(qt->type, cols);
    for (int r = 0; r < rows; r++) {
        dequant_row(qt->type, src + (size_t)r * rb, dst + (size_t)r * cols, cols);
    }
}

static void cpu_quant_q8_0_all(const float *src, uint8_t *dst, int rows, int cols, size_t row_bytes) {
    for (int r = 0; r < rows; r++) {
        quantize_row_q8_0_cpu(src + (size_t)r * cols, dst + (size_t)r * row_bytes, cols);
    }
}

static float cpu_dot_q2_k_q8_1_block(const uint8_t *w, const uint8_t *xq8) {
    const float d = ggml_fp16_to_fp32((uint16_t)(w[80] | ((uint16_t)w[81] << 8)));
    const float dmin = ggml_fp16_to_fp32((uint16_t)(w[82] | ((uint16_t)w[83] << 8)));
    float acc = 0.0f;

    for (int half = 0; half < 2; half++) {
        const uint8_t *qs = w + 16 + half * 32;
        for (int pair = 0; pair < 4; pair++) {
            const int shift = pair * 2;
            const uint8_t sc0 = w[half * 8 + pair * 2 + 0];
            const uint8_t sc1 = w[half * 8 + pair * 2 + 1];
            const uint8_t *xb = xq8 + (half * 4 + pair) * 36;
            const float dx = ggml_fp16_to_fp32((uint16_t)(xb[0] | ((uint16_t)xb[1] << 8)));
            const int8_t *xv = (const int8_t *)(xb + 4);
            int dot0 = 0, dot1 = 0, sum0 = 0, sum1 = 0;
            for (int j = 0; j < 16; j++) {
                const int x0 = xv[j];
                const int x1 = xv[16 + j];
                dot0 += (int)((qs[j] >> shift) & 3) * x0;
                dot1 += (int)((qs[16 + j] >> shift) & 3) * x1;
                sum0 += x0;
                sum1 += x1;
            }
            acc += d * dx * ((float)(sc0 & 15) * dot0 + (float)(sc1 & 15) * dot1);
            acc -= dmin * dx * ((float)(sc0 >> 4) * sum0 + (float)(sc1 >> 4) * sum1);
        }
    }

    return acc;
}

static void cpu_gemm_q2_k_q8_1(float *y, const uint8_t *w, const uint8_t *xq8,
                               int rows, int cols, int tokens, size_t row_bytes, size_t x_row_bytes) {
    const int nb = cols / 256;
    for (int t = 0; t < tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float acc = 0.0f;
            const uint8_t *wr = w + (size_t)r * row_bytes;
            const uint8_t *xr = xq8 + (size_t)t * x_row_bytes;
            for (int b = 0; b < nb; b++) acc += cpu_dot_q2_k_q8_1_block(wr + b * 84, xr + b * 8 * 36);
            y[(size_t)t * rows + r] = acc;
        }
    }
}

static void cpu_gemm(float *y, const float *w, const float *x, int rows, int cols, int tokens) {
    for (int t = 0; t < tokens; t++) {
        for (int r = 0; r < rows; r++) {
            double acc = 0.0;
            for (int k = 0; k < cols; k++) {
                acc += (double)w[(size_t)r * cols + k] * (double)x[(size_t)t * cols + k];
            }
            y[(size_t)t * rows + r] = (float)acc;
        }
    }
}

static void metrics(const float *a, const float *b, size_t n, float *max_abs, float *rms, float *cosine) {
    double maxv = 0.0, ss = 0.0, dot = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < n; i++) {
        double da = (double)a[i];
        double db = (double)b[i];
        double d = fabs(da - db);
        if (d > maxv) maxv = d;
        ss += d * d;
        dot += da * db;
        aa += da * da;
        bb += db * db;
    }
    *max_abs = (float)maxv;
    *rms = (float)sqrt(ss / (double)(n ? n : 1));
    *cosine = (float)(dot / (sqrt(aa) * sqrt(bb) + 1.0e-30));
}

static const quant_type_info *find_type(const char *name) {
    for (size_t i = 0; i < sizeof(g_types) / sizeof(g_types[0]); i++) {
        if (strcmp(name, g_types[i].name) == 0) return &g_types[i];
    }
    return NULL;
}

static uint32_t parse_modes(const char *s) {
    if (strcmp(s, "all") == 0) return MODE_ALL;
    if (strcmp(s, "dequant") == 0) return MODE_DEQUANT;
    if (strcmp(s, "bf16-wmma") == 0 || strcmp(s, "bf16") == 0) return MODE_BF16;
    if (strcmp(s, "int8-wmma") == 0 || strcmp(s, "int8") == 0) return MODE_INT8;
    if (strcmp(s, "fp8-wmma") == 0 || strcmp(s, "fp8") == 0) return MODE_FP8;
    if (strcmp(s, "quant") == 0 || strcmp(s, "q8_0-quant") == 0) return MODE_QUANT;
    if (strcmp(s, "q2fp8-wmma") == 0 || strcmp(s, "q2fp8") == 0) return MODE_Q2_FP8;
    if (strcmp(s, "q2llama") == 0 || strcmp(s, "q2-q8_1") == 0) return MODE_Q2_LLAMA;
    fprintf(stderr, "unknown --mode '%s'\n", s);
    exit(2);
}

static void parse_tokens(options *opt, const char *s) {
    opt->ntokens = 0;
    const char *p = s;
    while (*p) {
        if (opt->ntokens >= 16) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0) {
            fprintf(stderr, "bad --tokens '%s'\n", s);
            exit(2);
        }
        opt->tokens[opt->ntokens++] = (int)v;
        p = (*end == ',') ? end + 1 : end;
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [--verify] [--bench] [--type all|q4_0|...] [--mode all|dequant|quant|bf16-wmma|int8-wmma|fp8-wmma|q2fp8-wmma|q2llama]\n"
            "          [--rows N] [--cols N] [--tokens 1,16,64] [--iters N] [--device N] [--verbose]\n",
            argv0);
}

static options parse_args(int argc, char **argv) {
    options opt;
    memset(&opt, 0, sizeof(opt));
    opt.rows = 4096;
    opt.cols = 4096;
    opt.tokens[0] = 1;
    opt.tokens[1] = 16;
    opt.tokens[2] = 64;
    opt.ntokens = 3;
    opt.iters = 50;
    opt.verify = 0;
    opt.bench = 1;
    opt.modes = MODE_ALL;
    opt.device = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verify") == 0) opt.verify = 1;
        else if (strcmp(argv[i], "--bench") == 0) opt.bench = 1;
        else if (strcmp(argv[i], "--no-bench") == 0) opt.bench = 0;
        else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) opt.verbose++;
        else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) opt.rows = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc) opt.cols = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) opt.iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) opt.device = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) parse_tokens(&opt, argv[++i]);
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) opt.modes = parse_modes(argv[++i]);
        else if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            const char *name = argv[++i];
            if (strcmp(name, "all") == 0) opt.only_type = NULL;
            else {
                opt.only_type = find_type(name);
                if (!opt.only_type) {
                    fprintf(stderr, "unknown --type '%s'\n", name);
                    exit(2);
                }
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            usage(argv[0]);
            exit(2);
        }
    }
    if (!opt.verify && !opt.bench) opt.bench = 1;
    if (opt.cols <= 0 || opt.rows <= 0 || opt.iters <= 0) {
        fprintf(stderr, "rows, cols and iters must be positive\n");
        exit(2);
    }
    if (opt.cols % 256 != 0) {
        fprintf(stderr, "cols must be a multiple of 256 for the all-format benchmark\n");
        exit(2);
    }
    return opt;
}

static void run_one_type(const options *opt, const quant_type_info *qt,
                         hipFunction_t fn_deq, hipFunction_t fn_quant, hipFunction_t fn_bf16,
                         hipFunction_t fn_int8, hipFunction_t fn_fp8, hipFunction_t fn_q2fp8,
                         hipFunction_t fn_quant_q8_1, hipFunction_t fn_q2llama) {
    const int rows = opt->rows;
    const int cols = opt->cols;
    const size_t row_bytes = dequant_row_size(qt->type, cols);
    const size_t w_bytes = (size_t)rows * row_bytes;
    const size_t out_deq_bytes = (size_t)rows * cols * sizeof(float);

    uint8_t *h_w = (uint8_t *)malloc(w_bytes);
    float *h_ref = NULL;
    float *h_out = (float *)malloc(out_deq_bytes);
    if (!h_w || !h_out) {
        fprintf(stderr, "host allocation failed\n");
        exit(1);
    }
    fill_quant(h_w, qt, rows, cols);

    void *d_w = NULL, *d_out = NULL;
    CHECK_HIP(hipMalloc(&d_w, w_bytes));
    CHECK_HIP(hipMalloc(&d_out, out_deq_bytes));
    CHECK_HIP(hipMemcpy(d_w, h_w, w_bytes, hipMemcpyHostToDevice));

    int type = (int)qt->type, irows = rows, icols = cols, irow_bytes = (int)row_bytes;
    if (opt->modes & MODE_QUANT) {
        if (qt->type == GGML_TYPE_Q8_0) {
            const size_t in_bytes = (size_t)rows * cols * sizeof(float);
            float *h_qsrc = (float *)malloc(in_bytes);
            uint8_t *h_qref = (uint8_t *)malloc(w_bytes);
            uint8_t *h_qout = (uint8_t *)malloc(w_bytes);
            float *h_qref_f = NULL;
            float *h_qout_f = NULL;
            void *d_qsrc = NULL, *d_qout = NULL;
            if (!h_qsrc || !h_qref || !h_qout) {
                fprintf(stderr, "host quant allocation failed\n");
                exit(1);
            }
            fill_x(h_qsrc, rows, cols);
            CHECK_HIP(hipMalloc(&d_qsrc, in_bytes));
            CHECK_HIP(hipMalloc(&d_qout, w_bytes));
            CHECK_HIP(hipMemcpy(d_qsrc, h_qsrc, in_bytes, hipMemcpyHostToDevice));

            void *args[] = {&d_qsrc, &d_qout, &irows, &icols, &irow_bytes};
            unsigned gx = (unsigned)(((long)rows * (cols / 32) + 255) / 256);
            if (opt->verify) {
                CHECK_HIP(hipModuleLaunchKernel(fn_quant, gx, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
                CHECK_HIP(hipDeviceSynchronize());
                CHECK_HIP(hipMemcpy(h_qout, d_qout, w_bytes, hipMemcpyDeviceToHost));
                cpu_quant_q8_0_all(h_qsrc, h_qref, rows, cols, row_bytes);
                size_t mismatches = 0;
                for (size_t i = 0; i < w_bytes; i++) mismatches += h_qref[i] != h_qout[i];
                h_qref_f = (float *)malloc(out_deq_bytes);
                h_qout_f = (float *)malloc(out_deq_bytes);
                if (!h_qref_f || !h_qout_f) {
                    fprintf(stderr, "host quant dequant allocation failed\n");
                    exit(1);
                }
                cpu_dequant_all(h_qref, h_qref_f, qt, rows, cols);
                cpu_dequant_all(h_qout, h_qout_f, qt, rows, cols);
                float ma, rms, cosv;
                metrics(h_qref_f, h_qout_f, (size_t)rows * cols, &ma, &rms, &cosv);
                int pass = mismatches == 0 || ma <= 1.0e-5f;
                printf("VERIFY quant   %-5s packed_mismatch=%zu max_abs=%g rms=%g cosine=%.8f %s\n",
                       qt->name, mismatches, ma, rms, cosv, pass ? "PASS" : "FAIL");
                if (!pass) exit(1);
            }
            if (opt->bench) {
                float ms = elapsed_kernel_ms(fn_quant, gx, 1, 256, args, opt->iters);
                double total_gb = ((double)in_bytes + (double)w_bytes) / 1.0e9;
                double gelem = (double)rows * (double)cols / 1.0e9;
                printf("BENCH quant   %-5s rows=%d cols=%d ms=%.4f GB/s=%.2f Gelem/s=%.2f\n",
                       qt->name, rows, cols, ms, total_gb / (ms * 1.0e-3), gelem / (ms * 1.0e-3));
            }
            if (h_qref_f) free(h_qref_f);
            if (h_qout_f) free(h_qout_f);
            CHECK_HIP(hipFree(d_qsrc));
            CHECK_HIP(hipFree(d_qout));
            free(h_qsrc);
            free(h_qref);
            free(h_qout);
        } else if (opt->only_type && opt->modes == MODE_QUANT) {
            fprintf(stderr, "quant mode currently supports q8_0 only; skipping %s\n", qt->name);
        }
    }

    if (opt->modes & MODE_DEQUANT) {
        void *args[] = {&d_w, &d_out, &type, &irows, &icols, &irow_bytes};
        unsigned gx = (unsigned)(((long)rows * cols + 255) / 256);
        if (opt->verify) {
            CHECK_HIP(hipModuleLaunchKernel(fn_deq, gx, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_HIP(hipMemcpy(h_out, d_out, out_deq_bytes, hipMemcpyDeviceToHost));
            h_ref = (float *)malloc(out_deq_bytes);
            if (!h_ref) {
                fprintf(stderr, "host reference allocation failed\n");
                exit(1);
            }
            cpu_dequant_all(h_w, h_ref, qt, rows, cols);
            float ma, rms, cosv;
            metrics(h_ref, h_out, (size_t)rows * cols, &ma, &rms, &cosv);
            printf("VERIFY dequant %-5s max_abs=%g rms=%g cosine=%.8f %s\n",
                   qt->name, ma, rms, cosv, ma <= 1.0e-6f ? "PASS" : "FAIL");
            if (ma > 1.0e-6f) exit(1);
        }
        if (opt->bench) {
            float ms = elapsed_kernel_ms(fn_deq, gx, 1, 256, args, opt->iters);
            double total_gb = ((double)w_bytes + (double)out_deq_bytes) / 1.0e9;
            double gelem = (double)rows * (double)cols / 1.0e9;
            printf("BENCH dequant %-5s rows=%d cols=%d ms=%.4f GB/s=%.2f Gelem/s=%.2f\n",
                   qt->name, rows, cols, ms, total_gb / (ms * 1.0e-3), gelem / (ms * 1.0e-3));
        }
    }

    if ((opt->modes & (MODE_BF16 | MODE_INT8 | MODE_FP8 | MODE_Q2_FP8 | MODE_Q2_LLAMA)) == 0) {
        if (h_ref) free(h_ref);
        CHECK_HIP(hipFree(d_w));
        CHECK_HIP(hipFree(d_out));
        free(h_w);
        free(h_out);
        return;
    }

    for (int ti = 0; ti < opt->ntokens; ti++) {
        const int tokens = opt->tokens[ti];
        const size_t x_bytes = (size_t)tokens * cols * sizeof(float);
        const size_t y_bytes = (size_t)tokens * rows * sizeof(float);
        float *h_x = (float *)malloc(x_bytes);
        float *h_y = (float *)malloc(y_bytes);
        float *h_yref = NULL;
        void *d_x = NULL, *d_y = NULL;
        if (!h_x || !h_y) {
            fprintf(stderr, "host GEMM allocation failed\n");
            exit(1);
        }
        fill_x(h_x, tokens, cols);
        CHECK_HIP(hipMalloc(&d_x, x_bytes));
        CHECK_HIP(hipMalloc(&d_y, y_bytes));
        CHECK_HIP(hipMemcpy(d_x, h_x, x_bytes, hipMemcpyHostToDevice));

        if (opt->verify && !h_ref) {
            h_ref = (float *)malloc(out_deq_bytes);
            if (!h_ref) {
                fprintf(stderr, "host reference allocation failed\n");
                exit(1);
            }
            cpu_dequant_all(h_w, h_ref, qt, rows, cols);
        }
        if (opt->verify) {
            h_yref = (float *)malloc(y_bytes);
            if (!h_yref) {
                fprintf(stderr, "host GEMM reference allocation failed\n");
                exit(1);
            }
            double t0 = now_sec();
            cpu_gemm(h_yref, h_ref, h_x, rows, cols, tokens);
            if (opt->verbose) fprintf(stderr, "cpu ref %-5s tok=%d %.3fs\n", qt->name, tokens, now_sec() - t0);
        }

#define RUN_GEMM_MODE(label, bit, fn, strict_cos) do { \
        if ((opt->modes & (bit)) && (fn)) { \
            void *args[] = {&d_y, &d_w, &d_x, &type, &irows, &icols, &(int){tokens}, &irow_bytes}; \
            unsigned gx = (unsigned)((rows + 127) / 128); \
            unsigned gy = (unsigned)((tokens + 127) / 128); \
            if (opt->verify) { \
                CHECK_HIP(hipModuleLaunchKernel((fn), gx, gy, 1, 256, 1, 1, 0, NULL, args, NULL)); \
                CHECK_HIP(hipDeviceSynchronize()); \
                CHECK_HIP(hipMemcpy(h_y, d_y, y_bytes, hipMemcpyDeviceToHost)); \
                float ma, rm, co; metrics(h_yref, h_y, (size_t)tokens * rows, &ma, &rm, &co); \
                int pass = (co >= (strict_cos)) || ((strict_cos) < 0.999f && rm <= 1.0e-2f); \
                printf("VERIFY %-12s %-5s tok=%d max_abs=%g rms=%g cosine=%.8f %s\n", \
                       (label), qt->name, tokens, ma, rm, co, pass ? "PASS" : "FAIL"); \
                if (!pass) exit(1); \
            } \
            if (opt->bench) { \
                float ms = elapsed_kernel_ms((fn), gx, gy, 256, args, opt->iters); \
                double ops = 2.0 * (double)tokens * rows * cols; \
                printf("BENCH %-12s %-5s M=%d N=%d K=%d ms=%.4f %.2f TFLOP/s\n", \
                       (label), qt->name, tokens, rows, cols, ms, ops / (ms * 1.0e-3) / 1.0e12); \
            } \
        } \
    } while (0)

        RUN_GEMM_MODE("bf16-wmma", MODE_BF16, fn_bf16, 0.999f);
        RUN_GEMM_MODE("int8-wmma", MODE_INT8, fn_int8, 0.90f);
        RUN_GEMM_MODE("fp8-wmma", MODE_FP8, fn_fp8, 0.90f);
#undef RUN_GEMM_MODE

        if ((opt->modes & MODE_Q2_LLAMA) && qt->type == GGML_TYPE_Q2_K && fn_q2llama && fn_quant_q8_1) {
            const size_t xq_row_bytes = (size_t)(cols / 32) * 36;
            const size_t xq_bytes = (size_t)tokens * xq_row_bytes;
            uint8_t *h_xq = NULL;
            void *d_xq = NULL;
            CHECK_HIP(hipMalloc(&d_xq, xq_bytes));
            int ix_row_bytes = (int)xq_row_bytes;
            void *qargs[] = {&d_x, &d_xq, &(int){tokens}, &icols, &ix_row_bytes};
            unsigned qgx = (unsigned)(((long)tokens * (cols / 32) + 255) / 256);
            CHECK_HIP(hipModuleLaunchKernel(fn_quant_q8_1, qgx, 1, 1, 256, 1, 1, 0, NULL, qargs, NULL));
            CHECK_HIP(hipDeviceSynchronize());
            if (opt->verify) {
                h_xq = (uint8_t *)malloc(xq_bytes);
                if (!h_xq) {
                    fprintf(stderr, "host q8_1 allocation failed\n");
                    exit(1);
                }
                CHECK_HIP(hipMemcpy(h_xq, d_xq, xq_bytes, hipMemcpyDeviceToHost));
                cpu_gemm_q2_k_q8_1(h_yref, h_w, h_xq, rows, cols, tokens, row_bytes, xq_row_bytes);
            }
            void *args[] = {&d_y, &d_w, &d_xq, &irows, &icols, &(int){tokens}, &irow_bytes, &ix_row_bytes};
            unsigned gx = (unsigned)(((long)tokens * rows + 255) / 256);
            if (opt->verify) {
                CHECK_HIP(hipModuleLaunchKernel(fn_q2llama, gx, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
                CHECK_HIP(hipDeviceSynchronize());
                CHECK_HIP(hipMemcpy(h_y, d_y, y_bytes, hipMemcpyDeviceToHost));
                float ma, rm, co;
                metrics(h_yref, h_y, (size_t)tokens * rows, &ma, &rm, &co);
                int pass = ma <= 1.0e-3f || co >= 0.999f;
                printf("VERIFY %-12s %-5s tok=%d max_abs=%g rms=%g cosine=%.8f %s\n",
                       "q2llama", qt->name, tokens, ma, rm, co, pass ? "PASS" : "FAIL");
                if (!pass) exit(1);
            }
            if (opt->bench) {
                float msq = elapsed_kernel_ms(fn_quant_q8_1, qgx, 1, 256, qargs, opt->iters);
                float msg = elapsed_kernel_ms(fn_q2llama, gx, 1, 256, args, opt->iters);
                double ops = 2.0 * (double)tokens * rows * cols;
                printf("BENCH %-12s %-5s M=%d N=%d K=%d ms=%.4f quant_ms=%.4f gemm_ms=%.4f %.2f TFLOP/s\n",
                       "q2llama", qt->name, tokens, rows, cols, msq + msg, msq, msg,
                       ops / ((msq + msg) * 1.0e-3) / 1.0e12);
            }
            if (h_xq) free(h_xq);
            CHECK_HIP(hipFree(d_xq));
        } else if ((opt->modes & MODE_Q2_LLAMA) && opt->only_type && qt->type != GGML_TYPE_Q2_K) {
            fprintf(stderr, "q2llama mode currently supports q2_k only; skipping %s\n", qt->name);
        }

        if ((opt->modes & MODE_Q2_FP8) && qt->type == GGML_TYPE_Q2_K && fn_q2fp8) {
            void *args[] = {&d_y, &d_w, &d_x, &irows, &icols, &(int){tokens}, &irow_bytes};
            unsigned gx = (unsigned)((rows + 127) / 128);
            unsigned gy = (unsigned)((tokens + 127) / 128);
            if (opt->verify) {
                CHECK_HIP(hipModuleLaunchKernel(fn_q2fp8, gx, gy, 1, 256, 1, 1, 0, NULL, args, NULL));
                CHECK_HIP(hipDeviceSynchronize());
                CHECK_HIP(hipMemcpy(h_y, d_y, y_bytes, hipMemcpyDeviceToHost));
                float ma, rm, co;
                metrics(h_yref, h_y, (size_t)tokens * rows, &ma, &rm, &co);
                int pass = co >= 0.90f || rm <= 1.0e-2f;
                printf("VERIFY %-12s %-5s tok=%d max_abs=%g rms=%g cosine=%.8f %s\n",
                       "q2fp8-wmma", qt->name, tokens, ma, rm, co, pass ? "PASS" : "FAIL");
                if (!pass) exit(1);
            }
            if (opt->bench) {
                float ms = elapsed_kernel_ms(fn_q2fp8, gx, gy, 256, args, opt->iters);
                double ops = 2.0 * (double)tokens * rows * cols;
                printf("BENCH %-12s %-5s M=%d N=%d K=%d ms=%.4f %.2f TFLOP/s\n",
                       "q2fp8-wmma", qt->name, tokens, rows, cols, ms, ops / (ms * 1.0e-3) / 1.0e12);
            }
        } else if ((opt->modes & MODE_Q2_FP8) && opt->only_type && qt->type != GGML_TYPE_Q2_K) {
            fprintf(stderr, "q2fp8-wmma mode currently supports q2_k only; skipping %s\n", qt->name);
        }

        if (h_yref) free(h_yref);
        CHECK_HIP(hipFree(d_x));
        CHECK_HIP(hipFree(d_y));
        free(h_x);
        free(h_y);
    }

    if (h_ref) free(h_ref);
    CHECK_HIP(hipFree(d_w));
    CHECK_HIP(hipFree(d_out));
    free(h_w);
    free(h_out);
}

int main(int argc, char **argv) {
    options opt = parse_args(argc, argv);
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "failed to initialize ROCm/HIP runtime\n");
        return 1;
    }
    CHECK_HIP(hipSetDevice(opt.device));
    const char *arch = rocewGetRDNA4ArchString(opt.device);
    fprintf(stderr, "rdna4/dequant: device=%d arch=%s rows=%d cols=%d iters=%d\n",
            opt.device, arch ? arch : "?", opt.rows, opt.cols, opt.iters);

    char *full_kernel_src = build_kernel_source();
    hipModule_t module = NULL;
    if (hip_compile_kernels(&module, opt.device, full_kernel_src, "rdna4_dequant.hip", opt.verbose, "rdna4_dequant") < 0) {
        free(full_kernel_src);
        return 1;
    }
    free(full_kernel_src);

    hipFunction_t fn_deq = NULL, fn_quant = NULL, fn_bf16 = NULL, fn_int8 = NULL, fn_fp8 = NULL, fn_q2fp8 = NULL;
    hipFunction_t fn_quant_q8_1 = NULL, fn_q2llama = NULL;
    CHECK_HIP(hipModuleGetFunction(&fn_deq, module, "dequant_f32_kernel"));
    CHECK_HIP(hipModuleGetFunction(&fn_quant, module, "quant_q8_0_kernel"));
    CHECK_HIP(hipModuleGetFunction(&fn_quant_q8_1, module, "quant_q8_1_kernel"));
    CHECK_HIP(hipModuleGetFunction(&fn_q2llama, module, "gemm_q2k_q8_1_llama"));
    if (hipModuleGetFunction(&fn_bf16, module, "gemm_dequant_bf16_wmma") != hipSuccess) fn_bf16 = NULL;
    if (hipModuleGetFunction(&fn_int8, module, "gemm_dequant_int8_wmma") != hipSuccess) fn_int8 = NULL;
    if (hipModuleGetFunction(&fn_fp8, module, "gemm_dequant_fp8_wmma") != hipSuccess) fn_fp8 = NULL;
    if (hipModuleGetFunction(&fn_q2fp8, module, "gemm_q2k_fp8_wmma") != hipSuccess) fn_q2fp8 = NULL;
    if ((opt.modes & (MODE_BF16 | MODE_INT8 | MODE_FP8 | MODE_Q2_FP8 | MODE_Q2_LLAMA)) && (!fn_bf16 || !fn_int8 || !fn_fp8 || !fn_q2fp8 || !fn_q2llama)) {
        fprintf(stderr, "warning: one or more WMMA kernels are unavailable for this arch\n");
    }

    printf("# type/mode benchmark on %s\n", arch ? arch : "?");
    for (size_t i = 0; i < sizeof(g_types) / sizeof(g_types[0]); i++) {
        if (opt.only_type && opt.only_type != &g_types[i]) continue;
        run_one_type(&opt, &g_types[i], fn_deq, fn_quant, fn_bf16, fn_int8, fn_fp8, fn_q2fp8, fn_quant_q8_1, fn_q2llama);
    }
    return 0;
}
