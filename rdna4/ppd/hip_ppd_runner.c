/*
 * hip_ppd_runner.c - HIP/ROCm Pixel-Perfect-Depth via HIPRTC-compiled kernels
 *
 * Pipeline: DA2 semantic encoder -> DiT diffusion (4 Euler steps) -> depth
 * Compiles with plain gcc (no hipcc). Uses rocew for dynamic HIP/HIPRTC loading.
 * Targets RDNA4 (gfx1200/gfx1201).
 */

#define _GNU_SOURCE
#define PTH_LOADER_IMPLEMENTATION
#include "../../common/pth_loader.h"
#include "hip_ppd_runner.h"
#include "../rocew.h"
#include "../hip_kernels_common.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ======================================================================== */
/* PPD-specific HIP kernel source (compiled at runtime via HIPRTC)          */
/* Shared kernels are in hip_kernels_common.h (hip_kernels_common_src)      */
/* ======================================================================== */

static const char hip_ppd_specific_kernels[] =
"\n"
"/* ---- qk_norm_f32: per-head layernorm on strided QKV buffer ---- */\n"
"/* One thread per (token, head). Normalizes head_dim elements in-place. */\n"
"__global__ void qk_norm_f32(float *qkv, const float *w, const float *b,\n"
"                              int n_tok, int n_heads, int head_dim,\n"
"                              int stride, float eps) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * n_heads;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / n_heads;\n"
"    int h = idx % n_heads;\n"
"    float *base = qkv + tok * stride + h * head_dim;\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) sum += base[i];\n"
"    float mean = sum / (float)head_dim;\n"
"    float var_sum = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) { float d = base[i] - mean; var_sum += d*d; }\n"
"    float inv = rsqrtf(var_sum / (float)head_dim + eps);\n"
"    for (int i = 0; i < head_dim; i++)\n"
"        base[i] = (base[i] - mean) * inv * w[i] + b[i];\n"
"}\n"
"\n"
"/* ---- scalar_attn_f32: scalar attention (no MMA), one block per (head, query) ---- */\n"
"__global__ void scalar_attn_f32(float *out, const float *qkv,\n"
"                                 const float *K_t, const float *V_t,\n"
"                                 int n_tok, int dim, int n_heads, int head_dim,\n"
"                                 float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qi = blockIdx.y;\n"
"    if (h >= n_heads || qi >= n_tok) return;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int dim3 = 3 * dim;\n"
"    const float *q = qkv + qi * dim3 + h * head_dim;\n"
"    const float *kt_h = K_t + h * n_tok * head_dim;\n"
"    const float *vt_h = V_t + h * n_tok * head_dim;\n"
"    extern __shared__ float sdata[];\n"
"    float *scores = sdata;\n"            /* [n_tok] */
"    float *rbuf = sdata + n_tok;\n"      /* [nt] for reduction */
"\n"
"    /* Phase 1: Q @ K^T */\n"
"    for (int ki = tid; ki < n_tok; ki += nt) {\n"
"        const float *k = kt_h + ki * head_dim;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) dot += q[d] * k[d];\n"
"        scores[ki] = dot * scale;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 2: Softmax -- find max */\n"
"    float lmax = -1e30f;\n"
"    for (int ki = tid; ki < n_tok; ki += nt)\n"
"        lmax = fmaxf(lmax, scores[ki]);\n"
"    rbuf[tid] = lmax;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) rbuf[tid] = fmaxf(rbuf[tid], rbuf[tid+r]);\n"
"        __syncthreads();\n"
"    }\n"
"    float mx = rbuf[0];\n"
"    __syncthreads();\n"
"\n"
"    /* Exp + sum */\n"
"    float lsum = 0.0f;\n"
"    for (int ki = tid; ki < n_tok; ki += nt) {\n"
"        scores[ki] = expf(scores[ki] - mx);\n"
"        lsum += scores[ki];\n"
"    }\n"
"    rbuf[tid] = lsum;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) rbuf[tid] += rbuf[tid+r];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv_sum = (rbuf[0] > 0) ? 1.0f / rbuf[0] : 0.0f;\n"
"    for (int ki = tid; ki < n_tok; ki += nt)\n"
"        scores[ki] *= inv_sum;\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 3: Weighted sum of V */\n"
"    for (int d = tid; d < head_dim; d += nt) {\n"
"        float acc = 0.0f;\n"
"        for (int ki = 0; ki < n_tok; ki++)\n"
"            acc += scores[ki] * vt_h[ki * head_dim + d];\n"
"        out[qi * dim + h * head_dim + d] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* ---- kv_transpose_f16: transpose QKV and store K,V as FP16 ---- */\n"
"/* Halves global memory bandwidth for K/V during attention         */\n"
"__global__ void kv_transpose_f16(unsigned short *K_t, unsigned short *V_t,\n"
"                                   const float *qkv,\n"
"                                   int n_tok, int dim, int n_heads, int head_dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * dim;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / dim;\n"
"    int hd_idx = idx % dim;\n"
"    int h = hd_idx / head_dim;\n"
"    int d = hd_idx % head_dim;\n"
"    int dim3 = 3 * dim;\n"
"    int dst_idx = h * n_tok * head_dim + tok * head_dim + d;\n"
"    float kv = qkv[tok * dim3 + dim + hd_idx];\n"
"    float vv = qkv[tok * dim3 + 2*dim + hd_idx];\n"
"    __half kh_val = __float2half(kv);\n"
"    __half vh_val = __float2half(vv);\n"
"    K_t[dst_idx] = *((unsigned short*)&kh_val);\n"
"    V_t[dst_idx] = *((unsigned short*)&vh_val);\n"
"}\n"
"\n"
"/* ---- flash_attn_f16kv_f32: flash attention with FP16 global K/V    ---- */\n"
"/* K_t and V_t are FP16 (unsigned short) in global memory.              */\n"
"/* Grid: (n_heads, ceil(n_tok/FA_BQ)), blockDim=(FA_BQ=64)               */\n"
"#ifndef FA_BQ\n"
"#define FA_BQ       64\n"
"#define FA_BKV      16\n"
"#define FA_HEAD_DIM 64\n"
"#endif\n"
"__global__ void flash_attn_f16kv_f32(float *out, const float *qkv,\n"
"                            const unsigned short *K_t,\n"
"                            const unsigned short *V_t,\n"
"                            int n_tok, int dim, int n_heads,\n"
"                            int head_dim, float scale) {\n"
"    int h  = blockIdx.x;\n"
"    int qi = blockIdx.y * FA_BQ + threadIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int dim3 = 3 * dim;\n"
"    const float         *qt   = qkv + (qi < n_tok ? qi : 0) * dim3 + h * head_dim;\n"
"    const unsigned short *kt_h = K_t + (size_t)h * n_tok * head_dim;\n"
"    const unsigned short *vt_h = V_t + (size_t)h * n_tok * head_dim;\n"
"    extern __shared__ unsigned short smem_u16[];\n"
"    unsigned short *smK = smem_u16;\n"
"    unsigned short *smV = smem_u16 + FA_BKV * FA_HEAD_DIM;\n"
"    float q_reg[FA_HEAD_DIM];\n"
"#pragma unroll\n"
"    for (int d = 0; d < FA_HEAD_DIM; d++)\n"
"        q_reg[d] = (qi < n_tok) ? qt[d] : 0.0f;\n"
"    float m_i = -1e30f, l_i = 0.0f;\n"
"    float O_i[FA_HEAD_DIM];\n"
"#pragma unroll\n"
"    for (int d = 0; d < FA_HEAD_DIM; d++) O_i[d] = 0.0f;\n"
"    int tid = threadIdx.x;\n"
"    int kv_tiles = (n_tok + FA_BKV - 1) / FA_BKV;\n"
"    for (int tile = 0; tile < kv_tiles; tile++) {\n"
"        int kv = tile * FA_BKV;\n"
"        for (int idx = tid; idx < FA_BKV * FA_HEAD_DIM; idx += FA_BQ) {\n"
"            int kj = idx / FA_HEAD_DIM, d = idx % FA_HEAD_DIM;\n"
"            int kv_tok = kv + kj;\n"
"            smK[idx] = (kv_tok < n_tok) ? kt_h[(size_t)kv_tok * FA_HEAD_DIM + d] : 0;\n"
"            smV[idx] = (kv_tok < n_tok) ? vt_h[(size_t)kv_tok * FA_HEAD_DIM + d] : 0;\n"
"        }\n"
"        __syncthreads();\n"
"        float sc[FA_BKV];\n"
"#pragma unroll\n"
"        for (int kj = 0; kj < FA_BKV; kj++) sc[kj] = 0.0f;\n"
"#pragma unroll\n"
"        for (int d = 0; d < FA_HEAD_DIM; d++) {\n"
"            float qd = q_reg[d];\n"
"#pragma unroll\n"
"            for (int kj = 0; kj < FA_BKV; kj++) {\n"
"                sc[kj] += qd * half_to_float(smK[kj * FA_HEAD_DIM + d]);\n"
"            }\n"
"        }\n"
"        float mx_tile = -1e30f;\n"
"#pragma unroll\n"
"        for (int kj = 0; kj < FA_BKV; kj++) {\n"
"            sc[kj] = (kv + kj < n_tok) ? sc[kj] * scale : -1e30f;\n"
"            if (sc[kj] > mx_tile) mx_tile = sc[kj];\n"
"        }\n"
"        float mn_i  = fmaxf(m_i, mx_tile);\n"
"        float alpha = expf(m_i - mn_i);\n"
"        l_i *= alpha;\n"
"#pragma unroll\n"
"        for (int d = 0; d < FA_HEAD_DIM; d++) O_i[d] *= alpha;\n"
"        m_i = mn_i;\n"
"        float ej[FA_BKV];\n"
"#pragma unroll\n"
"        for (int kj = 0; kj < FA_BKV; kj++) {\n"
"            ej[kj] = (kv + kj < n_tok) ? expf(sc[kj] - m_i) : 0.0f;\n"
"            l_i += ej[kj];\n"
"        }\n"
"#pragma unroll\n"
"        for (int kj = 0; kj < FA_BKV; kj++) {\n"
"            float e = ej[kj];\n"
"#pragma unroll\n"
"            for (int d = 0; d < FA_HEAD_DIM; d++) {\n"
"                O_i[d] += e * half_to_float(smV[kj * FA_HEAD_DIM + d]);\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (qi < n_tok) {\n"
"        float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;\n"
"        float *out_qi = out + (size_t)qi * dim + h * head_dim;\n"
"#pragma unroll\n"
"        for (int d = 0; d < FA_HEAD_DIM; d++)\n"
"            out_qi[d] = O_i[d] * inv_l;\n"
"    }\n"
"}\n"
"\n"
"/* ---- flash_attn_warp_f16kv_f32: warp-cooperative flash attention     ---- */\n"
"/* 1 warp (32 threads on RDNA4 wavefront32) handles 1 query token.          */\n"
"/* Each thread holds q[lane*2], q[lane*2+1] and O[lane*2], O[lane*2+1].     */\n"
"/* Score = warp_reduce via __shfl_xor (no sync mask on HIP)                  */\n"
"/* Grid: (n_heads, ceil(n_tok/FA_WARPS)), Block: 32*FA_WARPS threads         */\n"
"#define FA_WARPS  8   /* warps per block = query-tokens per block */\n"
"__global__ void flash_attn_warp_f16kv_f32(\n"
"    float *out, const float *qkv,\n"
"    const unsigned short *K_t, const unsigned short *V_t,\n"
"    int n_tok, int dim, int n_heads, int head_dim, float scale) {\n"
"    int tid     = threadIdx.x;\n"
"    int warp_id = tid / 32;\n"
"    int lane    = tid % 32;\n"
"    int h  = blockIdx.x;\n"
"    int qi = blockIdx.y * FA_WARPS + warp_id;\n"
"    if (h >= n_heads) return;\n"
"    int dim3 = 3 * dim;\n"
"    const float         *qt   = qkv + (qi < n_tok ? qi : 0) * dim3 + h * FA_HEAD_DIM;\n"
"    const unsigned short *kt_h = K_t + (size_t)h * n_tok * FA_HEAD_DIM;\n"
"    const unsigned short *vt_h = V_t + (size_t)h * n_tok * FA_HEAD_DIM;\n"
"    extern __shared__ float smw[];\n"
"    float *smK = smw;\n"
"    float *smV = smw + FA_BKV * FA_HEAD_DIM;\n"
"    float q0 = (qi < n_tok) ? qt[lane * 2]     : 0.0f;\n"
"    float q1 = (qi < n_tok) ? qt[lane * 2 + 1] : 0.0f;\n"
"    float m_i = -1e30f, l_i = 0.0f;\n"
"    float O0 = 0.0f, O1 = 0.0f;\n"
"    int kv_tiles = (n_tok + FA_BKV - 1) / FA_BKV;\n"
"    for (int tile = 0; tile < kv_tiles; tile++) {\n"
"        int kv = tile * FA_BKV;\n"
"        for (int idx = tid; idx < FA_BKV * FA_HEAD_DIM; idx += 32 * FA_WARPS) {\n"
"            int kj = idx / FA_HEAD_DIM, d = idx % FA_HEAD_DIM;\n"
"            int kv_tok = kv + kj;\n"
"            unsigned short kh = (kv_tok < n_tok) ? kt_h[(size_t)kv_tok * FA_HEAD_DIM + d] : 0;\n"
"            unsigned short vh = (kv_tok < n_tok) ? vt_h[(size_t)kv_tok * FA_HEAD_DIM + d] : 0;\n"
"            smK[idx] = half_to_float(kh);\n"
"            smV[idx] = half_to_float(vh);\n"
"        }\n"
"        __syncthreads();\n"
"        float sc[FA_BKV];\n"
"#pragma unroll\n"
"        for (int kj = 0; kj < FA_BKV; kj++) {\n"
"            float p = q0 * smK[kj * FA_HEAD_DIM + lane * 2]\n"
"                    + q1 * smK[kj * FA_HEAD_DIM + lane * 2 + 1];\n"
"            p += __shfl_xor(p, 16);\n"
"            p += __shfl_xor(p,  8);\n"
"            p += __shfl_xor(p,  4);\n"
"            p += __shfl_xor(p,  2);\n"
"            p += __shfl_xor(p,  1);\n"
"            sc[kj] = p;\n"
"        }\n"
"        float mx_tile = -1e30f;\n"
"#pragma unroll\n"
"        for (int kj = 0; kj < FA_BKV; kj++) {\n"
"            sc[kj] = (kv + kj < n_tok) ? sc[kj] * scale : -1e30f;\n"
"            if (sc[kj] > mx_tile) mx_tile = sc[kj];\n"
"        }\n"
"        float mn_i  = fmaxf(m_i, mx_tile);\n"
"        float alpha = expf(m_i - mn_i);\n"
"        l_i *= alpha;\n"
"        O0  *= alpha;\n"
"        O1  *= alpha;\n"
"        m_i  = mn_i;\n"
"        float ej[FA_BKV];\n"
"#pragma unroll\n"
"        for (int kj = 0; kj < FA_BKV; kj++) {\n"
"            ej[kj] = (kv + kj < n_tok) ? expf(sc[kj] - m_i) : 0.0f;\n"
"            l_i += ej[kj];\n"
"        }\n"
"#pragma unroll\n"
"        for (int kj = 0; kj < FA_BKV; kj++) {\n"
"            float e = ej[kj];\n"
"            O0 += e * smV[kj * FA_HEAD_DIM + lane * 2];\n"
"            O1 += e * smV[kj * FA_HEAD_DIM + lane * 2 + 1];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (qi < n_tok) {\n"
"        float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;\n"
"        float *out_qi = out + (size_t)qi * dim + h * FA_HEAD_DIM;\n"
"        out_qi[lane * 2]     = O0 * inv_l;\n"
"        out_qi[lane * 2 + 1] = O1 * inv_l;\n"
"    }\n"
"}\n"
"\n"
"/* ======== DiT-specific kernels ======== */\n"
"\n"
"/* ---- adaLN_modulate: y = norm(x) * (1 + scale) + shift ---- */\n"
"__global__ void adaln_modulate_f32(float *dst, const float *src, const float *w,\n"
"                                    const float *b, const float *shift,\n"
"                                    const float *scale, int dim, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int tok = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    const float *x = src + tok * dim;\n"
"    float *y = dst + tok * dim;\n"
"    const float *sh = shift;\n"
"    const float *sc = scale;\n"
"    float s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) s += x[i];\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)dim;\n"
"    __syncthreads();\n"
"    s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) { float d = x[i] - mean; s += d*d; }\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)dim + eps);\n"
"    for (int i = tid; i < dim; i += nt)\n"
"        y[i] = ((x[i] - mean) * inv * w[i] + b[i]) * (1.0f + sc[i]) + sh[i];\n"
"}\n"
"\n"
"/* ---- gate_residual_add: dst += gate * src ---- */\n"
"__global__ void gate_residual_add_f32(float *dst, const float *src,\n"
"                                       const float *gate, int dim, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        dst[i] += gate[i % dim] * src[i];\n"
"    }\n"
"}\n"
"\n"
"/* ---- rope_2d_dit_f32: 2D RoPE for DiT (separate Q and K buffers) ---- */\n"
"__global__ void rope_2d_dit_f32(float *vec, const int *pos_y, const int *pos_x,\n"
"                                 int n_tok, int n_heads, int head_dim,\n"
"                                 int stride, float freq_base) {\n"
"    int t = blockIdx.x;\n"
"    if (t >= n_tok) return;\n"
"    int tid = threadIdx.x;\n"
"    int half = head_dim / 2;\n"
"    int quarter = half / 2;\n"
"    int total = n_heads * quarter;\n"
"    if (tid >= total) return;\n"
"    int h = tid / quarter;\n"
"    int j = tid % quarter;\n"
"    float py = (float)pos_y[t];\n"
"    float px = (float)pos_x[t];\n"
"    float *v = vec + t * stride + h * head_dim;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2*j) / (float)half);\n"
"    float ty = py * freq;\n"
"    float cy = cosf(ty), sy = sinf(ty);\n"
"    float v0y = v[j], v1y = v[j + quarter];\n"
"    v[j]           = v0y * cy - v1y * sy;\n"
"    v[j + quarter] = v0y * sy + v1y * cy;\n"
"    float tx = px * freq;\n"
"    float cx = cosf(tx), sx = sinf(tx);\n"
"    float v0x = v[half + j], v1x = v[half + j + quarter];\n"
"    v[half + j]           = v0x * cx - v1x * sx;\n"
"    v[half + j + quarter] = v0x * sx + v1x * cx;\n"
"}\n"
"\n"
"/* ---- unpatchify: reshape [n_tok, patch_size^2] -> [1, H, W] ---- */\n"
"__global__ void unpatchify_f32(float *dst, const float *src,\n"
"                                int gH, int gW, int ps) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int H = gH * ps, W = gW * ps;\n"
"    if (idx >= H * W) return;\n"
"    int oh = idx / W, ow = idx % W;\n"
"    int ph = oh / ps, pw = ow / ps;\n"
"    int kh = oh % ps, kw = ow % ps;\n"
"    int tok = ph * gW + pw;\n"
"    dst[idx] = src[tok * ps * ps + kh * ps + kw];\n"
"}\n"
"\n"
"/* ---- concat_4ch: cat([latent, cond], dim=1) for [1,H,W]+[3,H,W] -> [4,H,W] ---- */\n"
"__global__ void concat_4ch_f32(float *dst, const float *latent, const float *cond,\n"
"                                int HW) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= 4 * HW) return;\n"
"    int ch = i / HW;\n"
"    int hw = i % HW;\n"
"    dst[i] = (ch == 0) ? latent[hw] : cond[(ch-1) * HW + hw];\n"
"}\n"
"\n"
"/* ---- dit_patch_embed_conv2d: Conv2d(4, dim, k=ps, s=ps) for DiT ---- */\n"
"__global__ void dit_patch_embed_conv2d(float *out, const float *img, const float *w,\n"
"                                        const float *bias, int gw, int dim, int ps,\n"
"                                        int img_h, int img_w) {\n"
"    int patch = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    for (int co = tid; co < dim; co += blockDim.x) {\n"
"        float sum = bias ? bias[co] : 0.0f;\n"
"        for (int ci = 0; ci < 4; ci++)\n"
"            for (int kh = 0; kh < ps; kh++)\n"
"                for (int kw = 0; kw < ps; kw++)\n"
"                    sum += w[((co*4+ci)*ps+kh)*ps+kw]\n"
"                         * img[ci * img_h * img_w + (py*ps+kh) * img_w + (px*ps+kw)];\n"
"        out[patch * dim + co] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ---- dit_im2col: extract [n_ch,H,W] patches -> [n_patches, n_ch*ps*ps] ---- */\n"
"#define DIT_IM2COL_BLK 8\n"
"__global__ void dit_im2col(float *out, const float *img,\n"
"                            int n_patches, int gw, int ps, int img_h, int img_w, int n_ch) {\n"
"    int patch_base = blockIdx.x * DIT_IM2COL_BLK;\n"
"    int ci  = blockIdx.y;\n"
"    int tid = threadIdx.x;\n"
"    int kh = tid / ps, kw = tid % ps;\n"
"    int ci_stride = img_h * img_w;\n"
"    int in_ps2 = n_ch * ps * ps;\n"
"    for (int i = 0; i < DIT_IM2COL_BLK; i++) {\n"
"        int patch = patch_base + i;\n"
"        if (patch >= n_patches) break;\n"
"        int py = patch / gw, px = patch % gw;\n"
"        int src = ci * ci_stride + (py*ps+kh) * img_w + (px*ps+kw);\n"
"        int dst = patch * in_ps2 + ci * ps*ps + tid;\n"
"        out[dst] = img[src];\n"
"    }\n"
"}\n"
"\n"
"/* ---- euler_step: latent update via velocity prediction ---- */\n"
"__global__ void euler_step_f32(float *latent, const float *pred,\n"
"                                float t_ratio, float s_ratio, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float xt = latent[i];\n"
"    float v = pred[i];\n"
"    float pred_x0 = xt - t_ratio * v;\n"
"    float pred_xT = xt + (1.0f - t_ratio) * v;\n"
"    latent[i] = (1.0f - s_ratio) * pred_x0 + s_ratio * pred_xT;\n"
"}\n"
"\n"
"/* ---- add_scalar_f32: dst[i] += scalar ---- */\n"
"__global__ void add_scalar_f32(float *dst, float scalar, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) dst[i] += scalar;\n"
"}\n"
"\n"
"/* ---- sub_scalar_chw_f32: dst[i] = src[i] - scalar ---- */\n"
"__global__ void sub_scalar_chw_f32(float *dst, const float *src, float scalar, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) dst[i] = src[i] - scalar;\n"
"}\n"
"\n"
"/* ---- pixel_shuffle_2x_f32: [gH*gW, 4*dim] -> [2*gH*2*gW, dim] depth-to-space ---- */\n"
"__global__ void pixel_shuffle_2x_f32(float *dst, const float *src,\n"
"                                       int gH, int gW, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int oH = gH * 2, oW = gW * 2;\n"
"    int total = oH * oW * dim;\n"
"    if (idx >= total) return;\n"
"    int tok_out = idx / dim;\n"
"    int d = idx % dim;\n"
"    int oh = tok_out / oW, ow = tok_out % oW;\n"
"    int ih = oh / 2, iw = ow / 2;\n"
"    int sub = (oh % 2) * 2 + (ow % 2);\n"
"    int tok_in = ih * gW + iw;\n"
"    dst[idx] = src[tok_in * 4 * dim + sub * dim + d];\n"
"}\n"
"\n"
"/* ---- concat_tokens_f32: cat(a, b) along dim ---- */\n"
"__global__ void concat_tokens_f32(float *dst, const float *a, const float *b,\n"
"                                    int n_tok, int dim_a, int dim_b) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total_dim = dim_a + dim_b;\n"
"    int total = n_tok * total_dim;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / total_dim;\n"
"    int d = idx % total_dim;\n"
"    dst[idx] = (d < dim_a) ? a[tok * dim_a + d] : b[tok * dim_b + (d - dim_a)];\n"
"}\n"
"\n"
"/* ---- scalar_gemm_f16_f32: simple scalar GEMM for debugging ---- */\n"
"__global__ void scalar_gemm_f16_f32(float *Y, const half_raw *W, const float *X,\n"
"                                     const float *bias,\n"
"                                     int n_out, int n_in, int n_tok) {\n"
"    int out = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int tok = blockIdx.y;\n"
"    if (out >= n_out || tok >= n_tok) return;\n"
"    float sum = 0.0f;\n"
"    for (int k = 0; k < n_in; k++) {\n"
"        half_raw h = W[out * n_in + k];\n"
"        sum += half_to_float(h) * X[tok * n_in + k];\n"
"    }\n"
"    if (bias) sum += bias[out];\n"
"    Y[tok * n_out + out] = sum;\n"
"}\n"
"\n"
"} /* extern C */\n"
;

/* ======================================================================== */
/* Error checking (aliases to shared macros from hip_runner_common.h)       */
/* ======================================================================== */

#define CHECK_HIP HIP_CHECK
#define CHECK_HIP_NULL HIP_CHECK_NULL

/* ======================================================================== */
/* Runner state                                                             */
/* ======================================================================== */

/* Per-block weights for DINOv2 ViT (DA2 semantic encoder) */
typedef struct {
    void *ln1_w, *ln1_b;
    void *attn_qkv_w, *attn_qkv_b;
    void *attn_out_w, *attn_out_b;
    void *ln2_w, *ln2_b;
    void *ffn_up_w, *ffn_up_b;
    void *ffn_down_w, *ffn_down_b;
    void *ls1_gamma, *ls2_gamma;  /* LayerScale (DINOv2) */
    int qkv_rows, qkv_cols;
    int out_rows, out_cols;
    int ffn_up_rows, ffn_up_cols;
    int ffn_down_rows, ffn_down_cols;
} sem_layer;

/* Per-block weights for DiT */
typedef struct {
    void *ln1_w, *ln1_b;             /* norm1 */
    void *ln2_w, *ln2_b;             /* norm2 */
    void *attn_qkv_w, *attn_qkv_b;  /* qkv (bias=True) */
    void *attn_q_norm_w, *attn_q_norm_b; /* q_norm */
    void *attn_k_norm_w, *attn_k_norm_b; /* k_norm */
    void *attn_out_w, *attn_out_b;   /* proj */
    void *mlp_fc1_w, *mlp_fc1_b;     /* fc1 */
    void *mlp_fc2_w, *mlp_fc2_b;     /* fc2 */
    void *adaln_w, *adaln_b;         /* adaLN_modulation: Linear(dim, 6*dim) */
    int qkv_rows, qkv_cols;
    int out_rows, out_cols;
    int fc1_rows, fc1_cols;
    int fc2_rows, fc2_cols;
} dit_layer;

struct hip_ppd_runner {
    hipDevice_t device;
    hipCtx_t context;
    hipStream_t stream;
    int verbose;

    hipModule_t module;
    /* Kernel function handles */
    hipFunction_t fn_layernorm_f32;
    hipFunction_t fn_gemm_tiled_f16_f32;
    hipFunction_t fn_add_bias_f32;
    hipFunction_t fn_flash_attn_tiled_f32;
    hipFunction_t fn_flash_attn_f16kv_f32;
    hipFunction_t fn_flash_attn_warp_f16kv_f32;
    hipFunction_t fn_kv_transpose;
    hipFunction_t fn_kv_transpose_f16;
    hipFunction_t fn_gelu_f32;
    hipFunction_t fn_add_f32;
    hipFunction_t fn_resize_normalize;
    hipFunction_t fn_patch_embed_conv2d;
    hipFunction_t fn_cls_pos_embed;
    hipFunction_t fn_bilinear_upsample_f32;
    hipFunction_t fn_silu_f32;
    /* DiT-specific */
    hipFunction_t fn_adaln_modulate_f32;
    hipFunction_t fn_gate_residual_add_f32;
    hipFunction_t fn_rope_2d_dit_f32;
    hipFunction_t fn_unpatchify_f32;
    hipFunction_t fn_concat_4ch_f32;
    hipFunction_t fn_dit_patch_embed_conv2d;
    hipFunction_t fn_dit_im2col;
    hipFunction_t fn_euler_step_f32;
    hipFunction_t fn_add_scalar_f32;
    hipFunction_t fn_sub_scalar_chw_f32;
    hipFunction_t fn_pixel_shuffle_2x_f32;
    hipFunction_t fn_concat_tokens_f32;
    hipFunction_t fn_qk_norm_f32;
    hipFunction_t fn_scalar_gemm_f16_f32;
    hipFunction_t fn_scalar_attn_f32;

    /* ---- DA2 semantic encoder params ---- */
    int sem_n_blocks;       /* 24 */
    int sem_dim;            /* 1024 */
    int sem_n_heads;        /* 16 */
    int sem_head_dim;       /* 64 */
    int sem_ffn_hidden;     /* 4096 */
    int sem_patch_size;     /* 14 */
    float sem_ln_eps;
    void *sem_patch_embed_w, *sem_patch_embed_b;
    void *sem_cls_token, *sem_pos_embed;
    sem_layer *sem_layers;
    void *sem_norm_w, *sem_norm_b;  /* final layernorm */
    float *sem_pos_embed_host;   /* Host F32 copy for interpolation */
    int sem_pos_embed_total;     /* 1 + n_patches_orig */
    int sem_pos_embed_gH, sem_pos_embed_gW; /* original grid dims */

    /* ---- DiT params ---- */
    int dit_n_blocks;       /* 24 */
    int dit_dim;            /* 1024 */
    int dit_n_heads;        /* 16 */
    int dit_head_dim;       /* 64 */
    int dit_ffn_hidden;     /* 4096 */
    int dit_patch_size;     /* 16 (input stride), 8 (output unpatchify) */
    float dit_ln_eps;
    float dit_rope_freq;    /* 100.0 */
    void *dit_x_embed_w, *dit_x_embed_b;     /* PatchEmbed Conv2d(4,1024,16,16) */
    void *dit_t_mlp_w1, *dit_t_mlp_b1;       /* TimestepEmbedder fc1(256,1024) */
    void *dit_t_mlp_w2, *dit_t_mlp_b2;       /* TimestepEmbedder fc2(1024,1024) */
    void *dit_proj_fusion_w[3], *dit_proj_fusion_b[3]; /* proj_fusion 3-layer MLP */
    void *dit_final_ln_w, *dit_final_ln_b;    /* final_layer adaLN */
    void *dit_final_adaln_w, *dit_final_adaln_b; /* final_layer modulation */
    void *dit_final_proj_w, *dit_final_proj_b;    /* final_layer linear */
    dit_layer *dit_layers;
    void *dit_const_ones, *dit_const_zeros; /* shared constant norm buffers */

    /* Reusable buffers */
    void *d_img_raw;
    size_t d_img_raw_cap;
    void *d_result;
    size_t d_result_cap;

    /* Scratch buffers */
    void *d_hidden, *d_hidden2, *d_ln_buf;
    void *d_qkv, *d_attn_out;
    void *d_ffn_buf, *d_proj_out;
    void *d_img_norm;       /* preprocessed image [3, H, W] */
    void *d_semantics;      /* [n_tok, 1024] semantic tokens from DA2 */
    void *d_latent;         /* [1, H, W] diffusion latent */
    void *d_cond;           /* [3, H, W] condition (image - 0.5) */
    void *d_dit_input;      /* [4, H, W] concat(latent, cond) */
    void *d_dit_hidden;     /* [n_tok, dim] DiT hidden state */
    void *d_dit_qkv;
    void *d_dit_attn_out;
    void *d_dit_ffn_buf;
    void *d_dit_proj_out;
    void *d_dit_ln_buf;
    void *d_dit_modulation; /* [n_tok, 6*dim] adaLN output */
    void *d_dit_t_embed;    /* [1, dim] timestep embedding */
    void *d_dit_pos_y, *d_dit_pos_x; /* DiT 2D position indices */
    void *d_dit_concat;     /* scratch for semantic fusion concat */
    void *d_dit_pred;       /* [1, H, W] DiT prediction output */

    int loaded;
};

/* ======================================================================== */
/* HIPRTC compilation                                                       */
/* ======================================================================== */

static int ppd_compile_kernels(hip_ppd_runner *r) {
    /* Concatenate shared + PPD-specific kernel sources */
    size_t len1 = strlen(hip_kernels_common_src);
    size_t len2 = strlen(hip_ppd_specific_kernels);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, hip_kernels_common_src, len1);
    memcpy(full_src + len1, hip_ppd_specific_kernels, len2 + 1);

    int rc = hip_compile_kernels(&r->module, r->device, full_src,
                                  "ppd_kernels.hip", r->verbose, "hip_ppd");
    free(full_src);
    if (rc < 0) return -1;

    hipError_t err;
#define GET_FN(name) do { \
    err = hipModuleGetFunction(&r->fn_##name, r->module, #name); \
    if (err != hipSuccess) { \
        fprintf(stderr, "hip_ppd: kernel '%s' not found\n", #name); return -1; } \
} while(0)

    /* Shared kernels (from hip_kernels_common.h) */
    GET_FN(layernorm_f32);
    GET_FN(gemm_tiled_f16_f32);
    GET_FN(add_bias_f32);
    GET_FN(kv_transpose);
    GET_FN(gelu_f32);
    GET_FN(add_f32);
    GET_FN(resize_normalize);
    GET_FN(patch_embed_conv2d);
    GET_FN(cls_pos_embed);
    GET_FN(bilinear_upsample_f32);
    GET_FN(silu_f32);

    /* PPD-specific kernels */
    GET_FN(flash_attn_tiled_f32);
    GET_FN(flash_attn_f16kv_f32);
    GET_FN(flash_attn_warp_f16kv_f32);
    GET_FN(kv_transpose_f16);
    GET_FN(qk_norm_f32);
    GET_FN(scalar_attn_f32);
    GET_FN(scalar_gemm_f16_f32);
    /* DiT-specific */
    GET_FN(adaln_modulate_f32);
    GET_FN(gate_residual_add_f32);
    GET_FN(rope_2d_dit_f32);
    GET_FN(unpatchify_f32);
    GET_FN(concat_4ch_f32);
    GET_FN(dit_patch_embed_conv2d);
    GET_FN(dit_im2col);
    GET_FN(euler_step_f32);
    GET_FN(add_scalar_f32);
    GET_FN(sub_scalar_chw_f32);
    GET_FN(pixel_shuffle_2x_f32);
    GET_FN(concat_tokens_f32);

#undef GET_FN

    if (r->verbose >= 1)
        fprintf(stderr, "hip_ppd: kernels compiled OK\n");
    return 0;
}

/* ======================================================================== */
/* Tensor upload helpers                                                    */
/* ======================================================================== */

/* Upload F32 data -> GPU F32 */
static void *upload_f32(hipStream_t stream, const float *data, size_t n) {
    void *d = NULL;
    hipMalloc(&d, n * sizeof(float));
    hipMemcpyAsync(d, (void *)data, n * sizeof(float), hipMemcpyHostToDevice, stream);
    return d;
}

/* Upload F16 data -> GPU F16 (raw) */
static void *upload_f16_raw(hipStream_t stream, const void *data, size_t nbytes) {
    void *d = NULL;
    hipMalloc(&d, nbytes);
    hipMemcpyAsync(d, (void *)data, nbytes, hipMemcpyHostToDevice, stream);
    return d;
}

/* Upload BF16 data -> GPU F32 (convert on CPU) */
static void *upload_bf16_as_f32(hipStream_t stream, const void *data, size_t n) {
    float *tmp = (float *)malloc(n * sizeof(float));
    const uint16_t *src = (const uint16_t *)data;
    for (size_t i = 0; i < n; i++) {
        uint32_t f32 = (uint32_t)src[i] << 16;
        memcpy(&tmp[i], &f32, 4);
    }
    void *d = upload_f32(stream, tmp, n);
    free(tmp);
    return d;
}

/* Upload any-dtype tensor -> GPU F32 */
static void *upload_tensor_as_f32(hipStream_t stream, const void *data,
                                    const char *dtype, size_t n) {
    if (strcmp(dtype, "F32") == 0) {
        return upload_f32(stream, (const float *)data, n);
    } else if (strcmp(dtype, "BF16") == 0) {
        return upload_bf16_as_f32(stream, data, n);
    } else if (strcmp(dtype, "F16") == 0) {
        /* Convert F16 -> F32 on CPU */
        float *tmp = (float *)malloc(n * sizeof(float));
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint16_t h = src[i];
            int s = (h >> 15) & 1;
            int e = (h >> 10) & 0x1f;
            int m = h & 0x3ff;
            float f;
            if (e == 0) f = (float)m * (1.0f / 16777216.0f);
            else if (e == 31) f = m ? NAN : INFINITY;
            else f = (1.0f + m / 1024.0f) * powf(2.0f, (float)(e - 15));
            tmp[i] = s ? -f : f;
        }
        void *d = upload_f32(stream, tmp, n);
        free(tmp);
        return d;
    }
    fprintf(stderr, "hip_ppd: unsupported dtype %s for F32 upload\n", dtype);
    return NULL;
}

/* Upload any-dtype tensor -> GPU F16 */
static void *upload_tensor_as_f16(hipStream_t stream, const void *data,
                                    const char *dtype, size_t n) {
    if (strcmp(dtype, "F16") == 0) {
        return upload_f16_raw(stream, data, n * 2);
    } else if (strcmp(dtype, "F32") == 0) {
        /* Convert F32 -> F16 on CPU using simple truncation */
        uint16_t *tmp = (uint16_t *)malloc(n * 2);
        const float *src = (const float *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t fb;
            memcpy(&fb, &src[i], 4);
            int s = (fb >> 31) & 1;
            int e = ((fb >> 23) & 0xff) - 127 + 15;
            int m = (fb >> 13) & 0x3ff;
            if (e <= 0) { e = 0; m = 0; }
            if (e > 30) { e = 31; m = 0; }
            tmp[i] = (uint16_t)((s << 15) | (e << 10) | m);
        }
        void *d = upload_f16_raw(stream, tmp, n * 2);
        free(tmp);
        return d;
    } else if (strcmp(dtype, "BF16") == 0) {
        /* BF16 -> F32 -> F16 */
        float *tmp32 = (float *)malloc(n * sizeof(float));
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t f32 = (uint32_t)src[i] << 16;
            memcpy(&tmp32[i], &f32, 4);
        }
        uint16_t *tmp16 = (uint16_t *)malloc(n * 2);
        for (size_t i = 0; i < n; i++) {
            uint32_t fb;
            memcpy(&fb, &tmp32[i], 4);
            int s = (fb >> 31) & 1;
            int e = ((fb >> 23) & 0xff) - 127 + 15;
            int m = (fb >> 13) & 0x3ff;
            if (e <= 0) { e = 0; m = 0; }
            if (e > 30) { e = 31; m = 0; }
            tmp16[i] = (uint16_t)((s << 15) | (e << 10) | m);
        }
        free(tmp32);
        void *d = upload_f16_raw(stream, tmp16, n * 2);
        free(tmp16);
        return d;
    }
    fprintf(stderr, "hip_ppd: unsupported dtype %s for F16 upload\n", dtype);
    return NULL;
}

/* ======================================================================== */
/* Weight loading from .pth                                                 */
/* ======================================================================== */

/* Helper: find tensor in pth, upload as F32 */
static void *pth_upload_f32(pth_context *pth, hipStream_t stream, const char *name) {
    int idx = pth_find(pth, name);
    if (idx < 0) return NULL;
    size_t n = pth_nbytes(pth, idx) / pth_dtype_size(pth_dtype(pth, idx));
    return upload_tensor_as_f32(stream, pth_data(pth, idx), pth_dtype(pth, idx), n);
}

/* Helper: find tensor in pth, upload as F16 */
static void *pth_upload_f16(pth_context *pth, hipStream_t stream, const char *name) {
    int idx = pth_find(pth, name);
    if (idx < 0) return NULL;
    size_t n = pth_nbytes(pth, idx) / pth_dtype_size(pth_dtype(pth, idx));
    return upload_tensor_as_f16(stream, pth_data(pth, idx), pth_dtype(pth, idx), n);
}

/* Helper: upload as F16 (no FP8 on RDNA4) */
static void *pth_upload_backbone(pth_context *pth, hipStream_t stream,
                                   const char *name) {
    return pth_upload_f16(pth, stream, name);
}

/* Helper: upload a constant F32 buffer (all elements = val) */
static void *upload_constant_f32(hipStream_t stream, float val, int n) {
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) buf[i] = val;
    void *d = NULL;
    hipMalloc(&d, (size_t)n * sizeof(float));
    hipMemcpyAsync(d, buf, (size_t)n * sizeof(float), hipMemcpyHostToDevice, stream);
    free(buf);
    return d;
}

/* Load DA2 semantic encoder weights from .pth */
static int ppd_load_sem_weights(hip_ppd_runner *r, const char *path) {
    pth_context *pth = pth_open(path);
    if (!pth) return -1;

    if (r->verbose >= 1)
        fprintf(stderr, "hip_ppd: loading DA2 semantic encoder from %s (%d tensors)\n",
                path, pth_count(pth));

    /* Detect model size from first QKV weight */
    int qkv_idx = pth_find(pth, "blocks.0.attn.qkv.weight");
    if (qkv_idx < 0) {
        qkv_idx = pth_find(pth, "pretrained.blocks.0.attn.qkv.weight");
    }
    if (qkv_idx < 0) {
        fprintf(stderr, "hip_ppd: cannot find QKV weight in %s\n", path);
        pth_close(pth);
        return -1;
    }

    /* Detect prefix */
    const char *prefix = "";
    if (strncmp(pth_name(pth, qkv_idx), "pretrained.", 11) == 0)
        prefix = "pretrained.";

    const uint64_t *qkv_shape = pth_shape(pth, qkv_idx);
    r->sem_dim = (int)(qkv_shape[1]);
    r->sem_n_heads = r->sem_dim / 64;
    r->sem_head_dim = 64;
    r->sem_ffn_hidden = r->sem_dim * 4;
    r->sem_patch_size = 14;
    r->sem_ln_eps = 1e-6f;

    /* Count blocks */
    r->sem_n_blocks = 0;
    for (int i = 0; i < pth_count(pth); i++) {
        const char *n = pth_name(pth, i);
        const char *bp = strstr(n, "blocks.");
        if (bp) {
            int blk = atoi(bp + 7);
            if (blk + 1 > r->sem_n_blocks) r->sem_n_blocks = blk + 1;
        }
    }

    if (r->verbose >= 1)
        fprintf(stderr, "hip_ppd: DA2 dim=%d heads=%d blocks=%d\n",
                r->sem_dim, r->sem_n_heads, r->sem_n_blocks);

    r->sem_layers = (sem_layer *)calloc(r->sem_n_blocks, sizeof(sem_layer));

    /* Load per-block weights */
    for (int L = 0; L < r->sem_n_blocks; L++) {
        sem_layer *ly = &r->sem_layers[L];
        char buf[256];

        snprintf(buf, sizeof(buf), "%sblocks.%d.norm1.weight", prefix, L);
        ly->ln1_w = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.norm1.bias", prefix, L);
        ly->ln1_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.weight", prefix, L);
        ly->attn_qkv_w = pth_upload_backbone(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.bias", prefix, L);
        ly->attn_qkv_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.weight", prefix, L);
        ly->attn_out_w = pth_upload_backbone(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.bias", prefix, L);
        ly->attn_out_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.norm2.weight", prefix, L);
        ly->ln2_w = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.norm2.bias", prefix, L);
        ly->ln2_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.weight", prefix, L);
        ly->ffn_up_w = pth_upload_backbone(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.bias", prefix, L);
        ly->ffn_up_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.weight", prefix, L);
        ly->ffn_down_w = pth_upload_backbone(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.bias", prefix, L);
        ly->ffn_down_b = pth_upload_f32(pth, r->stream, buf);

        /* LayerScale (DINOv2 has these) */
        snprintf(buf, sizeof(buf), "%sblocks.%d.ls1.gamma", prefix, L);
        ly->ls1_gamma = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.ls2.gamma", prefix, L);
        ly->ls2_gamma = pth_upload_f32(pth, r->stream, buf);

        ly->qkv_rows = 3 * r->sem_dim;
        ly->qkv_cols = r->sem_dim;
        ly->out_rows = r->sem_dim;
        ly->out_cols = r->sem_dim;
        ly->ffn_up_rows = r->sem_ffn_hidden;
        ly->ffn_up_cols = r->sem_dim;
        ly->ffn_down_rows = r->sem_dim;
        ly->ffn_down_cols = r->sem_ffn_hidden;
    }

    /* Patch embedding, CLS token, position embedding */
    char buf[256];
    snprintf(buf, sizeof(buf), "%spatch_embed.proj.weight", prefix);
    r->sem_patch_embed_w = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%spatch_embed.proj.bias", prefix);
    r->sem_patch_embed_b = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%scls_token", prefix);
    r->sem_cls_token = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%spos_embed", prefix);
    r->sem_pos_embed = pth_upload_f32(pth, r->stream, buf);

    /* Store host copy of pos_embed for interpolation at predict time */
    {
        int pe_idx = pth_find(pth, buf);
        if (pe_idx >= 0) {
            const uint64_t *pe_shape = pth_shape(pth, pe_idx);
            int pe_ndims = pth_ndims(pth, pe_idx);
            int pe_n = (pe_ndims == 3) ? (int)pe_shape[1] : (int)pe_shape[0];
            r->sem_pos_embed_total = pe_n;
            int n_patches = pe_n - 1;
            int g = (int)sqrtf((float)n_patches);
            if (g * g != n_patches) {
                fprintf(stderr, "hip_ppd: warning: non-square pos_embed grid %d\n", n_patches);
                g = (int)sqrtf((float)n_patches);
            }
            r->sem_pos_embed_gH = g;
            r->sem_pos_embed_gW = g;
            /* Download GPU copy (already uploaded as F32) */
            hipStreamSynchronize(r->stream);
            size_t pe_bytes = (size_t)pe_n * r->sem_dim * sizeof(float);
            r->sem_pos_embed_host = (float *)malloc(pe_bytes);
            hipMemcpy(r->sem_pos_embed_host, r->sem_pos_embed, pe_bytes, hipMemcpyDeviceToHost);
            if (r->verbose >= 1)
                fprintf(stderr, "hip_ppd: pos_embed: %d tokens, grid %dx%d\n",
                        pe_n, g, g);
        }
    }

    /* Final layernorm */
    snprintf(buf, sizeof(buf), "%snorm.weight", prefix);
    r->sem_norm_w = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%snorm.bias", prefix);
    r->sem_norm_b = pth_upload_f32(pth, r->stream, buf);

    hipStreamSynchronize(r->stream);
    pth_close(pth);

    if (r->verbose >= 1)
        fprintf(stderr, "hip_ppd: DA2 weights loaded OK\n");
    return 0;
}

/* Load DiT weights from .pth */
static int ppd_load_dit_weights(hip_ppd_runner *r, const char *path) {
    pth_context *pth = pth_open(path);
    if (!pth) return -1;

    if (r->verbose >= 1)
        fprintf(stderr, "hip_ppd: loading DiT weights from %s (%d tensors)\n",
                path, pth_count(pth));

    /* Set DiT hyperparameters (fixed for PPD) */
    r->dit_n_blocks = 24;
    r->dit_dim = 1024;
    r->dit_n_heads = 16;
    r->dit_head_dim = 64;
    r->dit_ffn_hidden = 4096;
    r->dit_patch_size = 16;
    r->dit_ln_eps = 1e-6f;
    r->dit_rope_freq = 100.0f;

    /* Detect prefix */
    const char *dit_pfx = "";
    if (pth_find(pth, "dit.x_embedder.proj.weight") >= 0)
        dit_pfx = "dit.";
    else if (pth_find(pth, "x_embedder.proj.weight") < 0) {
        fprintf(stderr, "hip_ppd: cannot find x_embedder.proj.weight in %s\n", path);
        pth_close(pth);
        return -1;
    }

    r->dit_layers = (dit_layer *)calloc(r->dit_n_blocks, sizeof(dit_layer));

    char buf[256];

    /* PatchEmbed */
    snprintf(buf, sizeof(buf), "%sx_embedder.proj.weight", dit_pfx);
    r->dit_x_embed_w = pth_upload_backbone(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sx_embedder.proj.bias", dit_pfx);
    r->dit_x_embed_b = pth_upload_f32(pth, r->stream, buf);

    /* TimestepEmbedder */
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.0.weight", dit_pfx);
    r->dit_t_mlp_w1 = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.0.bias", dit_pfx);
    r->dit_t_mlp_b1 = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.2.weight", dit_pfx);
    r->dit_t_mlp_w2 = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.2.bias", dit_pfx);
    r->dit_t_mlp_b2 = pth_upload_f32(pth, r->stream, buf);

    /* Proj fusion (3 layers) */
    snprintf(buf, sizeof(buf), "%sproj_fusion.0.weight", dit_pfx);
    r->dit_proj_fusion_w[0] = pth_upload_backbone(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.0.bias", dit_pfx);
    r->dit_proj_fusion_b[0] = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.2.weight", dit_pfx);
    r->dit_proj_fusion_w[1] = pth_upload_backbone(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.2.bias", dit_pfx);
    r->dit_proj_fusion_b[1] = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.4.weight", dit_pfx);
    r->dit_proj_fusion_w[2] = pth_upload_backbone(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.4.bias", dit_pfx);
    r->dit_proj_fusion_b[2] = pth_upload_f32(pth, r->stream, buf);

    /* Final layer */
    snprintf(buf, sizeof(buf), "%sfinal_layer.adaLN_modulation.1.weight", dit_pfx);
    r->dit_final_ln_w = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sfinal_layer.adaLN_modulation.1.bias", dit_pfx);
    r->dit_final_ln_b = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sfinal_layer.linear.weight", dit_pfx);
    r->dit_final_proj_w = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sfinal_layer.linear.bias", dit_pfx);
    r->dit_final_proj_b = pth_upload_f32(pth, r->stream, buf);
    /* norm_final uses elementwise_affine=False -> constant weights */
    r->dit_final_adaln_w = upload_constant_f32(r->stream, 1.0f, r->dit_dim);
    r->dit_final_adaln_b = upload_constant_f32(r->stream, 0.0f, r->dit_dim);

    /* Constant norm buffers for DiT blocks (elementwise_affine=False) */
    r->dit_const_ones = upload_constant_f32(r->stream, 1.0f, r->dit_dim);
    r->dit_const_zeros = upload_constant_f32(r->stream, 0.0f, r->dit_dim);

    /* Per-block DiT weights */
    for (int L = 0; L < r->dit_n_blocks; L++) {
        dit_layer *ly = &r->dit_layers[L];

        ly->ln1_w = r->dit_const_ones;
        ly->ln1_b = r->dit_const_zeros;
        ly->ln2_w = r->dit_const_ones;
        ly->ln2_b = r->dit_const_zeros;

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.weight", dit_pfx, L);
        ly->attn_qkv_w = pth_upload_backbone(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.bias", dit_pfx, L);
        ly->attn_qkv_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.q_norm.weight", dit_pfx, L);
        ly->attn_q_norm_w = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.q_norm.bias", dit_pfx, L);
        ly->attn_q_norm_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.k_norm.weight", dit_pfx, L);
        ly->attn_k_norm_w = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.k_norm.bias", dit_pfx, L);
        ly->attn_k_norm_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.weight", dit_pfx, L);
        ly->attn_out_w = pth_upload_backbone(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.bias", dit_pfx, L);
        ly->attn_out_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.weight", dit_pfx, L);
        ly->mlp_fc1_w = pth_upload_backbone(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.bias", dit_pfx, L);
        ly->mlp_fc1_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.weight", dit_pfx, L);
        ly->mlp_fc2_w = pth_upload_backbone(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.bias", dit_pfx, L);
        ly->mlp_fc2_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.adaLN_modulation.1.weight", dit_pfx, L);
        ly->adaln_w = pth_upload_f16(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.adaLN_modulation.1.bias", dit_pfx, L);
        ly->adaln_b = pth_upload_f32(pth, r->stream, buf);

        ly->qkv_rows = 3 * r->dit_dim;
        ly->qkv_cols = r->dit_dim;
        ly->out_rows = r->dit_dim;
        ly->out_cols = r->dit_dim;
        ly->fc1_rows = r->dit_ffn_hidden;
        ly->fc1_cols = r->dit_dim;
        ly->fc2_rows = r->dit_dim;
        ly->fc2_cols = r->dit_ffn_hidden;
    }

    hipStreamSynchronize(r->stream);
    pth_close(pth);

    if (r->verbose >= 1)
        fprintf(stderr, "hip_ppd: DiT weights loaded OK\n");
    return 0;
}

/* ======================================================================== */
/* Public API: init, load, predict, free                                    */
/* ======================================================================== */

hip_ppd_runner *hip_ppd_init(int device_id, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_ppd: rocew init failed\n");
        return NULL;
    }
    hipInit(0);

    hip_ppd_runner *r = (hip_ppd_runner *)calloc(1, sizeof(hip_ppd_runner));
    r->verbose = verbose;
    r->device = device_id;

    HIP_CHECK_NULL(hipSetDevice(device_id));
    HIP_CHECK_NULL(hipCtxCreate(&r->context, 0, r->device));
    HIP_CHECK_NULL(hipStreamCreateWithFlags(&r->stream, hipStreamNonBlocking));

    if (ppd_compile_kernels(r) != 0) {
        fprintf(stderr, "hip_ppd: kernel compilation failed\n");
        free(r);
        return NULL;
    }

    return r;
}

int hip_ppd_load_weights(hip_ppd_runner *r,
                          const char *ppd_pth_path,
                          const char *sem_pth_path) {
    /* Load semantic encoder (DA2) */
    if (ppd_load_sem_weights(r, sem_pth_path) != 0) {
        fprintf(stderr, "hip_ppd: failed to load semantic encoder weights\n");
        return -1;
    }

    /* Load DiT */
    if (ppd_load_dit_weights(r, ppd_pth_path) != 0) {
        fprintf(stderr, "hip_ppd: failed to load DiT weights\n");
        return -1;
    }

    r->loaded = 1;
    return 0;
}

/* Kernel launch wrappers */
static void kl_layernorm(hip_ppd_runner *r, void *dst, void *src,
                          void *w, void *b, int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    hipModuleLaunchKernel(r->fn_layernorm_f32, (unsigned)n_tok, 1, 1, 256, 1, 1,
                          256 * sizeof(float), r->stream, args, NULL);
}

static void kl_gemm(hip_ppd_runner *r, void *Y, void *W_f16,
                     void *X, void *bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W_f16, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipModuleLaunchKernel(r->fn_gemm_tiled_f16_f32, gx, gy, 1, 16, 16, 1,
                          0, r->stream, args, NULL);
}

static void kl_backbone_gemm(hip_ppd_runner *r, void *Y, void *W,
                               void *X, void *bias, int n_out, int n_in, int n_tok) {
    kl_gemm(r, Y, W, X, bias, n_out, n_in, n_tok);
}

static void kl_kv_transpose(hip_ppd_runner *r, void *K_t, void *V_t,
                              void *qkv, int n_tok, int dim, int n_heads, int head_dim) {
    int total = n_tok * dim;
    int grid = (total + 255) / 256;
    void *args[] = {&K_t, &V_t, &qkv, &n_tok, &dim, &n_heads, &head_dim};
    hipModuleLaunchKernel(r->fn_kv_transpose, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_attn(hip_ppd_runner *r, void *out, void *qkv,
                     void *K_t, void *V_t,
                     int n_tok, int dim, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    void *args[] = {&out, &qkv, &K_t, &V_t, &n_tok, &dim, &n_heads, &head_dim, &scale};
    int bq = 64, bkv = 16;
    unsigned smem_size = (unsigned)(2 * bkv * head_dim * sizeof(float));
    unsigned gy = (unsigned)((n_tok + bq - 1) / bq);
    hipModuleLaunchKernel(r->fn_flash_attn_tiled_f32,
                          (unsigned)n_heads, gy, 1, (unsigned)bq, 1, 1,
                          smem_size, r->stream, args, NULL);
}

/* kl_kv_transpose_f16: transposes QKV K/V into FP16 K_t, V_t buffers */
static void kl_kv_transpose_f16(hip_ppd_runner *r, void *K_t_f16, void *V_t_f16,
                                  void *qkv, int n_tok, int dim, int n_heads, int head_dim) {
    int total = n_tok * dim;
    int grid = (total + 255) / 256;
    void *args[] = {&K_t_f16, &V_t_f16, &qkv, &n_tok, &dim, &n_heads, &head_dim};
    hipModuleLaunchKernel(r->fn_kv_transpose_f16, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

/* kl_attn_f16kv: flash attention reading FP16 K_t/V_t, output FP32 */
static void kl_attn_f16kv(hip_ppd_runner *r, void *out, void *qkv,
                            void *K_t_f16, void *V_t_f16,
                            int n_tok, int dim, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    void *args[] = {&out, &qkv, &K_t_f16, &V_t_f16, &n_tok, &dim, &n_heads, &head_dim, &scale};
    int bq = 64;
    unsigned smem_size = (unsigned)(2 * 16 * head_dim * sizeof(unsigned short));
    unsigned gy = (unsigned)((n_tok + bq - 1) / bq);
    hipModuleLaunchKernel(r->fn_flash_attn_f16kv_f32,
                          (unsigned)n_heads, gy, 1, (unsigned)bq, 1, 1,
                          smem_size, r->stream, args, NULL);
}

static void kl_gelu(hip_ppd_runner *r, void *x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_gelu_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_add(hip_ppd_runner *r, void *dst, void *src, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &n};
    hipModuleLaunchKernel(r->fn_add_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_silu(hip_ppd_runner *r, void *x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_silu_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_qk_norm(hip_ppd_runner *r, void *qkv, void *w, void *b,
                         int n_tok, int n_heads, int head_dim, int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    int grid = (total + 255) / 256;
    void *args[] = {&qkv, &w, &b, &n_tok, &n_heads, &head_dim, &stride, &eps};
    hipModuleLaunchKernel(r->fn_qk_norm_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_rope_2d_dit(hip_ppd_runner *r, void *vec, void *pos_y, void *pos_x,
                             int n_tok, int n_heads, int head_dim, int stride, float freq_base) {
    int quarter = head_dim / 4;
    int threads = n_heads * quarter;
    if (threads > 1024) threads = 1024;
    void *args[] = {&vec, &pos_y, &pos_x, &n_tok, &n_heads, &head_dim, &stride, &freq_base};
    hipModuleLaunchKernel(r->fn_rope_2d_dit_f32, (unsigned)n_tok, 1, 1, (unsigned)threads, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_adaln_modulate(hip_ppd_runner *r, void *dst, void *src,
                                void *w, void *b,
                                void *shift, void *scale,
                                int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &shift, &scale, &dim, &eps};
    hipModuleLaunchKernel(r->fn_adaln_modulate_f32, (unsigned)n_tok, 1, 1, 256, 1, 1,
                          256 * sizeof(float), r->stream, args, NULL);
}

static void kl_gate_residual_add(hip_ppd_runner *r, void *dst, void *src,
                                   void *gate, int dim, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &gate, &dim, &n};
    hipModuleLaunchKernel(r->fn_gate_residual_add_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_concat_4ch(hip_ppd_runner *r, void *dst, void *latent,
                            void *cond, int HW) {
    int n = 4 * HW;
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &latent, &cond, &HW};
    hipModuleLaunchKernel(r->fn_concat_4ch_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_unpatchify(hip_ppd_runner *r, void *dst, void *src,
                            int gH, int gW, int ps) {
    int n = gH * ps * gW * ps;
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &gH, &gW, &ps};
    hipModuleLaunchKernel(r->fn_unpatchify_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_euler_step(hip_ppd_runner *r, void *latent, void *pred,
                            float t_ratio, float s_ratio, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&latent, &pred, &t_ratio, &s_ratio, &n};
    hipModuleLaunchKernel(r->fn_euler_step_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_add_scalar(hip_ppd_runner *r, void *dst, float scalar, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &scalar, &n};
    hipModuleLaunchKernel(r->fn_add_scalar_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_concat_tokens(hip_ppd_runner *r, void *dst, void *a, void *b,
                               int n_tok, int dim_a, int dim_b) {
    int n = n_tok * (dim_a + dim_b);
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &a, &b, &n_tok, &dim_a, &dim_b};
    hipModuleLaunchKernel(r->fn_concat_tokens_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

static void kl_pixel_shuffle_2x(hip_ppd_runner *r, void *dst, void *src,
                                  int gH, int gW, int dim) {
    int n = gH * 2 * gW * 2 * dim;
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &gH, &gW, &dim};
    hipModuleLaunchKernel(r->fn_pixel_shuffle_2x_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                          0, r->stream, args, NULL);
}

/* ======================================================================== */
/* Helper functions for predict                                              */
/* ======================================================================== */

/* Bilinear interpolation of position embeddings on CPU */
static float *interpolate_pos_embed(const float *patch_pe, int dim,
                                      int orig_gH, int orig_gW,
                                      int new_gH, int new_gW) {
    int new_n = new_gH * new_gW;
    float *out = (float *)malloc((size_t)new_n * dim * sizeof(float));
    for (int oh = 0; oh < new_gH; oh++) {
        for (int ow = 0; ow < new_gW; ow++) {
            float fy = (new_gH > 1) ? (float)oh * (orig_gH - 1) / (new_gH - 1) : 0;
            float fx = (new_gW > 1) ? (float)ow * (orig_gW - 1) / (new_gW - 1) : 0;
            int y0 = (int)fy, x0 = (int)fx;
            int y1 = (y0 + 1 < orig_gH) ? y0 + 1 : y0;
            int x1 = (x0 + 1 < orig_gW) ? x0 + 1 : x0;
            float dy = fy - y0, dx = fx - x0;
            int dst_tok = oh * new_gW + ow;
            for (int d = 0; d < dim; d++) {
                out[dst_tok * dim + d] =
                    patch_pe[(y0 * orig_gW + x0) * dim + d] * (1-dy) * (1-dx) +
                    patch_pe[(y0 * orig_gW + x1) * dim + d] * (1-dy) * dx +
                    patch_pe[(y1 * orig_gW + x0) * dim + d] * dy * (1-dx) +
                    patch_pe[(y1 * orig_gW + x1) * dim + d] * dy * dx;
            }
        }
    }
    return out;
}

/* Generate standard normal random values via Box-Muller */
static void generate_randn(float *buf, int n) {
    for (int i = 0; i < n - 1; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        float rv = sqrtf(-2.0f * logf(u1));
        buf[i] = rv * cosf(2.0f * (float)M_PI * u2);
        buf[i + 1] = rv * sinf(2.0f * (float)M_PI * u2);
    }
    if (n % 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        buf[n - 1] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    }
}

/* Compute sinusoidal timestep embedding [256] for a given timestep */
static void compute_sinusoidal_embed(float *out, float t, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)(logf(10000.0f)) * (float)i / (float)half);
        float arg = t * freq;
        out[i] = cosf(arg);
        out[half + i] = sinf(arg);
    }
}

/* Generate 2D position indices for a grid */
static void generate_grid_pos(int *pos_y, int *pos_x, int gH, int gW) {
    for (int h = 0; h < gH; h++)
        for (int w = 0; w < gW; w++) {
            int idx = h * gW + w;
            pos_y[idx] = h;
            pos_x[idx] = w;
        }
}

/* ======================================================================== */
/* Predict: full inference pipeline                                          */
/* ======================================================================== */

ppd_result hip_ppd_predict(hip_ppd_runner *r, const uint8_t *rgb, int w, int h) {
    ppd_result res = {0};
    if (!r || !r->loaded) return res;

    struct timespec ts;
    double t0, t1;

    /* -- Step 1: Determine processing resolutions -- */
    int proc_h = ((h + 15) / 16) * 16;
    int proc_w = ((w + 15) / 16) * 16;

    /* DA2 semantic encoder resolution */
    int sem_ps = r->sem_patch_size; /* 14 */
    int sem_gH = proc_h / 16;
    int sem_gW = proc_w / 16;
    int sem_h = sem_gH * sem_ps;
    int sem_w = sem_gW * sem_ps;
    int sem_np = sem_gH * sem_gW;
    int sem_nt = 1 + sem_np;

    /* DiT resolutions */
    int dit_gH_lo = proc_h / 16, dit_gW_lo = proc_w / 16;
    int dit_nt_lo = dit_gH_lo * dit_gW_lo;
    int dit_gH_hi = proc_h / 8, dit_gW_hi = proc_w / 8;
    int dit_nt_hi = dit_gH_hi * dit_gW_hi;

    int sem_dim = r->sem_dim;
    int dit_dim = r->dit_dim;
    int HW = proc_h * proc_w;

    if (r->verbose >= 1) {
        fprintf(stderr, "hip_ppd: predict %dx%d -> proc %dx%d\n", w, h, proc_w, proc_h);
        fprintf(stderr, "hip_ppd: DA2 grid %dx%d (%d patches), DiT lo %dx%d hi %dx%d\n",
                sem_gH, sem_gW, sem_np, dit_gH_lo, dit_gW_lo, dit_gH_hi, dit_gW_hi);
    }

    /* -- Step 2: Allocate scratch buffers -- */
    int max_nt = dit_nt_hi;
    if (sem_nt > max_nt) max_nt = sem_nt;

    void *d_hidden = NULL, *d_ln_buf = NULL, *d_qkv = NULL, *d_attn_out = NULL;
    void *d_ffn_buf = NULL, *d_proj_out = NULL;
    hipMalloc(&d_hidden,   (size_t)max_nt * sem_dim * sizeof(float));
    hipMalloc(&d_ln_buf,   (size_t)max_nt * sem_dim * sizeof(float));
    hipMalloc(&d_qkv,      (size_t)max_nt * 3 * sem_dim * sizeof(float));
    hipMalloc(&d_attn_out, (size_t)max_nt * sem_dim * sizeof(float));
    hipMalloc(&d_ffn_buf,  (size_t)max_nt * r->sem_ffn_hidden * sizeof(float));
    hipMalloc(&d_proj_out, (size_t)max_nt * sem_dim * sizeof(float));

    void *d_img_norm = NULL, *d_cond = NULL, *d_latent = NULL;
    void *d_dit_input = NULL, *d_dit_pred = NULL;
    hipMalloc(&d_img_norm,  (size_t)3 * sem_h * sem_w * sizeof(float));
    hipMalloc(&d_cond,      (size_t)3 * HW * sizeof(float));
    hipMalloc(&d_latent,    (size_t)HW * sizeof(float));
    hipMalloc(&d_dit_input, (size_t)4 * HW * sizeof(float));
    hipMalloc(&d_dit_pred,  (size_t)HW * sizeof(float));

    void *d_semantics = NULL;
    void *d_t_embed = NULL;
    void *d_t_embed_silu = NULL;
    void *d_modulation = NULL;
    void *d_pos_y = NULL, *d_pos_x = NULL;
    void *d_concat = NULL;
    void *d_fusion_out = NULL;
    hipMalloc(&d_semantics,    (size_t)sem_np * sem_dim * sizeof(float));
    hipMalloc(&d_t_embed,      (size_t)dit_dim * sizeof(float));
    hipMalloc(&d_t_embed_silu, (size_t)dit_dim * sizeof(float));
    hipMalloc(&d_modulation,   (size_t)6 * dit_dim * sizeof(float));
    void *d_sin_buf = NULL;
    hipMalloc(&d_sin_buf,      256 * sizeof(float));
    hipMalloc(&d_pos_y,        (size_t)dit_nt_hi * sizeof(int));
    hipMalloc(&d_pos_x,        (size_t)dit_nt_hi * sizeof(int));
    hipMalloc(&d_concat,       (size_t)dit_nt_lo * 4 * dit_dim * sizeof(float));
    int ffn_buf_elems = max_nt * r->sem_ffn_hidden;
    int fusion_elems = dit_nt_lo * 4 * dit_dim;
    if (fusion_elems > ffn_buf_elems) {
        hipMalloc(&d_fusion_out, (size_t)fusion_elems * sizeof(float));
    } else {
        d_fusion_out = d_ffn_buf;
    }

    /* Upload raw RGB image to GPU */
    size_t img_bytes = (size_t)w * h * 3;
    if (img_bytes > r->d_img_raw_cap) {
        if (r->d_img_raw) hipFree(r->d_img_raw);
        hipMalloc(&r->d_img_raw, img_bytes);
        r->d_img_raw_cap = img_bytes;
    }
    hipMemcpyAsync(r->d_img_raw, (void *)rgb, img_bytes, hipMemcpyHostToDevice, r->stream);

    /* ================================================================ */
    /*  Step 3: DA2 Semantic Encoder                                     */
    /* ================================================================ */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    /* 3a. Resize + ImageNet normalize -> d_img_norm [3, sem_h, sem_w] */
    {
        int total = sem_h * sem_w;
        int grid = (total + 255) / 256;
        float m0 = 0.485f, m1 = 0.456f, m2 = 0.406f;
        float is0 = 1.0f/0.229f, is1 = 1.0f/0.224f, is2 = 1.0f/0.225f;
        int src_w = w, src_h = h, dst_w = sem_w, dst_h = sem_h;
        void *args[] = {&d_img_norm, &r->d_img_raw, &src_w, &src_h, &dst_w, &dst_h,
                        &m0, &m1, &m2, &is0, &is1, &is2};
        hipModuleLaunchKernel(r->fn_resize_normalize, (unsigned)grid, 1, 1, 256, 1, 1,
                              0, r->stream, args, NULL);
    }

    /* 3b. Patch embedding via im2col + tiled GEMM */
    {
        int in_ch = 3;
        int in_elems = in_ch * sem_ps * sem_ps;
        /* Step 1: im2col */
        {
            int blk = 8;
            unsigned gx = (unsigned)((sem_np + blk - 1) / blk);
            unsigned gy = (unsigned)in_ch;
            void *args[] = {&d_ffn_buf, &d_img_norm, &sem_np, &sem_gW,
                            &sem_ps, &sem_h, &sem_w, &in_ch};
            hipModuleLaunchKernel(r->fn_dit_im2col, gx, gy, 1,
                                  (unsigned)(sem_ps * sem_ps), 1, 1,
                                  0, r->stream, args, NULL);
        }
        /* Step 2: GEMM -> d_hidden[1..] (patch tokens, skip CLS at [0]) */
        void *d_hidden_patches = (char *)d_hidden + (size_t)sem_dim * sizeof(float);
        kl_gemm(r, d_hidden_patches, r->sem_patch_embed_w, d_ffn_buf,
                r->sem_patch_embed_b, sem_dim, in_elems, sem_np);
    }

    /* 3c. CLS token + position embedding (with interpolation if needed) */
    {
        void *d_pos = r->sem_pos_embed;
        void *d_pos_alloc = NULL;

        if (r->sem_pos_embed_host &&
            (sem_gH != r->sem_pos_embed_gH || sem_gW != r->sem_pos_embed_gW)) {
            float *cls_pe = r->sem_pos_embed_host;
            float *patch_pe = r->sem_pos_embed_host + r->sem_dim;
            float *interp = interpolate_pos_embed(patch_pe, r->sem_dim,
                                                    r->sem_pos_embed_gH, r->sem_pos_embed_gW,
                                                    sem_gH, sem_gW);
            size_t pe_bytes = (size_t)sem_nt * r->sem_dim * sizeof(float);
            float *full_pe = (float *)malloc(pe_bytes);
            memcpy(full_pe, cls_pe, (size_t)r->sem_dim * sizeof(float));
            memcpy(full_pe + r->sem_dim, interp, (size_t)sem_np * r->sem_dim * sizeof(float));
            free(interp);

            hipMalloc(&d_pos_alloc, pe_bytes);
            hipMemcpyAsync(d_pos_alloc, full_pe, pe_bytes, hipMemcpyHostToDevice, r->stream);
            free(full_pe);
            d_pos = d_pos_alloc;

            if (r->verbose >= 2)
                fprintf(stderr, "hip_ppd: interpolated pos_embed %dx%d -> %dx%d\n",
                        r->sem_pos_embed_gH, r->sem_pos_embed_gW, sem_gH, sem_gW);
        }

        int total = sem_nt * sem_dim;
        int grid = (total + 255) / 256;
        void *args[] = {&d_hidden, &r->sem_cls_token, &d_pos, &sem_nt, &sem_dim};
        hipModuleLaunchKernel(r->fn_cls_pos_embed, (unsigned)grid, 1, 1, 256, 1, 1,
                              0, r->stream, args, NULL);

        if (d_pos_alloc) { hipStreamSynchronize(r->stream); hipFree(d_pos_alloc); }
    }

    /* 3d. DA2 backbone: 24 ViT blocks */
    {
        int dim = sem_dim;
        int nt = sem_nt;
        int n_heads = r->sem_n_heads;
        int head_dim = r->sem_head_dim;

        for (int L = 0; L < r->sem_n_blocks; L++) {
            sem_layer *ly = &r->sem_layers[L];

            /* LN1 */
            kl_layernorm(r, d_ln_buf, d_hidden, ly->ln1_w, ly->ln1_b, nt, dim);

            /* QKV projection */
            kl_backbone_gemm(r, d_qkv, ly->attn_qkv_w, d_ln_buf, ly->attn_qkv_b,
                             ly->qkv_rows, ly->qkv_cols, nt);

            /* KV transpose + attention (F16 K/V) */
            {
                void *K_t = d_ffn_buf;
                void *V_t = (char *)d_ffn_buf + (size_t)nt * dim * sizeof(uint16_t);
                kl_kv_transpose_f16(r, K_t, V_t, d_qkv, nt, dim, n_heads, head_dim);
                kl_attn_f16kv(r, d_attn_out, d_qkv, K_t, V_t, nt, dim, n_heads, head_dim);
            }

            /* Output projection + LayerScale + residual */
            kl_backbone_gemm(r, d_proj_out, ly->attn_out_w, d_attn_out, ly->attn_out_b,
                             ly->out_rows, ly->out_cols, nt);
            if (ly->ls1_gamma)
                kl_gate_residual_add(r, d_hidden, d_proj_out, ly->ls1_gamma, dim, nt * dim);
            else
                kl_add(r, d_hidden, d_proj_out, nt * dim);

            /* LN2 */
            kl_layernorm(r, d_ln_buf, d_hidden, ly->ln2_w, ly->ln2_b, nt, dim);

            /* FFN: fc1 -> GELU -> fc2 + LayerScale + residual */
            kl_backbone_gemm(r, d_ffn_buf, ly->ffn_up_w, d_ln_buf, ly->ffn_up_b,
                             ly->ffn_up_rows, ly->ffn_up_cols, nt);
            kl_gelu(r, d_ffn_buf, nt * ly->ffn_up_rows);
            kl_backbone_gemm(r, d_proj_out, ly->ffn_down_w, d_ffn_buf, ly->ffn_down_b,
                             ly->ffn_down_rows, ly->ffn_down_cols, nt);
            if (ly->ls2_gamma)
                kl_gate_residual_add(r, d_hidden, d_proj_out, ly->ls2_gamma, dim, nt * dim);
            else
                kl_add(r, d_hidden, d_proj_out, nt * dim);
        }

        /* Final layernorm */
        kl_layernorm(r, d_hidden, d_hidden, r->sem_norm_w, r->sem_norm_b, nt, dim);
    }

    /* 3e. Extract patch tokens (skip CLS at index 0) -> d_semantics */
    hipMemcpyAsync(d_semantics,
                    (char *)d_hidden + (size_t)sem_dim * sizeof(float),
                    (size_t)sem_np * sem_dim * sizeof(float),
                    hipMemcpyDeviceToDevice, r->stream);

    hipStreamSynchronize(r->stream);
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "hip_ppd: DA2 semantic encoder: %.1f ms (%d patches)\n",
                (t1-t0)*1000, sem_np);

    /* ================================================================ */
    /*  Step 4: Prepare DiT inputs                                       */
    /* ================================================================ */

    /* 4a. Condition image */
    {
        int total = proc_h * proc_w;
        int grid = (total + 255) / 256;
        float m0 = 0.5f, m1 = 0.5f, m2 = 0.5f;
        float is0 = 1.0f, is1 = 1.0f, is2 = 1.0f;
        int src_w = w, src_h = h, dst_w = proc_w, dst_h = proc_h;
        void *args[] = {&d_cond, &r->d_img_raw, &src_w, &src_h, &dst_w, &dst_h,
                        &m0, &m1, &m2, &is0, &is1, &is2};
        hipModuleLaunchKernel(r->fn_resize_normalize, (unsigned)grid, 1, 1, 256, 1, 1,
                              0, r->stream, args, NULL);
    }

    /* 4b. Initialize latent ~ N(0, 1) */
    {
        float *noise = (float *)malloc((size_t)HW * sizeof(float));
        srand(42);
        generate_randn(noise, HW);
        hipMemcpyAsync(d_latent, noise, (size_t)HW * sizeof(float), hipMemcpyHostToDevice, r->stream);
        free(noise);
    }

    /* ================================================================ */
    /*  Step 5: DiT diffusion -- 4 Euler steps                          */
    /* ================================================================ */
    hipEvent_t ev_attn_start = NULL, ev_attn_stop = NULL;
    hipEvent_t ev_gemm_start = NULL, ev_gemm_stop = NULL;
    if (r->verbose >= 2) {
        hipEventCreate(&ev_attn_start);
        hipEventCreate(&ev_attn_stop);
        hipEventCreate(&ev_gemm_start);
        hipEventCreate(&ev_gemm_stop);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    float timesteps[] = {1000.0f, 750.0f, 500.0f, 250.0f};
    float T = 1000.0f;

    for (int step = 0; step < 4; step++) {
        float t_cur = timesteps[step];
        float t_next = (step < 3) ? timesteps[step + 1] : 0.0f;
        float t_ratio = t_cur / T;
        float s_ratio = t_next / T;

        if (r->verbose >= 2)
            fprintf(stderr, "hip_ppd: Euler step %d: t=%.0f -> s=%.0f\n", step, t_cur, t_next);

        /* 5a. Concat [latent, cond] -> d_dit_input [4, proc_h, proc_w] */
        kl_concat_4ch(r, d_dit_input, d_latent, d_cond, HW);

        /* 5b. DiT PatchEmbed */
        {
            int dit_ps = 16;
            int in_ch = 4, in_elems = in_ch * dit_ps * dit_ps;
            {
                int blk = 8;
                unsigned gx = (unsigned)((dit_nt_lo + blk - 1) / blk);
                unsigned gy = (unsigned)in_ch;
                int npatches = dit_nt_lo;
                void *args[] = {&d_ffn_buf, &d_dit_input, &npatches,
                                &dit_gW_lo, &dit_ps, &proc_h, &proc_w, &in_ch};
                hipModuleLaunchKernel(r->fn_dit_im2col, gx, gy, 1,
                                      (unsigned)(dit_ps * dit_ps), 1, 1,
                                      0, r->stream, args, NULL);
            }
            kl_backbone_gemm(r, d_hidden, r->dit_x_embed_w, d_ffn_buf,
                             r->dit_x_embed_b, dit_dim, in_elems, dit_nt_lo);
        }

        /* 5c. Timestep embedding */
        {
            float sin_embed[256];
            compute_sinusoidal_embed(sin_embed, t_cur, 256);
            hipMemcpyAsync(d_sin_buf, sin_embed, 256 * sizeof(float), hipMemcpyHostToDevice, r->stream);
            kl_gemm(r, d_t_embed, r->dit_t_mlp_w1, d_sin_buf, r->dit_t_mlp_b1, dit_dim, 256, 1);
            kl_silu(r, d_t_embed, dit_dim);
            kl_gemm(r, d_t_embed_silu, r->dit_t_mlp_w2, d_t_embed, r->dit_t_mlp_b2, dit_dim, dit_dim, 1);
            hipMemcpyAsync(d_t_embed, d_t_embed_silu, (size_t)dit_dim * sizeof(float),
                            hipMemcpyDeviceToDevice, r->stream);
            kl_silu(r, d_t_embed_silu, dit_dim);
        }

        /* 5d. Upload low-res position indices for RoPE */
        {
            int *py = (int *)malloc((size_t)dit_nt_lo * sizeof(int));
            int *px = (int *)malloc((size_t)dit_nt_lo * sizeof(int));
            generate_grid_pos(py, px, dit_gH_lo, dit_gW_lo);
            hipMemcpyAsync(d_pos_y, py, (size_t)dit_nt_lo * sizeof(int), hipMemcpyHostToDevice, r->stream);
            hipMemcpyAsync(d_pos_x, px, (size_t)dit_nt_lo * sizeof(int), hipMemcpyHostToDevice, r->stream);
            free(py); free(px);
        }

        /* 5e. DiT blocks 0-11 (low-res stage) */
        {
            int dim = dit_dim;
            int nt = dit_nt_lo;
            int n_heads = r->dit_n_heads;
            int head_dim = r->dit_head_dim;
            int stride_3dim = 3 * dim;

            struct timespec ts_lo;
            double t_lo0 = 0;
            if (step == 0 && r->verbose >= 2) {
                hipStreamSynchronize(r->stream);
                clock_gettime(CLOCK_MONOTONIC, &ts_lo);
                t_lo0 = ts_lo.tv_sec + ts_lo.tv_nsec * 1e-9;
            }

            for (int L = 0; L < 12; L++) {
                dit_layer *ly = &r->dit_layers[L];

                kl_gemm(r, d_modulation, ly->adaln_w, d_t_embed_silu, ly->adaln_b,
                        6 * dim, dim, 1);

                void *shift_msa = d_modulation;
                void *scale_msa = (char *)d_modulation + (size_t)1 * dim * sizeof(float);
                void *gate_msa  = (char *)d_modulation + (size_t)2 * dim * sizeof(float);
                void *shift_mlp = (char *)d_modulation + (size_t)3 * dim * sizeof(float);
                void *scale_mlp = (char *)d_modulation + (size_t)4 * dim * sizeof(float);
                void *gate_mlp  = (char *)d_modulation + (size_t)5 * dim * sizeof(float);

                kl_adaln_modulate(r, d_ln_buf, d_hidden, ly->ln1_w, ly->ln1_b,
                                  shift_msa, scale_msa, nt, dim);

                kl_backbone_gemm(r, d_qkv, ly->attn_qkv_w, d_ln_buf, ly->attn_qkv_b,
                                 ly->qkv_rows, ly->qkv_cols, nt);

                kl_qk_norm(r, d_qkv, ly->attn_q_norm_w, ly->attn_q_norm_b,
                           nt, n_heads, head_dim, stride_3dim);
                {
                    void *k_base = (char *)d_qkv + (size_t)dim * sizeof(float);
                    kl_qk_norm(r, k_base, ly->attn_k_norm_w, ly->attn_k_norm_b,
                               nt, n_heads, head_dim, stride_3dim);
                }

                kl_rope_2d_dit(r, d_qkv, d_pos_y, d_pos_x,
                               nt, n_heads, head_dim, stride_3dim, r->dit_rope_freq);
                {
                    void *k_base = (char *)d_qkv + (size_t)dim * sizeof(float);
                    kl_rope_2d_dit(r, k_base, d_pos_y, d_pos_x,
                                   nt, n_heads, head_dim, stride_3dim, r->dit_rope_freq);
                }

                {
                    void *K_t = d_ffn_buf;
                    void *V_t = (char *)d_ffn_buf + (size_t)nt * dim * sizeof(uint16_t);
                    kl_kv_transpose_f16(r, K_t, V_t, d_qkv, nt, dim, n_heads, head_dim);
                    if (L == 0 && ev_attn_start) hipEventRecord(ev_attn_start, r->stream);
                    kl_attn_f16kv(r, d_attn_out, d_qkv, K_t, V_t, nt, dim, n_heads, head_dim);
                    if (L == 0 && ev_attn_stop)  hipEventRecord(ev_attn_stop,  r->stream);
                }

                kl_backbone_gemm(r, d_proj_out, ly->attn_out_w, d_attn_out, ly->attn_out_b,
                                 ly->out_rows, ly->out_cols, nt);

                kl_gate_residual_add(r, d_hidden, d_proj_out, gate_msa, dim, nt * dim);

                kl_adaln_modulate(r, d_ln_buf, d_hidden, ly->ln2_w, ly->ln2_b,
                                  shift_mlp, scale_mlp, nt, dim);

                kl_backbone_gemm(r, d_ffn_buf, ly->mlp_fc1_w, d_ln_buf, ly->mlp_fc1_b,
                                 ly->fc1_rows, ly->fc1_cols, nt);
                kl_gelu(r, d_ffn_buf, nt * ly->fc1_rows);
                kl_backbone_gemm(r, d_proj_out, ly->mlp_fc2_w, d_ffn_buf, ly->mlp_fc2_b,
                                 ly->fc2_rows, ly->fc2_cols, nt);

                kl_gate_residual_add(r, d_hidden, d_proj_out, gate_mlp, dim, nt * dim);
            }

            if (step == 0 && r->verbose >= 2) {
                float ms_lo_attn = 0.0f;
                if (ev_attn_stop) {
                    hipEventSynchronize(ev_attn_stop);
                    hipEventElapsedTime(&ms_lo_attn, ev_attn_start, ev_attn_stop);
                }
                hipStreamSynchronize(r->stream);
                clock_gettime(CLOCK_MONOTONIC, &ts_lo);
                double t_lo1 = ts_lo.tv_sec + ts_lo.tv_nsec * 1e-9;
                fprintf(stderr, "hip_ppd: [profile] lo-res blocks (12 blocks, nt=%d): %.1f ms"
                        "  (L0 attn=%.2f ms, est. 12-blk attn=%.1f ms)\n",
                        nt, (t_lo1 - t_lo0) * 1000.0, ms_lo_attn, ms_lo_attn * 12.0f);
            }
        }

        /* 5f. Semantic fusion at midpoint */
        {
            int dim = dit_dim;
            int nt_lo = dit_nt_lo;

            kl_concat_tokens(r, d_concat, d_hidden, d_semantics, nt_lo, dim, dim);

            kl_backbone_gemm(r, d_fusion_out, r->dit_proj_fusion_w[0], d_concat, r->dit_proj_fusion_b[0],
                    4 * dim, 2 * dim, nt_lo);
            kl_silu(r, d_fusion_out, nt_lo * 4 * dim);
            kl_backbone_gemm(r, d_concat, r->dit_proj_fusion_w[1], d_fusion_out, r->dit_proj_fusion_b[1],
                    4 * dim, 4 * dim, nt_lo);
            kl_silu(r, d_concat, nt_lo * 4 * dim);
            kl_backbone_gemm(r, d_fusion_out, r->dit_proj_fusion_w[2], d_concat, r->dit_proj_fusion_b[2],
                    4 * dim, 4 * dim, nt_lo);
            kl_pixel_shuffle_2x(r, d_hidden, d_fusion_out, dit_gH_lo, dit_gW_lo, dim);
        }

        /* 5g. Upload high-res position indices */
        {
            int *py = (int *)malloc((size_t)dit_nt_hi * sizeof(int));
            int *px = (int *)malloc((size_t)dit_nt_hi * sizeof(int));
            generate_grid_pos(py, px, dit_gH_hi, dit_gW_hi);
            hipMemcpyAsync(d_pos_y, py, (size_t)dit_nt_hi * sizeof(int), hipMemcpyHostToDevice, r->stream);
            hipMemcpyAsync(d_pos_x, px, (size_t)dit_nt_hi * sizeof(int), hipMemcpyHostToDevice, r->stream);
            free(py); free(px);
        }

        /* 5h. DiT blocks 12-23 (high-res stage) */
        {
            int dim = dit_dim;
            int nt = dit_nt_hi;
            int n_heads = r->dit_n_heads;
            int head_dim = r->dit_head_dim;
            int stride_3dim = 3 * dim;

            struct timespec ts_hi;
            double t_hi0 = 0;
            if (step == 0 && r->verbose >= 2) {
                hipStreamSynchronize(r->stream);
                clock_gettime(CLOCK_MONOTONIC, &ts_hi);
                t_hi0 = ts_hi.tv_sec + ts_hi.tv_nsec * 1e-9;
            }

            for (int L = 12; L < 24; L++) {
                dit_layer *ly = &r->dit_layers[L];

                kl_gemm(r, d_modulation, ly->adaln_w, d_t_embed_silu, ly->adaln_b,
                        6 * dim, dim, 1);

                void *shift_msa = d_modulation;
                void *scale_msa = (char *)d_modulation + (size_t)1 * dim * sizeof(float);
                void *gate_msa  = (char *)d_modulation + (size_t)2 * dim * sizeof(float);
                void *shift_mlp = (char *)d_modulation + (size_t)3 * dim * sizeof(float);
                void *scale_mlp = (char *)d_modulation + (size_t)4 * dim * sizeof(float);
                void *gate_mlp  = (char *)d_modulation + (size_t)5 * dim * sizeof(float);

                kl_adaln_modulate(r, d_ln_buf, d_hidden, ly->ln1_w, ly->ln1_b,
                                  shift_msa, scale_msa, nt, dim);

                kl_backbone_gemm(r, d_qkv, ly->attn_qkv_w, d_ln_buf, ly->attn_qkv_b,
                                 ly->qkv_rows, ly->qkv_cols, nt);

                kl_qk_norm(r, d_qkv, ly->attn_q_norm_w, ly->attn_q_norm_b,
                           nt, n_heads, head_dim, stride_3dim);
                {
                    void *k_base = (char *)d_qkv + (size_t)dim * sizeof(float);
                    kl_qk_norm(r, k_base, ly->attn_k_norm_w, ly->attn_k_norm_b,
                               nt, n_heads, head_dim, stride_3dim);
                }

                kl_rope_2d_dit(r, d_qkv, d_pos_y, d_pos_x,
                               nt, n_heads, head_dim, stride_3dim, r->dit_rope_freq);
                {
                    void *k_base = (char *)d_qkv + (size_t)dim * sizeof(float);
                    kl_rope_2d_dit(r, k_base, d_pos_y, d_pos_x,
                                   nt, n_heads, head_dim, stride_3dim, r->dit_rope_freq);
                }

                {
                    void *K_t = d_ffn_buf;
                    void *V_t = (char *)d_ffn_buf + (size_t)nt * dim * sizeof(uint16_t);
                    kl_kv_transpose_f16(r, K_t, V_t, d_qkv, nt, dim, n_heads, head_dim);
                    if (L == 12 && ev_gemm_start) hipEventRecord(ev_gemm_start, r->stream);
                    kl_attn_f16kv(r, d_attn_out, d_qkv, K_t, V_t, nt, dim, n_heads, head_dim);
                    if (L == 12 && ev_gemm_stop)  hipEventRecord(ev_gemm_stop,  r->stream);
                }

                kl_backbone_gemm(r, d_proj_out, ly->attn_out_w, d_attn_out, ly->attn_out_b,
                                 ly->out_rows, ly->out_cols, nt);
                kl_gate_residual_add(r, d_hidden, d_proj_out, gate_msa, dim, nt * dim);

                kl_adaln_modulate(r, d_ln_buf, d_hidden, ly->ln2_w, ly->ln2_b,
                                  shift_mlp, scale_mlp, nt, dim);

                kl_backbone_gemm(r, d_ffn_buf, ly->mlp_fc1_w, d_ln_buf, ly->mlp_fc1_b,
                                 ly->fc1_rows, ly->fc1_cols, nt);
                kl_gelu(r, d_ffn_buf, nt * ly->fc1_rows);
                kl_backbone_gemm(r, d_proj_out, ly->mlp_fc2_w, d_ffn_buf, ly->mlp_fc2_b,
                                 ly->fc2_rows, ly->fc2_cols, nt);
                kl_gate_residual_add(r, d_hidden, d_proj_out, gate_mlp, dim, nt * dim);
            }

            if (step == 0 && r->verbose >= 2) {
                float ms_hi_attn = 0.0f;
                if (ev_gemm_stop) {
                    hipEventSynchronize(ev_gemm_stop);
                    hipEventElapsedTime(&ms_hi_attn, ev_gemm_start, ev_gemm_stop);
                }
                hipStreamSynchronize(r->stream);
                clock_gettime(CLOCK_MONOTONIC, &ts_hi);
                double t_hi1 = ts_hi.tv_sec + ts_hi.tv_nsec * 1e-9;
                fprintf(stderr, "hip_ppd: [profile] hi-res blocks (12 blocks, nt=%d): %.1f ms"
                        "  (L12 attn=%.2f ms, est. 12-blk attn=%.1f ms)\n",
                        nt, (t_hi1 - t_hi0) * 1000.0, ms_hi_attn, ms_hi_attn * 12.0f);
            }
        }

        /* 5i. Final layer: adaLN + Linear(dim, 64) + unpatchify */
        {
            int dim = dit_dim;
            int nt = dit_nt_hi;
            int out_ps = 8;

            kl_gemm(r, d_modulation, r->dit_final_ln_w, d_t_embed_silu, r->dit_final_ln_b,
                    2 * dim, dim, 1);
            void *final_shift = d_modulation;
            void *final_scale = (char *)d_modulation + (size_t)dim * sizeof(float);

            kl_adaln_modulate(r, d_ln_buf, d_hidden, r->dit_final_adaln_w, r->dit_final_adaln_b,
                              final_shift, final_scale, nt, dim);

            int out_dim = out_ps * out_ps;
            kl_gemm(r, d_proj_out, r->dit_final_proj_w, d_ln_buf, r->dit_final_proj_b,
                    out_dim, dim, nt);

            kl_unpatchify(r, d_dit_pred, d_proj_out, dit_gH_hi, dit_gW_hi, out_ps);
        }

        /* 5j. Euler step: update latent */
        kl_euler_step(r, d_latent, d_dit_pred, t_ratio, s_ratio, HW);
    }

    hipStreamSynchronize(r->stream);
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "hip_ppd: DiT diffusion (4 Euler steps): %.1f ms\n", (t1-t0)*1000);

    /* ================================================================ */
    /*  Step 6: Output depth                                             */
    /* ================================================================ */

    /* depth = latent + 0.5 */
    kl_add_scalar(r, d_latent, 0.5f, HW);

    /* Resize from proc_h x proc_w to original w x h */
    if (proc_h != h || proc_w != w) {
        void *d_depth_resized = NULL;
        hipMalloc(&d_depth_resized, (size_t)w * h * sizeof(float));
        {
            int n = h * w;
            int grid = (n + 255) / 256;
            int C = 1, Hi = proc_h, Wi = proc_w, Ho = h, Wo = w;
            void *args[] = {&d_depth_resized, &d_latent, &C, &Hi, &Wi, &Ho, &Wo};
            hipModuleLaunchKernel(r->fn_bilinear_upsample_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                                  0, r->stream, args, NULL);
        }
        res.width = w;
        res.height = h;
        res.depth = (float *)malloc((size_t)w * h * sizeof(float));
        hipMemcpy(res.depth, d_depth_resized, (size_t)w * h * sizeof(float), hipMemcpyDeviceToHost);
        hipFree(d_depth_resized);
    } else {
        res.width = w;
        res.height = h;
        res.depth = (float *)malloc((size_t)w * h * sizeof(float));
        hipMemcpy(res.depth, d_latent, (size_t)w * h * sizeof(float), hipMemcpyDeviceToHost);
    }

    /* -- Cleanup scratch buffers -- */
    hipFree(d_hidden);
    hipFree(d_ln_buf);
    hipFree(d_qkv);
    hipFree(d_attn_out);
    hipFree(d_ffn_buf);
    hipFree(d_proj_out);
    hipFree(d_img_norm);
    hipFree(d_cond);
    hipFree(d_latent);
    hipFree(d_dit_input);
    hipFree(d_dit_pred);
    hipFree(d_semantics);
    hipFree(d_t_embed);
    hipFree(d_t_embed_silu);
    hipFree(d_modulation);
    hipFree(d_sin_buf);
    hipFree(d_pos_y);
    hipFree(d_pos_x);
    hipFree(d_concat);
    if (d_fusion_out != d_ffn_buf) hipFree(d_fusion_out);

    if (ev_attn_start) hipEventDestroy(ev_attn_start);
    if (ev_attn_stop)  hipEventDestroy(ev_attn_stop);
    if (ev_gemm_start) hipEventDestroy(ev_gemm_start);
    if (ev_gemm_stop)  hipEventDestroy(ev_gemm_stop);

    if (r->verbose >= 1)
        fprintf(stderr, "hip_ppd: predict done (%dx%d)\n", w, h);

    return res;
}

void hip_ppd_free(hip_ppd_runner *r) {
    if (!r) return;

    /* Free semantic encoder weights */
    if (r->sem_layers) {
        for (int i = 0; i < r->sem_n_blocks; i++) {
            sem_layer *ly = &r->sem_layers[i];
            if (ly->ln1_w) hipFree(ly->ln1_w);
            if (ly->ln1_b) hipFree(ly->ln1_b);
            if (ly->attn_qkv_w) hipFree(ly->attn_qkv_w);
            if (ly->attn_qkv_b) hipFree(ly->attn_qkv_b);
            if (ly->attn_out_w) hipFree(ly->attn_out_w);
            if (ly->attn_out_b) hipFree(ly->attn_out_b);
            if (ly->ln2_w) hipFree(ly->ln2_w);
            if (ly->ln2_b) hipFree(ly->ln2_b);
            if (ly->ffn_up_w) hipFree(ly->ffn_up_w);
            if (ly->ffn_up_b) hipFree(ly->ffn_up_b);
            if (ly->ffn_down_w) hipFree(ly->ffn_down_w);
            if (ly->ffn_down_b) hipFree(ly->ffn_down_b);
            if (ly->ls1_gamma) hipFree(ly->ls1_gamma);
            if (ly->ls2_gamma) hipFree(ly->ls2_gamma);
        }
        free(r->sem_layers);
    }
    if (r->sem_patch_embed_w) hipFree(r->sem_patch_embed_w);
    if (r->sem_patch_embed_b) hipFree(r->sem_patch_embed_b);
    if (r->sem_cls_token) hipFree(r->sem_cls_token);
    if (r->sem_pos_embed) hipFree(r->sem_pos_embed);
    if (r->sem_norm_w) hipFree(r->sem_norm_w);
    if (r->sem_norm_b) hipFree(r->sem_norm_b);
    free(r->sem_pos_embed_host);

    /* Free DiT weights */
    if (r->dit_const_ones) hipFree(r->dit_const_ones);
    if (r->dit_const_zeros) hipFree(r->dit_const_zeros);
    if (r->dit_layers) {
        for (int i = 0; i < r->dit_n_blocks; i++) {
            dit_layer *ly = &r->dit_layers[i];
            /* ln1_w/b, ln2_w/b are shared constant buffers, already freed above */
            if (ly->attn_qkv_w) hipFree(ly->attn_qkv_w);
            if (ly->attn_qkv_b) hipFree(ly->attn_qkv_b);
            if (ly->attn_q_norm_w) hipFree(ly->attn_q_norm_w);
            if (ly->attn_q_norm_b) hipFree(ly->attn_q_norm_b);
            if (ly->attn_k_norm_w) hipFree(ly->attn_k_norm_w);
            if (ly->attn_k_norm_b) hipFree(ly->attn_k_norm_b);
            if (ly->attn_out_w) hipFree(ly->attn_out_w);
            if (ly->attn_out_b) hipFree(ly->attn_out_b);
            if (ly->mlp_fc1_w) hipFree(ly->mlp_fc1_w);
            if (ly->mlp_fc1_b) hipFree(ly->mlp_fc1_b);
            if (ly->mlp_fc2_w) hipFree(ly->mlp_fc2_w);
            if (ly->mlp_fc2_b) hipFree(ly->mlp_fc2_b);
            if (ly->adaln_w) hipFree(ly->adaln_w);
            if (ly->adaln_b) hipFree(ly->adaln_b);
        }
        free(r->dit_layers);
    }
    if (r->dit_x_embed_w) hipFree(r->dit_x_embed_w);
    if (r->dit_x_embed_b) hipFree(r->dit_x_embed_b);
    if (r->dit_t_mlp_w1) hipFree(r->dit_t_mlp_w1);
    if (r->dit_t_mlp_b1) hipFree(r->dit_t_mlp_b1);
    if (r->dit_t_mlp_w2) hipFree(r->dit_t_mlp_w2);
    if (r->dit_t_mlp_b2) hipFree(r->dit_t_mlp_b2);
    for (int i = 0; i < 3; i++) {
        if (r->dit_proj_fusion_w[i]) hipFree(r->dit_proj_fusion_w[i]);
        if (r->dit_proj_fusion_b[i]) hipFree(r->dit_proj_fusion_b[i]);
    }
    if (r->dit_final_ln_w) hipFree(r->dit_final_ln_w);
    if (r->dit_final_ln_b) hipFree(r->dit_final_ln_b);
    if (r->dit_final_proj_w) hipFree(r->dit_final_proj_w);
    if (r->dit_final_proj_b) hipFree(r->dit_final_proj_b);
    if (r->dit_final_adaln_w) hipFree(r->dit_final_adaln_w);
    if (r->dit_final_adaln_b) hipFree(r->dit_final_adaln_b);

    /* Free scratch/reusable buffers */
    if (r->d_img_raw) hipFree(r->d_img_raw);
    if (r->d_result) hipFree(r->d_result);

    if (r->module) hipModuleUnload(r->module);
    if (r->stream) hipStreamDestroy(r->stream);
    if (r->context) hipCtxDestroy(r->context);

    free(r);
}
