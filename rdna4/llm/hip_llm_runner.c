/*
 * hip_llm_runner.c - HIP/ROCm LLM inference via HIPRTC-compiled kernels
 *
 * Compiles with plain gcc (no hipcc). Uses rocew for dynamic HIP/HIPRTC loading.
 * Supports F16 and Q8_0 weights on GPU, F32 compute.
 * Q8_0 uses padded 36-byte blocks (2B scale + 2B pad + 32B INT8) for aligned access.
 * Single-stream sequential kernel launches.
 * Targets RDNA4 (wave size 32).
 */

#include "hip_llm_runner.h"
#include "../rocew.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* transformer.h header-only: gives us qtensor type + dequant declarations */
#include "../../common/ggml_dequant.h"
#include "../../common/transformer.h"

/* hip_runner_common.h: HIP_CHECK, hip_f32_to_f16, hip_upload_raw, hip_compile_kernels */
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#ifdef LLM_HIPBLASLT_ENABLED
#include "mm_blaslt_bridge.h"
#endif

/* ======================================================================== */
/* HIP C kernel source (compiled at runtime via HIPRTC)                     */
/* ======================================================================== */

static const char *hip_kernel_source =
"/* Minimal FP16 support -- no HIP SDK headers needed. */\n"
"/* We treat FP16 as unsigned short and convert via HIP builtins. */\n"
"typedef unsigned short half_raw;\n"
"\n"
"__device__ __forceinline__ float half_to_float(half_raw h) {\n"
"    __half hv;\n"
"    *((unsigned short*)&hv) = h;\n"
"    return __half2float(hv);\n"
"}\n"
"\n"
"extern \"C\" {\n"
"\n"
"/* ---- 1. embed_f16: F16 embedding lookup -> F32 ---- */\n"
"__global__ void embed_f16(float *dst, const half_raw *embd_table, int token_id, int n_embd) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n_embd) {\n"
"        dst[i] = half_to_float(embd_table[(size_t)token_id * n_embd + i]);\n"
"    }\n"
"}\n"
"\n"
"/* ---- 2. rmsnorm_f32: RMSNorm with parallel reduction ---- */\n"
"__global__ void rmsnorm_f32(float *dst, const float *x, const float *w,\n"
"                             int n, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"\n"
"    /* Sum x[i]^2 with stride loop */\n"
"    float sum = 0.0f;\n"
"    for (int i = tid; i < n; i += nthreads) {\n"
"        float v = x[i];\n"
"        sum += v * v;\n"
"    }\n"
"    sdata[tid] = sum;\n"
"    __syncthreads();\n"
"\n"
"    /* Tree reduction */\n"
"    for (int s = nthreads / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    float scale = rsqrtf(sdata[0] / (float)n + eps);\n"
"    for (int i = tid; i < n; i += nthreads) {\n"
"        dst[i] = x[i] * scale * w[i];\n"
"    }\n"
"}\n"
"\n"
"/* ---- 3. matvec_f16_f32: F16 matrix x F32 vector -> F32 ---- */\n"
"/* Each block computes one output row. 256 threads stride over K cols. */\n"
"__global__ void matvec_f16_f32(float *dst, const half_raw *mat, const float *x,\n"
"                                int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"\n"
"    const half_raw *row_ptr = mat + (size_t)row * n_cols;\n"
"    float sum = 0.0f;\n"
"    for (int j = tid; j < n_cols; j += nthreads) {\n"
"        sum += half_to_float(row_ptr[j]) * x[j];\n"
"    }\n"
"\n"
"    /* Warp shuffle reduction */\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        sum += __shfl_down(sum, offset);\n"
"    }\n"
"\n"
"    /* First thread per warp writes to shared memory, then reduce warps */\n"
"    __shared__ float warp_sums[8]; /* up to 256 threads = 8 warps */\n"
"    int warp_id = tid / 32;\n"
"    int lane = tid % 32;\n"
"    if (lane == 0) warp_sums[warp_id] = sum;\n"
"    __syncthreads();\n"
"\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += warp_sums[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 4. qknorm_f32: Per-head RMSNorm for Q/K ---- */\n"
"/* Grid: n_heads blocks, blockDim = head_dim threads (must be power of 2, <= 1024) */\n"
"__global__ void qknorm_f32(float *vec, const float *w, int n_heads, int head_dim, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    float *v = vec + h * head_dim;\n"
"\n"
"    /* Each thread handles one element (head_dim <= 256 typically) */\n"
"    float val = (tid < head_dim) ? v[tid] : 0.0f;\n"
"    sdata[tid] = val * val;\n"
"    __syncthreads();\n"
"\n"
"    /* Reduction */\n"
"    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);\n"
"    if (tid < head_dim) {\n"
"        v[tid] = val * scale * w[tid];\n"
"    }\n"
"}\n"
"\n"
"/* ---- 5. rope_neox_f32: NeoX-style RoPE, pairs (j, j+pair_offset) ---- */\n"
"/* Grid: n_heads blocks, blockDim: half_dim threads */\n"
"/* n_rope_pairs: number of dim pairs to rotate (0 or >= half_dim = rotate all) */\n"
"__global__ void rope_neox_f32(float *vec, int n_heads, int head_dim, int pos,\n"
"                               float freq_base, int n_rope_pairs) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half_dim = head_dim / 2;\n"
"    if (j >= half_dim) return;\n"
"    /* Only rotate first n_rope_pairs dimensions; leave the rest unchanged */\n"
"    if (n_rope_pairs > 0 && j >= n_rope_pairs) return;\n"
"\n"
"    /* Pair offset: n_rope_pairs when partial rotation, half_dim when full */\n"
"    int pair_off = (n_rope_pairs > 0 && n_rope_pairs < half_dim) ? n_rope_pairs : half_dim;\n"
"\n"
"    int rope_dim = (n_rope_pairs > 0) ? 2 * n_rope_pairs : head_dim;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)rope_dim);\n"
"    float theta = (float)pos * freq;\n"
"    float cos_t = cosf(theta);\n"
"    float sin_t = sinf(theta);\n"
"\n"
"    float *v = vec + h * head_dim;\n"
"    float v0 = v[j];\n"
"    float v1 = v[j + pair_off];\n"
"    v[j]             = v0 * cos_t - v1 * sin_t;\n"
"    v[j + pair_off]  = v0 * sin_t + v1 * cos_t;\n"
"}\n"
"\n"
"/* ---- 5b. rope_mrope_f32: M-RoPE with per-section frequencies ---- */\n"
"/* Grid: n_heads blocks, blockDim: half_dim threads */\n"
"/* sections[4] = {temporal, height, width, pad} dimension counts */\n"
"/* For text-only: pos_t = pos_h = pos_w = pos */\n"
"__global__ void rope_mrope_f32(float *vec, int n_heads, int head_dim,\n"
"                                int pos_t, int pos_h, int pos_w,\n"
"                                float freq_base,\n"
"                                int sect0, int sect1, int sect2, int sect3) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half_dim = head_dim / 2;\n"
"    if (j >= half_dim) return;\n"
"\n"
"    /* Determine which section this dimension pair belongs to */\n"
"    int sect_sum = sect0 + sect1 + sect2 + sect3;\n"
"    int rope_dim = 2 * sect_sum;\n"
"    int pos;\n"
"    if (j < sect0) pos = pos_t;\n"
"    else if (j < sect0 + sect1) pos = pos_h;\n"
"    else if (j < sect0 + sect1 + sect2) pos = pos_w;\n"
"    else pos = 0;  /* padding section */\n"
"\n"
"    float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)rope_dim);\n"
"    float theta = (float)pos * freq;\n"
"    float cos_t = cosf(theta);\n"
"    float sin_t = sinf(theta);\n"
"\n"
"    float *v = vec + h * head_dim;\n"
"    float v0 = v[j];\n"
"    float v1 = v[j + half_dim];\n"
"    v[j]             = v0 * cos_t - v1 * sin_t;\n"
"    v[j + half_dim]  = v0 * sin_t + v1 * cos_t;\n"
"}\n"
"\n"
"/* ---- 6. kv_cache_store: Copy K,V into cache at position ---- */\n"
"__global__ void kv_cache_store(float *key_cache, float *value_cache,\n"
"                                const float *k, const float *v,\n"
"                                int position, int kv_dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < kv_dim) {\n"
"        key_cache[(size_t)position * kv_dim + i] = k[i];\n"
"        value_cache[(size_t)position * kv_dim + i] = v[i];\n"
"    }\n"
"}\n"
"\n"
"/* ---- 7. attn_decode_f32: Single-token decode attention with GQA ---- */\n"
"/* Grid: n_heads blocks, blockDim: 256 threads */\n"
"/* Shared memory: seq_len * sizeof(float) for attention scores */\n"
"__global__ void attn_decode_f32(float *out, const float *q,\n"
"                                 const float *key_cache, const float *value_cache,\n"
"                                 int n_heads, int n_kv_heads, int head_dim,\n"
"                                 int kv_dim, int seq_len, float scale) {\n"
"    extern __shared__ float att[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"\n"
"    const float *q_h = q + h * head_dim;\n"
"\n"
"    /* Phase 1: Compute QK^T scores */\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        const float *k_t = key_cache + (size_t)t * kv_dim + kv_h * head_dim;\n"
"        float score = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) {\n"
"            score += q_h[d] * k_t[d];\n"
"        }\n"
"        att[t] = score * scale;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 2: Softmax over att[0..seq_len-1] */\n"
"    /* Find max (parallel reduction in registers) */\n"
"    float local_max = -1e30f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        if (att[t] > local_max) local_max = att[t];\n"
"    }\n"
"    /* Warp reduce max */\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        float other = __shfl_down(local_max, offset);\n"
"        if (other > local_max) local_max = other;\n"
"    }\n"
"    __shared__ float warp_max[8];\n"
"    int warp_id = tid / 32;\n"
"    int lane = tid % 32;\n"
"    if (lane == 0) warp_max[warp_id] = local_max;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float m = warp_max[0];\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 1; w < n_warps; w++) {\n"
"            if (warp_max[w] > m) m = warp_max[w];\n"
"        }\n"
"        warp_max[0] = m;\n"
"    }\n"
"    __syncthreads();\n"
"    float max_val = warp_max[0];\n"
"\n"
"    /* Exp and sum */\n"
"    float local_sum = 0.0f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        float e = expf(att[t] - max_val);\n"
"        att[t] = e;\n"
"        local_sum += e;\n"
"    }\n"
"    /* Warp reduce sum */\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        local_sum += __shfl_down(local_sum, offset);\n"
"    }\n"
"    __shared__ float warp_sum[8];\n"
"    if (lane == 0) warp_sum[warp_id] = local_sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float s = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) s += warp_sum[w];\n"
"        warp_sum[0] = s;\n"
"    }\n"
"    __syncthreads();\n"
"    float inv_sum = 1.0f / warp_sum[0];\n"
"\n"
"    /* Normalize */\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        att[t] *= inv_sum;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 3: Weighted V accumulation: out_h[d] = sum_t att[t] * V[t][d] */\n"
"    float *out_h = out + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += nthreads) {\n"
"        float sum = 0.0f;\n"
"        for (int t = 0; t < seq_len; t++) {\n"
"            sum += att[t] * value_cache[(size_t)t * kv_dim + kv_h * head_dim + d];\n"
"        }\n"
"        out_h[d] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 8. silu_mul_f32: gate = silu(gate) * up ---- */\n"
"__global__ void silu_mul_f32(float *gate, const float *up, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        float g = gate[i];\n"
"        g = g / (1.0f + expf(-g));\n"
"        gate[i] = g * up[i];\n"
"    }\n"
"}\n"
"\n"
"/* ---- 9. add_f32: dst += src (residual connection) ---- */\n"
"__global__ void add_f32(float *dst, const float *src, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        dst[i] += src[i];\n"
"    }\n"
"}\n"
"\n"
"/* ---- 9b. scale_add_f32: dst += scale * src (MoE expert accumulate) ---- */\n"
"__global__ void scale_add_f32(float *dst, const float *src, float scale, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        dst[i] += scale * src[i];\n"
"    }\n"
"}\n"
"\n"
"/* ---- 9c. matvec_f32_f32: F32 matrix x F32 vector -> F32 (for MoE router) ---- */\n"
"__global__ void matvec_f32_f32(float *dst, const float *mat, const float *x,\n"
"                                int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"\n"
"    const float *row_ptr = mat + (size_t)row * n_cols;\n"
"    float sum = 0.0f;\n"
"    for (int j = tid; j < n_cols; j += nthreads) {\n"
"        sum += row_ptr[j] * x[j];\n"
"    }\n"
"\n"
"    /* Warp shuffle reduction */\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        sum += __shfl_down(sum, offset);\n"
"    }\n"
"\n"
"    __shared__ float warp_sums[8];\n"
"    int warp_id = tid / 32;\n"
"    int lane = tid % 32;\n"
"    if (lane == 0) warp_sums[warp_id] = sum;\n"
"    __syncthreads();\n"
"\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += warp_sums[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- dp4a: INT8 dot product (software fallback for RDNA4) ---- */\n"
"__device__ __forceinline__ int dp4a_s8(int a, int b, int c) {\n"
"    signed char *av = (signed char*)&a;\n"
"    signed char *bv = (signed char*)&b;\n"
"    c += (int)av[0] * (int)bv[0];\n"
"    c += (int)av[1] * (int)bv[1];\n"
"    c += (int)av[2] * (int)bv[2];\n"
"    c += (int)av[3] * (int)bv[3];\n"
"    return c;\n"
"}\n"
"\n"
"/* ---- 10. quantize_f32_to_int8: F32 vector -> INT8 + scale ---- */\n"
"/* Single block, 256 threads. Writes scale_out[0] = absmax/127. */\n"
"__global__ void quantize_f32_to_int8(signed char *dst, float *scale_out,\n"
"                                       const float *src, int n) {\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"\n"
"    /* Find absmax via parallel reduction */\n"
"    float local_max = 0.0f;\n"
"    for (int i = tid; i < n; i += nthreads) {\n"
"        float v = src[i];\n"
"        if (v < 0.0f) v = -v;\n"
"        if (v > local_max) local_max = v;\n"
"    }\n"
"    sdata[tid] = local_max;\n"
"    __syncthreads();\n"
"\n"
"    for (int s = nthreads / 2; s > 0; s >>= 1) {\n"
"        if (tid < s && sdata[tid + s] > sdata[tid])\n"
"            sdata[tid] = sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    float absmax = sdata[0];\n"
"    float scale = absmax / 127.0f;\n"
"    float inv_scale = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;\n"
"    if (tid == 0) scale_out[0] = scale;\n"
"\n"
"    /* Quantize with rounding */\n"
"    for (int i = tid; i < n; i += nthreads) {\n"
"        float fv = src[i] * inv_scale;\n"
"        int iv = (int)rintf(fv);\n"
"        if (iv > 127) iv = 127;\n"
"        if (iv < -128) iv = -128;\n"
"        dst[i] = (signed char)iv;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 11. matvec_q8_0_dp4a: Q8_0 matrix x INT8 vector -> F32 via dp4a ---- */\n"
"/* Padded Q8_0 block: 36 bytes = [uint16 d_half][2B pad][int8 qs[32]] */\n"
"/* 4-byte aligned: qs data at offset 4 is always int-aligned. */\n"
"/* Each block computes one output row. 256 threads stride over Q8_0 blocks. */\n"
"__global__ void matvec_q8_0_dp4a(float *dst, const unsigned char *mat,\n"
"                                   const signed char *x_q, const float *x_scale_ptr,\n"
"                                   int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"\n"
"    int n_blocks_per_row = n_cols / 32;\n"
"    int row_bytes = n_blocks_per_row * 36;  /* padded stride */\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float x_scale = *x_scale_ptr;\n"
"\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < n_blocks_per_row; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 36;\n"
"\n"
"        /* Block scale (FP16 -> F32) */\n"
"        half_raw d_half = *(const half_raw *)bp;\n"
"        float block_scale = half_to_float(d_half);\n"
"\n"
"        /* 8 dp4a calls for 32 INT8 elements (4-byte aligned reads) */\n"
"        const int *w4 = (const int *)(bp + 4);  /* qs at offset 4 = aligned */\n"
"        const int *x4 = (const int *)(x_q + b * 32);\n"
"        int acc = 0;\n"
"        acc = dp4a_s8(w4[0], x4[0], acc);\n"
"        acc = dp4a_s8(w4[1], x4[1], acc);\n"
"        acc = dp4a_s8(w4[2], x4[2], acc);\n"
"        acc = dp4a_s8(w4[3], x4[3], acc);\n"
"        acc = dp4a_s8(w4[4], x4[4], acc);\n"
"        acc = dp4a_s8(w4[5], x4[5], acc);\n"
"        acc = dp4a_s8(w4[6], x4[6], acc);\n"
"        acc = dp4a_s8(w4[7], x4[7], acc);\n"
"\n"
"        sum += (float)acc * block_scale * x_scale;\n"
"    }\n"
"\n"
"    /* Warp shuffle reduction */\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        sum += __shfl_down(sum, offset);\n"
"    }\n"
"\n"
"    __shared__ float warp_sums[8];\n"
"    int warp_id = tid / 32;\n"
"    int lane = tid % 32;\n"
"    if (lane == 0) warp_sums[warp_id] = sum;\n"
"    __syncthreads();\n"
"\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += warp_sums[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 12b. matvec_q8_0_f32: Q8_0 matrix x F32 vector -> F32 (accurate) ---- */\n"
"/* Dequants Q8_0 weights on-the-fly, no input quantization needed. */\n"
"/* Padded block: 36 bytes = [uint16 d_half][2B pad][int8 qs[32]] */\n"
"__global__ void matvec_q8_0_f32(float *dst, const unsigned char *mat,\n"
"                                  const float *x, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"\n"
"    int n_blocks_per_row = n_cols / 32;\n"
"    int row_bytes = n_blocks_per_row * 36;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < n_blocks_per_row; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 36;\n"
"        half_raw d_half = *(const half_raw *)bp;\n"
"        float block_scale = half_to_float(d_half);\n"
"        const signed char *qs = (const signed char *)(bp + 4);\n"
"        const float *xb = x + b * 32;\n"
"\n"
"        float partial = 0.0f;\n"
"        for (int k = 0; k < 32; k++) {\n"
"            partial += (float)qs[k] * xb[k];\n"
"        }\n"
"        sum += partial * block_scale;\n"
"    }\n"
"\n"
"    /* Warp shuffle reduction */\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        sum += __shfl_down(sum, offset);\n"
"    }\n"
"    __shared__ float warp_sums[8];\n"
"    int warp_id = tid / 32;\n"
"    int lane = tid % 32;\n"
"    if (lane == 0) warp_sums[warp_id] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += warp_sums[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 13. embed_q8_0: Padded Q8_0 embedding lookup -> F32 ---- */\n"
"/* Block layout: 36 bytes = [uint16 d_half][2B pad][int8 qs[32]] */\n"
"__global__ void embed_q8_0(float *dst, const unsigned char *embd_table,\n"
"                             int token_id, int n_embd) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n_embd) return;\n"
"\n"
"    int n_blocks_per_row = n_embd / 32;\n"
"    int row_bytes = n_blocks_per_row * 36;  /* padded stride */\n"
"    const unsigned char *row = embd_table + (size_t)token_id * row_bytes;\n"
"\n"
"    int block_idx = i / 32;\n"
"    int block_off = i % 32;\n"
"    const unsigned char *bp = row + block_idx * 36;\n"
"\n"
"    half_raw d_half = *(const half_raw *)bp;\n"
"    float scale = half_to_float(d_half);\n"
"    signed char qs = *((const signed char *)(bp + 4) + block_off);  /* qs at offset 4 */\n"
"    dst[i] = (float)qs * scale;\n"
"}\n"
"\n"
"/* ---- 14. matvec_q2_K_f32: Q2_K matrix x F32 vector -> F32 ---- */\n"
"/* Q2_K block: 84 bytes = scales[16] + qs[64] + d(f16) + dmin(f16), 256 elements */\n"
"__global__ void matvec_q2_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 84;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 84;\n"
"        const unsigned char *scales = bp;\n"
"        const unsigned char *qs = bp + 16;\n"
"        float d = half_to_float(*(const half_raw *)(bp + 80));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 82));\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int is = 0, yi = 0;\n"
"        for (int n0 = 0; n0 < 2; n0++) {\n"
"            for (int j = 0; j < 4; j++) {\n"
"                int shift = j * 2;\n"
"                unsigned char sc = scales[is++];\n"
"                float dl = d * (sc & 0xF);\n"
"                float ml = dmin * (sc >> 4);\n"
"                for (int l = 0; l < 16; l++)\n"
"                    partial += (dl * ((qs[l] >> shift) & 3) - ml) * xb[yi++];\n"
"                sc = scales[is++];\n"
"                dl = d * (sc & 0xF);\n"
"                ml = dmin * (sc >> 4);\n"
"                for (int l = 0; l < 16; l++)\n"
"                    partial += (dl * ((qs[l + 16] >> shift) & 3) - ml) * xb[yi++];\n"
"            }\n"
"            qs += 32;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    __shared__ float warp_sums[8];\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    if (lane == 0) warp_sums[warp_id] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += warp_sums[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 15. matvec_q3_K_f32: Q3_K matrix x F32 vector -> F32 ---- */\n"
"/* Q3_K block: 110 bytes = hmask[32] + qs[64] + scales[12] + d(f16), 256 elements */\n"
"__global__ void matvec_q3_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 110;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        const unsigned char *hm = bp;\n"
"        const unsigned char *qs = bp + 32;\n"
"        const unsigned char *raw_sc = bp + 96;\n"
"        float d_all = half_to_float(*(const half_raw *)(bp + 108));\n"
"        unsigned int a0 = raw_sc[0]|(raw_sc[1]<<8)|(raw_sc[2]<<16)|(raw_sc[3]<<24);\n"
"        unsigned int a1 = raw_sc[4]|(raw_sc[5]<<8)|(raw_sc[6]<<16)|(raw_sc[7]<<24);\n"
"        unsigned int tmp = raw_sc[8]|(raw_sc[9]<<8)|(raw_sc[10]<<16)|(raw_sc[11]<<24);\n"
"        unsigned int km1 = 0x03030303u, km2 = 0x0f0f0f0fu;\n"
"        unsigned int aux[4];\n"
"        aux[0] = (a0 & km2) | (((tmp >> 0) & km1) << 4);\n"
"        aux[1] = (a1 & km2) | (((tmp >> 2) & km1) << 4);\n"
"        aux[2] = ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4);\n"
"        aux[3] = ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4);\n"
"        const signed char *scales = (const signed char *)aux;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int is = 0, yi = 0;\n"
"        unsigned char m_bit = 1;\n"
"        for (int n0 = 0; n0 < 2; n0++) {\n"
"            for (int j = 0; j < 4; j++) {\n"
"                int shift = j * 2;\n"
"                float dl1 = d_all * (scales[is++] - 32);\n"
"                for (int l = 0; l < 16; l++) {\n"
"                    int qv = ((qs[l] >> shift) & 3) - ((hm[l] & m_bit) ? 0 : 4);\n"
"                    partial += dl1 * qv * xb[yi++];\n"
"                }\n"
"                float dl2 = d_all * (scales[is++] - 32);\n"
"                for (int l = 0; l < 16; l++) {\n"
"                    int qv = ((qs[l+16] >> shift) & 3) - ((hm[l+16] & m_bit) ? 0 : 4);\n"
"                    partial += dl2 * qv * xb[yi++];\n"
"                }\n"
"                m_bit <<= 1;\n"
"            }\n"
"            qs += 32;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws3[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws3[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < nw; w++) total += ws3[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 16. matvec_q4_K_f32: Q4_K matrix x F32 vector -> F32 ---- */\n"
"/* Q4_K block: 144 bytes = d(f16) + dmin(f16) + scales[12] + qs[128], 256 elements */\n"
"__global__ void matvec_q4_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 144;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 144;\n"
"        float d = half_to_float(*(const half_raw *)(bp));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *sc = bp + 4;\n"
"        const unsigned char *qs = bp + 16;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        int is = 0;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            unsigned char sv0, mv0, sv1, mv1;\n"
"            if (is < 4) { sv0 = sc[is] & 63; mv0 = sc[is+4] & 63; }\n"
"            else { sv0 = (sc[is+4]&0xF)|((sc[is-4]>>6)<<4); mv0 = (sc[is+4]>>4)|((sc[is]>>6)<<4); }\n"
"            if (is+1 < 4) { sv1 = sc[is+1] & 63; mv1 = sc[is+1+4] & 63; }\n"
"            else { sv1 = (sc[is+1+4]&0xF)|((sc[is+1-4]>>6)<<4); mv1 = (sc[is+1+4]>>4)|((sc[is+1]>>6)<<4); }\n"
"            float d1 = d * sv0, m1 = dmin * mv0;\n"
"            float d2 = d * sv1, m2 = dmin * mv1;\n"
"            const unsigned char *q = qs + j * 32;\n"
"            for (int l = 0; l < 32; l++)\n"
"                partial += (d1 * (q[l] & 0xF) - m1) * xb[yi++];\n"
"            for (int l = 0; l < 32; l++)\n"
"                partial += (d2 * (q[l] >> 4) - m2) * xb[yi++];\n"
"            is += 2;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws4[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws4[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < nw; w++) total += ws4[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 17. matvec_q6_K_f32: Q6_K matrix x F32 vector -> F32 ---- */\n"
"/* Q6_K block: 210 bytes = ql[128] + qh[64] + scales[16] + d(f16), 256 elements */\n"
"__global__ void matvec_q6_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 210;\n"
"        const unsigned char *ql = bp;\n"
"        const unsigned char *qh = bp + 128;\n"
"        const signed char *sc = (const signed char *)(bp + 192);\n"
"        float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        for (int half = 0; half < 2; half++) {\n"
"            int base = half * 128;\n"
"            for (int l = 0; l < 32; l++) {\n"
"                int is = l / 16;\n"
"                int q1 = (int)((ql[l] & 0xF) | (((qh[l]>>0)&3)<<4)) - 32;\n"
"                int q2 = (int)((ql[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)) - 32;\n"
"                int q3 = (int)((ql[l] >> 4) | (((qh[l]>>4)&3)<<4)) - 32;\n"
"                int q4 = (int)((ql[l+32] >> 4) | (((qh[l]>>6)&3)<<4)) - 32;\n"
"                partial += d * sc[is+0] * q1 * xb[base+l];\n"
"                partial += d * sc[is+2] * q2 * xb[base+l+32];\n"
"                partial += d * sc[is+4] * q3 * xb[base+l+64];\n"
"                partial += d * sc[is+6] * q4 * xb[base+l+96];\n"
"            }\n"
"            ql += 64; qh += 32; sc += 8;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws6[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws6[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < nw; w++) total += ws6[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 17b. matvec_q5_K_f32: Q5_K matrix x F32 vector -> F32 ---- */\n"
"/* Q5_K block: 176 bytes = d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128], 256 elements */\n"
"__global__ void matvec_q5_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 176;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 176;\n"
"        float d = half_to_float(*(const half_raw *)(bp));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *sc = bp + 4;\n"
"        const unsigned char *qh = bp + 16;\n"
"        const unsigned char *qs = bp + 48;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        int is = 0;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            unsigned char sv0, mv0, sv1, mv1;\n"
"            if (is < 4) { sv0 = sc[is] & 63; mv0 = sc[is+4] & 63; }\n"
"            else { sv0 = (sc[is+4]&0xF)|((sc[is-4]>>6)<<4); mv0 = (sc[is+4]>>4)|((sc[is]>>6)<<4); }\n"
"            if (is+1 < 4) { sv1 = sc[is+1] & 63; mv1 = sc[is+1+4] & 63; }\n"
"            else { sv1 = (sc[is+1+4]&0xF)|((sc[is+1-4]>>6)<<4); mv1 = (sc[is+1+4]>>4)|((sc[is+1]>>6)<<4); }\n"
"            float d1 = d * sv0, m1 = dmin * mv0;\n"
"            float d2 = d * sv1, m2 = dmin * mv1;\n"
"            const unsigned char *q = qs + j * 32;\n"
"            for (int l = 0; l < 32; l++) {\n"
"                int qhbit = (qh[l] >> (2*j)) & 1;\n"
"                partial += (d1 * ((q[l] & 0xF) | (qhbit << 4)) - m1) * xb[yi++];\n"
"            }\n"
"            for (int l = 0; l < 32; l++) {\n"
"                int qhbit = (qh[l] >> (2*j + 1)) & 1;\n"
"                partial += (d2 * ((q[l] >> 4) | (qhbit << 4)) - m2) * xb[yi++];\n"
"            }\n"
"            is += 2;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws5[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws5[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < nw; w++) total += ws5[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 18. embed_q2_K: Q2_K embedding lookup -> F32 ---- */\n"
"__global__ void embed_q2_K(float *dst, const unsigned char *embd_table,\n"
"                             int token_id, int n_embd) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n_embd) return;\n"
"    int nb_per_row = n_embd / 256;\n"
"    int row_bytes = nb_per_row * 84;\n"
"    const unsigned char *row = embd_table + (size_t)token_id * row_bytes;\n"
"    int block_idx = i / 256;\n"
"    int elem = i % 256;\n"
"    const unsigned char *bp = row + block_idx * 84;\n"
"    const unsigned char *scales = bp;\n"
"    const unsigned char *qs = bp + 16;\n"
"    float d = half_to_float(*(const half_raw *)(bp + 80));\n"
"    float dmin = half_to_float(*(const half_raw *)(bp + 82));\n"
"    int half_idx = elem / 128;\n"
"    int within = elem % 128;\n"
"    int j = within / 32;\n"
"    int sub = (within % 32) / 16;\n"
"    int l = within % 16;\n"
"    int shift = j * 2;\n"
"    int qs_idx = half_idx * 32 + sub * 16 + l;\n"
"    int sc_idx = half_idx * 8 + j * 2 + sub;\n"
"    unsigned char sc = scales[sc_idx];\n"
"    float dl = d * (sc & 0xF);\n"
"    float ml = dmin * (sc >> 4);\n"
"    dst[i] = dl * ((qs[qs_idx] >> shift) & 3) - ml;\n"
"}\n"
"\n"
"/* ==== SSM Delta-Net kernels ==== */\n"
"\n"
"/* ---- 19. softplus_mul_f32: out[i] = softplus(in[i]+bias[i]) * a[i] ---- */\n"
"__global__ void softplus_mul_f32(float *out, const float *in, const float *bias,\n"
"                                  const float *a, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        float x = in[i] + bias[i];\n"
"        float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));\n"
"        out[i] = sp * a[i];\n"
"    }\n"
"}\n"
"\n"
"/* ---- 20. sigmoid_inplace_f32: data[i] = 1/(1+exp(-data[i])) ---- */\n"
"__global__ void sigmoid_inplace_f32(float *data, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) data[i] = 1.0f / (1.0f + expf(-data[i]));\n"
"}\n"
"\n"
"/* ---- 21. conv1d_depthwise_silu_f32: depthwise causal conv1d + SiLU + state update ---- */\n"
"__global__ void conv1d_depthwise_silu_f32(\n"
"    float *conv_out, float *conv_state, const float *input,\n"
"    const float *weight, int qkv_dim, int conv_k) {\n"
"    int j = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (j >= qkv_dim) return;\n"
"    float sum = 0.0f;\n"
"    for (int f = 0; f < conv_k - 1; f++)\n"
"        sum += weight[j * conv_k + f] * conv_state[f * qkv_dim + j];\n"
"    sum += weight[j * conv_k + (conv_k - 1)] * input[j];\n"
"    conv_out[j] = sum / (1.0f + expf(-sum));\n"
"    for (int f = 0; f < conv_k - 2; f++)\n"
"        conv_state[f * qkv_dim + j] = conv_state[(f + 1) * qkv_dim + j];\n"
"    conv_state[(conv_k - 2) * qkv_dim + j] = input[j];\n"
"}\n"
"\n"
"/* ---- 22. l2_norm_heads_f32: per-head L2 normalize ---- */\n"
"__global__ void l2_norm_heads_f32(float *data, int n_heads, int head_dim, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    float *v = data + h * head_dim;\n"
"    float sum = 0.0f;\n"
"    for (int i = tid; i < head_dim; i += blockDim.x) {\n"
"        float x = v[i]; sum += x * x;\n"
"    }\n"
"    sdata[tid] = sum;\n"
"    __syncthreads();\n"
"    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv = rsqrtf(sdata[0] + eps);\n"
"    for (int i = tid; i < head_dim; i += blockDim.x) v[i] *= inv;\n"
"}\n"
"\n"
"/* ---- 23. repeat_tile_f32 ---- */\n"
"__global__ void repeat_tile_f32(float *dst, const float *src,\n"
"                                 int dt_rank, int d_state, int n_group) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = dt_rank * d_state;\n"
"    if (idx >= total) return;\n"
"    int h = idx / d_state;\n"
"    int i = idx % d_state;\n"
"    dst[idx] = src[(h % n_group) * d_state + i];\n"
"}\n"
"\n"
"/* ---- 24. deltanet_step_f32 ---- */\n"
"__global__ void deltanet_step_f32(\n"
"    float *state, float *out,\n"
"    const float *Q, const float *K, const float *V,\n"
"    const float *alpha, const float *beta, int d_state) {\n"
"    int h = blockIdx.x;\n"
"    int r = threadIdx.x;\n"
"    if (r >= d_state) return;\n"
"    float *S = state + (size_t)h * d_state * d_state + r * d_state;\n"
"    const float *q = Q + h * d_state;\n"
"    const float *k = K + h * d_state;\n"
"    float v_r = V[h * d_state + r];\n"
"    float decay = expf(alpha[h]);\n"
"    float b = beta[h];\n"
"    for (int c = 0; c < d_state; c++) S[c] *= decay;\n"
"    float sk = 0.0f;\n"
"    for (int c = 0; c < d_state; c++) sk += S[c] * k[c];\n"
"    float delta = (v_r - sk) * b;\n"
"    for (int c = 0; c < d_state; c++) S[c] += delta * k[c];\n"
"    float scale = rsqrtf((float)d_state);\n"
"    float o = 0.0f;\n"
"    for (int c = 0; c < d_state; c++) o += S[c] * q[c];\n"
"    out[h * d_state + r] = o * scale;\n"
"}\n"
"\n"
"/* ---- 25. gated_rmsnorm_silu_f32 ---- */\n"
"__global__ void gated_rmsnorm_silu_f32(\n"
"    float *out, const float *z, const float *norm_w,\n"
"    int dt_rank, int d_state, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= dt_rank) return;\n"
"    int tid = threadIdx.x;\n"
"    float *o = out + h * d_state;\n"
"    const float *zh = z + h * d_state;\n"
"    float sum = 0.0f;\n"
"    for (int i = tid; i < d_state; i += blockDim.x) {\n"
"        float v = o[i]; sum += v * v;\n"
"    }\n"
"    sdata[tid] = sum;\n"
"    __syncthreads();\n"
"    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float scale = rsqrtf(sdata[0] / (float)d_state + eps);\n"
"    for (int i = tid; i < d_state; i += blockDim.x) {\n"
"        float normed = o[i] * scale * norm_w[i];\n"
"        float zv = zh[i];\n"
"        o[i] = normed * (zv / (1.0f + expf(-zv)));\n"
"    }\n"
"}\n"
"\n"
"/* ---- 26. sigmoid_mul_f32 ---- */\n"
"__global__ void sigmoid_mul_f32(float *data, const float *gate, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) data[i] *= 1.0f / (1.0f + expf(-gate[i]));\n"
"}\n"
"\n"
"/* ---- 27. deinterleave_qgate_f32 ---- */\n"
"__global__ void deinterleave_qgate_f32(\n"
"    float *q_out, float *gate_out, const float *qfull,\n"
"    int n_heads, int head_dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_heads * head_dim;\n"
"    if (idx >= total) return;\n"
"    int h = idx / head_dim;\n"
"    int i = idx % head_dim;\n"
"    q_out[idx]    = qfull[h * 2 * head_dim + i];\n"
"    gate_out[idx] = qfull[h * 2 * head_dim + head_dim + i];\n"
"}\n"
"\n"
"/* Placeholder stubs for less common quant kernels. */\n"
"/* Replace with full implementations from cuda_llm_runner.c as needed */\n"
"/* (just s/__shfl_down_sync(0xFFFFFFFF, X, Y)/__shfl_down(X, Y)/g). */\n"
"\n"
"#define STUB_MATVEC(name) \\\n"
"__global__ void name(float *dst, const unsigned char *mat, const float *x, \\\n"
"                     int n_rows, int n_cols) { \\\n"
"    int row = blockIdx.x; \\\n"
"    if (row >= n_rows) return; \\\n"
"    if (threadIdx.x == 0) dst[row] = 0.0f; \\\n"
"}\n"
"\n"
"STUB_MATVEC(matvec_iq2_xxs_f32)\n"
"STUB_MATVEC(matvec_q4_0_f32)\n"
"STUB_MATVEC(matvec_q4_1_f32)\n"
"STUB_MATVEC(matvec_q5_0_f32)\n"
"STUB_MATVEC(matvec_q5_1_f32)\n"
"STUB_MATVEC(matvec_iq4_nl_f32)\n"
"STUB_MATVEC(matvec_iq4_xs_f32)\n"
"STUB_MATVEC(matvec_iq2_xs_f32)\n"
"STUB_MATVEC(matvec_iq3_xxs_f32)\n"
"STUB_MATVEC(matvec_iq2_s_f32)\n"
"STUB_MATVEC(matvec_iq3_s_f32)\n"
"STUB_MATVEC(matvec_iq1_s_f32)\n"
"STUB_MATVEC(matvec_iq1_m_f32)\n"
"STUB_MATVEC(matvec_tq1_0_f32)\n"
"STUB_MATVEC(matvec_tq2_0_f32)\n"
"\n"
"/* ---- BF16 helpers (used by Phase-2 hipBLASLt prefill GEMMs) ---- */\n"
"typedef unsigned short bf16_raw;\n"
"\n"
"__device__ __forceinline__ bf16_raw f32_to_bf16(float f) {\n"
"    unsigned int u = __float_as_uint(f);\n"
"    /* Round-to-nearest-even */\n"
"    unsigned int rounded = u + ((u >> 16) & 1u) + 0x7FFFu;\n"
"    return (bf16_raw)(rounded >> 16);\n"
"}\n"
"\n"
"/* Vectorized F32 -> BF16 packer. Each thread packs 4 floats; tail is scalar. */\n"
"__global__ void pack_bf16_from_f32(bf16_raw *dst, const float *src, int n) {\n"
"    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int idx4 = gid * 4;\n"
"    if (idx4 + 3 < n) {\n"
"        float4 f4 = *((const float4 *)(src + idx4));\n"
"        unsigned short b0 = f32_to_bf16(f4.x);\n"
"        unsigned short b1 = f32_to_bf16(f4.y);\n"
"        unsigned short b2 = f32_to_bf16(f4.z);\n"
"        unsigned short b3 = f32_to_bf16(f4.w);\n"
"        unsigned long long packed = ((unsigned long long)b0)\n"
"            | (((unsigned long long)b1) << 16)\n"
"            | (((unsigned long long)b2) << 32)\n"
"            | (((unsigned long long)b3) << 48);\n"
"        *((unsigned long long *)(dst + idx4)) = packed;\n"
"    } else {\n"
"        for (int i = idx4; i < n && i < idx4 + 4; i++) {\n"
"            dst[i] = f32_to_bf16(src[i]);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* One-shot weight conversion: F16 [N*K] -> BF16 [N*K]. */\n"
"__global__ void convert_f16_to_bf16(bf16_raw *dst, const half_raw *src, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        dst[i] = f32_to_bf16(half_to_float(src[i]));\n"
"    }\n"
"}\n"
"\n"
"/* ===== Phase 3: causal WMMA flash-attention helpers ===== */\n"
"typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));\n"
"typedef float    float8 __attribute__((ext_vector_type(8)));\n"
"\n"
"__device__ __forceinline__ half_raw f32_to_f16_bits(float v) {\n"
"    __half hv = __float2half(v);\n"
"    return *((half_raw*)&hv);\n"
"}\n"
"\n"
"/* Vectorized F32 -> F16 packer (4 per thread). */\n"
"__global__ void pack_f16_from_f32(half_raw *dst, const float *src, int n) {\n"
"    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int idx = gid * 4;\n"
"    if (idx + 3 < n) {\n"
"        float4 f = *(const float4 *)(src + idx);\n"
"        ushort4 b;\n"
"        b.x = f32_to_f16_bits(f.x);\n"
"        b.y = f32_to_f16_bits(f.y);\n"
"        b.z = f32_to_f16_bits(f.z);\n"
"        b.w = f32_to_f16_bits(f.w);\n"
"        *(ushort4 *)(dst + idx) = b;\n"
"    } else {\n"
"        for (int i = idx; i < n && i < idx + 4; i++) {\n"
"            dst[i] = f32_to_f16_bits(src[i]);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* Pack KV cache F32 [kv_len, n_kv_heads, head_dim] -> F16 [n_kv_heads, kv_len, head_dim].\n"
" * Grid: (kv_len, n_kv_heads), block: head_dim (capped at 256). */\n"
"__global__ void pack_kv_cache_f16(half_raw *dst, const float *src,\n"
"                                   int kv_len, int n_kv_heads, int head_dim) {\n"
"    int t   = blockIdx.x;\n"
"    int hkv = blockIdx.y;\n"
"    int tid = threadIdx.x;\n"
"    int kv_dim = n_kv_heads * head_dim;\n"
"    for (int d = tid; d < head_dim; d += blockDim.x) {\n"
"        float v = src[(size_t)t * kv_dim + hkv * head_dim + d];\n"
"        dst[((size_t)hkv * kv_len + t) * head_dim + d] = f32_to_f16_bits(v);\n"
"    }\n"
"}\n"
"\n"
"/* Batched RoPE NeoX: applies RoPE to M consecutive rows in a [M, n_heads*head_dim]\n"
" * buffer. Grid: (n_heads, M), block: half_dim. */\n"
"__global__ void rope_neox_batch_f32(float *vec_batch, int n_heads, int head_dim,\n"
"                                     int position_start, float freq_base,\n"
"                                     int n_rope_pairs, int row_stride) {\n"
"    int h   = blockIdx.x;\n"
"    int row = blockIdx.y;\n"
"    if (h >= n_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half_dim = head_dim / 2;\n"
"    if (j >= half_dim) return;\n"
"    if (n_rope_pairs > 0 && j >= n_rope_pairs) return;\n"
"    int pair_off = (n_rope_pairs > 0 && n_rope_pairs < half_dim) ? n_rope_pairs : half_dim;\n"
"    int rope_dim = (n_rope_pairs > 0) ? 2 * n_rope_pairs : head_dim;\n"
"    int pos = position_start + row;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)rope_dim);\n"
"    float theta = (float)pos * freq;\n"
"    float cos_t = cosf(theta);\n"
"    float sin_t = sinf(theta);\n"
"    float *v = vec_batch + (size_t)row * row_stride + h * head_dim;\n"
"    float v0 = v[j];\n"
"    float v1 = v[j + pair_off];\n"
"    v[j]            = v0 * cos_t - v1 * sin_t;\n"
"    v[j + pair_off] = v0 * sin_t + v1 * cos_t;\n"
"}\n"
"\n"
"/* Batched M-RoPE; same shape as rope_mrope_f32 but with row dim. */\n"
"__global__ void rope_mrope_batch_f32(float *vec_batch, int n_heads, int head_dim,\n"
"                                      int position_start, float freq_base,\n"
"                                      int sect0, int sect1, int sect2, int sect3,\n"
"                                      int row_stride) {\n"
"    int h   = blockIdx.x;\n"
"    int row = blockIdx.y;\n"
"    if (h >= n_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half_dim = head_dim / 2;\n"
"    if (j >= half_dim) return;\n"
"    int sect_sum = sect0 + sect1 + sect2 + sect3;\n"
"    int rope_dim = 2 * sect_sum;\n"
"    int pos_base = position_start + row;\n"
"    int pos;\n"
"    if (j < sect0) pos = pos_base;\n"
"    else if (j < sect0 + sect1) pos = pos_base;\n"
"    else if (j < sect0 + sect1 + sect2) pos = pos_base;\n"
"    else pos = 0;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)rope_dim);\n"
"    float theta = (float)pos * freq;\n"
"    float cos_t = cosf(theta);\n"
"    float sin_t = sinf(theta);\n"
"    float *v = vec_batch + (size_t)row * row_stride + h * head_dim;\n"
"    float v0 = v[j];\n"
"    float v1 = v[j + half_dim];\n"
"    v[j]            = v0 * cos_t - v1 * sin_t;\n"
"    v[j + half_dim] = v0 * sin_t + v1 * cos_t;\n"
"}\n"
"\n"
"/* Batched KV cache store: copies M rows from k_batch/v_batch into the cache.\n"
" * Grid: ceil((kv_dim*M)/256), block: 256. */\n"
"__global__ void kv_cache_store_batch(float *key_cache, float *value_cache,\n"
"                                      const float *k_batch, const float *v_batch,\n"
"                                      int position_start, int M, int kv_dim) {\n"
"    int total = M * kv_dim;\n"
"    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (gid >= total) return;\n"
"    int row = gid / kv_dim;\n"
"    int i   = gid - row * kv_dim;\n"
"    size_t cache_idx = (size_t)(position_start + row) * kv_dim + i;\n"
"    size_t batch_idx = (size_t)row * kv_dim + i;\n"
"    key_cache[cache_idx]   = k_batch[batch_idx];\n"
"    value_cache[cache_idx] = v_batch[batch_idx];\n"
"}\n"
"\n"
"/* Causal WMMA flash-attention for prefill. BQ=64, 4 waves, F16, GQA, head_dim<=128.\n"
" * Q [M, n_heads*head_dim] F16, K_t/V_t [n_kv_heads, kv_len, head_dim] F16,\n"
" * out [M, n_heads*head_dim] F32.\n"
" * Grid: (n_heads, ceil(M/64)). Block: 128. Shared mem: see launcher. */\n"
"#define FA_LLM_BQ 64\n"
"#define FA_LLM_BKV 16\n"
"#define FA_LLM_HD_MAX 128\n"
"#define FA_LLM_K_NB (FA_LLM_HD_MAX / 16)\n"
"__global__ void flash_attn_wmma_f16_4w_causal(\n"
"    float *out, const half_raw *Q,\n"
"    const half_raw *K_t, const half_raw *V_t,\n"
"    int M, int kv_len, int position_start,\n"
"    int n_heads, int n_kv_heads, int head_dim, float scale) {\n"
"    int h  = blockIdx.x;\n"
"    int qb = blockIdx.y;\n"
"    int q0 = qb * FA_LLM_BQ;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"    int q_dim = n_heads * head_dim;\n"
"    int tid = threadIdx.x;\n"
"    int wid = tid >> 5;\n"
"    int lid = tid & 31;\n"
"    int half = lid >> 4;\n"
"    int idx  = lid & 15;\n"
"    extern __shared__ _Float16 smem_h[];\n"
"    _Float16 *smK0 = smem_h;\n"
"    _Float16 *smK1 = smK0 + 16 * FA_LLM_HD_MAX;\n"
"    _Float16 *smV0 = smK1 + 16 * FA_LLM_HD_MAX;\n"
"    _Float16 *smV1 = smV0 + FA_LLM_HD_MAX * 16;\n"
"    _Float16 *smP  = smV1 + FA_LLM_HD_MAX * 16;\n"
"    _Float16 *smP_w = smP + wid * 16 * 16;\n"
"    int q_base = q0 + wid * 16;\n"
"    /* Load Q tile (per wave: 16 rows, head_dim cols) into registers and apply scale. */\n"
"    f16x8 q_reg[FA_LLM_K_NB];\n"
"    for (int kb = 0; kb < FA_LLM_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int d = kb * 16 + half * 8 + i;\n"
"            int row = q_base + idx;\n"
"            float v = 0.0f;\n"
"            if (row < M && d < head_dim) {\n"
"                half_raw qh = Q[(size_t)row * q_dim + h * head_dim + d];\n"
"                v = half_to_float(qh);\n"
"            }\n"
"            q_reg[kb][i] = (_Float16)(v * scale);\n"
"        }\n"
"    }\n"
"    float8 O_acc[FA_LLM_K_NB];\n"
"    for (int kb = 0; kb < FA_LLM_K_NB; kb++)\n"
"        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"    /* Causal upper bound on KV iterations: max needed k = position_start + q0 + BQ - 1. */\n"
"    int max_q_global = position_start + q0 + FA_LLM_BQ - 1;\n"
"    int max_kv = max_q_global + 1;\n"
"    if (max_kv > kv_len) max_kv = kv_len;\n"
"    int n_kv = (max_kv + FA_LLM_BKV - 1) / FA_LLM_BKV;\n"
"    /* Each thread loads one (row, col-group) of the K/V tile. 128 threads,\n"
"     * 16 KV rows × 8 col-groups (16 cols each). */\n"
"    int ld_row  = tid >> 3;\n"
"    int ld_col0 = (tid & 7) * 16;\n"
"    {\n"
"        int kv = ld_row;\n"
"        size_t k_base = ((size_t)kv_h * kv_len + kv) * head_dim;\n"
"        bool kv_ok = (kv < max_kv);\n"
"        for (int j = 0; j < 16; j++) {\n"
"            int d = ld_col0 + j;\n"
"            _Float16 k_v = (_Float16)0.0f, v_v = (_Float16)0.0f;\n"
"            if (kv_ok && d < head_dim) {\n"
"                k_v = (_Float16)half_to_float(K_t[k_base + d]);\n"
"                v_v = (_Float16)half_to_float(V_t[k_base + d]);\n"
"            }\n"
"            smK0[ld_row * FA_LLM_HD_MAX + d] = k_v;\n"
"            smV0[d * 16 + ld_row] = v_v;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"    for (int t = 0; t < n_kv; t++) {\n"
"        _Float16 *smK_cur = (t & 1) ? smK1 : smK0;\n"
"        _Float16 *smV_cur = (t & 1) ? smV1 : smV0;\n"
"        _Float16 *smK_pre = (t & 1) ? smK0 : smK1;\n"
"        _Float16 *smV_pre = (t & 1) ? smV0 : smV1;\n"
"        int t_next = t + 1;\n"
"        bool prefetch = t_next < n_kv;\n"
"        if (prefetch) {\n"
"            int kv_pre = t_next * FA_LLM_BKV + ld_row;\n"
"            size_t k_base_pre = ((size_t)kv_h * kv_len + kv_pre) * head_dim;\n"
"            bool pre_ok = (kv_pre < max_kv);\n"
"            for (int j = 0; j < 16; j++) {\n"
"                int d = ld_col0 + j;\n"
"                _Float16 k_v = (_Float16)0.0f, v_v = (_Float16)0.0f;\n"
"                if (pre_ok && d < head_dim) {\n"
"                    k_v = (_Float16)half_to_float(K_t[k_base_pre + d]);\n"
"                    v_v = (_Float16)half_to_float(V_t[k_base_pre + d]);\n"
"                }\n"
"                smK_pre[ld_row * FA_LLM_HD_MAX + d] = k_v;\n"
"                smV_pre[d * 16 + ld_row] = v_v;\n"
"            }\n"
"        }\n"
"        int kv0 = t * FA_LLM_BKV;\n"
"        float8 score = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < FA_LLM_K_NB; kb++) {\n"
"            f16x8 b_K;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int dpos = kb * 16 + half * 8 + i;\n"
"                b_K[i] = smK_cur[idx * FA_LLM_HD_MAX + dpos];\n"
"            }\n"
"            score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_reg[kb], b_K, score);\n"
"        }\n"
"        /* Out-of-range KV columns: -inf. */\n"
"        bool col_valid = (kv0 + idx) < max_kv;\n"
"        for (int i = 0; i < 8; i++) score[i] = col_valid ? score[i] : -1e30f;\n"
"        /* Causal mask: for thread (idx, half*8+i):\n"
"         *   q_global = position_start + (q_base + half*8 + i)\n"
"         *   k_global = kv0 + idx\n"
"         * mask if q_global < k_global. */\n"
"        int k_global = kv0 + idx;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int q_row = q_base + half * 8 + i;\n"
"            int q_global = position_start + q_row;\n"
"            if (q_global < k_global) score[i] = -1e30f;\n"
"        }\n"
"        float row_max[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float v = score[i];\n"
"            v = fmaxf(v, __shfl_xor(v, 1, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 2, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 4, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 8, 32));\n"
"            row_max[i] = v;\n"
"        }\n"
"        float alpha[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float new_max = fmaxf(m_i[i], row_max[i]);\n"
"            alpha[i] = __expf(m_i[i] - new_max);\n"
"            float ej = __expf(score[i] - new_max);\n"
"            float s = ej;\n"
"            s += __shfl_xor(s, 1, 32);\n"
"            s += __shfl_xor(s, 2, 32);\n"
"            s += __shfl_xor(s, 4, 32);\n"
"            s += __shfl_xor(s, 8, 32);\n"
"            l_i[i] = l_i[i] * alpha[i] + s;\n"
"            m_i[i] = new_max;\n"
"            score[i] = ej;\n"
"        }\n"
"        for (int kb = 0; kb < FA_LLM_K_NB; kb++)\n"
"            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int n_col = idx;\n"
"            smP_w[m_row * 16 + n_col] = (_Float16)score[i];\n"
"        }\n"
"        f16x8 ap;\n"
"        for (int i = 0; i < 8; i++) ap[i] = smP_w[idx * 16 + half * 8 + i];\n"
"        for (int kb = 0; kb < FA_LLM_K_NB; kb++) {\n"
"            f16x8 bv;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d_col = kb * 16 + idx;\n"
"                int kv_k  = half * 8 + i;\n"
"                bv[i] = smV_cur[d_col * 16 + kv_k];\n"
"            }\n"
"            O_acc[kb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ap, bv, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int kb = 0; kb < FA_LLM_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int d = kb * 16 + idx;\n"
"            int row = q_base + m_row;\n"
"            if (row < M && d < head_dim) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * q_dim + h * head_dim + d] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"#undef FA_LLM_BQ\n"
"#undef FA_LLM_BKV\n"
"#undef FA_LLM_HD_MAX\n"
"#undef FA_LLM_K_NB\n"
"\n"
"/* ===== Phase 4: WMMA F16 matvec for decode (M=1) ===== */\n"
"/* Computes dst[n_rows] = mat[n_rows, n_cols] * x[n_cols] using 16-row WMMA tiles.\n"
" * x is staged into LDS as F16 once. 4 waves per WG, each wave handles its own\n"
" * 16-row tile -> WG covers 64 rows. B operand is x replicated across all 16\n"
" * N columns; we read column 0 of the result.\n"
" * Grid: ceil(n_rows / 64). Block: 128 (4 waves). Shared mem: n_cols * 2 bytes. */\n"
"__global__ void matvec_f16_wmma_f32(float *dst, const half_raw *mat,\n"
"                                     const float *x, int n_rows, int n_cols) {\n"
"    int tid = threadIdx.x;\n"
"    int wave = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int half = lane >> 4;\n"
"    int idx  = lane & 15;\n"
"    int row_tile = blockIdx.x * 4 + wave;\n"
"    int row0 = row_tile * 16;\n"
"    extern __shared__ _Float16 sm_x_h[];\n"
"    /* All threads in WG cooperate to load x to LDS as F16. */\n"
"    int n4 = n_cols & ~3;\n"
"    for (int i = tid * 4; i < n4; i += blockDim.x * 4) {\n"
"        float4 f = *(const float4 *)(x + i);\n"
"        sm_x_h[i + 0] = (_Float16)f.x;\n"
"        sm_x_h[i + 1] = (_Float16)f.y;\n"
"        sm_x_h[i + 2] = (_Float16)f.z;\n"
"        sm_x_h[i + 3] = (_Float16)f.w;\n"
"    }\n"
"    for (int i = n4 + tid; i < n_cols; i += blockDim.x) {\n"
"        sm_x_h[i] = (_Float16)x[i];\n"
"    }\n"
"    __syncthreads();\n"
"    if (row0 >= n_rows) return;\n"
"    int row = row0 + idx;\n"
"    int row_in = (row < n_rows) ? row : 0;\n"
"    const _Float16 *row_h = (const _Float16 *)mat + (size_t)row_in * n_cols;\n"
"    int n_kt = n_cols / 16;\n"
"    float8 acc = {0,0,0,0,0,0,0,0};\n"
"    for (int kt = 0; kt < n_kt; kt++) {\n"
"        /* Vector load: 8 consecutive F16 from this lane's row at kt*16 + half*8. */\n"
"        f16x8 a = *(const f16x8 *)(row_h + kt * 16 + half * 8);\n"
"        f16x8 b = *(const f16x8 *)(sm_x_h + kt * 16 + half * 8);\n"
"        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);\n"
"    }\n"
"    /* Tail: cols not multiple of 16. */\n"
"    int kt_tail = n_kt * 16;\n"
"    if (kt_tail < n_cols) {\n"
"        f16x8 a = {0,0,0,0,0,0,0,0};\n"
"        f16x8 b = {0,0,0,0,0,0,0,0};\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int col = kt_tail + half * 8 + i;\n"
"            if (col < n_cols) {\n"
"                a[i] = row_h[col];\n"
"                b[i] = sm_x_h[col];\n"
"            }\n"
"        }\n"
"        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);\n"
"    }\n"
"    /* acc[i] in lane (idx, half) holds C[M=half*8+i, N=idx]. Take any single N. */\n"
"    if (idx == 0) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = row0 + half * 8 + i;\n"
"            if (row < n_rows) dst[row] = acc[i];\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ===== Phase 5: device-pointer position kernels for HIP graph capture ===== */\n"
"/* These mirror rope_neox_f32 / rope_mrope_f32 / kv_cache_store / attn_decode_f32\n"
" * but read position from a device int so a captured graph can be replayed for\n"
" * any (token_id, position) by writing those scalars to *_p before launch. */\n"
"\n"
"__global__ void rope_neox_f32_devp(float *vec, int n_heads, int head_dim,\n"
"                                    const int *pos_p,\n"
"                                    float freq_base, int n_rope_pairs) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half_dim = head_dim / 2;\n"
"    if (j >= half_dim) return;\n"
"    if (n_rope_pairs > 0 && j >= n_rope_pairs) return;\n"
"    int pair_off = (n_rope_pairs > 0 && n_rope_pairs < half_dim) ? n_rope_pairs : half_dim;\n"
"    int rope_dim = (n_rope_pairs > 0) ? 2 * n_rope_pairs : head_dim;\n"
"    int pos = *pos_p;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)rope_dim);\n"
"    float theta = (float)pos * freq;\n"
"    float cos_t = cosf(theta);\n"
"    float sin_t = sinf(theta);\n"
"    float *v = vec + h * head_dim;\n"
"    float v0 = v[j];\n"
"    float v1 = v[j + pair_off];\n"
"    v[j]            = v0 * cos_t - v1 * sin_t;\n"
"    v[j + pair_off] = v0 * sin_t + v1 * cos_t;\n"
"}\n"
"\n"
"/* M-RoPE text-only: all three position axes share the same scalar. */\n"
"__global__ void rope_mrope_f32_devp(float *vec, int n_heads, int head_dim,\n"
"                                     const int *pos_p,\n"
"                                     float freq_base,\n"
"                                     int sect0, int sect1, int sect2, int sect3) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half_dim = head_dim / 2;\n"
"    if (j >= half_dim) return;\n"
"    int sect_sum = sect0 + sect1 + sect2 + sect3;\n"
"    int rope_dim = 2 * sect_sum;\n"
"    int pos_v = *pos_p;\n"
"    int pos;\n"
"    if (j < sect0) pos = pos_v;\n"
"    else if (j < sect0 + sect1) pos = pos_v;\n"
"    else if (j < sect0 + sect1 + sect2) pos = pos_v;\n"
"    else pos = 0;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)rope_dim);\n"
"    float theta = (float)pos * freq;\n"
"    float cos_t = cosf(theta);\n"
"    float sin_t = sinf(theta);\n"
"    float *v = vec + h * head_dim;\n"
"    float v0 = v[j];\n"
"    float v1 = v[j + half_dim];\n"
"    v[j]            = v0 * cos_t - v1 * sin_t;\n"
"    v[j + half_dim] = v0 * sin_t + v1 * cos_t;\n"
"}\n"
"\n"
"__global__ void kv_cache_store_devp(float *key_cache, float *value_cache,\n"
"                                     const float *k, const float *v,\n"
"                                     const int *pos_p, int kv_dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < kv_dim) {\n"
"        int position = *pos_p;\n"
"        key_cache[(size_t)position * kv_dim + i] = k[i];\n"
"        value_cache[(size_t)position * kv_dim + i] = v[i];\n"
"    }\n"
"}\n"
"\n"
"/* Same as attn_decode_f32 but seq_len comes from *pos_p + 1 at runtime. */\n"
"__global__ void attn_decode_f32_devp(float *out, const float *q,\n"
"                                      const float *key_cache, const float *value_cache,\n"
"                                      int n_heads, int n_kv_heads, int head_dim,\n"
"                                      int kv_dim, const int *pos_p, float scale) {\n"
"    extern __shared__ float att[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"    int seq_len = *pos_p + 1;\n"
"    const float *q_h = q + h * head_dim;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        const float *k_t = key_cache + (size_t)t * kv_dim + kv_h * head_dim;\n"
"        float score = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) score += q_h[d] * k_t[d];\n"
"        att[t] = score * scale;\n"
"    }\n"
"    __syncthreads();\n"
"    float local_max = -1e30f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        if (att[t] > local_max) local_max = att[t];\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        float other = __shfl_down(local_max, offset);\n"
"        if (other > local_max) local_max = other;\n"
"    }\n"
"    __shared__ float warp_max[8];\n"
"    int warp_id = tid / 32;\n"
"    int lane = tid % 32;\n"
"    if (lane == 0) warp_max[warp_id] = local_max;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float m = warp_max[0];\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 1; w < n_warps; w++) if (warp_max[w] > m) m = warp_max[w];\n"
"        warp_max[0] = m;\n"
"    }\n"
"    __syncthreads();\n"
"    float max_val = warp_max[0];\n"
"    float local_sum = 0.0f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        float e = expf(att[t] - max_val);\n"
"        att[t] = e;\n"
"        local_sum += e;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        local_sum += __shfl_down(local_sum, offset);\n"
"    }\n"
"    __shared__ float warp_sum[8];\n"
"    if (lane == 0) warp_sum[warp_id] = local_sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float s = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) s += warp_sum[w];\n"
"        warp_sum[0] = s;\n"
"    }\n"
"    __syncthreads();\n"
"    float inv_sum = 1.0f / warp_sum[0];\n"
"    for (int t = tid; t < seq_len; t += nthreads) att[t] *= inv_sum;\n"
"    __syncthreads();\n"
"    float *out_h = out + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += nthreads) {\n"
"        float acc = 0.0f;\n"
"        for (int t = 0; t < seq_len; t++) {\n"
"            const float *v_t = value_cache + (size_t)t * kv_dim + kv_h * head_dim;\n"
"            acc += att[t] * v_t[d];\n"
"        }\n"
"        out_h[d] = acc;\n"
"    }\n"
"}\n"
"\n"
"} /* extern \"C\" */\n"
;

/* ======================================================================== */
/* Error checking macros                                                    */
/* ======================================================================== */

#define CHECK_HIP(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        const char *errStr = "unknown"; \
        if (hipGetErrorString) hipGetErrorString(err, &errStr); \
        fprintf(stderr, "HIP error at %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, errStr, (int)err); \
        return -1; \
    } \
} while(0)

#define CHECK_HIP_NULL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        const char *errStr = "unknown"; \
        if (hipGetErrorString) hipGetErrorString(err, &errStr); \
        fprintf(stderr, "HIP error at %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, errStr, (int)err); \
        return NULL; \
    } \
} while(0)

#define CHECK_HIP_VOID(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        const char *errStr = "unknown"; \
        if (hipGetErrorString) hipGetErrorString(err, &errStr); \
        fprintf(stderr, "HIP error at %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, errStr, (int)err); \
        return; \
    } \
} while(0)

/* ======================================================================== */
/* Runner state                                                             */
/* ======================================================================== */

/* Per-layer GPU weight pointers */
typedef struct {
    void *attn_norm_w;    /* F32 [n_embd] */
    void *attn_q_w;       /* F16 [n_rows * n_cols] */
    void *attn_k_w;
    void *attn_v_w;
    void *attn_q_bias;    /* F32 [n_embd] Q projection bias (Qwen2.5-VL) */
    void *attn_k_bias;    /* F32 [kv_dim] K projection bias */
    void *attn_v_bias;    /* F32 [kv_dim] V projection bias */
    void *attn_q_norm_w;  /* F32 [head_dim] */
    void *attn_k_norm_w;  /* F32 [head_dim] */
    void *attn_output_w;  /* F16 */
    void *ffn_norm_w;     /* F32 [n_embd] */
    void *ffn_gate_w;     /* F16 */
    void *ffn_up_w;       /* F16 */
    void *ffn_down_w;     /* F16 */

    /* BF16 copies for hipBLASLt prefill GEMMs (Phase 2). NULL if not converted. */
    void *attn_q_w_bf16;
    void *attn_k_w_bf16;
    void *attn_v_w_bf16;
    void *attn_output_w_bf16;
    void *ffn_gate_w_bf16;
    void *ffn_up_w_bf16;
    void *ffn_down_w_bf16;

    /* Dimensions for matvec */
    int attn_q_rows, attn_q_cols;
    int attn_k_rows, attn_k_cols;
    int attn_v_rows, attn_v_cols;
    int attn_output_rows, attn_output_cols;
    int ffn_gate_rows, ffn_gate_cols;
    int ffn_up_rows, ffn_up_cols;
    int ffn_down_rows, ffn_down_cols;

    int has_qk_norm;

    /* Weight types */
    int attn_q_type, attn_k_type, attn_v_type, attn_output_type;
    int ffn_gate_type, ffn_up_type, ffn_down_type;

    /* MoE fields */
    int is_moe;
    void *moe_gate_w;
    int moe_gate_rows, moe_gate_cols;
    void *moe_gate_exps_w;
    void *moe_up_exps_w;
    void *moe_down_exps_w;
    int moe_gate_exps_type, moe_up_exps_type, moe_down_exps_type;
    int moe_exp_rows_gu, moe_exp_cols_gu;
    int moe_exp_rows_d, moe_exp_cols_d;
    size_t moe_exp_stride_gu;
    size_t moe_exp_stride_d;
    void *moe_shared_gate_w;
    void *moe_shared_ffn_gate_w;
    void *moe_shared_ffn_up_w;
    void *moe_shared_ffn_down_w;
    int moe_shared_gate_type, moe_shared_up_type, moe_shared_down_type;
    int moe_shared_gate_rows, moe_shared_gate_cols;
    int moe_shared_up_rows, moe_shared_up_cols;
    int moe_shared_down_rows, moe_shared_down_cols;

    /* SSM (Delta-Net) layer fields */
    int is_ssm;
    void *ssm_qkv_w;
    void *ssm_gate_w;
    void *ssm_alpha_w;
    void *ssm_beta_w;
    void *ssm_out_w;
    void *ssm_a;
    void *ssm_dt_bias;
    void *ssm_conv1d_w;
    void *ssm_norm_w;
    int ssm_qkv_type, ssm_gate_type, ssm_alpha_type, ssm_beta_type, ssm_out_type;
    int ssm_qkv_rows, ssm_qkv_cols;
    int ssm_gate_rows, ssm_gate_cols;
    int ssm_alpha_rows, ssm_alpha_cols;
    int ssm_beta_rows, ssm_beta_cols;
    int ssm_out_rows, ssm_out_cols;

    /* SSM persistent state */
    void *d_conv_state;
    void *d_recurrent_state;
} hip_layer;

struct hip_llm_runner {
    /* HIP context */
    hipDevice_t device;
    hipCtx_t context;
    hipStream_t stream;
    int verbose;

    /* Compiled module + kernels */
    hipModule_t module;
    hipFunction_t fn_embed_f16;
    hipFunction_t fn_rmsnorm_f32;
    hipFunction_t fn_matvec_f16_f32;
    hipFunction_t fn_qknorm_f32;
    hipFunction_t fn_rope_neox_f32;
    hipFunction_t fn_rope_mrope_f32;
    hipFunction_t fn_kv_cache_store;
    hipFunction_t fn_attn_decode_f32;
    hipFunction_t fn_silu_mul_f32;
    hipFunction_t fn_add_f32;
    hipFunction_t fn_quantize_f32_to_int8;
    hipFunction_t fn_matvec_q8_0_dp4a;
    hipFunction_t fn_matvec_q8_0_f32;
    hipFunction_t fn_embed_q8_0;
    hipFunction_t fn_matvec_q2_K_f32;
    hipFunction_t fn_matvec_q3_K_f32;
    hipFunction_t fn_matvec_q4_K_f32;
    hipFunction_t fn_matvec_q5_K_f32;
    hipFunction_t fn_matvec_q6_K_f32;
    hipFunction_t fn_embed_q2_K;
    /* SSM kernels */
    hipFunction_t fn_softplus_mul_f32;
    hipFunction_t fn_sigmoid_inplace_f32;
    hipFunction_t fn_conv1d_depthwise_silu_f32;
    hipFunction_t fn_l2_norm_heads_f32;
    hipFunction_t fn_repeat_tile_f32;
    hipFunction_t fn_deltanet_step_f32;
    hipFunction_t fn_gated_rmsnorm_silu_f32;
    hipFunction_t fn_sigmoid_mul_f32;
    hipFunction_t fn_deinterleave_qgate_f32;
    /* MoE kernels */
    hipFunction_t fn_scale_add_f32;
    hipFunction_t fn_matvec_f32_f32;
    hipFunction_t fn_matvec_iq2_xxs_f32;
    hipFunction_t fn_matvec_q4_0_f32;
    hipFunction_t fn_matvec_q4_1_f32;
    hipFunction_t fn_matvec_q5_0_f32;
    hipFunction_t fn_matvec_q5_1_f32;
    hipFunction_t fn_matvec_iq4_nl_f32;
    hipFunction_t fn_matvec_iq4_xs_f32;
    hipFunction_t fn_matvec_iq2_xs_f32;
    hipFunction_t fn_matvec_iq3_xxs_f32;
    hipFunction_t fn_matvec_iq2_s_f32;
    hipFunction_t fn_matvec_iq3_s_f32;
    hipFunction_t fn_matvec_iq1_s_f32;
    hipFunction_t fn_matvec_iq1_m_f32;
    hipFunction_t fn_matvec_tq1_0_f32;
    hipFunction_t fn_matvec_tq2_0_f32;

    /* Phase 2 BF16 helpers */
    hipFunction_t fn_pack_bf16_from_f32;
    hipFunction_t fn_convert_f16_to_bf16;

    /* Phase 3 flash-attention helpers */
    hipFunction_t fn_pack_f16_from_f32;
    hipFunction_t fn_pack_kv_cache_f16;
    hipFunction_t fn_rope_neox_batch_f32;
    hipFunction_t fn_rope_mrope_batch_f32;
    hipFunction_t fn_kv_cache_store_batch;
    hipFunction_t fn_flash_attn_wmma_f16_4w_causal;

    /* Phase 4 WMMA decode matvec */
    hipFunction_t fn_matvec_f16_wmma_f32;

    /* Model params */
    int n_layers;
    int n_embd;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int n_ff;
    int n_vocab;
    int max_seq_len;
    float rope_freq_base;
    int n_rope_pairs;
    int mrope_sections[4];  /* [temporal, height, width, pad] for M-RoPE */
    int use_mrope;
    float rms_norm_eps;
    int debug_layers;
    int max_layers;
    int n_deepstack;

    /* Hybrid SSM params (Qwen3.5) */
    int is_hybrid;
    int full_attn_interval;
    int ssm_conv_kernel;
    int ssm_d_state;
    int ssm_n_group;
    int ssm_dt_rank;
    int ssm_d_inner;
    int ssm_qkv_dim;

    /* MoE params */
    int is_moe;
    int n_experts;
    int n_experts_used;
    int expert_ff;
    int shared_expert_ff;

    /* MoE scratch buffers */
    void *d_router_logits;
    void *d_moe_accum;
    float *h_router_logits;

    /* GPU weights */
    int token_embd_type;
    void *d_token_embd;
    void *d_output_norm;
    void *d_output_w;
    int output_w_type;
    int has_lm_head;
    hip_layer *layers;

    /* KV cache */
    void **d_key_cache;
    void **d_value_cache;

    /* Scratch buffers on GPU */
    void *d_x;
    void *d_xb;
    void *d_xb2;
    void *d_q;
    void *d_k;
    void *d_v;
    void *d_gate;
    void *d_up;

    /* SSM scratch buffers */
    void *d_ssm_qkv;
    void *d_ssm_z;
    void *d_ssm_alpha;
    void *d_ssm_beta;
    void *d_ssm_Q_exp;
    void *d_ssm_K_exp;
    void *d_ssm_out;
    void *d_ssm_conv_out;
    void *d_attn_gate;

    /* Deepstack scratch */
    void *d_ds_tmp;

    /* INT8 quantization scratch */
    void *d_xb_q;
    void *d_xb_scale;

    /* Host output buffer */
    float *h_output;
    int    h_output_pinned;       /* 1 = hipHostMalloc-pinned (graph D2H requires this) */
    void *d_logits;

    /* Weight loading state */
    int weights_loaded;

    /* Deepstack injection state */
    const float *_ds_embd;
    int _ds_embd_stride;

    /* === Phase 2: batched dense path (hipBLASLt) === */
    int batch_path_ok;          /* 1 if BF16 weights + buffers are ready */
    int batch_max;              /* maximum M for prefill batch */
    int gemm_m_threshold;       /* dispatch to batched path when M >= this */
    /* Activation buffers sized [batch_max, dim]. F32 unless _bf16 suffix. */
    void *d_x_batch;            /* [BMAX, n_embd] F32 */
    void *d_xnorm_batch;        /* [BMAX, n_embd] F32 (post-rmsnorm) */
    void *d_xnorm_batch_bf16;   /* [BMAX, n_embd] BF16 (packed input) */
    void *d_q_batch;            /* [BMAX, n_heads*head_dim] F32 */
    void *d_k_batch;            /* [BMAX, kv_dim] F32 */
    void *d_v_batch;            /* [BMAX, kv_dim] F32 */
    void *d_attn_out_batch;     /* [BMAX, n_heads*head_dim] F32 */
    void *d_attn_out_batch_bf16;
    void *d_attn_proj_batch;    /* [BMAX, n_embd] F32 */
    void *d_ffn_norm_batch_bf16;/* [BMAX, n_embd] BF16 */
    void *d_gate_batch;         /* [BMAX, n_ff] F32 */
    void *d_up_batch;           /* [BMAX, n_ff] F32 */
    void *d_silu_batch_bf16;    /* [BMAX, n_ff] BF16 */
    void *d_down_batch;         /* [BMAX, n_embd] F32 */

    /* === Phase 3: flash-attention scratch === */
    int   fa_path_ok;           /* 1 if FA prefill path is enabled */
    void *d_q_batch_f16;        /* [BMAX, n_heads*head_dim] F16 */
    void *d_kv_pack_K_f16;      /* [n_kv_heads, max_seq_len, head_dim] F16 */
    void *d_kv_pack_V_f16;      /* [n_kv_heads, max_seq_len, head_dim] F16 */

    /* === Phase 4: WMMA decode matvec === */
    int decode_wmma;            /* 1 = use WMMA matvec for F16 weights at decode */

    /* === Phase 5: HIP graph capture for decode === */
    hipFunction_t fn_rope_neox_f32_devp;
    hipFunction_t fn_rope_mrope_f32_devp;
    hipFunction_t fn_kv_cache_store_devp;
    hipFunction_t fn_attn_decode_f32_devp;
    int        *d_position;       /* device int, written via hipMemcpyAsync per token */
    hipGraph_t  graph_logits;     /* captured forward+lm_head pipeline */
    hipGraphExec_t graph_exec_logits;
    hipGraph_t  graph_hidden;     /* captured forward (no lm_head) pipeline */
    hipGraphExec_t graph_exec_hidden;
    int         graph_eligible;   /* 1 if model is graph-capturable (set in load_weights) */
    int         graph_ready_logits;
    int         graph_ready_hidden;
    int         graph_disabled;   /* env LLM_GRAPH_DISABLE=1 */
    int         graph_verbose;    /* env LLM_GRAPH_VERBOSE=1 */
};

/* ======================================================================== */
/* HIPRTC kernel compilation                                                */
/* ======================================================================== */

static int compile_kernels(hip_llm_runner *r) {
    if (hip_compile_kernels(&r->module, r->device, hip_kernel_source,
                             "llm_kernels.hip", r->verbose, "hip_llm") <= 0) {
        return -1;
    }

    /* Look up kernel functions */
    #define GET_FUNC(name) do { \
        hipError_t err = hipModuleGetFunction(&r->fn_##name, r->module, #name); \
        if (err != hipSuccess) { \
            fprintf(stderr, "hip_llm: kernel '%s' not found\n", #name); \
            return -1; \
        } \
    } while(0)

    GET_FUNC(embed_f16);
    GET_FUNC(rmsnorm_f32);
    GET_FUNC(matvec_f16_f32);
    GET_FUNC(qknorm_f32);
    GET_FUNC(rope_neox_f32);
    GET_FUNC(rope_mrope_f32);
    GET_FUNC(kv_cache_store);
    GET_FUNC(attn_decode_f32);
    GET_FUNC(rope_neox_f32_devp);
    GET_FUNC(rope_mrope_f32_devp);
    GET_FUNC(kv_cache_store_devp);
    GET_FUNC(attn_decode_f32_devp);
    GET_FUNC(silu_mul_f32);
    GET_FUNC(add_f32);
    GET_FUNC(quantize_f32_to_int8);
    GET_FUNC(matvec_q8_0_dp4a);
    GET_FUNC(matvec_q8_0_f32);
    GET_FUNC(embed_q8_0);
    GET_FUNC(matvec_q2_K_f32);
    GET_FUNC(matvec_q3_K_f32);
    GET_FUNC(matvec_q4_K_f32);
    GET_FUNC(matvec_q5_K_f32);
    GET_FUNC(matvec_q6_K_f32);
    GET_FUNC(embed_q2_K);
    /* SSM kernels */
    GET_FUNC(softplus_mul_f32);
    GET_FUNC(sigmoid_inplace_f32);
    GET_FUNC(conv1d_depthwise_silu_f32);
    GET_FUNC(l2_norm_heads_f32);
    GET_FUNC(repeat_tile_f32);
    GET_FUNC(deltanet_step_f32);
    GET_FUNC(gated_rmsnorm_silu_f32);
    GET_FUNC(sigmoid_mul_f32);
    GET_FUNC(deinterleave_qgate_f32);
    /* MoE kernels */
    GET_FUNC(scale_add_f32);
    GET_FUNC(matvec_f32_f32);
    GET_FUNC(matvec_iq2_xxs_f32);
    GET_FUNC(matvec_q4_0_f32);
    GET_FUNC(matvec_q4_1_f32);
    GET_FUNC(matvec_q5_0_f32);
    GET_FUNC(matvec_q5_1_f32);
    GET_FUNC(matvec_iq4_nl_f32);
    GET_FUNC(matvec_iq4_xs_f32);
    GET_FUNC(matvec_iq2_xs_f32);
    GET_FUNC(matvec_iq3_xxs_f32);
    GET_FUNC(matvec_iq2_s_f32);
    GET_FUNC(matvec_iq3_s_f32);
    GET_FUNC(matvec_iq1_s_f32);
    GET_FUNC(matvec_iq1_m_f32);
    GET_FUNC(matvec_tq1_0_f32);
    GET_FUNC(matvec_tq2_0_f32);

    GET_FUNC(pack_bf16_from_f32);
    GET_FUNC(convert_f16_to_bf16);

    GET_FUNC(pack_f16_from_f32);
    GET_FUNC(pack_kv_cache_f16);
    GET_FUNC(rope_neox_batch_f32);
    GET_FUNC(rope_mrope_batch_f32);
    GET_FUNC(kv_cache_store_batch);
    GET_FUNC(flash_attn_wmma_f16_4w_causal);

    GET_FUNC(matvec_f16_wmma_f32);

    #undef GET_FUNC

    if (r->verbose >= 1) {
        fprintf(stderr, "hip_llm: all kernels compiled successfully\n");
    }
    return 0;
}

/* ======================================================================== */
/* Public API: init                                                         */
/* ======================================================================== */

hip_llm_runner *hip_llm_init(int device_id, int verbose) {
    /* Initialize HIP + HIPRTC via rocew */
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_llm: rocewInit failed (HIP/HIPRTC libraries not found)\n");
        return NULL;
    }

    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        fprintf(stderr, "hip_llm: hipInit failed\n");
        return NULL;
    }

    hip_llm_runner *r = (hip_llm_runner *)calloc(1, sizeof(hip_llm_runner));
    if (!r) return NULL;
    r->verbose = verbose;
    r->device = device_id;

    CHECK_HIP_NULL(hipSetDevice(device_id));
    r->context = NULL;  /* use default context (no hipCtxCreate) */
    CHECK_HIP_NULL(hipStreamCreateWithFlags(&r->stream, hipStreamNonBlocking));

    if (verbose >= 1) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, device_id);
        fprintf(stderr, "hip_llm: device %d: %s (%.1f GB VRAM)\n",
                device_id, props.name, (double)props.totalGlobalMem / (1024.0*1024.0*1024.0));
    }

    /* Compile kernels */
    if (compile_kernels(r) != 0) {
        hipStreamDestroy(r->stream);
        free(r);
        return NULL;
    }

    return r;
}

/* ======================================================================== */
/* Weight upload helpers                                                    */
/* ======================================================================== */

static int upload_f16_matrix(void **d_ptr, const qtensor *t) {
    if (!t->data) { *d_ptr = NULL; return 0; }
    size_t nbytes = (size_t)t->n_rows * t->n_cols * sizeof(uint16_t);
    hipError_t err = hipMalloc(d_ptr, nbytes);
    if (err != hipSuccess) { fprintf(stderr, "hip_llm: upload_f16_matrix alloc failed (%zu bytes, err=%d)\n", nbytes, (int)err); return -1; }
    err = hipMemcpy(*d_ptr, t->data, nbytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) { fprintf(stderr, "hip_llm: upload_f16_matrix copy failed\n"); hipFree(*d_ptr); *d_ptr = NULL; return -1; }
    return 0;
}

static int upload_norm_f32(void **d_ptr, const qtensor *t, int n) {
    if (!t->data) { *d_ptr = NULL; return 0; }
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) { fprintf(stderr, "hip_llm: upload_norm_f32 malloc failed (n=%d)\n", n); return -1; }
    dequant_row(t->type, t->data, buf, n);
    hipError_t err = hipMalloc(d_ptr, n * sizeof(float));
    if (err != hipSuccess) { fprintf(stderr, "hip_llm: upload_norm_f32 alloc failed (n=%d, err=%d)\n", n, (int)err); free(buf); return -1; }
    err = hipMemcpy(*d_ptr, buf, n * sizeof(float), hipMemcpyHostToDevice);
    free(buf);
    if (err != hipSuccess) { fprintf(stderr, "hip_llm: upload_norm_f32 copy failed\n"); hipFree(*d_ptr); *d_ptr = NULL; return -1; }
    return 0;
}

static int upload_q8_0_raw(void **d_ptr, const qtensor *t) {
    if (!t->data) { *d_ptr = NULL; return 0; }
    int n_elements = t->n_rows * t->n_cols;
    int n_blocks = n_elements / 32;
    size_t nbytes_padded = (size_t)n_blocks * 36;
    uint8_t *padded = (uint8_t *)malloc(nbytes_padded);
    if (!padded) return -1;
    const uint8_t *src = (const uint8_t *)t->data;
    for (int i = 0; i < n_blocks; i++) {
        uint8_t *dst = padded + (size_t)i * 36;
        const uint8_t *s = src + (size_t)i * 34;
        dst[0] = s[0]; dst[1] = s[1]; dst[2] = 0; dst[3] = 0;
        memcpy(dst + 4, s + 2, 32);
    }
    hipError_t err = hipMalloc(d_ptr, nbytes_padded);
    if (err != hipSuccess) { free(padded); return -1; }
    err = hipMemcpy(*d_ptr, padded, nbytes_padded, hipMemcpyHostToDevice);
    free(padded);
    if (err != hipSuccess) { hipFree(*d_ptr); *d_ptr = NULL; return -1; }
    return 0;
}

static int upload_kquant_raw(void **d_ptr, const qtensor *t) {
    if (!t->data) { *d_ptr = NULL; return 0; }
    size_t nbytes = dequant_row_size(t->type, t->n_cols) * (size_t)t->n_rows;
    hipError_t err = hipMalloc(d_ptr, nbytes);
    if (err != hipSuccess) { fprintf(stderr, "hip_llm: upload_kquant_raw alloc failed (%zu bytes, type=%d, err=%d)\n", nbytes, t->type, (int)err); return -1; }
    err = hipMemcpy(*d_ptr, t->data, nbytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) { fprintf(stderr, "hip_llm: upload_kquant_raw copy failed\n"); hipFree(*d_ptr); *d_ptr = NULL; return -1; }
    return 0;
}

/* F32 -> F16 conversion */
static uint16_t hllm_f32_to_f16(float f) {
    union { float f; uint32_t i; } u;
    u.f = f;
    uint32_t x = u.i;
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000);
    int32_t exp = ((x >> 23) & 0xFF) - 127;
    uint32_t mant = x & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) {
        if (exp < -24) return sign;
        mant |= 0x800000;
        mant >>= (-1 - exp);
        return sign | (uint16_t)(mant >> 13);
    }
    return sign | (uint16_t)((exp + 15) << 10) | (uint16_t)(mant >> 13);
}

static int upload_weight_matrix(void **d_ptr, const qtensor *t, int *out_type) {
    *out_type = t->type;
    if (t->type == GGML_TYPE_Q8_0) {
        return upload_q8_0_raw(d_ptr, t);
    } else if (t->type == GGML_TYPE_Q2_K || t->type == GGML_TYPE_Q3_K ||
               t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K ||
               t->type == GGML_TYPE_Q6_K) {
        /* K-quant types with real GPU kernels — upload raw */
        return upload_kquant_raw(d_ptr, t);
    } else if (t->type == GGML_TYPE_BF16) {
        *out_type = GGML_TYPE_F16;
        int n_elements = t->n_rows * t->n_cols;
        float *f32_buf = (float *)malloc((size_t)n_elements * sizeof(float));
        if (!f32_buf) return -1;
        dequant_row(GGML_TYPE_BF16, t->data, f32_buf, n_elements);
        uint16_t *f16_buf = (uint16_t *)malloc((size_t)n_elements * sizeof(uint16_t));
        if (!f16_buf) { free(f32_buf); return -1; }
        for (int i = 0; i < n_elements; i++) f16_buf[i] = hllm_f32_to_f16(f32_buf[i]);
        free(f32_buf);
        size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
        hipError_t err = hipMalloc(d_ptr, nbytes);
        if (err != hipSuccess) { free(f16_buf); return -1; }
        err = hipMemcpy(*d_ptr, f16_buf, nbytes, hipMemcpyHostToDevice);
        free(f16_buf);
        if (err != hipSuccess) { hipFree(*d_ptr); *d_ptr = NULL; return -1; }
        return 0;
    } else {
        /* IQ types + other unsupported quant types — dequant to F32 on CPU, convert to F16, upload */
        int n_elements = t->n_rows * t->n_cols;
        if (n_elements <= 0 || !t->data) {
            fprintf(stderr, "hip_llm: upload_weight_matrix: unhandled type %d, treating as F16 (rows=%d, cols=%d)\n", t->type, t->n_rows, t->n_cols);
            return upload_f16_matrix(d_ptr, t);
        }
        float *f32_buf = (float *)malloc((size_t)n_elements * sizeof(float));
        if (!f32_buf) return -1;
        dequant_row(t->type, t->data, f32_buf, n_elements);
        uint16_t *f16_buf = (uint16_t *)malloc((size_t)n_elements * sizeof(uint16_t));
        if (!f16_buf) { free(f32_buf); return -1; }
        for (int i = 0; i < n_elements; i++) f16_buf[i] = hllm_f32_to_f16(f32_buf[i]);
        free(f32_buf);
        size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
        hipError_t err = hipMalloc(d_ptr, nbytes);
        if (err != hipSuccess) { free(f16_buf); return -1; }
        err = hipMemcpy(*d_ptr, f16_buf, nbytes, hipMemcpyHostToDevice);
        free(f16_buf);
        *out_type = GGML_TYPE_F16;
        if (err != hipSuccess) { hipFree(*d_ptr); *d_ptr = NULL; return -1; }
        return 0;
    }
}

static int upload_3d_kquant_raw(void **d_ptr, const qtensor *t, size_t *out_stride) {
    if (!t->data) { *d_ptr = NULL; return 0; }
    size_t row_bytes = dequant_row_size(t->type, t->n_cols);
    int rows_per_expert = (t->n_dims >= 3) ? (int)t->dims[1] : t->n_rows;
    *out_stride = row_bytes * (size_t)rows_per_expert;
    size_t total_bytes = row_bytes * (size_t)t->n_rows;
    hipError_t err = hipMalloc(d_ptr, total_bytes);
    if (err != hipSuccess) { fprintf(stderr, "hip_llm: upload_3d_kquant alloc failed (%zu bytes, type=%d, err=%d)\n", total_bytes, t->type, (int)err); return -1; }
    err = hipMemcpy(*d_ptr, t->data, total_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) { fprintf(stderr, "hip_llm: upload_3d_kquant copy failed\n"); hipFree(*d_ptr); *d_ptr = NULL; return -1; }
    return 0;
}

/* ======================================================================== */
/* Helper: GGUF metadata readers                                            */
/* ======================================================================== */

static int hllm_get_int(const gguf_context *gguf, const char *key, int def) {
    int idx = gguf_find_key(gguf, key);
    if (idx < 0) return def;
    if (gguf->kv[idx].type == GGUF_TYPE_UINT32) return (int)gguf->kv[idx].value.u32;
    if (gguf->kv[idx].type == GGUF_TYPE_INT32)  return gguf->kv[idx].value.i32;
    return def;
}

static float hllm_get_float(const gguf_context *gguf, const char *key, float def) {
    int idx = gguf_find_key(gguf, key);
    if (idx < 0) return def;
    if (gguf->kv[idx].type == GGUF_TYPE_FLOAT32) return gguf->kv[idx].value.f32;
    return def;
}

static int hllm_find_tensor(const gguf_context *gguf, const char *name) {
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

static qtensor hllm_load_tensor(const gguf_context *gguf, const char *name, int required) {
    qtensor t = {0};
    int idx = hllm_find_tensor(gguf, name);
    if (idx < 0) {
        if (required) fprintf(stderr, "hip_llm: missing tensor '%s'\n", name);
        return t;
    }
    t.data = gguf_tensor_data(gguf, idx);
    t.type = gguf->tensors[idx].type;
    t.n_dims = (int)gguf->tensors[idx].n_dims;
    if (t.n_dims > 4) t.n_dims = 4;
    for (int d = 0; d < t.n_dims; d++) t.dims[d] = gguf->tensors[idx].dims[d];
    t.n_cols = (int)gguf->tensors[idx].dims[0];
    uint64_t n_rows = 1;
    for (uint32_t d = 1; d < gguf->tensors[idx].n_dims; d++)
        n_rows *= gguf->tensors[idx].dims[d];
    t.n_rows = (int)n_rows;
    return t;
}

/* ======================================================================== */
/* Forward-declared minimal launch helpers used during weight load.         */
/* Full launch helpers are defined later, alongside the rest of the kernel  */
/* surface; load_weights only needs the BF16 conversion + packer kernels.   */
/* ======================================================================== */
#ifdef LLM_HIPBLASLT_ENABLED
#define LAUNCH_CONV(fn, gx, gy, gz, bx, by, bz, smem, stream, args) \
    hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, bz, smem, stream, args, NULL)

static void launch_convert_f16_to_bf16_loadtime(hip_llm_runner *r, void *dst,
                                                void *src, int n) {
    void *args[] = { &dst, &src, &n };
    LAUNCH_CONV(r->fn_convert_f16_to_bf16, (n + 255) / 256, 1, 1, 256, 1, 1, 0,
                r->stream, args);
}
#endif

/* ======================================================================== */
/* Public API: load_weights                                                 */
/* ======================================================================== */

/* Phase 5 graph capture: defined far below (needs launch helpers + body). */
static void hip_llm_phase5_capture(hip_llm_runner *r);

int hip_llm_load_weights(hip_llm_runner *r, gguf_context *gguf, int max_seq_len) {
    if (!r || !gguf) return -1;

    const char *arch = "qwen2";
    if (gguf_find_key(gguf, "qwen35moe.block_count") >= 0) arch = "qwen35moe";
    else if (gguf_find_key(gguf, "qwen35.block_count") >= 0) arch = "qwen35";
    else if (gguf_find_key(gguf, "qwen3.block_count") >= 0) arch = "qwen3";
    else if (gguf_find_key(gguf, "qwen3vl.block_count") >= 0) arch = "qwen3vl";
    else if (gguf_find_key(gguf, "qwen2vl.block_count") >= 0) arch = "qwen2vl";

    char kbuf[128];
    #define ARCH_KEY(suffix) (snprintf(kbuf, sizeof(kbuf), "%s." suffix, arch), kbuf)

    r->n_embd      = hllm_get_int(gguf, ARCH_KEY("embedding_length"), 4096);
    r->n_heads     = hllm_get_int(gguf, ARCH_KEY("attention.head_count"), 32);
    r->n_kv_heads  = hllm_get_int(gguf, ARCH_KEY("attention.head_count_kv"), 8);
    r->n_layers    = hllm_get_int(gguf, ARCH_KEY("block_count"), 36);
    r->n_ff        = hllm_get_int(gguf, ARCH_KEY("feed_forward_length"), 12288);
    r->n_vocab     = hllm_get_int(gguf, ARCH_KEY("vocab_size"), 0);
    r->rms_norm_eps = hllm_get_float(gguf, ARCH_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);
    r->rope_freq_base = hllm_get_float(gguf, ARCH_KEY("rope.freq_base"), 5000000.0f);
    r->head_dim    = hllm_get_int(gguf, ARCH_KEY("attention.key_length"), r->n_embd / r->n_heads);

    {
        int rope_dim_count = hllm_get_int(gguf, ARCH_KEY("rope.dimension_count"), 0);
        r->n_rope_pairs = (rope_dim_count > 0) ? rope_dim_count / 2 : 0;
    }

    /* M-RoPE sections (Qwen2-VL, Qwen3-VL) */
    r->use_mrope = 0;
    memset(r->mrope_sections, 0, sizeof(r->mrope_sections));
    {
        int idx = gguf_find_key(gguf, ARCH_KEY("rope.dimension_sections"));
        if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY &&
            gguf->kv[idx].value.arr.type == GGUF_TYPE_INT32) {
            int n = (int)gguf->kv[idx].value.arr.n;
            if (n > 4) n = 4;
            int32_t *data = (int32_t *)gguf->kv[idx].value.arr.data;
            for (int i = 0; i < n; i++) r->mrope_sections[i] = data[i];
            int sect_sum = r->mrope_sections[0] + r->mrope_sections[1] +
                           r->mrope_sections[2] + r->mrope_sections[3];
            if (sect_sum > 0) {
                r->use_mrope = 1;
                fprintf(stderr, "hip_llm: M-RoPE sections=[%d, %d, %d, %d]\n",
                        r->mrope_sections[0], r->mrope_sections[1],
                        r->mrope_sections[2], r->mrope_sections[3]);
            }
        }
    }

    int ctx_len = hllm_get_int(gguf, ARCH_KEY("context_length"), 0);
    if (max_seq_len <= 0) {
        max_seq_len = (ctx_len > 0) ? ctx_len : 1024;
    } else if (ctx_len > 0 && max_seq_len > ctx_len) {
        max_seq_len = ctx_len;
    }
    r->max_seq_len = max_seq_len;

    /* Hybrid SSM (Qwen3.5) */
    r->is_hybrid = 0;
    r->is_moe = 0;
    r->full_attn_interval = 0;
    if (strcmp(arch, "qwen35") == 0 || strcmp(arch, "qwen35moe") == 0) {
        r->is_hybrid = 1;
        r->ssm_conv_kernel = hllm_get_int(gguf, ARCH_KEY("ssm.conv_kernel"), 4);
        r->ssm_d_state     = hllm_get_int(gguf, ARCH_KEY("ssm.state_size"), 128);
        r->ssm_n_group     = hllm_get_int(gguf, ARCH_KEY("ssm.group_count"), 16);
        r->ssm_dt_rank     = hllm_get_int(gguf, ARCH_KEY("ssm.time_step_rank"), 48);
        r->ssm_d_inner     = hllm_get_int(gguf, ARCH_KEY("ssm.inner_size"), 6144);
        r->full_attn_interval = hllm_get_int(gguf, ARCH_KEY("attention.full_attention_interval"), 4);
        r->ssm_qkv_dim = r->ssm_d_state * r->ssm_n_group * 2 + r->ssm_d_inner;
        if (r->verbose >= 1) {
            fprintf(stderr, "hip_llm: hybrid SSM: conv_k=%d d_state=%d n_group=%d dt_rank=%d d_inner=%d interval=%d qkv_dim=%d\n",
                    r->ssm_conv_kernel, r->ssm_d_state, r->ssm_n_group, r->ssm_dt_rank,
                    r->ssm_d_inner, r->full_attn_interval, r->ssm_qkv_dim);
        }
        if (strcmp(arch, "qwen35moe") == 0) {
            r->is_moe = 1;
            r->n_experts       = hllm_get_int(gguf, ARCH_KEY("expert_count"), 256);
            r->n_experts_used  = hllm_get_int(gguf, ARCH_KEY("expert_used_count"), 8);
            r->expert_ff       = hllm_get_int(gguf, ARCH_KEY("expert_feed_forward_length"), 512);
            r->shared_expert_ff = hllm_get_int(gguf, ARCH_KEY("expert_shared_feed_forward_length"), 512);
            r->n_ff = r->expert_ff;
            if (r->verbose >= 1) {
                fprintf(stderr, "hip_llm: MoE: n_experts=%d n_experts_used=%d expert_ff=%d shared_expert_ff=%d\n",
                        r->n_experts, r->n_experts_used, r->expert_ff, r->shared_expert_ff);
            }
        }
    }

    r->n_deepstack = hllm_get_int(gguf, ARCH_KEY("n_deepstack_layers"), 0);
    if (r->verbose >= 1 && r->n_deepstack > 0) {
        fprintf(stderr, "hip_llm: n_deepstack=%d\n", r->n_deepstack);
    }

    #undef ARCH_KEY

    int kv_dim = r->n_kv_heads * r->head_dim;
    int q_dim = r->n_heads * r->head_dim;

    if (r->verbose >= 1) {
        fprintf(stderr, "hip_llm: arch=%s n_embd=%d n_heads=%d n_kv_heads=%d n_layers=%d n_ff=%d head_dim=%d\n",
                arch, r->n_embd, r->n_heads, r->n_kv_heads, r->n_layers, r->n_ff, r->head_dim);
        fprintf(stderr, "hip_llm: rope_freq_base=%.0f rms_norm_eps=%.1e max_seq_len=%d n_rope_pairs=%d\n",
                r->rope_freq_base, r->rms_norm_eps, r->max_seq_len, r->n_rope_pairs);
    }

    /* Token embeddings */
    qtensor embd = hllm_load_tensor(gguf, "token_embd.weight", 1);
    if (!embd.data) return -1;
    if (r->n_vocab == 0) r->n_vocab = embd.n_rows;
    r->token_embd_type = embd.type;
    if (embd.type == GGML_TYPE_Q8_0) {
        if (upload_q8_0_raw(&r->d_token_embd, &embd) != 0) return -1;
    } else if (embd.type == GGML_TYPE_Q2_K) {
        if (upload_kquant_raw(&r->d_token_embd, &embd) != 0) return -1;
    } else if (embd.type == GGML_TYPE_Q3_K || embd.type == GGML_TYPE_Q4_K ||
               embd.type == GGML_TYPE_Q5_K || embd.type == GGML_TYPE_Q6_K) {
        /* Dequant to F16 for embedding lookup (no dedicated Q4_K/Q5_K embed kernel) */
        int n_elements = embd.n_rows * embd.n_cols;
        float *f32_buf = (float *)malloc((size_t)n_elements * sizeof(float));
        if (!f32_buf) return -1;
        for (int row = 0; row < embd.n_rows; row++) {
            const void *row_data = (const uint8_t *)embd.data +
                                    (size_t)row * dequant_row_size(embd.type, embd.n_cols);
            dequant_row(embd.type, row_data, f32_buf + (size_t)row * embd.n_cols, embd.n_cols);
        }
        uint16_t *f16_buf = (uint16_t *)malloc((size_t)n_elements * sizeof(uint16_t));
        if (!f16_buf) { free(f32_buf); return -1; }
        for (int i = 0; i < n_elements; i++) f16_buf[i] = hllm_f32_to_f16(f32_buf[i]);
        free(f32_buf);
        size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
        hipError_t err = hipMalloc(&r->d_token_embd, nbytes);
        if (err != hipSuccess) { free(f16_buf); return -1; }
        err = hipMemcpy(r->d_token_embd, f16_buf, nbytes, hipMemcpyHostToDevice);
        free(f16_buf);
        if (err != hipSuccess) { hipFree(r->d_token_embd); r->d_token_embd = NULL; return -1; }
        r->token_embd_type = GGML_TYPE_F16;
    } else {
        if (upload_f16_matrix(&r->d_token_embd, &embd) != 0) return -1;
    }

    /* Output norm */
    qtensor onorm = hllm_load_tensor(gguf, "output_norm.weight", 1);
    if (!onorm.data) return -1;
    if (upload_norm_f32(&r->d_output_norm, &onorm, r->n_embd) != 0) return -1;

    /* Output projection (lm_head) */
    {
        qtensor output = hllm_load_tensor(gguf, "output.weight", 0);
        if (output.data) {
            if (upload_weight_matrix(&r->d_output_w, &output, &r->output_w_type) != 0) return -1;
            r->has_lm_head = 1;
        } else {
            r->d_output_w = r->d_token_embd;
            r->output_w_type = r->token_embd_type;
            r->has_lm_head = 1;
        }
    }

    /* Per-layer weights */
    r->layers = (hip_layer *)calloc(r->n_layers, sizeof(hip_layer));
    if (!r->layers) return -1;

    for (int l = 0; l < r->n_layers; l++) {
        char name[128];
        hip_layer *cl = &r->layers[l];

        int is_ssm = (r->is_hybrid && r->full_attn_interval > 0 &&
                      (l + 1) % r->full_attn_interval != 0);
        cl->is_ssm = is_ssm;

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        qtensor t = hllm_load_tensor(gguf, name, 1);
        if (upload_norm_f32(&cl->attn_norm_w, &t, r->n_embd) != 0) return -1;

        if (is_ssm) {
            #define LOAD_SSM_W(field, suffix, rows_f, cols_f, type_f) do { \
                snprintf(name, sizeof(name), "blk.%d." suffix ".weight", l); \
                t = hllm_load_tensor(gguf, name, 1); \
                cl->rows_f = t.n_rows; cl->cols_f = t.n_cols; \
                if (upload_weight_matrix(&cl->field, &t, &cl->type_f) != 0) return -1; \
            } while(0)
            LOAD_SSM_W(ssm_qkv_w,   "attn_qkv",  ssm_qkv_rows,   ssm_qkv_cols,   ssm_qkv_type);
            LOAD_SSM_W(ssm_gate_w,   "attn_gate",  ssm_gate_rows,  ssm_gate_cols,  ssm_gate_type);
            LOAD_SSM_W(ssm_alpha_w,  "ssm_alpha",  ssm_alpha_rows, ssm_alpha_cols, ssm_alpha_type);
            LOAD_SSM_W(ssm_beta_w,   "ssm_beta",   ssm_beta_rows,  ssm_beta_cols,  ssm_beta_type);
            LOAD_SSM_W(ssm_out_w,    "ssm_out",    ssm_out_rows,   ssm_out_cols,   ssm_out_type);
            #undef LOAD_SSM_W

            snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ssm_a, &t, r->ssm_dt_rank) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ssm_dt_bias, &t, r->ssm_dt_rank) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ssm_conv1d_w, &t, t.n_rows * t.n_cols) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ssm_norm_w, &t, r->ssm_d_state) != 0) return -1;

            size_t conv_bytes = (size_t)(r->ssm_conv_kernel - 1) * r->ssm_qkv_dim * sizeof(float);
            CHECK_HIP(hipMalloc(&cl->d_conv_state, conv_bytes));
            CHECK_HIP(hipMemset(cl->d_conv_state, 0, conv_bytes));
            size_t rec_bytes = (size_t)r->ssm_dt_rank * r->ssm_d_state * r->ssm_d_state * sizeof(float);
            CHECK_HIP(hipMalloc(&cl->d_recurrent_state, rec_bytes));
            CHECK_HIP(hipMemset(cl->d_recurrent_state, 0, rec_bytes));
        } else {
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->attn_q_rows = t.n_rows; cl->attn_q_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_q_w, &t, &cl->attn_q_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->attn_k_rows = t.n_rows; cl->attn_k_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_k_w, &t, &cl->attn_k_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->attn_v_rows = t.n_rows; cl->attn_v_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_v_w, &t, &cl->attn_v_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->attn_output_rows = t.n_rows; cl->attn_output_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_output_w, &t, &cl->attn_output_type) != 0) return -1;

            /* Attention biases (optional, e.g. Qwen2.5-VL) */
            snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);
            t = hllm_load_tensor(gguf, name, 0);
            if (t.data) {
                if (l == 0) fprintf(stderr, "hip_llm: loading attention biases (Q/K/V)\n");
                if (upload_norm_f32(&cl->attn_q_bias, &t, cl->attn_q_rows) != 0) return -1;
            }
            snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
            t = hllm_load_tensor(gguf, name, 0);
            if (t.data) { if (upload_norm_f32(&cl->attn_k_bias, &t, cl->attn_k_rows) != 0) return -1; }
            snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);
            t = hllm_load_tensor(gguf, name, 0);
            if (t.data) { if (upload_norm_f32(&cl->attn_v_bias, &t, cl->attn_v_rows) != 0) return -1; }

            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 0);
            cl->has_qk_norm = (t.data != NULL);
            if (t.data) { if (upload_norm_f32(&cl->attn_q_norm_w, &t, r->head_dim) != 0) return -1; }
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 0);
            if (t.data) { if (upload_norm_f32(&cl->attn_k_norm_w, &t, r->head_dim) != 0) return -1; }
        }

        /* FFN norm */
        if (r->is_hybrid) {
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l);
        } else {
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        }
        t = hllm_load_tensor(gguf, name, 1);
        if (upload_norm_f32(&cl->ffn_norm_w, &t, r->n_embd) != 0) return -1;

        if (r->is_moe) {
            cl->is_moe = 1;
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_gate_rows = t.n_rows; cl->moe_gate_cols = t.n_cols;
            if (upload_norm_f32(&cl->moe_gate_w, &t, t.n_rows * t.n_cols) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_gate_exps_type = t.type;
            cl->moe_exp_cols_gu = t.n_cols;
            cl->moe_exp_rows_gu = (t.n_dims >= 3) ? (int)t.dims[1] : t.n_rows;
            if (upload_3d_kquant_raw(&cl->moe_gate_exps_w, &t, &cl->moe_exp_stride_gu) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_up_exps_type = t.type;
            if (upload_3d_kquant_raw(&cl->moe_up_exps_w, &t, &cl->moe_exp_stride_gu) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_down_exps_type = t.type;
            cl->moe_exp_cols_d = t.n_cols;
            cl->moe_exp_rows_d = (t.n_dims >= 3) ? (int)t.dims[1] : t.n_rows;
            if (upload_3d_kquant_raw(&cl->moe_down_exps_w, &t, &cl->moe_exp_stride_d) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->moe_shared_gate_w, &t, t.n_rows * t.n_cols) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_shared_gate_rows = t.n_rows; cl->moe_shared_gate_cols = t.n_cols;
            if (upload_weight_matrix(&cl->moe_shared_ffn_gate_w, &t, &cl->moe_shared_gate_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_shared_up_rows = t.n_rows; cl->moe_shared_up_cols = t.n_cols;
            if (upload_weight_matrix(&cl->moe_shared_ffn_up_w, &t, &cl->moe_shared_up_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_shared_down_rows = t.n_rows; cl->moe_shared_down_cols = t.n_cols;
            if (upload_weight_matrix(&cl->moe_shared_ffn_down_w, &t, &cl->moe_shared_down_type) != 0) return -1;
        } else {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->ffn_gate_rows = t.n_rows; cl->ffn_gate_cols = t.n_cols;
            if (upload_weight_matrix(&cl->ffn_gate_w, &t, &cl->ffn_gate_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->ffn_up_rows = t.n_rows; cl->ffn_up_cols = t.n_cols;
            if (upload_weight_matrix(&cl->ffn_up_w, &t, &cl->ffn_up_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->ffn_down_rows = t.n_rows; cl->ffn_down_cols = t.n_cols;
            if (upload_weight_matrix(&cl->ffn_down_w, &t, &cl->ffn_down_type) != 0) return -1;
        }
    }

    /* Allocate KV cache */
    r->d_key_cache = (void **)calloc(r->n_layers, sizeof(void *));
    r->d_value_cache = (void **)calloc(r->n_layers, sizeof(void *));
    size_t kv_cache_size = (size_t)max_seq_len * kv_dim * sizeof(float);
    for (int l = 0; l < r->n_layers; l++) {
        if (r->layers[l].is_ssm) continue;
        CHECK_HIP(hipMalloc(&r->d_key_cache[l], kv_cache_size));
        CHECK_HIP(hipMemset(r->d_key_cache[l], 0, kv_cache_size));
        CHECK_HIP(hipMalloc(&r->d_value_cache[l], kv_cache_size));
        CHECK_HIP(hipMemset(r->d_value_cache[l], 0, kv_cache_size));
    }

    /* Allocate scratch buffers */
    int max_dim = r->n_embd;
    if (q_dim > max_dim) max_dim = q_dim;
    if (r->n_ff > max_dim) max_dim = r->n_ff;
    int xb2_dim = max_dim;
    if (r->is_hybrid) {
        if (r->ssm_qkv_dim > xb2_dim) xb2_dim = r->ssm_qkv_dim;
        if (2 * q_dim > xb2_dim) xb2_dim = 2 * q_dim;
    }

    CHECK_HIP(hipMalloc(&r->d_x,   max_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_xb,  max_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_xb2, xb2_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_q,   q_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_k,   kv_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_v,   kv_dim * sizeof(float)));
    {
        int ff_dim = r->n_ff;
        if (r->is_moe && r->shared_expert_ff > ff_dim) ff_dim = r->shared_expert_ff;
        CHECK_HIP(hipMalloc(&r->d_gate, ff_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_up,   ff_dim * sizeof(float)));
    }

    if (r->is_moe) {
        CHECK_HIP(hipMalloc(&r->d_router_logits, r->n_experts * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_moe_accum, r->n_embd * sizeof(float)));
        r->h_router_logits = (float *)malloc(r->n_experts * sizeof(float));
        if (!r->h_router_logits) return -1;
    }

    if (r->is_hybrid) {
        int d_inner = r->ssm_d_inner;
        int dt_rank = r->ssm_dt_rank;
        int d_state = r->ssm_d_state;
        CHECK_HIP(hipMalloc(&r->d_ssm_qkv,      r->ssm_qkv_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ssm_z,         d_inner * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ssm_alpha,     dt_rank * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ssm_beta,      dt_rank * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ssm_Q_exp,     dt_rank * d_state * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ssm_K_exp,     dt_rank * d_state * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ssm_out,       d_inner * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ssm_conv_out,  r->ssm_qkv_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_attn_gate,     q_dim * sizeof(float)));
    }

    if (r->n_deepstack > 0) {
        CHECK_HIP(hipMalloc(&r->d_ds_tmp, r->n_embd * sizeof(float)));
    }

    CHECK_HIP(hipMalloc(&r->d_xb_q,     max_dim * sizeof(int8_t)));
    CHECK_HIP(hipMalloc(&r->d_xb_scale, sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_logits, (size_t)r->n_vocab * sizeof(float)));

    int out_sz = r->n_vocab > r->n_embd ? r->n_vocab : r->n_embd;
    /* Pinned host memory required by HIP graph captured D2H memcpy nodes. */
    if (hipHostMalloc((void **)&r->h_output, (size_t)out_sz * sizeof(float),
                      hipHostMallocDefault) != hipSuccess) {
        r->h_output = (float *)malloc((size_t)out_sz * sizeof(float));
        if (!r->h_output) return -1;
        r->h_output_pinned = 0;
    } else {
        r->h_output_pinned = 1;
    }

    /* === Phase 5: device int for position, written via hipMemcpyAsync per token === */
    CHECK_HIP(hipMalloc((void **)&r->d_position, sizeof(int)));
    CHECK_HIP(hipMemset(r->d_position, 0, sizeof(int)));

#ifdef LLM_HIPBLASLT_ENABLED
    /* === Phase 2: prepare BF16 weight copies + batched activation buffers
     * for hipBLASLt prefill GEMMs. Only enabled for non-hybrid, non-MoE,
     * F16-weight dense Qwen3-style models. =================================== */
    {
        int batch_max = 512;
        const char *env_bm = getenv("LLM_BMAX");
        if (env_bm) batch_max = atoi(env_bm);
        if (batch_max < 1) batch_max = 1;
        if (batch_max > 4096) batch_max = 4096;
        r->batch_max = batch_max;

        int gemm_thresh = 8;
        const char *env_th = getenv("LLM_GEMM_M_THRESHOLD");
        if (env_th) gemm_thresh = atoi(env_th);
        if (gemm_thresh < 1) gemm_thresh = 1;
        r->gemm_m_threshold = gemm_thresh;

        const char *env_dis = getenv("LLM_BATCH_DISABLE");
        int disabled = (env_dis && atoi(env_dis) != 0);

        /* Eligibility: dense Qwen3-style only. Hybrid SSM and MoE keep the
         * per-token path; they're out of scope for v1 batched GEMM routing. */
        int eligible = !disabled && !r->is_hybrid && !r->is_moe;

        /* Verify all dense F16 weights are present. */
        if (eligible) {
            for (int l = 0; l < r->n_layers; l++) {
                hip_layer *cl = &r->layers[l];
                if (cl->is_ssm || cl->is_moe) { eligible = 0; break; }
                if (cl->attn_q_type   != GGML_TYPE_F16) { eligible = 0; break; }
                if (cl->attn_k_type   != GGML_TYPE_F16) { eligible = 0; break; }
                if (cl->attn_v_type   != GGML_TYPE_F16) { eligible = 0; break; }
                if (cl->attn_output_type != GGML_TYPE_F16) { eligible = 0; break; }
                if (cl->ffn_gate_type != GGML_TYPE_F16) { eligible = 0; break; }
                if (cl->ffn_up_type   != GGML_TYPE_F16) { eligible = 0; break; }
                if (cl->ffn_down_type != GGML_TYPE_F16) { eligible = 0; break; }
            }
        }

        if (eligible) {
            if (mm_blaslt_init() != 0) {
                fprintf(stderr,
                    "hip_llm: mm_blaslt_init failed; falling back to per-token GEMV\n");
                eligible = 0;
            }
        }

        if (eligible) {
            int kv_dim = r->n_kv_heads * r->head_dim;
            int q_dim  = r->n_heads * r->head_dim;

            /* Allocate BF16 weight copies + populate via on-GPU F16->BF16 kernel. */
            for (int l = 0; l < r->n_layers; l++) {
                hip_layer *cl = &r->layers[l];
                size_t n_q  = (size_t)cl->attn_q_rows * cl->attn_q_cols;
                size_t n_k  = (size_t)cl->attn_k_rows * cl->attn_k_cols;
                size_t n_v  = (size_t)cl->attn_v_rows * cl->attn_v_cols;
                size_t n_o  = (size_t)cl->attn_output_rows * cl->attn_output_cols;
                size_t n_g  = (size_t)cl->ffn_gate_rows * cl->ffn_gate_cols;
                size_t n_u  = (size_t)cl->ffn_up_rows   * cl->ffn_up_cols;
                size_t n_d  = (size_t)cl->ffn_down_rows * cl->ffn_down_cols;
                CHECK_HIP(hipMalloc(&cl->attn_q_w_bf16,      n_q * 2));
                CHECK_HIP(hipMalloc(&cl->attn_k_w_bf16,      n_k * 2));
                CHECK_HIP(hipMalloc(&cl->attn_v_w_bf16,      n_v * 2));
                CHECK_HIP(hipMalloc(&cl->attn_output_w_bf16, n_o * 2));
                CHECK_HIP(hipMalloc(&cl->ffn_gate_w_bf16,    n_g * 2));
                CHECK_HIP(hipMalloc(&cl->ffn_up_w_bf16,      n_u * 2));
                CHECK_HIP(hipMalloc(&cl->ffn_down_w_bf16,    n_d * 2));

                launch_convert_f16_to_bf16_loadtime(r, cl->attn_q_w_bf16,      cl->attn_q_w,      (int)n_q);
                launch_convert_f16_to_bf16_loadtime(r, cl->attn_k_w_bf16,      cl->attn_k_w,      (int)n_k);
                launch_convert_f16_to_bf16_loadtime(r, cl->attn_v_w_bf16,      cl->attn_v_w,      (int)n_v);
                launch_convert_f16_to_bf16_loadtime(r, cl->attn_output_w_bf16, cl->attn_output_w, (int)n_o);
                launch_convert_f16_to_bf16_loadtime(r, cl->ffn_gate_w_bf16,    cl->ffn_gate_w,    (int)n_g);
                launch_convert_f16_to_bf16_loadtime(r, cl->ffn_up_w_bf16,      cl->ffn_up_w,      (int)n_u);
                launch_convert_f16_to_bf16_loadtime(r, cl->ffn_down_w_bf16,    cl->ffn_down_w,    (int)n_d);
            }
            hipDeviceSynchronize();

            /* Allocate batched activation buffers. */
            size_t bm = (size_t)batch_max;
            CHECK_HIP(hipMalloc(&r->d_x_batch,            bm * r->n_embd * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_xnorm_batch,        bm * r->n_embd * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_xnorm_batch_bf16,   bm * r->n_embd * 2));
            CHECK_HIP(hipMalloc(&r->d_q_batch,            bm * q_dim    * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_k_batch,            bm * kv_dim   * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_v_batch,            bm * kv_dim   * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_attn_out_batch,     bm * q_dim    * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_attn_out_batch_bf16,bm * q_dim    * 2));
            CHECK_HIP(hipMalloc(&r->d_attn_proj_batch,    bm * r->n_embd * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_ffn_norm_batch_bf16,bm * r->n_embd * 2));
            CHECK_HIP(hipMalloc(&r->d_gate_batch,         bm * r->n_ff  * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_up_batch,           bm * r->n_ff  * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_silu_batch_bf16,    bm * r->n_ff  * 2));
            CHECK_HIP(hipMalloc(&r->d_down_batch,         bm * r->n_embd * sizeof(float)));

            r->batch_path_ok = 1;
            if (r->verbose >= 1) {
                fprintf(stderr,
                    "hip_llm: Phase-2 batched path enabled (BMAX=%d, threshold=%d)\n",
                    batch_max, gemm_thresh);
            }

            /* Phase 3: flash-attention scratch.
             * Requires head_dim <= 128 and (n_heads % n_kv_heads) == 0. */
            int fa_disable = 0;
            const char *fa_disable_env = getenv("LLM_FA_DISABLE");
            if (fa_disable_env) fa_disable = atoi(fa_disable_env);
            int fa_eligible = !fa_disable
                && r->head_dim > 0 && r->head_dim <= 128
                && r->n_kv_heads > 0
                && (r->n_heads % r->n_kv_heads) == 0;
            if (fa_eligible) {
                size_t kv_pack_elems = (size_t)r->n_kv_heads
                                     * (size_t)r->max_seq_len
                                     * (size_t)r->head_dim;
                CHECK_HIP(hipMalloc(&r->d_q_batch_f16,   bm * (size_t)q_dim * 2));
                CHECK_HIP(hipMalloc(&r->d_kv_pack_K_f16, kv_pack_elems * 2));
                CHECK_HIP(hipMalloc(&r->d_kv_pack_V_f16, kv_pack_elems * 2));
                r->fa_path_ok = 1;
                if (r->verbose >= 1) {
                    fprintf(stderr,
                        "hip_llm: Phase-3 flash-attention path enabled (head_dim=%d kv_pack=%.1f MB)\n",
                        r->head_dim,
                        (double)(kv_pack_elems * 2 * 2) / (1024.0 * 1024.0));
                }
            } else {
                r->fa_path_ok = 0;
                if (r->verbose >= 1) {
                    fprintf(stderr,
                        "hip_llm: Phase-3 flash-attention disabled (head_dim=%d, n_heads=%d, n_kv_heads=%d, fa_disable=%d)\n",
                        r->head_dim, r->n_heads, r->n_kv_heads, fa_disable);
                }
            }
        } else {
            r->batch_path_ok = 0;
            if (r->verbose >= 1) {
                fprintf(stderr,
                    "hip_llm: Phase-2 batched path disabled (hybrid/moe/quant or LLM_BATCH_DISABLE)\n");
            }
        }
    }
#endif /* LLM_HIPBLASLT_ENABLED */

    /* === Phase 4: WMMA F16 matvec for decode (gated). =================== */
    r->decode_wmma = 0;
    {
        const char *env_dwmma = getenv("LLM_DECODE_WMMA");
        if (env_dwmma) r->decode_wmma = atoi(env_dwmma) != 0;
        if (r->verbose >= 1 && r->decode_wmma) {
            fprintf(stderr, "hip_llm: Phase-4 WMMA decode matvec enabled\n");
        }
    }

    r->weights_loaded = 1;

    /* === Phase 5: HIP graph capture (deferred — defined below all launch helpers
     * and forward_blocks_body). Captures post-embed pipeline so each decode token
     * issues one hipGraphLaunch instead of ~580 host launches. ============= */
    hip_llm_phase5_capture(r);

    if (r->verbose >= 1) {
        fprintf(stderr, "hip_llm: weights loaded successfully\n");
    }

    return 0;
}

/* ======================================================================== */
/* Kernel launch helpers                                                    */
/* ======================================================================== */

#define LAUNCH(fn, gx, gy, gz, bx, by, bz, smem, stream, args) \
    hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, bz, smem, stream, args, NULL)

static inline void launch_embed(hip_llm_runner *r, void *dst, void *embd_table,
                                int token_id, int n_embd) {
    void *args[] = { &dst, &embd_table, &token_id, &n_embd };
    LAUNCH(r->fn_embed_f16, (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_rmsnorm(hip_llm_runner *r, void *dst, void *x,
                                   void *w, int n, float eps) {
    void *args[] = { &dst, &x, &w, &n, &eps };
    LAUNCH(r->fn_rmsnorm_f32, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), r->stream, args);
}

static inline void launch_matvec(hip_llm_runner *r, void *dst, void *mat,
                                  void *x, int n_rows, int n_cols) {
    /* Phase 4: WMMA path needs n_rows multiple of 16 and n_cols >= 16. */
    if (r->decode_wmma && (n_rows & 15) == 0 && n_cols >= 16) {
        void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
        int n_blocks = (n_rows + 63) / 64;  /* 4 waves × 16 rows per WG */
        int smem = n_cols * (int)sizeof(short); /* sm_x_h is _Float16 */
        LAUNCH(r->fn_matvec_f16_wmma_f32, n_blocks, 1, 1, 128, 1, 1, smem,
               r->stream, args);
        return;
    }
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    LAUNCH(r->fn_matvec_f16_f32, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_qknorm(hip_llm_runner *r, void *vec, void *w,
                                   int n_heads, int head_dim, float eps) {
    int bdim = 1;
    while (bdim < head_dim) bdim <<= 1;
    if (bdim > 256) bdim = 256;
    void *args[] = { &vec, &w, &n_heads, &head_dim, &eps };
    LAUNCH(r->fn_qknorm_f32, n_heads, 1, 1, bdim, 1, 1, bdim * sizeof(float), r->stream, args);
}

static inline void launch_rope(hip_llm_runner *r, void *vec, int n_heads,
                                int head_dim, int pos, float freq_base) {
    if (r->use_mrope) {
        /* M-RoPE: for text-only, all position axes = pos */
        int half_dim = head_dim / 2;
        int s0 = r->mrope_sections[0], s1 = r->mrope_sections[1];
        int s2 = r->mrope_sections[2], s3 = r->mrope_sections[3];
        void *args[] = { &vec, &n_heads, &head_dim, &pos, &pos, &pos,
                         &freq_base, &s0, &s1, &s2, &s3 };
        LAUNCH(r->fn_rope_mrope_f32, n_heads, 1, 1, half_dim, 1, 1, 0, r->stream, args);
    } else {
        int half_dim = head_dim / 2;
        int n_rope_pairs = r->n_rope_pairs;
        void *args[] = { &vec, &n_heads, &head_dim, &pos, &freq_base, &n_rope_pairs };
        LAUNCH(r->fn_rope_neox_f32, n_heads, 1, 1, half_dim, 1, 1, 0, r->stream, args);
    }
}

static inline void launch_kv_store(hip_llm_runner *r, void *key_cache, void *value_cache,
                                    void *k, void *v, int position, int kv_dim) {
    void *args[] = { &key_cache, &value_cache, &k, &v, &position, &kv_dim };
    LAUNCH(r->fn_kv_cache_store, (kv_dim + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_attention(hip_llm_runner *r, void *out, void *q,
                                     void *key_cache, void *value_cache,
                                     int n_heads, int n_kv_heads, int head_dim,
                                     int kv_dim, int seq_len, float scale) {
    size_t smem = seq_len * sizeof(float);
    void *args[] = { &out, &q, &key_cache, &value_cache,
                     &n_heads, &n_kv_heads, &head_dim, &kv_dim, &seq_len, &scale };
    LAUNCH(r->fn_attn_decode_f32, n_heads, 1, 1, 256, 1, 1, smem, r->stream, args);
}

/* === Phase 5: device-pointer launchers (read position from r->d_position) === */
static inline void launch_rope_devp(hip_llm_runner *r, void *vec, int n_heads,
                                     int head_dim, float freq_base) {
    int half_dim = head_dim / 2;
    int *pos_p = r->d_position;
    if (r->use_mrope) {
        int s0 = r->mrope_sections[0], s1 = r->mrope_sections[1];
        int s2 = r->mrope_sections[2], s3 = r->mrope_sections[3];
        void *args[] = { &vec, &n_heads, &head_dim, &pos_p,
                         &freq_base, &s0, &s1, &s2, &s3 };
        LAUNCH(r->fn_rope_mrope_f32_devp, n_heads, 1, 1, half_dim, 1, 1, 0, r->stream, args);
    } else {
        int n_rope_pairs = r->n_rope_pairs;
        void *args[] = { &vec, &n_heads, &head_dim, &pos_p, &freq_base, &n_rope_pairs };
        LAUNCH(r->fn_rope_neox_f32_devp, n_heads, 1, 1, half_dim, 1, 1, 0, r->stream, args);
    }
}

static inline void launch_kv_store_devp(hip_llm_runner *r, void *key_cache, void *value_cache,
                                         void *k, void *v, int kv_dim) {
    int *pos_p = r->d_position;
    void *args[] = { &key_cache, &value_cache, &k, &v, &pos_p, &kv_dim };
    LAUNCH(r->fn_kv_cache_store_devp, (kv_dim + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

/* smem_bytes is the worst-case SMEM size — must be sized at capture time
 * to the maximum seq_len ever expected (max_seq_len * sizeof(float)). */
static inline void launch_attention_devp(hip_llm_runner *r, void *out, void *q,
                                          void *key_cache, void *value_cache,
                                          int n_heads, int n_kv_heads, int head_dim,
                                          int kv_dim, float scale, size_t smem_bytes) {
    int *pos_p = r->d_position;
    void *args[] = { &out, &q, &key_cache, &value_cache,
                     &n_heads, &n_kv_heads, &head_dim, &kv_dim, &pos_p, &scale };
    LAUNCH(r->fn_attn_decode_f32_devp, n_heads, 1, 1, 256, 1, 1, smem_bytes, r->stream, args);
}

static inline void launch_silu_mul(hip_llm_runner *r, void *gate, void *up, int n) {
    void *args[] = { &gate, &up, &n };
    LAUNCH(r->fn_silu_mul_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_add(hip_llm_runner *r, void *dst, void *src, int n) {
    void *args[] = { &dst, &src, &n };
    LAUNCH(r->fn_add_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_embed_q8_0(hip_llm_runner *r, void *dst, void *embd_table,
                                      int token_id, int n_embd) {
    void *args[] = { &dst, &embd_table, &token_id, &n_embd };
    LAUNCH(r->fn_embed_q8_0, (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_matvec_q8_f32(hip_llm_runner *r, void *dst, void *mat,
                                         void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    LAUNCH(r->fn_matvec_q8_0_f32, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args);
}

#define DEFINE_LAUNCH_MATVEC(name, fn_field) \
static inline void launch_matvec_##name(hip_llm_runner *r, void *dst, void *mat, \
                                        void *x, int n_rows, int n_cols) { \
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols }; \
    LAUNCH(r->fn_field, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args); \
}

DEFINE_LAUNCH_MATVEC(q2_K, fn_matvec_q2_K_f32)
DEFINE_LAUNCH_MATVEC(q3_K, fn_matvec_q3_K_f32)
DEFINE_LAUNCH_MATVEC(q4_K, fn_matvec_q4_K_f32)
DEFINE_LAUNCH_MATVEC(q5_K, fn_matvec_q5_K_f32)
DEFINE_LAUNCH_MATVEC(q6_K, fn_matvec_q6_K_f32)
DEFINE_LAUNCH_MATVEC(iq2_xxs, fn_matvec_iq2_xxs_f32)
DEFINE_LAUNCH_MATVEC(q4_0, fn_matvec_q4_0_f32)
DEFINE_LAUNCH_MATVEC(q4_1, fn_matvec_q4_1_f32)
DEFINE_LAUNCH_MATVEC(q5_0, fn_matvec_q5_0_f32)
DEFINE_LAUNCH_MATVEC(q5_1, fn_matvec_q5_1_f32)
DEFINE_LAUNCH_MATVEC(iq4_nl, fn_matvec_iq4_nl_f32)
DEFINE_LAUNCH_MATVEC(iq4_xs, fn_matvec_iq4_xs_f32)
DEFINE_LAUNCH_MATVEC(iq2_xs, fn_matvec_iq2_xs_f32)
DEFINE_LAUNCH_MATVEC(iq3_xxs, fn_matvec_iq3_xxs_f32)
DEFINE_LAUNCH_MATVEC(iq2_s, fn_matvec_iq2_s_f32)
DEFINE_LAUNCH_MATVEC(iq3_s, fn_matvec_iq3_s_f32)
DEFINE_LAUNCH_MATVEC(iq1_s, fn_matvec_iq1_s_f32)
DEFINE_LAUNCH_MATVEC(iq1_m, fn_matvec_iq1_m_f32)
DEFINE_LAUNCH_MATVEC(tq1_0, fn_matvec_tq1_0_f32)
DEFINE_LAUNCH_MATVEC(tq2_0, fn_matvec_tq2_0_f32)

#undef DEFINE_LAUNCH_MATVEC

static inline void launch_embed_q2_K(hip_llm_runner *r, void *dst, void *embd_table,
                                       int token_id, int n_embd) {
    void *args[] = { &dst, &embd_table, &token_id, &n_embd };
    LAUNCH(r->fn_embed_q2_K, (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_matvec_auto(hip_llm_runner *r, void *dst, void *mat,
                                       void *x, int n_rows, int n_cols, int weight_type) {
    switch (weight_type) {
        case GGML_TYPE_Q8_0: launch_matvec_q8_f32(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q2_K: launch_matvec_q2_K(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q3_K: launch_matvec_q3_K(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q4_K: launch_matvec_q4_K(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q5_K: launch_matvec_q5_K(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q6_K: launch_matvec_q6_K(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ2_XXS: launch_matvec_iq2_xxs(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q4_0:    launch_matvec_q4_0(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q4_1:    launch_matvec_q4_1(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q5_0:    launch_matvec_q5_0(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q5_1:    launch_matvec_q5_1(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ4_NL:  launch_matvec_iq4_nl(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ4_XS:  launch_matvec_iq4_xs(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ2_XS:  launch_matvec_iq2_xs(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ3_XXS: launch_matvec_iq3_xxs(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ2_S:   launch_matvec_iq2_s(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ3_S:   launch_matvec_iq3_s(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ1_S:   launch_matvec_iq1_s(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ1_M:   launch_matvec_iq1_m(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_TQ1_0:   launch_matvec_tq1_0(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_TQ2_0:   launch_matvec_tq2_0(r, dst, mat, x, n_rows, n_cols); break;
        default:             launch_matvec(r, dst, mat, x, n_rows, n_cols); break;
    }
}

/* Phase 2: BF16 packers */
static inline void launch_pack_bf16_from_f32(hip_llm_runner *r, void *dst,
                                             void *src, int n) {
    void *args[] = { &dst, &src, &n };
    int n4 = (n + 3) / 4;
    LAUNCH(r->fn_pack_bf16_from_f32, (n4 + 255) / 256, 1, 1, 256, 1, 1, 0,
           r->stream, args);
}

static inline void launch_convert_f16_to_bf16(hip_llm_runner *r, void *dst,
                                              void *src, int n) {
    void *args[] = { &dst, &src, &n };
    LAUNCH(r->fn_convert_f16_to_bf16, (n + 255) / 256, 1, 1, 256, 1, 1, 0,
           r->stream, args);
}

/* Phase 3: FA helpers */
static inline void launch_pack_f16_from_f32(hip_llm_runner *r, void *dst,
                                             void *src, int n) {
    void *args[] = { &dst, &src, &n };
    int n4 = (n + 3) / 4;
    LAUNCH(r->fn_pack_f16_from_f32, (n4 + 255) / 256, 1, 1, 256, 1, 1, 0,
           r->stream, args);
}

static inline void launch_pack_kv_cache_f16(hip_llm_runner *r,
                                             void *dst_k, void *dst_v,
                                             void *src_k, void *src_v,
                                             int kv_len, int n_kv_heads, int head_dim) {
    int block = head_dim < 256 ? head_dim : 256;
    {
        void *args[] = { &dst_k, &src_k, &kv_len, &n_kv_heads, &head_dim };
        LAUNCH(r->fn_pack_kv_cache_f16, kv_len, n_kv_heads, 1, block, 1, 1, 0,
               r->stream, args);
    }
    {
        void *args[] = { &dst_v, &src_v, &kv_len, &n_kv_heads, &head_dim };
        LAUNCH(r->fn_pack_kv_cache_f16, kv_len, n_kv_heads, 1, block, 1, 1, 0,
               r->stream, args);
    }
}

static inline void launch_rope_neox_batch(hip_llm_runner *r, void *vec_batch,
                                           int n_heads, int head_dim,
                                           int position_start, float freq_base,
                                           int n_rope_pairs, int row_stride, int M) {
    int half_dim = head_dim / 2;
    int block = half_dim;
    if (block < 1) block = 1;
    void *args[] = { &vec_batch, &n_heads, &head_dim, &position_start,
                     &freq_base, &n_rope_pairs, &row_stride };
    LAUNCH(r->fn_rope_neox_batch_f32, n_heads, M, 1, block, 1, 1, 0,
           r->stream, args);
}

static inline void launch_rope_mrope_batch(hip_llm_runner *r, void *vec_batch,
                                            int n_heads, int head_dim,
                                            int position_start, float freq_base,
                                            int s0, int s1, int s2, int s3,
                                            int row_stride, int M) {
    int half_dim = head_dim / 2;
    int block = half_dim;
    if (block < 1) block = 1;
    void *args[] = { &vec_batch, &n_heads, &head_dim, &position_start,
                     &freq_base, &s0, &s1, &s2, &s3, &row_stride };
    LAUNCH(r->fn_rope_mrope_batch_f32, n_heads, M, 1, block, 1, 1, 0,
           r->stream, args);
}

static inline void launch_kv_store_batch(hip_llm_runner *r,
                                          void *key_cache, void *value_cache,
                                          void *k_batch, void *v_batch,
                                          int position_start, int M, int kv_dim) {
    int total = M * kv_dim;
    void *args[] = { &key_cache, &value_cache, &k_batch, &v_batch,
                     &position_start, &M, &kv_dim };
    LAUNCH(r->fn_kv_cache_store_batch, (total + 255) / 256, 1, 1, 256, 1, 1, 0,
           r->stream, args);
}

static inline void launch_flash_attn_causal(hip_llm_runner *r,
                                             void *out, void *q_f16,
                                             void *k_f16, void *v_f16,
                                             int M, int kv_len, int position_start,
                                             int n_heads, int n_kv_heads, int head_dim,
                                             float scale) {
    void *args[] = { &out, &q_f16, &k_f16, &v_f16,
                     &M, &kv_len, &position_start,
                     &n_heads, &n_kv_heads, &head_dim, &scale };
    int q_blocks = (M + 63) / 64;
    /* Shared mem: 4*16*128 (smK0+smK1+smV0+smV1) + 4*16*16 (smP) f16 = 9216 elems = 18432 B. */
    int smem_bytes = (4 * 16 * 128 + 4 * 16 * 16) * 2;
    LAUNCH(r->fn_flash_attn_wmma_f16_4w_causal, n_heads, q_blocks, 1, 128, 1, 1,
           smem_bytes, r->stream, args);
}

/* SSM launch helpers */
static inline void launch_softplus_mul(hip_llm_runner *r, void *out,
    void *in, void *bias, void *a, int n) {
    void *args[] = { &out, &in, &bias, &a, &n };
    LAUNCH(r->fn_softplus_mul_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_sigmoid_inplace(hip_llm_runner *r, void *data, int n) {
    void *args[] = { &data, &n };
    LAUNCH(r->fn_sigmoid_inplace_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_conv1d(hip_llm_runner *r, void *conv_out,
    void *conv_state, void *input, void *weight, int qkv_dim, int conv_k) {
    void *args[] = { &conv_out, &conv_state, &input, &weight, &qkv_dim, &conv_k };
    LAUNCH(r->fn_conv1d_depthwise_silu_f32, (qkv_dim + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_l2_norm_heads(hip_llm_runner *r, void *data,
    int n_heads, int head_dim, float eps) {
    int threads = (head_dim <= 128) ? 128 : 256;
    void *args[] = { &data, &n_heads, &head_dim, &eps };
    LAUNCH(r->fn_l2_norm_heads_f32, n_heads, 1, 1, threads, 1, 1, threads * sizeof(float), r->stream, args);
}

static inline void launch_repeat_tile(hip_llm_runner *r, void *dst,
    void *src, int dt_rank, int d_state, int n_group) {
    int total = dt_rank * d_state;
    void *args[] = { &dst, &src, &dt_rank, &d_state, &n_group };
    LAUNCH(r->fn_repeat_tile_f32, (total + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_deltanet_step(hip_llm_runner *r, void *state,
    void *out, void *Q, void *K, void *V,
    void *alpha, void *beta, int dt_rank, int d_state) {
    void *args[] = { &state, &out, &Q, &K, &V, &alpha, &beta, &d_state };
    LAUNCH(r->fn_deltanet_step_f32, dt_rank, 1, 1, d_state, 1, 1, 0, r->stream, args);
}

static inline void launch_gated_rmsnorm_silu(hip_llm_runner *r, void *out,
    void *z, void *norm_w, int dt_rank, int d_state, float eps) {
    int threads = (d_state <= 128) ? 128 : 256;
    void *args[] = { &out, &z, &norm_w, &dt_rank, &d_state, &eps };
    LAUNCH(r->fn_gated_rmsnorm_silu_f32, dt_rank, 1, 1, threads, 1, 1, threads * sizeof(float), r->stream, args);
}

static inline void launch_sigmoid_mul(hip_llm_runner *r, void *data, void *gate, int n) {
    void *args[] = { &data, &gate, &n };
    LAUNCH(r->fn_sigmoid_mul_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_deinterleave_qgate(hip_llm_runner *r, void *q,
    void *gate, void *qfull, int n_heads, int head_dim) {
    int total = n_heads * head_dim;
    void *args[] = { &q, &gate, &qfull, &n_heads, &head_dim };
    LAUNCH(r->fn_deinterleave_qgate_f32, (total + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_scale_add(hip_llm_runner *r, void *dst, void *src,
                                     float scale, int n) {
    void *args[] = { &dst, &src, &scale, &n };
    LAUNCH(r->fn_scale_add_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_matvec_f32(hip_llm_runner *r, void *dst, void *mat,
                                      void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    LAUNCH(r->fn_matvec_f32_f32, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args);
}

/* Top-K softmax for MoE routing */
static void moe_topk_softmax(const float *logits, int n, int k, int *out_idx, float *out_weights) {
    for (int i = 0; i < k; i++) { out_idx[i] = -1; out_weights[i] = -1e30f; }
    for (int ki = 0; ki < k; ki++) {
        float best = -1e30f;
        int best_idx = -1;
        for (int i = 0; i < n; i++) {
            int skip = 0;
            for (int j = 0; j < ki; j++) {
                if (out_idx[j] == i) { skip = 1; break; }
            }
            if (skip) continue;
            if (logits[i] > best) { best = logits[i]; best_idx = i; }
        }
        out_idx[ki] = best_idx;
        out_weights[ki] = best;
    }
    float max_val = out_weights[0];
    for (int i = 1; i < k; i++) {
        if (out_weights[i] > max_val) max_val = out_weights[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        out_weights[i] = expf(out_weights[i] - max_val);
        sum += out_weights[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < k; i++) {
        out_weights[i] *= inv_sum;
    }
}

/* ======================================================================== */
/* Public API: forward                                                      */
/* ======================================================================== */

static void forward_blocks_body(hip_llm_runner *r);
static float *hip_llm_forward_blocks(hip_llm_runner *r, int position);

float *hip_llm_forward(hip_llm_runner *r, int32_t token_id, int position) {
    if (!r || !r->weights_loaded) return NULL;
    if (token_id < 0 || token_id >= r->n_vocab) return NULL;
    if (position < 0 || position >= r->max_seq_len) return NULL;

    int n_embd = r->n_embd;

    /* 1. Token embedding lookup -> F32 (outside captured graph) */
    if (r->token_embd_type == GGML_TYPE_Q8_0) {
        launch_embed_q8_0(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else if (r->token_embd_type == GGML_TYPE_Q2_K) {
        launch_embed_q2_K(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else {
        launch_embed(r, r->d_x, r->d_token_embd, token_id, n_embd);
    }

    return hip_llm_forward_blocks(r, position);
}

/* Public hidden-state path: writes d_position, then runs body or graph,
 * does D2H of d_x → h_output, syncs, returns h_output. */
static float *hip_llm_forward_blocks(hip_llm_runner *r, int position) {
    hipMemcpyAsync(r->d_position, &position, sizeof(int),
                   hipMemcpyHostToDevice, r->stream);

    if (r->graph_ready_hidden) {
        hipGraphLaunch(r->graph_exec_hidden, r->stream);
    } else {
        forward_blocks_body(r);
        hipMemcpyAsync(r->h_output, r->d_x, (size_t)r->n_embd * sizeof(float),
                       hipMemcpyDeviceToHost, r->stream);
    }
    hipStreamSynchronize(r->stream);
    return r->h_output;
}

/* Pure kernel sequence: layer loop + final RMSNorm. No sync, no D2H.
 * Reads position from r->d_position via the *_devp launchers.
 * Suitable for HIP stream capture (no host syncs in non-MoE/non-SSM/non-debug
 * paths). MoE and hybrid SSM paths contain hipDeviceSynchronize and are
 * NOT graph-capturable; graph_eligible must be 0 for those. */
static void forward_blocks_body(hip_llm_runner *r) {
    int n_embd = r->n_embd;
    int n_heads = r->n_heads;
    int n_kv_heads = r->n_kv_heads;
    int head_dim = r->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int n_ff = r->n_ff;
    float eps = r->rms_norm_eps;

    int n_run_layers = r->n_layers;
    if (r->max_layers > 0 && r->max_layers < r->n_layers) n_run_layers = r->max_layers;

    for (int l = 0; l < n_run_layers; l++) {
        hip_layer *cl = &r->layers[l];

        /* Pre-attention RMSNorm */
        launch_rmsnorm(r, r->d_xb, r->d_x, cl->attn_norm_w, n_embd, eps);

        if (r->is_hybrid && cl->is_ssm) {
            /* === SSM (Delta-Net) layer === */
            int qkv_dim = r->ssm_qkv_dim;
            int d_state = r->ssm_d_state;
            int n_group = r->ssm_n_group;
            int dt_rank = r->ssm_dt_rank;
            int conv_k  = r->ssm_conv_kernel;

            launch_matvec_auto(r, r->d_ssm_qkv,   cl->ssm_qkv_w,   r->d_xb, cl->ssm_qkv_rows,   cl->ssm_qkv_cols,   cl->ssm_qkv_type);
            launch_matvec_auto(r, r->d_ssm_z,      cl->ssm_gate_w,  r->d_xb, cl->ssm_gate_rows,  cl->ssm_gate_cols,  cl->ssm_gate_type);
            launch_matvec_auto(r, r->d_ssm_alpha,  cl->ssm_alpha_w, r->d_xb, cl->ssm_alpha_rows, cl->ssm_alpha_cols, cl->ssm_alpha_type);
            launch_matvec_auto(r, r->d_ssm_beta,   cl->ssm_beta_w,  r->d_xb, cl->ssm_beta_rows,  cl->ssm_beta_cols,  cl->ssm_beta_type);

            launch_softplus_mul(r, r->d_ssm_alpha, r->d_ssm_alpha, cl->ssm_dt_bias, cl->ssm_a, dt_rank);
            launch_sigmoid_inplace(r, r->d_ssm_beta, dt_rank);

            launch_conv1d(r, r->d_ssm_conv_out, cl->d_conv_state, r->d_ssm_qkv,
                         cl->ssm_conv1d_w, qkv_dim, conv_k);

            launch_l2_norm_heads(r, r->d_ssm_conv_out, n_group, d_state, eps);
            void *K_raw = (void *)((char *)r->d_ssm_conv_out + (size_t)n_group * d_state * sizeof(float));
            launch_l2_norm_heads(r, K_raw, n_group, d_state, eps);

            launch_repeat_tile(r, r->d_ssm_Q_exp, r->d_ssm_conv_out, dt_rank, d_state, n_group);
            launch_repeat_tile(r, r->d_ssm_K_exp, K_raw, dt_rank, d_state, n_group);

            void *V_ptr = (void *)((char *)r->d_ssm_conv_out + (size_t)2 * n_group * d_state * sizeof(float));

            launch_deltanet_step(r, cl->d_recurrent_state, r->d_ssm_out,
                                r->d_ssm_Q_exp, r->d_ssm_K_exp, V_ptr,
                                r->d_ssm_alpha, r->d_ssm_beta, dt_rank, d_state);

            launch_gated_rmsnorm_silu(r, r->d_ssm_out, r->d_ssm_z, cl->ssm_norm_w,
                                     dt_rank, d_state, eps);

            launch_matvec_auto(r, r->d_xb, cl->ssm_out_w, r->d_ssm_out,
                              cl->ssm_out_rows, cl->ssm_out_cols, cl->ssm_out_type);

        } else if (r->is_hybrid) {
            /* === Gated attention layer (Qwen3.5) === */
            launch_matvec_auto(r, r->d_xb2, cl->attn_q_w, r->d_xb,
                              cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);
            launch_deinterleave_qgate(r, r->d_q, r->d_attn_gate, r->d_xb2, n_heads, head_dim);

            launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb, cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
            launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb, cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);

            if (cl->has_qk_norm) {
                if (cl->attn_q_norm_w) launch_qknorm(r, r->d_q, cl->attn_q_norm_w, n_heads, head_dim, eps);
                if (cl->attn_k_norm_w) launch_qknorm(r, r->d_k, cl->attn_k_norm_w, n_kv_heads, head_dim, eps);
            }

            launch_rope_devp(r, r->d_q, n_heads, head_dim, r->rope_freq_base);
            launch_rope_devp(r, r->d_k, n_kv_heads, head_dim, r->rope_freq_base);

            launch_kv_store_devp(r, r->d_key_cache[l], r->d_value_cache[l],
                                 r->d_k, r->d_v, kv_dim);

            float scale = 1.0f / sqrtf((float)head_dim);
            size_t smem_attn = (size_t)r->max_seq_len * sizeof(float);
            launch_attention_devp(r, r->d_xb2, r->d_q,
                                   r->d_key_cache[l], r->d_value_cache[l],
                                   n_heads, n_kv_heads, head_dim, kv_dim, scale, smem_attn);

            int q_dim_local = n_heads * head_dim;
            launch_sigmoid_mul(r, r->d_xb2, r->d_attn_gate, q_dim_local);

            launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                              cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);

        } else {
            /* === Standard attention === */
            launch_matvec_auto(r, r->d_q, cl->attn_q_w, r->d_xb, cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);
            launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb, cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
            launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb, cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);

            /* Attention biases (Qwen2.5-VL) */
            if (cl->attn_q_bias) launch_add(r, r->d_q, cl->attn_q_bias, cl->attn_q_rows);
            if (cl->attn_k_bias) launch_add(r, r->d_k, cl->attn_k_bias, cl->attn_k_rows);
            if (cl->attn_v_bias) launch_add(r, r->d_v, cl->attn_v_bias, cl->attn_v_rows);

            if (cl->has_qk_norm) {
                if (cl->attn_q_norm_w) launch_qknorm(r, r->d_q, cl->attn_q_norm_w, n_heads, head_dim, eps);
                if (cl->attn_k_norm_w) launch_qknorm(r, r->d_k, cl->attn_k_norm_w, n_kv_heads, head_dim, eps);
            }

            launch_rope_devp(r, r->d_q, n_heads, head_dim, r->rope_freq_base);
            launch_rope_devp(r, r->d_k, n_kv_heads, head_dim, r->rope_freq_base);

            launch_kv_store_devp(r, r->d_key_cache[l], r->d_value_cache[l],
                                 r->d_k, r->d_v, kv_dim);

            float scale = 1.0f / sqrtf((float)head_dim);
            size_t smem_attn = (size_t)r->max_seq_len * sizeof(float);
            launch_attention_devp(r, r->d_xb2, r->d_q,
                                   r->d_key_cache[l], r->d_value_cache[l],
                                   n_heads, n_kv_heads, head_dim, kv_dim, scale, smem_attn);

            launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                              cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);
        }

        /* Residual: x += xb */
        launch_add(r, r->d_x, r->d_xb, n_embd);

        /* FFN RMSNorm */
        launch_rmsnorm(r, r->d_xb, r->d_x, cl->ffn_norm_w, n_embd, eps);

        if (cl->is_moe) {
            /* MoE FFN */
            int n_experts = r->n_experts;
            int n_experts_used = r->n_experts_used;
            int expert_ff = r->expert_ff;
            int shared_expert_ff = r->shared_expert_ff;

            launch_matvec_f32(r, r->d_router_logits, cl->moe_gate_w, r->d_xb,
                             cl->moe_gate_rows, cl->moe_gate_cols);

            hipDeviceSynchronize();
            hipMemcpy(r->h_router_logits, r->d_router_logits, n_experts * sizeof(float), hipMemcpyDeviceToHost);
            int top_k_idx[64];
            float top_k_weights[64];
            moe_topk_softmax(r->h_router_logits, n_experts, n_experts_used, top_k_idx, top_k_weights);

            hipMemset(r->d_moe_accum, 0, n_embd * sizeof(float));

            for (int e = 0; e < n_experts_used; e++) {
                int eidx = top_k_idx[e];
                void *gate_w = (void *)((char *)cl->moe_gate_exps_w + (size_t)eidx * cl->moe_exp_stride_gu);
                void *up_w   = (void *)((char *)cl->moe_up_exps_w   + (size_t)eidx * cl->moe_exp_stride_gu);
                void *down_w = (void *)((char *)cl->moe_down_exps_w  + (size_t)eidx * cl->moe_exp_stride_d);

                launch_matvec_auto(r, r->d_gate, gate_w, r->d_xb,
                                  cl->moe_exp_rows_gu, cl->moe_exp_cols_gu, cl->moe_gate_exps_type);
                launch_matvec_auto(r, r->d_up, up_w, r->d_xb,
                                  cl->moe_exp_rows_gu, cl->moe_exp_cols_gu, cl->moe_up_exps_type);
                launch_silu_mul(r, r->d_gate, r->d_up, expert_ff);
                launch_matvec_auto(r, r->d_xb2, down_w, r->d_gate,
                                  cl->moe_exp_rows_d, cl->moe_exp_cols_d, cl->moe_down_exps_type);
                launch_scale_add(r, r->d_moe_accum, r->d_xb2, top_k_weights[e], n_embd);
            }

            /* Shared expert */
            {
                launch_matvec_f32(r, r->d_router_logits, cl->moe_shared_gate_w, r->d_xb, 1, n_embd);
                hipDeviceSynchronize();
                float gate_val;
                hipMemcpy(&gate_val, r->d_router_logits, sizeof(float), hipMemcpyDeviceToHost);
                float shared_scale = 1.0f / (1.0f + expf(-gate_val));

                launch_matvec_auto(r, r->d_gate, cl->moe_shared_ffn_gate_w, r->d_xb,
                                  cl->moe_shared_gate_rows, cl->moe_shared_gate_cols, cl->moe_shared_gate_type);
                launch_matvec_auto(r, r->d_up, cl->moe_shared_ffn_up_w, r->d_xb,
                                  cl->moe_shared_up_rows, cl->moe_shared_up_cols, cl->moe_shared_up_type);
                launch_silu_mul(r, r->d_gate, r->d_up, shared_expert_ff);
                launch_matvec_auto(r, r->d_xb2, cl->moe_shared_ffn_down_w, r->d_gate,
                                  cl->moe_shared_down_rows, cl->moe_shared_down_cols, cl->moe_shared_down_type);
                launch_scale_add(r, r->d_moe_accum, r->d_xb2, shared_scale, n_embd);
            }

            hipMemcpyAsync(r->d_xb, r->d_moe_accum, n_embd * sizeof(float),
                          hipMemcpyDeviceToDevice, r->stream);

        } else {
            /* Dense FFN */
            launch_matvec_auto(r, r->d_gate, cl->ffn_gate_w, r->d_xb,
                          cl->ffn_gate_rows, cl->ffn_gate_cols, cl->ffn_gate_type);
            launch_matvec_auto(r, r->d_up, cl->ffn_up_w, r->d_xb,
                          cl->ffn_up_rows, cl->ffn_up_cols, cl->ffn_up_type);

            launch_silu_mul(r, r->d_gate, r->d_up, n_ff);
            launch_matvec_auto(r, r->d_xb, cl->ffn_down_w, r->d_gate,
                          cl->ffn_down_rows, cl->ffn_down_cols, cl->ffn_down_type);
        }

        /* Residual: x += xb */
        launch_add(r, r->d_x, r->d_xb, n_embd);

        /* DeepStack injection */
        if (r->_ds_embd && l < r->n_deepstack && r->_ds_embd_stride > n_embd) {
            const float *ds_slice = r->_ds_embd + (1 + l) * n_embd;
            hipMemcpyAsync(r->d_ds_tmp, (const void *)ds_slice, n_embd * sizeof(float),
                          hipMemcpyHostToDevice, r->stream);
            launch_add(r, r->d_x, r->d_ds_tmp, n_embd);
        }

        /* Debug: print hidden state norm */
        if (r->debug_layers) {
            hipDeviceSynchronize();
            float *full_x = (float *)malloc(n_embd * sizeof(float));
            hipMemcpy(full_x, r->d_x, n_embd * sizeof(float), hipMemcpyDeviceToHost);
            float ss = 0;
            for (int i = 0; i < n_embd; i++) ss += full_x[i] * full_x[i];
            fprintf(stderr, "  [L%02d %s] norm=%.2f first=[%.4f, %.4f, %.4f, %.4f]\n",
                    l, (r->is_hybrid && cl->is_ssm) ? "SSM" : "ATT",
                    sqrtf(ss), full_x[0], full_x[1], full_x[2], full_x[3]);
            free(full_x);
        }
    }

    /* Final RMSNorm */
    launch_rmsnorm(r, r->d_x, r->d_x, r->d_output_norm, n_embd, eps);
}

/* === Phase 5 graph capture =============================================== */
static void hip_llm_phase5_capture(hip_llm_runner *r) {
    r->graph_disabled = 0;
    r->graph_verbose  = 0;
    const char *env_gd = getenv("LLM_GRAPH_DISABLE");
    const char *env_gv = getenv("LLM_GRAPH_VERBOSE");
    if (env_gd) r->graph_disabled = atoi(env_gd) != 0;
    if (env_gv) r->graph_verbose  = atoi(env_gv) != 0;

    /* DeepStack only fires when r->_ds_embd is set (forward_embd path).
     * Decode (forward → forward_blocks_body with _ds_embd==NULL) skips it,
     * so it's safe to capture for n_deepstack>0 models like Qwen3-VL. */
    r->graph_eligible =
        !r->is_moe && !r->is_hybrid && !r->debug_layers;

    if (!r->graph_eligible || r->graph_disabled) {
        if (r->verbose >= 1) {
            fprintf(stderr,
                    "hip_llm: Phase-5 graph capture skipped (eligible=%d disabled=%d)\n",
                    r->graph_eligible, r->graph_disabled);
        }
        return;
    }

    if (!hipStreamBeginCapture || !hipStreamEndCapture ||
        !hipGraphInstantiate || !hipGraphLaunch) {
        if (r->verbose >= 1) {
            fprintf(stderr,
                    "hip_llm: Phase-5 graph capture skipped (HIP graph symbols unavailable)\n");
        }
        return;
    }

    int warm_pos = 0;
    hipMemcpy(r->d_position, &warm_pos, sizeof(int), hipMemcpyHostToDevice);

    /* --- logits graph: blocks + final norm + lm_head + D2H of d_logits --- */
    if (r->has_lm_head) {
        hipError_t err = hipStreamBeginCapture(r->stream,
                                               hipStreamCaptureModeThreadLocal);
        if (err == hipSuccess) {
            forward_blocks_body(r);
            launch_matvec_auto(r, r->d_logits, r->d_output_w, r->d_x,
                               r->n_vocab, r->n_embd, r->output_w_type);
            hipMemcpyAsync(r->h_output, r->d_logits,
                           (size_t)r->n_vocab * sizeof(float),
                           hipMemcpyDeviceToHost, r->stream);
            err = hipStreamEndCapture(r->stream, &r->graph_logits);
            if (err == hipSuccess && r->graph_logits) {
                err = hipGraphInstantiate(&r->graph_exec_logits, r->graph_logits,
                                          NULL, NULL, 0);
                if (err == hipSuccess) r->graph_ready_logits = 1;
            }
        }
        if (!r->graph_ready_logits && r->verbose >= 1) {
            fprintf(stderr,
                    "hip_llm: Phase-5 logits graph capture failed (err=%d)\n",
                    (int)err);
        }
    }

    /* --- hidden graph: blocks + final norm + D2H of d_x ------------------ */
    {
        hipError_t err = hipStreamBeginCapture(r->stream,
                                               hipStreamCaptureModeThreadLocal);
        if (err == hipSuccess) {
            forward_blocks_body(r);
            hipMemcpyAsync(r->h_output, r->d_x,
                           (size_t)r->n_embd * sizeof(float),
                           hipMemcpyDeviceToHost, r->stream);
            err = hipStreamEndCapture(r->stream, &r->graph_hidden);
            if (err == hipSuccess && r->graph_hidden) {
                err = hipGraphInstantiate(&r->graph_exec_hidden, r->graph_hidden,
                                          NULL, NULL, 0);
                if (err == hipSuccess) r->graph_ready_hidden = 1;
            }
        }
        if (!r->graph_ready_hidden && r->verbose >= 1) {
            fprintf(stderr,
                    "hip_llm: Phase-5 hidden graph capture failed (err=%d)\n",
                    (int)err);
        }
    }

    if (r->verbose >= 1 || r->graph_verbose) {
        fprintf(stderr,
                "hip_llm: Phase-5 graph capture: logits=%d hidden=%d\n",
                r->graph_ready_logits, r->graph_ready_hidden);
    }
}

float *hip_llm_forward_logits(hip_llm_runner *r, int32_t token_id, int position) {
    if (!r || !r->weights_loaded) return NULL;
    if (token_id < 0 || token_id >= r->n_vocab) return NULL;
    if (position < 0 || position >= r->max_seq_len) return NULL;
    if (!r->has_lm_head) return NULL;

    int n_embd = r->n_embd;

    /* Embed (outside captured graph — token_id varies) */
    if (r->token_embd_type == GGML_TYPE_Q8_0) {
        launch_embed_q8_0(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else if (r->token_embd_type == GGML_TYPE_Q2_K) {
        launch_embed_q2_K(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else {
        launch_embed(r, r->d_x, r->d_token_embd, token_id, n_embd);
    }

    /* Position is consumed by *_devp launchers via r->d_position */
    hipMemcpyAsync(r->d_position, &position, sizeof(int),
                   hipMemcpyHostToDevice, r->stream);

    if (r->graph_ready_logits) {
        hipGraphLaunch(r->graph_exec_logits, r->stream);
    } else {
        forward_blocks_body(r);
        launch_matvec_auto(r, r->d_logits, r->d_output_w, r->d_x,
                           r->n_vocab, r->n_embd, r->output_w_type);
        hipMemcpyAsync(r->h_output, r->d_logits, (size_t)r->n_vocab * sizeof(float),
                       hipMemcpyDeviceToHost, r->stream);
    }
    hipStreamSynchronize(r->stream);
    return r->h_output;
}

/* ---- Batched forward (Phase 2: hipBLASLt-routed dense path) ---- */

#ifdef LLM_HIPBLASLT_ENABLED
/* Run a single dense Qwen3 transformer block over [M, n_embd] activations.
 * d_x_in is [M, n_embd] F32; output is written into d_x_in (in place).
 * KV cache stride matches the per-token path.
 *
 * Strategy: for the 7 dense GEMMs (Q/K/V/O/gate/up/down) we issue a single
 * hipBLASLt call per (M, N, K) shape. For lightweight ops (rmsnorm, qknorm,
 * rope, kv_store, attention, residual) we loop over the M rows using the
 * existing per-row kernels — these are not the bottleneck.
 *
 * Returns 0 on success, -1 on error. */
static int forward_block_batched_dense(hip_llm_runner *r, int M,
                                       int position_start) {
    int n_embd     = r->n_embd;
    int n_heads    = r->n_heads;
    int n_kv_heads = r->n_kv_heads;
    int head_dim   = r->head_dim;
    int kv_dim     = n_kv_heads * head_dim;
    int q_dim      = n_heads * head_dim;
    int n_ff       = r->n_ff;
    float eps      = r->rms_norm_eps;
    float scale    = 1.0f / sqrtf((float)head_dim);

    int n_run_layers = r->n_layers;
    if (r->max_layers > 0 && r->max_layers < r->n_layers) n_run_layers = r->max_layers;

    for (int l = 0; l < n_run_layers; l++) {
        hip_layer *cl = &r->layers[l];

        /* ---- Pre-attention RMSNorm (M rows) ---- */
        for (int m = 0; m < M; m++) {
            float *xrow  = (float *)r->d_x_batch     + (size_t)m * n_embd;
            float *xnrow = (float *)r->d_xnorm_batch + (size_t)m * n_embd;
            launch_rmsnorm(r, xnrow, xrow, cl->attn_norm_w, n_embd, eps);
        }

        /* Pack BF16 input (single launch over M*n_embd) */
        launch_pack_bf16_from_f32(r, r->d_xnorm_batch_bf16,
                                  r->d_xnorm_batch, M * n_embd);

        /* ---- Q/K/V GEMMs (3 hipBLASLt calls) ---- */
        if (mm_blaslt_run_bf16(r->d_q_batch, cl->attn_q_w_bf16,
                               r->d_xnorm_batch_bf16, M, q_dim,  n_embd, r->stream) != 0) return -1;
        if (mm_blaslt_run_bf16(r->d_k_batch, cl->attn_k_w_bf16,
                               r->d_xnorm_batch_bf16, M, kv_dim, n_embd, r->stream) != 0) return -1;
        if (mm_blaslt_run_bf16(r->d_v_batch, cl->attn_v_w_bf16,
                               r->d_xnorm_batch_bf16, M, kv_dim, n_embd, r->stream) != 0) return -1;

        /* ---- Per-row biases / QK-norm (M rows) ---- */
        for (int m = 0; m < M; m++) {
            float *qrow = (float *)r->d_q_batch + (size_t)m * q_dim;
            float *krow = (float *)r->d_k_batch + (size_t)m * kv_dim;
            float *vrow = (float *)r->d_v_batch + (size_t)m * kv_dim;

            if (cl->attn_q_bias) launch_add(r, qrow, cl->attn_q_bias, q_dim);
            if (cl->attn_k_bias) launch_add(r, krow, cl->attn_k_bias, kv_dim);
            if (cl->attn_v_bias) launch_add(r, vrow, cl->attn_v_bias, kv_dim);

            if (cl->has_qk_norm) {
                if (cl->attn_q_norm_w) launch_qknorm(r, qrow, cl->attn_q_norm_w, n_heads,    head_dim, eps);
                if (cl->attn_k_norm_w) launch_qknorm(r, krow, cl->attn_k_norm_w, n_kv_heads, head_dim, eps);
            }
        }

        /* ---- Batched RoPE on Q and K (one launch each, grid=(n_heads, M)) ---- */
        if (r->use_mrope) {
            int s0 = r->mrope_sections[0], s1 = r->mrope_sections[1];
            int s2 = r->mrope_sections[2], s3 = r->mrope_sections[3];
            launch_rope_mrope_batch(r, r->d_q_batch, n_heads,    head_dim,
                                    position_start, r->rope_freq_base,
                                    s0, s1, s2, s3, q_dim,  M);
            launch_rope_mrope_batch(r, r->d_k_batch, n_kv_heads, head_dim,
                                    position_start, r->rope_freq_base,
                                    s0, s1, s2, s3, kv_dim, M);
        } else {
            launch_rope_neox_batch(r, r->d_q_batch, n_heads,    head_dim,
                                   position_start, r->rope_freq_base,
                                   r->n_rope_pairs, q_dim,  M);
            launch_rope_neox_batch(r, r->d_k_batch, n_kv_heads, head_dim,
                                   position_start, r->rope_freq_base,
                                   r->n_rope_pairs, kv_dim, M);
        }

        /* ---- Batched KV cache store (M rows in one launch) ---- */
        launch_kv_store_batch(r, r->d_key_cache[l], r->d_value_cache[l],
                              r->d_k_batch, r->d_v_batch,
                              position_start, M, kv_dim);

        /* ---- Attention: WMMA flash-attention if eligible, else per-row scalar ---- */
        if (r->fa_path_ok) {
            int kv_len = position_start + M;
            /* Pack K/V cache slice [0..kv_len) into transposed F16 scratch. */
            launch_pack_kv_cache_f16(r,
                                     r->d_kv_pack_K_f16, r->d_kv_pack_V_f16,
                                     r->d_key_cache[l],  r->d_value_cache[l],
                                     kv_len, n_kv_heads, head_dim);
            /* Pack Q [M, q_dim] -> F16. */
            launch_pack_f16_from_f32(r, r->d_q_batch_f16, r->d_q_batch, M * q_dim);
            /* Single FA grid launch. */
            launch_flash_attn_causal(r,
                                     r->d_attn_out_batch, r->d_q_batch_f16,
                                     r->d_kv_pack_K_f16,  r->d_kv_pack_V_f16,
                                     M, kv_len, position_start,
                                     n_heads, n_kv_heads, head_dim, scale);
        } else {
            for (int m = 0; m < M; m++) {
                int pos = position_start + m;
                float *qrow = (float *)r->d_q_batch + (size_t)m * q_dim;
                float *orow = (float *)r->d_attn_out_batch + (size_t)m * q_dim;
                int seq_len = pos + 1;
                launch_attention(r, orow, qrow,
                                 r->d_key_cache[l], r->d_value_cache[l],
                                 n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale);
            }
        }

        /* ---- Output projection: [M, q_dim] x W_o^T -> [M, n_embd] ---- */
        launch_pack_bf16_from_f32(r, r->d_attn_out_batch_bf16,
                                  r->d_attn_out_batch, M * q_dim);
        if (mm_blaslt_run_bf16(r->d_attn_proj_batch, cl->attn_output_w_bf16,
                               r->d_attn_out_batch_bf16, M, n_embd, q_dim, r->stream) != 0) return -1;

        /* Residual: x_batch += attn_proj */
        launch_add(r, r->d_x_batch, r->d_attn_proj_batch, M * n_embd);

        /* ---- Pre-FFN RMSNorm (M rows) ---- */
        for (int m = 0; m < M; m++) {
            float *xrow  = (float *)r->d_x_batch     + (size_t)m * n_embd;
            float *xnrow = (float *)r->d_xnorm_batch + (size_t)m * n_embd;
            launch_rmsnorm(r, xnrow, xrow, cl->ffn_norm_w, n_embd, eps);
        }
        launch_pack_bf16_from_f32(r, r->d_ffn_norm_batch_bf16,
                                  r->d_xnorm_batch, M * n_embd);

        /* ---- gate/up GEMMs ---- */
        if (mm_blaslt_run_bf16(r->d_gate_batch, cl->ffn_gate_w_bf16,
                               r->d_ffn_norm_batch_bf16, M, n_ff, n_embd, r->stream) != 0) return -1;
        if (mm_blaslt_run_bf16(r->d_up_batch,   cl->ffn_up_w_bf16,
                               r->d_ffn_norm_batch_bf16, M, n_ff, n_embd, r->stream) != 0) return -1;

        /* SiLU(gate) * up — elementwise across [M, n_ff] */
        launch_silu_mul(r, r->d_gate_batch, r->d_up_batch, M * n_ff);

        /* Pack and down projection */
        launch_pack_bf16_from_f32(r, r->d_silu_batch_bf16,
                                  r->d_gate_batch, M * n_ff);
        if (mm_blaslt_run_bf16(r->d_down_batch, cl->ffn_down_w_bf16,
                               r->d_silu_batch_bf16, M, n_embd, n_ff, r->stream) != 0) return -1;

        /* Residual */
        launch_add(r, r->d_x_batch, r->d_down_batch, M * n_embd);

        /* DeepStack injection (per-row) — keep parity with per-token path */
        if (r->_ds_embd && l < r->n_deepstack && r->_ds_embd_stride > n_embd) {
            for (int m = 0; m < M; m++) {
                const float *ds_slice = r->_ds_embd
                    + (size_t)m * r->_ds_embd_stride
                    + (size_t)(1 + l) * n_embd;
                float *xrow = (float *)r->d_x_batch + (size_t)m * n_embd;
                hipMemcpyAsync(r->d_ds_tmp, (const void *)ds_slice, n_embd * sizeof(float),
                               hipMemcpyHostToDevice, r->stream);
                launch_add(r, xrow, r->d_ds_tmp, n_embd);
            }
        }
    }

    /* ---- Final RMSNorm (M rows) ---- */
    for (int m = 0; m < M; m++) {
        float *xrow = (float *)r->d_x_batch + (size_t)m * n_embd;
        launch_rmsnorm(r, xrow, xrow, r->d_output_norm, n_embd, eps);
    }

    /* Copy last row into r->d_x so callers (and lm_head path) see the result. */
    float *last_row = (float *)r->d_x_batch + (size_t)(M - 1) * n_embd;
    hipMemcpyAsync(r->d_x, last_row, (size_t)n_embd * sizeof(float),
                   hipMemcpyDeviceToDevice, r->stream);
    hipStreamSynchronize(r->stream);
    return 0;
}

static int batched_path_eligible(const hip_llm_runner *r, int M) {
    if (!r->batch_path_ok) return 0;
    if (M < r->gemm_m_threshold) return 0;
    if (M > r->batch_max) return 0;
    if (r->_ds_embd != NULL) return 0;  /* DeepStack handled in embed path only */
    return 1;
}

/* Embed M tokens into d_x_batch [M, n_embd] F32 using existing embed kernels.
 * Each row is one token. Reuses launch_embed* helpers (per-row launches). */
static int embed_tokens_batch(hip_llm_runner *r,
                              const int32_t *tokens, int M) {
    int n_embd = r->n_embd;
    for (int m = 0; m < M; m++) {
        int32_t tid = tokens[m];
        if (tid < 0 || tid >= r->n_vocab) return -1;
        float *row = (float *)r->d_x_batch + (size_t)m * n_embd;
        if (r->token_embd_type == GGML_TYPE_Q8_0) {
            launch_embed_q8_0(r, row, r->d_token_embd, tid, n_embd);
        } else if (r->token_embd_type == GGML_TYPE_Q2_K) {
            launch_embed_q2_K(r, row, r->d_token_embd, tid, n_embd);
        } else {
            launch_embed(r, row, r->d_token_embd, tid, n_embd);
        }
    }
    return 0;
}
#endif /* LLM_HIPBLASLT_ENABLED */

/* ---- Batched forward entry points ---- */
float *hip_llm_forward_batch(hip_llm_runner *r,
                             const int32_t *tokens, int n_tokens,
                             int position_start) {
    if (!r || !r->weights_loaded || !tokens || n_tokens <= 0) return NULL;
    if (position_start < 0) return NULL;
    if (position_start + n_tokens > r->max_seq_len) return NULL;

#ifdef LLM_HIPBLASLT_ENABLED
    if (batched_path_eligible(r, n_tokens)) {
        if (embed_tokens_batch(r, tokens, n_tokens) != 0) return NULL;
        if (forward_block_batched_dense(r, n_tokens, position_start) != 0) return NULL;
        /* Returning NULL hidden buffer for now — callers needing logits use
         * forward_batch_logits. Most call sites only ask for the next token. */
        return (float *)r->d_x;
    }
#endif

    /* Fallback: per-token loop using existing kernels. */
    float *out = NULL;
    for (int i = 0; i < n_tokens; i++) {
        out = hip_llm_forward(r, tokens[i], position_start + i);
        if (!out) return NULL;
    }
    return out;
}

float *hip_llm_forward_batch_logits(hip_llm_runner *r,
                                    const int32_t *tokens, int n_tokens,
                                    int position_start) {
    if (!r || !r->weights_loaded || !tokens || n_tokens <= 0) return NULL;
    if (position_start < 0) return NULL;
    if (position_start + n_tokens > r->max_seq_len) return NULL;

#ifdef LLM_HIPBLASLT_ENABLED
    if (batched_path_eligible(r, n_tokens)) {
        if (embed_tokens_batch(r, tokens, n_tokens) != 0) return NULL;
        if (forward_block_batched_dense(r, n_tokens, position_start) != 0) return NULL;
        if (!r->has_lm_head) return NULL;
        /* lm_head only on last token — d_x already holds the last hidden row. */
        launch_matvec_auto(r, r->d_logits, r->d_output_w, r->d_x,
                           r->n_vocab, r->n_embd, r->output_w_type);
        hipMemcpyAsync(r->h_output, r->d_logits, (size_t)r->n_vocab * sizeof(float),
                       hipMemcpyDeviceToHost, r->stream);
        hipStreamSynchronize(r->stream);
        return r->h_output;
    }
#endif

    /* Fallback */
    for (int i = 0; i < n_tokens - 1; i++) {
        if (!hip_llm_forward(r, tokens[i], position_start + i)) return NULL;
    }
    return hip_llm_forward_logits(r, tokens[n_tokens - 1],
                                  position_start + n_tokens - 1);
}

float *hip_llm_forward_embd(hip_llm_runner *r, const float *embd, int embd_stride, int position) {
    if (!r || !r->weights_loaded || !embd) return NULL;
    if (position < 0 || position >= r->max_seq_len) return NULL;

    int n_embd = r->n_embd;
    hipMemcpyAsync(r->d_x, (const void *)embd, n_embd * sizeof(float),
                  hipMemcpyHostToDevice, r->stream);

    r->_ds_embd = embd;
    r->_ds_embd_stride = embd_stride;

    float *result = hip_llm_forward_blocks(r, position);

    r->_ds_embd = NULL;
    r->_ds_embd_stride = 0;

    return result;
}

float *hip_llm_forward_embd_logits(hip_llm_runner *r, const float *embd, int embd_stride, int position) {
    float *hidden = hip_llm_forward_embd(r, embd, embd_stride, position);
    if (!hidden || !r->has_lm_head) return NULL;

    launch_matvec_auto(r, r->d_logits, r->d_output_w, r->d_x,
                       r->n_vocab, r->n_embd, r->output_w_type);

    hipMemcpyAsync(r->h_output, r->d_logits, (size_t)r->n_vocab * sizeof(float),
                  hipMemcpyDeviceToHost, r->stream);
    hipStreamSynchronize(r->stream);

    return r->h_output;
}

int hip_llm_read_hidden(const hip_llm_runner *r, float *dst, int n) {
    if (!r || !dst || n <= 0) return -1;
    hipError_t err = hipMemcpy(dst, r->d_x, (size_t)n * sizeof(float), hipMemcpyDeviceToHost);
    return (err == hipSuccess) ? 0 : -1;
}

void hip_llm_set_debug(hip_llm_runner *r, int debug_layers) {
    if (r) r->debug_layers = debug_layers;
}

void hip_llm_set_max_layers(hip_llm_runner *r, int max_layers) {
    if (r) r->max_layers = max_layers;
}

/* Toggle Phase-2 batched path. Setting enable=0 forces hip_llm_forward_batch
 * to fall through to the per-token loop. Setting enable=1 only takes effect if
 * the path was eligible at load time (i.e., dense F16 model + LLM_HIPBLASLT_ENABLED). */
void hip_llm_set_batched_path(hip_llm_runner *r, int enable) {
    if (!r) return;
#ifdef LLM_HIPBLASLT_ENABLED
    /* Only flip on if buffers + BF16 weights are actually present. */
    if (enable) {
        r->batch_path_ok = (r->d_x_batch != NULL) ? 1 : 0;
    } else {
        r->batch_path_ok = 0;
    }
#else
    (void)enable;
#endif
}

int hip_llm_batched_path_available(const hip_llm_runner *r) {
#ifdef LLM_HIPBLASLT_ENABLED
    return r ? (r->d_x_batch != NULL) : 0;
#else
    (void)r;
    return 0;
#endif
}

/* ======================================================================== */
/* Public API: free                                                         */
/* ======================================================================== */

void hip_llm_offload(hip_llm_runner *r) {
    if (!r) return;
    /* Free GPU weight and activation buffers only — keep module, stream, context */
    if (r->d_x)    { hipFree(r->d_x);    r->d_x = NULL; }
    if (r->d_xb)   { hipFree(r->d_xb);   r->d_xb = NULL; }
    if (r->d_xb2)  { hipFree(r->d_xb2);  r->d_xb2 = NULL; }
    if (r->d_q)    { hipFree(r->d_q);    r->d_q = NULL; }
    if (r->d_k)    { hipFree(r->d_k);    r->d_k = NULL; }
    if (r->d_v)    { hipFree(r->d_v);    r->d_v = NULL; }
    if (r->d_gate) { hipFree(r->d_gate); r->d_gate = NULL; }
    if (r->d_up)   { hipFree(r->d_up);   r->d_up = NULL; }
    if (r->d_xb_q)    { hipFree(r->d_xb_q);    r->d_xb_q = NULL; }
    if (r->d_xb_scale){ hipFree(r->d_xb_scale); r->d_xb_scale = NULL; }
    if (r->d_logits)  { hipFree(r->d_logits);   r->d_logits = NULL; }
    /* Phase 2 batched buffers */
    if (r->d_x_batch)             { hipFree(r->d_x_batch);             r->d_x_batch = NULL; }
    if (r->d_xnorm_batch)         { hipFree(r->d_xnorm_batch);         r->d_xnorm_batch = NULL; }
    if (r->d_xnorm_batch_bf16)    { hipFree(r->d_xnorm_batch_bf16);    r->d_xnorm_batch_bf16 = NULL; }
    if (r->d_q_batch)             { hipFree(r->d_q_batch);             r->d_q_batch = NULL; }
    if (r->d_k_batch)             { hipFree(r->d_k_batch);             r->d_k_batch = NULL; }
    if (r->d_v_batch)             { hipFree(r->d_v_batch);             r->d_v_batch = NULL; }
    if (r->d_attn_out_batch)      { hipFree(r->d_attn_out_batch);      r->d_attn_out_batch = NULL; }
    if (r->d_attn_out_batch_bf16) { hipFree(r->d_attn_out_batch_bf16); r->d_attn_out_batch_bf16 = NULL; }
    if (r->d_attn_proj_batch)     { hipFree(r->d_attn_proj_batch);     r->d_attn_proj_batch = NULL; }
    if (r->d_ffn_norm_batch_bf16) { hipFree(r->d_ffn_norm_batch_bf16); r->d_ffn_norm_batch_bf16 = NULL; }
    if (r->d_gate_batch)          { hipFree(r->d_gate_batch);          r->d_gate_batch = NULL; }
    if (r->d_up_batch)            { hipFree(r->d_up_batch);            r->d_up_batch = NULL; }
    if (r->d_silu_batch_bf16)     { hipFree(r->d_silu_batch_bf16);     r->d_silu_batch_bf16 = NULL; }
    if (r->d_down_batch)          { hipFree(r->d_down_batch);          r->d_down_batch = NULL; }
    if (r->d_q_batch_f16)         { hipFree(r->d_q_batch_f16);         r->d_q_batch_f16 = NULL; }
    if (r->d_kv_pack_K_f16)       { hipFree(r->d_kv_pack_K_f16);       r->d_kv_pack_K_f16 = NULL; }
    if (r->d_kv_pack_V_f16)       { hipFree(r->d_kv_pack_V_f16);       r->d_kv_pack_V_f16 = NULL; }
    r->fa_path_ok = 0;
    r->batch_path_ok = 0;
    if (r->d_token_embd) { hipFree(r->d_token_embd); r->d_token_embd = NULL; }
    if (r->d_output_norm){ hipFree(r->d_output_norm); r->d_output_norm = NULL; }
    if (r->d_output_w && r->d_output_w != r->d_token_embd) { hipFree(r->d_output_w); }
    r->d_output_w = NULL;
    if (r->d_key_cache) {
        for (int l = 0; l < r->n_layers; l++)
            if (r->d_key_cache[l]) hipFree(r->d_key_cache[l]);
        free(r->d_key_cache); r->d_key_cache = NULL;
    }
    if (r->d_value_cache) {
        for (int l = 0; l < r->n_layers; l++)
            if (r->d_value_cache[l]) hipFree(r->d_value_cache[l]);
        free(r->d_value_cache); r->d_value_cache = NULL;
    }
    if (r->layers) {
        for (int l = 0; l < r->n_layers; l++) {
            hip_layer *cl = &r->layers[l];
            void **ptrs = (void **)cl;
            /* Free all non-NULL GPU pointers in the layer struct */
            for (size_t i = 0; i < sizeof(hip_layer) / sizeof(void *); i++)
                if (ptrs[i]) { hipFree(ptrs[i]); ptrs[i] = NULL; }
        }
        free(r->layers); r->layers = NULL;
    }
    hipDeviceSynchronize();
    fprintf(stderr, "hip_llm: offloaded GPU weights (VRAM freed)\n");
}

void hip_llm_free(hip_llm_runner *r) {
    if (!r) return;

    if (r->d_x)    hipFree(r->d_x);
    if (r->d_xb)   hipFree(r->d_xb);
    if (r->d_xb2)  hipFree(r->d_xb2);
    if (r->d_q)    hipFree(r->d_q);
    if (r->d_k)    hipFree(r->d_k);
    if (r->d_v)    hipFree(r->d_v);
    if (r->d_gate) hipFree(r->d_gate);
    if (r->d_up)   hipFree(r->d_up);
    if (r->d_ssm_qkv)     hipFree(r->d_ssm_qkv);
    if (r->d_ssm_z)        hipFree(r->d_ssm_z);
    if (r->d_ssm_alpha)    hipFree(r->d_ssm_alpha);
    if (r->d_ssm_beta)     hipFree(r->d_ssm_beta);
    if (r->d_ssm_Q_exp)    hipFree(r->d_ssm_Q_exp);
    if (r->d_ssm_K_exp)    hipFree(r->d_ssm_K_exp);
    if (r->d_ssm_out)      hipFree(r->d_ssm_out);
    if (r->d_ssm_conv_out) hipFree(r->d_ssm_conv_out);
    if (r->d_attn_gate)    hipFree(r->d_attn_gate);
    if (r->d_ds_tmp)   hipFree(r->d_ds_tmp);
    if (r->d_router_logits) hipFree(r->d_router_logits);
    if (r->d_moe_accum)    hipFree(r->d_moe_accum);
    free(r->h_router_logits);
    if (r->d_xb_q)    hipFree(r->d_xb_q);
    if (r->d_xb_scale) hipFree(r->d_xb_scale);

    /* Phase 2 batched buffers */
    if (r->d_x_batch)             hipFree(r->d_x_batch);
    if (r->d_xnorm_batch)         hipFree(r->d_xnorm_batch);
    if (r->d_xnorm_batch_bf16)    hipFree(r->d_xnorm_batch_bf16);
    if (r->d_q_batch)             hipFree(r->d_q_batch);
    if (r->d_k_batch)             hipFree(r->d_k_batch);
    if (r->d_v_batch)             hipFree(r->d_v_batch);
    if (r->d_attn_out_batch)      hipFree(r->d_attn_out_batch);
    if (r->d_attn_out_batch_bf16) hipFree(r->d_attn_out_batch_bf16);
    if (r->d_attn_proj_batch)     hipFree(r->d_attn_proj_batch);
    if (r->d_ffn_norm_batch_bf16) hipFree(r->d_ffn_norm_batch_bf16);
    if (r->d_gate_batch)          hipFree(r->d_gate_batch);
    if (r->d_up_batch)            hipFree(r->d_up_batch);
    if (r->d_silu_batch_bf16)     hipFree(r->d_silu_batch_bf16);
    if (r->d_down_batch)          hipFree(r->d_down_batch);
    if (r->d_q_batch_f16)         hipFree(r->d_q_batch_f16);
    if (r->d_kv_pack_K_f16)       hipFree(r->d_kv_pack_K_f16);
    if (r->d_kv_pack_V_f16)       hipFree(r->d_kv_pack_V_f16);

    if (r->d_key_cache) {
        for (int l = 0; l < r->n_layers; l++) {
            if (r->d_key_cache[l]) hipFree(r->d_key_cache[l]);
        }
        free(r->d_key_cache);
    }
    if (r->d_value_cache) {
        for (int l = 0; l < r->n_layers; l++) {
            if (r->d_value_cache[l]) hipFree(r->d_value_cache[l]);
        }
        free(r->d_value_cache);
    }

    if (r->layers) {
        for (int l = 0; l < r->n_layers; l++) {
            hip_layer *cl = &r->layers[l];
            if (cl->attn_norm_w)    hipFree(cl->attn_norm_w);
            if (cl->attn_q_w)      hipFree(cl->attn_q_w);
            if (cl->attn_k_w)      hipFree(cl->attn_k_w);
            if (cl->attn_v_w)      hipFree(cl->attn_v_w);
            if (cl->attn_q_bias)   hipFree(cl->attn_q_bias);
            if (cl->attn_k_bias)   hipFree(cl->attn_k_bias);
            if (cl->attn_v_bias)   hipFree(cl->attn_v_bias);
            if (cl->attn_q_norm_w) hipFree(cl->attn_q_norm_w);
            if (cl->attn_k_norm_w) hipFree(cl->attn_k_norm_w);
            if (cl->attn_output_w) hipFree(cl->attn_output_w);
            if (cl->ffn_norm_w)    hipFree(cl->ffn_norm_w);
            if (cl->ffn_gate_w)    hipFree(cl->ffn_gate_w);
            if (cl->ffn_up_w)     hipFree(cl->ffn_up_w);
            if (cl->ffn_down_w)   hipFree(cl->ffn_down_w);
            if (cl->attn_q_w_bf16)      hipFree(cl->attn_q_w_bf16);
            if (cl->attn_k_w_bf16)      hipFree(cl->attn_k_w_bf16);
            if (cl->attn_v_w_bf16)      hipFree(cl->attn_v_w_bf16);
            if (cl->attn_output_w_bf16) hipFree(cl->attn_output_w_bf16);
            if (cl->ffn_gate_w_bf16)    hipFree(cl->ffn_gate_w_bf16);
            if (cl->ffn_up_w_bf16)      hipFree(cl->ffn_up_w_bf16);
            if (cl->ffn_down_w_bf16)    hipFree(cl->ffn_down_w_bf16);
            if (cl->ssm_qkv_w)      hipFree(cl->ssm_qkv_w);
            if (cl->ssm_gate_w)     hipFree(cl->ssm_gate_w);
            if (cl->ssm_alpha_w)    hipFree(cl->ssm_alpha_w);
            if (cl->ssm_beta_w)     hipFree(cl->ssm_beta_w);
            if (cl->ssm_out_w)      hipFree(cl->ssm_out_w);
            if (cl->ssm_a)          hipFree(cl->ssm_a);
            if (cl->ssm_dt_bias)    hipFree(cl->ssm_dt_bias);
            if (cl->ssm_conv1d_w)   hipFree(cl->ssm_conv1d_w);
            if (cl->ssm_norm_w)     hipFree(cl->ssm_norm_w);
            if (cl->d_conv_state)      hipFree(cl->d_conv_state);
            if (cl->d_recurrent_state) hipFree(cl->d_recurrent_state);
            if (cl->moe_gate_w)             hipFree(cl->moe_gate_w);
            if (cl->moe_gate_exps_w)        hipFree(cl->moe_gate_exps_w);
            if (cl->moe_up_exps_w)          hipFree(cl->moe_up_exps_w);
            if (cl->moe_down_exps_w)        hipFree(cl->moe_down_exps_w);
            if (cl->moe_shared_gate_w)      hipFree(cl->moe_shared_gate_w);
            if (cl->moe_shared_ffn_gate_w)  hipFree(cl->moe_shared_ffn_gate_w);
            if (cl->moe_shared_ffn_up_w)    hipFree(cl->moe_shared_ffn_up_w);
            if (cl->moe_shared_ffn_down_w)  hipFree(cl->moe_shared_ffn_down_w);
        }
        free(r->layers);
    }

    if (r->d_token_embd) hipFree(r->d_token_embd);
    if (r->d_output_norm) hipFree(r->d_output_norm);
    if (r->d_output_w && r->d_output_w != r->d_token_embd) hipFree(r->d_output_w);
    if (r->d_logits) hipFree(r->d_logits);

    /* === Phase 5: graph + device-int cleanup === */
    if (r->graph_exec_logits) hipGraphExecDestroy(r->graph_exec_logits);
    if (r->graph_logits)      hipGraphDestroy(r->graph_logits);
    if (r->graph_exec_hidden) hipGraphExecDestroy(r->graph_exec_hidden);
    if (r->graph_hidden)      hipGraphDestroy(r->graph_hidden);
    if (r->d_position)        hipFree(r->d_position);

    if (r->module) hipModuleUnload(r->module);

#ifdef LLM_HIPBLASLT_ENABLED
    mm_blaslt_destroy();
#endif

    if (r->h_output_pinned) hipHostFree(r->h_output);
    else                    free(r->h_output);

    if (r->stream) hipStreamDestroy(r->stream);

    free(r);
}

/* ======================================================================== */
/* Public API: accessors                                                    */
/* ======================================================================== */

void hip_llm_reset_state(hip_llm_runner *r) {
    if (!r || !r->is_hybrid) return;
    for (int l = 0; l < r->n_layers; l++) {
        hip_layer *cl = &r->layers[l];
        if (!cl->is_ssm) continue;
        if (cl->d_conv_state) {
            size_t conv_bytes = (size_t)(r->ssm_conv_kernel - 1) * r->ssm_qkv_dim * sizeof(float);
            hipMemset(cl->d_conv_state, 0, conv_bytes);
        }
        if (cl->d_recurrent_state) {
            size_t rec_bytes = (size_t)r->ssm_dt_rank * r->ssm_d_state * r->ssm_d_state * sizeof(float);
            hipMemset(cl->d_recurrent_state, 0, rec_bytes);
        }
    }
}

int hip_llm_n_embd(const hip_llm_runner *r) { return r ? r->n_embd : 0; }
int hip_llm_n_layers(const hip_llm_runner *r) { return r ? r->n_layers : 0; }
int hip_llm_n_vocab(const hip_llm_runner *r) { return r ? r->n_vocab : 0; }
int hip_llm_max_seq_len(const hip_llm_runner *r) { return r ? r->max_seq_len : 0; }
