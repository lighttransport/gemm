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
"/* ---- 2b. rmsnorm_batch_f32: RMSNorm over n_rows rows, one block per row ---- */\n"
"__global__ void rmsnorm_batch_f32(float *dst, const float *x, const float *w,\n"
"                                    int n, int row_stride, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int row = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    const float *xrow = x   + (size_t)row * row_stride;\n"
"    float       *drow = dst + (size_t)row * row_stride;\n"
"    float sum = 0.0f;\n"
"    for (int i = tid; i < n; i += nthreads) {\n"
"        float v = xrow[i];\n"
"        sum += v * v;\n"
"    }\n"
"    sdata[tid] = sum;\n"
"    __syncthreads();\n"
"    for (int s = nthreads / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float scale = rsqrtf(sdata[0] / (float)n + eps);\n"
"    for (int i = tid; i < n; i += nthreads) {\n"
"        drow[i] = xrow[i] * scale * w[i];\n"
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
"/* ---- 4b. qknorm_batch_f32: per-row, per-head RMSNorm; grid=(n_heads, M) ---- */\n"
"__global__ void qknorm_batch_f32(float *vec, const float *w,\n"
"                                   int n_heads, int head_dim, int row_stride,\n"
"                                   float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int h   = blockIdx.x;\n"
"    int row = blockIdx.y;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    float *v = vec + (size_t)row * row_stride + h * head_dim;\n"
"    float val = (tid < head_dim) ? v[tid] : 0.0f;\n"
"    sdata[tid] = val * val;\n"
"    __syncthreads();\n"
"    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);\n"
"    if (tid < head_dim) v[tid] = val * scale * w[tid];\n"
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
"/* Same math as matvec_q4_K_f32, but one warp owns one output row.\n"
" * The decode shapes have nb ~= 16-20 blocks/row, so the old one-block-per-row\n"
" * kernel left most of its threads idle. This packs blockDim.x/32 rows/block. */\n"
"__global__ void matvec_q4_K_mw_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                     int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int rows_per_block = blockDim.x / 32;\n"
"    int row = blockIdx.x * rows_per_block + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 144;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 144;\n"
"        float d = half_to_float(*(const half_raw *)(bp));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *sc = bp + 4;\n"
"        const unsigned char *qs = bp + 16;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        int is = 0;\n"
"        #pragma unroll\n"
"        for (int j = 0; j < 4; j++) {\n"
"            unsigned char sv0, mv0, sv1, mv1;\n"
"            if (is < 4) { sv0 = sc[is] & 63; mv0 = sc[is+4] & 63; }\n"
"            else { sv0 = (sc[is+4]&0xF)|((sc[is-4]>>6)<<4); mv0 = (sc[is+4]>>4)|((sc[is]>>6)<<4); }\n"
"            if (is+1 < 4) { sv1 = sc[is+1] & 63; mv1 = sc[is+1+4] & 63; }\n"
"            else { sv1 = (sc[is+1+4]&0xF)|((sc[is+1-4]>>6)<<4); mv1 = (sc[is+1+4]>>4)|((sc[is+1]>>6)<<4); }\n"
"            float d1 = d * sv0, m1 = dmin * mv0;\n"
"            float d2 = d * sv1, m2 = dmin * mv1;\n"
"            const unsigned char *q = qs + j * 32;\n"
"            #pragma unroll 4\n"
"            for (int l = 0; l < 32; l++)\n"
"                partial += (d1 * (q[l] & 0xF) - m1) * xb[yi++];\n"
"            #pragma unroll 4\n"
"            for (int l = 0; l < 32; l++)\n"
"                partial += (d2 * (q[l] >> 4) - m2) * xb[yi++];\n"
"            is += 2;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
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
"        #pragma unroll\n"
"        for (int half = 0; half < 2; half++) {\n"
"            int base = half * 128;\n"
"            #pragma unroll 4\n"
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
"__global__ void matvec_q5_K_mw_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                     int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 176;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
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
"    if (lane == 0) dst[row] = sum;\n"
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
"/* ---- 19b. softplus_mul_batch_f32: same op over M rows of dt_rank ---- */\n"
"/* bias and a are dt_rank-wide constants; broadcast across the M rows. */\n"
"__global__ void softplus_mul_batch_f32(float *out, const float *in,\n"
"                                         const float *bias, const float *a,\n"
"                                         int dt_rank, int M) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = M * dt_rank;\n"
"    if (i >= total) return;\n"
"    int hi = i % dt_rank;\n"
"    float x = in[i] + bias[hi];\n"
"    float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));\n"
"    out[i] = sp * a[hi];\n"
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
"/* ---- 24b. deltanet_step_batch_f32: M sequential steps fused in one kernel ----\n"
" * Each thread r owns row r of the state matrix; the row is loaded into\n"
" * registers ONCE at the start, mutated through M token steps, written back\n"
" * ONCE at the end. Eliminates the M × (state_row R + W) global-memory\n"
" * traffic of the per-token launch_deltanet_step kernel.\n"
" *\n"
" * Layout: state is [n_heads, d_state, d_state] F32 persistent across calls.\n"
" *         Q/K/V_batch are [M, dt_rank, d_state] F32 (heads = dt_rank here).\n"
" *         alpha/beta_batch are [M, dt_rank] F32.\n"
" *         out_batch is [M, dt_rank, d_state] F32.\n"
" *\n"
" * Requires d_state <= 128 (compile-time max for the per-thread S_row array).\n"
" * The block dim is d_state; the grid dim is dt_rank. */\n"
"#define DLN_BATCH_MAX_DSTATE 128\n"
"__global__ void deltanet_step_batch_f32(\n"
"    float *state, float *out_batch,\n"
"    const float *Q_batch, const float *K_batch, const float *V_batch,\n"
"    const float *alpha_batch, const float *beta_batch,\n"
"    int dt_rank, int d_state, int v_row_stride, int M) {\n"
"    int h = blockIdx.x;\n"
"    int r = threadIdx.x;\n"
"    if (r >= d_state) return;\n"
"\n"
"    /* Load row r of the state matrix into per-thread registers. */\n"
"    float S_row[DLN_BATCH_MAX_DSTATE];\n"
"    const float *S_src = state + (size_t)h * d_state * d_state + r * d_state;\n"
"    for (int c = 0; c < d_state; c++) S_row[c] = S_src[c];\n"
"\n"
"    /* Shared K, Q broadcast staging — one warp's worth per axis. */\n"
"    __shared__ float sK[DLN_BATCH_MAX_DSTATE];\n"
"    __shared__ float sQ[DLN_BATCH_MAX_DSTATE];\n"
"\n"
"    float scale = rsqrtf((float)d_state);\n"
"    size_t per_tok_dt = (size_t)dt_rank * d_state;\n"
"\n"
"    for (int m = 0; m < M; m++) {\n"
"        size_t base    = (size_t)m * per_tok_dt + (size_t)h * d_state;\n"
"        size_t v_base  = (size_t)m * (size_t)v_row_stride + (size_t)h * d_state;\n"
"        float v_r   = V_batch[v_base + r];\n"
"        float decay = __expf(alpha_batch[(size_t)m * dt_rank + h]);\n"
"        float b     = beta_batch[(size_t)m * dt_rank + h];\n"
"\n"
"        /* Cooperatively stage k and q into LDS (every thread loads one elt). */\n"
"        sK[r] = K_batch[base + r];\n"
"        sQ[r] = Q_batch[base + r];\n"
"        __syncthreads();\n"
"\n"
"        /* Opp A2: 2-pass with 4-way unrolled accumulators to break the\n"
"         * per-FMA dependency chain. Per-step compute is dominated by two\n"
"         * dot products (sk, o); single-accumulator scalar form is bound by\n"
"         * the ~4-cycle FMA latency, single-issue. Splitting each dot into\n"
"         * 4 partial sums lets the SIMD lane issue 4 FMAs back-to-back. */\n"
"\n"
"        /* Pass 1: sk = decay * sum_c S_row[c] * sK[c]. The decay factor is\n"
"         * a per-step scalar, so pull it out of the sum and skip touching\n"
"         * S_row here — saves a full read+write pass. */\n"
"        float sk0 = 0.0f, sk1 = 0.0f, sk2 = 0.0f, sk3 = 0.0f;\n"
"        #pragma unroll 8\n"
"        for (int c = 0; c < d_state; c += 4) {\n"
"            sk0 += S_row[c+0] * sK[c+0];\n"
"            sk1 += S_row[c+1] * sK[c+1];\n"
"            sk2 += S_row[c+2] * sK[c+2];\n"
"            sk3 += S_row[c+3] * sK[c+3];\n"
"        }\n"
"        float sk    = ((sk0 + sk1) + (sk2 + sk3)) * decay;\n"
"        float delta = (v_r - sk) * b;\n"
"\n"
"        /* Pass 2: fuse state update with o-dot.\n"
"         *   S_row_new[c] = S_row[c] * decay + delta * sK[c]\n"
"         *   o            = sum_c S_row_new[c] * sQ[c]\n"
"         * One read + one write of S_row per c, 4-way unrolled. */\n"
"        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;\n"
"        #pragma unroll 8\n"
"        for (int c = 0; c < d_state; c += 4) {\n"
"            float s0 = S_row[c+0] * decay + delta * sK[c+0];\n"
"            float s1 = S_row[c+1] * decay + delta * sK[c+1];\n"
"            float s2 = S_row[c+2] * decay + delta * sK[c+2];\n"
"            float s3 = S_row[c+3] * decay + delta * sK[c+3];\n"
"            S_row[c+0] = s0; S_row[c+1] = s1;\n"
"            S_row[c+2] = s2; S_row[c+3] = s3;\n"
"            o0 += s0 * sQ[c+0];\n"
"            o1 += s1 * sQ[c+1];\n"
"            o2 += s2 * sQ[c+2];\n"
"            o3 += s3 * sQ[c+3];\n"
"        }\n"
"        float o = ((o0 + o1) + (o2 + o3)) * scale;\n"
"\n"
"        out_batch[base + r] = o;\n"
"        __syncthreads();   /* before next iter overwrites sK/sQ */\n"
"    }\n"
"\n"
"    /* Write the final state row back to global. */\n"
"    float *S_dst = state + (size_t)h * d_state * d_state + r * d_state;\n"
"    for (int c = 0; c < d_state; c++) S_dst[c] = S_row[c];\n"
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
"/* ===== Batched SSM aux ops (one launch over M rows) ===== */\n"
"\n"
"/* ---- 22b. l2_norm_heads_batch_f32: per-(m,h) RMSNorm over head_dim ----\n"
"   Grid (n_heads, M, 1); each block normalizes one head of one row. */\n"
"__global__ void l2_norm_heads_batch_f32(float *data, int n_heads, int head_dim,\n"
"                                          int row_stride, int M, float eps) {\n"
"    int h = blockIdx.x;\n"
"    int m = blockIdx.y;\n"
"    if (h >= n_heads || m >= M) return;\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    float *v = data + (size_t)m * row_stride + (size_t)h * head_dim;\n"
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
"/* ---- 23b. repeat_tile_batch_f32: broadcast n_group heads to dt_rank, M rows ---- */\n"
"__global__ void repeat_tile_batch_f32(float *dst, const float *src,\n"
"                                        int dt_rank, int d_state, int n_group,\n"
"                                        int src_row_stride, int dst_row_stride,\n"
"                                        int M) {\n"
"    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int per_row = dt_rank * d_state;\n"
"    int total = M * per_row;\n"
"    if (gid >= total) return;\n"
"    int m    = gid / per_row;\n"
"    int rest = gid - m * per_row;\n"
"    int h    = rest / d_state;\n"
"    int i    = rest - h * d_state;\n"
"    size_t dst_idx = (size_t)m * dst_row_stride + (size_t)h * d_state + i;\n"
"    size_t src_idx = (size_t)m * src_row_stride + (size_t)(h % n_group) * d_state + i;\n"
"    dst[dst_idx] = src[src_idx];\n"
"}\n"
"\n"
"/* ---- 25b. gated_rmsnorm_silu_batch_f32: per-(m,h) ----\n"
"   Grid (dt_rank, M, 1); each block does one row of [dt_rank, d_state]. */\n"
"__global__ void gated_rmsnorm_silu_batch_f32(\n"
"    float *out, const float *z, const float *norm_w,\n"
"    int dt_rank, int d_state, int row_stride, int M, float eps) {\n"
"    int h = blockIdx.x;\n"
"    int m = blockIdx.y;\n"
"    if (h >= dt_rank || m >= M) return;\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    float       *o  = out + (size_t)m * row_stride + (size_t)h * d_state;\n"
"    const float *zh = z   + (size_t)m * row_stride + (size_t)h * d_state;\n"
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
"/* ---- 21b. conv1d_depthwise_silu_batch_f32 ----\n"
"   Processes M sequential token steps for ONE column j per thread. Each thread\n"
"   keeps the conv_k weights + (conv_k-1)-element state in registers across\n"
"   the M iterations, eliminating the per-token global state R/W amplification\n"
"   (parallel to deltanet_step_batch). conv_k assumed <= 8. */\n"
"#define CONV1D_BATCH_MAX_K 8\n"
"__global__ void conv1d_depthwise_silu_batch_f32(\n"
"    float *conv_out_batch, float *conv_state,\n"
"    const float *input_batch, const float *weight,\n"
"    int qkv_dim, int conv_k, int row_stride, int M) {\n"
"    int j = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (j >= qkv_dim) return;\n"
"    /* Load this column's weights once. */\n"
"    float w[CONV1D_BATCH_MAX_K];\n"
"    for (int f = 0; f < conv_k; f++) w[f] = weight[j * conv_k + f];\n"
"    /* Load this column's persistent state row (conv_k-1 floats). */\n"
"    float st[CONV1D_BATCH_MAX_K];\n"
"    for (int f = 0; f < conv_k - 1; f++) st[f] = conv_state[f * qkv_dim + j];\n"
"    for (int m = 0; m < M; m++) {\n"
"        float in_val = input_batch[(size_t)m * row_stride + j];\n"
"        float sum = w[conv_k - 1] * in_val;\n"
"        for (int f = 0; f < conv_k - 1; f++) sum += w[f] * st[f];\n"
"        conv_out_batch[(size_t)m * row_stride + j] = sum / (1.0f + __expf(-sum));\n"
"        /* Shift state: drop oldest, append current input. */\n"
"        for (int f = 0; f < conv_k - 2; f++) st[f] = st[f + 1];\n"
"        st[conv_k - 2] = in_val;\n"
"    }\n"
"    /* Store final state back to global. */\n"
"    for (int f = 0; f < conv_k - 1; f++) conv_state[f * qkv_dim + j] = st[f];\n"
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
"/* ---- 27b. deinterleave_qgate_batch_f32: same op over M rows ---- */\n"
"__global__ void deinterleave_qgate_batch_f32(\n"
"    float *q_out, float *gate_out, const float *qfull,\n"
"    int n_heads, int head_dim, int M) {\n"
"    int q_dim = n_heads * head_dim;\n"
"    int total = M * q_dim;\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx >= total) return;\n"
"    int m  = idx / q_dim;\n"
"    int hi = idx % q_dim;\n"
"    int h  = hi / head_dim;\n"
"    int i  = hi % head_dim;\n"
"    size_t qf_base = (size_t)m * 2 * q_dim + (size_t)h * 2 * head_dim;\n"
"    q_out[idx]    = qfull[qf_base + i];\n"
"    gate_out[idx] = qfull[qf_base + head_dim + i];\n"
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
"/* IQ/TQ matvec kernels ported from cuda_llm_runner.c -- see block below */\n"
"\n"
"/* ---- matvec_q4_0_f32: Q4_0 matrix x F32 vector -> F32 ---- */\n"
"/* Native block: 18 bytes = [uint16 d][uint8 qs[16]], 32 elems, interleaved nibbles. */\n"
"__global__ void matvec_q4_0_f32(float *dst, const unsigned char *mat,\n"
"                                  const float *x, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int n_blocks_per_row = n_cols / 32;\n"
"    int row_bytes = n_blocks_per_row * 18;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < n_blocks_per_row; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 18;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 16; j++) {\n"
"            int v0 = (int)(qs[j] & 0x0F) - 8;\n"
"            int v1 = (int)(qs[j] >> 4)   - 8;\n"
"            partial += (float)v0 * xb[j] + (float)v1 * xb[j + 16];\n"
"        }\n"
"        sum += partial * d;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws40[8];\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    if (lane == 0) ws40[warp_id] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws40[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- matvec_q4_1_f32: Q4_1 matrix x F32 vector -> F32 ---- */\n"
"/* Native block: 20 bytes = [uint16 d][uint16 m][uint8 qs[16]], 32 elems. */\n"
"__global__ void matvec_q4_1_f32(float *dst, const unsigned char *mat,\n"
"                                  const float *x, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int n_blocks_per_row = n_cols / 32;\n"
"    int row_bytes = n_blocks_per_row * 20;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < n_blocks_per_row; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 20;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        float m = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *qs = bp + 4;\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 16; j++) {\n"
"            float v0 = (float)(qs[j] & 0x0F);\n"
"            float v1 = (float)(qs[j] >> 4);\n"
"            partial += (v0 * d + m) * xb[j] + (v1 * d + m) * xb[j + 16];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws41[8];\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    if (lane == 0) ws41[warp_id] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws41[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- matvec_q5_0_f32: Q5_0 matrix x F32 vector -> F32 ---- */\n"
"/* Native block: 22 bytes = [uint16 d][uint8 qh[4]][uint8 qs[16]], 32 elems. */\n"
"__global__ void matvec_q5_0_f32(float *dst, const unsigned char *mat,\n"
"                                  const float *x, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int n_blocks_per_row = n_cols / 32;\n"
"    int row_bytes = n_blocks_per_row * 22;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < n_blocks_per_row; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 22;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned int qh = (unsigned int)bp[2] | ((unsigned int)bp[3] << 8) |\n"
"                          ((unsigned int)bp[4] << 16) | ((unsigned int)bp[5] << 24);\n"
"        const unsigned char *qs = bp + 6;\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 16; j++) {\n"
"            unsigned int xh_0 = ((qh >> (j +  0)) << 4) & 0x10;\n"
"            unsigned int xh_1 = ((qh >> (j + 12))     ) & 0x10;\n"
"            int x0 = (int)((qs[j] & 0x0F) | xh_0) - 16;\n"
"            int x1 = (int)((qs[j] >> 4)   | xh_1) - 16;\n"
"            partial += (float)x0 * xb[j] + (float)x1 * xb[j + 16];\n"
"        }\n"
"        sum += partial * d;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws50[8];\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    if (lane == 0) ws50[warp_id] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws50[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- matvec_q5_1_f32: Q5_1 matrix x F32 vector -> F32 ---- */\n"
"/* Native block: 24 bytes = [uint16 d][uint16 m][uint8 qh[4]][uint8 qs[16]], 32 elems. */\n"
"__global__ void matvec_q5_1_f32(float *dst, const unsigned char *mat,\n"
"                                  const float *x, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int n_blocks_per_row = n_cols / 32;\n"
"    int row_bytes = n_blocks_per_row * 24;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < n_blocks_per_row; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 24;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        float m = half_to_float(*(const half_raw *)(bp + 2));\n"
"        unsigned int qh = (unsigned int)bp[4] | ((unsigned int)bp[5] << 8) |\n"
"                          ((unsigned int)bp[6] << 16) | ((unsigned int)bp[7] << 24);\n"
"        const unsigned char *qs = bp + 8;\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 16; j++) {\n"
"            unsigned int xh_0 = ((qh >> (j +  0)) << 4) & 0x10;\n"
"            unsigned int xh_1 = ((qh >> (j + 12))     ) & 0x10;\n"
"            float x0 = (float)((qs[j] & 0x0F) | xh_0);\n"
"            float x1 = (float)((qs[j] >> 4)   | xh_1);\n"
"            partial += (x0 * d + m) * xb[j] + (x1 * d + m) * xb[j + 16];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws51[8];\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    if (lane == 0) ws51[warp_id] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws51[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 31. matvec_iq2_xxs_f32: IQ2_XXS matrix x F32 vector -> F32 ---- */\n"
"/* IQ2_XXS block: 66 bytes = d(f16) + qs[32](uint16), 256 elements */\n"
"/* Lookup tables for IQ2_XXS codebook dequantization */\n"
"__device__ static const unsigned long long iq2xxs_grid_dev[256] = {\n"
"    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,\n"
"    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,\n"
"    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,\n"
"    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,\n"
"    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,\n"
"    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,\n"
"    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,\n"
"    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,\n"
"    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,\n"
"    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,\n"
"    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,\n"
"    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,\n"
"    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,\n"
"    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,\n"
"    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,\n"
"    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,\n"
"    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,\n"
"    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,\n"
"    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,\n"
"    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,\n"
"    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,\n"
"    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,\n"
"    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,\n"
"    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,\n"
"    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,\n"
"    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,\n"
"    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,\n"
"    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,\n"
"    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,\n"
"    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,\n"
"    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,\n"
"    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,\n"
"    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,\n"
"    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,\n"
"    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,\n"
"    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,\n"
"    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,\n"
"    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,\n"
"    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,\n"
"    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,\n"
"    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,\n"
"    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,\n"
"    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,\n"
"    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,\n"
"    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,\n"
"    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,\n"
"    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,\n"
"    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,\n"
"    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,\n"
"    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,\n"
"    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,\n"
"    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,\n"
"    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,\n"
"    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,\n"
"    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,\n"
"    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,\n"
"    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,\n"
"    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,\n"
"    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,\n"
"    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,\n"
"    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,\n"
"    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,\n"
"    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,\n"
"    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,\n"
"};\n"
"\n"
"__device__ static const unsigned char ksigns_iq2xs_dev[128] = {\n"
"      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,\n"
"    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,\n"
"    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,\n"
"     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,\n"
"    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,\n"
"     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,\n"
"     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,\n"
"    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,\n"
"};\n"
"\n"
"__global__ void matvec_iq2_xxs_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                     int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 66;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 66;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            unsigned int aux0 = qs[4*ib32] | ((unsigned int)qs[4*ib32+1] << 16);\n"
"            unsigned int aux1 = qs[4*ib32+2] | ((unsigned int)qs[4*ib32+3] << 16);\n"
"            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;\n"
"            const unsigned char *aux8 = (const unsigned char *)&aux0;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                const unsigned char *grid = (const unsigned char *)&iq2xxs_grid_dev[aux8[l]];\n"
"                unsigned char signs = ksigns_iq2xs_dev[(aux1 >> (7*l)) & 127];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    float w = db * (float)grid[j] * ((signs & (1 << j)) ? -1.0f : 1.0f);\n"
"                    partial += w * xb[yi++];\n"
"                }\n"
"            }\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"__device__ static const unsigned long long iq2xs_grid_dev[512] = {\n"
"    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,\n"
"    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL,\n"
"    0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL,\n"
"    0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL,\n"
"    0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL,\n"
"    0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL,\n"
"    0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL,\n"
"    0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL,\n"
"    0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL,\n"
"    0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL,\n"
"    0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL,\n"
"    0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL,\n"
"    0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL,\n"
"    0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL,\n"
"    0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL,\n"
"    0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL,\n"
"    0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL,\n"
"    0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL,\n"
"    0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL,\n"
"    0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL,\n"
"    0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL,\n"
"    0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL,\n"
"    0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL,\n"
"    0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL,\n"
"    0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL,\n"
"    0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL,\n"
"    0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL,\n"
"    0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL,\n"
"    0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL,\n"
"    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL,\n"
"    0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL,\n"
"    0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL,\n"
"    0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL,\n"
"    0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL,\n"
"    0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL,\n"
"    0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL,\n"
"    0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL,\n"
"    0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL,\n"
"    0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL,\n"
"    0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL,\n"
"    0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL,\n"
"    0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL,\n"
"    0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL,\n"
"    0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL,\n"
"    0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL,\n"
"    0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL,\n"
"    0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL,\n"
"    0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL,\n"
"    0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL,\n"
"    0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL,\n"
"    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL,\n"
"    0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL,\n"
"    0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL,\n"
"    0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL,\n"
"    0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL,\n"
"    0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL,\n"
"    0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL,\n"
"    0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL,\n"
"    0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL,\n"
"    0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL,\n"
"    0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL,\n"
"    0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL,\n"
"    0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL,\n"
"    0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL,\n"
"    0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL,\n"
"    0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL,\n"
"    0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL,\n"
"    0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL,\n"
"    0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL,\n"
"    0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL,\n"
"    0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL,\n"
"    0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL,\n"
"    0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL,\n"
"    0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL,\n"
"    0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL,\n"
"    0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL,\n"
"    0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL,\n"
"    0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL,\n"
"    0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL,\n"
"    0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL,\n"
"    0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL,\n"
"    0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL,\n"
"    0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL,\n"
"    0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL,\n"
"    0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL,\n"
"    0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL,\n"
"    0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL,\n"
"    0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL,\n"
"    0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL,\n"
"    0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL,\n"
"    0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL,\n"
"    0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL,\n"
"    0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL,\n"
"    0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL,\n"
"    0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL,\n"
"    0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL,\n"
"    0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL,\n"
"    0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL,\n"
"    0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL,\n"
"    0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL,\n"
"    0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL,\n"
"    0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL,\n"
"    0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL,\n"
"    0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL,\n"
"    0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL,\n"
"    0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL,\n"
"    0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL,\n"
"    0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL,\n"
"    0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL,\n"
"    0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL,\n"
"    0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL,\n"
"    0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL,\n"
"    0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL,\n"
"    0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL,\n"
"    0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL,\n"
"    0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL,\n"
"    0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL,\n"
"    0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL,\n"
"    0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL,\n"
"    0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL,\n"
"    0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL,\n"
"    0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL,\n"
"    0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL,\n"
"    0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL,\n"
"    0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL,\n"
"    0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL,\n"
"    0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL,\n"
"    0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL,\n"
"};\n"

"__device__ static const unsigned long long iq2s_grid_dev[1024] = {\n"
"    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,\n"
"    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL,\n"
"    0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL,\n"
"    0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL,\n"
"    0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL,\n"
"    0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x08080808192b192bULL,\n"
"    0x08080808192b2b19ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b081919ULL,\n"
"    0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b2b0808ULL,\n"
"    0x080808082b2b1919ULL, 0x080808082b2b2b2bULL, 0x0808081908080819ULL, 0x0808081908081908ULL,\n"
"    0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL,\n"
"    0x0808081908191919ULL, 0x0808081908192b08ULL, 0x08080819082b0819ULL, 0x08080819082b1908ULL,\n"
"    0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL,\n"
"    0x0808081919190819ULL, 0x0808081919191908ULL, 0x080808191919192bULL, 0x0808081919192b19ULL,\n"
"    0x08080819192b0808ULL, 0x08080819192b1919ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL,\n"
"    0x080808192b081908ULL, 0x080808192b190808ULL, 0x080808192b19082bULL, 0x080808192b191919ULL,\n"
"    0x080808192b2b0819ULL, 0x080808192b2b1908ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL,\n"
"    0x0808082b08081919ULL, 0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL,\n"
"    0x0808082b082b0808ULL, 0x0808082b082b2b2bULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL,\n"
"    0x0808082b1908192bULL, 0x0808082b19082b19ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL,\n"
"    0x0808082b2b080808ULL, 0x0808082b2b081919ULL, 0x0808082b2b082b2bULL, 0x0808082b2b191908ULL,\n"
"    0x0808082b2b2b082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL,\n"
"    0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL,\n"
"    0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x08081908082b192bULL,\n"
"    0x08081908082b2b19ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL,\n"
"    0x0808190819082b08ULL, 0x0808190819082b2bULL, 0x0808190819190819ULL, 0x0808190819191908ULL,\n"
"    0x080819081919192bULL, 0x0808190819192b19ULL, 0x08081908192b0808ULL, 0x08081908192b082bULL,\n"
"    0x08081908192b1919ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b08192bULL,\n"
"    0x080819082b082b19ULL, 0x080819082b190808ULL, 0x080819082b191919ULL, 0x080819082b192b08ULL,\n"
"    0x080819082b2b0819ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL,\n"
"    0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908082b2bULL, 0x0808191908190819ULL,\n"
"    0x0808191908191908ULL, 0x080819190819192bULL, 0x0808191908192b19ULL, 0x08081919082b0808ULL,\n"
"    0x08081919082b1919ULL, 0x08081919082b2b08ULL, 0x0808191919080819ULL, 0x0808191919081908ULL,\n"
"    0x080819191908192bULL, 0x0808191919082b19ULL, 0x0808191919190808ULL, 0x080819191919082bULL,\n"
"    0x0808191919191919ULL, 0x0808191919192b08ULL, 0x08081919192b0819ULL, 0x08081919192b1908ULL,\n"
"    0x080819192b080808ULL, 0x080819192b08082bULL, 0x080819192b081919ULL, 0x080819192b082b08ULL,\n"
"    0x080819192b190819ULL, 0x080819192b191908ULL, 0x080819192b2b0808ULL, 0x0808192b08080819ULL,\n"
"    0x0808192b08081908ULL, 0x0808192b0808192bULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL,\n"
"    0x0808192b08191919ULL, 0x0808192b19080808ULL, 0x0808192b19081919ULL, 0x0808192b19082b08ULL,\n"
"    0x0808192b19190819ULL, 0x0808192b19191908ULL, 0x0808192b192b0808ULL, 0x0808192b2b080819ULL,\n"
"    0x0808192b2b081908ULL, 0x0808192b2b190808ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL,\n"
"    0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL,\n"
"    0x08082b080819192bULL, 0x08082b0808192b19ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL,\n"
"    0x08082b08082b2b2bULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b081908192bULL,\n"
"    0x08082b0819082b19ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b0819191919ULL,\n"
"    0x08082b0819192b08ULL, 0x08082b08192b0819ULL, 0x08082b08192b1908ULL, 0x08082b082b080808ULL,\n"
"    0x08082b082b081919ULL, 0x08082b082b191908ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL,\n"
"    0x08082b1908081908ULL, 0x08082b1908190808ULL, 0x08082b190819082bULL, 0x08082b1908191919ULL,\n"
"    0x08082b1908192b08ULL, 0x08082b19082b0819ULL, 0x08082b1919080808ULL, 0x08082b1919081919ULL,\n"
"    0x08082b1919082b08ULL, 0x08082b1919190819ULL, 0x08082b1919191908ULL, 0x08082b19192b0808ULL,\n"
"    0x08082b192b080819ULL, 0x08082b192b190808ULL, 0x08082b2b08080808ULL, 0x08082b2b08190819ULL,\n"
"    0x08082b2b08191908ULL, 0x08082b2b082b082bULL, 0x08082b2b082b2b08ULL, 0x08082b2b082b2b2bULL,\n"
"    0x08082b2b19190808ULL, 0x08082b2b2b192b19ULL, 0x0819080808080819ULL, 0x0819080808081908ULL,\n"
"    0x081908080808192bULL, 0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL,\n"
"    0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL,\n"
"    0x08190808082b192bULL, 0x0819080819080808ULL, 0x081908081908082bULL, 0x0819080819081919ULL,\n"
"    0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x081908081919192bULL,\n"
"    0x0819080819192b19ULL, 0x08190808192b0808ULL, 0x08190808192b082bULL, 0x08190808192b1919ULL,\n"
"    0x08190808192b2b08ULL, 0x081908082b080819ULL, 0x081908082b081908ULL, 0x081908082b08192bULL,\n"
"    0x081908082b190808ULL, 0x081908082b191919ULL, 0x081908082b192b08ULL, 0x081908082b2b0819ULL,\n"
"    0x081908082b2b1908ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL,\n"
"    0x0819081908082b08ULL, 0x0819081908082b2bULL, 0x0819081908190819ULL, 0x0819081908191908ULL,\n"
"    0x081908190819192bULL, 0x0819081908192b19ULL, 0x08190819082b0808ULL, 0x08190819082b082bULL,\n"
"    0x08190819082b1919ULL, 0x08190819082b2b08ULL, 0x0819081919080819ULL, 0x0819081919081908ULL,\n"
"    0x081908191908192bULL, 0x0819081919082b19ULL, 0x0819081919190808ULL, 0x081908191919082bULL,\n"
"    0x0819081919191919ULL, 0x0819081919192b08ULL, 0x08190819192b0819ULL, 0x08190819192b1908ULL,\n"
"    0x081908192b080808ULL, 0x081908192b08082bULL, 0x081908192b081919ULL, 0x081908192b082b08ULL,\n"
"    0x081908192b190819ULL, 0x081908192b191908ULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL,\n"
"    0x0819082b08082b19ULL, 0x0819082b08190808ULL, 0x0819082b08191919ULL, 0x0819082b082b0819ULL,\n"
"    0x0819082b082b1908ULL, 0x0819082b19080808ULL, 0x0819082b19081919ULL, 0x0819082b19190819ULL,\n"
"    0x0819082b19191908ULL, 0x0819082b2b080819ULL, 0x0819082b2b081908ULL, 0x0819082b2b190808ULL,\n"
"    0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL, 0x0819190808082b08ULL,\n"
"    0x0819190808190819ULL, 0x0819190808191908ULL, 0x081919080819192bULL, 0x0819190808192b19ULL,\n"
"    0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x08191908082b2b08ULL, 0x0819190819080819ULL,\n"
"    0x0819190819081908ULL, 0x081919081908192bULL, 0x0819190819082b19ULL, 0x0819190819190808ULL,\n"
"    0x081919081919082bULL, 0x0819190819191919ULL, 0x0819190819192b08ULL, 0x08191908192b0819ULL,\n"
"    0x08191908192b1908ULL, 0x081919082b080808ULL, 0x081919082b08082bULL, 0x081919082b081919ULL,\n"
"    0x081919082b082b08ULL, 0x081919082b190819ULL, 0x081919082b191908ULL, 0x081919082b2b0808ULL,\n"
"    0x0819191908080819ULL, 0x0819191908081908ULL, 0x081919190808192bULL, 0x0819191908082b19ULL,\n"
"    0x0819191908190808ULL, 0x081919190819082bULL, 0x0819191908191919ULL, 0x0819191908192b08ULL,\n"
"    0x08191919082b0819ULL, 0x08191919082b1908ULL, 0x0819191919080808ULL, 0x081919191908082bULL,\n"
"    0x0819191919081919ULL, 0x0819191919082b08ULL, 0x0819191919190819ULL, 0x0819191919191908ULL,\n"
"    0x08191919192b0808ULL, 0x081919192b080819ULL, 0x081919192b081908ULL, 0x081919192b190808ULL,\n"
"    0x0819192b08080808ULL, 0x0819192b08081919ULL, 0x0819192b08082b08ULL, 0x0819192b08190819ULL,\n"
"    0x0819192b08191908ULL, 0x0819192b082b0808ULL, 0x0819192b19080819ULL, 0x0819192b19081908ULL,\n"
"    0x0819192b19190808ULL, 0x0819192b2b080808ULL, 0x0819192b2b2b2b2bULL, 0x08192b0808080819ULL,\n"
"    0x08192b0808081908ULL, 0x08192b080808192bULL, 0x08192b0808082b19ULL, 0x08192b0808190808ULL,\n"
"    0x08192b0808191919ULL, 0x08192b0808192b08ULL, 0x08192b08082b0819ULL, 0x08192b0819080808ULL,\n"
"    0x08192b081908082bULL, 0x08192b0819081919ULL, 0x08192b0819082b08ULL, 0x08192b0819190819ULL,\n"
"    0x08192b0819191908ULL, 0x08192b08192b0808ULL, 0x08192b082b080819ULL, 0x08192b082b081908ULL,\n"
"    0x08192b1908080808ULL, 0x08192b190808082bULL, 0x08192b1908081919ULL, 0x08192b1908082b08ULL,\n"
"    0x08192b1908190819ULL, 0x08192b1908191908ULL, 0x08192b19082b0808ULL, 0x08192b1919080819ULL,\n"
"    0x08192b1919081908ULL, 0x08192b1919190808ULL, 0x08192b19192b2b19ULL, 0x08192b192b2b082bULL,\n"
"    0x08192b2b08081908ULL, 0x08192b2b08190808ULL, 0x08192b2b19080808ULL, 0x08192b2b1919192bULL,\n"
"    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL, 0x082b080808082b08ULL,\n"
"    0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b08080819192bULL, 0x082b080808192b19ULL,\n"
"    0x082b0808082b0808ULL, 0x082b0808082b1919ULL, 0x082b0808082b2b2bULL, 0x082b080819080819ULL,\n"
"    0x082b080819081908ULL, 0x082b080819190808ULL, 0x082b08081919082bULL, 0x082b080819191919ULL,\n"
"    0x082b0808192b1908ULL, 0x082b08082b080808ULL, 0x082b08082b082b2bULL, 0x082b08082b191908ULL,\n"
"    0x082b08082b2b2b2bULL, 0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL,\n"
"    0x082b08190819082bULL, 0x082b081908191919ULL, 0x082b0819082b0819ULL, 0x082b081919080808ULL,\n"
"    0x082b08191908082bULL, 0x082b081919081919ULL, 0x082b081919190819ULL, 0x082b081919191908ULL,\n"
"    0x082b0819192b0808ULL, 0x082b08192b080819ULL, 0x082b08192b081908ULL, 0x082b08192b190808ULL,\n"
"    0x082b082b08080808ULL, 0x082b082b08082b2bULL, 0x082b082b082b082bULL, 0x082b082b082b2b08ULL,\n"
"    0x082b082b082b2b2bULL, 0x082b082b19081908ULL, 0x082b082b19190808ULL, 0x082b082b2b082b08ULL,\n"
"    0x082b082b2b082b2bULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL,\n"
"    0x082b19080808192bULL, 0x082b190808082b19ULL, 0x082b190808190808ULL, 0x082b190808191919ULL,\n"
"    0x082b190808192b08ULL, 0x082b1908082b0819ULL, 0x082b1908082b1908ULL, 0x082b190819080808ULL,\n"
"    0x082b19081908082bULL, 0x082b190819081919ULL, 0x082b190819082b08ULL, 0x082b190819190819ULL,\n"
"    0x082b190819191908ULL, 0x082b1908192b0808ULL, 0x082b19082b080819ULL, 0x082b19082b081908ULL,\n"
"    0x082b19082b190808ULL, 0x082b191908080808ULL, 0x082b191908081919ULL, 0x082b191908082b08ULL,\n"
"    0x082b191908190819ULL, 0x082b191908191908ULL, 0x082b1919082b0808ULL, 0x082b191919080819ULL,\n"
"    0x082b191919081908ULL, 0x082b191919190808ULL, 0x082b1919192b192bULL, 0x082b19192b080808ULL,\n"
"    0x082b192b08080819ULL, 0x082b192b08081908ULL, 0x082b192b08190808ULL, 0x082b192b19080808ULL,\n"
"    0x082b192b19192b19ULL, 0x082b2b0808080808ULL, 0x082b2b0808081919ULL, 0x082b2b0808190819ULL,\n"
"    0x082b2b0808191908ULL, 0x082b2b0819080819ULL, 0x082b2b0819081908ULL, 0x082b2b0819190808ULL,\n"
"    0x082b2b082b082b2bULL, 0x082b2b082b2b2b2bULL, 0x082b2b1908080819ULL, 0x082b2b1908081908ULL,\n"
"    0x082b2b1908190808ULL, 0x082b2b192b191919ULL, 0x082b2b2b08082b2bULL, 0x082b2b2b082b082bULL,\n"
"    0x082b2b2b192b1908ULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL,\n"
"    0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL,\n"
"    0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x1908080808192b2bULL,\n"
"    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x19080808082b192bULL, 0x1908080819080808ULL,\n"
"    0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL,\n"
"    0x1908080819190819ULL, 0x1908080819191908ULL, 0x190808081919192bULL, 0x1908080819192b19ULL,\n"
"    0x19080808192b0808ULL, 0x19080808192b082bULL, 0x19080808192b1919ULL, 0x190808082b080819ULL,\n"
"    0x190808082b081908ULL, 0x190808082b190808ULL, 0x190808082b191919ULL, 0x190808082b192b08ULL,\n"
"    0x190808082b2b0819ULL, 0x190808082b2b1908ULL, 0x1908081908080808ULL, 0x190808190808082bULL,\n"
"    0x1908081908081919ULL, 0x1908081908082b08ULL, 0x1908081908190819ULL, 0x1908081908191908ULL,\n"
"    0x190808190819192bULL, 0x1908081908192b19ULL, 0x19080819082b0808ULL, 0x19080819082b082bULL,\n"
"    0x19080819082b1919ULL, 0x1908081919080819ULL, 0x1908081919081908ULL, 0x190808191908192bULL,\n"
"    0x1908081919082b19ULL, 0x1908081919190808ULL, 0x190808191919082bULL, 0x1908081919191919ULL,\n"
"    0x1908081919192b08ULL, 0x19080819192b0819ULL, 0x19080819192b1908ULL, 0x190808192b080808ULL,\n"
"    0x190808192b08082bULL, 0x190808192b081919ULL, 0x190808192b082b08ULL, 0x190808192b190819ULL,\n"
"    0x190808192b191908ULL, 0x190808192b2b0808ULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL,\n"
"    0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b08191919ULL, 0x1908082b08192b08ULL,\n"
"    0x1908082b082b1908ULL, 0x1908082b19080808ULL, 0x1908082b19081919ULL, 0x1908082b19082b08ULL,\n"
"    0x1908082b19190819ULL, 0x1908082b19191908ULL, 0x1908082b192b0808ULL, 0x1908082b2b080819ULL,\n"
"    0x1908082b2b081908ULL, 0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL,\n"
"    0x1908190808082b08ULL, 0x1908190808082b2bULL, 0x1908190808190819ULL, 0x1908190808191908ULL,\n"
"    0x190819080819192bULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x19081908082b082bULL,\n"
"    0x19081908082b1919ULL, 0x19081908082b2b08ULL, 0x1908190819080819ULL, 0x1908190819081908ULL,\n"
"    0x190819081908192bULL, 0x1908190819082b19ULL, 0x1908190819190808ULL, 0x190819081919082bULL,\n"
"    0x1908190819191919ULL, 0x1908190819192b08ULL, 0x19081908192b0819ULL, 0x19081908192b1908ULL,\n"
"    0x190819082b080808ULL, 0x190819082b08082bULL, 0x190819082b081919ULL, 0x190819082b082b08ULL,\n"
"    0x190819082b190819ULL, 0x190819082b191908ULL, 0x190819082b2b0808ULL, 0x1908191908080819ULL,\n"
"    0x1908191908081908ULL, 0x190819190808192bULL, 0x1908191908082b19ULL, 0x1908191908190808ULL,\n"
"    0x190819190819082bULL, 0x1908191908191919ULL, 0x1908191908192b08ULL, 0x19081919082b0819ULL,\n"
"    0x19081919082b1908ULL, 0x1908191919080808ULL, 0x190819191908082bULL, 0x1908191919081919ULL,\n"
"    0x1908191919082b08ULL, 0x1908191919190819ULL, 0x1908191919191908ULL, 0x19081919192b0808ULL,\n"
"    0x19081919192b2b2bULL, 0x190819192b080819ULL, 0x190819192b081908ULL, 0x190819192b190808ULL,\n"
"    0x1908192b08080808ULL, 0x1908192b0808082bULL, 0x1908192b08081919ULL, 0x1908192b08082b08ULL,\n"
"    0x1908192b08190819ULL, 0x1908192b08191908ULL, 0x1908192b082b0808ULL, 0x1908192b19080819ULL,\n"
"    0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x1908192b2b080808ULL, 0x1908192b2b2b1919ULL,\n"
"    0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808082b19ULL, 0x19082b0808190808ULL,\n"
"    0x19082b080819082bULL, 0x19082b0808191919ULL, 0x19082b0808192b08ULL, 0x19082b08082b0819ULL,\n"
"    0x19082b08082b1908ULL, 0x19082b0819080808ULL, 0x19082b081908082bULL, 0x19082b0819081919ULL,\n"
"    0x19082b0819082b08ULL, 0x19082b0819190819ULL, 0x19082b0819191908ULL, 0x19082b08192b0808ULL,\n"
"    0x19082b082b081908ULL, 0x19082b082b190808ULL, 0x19082b1908080808ULL, 0x19082b190808082bULL,\n"
"    0x19082b1908081919ULL, 0x19082b1908082b08ULL, 0x19082b1908190819ULL, 0x19082b1908191908ULL,\n"
"    0x19082b19082b0808ULL, 0x19082b1919080819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL,\n"
"    0x19082b192b080808ULL, 0x19082b192b19192bULL, 0x19082b2b08080819ULL, 0x19082b2b08081908ULL,\n"
"    0x19082b2b08190808ULL, 0x19082b2b19080808ULL, 0x1919080808080808ULL, 0x191908080808082bULL,\n"
"    0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL,\n"
"    0x191908080819192bULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x19190808082b082bULL,\n"
"    0x19190808082b1919ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL, 0x1919080819081908ULL,\n"
"    0x191908081908192bULL, 0x1919080819082b19ULL, 0x1919080819190808ULL, 0x191908081919082bULL,\n"
"    0x1919080819191919ULL, 0x1919080819192b08ULL, 0x19190808192b0819ULL, 0x19190808192b1908ULL,\n"
"    0x191908082b080808ULL, 0x191908082b08082bULL, 0x191908082b081919ULL, 0x191908082b082b08ULL,\n"
"    0x191908082b190819ULL, 0x191908082b191908ULL, 0x1919081908080819ULL, 0x1919081908081908ULL,\n"
"    0x191908190808192bULL, 0x1919081908082b19ULL, 0x1919081908190808ULL, 0x191908190819082bULL,\n"
"    0x1919081908191919ULL, 0x1919081908192b08ULL, 0x19190819082b0819ULL, 0x19190819082b1908ULL,\n"
"    0x1919081919080808ULL, 0x191908191908082bULL, 0x1919081919081919ULL, 0x1919081919082b08ULL,\n"
"    0x1919081919190819ULL, 0x1919081919191908ULL, 0x19190819192b0808ULL, 0x191908192b080819ULL,\n"
"    0x191908192b081908ULL, 0x191908192b190808ULL, 0x1919082b08080808ULL, 0x1919082b08081919ULL,\n"
"    0x1919082b08082b08ULL, 0x1919082b08190819ULL, 0x1919082b08191908ULL, 0x1919082b082b0808ULL,\n"
"    0x1919082b19080819ULL, 0x1919082b19081908ULL, 0x1919082b19190808ULL, 0x1919082b192b2b19ULL,\n"
"    0x1919082b2b080808ULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x191919080808192bULL,\n"
"    0x1919190808082b19ULL, 0x1919190808190808ULL, 0x191919080819082bULL, 0x1919190808191919ULL,\n"
"    0x1919190808192b08ULL, 0x19191908082b0819ULL, 0x19191908082b1908ULL, 0x1919190819080808ULL,\n"
"    0x191919081908082bULL, 0x1919190819081919ULL, 0x1919190819082b08ULL, 0x1919190819190819ULL,\n"
"    0x1919190819191908ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b081908ULL,\n"
"    0x191919082b190808ULL, 0x1919191908080808ULL, 0x191919190808082bULL, 0x1919191908081919ULL,\n"
"    0x1919191908082b08ULL, 0x1919191908190819ULL, 0x1919191908191908ULL, 0x19191919082b0808ULL,\n"
"    0x1919191919080819ULL, 0x1919191919081908ULL, 0x1919191919190808ULL, 0x191919192b080808ULL,\n"
"    0x1919192b08080819ULL, 0x1919192b08081908ULL, 0x1919192b08190808ULL, 0x1919192b082b192bULL,\n"
"    0x1919192b19080808ULL, 0x19192b0808080808ULL, 0x19192b080808082bULL, 0x19192b0808081919ULL,\n"
"    0x19192b0808082b08ULL, 0x19192b0808190819ULL, 0x19192b0808191908ULL, 0x19192b08082b0808ULL,\n"
"    0x19192b0819080819ULL, 0x19192b0819081908ULL, 0x19192b0819190808ULL, 0x19192b0819192b2bULL,\n"
"    0x19192b082b080808ULL, 0x19192b1908080819ULL, 0x19192b1908081908ULL, 0x19192b1908190808ULL,\n"
"    0x19192b1919080808ULL, 0x19192b2b08080808ULL, 0x19192b2b08192b19ULL, 0x19192b2b2b081919ULL,\n"
"    0x19192b2b2b2b2b08ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b08080808192bULL,\n"
"    0x192b080808190808ULL, 0x192b08080819082bULL, 0x192b080808191919ULL, 0x192b080808192b08ULL,\n"
"    0x192b0808082b0819ULL, 0x192b0808082b1908ULL, 0x192b080819080808ULL, 0x192b080819081919ULL,\n"
"    0x192b080819082b08ULL, 0x192b080819190819ULL, 0x192b080819191908ULL, 0x192b0808192b0808ULL,\n"
"    0x192b08082b081908ULL, 0x192b08082b190808ULL, 0x192b081908080808ULL, 0x192b08190808082bULL,\n"
"    0x192b081908081919ULL, 0x192b081908082b08ULL, 0x192b081908190819ULL, 0x192b081908191908ULL,\n"
"    0x192b0819082b0808ULL, 0x192b081919080819ULL, 0x192b081919081908ULL, 0x192b081919190808ULL,\n"
"    0x192b08192b080808ULL, 0x192b08192b192b19ULL, 0x192b082b08081908ULL, 0x192b082b08190808ULL,\n"
"    0x192b082b19080808ULL, 0x192b082b1919192bULL, 0x192b082b2b2b0819ULL, 0x192b190808080808ULL,\n"
"    0x192b190808081919ULL, 0x192b190808082b08ULL, 0x192b190808190819ULL, 0x192b190808191908ULL,\n"
"    0x192b1908082b0808ULL, 0x192b190819080819ULL, 0x192b190819081908ULL, 0x192b190819190808ULL,\n"
"    0x192b19082b080808ULL, 0x192b191908080819ULL, 0x192b191908081908ULL, 0x192b191908190808ULL,\n"
"    0x192b191919080808ULL, 0x192b191919082b2bULL, 0x192b1919192b2b08ULL, 0x192b19192b19082bULL,\n"
"    0x192b192b08080808ULL, 0x192b192b2b191908ULL, 0x192b2b0808080819ULL, 0x192b2b0808081908ULL,\n"
"    0x192b2b0808190808ULL, 0x192b2b08192b1919ULL, 0x192b2b082b192b08ULL, 0x192b2b1908080808ULL,\n"
"    0x192b2b19082b2b2bULL, 0x192b2b2b1908082bULL, 0x192b2b2b2b2b0819ULL, 0x2b08080808080808ULL,\n"
"    0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL,\n"
"    0x2b08080808191908ULL, 0x2b08080808192b19ULL, 0x2b080808082b0808ULL, 0x2b080808082b1919ULL,\n"
"    0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808081919082bULL,\n"
"    0x2b08080819191919ULL, 0x2b08080819192b08ULL, 0x2b080808192b0819ULL, 0x2b0808082b080808ULL,\n"
"    0x2b0808082b081919ULL, 0x2b0808082b190819ULL, 0x2b0808082b191908ULL, 0x2b08081908080819ULL,\n"
"    0x2b08081908081908ULL, 0x2b08081908082b19ULL, 0x2b08081908190808ULL, 0x2b0808190819082bULL,\n"
"    0x2b08081908191919ULL, 0x2b08081908192b08ULL, 0x2b080819082b0819ULL, 0x2b080819082b1908ULL,\n"
"    0x2b08081919080808ULL, 0x2b0808191908082bULL, 0x2b08081919081919ULL, 0x2b08081919082b08ULL,\n"
"    0x2b08081919190819ULL, 0x2b08081919191908ULL, 0x2b0808192b080819ULL, 0x2b0808192b081908ULL,\n"
"    0x2b0808192b190808ULL, 0x2b0808192b2b2b19ULL, 0x2b08082b08080808ULL, 0x2b08082b08081919ULL,\n"
"    0x2b08082b08082b2bULL, 0x2b08082b08190819ULL, 0x2b08082b08191908ULL, 0x2b08082b19080819ULL,\n"
"    0x2b08082b19081908ULL, 0x2b08082b19190808ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL,\n"
"    0x2b0819080808192bULL, 0x2b08190808082b19ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL,\n"
"    0x2b08190808191919ULL, 0x2b08190808192b08ULL, 0x2b081908082b0819ULL, 0x2b08190819080808ULL,\n"
"    0x2b0819081908082bULL, 0x2b08190819081919ULL, 0x2b08190819082b08ULL, 0x2b08190819190819ULL,\n"
"    0x2b08190819191908ULL, 0x2b081908192b0808ULL, 0x2b0819082b080819ULL, 0x2b0819082b081908ULL,\n"
"    0x2b0819082b190808ULL, 0x2b08191908080808ULL, 0x2b0819190808082bULL, 0x2b08191908081919ULL,\n"
"    0x2b08191908082b08ULL, 0x2b08191908190819ULL, 0x2b08191908191908ULL, 0x2b081919082b0808ULL,\n"
"    0x2b08191919080819ULL, 0x2b08191919081908ULL, 0x2b08191919190808ULL, 0x2b0819192b080808ULL,\n"
"    0x2b0819192b082b2bULL, 0x2b08192b08080819ULL, 0x2b08192b08081908ULL, 0x2b08192b08190808ULL,\n"
"    0x2b08192b082b2b19ULL, 0x2b08192b19080808ULL, 0x2b082b0808080808ULL, 0x2b082b0808081919ULL,\n"
"    0x2b082b0808190819ULL, 0x2b082b0808191908ULL, 0x2b082b0819080819ULL, 0x2b082b0819081908ULL,\n"
"    0x2b082b0819190808ULL, 0x2b082b082b2b082bULL, 0x2b082b1908080819ULL, 0x2b082b1908081908ULL,\n"
"    0x2b082b1919080808ULL, 0x2b082b19192b1919ULL, 0x2b082b2b082b082bULL, 0x2b082b2b19192b08ULL,\n"
"    0x2b082b2b19192b2bULL, 0x2b082b2b2b08082bULL, 0x2b082b2b2b2b082bULL, 0x2b19080808080819ULL,\n"
"    0x2b19080808081908ULL, 0x2b19080808082b19ULL, 0x2b19080808190808ULL, 0x2b1908080819082bULL,\n"
"    0x2b19080808191919ULL, 0x2b19080808192b08ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL,\n"
"    0x2b1908081908082bULL, 0x2b19080819081919ULL, 0x2b19080819082b08ULL, 0x2b19080819190819ULL,\n"
"    0x2b19080819191908ULL, 0x2b190808192b0808ULL, 0x2b1908082b080819ULL, 0x2b1908082b081908ULL,\n"
"    0x2b1908082b190808ULL, 0x2b19081908080808ULL, 0x2b19081908081919ULL, 0x2b19081908190819ULL,\n"
"    0x2b19081908191908ULL, 0x2b19081919080819ULL, 0x2b19081919081908ULL, 0x2b19081919190808ULL,\n"
"    0x2b19081919192b2bULL, 0x2b19082b08080819ULL, 0x2b19082b08081908ULL, 0x2b19082b08190808ULL,\n"
"    0x2b19082b19080808ULL, 0x2b19082b2b2b192bULL, 0x2b19190808080808ULL, 0x2b1919080808082bULL,\n"
"    0x2b19190808081919ULL, 0x2b19190808082b08ULL, 0x2b19190808190819ULL, 0x2b19190808191908ULL,\n"
"    0x2b191908082b0808ULL, 0x2b19190819080819ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL,\n"
"    0x2b1919082b080808ULL, 0x2b1919082b19192bULL, 0x2b19191908080819ULL, 0x2b19191908081908ULL,\n"
"    0x2b19191908190808ULL, 0x2b19191919080808ULL, 0x2b1919192b192b08ULL, 0x2b1919192b2b0819ULL,\n"
"    0x2b19192b08080808ULL, 0x2b19192b1908192bULL, 0x2b19192b192b1908ULL, 0x2b192b0808080819ULL,\n"
"    0x2b192b0808081908ULL, 0x2b192b0808190808ULL, 0x2b192b08082b192bULL, 0x2b192b0819080808ULL,\n"
"    0x2b192b082b2b2b19ULL, 0x2b192b1908080808ULL, 0x2b192b1919082b19ULL, 0x2b192b191919082bULL,\n"
"    0x2b192b2b2b190808ULL, 0x2b2b080808080808ULL, 0x2b2b080808081919ULL, 0x2b2b080808082b2bULL,\n"
"    0x2b2b080808191908ULL, 0x2b2b0808082b082bULL, 0x2b2b0808082b2b2bULL, 0x2b2b080819080819ULL,\n"
"    0x2b2b080819081908ULL, 0x2b2b080819190808ULL, 0x2b2b08082b2b082bULL, 0x2b2b08082b2b2b2bULL,\n"
"    0x2b2b081919080808ULL, 0x2b2b0819192b1919ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b2bULL,\n"
"    0x2b2b082b082b082bULL, 0x2b2b082b082b2b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b08082bULL,\n"
"    0x2b2b082b2b082b08ULL, 0x2b2b082b2b082b2bULL, 0x2b2b082b2b2b2b08ULL, 0x2b2b190808080819ULL,\n"
"    0x2b2b190808081908ULL, 0x2b2b190808190808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b082b19ULL,\n"
"    0x2b2b19082b2b1908ULL, 0x2b2b191908080808ULL, 0x2b2b191908192b19ULL, 0x2b2b192b19190819ULL,\n"
"    0x2b2b2b0808082b2bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b082bULL, 0x2b2b2b1919191908ULL,\n"
"    0x2b2b2b192b08192bULL, 0x2b2b2b2b08082b08ULL, 0x2b2b2b2b08082b2bULL, 0x2b2b2b2b082b0808ULL,\n"
"    0x2b2b2b2b082b082bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b2b082b08ULL, 0x2b2b2b2b2b2b2b2bULL,\n"
"};\n"

"__device__ static const unsigned int iq3xxs_grid_dev[256] = {\n"
"    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,\n"
"    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,\n"
"    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,\n"
"    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,\n"
"    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,\n"
"    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,\n"
"    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,\n"
"    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,\n"
"    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,\n"
"    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,\n"
"    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,\n"
"    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,\n"
"    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,\n"
"    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,\n"
"    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,\n"
"    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,\n"
"    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,\n"
"    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,\n"
"    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,\n"
"    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,\n"
"    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,\n"
"    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,\n"
"    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,\n"
"    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,\n"
"    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,\n"
"    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,\n"
"    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,\n"
"    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,\n"
"    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,\n"
"    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,\n"
"    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,\n"
"    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,\n"
"};\n"

"__device__ static const unsigned int iq3s_grid_dev[512] = {\n"
"    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,\n"
"    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,\n"
"    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,\n"
"    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,\n"
"    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,\n"
"    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,\n"
"    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,\n"
"    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,\n"
"    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,\n"
"    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,\n"
"    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,\n"
"    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,\n"
"    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,\n"
"    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,\n"
"    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,\n"
"    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,\n"
"    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,\n"
"    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,\n"
"    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,\n"
"    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,\n"
"    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,\n"
"    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,\n"
"    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,\n"
"    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,\n"
"    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,\n"
"    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,\n"
"    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,\n"
"    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,\n"
"    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,\n"
"    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,\n"
"    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,\n"
"    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,\n"
"    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,\n"
"    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,\n"
"    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,\n"
"    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,\n"
"    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,\n"
"    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,\n"
"    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,\n"
"    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,\n"
"    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,\n"
"    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,\n"
"    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,\n"
"    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,\n"
"    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,\n"
"    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,\n"
"    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,\n"
"    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,\n"
"    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,\n"
"    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,\n"
"    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,\n"
"    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,\n"
"    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,\n"
"    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,\n"
"    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,\n"
"    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,\n"
"    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,\n"
"    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,\n"
"    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,\n"
"    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,\n"
"    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,\n"
"    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,\n"
"    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,\n"
"    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,\n"
"};\n"

"__device__ static const unsigned long long iq1s_grid_dev[2048] = {\n"
"    0xffffffffffffffffULL, 0xffffffffffffff01ULL, 0xffffffffffff0000ULL, 0xffffffffffff01ffULL,\n"
"    0xffffffffffff0101ULL, 0xffffffffff00ff00ULL, 0xffffffffff000000ULL, 0xffffffffff01ffffULL,\n"
"    0xffffffffff01ff01ULL, 0xffffffffff0101ffULL, 0xffffffffff010101ULL, 0xffffffff00ff0000ULL,\n"
"    0xffffffff0000ff00ULL, 0xffffffff000000ffULL, 0xffffffff00000001ULL, 0xffffffff00010000ULL,\n"
"    0xffffffff01ffffffULL, 0xffffffff01ffff01ULL, 0xffffffff01ff01ffULL, 0xffffffff01ff0101ULL,\n"
"    0xffffffff01000000ULL, 0xffffffff0101ffffULL, 0xffffffff0101ff01ULL, 0xffffffff010101ffULL,\n"
"    0xffffffff01010101ULL, 0xffffff00ffff00ffULL, 0xffffff00ffff0000ULL, 0xffffff00ff00ff00ULL,\n"
"    0xffffff00ff0000ffULL, 0xffffff00ff000001ULL, 0xffffff00ff000100ULL, 0xffffff00ff000101ULL,\n"
"    0xffffff00ff010000ULL, 0xffffff0000ffff00ULL, 0xffffff0000ff0001ULL, 0xffffff0000ff0100ULL,\n"
"    0xffffff000000ff01ULL, 0xffffff0000000000ULL, 0xffffff0000000101ULL, 0xffffff000001ff00ULL,\n"
"    0xffffff00000100ffULL, 0xffffff0000010001ULL, 0xffffff00000101ffULL, 0xffffff0001ff0000ULL,\n"
"    0xffffff000100ff00ULL, 0xffffff00010000ffULL, 0xffffff0001000001ULL, 0xffffff0001010000ULL,\n"
"    0xffffff01ffffffffULL, 0xffffff01ffffff01ULL, 0xffffff01ffff01ffULL, 0xffffff01ffff0101ULL,\n"
"    0xffffff01ff000000ULL, 0xffffff01ff01ffffULL, 0xffffff01ff01ff01ULL, 0xffffff01ff0101ffULL,\n"
"    0xffffff01ff010101ULL, 0xffffff0100ff0000ULL, 0xffffff010000ff00ULL, 0xffffff0100000100ULL,\n"
"    0xffffff01000100ffULL, 0xffffff0100010100ULL, 0xffffff0101ffffffULL, 0xffffff0101ffff01ULL,\n"
"    0xffffff0101ff01ffULL, 0xffffff0101ff0101ULL, 0xffffff010100ff00ULL, 0xffffff0101000000ULL,\n"
"    0xffffff0101000100ULL, 0xffffff010101ffffULL, 0xffffff010101ff01ULL, 0xffffff01010101ffULL,\n"
"    0xffffff0101010101ULL, 0xffff00ffff00ff00ULL, 0xffff00ffff0000ffULL, 0xffff00ffff000001ULL,\n"
"    0xffff00ffff010000ULL, 0xffff00ff00ffff00ULL, 0xffff00ff00ff0100ULL, 0xffff00ff00000000ULL,\n"
"    0xffff00ff00000101ULL, 0xffff00ff000100ffULL, 0xffff00ff00010000ULL, 0xffff00ff0100ff00ULL,\n"
"    0xffff00ff01000100ULL, 0xffff00ff01010000ULL, 0xffff0000ffffff00ULL, 0xffff0000ffff00ffULL,\n"
"    0xffff0000ffff0000ULL, 0xffff0000ffff0001ULL, 0xffff0000ff000000ULL, 0xffff0000ff0001ffULL,\n"
"    0xffff0000ff000101ULL, 0xffff0000ff010100ULL, 0xffff000000ffffffULL, 0xffff000000ff0000ULL,\n"
"    0xffff000000ff0101ULL, 0xffff00000000ffffULL, 0xffff00000000ff00ULL, 0xffff0000000000ffULL,\n"
"    0xffff000000000000ULL, 0xffff000000000001ULL, 0xffff000000000100ULL, 0xffff00000001ffffULL,\n"
"    0xffff00000001ff01ULL, 0xffff000000010000ULL, 0xffff0000000101ffULL, 0xffff000000010101ULL,\n"
"    0xffff000001ffff00ULL, 0xffff00000100ff00ULL, 0xffff000001000000ULL, 0xffff0000010001ffULL,\n"
"    0xffff000001000101ULL, 0xffff00000101ff00ULL, 0xffff0000010100ffULL, 0xffff000001010000ULL,\n"
"    0xffff000001010001ULL, 0xffff000001010100ULL, 0xffff0001ff0000ffULL, 0xffff0001ff000100ULL,\n"
"    0xffff000100ffff00ULL, 0xffff000100ff00ffULL, 0xffff00010000ffffULL, 0xffff00010000ff01ULL,\n"
"    0xffff000100000000ULL, 0xffff0001000001ffULL, 0xffff00010001ffffULL, 0xffff00010001ff00ULL,\n"
"    0xffff000100010001ULL, 0xffff000100010100ULL, 0xffff000101ff0000ULL, 0xffff00010100ff00ULL,\n"
"    0xffff0001010000ffULL, 0xffff000101000100ULL, 0xffff01ffffffffffULL, 0xffff01ffffffff01ULL,\n"
"    0xffff01ffffff01ffULL, 0xffff01ffffff0101ULL, 0xffff01ffff000000ULL, 0xffff01ffff01ffffULL,\n"
"    0xffff01ffff01ff01ULL, 0xffff01ffff0101ffULL, 0xffff01ffff010101ULL, 0xffff01ff00ff0000ULL,\n"
"    0xffff01ff0000ff00ULL, 0xffff01ff00000001ULL, 0xffff01ff00010000ULL, 0xffff01ff01ffffffULL,\n"
"    0xffff01ff01ffff01ULL, 0xffff01ff01ff01ffULL, 0xffff01ff01ff0101ULL, 0xffff01ff01000000ULL,\n"
"    0xffff01ff0101ffffULL, 0xffff01ff0101ff01ULL, 0xffff01ff010101ffULL, 0xffff01ff01010101ULL,\n"
"    0xffff0100ffff0000ULL, 0xffff0100ff00ff00ULL, 0xffff0100ff0000ffULL, 0xffff0100ff000100ULL,\n"
"    0xffff0100ff0100ffULL, 0xffff0100ff010000ULL, 0xffff010000ffff00ULL, 0xffff01000000ffffULL,\n"
"    0xffff01000000ff00ULL, 0xffff010000000000ULL, 0xffff01000001ff00ULL, 0xffff0100000100ffULL,\n"
"    0xffff010000010100ULL, 0xffff01000100ff00ULL, 0xffff0100010000ffULL, 0xffff010001000001ULL,\n"
"    0xffff010001000100ULL, 0xffff010001010000ULL, 0xffff0101ffffffffULL, 0xffff0101ffffff01ULL,\n"
"    0xffff0101ffff01ffULL, 0xffff0101ffff0101ULL, 0xffff0101ff000000ULL, 0xffff0101ff01ffffULL,\n"
"    0xffff0101ff01ff01ULL, 0xffff0101ff0101ffULL, 0xffff0101ff010101ULL, 0xffff010100ff0000ULL,\n"
"    0xffff01010000ff00ULL, 0xffff010100000100ULL, 0xffff01010001ff00ULL, 0xffff010100010000ULL,\n"
"    0xffff010101ffffffULL, 0xffff010101ffff01ULL, 0xffff010101ff0000ULL, 0xffff010101ff01ffULL,\n"
"    0xffff010101ff0101ULL, 0xffff010101000000ULL, 0xffff01010101ffffULL, 0xffff01010101ff01ULL,\n"
"    0xffff0101010101ffULL, 0xffff010101010101ULL, 0xff00ffffff00ffffULL, 0xff00ffffff00ff00ULL,\n"
"    0xff00ffffff0000ffULL, 0xff00ffffff000100ULL, 0xff00ffffff0100ffULL, 0xff00ffffff010000ULL,\n"
"    0xff00ffff00ffff00ULL, 0xff00ffff00ff00ffULL, 0xff00ffff0000ffffULL, 0xff00ffff00000000ULL,\n"
"    0xff00ffff000001ffULL, 0xff00ffff0001ff00ULL, 0xff00ffff000100ffULL, 0xff00ffff00010000ULL,\n"
"    0xff00ffff00010100ULL, 0xff00ffff0100ff00ULL, 0xff00ffff010000ffULL, 0xff00ffff01000001ULL,\n"
"    0xff00ffff0101ff00ULL, 0xff00ffff01010000ULL, 0xff00ff00ffffff00ULL, 0xff00ff00ffff00ffULL,\n"
"    0xff00ff00ffff0001ULL, 0xff00ff00ffff0100ULL, 0xff00ff00ff00ffffULL, 0xff00ff00ff00ff01ULL,\n"
"    0xff00ff00ff000000ULL, 0xff00ff00ff0001ffULL, 0xff00ff00ff01ff00ULL, 0xff00ff00ff0100ffULL,\n"
"    0xff00ff00ff010100ULL, 0xff00ff0000ff0000ULL, 0xff00ff0000ff0101ULL, 0xff00ff000000ffffULL,\n"
"    0xff00ff000000ff00ULL, 0xff00ff000000ff01ULL, 0xff00ff00000000ffULL, 0xff00ff0000000000ULL,\n"
"    0xff00ff0000000001ULL, 0xff00ff0000000100ULL, 0xff00ff000001ffffULL, 0xff00ff0000010000ULL,\n"
"    0xff00ff0001ff00ffULL, 0xff00ff000100ff01ULL, 0xff00ff0001000000ULL, 0xff00ff000101ff00ULL,\n"
"    0xff00ff00010100ffULL, 0xff00ff01ff00ff00ULL, 0xff00ff01ff0000ffULL, 0xff00ff01ff000001ULL,\n"
"    0xff00ff01ff010000ULL, 0xff00ff0100ffffffULL, 0xff00ff0100ff0001ULL, 0xff00ff0100ff0100ULL,\n"
"    0xff00ff010000ff01ULL, 0xff00ff0100000000ULL, 0xff00ff01000001ffULL, 0xff00ff0100000101ULL,\n"
"    0xff00ff01000100ffULL, 0xff00ff0100010001ULL, 0xff00ff0101ff0000ULL, 0xff00ff010100ff00ULL,\n"
"    0xff00ff01010000ffULL, 0xff00ff0101000001ULL, 0xff00ff0101010000ULL, 0xff0000ffffffff00ULL,\n"
"    0xff0000ffffff0001ULL, 0xff0000ffffff0100ULL, 0xff0000ffff0000ffULL, 0xff0000ffff000000ULL,\n"
"    0xff0000ffff0001ffULL, 0xff0000ffff000100ULL, 0xff0000ffff01ff00ULL, 0xff0000ffff010001ULL,\n"
"    0xff0000ff00ffff00ULL, 0xff0000ff00ff0000ULL, 0xff0000ff00ff0001ULL, 0xff0000ff00ff01ffULL,\n"
"    0xff0000ff00ff0101ULL, 0xff0000ff0000ff00ULL, 0xff0000ff000000ffULL, 0xff0000ff00000000ULL,\n"
"    0xff0000ff00000001ULL, 0xff0000ff00000100ULL, 0xff0000ff0001ff01ULL, 0xff0000ff00010000ULL,\n"
"    0xff0000ff000101ffULL, 0xff0000ff01ff00ffULL, 0xff0000ff01ff0100ULL, 0xff0000ff0100ffffULL,\n"
"    0xff0000ff010000ffULL, 0xff0000ff01000000ULL, 0xff0000ff010001ffULL, 0xff0000ff01000100ULL,\n"
"    0xff0000ff01000101ULL, 0xff0000ff0101ff00ULL, 0xff0000ff010100ffULL, 0xff0000ff01010000ULL,\n"
"    0xff0000ff01010100ULL, 0xff000000ffffff01ULL, 0xff000000ffff0000ULL, 0xff000000ffff0101ULL,\n"
"    0xff000000ff00ff00ULL, 0xff000000ff0000ffULL, 0xff000000ff000000ULL, 0xff000000ff000001ULL,\n"
"    0xff000000ff000100ULL, 0xff000000ff01ffffULL, 0xff000000ff01ff01ULL, 0xff000000ff010000ULL,\n"
"    0xff000000ff0101ffULL, 0xff000000ff010101ULL, 0xff00000000ffff00ULL, 0xff00000000ff00ffULL,\n"
"    0xff00000000ff0000ULL, 0xff00000000ff0001ULL, 0xff0000000000ff00ULL, 0xff0000000000ff01ULL,\n"
"    0xff000000000000ffULL, 0xff00000000000000ULL, 0xff00000000000001ULL, 0xff00000000000100ULL,\n"
"    0xff00000000000101ULL, 0xff0000000001ff00ULL, 0xff000000000100ffULL, 0xff00000000010000ULL,\n"
"    0xff00000000010001ULL, 0xff00000000010100ULL, 0xff00000001ffffffULL, 0xff00000001ffff01ULL,\n"
"    0xff00000001ff00ffULL, 0xff00000001ff0000ULL, 0xff00000001ff01ffULL, 0xff00000001ff0101ULL,\n"
"    0xff0000000100ffffULL, 0xff0000000100ff00ULL, 0xff000000010000ffULL, 0xff00000001000000ULL,\n"
"    0xff00000001000001ULL, 0xff00000001000100ULL, 0xff00000001000101ULL, 0xff0000000101ffffULL,\n"
"    0xff0000000101ff01ULL, 0xff00000001010000ULL, 0xff000001ffffff00ULL, 0xff000001ffff00ffULL,\n"
"    0xff000001ffff0000ULL, 0xff000001ffff0001ULL, 0xff000001ff000000ULL, 0xff000001ff000001ULL,\n"
"    0xff000001ff0001ffULL, 0xff000001ff000101ULL, 0xff000001ff01ff00ULL, 0xff000001ff010001ULL,\n"
"    0xff00000100ffffffULL, 0xff00000100ffff01ULL, 0xff00000100ff00ffULL, 0xff00000100ff0000ULL,\n"
"    0xff00000100ff01ffULL, 0xff00000100ff0101ULL, 0xff0000010000ff00ULL, 0xff00000100000000ULL,\n"
"    0xff00000100000001ULL, 0xff000001000001ffULL, 0xff00000100000100ULL, 0xff0000010001ff00ULL,\n"
"    0xff000001000100ffULL, 0xff00000100010000ULL, 0xff000001000101ffULL, 0xff00000100010100ULL,\n"
"    0xff00000100010101ULL, 0xff00000101ff0001ULL, 0xff00000101ff0101ULL, 0xff0000010100ff01ULL,\n"
"    0xff00000101000000ULL, 0xff000001010100ffULL, 0xff00000101010100ULL, 0xff0001ffff00ff00ULL,\n"
"    0xff0001ffff000001ULL, 0xff0001ffff010000ULL, 0xff0001ff00ffff00ULL, 0xff0001ff00ff00ffULL,\n"
"    0xff0001ff00ff0001ULL, 0xff0001ff00ff0100ULL, 0xff0001ff0000ffffULL, 0xff0001ff00000000ULL,\n"
"    0xff0001ff000001ffULL, 0xff0001ff00000101ULL, 0xff0001ff0001ffffULL, 0xff0001ff0001ff00ULL,\n"
"    0xff0001ff000100ffULL, 0xff0001ff00010001ULL, 0xff0001ff00010100ULL, 0xff0001ff01ff0000ULL,\n"
"    0xff0001ff0100ff00ULL, 0xff0001ff010000ffULL, 0xff0001ff01010000ULL, 0xff000100ff00ffffULL,\n"
"    0xff000100ff00ff01ULL, 0xff000100ff000000ULL, 0xff000100ff000101ULL, 0xff000100ff01ff00ULL,\n"
"    0xff000100ff010000ULL, 0xff00010000ffff01ULL, 0xff00010000ff00ffULL, 0xff00010000ff0000ULL,\n"
"    0xff00010000ff01ffULL, 0xff0001000000ff00ULL, 0xff000100000000ffULL, 0xff00010000000000ULL,\n"
"    0xff00010000000001ULL, 0xff00010000000100ULL, 0xff00010000000101ULL, 0xff0001000001ffffULL,\n"
"    0xff00010000010000ULL, 0xff00010000010101ULL, 0xff00010001ff0100ULL, 0xff0001000100ff00ULL,\n"
"    0xff0001000100ff01ULL, 0xff00010001000000ULL, 0xff000100010001ffULL, 0xff0001000101ff00ULL,\n"
"    0xff00010001010001ULL, 0xff00010001010100ULL, 0xff000101ffff0100ULL, 0xff000101ff000001ULL,\n"
"    0xff000101ff0100ffULL, 0xff000101ff010001ULL, 0xff00010100ff00ffULL, 0xff00010100ff0001ULL,\n"
"    0xff00010100ff0100ULL, 0xff0001010000ffffULL, 0xff0001010000ff01ULL, 0xff00010100000000ULL,\n"
"    0xff000101000001ffULL, 0xff0001010001ff00ULL, 0xff00010100010001ULL, 0xff00010100010100ULL,\n"
"    0xff00010101ff0000ULL, 0xff0001010100ff00ULL, 0xff00010101000001ULL, 0xff00010101000101ULL,\n"
"    0xff01ffffffffffffULL, 0xff01ffffffffff01ULL, 0xff01ffffffff01ffULL, 0xff01ffffffff0101ULL,\n"
"    0xff01ffffff000000ULL, 0xff01ffffff01ffffULL, 0xff01ffffff01ff01ULL, 0xff01ffffff010000ULL,\n"
"    0xff01ffffff0101ffULL, 0xff01ffffff010101ULL, 0xff01ffff00ff0000ULL, 0xff01ffff0000ff00ULL,\n"
"    0xff01ffff00000100ULL, 0xff01ffff0001ff00ULL, 0xff01ffff00010000ULL, 0xff01ffff01ffffffULL,\n"
"    0xff01ffff01ffff01ULL, 0xff01ffff01ff01ffULL, 0xff01ffff01ff0101ULL, 0xff01ffff01000000ULL,\n"
"    0xff01ffff0101ffffULL, 0xff01ffff0101ff01ULL, 0xff01ffff01010000ULL, 0xff01ffff010101ffULL,\n"
"    0xff01ffff01010101ULL, 0xff01ff00ffff0000ULL, 0xff01ff00ff00ff00ULL, 0xff01ff00ff0000ffULL,\n"
"    0xff01ff00ff000100ULL, 0xff01ff00ff010000ULL, 0xff01ff0000ffff01ULL, 0xff01ff0000ff00ffULL,\n"
"    0xff01ff0000ff0100ULL, 0xff01ff0000000000ULL, 0xff01ff00000001ffULL, 0xff01ff0000000101ULL,\n"
"    0xff01ff000001ff00ULL, 0xff01ff00000100ffULL, 0xff01ff0000010000ULL, 0xff01ff0000010001ULL,\n"
"    0xff01ff0001ff0000ULL, 0xff01ff000100ffffULL, 0xff01ff0001000001ULL, 0xff01ff0001000100ULL,\n"
"    0xff01ff0001010000ULL, 0xff01ff01ffffff00ULL, 0xff01ff01ffff01ffULL, 0xff01ff01ffff0101ULL,\n"
"    0xff01ff01ff00ff00ULL, 0xff01ff01ff000000ULL, 0xff01ff01ff01ffffULL, 0xff01ff01ff01ff01ULL,\n"
"    0xff01ff01ff0101ffULL, 0xff01ff01ff010101ULL, 0xff01ff0100ff0000ULL, 0xff01ff010000ff00ULL,\n"
"    0xff01ff0100000001ULL, 0xff01ff0100000100ULL, 0xff01ff0100010000ULL, 0xff01ff0101ffff00ULL,\n"
"    0xff01ff0101ff01ffULL, 0xff01ff0101ff0101ULL, 0xff01ff010100ff00ULL, 0xff01ff0101000000ULL,\n"
"    0xff01ff010101ffffULL, 0xff01ff010101ff01ULL, 0xff01ff01010101ffULL, 0xff01ff0101010101ULL,\n"
"    0xff0100ffffff0000ULL, 0xff0100ffff0000ffULL, 0xff0100ffff000001ULL, 0xff0100ffff000100ULL,\n"
"    0xff0100ffff010000ULL, 0xff0100ff00ff00ffULL, 0xff0100ff00ff0000ULL, 0xff0100ff00ff0001ULL,\n"
"    0xff0100ff00ff0100ULL, 0xff0100ff0000ff01ULL, 0xff0100ff00000000ULL, 0xff0100ff000001ffULL,\n"
"    0xff0100ff00000101ULL, 0xff0100ff00010001ULL, 0xff0100ff01ff0000ULL, 0xff0100ff0100ff00ULL,\n"
"    0xff0100ff010000ffULL, 0xff0100ff01000100ULL, 0xff0100ff0101ff00ULL, 0xff0100ff01010000ULL,\n"
"    0xff010000ffff0100ULL, 0xff010000ff000000ULL, 0xff010000ff01ff00ULL, 0xff010000ff010100ULL,\n"
"    0xff01000000ffffffULL, 0xff01000000ff0000ULL, 0xff01000000ff01ffULL, 0xff0100000000ff00ULL,\n"
"    0xff010000000000ffULL, 0xff01000000000000ULL, 0xff01000000000100ULL, 0xff0100000001ff01ULL,\n"
"    0xff01000000010000ULL, 0xff010000000101ffULL, 0xff01000001ff0100ULL, 0xff0100000100ffffULL,\n"
"    0xff010000010000ffULL, 0xff01000001000000ULL, 0xff010000010001ffULL, 0xff01000001000101ULL,\n"
"    0xff0100000101ff00ULL, 0xff010000010100ffULL, 0xff01000001010001ULL, 0xff01000001010100ULL,\n"
"    0xff010001ffff0000ULL, 0xff010001ff00ffffULL, 0xff010001ff00ff01ULL, 0xff010001ff000100ULL,\n"
"    0xff010001ff010000ULL, 0xff01000100ffff00ULL, 0xff01000100ff0100ULL, 0xff01000100000000ULL,\n"
"    0xff0100010001ffffULL, 0xff0100010001ff00ULL, 0xff01000100010100ULL, 0xff01000101ff00ffULL,\n"
"    0xff01000101ff0001ULL, 0xff0100010100ffffULL, 0xff01000101000101ULL, 0xff0101ffffffffffULL,\n"
"    0xff0101ffffffff01ULL, 0xff0101ffffff01ffULL, 0xff0101ffffff0101ULL, 0xff0101ffff000000ULL,\n"
"    0xff0101ffff01ffffULL, 0xff0101ffff01ff01ULL, 0xff0101ffff0101ffULL, 0xff0101ffff010101ULL,\n"
"    0xff0101ff00ff0000ULL, 0xff0101ff0000ff00ULL, 0xff0101ff000000ffULL, 0xff0101ff00010000ULL,\n"
"    0xff0101ff01ffffffULL, 0xff0101ff01ffff01ULL, 0xff0101ff01ff01ffULL, 0xff0101ff01ff0101ULL,\n"
"    0xff0101ff0101ffffULL, 0xff0101ff0101ff01ULL, 0xff0101ff010101ffULL, 0xff0101ff01010101ULL,\n"
"    0xff010100ffff0100ULL, 0xff010100ff00ff00ULL, 0xff010100ff0000ffULL, 0xff010100ff000100ULL,\n"
"    0xff010100ff010000ULL, 0xff01010000ff0001ULL, 0xff01010000ff0100ULL, 0xff0101000000ff01ULL,\n"
"    0xff01010000000000ULL, 0xff0101000001ff00ULL, 0xff010100000100ffULL, 0xff01010000010001ULL,\n"
"    0xff01010000010100ULL, 0xff01010001ff0000ULL, 0xff0101000100ffffULL, 0xff01010001000001ULL,\n"
"    0xff01010001000100ULL, 0xff010100010100ffULL, 0xff01010001010000ULL, 0xff010101ffffffffULL,\n"
"    0xff010101ffffff01ULL, 0xff010101ffff01ffULL, 0xff010101ffff0101ULL, 0xff010101ff01ffffULL,\n"
"    0xff010101ff01ff01ULL, 0xff010101ff0101ffULL, 0xff010101ff010101ULL, 0xff01010100ff0000ULL,\n"
"    0xff0101010000ff00ULL, 0xff01010100000001ULL, 0xff01010100000100ULL, 0xff01010100010000ULL,\n"
"    0xff01010101ffffffULL, 0xff01010101ffff01ULL, 0xff01010101ff01ffULL, 0xff01010101ff0101ULL,\n"
"    0xff01010101000000ULL, 0xff0101010101ffffULL, 0xff0101010101ff01ULL, 0xff010101010101ffULL,\n"
"    0xff01010101010101ULL, 0x00ffffffffff0000ULL, 0x00ffffffff00ff00ULL, 0x00ffffffff000001ULL,\n"
"    0x00ffffffff010000ULL, 0x00ffffff00ff0100ULL, 0x00ffffff0000ff01ULL, 0x00ffffff00000000ULL,\n"
"    0x00ffffff000001ffULL, 0x00ffffff00000101ULL, 0x00ffffff0001ff00ULL, 0x00ffffff000100ffULL,\n"
"    0x00ffffff00010001ULL, 0x00ffffff010000ffULL, 0x00ffffff01000100ULL, 0x00ffffff0101ff00ULL,\n"
"    0x00ffffff01010001ULL, 0x00ffff00ffffffffULL, 0x00ffff00ffffff00ULL, 0x00ffff00ffff00ffULL,\n"
"    0x00ffff00ffff0001ULL, 0x00ffff00ffff0100ULL, 0x00ffff00ff00ff01ULL, 0x00ffff00ff000000ULL,\n"
"    0x00ffff00ff000001ULL, 0x00ffff00ff0001ffULL, 0x00ffff00ff000101ULL, 0x00ffff00ff01ff00ULL,\n"
"    0x00ffff00ff010001ULL, 0x00ffff00ff010100ULL, 0x00ffff0000ff0000ULL, 0x00ffff0000ff01ffULL,\n"
"    0x00ffff0000ff0101ULL, 0x00ffff000000ff00ULL, 0x00ffff00000000ffULL, 0x00ffff0000000000ULL,\n"
"    0x00ffff0000000001ULL, 0x00ffff0000000100ULL, 0x00ffff0000000101ULL, 0x00ffff0000010000ULL,\n"
"    0x00ffff00000101ffULL, 0x00ffff0000010101ULL, 0x00ffff0001ffff00ULL, 0x00ffff0001ff00ffULL,\n"
"    0x00ffff0001ff0001ULL, 0x00ffff000100ffffULL, 0x00ffff000100ff01ULL, 0x00ffff0001000000ULL,\n"
"    0x00ffff000101ffffULL, 0x00ffff000101ff00ULL, 0x00ffff000101ff01ULL, 0x00ffff01ffff0000ULL,\n"
"    0x00ffff01ff00ff00ULL, 0x00ffff01ff0000ffULL, 0x00ffff01ff000001ULL, 0x00ffff01ff010000ULL,\n"
"    0x00ffff0100ffff00ULL, 0x00ffff010000ff01ULL, 0x00ffff0100000000ULL, 0x00ffff0100000101ULL,\n"
"    0x00ffff01000100ffULL, 0x00ffff0100010100ULL, 0x00ffff0101ff0100ULL, 0x00ffff01010000ffULL,\n"
"    0x00ffff0101010000ULL, 0x00ff00ffffffff00ULL, 0x00ff00ffff000000ULL, 0x00ff00ffff000100ULL,\n"
"    0x00ff00ffff010100ULL, 0x00ff00ff00ff0000ULL, 0x00ff00ff00ff01ffULL, 0x00ff00ff00ff0101ULL,\n"
"    0x00ff00ff0000ff00ULL, 0x00ff00ff000000ffULL, 0x00ff00ff00000000ULL, 0x00ff00ff00000001ULL,\n"
"    0x00ff00ff0001ff00ULL, 0x00ff00ff0001ff01ULL, 0x00ff00ff00010000ULL, 0x00ff00ff000101ffULL,\n"
"    0x00ff00ff00010101ULL, 0x00ff00ff01ffff00ULL, 0x00ff00ff01ff0001ULL, 0x00ff00ff01ff0100ULL,\n"
"    0x00ff00ff0100ffffULL, 0x00ff00ff0100ff01ULL, 0x00ff00ff01000000ULL, 0x00ff00ff0101ffffULL,\n"
"    0x00ff00ff0101ff00ULL, 0x00ff00ff01010100ULL, 0x00ff0000ffffff00ULL, 0x00ff0000ffffff01ULL,\n"
"    0x00ff0000ffff0000ULL, 0x00ff0000ffff0101ULL, 0x00ff0000ff00ff00ULL, 0x00ff0000ff0000ffULL,\n"
"    0x00ff0000ff000000ULL, 0x00ff0000ff000001ULL, 0x00ff0000ff000100ULL, 0x00ff0000ff01ffffULL,\n"
"    0x00ff0000ff010000ULL, 0x00ff0000ff010101ULL, 0x00ff000000ffff00ULL, 0x00ff000000ff00ffULL,\n"
"    0x00ff000000ff0000ULL, 0x00ff000000ff0001ULL, 0x00ff000000ff0100ULL, 0x00ff00000000ffffULL,\n"
"    0x00ff00000000ff00ULL, 0x00ff0000000000ffULL, 0x00ff000000000000ULL, 0x00ff000000000001ULL,\n"
"    0x00ff0000000001ffULL, 0x00ff000000000100ULL, 0x00ff00000001ff00ULL, 0x00ff0000000100ffULL,\n"
"    0x00ff000000010000ULL, 0x00ff000000010001ULL, 0x00ff000000010100ULL, 0x00ff000001ffff01ULL,\n"
"    0x00ff000001ff00ffULL, 0x00ff000001ff0000ULL, 0x00ff000001ff01ffULL, 0x00ff00000100ff00ULL,\n"
"    0x00ff0000010000ffULL, 0x00ff000001000000ULL, 0x00ff000001000001ULL, 0x00ff000001000100ULL,\n"
"    0x00ff000001000101ULL, 0x00ff000001010000ULL, 0x00ff0000010101ffULL, 0x00ff000001010101ULL,\n"
"    0x00ff0001ffffff00ULL, 0x00ff0001ffff0000ULL, 0x00ff0001ffff0100ULL, 0x00ff0001ff0000ffULL,\n"
"    0x00ff0001ff000000ULL, 0x00ff0001ff0001ffULL, 0x00ff0001ff000101ULL, 0x00ff0001ff01ff00ULL,\n"
"    0x00ff0001ff0100ffULL, 0x00ff0001ff010100ULL, 0x00ff000100ffffffULL, 0x00ff000100ffff01ULL,\n"
"    0x00ff000100ff0000ULL, 0x00ff000100ff01ffULL, 0x00ff00010000ffffULL, 0x00ff00010000ff00ULL,\n"
"    0x00ff00010000ff01ULL, 0x00ff000100000000ULL, 0x00ff000100000001ULL, 0x00ff000100000100ULL,\n"
"    0x00ff00010001ff01ULL, 0x00ff000100010000ULL, 0x00ff0001000101ffULL, 0x00ff000101ffff00ULL,\n"
"    0x00ff000101ff0000ULL, 0x00ff000101ff0101ULL, 0x00ff0001010000ffULL, 0x00ff000101000000ULL,\n"
"    0x00ff00010101ff00ULL, 0x00ff0001010100ffULL, 0x00ff000101010001ULL, 0x00ff01ffffff0000ULL,\n"
"    0x00ff01ffff00ff00ULL, 0x00ff01ffff000000ULL, 0x00ff01ffff000101ULL, 0x00ff01ffff010000ULL,\n"
"    0x00ff01ff00ffff01ULL, 0x00ff01ff00ff0100ULL, 0x00ff01ff0000ffffULL, 0x00ff01ff00000000ULL,\n"
"    0x00ff01ff000001ffULL, 0x00ff01ff0001ff00ULL, 0x00ff01ff000100ffULL, 0x00ff01ff00010001ULL,\n"
"    0x00ff01ff00010100ULL, 0x00ff01ff01ff0000ULL, 0x00ff01ff0100ff00ULL, 0x00ff01ff010000ffULL,\n"
"    0x00ff01ff01000001ULL, 0x00ff01ff01000100ULL, 0x00ff01ff01010000ULL, 0x00ff0100ffffff00ULL,\n"
"    0x00ff0100ffff0000ULL, 0x00ff0100ffff0001ULL, 0x00ff0100ffff0101ULL, 0x00ff0100ff00ffffULL,\n"
"    0x00ff0100ff0000ffULL, 0x00ff0100ff000000ULL, 0x00ff0100ff0001ffULL, 0x00ff0100ff01ff00ULL,\n"
"    0x00ff0100ff0100ffULL, 0x00ff0100ff010001ULL, 0x00ff010000ffffffULL, 0x00ff010000ff0000ULL,\n"
"    0x00ff010000ff0101ULL, 0x00ff01000000ff00ULL, 0x00ff01000000ff01ULL, 0x00ff0100000000ffULL,\n"
"    0x00ff010000000000ULL, 0x00ff010000000001ULL, 0x00ff010000000100ULL, 0x00ff01000001ffffULL,\n"
"    0x00ff01000001ff01ULL, 0x00ff010000010000ULL, 0x00ff010000010001ULL, 0x00ff010000010101ULL,\n"
"    0x00ff010001ff0001ULL, 0x00ff010001ff0100ULL, 0x00ff01000100ff01ULL, 0x00ff010001000000ULL,\n"
"    0x00ff010001000001ULL, 0x00ff0100010001ffULL, 0x00ff01000101ff00ULL, 0x00ff0100010100ffULL,\n"
"    0x00ff010001010001ULL, 0x00ff010001010100ULL, 0x00ff0101ff000001ULL, 0x00ff010100ff00ffULL,\n"
"    0x00ff010100ff0001ULL, 0x00ff010100ff0100ULL, 0x00ff010100000000ULL, 0x00ff0101000001ffULL,\n"
"    0x00ff010100000101ULL, 0x00ff0101000100ffULL, 0x00ff010100010100ULL, 0x00ff0101010000ffULL,\n"
"    0x00ff010101010000ULL, 0x0000ffffffffff00ULL, 0x0000ffffffff00ffULL, 0x0000ffffffff0000ULL,\n"
"    0x0000ffffffff0001ULL, 0x0000ffffffff0100ULL, 0x0000ffffff00ff01ULL, 0x0000ffffff000000ULL,\n"
"    0x0000ffffff000101ULL, 0x0000ffffff01ff00ULL, 0x0000ffffff0100ffULL, 0x0000ffffff010100ULL,\n"
"    0x0000ffff00ffffffULL, 0x0000ffff00ff0000ULL, 0x0000ffff00ff01ffULL, 0x0000ffff0000ff00ULL,\n"
"    0x0000ffff000000ffULL, 0x0000ffff00000000ULL, 0x0000ffff00000001ULL, 0x0000ffff00000100ULL,\n"
"    0x0000ffff00010000ULL, 0x0000ffff000101ffULL, 0x0000ffff01ff0001ULL, 0x0000ffff01ff0100ULL,\n"
"    0x0000ffff01000000ULL, 0x0000ffff010001ffULL, 0x0000ffff0101ffffULL, 0x0000ffff0101ff00ULL,\n"
"    0x0000ffff01010001ULL, 0x0000ffff01010100ULL, 0x0000ff00ffff0000ULL, 0x0000ff00ffff01ffULL,\n"
"    0x0000ff00ffff0100ULL, 0x0000ff00ffff0101ULL, 0x0000ff00ff00ff00ULL, 0x0000ff00ff0000ffULL,\n"
"    0x0000ff00ff000000ULL, 0x0000ff00ff000001ULL, 0x0000ff00ff0001ffULL, 0x0000ff00ff000100ULL,\n"
"    0x0000ff00ff01ffffULL, 0x0000ff00ff010000ULL, 0x0000ff00ff010001ULL, 0x0000ff00ff0101ffULL,\n"
"    0x0000ff00ff010101ULL, 0x0000ff0000ffff00ULL, 0x0000ff0000ff00ffULL, 0x0000ff0000ff0000ULL,\n"
"    0x0000ff0000ff0001ULL, 0x0000ff0000ff0100ULL, 0x0000ff000000ffffULL, 0x0000ff000000ff00ULL,\n"
"    0x0000ff000000ff01ULL, 0x0000ff00000000ffULL, 0x0000ff0000000000ULL, 0x0000ff0000000001ULL,\n"
"    0x0000ff00000001ffULL, 0x0000ff0000000100ULL, 0x0000ff0000000101ULL, 0x0000ff000001ff00ULL,\n"
"    0x0000ff00000100ffULL, 0x0000ff0000010000ULL, 0x0000ff0000010001ULL, 0x0000ff0000010100ULL,\n"
"    0x0000ff0001ffff01ULL, 0x0000ff0001ff0000ULL, 0x0000ff000100ff00ULL, 0x0000ff00010000ffULL,\n"
"    0x0000ff0001000000ULL, 0x0000ff0001000001ULL, 0x0000ff0001000100ULL, 0x0000ff000101ffffULL,\n"
"    0x0000ff0001010000ULL, 0x0000ff0001010101ULL, 0x0000ff01ffffff00ULL, 0x0000ff01ffff0001ULL,\n"
"    0x0000ff01ff00ff01ULL, 0x0000ff01ff000000ULL, 0x0000ff01ff000101ULL, 0x0000ff01ff01ff00ULL,\n"
"    0x0000ff01ff0100ffULL, 0x0000ff0100ffff01ULL, 0x0000ff0100ff0000ULL, 0x0000ff0100ff0101ULL,\n"
"    0x0000ff010000ff00ULL, 0x0000ff01000000ffULL, 0x0000ff0100000000ULL, 0x0000ff0100000001ULL,\n"
"    0x0000ff0100000100ULL, 0x0000ff010001ff01ULL, 0x0000ff0100010000ULL, 0x0000ff0101ff0000ULL,\n"
"    0x0000ff010100ffffULL, 0x0000ff010100ff01ULL, 0x0000ff0101000000ULL, 0x0000ff0101000100ULL,\n"
"    0x0000ff0101000101ULL, 0x0000ff01010100ffULL, 0x000000ffffff00ffULL, 0x000000ffffff0000ULL,\n"
"    0x000000ffff00ff00ULL, 0x000000ffff0000ffULL, 0x000000ffff000000ULL, 0x000000ffff000001ULL,\n"
"    0x000000ffff0001ffULL, 0x000000ffff000100ULL, 0x000000ffff01ff00ULL, 0x000000ffff010000ULL,\n"
"    0x000000ffff0101ffULL, 0x000000ffff010101ULL, 0x000000ff00ffff00ULL, 0x000000ff00ff00ffULL,\n"
"    0x000000ff00ff0000ULL, 0x000000ff00ff0001ULL, 0x000000ff00ff0100ULL, 0x000000ff00ff0101ULL,\n"
"    0x000000ff0000ffffULL, 0x000000ff0000ff00ULL, 0x000000ff000000ffULL, 0x000000ff00000000ULL,\n"
"    0x000000ff00000001ULL, 0x000000ff000001ffULL, 0x000000ff00000100ULL, 0x000000ff00000101ULL,\n"
"    0x000000ff0001ff00ULL, 0x000000ff0001ff01ULL, 0x000000ff000100ffULL, 0x000000ff00010000ULL,\n"
"    0x000000ff00010001ULL, 0x000000ff00010100ULL, 0x000000ff01ffffffULL, 0x000000ff01ff01ffULL,\n"
"    0x000000ff01ff0101ULL, 0x000000ff0100ff00ULL, 0x000000ff010000ffULL, 0x000000ff01000000ULL,\n"
"    0x000000ff01000001ULL, 0x000000ff01000100ULL, 0x000000ff0101ff00ULL, 0x000000ff010100ffULL,\n"
"    0x000000ff01010000ULL, 0x000000ff01010101ULL, 0x00000000ffffff00ULL, 0x00000000ffffff01ULL,\n"
"    0x00000000ffff00ffULL, 0x00000000ffff0000ULL, 0x00000000ffff0001ULL, 0x00000000ffff0100ULL,\n"
"    0x00000000ff00ffffULL, 0x00000000ff00ff00ULL, 0x00000000ff00ff01ULL, 0x00000000ff0000ffULL,\n"
"    0x00000000ff000000ULL, 0x00000000ff000001ULL, 0x00000000ff000100ULL, 0x00000000ff000101ULL,\n"
"    0x00000000ff01ff00ULL, 0x00000000ff0100ffULL, 0x00000000ff010000ULL, 0x00000000ff010001ULL,\n"
"    0x00000000ff010100ULL, 0x0000000000ffffffULL, 0x0000000000ffff00ULL, 0x0000000000ffff01ULL,\n"
"    0x0000000000ff00ffULL, 0x0000000000ff0000ULL, 0x0000000000ff0001ULL, 0x0000000000ff01ffULL,\n"
"    0x0000000000ff0100ULL, 0x000000000000ffffULL, 0x000000000000ff00ULL, 0x000000000000ff01ULL,\n"
"    0x00000000000000ffULL, 0x0000000000000000ULL, 0x0000000000000001ULL, 0x00000000000001ffULL,\n"
"    0x0000000000000100ULL, 0x0000000000000101ULL, 0x000000000001ffffULL, 0x000000000001ff00ULL,\n"
"    0x00000000000100ffULL, 0x0000000000010000ULL, 0x0000000000010001ULL, 0x00000000000101ffULL,\n"
"    0x0000000000010100ULL, 0x0000000000010101ULL, 0x0000000001ffff00ULL, 0x0000000001ff00ffULL,\n"
"    0x0000000001ff0000ULL, 0x0000000001ff0100ULL, 0x0000000001ff0101ULL, 0x000000000100ffffULL,\n"
"    0x000000000100ff00ULL, 0x00000000010000ffULL, 0x0000000001000000ULL, 0x0000000001000001ULL,\n"
"    0x00000000010001ffULL, 0x0000000001000100ULL, 0x000000000101ff00ULL, 0x00000000010100ffULL,\n"
"    0x0000000001010000ULL, 0x0000000001010001ULL, 0x0000000001010100ULL, 0x00000001ffffffffULL,\n"
"    0x00000001ffffff00ULL, 0x00000001ffffff01ULL, 0x00000001ffff00ffULL, 0x00000001ffff0001ULL,\n"
"    0x00000001ffff01ffULL, 0x00000001ffff0100ULL, 0x00000001ff00ff00ULL, 0x00000001ff0000ffULL,\n"
"    0x00000001ff000000ULL, 0x00000001ff0001ffULL, 0x00000001ff000100ULL, 0x00000001ff01ffffULL,\n"
"    0x00000001ff01ff00ULL, 0x00000001ff01ff01ULL, 0x00000001ff0100ffULL, 0x00000001ff010000ULL,\n"
"    0x00000001ff010001ULL, 0x00000001ff0101ffULL, 0x00000001ff010100ULL, 0x0000000100ffff00ULL,\n"
"    0x0000000100ff0000ULL, 0x0000000100ff0001ULL, 0x0000000100ff01ffULL, 0x0000000100ff0100ULL,\n"
"    0x0000000100ff0101ULL, 0x000000010000ffffULL, 0x000000010000ff00ULL, 0x000000010000ff01ULL,\n"
"    0x00000001000000ffULL, 0x0000000100000000ULL, 0x0000000100000001ULL, 0x00000001000001ffULL,\n"
"    0x0000000100000100ULL, 0x0000000100000101ULL, 0x000000010001ff00ULL, 0x00000001000100ffULL,\n"
"    0x0000000100010000ULL, 0x0000000100010100ULL, 0x0000000101ffff01ULL, 0x0000000101ff0000ULL,\n"
"    0x0000000101ff0001ULL, 0x0000000101ff01ffULL, 0x0000000101ff0100ULL, 0x0000000101ff0101ULL,\n"
"    0x000000010100ff00ULL, 0x0000000101000000ULL, 0x0000000101000101ULL, 0x000000010101ff01ULL,\n"
"    0x0000000101010000ULL, 0x0000000101010001ULL, 0x00000001010101ffULL, 0x0000000101010100ULL,\n"
"    0x000001ffffff00ffULL, 0x000001ffffff0000ULL, 0x000001ffffff0001ULL, 0x000001ffffff0100ULL,\n"
"    0x000001ffff00ffffULL, 0x000001ffff000000ULL, 0x000001ffff0001ffULL, 0x000001ffff01ff00ULL,\n"
"    0x000001ffff010101ULL, 0x000001ff00ff0000ULL, 0x000001ff00ff01ffULL, 0x000001ff00ff0101ULL,\n"
"    0x000001ff0000ff00ULL, 0x000001ff000000ffULL, 0x000001ff00000000ULL, 0x000001ff00000001ULL,\n"
"    0x000001ff000001ffULL, 0x000001ff00000100ULL, 0x000001ff0001ffffULL, 0x000001ff0001ff01ULL,\n"
"    0x000001ff000100ffULL, 0x000001ff00010000ULL, 0x000001ff01ffff01ULL, 0x000001ff01ff0100ULL,\n"
"    0x000001ff0100ffffULL, 0x000001ff0100ff01ULL, 0x000001ff01000000ULL, 0x000001ff010001ffULL,\n"
"    0x000001ff0101ff00ULL, 0x000001ff01010100ULL, 0x00000100ffffff00ULL, 0x00000100ffffff01ULL,\n"
"    0x00000100ffff0000ULL, 0x00000100ffff0101ULL, 0x00000100ff00ff00ULL, 0x00000100ff0000ffULL,\n"
"    0x00000100ff000000ULL, 0x00000100ff000001ULL, 0x00000100ff000100ULL, 0x00000100ff010000ULL,\n"
"    0x0000010000ffff00ULL, 0x0000010000ff00ffULL, 0x0000010000ff0000ULL, 0x0000010000ff0001ULL,\n"
"    0x0000010000ff0100ULL, 0x000001000000ffffULL, 0x000001000000ff00ULL, 0x000001000000ff01ULL,\n"
"    0x00000100000000ffULL, 0x0000010000000000ULL, 0x0000010000000001ULL, 0x00000100000001ffULL,\n"
"    0x0000010000000100ULL, 0x0000010000000101ULL, 0x000001000001ff00ULL, 0x00000100000100ffULL,\n"
"    0x0000010000010000ULL, 0x0000010000010001ULL, 0x0000010000010100ULL, 0x0000010001ffff00ULL,\n"
"    0x0000010001ff0000ULL, 0x0000010001ff0100ULL, 0x000001000100ff00ULL, 0x00000100010000ffULL,\n"
"    0x0000010001000000ULL, 0x0000010001000001ULL, 0x00000100010001ffULL, 0x0000010001000100ULL,\n"
"    0x0000010001010000ULL, 0x00000101ffff00ffULL, 0x00000101ffff01ffULL, 0x00000101ff000000ULL,\n"
"    0x00000101ff000101ULL, 0x00000101ff01ffffULL, 0x00000101ff010000ULL, 0x00000101ff010001ULL,\n"
"    0x00000101ff010100ULL, 0x0000010100ff0000ULL, 0x0000010100ff01ffULL, 0x0000010100ff0100ULL,\n"
"    0x000001010000ff00ULL, 0x0000010100000000ULL, 0x0000010100000001ULL, 0x00000101000001ffULL,\n"
"    0x0000010100000100ULL, 0x000001010001ff01ULL, 0x0000010100010000ULL, 0x00000101000101ffULL,\n"
"    0x0000010100010101ULL, 0x0000010101ffff00ULL, 0x0000010101ff0101ULL, 0x000001010100ff01ULL,\n"
"    0x0000010101000000ULL, 0x0000010101000001ULL, 0x00000101010001ffULL, 0x0000010101000101ULL,\n"
"    0x000001010101ff00ULL, 0x0001ffffffff0000ULL, 0x0001ffffff0000ffULL, 0x0001ffffff000001ULL,\n"
"    0x0001ffffff000100ULL, 0x0001ffffff010000ULL, 0x0001ffff00ff00ffULL, 0x0001ffff0000ffffULL,\n"
"    0x0001ffff00000000ULL, 0x0001ffff00000001ULL, 0x0001ffff000001ffULL, 0x0001ffff00000101ULL,\n"
"    0x0001ffff0001ff00ULL, 0x0001ffff000100ffULL, 0x0001ffff00010001ULL, 0x0001ffff00010100ULL,\n"
"    0x0001ffff01ffff00ULL, 0x0001ffff01000001ULL, 0x0001ffff01010000ULL, 0x0001ff00ffffff00ULL,\n"
"    0x0001ff00ffff00ffULL, 0x0001ff00ffff0001ULL, 0x0001ff00ffff0100ULL, 0x0001ff00ff00ff01ULL,\n"
"    0x0001ff00ff000000ULL, 0x0001ff00ff01ff00ULL, 0x0001ff00ff01ff01ULL, 0x0001ff00ff010001ULL,\n"
"    0x0001ff00ff010100ULL, 0x0001ff0000ff0000ULL, 0x0001ff0000ff0100ULL, 0x0001ff000000ff00ULL,\n"
"    0x0001ff0000000000ULL, 0x0001ff0000000001ULL, 0x0001ff0000000100ULL, 0x0001ff0000010000ULL,\n"
"    0x0001ff0000010001ULL, 0x0001ff0000010101ULL, 0x0001ff0001ff00ffULL, 0x0001ff0001ff0101ULL,\n"
"    0x0001ff000100ff01ULL, 0x0001ff0001000000ULL, 0x0001ff000101ff00ULL, 0x0001ff0001010001ULL,\n"
"    0x0001ff0001010100ULL, 0x0001ff01ff00ff00ULL, 0x0001ff01ff000001ULL, 0x0001ff01ff000100ULL,\n"
"    0x0001ff0100ffffffULL, 0x0001ff0100ffff00ULL, 0x0001ff0100ff0001ULL, 0x0001ff0100000000ULL,\n"
"    0x0001ff0100000001ULL, 0x0001ff01000001ffULL, 0x0001ff010001ffffULL, 0x0001ff0101ff0000ULL,\n"
"    0x0001ff010100ff00ULL, 0x0001ff0101000001ULL, 0x0001ff0101010000ULL, 0x000100ffff00ff00ULL,\n"
"    0x000100ffff00ff01ULL, 0x000100ffff000000ULL, 0x000100ffff000001ULL, 0x000100ffff000101ULL,\n"
"    0x000100ffff01ff00ULL, 0x000100ffff010001ULL, 0x000100ffff010100ULL, 0x000100ff00ffffffULL,\n"
"    0x000100ff00ffff01ULL, 0x000100ff00ff0000ULL, 0x000100ff00ff01ffULL, 0x000100ff00ff0101ULL,\n"
"    0x000100ff0000ff00ULL, 0x000100ff000000ffULL, 0x000100ff00000000ULL, 0x000100ff00000001ULL,\n"
"    0x000100ff00000100ULL, 0x000100ff00000101ULL, 0x000100ff0001ffffULL, 0x000100ff0001ff01ULL,\n"
"    0x000100ff00010000ULL, 0x000100ff01ff00ffULL, 0x000100ff01ff0000ULL, 0x000100ff01ff0100ULL,\n"
"    0x000100ff0100ffffULL, 0x000100ff0100ff01ULL, 0x000100ff010000ffULL, 0x000100ff01000000ULL,\n"
"    0x000100ff01000001ULL, 0x000100ff010001ffULL, 0x000100ff01000101ULL, 0x000100ff0101ff00ULL,\n"
"    0x000100ff010100ffULL, 0x000100ff01010100ULL, 0x00010000ffff0000ULL, 0x00010000ffff01ffULL,\n"
"    0x00010000ffff0101ULL, 0x00010000ff00ff00ULL, 0x00010000ff000000ULL, 0x00010000ff000001ULL,\n"
"    0x00010000ff000100ULL, 0x0001000000ff00ffULL, 0x0001000000ff0000ULL, 0x0001000000ff0001ULL,\n"
"    0x0001000000ff0100ULL, 0x000100000000ffffULL, 0x000100000000ff00ULL, 0x00010000000000ffULL,\n"
"    0x0001000000000000ULL, 0x0001000000000001ULL, 0x0001000000000100ULL, 0x000100000001ff00ULL,\n"
"    0x00010000000100ffULL, 0x0001000000010000ULL, 0x0001000000010001ULL, 0x0001000000010100ULL,\n"
"    0x0001000001ff0001ULL, 0x0001000001ff0100ULL, 0x0001000001ff0101ULL, 0x000100000100ff00ULL,\n"
"    0x0001000001000000ULL, 0x0001000001000001ULL, 0x0001000001000100ULL, 0x0001000001000101ULL,\n"
"    0x000100000101ff01ULL, 0x0001000001010000ULL, 0x0001000001010001ULL, 0x00010000010101ffULL,\n"
"    0x00010001ffffff01ULL, 0x00010001ffff0100ULL, 0x00010001ff000000ULL, 0x00010001ff01ffffULL,\n"
"    0x00010001ff010001ULL, 0x00010001ff0101ffULL, 0x00010001ff010100ULL, 0x0001000100ffffffULL,\n"
"    0x0001000100ff0000ULL, 0x0001000100ff01ffULL, 0x0001000100ff0101ULL, 0x000100010000ff00ULL,\n"
"    0x00010001000000ffULL, 0x0001000100000000ULL, 0x0001000100000001ULL, 0x00010001000001ffULL,\n"
"    0x0001000100000101ULL, 0x000100010001ffffULL, 0x0001000100010000ULL, 0x00010001000101ffULL,\n"
"    0x0001000101ffffffULL, 0x0001000101ffff01ULL, 0x0001000101ff0000ULL, 0x0001000101ff0101ULL,\n"
"    0x00010001010000ffULL, 0x0001000101000001ULL, 0x00010001010001ffULL, 0x0001000101000100ULL,\n"
"    0x000100010101ffffULL, 0x00010001010100ffULL, 0x0001000101010001ULL, 0x0001000101010101ULL,\n"
"    0x000101ffff000001ULL, 0x000101ffff000100ULL, 0x000101ffff010000ULL, 0x000101ff00ffff00ULL,\n"
"    0x000101ff0000ff01ULL, 0x000101ff00000000ULL, 0x000101ff00000101ULL, 0x000101ff0001ff00ULL,\n"
"    0x000101ff00010100ULL, 0x000101ff01ff0000ULL, 0x000101ff0100ff00ULL, 0x000101ff010001ffULL,\n"
"    0x000101ff01010001ULL, 0x00010100ffffff00ULL, 0x00010100ffff00ffULL, 0x00010100ff00ffffULL,\n"
"    0x00010100ff000000ULL, 0x00010100ff01ff00ULL, 0x00010100ff0100ffULL, 0x00010100ff010001ULL,\n"
"    0x00010100ff010100ULL, 0x0001010000ffffffULL, 0x0001010000ffff00ULL, 0x0001010000ff0000ULL,\n"
"    0x0001010000ff0001ULL, 0x0001010000ff01ffULL, 0x000101000000ff00ULL, 0x00010100000000ffULL,\n"
"    0x0001010000000000ULL, 0x0001010000000001ULL, 0x0001010000000100ULL, 0x000101000001ffffULL,\n"
"    0x0001010000010000ULL, 0x0001010000010101ULL, 0x0001010001ffff01ULL, 0x0001010001ff00ffULL,\n"
"    0x0001010001ff0101ULL, 0x0001010001000000ULL, 0x000101000101ff00ULL, 0x00010100010100ffULL,\n"
"    0x0001010001010000ULL, 0x0001010001010100ULL, 0x00010101ff00ff00ULL, 0x00010101ff000001ULL,\n"
"    0x00010101ff0001ffULL, 0x0001010100ffff00ULL, 0x0001010100ff00ffULL, 0x0001010100ff0100ULL,\n"
"    0x000101010000ffffULL, 0x0001010100000000ULL, 0x00010101000001ffULL, 0x0001010100000101ULL,\n"
"    0x00010101000100ffULL, 0x0001010100010000ULL, 0x0001010100010100ULL, 0x0001010101ff0001ULL,\n"
"    0x00010101010000ffULL, 0x00010101010001ffULL, 0x0001010101000101ULL, 0x0001010101010001ULL,\n"
"    0x01ffffffffffffffULL, 0x01ffffffffffff01ULL, 0x01ffffffffff01ffULL, 0x01ffffffffff0101ULL,\n"
"    0x01ffffffff01ffffULL, 0x01ffffffff01ff01ULL, 0x01ffffffff0101ffULL, 0x01ffffffff010101ULL,\n"
"    0x01ffffff00ff0000ULL, 0x01ffffff0000ffffULL, 0x01ffffff0000ff00ULL, 0x01ffffff000000ffULL,\n"
"    0x01ffffff00000001ULL, 0x01ffffff00000100ULL, 0x01ffffff00010000ULL, 0x01ffffff01ffffffULL,\n"
"    0x01ffffff01ffff01ULL, 0x01ffffff01ff01ffULL, 0x01ffffff01ff0101ULL, 0x01ffffff01000000ULL,\n"
"    0x01ffffff0101ffffULL, 0x01ffffff0101ff01ULL, 0x01ffffff010101ffULL, 0x01ffffff01010101ULL,\n"
"    0x01ffff00ffff0000ULL, 0x01ffff00ff00ff00ULL, 0x01ffff00ff0000ffULL, 0x01ffff00ff000001ULL,\n"
"    0x01ffff00ff000100ULL, 0x01ffff00ff010000ULL, 0x01ffff0000ffff00ULL, 0x01ffff0000ff00ffULL,\n"
"    0x01ffff0000ff0100ULL, 0x01ffff000000ffffULL, 0x01ffff000000ff01ULL, 0x01ffff0000000000ULL,\n"
"    0x01ffff0000000001ULL, 0x01ffff00000001ffULL, 0x01ffff0000000100ULL, 0x01ffff00000100ffULL,\n"
"    0x01ffff0000010001ULL, 0x01ffff0000010100ULL, 0x01ffff0001ff0000ULL, 0x01ffff0001ff0100ULL,\n"
"    0x01ffff00010000ffULL, 0x01ffff0001000001ULL, 0x01ffff0001000100ULL, 0x01ffff0001010000ULL,\n"
"    0x01ffff01ffffffffULL, 0x01ffff01ffffff01ULL, 0x01ffff01ffff01ffULL, 0x01ffff01ffff0101ULL,\n"
"    0x01ffff01ff000000ULL, 0x01ffff01ff01ffffULL, 0x01ffff01ff01ff01ULL, 0x01ffff01ff0101ffULL,\n"
"    0x01ffff01ff010101ULL, 0x01ffff010000ff00ULL, 0x01ffff01000000ffULL, 0x01ffff0100000100ULL,\n"
"    0x01ffff0100010000ULL, 0x01ffff0101ffffffULL, 0x01ffff0101ffff01ULL, 0x01ffff0101ff01ffULL,\n"
"    0x01ffff0101ff0101ULL, 0x01ffff0101000000ULL, 0x01ffff010101ffffULL, 0x01ffff010101ff01ULL,\n"
"    0x01ffff01010101ffULL, 0x01ffff0101010101ULL, 0x01ff00ffff0000ffULL, 0x01ff00ffff000100ULL,\n"
"    0x01ff00ff00ffff00ULL, 0x01ff00ff00ff00ffULL, 0x01ff00ff0000ff00ULL, 0x01ff00ff00000000ULL,\n"
"    0x01ff00ff00000101ULL, 0x01ff00ff0001ff00ULL, 0x01ff00ff000100ffULL, 0x01ff00ff00010100ULL,\n"
"    0x01ff00ff010000ffULL, 0x01ff00ff01000100ULL, 0x01ff0000ffffff00ULL, 0x01ff0000ffff0100ULL,\n"
"    0x01ff0000ff00ff01ULL, 0x01ff0000ff000000ULL, 0x01ff0000ff000101ULL, 0x01ff0000ff010001ULL,\n"
"    0x01ff0000ff010100ULL, 0x01ff000000ffffffULL, 0x01ff000000ffff00ULL, 0x01ff000000ff0000ULL,\n"
"    0x01ff000000ff01ffULL, 0x01ff00000000ff00ULL, 0x01ff0000000000ffULL, 0x01ff000000000000ULL,\n"
"    0x01ff000000000001ULL, 0x01ff000000000100ULL, 0x01ff000000000101ULL, 0x01ff000000010000ULL,\n"
"    0x01ff000000010001ULL, 0x01ff0000000101ffULL, 0x01ff000000010101ULL, 0x01ff000001ffff00ULL,\n"
"    0x01ff000001ff00ffULL, 0x01ff000001ff0001ULL, 0x01ff000001ff0100ULL, 0x01ff00000100ffffULL,\n"
"    0x01ff00000100ff01ULL, 0x01ff000001000000ULL, 0x01ff0000010001ffULL, 0x01ff000001010001ULL,\n"
"    0x01ff0001ff00ff00ULL, 0x01ff0001ff000001ULL, 0x01ff0001ff000100ULL, 0x01ff0001ff010000ULL,\n"
"    0x01ff000100ffff00ULL, 0x01ff000100ff00ffULL, 0x01ff000100ff0100ULL, 0x01ff000100ff0101ULL,\n"
"    0x01ff00010000ffffULL, 0x01ff000100000000ULL, 0x01ff000100000100ULL, 0x01ff000100000101ULL,\n"
"    0x01ff00010001ff00ULL, 0x01ff000100010001ULL, 0x01ff000100010101ULL, 0x01ff000101ff0000ULL,\n"
"    0x01ff00010100ff00ULL, 0x01ff000101000101ULL, 0x01ff0001010100ffULL, 0x01ff01ffffffffffULL,\n"
"    0x01ff01ffffffff01ULL, 0x01ff01ffffff01ffULL, 0x01ff01ffffff0101ULL, 0x01ff01ffff000000ULL,\n"
"    0x01ff01ffff01ffffULL, 0x01ff01ffff01ff01ULL, 0x01ff01ffff0101ffULL, 0x01ff01ffff010101ULL,\n"
"    0x01ff01ff00ffff00ULL, 0x01ff01ff00ff0000ULL, 0x01ff01ff0000ff00ULL, 0x01ff01ff000000ffULL,\n"
"    0x01ff01ff00000100ULL, 0x01ff01ff00010000ULL, 0x01ff01ff00010100ULL, 0x01ff01ff01ffffffULL,\n"
"    0x01ff01ff01ffff01ULL, 0x01ff01ff01ff01ffULL, 0x01ff01ff01ff0101ULL, 0x01ff01ff01000000ULL,\n"
"    0x01ff01ff0101ffffULL, 0x01ff01ff0101ff01ULL, 0x01ff01ff010101ffULL, 0x01ff01ff01010101ULL,\n"
"    0x01ff0100ffff0000ULL, 0x01ff0100ffff0001ULL, 0x01ff0100ff00ff00ULL, 0x01ff0100ff0000ffULL,\n"
"    0x01ff0100ff000001ULL, 0x01ff0100ff010000ULL, 0x01ff010000ffff00ULL, 0x01ff010000ff00ffULL,\n"
"    0x01ff010000ff0001ULL, 0x01ff010000ff0100ULL, 0x01ff01000000ffffULL, 0x01ff01000000ff01ULL,\n"
"    0x01ff010000000000ULL, 0x01ff010000000101ULL, 0x01ff01000001ff00ULL, 0x01ff0100000100ffULL,\n"
"    0x01ff010001ff0000ULL, 0x01ff010001000001ULL, 0x01ff010001000100ULL, 0x01ff010001010000ULL,\n"
"    0x01ff0101ffffffffULL, 0x01ff0101ffffff01ULL, 0x01ff0101ffff01ffULL, 0x01ff0101ffff0101ULL,\n"
"    0x01ff0101ff000000ULL, 0x01ff0101ff01ffffULL, 0x01ff0101ff01ff01ULL, 0x01ff0101ff0101ffULL,\n"
"    0x01ff0101ff010101ULL, 0x01ff010100ff0000ULL, 0x01ff01010000ff00ULL, 0x01ff0101000000ffULL,\n"
"    0x01ff010100000001ULL, 0x01ff010101ffffffULL, 0x01ff010101ffff01ULL, 0x01ff010101ff01ffULL,\n"
"    0x01ff010101ff0101ULL, 0x01ff010101000000ULL, 0x01ff01010101ffffULL, 0x01ff01010101ff01ULL,\n"
"    0x01ff0101010101ffULL, 0x01ff010101010101ULL, 0x0100ffffffff0000ULL, 0x0100ffffff00ff00ULL,\n"
"    0x0100ffffff000001ULL, 0x0100ffffff0001ffULL, 0x0100ffffff000100ULL, 0x0100ffffff010000ULL,\n"
"    0x0100ffff00ffff00ULL, 0x0100ffff00ff0001ULL, 0x0100ffff00ff0100ULL, 0x0100ffff00000000ULL,\n"
"    0x0100ffff000001ffULL, 0x0100ffff00000101ULL, 0x0100ffff00010100ULL, 0x0100ffff00010101ULL,\n"
"    0x0100ffff01ff0000ULL, 0x0100ffff0100ff00ULL, 0x0100ffff010000ffULL, 0x0100ffff01000001ULL,\n"
"    0x0100ffff01000100ULL, 0x0100ffff01010000ULL, 0x0100ff00ffffff00ULL, 0x0100ff00ffff00ffULL,\n"
"    0x0100ff00ffff0001ULL, 0x0100ff00ffff0100ULL, 0x0100ff00ff00ffffULL, 0x0100ff00ff000000ULL,\n"
"    0x0100ff00ff0001ffULL, 0x0100ff00ff000101ULL, 0x0100ff00ff01ff00ULL, 0x0100ff00ff0100ffULL,\n"
"    0x0100ff00ff010001ULL, 0x0100ff00ff010100ULL, 0x0100ff0000ffffffULL, 0x0100ff0000ff0000ULL,\n"
"    0x0100ff000000ffffULL, 0x0100ff000000ff00ULL, 0x0100ff00000000ffULL, 0x0100ff0000000000ULL,\n"
"    0x0100ff0000000001ULL, 0x0100ff0000000100ULL, 0x0100ff000001ff01ULL, 0x0100ff0000010000ULL,\n"
"    0x0100ff0001ff00ffULL, 0x0100ff0001ff0001ULL, 0x0100ff000100ff01ULL, 0x0100ff0001000000ULL,\n"
"    0x0100ff00010001ffULL, 0x0100ff000101ff00ULL, 0x0100ff00010100ffULL, 0x0100ff0001010001ULL,\n"
"    0x0100ff0001010100ULL, 0x0100ff01ffff0000ULL, 0x0100ff01ff00ff00ULL, 0x0100ff01ff0000ffULL,\n"
"    0x0100ff01ff000100ULL, 0x0100ff01ff010000ULL, 0x0100ff0100ff00ffULL, 0x0100ff0100ff0001ULL,\n"
"    0x0100ff0100ff0100ULL, 0x0100ff010000ffffULL, 0x0100ff010000ff01ULL, 0x0100ff0100000000ULL,\n"
"    0x0100ff01000001ffULL, 0x0100ff0100010001ULL, 0x0100ff0100010100ULL, 0x0100ff0101ff0000ULL,\n"
"    0x0100ff01010000ffULL, 0x0100ff0101000001ULL, 0x0100ff0101010100ULL, 0x010000ffffffff00ULL,\n"
"    0x010000ffffff00ffULL, 0x010000ffffff0001ULL, 0x010000ffff00ffffULL, 0x010000ffff000000ULL,\n"
"    0x010000ffff0001ffULL, 0x010000ffff010001ULL, 0x010000ff00ffffffULL, 0x010000ff00ff0101ULL,\n"
"    0x010000ff0000ff00ULL, 0x010000ff000000ffULL, 0x010000ff00000000ULL, 0x010000ff00000001ULL,\n"
"    0x010000ff000001ffULL, 0x010000ff00000100ULL, 0x010000ff0001ffffULL, 0x010000ff0001ff00ULL,\n"
"    0x010000ff0001ff01ULL, 0x010000ff00010000ULL, 0x010000ff01ff00ffULL, 0x010000ff01ff0001ULL,\n"
"    0x010000ff0100ff01ULL, 0x010000ff010000ffULL, 0x010000ff01000000ULL, 0x010000ff010001ffULL,\n"
"    0x010000ff0101ff00ULL, 0x010000ff01010100ULL, 0x01000000ffffffffULL, 0x01000000ffff0000ULL,\n"
"    0x01000000ffff01ffULL, 0x01000000ffff0101ULL, 0x01000000ff00ffffULL, 0x01000000ff00ff00ULL,\n"
"    0x01000000ff0000ffULL, 0x01000000ff000000ULL, 0x01000000ff000001ULL, 0x01000000ff000100ULL,\n"
"    0x01000000ff01ff00ULL, 0x01000000ff010000ULL, 0x01000000ff010100ULL, 0x01000000ff010101ULL,\n"
"    0x0100000000ffff00ULL, 0x0100000000ff00ffULL, 0x0100000000ff0000ULL, 0x0100000000ff0001ULL,\n"
"    0x0100000000ff0100ULL, 0x010000000000ffffULL, 0x010000000000ff00ULL, 0x010000000000ff01ULL,\n"
"    0x01000000000000ffULL, 0x0100000000000000ULL, 0x0100000000000001ULL, 0x01000000000001ffULL,\n"
"    0x0100000000000100ULL, 0x0100000000000101ULL, 0x010000000001ff00ULL, 0x01000000000100ffULL,\n"
"    0x0100000000010000ULL, 0x0100000000010001ULL, 0x0100000000010100ULL, 0x0100000001ffff00ULL,\n"
"    0x0100000001ff0000ULL, 0x0100000001ff01ffULL, 0x010000000100ff00ULL, 0x010000000100ff01ULL,\n"
"    0x01000000010000ffULL, 0x0100000001000000ULL, 0x0100000001000001ULL, 0x0100000001000100ULL,\n"
"    0x0100000001000101ULL, 0x010000000101ffffULL, 0x010000000101ff01ULL, 0x0100000001010000ULL,\n"
"    0x01000000010101ffULL, 0x0100000001010101ULL, 0x01000001ffffff00ULL, 0x01000001ffff00ffULL,\n"
"    0x01000001ff00ffffULL, 0x01000001ff000000ULL, 0x01000001ff000100ULL, 0x01000001ff01ffffULL,\n"
"    0x01000001ff010001ULL, 0x01000001ff010100ULL, 0x0100000100ff0000ULL, 0x0100000100ff01ffULL,\n"
"    0x0100000100ff0100ULL, 0x010000010000ff00ULL, 0x010000010000ff01ULL, 0x0100000100000000ULL,\n"
"    0x0100000100000001ULL, 0x0100000100000100ULL, 0x0100000100010000ULL, 0x01000001000101ffULL,\n"
"    0x0100000101ffff01ULL, 0x0100000101ff00ffULL, 0x0100000101ff0100ULL, 0x0100000101ff0101ULL,\n"
"    0x010000010100ff01ULL, 0x01000001010000ffULL, 0x0100000101000000ULL, 0x01000001010100ffULL,\n"
"    0x0100000101010001ULL, 0x0100000101010100ULL, 0x010001ffffff0000ULL, 0x010001ffff000001ULL,\n"
"    0x010001ffff000100ULL, 0x010001ffff010000ULL, 0x010001ff00ffff00ULL, 0x010001ff00ff0001ULL,\n"
"    0x010001ff0000ffffULL, 0x010001ff0000ff01ULL, 0x010001ff00000000ULL, 0x010001ff00000001ULL,\n"
"    0x010001ff00000101ULL, 0x010001ff000100ffULL, 0x010001ff00010000ULL, 0x010001ff01ff0000ULL,\n"
"    0x010001ff0100ff00ULL, 0x010001ff01000001ULL, 0x010001ff01000100ULL, 0x010001ff01010000ULL,\n"
"    0x01000100ffff00ffULL, 0x01000100ffff0001ULL, 0x01000100ffff0100ULL, 0x01000100ff00ffffULL,\n"
"    0x01000100ff00ff01ULL, 0x01000100ff000000ULL, 0x01000100ff0001ffULL, 0x01000100ff000101ULL,\n"
"    0x01000100ff01ffffULL, 0x01000100ff01ff00ULL, 0x01000100ff0100ffULL, 0x01000100ff010001ULL,\n"
"    0x0100010000ffffffULL, 0x0100010000ffff01ULL, 0x0100010000ff0000ULL, 0x0100010000ff01ffULL,\n"
"    0x0100010000ff0101ULL, 0x010001000000ff00ULL, 0x01000100000000ffULL, 0x0100010000000000ULL,\n"
"    0x0100010000000001ULL, 0x0100010000000100ULL, 0x010001000001ff01ULL, 0x0100010000010000ULL,\n"
"    0x0100010000010001ULL, 0x0100010000010101ULL, 0x0100010001ffff00ULL, 0x0100010001ff00ffULL,\n"
"    0x010001000100ffffULL, 0x010001000100ff01ULL, 0x0100010001000000ULL, 0x0100010001000101ULL,\n"
"    0x010001000101ff00ULL, 0x0100010001010001ULL, 0x01000101ffff0000ULL, 0x01000101ff000000ULL,\n"
"    0x01000101ff010000ULL, 0x0100010100ff00ffULL, 0x0100010100ff0001ULL, 0x0100010100ff0100ULL,\n"
"    0x010001010000ffffULL, 0x0100010100000000ULL, 0x01000101000001ffULL, 0x010001010001ff00ULL,\n"
"    0x0100010101ff0000ULL, 0x010001010100ff00ULL, 0x01000101010000ffULL, 0x0100010101000000ULL,\n"
"    0x0100010101000001ULL, 0x0101ffffffffffffULL, 0x0101ffffffffff01ULL, 0x0101ffffffff01ffULL,\n"
"    0x0101ffffffff0101ULL, 0x0101ffffff000000ULL, 0x0101ffffff01ffffULL, 0x0101ffffff01ff01ULL,\n"
"    0x0101ffffff0101ffULL, 0x0101ffffff010101ULL, 0x0101ffff00ff0000ULL, 0x0101ffff0000ff00ULL,\n"
"    0x0101ffff000000ffULL, 0x0101ffff00000001ULL, 0x0101ffff00000100ULL, 0x0101ffff01ffffffULL,\n"
"    0x0101ffff01ffff01ULL, 0x0101ffff01ff01ffULL, 0x0101ffff01ff0101ULL, 0x0101ffff01000000ULL,\n"
"    0x0101ffff0101ffffULL, 0x0101ffff0101ff01ULL, 0x0101ffff010101ffULL, 0x0101ffff01010101ULL,\n"
"    0x0101ff00ffff0000ULL, 0x0101ff00ffff0100ULL, 0x0101ff00ff00ff00ULL, 0x0101ff00ff0000ffULL,\n"
"    0x0101ff00ff000001ULL, 0x0101ff00ff000100ULL, 0x0101ff00ff000101ULL, 0x0101ff0000ff0001ULL,\n"
"    0x0101ff0000ff0100ULL, 0x0101ff000000ff00ULL, 0x0101ff0000000000ULL, 0x0101ff00000001ffULL,\n"
"    0x0101ff0000000101ULL, 0x0101ff000001ff00ULL, 0x0101ff00000100ffULL, 0x0101ff0001ff0000ULL,\n"
"    0x0101ff000100ffffULL, 0x0101ff000100ff01ULL, 0x0101ff0001000001ULL, 0x0101ff0001000100ULL,\n"
"    0x0101ff01ffffff01ULL, 0x0101ff01ffff01ffULL, 0x0101ff01ffff0101ULL, 0x0101ff01ff00ffffULL,\n"
"    0x0101ff01ff000100ULL, 0x0101ff01ff01ff01ULL, 0x0101ff01ff0101ffULL, 0x0101ff01ff010101ULL,\n"
"    0x0101ff0100ff0000ULL, 0x0101ff010000ff00ULL, 0x0101ff0100000001ULL, 0x0101ff0100000100ULL,\n"
"    0x0101ff0100010000ULL, 0x0101ff0101ffffffULL, 0x0101ff0101ffff01ULL, 0x0101ff0101ff01ffULL,\n"
"    0x0101ff0101ff0101ULL, 0x0101ff0101000000ULL, 0x0101ff010101ffffULL, 0x0101ff010101ff01ULL,\n"
"    0x0101ff01010101ffULL, 0x0101ff0101010101ULL, 0x010100ffff000100ULL, 0x010100ffff010000ULL,\n"
"    0x010100ff00ffff00ULL, 0x010100ff00ff00ffULL, 0x010100ff0000ffffULL, 0x010100ff000000ffULL,\n"
"    0x010100ff00000000ULL, 0x010100ff000001ffULL, 0x010100ff00000101ULL, 0x010100ff0001ff00ULL,\n"
"    0x010100ff00010000ULL, 0x010100ff00010001ULL, 0x010100ff000101ffULL, 0x010100ff00010100ULL,\n"
"    0x010100ff01ff0000ULL, 0x01010000ffff0001ULL, 0x01010000ffff0100ULL, 0x01010000ff00ffffULL,\n"
"    0x01010000ff00ff01ULL, 0x01010000ff000000ULL, 0x01010000ff0001ffULL, 0x01010000ff010001ULL,\n"
"    0x01010000ff010100ULL, 0x0101000000ffff01ULL, 0x0101000000ff0000ULL, 0x010100000000ff00ULL,\n"
"    0x01010000000000ffULL, 0x0101000000000000ULL, 0x0101000000000001ULL, 0x0101000000000100ULL,\n"
"    0x0101000000010000ULL, 0x0101000000010101ULL, 0x0101000001ffff00ULL, 0x0101000001ff00ffULL,\n"
"    0x0101000001ff0000ULL, 0x0101000001ff0001ULL, 0x0101000001ff0100ULL, 0x010100000100ff01ULL,\n"
"    0x0101000001000000ULL, 0x01010000010001ffULL, 0x01010001ffff0000ULL, 0x01010001ff00ff00ULL,\n"
"    0x01010001ff000001ULL, 0x01010001ff000101ULL, 0x01010001ff01ff00ULL, 0x01010001ff010000ULL,\n"
"    0x0101000100ff00ffULL, 0x0101000100ff0001ULL, 0x0101000100ff0101ULL, 0x010100010000ff01ULL,\n"
"    0x0101000100000000ULL, 0x0101000100000001ULL, 0x01010001000001ffULL, 0x010100010001ffffULL,\n"
"    0x010100010001ff01ULL, 0x0101000101ff0001ULL, 0x010100010100ffffULL, 0x0101000101000000ULL,\n"
"    0x0101000101000001ULL, 0x0101000101000100ULL, 0x010100010101ff00ULL, 0x01010001010100ffULL,\n"
"    0x0101000101010001ULL, 0x010101ffffffffffULL, 0x010101ffffffff01ULL, 0x010101ffffff01ffULL,\n"
"    0x010101ffffff0101ULL, 0x010101ffff01ffffULL, 0x010101ffff01ff01ULL, 0x010101ffff0101ffULL,\n"
"    0x010101ffff010101ULL, 0x010101ff0000ff00ULL, 0x010101ff000000ffULL, 0x010101ff00000001ULL,\n"
"    0x010101ff00000100ULL, 0x010101ff01ffffffULL, 0x010101ff01ffff01ULL, 0x010101ff01ff01ffULL,\n"
"    0x010101ff01ff0101ULL, 0x010101ff01000000ULL, 0x010101ff0101ffffULL, 0x010101ff0101ff01ULL,\n"
"    0x010101ff010101ffULL, 0x010101ff01010101ULL, 0x01010100ffff0000ULL, 0x01010100ff0000ffULL,\n"
"    0x01010100ff000100ULL, 0x01010100ff01ff00ULL, 0x01010100ff010000ULL, 0x0101010000ffff00ULL,\n"
"    0x010101000000ffffULL, 0x0101010000000000ULL, 0x0101010000000101ULL, 0x010101000001ff00ULL,\n"
"    0x0101010000010001ULL, 0x0101010000010100ULL, 0x010101000100ffffULL, 0x0101010001000001ULL,\n"
"    0x01010101ffffffffULL, 0x01010101ffffff01ULL, 0x01010101ffff01ffULL, 0x01010101ffff0101ULL,\n"
"    0x01010101ff01ffffULL, 0x01010101ff01ff01ULL, 0x01010101ff0101ffULL, 0x01010101ff010101ULL,\n"
"    0x010101010000ff00ULL, 0x01010101000000ffULL, 0x0101010100000001ULL, 0x0101010101ffffffULL,\n"
"    0x0101010101ffff01ULL, 0x0101010101ff01ffULL, 0x0101010101ff0101ULL, 0x0101010101000000ULL,\n"
"    0x010101010101ffffULL, 0x010101010101ff01ULL, 0x01010101010101ffULL, 0x0101010101010101ULL,\n"
"};\n"

"/* ---- Device LUT for IQ4_NL ---- */\n"
"__device__ static const signed char kvalues_iq4nl_dev[16] = {\n"
"    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113\n"
"};\n"
"\n"
"/* ---- matvec_iq4_nl_f32: IQ4_NL matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq4_nl_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                    int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 32;\n"
"    int row_bytes = nb * 18;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 18;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 16; j++) {\n"
"            float v0 = d * (float)kvalues_iq4nl_dev[qs[j] & 0xf];\n"
"            float v1 = d * (float)kvalues_iq4nl_dev[qs[j] >>  4];\n"
"            partial += v0 * xb[j] + v1 * xb[j + 16];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq4_xs_f32: IQ4_XS matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq4_xs_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                    int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 136;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 136;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned short scales_h = *(const unsigned short *)(bp + 2);\n"
"        const unsigned char *scales_l = bp + 4;\n"
"        const unsigned char *qs = bp + 8;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        for (int ib = 0; ib < 8; ib++) {\n"
"            int ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4);\n"
"            float dl = d * (float)(ls - 32);\n"
"            for (int j = 0; j < 16; j++) {\n"
"                float v0 = dl * (float)kvalues_iq4nl_dev[qs[j] & 0xf];\n"
"                float v1 = dl * (float)kvalues_iq4nl_dev[qs[j] >>  4];\n"
"                partial += v0 * xb[j] + v1 * xb[j + 16];\n"
"            }\n"
"            xb += 32;\n"
"            qs += 16;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq2_xs_f32: IQ2_XS matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq2_xs_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                    int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 74;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 74;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"        const unsigned char *scales = bp + 2 + 64;  /* 8 scale bytes at end */\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;\n"
"            float db1 = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                float dl = (l < 2) ? db0 : db1;\n"
"                unsigned short qval = qs[4*ib32 + l];\n"
"                const unsigned char *grid = (const unsigned char *)&iq2xs_grid_dev[qval & 511];\n"
"                unsigned char signs = ksigns_iq2xs_dev[qval >> 9];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    float w = dl * (float)grid[j] * ((signs & (1 << j)) ? -1.0f : 1.0f);\n"
"                    partial += w * xb[yi++];\n"
"                }\n"
"            }\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq3_xxs_f32: IQ3_XXS matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq3_xxs_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                     int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *scales_and_signs = qs + 64;  /* 32 bytes */\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        #pragma unroll\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            unsigned int aux32 = *(const unsigned int *)(scales_and_signs + 4*ib32);\n"
"            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;\n"
"            #pragma unroll\n"
"            for (int l = 0; l < 4; l++) {\n"
"                unsigned char signs = ksigns_iq2xs_dev[(aux32 >> (7*l)) & 127];\n"
"                const unsigned char *grid1 = (const unsigned char *)&iq3xxs_grid_dev[qs[2*l+0]];\n"
"                const unsigned char *grid2 = (const unsigned char *)&iq3xxs_grid_dev[qs[2*l+1]];\n"
"                #pragma unroll\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    float w0 = db * (float)grid1[j] * ((signs & (1 << j)) ? -1.0f : 1.0f);\n"
"                    float w1 = db * (float)grid2[j] * ((signs & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"                    partial += w0 * xb[yi+j] + w1 * xb[yi+j+4];\n"
"                }\n"
"                yi += 8;\n"
"            }\n"
"            qs += 8;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq2_s_f32: IQ2_S matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq2_s_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 82;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 82;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        /* block_iq2_s layout: d(2) + qs(64)[grid(32)+signs(32)] + qh(8) + scales(8) */\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *qh = bp + 66;\n"
"        const unsigned char *scales = bp + 74;\n"
"        const unsigned char *signs = bp + 34;  /* signs start at qs+32 */\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;\n"
"            float db1 = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                float dl = (l < 2) ? db0 : db1;\n"
"                int grid_idx = qs[l] | ((qh[ib32] << (8-2*l)) & 0x300);\n"
"                const unsigned char *grid = (const unsigned char *)&iq2s_grid_dev[grid_idx];\n"
"                unsigned char s = signs[l];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    float w = dl * (float)grid[j] * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"                    partial += w * xb[yi++];\n"
"                }\n"
"            }\n"
"            qs += 4;\n"
"            signs += 4;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq3_s_f32: IQ3_S matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq3_s_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 110;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        /* block_iq3_s layout: d(2) + qs(64) + qh(8) + signs(32) + scales(4) */\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *qh_base = bp + 2 + 64;\n"
"        const unsigned char *signs_base = bp + 2 + 64 + 8;\n"
"        const unsigned char *scales = bp + 2 + 64 + 8 + 32;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        int qhi = 0, si = 0;\n"
"        for (int ib32 = 0; ib32 < 8; ib32 += 2) {\n"
"            float db1 = d * (float)(1 + 2*(scales[ib32/2] & 0xf));\n"
"            float db2 = d * (float)(1 + 2*(scales[ib32/2] >>  4));\n"
"            for (int l = 0; l < 4; l++) {\n"
"                const unsigned char *grid1 = (const unsigned char *)&iq3s_grid_dev[qs[2*l+0] | ((qh_base[qhi] << (8-2*l)) & 256)];\n"
"                const unsigned char *grid2 = (const unsigned char *)&iq3s_grid_dev[qs[2*l+1] | ((qh_base[qhi] << (7-2*l)) & 256)];\n"
"                unsigned char s = signs_base[si + l];\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    float w0 = db1 * (float)grid1[j] * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"                    float w1 = db1 * (float)grid2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"                    partial += w0 * xb[yi+j] + w1 * xb[yi+j+4];\n"
"                }\n"
"                yi += 8;\n"
"            }\n"
"            qs += 8; si += 4;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                const unsigned char *grid1 = (const unsigned char *)&iq3s_grid_dev[qs[2*l+0] | ((qh_base[qhi+1] << (8-2*l)) & 256)];\n"
"                const unsigned char *grid2 = (const unsigned char *)&iq3s_grid_dev[qs[2*l+1] | ((qh_base[qhi+1] << (7-2*l)) & 256)];\n"
"                unsigned char s = signs_base[si + l];\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    float w0 = db2 * (float)grid1[j] * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"                    float w1 = db2 * (float)grid2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"                    partial += w0 * xb[yi+j] + w1 * xb[yi+j+4];\n"
"                }\n"
"                yi += 8;\n"
"            }\n"
"            qhi += 2; qs += 8; si += 4;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq1_s_f32: IQ1_S matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq1_s_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 50;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    const float IQ1S_DELTA = 0.125f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 50;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        /* block_iq1_s layout: d(2) + qs(32) + qh(16) */\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned short *qh = (const unsigned short *)(bp + 2 + 32);\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        for (int ib = 0; ib < 8; ib++) {\n"
"            float dl = d * (float)(2*((qh[ib] >> 12) & 7) + 1);\n"
"            float delta = (qh[ib] & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                int grid_idx = qs[l] | (((qh[ib] >> (3*l)) & 7) << 8);\n"
"                const signed char *grid = (const signed char *)&iq1s_grid_dev[grid_idx];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    float w = dl * ((float)grid[j] + delta);\n"
"                    partial += w * xb[yi++];\n"
"                }\n"
"            }\n"
"            qs += 4;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws_iq1s[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_iq1s[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_iq1s[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_iq1_m_f32: IQ1_M matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq1_m_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 56;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    const float IQ1S_DELTA = 0.125f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 56;\n"
"        /* block_iq1_m layout: qs(32) + qh(16) + scales(8) */\n"
"        const unsigned char *qs = bp;\n"
"        const unsigned char *qh = bp + 32;\n"
"        const unsigned short *sc = (const unsigned short *)(bp + 32 + 16);\n"
"        /* Reconstruct f16 scale from 4 stolen nibbles */\n"
"        unsigned short scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0u) | ((sc[2] >> 4) & 0x0f00u) | (sc[3] & 0xf000u);\n"
"        float d = half_to_float(*(const half_raw *)&scale_u16);\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        for (int ib = 0; ib < 8; ib++) {\n"
"            float dl1 = d * (float)(2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);\n"
"            float dl2 = d * (float)(2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);\n"
"            unsigned short idx0 = qs[0] | ((unsigned short)(qh[0] << 8) & 0x700u);\n"
"            unsigned short idx1 = qs[1] | ((unsigned short)(qh[0] << 4) & 0x700u);\n"
"            unsigned short idx2 = qs[2] | ((unsigned short)(qh[1] << 8) & 0x700u);\n"
"            unsigned short idx3 = qs[3] | ((unsigned short)(qh[1] << 4) & 0x700u);\n"
"            float delta0 = (qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;\n"
"            float delta1 = (qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;\n"
"            float delta2 = (qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;\n"
"            float delta3 = (qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;\n"
"            const signed char *g0 = (const signed char *)&iq1s_grid_dev[idx0];\n"
"            const signed char *g1 = (const signed char *)&iq1s_grid_dev[idx1];\n"
"            const signed char *g2 = (const signed char *)&iq1s_grid_dev[idx2];\n"
"            const signed char *g3 = (const signed char *)&iq1s_grid_dev[idx3];\n"
"            for (int j = 0; j < 8; j++) partial += dl1 * ((float)g0[j] + delta0) * xb[yi++];\n"
"            for (int j = 0; j < 8; j++) partial += dl1 * ((float)g1[j] + delta1) * xb[yi++];\n"
"            for (int j = 0; j < 8; j++) partial += dl2 * ((float)g2[j] + delta2) * xb[yi++];\n"
"            for (int j = 0; j < 8; j++) partial += dl2 * ((float)g3[j] + delta3) * xb[yi++];\n"
"            qs += 4; qh += 2;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws_iq1m[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_iq1m[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_iq1m[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_tq1_0_f32: TQ1_0 matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_tq1_0_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 54;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    const unsigned char pow3[6] = {1, 3, 9, 27, 81, 243};\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 54;\n"
"        /* block_tq1_0 layout: qs[48] + qh[4] + d(2) */\n"
"        float d = half_to_float(*(const half_raw *)(bp + 52));\n"
"        const unsigned char *qs = bp;\n"
"        const unsigned char *qh_ptr = bp + 48;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        /* Main qs: 48 bytes, first 32 bytes in groups of 32 */\n"
"        for (int j = 0; j < 32; j += 32) {\n"
"            for (int n = 0; n < 5; n++) {\n"
"                for (int m = 0; m < 32; m++) {\n"
"                    unsigned char q = qs[j+m] * pow3[n];\n"
"                    int xi = (((unsigned short)q * 3) >> 8);\n"
"                    partial += (float)(xi - 1) * d * xb[yi++];\n"
"                }\n"
"            }\n"
"        }\n"
"        /* Remaining 16 bytes of qs */\n"
"        for (int j = 32; j < 48; j += 16) {\n"
"            for (int n = 0; n < 5; n++) {\n"
"                for (int m = 0; m < 16; m++) {\n"
"                    unsigned char q = qs[j+m] * pow3[n];\n"
"                    int xi = (((unsigned short)q * 3) >> 8);\n"
"                    partial += (float)(xi - 1) * d * xb[yi++];\n"
"                }\n"
"            }\n"
"        }\n"
"        /* qh: 4 bytes, 4 trits per byte */\n"
"        for (int n = 0; n < 4; n++) {\n"
"            for (int j = 0; j < 4; j++) {\n"
"                unsigned char q = qh_ptr[j] * pow3[n];\n"
"                int xi = (((unsigned short)q * 3) >> 8);\n"
"                partial += (float)(xi - 1) * d * xb[yi++];\n"
"            }\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws_tq1[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_tq1[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_tq1[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_tq2_0_f32: TQ2_0 matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_tq2_0_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 66;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 66;\n"
"        /* block_tq2_0 layout: qs[64] + d(2) */\n"
"        float d = half_to_float(*(const half_raw *)(bp + 64));\n"
"        const unsigned char *qs = bp;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        for (int j = 0; j < 64; j += 32) {\n"
"            for (int l = 0; l < 4; l++) {\n"
"                for (int m = 0; m < 32; m++) {\n"
"                    int q = (qs[j+m] >> (l*2)) & 3;\n"
"                    partial += (float)(q - 1) * d * xb[yi++];\n"
"                }\n"
"            }\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    __shared__ float ws_tq2[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_tq2[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_tq2[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
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
"/* Per-call dequant of a Q4_K weight matrix [N,K] (raw 144-byte blocks per 256\n"
" * elements) to a contiguous BF16 buffer dst[N,K]. Used by the batched prefill\n"
" * path to stage quantized weights into hipBLASLt-callable BF16 once per GEMM.\n"
" * Grid: (n_rows, n_blocks_per_row). Block: 256 threads, one per element. */\n"
"__global__ void dequant_q4_K_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                       int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 144;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 144;\n"
"    float d    = half_to_float(*(const half_raw *)(bp));\n"
"    float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"    const unsigned char *sc = bp + 4;\n"
"    const unsigned char *qs = bp + 16;\n"
"    int tid = threadIdx.x;\n"
"    int sub = tid >> 5;            /* 0..7 sub-block index */\n"
"    int idx = tid & 31;            /* 0..31 element within sub-block */\n"
"    int j   = sub >> 1;            /* 0..3 paired-block index */\n"
"    int is  = sub;\n"
"    unsigned char sv, mv;\n"
"    if (is < 4) { sv = sc[is] & 63; mv = sc[is+4] & 63; }\n"
"    else { sv = (sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4);\n"
"           mv = (sc[is+4] >> 4)  | ((sc[is]     >> 6) << 4); }\n"
"    float dval = d * (float)sv;\n"
"    float mval = dmin * (float)mv;\n"
"    unsigned char q = qs[j * 32 + idx];\n"
"    int nibble = (sub & 1) ? (q >> 4) : (q & 0xF);\n"
"    float val = dval * (float)nibble - mval;\n"
"    int out_pos = j * 64 + ((sub & 1) ? 32 : 0) + idx;\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + out_pos;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"\n"
"/* Per-call dequant of Q5_K (176 B/block) to BF16. Layout: same paired-sub-block\n"
" * scheme as Q4_K plus one extra bit per element from qh[32]. */\n"
"__global__ void dequant_q5_K_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                       int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 176;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 176;\n"
"    float d    = half_to_float(*(const half_raw *)(bp));\n"
"    float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"    const unsigned char *sc = bp + 4;\n"
"    const unsigned char *qh = bp + 16;\n"
"    const unsigned char *qs = bp + 48;\n"
"    int tid = threadIdx.x;\n"
"    int sub = tid >> 5;            /* 0..7 */\n"
"    int idx = tid & 31;            /* 0..31 */\n"
"    int j   = sub >> 1;            /* 0..3 */\n"
"    int is  = sub;\n"
"    unsigned char sv, mv;\n"
"    if (is < 4) { sv = sc[is] & 63; mv = sc[is+4] & 63; }\n"
"    else { sv = (sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4);\n"
"           mv = (sc[is+4] >> 4)  | ((sc[is]     >> 6) << 4); }\n"
"    float dval = d * (float)sv;\n"
"    float mval = dmin * (float)mv;\n"
"    int shift = (sub & 1) ? (2 * j + 1) : (2 * j);\n"
"    int qhbit = (qh[idx] >> shift) & 1;\n"
"    unsigned char q = qs[j * 32 + idx];\n"
"    int nibble = (sub & 1) ? (q >> 4) : (q & 0xF);\n"
"    int weight = nibble | (qhbit << 4);\n"
"    float val = dval * (float)weight - mval;\n"
"    int out_pos = j * 64 + ((sub & 1) ? 32 : 0) + idx;\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + out_pos;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"\n"
"/* Per-call dequant of Q6_K (210 B/block) to BF16. Layout: 256 elements in two\n"
" * halves of 128. Each thread handles one output element. */\n"
"__global__ void dequant_q6_K_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                       int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 210;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 210;\n"
"    float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"    int tid   = threadIdx.x;\n"
"    int half  = tid >> 7;          /* 0 or 1 */\n"
"    int pos   = tid & 127;          /* 0..127 */\n"
"    int q_idx = pos >> 5;           /* 0..3 (q1/q2/q3/q4) */\n"
"    int l     = pos & 31;           /* 0..31 */\n"
"    int is    = l >> 4;             /* 0 or 1 */\n"
"    int use_low_nibble = (q_idx < 2);\n"
"    int ql_off = (q_idx & 1) ? (l + 32) : l;\n"
"    int ql_idx = half * 64 + ql_off;\n"
"    int qh_idx = 128 + half * 32 + l;\n"
"    int sc_idx = 192 + half * 8 + (is + q_idx * 2);\n"
"    int qh_bits = (bp[qh_idx] >> (q_idx * 2)) & 3;\n"
"    int ql_nibble = use_low_nibble ? (bp[ql_idx] & 0xF) : (bp[ql_idx] >> 4);\n"
"    int q = (int)(ql_nibble | (qh_bits << 4)) - 32;\n"
"    int scale = (int)((signed char)bp[sc_idx]);\n"
"    float val = d * (float)scale * (float)q;\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + tid;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"\n"
"/* Per-call dequant of IQ3_XXS (98 B/block, 256 elems) to BF16. Uses the\n"
" * iq3xxs_grid_dev codebook + ksigns_iq2xs_dev sign table that were emitted\n"
" * alongside matvec_iq3_xxs_f32. One thread per output element. */\n"
"__global__ void dequant_iq3_xxs_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                         int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 98;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 98;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    const unsigned char *qs = bp + 2;\n"
"    const unsigned char *scales_and_signs = bp + 66;\n"
"    int tid = threadIdx.x;\n"
"    int ib32 = tid >> 5;          /* 0..7 (which sub-block) */\n"
"    int elem_in_sub = tid & 31;   /* 0..31 */\n"
"    int pair_idx = elem_in_sub >> 3;     /* 0..3 (4 pair-iters per sub) */\n"
"    int elem_in_pair = elem_in_sub & 7;  /* 0..7 (8 elems per pair) */\n"
"    unsigned int aux32 = *(const unsigned int *)(scales_and_signs + 4 * ib32);\n"
"    float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;\n"
"    unsigned char signs = ksigns_iq2xs_dev[(aux32 >> (7 * pair_idx)) & 127];\n"
"    int use_grid2 = (elem_in_pair >= 4);\n"
"    int j = elem_in_pair & 3;\n"
"    unsigned char qs_idx = qs[ib32 * 8 + pair_idx * 2 + use_grid2];\n"
"    const unsigned char *grid = (const unsigned char *)&iq3xxs_grid_dev[qs_idx];\n"
"    int sign_bit = use_grid2 ? (j + 4) : j;\n"
"    float val = db * (float)grid[j] * ((signs & (1 << sign_bit)) ? -1.0f : 1.0f);\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + tid;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"\n"
"/* Per-call dequant of IQ4_XS (136 B/block, 256 elems) to BF16. Uses the\n"
" * kvalues_iq4nl_dev 16-entry lookup. One thread per output element. */\n"
"__global__ void dequant_iq4_xs_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                         int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 136;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 136;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    unsigned short scales_h = *(const unsigned short *)(bp + 2);\n"
"    const unsigned char *scales_l = bp + 4;\n"
"    const unsigned char *qs = bp + 8;\n"
"    int tid = threadIdx.x;\n"
"    int ib   = tid >> 5;            /* 0..7 sub-block */\n"
"    int elem = tid & 31;            /* 0..31 */\n"
"    int use_high_nibble = elem >> 4;\n"
"    int j    = elem & 15;\n"
"    int ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xf) |\n"
"             (((scales_h >> (2 * ib)) & 3) << 4);\n"
"    float dl = d * (float)(ls - 32);\n"
"    const unsigned char *qs_blk = qs + ib * 16;\n"
"    int q = use_high_nibble ? (qs_blk[j] >> 4) : (qs_blk[j] & 0xf);\n"
"    float val = dl * (float)kvalues_iq4nl_dev[q];\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + tid;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"\n"
"/* Per-call dequant of IQ2_XS (74 B/block) to BF16. Codebook + per-pair sign byte. */\n"
"__global__ void dequant_iq2_xs_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                         int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 74;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 74;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"    const unsigned char *scales = bp + 66;\n"
"    int tid  = threadIdx.x;\n"
"    int ib32 = tid >> 5;            /* 0..7 sub-block */\n"
"    int elem = tid & 31;            /* 0..31 */\n"
"    int l    = elem >> 3;           /* 0..3 (4 pair-iters) */\n"
"    int j    = elem & 7;            /* 0..7 in the pair */\n"
"    float dl = (l < 2)\n"
"        ? d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f\n"
"        : d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;\n"
"    unsigned short qval = qs[4 * ib32 + l];\n"
"    const unsigned char *grid = (const unsigned char *)&iq2xs_grid_dev[qval & 511];\n"
"    unsigned char signs = ksigns_iq2xs_dev[qval >> 9];\n"
"    float val = dl * (float)grid[j] * ((signs & (1 << j)) ? -1.0f : 1.0f);\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + tid;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"\n"
"/* Per-call dequant of IQ2_S (82 B/block) to BF16. Codebook + qh + per-pair signs. */\n"
"__global__ void dequant_iq2_s_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                        int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 82;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 82;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    const unsigned char *qs_base    = bp + 2;\n"
"    const unsigned char *signs_base = bp + 34;\n"
"    const unsigned char *qh         = bp + 66;\n"
"    const unsigned char *scales     = bp + 74;\n"
"    int tid  = threadIdx.x;\n"
"    int ib32 = tid >> 5;\n"
"    int elem = tid & 31;\n"
"    int l    = elem >> 3;\n"
"    int j    = elem & 7;\n"
"    float dl = (l < 2)\n"
"        ? d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f\n"
"        : d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;\n"
"    const unsigned char *qs    = qs_base    + 4 * ib32;\n"
"    const unsigned char *signs = signs_base + 4 * ib32;\n"
"    int grid_idx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);\n"
"    const unsigned char *grid = (const unsigned char *)&iq2s_grid_dev[grid_idx];\n"
"    unsigned char s = signs[l];\n"
"    float val = dl * (float)grid[j] * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + tid;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"\n"
"/* Per-call dequant of IQ3_S (110 B/block) to BF16. Each pair of sub-blocks\n"
" * shares a scale byte; sub-block 0 uses the low nibble, sub-block 1 the high.\n"
" * Each 8-element pair-iter draws from two consecutive grid entries, with the\n"
" * high bit coming from qh shifted by (8 - 2*l) for the first grid and\n"
" * (7 - 2*l) for the second. */\n"
"__global__ void dequant_iq3_s_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                        int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 110;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 110;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    const unsigned char *qs_base    = bp + 2;\n"
"    const unsigned char *qh_base    = bp + 66;\n"
"    const unsigned char *signs_base = bp + 74;\n"
"    const unsigned char *scales     = bp + 106;\n"
"    int tid     = threadIdx.x;\n"
"    int pair    = tid >> 6;          /* 0..3 (4 ib32-pair iters) */\n"
"    int inner   = tid & 63;\n"
"    int which   = inner >> 5;        /* 0=db1 (low scale), 1=db2 (high scale) */\n"
"    int sub_i   = inner & 31;\n"
"    int l       = sub_i >> 3;        /* 0..3 */\n"
"    int j       = sub_i & 7;         /* 0..7; j<4: grid1, j>=4: grid2 */\n"
"    int qhi     = 2 * pair + which;\n"
"    int scale_b = scales[pair];\n"
"    float db = (which == 0)\n"
"        ? d * (float)(1 + 2 * (scale_b & 0xf))\n"
"        : d * (float)(1 + 2 * (scale_b >> 4));\n"
"    const unsigned char *qs    = qs_base    + 16 * pair + 8 * which;\n"
"    const unsigned char *signs = signs_base +  8 * pair + 4 * which;\n"
"    int use_grid2 = (j >= 4);\n"
"    int j4    = j & 3;\n"
"    int qs_ix = 2 * l + (use_grid2 ? 1 : 0);\n"
"    int shift = use_grid2 ? (7 - 2 * l) : (8 - 2 * l);\n"
"    int grid_idx = qs[qs_ix] | ((qh_base[qhi] << shift) & 256);\n"
"    const unsigned char *grid = (const unsigned char *)&iq3s_grid_dev[grid_idx];\n"
"    unsigned char s = signs[l];\n"
"    float val = db * (float)grid[j4] * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + tid;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
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
"/* ---- head_dim=256 variant (qwen3.5/3.6 hybrid attn layers) ---- */\n"
"/* Same structure as flash_attn_wmma_f16_4w_causal, with HD_MAX doubled to\n"
" * 256 (K_NB=16). Per-thread register pressure roughly doubles for q_reg\n"
" * and O_acc (~192 VGPR/thread); occupancy drops to 1 wave/SIMD on RDNA4\n"
" * but the kernel still functions. LDS footprint: 16x256 K0+K1+V0+V1 +\n"
" * 4x16x16 smP = ~34 KB per block (fits 64 KB LDS budget).\n"
" *\n"
" * Cooperative K/V load partitions 16 rows x 256 cols across 128 threads:\n"
" * 8 threads/row x 32 cols/thread = 4096 elements. */\n"
"#define FA_LLM_BQ 64\n"
"#define FA_LLM_BKV 16\n"
"#define FA_LLM_HD_MAX 256\n"
"#define FA_LLM_K_NB (FA_LLM_HD_MAX / 16)\n"
"__global__ void flash_attn_wmma_f16_4w_causal_hd256(\n"
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
"    int max_q_global = position_start + q0 + FA_LLM_BQ - 1;\n"
"    int max_kv = max_q_global + 1;\n"
"    if (max_kv > kv_len) max_kv = kv_len;\n"
"    int n_kv = (max_kv + FA_LLM_BKV - 1) / FA_LLM_BKV;\n"
"    /* 128 threads -> 16 KV rows x 8 col-blocks (32 cols each, total HD_MAX=256). */\n"
"    int ld_row  = tid >> 3;\n"
"    int ld_col0 = (tid & 7) * 32;\n"
"    {\n"
"        int kv = ld_row;\n"
"        size_t k_base = ((size_t)kv_h * kv_len + kv) * head_dim;\n"
"        bool kv_ok = (kv < max_kv);\n"
"        for (int j = 0; j < 32; j++) {\n"
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
"            for (int j = 0; j < 32; j++) {\n"
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
"        bool col_valid = (kv0 + idx) < max_kv;\n"
"        for (int i = 0; i < 8; i++) score[i] = col_valid ? score[i] : -1e30f;\n"
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
    hipFunction_t fn_rmsnorm_batch_f32;
    hipFunction_t fn_matvec_f16_f32;
    hipFunction_t fn_qknorm_f32;
    hipFunction_t fn_qknorm_batch_f32;
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
    hipFunction_t fn_matvec_q4_K_mw_f32;
    hipFunction_t fn_matvec_q5_K_f32;
    hipFunction_t fn_matvec_q5_K_mw_f32;
    hipFunction_t fn_matvec_q6_K_f32;
    hipFunction_t fn_embed_q2_K;
    /* SSM kernels */
    hipFunction_t fn_softplus_mul_f32;
    hipFunction_t fn_softplus_mul_batch_f32;
    hipFunction_t fn_sigmoid_inplace_f32;
    hipFunction_t fn_conv1d_depthwise_silu_f32;
    hipFunction_t fn_l2_norm_heads_f32;
    hipFunction_t fn_repeat_tile_f32;
    hipFunction_t fn_deltanet_step_f32;
    hipFunction_t fn_deltanet_step_batch_f32;
    hipFunction_t fn_l2_norm_heads_batch_f32;
    hipFunction_t fn_repeat_tile_batch_f32;
    hipFunction_t fn_gated_rmsnorm_silu_batch_f32;
    hipFunction_t fn_conv1d_depthwise_silu_batch_f32;
    hipFunction_t fn_gated_rmsnorm_silu_f32;
    hipFunction_t fn_sigmoid_mul_f32;
    hipFunction_t fn_deinterleave_qgate_f32;
    hipFunction_t fn_deinterleave_qgate_batch_f32;
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
    /* K-quant batched-prefill dequant kernels (write BF16 to a staging buffer) */
    hipFunction_t fn_dequant_q4_K_to_bf16;
    hipFunction_t fn_dequant_q5_K_to_bf16;
    hipFunction_t fn_dequant_q6_K_to_bf16;
    hipFunction_t fn_dequant_iq3_xxs_to_bf16;
    hipFunction_t fn_dequant_iq4_xs_to_bf16;
    hipFunction_t fn_dequant_iq2_xs_to_bf16;
    hipFunction_t fn_dequant_iq2_s_to_bf16;
    hipFunction_t fn_dequant_iq3_s_to_bf16;

    /* Phase 3 flash-attention helpers */
    hipFunction_t fn_pack_f16_from_f32;
    hipFunction_t fn_pack_kv_cache_f16;
    hipFunction_t fn_rope_neox_batch_f32;
    hipFunction_t fn_rope_mrope_batch_f32;
    hipFunction_t fn_kv_cache_store_batch;
    hipFunction_t fn_flash_attn_wmma_f16_4w_causal;
    hipFunction_t fn_flash_attn_wmma_f16_4w_causal_hd256;

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

    /* Shared staging buffer for per-call dequant of K-quant weights to BF16
     * inside the batched prefill path. Sized to the largest single weight
     * tensor (typically ffn_down: n_embd × n_ff). */
    void *d_wbuf_bf16;
    size_t d_wbuf_bf16_bytes;

    /* Hybrid gated-attention batched buffers (qwen35-style). Allocated only
     * when r->is_hybrid; carry the Q projection's 2*q_dim-wide output before
     * deinterleave, plus the per-row gate vector consumed by sigmoid_mul. */
    void *d_qfull_batch;        /* [BMAX, 2*q_dim_max] F32 */
    void *d_attn_gate_batch;    /* [BMAX, q_dim] F32 */

    /* SSM batched intermediate buffers (B2 Phase 3). Allocated when r->is_hybrid.
     * The 5 SSM projections fan out from d_xnorm_batch_bf16 to these via
     * hipBLASLt; sequential ops (conv1d, deltanet_step) consume them per-row. */
    void *d_ssm_qkv_batch;      /* [BMAX, qkv_dim] F32 */
    void *d_ssm_z_batch;        /* [BMAX, d_inner]  F32 (=dt_rank*d_state) */
    void *d_ssm_alpha_batch;    /* [BMAX, dt_rank]  F32 */
    void *d_ssm_beta_batch;     /* [BMAX, dt_rank]  F32 */
    void *d_ssm_conv_out_batch; /* [BMAX, qkv_dim]  F32 (post-conv1d) */
    void *d_ssm_Q_exp_batch;    /* [BMAX, dt_rank*d_state] F32 */
    void *d_ssm_K_exp_batch;    /* [BMAX, dt_rank*d_state] F32 */
    void *d_ssm_out_batch;      /* [BMAX, d_inner]  F32 */

    /* === Phase 3: flash-attention scratch === */
    int   fa_path_ok;           /* 1 if FA prefill path is enabled */
    void *d_q_batch_f16;        /* [BMAX, n_heads*head_dim] F16 */
    void *d_kv_pack_K_f16;      /* [n_kv_heads, max_seq_len, head_dim] F16 */
    void *d_kv_pack_V_f16;      /* [n_kv_heads, max_seq_len, head_dim] F16 */

    /* === Phase 4: WMMA decode matvec === */
    int decode_wmma;            /* 1 = use WMMA matvec for F16 weights at decode */
    int quant_matvec_opt;       /* 1 = route experimental quant matvec variants */
    int q5_k_mw;                /* 1 = use one-warp-per-row Q5_K matvec */

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
    GET_FUNC(rmsnorm_batch_f32);
    GET_FUNC(matvec_f16_f32);
    GET_FUNC(qknorm_f32);
    GET_FUNC(qknorm_batch_f32);
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
    GET_FUNC(matvec_q4_K_mw_f32);
    GET_FUNC(matvec_q5_K_f32);
    GET_FUNC(matvec_q5_K_mw_f32);
    GET_FUNC(matvec_q6_K_f32);
    GET_FUNC(embed_q2_K);
    /* SSM kernels */
    GET_FUNC(softplus_mul_f32);
    GET_FUNC(softplus_mul_batch_f32);
    GET_FUNC(sigmoid_inplace_f32);
    GET_FUNC(conv1d_depthwise_silu_f32);
    GET_FUNC(l2_norm_heads_f32);
    GET_FUNC(repeat_tile_f32);
    GET_FUNC(deltanet_step_f32);
    GET_FUNC(deltanet_step_batch_f32);
    GET_FUNC(l2_norm_heads_batch_f32);
    GET_FUNC(repeat_tile_batch_f32);
    GET_FUNC(gated_rmsnorm_silu_batch_f32);
    GET_FUNC(conv1d_depthwise_silu_batch_f32);
    GET_FUNC(gated_rmsnorm_silu_f32);
    GET_FUNC(sigmoid_mul_f32);
    GET_FUNC(deinterleave_qgate_f32);
    GET_FUNC(deinterleave_qgate_batch_f32);
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
    GET_FUNC(dequant_q4_K_to_bf16);
    GET_FUNC(dequant_q5_K_to_bf16);
    GET_FUNC(dequant_q6_K_to_bf16);
    GET_FUNC(dequant_iq3_xxs_to_bf16);
    GET_FUNC(dequant_iq4_xs_to_bf16);
    GET_FUNC(dequant_iq2_xs_to_bf16);
    GET_FUNC(dequant_iq2_s_to_bf16);
    GET_FUNC(dequant_iq3_s_to_bf16);

    GET_FUNC(pack_f16_from_f32);
    GET_FUNC(pack_kv_cache_f16);
    GET_FUNC(rope_neox_batch_f32);
    GET_FUNC(rope_mrope_batch_f32);
    GET_FUNC(kv_cache_store_batch);
    GET_FUNC(flash_attn_wmma_f16_4w_causal);
    GET_FUNC(flash_attn_wmma_f16_4w_causal_hd256);

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
    r->quant_matvec_opt = 0;
    {
        const char *env_qmv = getenv("LLM_QUANT_MATVEC_OPT");
        if (env_qmv) r->quant_matvec_opt = atoi(env_qmv) != 0;
    }
    r->q5_k_mw = 1;
    {
        const char *env_q5 = getenv("LLM_Q5_K_MW");
        if (env_q5) r->q5_k_mw = atoi(env_q5) != 0;
    }

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
    } else if ((t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q4_1 ||
                t->type == GGML_TYPE_Q5_0 || t->type == GGML_TYPE_Q5_1) &&
               !getenv("HIP_LLM_LEGACY_CPU_DEQUANT")) {
        /* Legacy quants with real GPU kernels — upload raw (native block stride).
         * Set HIP_LLM_LEGACY_CPU_DEQUANT=1 to force the old CPU-dequant path. */
        return upload_kquant_raw(d_ptr, t);
    } else if ((t->type == GGML_TYPE_IQ2_XXS || t->type == GGML_TYPE_IQ2_XS ||
                t->type == GGML_TYPE_IQ2_S   || t->type == GGML_TYPE_IQ3_XXS ||
                t->type == GGML_TYPE_IQ3_S   || t->type == GGML_TYPE_IQ4_NL ||
                t->type == GGML_TYPE_IQ4_XS  || t->type == GGML_TYPE_IQ1_S ||
                t->type == GGML_TYPE_IQ1_M   || t->type == GGML_TYPE_TQ1_0 ||
                t->type == GGML_TYPE_TQ2_0) &&
               !getenv("HIP_LLM_LEGACY_CPU_DEQUANT")) {
        /* IQ/TQ types with real GPU kernels — upload raw (native block stride).
         * Set HIP_LLM_LEGACY_CPU_DEQUANT=1 to force the old CPU-dequant path. */
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
               embd.type == GGML_TYPE_Q5_K || embd.type == GGML_TYPE_Q6_K ||
               embd.type == GGML_TYPE_Q4_0 || embd.type == GGML_TYPE_Q4_1 ||
               embd.type == GGML_TYPE_Q5_0 || embd.type == GGML_TYPE_Q5_1 ||
               embd.type == GGML_TYPE_IQ2_XXS || embd.type == GGML_TYPE_IQ2_XS ||
               embd.type == GGML_TYPE_IQ2_S   || embd.type == GGML_TYPE_IQ3_XXS ||
               embd.type == GGML_TYPE_IQ3_S   || embd.type == GGML_TYPE_IQ4_NL ||
               embd.type == GGML_TYPE_IQ4_XS  || embd.type == GGML_TYPE_IQ1_S ||
               embd.type == GGML_TYPE_IQ1_M   || embd.type == GGML_TYPE_TQ1_0 ||
               embd.type == GGML_TYPE_TQ2_0) {
        /* Dequant to F16 for embedding lookup (no dedicated embed kernel for these) */
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
        /* Default BMAX:
         *   dense models: 4096 (covers prefill_len up to 4096 in one batch)
         *   hybrid SSM models: 1024 — SSM buffers scale with BMAX × qkv_dim
         *     (160-200 MB each), so default 4096 risks OOM on 27B-class models.
         * LLM_BMAX env still overrides. */
        int batch_max = r->is_hybrid ? 1024 : 4096;
        const char *env_bm = getenv("LLM_BMAX");
        if (env_bm) batch_max = atoi(env_bm);
        if (batch_max < 1) batch_max = 1;
        if (batch_max > 8192) batch_max = 8192;
        r->batch_max = batch_max;

        int gemm_thresh = 8;
        const char *env_th = getenv("LLM_GEMM_M_THRESHOLD");
        if (env_th) gemm_thresh = atoi(env_th);
        if (gemm_thresh < 1) gemm_thresh = 1;
        r->gemm_m_threshold = gemm_thresh;

        const char *env_dis = getenv("LLM_BATCH_DISABLE");
        int disabled = (env_dis && atoi(env_dis) != 0);

        /* Eligibility: any model where every standard-attn/FFN layer has a
         * batchable weight type. Hybrid models are allowed in B2 Phase 1 —
         * SSM layers and qwen35 gated-attn layers go per-row via
         * forward_one_layer(); only standard attn+FFN layers use the batched
         * code below. MoE is still per-token (separate sub-project). */
        int eligible = !disabled && !r->is_moe;

        /* Per-layer type batchability is decided at dispatch time (not here).
         * The gate only rejects MoE; everything else may have batchable layers,
         * non-batchable layers, or SSM layers — and the layer loop in
         * forward_block_batched_dense falls back to per-row for any layer
         * whose weight type lacks a dequant kernel. (For non-hybrid dense
         * models with all unsupported quants, the layer loop will per-row
         * every layer, which is functionally equivalent to the old fallback.) */
        if (eligible) {
            for (int l = 0; l < r->n_layers; l++) {
                hip_layer *cl = &r->layers[l];
                if (cl->is_moe) { eligible = 0; break; }
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

            /* Allocate BF16 weight copies + populate via on-GPU F16->BF16 kernel.
             * Only for F16-typed weights; Q4_K weights are dequantized per-call
             * into r->d_wbuf_bf16 staging buffer (allocated below). */
            size_t max_w_elems = 0;
            #define CONV_IF_F16(field, type, rows, cols)                                 \
                do {                                                                     \
                    size_t _n = (size_t)cl->rows * cl->cols;                             \
                    if (_n > max_w_elems) max_w_elems = _n;                              \
                    if (cl->type == GGML_TYPE_F16) {                                     \
                        CHECK_HIP(hipMalloc(&cl->field, _n * 2));                        \
                        launch_convert_f16_to_bf16_loadtime(r, cl->field,                \
                            cl->field##_src, (int)_n);                                   \
                    } else {                                                             \
                        cl->field = NULL;                                                \
                    }                                                                    \
                } while (0)
            /* The macro above wants a "src" tied to the bf16 field name; expand
             * the 7 cases by hand to keep things explicit. */
            for (int l = 0; l < r->n_layers; l++) {
                hip_layer *cl = &r->layers[l];
                size_t n_q  = (size_t)cl->attn_q_rows * cl->attn_q_cols;
                size_t n_k  = (size_t)cl->attn_k_rows * cl->attn_k_cols;
                size_t n_v  = (size_t)cl->attn_v_rows * cl->attn_v_cols;
                size_t n_o  = (size_t)cl->attn_output_rows * cl->attn_output_cols;
                size_t n_g  = (size_t)cl->ffn_gate_rows * cl->ffn_gate_cols;
                size_t n_u  = (size_t)cl->ffn_up_rows   * cl->ffn_up_cols;
                size_t n_d  = (size_t)cl->ffn_down_rows * cl->ffn_down_cols;
                if (n_q > max_w_elems) max_w_elems = n_q;
                if (n_k > max_w_elems) max_w_elems = n_k;
                if (n_v > max_w_elems) max_w_elems = n_v;
                if (n_o > max_w_elems) max_w_elems = n_o;
                if (n_g > max_w_elems) max_w_elems = n_g;
                if (n_u > max_w_elems) max_w_elems = n_u;
                if (n_d > max_w_elems) max_w_elems = n_d;
                #define DO_W(field, src, type, n) do {                                   \
                    if (cl->type == GGML_TYPE_F16) {                                     \
                        CHECK_HIP(hipMalloc(&cl->field, (n) * 2));                       \
                        launch_convert_f16_to_bf16_loadtime(r, cl->field, cl->src, (int)(n)); \
                    } else { cl->field = NULL; }                                         \
                } while (0)
                DO_W(attn_q_w_bf16,      attn_q_w,      attn_q_type,      n_q);
                DO_W(attn_k_w_bf16,      attn_k_w,      attn_k_type,      n_k);
                DO_W(attn_v_w_bf16,      attn_v_w,      attn_v_type,      n_v);
                DO_W(attn_output_w_bf16, attn_output_w, attn_output_type, n_o);
                DO_W(ffn_gate_w_bf16,    ffn_gate_w,    ffn_gate_type,    n_g);
                DO_W(ffn_up_w_bf16,      ffn_up_w,      ffn_up_type,      n_u);
                DO_W(ffn_down_w_bf16,    ffn_down_w,    ffn_down_type,    n_d);
                #undef DO_W
            }
            #undef CONV_IF_F16
            hipDeviceSynchronize();

            /* Per-call dequant staging buffer (sized to the largest weight). */
            r->d_wbuf_bf16_bytes = max_w_elems * 2;
            CHECK_HIP(hipMalloc(&r->d_wbuf_bf16, r->d_wbuf_bf16_bytes));
            if (r->verbose >= 1) {
                fprintf(stderr, "hip_llm: K-quant dequant staging buffer = %.1f MB\n",
                        (double)r->d_wbuf_bf16_bytes / (1024.0 * 1024.0));
            }

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

            /* Hybrid gated-attn extra buffers: Q projection emits 2*q_dim per row. */
            r->d_qfull_batch     = NULL;
            r->d_attn_gate_batch = NULL;
            r->d_ssm_qkv_batch     = NULL;
            r->d_ssm_z_batch       = NULL;
            r->d_ssm_alpha_batch   = NULL;
            r->d_ssm_beta_batch    = NULL;
            r->d_ssm_conv_out_batch= NULL;
            r->d_ssm_Q_exp_batch   = NULL;
            r->d_ssm_K_exp_batch   = NULL;
            r->d_ssm_out_batch     = NULL;
            if (r->is_hybrid) {
                /* Scan layers for the widest attn_q output. For pure-SSM layers
                 * attn_q_rows is 0; for gated-attn layers it's typically 2*q_dim. */
                int max_q_rows = q_dim;
                for (int l = 0; l < r->n_layers; l++) {
                    int r_ = r->layers[l].attn_q_rows;
                    if (r_ > max_q_rows) max_q_rows = r_;
                }
                CHECK_HIP(hipMalloc(&r->d_qfull_batch,     bm * (size_t)max_q_rows * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_attn_gate_batch, bm * (size_t)q_dim      * sizeof(float)));

                /* SSM Phase-3 buffers */
                int qkv_dim  = r->ssm_qkv_dim;
                int d_inner  = r->ssm_d_inner;
                int dt_rank  = r->ssm_dt_rank;
                int d_state  = r->ssm_d_state;
                CHECK_HIP(hipMalloc(&r->d_ssm_qkv_batch,      bm * (size_t)qkv_dim          * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_ssm_z_batch,        bm * (size_t)d_inner          * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_ssm_alpha_batch,    bm * (size_t)dt_rank          * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_ssm_beta_batch,     bm * (size_t)dt_rank          * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_ssm_conv_out_batch, bm * (size_t)qkv_dim          * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_ssm_Q_exp_batch,    bm * (size_t)dt_rank * d_state* sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_ssm_K_exp_batch,    bm * (size_t)dt_rank * d_state* sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_ssm_out_batch,      bm * (size_t)d_inner          * sizeof(float)));

                if (r->verbose >= 1) {
                    double ssm_total = (double)(bm * (2*(size_t)qkv_dim + 4*(size_t)d_inner
                                              + 2*(size_t)dt_rank + 2*(size_t)dt_rank*d_state) * sizeof(float));
                    fprintf(stderr, "hip_llm: hybrid batched buffers: d_qfull=%.1f MB, d_attn_gate=%.1f MB, SSM=%.1f MB\n",
                            (double)(bm * max_q_rows * sizeof(float)) / (1024.0*1024.0),
                            (double)(bm * q_dim      * sizeof(float)) / (1024.0*1024.0),
                            ssm_total / (1024.0*1024.0));
                }
            }

            r->batch_path_ok = 1;
            if (r->verbose >= 1) {
                fprintf(stderr,
                    "hip_llm: Phase-2 batched path enabled (BMAX=%d, threshold=%d)\n",
                    batch_max, gemm_thresh);
            }

            /* Plan pre-warm: fire the 7 hipBLASLt prefill GEMM shapes once at
             * representative M values so the per-(M,N,K) plan cache is hot on
             * the first user call. Set LLM_PLAN_PREWARM=0 to skip. */
            const char *prewarm_env = getenv("LLM_PLAN_PREWARM");
            int do_prewarm = (prewarm_env == NULL) ? 1 : atoi(prewarm_env);
            if (do_prewarm && r->n_layers > 0) {
                hip_layer *cl0 = &r->layers[0];
                /* Representative Ms — covers small prompts, medium, and BMAX. */
                int warm_Ms[3];
                int n_warm_Ms = 0;
                warm_Ms[n_warm_Ms++] = 64;
                if (batch_max > 64)  warm_Ms[n_warm_Ms++] = (batch_max < 256 ? batch_max : 256);
                if (batch_max > 256) warm_Ms[n_warm_Ms++] = batch_max;
                double pw0 = (r->verbose >= 1) ? 0.0 : 0.0;
                /* Pre-warm only the F16-pre-converted layers. K-quant layers
                 * skip pre-warm (one-time first-call cost is small). */
                for (int wi = 0; wi < n_warm_Ms; wi++) {
                    int M = warm_Ms[wi];
                    if (cl0->attn_q_w_bf16)
                        mm_blaslt_run_bf16(r->d_q_batch, cl0->attn_q_w_bf16,
                            r->d_xnorm_batch_bf16, M, q_dim, r->n_embd, r->stream);
                    if (cl0->attn_k_w_bf16)
                        mm_blaslt_run_bf16(r->d_k_batch, cl0->attn_k_w_bf16,
                            r->d_xnorm_batch_bf16, M, kv_dim, r->n_embd, r->stream);
                    if (cl0->attn_v_w_bf16)
                        mm_blaslt_run_bf16(r->d_v_batch, cl0->attn_v_w_bf16,
                            r->d_xnorm_batch_bf16, M, kv_dim, r->n_embd, r->stream);
                    if (cl0->attn_output_w_bf16)
                        mm_blaslt_run_bf16(r->d_attn_proj_batch, cl0->attn_output_w_bf16,
                            r->d_attn_out_batch_bf16, M, r->n_embd, q_dim, r->stream);
                    if (cl0->ffn_gate_w_bf16)
                        mm_blaslt_run_bf16(r->d_gate_batch, cl0->ffn_gate_w_bf16,
                            r->d_ffn_norm_batch_bf16, M, r->n_ff, r->n_embd, r->stream);
                    if (cl0->ffn_up_w_bf16)
                        mm_blaslt_run_bf16(r->d_up_batch, cl0->ffn_up_w_bf16,
                            r->d_ffn_norm_batch_bf16, M, r->n_ff, r->n_embd, r->stream);
                    if (cl0->ffn_down_w_bf16)
                        mm_blaslt_run_bf16(r->d_down_batch, cl0->ffn_down_w_bf16,
                            r->d_silu_batch_bf16, M, r->n_embd, r->n_ff, r->stream);
                }
                hipStreamSynchronize(r->stream);
                (void)pw0;
                if (r->verbose >= 1) {
                    fprintf(stderr, "hip_llm: Phase-2 plan pre-warm: %d shapes × %d Ms\n",
                            7, n_warm_Ms);
                }
            }

            /* Phase 3: flash-attention scratch.
             * Two WMMA-FA kernels available: head_dim<=128 (FA_LLM_HD_MAX=128)
             * and a dedicated head_dim<=256 variant (FA_LLM_HD_MAX=256, Opp B)
             * for qwen35-style models. (n_heads % n_kv_heads) == 0 required. */
            int fa_disable = 0;
            const char *fa_disable_env = getenv("LLM_FA_DISABLE");
            if (fa_disable_env) fa_disable = atoi(fa_disable_env);
            int fa_eligible = !fa_disable
                && r->head_dim > 0 && r->head_dim <= 256
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

/* Batched RMSNorm: one block per row, n_rows blocks. row_stride = n unless padded. */
static inline void launch_rmsnorm_batch(hip_llm_runner *r, void *dst, void *x,
                                          void *w, int n, int n_rows, int row_stride,
                                          float eps) {
    void *args[] = { &dst, &x, &w, &n, &row_stride, &eps };
    LAUNCH(r->fn_rmsnorm_batch_f32, n_rows, 1, 1, 256, 1, 1, 256 * sizeof(float),
           r->stream, args);
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

/* Batched QK-norm: grid = (n_heads, n_rows). row_stride = n_heads*head_dim. */
static inline void launch_qknorm_batch(hip_llm_runner *r, void *vec, void *w,
                                         int n_heads, int head_dim,
                                         int n_rows, int row_stride, float eps) {
    int bdim = 1;
    while (bdim < head_dim) bdim <<= 1;
    if (bdim > 256) bdim = 256;
    void *args[] = { &vec, &w, &n_heads, &head_dim, &row_stride, &eps };
    LAUNCH(r->fn_qknorm_batch_f32, n_heads, n_rows, 1, bdim, 1, 1,
           bdim * sizeof(float), r->stream, args);
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

/* Multi-warp variant: 8 warps/block, one warp per row, grid = ceil(n_rows/8).
 * Used by the IQ kernels that compute row = blockIdx.x * 8 + warp_id. */
#define DEFINE_LAUNCH_MATVEC_MW(name, fn_field) \
static inline void launch_matvec_##name(hip_llm_runner *r, void *dst, void *mat, \
                                        void *x, int n_rows, int n_cols) { \
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols }; \
    LAUNCH(r->fn_field, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args); \
}

DEFINE_LAUNCH_MATVEC(q2_K, fn_matvec_q2_K_f32)
DEFINE_LAUNCH_MATVEC(q3_K, fn_matvec_q3_K_f32)
static inline void launch_matvec_q4_K(hip_llm_runner *r, void *dst, void *mat,
                                      void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    if (r->quant_matvec_opt && n_rows <= 4096 && n_cols <= 4096) {
        LAUNCH(r->fn_matvec_q4_K_mw_f32, n_rows, 1, 1, 32, 1, 1, 0,
               r->stream, args);
    } else {
        LAUNCH(r->fn_matvec_q4_K_f32, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args);
    }
}
static inline void launch_matvec_q5_K(hip_llm_runner *r, void *dst, void *mat,
                                      void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    if (r->q5_k_mw) {
        LAUNCH(r->fn_matvec_q5_K_mw_f32, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0,
               r->stream, args);
    } else {
        LAUNCH(r->fn_matvec_q5_K_f32, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args);
    }
}
DEFINE_LAUNCH_MATVEC(q6_K, fn_matvec_q6_K_f32)
DEFINE_LAUNCH_MATVEC_MW(iq2_xxs, fn_matvec_iq2_xxs_f32)
DEFINE_LAUNCH_MATVEC(q4_0, fn_matvec_q4_0_f32)
DEFINE_LAUNCH_MATVEC(q4_1, fn_matvec_q4_1_f32)
DEFINE_LAUNCH_MATVEC(q5_0, fn_matvec_q5_0_f32)
DEFINE_LAUNCH_MATVEC(q5_1, fn_matvec_q5_1_f32)
DEFINE_LAUNCH_MATVEC_MW(iq4_nl, fn_matvec_iq4_nl_f32)
DEFINE_LAUNCH_MATVEC_MW(iq4_xs, fn_matvec_iq4_xs_f32)
DEFINE_LAUNCH_MATVEC_MW(iq2_xs, fn_matvec_iq2_xs_f32)
DEFINE_LAUNCH_MATVEC_MW(iq3_xxs, fn_matvec_iq3_xxs_f32)
DEFINE_LAUNCH_MATVEC_MW(iq2_s, fn_matvec_iq2_s_f32)
DEFINE_LAUNCH_MATVEC_MW(iq3_s, fn_matvec_iq3_s_f32)
DEFINE_LAUNCH_MATVEC(iq1_s, fn_matvec_iq1_s_f32)
DEFINE_LAUNCH_MATVEC(iq1_m, fn_matvec_iq1_m_f32)
DEFINE_LAUNCH_MATVEC(tq1_0, fn_matvec_tq1_0_f32)
DEFINE_LAUNCH_MATVEC(tq2_0, fn_matvec_tq2_0_f32)

#undef DEFINE_LAUNCH_MATVEC
#undef DEFINE_LAUNCH_MATVEC_MW

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

/* Per-call dequant a K-quant weight matrix to a contiguous BF16 buffer.
 * dst:  [n_rows, n_cols] BF16 row-major (typically r->d_wbuf_bf16).
 * mat:  raw block-quant data, native stride per quant type.
 * Launch grid (n_rows, n_blocks_per_row), 256 threads/block (one per element). */
#define DEFINE_LAUNCH_KQ_DEQUANT(qname)                                          \
static inline void launch_dequant_##qname##_to_bf16(hip_llm_runner *r,           \
                                                      void *dst, void *mat,      \
                                                      int n_rows, int n_cols) {  \
    void *args[] = { &dst, &mat, &n_rows, &n_cols };                             \
    int n_blocks_per_row = n_cols / 256;                                         \
    LAUNCH(r->fn_dequant_##qname##_to_bf16, n_rows, n_blocks_per_row, 1,         \
           256, 1, 1, 0, r->stream, args);                                       \
}
DEFINE_LAUNCH_KQ_DEQUANT(q4_K)
DEFINE_LAUNCH_KQ_DEQUANT(q5_K)
DEFINE_LAUNCH_KQ_DEQUANT(q6_K)
DEFINE_LAUNCH_KQ_DEQUANT(iq3_xxs)
DEFINE_LAUNCH_KQ_DEQUANT(iq4_xs)
DEFINE_LAUNCH_KQ_DEQUANT(iq2_xs)
DEFINE_LAUNCH_KQ_DEQUANT(iq2_s)
DEFINE_LAUNCH_KQ_DEQUANT(iq3_s)
#undef DEFINE_LAUNCH_KQ_DEQUANT

/* Return a BF16 weight pointer suitable for mm_blaslt_run_bf16, doing per-call
 * dequant into r->d_wbuf_bf16 if the weight is not F16-pre-converted.
 * NOTE: returns r->d_wbuf_bf16 for non-F16 paths; the caller must consume the
 * pointer before any subsequent get_bf16_weight call that would overwrite the
 * staging buffer. The current prefill code does this naturally — each GEMM is
 * called inline and the staging buffer's lifetime is exactly one mm_blaslt call. */
/* Forward decls used by get_bf16_weight (defined later in this file). */
static inline void launch_convert_f16_to_bf16(hip_llm_runner *r, void *dst,
                                              void *src, int n);

/* True if `type` has a per-call dequant kernel suitable for the batched path. */
static inline int batch_qtype_ok(int type) {
    return type == GGML_TYPE_F16     ||
           type == GGML_TYPE_Q4_K    || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K    || type == GGML_TYPE_IQ3_XXS ||
           type == GGML_TYPE_IQ4_XS  || type == GGML_TYPE_IQ2_XS  ||
           type == GGML_TYPE_IQ2_S   || type == GGML_TYPE_IQ3_S;
}

/* True if every projection weight of an attn+FFN layer has a batched path. */
static inline int layer_is_batched_eligible(const hip_layer *cl) {
    return batch_qtype_ok(cl->attn_q_type)      &&
           batch_qtype_ok(cl->attn_k_type)      &&
           batch_qtype_ok(cl->attn_v_type)      &&
           batch_qtype_ok(cl->attn_output_type) &&
           batch_qtype_ok(cl->ffn_gate_type)    &&
           batch_qtype_ok(cl->ffn_up_type)      &&
           batch_qtype_ok(cl->ffn_down_type);
}

/* True if every SSM projection weight has a batched path. */
static inline int ssm_layer_is_batched_eligible(const hip_layer *cl) {
    return batch_qtype_ok(cl->ssm_qkv_type)   &&
           batch_qtype_ok(cl->ssm_gate_type)  &&
           batch_qtype_ok(cl->ssm_alpha_type) &&
           batch_qtype_ok(cl->ssm_beta_type)  &&
           batch_qtype_ok(cl->ssm_out_type)   &&
           batch_qtype_ok(cl->ffn_gate_type)  &&
           batch_qtype_ok(cl->ffn_up_type)    &&
           batch_qtype_ok(cl->ffn_down_type);
}

static inline void *get_bf16_weight(hip_llm_runner *r, void *raw_w, void *bf16_w,
                                      int type, int n_rows, int n_cols) {
    if (bf16_w) return bf16_w;
    switch (type) {
        case GGML_TYPE_F16: {
            /* SSM weights are typically F16 and aren't pre-converted at load
             * (only attn+FFN F16 weights get cl->*_w_bf16). Convert into the
             * shared staging buffer on the fly. */
            size_t n = (size_t)n_rows * n_cols;
            launch_convert_f16_to_bf16(r, r->d_wbuf_bf16, raw_w, (int)n);
            return r->d_wbuf_bf16;
        }
        case GGML_TYPE_Q4_K:
            launch_dequant_q4_K_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        case GGML_TYPE_Q5_K:
            launch_dequant_q5_K_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        case GGML_TYPE_Q6_K:
            launch_dequant_q6_K_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        case GGML_TYPE_IQ3_XXS:
            launch_dequant_iq3_xxs_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        case GGML_TYPE_IQ4_XS:
            launch_dequant_iq4_xs_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        case GGML_TYPE_IQ2_XS:
            launch_dequant_iq2_xs_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        case GGML_TYPE_IQ2_S:
            launch_dequant_iq2_s_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        case GGML_TYPE_IQ3_S:
            launch_dequant_iq3_s_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        default:
            return NULL;  /* not batchable */
    }
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
    /* Pick the head_dim-appropriate kernel. The hd256 variant doubles LDS
     * footprint and per-thread register pressure but covers qwen35-style
     * head_dim=256 in one batched launch. */
    if (head_dim <= 128) {
        /* 4*16*128 (smK0+smK1+smV0+smV1) + 4*16*16 (smP) f16 = 18432 B. */
        int smem_bytes = (4 * 16 * 128 + 4 * 16 * 16) * 2;
        LAUNCH(r->fn_flash_attn_wmma_f16_4w_causal, n_heads, q_blocks, 1, 128, 1, 1,
               smem_bytes, r->stream, args);
    } else {
        /* 4*16*256 + 4*16*16 f16 = 34816 B. */
        int smem_bytes = (4 * 16 * 256 + 4 * 16 * 16) * 2;
        LAUNCH(r->fn_flash_attn_wmma_f16_4w_causal_hd256, n_heads, q_blocks, 1, 128, 1, 1,
               smem_bytes, r->stream, args);
    }
}

/* SSM launch helpers */
static inline void launch_softplus_mul(hip_llm_runner *r, void *out,
    void *in, void *bias, void *a, int n) {
    void *args[] = { &out, &in, &bias, &a, &n };
    LAUNCH(r->fn_softplus_mul_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

/* Batched softplus_mul: bias/a are dt_rank-wide layer constants, broadcast over M rows. */
static inline void launch_softplus_mul_batch(hip_llm_runner *r, void *out,
    void *in, void *bias, void *a, int dt_rank, int M) {
    void *args[] = { &out, &in, &bias, &a, &dt_rank, &M };
    int total = M * dt_rank;
    LAUNCH(r->fn_softplus_mul_batch_f32, (total + 255) / 256, 1, 1, 256, 1, 1, 0,
           r->stream, args);
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

/* Fused multi-token DeltaNet step: M sequential token steps in one launch.
 * Each thread keeps its row of the state matrix in registers across the M
 * iterations, eliminating the M × (state_row R + W) global traffic of the
 * per-token kernel. Requires d_state <= 128 (kernel-side compile-time max). */
static inline void launch_deltanet_step_batch(hip_llm_runner *r, void *state,
    void *out_batch, void *Q_batch, void *K_batch, void *V_batch,
    void *alpha_batch, void *beta_batch, int dt_rank, int d_state,
    int v_row_stride, int M) {
    void *args[] = { &state, &out_batch, &Q_batch, &K_batch, &V_batch,
                     &alpha_batch, &beta_batch, &dt_rank, &d_state,
                     &v_row_stride, &M };
    LAUNCH(r->fn_deltanet_step_batch_f32, dt_rank, 1, 1, d_state, 1, 1, 0,
           r->stream, args);
}

static inline void launch_gated_rmsnorm_silu(hip_llm_runner *r, void *out,
    void *z, void *norm_w, int dt_rank, int d_state, float eps) {
    int threads = (d_state <= 128) ? 128 : 256;
    void *args[] = { &out, &z, &norm_w, &dt_rank, &d_state, &eps };
    LAUNCH(r->fn_gated_rmsnorm_silu_f32, dt_rank, 1, 1, threads, 1, 1, threads * sizeof(float), r->stream, args);
}

/* ---- Batched SSM aux op launchers (one launch over M rows) ---- */

static inline void launch_l2_norm_heads_batch(hip_llm_runner *r, void *data,
    int n_heads, int head_dim, int row_stride, int M, float eps) {
    int threads = (head_dim <= 128) ? 128 : 256;
    void *args[] = { &data, &n_heads, &head_dim, &row_stride, &M, &eps };
    LAUNCH(r->fn_l2_norm_heads_batch_f32, n_heads, M, 1, threads, 1, 1,
           threads * sizeof(float), r->stream, args);
}

static inline void launch_repeat_tile_batch(hip_llm_runner *r, void *dst, void *src,
    int dt_rank, int d_state, int n_group,
    int src_row_stride, int dst_row_stride, int M) {
    int total = M * dt_rank * d_state;
    void *args[] = { &dst, &src, &dt_rank, &d_state, &n_group,
                     &src_row_stride, &dst_row_stride, &M };
    LAUNCH(r->fn_repeat_tile_batch_f32, (total + 255) / 256, 1, 1, 256, 1, 1, 0,
           r->stream, args);
}

static inline void launch_gated_rmsnorm_silu_batch(hip_llm_runner *r, void *out,
    void *z, void *norm_w, int dt_rank, int d_state, int row_stride, int M,
    float eps) {
    int threads = (d_state <= 128) ? 128 : 256;
    void *args[] = { &out, &z, &norm_w, &dt_rank, &d_state, &row_stride, &M, &eps };
    LAUNCH(r->fn_gated_rmsnorm_silu_batch_f32, dt_rank, M, 1, threads, 1, 1,
           threads * sizeof(float), r->stream, args);
}

static inline void launch_conv1d_batch(hip_llm_runner *r, void *conv_out_batch,
    void *conv_state, void *input_batch, void *weight,
    int qkv_dim, int conv_k, int row_stride, int M) {
    void *args[] = { &conv_out_batch, &conv_state, &input_batch, &weight,
                     &qkv_dim, &conv_k, &row_stride, &M };
    LAUNCH(r->fn_conv1d_depthwise_silu_batch_f32, (qkv_dim + 255) / 256, 1, 1,
           256, 1, 1, 0, r->stream, args);
}

static inline void launch_sigmoid_mul(hip_llm_runner *r, void *data, void *gate, int n) {
    void *args[] = { &data, &gate, &n };
    LAUNCH(r->fn_sigmoid_mul_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static inline void launch_deinterleave_qgate_batch(hip_llm_runner *r,
        void *q_out, void *gate_out, void *qfull,
        int n_heads, int head_dim, int M) {
    void *args[] = { &q_out, &gate_out, &qfull, &n_heads, &head_dim, &M };
    int total = M * n_heads * head_dim;
    LAUNCH(r->fn_deinterleave_qgate_batch_f32,
           (total + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
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
static void forward_one_layer(hip_llm_runner *r, int l);
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
/* Per-token compute for a single transformer layer. Reads r->d_x (the current
 * token's hidden state) and r->d_position (the current absolute token position
 * on the device, used by rope_devp / kv_store_devp). Writes the residual'd
 * hidden state back into r->d_x. Uses runner-scoped scratch buffers (r->d_xb,
 * r->d_q, r->d_k, r->d_v, r->d_xb2, r->d_ssm_*).
 *
 * This is the per-token layer body lifted out of forward_blocks_body so the
 * batched prefill dispatcher can also call it per-row for hybrid layers that
 * can't yet batch (B2 Phase 1). When the same function is invoked from the
 * decode-time loop AND the batched-prefill per-row fallback, all that differs
 * is the pointer r->d_x and the value at *r->d_position. */
static void forward_one_layer(hip_llm_runner *r, int l) {
    int n_embd     = r->n_embd;
    int n_heads    = r->n_heads;
    int n_kv_heads = r->n_kv_heads;
    int head_dim   = r->head_dim;
    int kv_dim     = n_kv_heads * head_dim;
    int n_ff       = r->n_ff;
    float eps      = r->rms_norm_eps;
    hip_layer *cl  = &r->layers[l];

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

/* Per-token forward over all layers + final RMSNorm. */
static void forward_blocks_body(hip_llm_runner *r) {
    int n_run_layers = r->n_layers;
    if (r->max_layers > 0 && r->max_layers < r->n_layers) n_run_layers = r->max_layers;
    for (int l = 0; l < n_run_layers; l++) {
        forward_one_layer(r, l);
    }
    /* Final RMSNorm */
    launch_rmsnorm(r, r->d_x, r->d_x, r->d_output_norm, r->n_embd, r->rms_norm_eps);
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
     * so it's safe to capture for n_deepstack>0 models like Qwen3-VL.
     *
     * Hybrid (dense+SSM) is also capturable: SSM kernels mutate
     * cl->d_conv_state / cl->d_recurrent_state in place through fixed device
     * pointers, so replay re-executes the recurrence correctly. Gated-attn
     * uses the same *_devp launchers as standard attn (position via
     * r->d_position). NOTE: the capture warm-pass at warm_pos=0 advances SSM
     * state by one bogus step, so hip_llm_reset_state() is called after
     * capture to zero it back out (see decode-graph-capture-audit.md).
     *
     * MoE stays gated: cl->is_moe branch synchronizes the stream and reads
     * router logits to host for top-K selection — incompatible with capture. */
    r->graph_eligible =
        !r->is_moe && !r->debug_layers;

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

    /* The capture's warm-pass at position 0 advances SSM state by one bogus
     * step (conv_state / recurrent_state are mutated in-place by SSM kernels).
     * For non-hybrid this is harmless (KV slot 0 gets overwritten on the first
     * real call). For hybrid we must zero the state back out — otherwise the
     * first real decode token sees a primed state from the warm pass. */
    if (r->is_hybrid && (r->graph_ready_logits || r->graph_ready_hidden)) {
        hip_llm_reset_state(r);
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

    /* One-shot per-runner dispatch summary, gated by LLM_DEBUG_DISPATCH=1.
     * Helpful for diagnosing hybrid / mixed-quant models whose layers fall
     * back to per-row because a weight type lacks a dequant kernel
     * (e.g. IQ2_S, IQ3_S, IQ4_XS — not yet ported). */
    static int s_dump_dispatch_once = -1;
    int dump_dispatch = 0;
    if (s_dump_dispatch_once == -1) {
        const char *env_dd = getenv("LLM_DEBUG_DISPATCH");
        if (env_dd && atoi(env_dd) != 0) dump_dispatch = 1;
        s_dump_dispatch_once = 0;
    }

    int n_ssm = 0, n_per_row_quant = 0, n_batched = 0, n_gated = 0;

    for (int l = 0; l < n_run_layers; l++) {
        hip_layer *cl = &r->layers[l];

        if (dump_dispatch) {
            const char *kind;
            if (cl->is_ssm) {
                if (ssm_layer_is_batched_eligible(cl)) { kind = "BATCH-SSM"; n_batched++; }
                else { kind = "SSM per-row";   n_ssm++; }
            } else if (!layer_is_batched_eligible(cl)) {
                kind = "per-row(quant)"; n_per_row_quant++;
            } else if (cl->attn_q_rows == 2 * (r->n_heads * r->head_dim)) {
                kind = "BATCH-gated";  n_batched++; n_gated++;
            } else {
                kind = "BATCH-std";    n_batched++;
            }
            if (cl->is_ssm) {
                fprintf(stderr, "  L%02d %-14s  ssm:qkv=%d gate=%d a=%d b=%d out=%d  ffn:g=%d u=%d d=%d\n",
                        l, kind, cl->ssm_qkv_type, cl->ssm_gate_type,
                        cl->ssm_alpha_type, cl->ssm_beta_type, cl->ssm_out_type,
                        cl->ffn_gate_type, cl->ffn_up_type, cl->ffn_down_type);
            } else {
                fprintf(stderr, "  L%02d %-14s  q=%d k=%d v=%d o=%d g=%d u=%d d=%d\n",
                        l, kind, cl->attn_q_type, cl->attn_k_type, cl->attn_v_type,
                        cl->attn_output_type, cl->ffn_gate_type, cl->ffn_up_type, cl->ffn_down_type);
            }
        }

        /* Per-layer dispatch:
         *   - SSM layer + all 5 SSM weights batchable  -> batched SSM (B2 P3)
         *   - SSM layer + any SSM weight not batchable -> per-row fallback
         *   - non-SSM + all 7 attn/FFN weights batchable -> existing batched
         *   - non-SSM + any weight not batchable         -> per-row fallback
         */
        int can_batch_ssm  =  cl->is_ssm && ssm_layer_is_batched_eligible(cl);
        int can_batch_attn = !cl->is_ssm && layer_is_batched_eligible(cl);
        if (!can_batch_ssm && !can_batch_attn) {
            void *saved_d_x = r->d_x;
            for (int m = 0; m < M; m++) {
                int pos = position_start + m;
                r->d_x = (char *)r->d_x_batch + (size_t)m * n_embd * sizeof(float);
                hipMemcpyAsync(r->d_position, &pos, sizeof(int),
                               hipMemcpyHostToDevice, r->stream);
                forward_one_layer(r, l);
            }
            r->d_x = saved_d_x;
            continue;
        }

        /* B2 Phase 2: qwen35 gated-attn layers (is_hybrid && attn_q_rows == 2*q_dim)
         * batch the projections + attention through the existing flow, with extra
         * deinterleave_qgate_batch after Q proj and sigmoid_mul before output proj. */
        int is_gated_attn = (cl->attn_q_rows == 2 * q_dim);

        /* ---- Pre-attention RMSNorm: one batched launch over M rows ---- */
        launch_rmsnorm_batch(r, r->d_xnorm_batch, r->d_x_batch,
                             cl->attn_norm_w, n_embd, M, n_embd, eps);

        /* Pack BF16 input (single launch over M*n_embd) */
        launch_pack_bf16_from_f32(r, r->d_xnorm_batch_bf16,
                                  r->d_xnorm_batch, M * n_embd);

        if (can_batch_ssm) {
            /* === B2 Phase 3: SSM batched compute ===
             * 5 SSM projections (qkv/gate/alpha/beta/out) go through hipBLASLt;
             * elementwise softplus_mul and sigmoid_inplace are batched; the
             * stateful conv1d + deltanet_step + their per-token aux ops run
             * in a per-row host loop (the conv state and DeltaNet recurrent
             * state are intrinsically sequential). */
            int qkv_dim = r->ssm_qkv_dim;
            int d_state = r->ssm_d_state;
            int n_group = r->ssm_n_group;
            int dt_rank = r->ssm_dt_rank;
            int d_inner = r->ssm_d_inner;
            int conv_k  = r->ssm_conv_kernel;

            /* 4 input projections (qkv, gate, alpha, beta) batched */
            #define SSM_GEMM(dst, w_field, type_field, rows_field, cols_field) do {       \
                void *_w = get_bf16_weight(r, cl->w_field, NULL,                          \
                                cl->type_field, cl->rows_field, cl->cols_field);          \
                if (!_w) return -1;                                                       \
                if (mm_blaslt_run_bf16(dst, _w, r->d_xnorm_batch_bf16, M,                 \
                                       cl->rows_field, n_embd, r->stream) != 0) return -1;\
            } while(0)
            SSM_GEMM(r->d_ssm_qkv_batch,   ssm_qkv_w,   ssm_qkv_type,   ssm_qkv_rows,   ssm_qkv_cols);
            SSM_GEMM(r->d_ssm_z_batch,     ssm_gate_w,  ssm_gate_type,  ssm_gate_rows,  ssm_gate_cols);
            SSM_GEMM(r->d_ssm_alpha_batch, ssm_alpha_w, ssm_alpha_type, ssm_alpha_rows, ssm_alpha_cols);
            SSM_GEMM(r->d_ssm_beta_batch,  ssm_beta_w,  ssm_beta_type,  ssm_beta_rows,  ssm_beta_cols);
            #undef SSM_GEMM

            /* softplus_mul broadcasts dt_rank-wide bias/a over M rows; sigmoid
             * is a flat elementwise op so the per-token kernel works as-is. */
            launch_softplus_mul_batch(r, r->d_ssm_alpha_batch, r->d_ssm_alpha_batch,
                                      cl->ssm_dt_bias, cl->ssm_a, dt_rank, M);
            launch_sigmoid_inplace(r, r->d_ssm_beta_batch, M * dt_rank);

            /* All aux ops now batched over M rows:
             *   conv1d:           one launch, M sequential steps fused
             *                     inside the kernel (state in registers)
             *   l2_norm_heads:    one launch covering n_group × M heads,
             *                     once for the Q slice and once for the K slice
             *   repeat_tile:      one launch each for Q→Q_exp and K→K_exp
             *   deltanet_step:    one fused launch over all M rows (Opp A+A2)
             *   gated_rmsnorm:    one launch covering dt_rank × M rows
             * Replaces ~6*M per-row launches per layer with ~6 launches. */
            launch_conv1d_batch(r, r->d_ssm_conv_out_batch, cl->d_conv_state,
                                r->d_ssm_qkv_batch, cl->ssm_conv1d_w,
                                qkv_dim, conv_k, qkv_dim, M);

            float *conv_out_base = (float *)r->d_ssm_conv_out_batch;
            float *K_raw_base    = conv_out_base + (size_t)n_group * d_state;
            launch_l2_norm_heads_batch(r, conv_out_base, n_group, d_state,
                                       qkv_dim, M, eps);                 /* Q */
            launch_l2_norm_heads_batch(r, K_raw_base,    n_group, d_state,
                                       qkv_dim, M, eps);                 /* K */
            launch_repeat_tile_batch(r, r->d_ssm_Q_exp_batch, conv_out_base,
                                     dt_rank, d_state, n_group,
                                     qkv_dim, dt_rank * d_state, M);
            launch_repeat_tile_batch(r, r->d_ssm_K_exp_batch, K_raw_base,
                                     dt_rank, d_state, n_group,
                                     qkv_dim, dt_rank * d_state, M);

            /* Phase B: fused multi-token deltanet step. V lives inside conv_out
             * at offset 2*n_group*d_state with stride qkv_dim per row. */
            float *V_base = (float *)r->d_ssm_conv_out_batch
                              + (size_t)2 * n_group * d_state;
            int use_deltanet_batch = (d_state <= 128);
            if (use_deltanet_batch) {
                launch_deltanet_step_batch(r, cl->d_recurrent_state,
                    r->d_ssm_out_batch,
                    r->d_ssm_Q_exp_batch, r->d_ssm_K_exp_batch, V_base,
                    r->d_ssm_alpha_batch, r->d_ssm_beta_batch,
                    dt_rank, d_state, qkv_dim, M);
            } else {
                /* Fallback: per-row deltanet for d_state > 128. */
                for (int m = 0; m < M; m++) {
                    size_t qkv_off    = (size_t)m * qkv_dim;
                    size_t dt_off     = (size_t)m * dt_rank;
                    size_t dinner_off = (size_t)m * d_inner;
                    size_t exp_off    = (size_t)m * dt_rank * d_state;
                    launch_deltanet_step(r, cl->d_recurrent_state,
                        (float *)r->d_ssm_out_batch    + dinner_off,
                        (float *)r->d_ssm_Q_exp_batch  + exp_off,
                        (float *)r->d_ssm_K_exp_batch  + exp_off,
                        (float *)r->d_ssm_conv_out_batch + qkv_off
                                    + (size_t)2 * n_group * d_state,
                        (float *)r->d_ssm_alpha_batch + dt_off,
                        (float *)r->d_ssm_beta_batch  + dt_off,
                        dt_rank, d_state);
                }
            }

            /* Batched gated_rmsnorm_silu: one launch over dt_rank × M heads. */
            launch_gated_rmsnorm_silu_batch(r, r->d_ssm_out_batch, r->d_ssm_z_batch,
                                            cl->ssm_norm_w, dt_rank, d_state,
                                            dt_rank * d_state, M, eps);

            /* ssm_out projection: d_inner -> n_embd, batched. Reuse d_silu_batch_bf16
             * as packing scratch (sized for n_ff >= d_inner). */
            launch_pack_bf16_from_f32(r, r->d_silu_batch_bf16,
                                      r->d_ssm_out_batch, M * d_inner);
            {
                void *ow = get_bf16_weight(r, cl->ssm_out_w, NULL,
                                cl->ssm_out_type, cl->ssm_out_rows, cl->ssm_out_cols);
                if (!ow) return -1;
                if (mm_blaslt_run_bf16(r->d_attn_proj_batch, ow,
                                       r->d_silu_batch_bf16, M,
                                       n_embd, d_inner, r->stream) != 0) return -1;
            }

            /* Residual: x_batch += ssm_proj */
            launch_add(r, r->d_x_batch, r->d_attn_proj_batch, M * n_embd);

            /* Jump past the attn block to the FFN section. */
            goto ffn_section;
        }

        /* ---- Q/K/V GEMMs (3 hipBLASLt calls; per-call dequant for K-quant rows).
         * Gated attn: Q projection produces [M, 2*q_dim] into d_qfull_batch,
         * then deinterleave_qgate_batch splits into d_q_batch + d_attn_gate_batch. */
        {
            void *qw = get_bf16_weight(r, cl->attn_q_w, cl->attn_q_w_bf16,
                                       cl->attn_q_type, cl->attn_q_rows, cl->attn_q_cols);
            if (!qw) return -1;
            int q_proj_rows = cl->attn_q_rows;  /* q_dim or 2*q_dim */
            void *q_dst = is_gated_attn ? r->d_qfull_batch : r->d_q_batch;
            if (mm_blaslt_run_bf16(q_dst, qw,
                                   r->d_xnorm_batch_bf16, M, q_proj_rows, n_embd, r->stream) != 0) return -1;
            if (is_gated_attn) {
                launch_deinterleave_qgate_batch(r, r->d_q_batch, r->d_attn_gate_batch,
                                                r->d_qfull_batch, n_heads, head_dim, M);
            }
        }
        {
            void *kw = get_bf16_weight(r, cl->attn_k_w, cl->attn_k_w_bf16,
                                       cl->attn_k_type, cl->attn_k_rows, cl->attn_k_cols);
            if (!kw) return -1;
            if (mm_blaslt_run_bf16(r->d_k_batch, kw,
                                   r->d_xnorm_batch_bf16, M, kv_dim, n_embd, r->stream) != 0) return -1;
        }
        {
            void *vw = get_bf16_weight(r, cl->attn_v_w, cl->attn_v_w_bf16,
                                       cl->attn_v_type, cl->attn_v_rows, cl->attn_v_cols);
            if (!vw) return -1;
            if (mm_blaslt_run_bf16(r->d_v_batch, vw,
                                   r->d_xnorm_batch_bf16, M, kv_dim, n_embd, r->stream) != 0) return -1;
        }

        /* ---- Biases (per-row loop kept; one launch per row is cheap, only 3 × M
         *      ~ several thousand launches at L=1024 — defer to a batched bias kernel
         *      if it shows up in profile. QK-norm is the heavy one and is batched.) ---- */
        if (cl->attn_q_bias || cl->attn_k_bias || cl->attn_v_bias) {
            for (int m = 0; m < M; m++) {
                float *qrow = (float *)r->d_q_batch + (size_t)m * q_dim;
                float *krow = (float *)r->d_k_batch + (size_t)m * kv_dim;
                float *vrow = (float *)r->d_v_batch + (size_t)m * kv_dim;
                if (cl->attn_q_bias) launch_add(r, qrow, cl->attn_q_bias, q_dim);
                if (cl->attn_k_bias) launch_add(r, krow, cl->attn_k_bias, kv_dim);
                if (cl->attn_v_bias) launch_add(r, vrow, cl->attn_v_bias, kv_dim);
            }
        }
        /* ---- QK-norm: batched over M rows ---- */
        if (cl->has_qk_norm) {
            if (cl->attn_q_norm_w)
                launch_qknorm_batch(r, r->d_q_batch, cl->attn_q_norm_w,
                                    n_heads,    head_dim, M, q_dim,  eps);
            if (cl->attn_k_norm_w)
                launch_qknorm_batch(r, r->d_k_batch, cl->attn_k_norm_w,
                                    n_kv_heads, head_dim, M, kv_dim, eps);
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

        /* Gated attn: d_attn_out_batch *= sigmoid(d_attn_gate_batch). The existing
         * sigmoid_mul_f32 kernel is flat-indexed, so a single launch over M*q_dim
         * elements covers the whole batch. */
        if (is_gated_attn) {
            launch_sigmoid_mul(r, r->d_attn_out_batch, r->d_attn_gate_batch, M * q_dim);
        }

        /* ---- Output projection: [M, q_dim] x W_o^T -> [M, n_embd] ---- */
        launch_pack_bf16_from_f32(r, r->d_attn_out_batch_bf16,
                                  r->d_attn_out_batch, M * q_dim);
        {
            void *ow = get_bf16_weight(r, cl->attn_output_w, cl->attn_output_w_bf16,
                                       cl->attn_output_type, cl->attn_output_rows, cl->attn_output_cols);
            if (!ow) return -1;
            if (mm_blaslt_run_bf16(r->d_attn_proj_batch, ow,
                                   r->d_attn_out_batch_bf16, M, n_embd, q_dim, r->stream) != 0) return -1;
        }

        /* Residual: x_batch += attn_proj */
        launch_add(r, r->d_x_batch, r->d_attn_proj_batch, M * n_embd);

ffn_section:
        /* ---- Pre-FFN RMSNorm: one batched launch over M rows ---- */
        launch_rmsnorm_batch(r, r->d_xnorm_batch, r->d_x_batch,
                             cl->ffn_norm_w, n_embd, M, n_embd, eps);
        launch_pack_bf16_from_f32(r, r->d_ffn_norm_batch_bf16,
                                  r->d_xnorm_batch, M * n_embd);

        /* ---- gate/up GEMMs ---- */
        {
            void *gw = get_bf16_weight(r, cl->ffn_gate_w, cl->ffn_gate_w_bf16,
                                       cl->ffn_gate_type, cl->ffn_gate_rows, cl->ffn_gate_cols);
            if (!gw) return -1;
            if (mm_blaslt_run_bf16(r->d_gate_batch, gw,
                                   r->d_ffn_norm_batch_bf16, M, n_ff, n_embd, r->stream) != 0) return -1;
        }
        {
            void *uw = get_bf16_weight(r, cl->ffn_up_w, cl->ffn_up_w_bf16,
                                       cl->ffn_up_type, cl->ffn_up_rows, cl->ffn_up_cols);
            if (!uw) return -1;
            if (mm_blaslt_run_bf16(r->d_up_batch, uw,
                                   r->d_ffn_norm_batch_bf16, M, n_ff, n_embd, r->stream) != 0) return -1;
        }

        /* SiLU(gate) * up — elementwise across [M, n_ff] */
        launch_silu_mul(r, r->d_gate_batch, r->d_up_batch, M * n_ff);

        /* Pack and down projection */
        launch_pack_bf16_from_f32(r, r->d_silu_batch_bf16,
                                  r->d_gate_batch, M * n_ff);
        {
            void *dw = get_bf16_weight(r, cl->ffn_down_w, cl->ffn_down_w_bf16,
                                       cl->ffn_down_type, cl->ffn_down_rows, cl->ffn_down_cols);
            if (!dw) return -1;
            if (mm_blaslt_run_bf16(r->d_down_batch, dw,
                                   r->d_silu_batch_bf16, M, n_embd, n_ff, r->stream) != 0) return -1;
        }

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

    if (dump_dispatch) {
        fprintf(stderr, "  dispatch summary: %d batched (%d gated) / %d SSM / %d per-row (unsupported quant)\n",
                n_batched, n_gated, n_ssm, n_per_row_quant);
    }

    /* ---- Final RMSNorm: one batched launch over M rows ---- */
    launch_rmsnorm_batch(r, r->d_x_batch, r->d_x_batch,
                         r->d_output_norm, n_embd, M, n_embd, eps);

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

/* Batched embedding forward: feeds M pre-computed embeddings through the
 * batched prefill path in one shot (positions [position_start, +M)). The
 * input layout is M rows of `embd_stride` floats each — the first n_embd
 * floats are the main embedding, the rest (if embd_stride > n_embd) are
 * DeepStack slices for layers 0..n_deepstack-1 (same convention as the
 * per-token hip_llm_forward_embd).
 *
 * Falls back to a per-token loop if the batched path isn't eligible (e.g.
 * M > BMAX, or LLM_BATCH_DISABLE). Returns r->h_output (hidden state of
 * the last fed token) on success, NULL on failure.
 *
 * Limits: M must satisfy position_start + M <= max_seq_len. If M exceeds
 * BMAX the caller should chunk; this implementation does NOT split. */
float *hip_llm_forward_batch_embd(hip_llm_runner *r, const float *embds,
                                    int M, int embd_stride, int position_start) {
    if (!r || !r->weights_loaded || !embds || M <= 0) return NULL;
    if (position_start < 0 || position_start + M > r->max_seq_len) return NULL;

    int n_embd = r->n_embd;

#ifdef LLM_HIPBLASLT_ENABLED
    if (r->batch_path_ok && M >= r->gemm_m_threshold && M <= r->batch_max) {
        /* Copy the M main embeddings (first n_embd floats of each row) into
         * d_x_batch. Rows are non-contiguous in source when embd_stride > n_embd
         * (deepstack layout), so issue one memcpy per row — M is small (a few
         * hundred at most). */
        for (int m = 0; m < M; m++) {
            const float *src = embds + (size_t)m * embd_stride;
            float *dst = (float *)r->d_x_batch + (size_t)m * n_embd;
            hipMemcpyAsync(dst, src, (size_t)n_embd * sizeof(float),
                           hipMemcpyHostToDevice, r->stream);
        }

        /* DeepStack: the layer loop reads slices on the fly from the host
         * pointer via hipMemcpyAsync per (layer, row). For VLM workloads
         * with n_deepstack=3 and M ~ 260, that's ~780 small uploads — a
         * known overhead but still vastly less than the per-token forward. */
        r->_ds_embd        = embds;
        r->_ds_embd_stride = embd_stride;

        int rc = forward_block_batched_dense(r, M, position_start);

        r->_ds_embd        = NULL;
        r->_ds_embd_stride = 0;

        if (rc != 0) return NULL;

        /* Hand back the last row's hidden state in r->h_output (same contract
         * as hip_llm_forward_embd). r->d_x already holds it after
         * forward_block_batched_dense. */
        hipMemcpyAsync(r->h_output, r->d_x, (size_t)n_embd * sizeof(float),
                       hipMemcpyDeviceToHost, r->stream);
        hipStreamSynchronize(r->stream);
        return r->h_output;
    }
#endif

    /* Fallback: per-token loop using the existing forward_embd path. */
    float *last = NULL;
    for (int m = 0; m < M; m++) {
        last = hip_llm_forward_embd(r, embds + (size_t)m * embd_stride,
                                    embd_stride, position_start + m);
        if (!last) return NULL;
    }
    return last;
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
    if (r->d_wbuf_bf16)           { hipFree(r->d_wbuf_bf16);           r->d_wbuf_bf16 = NULL; }
    if (r->d_qfull_batch)         { hipFree(r->d_qfull_batch);         r->d_qfull_batch = NULL; }
    if (r->d_attn_gate_batch)     { hipFree(r->d_attn_gate_batch);     r->d_attn_gate_batch = NULL; }
    if (r->d_ssm_qkv_batch)       { hipFree(r->d_ssm_qkv_batch);       r->d_ssm_qkv_batch = NULL; }
    if (r->d_ssm_z_batch)         { hipFree(r->d_ssm_z_batch);         r->d_ssm_z_batch = NULL; }
    if (r->d_ssm_alpha_batch)     { hipFree(r->d_ssm_alpha_batch);     r->d_ssm_alpha_batch = NULL; }
    if (r->d_ssm_beta_batch)      { hipFree(r->d_ssm_beta_batch);      r->d_ssm_beta_batch = NULL; }
    if (r->d_ssm_conv_out_batch)  { hipFree(r->d_ssm_conv_out_batch);  r->d_ssm_conv_out_batch = NULL; }
    if (r->d_ssm_Q_exp_batch)     { hipFree(r->d_ssm_Q_exp_batch);     r->d_ssm_Q_exp_batch = NULL; }
    if (r->d_ssm_K_exp_batch)     { hipFree(r->d_ssm_K_exp_batch);     r->d_ssm_K_exp_batch = NULL; }
    if (r->d_ssm_out_batch)       { hipFree(r->d_ssm_out_batch);       r->d_ssm_out_batch = NULL; }
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
    if (r->d_wbuf_bf16)           hipFree(r->d_wbuf_bf16);
    if (r->d_qfull_batch)         hipFree(r->d_qfull_batch);
    if (r->d_attn_gate_batch)     hipFree(r->d_attn_gate_batch);
    if (r->d_ssm_qkv_batch)       hipFree(r->d_ssm_qkv_batch);
    if (r->d_ssm_z_batch)         hipFree(r->d_ssm_z_batch);
    if (r->d_ssm_alpha_batch)     hipFree(r->d_ssm_alpha_batch);
    if (r->d_ssm_beta_batch)      hipFree(r->d_ssm_beta_batch);
    if (r->d_ssm_conv_out_batch)  hipFree(r->d_ssm_conv_out_batch);
    if (r->d_ssm_Q_exp_batch)     hipFree(r->d_ssm_Q_exp_batch);
    if (r->d_ssm_K_exp_batch)     hipFree(r->d_ssm_K_exp_batch);
    if (r->d_ssm_out_batch)       hipFree(r->d_ssm_out_batch);
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

/* Local block metadata for quant matvec kernels. ggml_type_info is a static
 * const in gguf_loader.h, only visible to translation units that define
 * GGUF_LOADER_IMPLEMENTATION, so the runner keeps the small table it needs. */
static int quant_matvec_block_info(int weight_type, int *blk_elems, int *blk_bytes) {
    if (!blk_elems || !blk_bytes) return -1;
    *blk_elems = 0;
    *blk_bytes = 0;
    switch (weight_type) {
        case GGML_TYPE_Q4_K:    *blk_elems = 256; *blk_bytes = 144; break;
        case GGML_TYPE_Q5_K:    *blk_elems = 256; *blk_bytes = 176; break;
        case GGML_TYPE_Q6_K:    *blk_elems = 256; *blk_bytes = 210; break;
        case GGML_TYPE_Q4_0:    *blk_elems = 32;  *blk_bytes = 18;  break;
        case GGML_TYPE_Q4_1:    *blk_elems = 32;  *blk_bytes = 20;  break;
        case GGML_TYPE_Q5_0:    *blk_elems = 32;  *blk_bytes = 22;  break;
        case GGML_TYPE_Q5_1:    *blk_elems = 32;  *blk_bytes = 24;  break;
        case GGML_TYPE_IQ2_XXS: *blk_elems = 256; *blk_bytes = 66;  break;
        case GGML_TYPE_IQ2_XS:  *blk_elems = 256; *blk_bytes = 74;  break;
        case GGML_TYPE_IQ2_S:   *blk_elems = 256; *blk_bytes = 82;  break;
        case GGML_TYPE_IQ3_XXS: *blk_elems = 256; *blk_bytes = 98;  break;
        case GGML_TYPE_IQ3_S:   *blk_elems = 256; *blk_bytes = 110; break;
        case GGML_TYPE_IQ1_S:   *blk_elems = 256; *blk_bytes = 50;  break;
        case GGML_TYPE_IQ1_M:   *blk_elems = 256; *blk_bytes = 56;  break;
        case GGML_TYPE_IQ4_NL:  *blk_elems = 32;  *blk_bytes = 18;  break;
        case GGML_TYPE_IQ4_XS:  *blk_elems = 256; *blk_bytes = 136; break;
        case GGML_TYPE_TQ1_0:   *blk_elems = 256; *blk_bytes = 54;  break;
        case GGML_TYPE_TQ2_0:   *blk_elems = 256; *blk_bytes = 66;  break;
        default: return -1;
    }
    return 0;
}

static void fill_quant_matvec_inputs(unsigned char *h_mat, size_t mat_bytes,
                                     float *h_x, int n_cols, int weight_type) {
    unsigned s = 0xC0FFEEu ^ (unsigned)weight_type;
    for (size_t i = 0; i < mat_bytes; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        h_mat[i] = (unsigned char)s;
    }
    for (int i = 0; i < n_cols; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        h_x[i] = (((float)((s >> 8) & 0xFFFFFF) / 16777216.0f) - 0.5f) * 0.1f;
    }
}

/* A/B verify: GPU matvec_<type> vs caller-provided CPU dequant + scalar matvec
 * on identical raw quant bytes. Used by test_hip_llm --verify-quant-kernels. */
int hip_llm_verify_quant_matvec(
        hip_llm_runner *r, int weight_type,
        void (*cpu_dequant_row)(const void *src, float *dst, int n),
        int n_rows, int n_cols,
        double *out_rel_l2, double *out_max_abs) {
    if (!r || !cpu_dequant_row || n_rows <= 0 || n_cols <= 0) return -1;

    int blk_elems = 0, blk_bytes = 0;
    if (quant_matvec_block_info(weight_type, &blk_elems, &blk_bytes) != 0) return -1;
    if ((n_cols % blk_elems) != 0) return -1;
    int n_blk_per_row = n_cols / blk_elems;
    size_t row_bytes  = (size_t)n_blk_per_row * blk_bytes;
    size_t mat_bytes  = (size_t)n_rows * row_bytes;

    unsigned char *h_mat  = (unsigned char *)malloc(mat_bytes);
    float *h_x       = (float *)malloc((size_t)n_cols * sizeof(float));
    float *h_dst_hip = (float *)malloc((size_t)n_rows * sizeof(float));
    float *h_dst_ref = (float *)malloc((size_t)n_rows * sizeof(float));
    float *h_row_f32 = (float *)malloc((size_t)n_cols * sizeof(float));
    if (!h_mat || !h_x || !h_dst_hip || !h_dst_ref || !h_row_f32) {
        free(h_mat); free(h_x); free(h_dst_hip); free(h_dst_ref); free(h_row_f32);
        return -2;
    }

    fill_quant_matvec_inputs(h_mat, mat_bytes, h_x, n_cols, weight_type);

    void *d_mat = NULL, *d_x = NULL, *d_dst = NULL;
    if (hipMalloc(&d_mat, mat_bytes) != hipSuccess ||
        hipMalloc(&d_x,   (size_t)n_cols * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_dst, (size_t)n_rows * sizeof(float)) != hipSuccess) {
        if (d_mat) hipFree(d_mat);
        if (d_x) hipFree(d_x);
        if (d_dst) hipFree(d_dst);
        free(h_mat); free(h_x); free(h_dst_hip); free(h_dst_ref); free(h_row_f32);
        return -2;
    }
    hipMemcpy(d_mat, h_mat, mat_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_x,   h_x,   (size_t)n_cols * sizeof(float), hipMemcpyHostToDevice);

    launch_matvec_auto(r, d_dst, d_mat, d_x, n_rows, n_cols, weight_type);
    hipStreamSynchronize(r->stream);
    hipMemcpy(h_dst_hip, d_dst, (size_t)n_rows * sizeof(float), hipMemcpyDeviceToHost);

    /* CPU reference. */
    for (int row = 0; row < n_rows; row++) {
        const void *row_ptr = h_mat + (size_t)row * row_bytes;
        cpu_dequant_row(row_ptr, h_row_f32, n_cols);
        double acc = 0.0;
        for (int k = 0; k < n_cols; k++) acc += (double)h_row_f32[k] * (double)h_x[k];
        h_dst_ref[row] = (float)acc;
    }

    /* Random raw bytes can include FP16 NaN/Inf in the scale field, which
     * propagates through the dequant + dot product on both sides. A row where
     * either side is non-finite is treated as "invalid input" (NaN-poisoned)
     * and skipped from the rel_l2 sum. The verifier still notices real
     * divergence on the rows that do produce finite outputs — there are
     * plenty (typically ~95%+ of rows on a 64-row sample). */
    double num = 0.0, den = 0.0, max_abs = 0.0;
    int n_valid = 0;
    for (int i = 0; i < n_rows; i++) {
        float h = h_dst_hip[i], rf = h_dst_ref[i];
        int ok = !(h != h) && !(rf != rf) &&
                 (h - h == 0.0f) && (rf - rf == 0.0f);
        if (!ok) continue;
        n_valid++;
        double diff = (double)h - (double)rf;
        double ad = diff < 0 ? -diff : diff;
        if (ad > max_abs) max_abs = ad;
        num += diff * diff;
        den += (double)rf * (double)rf;
    }
    double rel_l2 = (n_valid == 0) ? 1.0
                                   : ((den > 1e-30) ? sqrt(num / den) : sqrt(num));
    if (out_rel_l2)  *out_rel_l2 = rel_l2;
    if (out_max_abs) *out_max_abs = max_abs;

    hipFree(d_mat); hipFree(d_x); hipFree(d_dst);
    free(h_mat); free(h_x); free(h_dst_hip); free(h_dst_ref); free(h_row_f32);
    return 0;
}

int hip_llm_bench_quant_matvec(
        hip_llm_runner *r, int weight_type,
        int n_rows, int n_cols,
        int warmup, int iters,
        float *out_ms) {
    if (!r || n_rows <= 0 || n_cols <= 0 || iters <= 0 || !out_ms) return -1;

    int blk_elems = 0, blk_bytes = 0;
    if (quant_matvec_block_info(weight_type, &blk_elems, &blk_bytes) != 0) return -1;
    if ((n_cols % blk_elems) != 0) return -1;
    int n_blk_per_row = n_cols / blk_elems;
    size_t row_bytes = (size_t)n_blk_per_row * blk_bytes;
    size_t mat_bytes = (size_t)n_rows * row_bytes;
    size_t x_bytes = (size_t)n_cols * sizeof(float);
    size_t dst_bytes = (size_t)n_rows * sizeof(float);

    unsigned char *h_mat = (unsigned char *)malloc(mat_bytes);
    float *h_x = (float *)malloc(x_bytes);
    if (!h_mat || !h_x) {
        free(h_mat); free(h_x);
        return -2;
    }
    fill_quant_matvec_inputs(h_mat, mat_bytes, h_x, n_cols, weight_type);

    void *d_mat = NULL, *d_x = NULL, *d_dst = NULL;
    hipEvent_t ev_start = NULL, ev_stop = NULL;
    int rc = -2;
    if (hipMalloc(&d_mat, mat_bytes) != hipSuccess ||
        hipMalloc(&d_x,   x_bytes)   != hipSuccess ||
        hipMalloc(&d_dst, dst_bytes) != hipSuccess) {
        goto done;
    }
    if (hipMemcpy(d_mat, h_mat, mat_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_x, h_x, x_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        goto done;
    }
    if (hipEventCreate(&ev_start) != hipSuccess ||
        hipEventCreate(&ev_stop)  != hipSuccess) {
        goto done;
    }

    for (int i = 0; i < warmup; i++) {
        launch_matvec_auto(r, d_dst, d_mat, d_x, n_rows, n_cols, weight_type);
    }
    if (hipStreamSynchronize(r->stream) != hipSuccess) goto done;

    if (hipEventRecord(ev_start, r->stream) != hipSuccess) goto done;
    for (int i = 0; i < iters; i++) {
        launch_matvec_auto(r, d_dst, d_mat, d_x, n_rows, n_cols, weight_type);
    }
    if (hipEventRecord(ev_stop, r->stream) != hipSuccess) goto done;
    if (hipEventSynchronize(ev_stop) != hipSuccess) goto done;

    float elapsed = 0.0f;
    if (hipEventElapsedTime(&elapsed, ev_start, ev_stop) != hipSuccess) goto done;
    *out_ms = elapsed / (float)iters;
    rc = 0;

done:
    if (ev_start) hipEventDestroy(ev_start);
    if (ev_stop) hipEventDestroy(ev_stop);
    if (d_mat) hipFree(d_mat);
    if (d_x) hipFree(d_x);
    if (d_dst) hipFree(d_dst);
    free(h_mat);
    free(h_x);
    return rc;
}

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
