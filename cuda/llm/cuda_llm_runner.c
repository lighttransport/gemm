/*
 * cuda_llm_runner.c - CUDA LLM inference via NVRTC-compiled kernels
 *
 * Compiles with plain gcc (no nvcc). Uses cuew for dynamic CUDA/NVRTC loading.
 * Supports F16 and Q8_0 weights on GPU, F32 compute.
 * Q8_0 uses padded 36-byte blocks (2B scale + 2B pad + 32B INT8) for aligned access.
 * Single-stream sequential kernel launches.
 */

#include "cuda_llm_runner.h"
#include "../cuew.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* transformer.h header-only: gives us qtensor type + dequant declarations */
#include "../../common/ggml_dequant.h"
#include "../../common/transformer.h"

/* ======================================================================== */
/* CUDA C kernel source (compiled at runtime via NVRTC)                     */
/* ======================================================================== */

static const char *cuda_kernel_source =
"/* Minimal FP16 support — no CUDA SDK headers needed. */\n"
"/* We treat FP16 as unsigned short and convert via inline PTX. */\n"
"typedef unsigned short half_raw;\n"
"\n"
"__device__ __forceinline__ float half_to_float(half_raw h) {\n"
"    float f;\n"
"    asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h));\n"
"    return f;\n"
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
"/* ---- 3. matvec_f16_f32: F16 matrix × F32 vector -> F32 ---- */\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"/* When partial rotation (n_rope_pairs < half_dim), pair offset = n_rope_pairs, */\n"
"/* NOT half_dim. For Qwen3.5: n_rope_pairs=32 -> rotate 32 pairs, NeoX pairing */\n"
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
"    /* Use rope_dim_count (2*n_rope_pairs) as freq denominator, not head_dim.\n"
"       For models where rope_dim < head_dim (e.g. Qwen3.5: 64 vs 256). */\n"
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
"        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);\n"
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
"        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"/* ---- dp4a: INT8 dot product via inline PTX (sm_61+) ---- */\n"
"__device__ __forceinline__ int dp4a_s8(int a, int b, int c) {\n"
"    asm(\"dp4a.s32.s32 %0, %1, %2, %0;\" : \"+r\"(c) : \"r\"(a), \"r\"(b));\n"
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
"        int iv = __float2int_rn(fv);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        /* Note: qhbits extracted inline below */\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"/* conv_state: [(conv_k-1) * qkv_dim], weight: [qkv_dim * conv_k] (row=channel, col=filter) */\n"
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
"/* One block per head. Shared mem reduction. */\n"
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
"/* ---- 23. repeat_tile_f32: expand n_group heads to dt_rank heads (tiling) ---- */\n"
"/* dst[h*d + i] = src[(h % n_group)*d + i] for h in [0,dt_rank), i in [0,d) */\n"
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
"/* ---- 24. deltanet_step_f32: Delta-Net autoregressive step ---- */\n"
"/* One block per head (blockIdx.x = head), d_state threads per block.\n"
"   Each thread handles one row of the d_state x d_state state matrix.\n"
"   State is stored as S^T (transposed), matching llama.cpp convention:\n"
"     sk = S^T @ k, output = S^T @ q, update: S^T += outer(delta, k). */\n"
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
"    /* 1. Decay */\n"
"    for (int c = 0; c < d_state; c++) S[c] *= decay;\n"
"    /* 2. sk = S^T @ k (row dot product in S^T storage) */\n"
"    float sk = 0.0f;\n"
"    for (int c = 0; c < d_state; c++) sk += S[c] * k[c];\n"
"    /* 3. delta = (v - sk) * beta (scalar for this row/thread) */\n"
"    float delta = (v_r - sk) * b;\n"
"    /* 4. Rank-1 update: S^T[r,c] += delta[r] * k[c] = outer(delta, k) */\n"
"    for (int c = 0; c < d_state; c++) S[c] += delta * k[c];\n"
"    /* 5. Output: o = S^T @ q / sqrt(d_state) */\n"
"    float scale = rsqrtf((float)d_state);\n"
"    float o = 0.0f;\n"
"    for (int c = 0; c < d_state; c++) o += S[c] * q[c];\n"
"    out[h * d_state + r] = o * scale;\n"
"}\n"
"\n"
"/* ---- 25. gated_rmsnorm_silu_f32: out = rmsnorm(out, w) * silu(z), per head ---- */\n"
"/* One block per head. norm_w has [d_state] elements (shared across heads). */\n"
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
"/* ---- 26. sigmoid_mul_f32: data[i] *= sigmoid(gate[i]) ---- */\n"
"__global__ void sigmoid_mul_f32(float *data, const float *gate, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) data[i] *= 1.0f / (1.0f + expf(-gate[i]));\n"
"}\n"
"\n"
"/* ---- 27. deinterleave_qgate_f32: split [Q0,g0,Q1,g1,...] -> Q[], gate[] ---- */\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_iq2[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_iq2[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_iq2[w];\n"
"        dst[row] = total;\n"
"    }\n"
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

/* ---- NEW QUANTIZATION TYPE KERNELS ---- */

"/* ---- matvec_q4_0_f32: Q4_0 matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_q4_0_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 32;\n"
"    int row_bytes = nb * 18;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 18;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 16; j++) {\n"
"            float v0 = (float)((int)(qs[j] & 0x0F) - 8) * d;\n"
"            float v1 = (float)((int)(qs[j] >>    4) - 8) * d;\n"
"            partial += v0 * xb[j] + v1 * xb[j + 16];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_q4_0[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_q4_0[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_q4_0[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_q4_1_f32: Q4_1 matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_q4_1_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 32;\n"
"    int row_bytes = nb * 20;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 20;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        float m = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *qs = bp + 4;\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 16; j++) {\n"
"            float v0 = (float)(qs[j] & 0x0F) * d + m;\n"
"            float v1 = (float)(qs[j] >>    4) * d + m;\n"
"            partial += v0 * xb[j] + v1 * xb[j + 16];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_q4_1[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_q4_1[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_q4_1[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_q5_0_f32: Q5_0 matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_q5_0_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 32;\n"
"    int row_bytes = nb * 22;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 22;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned int qh;\n"
"        memcpy(&qh, bp + 2, 4);\n"
"        const unsigned char *qs = bp + 6;\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 16; j++) {\n"
"            unsigned char xh_0 = ((qh >> (j +  0)) << 4) & 0x10;\n"
"            unsigned char xh_1 = ((qh >> (j + 12))     ) & 0x10;\n"
"            int x0 = ((int)(qs[j] & 0x0F) | xh_0) - 16;\n"
"            int x1 = ((int)(qs[j] >>    4) | xh_1) - 16;\n"
"            partial += (float)x0 * d * xb[j] + (float)x1 * d * xb[j + 16];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_q5_0[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_q5_0[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_q5_0[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_q5_1_f32: Q5_1 matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_q5_1_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 32;\n"
"    int row_bytes = nb * 24;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 24;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        float m = half_to_float(*(const half_raw *)(bp + 2));\n"
"        unsigned int qh;\n"
"        memcpy(&qh, bp + 4, 4);\n"
"        const unsigned char *qs = bp + 8;\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 16; j++) {\n"
"            unsigned char xh_0 = ((qh >> (j +  0)) << 4) & 0x10;\n"
"            unsigned char xh_1 = ((qh >> (j + 12))     ) & 0x10;\n"
"            float v0 = (float)((int)(qs[j] & 0x0F) | xh_0) * d + m;\n"
"            float v1 = (float)((int)(qs[j] >>    4) | xh_1) * d + m;\n"
"            partial += v0 * xb[j] + v1 * xb[j + 16];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_q5_1[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_q5_1[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_q5_1[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- Device LUT for IQ4_NL ---- */\n"
"__device__ static const signed char kvalues_iq4nl_dev[16] = {\n"
"    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113\n"
"};\n"
"\n"
"/* ---- matvec_iq4_nl_f32: IQ4_NL matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq4_nl_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                    int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 32;\n"
"    int row_bytes = nb * 18;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_iq4nl[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_iq4nl[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_iq4nl[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_iq4_xs_f32: IQ4_XS matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq4_xs_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                    int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 136;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_iq4xs[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_iq4xs[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_iq4xs[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_iq2_xs_f32: IQ2_XS matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq2_xs_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                    int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 74;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_iq2xs[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_iq2xs[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_iq2xs[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_iq3_xxs_f32: IQ3_XXS matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq3_xxs_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                     int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *scales_and_signs = qs + 64;  /* 32 bytes */\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            unsigned int aux32;\n"
"            memcpy(&aux32, scales_and_signs + 4*ib32, 4);\n"
"            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                unsigned char signs = ksigns_iq2xs_dev[(aux32 >> (7*l)) & 127];\n"
"                const unsigned char *grid1 = (const unsigned char *)&iq3xxs_grid_dev[qs[2*l+0]];\n"
"                const unsigned char *grid2 = (const unsigned char *)&iq3xxs_grid_dev[qs[2*l+1]];\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_iq3xxs[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_iq3xxs[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_iq3xxs[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_iq2_s_f32: IQ2_S matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq2_s_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 82;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_iq2s[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_iq2s[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_iq2s[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"

"/* ---- matvec_iq3_s_f32: IQ3_S matrix x F32 vector -> F32 ---- */\n"
"__global__ void matvec_iq3_s_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_iq3s[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_iq3s[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_iq3s[w];\n"
"        dst[row] = total;\n"
"    }\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"} /* extern \"C\" */\n"
;

/* ======================================================================== */
/* Error checking macros                                                    */
/* ======================================================================== */

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *errStr = "unknown"; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, errStr, (int)err); \
        return -1; \
    } \
} while(0)

#define CHECK_CU_NULL(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *errStr = "unknown"; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, errStr, (int)err); \
        return NULL; \
    } \
} while(0)

#define CHECK_CU_VOID(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *errStr = "unknown"; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, errStr, (int)err); \
        return; \
    } \
} while(0)

/* ======================================================================== */
/* Runner state                                                             */
/* ======================================================================== */

/* Per-layer GPU weight pointers */
typedef struct {
    CUdeviceptr attn_norm_w;   /* F32 [n_embd] */
    CUdeviceptr attn_q_w;     /* F16 [n_rows * n_cols] */
    CUdeviceptr attn_k_w;
    CUdeviceptr attn_v_w;
    CUdeviceptr attn_q_norm_w; /* F32 [head_dim] */
    CUdeviceptr attn_k_norm_w; /* F32 [head_dim] */
    CUdeviceptr attn_output_w; /* F16 */
    CUdeviceptr ffn_norm_w;    /* F32 [n_embd] */
    CUdeviceptr ffn_gate_w;    /* F16 */
    CUdeviceptr ffn_up_w;      /* F16 */
    CUdeviceptr ffn_down_w;    /* F16 */

    /* Dimensions for matvec */
    int attn_q_rows, attn_q_cols;
    int attn_k_rows, attn_k_cols;
    int attn_v_rows, attn_v_cols;
    int attn_output_rows, attn_output_cols;
    int ffn_gate_rows, ffn_gate_cols;
    int ffn_up_rows, ffn_up_cols;
    int ffn_down_rows, ffn_down_cols;

    int has_qk_norm;

    /* Weight types (GGML_TYPE_F16=1 or GGML_TYPE_Q8_0=8 or K-quant) */
    int attn_q_type, attn_k_type, attn_v_type, attn_output_type;
    int ffn_gate_type, ffn_up_type, ffn_down_type;

    /* MoE fields — only used when is_moe=1 */
    int is_moe;
    CUdeviceptr moe_gate_w;         /* F32 [n_experts, n_embd] — router */
    int moe_gate_rows, moe_gate_cols;

    CUdeviceptr moe_gate_exps_w;    /* K-quant [expert_ff, n_embd, n_experts] — 3D packed */
    CUdeviceptr moe_up_exps_w;      /* K-quant [expert_ff, n_embd, n_experts] */
    CUdeviceptr moe_down_exps_w;    /* K-quant [n_embd, expert_ff, n_experts] */
    int moe_gate_exps_type, moe_up_exps_type, moe_down_exps_type;
    int moe_exp_rows_gu, moe_exp_cols_gu;   /* per-expert dims for gate/up: [expert_ff, n_embd] */
    int moe_exp_rows_d, moe_exp_cols_d;     /* per-expert dims for down: [n_embd, expert_ff] */
    size_t moe_exp_stride_gu;       /* byte stride between experts in gate/up tensors */
    size_t moe_exp_stride_d;        /* byte stride between experts in down tensor */

    CUdeviceptr moe_shared_gate_w;      /* F32 [n_embd] — shared expert sigmoid gate */
    CUdeviceptr moe_shared_ffn_gate_w;  /* F16/BF16 [shared_ff, n_embd] — shared expert */
    CUdeviceptr moe_shared_ffn_up_w;    /* F16/BF16 [shared_ff, n_embd] */
    CUdeviceptr moe_shared_ffn_down_w;  /* F16/BF16 [n_embd, shared_ff] */
    int moe_shared_gate_type, moe_shared_up_type, moe_shared_down_type;
    int moe_shared_gate_rows, moe_shared_gate_cols;
    int moe_shared_up_rows, moe_shared_up_cols;
    int moe_shared_down_rows, moe_shared_down_cols;

    /* SSM (Delta-Net) layer fields — only used when is_ssm=1 */
    int is_ssm;
    CUdeviceptr ssm_qkv_w;      /* [qkv_dim, n_embd] quantized */
    CUdeviceptr ssm_gate_w;     /* [d_inner, n_embd] quantized */
    CUdeviceptr ssm_alpha_w;    /* [dt_rank, n_embd] quantized */
    CUdeviceptr ssm_beta_w;     /* [dt_rank, n_embd] quantized */
    CUdeviceptr ssm_out_w;      /* [n_embd, d_inner] quantized */
    CUdeviceptr ssm_a;          /* [dt_rank] F32 */
    CUdeviceptr ssm_dt_bias;    /* [dt_rank] F32 */
    CUdeviceptr ssm_conv1d_w;   /* [qkv_dim * conv_k] F32 */
    CUdeviceptr ssm_norm_w;     /* [d_state] F32 */
    int ssm_qkv_type, ssm_gate_type, ssm_alpha_type, ssm_beta_type, ssm_out_type;
    int ssm_qkv_rows, ssm_qkv_cols;
    int ssm_gate_rows, ssm_gate_cols;
    int ssm_alpha_rows, ssm_alpha_cols;
    int ssm_beta_rows, ssm_beta_cols;
    int ssm_out_rows, ssm_out_cols;

    /* SSM persistent state */
    CUdeviceptr d_conv_state;    /* [(conv_k-1) * qkv_dim] F32 */
    CUdeviceptr d_recurrent_state; /* [dt_rank * d_state * d_state] F32 */
} cuda_layer;

struct cuda_llm_runner {
    /* CUDA context */
    CUdevice device;
    CUcontext context;
    CUstream stream;
    int verbose;

    /* Compiled module + kernels */
    CUmodule module;
    CUfunction fn_embed_f16;
    CUfunction fn_rmsnorm_f32;
    CUfunction fn_matvec_f16_f32;
    CUfunction fn_qknorm_f32;
    CUfunction fn_rope_neox_f32;
    CUfunction fn_kv_cache_store;
    CUfunction fn_attn_decode_f32;
    CUfunction fn_silu_mul_f32;
    CUfunction fn_add_f32;
    CUfunction fn_quantize_f32_to_int8;
    CUfunction fn_matvec_q8_0_dp4a;
    CUfunction fn_matvec_q8_0_f32;
    CUfunction fn_embed_q8_0;
    CUfunction fn_matvec_q2_K_f32;
    CUfunction fn_matvec_q3_K_f32;
    CUfunction fn_matvec_q4_K_f32;
    CUfunction fn_matvec_q5_K_f32;
    CUfunction fn_matvec_q6_K_f32;
    CUfunction fn_embed_q2_K;
    /* SSM kernels */
    CUfunction fn_softplus_mul_f32;
    CUfunction fn_sigmoid_inplace_f32;
    CUfunction fn_conv1d_depthwise_silu_f32;
    CUfunction fn_l2_norm_heads_f32;
    CUfunction fn_repeat_tile_f32;
    CUfunction fn_deltanet_step_f32;
    CUfunction fn_gated_rmsnorm_silu_f32;
    CUfunction fn_sigmoid_mul_f32;
    CUfunction fn_deinterleave_qgate_f32;
    /* MoE kernels */
    CUfunction fn_scale_add_f32;
    CUfunction fn_matvec_f32_f32;
    CUfunction fn_matvec_iq2_xxs_f32;
    CUfunction fn_matvec_q4_0_f32;
    CUfunction fn_matvec_q4_1_f32;
    CUfunction fn_matvec_q5_0_f32;
    CUfunction fn_matvec_q5_1_f32;
    CUfunction fn_matvec_iq4_nl_f32;
    CUfunction fn_matvec_iq4_xs_f32;
    CUfunction fn_matvec_iq2_xs_f32;
    CUfunction fn_matvec_iq3_xxs_f32;
    CUfunction fn_matvec_iq2_s_f32;
    CUfunction fn_matvec_iq3_s_f32;
    CUfunction fn_matvec_iq1_s_f32;
    CUfunction fn_matvec_iq1_m_f32;
    CUfunction fn_matvec_tq1_0_f32;
    CUfunction fn_matvec_tq2_0_f32;

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
    int n_rope_pairs;       /* number of dimension pairs to rotate (0 = all) */
    float rms_norm_eps;
    int debug_layers;       /* if > 0, print hidden state norm after each layer */
    int max_layers;         /* if > 0, process only first N layers (debug) */
    int n_deepstack;        /* number of deepstack layers (VLM injection, 0 = none) */

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
    int n_experts_used;   /* top-k */
    int expert_ff;        /* per-expert FFN dim */
    int shared_expert_ff; /* shared expert FFN dim */

    /* MoE scratch buffers */
    CUdeviceptr d_router_logits;  /* [n_experts] F32 */
    CUdeviceptr d_moe_accum;      /* [n_embd] F32 */
    float *h_router_logits;       /* host copy for top-k selection */

    /* GPU weights */
    int token_embd_type;        /* GGML_TYPE_F16 or GGML_TYPE_Q8_0 */
    CUdeviceptr d_token_embd;   /* F16 or Q8_0 [n_vocab * n_embd] */
    CUdeviceptr d_output_norm;  /* F32 [n_embd] */
    CUdeviceptr d_output_w;    /* lm_head [n_vocab * n_embd] */
    int output_w_type;
    int has_lm_head;
    cuda_layer *layers;

    /* KV cache: one allocation per layer */
    CUdeviceptr *d_key_cache;    /* [n_layers] -> [max_seq_len * kv_dim] F32 */
    CUdeviceptr *d_value_cache;

    /* Scratch buffers on GPU */
    CUdeviceptr d_x;     /* [n_embd] */
    CUdeviceptr d_xb;    /* [max(n_embd, n_heads*head_dim)] */
    CUdeviceptr d_xb2;   /* [max(n_embd, n_heads*head_dim)] */
    CUdeviceptr d_q;     /* [n_heads * head_dim] */
    CUdeviceptr d_k;     /* [n_kv_heads * head_dim] */
    CUdeviceptr d_v;     /* [n_kv_heads * head_dim] */
    CUdeviceptr d_gate;  /* [n_ff] */
    CUdeviceptr d_up;    /* [n_ff] */

    /* SSM scratch buffers */
    CUdeviceptr d_ssm_qkv;    /* [qkv_dim] conv input/output */
    CUdeviceptr d_ssm_z;      /* [d_inner] gate projection */
    CUdeviceptr d_ssm_alpha;  /* [dt_rank] */
    CUdeviceptr d_ssm_beta;   /* [dt_rank] */
    CUdeviceptr d_ssm_Q_exp;  /* [dt_rank * d_state] expanded Q */
    CUdeviceptr d_ssm_K_exp;  /* [dt_rank * d_state] expanded K */
    CUdeviceptr d_ssm_out;    /* [d_inner] delta-net output */
    CUdeviceptr d_ssm_conv_out; /* [qkv_dim] conv1d output */
    CUdeviceptr d_attn_gate;  /* [q_dim] for gated attention gate */

    /* Deepstack scratch (for VLM embedding injection) */
    CUdeviceptr d_ds_tmp;   /* [n_embd] for deepstack slice upload */

    /* INT8 quantization scratch (for dp4a path) */
    CUdeviceptr d_xb_q;     /* INT8 [max_dim] */
    CUdeviceptr d_xb_scale; /* F32 [1] */

    /* Host output buffer */
    float *h_output;     /* [n_embd] or [n_vocab] for logits */
    CUdeviceptr d_logits; /* [n_vocab] GPU logits buffer */

    /* Weight loading state */
    int weights_loaded;

    /* Deepstack injection state (set during forward_embd, NULL otherwise) */
    const float *_ds_embd;
    int _ds_embd_stride;
};

/* ======================================================================== */
/* NVRTC kernel compilation                                                 */
/* ======================================================================== */

static int compile_kernels(cuda_llm_runner *r) {
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, r->device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, r->device);
    int sm = major * 10 + minor;

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: compiling kernels for sm_%d via NVRTC...\n", sm);
    }

    nvrtcProgram prog;
    nvrtcResult nres = nvrtcCreateProgram(&prog, cuda_kernel_source, "llm_kernels.cu",
                                          0, NULL, NULL);
    if (nres != NVRTC_SUCCESS) {
        fprintf(stderr, "cuda_llm: nvrtcCreateProgram failed: %d\n", (int)nres);
        return -1;
    }

    char arch_opt[32];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=sm_%d", sm);
    const char *opts[] = { arch_opt, "--use_fast_math" };
    nres = nvrtcCompileProgram(prog, 2, opts);

    if (nres != NVRTC_SUCCESS) {
        /* Print compilation log */
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char *log = (char *)malloc(log_size);
            nvrtcGetProgramLog(prog, log);
            fprintf(stderr, "cuda_llm: NVRTC compilation log:\n%s\n", log);
            free(log);
        }
        nvrtcDestroyProgram(&prog);
        return -1;
    }

    if (r->verbose >= 1) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char *log = (char *)malloc(log_size);
            nvrtcGetProgramLog(prog, log);
            fprintf(stderr, "cuda_llm: NVRTC log: %s\n", log);
            free(log);
        }
    }

    /* Get PTX */
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char *ptx = (char *)malloc(ptx_size);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    if (r->verbose >= 2) {
        fprintf(stderr, "cuda_llm: PTX size = %zu bytes\n", ptx_size);
    }

    /* Load module from PTX */
    CUresult err = cuModuleLoadDataEx(&r->module, ptx, 0, NULL, NULL);
    free(ptx);
    if (err != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "cuda_llm: cuModuleLoadDataEx failed: %s\n", errStr);
        return -1;
    }

    /* Look up kernel functions */
    #define GET_FUNC(name) do { \
        err = cuModuleGetFunction(&r->fn_##name, r->module, #name); \
        if (err != CUDA_SUCCESS) { \
            fprintf(stderr, "cuda_llm: kernel '%s' not found\n", #name); \
            return -1; \
        } \
    } while(0)

    GET_FUNC(embed_f16);
    GET_FUNC(rmsnorm_f32);
    GET_FUNC(matvec_f16_f32);
    GET_FUNC(qknorm_f32);
    GET_FUNC(rope_neox_f32);
    GET_FUNC(kv_cache_store);
    GET_FUNC(attn_decode_f32);
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

    #undef GET_FUNC

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: all %d kernels compiled successfully\n", 31 + 14);
    }
    return 0;
}

/* ======================================================================== */
/* Public API: init                                                         */
/* ======================================================================== */

cuda_llm_runner *cuda_llm_init(int device_id, int verbose) {
    /* Initialize CUDA + NVRTC via cuew */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuda_llm: cuewInit failed (CUDA/NVRTC libraries not found)\n");
        return NULL;
    }

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_llm: cuInit failed\n");
        return NULL;
    }

    cuda_llm_runner *r = (cuda_llm_runner *)calloc(1, sizeof(cuda_llm_runner));
    if (!r) return NULL;
    r->verbose = verbose;

    CHECK_CU_NULL(cuDeviceGet(&r->device, device_id));
    CHECK_CU_NULL(cuCtxCreate(&r->context, 0, r->device));
    CHECK_CU_NULL(cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING));

    if (verbose >= 1) {
        char name[256];
        cuDeviceGetName(name, sizeof(name), r->device);
        int major, minor;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, r->device);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, r->device);
        size_t mem_total;
        cuDeviceTotalMem(&mem_total, r->device);
        fprintf(stderr, "cuda_llm: device %d: %s (sm_%d%d, %.1f GB VRAM)\n",
                device_id, name, major, minor, (double)mem_total / (1024.0*1024.0*1024.0));
    }

    /* Compile kernels */
    if (compile_kernels(r) != 0) {
        cuStreamDestroy(r->stream);
        cuCtxDestroy(r->context);
        free(r);
        return NULL;
    }

    return r;
}

/* ======================================================================== */
/* Weight upload helpers                                                    */
/* ======================================================================== */

/* Upload a qtensor as F16 to GPU (for F16 weights, direct copy; others dequant to F32 then... */
/* For this model (Qwen3-Embedding-0.6B-f16), all weight matrices are F16. */
/* Norms are F32. We handle both cases. */

/* Upload F16 tensor data directly to GPU */
static int upload_f16_matrix(CUdeviceptr *d_ptr, const qtensor *t) {
    if (!t->data) { *d_ptr = 0; return 0; }
    size_t nbytes = (size_t)t->n_rows * t->n_cols * sizeof(uint16_t);  /* F16 = 2 bytes */
    CUresult err = cuMemAlloc(d_ptr, nbytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_f16_matrix alloc failed (%zu bytes, err=%d)\n", nbytes, (int)err); return -1; }
    err = cuMemcpyHtoD(*d_ptr, t->data, nbytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_f16_matrix copy failed\n"); cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}

/* Dequant a 1D norm tensor to F32 and upload */
static int upload_norm_f32(CUdeviceptr *d_ptr, const qtensor *t, int n) {
    if (!t->data) { *d_ptr = 0; return 0; }
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) { fprintf(stderr, "cuda_llm: upload_norm_f32 malloc failed (n=%d)\n", n); return -1; }
    dequant_row(t->type, t->data, buf, n);
    CUresult err = cuMemAlloc(d_ptr, n * sizeof(float));
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_norm_f32 alloc failed (n=%d, err=%d)\n", n, (int)err); free(buf); return -1; }
    err = cuMemcpyHtoD(*d_ptr, buf, n * sizeof(float));
    free(buf);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_norm_f32 copy failed\n"); cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}

/* Upload Q8_0 tensor data to GPU with padding for alignment.
 * Each 34-byte Q8_0 block (2B scale + 32B qs) is padded to 36 bytes
 * (2B scale + 2B pad + 32B qs) so int32 reads of qs data are 4-byte aligned. */
static int upload_q8_0_raw(CUdeviceptr *d_ptr, const qtensor *t) {
    if (!t->data) { *d_ptr = 0; return 0; }
    int n_elements = t->n_rows * t->n_cols;
    int n_blocks = n_elements / 32;
    size_t nbytes_padded = (size_t)n_blocks * 36;  /* 36 bytes per padded block */

    /* Repack on host: insert 2-byte padding after each scale */
    uint8_t *padded = (uint8_t *)malloc(nbytes_padded);
    if (!padded) return -1;
    const uint8_t *src = (const uint8_t *)t->data;
    for (int i = 0; i < n_blocks; i++) {
        uint8_t *dst = padded + (size_t)i * 36;
        const uint8_t *s = src + (size_t)i * 34;
        dst[0] = s[0];  /* scale low byte */
        dst[1] = s[1];  /* scale high byte */
        dst[2] = 0;     /* padding */
        dst[3] = 0;     /* padding */
        memcpy(dst + 4, s + 2, 32);  /* 32 INT8 values */
    }

    CUresult err = cuMemAlloc(d_ptr, nbytes_padded);
    if (err != CUDA_SUCCESS) { free(padded); return -1; }
    err = cuMemcpyHtoD(*d_ptr, padded, nbytes_padded);
    free(padded);
    if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}

/* Upload a K-quant tensor directly to GPU (no repack needed — already aligned) */
static int upload_kquant_raw(CUdeviceptr *d_ptr, const qtensor *t) {
    if (!t->data) { *d_ptr = 0; return 0; }
    size_t nbytes = dequant_row_size(t->type, t->n_cols) * (size_t)t->n_rows;
    CUresult err = cuMemAlloc(d_ptr, nbytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_kquant_raw alloc failed (%zu bytes, type=%d, err=%d)\n", nbytes, t->type, (int)err); return -1; }
    err = cuMemcpyHtoD(*d_ptr, t->data, nbytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_kquant_raw copy failed\n"); cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}

/* F32 → F16 conversion (truncation, no rounding) */
static uint16_t cllm_f32_to_f16(float f) {
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

/* Upload a weight matrix - dispatches based on type */
static int upload_weight_matrix(CUdeviceptr *d_ptr, const qtensor *t, int *out_type) {
    *out_type = t->type;
    if (t->type == GGML_TYPE_Q8_0) {
        return upload_q8_0_raw(d_ptr, t);
    } else if (t->type == GGML_TYPE_Q2_K || t->type == GGML_TYPE_Q3_K ||
               t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K ||
               t->type == GGML_TYPE_Q6_K) {
        return upload_kquant_raw(d_ptr, t);
    } else if (t->type == GGML_TYPE_BF16) {
        /* BF16 → F32 → F16, then upload as F16 */
        *out_type = GGML_TYPE_F16;
        int n_elements = t->n_rows * t->n_cols;
        float *f32_buf = (float *)malloc((size_t)n_elements * sizeof(float));
        if (!f32_buf) return -1;
        dequant_row(GGML_TYPE_BF16, t->data, f32_buf, n_elements);
        uint16_t *f16_buf = (uint16_t *)malloc((size_t)n_elements * sizeof(uint16_t));
        if (!f16_buf) { free(f32_buf); return -1; }
        for (int i = 0; i < n_elements; i++) f16_buf[i] = cllm_f32_to_f16(f32_buf[i]);
        free(f32_buf);
        size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
        CUresult err = cuMemAlloc(d_ptr, nbytes);
        if (err != CUDA_SUCCESS) { free(f16_buf); return -1; }
        err = cuMemcpyHtoD(*d_ptr, f16_buf, nbytes);
        free(f16_buf);
        if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
        return 0;
    } else {
        /* Default: F16 (or treat as F16) */
        fprintf(stderr, "cuda_llm: upload_weight_matrix: unhandled type %d, treating as F16 (rows=%d, cols=%d)\n", t->type, t->n_rows, t->n_cols);
        return upload_f16_matrix(d_ptr, t);
    }
}

/* Upload a 3D K-quant tensor (stacked experts) directly to GPU.
 * Returns per-expert byte stride via out_stride. */
static int upload_3d_kquant_raw(CUdeviceptr *d_ptr, const qtensor *t, size_t *out_stride) {
    if (!t->data) { *d_ptr = 0; return 0; }
    /* For 3D tensors: dims[0]=cols, dims[1]=rows_per_expert, dims[2]=n_experts
     * Total rows = dims[1] * dims[2] (already computed in t->n_rows by cllm_load_tensor) */
    size_t row_bytes = dequant_row_size(t->type, t->n_cols);
    int rows_per_expert = (t->n_dims >= 3) ? (int)t->dims[1] : t->n_rows;
    *out_stride = row_bytes * (size_t)rows_per_expert;
    size_t total_bytes = row_bytes * (size_t)t->n_rows;
    CUresult err = cuMemAlloc(d_ptr, total_bytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_3d_kquant alloc failed (%zu bytes, type=%d, err=%d)\n", total_bytes, t->type, (int)err); return -1; }
    err = cuMemcpyHtoD(*d_ptr, t->data, total_bytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_3d_kquant copy failed\n"); cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}

/* ======================================================================== */
/* Helper: GGUF metadata readers (duplicated from transformer.h for standalone) */
/* ======================================================================== */

static int cllm_get_int(const gguf_context *gguf, const char *key, int def) {
    int idx = gguf_find_key(gguf, key);
    if (idx < 0) return def;
    if (gguf->kv[idx].type == GGUF_TYPE_UINT32) return (int)gguf->kv[idx].value.u32;
    if (gguf->kv[idx].type == GGUF_TYPE_INT32)  return gguf->kv[idx].value.i32;
    return def;
}

static float cllm_get_float(const gguf_context *gguf, const char *key, float def) {
    int idx = gguf_find_key(gguf, key);
    if (idx < 0) return def;
    if (gguf->kv[idx].type == GGUF_TYPE_FLOAT32) return gguf->kv[idx].value.f32;
    return def;
}

static int cllm_find_tensor(const gguf_context *gguf, const char *name) {
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

static qtensor cllm_load_tensor(const gguf_context *gguf, const char *name, int required) {
    qtensor t = {0};
    int idx = cllm_find_tensor(gguf, name);
    if (idx < 0) {
        if (required) fprintf(stderr, "cuda_llm: missing tensor '%s'\n", name);
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
/* Public API: load_weights                                                 */
/* ======================================================================== */

int cuda_llm_load_weights(cuda_llm_runner *r, gguf_context *gguf, int max_seq_len) {
    if (!r || !gguf) return -1;

    /* Detect architecture prefix */
    const char *arch = "qwen2";
    if (gguf_find_key(gguf, "qwen35moe.block_count") >= 0) arch = "qwen35moe";
    else if (gguf_find_key(gguf, "qwen35.block_count") >= 0) arch = "qwen35";
    else if (gguf_find_key(gguf, "qwen3.block_count") >= 0) arch = "qwen3";
    else if (gguf_find_key(gguf, "qwen3vl.block_count") >= 0) arch = "qwen3vl";

    char kbuf[128];
    #define ARCH_KEY(suffix) (snprintf(kbuf, sizeof(kbuf), "%s." suffix, arch), kbuf)

    r->n_embd      = cllm_get_int(gguf, ARCH_KEY("embedding_length"), 4096);
    r->n_heads     = cllm_get_int(gguf, ARCH_KEY("attention.head_count"), 32);
    r->n_kv_heads  = cllm_get_int(gguf, ARCH_KEY("attention.head_count_kv"), 8);
    r->n_layers    = cllm_get_int(gguf, ARCH_KEY("block_count"), 36);
    r->n_ff        = cllm_get_int(gguf, ARCH_KEY("feed_forward_length"), 12288);
    r->n_vocab     = cllm_get_int(gguf, ARCH_KEY("vocab_size"), 0);
    r->rms_norm_eps = cllm_get_float(gguf, ARCH_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);
    r->rope_freq_base = cllm_get_float(gguf, ARCH_KEY("rope.freq_base"), 5000000.0f);
    r->head_dim    = cllm_get_int(gguf, ARCH_KEY("attention.key_length"), r->n_embd / r->n_heads);

    /* RoPE dimension count: only rotate this many dimensions (0 = all).
       NOTE: dimension_sections (e.g. Qwen3.5: [11,11,10,0]) is for MROPE position
       channel splitting, NOT for limiting which pairs get rotated. In llama.cpp,
       n_rot defaults to n_embd_head_k (=head_dim) when rope.dimension_count is absent.
       So n_rope_pairs=0 means "rotate all pairs with freq_denom=head_dim". */
    {
        int rope_dim_count = cllm_get_int(gguf, ARCH_KEY("rope.dimension_count"), 0);
        r->n_rope_pairs = (rope_dim_count > 0) ? rope_dim_count / 2 : 0;
    }

    int ctx_len = cllm_get_int(gguf, ARCH_KEY("context_length"), 0);
    if (max_seq_len <= 0) {
        max_seq_len = (ctx_len > 0) ? ctx_len : 1024;
    } else if (ctx_len > 0 && max_seq_len > ctx_len) {
        max_seq_len = ctx_len;
    }
    r->max_seq_len = max_seq_len;

    /* Hybrid SSM (Qwen3.5 / Qwen3.5-MoE) */
    r->is_hybrid = 0;
    r->is_moe = 0;
    r->full_attn_interval = 0;
    if (strcmp(arch, "qwen35") == 0 || strcmp(arch, "qwen35moe") == 0) {
        r->is_hybrid = 1;
        r->ssm_conv_kernel = cllm_get_int(gguf, ARCH_KEY("ssm.conv_kernel"), 4);
        r->ssm_d_state     = cllm_get_int(gguf, ARCH_KEY("ssm.state_size"), 128);
        r->ssm_n_group     = cllm_get_int(gguf, ARCH_KEY("ssm.group_count"), 16);
        r->ssm_dt_rank     = cllm_get_int(gguf, ARCH_KEY("ssm.time_step_rank"), 48);
        r->ssm_d_inner     = cllm_get_int(gguf, ARCH_KEY("ssm.inner_size"), 6144);
        r->full_attn_interval = cllm_get_int(gguf, ARCH_KEY("attention.full_attention_interval"), 4);
        r->ssm_qkv_dim = r->ssm_d_state * r->ssm_n_group * 2 + r->ssm_d_inner;
        if (r->verbose >= 1) {
            fprintf(stderr, "cuda_llm: hybrid SSM: conv_k=%d d_state=%d n_group=%d dt_rank=%d d_inner=%d interval=%d qkv_dim=%d\n",
                    r->ssm_conv_kernel, r->ssm_d_state, r->ssm_n_group, r->ssm_dt_rank,
                    r->ssm_d_inner, r->full_attn_interval, r->ssm_qkv_dim);
        }

        /* MoE params (Qwen3.5-MoE) */
        if (strcmp(arch, "qwen35moe") == 0) {
            r->is_moe = 1;
            r->n_experts       = cllm_get_int(gguf, ARCH_KEY("expert_count"), 256);
            r->n_experts_used  = cllm_get_int(gguf, ARCH_KEY("expert_used_count"), 8);
            r->expert_ff       = cllm_get_int(gguf, ARCH_KEY("expert_feed_forward_length"), 512);
            r->shared_expert_ff = cllm_get_int(gguf, ARCH_KEY("expert_shared_feed_forward_length"), 512);
            /* n_ff is not meaningful for MoE layers; set to expert_ff for scratch sizing */
            r->n_ff = r->expert_ff;
            if (r->verbose >= 1) {
                fprintf(stderr, "cuda_llm: MoE: n_experts=%d n_experts_used=%d expert_ff=%d shared_expert_ff=%d\n",
                        r->n_experts, r->n_experts_used, r->expert_ff, r->shared_expert_ff);
            }
        }
    }

    /* DeepStack layers (VLM injection) */
    r->n_deepstack = cllm_get_int(gguf, ARCH_KEY("n_deepstack_layers"), 0);
    if (r->verbose >= 1 && r->n_deepstack > 0) {
        fprintf(stderr, "cuda_llm: n_deepstack=%d\n", r->n_deepstack);
    }

    #undef ARCH_KEY

    int kv_dim = r->n_kv_heads * r->head_dim;
    int q_dim = r->n_heads * r->head_dim;

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: arch=%s n_embd=%d n_heads=%d n_kv_heads=%d n_layers=%d n_ff=%d head_dim=%d\n",
                arch, r->n_embd, r->n_heads, r->n_kv_heads, r->n_layers, r->n_ff, r->head_dim);
        fprintf(stderr, "cuda_llm: rope_freq_base=%.0f rms_norm_eps=%.1e max_seq_len=%d n_rope_pairs=%d\n",
                r->rope_freq_base, r->rms_norm_eps, r->max_seq_len, r->n_rope_pairs);
    }

    /* Token embeddings (F16 or Q8_0) */
    qtensor embd = cllm_load_tensor(gguf, "token_embd.weight", 1);
    if (!embd.data) return -1;
    if (r->n_vocab == 0) r->n_vocab = embd.n_rows;
    r->token_embd_type = embd.type;
    if (embd.type == GGML_TYPE_Q8_0) {
        if (upload_q8_0_raw(&r->d_token_embd, &embd) != 0) return -1;
    } else if (embd.type == GGML_TYPE_Q2_K || embd.type == GGML_TYPE_Q3_K ||
               embd.type == GGML_TYPE_Q4_K || embd.type == GGML_TYPE_Q6_K) {
        if (upload_kquant_raw(&r->d_token_embd, &embd) != 0) return -1;
    } else if (embd.type == GGML_TYPE_Q5_K) {
        /* Q5_K → dequant to F16 at load time (no Q5_K embed kernel) */
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
        for (int i = 0; i < n_elements; i++) f16_buf[i] = cllm_f32_to_f16(f32_buf[i]);
        free(f32_buf);
        size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
        CUresult err = cuMemAlloc(&r->d_token_embd, nbytes);
        if (err != CUDA_SUCCESS) { free(f16_buf); return -1; }
        err = cuMemcpyHtoD(r->d_token_embd, f16_buf, nbytes);
        free(f16_buf);
        if (err != CUDA_SUCCESS) { cuMemFree(r->d_token_embd); r->d_token_embd = 0; return -1; }
        r->token_embd_type = GGML_TYPE_F16;  /* now stored as F16 */
    } else {
        if (upload_f16_matrix(&r->d_token_embd, &embd) != 0) return -1;
    }
    if (r->verbose >= 1) {
        static const char *type_names[] = {
            [GGML_TYPE_F16] = "F16", [GGML_TYPE_Q8_0] = "Q8_0",
            [GGML_TYPE_Q2_K] = "Q2_K", [GGML_TYPE_Q3_K] = "Q3_K",
            [GGML_TYPE_Q4_K] = "Q4_K", [GGML_TYPE_Q5_K] = "Q5_K", [GGML_TYPE_Q6_K] = "Q6_K",
            [GGML_TYPE_IQ2_XXS] = "IQ2_XXS",
            [GGML_TYPE_Q4_0] = "Q4_0", [GGML_TYPE_Q4_1] = "Q4_1",
            [GGML_TYPE_Q5_0] = "Q5_0", [GGML_TYPE_Q5_1] = "Q5_1",
            [GGML_TYPE_IQ4_NL] = "IQ4_NL", [GGML_TYPE_IQ4_XS] = "IQ4_XS",
            [GGML_TYPE_IQ2_XS] = "IQ2_XS", [GGML_TYPE_IQ2_S] = "IQ2_S",
            [GGML_TYPE_IQ3_XXS] = "IQ3_XXS", [GGML_TYPE_IQ3_S] = "IQ3_S",
            [GGML_TYPE_IQ1_S] = "IQ1_S", [GGML_TYPE_IQ1_M] = "IQ1_M",
            [GGML_TYPE_TQ1_0] = "TQ1_0", [GGML_TYPE_TQ2_0] = "TQ2_0",
        };
        const char *tn = (embd.type < sizeof(type_names)/sizeof(type_names[0]) && type_names[embd.type]) ? type_names[embd.type] : "unknown";
        fprintf(stderr, "cuda_llm: token_embd type=%s\n", tn);
    }

    /* Output norm (F32) */
    if (r->verbose) fprintf(stderr, "cuda_llm: loading output_norm...\n");
    qtensor onorm = cllm_load_tensor(gguf, "output_norm.weight", 1);
    if (!onorm.data) { fprintf(stderr, "cuda_llm: output_norm.weight not found!\n"); return -1; }
    if (r->verbose) fprintf(stderr, "cuda_llm: output_norm type=%d n_rows=%d n_cols=%d\n",
        onorm.type, onorm.n_rows, onorm.n_cols);
    if (upload_norm_f32(&r->d_output_norm, &onorm, r->n_embd) != 0) {
        fprintf(stderr, "cuda_llm: upload output_norm failed!\n"); return -1;
    }

    /* Output projection (lm_head) — may be weight-tied with token_embd */
    if (r->verbose) fprintf(stderr, "cuda_llm: loading output.weight...\n");
    {
        qtensor output = cllm_load_tensor(gguf, "output.weight", 0);
        if (output.data) {
            if (upload_weight_matrix(&r->d_output_w, &output, &r->output_w_type) != 0) { fprintf(stderr, "cuda_llm: output.weight upload failed!\n"); return -1; }
            r->has_lm_head = 1;
            if (r->verbose) fprintf(stderr, "cuda_llm: output.weight loaded (type=%d)\n", r->output_w_type);
        } else {
            /* Weight tying: share token_embd */
            r->d_output_w = r->d_token_embd;
            r->output_w_type = r->token_embd_type;
            r->has_lm_head = 1;
            if (r->verbose) fprintf(stderr, "cuda_llm: using weight-tied output (token_embd)\n");
        }
    }

    /* Per-layer weights */
    if (r->verbose) fprintf(stderr, "cuda_llm: loading per-layer weights (%d layers)...\n", r->n_layers);
    r->layers = (cuda_layer *)calloc(r->n_layers, sizeof(cuda_layer));
    if (!r->layers) return -1;

    for (int l = 0; l < r->n_layers; l++) {
        char name[128];
        cuda_layer *cl = &r->layers[l];
        if (r->verbose >= 2) fprintf(stderr, "cuda_llm: loading layer %d/%d\n", l, r->n_layers);

        /* Determine layer type for hybrid models */
        int is_ssm = (r->is_hybrid && r->full_attn_interval > 0 &&
                      (l + 1) % r->full_attn_interval != 0);
        cl->is_ssm = is_ssm;

        /* Attention norm (F32) — shared by all layer types */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        qtensor t = cllm_load_tensor(gguf, name, 1);
        if (upload_norm_f32(&cl->attn_norm_w, &t, r->n_embd) != 0) return -1;

        if (r->verbose >= 2) fprintf(stderr, "cuda_llm: layer %d (%s)...\n", l, is_ssm ? "SSM" : "ATT");
        if (is_ssm) {
            /* --- SSM (Delta-Net) layer weights --- */
            #define LOAD_SSM_W(field, suffix, rows_f, cols_f, type_f) do { \
                snprintf(name, sizeof(name), "blk.%d." suffix ".weight", l); \
                t = cllm_load_tensor(gguf, name, 1); \
                cl->rows_f = t.n_rows; cl->cols_f = t.n_cols; \
                if (upload_weight_matrix(&cl->field, &t, &cl->type_f) != 0) return -1; \
            } while(0)
            LOAD_SSM_W(ssm_qkv_w,   "attn_qkv",  ssm_qkv_rows,   ssm_qkv_cols,   ssm_qkv_type);
            LOAD_SSM_W(ssm_gate_w,   "attn_gate",  ssm_gate_rows,  ssm_gate_cols,  ssm_gate_type);
            LOAD_SSM_W(ssm_alpha_w,  "ssm_alpha",  ssm_alpha_rows, ssm_alpha_cols, ssm_alpha_type);
            LOAD_SSM_W(ssm_beta_w,   "ssm_beta",   ssm_beta_rows,  ssm_beta_cols,  ssm_beta_type);
            LOAD_SSM_W(ssm_out_w,    "ssm_out",    ssm_out_rows,   ssm_out_cols,   ssm_out_type);
            #undef LOAD_SSM_W

            /* ssm_a (F32, no .weight suffix) */
            snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ssm_a, &t, r->ssm_dt_rank) != 0) return -1;

            /* ssm_dt.bias (F32) */
            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ssm_dt_bias, &t, r->ssm_dt_rank) != 0) return -1;

            /* ssm_conv1d (F32) */
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ssm_conv1d_w, &t, t.n_rows * t.n_cols) != 0) return -1;

            /* ssm_norm (F32) */
            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ssm_norm_w, &t, r->ssm_d_state) != 0) return -1;

            /* Allocate SSM persistent state */
            size_t conv_bytes = (size_t)(r->ssm_conv_kernel - 1) * r->ssm_qkv_dim * sizeof(float);
            CHECK_CU(cuMemAlloc(&cl->d_conv_state, conv_bytes));
            CHECK_CU(cuMemsetD8(cl->d_conv_state, 0, conv_bytes));
            size_t rec_bytes = (size_t)r->ssm_dt_rank * r->ssm_d_state * r->ssm_d_state * sizeof(float);
            CHECK_CU(cuMemAlloc(&cl->d_recurrent_state, rec_bytes));
            CHECK_CU(cuMemsetD8(cl->d_recurrent_state, 0, rec_bytes));

            if (r->verbose >= 2) {
                fprintf(stderr, "  layer %d [SSM]: qkv[%d×%d] gate[%d×%d] out[%d×%d]\n",
                        l, cl->ssm_qkv_rows, cl->ssm_qkv_cols,
                        cl->ssm_gate_rows, cl->ssm_gate_cols,
                        cl->ssm_out_rows, cl->ssm_out_cols);
            }
        } else {
            /* --- Attention layer weights --- */
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->attn_q_rows = t.n_rows; cl->attn_q_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_q_w, &t, &cl->attn_q_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->attn_k_rows = t.n_rows; cl->attn_k_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_k_w, &t, &cl->attn_k_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->attn_v_rows = t.n_rows; cl->attn_v_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_v_w, &t, &cl->attn_v_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->attn_output_rows = t.n_rows; cl->attn_output_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_output_w, &t, &cl->attn_output_type) != 0) return -1;

            /* QK norms (F32, optional) */
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
            t = cllm_load_tensor(gguf, name, 0);
            cl->has_qk_norm = (t.data != NULL);
            if (t.data) {
                if (upload_norm_f32(&cl->attn_q_norm_w, &t, r->head_dim) != 0) return -1;
            }
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            t = cllm_load_tensor(gguf, name, 0);
            if (t.data) {
                if (upload_norm_f32(&cl->attn_k_norm_w, &t, r->head_dim) != 0) return -1;
            }

            if (r->verbose >= 2) {
                fprintf(stderr, "  layer %d [ATT]: Q[%d×%d] K[%d×%d] V[%d×%d] O[%d×%d] qk_norm=%d\n",
                        l, cl->attn_q_rows, cl->attn_q_cols,
                        cl->attn_k_rows, cl->attn_k_cols,
                        cl->attn_v_rows, cl->attn_v_cols,
                        cl->attn_output_rows, cl->attn_output_cols,
                        cl->has_qk_norm);
            }
        }

        /* FFN norm (F32) — shared by all layer types */
        if (r->is_hybrid) {
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l);
        } else {
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        }
        t = cllm_load_tensor(gguf, name, 1);
        if (upload_norm_f32(&cl->ffn_norm_w, &t, r->n_embd) != 0) return -1;

        /* FFN weights */
        if (r->is_moe) {
            /* --- MoE FFN weights --- */
            cl->is_moe = 1;

            /* Router: ffn_gate_inp [n_experts, n_embd] F32 */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_gate_rows = t.n_rows; cl->moe_gate_cols = t.n_cols;
            if (upload_norm_f32(&cl->moe_gate_w, &t, t.n_rows * t.n_cols) != 0) return -1;

            /* Expert 3D weights (K-quant packed) */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_gate_exps_type = t.type;
            cl->moe_exp_cols_gu = t.n_cols;  /* n_embd (input dim) */
            cl->moe_exp_rows_gu = (t.n_dims >= 3) ? (int)t.dims[1] : t.n_rows;  /* expert_ff */
            if (upload_3d_kquant_raw(&cl->moe_gate_exps_w, &t, &cl->moe_exp_stride_gu) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_up_exps_type = t.type;
            if (upload_3d_kquant_raw(&cl->moe_up_exps_w, &t, &cl->moe_exp_stride_gu) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_down_exps_type = t.type;
            cl->moe_exp_cols_d = t.n_cols;   /* expert_ff (input dim) */
            cl->moe_exp_rows_d = (t.n_dims >= 3) ? (int)t.dims[1] : t.n_rows;  /* n_embd */
            if (upload_3d_kquant_raw(&cl->moe_down_exps_w, &t, &cl->moe_exp_stride_d) != 0) return -1;

            /* Shared expert gate: ffn_gate_inp_shexp [n_embd] F32 (1D sigmoid gate) */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->moe_shared_gate_w, &t, t.n_rows * t.n_cols) != 0) return -1;

            /* Shared expert FFN (BF16 → F16 via upload_weight_matrix) */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_shared_gate_rows = t.n_rows; cl->moe_shared_gate_cols = t.n_cols;
            if (upload_weight_matrix(&cl->moe_shared_ffn_gate_w, &t, &cl->moe_shared_gate_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_shared_up_rows = t.n_rows; cl->moe_shared_up_cols = t.n_cols;
            if (upload_weight_matrix(&cl->moe_shared_ffn_up_w, &t, &cl->moe_shared_up_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_shared_down_rows = t.n_rows; cl->moe_shared_down_cols = t.n_cols;
            if (upload_weight_matrix(&cl->moe_shared_ffn_down_w, &t, &cl->moe_shared_down_type) != 0) return -1;

            if (r->verbose >= 2) {
                fprintf(stderr, "  layer %d [MoE]: router[%d×%d] exp_gu[%d×%d] exp_d[%d×%d] stride_gu=%zu stride_d=%zu\n",
                        l, cl->moe_gate_rows, cl->moe_gate_cols,
                        cl->moe_exp_rows_gu, cl->moe_exp_cols_gu,
                        cl->moe_exp_rows_d, cl->moe_exp_cols_d,
                        cl->moe_exp_stride_gu, cl->moe_exp_stride_d);
                fprintf(stderr, "         shared_gate[%d×%d] shared_up[%d×%d] shared_down[%d×%d]\n",
                        cl->moe_shared_gate_rows, cl->moe_shared_gate_cols,
                        cl->moe_shared_up_rows, cl->moe_shared_up_cols,
                        cl->moe_shared_down_rows, cl->moe_shared_down_cols);
            }
        } else {
            /* Dense FFN gate/up/down */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->ffn_gate_rows = t.n_rows; cl->ffn_gate_cols = t.n_cols;
            if (upload_weight_matrix(&cl->ffn_gate_w, &t, &cl->ffn_gate_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->ffn_up_rows = t.n_rows; cl->ffn_up_cols = t.n_cols;
            if (upload_weight_matrix(&cl->ffn_up_w, &t, &cl->ffn_up_type) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->ffn_down_rows = t.n_rows; cl->ffn_down_cols = t.n_cols;
            if (upload_weight_matrix(&cl->ffn_down_w, &t, &cl->ffn_down_type) != 0) return -1;
        }
    }

    /* Allocate KV cache (skip SSM layers) */
    r->d_key_cache = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
    r->d_value_cache = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
    size_t kv_cache_size = (size_t)max_seq_len * kv_dim * sizeof(float);
    int n_attn_layers = 0;
    for (int l = 0; l < r->n_layers; l++) {
        if (r->layers[l].is_ssm) continue;
        CHECK_CU(cuMemAlloc(&r->d_key_cache[l], kv_cache_size));
        CHECK_CU(cuMemsetD8(r->d_key_cache[l], 0, kv_cache_size));
        CHECK_CU(cuMemAlloc(&r->d_value_cache[l], kv_cache_size));
        CHECK_CU(cuMemsetD8(r->d_value_cache[l], 0, kv_cache_size));
        n_attn_layers++;
    }
    if (r->is_hybrid && r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: %d attention layers (KV cache), %d SSM layers\n",
                n_attn_layers, r->n_layers - n_attn_layers);
    }

    /* Allocate scratch buffers */
    int max_dim = r->n_embd;
    if (q_dim > max_dim) max_dim = q_dim;
    if (r->n_ff > max_dim) max_dim = r->n_ff;
    /* For hybrid: xb2 must hold Q+gate (2*q_dim) or SSM qkv (qkv_dim) */
    int xb2_dim = max_dim;
    if (r->is_hybrid) {
        if (r->ssm_qkv_dim > xb2_dim) xb2_dim = r->ssm_qkv_dim;
        if (2 * q_dim > xb2_dim) xb2_dim = 2 * q_dim;
    }

    CHECK_CU(cuMemAlloc(&r->d_x,   max_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_xb,  max_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_xb2, xb2_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_q,   q_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_k,   kv_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_v,   kv_dim * sizeof(float)));
    {
        int ff_dim = r->n_ff;
        if (r->is_moe && r->shared_expert_ff > ff_dim) ff_dim = r->shared_expert_ff;
        CHECK_CU(cuMemAlloc(&r->d_gate, ff_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_up,   ff_dim * sizeof(float)));
    }

    /* MoE scratch buffers */
    if (r->is_moe) {
        CHECK_CU(cuMemAlloc(&r->d_router_logits, r->n_experts * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_moe_accum, r->n_embd * sizeof(float)));
        r->h_router_logits = (float *)malloc(r->n_experts * sizeof(float));
        if (!r->h_router_logits) return -1;
    }

    /* SSM scratch buffers */
    if (r->is_hybrid) {
        int d_inner = r->ssm_d_inner;
        int dt_rank = r->ssm_dt_rank;
        int d_state = r->ssm_d_state;
        CHECK_CU(cuMemAlloc(&r->d_ssm_qkv,      r->ssm_qkv_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ssm_z,         d_inner * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ssm_alpha,     dt_rank * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ssm_beta,      dt_rank * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ssm_Q_exp,     dt_rank * d_state * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ssm_K_exp,     dt_rank * d_state * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ssm_out,       d_inner * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ssm_conv_out,  r->ssm_qkv_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_attn_gate,     q_dim * sizeof(float)));
    }

    /* DeepStack scratch buffer */
    if (r->n_deepstack > 0) {
        CHECK_CU(cuMemAlloc(&r->d_ds_tmp, r->n_embd * sizeof(float)));
    }

    /* INT8 quantization scratch (for dp4a path) */
    CHECK_CU(cuMemAlloc(&r->d_xb_q,     max_dim * sizeof(int8_t)));
    CHECK_CU(cuMemAlloc(&r->d_xb_scale, sizeof(float)));

    /* Logits buffer (GPU + host) */
    CHECK_CU(cuMemAlloc(&r->d_logits, (size_t)r->n_vocab * sizeof(float)));

    /* Host output buffer (large enough for logits) */
    int out_sz = r->n_vocab > r->n_embd ? r->n_vocab : r->n_embd;
    r->h_output = (float *)malloc((size_t)out_sz * sizeof(float));
    if (!r->h_output) return -1;

    r->weights_loaded = 1;

    if (r->verbose >= 1) {
        /* Helper: bytes on GPU for a weight type (Q8_0 uses padded 36B blocks) */
        #define WEIGHT_BYTES(type, n_elements) \
            ((type) == GGML_TYPE_Q8_0 ? (size_t)((n_elements) / 32) * 36 : \
             ((type) == GGML_TYPE_Q2_K || (type) == GGML_TYPE_Q3_K || \
              (type) == GGML_TYPE_Q4_K || (type) == GGML_TYPE_Q5_K || \
              (type) == GGML_TYPE_Q6_K || (type) == GGML_TYPE_IQ2_XXS || \
              (type) == GGML_TYPE_Q4_0 || (type) == GGML_TYPE_Q4_1 || \
              (type) == GGML_TYPE_Q5_0 || (type) == GGML_TYPE_Q5_1 || \
              (type) == GGML_TYPE_IQ2_XS || (type) == GGML_TYPE_IQ2_S || \
              (type) == GGML_TYPE_IQ3_XXS || (type) == GGML_TYPE_IQ3_S || \
              (type) == GGML_TYPE_IQ4_NL || (type) == GGML_TYPE_IQ4_XS || \
              (type) == GGML_TYPE_IQ1_S || (type) == GGML_TYPE_IQ1_M || \
              (type) == GGML_TYPE_TQ1_0 || (type) == GGML_TYPE_TQ2_0) ? dequant_row_size(type, n_elements) : \
             (size_t)(n_elements) * 2)

        /* VRAM summary */
        size_t weight_bytes = WEIGHT_BYTES(r->token_embd_type, (size_t)r->n_vocab * r->n_embd);
        weight_bytes += r->n_embd * 4;  /* output_norm */
        for (int l = 0; l < r->n_layers; l++) {
            cuda_layer *cl = &r->layers[l];
            weight_bytes += r->n_embd * 4;  /* attn_norm */
            weight_bytes += WEIGHT_BYTES(cl->attn_q_type, (size_t)cl->attn_q_rows * cl->attn_q_cols);
            weight_bytes += WEIGHT_BYTES(cl->attn_k_type, (size_t)cl->attn_k_rows * cl->attn_k_cols);
            weight_bytes += WEIGHT_BYTES(cl->attn_v_type, (size_t)cl->attn_v_rows * cl->attn_v_cols);
            weight_bytes += WEIGHT_BYTES(cl->attn_output_type, (size_t)cl->attn_output_rows * cl->attn_output_cols);
            if (cl->has_qk_norm) weight_bytes += r->head_dim * 4 * 2;
            weight_bytes += r->n_embd * 4;  /* ffn_norm */
            if (cl->is_moe) {
                /* MoE: router + expert weights + shared expert */
                weight_bytes += (size_t)cl->moe_gate_rows * cl->moe_gate_cols * 4;  /* router F32 */
                weight_bytes += cl->moe_exp_stride_gu * r->n_experts * 2;  /* gate_exps + up_exps */
                weight_bytes += cl->moe_exp_stride_d * r->n_experts;  /* down_exps */
                weight_bytes += r->n_embd * 4;  /* shared gate F32 */
                weight_bytes += WEIGHT_BYTES(cl->moe_shared_gate_type, (size_t)cl->moe_shared_gate_rows * cl->moe_shared_gate_cols);
                weight_bytes += WEIGHT_BYTES(cl->moe_shared_up_type, (size_t)cl->moe_shared_up_rows * cl->moe_shared_up_cols);
                weight_bytes += WEIGHT_BYTES(cl->moe_shared_down_type, (size_t)cl->moe_shared_down_rows * cl->moe_shared_down_cols);
            } else {
                weight_bytes += WEIGHT_BYTES(cl->ffn_gate_type, (size_t)cl->ffn_gate_rows * cl->ffn_gate_cols);
                weight_bytes += WEIGHT_BYTES(cl->ffn_up_type, (size_t)cl->ffn_up_rows * cl->ffn_up_cols);
                weight_bytes += WEIGHT_BYTES(cl->ffn_down_type, (size_t)cl->ffn_down_rows * cl->ffn_down_cols);
            }
        }
        #undef WEIGHT_BYTES
        size_t cache_bytes = (size_t)r->n_layers * 2 * kv_cache_size;
        size_t scratch_bytes = (size_t)(max_dim * 3 + q_dim + kv_dim * 2 + r->n_ff * 2) * sizeof(float)
                             + max_dim + sizeof(float);  /* INT8 scratch */
        cuda_layer *cl0 = &r->layers[0];
        const char *wtype = "F16";
        if (cl0->attn_q_type == GGML_TYPE_Q8_0) wtype = "Q8_0+dp4a";
        else if (cl0->attn_q_type == GGML_TYPE_Q2_K) wtype = "Q2_K";
        else if (cl0->attn_q_type == GGML_TYPE_Q3_K) wtype = "Q3_K";
        else if (cl0->attn_q_type == GGML_TYPE_Q4_K) wtype = "Q4_K";
        else if (cl0->attn_q_type == GGML_TYPE_Q5_K) wtype = "Q5_K";
        else if (cl0->attn_q_type == GGML_TYPE_Q6_K) wtype = "Q6_K";
        if (r->is_moe) wtype = "MoE+K-quant";
        fprintf(stderr, "cuda_llm: VRAM usage: weights=%.1f MB (%s), KV cache=%.1f MB, scratch=%.1f KB\n",
                (double)weight_bytes / (1024.0*1024.0), wtype,
                (double)cache_bytes / (1024.0*1024.0),
                (double)scratch_bytes / 1024.0);
    }

    return 0;
}

/* ======================================================================== */
/* Kernel launch helpers                                                    */
/* ======================================================================== */

static inline void launch_embed(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr embd_table,
                                int token_id, int n_embd) {
    void *args[] = { &dst, &embd_table, &token_id, &n_embd };
    cuLaunchKernel(r->fn_embed_f16,
                   (n_embd + 255) / 256, 1, 1,
                   256, 1, 1,
                   0, r->stream, args, NULL);
}

static inline void launch_rmsnorm(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr x,
                                   CUdeviceptr w, int n, float eps) {
    void *args[] = { &dst, &x, &w, &n, &eps };
    cuLaunchKernel(r->fn_rmsnorm_f32,
                   1, 1, 1,
                   256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static inline void launch_matvec(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                  CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_f16_f32,
                   n_rows, 1, 1,
                   256, 1, 1,
                   0, r->stream, args, NULL);
}

static inline void launch_qknorm(cuda_llm_runner *r, CUdeviceptr vec, CUdeviceptr w,
                                   int n_heads, int head_dim, float eps) {
    /* blockDim = next power of 2 >= head_dim, capped at 256 */
    int bdim = 1;
    while (bdim < head_dim) bdim <<= 1;
    if (bdim > 256) bdim = 256;
    void *args[] = { &vec, &w, &n_heads, &head_dim, &eps };
    cuLaunchKernel(r->fn_qknorm_f32,
                   n_heads, 1, 1,
                   bdim, 1, 1,
                   bdim * sizeof(float), r->stream, args, NULL);
}

static inline void launch_rope(cuda_llm_runner *r, CUdeviceptr vec, int n_heads,
                                int head_dim, int pos, float freq_base) {
    int half_dim = head_dim / 2;
    int n_rope_pairs = r->n_rope_pairs;
    void *args[] = { &vec, &n_heads, &head_dim, &pos, &freq_base, &n_rope_pairs };
    cuLaunchKernel(r->fn_rope_neox_f32,
                   n_heads, 1, 1,
                   half_dim, 1, 1,
                   0, r->stream, args, NULL);
}

static inline void launch_kv_store(cuda_llm_runner *r, CUdeviceptr key_cache, CUdeviceptr value_cache,
                                    CUdeviceptr k, CUdeviceptr v, int position, int kv_dim) {
    void *args[] = { &key_cache, &value_cache, &k, &v, &position, &kv_dim };
    cuLaunchKernel(r->fn_kv_cache_store,
                   (kv_dim + 255) / 256, 1, 1,
                   256, 1, 1,
                   0, r->stream, args, NULL);
}

static inline void launch_attention(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr q,
                                     CUdeviceptr key_cache, CUdeviceptr value_cache,
                                     int n_heads, int n_kv_heads, int head_dim,
                                     int kv_dim, int seq_len, float scale) {
    size_t smem = seq_len * sizeof(float);
    /* Cap shared memory; for very long sequences this may need adjustment */
    void *args[] = { &out, &q, &key_cache, &value_cache,
                     &n_heads, &n_kv_heads, &head_dim, &kv_dim, &seq_len, &scale };
    cuLaunchKernel(r->fn_attn_decode_f32,
                   n_heads, 1, 1,
                   256, 1, 1,
                   smem, r->stream, args, NULL);
}

static inline void launch_silu_mul(cuda_llm_runner *r, CUdeviceptr gate, CUdeviceptr up, int n) {
    void *args[] = { &gate, &up, &n };
    cuLaunchKernel(r->fn_silu_mul_f32,
                   (n + 255) / 256, 1, 1,
                   256, 1, 1,
                   0, r->stream, args, NULL);
}

static inline void launch_add(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr src, int n) {
    void *args[] = { &dst, &src, &n };
    cuLaunchKernel(r->fn_add_f32,
                   (n + 255) / 256, 1, 1,
                   256, 1, 1,
                   0, r->stream, args, NULL);
}

/* ---- Q8_0 dp4a launch helpers ---- */

static inline void launch_embed_q8_0(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr embd_table,
                                      int token_id, int n_embd) {
    void *args[] = { &dst, &embd_table, &token_id, &n_embd };
    cuLaunchKernel(r->fn_embed_q8_0,
                   (n_embd + 255) / 256, 1, 1,
                   256, 1, 1,
                   0, r->stream, args, NULL);
}

static inline void launch_quantize(cuda_llm_runner *r, CUdeviceptr dst_q, CUdeviceptr dst_scale,
                                    CUdeviceptr src, int n) {
    void *args[] = { &dst_q, &dst_scale, &src, &n };
    cuLaunchKernel(r->fn_quantize_f32_to_int8,
                   1, 1, 1,
                   256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static inline void launch_matvec_q8(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                     CUdeviceptr x_q, CUdeviceptr x_scale,
                                     int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q, &x_scale, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q8_0_dp4a,
                   n_rows, 1, 1,
                   256, 1, 1,
                   0, r->stream, args, NULL);
}

static inline void launch_matvec_q8_f32(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                         CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q8_0_f32,
                   n_rows, 1, 1,
                   256, 1, 1,
                   0, r->stream, args, NULL);
}

/* ---- K-quant launch helpers ---- */

static inline void launch_matvec_q2_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q2_K_f32,
                   n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q3_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q3_K_f32,
                   n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q4_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q4_K_f32,
                   n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q5_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q5_K_f32,
                   n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q6_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q6_K_f32,
                   n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_iq2_xxs(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_iq2_xxs_f32,
                   n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

#define DEFINE_LAUNCH_MATVEC(name, fn_field) \
static inline void launch_matvec_##name(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat, \
                                        CUdeviceptr x, int n_rows, int n_cols) { \
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols }; \
    cuLaunchKernel(r->fn_field, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL); \
}

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

static inline void launch_embed_q2_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr embd_table,
                                       int token_id, int n_embd) {
    void *args[] = { &dst, &embd_table, &token_id, &n_embd };
    cuLaunchKernel(r->fn_embed_q2_K,
                   (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* Auto-dispatch matvec based on weight type */
static inline void launch_matvec_auto(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                       CUdeviceptr x, int n_rows, int n_cols, int weight_type) {
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

/* ---- SSM launch helpers ---- */

static inline void launch_softplus_mul(cuda_llm_runner *r, CUdeviceptr out,
    CUdeviceptr in, CUdeviceptr bias, CUdeviceptr a, int n) {
    void *args[] = { &out, &in, &bias, &a, &n };
    cuLaunchKernel(r->fn_softplus_mul_f32,
                   (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_sigmoid_inplace(cuda_llm_runner *r, CUdeviceptr data, int n) {
    void *args[] = { &data, &n };
    cuLaunchKernel(r->fn_sigmoid_inplace_f32,
                   (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_conv1d(cuda_llm_runner *r, CUdeviceptr conv_out,
    CUdeviceptr conv_state, CUdeviceptr input, CUdeviceptr weight,
    int qkv_dim, int conv_k) {
    void *args[] = { &conv_out, &conv_state, &input, &weight, &qkv_dim, &conv_k };
    cuLaunchKernel(r->fn_conv1d_depthwise_silu_f32,
                   (qkv_dim + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_l2_norm_heads(cuda_llm_runner *r, CUdeviceptr data,
    int n_heads, int head_dim, float eps) {
    int threads = (head_dim <= 128) ? 128 : 256;
    void *args[] = { &data, &n_heads, &head_dim, &eps };
    cuLaunchKernel(r->fn_l2_norm_heads_f32,
                   n_heads, 1, 1, threads, 1, 1, threads * sizeof(float),
                   r->stream, args, NULL);
}

static inline void launch_repeat_tile(cuda_llm_runner *r, CUdeviceptr dst,
    CUdeviceptr src, int dt_rank, int d_state, int n_group) {
    int total = dt_rank * d_state;
    void *args[] = { &dst, &src, &dt_rank, &d_state, &n_group };
    cuLaunchKernel(r->fn_repeat_tile_f32,
                   (total + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_deltanet_step(cuda_llm_runner *r, CUdeviceptr state,
    CUdeviceptr out, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
    CUdeviceptr alpha, CUdeviceptr beta, int dt_rank, int d_state) {
    void *args[] = { &state, &out, &Q, &K, &V, &alpha, &beta, &d_state };
    cuLaunchKernel(r->fn_deltanet_step_f32,
                   dt_rank, 1, 1, d_state, 1, 1, 0,
                   r->stream, args, NULL);
}

static inline void launch_gated_rmsnorm_silu(cuda_llm_runner *r, CUdeviceptr out,
    CUdeviceptr z, CUdeviceptr norm_w, int dt_rank, int d_state, float eps) {
    int threads = (d_state <= 128) ? 128 : 256;
    void *args[] = { &out, &z, &norm_w, &dt_rank, &d_state, &eps };
    cuLaunchKernel(r->fn_gated_rmsnorm_silu_f32,
                   dt_rank, 1, 1, threads, 1, 1, threads * sizeof(float),
                   r->stream, args, NULL);
}

static inline void launch_sigmoid_mul(cuda_llm_runner *r, CUdeviceptr data,
    CUdeviceptr gate, int n) {
    void *args[] = { &data, &gate, &n };
    cuLaunchKernel(r->fn_sigmoid_mul_f32,
                   (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_deinterleave_qgate(cuda_llm_runner *r, CUdeviceptr q,
    CUdeviceptr gate, CUdeviceptr qfull, int n_heads, int head_dim) {
    int total = n_heads * head_dim;
    void *args[] = { &q, &gate, &qfull, &n_heads, &head_dim };
    cuLaunchKernel(r->fn_deinterleave_qgate_f32,
                   (total + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* ---- MoE launch helpers ---- */

static inline void launch_scale_add(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr src,
                                     float scale, int n) {
    void *args[] = { &dst, &src, &scale, &n };
    cuLaunchKernel(r->fn_scale_add_f32,
                   (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_f32(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                      CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_f32_f32,
                   n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* Top-K softmax for MoE routing: select top-k experts and compute softmax weights */
static void moe_topk_softmax(const float *logits, int n, int k, int *out_idx, float *out_weights) {
    /* Initialize with -inf */
    for (int i = 0; i < k; i++) { out_idx[i] = -1; out_weights[i] = -1e30f; }

    /* Simple selection sort for top-k (k is small, typically 8) */
    for (int ki = 0; ki < k; ki++) {
        float best = -1e30f;
        int best_idx = -1;
        for (int i = 0; i < n; i++) {
            /* Skip already selected */
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

    /* Softmax over selected weights */
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

static float *cuda_llm_forward_blocks(cuda_llm_runner *r, int position);

float *cuda_llm_forward(cuda_llm_runner *r, int32_t token_id, int position) {
    if (!r || !r->weights_loaded) return NULL;
    if (token_id < 0 || token_id >= r->n_vocab) return NULL;
    if (position < 0 || position >= r->max_seq_len) return NULL;

    int n_embd = r->n_embd;

    /* 1. Token embedding lookup -> F32 */
    if (r->token_embd_type == GGML_TYPE_Q8_0) {
        launch_embed_q8_0(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else if (r->token_embd_type == GGML_TYPE_Q2_K) {
        launch_embed_q2_K(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else {
        launch_embed(r, r->d_x, r->d_token_embd, token_id, n_embd);
    }

    if (r->debug_layers >= 2) {
        cuStreamSynchronize(r->stream);
        float dbg[8];
        cuMemcpyDtoH(dbg, r->d_x, 8 * sizeof(float));
        fprintf(stderr, "  [EMB DBG] embd[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
    }

    /* Embedding cross-check: verify Q2_K embed kernel against CPU dequant */
    if (r->debug_layers >= 3 && r->token_embd_type == GGML_TYPE_Q2_K) {
        cuStreamSynchronize(r->stream);
        float *gpu_emb = (float *)malloc(n_embd * sizeof(float));
        cuMemcpyDtoH(gpu_emb, r->d_x, n_embd * sizeof(float));

        /* Download Q2_K embedding weights from GPU */
        int nb_per_row = n_embd / 256;
        int row_bytes = nb_per_row * 84;
        size_t total_embd_bytes = (size_t)r->n_vocab * row_bytes;
        /* Only download the one row we need */
        unsigned char *row_data = (unsigned char *)malloc(row_bytes);
        cuMemcpyDtoH(row_data, r->d_token_embd + (size_t)token_id * row_bytes, row_bytes);

        /* CPU dequantize this row */
        float *cpu_emb = (float *)malloc(n_embd * sizeof(float));
        dequant_row(GGML_TYPE_Q2_K, row_data, cpu_emb, n_embd);

        float emb_max_err = 0; int emb_max_idx = 0;
        for (int i = 0; i < n_embd; i++) {
            float err = fabsf(gpu_emb[i] - cpu_emb[i]);
            if (err > emb_max_err) { emb_max_err = err; emb_max_idx = i; }
        }
        fprintf(stderr, "  [EMB xchk] Q2_K embed max_err=%.6f at idx=%d (GPU=%.6f CPU=%.6f)\n",
                emb_max_err, emb_max_idx, gpu_emb[emb_max_idx], cpu_emb[emb_max_idx]);
        if (emb_max_err > 0.001f) {
            fprintf(stderr, "    GPU[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                    gpu_emb[0], gpu_emb[1], gpu_emb[2], gpu_emb[3]);
            fprintf(stderr, "    CPU[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                    cpu_emb[0], cpu_emb[1], cpu_emb[2], cpu_emb[3]);
        }
        free(gpu_emb); free(cpu_emb); free(row_data);
    }

    /* Run transformer blocks (shared with forward_embd) */
    return cuda_llm_forward_blocks(r, position);
}

/* Internal: run transformer blocks + final norm on d_x.
 * Assumes d_x already contains the input embedding on GPU. */
static float *cuda_llm_forward_blocks(cuda_llm_runner *r, int position) {
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
        cuda_layer *cl = &r->layers[l];

        /* Pre-attention RMSNorm: xb = rmsnorm(x, attn_norm) */
        launch_rmsnorm(r, r->d_xb, r->d_x, cl->attn_norm_w, n_embd, eps);

        /* RMSNorm cross-check for layer 0 (debug >= 3) */
        if (r->debug_layers >= 3 && l == 0) {
            cuStreamSynchronize(r->stream);
            float *gpu_x = (float *)malloc(n_embd * sizeof(float));
            float *gpu_xb = (float *)malloc(n_embd * sizeof(float));
            float *gpu_norm_w = (float *)malloc(n_embd * sizeof(float));
            cuMemcpyDtoH(gpu_x, r->d_x, n_embd * sizeof(float));
            cuMemcpyDtoH(gpu_xb, r->d_xb, n_embd * sizeof(float));
            cuMemcpyDtoH(gpu_norm_w, cl->attn_norm_w, n_embd * sizeof(float));

            /* CPU RMSNorm */
            float ss = 0;
            for (int i = 0; i < n_embd; i++) ss += gpu_x[i] * gpu_x[i];
            float cpu_scale = 1.0f / sqrtf(ss / (float)n_embd + eps);
            float norm_max_err = 0; int norm_max_idx = 0;
            for (int i = 0; i < n_embd; i++) {
                float cpu_xb = gpu_x[i] * cpu_scale * gpu_norm_w[i];
                float err = fabsf(cpu_xb - gpu_xb[i]);
                if (err > norm_max_err) { norm_max_err = err; norm_max_idx = i; }
            }
            float emb_norm = sqrtf(ss);
            fprintf(stderr, "  [L0 RMSNorm xchk] max_err=%.6f at idx=%d |emb|=%.4f scale=%.6f\n",
                    norm_max_err, norm_max_idx, emb_norm, cpu_scale);
            if (norm_max_err > 0.001f) {
                fprintf(stderr, "    GPU_xb[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                        gpu_xb[0], gpu_xb[1], gpu_xb[2], gpu_xb[3]);
                float cpu_xb0 = gpu_x[0] * cpu_scale * gpu_norm_w[0];
                float cpu_xb1 = gpu_x[1] * cpu_scale * gpu_norm_w[1];
                fprintf(stderr, "    CPU_xb[0:2]=[%.6f,%.6f]\n", cpu_xb0, cpu_xb1);
            }
            free(gpu_x); free(gpu_xb); free(gpu_norm_w);
        }

        if (r->is_hybrid && cl->is_ssm) {
            /* === SSM (Delta-Net) layer === */
            int qkv_dim = r->ssm_qkv_dim;
            int d_state = r->ssm_d_state;
            int n_group = r->ssm_n_group;
            int dt_rank = r->ssm_dt_rank;
            int conv_k  = r->ssm_conv_kernel;

            /* 1. Linear projections */
            launch_matvec_auto(r, r->d_ssm_qkv,   cl->ssm_qkv_w,   r->d_xb, cl->ssm_qkv_rows,   cl->ssm_qkv_cols,   cl->ssm_qkv_type);
            launch_matvec_auto(r, r->d_ssm_z,      cl->ssm_gate_w,  r->d_xb, cl->ssm_gate_rows,  cl->ssm_gate_cols,  cl->ssm_gate_type);
            launch_matvec_auto(r, r->d_ssm_alpha,  cl->ssm_alpha_w, r->d_xb, cl->ssm_alpha_rows, cl->ssm_alpha_cols, cl->ssm_alpha_type);
            launch_matvec_auto(r, r->d_ssm_beta,   cl->ssm_beta_w,  r->d_xb, cl->ssm_beta_rows,  cl->ssm_beta_cols,  cl->ssm_beta_type);

            /* Detailed debug output for layer 0 */
            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[10];
                cuMemcpyDtoH(dbg, r->d_xb, 8 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] normed[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                cuMemcpyDtoH(dbg, r->d_ssm_qkv, 10 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] qkv[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7],dbg[8],dbg[9]);
                cuMemcpyDtoH(dbg, r->d_ssm_z, 10 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] z[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7],dbg[8],dbg[9]);
                float alpha_dbg[48], beta_dbg[48];
                cuMemcpyDtoH(alpha_dbg, r->d_ssm_alpha, dt_rank * sizeof(float));
                cuMemcpyDtoH(beta_dbg, r->d_ssm_beta, dt_rank * sizeof(float));
                fprintf(stderr, "  [L0 DBG] alpha_raw[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        alpha_dbg[0],alpha_dbg[1],alpha_dbg[2],alpha_dbg[3],alpha_dbg[4],alpha_dbg[5],alpha_dbg[6],alpha_dbg[7],alpha_dbg[8],alpha_dbg[9]);
                fprintf(stderr, "  [L0 DBG] beta_raw[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        beta_dbg[0],beta_dbg[1],beta_dbg[2],beta_dbg[3],beta_dbg[4],beta_dbg[5],beta_dbg[6],beta_dbg[7],beta_dbg[8],beta_dbg[9]);
            }

            /* 2. alpha = softplus(alpha + dt_bias) * ssm_a */
            launch_softplus_mul(r, r->d_ssm_alpha, r->d_ssm_alpha, cl->ssm_dt_bias, cl->ssm_a, dt_rank);

            /* 3. beta = sigmoid(beta) */
            launch_sigmoid_inplace(r, r->d_ssm_beta, dt_rank);

            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float alpha_dbg[10], beta_dbg[10];
                cuMemcpyDtoH(alpha_dbg, r->d_ssm_alpha, 10 * sizeof(float));
                cuMemcpyDtoH(beta_dbg, r->d_ssm_beta, 10 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] alpha_final[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        alpha_dbg[0],alpha_dbg[1],alpha_dbg[2],alpha_dbg[3],alpha_dbg[4],alpha_dbg[5],alpha_dbg[6],alpha_dbg[7],alpha_dbg[8],alpha_dbg[9]);
                fprintf(stderr, "  [L0 DBG] beta_sigmoid[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        beta_dbg[0],beta_dbg[1],beta_dbg[2],beta_dbg[3],beta_dbg[4],beta_dbg[5],beta_dbg[6],beta_dbg[7],beta_dbg[8],beta_dbg[9]);
            }

            /* 4. Conv1d + SiLU + state update */
            launch_conv1d(r, r->d_ssm_conv_out, cl->d_conv_state, r->d_ssm_qkv,
                         cl->ssm_conv1d_w, qkv_dim, conv_k);

            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_ssm_conv_out, 8 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] conv_Q[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                cuMemcpyDtoH(dbg, r->d_ssm_conv_out + (size_t)n_group * d_state * sizeof(float), 8 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] conv_K[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                cuMemcpyDtoH(dbg, r->d_ssm_conv_out + (size_t)2 * n_group * d_state * sizeof(float), 8 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] conv_V[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }

            /* Save conv_out for cross-check before L2 norm modifies it */
            float *xchk_conv_out = NULL;
            float *xchk_qkv = NULL;
            float *xchk_alpha = NULL;
            float *xchk_beta = NULL;
            float *xchk_z = NULL;
            float *xchk_conv_w = NULL;
            if (r->debug_layers >= 3 && l == 0) {
                cuStreamSynchronize(r->stream);
                xchk_conv_out = (float *)malloc(qkv_dim * sizeof(float));
                xchk_qkv = (float *)malloc(qkv_dim * sizeof(float));
                xchk_alpha = (float *)malloc(dt_rank * sizeof(float));
                xchk_beta = (float *)malloc(dt_rank * sizeof(float));
                xchk_z = (float *)malloc(dt_rank * d_state * sizeof(float));
                xchk_conv_w = (float *)malloc(qkv_dim * conv_k * sizeof(float));
                cuMemcpyDtoH(xchk_conv_out, r->d_ssm_conv_out, qkv_dim * sizeof(float));
                cuMemcpyDtoH(xchk_qkv, r->d_ssm_qkv, qkv_dim * sizeof(float));
                cuMemcpyDtoH(xchk_alpha, r->d_ssm_alpha, dt_rank * sizeof(float));
                cuMemcpyDtoH(xchk_beta, r->d_ssm_beta, dt_rank * sizeof(float));
                cuMemcpyDtoH(xchk_z, r->d_ssm_z, dt_rank * d_state * sizeof(float));
                cuMemcpyDtoH(xchk_conv_w, cl->ssm_conv1d_w, qkv_dim * conv_k * sizeof(float));

                fprintf(stderr, "\n  === L0 SSM FULL CPU CROSS-CHECK ===\n");

                /* Verify conv1d for first token */
                if (position == 0) {
                    float conv_max_err = 0; int conv_max_j = 0;
                    for (int j = 0; j < qkv_dim; j++) {
                        float s_val = xchk_conv_w[j * conv_k + (conv_k - 1)] * xchk_qkv[j];
                        float cpu_c = s_val / (1.0f + expf(-s_val));
                        float err = fabsf(cpu_c - xchk_conv_out[j]);
                        if (err > conv_max_err) { conv_max_err = err; conv_max_j = j; }
                    }
                    fprintf(stderr, "  [Conv1d] max_err=%.6f at j=%d\n", conv_max_err, conv_max_j);
                }
            }

            /* 5. L2-normalize Q and K (first and last n_group*d_state slices) */
            launch_l2_norm_heads(r, r->d_ssm_conv_out, n_group, d_state, eps);
            CUdeviceptr K_raw = r->d_ssm_conv_out + (size_t)n_group * d_state * sizeof(float);
            launch_l2_norm_heads(r, K_raw, n_group, d_state, eps);

            /* 6. Repeat Q and K from n_group to dt_rank heads (tiling) */
            launch_repeat_tile(r, r->d_ssm_Q_exp, r->d_ssm_conv_out, dt_rank, d_state, n_group);
            launch_repeat_tile(r, r->d_ssm_K_exp, K_raw, dt_rank, d_state, n_group);

            /* V is at offset 2*n_group*d_state in conv_out */
            CUdeviceptr V_ptr = r->d_ssm_conv_out + (size_t)2 * n_group * d_state * sizeof(float);

            /* 7. Delta-Net recurrence: one block per head, d_state threads */
            launch_deltanet_step(r, cl->d_recurrent_state, r->d_ssm_out,
                                r->d_ssm_Q_exp, r->d_ssm_K_exp, V_ptr,
                                r->d_ssm_alpha, r->d_ssm_beta, dt_rank, d_state);

            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_ssm_out, 8 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] deltanet_out[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }

            /* 8. Gated RMSNorm + SiLU: out = rmsnorm(out, norm_w) * silu(z) */
            launch_gated_rmsnorm_silu(r, r->d_ssm_out, r->d_ssm_z, cl->ssm_norm_w,
                                     dt_rank, d_state, eps);

            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_ssm_out, 8 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] gated_out[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }

            /* SSM intermediate norms for ALL SSM layers (debug_layers >= 2) */
            if (r->debug_layers >= 2 && position == 0) {
                cuStreamSynchronize(r->stream);
                int d_inner = dt_rank * d_state;
                float gnbuf[512];
                int gn_n = d_inner < 512 ? d_inner : 512;
                cuMemcpyDtoH(gnbuf, r->d_ssm_out, gn_n * sizeof(float));
                float gn = 0;
                for (int gi = 0; gi < gn_n; gi++) gn += gnbuf[gi] * gnbuf[gi];
                float gated_norm = sqrtf(gn * (float)d_inner / (float)gn_n);
                /* Also get z norm */
                cuMemcpyDtoH(gnbuf, r->d_ssm_z, gn_n * sizeof(float));
                float zn = 0;
                for (int gi = 0; gi < gn_n; gi++) zn += gnbuf[gi] * gnbuf[gi];
                float z_norm = sqrtf(zn * (float)d_inner / (float)gn_n);
                fprintf(stderr, "  [L%02d SSM detail] gated_norm=%.2f z_norm=%.2f\n",
                        l, gated_norm, z_norm);
            }

            /* Full CPU cross-check AFTER all SSM ops have run (debug_layers >= 3, layer 0) */
            if (r->debug_layers >= 3 && l == 0 && xchk_conv_out) {
                cuStreamSynchronize(r->stream);

                /* Download GPU results AFTER L2norm, repeat, deltanet, gated norm */
                float *gpu_Q_exp = (float *)malloc(dt_rank * d_state * sizeof(float));
                float *gpu_K_exp = (float *)malloc(dt_rank * d_state * sizeof(float));
                float *gpu_dnet_out = (float *)malloc(dt_rank * d_state * sizeof(float));
                float *gpu_gated_final = (float *)malloc(dt_rank * d_state * sizeof(float));
                float *norm_w = (float *)malloc(d_state * sizeof(float));

                cuMemcpyDtoH(gpu_Q_exp, r->d_ssm_Q_exp, dt_rank * d_state * sizeof(float));
                cuMemcpyDtoH(gpu_K_exp, r->d_ssm_K_exp, dt_rank * d_state * sizeof(float));
                /* Note: d_ssm_out was overwritten by gated norm, so download from before that */
                /* Actually we need the delta-net output before gated norm. But it's already overwritten. */
                /* Let's just check the final gated output against full CPU pipeline */
                cuMemcpyDtoH(gpu_gated_final, r->d_ssm_out, dt_rank * d_state * sizeof(float));
                cuMemcpyDtoH(norm_w, cl->ssm_norm_w, d_state * sizeof(float));

                /* CPU L2 norm per head for Q and K (from saved conv_out) */
                float *cpu_q = (float *)malloc(n_group * d_state * sizeof(float));
                float *cpu_k = (float *)malloc(n_group * d_state * sizeof(float));
                memcpy(cpu_q, xchk_conv_out, n_group * d_state * sizeof(float));
                memcpy(cpu_k, xchk_conv_out + n_group * d_state, n_group * d_state * sizeof(float));
                for (int h = 0; h < n_group; h++) {
                    float sq = 0;
                    for (int i = 0; i < d_state; i++) sq += cpu_q[h * d_state + i] * cpu_q[h * d_state + i];
                    float inv = 1.0f / sqrtf(sq + eps);
                    for (int i = 0; i < d_state; i++) cpu_q[h * d_state + i] *= inv;
                    sq = 0;
                    for (int i = 0; i < d_state; i++) sq += cpu_k[h * d_state + i] * cpu_k[h * d_state + i];
                    inv = 1.0f / sqrtf(sq + eps);
                    for (int i = 0; i < d_state; i++) cpu_k[h * d_state + i] *= inv;
                }

                /* CPU repeat/tile */
                float *cpu_Q_exp = (float *)malloc(dt_rank * d_state * sizeof(float));
                float *cpu_K_exp = (float *)malloc(dt_rank * d_state * sizeof(float));
                for (int h = 0; h < dt_rank; h++) {
                    memcpy(cpu_Q_exp + h * d_state, cpu_q + (h % n_group) * d_state, d_state * sizeof(float));
                    memcpy(cpu_K_exp + h * d_state, cpu_k + (h % n_group) * d_state, d_state * sizeof(float));
                }

                /* Compare Q_exp */
                float qexp_max_err = 0; int qexp_max_idx = 0;
                for (int i = 0; i < dt_rank * d_state; i++) {
                    float err = fabsf(cpu_Q_exp[i] - gpu_Q_exp[i]);
                    if (err > qexp_max_err) { qexp_max_err = err; qexp_max_idx = i; }
                }
                fprintf(stderr, "  [L2norm+repeat Q] max_err=%.6f at idx=%d (CPU=%.6f GPU=%.6f)\n",
                        qexp_max_err, qexp_max_idx, cpu_Q_exp[qexp_max_idx], gpu_Q_exp[qexp_max_idx]);
                fprintf(stderr, "    CPU_Q[0:4]=[%.6f,%.6f,%.6f,%.6f] GPU_Q[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                        cpu_Q_exp[0],cpu_Q_exp[1],cpu_Q_exp[2],cpu_Q_exp[3],
                        gpu_Q_exp[0],gpu_Q_exp[1],gpu_Q_exp[2],gpu_Q_exp[3]);

                /* Compare K_exp */
                float kexp_max_err = 0; int kexp_max_idx = 0;
                for (int i = 0; i < dt_rank * d_state; i++) {
                    float err = fabsf(cpu_K_exp[i] - gpu_K_exp[i]);
                    if (err > kexp_max_err) { kexp_max_err = err; kexp_max_idx = i; }
                }
                fprintf(stderr, "  [L2norm+repeat K] max_err=%.6f at idx=%d (CPU=%.6f GPU=%.6f)\n",
                        kexp_max_err, kexp_max_idx, cpu_K_exp[kexp_max_idx], gpu_K_exp[kexp_max_idx]);

                /* V is at conv_out[2*n_group*d_state ..] */
                float *cpu_V = xchk_conv_out + 2 * n_group * d_state;

                /* CPU delta-net for first token (zero state) */
                /* With S^T convention: S^T += outer(delta, k), output = S^T @ q / sqrt(d) */
                /* At pos 0 (zero state): delta = v*beta, S^T[r][c] = delta[r]*k[c] = v[r]*beta*k[c],
                   o[r] = sum_c(S^T[r][c]*q[c]) = v[r]*beta*(k.q)/sqrt(d) */
                float *cpu_dnet_out = (float *)calloc(dt_rank * d_state, sizeof(float));
                float scale_dn = 1.0f / sqrtf((float)d_state);
                if (position == 0) {
                    for (int h = 0; h < dt_rank; h++) {
                        float *q_h = cpu_Q_exp + h * d_state;
                        float *k_h = cpu_K_exp + h * d_state;
                        float b = xchk_beta[h];
                        float kq_dot = 0;
                        for (int c = 0; c < d_state; c++) kq_dot += k_h[c] * q_h[c];
                        for (int ri = 0; ri < d_state; ri++) {
                            float v_r = cpu_V[h * d_state + ri];
                            cpu_dnet_out[h * d_state + ri] = v_r * b * kq_dot * scale_dn;
                        }
                    }
                }

                /* CPU gated rmsnorm+silu */
                float *cpu_gated = (float *)malloc(dt_rank * d_state * sizeof(float));
                for (int h = 0; h < dt_rank; h++) {
                    float *o = cpu_dnet_out + h * d_state;
                    float *zh = xchk_z + h * d_state;
                    float ss = 0;
                    for (int i = 0; i < d_state; i++) ss += o[i] * o[i];
                    float sc = 1.0f / sqrtf(ss / (float)d_state + eps);
                    for (int i = 0; i < d_state; i++) {
                        float normed = o[i] * sc * norm_w[i];
                        float zv = zh[i];
                        cpu_gated[h * d_state + i] = normed * (zv / (1.0f + expf(-zv)));
                    }
                }

                /* Compare gated output */
                float gated_max_err = 0; int gated_max_idx = 0;
                for (int i = 0; i < dt_rank * d_state; i++) {
                    float err = fabsf(cpu_gated[i] - gpu_gated_final[i]);
                    if (err > gated_max_err) { gated_max_err = err; gated_max_idx = i; }
                }
                fprintf(stderr, "  [Gated out] max_err=%.6f at idx=%d (CPU=%.6f GPU=%.6f)\n",
                        gated_max_err, gated_max_idx, cpu_gated[gated_max_idx], gpu_gated_final[gated_max_idx]);
                fprintf(stderr, "    CPU[0:4]=[%.6f,%.6f,%.6f,%.6f] GPU[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                        cpu_gated[0],cpu_gated[1],cpu_gated[2],cpu_gated[3],
                        gpu_gated_final[0],gpu_gated_final[1],gpu_gated_final[2],gpu_gated_final[3]);
                float gn = 0;
                for (int i = 0; i < dt_rank * d_state; i++) gn += gpu_gated_final[i] * gpu_gated_final[i];
                fprintf(stderr, "  [Gated out] GPU L2 norm = %.6f\n", sqrtf(gn));

                /* Print alpha and beta for reference */
                fprintf(stderr, "    alpha[0:4]=[%.6f,%.6f,%.6f,%.6f] beta[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                        xchk_alpha[0],xchk_alpha[1],xchk_alpha[2],xchk_alpha[3],
                        xchk_beta[0],xchk_beta[1],xchk_beta[2],xchk_beta[3]);

                free(gpu_Q_exp); free(gpu_K_exp); free(gpu_dnet_out); free(gpu_gated_final);
                free(norm_w); free(cpu_q); free(cpu_k);
                free(cpu_Q_exp); free(cpu_K_exp); free(cpu_dnet_out); free(cpu_gated);
                free(xchk_conv_out); free(xchk_qkv); free(xchk_alpha); free(xchk_beta);
                free(xchk_z); free(xchk_conv_w);
                xchk_conv_out = NULL; /* prevent double free */
            }

            /* 9. Output projection: xb = ssm_out @ out */
            launch_matvec_auto(r, r->d_xb, cl->ssm_out_w, r->d_ssm_out,
                              cl->ssm_out_rows, cl->ssm_out_cols, cl->ssm_out_type);

        } else if (r->is_hybrid) {
            /* === Gated attention layer (Qwen3.5) === */
            /* Q+gate combined projection into xb2 */
            launch_matvec_auto(r, r->d_xb2, cl->attn_q_w, r->d_xb,
                              cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);

            /* De-interleave Q and gate */
            launch_deinterleave_qgate(r, r->d_q, r->d_attn_gate, r->d_xb2, n_heads, head_dim);

            if (r->debug_layers >= 2 && l == 3) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_q, 8 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] Q_deint[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                cuMemcpyDtoH(dbg, r->d_attn_gate, 8 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] gate[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }

            /* K, V projections */
            launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb, cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
            launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb, cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);

            if (r->debug_layers >= 2 && l == 3) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_k, 8 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] K[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                cuMemcpyDtoH(dbg, r->d_v, 8 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] V[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }

            /* QK-Norm */
            if (cl->has_qk_norm) {
                if (cl->attn_q_norm_w) launch_qknorm(r, r->d_q, cl->attn_q_norm_w, n_heads, head_dim, eps);
                if (cl->attn_k_norm_w) launch_qknorm(r, r->d_k, cl->attn_k_norm_w, n_kv_heads, head_dim, eps);
            }

            if (r->debug_layers >= 2 && l == 3) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_q, 8 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] Q_normed[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                cuMemcpyDtoH(dbg, r->d_k, 8 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] K_normed[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }

            /* RoPE */
            launch_rope(r, r->d_q, n_heads, head_dim, position, r->rope_freq_base);
            launch_rope(r, r->d_k, n_kv_heads, head_dim, position, r->rope_freq_base);

            if (r->debug_layers >= 2 && l == 3) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_q, 8 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] Q_rope[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                /* Also dump Q[32] and Q[64] to verify RoPE pairing */
                float dbg2[4];
                cuMemcpyDtoH(dbg2, r->d_q + 32 * sizeof(float), 4 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] Q_rope[32:36]=[%.6f,%.6f,%.6f,%.6f]\n",
                        dbg2[0],dbg2[1],dbg2[2],dbg2[3]);
                cuMemcpyDtoH(dbg2, r->d_q + 64 * sizeof(float), 4 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] Q_rope[64:68]=[%.6f,%.6f,%.6f,%.6f] (should be unchanged by RoPE)\n",
                        dbg2[0],dbg2[1],dbg2[2],dbg2[3]);
            }

            /* KV cache */
            launch_kv_store(r, r->d_key_cache[l], r->d_value_cache[l],
                           r->d_k, r->d_v, position, kv_dim);

            /* Attention decode */
            int seq_len = position + 1;
            float scale = 1.0f / sqrtf((float)head_dim);
            launch_attention(r, r->d_xb2, r->d_q,
                           r->d_key_cache[l], r->d_value_cache[l],
                           n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale);

            if (r->debug_layers >= 2 && l == 3) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_xb2, 8 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] attn_out[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }

            /* Attention cross-check (debug_layers >= 3, first attention layer at position 0) */
            if (r->debug_layers >= 3 && !cl->is_ssm && position == 0) {
                cuStreamSynchronize(r->stream);
                int q_dim_total = n_heads * head_dim;
                int gqa_ratio = n_heads / n_kv_heads;

                /* At position 0, seq_len=1, softmax=[1.0], attention output = V with GQA repeat */
                float *gpu_v = (float *)malloc(kv_dim * sizeof(float));
                float *gpu_attn_out = (float *)malloc(q_dim_total * sizeof(float));
                float *gpu_q = (float *)malloc(q_dim_total * sizeof(float));
                float *gpu_k = (float *)malloc(kv_dim * sizeof(float));
                float *gpu_gate = (float *)malloc(q_dim_total * sizeof(float));

                cuMemcpyDtoH(gpu_v, r->d_v, kv_dim * sizeof(float));
                cuMemcpyDtoH(gpu_attn_out, r->d_xb2, q_dim_total * sizeof(float));
                cuMemcpyDtoH(gpu_q, r->d_q, q_dim_total * sizeof(float));
                cuMemcpyDtoH(gpu_k, r->d_k, kv_dim * sizeof(float));
                cuMemcpyDtoH(gpu_gate, r->d_attn_gate, q_dim_total * sizeof(float));

                /* Read V from KV cache at position 0 to verify it was stored correctly */
                float *cache_v = (float *)malloc(kv_dim * sizeof(float));
                cuMemcpyDtoH(cache_v, r->d_value_cache[l], kv_dim * sizeof(float));

                /* Check V was stored correctly in cache */
                float v_cache_err = 0;
                for (int i = 0; i < kv_dim; i++) {
                    float err = fabsf(gpu_v[i] - cache_v[i]);
                    if (err > v_cache_err) v_cache_err = err;
                }
                fprintf(stderr, "  [L%d ATT xchk pos=0] V cache store err: %.6f\n", l, v_cache_err);

                /* Check attention output == V (with GQA repeat) */
                float attn_v_max_err = 0; int attn_v_max_idx = 0;
                for (int h = 0; h < n_heads; h++) {
                    int kv_h = h / gqa_ratio;
                    for (int d = 0; d < head_dim; d++) {
                        float expected = cache_v[kv_h * head_dim + d];
                        float actual = gpu_attn_out[h * head_dim + d];
                        float err = fabsf(expected - actual);
                        if (err > attn_v_max_err) { attn_v_max_err = err; attn_v_max_idx = h * head_dim + d; }
                    }
                }
                fprintf(stderr, "  [L%d ATT xchk pos=0] attn_out==V? max_err=%.6f at idx=%d\n",
                        l, attn_v_max_err, attn_v_max_idx);

                if (attn_v_max_err > 0.001f) {
                    fprintf(stderr, "    V[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                            gpu_v[0], gpu_v[1], gpu_v[2], gpu_v[3]);
                    fprintf(stderr, "    cache_V[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                            cache_v[0], cache_v[1], cache_v[2], cache_v[3]);
                    fprintf(stderr, "    attn_out[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                            gpu_attn_out[0], gpu_attn_out[1], gpu_attn_out[2], gpu_attn_out[3]);
                }

                /* Verify QK dot product at position 0: should be Q·K * scale */
                /* With softmax of single element = 1.0, this doesn't affect output but verify for sanity */
                float qk_dot_h0 = 0;
                for (int d = 0; d < head_dim; d++) qk_dot_h0 += gpu_q[d] * gpu_k[d];
                fprintf(stderr, "  [L%d ATT xchk] QK_dot_h0=%.6f (scale=%.6f, scaled=%.6f)\n",
                        l, qk_dot_h0, scale, qk_dot_h0 * scale);

                /* Print V, Q norms for sanity */
                float v_norm = 0, q_norm = 0, k_norm = 0;
                for (int d = 0; d < head_dim; d++) {
                    v_norm += gpu_v[d] * gpu_v[d];
                    q_norm += gpu_q[d] * gpu_q[d];
                    k_norm += gpu_k[d] * gpu_k[d];
                }
                fprintf(stderr, "  [L%d ATT xchk] |Q_h0|=%.4f |K_h0|=%.4f |V_h0|=%.4f\n",
                        l, sqrtf(q_norm), sqrtf(k_norm), sqrtf(v_norm));

                free(gpu_v); free(gpu_attn_out); free(gpu_q); free(gpu_k); free(gpu_gate);
                free(cache_v);
            }

            /* Apply sigmoid gate */
            int q_dim_local = n_heads * head_dim;
            launch_sigmoid_mul(r, r->d_xb2, r->d_attn_gate, q_dim_local);

            if (r->debug_layers >= 2 && l == 3) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_xb2, 8 * sizeof(float));
                fprintf(stderr, "  [L3 ATT DBG] gated_out[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }

            /* Output projection */
            launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                              cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);

        } else {
            /* === Standard attention (non-hybrid) === */
            /* Q/K/V projections (auto-dispatch) */
            launch_matvec_auto(r, r->d_q, cl->attn_q_w, r->d_xb, cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);
            launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb, cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
            launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb, cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);

            /* QK-Norm (if present) */
            if (cl->has_qk_norm) {
                if (cl->attn_q_norm_w) launch_qknorm(r, r->d_q, cl->attn_q_norm_w, n_heads, head_dim, eps);
                if (cl->attn_k_norm_w) launch_qknorm(r, r->d_k, cl->attn_k_norm_w, n_kv_heads, head_dim, eps);
            }

            /* RoPE on Q and K */
            launch_rope(r, r->d_q, n_heads, head_dim, position, r->rope_freq_base);
            launch_rope(r, r->d_k, n_kv_heads, head_dim, position, r->rope_freq_base);

            /* Store K,V to cache */
            launch_kv_store(r, r->d_key_cache[l], r->d_value_cache[l],
                           r->d_k, r->d_v, position, kv_dim);

            /* Attention decode */
            int seq_len = position + 1;
            float scale = 1.0f / sqrtf((float)head_dim);
            launch_attention(r, r->d_xb2, r->d_q,
                           r->d_key_cache[l], r->d_value_cache[l],
                           n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale);

            /* Output projection */
            launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                              cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);
        }

        /* Residual: x += xb */
        launch_add(r, r->d_x, r->d_xb, n_embd);

        /* Debug: post-attention norm (before FFN) */
        if (r->debug_layers >= 2) {
            cuStreamSynchronize(r->stream);
            float dbg[4];
            cuMemcpyDtoH(dbg, r->d_xb, 4 * sizeof(float));
            float xb_buf[512];
            int nrd = n_embd < 512 ? n_embd : 512;
            cuMemcpyDtoH(xb_buf, r->d_xb, nrd * sizeof(float));
            float ss2 = 0;
            for (int qi = 0; qi < nrd; qi++) ss2 += xb_buf[qi] * xb_buf[qi];
            fprintf(stderr, "  [L%02d %s] attn_out_norm~%.2f [%.4f,%.4f,%.4f,%.4f]\n",
                    l, cl->is_ssm ? "SSM" : "ATT", sqrtf(ss2*(float)n_embd/(float)nrd),
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }

        /* FFN RMSNorm: xb = rmsnorm(x, ffn_norm) */
        launch_rmsnorm(r, r->d_xb, r->d_x, cl->ffn_norm_w, n_embd, eps);

        if (cl->is_moe) {
            /* ---- MoE FFN ---- */
            int n_experts = r->n_experts;
            int n_experts_used = r->n_experts_used;
            int expert_ff = r->expert_ff;
            int shared_expert_ff = r->shared_expert_ff;

            /* 1. Router: logits = gate_inp @ xb → [n_experts] (F32 matvec) */
            launch_matvec_f32(r, r->d_router_logits, cl->moe_gate_w, r->d_xb,
                             cl->moe_gate_rows, cl->moe_gate_cols);

            /* 2. Top-k on CPU (sync + download n_experts floats → pick top-k + softmax) */
            cuStreamSynchronize(r->stream);
            cuMemcpyDtoH(r->h_router_logits, r->d_router_logits, n_experts * sizeof(float));
            int top_k_idx[64];  /* max 64 experts selected */
            float top_k_weights[64];
            moe_topk_softmax(r->h_router_logits, n_experts, n_experts_used, top_k_idx, top_k_weights);

            /* MoE debug helper macro */
            #define MOE_DBG_NORM(label, dptr, sz) do { \
                if (r->debug_layers >= 2 && l == 0) { \
                    cuStreamSynchronize(r->stream); \
                    float _dbg[8]; int _dn = (sz) < 8 ? (sz) : 8; \
                    cuMemcpyDtoH(_dbg, (dptr), _dn * sizeof(float)); \
                    float _s = 0; for (int _i = 0; _i < _dn; _i++) _s += _dbg[_i]*_dbg[_i]; \
                    fprintf(stderr, "  [L0 MoE] %s norm~%.4f first=[%.6f,%.6f,%.6f,%.6f]\n", \
                            (label), sqrtf(_s*(float)(sz)/(float)_dn), _dbg[0],_dbg[1],_dbg[2],_dbg[3]); \
                } } while(0)

            /* Debug: routing info */
            if (r->debug_layers >= 2 && l == 0) {
                fprintf(stderr, "  [L0 MoE] router top-k:");
                for (int e = 0; e < n_experts_used; e++)
                    fprintf(stderr, " e%d(%.4f)", top_k_idx[e], top_k_weights[e]);
                fprintf(stderr, "\n");
            }

            /* 3. Zero accumulator */
            cuMemsetD32(r->d_moe_accum, 0, n_embd);

            /* 4. For each selected expert */
            for (int e = 0; e < n_experts_used; e++) {
                int eidx = top_k_idx[e];
                CUdeviceptr gate_w = cl->moe_gate_exps_w + (size_t)eidx * cl->moe_exp_stride_gu;
                CUdeviceptr up_w   = cl->moe_up_exps_w   + (size_t)eidx * cl->moe_exp_stride_gu;
                CUdeviceptr down_w = cl->moe_down_exps_w  + (size_t)eidx * cl->moe_exp_stride_d;

                /* gate = expert_gate @ xb, up = expert_up @ xb */
                launch_matvec_auto(r, r->d_gate, gate_w, r->d_xb,
                                  cl->moe_exp_rows_gu, cl->moe_exp_cols_gu, cl->moe_gate_exps_type);
                launch_matvec_auto(r, r->d_up, up_w, r->d_xb,
                                  cl->moe_exp_rows_gu, cl->moe_exp_cols_gu, cl->moe_up_exps_type);

                if (r->debug_layers >= 2 && l == 0 && e == 0) {
                    MOE_DBG_NORM("e0 gate_out", r->d_gate, expert_ff);
                    MOE_DBG_NORM("e0 up_out", r->d_up, expert_ff);

                    /* CPU cross-check: verify Q2_K expert matvec for expert 0 */
                    {
                        cuStreamSynchronize(r->stream);
                        /* Download FFN input (d_xb = post-norm) */
                        float *cpu_xb = (float *)malloc(n_embd * sizeof(float));
                        cuMemcpyDtoH(cpu_xb, r->d_xb, n_embd * sizeof(float));
                        /* Download GPU gate output for comparison */
                        float *gpu_gate_out = (float *)malloc(expert_ff * sizeof(float));
                        cuMemcpyDtoH(gpu_gate_out, r->d_gate, expert_ff * sizeof(float));
                        /* Download expert gate weight raw bytes from GPU */
                        size_t exp_bytes = cl->moe_exp_stride_gu;
                        unsigned char *gate_raw = (unsigned char *)malloc(exp_bytes);
                        cuMemcpyDtoH(gate_raw, gate_w, exp_bytes);
                        /* CPU dequant + matvec for a few rows */
                        int exp_cols = cl->moe_exp_cols_gu;
                        int exp_rows = cl->moe_exp_rows_gu;
                        int row_bytes = (int)dequant_row_size(cl->moe_gate_exps_type, exp_cols);
                        float *dqbuf = (float *)malloc(exp_cols * sizeof(float));
                        fprintf(stderr, "  [L0 MoE CPU cross-check] exp type=%d rows=%d cols=%d stride=%zu row_bytes=%d\n",
                                cl->moe_gate_exps_type, exp_rows, exp_cols, exp_bytes, row_bytes);
                        fprintf(stderr, "  [L0 MoE CPU cross-check] xb_input first=[%.6f,%.6f,%.6f,%.6f]\n",
                                cpu_xb[0], cpu_xb[1], cpu_xb[2], cpu_xb[3]);
                        int check_rows[] = {0, 1, 10, 100, exp_rows/2, exp_rows-1};
                        float max_err = 0;
                        for (int ci = 0; ci < 6; ci++) {
                            int row = check_rows[ci];
                            if (row >= exp_rows) continue;
                            dequant_row(cl->moe_gate_exps_type, gate_raw + (size_t)row * row_bytes, dqbuf, exp_cols);
                            float cpu_dot = 0;
                            for (int j = 0; j < exp_cols; j++) cpu_dot += dqbuf[j] * cpu_xb[j];
                            float err = fabsf(cpu_dot - gpu_gate_out[row]);
                            if (err > max_err) max_err = err;
                            fprintf(stderr, "    row %3d: CPU=%.6f GPU=%.6f err=%.6f\n",
                                    row, cpu_dot, gpu_gate_out[row], err);
                        }
                        fprintf(stderr, "    max_err=%.6f\n", max_err);
                        free(cpu_xb); free(gpu_gate_out); free(gate_raw); free(dqbuf);
                    }
                }

                /* SiLU(gate) * up */
                launch_silu_mul(r, r->d_gate, r->d_up, expert_ff);

                /* down = expert_down @ gate → d_xb2 */
                launch_matvec_auto(r, r->d_xb2, down_w, r->d_gate,
                                  cl->moe_exp_rows_d, cl->moe_exp_cols_d, cl->moe_down_exps_type);

                if (r->debug_layers >= 2 && l == 0 && e == 0) {
                    MOE_DBG_NORM("e0 silu_mul", r->d_gate, expert_ff);
                    MOE_DBG_NORM("e0 down_out", r->d_xb2, n_embd);
                }

                /* Weighted accumulate: moe_accum += weight * xb2 */
                launch_scale_add(r, r->d_moe_accum, r->d_xb2, top_k_weights[e], n_embd);
            }

            MOE_DBG_NORM("experts_accum", r->d_moe_accum, n_embd);

            /* 5. Shared expert */
            {
                /* Compute gate scalar: sigmoid(dot(xb, shared_gate_w))
                 * shared_gate_w is [n_embd] F32, treated as [1, n_embd] matvec → [1] */
                launch_matvec_f32(r, r->d_router_logits, cl->moe_shared_gate_w, r->d_xb,
                                 1, n_embd);
                cuStreamSynchronize(r->stream);
                float gate_val;
                cuMemcpyDtoH(&gate_val, r->d_router_logits, sizeof(float));
                float shared_scale = 1.0f / (1.0f + expf(-gate_val));  /* sigmoid */

                if (r->debug_layers >= 2 && l == 0)
                    fprintf(stderr, "  [L0 MoE] shared_gate=%.6f sigmoid=%.6f\n", gate_val, shared_scale);

                /* Shared expert FFN: gate, up, silu_mul, down */
                launch_matvec_auto(r, r->d_gate, cl->moe_shared_ffn_gate_w, r->d_xb,
                                  cl->moe_shared_gate_rows, cl->moe_shared_gate_cols, cl->moe_shared_gate_type);
                launch_matvec_auto(r, r->d_up, cl->moe_shared_ffn_up_w, r->d_xb,
                                  cl->moe_shared_up_rows, cl->moe_shared_up_cols, cl->moe_shared_up_type);
                launch_silu_mul(r, r->d_gate, r->d_up, shared_expert_ff);
                launch_matvec_auto(r, r->d_xb2, cl->moe_shared_ffn_down_w, r->d_gate,
                                  cl->moe_shared_down_rows, cl->moe_shared_down_cols, cl->moe_shared_down_type);

                MOE_DBG_NORM("shared_down_out", r->d_xb2, n_embd);

                /* moe_accum += shared_scale * xb2 */
                launch_scale_add(r, r->d_moe_accum, r->d_xb2, shared_scale, n_embd);
            }

            MOE_DBG_NORM("final_moe_out", r->d_moe_accum, n_embd);
            #undef MOE_DBG_NORM

            /* 6. Copy accumulated result to xb for residual */
            cuMemcpyDtoDAsync(r->d_xb, r->d_moe_accum, n_embd * sizeof(float), r->stream);

        } else {
            /* ---- Dense FFN ---- */
            /* FFN gate and up projections */
            launch_matvec_auto(r, r->d_gate, cl->ffn_gate_w, r->d_xb,
                          cl->ffn_gate_rows, cl->ffn_gate_cols, cl->ffn_gate_type);
            launch_matvec_auto(r, r->d_up, cl->ffn_up_w, r->d_xb,
                          cl->ffn_up_rows, cl->ffn_up_cols, cl->ffn_up_type);

            /* SiLU(gate) * up */
            launch_silu_mul(r, r->d_gate, r->d_up, n_ff);

            /* FFN down projection: xb = ffn_down @ gate */
            launch_matvec_auto(r, r->d_xb, cl->ffn_down_w, r->d_gate,
                          cl->ffn_down_rows, cl->ffn_down_cols, cl->ffn_down_type);
        }

        /* CPU cross-check of FFN matvecs for layer 0 (debug_layers >= 3, dense only) */
        if (r->debug_layers >= 3 && l == 0 && !cl->is_moe) {
            cuStreamSynchronize(r->stream);

            /* Download GPU input (d_xb = RMSNorm output, input to FFN gate/up) */
            /* and GPU output (d_gate = after silu_mul, input to FFN down) */
            /* and GPU FFN down output (d_xb) */
            int nc_gate = cl->ffn_gate_cols;
            int nr_gate = cl->ffn_gate_rows;
            int nc_down = cl->ffn_down_cols;
            int nr_down = cl->ffn_down_rows;

            float *gpu_gate_input = (float *)malloc(nc_gate * sizeof(float));
            float *gpu_silu_mul_out = (float *)malloc(nc_down * sizeof(float));
            float *gpu_ffn_down_out = (float *)malloc(nr_down * sizeof(float));

            /* Read the input to FFN (d_xb was overwritten by ffn_down; read d_x - d_xb for residual) */
            /* Actually, d_xb now holds ffn_down output. Let's read d_gate (silu_mul output) */
            cuMemcpyDtoH(gpu_silu_mul_out, r->d_gate, nc_down * sizeof(float));
            cuMemcpyDtoH(gpu_ffn_down_out, r->d_xb, nr_down * sizeof(float));

            /* Download the raw weight data for ffn_down from GPU */
            size_t ffn_down_bytes = dequant_row_size(cl->ffn_down_type, nc_down);
            size_t total_weight_bytes = ffn_down_bytes * nr_down;
            unsigned char *ffn_down_raw = (unsigned char *)malloc(total_weight_bytes);
            cuMemcpyDtoH(ffn_down_raw, cl->ffn_down_w, total_weight_bytes);

            /* CPU cross-check: dequant+dot for a few rows */
            float *dequant_buf = (float *)malloc(nc_down * sizeof(float));
            fprintf(stderr, "\n  [L0 FFN cross-check] ffn_down type=%d [%d×%d]\n",
                    cl->ffn_down_type, nr_down, nc_down);
            fprintf(stderr, "    silu_mul_out[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                    gpu_silu_mul_out[0], gpu_silu_mul_out[1], gpu_silu_mul_out[2], gpu_silu_mul_out[3]);

            int check_rows[] = {0, 1, 100, 1000, 2000, 3000, 4000, nr_down-1};
            float max_err = 0;
            for (int ci = 0; ci < 8; ci++) {
                int row = check_rows[ci];
                if (row >= nr_down) continue;
                const void *row_data = ffn_down_raw + (size_t)row * ffn_down_bytes;
                dequant_row(cl->ffn_down_type, row_data, dequant_buf, nc_down);
                float cpu_dot = 0.0f;
                for (int j = 0; j < nc_down; j++) cpu_dot += dequant_buf[j] * gpu_silu_mul_out[j];
                float err = fabsf(cpu_dot - gpu_ffn_down_out[row]);
                if (err > max_err) max_err = err;
                fprintf(stderr, "    ffn_down row %4d: CPU=%.6f GPU=%.6f err=%.6f\n",
                        row, cpu_dot, gpu_ffn_down_out[row], err);
            }
            fprintf(stderr, "    max_err=%.6f\n", max_err);

            /* Also verify FFN gate (Q2_K) for a couple rows */
            {
                size_t gate_bytes = dequant_row_size(cl->ffn_gate_type, nc_gate);
                size_t total_gate = gate_bytes * nr_gate;
                unsigned char *gate_raw = (unsigned char *)malloc(total_gate);
                cuMemcpyDtoH(gate_raw, cl->ffn_gate_w, total_gate);

                /* Need the RMSNorm input to gate. We don't have it anymore since d_xb was overwritten.
                 * Skip this for now. */
                free(gate_raw);
            }

            /* Verify SSM out (Q4_K) */
            if (cl->is_ssm) {
                int nc_ssm_out = cl->ssm_out_cols;
                int nr_ssm_out = cl->ssm_out_rows;
                size_t ssm_out_bytes = dequant_row_size(cl->ssm_out_type, nc_ssm_out);

                unsigned char *ssm_out_raw = (unsigned char *)malloc(ssm_out_bytes * nr_ssm_out);
                cuMemcpyDtoH(ssm_out_raw, cl->ssm_out_w, ssm_out_bytes * nr_ssm_out);

                /* Download the ssm gated output (input to ssm_out) */
                float *ssm_gated = (float *)malloc(nc_ssm_out * sizeof(float));
                cuMemcpyDtoH(ssm_gated, r->d_ssm_out, nc_ssm_out * sizeof(float));

                /* Also download the d_xb which should hold the ssm_out matvec result
                 * But d_xb was overwritten by FFN. So this check only works if we save it earlier.
                 * Let's skip the full check and just verify dequant for a known input. */

                fprintf(stderr, "\n  [L0 SSM out cross-check] ssm_out type=%d [%d×%d]\n",
                        cl->ssm_out_type, nr_ssm_out, nc_ssm_out);
                fprintf(stderr, "    ssm_gated[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                        ssm_gated[0], ssm_gated[1], ssm_gated[2], ssm_gated[3]);

                float *deq = (float *)malloc(nc_ssm_out * sizeof(float));
                for (int ci = 0; ci < 5; ci++) {
                    int row = ci;
                    if (row >= nr_ssm_out) break;
                    const void *row_data = ssm_out_raw + (size_t)row * ssm_out_bytes;
                    dequant_row(cl->ssm_out_type, row_data, deq, nc_ssm_out);
                    float cpu_dot = 0.0f;
                    for (int j = 0; j < nc_ssm_out; j++) cpu_dot += deq[j] * ssm_gated[j];
                    fprintf(stderr, "    ssm_out row %4d: CPU=%.6f\n", row, cpu_dot);
                }
                free(deq);
                free(ssm_gated);
                free(ssm_out_raw);
            }

            free(gpu_gate_input);
            free(gpu_silu_mul_out);
            free(gpu_ffn_down_out);
            free(ffn_down_raw);
            free(dequant_buf);
        }

        /* Debug: FFN output norm for all layers */
        if (r->debug_layers >= 2 && position == 0) {
            cuStreamSynchronize(r->stream);
            float ffn_buf[512];
            int fn = n_embd < 512 ? n_embd : 512;
            cuMemcpyDtoH(ffn_buf, r->d_xb, fn * sizeof(float));
            float fs = 0;
            for (int fi = 0; fi < fn; fi++) fs += ffn_buf[fi] * ffn_buf[fi];
            fprintf(stderr, "  [L%02d %s] ffn_out_norm~%.2f\n",
                    l, cl->is_ssm ? "SSM" : "ATT", sqrtf(fs*(float)n_embd/(float)fn));
        }

        /* Residual: x += xb */
        launch_add(r, r->d_x, r->d_xb, n_embd);

        /* DeepStack injection: add deepstack slice after each early layer */
        if (r->_ds_embd && l < r->n_deepstack && r->_ds_embd_stride > n_embd) {
            const float *ds_slice = r->_ds_embd + (1 + l) * n_embd;
            cuMemcpyHtoDAsync(r->d_ds_tmp, (const void *)ds_slice, n_embd * sizeof(float), r->stream);
            launch_add(r, r->d_x, r->d_ds_tmp, n_embd);
        }

        /* Debug: print hidden state norm after each layer */
        if (r->debug_layers) {
            cuStreamSynchronize(r->stream);
            float *full_x = (float *)malloc(n_embd * sizeof(float));
            cuMemcpyDtoH(full_x, r->d_x, n_embd * sizeof(float));
            float ss = 0;
            for (int i = 0; i < n_embd; i++) ss += full_x[i] * full_x[i];
            float pnorm = sqrtf(ss);
            fprintf(stderr, "  [L%02d %s] norm=%.2f first=[%.4f, %.4f, %.4f, %.4f]\n",
                    l, (r->is_hybrid && cl->is_ssm) ? "SSM" : "ATT", pnorm, full_x[0], full_x[1], full_x[2], full_x[3]);
            /* Dump per-layer hidden state to binary file (debug_layers >= 4) */
            if (r->debug_layers >= 4) {
                char fname[256];
                snprintf(fname, sizeof(fname), "/tmp/cuda_layer_%02d_pos%d.bin", l, position);
                FILE *f = fopen(fname, "wb");
                if (f) { fwrite(full_x, sizeof(float), n_embd, f); fclose(f); }
            }
            free(full_x);
        }

    }

    /* Final RMSNorm */
    launch_rmsnorm(r, r->d_x, r->d_x, r->d_output_norm, n_embd, eps);

    /* Copy result to host */
    cuMemcpyDtoHAsync(r->h_output, r->d_x, n_embd * sizeof(float), r->stream);
    cuStreamSynchronize(r->stream);

    return r->h_output;
}

/* Forward pass returning logits [n_vocab] instead of hidden state */
float *cuda_llm_forward_logits(cuda_llm_runner *r, int32_t token_id, int position) {
    float *hidden = cuda_llm_forward(r, token_id, position);
    if (!hidden || !r->has_lm_head) return NULL;

    /* Copy hidden state back to GPU d_x (forward already did rmsnorm) */
    /* d_x already contains the normed hidden state on GPU — use it directly */

    /* LM head: logits = output_w @ hidden */
    launch_matvec_auto(r, r->d_logits, r->d_output_w, r->d_x,
                       r->n_vocab, r->n_embd, r->output_w_type);

    /* Copy logits to host */
    cuMemcpyDtoHAsync(r->h_output, r->d_logits, (size_t)r->n_vocab * sizeof(float), r->stream);
    cuStreamSynchronize(r->stream);

    return r->h_output;
}

/* Forward pass with pre-computed F32 embedding (for VLM vision token injection).
 * Uploads embd[0..n_embd-1] to d_x, then runs the same transformer blocks as
 * cuda_llm_forward(). Deepstack injection adds embd[(1+l)*n_embd .. (2+l)*n_embd-1]
 * after each early layer l < n_deepstack, if embd_stride > n_embd. */
float *cuda_llm_forward_embd(cuda_llm_runner *r, const float *embd, int embd_stride, int position) {
    if (!r || !r->weights_loaded || !embd) return NULL;
    if (position < 0 || position >= r->max_seq_len) return NULL;

    int n_embd = r->n_embd;

    /* Upload F32 embedding to d_x (first n_embd floats) */
    cuMemcpyHtoDAsync(r->d_x, (const void *)embd, n_embd * sizeof(float), r->stream);

    /* Set deepstack state for the forward loop */
    r->_ds_embd = embd;
    r->_ds_embd_stride = embd_stride;

    /* Run transformer blocks + final norm (same code path as cuda_llm_forward) */
    float *result = cuda_llm_forward_blocks(r, position);

    r->_ds_embd = NULL;
    r->_ds_embd_stride = 0;

    return result;
}

/* Forward pass with embedding returning logits */
float *cuda_llm_forward_embd_logits(cuda_llm_runner *r, const float *embd, int embd_stride, int position) {
    float *hidden = cuda_llm_forward_embd(r, embd, embd_stride, position);
    if (!hidden || !r->has_lm_head) return NULL;

    /* LM head: logits = output_w @ hidden (d_x still holds the normed state) */
    launch_matvec_auto(r, r->d_logits, r->d_output_w, r->d_x,
                       r->n_vocab, r->n_embd, r->output_w_type);

    /* Copy logits to host */
    cuMemcpyDtoHAsync(r->h_output, r->d_logits, (size_t)r->n_vocab * sizeof(float), r->stream);
    cuStreamSynchronize(r->stream);

    return r->h_output;
}

/* Read last hidden state (d_x) from GPU */
int cuda_llm_read_hidden(const cuda_llm_runner *r, float *dst, int n) {
    if (!r || !dst || n <= 0) return -1;
    CUresult err = cuMemcpyDtoH(dst, r->d_x, (size_t)n * sizeof(float));
    return (err == CUDA_SUCCESS) ? 0 : -1;
}

void cuda_llm_set_debug(cuda_llm_runner *r, int debug_layers) {
    if (r) r->debug_layers = debug_layers;
}

void cuda_llm_set_max_layers(cuda_llm_runner *r, int max_layers) {
    if (r) r->max_layers = max_layers;
}

/* ======================================================================== */
/* Public API: free                                                         */
/* ======================================================================== */

void cuda_llm_free(cuda_llm_runner *r) {
    if (!r) return;

    /* Free scratch */
    if (r->d_x)    cuMemFree(r->d_x);
    if (r->d_xb)   cuMemFree(r->d_xb);
    if (r->d_xb2)  cuMemFree(r->d_xb2);
    if (r->d_q)    cuMemFree(r->d_q);
    if (r->d_k)    cuMemFree(r->d_k);
    if (r->d_v)    cuMemFree(r->d_v);
    if (r->d_gate) cuMemFree(r->d_gate);
    if (r->d_up)   cuMemFree(r->d_up);
    if (r->d_ssm_qkv)     cuMemFree(r->d_ssm_qkv);
    if (r->d_ssm_z)        cuMemFree(r->d_ssm_z);
    if (r->d_ssm_alpha)    cuMemFree(r->d_ssm_alpha);
    if (r->d_ssm_beta)     cuMemFree(r->d_ssm_beta);
    if (r->d_ssm_Q_exp)    cuMemFree(r->d_ssm_Q_exp);
    if (r->d_ssm_K_exp)    cuMemFree(r->d_ssm_K_exp);
    if (r->d_ssm_out)      cuMemFree(r->d_ssm_out);
    if (r->d_ssm_conv_out) cuMemFree(r->d_ssm_conv_out);
    if (r->d_attn_gate)    cuMemFree(r->d_attn_gate);
    if (r->d_ds_tmp)   cuMemFree(r->d_ds_tmp);
    if (r->d_router_logits) cuMemFree(r->d_router_logits);
    if (r->d_moe_accum)    cuMemFree(r->d_moe_accum);
    free(r->h_router_logits);
    if (r->d_xb_q)    cuMemFree(r->d_xb_q);
    if (r->d_xb_scale) cuMemFree(r->d_xb_scale);

    /* Free KV cache */
    if (r->d_key_cache) {
        for (int l = 0; l < r->n_layers; l++) {
            if (r->d_key_cache[l]) cuMemFree(r->d_key_cache[l]);
        }
        free(r->d_key_cache);
    }
    if (r->d_value_cache) {
        for (int l = 0; l < r->n_layers; l++) {
            if (r->d_value_cache[l]) cuMemFree(r->d_value_cache[l]);
        }
        free(r->d_value_cache);
    }

    /* Free per-layer weights */
    if (r->layers) {
        for (int l = 0; l < r->n_layers; l++) {
            cuda_layer *cl = &r->layers[l];
            if (cl->attn_norm_w)    cuMemFree(cl->attn_norm_w);
            if (cl->attn_q_w)      cuMemFree(cl->attn_q_w);
            if (cl->attn_k_w)      cuMemFree(cl->attn_k_w);
            if (cl->attn_v_w)      cuMemFree(cl->attn_v_w);
            if (cl->attn_q_norm_w) cuMemFree(cl->attn_q_norm_w);
            if (cl->attn_k_norm_w) cuMemFree(cl->attn_k_norm_w);
            if (cl->attn_output_w) cuMemFree(cl->attn_output_w);
            if (cl->ffn_norm_w)    cuMemFree(cl->ffn_norm_w);
            if (cl->ffn_gate_w)    cuMemFree(cl->ffn_gate_w);
            if (cl->ffn_up_w)     cuMemFree(cl->ffn_up_w);
            if (cl->ffn_down_w)   cuMemFree(cl->ffn_down_w);
            /* SSM weights + state */
            if (cl->ssm_qkv_w)      cuMemFree(cl->ssm_qkv_w);
            if (cl->ssm_gate_w)     cuMemFree(cl->ssm_gate_w);
            if (cl->ssm_alpha_w)    cuMemFree(cl->ssm_alpha_w);
            if (cl->ssm_beta_w)     cuMemFree(cl->ssm_beta_w);
            if (cl->ssm_out_w)      cuMemFree(cl->ssm_out_w);
            if (cl->ssm_a)          cuMemFree(cl->ssm_a);
            if (cl->ssm_dt_bias)    cuMemFree(cl->ssm_dt_bias);
            if (cl->ssm_conv1d_w)   cuMemFree(cl->ssm_conv1d_w);
            if (cl->ssm_norm_w)     cuMemFree(cl->ssm_norm_w);
            if (cl->d_conv_state)      cuMemFree(cl->d_conv_state);
            if (cl->d_recurrent_state) cuMemFree(cl->d_recurrent_state);
            /* MoE weights */
            if (cl->moe_gate_w)             cuMemFree(cl->moe_gate_w);
            if (cl->moe_gate_exps_w)        cuMemFree(cl->moe_gate_exps_w);
            if (cl->moe_up_exps_w)          cuMemFree(cl->moe_up_exps_w);
            if (cl->moe_down_exps_w)        cuMemFree(cl->moe_down_exps_w);
            if (cl->moe_shared_gate_w)      cuMemFree(cl->moe_shared_gate_w);
            if (cl->moe_shared_ffn_gate_w)  cuMemFree(cl->moe_shared_ffn_gate_w);
            if (cl->moe_shared_ffn_up_w)    cuMemFree(cl->moe_shared_ffn_up_w);
            if (cl->moe_shared_ffn_down_w)  cuMemFree(cl->moe_shared_ffn_down_w);
        }
        free(r->layers);
    }

    /* Free global weights */
    if (r->d_token_embd) cuMemFree(r->d_token_embd);
    if (r->d_output_norm) cuMemFree(r->d_output_norm);
    if (r->d_output_w && r->d_output_w != r->d_token_embd) cuMemFree(r->d_output_w);
    if (r->d_logits) cuMemFree(r->d_logits);

    /* Free module */
    if (r->module) cuModuleUnload(r->module);

    /* Free host buffer */
    free(r->h_output);

    /* Destroy CUDA objects */
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->context) cuCtxDestroy(r->context);

    free(r);
}

/* ======================================================================== */
/* Public API: accessors                                                    */
/* ======================================================================== */

void cuda_llm_reset_state(cuda_llm_runner *r) {
    if (!r || !r->is_hybrid) return;
    for (int l = 0; l < r->n_layers; l++) {
        cuda_layer *cl = &r->layers[l];
        if (!cl->is_ssm) continue;
        if (cl->d_conv_state) {
            size_t conv_bytes = (size_t)(r->ssm_conv_kernel - 1) * r->ssm_qkv_dim * sizeof(float);
            cuMemsetD8(cl->d_conv_state, 0, conv_bytes);
        }
        if (cl->d_recurrent_state) {
            size_t rec_bytes = (size_t)r->ssm_dt_rank * r->ssm_d_state * r->ssm_d_state * sizeof(float);
            cuMemsetD8(cl->d_recurrent_state, 0, rec_bytes);
        }
    }
}

int cuda_llm_n_embd(const cuda_llm_runner *r) { return r ? r->n_embd : 0; }
int cuda_llm_n_layers(const cuda_llm_runner *r) { return r ? r->n_layers : 0; }
int cuda_llm_n_vocab(const cuda_llm_runner *r) { return r ? r->n_vocab : 0; }
int cuda_llm_max_seq_len(const cuda_llm_runner *r) { return r ? r->max_seq_len : 0; }
