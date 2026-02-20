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
"/* ---- 5. rope_neox_f32: NeoX-style RoPE, pairs (j, j+half_dim) ---- */\n"
"/* Grid: n_heads blocks, blockDim: half_dim threads */\n"
"__global__ void rope_neox_f32(float *vec, int n_heads, int head_dim, int pos,\n"
"                               float freq_base) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half_dim = head_dim / 2;\n"
"    if (j >= half_dim) return;\n"
"\n"
"    float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)head_dim);\n"
"    float theta = (float)pos * freq;\n"
"    float cos_t = cosf(theta);\n"
"    float sin_t = sinf(theta);\n"
"\n"
"    float *v = vec + h * head_dim;\n"
"    float v0 = v[j];\n"
"    float v1 = v[j + half_dim];\n"
"    v[j]            = v0 * cos_t - v1 * sin_t;\n"
"    v[j + half_dim] = v0 * sin_t + v1 * cos_t;\n"
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

    /* Weight types (GGML_TYPE_F16=1 or GGML_TYPE_Q8_0=8) */
    int attn_q_type, attn_k_type, attn_v_type, attn_output_type;
    int ffn_gate_type, ffn_up_type, ffn_down_type;
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
    float rms_norm_eps;

    /* GPU weights */
    int token_embd_type;        /* GGML_TYPE_F16 or GGML_TYPE_Q8_0 */
    CUdeviceptr d_token_embd;   /* F16 or Q8_0 [n_vocab * n_embd] */
    CUdeviceptr d_output_norm;  /* F32 [n_embd] */
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

    /* INT8 quantization scratch (for dp4a path) */
    CUdeviceptr d_xb_q;     /* INT8 [max_dim] */
    CUdeviceptr d_xb_scale; /* F32 [1] */

    /* Host output buffer */
    float *h_output;     /* [n_embd] */

    /* Weight loading state */
    int weights_loaded;
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

    #undef GET_FUNC

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: all 13 kernels compiled successfully\n");
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
    if (err != CUDA_SUCCESS) return -1;
    err = cuMemcpyHtoD(*d_ptr, t->data, nbytes);
    if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}

/* Dequant a 1D norm tensor to F32 and upload */
static int upload_norm_f32(CUdeviceptr *d_ptr, const qtensor *t, int n) {
    if (!t->data) { *d_ptr = 0; return 0; }
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) return -1;
    dequant_row(t->type, t->data, buf, n);
    CUresult err = cuMemAlloc(d_ptr, n * sizeof(float));
    if (err != CUDA_SUCCESS) { free(buf); return -1; }
    err = cuMemcpyHtoD(*d_ptr, buf, n * sizeof(float));
    free(buf);
    if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
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

/* Upload a weight matrix - dispatches F16 or Q8_0 based on type */
static int upload_weight_matrix(CUdeviceptr *d_ptr, const qtensor *t, int *out_type) {
    *out_type = t->type;
    if (t->type == GGML_TYPE_Q8_0) {
        return upload_q8_0_raw(d_ptr, t);
    } else {
        /* Default: F16 (or treat as F16) */
        return upload_f16_matrix(d_ptr, t);
    }
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
    if (gguf_find_key(gguf, "qwen3.block_count") >= 0) arch = "qwen3";
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

    int ctx_len = cllm_get_int(gguf, ARCH_KEY("context_length"), 0);
    if (max_seq_len <= 0) {
        max_seq_len = (ctx_len > 0) ? ctx_len : 1024;
    } else if (ctx_len > 0 && max_seq_len > ctx_len) {
        max_seq_len = ctx_len;
    }
    r->max_seq_len = max_seq_len;

    #undef ARCH_KEY

    int kv_dim = r->n_kv_heads * r->head_dim;
    int q_dim = r->n_heads * r->head_dim;

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: arch=%s n_embd=%d n_heads=%d n_kv_heads=%d n_layers=%d n_ff=%d head_dim=%d\n",
                arch, r->n_embd, r->n_heads, r->n_kv_heads, r->n_layers, r->n_ff, r->head_dim);
        fprintf(stderr, "cuda_llm: rope_freq_base=%.0f rms_norm_eps=%.1e max_seq_len=%d\n",
                r->rope_freq_base, r->rms_norm_eps, r->max_seq_len);
    }

    /* Token embeddings (F16 or Q8_0) */
    qtensor embd = cllm_load_tensor(gguf, "token_embd.weight", 1);
    if (!embd.data) return -1;
    if (r->n_vocab == 0) r->n_vocab = embd.n_rows;
    r->token_embd_type = embd.type;
    if (embd.type == GGML_TYPE_Q8_0) {
        if (upload_q8_0_raw(&r->d_token_embd, &embd) != 0) return -1;
    } else {
        if (upload_f16_matrix(&r->d_token_embd, &embd) != 0) return -1;
    }
    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: token_embd type=%s\n",
                embd.type == GGML_TYPE_Q8_0 ? "Q8_0" : "F16");
    }

    /* Output norm (F32) */
    qtensor onorm = cllm_load_tensor(gguf, "output_norm.weight", 1);
    if (!onorm.data) return -1;
    if (upload_norm_f32(&r->d_output_norm, &onorm, r->n_embd) != 0) return -1;

    /* Per-layer weights */
    r->layers = (cuda_layer *)calloc(r->n_layers, sizeof(cuda_layer));
    if (!r->layers) return -1;

    for (int l = 0; l < r->n_layers; l++) {
        char name[128];
        cuda_layer *cl = &r->layers[l];

        /* Attention norm (F32) */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        qtensor t = cllm_load_tensor(gguf, name, 1);
        if (upload_norm_f32(&cl->attn_norm_w, &t, r->n_embd) != 0) return -1;

        /* Q/K/V/Output projections (F16 or Q8_0) */
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

        /* FFN norm (F32) */
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        t = cllm_load_tensor(gguf, name, 1);
        if (upload_norm_f32(&cl->ffn_norm_w, &t, r->n_embd) != 0) return -1;

        /* FFN gate/up/down (F16 or Q8_0) */
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

        if (r->verbose >= 2) {
            fprintf(stderr, "  layer %d: Q[%d×%d] K[%d×%d] V[%d×%d] O[%d×%d] gate[%d×%d] up[%d×%d] down[%d×%d] qk_norm=%d\n",
                    l, cl->attn_q_rows, cl->attn_q_cols,
                    cl->attn_k_rows, cl->attn_k_cols,
                    cl->attn_v_rows, cl->attn_v_cols,
                    cl->attn_output_rows, cl->attn_output_cols,
                    cl->ffn_gate_rows, cl->ffn_gate_cols,
                    cl->ffn_up_rows, cl->ffn_up_cols,
                    cl->ffn_down_rows, cl->ffn_down_cols,
                    cl->has_qk_norm);
        }
    }

    /* Allocate KV cache */
    r->d_key_cache = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
    r->d_value_cache = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
    size_t kv_cache_size = (size_t)max_seq_len * kv_dim * sizeof(float);
    for (int l = 0; l < r->n_layers; l++) {
        CHECK_CU(cuMemAlloc(&r->d_key_cache[l], kv_cache_size));
        CHECK_CU(cuMemsetD8(r->d_key_cache[l], 0, kv_cache_size));
        CHECK_CU(cuMemAlloc(&r->d_value_cache[l], kv_cache_size));
        CHECK_CU(cuMemsetD8(r->d_value_cache[l], 0, kv_cache_size));
    }

    /* Allocate scratch buffers */
    int max_dim = r->n_embd;
    if (q_dim > max_dim) max_dim = q_dim;
    if (r->n_ff > max_dim) max_dim = r->n_ff;

    CHECK_CU(cuMemAlloc(&r->d_x,   max_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_xb,  max_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_xb2, max_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_q,   q_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_k,   kv_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_v,   kv_dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_gate, r->n_ff * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_up,   r->n_ff * sizeof(float)));

    /* INT8 quantization scratch (for dp4a path) */
    CHECK_CU(cuMemAlloc(&r->d_xb_q,     max_dim * sizeof(int8_t)));
    CHECK_CU(cuMemAlloc(&r->d_xb_scale, sizeof(float)));

    /* Host output buffer */
    r->h_output = (float *)malloc(r->n_embd * sizeof(float));
    if (!r->h_output) return -1;

    r->weights_loaded = 1;

    if (r->verbose >= 1) {
        /* Helper: bytes per element for a weight type */
        #define WEIGHT_BYTES(type, n_elements) \
            ((type) == GGML_TYPE_Q8_0 ? (size_t)((n_elements) / 32) * 36 : (size_t)(n_elements) * 2)

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
            weight_bytes += WEIGHT_BYTES(cl->ffn_gate_type, (size_t)cl->ffn_gate_rows * cl->ffn_gate_cols);
            weight_bytes += WEIGHT_BYTES(cl->ffn_up_type, (size_t)cl->ffn_up_rows * cl->ffn_up_cols);
            weight_bytes += WEIGHT_BYTES(cl->ffn_down_type, (size_t)cl->ffn_down_rows * cl->ffn_down_cols);
        }
        #undef WEIGHT_BYTES
        size_t cache_bytes = (size_t)r->n_layers * 2 * kv_cache_size;
        size_t scratch_bytes = (size_t)(max_dim * 3 + q_dim + kv_dim * 2 + r->n_ff * 2) * sizeof(float)
                             + max_dim + sizeof(float);  /* INT8 scratch */
        cuda_layer *cl0 = &r->layers[0];
        const char *wtype = (cl0->attn_q_type == GGML_TYPE_Q8_0) ? "Q8_0+dp4a" : "F16";
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
    void *args[] = { &vec, &n_heads, &head_dim, &pos, &freq_base };
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

/* Auto-dispatch matvec: Q8_0 uses dequant-to-F32 path (accurate), F16 uses direct path */
static inline void launch_matvec_auto(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                       CUdeviceptr x, int n_rows, int n_cols, int weight_type) {
    if (weight_type == GGML_TYPE_Q8_0) {
        launch_matvec_q8_f32(r, dst, mat, x, n_rows, n_cols);
    } else {
        launch_matvec(r, dst, mat, x, n_rows, n_cols);
    }
}

/* ======================================================================== */
/* Public API: forward                                                      */
/* ======================================================================== */

float *cuda_llm_forward(cuda_llm_runner *r, int32_t token_id, int position) {
    if (!r || !r->weights_loaded) return NULL;
    if (token_id < 0 || token_id >= r->n_vocab) return NULL;
    if (position < 0 || position >= r->max_seq_len) return NULL;

    int n_embd = r->n_embd;
    int n_heads = r->n_heads;
    int n_kv_heads = r->n_kv_heads;
    int head_dim = r->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int n_ff = r->n_ff;
    float eps = r->rms_norm_eps;

    /* 1. Token embedding lookup -> F32 */
    if (r->token_embd_type == GGML_TYPE_Q8_0) {
        launch_embed_q8_0(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else {
        launch_embed(r, r->d_x, r->d_token_embd, token_id, n_embd);
    }

    /* 2. Transformer blocks */
    for (int l = 0; l < r->n_layers; l++) {
        cuda_layer *cl = &r->layers[l];

        /* Attention RMSNorm: xb = rmsnorm(x, attn_norm) */
        launch_rmsnorm(r, r->d_xb, r->d_x, cl->attn_norm_w, n_embd, eps);

        /* Q/K/V projections (auto-dispatch F16 or Q8_0+dp4a) */
        launch_matvec_auto(r, r->d_q, cl->attn_q_w, r->d_xb, cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);
        launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb, cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
        launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb, cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);

        /* QK-Norm (if present) */
        if (cl->has_qk_norm) {
            if (cl->attn_q_norm_w)
                launch_qknorm(r, r->d_q, cl->attn_q_norm_w, n_heads, head_dim, eps);
            if (cl->attn_k_norm_w)
                launch_qknorm(r, r->d_k, cl->attn_k_norm_w, n_kv_heads, head_dim, eps);
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

        /* Output projection: xb = attn_output @ xb2 */
        launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                      cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);

        /* Residual: x += xb */
        launch_add(r, r->d_x, r->d_xb, n_embd);

        /* FFN RMSNorm: xb = rmsnorm(x, ffn_norm) */
        launch_rmsnorm(r, r->d_xb, r->d_x, cl->ffn_norm_w, n_embd, eps);

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

        /* Residual: x += xb */
        launch_add(r, r->d_x, r->d_xb, n_embd);
    }

    /* Final RMSNorm */
    launch_rmsnorm(r, r->d_x, r->d_x, r->d_output_norm, n_embd, eps);

    /* Copy result to host */
    cuMemcpyDtoHAsync(r->h_output, r->d_x, n_embd * sizeof(float), r->stream);
    cuStreamSynchronize(r->stream);

    return r->h_output;
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
        }
        free(r->layers);
    }

    /* Free global weights */
    if (r->d_token_embd) cuMemFree(r->d_token_embd);
    if (r->d_output_norm) cuMemFree(r->d_output_norm);

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

int cuda_llm_n_embd(const cuda_llm_runner *r) { return r ? r->n_embd : 0; }
int cuda_llm_n_layers(const cuda_llm_runner *r) { return r ? r->n_layers : 0; }
int cuda_llm_n_vocab(const cuda_llm_runner *r) { return r ? r->n_vocab : 0; }
int cuda_llm_max_seq_len(const cuda_llm_runner *r) { return r ? r->max_seq_len : 0; }
