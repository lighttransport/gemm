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

/* safetensors loader (header-only) for the Qwen3 text-encoder weight path.
 * When linked into a host TU that already carries the safetensors/json
 * implementation (e.g. diffusion-server's server.c), define
 * HIP_LLM_RUNNER_EXTERNAL_IMPLS=1 to use declarations only and avoid
 * multiple-definition link errors. The standalone rdna4/llm build leaves it
 * unset and owns the implementation here. */
#ifndef HIP_LLM_RUNNER_EXTERNAL_IMPLS
#define SAFETENSORS_IMPLEMENTATION
#endif
#include "../../common/safetensors.h"

/* hip_runner_common.h: HIP_CHECK, hip_f32_to_f16, hip_upload_raw, hip_compile_kernels */
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#ifdef LLM_HIPBLASLT_ENABLED
#include "mm_blaslt_bridge.h"
#else
/* No-hipBLASLt build: the batched prefill path stays fully functional via the
 * self-owned WMMA GEMM (gemm_bf16_own); these stubs make the blaslt branch
 * unreachable (init fails -> gemm_own forced on). */
static int mm_blaslt_init(void) { return -1; }
static int mm_blaslt_run_bf16(void *y, const void *w, const void *x,
                              int M, int N, int K, void *stream) {
    (void)y;(void)w;(void)x;(void)M;(void)N;(void)K;(void)stream; return -1;
}
static void mm_blaslt_destroy(void) {}
#define LLM_HIPBLASLT_ENABLED 1   /* compile the batched path unconditionally */
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
"    int NT  = blockDim.x;\n"
"    float *v = vec + h * head_dim;\n"
"    /* grid-stride so head_dim may exceed blockDim (Gemma4 full-attn head_dim=512). */\n"
"    float sum = 0.0f;\n"
"    for (int d = tid; d < head_dim; d += NT) { float x = v[d]; sum += x * x; }\n"
"    sdata[tid] = sum;\n"
"    __syncthreads();\n"
"    for (int s = NT / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);\n"
"    for (int d = tid; d < head_dim; d += NT) v[d] = v[d] * scale * w[d];\n"
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
"    int NT  = blockDim.x;\n"
"    float *v = vec + (size_t)row * row_stride + h * head_dim;\n"
"    /* grid-stride so head_dim may exceed blockDim (Gemma4 full-attn head_dim=512). */\n"
"    float sum = 0.0f;\n"
"    for (int d = tid; d < head_dim; d += NT) { float x = v[d]; sum += x * x; }\n"
"    sdata[tid] = sum;\n"
"    __syncthreads();\n"
"    for (int s = NT / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);\n"
"    for (int d = tid; d < head_dim; d += NT) v[d] = v[d] * scale * w[d];\n"
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
"/* ---- hardware DP4A on RDNA4 (gfx12): signed 4x int8 MAC ---- */\n"
"__device__ __forceinline__ int dp4a_hw(int a, int b, int c) {\n"
"    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);\n"
"}\n"
"/* Build a packed int8x4 from 4 uint8 grid magnitudes, negating byte k when\n"
"   sign bit (base+k) of sb is set (matches the scalar IQ sign convention). */\n"
"__device__ __forceinline__ int apply_sign4(unsigned int g, unsigned int sb, int base) {\n"
"    int r = 0; signed char *o = (signed char *)&r; const unsigned char *gb = (const unsigned char *)&g;\n"
"    o[0] = (sb & (1u << (base+0))) ? -(int)gb[0] : (int)gb[0];\n"
"    o[1] = (sb & (1u << (base+1))) ? -(int)gb[1] : (int)gb[1];\n"
"    o[2] = (sb & (1u << (base+2))) ? -(int)gb[2] : (int)gb[2];\n"
"    o[3] = (sb & (1u << (base+3))) ? -(int)gb[3] : (int)gb[3];\n"
"    return r;\n"
"}\n"
"/* ---- quantize_q8_32: F32 vector -> int8 per-32-block + fp32 scale ---- */\n"
"/* grid = n/32 blocks, blockDim 32 (one warp per 32-element block).          */\n"
"__global__ void quantize_q8_32(signed char *qs, float *scale, const float *x, int n) {\n"
"    int g = blockIdx.x; int lane = threadIdx.x; int idx = g * 32 + lane;\n"
"    float v = (idx < n) ? x[idx] : 0.0f;\n"
"    float a = fabsf(v);\n"
"    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_down(a, o));\n"
"    a = __shfl(a, 0);\n"
"    float inv = (a > 0.0f) ? 127.0f / a : 0.0f;\n"
"    if (lane == 0) scale[g] = a / 127.0f;\n"
"    if (idx < n) {\n"
"        int q = (int)rintf(v * inv);\n"
"        q = q > 127 ? 127 : (q < -127 ? -127 : q);\n"
"        qs[idx] = (signed char)q;\n"
"    }\n"
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
"/* Q2_K warp-per-row, G=nb*4: one lane per 64-element chunk (n0,half) -> 16 qs    */\n"
"/* bytes shared across all 4 scale-groups (no redundant reads), all 32 lanes      */\n"
"/* active, intra-warp reduction. 8 rows/block. Q2_K: scales[16]+qs[64]+d@80+dmin@82.*/\n"
"__global__ void matvec_q2_K_g4_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                     int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 84;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int G = nb * 4;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 2; int gi = g & 3; int n0 = gi >> 1; int half = gi & 1;\n"
"        const unsigned char *bp = row_ptr + b * 84;\n"
"        const unsigned char *scales = bp;\n"
"        const unsigned char *qsb = bp + 16 + n0 * 32 + half * 16;\n"
"        float d = half_to_float(*(const half_raw *)(bp + 80));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 82));\n"
"        const float *xb = x + b * 256 + n0 * 128 + half * 16;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            unsigned char sc = scales[n0 * 8 + j * 2 + half];\n"
"            float dl = d * (float)(sc & 0xF);\n"
"            float ml = dmin * (float)(sc >> 4);\n"
"            int shift = j * 2;\n"
"            const float *xj = xb + j * 32;\n"
"            for (int l = 0; l < 16; l++)\n"
"                partial += (dl * (float)((qsb[l] >> shift) & 3) - ml) * xj[l];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
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
"/* Q3_K warp-per-row, G=nb*4: one lane per 64-element chunk (n0,half), all 32     */\n"
"/* lanes active, intra-warp reduction. 8 rows/block. */\n"
"__global__ void matvec_q3_K_g4_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                     int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 110;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int G = nb * 4;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 2; int gi = g & 3; int n0 = gi >> 1; int half = gi & 1;\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        const unsigned char *hmb = bp + half * 16;\n"
"        const unsigned char *qsb = bp + 32 + n0 * 32 + half * 16;\n"
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
"        const float *xb = x + b * 256 + n0 * 128 + half * 16;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float dl = d_all * (float)((int)scales[n0 * 8 + j * 2 + half] - 32);\n"
"            int shift = j * 2;\n"
"            unsigned char m_bit = (unsigned char)(1u << (n0 * 4 + j));\n"
"            const float *xj = xb + j * 32;\n"
"            for (int l = 0; l < 16; l++) {\n"
"                int qv = ((qsb[l] >> shift) & 3) - ((hmb[l] & m_bit) ? 0 : 4);\n"
"                partial += dl * (float)qv * xj[l];\n"
"            }\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
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
"/* Q4_K warp-per-row, G=nb*4: one lane per 64-element chunk (32 qs bytes, low+   */\n"
"/* high nibbles), all 32 lanes active, no redundant qs reads. 8 rows/block.      */\n"
"__global__ void matvec_q4_K_g4_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                     int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 144;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int G = nb * 4;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 2; int c = g & 3; int is = 2 * c;\n"
"        const unsigned char *bp = row_ptr + b * 144;\n"
"        float d = half_to_float(*(const half_raw *)(bp));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *sc = bp + 4;\n"
"        unsigned char sv0, mv0, sv1, mv1;\n"
"        if (is < 4) { sv0 = sc[is] & 63; mv0 = sc[is+4] & 63; }\n"
"        else { sv0 = (sc[is+4]&0xF)|((sc[is-4]>>6)<<4); mv0 = (sc[is+4]>>4)|((sc[is]>>6)<<4); }\n"
"        if (is+1 < 4) { sv1 = sc[is+1] & 63; mv1 = sc[is+1+4] & 63; }\n"
"        else { sv1 = (sc[is+1+4]&0xF)|((sc[is+1-4]>>6)<<4); mv1 = (sc[is+1+4]>>4)|((sc[is+1]>>6)<<4); }\n"
"        float d1 = d * sv0, m1 = dmin * mv0;\n"
"        float d2 = d * sv1, m2 = dmin * mv1;\n"
"        const unsigned char *q = bp + 16 + c * 32;\n"
"        const float *xlo = x + b * 256 + c * 64;\n"
"        const float *xhi = xlo + 32;\n"
"        float partial = 0.0f;\n"
"        for (int l = 0; l < 32; l++)\n"
"            partial += (d1 * (q[l] & 0xF) - m1) * xlo[l] + (d2 * (q[l] >> 4) - m2) * xhi[l];\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- 17. matvec_q6_K_f32: Q6_K matrix x F32 vector -> F32 ---- */\n"
"/* Q6_K block: 210 bytes = ql[128] + qh[64] + scales[16] + d(f16), 256 elements */\n"
"/* Vectorized Q6_K row-dot: lane g covers 4 consecutive l of (block,half).   */\n"
"/* Loads ql/qh as dwords (memcpy, 2-aligned), x as 4x float4. */\n"
"__device__ __forceinline__ float q6k_dot4(const unsigned char *bp, const float *xb,\n"
"                                            int half, int lp) {\n"
"    const unsigned char *ql = bp + half * 64 + lp * 4;\n"
"    const unsigned char *qh = bp + 128 + half * 32 + lp * 4;\n"
"    const signed char *sc = (const signed char *)(bp + 192 + half * 8);\n"
"    float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"    unsigned int qlo, qhi, qhv;\n"
"    __builtin_memcpy(&qlo, ql, 4);\n"
"    __builtin_memcpy(&qhi, ql + 32, 4);\n"
"    __builtin_memcpy(&qhv, qh, 4);\n"
"    int hi = (lp < 4) ? 0 : 1;\n"
"    float sA = d * (float)sc[0 + hi], sB = d * (float)sc[2 + hi];\n"
"    float sC = d * (float)sc[4 + hi], sD = d * (float)sc[6 + hi];\n"
"    const float *x0 = xb + half * 128 + lp * 4;\n"
"    float4 xa = *(const float4 *)(x0);\n"
"    float4 xb2 = *(const float4 *)(x0 + 32);\n"
"    float4 xc = *(const float4 *)(x0 + 64);\n"
"    float4 xd = *(const float4 *)(x0 + 96);\n"
"    float acc = 0.0f;\n"
"    #pragma unroll\n"
"    for (int j = 0; j < 4; j++) {\n"
"        int lo = (qlo >> (8*j)) & 0xFF, hi8 = (qhi >> (8*j)) & 0xFF, h2 = (qhv >> (8*j)) & 0xFF;\n"
"        int q1 = (lo & 0xF) | ((h2 & 3) << 4); q1 -= 32;\n"
"        int q2 = (hi8 & 0xF) | (((h2 >> 2) & 3) << 4); q2 -= 32;\n"
"        int q3 = (lo >> 4) | (((h2 >> 4) & 3) << 4); q3 -= 32;\n"
"        int q4 = (hi8 >> 4) | (((h2 >> 6) & 3) << 4); q4 -= 32;\n"
"        float xj = (&xa.x)[j], xj32 = (&xb2.x)[j], xj64 = (&xc.x)[j], xj96 = (&xd.x)[j];\n"
"        acc += sA*q1*xj + sB*q2*xj32 + sC*q3*xj64 + sD*q4*xj96;\n"
"    }\n"
"    return acc;\n"
"}\n"
"/* Full thread utilization: groups of (block,half,l) -> 4 products each, strided */\n"
"/* across all nthreads. Old kernel strided nb blocks -> for small n_cols only nb */\n"
"/* (2-8) of 64 threads were active (LM head / shared-expert / attention).        */\n"
"__global__ void matvec_q6_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int G = nb * 16;  /* (block, half, lp(8)) groups, 16 products each */\n"
"    float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        sum += q6k_dot4(row_ptr + b * 210, x + b * 256, half, lp);\n"
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
"    /* G=nb*32 group decomposition: one lane per 8-element (sub-block,l) group,  */\n"
"    /* all 32 lanes active (was nb-stride: only nb of 32 lanes active).          */\n"
"    int G = nb * 32;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        unsigned int aux = *(const unsigned int *)(bp + 66 + 4*sb);\n"
"        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;\n"
"        unsigned char sgn = ksigns_iq2xs_dev[(aux >> (7*l)) & 127];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l]];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l+1]];\n"
"        const float *xb = x + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float w0 = db * (float)g1[j] * ((sgn & (1 << j)) ? -1.0f : 1.0f);\n"
"            float w1 = db * (float)g2[j] * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            sum += w0 * xb[j] + w1 * xb[j+4];\n"
"        }\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq2_s_f32: IQ2_S matrix x F32 vector -> F32 ---- */\n"
"/* Full 32-lane utilization: each lane processes 8-element groups (one (ib32,l)  */\n"
"/* pair = a single grid entry) striding by 32 over the row's nb*32 groups. Avoids */\n"
"/* the old nb-strided loop that left 24/32 lanes idle for small n_cols (experts). */\n"
"__global__ void matvec_iq2_s_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 82;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    /* block_iq2_s layout: d(2) + qs(32) + signs(32) + qh(8) + scales(8) */\n"
"    int G = nb * 32;  /* groups of 8 elements per row */\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 5; int gl = g & 31; int ib32 = gl >> 2; int l = gl & 3;\n"
"        const unsigned char *bp = row_ptr + b * 82;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char scale = bp[74 + ib32];\n"
"        float db = (l < 2) ? d * (0.5f + (float)(scale & 0xf)) * 0.25f\n"
"                           : d * (0.5f + (float)(scale >>  4)) * 0.25f;\n"
"        unsigned char qsl = bp[2 + ib32 * 4 + l];\n"
"        unsigned char qh  = bp[66 + ib32];\n"
"        int grid_idx = qsl | ((qh << (8 - 2 * l)) & 0x300);\n"
"        const unsigned char *grid = (const unsigned char *)&iq2s_grid_dev[grid_idx];\n"
"        unsigned char s = bp[34 + ib32 * 4 + l];\n"
"        const float *xb = x + b * 256 + ib32 * 32 + l * 8;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 8; j++)\n"
"            partial += db * (float)grid[j] * ((s & (1 << j)) ? -1.0f : 1.0f) * xb[j];\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq3_s_f32: IQ3_S matrix x F32 vector -> F32 ---- */\n"
"/* Full 32-lane utilization: one lane per 8-element (sub-block,l) group.        */\n"
"/* block_iq3_s layout: d(2) + qs(64) + qh(8) + signs(32) + scales(4)            */\n"
"__global__ void matvec_iq3_s_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                   int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 110;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int G = nb * 32;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 5; int g32 = g & 31; int sb = g32 >> 2; int l = g32 & 3;\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char sc = bp[106 + (sb >> 1)];\n"
"        float db = d * (float)(1 + 2 * ((sb & 1) ? (sc >> 4) : (sc & 0xf)));\n"
"        unsigned char qh = bp[66 + sb];\n"
"        unsigned char qs0 = bp[2 + sb * 8 + 2 * l + 0];\n"
"        unsigned char qs1 = bp[2 + sb * 8 + 2 * l + 1];\n"
"        const unsigned char *grid1 = (const unsigned char *)&iq3s_grid_dev[qs0 | ((qh << (8 - 2 * l)) & 256)];\n"
"        const unsigned char *grid2 = (const unsigned char *)&iq3s_grid_dev[qs1 | ((qh << (7 - 2 * l)) & 256)];\n"
"        unsigned char s = bp[74 + sb * 4 + l];\n"
"        const float *xb = x + b * 256 + sb * 32 + l * 8;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            partial += db * (float)grid1[j] * ((s & (1 << j)) ? -1.0f : 1.0f) * xb[j];\n"
"            partial += db * (float)grid2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f) * xb[j+4];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ======================================================================== */\n"
"/* MoE device-side dispatch kernels (sync-free, graph-capturable).          */\n"
"/* Expert matvec variants resolve the per-expert weight base on-device from  */\n"
"/* a device index buffer, so no host round-trip is needed for routing.       */\n"
"/* ======================================================================== */\n"
"\n"
"/* ---- moe_topk_softmax_gpu: top-k by logit, softmax over the k selected ---- */\n"
"/* One block, n threads. Parallel argmax reduction per round (k rounds), then  */\n"
"/* softmax over the k selected. Bit-identical selection to the host path        */\n"
"/* (ties broken by lowest index, matching the host '>' comparison).             */\n"
"__global__ void moe_topk_softmax_gpu(const float *logits, int n, int k,\n"
"                                       int *out_idx, float *out_w) {\n"
"    __shared__ float sval[256];\n"
"    __shared__ int   sidx[256];\n"
"    __shared__ int   chosen[64];\n"
"    int tid = threadIdx.x;\n"
"    float my = (tid < n) ? logits[tid] : -1e30f;\n"
"    for (int ki = 0; ki < k; ki++) {\n"
"        float v = my;\n"
"        /* mask already-selected indices */\n"
"        for (int j = 0; j < ki; j++) if (chosen[j] == tid) v = -1e30f;\n"
"        sval[tid] = v; sidx[tid] = tid;\n"
"        __syncthreads();\n"
"        for (int s = 128; s > 0; s >>= 1) {\n"
"            if (tid < s && tid + s < 256) {\n"
"                /* '>' so ties keep the lower index (smaller of the two) */\n"
"                if (sval[tid + s] > sval[tid]) { sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s]; }\n"
"                else if (sval[tid + s] == sval[tid] && sidx[tid + s] < sidx[tid]) { sidx[tid] = sidx[tid + s]; }\n"
"            }\n"
"            __syncthreads();\n"
"        }\n"
"        if (tid == 0) { chosen[ki] = sidx[0]; out_idx[ki] = sidx[0]; out_w[ki] = sval[0]; }\n"
"        __syncthreads();\n"
"    }\n"
"    if (tid == 0) {\n"
"        float mx = out_w[0];\n"
"        for (int i = 1; i < k; i++) if (out_w[i] > mx) mx = out_w[i];\n"
"        float sum = 0.0f;\n"
"        for (int i = 0; i < k; i++) { out_w[i] = expf(out_w[i] - mx); sum += out_w[i]; }\n"
"        float inv = 1.0f / sum;\n"
"        for (int i = 0; i < k; i++) out_w[i] *= inv;\n"
"    }\n"
"}\n"
"\n"
"/* ---- sigmoid_scalar_f32: out[0] = sigmoid(in[0]) (shared-expert gate) ---- */\n"
"__global__ void sigmoid_scalar_f32(float *out, const float *in) {\n"
"    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 1.0f / (1.0f + expf(-in[0]));\n"
"}\n"
"\n"
"/* ---- scale_add_dev_f32: dst += scale_ptr[slot] * src (device scalar) ---- */\n"
"__global__ void scale_add_dev_f32(float *dst, const float *src,\n"
"                                    const float *scale_ptr, int slot, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) dst[i] += scale_ptr[slot] * src[i];\n"
"}\n"
"/* ---- Batched token-grouped MoE: gather / scatter-accumulate / row-scale ---- */\n"
"/* gather: dst[i, :] = src[idx[i], :]  (assignment i pulls its token's row) */\n"
"__global__ void moe_gather_rows(float *dst, const float *src, const int *idx,\n"
"                                  int n_assign, int n_embd) {\n"
"    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"    long total = (long)n_assign * n_embd;\n"
"    if (t >= total) return;\n"
"    int i = t / n_embd; int j = t - (long)i * n_embd;\n"
"    dst[t] = src[(long)idx[i] * n_embd + j];\n"
"}\n"
"/* scatter+accumulate: dst[idx[i], :] += w[i] * src[i, :]  (atomic; a token gets\n"
"   one contribution per selected expert). */\n"
"__global__ void moe_scatter_accum(float *dst, const float *src, const int *idx,\n"
"                                    const float *w, int n_assign, int n_embd) {\n"
"    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"    long total = (long)n_assign * n_embd;\n"
"    if (t >= total) return;\n"
"    int i = t / n_embd; int j = t - (long)i * n_embd;\n"
"    atomicAdd(&dst[(long)idx[i] * n_embd + j], w[i] * src[t]);\n"
"}\n"
"/* row-scale add: dst[m, :] += scale[m] * src[m, :]  (shared-expert gating). */\n"
"__global__ void moe_row_scale_add(float *dst, const float *src, const float *scale,\n"
"                                    int M, int n_embd) {\n"
"    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"    long total = (long)M * n_embd;\n"
"    if (t >= total) return;\n"
"    int m = t / n_embd;\n"
"    dst[t] += scale[m] * src[t];\n"
"}\n"
"\n"
"/* ---- expert-indexed matvec variants: base resolved from eidx[slot] ----   */\n"
"__global__ void matvec_iq2_s_expert_f32(float *dst, const unsigned char *base, const float *x,\n"
"                                          int n_rows, int n_cols,\n"
"                                          const int *eidx, int slot, long long stride) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    const unsigned char *mat = base + (long long)eidx[slot] * stride;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 82;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int G = nb * 32;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 5; int gl = g & 31; int ib32 = gl >> 2; int l = gl & 3;\n"
"        const unsigned char *bp = row_ptr + b * 82;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char scale = bp[74 + ib32];\n"
"        float db = (l < 2) ? d * (0.5f + (float)(scale & 0xf)) * 0.25f\n"
"                           : d * (0.5f + (float)(scale >>  4)) * 0.25f;\n"
"        unsigned char qsl = bp[2 + ib32 * 4 + l];\n"
"        unsigned char qh  = bp[66 + ib32];\n"
"        int grid_idx = qsl | ((qh << (8 - 2 * l)) & 0x300);\n"
"        const unsigned char *grid = (const unsigned char *)&iq2s_grid_dev[grid_idx];\n"
"        unsigned char s = bp[34 + ib32 * 4 + l];\n"
"        const float *xb = x + b * 256 + ib32 * 32 + l * 8;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 8; j++)\n"
"            partial += db * (float)grid[j] * ((s & (1 << j)) ? -1.0f : 1.0f) * xb[j];\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"__global__ void matvec_iq3_s_expert_f32(float *dst, const unsigned char *base, const float *x,\n"
"                                          int n_rows, int n_cols,\n"
"                                          const int *eidx, int slot, long long stride) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    const unsigned char *mat = base + (long long)eidx[slot] * stride;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 110;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int G = nb * 32;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 5; int g32 = g & 31; int sb = g32 >> 2; int l = g32 & 3;\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char sc = bp[106 + (sb >> 1)];\n"
"        float db = d * (float)(1 + 2 * ((sb & 1) ? (sc >> 4) : (sc & 0xf)));\n"
"        unsigned char qh = bp[66 + sb];\n"
"        unsigned char qs0 = bp[2 + sb * 8 + 2 * l + 0];\n"
"        unsigned char qs1 = bp[2 + sb * 8 + 2 * l + 1];\n"
"        const unsigned char *grid1 = (const unsigned char *)&iq3s_grid_dev[qs0 | ((qh << (8 - 2 * l)) & 256)];\n"
"        const unsigned char *grid2 = (const unsigned char *)&iq3s_grid_dev[qs1 | ((qh << (7 - 2 * l)) & 256)];\n"
"        unsigned char s = bp[74 + sb * 4 + l];\n"
"        const float *xb = x + b * 256 + sb * 32 + l * 8;\n"
"        float partial = 0.0f;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            partial += db * (float)grid1[j] * ((s & (1 << j)) ? -1.0f : 1.0f) * xb[j];\n"
"            partial += db * (float)grid2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f) * xb[j+4];\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down(sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"__global__ void matvec_iq4_xs_expert_f32(float *dst, const unsigned char *base, const float *x,\n"
"                                           int n_rows, int n_cols,\n"
"                                           const int *eidx, int slot, long long stride) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    const unsigned char *mat = base + (long long)eidx[slot] * stride;\n"
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

"/* ======================================================================== */\n"
"/* DP4A decode matvecs: q8 per-32-block activation + hardware int8 dot.       */\n"
"/* Half-split: one lane per 16-element half so even nb=2 (down-proj) keeps     */\n"
"/* full 32-lane occupancy. xq/xs are the activation pre-quantized by           */\n"
"/* quantize_q8_32 (per-32-block scale). Grids fit int8 (iq2s<=43, iq3s<=15).   */\n"
"/* ======================================================================== */\n"
"#define IQ3S_DP4A_BODY \\\n"
"    int nb = n_cols / 256; int row_bytes = nb * 110; \\\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes; \\\n"
"    int G = nb * 16; float sum = 0.0f; \\\n"
"    for (int gg = lane; gg < G; gg += 32) { \\\n"
"        int ibg = gg >> 1; int half = gg & 1; int b = ibg >> 3; int ib32 = ibg & 7; \\\n"
"        const unsigned char *bp = row_ptr + b * 110; \\\n"
"        float dw = half_to_float(*(const half_raw *)bp); \\\n"
"        unsigned char qh = bp[66 + ib32]; \\\n"
"        const unsigned char *qsb = bp + 2 + ib32 * 8; \\\n"
"        const unsigned char *sgb = bp + 74 + ib32 * 4; \\\n"
"        unsigned char scb = bp[106 + (ib32 >> 1)]; \\\n"
"        int ls = (ib32 & 1) ? ((scb >> 4) & 0xf) : (scb & 0xf); \\\n"
"        const int *u = (const int *)(xq + (size_t)(b * 8 + ib32) * 32); \\\n"
"        float asc = xs[b * 8 + ib32]; \\\n"
"        int sumi = 0; \\\n"
"        for (int t = 0; t < 2; t++) { \\\n"
"            int l0 = half * 4 + t * 2; \\\n"
"            unsigned int gx = iq3s_grid_dev[qsb[l0]   | ((qh << (8 - l0)) & 0x100)]; \\\n"
"            unsigned int gy = iq3s_grid_dev[qsb[l0+1] | ((qh << (7 - l0)) & 0x100)]; \\\n"
"            unsigned char sb = sgb[l0 >> 1]; \\\n"
"            sumi = dp4a_hw(apply_sign4(gx, sb, 0), u[l0],   sumi); \\\n"
"            sumi = dp4a_hw(apply_sign4(gy, sb, 4), u[l0+1], sumi); \\\n"
"        } \\\n"
"        sum += dw * asc * (float)(1 + 2 * ls) * (float)sumi; \\\n"
"    } \\\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o); \\\n"
"    if (lane == 0) dst[row] = sum;\n"
"__global__ void matvec_iq3_s_dp4a(float *dst, const unsigned char *mat,\n"
"                                  const signed char *xq, const float *xs,\n"
"                                  int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id; if (row >= n_rows) return;\n"
"    IQ3S_DP4A_BODY\n"
"}\n"
"__global__ void matvec_iq3_s_expert_dp4a(float *dst, const unsigned char *base,\n"
"                                         const signed char *xq, const float *xs,\n"
"                                         int n_rows, int n_cols,\n"
"                                         const int *eidx, int slot, long long stride) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id; if (row >= n_rows) return;\n"
"    const unsigned char *mat = base + (long long)eidx[slot] * stride;\n"
"    IQ3S_DP4A_BODY\n"
"}\n"
"#define IQ2S_DP4A_BODY \\\n"
"    int nb = n_cols / 256; int row_bytes = nb * 82; \\\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes; \\\n"
"    int G = nb * 16; float sum = 0.0f; \\\n"
"    for (int gg = lane; gg < G; gg += 32) { \\\n"
"        int ibg = gg >> 1; int half = gg & 1; int b = ibg >> 3; int ib32 = ibg & 7; \\\n"
"        const unsigned char *bp = row_ptr + b * 82; \\\n"
"        float dw = half_to_float(*(const half_raw *)bp); \\\n"
"        const unsigned char *qsb = bp + 2 + ib32 * 4; \\\n"
"        unsigned char qh = bp[66 + ib32]; \\\n"
"        const unsigned char *sgb = bp + 34 + ib32 * 4; \\\n"
"        unsigned char scb = bp[74 + ib32]; \\\n"
"        int ls = half ? (scb >> 4) : (scb & 0xf); \\\n"
"        const int *u = (const int *)(xq + (size_t)(b * 8 + ib32) * 32); \\\n"
"        float asc = xs[b * 8 + ib32]; \\\n"
"        int sumi = 0; \\\n"
"        for (int t = 0; t < 2; t++) { \\\n"
"            int l = half * 2 + t; \\\n"
"            int idx = qsb[l] | ((qh << (8 - 2 * l)) & 0x300); \\\n"
"            const int *gp = (const int *)&iq2s_grid_dev[idx]; \\\n"
"            unsigned char sb = sgb[l]; \\\n"
"            sumi = dp4a_hw(apply_sign4(gp[0], sb, 0), u[l*2],   sumi); \\\n"
"            sumi = dp4a_hw(apply_sign4(gp[1], sb, 4), u[l*2+1], sumi); \\\n"
"        } \\\n"
"        sum += dw * asc * 0.25f * ((float)ls + 0.5f) * (float)sumi; \\\n"
"    } \\\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o); \\\n"
"    if (lane == 0) dst[row] = sum;\n"
"__global__ void matvec_iq2_s_dp4a(float *dst, const unsigned char *mat,\n"
"                                  const signed char *xq, const float *xs,\n"
"                                  int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id; if (row >= n_rows) return;\n"
"    IQ2S_DP4A_BODY\n"
"}\n"
"__global__ void matvec_iq2_s_expert_dp4a(float *dst, const unsigned char *base,\n"
"                                         const signed char *xq, const float *xs,\n"
"                                         int n_rows, int n_cols,\n"
"                                         const int *eidx, int slot, long long stride) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id; if (row >= n_rows) return;\n"
"    const unsigned char *mat = base + (long long)eidx[slot] * stride;\n"
"    IQ2S_DP4A_BODY\n"
"}\n"
"/* ======================================================================== */\n"
"/* LDS-cached grid decode matvecs: copy the iq2s/iq3s grid table into shared   */\n"
"/* memory once per block so the divergent per-lane lookups hit LDS (~20 cyc)    */\n"
"/* instead of L2 (~200 cyc). Same full-utilization math as the F32 kernels.     */\n"
"/* ======================================================================== */\n"
"#define IQ3S_FU_BODY(GRID) \\\n"
"    int nb = n_cols / 256; int row_bytes = nb * 110; \\\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes; \\\n"
"    int G = nb * 32; float sum = 0.0f; \\\n"
"    for (int g = lane; g < G; g += 32) { \\\n"
"        int b = g >> 5; int g32 = g & 31; int sb = g32 >> 2; int l = g32 & 3; \\\n"
"        const unsigned char *bp = row_ptr + b * 110; \\\n"
"        float d = half_to_float(*(const half_raw *)bp); \\\n"
"        unsigned char sc = bp[106 + (sb >> 1)]; \\\n"
"        float db = d * (float)(1 + 2 * ((sb & 1) ? (sc >> 4) : (sc & 0xf))); \\\n"
"        unsigned char qh = bp[66 + sb]; \\\n"
"        unsigned char qs0 = bp[2 + sb * 8 + 2 * l + 0]; \\\n"
"        unsigned char qs1 = bp[2 + sb * 8 + 2 * l + 1]; \\\n"
"        const unsigned char *grid1 = (const unsigned char *)&GRID[qs0 | ((qh << (8 - 2 * l)) & 256)]; \\\n"
"        const unsigned char *grid2 = (const unsigned char *)&GRID[qs1 | ((qh << (7 - 2 * l)) & 256)]; \\\n"
"        unsigned char s = bp[74 + sb * 4 + l]; \\\n"
"        const float *xb = x + b * 256 + sb * 32 + l * 8; \\\n"
"        float partial = 0.0f; \\\n"
"        for (int j = 0; j < 4; j++) { \\\n"
"            partial += db * (float)grid1[j] * ((s & (1 << j)) ? -1.0f : 1.0f) * xb[j]; \\\n"
"            partial += db * (float)grid2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f) * xb[j+4]; \\\n"
"        } \\\n"
"        sum += partial; \\\n"
"    } \\\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o); \\\n"
"    if (lane == 0) dst[row] = sum;\n"
"#define IQ2S_FU_BODY(GRID) \\\n"
"    int nb = n_cols / 256; int row_bytes = nb * 82; \\\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes; \\\n"
"    int G = nb * 32; float sum = 0.0f; \\\n"
"    for (int g = lane; g < G; g += 32) { \\\n"
"        int b = g >> 5; int gl = g & 31; int ib32 = gl >> 2; int l = gl & 3; \\\n"
"        const unsigned char *bp = row_ptr + b * 82; \\\n"
"        float d = half_to_float(*(const half_raw *)bp); \\\n"
"        unsigned char scale = bp[74 + ib32]; \\\n"
"        float db = (l < 2) ? d * (0.5f + (float)(scale & 0xf)) * 0.25f \\\n"
"                           : d * (0.5f + (float)(scale >>  4)) * 0.25f; \\\n"
"        unsigned char qsl = bp[2 + ib32 * 4 + l]; \\\n"
"        unsigned char qh  = bp[66 + ib32]; \\\n"
"        int grid_idx = qsl | ((qh << (8 - 2 * l)) & 0x300); \\\n"
"        const unsigned char *grid = (const unsigned char *)&GRID[grid_idx]; \\\n"
"        unsigned char s = bp[34 + ib32 * 4 + l]; \\\n"
"        const float *xb = x + b * 256 + ib32 * 32 + l * 8; \\\n"
"        float partial = 0.0f; \\\n"
"        for (int j = 0; j < 8; j++) \\\n"
"            partial += db * (float)grid[j] * ((s & (1 << j)) ? -1.0f : 1.0f) * xb[j]; \\\n"
"        sum += partial; \\\n"
"    } \\\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o); \\\n"
"    if (lane == 0) dst[row] = sum;\n"
"__global__ void matvec_iq3_s_lds_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                       int n_rows, int n_cols) {\n"
"    __shared__ unsigned int sgrid[512];\n"
"    for (int i = threadIdx.x; i < 512; i += blockDim.x) sgrid[i] = iq3s_grid_dev[i];\n"
"    __syncthreads();\n"
"    int lane = threadIdx.x % 32; int row = blockIdx.x * 8 + threadIdx.x / 32;\n"
"    if (row < n_rows) { IQ3S_FU_BODY(sgrid) }\n"
"}\n"
"__global__ void matvec_iq3_s_expert_lds_f32(float *dst, const unsigned char *base, const float *x,\n"
"                                              int n_rows, int n_cols,\n"
"                                              const int *eidx, int slot, long long stride) {\n"
"    __shared__ unsigned int sgrid[512];\n"
"    for (int i = threadIdx.x; i < 512; i += blockDim.x) sgrid[i] = iq3s_grid_dev[i];\n"
"    __syncthreads();\n"
"    int lane = threadIdx.x % 32; int row = blockIdx.x * 8 + threadIdx.x / 32;\n"
"    if (row < n_rows) { const unsigned char *mat = base + (long long)eidx[slot] * stride; IQ3S_FU_BODY(sgrid) }\n"
"}\n"
"__global__ void matvec_iq2_s_lds_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                       int n_rows, int n_cols) {\n"
"    __shared__ unsigned long long sgrid[1024];\n"
"    for (int i = threadIdx.x; i < 1024; i += blockDim.x) sgrid[i] = iq2s_grid_dev[i];\n"
"    __syncthreads();\n"
"    int lane = threadIdx.x % 32; int row = blockIdx.x * 8 + threadIdx.x / 32;\n"
"    if (row < n_rows) { IQ2S_FU_BODY(sgrid) }\n"
"}\n"
"__global__ void matvec_iq2_s_expert_lds_f32(float *dst, const unsigned char *base, const float *x,\n"
"                                              int n_rows, int n_cols,\n"
"                                              const int *eidx, int slot, long long stride) {\n"
"    __shared__ unsigned long long sgrid[1024];\n"
"    for (int i = threadIdx.x; i < 1024; i += blockDim.x) sgrid[i] = iq2s_grid_dev[i];\n"
"    __syncthreads();\n"
"    int lane = threadIdx.x % 32; int row = blockIdx.x * 8 + threadIdx.x / 32;\n"
"    if (row < n_rows) { const unsigned char *mat = base + (long long)eidx[slot] * stride; IQ2S_FU_BODY(sgrid) }\n"
"}\n"
"/* ======================================================================== */\n"
"/* Fused quantized MoE GEMM (mmq-style): Y[cnt,N] = X[cnt,K] x W[N,K]^T with W   */\n"
"/* kept QUANTIZED. One warp owns weight-row n; it dequants each weight group ONCE */\n"
"/* (grid lookup) and reuses it across all cnt tokens — amortizing the dequant +   */\n"
"/* weight read over the token tile. cnt <= 32 per launch (host chunks). X is f32. */\n"
"/* ======================================================================== */\n"
"__global__ void mmq_iq2s_f32(float *Y, const unsigned char *W, const float *X,\n"
"                              int cnt, int N, int K) {\n"
"    int warp = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int n = blockIdx.x * 8 + warp; if (n >= N) return;\n"
"    int nb = K / 256;\n"
"    const unsigned char *wrow = W + (size_t)n * nb * 82;\n"
"    float acc[32];\n"
"    for (int t = 0; t < cnt; t++) acc[t] = 0.0f;\n"
"    for (int g = lane; g < nb * 32; g += 32) {\n"
"        int b = g >> 5; int gl = g & 31; int ib32 = gl >> 2; int l = gl & 3;\n"
"        const unsigned char *bp = wrow + b * 82;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char scale = bp[74 + ib32];\n"
"        float db = (l < 2) ? d * (0.5f + (float)(scale & 0xf)) * 0.25f\n"
"                           : d * (0.5f + (float)(scale >>  4)) * 0.25f;\n"
"        unsigned char qsl = bp[2 + ib32 * 4 + l];\n"
"        unsigned char qh  = bp[66 + ib32];\n"
"        int gi = qsl | ((qh << (8 - 2 * l)) & 0x300);\n"
"        const unsigned char *grid = (const unsigned char *)&iq2s_grid_dev[gi];\n"
"        unsigned char s = bp[34 + ib32 * 4 + l];\n"
"        float w8[8];\n"
"        for (int j = 0; j < 8; j++) w8[j] = db * (float)grid[j] * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"        int xbase = b * 256 + ib32 * 32 + l * 8;\n"
"        for (int t = 0; t < cnt; t++) {\n"
"            const float *xt = X + (size_t)t * K + xbase;\n"
"            acc[t] += w8[0]*xt[0]+w8[1]*xt[1]+w8[2]*xt[2]+w8[3]*xt[3]\n"
"                    + w8[4]*xt[4]+w8[5]*xt[5]+w8[6]*xt[6]+w8[7]*xt[7];\n"
"        }\n"
"    }\n"
"    for (int t = 0; t < cnt; t++) {\n"
"        float v = acc[t];\n"
"        for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);\n"
"        if (lane == 0) Y[(size_t)t * N + n] = v;\n"
"    }\n"
"}\n"
"__global__ void mmq_iq3s_f32(float *Y, const unsigned char *W, const float *X,\n"
"                              int cnt, int N, int K) {\n"
"    int warp = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int n = blockIdx.x * 8 + warp; if (n >= N) return;\n"
"    int nb = K / 256;\n"
"    const unsigned char *wrow = W + (size_t)n * nb * 110;\n"
"    float acc[32];\n"
"    for (int t = 0; t < cnt; t++) acc[t] = 0.0f;\n"
"    for (int g = lane; g < nb * 32; g += 32) {\n"
"        int b = g >> 5; int g32 = g & 31; int sb = g32 >> 2; int l = g32 & 3;\n"
"        const unsigned char *bp = wrow + b * 110;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char sc = bp[106 + (sb >> 1)];\n"
"        float db = d * (float)(1 + 2 * ((sb & 1) ? (sc >> 4) : (sc & 0xf)));\n"
"        unsigned char qh = bp[66 + sb];\n"
"        unsigned char qs0 = bp[2 + sb * 8 + 2 * l + 0];\n"
"        unsigned char qs1 = bp[2 + sb * 8 + 2 * l + 1];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3s_grid_dev[qs0 | ((qh << (8 - 2 * l)) & 256)];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3s_grid_dev[qs1 | ((qh << (7 - 2 * l)) & 256)];\n"
"        unsigned char s = bp[74 + sb * 4 + l];\n"
"        float w8[8];\n"
"        for (int j = 0; j < 4; j++) {\n"
"            w8[j]   = db * (float)g1[j] * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"            w8[j+4] = db * (float)g2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"        }\n"
"        int xbase = b * 256 + sb * 32 + l * 8;\n"
"        for (int t = 0; t < cnt; t++) {\n"
"            const float *xt = X + (size_t)t * K + xbase;\n"
"            acc[t] += w8[0]*xt[0]+w8[1]*xt[1]+w8[2]*xt[2]+w8[3]*xt[3]\n"
"                    + w8[4]*xt[4]+w8[5]*xt[5]+w8[6]*xt[6]+w8[7]*xt[7];\n"
"        }\n"
"    }\n"
"    for (int t = 0; t < cnt; t++) {\n"
"        float v = acc[t];\n"
"        for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);\n"
"        if (lane == 0) Y[(size_t)t * N + n] = v;\n"
"    }\n"
"}\n"
"/* ======================================================================== */\n"
"/* Fused decode MoE matvecs: one launch covers all K selected experts.        */\n"
"/* Cuts ~40 small dependent launches/layer to 2 (gate+up+silu, down+accum).   */\n"
"/* blockIdx.y = expert slot; expert base resolved on-device from eidx[].      */\n"
"/* ======================================================================== */\n"
"__global__ void moe_gateup_silu_iq2s(float *out /*[K+1,eff]*/,\n"
"        const unsigned char *gate_base, const unsigned char *up_base,\n"
"        const float *x, int eff, int n_cols,\n"
"        const int *eidx, long long stride, int n_slots,\n"
"        const unsigned char *shg, const unsigned char *shu, float *accum, int n_embd) {\n"
"    int slot = blockIdx.y;\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    /* slot 0 also zeroes the accumulator (removes a memset launch) */\n"
"    if (slot == 0) {\n"
"        int t = blockIdx.x * blockDim.x + threadIdx.x;\n"
"        if (t < n_embd) accum[t] = 0.0f;\n"
"    }\n"
"    if (row >= eff) return;\n"
"    if (slot == n_slots - 1 && shg) {\n"
"        /* shared expert: Q6_K gate+up+silu */\n"
"        int nb = n_cols / 256; int row_bytes = nb * 210;\n"
"        int G = nb * 16; float sg6 = 0.0f, su6 = 0.0f;\n"
"        for (int g = lane; g < G; g += 32) {\n"
"            int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"            sg6 += q6k_dot4(shg + (size_t)row * row_bytes + b * 210, x + b * 256, half, lp);\n"
"            su6 += q6k_dot4(shu + (size_t)row * row_bytes + b * 210, x + b * 256, half, lp);\n"
"        }\n"
"        for (int o = 16; o > 0; o >>= 1) { sg6 += __shfl_down(sg6, o); su6 += __shfl_down(su6, o); }\n"
"        if (lane == 0) out[(size_t)slot * eff + row] = (sg6 / (1.0f + expf(-sg6))) * su6;\n"
"        return;\n"
"    }\n"
"    long long wb = (long long)eidx[slot] * stride;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 82;\n"
"    int G = nb * 32;\n"
"    float sg = 0.0f, su = 0.0f;\n"
"    const unsigned char *grow = gate_base + wb + (size_t)row * row_bytes;\n"
"    const unsigned char *urow = up_base   + wb + (size_t)row * row_bytes;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 5; int gl = g & 31; int ib32 = gl >> 2; int l = gl & 3;\n"
"        const float *xb = x + b * 256 + ib32 * 32 + l * 8;\n"
"        #pragma unroll\n"
"        for (int w = 0; w < 2; w++) {\n"
"            const unsigned char *bp = (w ? urow : grow) + b * 82;\n"
"            float d = half_to_float(*(const half_raw *)bp);\n"
"            unsigned char scale = bp[74 + ib32];\n"
"            float db = (l < 2) ? d * (0.5f + (float)(scale & 0xf)) * 0.25f\n"
"                               : d * (0.5f + (float)(scale >>  4)) * 0.25f;\n"
"            int gi = bp[2 + ib32*4 + l] | ((bp[66 + ib32] << (8 - 2*l)) & 0x300);\n"
"            const unsigned char *grid = (const unsigned char *)&iq2s_grid_dev[gi];\n"
"            unsigned char s = bp[34 + ib32*4 + l];\n"
"            float p = 0.0f;\n"
"            for (int j = 0; j < 8; j++)\n"
"                p += db * (float)grid[j] * ((s & (1 << j)) ? -1.0f : 1.0f) * xb[j];\n"
"            if (w) su += p; else sg += p;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) { sg += __shfl_down(sg, o); su += __shfl_down(su, o); }\n"
"    if (lane == 0) {\n"
"        float gate = sg / (1.0f + expf(-sg));   /* SiLU(gate) */\n"
"        out[(size_t)slot * eff + row] = gate * su;\n"
"    }\n"
"}\n"
"__global__ void moe_down_accum_iq3s(float *accum /*[n_embd]*/,\n"
"        const unsigned char *down_base, const float *acts /*[K+1,eff]*/,\n"
"        int n_embd, int eff, const int *eidx, const float *ew, long long stride,\n"
"        int n_slots, const unsigned char *shd, const float *shscale, int dtype) {\n"
"    int slot = blockIdx.y;\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id; if (row >= n_embd) return;\n"
"    const float *x = acts + (size_t)slot * eff;\n"
"    if (slot == n_slots - 1 && shd) {\n"
"        /* shared expert down: Q6_K, scaled by sigmoid gate */\n"
"        int nb6 = eff / 256; int rb6 = nb6 * 210;\n"
"        const unsigned char *rp = shd + (size_t)row * rb6;\n"
"        int G6 = nb6 * 16; float s6 = 0.0f;\n"
"        for (int g = lane; g < G6; g += 32) {\n"
"            int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"            s6 += q6k_dot4(rp + b * 210, x + b * 256, half, lp);\n"
"        }\n"
"        for (int o = 16; o > 0; o >>= 1) s6 += __shfl_down(s6, o);\n"
"        if (lane == 0) atomicAdd(&accum[row], shscale[0] * s6);\n"
"        return;\n"
"    }\n"
"    const unsigned char *mat = down_base + (long long)eidx[slot] * stride;\n"
"    int nb = eff / 256;\n"
"    if (dtype == 1) {  /* IQ4_XS */\n"
"        const unsigned char *row_ptr4 = mat + (size_t)row * nb * 136;\n"
"        float sum4 = 0.0f;\n"
"        for (int g = lane; g < nb * 8; g += 32) {\n"
"            int b = g >> 3; int ib = g & 7;\n"
"            const unsigned char *bp = row_ptr4 + b * 136;\n"
"            float d = half_to_float(*(const half_raw *)bp);\n"
"            unsigned short scales_h = *(const unsigned short *)(bp + 2);\n"
"            const unsigned char *scales_l = bp + 4;\n"
"            const unsigned char *qs = bp + 8 + ib * 16;\n"
"            int ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4);\n"
"            float dl = d * (float)(ls - 32);\n"
"            const float *xb = x + b * 256 + ib * 32;\n"
"            for (int j = 0; j < 16; j++) {\n"
"                sum4 += dl * (float)kvalues_iq4nl_dev[qs[j] & 0xf] * xb[j];\n"
"                sum4 += dl * (float)kvalues_iq4nl_dev[qs[j] >>  4] * xb[j + 16];\n"
"            }\n"
"        }\n"
"        for (int o = 16; o > 0; o >>= 1) sum4 += __shfl_down(sum4, o);\n"
"        if (lane == 0) atomicAdd(&accum[row], ew[slot] * sum4);\n"
"        return;\n"
"    }\n"
"    int row_bytes = nb * 110;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int G = nb * 32;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 5; int g32 = g & 31; int sb = g32 >> 2; int l = g32 & 3;\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char sc = bp[106 + (sb >> 1)];\n"
"        float db = d * (float)(1 + 2 * ((sb & 1) ? (sc >> 4) : (sc & 0xf)));\n"
"        unsigned char qh = bp[66 + sb];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3s_grid_dev[bp[2+sb*8+2*l]   | ((qh << (8-2*l)) & 256)];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3s_grid_dev[bp[2+sb*8+2*l+1] | ((qh << (7-2*l)) & 256)];\n"
"        unsigned char s = bp[74 + sb*4 + l];\n"
"        const float *xb = x + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            sum += db * (float)g1[j] * ((s & (1 << j)) ? -1.0f : 1.0f) * xb[j];\n"
"            sum += db * (float)g2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f) * xb[j+4];\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    if (lane == 0) atomicAdd(&accum[row], ew[slot] * sum);\n"
"}\n"
"/* ---- ssm_matvec4_q6k: fused qkv(Q6K) + z(Q6K) + alpha(F32) + beta(F32) matvecs. */\n"
"__global__ void ssm_matvec4_q6k(float *qkv, float *z, float *alpha, float *beta,\n"
"        const unsigned char *wq, const unsigned char *wz,\n"
"        const half_raw *wa, const half_raw *wb, const float *x,\n"
"        int qkv_rows, int z_rows, int dt_rank, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    const unsigned char *mat = 0; float *dst = 0; int r = row;\n"
"    if (r < qkv_rows) { mat = wq; dst = qkv + r; }\n"
"    else { r -= qkv_rows;\n"
"      if (r < z_rows) { mat = wz; dst = z + r; }\n"
"      else { r -= z_rows;\n"
"        const half_raw *wf = (r < dt_rank) ? wa + (size_t)r * n_cols : wb + (size_t)(r - dt_rank) * n_cols;\n"
"        float *df = (r < dt_rank) ? alpha + r : beta + (r - dt_rank);\n"
"        float sum = 0.0f;\n"
"        for (int j = tid; j < n_cols; j += nthreads) sum += half_to_float(wf[j]) * x[j];\n"
"        for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"        __shared__ float wsf[8]; int wid = tid/32, ln = tid%32;\n"
"        if (ln == 0) wsf[wid] = sum; __syncthreads();\n"
"        if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += wsf[w]; *df = t; }\n"
"        return; } }\n"
"    int nb = n_cols / 256; int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = mat + (size_t)r * row_bytes;\n"
"    int G = nb * 16; float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        sum += q6k_dot4(row_ptr + b * 210, x + b * 256, half, lp);\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float ws[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) ws[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += ws[w]; *dst = t; }\n"
"}\n"
"/* ---- matvec_qkv_q6k: fused attn Q/K/V Q6_K matvecs (one launch). */\n"
"__global__ void matvec_qkv_q6k(float *q, float *k, float *v,\n"
"        const unsigned char *wq, const unsigned char *wk, const unsigned char *wv,\n"
"        int q_rows, int k_rows, int v_rows, int n_cols, const float *x) {\n"
"    int row = blockIdx.x;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    const unsigned char *mat; float *dst; int r = row;\n"
"    if (r < q_rows) { mat = wq; dst = q + r; }\n"
"    else if (r < q_rows + k_rows) { r -= q_rows; mat = wk; dst = k + r; }\n"
"    else { r -= q_rows + k_rows; if (r >= v_rows) return; mat = wv; dst = v + r; }\n"
"    int nb = n_cols / 256; int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = mat + (size_t)r * row_bytes;\n"
"    int G = nb * 16; float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        sum += q6k_dot4(row_ptr + b * 210, x + b * 256, half, lp);\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float ws[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) ws[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += ws[w]; *dst = t; }\n"
"}\n"
"/* ---- matvec_down_residual_q6k: fused W_down * gate + x += result (Q6_K). */\n"
"/* Replaces matvec + add (2 launches) with 1 per dense FFN decode layer.\n"
" * x[row] += dot(W_down[row], gate). */\n"
"__global__ void matvec_down_residual_q6k(float *x, const unsigned char *down_w,\n"
"        const float *gate, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = down_w + (size_t)row * row_bytes;\n"
"    int G = nb * 16;\n"
"    float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        sum += q6k_dot4(row_ptr + b * 210, gate + b * 256, half, lp);\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float ws[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) ws[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += ws[w]; x[row] += t; }\n"
"}\n"
"/* ---- matvec_out_gated_q6k: fused gated attention output matvec (Q6_K). */\n"
"/* Fuses sigmoid_mul + output_w matvec into one launch.\n"
" * out[row] = dot(W_out[row], sigmoid(gate) .* in). */\n"
"__global__ void matvec_out_gated_q6k(float *out, const unsigned char *w_out,\n"
"        const float *in, const float *gate, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256; int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = w_out + (size_t)row * row_bytes;\n"
"    int G = nb * 16; float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        const unsigned char *bp = row_ptr + b * 210;\n"
"        const unsigned char *ql = bp + half * 64 + lp * 4;\n"
"        const unsigned char *qh = bp + 128 + half * 32 + lp * 4;\n"
"        const signed char *sc = (const signed char *)(bp + 192 + half * 8);\n"
"        float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"        unsigned int qlo, qhi, qhv;\n"
"        __builtin_memcpy(&qlo, ql, 4);\n"
"        __builtin_memcpy(&qhi, ql + 32, 4);\n"
"        __builtin_memcpy(&qhv, qh, 4);\n"
"        int hi = (lp < 4) ? 0 : 1;\n"
"        float sA = d * (float)sc[0 + hi], sB = d * (float)sc[2 + hi];\n"
"        float sC = d * (float)sc[4 + hi], sD = d * (float)sc[6 + hi];\n"
"        const float *x0 = in + b * 256 + half * 128 + lp * 4;\n"
"        const float *g0 = gate + b * 256 + half * 128 + lp * 4;\n"
"        float4 xv = *(const float4 *)x0;\n"
"        float4 xv2 = *(const float4 *)(x0 + 32);\n"
"        float4 xv3 = *(const float4 *)(x0 + 64);\n"
"        float4 xv4 = *(const float4 *)(x0 + 96);\n"
"        float4 gv = *(const float4 *)g0;\n"
"        float4 gv2 = *(const float4 *)(g0 + 32);\n"
"        float4 gv3 = *(const float4 *)(g0 + 64);\n"
"        float4 gv4 = *(const float4 *)(g0 + 96);\n"
"        for (int j = 0; j < 4; j++) {\n"
"            int lo = (qlo >> (8*j)) & 0xFF, hi8 = (qhi >> (8*j)) & 0xFF, h2 = (qhv >> (8*j)) & 0xFF;\n"
"            int q1 = (lo & 0xF) | ((h2 & 3) << 4); q1 -= 32;\n"
"            int q2 = (hi8 & 0xF) | (((h2 >> 2) & 3) << 4); q2 -= 32;\n"
"            int q3 = (lo >> 4) | (((h2 >> 4) & 3) << 4); q3 -= 32;\n"
"            int q4 = (hi8 >> 4) | (((h2 >> 6) & 3) << 4); q4 -= 32;\n"
"            float xa = (&xv.x)[j], xa2 = (&xv2.x)[j], xa3 = (&xv3.x)[j], xa4 = (&xv4.x)[j];\n"
"            float ga = (&gv.x)[j], ga2 = (&gv2.x)[j], ga3 = (&gv3.x)[j], ga4 = (&gv4.x)[j];\n"
"            float sg = 1.0f / (1.0f + __expf(-ga));\n"
"            float sg2 = 1.0f / (1.0f + __expf(-ga2));\n"
"            float sg3 = 1.0f / (1.0f + __expf(-ga3));\n"
"            float sg4 = 1.0f / (1.0f + __expf(-ga4));\n"
"            sum += sA * q1 * xa * sg + sB * q2 * xa2 * sg2\n"
"                 + sC * q3 * xa3 * sg3 + sD * q4 * xa4 * sg4;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float ws[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) ws[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += ws[w]; out[row] = t; }\n"
"}\n"
"/* ---- fused_ssm_out_gated_q6k: fused gated RMSNorm + ssm_out matvec (Q6_K). */\n"
"/* Fuses gated_rmsnorm_silu + ssm_out matvec into one launch per SSM decode\n"
" * layer. xb[row] = dot(W_out[row], ssm_out_normed) where ssm_out_normed =\n"
" * RMSNorm(ssm_out) * norm_w * SiLU(z). Per-head norm_w shared across heads. */\n"
"__global__ void fused_ssm_out_gated_q6k(float *xb, const unsigned char *w_out,\n"
"        const float *ssm_out, const float *z, const float *norm_w,\n"
"        int n_rows, int n_cols, int dt_rank, int d_state, float eps) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = w_out + (size_t)row * row_bytes;\n"
"    int G = nb * 16;\n"
"    float i_d_state = 1.0f / (float)d_state;\n"
"    /* Phase 1: accumulate per-head sum of squares (atomic on shared mem). */\n"
"    __shared__ float head_sq[256];\n"
"    if (tid < dt_rank) head_sq[tid] = 0.0f;\n"
"    __syncthreads();\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        int head = b * 2 + half;\n"
"        if (head >= dt_rank) continue;\n"
"        const float *x0 = ssm_out + b * 256 + half * 128 + lp * 4;\n"
"        float4 xv = *(const float4 *)x0;\n"
"        atomicAdd(&head_sq[head], xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w);\n"
"    }\n"
"    __syncthreads();\n"
"    /* Phase 2: matvec with RMSNorm * norm_w * SiLU(z) gating. */\n"
"    float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        int head = b * 2 + half;\n"
"        if (head >= dt_rank) continue;\n"
"        float inv_mean = rsqrtf(head_sq[head] * i_d_state + eps);\n"
"        /* Load z and norm_w for all 4 position groups within this Q6K subgroup */\n"
"        const float *zx0 = z + b * 256 + half * 128 + lp * 4;\n"
"        const float *nw0 = norm_w + half * 128 + lp * 4;\n"
"        float4 zv0 = *(const float4 *)zx0, zv32 = *(const float4 *)(zx0 + 32);\n"
"        float4 zv64 = *(const float4 *)(zx0 + 64), zv96 = *(const float4 *)(zx0 + 96);\n"
"        float4 nwv0 = *(const float4 *)nw0, nwv32 = *(const float4 *)(nw0 + 32);\n"
"        float4 nwv64 = *(const float4 *)(nw0 + 64), nwv96 = *(const float4 *)(nw0 + 96);\n"
"        const unsigned char *bp = row_ptr + b * 210;\n"
"        const unsigned char *ql = bp + half * 64 + lp * 4;\n"
"        const unsigned char *qh = bp + 128 + half * 32 + lp * 4;\n"
"        const signed char *sc = (const signed char *)(bp + 192 + half * 8);\n"
"        float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"        unsigned int qlo, qhi, qhv;\n"
"        __builtin_memcpy(&qlo, ql, 4);\n"
"        __builtin_memcpy(&qhi, ql + 32, 4);\n"
"        __builtin_memcpy(&qhv, qh, 4);\n"
"        int hi = (lp < 4) ? 0 : 1;\n"
"        float sA = d * (float)sc[0 + hi], sB = d * (float)sc[2 + hi];\n"
"        float sC = d * (float)sc[4 + hi], sD = d * (float)sc[6 + hi];\n"
"        const float *x0 = ssm_out + b * 256 + half * 128 + lp * 4;\n"
"        float4 xv0 = *(const float4 *)x0, xv32 = *(const float4 *)(x0 + 32);\n"
"        float4 xv64 = *(const float4 *)(x0 + 64), xv96 = *(const float4 *)(x0 + 96);\n"
"        for (int j = 0; j < 4; j++) {\n"
"            int lo = (qlo >> (8*j)) & 0xFF, hi8 = (qhi >> (8*j)) & 0xFF, h2 = (qhv >> (8*j)) & 0xFF;\n"
"            int q1 = (lo & 0xF) | ((h2 & 3) << 4); q1 -= 32;\n"
"            int q2 = (hi8 & 0xF) | (((h2 >> 2) & 3) << 4); q2 -= 32;\n"
"            int q3 = (lo >> 4) | (((h2 >> 4) & 3) << 4); q3 -= 32;\n"
"            int q4 = (hi8 >> 4) | (((h2 >> 6) & 3) << 4); q4 -= 32;\n"
"            float s0 = (&xv0.x)[j] * inv_mean * (&nwv0.x)[j] * ((&zv0.x)[j] / (1.0f + __expf(-(&zv0.x)[j])));\n"
"            float s32 = (&xv32.x)[j] * inv_mean * (&nwv32.x)[j] * ((&zv32.x)[j] / (1.0f + __expf(-(&zv32.x)[j])));\n"
"            float s64 = (&xv64.x)[j] * inv_mean * (&nwv64.x)[j] * ((&zv64.x)[j] / (1.0f + __expf(-(&zv64.x)[j])));\n"
"            float s96 = (&xv96.x)[j] * inv_mean * (&nwv96.x)[j] * ((&zv96.x)[j] / (1.0f + __expf(-(&zv96.x)[j])));\n"
"            sum += sA * q1 * s0 + sB * q2 * s32 + sC * q3 * s64 + sD * q4 * s96;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float ws[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) ws[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += ws[w]; xb[row] = t; }\n"
"}\n"
"/* Precompute per-head RMSNorm inv_mean from ssm_out ONCE (it is independent of  */\n"
"/* the output row). One warp per head; head h owns ssm_out[h*d_state .. +d_state].*/\n"
"__global__ void ssm_inv_mean_f32(float *inv_mean, const float *ssm_out,\n"
"        int dt_rank, int d_state, float eps) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int head = blockIdx.x * 8 + warp_id;\n"
"    if (head >= dt_rank) return;\n"
"    const float *p = ssm_out + (size_t)head * d_state;\n"
"    float ss = 0.0f;\n"
"    for (int i = lane; i < d_state; i += 32) { float v = p[i]; ss += v * v; }\n"
"    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down(ss, o);\n"
"    if (lane == 0) inv_mean[head] = rsqrtf(ss / (float)d_state + eps);\n"
"}\n"
"/* Warp-per-row SSM-out twin: reads precomputed inv_mean[head] (no Phase-1 atomic */\n"
"/* sum-of-squares, no __shared__/__syncthreads). q6k_dot4-style float4 loads.     */\n"
"__global__ void fused_ssm_out_gated_q6k_mw(float *xb, const unsigned char *w_out,\n"
"        const float *ssm_out, const float *z, const float *norm_w,\n"
"        const float *inv_mean, int n_rows, int n_cols, int dt_rank) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = w_out + (size_t)row * row_bytes;\n"
"    int G = nb * 16;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        int head = b * 2 + half;\n"
"        if (head >= dt_rank) continue;\n"
"        float inv_m = inv_mean[head];\n"
"        const float *zx0 = z + b * 256 + half * 128 + lp * 4;\n"
"        const float *nw0 = norm_w + half * 128 + lp * 4;\n"
"        float4 zv0 = *(const float4 *)zx0, zv32 = *(const float4 *)(zx0 + 32);\n"
"        float4 zv64 = *(const float4 *)(zx0 + 64), zv96 = *(const float4 *)(zx0 + 96);\n"
"        float4 nwv0 = *(const float4 *)nw0, nwv32 = *(const float4 *)(nw0 + 32);\n"
"        float4 nwv64 = *(const float4 *)(nw0 + 64), nwv96 = *(const float4 *)(nw0 + 96);\n"
"        const unsigned char *bp = row_ptr + b * 210;\n"
"        const unsigned char *ql = bp + half * 64 + lp * 4;\n"
"        const unsigned char *qh = bp + 128 + half * 32 + lp * 4;\n"
"        const signed char *sc = (const signed char *)(bp + 192 + half * 8);\n"
"        float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"        unsigned int qlo, qhi, qhv;\n"
"        __builtin_memcpy(&qlo, ql, 4);\n"
"        __builtin_memcpy(&qhi, ql + 32, 4);\n"
"        __builtin_memcpy(&qhv, qh, 4);\n"
"        int hi = (lp < 4) ? 0 : 1;\n"
"        float sA = d * (float)sc[0 + hi], sB = d * (float)sc[2 + hi];\n"
"        float sC = d * (float)sc[4 + hi], sD = d * (float)sc[6 + hi];\n"
"        const float *x0 = ssm_out + b * 256 + half * 128 + lp * 4;\n"
"        float4 xv0 = *(const float4 *)x0, xv32 = *(const float4 *)(x0 + 32);\n"
"        float4 xv64 = *(const float4 *)(x0 + 64), xv96 = *(const float4 *)(x0 + 96);\n"
"        for (int j = 0; j < 4; j++) {\n"
"            int lo = (qlo >> (8*j)) & 0xFF, hi8 = (qhi >> (8*j)) & 0xFF, h2 = (qhv >> (8*j)) & 0xFF;\n"
"            int q1 = (lo & 0xF) | ((h2 & 3) << 4); q1 -= 32;\n"
"            int q2 = (hi8 & 0xF) | (((h2 >> 2) & 3) << 4); q2 -= 32;\n"
"            int q3 = (lo >> 4) | (((h2 >> 4) & 3) << 4); q3 -= 32;\n"
"            int q4 = (hi8 >> 4) | (((h2 >> 6) & 3) << 4); q4 -= 32;\n"
"            float s0 = (&xv0.x)[j] * inv_m * (&nwv0.x)[j] * ((&zv0.x)[j] / (1.0f + __expf(-(&zv0.x)[j])));\n"
"            float s32 = (&xv32.x)[j] * inv_m * (&nwv32.x)[j] * ((&zv32.x)[j] / (1.0f + __expf(-(&zv32.x)[j])));\n"
"            float s64 = (&xv64.x)[j] * inv_m * (&nwv64.x)[j] * ((&zv64.x)[j] / (1.0f + __expf(-(&zv64.x)[j])));\n"
"            float s96 = (&xv96.x)[j] * inv_m * (&nwv96.x)[j] * ((&zv96.x)[j] / (1.0f + __expf(-(&zv96.x)[j])));\n"
"            sum += sA * q1 * s0 + sB * q2 * s32 + sC * q3 * s64 + sD * q4 * s96;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    if (lane == 0) xb[row] = sum;\n"
"}\n"
"/* ---- ffn_gate_up_silu_q6k: fused decode FFN gate + up matvec + SiLU (Q6_K). */\n"
"__global__ void ffn_gate_up_silu_q6k(float *gate_out,\n"
"        const unsigned char *gate_w, const unsigned char *up_w,\n"
"        const float *x, int n_ff, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_ff) return;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 210;\n"
"    int G = nb * 16;\n"
"    /* Gate dot product */\n"
"    const unsigned char *gp = gate_w + (size_t)row * row_bytes;\n"
"    float gsum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        gsum += q6k_dot4(gp + b * 210, x + b * 256, half, lp);\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) gsum += __shfl_down(gsum, o);\n"
"    __shared__ float sf[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) sf[wid] = gsum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += sf[w]; gsum = t; }\n"
"    __syncthreads();\n"
"    /* Up dot product */\n"
"    const unsigned char *up = up_w + (size_t)row * row_bytes;\n"
"    float usum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        usum += q6k_dot4(up + b * 210, x + b * 256, half, lp);\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) usum += __shfl_down(usum, o);\n"
"    if (ln == 0) sf[wid] = usum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += sf[w]; usum = t; }\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float g = gsum / (1.0f + __expf(-gsum));\n"
"        gate_out[row] = g * usum;\n"
"    }\n"
"}\n"
"/* ---- ffn_gate_up_silu_iq3xxs: fused FFN gate+up+SiLU for IQ3_XXS weights. */\n"
"__global__ void ffn_gate_up_silu_iq3xxs(float *gate_out,\n"
"        const unsigned char *gate_w, const unsigned char *up_w,\n"
"        const float *x, int n_ff, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_ff) return;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 98;\n"
"    int G = nb * 32;\n"
"    float gsum = 0.0f, usum = 0.0f;\n"
"    const unsigned char *gp = gate_w + (size_t)row * row_bytes;\n"
"    const unsigned char *up = up_w + (size_t)row * row_bytes;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        const unsigned char *bp_g = gp + b * 98;\n"
"        const unsigned char *bp_u = up + b * 98;\n"
"        float d_g = half_to_float(*(const half_raw *)bp_g);\n"
"        float d_u = half_to_float(*(const half_raw *)bp_u);\n"
"        const unsigned char *qs_g = bp_g + 2, *qs_u = bp_u + 2;\n"
"        unsigned int aux_g = *(const unsigned int *)(bp_g + 66 + 4*sb);\n"
"        unsigned int aux_u = *(const unsigned int *)(bp_u + 66 + 4*sb);\n"
"        float db_g = d_g * (0.5f + (float)(aux_g >> 28)) * 0.5f;\n"
"        float db_u = d_u * (0.5f + (float)(aux_u >> 28)) * 0.5f;\n"
"        unsigned char sgn_g = ksigns_iq2xs_dev[(aux_g >> (7*l)) & 127];\n"
"        unsigned char sgn_u = ksigns_iq2xs_dev[(aux_u >> (7*l)) & 127];\n"
"        const unsigned char *g1_g = (const unsigned char *)&iq3xxs_grid_dev[qs_g[8*sb+2*l]];\n"
"        const unsigned char *g2_g = (const unsigned char *)&iq3xxs_grid_dev[qs_g[8*sb+2*l+1]];\n"
"        const unsigned char *g1_u = (const unsigned char *)&iq3xxs_grid_dev[qs_u[8*sb+2*l]];\n"
"        const unsigned char *g2_u = (const unsigned char *)&iq3xxs_grid_dev[qs_u[8*sb+2*l+1]];\n"
"        const float *xb = x + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float xv0 = xb[j], xv1 = xb[j+4];\n"
"            float wg0 = db_g * (float)g1_g[j] * ((sgn_g & (1 << j)) ? -1.0f : 1.0f);\n"
"            float wg1 = db_g * (float)g2_g[j] * ((sgn_g & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            float wu0 = db_u * (float)g1_u[j] * ((sgn_u & (1 << j)) ? -1.0f : 1.0f);\n"
"            float wu1 = db_u * (float)g2_u[j] * ((sgn_u & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            gsum += wg0 * xv0 + wg1 * xv1;\n"
"            usum += wu0 * xv0 + wu1 * xv1;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) { gsum += __shfl_down(gsum, o); usum += __shfl_down(usum, o); }\n"
"    __shared__ float sf[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) { sf[wid] = gsum; } __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += sf[w]; gsum = t; }\n"
"    __syncthreads();\n"
"    if (ln == 0) { sf[wid] = usum; } __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += sf[w]; usum = t; }\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float g = gsum / (1.0f + __expf(-gsum));\n"
"        gate_out[row] = g * usum;\n"
"    }\n"
"}\n"
"/* Warp-per-row twin: one warp owns a row, G=nb*32 (all 32 lanes active), intra- */\n"
"/* warp reduction only (no cross-warp __shared__/__syncthreads). 8 rows/block.   */\n"
"__global__ void ffn_gate_up_silu_iq3xxs_mw(float *gate_out,\n"
"        const unsigned char *gate_w, const unsigned char *up_w,\n"
"        const float *x, int n_ff, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * (blockDim.x / 32) + warp_id;\n"
"    if (row >= n_ff) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 98;\n"
"    int G = nb * 32;\n"
"    float gsum = 0.0f, usum = 0.0f;\n"
"    const unsigned char *gp = gate_w + (size_t)row * row_bytes;\n"
"    const unsigned char *up = up_w + (size_t)row * row_bytes;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        const unsigned char *bp_g = gp + b * 98;\n"
"        const unsigned char *bp_u = up + b * 98;\n"
"        float d_g = half_to_float(*(const half_raw *)bp_g);\n"
"        float d_u = half_to_float(*(const half_raw *)bp_u);\n"
"        const unsigned char *qs_g = bp_g + 2, *qs_u = bp_u + 2;\n"
"        unsigned int aux_g = *(const unsigned int *)(bp_g + 66 + 4*sb);\n"
"        unsigned int aux_u = *(const unsigned int *)(bp_u + 66 + 4*sb);\n"
"        float db_g = d_g * (0.5f + (float)(aux_g >> 28)) * 0.5f;\n"
"        float db_u = d_u * (0.5f + (float)(aux_u >> 28)) * 0.5f;\n"
"        unsigned char sgn_g = ksigns_iq2xs_dev[(aux_g >> (7*l)) & 127];\n"
"        unsigned char sgn_u = ksigns_iq2xs_dev[(aux_u >> (7*l)) & 127];\n"
"        const unsigned char *g1_g = (const unsigned char *)&iq3xxs_grid_dev[qs_g[8*sb+2*l]];\n"
"        const unsigned char *g2_g = (const unsigned char *)&iq3xxs_grid_dev[qs_g[8*sb+2*l+1]];\n"
"        const unsigned char *g1_u = (const unsigned char *)&iq3xxs_grid_dev[qs_u[8*sb+2*l]];\n"
"        const unsigned char *g2_u = (const unsigned char *)&iq3xxs_grid_dev[qs_u[8*sb+2*l+1]];\n"
"        const float *xb = x + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float xv0 = xb[j], xv1 = xb[j+4];\n"
"            float wg0 = db_g * (float)g1_g[j] * ((sgn_g & (1 << j)) ? -1.0f : 1.0f);\n"
"            float wg1 = db_g * (float)g2_g[j] * ((sgn_g & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            float wu0 = db_u * (float)g1_u[j] * ((sgn_u & (1 << j)) ? -1.0f : 1.0f);\n"
"            float wu1 = db_u * (float)g2_u[j] * ((sgn_u & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            gsum += wg0 * xv0 + wg1 * xv1;\n"
"            usum += wu0 * xv0 + wu1 * xv1;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) { gsum += __shfl_down(gsum, o); usum += __shfl_down(usum, o); }\n"
"    if (lane == 0) {\n"
"        float g = gsum / (1.0f + __expf(-gsum));\n"
"        gate_out[row] = g * usum;\n"
"    }\n"
"}\n"
"/* ---- matvec_down_residual_iq3xxs: fused down matvec + residual for IQ3_XXS. */\n"
"__global__ void matvec_down_residual_iq3xxs(float *x, const unsigned char *down_w,\n"
"        const float *gate, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = down_w + (size_t)row * row_bytes;\n"
"    int G = nb * 32;\n"
"    float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        unsigned int aux = *(const unsigned int *)(bp + 66 + 4*sb);\n"
"        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;\n"
"        unsigned char sgn = ksigns_iq2xs_dev[(aux >> (7*l)) & 127];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l]];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l+1]];\n"
"        const float *gb = gate + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float gv0 = gb[j], gv1 = gb[j+4];\n"
"            float w0 = db * (float)g1[j] * ((sgn & (1 << j)) ? -1.0f : 1.0f);\n"
"            float w1 = db * (float)g2[j] * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            sum += w0 * gv0 + w1 * gv1;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float ws[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) ws[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += ws[w]; x[row] += t; }\n"
"}\n"
"/* Warp-per-row twin (G=nb*32, intra-warp reduction, 8 rows/block). */\n"
"__global__ void matvec_down_residual_iq3xxs_mw(float *x, const unsigned char *down_w,\n"
"        const float *gate, int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = down_w + (size_t)row * row_bytes;\n"
"    int G = nb * 32;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        unsigned int aux = *(const unsigned int *)(bp + 66 + 4*sb);\n"
"        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;\n"
"        unsigned char sgn = ksigns_iq2xs_dev[(aux >> (7*l)) & 127];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l]];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l+1]];\n"
"        const float *gb = gate + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float gv0 = gb[j], gv1 = gb[j+4];\n"
"            float w0 = db * (float)g1[j] * ((sgn & (1 << j)) ? -1.0f : 1.0f);\n"
"            float w1 = db * (float)g2[j] * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            sum += w0 * gv0 + w1 * gv1;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    if (lane == 0) x[row] += sum;\n"
"}\n"
"/* Split-K warp-per-row twin: grid.y = KSPLIT warps cooperate per row over the   */\n"
"/* nb-block range, atomicAdd partials into x[row] (residual already resident).    */\n"
"/* Raises block count for low-row matvecs (down: 5120 rows) to fill the GPU.      */\n"
"__global__ void matvec_down_residual_iq3xxs_splitk(float *x, const unsigned char *down_w,\n"
"        const float *gate, int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int ks = blockIdx.y; int KS = gridDim.y;\n"
"    int nb = n_cols / 256;\n"
"    int b0 = ks * nb / KS, b1 = (ks + 1) * nb / KS;\n"
"    int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = down_w + (size_t)row * row_bytes;\n"
"    int sb = lane >> 2, l = lane & 3;\n"
"    float sum = 0.0f;\n"
"    for (int b = b0; b < b1; b++) {\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        unsigned int aux = *(const unsigned int *)(bp + 66 + 4*sb);\n"
"        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;\n"
"        unsigned char sgn = ksigns_iq2xs_dev[(aux >> (7*l)) & 127];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l]];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l+1]];\n"
"        const float *gb = gate + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float gv0 = gb[j], gv1 = gb[j+4];\n"
"            float w0 = db * (float)g1[j] * ((sgn & (1 << j)) ? -1.0f : 1.0f);\n"
"            float w1 = db * (float)g2[j] * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            sum += w0 * gv0 + w1 * gv1;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    if (lane == 0) { float v = sum; if (!isnan(v)) atomicAdd(&x[row], v); }\n"
"}\n"
"/* ---- matvec_qkv_iq3xxs: fused attn Q/K/V IQ3_XXS matvecs (one launch). */\n"
"__global__ void matvec_qkv_iq3xxs(float *q, float *k, float *v,\n"
"        const unsigned char *wq, const unsigned char *wk, const unsigned char *wv,\n"
"        int q_rows, int k_rows, int v_rows, int n_cols, const float *x) {\n"
"    int row = blockIdx.x;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    const unsigned char *mat; float *dst; int r = row;\n"
"    if (r < q_rows) { mat = wq; dst = q + r; }\n"
"    else if (r < q_rows + k_rows) { r -= q_rows; mat = wk; dst = k + r; }\n"
"    else { r -= q_rows + k_rows; if (r >= v_rows) return; mat = wv; dst = v + r; }\n"
"    int nb = n_cols / 256; int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = mat + (size_t)r * row_bytes;\n"
"    int G = nb * 32; float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        unsigned int aux = *(const unsigned int *)(bp + 66 + 4*sb);\n"
"        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;\n"
"        unsigned char sgn = ksigns_iq2xs_dev[(aux >> (7*l)) & 127];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l]];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l+1]];\n"
"        const float *xb = x + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float xv0 = xb[j], xv1 = xb[j+4];\n"
"            float w0 = db * (float)g1[j] * ((sgn & (1 << j)) ? -1.0f : 1.0f);\n"
"            float w1 = db * (float)g2[j] * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            sum += w0 * xv0 + w1 * xv1;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float ws[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) ws[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += ws[w]; *dst = t; }\n"
"}\n"
"/* ---- matvec_out_gated_iq3xxs: fused sigmoid_mul + output matvec for IQ3_XXS. */\n"
"__global__ void matvec_out_gated_iq3xxs(float *out, const unsigned char *w_out,\n"
"        const float *in, const float *gate, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256; int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = w_out + (size_t)row * row_bytes;\n"
"    int G = nb * 32; float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        unsigned int aux = *(const unsigned int *)(bp + 66 + 4*sb);\n"
"        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;\n"
"        unsigned char sgn = ksigns_iq2xs_dev[(aux >> (7*l)) & 127];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l]];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l+1]];\n"
"        const float *x0 = in + b * 256 + sb * 32 + l * 8;\n"
"        const float *g0 = gate + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float sg0 = 1.0f / (1.0f + __expf(-g0[j]));\n"
"            float sg1 = 1.0f / (1.0f + __expf(-g0[j+4]));\n"
"            float w0 = db * (float)g1[j] * ((sgn & (1 << j)) ? -1.0f : 1.0f);\n"
"            float w1 = db * (float)g2[j] * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            sum += (w0 * x0[j] * sg0) + (w1 * x0[j+4] * sg1);\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float ws[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) ws[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += ws[w]; out[row] = t; }\n"
"}\n"
"/* ---- ssm_matvec4_iq3xxs: fused qkv + z + alpha + beta IQ3_XXS matvecs. */\n"
"__global__ void ssm_matvec4_iq3xxs(float *qkv, float *z, float *alpha, float *beta,\n"
"        const unsigned char *wq, const unsigned char *wz,\n"
"        const half_raw *wa, const half_raw *wb, const float *x,\n"
"        int qkv_rows, int z_rows, int dt_rank, int n_cols) {\n"
"    int row = blockIdx.x; int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    const unsigned char *mat = 0; float *dst = 0; int r = row;\n"
"    if (r < qkv_rows) { mat = wq; dst = qkv + r; }\n"
"    else { r -= qkv_rows;\n"
"      if (r < z_rows) { mat = wz; dst = z + r; }\n"
"      else { r -= z_rows;\n"
"        const half_raw *wf = (r < dt_rank) ? wa + (size_t)r * n_cols : wb + (size_t)(r - dt_rank) * n_cols;\n"
"        float *df = (r < dt_rank) ? alpha + r : beta + (r - dt_rank);\n"
"        float sum = 0.0f;\n"
"        for (int j = tid; j < n_cols; j += nthreads) sum += half_to_float(wf[j]) * x[j];\n"
"        for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"        __shared__ float wsf[8]; int wid = tid/32, ln = tid%32;\n"
"        if (ln == 0) wsf[wid] = sum; __syncthreads();\n"
"        if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += wsf[w]; *df = t; }\n"
"        return; } }\n"
"    int nb = n_cols / 256; int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = mat + (size_t)r * row_bytes;\n"
"    int G = nb * 32; float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        unsigned int aux = *(const unsigned int *)(bp + 66 + 4*sb);\n"
"        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;\n"
"        unsigned char sgn = ksigns_iq2xs_dev[(aux >> (7*l)) & 127];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l]];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l+1]];\n"
"        const float *xb = x + b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float xv0 = xb[j], xv1 = xb[j+4];\n"
"            float w0 = db * (float)g1[j] * ((sgn & (1 << j)) ? -1.0f : 1.0f);\n"
"            float w1 = db * (float)g2[j] * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            sum += w0 * xv0 + w1 * xv1;\n"
"        }\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float ws[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) ws[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += ws[w]; *dst = t; }\n"
"}\n"
"/* ---- fused_ssm_out_gated_iq3xxs: fused gated RMSNorm + ssm_out matvec (IQ3_XXS). */\n"
"__global__ void fused_ssm_out_gated_iq3xxs(float *xb, const unsigned char *w_out,\n"
"        const float *ssm_out, const float *z, const float *norm_w,\n"
"        int n_rows, int n_cols, int dt_rank, int d_state, float eps) {\n"
"    int row = blockIdx.x; if (row >= n_rows) return;\n"
"    int tid = threadIdx.x; int nthreads = blockDim.x;\n"
"    int nb = n_cols / 256; int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = w_out + (size_t)row * row_bytes;\n"
"    int G = nb * 32;\n"
"    float i_d_state = 1.0f / (float)d_state;\n"
"    __shared__ float head_sq[256];\n"
"    if (tid < dt_rank) head_sq[tid] = 0.0f;\n"
"    __syncthreads();\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        int head = b * 8 + sb;\n"
"        if (head >= dt_rank) continue;\n"
"        const float *x0 = ssm_out + b * 256 + sb * 32 + l * 8;\n"
"        float4 xv = *(const float4 *)x0;\n"
"        atomicAdd(&head_sq[head], xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w);\n"
"    }\n"
"    __syncthreads();\n"
"    float sum = 0.0f;\n"
"    for (int g = tid; g < G; g += nthreads) {\n"
"        int b = g >> 5; int rem = g & 31; int sb = rem >> 2; int l = rem & 3;\n"
"        int head = b * 8 + sb;\n"
"        if (head >= dt_rank) continue;\n"
"        float inv_mean = rsqrtf(head_sq[head] * i_d_state + eps);\n"
"        const float *x0 = ssm_out + b * 256 + sb * 32 + l * 8;\n"
"        const float *zx0 = z + b * 256 + sb * 32 + l * 8;\n"
"        float4 xv = *(const float4 *)x0;\n"
"        float4 zv = *(const float4 *)zx0;\n"
"        float nw = norm_w[head];\n"
"        float nv0 = xv.x * inv_mean * nw;\n"
"        float nv1 = xv.y * inv_mean * nw;\n"
"        float nv2 = xv.z * inv_mean * nw;\n"
"        float nv3 = xv.w * inv_mean * nw;\n"
"        float sg0 = 1.0f / (1.0f + __expf(-zv.x));\n"
"        float sg1 = 1.0f / (1.0f + __expf(-zv.y));\n"
"        float sg2 = 1.0f / (1.0f + __expf(-zv.z));\n"
"        float sg3 = 1.0f / (1.0f + __expf(-zv.w));\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        unsigned int aux = *(const unsigned int *)(bp + 66 + 4*sb);\n"
"        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;\n"
"        unsigned char sgn = ksigns_iq2xs_dev[(aux >> (7*l)) & 127];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l]];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*sb+2*l+1]];\n"
"        sum += db * (float)g1[0] * ((sgn & 1) ? -nv0 : nv0) * sg0\n"
"              + db * (float)g1[1] * ((sgn & 2) ? -nv1 : nv1) * sg1\n"
"              + db * (float)g1[2] * ((sgn & 4) ? -nv2 : nv2) * sg2\n"
"              + db * (float)g1[3] * ((sgn & 8) ? -nv3 : nv3) * sg3;\n"
"        sum += db * (float)g2[0] * ((sgn & 16) ? -(nv0*sg0) : (nv0*sg0))\n"
"              + db * (float)g2[1] * ((sgn & 32) ? -(nv1*sg1) : (nv1*sg1))\n"
"              + db * (float)g2[2] * ((sgn & 64) ? -(nv2*sg2) : (nv2*sg2))\n"
"              + db * (float)g2[3] * ((sgn & 128) ? -(nv3*sg3) : (nv3*sg3));\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    __shared__ float wsf[8]; int wid = tid/32, ln = tid%32;\n"
"    if (ln == 0) wsf[wid] = sum; __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < nthreads/32; w++) t += wsf[w]; xb[row] = t; }\n"
"}\n"
"/* ---- moe_route_decode: router logits + topk + softmax + shared-gate sigmoid. */\n"
"/* One block 256 thr: thread e = expert dot, then topk/softmax/sgate. */\n"
"__global__ void moe_route_decode(const float *gate_w, const float *sgate_w,\n"
"        const float *x, int ne, int K, int n_cols,\n"
"        int *out_idx, float *out_w, float *shared_scale) {\n"
"    int tid = threadIdx.x;\n"
"    float my = -1e30f;\n"
"    if (tid < ne) {\n"
"        const float *wr = gate_w + (size_t)tid * n_cols;\n"
"        float sum = 0.0f;\n"
"        for (int j = 0; j < n_cols; j++) sum += wr[j] * x[j];\n"
"        my = sum;\n"
"    }\n"
"    __shared__ float sval[256]; __shared__ int sidx[256]; __shared__ int chosen[64];\n"
"    for (int ki = 0; ki < K; ki++) {\n"
"        float v = my;\n"
"        for (int j = 0; j < ki; j++) if (chosen[j] == tid) v = -1e30f;\n"
"        sval[tid] = v; sidx[tid] = tid; __syncthreads();\n"
"        for (int st = 128; st > 0; st >>= 1) {\n"
"            if (tid < st && tid + st < 256) {\n"
"                if (sval[tid+st] > sval[tid]) { sval[tid]=sval[tid+st]; sidx[tid]=sidx[tid+st]; }\n"
"                else if (sval[tid+st] == sval[tid] && sidx[tid+st] < sidx[tid]) sidx[tid]=sidx[tid+st];\n"
"            } __syncthreads();\n"
"        }\n"
"        if (tid == 0) { chosen[ki]=sidx[0]; out_idx[ki]=sidx[0]; out_w[ki]=sval[0]; }\n"
"        __syncthreads();\n"
"    }\n"
"    if (tid == 0) {\n"
"        float mx = out_w[0]; for (int i=1;i<K;i++) if (out_w[i]>mx) mx=out_w[i];\n"
"        float sum=0; for (int i=0;i<K;i++){ out_w[i]=expf(out_w[i]-mx); sum+=out_w[i]; }\n"
"        float inv=1.0f/sum; for (int i=0;i<K;i++) out_w[i]*=inv;\n"
"    }\n"
"    /* shared-expert gate: 256-thread dot + sigmoid */\n"
"    float p = 0.0f;\n"
"    for (int j = tid; j < n_cols; j += 256) p += sgate_w[j] * x[j];\n"
"    for (int o = 16; o > 0; o >>= 1) p += __shfl_down(p, o);\n"
"    __shared__ float ws[8];\n"
"    if ((tid & 31) == 0) ws[tid >> 5] = p;\n"
"    __syncthreads();\n"
"    if (tid == 0) { float t = 0; for (int w = 0; w < 8; w++) t += ws[w];\n"
"        shared_scale[0] = 1.0f / (1.0f + expf(-t)); }\n"
"}\n"
"/* ---- res_rmsnorm_f32: x += res; xb = rmsnorm(x) * w. One block (n<=4096). */\n"
"__global__ void res_rmsnorm_f32(float *x, const float *res, float *xb,\n"
"        const float *w, int n, float eps) {\n"
"    extern __shared__ float sd[];\n"
"    int tid = threadIdx.x;\n"
"    float ss = 0.0f;\n"
"    for (int i = tid; i < n; i += blockDim.x) {\n"
"        float v = x[i] + res[i]; x[i] = v; ss += v * v;\n"
"    }\n"
"    sd[tid] = ss; __syncthreads();\n"
"    for (int st = blockDim.x / 2; st > 0; st >>= 1) { if (tid < st) sd[tid] += sd[tid + st]; __syncthreads(); }\n"
"    float inv = rsqrtf(sd[0] / n + eps);\n"
"    for (int i = tid; i < n; i += blockDim.x) xb[i] = x[i] * inv * w[i];\n"
"}\n"
"/* ---- moe_router_fused: per-block expert dot, last block does topK+softmax. */\n"
"/* grid = ne+1 blocks x 256 thr. Block ne = shared-gate dot + sigmoid. Counter */\n"
"/* must be 0 before launch; the topk block resets it for the next layer.       */\n"
"__global__ void moe_router_fused(const unsigned short *gate_w, const unsigned short *sgate_w,\n"
"        const float *x, int ne, int K, int n_cols,\n"
"        float *logits, int *out_idx, float *out_w, float *shared_scale,\n"
"        unsigned int *counter) {\n"
"    int e = blockIdx.x; int tid = threadIdx.x;\n"
"    const unsigned short *wr = (e < ne) ? gate_w + (size_t)e * n_cols : sgate_w;\n"
"    float p = 0.0f;\n"
"    for (int j = tid; j < n_cols; j += blockDim.x) {\n"
"        unsigned int b16 = (unsigned int)wr[j] << 16; float w; __builtin_memcpy(&w, &b16, 4);\n"
"        p += w * x[j];\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) p += __shfl_down(p, o);\n"
"    __shared__ float ws[8];\n"
"    if ((tid & 31) == 0) ws[tid >> 5] = p;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float t = 0; for (int w = 0; w < 8; w++) t += ws[w];\n"
"        if (e < ne) logits[e] = t;\n"
"        else shared_scale[0] = 1.0f / (1.0f + expf(-t));\n"
"    }\n"
"    __threadfence();\n"
"    __shared__ unsigned int rank;\n"
"    if (tid == 0) rank = atomicAdd(counter, 1u);\n"
"    __syncthreads();\n"
"    if (rank != (unsigned)ne) return;   /* only the last-finished block continues */\n"
"    /* topK + softmax over logits[0..ne) */\n"
"    __shared__ float sval[256]; __shared__ int sidx[256]; __shared__ int chosen[64];\n"
"    float my = (tid < ne) ? logits[tid] : -1e30f;\n"
"    for (int ki = 0; ki < K; ki++) {\n"
"        float v = my;\n"
"        for (int j = 0; j < ki; j++) if (chosen[j] == tid) v = -1e30f;\n"
"        sval[tid] = v; sidx[tid] = tid; __syncthreads();\n"
"        for (int st = 128; st > 0; st >>= 1) {\n"
"            if (tid < st && tid + st < 256) {\n"
"                if (sval[tid+st] > sval[tid]) { sval[tid]=sval[tid+st]; sidx[tid]=sidx[tid+st]; }\n"
"                else if (sval[tid+st] == sval[tid] && sidx[tid+st] < sidx[tid]) sidx[tid]=sidx[tid+st];\n"
"            } __syncthreads();\n"
"        }\n"
"        if (tid == 0) { chosen[ki]=sidx[0]; out_idx[ki]=sidx[0]; out_w[ki]=sval[0]; }\n"
"        __syncthreads();\n"
"    }\n"
"    if (tid == 0) {\n"
"        float mx = out_w[0]; for (int i=1;i<K;i++) if (out_w[i]>mx) mx=out_w[i];\n"
"        float sum=0; for (int i=0;i<K;i++){ out_w[i]=expf(out_w[i]-mx); sum+=out_w[i]; }\n"
"        float inv=1.0f/sum; for (int i=0;i<K;i++) out_w[i]*=inv;\n"
"        counter[0] = 0;  /* reset for next layer */\n"
"    }\n"
"}\n"
"/* ---- deltanet_step_warp_f32: warp-per-state-row decode recurrence.        */\n"
"/* 8 warps/block; warp owns S[h,r,0..127] in regs (4 f32/lane). 512 blocks   */\n"
"/* (32 heads x 16 row-warps) vs old 32 blocks; single S read+write.          */\n"
"__global__ void deltanet_step_warp_f32(\n"
"    float *state, float *out,\n"
"    const float *Q, const float *K, const float *V,\n"
"    const float *alpha, const float *beta, int d_state) {\n"
"    int warps_per_head = d_state / 8;\n"
"    int gw = blockIdx.x * 8 + (threadIdx.x >> 5);\n"
"    int h = gw / warps_per_head;\n"
"    int rb = gw % warps_per_head;\n"
"    int lane = threadIdx.x & 31;\n"
"    int rows_per_warp = 8;  /* 8 rows handled per warp, 4 lanes each */\n"
"    int r = rb * rows_per_warp + (lane >> 2);\n"
"    int c0 = (lane & 3) * 32;\n"
"    float *S = state + ((size_t)h * d_state + r) * d_state + c0;\n"
"    const float *k = K + h * d_state + c0;\n"
"    const float *q = Q + h * d_state + c0;\n"
"    float decay = expf(alpha[h]);\n"
"    float b = beta[h];\n"
"    float v_r = V[h * d_state + r];\n"
"    float scale = rsqrtf((float)d_state);\n"
"    float s_[32], k_[32], q_[32];\n"
"    float sk = 0.0f;\n"
"    for (int j = 0; j < 32; j += 4) {\n"
"        float4 sv = *(const float4 *)&S[j];\n"
"        float4 kv = *(const float4 *)&k[j];\n"
"        float4 qv = *(const float4 *)&q[j];\n"
"        s_[j+0]=sv.x*decay; s_[j+1]=sv.y*decay; s_[j+2]=sv.z*decay; s_[j+3]=sv.w*decay;\n"
"        k_[j+0]=kv.x; k_[j+1]=kv.y; k_[j+2]=kv.z; k_[j+3]=kv.w;\n"
"        q_[j+0]=qv.x; q_[j+1]=qv.y; q_[j+2]=qv.z; q_[j+3]=qv.w;\n"
"        sk += s_[j]*k_[j] + s_[j+1]*k_[j+1] + s_[j+2]*k_[j+2] + s_[j+3]*k_[j+3];\n"
"    }\n"
"    /* reduce sk across the 4 lanes of this row (lanes r*4..r*4+3) */\n"
"    sk += __shfl_xor(sk, 1); sk += __shfl_xor(sk, 2);\n"
"    float delta = (v_r - sk) * b;\n"
"    float o = 0.0f;\n"
"    for (int j = 0; j < 32; j += 4) {\n"
"        s_[j+0] += delta * k_[j+0]; s_[j+1] += delta * k_[j+1];\n"
"        s_[j+2] += delta * k_[j+2]; s_[j+3] += delta * k_[j+3];\n"
"        o += s_[j]*q_[j] + s_[j+1]*q_[j+1] + s_[j+2]*q_[j+2] + s_[j+3]*q_[j+3];\n"
"        float4 sv = { s_[j+0], s_[j+1], s_[j+2], s_[j+3] };\n"
"        *(float4 *)&S[j] = sv;\n"
"    }\n"
"    o += __shfl_xor(o, 1); o += __shfl_xor(o, 2);\n"
"    if ((lane & 3) == 0) out[h * d_state + r] = o * scale;\n"
"}\n"
"/* ---- deltanet_step_batch_warp_f32: warp-per-row M-step recurrence.        */\n"
"/* 4 lanes/row x 32 cols in regs across M tokens; 8 rows/warp, 8 warps/blk.  */\n"
"__global__ void deltanet_step_batch_warp_f32(\n"
"    float *state, float *out_batch,\n"
"    const float *Q_batch, const float *K_batch, const float *V_batch,\n"
"    const float *alpha_batch, const float *beta_batch,\n"
"    int dt_rank, int d_state, int v_row_stride, int M) {\n"
"    int warps_per_head = d_state / 8;\n"
"    int gw = blockIdx.x * 8 + (threadIdx.x >> 5);\n"
"    int h = gw / warps_per_head;\n"
"    int rb = gw % warps_per_head;\n"
"    int lane = threadIdx.x & 31;\n"
"    int r = rb * 8 + (lane >> 2);\n"
"    int c0 = (lane & 3) * 32;\n"
"    float *S = state + ((size_t)h * d_state + r) * d_state + c0;\n"
"    float s_[32];\n"
"    for (int j = 0; j < 32; j += 4) *(float4 *)&s_[j] = *(const float4 *)&S[j];\n"
"    float scale = rsqrtf((float)d_state);\n"
"    size_t per_tok = (size_t)dt_rank * d_state;\n"
"    for (int m = 0; m < M; m++) {\n"
"        const float *k = K_batch + (size_t)m * per_tok + h * d_state + c0;\n"
"        const float *q = Q_batch + (size_t)m * per_tok + h * d_state + c0;\n"
"        float v_r   = V_batch[(size_t)m * v_row_stride + h * d_state + r];\n"
"        float decay = __expf(alpha_batch[(size_t)m * dt_rank + h]);\n"
"        float b     = beta_batch[(size_t)m * dt_rank + h];\n"
"        float k_[32], q_[32];\n"
"        float sk = 0.0f;\n"
"        for (int j = 0; j < 32; j += 4) {\n"
"            float4 kv = *(const float4 *)&k[j];\n"
"            float4 qv = *(const float4 *)&q[j];\n"
"            k_[j]=kv.x;k_[j+1]=kv.y;k_[j+2]=kv.z;k_[j+3]=kv.w;\n"
"            q_[j]=qv.x;q_[j+1]=qv.y;q_[j+2]=qv.z;q_[j+3]=qv.w;\n"
"            s_[j]*=decay;s_[j+1]*=decay;s_[j+2]*=decay;s_[j+3]*=decay;\n"
"            sk += s_[j]*k_[j]+s_[j+1]*k_[j+1]+s_[j+2]*k_[j+2]+s_[j+3]*k_[j+3];\n"
"        }\n"
"        sk += __shfl_xor(sk, 1); sk += __shfl_xor(sk, 2);\n"
"        float delta = (v_r - sk) * b;\n"
"        float o = 0.0f;\n"
"        for (int j = 0; j < 32; j += 4) {\n"
"            s_[j]+=delta*k_[j];s_[j+1]+=delta*k_[j+1];s_[j+2]+=delta*k_[j+2];s_[j+3]+=delta*k_[j+3];\n"
"            o += s_[j]*q_[j]+s_[j+1]*q_[j+1]+s_[j+2]*q_[j+2]+s_[j+3]*q_[j+3];\n"
"        }\n"
"        o += __shfl_xor(o, 1); o += __shfl_xor(o, 2);\n"
"        if ((lane & 3) == 0) out_batch[(size_t)m * per_tok + h * d_state + r] = o * scale;\n"
"    }\n"
"    for (int j = 0; j < 32; j += 4) *(float4 *)&S[j] = *(const float4 *)&s_[j];\n"
"}\n"
"/* ---- ssm_prep_f32: fused decode SSM aux chain (conv+silu+state, l2norm Q/K,  */\n"
"/* repeat-tile to Q_exp/K_exp, alpha softplus, beta sigmoid). One launch       */\n"
"/* replaces 7. grid = n_group blocks x 256 thr. Layout: conv_out =             */\n"
"/* [Q: n_group*d_state][K: n_group*d_state][V: rest of qkv_dim].               */\n"
"__global__ void ssm_prep_f32(float *conv_out, float *conv_state,\n"
"        const float *qkv_in, const float *conv_w,\n"
"        float *alpha, float *beta, const float *dt_bias, const float *a_arr,\n"
"        float *Q_exp, float *K_exp,\n"
"        int qkv_dim, int conv_k, int d_state, int n_group, int dt_rank, float eps) {\n"
"    int g = blockIdx.x; int tid = threadIdx.x;\n"
"    int v_per_blk = (qkv_dim - 2 * n_group * d_state) / n_group;\n"
"    /* conv channels owned by this block: Q group g, K group g, V slice g */\n"
"    for (int t = tid; t < 2 * d_state + v_per_blk; t += blockDim.x) {\n"
"        int j;\n"
"        if (t < d_state) j = g * d_state + t;\n"
"        else if (t < 2 * d_state) j = n_group * d_state + g * d_state + (t - d_state);\n"
"        else j = 2 * n_group * d_state + g * v_per_blk + (t - 2 * d_state);\n"
"        float sum = 0.0f;\n"
"        for (int f = 0; f < conv_k - 1; f++)\n"
"            sum += conv_w[j * conv_k + f] * conv_state[f * qkv_dim + j];\n"
"        sum += conv_w[j * conv_k + (conv_k - 1)] * qkv_in[j];\n"
"        conv_out[j] = sum / (1.0f + expf(-sum));\n"
"        for (int f = 0; f < conv_k - 2; f++)\n"
"            conv_state[f * qkv_dim + j] = conv_state[(f + 1) * qkv_dim + j];\n"
"        conv_state[(conv_k - 2) * qkv_dim + j] = qkv_in[j];\n"
"    }\n"
"    /* alpha/beta prep on block 0 (dt_rank small) */\n"
"    if (g == 0) {\n"
"        for (int i = tid; i < dt_rank; i += blockDim.x) {\n"
"            float x = alpha[i] + dt_bias[i];\n"
"            float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));\n"
"            alpha[i] = sp * a_arr[i];\n"
"            beta[i]  = 1.0f / (1.0f + expf(-beta[i]));\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"    /* L2 norm of Q group g and K group g (d_state each) */\n"
"    __shared__ float sq[256], sk[256];\n"
"    float aq = 0.0f, ak = 0.0f;\n"
"    float *Qg = conv_out + g * d_state;\n"
"    float *Kg = conv_out + n_group * d_state + g * d_state;\n"
"    for (int i = tid; i < d_state; i += blockDim.x) { aq += Qg[i]*Qg[i]; ak += Kg[i]*Kg[i]; }\n"
"    sq[tid] = aq; sk[tid] = ak; __syncthreads();\n"
"    for (int s = blockDim.x/2; s > 0; s >>= 1) {\n"
"        if (tid < s) { sq[tid] += sq[tid+s]; sk[tid] += sk[tid+s]; } __syncthreads();\n"
"    }\n"
"    float iq = rsqrtf(sq[0] + eps), ik = rsqrtf(sk[0] + eps);\n"
"    for (int i = tid; i < d_state; i += blockDim.x) { Qg[i] *= iq; Kg[i] *= ik; }\n"
"    __syncthreads();\n"
"    /* repeat-tile: every head h with h % n_group == g gets this group's row */\n"
"    for (int h = g; h < dt_rank; h += n_group) {\n"
"        for (int i = tid; i < d_state; i += blockDim.x) {\n"
"            Q_exp[h * d_state + i] = Qg[i];\n"
"            K_exp[h * d_state + i] = Kg[i];\n"
"        }\n"
"    }\n"
"}\n"
"/* ---- Fused shared-expert decode (Q6_K): gate+up+silu, then down+scale-add. ---- */\n"
"__global__ void shexp_gateup_silu_q6k(float *out, const unsigned char *gw,\n"
"        const unsigned char *uw, const float *x, int eff, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id; if (row >= eff) return;\n"
"    int nb = n_cols / 256; int row_bytes = nb * 210;\n"
"    int G = nb * 16;\n"
"    float sg = 0.0f, su = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 4; int rem = g & 15; int half = rem >> 3; int lp = rem & 7;\n"
"        sg += q6k_dot4(gw + (size_t)row * row_bytes + b * 210, x + b * 256, half, lp);\n"
"        su += q6k_dot4(uw + (size_t)row * row_bytes + b * 210, x + b * 256, half, lp);\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) { sg += __shfl_down(sg, o); su += __shfl_down(su, o); }\n"
"    if (lane == 0) out[row] = (sg / (1.0f + expf(-sg))) * su;\n"
"}\n"
"__global__ void shexp_down_accum_q6k(float *accum, const unsigned char *dw,\n"
"        const float *x, int n_embd, int eff, const float *scale_ptr) {\n"
"    int warp_id = threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id; if (row >= n_embd) return;\n"
"    int nb = eff / 256; int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = dw + (size_t)row * row_bytes;\n"
"    int G = nb * 64;\n"
"    float sum = 0.0f;\n"
"    for (int g = lane; g < G; g += 32) {\n"
"        int b = g >> 6; int rem = g & 63; int half = rem >> 5; int l = rem & 31;\n"
"        const unsigned char *bp = row_ptr + b * 210;\n"
"        const unsigned char *ql = bp + half * 64;\n"
"        const unsigned char *qh = bp + 128 + half * 32;\n"
"        const signed char *sc = (const signed char *)(bp + 192 + half * 8);\n"
"        float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"        const float *xb = x + b * 256 + half * 128;\n"
"        int hi = (l < 16) ? 0 : 1;\n"
"        float sA = d * (float)sc[0 + hi], sB = d * (float)sc[2 + hi];\n"
"        float sC = d * (float)sc[4 + hi], sD = d * (float)sc[6 + hi];\n"
"        int q1 = (int)((ql[l]    & 0xF) | (((qh[l]>>0)&3)<<4)) - 32;\n"
"        int q2 = (int)((ql[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)) - 32;\n"
"        int q3 = (int)((ql[l]    >> 4)  | (((qh[l]>>4)&3)<<4)) - 32;\n"
"        int q4 = (int)((ql[l+32] >> 4)  | (((qh[l]>>6)&3)<<4)) - 32;\n"
"        sum += sA*q1*xb[l] + sB*q2*xb[l+32] + sC*q3*xb[l+64] + sD*q4*xb[l+96];\n"
"    }\n"
"    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down(sum, o);\n"
"    if (lane == 0) accum[row] += scale_ptr[0] * sum;\n"
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
"/* Per-call dequant of IQ2_XXS (66 B/block) to BF16. */\n"
"__global__ void dequant_iq2_xxs_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                          int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 66;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 66;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"    int tid = threadIdx.x;\n"
"    int ib32 = tid >> 5;\n"
"    int sub  = tid & 31;\n"
"    int l    = sub >> 3;\n"
"    int j    = sub & 7;\n"
"    unsigned int aux0 = qs[4*ib32] | ((unsigned int)qs[4*ib32+1] << 16);\n"
"    unsigned int aux1 = qs[4*ib32+2] | ((unsigned int)qs[4*ib32+3] << 16);\n"
"    float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;\n"
"    const unsigned char *aux8 = (const unsigned char *)&aux0;\n"
"    const unsigned char *grid = (const unsigned char *)&iq2xxs_grid_dev[aux8[l]];\n"
"    unsigned char signs = ksigns_iq2xs_dev[(aux1 >> (7*l)) & 127];\n"
"    float val = db * (float)grid[j] * ((signs & (1 << j)) ? -1.0f : 1.0f);\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + tid;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"\n"
"/* Per-call dequant of IQ1_S (50 B/block) to BF16. */\n"
"__global__ void dequant_iq1_s_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                        int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 50;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 50;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    const unsigned char *qs  = bp + 2;\n"
"    const unsigned short *qh = (const unsigned short *)(bp + 2 + 32);\n"
"    const float IQ1S_DELTA = 0.125f;\n"
"    int tid   = threadIdx.x;\n"
"    int ib    = tid >> 5;\n"
"    int sub   = tid & 31;\n"
"    int l     = sub >> 3;\n"
"    int j     = sub & 7;\n"
"    float dl  = d * (float)(2 * ((qh[ib] >> 12) & 7) + 1);\n"
"    float delta = (qh[ib] & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;\n"
"    int grid_idx = qs[ib * 4 + l] | (((qh[ib] >> (3*l)) & 7) << 8);\n"
"    const signed char *grid = (const signed char *)&iq1s_grid_dev[grid_idx];\n"
"    float val = dl * ((float)grid[j] + delta);\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + tid;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"\n"
"/* Per-call dequant of TQ1_0 (54 B/block) to BF16. */\n"
"__global__ void dequant_tq1_0_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                        int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    int b   = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 54;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 54;\n"
"    float d = half_to_float(*(const half_raw *)(bp + 52));\n"
"    const unsigned char *qs  = bp;\n"
"    const unsigned char *qh_ptr = bp + 48;\n"
"    const unsigned char pow3[6] = {1, 3, 9, 27, 81, 243};\n"
"    int tid = threadIdx.x;\n"
"    unsigned char q; int xi;\n"
"    if (tid < 240) {\n"
"        int j = tid / 5, n = tid % 5;\n"
"        q = qs[j] * pow3[n];\n"
"    } else {\n"
"        int j = (tid - 240) / 4, n = (tid - 240) % 4;\n"
"        q = qh_ptr[j] * pow3[n];\n"
"    }\n"
"    xi = (((unsigned short)q * 3) >> 8);\n"
"    float val = d * (float)(xi - 1);\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + tid;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
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
"/* Per-call dequant of Q2_K (84 B/block) to BF16. One thread per output column;  */\n"
"/* column order matches matvec_q2_K_f32's xb traversal. Layout: scales[16] +     */\n"
"/* qs[64] + d(f16)@80 + dmin(f16)@82. */\n"
"__global__ void dequant_q2_K_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                       int n_rows, int n_cols) {\n"
"    int row = blockIdx.x; int b = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 84;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 84;\n"
"    const unsigned char *scales = bp;\n"
"    const unsigned char *qs = bp + 16;\n"
"    float d = half_to_float(*(const half_raw *)(bp + 80));\n"
"    float dmin = half_to_float(*(const half_raw *)(bp + 82));\n"
"    int c = threadIdx.x;\n"
"    int n0 = c >> 7; int local = c & 127;\n"
"    int j = local >> 5; int s16 = (local >> 4) & 1; int l = local & 15;\n"
"    int is = n0 * 8 + j * 2 + s16;\n"
"    unsigned char sc = scales[is];\n"
"    float dl = d * (float)(sc & 0xF);\n"
"    float ml = dmin * (float)(sc >> 4);\n"
"    int qs_byte = n0 * 32 + s16 * 16 + l;\n"
"    int shift = j * 2;\n"
"    float val = dl * (float)((qs[qs_byte] >> shift) & 3) - ml;\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + c;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"/* Per-call dequant of Q3_K (110 B/block) to BF16. Layout: hmask[32] + qs[64] +  */\n"
"/* scales[12]@96 + d(f16)@108. Column order matches matvec_q3_K_f32. */\n"
"__global__ void dequant_q3_K_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                       int n_rows, int n_cols) {\n"
"    int row = blockIdx.x; int b = blockIdx.y;\n"
"    int n_blocks_per_row = n_cols / 256;\n"
"    int row_bytes = n_blocks_per_row * 110;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + b * 110;\n"
"    const unsigned char *hm = bp;\n"
"    const unsigned char *qs = bp + 32;\n"
"    const unsigned char *raw_sc = bp + 96;\n"
"    float d_all = half_to_float(*(const half_raw *)(bp + 108));\n"
"    unsigned int a0 = raw_sc[0]|(raw_sc[1]<<8)|(raw_sc[2]<<16)|(raw_sc[3]<<24);\n"
"    unsigned int a1 = raw_sc[4]|(raw_sc[5]<<8)|(raw_sc[6]<<16)|(raw_sc[7]<<24);\n"
"    unsigned int tmp = raw_sc[8]|(raw_sc[9]<<8)|(raw_sc[10]<<16)|(raw_sc[11]<<24);\n"
"    unsigned int km1 = 0x03030303u, km2 = 0x0f0f0f0fu;\n"
"    unsigned int aux[4];\n"
"    aux[0] = (a0 & km2) | (((tmp >> 0) & km1) << 4);\n"
"    aux[1] = (a1 & km2) | (((tmp >> 2) & km1) << 4);\n"
"    aux[2] = ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4);\n"
"    aux[3] = ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4);\n"
"    const signed char *scales = (const signed char *)aux;\n"
"    int c = threadIdx.x;\n"
"    int n0 = c >> 7; int local = c & 127;\n"
"    int j = local >> 5; int s16 = (local >> 4) & 1; int l = local & 15;\n"
"    int is = n0 * 8 + j * 2 + s16;\n"
"    int shift = j * 2;\n"
"    unsigned char m_bit = (unsigned char)(1u << (n0 * 4 + j));\n"
"    int hm_idx = s16 * 16 + l;\n"
"    int qs_byte = n0 * 32 + s16 * 16 + l;\n"
"    float dl = d_all * (float)((int)scales[is] - 32);\n"
"    int qv = ((qs[qs_byte] >> shift) & 3) - ((hm[hm_idx] & m_bit) ? 0 : 4);\n"
"    float val = dl * (float)qv;\n"
"    size_t out_idx = (size_t)row * n_cols + (size_t)b * 256 + c;\n"
"    dst[out_idx] = f32_to_bf16(val);\n"
"}\n"
"/* Per-call dequant of Q8_0 to BF16. The runner uploads Q8_0 in a PADDED 36-byte  */\n"
"/* block: [d(f16) 2B][pad 2B][int8 qs[32]] (see upload_q8_0_raw / matvec_q8_0_f32).*/\n"
"/* Same (n_rows, ceil(n_cols/256), 256-thread) launch; one element/thread. */\n"
"__global__ void dequant_q8_0_to_bf16(bf16_raw *dst, const unsigned char *mat,\n"
"                                       int n_rows, int n_cols) {\n"
"    int row = blockIdx.x; int b = blockIdx.y;\n"
"    int c = b * 256 + threadIdx.x;       /* global column 0..n_cols-1 */\n"
"    if (c >= n_cols) return;\n"
"    int n_blocks_per_row = n_cols / 32;\n"
"    int row_bytes = n_blocks_per_row * 36;\n"
"    const unsigned char *rp = mat + (size_t)row * row_bytes;\n"
"    int blk = c >> 5; int within = c & 31;\n"
"    const unsigned char *bp = rp + (size_t)blk * 36;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    signed char q = (signed char)bp[4 + within];\n"
"    float val = d * (float)q;\n"
"    size_t out_idx = (size_t)row * n_cols + c;\n"
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
"typedef unsigned short bf16x8 __attribute__((ext_vector_type(8)));\n"
"\n"
"/* ---- Self-owned BF16 GEMM (no hipBLASLt): Y[M,N]f32 = X[M,K]bf16 x W[N,K]^T. */\n"
"/* 128x128 CTA, 4 waves, WMMA 16x16x16, bounds-checked tails (ported vlm).     */\n"
"__global__ void gemm_bf16_own(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                              int N, int K, int M) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ unsigned short smA[128*32];\n"
"    __shared__ unsigned short smB[128*32];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    int interior = (cta_m0 + 128 <= M) && (cta_n0 + 128 <= N) && ((K & 31) == 0);\n"
"    for (int k = 0; k < K; k += 32) {\n"
"        if (interior) {\n"
"            int er = tid >> 1, ek = (tid & 1) * 16;\n"
"            bf16x8 *da = (bf16x8 *)&smA[er * 32 + ek];\n"
"            const bf16x8 *sa = (const bf16x8 *)&X[(size_t)(cta_m0 + er) * K + k + ek];\n"
"            da[0] = sa[0]; da[1] = sa[1];\n"
"            bf16x8 *db = (bf16x8 *)&smB[er * 32 + ek];\n"
"            const bf16x8 *sb = (const bf16x8 *)&W[(size_t)(cta_n0 + er) * K + k + ek];\n"
"            db[0] = sb[0]; db[1] = sb[1];\n"
"        } else {\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int e = tid * 16 + it;\n"
"            int er = e >> 5, ek = e & 31;\n"
"            int row = cta_m0 + er, kp = k + ek;\n"
"            smA[e] = (row < M && kp < K) ? X[(size_t)row * K + kp] : 0;\n"
"        }\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int e = tid * 16 + it;\n"
"            int er = e >> 5, ek = e & 31;\n"
"            int col = cta_n0 + er, kp = k + ek;\n"
"            smB[e] = (col < N && kp < K) ? W[(size_t)col * K + kp] : 0;\n"
"        }\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            bf16x8 a0,a1,a2,a3,b0,b1;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                a0[i]=smA[(a_base+0 +idx)*32+kk0+k_off+i];\n"
"                a1[i]=smA[(a_base+16+idx)*32+kk0+k_off+i];\n"
"                a2[i]=smA[(a_base+32+idx)*32+kk0+k_off+i];\n"
"                a3[i]=smA[(a_base+48+idx)*32+kk0+k_off+i];\n"
"                b0[i]=smB[(b_base+0 +idx)*32+kk0+k_off+i];\n"
"                b1[i]=smB[(b_base+16+idx)*32+kk0+k_off+i];\n"
"            }\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        if (col >= N) continue;\n"
"        float8 acc = *accs[t];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row >= M) continue;\n"
"            Y[(size_t)row * N + col] = acc[i];\n"
"        }\n"
"    }\n"
"}\n"
"/* ---- BF16 GEMM with double-buffered LDS (hides global load behind WMMA).       */\n"
"__global__ void gemm_bf16_own_db(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                int N, int K, int M) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5; int lane = tid & 31;\n"
"    int wM = wave_id & 1, wN = wave_id >> 1;\n"
"    int half = lane >> 4, idx = lane & 15, k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128, cta_n0 = blockIdx.x * 128;\n"
"    __shared__ unsigned short smA[2][128*32], smB[2][128*32];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    int interior = (cta_m0 + 128 <= M) && (cta_n0 + 128 <= N) && ((K & 31) == 0);\n"
"    int buf = 0;\n"
"    if (interior) {\n"
"        int er = tid >> 1, ek = (tid & 1) * 16;\n"
"        *(bf16x8 *)&smA[0][er*32+ek]   = *(const bf16x8 *)&X[(size_t)(cta_m0+er)*K+ek];\n"
"        *(bf16x8 *)&smA[0][er*32+ek+8] = *(const bf16x8 *)&X[(size_t)(cta_m0+er)*K+ek+8];\n"
"        *(bf16x8 *)&smB[0][er*32+ek]   = *(const bf16x8 *)&W[(size_t)(cta_n0+er)*K+ek];\n"
"        *(bf16x8 *)&smB[0][er*32+ek+8] = *(const bf16x8 *)&W[(size_t)(cta_n0+er)*K+ek+8];\n"
"    } else {\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int e = tid*16+it, er = e>>5, ek = e&31;\n"
"            int row = cta_m0+er, kp = ek;\n"
"            smA[0][e] = (row<M && kp<K) ? X[(size_t)row*K+kp] : 0; }\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int e = tid*16+it, er = e>>5, ek = e&31;\n"
"            int col = cta_n0+er, kp = ek;\n"
"            smB[0][e] = (col<N && kp<K) ? W[(size_t)col*K+kp] : 0; }\n"
"    }\n"
"    __syncthreads();\n"
"    for (int k = 0; k < K; k += 64) {\n"
"        int nbuf = buf ^ 1;\n"
"        int a_base = wM*64, b_base = wN*32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            bf16x8 a0,a1,a2,a3,b0,b1;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                a0[i]=smA[buf][(a_base+0+idx)*32+kk0+k_off+i];\n"
"                a1[i]=smA[buf][(a_base+16+idx)*32+kk0+k_off+i];\n"
"                a2[i]=smA[buf][(a_base+32+idx)*32+kk0+k_off+i];\n"
"                a3[i]=smA[buf][(a_base+48+idx)*32+kk0+k_off+i];\n"
"                b0[i]=smB[buf][(b_base+0+idx)*32+kk0+k_off+i];\n"
"                b1[i]=smB[buf][(b_base+16+idx)*32+kk0+k_off+i];\n"
"            }\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        int k2 = k + 32;\n"
"        if (k2 < K) {\n"
"            if (interior) {\n"
"                int er = tid>>1, ek = (tid&1)*16;\n"
"                *(bf16x8 *)&smA[nbuf][er*32+ek]   = *(const bf16x8 *)&X[(size_t)(cta_m0+er)*K+k2+ek];\n"
"                *(bf16x8 *)&smA[nbuf][er*32+ek+8] = *(const bf16x8 *)&X[(size_t)(cta_m0+er)*K+k2+ek+8];\n"
"                *(bf16x8 *)&smB[nbuf][er*32+ek]   = *(const bf16x8 *)&W[(size_t)(cta_n0+er)*K+k2+ek];\n"
"                *(bf16x8 *)&smB[nbuf][er*32+ek+8] = *(const bf16x8 *)&W[(size_t)(cta_n0+er)*K+k2+ek+8];\n"
"            } else {\n"
"                for (int it = 0; it < 16; it++) {\n"
"                    int e = tid*16+it, er = e>>5, ek = e&31;\n"
"                    int row = cta_m0+er, kp = k2+ek;\n"
"                    smA[nbuf][e] = (row<M && kp<K) ? X[(size_t)row*K+kp] : 0; }\n"
"                for (int it = 0; it < 16; it++) {\n"
"                    int e = tid*16+it, er = e>>5, ek = e&31;\n"
"                    int col = cta_n0+er, kp = k2+ek;\n"
"                    smB[nbuf][e] = (col<N && kp<K) ? W[(size_t)col*K+kp] : 0; }\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"        a_base = wM*64; b_base = wN*32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            bf16x8 a0,a1,a2,a3,b0,b1;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                a0[i]=smA[nbuf][(a_base+0+idx)*32+kk0+k_off+i];\n"
"                a1[i]=smA[nbuf][(a_base+16+idx)*32+kk0+k_off+i];\n"
"                a2[i]=smA[nbuf][(a_base+32+idx)*32+kk0+k_off+i];\n"
"                a3[i]=smA[nbuf][(a_base+48+idx)*32+kk0+k_off+i];\n"
"                b0[i]=smB[nbuf][(b_base+0+idx)*32+kk0+k_off+i];\n"
"                b1[i]=smB[nbuf][(b_base+16+idx)*32+kk0+k_off+i];\n"
"            }\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        int k3 = k + 64;\n"
"        if (k3 < K) {\n"
"            if (interior) {\n"
"                int er = tid>>1, ek = (tid&1)*16;\n"
"                *(bf16x8 *)&smA[buf][er*32+ek]   = *(const bf16x8 *)&X[(size_t)(cta_m0+er)*K+k3+ek];\n"
"                *(bf16x8 *)&smA[buf][er*32+ek+8] = *(const bf16x8 *)&X[(size_t)(cta_m0+er)*K+k3+ek+8];\n"
"                *(bf16x8 *)&smB[buf][er*32+ek]   = *(const bf16x8 *)&W[(size_t)(cta_n0+er)*K+k3+ek];\n"
"                *(bf16x8 *)&smB[buf][er*32+ek+8] = *(const bf16x8 *)&W[(size_t)(cta_n0+er)*K+k3+ek+8];\n"
"            } else {\n"
"                for (int it = 0; it < 16; it++) {\n"
"                    int e = tid*16+it, er = e>>5, ek = e&31;\n"
"                    int row = cta_m0+er, kp = k3+ek;\n"
"                    smA[buf][e] = (row<M && kp<K) ? X[(size_t)row*K+kp] : 0; }\n"
"                for (int it = 0; it < 16; it++) {\n"
"                    int e = tid*16+it, er = e>>5, ek = e&31;\n"
"                    int col = cta_n0+er, kp = k3+ek;\n"
"                    smB[buf][e] = (col<N && kp<K) ? W[(size_t)col*K+kp] : 0; }\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64, wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48}, ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        if (col >= N) continue;\n"
"        float8 acc = *accs[t];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row >= M) continue;\n"
"            Y[(size_t)row * N + col] = acc[i];\n"
"        }\n"
"    }\n"
"}\n"
"/* ---- Grouped MoE prefill kernels: blockIdx.y/z = expert. One launch per layer. */\n"
"/* Dequant ALL experts' IQ2_S weights -> bf16 (skips experts with 0 tokens). */\n"
"__global__ void dequant_iq2s_all(bf16_raw *dst, const unsigned char *base,\n"
"        const int *offs, int rows, int cols, long long stride, int bm) {\n"
"    int e = blockIdx.y;\n"
"    if (offs[e + 1] == offs[e]) return;\n"
"    int row = blockIdx.x * 8 + threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    if (row >= rows) return;\n"
"    const unsigned char *mat = base + (long long)e * stride;\n"
"    int nb = cols / 256;\n"
"    bf16_raw *orow = dst + ((size_t)e * rows + row) * cols;\n"
"    for (int g = lane; g < nb * 32; g += 32) {\n"
"        int b = g >> 5; int gl = g & 31; int ib32 = gl >> 2; int l = gl & 3;\n"
"        const unsigned char *bp = bm ? (mat + (size_t)b * rows * 82 + (size_t)row * 82)\n"
"                                     : (mat + (size_t)row * nb * 82 + (size_t)b * 82);\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char scale = bp[74 + ib32];\n"
"        float db = (l < 2) ? d * (0.5f + (float)(scale & 0xf)) * 0.25f\n"
"                           : d * (0.5f + (float)(scale >>  4)) * 0.25f;\n"
"        int gi = bp[2 + ib32*4 + l] | ((bp[66 + ib32] << (8 - 2*l)) & 0x300);\n"
"        const unsigned char *grid = (const unsigned char *)&iq2s_grid_dev[gi];\n"
"        unsigned char s = bp[34 + ib32*4 + l];\n"
"        int o = b * 256 + ib32 * 32 + l * 8;\n"
"        for (int j = 0; j < 8; j++)\n"
"            orow[o + j] = f32_to_bf16(db * (float)grid[j] * ((s & (1 << j)) ? -1.0f : 1.0f));\n"
"    }\n"
"}\n"
"__global__ void dequant_iq3s_all(bf16_raw *dst, const unsigned char *base,\n"
"        const int *offs, int rows, int cols, long long stride, int bm) {\n"
"    int e = blockIdx.y;\n"
"    if (offs[e + 1] == offs[e]) return;\n"
"    int row = blockIdx.x * 8 + threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    if (row >= rows) return;\n"
"    const unsigned char *mat = base + (long long)e * stride;\n"
"    int nb = cols / 256;\n"
"    bf16_raw *orow = dst + ((size_t)e * rows + row) * cols;\n"
"    for (int g = lane; g < nb * 32; g += 32) {\n"
"        int b = g >> 5; int g32 = g & 31; int sb = g32 >> 2; int l = g32 & 3;\n"
"        const unsigned char *bp = bm ? (mat + (size_t)b * rows * 110 + (size_t)row * 110)\n"
"                                     : (mat + (size_t)row * nb * 110 + (size_t)b * 110);\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char sc = bp[106 + (sb >> 1)];\n"
"        float db = d * (float)(1 + 2 * ((sb & 1) ? (sc >> 4) : (sc & 0xf)));\n"
"        unsigned char qh = bp[66 + sb];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3s_grid_dev[bp[2+sb*8+2*l]   | ((qh << (8-2*l)) & 256)];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3s_grid_dev[bp[2+sb*8+2*l+1] | ((qh << (7-2*l)) & 256)];\n"
"        unsigned char s = bp[74 + sb*4 + l];\n"
"        int o = b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            orow[o + j]     = f32_to_bf16(db * (float)g1[j] * ((s & (1 << j)) ? -1.0f : 1.0f));\n"
"            orow[o + 4 + j] = f32_to_bf16(db * (float)g2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f));\n"
"        }\n"
"    }\n"
"}\n"
"__global__ void dequant_iq2_xxs_all(bf16_raw *dst, const unsigned char *base,\n"
"        const int *offs, int rows, int cols, long long stride, int bm) {\n"
"    int e = blockIdx.y;\n"
"    if (offs[e + 1] == offs[e]) return;\n"
"    int row = blockIdx.x * 8 + threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    if (row >= rows) return;\n"
"    const unsigned char *mat = base + (long long)e * stride;\n"
"    int nb = cols / 256;\n"
"    bf16_raw *orow = dst + ((size_t)e * rows + row) * cols;\n"
"    for (int g = lane; g < nb * 32; g += 32) {\n"
"        int b = g >> 5; int gl = g & 31; int ib32 = gl >> 2; int l = gl & 3;\n"
"        const unsigned char *bp = bm ? (mat + (size_t)b * rows * 66 + (size_t)row * 66)\n"
"                                     : (mat + (size_t)row * nb * 66 + (size_t)b * 66);\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"        unsigned int aux0 = qs[4*ib32] | ((unsigned int)qs[4*ib32+1] << 16);\n"
"        unsigned int aux1 = qs[4*ib32+2] | ((unsigned int)qs[4*ib32+3] << 16);\n"
"        float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;\n"
"        const unsigned char *aux8 = (const unsigned char *)&aux0;\n"
"        const unsigned char *grid = (const unsigned char *)&iq2xxs_grid_dev[aux8[l]];\n"
"        unsigned char signs = ksigns_iq2xs_dev[(aux1 >> (7*l)) & 127];\n"
"        int o = b * 256 + ib32 * 32 + l * 8;\n"
"        for (int j = 0; j < 8; j++)\n"
"            orow[o + j] = f32_to_bf16(db * (float)grid[j] * ((signs & (1 << j)) ? -1.0f : 1.0f));\n"
"    }\n"
"}\n"
"__global__ void dequant_iq3_xxs_all(bf16_raw *dst, const unsigned char *base,\n"
"        const int *offs, int rows, int cols, long long stride, int bm) {\n"
"    int e = blockIdx.y;\n"
"    if (offs[e + 1] == offs[e]) return;\n"
"    int row = blockIdx.x * 8 + threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    if (row >= rows) return;\n"
"    const unsigned char *mat = base + (long long)e * stride;\n"
"    int nb = cols / 256;\n"
"    bf16_raw *orow = dst + ((size_t)e * rows + row) * cols;\n"
"    for (int g = lane; g < nb * 32; g += 32) {\n"
"        int b = g >> 5; int gl = g & 31; int ib32 = gl >> 2; int l = gl & 3;\n"
"        const unsigned char *bp = bm ? (mat + (size_t)b * rows * 98 + (size_t)row * 98)\n"
"                                     : (mat + (size_t)row * nb * 98 + (size_t)b * 98);\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        unsigned int aux32 = *(const unsigned int *)(bp + 66 + 4*ib32);\n"
"        float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;\n"
"        unsigned char signs = ksigns_iq2xs_dev[(aux32 >> (7*l)) & 127];\n"
"        const unsigned char *grid1 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*ib32 + 2*l]];\n"
"        const unsigned char *grid2 = (const unsigned char *)&iq3xxs_grid_dev[qs[8*ib32 + 2*l + 1]];\n"
"        int o = b * 256 + ib32 * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            orow[o + j]     = f32_to_bf16(db * (float)grid1[j] * ((signs & (1 << j)) ? -1.0f : 1.0f));\n"
"            orow[o + 4 + j] = f32_to_bf16(db * (float)grid2[j] * ((signs & (1 << (j+4))) ? -1.0f : 1.0f));\n"
"        }\n"
"    }\n"
"}\n"

"__global__ void dequant_iq2s_all_int8(char *dst, const unsigned char *base,\n"
"        const int *offs, int rows, int cols, long long stride) {\n"
"    int e = blockIdx.y;\n"
"    if (offs[e + 1] == offs[e]) return;\n"
"    int row = blockIdx.x * 8 + threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    if (row >= rows) return;\n"
"    const unsigned char *mat = base + (long long)e * stride;\n"
"    int nb = cols / 256;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * nb * 82;\n"
"    char *orow = dst + ((size_t)e * rows + row) * cols;\n"
"    for (int g = lane; g < nb * 32; g += 32) {\n"
"        int b = g >> 5; int gl = g & 31; int ib32 = gl >> 2; int l = gl & 3;\n"
"        const unsigned char *bp = row_ptr + b * 82;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char scale = bp[74 + ib32];\n"
"        float db = (l < 2) ? d * (0.5f + (float)(scale & 0xf)) * 0.25f\n"
"                           : d * (0.5f + (float)(scale >>  4)) * 0.25f;\n"
"        int gi = bp[2 + ib32*4 + l] | ((bp[66 + ib32] << (8 - 2*l)) & 0x300);\n"
"        const unsigned char *grid = (const unsigned char *)&iq2s_grid_dev[gi];\n"
"        unsigned char s = bp[34 + ib32*4 + l];\n"
"        int o = b * 256 + ib32 * 32 + l * 8;\n"
"        for (int j = 0; j < 8; j++) {\n"
"            float val = db * (float)grid[j] * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"            orow[o + j] = (char)(int)rintf(val);\n"
"        }\n"
"    }\n"
"}\n"
"__global__ void dequant_iq3s_all_int8(char *dst, const unsigned char *base,\n"
"        const int *offs, int rows, int cols, long long stride) {\n"
"    int e = blockIdx.y;\n"
"    if (offs[e + 1] == offs[e]) return;\n"
"    int row = blockIdx.x * 8 + threadIdx.x / 32; int lane = threadIdx.x % 32;\n"
"    if (row >= rows) return;\n"
"    const unsigned char *mat = base + (long long)e * stride;\n"
"    int nb = cols / 256;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * nb * 110;\n"
"    char *orow = dst + ((size_t)e * rows + row) * cols;\n"
"    for (int g = lane; g < nb * 32; g += 32) {\n"
"        int b = g >> 5; int g32 = g & 31; int sb = g32 >> 2; int l = g32 & 3;\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned char sc = bp[106 + (sb >> 1)];\n"
"        float db = d * (float)(1 + 2 * ((sb & 1) ? (sc >> 4) : (sc & 0xf)));\n"
"        unsigned char qh = bp[66 + sb];\n"
"        const unsigned char *g1 = (const unsigned char *)&iq3s_grid_dev[bp[2+sb*8+2*l]   | ((qh << (8-2*l)) & 256)];\n"
"        const unsigned char *g2 = (const unsigned char *)&iq3s_grid_dev[bp[2+sb*8+2*l+1] | ((qh << (7-2*l)) & 256)];\n"
"        unsigned char s = bp[74 + sb*4 + l];\n"
"        int o = b * 256 + sb * 32 + l * 8;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            float v0 = db * (float)g1[j] * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"            float v1 = db * (float)g2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"            orow[o + j]     = (char)(int)rintf(v0);\n"
"            orow[o + 4 + j] = (char)(int)rintf(v1);\n"
"        }\n"
"    }\n"
"}\n""/* Batch top-K + softmax: one block per token. */\n"
"__global__ void moe_topk_batch(const float *logits /*[M,ne]*/, int ne, int K,\n"
"                                 int *tok_idx /*[M,K]*/, float *tok_w /*[M,K]*/) {\n"
"    int m = blockIdx.x; int tid = threadIdx.x;\n"
"    const float *lg = logits + (size_t)m * ne;\n"
"    __shared__ float sval[256];\n"
"    __shared__ int   sidx[256];\n"
"    __shared__ int   chosen[64];\n"
"    float my = (tid < ne) ? lg[tid] : -1e30f;\n"
"    int *oi = tok_idx + (size_t)m * K; float *ow = tok_w + (size_t)m * K;\n"
"    for (int ki = 0; ki < K; ki++) {\n"
"        float v = my;\n"
"        for (int j = 0; j < ki; j++) if (chosen[j] == tid) v = -1e30f;\n"
"        sval[tid] = v; sidx[tid] = tid; __syncthreads();\n"
"        for (int s = 128; s > 0; s >>= 1) {\n"
"            if (tid < s && tid + s < 256) {\n"
"                if (sval[tid+s] > sval[tid]) { sval[tid]=sval[tid+s]; sidx[tid]=sidx[tid+s]; }\n"
"                else if (sval[tid+s] == sval[tid] && sidx[tid+s] < sidx[tid]) sidx[tid]=sidx[tid+s];\n"
"            } __syncthreads();\n"
"        }\n"
"        if (tid == 0) { chosen[ki]=sidx[0]; oi[ki]=sidx[0]; ow[ki]=sval[0]; }\n"
"        __syncthreads();\n"
"    }\n"
"    if (tid == 0) {\n"
"        float mx = ow[0]; for (int i=1;i<K;i++) if (ow[i]>mx) mx=ow[i];\n"
"        float sum=0; for (int i=0;i<K;i++){ ow[i]=expf(ow[i]-mx); sum+=ow[i]; }\n"
"        float inv=1.0f/sum; for (int i=0;i<K;i++) ow[i]*=inv;\n"
"    }\n"
"}\n"
"/* Count per-expert assignments + exclusive prefix into offs (1 block, ne<=1024). */\n"
"__global__ void moe_count_offs(const int *tok_idx, int M, int K, int ne,\n"
"                                 int *offs /*[ne+1]*/, int *cursor /*[ne]*/) {\n"
"    __shared__ int cnt[1024];\n"
"    int tid = threadIdx.x;\n"
"    for (int e = tid; e < ne; e += blockDim.x) cnt[e] = 0;\n"
"    __syncthreads();\n"
"    for (int t = tid; t < M * K; t += blockDim.x) atomicAdd(&cnt[tok_idx[t]], 1);\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        int acc = 0;\n"
"        for (int e = 0; e < ne; e++) { offs[e] = acc; cursor[e] = acc; acc += cnt[e]; }\n"
"        offs[ne] = acc;\n"
"    }\n"
"}\n"
"/* Scatter assignments into expert-grouped order via atomic cursor. */\n"
"__global__ void moe_fill_gather(const int *tok_idx, const float *tok_w, int M, int K,\n"
"                                  int *cursor, int *gsrc, float *gw) {\n"
"    int t = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (t >= M * K) return;\n"
"    int e = tok_idx[t];\n"
"    int p = atomicAdd(&cursor[e], 1);\n"
"    gsrc[p] = t / K;\n"
"    gw[p]   = tok_w[t];\n"
"}\n"
"/* Grouped GEMM: per expert e, Y[offs[e]..offs[e+1], 0..N) = X[offs[e].., K] x W_e^T. */\n"
"__global__ void gemm_bf16_grouped(float *Y, const bf16_raw *Wall, const bf16_raw *X,\n"
"                                  const int *offs, int N, int K) {\n"
"    int e = blockIdx.z;\n"
"    int m0 = offs[e], m1 = offs[e + 1];\n"
"    int M = m1 - m0;\n"
"    if (M <= 0 || (int)blockIdx.y * 128 >= M) return;\n"
"    const bf16_raw *W = Wall + (size_t)e * N * K;\n"
"    const bf16_raw *Xe = X + (size_t)m0 * K;\n"
"    float *Ye = Y + (size_t)m0 * N;\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ unsigned short smA[128*32];\n"
"    __shared__ unsigned short smB[128*32];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    int interior = (cta_m0 + 128 <= M) && (cta_n0 + 128 <= N) && ((K & 31) == 0);\n"
"    for (int k = 0; k < K; k += 32) {\n"
"        if (interior) {\n"
"            int er = tid >> 1, ek = (tid & 1) * 16;\n"
"            bf16x8 *da = (bf16x8 *)&smA[er * 32 + ek];\n"
"            const bf16x8 *sa = (const bf16x8 *)&Xe[(size_t)(cta_m0 + er) * K + k + ek];\n"
"            da[0] = sa[0]; da[1] = sa[1];\n"
"            bf16x8 *db = (bf16x8 *)&smB[er * 32 + ek];\n"
"            const bf16x8 *sb = (const bf16x8 *)&W[(size_t)(cta_n0 + er) * K + k + ek];\n"
"            db[0] = sb[0]; db[1] = sb[1];\n"
"        } else {\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int el = tid * 16 + it;\n"
"            int er = el >> 5, ek = el & 31;\n"
"            int row = cta_m0 + er, kp = k + ek;\n"
"            smA[el] = (row < M && kp < K) ? Xe[(size_t)row * K + kp] : 0;\n"
"        }\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int el = tid * 16 + it;\n"
"            int er = el >> 5, ek = el & 31;\n"
"            int col = cta_n0 + er, kp = k + ek;\n"
"            smB[el] = (col < N && kp < K) ? W[(size_t)col * K + kp] : 0;\n"
"        }\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            bf16x8 a0,a1,a2,a3,b0,b1;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                a0[i]=smA[(a_base+0 +idx)*32+kk0+k_off+i];\n"
"                a1[i]=smA[(a_base+16+idx)*32+kk0+k_off+i];\n"
"                a2[i]=smA[(a_base+32+idx)*32+kk0+k_off+i];\n"
"                a3[i]=smA[(a_base+48+idx)*32+kk0+k_off+i];\n"
"                b0[i]=smB[(b_base+0 +idx)*32+kk0+k_off+i];\n"
"                b1[i]=smB[(b_base+16+idx)*32+kk0+k_off+i];\n"
"            }\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        if (col >= N) continue;\n"
"        float8 acc = *accs[t];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row >= M) continue;\n"
"            Ye[(size_t)row * N + col] = acc[i];\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__device__ __forceinline__ half_raw f32_to_f16_bits(float v) {\n"
"    __half hv = __float2half(v);\n"
"    return *((half_raw*)&hv);\n"
"}\n"
"\n"

"/* INT8 grouped GEMM: Y[t][n] = W_int8[n][:] * X_int8[t][:] via INT8 WMMA.     */\n"
"/* Grid: (N/128, mtiles, ne) x 256 thr. Same layout as gemm_bf16_grouped but INT8. */\n"
"__global__ void gemm_int8_grouped(float *Y, const signed char *Wall, const signed char *X,\n"
"                                  const int *offs, int N, int K) {\n"
"    typedef int i32x2 __attribute__((ext_vector_type(2)));\n"
"    typedef int i32x8 __attribute__((ext_vector_type(8)));\n"
"    int e = blockIdx.z;\n"
"    int m0 = offs[e], m1 = offs[e + 1];\n"
"    int M = m1 - m0;\n"
"    if (M <= 0 || (int)blockIdx.y * 128 >= M) return;\n"
"    const signed char *W = Wall + (size_t)e * N * K;\n"
"    const signed char *Xe = X + (size_t)m0 * K;\n"
"    float *Ye = Y + (size_t)m0 * N;\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int idx_lane = lane & 15;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ signed char smA[128*32];\n"
"    __shared__ signed char smB[128*32];\n"
"    i32x8 z = {0,0,0,0,0,0,0,0};\n"
"    i32x8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    int interior = (cta_m0 + 128 <= M) && (cta_n0 + 128 <= N) && ((K & 31) == 0);\n"
"    for (int k = 0; k < K; k += 32) {\n"
"        if (interior) {\n"
"            int er = tid >> 1, ek = (tid & 1) * 16;\n"
"            // Load smA: 16 INT8 per thread (2 chunks of 8)\n"
"            for (int it = 0; it < 2; it++) {\n"
"                int off = it * 8;\n"
"                int a0 = Xe[(size_t)(cta_m0 + er) * K + k + ek + off];\n"
"                smA[er * 32 + ek + off] = a0;\n"
"            }\n"
"            for (int it = 0; it < 2; it++) {\n"
"                int off = it * 8;\n"
"                int b0 = W[(size_t)(cta_n0 + er) * K + k + ek + off];\n"
"                smB[er * 32 + ek + off] = b0;\n"
"            }\n"
"        } else {\n"
"            for (int it = 0; it < 16; it++) {\n"
"                int el = tid * 16 + it;\n"
"                int er = el >> 5, ek = el & 31;\n"
"                int row = cta_m0 + er, kp = k + ek;\n"
"                smA[el] = (row < M && kp < K) ? Xe[(size_t)row * K + kp] : 0;\n"
"            }\n"
"            for (int it = 0; it < 16; it++) {\n"
"                int el = tid * 16 + it;\n"
"                int er = el >> 5, ek = el & 31;\n"
"                int col = cta_n0 + er, kp = k + ek;\n"
"                smB[el] = (col < N && kp < K) ? W[(size_t)col * K + kp] : 0;\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            // Load 8 INT8 values from smA row (a_base+0+idx_lane) at col kk0..kk0+7\n"
"            int addr_a0 = (a_base + 0 + idx_lane) * 32 + kk0;\n"
"            i32x2 a0 = { *(const int*)&smA[addr_a0], *(const int*)&smA[addr_a0 + 4] };\n"
"            int addr_a1 = (a_base + 16 + idx_lane) * 32 + kk0;\n"
"            i32x2 a1 = { *(const int*)&smA[addr_a1], *(const int*)&smA[addr_a1 + 4] };\n"
"            int addr_a2 = (a_base + 32 + idx_lane) * 32 + kk0;\n"
"            i32x2 a2 = { *(const int*)&smA[addr_a2], *(const int*)&smA[addr_a2 + 4] };\n"
"            int addr_a3 = (a_base + 48 + idx_lane) * 32 + kk0;\n"
"            i32x2 a3 = { *(const int*)&smA[addr_a3], *(const int*)&smA[addr_a3 + 4] };\n"
"            int addr_b0 = (b_base + 0 + idx_lane) * 32 + kk0;\n"
"            i32x2 b0 = { *(const int*)&smB[addr_b0], *(const int*)&smB[addr_b0 + 4] };\n"
"            int addr_b1 = (b_base + 16 + idx_lane) * 32 + kk0;\n"
"            i32x2 b1 = { *(const int*)&smB[addr_b1], *(const int*)&smB[addr_b1 + 4] };\n"
"            cv00=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a0,true,b0,cv00,false);\n"
"            cv01=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a0,true,b1,cv01,false);\n"
"            cv10=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a1,true,b0,cv10,false);\n"
"            cv11=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a1,true,b1,cv11,false);\n"
"            cv20=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a2,true,b0,cv20,false);\n"
"            cv21=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a2,true,b1,cv21,false);\n"
"            cv30=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a3,true,b0,cv30,false);\n"
"            cv31=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,a3,true,b1,cv31,false);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    i32x8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    // Convert INT32 accumulators to F32 and write output\n"
"    int half = idx_lane >> 3;\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx_lane;\n"
"        if (col >= N) continue;\n"
"        i32x8 acc = *accs[t];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row >= M) continue;\n"
"            Ye[(size_t)row * N + col] = (float)acc[i];\n"
"        }\n"
"    }\n"
"}\n"

"/* Quantize F32 to INT8 (range-limited, per-element). */\n"
"__global__ void quantize_f32_act_to_int8(signed char *dst, const float *src, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        float v = src[i];\n"
"        int q = (int)rintf(v);\n"
"        q = q > 127 ? 127 : (q < -127 ? -127 : q);\n"
"        dst[i] = (signed char)q;\n"
"    }\n"
"}\n"
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
"                                     int n_rope_pairs, int row_stride,\n"
"                                     const float *freq_factors) {\n"
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
"    if (freq_factors) freq /= freq_factors[j];\n"
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
"/* Variant with separate source/dest strides */\n"
"__global__ void kv_cache_store_batch_strided(float *key_cache, float *value_cache,\n"
"    const float *k_batch, const float *v_batch,\n"
"    int position_start, int M, int kv_dim, int batch_stride, int cache_len) {\n"
"    int total = M * kv_dim;\n"
"    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (gid >= total) return;\n"
"    int row = gid / kv_dim;\n"
"    int i   = gid - row * kv_dim;\n"
"    int pos = position_start + row;\n"
"    int slot = (cache_len > 0) ? (pos % cache_len) : pos;\n"
"    size_t cache_idx = (size_t)slot * kv_dim + i;\n"
"    size_t batch_idx = (size_t)row * batch_stride + i;\n"
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
"/* ---- head_dim=512 WMMA flash for Gemma4 full-attention (4 waves, head_dim split).\n"
" * head_dim 512 makes a single-wave O accumulator 256 VGPR (or 32 KB LDS), which\n"
" * collapses occupancy. Instead split the head_dim across 4 waves: each wave owns\n"
" * 128 output dims, so its O lives in registers (8 f32x8 = 64 VGPR). For QK each\n"
" * wave contracts only its 128 dims into a PARTIAL score; the 4 partials are summed\n"
" * via LDS to the full score, which all waves then softmax. Reads the F32 KV cache\n"
" * directly (K = RoPE'd key, V = un-RoPE'd normed key). LDS ~36.5 KB. BQ=16.\n"
" * Grid: (n_heads, ceil(M/16)). Block: 128. */\n"
"#define FW512_BQ 16\n"
"#define FW512_BKV 16\n"
"__global__ void flash_attn_wmma_hd512_causal(\n"
"    float *out, const float *Q, const float *key_cache, const float *value_cache,\n"
"    int M, int kv_len, int position_start, int n_heads, int n_kv_heads,\n"
"    int head_dim, int kv_dim, float scale) {\n"
"    int h  = blockIdx.x;\n"
"    int qb = blockIdx.y;\n"
"    int q0 = qb * FW512_BQ;\n"
"    int gqa = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa;\n"
"    int q_dim = n_heads * head_dim;\n"
"    int tid = threadIdx.x;\n"
"    int wid = tid >> 5;       /* 0..3, which 128-dim head slice this wave owns */\n"
"    int lid = tid & 31;\n"
"    int half = lid >> 4;\n"
"    int idx  = lid & 15;\n"
"    int dbase = wid * 128;    /* this wave's head_dim offset */\n"
"    extern __shared__ _Float16 smem512[];\n"
"    _Float16 *smK = smem512;                  /* [16*512] f16 (key*512 + d) */\n"
"    _Float16 *smV = smK + 16 * 512;           /* [16*512] f16 (d*16 + key)  */\n"
"    float    *smScore = (float*)(smV + 16 * 512); /* [4*256] partial scores  */\n"
"    _Float16 *smP = smK;                       /* [16*16] overlaps smK after QK */\n"
"    /* Q for this wave's 8 k-chunks (128 dims), scaled. */\n"
"    f16x8 q_reg[8];\n"
"    for (int kb = 0; kb < 8; kb++)\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int d = dbase + kb * 16 + half * 8 + i;\n"
"            int row = q0 + idx;\n"
"            float v = (row < M && d < head_dim) ? Q[(size_t)row * q_dim + h * head_dim + d] : 0.0f;\n"
"            q_reg[kb][i] = (_Float16)(v * scale);\n"
"        }\n"
"    float8 O_acc[8];\n"
"    for (int kb = 0; kb < 8; kb++) for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"    int max_kv = position_start + q0 + FW512_BQ;\n"
"    if (max_kv > kv_len) max_kv = kv_len;\n"
"    int n_kv = (max_kv + FW512_BKV - 1) / FW512_BKV;\n"
"    for (int t = 0; t < n_kv; t++) {\n"
"        int kv0 = t * FW512_BKV;\n"
"        for (int e = tid; e < 16 * 512; e += 128) {\n"
"            int key = e >> 9; int d = e & 511;\n"
"            int kpos = kv0 + key;\n"
"            float kf = 0.0f, vf = 0.0f;\n"
"            if (kpos < max_kv) {\n"
"                size_t base = (size_t)kpos * kv_dim + kv_h * head_dim + d;\n"
"                kf = key_cache[base]; vf = value_cache[base];\n"
"            }\n"
"            smK[key * 512 + d] = (_Float16)kf;\n"
"            smV[d * 16 + key]  = (_Float16)vf;\n"
"        }\n"
"        __syncthreads();\n"
"        /* Partial QK over this wave's 128 dims. */\n"
"        float8 sp = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < 8; kb++) {\n"
"            f16x8 bK;\n"
"            for (int i = 0; i < 8; i++) { int dpos = dbase + kb * 16 + half * 8 + i; bK[i] = smK[idx * 512 + dpos]; }\n"
"            sp = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_reg[kb], bK, sp);\n"
"        }\n"
"        /* Reduce the 4 waves' partial scores to the full score via LDS. */\n"
"        for (int i = 0; i < 8; i++) smScore[wid * 256 + lid * 8 + i] = sp[i];\n"
"        __syncthreads();\n"
"        float8 score;\n"
"        for (int i = 0; i < 8; i++)\n"
"            score[i] = smScore[0 * 256 + lid * 8 + i] + smScore[1 * 256 + lid * 8 + i]\n"
"                     + smScore[2 * 256 + lid * 8 + i] + smScore[3 * 256 + lid * 8 + i];\n"
"        bool col_valid = (kv0 + idx) < max_kv;\n"
"        for (int i = 0; i < 8; i++) score[i] = col_valid ? score[i] : -1e30f;\n"
"        int k_global = kv0 + idx;\n"
"        for (int i = 0; i < 8; i++) { int qg = position_start + q0 + half * 8 + i; if (qg < k_global) score[i] = -1e30f; }\n"
"        float row_max[8];\n"
"        for (int i = 0; i < 8; i++) { float v = score[i];\n"
"            v = fmaxf(v, __shfl_xor(v,1,32)); v = fmaxf(v, __shfl_xor(v,2,32));\n"
"            v = fmaxf(v, __shfl_xor(v,4,32)); v = fmaxf(v, __shfl_xor(v,8,32)); row_max[i] = v; }\n"
"        float alpha[8];\n"
"        for (int i = 0; i < 8; i++) { float nm = fmaxf(m_i[i], row_max[i]);\n"
"            alpha[i] = (m_i[i] < -1e29f) ? 0.0f : __expf(m_i[i] - nm);\n"
"            float ej = __expf(score[i] - nm); float s = ej;\n"
"            s += __shfl_xor(s,1,32); s += __shfl_xor(s,2,32); s += __shfl_xor(s,4,32); s += __shfl_xor(s,8,32);\n"
"            l_i[i] = l_i[i] * alpha[i] + s; m_i[i] = nm; score[i] = ej; }\n"
"        for (int kb = 0; kb < 8; kb++) for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"        __syncthreads();   /* before overwriting smK with smP */\n"
"        for (int i = 0; i < 8; i++) smP[(half * 8 + i) * 16 + idx] = (_Float16)score[i];\n"
"        __syncthreads();\n"
"        f16x8 ap;\n"
"        for (int i = 0; i < 8; i++) ap[i] = smP[idx * 16 + half * 8 + i];\n"
"        for (int kb = 0; kb < 8; kb++) {\n"
"            f16x8 bv;\n"
"            for (int i = 0; i < 8; i++) { int d_col = dbase + kb * 16 + idx; int kv_k = half * 8 + i; bv[i] = smV[d_col * 16 + kv_k]; }\n"
"            O_acc[kb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ap, bv, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int kb = 0; kb < 8; kb++)\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i; int d = dbase + kb * 16 + idx; int row = q0 + m_row;\n"
"            if (row < M && d < head_dim) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * q_dim + h * head_dim + d] = v;\n"
"            }\n"
"        }\n"
"}\n"
"#undef FW512_BQ\n"
"#undef FW512_BKV\n"
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
"                                    float freq_base, int n_rope_pairs,\n"
"                                    const float *freq_factors) {\n"
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
"    if (freq_factors) freq /= freq_factors[j];\n"
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
"                                     const int *pos_p, int kv_dim, int cache_len) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < kv_dim) {\n"
"        int position = *pos_p;\n"
"        int slot = (cache_len > 0) ? (position % cache_len) : position;\n"
"        key_cache[(size_t)slot * kv_dim + i] = k[i];\n"
"        value_cache[(size_t)slot * kv_dim + i] = v[i];\n"
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
"/* ---- Gemma4 kernels ---- */\n"
"\n"
"/* GELU * up: gate[i] = gelu(gate[i]) * up[i] */\n"
"__global__ void gelu_mul_f32(float *gate, const float *up, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float g = gate[i];\n"
"    float u = up[i];\n"
"    float gelu_g = g * 0.5f * (1.0f + erff(g * 0.7071067811865476f));\n"
"    gate[i] = gelu_g * u;\n"
"}\n"
"\n"
"/* Logit softcapping: x[i] = cap * tanh(x[i] / cap) */\n"
"__global__ void logit_softcap_f32(float *x, int n, float cap) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    x[i] = cap * tanhf(x[i] / cap);\n"
"}\n"
"\n"
"/* Scale: x[i] *= scale */\n"
"__global__ void scale_f32(float *x, float scale, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    x[i] *= scale;\n"
"}\n"
"\n"
"\n"
"/* Batched weightless per-head RMS norm. grid=(n_heads, M), block 256, smem 256 floats.\n"
" * Handles head_dim > blockDim (e.g. 512) via a grid-stride loop, unlike the single\n"
" * tid<head_dim variant above. Used for Gemma4 V normalization (V is rms-normed per head). */\n"
"__global__ void raw_rmsnorm_heads_batch_f32(float *vec_batch, int n_heads, int head_dim,\n"
"                                            int M, int row_stride, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int h = blockIdx.x;\n"
"    int row = blockIdx.y;\n"
"    if (h >= n_heads || row >= M) return;\n"
"    int tid = threadIdx.x;\n"
"    int NT = blockDim.x;\n"
"    float *v = vec_batch + (size_t)row * row_stride + h * head_dim;\n"
"    float sum = 0.0f;\n"
"    for (int d = tid; d < head_dim; d += NT) { float x = v[d]; sum += x * x; }\n"
"    sdata[tid] = sum; __syncthreads();\n"
"    for (int s = NT / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }\n"
"    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);\n"
"    for (int d = tid; d < head_dim; d += NT) v[d] *= scale;\n"
"}\n"
"\n"
"\n"
"\n"
"/* Flash (online-softmax) prefill attention. One block per (head, query row).\n"
" * Bounded shared memory = (2*head_dim + 2*blockDim) floats regardless of seq_len,\n"
" * so it scales to arbitrarily long sequences (pp32k) without exceeding LDS.\n"
" *   window > 0 : sliding-window attention, keys restricted to [pos-window+1, pos].\n"
" *   window == 0: full causal attention, keys [0, pos].\n"
" *   cache_len  : circular cache size (slot = pos %% cache_len). For full-attn this\n"
" *                equals max_seq_len so the modulo is a no-op. */\n"
"__global__ void attn_prefill_flash_f32(float *out, const float *q,\n"
"    const float *key_cache, const float *value_cache,\n"
"    int n_heads, int n_kv_heads, int head_dim, int kv_dim,\n"
"    int M, int position_start, float scale, int window, int cache_len) {\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    int m = blockIdx.y;\n"
"    if (h >= n_heads || m >= M) return;\n"
"    int kv_h = h / (n_heads / n_kv_heads);\n"
"    int pos = position_start + m;\n"
"    int lo = (window > 0 && (pos - window + 1) > 0) ? (pos - window + 1) : 0;\n"
"    int hi = pos;\n"
"    int tid = threadIdx.x;\n"
"    int NT  = blockDim.x;\n"
"    float *q_sh   = smem;                  /* head_dim */\n"
"    float *acc_sh = q_sh + head_dim;       /* head_dim */\n"
"    float *p_sh   = acc_sh + head_dim;     /* NT       */\n"
"    float *red    = p_sh + NT;             /* NT       */\n"
"    const float *qh = q + (size_t)m * n_heads * head_dim + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += NT) { q_sh[d] = qh[d] * scale; acc_sh[d] = 0.0f; }\n"
"    __syncthreads();\n"
"    float m_i = -1e30f, l_i = 0.0f;\n"
"    for (int ts = lo; ts <= hi; ts += NT) {\n"
"        int kpos = ts + tid;\n"
"        int tile_n = hi - ts + 1; if (tile_n > NT) tile_n = NT;\n"
"        float score = -1e30f;\n"
"        if (kpos <= hi) {\n"
"            int slot = (cache_len > 0) ? (kpos % cache_len) : kpos;\n"
"            const float *kp = key_cache + (size_t)slot * kv_dim + kv_h * head_dim;\n"
"            float s = 0.0f;\n"
"            for (int d = 0; d < head_dim; d++) s += q_sh[d] * kp[d];\n"
"            score = s;\n"
"        }\n"
"        red[tid] = score; __syncthreads();\n"
"        for (int s2 = NT / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] = fmaxf(red[tid], red[tid + s2]); __syncthreads(); }\n"
"        float tile_max = red[0];\n"
"        float new_m = fmaxf(m_i, tile_max);\n"
"        float corr  = (m_i < -1e29f) ? 0.0f : __expf(m_i - new_m);\n"
"        float pv = (kpos <= hi) ? __expf(score - new_m) : 0.0f;\n"
"        __syncthreads();\n"
"        p_sh[tid] = pv; red[tid] = pv; __syncthreads();\n"
"        for (int s2 = NT / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }\n"
"        float tile_sum = red[0];\n"
"        l_i = l_i * corr + tile_sum;\n"
"        for (int d = tid; d < head_dim; d += NT) {\n"
"            float a = acc_sh[d] * corr;\n"
"            for (int t = 0; t < tile_n; t++) {\n"
"                int slot2 = (cache_len > 0) ? ((ts + t) % cache_len) : (ts + t);\n"
"                a += p_sh[t] * value_cache[(size_t)slot2 * kv_dim + kv_h * head_dim + d];\n"
"            }\n"
"            acc_sh[d] = a;\n"
"        }\n"
"        m_i = new_m;\n"
"        __syncthreads();\n"
"    }\n"
"    float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;\n"
"    float *out_h = out + (size_t)m * n_heads * head_dim + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += NT) out_h[d] = acc_sh[d] * inv;\n"
"}\n"
"\n"
"/* Flash (online-softmax) decode attention (single query at *pos_p). Device-pointer\n"
" * position so it is HIP-graph-capturable. Bounded shared memory. window/cache_len\n"
" * semantics identical to attn_prefill_flash_f32. */\n"
"__global__ void attn_decode_flash_f32_devp(float *out, const float *q,\n"
"    const float *key_cache, const float *value_cache,\n"
"    int n_heads, int n_kv_heads, int head_dim, int kv_dim,\n"
"    const int *pos_p, float scale, int window, int cache_len) {\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int kv_h = h / (n_heads / n_kv_heads);\n"
"    int pos = *pos_p;\n"
"    int lo = (window > 0 && (pos - window + 1) > 0) ? (pos - window + 1) : 0;\n"
"    int hi = pos;\n"
"    int tid = threadIdx.x;\n"
"    int NT  = blockDim.x;\n"
"    float *q_sh   = smem;\n"
"    float *acc_sh = q_sh + head_dim;\n"
"    float *p_sh   = acc_sh + head_dim;\n"
"    float *red    = p_sh + NT;\n"
"    const float *qh = q + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += NT) { q_sh[d] = qh[d] * scale; acc_sh[d] = 0.0f; }\n"
"    __syncthreads();\n"
"    float m_i = -1e30f, l_i = 0.0f;\n"
"    for (int ts = lo; ts <= hi; ts += NT) {\n"
"        int kpos = ts + tid;\n"
"        int tile_n = hi - ts + 1; if (tile_n > NT) tile_n = NT;\n"
"        float score = -1e30f;\n"
"        if (kpos <= hi) {\n"
"            int slot = (cache_len > 0) ? (kpos % cache_len) : kpos;\n"
"            const float *kp = key_cache + (size_t)slot * kv_dim + kv_h * head_dim;\n"
"            float s = 0.0f;\n"
"            for (int d = 0; d < head_dim; d++) s += q_sh[d] * kp[d];\n"
"            score = s;\n"
"        }\n"
"        red[tid] = score; __syncthreads();\n"
"        for (int s2 = NT / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] = fmaxf(red[tid], red[tid + s2]); __syncthreads(); }\n"
"        float tile_max = red[0];\n"
"        float new_m = fmaxf(m_i, tile_max);\n"
"        float corr  = (m_i < -1e29f) ? 0.0f : __expf(m_i - new_m);\n"
"        float pv = (kpos <= hi) ? __expf(score - new_m) : 0.0f;\n"
"        __syncthreads();\n"
"        p_sh[tid] = pv; red[tid] = pv; __syncthreads();\n"
"        for (int s2 = NT / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }\n"
"        float tile_sum = red[0];\n"
"        l_i = l_i * corr + tile_sum;\n"
"        for (int d = tid; d < head_dim; d += NT) {\n"
"            float a = acc_sh[d] * corr;\n"
"            for (int t = 0; t < tile_n; t++) {\n"
"                int slot2 = (cache_len > 0) ? ((ts + t) % cache_len) : (ts + t);\n"
"                a += p_sh[t] * value_cache[(size_t)slot2 * kv_dim + kv_h * head_dim + d];\n"
"            }\n"
"            acc_sh[d] = a;\n"
"        }\n"
"        m_i = new_m;\n"
"        __syncthreads();\n"
"    }\n"
"    float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;\n"
"    float *out_h = out + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += NT) out_h[d] = acc_sh[d] * inv;\n"
"}\n"
"\n"
"/* Repack batch buffer from compact per-layer stride to padded max stride.\n"
" * dst [M * dst_stride] = src [M * src_stride], copying n_per_row elements per row. */\n"
"__global__ void repack_batch_f32(float *dst, const float *src,\n"
"    int M, int n_per_row, int src_stride, int dst_stride) {\n"
"    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = M * n_per_row;\n"
"    if (gid >= total) return;\n"
"    int row = gid / n_per_row;\n"
"    int col = gid - row * n_per_row;\n"
"    dst[(size_t)row * dst_stride + col] = src[(size_t)row * src_stride + col];\n"
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
    void *moe_gate_w_bf16;        /* bf16 router weights (fused decode) */
    void *moe_shared_gate_w_bf16;
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

    /* BF16 copies for hipBLASLt prefill GEMMs (Phase 2/3). NULL if not converted. */
    void *ssm_qkv_w_bf16;
    void *ssm_gate_w_bf16;
    void *ssm_alpha_w_bf16;
    void *ssm_beta_w_bf16;
    void *ssm_out_w_bf16;

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

    /* Gemma4 fields */
    int is_swa;                 /* 1 = sliding window attention layer */
    int shared_kv_source;       /* -1 = own KV, else layer index to share from */
    int local_kv_heads;         /* per-layer KV head count (may differ from r->n_kv_heads) */
    int local_head_dim;         /* head_dim_full or head_dim_swa */
    void *post_attn_norm_w;     /* F32 [n_embd] post-attention norm */
    void *post_ffw_norm_w;      /* F32 [n_embd] post-FFN norm */
    void *ple_inp_gate_w;       /* F32 [ple_dim, n_embd] PLE input gate */
    void *ple_proj_w;           /* F32 [n_embd, ple_dim] PLE projection */
    void *ple_post_norm_w;      /* F32 [n_embd] PLE post-projection norm */
    float layer_scale_val;      /* layer output scale (default 1.0) */
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
    hipFunction_t fn_quantize_f32_act_to_int8;
    hipFunction_t fn_matvec_q8_0_dp4a;
    hipFunction_t fn_matvec_q8_0_f32;
    hipFunction_t fn_embed_q8_0;
    hipFunction_t fn_matvec_q2_K_f32;
    hipFunction_t fn_matvec_q2_K_g4_f32;
    hipFunction_t fn_matvec_q3_K_f32;
    hipFunction_t fn_matvec_q3_K_g4_f32;
    int q2k_g4, q3k_g4;                     /* LLM_Q2K_G4 / LLM_Q3K_G4 warp-per-row */
    hipFunction_t fn_matvec_q4_K_f32;
    hipFunction_t fn_matvec_q4_K_mw_f32;
    hipFunction_t fn_matvec_q4_K_g4_f32;
    int q4k_g4;                             /* LLM_Q4K_G4: warp-per-row G=nb*4 Q4_K */
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
    /* MoE device-side dispatch (sync-free, graph-capturable) */
    hipFunction_t fn_moe_topk_softmax_gpu;
    hipFunction_t fn_sigmoid_scalar_f32;
    hipFunction_t fn_scale_add_dev_f32;
    hipFunction_t fn_moe_gather_rows;
    hipFunction_t fn_moe_scatter_accum;
    hipFunction_t fn_moe_row_scale_add;
    hipFunction_t fn_mmq_iq2s_f32;   /* fused quantized MoE GEMM (gate/up) */
    hipFunction_t fn_mmq_iq3s_f32;   /* fused quantized MoE GEMM (down) */
    hipFunction_t fn_moe_gateup_silu_iq2s;  /* decode: all-K experts gate+up+silu */
    hipFunction_t fn_moe_down_accum_iq3s;   /* decode: all-K experts down+w-accum */
    hipFunction_t fn_shexp_gateup_silu_q6k; /* decode: shared expert gate+up+silu */
    hipFunction_t fn_shexp_down_accum_q6k;  /* decode: shared expert down+scale-add */
    hipFunction_t fn_ssm_prep_f32;          /* decode: fused SSM aux chain */
    hipFunction_t fn_ssm_matvec4_q6k;       /* decode: fused 4 SSM input matvecs */
    hipFunction_t fn_matvec_qkv_q6k;        /* decode: fused attn q/k/v matvecs */
    hipFunction_t fn_ffn_gate_up_silu_q6k;  /* decode: fused FFN gate+up+SiLU (Q6_K) */
    hipFunction_t fn_ffn_gate_up_silu_iq3xxs; /* decode: fused FFN gate+up+SiLU (IQ3_XXS) */
    hipFunction_t fn_ffn_gate_up_silu_iq3xxs_mw; /* warp-per-row twin */
    hipFunction_t fn_matvec_down_residual_q6k; /* decode: fused down matvec + residual (Q6_K) */
    hipFunction_t fn_matvec_down_residual_iq3xxs; /* decode: fused down matvec + residual (IQ3_XXS) */
    hipFunction_t fn_matvec_down_residual_iq3xxs_mw; /* warp-per-row twin */
    hipFunction_t fn_matvec_down_residual_iq3xxs_splitk; /* split-K warp-per-row twin */
    int down_ksplit;                        /* LLM_DOWN_KSPLIT (default 1) */
    hipFunction_t fn_matvec_qkv_iq3xxs;       /* decode: fused attn Q/K/V IQ3_XXS matvecs */
    hipFunction_t fn_matvec_out_gated_iq3xxs; /* decode: fused sigmoid_mul + output matvec (IQ3_XXS) */
    hipFunction_t fn_ssm_matvec4_iq3xxs;      /* decode: fused 4 SSM input matvecs (IQ3_XXS) */
    hipFunction_t fn_fused_ssm_out_gated_iq3xxs; /* decode: fused SSM out + gated RMSNorm (IQ3_XXS) */
    hipFunction_t fn_matvec_out_gated_q6k;     /* decode: fused gated output matvec (Q6_K) */
    hipFunction_t fn_fused_ssm_out_gated_q6k;  /* decode: fused SSM out + gated RMSNorm (Q6_K) */
    hipFunction_t fn_fused_ssm_out_gated_q6k_mw; /* warp-per-row twin (precomputed inv_mean) */
    hipFunction_t fn_ssm_inv_mean_f32;         /* precompute per-head RMSNorm inv_mean */
    void *d_ssm_inv_mean;                       /* scratch: dt_rank floats */
    int ssm_out_mw;                             /* LLM_SSM_OUT_MW */
    hipFunction_t fn_moe_route_decode;      /* decode: router+topk+sgate fused */
    hipFunction_t fn_res_rmsnorm_f32;       /* decode: residual + rmsnorm fused */
    hipFunction_t fn_moe_router_fused;      /* decode: router+topk+sgate, 1 launch */
    hipFunction_t fn_deltanet_step_warp_f32; /* decode: warp-per-row deltanet */
    int moe_add_pending;                    /* decode: pending x += d_moe_accum fold */
    hipFunction_t fn_deltanet_step_batch_warp_f32; /* prefill: warp-per-row M-step */
    unsigned int *d_router_counter;
    int ssm_fused_decode;                   /* LLM_SSM_FUSED (default on) */
    int ffn_iq3_mw;                         /* LLM_FFN_IQ3_MW: warp-per-row FFN IQ3_XXS */
    int mw_threads;                         /* LLM_MW_THREADS: threads/block for warp-per-row (default 256) */
    hipFunction_t fn_gemm_bf16_own;         /* self-owned BF16 WMMA GEMM */
    hipFunction_t fn_gemm_bf16_own_db;      /* double-buffered LDS variant */
    int gemm_own;                           /* LLM_GEMM=own -> 1, blaslt -> 0 */
    hipFunction_t fn_dequant_iq2s_all;      /* all-expert dequant (grouped prefill) */
    hipFunction_t fn_dequant_iq3s_all;
    hipFunction_t fn_dequant_iq2_xxs_all;
    hipFunction_t fn_dequant_iq3_xxs_all;
    hipFunction_t fn_gemm_bf16_grouped;     /* per-expert grouped WMMA GEMM */
    hipFunction_t fn_gemm_int8_grouped;     /* INT8 WMMA grouped GEMM */
    hipFunction_t fn_dequant_iq2s_all_int8; /* dequant IQ2_S->INT8 (all experts) */
    hipFunction_t fn_dequant_iq3s_all_int8; /* dequant IQ3_S->INT8 (all experts) */
    void *d_expw_int8;                     /* [ne*N*K] int8 weight staging (one type) */
    void *d_act_int8;                      /* [max_assignments*max_dim] int8 activations */
    hipFunction_t fn_moe_topk_batch;
    hipFunction_t fn_moe_count_offs;
    hipFunction_t fn_moe_fill_gather;
    void *d_expw_bf16;                      /* [ne*rows*cols] bf16 staging (one type) */
    int  *d_moe_offs;                       /* [ne+1] device expert offsets */
    int  *d_tok_idx;                        /* [Mmax*K] per-token expert idx (GPU group) */
    float *d_tok_w;                         /* [Mmax*K] per-token softmax w */
    int  *d_cursor;                         /* [ne] scatter cursor */
    int moe_fused_decode;            /* LLM_MOE_FUSED (default on if types match) */
    int moe_iq2_bm;                  /* IQ2_XXS/IQ3_XXS weights repacked block-major */
    int moe_int8;                    /* use INT8 WMMA grouped GEMM */
    hipFunction_t fn_matvec_iq2_s_expert_f32;
    hipFunction_t fn_matvec_iq3_s_expert_f32;
    hipFunction_t fn_matvec_iq4_xs_expert_f32;
    /* DP4A decode path: per-32-block q8 activation + hardware int8 dot */
    hipFunction_t fn_quantize_q8_32;
    hipFunction_t fn_matvec_iq2_s_dp4a;
    hipFunction_t fn_matvec_iq3_s_dp4a;
    hipFunction_t fn_matvec_iq2_s_expert_dp4a;
    hipFunction_t fn_matvec_iq3_s_expert_dp4a;
    /* LDS-cached grid decode path */
    hipFunction_t fn_matvec_iq3_s_lds_f32;
    hipFunction_t fn_matvec_iq3_s_expert_lds_f32;
    hipFunction_t fn_matvec_iq2_s_lds_f32;
    hipFunction_t fn_matvec_iq2_s_expert_lds_f32;
    int lds_grid;        /* LLM_LDS_GRID: cache iq2s/iq3s grids in shared memory */
    void *d_act_q8;      /* int8 activation A (d_xb), per-32-block (<= 8192 cols) */
    void *d_act_scale;   /* per-32-block fp32 scale A */
    void *d_act_q8_b;    /* int8 activation B (expert d_gate for down-proj) */
    void *d_act_scale_b; /* per-32-block fp32 scale B */
    int   decode_dp4a;   /* LLM_DECODE_DP4A (default on) */
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
    hipFunction_t fn_dequant_q8_0_to_bf16;
    hipFunction_t fn_dequant_q2_K_to_bf16;
    hipFunction_t fn_dequant_q3_K_to_bf16;
    hipFunction_t fn_dequant_q4_K_to_bf16;
    hipFunction_t fn_dequant_q5_K_to_bf16;
    hipFunction_t fn_dequant_q6_K_to_bf16;
    hipFunction_t fn_dequant_iq3_xxs_to_bf16;
    hipFunction_t fn_dequant_iq4_xs_to_bf16;
    hipFunction_t fn_dequant_iq2_xs_to_bf16;
    hipFunction_t fn_dequant_iq2_s_to_bf16;
    hipFunction_t fn_dequant_iq3_s_to_bf16;
    hipFunction_t fn_dequant_iq1_s_to_bf16;
    hipFunction_t fn_dequant_tq1_0_to_bf16;
    hipFunction_t fn_dequant_iq2_xxs_to_bf16;

    /* Phase 3 flash-attention helpers */
    hipFunction_t fn_pack_f16_from_f32;
    hipFunction_t fn_pack_kv_cache_f16;
    hipFunction_t fn_rope_neox_batch_f32;
    hipFunction_t fn_rope_mrope_batch_f32;
    hipFunction_t fn_kv_cache_store_batch;
    hipFunction_t fn_kv_cache_store_batch_strided;
    hipFunction_t fn_flash_attn_wmma_f16_4w_causal;
    hipFunction_t fn_flash_attn_wmma_f16_4w_causal_hd256;

    /* Phase 4 WMMA decode matvec */
    hipFunction_t fn_matvec_f16_wmma_f32;

    /* Hidden-state snapshots (text-encoder path): capture up to 3 layers'
     * post-residual hidden state during forward into d_hidden_snapshots. */
    int   hidden_snapshot_layers[3];
    int   n_hidden_snapshots;
    void *d_hidden_snapshots;   /* device [3 * n_embd] f32 */

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
    /* MoE device-side dispatch buffers */
    int  *d_moe_idx;       /* [n_experts_used] selected expert indices */
    float *d_moe_w;        /* [n_experts_used] softmax weights */
    float *d_shared_scale; /* [1] shared-expert sigmoid gate */
    float *d_moe_act8;     /* [n_experts_used, expert_ff] fused decode activations */
    int   moe_dev_dispatch_ok; /* 1 if all expert types are device-dispatchable */
    /* Batched token-grouped MoE prefill (Phase 2) */
    int   moe_prefill_batched;   /* LLM_MOE_PREFILL=1 enables the batched-expert path */
    void *d_router_logits_batch; /* [Mmax, n_experts] f32 */
    void *d_router_w_bf16;       /* [n_experts, n_embd] bf16 router weight (packed) */
    void *d_moe_gather_in;       /* [Mmax*K, n_embd] f32 gathered activations */
    void *d_moe_gather_in_bf16;  /* bf16 */
    void *d_moe_eg;              /* [Mmax*K, expert_ff] f32 expert gate */
    void *d_moe_eu;              /* [Mmax*K, expert_ff] f32 expert up */
    void *d_moe_esilu_bf16;      /* [Mmax*K, expert_ff] bf16 silu(gate)*up */
    void *d_moe_eout;            /* [Mmax*K, n_embd] f32 expert outputs */
    void *d_moe_out_batch;       /* [Mmax, n_embd] f32 accumulated MoE output */
    void *d_xnorm_batch_bf16_moe;/* [Mmax, n_embd] bf16 normed input (router+shared) */
    void *d_shared_scale_batch;  /* [Mmax] f32 shared-gate sigmoid per token */
    int  *d_moe_gather_src;      /* [Mmax*K] token index per assignment */
    float *d_moe_gather_w;       /* [Mmax*K] softmax weight per assignment */
    float *h_router_batch;       /* host [Mmax*n_experts] */
    int   *h_moe_gather_src;     /* host [Mmax*K] expert-grouped token index */
    float *h_moe_gather_w;       /* host [Mmax*K] expert-grouped weight */
    int   *h_moe_tok_idx;        /* host [Mmax*K] per-token top-K expert idx */
    float *h_moe_tok_w;          /* host [Mmax*K] per-token top-K softmax weight */
    int   *h_moe_offsets;        /* host [n_experts+1] */

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

    /* Gemma4 architecture */
    int is_gemma4;
    int head_dim_full;              /* head dim for full-attention layers (512) */
    int head_dim_swa;               /* head dim for SWA layers (256) */
    int swa_window_size;            /* sliding window size (512) */
    int *swa_pattern;               /* [n_layers] 1=SWA, 0=full attention */
    int n_layer_kv_from_start;      /* layers 0..N-1 have own KV, rest share */
    float final_logit_softcapping;  /* tanh softcap on output logits (30.0) */
    float rope_freq_base_swa;       /* RoPE freq base for SWA layers (10000) */
    float embd_scale;               /* sqrt(n_embd) for token embedding scaling */
    int n_embd_per_layer;           /* per-layer embedding dim (256) */
    int *per_layer_kv_heads;        /* [n_layers] per-layer KV head count */
    void *d_ple_combined;           /* [n_layers * ple_dim] f32 */
    void *d_ple_buf;                /* [ple_dim] f32 */
    void *d_ple_proj;               /* [n_embd] f32 */
    void *d_per_layer_token_embd;   /* Q8_0 [n_vocab, n_layers * ple_dim] */
    void *d_per_layer_model_proj;   /* BF16 [n_layers * ple_dim, n_embd] */
    void *d_per_layer_proj_norm;    /* F32 [ple_dim] */
    float *h_rope_freq_factors;     /* [head_dim_full/2] proportional RoPE factors */
    void  *d_rope_freq_factors;     /* GPU copy of the above (full-attn proportional rope) */
    int    n_rope_freq_factors;     /* length of freq_factors (0 if absent) */
    float *h_rope_inv_freq_full;    /* precomputed inv_freq for full-attn layers */
    float *h_rope_inv_freq_swa;     /* precomputed inv_freq for SWA layers */
    int rope_full_dim;              /* head_dim_full/2 */
    int rope_swa_dim;               /* head_dim_swa/2 */
    /* Gemma4 GPU kernels */
    hipFunction_t fn_gelu_mul_f32;
    hipFunction_t fn_logit_softcap_f32;
    hipFunction_t fn_scale_f32;
    hipFunction_t fn_raw_rmsnorm_heads_batch_f32;
    hipFunction_t fn_attn_prefill_flash_f32;    /* bounded online-softmax prefill */
    hipFunction_t fn_attn_decode_flash_f32_devp;/* bounded online-softmax decode  */
    hipFunction_t fn_flash_attn_wmma_hd512_causal; /* WMMA full-attn (head_dim 512) */
    int           gemma_fa_wmma;                /* use WMMA kernel for full-attn layers */
    int         swa_cache_len;      /* circular SWA cache size = window + chunk */
    int         gemma_prefill_chunk;/* prefill chunk size (<= swa_window) */
    int         cur_position;       /* host-side current position for SWA kernel */
};

static inline int gemm_run_bf16_w(hip_llm_runner *r, void *Y, const void *W,
                                  const void *X, int M, int N, int K, void *stream);

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
    GET_FUNC(quantize_f32_act_to_int8);
    GET_FUNC(matvec_q8_0_f32);
    GET_FUNC(embed_q8_0);
    GET_FUNC(matvec_q2_K_f32);
    GET_FUNC(matvec_q2_K_g4_f32);
    GET_FUNC(matvec_q3_K_f32);
    GET_FUNC(matvec_q3_K_g4_f32);
    GET_FUNC(matvec_q4_K_f32);
    GET_FUNC(matvec_q4_K_mw_f32);
    GET_FUNC(matvec_q4_K_g4_f32);
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
    /* MoE device-side dispatch */
    GET_FUNC(moe_topk_softmax_gpu);
    GET_FUNC(sigmoid_scalar_f32);
    GET_FUNC(scale_add_dev_f32);
    GET_FUNC(moe_gather_rows);
    GET_FUNC(moe_scatter_accum);
    GET_FUNC(moe_row_scale_add);
    GET_FUNC(mmq_iq2s_f32);
    GET_FUNC(mmq_iq3s_f32);
    GET_FUNC(moe_gateup_silu_iq2s);
    GET_FUNC(moe_down_accum_iq3s);
    GET_FUNC(shexp_gateup_silu_q6k);
    GET_FUNC(shexp_down_accum_q6k);
    GET_FUNC(ssm_prep_f32);
    GET_FUNC(ssm_matvec4_q6k);
    GET_FUNC(matvec_qkv_q6k);
    GET_FUNC(ffn_gate_up_silu_q6k);
    GET_FUNC(ffn_gate_up_silu_iq3xxs);
    GET_FUNC(ffn_gate_up_silu_iq3xxs_mw);
    GET_FUNC(matvec_down_residual_q6k);
    GET_FUNC(matvec_down_residual_iq3xxs);
    GET_FUNC(matvec_down_residual_iq3xxs_mw);
    GET_FUNC(matvec_down_residual_iq3xxs_splitk);
    GET_FUNC(matvec_qkv_iq3xxs);
    GET_FUNC(matvec_out_gated_iq3xxs);
    GET_FUNC(ssm_matvec4_iq3xxs);
    GET_FUNC(fused_ssm_out_gated_iq3xxs);
    GET_FUNC(matvec_out_gated_q6k);
    GET_FUNC(fused_ssm_out_gated_q6k);
    GET_FUNC(fused_ssm_out_gated_q6k_mw);
    GET_FUNC(ssm_inv_mean_f32);
    GET_FUNC(moe_route_decode);
    GET_FUNC(res_rmsnorm_f32);
    GET_FUNC(moe_router_fused);
    GET_FUNC(deltanet_step_warp_f32);
    GET_FUNC(deltanet_step_batch_warp_f32);
    GET_FUNC(gemm_bf16_own);
    GET_FUNC(gemm_bf16_own_db);
    GET_FUNC(dequant_iq2s_all);
    GET_FUNC(dequant_iq3s_all);
    GET_FUNC(dequant_iq2_xxs_all);
    GET_FUNC(dequant_iq3_xxs_all);
    GET_FUNC(gemm_bf16_grouped);
    GET_FUNC(gemm_int8_grouped);
    GET_FUNC(dequant_iq2s_all_int8);
    GET_FUNC(dequant_iq3s_all_int8);
    GET_FUNC(moe_topk_batch);
    GET_FUNC(moe_count_offs);
    GET_FUNC(moe_fill_gather);
    GET_FUNC(matvec_iq2_s_expert_f32);
    GET_FUNC(matvec_iq3_s_expert_f32);
    GET_FUNC(matvec_iq4_xs_expert_f32);
    /* DP4A decode path */
    GET_FUNC(quantize_q8_32);
    GET_FUNC(matvec_iq2_s_dp4a);
    GET_FUNC(matvec_iq3_s_dp4a);
    GET_FUNC(matvec_iq2_s_expert_dp4a);
    GET_FUNC(matvec_iq3_s_expert_dp4a);
    GET_FUNC(matvec_iq3_s_lds_f32);
    GET_FUNC(matvec_iq3_s_expert_lds_f32);
    GET_FUNC(matvec_iq2_s_lds_f32);
    GET_FUNC(matvec_iq2_s_expert_lds_f32);
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
    /* Gemma4 kernels */
    GET_FUNC(gelu_mul_f32);
    GET_FUNC(logit_softcap_f32);
    GET_FUNC(scale_f32);
    GET_FUNC(raw_rmsnorm_heads_batch_f32);
    GET_FUNC(attn_prefill_flash_f32);
    GET_FUNC(attn_decode_flash_f32_devp);
    GET_FUNC(flash_attn_wmma_hd512_causal);

    GET_FUNC(pack_bf16_from_f32);
    GET_FUNC(convert_f16_to_bf16);
    GET_FUNC(dequant_q8_0_to_bf16);
    GET_FUNC(dequant_q2_K_to_bf16);
    GET_FUNC(dequant_q3_K_to_bf16);
    GET_FUNC(dequant_q4_K_to_bf16);
    GET_FUNC(dequant_q5_K_to_bf16);
    GET_FUNC(dequant_q6_K_to_bf16);
    GET_FUNC(dequant_iq3_xxs_to_bf16);
    GET_FUNC(dequant_iq4_xs_to_bf16);
    GET_FUNC(dequant_iq2_xs_to_bf16);
    GET_FUNC(dequant_iq2_s_to_bf16);
    GET_FUNC(dequant_iq3_s_to_bf16);
    GET_FUNC(dequant_iq1_s_to_bf16);
    GET_FUNC(dequant_tq1_0_to_bf16);
    GET_FUNC(dequant_iq2_xxs_to_bf16);

    GET_FUNC(pack_f16_from_f32);
    GET_FUNC(pack_kv_cache_f16);
    GET_FUNC(rope_neox_batch_f32);
    GET_FUNC(rope_mrope_batch_f32);
    GET_FUNC(kv_cache_store_batch);
    GET_FUNC(kv_cache_store_batch_strided);
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

    /* DP4A decode path (opt-in via LLM_DECODE_DP4A=1). Default OFF: on gfx1201
     * the IQ2_S/IQ3_S matvecs are grid-lookup-dequant bound, not multiply bound,
     * so int8 DP4A + q8 activation is ~2% slower than the full-utilization F32
     * kernels. Kept gated/validated (rel_l2 ~1e-3 vs float ref) for reference and
     * future tuning (e.g. SIMD sign via __vsub4, LDS-cached grids). */
    r->decode_dp4a = 0;
    { const char *e = getenv("LLM_DECODE_DP4A"); if (e) r->decode_dp4a = atoi(e) != 0; }
    r->lds_grid = 0;
    { const char *e = getenv("LLM_LDS_GRID"); if (e) r->lds_grid = atoi(e) != 0; }
    r->moe_fused_decode = 1;
    { const char *e = getenv("LLM_MOE_FUSED"); if (e) r->moe_fused_decode = atoi(e) != 0; }
    r->moe_iq2_bm = 0;
    { const char *e = getenv("LLM_MOE_BM");
      if (e && atoi(e)) { r->moe_iq2_bm = 1; } }
    r->moe_int8 = 0;
    { const char *e = getenv("LLM_MOE_INT8");
      if (e && atoi(e)) { r->moe_int8 = 1; } }
    r->ssm_fused_decode = 1;
    r->ffn_iq3_mw = 1;  /* warp-per-row gate_up: 313->400 GB/s, validated */
    { const char *e = getenv("LLM_FFN_IQ3_MW"); if (e) r->ffn_iq3_mw = atoi(e) != 0; }
    r->mw_threads = 256;
    { const char *e = getenv("LLM_MW_THREADS"); if (e) { int t = atoi(e); if (t==64||t==128||t==256||t==512) r->mw_threads = t; } }
    r->q4k_g4 = 1;  /* warp-per-row G=nb*4 Q4_K: 314->513 GB/s, validated */
    r->q2k_g4 = 1; { const char *e = getenv("LLM_Q2K_G4"); if (e) r->q2k_g4 = atoi(e) != 0; }
    r->q3k_g4 = 1; { const char *e = getenv("LLM_Q3K_G4"); if (e) r->q3k_g4 = atoi(e) != 0; }
    { const char *e = getenv("LLM_Q4K_G4"); if (e) r->q4k_g4 = atoi(e) != 0; }
    r->down_ksplit = 1;
    { const char *e = getenv("LLM_DOWN_KSPLIT"); if (e) { r->down_ksplit = atoi(e); if (r->down_ksplit < 1) r->down_ksplit = 1; } }
    r->ssm_out_mw = 1;  /* warp-per-row + precomputed inv_mean: 240->404 GB/s, validated */
    { const char *e = getenv("LLM_SSM_OUT_MW"); if (e) r->ssm_out_mw = atoi(e) != 0; }
    { const char *e = getenv("LLM_SSM_FUSED"); if (e) r->ssm_fused_decode = atoi(e) != 0; }
    r->gemm_own = 1;  /* self-owned WMMA GEMM default (faster than blaslt path) */
    { const char *e = getenv("LLM_GEMM"); if (e && strcmp(e, "blaslt") == 0) r->gemm_own = 0; }
    {
        const size_t QCAP = 8192;
        if (hipMalloc(&r->d_act_q8,     QCAP)                     != hipSuccess ||
            hipMalloc(&r->d_act_scale,  (QCAP/32) * sizeof(float)) != hipSuccess ||
            hipMalloc(&r->d_act_q8_b,   QCAP)                     != hipSuccess ||
            hipMalloc(&r->d_act_scale_b,(QCAP/32) * sizeof(float)) != hipSuccess) {
            fprintf(stderr, "hip_llm: q8 activation scratch alloc failed; DP4A disabled\n");
            r->decode_dp4a = 0;
        }
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

static int upload_norm_bf16(void **d_ptr, const qtensor *t, int n) {
    if (!t->data) { *d_ptr = NULL; return 0; }
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) return -1;
    dequant_row(t->type, t->data, buf, n);
    uint16_t *h = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!h) { free(buf); return -1; }
    for (int i = 0; i < n; i++) {
        uint32_t bits; memcpy(&bits, &buf[i], 4);
        uint32_t r = bits + 0x7fff + ((bits >> 16) & 1);
        h[i] = (uint16_t)(r >> 16);
    }
    free(buf);
    if (hipMalloc(d_ptr, n * 2) != hipSuccess) { free(h); return -1; }
    hipError_t err = hipMemcpy(*d_ptr, h, n * 2, hipMemcpyHostToDevice);
    free(h);
    if (err != hipSuccess) { hipFree(*d_ptr); *d_ptr = NULL; return -1; }
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
/* Upload with optional block-major repack. When repack_bm=1 and type matches,
 * transpose each expert from row-major [N][nb][bs] to block-major [nb][N][bs]
 * on the host before upload, so the dequant kernel's per-row-tile read of one
 * 256-block coalesces instead of stride (recovers cache-line over-fetch). */
static int upload_3d_kquant_raw_bm(void **d_ptr, const qtensor *t, size_t *out_stride,
                                    int repack_bm) {
    if (!t->data) { *d_ptr = NULL; return 0; }
    size_t row_bytes = dequant_row_size(t->type, t->n_cols);
    int rows_per_expert = (t->n_dims >= 3) ? (int)t->dims[1] : t->n_rows;
    *out_stride = row_bytes * (size_t)rows_per_expert;
    size_t total_bytes = row_bytes * (size_t)t->n_rows;
    int repackable = (t->n_cols % 256 == 0) &&
        (t->type == GGML_TYPE_IQ2_XXS || t->type == GGML_TYPE_IQ3_XXS ||
         t->type == GGML_TYPE_IQ2_S   || t->type == GGML_TYPE_IQ3_S);
    const void *src = t->data;
    void *repacked = NULL;
    if (repack_bm && repackable) {
        int nb = t->n_cols / 256, N = rows_per_expert, n_exp = t->n_rows / rows_per_expert;
        size_t bs = row_bytes / nb;  /* bytes per 256-block */
        size_t estride = (size_t)N * row_bytes;  /* per-expert byte stride */
        repacked = malloc(total_bytes);
        if (!repacked) { fprintf(stderr, "hip_llm: repack malloc failed\n"); return -1; }
        const unsigned char *s = (const unsigned char *)t->data;
        unsigned char *d = (unsigned char *)repacked;
        for (int e = 0; e < n_exp; e++) {
            const unsigned char *se = s + (size_t)e * estride;
            unsigned char *de = d + (size_t)e * estride;
            for (int n = 0; n < N; n++)
                for (int bg = 0; bg < nb; bg++)
                    memcpy(de + (size_t)bg * N * bs + (size_t)n * bs,
                           se + (size_t)n * nb * bs + (size_t)bg * bs, bs);
        }
        src = repacked;
    }
    hipError_t err = hipMalloc(d_ptr, total_bytes);
    if (err != hipSuccess) { fprintf(stderr, "hip_llm: upload_3d_kquant alloc failed (%zu bytes, type=%d, err=%d)\n", total_bytes, t->type, (int)err); free(repacked); return -1; }
    err = hipMemcpy(*d_ptr, src, total_bytes, hipMemcpyHostToDevice);
    free(repacked);
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
    if (gguf->kv[idx].type == GGUF_TYPE_UINT64) return (int)gguf->kv[idx].value.u64;
    if (gguf->kv[idx].type == GGUF_TYPE_INT64)  return (int)gguf->kv[idx].value.i64;
    if (gguf->kv[idx].type == GGUF_TYPE_BOOL)   return gguf->kv[idx].value.b ? 1 : 0;
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

static int hip_llm_finalize_load(hip_llm_runner *r, int max_seq_len);

/* ======================================================================== */
/* Qwen3 safetensors weight load (text-encoder path for Flux.2 Klein).      */
/* Builds qtensor views over the BF16/F32 safetensors data and reuses the   */
/* same upload_weight_matrix / upload_norm_f32 helpers as the GGUF path.    */
/* ======================================================================== */

/* Build a qtensor view of a safetensors tensor (no copy). n_cols = innermost
 * (contiguous) dim, matching the GGUF qtensor convention used by the upload
 * helpers; safetensors [out,in] row-major == GGUF byte layout (no transpose). */
static qtensor hllm_st_load_tensor(st_context *st, const char *name, int required) {
    qtensor t = {0};
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (required) fprintf(stderr, "hip_llm: missing safetensors tensor '%s'\n", name);
        return t;
    }
    const char *dt = safetensors_dtype(st, idx);
    if      (strcmp(dt, "BF16") == 0) t.type = GGML_TYPE_BF16;
    else if (strcmp(dt, "F16")  == 0) t.type = GGML_TYPE_F16;
    else if (strcmp(dt, "F32")  == 0) t.type = GGML_TYPE_F32;
    else { fprintf(stderr, "hip_llm: unsupported safetensors dtype '%s' for '%s'\n", dt, name); return t; }
    int nd = safetensors_ndims(st, idx);
    const uint64_t *sh = safetensors_shape(st, idx);
    t.n_dims = nd;
    if (nd >= 2) { t.n_cols = (int)sh[nd - 1]; uint64_t rows = 1; for (int d = 0; d < nd - 1; d++) rows *= sh[d]; t.n_rows = (int)rows; }
    else if (nd == 1) { t.n_cols = (int)sh[0]; t.n_rows = 1; }
    for (int d = 0; d < nd && d < 4; d++) t.dims[d] = (int)sh[d];
    t.data = safetensors_data(st, idx);
    return t;
}

int hip_llm_load_weights_qwen3_safetensors(hip_llm_runner *r, const char *model_path, int max_seq_len) {
    if (!r || !model_path) return -1;
    st_context *st = safetensors_open(model_path);
    if (!st) { fprintf(stderr, "hip_llm: failed to open safetensors '%s'\n", model_path); return -1; }

    /* Discover layer count from tensor names. */
    int n_layers = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strncmp(nm, "model.layers.", 13) == 0) {
            int l = atoi(nm + 13);
            if (l + 1 > n_layers) n_layers = l + 1;
        }
    }
    if (n_layers <= 0) { fprintf(stderr, "hip_llm: no model.layers.* tensors in '%s'\n", model_path); safetensors_close(st); return -1; }

    qtensor embd = hllm_st_load_tensor(st, "model.embed_tokens.weight", 1);
    qtensor q0   = hllm_st_load_tensor(st, "model.layers.0.self_attn.q_proj.weight", 1);
    qtensor k0   = hllm_st_load_tensor(st, "model.layers.0.self_attn.k_proj.weight", 1);
    qtensor qn0  = hllm_st_load_tensor(st, "model.layers.0.self_attn.q_norm.weight", 1);
    qtensor ff0  = hllm_st_load_tensor(st, "model.layers.0.mlp.gate_proj.weight", 1);
    if (!embd.data || !q0.data || !k0.data || !ff0.data) { safetensors_close(st); return -1; }

    /* embed_tokens.weight is [n_vocab, n_embd] -> n_cols=n_embd, n_rows=n_vocab. */
    r->n_embd     = embd.n_cols;
    r->n_vocab    = embd.n_rows;
    r->n_layers   = n_layers;
    r->head_dim   = qn0.n_cols ? qn0.n_cols : 128;
    r->n_heads    = q0.n_rows / r->head_dim;     /* q_proj [n_heads*head_dim, n_embd] */
    r->n_kv_heads = k0.n_rows / r->head_dim;
    r->n_ff       = ff0.n_rows;                  /* gate_proj [n_ff, n_embd] */
    r->rms_norm_eps   = 1e-6f;
    r->rope_freq_base = 1000000.0f;              /* Qwen3 */
    r->n_rope_pairs   = 0;
    r->use_mrope      = 0;
    r->is_hybrid      = 0;
    r->is_moe         = 0;
    r->n_deepstack    = 0;
    if (max_seq_len <= 0) max_seq_len = 2048;
    r->max_seq_len = max_seq_len;

    if (r->verbose >= 1)
        fprintf(stderr, "hip_llm: safetensors qwen3 n_embd=%d n_heads=%d n_kv_heads=%d n_layers=%d n_ff=%d head_dim=%d n_vocab=%d\n",
                r->n_embd, r->n_heads, r->n_kv_heads, r->n_layers, r->n_ff, r->head_dim, r->n_vocab);

    /* token embeddings (BF16 -> F16 via upload_weight_matrix) */
    if (upload_weight_matrix(&r->d_token_embd, &embd, &r->token_embd_type) != 0) { safetensors_close(st); return -1; }

    qtensor onorm = hllm_st_load_tensor(st, "model.norm.weight", 1);
    if (!onorm.data || upload_norm_f32(&r->d_output_norm, &onorm, r->n_embd) != 0) { safetensors_close(st); return -1; }

    /* lm_head tied to embeddings (Qwen3-4B) */
    r->d_output_w   = r->d_token_embd;
    r->output_w_type = r->token_embd_type;
    r->has_lm_head  = 1;

    r->layers = (hip_layer *)calloc(r->n_layers, sizeof(hip_layer));
    if (!r->layers) { safetensors_close(st); return -1; }

    for (int l = 0; l < r->n_layers; l++) {
        char name[160];
        hip_layer *cl = &r->layers[l];
        qtensor t;
        cl->is_ssm = 0;

        #define ST_NORM(field, suffix, n) do { \
            snprintf(name, sizeof(name), "model.layers.%d." suffix ".weight", l); \
            t = hllm_st_load_tensor(st, name, 1); \
            if (!t.data || upload_norm_f32(&cl->field, &t, (n)) != 0) { safetensors_close(st); return -1; } \
        } while (0)
        #define ST_MAT(field, suffix, rf, cf, tf) do { \
            snprintf(name, sizeof(name), "model.layers.%d." suffix ".weight", l); \
            t = hllm_st_load_tensor(st, name, 1); \
            cl->rf = t.n_rows; cl->cf = t.n_cols; \
            if (!t.data || upload_weight_matrix(&cl->field, &t, &cl->tf) != 0) { safetensors_close(st); return -1; } \
        } while (0)

        ST_NORM(attn_norm_w,   "input_layernorm",          r->n_embd);
        ST_MAT (attn_q_w,      "self_attn.q_proj",  attn_q_rows,      attn_q_cols,      attn_q_type);
        ST_MAT (attn_k_w,      "self_attn.k_proj",  attn_k_rows,      attn_k_cols,      attn_k_type);
        ST_MAT (attn_v_w,      "self_attn.v_proj",  attn_v_rows,      attn_v_cols,      attn_v_type);
        ST_MAT (attn_output_w, "self_attn.o_proj",  attn_output_rows, attn_output_cols, attn_output_type);
        cl->has_qk_norm = 1;
        ST_NORM(attn_q_norm_w, "self_attn.q_norm",         r->head_dim);
        ST_NORM(attn_k_norm_w, "self_attn.k_norm",         r->head_dim);
        ST_NORM(ffn_norm_w,    "post_attention_layernorm", r->n_embd);
        ST_MAT (ffn_gate_w,    "mlp.gate_proj",     ffn_gate_rows, ffn_gate_cols, ffn_gate_type);
        ST_MAT (ffn_up_w,      "mlp.up_proj",       ffn_up_rows,   ffn_up_cols,   ffn_up_type);
        ST_MAT (ffn_down_w,    "mlp.down_proj",     ffn_down_rows, ffn_down_cols, ffn_down_type);
        #undef ST_NORM
        #undef ST_MAT
    }

    safetensors_close(st);
    return hip_llm_finalize_load(r, max_seq_len);
}

int hip_llm_load_weights(hip_llm_runner *r, gguf_context *gguf, int max_seq_len) {
    if (!r || !gguf) return -1;

    const char *arch = "qwen2";
    if (gguf_find_key(gguf, "gemma4.block_count") >= 0) arch = "gemma4";
    else if (gguf_find_key(gguf, "qwen35moe.block_count") >= 0) arch = "qwen35moe";
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

    /* Gemma4 architecture */
    r->is_gemma4 = 0;
    r->swa_pattern = NULL;
    r->per_layer_kv_heads = NULL;
    r->h_rope_freq_factors = NULL;
    r->h_rope_inv_freq_full = NULL;
    r->h_rope_inv_freq_swa = NULL;
    if (strcmp(arch, "gemma4") == 0) {
        r->is_gemma4 = 1;
        r->head_dim_full = hllm_get_int(gguf, ARCH_KEY("attention.key_length"), 512);
        r->head_dim_swa  = hllm_get_int(gguf, ARCH_KEY("attention.key_length_swa"), 256);
        r->head_dim = r->head_dim_full; /* use max for buffer sizing */
        r->swa_window_size = hllm_get_int(gguf, ARCH_KEY("attention.sliding_window"), 512);
        /* Prefill chunk for SWA correctness: a query at position p attends to keys
         * [p-window+1, p]. With a circular SWA cache the store of a whole chunk must
         * not overwrite keys still needed by earlier queries in the same chunk, so
         * the cache must hold window+chunk recent positions. We set chunk = window
         * and cache = 2*window (capped to batch_max later at the call site). */
        r->gemma_prefill_chunk = r->swa_window_size;
        r->swa_cache_len = r->swa_window_size + r->gemma_prefill_chunk;
        /* Gemma4 rotates the full per-layer head_dim (n_rot == head_dim for both SWA
         * and full layers). Force n_rope_pairs=0 so the rope kernels derive rope_dim
         * from the per-call head_dim instead of a single global value (which would be
         * 512 and wrong for the 256-wide SWA heads). */
        r->n_rope_pairs = 0;
        /* WMMA flash kernel for the full-attention layers (head_dim 512), the O(M^2)
         * long-context bottleneck. Splits the 512 head_dim across 4 waves so the O
         * accumulator stays in registers (good occupancy); 1.1-1.4x+ faster than the
         * scalar online-softmax kernel and growing with context. Default ON;
         * GEMMA_FA_WMMA=0 falls back to the scalar kernel. */
        { const char *e = getenv("GEMMA_FA_WMMA"); r->gemma_fa_wmma = (e && atoi(e) == 0) ? 0 : 1; }
        r->n_embd_per_layer = hllm_get_int(gguf, ARCH_KEY("embedding_length_per_layer_input"), 256);
        int shared_kv_layers = hllm_get_int(gguf, ARCH_KEY("attention.shared_kv_layers"), 0);
        r->n_layer_kv_from_start = r->n_layers - shared_kv_layers;
        r->final_logit_softcapping = hllm_get_float(gguf, ARCH_KEY("final_logit_softcapping"), 30.0f);
        r->rope_freq_base_swa = hllm_get_float(gguf, ARCH_KEY("rope.freq_base_swa"), 10000.0f);
        r->embd_scale = sqrtf((float)r->n_embd);

        /* Parse SWA layer pattern */
        r->swa_pattern = (int *)calloc(r->n_layers, sizeof(int));
        {
            int idx = gguf_find_key(gguf, ARCH_KEY("attention.sliding_window_pattern"));
            if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY) {
                int n = (int)gguf->kv[idx].value.arr.n;
                if (n > r->n_layers) n = r->n_layers;
                uint8_t *data = (uint8_t *)gguf->kv[idx].value.arr.data;
                for (int i = 0; i < n; i++)
                    r->swa_pattern[i] = data[i] ? 1 : 0;
            } else {
                for (int i = 0; i < r->n_layers; i++)
                    r->swa_pattern[i] = ((i + 1) % 6 != 0) ? 1 : 0;
            }
        }

        /* Parse per-layer KV head count */
        r->per_layer_kv_heads = (int *)calloc(r->n_layers, sizeof(int));
        {
            int idx = gguf_find_key(gguf, ARCH_KEY("attention.head_count_kv"));
            if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY) {
                int n = (int)gguf->kv[idx].value.arr.n;
                int32_t *data = (int32_t *)gguf->kv[idx].value.arr.data;
                for (int i = 0; i < r->n_layers && i < n; i++)
                    r->per_layer_kv_heads[i] = data[i];
                for (int i = n; i < r->n_layers; i++)
                    r->per_layer_kv_heads[i] = r->n_kv_heads;
            } else {
                for (int i = 0; i < r->n_layers; i++)
                    r->per_layer_kv_heads[i] = r->n_kv_heads;
            }
        }

        /* Load proportional RoPE frequency factors (used by full-attention layers). */
        r->d_rope_freq_factors = NULL;
        r->n_rope_freq_factors = 0;
        {
            qtensor rope_freqs = hllm_load_tensor(gguf, "rope_freqs.weight", 0);
            if (rope_freqs.data) {
                int n_freq = rope_freqs.n_cols;
                r->h_rope_freq_factors = (float *)calloc(n_freq, sizeof(float));
                dequant_row(rope_freqs.type, rope_freqs.data, r->h_rope_freq_factors, n_freq);
                r->n_rope_freq_factors = n_freq;
                if (hipMalloc(&r->d_rope_freq_factors, (size_t)n_freq * sizeof(float)) == hipSuccess)
                    hipMemcpy(r->d_rope_freq_factors, r->h_rope_freq_factors,
                              (size_t)n_freq * sizeof(float), hipMemcpyHostToDevice);
            }
        }

        /* Precompute inverse frequencies for full-attn layers */
        {
            int half = r->head_dim_full / 2;
            r->rope_full_dim = half;
            r->h_rope_inv_freq_full = (float *)calloc(half, sizeof(float));
            float base = r->rope_freq_base;
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(base, (float)(2 * i) / (float)r->head_dim_full);
                if (r->h_rope_freq_factors && i < half)
                    freq /= r->h_rope_freq_factors[i];
                r->h_rope_inv_freq_full[i] = freq;
            }
        }

        /* Precompute inverse frequencies for SWA layers */
        {
            int half = r->head_dim_swa / 2;
            r->rope_swa_dim = half;
            r->h_rope_inv_freq_swa = (float *)calloc(half, sizeof(float));
            float base = r->rope_freq_base_swa;
            for (int i = 0; i < half; i++) {
                r->h_rope_inv_freq_swa[i] = 1.0f / powf(base, (float)(2 * i) / (float)r->head_dim_swa);
            }
        }

        if (r->verbose >= 1) {
            fprintf(stderr, "hip_llm: Gemma4: head_dim_full=%d head_dim_swa=%d swa_window=%d\n",
                    r->head_dim_full, r->head_dim_swa, r->swa_window_size);
            fprintf(stderr, "hip_llm: Gemma4: n_embd_per_layer=%d n_layer_kv_from_start=%d\n",
                    r->n_embd_per_layer, r->n_layer_kv_from_start);
            fprintf(stderr, "hip_llm: Gemma4: softcap=%.1f rope_base_swa=%.0f embd_scale=%.1f\n",
                    r->final_logit_softcapping, r->rope_freq_base_swa, r->embd_scale);
        }
    }

    r->n_deepstack = hllm_get_int(gguf, ARCH_KEY("n_deepstack_layers"), 0);
    if (r->verbose >= 1 && r->n_deepstack > 0) {
        fprintf(stderr, "hip_llm: n_deepstack=%d\n", r->n_deepstack);
    }

    #undef ARCH_KEY

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

        if (r->is_gemma4) {
            /* === Gemma4 per-layer setup === */
            cl->is_swa = r->swa_pattern[l];
            cl->shared_kv_source = -1;
            cl->layer_scale_val = 1.0f;
            cl->post_attn_norm_w = NULL;
            cl->post_ffw_norm_w = NULL;
            cl->ple_inp_gate_w = NULL;
            cl->ple_proj_w = NULL;
            cl->ple_post_norm_w = NULL;

            int hd = cl->is_swa ? r->head_dim_swa : r->head_dim_full;
            int local_kv_heads = r->per_layer_kv_heads[l];
            cl->local_head_dim = hd;
            cl->local_kv_heads = local_kv_heads;

            /* Determine shared KV source */
            if (l >= r->n_layer_kv_from_start) {
                cl->shared_kv_source = r->n_layer_kv_from_start - (cl->is_swa ? 2 : 1);
                if (cl->shared_kv_source < 0) cl->shared_kv_source = 0;
            }

            /* Q projection (always present) */
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->attn_q_rows = t.n_rows; cl->attn_q_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_q_w, &t, &cl->attn_q_type) != 0) return -1;

            /* K/V only for layers with own KV */
            if (cl->shared_kv_source < 0) {
                snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
                t = hllm_load_tensor(gguf, name, 1);
                cl->attn_k_rows = t.n_rows; cl->attn_k_cols = t.n_cols;
                if (upload_weight_matrix(&cl->attn_k_w, &t, &cl->attn_k_type) != 0) return -1;

                snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
                t = hllm_load_tensor(gguf, name, 1);
                cl->attn_v_rows = t.n_rows; cl->attn_v_cols = t.n_cols;
                if (upload_weight_matrix(&cl->attn_v_w, &t, &cl->attn_v_type) != 0) return -1;
            }

            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->attn_output_rows = t.n_rows; cl->attn_output_cols = t.n_cols;
            if (upload_weight_matrix(&cl->attn_output_w, &t, &cl->attn_output_type) != 0) return -1;

            /* QK norm (per-layer head_dim) */
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->has_qk_norm = (t.data != NULL);
            if (t.data) { if (upload_norm_f32(&cl->attn_q_norm_w, &t, hd) != 0) return -1; }
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (t.data) { if (upload_norm_f32(&cl->attn_k_norm_w, &t, hd) != 0) return -1; }

            /* Post-attention norm */
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (t.data) { if (upload_norm_f32(&cl->post_attn_norm_w, &t, r->n_embd) != 0) return -1; }

            /* PLE weights */
            snprintf(name, sizeof(name), "blk.%d.inp_gate.weight", l);
            t = hllm_load_tensor(gguf, name, 0);
            if (t.data) { if (upload_weight_matrix(&cl->ple_inp_gate_w, &t, &(int){0}) != 0) return -1; }
            snprintf(name, sizeof(name), "blk.%d.proj.weight", l);
            t = hllm_load_tensor(gguf, name, 0);
            if (t.data) { if (upload_weight_matrix(&cl->ple_proj_w, &t, &(int){0}) != 0) return -1; }
            snprintf(name, sizeof(name), "blk.%d.post_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 0);
            if (t.data) { if (upload_norm_f32(&cl->ple_post_norm_w, &t, r->n_embd) != 0) return -1; }

            /* Layer output scale */
            snprintf(name, sizeof(name), "blk.%d.layer_output_scale.weight", l);
            t = hllm_load_tensor(gguf, name, 0);
            if (t.data && t.type == GGML_TYPE_F32) {
                cl->layer_scale_val = *(float *)t.data;
            } else if (t.data) {
                float sv; dequant_row(t.type, t.data, &sv, 1);
                cl->layer_scale_val = sv;
            }

        } else if (is_ssm) {
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
        if (r->is_gemma4) {
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ffn_norm_w, &t, r->n_embd) != 0) return -1;
            /* Post-FFN norm */
            snprintf(name, sizeof(name), "blk.%d.post_ffw_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (t.data) { if (upload_norm_f32(&cl->post_ffw_norm_w, &t, r->n_embd) != 0) return -1; }
        } else if (r->is_hybrid) {
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ffn_norm_w, &t, r->n_embd) != 0) return -1;
        } else {
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->ffn_norm_w, &t, r->n_embd) != 0) return -1;
        }

        if (r->is_moe) {
            cl->is_moe = 1;
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_gate_rows = t.n_rows; cl->moe_gate_cols = t.n_cols;
            if (upload_norm_f32(&cl->moe_gate_w, &t, t.n_rows * t.n_cols) != 0) return -1;
            if (upload_norm_bf16(&cl->moe_gate_w_bf16, &t, t.n_rows * t.n_cols) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_gate_exps_type = t.type;
            cl->moe_exp_cols_gu = t.n_cols;
            cl->moe_exp_rows_gu = (t.n_dims >= 3) ? (int)t.dims[1] : t.n_rows;
            if (upload_3d_kquant_raw_bm(&cl->moe_gate_exps_w, &t, &cl->moe_exp_stride_gu, r->moe_iq2_bm) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_up_exps_type = t.type;
            if (upload_3d_kquant_raw_bm(&cl->moe_up_exps_w, &t, &cl->moe_exp_stride_gu, r->moe_iq2_bm) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            cl->moe_down_exps_type = t.type;
            cl->moe_exp_cols_d = t.n_cols;
            cl->moe_exp_rows_d = (t.n_dims >= 3) ? (int)t.dims[1] : t.n_rows;
            if (upload_3d_kquant_raw_bm(&cl->moe_down_exps_w, &t, &cl->moe_exp_stride_d, r->moe_iq2_bm) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", l);
            t = hllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(&cl->moe_shared_gate_w, &t, t.n_rows * t.n_cols) != 0) return -1;
            if (upload_norm_bf16(&cl->moe_shared_gate_w_bf16, &t, t.n_rows * t.n_cols) != 0) return -1;

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

    return hip_llm_finalize_load(r, max_seq_len);
}

static int hip_llm_finalize_load(hip_llm_runner *r, int max_seq_len) {
    /* For Gemma4, use the maximum head_dim for buffer sizing */
    int kv_dim = r->n_kv_heads * r->head_dim;
    int q_dim = r->n_heads * r->head_dim;
    if (r->is_gemma4) {
        /* Gemma4: q_dim uses head_dim_full, kv_dim uses per-layer max */
        q_dim = r->n_heads * r->head_dim_full;
        kv_dim = r->n_kv_heads * r->head_dim_full;
    }

    /* Allocate KV cache */
    r->d_key_cache = (void **)calloc(r->n_layers, sizeof(void *));
    r->d_value_cache = (void **)calloc(r->n_layers, sizeof(void *));
    for (int l = 0; l < r->n_layers; l++) {
        if (r->layers[l].is_ssm) continue;
        size_t kv_cache_size;
        if (r->is_gemma4) {
            int local_kv = r->layers[l].local_kv_heads;
            int local_hd = r->layers[l].local_head_dim;
            size_t layer_kv = (size_t)local_kv * local_hd * sizeof(float);
            if (r->layers[l].shared_kv_source >= 0) {
                /* Shared KV: alias to source layer */
                int src = r->layers[l].shared_kv_source;
                r->d_key_cache[l] = r->d_key_cache[src];
                r->d_value_cache[l] = r->d_value_cache[src];
                continue;
            } else if (r->layers[l].is_swa) {
                /* Circular cache holds window+chunk recent positions (see loader). */
                kv_cache_size = (size_t)r->swa_cache_len * layer_kv;
            } else {
                /* Full-attention layer. Only 1 KV head (num_global_key_value_heads),
                 * so layer_kv is small even at long context; allocate the linear
                 * max_seq_len cache. Value-less (V=K) layers still need a V cache:
                 * V is the un-RoPE'd normed K, which differs from the RoPE'd K cache. */
                kv_cache_size = (size_t)max_seq_len * layer_kv;
            }
        } else {
            kv_cache_size = (size_t)max_seq_len * kv_dim * sizeof(float);
        }
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
        /* Device-side dispatch buffers (sync-free MoE routing) */
        CHECK_HIP(hipMalloc(&r->d_moe_idx, r->n_experts_used * sizeof(int)));
        CHECK_HIP(hipMalloc(&r->d_moe_w,   r->n_experts_used * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_shared_scale, sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_router_counter, sizeof(unsigned int)));
        CHECK_HIP(hipMemset(r->d_router_counter, 0, sizeof(unsigned int)));
        CHECK_HIP(hipMalloc(&r->d_moe_act8,
                            (size_t)(r->n_experts_used + 1) * r->expert_ff * sizeof(float)));
        /* Device-side expert dispatch is supported only when every routed-expert
         * weight type has an expert-indexed kernel (IQ2_S / IQ3_S / IQ4_XS). */
        r->moe_dev_dispatch_ok = 1;
        for (int l = 0; l < r->n_layers; l++) {
            hip_layer *cl = &r->layers[l];
            if (!cl->is_moe) continue;
            int ts[3] = { cl->moe_gate_exps_type, cl->moe_up_exps_type, cl->moe_down_exps_type };
            for (int j = 0; j < 3; j++) {
                if (ts[j] != GGML_TYPE_IQ2_S && ts[j] != GGML_TYPE_IQ3_S &&
                    ts[j] != GGML_TYPE_IQ4_XS) { r->moe_dev_dispatch_ok = 0; }
            }
        }
        if (r->verbose >= 1) {
            fprintf(stderr, "hip_llm: MoE device-side dispatch %s\n",
                    r->moe_dev_dispatch_ok ? "ENABLED (sync-free, graph-capturable)"
                                           : "disabled (unsupported expert type)");
        }
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
        CHECK_HIP(hipMalloc(&r->d_ssm_inv_mean,  (dt_rank > 0 ? dt_rank : 256) * sizeof(float)));
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

    /* Gemma4 PLE buffers */
    if (r->is_gemma4 && r->n_embd_per_layer > 0) {
        int ple_dim = r->n_embd_per_layer;
        CHECK_HIP(hipMalloc(&r->d_ple_combined, (size_t)r->n_layers * ple_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ple_buf, ple_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ple_proj, r->n_embd * sizeof(float)));
        if (r->verbose >= 1)
            fprintf(stderr, "hip_llm: Gemma4 PLE buffers allocated (ple_dim=%d, %.1f MB)\n",
                    ple_dim, (double)(r->n_layers * ple_dim + ple_dim + r->n_embd) * sizeof(float) / (1024.0*1024.0));
    }

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
        /* Gemma4 prefill is chunked at gemma_prefill_chunk rows; sizing the batch
         * scratch buffers larger than that only wastes VRAM (and risks OOM at the
         * large KV caches needed for long context). */
        if (r->is_gemma4 && r->gemma_prefill_chunk > 0) batch_max = r->gemma_prefill_chunk;
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

        /* Eligibility: any model where the standard-attn/SSM/FFN projections have a
         * batchable weight type. Hybrid models use B2 Phase 2/3 for gated-attn / SSM.
         * Phase 2 (prefill MoE): hybrid+MoE models are now eligible — the SSM and
         * attention projections batch via hipBLASLt, and the MoE FFN runs per-row of
         * the batch (forward_moe_ffn). Requires the sync-free device MoE dispatch so
         * the per-row loop has no host round-trips. Gate: LLM_MOE_PREFILL (default 1). */
        int eligible = !disabled;
        if (r->is_moe) {
         /* Batched MoE prefill (LLM_MOE_PREFILL, default ON): SSM/attn projections
              * batch via hipBLASLt; experts are grouped by token and run through the
              * fused QUANTIZED MoE GEMM (mmq_iq2s/iq3s — weight stays quantized, dequant
              * amortized over the token tile). Batch chunked to <=256 (Tensile plan
              * limit). Prefill ~400 tok/s vs ~48 per-token (8x). Set =0 to disable.
              * Does NOT require moe_dev_dispatch_ok — GPU grouping (topK/gather) works
              * for any expert type; the batched MoE GEMM path dequantizes each expert's
              * weights to BF16 on-the-fly via get_bf16_weight. */
             const char *env_mp = getenv("LLM_MOE_PREFILL");
             int moe_prefill = (env_mp ? atoi(env_mp) != 0 : 1);
             r->moe_prefill_batched = moe_prefill;
             if (!moe_prefill || !r->is_hybrid) eligible = 0;
        }

        /* Smart default for large DENSE (non-MoE) models: the FFN GEMMs have big
         * N (n_ff) and K (n_embd), where hipBLASLt markedly outperforms the
         * self-owned WMMA GEMM (measured 1.73x prefill on Qwen3.6-27B dense,
         * n_embd=5120/n_ff=17408). The self-owned GEMM still wins for the 35B
         * MoE grouped path (which requires gemm_own) and for small dense models
         * (n_embd<4096, e.g. Qwen3-VL-2B), so restrict the flip accordingly.
         * An explicit LLM_GEMM=own/blaslt always overrides. */
        {
            const char *eg = getenv("LLM_GEMM");
            int user_forced = (eg && (strcmp(eg, "own") == 0 || strcmp(eg, "blaslt") == 0));
            if (!user_forced && eligible && !r->is_moe && (r->n_embd >= 4096 || r->is_gemma4)) {
                r->gemm_own = 0;  /* prefer hipBLASLt for large dense GEMMs */
            }
        }
        if (eligible && !r->gemm_own) {
            if (mm_blaslt_init() != 0) {
                fprintf(stderr,
                    "hip_llm: mm_blaslt_init unavailable; using self-owned WMMA GEMM\n");
                r->gemm_own = 1;
            }
        }
        if (r->verbose >= 1 && eligible)
            fprintf(stderr, "hip_llm: prefill GEMM backend = %s\n",
                    r->gemm_own ? "own (WMMA, no hipBLASLt)" : "hipBLASLt");

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
                size_t n_qkv  = (size_t)cl->ssm_qkv_rows  * cl->ssm_qkv_cols;
                size_t n_sg   = (size_t)cl->ssm_gate_rows * cl->ssm_gate_cols;
                size_t n_a    = (size_t)cl->ssm_alpha_rows * cl->ssm_alpha_cols;
                size_t n_b    = (size_t)cl->ssm_beta_rows  * cl->ssm_beta_cols;
                size_t n_so   = (size_t)cl->ssm_out_rows   * cl->ssm_out_cols;
                if (n_q > max_w_elems) max_w_elems = n_q;
                if (n_k > max_w_elems) max_w_elems = n_k;
                if (n_v > max_w_elems) max_w_elems = n_v;
                if (n_o > max_w_elems) max_w_elems = n_o;
                if (n_g > max_w_elems) max_w_elems = n_g;
                if (n_u > max_w_elems) max_w_elems = n_u;
                if (n_d > max_w_elems) max_w_elems = n_d;
                if (n_qkv > max_w_elems) max_w_elems = n_qkv;
                if (n_sg  > max_w_elems) max_w_elems = n_sg;
                if (n_a   > max_w_elems) max_w_elems = n_a;
                if (n_b   > max_w_elems) max_w_elems = n_b;
                if (n_so  > max_w_elems) max_w_elems = n_so;
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
                DO_W(ssm_qkv_w_bf16,     ssm_qkv_w,     ssm_qkv_type,     n_qkv);
                DO_W(ssm_gate_w_bf16,    ssm_gate_w,    ssm_gate_type,    n_sg);
                DO_W(ssm_alpha_w_bf16,   ssm_alpha_w,   ssm_alpha_type,   n_a);
                DO_W(ssm_beta_w_bf16,    ssm_beta_w,    ssm_beta_type,    n_b);
                DO_W(ssm_out_w_bf16,     ssm_out_w,     ssm_out_type,     n_so);
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
            {
                /* Also used as the packed ssm_out scratch — must hold d_inner. */
                size_t silu_cols = (size_t)r->n_ff;
                if (r->is_hybrid && (size_t)r->ssm_d_inner > silu_cols) silu_cols = r->ssm_d_inner;
                CHECK_HIP(hipMalloc(&r->d_silu_batch_bf16, (size_t)bm * silu_cols * 2));
            }
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

            /* Batched token-grouped MoE prefill buffers (LLM_MOE_PREFILL=1). */
            if (r->is_moe && r->moe_prefill_batched) {
                int bm = r->batch_max;
                int ne = r->n_experts, K = r->n_experts_used;
                int eff = r->expert_ff;
                size_t TA = (size_t)bm * K;  /* max assignments */
                CHECK_HIP(hipMalloc(&r->d_router_logits_batch, (size_t)bm * ne * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_router_w_bf16,       (size_t)ne * r->n_embd * 2));
                CHECK_HIP(hipMalloc(&r->d_moe_gather_in,       TA * r->n_embd * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_moe_gather_in_bf16,  TA * r->n_embd * 2));
                CHECK_HIP(hipMalloc(&r->d_moe_eg,              TA * eff * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_moe_eu,              TA * eff * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_moe_esilu_bf16,      TA * eff * 2));
                CHECK_HIP(hipMalloc(&r->d_moe_eout,            TA * r->n_embd * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_moe_out_batch,       (size_t)bm * r->n_embd * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_xnorm_batch_bf16_moe,(size_t)bm * r->n_embd * 2));
                CHECK_HIP(hipMalloc(&r->d_shared_scale_batch,  (size_t)bm * sizeof(float)));
                CHECK_HIP(hipMalloc(&r->d_moe_gather_src,      TA * sizeof(int)));
                CHECK_HIP(hipMalloc(&r->d_moe_gather_w,        TA * sizeof(float)));
                /* All-expert bf16 staging (one weight type at a time) + device offsets */
                {
                    size_t gu = (size_t)ne * r->expert_ff * r->n_embd * 2;
                    size_t dn = (size_t)ne * r->n_embd * r->expert_ff * 2;
                    size_t sz = gu > dn ? gu : dn;
                    CHECK_HIP(hipMalloc(&r->d_expw_bf16, sz));
                    CHECK_HIP(hipMalloc(&r->d_expw_int8, sz / 2));  /* INT8: 1 byte vs BF16's 2 */
                    { size_t _act_sz = (size_t)TA * (r->n_embd > eff ? r->n_embd : eff);
                      CHECK_HIP(hipMalloc(&r->d_act_int8, _act_sz)); }
                    CHECK_HIP(hipMalloc(&r->d_moe_offs, (size_t)(ne + 1) * sizeof(int)));
                    CHECK_HIP(hipMalloc(&r->d_tok_idx, TA * sizeof(int)));
                    CHECK_HIP(hipMalloc(&r->d_tok_w,   TA * sizeof(float)));
                    CHECK_HIP(hipMalloc(&r->d_cursor,  (size_t)ne * sizeof(int)));
                    if (r->verbose >= 1)
                        fprintf(stderr, "hip_llm: grouped-expert staging %.0f MB\n", sz / 1048576.0);
                }
                r->h_router_batch    = (float *)malloc((size_t)bm * ne * sizeof(float));
                r->h_moe_gather_src  = (int *)malloc(TA * sizeof(int));
                r->h_moe_gather_w    = (float *)malloc(TA * sizeof(float));
                r->h_moe_tok_idx     = (int *)malloc(TA * sizeof(int));
                r->h_moe_tok_w       = (float *)malloc(TA * sizeof(float));
                r->h_moe_offsets     = (int *)malloc((size_t)(ne + 1) * sizeof(int));
                if (!r->h_router_batch || !r->h_moe_gather_src || !r->h_moe_gather_w ||
                    !r->h_moe_tok_idx || !r->h_moe_tok_w || !r->h_moe_offsets) return -1;
                if (r->verbose >= 1)
                    fprintf(stderr, "hip_llm: batched MoE prefill buffers: ~%.0f MB\n",
                            (double)(TA * (r->n_embd*6 + eff*5)) / (1024.0*1024.0));
            }

            /* Plan pre-warm: fire the 7 hipBLASLt prefill GEMM shapes once at
             * representative M values so the per-(M,N,K) plan cache is hot on
             * the first user call. Set LLM_PLAN_PREWARM=0 to skip. */
            const char *prewarm_env = getenv("LLM_PLAN_PREWARM");
            int do_prewarm = (prewarm_env == NULL) ? 1 : atoi(prewarm_env);
            if (r->gemm_own) do_prewarm = 0;  /* own kernel needs no plan cache */
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
                        gemm_run_bf16_w(r, r->d_q_batch, cl0->attn_q_w_bf16,
                            r->d_xnorm_batch_bf16, M, q_dim, r->n_embd, r->stream);
                    if (cl0->attn_k_w_bf16)
                        gemm_run_bf16_w(r, r->d_k_batch, cl0->attn_k_w_bf16,
                            r->d_xnorm_batch_bf16, M, kv_dim, r->n_embd, r->stream);
                    if (cl0->attn_v_w_bf16)
                        gemm_run_bf16_w(r, r->d_v_batch, cl0->attn_v_w_bf16,
                            r->d_xnorm_batch_bf16, M, kv_dim, r->n_embd, r->stream);
                    if (cl0->attn_output_w_bf16)
                        gemm_run_bf16_w(r, r->d_attn_proj_batch, cl0->attn_output_w_bf16,
                            r->d_attn_out_batch_bf16, M, r->n_embd, q_dim, r->stream);
                    if (cl0->ffn_gate_w_bf16)
                        gemm_run_bf16_w(r, r->d_gate_batch, cl0->ffn_gate_w_bf16,
                            r->d_ffn_norm_batch_bf16, M, r->n_ff, r->n_embd, r->stream);
                    if (cl0->ffn_up_w_bf16)
                        gemm_run_bf16_w(r, r->d_up_batch, cl0->ffn_up_w_bf16,
                            r->d_ffn_norm_batch_bf16, M, r->n_ff, r->n_embd, r->stream);
                    if (cl0->ffn_down_w_bf16)
                        gemm_run_bf16_w(r, r->d_down_batch, cl0->ffn_down_w_bf16,
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

/* GEMM router: hipBLASLt or self-owned WMMA GEMM (LLM_GEMM=own / no-blaslt build).
 * Drop-in for mm_blaslt_run_bf16: Y[M,N]f32 = X[M,K]bf16 x W[N,K]^T bf16. */
static inline int gemm_run_bf16_w(hip_llm_runner *r, void *Y, const void *W,
                                  const void *X, int M, int N, int K, void *stream) {
    if (!r->gemm_own)
        return mm_blaslt_run_bf16(Y, W, X, M, N, K, stream);
    void *args[] = { &Y, &W, &X, &N, &K, &M };
    hipFunction_t fn = (K >= 128 && r->fn_gemm_bf16_own_db) ? r->fn_gemm_bf16_own_db
                                                            : r->fn_gemm_bf16_own;
    hipError_t err = LAUNCH(fn, (unsigned)((N + 127) / 128),
                            (unsigned)((M + 127) / 128), 1, 256, 1, 1, 0,
                            (hipStream_t)stream, args);
    if (err != hipSuccess)
        fprintf(stderr, "hip_llm: gemm_bf16_own launch failed (M=%d N=%d K=%d err=%d)\n",
                M, N, K, (int)err);
    return err == hipSuccess ? 0 : -1;
}

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

/* Weightless per-head RMS norm over n_rows (n_rows=1 for decode). Handles head_dim>256. */
static inline void launch_raw_rmsnorm_heads_batch(hip_llm_runner *r, void *vec,
                                                  int n_heads, int head_dim,
                                                  int n_rows, int row_stride, float eps) {
    void *args[] = { &vec, &n_heads, &head_dim, &n_rows, &row_stride, &eps };
    LAUNCH(r->fn_raw_rmsnorm_heads_batch_f32, n_heads, n_rows, 1, 256, 1, 1,
           256 * sizeof(float), r->stream, args);
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
static inline void launch_rope_devp_ff(hip_llm_runner *r, void *vec, int n_heads,
                                        int head_dim, float freq_base, void *freq_factors) {
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
        void *args[] = { &vec, &n_heads, &head_dim, &pos_p, &freq_base, &n_rope_pairs, &freq_factors };
        LAUNCH(r->fn_rope_neox_f32_devp, n_heads, 1, 1, half_dim, 1, 1, 0, r->stream, args);
    }
}
static inline void launch_rope_devp(hip_llm_runner *r, void *vec, int n_heads,
                                     int head_dim, float freq_base) {
    launch_rope_devp_ff(r, vec, n_heads, head_dim, freq_base, NULL);
}

static inline void launch_kv_store_devp(hip_llm_runner *r, void *key_cache, void *value_cache,
                                         void *k, void *v, int kv_dim, int cache_len) {
    int *pos_p = r->d_position;
    void *args[] = { &key_cache, &value_cache, &k, &v, &pos_p, &kv_dim, &cache_len };
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

static inline void launch_matvec_q2_K(hip_llm_runner *r, void *dst, void *mat,
                                      void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    if (r->q2k_g4)
        LAUNCH(r->fn_matvec_q2_K_g4_f32, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args);
    else
        LAUNCH(r->fn_matvec_q2_K_f32, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args);
}
static inline void launch_matvec_q3_K(hip_llm_runner *r, void *dst, void *mat,
                                      void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    if (r->q3k_g4)
        LAUNCH(r->fn_matvec_q3_K_g4_f32, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args);
    else
        LAUNCH(r->fn_matvec_q3_K_f32, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args);
}
static inline void launch_matvec_q4_K(hip_llm_runner *r, void *dst, void *mat,
                                      void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    if (r->q4k_g4) {
        LAUNCH(r->fn_matvec_q4_K_g4_f32, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args);
    } else if (r->quant_matvec_opt && n_rows <= 4096 && n_cols <= 4096) {
        LAUNCH(r->fn_matvec_q4_K_mw_f32, n_rows, 1, 1, 32, 1, 1, 0,
               r->stream, args);
    } else {
        LAUNCH(r->fn_matvec_q4_K_f32, n_rows, 1, 1, 64, 1, 1, 0, r->stream, args);
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
static inline void launch_matvec_q6_K(hip_llm_runner *r, void *dst, void *mat,
                                      void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    LAUNCH(r->fn_matvec_q6_K_f32, n_rows, 1, 1, 64, 1, 1, 0, r->stream, args);
}
DEFINE_LAUNCH_MATVEC_MW(iq2_xxs, fn_matvec_iq2_xxs_f32)
DEFINE_LAUNCH_MATVEC(q4_0, fn_matvec_q4_0_f32)
DEFINE_LAUNCH_MATVEC(q4_1, fn_matvec_q4_1_f32)
DEFINE_LAUNCH_MATVEC(q5_0, fn_matvec_q5_0_f32)
DEFINE_LAUNCH_MATVEC(q5_1, fn_matvec_q5_1_f32)
DEFINE_LAUNCH_MATVEC_MW(iq4_nl, fn_matvec_iq4_nl_f32)
DEFINE_LAUNCH_MATVEC_MW(iq4_xs, fn_matvec_iq4_xs_f32)
DEFINE_LAUNCH_MATVEC_MW(iq2_xs, fn_matvec_iq2_xs_f32)
DEFINE_LAUNCH_MATVEC_MW(iq3_xxs, fn_matvec_iq3_xxs_f32)
/* Quantize a F32 activation vector x[n] -> int8 per-32-block (qs) + fp32 scale. */
static inline void launch_quantize_q8(hip_llm_runner *r, void *x, int n,
                                       void *qs, void *scale) {
    void *args[] = { &qs, &scale, &x, &n };
    LAUNCH(r->fn_quantize_q8_32, (n + 31) / 32, 1, 1, 32, 1, 1, 0, r->stream, args);
}
/* IQ2_S / IQ3_S: DP4A path when enabled (quantize x then int8 dot), else the
 * full-utilization F32 fallback. Used by launch_matvec_auto + verify harness. */
static inline void launch_matvec_iq2_s(hip_llm_runner *r, void *dst, void *mat,
                                       void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    if (r->lds_grid) {
        LAUNCH(r->fn_matvec_iq2_s_lds_f32, (n_rows + 7) / 8, 1, 1, 256, 1, 1,
               1024 * sizeof(unsigned long long), r->stream, args);
    } else if (r->decode_dp4a && (n_cols % 256) == 0 && n_cols <= 8192) {
        launch_quantize_q8(r, x, n_cols, r->d_act_q8, r->d_act_scale);
        void *a2[] = { &dst, &mat, &r->d_act_q8, &r->d_act_scale, &n_rows, &n_cols };
        LAUNCH(r->fn_matvec_iq2_s_dp4a, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, a2);
    } else {
        LAUNCH(r->fn_matvec_iq2_s_f32, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args);
    }
}
static inline void launch_matvec_iq3_s(hip_llm_runner *r, void *dst, void *mat,
                                       void *x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    if (r->lds_grid) {
        LAUNCH(r->fn_matvec_iq3_s_lds_f32, (n_rows + 7) / 8, 1, 1, 256, 1, 1,
               512 * sizeof(unsigned int), r->stream, args);
    } else if (r->decode_dp4a && (n_cols % 256) == 0 && n_cols <= 8192) {
        launch_quantize_q8(r, x, n_cols, r->d_act_q8, r->d_act_scale);
        void *a2[] = { &dst, &mat, &r->d_act_q8, &r->d_act_scale, &n_rows, &n_cols };
        LAUNCH(r->fn_matvec_iq3_s_dp4a, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, a2);
    } else {
        LAUNCH(r->fn_matvec_iq3_s_f32, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args);
    }
}
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
static inline int launch_dequant_##qname##_to_bf16(hip_llm_runner *r,           \
                                                       void *dst, void *mat,      \
                                                       int n_rows, int n_cols) {  \
    void *args[] = { &dst, &mat, &n_rows, &n_cols };                             \
    int n_blocks_per_row = n_cols / 256;                                         \
    hipError_t e = LAUNCH(r->fn_dequant_##qname##_to_bf16, n_rows, n_blocks_per_row, 1,         \
           256, 1, 1, 0, r->stream, args);                                       \
    if (e != hipSuccess) { fprintf(stderr, "hip_llm: dequant_" #qname " launch failed: %d\n", e); return -1; } \
    return 0;                                                                     \
}
DEFINE_LAUNCH_KQ_DEQUANT(q2_K)
DEFINE_LAUNCH_KQ_DEQUANT(q3_K)
DEFINE_LAUNCH_KQ_DEQUANT(q4_K)
DEFINE_LAUNCH_KQ_DEQUANT(q5_K)
DEFINE_LAUNCH_KQ_DEQUANT(q6_K)
DEFINE_LAUNCH_KQ_DEQUANT(iq3_xxs)
DEFINE_LAUNCH_KQ_DEQUANT(iq4_xs)
DEFINE_LAUNCH_KQ_DEQUANT(iq2_xs)
DEFINE_LAUNCH_KQ_DEQUANT(iq2_s)
DEFINE_LAUNCH_KQ_DEQUANT(iq3_s)
DEFINE_LAUNCH_KQ_DEQUANT(iq1_s)
DEFINE_LAUNCH_KQ_DEQUANT(tq1_0)
DEFINE_LAUNCH_KQ_DEQUANT(iq2_xxs)
#undef DEFINE_LAUNCH_KQ_DEQUANT

static inline int launch_dequant_q8_0_to_bf16(hip_llm_runner *r,
                                              void *dst, void *mat,
                                              int n_rows, int n_cols) {
    if ((n_cols % 32) != 0) return -1;
    void *args[] = { &dst, &mat, &n_rows, &n_cols };
    int chunks = (n_cols + 255) / 256;
    hipError_t e = LAUNCH(r->fn_dequant_q8_0_to_bf16, n_rows, chunks, 1,
                          256, 1, 1, 0, r->stream, args);
    if (e != hipSuccess) {
        fprintf(stderr, "hip_llm: dequant_q8_0 launch failed: %d\n", e);
        return -1;
    }
    return 0;
}

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
    return type == GGML_TYPE_F32     || type == GGML_TYPE_F16 ||
           type == GGML_TYPE_Q8_0    ||
           type == GGML_TYPE_Q2_K    || type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q4_K    || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K    || type == GGML_TYPE_IQ3_XXS ||
           type == GGML_TYPE_IQ4_XS  || type == GGML_TYPE_IQ2_XS  ||
           type == GGML_TYPE_IQ2_S   || type == GGML_TYPE_IQ3_S   ||
           type == GGML_TYPE_IQ1_S   || type == GGML_TYPE_TQ1_0   ||
           type == GGML_TYPE_IQ2_XXS;
}
 
/* True if every projection weight of an attn+FFN layer has a batched path.
 * For MoE layers the dense ffn_* weights are unused (the MoE FFN is batched
 * separately by forward_moe_ffn_batched), so skip those checks. */
static inline int layer_is_batched_eligible(const hip_layer *cl) {
    int ffn_ok = cl->is_moe ? 1 :
        (batch_qtype_ok(cl->ffn_gate_type) && batch_qtype_ok(cl->ffn_up_type) &&
         batch_qtype_ok(cl->ffn_down_type));
    return batch_qtype_ok(cl->attn_q_type)      &&
           batch_qtype_ok(cl->attn_k_type)      &&
           batch_qtype_ok(cl->attn_v_type)      &&
           batch_qtype_ok(cl->attn_output_type) && ffn_ok;
}

/* True if every SSM projection weight has a batched path. MoE layers skip the
 * dense ffn_* checks (MoE FFN batched separately). */
static inline int ssm_layer_is_batched_eligible(const hip_layer *cl) {
    int ffn_ok = cl->is_moe ? 1 :
        (batch_qtype_ok(cl->ffn_gate_type) && batch_qtype_ok(cl->ffn_up_type) &&
         batch_qtype_ok(cl->ffn_down_type));
    return batch_qtype_ok(cl->ssm_qkv_type)   &&
           batch_qtype_ok(cl->ssm_gate_type)  &&
           batch_qtype_ok(cl->ssm_alpha_type) &&
           batch_qtype_ok(cl->ssm_beta_type)  &&
           batch_qtype_ok(cl->ssm_out_type)   && ffn_ok;
}

static inline void *get_bf16_weight(hip_llm_runner *r, void *raw_w, void *bf16_w,
                                      int type, int n_rows, int n_cols) {
    if (bf16_w) return bf16_w;
    switch (type) {
        case GGML_TYPE_F32: {
            /* F32 SSM weights (gate/alpha/beta/out on qwen35moe): truncate to bf16
             * into the shared staging buffer on the fly. */
            size_t n = (size_t)n_rows * n_cols;
            launch_pack_bf16_from_f32(r, r->d_wbuf_bf16, raw_w, (int)n);
            return r->d_wbuf_bf16;
        }
        case GGML_TYPE_F16: {
            /* SSM weights are typically F16 and aren't pre-converted at load
             * (only attn+FFN F16 weights get cl->*_w_bf16). Convert into the
             * shared staging buffer on the fly. */
            size_t n = (size_t)n_rows * n_cols;
            launch_convert_f16_to_bf16(r, r->d_wbuf_bf16, raw_w, (int)n);
            return r->d_wbuf_bf16;
        }
        case GGML_TYPE_Q8_0:
            if (launch_dequant_q8_0_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols) != 0) return NULL;
            return r->d_wbuf_bf16;
        case GGML_TYPE_Q2_K:
            launch_dequant_q2_K_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        case GGML_TYPE_Q3_K:
            launch_dequant_q3_K_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
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
        case GGML_TYPE_IQ2_XXS:
            launch_dequant_iq2_xxs_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols);
            return r->d_wbuf_bf16;
        case GGML_TYPE_IQ1_S:
            if (launch_dequant_iq1_s_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols) != 0) return NULL;
            return r->d_wbuf_bf16;
        case GGML_TYPE_TQ1_0:
            if (launch_dequant_tq1_0_to_bf16(r, r->d_wbuf_bf16, raw_w, n_rows, n_cols) != 0) return NULL;
            return r->d_wbuf_bf16;
        default:
            fprintf(stderr, "hip_llm: get_bf16_weight: unsupported type %d for %dx%d\n", type, n_rows, n_cols);
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

static inline void launch_rope_neox_batch_ff(hip_llm_runner *r, void *vec_batch,
                                           int n_heads, int head_dim,
                                           int position_start, float freq_base,
                                           int n_rope_pairs, int row_stride, int M,
                                           void *freq_factors) {
    int half_dim = head_dim / 2;
    int block = half_dim;
    if (block < 1) block = 1;
    void *args[] = { &vec_batch, &n_heads, &head_dim, &position_start,
                     &freq_base, &n_rope_pairs, &row_stride, &freq_factors };
    LAUNCH(r->fn_rope_neox_batch_f32, n_heads, M, 1, block, 1, 1, 0,
           r->stream, args);
}
static inline void launch_rope_neox_batch(hip_llm_runner *r, void *vec_batch,
                                           int n_heads, int head_dim,
                                           int position_start, float freq_base,
                                           int n_rope_pairs, int row_stride, int M) {
    launch_rope_neox_batch_ff(r, vec_batch, n_heads, head_dim, position_start,
                              freq_base, n_rope_pairs, row_stride, M, NULL);
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

/* Strided variant: kv_dim is the destination stride, batch_stride the source.
 * cache_len > 0 enables circular indexing (slot = pos % cache_len) for SWA. */
static inline void launch_kv_store_batch_strided(hip_llm_runner *r,
    void *key_cache, void *value_cache,
    void *k_batch, void *v_batch,
    int position_start, int M, int kv_dim, int batch_stride, int cache_len) {
    int total = M * kv_dim;
    void *args[] = { &key_cache, &value_cache, &k_batch, &v_batch,
                     &position_start, &M, &kv_dim, &batch_stride, &cache_len };
    LAUNCH(r->fn_kv_cache_store_batch_strided, (total + 255) / 256, 1, 1, 256, 1, 1, 0,
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

/* Bounded online-softmax (flash) prefill attention. Grid (n_heads, M), block 256.
 * Smem = (2*head_dim + 2*256) floats, independent of seq_len. window>0 => SWA,
 * cache_len = circular cache size. Launch error is reported (these used to fail
 * silently when the old kernel requested > 64KB LDS). */
static inline void launch_attn_prefill_flash(hip_llm_runner *r,
    void *out, void *q, void *key_cache, void *value_cache,
    int n_heads, int n_kv_heads, int head_dim, int kv_dim,
    int M, int position_start, float scale, int window, int cache_len) {
    void *args[] = { &out, &q, &key_cache, &value_cache,
        &n_heads, &n_kv_heads, &head_dim, &kv_dim,
        &M, &position_start, &scale, &window, &cache_len };
    size_t smem = ((size_t)(2 * head_dim + 2 * 256)) * sizeof(float);
    hipError_t err = LAUNCH(r->fn_attn_prefill_flash_f32, n_heads, M, 1, 256, 1, 1,
                            smem, r->stream, args);
    if (err != hipSuccess)
        fprintf(stderr, "hip_llm: attn_prefill_flash launch failed (M=%d hd=%d smem=%zu err=%d)\n",
                M, head_dim, smem, (int)err);
}

/* Bounded online-softmax (flash) decode attention. Grid (n_heads,1), block 256.
 * Reads position from device pointer (graph-capturable). */
static inline void launch_attn_decode_flash(hip_llm_runner *r,
    void *out, void *q, void *key_cache, void *value_cache,
    int n_heads, int n_kv_heads, int head_dim, int kv_dim,
    float scale, int window, int cache_len) {
    int *pos_p = r->d_position;
    void *args[] = { &out, &q, &key_cache, &value_cache,
        &n_heads, &n_kv_heads, &head_dim, &kv_dim,
        &pos_p, &scale, &window, &cache_len };
    size_t smem = ((size_t)(2 * head_dim + 2 * 256)) * sizeof(float);
    LAUNCH(r->fn_attn_decode_flash_f32_devp, n_heads, 1, 1, 256, 1, 1,
           smem, r->stream, args);
}

/* WMMA flash attention for Gemma4 full-attention layers (head_dim 512). Reads the
 * F32 KV cache directly. Grid (n_heads, ceil(M/16)), 1 wave/block, O in LDS. */
static inline void launch_flash_attn_wmma_hd512(hip_llm_runner *r,
    void *out, void *q, void *key_cache, void *value_cache,
    int M, int kv_len, int position_start,
    int n_heads, int n_kv_heads, int head_dim, int kv_dim, float scale) {
    void *args[] = { &out, &q, &key_cache, &value_cache,
        &M, &kv_len, &position_start, &n_heads, &n_kv_heads, &head_dim, &kv_dim, &scale };
    /* smK 16K + smV 16K (f16) + smScore 4*256 f32 = ~36.5 KB (smP overlaps smK). */
    size_t smem = (size_t)(2 * 16 * 512) * 2 + (size_t)(4 * 256) * 4;
    int qblocks = (M + 15) / 16;
    hipError_t err = LAUNCH(r->fn_flash_attn_wmma_hd512_causal, n_heads, qblocks, 1, 128, 1, 1,
                            smem, r->stream, args);
    if (err != hipSuccess)
        fprintf(stderr, "hip_llm: flash_attn_wmma_hd512 launch failed (M=%d smem=%zu err=%d)\n",
                M, smem, (int)err);
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
    if (r->ssm_fused_decode && d_state == 128) {
        /* warp-per-row: 8 warps/block, 8 rows/warp -> blocks = heads*rows/64 */
        int blocks = dt_rank * d_state / 64;
        LAUNCH(r->fn_deltanet_step_warp_f32, blocks, 1, 1, 256, 1, 1, 0, r->stream, args);
    } else {
        LAUNCH(r->fn_deltanet_step_f32, dt_rank, 1, 1, d_state, 1, 1, 0, r->stream, args);
    }
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
    if (d_state == 128) {
        int blocks = dt_rank * d_state / 64;
        LAUNCH(r->fn_deltanet_step_batch_warp_f32, blocks, 1, 1, 256, 1, 1, 0,
               r->stream, args);
    } else {
        LAUNCH(r->fn_deltanet_step_batch_f32, dt_rank, 1, 1, d_state, 1, 1, 0,
               r->stream, args);
    }
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

/* === MoE device-side dispatch launchers (sync-free, graph-capturable) === */
static inline void launch_moe_topk(hip_llm_runner *r) {
    int n = r->n_experts, k = r->n_experts_used;
    void *args[] = { &r->d_router_logits, &n, &k, &r->d_moe_idx, &r->d_moe_w };
    LAUNCH(r->fn_moe_topk_softmax_gpu, 1, 1, 1, 256, 1, 1, 0, r->stream, args);
}
static inline void launch_sigmoid_scalar(hip_llm_runner *r, void *out, void *in) {
    void *args[] = { &out, &in };
    LAUNCH(r->fn_sigmoid_scalar_f32, 1, 1, 1, 1, 1, 1, 0, r->stream, args);
}
static inline void launch_scale_add_dev(hip_llm_runner *r, void *dst, void *src,
                                          void *scale_ptr, int slot, int n) {
    void *args[] = { &dst, &src, &scale_ptr, &slot, &n };
    LAUNCH(r->fn_scale_add_dev_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args);
}
static inline void launch_moe_gather(hip_llm_runner *r, void *dst, void *src,
                                      void *idx, int n_assign, int n_embd) {
    void *args[] = { &dst, &src, &idx, &n_assign, &n_embd };
    long total = (long)n_assign * n_embd;
    LAUNCH(r->fn_moe_gather_rows, (unsigned)((total + 255) / 256), 1, 1, 256, 1, 1, 0, r->stream, args);
}
static inline void launch_moe_scatter_accum(hip_llm_runner *r, void *dst, void *src,
                                             void *idx, void *w, int n_assign, int n_embd) {
    void *args[] = { &dst, &src, &idx, &w, &n_assign, &n_embd };
    long total = (long)n_assign * n_embd;
    LAUNCH(r->fn_moe_scatter_accum, (unsigned)((total + 255) / 256), 1, 1, 256, 1, 1, 0, r->stream, args);
}
static inline void launch_moe_row_scale_add(hip_llm_runner *r, void *dst, void *src,
                                             void *scale, int M, int n_embd) {
    void *args[] = { &dst, &src, &scale, &M, &n_embd };
    long total = (long)M * n_embd;
    LAUNCH(r->fn_moe_row_scale_add, (unsigned)((total + 255) / 256), 1, 1, 256, 1, 1, 0, r->stream, args);
}
/* Fused quantized MoE GEMM: Y[cnt,N] = X[cnt,K] x W[N,K]^T (W quantized in place).
 * Tiles cnt into chunks of <=32 (the in-kernel acc[] cap). type = IQ2_S or IQ3_S. */
static inline void launch_mmq(hip_llm_runner *r, void *Y, void *W, void *X,
                               int cnt, int N, int K, int type) {
    hipFunction_t fn = (type == GGML_TYPE_IQ3_S) ? r->fn_mmq_iq3s_f32 : r->fn_mmq_iq2s_f32;
    for (int t0 = 0; t0 < cnt; t0 += 32) {
        int cc = cnt - t0; if (cc > 32) cc = 32;
        void *Yc = (float *)Y + (size_t)t0 * N;
        void *Xc = (float *)X + (size_t)t0 * K;
        void *args[] = { &Yc, &W, &Xc, &cc, &N, &K };
        LAUNCH(fn, (unsigned)((N + 7) / 8), 1, 1, 256, 1, 1, 0, r->stream, args);
    }
}
/* Fused decode MoE: all-K experts gate+up+silu in one launch, down+accum in one. */
/* n_slots = K (+1 when shg/shu/shd given: last slot = shared expert, Q6_K, and
 * slot 0 zeroes the accumulator). */
static inline void launch_moe_gateup_silu(hip_llm_runner *r, hip_layer *cl, int n_slots,
                                          void *shg, void *shu) {
    int eff = r->expert_ff, n_cols = r->n_embd, n_embd = r->n_embd;
    long long stride = (long long)cl->moe_exp_stride_gu;
    void *args[] = { &r->d_moe_act8, &cl->moe_gate_exps_w, &cl->moe_up_exps_w,
                     &r->d_xb, &eff, &n_cols, &r->d_moe_idx, &stride,
                     &n_slots, &shg, &shu, &r->d_moe_accum, &n_embd };
    LAUNCH(r->fn_moe_gateup_silu_iq2s, (eff + 7) / 8, n_slots, 1, 256, 1, 1, 0, r->stream, args);
}
static inline void launch_moe_down_accum(hip_llm_runner *r, hip_layer *cl, int n_slots,
                                         void *shd) {
    int n_embd = r->n_embd, eff = r->expert_ff;
    long long stride = (long long)cl->moe_exp_stride_d;
    int dtype = (cl->moe_down_exps_type == GGML_TYPE_IQ4_XS) ? 1 : 0;
    void *args[] = { &r->d_moe_accum, &cl->moe_down_exps_w, &r->d_moe_act8,
                     &n_embd, &eff, &r->d_moe_idx, &r->d_moe_w, &stride,
                     &n_slots, &shd, &r->d_shared_scale, &dtype };
    LAUNCH(r->fn_moe_down_accum_iq3s, (n_embd + 7) / 8, n_slots, 1, 256, 1, 1, 0, r->stream, args);
}
/* Expert-indexed DP4A matvec for IQ2_S/IQ3_S; activation pre-quantized to (qs,scale). */
static inline void launch_matvec_expert_dp4a(hip_llm_runner *r, void *dst, void *base,
                                              void *qs, void *scale, int n_rows, int n_cols,
                                              int type, int slot, long long stride) {
    void *args[] = { &dst, &base, &qs, &scale, &n_rows, &n_cols, &r->d_moe_idx, &slot, &stride };
    hipFunction_t fn = (type == GGML_TYPE_IQ3_S) ? r->fn_matvec_iq3_s_expert_dp4a
                                                 : r->fn_matvec_iq2_s_expert_dp4a;
    LAUNCH(fn, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args);
}
/* Expert-indexed matvec: base resolved on-device from r->d_moe_idx[slot]. */
static inline void launch_matvec_expert_auto(hip_llm_runner *r, void *dst, void *base,
                                              void *x, int n_rows, int n_cols, int type,
                                              int slot, long long stride) {
    void *args[] = { &dst, &base, &x, &n_rows, &n_cols, &r->d_moe_idx, &slot, &stride };
    hipFunction_t fn;
    unsigned int smem = 0;
    if (r->lds_grid && type == GGML_TYPE_IQ2_S) {
        fn = r->fn_matvec_iq2_s_expert_lds_f32; smem = 1024 * sizeof(unsigned long long);
    } else if (r->lds_grid && type == GGML_TYPE_IQ3_S) {
        fn = r->fn_matvec_iq3_s_expert_lds_f32; smem = 512 * sizeof(unsigned int);
    } else switch (type) {
        case GGML_TYPE_IQ2_S:  fn = r->fn_matvec_iq2_s_expert_f32;  break;
        case GGML_TYPE_IQ3_S:  fn = r->fn_matvec_iq3_s_expert_f32;  break;
        case GGML_TYPE_IQ4_XS: fn = r->fn_matvec_iq4_xs_expert_f32; break;
        default: fn = r->fn_matvec_iq2_s_expert_f32; break; /* gated by moe_dev_dispatch_ok */
    }
    LAUNCH(fn, (n_rows + 7) / 8, 1, 1, 256, 1, 1, smem, r->stream, args);
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

    /* Gemma4 scales token embeddings by sqrt(n_embd). This must run for EVERY token
     * (hip_llm_forward_logits also does it, but this per-token prefill entry must
     * too, or the residual stream uses an unscaled embedding for all but the last). */
    if (r->is_gemma4) {
        int n = n_embd;
        float scale = r->embd_scale;
        void *a[] = { &r->d_x, &scale, &n };
        LAUNCH(r->fn_scale_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a);
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
/* MoE FFN for one token: reads the pre-normed input in r->d_xb and overwrites
 * r->d_xb with the MoE output (router top-K experts + shared expert, weighted).
 * Caller adds the residual. Shared by the decode layer body and the batched
 * prefill per-row MoE loop. */
static void forward_moe_ffn(hip_llm_runner *r, hip_layer *cl) {
    int n_embd = r->n_embd;
    int n_experts = r->n_experts;
    int n_experts_used = r->n_experts_used;
    int expert_ff = r->expert_ff;
    int shared_expert_ff = r->shared_expert_ff;

  /* Fully-fused MoE decode path: router+topk+sgate (1 launch, last-block topk),
   * gateup incl. shared expert slot + accum zero (1), down incl. shared (1). */
  if (r->moe_dev_dispatch_ok && r->moe_fused_decode && r->n_experts <= 256 &&
      cl->moe_gate_exps_type == GGML_TYPE_IQ2_S &&
      cl->moe_up_exps_type   == GGML_TYPE_IQ2_S &&
      (cl->moe_down_exps_type == GGML_TYPE_IQ3_S || cl->moe_down_exps_type == GGML_TYPE_IQ4_XS) &&
      shared_expert_ff == expert_ff &&
      cl->moe_shared_gate_type == GGML_TYPE_Q6_K &&
      cl->moe_shared_up_type   == GGML_TYPE_Q6_K &&
      cl->moe_shared_down_type == GGML_TYPE_Q6_K) {
      int ne_ = n_experts, K_ = n_experts_used, nc_ = n_embd;
      void *a[] = { &cl->moe_gate_w_bf16, &cl->moe_shared_gate_w_bf16, &r->d_xb,
                    &ne_, &K_, &nc_, &r->d_router_logits, &r->d_moe_idx, &r->d_moe_w,
                    &r->d_shared_scale, &r->d_router_counter };
      LAUNCH(r->fn_moe_router_fused, ne_ + 1, 1, 1, 256, 1, 1, 0, r->stream, a);
      launch_moe_gateup_silu(r, cl, n_experts_used + 1,
                             cl->moe_shared_ffn_gate_w, cl->moe_shared_ffn_up_w);
      launch_moe_down_accum(r, cl, n_experts_used + 1, cl->moe_shared_ffn_down_w);
      return;
  }

  launch_matvec_f32(r, r->d_router_logits, cl->moe_gate_w, r->d_xb,
                   cl->moe_gate_rows, cl->moe_gate_cols);

  if (r->moe_dev_dispatch_ok) {
    launch_moe_topk(r);

    /* Fused all-experts path: 2 launches instead of ~40 small dependent ones.
     * Requires gate/up = IQ2_S and down = IQ3_S (covers 37/40 layers). */
    if (r->moe_fused_decode &&
        cl->moe_gate_exps_type == GGML_TYPE_IQ2_S &&
        cl->moe_up_exps_type   == GGML_TYPE_IQ2_S &&
        cl->moe_down_exps_type == GGML_TYPE_IQ3_S) {
        int sh_fold = shared_expert_ff == expert_ff &&
            cl->moe_shared_gate_type == GGML_TYPE_Q6_K &&
            cl->moe_shared_up_type   == GGML_TYPE_Q6_K &&
            cl->moe_shared_down_type == GGML_TYPE_Q6_K;
        if (sh_fold) {
            launch_matvec_f32(r, r->d_shared_scale, cl->moe_shared_gate_w, r->d_xb, 1, n_embd);
            launch_sigmoid_scalar(r, r->d_shared_scale, r->d_shared_scale);
            launch_moe_gateup_silu(r, cl, n_experts_used + 1,
                                   cl->moe_shared_ffn_gate_w, cl->moe_shared_ffn_up_w);
            launch_moe_down_accum(r, cl, n_experts_used + 1, cl->moe_shared_ffn_down_w);
            return;
        }
        hipMemsetAsync(r->d_moe_accum, 0, n_embd * sizeof(float), r->stream);
        launch_moe_gateup_silu(r, cl, n_experts_used, NULL, NULL);
        launch_moe_down_accum(r, cl, n_experts_used, NULL);
        goto shared_expert;
    }

    hipMemsetAsync(r->d_moe_accum, 0, n_embd * sizeof(float), r->stream);
    int gu_dp4a = r->decode_dp4a &&
        (cl->moe_gate_exps_type == GGML_TYPE_IQ2_S || cl->moe_gate_exps_type == GGML_TYPE_IQ3_S) &&
        (cl->moe_up_exps_type   == GGML_TYPE_IQ2_S || cl->moe_up_exps_type   == GGML_TYPE_IQ3_S);
    int dn_dp4a = r->decode_dp4a &&
        (cl->moe_down_exps_type == GGML_TYPE_IQ2_S || cl->moe_down_exps_type == GGML_TYPE_IQ3_S);
    if (gu_dp4a)
        launch_quantize_q8(r, r->d_xb, n_embd, r->d_act_q8, r->d_act_scale);

    for (int e = 0; e < n_experts_used; e++) {
        if (gu_dp4a) {
            launch_matvec_expert_dp4a(r, r->d_gate, cl->moe_gate_exps_w, r->d_act_q8, r->d_act_scale,
                              cl->moe_exp_rows_gu, cl->moe_exp_cols_gu,
                              cl->moe_gate_exps_type, e, (long long)cl->moe_exp_stride_gu);
            launch_matvec_expert_dp4a(r, r->d_up, cl->moe_up_exps_w, r->d_act_q8, r->d_act_scale,
                              cl->moe_exp_rows_gu, cl->moe_exp_cols_gu,
                              cl->moe_up_exps_type, e, (long long)cl->moe_exp_stride_gu);
        } else {
            launch_matvec_expert_auto(r, r->d_gate, cl->moe_gate_exps_w, r->d_xb,
                              cl->moe_exp_rows_gu, cl->moe_exp_cols_gu,
                              cl->moe_gate_exps_type, e, (long long)cl->moe_exp_stride_gu);
            launch_matvec_expert_auto(r, r->d_up, cl->moe_up_exps_w, r->d_xb,
                              cl->moe_exp_rows_gu, cl->moe_exp_cols_gu,
                              cl->moe_up_exps_type, e, (long long)cl->moe_exp_stride_gu);
        }
        launch_silu_mul(r, r->d_gate, r->d_up, expert_ff);
        if (dn_dp4a) {
            launch_quantize_q8(r, r->d_gate, expert_ff, r->d_act_q8_b, r->d_act_scale_b);
            launch_matvec_expert_dp4a(r, r->d_xb2, cl->moe_down_exps_w, r->d_act_q8_b, r->d_act_scale_b,
                              cl->moe_exp_rows_d, cl->moe_exp_cols_d,
                              cl->moe_down_exps_type, e, (long long)cl->moe_exp_stride_d);
        } else {
            launch_matvec_expert_auto(r, r->d_xb2, cl->moe_down_exps_w, r->d_gate,
                              cl->moe_exp_rows_d, cl->moe_exp_cols_d,
                              cl->moe_down_exps_type, e, (long long)cl->moe_exp_stride_d);
        }
        launch_scale_add_dev(r, r->d_moe_accum, r->d_xb2, r->d_moe_w, e, n_embd);
    }

shared_expert:
    {
        launch_matvec_f32(r, r->d_shared_scale, cl->moe_shared_gate_w, r->d_xb, 1, n_embd);
        launch_sigmoid_scalar(r, r->d_shared_scale, r->d_shared_scale);
        if (r->moe_fused_decode &&
            cl->moe_shared_gate_type == GGML_TYPE_Q6_K &&
            cl->moe_shared_up_type   == GGML_TYPE_Q6_K &&
            cl->moe_shared_down_type == GGML_TYPE_Q6_K) {
            /* Fused: 5 launches -> 2 (gate+up+silu, down+scale-add). */
            int eff = shared_expert_ff;
            void *a1[] = { &r->d_gate, &cl->moe_shared_ffn_gate_w, &cl->moe_shared_ffn_up_w,
                           &r->d_xb, &eff, &n_embd };
            LAUNCH(r->fn_shexp_gateup_silu_q6k, (eff + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, a1);
            void *a2[] = { &r->d_moe_accum, &cl->moe_shared_ffn_down_w, &r->d_gate,
                           &n_embd, &eff, &r->d_shared_scale };
            LAUNCH(r->fn_shexp_down_accum_q6k, (n_embd + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, a2);
        } else {
            launch_matvec_auto(r, r->d_gate, cl->moe_shared_ffn_gate_w, r->d_xb,
                              cl->moe_shared_gate_rows, cl->moe_shared_gate_cols, cl->moe_shared_gate_type);
            launch_matvec_auto(r, r->d_up, cl->moe_shared_ffn_up_w, r->d_xb,
                              cl->moe_shared_up_rows, cl->moe_shared_up_cols, cl->moe_shared_up_type);
            launch_silu_mul(r, r->d_gate, r->d_up, shared_expert_ff);
            launch_matvec_auto(r, r->d_xb2, cl->moe_shared_ffn_down_w, r->d_gate,
                              cl->moe_shared_down_rows, cl->moe_shared_down_cols, cl->moe_shared_down_type);
            launch_scale_add_dev(r, r->d_moe_accum, r->d_xb2, r->d_shared_scale, 0, n_embd);
        }
    }
  } else {
    hipDeviceSynchronize();
    hipMemcpy(r->h_router_logits, r->d_router_logits, n_experts * sizeof(float), hipMemcpyDeviceToHost);
    int top_k_idx[64]; float top_k_weights[64];
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
  }
}

/* Batched token-grouped MoE FFN for M tokens (prefill). Input: r->d_xnorm_batch
 * [M, n_embd] (pre-normed). Adds the MoE output into r->d_x_batch (residual).
 * Experts are grouped by token: router GEMM -> host top-K -> gather by expert ->
 * per-expert dequant+GEMM (gate/up/silu/down) -> scatter-accumulate; shared
 * expert is dense over all M. Returns 0 on success. */
static int forward_moe_ffn_batched(hip_llm_runner *r, hip_layer *cl, int M) {
    int n_embd = r->n_embd, ne = r->n_experts, K = r->n_experts_used;
    int eff = r->expert_ff, sff = r->shared_expert_ff;
    if (ne > 1024) return -1;  /* cursor[] cap */
    /* 1. Router GEMM: [M,ne] = xnorm[M,n_embd] x Wg[ne,n_embd] (Wg is F32 -> bf16). */
    launch_pack_bf16_from_f32(r, r->d_xnorm_batch_bf16_moe, r->d_xnorm_batch, M * n_embd);
    launch_pack_bf16_from_f32(r, r->d_router_w_bf16, cl->moe_gate_w, ne * n_embd);
    if (gemm_run_bf16_w(r, r->d_router_logits_batch, r->d_router_w_bf16,
                           r->d_xnorm_batch_bf16_moe, M, ne, n_embd, r->stream) != 0) return -1;

    /* 2. Top-K + softmax per token, group assignments by expert.
     * GPU grouping (no host sync) for supported quant types; host fallback otherwise. */
    int total = M * K;
    int *offs = r->h_moe_offsets;
    int gu_is_iq2s = cl->moe_gate_exps_type == GGML_TYPE_IQ2_S;
    int gu_is_iq1s = cl->moe_gate_exps_type == GGML_TYPE_IQ1_S;
    int gpu_group = r->gemm_own && r->d_tok_idx && cl->moe_down_exps_type == GGML_TYPE_IQ3_S &&
        ((gu_is_iq2s && cl->moe_up_exps_type == GGML_TYPE_IQ2_S) ||
         (gu_is_iq1s && cl->moe_up_exps_type == GGML_TYPE_IQ1_S));
    int _gpu_ok = r->gemm_own && r->d_tok_idx && cl->moe_down_exps_type == GGML_TYPE_IQ3_XXS &&
        ((cl->moe_gate_exps_type == GGML_TYPE_IQ1_S && cl->moe_up_exps_type == GGML_TYPE_IQ1_S) ||
         (cl->moe_gate_exps_type == GGML_TYPE_IQ2_XXS && cl->moe_up_exps_type == GGML_TYPE_IQ2_XXS));
    if (gpu_group || _gpu_ok) {
        gpu_group = 1;
        { void *a[] = { &r->d_router_logits_batch, &ne, &K, &r->d_tok_idx, &r->d_tok_w };
          LAUNCH(r->fn_moe_topk_batch, M, 1, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_tok_idx, &M, &K, &ne, &r->d_moe_offs, &r->d_cursor };
          LAUNCH(r->fn_moe_count_offs, 1, 1, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_tok_idx, &r->d_tok_w, &M, &K, &r->d_cursor,
                        &r->d_moe_gather_src, &r->d_moe_gather_w };
          LAUNCH(r->fn_moe_fill_gather, (M * K + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a); }
        hipMemcpyAsync(r->h_moe_offsets, r->d_moe_offs, (size_t)(ne + 1) * sizeof(int),
                       hipMemcpyDeviceToHost, r->stream);
        hipStreamSynchronize(r->stream);

    } else {
        hipMemcpyAsync(r->h_router_batch, r->d_router_logits_batch,
                       (size_t)M * ne * sizeof(float), hipMemcpyDeviceToHost, r->stream);
        hipStreamSynchronize(r->stream);
        for (int e = 0; e <= ne; e++) offs[e] = 0;
        for (int m = 0; m < M; m++) {
            moe_topk_softmax(r->h_router_batch + (size_t)m * ne, ne, K,
                             r->h_moe_tok_idx + (size_t)m * K, r->h_moe_tok_w + (size_t)m * K);
            for (int k = 0; k < K; k++) offs[r->h_moe_tok_idx[(size_t)m * K + k] + 1]++;
        }
        for (int e = 0; e < ne; e++) offs[e + 1] += offs[e];
        {
            int cursor[1024];
            for (int e = 0; e < ne; e++) cursor[e] = offs[e];
            for (int m = 0; m < M; m++)
                for (int k = 0; k < K; k++) {
                    int e = r->h_moe_tok_idx[(size_t)m * K + k];
                    int p = cursor[e]++;
                    r->h_moe_gather_src[p] = m;
                    r->h_moe_gather_w[p]   = r->h_moe_tok_w[(size_t)m * K + k];
                }
        }
        hipMemcpyAsync(r->d_moe_gather_src, r->h_moe_gather_src, (size_t)total * sizeof(int),
                       hipMemcpyHostToDevice, r->stream);
        hipMemcpyAsync(r->d_moe_gather_w, r->h_moe_gather_w, (size_t)total * sizeof(float),
                       hipMemcpyHostToDevice, r->stream);
    }

    /* 3. Gather activations grouped by expert (f32, fed straight to mmq). */
    launch_moe_gather(r, r->d_moe_gather_in, r->d_xnorm_batch, r->d_moe_gather_src, total, n_embd);
    hipMemsetAsync(r->d_moe_out_batch, 0, (size_t)M * n_embd * sizeof(float), r->stream);

    /* 4. Per-expert GEMMs.
     * own-GEMM path: dequant the expert weight to bf16 once and run the WMMA
     * GEMM — far faster than scalar mmq, which re-reads x per output row
     * (mmq measured 65% of prefill). mmq remains the blaslt-build fallback. */
    int use_wmma_exp = r->gemm_own;
    /* Grouped path (own GEMM): per weight type, ONE all-expert dequant + ONE grouped
     * GEMM (blockIdx.z=expert). ~8 launches/layer vs ~1500 in the per-expert loop. */
    if (use_wmma_exp && r->d_expw_bf16 &&
        cl->moe_gate_exps_type == GGML_TYPE_IQ2_S &&
        cl->moe_up_exps_type   == GGML_TYPE_IQ2_S &&
        cl->moe_down_exps_type == GGML_TYPE_IQ3_S) {
        launch_pack_bf16_from_f32(r, r->d_moe_gather_in_bf16, r->d_moe_gather_in, total * n_embd);
        if (!gpu_group)
            hipMemcpyAsync(r->d_moe_offs, offs, (size_t)(ne + 1) * sizeof(int),
                           hipMemcpyHostToDevice, r->stream);
        unsigned mtiles = (unsigned)((M + 127) / 128); /* covers max per-expert cnt */
        long long sgu = (long long)cl->moe_exp_stride_gu;
        long long sdn = (long long)cl->moe_exp_stride_d;
        int _bm = r->moe_iq2_bm;
        /* gate */
        { void *a[] = { &r->d_expw_bf16, &cl->moe_gate_exps_w, &r->d_moe_offs, &eff, &n_embd, &sgu, &_bm };
          LAUNCH(r->fn_dequant_iq2s_all, (unsigned)((eff + 7) / 8), ne, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_moe_eg, &r->d_expw_bf16, &r->d_moe_gather_in_bf16, &r->d_moe_offs, &eff, &n_embd };
          LAUNCH(r->fn_gemm_bf16_grouped, (unsigned)((eff + 127) / 128), mtiles, ne, 256, 1, 1, 0, r->stream, a); }
        /* up */
        { void *a[] = { &r->d_expw_bf16, &cl->moe_up_exps_w, &r->d_moe_offs, &eff, &n_embd, &sgu, &_bm };
          LAUNCH(r->fn_dequant_iq2s_all, (unsigned)((eff + 7) / 8), ne, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_moe_eu, &r->d_expw_bf16, &r->d_moe_gather_in_bf16, &r->d_moe_offs, &eff, &n_embd };
          LAUNCH(r->fn_gemm_bf16_grouped, (unsigned)((eff + 127) / 128), mtiles, ne, 256, 1, 1, 0, r->stream, a); }
         launch_silu_mul(r, r->d_moe_eg, r->d_moe_eu, total * eff);
        launch_pack_bf16_from_f32(r, r->d_moe_esilu_bf16, r->d_moe_eg, total * eff);
        /* down */
        { void *a[] = { &r->d_expw_bf16, &cl->moe_down_exps_w, &r->d_moe_offs, &n_embd, &eff, &sdn, &_bm };
          LAUNCH(r->fn_dequant_iq3s_all, (unsigned)((n_embd + 7) / 8), ne, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_moe_eout, &r->d_expw_bf16, &r->d_moe_esilu_bf16, &r->d_moe_offs, &n_embd, &eff };
          LAUNCH(r->fn_gemm_bf16_grouped, (unsigned)((n_embd + 127) / 128), mtiles, ne, 256, 1, 1, 0, r->stream, a); }
        goto experts_done;
    }
    /* INT8 WMMA grouped path: quantize activations to INT8, dequant weights to INT8,
     * then INT8 WMMA GEMM. 2x less BW than BF16 path. Gate: LLM_MOE_INT8=1 */
    if (r->moe_int8 && r->d_act_int8 &&
        cl->moe_gate_exps_type == GGML_TYPE_IQ2_S &&
        cl->moe_up_exps_type   == GGML_TYPE_IQ2_S &&
        cl->moe_down_exps_type == GGML_TYPE_IQ3_S) {
        if (!gpu_group)
            hipMemcpyAsync(r->d_moe_offs, offs, (size_t)(ne + 1) * sizeof(int),
                           hipMemcpyHostToDevice, r->stream);
        unsigned mtiles = (unsigned)((M + 127) / 128);
        long long sgu = (long long)cl->moe_exp_stride_gu;
        long long sdn = (long long)cl->moe_exp_stride_d;
        /* quantize activations: F32 gather_in -> INT8 act */
        { int _nq = total * n_embd;
          void *a[] = { &r->d_act_int8, &r->d_moe_gather_in, &_nq };
          LAUNCH(r->fn_quantize_f32_act_to_int8, (unsigned)((_nq + 255) / 256), 1, 1, 256, 1, 1, 0, r->stream, a); }
        /* gate: dequant IQ2_S->INT8 + INT8 grouped GEMM */
        { void *a[] = { &r->d_expw_int8, &cl->moe_gate_exps_w, &r->d_moe_offs, &eff, &n_embd, &sgu };
          LAUNCH(r->fn_dequant_iq2s_all_int8, (unsigned)((eff + 7) / 8), ne, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_moe_eg, &r->d_expw_int8, &r->d_act_int8, &r->d_moe_offs, &eff, &n_embd };
          LAUNCH(r->fn_gemm_int8_grouped, (unsigned)((eff + 127) / 128), mtiles, ne, 256, 1, 1, 0, r->stream, a); }
        /* up: reuses same INT8 activations, dequant weights again */
        { void *a[] = { &r->d_expw_int8, &cl->moe_up_exps_w, &r->d_moe_offs, &eff, &n_embd, &sgu };
          LAUNCH(r->fn_dequant_iq2s_all_int8, (unsigned)((eff + 7) / 8), ne, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_moe_eu, &r->d_expw_int8, &r->d_act_int8, &r->d_moe_offs, &eff, &n_embd };
          LAUNCH(r->fn_gemm_int8_grouped, (unsigned)((eff + 127) / 128), mtiles, ne, 256, 1, 1, 0, r->stream, a); }
        launch_silu_mul(r, r->d_moe_eg, r->d_moe_eu, total * eff);
        /* quantize silu output -> INT8 for down */
        { int _nq = total * eff;
          void *a[] = { &r->d_act_int8, &r->d_moe_eg, &_nq };
          LAUNCH(r->fn_quantize_f32_act_to_int8, (unsigned)((_nq + 255) / 256), 1, 1, 256, 1, 1, 0, r->stream, a); }
        /* down: dequant IQ3_S->INT8 + INT8 grouped GEMM */
        { void *a[] = { &r->d_expw_int8, &cl->moe_down_exps_w, &r->d_moe_offs, &n_embd, &eff, &sdn };
          LAUNCH(r->fn_dequant_iq3s_all_int8, (unsigned)((n_embd + 7) / 8), ne, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_moe_eout, &r->d_expw_int8, &r->d_act_int8, &r->d_moe_offs, &n_embd, &eff };
          LAUNCH(r->fn_gemm_int8_grouped, (unsigned)((n_embd + 127) / 128), mtiles, ne, 256, 1, 1, 0, r->stream, a); }
        goto experts_done;
    }
    /* Grouped path for IQ2_XXS gate/up + IQ3_XXS down (IQ2_M model). */
    if (use_wmma_exp && r->d_expw_bf16 &&
        cl->moe_gate_exps_type == GGML_TYPE_IQ2_XXS &&
        cl->moe_up_exps_type   == GGML_TYPE_IQ2_XXS &&
        cl->moe_down_exps_type == GGML_TYPE_IQ3_XXS) {
        launch_pack_bf16_from_f32(r, r->d_moe_gather_in_bf16, r->d_moe_gather_in, total * n_embd);
        if (!gpu_group)
            hipMemcpyAsync(r->d_moe_offs, offs, (size_t)(ne + 1) * sizeof(int),
                           hipMemcpyHostToDevice, r->stream);
        unsigned mtiles = (unsigned)((M + 127) / 128);
        long long sgu = (long long)cl->moe_exp_stride_gu;
        long long sdn = (long long)cl->moe_exp_stride_d;
        int _bm2 = r->moe_iq2_bm;
        /* gate */
        { void *a[] = { &r->d_expw_bf16, &cl->moe_gate_exps_w, &r->d_moe_offs, &eff, &n_embd, &sgu, &_bm2 };
          LAUNCH(r->fn_dequant_iq2_xxs_all, (unsigned)((eff + 7) / 8), ne, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_moe_eg, &r->d_expw_bf16, &r->d_moe_gather_in_bf16, &r->d_moe_offs, &eff, &n_embd };
          LAUNCH(r->fn_gemm_bf16_grouped, (unsigned)((eff + 127) / 128), mtiles, ne, 256, 1, 1, 0, r->stream, a); }
        /* up */
        { void *a[] = { &r->d_expw_bf16, &cl->moe_up_exps_w, &r->d_moe_offs, &eff, &n_embd, &sgu, &_bm2 };
          LAUNCH(r->fn_dequant_iq2_xxs_all, (unsigned)((eff + 7) / 8), ne, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_moe_eu, &r->d_expw_bf16, &r->d_moe_gather_in_bf16, &r->d_moe_offs, &eff, &n_embd };
          LAUNCH(r->fn_gemm_bf16_grouped, (unsigned)((eff + 127) / 128), mtiles, ne, 256, 1, 1, 0, r->stream, a); }
        launch_silu_mul(r, r->d_moe_eg, r->d_moe_eu, total * eff);
        launch_pack_bf16_from_f32(r, r->d_moe_esilu_bf16, r->d_moe_eg, total * eff);
        /* down */
        { void *a[] = { &r->d_expw_bf16, &cl->moe_down_exps_w, &r->d_moe_offs, &n_embd, &eff, &sdn, &_bm2 };
          LAUNCH(r->fn_dequant_iq3_xxs_all, (unsigned)((n_embd + 7) / 8), ne, 1, 256, 1, 1, 0, r->stream, a); }
        { void *a[] = { &r->d_moe_eout, &r->d_expw_bf16, &r->d_moe_esilu_bf16, &r->d_moe_offs, &n_embd, &eff };
          LAUNCH(r->fn_gemm_bf16_grouped, (unsigned)((n_embd + 127) / 128), mtiles, ne, 256, 1, 1, 0, r->stream, a); }
        goto experts_done;
    }
    if (use_wmma_exp)
        launch_pack_bf16_from_f32(r, r->d_moe_gather_in_bf16, r->d_moe_gather_in, total * n_embd);
    for (int e = 0; e < ne; e++) {
        int cnt = offs[e + 1] - offs[e];
        if (cnt == 0) continue;
        size_t off = (size_t)offs[e];
        float *xin = (float *)r->d_moe_gather_in + off * n_embd;
        void *gate_w = (char *)cl->moe_gate_exps_w + (size_t)e * cl->moe_exp_stride_gu;
        void *up_w   = (char *)cl->moe_up_exps_w   + (size_t)e * cl->moe_exp_stride_gu;
        void *down_w = (char *)cl->moe_down_exps_w  + (size_t)e * cl->moe_exp_stride_d;
        if (use_wmma_exp) {
            void *xin_bf16 = (char *)r->d_moe_gather_in_bf16 + off * n_embd * 2;
            void *gw = get_bf16_weight(r, gate_w, NULL, cl->moe_gate_exps_type,
                                       cl->moe_exp_rows_gu, cl->moe_exp_cols_gu);
            if (!gw) return -1;
            if (gemm_run_bf16_w(r, (float *)r->d_moe_eg + off * eff, gw, xin_bf16,
                                cnt, eff, n_embd, r->stream) != 0) return -1;
            void *uw = get_bf16_weight(r, up_w, NULL, cl->moe_up_exps_type,
                                       cl->moe_exp_rows_gu, cl->moe_exp_cols_gu);
            if (!uw) return -1;
            if (gemm_run_bf16_w(r, (float *)r->d_moe_eu + off * eff, uw, xin_bf16,
                                cnt, eff, n_embd, r->stream) != 0) return -1;
            launch_silu_mul(r, (float *)r->d_moe_eg + off * eff, (float *)r->d_moe_eu + off * eff, cnt * eff);
            launch_pack_bf16_from_f32(r, (char *)r->d_moe_esilu_bf16 + off * eff * 2,
                                      (float *)r->d_moe_eg + off * eff, cnt * eff);
            void *dw = get_bf16_weight(r, down_w, NULL, cl->moe_down_exps_type,
                                       cl->moe_exp_rows_d, cl->moe_exp_cols_d);
            if (!dw) return -1;
            if (gemm_run_bf16_w(r, (float *)r->d_moe_eout + off * n_embd, dw,
                                (char *)r->d_moe_esilu_bf16 + off * eff * 2,
                                cnt, n_embd, eff, r->stream) != 0) return -1;
            continue;
        }
        launch_mmq(r, (float *)r->d_moe_eg + off * eff, gate_w, xin, cnt, eff, n_embd, cl->moe_gate_exps_type);
        launch_mmq(r, (float *)r->d_moe_eu + off * eff, up_w,   xin, cnt, eff, n_embd, cl->moe_up_exps_type);
        launch_silu_mul(r, (float *)r->d_moe_eg + off * eff, (float *)r->d_moe_eu + off * eff, cnt * eff);
        float *silu = (float *)r->d_moe_eg + off * eff;
        if (cl->moe_down_exps_type == GGML_TYPE_IQ2_S || cl->moe_down_exps_type == GGML_TYPE_IQ3_S) {
            launch_mmq(r, (float *)r->d_moe_eout + off * n_embd, down_w, silu, cnt, n_embd, eff, cl->moe_down_exps_type);
        } else {
            launch_pack_bf16_from_f32(r, (char *)r->d_moe_esilu_bf16 + off * eff * 2, silu, cnt * eff);
            void *dw = get_bf16_weight(r, down_w, NULL, cl->moe_down_exps_type, cl->moe_exp_rows_d, cl->moe_exp_cols_d);
            if (!dw) return -1;
            if (gemm_run_bf16_w(r, (float *)r->d_moe_eout + off * n_embd, dw,
                                   (char *)r->d_moe_esilu_bf16 + off * eff * 2, cnt, n_embd, eff, r->stream) != 0) return -1;
        }
    }

experts_done:
    /* 5. Scatter + weighted accumulate into d_moe_out_batch. */
    launch_moe_scatter_accum(r, r->d_moe_out_batch, r->d_moe_eout, r->d_moe_gather_src,
                             r->d_moe_gather_w, total, n_embd);

    /* 6. Shared expert (dense over all M): gate logit -> sigmoid -> gate/up/silu/down -> row-scale add. */
    launch_pack_bf16_from_f32(r, r->d_router_w_bf16, cl->moe_shared_gate_w, n_embd);
    if (gemm_run_bf16_w(r, r->d_shared_scale_batch, r->d_router_w_bf16,
                           r->d_xnorm_batch_bf16_moe, M, 1, n_embd, r->stream) != 0) return -1;
    launch_sigmoid_inplace(r, r->d_shared_scale_batch, M);
    {
        void *sg = get_bf16_weight(r, cl->moe_shared_ffn_gate_w, NULL, cl->moe_shared_gate_type,
                                   cl->moe_shared_gate_rows, cl->moe_shared_gate_cols);
        if (!sg) return -1;
        if (gemm_run_bf16_w(r, r->d_moe_eg, sg, r->d_xnorm_batch_bf16_moe, M, sff, n_embd, r->stream) != 0) return -1;
        void *su = get_bf16_weight(r, cl->moe_shared_ffn_up_w, NULL, cl->moe_shared_up_type,
                                   cl->moe_shared_up_rows, cl->moe_shared_up_cols);
        if (!su) return -1;
        if (gemm_run_bf16_w(r, r->d_moe_eu, su, r->d_xnorm_batch_bf16_moe, M, sff, n_embd, r->stream) != 0) return -1;
        launch_silu_mul(r, r->d_moe_eg, r->d_moe_eu, M * sff);
        launch_pack_bf16_from_f32(r, r->d_moe_esilu_bf16, r->d_moe_eg, M * sff);
        void *sd = get_bf16_weight(r, cl->moe_shared_ffn_down_w, NULL, cl->moe_shared_down_type,
                                   cl->moe_shared_down_rows, cl->moe_shared_down_cols);
        if (!sd) return -1;
        if (gemm_run_bf16_w(r, r->d_moe_eout, sd, r->d_moe_esilu_bf16, M, n_embd, sff, r->stream) != 0) return -1;
        launch_moe_row_scale_add(r, r->d_moe_out_batch, r->d_moe_eout, r->d_shared_scale_batch, M, n_embd);
    }

    /* 7. Residual: x_batch += moe_out. */
    launch_add(r, r->d_x_batch, r->d_moe_out_batch, M * n_embd);
    return 0;
}

static void forward_one_layer(hip_llm_runner *r, int l) {
    int n_embd     = r->n_embd;
    int n_heads    = r->n_heads;
    int n_kv_heads = r->n_kv_heads;
    int head_dim   = r->head_dim;
    int kv_dim     = n_kv_heads * head_dim;
    int n_ff       = r->n_ff;
    float eps      = r->rms_norm_eps;
    hip_layer *cl  = &r->layers[l];

    /* Pre-attention RMSNorm; fused with pending MoE residual from previous layer. */
    if (r->moe_add_pending) {
        void *a[] = { &r->d_x, &r->d_moe_accum, &r->d_xb, &cl->attn_norm_w, &n_embd, &eps };
        LAUNCH(r->fn_res_rmsnorm_f32, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), r->stream, a);
        r->moe_add_pending = 0;
    } else {
        launch_rmsnorm(r, r->d_xb, r->d_x, cl->attn_norm_w, n_embd, eps);
    }

    /* === Gemma4 layer === */
    if (r->is_gemma4) {
        int hd = cl->local_head_dim;
        int local_kv_heads = cl->local_kv_heads;
        int local_kv_dim = local_kv_heads * hd;
        int local_q_dim = n_heads * hd;
        int local_gqa = n_heads / local_kv_heads;
        int kv_src = (cl->shared_kv_source >= 0) ? cl->shared_kv_source : l;
        /* Value-less (global/full-attn) layers: V = K (config attention_k_eq_v). */
        int v_eq_k = (cl->shared_kv_source < 0) && (cl->attn_v_rows <= 0);

        /* Q projection */
        launch_matvec_auto(r, r->d_q, cl->attn_q_w, r->d_xb,
                          cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);

        /* K/V projections (skip if sharing KV; skip V when V==K) */
        if (cl->shared_kv_source < 0) {
            launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb,
                              cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
            if (!v_eq_k)
                launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb,
                                  cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);
        }

        /* V = K (raw k_proj, BEFORE k_norm) for value-less layers. matches llama.cpp:
         * Vcur = Kcur (pre-k_norm), then a weightless per-head RMS norm; K instead gets
         * the learned k_norm + RoPE. So copy here, before k_norm runs. */
        if (v_eq_k)
            hipMemcpyAsync(r->d_v, r->d_k, (size_t)local_kv_dim * sizeof(float),
                           hipMemcpyDeviceToDevice, r->stream);

        /* QK norm (per-layer head_dim) — applies to K only (not the V copy). */
        if (cl->has_qk_norm) {
            if (cl->attn_q_norm_w) launch_qknorm(r, r->d_q, cl->attn_q_norm_w, n_heads, hd, eps);
            if (cl->attn_k_norm_w) launch_qknorm(r, r->d_k, cl->attn_k_norm_w, local_kv_heads, hd, eps);
        }

        /* V raw RMSNorm (weightless, per head) — applied to V for every KV-bearing
         * layer (matches llama.cpp Vcur = ggml_rms_norm(Vcur, eps)). V is not RoPE'd.
         * Use the batched kernel (n_rows=1): it handles head_dim>256 (full-attn hd=512). */
        if (cl->shared_kv_source < 0)
            launch_raw_rmsnorm_heads_batch(r, r->d_v, local_kv_heads, hd, 1, local_kv_dim, eps);

        /* RoPE: SWA uses base_swa (no freq_factors); full-attn uses base + proportional
         * freq_factors. V is never RoPE'd. */
        if (cl->is_swa) {
            launch_rope_devp(r, r->d_q, n_heads, hd, r->rope_freq_base_swa);
            if (cl->shared_kv_source < 0)
                launch_rope_devp(r, r->d_k, local_kv_heads, hd, r->rope_freq_base_swa);
        } else {
            launch_rope_devp_ff(r, r->d_q, n_heads, hd, r->rope_freq_base, r->d_rope_freq_factors);
            if (cl->shared_kv_source < 0)
                launch_rope_devp_ff(r, r->d_k, local_kv_heads, hd, r->rope_freq_base, r->d_rope_freq_factors);
        }

        /* KV cache store. SWA uses a circular cache (slot = pos % swa_cache_len);
         * full-attn uses a linear cache (cache_len = max_seq_len => no wrap). */
        int dec_cache_len = cl->is_swa ? r->swa_cache_len : r->max_seq_len;
        if (cl->shared_kv_source < 0) {
            launch_kv_store_devp(r, r->d_key_cache[l], r->d_value_cache[l],
                                 r->d_k, r->d_v, local_kv_dim, dec_cache_len);
        }

        /* Attention: scale=1.0 (QK norms handle scaling). Bounded online-softmax
         * kernel works for both SWA (window>0) and full (window=0) and never
         * exceeds LDS regardless of -s. */
        {
            float scale = 1.0f;
            int win = cl->is_swa ? r->swa_window_size : 0;
            launch_attn_decode_flash(r, r->d_xb2, r->d_q,
                                     r->d_key_cache[kv_src], r->d_value_cache[kv_src],
                                     n_heads, local_kv_heads, hd, local_kv_dim,
                                     scale, win, dec_cache_len);
        }

        /* Output projection */
        launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                          cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);

        /* Post-attention norm */
        if (cl->post_attn_norm_w) {
            launch_rmsnorm(r, r->d_xb, r->d_xb, cl->post_attn_norm_w, n_embd, eps);
        }

        /* Attention residual: x += xb */
        launch_add(r, r->d_x, r->d_xb, n_embd);

        /* FFN: RMSNorm */
        launch_rmsnorm(r, r->d_xb, r->d_x, cl->ffn_norm_w, n_embd, eps);

        /* FFN gate + up */
        launch_matvec_auto(r, r->d_gate, cl->ffn_gate_w, r->d_xb,
                          cl->ffn_gate_rows, cl->ffn_gate_cols, cl->ffn_gate_type);
        launch_matvec_auto(r, r->d_up, cl->ffn_up_w, r->d_xb,
                          cl->ffn_up_rows, cl->ffn_up_cols, cl->ffn_up_type);

        /* GELU * up (not SiLU) */
        {
            int n = cl->ffn_gate_rows;
            void *a[] = { &r->d_gate, &r->d_up, &n };
            LAUNCH(r->fn_gelu_mul_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a);
        }

        /* FFN down */
        launch_matvec_auto(r, r->d_xb, cl->ffn_down_w, r->d_gate,
                          cl->ffn_down_rows, cl->ffn_down_cols, cl->ffn_down_type);

        /* Post-FFN norm */
        if (cl->post_ffw_norm_w) {
            launch_rmsnorm(r, r->d_xb, r->d_xb, cl->post_ffw_norm_w, n_embd, eps);
        }

        /* FFN residual: x += xb */
        launch_add(r, r->d_x, r->d_xb, n_embd);

        /* Layer output scale */
        if (cl->layer_scale_val != 1.0f) {
            int n = n_embd;
            void *a[] = { &r->d_x, &cl->layer_scale_val, &n };
            LAUNCH(r->fn_scale_f32, (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a);
        }

        goto gemma4_layer_done;

    } else if (r->is_hybrid && cl->is_ssm) {
            /* === SSM (Delta-Net) layer === */
            int qkv_dim = r->ssm_qkv_dim;
            int d_state = r->ssm_d_state;
            int n_group = r->ssm_n_group;
            int dt_rank = r->ssm_dt_rank;
            int conv_k  = r->ssm_conv_kernel;

            if (r->ssm_fused_decode &&
                cl->ssm_qkv_type == GGML_TYPE_Q6_K && cl->ssm_gate_type == GGML_TYPE_Q6_K &&
                cl->ssm_alpha_type == GGML_TYPE_F16 && cl->ssm_beta_type == GGML_TYPE_F16) {
                int qkv_rows = cl->ssm_qkv_rows, z_rows = cl->ssm_gate_rows, n_cols = cl->ssm_qkv_cols;
                void *a[] = { &r->d_ssm_qkv, &r->d_ssm_z, &r->d_ssm_alpha, &r->d_ssm_beta,
                              &cl->ssm_qkv_w, &cl->ssm_gate_w, &cl->ssm_alpha_w, &cl->ssm_beta_w,
                              &r->d_xb, &qkv_rows, &z_rows, &dt_rank, &n_cols };
                LAUNCH(r->fn_ssm_matvec4_q6k, qkv_rows + z_rows + 2 * dt_rank, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else if (r->ssm_fused_decode &&
                cl->ssm_qkv_type == GGML_TYPE_IQ3_XXS && cl->ssm_gate_type == GGML_TYPE_IQ3_XXS &&
                cl->ssm_alpha_type == GGML_TYPE_F16 && cl->ssm_beta_type == GGML_TYPE_F16) {
                int qkv_rows = cl->ssm_qkv_rows, z_rows = cl->ssm_gate_rows, n_cols = cl->ssm_qkv_cols;
                void *a[] = { &r->d_ssm_qkv, &r->d_ssm_z, &r->d_ssm_alpha, &r->d_ssm_beta,
                              &cl->ssm_qkv_w, &cl->ssm_gate_w, &cl->ssm_alpha_w, &cl->ssm_beta_w,
                              &r->d_xb, &qkv_rows, &z_rows, &dt_rank, &n_cols };
                LAUNCH(r->fn_ssm_matvec4_iq3xxs, qkv_rows + z_rows + 2 * dt_rank, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else {
            launch_matvec_auto(r, r->d_ssm_qkv,   cl->ssm_qkv_w,   r->d_xb, cl->ssm_qkv_rows,   cl->ssm_qkv_cols,   cl->ssm_qkv_type);
            launch_matvec_auto(r, r->d_ssm_z,      cl->ssm_gate_w,  r->d_xb, cl->ssm_gate_rows,  cl->ssm_gate_cols,  cl->ssm_gate_type);
            launch_matvec_auto(r, r->d_ssm_alpha,  cl->ssm_alpha_w, r->d_xb, cl->ssm_alpha_rows, cl->ssm_alpha_cols, cl->ssm_alpha_type);
            launch_matvec_auto(r, r->d_ssm_beta,   cl->ssm_beta_w,  r->d_xb, cl->ssm_beta_rows,  cl->ssm_beta_cols,  cl->ssm_beta_type);
            }

            if (r->ssm_fused_decode &&
                (qkv_dim - 2 * n_group * d_state) % n_group == 0 &&
                dt_rank % n_group == 0 && d_state <= 256) {
                /* Fused conv+silu+state, l2norm Q/K, repeat-tile, softplus, sigmoid:
                 * 7 launches -> 1. */
                void *args[] = { &r->d_ssm_conv_out, &cl->d_conv_state, &r->d_ssm_qkv,
                                 &cl->ssm_conv1d_w, &r->d_ssm_alpha, &r->d_ssm_beta,
                                 &cl->ssm_dt_bias, &cl->ssm_a,
                                 &r->d_ssm_Q_exp, &r->d_ssm_K_exp,
                                 &qkv_dim, &conv_k, &d_state, &n_group, &dt_rank, &eps };
                LAUNCH(r->fn_ssm_prep_f32, n_group, 1, 1, 256, 1, 1, 0, r->stream, args);
            } else {
            launch_softplus_mul(r, r->d_ssm_alpha, r->d_ssm_alpha, cl->ssm_dt_bias, cl->ssm_a, dt_rank);
            launch_sigmoid_inplace(r, r->d_ssm_beta, dt_rank);

            launch_conv1d(r, r->d_ssm_conv_out, cl->d_conv_state, r->d_ssm_qkv,
                         cl->ssm_conv1d_w, qkv_dim, conv_k);

            launch_l2_norm_heads(r, r->d_ssm_conv_out, n_group, d_state, eps);
            void *K_raw = (void *)((char *)r->d_ssm_conv_out + (size_t)n_group * d_state * sizeof(float));
            launch_l2_norm_heads(r, K_raw, n_group, d_state, eps);

            launch_repeat_tile(r, r->d_ssm_Q_exp, r->d_ssm_conv_out, dt_rank, d_state, n_group);
            launch_repeat_tile(r, r->d_ssm_K_exp, K_raw, dt_rank, d_state, n_group);
            }

            void *V_ptr = (void *)((char *)r->d_ssm_conv_out + (size_t)2 * n_group * d_state * sizeof(float));

            launch_deltanet_step(r, cl->d_recurrent_state, r->d_ssm_out,
                                r->d_ssm_Q_exp, r->d_ssm_K_exp, V_ptr,
                                r->d_ssm_alpha, r->d_ssm_beta, dt_rank, d_state);

             if (r->ssm_fused_decode && cl->ssm_out_type == GGML_TYPE_Q6_K && r->ssm_out_mw) {
                int nr = cl->ssm_out_rows;
                int nc = cl->ssm_out_cols;
                /* precompute per-head inv_mean once (independent of output row) */
                void *ia[] = { &r->d_ssm_inv_mean, &r->d_ssm_out, &dt_rank, &d_state, &eps };
                LAUNCH(r->fn_ssm_inv_mean_f32, (dt_rank + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, ia);
                void *a[] = { &r->d_xb, &cl->ssm_out_w, &r->d_ssm_out, &r->d_ssm_z, &cl->ssm_norm_w,
                              &r->d_ssm_inv_mean, &nr, &nc, &dt_rank };
                LAUNCH(r->fn_fused_ssm_out_gated_q6k_mw, (nr + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else if (r->ssm_fused_decode && cl->ssm_out_type == GGML_TYPE_Q6_K) {
                int nr = cl->ssm_out_rows;
                int nc = cl->ssm_out_cols;
                void *a[] = { &r->d_xb, &cl->ssm_out_w, &r->d_ssm_out, &r->d_ssm_z, &cl->ssm_norm_w,
                              &nr, &nc, &dt_rank, &d_state, &eps };
                LAUNCH(r->fn_fused_ssm_out_gated_q6k, nr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else if (r->ssm_fused_decode && cl->ssm_out_type == GGML_TYPE_IQ3_XXS) {
                int nr = cl->ssm_out_rows;
                int nc = cl->ssm_out_cols;
                void *a[] = { &r->d_xb, &cl->ssm_out_w, &r->d_ssm_out, &r->d_ssm_z, &cl->ssm_norm_w,
                              &nr, &nc, &dt_rank, &d_state, &eps };
                LAUNCH(r->fn_fused_ssm_out_gated_iq3xxs, nr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else {
                launch_gated_rmsnorm_silu(r, r->d_ssm_out, r->d_ssm_z, cl->ssm_norm_w,
                                         dt_rank, d_state, eps);
                launch_matvec_auto(r, r->d_xb, cl->ssm_out_w, r->d_ssm_out,
                                  cl->ssm_out_rows, cl->ssm_out_cols, cl->ssm_out_type);
            }

        } else if (r->is_hybrid) {
            /* === Gated attention layer (Qwen3.5) === */
            if (r->ssm_fused_decode &&
                cl->attn_q_type == GGML_TYPE_Q6_K && cl->attn_k_type == GGML_TYPE_Q6_K &&
                cl->attn_v_type == GGML_TYPE_Q6_K) {
                int qr = cl->attn_q_rows, kr = cl->attn_k_rows, vr = cl->attn_v_rows;
                int nc = cl->attn_q_cols;
                void *a[] = { &r->d_xb2, &r->d_k, &r->d_v, &cl->attn_q_w, &cl->attn_k_w, &cl->attn_v_w,
                              &qr, &kr, &vr, &nc, &r->d_xb };
                LAUNCH(r->fn_matvec_qkv_q6k, qr + kr + vr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else if (r->ssm_fused_decode &&
                cl->attn_q_type == GGML_TYPE_IQ3_XXS && cl->attn_k_type == GGML_TYPE_IQ3_XXS &&
                cl->attn_v_type == GGML_TYPE_IQ3_XXS) {
                int qr = cl->attn_q_rows, kr = cl->attn_k_rows, vr = cl->attn_v_rows;
                int nc = cl->attn_q_cols;
                void *a[] = { &r->d_xb2, &r->d_k, &r->d_v, &cl->attn_q_w, &cl->attn_k_w, &cl->attn_v_w,
                              &qr, &kr, &vr, &nc, &r->d_xb };
                LAUNCH(r->fn_matvec_qkv_iq3xxs, qr + kr + vr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else {
            launch_matvec_auto(r, r->d_xb2, cl->attn_q_w, r->d_xb,
                              cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);
            launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb, cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
            launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb, cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);
            }
            launch_deinterleave_qgate(r, r->d_q, r->d_attn_gate, r->d_xb2, n_heads, head_dim);

            if (cl->has_qk_norm) {
                if (cl->attn_q_norm_w) launch_qknorm(r, r->d_q, cl->attn_q_norm_w, n_heads, head_dim, eps);
                if (cl->attn_k_norm_w) launch_qknorm(r, r->d_k, cl->attn_k_norm_w, n_kv_heads, head_dim, eps);
            }

            launch_rope_devp(r, r->d_q, n_heads, head_dim, r->rope_freq_base);
            launch_rope_devp(r, r->d_k, n_kv_heads, head_dim, r->rope_freq_base);

            launch_kv_store_devp(r, r->d_key_cache[l], r->d_value_cache[l],
                                 r->d_k, r->d_v, kv_dim, r->max_seq_len);

            float scale = 1.0f / sqrtf((float)head_dim);
            size_t smem_attn = (size_t)r->max_seq_len * sizeof(float);
            launch_attention_devp(r, r->d_xb2, r->d_q,
                                   r->d_key_cache[l], r->d_value_cache[l],
                                   n_heads, n_kv_heads, head_dim, kv_dim, scale, smem_attn);

            int q_dim_local = n_heads * head_dim;
            if (r->ssm_fused_decode && cl->attn_output_type == GGML_TYPE_Q6_K) {
                int nr = cl->attn_output_rows;
                int nc = cl->attn_output_cols;
                void *a[] = { &r->d_xb, &cl->attn_output_w, &r->d_xb2, &r->d_attn_gate, &nr, &nc };
                LAUNCH(r->fn_matvec_out_gated_q6k, nr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else if (r->ssm_fused_decode && cl->attn_output_type == GGML_TYPE_IQ3_XXS) {
                int nr = cl->attn_output_rows;
                int nc = cl->attn_output_cols;
                void *a[] = { &r->d_xb, &cl->attn_output_w, &r->d_xb2, &r->d_attn_gate, &nr, &nc };
                LAUNCH(r->fn_matvec_out_gated_iq3xxs, nr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else {
                launch_sigmoid_mul(r, r->d_xb2, r->d_attn_gate, q_dim_local);
                launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                                  cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);
            }

        } else {
            /* === Standard attention === */
            if (r->ssm_fused_decode &&
                cl->attn_q_type == GGML_TYPE_Q6_K && cl->attn_k_type == GGML_TYPE_Q6_K &&
                cl->attn_v_type == GGML_TYPE_Q6_K) {
                int qr = cl->attn_q_rows, kr = cl->attn_k_rows, vr = cl->attn_v_rows;
                int nc = cl->attn_q_cols;
                void *a[] = { &r->d_q, &r->d_k, &r->d_v, &cl->attn_q_w, &cl->attn_k_w, &cl->attn_v_w,
                              &qr, &kr, &vr, &nc, &r->d_xb };
                LAUNCH(r->fn_matvec_qkv_q6k, qr + kr + vr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else if (r->ssm_fused_decode &&
                cl->attn_q_type == GGML_TYPE_IQ3_XXS && cl->attn_k_type == GGML_TYPE_IQ3_XXS &&
                cl->attn_v_type == GGML_TYPE_IQ3_XXS) {
                int qr = cl->attn_q_rows, kr = cl->attn_k_rows, vr = cl->attn_v_rows;
                int nc = cl->attn_q_cols;
                void *a[] = { &r->d_q, &r->d_k, &r->d_v, &cl->attn_q_w, &cl->attn_k_w, &cl->attn_v_w,
                              &qr, &kr, &vr, &nc, &r->d_xb };
                LAUNCH(r->fn_matvec_qkv_iq3xxs, qr + kr + vr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else {
                launch_matvec_auto(r, r->d_q, cl->attn_q_w, r->d_xb, cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);
                launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb, cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
                launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb, cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);
            }

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
                                 r->d_k, r->d_v, kv_dim, r->max_seq_len);

            float scale = 1.0f / sqrtf((float)head_dim);
            size_t smem_attn = (size_t)r->max_seq_len * sizeof(float);
            launch_attention_devp(r, r->d_xb2, r->d_q,
                                   r->d_key_cache[l], r->d_value_cache[l],
                                   n_heads, n_kv_heads, head_dim, kv_dim, scale, smem_attn);

            launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                              cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);
        }

        /* Residual + FFN RMSNorm fused: x += xb; xb = rmsnorm(x)*w */
        {
            void *a[] = { &r->d_x, &r->d_xb, &r->d_xb, &cl->ffn_norm_w, &n_embd, &eps };
            LAUNCH(r->fn_res_rmsnorm_f32, 1, 1, 1, 256, 1, 1, 256 * sizeof(float), r->stream, a);
        }

        if (cl->is_moe) {
            forward_moe_ffn(r, cl);
            r->moe_add_pending = 1;   /* folded into next layer's pre-attn norm */
            goto ffn_done;
        } else {
            /* Dense FFN */
            if (r->ssm_fused_decode &&
                cl->ffn_gate_type == GGML_TYPE_Q6_K && cl->ffn_up_type == GGML_TYPE_Q6_K) {
                int n_ff = cl->ffn_gate_rows;
                int nc = cl->ffn_gate_cols;
                void *a[] = { &r->d_gate, &cl->ffn_gate_w, &cl->ffn_up_w, &r->d_xb, &n_ff, &nc };
                LAUNCH(r->fn_ffn_gate_up_silu_q6k, n_ff, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else if (r->ssm_fused_decode &&
                cl->ffn_gate_type == GGML_TYPE_IQ3_XXS && cl->ffn_up_type == GGML_TYPE_IQ3_XXS) {
                int n_ff = cl->ffn_gate_rows;
                int nc = cl->ffn_gate_cols;
                void *a[] = { &r->d_gate, &cl->ffn_gate_w, &cl->ffn_up_w, &r->d_xb, &n_ff, &nc };
                if (r->ffn_iq3_mw) {
                    int rpb = r->mw_threads / 32;
                    LAUNCH(r->fn_ffn_gate_up_silu_iq3xxs_mw, (n_ff + rpb - 1) / rpb, 1, 1, (unsigned)r->mw_threads, 1, 1, 0, r->stream, a);
                } else
                    LAUNCH(r->fn_ffn_gate_up_silu_iq3xxs, n_ff, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else {
                launch_matvec_auto(r, r->d_gate, cl->ffn_gate_w, r->d_xb,
                              cl->ffn_gate_rows, cl->ffn_gate_cols, cl->ffn_gate_type);
                launch_matvec_auto(r, r->d_up, cl->ffn_up_w, r->d_xb,
                              cl->ffn_up_rows, cl->ffn_up_cols, cl->ffn_up_type);
                launch_silu_mul(r, r->d_gate, r->d_up, n_ff);
            }
            if (r->ssm_fused_decode && cl->ffn_down_type == GGML_TYPE_Q6_K) {
                int nr = cl->ffn_down_rows; /* n_embd */
                int nc = cl->ffn_down_cols; /* n_ff */
                void *a[] = { &r->d_x, &cl->ffn_down_w, &r->d_gate, &nr, &nc };
                LAUNCH(r->fn_matvec_down_residual_q6k, nr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else if (r->ssm_fused_decode && cl->ffn_down_type == GGML_TYPE_IQ3_XXS) {
                int nr = cl->ffn_down_rows; /* n_embd */
                int nc = cl->ffn_down_cols; /* n_ff */
                void *a[] = { &r->d_x, &cl->ffn_down_w, &r->d_gate, &nr, &nc };
                /* down warp-per-row is neutral-to-worse (5120 rows underfill);
                 * keep the original block-per-row unless split-K is requested. */
                if (r->ffn_iq3_mw && r->down_ksplit > 1)
                    LAUNCH(r->fn_matvec_down_residual_iq3xxs_splitk, (nr + 7) / 8, (unsigned)r->down_ksplit, 1, 256, 1, 1, 0, r->stream, a);
                else
                    LAUNCH(r->fn_matvec_down_residual_iq3xxs, nr, 1, 1, 256, 1, 1, 0, r->stream, a);
            } else {
                launch_matvec_auto(r, r->d_xb, cl->ffn_down_w, r->d_gate,
                              cl->ffn_down_rows, cl->ffn_down_cols, cl->ffn_down_type);
                launch_add(r, r->d_x, r->d_xb, n_embd);
            }
        }

gemma4_layer_done:
ffn_done:
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
        /* Text-encoder hidden snapshots: copy this layer's post-residual hidden
         * state (d_x) into the snapshot buffer. Only active when layers are set
         * (which also disables graph capture so this inline copy runs). */
        for (int si = 0; si < r->n_hidden_snapshots; si++) {
            if (r->hidden_snapshot_layers[si] == l && r->d_hidden_snapshots) {
                hipMemcpyAsync((char *)r->d_hidden_snapshots + (size_t)si * r->n_embd * sizeof(float),
                               r->d_x, (size_t)r->n_embd * sizeof(float),
                               hipMemcpyDeviceToDevice, r->stream);
            }
        }
    }
    /* Final RMSNorm */
    if (r->moe_add_pending) { launch_add(r, r->d_x, r->d_moe_accum, r->n_embd); r->moe_add_pending = 0; }
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
     * MoE: capturable ONLY when device-side dispatch is active (moe_dev_dispatch_ok),
     * i.e. router top-K + softmax + shared-gate sigmoid run on-device with no host
     * sync. The legacy host-sync MoE path stays gated out. */
    r->graph_eligible =
        !r->debug_layers && (!r->is_moe || r->moe_dev_dispatch_ok);

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

    /* Gemma4: scale token embeddings by sqrt(n_embd) */
    if (r->is_gemma4) {
        int n = n_embd;
        float scale = r->embd_scale;
        void *a[] = { &r->d_x, &scale, &n };
        LAUNCH(r->fn_scale_f32, (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a);
    }

    /* Position is consumed by *_devp launchers via r->d_position */
    hipMemcpyAsync(r->d_position, &position, sizeof(int),
                   hipMemcpyHostToDevice, r->stream);
    r->cur_position = position;

    if (r->graph_ready_logits) {
        hipGraphLaunch(r->graph_exec_logits, r->stream);
    } else {
        forward_blocks_body(r);
        launch_matvec_auto(r, r->d_logits, r->d_output_w, r->d_x,
                           r->n_vocab, r->n_embd, r->output_w_type);
        /* Gemma4: final logit softcapping */
        if (r->is_gemma4 && r->final_logit_softcapping > 0.0f) {
            float cap = r->final_logit_softcapping;
            int n = r->n_vocab;
            void *a[] = { &r->d_logits, &n, &cap };
            LAUNCH(r->fn_logit_softcap_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a);
        }
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
                void *_w = get_bf16_weight(r, cl->w_field, cl->w_field##_bf16,            \
                                cl->type_field, cl->rows_field, cl->cols_field);          \
                if (!_w) return -1;                                                       \
                if (gemm_run_bf16_w(r, dst, _w, r->d_xnorm_batch_bf16, M,                 \
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
                void *ow = get_bf16_weight(r, cl->ssm_out_w, cl->ssm_out_w_bf16,
                                cl->ssm_out_type, cl->ssm_out_rows, cl->ssm_out_cols);
                if (!ow) return -1;
                if (gemm_run_bf16_w(r, r->d_attn_proj_batch, ow,
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
            if (gemm_run_bf16_w(r, q_dst, qw,
                                   r->d_xnorm_batch_bf16, M, q_proj_rows, n_embd, r->stream) != 0) return -1;
            if (is_gated_attn) {
                launch_deinterleave_qgate_batch(r, r->d_q_batch, r->d_attn_gate_batch,
                                                r->d_qfull_batch, n_heads, head_dim, M);
            }
        }

        /* Gemma4 per-layer dimensions (used by K/V GEMMs, bias, QK-norm, RoPE, KV store, attention) */
        int l_hd = head_dim, l_qdim = q_dim, l_kvdim = kv_dim, l_nkv = n_kv_heads;
        if (r->is_gemma4) {
            l_hd = cl->local_head_dim ? cl->local_head_dim : head_dim;
            l_nkv = cl->local_kv_heads ? cl->local_kv_heads : n_kv_heads;
            l_qdim = n_heads * l_hd;
            l_kvdim = l_nkv * l_hd;
        }

        {
            void *kw = get_bf16_weight(r, cl->attn_k_w, cl->attn_k_w_bf16,
                                       cl->attn_k_type, cl->attn_k_rows, cl->attn_k_cols);
            if (!kw) return -1;
            if (gemm_run_bf16_w(r, r->d_k_batch, kw,
                                   r->d_xnorm_batch_bf16, M, l_kvdim, n_embd, r->stream) != 0) return -1;
        }
        /* Gemma4 "global"/full-attention layers have no v_proj: V = K (config
         * attention_k_eq_v). matches llama.cpp: Vcur = Kcur (the RAW k_proj, before
         * k_norm), then a weightless per-head RMS norm; V is not RoPE'd. So copy K->V
         * HERE (pre-k_norm), then k_norm runs on K only, then V gets its raw RMS norm. */
        int v_eq_k = r->is_gemma4 && (cl->shared_kv_source < 0) && (cl->attn_v_rows <= 0);
        if (v_eq_k) {
            hipMemcpyAsync(r->d_v_batch, r->d_k_batch, (size_t)M * l_kvdim * sizeof(float),
                           hipMemcpyDeviceToDevice, r->stream);
        } else {
            void *vw = get_bf16_weight(r, cl->attn_v_w, cl->attn_v_w_bf16,
                                       cl->attn_v_type, cl->attn_v_rows, cl->attn_v_cols);
            if (!vw || cl->attn_v_rows <= 0) {
                hipMemsetAsync(r->d_v_batch, 0, (size_t)M * l_kvdim * sizeof(float), r->stream);
            } else if (gemm_run_bf16_w(r, r->d_v_batch, vw,
                                           r->d_xnorm_batch_bf16, M, l_kvdim, n_embd, r->stream) != 0) return -1;
        }

        /* ---- Biases ---- */
        if (cl->attn_q_bias || cl->attn_k_bias || cl->attn_v_bias) {
            for (int m = 0; m < M; m++) {
                float *qrow = (float *)r->d_q_batch + (size_t)m * l_qdim;
                float *krow = (float *)r->d_k_batch + (size_t)m * l_kvdim;
                float *vrow = (float *)r->d_v_batch + (size_t)m * l_kvdim;
                if (cl->attn_q_bias) launch_add(r, qrow, cl->attn_q_bias, l_qdim);
                if (cl->attn_k_bias) launch_add(r, krow, cl->attn_k_bias, l_kvdim);
                if (cl->attn_v_bias) launch_add(r, vrow, cl->attn_v_bias, l_kvdim);
            }
        }
        /* ---- QK-norm: batched over M rows ----
         * NOTE: row strides MUST be the per-layer packed widths (l_qdim/l_kvdim),
         * not the global q_dim/kv_dim. The Q/K GEMMs write rows contiguously at the
         * local width; for Gemma4 SWA layers (head_dim 256 vs full 512) the global
         * stride is 2x too large and would read stale data from the prior layer. */
        if (cl->has_qk_norm) {
            if (cl->attn_q_norm_w)
                launch_qknorm_batch(r, r->d_q_batch, cl->attn_q_norm_w,
                                    n_heads, l_hd, M, l_qdim, eps);
            if (cl->attn_k_norm_w)
                launch_qknorm_batch(r, r->d_k_batch, cl->attn_k_norm_w,
                                    l_nkv, l_hd, M, l_kvdim, eps);
        }

        /* V raw RMSNorm (weightless, per head) for every KV-bearing layer — matches
         * llama.cpp Vcur = ggml_rms_norm(Vcur, eps). V is not RoPE'd. */
        if (r->is_gemma4 && cl->shared_kv_source < 0)
            launch_raw_rmsnorm_heads_batch(r, r->d_v_batch, l_nkv, l_hd, M, l_kvdim, eps);

        /* ---- Batched RoPE on Q and K (one launch each, grid=(n_heads, M)) ---- */
        if (r->use_mrope) {
            int s0 = r->mrope_sections[0], s1 = r->mrope_sections[1];
            int s2 = r->mrope_sections[2], s3 = r->mrope_sections[3];
            launch_rope_mrope_batch(r, r->d_q_batch, n_heads,    l_hd,
                                    position_start, r->rope_freq_base,
                                    s0, s1, s2, s3, l_qdim,  M);
            launch_rope_mrope_batch(r, r->d_k_batch, l_nkv, l_hd,
                                    position_start, r->rope_freq_base,
                                    s0, s1, s2, s3, l_kvdim, M);
        } else {
            /* Per-layer RoPE base (Gemma4 SWA uses freq_base_swa, full uses freq_base)
             * and n_rope_pairs=0 for Gemma4 so the kernel rotates the full per-layer
             * head_dim (l_hd); the global n_rope_pairs=256 would force rope_dim=512 on
             * the 256-wide SWA heads, corrupting their positional encoding. */
            float l_base = (r->is_gemma4 && cl->is_swa) ? r->rope_freq_base_swa : r->rope_freq_base;
            int   l_npairs = r->is_gemma4 ? 0 : r->n_rope_pairs;
            /* Full-attention layers use proportional RoPE (freq_factors); SWA does not. */
            void *l_ff = (r->is_gemma4 && !cl->is_swa) ? r->d_rope_freq_factors : NULL;
            launch_rope_neox_batch_ff(r, r->d_q_batch, n_heads,    l_hd,
                                   position_start, l_base,
                                   l_npairs, l_qdim,  M, l_ff);
            launch_rope_neox_batch_ff(r, r->d_k_batch, l_nkv, l_hd,
                                   position_start, l_base,
                                   l_npairs, l_kvdim, M, l_ff);
        }

        /* ---- Batched KV cache store ---- */
        /* SWA layers use a circular cache (slot = pos % swa_cache_len); full-attn
         * layers use a linear cache (cache_len = max_seq_len => no wrap). */
        int pf_cache_len = (r->is_gemma4 && cl->is_swa) ? r->swa_cache_len : r->max_seq_len;
        int pf_window    = (r->is_gemma4 && cl->is_swa) ? r->swa_window_size : 0;
        if (r->is_gemma4) {
            /* Source (batch) stride must be l_kvdim — the packed width the V/K GEMMs
             * wrote — not the global kv_dim (2x too large for SWA layers). */
            launch_kv_store_batch_strided(r, r->d_key_cache[l], r->d_value_cache[l],
                                          r->d_k_batch, r->d_v_batch,
                                          position_start, M, l_kvdim, l_kvdim, pf_cache_len);
        } else {
            launch_kv_store_batch(r, r->d_key_cache[l], r->d_value_cache[l],
                                  r->d_k_batch, r->d_v_batch,
                                  position_start, M, kv_dim);
        }

        /* ---- Attention ---- */
        if (r->is_gemma4) {
            /* Gemma4: bounded online-softmax (flash) attention. Handles SWA
             * (window>0, circular cache) and full (window=0, linear cache) and
             * never exceeds LDS regardless of sequence length (scales to pp32k).
             * Gemma4 uses attention scale = 1.0 (self.scaling=1.0); the Q/K RMS
             * norms set the magnitude, so there is NO 1/sqrt(head_dim) factor. */
            float l_scale = 1.0f;
            if (!cl->is_swa && r->gemma_fa_wmma && l_hd == 512 && l_nkv == 1) {
                /* Full-attention layers (the O(M^2) bottleneck): WMMA flash kernel. */
                launch_flash_attn_wmma_hd512(r, r->d_attn_out_batch, r->d_q_batch,
                                             r->d_key_cache[l], r->d_value_cache[l],
                                             M, position_start + M, position_start,
                                             n_heads, l_nkv, l_hd, l_kvdim, l_scale);
            } else {
                launch_attn_prefill_flash(r, r->d_attn_out_batch, r->d_q_batch,
                                           r->d_key_cache[l], r->d_value_cache[l],
                                           n_heads, l_nkv, l_hd, l_kvdim,
                                           M, position_start, l_scale, pf_window, pf_cache_len);
            }
        } else if (r->fa_path_ok) {
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
        int attn_out_elems = M * (r->is_gemma4 ? l_qdim : q_dim);
        launch_pack_bf16_from_f32(r, r->d_attn_out_batch_bf16,
                                  r->d_attn_out_batch, attn_out_elems);
        {
            void *ow = get_bf16_weight(r, cl->attn_output_w, cl->attn_output_w_bf16,
                                       cl->attn_output_type, cl->attn_output_rows, cl->attn_output_cols);
            if (!ow) return -1;
            int o_rows = r->is_gemma4 ? l_qdim : q_dim;
            if (gemm_run_bf16_w(r, r->d_attn_proj_batch, ow,
                                   r->d_attn_out_batch_bf16, M, n_embd, o_rows, r->stream) != 0) return -1;
        }

        /* Residual with Gemma4 post-attention norm. llama.cpp applies the norm to the
         * attention output (Wo·attn) and THEN adds the residual: x += post_attn_norm(attn_proj).
         * (Earlier this normed the post-residual x, which is structurally wrong.) */
        if (r->is_gemma4 && cl->post_attn_norm_w) {
            launch_rmsnorm_batch(r, r->d_xnorm_batch, r->d_attn_proj_batch,
                                 cl->post_attn_norm_w, n_embd, M, n_embd, eps);
            launch_add(r, r->d_x_batch, r->d_xnorm_batch, M * n_embd);
        } else {
            launch_add(r, r->d_x_batch, r->d_attn_proj_batch, M * n_embd);
        }

ffn_section:
        /* ---- Pre-FFN RMSNorm: one batched launch over M rows ---- */
        launch_rmsnorm_batch(r, r->d_xnorm_batch, r->d_x_batch,
                             cl->ffn_norm_w, n_embd, M, n_embd, eps);

        /* ---- MoE FFN: per-row over the batch (Phase 2 increment 1). The router
         * top-K + sparse experts are applied to each of the M normed rows; the
         * attention/SSM projections above are already batched via hipBLASLt.
         * (Batched token-grouped experts are the next increment.) ---- */
        if (cl->is_moe) {
            if (r->moe_prefill_batched) {
                if (forward_moe_ffn_batched(r, cl, M) != 0) return -1;
            } else {
                void *saved_xb = r->d_xb;
                for (int m = 0; m < M; m++) {
                    r->d_xb = (char *)r->d_xnorm_batch + (size_t)m * n_embd * sizeof(float);
                    forward_moe_ffn(r, cl);  /* reads + overwrites that normed row */
                    launch_add(r, (float *)r->d_x_batch + (size_t)m * n_embd, r->d_xb, n_embd);
                }
                r->d_xb = saved_xb;
            }
            continue;
        }

        launch_pack_bf16_from_f32(r, r->d_ffn_norm_batch_bf16,
                                  r->d_xnorm_batch, M * n_embd);

        /* ---- gate/up GEMMs ---- */
        {
            void *gw = get_bf16_weight(r, cl->ffn_gate_w, cl->ffn_gate_w_bf16,
                                       cl->ffn_gate_type, cl->ffn_gate_rows, cl->ffn_gate_cols);
            if (!gw) return -1;
            if (gemm_run_bf16_w(r, r->d_gate_batch, gw,
                                   r->d_ffn_norm_batch_bf16, M, n_ff, n_embd, r->stream) != 0) return -1;
        }
        {
            void *uw = get_bf16_weight(r, cl->ffn_up_w, cl->ffn_up_w_bf16,
                                       cl->ffn_up_type, cl->ffn_up_rows, cl->ffn_up_cols);
            if (!uw) return -1;
            if (gemm_run_bf16_w(r, r->d_up_batch, uw,
                                   r->d_ffn_norm_batch_bf16, M, n_ff, n_embd, r->stream) != 0) return -1;
        }

        /* SiLU(gate) * up — elementwise across [M, n_ff] */
        if (r->is_gemma4) {
            int n = M * n_ff;
            void *a[] = { &r->d_gate_batch, &r->d_up_batch, &n };
            LAUNCH(r->fn_gelu_mul_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a);
        } else {
            launch_silu_mul(r, r->d_gate_batch, r->d_up_batch, M * n_ff);
        }

        /* Pack and down projection */
        launch_pack_bf16_from_f32(r, r->d_silu_batch_bf16,
                                  r->d_gate_batch, M * n_ff);
        {
            void *dw = get_bf16_weight(r, cl->ffn_down_w, cl->ffn_down_w_bf16,
                                       cl->ffn_down_type, cl->ffn_down_rows, cl->ffn_down_cols);
            if (!dw) return -1;
            if (gemm_run_bf16_w(r, r->d_down_batch, dw,
                                   r->d_silu_batch_bf16, M, n_embd, n_ff, r->stream) != 0) return -1;
        }

        /* Residual + Gemma4 post-FFN norm */
        if (r->is_gemma4 && cl->post_ffw_norm_w) {
            launch_rmsnorm_batch(r, r->d_xnorm_batch, r->d_down_batch,
                                 cl->post_ffw_norm_w, n_embd, M, n_embd, eps);
            launch_add(r, r->d_x_batch, r->d_xnorm_batch, M * n_embd);
        } else {
            launch_add(r, r->d_x_batch, r->d_down_batch, M * n_embd);
        }

        /* Gemma4 layer output scale */
        if (r->is_gemma4 && cl->layer_scale_val != 1.0f) {
            int n = M * n_embd;
            float sv = cl->layer_scale_val;
            void *a[] = { &r->d_x_batch, &sv, &n };
            LAUNCH(r->fn_scale_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a);
        }

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
    /* Paths that internally chunk the prefill (Gemma4 and batched-MoE) can accept
     * M > batch_max; only the unchunked dense path is hard-capped at batch_max. */
    if (M > r->batch_max && !r->is_gemma4 && !r->moe_prefill_batched) return 0;
    if (r->_ds_embd != NULL) return 0;  /* DeepStack handled in embed path only */
    return 1;
}

/* Per-chunk size for the batched prefill. Gemma4 uses a fixed chunk (SWA circular
 * cache correctness); other models follow the legacy MoE/dense policy. Always
 * capped at batch_max so the per-chunk activation buffers never overflow. */
static int prefill_chunk_size(const hip_llm_runner *r, int n_tokens) {
    int chunk;
    if (r->is_gemma4)            chunk = r->gemma_prefill_chunk;
    else if (r->moe_prefill_batched) chunk = r->gemm_own ? r->batch_max : 256;
    else                         chunk = n_tokens;
    { const char *e = getenv("LLM_MOE_CHUNK"); if (e) { int c = atoi(e); if (c >= 1) chunk = c; } }
    { const char *e = getenv("LLM_PREFILL_CHUNK"); if (e) { int c = atoi(e); if (c >= 1) chunk = c; } }
    if (chunk > r->batch_max) chunk = r->batch_max;
    /* Gemma4 SWA circular cache holds window+gemma_prefill_chunk positions; a larger
     * chunk would overwrite still-needed keys mid-batch, so clamp hard. */
    if (r->is_gemma4 && chunk > r->gemma_prefill_chunk) chunk = r->gemma_prefill_chunk;
    if (chunk < 1) chunk = 1;
    return chunk;
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
    /* Gemma4 scales token embeddings by sqrt(n_embd). The per-token path does this
     * (forward_blocks); the batched path must too or hidden states are ~62x too small. */
    if (r->is_gemma4) {
        int n = M * n_embd;
        float scale = r->embd_scale;
        void *a[] = { &r->d_x_batch, &scale, &n };
        LAUNCH(r->fn_scale_f32, (n + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a);
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
        int chunk = prefill_chunk_size(r, n_tokens);
        for (int off = 0; off < n_tokens; off += chunk) {
            int cc = n_tokens - off; if (cc > chunk) cc = chunk;
            if (embed_tokens_batch(r, tokens + off, cc) != 0) return NULL;
            if (forward_block_batched_dense(r, cc, position_start + off) != 0) return NULL;
        }
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
        /* Chunk the batch: hipBLASLt/Tensile fails to build plans for this model's
         * large SSM/MoE GEMM shapes at M>256, so process the prefill in <=256-token
         * sub-batches. SSM conv/recurrent state and KV cache carry across chunks.
         * Gemma4 uses a fixed chunk for SWA circular-cache correctness. */
        int chunk = prefill_chunk_size(r, n_tokens);
        for (int off = 0; off < n_tokens; off += chunk) {
            int cc = n_tokens - off; if (cc > chunk) cc = chunk;
            if (embed_tokens_batch(r, tokens + off, cc) != 0) return NULL;
            if (forward_block_batched_dense(r, cc, position_start + off) != 0) return NULL;
        }
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
        /* Chunk to <=256 rows (Tensile plan limit for this model's GEMMs); SSM/KV
         * state carries across chunks. _ds_embd is offset per chunk so the layer
         * loop's deepstack slice indexing (local row m) maps to the global row. */
        int chunk = r->moe_prefill_batched ? (r->gemm_own ? r->batch_max : 256) : M;
        { const char *e = getenv("LLM_MOE_CHUNK"); if (e) { chunk = atoi(e); if (chunk < 1) chunk = M; } }
        for (int off = 0; off < M; off += chunk) {
            int cc = M - off; if (cc > chunk) cc = chunk;
            for (int m = 0; m < cc; m++) {
                const float *src = embds + (size_t)(off + m) * embd_stride;
                float *dst = (float *)r->d_x_batch + (size_t)m * n_embd;
                hipMemcpyAsync(dst, src, (size_t)n_embd * sizeof(float),
                               hipMemcpyHostToDevice, r->stream);
            }
            r->_ds_embd        = embds + (size_t)off * embd_stride;
            r->_ds_embd_stride = embd_stride;
            int rc = forward_block_batched_dense(r, cc, position_start + off);
            r->_ds_embd        = NULL;
            r->_ds_embd_stride = 0;
            if (rc != 0) return NULL;
        }

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

/* Select up to 3 layers whose post-residual hidden state is snapshotted during
 * each forward (text-encoder path). Allocates the snapshot buffer and disables
 * graph capture (so the inline snapshot copy in forward_blocks_body runs). */
int hip_llm_set_hidden_snapshot_layers(hip_llm_runner *r, const int *layers, int n_slots) {
    if (!r) return -1;
    if (n_slots < 0 || n_slots > 3) return -1;
    r->n_hidden_snapshots = 0;
    for (int i = 0; i < 3; i++) r->hidden_snapshot_layers[i] = -1;
    for (int i = 0; i < n_slots; i++) {
        int layer = layers ? layers[i] : -1;
        if (layer < 0 || layer >= r->n_layers) return -1;
        r->hidden_snapshot_layers[i] = layer;
        r->n_hidden_snapshots++;
    }
    if (n_slots > 0) {
        if (!r->d_hidden_snapshots &&
            hipMalloc(&r->d_hidden_snapshots, (size_t)3 * r->n_embd * sizeof(float)) != hipSuccess)
            return -1;
        /* Snapshots require the non-graph forward path (the graph captured at load
         * predates the snapshot copies). Disable + drop the captured graphs so
         * hip_llm_forward falls back to forward_blocks_body. */
        r->graph_disabled = 1;
        r->graph_ready_hidden = 0;
        r->graph_ready_logits = 0;
    }
    return 0;
}

/* Read the captured snapshots: dst receives n_slots contiguous [n] blocks
 * (layout [n_slots, n]). n must equal n_embd. */
int hip_llm_read_hidden_snapshots(const hip_llm_runner *r, float *dst, int n_slots, int n) {
    if (!r || !dst || !r->d_hidden_snapshots) return -1;
    if (n != r->n_embd) return -1;
    if (n_slots > r->n_hidden_snapshots) n_slots = r->n_hidden_snapshots;
    hipStreamSynchronize(r->stream);
    for (int si = 0; si < n_slots; si++) {
        if (hipMemcpy(dst + (size_t)si * n,
                      (char *)r->d_hidden_snapshots + (size_t)si * r->n_embd * sizeof(float),
                      (size_t)n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
            return -1;
    }
    return 0;
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
            if (r->d_key_cache[l]) {
                /* Skip shared-KV aliases to avoid double-free */
                int is_alias = 0;
                if (r->is_gemma4) {
                    for (int k = 0; k < l; k++) {
                        if (r->layers[k].shared_kv_source == l && r->d_key_cache[k] == r->d_key_cache[l]) {
                            is_alias = 1; break;
                        }
                    }
                }
                if (!is_alias) hipFree(r->d_key_cache[l]);
            }
        }
        free(r->d_key_cache);
    }
    if (r->d_value_cache) {
        for (int l = 0; l < r->n_layers; l++) {
            if (r->d_value_cache[l]) {
                int is_alias = 0;
                if (r->is_gemma4) {
                    for (int k = 0; k < l; k++) {
                        if (r->layers[k].shared_kv_source == l && r->d_value_cache[k] == r->d_value_cache[l]) {
                            is_alias = 1; break;
                        }
                    }
                }
                if (!is_alias) hipFree(r->d_value_cache[l]);
            }
        }
        free(r->d_value_cache);
    }

    /* Gemma4 arrays and PLE buffers */
    if (r->is_gemma4) {
        free(r->swa_pattern);
        free(r->per_layer_kv_heads);
        free(r->h_rope_freq_factors);
        free(r->h_rope_inv_freq_full);
        free(r->h_rope_inv_freq_swa);
        if (r->d_ple_combined) hipFree(r->d_ple_combined);
        if (r->d_ple_buf)      hipFree(r->d_ple_buf);
        if (r->d_ple_proj)     hipFree(r->d_ple_proj);
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
            if (cl->ssm_qkv_w_bf16)   hipFree(cl->ssm_qkv_w_bf16);
            if (cl->ssm_gate_w_bf16)  hipFree(cl->ssm_gate_w_bf16);
            if (cl->ssm_alpha_w_bf16) hipFree(cl->ssm_alpha_w_bf16);
            if (cl->ssm_beta_w_bf16)  hipFree(cl->ssm_beta_w_bf16);
            if (cl->ssm_out_w_bf16)   hipFree(cl->ssm_out_w_bf16);
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
            /* Gemma4 per-layer weights */
            if (cl->post_attn_norm_w)  hipFree(cl->post_attn_norm_w);
            if (cl->post_ffw_norm_w)   hipFree(cl->post_ffw_norm_w);
            if (cl->ple_inp_gate_w)    hipFree(cl->ple_inp_gate_w);
            if (cl->ple_proj_w)        hipFree(cl->ple_proj_w);
            if (cl->ple_post_norm_w)   hipFree(cl->ple_post_norm_w);
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
        case GGML_TYPE_Q8_0:    *blk_elems = 32;  *blk_bytes = 36;  break;
        case GGML_TYPE_Q2_K:    *blk_elems = 256; *blk_bytes = 84;  break;
        case GGML_TYPE_Q3_K:    *blk_elems = 256; *blk_bytes = 110; break;
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
    if (!r) return;
    if (r->is_hybrid) {
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
    if (r->is_gemma4) {
        for (int l = 0; l < r->n_layers; l++) {
            hip_layer *cl = &r->layers[l];
            if (cl->shared_kv_source >= 0) continue;
            int local_kv = cl->local_kv_heads;
            int local_hd = cl->local_head_dim;
            size_t kv_bytes = (size_t)local_kv * local_hd * sizeof(float);
            if (cl->is_swa) {
                kv_bytes *= r->swa_window_size;
            } else {
                kv_bytes *= r->max_seq_len;
            }
            if (r->d_key_cache[l])   hipMemset(r->d_key_cache[l], 0, kv_bytes);
            if (r->d_value_cache[l]) hipMemset(r->d_value_cache[l], 0, kv_bytes);
        }
    }
}

int hip_llm_n_embd(const hip_llm_runner *r) { return r ? r->n_embd : 0; }
int hip_llm_n_layers(const hip_llm_runner *r) { return r ? r->n_layers : 0; }
int hip_llm_n_vocab(const hip_llm_runner *r) { return r ? r->n_vocab : 0; }
int hip_llm_max_seq_len(const hip_llm_runner *r) { return r ? r->max_seq_len : 0; }
