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
#include "../cublasew.h"
#include "../cuda_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "moe_grid_data.h"

#define CLLM_MOE_PREFILL_AUTO   0
#define CLLM_MOE_PREFILL_CUBLAS 1
#define CLLM_MOE_PREFILL_FUSED  2
#define CLLM_MOE_PREFILL_EXACT  3
#define CLLM_PREFILL_EXACT_MAX_TOKENS_DEFAULT 32
/* 0 = always use the cuBLAS dequant+GEMM expert path. The per-expert direct IQ2
   matvec is slower for the modal small-expert case (avg ~16 rows/expert at 512
   tokens, top-8/256 experts), so it regressed prefill 4.5x as a default; opt in
   per-run via CUDA_LLM_MOE_DIRECT_BATCH_THRESHOLD if a workload benefits. */
#define CLLM_MOE_DIRECT_BATCH_THRESHOLD_DEFAULT 0

#if defined(__SSE4_2__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <nmmintrin.h>
#endif

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* transformer.h header-only: gives us qtensor type + dequant declarations */
#include "../../common/ggml_dequant.h"
#include "../../common/transformer.h"
/* safetensors.h for cuda_llm_inject_biases — IMPLEMENTATION must be defined
 * by the translation unit that includes this .c file (e.g. test_cuda_qimg.c) */
#include "../../common/safetensors.h"

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
"__device__ __forceinline__ half_raw float_to_half(float f) {\n"
"    half_raw h;\n"
"    asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(f));\n"
"    return h;\n"
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
"/* ---- 6. kv_cache_store_f16: convert F32 K/V to F16 and write at position ---- */\n"
"__global__ void kv_cache_store_f16(half_raw *key_cache, half_raw *value_cache,\n"
"                                    const float *k, const float *v,\n"
"                                    int position, int kv_dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < kv_dim) {\n"
"        key_cache[(size_t)position * kv_dim + i] = float_to_half(k[i]);\n"
"        value_cache[(size_t)position * kv_dim + i] = float_to_half(v[i]);\n"
"    }\n"
"}\n"
"\n"
"/* ---- 6b. kv_cache_store_q8: quantize F32 K/V to INT8 with per-head scale and store ---- */\n"
"__global__ void kv_cache_store_q8(signed char *key_cache_q8, signed char *value_cache_q8,\n"
"                                    float *key_scale, float *value_scale,\n"
"                                    const float *k, const float *v,\n"
"                                    int position, int kv_dim, int n_kv_heads, int head_dim) {\n"
"    int h = blockIdx.x;\n"
"    int t = threadIdx.x;\n"
"    if (h >= n_kv_heads || t >= head_dim) return;\n"
"    int i = h * head_dim + t;\n"
"    if (i >= kv_dim) return;\n"
"    float kv = k[i], vv = v[i];\n"
"    float ak = fabsf(kv), av = fabsf(vv);\n"
"    for (int o = 16; o > 0; o >>= 1) { ak = fmaxf(ak, __shfl_xor_sync(0xffffffff, ak, o));\n"
"                                         av = fmaxf(av, __shfl_xor_sync(0xffffffff, av, o)); }\n"
"    if (t == 0) { key_scale[(size_t)position * n_kv_heads + h] = ak / 127.0f;\n"
"                  value_scale[(size_t)position * n_kv_heads + h] = av / 127.0f; }\n"
"    float iks = ak > 0.0f ? 127.0f / ak : 0.0f;\n"
"    float ivs = av > 0.0f ? 127.0f / av : 0.0f;\n"
"    int qk = (int)rintf(kv * iks); qk = qk < -127 ? -127 : (qk > 127 ? 127 : qk);\n"
"    int qv = (int)rintf(vv * ivs); qv = qv < -127 ? -127 : (qv > 127 ? 127 : qv);\n"
"    key_cache_q8[(size_t)position * kv_dim + i] = (signed char)qk;\n"
"    value_cache_q8[(size_t)position * kv_dim + i] = (signed char)qv;\n"
"}\n"
"\n"
"/* ---- 7q. attn_decode_q8: Flash-style decode attention reading INT8 quantized KV cache ---- */\n"
"__global__ void attn_decode_q8(float *out, const float *q,\n"
"                                 const signed char *key_cache_q8, const signed char *value_cache_q8,\n"
"                                 const float *key_scale, const float *value_scale,\n"
"                                 int n_heads, int n_kv_heads, int head_dim,\n"
"                                 int kv_dim, int seq_len, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x, nthreads = blockDim.x;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"    const float *q_h = q + h * head_dim;\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    float *scores = smem;\n"
"    float local_max = -1e30f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        const signed char *k_t = key_cache_q8 + (size_t)t * kv_dim + kv_h * head_dim;\n"
"        float ks = key_scale[(size_t)t * n_kv_heads + kv_h];\n"
"        float s = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) s += q_h[d] * ((float)k_t[d] * ks);\n"
"        s *= scale;\n"
"        scores[t] = s;\n"
"        if (s > local_max) local_max = s;\n"
"    }\n"
"    for (int off = 16; off > 0; off >>= 1)\n"
"        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off));\n"
"    __shared__ float wm[8];\n"
"    if (lane == 0) wm[warp_id] = local_max;\n"
"    __syncthreads();\n"
"    if (tid == 0) { float m = wm[0]; for (int w = 1; w < (nthreads+31)/32; w++) if(wm[w]>m) m=wm[w]; wm[0]=m; }\n"
"    __syncthreads();\n"
"    float max_val = wm[0];\n"
"    float local_sum = 0.0f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        float e = expf(scores[t] - max_val); scores[t] = e; local_sum += e; }\n"
"    for (int off = 16; off > 0; off >>= 1)\n"
"        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);\n"
"    __shared__ float ws[8];\n"
"    if (lane == 0) ws[warp_id] = local_sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) { float s = 0.0f; for (int w = 0; w < (nthreads+31)/32; w++) s+=ws[w]; ws[0]=1.0f/s; }\n"
"    __syncthreads();\n"
"    float inv_sum = ws[0];\n"
"    for (int t = tid; t < seq_len; t += nthreads) scores[t] *= inv_sum;\n"
"    __syncthreads();\n"
"    float *out_h = out + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += nthreads) {\n"
"        float acc = 0.0f;\n"
"        for (int t = 0; t < seq_len; t++)\n"
"            acc += scores[t] * ((float)value_cache_q8[(size_t)t * kv_dim + kv_h * head_dim + d] *\n"
"                                value_scale[(size_t)t * n_kv_heads + kv_h]);\n"
"        out_h[d] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* Device-pointer variant for CUDA graph capture: position read from device memory */\n"
"__global__ void rope_neox_f32_ptr(float *vec, int n_heads, int head_dim,\n"
"                                   const int *pos_ptr,\n"
"                                   float freq_base, int n_rope_pairs) {\n"
"    int pos = *pos_ptr;\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half_dim = head_dim / 2;\n"
"    if (j >= half_dim) return;\n"
"    if (n_rope_pairs > 0 && j >= n_rope_pairs) return;\n"
"    int pair_off = (n_rope_pairs > 0 && n_rope_pairs < half_dim) ? n_rope_pairs : half_dim;\n"
"    int rope_dim = (n_rope_pairs > 0) ? 2 * n_rope_pairs : head_dim;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)rope_dim);\n"
"    float theta = (float)pos * freq;\n"
"    float cos_t = cosf(theta);\n"
"    float sin_t = sinf(theta);\n"
"    float *vx = vec + h * head_dim;\n"
"    float a = vx[j];\n"
"    float b = vx[j + pair_off];\n"
"    vx[j] = a * cos_t - b * sin_t;\n"
"    vx[j + pair_off] = a * sin_t + b * cos_t;\n"
"}\n"
"\n"
"/* Device-pointer variant of attn_decode_f32 for graph capture */\n"
"__global__ void attn_decode_f32_ptr(float *out, const float *q,\n"
"                                     const half_raw *key_cache, const half_raw *value_cache,\n"
"                                     int n_heads, int n_kv_heads, int head_dim,\n"
"                                     int kv_dim, const int *seq_len_ptr, float scale) {\n"
"    int seq_len = *seq_len_ptr;\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"    const float *q_h = q + h * head_dim;\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    float *scores = smem;\n"
"    float local_max = -1e30f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        const half_raw *k_t = key_cache + (size_t)t * kv_dim + kv_h * head_dim;\n"
"        float s = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) s += q_h[d] * half_to_float(k_t[d]);\n"
"        s *= scale;\n"
"        scores[t] = s;\n"
"        if (s > local_max) local_max = s;\n"
"    }\n"
"    for (int off = 16; off > 0; off >>= 1)\n"
"        local_max = max(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off));\n"
"    if ((tid & 0x1f) == 0) smem[0] = local_max;\n"
"    __syncthreads();\n"
"    local_max = smem[0];\n"
"    __syncthreads();\n"
"    float local_sum = 0.0f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        scores[t] = __expf(scores[t] - local_max);\n"
"        local_sum += scores[t];\n"
"    }\n"
"    for (int off = 16; off > 0; off >>= 1)\n"
"        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);\n"
"    if ((tid & 0x1f) == 0) smem[0] = local_sum;\n"
"    __syncthreads();\n"
"    float inv_sum = 1.0f / smem[0];\n"
"    float *out_h = out + h * head_dim;\n"
"    float acc = 0.0f;\n"
"    for (int d = tid; d < head_dim; d += nthreads) {\n"
"        acc = 0.0f;\n"
"        for (int t = 0; t < seq_len; t++)\n"
"            acc += scores[t] * half_to_float(value_cache[(size_t)t * kv_dim + kv_h * head_dim + d]);\n"
"        out_h[d] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* Device-pointer variant of kv_cache_store_f16 */\n"
"__global__ void kv_cache_store_f16_ptr(half_raw *key_cache, half_raw *value_cache,\n"
"                                        const float *k, const float *v,\n"
"                                        const int *pos_ptr, int kv_dim) {\n"
"    int position = *pos_ptr;\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < kv_dim) {\n"
"        key_cache[(size_t)position * kv_dim + i] = float_to_half(k[i]);\n"
"        value_cache[(size_t)position * kv_dim + i] = float_to_half(v[i]);\n"
"    }\n"
"}\n"
"\n"
"/* ---- 7. attn_decode_f32: Flash-style decode attention with online softmax ---- */\n"
"/* Fuses QK scoring + softmax + V accumulation into 2 passes over KV cache. */\n"
"/* Pass 1: compute scores + online max/sum. Pass 2: V-weighted accumulation. */\n"
"/* Grid: n_heads, blockDim: 256, shared: max_seq_len * sizeof(float) */\n"
"/* Original: takes seq_len by value; _ptr variant takes const int* for graph capture. */\n"
"__global__ void attn_decode_f32(float *out, const float *q,\n"
"                                 const half_raw *key_cache, const half_raw *value_cache,\n"
"                                 int n_heads, int n_kv_heads, int head_dim,\n"
"                                 int kv_dim, int seq_len, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"    const float *q_h = q + h * head_dim;\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"\n"
"    /* Pass 1: QK scores with fused online softmax max+sum */\n"
"    float *scores = smem;\n"
"    float local_max = -1e30f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        const half_raw *k_t = key_cache + (size_t)t * kv_dim + kv_h * head_dim;\n"
"        float s = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) s += q_h[d] * half_to_float(k_t[d]);\n"
"        s *= scale;\n"
"        scores[t] = s;\n"
"        if (s > local_max) local_max = s;\n"
"    }\n"
"    /* Cross-warp max reduction */\n"
"    for (int off = 16; off > 0; off >>= 1)\n"
"        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off));\n"
"    __shared__ float wm[8];\n"
"    if (lane == 0) wm[warp_id] = local_max;\n"
"    __syncthreads();\n"
"    if (tid == 0) { float m = wm[0]; for (int w = 1; w < (nthreads+31)/32; w++) if(wm[w]>m) m=wm[w]; wm[0]=m; }\n"
"    __syncthreads();\n"
"    float max_val = wm[0];\n"
"\n"
"    /* Exp + sum (fused) */\n"
"    float local_sum = 0.0f;\n"
"    for (int t = tid; t < seq_len; t += nthreads) {\n"
"        float e = expf(scores[t] - max_val);\n"
"        scores[t] = e;\n"
"        local_sum += e;\n"
"    }\n"
"    for (int off = 16; off > 0; off >>= 1)\n"
"        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);\n"
"    __shared__ float ws[8];\n"
"    if (lane == 0) ws[warp_id] = local_sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) { float s = 0.0f; for (int w = 0; w < (nthreads+31)/32; w++) s+=ws[w]; ws[0]=1.0f/s; }\n"
"    __syncthreads();\n"
"    float inv_sum = ws[0];\n"
"    for (int t = tid; t < seq_len; t += nthreads) scores[t] *= inv_sum;\n"
"    __syncthreads();\n"
"\n"
"    /* Pass 2: V accumulation */\n"
"    float *out_h = out + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += nthreads) {\n"
"        float acc = 0.0f;\n"
"        for (int t = 0; t < seq_len; t++)\n"
"            acc += scores[t] * half_to_float(value_cache[(size_t)t * kv_dim + kv_h * head_dim + d]);\n"
"        out_h[d] = acc;\n"
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
"/* ---- 9c. batch_scale_add_sigmoid_f32: dst[t,i] += sigmoid(gate[t]) * src[t,i] ---- */\n"
"__global__ void batch_scale_add_sigmoid_f32(float *dst, const float *src, const float *gate,\n"
"                                           int n_tokens, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tokens * dim;\n"
"    if (i < total) {\n"
"        int t = i / dim;\n"
"        float g = gate[t];\n"
"        float s = 1.0f / (1.0f + expf(-g));\n"
"        dst[i] += s * src[i];\n"
"    }\n"
"}\n"
"\n"
"/* ---- 9d. matvec_f32_f32: F32 matrix x F32 vector -> F32 (for MoE router) ---- */\n"
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
"/* ---- 9e. batch_matvec_f32_f32: batched F32 matrix x F32 vectors -> F32 ---- */\n"
"__global__ void batch_matvec_f32_f32(float *output, const float *mat, const float *input,\n"
"                                    int out_dim, int in_dim, int n_tokens) {\n"
"    int row = blockIdx.x;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int tid = threadIdx.x;\n"
"    const float *row_ptr = mat + (size_t)row * in_dim;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int j = tid; j < in_dim; j += blockDim.x) {\n"
"        sum += row_ptr[j] * x[j];\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    }\n"
"    __shared__ float warp_sums[8];\n"
"    int warp_id = tid / 32;\n"
"    int lane = tid & 31;\n"
"    if (lane == 0) warp_sums[warp_id] = sum;\n"
"    __syncthreads();\n"
"    if (tid < 32) {\n"
"        float total = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0.0f;\n"
"        for (int offset = 16; offset > 0; offset >>= 1) {\n"
"            total += __shfl_down_sync(0xFFFFFFFF, total, offset);\n"
"        }\n"
"        if (tid == 0) output[(size_t)token * out_dim + row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* ---- dp4a: INT8 dot product via inline PTX (sm_61+) ---- */\n"
"__device__ __forceinline__ int dp4a_s8(int a, int b, int c) {\n"
"    asm(\"dp4a.s32.s32 %0, %1, %2, %0;\" : \"+r\"(c) : \"r\"(a), \"r\"(b));\n"
"    return c;\n"
"}\n"
"\n"
"/* Pack 4 consecutive bytes into an int32 (little-endian, byte 0 = LSB). */\n"
"/* Works for any alignment; used for Q3_K/Q6_K whose super-blocks have odd size. */\n"
"__device__ __forceinline__ int load_u8x4(const unsigned char *p) {\n"
"    return (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);\n"
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
"/* ---- 11a2. matvec_q8_0_q8_1_dp4a: Q8_0 weight x Q8_1 (block-wise) activation ---- */\n"
"/* Like matvec_q8_0_dp4a but the activation carries a PER-32-BLOCK scale (q8_1)\n"
"   instead of one per-row absmax scale. Per-row quant crushes small activations\n"
"   when any of the 3840 values is an outlier (common post-residual), which made\n"
"   the Q8_0 LM head + attn projections the dominant dp4a error (~1.8 rel_L2 alone\n"
"   on the 12B Q6_K model). Block-wise matches llama.cpp and drops it to ~0.07.\n"
"   Q8_0 is symmetric (zero-point 0) so no activation-sum term is needed.\n"
"   8 warps/block, 1 warp/row, grid = ceil(n_rows/8). Q8_0 block = 36B (f16 d@0,\n"
"   32 int8 @4); q8_1 block = 36B (32 int8 @0, f16 d8@32, f16 sum@34). */\n"
"__global__ void matvec_q8_0_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                        const unsigned char *x_q81,\n"
"                                        int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 32;\n"
"    int row_bytes = nb * 36;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sumf = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 36;\n"
"        float dw = half_to_float(*(const half_raw *)bp);\n"
"        const int *w4 = (const int *)(bp + 4);\n"
"        const unsigned char *q81 = x_q81 + (size_t)b * 36;\n"
"        const int *u = (const int *)q81;\n"
"        float d8 = half_to_float(*(const half_raw *)(q81 + 32));\n"
"        int acc = 0;\n"
"        #pragma unroll\n"
"        for (int k = 0; k < 8; k++) acc = dp4a_s8(w4[k], u[k], acc);\n"
"        sumf += dw * d8 * (float)acc;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);\n"
"    if (lane == 0) dst[row] = sumf;\n"
"}\n"
"\n"
"/* ---- 11b. matvec_q8_0_dp4a_fused2: Fused gate+up Q8_0 × INT8 via dp4a ---- */\n"
"__global__ void matvec_q8_0_dp4a_fused2(float *dst1, float *dst2,\n"
"                                         const unsigned char *mat1, const unsigned char *mat2,\n"
"                                         const signed char *x_q, const float *x_scale_ptr,\n"
"                                         int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int nb = n_cols / 32;\n"
"    int row_bytes = nb * 36;\n"
"    const unsigned char *rp1 = mat1 + (size_t)row * row_bytes;\n"
"    const unsigned char *rp2 = mat2 + (size_t)row * row_bytes;\n"
"    float x_scale = *x_scale_ptr;\n"
"    float sum1 = 0.0f, sum2 = 0.0f;\n"
"    for (int b = tid; b < nb; b += nthreads) {\n"
"        const unsigned char *bp1 = rp1 + b * 36;\n"
"        const unsigned char *bp2 = rp2 + b * 36;\n"
"        float d1 = half_to_float(*(const half_raw *)bp1);\n"
"        float d2 = half_to_float(*(const half_raw *)bp2);\n"
"        const int *w1 = (const int *)(bp1 + 4);\n"
"        const int *w2 = (const int *)(bp2 + 4);\n"
"        const int *x4 = (const int *)(x_q + b * 32);\n"
"        int a1 = 0, a2 = 0;\n"
"        a1 = dp4a_s8(w1[0], x4[0], a1); a2 = dp4a_s8(w2[0], x4[0], a2);\n"
"        a1 = dp4a_s8(w1[1], x4[1], a1); a2 = dp4a_s8(w2[1], x4[1], a2);\n"
"        a1 = dp4a_s8(w1[2], x4[2], a1); a2 = dp4a_s8(w2[2], x4[2], a2);\n"
"        a1 = dp4a_s8(w1[3], x4[3], a1); a2 = dp4a_s8(w2[3], x4[3], a2);\n"
"        a1 = dp4a_s8(w1[4], x4[4], a1); a2 = dp4a_s8(w2[4], x4[4], a2);\n"
"        a1 = dp4a_s8(w1[5], x4[5], a1); a2 = dp4a_s8(w2[5], x4[5], a2);\n"
"        a1 = dp4a_s8(w1[6], x4[6], a1); a2 = dp4a_s8(w2[6], x4[6], a2);\n"
"        a1 = dp4a_s8(w1[7], x4[7], a1); a2 = dp4a_s8(w2[7], x4[7], a2);\n"
"        sum1 += (float)a1 * d1 * x_scale;\n"
"        sum2 += (float)a2 * d2 * x_scale;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        sum1 += __shfl_down_sync(0xFFFFFFFF, sum1, offset);\n"
"        sum2 += __shfl_down_sync(0xFFFFFFFF, sum2, offset);\n"
"    }\n"
"    __shared__ float ws1[8], ws2[8];\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    if (lane == 0) { ws1[warp_id] = sum1; ws2[warp_id] = sum2; }\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float t1 = 0.0f, t2 = 0.0f;\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < nw; w++) { t1 += ws1[w]; t2 += ws2[w]; }\n"
"        dst1[row] = t1; dst2[row] = t2;\n"
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
"/* ---- 12c. matvec_q8_0_f32_fused2: Fused gate+up Q8_0 matvec ---- */\n"
"/* Computes dst1[row] = mat1[row] . x and dst2[row] = mat2[row] . x simultaneously. */\n"
"/* Each block handles one row, reading input x from shared memory. */\n"
"__global__ void matvec_q8_0_f32_fused2(float *dst1, float *dst2,\n"
"                                        const unsigned char *mat1, const unsigned char *mat2,\n"
"                                        const float *x, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int n_blocks_per_row = n_cols / 32;\n"
"    int row_bytes = n_blocks_per_row * 36;\n"
"    const unsigned char *rp1 = mat1 + (size_t)row * row_bytes;\n"
"    const unsigned char *rp2 = mat2 + (size_t)row * row_bytes;\n"
"    float sum1 = 0.0f, sum2 = 0.0f;\n"
"    for (int b = tid; b < n_blocks_per_row; b += nthreads) {\n"
"        const unsigned char *bp1 = rp1 + b * 36;\n"
"        const unsigned char *bp2 = rp2 + b * 36;\n"
"        float d1 = half_to_float(*(const half_raw *)bp1);\n"
"        float d2 = half_to_float(*(const half_raw *)bp2);\n"
"        const signed char *qs1 = (const signed char *)(bp1 + 4);\n"
"        const signed char *qs2 = (const signed char *)(bp2 + 4);\n"
"        const float *xb = x + b * 32;\n"
"        float p1 = 0.0f, p2 = 0.0f;\n"
"        for (int k = 0; k < 32; k++) {\n"
"            float xv = xb[k];\n"
"            p1 += (float)qs1[k] * xv;\n"
"            p2 += (float)qs2[k] * xv;\n"
"        }\n"
"        sum1 += p1 * d1;\n"
"        sum2 += p2 * d2;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        sum1 += __shfl_down_sync(0xFFFFFFFF, sum1, offset);\n"
"        sum2 += __shfl_down_sync(0xFFFFFFFF, sum2, offset);\n"
"    }\n"
"    __shared__ float ws1[8], ws2[8];\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    if (lane == 0) { ws1[warp_id] = sum1; ws2[warp_id] = sum2; }\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float t1 = 0.0f, t2 = 0.0f;\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < nw; w++) { t1 += ws1[w]; t2 += ws2[w]; }\n"
"        dst1[row] = t1; dst2[row] = t2;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 12d. matvec_f16_f32_fused2: Fused gate+up F16 matvec ---- */\n"
"__global__ void matvec_f16_f32_fused2(float *dst1, float *dst2,\n"
"                                       const half_raw *mat1, const half_raw *mat2,\n"
"                                       const float *x, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    const half_raw *rp1 = mat1 + (size_t)row * n_cols;\n"
"    const half_raw *rp2 = mat2 + (size_t)row * n_cols;\n"
"    float sum1 = 0.0f, sum2 = 0.0f;\n"
"    for (int i = tid; i < n_cols; i += nthreads) {\n"
"        float xv = x[i];\n"
"        sum1 += half_to_float(rp1[i]) * xv;\n"
"        sum2 += half_to_float(rp2[i]) * xv;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        sum1 += __shfl_down_sync(0xFFFFFFFF, sum1, offset);\n"
"        sum2 += __shfl_down_sync(0xFFFFFFFF, sum2, offset);\n"
"    }\n"
"    __shared__ float ws1[8], ws2[8];\n"
"    int warp_id = tid / 32, lane = tid % 32;\n"
"    if (lane == 0) { ws1[warp_id] = sum1; ws2[warp_id] = sum2; }\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float t1 = 0.0f, t2 = 0.0f;\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < nw; w++) { t1 += ws1[w]; t2 += ws2[w]; }\n"
"        dst1[row] = t1; dst2[row] = t2;\n"
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
"/* Multi-warp: 8 warps per block, each warp handles one row */\n"
"/* Q2_K block: 84 bytes = scales[16] + qs[64] + d(f16) + dmin(f16), 256 elements */\n"
"__global__ void matvec_q2_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 84;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
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
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- dequant_q2_K_to_f16: Q2_K weight -> F16, element order matching matvec_q2_K_f32 ---- */\n"
"/* For the gemma4 dequant->F16->cuBLAS batched-prefill path. 8 warps/block, 1 warp/row. */\n"
"__global__ void dequant_q2_K_to_f16(half_raw *dst, const unsigned char *mat,\n"
"                                      int rows, int cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= rows) return;\n"
"    int nb = cols / 256;\n"
"    int row_bytes = nb * 84;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    half_raw *dst_row = dst + (size_t)row * cols;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 84;\n"
"        const unsigned char *scales = bp;\n"
"        const unsigned char *qs = bp + 16;\n"
"        float d = half_to_float(*(const half_raw *)(bp + 80));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 82));\n"
"        half_raw *d_b = dst_row + b * 256;\n"
"        int is = 0, yi = 0;\n"
"        for (int n0 = 0; n0 < 2; n0++) {\n"
"            for (int j = 0; j < 4; j++) {\n"
"                int shift = j * 2;\n"
"                unsigned char sc = scales[is++];\n"
"                float dl = d * (sc & 0xF);\n"
"                float ml = dmin * (sc >> 4);\n"
"                for (int l = 0; l < 16; l++)\n"
"                    d_b[yi++] = float_to_half(dl * ((qs[l] >> shift) & 3) - ml);\n"
"                sc = scales[is++];\n"
"                dl = d * (sc & 0xF);\n"
"                ml = dmin * (sc >> 4);\n"
"                for (int l = 0; l < 16; l++)\n"
"                    d_b[yi++] = float_to_half(dl * ((qs[l + 16] >> shift) & 3) - ml);\n"
"            }\n"
"            qs += 32;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- 14b. quantize_f32_to_q8_1: per-32 block F32 -> Q8_1 (int8 qs[32] + half2(d,sum)) ---- */\n"
"/* One warp per Q8_1 block; launch with blockDim=(32, N), grid=(n_blocks/N). */\n"
"/* Output format: 36 bytes per block = 32 int8 qs + half d + half sum. */\n"
"__global__ void quantize_f32_to_q8_1(unsigned char *dst, const float *src, int n) {\n"
"    int block_id = blockIdx.x * blockDim.y + threadIdx.y;\n"
"    int n_blocks = n / 32;\n"
"    if (block_id >= n_blocks) return;\n"
"    int lane = threadIdx.x;\n"
"    float v = src[block_id * 32 + lane];\n"
"    float amax = fabsf(v);\n"
"    float sum = v;\n"
"    for (int offset = 16; offset > 0; offset >>= 1) {\n"
"        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));\n"
"        sum  += __shfl_xor_sync(0xFFFFFFFF, sum,  offset);\n"
"    }\n"
"    float d = amax / 127.0f;\n"
"    float id_scale = (amax > 0.0f) ? 127.0f / amax : 0.0f;\n"
"    int iv = __float2int_rn(v * id_scale);\n"
"    if (iv >  127) iv =  127;\n"
"    if (iv < -128) iv = -128;\n"
"    unsigned char *bp = dst + (size_t)block_id * 36;\n"
"    ((signed char *)bp)[lane] = (signed char)iv;\n"
"    if (lane == 0) {\n"
"        half_raw *ds = (half_raw *)(bp + 32);\n"
"        ds[0] = float_to_half(d);\n"
"        ds[1] = float_to_half(sum);\n"
"    }\n"
"}\n"
"\n"
"/* ---- 14c. matvec_q2_K_q8_1_dp4a: Q2_K weight x Q8_1 activation via dp4a ---- */\n"
"/* Warp per row, 8 warps per block, grid = ceil(n_rows/8). */\n"
"/* Each super-block has 256 elements = 8 Q8_1 blocks.                         */\n"
"/* Per Q8_1 block i: 2 sub-blocks of 16 weights using scales[2*i], scales[2*i+1]. */\n"
"/* Sub-block maps to n0=i/4 (half of super-block), j=i%4 (shift=j*2).          */\n"
"__global__ void matvec_q2_K_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                        const unsigned char *x_q81,\n"
"                                        int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 84;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 84;\n"
"        const unsigned char *scales = bp;\n"
"        const unsigned char *qs = bp + 16;\n"
"        float d    = half_to_float(*(const half_raw *)(bp + 80));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 82));\n"
"        const unsigned char *bq81_base = x_q81 + (size_t)b * 8 * 36;\n"
"        float sumf_d = 0.0f, sumf_m = 0.0f;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int n0 = i >> 2;\n"
"            int jj = i & 3;\n"
"            int shift = jj * 2;\n"
"            const int *qs_w = (const int *)(qs + n0 * 32);\n"
"            const unsigned char *q81 = bq81_base + i * 36;\n"
"            const int *u_w = (const int *)q81;\n"
"            float d8 = half_to_float(*(const half_raw *)(q81 + 32));\n"
"            int sc0 = scales[2*i];\n"
"            int sc1 = scales[2*i + 1];\n"
"            int dlow0 = sc0 & 0xF;\n"
"            int dlow1 = sc1 & 0xF;\n"
"            int mhi0  = sc0 >> 4;\n"
"            int mhi1  = sc1 >> 4;\n"
"            int m0w = mhi0 | (mhi0 << 8); m0w |= m0w << 16;\n"
"            int m1w = mhi1 | (mhi1 << 8); m1w |= m1w << 16;\n"
"            int acc_d0 = 0, acc_m0 = 0, acc_d1 = 0, acc_m1 = 0;\n"
"            #pragma unroll\n"
"            for (int k = 0; k < 4; k++) {\n"
"                int vi0 = (qs_w[k]     >> shift) & 0x03030303;\n"
"                int vi1 = (qs_w[k + 4] >> shift) & 0x03030303;\n"
"                acc_d0 = dp4a_s8(vi0, u_w[k],     acc_d0);\n"
"                acc_m0 = dp4a_s8(m0w, u_w[k],     acc_m0);\n"
"                acc_d1 = dp4a_s8(vi1, u_w[k + 4], acc_d1);\n"
"                acc_m1 = dp4a_s8(m1w, u_w[k + 4], acc_m1);\n"
"            }\n"
"            sumf_d += d8 * ((float)acc_d0 * (float)dlow0 + (float)acc_d1 * (float)dlow1);\n"
"            sumf_m += d8 * ((float)acc_m0 + (float)acc_m1);\n"
"        }\n"
"        sum += d * sumf_d - dmin * sumf_m;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- 15. matvec_q3_K_f32: Q3_K matrix x F32 vector -> F32 ---- */\n"
"/* Multi-warp: 8 warps per block, each warp handles one row */\n"
"/* Q3_K block: 110 bytes = hmask[32] + qs[64] + scales[12] + d(f16), 256 elements */\n"
"__global__ void matvec_q3_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
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
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- 15b. matvec_q3_K_q8_1_dp4a: Q3_K weight x Q8_1 activation via dp4a ---- */\n"
"/* Signed weight = ((qs>>shift)&3) - 4*(~hm_bit) = vil + 4*vih - 4.           */\n"
"/* Per sub-block, compute: dot = dp4a(vil,u,0) + 4*dp4a(vih,u,0) - 4*sum(u).  */\n"
"__global__ void matvec_q3_K_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                        const unsigned char *x_q81,\n"
"                                        int n_rows, int n_cols) {\n"
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
"        const unsigned char *hm = bp;\n"
"        const unsigned char *qs = bp + 32;\n"
"        const unsigned char *raw_sc = bp + 96;\n"
"        float d_all = half_to_float(*(const half_raw *)(bp + 108));\n"
"        unsigned int a0  = raw_sc[0]|(raw_sc[1]<<8)|(raw_sc[2]<<16)|(raw_sc[3]<<24);\n"
"        unsigned int a1  = raw_sc[4]|(raw_sc[5]<<8)|(raw_sc[6]<<16)|(raw_sc[7]<<24);\n"
"        unsigned int tmp = raw_sc[8]|(raw_sc[9]<<8)|(raw_sc[10]<<16)|(raw_sc[11]<<24);\n"
"        unsigned int km1 = 0x03030303u, km2 = 0x0f0f0f0fu;\n"
"        unsigned int aux[4];\n"
"        aux[0] = (a0 & km2)        | (((tmp >> 0) & km1) << 4);\n"
"        aux[1] = (a1 & km2)        | (((tmp >> 2) & km1) << 4);\n"
"        aux[2] = ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4);\n"
"        aux[3] = ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4);\n"
"        const signed char *scales = (const signed char *)aux;\n"
"        const unsigned char *bq81_base = x_q81 + (size_t)b * 8 * 36;\n"
"        float partial = 0.0f;\n"
"        int is = 0;\n"
"        for (int n0 = 0; n0 < 2; n0++) {\n"
"            const unsigned char *qs_p = qs + n0 * 32;\n"
"            for (int j = 0; j < 4; j++) {\n"
"                int shift = j * 2;\n"
"                int m_bit_idx = n0 * 4 + j;\n"
"                int i = n0 * 4 + j;\n"
"                const unsigned char *q81 = bq81_base + i * 36;\n"
"                const int *u_w = (const int *)q81;\n"
"                float d8 = half_to_float(*(const half_raw *)(q81 + 32));\n"
"                int sc0_s = (int)scales[is++] - 32;\n"
"                int sc1_s = (int)scales[is++] - 32;\n"
"                int acc_q0 = 0, acc_h0 = 0, acc_u0 = 0;\n"
"                int acc_q1 = 0, acc_h1 = 0, acc_u1 = 0;\n"
"                #pragma unroll\n"
"                for (int k = 0; k < 4; k++) {\n"
"                    int qs0 = load_u8x4(qs_p + k * 4);\n"
"                    int qs1 = load_u8x4(qs_p + 16 + k * 4);\n"
"                    int hm0 = load_u8x4(hm   + k * 4);\n"
"                    int hm1 = load_u8x4(hm   + 16 + k * 4);\n"
"                    int vil0 = (qs0 >> shift) & 0x03030303;\n"
"                    int vih0 = (hm0 >> m_bit_idx) & 0x01010101;\n"
"                    int vil1 = (qs1 >> shift) & 0x03030303;\n"
"                    int vih1 = (hm1 >> m_bit_idx) & 0x01010101;\n"
"                    acc_q0 = dp4a_s8(vil0, u_w[k],     acc_q0);\n"
"                    acc_h0 = dp4a_s8(vih0, u_w[k],     acc_h0);\n"
"                    acc_u0 = dp4a_s8(0x01010101, u_w[k],     acc_u0);\n"
"                    acc_q1 = dp4a_s8(vil1, u_w[k + 4], acc_q1);\n"
"                    acc_h1 = dp4a_s8(vih1, u_w[k + 4], acc_h1);\n"
"                    acc_u1 = dp4a_s8(0x01010101, u_w[k + 4], acc_u1);\n"
"                }\n"
"                int signed0 = acc_q0 + 4 * acc_h0 - 4 * acc_u0;\n"
"                int signed1 = acc_q1 + 4 * acc_h1 - 4 * acc_u1;\n"
"                partial += d8 * ((float)signed0 * (float)sc0_s + (float)signed1 * (float)sc1_s);\n"
"            }\n"
"        }\n"
"        sum += d_all * partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- 16. matvec_q4_K_f32: Q4_K matrix x F32 vector -> F32 ---- */\n"
"/* Multi-warp: 8 warps per block, each warp handles one row */\n"
"/* Q4_K block: 144 bytes = d(f16) + dmin(f16) + scales[12] + qs[128], 256 elements */\n"
"__global__ void matvec_q4_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
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
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- 16b. matvec_q4_K_q8_1_dp4a: Q4_K weight x Q8_1 activation via dp4a ---- */\n"
"/* Warp-per-row (8 warps per block). Per super-block: 8 Q8_1 blocks, 4 groups      */\n"
"/* of 32 qs bytes carrying sub-block 2j (low nibble) and 2j+1 (high nibble).       */\n"
"/* sumf_d = d * sum(sc[i] * d8[i] * dp4a(v_i, u_i, 0));                             */\n"
"/* sumf_m = dmin * sum(mv[i] * sum8[i]),  where sum8[i] = sum of xi in q8_1 block i */\n"
"__global__ void matvec_q4_K_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                        const unsigned char *x_q81,\n"
"                                        int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 144;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sumf_d_all = 0.0f;\n"
"    float sumf_m_all = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 144;\n"
"        float d    = half_to_float(*(const half_raw *)(bp));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *sc = bp + 4;\n"
"        const unsigned char *qs = bp + 16;\n"
"        unsigned char svs[8], mvs[8];\n"
"        svs[0] = sc[0] & 0x3F; svs[1] = sc[1] & 0x3F;\n"
"        svs[2] = sc[2] & 0x3F; svs[3] = sc[3] & 0x3F;\n"
"        mvs[0] = sc[4] & 0x3F; mvs[1] = sc[5] & 0x3F;\n"
"        mvs[2] = sc[6] & 0x3F; mvs[3] = sc[7] & 0x3F;\n"
"        svs[4] = (sc[8]  & 0x0F) | (((sc[0] >> 6) & 0x03) << 4);\n"
"        svs[5] = (sc[9]  & 0x0F) | (((sc[1] >> 6) & 0x03) << 4);\n"
"        svs[6] = (sc[10] & 0x0F) | (((sc[2] >> 6) & 0x03) << 4);\n"
"        svs[7] = (sc[11] & 0x0F) | (((sc[3] >> 6) & 0x03) << 4);\n"
"        mvs[4] = (sc[8]  >> 4)   | (((sc[4] >> 6) & 0x03) << 4);\n"
"        mvs[5] = (sc[9]  >> 4)   | (((sc[5] >> 6) & 0x03) << 4);\n"
"        mvs[6] = (sc[10] >> 4)   | (((sc[6] >> 6) & 0x03) << 4);\n"
"        mvs[7] = (sc[11] >> 4)   | (((sc[7] >> 6) & 0x03) << 4);\n"
"        const unsigned char *bq81_base = x_q81 + (size_t)b * 8 * 36;\n"
"        float sumf_d_blk = 0.0f;\n"
"        float sumf_m_blk = 0.0f;\n"
"        #pragma unroll\n"
"        for (int j = 0; j < 4; j++) {\n"
"            const int *qs_w = (const int *)(qs + j * 32);\n"
"            const unsigned char *q81_lo = bq81_base + (2 * j)     * 36;\n"
"            const unsigned char *q81_hi = bq81_base + (2 * j + 1) * 36;\n"
"            const int *u_lo = (const int *)q81_lo;\n"
"            const int *u_hi = (const int *)q81_hi;\n"
"            float d8_lo = half_to_float(*(const half_raw *)(q81_lo + 32));\n"
"            float s8_lo = half_to_float(*(const half_raw *)(q81_lo + 34));\n"
"            float d8_hi = half_to_float(*(const half_raw *)(q81_hi + 32));\n"
"            float s8_hi = half_to_float(*(const half_raw *)(q81_hi + 34));\n"
"            int acc_lo = 0;\n"
"            int acc_hi = 0;\n"
"            #pragma unroll\n"
"            for (int k = 0; k < 8; k++) {\n"
"                int qb = qs_w[k];\n"
"                int v_lo = qb & 0x0F0F0F0F;\n"
"                int v_hi = (qb >> 4) & 0x0F0F0F0F;\n"
"                acc_lo = dp4a_s8(v_lo, u_lo[k], acc_lo);\n"
"                acc_hi = dp4a_s8(v_hi, u_hi[k], acc_hi);\n"
"            }\n"
"            sumf_d_blk += d8_lo * (float)acc_lo * (float)svs[2*j]\n"
"                        + d8_hi * (float)acc_hi * (float)svs[2*j+1];\n"
"            sumf_m_blk += s8_lo * (float)mvs[2*j] + s8_hi * (float)mvs[2*j+1];\n"
"        }\n"
"        sumf_d_all += d    * sumf_d_blk;\n"
"        sumf_m_all += dmin * sumf_m_blk;\n"
"    }\n"
"    float sum = sumf_d_all - sumf_m_all;\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- 17. matvec_q6_K_f32: Q6_K matrix x F32 vector -> F32 ---- */\n"
"/* Q6_K block: 210 bytes = ql[128] + qh[64] + scales[16] + d(f16), 256 elements */\n"
"__global__ void matvec_q6_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
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
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- 17a. matvec_q6_K_q8_1_dp4a: Q6_K weight x Q8_1 activation via dp4a ---- */\n"
"/* Q6_K super-block: 16 sub-sub-blocks of 16 elements, each with int8 scale.     */\n"
"/* Q6_K block bytes = 210 (not 4-aligned) — use load_u8x4 for all weight reads. */\n"
"/* Q8_1 blocks are 36 bytes each, cuMemAlloc-aligned, safe for int cast.         */\n"
"/* Unsigned weight vi = (ql&0xF) | ((qh&3)<<4), range [0..63]; signed = vi - 32. */\n"
"__global__ void matvec_q6_K_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                        const unsigned char *x_q81,\n"
"                                        int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 210;\n"
"        const unsigned char *ql = bp;\n"
"        const unsigned char *qh = bp + 128;\n"
"        const signed char *sc = (const signed char *)(bp + 192);\n"
"        float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"        const unsigned char *bq81_base = x_q81 + (size_t)b * 8 * 36;\n"
"        float sumf_d = 0.0f;\n"
"        #pragma unroll\n"
"        for (int half = 0; half < 2; half++) {\n"
"            int ql_half_off = half * 64;\n"
"            int qh_half_off = half * 32;\n"
"            int sc_half_off = half * 8;\n"
"            #pragma unroll\n"
"            for (int g = 0; g < 4; g++) {\n"
"                int q81_idx = half * 4 + g;\n"
"                const unsigned char *q81 = bq81_base + q81_idx * 36;\n"
"                const int *u_w = (const int *)q81;\n"
"                float d8 = half_to_float(*(const half_raw *)(q81 + 32));\n"
"                int sc0 = (int)sc[sc_half_off + g * 2 + 0];\n"
"                int sc1 = (int)sc[sc_half_off + g * 2 + 1];\n"
"                int ql_off = ql_half_off + ((g & 1) ? 32 : 0);\n"
"                int shift_q = 2 * g;\n"
"                int use_high = (g >= 2);\n"
"                int acc_r_lo = 0, acc_m_lo = 0;\n"
"                int acc_r_hi = 0, acc_m_hi = 0;\n"
"                #pragma unroll\n"
"                for (int k = 0; k < 8; k++) {\n"
"                    int ql_int = load_u8x4(ql + ql_off + k * 4);\n"
"                    int qh_int = load_u8x4(qh + qh_half_off + k * 4);\n"
"                    int vil = use_high ? ((ql_int >> 4) & 0x0F0F0F0F) : (ql_int & 0x0F0F0F0F);\n"
"                    int vih = ((qh_int >> shift_q) & 0x03030303) << 4;\n"
"                    int vi  = vil | vih;\n"
"                    int u_int = u_w[k];\n"
"                    if (k < 4) {\n"
"                        acc_r_lo = dp4a_s8(vi, u_int, acc_r_lo);\n"
"                        acc_m_lo = dp4a_s8(0x20202020, u_int, acc_m_lo);\n"
"                    } else {\n"
"                        acc_r_hi = dp4a_s8(vi, u_int, acc_r_hi);\n"
"                        acc_m_hi = dp4a_s8(0x20202020, u_int, acc_m_hi);\n"
"                    }\n"
"                }\n"
"                sumf_d += d8 * ((float)sc0 * (float)(acc_r_lo - acc_m_lo)\n"
"                              + (float)sc1 * (float)(acc_r_hi - acc_m_hi));\n"
"            }\n"
"        }\n"
"        sum += d * sumf_d;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- 17b. matvec_q5_K_f32: Q5_K matrix x F32 vector -> F32 ---- */\n"
"/* Q5_K block: 176 bytes = d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128], 256 elements */\n"
"__global__ void matvec_q5_K_f32(float *dst, const unsigned char *mat, const float *x,\n"
"                                  int n_rows, int n_cols) {\n"
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
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- 17c. matvec_q5_K_q8_1_dp4a: Q5_K weight x Q8_1 activation via dp4a ---- */\n"
"/* Signed nibble = (qs>>shift)&0x0F, plus high bit from qh; total weight in [0,31]. */\n"
"/* Per group j (4 groups) we handle sub-blocks 2j (low nibble) and 2j+1 (high nibble). */\n"
"__global__ void matvec_q5_K_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                        const unsigned char *x_q81,\n"
"                                        int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 176;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sumf_d_all = 0.0f;\n"
"    float sumf_m_all = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 176;\n"
"        float d    = half_to_float(*(const half_raw *)(bp));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *sc = bp + 4;\n"
"        const unsigned char *qh = bp + 16;\n"
"        const unsigned char *qs = bp + 48;\n"
"        unsigned char svs[8], mvs[8];\n"
"        svs[0] = sc[0] & 0x3F; svs[1] = sc[1] & 0x3F;\n"
"        svs[2] = sc[2] & 0x3F; svs[3] = sc[3] & 0x3F;\n"
"        mvs[0] = sc[4] & 0x3F; mvs[1] = sc[5] & 0x3F;\n"
"        mvs[2] = sc[6] & 0x3F; mvs[3] = sc[7] & 0x3F;\n"
"        svs[4] = (sc[8]  & 0x0F) | (((sc[0] >> 6) & 0x03) << 4);\n"
"        svs[5] = (sc[9]  & 0x0F) | (((sc[1] >> 6) & 0x03) << 4);\n"
"        svs[6] = (sc[10] & 0x0F) | (((sc[2] >> 6) & 0x03) << 4);\n"
"        svs[7] = (sc[11] & 0x0F) | (((sc[3] >> 6) & 0x03) << 4);\n"
"        mvs[4] = (sc[8]  >> 4)   | (((sc[4] >> 6) & 0x03) << 4);\n"
"        mvs[5] = (sc[9]  >> 4)   | (((sc[5] >> 6) & 0x03) << 4);\n"
"        mvs[6] = (sc[10] >> 4)   | (((sc[6] >> 6) & 0x03) << 4);\n"
"        mvs[7] = (sc[11] >> 4)   | (((sc[7] >> 6) & 0x03) << 4);\n"
"        const int *qh_w = (const int *)qh;\n"
"        const unsigned char *bq81_base = x_q81 + (size_t)b * 8 * 36;\n"
"        float sumf_d_blk = 0.0f;\n"
"        float sumf_m_blk = 0.0f;\n"
"        #pragma unroll\n"
"        for (int j = 0; j < 4; j++) {\n"
"            const int *qs_w = (const int *)(qs + j * 32);\n"
"            const unsigned char *q81_lo = bq81_base + (2 * j)     * 36;\n"
"            const unsigned char *q81_hi = bq81_base + (2 * j + 1) * 36;\n"
"            const int *u_lo = (const int *)q81_lo;\n"
"            const int *u_hi = (const int *)q81_hi;\n"
"            float d8_lo = half_to_float(*(const half_raw *)(q81_lo + 32));\n"
"            float s8_lo = half_to_float(*(const half_raw *)(q81_lo + 34));\n"
"            float d8_hi = half_to_float(*(const half_raw *)(q81_hi + 32));\n"
"            float s8_hi = half_to_float(*(const half_raw *)(q81_hi + 34));\n"
"            int shift_lo = 2 * j;\n"
"            int shift_hi = 2 * j + 1;\n"
"            int acc_lo = 0;\n"
"            int acc_hi = 0;\n"
"            #pragma unroll\n"
"            for (int k = 0; k < 8; k++) {\n"
"                int qb = qs_w[k];\n"
"                int qhb = qh_w[k];\n"
"                int v_lo = (qb & 0x0F0F0F0F) | (((qhb >> shift_lo) & 0x01010101) << 4);\n"
"                int v_hi = ((qb >> 4) & 0x0F0F0F0F) | (((qhb >> shift_hi) & 0x01010101) << 4);\n"
"                acc_lo = dp4a_s8(v_lo, u_lo[k], acc_lo);\n"
"                acc_hi = dp4a_s8(v_hi, u_hi[k], acc_hi);\n"
"            }\n"
"            sumf_d_blk += d8_lo * (float)acc_lo * (float)svs[2*j]\n"
"                        + d8_hi * (float)acc_hi * (float)svs[2*j+1];\n"
"            sumf_m_blk += s8_lo * (float)mvs[2*j] + s8_hi * (float)mvs[2*j+1];\n"
"        }\n"
"        sumf_d_all += d    * sumf_d_blk;\n"
"        sumf_m_all += dmin * sumf_m_blk;\n"
"    }\n"
"    float sum = sumf_d_all - sumf_m_all;\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"/* ---- 18b. embed_q4_K: Q4_K embedding lookup -> F32 ---- */\n"
"__global__ void embed_q4_K(float *dst, const unsigned char *embd_table,\n"
"                             int token_id, int n_embd) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n_embd) return;\n"
"    int nb_per_row = n_embd / 256;\n"
"    int row_bytes = nb_per_row * 144;\n"
"    const unsigned char *row = embd_table + (size_t)token_id * row_bytes;\n"
"    int block_idx = i / 256;\n"
"    int elem = i % 256;\n"
"    const unsigned char *bp = row + block_idx * 144;\n"
"    float d = half_to_float(*(const half_raw *)(bp));\n"
"    float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"    const unsigned char *sc = bp + 4;\n"
"    const unsigned char *qs = bp + 16;\n"
"    /* 256 elements = 4 groups of 64, each group has 2 sub-blocks of 32 */\n"
"    int j = elem / 64;\n"             /* group 0-3 */
"    int within = elem % 64;\n"
"    int sub = within / 32;\n"          /* sub-block 0-1 */
"    int l = within % 32;\n"            /* element within sub-block */
"    int is = j * 2 + sub;\n"
"    unsigned char sv, mv;\n"
"    if (is < 4) { sv = sc[is] & 63; mv = sc[is+4] & 63; }\n"
"    else { sv = (sc[is+4]&0xF)|((sc[is-4]>>6)<<4); mv = (sc[is+4]>>4)|((sc[is]>>6)<<4); }\n"
"    float dl = d * sv;\n"
"    float ml = dmin * mv;\n"
"    unsigned char q = qs[j * 32 + l];\n"
"    float val = sub == 0 ? (dl * (q & 0xF) - ml) : (dl * (q >> 4) - ml);\n"
"    dst[i] = val;\n"
"}\n"
"\n"
"/* ---- 18c. embed_q4_0: Q4_0 embedding lookup -> F32 ---- */\n"
"/* Q4_0 block: 18 bytes = [half d][qs[16]], 32 elements; w[j]=(qs[j]&0xF-8)*d for\n"
"   j<16, (qs[j-16]>>4-8)*d for j>=16 (matches matvec_q4_0_f32). */\n"
"__global__ void embed_q4_0(float *dst, const unsigned char *embd_table,\n"
"                             int token_id, int n_embd) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n_embd) return;\n"
"    int nb_per_row = n_embd / 32;\n"
"    int row_bytes = nb_per_row * 18;\n"
"    const unsigned char *row = embd_table + (size_t)token_id * row_bytes;\n"
"    int block_idx = i / 32;\n"
"    int elem = i % 32;\n"
"    const unsigned char *bp = row + block_idx * 18;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    const unsigned char *qs = bp + 2;\n"
"    unsigned char q = qs[elem & 15];\n"
"    int v = (elem < 16) ? (int)(q & 0x0F) : (int)(q >> 4);\n"
"    dst[i] = (float)(v - 8) * d;\n"
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
"                                     int n_rows, int n_cols, int bm) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 66;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        /* bm=1: block-major repacked weights (block b of all rows contiguous); bm=0: row-major. */\n"
"        const unsigned char *bp = bm ? (mat + (size_t)b*n_rows*66 + (size_t)row*66)\n"
"                                      : (mat + (size_t)row*row_bytes + (size_t)b*66);\n"
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
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- matvec_iq2_xxs_q8_1_dp4a: IQ2_XXS weight x Q8_1 activation via dp4a (decode) ---- */\n"
"/* 8 warps/block, one warp per row, lane strides over 256-blocks. Branchless int8\n"
"   weight build: grid byte = magnitude, sign applied via per-byte two's complement\n"
"   __vadd4((g ^ mask), mask & 0x01010101) so dp4a_s8 sees signed int8. IQ2_XXS is\n"
"   zero-centered (signed) so there is no Q8_1 s8 offset term. Row-major weights. */\n"
"__global__ void matvec_iq2_xxs_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                           const unsigned char *x_q81,\n"
"                                           int n_rows, int n_cols) {\n"
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
"        const unsigned char *q81 = x_q81 + (size_t)(b * 8) * 36;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            unsigned int aux0 = qs[4*ib32] | ((unsigned int)qs[4*ib32+1] << 16);\n"
"            unsigned int aux1 = qs[4*ib32+2] | ((unsigned int)qs[4*ib32+3] << 16);\n"
"            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;\n"
"            const unsigned char *aq = q81 + ib32 * 36;\n"
"            const int *a = (const int *)aq;\n"
"            float d8 = half_to_float(*(const half_raw *)(aq + 32));\n"
"            int acc = 0;\n"
"            #pragma unroll\n"
"            for (int l = 0; l < 4; l++) {\n"
"                unsigned long long gv = iq2xxs_grid_dev[(aux0 >> (8*l)) & 255];\n"
"                unsigned int signs = ksigns_iq2xs_dev[(aux1 >> (7*l)) & 127];\n"
"                unsigned int slo = signs & 0xF, shi = (signs >> 4) & 0xF;\n"
"                unsigned int mlo = ((slo&1)?0x000000FFu:0u)|((slo&2)?0x0000FF00u:0u)|((slo&4)?0x00FF0000u:0u)|((slo&8)?0xFF000000u:0u);\n"
"                unsigned int mhi = ((shi&1)?0x000000FFu:0u)|((shi&2)?0x0000FF00u:0u)|((shi&4)?0x00FF0000u:0u)|((shi&8)?0xFF000000u:0u);\n"
"                unsigned int glo = (unsigned int)(gv & 0xFFFFFFFFu);\n"
"                unsigned int ghi = (unsigned int)(gv >> 32);\n"
"                unsigned int wlo = __vadd4(glo ^ mlo, mlo & 0x01010101u);\n"
"                unsigned int whi = __vadd4(ghi ^ mhi, mhi & 0x01010101u);\n"
"                acc = dp4a_s8((int)wlo, a[2*l],     acc);\n"
"                acc = dp4a_s8((int)whi, a[2*l + 1], acc);\n"
"            }\n"
"            sum += db * d8 * (float)acc;\n"
"        }\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"
"\n"
"/* ---- matvec_iq2_xxs_q8_1_dp4a_coal: coalesced-read variant of the kernel above ---- */\n"
"/* lane -> qs-short (NOT lane -> block): the 32 lanes read 32 consecutive qs shorts of\n"
"   ONE block => one fully coalesced 64-byte read, vs the strided 2-byte reads of the\n"
"   per-block-per-lane kernel (~220 GB/s ceiling). Decomposition: g=lane/4 = ib32 group\n"
"   (0..7), sub=lane%4 = l-subgroup (0..3). ib32=g needs exactly its 4 qs shorts, held\n"
"   by lanes g*4..g*4+3; aux0/aux1 are gathered within the 4-lane group via __shfl. Each\n"
"   lane decodes its 8 elements (2 dp4a); the 32 lanes (8 ib32 x 4 sub = whole 256-block)\n"
"   are warp-reduced, accumulated over blocks. Bit-identical to matvec_iq2_xxs_q8_1_dp4a. */\n"
"__global__ void matvec_iq2_xxs_q8_1_dp4a_coal(float *dst, const unsigned char *mat,\n"
"                                                const unsigned char *x_q81,\n"
"                                                int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 66;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int g = lane >> 2;        /* ib32 group 0..7 */\n"
"    int sub = lane & 3;       /* l-subgroup 0..3 */\n"
"    int g4 = g << 2;          /* base lane of this group */\n"
"    float sumf = 0.0f;\n"
"    for (int b = 0; b < nb; b++) {\n"
"        const unsigned char *bp = row_ptr + b * 66;\n"
"        unsigned int qs_l = ((const unsigned short *)(bp + 2))[lane];  /* coalesced 64B */\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned int s0 = __shfl_sync(0xFFFFFFFFu, qs_l, g4 + 0);\n"
"        unsigned int s1 = __shfl_sync(0xFFFFFFFFu, qs_l, g4 + 1);\n"
"        unsigned int s2 = __shfl_sync(0xFFFFFFFFu, qs_l, g4 + 2);\n"
"        unsigned int s3 = __shfl_sync(0xFFFFFFFFu, qs_l, g4 + 3);\n"
"        unsigned int aux0 = s0 | (s1 << 16);\n"
"        unsigned int aux1 = s2 | (s3 << 16);\n"
"        float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;\n"
"        unsigned long long gv = iq2xxs_grid_dev[(aux0 >> (8*sub)) & 255];\n"
"        unsigned int signs = ksigns_iq2xs_dev[(aux1 >> (7*sub)) & 127];\n"
"        unsigned int slo = signs & 0xF, shi = (signs >> 4) & 0xF;\n"
"        unsigned int mlo = ((slo&1)?0x000000FFu:0u)|((slo&2)?0x0000FF00u:0u)|((slo&4)?0x00FF0000u:0u)|((slo&8)?0xFF000000u:0u);\n"
"        unsigned int mhi = ((shi&1)?0x000000FFu:0u)|((shi&2)?0x0000FF00u:0u)|((shi&4)?0x00FF0000u:0u)|((shi&8)?0xFF000000u:0u);\n"
"        unsigned int glo = (unsigned int)(gv & 0xFFFFFFFFu);\n"
"        unsigned int ghi = (unsigned int)(gv >> 32);\n"
"        unsigned int wlo = __vadd4(glo ^ mlo, mlo & 0x01010101u);\n"
"        unsigned int whi = __vadd4(ghi ^ mhi, mhi & 0x01010101u);\n"
"        const unsigned char *aq = x_q81 + (size_t)(b*8 + g) * 36;\n"
"        const int *a = (const int *)aq;\n"
"        float d8 = half_to_float(*(const half_raw *)(aq + 32));\n"
"        int acc = dp4a_s8((int)wlo, a[2*sub], 0);\n"
"        acc = dp4a_s8((int)whi, a[2*sub + 1], acc);\n"
"        sumf += db * d8 * (float)acc;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sumf += __shfl_down_sync(0xFFFFFFFFu, sumf, offset);\n"
"    if (lane == 0) dst[row] = sumf;\n"
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

"/* ---- matvec_q4_0_q8_1_dp4a: Q4_0 weight x Q8_1 activation via dp4a ---- */\n"
"/* 8 warps/block, each warp = one row; lane strides over 32-elem blocks. Q4_0\n"
"   value=(nibble-8)*d_w; per block, dot = d_w*(d8*sum(nibble*a_int) - 8*s8) where\n"
"   the Q8_1 block holds a_int (int8 x32), d8=scale, s8=d8*sum(a_int). The 18-byte\n"
"   Q4_0 block leaves qs at offset 2 (not 4-aligned), so qs is read as bytes and\n"
"   packed; the Q8_1 quants (x_q81, 36B blocks) ARE 4-aligned and read as int. */\n"
"__global__ void matvec_q4_0_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                        const unsigned char *x_q81,\n"
"                                        int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 32;\n"
"    int row_bytes = nb * 18;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sumf = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 18;\n"
"        float dw = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *q81 = x_q81 + (size_t)b * 36;\n"
"        const int *u = (const int *)q81;\n"
"        float d8 = half_to_float(*(const half_raw *)(q81 + 32));\n"
"        float s8 = half_to_float(*(const half_raw *)(q81 + 34));\n"
"        int acc = 0;\n"
"        #pragma unroll\n"
"        for (int k = 0; k < 4; k++) {\n"
"            unsigned int c0 = qs[4*k], c1 = qs[4*k+1], c2 = qs[4*k+2], c3 = qs[4*k+3];\n"
"            int v_lo = (int)((c0 & 0xF) | ((c1 & 0xF) << 8) | ((c2 & 0xF) << 16) | ((c3 & 0xF) << 24));\n"
"            int v_hi = (int)((c0 >> 4)  | ((c1 >> 4)  << 8) | ((c2 >> 4)  << 16) | ((c3 >> 4)  << 24));\n"
"            acc = dp4a_s8(v_lo, u[k],     acc);\n"
"            acc = dp4a_s8(v_hi, u[k + 4], acc);\n"
"        }\n"
"        sumf += dw * (d8 * (float)acc - 8.0f * s8);\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);\n"
"    if (lane == 0) dst[row] = sumf;\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
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
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq3_xxs_q8_1_dp4a: IQ3_XXS weight x Q8_1 activation via dp4a (decode) ---- */\n"
"/* Mirrors matvec_iq3_xxs_f32: 98-byte block = d + qs[64] + scales_and_signs[32];\n"
"   uint32 grid (4 magnitudes), db = d*(0.5+(aux>>28))*0.5. Branchless signed int8 +\n"
"   dp4a, no Q8_1 offset term (zero-centered). Row-major. */\n"
"/* Coalesced variant: lane -> qs uint16 (one 64B coalesced read of the block's qs).\n"
"   g=lane/4=ib32, sub=lane%4=l. qs uint16[lane] holds exactly qsi[2*sub],qsi[2*sub+1]\n"
"   for (g,sub); aux32 = sas word g (broadcast within the 4-lane group). Each lane does\n"
"   its 2 dp4a; the 32 lanes (8 ib32 x 4 sub) warp-reduce. Bit-identical to the strided\n"
"   form above. */\n"
"__global__ void matvec_iq3_xxs_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                           const unsigned char *x_q81,\n"
"                                           int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int g = lane >> 2;        /* ib32 0..7 */\n"
"    int sub = lane & 3;       /* l 0..3 */\n"
"    float sumf = 0.0f;\n"
"    for (int b = 0; b < nb; b++) {\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned int qs_l = ((const unsigned short *)(bp + 2))[lane];   /* coalesced 64B */\n"
"        const unsigned short *sas16 = (const unsigned short *)(bp + 66); /* 2-byte aligned */\n"
"        unsigned int aux32 = (unsigned int)sas16[2*g] | ((unsigned int)sas16[2*g+1] << 16);\n"
"        float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;\n"
"        unsigned int signs = ksigns_iq2xs_dev[(aux32 >> (7*sub)) & 127];\n"
"        unsigned int g1 = iq3xxs_grid_dev[qs_l & 0xFF];\n"
"        unsigned int g2 = iq3xxs_grid_dev[(qs_l >> 8) & 0xFF];\n"
"        unsigned int slo = signs & 0xF, shi = (signs >> 4) & 0xF;\n"
"        unsigned int mlo = ((slo&1)?0x000000FFu:0u)|((slo&2)?0x0000FF00u:0u)|((slo&4)?0x00FF0000u:0u)|((slo&8)?0xFF000000u:0u);\n"
"        unsigned int mhi = ((shi&1)?0x000000FFu:0u)|((shi&2)?0x0000FF00u:0u)|((shi&4)?0x00FF0000u:0u)|((shi&8)?0xFF000000u:0u);\n"
"        unsigned int w1 = __vadd4(g1 ^ mlo, mlo & 0x01010101u);\n"
"        unsigned int w2 = __vadd4(g2 ^ mhi, mhi & 0x01010101u);\n"
"        const unsigned char *aq = x_q81 + (size_t)(b*8 + g) * 36;\n"
"        const int *a = (const int *)aq;\n"
"        float d8 = half_to_float(*(const half_raw *)(aq + 32));\n"
"        int acc = dp4a_s8((int)w1, a[2*sub],     0);\n"
"        acc = dp4a_s8((int)w2, a[2*sub + 1], acc);\n"
"        sumf += db * d8 * (float)acc;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);\n"
"    if (lane == 0) dst[row] = sumf;\n"
"}\n"
"\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- matvec_iq2_s_q8_1_dp4a: IQ2_S weight x Q8_1 activation via dp4a (decode) ---- */\n"
"/* Mirrors matvec_iq2_s_f32: 82B = d + qs[32]@2 + signs[32]@34 + qh[8]@66 + scales[8]@74;\n"
"   uint64 grid (8 mag, 10-bit idx via qh), two scales per ib32 (l<2 vs l>=2) -> two\n"
"   accumulators. Branchless signed int8 + dp4a, no Q8_1 offset term. Row-major. */\n"
"/* Coalesced: lane -> (g=ib32, sub=l). qs[lane]/signs[lane] are exactly this lane's\n"
"   byte (g*4+sub=lane); qh[g]/scales[g] broadcast within the 4-lane group. Each lane\n"
"   does its 2 dp4a; the 32 lanes warp-reduce. Numerically identical to the strided form. */\n"
"__global__ void matvec_iq2_s_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                         const unsigned char *x_q81,\n"
"                                         int n_rows, int n_cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 82;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int g = lane >> 2;        /* ib32 0..7 */\n"
"    int sub = lane & 3;       /* l 0..3 */\n"
"    float sumf = 0.0f;\n"
"    for (int b = 0; b < nb; b++) {\n"
"        const unsigned char *bp = row_ptr + b * 82;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned int qsb = ((const unsigned char *)(bp + 2))[lane];     /* coalesced 32B */\n"
"        unsigned int sgb = ((const unsigned char *)(bp + 34))[lane];    /* coalesced 32B */\n"
"        unsigned int qhg = ((const unsigned char *)(bp + 66))[g];       /* broadcast */\n"
"        unsigned int scg = ((const unsigned char *)(bp + 74))[g];       /* broadcast */\n"
"        float db = (sub < 2) ? d * (0.5f + (float)(scg & 0xf)) * 0.25f\n"
"                             : d * (0.5f + (float)(scg >>  4)) * 0.25f;\n"
"        int gid = qsb | ((qhg << (8-2*sub)) & 0x300);\n"
"        unsigned long long gv = iq2s_grid_dev[gid];\n"
"        unsigned int slo = sgb & 0xF, shi = (sgb >> 4) & 0xF;\n"
"        unsigned int mlo = ((slo&1)?0x000000FFu:0u)|((slo&2)?0x0000FF00u:0u)|((slo&4)?0x00FF0000u:0u)|((slo&8)?0xFF000000u:0u);\n"
"        unsigned int mhi = ((shi&1)?0x000000FFu:0u)|((shi&2)?0x0000FF00u:0u)|((shi&4)?0x00FF0000u:0u)|((shi&8)?0xFF000000u:0u);\n"
"        unsigned int glo = (unsigned int)(gv & 0xFFFFFFFFu);\n"
"        unsigned int ghi = (unsigned int)(gv >> 32);\n"
"        unsigned int wlo = __vadd4(glo ^ mlo, mlo & 0x01010101u);\n"
"        unsigned int whi = __vadd4(ghi ^ mhi, mhi & 0x01010101u);\n"
"        const unsigned char *aq = x_q81 + (size_t)(b*8 + g) * 36;\n"
"        const int *a = (const int *)aq;\n"
"        float d8 = half_to_float(*(const half_raw *)(aq + 32));\n"
"        int acc = dp4a_s8((int)wlo, a[2*sub],     0);\n"
"        acc = dp4a_s8((int)whi, a[2*sub + 1], acc);\n"
"        sumf += db * d8 * (float)acc;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);\n"
"    if (lane == 0) dst[row] = sumf;\n"
"}\n"
"\n"
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
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) dst[row] = sum;\n"
"}\n"

"/* ---- dequant_iq3_s_to_f16: IQ3_S weight -> F16, element order matching matvec_iq3_s_f32 ---- */\n"
"__global__ void dequant_iq3_s_to_f16(half_raw *dst, const unsigned char *mat,\n"
"                                       int rows, int cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= rows) return;\n"
"    int nb = cols / 256;\n"
"    int row_bytes = nb * 110;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    half_raw *dst_row = dst + (size_t)row * cols;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *qh_base = bp + 2 + 64;\n"
"        const unsigned char *signs_base = bp + 2 + 64 + 8;\n"
"        const unsigned char *scales = bp + 2 + 64 + 8 + 32;\n"
"        half_raw *d_b = dst_row + b * 256;\n"
"        int yi = 0, qhi = 0, si = 0;\n"
"        for (int ib32 = 0; ib32 < 8; ib32 += 2) {\n"
"            float db1 = d * (float)(1 + 2*(scales[ib32/2] & 0xf));\n"
"            float db2 = d * (float)(1 + 2*(scales[ib32/2] >>  4));\n"
"            for (int l = 0; l < 4; l++) {\n"
"                const unsigned char *grid1 = (const unsigned char *)&iq3s_grid_dev[qs[2*l+0] | ((qh_base[qhi] << (8-2*l)) & 256)];\n"
"                const unsigned char *grid2 = (const unsigned char *)&iq3s_grid_dev[qs[2*l+1] | ((qh_base[qhi] << (7-2*l)) & 256)];\n"
"                unsigned char s = signs_base[si + l];\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    d_b[yi+j]   = float_to_half(db1 * (float)grid1[j] * ((s & (1 << j)) ? -1.0f : 1.0f));\n"
"                    d_b[yi+j+4] = float_to_half(db1 * (float)grid2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f));\n"
"                }\n"
"                yi += 8;\n"
"            }\n"
"            qs += 8; si += 4;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                const unsigned char *grid1 = (const unsigned char *)&iq3s_grid_dev[qs[2*l+0] | ((qh_base[qhi+1] << (8-2*l)) & 256)];\n"
"                const unsigned char *grid2 = (const unsigned char *)&iq3s_grid_dev[qs[2*l+1] | ((qh_base[qhi+1] << (7-2*l)) & 256)];\n"
"                unsigned char s = signs_base[si + l];\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    d_b[yi+j]   = float_to_half(db2 * (float)grid1[j] * ((s & (1 << j)) ? -1.0f : 1.0f));\n"
"                    d_b[yi+j+4] = float_to_half(db2 * (float)grid2[j] * ((s & (1 << (j+4))) ? -1.0f : 1.0f));\n"
"                }\n"
"                yi += 8;\n"
"            }\n"
"            qhi += 2; qs += 8; si += 4;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- matvec_iq3_s_q8_1_dp4a: IQ3_S weight x Q8_1 activation via dp4a (decode) ---- */\n"
"/* Mirrors matvec_iq3_s_f32: 110B = d + qs[64]@2 + qh[8]@66 + signs[32]@74 + scales[4]@106;\n"
"   uint32 grid (4 mag, 9-bit idx via qh bit 8), one scale per 32-block. Processed per\n"
"   32-sub-block (imb). Branchless signed int8 + dp4a, no Q8_1 offset term. Row-major. */\n"
"__global__ void matvec_iq3_s_q8_1_dp4a(float *dst, const unsigned char *mat,\n"
"                                         const unsigned char *x_q81,\n"
"                                         int n_rows, int n_cols) {\n"
"    /* Coalesced: lane -> (g=imb, sub=l). qs uint16[lane] = qsi[2*sub],qsi[2*sub+1];\n"
"       signs[lane] = sgi[sub]; qh[g]/scales[g>>1] broadcast. Numerically identical. */\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 110;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    int g = lane >> 2;        /* imb 0..7 */\n"
"    int sub = lane & 3;       /* l 0..3 */\n"
"    float sumf = 0.0f;\n"
"    for (int b = 0; b < nb; b++) {\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        unsigned int qs_l = ((const unsigned short *)(bp + 2))[lane];   /* coalesced 64B */\n"
"        unsigned int qhg = ((const unsigned char *)(bp + 66))[g];       /* broadcast */\n"
"        unsigned int sgb = ((const unsigned char *)(bp + 74))[lane];    /* coalesced 32B */\n"
"        unsigned int scg = ((const unsigned char *)(bp + 106))[g >> 1]; /* broadcast */\n"
"        float db = d * (float)(1 + 2*((g & 1) ? (scg >> 4) : (scg & 0xf)));\n"
"        unsigned int g1 = iq3s_grid_dev[(qs_l & 0xFF)        | ((qhg << (8-2*sub)) & 256)];\n"
"        unsigned int g2 = iq3s_grid_dev[((qs_l >> 8) & 0xFF) | ((qhg << (7-2*sub)) & 256)];\n"
"        unsigned int slo = sgb & 0xF, shi = (sgb >> 4) & 0xF;\n"
"        unsigned int mlo = ((slo&1)?0x000000FFu:0u)|((slo&2)?0x0000FF00u:0u)|((slo&4)?0x00FF0000u:0u)|((slo&8)?0xFF000000u:0u);\n"
"        unsigned int mhi = ((shi&1)?0x000000FFu:0u)|((shi&2)?0x0000FF00u:0u)|((shi&4)?0x00FF0000u:0u)|((shi&8)?0xFF000000u:0u);\n"
"        unsigned int w1 = __vadd4(g1 ^ mlo, mlo & 0x01010101u);\n"
"        unsigned int w2 = __vadd4(g2 ^ mhi, mhi & 0x01010101u);\n"
"        const unsigned char *aq = x_q81 + (size_t)(b*8 + g) * 36;\n"
"        const int *a = (const int *)aq;\n"
"        float d8 = half_to_float(*(const half_raw *)(aq + 32));\n"
"        int acc = dp4a_s8((int)w1, a[2*sub],     0);\n"
"        acc = dp4a_s8((int)w2, a[2*sub + 1], acc);\n"
"        sumf += db * d8 * (float)acc;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sumf += __shfl_down_sync(0xFFFFFFFF, sumf, offset);\n"
"    if (lane == 0) dst[row] = sumf;\n"
"}\n"
"\n"
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
"/* ---- 4b. raw_rmsnorm_heads_f32: Per-head RMSNorm without learned weight (for V norm) ---- */\n"
"__global__ void raw_rmsnorm_heads_f32(float *vec, int n_heads, int head_dim, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    float *v = vec + h * head_dim;\n"
"    float val = (tid < head_dim) ? v[tid] : 0.0f;\n"
"    sdata[tid] = val * val;\n"
"    __syncthreads();\n"
"    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);\n"
"    if (tid < head_dim) v[tid] = val * scale;\n"
"}\n"
"\n"
"/* ---- Gemma4 kernels ---- */\n"
"\n"
"/* GELU(gate) * up: gate[i] = gelu(gate[i]) * up[i] */\n"
"__global__ void gelu_mul_f32(float *gate, const float *up, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float g = gate[i];\n"
"    float gelu_g = g * 0.5f * (1.0f + erff(g * 0.7071067811865476f));\n"
"    gate[i] = gelu_g * up[i];\n"
"}\n"
"\n"
"/* GELU element-wise multiply: out[i] = gelu(gate[i]) * ple[i] */\n"
"__global__ void gelu_elementwise_mul_f32(float *ple, const float *gate, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float g = gate[i];\n"
"    float gelu_g = g * 0.5f * (1.0f + erff(g * 0.7071067811865476f));\n"
"    ple[i] *= gelu_g;\n"
"}\n"
"\n"
"/* Scale vector: x[i] *= scale */\n"
"__global__ void scale_f32(float *x, float scale_val, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    x[i] *= scale_val;\n"
"}\n"
"\n"
"/* Logit soft-capping: x[i] = cap * tanh(x[i] / cap) */\n"
"__global__ void logit_softcap_f32(float *x, int n, float cap) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    x[i] = cap * tanhf(x[i] / cap);\n"
"}\n"
"\n"
"/* SWA attention decode with circular KV cache.\n"
"   One block per head, blockDim.x threads for head_dim parallelism. */\n"
"/* SWA attention decode with circular KV cache.\n"
"   One block per head, 256 threads. Shared memory: seq_len floats for scores.\n"
"   Uses cross-warp reduction for softmax (same pattern as attn_decode_f32). */\n"
"__global__ void attn_decode_swa_f32(\n"
"    float *out, const float *Q,\n"
"    const half_raw *K_cache, const half_raw *V_cache,\n"
"    int n_heads, int n_kv_heads, int head_dim, int kv_dim,\n"
"    int position, int window_size, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"    int start = (position >= window_size) ? (position - window_size + 1) : 0;\n"
"    int seq_len = position - start + 1;\n"
"    float *scores = smem;\n"
"    const float *q_h = Q + h * head_dim;\n"
"    int warp_id = tid / 32;\n"
"    int lane = tid % 32;\n"
"\n"
"    /* Phase 1: QK scores */\n"
"    for (int p = tid; p < seq_len; p += nthreads) {\n"
"        int slot = (start + p) % window_size;\n"
"        const half_raw *k_p = K_cache + slot * kv_dim + kv_h * head_dim;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) dot += q_h[d] * half_to_float(k_p[d]);\n"
"        scores[p] = dot * scale;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 2: Softmax with cross-warp reduction */\n"
"    float local_max = -1e30f;\n"
"    for (int p = tid; p < seq_len; p += nthreads)\n"
"        if (scores[p] > local_max) local_max = scores[p];\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));\n"
"    __shared__ float warp_max[8];\n"
"    if (lane == 0) warp_max[warp_id] = local_max;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float m = warp_max[0];\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 1; w < n_warps; w++)\n"
"            if (warp_max[w] > m) m = warp_max[w];\n"
"        warp_max[0] = m;\n"
"    }\n"
"    __syncthreads();\n"
"    float max_val = warp_max[0];\n"
"\n"
"    float local_sum = 0.0f;\n"
"    for (int p = tid; p < seq_len; p += nthreads) {\n"
"        float e = expf(scores[p] - max_val);\n"
"        scores[p] = e;\n"
"        local_sum += e;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);\n"
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
"    for (int p = tid; p < seq_len; p += nthreads)\n"
"        scores[p] *= inv_sum;\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 3: Weighted V sum */\n"
"    float *out_h = out + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += nthreads) {\n"
"        float acc = 0.0f;\n"
"        for (int p = 0; p < seq_len; p++) {\n"
"            int slot = (start + p) % window_size;\n"
"            acc += scores[p] * half_to_float(V_cache[slot * kv_dim + kv_h * head_dim + d]);\n"
"        }\n"
"        out_h[d] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* BF16 matvec: out[row] = sum(mat[row][c] * vec[c]) for BF16 matrix */\n"
"__global__ void matvec_bf16_f32(float *dst, const unsigned short *mat,\n"
"    const float *vec, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    const unsigned short *row_data = mat + (long long)row * n_cols;\n"
"    float sum = 0.0f;\n"
"    for (int c = tid; c < n_cols; c += nthreads) {\n"
"        /* BF16 to F32: shift left by 16 bits */\n"
"        unsigned int bits = (unsigned int)row_data[c] << 16;\n"
"        float val;\n"
"        asm volatile(\"mov.b32 %0, %1;\" : \"=f\"(val) : \"r\"(bits));\n"
"        sum += val * vec[c];\n"
"    }\n"
"    /* Warp reduce */\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws_bf16[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws_bf16[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < n_warps; w++) total += ws_bf16[w];\n"
"        dst[row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* RoPE with per-dimension frequency factors (proportional RoPE for Gemma4 full-attn).\n"
"   freq_factors[j] is the precomputed inv_freq (already divided by rope_freqs). */\n"
"__global__ void rope_with_factors_f32(\n"
"    float *vec, int n_heads, int head_dim, int position,\n"
"    const float *inv_freq, int n_pairs) {\n"
"    int h = blockIdx.x;\n"
"    int j = threadIdx.x;\n"
"    if (j >= n_pairs) return;\n"
"    float freq = (float)position * inv_freq[j];\n"
"    float cos_v = cosf(freq), sin_v = sinf(freq);\n"
"    float *v = vec + h * head_dim;\n"
"    float r0 = v[j], r1 = v[j + n_pairs];\n"
"    v[j]           = r0 * cos_v - r1 * sin_v;\n"
"    v[j + n_pairs] = r0 * sin_v + r1 * cos_v;\n"
"}\n"
"\n"
"/* Device-pointer variant of rope_with_factors_f32 for graph capture */\n"
"__global__ void rope_with_factors_f32_ptr(\n"
"    float *vec, int n_heads, int head_dim,\n"
"    const int *pos_ptr,\n"
"    const float *inv_freq, int n_pairs) {\n"
"    int position = *pos_ptr;\n"
"    int h = blockIdx.x;\n"
"    int j = threadIdx.x;\n"
"    if (j >= n_pairs) return;\n"
"    float freq = (float)position * inv_freq[j];\n"
"    float cos_v = cosf(freq), sin_v = sinf(freq);\n"
"    float *v = vec + h * head_dim;\n"
"    float r0 = v[j], r1 = v[j + n_pairs];\n"
"    v[j]           = r0 * cos_v - r1 * sin_v;\n"
"    v[j + n_pairs] = r0 * sin_v + r1 * cos_v;\n"
"}\n"
"\n"
"/* Device-pointer variant of attn_decode_swa_f32 for graph capture */\n"
"__global__ void attn_decode_swa_f32_ptr(\n"
"    float *out, const float *Q,\n"
"    const half_raw *K_cache, const half_raw *V_cache,\n"
"    int n_heads, int n_kv_heads, int head_dim, int kv_dim,\n"
"    const int *pos_ptr, int window_size, float scale) {\n"
"    int position = *pos_ptr;\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"    int start = (position >= window_size) ? (position - window_size + 1) : 0;\n"
"    int seq_len = position - start + 1;\n"
"    float *scores = smem;\n"
"    const float *q_h = Q + h * head_dim;\n"
"    int warp_id = tid / 32;\n"
"    int lane = tid % 32;\n"
"\n"
"    for (int p = tid; p < seq_len; p += nthreads) {\n"
"        int slot = (start + p) % window_size;\n"
"        const half_raw *k_p = K_cache + slot * kv_dim + kv_h * head_dim;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) dot += q_h[d] * half_to_float(k_p[d]);\n"
"        scores[p] = dot * scale;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    float local_max = -1e30f;\n"
"    for (int p = tid; p < seq_len; p += nthreads)\n"
"        if (scores[p] > local_max) local_max = scores[p];\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));\n"
"    __shared__ float warp_max[8];\n"
"    if (lane == 0) warp_max[warp_id] = local_max;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float m = warp_max[0];\n"
"        int n_warps = (nthreads + 31) / 32;\n"
"        for (int w = 1; w < n_warps; w++)\n"
"            if (warp_max[w] > m) m = warp_max[w];\n"
"        warp_max[0] = m;\n"
"    }\n"
"    __syncthreads();\n"
"    float max_val = warp_max[0];\n"
"\n"
"    float local_sum = 0.0f;\n"
"    for (int p = tid; p < seq_len; p += nthreads) {\n"
"        float e = expf(scores[p] - max_val);\n"
"        scores[p] = e;\n"
"        local_sum += e;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);\n"
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
"    for (int p = tid; p < seq_len; p += nthreads)\n"
"        scores[p] *= inv_sum;\n"
"    __syncthreads();\n"
"\n"
"    float *out_h = out + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += nthreads) {\n"
"        float acc = 0.0f;\n"
"        for (int p = 0; p < seq_len; p++) {\n"
"            int slot = (start + p) % window_size;\n"
"            acc += scores[p] * half_to_float(V_cache[slot * kv_dim + kv_h * head_dim + d]);\n"
"        }\n"
"        out_h[d] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* ==== Batched Prefill Kernels ==== */\n"
"\n"
"/* Convert F32 array to F16 in-place (output to separate buffer) */\n"
"__global__ void convert_f32_to_f16(half_raw *out, const float *in, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) out[i] = float_to_half(in[i]);\n"
"}\n"
"\n"
"/* ---- convert_f16_to_f32: F16 to F32 conversion ---- */\n"
"__global__ void convert_f16_to_f32(float *out, const half_raw *in, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) out[i] = half_to_float(in[i]);\n"
"}\n"
"\n"
"__device__ __forceinline__ unsigned short bf16_to_f16_trunc(unsigned short v) {\n"
"    unsigned int bits = ((unsigned int)v) << 16;\n"
"    unsigned short sign = (unsigned short)((bits >> 16) & 0x8000u);\n"
"    int exp = (int)((bits >> 23) & 0xffu) - 127;\n"
"    unsigned int mant = bits & 0x7fffffu;\n"
"    if (exp > 15) return (unsigned short)(sign | 0x7c00u);\n"
"    if (exp < -14) {\n"
"        if (exp < -24) return sign;\n"
"        mant |= 0x800000u;\n"
"        mant >>= (-1 - exp);\n"
"        return (unsigned short)(sign | (unsigned short)(mant >> 13));\n"
"    }\n"
"    return (unsigned short)(sign | (unsigned short)((exp + 15) << 10) | (unsigned short)(mant >> 13));\n"
"}\n"
"\n"
"__global__ void bf16_to_f16_inplace(unsigned short *data, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    data[i] = bf16_to_f16_trunc(data[i]);\n"
"}\n"
"\n"
"/* Batched F16 embedding lookup: output[token, i] = embd[token_ids[token], i] */\n"
"__global__ void batch_embed_f16(float *output, const half_raw *embd_table, const int *token_ids,\n"
"                                 int n_embd, int n_tokens) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int token = blockIdx.y;\n"
"    if (i >= n_embd || token >= n_tokens) return;\n"
"    int token_id = token_ids[token];\n"
"    output[(size_t)token * n_embd + i] = half_to_float(embd_table[(size_t)token_id * n_embd + i]);\n"
"}\n"
"\n"
"/* Batched Q8_0 matvec: process n_tokens through weight matrix simultaneously */\n"
"/* Grid: [ceil(out_dim/8), n_tokens], Block: 256 (8 warps) */\n"
"__global__ void batch_matvec_q8_0_f32(float *output, const unsigned char *mat, const float *input,\n"
"                                       int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 32;\n"
"    int row_bytes = nb * 36;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 36;\n"
"        float d = half_to_float(*(const half_raw *)(bp));\n"
"        const signed char *qs = (const signed char *)(bp + 4);\n"
"        const float *xb = x + b * 32;\n"
"        float partial = 0.0f;\n"
"        for (int i = 0; i < 32; i++) partial += (float)qs[i] * xb[i];\n"
"        sum += partial * d;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* ---- Chunked batch_matvec_q8_0: 4 tokens per block (reduces weight reads 4x) ---- */\n"
"__global__ void batch_matvec_q8_0_x4(float *output, const unsigned char *mat, const float *input,\n"
"                                      int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= out_dim) return;\n"
"    int t0 = blockIdx.y * 4;\n"
"    int nt = n_tokens - t0;\n"
"    if (nt > 4) nt = 4;\n"
"    if (nt <= 0) return;\n"
"    int nb = in_dim / 32;\n"
"    int row_bytes = nb * 36;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum[4];\n"
"    for (int t = 0; t < nt; t++) sum[t] = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 36;\n"
"        float d = half_to_float(*(const half_raw *)(bp));\n"
"        const signed char *qs = (const signed char *)(bp + 4);\n"
"        float partial[4];\n"
"        for (int t = 0; t < nt; t++) partial[t] = 0.0f;\n"
"        for (int i = 0; i < 32; i++) {\n"
"            float w = d * (float)qs[i];\n"
"            for (int t = 0; t < nt; t++)\n"
"                partial[t] += w * input[(size_t)(t0+t) * in_dim + b * 32 + i];\n"
"        }\n"
"        for (int t = 0; t < nt; t++) sum[t] += partial[t];\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        for (int t = 0; t < nt; t++)\n"
"            sum[t] += __shfl_down_sync(0xFFFFFFFF, sum[t], offset);\n"
"    if (lane == 0)\n"
"        for (int t = 0; t < nt; t++)\n"
"            output[(size_t)(t0+t) * out_dim + row] = sum[t];\n"
"}\n"
"\n"
"/* Batched Q2_K matvec: process n_tokens through weight matrix simultaneously */\n"
"/* Grid: [ceil(out_dim/8), n_tokens], Block: 256 (8 warps) */\n"
"__global__ void batch_matvec_q2_K(float *output, const unsigned char *mat, const float *input,\n"
"                                    int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * nb * 84;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
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
"                float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);\n"
"                for (int l = 0; l < 16; l++) partial += (dl * ((qs[l] >> shift) & 3) - ml) * xb[yi++];\n"
"                sc = scales[is++];\n"
"                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);\n"
"                for (int l = 0; l < 16; l++) partial += (dl * ((qs[l+16] >> shift) & 3) - ml) * xb[yi++];\n"
"            }\n"
"            qs += 32;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Batched Q3_K matvec */\n"
"__global__ void batch_matvec_q3_K(float *output, const unsigned char *mat, const float *input,\n"
"                                    int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * nb * 110;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        const unsigned char *hm = bp;\n"
"        const unsigned char *qs = bp + 32;\n"
"        const unsigned char *raw_sc = bp + 96;\n"
"        float d_all = half_to_float(*(const half_raw *)(bp + 108));\n"
"        unsigned int a0=raw_sc[0]|(raw_sc[1]<<8)|(raw_sc[2]<<16)|(raw_sc[3]<<24);\n"
"        unsigned int a1=raw_sc[4]|(raw_sc[5]<<8)|(raw_sc[6]<<16)|(raw_sc[7]<<24);\n"
"        unsigned int tmp2=raw_sc[8]|(raw_sc[9]<<8)|(raw_sc[10]<<16)|(raw_sc[11]<<24);\n"
"        unsigned int km1=0x03030303u, km2=0x0f0f0f0fu;\n"
"        unsigned int aux[4];\n"
"        aux[0]=(a0&km2)|(((tmp2>>0)&km1)<<4); aux[1]=(a1&km2)|(((tmp2>>2)&km1)<<4);\n"
"        aux[2]=((a0>>4)&km2)|(((tmp2>>4)&km1)<<4); aux[3]=((a1>>4)&km2)|(((tmp2>>6)&km1)<<4);\n"
"        const signed char *scales = (const signed char *)aux;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int is = 0, yi = 0;\n"
"        unsigned char m_bit = 1;\n"
"        for (int n0 = 0; n0 < 2; n0++) {\n"
"            for (int j = 0; j < 4; j++) {\n"
"                int shift = j * 2;\n"
"                float dl1 = d_all * (scales[is++] - 32);\n"
"                for (int l = 0; l < 16; l++) { int qv=((qs[l]>>shift)&3)-((hm[l]&m_bit)?0:4); partial+=dl1*qv*xb[yi++]; }\n"
"                float dl2 = d_all * (scales[is++] - 32);\n"
"                for (int l = 0; l < 16; l++) { int qv=((qs[l+16]>>shift)&3)-((hm[l+16]&m_bit)?0:4); partial+=dl2*qv*xb[yi++]; }\n"
"                m_bit <<= 1;\n"
"            }\n"
"            qs += 32;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Batched IQ2_S matvec */\n"
"__global__ void batch_matvec_iq2_s(float *output, const unsigned char *mat, const float *input,\n"
"                                    int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 82;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 82;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *qh = bp + 66;\n"
"        const unsigned char *scales = bp + 74;\n"
"        const unsigned char *signs = bp + 34;\n"
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
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Batched IQ2_XXS matvec */\n"
"__global__ void batch_matvec_iq2_xxs(float *output, const unsigned char *mat, const float *input,\n"
"                                       int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 66;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
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
"                unsigned char signs = ksigns_iq2xs_dev[(aux1 >> (7*l)) & 127];\n"
"                const unsigned char *grid = (const unsigned char *)&iq2xxs_grid_dev[aux8[l]];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    float w = db * (float)grid[j] * ((signs & (1 << j)) ? -1.0f : 1.0f);\n"
"                    partial += w * xb[yi++];\n"
"                }\n"
"            }\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Batched IQ2_XS matvec */\n"
"__global__ void batch_matvec_iq2_xs(float *output, const unsigned char *mat, const float *input,\n"
"                                      int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 74;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 74;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"        const unsigned char *scales = bp + 2 + 64;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;\n"
"            float db1 = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                float dl = (l < 2) ? db0 : db1;\n"
"                unsigned short qval = qs[4*ib32 + l];\n"
"                unsigned long long grid_val = iq2xs_grid_dev[qval & 511];\n"
"                unsigned char signs = ksigns_iq2xs_dev[qval >> 9];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    float w = dl * (float)(unsigned char)(grid_val >> (8*j)) * ((signs & (1 << j)) ? -1.0f : 1.0f);\n"
"                    partial += w * xb[yi++];\n"
"                }\n"
"            }\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Batched IQ3_XXS matvec */\n"
"__global__ void batch_matvec_iq3_xxs(float *output, const unsigned char *mat, const float *input,\n"
"                                       int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *scales_and_signs = qs + 64;\n"
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
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Batched IQ3_S matvec */\n"
"__global__ void batch_matvec_iq3_s(float *output, const unsigned char *mat, const float *input,\n"
"                                    int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 110;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 110;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *qh_base = bp + 2 + 64;\n"
"        const unsigned char *signs_base = bp + 2 + 64 + 8;\n"
"        const unsigned char *scales = bp + 2 + 64 + 8 + 32;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0, qhi = 0, si = 0;\n"
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
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Batched Q4_K matvec (float path) */\n"
"__global__ void batch_matvec_q4_K(float *output, const unsigned char *mat, const float *input,\n"
"                                    int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 144;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 144;\n"
"        float d = half_to_float(*(const half_raw *)(bp));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *sc = bp + 4;\n"
"        const unsigned char *qs = bp + 16;\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        int yi = 0, is = 0;\n"
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
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"

"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* ---- Chunked batch_matvec_q4_K: processes 4 tokens per block ---- */\n"
"__global__ void batch_matvec_q4_K_x4(float *output, const unsigned char *mat, const float *input,\n"
"                                      int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= out_dim) return;\n"
"    int t0 = blockIdx.y * 4;\n"
"    int nt = n_tokens - t0;\n"
"    if (nt > 4) nt = 4;\n"
"    if (nt <= 0) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 144;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum[4];\n"
"    for (int t = 0; t < nt; t++) sum[t] = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 144;\n"
"        float d = half_to_float(*(const half_raw *)(bp));\n"
"        float dmin = half_to_float(*(const half_raw *)(bp + 2));\n"
"        const unsigned char *sc = bp + 4;\n"
"        const unsigned char *qs = bp + 16;\n"
"        int yi = 0, is = 0;\n"
"        float p0[4], p1[4];\n"
"        for (int t = 0; t < nt; t++) p0[t] = 0.0f, p1[t] = 0.0f;\n"
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
"                float v0 = d1 * (q[l] & 0xF) - m1;\n"
"                float v1 = d2 * (q[l] >> 4) - m2;\n"
"                for (int t = 0; t < nt; t++) {\n"
"                    const float *x = input + (size_t)(t0+t) * in_dim + (size_t)b * 256 + yi;\n"
"                    p0[t] += v0 * x[0];\n"
"                    p1[t] += v1 * x[0];\n"
"                }\n"
"                yi++;\n"
"            }\n"
"            is += 2;\n"
"        }\n"
"        for (int t = 0; t < nt; t++) sum[t] += p0[t] + p1[t];\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        for (int t = 0; t < nt; t++)\n"
"            sum[t] += __shfl_down_sync(0xFFFFFFFF, sum[t], offset);\n"
"    if (lane == 0)\n"
"        for (int t = 0; t < nt; t++)\n"
"            output[(size_t)(t0+t) * out_dim + row] = sum[t];\n"
"}\n"
"\n"
"/* Batched Q5_K matvec (float path) */\n"
"__global__ void batch_matvec_q5_K(float *output, const unsigned char *mat, const float *input,\n"
"                                    int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 176;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
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
"        int yi = 0, is = 0;\n"
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
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Batched Q6_K matvec (float path) */\n"
"__global__ void batch_matvec_q6_K(float *output, const unsigned char *mat, const float *input,\n"
"                                    int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 210;\n"
"        const unsigned char *ql = bp;\n"
"        const unsigned char *qh = bp + 128;\n"
"        const signed char *sc = (const signed char *)(bp + 192);\n"
"        float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"        const float *xb = x + b * 256;\n"
"        float partial = 0.0f;\n"
"        for (int half = 0; half < 2; half++) {\n"
"            for (int l = 0; l < 32; l++) {\n"
"                int is = l / 16;\n"
"                int q1 = (int)((ql[l] & 0xF) | (((qh[l]>>0)&3)<<4)) - 32;\n"
"                int q2 = (int)((ql[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)) - 32;\n"
"                int q3 = (int)((ql[l] >> 4) | (((qh[l]>>4)&3)<<4)) - 32;\n"
"                int q4 = (int)((ql[l+32] >> 4) | (((qh[l]>>6)&3)<<4)) - 32;\n"
"                partial += d * sc[is+0] * q1 * xb[half*128+l];\n"
"                partial += d * sc[is+2] * q2 * xb[half*128+l+32];\n"
"                partial += d * sc[is+4] * q3 * xb[half*128+l+64];\n"
"                partial += d * sc[is+6] * q4 * xb[half*128+l+96];\n"
"            }\n"
"            ql += 64; qh += 32; sc += 8;\n"
"        }\n"
"        sum += partial;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"\n"
"/* ---- Chunked batch_matvec_q6_K: processes 4 tokens per block ---- */\n"
"/* Grid: (out_dim/8, ceil(n_tokens/4)), blockDim: 256.\n"
"   8 warps x 8 rows. Each warp loads 1 row of weights, processes 4 tokens. */\n"
"__global__ void batch_matvec_q6_K_x4(float *output, const unsigned char *mat,\n"
"                                      const float *input,\n"
"                                      int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= out_dim) return;\n"
"    int t0 = blockIdx.y * 4;\n"
"    int nt = n_tokens - t0;\n"
"    if (nt > 4) nt = 4;\n"
"    if (nt <= 0) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 210;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    float sum[4];\n"
"    for (int t = 0; t < nt; t++) sum[t] = 0.0f;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 210;\n"
"        const unsigned char *ql = bp;\n"
"        const unsigned char *qh = bp + 128;\n"
"        const signed char *sc = (const signed char *)(bp + 192);\n"
"        float d = half_to_float(*(const half_raw *)(bp + 208));\n"
"        for (int half = 0; half < 2; half++) {\n"
"            for (int l = 0; l < 32; l++) {\n"
"                int is = l / 16;\n"
"                int q1 = (int)((ql[l] & 0xF) | (((qh[l]>>0)&3)<<4)) - 32;\n"
"                int q2 = (int)((ql[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)) - 32;\n"
"                int q3 = (int)((ql[l] >> 4) | (((qh[l]>>4)&3)<<4)) - 32;\n"
"                int q4 = (int)((ql[l+32] >> 4) | (((qh[l]>>6)&3)<<4)) - 32;\n"
"                float w0 = d * sc[is+0] * q1;\n"
"                float w1 = d * sc[is+2] * q2;\n"
"                float w2 = d * sc[is+4] * q3;\n"
"                float w3 = d * sc[is+6] * q4;\n"
"                int b256 = b * 256 + half * 128;\n"
"                for (int t = 0; t < nt; t++) {\n"
"                    const float *x = input + (size_t)(t0+t) * in_dim + b256 + l;\n"
"                    sum[t] += w0 * x[0] + w1 * x[32] + w2 * x[64] + w3 * x[96];\n"
"                }\n"
"            }\n"
"            ql += 64; qh += 32; sc += 8;\n"
"        }\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        for (int t = 0; t < nt; t++)\n"
"            sum[t] += __shfl_down_sync(0xFFFFFFFF, sum[t], offset);\n"
"    if (lane == 0)\n"
"        for (int t = 0; t < nt; t++)\n"
"            output[(size_t)(t0+t) * out_dim + row] = sum[t];\n"
"}\n"
"\n"
"/* ---- dequant_iq2_s_to_f16: IQ2_S weight matrix -> F16 ---- */\n"
"__global__ void dequant_iq2_s_to_f16(half_raw *dst, const unsigned char *mat,\n"
"                                       int n_rows, int n_cols) {\n"
"    int row = blockIdx.x * 8 + threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    if (row >= n_rows) return;\n"
"    int nb = n_cols / 256;\n"
"    int row_bytes = nb * 82;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    half_raw *dst_row = dst + (size_t)row * n_cols;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 82;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *qh = bp + 66;\n"
"        const unsigned char *scales = bp + 74;\n"
"        const unsigned char *signs = bp + 34;\n"
"        half_raw *d_b = dst_row + b * 256;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            float db0 = d * (0.5f + (float)(scales[ib32] & 0xf)) * 0.25f;\n"
"            float db1 = d * (0.5f + (float)(scales[ib32] >>  4)) * 0.25f;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                float dl = (l < 2) ? db0 : db1;\n"
"                int grid_idx = qs[l] | ((qh[ib32] << (8-2*l)) & 0x300);\n"
"                unsigned long long grid_val = iq2s_grid_dev[grid_idx];\n"
"                unsigned char s = signs[l];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    float w = dl * (float)(unsigned char)(grid_val >> (8*j)) * ((s & (1 << j)) ? -1.0f : 1.0f);\n"
"                    d_b[ib32 * 32 + l * 8 + j] = float_to_half(w);\n"
"                }\n"
"            }\n"
"            qs += 4;\n"
"            signs += 4;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- dequant_iq2_xxs_to_f16: IQ2_XXS expert slice -> F16 ---- */\n"
"/* Writes rows_per_expert x n_cols F16 values to dst */\n"
"__global__ void dequant_iq2_xxs_to_f16(half_raw *dst, const unsigned char *mat,\n"
"                                         int rows, int cols) {\n"
"    /* one thread per 32-element sub-block (ib32); coalesced F16 writes */\n"
"    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int units_per_row = cols >> 5;            /* (cols/256)*8 = cols/32 */\n"
"    long long total = (long long)rows * units_per_row;\n"
"    if (tid >= total) return;\n"
"    int row = (int)(tid / units_per_row);\n"
"    int u = (int)(tid - (long long)row * units_per_row);\n"
"    int b = u >> 3;        /* which 256-element block */\n"
"    int ib32 = u & 7;      /* sub-block within it */\n"
"    int nb = cols / 256;\n"
"    int row_bytes = nb * 66;\n"
"    const unsigned char *bp = mat + (size_t)row * row_bytes + (size_t)b * 66;\n"
"    float d = half_to_float(*(const half_raw *)bp);\n"
"    const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"    unsigned int aux0 = qs[4*ib32] | ((unsigned int)qs[4*ib32+1] << 16);\n"
"    unsigned int aux1 = qs[4*ib32+2] | ((unsigned int)qs[4*ib32+3] << 16);\n"
"    float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;\n"
"    const unsigned char *aux8 = (const unsigned char *)&aux0;\n"
"    half_raw *d_b = dst + (size_t)row * cols + (size_t)b * 256 + ib32 * 32;\n"
"    for (int l = 0; l < 4; l++) {\n"
"        unsigned long long gv = iq2xxs_grid_dev[aux8[l]];\n"
"        unsigned char signs = ksigns_iq2xs_dev[(aux1 >> (7*l)) & 127];\n"
"        for (int j = 0; j < 8; j++) {\n"
"            float w = db * (float)(unsigned char)(gv >> (8*j)) * ((signs & (1 << j)) ? -1.0f : 1.0f);\n"
"            d_b[l * 8 + j] = float_to_half(w);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- dequant_iq2_xxs_pair_to_f16: two IQ2_XXS expert slices -> two F16 buffers ---- */\n"
"__global__ void dequant_iq2_xxs_pair_to_f16(half_raw *dst0, half_raw *dst1,\n"
"                                             const unsigned char *mat0, const unsigned char *mat1,\n"
"                                             int rows, int cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= rows) return;\n"
"    int nb = cols / 256;\n"
"    int row_bytes = nb * 66;\n"
"    const unsigned char *row_ptr0 = mat0 + (size_t)row * row_bytes;\n"
"    const unsigned char *row_ptr1 = mat1 + (size_t)row * row_bytes;\n"
"    half_raw *dst_row0 = dst0 + (size_t)row * cols;\n"
"    half_raw *dst_row1 = dst1 + (size_t)row * cols;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp0 = row_ptr0 + b * 66;\n"
"        const unsigned char *bp1 = row_ptr1 + b * 66;\n"
"        half_raw *d_b0 = dst_row0 + b * 256;\n"
"        half_raw *d_b1 = dst_row1 + b * 256;\n"
"        float d0 = half_to_float(*(const half_raw *)bp0);\n"
"        float d1 = half_to_float(*(const half_raw *)bp1);\n"
"        const unsigned short *qs0 = (const unsigned short *)(bp0 + 2);\n"
"        const unsigned short *qs1 = (const unsigned short *)(bp1 + 2);\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            unsigned int aux00 = qs0[4*ib32] | ((unsigned int)qs0[4*ib32+1] << 16);\n"
"            unsigned int aux01 = qs0[4*ib32+2] | ((unsigned int)qs0[4*ib32+3] << 16);\n"
"            unsigned int aux10 = qs1[4*ib32] | ((unsigned int)qs1[4*ib32+1] << 16);\n"
"            unsigned int aux11 = qs1[4*ib32+2] | ((unsigned int)qs1[4*ib32+3] << 16);\n"
"            float db0 = d0 * (0.5f + (float)(aux01 >> 28)) * 0.25f;\n"
"            float db1 = d1 * (0.5f + (float)(aux11 >> 28)) * 0.25f;\n"
"            const unsigned char *aux80 = (const unsigned char *)&aux00;\n"
"            const unsigned char *aux81 = (const unsigned char *)&aux10;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                unsigned long long gv0 = iq2xxs_grid_dev[aux80[l]];\n"
"                unsigned long long gv1 = iq2xxs_grid_dev[aux81[l]];\n"
"                unsigned char signs0 = ksigns_iq2xs_dev[(aux01 >> (7*l)) & 127];\n"
"                unsigned char signs1 = ksigns_iq2xs_dev[(aux11 >> (7*l)) & 127];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    float w0 = db0 * (float)(unsigned char)(gv0 >> (8*j)) * ((signs0 & (1 << j)) ? -1.0f : 1.0f);\n"
"                    float w1 = db1 * (float)(unsigned char)(gv1 >> (8*j)) * ((signs1 & (1 << j)) ? -1.0f : 1.0f);\n"
"                    d_b0[ib32 * 32 + l * 8 + j] = float_to_half(w0);\n"
"                    d_b1[ib32 * 32 + l * 8 + j] = float_to_half(w1);\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- dequant_iq2_xxs_triplet_to_f16: gate/up/down IQ2_XXS slices -> F16 ---- */\n"
"__global__ void dequant_iq2_xxs_triplet_to_f16(half_raw *dst_gate, half_raw *dst_up, half_raw *dst_down,\n"
"                                                const unsigned char *gate, const unsigned char *up,\n"
"                                                const unsigned char *down, int n_embd, int expert_ff) {\n"
"    int lane = threadIdx.x & 31;\n"
"    int warp = threadIdx.x >> 5;\n"
"    int warps_per_cta = blockDim.x >> 5;\n"
"    int unit = blockIdx.x * warps_per_cta + warp;\n"
"    int nb_gu = n_embd / 256;\n"
"    int nb_d = expert_ff / 256;\n"
"    int gu_units = expert_ff * nb_gu;\n"
"    int d_units = n_embd * nb_d;\n"
"    int l = lane >> 3;\n"
"    int j = lane & 7;\n"
"    if (unit < gu_units) {\n"
"        int row = unit / nb_gu;\n"
"        int b = unit - row * nb_gu;\n"
"        int row_bytes = nb_gu * 66;\n"
"        const unsigned char *bp0 = gate + (size_t)row * row_bytes + b * 66;\n"
"        const unsigned char *bp1 = up + (size_t)row * row_bytes + b * 66;\n"
"        float d0 = half_to_float(*(const half_raw *)bp0);\n"
"        float d1 = half_to_float(*(const half_raw *)bp1);\n"
"        const unsigned short *qs0 = (const unsigned short *)(bp0 + 2);\n"
"        const unsigned short *qs1 = (const unsigned short *)(bp1 + 2);\n"
"        half_raw *dg_b = dst_gate + (size_t)row * n_embd + b * 256;\n"
"        half_raw *du_b = dst_up + (size_t)row * n_embd + b * 256;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            unsigned int aux00 = qs0[4*ib32] | ((unsigned int)qs0[4*ib32+1] << 16);\n"
"            unsigned int aux01 = qs0[4*ib32+2] | ((unsigned int)qs0[4*ib32+3] << 16);\n"
"            unsigned int aux10 = qs1[4*ib32] | ((unsigned int)qs1[4*ib32+1] << 16);\n"
"            unsigned int aux11 = qs1[4*ib32+2] | ((unsigned int)qs1[4*ib32+3] << 16);\n"
"            float db0 = d0 * (0.5f + (float)(aux01 >> 28)) * 0.25f;\n"
"            float db1 = d1 * (0.5f + (float)(aux11 >> 28)) * 0.25f;\n"
"            unsigned char idx0 = (unsigned char)((aux00 >> (8*l)) & 255);\n"
"            unsigned char idx1 = (unsigned char)((aux10 >> (8*l)) & 255);\n"
"            unsigned long long gv0 = iq2xxs_grid_dev[idx0];\n"
"            unsigned long long gv1 = iq2xxs_grid_dev[idx1];\n"
"            unsigned char s0 = ksigns_iq2xs_dev[(aux01 >> (7*l)) & 127];\n"
"            unsigned char s1 = ksigns_iq2xs_dev[(aux11 >> (7*l)) & 127];\n"
"            dg_b[ib32 * 32 + lane] = float_to_half(db0 * (float)(unsigned char)(gv0 >> (8*j)) * ((s0 & (1 << j)) ? -1.0f : 1.0f));\n"
"            du_b[ib32 * 32 + lane] = float_to_half(db1 * (float)(unsigned char)(gv1 >> (8*j)) * ((s1 & (1 << j)) ? -1.0f : 1.0f));\n"
"        }\n"
"    } else if (unit < gu_units + d_units) {\n"
"        unit -= gu_units;\n"
"        int row = unit / nb_d;\n"
"        int b = unit - row * nb_d;\n"
"        int row_bytes = nb_d * 66;\n"
"        const unsigned char *bp = down + (size_t)row * row_bytes + b * 66;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"        half_raw *dd_b = dst_down + (size_t)row * expert_ff + b * 256;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            unsigned int aux0 = qs[4*ib32] | ((unsigned int)qs[4*ib32+1] << 16);\n"
"            unsigned int aux1 = qs[4*ib32+2] | ((unsigned int)qs[4*ib32+3] << 16);\n"
"            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;\n"
"            unsigned char idx = (unsigned char)((aux0 >> (8*l)) & 255);\n"
"            unsigned long long gv = iq2xxs_grid_dev[idx];\n"
"            unsigned char signs = ksigns_iq2xs_dev[(aux1 >> (7*l)) & 127];\n"
"            dd_b[ib32 * 32 + lane] = float_to_half(db * (float)(unsigned char)(gv >> (8*j)) * ((signs & (1 << j)) ? -1.0f : 1.0f));\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ================= IQ2_XXS MMQ (mul_mat_id) kernels =================\n"
"   Grouped quantized MoE expert matmul: consume IQ2_XXS weights directly (no\n"
"   F16 materialization) + q8_1 int8 activations via mma.sync.m16n8k32.s8.s8.s32.\n"
"   Validated standalone in cuda/llm/mmq/. */\n"
"/* gather rows by token index + quantize to q8_1 (per-32 scale). grid=(K/32,total), block=32 */\n"
"/* Optimized gather+quantize: 32 threads, 8 elements/thread = 256 K elements/block. */\n"
"/* Grid: (K/256, total_rows). Requires K %% 256 == 0.                              */\n"
"__global__ void mmq_gather_quant_q8_1(const float *X, const int *ids_token,\n"
"                                       signed char *cxq8, float *cxs, int total_rows, int K) {\n"
"    int kb = blockIdx.x, c = blockIdx.y, t = threadIdx.x;\n"
"    if (c >= total_rows) return;\n"
"    int tok = ids_token[c];\n"
"    int k_base = kb * 256 + t;\n"
"    float v[8];\n"
"    for (int g = 0; g < 8; g++) {\n"
"        int k = k_base + g * 32;\n"
"        v[g] = (k < K) ? X[(size_t)tok*K + k] : 0.0f;\n"
"    }\n"
"    for (int g = 0; g < 8; g++) {\n"
"        float a = fabsf(v[g]);\n"
"        for (int o = 16; o > 0; o >>= 1)\n"
"            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));\n"
"        float scale = a / 127.0f, inv = scale > 0.0f ? 1.0f / scale : 0.0f;\n"
"        int q = (int)rintf(v[g] * inv); q = q < -127 ? -127 : (q > 127 ? 127 : q);\n"
"        cxq8[(size_t)c*K + k_base + g*32] = (signed char)q;\n"
"        if (t == 0) cxs[(size_t)c*(K/32) + kb*8 + g] = scale;\n"
"    }\n"
"}\n"
"/* quantize compact rows (already gathered) to q8_1. grid=(K/32,total), block=32 */\n"
"/* Optimized quantize: 32 threads, 8 elements/thread = 256 K elements/block. */\n"
"/* Grid: (K/256, total_rows). Requires K %% 256 == 0. Each group of 32 gets  */\n"
"/* its own scale (8 groups/block, 1 element/thread per group).              */\n"
"__global__ void mmq_quant_q8_1(const float *X, signed char *cxq8, float *cxs, int total_rows, int K) {\n"
"    int kb = blockIdx.x, c = blockIdx.y, t = threadIdx.x;\n"
"    if (c >= total_rows) return;\n"
"    int k_base = kb * 256 + t;\n"
"    float v[8];\n"
"    for (int g = 0; g < 8; g++) {\n"
"        int k = k_base + g * 32;\n"
"        v[g] = (k < K) ? X[(size_t)c*K + k] : 0.0f;\n"
"    }\n"
"    for (int g = 0; g < 8; g++) {\n"
"        float a = fabsf(v[g]);\n"
"        for (int o = 16; o > 0; o >>= 1)\n"
"            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));\n"
"        float scale = a / 127.0f, inv = scale > 0.0f ? 1.0f / scale : 0.0f;\n"
"        int q = (int)rintf(v[g] * inv); q = q < -127 ? -127 : (q > 127 ? 127 : q);\n"
"        cxq8[(size_t)c*K + k_base + g*32] = (signed char)q;\n"
"        if (t == 0) cxs[(size_t)c*(K/32) + kb*8 + g] = scale;\n"
"    }\n"
"}\n"
"static __device__ __forceinline__ int mmq_pack4(const signed char*p){\n"
"    return (p[0]&0xff)|((p[1]&0xff)<<8)|((p[2]&0xff)<<16)|((p[3]&0xff)<<24); }\n"
"/* grouped MMQ (decode-amortized): one dispatch over all experts (grid.z=n_experts,\n"
"   WN=4 warps/block). Each block decodes its 64 weight-rows for a sub-block ONCE\n"
"   and reuses across up to TG=4 token-groups (32 tokens) of its expert — the\n"
"   IQ2_XXS decode is the bottleneck, so amortizing it is the win (2.1x vs 8/block).\n"
"   Flattened work-list grid: blockIdx.y indexes a real (expert,group) pair in\n"
"   worklist[] (packed (e<<16)|g), so grid=(N/64, n_work, 1) launches exactly the\n"
"   non-empty blocks. This kills the ~80%% empty blocks AND the hot-expert tail that\n"
"   the old (N/64, ceil(max_tok/32), n_experts) grid produced on skewed routing\n"
"   (max_tok=163 vs mean 16). block=128 */\n"
"__global__ void mmq_iq2xxs_grouped(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 4;\n"
"    /* Stage the IQ2_XXS codebook into shared once per block (the per-lane random-index\n"
"       gather is decode's latency source in global mem). sSignMask[i] = the 8 sign bits\n"
"       of ksigns[i] expanded to 8 bytes of 0x00/0xFF, so the decode can negate via a\n"
"       branchless __vsub4 ((g^0xFF)-0xFF = -g) instead of a per-byte conditional multiply. */\n"
"    __shared__ unsigned long long sGrid[256]; __shared__ unsigned long long sSignMask[128];\n"
"    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=iq2xxs_grid_dev[i];\n"
"    for (int i=threadIdx.x;i<128;i+=blockDim.x){ unsigned char s=ksigns_iq2xs_dev[i];\n"
"        unsigned long long m=0;\n"
"        for(int j=0;j<8;j++) if(s&(1<<j)) m|=(unsigned long long)0xFFu<<(8*j);\n"
"        sSignMask[i]=m; }\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*66, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[32][36]; __shared__ float sXs[32];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[4][4]; for(int g=0;g<4;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;  /* 32-lane decode: 2 lanes/row */\n"
"            /* bm=1: block-major repacked layout (block bg=(sb/8) of all N rows contiguous), so a\n"
"               row-tile reads one 256-block coalesced (vs row_bytes-strided over-fetch). bm=0: row-major. */\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*66+(size_t)n*66)\n"
"                                          : (We+(size_t)n*row_bytes+(size_t)(sb/8)*66);\n"
"            float d=half_to_float(*(const half_raw*)bp); const unsigned short *qs=(const unsigned short*)(bp+2); int ib=sb&7;\n"
"            unsigned int a0=(unsigned int)qs[4*ib]|((unsigned int)qs[4*ib+1]<<16);\n"
"            unsigned int a1=(unsigned int)qs[4*ib+2]|((unsigned int)qs[4*ib+3]<<16);\n"
"            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;\n"
"            for(int l=half*2;l<half*2+2;l++){ unsigned char idx=(a0>>(8*l))&255;\n"
"                unsigned long long gv=sGrid[idx], m=sSignMask[(a1>>(7*l))&127];\n"
"                unsigned int glo=(unsigned int)gv, ghi=(unsigned int)(gv>>32);\n"
"                unsigned int mlo=(unsigned int)m,  mhi=(unsigned int)(m>>32);\n"
"                *(unsigned int*)&sW[r][l*8]   = __vsub4(glo^mlo, mlo);\n"
"                *(unsigned int*)&sW[r][l*8+4] = __vsub4(ghi^mhi, mhi); } }\n"
"        /* vectorized activation staging: 4 bytes (one int) per thread, cxq8/sX rows 4-aligned */\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        /* sW/sX rows are 32B and written via *(uint*); [k*4] offsets are 4-aligned,\n"
"           so read the int directly instead of mmq_pack4's per-byte reassembly. */\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4], qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_iq2xxs_grouped with TG=16 (128 tokens/block instead of 32):\n"
"   the decode-amortize kernel re-reads the weight ceil(n_tokens/(8*TG)) times, so a\n"
"   bigger TG cuts the redundant weight reads ~4x for large prefill batches. f[16][4]\n"
"   raises register use but stays within budget (TG=32 spills and is slower). Identical\n"
"   math to grouped; worklist groups by 128. */\n"
"__global__ void mmq_iq2xxs_grouped8(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 16;\n"
"    __shared__ unsigned long long sGrid[256]; __shared__ unsigned long long sSignMask[128];\n"
"    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=iq2xxs_grid_dev[i];\n"
"    for (int i=threadIdx.x;i<128;i+=blockDim.x){ unsigned char s=ksigns_iq2xs_dev[i];\n"
"        unsigned long long m=0;\n"
"        for(int j=0;j<8;j++) if(s&(1<<j)) m|=(unsigned long long)0xFFu<<(8*j);\n"
"        sSignMask[i]=m; }\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*66, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[128][36]; __shared__ float sXs[128];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[16][4]; for(int g=0;g<16;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*66+(size_t)n*66)\n"
"                                          : (We+(size_t)n*row_bytes+(size_t)(sb/8)*66);\n"
"            float d=half_to_float(*(const half_raw*)bp); const unsigned short *qs=(const unsigned short*)(bp+2); int ib=sb&7;\n"
"            unsigned int a0=(unsigned int)qs[4*ib]|((unsigned int)qs[4*ib+1]<<16);\n"
"            unsigned int a1=(unsigned int)qs[4*ib+2]|((unsigned int)qs[4*ib+3]<<16);\n"
"            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;\n"
"            for(int l=half*2;l<half*2+2;l++){ unsigned char idx=(a0>>(8*l))&255;\n"
"                unsigned long long gv=sGrid[idx], m=sSignMask[(a1>>(7*l))&127];\n"
"                unsigned int glo=(unsigned int)gv, ghi=(unsigned int)(gv>>32);\n"
"                unsigned int mlo=(unsigned int)m,  mhi=(unsigned int)(m>>32);\n"
"                *(unsigned int*)&sW[r][l*8]   = __vsub4(glo^mlo, mlo);\n"
"                *(unsigned int*)&sW[r][l*8+4] = __vsub4(ghi^mhi, mhi); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4], qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* weighted scatter: final[ids_token[c]] += cw[c]*out_compact[c]. grid=total, block=256 */\n"
"__global__ void mmq_scatter_weighted(float *final_out, const float *out_compact,\n"
"                                      const int *ids_token, const float *cw, int total_rows, int N) {\n"
"    int c = blockIdx.x; if (c>=total_rows) return;\n"
"    int tok = ids_token[c]; float w = cw[c];\n"
"    const float *s = out_compact + (size_t)c*N; float *d = final_out + (size_t)tok*N;\n"
"    for (int n=threadIdx.x;n<N;n+=blockDim.x) atomicAdd(&d[n], w*s[n]);\n"
"}\n"
"\n"
"/* ---- dequant_iq3_xxs_to_f16: IQ3_XXS expert slice -> F16 ---- */\n"
"__global__ void dequant_iq3_xxs_to_f16(half_raw *dst, const unsigned char *mat,\n"
"                                         int rows, int cols) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    if (row >= rows) return;\n"
"    int nb = cols / 256;\n"
"    int row_bytes = nb * 98;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    half_raw *dst_row = dst + (size_t)row * cols;\n"
"    for (int b = lane; b < nb; b += 32) {\n"
"        const unsigned char *bp = row_ptr + b * 98;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        const unsigned char *qs = bp + 2;\n"
"        const unsigned char *scales_and_signs = qs + 64;\n"
"        half_raw *d_b = dst_row + b * 256;\n"
"        for (int ib32 = 0; ib32 < 8; ib32++) {\n"
"            unsigned int aux32;\n"
"            memcpy(&aux32, scales_and_signs + 4*ib32, 4);\n"
"            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;\n"
"            for (int l = 0; l < 4; l++) {\n"
"                unsigned char sgn = (unsigned char)ksigns_iq2xs_dev[(aux32 >> (7*l)) & 127];\n"
"                unsigned int gv1 = iq3xxs_grid_dev[qs[2*l+0]];\n"
"                unsigned int gv2 = iq3xxs_grid_dev[qs[2*l+1]];\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    float w0 = db * (float)(unsigned char)(gv1 >> (8*j)) * ((sgn & (1 << j)) ? -1.0f : 1.0f);\n"
"                    float w1 = db * (float)(unsigned char)(gv2 >> (8*j)) * ((sgn & (1 << (j+4))) ? -1.0f : 1.0f);\n"
"                    d_b[ib32 * 32 + l * 8 + j] = float_to_half(w0);\n"
"                    d_b[ib32 * 32 + l * 8 + j + 4] = float_to_half(w1);\n"
"                }\n"
"            }\n"
"            qs += 8;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* Prefill variant of mmq_iq2xxs_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_iq2xxs_grouped32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    __shared__ unsigned long long sGrid[256]; __shared__ unsigned long long sSignMask[128];\n"
"    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=iq2xxs_grid_dev[i];\n"
"    for (int i=threadIdx.x;i<128;i+=blockDim.x){ unsigned char s=ksigns_iq2xs_dev[i];\n"
"        unsigned long long m=0;\n"
"        for(int j=0;j<8;j++) if(s&(1<<j)) m|=(unsigned long long)0xFFu<<(8*j);\n"
"        sSignMask[i]=m; }\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*66, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*66+(size_t)n*66)\n"
"                                          : (We+(size_t)n*row_bytes+(size_t)(sb/8)*66);\n"
"            float d=half_to_float(*(const half_raw*)bp); const unsigned short *qs=(const unsigned short*)(bp+2); int ib=sb&7;\n"
"            unsigned int a0=(unsigned int)qs[4*ib]|((unsigned int)qs[4*ib+1]<<16);\n"
"            unsigned int a1=(unsigned int)qs[4*ib+2]|((unsigned int)qs[4*ib+3]<<16);\n"
"            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;\n"
"            for(int l=half*2;l<half*2+2;l++){ unsigned char idx=(a0>>(8*l))&255;\n"
"                unsigned long long gv=sGrid[idx], m=sSignMask[(a1>>(7*l))&127];\n"
"                unsigned int glo=(unsigned int)gv, ghi=(unsigned int)(gv>>32);\n"
"                unsigned int mlo=(unsigned int)m,  mhi=(unsigned int)(m>>32);\n"
"                *(unsigned int*)&sW[r][l*8]   = __vsub4(glo^mlo, mlo);\n"
"                *(unsigned int*)&sW[r][l*8+4] = __vsub4(ghi^mhi, mhi); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4], qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Fused prefill variant of mmq_iq2xxs_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_iq2xxs_fused32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const float *input, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    __shared__ unsigned long long sGrid[256]; __shared__ unsigned long long sSignMask[128];\n"
"    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=iq2xxs_grid_dev[i];\n"
"    for (int i=threadIdx.x;i<128;i+=blockDim.x){ unsigned char s=ksigns_iq2xs_dev[i];\n"
"        unsigned long long m=0;\n"
"        for(int j=0;j<8;j++) if(s&(1<<j)) m|=(unsigned long long)0xFFu<<(8*j);\n"
"        sSignMask[i]=m; }\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*66, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*66+(size_t)n*66)\n"
"                                          : (We+(size_t)n*row_bytes+(size_t)(sb/8)*66);\n"
"            float d=half_to_float(*(const half_raw*)bp); const unsigned short *qs=(const unsigned short*)(bp+2); int ib=sb&7;\n"
"            unsigned int a0=(unsigned int)qs[4*ib]|((unsigned int)qs[4*ib+1]<<16);\n"
"            unsigned int a1=(unsigned int)qs[4*ib+2]|((unsigned int)qs[4*ib+3]<<16);\n"
"            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;\n"
"            for(int l=half*2;l<half*2+2;l++){ unsigned char idx=(a0>>(8*l))&255;\n"
"                unsigned long long gv=sGrid[idx], m=sSignMask[(a1>>(7*l))&127];\n"
"                unsigned int glo=(unsigned int)gv, ghi=(unsigned int)(gv>>32);\n"
"                unsigned int mlo=(unsigned int)m,  mhi=(unsigned int)(m>>32);\n"
"                *(unsigned int*)&sW[r][l*8]   = __vsub4(glo^mlo, mlo);\n"
"                *(unsigned int*)&sW[r][l*8+4] = __vsub4(ghi^mhi, mhi); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            float v0=(m<eb1)?input[(size_t)m*K+sb*32+j*4+0]:0.0f;\n"
"            float v1=(m<eb1)?input[(size_t)m*K+sb*32+j*4+1]:0.0f;\n"
"            float v2=(m<eb1)?input[(size_t)m*K+sb*32+j*4+2]:0.0f;\n"
"            float v3=(m<eb1)?input[(size_t)m*K+sb*32+j*4+3]:0.0f;\n"
"            float local_max=fmaxf(fabsf(v0),fmaxf(fabsf(v1),fmaxf(fabsf(v2),fabsf(v3))));\n"
"            unsigned mask=0xFF<<((threadIdx.x&31)&~7);\n"
"            float a=local_max;\n"
"            for(int o=4;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(mask,a,o));\n"
"            float scale=a/127.0f,inv=scale>0.0f?1.0f/scale:0.0f;\n"
"            int q0=(int)rintf(v0*inv);q0=q0<-127?-127:(q0>127?127:q0);\n"
"            int q1=(int)rintf(v1*inv);q1=q1<-127?-127:(q1>127?127:q1);\n"
"            int q2=(int)rintf(v2*inv);q2=q2<-127?-127:(q2>127?127:q2);\n"
"            int q3=(int)rintf(v3*inv);q3=q3<-127?-127:(q3>127?127:q3);\n"
"            int packed=(unsigned char)q0|((unsigned char)q1<<8)|((unsigned char)q2<<16)|((unsigned char)q3<<24);\n"
"            *(int*)&sX[tt][j*4]=packed;\n"
"            if(j==0)sXs[tt]=scale; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4], qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* IQ3_XXS int8 tensor-core MMQ (multi-token grouped, TG=4, 32 tok/block). */\n"
"__global__ void mmq_iq3xxs_grouped(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 4;\n"
"    __shared__ unsigned int sGrid[256]; __shared__ unsigned long long sSignMask[128];\n"
"    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=iq3xxs_grid_dev[i];\n"
"    for (int i=threadIdx.x;i<128;i+=blockDim.x){ unsigned char s=ksigns_iq2xs_dev[i];\n"
"        unsigned long long m=0;\n"
"        for(int j=0;j<8;j++) if(s&(1<<j)) m|=(unsigned long long)0xFFu<<(8*j);\n"
"        sSignMask[i]=m; }\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*98, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[32][36]; __shared__ float sXs[32];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[4][4]; for(int g=0;g<4;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*98+(size_t)n*98)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*98);\n"
"            float d=half_to_float(*(const half_raw*)bp);\n"
"            const unsigned char *qs = bp + 2;\n"
"            const unsigned short *sas16 = (const unsigned short *)(bp + 66); int ib=sb&7;\n"
"            unsigned int sas_word = (unsigned int)sas16[2*ib] | ((unsigned int)sas16[2*ib+1] << 16);\n"
"            if (half==0) sWs[r]=d*(0.5f+(float)(sas_word>>28))*0.5f;\n"
"            for(int l=half*2;l<half*2+2;l++){\n"
"                unsigned int gv1=sGrid[qs[8*ib+2*l+0]], gv2=sGrid[qs[8*ib+2*l+1]];\n"
"                unsigned long long m=sSignMask[(sas_word>>(7*l))&127];\n"
"                unsigned int mlo=(unsigned int)m, mhi=(unsigned int)(m>>32);\n"
"                *(unsigned int*)&sW[r][l*8]   = __vsub4(gv1^mlo, mlo);\n"
"                *(unsigned int*)&sW[r][l*8+4] = __vsub4(gv2^mhi, mhi); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4], qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_iq3xxs_grouped with TG=16 (128 tokens/block). */\n"
"__global__ void mmq_iq3xxs_grouped8(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 16;\n"
"    __shared__ unsigned int sGrid[256]; __shared__ unsigned long long sSignMask[128];\n"
"    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=iq3xxs_grid_dev[i];\n"
"    for (int i=threadIdx.x;i<128;i+=blockDim.x){ unsigned char s=ksigns_iq2xs_dev[i];\n"
"        unsigned long long m=0;\n"
"        for(int j=0;j<8;j++) if(s&(1<<j)) m|=(unsigned long long)0xFFu<<(8*j);\n"
"        sSignMask[i]=m; }\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*98, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[128][36]; __shared__ float sXs[128];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[16][4]; for(int g=0;g<16;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*98+(size_t)n*98)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*98);\n"
"            float d=half_to_float(*(const half_raw*)bp);\n"
"            const unsigned char *qs = bp + 2;\n"
"            const unsigned short *sas16 = (const unsigned short *)(bp + 66); int ib=sb&7;\n"
"            unsigned int sas_word = (unsigned int)sas16[2*ib] | ((unsigned int)sas16[2*ib+1] << 16);\n"
"            if (half==0) sWs[r]=d*(0.5f+(float)(sas_word>>28))*0.5f;\n"
"            for(int l=half*2;l<half*2+2;l++){\n"
"                unsigned int gv1=sGrid[qs[8*ib+2*l+0]], gv2=sGrid[qs[8*ib+2*l+1]];\n"
"                unsigned long long m=sSignMask[(sas_word>>(7*l))&127];\n"
"                unsigned int mlo=(unsigned int)m, mhi=(unsigned int)(m>>32);\n"
"                *(unsigned int*)&sW[r][l*8]   = __vsub4(gv1^mlo, mlo);\n"
"                *(unsigned int*)&sW[r][l*8+4] = __vsub4(gv2^mhi, mhi); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4], qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* Prefill variant of mmq_iq3xxs_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_iq3xxs_grouped32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    __shared__ unsigned int sGrid[256]; __shared__ unsigned long long sSignMask[128];\n"
"    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=iq3xxs_grid_dev[i];\n"
"    for (int i=threadIdx.x;i<128;i+=blockDim.x){ unsigned char s=ksigns_iq2xs_dev[i];\n"
"        unsigned long long m=0;\n"
"        for(int j=0;j<8;j++) if(s&(1<<j)) m|=(unsigned long long)0xFFu<<(8*j);\n"
"        sSignMask[i]=m; }\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*98, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*98+(size_t)n*98)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*98);\n"
"            float d=half_to_float(*(const half_raw*)bp);\n"
"            const unsigned char *qs = bp + 2;\n"
"            const unsigned short *sas16 = (const unsigned short *)(bp + 66); int ib=sb&7;\n"
"            unsigned int sas_word = (unsigned int)sas16[2*ib] | ((unsigned int)sas16[2*ib+1] << 16);\n"
"            if (half==0) sWs[r]=d*(0.5f+(float)(sas_word>>28))*0.5f;\n"
"            for(int l=half*2;l<half*2+2;l++){\n"
"                unsigned int gv1=sGrid[qs[8*ib+2*l+0]], gv2=sGrid[qs[8*ib+2*l+1]];\n"
"                unsigned long long m=sSignMask[(sas_word>>(7*l))&127];\n"
"                unsigned int mlo=(unsigned int)m, mhi=(unsigned int)(m>>32);\n"
"                *(unsigned int*)&sW[r][l*8]   = __vsub4(gv1^mlo, mlo);\n"
"                *(unsigned int*)&sW[r][l*8+4] = __vsub4(gv2^mhi, mhi); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4], qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Fused prefill variant of mmq_iq3xxs_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_iq3xxs_fused32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const float *input, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    __shared__ unsigned int sGrid[256]; __shared__ unsigned long long sSignMask[128];\n"
"    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=iq3xxs_grid_dev[i];\n"
"    for (int i=threadIdx.x;i<128;i+=blockDim.x){ unsigned char s=ksigns_iq2xs_dev[i];\n"
"        unsigned long long m=0;\n"
"        for(int j=0;j<8;j++) if(s&(1<<j)) m|=(unsigned long long)0xFFu<<(8*j);\n"
"        sSignMask[i]=m; }\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*98, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*98+(size_t)n*98)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*98);\n"
"            float d=half_to_float(*(const half_raw*)bp);\n"
"            const unsigned char *qs = bp + 2;\n"
"            const unsigned short *sas16 = (const unsigned short *)(bp + 66); int ib=sb&7;\n"
"            unsigned int sas_word = (unsigned int)sas16[2*ib] | ((unsigned int)sas16[2*ib+1] << 16);\n"
"            if (half==0) sWs[r]=d*(0.5f+(float)(sas_word>>28))*0.5f;\n"
"            for(int l=half*2;l<half*2+2;l++){\n"
"                unsigned int gv1=sGrid[qs[8*ib+2*l+0]], gv2=sGrid[qs[8*ib+2*l+1]];\n"
"                unsigned long long m=sSignMask[(sas_word>>(7*l))&127];\n"
"                unsigned int mlo=(unsigned int)m, mhi=(unsigned int)(m>>32);\n"
"                *(unsigned int*)&sW[r][l*8]   = __vsub4(gv1^mlo, mlo);\n"
"                *(unsigned int*)&sW[r][l*8+4] = __vsub4(gv2^mhi, mhi); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            float v0=(m<eb1)?input[(size_t)m*K+sb*32+j*4+0]:0.0f;\n"
"            float v1=(m<eb1)?input[(size_t)m*K+sb*32+j*4+1]:0.0f;\n"
"            float v2=(m<eb1)?input[(size_t)m*K+sb*32+j*4+2]:0.0f;\n"
"            float v3=(m<eb1)?input[(size_t)m*K+sb*32+j*4+3]:0.0f;\n"
"            float local_max=fmaxf(fabsf(v0),fmaxf(fabsf(v1),fmaxf(fabsf(v2),fabsf(v3))));\n"
"            unsigned mask=0xFF<<((threadIdx.x&31)&~7);\n"
"            float a=local_max;\n"
"            for(int o=4;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(mask,a,o));\n"
"            float scale=a/127.0f,inv=scale>0.0f?1.0f/scale:0.0f;\n"
"            int q0=(int)rintf(v0*inv);q0=q0<-127?-127:(q0>127?127:q0);\n"
"            int q1=(int)rintf(v1*inv);q1=q1<-127?-127:(q1>127?127:q1);\n"
"            int q2=(int)rintf(v2*inv);q2=q2<-127?-127:(q2>127?127:q2);\n"
"            int q3=(int)rintf(v3*inv);q3=q3<-127?-127:(q3>127?127:q3);\n"
"            int packed=(unsigned char)q0|((unsigned char)q1<<8)|((unsigned char)q2<<16)|((unsigned char)q3<<24);\n"
"            *(int*)&sX[tt][j*4]=packed;\n"
"            if(j==0)sXs[tt]=scale; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4], qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* ---- 19g. mmq_q2_K_grouped: Q2_K weight x Q8_1 activation via int8 tensor-core MMQ ---- */\n"
"/* Q2_K block: 84 bytes = scales[16] + qs[64] + d(f16) + dmin(f16), 256 elements.           */\n"
"/* 2-MMA approach: weight = d*dl*qs_val - dmin*ml. C_plus = sum(dl*qs_val*x),                */\n"
"/* C_minus = sum(ml*x). Result = d*d8*C_plus - dmin*d8*C_minus.                              */\n"
"__global__ void mmq_q2_K_grouped(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 4;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*84, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW_plus[64][32]; __shared__ signed char sW_minus[64][32];\n"
"    __shared__ float sWs_d[64], sWs_dmin[64];\n"
"    __shared__ signed char sX[32][36]; __shared__ float sXs[32];\n"
"    float f[4][4]; for(int g=0;g<4;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*84+(size_t)n*84)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*84);\n"
"            float d_val  = half_to_float(*(const half_raw*)(bp + 80));\n"
"            float dmin_val = half_to_float(*(const half_raw*)(bp + 82));\n"
"            const unsigned char *scales = bp;\n"
"            const unsigned char *qs_full = bp + 16;\n"
"            int ib = sb & 7;\n"
"            int n0_qs = (ib >> 2) * 32;\n"
"            int shift = (ib & 3) * 2;\n"
"            const unsigned char *qs = qs_full + n0_qs;\n"
"            int sc0 = scales[2*ib], sc1 = scales[2*ib+1];\n"
"            int dl0 = sc0 & 0xF, ml0 = sc0 >> 4;\n"
"            int dl1 = sc1 & 0xF, ml1 = sc1 >> 4;\n"
"            if (half==0){ sWs_d[r]=d_val; sWs_dmin[r]=dmin_val; }\n"
"            int dl = half ? dl1 : dl0;\n"
"            int ml = half ? ml1 : ml0;\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int raw = (qs[koff+k] >> shift) & 3;\n"
"                sW_plus[r][koff+k] = (signed char)(dl * raw);\n"
"                sW_minus[r][koff+k] = (signed char)ml; } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW_plus[wr+gid][tid*4],   qp1=*(const int*)&sW_plus[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW_plus[wr+gid][tid*4+16], qp3=*(const int*)&sW_plus[wr+gid+8][tid*4+16];\n"
"        int qm0=*(const int*)&sW_minus[wr+gid][tid*4],   qm1=*(const int*)&sW_minus[wr+gid+8][tid*4];\n"
"        int qm2=*(const int*)&sW_minus[wr+gid][tid*4+16], qm3=*(const int*)&sW_minus[wr+gid+8][tid*4+16];\n"
"        float wr_d_0=sWs_d[wr+gid], wr_d_8=sWs_d[wr+gid+8];\n"
"        float wr_m_0=sWs_dmin[wr+gid], wr_m_8=sWs_dmin[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0, cm0=0,cm1=0,cm2=0,cm3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cm0),\"=r\"(cm1),\"=r\"(cm2),\"=r\"(cm3)\n"
"                :\"r\"(qm0),\"r\"(qm1),\"r\"(qm2),\"r\"(qm3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_d_0*xc0*(float)cp0 - wr_m_0*xc0*(float)cm0;\n"
"            f[g][1]+=wr_d_0*xc1*(float)cp1 - wr_m_0*xc1*(float)cm1;\n"
"            f[g][2]+=wr_d_8*xc0*(float)cp2 - wr_m_8*xc0*(float)cm2;\n"
"            f[g][3]+=wr_d_8*xc1*(float)cp3 - wr_m_8*xc1*(float)cm3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_q2_K_grouped with TG=16 (128 tokens/block). */\n"
"__global__ void mmq_q2_K_grouped8(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 16;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*84, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW_plus[64][32]; __shared__ signed char sW_minus[64][32];\n"
"    __shared__ float sWs_d[64], sWs_dmin[64];\n"
"    __shared__ signed char sX[128][36]; __shared__ float sXs[128];\n"
"    float f[16][4]; for(int g=0;g<16;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*84+(size_t)n*84)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*84);\n"
"            float d_val  = half_to_float(*(const half_raw*)(bp + 80));\n"
"            float dmin_val = half_to_float(*(const half_raw*)(bp + 82));\n"
"            const unsigned char *scales = bp;\n"
"            const unsigned char *qs_full = bp + 16;\n"
"            int ib = sb & 7;\n"
"            int n0_qs = (ib >> 2) * 32;\n"
"            int shift = (ib & 3) * 2;\n"
"            const unsigned char *qs = qs_full + n0_qs;\n"
"            int sc0 = scales[2*ib], sc1 = scales[2*ib+1];\n"
"            int dl0 = sc0 & 0xF, ml0 = sc0 >> 4;\n"
"            int dl1 = sc1 & 0xF, ml1 = sc1 >> 4;\n"
"            if (half==0){ sWs_d[r]=d_val; sWs_dmin[r]=dmin_val; }\n"
"            int dl = half ? dl1 : dl0;\n"
"            int ml = half ? ml1 : ml0;\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int raw = (qs[koff+k] >> shift) & 3;\n"
"                sW_plus[r][koff+k] = (signed char)(dl * raw);\n"
"                sW_minus[r][koff+k] = (signed char)ml; } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW_plus[wr+gid][tid*4],   qp1=*(const int*)&sW_plus[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW_plus[wr+gid][tid*4+16], qp3=*(const int*)&sW_plus[wr+gid+8][tid*4+16];\n"
"        int qm0=*(const int*)&sW_minus[wr+gid][tid*4],   qm1=*(const int*)&sW_minus[wr+gid+8][tid*4];\n"
"        int qm2=*(const int*)&sW_minus[wr+gid][tid*4+16], qm3=*(const int*)&sW_minus[wr+gid+8][tid*4+16];\n"
"        float wr_d_0=sWs_d[wr+gid], wr_d_8=sWs_d[wr+gid+8];\n"
"        float wr_m_0=sWs_dmin[wr+gid], wr_m_8=sWs_dmin[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0, cm0=0,cm1=0,cm2=0,cm3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cm0),\"=r\"(cm1),\"=r\"(cm2),\"=r\"(cm3)\n"
"                :\"r\"(qm0),\"r\"(qm1),\"r\"(qm2),\"r\"(qm3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_d_0*xc0*(float)cp0 - wr_m_0*xc0*(float)cm0;\n"
"            f[g][1]+=wr_d_0*xc1*(float)cp1 - wr_m_0*xc1*(float)cm1;\n"
"            f[g][2]+=wr_d_8*xc0*(float)cp2 - wr_m_8*xc0*(float)cm2;\n"
"            f[g][3]+=wr_d_8*xc1*(float)cp3 - wr_m_8*xc1*(float)cm3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* Prefill variant of mmq_q2_K_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_q2_K_grouped32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*84, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW_plus[64][32]; __shared__ signed char sW_minus[64][32];\n"
"    __shared__ float sWs_d[64], sWs_dmin[64];\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*84+(size_t)n*84)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*84);\n"
"            float d_val  = half_to_float(*(const half_raw*)(bp + 80));\n"
"            float dmin_val = half_to_float(*(const half_raw*)(bp + 82));\n"
"            const unsigned char *scales = bp;\n"
"            const unsigned char *qs_full = bp + 16;\n"
"            int ib = sb & 7;\n"
"            int n0_qs = (ib >> 2) * 32;\n"
"            int shift = (ib & 3) * 2;\n"
"            const unsigned char *qs = qs_full + n0_qs;\n"
"            int sc0 = scales[2*ib], sc1 = scales[2*ib+1];\n"
"            int dl0 = sc0 & 0xF, ml0 = sc0 >> 4;\n"
"            int dl1 = sc1 & 0xF, ml1 = sc1 >> 4;\n"
"            if (half==0){ sWs_d[r]=d_val; sWs_dmin[r]=dmin_val; }\n"
"            int dl = half ? dl1 : dl0;\n"
"            int ml = half ? ml1 : ml0;\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int raw = (qs[koff+k] >> shift) & 3;\n"
"                sW_plus[r][koff+k] = (signed char)(dl * raw);\n"
"                sW_minus[r][koff+k] = (signed char)ml; } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW_plus[wr+gid][tid*4],   qp1=*(const int*)&sW_plus[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW_plus[wr+gid][tid*4+16], qp3=*(const int*)&sW_plus[wr+gid+8][tid*4+16];\n"
"        int qm0=*(const int*)&sW_minus[wr+gid][tid*4],   qm1=*(const int*)&sW_minus[wr+gid+8][tid*4];\n"
"        int qm2=*(const int*)&sW_minus[wr+gid][tid*4+16], qm3=*(const int*)&sW_minus[wr+gid+8][tid*4+16];\n"
"        float wr_d_0=sWs_d[wr+gid], wr_d_8=sWs_d[wr+gid+8];\n"
"        float wr_m_0=sWs_dmin[wr+gid], wr_m_8=sWs_dmin[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0, cm0=0,cm1=0,cm2=0,cm3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cm0),\"=r\"(cm1),\"=r\"(cm2),\"=r\"(cm3)\n"
"                :\"r\"(qm0),\"r\"(qm1),\"r\"(qm2),\"r\"(qm3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_d_0*xc0*(float)cp0 - wr_m_0*xc0*(float)cm0;\n"
"            f[g][1]+=wr_d_0*xc1*(float)cp1 - wr_m_0*xc1*(float)cm1;\n"
"            f[g][2]+=wr_d_8*xc0*(float)cp2 - wr_m_8*xc0*(float)cm2;\n"
"            f[g][3]+=wr_d_8*xc1*(float)cp3 - wr_m_8*xc1*(float)cm3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* Fused prefill variant of mmq_q2_K_grouped with TG=32 (256 tokens/block).                    */\n"
"/* Loads F32 activation directly and quantizes to int8 on-the-fly.                              */\n"
"__global__ void mmq_q2_K_fused32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const float *input, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*84, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW_plus[64][32]; __shared__ signed char sW_minus[64][32];\n"
"    __shared__ float sWs_d[64], sWs_dmin[64];\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*84+(size_t)n*84)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*84);\n"
"            float d_val  = half_to_float(*(const half_raw*)(bp + 80));\n"
"            float dmin_val = half_to_float(*(const half_raw*)(bp + 82));\n"
"            const unsigned char *scales = bp;\n"
"            const unsigned char *qs_full = bp + 16;\n"
"            int ib = sb & 7;\n"
"            int n0_qs = (ib >> 2) * 32;\n"
"            int shift = (ib & 3) * 2;\n"
"            const unsigned char *qs = qs_full + n0_qs;\n"
"            int sc0 = scales[2*ib], sc1 = scales[2*ib+1];\n"
"            int dl0 = sc0 & 0xF, ml0 = sc0 >> 4;\n"
"            int dl1 = sc1 & 0xF, ml1 = sc1 >> 4;\n"
"            if (half==0){ sWs_d[r]=d_val; sWs_dmin[r]=dmin_val; }\n"
"            int dl = half ? dl1 : dl0;\n"
"            int ml = half ? ml1 : ml0;\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int raw = (qs[koff+k] >> shift) & 3;\n"
"                sW_plus[r][koff+k] = (signed char)(dl * raw);\n"
"                sW_minus[r][koff+k] = (signed char)ml; } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            float v0=(m<eb1)?input[(size_t)m*K+sb*32+j*4+0]:0.0f;\n"
"            float v1=(m<eb1)?input[(size_t)m*K+sb*32+j*4+1]:0.0f;\n"
"            float v2=(m<eb1)?input[(size_t)m*K+sb*32+j*4+2]:0.0f;\n"
"            float v3=(m<eb1)?input[(size_t)m*K+sb*32+j*4+3]:0.0f;\n"
"            float local_max=fmaxf(fabsf(v0),fmaxf(fabsf(v1),fmaxf(fabsf(v2),fabsf(v3))));\n"
"            unsigned mask=0xFF<<((threadIdx.x&31)&~7);\n"
"            float a=local_max;\n"
"            for(int o=4;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(mask,a,o));\n"
"            float scale=a/127.0f,inv=scale>0.0f?1.0f/scale:0.0f;\n"
"            int q0=(int)rintf(v0*inv);q0=q0<-127?-127:(q0>127?127:q0);\n"
"            int q1=(int)rintf(v1*inv);q1=q1<-127?-127:(q1>127?127:q1);\n"
"            int q2=(int)rintf(v2*inv);q2=q2<-127?-127:(q2>127?127:q2);\n"
"            int q3=(int)rintf(v3*inv);q3=q3<-127?-127:(q3>127?127:q3);\n"
"            int packed=(unsigned char)q0|((unsigned char)q1<<8)|((unsigned char)q2<<16)|((unsigned char)q3<<24);\n"
"            *(int*)&sX[tt][j*4]=packed;\n"
"            if(j==0)sXs[tt]=scale; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW_plus[wr+gid][tid*4],   qp1=*(const int*)&sW_plus[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW_plus[wr+gid][tid*4+16], qp3=*(const int*)&sW_plus[wr+gid+8][tid*4+16];\n"
"        int qm0=*(const int*)&sW_minus[wr+gid][tid*4],   qm1=*(const int*)&sW_minus[wr+gid+8][tid*4];\n"
"        int qm2=*(const int*)&sW_minus[wr+gid][tid*4+16], qm3=*(const int*)&sW_minus[wr+gid+8][tid*4+16];\n"
"        float wr_d_0=sWs_d[wr+gid], wr_d_8=sWs_d[wr+gid+8];\n"
"        float wr_m_0=sWs_dmin[wr+gid], wr_m_8=sWs_dmin[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0, cm0=0,cm1=0,cm2=0,cm3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cm0),\"=r\"(cm1),\"=r\"(cm2),\"=r\"(cm3)\n"
"                :\"r\"(qm0),\"r\"(qm1),\"r\"(qm2),\"r\"(qm3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_d_0*xc0*(float)cp0 - wr_m_0*xc0*(float)cm0;\n"
"            f[g][1]+=wr_d_0*xc1*(float)cp1 - wr_m_0*xc1*(float)cm1;\n"
"            f[g][2]+=wr_d_8*xc0*(float)cp2 - wr_m_8*xc0*(float)cm2;\n"
"            f[g][3]+=wr_d_8*xc1*(float)cp3 - wr_m_8*xc1*(float)cm3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* ---- 19g1b. mmq_q4_K_grouped: Q4_K weight x Q8_1 activation via int8 tensor-core MMQ ---- */\n"
"/* Q4_K block: 144 bytes = d(f16)+dmin(f16)+sc[12]+qs[128], 256 elements, 8 sub-blocks of 32. */\n"
"/* weight = d*sv*nibble - dmin*mv. sv(6-bit,<=63)*nibble(4-bit,<=15) overflows int8, so sv is  */\n"
"/* NOT folded: sW_plus=nibble, sW_minus=1, and d*sv / dmin*mv applied as floats after the      */\n"
"/* m16n8k32 MMA (which spans exactly one 32-elem sub-block). C_plus=sum(nibble*x), C_minus=sum(x). */\n"
"__global__ void mmq_q4_K_grouped(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 4;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*144, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW_plus[64][32]; __shared__ signed char sW_minus[64][32];\n"
"    __shared__ float sWs_dsv[64], sWs_dm[64];\n"
"    __shared__ signed char sX[32][36]; __shared__ float sXs[32];\n"
"    float f[4][4]; for(int g=0;g<4;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*144+(size_t)n*144)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*144);\n"
"            float d_val  = half_to_float(*(const half_raw*)(bp + 0));\n"
"            float dmin_val = half_to_float(*(const half_raw*)(bp + 2));\n"
"            const unsigned char *sc = bp + 4;\n"
"            const unsigned char *qs_full = bp + 16;\n"
"            int ib = sb & 7, sv, mv;\n"
"            if (ib < 4){ sv = sc[ib] & 0x3F; mv = sc[ib+4] & 0x3F; }\n"
"            else { int i4=ib-4; sv=(sc[8+i4]&0x0F)|(((sc[i4]>>6)&3)<<4); mv=(sc[8+i4]>>4)|(((sc[4+i4]>>6)&3)<<4); }\n"
"            const unsigned char *qs = qs_full + (size_t)(ib>>1)*32;\n"
"            int hi = ib & 1;\n"
"            if (half==0){ sWs_dsv[r]=d_val*(float)sv; sWs_dm[r]=dmin_val*(float)mv; }\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int nib = hi ? (qs[koff+k] >> 4) : (qs[koff+k] & 0xF);\n"
"                sW_plus[r][koff+k] = (signed char)nib;\n"
"                sW_minus[r][koff+k] = (signed char)1; } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW_plus[wr+gid][tid*4],   qp1=*(const int*)&sW_plus[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW_plus[wr+gid][tid*4+16], qp3=*(const int*)&sW_plus[wr+gid+8][tid*4+16];\n"
"        int qm0=*(const int*)&sW_minus[wr+gid][tid*4],   qm1=*(const int*)&sW_minus[wr+gid+8][tid*4];\n"
"        int qm2=*(const int*)&sW_minus[wr+gid][tid*4+16], qm3=*(const int*)&sW_minus[wr+gid+8][tid*4+16];\n"
"        float wr_dsv_0=sWs_dsv[wr+gid], wr_dsv_8=sWs_dsv[wr+gid+8];\n"
"        float wr_dm_0=sWs_dm[wr+gid], wr_dm_8=sWs_dm[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0, cm0=0,cm1=0,cm2=0,cm3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cm0),\"=r\"(cm1),\"=r\"(cm2),\"=r\"(cm3)\n"
"                :\"r\"(qm0),\"r\"(qm1),\"r\"(qm2),\"r\"(qm3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_dsv_0*xc0*(float)cp0 - wr_dm_0*xc0*(float)cm0;\n"
"            f[g][1]+=wr_dsv_0*xc1*(float)cp1 - wr_dm_0*xc1*(float)cm1;\n"
"            f[g][2]+=wr_dsv_8*xc0*(float)cp2 - wr_dm_8*xc0*(float)cm2;\n"
"            f[g][3]+=wr_dsv_8*xc1*(float)cp3 - wr_dm_8*xc1*(float)cm3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_q4_K_grouped with TG=16 (128 tokens/block). */\n"
"__global__ void mmq_q4_K_grouped8(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 16;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*144, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW_plus[64][32]; __shared__ signed char sW_minus[64][32];\n"
"    __shared__ float sWs_dsv[64], sWs_dm[64];\n"
"    __shared__ signed char sX[128][36]; __shared__ float sXs[128];\n"
"    float f[16][4]; for(int g=0;g<16;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*144+(size_t)n*144)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*144);\n"
"            float d_val  = half_to_float(*(const half_raw*)(bp + 0));\n"
"            float dmin_val = half_to_float(*(const half_raw*)(bp + 2));\n"
"            const unsigned char *sc = bp + 4;\n"
"            const unsigned char *qs_full = bp + 16;\n"
"            int ib = sb & 7, sv, mv;\n"
"            if (ib < 4){ sv = sc[ib] & 0x3F; mv = sc[ib+4] & 0x3F; }\n"
"            else { int i4=ib-4; sv=(sc[8+i4]&0x0F)|(((sc[i4]>>6)&3)<<4); mv=(sc[8+i4]>>4)|(((sc[4+i4]>>6)&3)<<4); }\n"
"            const unsigned char *qs = qs_full + (size_t)(ib>>1)*32;\n"
"            int hi = ib & 1;\n"
"            if (half==0){ sWs_dsv[r]=d_val*(float)sv; sWs_dm[r]=dmin_val*(float)mv; }\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int nib = hi ? (qs[koff+k] >> 4) : (qs[koff+k] & 0xF);\n"
"                sW_plus[r][koff+k] = (signed char)nib;\n"
"                sW_minus[r][koff+k] = (signed char)1; } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW_plus[wr+gid][tid*4],   qp1=*(const int*)&sW_plus[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW_plus[wr+gid][tid*4+16], qp3=*(const int*)&sW_plus[wr+gid+8][tid*4+16];\n"
"        int qm0=*(const int*)&sW_minus[wr+gid][tid*4],   qm1=*(const int*)&sW_minus[wr+gid+8][tid*4];\n"
"        int qm2=*(const int*)&sW_minus[wr+gid][tid*4+16], qm3=*(const int*)&sW_minus[wr+gid+8][tid*4+16];\n"
"        float wr_dsv_0=sWs_dsv[wr+gid], wr_dsv_8=sWs_dsv[wr+gid+8];\n"
"        float wr_dm_0=sWs_dm[wr+gid], wr_dm_8=sWs_dm[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0, cm0=0,cm1=0,cm2=0,cm3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cm0),\"=r\"(cm1),\"=r\"(cm2),\"=r\"(cm3)\n"
"                :\"r\"(qm0),\"r\"(qm1),\"r\"(qm2),\"r\"(qm3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_dsv_0*xc0*(float)cp0 - wr_dm_0*xc0*(float)cm0;\n"
"            f[g][1]+=wr_dsv_0*xc1*(float)cp1 - wr_dm_0*xc1*(float)cm1;\n"
"            f[g][2]+=wr_dsv_8*xc0*(float)cp2 - wr_dm_8*xc0*(float)cm2;\n"
"            f[g][3]+=wr_dsv_8*xc1*(float)cp3 - wr_dm_8*xc1*(float)cm3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_q4_K_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_q4_K_grouped32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*144, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW_plus[64][32]; __shared__ signed char sW_minus[64][32];\n"
"    __shared__ float sWs_dsv[64], sWs_dm[64];\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*144+(size_t)n*144)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*144);\n"
"            float d_val  = half_to_float(*(const half_raw*)(bp + 0));\n"
"            float dmin_val = half_to_float(*(const half_raw*)(bp + 2));\n"
"            const unsigned char *sc = bp + 4;\n"
"            const unsigned char *qs_full = bp + 16;\n"
"            int ib = sb & 7, sv, mv;\n"
"            if (ib < 4){ sv = sc[ib] & 0x3F; mv = sc[ib+4] & 0x3F; }\n"
"            else { int i4=ib-4; sv=(sc[8+i4]&0x0F)|(((sc[i4]>>6)&3)<<4); mv=(sc[8+i4]>>4)|(((sc[4+i4]>>6)&3)<<4); }\n"
"            const unsigned char *qs = qs_full + (size_t)(ib>>1)*32;\n"
"            int hi = ib & 1;\n"
"            if (half==0){ sWs_dsv[r]=d_val*(float)sv; sWs_dm[r]=dmin_val*(float)mv; }\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int nib = hi ? (qs[koff+k] >> 4) : (qs[koff+k] & 0xF);\n"
"                sW_plus[r][koff+k] = (signed char)nib;\n"
"                sW_minus[r][koff+k] = (signed char)1; } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW_plus[wr+gid][tid*4],   qp1=*(const int*)&sW_plus[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW_plus[wr+gid][tid*4+16], qp3=*(const int*)&sW_plus[wr+gid+8][tid*4+16];\n"
"        int qm0=*(const int*)&sW_minus[wr+gid][tid*4],   qm1=*(const int*)&sW_minus[wr+gid+8][tid*4];\n"
"        int qm2=*(const int*)&sW_minus[wr+gid][tid*4+16], qm3=*(const int*)&sW_minus[wr+gid+8][tid*4+16];\n"
"        float wr_dsv_0=sWs_dsv[wr+gid], wr_dsv_8=sWs_dsv[wr+gid+8];\n"
"        float wr_dm_0=sWs_dm[wr+gid], wr_dm_8=sWs_dm[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0, cm0=0,cm1=0,cm2=0,cm3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cm0),\"=r\"(cm1),\"=r\"(cm2),\"=r\"(cm3)\n"
"                :\"r\"(qm0),\"r\"(qm1),\"r\"(qm2),\"r\"(qm3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_dsv_0*xc0*(float)cp0 - wr_dm_0*xc0*(float)cm0;\n"
"            f[g][1]+=wr_dsv_0*xc1*(float)cp1 - wr_dm_0*xc1*(float)cm1;\n"
"            f[g][2]+=wr_dsv_8*xc0*(float)cp2 - wr_dm_8*xc0*(float)cm2;\n"
"            f[g][3]+=wr_dsv_8*xc1*(float)cp3 - wr_dm_8*xc1*(float)cm3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* ---- 19g1c. mmq_q4_0_grouped: Q4_0 weight x Q8_1 activation via int8 tensor-core MMQ ---- */\n"
"/* Q4_0 block: 18 bytes = d(f16)+qs[16], 32 elements. weight=(nibble-8)*d. nibble-8 in    */\n"
"/* [-8,7] fits signed int8 -> folds straight in: sW=nibble-8, SINGLE m16n8k32 MMA (no       */\n"
"/* plus/minus split). d is per-32 block; result += d*d8*MMA(nibble-8, x_int8).              */\n"
"__global__ void mmq_q4_0_grouped(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 4;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/32, row_bytes = nb*18, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs_d[64];\n"
"    __shared__ signed char sX[32][36]; __shared__ float sXs[32];\n"
"    float f[4][4]; for(int g=0;g<4;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)sb*N*18+(size_t)n*18)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)sb*18);\n"
"            float d_val = half_to_float(*(const half_raw*)(bp + 0));\n"
"            const unsigned char *qs = bp + 2;\n"
"            if (half==0){ sWs_d[r]=d_val; }\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int nib = (half==0) ? (qs[k] & 0xF) : (qs[k] >> 4);\n"
"                sW[r][koff+k] = (signed char)(nib - 8); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW[wr+gid][tid*4],   qp1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW[wr+gid][tid*4+16], qp3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr_d_0=sWs_d[wr+gid], wr_d_8=sWs_d[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_d_0*xc0*(float)cp0;\n"
"            f[g][1]+=wr_d_0*xc1*(float)cp1;\n"
"            f[g][2]+=wr_d_8*xc0*(float)cp2;\n"
"            f[g][3]+=wr_d_8*xc1*(float)cp3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_q4_0_grouped with TG=16 (128 tokens/block). */\n"
"__global__ void mmq_q4_0_grouped8(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 16;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/32, row_bytes = nb*18, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs_d[64];\n"
"    __shared__ signed char sX[128][36]; __shared__ float sXs[128];\n"
"    float f[16][4]; for(int g=0;g<16;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)sb*N*18+(size_t)n*18)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)sb*18);\n"
"            float d_val = half_to_float(*(const half_raw*)(bp + 0));\n"
"            const unsigned char *qs = bp + 2;\n"
"            if (half==0){ sWs_d[r]=d_val; }\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int nib = (half==0) ? (qs[k] & 0xF) : (qs[k] >> 4);\n"
"                sW[r][koff+k] = (signed char)(nib - 8); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW[wr+gid][tid*4],   qp1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW[wr+gid][tid*4+16], qp3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr_d_0=sWs_d[wr+gid], wr_d_8=sWs_d[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_d_0*xc0*(float)cp0;\n"
"            f[g][1]+=wr_d_0*xc1*(float)cp1;\n"
"            f[g][2]+=wr_d_8*xc0*(float)cp2;\n"
"            f[g][3]+=wr_d_8*xc1*(float)cp3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_q4_0_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_q4_0_grouped32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/32, row_bytes = nb*18, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs_d[64];\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)sb*N*18+(size_t)n*18)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)sb*18);\n"
"            float d_val = half_to_float(*(const half_raw*)(bp + 0));\n"
"            const unsigned char *qs = bp + 2;\n"
"            if (half==0){ sWs_d[r]=d_val; }\n"
"            int koff = half * 16;\n"
"            for(int k=0;k<16;k++){\n"
"                int nib = (half==0) ? (qs[k] & 0xF) : (qs[k] >> 4);\n"
"                sW[r][koff+k] = (signed char)(nib - 8); } }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qp0=*(const int*)&sW[wr+gid][tid*4],   qp1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qp2=*(const int*)&sW[wr+gid][tid*4+16], qp3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr_d_0=sWs_d[wr+gid], wr_d_8=sWs_d[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int cp0=0,cp1=0,cp2=0,cp3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(cp0),\"=r\"(cp1),\"=r\"(cp2),\"=r\"(cp3)\n"
"                :\"r\"(qp0),\"r\"(qp1),\"r\"(qp2),\"r\"(qp3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr_d_0*xc0*(float)cp0;\n"
"            f[g][1]+=wr_d_0*xc1*(float)cp1;\n"
"            f[g][2]+=wr_d_8*xc0*(float)cp2;\n"
"            f[g][3]+=wr_d_8*xc1*(float)cp3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* ---- 19g2. mmq_q3_K_grouped: Q3_K weight x Q8_1 activation via int8 tensor-core MMQ ---- */\n"
"/* Q3_K block: 110 bytes = hmask[32]+qs[64]+scales[12]+d(f16), 256 elements.                */\n"
"/* Two dls per sub-block (dl0 k=0..15, dl1 k=16..31). Decode to int8 (-4..3), then           */\n"
"/* 2 x m16n8k16 MMA per sub-block (no plus/minus split since Q3_K is naturally signed).       */\n"
"__global__ void mmq_q3_K_grouped(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 4;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*110, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[32][36]; __shared__ float sXs[32];\n"
"    __shared__ signed char sW[64][32];\n"
"    __shared__ float sWs_db0[64]; __shared__ float sWs_db1[64];\n"
"    float f[4][4]; for(int g=0;g<4;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*110+(size_t)n*110)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*110);\n"
"            float d = half_to_float(*(const half_raw*)(bp + 108));\n"
"            int ib = sb & 7;\n"
"            /* Unpack 6-bit scales to int8 */\n"
"            unsigned int aux[4];\n"
"            *(unsigned int*)&aux[0] = *(const unsigned int*)(bp + 96);\n"
"            *(unsigned int*)&aux[1] = *(const unsigned int*)(bp + 100);\n"
"            *(unsigned int*)&aux[2] = *(const unsigned int*)(bp + 104);\n"
"            unsigned int tmp = aux[2];\n"
"            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);\n"
"            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);\n"
"            aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);\n"
"            aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);\n"
"            const signed char *scales = (const signed char *)aux;\n"
"            float dl0 = d * (float)(scales[2*ib] - 32);\n"
"            float dl1 = d * (float)(scales[2*ib+1] - 32);\n"
"            if (half==0){ sWs_db0[r]=dl0; sWs_db1[r]=dl1; }\n"
"            int q_base = (ib / 4) * 32;\n"
"            int shift = (ib % 4) * 2;\n"
"            int hm_bit = ib;\n"
"            const unsigned char *hm = bp;\n"
"            const unsigned char *qs = bp + 32;\n"
"            int koff = half * 16;\n"
"            for (int l = 0; l < 16; l++) {\n"
"                int idx = koff + l;\n"
"                int q_off = (idx < 16) ? 0 : 16;\n"
"                int qv = (qs[q_base + q_off + l] >> shift) & 3;\n"
"                int hmv = (hm[(idx < 16) ? l : (l + 16)] >> hm_bit) & 1;\n"
"                sW[r][idx] = (signed char)(hmv ? qv : qv - 4);\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        /* MMA_0: k=0..15 */\n"
"        int a0_h0 = *(const int*)&sW[wr+gid][tid*4];\n"
"        int a1_h0 = *(const int*)&sW[wr+gid+8][tid*4];\n"
"        int b0_h0 = *(const int*)&sX[gid][tid*4];\n"
"        float wr0_db0 = sWs_db0[wr+gid], wr8_db0 = sWs_db0[wr+gid+8];\n"
"        /* MMA_1: k=16..31 */\n"
"        int a0_h1 = *(const int*)&sW[wr+gid][16+tid*4];\n"
"        int a1_h1 = *(const int*)&sW[wr+gid+8][16+tid*4];\n"
"        int b0_h1 = *(const int*)&sX[gid][16+tid*4];\n"
"        float wr0_db1 = sWs_db1[wr+gid], wr8_db1 = sWs_db1[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int b0_x = *(const int*)&sX[g*8+gid][tid*4];\n"
"            int b1_x = *(const int*)&sX[g*8+gid][16+tid*4];\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h0),\"r\"(a1_h0),\"r\"(b0_h0),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db0*xc0*(float)c0; f[g][1]+=wr0_db0*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db0*xc0*(float)c2; f[g][3]+=wr8_db0*xc1*(float)c3; }\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h1),\"r\"(a1_h1),\"r\"(b1_x),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db1*xc0*(float)c0; f[g][1]+=wr0_db1*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db1*xc0*(float)c2; f[g][3]+=wr8_db1*xc1*(float)c3; }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_q3_K_grouped with TG=16 (128 tokens/block). */\n"
"__global__ void mmq_q3_K_grouped8(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 16;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*110, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[128][36]; __shared__ float sXs[128];\n"
"    __shared__ signed char sW[64][32];\n"
"    __shared__ float sWs_db0[64]; __shared__ float sWs_db1[64];\n"
"    float f[16][4]; for(int g=0;g<16;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*110+(size_t)n*110)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*110);\n"
"            float d = half_to_float(*(const half_raw*)(bp + 108));\n"
"            int ib = sb & 7;\n"
"            unsigned int aux[4];\n"
"            *(unsigned int*)&aux[0] = *(const unsigned int*)(bp + 96);\n"
"            *(unsigned int*)&aux[1] = *(const unsigned int*)(bp + 100);\n"
"            *(unsigned int*)&aux[2] = *(const unsigned int*)(bp + 104);\n"
"            unsigned int tmp = aux[2];\n"
"            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);\n"
"            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);\n"
"            aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);\n"
"            aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);\n"
"            const signed char *scales = (const signed char *)aux;\n"
"            float dl0 = d * (float)(scales[2*ib] - 32);\n"
"            float dl1 = d * (float)(scales[2*ib+1] - 32);\n"
"            if (half==0){ sWs_db0[r]=dl0; sWs_db1[r]=dl1; }\n"
"            int q_base = (ib / 4) * 32;\n"
"            int shift = (ib % 4) * 2;\n"
"            int hm_bit = ib;\n"
"            const unsigned char *hm = bp;\n"
"            const unsigned char *qs = bp + 32;\n"
"            int koff = half * 16;\n"
"            for (int l = 0; l < 16; l++) {\n"
"                int idx = koff + l;\n"
"                int q_off = (idx < 16) ? 0 : 16;\n"
"                int qv = (qs[q_base + q_off + l] >> shift) & 3;\n"
"                int hmv = (hm[(idx < 16) ? l : (l + 16)] >> hm_bit) & 1;\n"
"                sW[r][idx] = (signed char)(hmv ? qv : qv - 4);\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int a0_h0 = *(const int*)&sW[wr+gid][tid*4];\n"
"        int a1_h0 = *(const int*)&sW[wr+gid+8][tid*4];\n"
"        int b0_h0 = *(const int*)&sX[gid][tid*4];\n"
"        float wr0_db0 = sWs_db0[wr+gid], wr8_db0 = sWs_db0[wr+gid+8];\n"
"        int a0_h1 = *(const int*)&sW[wr+gid][16+tid*4];\n"
"        int a1_h1 = *(const int*)&sW[wr+gid+8][16+tid*4];\n"
"        int b0_h1 = *(const int*)&sX[gid][16+tid*4];\n"
"        float wr0_db1 = sWs_db1[wr+gid], wr8_db1 = sWs_db1[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int b0_x = *(const int*)&sX[g*8+gid][tid*4];\n"
"            int b1_x = *(const int*)&sX[g*8+gid][16+tid*4];\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h0),\"r\"(a1_h0),\"r\"(b0_h0),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db0*xc0*(float)c0; f[g][1]+=wr0_db0*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db0*xc0*(float)c2; f[g][3]+=wr8_db0*xc1*(float)c3; }\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h1),\"r\"(a1_h1),\"r\"(b1_x),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db1*xc0*(float)c0; f[g][1]+=wr0_db1*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db1*xc0*(float)c2; f[g][3]+=wr8_db1*xc1*(float)c3; }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_q3_K_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_q3_K_grouped32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*110, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32];\n"
"    __shared__ float sWs_db0[64]; __shared__ float sWs_db1[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*110+(size_t)n*110)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*110);\n"
"            float d = half_to_float(*(const half_raw*)(bp + 108));\n"
"            int ib = sb & 7;\n"
"            unsigned int aux[4];\n"
"            *(unsigned int*)&aux[0] = *(const unsigned int*)(bp + 96);\n"
"            *(unsigned int*)&aux[1] = *(const unsigned int*)(bp + 100);\n"
"            *(unsigned int*)&aux[2] = *(const unsigned int*)(bp + 104);\n"
"            unsigned int tmp = aux[2];\n"
"            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);\n"
"            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);\n"
"            aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);\n"
"            aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);\n"
"            const signed char *scales = (const signed char *)aux;\n"
"            float dl0 = d * (float)(scales[2*ib] - 32);\n"
"            float dl1 = d * (float)(scales[2*ib+1] - 32);\n"
"            if (half==0){ sWs_db0[r]=dl0; sWs_db1[r]=dl1; }\n"
"            int q_base = (ib / 4) * 32;\n"
"            int shift = (ib % 4) * 2;\n"
"            int hm_bit = ib;\n"
"            const unsigned char *hm = bp;\n"
"            const unsigned char *qs = bp + 32;\n"
"            int koff = half * 16;\n"
"            for (int l = 0; l < 16; l++) {\n"
"                int idx = koff + l;\n"
"                int q_off = (idx < 16) ? 0 : 16;\n"
"                int qv = (qs[q_base + q_off + l] >> shift) & 3;\n"
"                int hmv = (hm[(idx < 16) ? l : (l + 16)] >> hm_bit) & 1;\n"
"                sW[r][idx] = (signed char)(hmv ? qv : qv - 4);\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int a0_h0 = *(const int*)&sW[wr+gid][tid*4];\n"
"        int a1_h0 = *(const int*)&sW[wr+gid+8][tid*4];\n"
"        int b0_h0 = *(const int*)&sX[gid][tid*4];\n"
"        float wr0_db0 = sWs_db0[wr+gid], wr8_db0 = sWs_db0[wr+gid+8];\n"
"        int a0_h1 = *(const int*)&sW[wr+gid][16+tid*4];\n"
"        int a1_h1 = *(const int*)&sW[wr+gid+8][16+tid*4];\n"
"        int b0_h1 = *(const int*)&sX[gid][16+tid*4];\n"
"        float wr0_db1 = sWs_db1[wr+gid], wr8_db1 = sWs_db1[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int b0_x = *(const int*)&sX[g*8+gid][tid*4];\n"
"            int b1_x = *(const int*)&sX[g*8+gid][16+tid*4];\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h0),\"r\"(a1_h0),\"r\"(b0_h0),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db0*xc0*(float)c0; f[g][1]+=wr0_db0*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db0*xc0*(float)c2; f[g][3]+=wr8_db0*xc1*(float)c3; }\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h1),\"r\"(a1_h1),\"r\"(b1_x),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db1*xc0*(float)c0; f[g][1]+=wr0_db1*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db1*xc0*(float)c2; f[g][3]+=wr8_db1*xc1*(float)c3; }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Fused prefill variant of mmq_q3_K_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_q3_K_fused32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const float *input, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*110, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32];\n"
"    __shared__ float sWs_db0[64]; __shared__ float sWs_db1[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*110+(size_t)n*110)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*110);\n"
"            float d = half_to_float(*(const half_raw*)(bp + 108));\n"
"            int ib = sb & 7;\n"
"            unsigned int aux[4];\n"
"            *(unsigned int*)&aux[0] = *(const unsigned int*)(bp + 96);\n"
"            *(unsigned int*)&aux[1] = *(const unsigned int*)(bp + 100);\n"
"            *(unsigned int*)&aux[2] = *(const unsigned int*)(bp + 104);\n"
"            unsigned int tmp = aux[2];\n"
"            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);\n"
"            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);\n"
"            aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);\n"
"            aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);\n"
"            const signed char *scales = (const signed char *)aux;\n"
"            float dl0 = d * (float)(scales[2*ib] - 32);\n"
"            float dl1 = d * (float)(scales[2*ib+1] - 32);\n"
"            if (half==0){ sWs_db0[r]=dl0; sWs_db1[r]=dl1; }\n"
"            int q_base = (ib / 4) * 32;\n"
"            int shift = (ib % 4) * 2;\n"
"            int hm_bit = ib;\n"
"            const unsigned char *hm = bp;\n"
"            const unsigned char *qs = bp + 32;\n"
"            int koff = half * 16;\n"
"            for (int l = 0; l < 16; l++) {\n"
"                int idx = koff + l;\n"
"                int q_off = (idx < 16) ? 0 : 16;\n"
"                int qv = (qs[q_base + q_off + l] >> shift) & 3;\n"
"                int hmv = (hm[(idx < 16) ? l : (l + 16)] >> hm_bit) & 1;\n"
"                sW[r][idx] = (signed char)(hmv ? qv : qv - 4);\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            float v0=(m<eb1)?input[(size_t)m*K+sb*32+j*4+0]:0.0f;\n"
"            float v1=(m<eb1)?input[(size_t)m*K+sb*32+j*4+1]:0.0f;\n"
"            float v2=(m<eb1)?input[(size_t)m*K+sb*32+j*4+2]:0.0f;\n"
"            float v3=(m<eb1)?input[(size_t)m*K+sb*32+j*4+3]:0.0f;\n"
"            float local_max=fmaxf(fabsf(v0),fmaxf(fabsf(v1),fmaxf(fabsf(v2),fabsf(v3))));\n"
"            unsigned mask=0xFF<<((threadIdx.x&31)&~7);\n"
"            float a=local_max;\n"
"            for(int o=4;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(mask,a,o));\n"
"            float scale=a/127.0f,inv=scale>0.0f?1.0f/scale:0.0f;\n"
"            int q0=(int)rintf(v0*inv);q0=q0<-127?-127:(q0>127?127:q0);\n"
"            int q1=(int)rintf(v1*inv);q1=q1<-127?-127:(q1>127?127:q1);\n"
"            int q2=(int)rintf(v2*inv);q2=q2<-127?-127:(q2>127?127:q2);\n"
"            int q3=(int)rintf(v3*inv);q3=q3<-127?-127:(q3>127?127:q3);\n"
"            int packed=(unsigned char)q0|((unsigned char)q1<<8)|((unsigned char)q2<<16)|((unsigned char)q3<<24);\n"
"            *(int*)&sX[tt][j*4]=packed;\n"
"            if(j==0)sXs[tt]=scale; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int a0_h0 = *(const int*)&sW[wr+gid][tid*4];\n"
"        int a1_h0 = *(const int*)&sW[wr+gid+8][tid*4];\n"
"        int b0_h0 = *(const int*)&sX[gid][tid*4];\n"
"        float wr0_db0 = sWs_db0[wr+gid], wr8_db0 = sWs_db0[wr+gid+8];\n"
"        int a0_h1 = *(const int*)&sW[wr+gid][16+tid*4];\n"
"        int a1_h1 = *(const int*)&sW[wr+gid+8][16+tid*4];\n"
"        int b0_h1 = *(const int*)&sX[gid][16+tid*4];\n"
"        float wr0_db1 = sWs_db1[wr+gid], wr8_db1 = sWs_db1[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int b0_x = *(const int*)&sX[g*8+gid][tid*4];\n"
"            int b1_x = *(const int*)&sX[g*8+gid][16+tid*4];\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h0),\"r\"(a1_h0),\"r\"(b0_h0),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db0*xc0*(float)c0; f[g][1]+=wr0_db0*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db0*xc0*(float)c2; f[g][3]+=wr8_db0*xc1*(float)c3; }\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h1),\"r\"(a1_h1),\"r\"(b1_x),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db1*xc0*(float)c0; f[g][1]+=wr0_db1*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db1*xc0*(float)c2; f[g][3]+=wr8_db1*xc1*(float)c3; }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* ---- 19h. mmq_iq3_s_grouped: IQ3_S weight x Q8_1 activation via int8 tensor-core MMQ ---- */\n"
"/* IQ3_S block: 110 bytes = d(f16)+qs[64]+qh[8]+signs[32]+scales[4], 256 elements.           */\n"
"/* Single db per 32-element sub-block; 1 MMA per sub-block (k=32).                            */\n"
"__global__ void mmq_iq3_s_grouped(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 4;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*110, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[32][36]; __shared__ float sXs[32];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[4][4]; for(int g=0;g<4;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*110+(size_t)n*110)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*110);\n"
"            float d = half_to_float(*(const half_raw*)bp);\n"
"            int ib32 = sb & 7;\n"
"            int pair = ib32 / 2, sub_in_pair = ib32 & 1;\n"
"            if (half==0) sWs[r] = d * (float)(1 + 2 * (sub_in_pair ? (bp[106+pair] >> 4) : (bp[106+pair] & 0xf)));\n"
"            unsigned char qh_byte = bp[66+ib32];\n"
"            int qs_off = ib32 * 8, sign_off = ib32 * 4;\n"
"            for (int l = half*2; l < half*2+2; l++) {\n"
"                int g1_idx = (int)(bp[2+qs_off + 2*l + 0]) | ((qh_byte << (8-2*l)) & 256);\n"
"                int g2_idx = (int)(bp[2+qs_off + 2*l + 1]) | ((qh_byte << (7-2*l)) & 256);\n"
"                unsigned int gv1 = iq3s_grid_dev[g1_idx];\n"
"                unsigned int gv2 = iq3s_grid_dev[g2_idx];\n"
"                unsigned char sgn = bp[74+sign_off + l];\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    int w1 = (int)(unsigned char)(gv1 >> (8*j));\n"
"                    int w2 = (int)(unsigned char)(gv2 >> (8*j));\n"
"                    sW[r][l*8 + j]     = (signed char)((sgn & (1 << j))     ? -w1 : w1);\n"
"                    sW[r][l*8 + j + 4] = (signed char)((sgn & (1 << (j+4))) ? -w2 : w2);\n"
"                }\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4],   qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_iq3_s_grouped with TG=16 (128 tokens/block). */\n"
"__global__ void mmq_iq3_s_grouped8(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 16;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*110, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[128][36]; __shared__ float sXs[128];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[16][4]; for(int g=0;g<16;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*110+(size_t)n*110)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*110);\n"
"            float d = half_to_float(*(const half_raw*)bp);\n"
"            int ib32 = sb & 7;\n"
"            int pair = ib32 / 2, sub_in_pair = ib32 & 1;\n"
"            if (half==0) sWs[r] = d * (float)(1 + 2 * (sub_in_pair ? (bp[106+pair] >> 4) : (bp[106+pair] & 0xf)));\n"
"            unsigned char qh_byte = bp[66+ib32];\n"
"            int qs_off = ib32 * 8, sign_off = ib32 * 4;\n"
"            for (int l = half*2; l < half*2+2; l++) {\n"
"                int g1_idx = (int)(bp[2+qs_off + 2*l + 0]) | ((qh_byte << (8-2*l)) & 256);\n"
"                int g2_idx = (int)(bp[2+qs_off + 2*l + 1]) | ((qh_byte << (7-2*l)) & 256);\n"
"                unsigned int gv1 = iq3s_grid_dev[g1_idx];\n"
"                unsigned int gv2 = iq3s_grid_dev[g2_idx];\n"
"                unsigned char sgn = bp[74+sign_off + l];\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    int w1 = (int)(unsigned char)(gv1 >> (8*j));\n"
"                    int w2 = (int)(unsigned char)(gv2 >> (8*j));\n"
"                    sW[r][l*8 + j]     = (signed char)((sgn & (1 << j))     ? -w1 : w1);\n"
"                    sW[r][l*8 + j + 4] = (signed char)((sgn & (1 << (j+4))) ? -w2 : w2);\n"
"                }\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4],   qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* Prefill variant of mmq_iq3_s_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_iq3_s_grouped32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*110, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*110+(size_t)n*110)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*110);\n"
"            float d = half_to_float(*(const half_raw*)bp);\n"
"            int ib32 = sb & 7;\n"
"            int pair = ib32 / 2, sub_in_pair = ib32 & 1;\n"
"            if (half==0) sWs[r] = d * (float)(1 + 2 * (sub_in_pair ? (bp[106+pair] >> 4) : (bp[106+pair] & 0xf)));\n"
"            unsigned char qh_byte = bp[66+ib32];\n"
"            int qs_off = ib32 * 8, sign_off = ib32 * 4;\n"
"            for (int l = half*2; l < half*2+2; l++) {\n"
"                int g1_idx = (int)(bp[2+qs_off + 2*l + 0]) | ((qh_byte << (8-2*l)) & 256);\n"
"                int g2_idx = (int)(bp[2+qs_off + 2*l + 1]) | ((qh_byte << (7-2*l)) & 256);\n"
"                unsigned int gv1 = iq3s_grid_dev[g1_idx];\n"
"                unsigned int gv2 = iq3s_grid_dev[g2_idx];\n"
"                unsigned char sgn = bp[74+sign_off + l];\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    int w1 = (int)(unsigned char)(gv1 >> (8*j));\n"
"                    int w2 = (int)(unsigned char)(gv2 >> (8*j));\n"
"                    sW[r][l*8 + j]     = (signed char)((sgn & (1 << j))     ? -w1 : w1);\n"
"                    sW[r][l*8 + j + 4] = (signed char)((sgn & (1 << (j+4))) ? -w2 : w2);\n"
"                }\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4],   qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* Fused prefill variant of mmq_iq3_s_grouped with TG=32 (256 tokens/block).                 */\n"
"/* Loads F32 activation directly and quantizes to int8 on-the-fly, eliminating the separate    */\n"
"/* mmq_quant_q8_1 launch and the global-memory round-trip of cxq8/cxs.                         */\n"
"__global__ void mmq_iq3_s_fused32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const float *input, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*110, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32]; __shared__ float sWs[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*110+(size_t)n*110)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*110);\n"
"            float d = half_to_float(*(const half_raw*)bp);\n"
"            int ib32 = sb & 7;\n"
"            int pair = ib32 / 2, sub_in_pair = ib32 & 1;\n"
"            if (half==0) sWs[r] = d * (float)(1 + 2 * (sub_in_pair ? (bp[106+pair] >> 4) : (bp[106+pair] & 0xf)));\n"
"            unsigned char qh_byte = bp[66+ib32];\n"
"            int qs_off = ib32 * 8, sign_off = ib32 * 4;\n"
"            for (int l = half*2; l < half*2+2; l++) {\n"
"                int g1_idx = (int)(bp[2+qs_off + 2*l + 0]) | ((qh_byte << (8-2*l)) & 256);\n"
"                int g2_idx = (int)(bp[2+qs_off + 2*l + 1]) | ((qh_byte << (7-2*l)) & 256);\n"
"                unsigned int gv1 = iq3s_grid_dev[g1_idx];\n"
"                unsigned int gv2 = iq3s_grid_dev[g2_idx];\n"
"                unsigned char sgn = bp[74+sign_off + l];\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    int w1 = (int)(unsigned char)(gv1 >> (8*j));\n"
"                    int w2 = (int)(unsigned char)(gv2 >> (8*j));\n"
"                    sW[r][l*8 + j]     = (signed char)((sgn & (1 << j))     ? -w1 : w1);\n"
"                    sW[r][l*8 + j + 4] = (signed char)((sgn & (1 << (j+4))) ? -w2 : w2);\n"
"                }\n"
"            }\n"
"        }\n"
"        /* Fused quantize: load F32 from input, compute per-32-element scale, write int8 to sX */\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            float v0=(m<eb1)?input[(size_t)m*K+sb*32+j*4+0]:0.0f;\n"
"            float v1=(m<eb1)?input[(size_t)m*K+sb*32+j*4+1]:0.0f;\n"
"            float v2=(m<eb1)?input[(size_t)m*K+sb*32+j*4+2]:0.0f;\n"
"            float v3=(m<eb1)?input[(size_t)m*K+sb*32+j*4+3]:0.0f;\n"
"            float local_max=fmaxf(fabsf(v0),fmaxf(fabsf(v1),fmaxf(fabsf(v2),fabsf(v3))));\n"
"            unsigned mask=0xFF<<((threadIdx.x&31)&~7);\n"
"            float a=local_max;\n"
"            for(int o=4;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(mask,a,o));\n"
"            float scale=a/127.0f,inv=scale>0.0f?1.0f/scale:0.0f;\n"
"            int q0=(int)rintf(v0*inv);q0=q0<-127?-127:(q0>127?127:q0);\n"
"            int q1=(int)rintf(v1*inv);q1=q1<-127?-127:(q1>127?127:q1);\n"
"            int q2=(int)rintf(v2*inv);q2=q2<-127?-127:(q2>127?127:q2);\n"
"            int q3=(int)rintf(v3*inv);q3=q3<-127?-127:(q3>127?127:q3);\n"
"            int packed=(unsigned char)q0|((unsigned char)q1<<8)|((unsigned char)q2<<16)|((unsigned char)q3<<24);\n"
"            *(int*)&sX[tt][j*4]=packed;\n"
"            if(j==0)sXs[tt]=scale; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int qa0=*(const int*)&sW[wr+gid][tid*4],   qa1=*(const int*)&sW[wr+gid+8][tid*4];\n"
"        int qa2=*(const int*)&sW[wr+gid][tid*4+16], qa3=*(const int*)&sW[wr+gid+8][tid*4+16];\n"
"        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int qb0=*(const int*)&sX[g*8+gid][tid*4], qb1=*(const int*)&sX[g*8+gid][tid*4+16];\n"
"            int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(qa0),\"r\"(qa1),\"r\"(qa2),\"r\"(qa3),\"r\"(qb0),\"r\"(qb1),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;\n"
"            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* ---- 19i. mmq_iq2_s_grouped: IQ2_S weight x Q8_1 activation via int8 tensor-core MMQ ---- */\n"
"/* IQ2_S block: 82 bytes = d(f16)+qs[32]+signs[32]+qh[8]+scales[8], 256 elements.            */\n"
"/* Two dbs per sub-block (db0 k=0..15, db1 k=16..31); 2 x m16n8k16 MMA per sub-block.          */\n"
"__global__ void mmq_iq2_s_grouped(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 4;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*82, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[32][36]; __shared__ float sXs[32];\n"
"    __shared__ signed char sW[64][32];\n"
"    __shared__ float sWs_db0[64]; __shared__ float sWs_db1[64];\n"
"    float f[4][4]; for(int g=0;g<4;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*82+(size_t)n*82)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*82);\n"
"            float d = half_to_float(*(const half_raw*)bp);\n"
"            int ib32 = sb & 7;\n"
"            if (half==0){ sWs_db0[r]=d*(0.5f+(float)(bp[74+ib32]&0xf))*0.25f; sWs_db1[r]=d*(0.5f+(float)(bp[74+ib32]>>4))*0.25f; }\n"
"            unsigned char qh_byte = bp[66+ib32];\n"
"            int qs_off = ib32 * 4, sign_off = ib32 * 4;\n"
"            for (int l = half*2; l < half*2+2; l++) {\n"
"                int grid_idx = (int)(bp[2+qs_off + l]) | ((qh_byte << (8-2*l)) & 0x300);\n"
"                unsigned long long gv = iq2s_grid_dev[grid_idx];\n"
"                unsigned char sgn = bp[34+sign_off + l];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    int wv = (int)(unsigned char)(gv >> (8*j));\n"
"                    sW[r][l*8 + j] = (signed char)((sgn & (1 << j)) ? -wv : wv);\n"
"                }\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        /* MMA_0: k=0..15 */\n"
"        int a0_h0 = *(const int*)&sW[wr+gid][tid*4];\n"
"        int a1_h0 = *(const int*)&sW[wr+gid+8][tid*4];\n"
"        int b0_h0 = *(const int*)&sX[gid][tid*4];\n"
"        float wr0_db0 = sWs_db0[wr+gid], wr8_db0 = sWs_db0[wr+gid+8];\n"
"        /* MMA_1: k=16..31 */\n"
"        int a0_h1 = *(const int*)&sW[wr+gid][16+tid*4];\n"
"        int a1_h1 = *(const int*)&sW[wr+gid+8][16+tid*4];\n"
"        int b0_h1 = *(const int*)&sX[gid][16+tid*4];\n"
"        float wr0_db1 = sWs_db1[wr+gid], wr8_db1 = sWs_db1[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int b0_x = *(const int*)&sX[g*8+gid][tid*4];\n"
"            int b1_x = *(const int*)&sX[g*8+gid][16+tid*4];\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h0),\"r\"(a1_h0),\"r\"(b0_h0),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db0*xc0*(float)c0; f[g][1]+=wr0_db0*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db0*xc0*(float)c2; f[g][3]+=wr8_db0*xc1*(float)c3; }\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h1),\"r\"(a1_h1),\"r\"(b1_x),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db1*xc0*(float)c0; f[g][1]+=wr0_db1*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db1*xc0*(float)c2; f[g][3]+=wr8_db1*xc1*(float)c3; }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"/* Prefill variant of mmq_iq2_s_grouped with TG=16 (128 tokens/block). */\n"
"__global__ void mmq_iq2_s_grouped8(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 16;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*82, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[128][36]; __shared__ float sXs[128];\n"
"    __shared__ signed char sW[64][32];\n"
"    __shared__ float sWs_db0[64]; __shared__ float sWs_db1[64];\n"
"    float f[16][4]; for(int g=0;g<16;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*82+(size_t)n*82)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*82);\n"
"            float d = half_to_float(*(const half_raw*)bp);\n"
"            int ib32 = sb & 7;\n"
"            if (half==0){ sWs_db0[r]=d*(0.5f+(float)(bp[74+ib32]&0xf))*0.25f; sWs_db1[r]=d*(0.5f+(float)(bp[74+ib32]>>4))*0.25f; }\n"
"            unsigned char qh_byte = bp[66+ib32];\n"
"            int qs_off = ib32 * 4, sign_off = ib32 * 4;\n"
"            for (int l = half*2; l < half*2+2; l++) {\n"
"                int grid_idx = (int)(bp[2+qs_off + l]) | ((qh_byte << (8-2*l)) & 0x300);\n"
"                unsigned long long gv = iq2s_grid_dev[grid_idx];\n"
"                unsigned char sgn = bp[34+sign_off + l];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    int wv = (int)(unsigned char)(gv >> (8*j));\n"
"                    sW[r][l*8 + j] = (signed char)((sgn & (1 << j)) ? -wv : wv);\n"
"                }\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        /* MMA_0: k=0..15 */\n"
"        int a0_h0 = *(const int*)&sW[wr+gid][tid*4];\n"
"        int a1_h0 = *(const int*)&sW[wr+gid+8][tid*4];\n"
"        int b0_h0 = *(const int*)&sX[gid][tid*4];\n"
"        float wr0_db0 = sWs_db0[wr+gid], wr8_db0 = sWs_db0[wr+gid+8];\n"
"        /* MMA_1: k=16..31 */\n"
"        int a0_h1 = *(const int*)&sW[wr+gid][16+tid*4];\n"
"        int a1_h1 = *(const int*)&sW[wr+gid+8][16+tid*4];\n"
"        int b0_h1 = *(const int*)&sX[gid][16+tid*4];\n"
"        float wr0_db1 = sWs_db1[wr+gid], wr8_db1 = sWs_db1[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int b0_x = *(const int*)&sX[g*8+gid][tid*4];\n"
"            int b1_x = *(const int*)&sX[g*8+gid][16+tid*4];\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h0),\"r\"(a1_h0),\"r\"(b0_h0),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db0*xc0*(float)c0; f[g][1]+=wr0_db0*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db0*xc0*(float)c2; f[g][3]+=wr8_db0*xc1*(float)c3; }\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h1),\"r\"(a1_h1),\"r\"(b1_x),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db1*xc0*(float)c0; f[g][1]+=wr0_db1*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db1*xc0*(float)c2; f[g][3]+=wr8_db1*xc1*(float)c3; }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* Prefill variant of mmq_iq2_s_grouped with TG=32 (256 tokens/block). */\n"
"__global__ void mmq_iq2_s_grouped32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const signed char *cxq8, const float *cxs, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*82, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32];\n"
"    __shared__ float sWs_db0[64]; __shared__ float sWs_db1[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*82+(size_t)n*82)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*82);\n"
"            float d = half_to_float(*(const half_raw*)bp);\n"
"            int ib32 = sb & 7;\n"
"            if (half==0){ sWs_db0[r]=d*(0.5f+(float)(bp[74+ib32]&0xf))*0.25f; sWs_db1[r]=d*(0.5f+(float)(bp[74+ib32]>>4))*0.25f; }\n"
"            unsigned char qh_byte = bp[66+ib32];\n"
"            int qs_off = ib32 * 4, sign_off = ib32 * 4;\n"
"            for (int l = half*2; l < half*2+2; l++) {\n"
"                int grid_idx = (int)(bp[2+qs_off + l]) | ((qh_byte << (8-2*l)) & 0x300);\n"
"                unsigned long long gv = iq2s_grid_dev[grid_idx];\n"
"                unsigned char sgn = bp[34+sign_off + l];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    int wv = (int)(unsigned char)(gv >> (8*j));\n"
"                    sW[r][l*8 + j] = (signed char)((sgn & (1 << j)) ? -wv : wv);\n"
"                }\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            *(int*)&sX[tt][j*4]=(m<eb1)?*(const int*)&cxq8[(size_t)m*K+sb*32+j*4]:0; }\n"
"        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        /* MMA_0: k=0..15 */\n"
"        int a0_h0 = *(const int*)&sW[wr+gid][tid*4];\n"
"        int a1_h0 = *(const int*)&sW[wr+gid+8][tid*4];\n"
"        int b0_h0 = *(const int*)&sX[gid][tid*4];\n"
"        float wr0_db0 = sWs_db0[wr+gid], wr8_db0 = sWs_db0[wr+gid+8];\n"
"        /* MMA_1: k=16..31 */\n"
"        int a0_h1 = *(const int*)&sW[wr+gid][16+tid*4];\n"
"        int a1_h1 = *(const int*)&sW[wr+gid+8][16+tid*4];\n"
"        int b0_h1 = *(const int*)&sX[gid][16+tid*4];\n"
"        float wr0_db1 = sWs_db1[wr+gid], wr8_db1 = sWs_db1[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int b0_x = *(const int*)&sX[g*8+gid][tid*4];\n"
"            int b1_x = *(const int*)&sX[g*8+gid][16+tid*4];\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h0),\"r\"(a1_h0),\"r\"(b0_h0),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db0*xc0*(float)c0; f[g][1]+=wr0_db0*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db0*xc0*(float)c2; f[g][3]+=wr8_db0*xc1*(float)c3; }\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h1),\"r\"(a1_h1),\"r\"(b1_x),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db1*xc0*(float)c0; f[g][1]+=wr0_db1*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db1*xc0*(float)c2; f[g][3]+=wr8_db1*xc1*(float)c3; }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* Fused prefill variant of mmq_iq2_s_grouped with TG=32 (256 tokens/block).                    */\n"
"/* Loads F32 activation directly and quantizes to int8 on-the-fly.                              */\n"
"__global__ void mmq_iq2_s_fused32(float *out, const unsigned char *W, unsigned long long estride,\n"
"                                    const float *input, const int *ebounds,\n"
"                                    const int *worklist, int bm, int N, int K) {\n"
"    const int TG = 32;\n"
"    int packed = worklist[blockIdx.y]; int e = packed >> 16, g0 = packed & 0xffff;\n"
"    int eb0 = ebounds[e], eb1 = ebounds[e+1];\n"
"    int m_base = eb0 + g0*(8*TG);\n"
"    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;\n"
"    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n"
"    int gid = lane>>2, tid = lane&3;\n"
"    int n0 = blockIdx.x*64 + warp*16;\n"
"    int nb = K/256, row_bytes = nb*82, nsb = K/32;\n"
"    const unsigned char *We = W + (size_t)e*estride;\n"
"    __shared__ signed char sX[256][36]; __shared__ float sXs[256];\n"
"    __shared__ signed char sW[64][32];\n"
"    __shared__ float sWs_db0[64]; __shared__ float sWs_db1[64];\n"
"    float f[32][4]; for(int g=0;g<32;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}\n"
"    __syncthreads();\n"
"    for (int sb=0; sb<nsb; sb++) {\n"
"        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;\n"
"            const unsigned char *bp = bm ? (We+(size_t)(sb/8)*N*82+(size_t)n*82)\n"
"                              : (We + (size_t)n*row_bytes + (size_t)(sb/8)*82);\n"
"            float d = half_to_float(*(const half_raw*)bp);\n"
"            int ib32 = sb & 7;\n"
"            if (half==0){ sWs_db0[r]=d*(0.5f+(float)(bp[74+ib32]&0xf))*0.25f; sWs_db1[r]=d*(0.5f+(float)(bp[74+ib32]>>4))*0.25f; }\n"
"            unsigned char qh_byte = bp[66+ib32];\n"
"            int qs_off = ib32 * 4, sign_off = ib32 * 4;\n"
"            for (int l = half*2; l < half*2+2; l++) {\n"
"                int grid_idx = (int)(bp[2+qs_off + l]) | ((qh_byte << (8-2*l)) & 0x300);\n"
"                unsigned long long gv = iq2s_grid_dev[grid_idx];\n"
"                unsigned char sgn = bp[34+sign_off + l];\n"
"                for (int j = 0; j < 8; j++) {\n"
"                    int wv = (int)(unsigned char)(gv >> (8*j));\n"
"                    sW[r][l*8 + j] = (signed char)((sgn & (1 << j)) ? -wv : wv);\n"
"                }\n"
"            }\n"
"        }\n"
"        for (int i=threadIdx.x;i<ntg*8*8;i+=blockDim.x){ int tt=i>>3,j=i&7,m=m_base+tt;\n"
"            float v0=(m<eb1)?input[(size_t)m*K+sb*32+j*4+0]:0.0f;\n"
"            float v1=(m<eb1)?input[(size_t)m*K+sb*32+j*4+1]:0.0f;\n"
"            float v2=(m<eb1)?input[(size_t)m*K+sb*32+j*4+2]:0.0f;\n"
"            float v3=(m<eb1)?input[(size_t)m*K+sb*32+j*4+3]:0.0f;\n"
"            float local_max=fmaxf(fabsf(v0),fmaxf(fabsf(v1),fmaxf(fabsf(v2),fabsf(v3))));\n"
"            unsigned mask=0xFF<<((threadIdx.x&31)&~7);\n"
"            float a=local_max;\n"
"            for(int o=4;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(mask,a,o));\n"
"            float scale=a/127.0f,inv=scale>0.0f?1.0f/scale:0.0f;\n"
"            int q0=(int)rintf(v0*inv);q0=q0<-127?-127:(q0>127?127:q0);\n"
"            int q1=(int)rintf(v1*inv);q1=q1<-127?-127:(q1>127?127:q1);\n"
"            int q2=(int)rintf(v2*inv);q2=q2<-127?-127:(q2>127?127:q2);\n"
"            int q3=(int)rintf(v3*inv);q3=q3<-127?-127:(q3>127?127:q3);\n"
"            int packed=(unsigned char)q0|((unsigned char)q1<<8)|((unsigned char)q2<<16)|((unsigned char)q3<<24);\n"
"            *(int*)&sX[tt][j*4]=packed;\n"
"            if(j==0)sXs[tt]=scale; }\n"
"        __syncthreads();\n"
"        int wr=warp*16;\n"
"        int a0_h0 = *(const int*)&sW[wr+gid][tid*4];\n"
"        int a1_h0 = *(const int*)&sW[wr+gid+8][tid*4];\n"
"        int b0_h0 = *(const int*)&sX[gid][tid*4];\n"
"        float wr0_db0 = sWs_db0[wr+gid], wr8_db0 = sWs_db0[wr+gid+8];\n"
"        int a0_h1 = *(const int*)&sW[wr+gid][16+tid*4];\n"
"        int a1_h1 = *(const int*)&sW[wr+gid+8][16+tid*4];\n"
"        int b0_h1 = *(const int*)&sX[gid][16+tid*4];\n"
"        float wr0_db1 = sWs_db1[wr+gid], wr8_db1 = sWs_db1[wr+gid+8];\n"
"        for (int g=0; g<ntg; g++) {\n"
"            int b0_x = *(const int*)&sX[g*8+gid][tid*4];\n"
"            int b1_x = *(const int*)&sX[g*8+gid][16+tid*4];\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h0),\"r\"(a1_h0),\"r\"(b0_h0),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db0*xc0*(float)c0; f[g][1]+=wr0_db0*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db0*xc0*(float)c2; f[g][3]+=wr8_db0*xc1*(float)c3; }\n"
"            { int c0=0,c1=0,c2=0,c3=0;\n"
"            asm(\"mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\"\n"
"                :\"=r\"(c0),\"=r\"(c1),\"=r\"(c2),\"=r\"(c3)\n"
"                :\"r\"(a0_h1),\"r\"(a1_h1),\"r\"(b1_x),\"r\"(0),\"r\"(0),\"r\"(0),\"r\"(0));\n"
"            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];\n"
"            f[g][0]+=wr0_db1*xc0*(float)c0; f[g][1]+=wr0_db1*xc1*(float)c1;\n"
"            f[g][2]+=wr8_db1*xc0*(float)c2; f[g][3]+=wr8_db1*xc1*(float)c3; }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int n_a=n0+gid, n_b=n0+gid+8;\n"
"    for (int g=0; g<ntg; g++) {\n"
"        int m_a=m_base+g*8+tid*2, m_b=m_a+1;\n"
"        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }\n"
"        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }\n"
"    }\n"
"}\n"
"\n"
"/* Batched IQ2_S TensorCore matvec: 16 rows x 8 tokens via mma.sync */\n"
"__global__ void batch_matvec_iq2_s_tc(float *output, const unsigned char *mat, const float *input,\n"
"                                        int out_dim, int in_dim, int n_tokens) {\n"
"    int lane = threadIdx.x;\n"
"    int m_start = blockIdx.x * 16;\n"
"    int t_start = blockIdx.y * 8;\n"
"    if (m_start >= out_dim || t_start >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 82;\n"
"    float rc[4] = {0, 0, 0, 0};\n"
"    for (int k = 0; k < in_dim; k += 16) {\n"
"        int a_row_off = lane / 2;\n"
"        int a_col_off = (lane % 2) * 8;\n"
"        int row = m_start + a_row_off;\n"
"        int col_k = k + a_col_off;\n"
"        int bi = col_k / 256;\n"
"        int ib32 = (col_k % 256) / 32;\n"
"        int tl = ((k + a_col_off) % 32) / 8;\n"
"        const unsigned char *rp = mat + (size_t)row * row_bytes;\n"
"        const unsigned char *bp = rp + bi * 82;\n"
"        float d = half_to_float(*(const half_raw *)bp);\n"
"        float db0 = d * (0.5f + (float)(bp[74+ib32] & 0xf)) * 0.25f;\n"
"        float db1 = d * (0.5f + (float)(bp[74+ib32] >>  4)) * 0.25f;\n"
"        float dl = (tl < 2) ? db0 : db1;\n"
"        int gidx = bp[2+ib32*4+tl] | ((bp[66+ib32] << (8-2*tl)) & 0x300);\n"
"        unsigned long long gv = iq2s_grid_dev[gidx];\n"
"        unsigned char sgn = bp[34+ib32*4+tl];\n"
"        half_raw w16[8];\n"
"        for (int j = 0; j < 8; j++) {\n"
"            float w = dl * (float)(unsigned char)(gv >> (8*j));\n"
"            if ((sgn >> j) & 1) w = -w;\n"
"            w16[j] = float_to_half(w);\n"
"        }\n"
"        unsigned int ra[4];\n"
"        ra[0] = *(const unsigned int *)&w16[0]; ra[1] = *(const unsigned int *)&w16[2];\n"
"        ra[2] = *(const unsigned int *)&w16[4]; ra[3] = *(const unsigned int *)&w16[6];\n"
"        half_raw b16[4];\n"
"        int b_tok_off = (lane / 8) * 2;\n"
"        int b_k_off = ((lane % 8) / 2) * 2;\n"
"        int t0 = t_start + b_tok_off, t1 = t_start + b_tok_off + 1;\n"
"        int kk = k + b_k_off;\n"
"        b16[0] = (t0 < n_tokens) ? float_to_half(input[(size_t)t0 * in_dim + kk])     : 0;\n"
"        b16[1] = (t0 < n_tokens) ? float_to_half(input[(size_t)t0 * in_dim + kk + 1]) : 0;\n"
"        b16[2] = (t1 < n_tokens) ? float_to_half(input[(size_t)t1 * in_dim + kk])     : 0;\n"
"        b16[3] = (t1 < n_tokens) ? float_to_half(input[(size_t)t1 * in_dim + kk + 1]) : 0;\n"
"        unsigned int rb[2];\n"
"        rb[0] = *(const unsigned int *)&b16[0];\n"
"        rb[1] = *(const unsigned int *)&b16[2];\n"
"        asm volatile(\n"
"            \"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\"\n"
"            \" {%0, %1, %2, %3},\"\n"
"            \" {%4, %5, %6, %7},\"\n"
"            \" {%8, %9},\"\n"
"            \" {%10, %11, %12, %13};\"\n"
"            : \"=f\"(rc[0]), \"=f\"(rc[1]), \"=f\"(rc[2]), \"=f\"(rc[3])\n"
"            : \"r\"(ra[0]), \"r\"(ra[1]), \"r\"(ra[2]), \"r\"(ra[3]),\n"
"              \"r\"(rb[0]), \"r\"(rb[1]),\n"
"              \"f\"(rc[0]), \"f\"(rc[1]), \"f\"(rc[2]), \"f\"(rc[3]));\n"
"    }\n"
"        int m0 = lane / 4, m1 = m0 + 8;\n"
"        int n0 = (lane % 4) * 2, n1 = n0 + 1;\n"
"        int token0 = t_start + n0, token1 = t_start + n1;\n"
"        int row0 = m_start + m0;\n"
"        /* D[l/4] holds even-K cols, D[l/4+8] holds odd-K cols for same weight row */\n"
"        /* Combine both for the full dot product */\n"
"        if (token0 < n_tokens) {\n"
"            output[(size_t)token0 * out_dim + row0] += rc[0] + rc[2];\n"
"        }\n"
"        if (token1 < n_tokens) {\n"
"            output[(size_t)token1 * out_dim + row0] += rc[1] + rc[3];\n"
"        }\n"
"}\n"
"\n"
"/* Batched IQ4_NL matvec */\n"
"__global__ void batch_matvec_iq4_nl(float *output, const unsigned char *mat, const float *input,\n"
"                                      int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 32;\n"
"    int row_bytes = nb * 18;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
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
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Batched IQ4_XS matvec */\n"
"__global__ void batch_matvec_iq4_xs(float *output, const unsigned char *mat, const float *input,\n"
"                                      int out_dim, int in_dim, int n_tokens) {\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int row = blockIdx.x * 8 + warp_id;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int nb = in_dim / 256;\n"
"    int row_bytes = nb * 136;\n"
"    const unsigned char *row_ptr = mat + (size_t)row * row_bytes;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
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
"    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    if (lane == 0) output[(size_t)token * out_dim + row] = sum;\n"
"}\n"
"\n"
"/* Fused MoE FFN: processes one token's routed experts in a single block */\n"
"/* grid = n_tokens, block = 256 */\n"
"__global__ void moe_fused_ffn(float *output, const float *input,\n"
"    const unsigned char *gate_exps, const unsigned char *up_exps, const unsigned char *down_exps,\n"
"    const int *topk_idx, const float *topk_wgt, int n_tokens, int n_used,\n"
"    int n_embd, int expert_ff, size_t stride_gu, size_t stride_d,\n"
"    int gate_type, int up_type, int down_type) {\n"
"    int token = blockIdx.x;\n"
"    if (token >= n_tokens) return;\n"
"    __shared__ float s_accum[2048+512+512]; /* n_embd + expert_ff + expert_ff (up_buf) */\n"
"    __shared__ float s_xb[2048];\n"
"    int tid = threadIdx.x;\n"
"    int lane = tid % 32, warp = tid / 32;\n"
"    const float *xb = input + (size_t)token * n_embd;\n"
"    for (int i = tid; i < n_embd; i += 256) s_xb[i] = xb[i];\n"
"    __syncthreads();\n"
"    float *accum = s_accum;\n"
"    float *gate_buf = s_accum + n_embd;\n"
"    float *up_buf = gate_buf + expert_ff;\n"
"    for (int i = tid; i < n_embd; i += 256) accum[i] = 0.0f;\n"
"    __syncthreads();\n"
"    for (int ei = 0; ei < n_used; ei++) {\n"
"        int eidx = topk_idx[(size_t)token * n_used + ei];\n"
"        float weight = topk_wgt[(size_t)token * n_used + ei];\n"
"        if (eidx < 0) continue;\n"
"        for (int r = warp; r < expert_ff; r += 8) {\n"
"            int nb = n_embd / 256, row_bytes = nb * 66;\n"
"            const unsigned char *rp = gate_exps + (size_t)eidx * stride_gu + (size_t)r * row_bytes;\n"
"            float sum = 0.0f;\n"
"            for (int b = lane; b < nb; b += 32) {\n"
"                const unsigned char *bp = rp + b * 66;\n"
"                float d = half_to_float(*(const half_raw *)bp);\n"
"                const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"                const float *xbb = s_xb + b * 256;\n"
"                float p = 0.0f; int yi = 0;\n"
"                for (int ib = 0; ib < 8; ib++) {\n"
"                    unsigned int a0 = qs[4*ib] | ((unsigned int)qs[4*ib+1]<<16);\n"
"                    unsigned int a1 = qs[4*ib+2] | ((unsigned int)qs[4*ib+3]<<16);\n"
"                    float db = d * (0.5f + (float)(a1>>28)) * 0.25f;\n"
"                    const unsigned char *a8 = (const unsigned char *)&a0;\n"
"                    for (int l = 0; l < 4; l++) {\n"
"                        unsigned long long gv = iq2xxs_grid_dev[a8[l]];\n"
"                        unsigned char sn = ksigns_iq2xs_dev[(a1>>(7*l))&127];\n"
"                        for (int j = 0; j < 8; j++) {\n"
"                            float w = db * (float)(unsigned char)(gv>>(8*j)) * ((sn&(1<<j))?-1.0f:1.0f);\n"
"                            p += w * xbb[yi++];\n"
"                        }\n"
"                    }\n"
"                }\n"
"                sum += p;\n"
"            }\n"
"            for (int o=16; o>0; o>>=1) sum += __shfl_down_sync(0xFFFFFFFF, sum, o);\n"
"            if (lane==0) gate_buf[r] = sum;\n"
"        }\n"
"        __syncthreads();\n"
"        for (int r = warp; r < expert_ff; r += 8) {\n"
"            int nb = n_embd / 256, row_bytes = nb * 66;\n"
"            const unsigned char *rp = up_exps + (size_t)eidx * stride_gu + (size_t)r * row_bytes;\n"
"            float sum = 0.0f;\n"
"            for (int b = lane; b < nb; b += 32) {\n"
"                const unsigned char *bp = rp + b * 66;\n"
"                float d = half_to_float(*(const half_raw *)bp);\n"
"                const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"                const float *xbb = s_xb + b * 256;\n"
"                float p = 0.0f; int yi = 0;\n"
"                for (int ib = 0; ib < 8; ib++) {\n"
"                    unsigned int a0 = qs[4*ib] | ((unsigned int)qs[4*ib+1]<<16);\n"
"                    unsigned int a1 = qs[4*ib+2] | ((unsigned int)qs[4*ib+3]<<16);\n"
"                    float db = d * (0.5f + (float)(a1>>28)) * 0.25f;\n"
"                    const unsigned char *a8 = (const unsigned char *)&a0;\n"
"                    for (int l = 0; l < 4; l++) {\n"
"                        unsigned long long gv = iq2xxs_grid_dev[a8[l]];\n"
"                        unsigned char sn = ksigns_iq2xs_dev[(a1>>(7*l))&127];\n"
"                        for (int j = 0; j < 8; j++) {\n"
"                            float w = db * (float)(unsigned char)(gv>>(8*j)) * ((sn&(1<<j))?-1.0f:1.0f);\n"
"                            p += w * xbb[yi++];\n"
"                        }\n"
"                    }\n"
"                }\n"
"                sum += p;\n"
"            }\n"
"            for (int o=16; o>0; o>>=1) sum += __shfl_down_sync(0xFFFFFFFF, sum, o);\n"
"            if (lane==0) up_buf[r] = sum;\n"
"        }\n"
"        __syncthreads();\n"
"        for (int i = tid; i < expert_ff; i += 256) {\n"
"            float g = gate_buf[i];\n"
"            gate_buf[i] = (g/(1.0f+expf(-g))) * up_buf[i];\n"
"        }\n"
"        __syncthreads();\n"
"        for (int r = warp; r < n_embd; r += 8) {\n"
"            int nb = expert_ff / 256, row_bytes = nb * 66;\n"
"            const unsigned char *rp = down_exps + (size_t)eidx * stride_d + (size_t)r * row_bytes;\n"
"            float sum = 0.0f;\n"
"            for (int b = lane; b < nb; b += 32) {\n"
"                const unsigned char *bp = rp + b * 66;\n"
"                float d = half_to_float(*(const half_raw *)bp);\n"
"                const unsigned short *qs = (const unsigned short *)(bp + 2);\n"
"                const float *xbb = gate_buf + b * 256;\n"
"                float p = 0.0f; int yi = 0;\n"
"                for (int ib = 0; ib < 8; ib++) {\n"
"                    unsigned int a0 = qs[4*ib] | ((unsigned int)qs[4*ib+1]<<16);\n"
"                    unsigned int a1 = qs[4*ib+2] | ((unsigned int)qs[4*ib+3]<<16);\n"
"                    float db = d * (0.5f + (float)(a1>>28)) * 0.25f;\n"
"                    const unsigned char *a8 = (const unsigned char *)&a0;\n"
"                    for (int l = 0; l < 4; l++) {\n"
"                        unsigned long long gv = iq2xxs_grid_dev[a8[l]];\n"
"                        unsigned char sn = ksigns_iq2xs_dev[(a1>>(7*l))&127];\n"
"                        for (int j = 0; j < 8; j++) {\n"
"                            float w = db * (float)(unsigned char)(gv>>(8*j)) * ((sn&(1<<j))?-1.0f:1.0f);\n"
"                            p += w * xbb[yi++];\n"
"                        }\n"
"                    }\n"
"                }\n"
"                sum += p;\n"
"            }\n"
"            for (int o=16; o>0; o>>=1) sum += __shfl_down_sync(0xFFFFFFFF, sum, o);\n"
"            if (lane==0) accum[r] += weight * sum;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    float *out = output + (size_t)token * n_embd;\n"
"    for (int i = tid; i < n_embd; i += 256) out[i] += accum[i];\n"
"}\n"
"\n"
"__global__ void moe_decode_fused_tc(float *output, const float *xb,\n""    const float *router_logits,\n""    const unsigned char *gate_exps, const unsigned char *up_exps, const unsigned char *down_exps,\n""    const unsigned short *shared_gate, const unsigned short *shared_up, const unsigned short *shared_down,\n""    const float *shared_gate_scalar,\n""    int n_experts, int n_used, int n_embd, int expert_ff,\n""    size_t stride_gu, size_t stride_d,\n""    int gate_type, int up_type, int down_type) {\n""    __shared__ float sx[2048];\n""    __shared__ float sg[512];\n""    __shared__ float su[512];\n""    __shared__ float sa[2048];\n""    __shared__ int sidx[8];\n""    __shared__ float sw[8];\n""    int tid = threadIdx.x;\n""    int warp = tid / 32;\n""    int lane = tid % 32;\n""\n""    for (int i = tid; i < n_embd; i += 128) sx[i] = xb[i];\n""    __syncthreads();\n""\n""    float lv[8]; int li[8];\n""    for (int i = 0; i < n_used; i++) { lv[i] = -1e38f; li[i] = -1; }\n""    for (int e = tid; e < n_experts; e += 128) {\n""        float v = router_logits[e];\n""        if (v > lv[n_used-1]) {\n""            lv[n_used-1] = v; li[n_used-1] = e;\n""            for (int i = n_used-2; i >= 0 && lv[i+1] > lv[i]; i--) {\n""                float xv = lv[i]; int xi = li[i];\n""                lv[i] = lv[i+1]; li[i] = li[i+1];\n""                lv[i+1] = xv; li[i+1] = xi;\n""            }\n""        }\n""    }\n""    for (int rr = 16; rr > 0; rr >>= 1) {\n""        if (tid + rr < 128) {\n""            for (int kk = 0; kk < n_used; kk++) {\n""                if (lv[kk] > lv[n_used-1]) {\n""                    lv[n_used-1] = lv[kk]; li[n_used-1] = li[kk];\n""                    for (int ii = n_used-2; ii >= 0 && lv[ii+1] > lv[ii]; ii--) {\n""                        float xv = lv[ii]; int xi = li[ii];\n""                        lv[ii] = lv[ii+1]; li[ii] = li[ii+1];\n""                        lv[ii+1] = xv; li[ii+1] = xi;\n""                    }\n""                }\n""            }\n""        }\n""        __syncthreads();\n""    }\n""    if (tid < n_used) { sidx[tid] = li[tid]; sw[tid] = 1.0f/(1.0f+expf(-lv[tid])); }\n""    __syncthreads();\n""\n""    for (int i = tid; i < n_embd; i += 128) sa[i] = 0.0f;\n""    __syncthreads();\n""\n""    for (int ei = 0; ei < n_used; ei++) {\n""        int eidx = sidx[ei];\n""        float wgt = sw[ei];\n""        for (int i = tid; i < expert_ff; i += 128) { sg[i] = 0.0f; su[i] = 0.0f; }\n""        __syncthreads();\n""\n""        int nb = n_embd / 256, rb = nb * 82;\n""        for (int r = warp; r < expert_ff; r += 4) {\n""            const unsigned char *rp = gate_exps + (size_t)eidx * stride_gu + (size_t)r * rb;\n""            float s = 0;\n""            for (int b = lane; b < nb; b += 32) {\n""                const unsigned char *bp = rp + b * 82;\n""                float d = half_to_float(*(const half_raw *)bp);\n""                const unsigned char *qs = bp + 2, *qh = bp + 66, *sc = bp + 74, *signp = bp + 34;\n""                const float *xx = sx + b * 256;\n""                float p = 0; int yi = 0;\n""                for (int ib = 0; ib < 8; ib++) {\n""                    float da = d * (0.5f + (sc[ib] & 0xf)) * 0.25f;\n""                    float db = d * (0.5f + (sc[ib] >> 4)) * 0.25f;\n""                    for (int lx = 0; lx < 4; lx++) {\n""                        float dl = (lx < 2) ? da : db;\n""                        int gid = qs[lx] | ((qh[ib] << (8-2*lx)) & 0x300);\n""                        const unsigned char *grd = (const unsigned char *)&iq2s_grid_dev[gid];\n""                        unsigned char sn = signp[lx];\n""                        for (int j = 0; j < 8; j++)\n""                            p += (dl * (float)grd[j] * ((sn & (1<<j))?-1.0f:1.0f)) * xx[yi++];\n""                    }\n""                    qs += 4; signp += 4;\n""                }\n""                s += p;\n""            }\n""            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);\n""            if (lane == 0) sg[r] = s;\n""        }\n""        __syncthreads();\n""\n""        for (int r = warp; r < expert_ff; r += 4) {\n""            const unsigned char *rp = up_exps + (size_t)eidx * stride_gu + (size_t)r * rb;\n""            float s = 0;\n""            for (int b = lane; b < nb; b += 32) {\n""                const unsigned char *bp = rp + b * 82;\n""                float d = half_to_float(*(const half_raw *)bp);\n""                const unsigned char *qs = bp + 2, *qh = bp + 66, *sc = bp + 74, *signp = bp + 34;\n""                const float *xx = sx + b * 256;\n""                float p = 0; int yi = 0;\n""                for (int ib = 0; ib < 8; ib++) {\n""                    float da = d * (0.5f + (sc[ib] & 0xf)) * 0.25f;\n""                    float db = d * (0.5f + (sc[ib] >> 4)) * 0.25f;\n""                    for (int lx = 0; lx < 4; lx++) {\n""                        float dl = (lx < 2) ? da : db;\n""                        int gid = qs[lx] | ((qh[ib] << (8-2*lx)) & 0x300);\n""                        const unsigned char *grd = (const unsigned char *)&iq2s_grid_dev[gid];\n""                        unsigned char sn = signp[lx];\n""                        for (int j = 0; j < 8; j++)\n""                            p += (dl * (float)grd[j] * ((sn & (1<<j))?-1.0f:1.0f)) * xx[yi++];\n""                    }\n""                    qs += 4; signp += 4;\n""                }\n""                s += p;\n""            }\n""            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);\n""            if (lane == 0) su[r] = s;\n""        }\n""        __syncthreads();\n""\n""        for (int i = tid; i < expert_ff; i += 128)\n""            sg[i] = (sg[i] / (1.0f + expf(-sg[i]))) * su[i];\n""        __syncthreads();\n""\n""        nb = expert_ff / 256; rb = nb * 110;\n""        for (int r = warp; r < n_embd; r += 4) {\n""            const unsigned char *rp = down_exps + (size_t)eidx * stride_d + (size_t)r * rb;\n""            float s = 0;\n""            for (int b = lane; b < nb; b += 32) {\n""                const unsigned char *bp = rp + b * 110;\n""                float d = half_to_float(*(const half_raw *)bp);\n""                const unsigned char *qs = bp + 2, *ss = qs + 64;\n""                const float *gg = sg + b * 256;\n""                float p = 0; int yi = 0;\n""                for (int ib = 0; ib < 8; ib++) {\n""                    unsigned int aux32;\n""                    memcpy(&aux32, ss + 4*ib, 4);\n""                    float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;\n""                    for (int lx = 0; lx < 4; lx++) {\n""                        unsigned char sb = (unsigned char)ksigns_iq2xs_dev[(aux32>>(7*lx))&127];\n""                        unsigned int g1 = iq3xxs_grid_dev[qs[2*lx+0]];\n""                        unsigned int g2 = iq3xxs_grid_dev[qs[2*lx+1]];\n""                        for (int j = 0; j < 4; j++) {\n""                            float w0 = db*(float)(unsigned char)(g1>>(8*j))*((sb&(1<<j))?-1.0f:1.0f);\n""                            float w1 = db*(float)(unsigned char)(g2>>(8*j))*((sb&(1<<(j+4)))?-1.0f:1.0f);\n""                            p += w0*gg[yi+j] + w1*gg[yi+j+4];\n""                        }\n""                        yi += 8;\n""                    }\n""                    qs += 8;\n""                }\n""                s += p;\n""            }\n""            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);\n""            if (lane == 0) sa[r] += wgt * s;\n""        }\n""        __syncthreads();\n""    }\n""\n""    {\n""        float gs = 0;\n""        for (int i = lane; i < n_embd; i += 32) gs += shared_gate_scalar[i] * sx[i];\n""        for (int o = 16; o > 0; o >>= 1) gs += __shfl_down_sync(0xFFFFFFFF, gs, o);\n""        float sscale = 1.0f/(1.0f+expf(-gs));\n""        for (int r = warp; r < expert_ff; r += 4) {\n""            float s = 0;\n""            for (int i = lane; i < n_embd; i += 32)\n""                s += half_to_float(shared_gate[(size_t)r*n_embd+i]) * sx[i];\n""            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);\n""            if (lane == 0) sg[r] = s;\n""        }\n""        __syncthreads();\n""        for (int r = warp; r < expert_ff; r += 4) {\n""            float s = 0;\n""            for (int i = lane; i < n_embd; i += 32)\n""                s += half_to_float(shared_up[(size_t)r*n_embd+i]) * sx[i];\n""            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);\n""            if (lane == 0) su[r] = s;\n""        }\n""        __syncthreads();\n""        for (int i = tid; i < expert_ff; i += 128)\n""            sg[i] = (sg[i]/(1.0f+expf(-sg[i])))*su[i];\n""        __syncthreads();\n""        for (int r = warp; r < n_embd; r += 4) {\n""            float s = 0;\n""            for (int i = lane; i < expert_ff; i += 32)\n""                s += half_to_float(shared_down[(size_t)r*expert_ff+i]) * sg[i];\n""            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);\n""            if (lane == 0) sa[r] += sscale * s;\n""        }\n""        __syncthreads();\n""    }\n""\n""    for (int i = tid; i < n_embd; i += 128) output[i] = sa[i];\n""}\n"
"__global__ void gather_rows_f32(float *dst, const float *src, const int *indices,\n"
"                                 int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int idx = indices[row];\n"
"    const float *s = src + (size_t)idx * n_cols;\n"
"    float *d = dst + (size_t)row * n_cols;\n"
"    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)\n"
"        d[col] = s[col];\n"
"}\n"
"\n"
"/* Scatter-add weighted results: dst[indices[row], cols] += weight[row] * src[row, cols] */\n"
"__global__ void scatter_add_weighted_f32(float *dst, const float *src, const int *indices,\n"
"                                          const float *weights, int n_rows, int n_cols) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    int idx = indices[row];\n"
"    float w = weights[row];\n"
"    float *d = dst + (size_t)idx * n_cols;\n"
"    const float *s = src + (size_t)row * n_cols;\n"
"    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)\n"
"        d[col] += w * s[col];\n"
"}\n"
"\n"
"/* MoE top-k selection: for each token, find top-k expert indices and softmax weights */\n"
"/* Input: logits[n_tokens, n_experts] F32 */\n"
"/* Output: indices[n_tokens, k] int32, weights[n_tokens, k] float (softmax over selected experts) */\n"
"__global__ void moe_topk_kernel(const float *logits, int *indices, float *weights,\n"
"                                  int n_tokens, int n_experts, int k) {\n"
"    int token = blockIdx.x;\n"
"    if (token >= n_tokens) return;\n"
"    const float *row = logits + (size_t)token * n_experts;\n"
"    int tid = threadIdx.x;\n"
"    /* Each thread scans a subset of experts, tracking top-k */\n"
"    int lbi[8]; float lbv[8];\n"
"    for (int i = 0; i < k; i++) { lbi[i] = -1; lbv[i] = -1e38f; }\n"
"    for (int e = tid; e < n_experts; e += blockDim.x) {\n"
"        float v = row[e];\n"
"        if (!(v == v)) v = -3.402823466e+38F;\n"
"        if (v > lbv[k-1] || (v == lbv[k-1] && (lbi[k-1] < 0 || e < lbi[k-1]))) {\n"
"            lbv[k-1] = v;\n"
"            lbi[k-1] = e;\n"
"            /* Insertion sort: bubble down */\n"
"            for (int i = k-2; i >= 0 &&\n"
"                 (lbv[i+1] > lbv[i] || (lbv[i+1] == lbv[i] && lbi[i+1] >= 0 && (lbi[i] < 0 || lbi[i+1] < lbi[i]))); i--) {\n"
"                float tv = lbv[i]; int ti = lbi[i];\n"
"                lbv[i] = lbv[i+1]; lbi[i] = lbi[i+1];\n"
"                lbv[i+1] = tv; lbi[i+1] = ti;\n"
"            }\n"
"        }\n"
"    }\n"
"    /* Parallel reduction across threads: each thread has its top-k */\n"
"    /* Shared memory is sized for up to 128 threads and k <= 8. */\n"
"    __shared__ float s_vals[1024];\n"
"    __shared__ int s_idxs[1024];\n"
"    for (int i = 0; i < k; i++) {\n"
"        s_vals[tid * k + i] = lbv[i];\n"
"        s_idxs[tid * k + i] = lbi[i];\n"
"    }\n"
"    __syncthreads();\n"
"    /* Tree reduction: merge pairs of threads */\n"
"    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) {\n"
"            for (int i = 0; i < k; i++) {\n"
"                /* Merge s_vals[tid*k + i] into best list from other thread */\n"
"                for (int j = 0; j < k; j++) {\n"
"                    float ov = s_vals[(tid+stride)*k + j];\n"
"                    int oi = s_idxs[(tid+stride)*k + j];\n"
"                    if (!(ov == ov)) ov = -3.402823466e+38F;\n"
"                    if (oi >= 0 && (ov > lbv[k-1] || (ov == lbv[k-1] && (lbi[k-1] < 0 || oi < lbi[k-1])))) {\n"
"                        lbv[k-1] = ov; lbi[k-1] = oi;\n"
"                        for (int ii = k-2; ii >= 0 &&\n"
"                             (lbv[ii+1] > lbv[ii] || (lbv[ii+1] == lbv[ii] && lbi[ii+1] >= 0 && (lbi[ii] < 0 || lbi[ii+1] < lbi[ii]))); ii--) {\n"
"                            float tv = lbv[ii]; int ti = lbi[ii];\n"
"                            lbv[ii] = lbv[ii+1]; lbi[ii] = lbi[ii+1];\n"
"                            lbv[ii+1] = tv; lbi[ii+1] = ti;\n"
"                        }\n"
"                    }\n"
"                }\n"
"            }\n"
"            for (int i = 0; i < k; i++) {\n"
"                s_vals[tid * k + i] = lbv[i];\n"
"                s_idxs[tid * k + i] = lbi[i];\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (tid == 0) {\n"
"        float max_val = s_vals[0];\n"
"        for (int i = 1; i < k; i++) {\n"
"            if (s_vals[i] > max_val) max_val = s_vals[i];\n"
"        }\n"
"        float sum = 0.0f;\n"
"        for (int i = 0; i < k; i++) {\n"
"            indices[token * k + i] = s_idxs[i];\n"
"            float v = expf(s_vals[i] - max_val);\n"
"            weights[token * k + i] = v;\n"
"            sum += v;\n"
"        }\n"
"        float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;\n"
"        for (int i = 0; i < k; i++) {\n"
"            weights[token * k + i] *= inv_sum;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* Batched softplus+bias+mul: out[t, i] = softplus(in[t, i] + bias[i]) * a[i] */\n"
"__global__ void batch_softplus_mul_f32(float *out, const float *in, const float *bias,\n"
"                                        const float *a, int n, int n_tokens) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n * n_tokens;\n"
"    if (idx >= total) return;\n"
"    int i = idx % n;\n"
"    float x = in[idx] + bias[i];\n"
"    float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));\n"
"    out[idx] = sp * a[i];\n"
"}\n"
"\n"
"/* In-place exp for precomputed Delta-Net decay: data[i] = exp(data[i]) */\n"
"__global__ void exp_inplace_f32(float *data, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) data[i] = expf(data[i]);\n"
"}\n"
"\n"
"/* Batched NeoX RoPE for flattened [token][head][dim] layout. */\n"
"__global__ void batch_rope_neox_f32(float *vec, int heads_per_token, int total_heads,\n"
"                                     int head_dim, int start_pos, float freq_base,\n"
"                                     int n_rope_pairs) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= total_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half_dim = head_dim / 2;\n"
"    if (j >= half_dim) return;\n"
"    if (n_rope_pairs > 0 && j >= n_rope_pairs) return;\n"
"    int pair_off = (n_rope_pairs > 0 && n_rope_pairs < half_dim) ? n_rope_pairs : half_dim;\n"
"    int rope_dim = (n_rope_pairs > 0) ? 2 * n_rope_pairs : head_dim;\n"
"    int token = h / heads_per_token;\n"
"    int pos = start_pos + token;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)rope_dim);\n"
"    float theta = (float)pos * freq;\n"
"    float cos_t = cosf(theta);\n"
"    float sin_t = sinf(theta);\n"
"    float *v = vec + (size_t)h * head_dim;\n"
"    float v0 = v[j];\n"
"    float v1 = v[j + pair_off];\n"
"    v[j] = v0 * cos_t - v1 * sin_t;\n"
"    v[j + pair_off] = v0 * sin_t + v1 * cos_t;\n"
"}\n"
"\n"
"/* Batched factored RoPE: like rope_with_factors_f32 but derives a PER-TOKEN\n"
"   position (start_pos + h/heads_per_token) instead of a single constant. The\n"
"   non-batched rope_with_factors_f32 applies one position to every block, which\n"
"   is only correct for a single token; batched prefill must rope token t at\n"
"   start_pos+t. inv_freq holds the proportional/NTK-scaled frequencies. */\n"
"__global__ void batch_rope_with_factors_f32(float *vec, int heads_per_token,\n"
"                                             int total_heads, int head_dim,\n"
"                                             int start_pos, const float *inv_freq,\n"
"                                             int n_pairs) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= total_heads) return;\n"
"    int j = threadIdx.x;\n"
"    if (j >= n_pairs) return;\n"
"    int token = h / heads_per_token;\n"
"    int pos = start_pos + token;\n"
"    float freq = (float)pos * inv_freq[j];\n"
"    float cos_v = cosf(freq), sin_v = sinf(freq);\n"
"    float *v = vec + (size_t)h * head_dim;\n"
"    float r0 = v[j], r1 = v[j + n_pairs];\n"
"    v[j]           = r0 * cos_v - r1 * sin_v;\n"
"    v[j + n_pairs] = r0 * sin_v + r1 * cos_v;\n"
"}\n"
"\n"
"/* Batched depthwise causal conv1d + SiLU over a token chunk.\n"
"   data is laid out as [n_tokens, qkv_dim] and updated in-place to hold conv output. */\n"
"__global__ void batch_conv1d_depthwise_silu_f32(float *data, float *conv_state,\n"
"                                                 const float *weight, int qkv_dim,\n"
"                                                 int conv_k, int n_tokens) {\n"
"    int j = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (j >= qkv_dim) return;\n"
"    float hist[8];\n"
"    for (int f = 0; f < conv_k - 1; f++) hist[f] = conv_state[f * qkv_dim + j];\n"
"    for (int t = 0; t < n_tokens; t++) {\n"
"        float in = data[(size_t)t * qkv_dim + j];\n"
"        float sum = 0.0f;\n"
"        for (int f = 0; f < conv_k - 1; f++) sum += weight[j * conv_k + f] * hist[f];\n"
"        sum += weight[j * conv_k + (conv_k - 1)] * in;\n"
"        data[(size_t)t * qkv_dim + j] = sum / (1.0f + expf(-sum));\n"
"        for (int f = 0; f < conv_k - 2; f++) hist[f] = hist[f + 1];\n"
"        hist[conv_k - 2] = in;\n"
"    }\n"
"    for (int f = 0; f < conv_k - 1; f++) conv_state[f * qkv_dim + j] = hist[f];\n"
"}\n"
"\n"
"/* Batched Delta-Net scan over a token chunk.\n"
"   2D grid: blockIdx.x = h (dt_rank), blockIdx.y = r (d_state).\n"
"   Each block processes 1 (h, r) pair with 1 warp (32 threads).\n"
"   qkv holds [Q(group), K(group), V(dt_rank)] per token after conv+norm. */\n"
"__global__ void batch_deltanet_scan_f32(float *out, float *state, const float *qkv,\n"
"                                         const float *alpha, const float *beta,\n"
"                                         int n_tokens, int dt_rank, int d_state,\n"
"                                         int n_group) {\n"
"    int h = blockIdx.x;\n"
"    int r = blockIdx.y * blockDim.y + threadIdx.y;\n"  /* blockDim.y warps/block, each an independent r (no shared, no sync) */
"    int lane = threadIdx.x;\n"
"    if (h >= dt_rank || r >= d_state) return;\n"
"    int qkv_dim = 2 * n_group * d_state + dt_rank * d_state;\n"
"    float scale = rsqrtf((float)d_state);\n"
"    if (d_state == 128) {\n"
"        size_t base = (size_t)h * 128 * 128 + (size_t)r * 128;\n"
"        float s0 = state[base + lane];\n"
"        float s1 = state[base + 32 + lane];\n"
"        float s2 = state[base + 64 + lane];\n"
"        float s3 = state[base + 96 + lane];\n"
"        int gh = h % n_group;\n"
"        for (int t = 0; t < n_tokens; t++) {\n"
"            const float *tok = qkv + (size_t)t * qkv_dim;\n"
"            const float *q = tok + gh * 128;\n"
"            const float *k = tok + (size_t)n_group * 128 + gh * 128;\n"
"            const float *v = tok + (size_t)2 * n_group * 128 + h * 128;\n"
"            float decay = alpha[(size_t)t * dt_rank + h];\n"
"            float b = beta[(size_t)t * dt_rank + h];\n"
"            float k0 = k[lane], k1 = k[32 + lane], k2 = k[64 + lane], k3 = k[96 + lane];\n"
"            float q0 = q[lane], q1 = q[32 + lane], q2 = q[64 + lane], q3 = q[96 + lane];\n"
"            s0 *= decay; s1 *= decay; s2 *= decay; s3 *= decay;\n"
"            float sk = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;\n"
"            /* All-reduce (butterfly): every lane must hold the full S.k dot product, */\n"
"            /* because delta is a per-row scalar applied to all 128 state columns. */\n"
"            for (int o = 16; o > 0; o >>= 1) sk += __shfl_xor_sync(0xffffffff, sk, o);\n"
"            float delta = (v[r] - sk) * b;\n"
"            s0 += delta * k0; s1 += delta * k1; s2 += delta * k2; s3 += delta * k3;\n"
"            float o_val = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;\n"
"            for (int o = 16; o > 0; o >>= 1) o_val += __shfl_down_sync(0xffffffff, o_val, o);\n"
"            if (lane == 0) out[(size_t)t * dt_rank * 128 + h * 128 + r] = o_val * scale;\n"
"        }\n"
"        state[base + lane] = s0;\n"
"        state[base + 32 + lane] = s1;\n"
"        state[base + 64 + lane] = s2;\n"
"        state[base + 96 + lane] = s3;\n"
"        return;\n"
"    }\n"
"    __shared__ float S[128];\n"
"    for (int c = lane; c < d_state; c += 32)\n"
"        S[c] = state[(size_t)h * d_state * d_state + (size_t)r * d_state + c];\n"
"    __syncthreads();\n"
"    for (int t = 0; t < n_tokens; t++) {\n"
"        const float *tok = qkv + (size_t)t * qkv_dim;\n"
"        int gh = h % n_group;\n"
"        const float *q = tok + gh * d_state;\n"
"        const float *k = tok + (size_t)n_group * d_state + gh * d_state;\n"
"        const float *v = tok + (size_t)2 * n_group * d_state + h * d_state;\n"
"        float decay = alpha[(size_t)t * dt_rank + h];\n"
"        float b = beta[(size_t)t * dt_rank + h];\n"
"        for (int c = lane; c < d_state; c += 32) S[c] *= decay;\n"
"        __syncthreads();\n"
"        float sk = 0.0f;\n"
"        for (int c = lane; c < d_state; c += 32) sk += S[c] * k[c];\n"
"        /* All-reduce (butterfly): delta is a per-row scalar applied to every column, */\n"
"        /* so all lanes need the full S.k dot product, not just lane 0. */\n"
"        for (int o = 16; o > 0; o >>= 1) sk += __shfl_xor_sync(0xffffffff, sk, o);\n"
"        float delta = (v[r] - sk) * b;\n"
"        for (int c = lane; c < d_state; c += 32) S[c] += delta * k[c];\n"
"        __syncthreads();\n"
"        float o_val = 0.0f;\n"
"        for (int c = lane; c < d_state; c += 32) o_val += S[c] * q[c];\n"
"        for (int offset = 16; offset > 0; offset >>= 1)\n"
"            o_val += __shfl_down_sync(0xffffffff, o_val, offset);\n"
"        if (lane == 0)\n"
"            out[(size_t)t * dt_rank * d_state + h * d_state + r] = o_val * scale;\n"
"    }\n"
"    for (int c = lane; c < d_state; c += 32)\n"
"        state[(size_t)h * d_state * d_state + (size_t)r * d_state + c] = S[c];\n"
"}\n"
"\n"
"/* Batched F16 KV store: store n_tokens K/V at consecutive positions */\n"
"/* Grid: [ceil(kv_dim/256), n_tokens], Block: 256 */\n"
"__global__ void batch_kv_store_f16(half_raw *key_cache, half_raw *value_cache,\n"
"                                    const float *k_batch, const float *v_batch,\n"
"                                    int start_pos, int kv_dim, int n_tokens) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int t = blockIdx.y;\n"
"    if (i >= kv_dim || t >= n_tokens) return;\n"
"    int pos = start_pos + t;\n"
"    key_cache[(size_t)pos * kv_dim + i] = float_to_half(k_batch[(size_t)t * kv_dim + i]);\n"
"    value_cache[(size_t)pos * kv_dim + i] = float_to_half(v_batch[(size_t)t * kv_dim + i]);\n"
"}\n"
"\n"
"/* Batched SWA KV store: store at circular buffer positions */\n"
"__global__ void batch_kv_store_swa_f16(half_raw *key_cache, half_raw *value_cache,\n"
"                                        const float *k_batch, const float *v_batch,\n"
"                                        int start_pos, int kv_dim, int n_tokens, int window_size) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int t = blockIdx.y;\n"
"    if (i >= kv_dim || t >= n_tokens) return;\n"
"    int slot = (start_pos + t) % window_size;\n"
"    key_cache[(size_t)slot * kv_dim + i] = float_to_half(k_batch[(size_t)t * kv_dim + i]);\n"
"    value_cache[(size_t)slot * kv_dim + i] = float_to_half(v_batch[(size_t)t * kv_dim + i]);\n"
"}\n"
"\n"
"/* ==== Vision Encoder Kernels ==== */\n"
"\n"
"/* Batched F16 linear: weight[out_dim, in_dim] (F16) × input[N, in_dim] (F32) → output[N, out_dim] (F32) */\n"
"/* Grid: [out_dim, N], Block: 256 */\n"
"__global__ void vision_linear_f16(float *output, const half_raw *weight,\n"
"                                   const float *input, int out_dim, int in_dim, int n_tokens) {\n"
"    int row = blockIdx.x;\n"
"    int token = blockIdx.y;\n"
"    if (row >= out_dim || token >= n_tokens) return;\n"
"    int tid = threadIdx.x;\n"
"    const half_raw *w = weight + (size_t)row * in_dim;\n"
"    const float *x = input + (size_t)token * in_dim;\n"
"    float sum = 0.0f;\n"
"    for (int i = tid; i < in_dim; i += blockDim.x)\n"
"        sum += half_to_float(w[i]) * x[i];\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);\n"
"    __shared__ float ws[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int nw = (blockDim.x + 31) / 32;\n"
"        for (int w2 = 0; w2 < nw; w2++) total += ws[w2];\n"
"        output[(size_t)token * out_dim + row] = total;\n"
"    }\n"
"}\n"
"\n"
"/* Batched RMSNorm: normalize each of N tokens independently */\n"
"/* Grid: [N], Block: 256.  If weight != NULL, multiply by weight. */\n"
"__global__ void vision_rmsnorm(float *x, const half_raw *weight, int dim, float eps, int n_tokens) {\n"
"    int token = blockIdx.x;\n"
"    if (token >= n_tokens) return;\n"
"    int tid = threadIdx.x;\n"
"    float *v = x + (size_t)token * dim;\n"
"    float ss = 0.0f;\n"
"    for (int i = tid; i < dim; i += blockDim.x) ss += v[i] * v[i];\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        ss += __shfl_down_sync(0xFFFFFFFF, ss, offset);\n"
"    __shared__ float ws[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) ws[wid] = ss;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float s = 0.0f;\n"
"        int nw = (blockDim.x + 31) / 32;\n"
"        for (int w2 = 0; w2 < nw; w2++) s += ws[w2];\n"
"        ws[0] = rsqrtf(s / dim + eps);\n"
"    }\n"
"    __syncthreads();\n"
"    float scale = ws[0];\n"
"    if (weight) {\n"
"        for (int i = tid; i < dim; i += blockDim.x)\n"
"            v[i] = v[i] * scale * half_to_float(weight[i]);\n"
"    } else {\n"
"        for (int i = tid; i < dim; i += blockDim.x)\n"
"            v[i] *= scale;\n"
"    }\n"
"}\n"
"\n"
"/* Batched RMSNorm with F32 weights (for LLM layer norms) */\n"
"__global__ void batch_rmsnorm_f32(float *x, const float *weight, int dim, float eps, int n_tokens) {\n"
"    int token = blockIdx.x;\n"
"    if (token >= n_tokens) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    float *v = x + (size_t)token * dim;\n"
"    __shared__ float sdata[256];\n"
"    float ss = 0.0f;\n"
"    for (int i = tid; i < dim; i += nthreads) ss += v[i] * v[i];\n"
"    sdata[tid] = ss;\n"
"    __syncthreads();\n"
"    for (int s = nthreads / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float scale = rsqrtf(sdata[0] / (float)dim + eps);\n"
"    if (weight) {\n"
"        for (int i = tid; i < dim; i += nthreads)\n"
"            v[i] = v[i] * scale * weight[i];\n"
"    } else {\n"
"        for (int i = tid; i < dim; i += nthreads)\n"
"            v[i] *= scale;\n"
"    }\n"
"}\n"
"\n"
"/* Batched per-head L2 normalize without learned weights, using a token stride. */\n"
"__global__ void batch_l2_norm_heads_strided_f32(float *vec, int heads_per_token,\n"
"                                                 int n_tokens, int head_dim,\n"
"                                                 int token_stride, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int h = blockIdx.x;\n"
"    int total_heads = n_tokens * heads_per_token;\n"
"    if (h >= total_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int token = h / heads_per_token;\n"
"    int head = h % heads_per_token;\n"
"    float *v = vec + (size_t)token * token_stride + head * head_dim;\n"
"    float val = (tid < head_dim) ? v[tid] : 0.0f;\n"
"    sdata[tid] = val * val;\n"
"    __syncthreads();\n"
"    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) sdata[tid] += sdata[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float scale = rsqrtf(sdata[0] + eps);\n"
"    if (tid < head_dim) v[tid] = val * scale;\n"
"}\n"
"\n"
"/* Batched causal attention for a chunk of query tokens against the F16 KV cache. */\n"
"__global__ void batch_attn_causal_f32(float *out, const float *q_batch,\n"
"                                       const half_raw *key_cache, const half_raw *value_cache,\n"
"                                       int n_tokens, int start_pos,\n"
"                                       int n_heads, int n_kv_heads,\n"
"                                       int head_dim, int kv_dim, float scale) {\n"
"    extern __shared__ float scores[];\n"
"    __shared__ float warp_vals[8];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int q_dim = n_heads * head_dim;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"    int qi = blockIdx.y;\n"
"    if (qi >= n_tokens) return;\n"
"    int seq_len = start_pos + qi + 1;\n"
"    const float *q_h = q_batch + (size_t)qi * q_dim + h * head_dim;\n"
"    int lane_qk = tid & 31;\n"
"    int warp_qk = tid >> 5;\n"
"    int nwarps_qk = nt >> 5;\n"
"    for (int base = 0; base < seq_len; base += nwarps_qk) {\n"
"        int t = base + warp_qk;\n"
"        float dot = 0.0f;\n"
"        if (t < seq_len) {\n"
"            const half_raw *k_t = key_cache + (size_t)t * kv_dim + kv_h * head_dim;\n"
"            for (int d = lane_qk; d < head_dim; d += 32) dot += q_h[d] * half_to_float(k_t[d]);\n"
"        }\n"
"        for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);\n"
"        if (lane_qk == 0 && t < seq_len) scores[t] = dot * scale;\n"
"    }\n"
"    __syncthreads();\n"
"    float local_max = -1e30f;\n"
"    for (int t = tid; t < seq_len; t += nt)\n"
"        if (scores[t] > local_max) local_max = scores[t];\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) warp_vals[wid] = local_max;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float m = warp_vals[0];\n"
"        int nw = (nt + 31) / 32;\n"
"        for (int w = 1; w < nw; w++) if (warp_vals[w] > m) m = warp_vals[w];\n"
"        warp_vals[0] = m;\n"
"    }\n"
"    __syncthreads();\n"
"    float max_val = warp_vals[0];\n"
"    float local_sum = 0.0f;\n"
"    for (int t = tid; t < seq_len; t += nt) {\n"
"        float e = expf(scores[t] - max_val);\n"
"        scores[t] = e;\n"
"        local_sum += e;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);\n"
"    if (ln == 0) warp_vals[wid] = local_sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float s = 0.0f;\n"
"        int nw = (nt + 31) / 32;\n"
"        for (int w = 0; w < nw; w++) s += warp_vals[w];\n"
"        warp_vals[0] = 1.0f / s;\n"
"    }\n"
"    __syncthreads();\n"
"    float inv_sum = warp_vals[0];\n"
"    for (int t = tid; t < seq_len; t += nt) scores[t] *= inv_sum;\n"
"    __syncthreads();\n"
"    float *out_h = out + (size_t)qi * q_dim + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += nt) {\n"
"        float acc = 0.0f;\n"
"        for (int t = 0; t < seq_len; t++) {\n"
"            const half_raw *v_t = value_cache + (size_t)t * kv_dim + kv_h * head_dim;\n"
"            acc += scores[t] * half_to_float(v_t[d]);\n"
"        }\n"
"        out_h[d] = acc;\n"
"    }\n"
"}\n"
"\n"
"\n"
"/* ---- batch_attn_all_tokens_f32: all tokens per head, D=512 optimized ---- */\n"
"/* Grid: (n_heads, 1), Block: 256. Reads KV cache ONCE per head for all tokens.\n"
"   Uses shared memory for scores [n_tokens x seq_len]. */\n"
"__global__ void batch_attn_all_tokens_f32(float *out, const float *q_batch,\n"
"                                            const half_raw *key_cache, const half_raw *value_cache,\n"
"                                            int n_tokens, int start_pos,\n"
"                                            int n_heads, int n_kv_heads,\n"
"                                            int head_dim, int kv_dim, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    float *scores = smem;\n"
"    int h = blockIdx.x;\n"
"    int gqa_ratio = n_heads / n_kv_heads;\n"
"    int kv_h = h / gqa_ratio;\n"
"    int q_dim = n_heads * head_dim;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int max_seq = start_pos + n_tokens;\n"
"\n"
"    /* Compute QK scores: for each token qi, dot product over head_dim */\n"
"    for (int qi = tid; qi < n_tokens; qi += nt) {\n"
"        const float *q_h = q_batch + (size_t)qi * q_dim + h * head_dim;\n"
"        int kv_lim = start_pos + qi + 1;\n"
"        if (kv_lim > max_seq) kv_lim = max_seq;\n"
"        int row_off = qi * max_seq;\n"
"        float local_max = -1e30f;\n"
"        for (int t = 0; t < kv_lim; t++) {\n"
"            const half_raw *k_t = key_cache + (size_t)t * kv_dim + kv_h * head_dim;\n"
"            float s = 0.0f;\n"
"            for (int d = 0; d < head_dim; d++) s += q_h[d] * half_to_float(k_t[d]);\n"
"            s *= scale;\n"
"            scores[row_off + t] = s;\n"
"            if (s > local_max) local_max = s;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* Softmax: find max, sum, normalize */\n"
"    for (int qi = tid; qi < n_tokens; qi += nt) {\n"
"        int kv_lim = start_pos + qi + 1;\n"
"        if (kv_lim > max_seq) kv_lim = max_seq;\n"
"        int row_off = qi * max_seq;\n"
"        float mx = scores[row_off];\n"
"        for (int t = 1; t < kv_lim; t++) if (scores[row_off + t] > mx) mx = scores[row_off + t];\n"
"        float sum = 0.0f;\n"
"        for (int t = 0; t < kv_lim; t++) {\n"
"            float e = expf(scores[row_off + t] - mx);\n"
"            scores[row_off + t] = e;\n"
"            sum += e;\n"
"        }\n"
"        float inv = 1.0f / sum;\n"
"        for (int t = 0; t < kv_lim; t++) scores[row_off + t] *= inv;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* V accumulation: each thread handles a subset of d values */\n"
"    for (int d = tid; d < head_dim; d += nt) {\n"
"        for (int qi = 0; qi < n_tokens; qi++) {\n"
"            int kv_lim = start_pos + qi + 1;\n"
"            if (kv_lim > max_seq) kv_lim = max_seq;\n"
"            float acc = 0.0f;\n"
"            int row_off = qi * max_seq;\n"
"            for (int t = 0; t < kv_lim; t++) {\n"
"                const half_raw *v_t = value_cache + (size_t)t * kv_dim + kv_h * head_dim;\n"
"                acc += scores[row_off + t] * half_to_float(v_t[d]);\n"
"            }\n"
"            float *out_h = out + (size_t)qi * q_dim + h * head_dim;\n"
"            out_h[d] = acc;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- Fused causal softmax: applies causal mask + softmax to scores ---- */\n"
"/* scores [n_q_tokens x seq_len] in-place. Grid: n_q_tokens. */\n"
"/* tok_div = token index divisor for GQA groups (e.g., 2 for gqa_ratio=2) */\n"
"__global__ void causal_softmax_f32(float *scores, int n_q_tokens,\n"
"                                    int seq_len, int start_pos, int tok_div) {\n"
"    int qi = blockIdx.x;\n"
"    if (qi >= n_q_tokens) return;\n"
"    int tok = qi / tok_div;\n"
"    int kv_lim = start_pos + tok + 1;\n"
"    if (kv_lim > seq_len) kv_lim = seq_len;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    float *row = scores + (size_t)qi * seq_len;\n"
"    float mx = -1e30f;\n"
"    for (int t = tid; t < kv_lim; t += nt)\n"
"        if (row[t] > mx) mx = row[t];\n"
"    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xFFFFFFFF, mx, off));\n"
"    __shared__ float warp_vals[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) warp_vals[wid] = mx;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float m = warp_vals[0];\n"
"        for (int w = 1; w < (nt + 31) / 32; w++) if (warp_vals[w] > m) m = warp_vals[w];\n"
"        warp_vals[0] = m;\n"
"    }\n"
"    __syncthreads();\n"
"    mx = warp_vals[0];\n"
"    float sum = 0.0f;\n"
"    for (int t = tid; t < kv_lim; t += nt) {\n"
"        float e = expf(row[t] - mx);\n"
"        row[t] = e;\n"
"        sum += e;\n"
"    }\n"
"    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, off);\n"
"    if (ln == 0) warp_vals[wid] = sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float s = 0.0f;\n"
"        for (int w = 0; w < (nt + 31) / 32; w++) s += warp_vals[w];\n"
"        warp_vals[0] = 1.0f / s;\n"
"    }\n"
"    __syncthreads();\n"
"    float inv = warp_vals[0];\n"
"    for (int t = tid; t < kv_lim; t += nt) row[t] *= inv;\n"
"    /* Zero out future positions */\n"
"    for (int t = kv_lim + tid; t < seq_len; t += nt) row[t] = 0.0f;\n"
"}\n"
"\n"
"/* Per-head RMSNorm for Q/K: normalize each head independently */\n"
"/* Grid: [n_tokens, n_heads], Block: head_dim (padded to power of 2) */\n"
"__global__ void vision_head_rmsnorm(float *x, const half_raw *weight, int n_tokens,\n"
"                                     int n_heads, int head_dim, int dim, float eps) {\n"
"    int token = blockIdx.x;\n"
"    int head = blockIdx.y;\n"
"    if (token >= n_tokens || head >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    float *v = x + (size_t)token * dim + head * head_dim;\n"
"    float ss = 0.0f;\n"
"    if (tid < head_dim) ss = v[tid] * v[tid];\n"
"    /* Cross-warp reduction */\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        ss += __shfl_down_sync(0xFFFFFFFF, ss, offset);\n"
"    __shared__ float warp_ss[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) warp_ss[wid] = ss;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w = 0; w < nw; w++) total += warp_ss[w];\n"
"        warp_ss[0] = rsqrtf(total / head_dim + eps);\n"
"    }\n"
"    __syncthreads();\n"
"    float scale = warp_ss[0];\n"
"    if (tid < head_dim) {\n"
"        v[tid] = v[tid] * scale * (weight ? half_to_float(weight[tid]) : 1.0f);\n"
"    }\n"
"}\n"
"\n"
"/* 2D RoPE for vision tokens: apply rotation based on pos_x/pos_y coordinates */\n"
"/* Grid: [n_tokens, n_heads], Block: head_dim/4 */\n"
"__global__ void vision_rope_2d(float *x, const int *pos_x, const int *pos_y,\n"
"                                int n_tokens, int n_heads, int head_dim, int dim, float theta) {\n"
"    int token = blockIdx.x;\n"
"    int head = blockIdx.y;\n"
"    if (token >= n_tokens || head >= n_heads) return;\n"
"    int j = threadIdx.x;\n"
"    int half = head_dim / 2;\n"
"    int quarter = half / 2;\n"
"    if (j >= quarter) return;\n"
"    float *v = x + (size_t)token * dim + head * head_dim;\n"
"    /* First half: rotate by pos_x */\n"
"    float freq_x = (float)pos_x[token] / powf(theta, (float)(2*j) / half);\n"
"    float cx = cosf(freq_x), sx = sinf(freq_x);\n"
"    float r0 = v[j], r1 = v[j + quarter];\n"
"    v[j]           = r0 * cx - r1 * sx;\n"
"    v[j + quarter] = r0 * sx + r1 * cx;\n"
"    /* Second half: rotate by pos_y */\n"
"    float freq_y = (float)pos_y[token] / powf(theta, (float)(2*j) / half);\n"
"    float cy = cosf(freq_y), sy = sinf(freq_y);\n"
"    float r2 = v[half + j], r3 = v[half + j + quarter];\n"
"    v[half + j]           = r2 * cy - r3 * sy;\n"
"    v[half + j + quarter] = r2 * sy + r3 * cy;\n"
"}\n"
"\n"
"/* Vision attention: full NxN attention for all heads */\n"
"/* Grid: [n_heads, n_tokens], Block: 256 */\n"
"/* Each block computes one output row (one token, one head) */\n"
"__global__ void vision_attention(float *out, const float *Q, const float *K, const float *V,\n"
"                                  int n_tokens, int n_heads, int head_dim, int dim) {\n"
"    int head = blockIdx.x;\n"
"    int qi = blockIdx.y;  /* query token index */\n"
"    if (head >= n_heads || qi >= n_tokens) return;\n"
"    int tid = threadIdx.x;\n"
"    int nthreads = blockDim.x;\n"
"    extern __shared__ float smem[];\n"
"    float *scores = smem;\n"
"\n"
"    const float *q_h = Q + (size_t)qi * dim + head * head_dim;\n"
"    float scale = 1.0f;  /* Gemma4 uses Q/K norm so scale=1 */\n"
"\n"
"    /* Compute Q·K^T scores for this query against all keys */\n"
"    for (int ki = tid; ki < n_tokens; ki += nthreads) {\n"
"        const float *k_h = K + (size_t)ki * dim + head * head_dim;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) dot += q_h[d] * k_h[d];\n"
"        scores[ki] = dot * scale;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* Softmax */\n"
"    float local_max = -1e30f;\n"
"    for (int t = tid; t < n_tokens; t += nthreads)\n"
"        if (scores[t] > local_max) local_max = scores[t];\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));\n"
"    __shared__ float warp_vals[8];\n"
"    int wid = tid / 32, ln = tid % 32;\n"
"    if (ln == 0) warp_vals[wid] = local_max;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float m = warp_vals[0];\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w2 = 1; w2 < nw; w2++) if (warp_vals[w2] > m) m = warp_vals[w2];\n"
"        warp_vals[0] = m;\n"
"    }\n"
"    __syncthreads();\n"
"    float max_val = warp_vals[0];\n"
"\n"
"    float local_sum = 0.0f;\n"
"    for (int t = tid; t < n_tokens; t += nthreads) {\n"
"        float e = expf(scores[t] - max_val);\n"
"        scores[t] = e;\n"
"        local_sum += e;\n"
"    }\n"
"    for (int offset = 16; offset > 0; offset >>= 1)\n"
"        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);\n"
"    if (ln == 0) warp_vals[wid] = local_sum;\n"
"    __syncthreads();\n"
"    if (tid == 0) {\n"
"        float s = 0.0f;\n"
"        int nw = (nthreads + 31) / 32;\n"
"        for (int w2 = 0; w2 < nw; w2++) s += warp_vals[w2];\n"
"        warp_vals[0] = s;\n"
"    }\n"
"    __syncthreads();\n"
"    float inv_sum = 1.0f / warp_vals[0];\n"
"    for (int t = tid; t < n_tokens; t += nthreads) scores[t] *= inv_sum;\n"
"    __syncthreads();\n"
"\n"
"    /* Weighted V sum */\n"
"    float *out_h = out + (size_t)qi * dim + head * head_dim;\n"
"    for (int d = tid; d < head_dim; d += nthreads) {\n"
"        float acc = 0.0f;\n"
"        for (int t = 0; t < n_tokens; t++)\n"
"            acc += scores[t] * V[(size_t)t * dim + head * head_dim + d];\n"
"        out_h[d] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* GELU gate: out[i] = gelu(gate[i]) * up[i] */\n"
"__global__ void vision_gelu_gate(float *gate, const float *up, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        float x = gate[i];\n"
"        float g = x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));\n"
"        gate[i] = g * up[i];\n"
"    }\n"
"}\n"
"\n"
"/* Element-wise add: dst[i] += src[i] */\n"
"__global__ void vision_add(float *dst, const float *src, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) dst[i] += src[i];\n"
"}\n"
"\n"
"/* Scale: dst[i] *= scale */\n"
"__global__ void vision_scale(float *dst, float scale, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) dst[i] *= scale;\n"
"}\n"
"\n"
"/* Add bias from F16 lookup: dst[token*dim + i] += half_to_float(table[idx*dim + i]) */\n"
"/* Grid: [n_tokens], Block: 256 */\n"
"__global__ void vision_add_pos_embd(float *dst, const half_raw *table, const int *indices,\n"
"                                      int dim, int n_tokens) {\n"
"    int token = blockIdx.x;\n"
"    if (token >= n_tokens) return;\n"
"    int idx = indices[token];\n"
"    float *d = dst + (size_t)token * dim;\n"
"    const half_raw *s = table + (size_t)idx * dim;\n"
"    for (int i = threadIdx.x; i < dim; i += blockDim.x)\n"
"        d[i] += half_to_float(s[i]);\n"
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
    CUdeviceptr attn_q_w_f16; /* optional F16 shadow copy for batched Q8_0 prefill */
    CUdeviceptr attn_q_w_bm;  /* block-major repacked (0 if not repacked) */
    CUdeviceptr attn_k_w;
    CUdeviceptr attn_k_w_f16;
    CUdeviceptr attn_k_w_bm;
    CUdeviceptr attn_v_w;
    CUdeviceptr attn_v_w_f16;
    CUdeviceptr attn_v_w_bm;
    CUdeviceptr attn_q_norm_w; /* F32 [head_dim] */
    CUdeviceptr attn_k_norm_w; /* F32 [head_dim] */
    CUdeviceptr attn_q_bias;   /* F32 [q_dim], 0 if absent (Qwen2.5-VL) */
    CUdeviceptr attn_k_bias;   /* F32 [kv_dim] */
    CUdeviceptr attn_v_bias;   /* F32 [kv_dim] */
    CUdeviceptr attn_output_w; /* F16 */
    CUdeviceptr attn_output_w_f16;
    CUdeviceptr attn_output_w_bm;
    CUdeviceptr ffn_norm_w;    /* F32 [n_embd] */
    CUdeviceptr ffn_gate_w;    /* F16 */
    CUdeviceptr ffn_gate_w_f16;
    CUdeviceptr ffn_gate_w_bm;
    CUdeviceptr ffn_up_w;      /* F16 */
    CUdeviceptr ffn_up_w_f16;
    CUdeviceptr ffn_up_w_bm;
    CUdeviceptr ffn_down_w;    /* F16 */
    CUdeviceptr ffn_down_w_f16;
    CUdeviceptr ffn_down_w_bm;

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
    CUdeviceptr moe_gate_exps_w_f16; /* F16 shadow for cuBLAS */
    CUdeviceptr moe_up_exps_w;      /* K-quant [expert_ff, n_embd, n_experts] */
    CUdeviceptr moe_up_exps_w_f16;  /* F16 shadow for cuBLAS */
    CUdeviceptr moe_down_exps_w;    /* K-quant [n_embd, expert_ff, n_experts] */
    CUdeviceptr moe_down_exps_w_f16; /* F16 shadow for cuBLAS */
    uint64_t moe_f16_mask[4];       /* bitmask: which experts have F16 uploaded (256 bits) */
    int moe_gate_exps_type, moe_up_exps_type, moe_down_exps_type;
    int moe_exp_rows_gu, moe_exp_cols_gu;   /* per-expert dims for gate/up: [expert_ff, n_embd] */
    int moe_exp_rows_d, moe_exp_cols_d;     /* per-expert dims for down: [n_embd, expert_ff] */
    size_t moe_exp_stride_gu;       /* byte stride between experts in gate/up tensors */
    size_t moe_exp_stride_d;        /* byte stride between experts in down tensor */

    CUdeviceptr moe_shared_gate_w;      /* F32 [n_embd] — shared expert sigmoid gate */
    CUdeviceptr moe_shared_ffn_gate_w;  /* F16/BF16 [shared_ff, n_embd] — shared expert */
    CUdeviceptr moe_shared_ffn_gate_w_f16;
    CUdeviceptr moe_shared_ffn_up_w;    /* F16/BF16 [shared_ff, n_embd] */
    CUdeviceptr moe_shared_ffn_up_w_f16;
    CUdeviceptr moe_shared_ffn_down_w;  /* F16/BF16 [n_embd, shared_ff] */
    CUdeviceptr moe_shared_ffn_down_w_f16;
    int moe_shared_gate_type, moe_shared_up_type, moe_shared_down_type;
    int moe_shared_gate_rows, moe_shared_gate_cols;
    int moe_shared_up_rows, moe_shared_up_cols;
    int moe_shared_down_rows, moe_shared_down_cols;

    /* SSM (Delta-Net) layer fields — only used when is_ssm=1 */
    int is_ssm;
    CUdeviceptr ssm_qkv_w;      /* [qkv_dim, n_embd] quantized */
    CUdeviceptr ssm_qkv_w_f16;  /* optional F16 shadow copy for batched Q8_0 prefill */
    CUdeviceptr ssm_gate_w;     /* [d_inner, n_embd] quantized */
    CUdeviceptr ssm_gate_w_f16; /* optional F16 shadow */
    CUdeviceptr ssm_alpha_w;    /* [dt_rank, n_embd] quantized */
    CUdeviceptr ssm_beta_w;     /* [dt_rank, n_embd] quantized */
    CUdeviceptr ssm_out_w;      /* [n_embd, d_inner] quantized */
    CUdeviceptr ssm_out_w_f16;  /* optional F16 shadow */
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

    /* Gemma4 per-layer */
    CUdeviceptr post_attn_norm_w;  /* F32 [n_embd] */
    CUdeviceptr post_ffw_norm_w;   /* F32 [n_embd] */
    float layer_scale_val;         /* cached output scale scalar */
    CUdeviceptr ple_inp_gate_w;    /* per-layer embedding gate weight */
    CUdeviceptr ple_proj_w;        /* per-layer embedding projection */
    CUdeviceptr ple_post_norm_w;   /* per-layer embedding post-norm */
    int ple_inp_gate_type, ple_proj_type;
    int ple_inp_gate_rows, ple_inp_gate_cols;
    int ple_proj_rows, ple_proj_cols;
    int is_swa;
    int shared_kv_source; /* -1 = own KV, >= 0 = source layer index */
    int n_kv_heads;       /* per-layer KV head count (may differ from global for Gemma4 31B) */
    int has_v_proj;       /* 0 = V uses K (Gemma4 full-attn layers without V tensor) */
} cuda_layer;

struct cuda_llm_runner {
    /* CUDA context */
    CUdevice device;
    CUcontext context;
    CUstream stream;
    int verbose;
    cublasew_context *cublas;
    int use_cublas;
    int moe_prefill_backend;

    /* Compiled module + kernels */
    CUmodule module;
    CUfunction fn_embed_f16;
    CUfunction fn_rmsnorm_f32;
    CUfunction fn_matvec_f16_f32;
    CUfunction fn_qknorm_f32;
    CUfunction fn_rope_neox_f32;
    CUfunction fn_rope_neox_f32_ptr;
    CUfunction fn_attn_decode_f32_ptr;
    CUfunction fn_kv_cache_store_f16;
    CUfunction fn_kv_cache_store_f16_ptr;
    CUfunction fn_kv_cache_store_q8;
    CUfunction fn_attn_decode_q8;
    CUfunction fn_attn_decode_f32;
    CUfunction fn_silu_mul_f32;
    CUfunction fn_add_f32;
    CUfunction fn_quantize_f32_to_int8;
    CUfunction fn_matvec_q8_0_dp4a;
    CUfunction fn_matvec_q8_0_q8_1_dp4a;
    CUfunction fn_matvec_q8_0_f32;
    CUfunction fn_matvec_q8_0_f32_fused2;
    CUfunction fn_matvec_q8_0_dp4a_fused2;
    CUfunction fn_matvec_f16_f32_fused2;
    CUfunction fn_embed_q8_0;
    CUfunction fn_matvec_q2_K_f32;
    CUfunction fn_quantize_f32_to_q8_1;
    CUfunction fn_matvec_q2_K_q8_1_dp4a;
    CUfunction fn_matvec_q3_K_f32;
    CUfunction fn_matvec_q3_K_q8_1_dp4a;
    CUfunction fn_matvec_q4_K_f32;
    CUfunction fn_matvec_q4_K_q8_1_dp4a;
    CUfunction fn_matvec_q5_K_f32;
    CUfunction fn_matvec_q5_K_q8_1_dp4a;
    CUfunction fn_matvec_q6_K_f32;
    CUfunction fn_matvec_q6_K_q8_1_dp4a;
    CUfunction fn_embed_q2_K;
    CUfunction fn_embed_q4_K;
    CUfunction fn_embed_q4_0;
    /* SSM kernels */
    CUfunction fn_softplus_mul_f32;
    CUfunction fn_sigmoid_inplace_f32;
    CUfunction fn_exp_inplace_f32;
    CUfunction fn_conv1d_depthwise_silu_f32;
    CUfunction fn_l2_norm_heads_f32;
    CUfunction fn_repeat_tile_f32;
    CUfunction fn_deltanet_step_f32;
    CUfunction fn_gated_rmsnorm_silu_f32;
    CUfunction fn_sigmoid_mul_f32;
    CUfunction fn_deinterleave_qgate_f32;
    /* MoE kernels */
    CUfunction fn_scale_add_f32;
    CUfunction fn_batch_scale_add_sigmoid_f32;
    CUfunction fn_matvec_f32_f32;
    CUfunction fn_batch_matvec_f32_f32;
    CUfunction fn_matvec_iq2_xxs_f32;
    CUfunction fn_matvec_iq2_xxs_q8_1_dp4a;
    CUfunction fn_matvec_iq2_xxs_q8_1_dp4a_coal;
    CUfunction fn_matvec_q4_0_f32;
    CUfunction fn_matvec_q4_0_q8_1_dp4a;
    CUfunction fn_matvec_q4_1_f32;
    CUfunction fn_matvec_q5_0_f32;
    CUfunction fn_matvec_q5_1_f32;
    CUfunction fn_matvec_iq4_nl_f32;
    CUfunction fn_matvec_iq4_xs_f32;
    CUfunction fn_matvec_iq2_xs_f32;
    CUfunction fn_matvec_iq3_xxs_f32;
    CUfunction fn_matvec_iq2_s_f32;
    CUfunction fn_matvec_iq3_s_f32;
    CUfunction fn_matvec_iq3_xxs_q8_1_dp4a;
    CUfunction fn_matvec_iq2_s_q8_1_dp4a;
    CUfunction fn_matvec_iq3_s_q8_1_dp4a;
    CUfunction fn_matvec_iq1_s_f32;
    CUfunction fn_matvec_iq1_m_f32;
    CUfunction fn_matvec_tq1_0_f32;
    CUfunction fn_matvec_tq2_0_f32;

    /* Batched prefill kernels */
    CUfunction fn_convert_f32_to_f16;
    CUfunction fn_convert_f16_to_f32;
    CUfunction fn_bf16_to_f16_inplace;
    CUfunction fn_batch_embed_f16;
    CUfunction fn_batch_matvec_q8_0_f32;
    CUfunction fn_batch_matvec_q8_0_x4;
    CUfunction fn_batch_matvec_q2_K;
    CUfunction fn_batch_matvec_q3_K;
    CUfunction fn_batch_matvec_iq2_xxs;
    CUfunction fn_batch_matvec_iq2_xs;
    CUfunction fn_batch_matvec_iq2_s;
    CUfunction fn_batch_matvec_iq4_nl;
    CUfunction fn_batch_matvec_iq4_xs;
    CUfunction fn_moe_topk_kernel;
    CUfunction fn_gather_rows_f32;
    CUfunction fn_scatter_add_weighted_f32;
    CUfunction fn_moe_fused_ffn;
    CUfunction fn_dequant_iq2_s_to_f16;
    CUfunction fn_dequant_iq2_xxs_to_f16;
    CUfunction fn_dequant_iq2_xxs_pair_to_f16;
    CUfunction fn_dequant_iq2_xxs_triplet_to_f16;
    /* IQ2_XXS MMQ (mul_mat_id) path */
    CUfunction fn_mmq_gather_quant_q8_1;
    CUfunction fn_mmq_quant_q8_1;
    CUfunction fn_mmq_iq2xxs_grouped;
    CUfunction fn_mmq_scatter_weighted;
    CUfunction fn_mmq_iq2xxs_grouped8;
    CUfunction fn_mmq_iq2xxs_grouped32;
    CUfunction fn_mmq_iq2xxs_fused32;
    /* IQ3_XXS MMQ (mul_mat_id) path */
    CUfunction fn_mmq_iq3xxs_grouped;
    CUfunction fn_mmq_iq3xxs_grouped8;
    CUfunction fn_mmq_iq3xxs_grouped32;
    CUfunction fn_mmq_iq3xxs_fused32;
    /* Q2_K MMQ (dense prefill) path */
    CUfunction fn_mmq_q2_K_grouped;
    CUfunction fn_mmq_q2_K_grouped8;
    CUfunction fn_mmq_q2_K_grouped32;
    CUfunction fn_mmq_q2_K_fused32;
    /* Q4_K MMQ (dense prefill) path */
    CUfunction fn_mmq_q4_K_grouped;
    CUfunction fn_mmq_q4_K_grouped8;
    CUfunction fn_mmq_q4_K_grouped32;
    /* Q4_0 MMQ (dense prefill) path */
    CUfunction fn_mmq_q4_0_grouped;
    CUfunction fn_mmq_q4_0_grouped8;
    CUfunction fn_mmq_q4_0_grouped32;
    /* Q3_K MMQ (dense prefill) path */
    CUfunction fn_mmq_q3_K_grouped;
    CUfunction fn_mmq_q3_K_grouped8;
    CUfunction fn_mmq_q3_K_grouped32;
    CUfunction fn_mmq_q3_K_fused32;
    /* IQ3_S MMQ (dense prefill) path */
    CUfunction fn_mmq_iq3_s_grouped;
    CUfunction fn_mmq_iq3_s_grouped8;
    CUfunction fn_mmq_iq3_s_grouped32;
    CUfunction fn_mmq_iq3_s_fused32;
    /* IQ2_S MMQ (dense prefill) path */
    CUfunction fn_mmq_iq2_s_grouped;
    CUfunction fn_mmq_iq2_s_grouped8;
    CUfunction fn_mmq_iq2_s_grouped32;
    CUfunction fn_mmq_iq2_s_fused32;
    CUfunction fn_dequant_iq3_xxs_to_f16;
    CUfunction fn_dequant_q2_K_to_f16;
    CUfunction fn_dequant_iq3_s_to_f16;
    CUfunction fn_batch_matvec_iq2_s_tc;
    CUfunction fn_batch_matvec_iq3_xxs;
    CUfunction fn_batch_matvec_iq3_s;
    CUfunction fn_batch_matvec_q4_K;
    CUfunction fn_batch_matvec_q4_K_x4;
    CUfunction fn_batch_matvec_q5_K;
    CUfunction fn_batch_matvec_q6_K;
    CUfunction fn_batch_matvec_q6_K_x4;
    CUfunction fn_batch_softplus_mul_f32;
    CUfunction fn_batch_rope_neox_f32;
    CUfunction fn_batch_conv1d_depthwise_silu_f32;
    CUfunction fn_batch_deltanet_scan_f32;
    CUfunction fn_batch_l2_norm_heads_strided_f32;
    CUfunction fn_batch_attn_causal_f32;
    CUfunction fn_batch_attn_all_tokens_f32;
    CUfunction fn_causal_softmax_f32;
    CUfunction fn_batch_kv_store_f16;
    CUfunction fn_batch_kv_store_swa_f16;
    CUfunction fn_batch_rmsnorm_f32;

    /* Vision encoder kernels */
    CUfunction fn_vision_linear_f16;
    CUfunction fn_vision_rmsnorm;
    CUfunction fn_vision_head_rmsnorm;
    CUfunction fn_vision_rope_2d;
    CUfunction fn_vision_attention;
    CUfunction fn_vision_gelu_gate;
    CUfunction fn_vision_add;
    CUfunction fn_vision_scale;
    CUfunction fn_vision_add_pos_embd;

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
    int hidden_snapshot_layers[3];
    int n_hidden_snapshots;

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
    int moe_iq2_bm;       /* IQ2_XXS expert weights repacked block-major (CUDA_LLM_MMQ_REPACK) */
    int dense_bm;         /* Dense weights repacked block-major (CUDA_LLM_DENSE_BM) */
    int n_experts;
    int n_experts_used;   /* top-k */
    int expert_ff;        /* per-expert FFN dim */
    int shared_expert_ff; /* shared expert FFN dim */

    /* Gemma4 params */
    int is_gemma4;
    int head_dim_full, head_dim_swa;
    int swa_window_size;
    int n_layer_kv_from_start;
    int n_embd_per_layer;
    float final_logit_softcapping;
    float rope_freq_base_swa;
    float embd_scale;
    int *swa_pattern;               /* [n_layers] host array: 1=SWA, 0=full */
    int *per_layer_kv_heads;        /* [n_layers] per-layer KV head count (NULL = use global n_kv_heads) */
    CUdeviceptr d_rope_inv_freq_full; /* F32 [head_dim_full/2] with proportional factors */
    CUdeviceptr d_rope_inv_freq_swa;  /* F32 [head_dim_swa/2] */
    /* Per-layer embedding globals on GPU */
    CUdeviceptr d_ple_combined;     /* F32 [n_layers * ple_dim] precomputed per token */
    CUdeviceptr d_ple_buf;          /* F32 [ple_dim] scratch */
    CUdeviceptr d_ple_proj;         /* F32 [n_embd] scratch */
    int current_token_id;           /* stashed for PLE lookup */
    int ple_use_f32;                /* 1 = keep PLE weights in F32 for accuracy */
    /* AOT-compiled GPU kernels (moe_gpu_kernels.cubin, no device globals) */
    CUmodule moe_gpu_mod;
    /* FA2 Flash Attention NVRTC module (separately compiled for D=256 and D=512) */
    CUmodule fa2_mod;
    CUfunction fn_fa2_attn_256;
    CUfunction fn_fa2_attn_512;
    CUfunction fn_moe_topk_gpu;
    CUfunction fn_moe_shared_gate_gpu;
    CUfunction fn_moe_iq2s_tc;
    CUfunction fn_moe_iq3s_tc;
    CUfunction fn_moe_accum_gpu;
    CUfunction fn_moe_f16_tc;
    CUfunction fn_moe_prefill_q4k;
    CUfunction fn_dequant_q4_K_to_f16;
    CUfunction fn_dequant_q4_0_to_f16;
    CUfunction fn_dequant_q8_0_to_f16;
    CUfunction fn_dequant_q8_0_to_f16_h;
    CUfunction fn_dequant_q6_K_to_f16;
    CUfunction fn_moe_expert_fused_q4k;
    CUdeviceptr d_topk_idx;   /* [n_used * max_tokens] int */
    CUdeviceptr d_topk_wgt;   /* [n_used * max_tokens] float */
    int d_topk_idx_entries;   /* allocated entries in d_topk_idx/wgt */
    int h_topk_idx[8];
    float h_topk_wgt[8];
    CUdeviceptr d_grid_ksigns;  /* grid tables for cubin kernels */
    CUdeviceptr d_grid_iq2s;
    CUdeviceptr d_grid_iq3;   /* iq3xxs_grid (256) — not used, keep for compat */
    CUdeviceptr d_grid_iq3s;  /* iq3s_grid (512) — correct for IQ3_S down */
    /* Device buffers for graph-captured decode (position, seq_len — updated each step) */
    CUdeviceptr d_pos_seq;    /* [2] int32: [position, seq_len] */
    CUdeviceptr d_seq_ptr;    /* points to d_pos_seq[1] (seq_len) — persistent for graph capture */
    CUgraph d_graph;
    CUgraphExec d_graph_exec;
    int graph_captured;
    int disable_graph;
    /* Host-side PLE data (mmap pointers from GGUF, valid while GGUF is open) */
    qtensor h_ple_token_embd;       /* Q8_0 per_layer_token_embd */
    qtensor h_ple_model_proj;       /* BF16 per_layer_model_proj */
    qtensor h_ple_proj_norm;        /* F32 per_layer_proj_norm */
    /* Gemma4 kernel functions */
    CUfunction fn_raw_rmsnorm_heads_f32;
    CUfunction fn_gelu_mul_f32;
    CUfunction fn_gelu_elementwise_mul_f32;
    CUfunction fn_scale_f32;
    CUfunction fn_logit_softcap_f32;
    CUfunction fn_attn_decode_swa_f32;
    CUfunction fn_matvec_bf16_f32;
    CUfunction fn_rope_with_factors_f32;
    CUfunction fn_rope_with_factors_f32_ptr;
    CUfunction fn_attn_decode_swa_f32_ptr;
    CUfunction fn_batch_rope_with_factors_f32;

    /* MoE scratch buffers */
    CUdeviceptr d_router_logits;  /* [d_router_logits_entries] F32 */
    int d_router_logits_entries;
    CUdeviceptr d_moe_accum;      /* [n_embd] F32 */
    CUdeviceptr d_moe_f16w;       /* [expert_ff * n_embd] F16 transient dequant scratch */
    CUdeviceptr d_moe_f16w2;      /* second F16 scratch for paired gate/up dequant */
    CUdeviceptr d_moe_f16w3;      /* third F16 scratch for triplet gate/up/down dequant */
    /* IQ2_XXS MMQ (mul_mat_id) scratch — lazily allocated, sized for compacted rows */
    CUdeviceptr d_mmq_cxq8;       /* int8 [total_rows * max(n_embd,expert_ff)] q8_1 activations */
    CUdeviceptr d_mmq_cxs;        /* f32  [total_rows * max(n_embd,expert_ff)/32] q8_1 scales */
    CUdeviceptr d_mmq_outg;       /* f32  [total_rows * expert_ff] gate out */
    CUdeviceptr d_mmq_outu;       /* f32  [total_rows * expert_ff] up out */
    CUdeviceptr d_mmq_outd;       /* f32  [total_rows * n_embd] down out */
    CUdeviceptr d_mmq_ebounds;    /* int  [n_experts+1] expert bounds */
    CUdeviceptr d_mmq_worklist;   /* int  [<= n_experts*ceil(maxtok/32)] packed (e<<16)|group */
    int d_mmq_alloc_rows;         /* allocated compact-row capacity (0 = unallocated) */
    /* Dense IQ2_XXS MMQ prefill scratch (non-MoE; tensor-core int8, weights read once) */
    CUdeviceptr d_mmqd_cxq8;      /* int8 [rows*K] q8_1 activations */
    CUdeviceptr d_mmqd_cxs;       /* f32  [rows*K/32] q8_1 scales */
    CUdeviceptr d_mmqd_eb;        /* int  [2] = {0, n_tokens} */
    CUdeviceptr d_mmqd_wl;        /* int  worklist */
    size_t d_mmqd_cap;            /* allocated cxq8 capacity in bytes (0 = unallocated) */
    int d_mmqd_wl_ntok;           /* n_tokens the cached worklist/eb were built for */
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
    /* INT8 quantized KV cache (opt-in, CUDA_LLM_KV_CACHE_Q8) */
    CUdeviceptr *d_key_cache_q8;  /* [n_layers] -> [max_seq_len * kv_dim] INT8 */
    CUdeviceptr *d_value_cache_q8;
    CUdeviceptr *d_key_scale;     /* [n_layers] -> [max_seq_len * n_kv_heads] F32 */
    CUdeviceptr *d_value_scale;
    int kv_cache_q8;              /* flag: use INT8 KV cache */

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
    CUdeviceptr d_xb_q2;    /* INT8 [max_dim] second buffer for FFN down / ATT output */
    CUdeviceptr d_xb_scale2;/* F32 [1] */
    /* Q8_1 quantization scratch (for K-quant dp4a path) */
    CUdeviceptr d_xb_q81;   /* Q8_1 blocks [ceil(max_dim/32)*36 bytes] */
    CUdeviceptr d_xb_q81_2; /* Q8_1 blocks [ceil(max_dim/32)*36 bytes] second buffer */
    int use_dp4a;            /* 1 = use dp4a INT8 path for Q8_0 matvecs */
    CUdeviceptr d_hidden_snapshots; /* optional packed [3 * n_embd] snapshot buffer */

    /* Pre-allocated batch prefill buffers (allocated on first use, reused) */
    int batch_buf_max_tokens;   /* max tokens these buffers can handle */
    CUdeviceptr d_batch_x;      /* [max_tokens * n_embd] */
    CUdeviceptr d_batch_xb;     /* [max_tokens * n_embd] */
    CUdeviceptr d_batch_wide;   /* [max_tokens * wide_dim] */
    CUdeviceptr d_batch_q;      /* [max_tokens * q_dim] */
    CUdeviceptr d_batch_mid;    /* [max_tokens * mid_dim] */
    CUdeviceptr d_batch_k;      /* [max_tokens * kv_dim] */
    CUdeviceptr d_batch_v;      /* [max_tokens * kv_dim] */
    CUdeviceptr d_batch_ff1;    /* [max_tokens * n_ff] */
    CUdeviceptr d_batch_ff2;    /* [max_tokens * n_ff] */
    CUdeviceptr d_batch_alpha;  /* [max_tokens * dt_rank] */
    CUdeviceptr d_batch_beta;   /* [max_tokens * dt_rank] */
    CUdeviceptr d_batch_token_ids; /* [max_tokens] int32 */
    CUdeviceptr d_fa2_q_f16; /* [max_tokens * max_dim_h] F16 for FA2 Q input */
    CUdeviceptr d_fa2_o_f16; /* [max_tokens * max_dim_h] F16 for FA2 O output / d512 F32 scores */
    CUdeviceptr d_fa2_scores_f16; /* [sq * total_s] F16 score matrix for cuBLAS-d512 attention */
    int fa2_buf_max_s; /* token capacity the d_fa2_* buffers were sized for */
    size_t d512_cap_elems; /* effective per-chunk score-matrix cap (elems) the d_fa2_* buffers were sized for; the d512 attn loop must use this exact value to chunk so it never overflows the buffer */
    CUdeviceptr d_swa_k_lin; /* [n_tokens * kv_dim] F16 linear K for batched windowed SWA prefill */
    CUdeviceptr d_swa_v_lin; /* [n_tokens * kv_dim] F16 linear V */
    int swa_lin_max_tokens;  /* token capacity the d_swa_*_lin buffers were sized for */
    CUdeviceptr d_f16_scratch; /* [max_out * max_in] F16 dequant scratch for Q8_0 GEMM */
    CUdeviceptr d_f16_scratch2; /* second weight-dequant buffer for double-buffered overlap */
    CUstream stream_dq;        /* weight dequant runs here, overlapping GEMM on the main stream */
    CUevent dq_evt[2];         /* dequant-done events (ping-pong) */
    CUevent gemm_evt[2];       /* gemm-done events guarding buffer reuse (ping-pong) */
    int dq_pp;                 /* ping-pong index */
    int prefill_overlap;       /* 1 = double-buffer weight dequant vs GEMM */
    CUevent gate_done_evt;     /* CUDA_LLM_PAR_FFN: gate matvec done on main stream */
    CUdeviceptr d_batch_f16_scratch; /* [max_tokens * max_dim] F16 for cuBLAS input conversion */
    size_t batch_f16_scratch_bytes; /* current byte capacity of d_batch_f16_scratch */

    /* Host output buffer */
    float *h_output;     /* [n_embd] or [n_vocab] for logits */
    CUdeviceptr d_logits; /* [n_vocab] GPU logits buffer */
    void *h_stage;
    size_t h_stage_bytes;

    /* Weight loading state */
    int weights_loaded;

    /* Deepstack injection state (set during forward_embd, NULL otherwise) */
    const float *_ds_embd;
    int _ds_embd_stride;
};

static int cuda_llm_bind_context(cuda_llm_runner *r) {
    if (!r) return -1;
    CUresult err = cuCtxSetCurrent(r->context);
    if (err != CUDA_SUCCESS) {
        if (r->verbose >= 1) {
            fprintf(stderr, "cuda_llm: cuCtxSetCurrent failed (%d)\n", (int)err);
        }
        return -1;
    }
    return 0;
}

/* ======================================================================== */
/* NVRTC kernel compilation                                                 */
/* ======================================================================== */

#define CLLM_PTX_CACHE_VERSION 25

static uint64_t cllm_kernel_source_hash(const char *s) {
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static int cllm_read_file(const char *path, char **out_data, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return -1; }
    long sz = ftell(fp);
    if (sz <= 0) { fclose(fp); return -1; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return -1; }
    char *buf = (char *)malloc((size_t)sz);
    if (!buf) { fclose(fp); return -1; }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *out_data = buf;
    *out_size = (size_t)sz;
    return 0;
}

/* Open a cubin by leaf name, robust to the current working directory. The cubins
 * live next to the binary (cuda/llm/). A hardcoded relative path silently fails
 * when the binary is run from any other directory -> FA2/moe kernels don't load
 * -> silent fall back to slow per-token attention / per-token matvec (a 5x prefill
 * cliff). Try, in order: next to the executable (/proc/self/exe dir), the repo-root
 * relative path "cuda/llm/<leaf>", and the bare leaf (cwd == cuda/llm). Returns an
 * open FILE* (caller closes) and writes the path used into used_path (optional). */
static FILE *cllm_fopen_cubin(const char *leaf, const char **used_path) {
    static char exe_path[4096];
    /* 1) directory of the running executable */
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n > 0) {
        exe_path[n] = '\0';
        char *slash = strrchr(exe_path, '/');
        if (slash) {
            size_t dlen = (size_t)(slash - exe_path) + 1;
            if (dlen + strlen(leaf) < sizeof(exe_path)) {
                memcpy(slash + 1, leaf, strlen(leaf) + 1);
                FILE *fp = fopen(exe_path, "rb");
                if (fp) { if (used_path) *used_path = exe_path; return fp; }
            }
        }
    }
    /* 2) repo-root relative (legacy default) */
    {
        static char rp[4096];
        snprintf(rp, sizeof(rp), "cuda/llm/%s", leaf);
        FILE *fp = fopen(rp, "rb");
        if (fp) { if (used_path) *used_path = rp; return fp; }
    }
    /* 3) bare leaf (cwd already inside cuda/llm) */
    {
        FILE *fp = fopen(leaf, "rb");
        if (fp) { if (used_path) *used_path = leaf; return fp; }
    }
    if (used_path) *used_path = leaf;
    return NULL;
}

static void cllm_write_file(const char *path, const char *data, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fwrite(data, 1, size, fp);
    fclose(fp);
}

static int compile_kernels(cuda_llm_runner *r) {
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, r->device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, r->device);
    int sm = major * 10 + minor;
    char cache_path[128];
    uint64_t source_hash = cllm_kernel_source_hash(cuda_kernel_source);
    snprintf(cache_path, sizeof(cache_path), "/tmp/cuda_llm_sm_%d_v%d_%016llx.ptx",
             sm, CLLM_PTX_CACHE_VERSION, (unsigned long long)source_hash);

    char *ptx = NULL;
    size_t ptx_size = 0;
    if (cllm_read_file(cache_path, &ptx, &ptx_size) == 0) {
        if (r->verbose >= 1) {
            fprintf(stderr, "cuda_llm: loading cached PTX %s\n", cache_path);
        }
        CUresult err = cuModuleLoadDataEx(&r->module, ptx, 0, NULL, NULL);
        free(ptx);
        if (err == CUDA_SUCCESS) goto lookup_funcs;
        if (r->verbose >= 1) {
            const char *errStr = NULL;
            cuGetErrorString(err, &errStr);
            fprintf(stderr, "cuda_llm: cached PTX load failed (%s), recompiling\n",
                    errStr ? errStr : "unknown");
        }
    }

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
    ptx_size = 0;
    nvrtcGetPTXSize(prog, &ptx_size);
    ptx = (char *)malloc(ptx_size);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    if (r->verbose >= 2) {
        fprintf(stderr, "cuda_llm: PTX size = %zu bytes\n", ptx_size);
    }

    /* Load module from PTX */
    CUresult err = cuModuleLoadDataEx(&r->module, ptx, 0, NULL, NULL);
    if (err == CUDA_SUCCESS) cllm_write_file(cache_path, ptx, ptx_size);
    free(ptx);
    if (err != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "cuda_llm: cuModuleLoadDataEx failed: %s\n", errStr);
        return -1;
    }

    /* Look up kernel functions */
lookup_funcs:
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
    GET_FUNC(rope_neox_f32_ptr);
    GET_FUNC(attn_decode_f32_ptr);
    GET_FUNC(kv_cache_store_f16);
    GET_FUNC(kv_cache_store_f16_ptr);
    GET_FUNC(kv_cache_store_q8);
    GET_FUNC(attn_decode_q8);
    GET_FUNC(attn_decode_f32);
    GET_FUNC(silu_mul_f32);
    GET_FUNC(add_f32);
    GET_FUNC(quantize_f32_to_int8);
    GET_FUNC(matvec_q8_0_dp4a);
    GET_FUNC(matvec_q8_0_q8_1_dp4a);
    GET_FUNC(matvec_q8_0_f32);
    GET_FUNC(matvec_q8_0_f32_fused2);
    GET_FUNC(matvec_q8_0_dp4a_fused2);
    GET_FUNC(matvec_f16_f32_fused2);
    GET_FUNC(embed_q8_0);
    GET_FUNC(matvec_q2_K_f32);
    GET_FUNC(quantize_f32_to_q8_1);
    GET_FUNC(matvec_q2_K_q8_1_dp4a);
    GET_FUNC(matvec_q3_K_f32);
    GET_FUNC(matvec_q3_K_q8_1_dp4a);
    GET_FUNC(matvec_q4_K_f32);
    GET_FUNC(matvec_q4_K_q8_1_dp4a);
    GET_FUNC(matvec_q5_K_f32);
    GET_FUNC(matvec_q5_K_q8_1_dp4a);
    GET_FUNC(matvec_q6_K_f32);
    GET_FUNC(matvec_q6_K_q8_1_dp4a);
    GET_FUNC(embed_q2_K);
    GET_FUNC(embed_q4_K);
    GET_FUNC(embed_q4_0);
    /* SSM kernels */
    GET_FUNC(softplus_mul_f32);
    GET_FUNC(sigmoid_inplace_f32);
    GET_FUNC(exp_inplace_f32);
    GET_FUNC(conv1d_depthwise_silu_f32);
    GET_FUNC(l2_norm_heads_f32);
    GET_FUNC(repeat_tile_f32);
    GET_FUNC(deltanet_step_f32);
    GET_FUNC(gated_rmsnorm_silu_f32);
    GET_FUNC(sigmoid_mul_f32);
    GET_FUNC(deinterleave_qgate_f32);
    /* MoE kernels */
    GET_FUNC(scale_add_f32);
    GET_FUNC(batch_scale_add_sigmoid_f32);
    GET_FUNC(matvec_f32_f32);
    GET_FUNC(batch_matvec_f32_f32);
    GET_FUNC(matvec_iq2_xxs_f32);
    GET_FUNC(matvec_iq2_xxs_q8_1_dp4a);
    GET_FUNC(matvec_iq2_xxs_q8_1_dp4a_coal);
    GET_FUNC(matvec_q4_0_f32);
    GET_FUNC(matvec_q4_0_q8_1_dp4a);
    GET_FUNC(matvec_q4_1_f32);
    GET_FUNC(matvec_q5_0_f32);
    GET_FUNC(matvec_q5_1_f32);
    GET_FUNC(matvec_iq4_nl_f32);
    GET_FUNC(matvec_iq4_xs_f32);
    GET_FUNC(matvec_iq2_xs_f32);
    GET_FUNC(matvec_iq3_xxs_f32);
    GET_FUNC(matvec_iq2_s_f32);
    GET_FUNC(matvec_iq3_s_f32);
    GET_FUNC(matvec_iq3_xxs_q8_1_dp4a);
    GET_FUNC(matvec_iq2_s_q8_1_dp4a);
    GET_FUNC(matvec_iq3_s_q8_1_dp4a);
    GET_FUNC(matvec_iq1_s_f32);
    GET_FUNC(matvec_iq1_m_f32);
    GET_FUNC(matvec_tq1_0_f32);
    GET_FUNC(matvec_tq2_0_f32);
    GET_FUNC(raw_rmsnorm_heads_f32);
    /* Gemma4 kernels */
    GET_FUNC(gelu_mul_f32);
    GET_FUNC(gelu_elementwise_mul_f32);
    GET_FUNC(scale_f32);
    GET_FUNC(logit_softcap_f32);
    GET_FUNC(attn_decode_swa_f32);
    GET_FUNC(matvec_bf16_f32);
    GET_FUNC(rope_with_factors_f32);
    GET_FUNC(rope_with_factors_f32_ptr);
    GET_FUNC(attn_decode_swa_f32_ptr);
    GET_FUNC(batch_rope_with_factors_f32);
    /* Batched prefill kernels */
    GET_FUNC(convert_f32_to_f16);
    GET_FUNC(convert_f16_to_f32);
    GET_FUNC(bf16_to_f16_inplace);
    GET_FUNC(batch_embed_f16);
    GET_FUNC(batch_matvec_q8_0_f32);
    GET_FUNC(batch_matvec_q8_0_x4);
    GET_FUNC(batch_matvec_q2_K);
    GET_FUNC(batch_matvec_q3_K);
    GET_FUNC(batch_matvec_iq2_xxs);
    GET_FUNC(batch_matvec_iq2_xs);
    GET_FUNC(batch_matvec_iq2_s);
    GET_FUNC(batch_matvec_iq4_nl);
    GET_FUNC(batch_matvec_iq4_xs);
    GET_FUNC(moe_topk_kernel);
    GET_FUNC(gather_rows_f32);
    GET_FUNC(scatter_add_weighted_f32);
    GET_FUNC(moe_fused_ffn);
    GET_FUNC(dequant_iq2_s_to_f16);
    GET_FUNC(dequant_iq2_xxs_to_f16);
    GET_FUNC(dequant_iq2_xxs_pair_to_f16);
    GET_FUNC(dequant_iq2_xxs_triplet_to_f16);
    GET_FUNC(mmq_gather_quant_q8_1);
    GET_FUNC(mmq_quant_q8_1);
    GET_FUNC(mmq_iq2xxs_grouped);
    GET_FUNC(mmq_scatter_weighted);
    GET_FUNC(mmq_iq2xxs_grouped8);
    GET_FUNC(mmq_iq2xxs_grouped32);
    GET_FUNC(mmq_iq2xxs_fused32);
    GET_FUNC(mmq_iq3xxs_grouped);
    GET_FUNC(mmq_iq3xxs_grouped8);
    GET_FUNC(mmq_iq3xxs_grouped32);
    GET_FUNC(mmq_iq3xxs_fused32);
    GET_FUNC(mmq_q2_K_grouped);
    GET_FUNC(mmq_q2_K_grouped8);
    GET_FUNC(mmq_q2_K_grouped32);
    GET_FUNC(mmq_q2_K_fused32);
    GET_FUNC(mmq_q4_K_grouped);
    GET_FUNC(mmq_q4_K_grouped8);
    GET_FUNC(mmq_q4_K_grouped32);
    GET_FUNC(mmq_q4_0_grouped);
    GET_FUNC(mmq_q4_0_grouped8);
    GET_FUNC(mmq_q4_0_grouped32);
    GET_FUNC(mmq_q3_K_grouped);
    GET_FUNC(mmq_q3_K_grouped8);
    GET_FUNC(mmq_q3_K_grouped32);
    GET_FUNC(mmq_q3_K_fused32);
    GET_FUNC(mmq_iq3_s_grouped);
    GET_FUNC(mmq_iq3_s_grouped8);
    GET_FUNC(mmq_iq3_s_grouped32);
    GET_FUNC(mmq_iq3_s_fused32);
    GET_FUNC(mmq_iq2_s_grouped);
    GET_FUNC(mmq_iq2_s_grouped8);
    GET_FUNC(mmq_iq2_s_grouped32);
    GET_FUNC(mmq_iq2_s_fused32);
    GET_FUNC(dequant_iq3_xxs_to_f16);
    GET_FUNC(dequant_q2_K_to_f16);
    GET_FUNC(dequant_iq3_s_to_f16);
    GET_FUNC(batch_matvec_iq2_s_tc);
    GET_FUNC(batch_matvec_iq3_xxs);
    GET_FUNC(batch_matvec_iq3_s);
    GET_FUNC(batch_matvec_q4_K);
    GET_FUNC(batch_matvec_q4_K_x4);
    GET_FUNC(batch_matvec_q5_K);
    GET_FUNC(batch_matvec_q6_K);
    GET_FUNC(batch_matvec_q6_K_x4);
    GET_FUNC(batch_softplus_mul_f32);
    GET_FUNC(batch_rope_neox_f32);
    GET_FUNC(batch_conv1d_depthwise_silu_f32);
    GET_FUNC(batch_deltanet_scan_f32);
    GET_FUNC(batch_l2_norm_heads_strided_f32);
    GET_FUNC(batch_attn_causal_f32);
    GET_FUNC(batch_attn_all_tokens_f32);
    GET_FUNC(causal_softmax_f32);
    GET_FUNC(batch_kv_store_f16);
    GET_FUNC(batch_kv_store_swa_f16);
    GET_FUNC(batch_rmsnorm_f32);
    /* Vision encoder kernels */
    GET_FUNC(vision_linear_f16);
    GET_FUNC(vision_rmsnorm);
    GET_FUNC(vision_head_rmsnorm);
    GET_FUNC(vision_rope_2d);
    GET_FUNC(vision_attention);
    GET_FUNC(vision_gelu_gate);
    GET_FUNC(vision_add);
    GET_FUNC(vision_scale);
    GET_FUNC(vision_add_pos_embd);

    #undef GET_FUNC

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: all kernels compiled successfully\n");
    }

    r->moe_gpu_mod = 0; r->fn_moe_topk_gpu = NULL; r->fn_moe_shared_gate_gpu = NULL;
    {
        const char *cp = "cuda/llm/moe_gpu_kernels.cubin";
        FILE *fp = cllm_fopen_cubin("moe_gpu_kernels.cubin", &cp);
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            char *d = (char *)malloc((size_t)sz);
            if (d) {
                size_t nr = fread(d, 1, (size_t)sz, fp);
                fclose(fp);
                if (nr == (size_t)sz && cuModuleLoadData(&r->moe_gpu_mod, d) == CUDA_SUCCESS) {
#define CLLM_LOAD_OPT_FUNC(dst, name) do { \
                        CUfunction _fn = NULL; \
                        if (cuModuleGetFunction(&_fn, r->moe_gpu_mod, (name)) == CUDA_SUCCESS) (dst) = _fn; \
                        else (dst) = NULL; \
                    } while (0)
                    CLLM_LOAD_OPT_FUNC(r->fn_moe_topk_gpu, "moe_topk_gpu");
                    CLLM_LOAD_OPT_FUNC(r->fn_moe_shared_gate_gpu, "moe_shared_gate_gpu");
                    r->fn_moe_iq2s_tc = NULL;
                    r->fn_moe_iq3s_tc = NULL;
                    CLLM_LOAD_OPT_FUNC(r->fn_moe_accum_gpu, "moe_accum_gpu");
                    CLLM_LOAD_OPT_FUNC(r->fn_moe_f16_tc, "moe_f16_tc");
                    CLLM_LOAD_OPT_FUNC(r->fn_moe_prefill_q4k, "moe_prefill_q4k");
                    CLLM_LOAD_OPT_FUNC(r->fn_dequant_q4_K_to_f16, "dequant_q4_K_to_f16");
                    CLLM_LOAD_OPT_FUNC(r->fn_dequant_q4_0_to_f16, "dequant_q4_0_to_f16");
                    CLLM_LOAD_OPT_FUNC(r->fn_dequant_q8_0_to_f16, "dequant_q8_0_to_f16");
                    CLLM_LOAD_OPT_FUNC(r->fn_dequant_q8_0_to_f16_h, "dequant_q8_0_to_f16_h");
                    CLLM_LOAD_OPT_FUNC(r->fn_dequant_q6_K_to_f16, "dequant_q6_K_to_f16");
                    CLLM_LOAD_OPT_FUNC(r->fn_moe_expert_fused_q4k, "moe_expert_fused_q4k");
#undef CLLM_LOAD_OPT_FUNC
                    r->d_grid_ksigns = 0; r->d_grid_iq2s = 0; r->d_grid_iq3 = 0;
                    if (r->verbose >= 1) fprintf(stderr, "cuda_llm: loaded %s\n", cp);
                } else if (r->verbose >= 1) {
                    fprintf(stderr, "cuda_llm: failed to load %s\n", cp);
                }
                free(d);
            } else {
                fclose(fp);
            }
        } else if (r->verbose >= 1) {
            fprintf(stderr, "cuda_llm: %s not found; AOT MoE kernels disabled\n", cp);
        }
    }
    /* Load FA2 flash attention cubin */
    r->fa2_mod = 0; r->fn_fa2_attn_256 = NULL; r->fn_fa2_attn_512 = NULL;
    {
        const char *cp = "cuda/llm/fa2_kernels.cubin";
        FILE *fp = cllm_fopen_cubin("fa2_kernels.cubin", &cp);
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            char *d = (char *)malloc((size_t)sz);
            if (d) {
                size_t nr = fread(d, 1, (size_t)sz, fp);
                fclose(fp);
                if (nr == (size_t)sz && cuModuleLoadDataEx(&r->fa2_mod, d, 0, NULL, NULL) == CUDA_SUCCESS) {
                    cuModuleGetFunction(&r->fn_fa2_attn_256, r->fa2_mod, "fa2_attn_f");
                    cuModuleGetFunction(&r->fn_fa2_attn_512, r->fa2_mod, "fa2_attn_d512_f");
                    /* fa2_attn_f uses 4*32*(head_dim+8)*sizeof(f16) dynamic smem
                     * (= 67584 B for head_dim=256), which exceeds the 48 KB default.
                     * Opt in to the larger dynamic-smem limit or the launch silently
                     * fails (invalid argument) -> stale attention output. */
                    if (r->fn_fa2_attn_256) {
                        CUresult _sa = cuFuncSetAttribute(r->fn_fa2_attn_256,
                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 96*1024);
                        if (_sa != CUDA_SUCCESS && r->verbose >= 1)
                            fprintf(stderr, "cuda_llm: FA2 d=256 smem opt-in failed (err=%d)\n", (int)_sa);
                    }
                    /* fa2_attn_d512_f uses 4*16*(512+8)*sizeof(f16) = 66560 B
                     * dynamic smem (BC=16, head_dim=512). Opt in to 99 KB. */
                    if (r->fn_fa2_attn_512) {
                        CUresult _sa = cuFuncSetAttribute(r->fn_fa2_attn_512,
                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 99*1024);
                        if (_sa != CUDA_SUCCESS && r->verbose >= 1)
                            fprintf(stderr, "cuda_llm: FA2 d=512 smem opt-in failed (err=%d)\n", (int)_sa);
                    }
                    if (r->verbose >= 1)
                        fprintf(stderr, "cuda_llm: loaded %s (FA2 flash attention: %s%s%s)\n", cp,
                            r->fn_fa2_attn_256 ? "d=256" : "",
                            (r->fn_fa2_attn_256 && r->fn_fa2_attn_512) ? "," : "",
                            r->fn_fa2_attn_512 ? "d=512" : "");
                } else if (r->verbose >= 1) {
                    fprintf(stderr, "cuda_llm: failed to load %s\n", cp);
                }
                free(d);
            } else {
                fclose(fp);
            }
        } else if (r->verbose >= 1) {
            fprintf(stderr, "cuda_llm: %s not found; FA2 flash attention disabled\n", cp);
        }
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
    CHECK_CU_NULL(cuDevicePrimaryCtxRetain(&r->context, r->device));
    CHECK_CU_NULL(cuCtxSetCurrent(r->context));
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
        free(r);
        return NULL;
    }

    /* Enable cuBLAS for verification/baseline when CUDA_LLM_USE_CUBLAS=1.
     * Handle is created lazily on first use to avoid CUDA context issues. */
    {
        const char *env = getenv("CUDA_LLM_USE_CUBLAS");
        r->use_cublas = (env && atoi(env) > 0) ? 1 : 0;
        if (r->use_cublas && verbose >= 1) {
            fprintf(stderr, "cuda_llm: cuBLAS enabled (fully lazy, set CUDA_LLM_USE_CUBLAS=1)\n");
        }
    }

    {
        const char *env = getenv("CUDA_LLM_MOE_PREFILL_BACKEND");
        r->moe_prefill_backend = CLLM_MOE_PREFILL_AUTO;
        if (env && strcmp(env, "cublas") == 0) {
            r->moe_prefill_backend = CLLM_MOE_PREFILL_CUBLAS;
        } else if (env && strcmp(env, "fused") == 0) {
            r->moe_prefill_backend = CLLM_MOE_PREFILL_FUSED;
        } else if (env && strcmp(env, "exact") == 0) {
            r->moe_prefill_backend = CLLM_MOE_PREFILL_EXACT;
        } else if (env && strcmp(env, "auto") != 0 && verbose >= 1) {
            fprintf(stderr, "cuda_llm: unknown CUDA_LLM_MOE_PREFILL_BACKEND='%s' (using auto)\n", env);
        }
        if (verbose >= 1) {
            const char *name = r->moe_prefill_backend == CLLM_MOE_PREFILL_CUBLAS ? "cublas" :
                               r->moe_prefill_backend == CLLM_MOE_PREFILL_FUSED ? "fused" :
                               r->moe_prefill_backend == CLLM_MOE_PREFILL_EXACT ? "exact" : "auto";
            fprintf(stderr, "cuda_llm: MoE prefill backend=%s\n", name);
        }
    }

    /* Enable dp4a INT8 path for sm >= 6.1 (Pascal+) unless disabled */
    {
        int major = 0, minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, r->device);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, r->device);
        int sm = major * 10 + minor;
        r->use_dp4a = (sm >= 61 && !getenv("CUDA_LLM_NO_DP4A"));
        if (r->use_dp4a && verbose >= 1)
            fprintf(stderr, "cuda_llm: dp4a INT8 matvec enabled (sm_%d)\n", sm);
    }

    r->disable_graph = getenv("CUDA_LLM_NO_GRAPH") ? 1 : 0;
    if (r->disable_graph && verbose >= 1)
        fprintf(stderr, "cuda_llm: CUDA graph capture disabled (CUDA_LLM_NO_GRAPH=1)\n");

    return r;
}

/* ======================================================================== */
/* Weight upload helpers                                                    */
/* ======================================================================== */

/* Local safetensors-only FP8 type. Uploaded weights are expanded to F16, so
 * runtime dispatch still sees GGML_TYPE_F16. */
#define CLLM_TYPE_F8_E4M3 200

/* Upload a qtensor as F16 to GPU (for F16 weights, direct copy; others dequant to F32 then... */
/* For this model (Qwen3-Embedding-0.6B-f16), all weight matrices are F16. */
/* Norms are F32. We handle both cases. */

static void *cllm_stage_buf(cuda_llm_runner *r, size_t nbytes) {
    if (!r) return NULL;
    if (nbytes <= r->h_stage_bytes) return r->h_stage;
    void *p = realloc(r->h_stage, nbytes);
    if (!p) return NULL;
    r->h_stage = p;
    r->h_stage_bytes = nbytes;
    return p;
}

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
static int upload_norm_f32(cuda_llm_runner *r, CUdeviceptr *d_ptr, const qtensor *t, int n) {
    if (!t->data) { *d_ptr = 0; return 0; }
    const void *src = t->data;
    if (t->type != GGML_TYPE_F32) {
        float *buf = (float *)cllm_stage_buf(r, (size_t)n * sizeof(float));
        if (!buf) { fprintf(stderr, "cuda_llm: upload_norm_f32 stage alloc failed (n=%d)\n", n); return -1; }
        dequant_row(t->type, t->data, buf, n);
        src = buf;
    }
    CUresult err = cuMemAlloc(d_ptr, n * sizeof(float));
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_norm_f32 alloc failed (n=%d, err=%d)\n", n, (int)err); return -1; }
    err = cuMemcpyHtoD(*d_ptr, src, n * sizeof(float));
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_norm_f32 copy failed\n"); cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}

/* Upload Q8_0 tensor data to GPU with padding for alignment.
 * Each 34-byte Q8_0 block (2B scale + 32B qs) is padded to 36 bytes
 * (2B scale + 2B pad + 32B qs) so int32 reads of qs data are 4-byte aligned. */
static uint16_t cllm_f32_to_f16(float f);

static int upload_q8_0_raw(cuda_llm_runner *r, CUdeviceptr *d_ptr, const qtensor *t) {
    if (!t->data) { *d_ptr = 0; return 0; }
    int n_elements = t->n_rows * t->n_cols;
    int n_blocks = n_elements / 32;
    size_t nbytes_padded = (size_t)n_blocks * 36;  /* 36 bytes per padded block */

    /* Repack on host: insert 2-byte padding after each scale */
    uint8_t *padded = (uint8_t *)cllm_stage_buf(r, nbytes_padded);
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
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_llm: upload_q8_0_raw alloc failed (%zu bytes, err=%d)\n", nbytes_padded, (int)err);
        return -1;
    }
    err = cuMemcpyHtoD(*d_ptr, padded, nbytes_padded);
    if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}

static int upload_q8_0_shadow_f16(cuda_llm_runner *r, CUdeviceptr *d_ptr, const qtensor *t) {
    /* Create F16 shadow when cuBLAS is enabled, for smaller quantized types.
     * Skip large tensors (>10M elements or >20MB F16) to avoid OOM. */
    if (!r || !r->use_cublas || !t->data ||
        t->type == GGML_TYPE_F32 ||
        (t->type != GGML_TYPE_Q8_0 && t->type != GGML_TYPE_F16)) {
        *d_ptr = 0;
        return 0;
    }
    if ((int64_t)t->n_rows * t->n_cols > 10000000) {  /* >10M elements = >20MB F16 → skip */
        *d_ptr = 0;
        return 0;
    }

    int n_rows = t->n_rows;
    int n_cols = t->n_cols;
    int n_elements = n_rows * n_cols;
    uint16_t *f16_buf = (uint16_t *)cllm_stage_buf(r, (size_t)n_elements * sizeof(uint16_t));
    float *row_tmp = NULL;
    if (!f16_buf) return -1;
    row_tmp = (float *)malloc((size_t)n_cols * sizeof(float));
    if (!row_tmp) return -1;

    size_t row_bytes = dequant_row_size(t->type, t->n_cols);
    const uint8_t *base = (const uint8_t *)t->data;
    for (int row = 0; row < n_rows; row++) {
        dequant_row(t->type, base + (size_t)row * row_bytes, row_tmp, n_cols);
        for (int col = 0; col < n_cols; col++) {
            f16_buf[(size_t)row * n_cols + col] = cllm_f32_to_f16(row_tmp[col]);
        }
    }
    free(row_tmp);

    size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
    CUresult err = cuMemAlloc(d_ptr, nbytes);
    if (err != CUDA_SUCCESS) {
        *d_ptr = 0;
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_llm: F16 shadow alloc failed (%zu bytes, type=%d, continuing without cuBLAS for this weight)\n",
                    nbytes, t->type);
        return 0;
    }
    err = cuMemcpyHtoD(*d_ptr, f16_buf, nbytes);
    if (err != CUDA_SUCCESS) {
        cuMemFree(*d_ptr);
        *d_ptr = 0;
        return -1;
    }
    return 0;
}

/* Upload a K-quant tensor directly to GPU (no repack needed — already aligned) */
static int upload_kquant_raw(CUdeviceptr *d_ptr, const qtensor *t) {
    if (!t->data) { *d_ptr = 0; return 0; }
    size_t nbytes = dequant_row_size(t->type, t->n_cols) * (size_t)t->n_rows;
    CUresult err = cuMemAlloc(d_ptr, nbytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_kquant_raw alloc failed (%zu bytes, type=%d, err=%d, n_cols=%d, n_rows=%d)\n", nbytes, t->type, (int)err, t->n_cols, t->n_rows); return -1; }
    err = cuMemcpyHtoD(*d_ptr, t->data, nbytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_kquant_raw copy failed\n"); cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}

/* Block-major repack for dense MMQ weights. Transforms row-major [N][nb][bs] to
 * block-major [nb][N][bs] so the kernel reads 64 contiguous rows per block group.
 * The original data is preserved (row-major stays on device for fallback paths). */
static int upload_kquant_raw_bm(CUdeviceptr *d_ptr_bm, const qtensor *t, int bm_enabled) {
    if (!t->data || !bm_enabled) { *d_ptr_bm = 0; return 0; }
    int N = t->n_rows, nb = t->n_cols / 256;
    int bs = (int)dequant_row_size(t->type, t->n_cols) / N; /* bytes per row */
    if (bs <= 0 || nb <= 0) { *d_ptr_bm = 0; return 0; }
    size_t row_bytes = (size_t)nb * bs;
    size_t total = (size_t)N * row_bytes;
    unsigned char *repacked = (unsigned char *)malloc(total);
    if (!repacked) { *d_ptr_bm = 0; return -1; }
    const unsigned char *src = (const unsigned char *)t->data;
    for (int b = 0; b < nb; b++)
        for (int n = 0; n < N; n++)
            memcpy(repacked + ((size_t)b * N + n) * bs,
                   src + ((size_t)n * nb + b) * bs, bs);
    CUresult err = cuMemAlloc(d_ptr_bm, total);
    if (err != CUDA_SUCCESS) { free(repacked); *d_ptr_bm = 0; return -1; }
    err = cuMemcpyHtoD(*d_ptr_bm, repacked, total);
    free(repacked);
    if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr_bm); *d_ptr_bm = 0; return -1; }
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

static uint16_t cllm_bf16_to_f16(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
    int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) {
        if (exp < -24) return sign;
        mant |= 0x800000;
        mant >>= (-1 - exp);
        return sign | (uint16_t)(mant >> 13);
    }
    return sign | (uint16_t)((exp + 15) << 10) | (uint16_t)(mant >> 13);
}

static float cllm_fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1u;
    uint32_t exp  = (b >> 3) & 0xFu;
    uint32_t mant = b & 0x7u;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0) {
        f = ldexpf((float)mant / 8.0f, -6);
    } else if (exp == 15 && mant == 7) {
        return 0.0f;
    } else {
        f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -f : f;
}

static int upload_bf16_matrix_gpu_as_f16(cuda_llm_runner *r, CUdeviceptr *d_ptr,
                                         const qtensor *t, const char *name) {
    if (!r || !r->fn_bf16_to_f16_inplace || !t->data) return -1;
    int n_elements = t->n_rows * t->n_cols;
    size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
    CUresult err = cuMemAlloc(d_ptr, nbytes);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_llm: GPU BF16->F16 alloc failed (%zu bytes, err=%d)\n",
                nbytes, (int)err);
        *d_ptr = 0;
        return -1;
    }
    err = cuMemcpyHtoDAsync(*d_ptr, t->data, nbytes, r->stream);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_llm: GPU BF16->F16 upload failed (err=%d)\n", (int)err);
        cuMemFree(*d_ptr);
        *d_ptr = 0;
        return -1;
    }
    CUdeviceptr data = *d_ptr;
    void *args[] = { &data, &n_elements };
    err = cuLaunchKernel(r->fn_bf16_to_f16_inplace,
                         (unsigned)((n_elements + 255) / 256), 1, 1,
                         256, 1, 1, 0, r->stream, args, NULL);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_llm: GPU BF16->F16 kernel launch failed (err=%d)\n", (int)err);
        cuMemFree(*d_ptr);
        *d_ptr = 0;
        return -1;
    }
    err = cuStreamSynchronize(r->stream);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_llm: GPU BF16->F16 sync failed (err=%d)\n", (int)err);
        cuMemFree(*d_ptr);
        *d_ptr = 0;
        return -1;
    }
    if (getenv("CUDA_LLM_GPU_BF16_VERIFY")) {
        uint16_t *got = (uint16_t *)malloc(nbytes);
        if (!got) {
            cuMemFree(*d_ptr);
            *d_ptr = 0;
            return -1;
        }
        err = cuMemcpyDtoH(got, *d_ptr, nbytes);
        if (err != CUDA_SUCCESS) {
            fprintf(stderr, "cuda_llm: GPU BF16->F16 verify copy failed (err=%d)\n", (int)err);
            free(got);
            cuMemFree(*d_ptr);
            *d_ptr = 0;
            return -1;
        }
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n_elements; i++) {
            uint16_t want = cllm_bf16_to_f16(src[i]);
            if (got[i] != want) {
                fprintf(stderr,
                        "cuda_llm: GPU BF16->F16 mismatch at %s[%d]: bf16=%04x got=%04x want=%04x\n",
                        name ? name : "?", i, src[i], got[i], want);
                free(got);
                cuMemFree(*d_ptr);
                *d_ptr = 0;
                return -1;
            }
        }
        free(got);
    }
    return 0;
}

/* ---- F16 weight cache (persistent BF16→F16 sidecar) -----------------------
 *
 * On the cold path the runner converts every BF16 safetensors weight to F16
 * on the host before HtoD. For the flux2 text encoder this dominates cold
 * start (per doc/flux2-klein.md:460). The cache is a stream-style sidecar
 * `<model_path>.f16cache` (or `<dir>/cache.f16cache` for shard dirs): on
 * miss the writer dumps every converted F16 buffer in upload order; on hit
 * the reader serves them back in the same order, skipping host conversion.
 *
 * Disable with FLUX2_F16CACHE_DISABLE=1.  The cache is keyed by the sum of
 * (mtime, size) over all shard files — touching any shard invalidates.
 */

typedef struct {
    int        active_read;
    int        active_write;
    int        aborted;
    const uint8_t *map_base;
    size_t     map_size;
    size_t     read_pos;
    FILE      *write_fp;
    char       path_tmp[1200];
    char       path_final[1100];
    uint64_t   key;
} cllm_f16cache_t;

static cllm_f16cache_t g_f16cache = {0};

#define CLLM_F16CACHE_MAGIC 0x43464C4331u  /* "1CLFC" */
#define CLLM_F16CACHE_VERSION 3u

static uint64_t cllm_f16cache_hash64(const void *data, size_t n, uint64_t h) {
    const uint8_t *p = (const uint8_t *)data;
    if (!h) h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t cllm_f16cache_record_sig(const char *name, const qtensor *t, size_t n_bytes) {
    uint64_t h = cllm_f16cache_hash64(name ? name : "?", strlen(name ? name : "?"), 0);
    uint64_t v[7] = {
        (uint64_t)t->type,
        (uint64_t)t->n_dims,
        (uint64_t)t->n_rows,
        (uint64_t)t->n_cols,
        (uint64_t)t->dims[0],
        (uint64_t)t->dims[1],
        (uint64_t)n_bytes
    };
    return cllm_f16cache_hash64(v, sizeof(v), h);
}

static uint32_t cllm_f16cache_crc32(const void *data, size_t n) {
#if defined(__SSE4_2__) && (defined(__x86_64__) || defined(_M_X64))
    const uint8_t *p = (const uint8_t *)data;
    uint64_t c = 0xFFFFFFFFu;
    while (n >= 8) {
        uint64_t v;
        memcpy(&v, p, sizeof(v));
        c = _mm_crc32_u64(c, v);
        p += 8;
        n -= 8;
    }
    uint32_t c32 = (uint32_t)c;
    while (n > 0) {
        c32 = _mm_crc32_u8(c32, *p++);
        n--;
    }
    return c32 ^ 0xFFFFFFFFu;
#elif defined(__SSE4_2__) && (defined(__i386__) || defined(_M_IX86))
    const uint8_t *p = (const uint8_t *)data;
    uint32_t c = 0xFFFFFFFFu;
    while (n >= 4) {
        uint32_t v;
        memcpy(&v, p, sizeof(v));
        c = _mm_crc32_u32(c, v);
        p += 4;
        n -= 4;
    }
    while (n > 0) {
        c = _mm_crc32_u8(c, *p++);
        n--;
    }
    return c ^ 0xFFFFFFFFu;
#else
    static uint32_t table[256];
    static int table_ready = 0;
    if (!table_ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int j = 0; j < 8; j++)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        table_ready = 1;
    }
    const uint8_t *p = (const uint8_t *)data;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++)
        c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
#endif
}

static uint64_t cllm_f16cache_compute_key(const char *model_path) {
    struct stat sb;
    if (stat(model_path, &sb) != 0) return 0;
    uint64_t key = 0;
    if (S_ISDIR(sb.st_mode)) {
        char p[1100];
        for (int i = 1; i <= 999; i++) {
            int hit = 0;
            for (int total = i; total <= 999; total++) {
                snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", model_path, i, total);
                if (stat(p, &sb) == 0) { hit = 1; break; }
            }
            if (!hit) break;
            key = key * 1315423911ull + (uint64_t)sb.st_mtime;
            key = key * 1315423911ull + (uint64_t)sb.st_size;
        }
    } else {
        key = (uint64_t)sb.st_mtime * 1315423911ull + (uint64_t)sb.st_size;
    }
    return key;
}

static void cllm_f16cache_sidecar_path(const char *model_path, char *out, size_t out_sz) {
    struct stat sb;
    if (stat(model_path, &sb) == 0 && S_ISDIR(sb.st_mode)) {
        snprintf(out, out_sz, "%s/cache.f16cache", model_path);
    } else {
        snprintf(out, out_sz, "%s.f16cache", model_path);
    }
}

static void cllm_f16cache_begin(const char *model_path, int verbose) {
    memset(&g_f16cache, 0, sizeof(g_f16cache));
    if (getenv("FLUX2_F16CACHE_DISABLE")) return;
    if (getenv("CUDA_LLM_GPU_BF16_TO_F16")) return;
    if (!model_path) return;

    g_f16cache.key = cllm_f16cache_compute_key(model_path);
    if (!g_f16cache.key) return;
    cllm_f16cache_sidecar_path(model_path, g_f16cache.path_final, sizeof(g_f16cache.path_final));

    /* Try reader first */
    int fd = open(g_f16cache.path_final, O_RDONLY);
    if (fd >= 0) {
        struct stat sb;
        if (fstat(fd, &sb) == 0 && sb.st_size >= (off_t)(4 * sizeof(uint64_t))) {
            void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (map != MAP_FAILED) {
                const uint64_t *hdr = (const uint64_t *)map;
                uint64_t magic = hdr[0], version = hdr[1], key = hdr[2], payload_bytes = hdr[3];
                if (magic == CLLM_F16CACHE_MAGIC && version == CLLM_F16CACHE_VERSION &&
                    key == g_f16cache.key &&
                    payload_bytes <= (uint64_t)sb.st_size - 4 * sizeof(uint64_t)) {
                    g_f16cache.map_base = (const uint8_t *)map;
                    g_f16cache.map_size = (size_t)sb.st_size;
                    g_f16cache.read_pos = 4 * sizeof(uint64_t);
                    g_f16cache.active_read = 1;
                    close(fd);
                    if (verbose >= 1)
                        fprintf(stderr, "cuda_llm: f16 cache HIT %s (%.1f MB)\n",
                                g_f16cache.path_final,
                                (double)payload_bytes / (1<<20));
                    return;
                }
                munmap(map, sb.st_size);
            }
        }
        close(fd);
    }

    /* Writer. Use a unique temp path so concurrent cold processes do not
     * clobber each other's sidecar stream before the final atomic rename. */
    snprintf(g_f16cache.path_tmp, sizeof(g_f16cache.path_tmp), "%s.tmp.%ld",
             g_f16cache.path_final, (long)getpid());
    int wfd = open(g_f16cache.path_tmp, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (wfd < 0) return;
    g_f16cache.write_fp = fdopen(wfd, "wb");
    if (!g_f16cache.write_fp) { close(wfd); unlink(g_f16cache.path_tmp); return; }
    uint64_t hdr[4] = { CLLM_F16CACHE_MAGIC, CLLM_F16CACHE_VERSION, g_f16cache.key, 0 };
    if (fwrite(hdr, sizeof(hdr), 1, g_f16cache.write_fp) != 1) {
        fclose(g_f16cache.write_fp); g_f16cache.write_fp = NULL;
        unlink(g_f16cache.path_tmp);
        return;
    }
    g_f16cache.active_write = 1;
    if (verbose >= 1)
        fprintf(stderr, "cuda_llm: f16 cache MISS, writing %s\n", g_f16cache.path_final);
}

/* Reader: returns pointer to n_bytes of cached F16 data (consumed from cursor),
 * or NULL if cache is in write mode / aborted / underflow / signature mismatch. */
static const void *cllm_f16cache_read_consume(const char *name, const qtensor *t, size_t n_bytes) {
    if (!g_f16cache.active_read || g_f16cache.aborted) return NULL;
    if (g_f16cache.read_pos + 3 * sizeof(uint64_t) > g_f16cache.map_size) {
        fprintf(stderr, "cuda_llm: f16 cache record header underflow at %s\n",
                name ? name : "?");
        g_f16cache.aborted = 1;
        return NULL;
    }
    const uint64_t *rec = (const uint64_t *)(g_f16cache.map_base + g_f16cache.read_pos);
    uint64_t sig = cllm_f16cache_record_sig(name, t, n_bytes);
    uint64_t rec_sig = rec[0];
    uint64_t rec_bytes = rec[1];
    uint32_t rec_crc = (uint32_t)rec[2];
    g_f16cache.read_pos += 3 * sizeof(uint64_t);
    if (rec_sig != sig || rec_bytes != (uint64_t)n_bytes ||
        g_f16cache.read_pos + n_bytes > g_f16cache.map_size) {
        fprintf(stderr,
                "cuda_llm: f16 cache record mismatch at %s "
                "(sig %llx/%llx, bytes %llu/%zu); invalidating\n",
                name ? name : "?",
                (unsigned long long)rec_sig, (unsigned long long)sig,
                (unsigned long long)rec_bytes, n_bytes);
        g_f16cache.aborted = 1;
        return NULL;
    }
    const void *p = g_f16cache.map_base + g_f16cache.read_pos;
    uint32_t crc = cllm_f16cache_crc32(p, n_bytes);
    if (crc != rec_crc) {
        fprintf(stderr,
                "cuda_llm: f16 cache CRC mismatch at %s (%08x/%08x); invalidating\n",
                name ? name : "?", rec_crc, crc);
        g_f16cache.aborted = 1;
        return NULL;
    }
    g_f16cache.read_pos += n_bytes;
    return p;
}

/* Writer: append n_bytes of converted F16 to the sidecar. */
static void cllm_f16cache_write_append(const char *name, const qtensor *t,
                                       const void *data, size_t n_bytes) {
    if (!g_f16cache.active_write || g_f16cache.aborted) return;
    uint64_t rec[3] = {
        cllm_f16cache_record_sig(name, t, n_bytes),
        (uint64_t)n_bytes,
        (uint64_t)cllm_f16cache_crc32(data, n_bytes)
    };
    if (fwrite(rec, sizeof(rec), 1, g_f16cache.write_fp) != 1) {
        fprintf(stderr, "cuda_llm: f16 cache record write failed; aborting\n");
        g_f16cache.aborted = 1;
        fclose(g_f16cache.write_fp); g_f16cache.write_fp = NULL;
        unlink(g_f16cache.path_tmp);
        return;
    }
    if (fwrite(data, 1, n_bytes, g_f16cache.write_fp) != n_bytes) {
        fprintf(stderr, "cuda_llm: f16 cache write failed; aborting\n");
        g_f16cache.aborted = 1;
        fclose(g_f16cache.write_fp); g_f16cache.write_fp = NULL;
        unlink(g_f16cache.path_tmp);
    }
}

static void cllm_f16cache_end(int success, int verbose) {
    if (g_f16cache.active_read) {
        if (g_f16cache.map_base) munmap((void *)g_f16cache.map_base, g_f16cache.map_size);
        if (g_f16cache.aborted) {
            unlink(g_f16cache.path_final);
            if (verbose >= 1)
                fprintf(stderr, "cuda_llm: f16 cache invalidated %s\n", g_f16cache.path_final);
        }
    } else if (g_f16cache.active_write && g_f16cache.write_fp) {
        if (success && !g_f16cache.aborted) {
            long payload_end = ftell(g_f16cache.write_fp);
            uint64_t payload_bytes = (uint64_t)payload_end - 4 * sizeof(uint64_t);
            fseek(g_f16cache.write_fp, 3 * sizeof(uint64_t), SEEK_SET);
            fwrite(&payload_bytes, sizeof(payload_bytes), 1, g_f16cache.write_fp);
            fclose(g_f16cache.write_fp);
            if (rename(g_f16cache.path_tmp, g_f16cache.path_final) != 0) {
                unlink(g_f16cache.path_tmp);
            } else if (verbose >= 1) {
                fprintf(stderr, "cuda_llm: f16 cache wrote %s (%.1f MB)\n",
                        g_f16cache.path_final, (double)payload_bytes / (1<<20));
            }
        } else {
            fclose(g_f16cache.write_fp);
            unlink(g_f16cache.path_tmp);
        }
    }
    memset(&g_f16cache, 0, sizeof(g_f16cache));
}

/* Upload a weight matrix - dispatches based on type */
/* Upload weight matrix as F32 (dequant any type to F32, copy to GPU) */
static int upload_weight_f32(CUdeviceptr *d_ptr, const qtensor *t) {
    if (!t->data) { *d_ptr = 0; return 0; }
    int n_elements = t->n_rows * t->n_cols;
    float *f32_buf = (float *)malloc((size_t)n_elements * sizeof(float));
    if (!f32_buf) {
        fprintf(stderr, "cuda_llm: upload_weight_f32 malloc failed (rows=%d, cols=%d)\n",
                t->n_rows, t->n_cols);
        return -1;
    }
    size_t row_bytes = dequant_row_size(t->type, t->n_cols);
    const uint8_t *base = (const uint8_t *)t->data;
    for (int row = 0; row < t->n_rows; row++) {
        dequant_row(t->type,
                    base + (size_t)row * row_bytes,
                    f32_buf + (size_t)row * t->n_cols,
                    t->n_cols);
    }
    size_t nbytes = (size_t)n_elements * sizeof(float);
    CUresult err = cuMemAlloc(d_ptr, nbytes);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_llm: upload_weight_f32 alloc failed (%zu bytes, err=%d)\n",
                nbytes, (int)err);
        free(f32_buf);
        return -1;
    }
    err = cuMemcpyHtoD(*d_ptr, f32_buf, nbytes);
    free(f32_buf);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_llm: upload_weight_f32 copy failed\n");
        cuMemFree(*d_ptr);
        *d_ptr = 0;
        return -1;
    }
    return 0;
}

/* Upload weight + optional block-major repack for dense MMQ.
 * When dense_bm is set, replaces the original row-major pointer with the
 * block-major repacked version. The launch helper passes bm=1. */
static int upload_weight_matrix(cuda_llm_runner *r, CUdeviceptr *d_ptr, const qtensor *t,
                                int *out_type, const char *cache_name);
 static int upload_weight_matrix_bm(cuda_llm_runner *r, CUdeviceptr *d_ptr, CUdeviceptr *d_ptr_bm,
                                    const qtensor *t, int *out_type, const char *cache_name) {
     if (upload_weight_matrix(r, d_ptr, t, out_type, cache_name) != 0) return -1;
     if (r->dense_bm && d_ptr_bm) {
         if (upload_kquant_raw_bm(d_ptr_bm, t, 1) != 0) *d_ptr_bm = 0;
     } else if (d_ptr_bm) {
         *d_ptr_bm = 0;
     }
     return 0;
 }

static int upload_weight_matrix(cuda_llm_runner *r, CUdeviceptr *d_ptr, const qtensor *t,
                                int *out_type, const char *cache_name) {
    *out_type = t->type;
    if (t->type == GGML_TYPE_Q8_0) {
        return upload_q8_0_raw(r, d_ptr, t);
    } else if (t->type == GGML_TYPE_Q2_K || t->type == GGML_TYPE_Q3_K ||
               t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K ||
               t->type == GGML_TYPE_Q6_K ||
               t->type == GGML_TYPE_IQ4_XS || t->type == GGML_TYPE_IQ4_NL) {
        return upload_kquant_raw(d_ptr, t);
    } else if (t->type == GGML_TYPE_F32) {
        /* F32 → F16, then upload */
        *out_type = GGML_TYPE_F16;
        int n_elements = t->n_rows * t->n_cols;
        const float *f32_data = (const float *)t->data;
        uint16_t *f16_buf = (uint16_t *)malloc((size_t)n_elements * sizeof(uint16_t));
        if (!f16_buf) return -1;
        for (int i = 0; i < n_elements; i++) f16_buf[i] = cllm_f32_to_f16(f32_data[i]);
        size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
        CUresult err = cuMemAlloc(d_ptr, nbytes);
        if (err != CUDA_SUCCESS) { free(f16_buf); return -1; }
        err = cuMemcpyHtoD(*d_ptr, f16_buf, nbytes);
        free(f16_buf);
        if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
        return 0;
    } else if (t->type == GGML_TYPE_BF16) {
        /* BF16 -> F16 in one host pass, then upload as F16.  If the f16 cache
         * is active (read mode), skip conversion and upload directly from the
         * mmap'd sidecar; if in write mode, append the converted buffer. */
        *out_type = GGML_TYPE_F16;
        int n_elements = t->n_rows * t->n_cols;
        size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
        CUresult err;
        const void *cached = cllm_f16cache_read_consume(cache_name, t, nbytes);
        if (cached) {
            err = cuMemAlloc(d_ptr, nbytes);
            if (err != CUDA_SUCCESS) return -1;
            err = cuMemcpyHtoD(*d_ptr, cached, nbytes);
            if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
            return 0;
        }
        if (getenv("CUDA_LLM_GPU_BF16_TO_F16")) {
            return upload_bf16_matrix_gpu_as_f16(r, d_ptr, t, cache_name);
        }
        uint16_t *f16_buf = (uint16_t *)cllm_stage_buf(r, nbytes);
        if (!f16_buf) return -1;
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n_elements; i++) f16_buf[i] = cllm_bf16_to_f16(src[i]);
        cllm_f16cache_write_append(cache_name, t, f16_buf, nbytes);
        err = cuMemAlloc(d_ptr, nbytes);
        if (err != CUDA_SUCCESS) { return -1; }
        err = cuMemcpyHtoD(*d_ptr, f16_buf, nbytes);
        if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
        return 0;
    } else if (t->type == CLLM_TYPE_F8_E4M3) {
        *out_type = GGML_TYPE_F16;
        int n_elements = t->n_rows * t->n_cols;
        size_t nbytes = (size_t)n_elements * sizeof(uint16_t);
        uint16_t *f16_buf = (uint16_t *)cllm_stage_buf(r, nbytes);
        if (!f16_buf) return -1;
        const uint8_t *src = (const uint8_t *)t->data;
        float scale = t->has_scale ? t->scale : 1.0f;
        for (int i = 0; i < n_elements; i++)
            f16_buf[i] = cllm_f32_to_f16(cllm_fp8_e4m3_to_f32(src[i]) * scale);
        CUresult err = cuMemAlloc(d_ptr, nbytes);
        if (err != CUDA_SUCCESS) { return -1; }
        err = cuMemcpyHtoD(*d_ptr, f16_buf, nbytes);
        if (err != CUDA_SUCCESS) { cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
        return 0;
    } else if (t->type == GGML_TYPE_F16) {
        return upload_f16_matrix(d_ptr, t);
    } else if (t->type == GGML_TYPE_IQ2_XXS || t->type == GGML_TYPE_IQ2_XS ||
               t->type == GGML_TYPE_IQ2_S   || t->type == GGML_TYPE_IQ3_XXS ||
               t->type == GGML_TYPE_IQ3_S   || t->type == GGML_TYPE_IQ4_NL ||
               t->type == GGML_TYPE_IQ4_XS  || t->type == GGML_TYPE_IQ1_S ||
               t->type == GGML_TYPE_IQ1_M   || t->type == GGML_TYPE_Q4_0 ||
               t->type == GGML_TYPE_Q4_1    || t->type == GGML_TYPE_Q5_0 ||
               t->type == GGML_TYPE_Q5_1    || t->type == GGML_TYPE_TQ1_0 ||
               t->type == GGML_TYPE_TQ2_0) {
        /* Upload raw quantized data — dequant happens in-kernel during matvec */
        return upload_kquant_raw(d_ptr, t);
    } else {
        /* Default: unknown type */
        fprintf(stderr, "cuda_llm: upload_weight_matrix: unhandled type %d (rows=%d, cols=%d)\n", t->type, t->n_rows, t->n_cols);
        return upload_f16_matrix(d_ptr, t);
    }
}

/* Upload a 3D K-quant tensor (stacked experts) directly to GPU.
 * Returns per-expert byte stride via out_stride.
 * If repack_bm and type==IQ2_XXS: transpose each expert from row-major [N][nb][66] to
 * block-major [nb][N][66] on the host before upload, so the MMQ kernel's per-row-tile read
 * of one 256-block is contiguous (recovers the cache-line over-fetch). estride is unchanged. */
static int upload_3d_kquant_raw_ex(CUdeviceptr *d_ptr, const qtensor *t, size_t *out_stride, int repack_bm) {
    if (!t->data) { *d_ptr = 0; return 0; }
    /* For 3D tensors: dims[0]=cols, dims[1]=rows_per_expert, dims[2]=n_experts
     * Total rows = dims[1] * dims[2] (already computed in t->n_rows by cllm_load_tensor) */
    size_t row_bytes = dequant_row_size(t->type, t->n_cols);
    int rows_per_expert = (t->n_dims >= 3) ? (int)t->dims[1] : t->n_rows;
    *out_stride = row_bytes * (size_t)rows_per_expert;
    size_t total_bytes = row_bytes * (size_t)t->n_rows;
    const void *src = t->data;
    void *repacked = NULL;
    if (repack_bm && t->type == GGML_TYPE_IQ2_XXS && (t->n_cols % 256) == 0) {
        int N = rows_per_expert, nb = t->n_cols / 256;
        int n_exp = (int)(t->n_rows / rows_per_expert);
        size_t estride = (size_t)N * nb * 66;
        repacked = malloc(total_bytes);
        if (!repacked) { fprintf(stderr, "cuda_llm: IQ2 repack malloc failed\n"); return -1; }
        const unsigned char *s = (const unsigned char *)t->data;
        unsigned char *d = (unsigned char *)repacked;
        for (int e = 0; e < n_exp; e++) {
            const unsigned char *se = s + (size_t)e * estride;
            unsigned char *de = d + (size_t)e * estride;
            for (int n = 0; n < N; n++)
                for (int bg = 0; bg < nb; bg++)
                    memcpy(de + (size_t)bg * N * 66 + (size_t)n * 66,
                           se + (size_t)n * nb * 66 + (size_t)bg * 66, 66);
        }
        src = repacked;
    }
    CUresult err = cuMemAlloc(d_ptr, total_bytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_3d_kquant alloc failed (%zu bytes, type=%d, err=%d)\n", total_bytes, t->type, (int)err); free(repacked); return -1; }
    err = cuMemcpyHtoD(*d_ptr, src, total_bytes);
    free(repacked);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: upload_3d_kquant copy failed\n"); cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
    return 0;
}
static int upload_3d_kquant_raw(CUdeviceptr *d_ptr, const qtensor *t, size_t *out_stride) {
    return upload_3d_kquant_raw_ex(d_ptr, t, out_stride, 0);
}

/* Upload F16 shadow of a 3D K-quant tensor for cuBLAS.
 * Converts each expert's quantized rows to F16 on CPU, uploads as flat F16 array.
 * The F16 shadow has the same layout as the raw data but with F16 elements.
 * stride = row_bytes_per_expert_in_f16 (= rows_per_expert * cols * sizeof(uint16_t)) */
static int upload_3d_kquant_f16(cuda_llm_runner *r, CUdeviceptr *d_ptr, const qtensor *t,
                                  int rows_per_expert, int cols, size_t raw_stride) {
    if (!r || !r->use_cublas || !t->data) { *d_ptr = 0; return 0; }
    int n_experts = t->n_rows / rows_per_expert;
    size_t f16_stride = (size_t)rows_per_expert * cols * sizeof(uint16_t);
    size_t total_bytes = f16_stride * n_experts;
    
    /* For large expert tensors (>10 MB), skip bulk upload — lazy per-expert upload at runtime */
    if (total_bytes > 10 * 1024 * 1024) {
        *d_ptr = 0; return 0;
    }
    
    uint16_t *f16_buf = (uint16_t *)malloc(total_bytes);
    if (!f16_buf) { fprintf(stderr, "cuda_llm: malloc failed for moe F16 shadow (%zu bytes)\n", total_bytes); *d_ptr = 0; return -1; }
    float *row_tmp = (float *)malloc((size_t)cols * sizeof(float));
    if (!row_tmp) { free(f16_buf); *d_ptr = 0; return -1; }
    
    size_t row_bytes = dequant_row_size(t->type, cols);
    const uint8_t *base = (const uint8_t *)t->data;
    
    for (int e = 0; e < n_experts; e++) {
        for (int r = 0; r < rows_per_expert; r++) {
            dequant_row(t->type, base + (size_t)e * raw_stride + (size_t)r * row_bytes, row_tmp, cols);
            for (int c = 0; c < cols; c++) {
                f16_buf[(size_t)e * f16_stride / sizeof(uint16_t) + (size_t)r * cols + c] = cllm_f32_to_f16(row_tmp[c]);
            }
        }
    }
    free(row_tmp);
    
    CUresult err = cuMemAlloc(d_ptr, total_bytes);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: moe F16 shadow cuMemAlloc failed (%zu bytes, err=%d)\n", total_bytes, (int)err); free(f16_buf); *d_ptr = 0; return -1; }
    err = cuMemcpyHtoD(*d_ptr, f16_buf, total_bytes);
    free(f16_buf);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: moe F16 shadow cuMemcpyHtoD failed\n"); cuMemFree(*d_ptr); *d_ptr = 0; return -1; }
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

static int cllm_st_dtype_to_ggml(const char *dtype) {
    if (!dtype) return -1;
    if (strcmp(dtype, "F32") == 0) return GGML_TYPE_F32;
    if (strcmp(dtype, "F16") == 0) return GGML_TYPE_F16;
    if (strcmp(dtype, "BF16") == 0) return GGML_TYPE_BF16;
    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0)
        return CLLM_TYPE_F8_E4M3;
    return -1;
}

static int cllm_st_read_scalar_f32_multi(st_context **shards, int n_shards,
                                         const char *name, float *out) {
    for (int s = 0; s < n_shards; s++) {
        int idx = safetensors_find(shards[s], name);
        if (idx < 0) continue;
        if (strcmp(safetensors_dtype(shards[s], idx), "F32") != 0 ||
            safetensors_nbytes(shards[s], idx) != sizeof(float)) {
            return -1;
        }
        memcpy(out, safetensors_data(shards[s], idx), sizeof(float));
        return 0;
    }
    return -1;
}

static qtensor cllm_st_load_tensor_multi(st_context **shards, int n_shards,
                                         const char *name, int required) {
    qtensor t = {0};
    st_context *st = NULL;
    int idx = -1;
    for (int s = 0; s < n_shards && idx < 0; s++) {
        idx = safetensors_find(shards[s], name);
        if (idx >= 0) st = shards[s];
    }
    if (idx < 0) {
        if (required) fprintf(stderr, "cuda_llm: missing safetensors tensor '%s'\n", name);
        return t;
    }

    int ggml_type = cllm_st_dtype_to_ggml(safetensors_dtype(st, idx));
    if (ggml_type < 0) {
        fprintf(stderr, "cuda_llm: unsupported safetensors dtype '%s' for %s\n",
                safetensors_dtype(st, idx), name);
        return t;
    }

    const uint64_t *sh = safetensors_shape(st, idx);
    int nd = safetensors_ndims(st, idx);
    t.data = safetensors_data(st, idx);
    t.type = ggml_type;
    t.n_dims = nd > 4 ? 4 : nd;
    if (ggml_type == CLLM_TYPE_F8_E4M3) {
        size_t name_len = strlen(name);
        if (name_len > 7 && strcmp(name + name_len - 7, ".weight") == 0) {
            char scale_name[512];
            if (name_len - 7 + strlen(".scale_weight") < sizeof(scale_name)) {
                memcpy(scale_name, name, name_len - 7);
                strcpy(scale_name + name_len - 7, ".scale_weight");
                float scale = 1.0f;
                if (cllm_st_read_scalar_f32_multi(shards, n_shards, scale_name, &scale) == 0) {
                    t.scale = scale;
                    t.has_scale = 1;
                } else {
                    fprintf(stderr,
                            "cuda_llm: FP8 safetensors tensor %s requires scalar F32 scale tensor %s\n",
                            name, scale_name);
                    memset(&t, 0, sizeof(t));
                    return t;
                }
            } else {
                fprintf(stderr, "cuda_llm: FP8 safetensors scale name too long for %s\n", name);
                memset(&t, 0, sizeof(t));
                return t;
            }
        }
    }
    if (nd <= 1) {
        int n = (nd == 0) ? 1 : (int)sh[0];
        t.n_rows = 1;
        t.n_cols = n;
        t.dims[0] = (uint64_t)n;
    } else {
        t.n_rows = (int)sh[0];
        t.n_cols = (int)sh[1];
        t.dims[0] = sh[1];
        t.dims[1] = sh[0];
        for (int d = 2; d < t.n_dims; d++) t.dims[d] = sh[d];
    }
    return t;
}

static int cllm_open_safetensors_shards(const char *model_path, st_context **shards,
                                        int max_shards) {
    struct stat sb;
    if (stat(model_path, &sb) != 0) return -1;

    if (S_ISDIR(sb.st_mode)) {
        char path[512];
        int n_shards = 0;
        for (int i = 1; i <= max_shards && n_shards < max_shards; i++) {
            int found = 0;
            for (int total = i; total <= max_shards; total++) {
                snprintf(path, sizeof(path), "%s/model-%05d-of-%05d.safetensors",
                         model_path, i, total);
                if (stat(path, &sb) != 0) continue;
                st_context *s = safetensors_open(path);
                if (s) {
                    shards[n_shards++] = s;
                    found = 1;
                    break;
                }
            }
            if (!found) break;
        }
        return n_shards;
    }

    st_context *st = safetensors_open(model_path);
    if (!st) return -1;
    shards[0] = st;
    return 1;
}

static void cllm_close_safetensors_shards(st_context **shards, int n_shards) {
    for (int i = 0; i < n_shards; i++) {
        if (shards[i]) safetensors_close(shards[i]);
    }
}

static int cuda_llm_alloc_runtime_buffers(cuda_llm_runner *r, int q_dim, int kv_dim,
                                          int max_seq_len) {
    size_t kv_cache_size = (size_t)max_seq_len * kv_dim * sizeof(uint16_t);
    int n_attn_layers = 0;
    r->d_key_cache = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
    r->d_value_cache = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
    if (!r->d_key_cache || !r->d_value_cache) return -1;
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

    int max_dim = r->n_embd;
    if (q_dim > max_dim) max_dim = q_dim;
    if (r->n_ff > max_dim) max_dim = r->n_ff;
    int xb2_dim = max_dim;
    if (r->is_hybrid) {
        if (r->ssm_qkv_dim > xb2_dim) xb2_dim = r->ssm_qkv_dim;
        if (2 * q_dim > xb2_dim) xb2_dim = 2 * q_dim;
    }
    if (r->is_moe) {
        int tc_down_dim = r->n_experts_used * r->n_embd;
        if (tc_down_dim > xb2_dim) xb2_dim = tc_down_dim;
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
        int gate_dim = ff_dim;
        if (r->is_moe) {
            int tc_dim = r->n_experts_used * r->expert_ff;
            if (tc_dim > gate_dim) gate_dim = tc_dim;
        }
        CHECK_CU(cuMemAlloc(&r->d_gate, gate_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_up,   gate_dim * sizeof(float)));
    }

    if (r->is_moe) {
        CHECK_CU(cuMemAlloc(&r->d_router_logits, r->n_experts * sizeof(float)));
        r->d_router_logits_entries = r->n_experts;
        CHECK_CU(cuMemAlloc(&r->d_moe_accum, r->n_embd * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_moe_f16w, (size_t)r->expert_ff * r->n_embd * sizeof(uint16_t)));
        CHECK_CU(cuMemAlloc(&r->d_moe_f16w2, (size_t)r->expert_ff * r->n_embd * sizeof(uint16_t)));
        CHECK_CU(cuMemAlloc(&r->d_moe_f16w3, (size_t)r->expert_ff * r->n_embd * sizeof(uint16_t)));
        r->h_router_logits = (float *)malloc(r->n_experts * sizeof(float));
        if (!r->h_router_logits) return -1;
    }



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

    if (r->n_deepstack > 0) {
        CHECK_CU(cuMemAlloc(&r->d_ds_tmp, r->n_embd * sizeof(float)));
    }

    CHECK_CU(cuMemAlloc(&r->d_xb_q,      max_dim * sizeof(int8_t)));
    CHECK_CU(cuMemAlloc(&r->d_xb_scale,  sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_xb_q2,     max_dim * sizeof(int8_t)));
    CHECK_CU(cuMemAlloc(&r->d_xb_scale2, sizeof(float)));
    {
        size_t q81_bytes = ((size_t)max_dim / 32 + 1) * 36;
        CHECK_CU(cuMemAlloc(&r->d_xb_q81,   q81_bytes));
        CHECK_CU(cuMemAlloc(&r->d_xb_q81_2, q81_bytes));
    }
    CHECK_CU(cuMemAlloc(&r->d_hidden_snapshots, (size_t)3 * r->n_embd * sizeof(float)));

    CHECK_CU(cuMemAlloc(&r->d_logits, (size_t)r->n_vocab * sizeof(float)));

    {
        int out_sz = r->n_vocab > r->n_embd ? r->n_vocab : r->n_embd;
        r->h_output = (float *)malloc((size_t)out_sz * sizeof(float));
        if (!r->h_output) return -1;
    }

    r->weights_loaded = 1;
    return 0;
}

/* ======================================================================== */
/* Public API: load_weights                                                 */
/* ======================================================================== */

int cuda_llm_load_weights(cuda_llm_runner *r, gguf_context *gguf, int max_seq_len) {
    if (!r || !gguf) return -1;
    if (cuda_llm_bind_context(r) != 0) return -1;

    /* Detect architecture prefix */
    const char *arch = "qwen2";
    if (gguf_find_key(gguf, "gemma4.block_count") >= 0) arch = "gemma4";
    else if (gguf_find_key(gguf, "qwen35moe.block_count") >= 0) arch = "qwen35moe";
    else if (gguf_find_key(gguf, "qwen35.block_count") >= 0) arch = "qwen35";
    else if (gguf_find_key(gguf, "qwen3.block_count") >= 0) arch = "qwen3";
    else if (gguf_find_key(gguf, "qwen3vl.block_count") >= 0) arch = "qwen3vl";
    else if (gguf_find_key(gguf, "qwen2vl.block_count") >= 0) arch = "qwen2vl";

    char kbuf[128];
    #define ARCH_KEY(suffix) (snprintf(kbuf, sizeof(kbuf), "%s." suffix, arch), kbuf)

    r->n_embd      = cllm_get_int(gguf, ARCH_KEY("embedding_length"), 4096);
    r->n_heads     = cllm_get_int(gguf, ARCH_KEY("attention.head_count"), 32);
    r->n_kv_heads  = cllm_get_int(gguf, ARCH_KEY("attention.head_count_kv"), 8);
    /* head_count_kv may be a per-layer array (e.g. Gemma4 31B) — use first element */
    {
        int idx = gguf_find_key(gguf, ARCH_KEY("attention.head_count_kv"));
        if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY &&
            gguf->kv[idx].value.arr.n > 0) {
            uint32_t atype = gguf->kv[idx].value.arr.type;
            if (atype == GGUF_TYPE_UINT32) r->n_kv_heads = (int)((uint32_t *)gguf->kv[idx].value.arr.data)[0];
            else if (atype == GGUF_TYPE_INT32) r->n_kv_heads = ((int32_t *)gguf->kv[idx].value.arr.data)[0];
        }
    }
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

    /* Gemma4 architecture */
    r->is_gemma4 = 0;
    r->swa_pattern = NULL;
    if (strcmp(arch, "gemma4") == 0) {
        r->is_gemma4 = 1;
        /* The batched Gemma4 prefill is GEMM-based (dequant→F16→cuBLAS tensor cores),
         * which is the whole point of the fast path — enable cuBLAS by default unless
         * the user explicitly disabled it via CUDA_LLM_USE_CUBLAS=0. */
        {
            const char *ce = getenv("CUDA_LLM_USE_CUBLAS");
            if (!ce || atoi(ce) != 0) r->use_cublas = 1;
        }
        r->ple_use_f32 = 1; /* F32 PLE weights for accuracy (costs ~105MB extra VRAM) */
        r->head_dim_full = cllm_get_int(gguf, ARCH_KEY("attention.key_length"), 512);
        r->head_dim_swa  = cllm_get_int(gguf, ARCH_KEY("attention.key_length_swa"), 256);
        r->head_dim = r->head_dim_full; /* max for buffer sizing */
        r->swa_window_size = cllm_get_int(gguf, ARCH_KEY("attention.sliding_window"), 512);
        { const char *we = getenv("CUDA_LLM_SWA_WINDOW"); /* test override */
          if (we && atoi(we) > 0) r->swa_window_size = atoi(we); }
        r->n_embd_per_layer = cllm_get_int(gguf, ARCH_KEY("embedding_length_per_layer_input"), 256);
        int shared_kv_layers = cllm_get_int(gguf, ARCH_KEY("attention.shared_kv_layers"), 0);
        r->n_layer_kv_from_start = r->n_layers - shared_kv_layers;
        r->final_logit_softcapping = cllm_get_float(gguf, ARCH_KEY("final_logit_softcapping"), 30.0f);
        r->rope_freq_base_swa = cllm_get_float(gguf, ARCH_KEY("rope.freq_base_swa"), 10000.0f);
        r->embd_scale = sqrtf((float)r->n_embd);

        /* Parse SWA layer pattern */
        r->swa_pattern = (int *)calloc(r->n_layers, sizeof(int));
        if (!r->swa_pattern) {
            fprintf(stderr, "cuda_llm: Gemma4 swa_pattern alloc failed (n_layers=%d)\n", r->n_layers);
            return -1;
        }
        {
            int idx = gguf_find_key(gguf, ARCH_KEY("attention.sliding_window_pattern"));
            if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY) {
                int n = (int)gguf->kv[idx].value.arr.n;
                if (n > r->n_layers) n = r->n_layers;
                uint8_t *data = (uint8_t *)gguf->kv[idx].value.arr.data;
                for (int i = 0; i < n; i++) r->swa_pattern[i] = data[i] ? 1 : 0;
                if (r->verbose >= 1)
                    fprintf(stderr, "cuda_llm: Gemma4 SWA pattern loaded (%d layers, first=%d)\n", n, r->swa_pattern[0]);
            } else {
                for (int i = 0; i < r->n_layers; i++)
                    r->swa_pattern[i] = ((i + 1) % 6 != 0) ? 1 : 0;
            }
        }

        /* Parse per-layer KV head counts (may vary for 31B: 16 for SWA, 4 for full) */
        r->per_layer_kv_heads = NULL;
        {
            int idx = gguf_find_key(gguf, ARCH_KEY("attention.head_count_kv"));
            if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_ARRAY &&
                gguf->kv[idx].value.arr.n >= (uint64_t)r->n_layers) {
                r->per_layer_kv_heads = (int *)calloc(r->n_layers, sizeof(int));
                int n = r->n_layers;
                uint32_t atype = gguf->kv[idx].value.arr.type;
                void *data = gguf->kv[idx].value.arr.data;
                for (int i = 0; i < n; i++) {
                    if (atype == GGUF_TYPE_UINT32) r->per_layer_kv_heads[i] = (int)((uint32_t *)data)[i];
                    else if (atype == GGUF_TYPE_INT32) r->per_layer_kv_heads[i] = ((int32_t *)data)[i];
                    else r->per_layer_kv_heads[i] = r->n_kv_heads;
                }
                if (r->verbose >= 1) {
                    int same = 1;
                    for (int i = 1; i < n; i++)
                        if (r->per_layer_kv_heads[i] != r->per_layer_kv_heads[0]) { same = 0; break; }
                    fprintf(stderr, "cuda_llm: Gemma4 per-layer kv_heads: %s", same ? "uniform" : "varies");
                    if (!same) {
                        fprintf(stderr, " [");
                        for (int i = 0; i < n && i < 10; i++) fprintf(stderr, "%s%d", i?",":"", r->per_layer_kv_heads[i]);
                        if (n > 10) fprintf(stderr, ",...");
                        fprintf(stderr, "]");
                    } else {
                        fprintf(stderr, " (%d)", r->per_layer_kv_heads[0]);
                    }
                    fprintf(stderr, "\n");
                }
            }
        }

        /* Precompute RoPE inv_freq tables and upload to GPU */
        {
            /* SWA RoPE */
            int half_swa = r->head_dim_swa / 2;
            float *inv_freq_swa = (float *)malloc(half_swa * sizeof(float));
            if (!inv_freq_swa) {
                fprintf(stderr, "cuda_llm: Gemma4 inv_freq_swa alloc failed (n=%d)\n", half_swa);
                return -1;
            }
            for (int j = 0; j < half_swa; j++)
                inv_freq_swa[j] = 1.0f / powf(r->rope_freq_base_swa, (float)(2*j) / r->head_dim_swa);
            {
                CUresult err = cuMemAlloc(&r->d_rope_inv_freq_swa, half_swa * sizeof(float));
                if (err != CUDA_SUCCESS) {
                    fprintf(stderr, "cuda_llm: Gemma4 d_rope_inv_freq_swa alloc failed (err=%d)\n", (int)err);
                    free(inv_freq_swa);
                    return -1;
                }
                err = cuMemcpyHtoD(r->d_rope_inv_freq_swa, inv_freq_swa, half_swa * sizeof(float));
                if (err != CUDA_SUCCESS) {
                    fprintf(stderr, "cuda_llm: Gemma4 d_rope_inv_freq_swa copy failed (err=%d)\n", (int)err);
                    cuMemFree(r->d_rope_inv_freq_swa);
                    r->d_rope_inv_freq_swa = 0;
                    free(inv_freq_swa);
                    return -1;
                }
            }
            free(inv_freq_swa);

            /* Full-attention proportional RoPE */
            int half_full = r->head_dim_full / 2;
            float *inv_freq_full = (float *)malloc(half_full * sizeof(float));
            if (!inv_freq_full) {
                fprintf(stderr, "cuda_llm: Gemma4 inv_freq_full alloc failed (n=%d)\n", half_full);
                return -1;
            }
            for (int j = 0; j < half_full; j++)
                inv_freq_full[j] = 1.0f / powf(r->rope_freq_base, (float)(2*j) / r->head_dim_full);

            /* Apply proportional freq_factors from rope_freqs.weight */
            qtensor rope_freqs_qt = cllm_load_tensor(gguf, "rope_freqs.weight", 0);
            if (rope_freqs_qt.data) {
                float *ff = (float *)malloc(half_full * sizeof(float));
                if (!ff) {
                    fprintf(stderr, "cuda_llm: Gemma4 rope_freqs host buffer alloc failed (n=%d)\n", half_full);
                    free(inv_freq_full);
                    return -1;
                }
                dequant_row(rope_freqs_qt.type, rope_freqs_qt.data, ff, half_full);
                for (int j = 0; j < half_full; j++) inv_freq_full[j] /= ff[j];
                free(ff);
            }
            {
                CUresult err = cuMemAlloc(&r->d_rope_inv_freq_full, half_full * sizeof(float));
                if (err != CUDA_SUCCESS) {
                    fprintf(stderr, "cuda_llm: Gemma4 d_rope_inv_freq_full alloc failed (err=%d)\n", (int)err);
                    free(inv_freq_full);
                    return -1;
                }
                err = cuMemcpyHtoD(r->d_rope_inv_freq_full, inv_freq_full, half_full * sizeof(float));
                if (err != CUDA_SUCCESS) {
                    fprintf(stderr, "cuda_llm: Gemma4 d_rope_inv_freq_full copy failed (err=%d)\n", (int)err);
                    cuMemFree(r->d_rope_inv_freq_full);
                    r->d_rope_inv_freq_full = 0;
                    free(inv_freq_full);
                    return -1;
                }
            }
            free(inv_freq_full);
        }

        if (r->verbose >= 1) {
            fprintf(stderr, "cuda_llm: Gemma4: head_full=%d head_swa=%d swa_window=%d ple_dim=%d\n",
                    r->head_dim_full, r->head_dim_swa, r->swa_window_size, r->n_embd_per_layer);
            fprintf(stderr, "cuda_llm: Gemma4: n_layer_kv_start=%d softcap=%.1f\n",
                    r->n_layer_kv_from_start, r->final_logit_softcapping);
        }
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
        if (upload_q8_0_raw(r, &r->d_token_embd, &embd) != 0) return -1;
    } else if (embd.type == GGML_TYPE_Q2_K) {
        /* Q2_K has a dedicated GPU embed kernel */
        if (upload_kquant_raw(&r->d_token_embd, &embd) != 0) return -1;
    } else if (embd.type == GGML_TYPE_Q3_K || embd.type == GGML_TYPE_Q4_K ||
               embd.type == GGML_TYPE_Q5_K || embd.type == GGML_TYPE_Q6_K ||
               embd.type == GGML_TYPE_IQ4_XS || embd.type == GGML_TYPE_IQ4_NL) {
        /* K-quants without GPU embed kernels → dequant to F16 at load time */
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
    if (upload_norm_f32(r, &r->d_output_norm, &onorm, r->n_embd) != 0) {
        fprintf(stderr, "cuda_llm: upload output_norm failed!\n"); return -1;
    }

    /* Output projection (lm_head) — may be weight-tied with token_embd */
    if (r->verbose) fprintf(stderr, "cuda_llm: loading output.weight...\n");
    {
        qtensor output = cllm_load_tensor(gguf, "output.weight", 0);
        if (output.data) {
            if (upload_weight_matrix(r, &r->d_output_w, &output, &r->output_w_type, "output.weight") != 0) { fprintf(stderr, "cuda_llm: output.weight upload failed!\n"); return -1; }
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

    /* MMQ block-major repack (opt-in CUDA_LLM_MMQ_REPACK, default OFF). Transpose IQ2_XXS
     * expert weights to block-major at upload so the MMQ kernel reads each row-tile's 256-block
     * contiguously (recovers the cache-line over-fetch → ~2500 tok/s prefill). The MMQ kernel
     * is layout-parameterized (bm flag), so default-off keeps row-major everywhere (decode and
     * the cuBLAS fallback unchanged). NOTE: with the flag ON the decode matvec path (row-major)
     * is NOT yet block-major-aware — this is currently a prefill-throughput mode. */
    int moe_repack_bm = 0;
    { const char *e = getenv("CUDA_LLM_MMQ_REPACK");   /* default ON with MMQ; set =0 to opt out */
      const char *m = getenv("CUDA_LLM_MOE_MMQ");
      int repack_off = e && e[0] && strcmp(e, "0") == 0;
      int mmq_on = m && m[0] && strcmp(m, "0") != 0;  /* repack only with MMQ (block-major prefill path) */
      moe_repack_bm = !repack_off && mmq_on
                      && (r->n_embd % 64 == 0) && (r->expert_ff % 64 == 0); }
     r->moe_iq2_bm = moe_repack_bm;
    /* Dense weight block-major repack (opt-out CUDA_LLM_NO_DENSE_BM, default ON with MMQ).
     * Repacks all MMQ-compatible dense weights to block-major for contiguous 64-row reads. */
    { const char *e = getenv("CUDA_LLM_NO_DENSE_BM");
      int no_bm = e && e[0] && strcmp(e, "0") != 0;
      int mmq_possible = !getenv("CUDA_LLM_NO_MMQ_DENSE");
      r->dense_bm = !no_bm && mmq_possible; }


    for (int l = 0; l < r->n_layers; l++) {
        char name[128];
        cuda_layer *cl = &r->layers[l];
        if (r->verbose >= 2) fprintf(stderr, "cuda_llm: loading layer %d/%d\n", l, r->n_layers);

        /* Determine layer type */
        int is_ssm = (r->is_hybrid && r->full_attn_interval > 0 &&
                      (l + 1) % r->full_attn_interval != 0);
        cl->is_ssm = is_ssm;
        cl->is_swa = 0;
        cl->shared_kv_source = -1;
        cl->n_kv_heads = r->n_kv_heads;
        if (r->is_gemma4) {
            cl->is_swa = r->swa_pattern[l];
            if (r->per_layer_kv_heads) cl->n_kv_heads = r->per_layer_kv_heads[l];
            if (l >= r->n_layer_kv_from_start)
                cl->shared_kv_source = r->n_layer_kv_from_start - (cl->is_swa ? 2 : 1);
        }

        /* Attention norm (F32) — shared by all layer types */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        qtensor t = cllm_load_tensor(gguf, name, 1);
        if (upload_norm_f32(r, &cl->attn_norm_w, &t, r->n_embd) != 0) return -1;

        if (r->verbose >= 2) fprintf(stderr, "cuda_llm: layer %d (%s)...\n", l, is_ssm ? "SSM" : "ATT");
        if (is_ssm) {
            /* --- SSM (Delta-Net) layer weights --- */
            #define LOAD_SSM_W(field, suffix, rows_f, cols_f, type_f) do { \
                snprintf(name, sizeof(name), "blk.%d." suffix ".weight", l); \
                t = cllm_load_tensor(gguf, name, 1); \
                cl->rows_f = t.n_rows; cl->cols_f = t.n_cols; \
                if (upload_weight_matrix(r, &cl->field, &t, &cl->type_f, name) != 0) return -1; \
            } while(0)
            LOAD_SSM_W(ssm_qkv_w,   "attn_qkv",  ssm_qkv_rows,   ssm_qkv_cols,   ssm_qkv_type);
            if (upload_q8_0_shadow_f16(r, &cl->ssm_qkv_w_f16, &t) != 0) return -1;
            LOAD_SSM_W(ssm_gate_w,   "attn_gate",  ssm_gate_rows,  ssm_gate_cols,  ssm_gate_type);
            if (upload_q8_0_shadow_f16(r, &cl->ssm_gate_w_f16, &t) != 0) return -1;
            LOAD_SSM_W(ssm_alpha_w,  "ssm_alpha",  ssm_alpha_rows, ssm_alpha_cols, ssm_alpha_type);
            LOAD_SSM_W(ssm_beta_w,   "ssm_beta",   ssm_beta_rows,  ssm_beta_cols,  ssm_beta_type);
            LOAD_SSM_W(ssm_out_w,    "ssm_out",    ssm_out_rows,   ssm_out_cols,   ssm_out_type);
            if (upload_q8_0_shadow_f16(r, &cl->ssm_out_w_f16, &t) != 0) return -1;
            #undef LOAD_SSM_W

            /* ssm_a (F32, no .weight suffix) */
            snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(r, &cl->ssm_a, &t, r->ssm_dt_rank) != 0) return -1;

            /* ssm_dt.bias (F32) */
            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(r, &cl->ssm_dt_bias, &t, r->ssm_dt_rank) != 0) return -1;

            /* ssm_conv1d (F32) */
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(r, &cl->ssm_conv1d_w, &t, t.n_rows * t.n_cols) != 0) return -1;

            /* ssm_norm (F32) */
            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(r, &cl->ssm_norm_w, &t, r->ssm_d_state) != 0) return -1;

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
            if (upload_weight_matrix_bm(r, &cl->attn_q_w, &cl->attn_q_w_bm, &t, &cl->attn_q_type, name) != 0) return -1;
            if (upload_q8_0_shadow_f16(r, &cl->attn_q_w_f16, &t) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->attn_k_rows = t.n_rows; cl->attn_k_cols = t.n_cols;
            if (upload_weight_matrix_bm(r, &cl->attn_k_w, &cl->attn_k_w_bm, &t, &cl->attn_k_type, name) != 0) return -1;
            if (upload_q8_0_shadow_f16(r, &cl->attn_k_w_f16, &t) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
            t = cllm_load_tensor(gguf, name, r->is_gemma4 ? 0 : 1);
            cl->has_v_proj = (t.data != NULL);
            if (cl->has_v_proj) {
                cl->attn_v_rows = t.n_rows; cl->attn_v_cols = t.n_cols;
                if (upload_weight_matrix_bm(r, &cl->attn_v_w, &cl->attn_v_w_bm, &t, &cl->attn_v_type, name) != 0) return -1;
                if (upload_q8_0_shadow_f16(r, &cl->attn_v_w_f16, &t) != 0) return -1;
            } else {
                /* V = K for this layer (Gemma4 pattern) */
                cl->attn_v_rows = cl->attn_k_rows;
                cl->attn_v_cols = cl->attn_k_cols;
                cl->attn_v_w = 0;
                cl->attn_v_type = 0;
                if (r->verbose >= 2)
                    fprintf(stderr, "cuda_llm: layer %d: no V tensor, will use K as V\n", l);
            }

            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->attn_output_rows = t.n_rows; cl->attn_output_cols = t.n_cols;
            if (upload_weight_matrix_bm(r, &cl->attn_output_w, &cl->attn_output_w_bm, &t, &cl->attn_output_type, name) != 0) return -1;
            if (upload_q8_0_shadow_f16(r, &cl->attn_output_w_f16, &t) != 0) return -1;

            /* QK norms (F32, optional) */
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
            t = cllm_load_tensor(gguf, name, 0);
            cl->has_qk_norm = (t.data != NULL);
            if (t.data) {
                if (upload_norm_f32(r, &cl->attn_q_norm_w, &t, r->head_dim) != 0) return -1;
            }
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            t = cllm_load_tensor(gguf, name, 0);
            if (t.data) {
                if (upload_norm_f32(r, &cl->attn_k_norm_w, &t, r->head_dim) != 0) return -1;
            }

            /* Q/K/V biases (F32, optional — Qwen2.5-VL has these) */
            snprintf(name, sizeof(name), "blk.%d.attn_q.bias", l);
            t = cllm_load_tensor(gguf, name, 0);
            if (t.data) {
                size_t sz = (size_t)t.n_cols * sizeof(float);
                cuMemAlloc(&cl->attn_q_bias, sz);
                /* Dequant to F32 and upload */
                float *tmp = (float *)malloc(sz);
                dequant_row(t.type, t.data, tmp, t.n_cols);
                cuMemcpyHtoD(cl->attn_q_bias, tmp, sz);
                free(tmp);
            }
            snprintf(name, sizeof(name), "blk.%d.attn_k.bias", l);
            t = cllm_load_tensor(gguf, name, 0);
            if (t.data) {
                size_t sz = (size_t)t.n_cols * sizeof(float);
                cuMemAlloc(&cl->attn_k_bias, sz);
                float *tmp = (float *)malloc(sz);
                dequant_row(t.type, t.data, tmp, t.n_cols);
                cuMemcpyHtoD(cl->attn_k_bias, tmp, sz);
                free(tmp);
            }
            snprintf(name, sizeof(name), "blk.%d.attn_v.bias", l);
            t = cllm_load_tensor(gguf, name, 0);
            if (t.data) {
                size_t sz = (size_t)t.n_cols * sizeof(float);
                cuMemAlloc(&cl->attn_v_bias, sz);
                float *tmp = (float *)malloc(sz);
                dequant_row(t.type, t.data, tmp, t.n_cols);
                cuMemcpyHtoD(cl->attn_v_bias, tmp, sz);
                free(tmp);
            }

            if (r->verbose >= 2) {
                fprintf(stderr, "  layer %d [ATT]: Q[%d×%d] K[%d×%d] V[%d×%d] O[%d×%d] qk_norm=%d bias=%d\n",
                        l, cl->attn_q_rows, cl->attn_q_cols,
                        cl->attn_k_rows, cl->attn_k_cols,
                        cl->attn_v_rows, cl->attn_v_cols,
                        cl->attn_output_rows, cl->attn_output_cols,
                        cl->has_qk_norm, cl->attn_q_bias ? 1 : 0);
            }
        }

        /* FFN norm (F32) — shared by all layer types */
        if (r->is_hybrid && !r->is_gemma4) {
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l);
        } else {
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        }
        t = cllm_load_tensor(gguf, name, 1);
        if (upload_norm_f32(r, &cl->ffn_norm_w, &t, r->n_embd) != 0) return -1;

        /* FFN weights */
        if (r->is_moe) {
            /* --- MoE FFN weights --- */
            cl->is_moe = 1;

            /* Router: ffn_gate_inp [n_experts, n_embd] F32 */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_gate_rows = t.n_rows; cl->moe_gate_cols = t.n_cols;
            if (upload_norm_f32(r, &cl->moe_gate_w, &t, t.n_rows * t.n_cols) != 0) return -1;

            /* Expert 3D weights (K-quant packed) */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_gate_exps_type = t.type;
            cl->moe_exp_cols_gu = t.n_cols;  /* n_embd (input dim) */
            cl->moe_exp_rows_gu = (t.n_dims >= 3) ? (int)t.dims[1] : t.n_rows;  /* expert_ff */
            if (upload_3d_kquant_raw_ex(&cl->moe_gate_exps_w, &t, &cl->moe_exp_stride_gu, moe_repack_bm) != 0) return -1;
            /* F16 shadow for cuBLAS */
            if (r->use_cublas) {
                upload_3d_kquant_f16(r, &cl->moe_gate_exps_w_f16, &t,
                                     cl->moe_exp_rows_gu, cl->moe_exp_cols_gu,
                                     cl->moe_exp_stride_gu);
            } else {
                cl->moe_gate_exps_w_f16 = 0;
            }

            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_up_exps_type = t.type;
            if (upload_3d_kquant_raw_ex(&cl->moe_up_exps_w, &t, &cl->moe_exp_stride_gu, moe_repack_bm) != 0) return -1;
            if (r->use_cublas) {
                upload_3d_kquant_f16(r, &cl->moe_up_exps_w_f16, &t,
                                     cl->moe_exp_rows_gu, cl->moe_exp_cols_gu,
                                     cl->moe_exp_stride_gu);
            } else {
                cl->moe_up_exps_w_f16 = 0;
            }

            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_down_exps_type = t.type;
            cl->moe_exp_cols_d = t.n_cols;   /* expert_ff (input dim) */
            cl->moe_exp_rows_d = (t.n_dims >= 3) ? (int)t.dims[1] : t.n_rows;  /* n_embd */
            if (upload_3d_kquant_raw_ex(&cl->moe_down_exps_w, &t, &cl->moe_exp_stride_d, moe_repack_bm) != 0) return -1;
            if (r->use_cublas) {
                upload_3d_kquant_f16(r, &cl->moe_down_exps_w_f16, &t,
                                     cl->moe_exp_rows_d, cl->moe_exp_cols_d,
                                     cl->moe_exp_stride_d);
            } else {
                cl->moe_down_exps_w_f16 = 0;
            }

            /* Shared expert gate: ffn_gate_inp_shexp [n_embd] F32 (1D sigmoid gate) */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(r, &cl->moe_shared_gate_w, &t, t.n_rows * t.n_cols) != 0) return -1;

            /* Shared expert FFN (BF16 → F16 via upload_weight_matrix) */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_shared_gate_rows = t.n_rows; cl->moe_shared_gate_cols = t.n_cols;
            if (upload_weight_matrix(r, &cl->moe_shared_ffn_gate_w, &t, &cl->moe_shared_gate_type, name) != 0) return -1;
            if (upload_q8_0_shadow_f16(r, &cl->moe_shared_ffn_gate_w_f16, &t) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_shared_up_rows = t.n_rows; cl->moe_shared_up_cols = t.n_cols;
            if (upload_weight_matrix(r, &cl->moe_shared_ffn_up_w, &t, &cl->moe_shared_up_type, name) != 0) return -1;
            if (upload_q8_0_shadow_f16(r, &cl->moe_shared_ffn_up_w_f16, &t) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->moe_shared_down_rows = t.n_rows; cl->moe_shared_down_cols = t.n_cols;
            if (upload_weight_matrix(r, &cl->moe_shared_ffn_down_w, &t, &cl->moe_shared_down_type, name) != 0) return -1;
            if (upload_q8_0_shadow_f16(r, &cl->moe_shared_ffn_down_w_f16, &t) != 0) return -1;

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
            if (upload_weight_matrix_bm(r, &cl->ffn_gate_w, &cl->ffn_gate_w_bm, &t, &cl->ffn_gate_type, name) != 0) return -1;
            if (upload_q8_0_shadow_f16(r, &cl->ffn_gate_w_f16, &t) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->ffn_up_rows = t.n_rows; cl->ffn_up_cols = t.n_cols;
            if (upload_weight_matrix_bm(r, &cl->ffn_up_w, &cl->ffn_up_w_bm, &t, &cl->ffn_up_type, name) != 0) return -1;
            if (upload_q8_0_shadow_f16(r, &cl->ffn_up_w_f16, &t) != 0) return -1;

            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            cl->ffn_down_rows = t.n_rows; cl->ffn_down_cols = t.n_cols;
            if (upload_weight_matrix_bm(r, &cl->ffn_down_w, &cl->ffn_down_w_bm, &t, &cl->ffn_down_type, name) != 0) return -1;
            if (upload_q8_0_shadow_f16(r, &cl->ffn_down_w_f16, &t) != 0) return -1;
        }

        /* Gemma4 per-layer extra tensors */
        if (r->is_gemma4) {
            /* Post-attention norm */
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(r, &cl->post_attn_norm_w, &t, r->n_embd) != 0) return -1;

            /* Post-FFW norm */
            snprintf(name, sizeof(name), "blk.%d.post_ffw_norm.weight", l);
            t = cllm_load_tensor(gguf, name, 1);
            if (upload_norm_f32(r, &cl->post_ffw_norm_w, &t, r->n_embd) != 0) return -1;

            /* Layer output scale (cached as float) */
            snprintf(name, sizeof(name), "blk.%d.layer_output_scale.weight", l);
            t = cllm_load_tensor(gguf, name, 0);
            cl->layer_scale_val = 1.0f;
            if (t.data) {
                float sv;
                dequant_row(t.type, t.data, &sv, 1);
                cl->layer_scale_val = sv;
            }

            /* Per-layer embedding tensors (only if model has PLE, i.e. n_embd_per_layer > 0) */
            if (r->n_embd_per_layer > 0) {
                snprintf(name, sizeof(name), "blk.%d.inp_gate.weight", l);
                t = cllm_load_tensor(gguf, name, 1);
                cl->ple_inp_gate_rows = t.n_rows; cl->ple_inp_gate_cols = t.n_cols;
                if (r->ple_use_f32) {
                    cl->ple_inp_gate_type = GGML_TYPE_F32;
                    if (upload_weight_f32(&cl->ple_inp_gate_w, &t) != 0) return -1;
                } else {
                    if (upload_weight_matrix(r, &cl->ple_inp_gate_w, &t, &cl->ple_inp_gate_type, name) != 0) return -1;
                }

                snprintf(name, sizeof(name), "blk.%d.proj.weight", l);
                t = cllm_load_tensor(gguf, name, 1);
                cl->ple_proj_rows = t.n_rows; cl->ple_proj_cols = t.n_cols;
                if (r->ple_use_f32) {
                    cl->ple_proj_type = GGML_TYPE_F32;
                    if (upload_weight_f32(&cl->ple_proj_w, &t) != 0) return -1;
                } else {
                    if (upload_weight_matrix(r, &cl->ple_proj_w, &t, &cl->ple_proj_type, name) != 0) return -1;
                }

                /* PLE post-norm (F32) */
                snprintf(name, sizeof(name), "blk.%d.post_norm.weight", l);
                t = cllm_load_tensor(gguf, name, 1);
                if (upload_norm_f32(r, &cl->ple_post_norm_w, &t, r->n_embd) != 0) return -1;
            }

            /* QK norm for Gemma4 uses per-layer head_dim */
            if (cl->has_qk_norm) {
                int hd = cl->is_swa ? r->head_dim_swa : r->head_dim_full;
                /* Re-upload if head_dim differs — already uploaded with r->head_dim above,
                 * but Gemma4 head_dim varies. Re-upload with correct size. */
                if (hd != r->head_dim) {
                    if (cl->attn_q_norm_w) cuMemFree(cl->attn_q_norm_w);
                    if (cl->attn_k_norm_w) cuMemFree(cl->attn_k_norm_w);
                    cl->attn_q_norm_w = 0;
                    cl->attn_k_norm_w = 0;
                    snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
                    t = cllm_load_tensor(gguf, name, 0);
                    if (t.data && upload_norm_f32(r, &cl->attn_q_norm_w, &t, hd) != 0) return -1;
                    snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
                    t = cllm_load_tensor(gguf, name, 0);
                    if (t.data && upload_norm_f32(r, &cl->attn_k_norm_w, &t, hd) != 0) return -1;
                }
            }
        }
    }

    /* Gemma4: load global per-layer embedding tensors and allocate PLE scratch */
    if (r->is_gemma4 && r->n_embd_per_layer > 0) {
        int ple_dim = r->n_embd_per_layer;
        int total_ple = ple_dim * r->n_layers;
        /* Store host-side PLE tensor pointers for CPU precomputation */
        r->h_ple_token_embd = cllm_load_tensor(gguf, "per_layer_token_embd.weight", 1);
        r->h_ple_model_proj = cllm_load_tensor(gguf, "per_layer_model_proj.weight", 1);
        r->h_ple_proj_norm = cllm_load_tensor(gguf, "per_layer_proj_norm.weight", 0);
        if (!r->h_ple_token_embd.data || !r->h_ple_model_proj.data) {
            fprintf(stderr, "cuda_llm: Gemma4 missing required global PLE tensors\n");
            return -1;
        }

        CHECK_CU(cuMemAlloc(&r->d_ple_combined, (size_t)total_ple * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ple_buf, (size_t)ple_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ple_proj, (size_t)r->n_embd * sizeof(float)));

        if (r->verbose >= 1)
            fprintf(stderr, "cuda_llm: Gemma4 PLE buffers allocated (ple_dim=%d, tok_embd=%s, proj=%s)\n",
                    ple_dim, r->h_ple_token_embd.data ? "OK" : "MISSING",
                    r->h_ple_model_proj.data ? "OK" : "MISSING");
    } else if (r->is_gemma4 && r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: Gemma4 model has no PLE (n_embd_per_layer=0)\n");
    }

    /* Allocate KV cache (skip SSM layers; Gemma4 has per-layer sizing) */
    r->d_key_cache = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
    r->d_value_cache = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
    r->kv_cache_q8 = getenv("CUDA_LLM_KV_CACHE_Q8") && !getenv("CUDA_LLM_NO_KV_CACHE_Q8");
    if (r->kv_cache_q8) {
        r->d_key_cache_q8 = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
        r->d_value_cache_q8 = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
        r->d_key_scale = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
        r->d_value_scale = (CUdeviceptr *)calloc(r->n_layers, sizeof(CUdeviceptr));
    } else {
        r->d_key_cache_q8 = NULL; r->d_value_cache_q8 = NULL;
        r->d_key_scale = NULL; r->d_value_scale = NULL;
    }
    int n_attn_layers = 0;
    if (r->is_gemma4) {
        for (int l = 0; l < r->n_layers; l++) {
            if (r->layers[l].shared_kv_source >= 0) continue;
            int layer_kv_heads = r->layers[l].n_kv_heads;
            int hd = r->layers[l].is_swa ? r->head_dim_swa : r->head_dim_full;
            int kv_dim = layer_kv_heads * hd;
            size_t cache_sz;
            if (r->layers[l].is_swa) {
                cache_sz = (size_t)r->swa_window_size * kv_dim * sizeof(uint16_t);
            } else {
                cache_sz = (size_t)max_seq_len * kv_dim * sizeof(uint16_t);
            }
            if (r->kv_cache_q8) {
                size_t q8_sz = (size_t)max_seq_len * kv_dim * sizeof(signed char);
                size_t scale_sz = (size_t)max_seq_len * layer_kv_heads * sizeof(float);
                CHECK_CU(cuMemAlloc(&r->d_key_cache_q8[l], q8_sz));
                CHECK_CU(cuMemAlloc(&r->d_value_cache_q8[l], q8_sz));
                CHECK_CU(cuMemAlloc(&r->d_key_scale[l], scale_sz));
                CHECK_CU(cuMemAlloc(&r->d_value_scale[l], scale_sz));
                /* Point FP16 cache to Q8 (prevents allocation; FP16 path unused) */
                r->d_key_cache[l] = r->d_key_cache_q8[l];
                r->d_value_cache[l] = r->d_value_cache_q8[l];
                cuMemsetD8(r->d_key_scale[l], 0, scale_sz);
                cuMemsetD8(r->d_value_scale[l], 0, scale_sz);
            } else {
                CHECK_CU(cuMemAlloc(&r->d_key_cache[l], cache_sz));
                CHECK_CU(cuMemsetD8(r->d_key_cache[l], 0, cache_sz));
                CHECK_CU(cuMemAlloc(&r->d_value_cache[l], cache_sz));
                CHECK_CU(cuMemsetD8(r->d_value_cache[l], 0, cache_sz));
            }
            n_attn_layers++;
        }
        /* Point shared layers to source */
        for (int l = 0; l < r->n_layers; l++) {
            int src = r->layers[l].shared_kv_source;
            if (src >= 0 && src < r->n_layers) {
                r->d_key_cache[l] = r->d_key_cache[src];
                r->d_value_cache[l] = r->d_value_cache[src];
                if (r->kv_cache_q8) {
                    r->d_key_cache_q8[l] = r->d_key_cache_q8[src];
                    r->d_value_cache_q8[l] = r->d_value_cache_q8[src];
                    r->d_key_scale[l] = r->d_key_scale[src];
                    r->d_value_scale[l] = r->d_value_scale[src];
                }
            }
        }
        if (r->verbose >= 1) {
            int n_shared = r->n_layers - r->n_layer_kv_from_start;
            fprintf(stderr, "cuda_llm: Gemma4 KV cache: %d own, %d shared\n", n_attn_layers, n_shared);
        }
    } else {
        size_t kv_cache_size = (size_t)max_seq_len * kv_dim * sizeof(uint16_t);
        for (int l = 0; l < r->n_layers; l++) {
            if (r->layers[l].is_ssm) continue;
            CHECK_CU(cuMemAlloc(&r->d_key_cache[l], kv_cache_size));
            CHECK_CU(cuMemsetD8(r->d_key_cache[l], 0, kv_cache_size));
            CHECK_CU(cuMemAlloc(&r->d_value_cache[l], kv_cache_size));
            CHECK_CU(cuMemsetD8(r->d_value_cache[l], 0, kv_cache_size));
            n_attn_layers++;
        }
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
    if (r->is_moe) {
        int tc_down_dim = r->n_experts_used * r->n_embd;
        if (tc_down_dim > xb2_dim) xb2_dim = tc_down_dim;
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
        int gate_dim = ff_dim;
        if (r->is_moe) {
            int tc_dim = r->n_experts_used * r->expert_ff;
            if (tc_dim > gate_dim) gate_dim = tc_dim;
        }
        CHECK_CU(cuMemAlloc(&r->d_gate, gate_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_up,   gate_dim * sizeof(float)));
    }

    /* MoE scratch buffers */
    if (r->is_moe) {
        CHECK_CU(cuMemAlloc(&r->d_router_logits, r->n_experts * sizeof(float)));
        r->d_router_logits_entries = r->n_experts;
        CHECK_CU(cuMemAlloc(&r->d_moe_accum, r->n_embd * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_moe_f16w, (size_t)r->expert_ff * r->n_embd * sizeof(uint16_t)));
        CHECK_CU(cuMemAlloc(&r->d_moe_f16w2, (size_t)r->expert_ff * r->n_embd * sizeof(uint16_t)));
        CHECK_CU(cuMemAlloc(&r->d_moe_f16w3, (size_t)r->expert_ff * r->n_embd * sizeof(uint16_t)));
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
    CHECK_CU(cuMemAlloc(&r->d_xb_q,      max_dim * sizeof(int8_t)));
    CHECK_CU(cuMemAlloc(&r->d_xb_scale,  sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_xb_q2,     max_dim * sizeof(int8_t)));
    CHECK_CU(cuMemAlloc(&r->d_xb_scale2, sizeof(float)));
    {
        size_t q81_bytes = ((size_t)max_dim / 32 + 1) * 36;
        CHECK_CU(cuMemAlloc(&r->d_xb_q81,   q81_bytes));
        CHECK_CU(cuMemAlloc(&r->d_xb_q81_2, q81_bytes));
    }
    CHECK_CU(cuMemAlloc(&r->d_hidden_snapshots, (size_t)3 * r->n_embd * sizeof(float)));

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
        size_t kv_cache_size_est = (size_t)max_seq_len * kv_dim * sizeof(uint16_t); /* F16 KV cache */
        size_t cache_bytes = (size_t)r->n_layers * 2 * kv_cache_size_est;
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

int cuda_llm_load_weights_qwen3_safetensors(cuda_llm_runner *r,
                                            const char *model_path,
                                            int max_seq_len) {
    if (!r || !model_path) return -1;

    cllm_f16cache_begin(model_path, r->verbose);

    st_context *shards[16] = {0};
    int n_shards = cllm_open_safetensors_shards(model_path, shards, 16);
    if (n_shards <= 0) {
        fprintf(stderr, "cuda_llm: failed to open safetensors model %s\n", model_path);
        cllm_f16cache_end(0, r->verbose);
        return -1;
    }

    int n_layers = 0;
    for (int s = 0; s < n_shards; s++) {
        for (int i = 0; i < shards[s]->n_tensors; i++) {
            const char *nm = safetensors_name(shards[s], i);
            if (strncmp(nm, "model.layers.", 13) == 0) {
                int l = atoi(nm + 13);
                if (l + 1 > n_layers) n_layers = l + 1;
            }
        }
    }

    int r0, c0;
    qtensor embd = cllm_st_load_tensor_multi(shards, n_shards, "model.embed_tokens.weight", 1);
    if (!embd.data) { cllm_close_safetensors_shards(shards, n_shards); cllm_f16cache_end(0, r->verbose); return -1; }
    qtensor q0 = cllm_st_load_tensor_multi(shards, n_shards, "model.layers.0.self_attn.q_proj.weight", 1);
    qtensor k0 = cllm_st_load_tensor_multi(shards, n_shards, "model.layers.0.self_attn.k_proj.weight", 1);
    qtensor qn0 = cllm_st_load_tensor_multi(shards, n_shards, "model.layers.0.self_attn.q_norm.weight", 0);
    qtensor ff0 = cllm_st_load_tensor_multi(shards, n_shards, "model.layers.0.mlp.gate_proj.weight", 1);
    (void)r0; (void)c0;

    r->n_embd = embd.n_cols;
    r->n_vocab = embd.n_rows;
    r->n_layers = n_layers;
    r->head_dim = qn0.n_cols ? qn0.n_cols : 128;
    r->n_heads = q0.n_rows / r->head_dim;
    r->n_kv_heads = k0.n_rows / r->head_dim;
    r->n_ff = ff0.n_rows;
    r->rms_norm_eps = 1e-6f;
    r->rope_freq_base = 1000000.0f;
    r->n_rope_pairs = 0;
    r->is_hybrid = 0;
    r->is_moe = 0;
    r->n_deepstack = 0;
    if (max_seq_len <= 0) max_seq_len = 2048;
    r->max_seq_len = max_seq_len;

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: safetensors qwen3 n_embd=%d n_heads=%d n_kv_heads=%d n_layers=%d n_ff=%d head_dim=%d\n",
                r->n_embd, r->n_heads, r->n_kv_heads, r->n_layers, r->n_ff, r->head_dim);
    }

    if (upload_weight_matrix(r, &r->d_token_embd, &embd, &r->token_embd_type,
                             "model.embed_tokens.weight") != 0) {
        cllm_close_safetensors_shards(shards, n_shards);
        cllm_f16cache_end(0, r->verbose);
        return -1;
    }

    qtensor onorm = cllm_st_load_tensor_multi(shards, n_shards, "model.norm.weight", 1);
    if (!onorm.data || upload_norm_f32(r, &r->d_output_norm, &onorm, r->n_embd) != 0) {
        cllm_close_safetensors_shards(shards, n_shards);
        cllm_f16cache_end(0, r->verbose);
        return -1;
    }

    r->d_output_w = r->d_token_embd;
    r->output_w_type = r->token_embd_type;
    r->has_lm_head = 1;

    r->layers = (cuda_layer *)calloc(r->n_layers, sizeof(cuda_layer));
    if (!r->layers) {
        cllm_close_safetensors_shards(shards, n_shards);
        cllm_f16cache_end(0, r->verbose);
        return -1;
    }

    for (int l = 0; l < r->n_layers; l++) {
        char name[160];
        cuda_layer *cl = &r->layers[l];
        qtensor t;

        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 1);
        if (!t.data || upload_norm_f32(r, &cl->attn_norm_w, &t, r->n_embd) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 1);
        cl->attn_q_rows = t.n_rows; cl->attn_q_cols = t.n_cols;
        if (!t.data || upload_weight_matrix(r, &cl->attn_q_w, &t, &cl->attn_q_type, name) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 1);
        cl->attn_k_rows = t.n_rows; cl->attn_k_cols = t.n_cols;
        if (!t.data || upload_weight_matrix(r, &cl->attn_k_w, &t, &cl->attn_k_type, name) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 1);
        cl->attn_v_rows = t.n_rows; cl->attn_v_cols = t.n_cols;
        if (!t.data || upload_weight_matrix(r, &cl->attn_v_w, &t, &cl->attn_v_type, name) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 1);
        cl->attn_output_rows = t.n_rows; cl->attn_output_cols = t.n_cols;
        if (!t.data || upload_weight_matrix(r, &cl->attn_output_w, &t, &cl->attn_output_type, name) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 0);
        cl->has_qk_norm = (t.data != NULL);
        if (t.data && upload_norm_f32(r, &cl->attn_q_norm_w, &t, r->head_dim) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 0);
        if (t.data) cl->has_qk_norm = 1;
        if (t.data && upload_norm_f32(r, &cl->attn_k_norm_w, &t, r->head_dim) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 1);
        if (!t.data || upload_norm_f32(r, &cl->ffn_norm_w, &t, r->n_embd) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 1);
        cl->ffn_gate_rows = t.n_rows; cl->ffn_gate_cols = t.n_cols;
        if (!t.data || upload_weight_matrix(r, &cl->ffn_gate_w, &t, &cl->ffn_gate_type, name) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 1);
        cl->ffn_up_rows = t.n_rows; cl->ffn_up_cols = t.n_cols;
        if (!t.data || upload_weight_matrix(r, &cl->ffn_up_w, &t, &cl->ffn_up_type, name) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }

        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", l);
        t = cllm_st_load_tensor_multi(shards, n_shards, name, 1);
        cl->ffn_down_rows = t.n_rows; cl->ffn_down_cols = t.n_cols;
        if (!t.data || upload_weight_matrix(r, &cl->ffn_down_w, &t, &cl->ffn_down_type, name) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }
    }

    {
        int kv_dim = r->n_kv_heads * r->head_dim;
        int q_dim = r->n_heads * r->head_dim;
        if (cuda_llm_alloc_runtime_buffers(r, q_dim, kv_dim, max_seq_len) != 0) {
            cllm_close_safetensors_shards(shards, n_shards);
            cllm_f16cache_end(0, r->verbose);
            return -1;
        }
    }

    cllm_close_safetensors_shards(shards, n_shards);
    cllm_f16cache_end(1, r->verbose);
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
    /* 1024 threads (32 warps) hides the load latency of the single-block reduction
       far better than 256 (n_embd=5376 -> ~5 elems/thread). */
    int nthr = (n >= 1024) ? 1024 : 256;
    cuLaunchKernel(r->fn_rmsnorm_f32,
                   1, 1, 1,
                   nthr, 1, 1,
                   nthr * sizeof(float), r->stream, args, NULL);
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
    /* blockDim = next power of 2 >= head_dim (Gemma4 needs 512 for full-attn heads) */
    int bdim = 1;
    while (bdim < head_dim) bdim <<= 1;
    if (bdim > 1024) bdim = 1024;
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
    int use_ptr = !r->disable_graph && r->d_pos_seq;
    /* Use pointer variant only for graph capture/replay. */
    CUfunction fn = use_ptr ? r->fn_rope_neox_f32_ptr : r->fn_rope_neox_f32;
    void *pos_arg = use_ptr ? (void*)&r->d_pos_seq : (void*)&pos;
    void *args[] = { &vec, &n_heads, &head_dim, pos_arg, &freq_base, &n_rope_pairs };
    cuLaunchKernel(fn, n_heads, 1, 1, half_dim, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_kv_store(cuda_llm_runner *r, CUdeviceptr key_cache, CUdeviceptr value_cache,
                                    CUdeviceptr k, CUdeviceptr v, int position, int kv_dim) {
    if (r->kv_cache_q8) return; /* Q8 path handles store separately */
    int use_ptr = !r->disable_graph && r->d_pos_seq;
    CUfunction fn = use_ptr ? r->fn_kv_cache_store_f16_ptr : r->fn_kv_cache_store_f16;
    void *pos_arg = use_ptr ? (void*)&r->d_pos_seq : (void*)&position;
    void *args[] = { &key_cache, &value_cache, &k, &v, pos_arg, &kv_dim };
    cuLaunchKernel(fn, (kv_dim + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_kv_store_q8(cuda_llm_runner *r, int layer,
                                       CUdeviceptr k, CUdeviceptr v, int position, int kv_dim) {
    if (!r->kv_cache_q8 || !r->fn_kv_cache_store_q8) return;
    cuda_layer *cl = &r->layers[layer];
    int n_kv_heads = cl->n_kv_heads;
    int hd = cl->is_swa ? r->head_dim_swa : r->head_dim_full;
    void *a[] = { &r->d_key_cache_q8[layer], &r->d_value_cache_q8[layer],
                  &r->d_key_scale[layer], &r->d_value_scale[layer],
                  &k, &v, &position, &kv_dim, &n_kv_heads, &hd };
    cuLaunchKernel(r->fn_kv_cache_store_q8, n_kv_heads, 1, 1, hd, 1, 1, 0, r->stream, a, NULL);
}

static inline void launch_attention(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr q,
                                     CUdeviceptr key_cache, CUdeviceptr value_cache,
                                     int n_heads, int n_kv_heads, int head_dim,
                                     int kv_dim, int seq_len, float scale) {
    int use_ptr = !r->disable_graph && r->d_pos_seq;
    /* Use pointer variant only for graph capture/replay. */
    CUfunction fn = use_ptr ? r->fn_attn_decode_f32_ptr : r->fn_attn_decode_f32;
    /* seq_len is at d_seq_ptr (d_pos_seq[1]), pre-computed for graph-capture persistence */
    void *sl_arg = use_ptr ? (void*)&r->d_seq_ptr : (void*)&seq_len;
    size_t smem = (use_ptr ? (size_t)r->max_seq_len : seq_len) * sizeof(float);
    void *args[] = { &out, &q, &key_cache, &value_cache,
                     &n_heads, &n_kv_heads, &head_dim, &kv_dim, sl_arg, &scale };
    cuLaunchKernel(fn, n_heads, 1, 1, 256, 1, 1, (unsigned int)smem, r->stream, args, NULL);
}

/* Q8 quantized KV cache attention (INT8, per-head scale). Replaces FP16 attention. */
static inline void launch_attention_q8(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr q,
                                        CUdeviceptr key_cache_q8, CUdeviceptr value_cache_q8,
                                        CUdeviceptr key_scale, CUdeviceptr value_scale,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        int kv_dim, int seq_len, float scale) {
    if (!r->fn_attn_decode_q8) return;
    size_t smem = (size_t)seq_len * sizeof(float);
    void *args[] = { &out, &q, &key_cache_q8, &value_cache_q8,
                     &key_scale, &value_scale,
                     &n_heads, &n_kv_heads, &head_dim, &kv_dim, &seq_len, &scale };
    cuLaunchKernel(r->fn_attn_decode_q8, n_heads, 1, 1, 256, 1, 1, (unsigned int)smem, r->stream, args, NULL);
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

/* ---- Gemma4 launch helpers ---- */

static inline void launch_gelu_mul(cuda_llm_runner *r, CUdeviceptr gate, CUdeviceptr up, int n) {
    void *args[] = { &gate, &up, &n };
    cuLaunchKernel(r->fn_gelu_mul_f32, (n+255)/256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_gelu_elementwise_mul(cuda_llm_runner *r, CUdeviceptr ple, CUdeviceptr gate, int n) {
    void *args[] = { &ple, &gate, &n };
    cuLaunchKernel(r->fn_gelu_elementwise_mul_f32, (n+255)/256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_scale(cuda_llm_runner *r, CUdeviceptr x, float scale_val, int n) {
    void *args[] = { &x, &scale_val, &n };
    cuLaunchKernel(r->fn_scale_f32, (n+255)/256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_logit_softcap(cuda_llm_runner *r, CUdeviceptr x, int n, float cap) {
    void *args[] = { &x, &n, &cap };
    cuLaunchKernel(r->fn_logit_softcap_f32, (n+255)/256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_attention_swa(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr q,
                                         CUdeviceptr key_cache, CUdeviceptr value_cache,
                                         int n_heads, int n_kv_heads, int head_dim,
                                         int kv_dim, int position, int window_size, float scale) {
    int start = (position >= window_size) ? (position - window_size + 1) : 0;
    int seq_len = position - start + 1;
    int use_ptr = !r->disable_graph && r->d_pos_seq;
    CUfunction fn = use_ptr ? r->fn_attn_decode_swa_f32_ptr : r->fn_attn_decode_swa_f32;
    /* For ptr variant (graph capture), allocate max possible shared memory so that
       the captured smem is large enough for any replay position within the window. */
    size_t smem = use_ptr ? ((size_t)window_size * sizeof(float)) : ((size_t)seq_len * sizeof(float));
    void *pos_arg = use_ptr ? (void*)&r->d_pos_seq : (void*)&position;
    void *args[] = { &out, &q, &key_cache, &value_cache,
                     &n_heads, &n_kv_heads, &head_dim, &kv_dim,
                     pos_arg, &window_size, &scale };
    cuLaunchKernel(fn, n_heads, 1, 1, 256, 1, 1, smem, r->stream, args, NULL);
}

static inline void launch_rope_with_factors(cuda_llm_runner *r, CUdeviceptr vec, int n_heads,
                                             int head_dim, int pos, CUdeviceptr inv_freq) {
    int n_pairs = head_dim / 2;
    int use_ptr = !r->disable_graph && r->d_pos_seq;
    CUfunction fn = use_ptr ? r->fn_rope_with_factors_f32_ptr : r->fn_rope_with_factors_f32;
    void *pos_arg = use_ptr ? (void*)&r->d_pos_seq : (void*)&pos;
    void *args[] = { &vec, &n_heads, &head_dim, pos_arg, &inv_freq, &n_pairs };
    cuLaunchKernel(fn, n_heads, 1, 1, n_pairs, 1, 1, 0, r->stream, args, NULL);
}

/* Batched version: processes n_tokens * n_heads heads with position p0 for all */
static inline void launch_batch_rope_with_factors(cuda_llm_runner *r, CUdeviceptr vec,
                                                   int heads_per_token, int n_tokens,
                                                   int head_dim, int start_pos,
                                                   CUdeviceptr inv_freq) {
    int total_heads = heads_per_token * n_tokens;
    int n_pairs = head_dim / 2;
    /* Per-token batched kernel: token = blockIdx.x / heads_per_token, pos =
     * start_pos + token. The old fn_rope_with_factors_f32 applied a single
     * constant position to every block — correct only for n_tokens==1, and a
     * latent bug for batched prefill (every token roped at start_pos). */
    void *args[] = { &vec, &heads_per_token, &total_heads, &head_dim, &start_pos, &inv_freq, &n_pairs };
    cuLaunchKernel(r->fn_batch_rope_with_factors_f32,
                   total_heads, 1, 1, n_pairs, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_bf16(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                       CUdeviceptr vec, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &vec, &n_rows, &n_cols };
    int threads = (n_cols < 256) ? ((n_cols + 31) / 32 * 32) : 256;
    cuLaunchKernel(r->fn_matvec_bf16_f32, n_rows, 1, 1, threads, 1, 1, 0, r->stream, args, NULL);
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

static inline void launch_embed_q4_0(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr embd_table,
                                      int token_id, int n_embd) {
    void *args[] = { &dst, &embd_table, &token_id, &n_embd };
    cuLaunchKernel(r->fn_embed_q4_0,
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

/* Q8_0 weight x block-wise (q8_1) activation. The q8_1 buffer must already hold
 * the quantized activation (caller runs launch_quantize_q8_1). Block-wise scales
 * are far more accurate than the per-row matvec_q8_0_dp4a. 1 warp/row, 8/block. */
static inline void launch_matvec_q8_q81(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                         CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q8_0_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1,
                   256, 1, 1,
                   0, r->stream, args, NULL);
}

static inline void launch_matvec_auto(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                       CUdeviceptr x, int n_rows, int n_cols, int weight_type);

static inline void launch_matvec_fused2(cuda_llm_runner *r,
                                         CUdeviceptr dst1, CUdeviceptr dst2,
                                         CUdeviceptr mat1, CUdeviceptr mat2,
                                         CUdeviceptr x, int n_rows, int n_cols, int type) {
    if (type == GGML_TYPE_Q8_0) {
        void *args[] = { &dst1, &dst2, &mat1, &mat2, &x, &n_rows, &n_cols };
        cuLaunchKernel(r->fn_matvec_q8_0_f32_fused2,
                       n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    } else if (type == GGML_TYPE_F16) {
        void *args[] = { &dst1, &dst2, &mat1, &mat2, &x, &n_rows, &n_cols };
        cuLaunchKernel(r->fn_matvec_f16_f32_fused2,
                       n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    } else {
        /* Fallback: two separate launches */
        launch_matvec_auto(r, dst1, mat1, x, n_rows, n_cols, type);
        launch_matvec_auto(r, dst2, mat2, x, n_rows, n_cols, type);
    }
}

/* dp4a fused gate+up: both matrices are Q8_0, input is pre-quantized INT8 */
static inline void launch_matvec_dp4a_fused2(cuda_llm_runner *r,
                                              CUdeviceptr dst1, CUdeviceptr dst2,
                                              CUdeviceptr mat1, CUdeviceptr mat2,
                                              CUdeviceptr x_q, CUdeviceptr x_scale,
                                              int n_rows, int n_cols) {
    void *args[] = { &dst1, &dst2, &mat1, &mat2, &x_q, &x_scale, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q8_0_dp4a_fused2,
                   n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
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
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* Quantize F32 -> Q8_1 (per-32 block). Launch n/32 blocks, 32 threads each
 * (blockDim=(32,1), gridDim=(n/32,1,1)). */
static inline void launch_quantize_q8_1(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr src, int n) {
    void *args[] = { &dst, &src, &n };
    int n_blocks = n / 32;
    /* Pack 4 Q8_1 blocks per CTA: blockDim=(32,4), grid=ceil(n_blocks/4). */
    int ctas = (n_blocks + 3) / 4;
    cuLaunchKernel(r->fn_quantize_f32_to_q8_1,
                   ctas, 1, 1, 32, 4, 1, 0, r->stream, args, NULL);
}

/* Q2_K dp4a: assumes Q8_1-quantized x has already been written to x_q81. */
static inline void launch_matvec_q2_K_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                            CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q2_K_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q3_K_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                            CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q3_K_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q3_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q3_K_f32,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q4_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q4_K_f32,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q4_K_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                            CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q4_K_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q4_0_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                            CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q4_0_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q5_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q5_K_f32,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q5_K_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                            CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q5_K_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q6_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q6_K_f32,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_q6_K_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                            CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_q6_K_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

static inline void launch_matvec_iq2_xxs_ex(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols, int bm) {
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols, &bm };
    cuLaunchKernel(r->fn_matvec_iq2_xxs_f32,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}
static inline void launch_matvec_iq2_xxs(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                        CUdeviceptr x, int n_rows, int n_cols) {
    launch_matvec_iq2_xxs_ex(r, dst, mat, x, n_rows, n_cols, 0);
}
static inline void launch_matvec_iq2_xxs_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                              CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    /* Coalesced (lane->qs-short) variant is default; CUDA_LLM_NO_IQ2XXS_COAL falls
       back to the strided (lane->block) kernel for A/B. */
    CUfunction fn = (r->fn_matvec_iq2_xxs_q8_1_dp4a_coal && !getenv("CUDA_LLM_NO_IQ2XXS_COAL"))
                    ? r->fn_matvec_iq2_xxs_q8_1_dp4a_coal : r->fn_matvec_iq2_xxs_q8_1_dp4a;
    cuLaunchKernel(fn, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}
static inline void launch_matvec_iq3_xxs_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                              CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_iq3_xxs_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}
static inline void launch_matvec_iq2_s_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                            CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_iq2_s_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}
static inline void launch_matvec_iq3_s_dp4a(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                            CUdeviceptr x_q81, int n_rows, int n_cols) {
    void *args[] = { &dst, &mat, &x_q81, &n_rows, &n_cols };
    cuLaunchKernel(r->fn_matvec_iq3_s_q8_1_dp4a,
                   (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}
/* MoE expert matvec: routes IQ2_XXS to the block-major path when expert weights are repacked. */
static inline void launch_moe_expert_matvec(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                            CUdeviceptr x, int n_rows, int n_cols, int type) {
    if (r->moe_iq2_bm && type == GGML_TYPE_IQ2_XXS)
        launch_matvec_iq2_xxs_ex(r, dst, mat, x, n_rows, n_cols, 1);
    else
        launch_matvec_auto(r, dst, mat, x, n_rows, n_cols, type);
}

#define DEFINE_LAUNCH_MATVEC(name, fn_field) \
static inline void launch_matvec_##name(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat, \
                                        CUdeviceptr x, int n_rows, int n_cols) { \
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols }; \
    cuLaunchKernel(r->fn_field, n_rows, 1, 1, 256, 1, 1, 0, r->stream, args, NULL); \
}

/* Multi-warp variant: 8 warps/block, grid = ceil(n_rows/8) */
#define DEFINE_LAUNCH_MATVEC_MW(name, fn_field) \
static inline void launch_matvec_##name(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat, \
                                        CUdeviceptr x, int n_rows, int n_cols) { \
    void *args[] = { &dst, &mat, &x, &n_rows, &n_cols }; \
    cuLaunchKernel(r->fn_field, (n_rows + 7) / 8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL); \
}

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

static inline void launch_embed_q2_K(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr embd_table,
                                       int token_id, int n_embd) {
    void *args[] = { &dst, &embd_table, &token_id, &n_embd };
    cuLaunchKernel(r->fn_embed_q2_K,
                   (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* Forward declaration for F32 matvec (defined later, used by auto-dispatch) */
static inline void launch_matvec_f32(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                      CUdeviceptr x, int n_rows, int n_cols);

/* Auto-dispatch matvec based on weight type */
static inline void launch_matvec_auto(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                       CUdeviceptr x, int n_rows, int n_cols, int weight_type) {
    switch (weight_type) {
        case GGML_TYPE_Q8_0:
            if (r->use_dp4a && (n_cols % 256) == 0 && (n_cols % 32) == 0 && !getenv("CUDA_LLM_NO_Q8K_DP4A")) {
                if (!getenv("CUDA_LLM_Q8_0_PERROW")) {
                    /* Block-wise (q8_1) activation quant: per-32 scales instead of
                     * one per-row absmax. Far more accurate (per-row crushed small
                     * activations around outliers -> the Q8_0 LM head/attn were the
                     * dominant dp4a error). CUDA_LLM_Q8_0_PERROW reverts for A/B. */
                    launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                    launch_matvec_q8_q81(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
                } else {
                    /* Legacy per-row path. matvec_q8_0_dp4a reads scales from a
                     * separate buffer; quantize to d_xb_q + d_xb_scale. */
                    launch_quantize(r, r->d_xb_q, r->d_xb_scale, x, n_cols);
                    launch_matvec_q8(r, dst, mat, r->d_xb_q, r->d_xb_scale, n_rows, n_cols);
                }
            } else {
                launch_matvec_q8_f32(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_Q2_K:
            if (r->use_dp4a && (n_cols % 256) == 0 && (n_cols % 32) == 0) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_q2_K_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_q2_K(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_Q3_K:
            if (r->use_dp4a && (n_cols % 256) == 0 && (n_cols % 32) == 0 && !getenv("CUDA_LLM_NO_Q3K_DP4A")) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_q3_K_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_q3_K(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_Q4_K:
            if (r->use_dp4a && (n_cols % 256) == 0 && (n_cols % 32) == 0 && !getenv("CUDA_LLM_NO_Q4K_DP4A")) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_q4_K_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_q4_K(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_Q5_K:
            if (r->use_dp4a && (n_cols % 256) == 0 && (n_cols % 32) == 0 && !getenv("CUDA_LLM_NO_Q5K_DP4A")) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_q5_K_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_q5_K(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_Q6_K:
            if (r->use_dp4a && (n_cols % 256) == 0 && (n_cols % 32) == 0 && !getenv("CUDA_LLM_NO_Q6K_DP4A")) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_q6_K_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_q6_K(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_IQ2_XXS:
            if (r->use_dp4a && (n_cols % 256) == 0 && !getenv("CUDA_LLM_NO_IQ2XXS_DP4A")) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_iq2_xxs_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_iq2_xxs(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_Q4_0:
            if (r->use_dp4a && (n_cols % 32) == 0 && !getenv("CUDA_LLM_NO_Q40_DP4A")) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_q4_0_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_q4_0(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_Q4_1:    launch_matvec_q4_1(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q5_0:    launch_matvec_q5_0(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_Q5_1:    launch_matvec_q5_1(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ4_NL:  launch_matvec_iq4_nl(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ4_XS:  launch_matvec_iq4_xs(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ2_XS:  launch_matvec_iq2_xs(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ3_XXS:
            if (r->use_dp4a && (n_cols % 256) == 0 && !getenv("CUDA_LLM_NO_IQ3XXS_DP4A")) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_iq3_xxs_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_iq3_xxs(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_IQ2_S:
            if (r->use_dp4a && (n_cols % 256) == 0 && !getenv("CUDA_LLM_NO_IQ2S_DP4A")) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_iq2_s_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_iq2_s(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_IQ3_S:
            if (r->use_dp4a && (n_cols % 256) == 0 && !getenv("CUDA_LLM_NO_IQ3S_DP4A")) {
                launch_quantize_q8_1(r, r->d_xb_q81, x, n_cols);
                launch_matvec_iq3_s_dp4a(r, dst, mat, r->d_xb_q81, n_rows, n_cols);
            } else {
                launch_matvec_iq3_s(r, dst, mat, x, n_rows, n_cols);
            }
            break;
        case GGML_TYPE_IQ1_S:   launch_matvec_iq1_s(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_IQ1_M:   launch_matvec_iq1_m(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_TQ1_0:   launch_matvec_tq1_0(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_TQ2_0:   launch_matvec_tq2_0(r, dst, mat, x, n_rows, n_cols); break;
        case GGML_TYPE_F32:     launch_matvec_f32(r, dst, mat, x, n_rows, n_cols); break;
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

static inline void launch_exp_inplace(cuda_llm_runner *r, CUdeviceptr data, int n) {
    void *args[] = { &data, &n };
    cuLaunchKernel(r->fn_exp_inplace_f32,
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

static inline void launch_batch_scale_add_sigmoid(cuda_llm_runner *r, CUdeviceptr dst,
                                                   CUdeviceptr src, CUdeviceptr gate,
                                                   int n_tokens, int dim) {
    int total = n_tokens * dim;
    void *args[] = { &dst, &src, &gate, &n_tokens, &dim };
    cuLaunchKernel(r->fn_batch_scale_add_sigmoid_f32,
                   (total + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
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
        int lbi = -1;
        for (int i = 0; i < n; i++) {
            float v = logits[i];
            if (!(v == v)) v = -FLT_MAX;
            /* Skip already selected */
            int skip = 0;
            for (int j = 0; j < ki; j++) {
                if (out_idx[j] == i) { skip = 1; break; }
            }
            if (skip) continue;
            if (v > best || (v == best && (lbi < 0 || i < lbi))) { best = v; lbi = i; }
        }
        out_idx[ki] = lbi;
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

static int launch_moe_topk_prefill(cuda_llm_runner *r, int n_tokens) {
    int tke = n_tokens * r->n_experts_used;
    if (r->d_topk_idx_entries < tke) {
        if (r->d_topk_idx) { cuMemFree(r->d_topk_idx); r->d_topk_idx = 0; }
        if (r->d_topk_wgt) { cuMemFree(r->d_topk_wgt); r->d_topk_wgt = 0; }
        CUdeviceptr new_idx = 0, new_wgt = 0;
        if (cuMemAlloc(&new_idx, (size_t)tke * sizeof(int)) != CUDA_SUCCESS ||
            cuMemAlloc(&new_wgt, (size_t)tke * sizeof(float)) != CUDA_SUCCESS) {
            if (new_idx) cuMemFree(new_idx);
            if (new_wgt) cuMemFree(new_wgt);
            fprintf(stderr, "cuda_llm: top-k buffer allocation failed\n");
            return -1;
        }
        r->d_topk_idx = new_idx;
        r->d_topk_wgt = new_wgt;
        r->d_topk_idx_entries = tke;
    }

    if (r->fn_moe_topk_gpu) {
        int n_experts = r->n_experts;
        int n_used = r->n_experts_used;
        void *args[] = { &r->d_topk_idx, &r->d_topk_wgt, &r->d_router_logits, &n_experts, &n_used, &n_tokens };
        cuLaunchKernel(r->fn_moe_topk_gpu, n_tokens, 1, 1, 128, 1, 1, 0, r->stream, args, NULL);
        return 0;
    }

    if (r->fn_moe_topk_kernel) {
        int n_experts = r->n_experts;
        int n_used = r->n_experts_used;
        void *args[] = { &r->d_router_logits, &r->d_topk_idx, &r->d_topk_wgt, &n_tokens, &n_experts, &n_used };
        cuLaunchKernel(r->fn_moe_topk_kernel, n_tokens, 1, 1, 128, 1, 1, 0, r->stream, args, NULL);
        return 0;
    }

    {
        size_t logits_count = (size_t)n_tokens * r->n_experts;
        int *h_idx = (int *)malloc((size_t)tke * sizeof(int));
        float *h_wgt = (float *)malloc((size_t)tke * sizeof(float));
        float *h_logits = (float *)malloc(logits_count * sizeof(float));
        if (!h_idx || !h_wgt || !h_logits) {
            free(h_idx);
            free(h_wgt);
            free(h_logits);
            fprintf(stderr, "cuda_llm: CPU top-k fallback allocation failed\n");
            return -1;
        }
        if (cuStreamSynchronize(r->stream) != CUDA_SUCCESS ||
            cuMemcpyDtoH(h_logits, r->d_router_logits, logits_count * sizeof(float)) != CUDA_SUCCESS) {
            free(h_idx);
            free(h_wgt);
            free(h_logits);
            fprintf(stderr, "cuda_llm: CPU top-k fallback download failed\n");
            return -1;
        }
        for (int t = 0; t < n_tokens; t++) {
            moe_topk_softmax(h_logits + (size_t)t * r->n_experts, r->n_experts, r->n_experts_used,
                             h_idx + (size_t)t * r->n_experts_used,
                             h_wgt + (size_t)t * r->n_experts_used);
        }
        if (cuMemcpyHtoD(r->d_topk_idx, h_idx, (size_t)tke * sizeof(int)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(r->d_topk_wgt, h_wgt, (size_t)tke * sizeof(float)) != CUDA_SUCCESS) {
            free(h_idx);
            free(h_wgt);
            free(h_logits);
            fprintf(stderr, "cuda_llm: CPU top-k fallback upload failed\n");
            return -1;
        }
        free(h_idx);
        free(h_wgt);
        free(h_logits);
    }

    return 0;
}

/* ======================================================================== */
/* Public API: forward                                                      */
/* ======================================================================== */

/* Forward declaration for MoE cuBLAS helper (used in cuda_llm_forward_blocks) */
static int launch_moe_expert_cublas(cuda_llm_runner *r, cuda_layer *cl,
    CUdeviceptr dst, int expert_idx, int is_down,
    CUdeviceptr x, int n_tokens, int out_dim, int in_dim,
    CUdeviceptr d_x_f16);

static float *cuda_llm_forward_blocks(cuda_llm_runner *r, int position, int apply_final_norm, int copy_to_host);
static float *cuda_llm_prefill_sequential(cuda_llm_runner *r, const int32_t *token_ids,
                                           const float *embeddings, int embd_stride,
                                           int n_tokens, int start_pos);

float *cuda_llm_forward(cuda_llm_runner *r, int32_t token_id, int position) {
    if (!r || !r->weights_loaded) return NULL;
    if (token_id < 0 || token_id >= r->n_vocab) return NULL;
    if (position < 0 || position >= r->max_seq_len) return NULL;
    if (cuda_llm_bind_context(r) != 0) return NULL;

    int n_embd = r->n_embd;

    /* 1. Token embedding lookup -> F32 */
    if (r->token_embd_type == GGML_TYPE_Q8_0) {
        launch_embed_q8_0(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else if (r->token_embd_type == GGML_TYPE_Q2_K) {
        launch_embed_q2_K(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else if (r->token_embd_type == GGML_TYPE_Q4_K) {
        void *args[] = { &r->d_x, &r->d_token_embd, &token_id, &n_embd };
        cuLaunchKernel(r->fn_embed_q4_K,
                       (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    } else if (r->token_embd_type == GGML_TYPE_Q4_0) {
        launch_embed_q4_0(r, r->d_x, r->d_token_embd, token_id, n_embd);
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

    /* Embedding cross-check: verify embed kernel against CPU dequant */
    if (r->debug_layers >= 3 && (r->token_embd_type == GGML_TYPE_Q2_K || r->token_embd_type == GGML_TYPE_Q4_K)) {
        cuStreamSynchronize(r->stream);
        float *gpu_emb = (float *)malloc(n_embd * sizeof(float));
        cuMemcpyDtoH(gpu_emb, r->d_x, n_embd * sizeof(float));

        /* Download embedding weights from GPU */
        int nb_per_row = n_embd / 256;
        int blk_bytes = (r->token_embd_type == GGML_TYPE_Q4_K) ? 144 : 84;
        int row_bytes = nb_per_row * blk_bytes;
        /* Only download the one row we need */
        unsigned char *row_data = (unsigned char *)malloc(row_bytes);
        cuMemcpyDtoH(row_data, r->d_token_embd + (size_t)token_id * row_bytes, row_bytes);

        /* CPU dequantize this row */
        float *cpu_emb = (float *)malloc(n_embd * sizeof(float));
        dequant_row(r->token_embd_type, row_data, cpu_emb, n_embd);

        float emb_max_err = 0; int emb_max_idx = 0;
        for (int i = 0; i < n_embd; i++) {
            float err = fabsf(gpu_emb[i] - cpu_emb[i]);
            if (err > emb_max_err) { emb_max_err = err; emb_max_idx = i; }
        }
        fprintf(stderr, "  [EMB xchk] embed max_err=%.6f at idx=%d (GPU=%.6f CPU=%.6f)\n",
                emb_max_err, emb_max_idx, gpu_emb[emb_max_idx], cpu_emb[emb_max_idx]);
        if (emb_max_err > 0.001f) {
            fprintf(stderr, "    GPU[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                    gpu_emb[0], gpu_emb[1], gpu_emb[2], gpu_emb[3]);
            fprintf(stderr, "    CPU[0:4]=[%.6f,%.6f,%.6f,%.6f]\n",
                    cpu_emb[0], cpu_emb[1], cpu_emb[2], cpu_emb[3]);
        }
        free(gpu_emb); free(cpu_emb); free(row_data);
    }

    /* Gemma4: scale token embeddings by sqrt(n_embd), stash token_id for PLE */
    if (r->is_gemma4) {
        launch_scale(r, r->d_x, r->embd_scale, n_embd);
        r->current_token_id = token_id;
    }

    /* Run transformer blocks (shared with forward_embd) */
    return cuda_llm_forward_blocks(r, position, 1, 1);
}

int cuda_llm_forward_nohost(cuda_llm_runner *r, int32_t token_id, int position) {
    if (!r || !r->weights_loaded) return -1;
    if (token_id < 0 || token_id >= r->n_vocab) return -1;
    if (position < 0 || position >= r->max_seq_len) return -1;
    if (cuda_llm_bind_context(r) != 0) return -1;

    int n_embd = r->n_embd;

    if (r->token_embd_type == GGML_TYPE_Q8_0) {
        launch_embed_q8_0(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else if (r->token_embd_type == GGML_TYPE_Q2_K) {
        launch_embed_q2_K(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else if (r->token_embd_type == GGML_TYPE_Q4_K) {
        void *args[] = { &r->d_x, &r->d_token_embd, &token_id, &n_embd };
        cuLaunchKernel(r->fn_embed_q4_K,
                       (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    } else if (r->token_embd_type == GGML_TYPE_Q4_0) {
        launch_embed_q4_0(r, r->d_x, r->d_token_embd, token_id, n_embd);
    } else {
        launch_embed(r, r->d_x, r->d_token_embd, token_id, n_embd);
    }

    if (r->is_gemma4) {
        launch_scale(r, r->d_x, r->embd_scale, n_embd);
        r->current_token_id = token_id;
    }

    cuda_llm_forward_blocks(r, position, 0, 0);
    return 0;
}

/* Internal: run transformer blocks + final norm on d_x.
 * Assumes d_x already contains the input embedding on GPU. */
static float *cuda_llm_forward_blocks(cuda_llm_runner *r, int position, int apply_final_norm, int copy_to_host) {
    int n_embd = r->n_embd;
    int n_heads = r->n_heads;
    int n_kv_heads = r->n_kv_heads;
    int head_dim = r->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int n_ff = r->n_ff;
    float eps = r->rms_norm_eps;

    int n_run_layers = r->n_layers;
    if (r->max_layers > 0 && r->max_layers < r->n_layers) n_run_layers = r->max_layers;
    const char *stop_attn_env = getenv("CUDA_LLM_STOP_AFTER_ATTN_RESIDUAL");
    int stop_after_attn_residual = stop_attn_env && stop_attn_env[0] && strcmp(stop_attn_env, "0") != 0;
    int graph_enabled = !r->disable_graph && !r->is_hybrid && !r->is_gemma4 &&
                        r->debug_layers == 0 && r->max_layers == 0 &&
                        !stop_after_attn_residual;

    /* Gemma4: precompute per-layer embeddings on CPU and upload to GPU.
     * Mirrors transformer.h: combine token_embd lookup + model_proj @ x + normalize */
    if (r->is_gemma4 && r->current_token_id >= 0 &&
        r->h_ple_token_embd.data && r->h_ple_model_proj.data) {
        int ple_dim = r->n_embd_per_layer;
        int total_ple = ple_dim * r->n_layers;

        /* Read current hidden state from GPU for projection */
        float *h_x = (float *)malloc(n_embd * sizeof(float));
        float *tok_ple = NULL;
        float *proj_out = NULL;
        float *ple_host = NULL;
        if (!h_x) {
            fprintf(stderr, "cuda_llm: Gemma4 PLE hidden buffer alloc failed (n_embd=%d)\n", n_embd);
            return NULL;
        }
        CHECK_CU_NULL(cuStreamSynchronize(r->stream));
        CHECK_CU_NULL(cuMemcpyDtoH(h_x, r->d_x, n_embd * sizeof(float)));

        /* 1. Token embedding lookup */
        tok_ple = (float *)malloc(total_ple * sizeof(float));
        if (!tok_ple) {
            fprintf(stderr, "cuda_llm: Gemma4 PLE token buffer alloc failed (n=%d)\n", total_ple);
            free(h_x);
            return NULL;
        }
        {
            size_t row_bytes = dequant_row_size(r->h_ple_token_embd.type, total_ple);
            dequant_row(r->h_ple_token_embd.type,
                        (const uint8_t *)r->h_ple_token_embd.data + (size_t)r->current_token_id * row_bytes,
                        tok_ple, total_ple);
            float tok_scale = sqrtf((float)ple_dim);
            for (int i = 0; i < total_ple; i++) tok_ple[i] *= tok_scale;
        }

        /* 2. Project hidden state: proj = per_layer_model_proj @ h_x → [total_ple] */
        proj_out = (float *)malloc(total_ple * sizeof(float));
        if (!proj_out) {
            fprintf(stderr, "cuda_llm: Gemma4 PLE proj buffer alloc failed (n=%d)\n", total_ple);
            free(tok_ple);
            free(h_x);
            return NULL;
        }
        {
            int n_cols = r->h_ple_model_proj.n_cols;
            size_t row_bytes = dequant_row_size(r->h_ple_model_proj.type, n_cols);
            float *row_buf = (float *)malloc(n_cols * sizeof(float));
            if (!row_buf) {
                fprintf(stderr, "cuda_llm: Gemma4 PLE row buffer alloc failed (n_cols=%d)\n", n_cols);
                free(proj_out);
                free(tok_ple);
                free(h_x);
                return NULL;
            }
            const uint8_t *base = (const uint8_t *)r->h_ple_model_proj.data;
            for (int rr = 0; rr < total_ple; rr++) {
                dequant_row(r->h_ple_model_proj.type, base + (size_t)rr * row_bytes, row_buf, n_cols);
                float dot = 0.0f;
                for (int c = 0; c < n_cols; c++) dot += row_buf[c] * h_x[c];
                proj_out[rr] = dot;
            }
            free(row_buf);
        }

        /* Scale by 1/sqrt(n_embd) */
        float proj_scale = 1.0f / sqrtf((float)n_embd);
        for (int i = 0; i < total_ple; i++) proj_out[i] *= proj_scale;

        /* 3. RMSNorm with shared proj_norm [ple_dim] per layer slice */
        if (r->h_ple_proj_norm.data) {
            float norm_w[256];
            dequant_row(r->h_ple_proj_norm.type, r->h_ple_proj_norm.data, norm_w, ple_dim);
            for (int ll = 0; ll < r->n_layers; ll++) {
                float *slice = proj_out + ll * ple_dim;
                float ss = 0.0f;
                for (int i = 0; i < ple_dim; i++) ss += slice[i] * slice[i];
                ss = 1.0f / sqrtf(ss / ple_dim + eps);
                for (int i = 0; i < ple_dim; i++) slice[i] = slice[i] * ss * norm_w[i];
            }
        }

        /* 4. Combine: (proj + tok) * 1/sqrt(2) */
        float input_scale = 1.0f / sqrtf(2.0f);
        ple_host = (float *)malloc(total_ple * sizeof(float));
        if (!ple_host) {
            fprintf(stderr, "cuda_llm: Gemma4 PLE upload buffer alloc failed (n=%d)\n", total_ple);
            free(proj_out);
            free(tok_ple);
            free(h_x);
            return NULL;
        }
        for (int i = 0; i < total_ple; i++)
            ple_host[i] = (proj_out[i] + tok_ple[i]) * input_scale;

        /* Upload to GPU */
        CHECK_CU_NULL(cuMemcpyHtoDAsync(r->d_ple_combined, ple_host,
                                        total_ple * sizeof(float), r->stream));
        free(ple_host); free(proj_out); free(tok_ple); free(h_x);
    }

    /* Upload position and seq_len to device for graph-captured ptr-variant kernels. */
    if (graph_enabled && !r->d_pos_seq) {
        cuMemAlloc(&r->d_pos_seq, 2 * sizeof(int));
        r->d_seq_ptr = r->d_pos_seq + sizeof(int);
    }
    if (graph_enabled && r->d_pos_seq) {
        int ps[2] = {position, position + 1};
        cuMemcpyHtoDAsync(r->d_pos_seq, ps, 2 * sizeof(int), r->stream);
    }

    /* Decode profiling (debug_layers >= 2 and first 3 positions) */
    double dec_prof_ssm = 0, dec_prof_attn = 0, dec_prof_ffn = 0, dec_prof_proj = 0;
    int dec_do_prof = (r->debug_layers >= 2 && position < 3);

    if (graph_enabled && r->graph_captured == 1 && position > 0) {
        cuStreamSynchronize(r->stream);
        if (r->verbose >= 2 && position < 4)
            fprintf(stderr, "cuda_llm: CUDA graph replay pos=%d\n", position);
        cuGraphLaunch(r->d_graph_exec, r->stream);
        goto after_layers;
    }
    if (graph_enabled && r->graph_captured == 0 && position == 0) {
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_llm: CUDA graph capture starting\n");
        cuStreamSynchronize(r->stream);
        r->graph_captured = -1;
        cuStreamBeginCapture(r->stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
    }
    {
    int l;
    for (l = 0; l < n_run_layers; l++) {
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

        if (r->is_gemma4) {
            /* === Gemma4 layer === */
            int hd = cl->is_swa ? r->head_dim_swa : r->head_dim_full;
            int layer_kv_heads = cl->n_kv_heads;
            int local_kv_dim = layer_kv_heads * hd;

            /* Q projection */
            launch_matvec_auto(r, r->d_q, cl->attn_q_w, r->d_xb,
                              cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);

            /* K/V projections (skip if sharing KV) */
            if (cl->shared_kv_source < 0) {
                launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb,
                                  cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
                if (cl->has_v_proj) {
                    launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb,
                                      cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);
                } else {
                    /* V = K for layers without V projection */
                    cuMemcpyDtoDAsync(r->d_v, r->d_k, (size_t)local_kv_dim * sizeof(float), r->stream);
                }
            }

            /* Q/K/V norm */
            if (cl->has_qk_norm) {
                if (cl->attn_q_norm_w) launch_qknorm(r, r->d_q, cl->attn_q_norm_w, n_heads, hd, eps);
                if (cl->shared_kv_source < 0) {
                    if (cl->attn_k_norm_w) launch_qknorm(r, r->d_k, cl->attn_k_norm_w, layer_kv_heads, hd, eps);
                    /* V norm: raw RMSNorm per-head (no learned weight) */
                    {
                        int block_dim = hd;
                        /* Round up to power of 2 for reduction */
                        int bd = 1; while (bd < block_dim) bd <<= 1;
                        void *vargs[] = { &r->d_v, &layer_kv_heads, &hd, &eps };
                        cuLaunchKernel(r->fn_raw_rmsnorm_heads_f32,
                                       layer_kv_heads, 1, 1, bd, 1, 1,
                                       bd * sizeof(float), r->stream, vargs, NULL);
                    }
                }
            }

            /* RoPE */
            if (cl->is_swa) {
                /* SWA: standard RoPE with freq_base_swa and ALL dims rotated (n_rope_pairs=0) */
                int half_swa = hd / 2;
                int zero_pairs = 0; /* 0 = rotate all, rope_dim = head_dim */
                int use_ptr = !r->disable_graph && r->d_pos_seq;
                CUfunction fn_rope = use_ptr ? r->fn_rope_neox_f32_ptr : r->fn_rope_neox_f32;
                void *pos_arg = use_ptr ? (void*)&r->d_pos_seq : (void*)&position;
                void *qargs[] = { &r->d_q, &n_heads, &hd, pos_arg, &r->rope_freq_base_swa, &zero_pairs };
                cuLaunchKernel(fn_rope, n_heads, 1, 1, half_swa, 1, 1, 0, r->stream, qargs, NULL);
                if (cl->shared_kv_source < 0) {
                    void *kargs[] = { &r->d_k, &layer_kv_heads, &hd, pos_arg, &r->rope_freq_base_swa, &zero_pairs };
                    cuLaunchKernel(fn_rope, layer_kv_heads, 1, 1, half_swa, 1, 1, 0, r->stream, kargs, NULL);
                }
            } else {
                /* Full-attention: proportional RoPE with precomputed freq factors */
                launch_rope_with_factors(r, r->d_q, n_heads, hd, position, r->d_rope_inv_freq_full);
                if (cl->shared_kv_source < 0)
                    launch_rope_with_factors(r, r->d_k, layer_kv_heads, hd, position, r->d_rope_inv_freq_full);
            }

            /* KV cache store (F16) */
            if (cl->shared_kv_source < 0) {
                if (cl->is_swa) {
                    int slot = position % r->swa_window_size;
                    launch_kv_store(r, r->d_key_cache[l], r->d_value_cache[l],
                                    r->d_k, r->d_v, slot, local_kv_dim);
                    if (r->kv_cache_q8) launch_kv_store_q8(r, l, r->d_k, r->d_v, slot, local_kv_dim);
                } else {
                    launch_kv_store(r, r->d_key_cache[l], r->d_value_cache[l],
                                    r->d_k, r->d_v, position, local_kv_dim);
                    if (r->kv_cache_q8) launch_kv_store_q8(r, l, r->d_k, r->d_v, position, local_kv_dim);
                }
            }

            /* Attention (scale=1.0 since Q/K are normalized) */
            {
                int kv_src = (cl->shared_kv_source >= 0) ? cl->shared_kv_source : l;
                float attn_scale = 1.0f;
                if (cl->is_swa) {
                    launch_attention_swa(r, r->d_xb2, r->d_q,
                                        r->d_key_cache[kv_src], r->d_value_cache[kv_src],
                                        n_heads, layer_kv_heads, hd, local_kv_dim,
                                        position, r->swa_window_size, attn_scale);
                } else {
                    int seq_len = position + 1;
                    launch_attention(r, r->d_xb2, r->d_q,
                                   r->d_key_cache[kv_src], r->d_value_cache[kv_src],
                                   n_heads, layer_kv_heads, hd, local_kv_dim, seq_len, attn_scale);
                }
            }

            /* Output projection */
            launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                              cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);

            /* Post-attention RMSNorm */
            launch_rmsnorm(r, r->d_xb, r->d_xb, cl->post_attn_norm_w, n_embd, eps);

            /* Residual: x += xb */
            launch_add(r, r->d_x, r->d_xb, n_embd);

            /* FFN with GELU */
            launch_rmsnorm(r, r->d_xb, r->d_x, cl->ffn_norm_w, n_embd, eps);
            launch_matvec_auto(r, r->d_gate, cl->ffn_gate_w, r->d_xb,
                              cl->ffn_gate_rows, cl->ffn_gate_cols, cl->ffn_gate_type);
            launch_matvec_auto(r, r->d_up, cl->ffn_up_w, r->d_xb,
                              cl->ffn_up_rows, cl->ffn_up_cols, cl->ffn_up_type);
            launch_gelu_mul(r, r->d_gate, r->d_up, n_ff);
            launch_matvec_auto(r, r->d_xb, cl->ffn_down_w, r->d_gate,
                              cl->ffn_down_rows, cl->ffn_down_cols, cl->ffn_down_type);

            /* Post-FFN RMSNorm */
            launch_rmsnorm(r, r->d_xb, r->d_xb, cl->post_ffw_norm_w, n_embd, eps);

            /* Residual: x += xb */
            launch_add(r, r->d_x, r->d_xb, n_embd);

            /* Per-layer embedding injection */
            if (r->d_ple_combined) {
                int ple_dim = r->n_embd_per_layer;
                /* Copy slice for this layer from d_ple_combined to d_ple_buf */
                CHECK_CU_NULL(cuMemcpyDtoDAsync(r->d_ple_buf,
                                                r->d_ple_combined + (size_t)l * ple_dim * sizeof(float),
                                                ple_dim * sizeof(float), r->stream));

                /* inp_gate: gate_out = inp_gate @ x → [ple_dim] */
                launch_matvec_auto(r, r->d_ple_proj, cl->ple_inp_gate_w, r->d_x,
                                  cl->ple_inp_gate_rows, cl->ple_inp_gate_cols, cl->ple_inp_gate_type);

                /* GELU(gate) * ple */
                launch_gelu_elementwise_mul(r, r->d_ple_buf, r->d_ple_proj, ple_dim);

                /* Project back: proj = ple_proj @ ple → [n_embd] */
                launch_matvec_auto(r, r->d_ple_proj, cl->ple_proj_w, r->d_ple_buf,
                                  cl->ple_proj_rows, cl->ple_proj_cols, cl->ple_proj_type);

                /* Post-norm */
                if (cl->ple_post_norm_w)
                    launch_rmsnorm(r, r->d_ple_proj, r->d_ple_proj, cl->ple_post_norm_w, n_embd, eps);

                /* Residual add */
                launch_add(r, r->d_x, r->d_ple_proj, n_embd);
            }

            /* Layer output scale */
            if (cl->layer_scale_val != 1.0f) {
                launch_scale(r, r->d_x, cl->layer_scale_val, n_embd);
            }

        } else if (r->is_hybrid && cl->is_ssm) {
            /* === SSM (Delta-Net) layer === */
            int qkv_dim = r->ssm_qkv_dim;
            int d_state = r->ssm_d_state;
            int n_group = r->ssm_n_group;
            int dt_rank = r->ssm_dt_rank;
            int conv_k  = r->ssm_conv_kernel;

            if (dec_do_prof) cuStreamSynchronize(r->stream);
            double _dp0 = dec_do_prof ? get_time_ms() : 0;
            /* 1. Linear projections (dp4a for Q8_0 weights) */
            if (r->use_dp4a && cl->ssm_qkv_type == GGML_TYPE_Q8_0) {
                launch_quantize(r, r->d_xb_q, r->d_xb_scale, r->d_xb, n_embd);
                launch_matvec_q8(r, r->d_ssm_qkv, cl->ssm_qkv_w, r->d_xb_q, r->d_xb_scale,
                                 cl->ssm_qkv_rows, cl->ssm_qkv_cols);
            } else {
                launch_matvec_auto(r, r->d_ssm_qkv, cl->ssm_qkv_w, r->d_xb, cl->ssm_qkv_rows, cl->ssm_qkv_cols, cl->ssm_qkv_type);
            }
            launch_matvec_auto(r, r->d_ssm_z,      cl->ssm_gate_w,  r->d_xb, cl->ssm_gate_rows,  cl->ssm_gate_cols,  cl->ssm_gate_type);
            /* Fuse alpha+beta when same type+dims */
            if (cl->ssm_alpha_type == cl->ssm_beta_type &&
                cl->ssm_alpha_rows == cl->ssm_beta_rows &&
                cl->ssm_alpha_cols == cl->ssm_beta_cols) {
                launch_matvec_fused2(r, r->d_ssm_alpha, r->d_ssm_beta,
                    cl->ssm_alpha_w, cl->ssm_beta_w, r->d_xb,
                    cl->ssm_alpha_rows, cl->ssm_alpha_cols, cl->ssm_alpha_type);
            } else {
                launch_matvec_auto(r, r->d_ssm_alpha,  cl->ssm_alpha_w, r->d_xb, cl->ssm_alpha_rows, cl->ssm_alpha_cols, cl->ssm_alpha_type);
                launch_matvec_auto(r, r->d_ssm_beta,   cl->ssm_beta_w,  r->d_xb, cl->ssm_beta_rows,  cl->ssm_beta_cols,  cl->ssm_beta_type);
            }
            if (dec_do_prof) { cuStreamSynchronize(r->stream); dec_prof_proj += get_time_ms() - _dp0; }

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

            if (dec_do_prof) { cuStreamSynchronize(r->stream); _dp0 = get_time_ms(); }
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

            if (dec_do_prof) { cuStreamSynchronize(r->stream); dec_prof_ssm += get_time_ms() - _dp0; _dp0 = get_time_ms(); }
            /* 9. Output projection: xb = ssm_out @ out */
            launch_matvec_auto(r, r->d_xb, cl->ssm_out_w, r->d_ssm_out,
                              cl->ssm_out_rows, cl->ssm_out_cols, cl->ssm_out_type);
            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, r->d_xb, 8 * sizeof(float));
                fprintf(stderr, "  [L0 DBG] ssm_out_proj[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }
            if (dec_do_prof) { cuStreamSynchronize(r->stream); dec_prof_proj += get_time_ms() - _dp0; }

        } else if (r->is_hybrid) {
            /* === Gated attention layer (Qwen3.5) === */
            if (dec_do_prof) cuStreamSynchronize(r->stream);
            double _dp0 = dec_do_prof ? get_time_ms() : 0;
            /* Quantize input for dp4a Q8_0 projections */
            if (r->use_dp4a && cl->attn_q_type == GGML_TYPE_Q8_0) {
                launch_quantize(r, r->d_xb_q, r->d_xb_scale, r->d_xb, n_embd);
                /* Q+gate combined projection via dp4a */
                launch_matvec_q8(r, r->d_xb2, cl->attn_q_w, r->d_xb_q, r->d_xb_scale,
                                 cl->attn_q_rows, cl->attn_q_cols);
            } else {
                launch_matvec_auto(r, r->d_xb2, cl->attn_q_w, r->d_xb,
                                  cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);
            }

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

            /* K, V projections (dp4a reuses d_xb_q from above) */
            if (r->use_dp4a && cl->attn_k_type == GGML_TYPE_Q8_0) {
                launch_matvec_q8(r, r->d_k, cl->attn_k_w, r->d_xb_q, r->d_xb_scale,
                                 cl->attn_k_rows, cl->attn_k_cols);
                launch_matvec_q8(r, r->d_v, cl->attn_v_w, r->d_xb_q, r->d_xb_scale,
                                 cl->attn_v_rows, cl->attn_v_cols);
            } else {
                launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb, cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
                launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb, cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);
            }

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
            if (r->kv_cache_q8) launch_kv_store_q8(r, l, r->d_k, r->d_v, position, kv_dim);

            /* Attention decode */
            int seq_len = position + 1;
            float scale = 1.0f / sqrtf((float)head_dim);
            if (r->kv_cache_q8) {
                launch_attention_q8(r, r->d_xb2, r->d_q,
                                   r->d_key_cache_q8[l], r->d_value_cache_q8[l],
                                   r->d_key_scale[l], r->d_value_scale[l],
                                   n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale);
            } else {
                launch_attention(r, r->d_xb2, r->d_q,
                                r->d_key_cache[l], r->d_value_cache[l],
                                n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale);
            }

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

                /* Read V from KV cache at position 0 to verify it was stored correctly (F16 cache) */
                uint16_t *cache_v_f16 = (uint16_t *)malloc(kv_dim * sizeof(uint16_t));
                float *cache_v = (float *)malloc(kv_dim * sizeof(float));
                cuMemcpyDtoH(cache_v_f16, r->d_value_cache[l], kv_dim * sizeof(uint16_t));
                for (int i = 0; i < kv_dim; i++) cache_v[i] = ggml_fp16_to_fp32(cache_v_f16[i]);
                free(cache_v_f16);

                /* Check V was stored correctly in cache (F16 precision) */
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

            /* Output projection (dp4a for Q8_0) */
            if (r->use_dp4a && cl->attn_output_type == GGML_TYPE_Q8_0) {
                launch_quantize(r, r->d_xb_q2, r->d_xb_scale2, r->d_xb2, cl->attn_output_cols);
                launch_matvec_q8(r, r->d_xb, cl->attn_output_w, r->d_xb_q2, r->d_xb_scale2,
                                 cl->attn_output_rows, cl->attn_output_cols);
            } else {
                launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                                  cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);
            }
            if (dec_do_prof) { cuStreamSynchronize(r->stream); dec_prof_attn += get_time_ms() - _dp0; }

        } else {
            /* === Standard attention (non-hybrid) === */
            if (dec_do_prof) cuStreamSynchronize(r->stream);
            double _dp0 = dec_do_prof ? get_time_ms() : 0;
            /* Q/K/V projections (auto-dispatch) */
            launch_matvec_auto(r, r->d_q, cl->attn_q_w, r->d_xb, cl->attn_q_rows, cl->attn_q_cols, cl->attn_q_type);
            if (cl->attn_q_bias) launch_add(r, r->d_q, cl->attn_q_bias, cl->attn_q_rows);
            launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb, cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
            if (cl->attn_k_bias) launch_add(r, r->d_k, cl->attn_k_bias, cl->attn_k_rows);
            launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb, cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);
            if (cl->attn_v_bias) launch_add(r, r->d_v, cl->attn_v_bias, cl->attn_v_rows);

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
            if (r->kv_cache_q8) launch_kv_store_q8(r, l, r->d_k, r->d_v, position, kv_dim);

            /* Attention decode */
            int seq_len = position + 1;
            float scale = 1.0f / sqrtf((float)head_dim);
            if (r->kv_cache_q8) {
                launch_attention_q8(r, r->d_xb2, r->d_q,
                                   r->d_key_cache_q8[l], r->d_value_cache_q8[l],
                                   r->d_key_scale[l], r->d_value_scale[l],
                                   n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale);
            } else {
                launch_attention(r, r->d_xb2, r->d_q,
                                r->d_key_cache[l], r->d_value_cache[l],
                                n_heads, n_kv_heads, head_dim, kv_dim, seq_len, scale);
            }

            /* Output projection */
            launch_matvec_auto(r, r->d_xb, cl->attn_output_w, r->d_xb2,
                              cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);
            if (dec_do_prof) { cuStreamSynchronize(r->stream); dec_prof_attn += get_time_ms() - _dp0; }
        }

        /* Gemma4 already did its own residual + FFN above — skip shared code */
        if (r->is_gemma4) goto cuda_layer_done;

        /* Residual: x += xb */
        launch_add(r, r->d_x, r->d_xb, n_embd);
        if (stop_after_attn_residual) goto after_layers;

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
        if (dec_do_prof) cuStreamSynchronize(r->stream);
        double _dfn0 = dec_do_prof ? get_time_ms() : 0;
        launch_rmsnorm(r, r->d_xb, r->d_x, cl->ffn_norm_w, n_embd, eps);

        if (cl->is_moe) {
            int n_experts = r->n_experts;
            int n_used = r->n_experts_used;
            int expert_ff = r->expert_ff;
            int shared_expert_ff = r->shared_expert_ff;

            launch_matvec_f32(r, r->d_router_logits, cl->moe_gate_w, r->d_xb,
                             cl->moe_gate_rows, cl->moe_gate_cols);

            /* Fused decode: GPU top-k + 8 parallel expert blocks + shared gate */
            /* Fused expert kernel disabled: 8 blocks × 128 threads underutilizes SMs.
             * Each matvec needs grid=(out_dim/8) for full SM parallelism.
             * Use fallback path (24 launches/layer × 65 blocks/launch = full occupancy). */
            if (0 && r->fn_moe_iq2s_tc /* was expert_fused */ && r->fn_moe_topk_gpu) {
                if (!r->d_topk_idx) cuMemAlloc(&r->d_topk_idx, 8 * sizeof(int));
                if (!r->d_topk_wgt) cuMemAlloc(&r->d_topk_wgt, 8 * sizeof(float));
                /* grid upload is in fallback path */
                /* Lazy upload of grid tables to device (cubin kernels need them as params) */
                if (!r->d_grid_ksigns) {
                    /* data from moe_grid_data.h */
                    /* data from moe_grid_data.h */
                    /* data from moe_grid_data.h */
                    // Can't use #include here directly — arrays are defined elsewhere
                    // For now, allocate and the data will be filled below
                }

                void *tka[] = { &r->d_topk_idx, &r->d_topk_wgt,
                                &r->d_router_logits, &n_experts, &n_used };
                cuLaunchKernel(r->fn_moe_topk_gpu, 1,1,1, 128,1,1, 0, r->stream, tka, NULL);

                cuMemsetD32(r->d_moe_accum, 0, n_embd);

                {
                    /* grid upload is in fallback path */
                    CUdeviceptr d_ge = cl->moe_gate_exps_w;
                    CUdeviceptr d_ue = cl->moe_up_exps_w;
                    CUdeviceptr d_de = cl->moe_down_exps_w;
                    void *efa[] = { &r->d_moe_accum, &r->d_xb,
                                    &r->d_topk_idx, &r->d_topk_wgt,
                                    &d_ge, &d_ue, &d_de,
                                    &n_embd, &expert_ff,
                                    &cl->moe_exp_stride_gu, &cl->moe_exp_stride_d,
                                    &r->d_grid_ksigns, &r->d_grid_iq2s, &r->d_grid_iq3s };
                    cuLaunchKernel(r->fn_moe_iq2s_tc /* was expert_fused */, n_used,1,1, 128,1,1,
                                   0, r->stream, efa, NULL);
                }

                {
                    void *sga[] = { &r->d_topk_wgt, &r->d_xb,
                                    &cl->moe_shared_gate_w, &n_embd };
                    cuLaunchKernel(r->fn_moe_shared_gate_gpu, 1,1,1, 128,1,1,
                                   0, r->stream, sga, NULL);
                }

                {
                    float shared_scale;
                    cuMemcpyDtoH(&shared_scale, r->d_topk_wgt, sizeof(float));
                    launch_matvec_auto(r, r->d_gate, cl->moe_shared_ffn_gate_w, r->d_xb,
                                      cl->moe_shared_gate_rows, cl->moe_shared_gate_cols, cl->moe_shared_gate_type);
                    launch_matvec_auto(r, r->d_up, cl->moe_shared_ffn_up_w, r->d_xb,
                                      cl->moe_shared_up_rows, cl->moe_shared_up_cols, cl->moe_shared_up_type);
                    launch_silu_mul(r, r->d_gate, r->d_up, shared_expert_ff);
                    launch_matvec_auto(r, r->d_xb2, cl->moe_shared_ffn_down_w, r->d_gate,
                                      cl->moe_shared_down_rows, cl->moe_shared_down_cols, cl->moe_shared_down_type);
                    launch_scale_add(r, r->d_moe_accum, r->d_xb2, shared_scale, n_embd);
                }

                cuMemcpyDtoDAsync(r->d_xb, r->d_moe_accum, n_embd * sizeof(float), r->stream);
            } else {
                /* Fallback: CPU top-k + per-expert matvecs */
                cuStreamSynchronize(r->stream);
                cuMemcpyDtoH(r->h_router_logits, r->d_router_logits, n_experts * sizeof(float));
                int tki[64]; float tkw[64];
                moe_topk_softmax(r->h_router_logits, n_experts, n_used, tki, tkw);
                int hti[8]; float htw[8];
                for (int i = 0; i < 8; i++) { hti[i] = tki[i]; htw[i] = tkw[i]; }
                /* Upload CPU top-k results to GPU for optional TC kernels. */
                if (r->fn_moe_iq2s_tc && !r->d_topk_idx)
                    cuMemAlloc(&r->d_topk_idx, 8 * sizeof(int));
                if ((r->fn_moe_iq2s_tc || r->fn_moe_shared_gate_gpu) && !r->d_topk_wgt)
                    cuMemAlloc(&r->d_topk_wgt, 8 * sizeof(float));
                if (r->fn_moe_iq2s_tc && r->d_topk_idx && r->d_topk_wgt) {
                    cuMemcpyHtoD(r->d_topk_idx, hti, 8 * sizeof(int));
                    cuMemcpyHtoD(r->d_topk_wgt, htw, 8 * sizeof(float));
                }
                if (r->fn_moe_iq2s_tc && (!r->d_topk_idx || !r->d_topk_wgt)) {
                    r->fn_moe_iq2s_tc = NULL;
                    r->fn_moe_iq3s_tc = NULL;
                    r->fn_moe_accum_gpu = NULL;
                }
                if (r->fn_moe_shared_gate_gpu && !r->d_topk_wgt) {
                    r->fn_moe_shared_gate_gpu = NULL;
                }
                cuMemsetD32(r->d_moe_accum, 0, n_embd);
                if (!r->d_grid_iq2s && r->fn_moe_iq2s_tc) {
                    CUresult _ge = cuMemAlloc(&r->d_grid_ksigns, sizeof(_grid_ksigns));
                    _ge |= cuMemAlloc(&r->d_grid_iq2s, sizeof(_grid_iq2s));
                    _ge |= cuMemAlloc(&r->d_grid_iq3s, sizeof(_grid_iq3s));
                    if (_ge == CUDA_SUCCESS) {
                        cuMemcpyHtoD(r->d_grid_ksigns, _grid_ksigns, sizeof(_grid_ksigns));
                        cuMemcpyHtoD(r->d_grid_iq2s, _grid_iq2s, sizeof(_grid_iq2s));
                        cuMemcpyHtoD(r->d_grid_iq3s, _grid_iq3s, sizeof(_grid_iq3s));
                    } else { r->d_grid_ksigns = 0; r->d_grid_iq2s = 0; r->d_grid_iq3s = 0; }
                }
                    /* 2D-grid TC: all 8 experts' gate+up in 2 launches (256 blocks each) */
                if (r->fn_moe_iq2s_tc && r->d_grid_iq2s && cl->moe_gate_exps_type == 22) {
                    {
                        void *ga[] = { &r->d_gate, &r->d_xb, &cl->moe_gate_exps_w,
                                       &r->d_topk_idx, &r->d_grid_iq2s,
                                       &n_used, &expert_ff, &n_embd,
                                       &cl->moe_exp_stride_gu };
                        cuLaunchKernel(r->fn_moe_iq2s_tc, expert_ff/16, n_used, 1, 32,1,1, 0, r->stream, ga, NULL);
                    }
                    {
                        void *ua[] = { &r->d_up, &r->d_xb, &cl->moe_up_exps_w,
                                       &r->d_topk_idx, &r->d_grid_iq2s,
                                       &n_used, &expert_ff, &n_embd,
                                       &cl->moe_exp_stride_gu };
                        cuLaunchKernel(r->fn_moe_iq2s_tc, expert_ff/64, n_used, 1, 128,1,1, 0, r->stream, ua, NULL);
                    }
                    /* Fused SiLU: all 8 experts in one launch (n_used * expert_ff = 4096) */
                    launch_silu_mul(r, r->d_gate, r->d_up, n_used * expert_ff);

                    /* IQ3_S TC down matvec for all 8 experts (1024 blocks) */
                    if (r->fn_moe_iq3s_tc && r->d_grid_iq3s && cl->moe_down_exps_type == 21) {
                        void *dna[] = { &r->d_xb2, &r->d_gate,
                                       &cl->moe_down_exps_w,
                                       &r->d_topk_idx, &r->d_grid_iq3s,
                                       &n_used, &n_embd, &expert_ff,
                                       &cl->moe_exp_stride_d };
                        cuLaunchKernel(r->fn_moe_iq3s_tc, n_embd/16, n_used, 1, 32,1,1, 0, r->stream, dna, NULL);
                        /* Fused accumulate: all 8 experts in one launch */
                    if (r->fn_moe_accum_gpu) {
                        void *aca[] = { &r->d_moe_accum, &r->d_xb2,
                                        &r->d_topk_wgt, &n_used, &n_embd };
                        cuLaunchKernel(r->fn_moe_accum_gpu, (n_embd+255)/256,1,1, 256,1,1,
                                       0, r->stream, aca, NULL);
                    } else {
                        for (int e = 0; e < n_used; e++) {
                            float wgt = htw[e];
                            CUdeviceptr down_e = r->d_xb2 + (size_t)e * n_embd * sizeof(float);
                            launch_scale_add(r, r->d_moe_accum, down_e, wgt, n_embd);
                        }
                    }
                    } else {
                        /* Scalar down per expert + accumulate */
                        for (int e = 0; e < n_used; e++) {
                            float wgt = htw[e];
                            CUdeviceptr gate_e = r->d_gate + (size_t)e * expert_ff * sizeof(float);
                            CUdeviceptr dw = cl->moe_down_exps_w + (size_t)hti[e] * cl->moe_exp_stride_d;
                            launch_moe_expert_matvec(r, r->d_xb2, dw, gate_e,
                                              cl->moe_exp_rows_d, cl->moe_exp_cols_d, cl->moe_down_exps_type);
                            launch_scale_add(r, r->d_moe_accum, r->d_xb2, wgt, n_embd);
                        }
                    }
                } else {
                    /* Scalar fallback per expert */
                    for (int e = 0; e < n_used; e++) {
                        int eidx = hti[e]; float wgt = htw[e];
                        if (eidx < 0 || eidx >= n_experts) continue;
                        CUdeviceptr gw = cl->moe_gate_exps_w + (size_t)eidx * cl->moe_exp_stride_gu;
                        CUdeviceptr uw = cl->moe_up_exps_w   + (size_t)eidx * cl->moe_exp_stride_gu;
                        CUdeviceptr dw = cl->moe_down_exps_w  + (size_t)eidx * cl->moe_exp_stride_d;
                        launch_moe_expert_matvec(r, r->d_gate, gw, r->d_xb, cl->moe_exp_rows_gu, cl->moe_exp_cols_gu, cl->moe_gate_exps_type);
                        launch_moe_expert_matvec(r, r->d_up, uw, r->d_xb, cl->moe_exp_rows_gu, cl->moe_exp_cols_gu, cl->moe_up_exps_type);
                        launch_silu_mul(r, r->d_gate, r->d_up, expert_ff);
                        launch_moe_expert_matvec(r, r->d_xb2, dw, r->d_gate, cl->moe_exp_rows_d, cl->moe_exp_cols_d, cl->moe_down_exps_type);
                        launch_scale_add(r, r->d_moe_accum, r->d_xb2, wgt, n_embd);
                    }
                }
                {
                    float shared_scale;
                    if (0 && r->fn_moe_shared_gate_gpu) {
                        void *sga[] = { &r->d_topk_wgt, &r->d_xb,
                                        &cl->moe_shared_gate_w, &n_embd };
                        cuLaunchKernel(r->fn_moe_shared_gate_gpu, 1,1,1, 128,1,1,
                                       0, r->stream, sga, NULL);
                        cuMemcpyDtoH(&shared_scale, r->d_topk_wgt, sizeof(float));
                    } else {
                        launch_matvec_f32(r, r->d_router_logits, cl->moe_shared_gate_w, r->d_xb, 1, n_embd);
                        cuStreamSynchronize(r->stream);
                        float gv; cuMemcpyDtoH(&gv, r->d_router_logits, sizeof(float));
                        shared_scale = 1.0f / (1.0f + expf(-gv));
                    }
                    /* Shared expert via TC when weights are F16 */
                    int sh_rows = cl->moe_shared_gate_rows;
                    int sh_cols = cl->moe_shared_gate_cols;
                    int su_rows = cl->moe_shared_up_rows;
                    int sd_rows = cl->moe_shared_down_rows;
                    int sd_cols = cl->moe_shared_down_cols;
                    /* Shared expert: TC for gate/up (512×2048 F16), scalar for down (2048×512 needs more parallelism) */
                    if (0 && r->fn_moe_f16_tc && cl->moe_shared_gate_type == GGML_TYPE_F16) {
                        void *sga[] = { &r->d_gate, &cl->moe_shared_ffn_gate_w, &r->d_xb, &sh_rows, &sh_cols };
                        cuLaunchKernel(r->fn_moe_f16_tc, (sh_rows+15)/16,1,1, 32,1,1, 0, r->stream, sga, NULL);
                        void *sua[] = { &r->d_up, &cl->moe_shared_ffn_up_w, &r->d_xb, &su_rows, &sh_cols };
                        cuLaunchKernel(r->fn_moe_f16_tc, (su_rows+15)/16,1,1, 32,1,1, 0, r->stream, sua, NULL);
                    } else {
                        launch_matvec_auto(r, r->d_gate, cl->moe_shared_ffn_gate_w, r->d_xb,
                                          cl->moe_shared_gate_rows, cl->moe_shared_gate_cols, cl->moe_shared_gate_type);
                        launch_matvec_auto(r, r->d_up, cl->moe_shared_ffn_up_w, r->d_xb,
                                          cl->moe_shared_up_rows, cl->moe_shared_up_cols, cl->moe_shared_up_type);
                    }
                    launch_silu_mul(r, r->d_gate, r->d_up, shared_expert_ff);
                    /* Down matvec: scalar (more parallelism with 256 blocks × 256 threads) */
                    launch_matvec_auto(r, r->d_xb2, cl->moe_shared_ffn_down_w, r->d_gate,
                                      cl->moe_shared_down_rows, cl->moe_shared_down_cols, cl->moe_shared_down_type);
                    launch_scale_add(r, r->d_moe_accum, r->d_xb2, shared_scale, n_embd);
                }
                cuMemcpyDtoDAsync(r->d_xb, r->d_moe_accum, n_embd * sizeof(float), r->stream);
            }
        } else {
            /* ---- Dense FFN ---- */
            /* FFN gate and up projections (fused, dp4a when Q8_0) */
            if (r->use_dp4a && cl->ffn_gate_type == GGML_TYPE_Q8_0 &&
                cl->ffn_gate_type == cl->ffn_up_type &&
                cl->ffn_gate_rows == cl->ffn_up_rows &&
                cl->ffn_gate_cols == cl->ffn_up_cols) {
                launch_quantize(r, r->d_xb_q, r->d_xb_scale, r->d_xb, cl->ffn_gate_cols);
                launch_matvec_dp4a_fused2(r, r->d_gate, r->d_up,
                    cl->ffn_gate_w, cl->ffn_up_w, r->d_xb_q, r->d_xb_scale,
                    cl->ffn_gate_rows, cl->ffn_gate_cols);
            } else if (cl->ffn_gate_type == cl->ffn_up_type &&
                cl->ffn_gate_rows == cl->ffn_up_rows &&
                cl->ffn_gate_cols == cl->ffn_up_cols) {
                launch_matvec_fused2(r, r->d_gate, r->d_up,
                    cl->ffn_gate_w, cl->ffn_up_w, r->d_xb,
                    cl->ffn_gate_rows, cl->ffn_gate_cols, cl->ffn_gate_type);
            } else {
                launch_matvec_auto(r, r->d_gate, cl->ffn_gate_w, r->d_xb,
                              cl->ffn_gate_rows, cl->ffn_gate_cols, cl->ffn_gate_type);
                launch_matvec_auto(r, r->d_up, cl->ffn_up_w, r->d_xb,
                              cl->ffn_up_rows, cl->ffn_up_cols, cl->ffn_up_type);
            }

            /* SiLU(gate) * up */
            launch_silu_mul(r, r->d_gate, r->d_up, n_ff);

            /* FFN down projection (dp4a for Q8_0) */
            if (r->use_dp4a && cl->ffn_down_type == GGML_TYPE_Q8_0) {
                launch_quantize(r, r->d_xb_q2, r->d_xb_scale2, r->d_gate, cl->ffn_down_cols);
                launch_matvec_q8(r, r->d_xb, cl->ffn_down_w, r->d_xb_q2, r->d_xb_scale2,
                                 cl->ffn_down_rows, cl->ffn_down_cols);
            } else {
                launch_matvec_auto(r, r->d_xb, cl->ffn_down_w, r->d_gate,
                              cl->ffn_down_rows, cl->ffn_down_cols, cl->ffn_down_type);
            }
        }

        if (dec_do_prof) { cuStreamSynchronize(r->stream); dec_prof_ffn += get_time_ms() - _dfn0; }

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

        cuda_layer_done:

        /* DeepStack injection: add deepstack slice after each early layer */
        if (r->_ds_embd && l < r->n_deepstack && r->_ds_embd_stride > n_embd) {
            const float *ds_slice = r->_ds_embd + (1 + l) * n_embd;
            cuMemcpyHtoDAsync(r->d_ds_tmp, (const void *)ds_slice, n_embd * sizeof(float), r->stream);
            launch_add(r, r->d_x, r->d_ds_tmp, n_embd);
        }

        for (int si = 0; si < r->n_hidden_snapshots; si++) {
            if (r->hidden_snapshot_layers[si] == l) {
                CUdeviceptr dst = r->d_hidden_snapshots + (size_t)si * n_embd * sizeof(float);
                cuMemcpyDtoDAsync(dst, r->d_x,
                                  (size_t)n_embd * sizeof(float), r->stream);
            }
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
    }

after_layers:

    /* End graph capture and instantiate */
    if (graph_enabled && r->graph_captured == -1 && position == 0) {
        CUgraph _g;
        CUresult _ge = cuStreamEndCapture(r->stream, &_g);
        if (_ge == CUDA_SUCCESS && _g) {
            _ge = cuGraphInstantiate(&r->d_graph_exec, _g, 0);
            if (_ge == CUDA_SUCCESS) { r->d_graph = _g; r->graph_captured = 1;
                if (r->verbose >= 1)
                    fprintf(stderr, "cuda_llm: CUDA graph captured and instantiated\n"); }
            else r->graph_captured = 0;
        } else r->graph_captured = 0;
    }

    if (dec_do_prof) {
        fprintf(stderr, "cuda_llm: decode profile (pos=%d): proj=%.2fms ssm=%.2fms attn=%.2fms ffn=%.2fms total=%.2fms\n",
                position, dec_prof_proj, dec_prof_ssm, dec_prof_attn, dec_prof_ffn,
                dec_prof_proj + dec_prof_ssm + dec_prof_attn + dec_prof_ffn);
    }

    if (apply_final_norm) {
        /* Final RMSNorm */
        launch_rmsnorm(r, r->d_x, r->d_x, r->d_output_norm, n_embd, eps);
    }

    if (copy_to_host) {
        /* Copy result to host */
        cuMemcpyDtoHAsync(r->h_output, r->d_x, n_embd * sizeof(float), r->stream);
        cuStreamSynchronize(r->stream);
    }

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

    /* Gemma4: logit soft-capping */
    if (r->is_gemma4 && r->final_logit_softcapping > 0.0f) {
        launch_logit_softcap(r, r->d_logits, r->n_vocab, r->final_logit_softcapping);
    }

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
    if (cuda_llm_bind_context(r) != 0) return NULL;

    int n_embd = r->n_embd;

    /* Upload F32 embedding to d_x (first n_embd floats) — no scaling for vision tokens */
    cuMemcpyHtoDAsync(r->d_x, (const void *)embd, n_embd * sizeof(float), r->stream);

    /* Gemma4: use pad token (ID=0) for per-layer embeddings on vision tokens */
    if (r->is_gemma4) r->current_token_id = 0;

    /* Set deepstack state for the forward loop */
    r->_ds_embd = embd;
    r->_ds_embd_stride = embd_stride;

    /* Run transformer blocks + final norm (same code path as cuda_llm_forward) */
    float *result = cuda_llm_forward_blocks(r, position, 1, 1);

    r->_ds_embd = NULL;
    r->_ds_embd_stride = 0;

    return result;
}

int cuda_llm_forward_embd_nohost(cuda_llm_runner *r, const float *embd, int embd_stride, int position) {
    if (!r || !r->weights_loaded || !embd) return -1;
    if (position < 0 || position >= r->max_seq_len) return -1;
    if (cuda_llm_bind_context(r) != 0) return -1;

    int n_embd = r->n_embd;

    cuMemcpyHtoDAsync(r->d_x, (const void *)embd, n_embd * sizeof(float), r->stream);

    if (r->is_gemma4) r->current_token_id = 0;

    r->_ds_embd = embd;
    r->_ds_embd_stride = embd_stride;

    cuda_llm_forward_blocks(r, position, 0, 0);

    r->_ds_embd = NULL;
    r->_ds_embd_stride = 0;
    return 0;
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
    if (cuda_llm_bind_context((cuda_llm_runner *)r) != 0) return -1;
    CUresult err = cuStreamSynchronize(r->stream);
    if (err != CUDA_SUCCESS) return -1;
    err = cuMemcpyDtoH(dst, r->d_x, (size_t)n * sizeof(float));
    return (err == CUDA_SUCCESS) ? 0 : -1;
}

int cuda_llm_set_hidden_snapshot_layers(cuda_llm_runner *r, const int *layers, int n_slots) {
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
    return 0;
}

int cuda_llm_read_hidden_snapshot(const cuda_llm_runner *r, int slot, float *dst, int n) {
    if (!r || !dst || n <= 0) return -1;
    if (slot < 0 || slot >= r->n_hidden_snapshots) return -1;
    if (cuda_llm_bind_context((cuda_llm_runner *)r) != 0) return -1;
    CUresult err = cuStreamSynchronize(r->stream);
    if (err != CUDA_SUCCESS) return -1;
    CUdeviceptr src = r->d_hidden_snapshots + (size_t)slot * n * sizeof(float);
    err = cuMemcpyDtoH(dst, src, (size_t)n * sizeof(float));
    return (err == CUDA_SUCCESS) ? 0 : -1;
}

int cuda_llm_read_hidden_snapshots(const cuda_llm_runner *r, float *dst, int n_slots, int n) {
    if (!r || !dst || n <= 0) return -1;
    if (n_slots < 0 || n_slots > r->n_hidden_snapshots) return -1;
    if (n_slots == 0) return 0;
    if (cuda_llm_bind_context((cuda_llm_runner *)r) != 0) return -1;
    CUresult err = cuStreamSynchronize(r->stream);
    if (err != CUDA_SUCCESS) return -1;
    err = cuMemcpyDtoH(dst, r->d_hidden_snapshots, (size_t)n_slots * n * sizeof(float));
    return (err == CUDA_SUCCESS) ? 0 : -1;
}

void cuda_llm_set_debug(cuda_llm_runner *r, int debug_layers) {
    if (r) r->debug_layers = debug_layers;
}

void cuda_llm_set_max_layers(cuda_llm_runner *r, int max_layers) {
    if (r) r->max_layers = max_layers;
}

/* ======================================================================== */
/* GPU Vision Encoder                                                       */
/* ======================================================================== */

/* ======================================================================== */
/* Batched Prefill                                                          */
/* ======================================================================== */

static void launch_batch_embed_f16(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr embd_table,
                                    CUdeviceptr token_ids, int n_embd, int n_tokens) {
    void *a[] = { &dst, &embd_table, &token_ids, &n_embd, &n_tokens };
    cuLaunchKernel(r->fn_batch_embed_f16,
                   (n_embd + 255) / 256, n_tokens, 1,
                   256, 1, 1, 0, r->stream, a, NULL);
}

/* Helper: ensure F16 shadow exists for MoE expert, then run cuBLAS matvec.
 * n_tokens = 1 for single-token decode, >1 for batched prefill.
 * If d_x_f16 is non-zero, use it as pre-converted F16 input (avoids re-conversion).
 * Returns 0 on success (cuBLAS used), -1 to fall back to custom kernel. */
static int launch_moe_expert_cublas(cuda_llm_runner *r, cuda_layer *cl,
    CUdeviceptr dst, int expert_idx, int is_down,
    CUdeviceptr x, int n_tokens, int out_dim, int in_dim,
    CUdeviceptr d_x_f16) {
    if (!r->use_cublas || !r->cublas) return -1;
    
    CUdeviceptr *w_f16;
    CUdeviceptr w_quant;
    int rows, cols, type;
    size_t raw_stride;
    CUfunction fn_dequant;
    
    if (is_down) {
        w_f16 = &cl->moe_down_exps_w_f16;
        w_quant = cl->moe_down_exps_w;
        raw_stride = cl->moe_exp_stride_d;
        type = cl->moe_down_exps_type;
        rows = cl->moe_exp_rows_d;
        cols = cl->moe_exp_cols_d;
        fn_dequant = r->fn_dequant_iq3_xxs_to_f16;
    } else {
        w_f16 = &cl->moe_gate_exps_w_f16;
        w_quant = cl->moe_gate_exps_w;
        raw_stride = cl->moe_exp_stride_gu;
        type = cl->moe_gate_exps_type;
        rows = cl->moe_exp_rows_gu;
        cols = cl->moe_exp_cols_gu;
        fn_dequant = r->fn_dequant_iq2_xxs_to_f16;
    }
    
    /* Check if F16 shadow exists */
    int word = expert_idx / 64, bit = expert_idx % 64;
    if (!((cl->moe_f16_mask[word] >> bit) & 1)) {
        int max_experts = r->n_experts < 96 ? r->n_experts : 96;
        if (expert_idx >= max_experts) return -1; /* Can't lazy-upload, fall back */
        
        if (!*w_f16) {
            size_t f16_total = (size_t)rows * cols * sizeof(uint16_t) * max_experts;
            CUresult err = cuMemAlloc(w_f16, f16_total);
            if (err != CUDA_SUCCESS) return -1;
        }
        
        size_t src_off = (size_t)expert_idx * raw_stride;
        size_t dst_off = (size_t)expert_idx * rows * cols * sizeof(uint16_t);
        CUdeviceptr src_ptr = w_quant + src_off;
        CUdeviceptr dst_ptr = *w_f16 + dst_off;
        void *args[] = { &dst_ptr, &src_ptr, &rows, &cols };
        cuLaunchKernel(fn_dequant, (rows+7)/8, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
        cuStreamSynchronize(r->stream);
        cl->moe_f16_mask[word] |= (1ULL << bit);
    }
    
    /* Get or create F16 input buffer */
    CUdeviceptr d_x_f16_local;
    if (d_x_f16) {
        d_x_f16_local = d_x_f16;  /* pre-converted (batched path) */
    } else {
        /* Single-token: convert F32→F16 on the fly */
        d_x_f16_local = r->d_batch_f16_scratch;
        if (!d_x_f16_local) return -1;
        int n_elems = n_tokens * in_dim;
        void *cv_args[] = { &d_x_f16_local, &x, &n_elems };
        cuLaunchKernel(r->fn_convert_f32_to_f16,
            (n_elems + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, cv_args, NULL);
    }
    
    /* cuBLAS F16×F16 GEMM: Y[n_tokens×out_dim] = X_F16[n_tokens×in_dim] × W_F16[out_dim×in_dim]^T */
    size_t f16_off = (size_t)expert_idx * rows * cols * sizeof(uint16_t);
    CUdeviceptr w_ptr = *w_f16 + f16_off;
    int ret = cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas,
        dst, w_ptr, d_x_f16_local, n_tokens, out_dim, in_dim);
    return (ret == 0) ? 0 : -1;
}

/* Helper: launch batched matvec for a given weight type */
/* Dequant a Q6_K or Q8_0 weight to F16 + cuBLAS tensor-core F16 GEMM.
 * When r->prefill_overlap, the weight dequant runs on a second stream into a
 * ping-pong buffer so weight[i+1]'s dequant overlaps GEMM[i] (dequant is ~19%
 * of prefill and fits entirely under the ~66% spent in GEMM). Returns 0 on
 * success, -1 to fall back. The input F32->F16 convert stays on the main stream
 * (ordered with the GEMM that reads it). */
static int launch_dequant_gemm_f16(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                    CUdeviceptr input, int out_dim, int in_dim,
                                    int n_tokens, int weight_type) {
    if (!r->cublas || !r->d_batch_f16_scratch || !r->fn_convert_f32_to_f16) return -1;
    CUfunction dq_fn; int dq_grid, dq_block;
    int nb;
    if (weight_type == GGML_TYPE_Q6_K) {
        if (!r->fn_dequant_q6_K_to_f16) return -1;
        nb = in_dim / 256; dq_grid = (out_dim * nb + 3) / 4; dq_block = 256; dq_fn = r->fn_dequant_q6_K_to_f16;
    } else if (weight_type == GGML_TYPE_Q4_K) {
        if (!r->fn_dequant_q4_K_to_f16) return -1;
        nb = in_dim / 256; dq_grid = (int)(((size_t)out_dim * nb + 127) / 128); dq_block = 128; dq_fn = r->fn_dequant_q4_K_to_f16;
    } else if (weight_type == GGML_TYPE_Q8_0) {
        if (!r->fn_dequant_q8_0_to_f16_h) return -1;
        nb = in_dim / 32; dq_grid = (out_dim * nb + 7) / 8; dq_block = 256; dq_fn = r->fn_dequant_q8_0_to_f16_h;
    } else if (weight_type == GGML_TYPE_Q4_0) {
        if (!r->fn_dequant_q4_0_to_f16) return -1;
        /* one thread per output element (coalesced); grid over rows*cols */
        dq_block = 256; dq_grid = (int)(((size_t)out_dim * in_dim + 255) / 256); dq_fn = r->fn_dequant_q4_0_to_f16;
    } else if (weight_type == GGML_TYPE_IQ2_XXS) {
        if (!r->fn_dequant_iq2_xxs_to_f16) return -1;
        /* one thread per 32-elem sub-block (coalesced); grid over rows*cols/32 */
        dq_block = 256; dq_grid = (int)(((size_t)out_dim * in_dim / 32 + 255) / 256); dq_fn = r->fn_dequant_iq2_xxs_to_f16;
    } else if (weight_type == GGML_TYPE_IQ3_XXS) {
        if (!r->fn_dequant_iq3_xxs_to_f16) return -1;
        dq_block = 256; dq_grid = (out_dim + 7) / 8; dq_fn = r->fn_dequant_iq3_xxs_to_f16;
    } else if (weight_type == GGML_TYPE_Q2_K) {
        if (!r->fn_dequant_q2_K_to_f16) return -1;
        dq_block = 256; dq_grid = (out_dim + 7) / 8; dq_fn = r->fn_dequant_q2_K_to_f16;
    } else if (weight_type == GGML_TYPE_IQ3_S) {
        if (!r->fn_dequant_iq3_s_to_f16) return -1;
        dq_block = 256; dq_grid = (out_dim + 7) / 8; dq_fn = r->fn_dequant_iq3_s_to_f16;
    } else if (weight_type == GGML_TYPE_IQ2_S) {
        if (!r->fn_dequant_iq2_s_to_f16) return -1;
        dq_block = 256; dq_grid = (out_dim + 7) / 8; dq_fn = r->fn_dequant_iq2_s_to_f16;
    } else return -1;

    size_t w_f16_bytes = (size_t)out_dim * in_dim * sizeof(uint16_t);
    if (w_f16_bytes > 256*1024*1024) return -1;
    if (!r->d_f16_scratch) {
        if (cuMemAlloc(&r->d_f16_scratch, 256*1024*1024) != CUDA_SUCCESS) return -1;
    }

    int overlap = r->prefill_overlap && r->stream_dq && r->d_f16_scratch2;
    CUstream dq_stream = overlap ? r->stream_dq : r->stream;
    int pp = overlap ? r->dq_pp : 0;
    CUdeviceptr wbuf = (overlap && pp) ? r->d_f16_scratch2 : r->d_f16_scratch;

    if (overlap) cuStreamWaitEvent(dq_stream, r->gemm_evt[pp], 0); /* buffer free? */
    void *dq[] = { &wbuf, &mat, &out_dim, &in_dim };
    cuLaunchKernel(dq_fn, dq_grid, 1, 1, dq_block, 1, 1, 0, dq_stream, dq, NULL);
    if (overlap) cuEventRecord(r->dq_evt[pp], dq_stream);

    /* input F32 -> F16 on the main stream */
    int n_elems = n_tokens * in_dim;
    CUdeviceptr d_x_f16 = r->d_batch_f16_scratch;
    void *cv[] = { &d_x_f16, &input, &n_elems };
    cuLaunchKernel(r->fn_convert_f32_to_f16, (n_elems+255)/256, 1, 1, 256, 1, 1, 0, r->stream, cv, NULL);

    if (overlap) cuStreamWaitEvent(r->stream, r->dq_evt[pp], 0); /* GEMM waits dequant */
    int ret = cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, dst, wbuf, d_x_f16,
                                                     n_tokens, out_dim, in_dim);
    if (ret != 0) return -1;
    if (overlap) { cuEventRecord(r->gemm_evt[pp], r->stream); r->dq_pp ^= 1; }
    return 0;
}

/* Dense IQ2_XXS prefill via the validated MMQ int8 tensor-core kernel
 * (mma.sync m16n8k32.s8.s8.s32). Decodes IQ2_XXS->int8 in shared memory and reads
 * the 2-bit weight ONCE (no 8x F16 materialization like launch_dequant_gemm_f16).
 * Treats the dense weight as a single expert with all n_tokens routed to it.
 * dst is [n_tokens x out_dim] row-major (matches the cuBLAS path).
 * Returns 0 on success, -1 to fall back. */
static int launch_mmq_iq2xxs_dense(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                    CUdeviceptr input, int out_dim, int in_dim, int n_tokens) {
    if (!r->fn_mmq_iq2xxs_grouped || !r->fn_mmq_quant_q8_1) return -1;
    if (out_dim % 64 != 0 || in_dim % 256 != 0 || n_tokens < 1) return -1;
    /* Prefer fused kernel: loads F32 activations and quantizes on-the-fly. */
    int use_fused = (r->fn_mmq_iq2xxs_fused32 != 0) && getenv("CUDA_LLM_MMQ_FUSED");
    int use16 = !use_fused && (r->fn_mmq_iq2xxs_grouped8 != 0) && !getenv("CUDA_LLM_MMQ_TG32") && !getenv("CUDA_LLM_MMQ_TG4");
    int use32 = !use_fused && !use16 && (r->fn_mmq_iq2xxs_grouped32 != 0) && !getenv("CUDA_LLM_MMQ_TG4");
    int grp = use_fused ? 256 : (use16 ? 128 : (use32 ? 256 : 32));
    if (r->d_mmqd_wl_ntok != n_tokens) {
        int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
        if (!r->d_mmqd_eb && cuMemAlloc(&r->d_mmqd_eb, 2*sizeof(int)) != CUDA_SUCCESS) return -1;
        if (r->d_mmqd_wl) { cuMemFree(r->d_mmqd_wl); r->d_mmqd_wl = 0; }
        if (cuMemAlloc(&r->d_mmqd_wl, (size_t)n_work*sizeof(int)) != CUDA_SUCCESS) return -1;
        int eb[2] = { 0, n_tokens };
        cuMemcpyHtoD(r->d_mmqd_eb, eb, 2*sizeof(int));
        int *wl = (int *)alloca((size_t)n_work*sizeof(int));
        for (int g = 0; g < n_work; g++) wl[g] = g;
        cuMemcpyHtoD(r->d_mmqd_wl, wl, (size_t)n_work*sizeof(int));
        r->d_mmqd_wl_ntok = n_tokens;
    }
    int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
    if (use_fused) {
        unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
        void *a[] = { &dst, &mat, &st, &input, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
        cuLaunchKernel(r->fn_mmq_iq2xxs_fused32, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL);
        return 0;
    }
    size_t need_q8 = (size_t)n_tokens * in_dim;
    if (r->d_mmqd_cap < need_q8) {
        if (r->d_mmqd_cxq8) cuMemFree(r->d_mmqd_cxq8);
        if (r->d_mmqd_cxs)  cuMemFree(r->d_mmqd_cxs);
        r->d_mmqd_cxq8 = 0; r->d_mmqd_cxs = 0; r->d_mmqd_cap = 0;
        if (cuMemAlloc(&r->d_mmqd_cxq8, need_q8) != CUDA_SUCCESS) return -1;
        if (cuMemAlloc(&r->d_mmqd_cxs, need_q8/8) != CUDA_SUCCESS) {
            cuMemFree(r->d_mmqd_cxq8); r->d_mmqd_cxq8 = 0; return -1; }
        r->d_mmqd_cap = need_q8;
    }
    { int tr = n_tokens, kk = in_dim;
      void *a[] = { &input, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &tr, &kk };
      cuLaunchKernel(r->fn_mmq_quant_q8_1, in_dim/256, n_tokens, 1, 32, 1, 1, 0, r->stream, a, NULL); }
    { unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
      CUfunction mmqfn = use32 ? r->fn_mmq_iq2xxs_grouped32 : (use16 ? r->fn_mmq_iq2xxs_grouped8 : r->fn_mmq_iq2xxs_grouped);
      void *a[] = { &dst, &mat, &st, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
      cuLaunchKernel(mmqfn, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL); }
    return 0;
}

/* Dense IQ3_XXS prefill via the validated MMQ int8 tensor-core kernel.
 * Same API and worklist convention as launch_mmq_iq2xxs_dense (IQ3_XXS variant). */
static int launch_mmq_iq3xxs_dense(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                    CUdeviceptr input, int out_dim, int in_dim, int n_tokens) {
    if (!r->fn_mmq_iq3xxs_grouped || !r->fn_mmq_quant_q8_1) return -1;
    if (out_dim % 64 != 0 || in_dim % 256 != 0 || n_tokens < 1) return -1;
    /* Prefer fused kernel: loads F32 activations and quantizes on-the-fly. */
    int use_fused = (r->fn_mmq_iq3xxs_fused32 != 0) && getenv("CUDA_LLM_MMQ_FUSED");
    int use16 = !use_fused && (r->fn_mmq_iq3xxs_grouped8 != 0) && !getenv("CUDA_LLM_MMQ_TG32") && !getenv("CUDA_LLM_MMQ_TG4");
    int use32 = !use_fused && !use16 && (r->fn_mmq_iq3xxs_grouped32 != 0) && !getenv("CUDA_LLM_MMQ_TG4");
    int grp = use_fused ? 256 : (use16 ? 128 : (use32 ? 256 : 32));
    if (r->d_mmqd_wl_ntok != n_tokens) {
        int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
        if (!r->d_mmqd_eb && cuMemAlloc(&r->d_mmqd_eb, 2*sizeof(int)) != CUDA_SUCCESS) return -1;
        if (r->d_mmqd_wl) { cuMemFree(r->d_mmqd_wl); r->d_mmqd_wl = 0; }
        if (cuMemAlloc(&r->d_mmqd_wl, (size_t)n_work*sizeof(int)) != CUDA_SUCCESS) return -1;
        int eb[2] = { 0, n_tokens };
        cuMemcpyHtoD(r->d_mmqd_eb, eb, 2*sizeof(int));
        int *wl = (int *)alloca((size_t)n_work*sizeof(int));
        for (int g = 0; g < n_work; g++) wl[g] = g;
        cuMemcpyHtoD(r->d_mmqd_wl, wl, (size_t)n_work*sizeof(int));
        r->d_mmqd_wl_ntok = n_tokens;
    }
    int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
    if (use_fused) {
        unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
        void *a[] = { &dst, &mat, &st, &input, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
        cuLaunchKernel(r->fn_mmq_iq3xxs_fused32, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL);
        return 0;
    }
    size_t need_q8 = (size_t)n_tokens * in_dim;
    if (r->d_mmqd_cap < need_q8) {
        if (r->d_mmqd_cxq8) cuMemFree(r->d_mmqd_cxq8);
        if (r->d_mmqd_cxs)  cuMemFree(r->d_mmqd_cxs);
        r->d_mmqd_cxq8 = 0; r->d_mmqd_cxs = 0; r->d_mmqd_cap = 0;
        if (cuMemAlloc(&r->d_mmqd_cxq8, need_q8) != CUDA_SUCCESS) return -1;
        if (cuMemAlloc(&r->d_mmqd_cxs, need_q8/8) != CUDA_SUCCESS) {
            cuMemFree(r->d_mmqd_cxq8); r->d_mmqd_cxq8 = 0; return -1; }
        r->d_mmqd_cap = need_q8;
    }
    { int tr = n_tokens, kk = in_dim;
      void *a[] = { &input, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &tr, &kk };
      cuLaunchKernel(r->fn_mmq_quant_q8_1, in_dim/256, n_tokens, 1, 32, 1, 1, 0, r->stream, a, NULL); }
    { unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
      CUfunction mmqfn = use32 ? r->fn_mmq_iq3xxs_grouped32 : (use16 ? r->fn_mmq_iq3xxs_grouped8 : r->fn_mmq_iq3xxs_grouped);
      void *a[] = { &dst, &mat, &st, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
      cuLaunchKernel(mmqfn, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL); }
    return 0;
}

/* Dense Q2_K prefill via the validated MMQ int8 tensor-core kernel.
 * 2-MMA approach: weight = d*dl*qs_val - dmin*ml.
 * Same API and worklist convention as launch_mmq_iq2xxs_dense. */
static int launch_mmq_q2_K_dense(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                  CUdeviceptr input, int out_dim, int in_dim, int n_tokens) {
    if (!r->fn_mmq_q2_K_grouped || !r->fn_mmq_quant_q8_1) return -1;
    if (out_dim % 64 != 0 || in_dim % 256 != 0 || n_tokens < 1) return -1;
    /* Prefer fused kernel: loads F32 activations and quantizes on-the-fly. */
    int use_fused = (r->fn_mmq_q2_K_fused32 != 0) && getenv("CUDA_LLM_MMQ_FUSED");
    int use16 = !use_fused && (r->fn_mmq_q2_K_grouped8 != 0) && !getenv("CUDA_LLM_MMQ_TG32") && !getenv("CUDA_LLM_MMQ_TG4");
    int use32 = !use_fused && !use16 && (r->fn_mmq_q2_K_grouped32 != 0) && !getenv("CUDA_LLM_MMQ_TG4");
    int grp = use_fused ? 256 : (use16 ? 128 : (use32 ? 256 : 32));
    if (r->d_mmqd_wl_ntok != n_tokens) {
        int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
        if (!r->d_mmqd_eb && cuMemAlloc(&r->d_mmqd_eb, 2*sizeof(int)) != CUDA_SUCCESS) return -1;
        if (r->d_mmqd_wl) { cuMemFree(r->d_mmqd_wl); r->d_mmqd_wl = 0; }
        if (cuMemAlloc(&r->d_mmqd_wl, (size_t)n_work*sizeof(int)) != CUDA_SUCCESS) return -1;
        int eb[2] = { 0, n_tokens };
        cuMemcpyHtoD(r->d_mmqd_eb, eb, 2*sizeof(int));
        int *wl = (int *)alloca((size_t)n_work*sizeof(int));
        for (int g = 0; g < n_work; g++) wl[g] = g;
        cuMemcpyHtoD(r->d_mmqd_wl, wl, (size_t)n_work*sizeof(int));
        r->d_mmqd_wl_ntok = n_tokens;
    }
    int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
    if (use_fused) {
        unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
        void *a[] = { &dst, &mat, &st, &input, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
        cuLaunchKernel(r->fn_mmq_q2_K_fused32, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL);
        return 0;
    }
    size_t need_q8 = (size_t)n_tokens * in_dim;
    if (r->d_mmqd_cap < need_q8) {
        if (r->d_mmqd_cxq8) cuMemFree(r->d_mmqd_cxq8);
        if (r->d_mmqd_cxs)  cuMemFree(r->d_mmqd_cxs);
        r->d_mmqd_cxq8 = 0; r->d_mmqd_cxs = 0; r->d_mmqd_cap = 0;
        if (cuMemAlloc(&r->d_mmqd_cxq8, need_q8) != CUDA_SUCCESS) return -1;
        if (cuMemAlloc(&r->d_mmqd_cxs, need_q8/8) != CUDA_SUCCESS) {
            cuMemFree(r->d_mmqd_cxq8); r->d_mmqd_cxq8 = 0; return -1; }
        r->d_mmqd_cap = need_q8;
    }
    { int tr = n_tokens, kk = in_dim;
      void *a[] = { &input, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &tr, &kk };
      cuLaunchKernel(r->fn_mmq_quant_q8_1, in_dim/256, n_tokens, 1, 32, 1, 1, 0, r->stream, a, NULL); }
    { unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
      CUfunction mmqfn = use32 ? r->fn_mmq_q2_K_grouped32 : (use16 ? r->fn_mmq_q2_K_grouped8 : r->fn_mmq_q2_K_grouped);
      void *a[] = { &dst, &mat, &st, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
      cuLaunchKernel(mmqfn, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL); }
    return 0;
}

/* Dense Q4_K prefill via the validated MMQ int8 tensor-core kernel.
 * 2-MMA approach: weight = d*sv*nibble - dmin*mv. sv (6-bit) is NOT folded into the
 * int8 weight (sv*nibble overflows): sW_plus=nibble, sW_minus=1, and d*sv / dmin*mv
 * applied as floats after the m16n8k32 MMA (which spans one 32-elem sub-block).
 * Same API and worklist convention as launch_mmq_q2_K_dense. */
static int launch_mmq_q4_K_dense(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                  CUdeviceptr input, int out_dim, int in_dim, int n_tokens) {
    if (!r->fn_mmq_q4_K_grouped || !r->fn_mmq_quant_q8_1) return -1;
    if (out_dim % 64 != 0 || in_dim % 256 != 0 || n_tokens < 1) return -1;
    int use16 = (r->fn_mmq_q4_K_grouped8 != 0) && !getenv("CUDA_LLM_MMQ_TG32") && !getenv("CUDA_LLM_MMQ_TG4");
    int use32 = !use16 && (r->fn_mmq_q4_K_grouped32 != 0) && !getenv("CUDA_LLM_MMQ_TG4");
    int grp = use16 ? 128 : (use32 ? 256 : 32);
    if (r->d_mmqd_wl_ntok != n_tokens) {
        int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
        if (!r->d_mmqd_eb && cuMemAlloc(&r->d_mmqd_eb, 2*sizeof(int)) != CUDA_SUCCESS) return -1;
        if (r->d_mmqd_wl) { cuMemFree(r->d_mmqd_wl); r->d_mmqd_wl = 0; }
        if (cuMemAlloc(&r->d_mmqd_wl, (size_t)n_work*sizeof(int)) != CUDA_SUCCESS) return -1;
        int eb[2] = { 0, n_tokens };
        cuMemcpyHtoD(r->d_mmqd_eb, eb, 2*sizeof(int));
        int *wl = (int *)alloca((size_t)n_work*sizeof(int));
        for (int g = 0; g < n_work; g++) wl[g] = g;
        cuMemcpyHtoD(r->d_mmqd_wl, wl, (size_t)n_work*sizeof(int));
        r->d_mmqd_wl_ntok = n_tokens;
    }
    int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
    size_t need_q8 = (size_t)n_tokens * in_dim;
    if (r->d_mmqd_cap < need_q8) {
        if (r->d_mmqd_cxq8) cuMemFree(r->d_mmqd_cxq8);
        if (r->d_mmqd_cxs)  cuMemFree(r->d_mmqd_cxs);
        r->d_mmqd_cxq8 = 0; r->d_mmqd_cxs = 0; r->d_mmqd_cap = 0;
        if (cuMemAlloc(&r->d_mmqd_cxq8, need_q8) != CUDA_SUCCESS) return -1;
        if (cuMemAlloc(&r->d_mmqd_cxs, need_q8/8) != CUDA_SUCCESS) {
            cuMemFree(r->d_mmqd_cxq8); r->d_mmqd_cxq8 = 0; return -1; }
        r->d_mmqd_cap = need_q8;
    }
    { int tr = n_tokens, kk = in_dim;
      void *a[] = { &input, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &tr, &kk };
      cuLaunchKernel(r->fn_mmq_quant_q8_1, in_dim/256, n_tokens, 1, 32, 1, 1, 0, r->stream, a, NULL); }
    { unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
      CUfunction mmqfn = use32 ? r->fn_mmq_q4_K_grouped32 : (use16 ? r->fn_mmq_q4_K_grouped8 : r->fn_mmq_q4_K_grouped);
      void *a[] = { &dst, &mat, &st, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
      cuLaunchKernel(mmqfn, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL); }
    return 0;
}

/* Dense Q4_0 prefill via the validated MMQ int8 tensor-core kernel.
 * SINGLE-MMA: weight = (nibble-8)*d, nibble-8 in [-8,7] folds straight into the
 * int8 weight, so result = d*d8*MMA(nibble-8, x_int8) (no plus/minus split).
 * Same API and worklist convention as launch_mmq_q4_K_dense.
 * (The gemma-4-12B "Q4_K_XL" QAT model is actually 100% Q4_0.) */
static int launch_mmq_q4_0_dense(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                  CUdeviceptr input, int out_dim, int in_dim, int n_tokens) {
    if (!r->fn_mmq_q4_0_grouped || !r->fn_mmq_quant_q8_1) return -1;
    if (out_dim % 64 != 0 || in_dim % 256 != 0 || n_tokens < 1) return -1;
    int use16 = (r->fn_mmq_q4_0_grouped8 != 0) && !getenv("CUDA_LLM_MMQ_TG32") && !getenv("CUDA_LLM_MMQ_TG4");
    int use32 = !use16 && (r->fn_mmq_q4_0_grouped32 != 0) && !getenv("CUDA_LLM_MMQ_TG4");
    int grp = use16 ? 128 : (use32 ? 256 : 32);
    if (r->d_mmqd_wl_ntok != n_tokens) {
        int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
        if (!r->d_mmqd_eb && cuMemAlloc(&r->d_mmqd_eb, 2*sizeof(int)) != CUDA_SUCCESS) return -1;
        if (r->d_mmqd_wl) { cuMemFree(r->d_mmqd_wl); r->d_mmqd_wl = 0; }
        if (cuMemAlloc(&r->d_mmqd_wl, (size_t)n_work*sizeof(int)) != CUDA_SUCCESS) return -1;
        int eb[2] = { 0, n_tokens };
        cuMemcpyHtoD(r->d_mmqd_eb, eb, 2*sizeof(int));
        int *wl = (int *)alloca((size_t)n_work*sizeof(int));
        for (int g = 0; g < n_work; g++) wl[g] = g;
        cuMemcpyHtoD(r->d_mmqd_wl, wl, (size_t)n_work*sizeof(int));
        r->d_mmqd_wl_ntok = n_tokens;
    }
    int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
    size_t need_q8 = (size_t)n_tokens * in_dim;
    if (r->d_mmqd_cap < need_q8) {
        if (r->d_mmqd_cxq8) cuMemFree(r->d_mmqd_cxq8);
        if (r->d_mmqd_cxs)  cuMemFree(r->d_mmqd_cxs);
        r->d_mmqd_cxq8 = 0; r->d_mmqd_cxs = 0; r->d_mmqd_cap = 0;
        if (cuMemAlloc(&r->d_mmqd_cxq8, need_q8) != CUDA_SUCCESS) return -1;
        if (cuMemAlloc(&r->d_mmqd_cxs, need_q8/8) != CUDA_SUCCESS) {
            cuMemFree(r->d_mmqd_cxq8); r->d_mmqd_cxq8 = 0; return -1; }
        r->d_mmqd_cap = need_q8;
    }
    { int tr = n_tokens, kk = in_dim;
      void *a[] = { &input, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &tr, &kk };
      cuLaunchKernel(r->fn_mmq_quant_q8_1, in_dim/256, n_tokens, 1, 32, 1, 1, 0, r->stream, a, NULL); }
    { unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
      CUfunction mmqfn = use32 ? r->fn_mmq_q4_0_grouped32 : (use16 ? r->fn_mmq_q4_0_grouped8 : r->fn_mmq_q4_0_grouped);
      void *a[] = { &dst, &mat, &st, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
      cuLaunchKernel(mmqfn, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL); }
    return 0;
}

/* Dense Q3_K prefill via the int8 tensor-core MMQ (2 x m16n8k16 per sub-block).
 * Same API and worklist convention as launch_mmq_iq2xxs_dense. */
static int launch_mmq_q3_K_dense(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                  CUdeviceptr input, int out_dim, int in_dim, int n_tokens) {
    if (!r->fn_mmq_q3_K_grouped || !r->fn_mmq_quant_q8_1) return -1;
    if (out_dim % 64 != 0 || in_dim % 256 != 0 || n_tokens < 1) return -1;
    /* Prefer fused kernel: loads F32 activations and quantizes on-the-fly. */
    int use_fused = (r->fn_mmq_q3_K_fused32 != 0) && getenv("CUDA_LLM_MMQ_FUSED");
    int use16 = !use_fused && (r->fn_mmq_q3_K_grouped8 != 0) && !getenv("CUDA_LLM_MMQ_TG32") && !getenv("CUDA_LLM_MMQ_TG4");
    int use32 = !use_fused && !use16 && (r->fn_mmq_q3_K_grouped32 != 0) && !getenv("CUDA_LLM_MMQ_TG4");
    int grp = use_fused ? 256 : (use16 ? 128 : (use32 ? 256 : 32));
    if (r->d_mmqd_wl_ntok != n_tokens) {
        int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
        if (!r->d_mmqd_eb && cuMemAlloc(&r->d_mmqd_eb, 2*sizeof(int)) != CUDA_SUCCESS) return -1;
        if (r->d_mmqd_wl) { cuMemFree(r->d_mmqd_wl); r->d_mmqd_wl = 0; }
        if (cuMemAlloc(&r->d_mmqd_wl, (size_t)n_work*sizeof(int)) != CUDA_SUCCESS) return -1;
        int eb[2] = { 0, n_tokens };
        cuMemcpyHtoD(r->d_mmqd_eb, eb, 2*sizeof(int));
        int *wl = (int *)alloca((size_t)n_work*sizeof(int));
        for (int g = 0; g < n_work; g++) wl[g] = g;
        cuMemcpyHtoD(r->d_mmqd_wl, wl, (size_t)n_work*sizeof(int));
        r->d_mmqd_wl_ntok = n_tokens;
    }
    int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
    if (use_fused) {
        unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
        void *a[] = { &dst, &mat, &st, &input, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
        cuLaunchKernel(r->fn_mmq_q3_K_fused32, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL);
        return 0;
    }
    size_t need_q8 = (size_t)n_tokens * in_dim;
    if (r->d_mmqd_cap < need_q8) {
        if (r->d_mmqd_cxq8) cuMemFree(r->d_mmqd_cxq8);
        if (r->d_mmqd_cxs)  cuMemFree(r->d_mmqd_cxs);
        r->d_mmqd_cxq8 = 0; r->d_mmqd_cxs = 0; r->d_mmqd_cap = 0;
        if (cuMemAlloc(&r->d_mmqd_cxq8, need_q8) != CUDA_SUCCESS) return -1;
        if (cuMemAlloc(&r->d_mmqd_cxs, need_q8/8) != CUDA_SUCCESS) {
            cuMemFree(r->d_mmqd_cxq8); r->d_mmqd_cxq8 = 0; return -1; }
        r->d_mmqd_cap = need_q8;
    }
    { int tr = n_tokens, kk = in_dim;
      void *a[] = { &input, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &tr, &kk };
      cuLaunchKernel(r->fn_mmq_quant_q8_1, in_dim/256, n_tokens, 1, 32, 1, 1, 0, r->stream, a, NULL); }
    { unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
      CUfunction mmqfn = use32 ? r->fn_mmq_q3_K_grouped32 : (use16 ? r->fn_mmq_q3_K_grouped8 : r->fn_mmq_q3_K_grouped);
      void *a[] = { &dst, &mat, &st, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
      cuLaunchKernel(mmqfn, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL); }
    return 0;
}

/* Dense IQ3_S prefill via the validated MMQ int8 tensor-core kernel.
 * Same API and worklist convention as launch_mmq_iq2xxs_dense (IQ3_S variant). */
static int launch_mmq_iq3_s_dense(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                    CUdeviceptr input, int out_dim, int in_dim, int n_tokens) {
    if (!r->fn_mmq_iq3_s_grouped || !r->fn_mmq_quant_q8_1) return -1;
    if (out_dim % 64 != 0 || in_dim % 256 != 0 || n_tokens < 1) return -1;
    /* Prefer fused kernel: loads F32 activations and quantizes on-the-fly,
       eliminating the separate mmq_quant_q8_1 launch and global memory round-trip. */
    int use_fused = (r->fn_mmq_iq3_s_fused32 != 0) && getenv("CUDA_LLM_MMQ_FUSED");
    int use16 = !use_fused && (r->fn_mmq_iq3_s_grouped8 != 0) && !getenv("CUDA_LLM_MMQ_TG32") && !getenv("CUDA_LLM_MMQ_TG4");
    int use32 = !use_fused && !use16 && (r->fn_mmq_iq3_s_grouped32 != 0) && !getenv("CUDA_LLM_MMQ_TG4");
    int grp = use_fused ? 256 : (use16 ? 128 : (use32 ? 256 : 32));
    if (r->d_mmqd_wl_ntok != n_tokens) {
        int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
        if (!r->d_mmqd_eb && cuMemAlloc(&r->d_mmqd_eb, 2*sizeof(int)) != CUDA_SUCCESS) return -1;
        if (r->d_mmqd_wl) { cuMemFree(r->d_mmqd_wl); r->d_mmqd_wl = 0; }
        if (cuMemAlloc(&r->d_mmqd_wl, (size_t)n_work*sizeof(int)) != CUDA_SUCCESS) return -1;
        int eb[2] = { 0, n_tokens };
        cuMemcpyHtoD(r->d_mmqd_eb, eb, 2*sizeof(int));
        int *wl = (int *)alloca((size_t)n_work*sizeof(int));
        for (int g = 0; g < n_work; g++) wl[g] = g;
        cuMemcpyHtoD(r->d_mmqd_wl, wl, (size_t)n_work*sizeof(int));
        r->d_mmqd_wl_ntok = n_tokens;
    }
    int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
    if (use_fused) {
        unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
        void *a[] = { &dst, &mat, &st, &input, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
        cuLaunchKernel(r->fn_mmq_iq3_s_fused32, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL);
        return 0;
    }
    size_t need_q8 = (size_t)n_tokens * in_dim;
    if (r->d_mmqd_cap < need_q8) {
        if (r->d_mmqd_cxq8) cuMemFree(r->d_mmqd_cxq8);
        if (r->d_mmqd_cxs)  cuMemFree(r->d_mmqd_cxs);
        r->d_mmqd_cxq8 = 0; r->d_mmqd_cxs = 0; r->d_mmqd_cap = 0;
        if (cuMemAlloc(&r->d_mmqd_cxq8, need_q8) != CUDA_SUCCESS) return -1;
        if (cuMemAlloc(&r->d_mmqd_cxs, need_q8/8) != CUDA_SUCCESS) {
            cuMemFree(r->d_mmqd_cxq8); r->d_mmqd_cxq8 = 0; return -1; }
        r->d_mmqd_cap = need_q8;
    }
    { int tr = n_tokens, kk = in_dim;
      void *a[] = { &input, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &tr, &kk };
      cuLaunchKernel(r->fn_mmq_quant_q8_1, in_dim/256, n_tokens, 1, 32, 1, 1, 0, r->stream, a, NULL); }
    { unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
      CUfunction mmqfn = use32 ? r->fn_mmq_iq3_s_grouped32 : (use16 ? r->fn_mmq_iq3_s_grouped8 : r->fn_mmq_iq3_s_grouped);
      void *a[] = { &dst, &mat, &st, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
      cuLaunchKernel(mmqfn, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL); }
    return 0;
}

/* Dense IQ2_S prefill via the validated MMQ int8 tensor-core kernel.
 * 2-MMA approach (m16n8k16 x 2 per sub-block).
 * Same API and worklist convention as launch_mmq_iq2xxs_dense. */
static int launch_mmq_iq2_s_dense(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                    CUdeviceptr input, int out_dim, int in_dim, int n_tokens) {
    if (!r->fn_mmq_iq2_s_grouped || !r->fn_mmq_quant_q8_1) return -1;
    if (out_dim % 64 != 0 || in_dim % 256 != 0 || n_tokens < 1) return -1;
    /* Prefer fused kernel: loads F32 activations and quantizes on-the-fly. */
    int use_fused = (r->fn_mmq_iq2_s_fused32 != 0) && getenv("CUDA_LLM_MMQ_FUSED");
    int use16 = !use_fused && (r->fn_mmq_iq2_s_grouped8 != 0) && !getenv("CUDA_LLM_MMQ_TG32") && !getenv("CUDA_LLM_MMQ_TG4");
    int use32 = !use_fused && !use16 && (r->fn_mmq_iq2_s_grouped32 != 0) && !getenv("CUDA_LLM_MMQ_TG4");
    int grp = use_fused ? 256 : (use16 ? 128 : (use32 ? 256 : 32));
    if (r->d_mmqd_wl_ntok != n_tokens) {
        int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
        if (!r->d_mmqd_eb && cuMemAlloc(&r->d_mmqd_eb, 2*sizeof(int)) != CUDA_SUCCESS) return -1;
        if (r->d_mmqd_wl) { cuMemFree(r->d_mmqd_wl); r->d_mmqd_wl = 0; }
        if (cuMemAlloc(&r->d_mmqd_wl, (size_t)n_work*sizeof(int)) != CUDA_SUCCESS) return -1;
        int eb[2] = { 0, n_tokens };
        cuMemcpyHtoD(r->d_mmqd_eb, eb, 2*sizeof(int));
        int *wl = (int *)alloca((size_t)n_work*sizeof(int));
        for (int g = 0; g < n_work; g++) wl[g] = g;
        cuMemcpyHtoD(r->d_mmqd_wl, wl, (size_t)n_work*sizeof(int));
        r->d_mmqd_wl_ntok = n_tokens;
    }
    int n_work = (n_tokens + grp - 1) / grp; if (n_work < 1) n_work = 1;
    if (use_fused) {
        unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
        void *a[] = { &dst, &mat, &st, &input, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
        cuLaunchKernel(r->fn_mmq_iq2_s_fused32, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL);
        return 0;
    }
    size_t need_q8 = (size_t)n_tokens * in_dim;
    if (r->d_mmqd_cap < need_q8) {
        if (r->d_mmqd_cxq8) cuMemFree(r->d_mmqd_cxq8);
        if (r->d_mmqd_cxs)  cuMemFree(r->d_mmqd_cxs);
        r->d_mmqd_cxq8 = 0; r->d_mmqd_cxs = 0; r->d_mmqd_cap = 0;
        if (cuMemAlloc(&r->d_mmqd_cxq8, need_q8) != CUDA_SUCCESS) return -1;
        if (cuMemAlloc(&r->d_mmqd_cxs, need_q8/8) != CUDA_SUCCESS) {
            cuMemFree(r->d_mmqd_cxq8); r->d_mmqd_cxq8 = 0; return -1; }
        r->d_mmqd_cap = need_q8;
    }
    { int tr = n_tokens, kk = in_dim;
      void *a[] = { &input, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &tr, &kk };
      cuLaunchKernel(r->fn_mmq_quant_q8_1, in_dim/256, n_tokens, 1, 32, 1, 1, 0, r->stream, a, NULL); }
    { unsigned long long st = 0; int bm = 0, nN = out_dim, nK = in_dim;
      CUfunction mmqfn = use32 ? r->fn_mmq_iq2_s_grouped32 : (use16 ? r->fn_mmq_iq2_s_grouped8 : r->fn_mmq_iq2_s_grouped);
      void *a[] = { &dst, &mat, &st, &r->d_mmqd_cxq8, &r->d_mmqd_cxs, &r->d_mmqd_eb, &r->d_mmqd_wl, &bm, &nN, &nK };
      cuLaunchKernel(mmqfn, out_dim/64, n_work, 1, 128, 1, 1, 0, r->stream, a, NULL); }
    return 0;
}

/* Max n_tokens for which dense int8 MMQ beats the dequant->F16->cuBLAS GEMM.
 * MMQ wins at small batch (skinny cuBLAS GEMM); cuBLAS wins at large batch (weight
 * reuse). Default crossover ~256 (measured on gemma-4-12B Q4_0, RTX 5060 Ti).
 * CUDA_LLM_MMQ_MAX_TOKENS overrides: 0 disables MMQ, a large value forces it. */
static int cllm_mmq_dense_max_tokens(void) {
    const char *e = getenv("CUDA_LLM_MMQ_MAX_TOKENS");
    if (e) { int v = atoi(e); return v < 0 ? 0 : v; }
    if (getenv("CUDA_LLM_NO_MMQ_DENSE")) return 0;
    return 256;
}

static void launch_batch_matvec(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                 CUdeviceptr mat_f16, CUdeviceptr input, int out_dim, int in_dim,
                                 int n_tokens, int weight_type);

/* Same as launch_batch_matvec but routes all launches (dequant, conversions,
 * cuBLAS GEMM) onto the caller-supplied stream. Used for FFN gate/up
 * parallelization via CUDA_LLM_PAR_FFN. */
static void launch_batch_matvec_stream(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                       CUdeviceptr mat_f16, CUdeviceptr input, int out_dim, int in_dim,
                                       int n_tokens, int weight_type, CUstream stream) {
    CUstream saved = r->stream;
    r->stream = stream;
    launch_batch_matvec(r, dst, mat, mat_f16, input, out_dim, in_dim, n_tokens, weight_type);
    r->stream = saved;
}

static void launch_batch_matvec(cuda_llm_runner *r, CUdeviceptr dst, CUdeviceptr mat,
                                 CUdeviceptr mat_f16, CUdeviceptr input, int out_dim, int in_dim,
                                 int n_tokens, int weight_type) {
    /* cuBLAS path: use F16 shadow + Tensor Core GEMM when available (batched only). */
    if (n_tokens > 1 && r->use_cublas && mat_f16 && r->cublas) {
        /* Convert F32 input to F16 (required for Blackwell Tensor Core GEMM) */
        int n_elems = n_tokens * in_dim;
        CUdeviceptr d_x_f16 = r->d_batch_f16_scratch;
        if (d_x_f16) {
            void *args[] = { &d_x_f16, &input, &n_elems };
            cuLaunchKernel(r->fn_convert_f32_to_f16,
                (n_elems + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
            int ret = cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas,
                dst, mat_f16, d_x_f16,
                n_tokens, out_dim, in_dim);
            if (ret == 0) return;
        }
        /* Fall through to custom kernel on failure */
    }

    if (weight_type == GGML_TYPE_F32) {
        if (r->use_cublas && r->cublas && n_tokens > 1) {
            int ret = cublasew_gemm_f32_rowmajor_nt(r->cublas, dst, mat, input,
                                                     n_tokens, out_dim, in_dim);
            if (ret == 0) return;
        }
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_f32_f32, out_dim, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_Q8_0) {
        /* Dequant Q8_0 → F16 + cuBLAS tensor-core F16 GEMM (reads weights once).
         * F16 tensor cores are far faster than the F32 SIMT GEMM the old path used. */
        if (n_tokens > 1) {
            if (launch_dequant_gemm_f16(r, dst, mat, input, out_dim, in_dim, n_tokens, weight_type) == 0)
                return;
        }
        /* Legacy F32 dequant + SIMT GEMM fallback */
        if (n_tokens > 1 && r->fn_dequant_q8_0_to_f16 && r->cublas && r->d_f16_scratch) {
            CUdeviceptr d_f32 = r->d_f16_scratch;
            int nb = in_dim / 32;
            int total_blocks = out_dim * nb;
            int dequant_grid = (total_blocks + 31) / 32;
            void *args[] = { &d_f32, &mat, &out_dim, &in_dim };
            cuLaunchKernel(r->fn_dequant_q8_0_to_f16, dequant_grid, 1, 1, 32, 1, 1, 0, r->stream, args, NULL);
            int gemm_ret = cublasew_gemm_f32_rowmajor_nt(r->cublas, dst, d_f32, input,
                                                          n_tokens, out_dim, in_dim);
            if (gemm_ret == 0) return;
        }
        /* Fallback: x4 chunked (4 tok/block) or per-token */
        if (n_tokens > 1 && r->fn_batch_matvec_q8_0_x4) {
            void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
            cuLaunchKernel(r->fn_batch_matvec_q8_0_x4, (out_dim+7)/8, (n_tokens+3)/4, 1, 256, 1, 1, 0, r->stream, a, NULL);
        } else {
            void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
            cuLaunchKernel(r->fn_batch_matvec_q8_0_f32, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
        }
    } else if (weight_type == GGML_TYPE_Q2_K) {
        /* Prefer the int8 tensor-core MMQ for all batch sizes.
           CUDA_LLM_NO_MMQ_DENSE forces the dequant->cuBLAS path. */
        if (!getenv("CUDA_LLM_NO_MMQ_DENSE") &&
            launch_mmq_q2_K_dense(r, dst, mat, input, out_dim, in_dim, n_tokens) == 0)
            return;
        if (n_tokens > 1 && launch_dequant_gemm_f16(r, dst, mat, input, out_dim, in_dim, n_tokens, weight_type) == 0)
            return;
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_q2_K, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_Q3_K) {
        /* Prefer the int8 tensor-core MMQ for all batch sizes. */
        if (!getenv("CUDA_LLM_NO_MMQ_DENSE") &&
            launch_mmq_q3_K_dense(r, dst, mat, input, out_dim, in_dim, n_tokens) == 0)
            return;
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_q3_K, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_F16) {
        /* F16 weight is already F16: convert input -> F16 + cuBLAS tensor-core GEMM. */
        if (n_tokens > 1 && r->cublas && r->d_batch_f16_scratch && r->fn_convert_f32_to_f16) {
            int n_elems = n_tokens * in_dim;
            CUdeviceptr d_x_f16 = r->d_batch_f16_scratch;
            void *cv[] = { &d_x_f16, &input, &n_elems };
            cuLaunchKernel(r->fn_convert_f32_to_f16, (n_elems+255)/256, 1, 1, 256, 1, 1, 0, r->stream, cv, NULL);
            if (cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, dst, mat, d_x_f16,
                                                      n_tokens, out_dim, in_dim) == 0)
                return;
        }
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_vision_linear_f16, out_dim, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_IQ2_XXS) {
        /* Prefer the int8 tensor-core MMQ for all batch sizes.
           CUDA_LLM_NO_MMQ_DENSE forces the dequant->cuBLAS path. */
        if (!getenv("CUDA_LLM_NO_MMQ_DENSE") &&
            launch_mmq_iq2xxs_dense(r, dst, mat, input, out_dim, in_dim, n_tokens) == 0)
            return;
        if (n_tokens > 1 && launch_dequant_gemm_f16(r, dst, mat, input, out_dim, in_dim, n_tokens, weight_type) == 0)
            return;
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_iq2_xxs, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_IQ2_XS) {
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_iq2_xs, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_IQ2_S) {
        /* Prefer the int8 tensor-core MMQ for all batch sizes. */
        if (!getenv("CUDA_LLM_NO_MMQ_DENSE") &&
            launch_mmq_iq2_s_dense(r, dst, mat, input, out_dim, in_dim, n_tokens) == 0)
            return;
        if (n_tokens > 1 && launch_dequant_gemm_f16(r, dst, mat, input, out_dim, in_dim, n_tokens, weight_type) == 0)
            return;
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_iq2_s, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_IQ3_XXS) {
        /* Prefer the int8 tensor-core MMQ for all batch sizes. */
        if (!getenv("CUDA_LLM_NO_MMQ_DENSE") &&
            launch_mmq_iq3xxs_dense(r, dst, mat, input, out_dim, in_dim, n_tokens) == 0)
            return;
        if (n_tokens > 1 && launch_dequant_gemm_f16(r, dst, mat, input, out_dim, in_dim, n_tokens, weight_type) == 0)
            return;
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_iq3_xxs, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_IQ3_S) {
        /* Prefer the int8 tensor-core MMQ for all batch sizes. */
        if (!getenv("CUDA_LLM_NO_MMQ_DENSE") &&
            launch_mmq_iq3_s_dense(r, dst, mat, input, out_dim, in_dim, n_tokens) == 0)
            return;
        if (n_tokens > 1 && launch_dequant_gemm_f16(r, dst, mat, input, out_dim, in_dim, n_tokens, weight_type) == 0)
            return;
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_iq3_s, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_IQ4_NL) {
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_iq4_nl, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_IQ4_XS) {
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_iq4_xs, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_Q4_0) {
        /* Crossover: dense int8 MMQ wins only at SMALL batch (skinny cuBLAS GEMM);
         * at large batch the dequant->F16->cuBLAS tensor-core GEMM has far better
         * weight reuse and wins (measured gemma-4-12B Q4_0, RTX 5060 Ti:
         * pp64/128 MMQ 519/784 vs cuBLAS 345/627; pp512/1024/2048 MMQ 1044/1031/1016
         * vs cuBLAS 1240/1460/1526). So use MMQ only for n_tokens <= threshold.
         * CUDA_LLM_MMQ_MAX_TOKENS overrides (0 disables MMQ; large value forces it). */
        if (n_tokens > 1 && n_tokens <= cllm_mmq_dense_max_tokens() &&
            launch_mmq_q4_0_dense(r, dst, mat, input, out_dim, in_dim, n_tokens) == 0)
            return;
        /* Dequant Q4_0 -> F16 (once) + cuBLAS F16 tensor-core GEMM, like Q6_K/Q8_0.
         * launch_dequant_gemm_f16 also converts the F32 input to F16 (the GEMM
         * needs F16 on both sides), applies the >256 MB guard (skips the 2 GB
         * lm_head), and lazily allocates d_f16_scratch. */
        if (n_tokens > 1) {
            if (launch_dequant_gemm_f16(r, dst, mat, input, out_dim, in_dim, n_tokens, weight_type) == 0)
                return;
        }
        /* Fallback (n_tokens==1, oversized lm_head, or no cuBLAS): per-token
         * matvec_q4_0_f32, the known-correct decode kernel. */
        for (int t = 0; t < n_tokens; t++) {
            launch_matvec_q4_0(r, dst + (size_t)t * out_dim * sizeof(float), mat,
                               input + (size_t)t * in_dim * sizeof(float), out_dim, in_dim);
        }
    } else if (weight_type == GGML_TYPE_Q4_K) {
        /* Skip dequant+cuBLAS path and use the x4 chunked batch kernel directly.
         * The dequant materializes 113 MB F16 per call, which is the FFN
         * bottleneck on this GPU; the x4 kernel reads Q4_K directly and
         * computes 4 tokens per block. CUDA_LLM_Q4K_X4=1 forces this. */
        if (n_tokens > 1 && getenv("CUDA_LLM_Q4K_X4") && r->fn_batch_matvec_q4_K_x4) {
            int n_groups = (n_tokens + 3) / 4;
            void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
            cuLaunchKernel(r->fn_batch_matvec_q4_K_x4, (out_dim+7)/8, n_groups, 1, 256, 1, 1, 0, r->stream, a, NULL);
            return;
        }
        /* Dense int8 MMQ only at small batch (crossover; see the Q4_0 arm above).
         * Large-batch prefill uses the dequant->F16->cuBLAS GEMM (better weight reuse). */
        if (n_tokens > 1 && n_tokens <= cllm_mmq_dense_max_tokens() &&
            launch_mmq_q4_K_dense(r, dst, mat, input, out_dim, in_dim, n_tokens) == 0)
            return;
        /* Dequant + cuBLAS F16 GEMM via the shared helper (reuses the 256 MB
         * pre-allocated d_f16_scratch + double-buffered overlap) — same fast path
         * used by Q6_K/Q4_0. Replaces the old per-call cuMemAlloc/cuMemFree which
         * allocated 113 MB per FFN gate/up call and was the FFN bottleneck. */
        if (n_tokens > 1) {
            if (launch_dequant_gemm_f16(r, dst, mat, input, out_dim, in_dim, n_tokens, weight_type) == 0)
                return;
        }
        /* Fallback (n_tokens==1, oversized lm_head, or no cuBLAS): x4 chunked batch kernel. */
        if (n_tokens > 1 && r->fn_batch_matvec_q4_K_x4) {
            int n_groups = (n_tokens + 3) / 4;
            void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
            cuLaunchKernel(r->fn_batch_matvec_q4_K_x4, (out_dim+7)/8, n_groups, 1, 256, 1, 1, 0, r->stream, a, NULL);
        } else {
            void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
            cuLaunchKernel(r->fn_batch_matvec_q4_K, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
        }
    } else if (weight_type == GGML_TYPE_Q5_K) {
        void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
        cuLaunchKernel(r->fn_batch_matvec_q5_K, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
    } else if (weight_type == GGML_TYPE_Q6_K) {
        /* Dequant Q6_K -> F16 (once) + cuBLAS F16 tensor-core GEMM (reads the
         * weight once vs ceil(n_tokens/4)×; tensor cores; double-buffered overlap). */
        if (n_tokens > 1) {
            if (launch_dequant_gemm_f16(r, dst, mat, input, out_dim, in_dim, n_tokens, weight_type) == 0)
                return;
        }
        /* Use chunked kernel (4 tokens/block) for batched mode, fallback for single token */
        if (n_tokens > 1 && r->fn_batch_matvec_q6_K_x4) {
            int n_groups = (n_tokens + 3) / 4;
            void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
            cuLaunchKernel(r->fn_batch_matvec_q6_K_x4, (out_dim+7)/8, n_groups, 1, 256, 1, 1, 0, r->stream, a, NULL);
        } else {
            void *a[] = { &dst, &mat, &input, &out_dim, &in_dim, &n_tokens };
            cuLaunchKernel(r->fn_batch_matvec_q6_K, (out_dim+7)/8, n_tokens, 1, 256, 1, 1, 0, r->stream, a, NULL);
        }
    } else {
        /* Fallback: sequential per-token matvec */
        for (int t = 0; t < n_tokens; t++) {
            CUdeviceptr in_t = input + (size_t)t * in_dim * sizeof(float);
            CUdeviceptr out_t = dst + (size_t)t * out_dim * sizeof(float);
            launch_matvec_auto(r, out_t, mat, in_t, out_dim, in_dim, weight_type);
        }
    }
}

static void launch_batch_softplus_mul(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr in,
                                       CUdeviceptr bias, CUdeviceptr a, int n, int n_tokens) {
    int total = n * n_tokens;
    void *args[] = { &out, &in, &bias, &a, &n, &n_tokens };
    cuLaunchKernel(r->fn_batch_softplus_mul_f32,
                   (total + 255) / 256, 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void launch_batch_rope(cuda_llm_runner *r, CUdeviceptr vec, int heads_per_token,
                               int n_tokens, int head_dim, int start_pos, float freq_base,
                               int n_rope_pairs) {
    int total_heads = heads_per_token * n_tokens;
    int half_dim = head_dim / 2;
    void *args[] = { &vec, &heads_per_token, &total_heads, &head_dim,
                     &start_pos, &freq_base, &n_rope_pairs };
    cuLaunchKernel(r->fn_batch_rope_neox_f32,
                   total_heads, 1, 1,
                   half_dim, 1, 1, 0, r->stream, args, NULL);
}

static void launch_batch_conv1d(cuda_llm_runner *r, CUdeviceptr data, CUdeviceptr conv_state,
                                 CUdeviceptr weight, int qkv_dim, int conv_k, int n_tokens) {
    void *args[] = { &data, &conv_state, &weight, &qkv_dim, &conv_k, &n_tokens };
    cuLaunchKernel(r->fn_batch_conv1d_depthwise_silu_f32,
                   (qkv_dim + 255) / 256, 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void launch_batch_deltanet_scan(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr state,
                                        CUdeviceptr qkv, CUdeviceptr alpha, CUdeviceptr beta,
                                        int n_tokens, int dt_rank, int d_state, int n_group) {
    void *args[] = { &out, &state, &qkv, &alpha, &beta, &n_tokens, &dt_rank, &d_state, &n_group };
    /* W independent warps/block (d_state==128 register path only) raises warps/SM past
       the 1-warp/block occupancy cap. No shared/sync; r = blockIdx.y*W + threadIdx.y.
       CUDA_LLM_SCAN_W in {1,2,4}; only used when d_state==128. */
    static int scan_W = -1;
    if (scan_W < 0) {
        const char *e = getenv("CUDA_LLM_SCAN_W");
        scan_W = e && e[0] ? atoi(e) : 4;
        if (scan_W != 1 && scan_W != 2 && scan_W != 4) scan_W = 2;
    }
    int W = (d_state == 128 && (d_state % scan_W) == 0) ? scan_W : 1;
    cuLaunchKernel(r->fn_batch_deltanet_scan_f32,
                   dt_rank, d_state / W, 1,
                   32, W, 1, W > 1 ? 0 : sizeof(float) * d_state, r->stream, args, NULL);
}

static void launch_batch_l2_norm_heads_strided(cuda_llm_runner *r, CUdeviceptr vec,
                                                    int heads_per_token, int n_tokens,
                                                    int head_dim, int token_stride,
                                                    float eps) {
    int total_heads = heads_per_token * n_tokens;
    int threads = 1;
    while (threads < head_dim && threads < 256) threads <<= 1;
    if (threads < 32) threads = 32;
    void *args[] = { &vec, &heads_per_token, &n_tokens, &head_dim, &token_stride, &eps };
    cuLaunchKernel(r->fn_batch_l2_norm_heads_strided_f32,
                   total_heads, 1, 1,
                   threads, 1, 1, threads * sizeof(float), r->stream, args, NULL);
}

static void launch_batch_attention(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr q_batch,
                                    CUdeviceptr key_cache, CUdeviceptr value_cache,
                                    int n_tokens, int start_pos, int n_heads, int n_kv_heads,
                                    int head_dim, int kv_dim, float scale) {
    size_t smem = (size_t)(start_pos + n_tokens) * sizeof(float);
    void *args[] = { &out, &q_batch, &key_cache, &value_cache, &n_tokens, &start_pos,
                     &n_heads, &n_kv_heads, &head_dim, &kv_dim, &scale };
    cuLaunchKernel(r->fn_batch_attn_causal_f32,
                   n_heads, n_tokens, 1,
                    256, 1, 1, smem, r->stream, args, NULL);
}

/* ---- FA2 Flash Attention launch helper (flat layout) ---- */
/* Runs fa2_attn_f kernel for one layer's attention.
   Q_batch: F32 [n_tokens * q_dim] -> converted to F16 on the fly
   K/V cache: F16 [max_seq_len * kv_dim] flat layout
   Output: F16 -> converted back to F32 [n_tokens * q_dim] */
static void launch_fa2_attention(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr q_batch,
                                  CUdeviceptr key_cache, CUdeviceptr value_cache,
                                  int n_tokens, int start_pos, int n_heads, int n_kv_heads,
                                  int head_dim, int kv_dim, float scale, int window) {
    if (!r->fn_fa2_attn_256 || head_dim != 256) return;
    int q_dim = n_heads * head_dim;
    int hstride_q = q_dim;  /* Q layout: [n_tokens, q_dim] */
    int hstride_kv = kv_dim;  /* KV layout: [max_seq_len, kv_dim] */
    int gqa_ratio = n_heads / n_kv_heads;

    /* Convert Q from F32 to F16 */
    int total_q = n_tokens * q_dim;
    CUdeviceptr d_q_f16 = r->d_fa2_q_f16;
    if (d_q_f16) {
        void *args_conv[] = { &d_q_f16, &q_batch, &total_q };
        cuLaunchKernel(r->fn_convert_f32_to_f16,
            (total_q + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args_conv, NULL);
    }
    CUdeviceptr d_o_f16 = r->d_fa2_o_f16 ? r->d_fa2_o_f16 : d_q_f16;

    /* Run FA2 - process all tokens in one call.
     * Keys span [0, total_s); the n_tokens queries sit at global positions
     * [start_pos, total_s). start_pos>0 (chunked prefill, queries attend a
     * [history ++ chunk] key buffer) feeds q_off=start_pos; start_pos==0 gives
     * n_q==S, q_off==0 -> byte-identical to the original kernel. */
    int total_s = start_pos + n_tokens;
    int fa2_n_q = n_tokens;
    int fa2_q_off = start_pos;
#define FA2_BR_HOST 64
#define FA2_BC_HOST 32
#define FA2_NTHR_HOST 128
    int n_q_tiles = (n_tokens + FA2_BR_HOST - 1) / FA2_BR_HOST;
    size_t fa2_smem = 4 * FA2_BC_HOST * (head_dim + 8) * 2; /* 2 = sizeof(dt_t) = F16 */
    void *fa2_args[] = { &d_o_f16, &d_q_f16, &key_cache, &value_cache,
                         &total_s, &head_dim, &hstride_q, &hstride_kv, &gqa_ratio, &scale, &window,
                         &fa2_n_q, &fa2_q_off };
    cuLaunchKernel(r->fn_fa2_attn_256,
        (unsigned)n_q_tiles, (unsigned)n_heads, 1,
        FA2_NTHR_HOST, 1, 1, (unsigned)fa2_smem, r->stream, fa2_args, NULL);

    /* Convert output from F16 back to F32 */
    if (r->fn_convert_f16_to_f32) {
        void *args_deconv[] = { &out, &d_o_f16, &total_q };
        cuLaunchKernel(r->fn_convert_f16_to_f32,
            (total_q + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args_deconv, NULL);
    }
}

/* ---- SWA FA2 launch helper ---- */
static void launch_fa2_attention_swa(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr q_batch,
                                      CUdeviceptr key_cache, CUdeviceptr value_cache,
                                      int n_tokens, int start_pos, int n_heads, int n_kv_heads,
                                      int head_dim, int kv_dim, int window_size, float scale) {
    /* SWA: restrict KV range to window */
    int first = (start_pos + n_tokens > window_size) ? (start_pos + n_tokens - window_size) : 0;
    int kv_tokens = start_pos + n_tokens - first;
    /* FA2 with offset: use key_cache + first * hstride_kv and V similarly */
    CUdeviceptr k_sub = key_cache + (size_t)first * kv_dim * 2; /* 2 = sizeof(F16) */
    CUdeviceptr v_sub = value_cache + (size_t)first * kv_dim * 2;
    /* Adjust scale for windowed context */
    float adj_scale = scale; /* FA2 handles causal mask internally */
    int dummy;
    (void)kv_tokens;
    launch_fa2_attention(r, out, q_batch, k_sub, v_sub,
                         n_tokens, start_pos, n_heads, n_kv_heads,
                         head_dim, kv_dim, adj_scale, 0);
}

/* ---- FA2 d=512 launch helper ---- */
/* Same shape as launch_fa2_attention but dispatches to fa2_attn_d512_f which
 * uses BC=16 (vs 32 for d=256). Inputs are F32 Q (converted to F16) and F16
 * KV cache. Output is F16 (converted back to F32). */
static void launch_fa2_attention_d512(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr q_batch,
                                      CUdeviceptr key_cache, CUdeviceptr value_cache,
                                      int n_tokens, int start_pos, int n_heads, int n_kv_heads,
                                      int head_dim, int kv_dim, float scale, int window) {
    if (!r->fn_fa2_attn_512 || head_dim != 512) return;
    int q_dim = n_heads * head_dim;
    int hstride_q = q_dim;
    int hstride_kv = kv_dim;
    int gqa_ratio = n_heads / n_kv_heads;

    /* Convert Q from F32 to F16 */
    int total_q = n_tokens * q_dim;
    CUdeviceptr d_q_f16 = r->d_fa2_q_f16;
    if (d_q_f16 && r->fn_convert_f32_to_f16) {
        void *args_conv[] = { &d_q_f16, &q_batch, &total_q };
        cuLaunchKernel(r->fn_convert_f32_to_f16,
            (total_q + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args_conv, NULL);
    }
    CUdeviceptr d_o_f16 = r->d_fa2_o_f16 ? r->d_fa2_o_f16 : d_q_f16;

    int total_s = start_pos + n_tokens;
    int fa2_n_q = n_tokens;
    int fa2_q_off = start_pos;
#define FA2_BR_D512 64
#define FA2_BC_D512 16
#define FA2_NTHR_D512 128
    int n_q_tiles = (n_tokens + FA2_BR_D512 - 1) / FA2_BR_D512;
    size_t fa2_smem = 4 * FA2_BC_D512 * (head_dim + 8) * 2; /* F16, +8 to avoid bank conflicts */
    void *fa2_args[] = { &d_o_f16, &d_q_f16, &key_cache, &value_cache,
                         &total_s, &head_dim, &hstride_q, &hstride_kv, &gqa_ratio, &scale, &window,
                         &fa2_n_q, &fa2_q_off };
    cuLaunchKernel(r->fn_fa2_attn_512,
        (unsigned)n_q_tiles, (unsigned)n_heads, 1,
        FA2_NTHR_D512, 1, 1, (unsigned)fa2_smem, r->stream, fa2_args, NULL);

    if (r->fn_convert_f16_to_f32) {
        void *args_deconv[] = { &out, &d_o_f16, &total_q };
        cuLaunchKernel(r->fn_convert_f16_to_f32,
            (total_q + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, args_deconv, NULL);
    }
}
#undef FA2_BR_D512
#undef FA2_BC_D512
#undef FA2_NTHR_D512

/* cuBLAS-d512 score scratch cap (elements): the [gqa*q_chunk x total_s] score
 * matrix is bounded to this many elements by chunking the query dim. Without
 * chunking, n_heads*N^2 is 4 GB+ at N=8192 and fails to allocate. Used as the
 * ceiling both here and in the FA2 buffer sizing in cuda_llm_prefill.
 * 16M elems = 64 MB F32 + 32 MB F16 score scratch (+128 MB Q). On a 16 GB card
 * at 8192 tokens this leaves cuBLAS ~1 GB free for its on-demand GEMM workspace;
 * a bigger cap (>=64M -> >=512 MB scratch) starves cuBLAS off the tensor-core
 * path and the prefill collapses (>200s vs 5.5s). Chunking finer than this costs
 * nothing measurable (8192 prefill: 16M=1483, 32M=1469 tok/s). The adaptive
 * clamp in cuda_llm_prefill lowers it further when free memory is tighter. */
#define D512_SCORE_CAP_ELEMS_DEFAULT ((size_t)16 * 1024 * 1024)
/* Env override (in elements) lets tests force chunking at small n_tokens to
 * validate the chunk-boundary math against the sequential path, or raise the
 * cap on cards with more spare memory. */
static size_t d512_score_cap_elems(void) {
    const char *e = getenv("CUDA_LLM_D512_CAP");
    if (e) { long v = atol(e); if (v > 0) return (size_t)v; }
    return D512_SCORE_CAP_ELEMS_DEFAULT;
}
#define D512_SCORE_CAP_ELEMS (d512_score_cap_elems())

/* ---- cuBLAS-based D=512 attention: per-KV-head GEMM + softmax + GEMM ---- */
/* Q_batch: F32 [n_tokens * q_dim]. K/V cache: F16 [max_seq_len * kv_dim].
   Uses cuBLAS F16 GEMM for Q@K^T and P@V with custom causal softmax.
   The query dim is processed in chunks so the materialized score matrix stays
   bounded (D512_SCORE_CAP_ELEMS) regardless of n_tokens.

   Falls through to launch_fa2_attention_d512 when the FA2 d=512 cubin
   function is available, since FA2 avoids materializing scores entirely. */
static void launch_cublas_d512_attention(cuda_llm_runner *r, CUdeviceptr out, CUdeviceptr q_batch,
                                          CUdeviceptr key_cache, CUdeviceptr value_cache,
                                          int n_tokens, int start_pos, int n_heads, int n_kv_heads,
                                          int head_dim, int kv_dim, float scale) {
    if (head_dim == 512 && r->fn_fa2_attn_512) {
        launch_fa2_attention_d512(r, out, q_batch, key_cache, value_cache,
                                   n_tokens, start_pos, n_heads, n_kv_heads,
                                   head_dim, kv_dim, scale, 0);
        return;
    }
    if (!r->cublas || !r->fn_fa2_attn_256 || !r->d_fa2_q_f16 || !r->d_fa2_scores_f16) return;
    int gqa = n_heads / n_kv_heads;
    int q_dim = n_heads * head_dim;
    int total_s = start_pos + n_tokens;

    /* Convert the full Q batch F32 -> F16 once */
    int n_q = n_tokens * q_dim;
    CUdeviceptr q_f16 = r->d_fa2_q_f16;
    {
        void *args[] = { &q_f16, &q_batch, &n_q };
        cuLaunchKernel(r->fn_convert_f32_to_f16, (n_q+255)/256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }

    /* Scores scratch: F32 [chunk_sq * total_s] (d_fa2_o_f16) + F16 copy
     * (d_fa2_scores_f16). Both sized for one query-chunk in cuda_llm_prefill. */
    CUdeviceptr scores = r->d_fa2_o_f16;
    CUdeviceptr scores_f16 = r->d_fa2_scores_f16;

    /* Chunk the query (token) dim so gqa*q_chunk*total_s <= cap. Computed with
     * n_heads (worst-case gqa) and r->d512_cap_elems — the SAME cap the buffers
     * were sized for in cuda_llm_prefill — so it matches the allocation exactly. */
    size_t cap = r->d512_cap_elems ? r->d512_cap_elems : D512_SCORE_CAP_ELEMS;
    int q_chunk = (int)(cap / ((size_t)n_heads * (size_t)total_s));
    if (q_chunk < 1) q_chunk = 1;
    if (q_chunk > n_tokens) q_chunk = n_tokens;

    /* Per KV head: chunked GEMM, softmax, GEMM */
    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        /* Q_group base: F16 [gqa*n_tokens, head_dim] for this kv-head */
        CUdeviceptr qg = q_f16 + (size_t)kv_h * gqa * head_dim * sizeof(uint16_t);
        /* K/V_head: F16 [total_s, head_dim] (interleaved stride = kv_dim) */
        CUdeviceptr k_head = key_cache + (size_t)kv_h * head_dim * sizeof(uint16_t);
        CUdeviceptr v_head = value_cache + (size_t)kv_h * head_dim * sizeof(uint16_t);
        CUdeviceptr out_group = out + (size_t)kv_h * gqa * head_dim * sizeof(float);

        for (int q0 = 0; q0 < n_tokens; q0 += q_chunk) {
            int qc = n_tokens - q0;
            if (qc > q_chunk) qc = q_chunk;
            int chunk_sq = qc * gqa;
            /* Rows for tokens [q0, q0+qc) are contiguous in qg (token is outer dim) */
            CUdeviceptr qg_c = qg + (size_t)q0 * gqa * head_dim * sizeof(uint16_t);

            /* Step 1: Q_chunk @ K^T -> scores [chunk_sq, total_s] F32 */
            cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, scores, k_head, qg_c,
                                                   chunk_sq, total_s, head_dim);

            /* Step 2: causal softmax. Pass start_pos+q0 so each chunk row's
             * kv_lim = (start_pos+q0) + local_tok + 1 = correct absolute limit. */
            {
                int sp = start_pos + q0;
                void *args[] = { &scores, &chunk_sq, &total_s, &sp, &gqa };
                cuLaunchKernel(r->fn_causal_softmax_f32, chunk_sq, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
            }

            /* Step 3: F32 scores -> F16 (cuBLAS needs F16 input) */
            {
                int n_scores = chunk_sq * total_s;
                void *args[] = { &scores_f16, &scores, &n_scores };
                cuLaunchKernel(r->fn_convert_f32_to_f16, (n_scores+255)/256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
            }

            /* Step 4: P(F16) @ V(F16)^T -> out [chunk_sq, head_dim] F32 */
            CUdeviceptr out_c = out_group + (size_t)q0 * gqa * head_dim * sizeof(float);
            cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, out_c, v_head, scores_f16,
                                                   chunk_sq, head_dim, total_s);
        }
    }
}

/* Ensure pre-allocated batch buffers are large enough for n_tokens */
static int cuda_llm_ensure_batch_buffers(cuda_llm_runner *r, int n_tokens) {
    if (n_tokens <= r->batch_buf_max_tokens) return 0;

    int n_embd = r->n_embd;
    int n_heads = r->n_heads;
    int n_kv_heads = r->n_kv_heads;
    int head_dim = r->head_dim;
    int n_ff = r->n_ff;
    int q_dim = n_heads * head_dim;
    int q2_dim = 2 * q_dim;
    int kv_dim = n_kv_heads * head_dim;
    int qkv_dim = r->ssm_qkv_dim;
    int d_inner = r->ssm_d_inner;
    int dt_rank = r->ssm_dt_rank;
    int wide_dim = q2_dim > qkv_dim ? q2_dim : qkv_dim;
    int mid_dim = q_dim > d_inner ? q_dim : d_inner;
    int ff_dim = n_ff;
    if (r->is_moe && r->shared_expert_ff > ff_dim) ff_dim = r->shared_expert_ff;

    /* Free old buffers */
    if (r->d_batch_x)     { cuMemFree(r->d_batch_x);     r->d_batch_x = 0; }
    if (r->d_batch_xb)    { cuMemFree(r->d_batch_xb);    r->d_batch_xb = 0; }
    if (r->d_batch_wide)  { cuMemFree(r->d_batch_wide);  r->d_batch_wide = 0; }
    if (r->d_batch_q)     { cuMemFree(r->d_batch_q);     r->d_batch_q = 0; }
    if (r->d_batch_mid)   { cuMemFree(r->d_batch_mid);   r->d_batch_mid = 0; }
    if (r->d_batch_k)     { cuMemFree(r->d_batch_k);     r->d_batch_k = 0; }
    if (r->d_batch_v)     { cuMemFree(r->d_batch_v);     r->d_batch_v = 0; }
    if (r->d_batch_ff1)   { cuMemFree(r->d_batch_ff1);   r->d_batch_ff1 = 0; }
    if (r->d_batch_ff2)   { cuMemFree(r->d_batch_ff2);   r->d_batch_ff2 = 0; }
    if (r->d_batch_alpha) { cuMemFree(r->d_batch_alpha);  r->d_batch_alpha = 0; }
    if (r->d_batch_beta)  { cuMemFree(r->d_batch_beta);   r->d_batch_beta = 0; }
    if (r->d_batch_token_ids) { cuMemFree(r->d_batch_token_ids); r->d_batch_token_ids = 0; }
    r->batch_buf_max_tokens = 0;

    /* Allocate with some headroom to avoid frequent reallocations */
    int alloc_tokens = n_tokens < 128 ? 128 : ((n_tokens + 63) & ~63);

    #define ALLOC_BATCH(field, elems) do { \
        size_t _sz = (size_t)(alloc_tokens) * (elems) * sizeof(float); \
        if (cuMemAlloc(&r->field, _sz) != CUDA_SUCCESS) { \
            fprintf(stderr, "cuda_llm: batch alloc failed for " #field " (%zu bytes)\n", _sz); \
            return -1; \
        } \
    } while(0)
    ALLOC_BATCH(d_batch_x, n_embd);
    ALLOC_BATCH(d_batch_xb, n_embd);
    ALLOC_BATCH(d_batch_wide, wide_dim);
    ALLOC_BATCH(d_batch_q, q_dim);
    ALLOC_BATCH(d_batch_mid, mid_dim);
    ALLOC_BATCH(d_batch_k, kv_dim);
    ALLOC_BATCH(d_batch_v, kv_dim);
    ALLOC_BATCH(d_batch_ff1, ff_dim);
    ALLOC_BATCH(d_batch_ff2, ff_dim);
    ALLOC_BATCH(d_batch_alpha, dt_rank);
    ALLOC_BATCH(d_batch_beta, dt_rank);
    #undef ALLOC_BATCH
    if (cuMemAlloc(&r->d_batch_token_ids, (size_t)alloc_tokens * sizeof(int32_t)) != CUDA_SUCCESS) { fprintf(stderr, "cuda_llm: batch alloc failed for d_batch_token_ids\n"); return -1; }

    /* Allocate F16 scratch for cuBLAS input conversion if needed */
    int max_in_dim = n_embd > ff_dim ? n_embd : ff_dim;
    if (max_in_dim < qkv_dim) max_in_dim = qkv_dim;
    if (max_in_dim < d_inner) max_in_dim = d_inner;
    if (r->d_batch_f16_scratch) { cuMemFree(r->d_batch_f16_scratch); r->d_batch_f16_scratch = 0; r->batch_f16_scratch_bytes = 0; }
    /* Lazily create cuBLAS handle on first batch buffer allocation */
    if (r->use_cublas && !r->cublas) {
        int cublas_ret = cublasewCreate(&r->cublas, r->stream);
        fprintf(stderr, "cuda_llm: cuBLAS handle creation: ret=%d handle=%p\n", cublas_ret, (void*)r->cublas);
        if (cublas_ret == 0 && r->verbose >= 1)
            fprintf(stderr, "cuda_llm: cuBLAS handle created\n");
    }
    if (r->use_cublas && r->cublas) {
        size_t bytes = (size_t)alloc_tokens * max_in_dim * sizeof(uint16_t);
        if (cuMemAlloc(&r->d_batch_f16_scratch, bytes) != CUDA_SUCCESS) {
            r->d_batch_f16_scratch = 0;
        } else {
            r->batch_f16_scratch_bytes = bytes;
        }
    }

    /* Allocate FA2 flash attention F16 buffers (Q and O conversion) */
    if (r->fn_fa2_attn_256) {
        int max_hd = q_dim > kv_dim ? q_dim : kv_dim;
        if (r->d_fa2_q_f16) { cuMemFree(r->d_fa2_q_f16); r->d_fa2_q_f16 = 0; }
        if (r->d_fa2_o_f16) { cuMemFree(r->d_fa2_o_f16); r->d_fa2_o_f16 = 0; }
        if (r->d_f16_scratch) { cuMemFree(r->d_f16_scratch); r->d_f16_scratch = 0; }
        if (cuMemAlloc(&r->d_fa2_q_f16, (size_t)alloc_tokens * max_hd * sizeof(uint16_t)) != CUDA_SUCCESS)
            r->d_fa2_q_f16 = 0;
        if (cuMemAlloc(&r->d_fa2_o_f16, (size_t)alloc_tokens * max_hd * sizeof(uint16_t)) != CUDA_SUCCESS)
            r->d_fa2_o_f16 = 0;
    }

    if (r->is_moe) {
        int router_entries = alloc_tokens * r->n_experts;
        if (router_entries < r->n_experts) router_entries = r->n_experts;
        if (r->d_router_logits_entries < router_entries) {
            CUdeviceptr new_logits = 0;
            if (cuMemAlloc(&new_logits, (size_t)router_entries * sizeof(float)) != CUDA_SUCCESS) {
                fprintf(stderr, "cuda_llm: batch alloc failed for d_router_logits\n");
                return -1;
            }
            if (r->d_router_logits) cuMemFree(r->d_router_logits);
            r->d_router_logits = new_logits;
            r->d_router_logits_entries = router_entries;
        }
    }

    r->batch_buf_max_tokens = alloc_tokens;
    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_llm: allocated batch buffers for %d tokens\n", alloc_tokens);
    }
    return 0;
}

static float *cuda_llm_prefill_qwen35(cuda_llm_runner *r, const int32_t *token_ids,
                                       const float *embeddings, int embd_stride,
                                       int n_tokens, int start_pos) {
    if (cuda_llm_bind_context(r) != 0) return NULL;

    /* MoE models: use batched prefill with per-layer MoE FFN */
    if (cuda_llm_ensure_batch_buffers(r, n_tokens) != 0) return NULL;

    for (int l = 0; l < r->n_layers; l++) {
        cuda_layer *cl = &r->layers[l];
        if (!cl->is_moe) continue;
        if (r->max_layers > 0 && l >= r->max_layers) break;

        {
            int can_cublas = r->use_cublas && r->cublas && r->d_batch_f16_scratch &&
                             cl->moe_gate_exps_type == GGML_TYPE_IQ2_XXS &&
                             cl->moe_up_exps_type == GGML_TYPE_IQ2_XXS &&
                             cl->moe_down_exps_type == GGML_TYPE_IQ2_XXS;
            int can_fused = cl->moe_gate_exps_type == GGML_TYPE_IQ2_XXS &&
                            cl->moe_up_exps_type == GGML_TYPE_IQ2_XXS &&
                            cl->moe_down_exps_type == GGML_TYPE_IQ2_XXS;

            if (r->moe_prefill_backend == CLLM_MOE_PREFILL_CUBLAS && !can_cublas) {
                fprintf(stderr, "cuda_llm: requested cublas MoE prefill backend is unavailable\n");
                return NULL;
            }
            if (r->moe_prefill_backend == CLLM_MOE_PREFILL_FUSED && !can_fused) {
                fprintf(stderr, "cuda_llm: requested fused MoE prefill backend requires IQ2_XXS gate/up/down\n");
                return NULL;
            }
            if (r->moe_prefill_backend == CLLM_MOE_PREFILL_AUTO && !can_cublas && !can_fused) {
                if (r->verbose >= 1) {
                    fprintf(stderr,
                            "cuda_llm: falling back to exact MoE prefill for layer %d (unsupported batched expert weights)\n",
                            l);
                }
                return cuda_llm_prefill_sequential(r, token_ids, embeddings, embd_stride,
                                                   n_tokens, start_pos);
            }
        }
    }

    int n_embd = r->n_embd;
    int n_heads = r->n_heads;
    int n_kv_heads = r->n_kv_heads;
    int head_dim = r->head_dim;
    int n_ff = r->n_ff;
    int q_dim = n_heads * head_dim;
    int q2_dim = 2 * q_dim;
    int kv_dim = n_kv_heads * head_dim;
    int qkv_dim = r->ssm_qkv_dim;
    int d_inner = r->ssm_d_inner;
    int dt_rank = r->ssm_dt_rank;
    int d_state = r->ssm_d_state;
    int n_group = r->ssm_n_group;
    int conv_k = r->ssm_conv_kernel;
    int n_layers = r->n_layers;
    float eps = r->rms_norm_eps;

    if (r->max_layers > 0 && r->max_layers < n_layers) n_layers = r->max_layers;

    size_t batch_embd_bytes = (size_t)n_tokens * n_embd * sizeof(float);

    CUdeviceptr d_batch_x = r->d_batch_x;
    CUdeviceptr d_batch_xb = r->d_batch_xb;
    CUdeviceptr d_batch_wide = r->d_batch_wide;
    CUdeviceptr d_batch_q = r->d_batch_q;
    CUdeviceptr d_batch_mid = r->d_batch_mid;
    CUdeviceptr d_batch_k = r->d_batch_k;
    CUdeviceptr d_batch_v = r->d_batch_v;
    CUdeviceptr d_batch_ff1 = r->d_batch_ff1;
    CUdeviceptr d_batch_ff2 = r->d_batch_ff2;
    CUdeviceptr d_batch_alpha = r->d_batch_alpha;
    CUdeviceptr d_batch_beta = r->d_batch_beta;
    float *result = NULL;

    if (token_ids) {
        if (r->token_embd_type == GGML_TYPE_F16) {
            if (cuMemcpyHtoD(r->d_batch_token_ids, token_ids, (size_t)n_tokens * sizeof(int32_t)) != CUDA_SUCCESS) return NULL;
            launch_batch_embed_f16(r, d_batch_x, r->d_token_embd, r->d_batch_token_ids, n_embd, n_tokens);
        } else {
            for (int t = 0; t < n_tokens; t++) {
                if (r->token_embd_type == GGML_TYPE_Q8_0) {
                    launch_embed_q8_0(r, r->d_x, r->d_token_embd, token_ids[t], n_embd);
                } else if (r->token_embd_type == GGML_TYPE_Q2_K) {
                    launch_embed_q2_K(r, r->d_x, r->d_token_embd, token_ids[t], n_embd);
                } else if (r->token_embd_type == GGML_TYPE_Q4_K) {
                    int32_t tok = token_ids[t];
                    void *ea[] = { &r->d_x, &r->d_token_embd, &tok, &n_embd };
                    cuLaunchKernel(r->fn_embed_q4_K, (n_embd + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, ea, NULL);
                } else if (r->token_embd_type == GGML_TYPE_Q4_0) {
                    launch_embed_q4_0(r, r->d_x, r->d_token_embd, token_ids[t], n_embd);
                } else {
                    launch_embed(r, r->d_x, r->d_token_embd, token_ids[t], n_embd);
                }
                CUdeviceptr dst_t = d_batch_x + (size_t)t * n_embd * sizeof(float);
                cuMemcpyDtoDAsync(dst_t, r->d_x, (size_t)n_embd * sizeof(float), r->stream);
            }
        }
    } else if (embeddings) {
        if (embd_stride == n_embd) {
            if (cuMemcpyHtoD(d_batch_x, embeddings, batch_embd_bytes) != CUDA_SUCCESS) return NULL;
        } else {
            for (int t = 0; t < n_tokens; t++) {
                CUdeviceptr dst_t = d_batch_x + (size_t)t * n_embd * sizeof(float);
                const float *src_t = embeddings + (size_t)t * embd_stride;
                cuMemcpyHtoDAsync(dst_t, src_t, (size_t)n_embd * sizeof(float), r->stream);
            }
        }
    } else {
        return NULL;
    }

    /* Profiling accumulators */
    double prof_gemm_ms = 0, prof_ssm_ms = 0, prof_attn_ms = 0, prof_ffn_ms = 0;
    double detail_ssm_param_ms = 0, detail_ssm_conv_ms = 0, detail_ssm_norm_ms = 0;
    double detail_ssm_scan_ms = 0, detail_ssm_gate_ms = 0;
    double detail_attn_prep_ms = 0, detail_attn_core_ms = 0, detail_attn_post_ms = 0;
    double detail_ffn_norm_ms = 0, detail_ffn_router_ms = 0, detail_ffn_topk_ms = 0;
    double detail_ffn_pack_ms = 0, detail_ffn_experts_ms = 0, detail_ffn_shared_ms = 0;
    double detail_exp_gather_ms = 0, detail_exp_dequant_ms = 0, detail_exp_gemm_ms = 0;
    double detail_exp_act_ms = 0, detail_exp_scatter_ms = 0;
    int do_prof = (r->verbose >= 1 && n_tokens > 1);
    const char *detail_env = getenv("CUDA_LLM_PREFILL_DETAIL");
    int do_detail = do_prof && detail_env && detail_env[0] && strcmp(detail_env, "0") != 0;
    const char *stop_attn_env = getenv("CUDA_LLM_STOP_AFTER_ATTN_RESIDUAL");
    int stop_after_attn_residual = stop_attn_env && stop_attn_env[0] && strcmp(stop_attn_env, "0") != 0;

    for (int l = 0; l < n_layers; l++) {
        cuda_layer *cl = &r->layers[l];

        cuMemcpyDtoDAsync(d_batch_xb, d_batch_x, batch_embd_bytes, r->stream);
        {
            void *a[] = { &d_batch_xb, &cl->attn_norm_w, &n_embd, &eps, &n_tokens };
            cuLaunchKernel(r->fn_batch_rmsnorm_f32, n_tokens, 1, 1, 256, 1, 1, 0, r->stream, a, NULL);
        }

        if (cl->is_ssm) {
            if (do_prof) cuStreamSynchronize(r->stream);
            double _t0 = do_prof ? get_time_ms() : 0;
            launch_batch_matvec(r, d_batch_wide, cl->ssm_qkv_w, cl->ssm_qkv_w_f16, d_batch_xb,
                                qkv_dim, n_embd, n_tokens, cl->ssm_qkv_type);
            launch_batch_matvec(r, d_batch_q, cl->ssm_gate_w, cl->ssm_gate_w_f16, d_batch_xb,
                                d_inner, n_embd, n_tokens, cl->ssm_gate_type);
            launch_batch_matvec(r, d_batch_alpha, cl->ssm_alpha_w, 0, d_batch_xb,
                                dt_rank, n_embd, n_tokens, cl->ssm_alpha_type);
            launch_batch_matvec(r, d_batch_beta, cl->ssm_beta_w, 0, d_batch_xb,
                                dt_rank, n_embd, n_tokens, cl->ssm_beta_type);
            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[10];
                cuMemcpyDtoH(dbg, d_batch_xb, 8 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] normed[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                cuMemcpyDtoH(dbg, d_batch_wide, 10 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] qkv[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7],dbg[8],dbg[9]);
                cuMemcpyDtoH(dbg, d_batch_q, 10 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] z[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7],dbg[8],dbg[9]);
                float alpha_dbg[48], beta_dbg[48];
                cuMemcpyDtoH(alpha_dbg, d_batch_alpha, dt_rank * sizeof(float));
                cuMemcpyDtoH(beta_dbg, d_batch_beta, dt_rank * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] alpha_raw[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        alpha_dbg[0],alpha_dbg[1],alpha_dbg[2],alpha_dbg[3],alpha_dbg[4],alpha_dbg[5],alpha_dbg[6],alpha_dbg[7],alpha_dbg[8],alpha_dbg[9]);
                fprintf(stderr, "  [BATCH L0 DBG] beta_raw[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        beta_dbg[0],beta_dbg[1],beta_dbg[2],beta_dbg[3],beta_dbg[4],beta_dbg[5],beta_dbg[6],beta_dbg[7],beta_dbg[8],beta_dbg[9]);
            }
            if (do_prof) { cuStreamSynchronize(r->stream); prof_gemm_ms += get_time_ms() - _t0; _t0 = get_time_ms(); }

            double _detail_t0 = do_detail ? get_time_ms() : 0;
            launch_batch_softplus_mul(r, d_batch_alpha, d_batch_alpha,
                                      cl->ssm_dt_bias, cl->ssm_a, dt_rank, n_tokens);
            launch_exp_inplace(r, d_batch_alpha, n_tokens * dt_rank);
            launch_sigmoid_inplace(r, d_batch_beta, n_tokens * dt_rank);
            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float alpha_dbg[10], beta_dbg[10];
                cuMemcpyDtoH(alpha_dbg, d_batch_alpha, 10 * sizeof(float));
                cuMemcpyDtoH(beta_dbg, d_batch_beta, 10 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] alpha_decay[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        alpha_dbg[0],alpha_dbg[1],alpha_dbg[2],alpha_dbg[3],alpha_dbg[4],alpha_dbg[5],alpha_dbg[6],alpha_dbg[7],alpha_dbg[8],alpha_dbg[9]);
                fprintf(stderr, "  [BATCH L0 DBG] beta_sigmoid[0:10]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        beta_dbg[0],beta_dbg[1],beta_dbg[2],beta_dbg[3],beta_dbg[4],beta_dbg[5],beta_dbg[6],beta_dbg[7],beta_dbg[8],beta_dbg[9]);
            }
            if (do_detail) { cuStreamSynchronize(r->stream); detail_ssm_param_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }
            launch_batch_conv1d(r, d_batch_wide, cl->d_conv_state, cl->ssm_conv1d_w,
                                qkv_dim, conv_k, n_tokens);
            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, d_batch_wide, 8 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] conv_Q[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                cuMemcpyDtoH(dbg, d_batch_wide + (size_t)n_group * d_state * sizeof(float), 8 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] conv_K[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
                cuMemcpyDtoH(dbg, d_batch_wide + (size_t)2 * n_group * d_state * sizeof(float), 8 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] conv_V[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }
            if (do_detail) { cuStreamSynchronize(r->stream); detail_ssm_conv_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }
            launch_batch_l2_norm_heads_strided(r, d_batch_wide,
                                                   n_group, n_tokens, d_state, qkv_dim, eps);
            launch_batch_l2_norm_heads_strided(r,
                                                   d_batch_wide + (size_t)n_group * d_state * sizeof(float),
                                                   n_group, n_tokens, d_state, qkv_dim, eps);
            if (do_detail) { cuStreamSynchronize(r->stream); detail_ssm_norm_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }

            launch_batch_deltanet_scan(r, d_batch_mid, cl->d_recurrent_state, d_batch_wide,
                                       d_batch_alpha, d_batch_beta, n_tokens, dt_rank, d_state, n_group);
            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, d_batch_mid, 8 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] deltanet_out[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }
            if (do_detail) { cuStreamSynchronize(r->stream); detail_ssm_scan_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }
            launch_gated_rmsnorm_silu(r, d_batch_mid, d_batch_q, cl->ssm_norm_w,
                                      n_tokens * dt_rank, d_state, eps);
            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, d_batch_mid, 8 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] gated_out[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }
            if (do_detail) { cuStreamSynchronize(r->stream); detail_ssm_gate_ms += get_time_ms() - _detail_t0; }
            if (do_prof) { cuStreamSynchronize(r->stream); prof_ssm_ms += get_time_ms() - _t0; _t0 = get_time_ms(); }
            launch_batch_matvec(r, d_batch_xb, cl->ssm_out_w, cl->ssm_out_w_f16, d_batch_mid,
                                n_embd, d_inner, n_tokens, cl->ssm_out_type);
            if (r->debug_layers >= 2 && l == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, d_batch_xb, 8 * sizeof(float));
                fprintf(stderr, "  [BATCH L0 DBG] ssm_out_proj[0:8]=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        dbg[0],dbg[1],dbg[2],dbg[3],dbg[4],dbg[5],dbg[6],dbg[7]);
            }
            if (do_prof) { cuStreamSynchronize(r->stream); prof_gemm_ms += get_time_ms() - _t0; }
        } else {
            if (do_prof) cuStreamSynchronize(r->stream);
            double _t0 = do_prof ? get_time_ms() : 0;
            launch_batch_matvec(r, d_batch_wide, cl->attn_q_w, cl->attn_q_w_f16, d_batch_xb,
                                q2_dim, n_embd, n_tokens, cl->attn_q_type);
            launch_batch_matvec(r, d_batch_k, cl->attn_k_w, cl->attn_k_w_f16, d_batch_xb,
                                kv_dim, n_embd, n_tokens, cl->attn_k_type);
            launch_batch_matvec(r, d_batch_v, cl->attn_v_w, cl->attn_v_w_f16, d_batch_xb,
                                kv_dim, n_embd, n_tokens, cl->attn_v_type);
            if (do_prof) { cuStreamSynchronize(r->stream); prof_gemm_ms += get_time_ms() - _t0; _t0 = get_time_ms(); }

            double _detail_t0 = do_detail ? get_time_ms() : 0;
            launch_deinterleave_qgate(r, d_batch_q, d_batch_mid, d_batch_wide, n_tokens * n_heads, head_dim);

            if (cl->has_qk_norm) {
                if (cl->attn_q_norm_w) launch_qknorm(r, d_batch_q, cl->attn_q_norm_w, n_tokens * n_heads, head_dim, eps);
                if (cl->attn_k_norm_w) launch_qknorm(r, d_batch_k, cl->attn_k_norm_w, n_tokens * n_kv_heads, head_dim, eps);
            }

            launch_batch_rope(r, d_batch_q, n_heads, n_tokens, head_dim, start_pos, r->rope_freq_base, r->n_rope_pairs);
            launch_batch_rope(r, d_batch_k, n_kv_heads, n_tokens, head_dim, start_pos, r->rope_freq_base, r->n_rope_pairs);

            {
                void *args[] = { &r->d_key_cache[l], &r->d_value_cache[l], &d_batch_k, &d_batch_v,
                                 &start_pos, &kv_dim, &n_tokens };
                cuLaunchKernel(r->fn_batch_kv_store_f16, (kv_dim + 255) / 256, n_tokens, 1,
                               256, 1, 1, 0, r->stream, args, NULL);
            }
            if (do_detail) { cuStreamSynchronize(r->stream); detail_attn_prep_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }

            {
                float scale = 1.0f / sqrtf((float)head_dim);
                launch_batch_attention(r, d_batch_wide, d_batch_q,
                                       r->d_key_cache[l], r->d_value_cache[l],
                                       n_tokens, start_pos, n_heads, n_kv_heads, head_dim, kv_dim, scale);
            }
            if (do_detail) { cuStreamSynchronize(r->stream); detail_attn_core_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }

            launch_sigmoid_mul(r, d_batch_wide, d_batch_mid, n_tokens * q_dim);
            launch_batch_matvec(r, d_batch_xb, cl->attn_output_w, cl->attn_output_w_f16, d_batch_wide,
                                n_embd, q_dim, n_tokens, cl->attn_output_type);
            if (do_detail) { cuStreamSynchronize(r->stream); detail_attn_post_ms += get_time_ms() - _detail_t0; }
            if (do_prof) { cuStreamSynchronize(r->stream); prof_attn_ms += get_time_ms() - _t0; }
        }

        launch_add(r, d_batch_x, d_batch_xb, n_tokens * n_embd);
        if (stop_after_attn_residual) break;

        if (do_prof) cuStreamSynchronize(r->stream);
        double _ft0 = do_prof ? get_time_ms() : 0;
        cuMemcpyDtoDAsync(d_batch_xb, d_batch_x, batch_embd_bytes, r->stream);
        {
            void *a[] = { &d_batch_xb, &cl->ffn_norm_w, &n_embd, &eps, &n_tokens };
            cuLaunchKernel(r->fn_batch_rmsnorm_f32, n_tokens, 1, 1, 256, 1, 1, 0, r->stream, a, NULL);
        }
        if (do_detail) { cuStreamSynchronize(r->stream); detail_ffn_norm_ms += get_time_ms() - _ft0; }

        if (cl->is_moe) {
            double _detail_t0 = do_detail ? get_time_ms() : 0;
            /* Router: logits[n_tokens, n_experts] = xb @ gate_inp^T */
            launch_batch_matvec(r, r->d_router_logits, cl->moe_gate_w, 0, d_batch_xb,
                                r->n_experts, n_embd, n_tokens, GGML_TYPE_F32);
            if (do_detail) { cuStreamSynchronize(r->stream); detail_ffn_router_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }

            /* MoE FFN: shared GPU top-k, then selected prefill backend. */
            {
                int can_cublas = r->use_cublas && r->cublas && r->d_batch_f16_scratch &&
                                 cl->moe_gate_exps_type == GGML_TYPE_IQ2_XXS &&
                                 cl->moe_up_exps_type == GGML_TYPE_IQ2_XXS &&
                                 cl->moe_down_exps_type == GGML_TYPE_IQ2_XXS;
                int can_fused = cl->moe_gate_exps_type == GGML_TYPE_IQ2_XXS &&
                                cl->moe_up_exps_type == GGML_TYPE_IQ2_XXS &&
                                cl->moe_down_exps_type == GGML_TYPE_IQ2_XXS;
                int backend = r->moe_prefill_backend;
                if (backend == CLLM_MOE_PREFILL_AUTO)
                    backend = can_cublas ? CLLM_MOE_PREFILL_CUBLAS : CLLM_MOE_PREFILL_FUSED;
                if (backend == CLLM_MOE_PREFILL_CUBLAS && !can_cublas) {
                    fprintf(stderr, "cuda_llm: requested cublas MoE prefill backend is unavailable\n");
                    return NULL;
                }
                if (backend == CLLM_MOE_PREFILL_FUSED && !can_fused) {
                    fprintf(stderr, "cuda_llm: requested fused MoE prefill backend requires IQ2_XXS gate/up/down\n");
                    return NULL;
                }
                if (launch_moe_topk_prefill(r, n_tokens) != 0) return NULL;
                if (do_detail) { cuStreamSynchronize(r->stream); detail_ffn_topk_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }

                if (backend == CLLM_MOE_PREFILL_CUBLAS) {
                    int tke = n_tokens * r->n_experts_used;
                    int *h_tki = (int *)malloc((size_t)tke * sizeof(int));
                    float *h_tkw = (float *)malloc((size_t)tke * sizeof(float));
                    int *ec = (int *)calloc((size_t)r->n_experts, sizeof(int));
                    int *offs = (int *)malloc((size_t)(r->n_experts + 1) * sizeof(int));
                    int *cursor = (int *)malloc((size_t)r->n_experts * sizeof(int));
                    int *etok = (int *)malloc((size_t)tke * sizeof(int));
                    float *etw = (float *)malloc((size_t)tke * sizeof(float));
                    if (!h_tki || !h_tkw || !ec || !offs || !cursor || !etok || !etw) {
                        free(h_tki); free(h_tkw); free(ec); free(offs); free(cursor); free(etok); free(etw);
                        return NULL;
                    }
                    cuStreamSynchronize(r->stream);
                    cuMemcpyDtoH(h_tki, r->d_topk_idx, tke * sizeof(int));
                    cuMemcpyDtoH(h_tkw, r->d_topk_wgt, tke * sizeof(float));
                    for (int t = 0; t < n_tokens; t++) {
                        for (int ei = 0; ei < r->n_experts_used; ei++) {
                            int eidx = h_tki[t * r->n_experts_used + ei];
                            if (eidx >= 0 && eidx < r->n_experts) ec[eidx]++;
                        }
                    }
                    offs[0] = 0;
                    for (int e = 0; e < r->n_experts; e++) {
                        offs[e + 1] = offs[e] + ec[e];
                        cursor[e] = offs[e];
                    }
                    for (int t = 0; t < n_tokens; t++) {
                        for (int ei = 0; ei < r->n_experts_used; ei++) {
                            int src = t * r->n_experts_used + ei;
                            int eidx = h_tki[src];
                            if (eidx < 0 || eidx >= r->n_experts) continue;
                            int dst = cursor[eidx]++;
                            etok[dst] = t;
                            etw[dst] = h_tkw[src];
                        }
                    }
                    cuMemcpyHtoD(r->d_topk_idx, etok, (size_t)tke * sizeof(int));
                    cuMemcpyHtoD(r->d_topk_wgt, etw, (size_t)tke * sizeof(float));
                    if (do_detail) { detail_ffn_pack_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }
                    const char *mmq_env = getenv("CUDA_LLM_MOE_MMQ");
                    int use_mmq = mmq_env && mmq_env[0] && strcmp(mmq_env, "0") != 0 &&
                        cl->moe_gate_exps_type == GGML_TYPE_IQ2_XXS &&
                        cl->moe_up_exps_type == GGML_TYPE_IQ2_XXS &&
                        cl->moe_down_exps_type == GGML_TYPE_IQ2_XXS &&
                        (n_embd % 64 == 0) && (r->expert_ff % 64 == 0) &&
                        r->fn_mmq_iq2xxs_grouped != NULL;
                    if (use_mmq) {
                        /* ===== IQ2_XXS MMQ (mul_mat_id) path: one grouped dispatch over all experts ===== */
                        int total = offs[r->n_experts];           /* compacted (token,expert) rows */
                        int ef = r->expert_ff;
                        int maxcol = n_embd > ef ? n_embd : ef;
                        /* Prefer larger token group size for better amortization. */
                        int use32 = (r->fn_mmq_iq2xxs_grouped32 != 0) && !getenv("CUDA_LLM_MMQ_TG16") && !getenv("CUDA_LLM_MMQ_TG4");
                        int use16 = !use32 && (r->fn_mmq_iq2xxs_grouped8 != 0) && !getenv("CUDA_LLM_MMQ_TG4");
                        int grp = use32 ? 256 : (use16 ? 128 : 32);
                        if (r->d_mmq_alloc_rows < total) {
                            if (r->d_mmq_cxq8) { cuMemFree(r->d_mmq_cxq8); cuMemFree(r->d_mmq_cxs);
                                cuMemFree(r->d_mmq_outg); cuMemFree(r->d_mmq_outu); cuMemFree(r->d_mmq_outd); }
                            if (cuMemAlloc(&r->d_mmq_cxq8, (size_t)total * maxcol) != CUDA_SUCCESS ||
                                cuMemAlloc(&r->d_mmq_cxs, (size_t)total * (maxcol / 32) * sizeof(float)) != CUDA_SUCCESS ||
                                cuMemAlloc(&r->d_mmq_outg, (size_t)total * ef * sizeof(float)) != CUDA_SUCCESS ||
                                cuMemAlloc(&r->d_mmq_outu, (size_t)total * ef * sizeof(float)) != CUDA_SUCCESS ||
                                cuMemAlloc(&r->d_mmq_outd, (size_t)total * n_embd * sizeof(float)) != CUDA_SUCCESS) {
                                fprintf(stderr, "cuda_llm: MMQ buffer alloc failed\n");
                                free(h_tki); free(h_tkw); free(ec); free(offs); free(cursor); free(etok); free(etw);
                                return NULL;
                            }
                            if (!r->d_mmq_ebounds) cuMemAlloc(&r->d_mmq_ebounds, (size_t)(r->n_experts + 1) * sizeof(int));
                            if (r->d_mmq_worklist) { cuMemFree(r->d_mmq_worklist); r->d_mmq_worklist = 0; }
                            cuMemAlloc(&r->d_mmq_worklist, (size_t)(r->n_experts + total / grp + 8) * sizeof(int));
                            r->d_mmq_alloc_rows = total;
                        }
                        cuMemcpyHtoD(r->d_mmq_ebounds, offs, (size_t)(r->n_experts + 1) * sizeof(int));
                        int maxtok = 0; for (int e = 0; e < r->n_experts; e++) if (ec[e] > maxtok) maxtok = ec[e];
                        /* flattened work-list: one entry per real (expert,group) pair */
                        int n_work = 0;
                        { int *wl = (int *)alloca((size_t)(r->n_experts + total / grp + 8) * sizeof(int));
                          for (int e = 0; e < r->n_experts; e++) { int ng = (ec[e] + grp - 1) / grp;
                              for (int g = 0; g < ng; g++) wl[n_work++] = (e << 16) | g; }
                          if (n_work == 0) wl[n_work++] = 0;
                          cuMemcpyHtoD(r->d_mmq_worklist, wl, (size_t)n_work * sizeof(int)); }
                        if (getenv("CUDA_LLM_MOE_DUMP_DIST")) {
                            static int dumped = 0;
                            if (!dumped) { dumped = 1;
                                int nz=0,h32=0,h64=0; long sum=0;
                                int hist[9]={0,0,0,0,0,0,0,0,0};
                                for (int e=0;e<r->n_experts;e++){ int c=ec[e]; sum+=c; if(c>0)nz++; if(c>32)h32++; if(c>64)h64++;
                                    int b=c==0?0:(c<=8?1:c<=16?2:c<=24?3:c<=32?4:c<=48?5:c<=64?6:c<=96?7:8); hist[b]++; }
                                fprintf(stderr,"[moe-dist] total=%d experts=%d maxtok=%d mean=%.1f nonzero=%d >32grp=%d >64=%d\n",
                                        (int)sum, r->n_experts, maxtok, (double)sum/r->n_experts, nz, h32, h64);
                                fprintf(stderr,"[moe-dist] hist =0:%d 1-8:%d 9-16:%d 17-24:%d 25-32:%d 33-48:%d 49-64:%d 65-96:%d 97+:%d\n",
                                        hist[0],hist[1],hist[2],hist[3],hist[4],hist[5],hist[6],hist[7],hist[8]);
                            }
                        }
                        /* 1. gather d_batch_xb rows by ids_token (=etok in d_topk_idx) + quant to q8_1 */
                        { int tr = total, kk = n_embd;
                          void *a[] = { &d_batch_xb, &r->d_topk_idx, &r->d_mmq_cxq8, &r->d_mmq_cxs, &tr, &kk };
                          cuLaunchKernel(r->fn_mmq_gather_quant_q8_1, n_embd / 256, total, 1, 32, 1, 1, 0, r->stream, a, NULL); }
                        /* 2. grouped gate + up: out[total][ef] */
                        int bm = r->moe_iq2_bm;
                        CUfunction mmqfn = use32 ? r->fn_mmq_iq2xxs_grouped32 : (use16 ? r->fn_mmq_iq2xxs_grouped8 : r->fn_mmq_iq2xxs_grouped);
                        { unsigned long long st = cl->moe_exp_stride_gu; int nN = ef, nK = n_embd;
                          void *ag[] = { &r->d_mmq_outg, &cl->moe_gate_exps_w, &st, &r->d_mmq_cxq8, &r->d_mmq_cxs, &r->d_mmq_ebounds, &r->d_mmq_worklist, &bm, &nN, &nK };
                          cuLaunchKernel(mmqfn, ef / 64, n_work, 1, 128, 1, 1, 0, r->stream, ag, NULL);
                          void *au[] = { &r->d_mmq_outu, &cl->moe_up_exps_w, &st, &r->d_mmq_cxq8, &r->d_mmq_cxs, &r->d_mmq_ebounds, &r->d_mmq_worklist, &bm, &nN, &nK };
                          cuLaunchKernel(mmqfn, ef / 64, n_work, 1, 128, 1, 1, 0, r->stream, au, NULL); }
                        /* 3. silu: outg = silu(outg) * outu */
                        launch_silu_mul(r, r->d_mmq_outg, r->d_mmq_outu, total * ef);
                        /* 4. quantize silu result (ef cols) into cxq8 for the down matmul */
                        { int tr = total, kk = ef;
                          void *a[] = { &r->d_mmq_outg, &r->d_mmq_cxq8, &r->d_mmq_cxs, &tr, &kk };
                          cuLaunchKernel(r->fn_mmq_quant_q8_1, ef / 256, total, 1, 32, 1, 1, 0, r->stream, a, NULL); }
                        /* 5. grouped down: out[total][n_embd] */
                        { unsigned long long st = cl->moe_exp_stride_d; int nN = n_embd, nK = ef;
                          void *ad[] = { &r->d_mmq_outd, &cl->moe_down_exps_w, &st, &r->d_mmq_cxq8, &r->d_mmq_cxs, &r->d_mmq_ebounds, &r->d_mmq_worklist, &bm, &nN, &nK };
                          cuLaunchKernel(mmqfn, n_embd / 64, n_work, 1, 128, 1, 1, 0, r->stream, ad, NULL); }
                        /* 6. weighted scatter into d_batch_x (ids_token=etok, weights=etw) */
                        { int tr = total, nN = n_embd;
                          void *a[] = { &d_batch_x, &r->d_mmq_outd, &r->d_topk_idx, &r->d_topk_wgt, &tr, &nN };
                          cuLaunchKernel(r->fn_mmq_scatter_weighted, total, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }
                    } else {
                    CUdeviceptr d_gd = d_batch_mid;
                    CUdeviceptr d_eg = d_batch_ff1, d_eu = d_batch_ff2, d_ed = d_batch_wide;
                    CUdeviceptr d_f16w = r->d_moe_f16w;
                    CUdeviceptr d_f16w2 = r->d_moe_f16w2;
                    CUdeviceptr d_f16w3 = r->d_moe_f16w3;
                    if (!d_f16w || !d_f16w2 || !d_f16w3) {
                        free(h_tki); free(h_tkw); free(ec); free(offs); free(cursor); free(etok); free(etw);
                        return NULL;
                    }
                    int direct_batch_threshold = CLLM_MOE_DIRECT_BATCH_THRESHOLD_DEFAULT;
                    {
                        const char *env = getenv("CUDA_LLM_MOE_DIRECT_BATCH_THRESHOLD");
                        if (env && env[0]) direct_batch_threshold = atoi(env);
                    }
                    for (int e = 0; e < r->n_experts; e++) {
                        int n_e = ec[e]; if (n_e <= 0) continue;
                        CUdeviceptr d_idx_e = r->d_topk_idx + (size_t)offs[e] * sizeof(int);
                        CUdeviceptr d_wgt_e = r->d_topk_wgt + (size_t)offs[e] * sizeof(float);
                        double _exp_t0 = do_detail ? get_time_ms() : 0;
                        { int _nr=n_e,_nc=n_embd;
                        void *ga[] = { &d_gd, &d_batch_xb, &d_idx_e, &_nr, &_nc };
                        cuLaunchKernel(r->fn_gather_rows_f32, n_e, 1, 1, 256, 1, 1, 0, r->stream, ga, NULL); }
                        CUdeviceptr src_gate = cl->moe_gate_exps_w + (size_t)e * cl->moe_exp_stride_gu;
                        CUdeviceptr src_up = cl->moe_up_exps_w + (size_t)e * cl->moe_exp_stride_gu;
                        CUdeviceptr src_down = cl->moe_down_exps_w + (size_t)e * cl->moe_exp_stride_d;
                        int use_direct_batch = direct_batch_threshold > 0 &&
                                               cl->moe_gate_exps_type == GGML_TYPE_IQ2_XXS &&
                                               cl->moe_up_exps_type == GGML_TYPE_IQ2_XXS &&
                                               cl->moe_down_exps_type == GGML_TYPE_IQ2_XXS &&
                                               n_e <= direct_batch_threshold;
                        if (use_direct_batch) {
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_gather_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                            launch_batch_matvec(r, d_eg, src_gate, 0, d_gd,
                                                r->expert_ff, n_embd, n_e, cl->moe_gate_exps_type);
                            launch_batch_matvec(r, d_eu, src_up, 0, d_gd,
                                                r->expert_ff, n_embd, n_e, cl->moe_up_exps_type);
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_gemm_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                        } else {
                            /* Convert F32 gathered -> F16 for cuBLAS input. */
                            { int _ne = n_e * n_embd;
                            void *cv[] = { &r->d_batch_f16_scratch, &d_gd, &_ne };
                            cuLaunchKernel(r->fn_convert_f32_to_f16, (_ne+255)/256, 1, 1, 256, 1, 1, 0, r->stream, cv, NULL); }
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_gather_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                            { int _ne = n_embd, _ef = r->expert_ff;
                            int _dq_units = r->expert_ff * (n_embd / 256) + n_embd * (r->expert_ff / 256);
                            void *dqt[] = { &d_f16w, &d_f16w2, &d_f16w3, &src_gate, &src_up, &src_down, &_ne, &_ef };
                            cuLaunchKernel(r->fn_dequant_iq2_xxs_triplet_to_f16,
                                           (_dq_units + 3) / 4, 1, 1, 128, 1, 1, 0, r->stream, dqt, NULL); }
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_dequant_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                            /* cuBLAS gate/up GEMMs */
                            cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, d_eg, d_f16w, r->d_batch_f16_scratch, n_e, r->expert_ff, n_embd);
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_gemm_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                            cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, d_eu, d_f16w2, r->d_batch_f16_scratch, n_e, r->expert_ff, n_embd);
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_gemm_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                        }
                        launch_silu_mul(r, d_eg, d_eu, n_e * r->expert_ff);
                        if (use_direct_batch) {
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_act_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                            launch_batch_matvec(r, d_ed, src_down, 0, d_eg,
                                                n_embd, r->expert_ff, n_e, cl->moe_down_exps_type);
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_gemm_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                        } else {
                            /* Convert F32 SiLU -> F16 for down input */
                            { int _ne = n_e * r->expert_ff;
                            void *cv2[] = { &r->d_batch_f16_scratch, &d_eg, &_ne };
                            cuLaunchKernel(r->fn_convert_f32_to_f16, (_ne+255)/256, 1, 1, 256, 1, 1, 0, r->stream, cv2, NULL); }
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_act_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                            cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, d_ed, d_f16w3, r->d_batch_f16_scratch, n_e, n_embd, r->expert_ff);
                            if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_gemm_ms += get_time_ms() - _exp_t0; _exp_t0 = get_time_ms(); }
                        }
                        /* Scatter-add weighted */
                        { int _sr=n_e,_sc=n_embd;
                        void *sa[] = { &d_batch_x, &d_ed, &d_idx_e, &d_wgt_e, &_sr, &_sc };
                        cuLaunchKernel(r->fn_scatter_add_weighted_f32, n_e, 1, 1, 256, 1, 1, 0, r->stream, sa, NULL); }
                        if (do_detail) { cuStreamSynchronize(r->stream); detail_exp_scatter_ms += get_time_ms() - _exp_t0; }
                    }
                    }  /* end else (per-expert cuBLAS loop) */
                    if (do_detail) { cuStreamSynchronize(r->stream); detail_ffn_experts_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }
                    free(h_tki); free(h_tkw); free(ec); free(offs); free(cursor); free(etok); free(etw);
                } else {
                    CUdeviceptr d_out = d_batch_x, d_in = d_batch_xb;
                    CUdeviceptr d_ge = cl->moe_gate_exps_w, d_ue = cl->moe_up_exps_w, d_de = cl->moe_down_exps_w;
                    int i6 = n_tokens, i8 = r->n_experts_used;
                    int i9 = n_embd, i10 = r->expert_ff;
                    size_t sz11 = cl->moe_exp_stride_gu, sz12 = cl->moe_exp_stride_d;
                    int i13 = cl->moe_gate_exps_type, i14 = cl->moe_up_exps_type, i15 = cl->moe_down_exps_type;
                    void *kargs[] = { &d_out,&d_in,&d_ge,&d_ue,&d_de,&r->d_topk_idx,&r->d_topk_wgt,
                        &i6,&i8,&i9,&i10,&sz11,&sz12,&i13,&i14,&i15 };
                    cuLaunchKernel(r->fn_moe_fused_ffn, n_tokens,1,1, 256,1,1, 0, r->stream, kargs, NULL);
                    if (do_detail) { cuStreamSynchronize(r->stream); detail_ffn_experts_ms += get_time_ms() - _detail_t0; _detail_t0 = get_time_ms(); }
                }
                /* === Shared expert === */
                {
                    int sff = r->shared_expert_ff;
                    launch_batch_matvec(r, r->d_router_logits, cl->moe_shared_gate_w, 0, d_batch_xb,
                                        1, n_embd, n_tokens, GGML_TYPE_F32);
                    launch_batch_matvec(r, d_batch_ff1, cl->moe_shared_ffn_gate_w,
                                        cl->moe_shared_ffn_gate_w_f16, d_batch_xb,
                                        cl->moe_shared_gate_rows, cl->moe_shared_gate_cols,
                                        n_tokens, cl->moe_shared_gate_type);
                    launch_batch_matvec(r, d_batch_ff2, cl->moe_shared_ffn_up_w,
                                        cl->moe_shared_ffn_up_w_f16, d_batch_xb,
                                        cl->moe_shared_up_rows, cl->moe_shared_up_cols,
                                        n_tokens, cl->moe_shared_up_type);
                    launch_silu_mul(r, d_batch_ff1, d_batch_ff2, n_tokens * sff);
                    launch_batch_matvec(r, d_batch_xb, cl->moe_shared_ffn_down_w,
                                        cl->moe_shared_ffn_down_w_f16, d_batch_ff1,
                                        cl->moe_shared_down_rows, cl->moe_shared_down_cols,
                                        n_tokens, cl->moe_shared_down_type);
                    launch_batch_scale_add_sigmoid(r, d_batch_x, d_batch_xb, r->d_router_logits,
                                                   n_tokens, n_embd);
                }
                if (do_detail) { cuStreamSynchronize(r->stream); detail_ffn_shared_ms += get_time_ms() - _detail_t0; }
            }
            if (do_prof) { cuStreamSynchronize(r->stream); prof_ffn_ms += get_time_ms() - _ft0; }
        } else {
            launch_batch_matvec(r, d_batch_ff1, cl->ffn_gate_w, cl->ffn_gate_w_f16, d_batch_xb,
                                n_ff, n_embd, n_tokens, cl->ffn_gate_type);
            launch_batch_matvec(r, d_batch_ff2, cl->ffn_up_w, cl->ffn_up_w_f16, d_batch_xb,
                                n_ff, n_embd, n_tokens, cl->ffn_up_type);
            launch_silu_mul(r, d_batch_ff1, d_batch_ff2, n_tokens * n_ff);
            launch_batch_matvec(r, d_batch_xb, cl->ffn_down_w, cl->ffn_down_w_f16, d_batch_ff1,
                                n_embd, n_ff, n_tokens, cl->ffn_down_type);
            launch_add(r, d_batch_x, d_batch_xb, n_tokens * n_embd);
            if (do_prof) { cuStreamSynchronize(r->stream); prof_ffn_ms += get_time_ms() - _ft0; }
        }

        if (embeddings && l < r->n_deepstack && embd_stride > n_embd) {
            const float *base = embeddings + (size_t)(1 + l) * n_embd;
            for (int t = 0; t < n_tokens; t++) {
                CUdeviceptr dst_t = d_batch_xb + (size_t)t * n_embd * sizeof(float);
                const float *src_t = base + (size_t)t * embd_stride;
                cuMemcpyHtoDAsync(dst_t, src_t, (size_t)n_embd * sizeof(float), r->stream);
            }
            launch_add(r, d_batch_x, d_batch_xb, n_tokens * n_embd);
        }
    }

    if (do_prof) {
        fprintf(stderr, "cuda_llm: prefill profile (%d tok): gemm=%.1fms ssm=%.1fms attn=%.1fms ffn=%.1fms total=%.1fms\n",
                n_tokens, prof_gemm_ms, prof_ssm_ms, prof_attn_ms, prof_ffn_ms,
                prof_gemm_ms + prof_ssm_ms + prof_attn_ms + prof_ffn_ms);
        if (do_detail) {
            fprintf(stderr,
                    "cuda_llm: prefill detail (%d tok): "
                    "ssm[param=%.1f conv=%.1f norm=%.1f scan=%.1f gate=%.1f] "
                    "attn[prep=%.1f core=%.1f post=%.1f] "
                    "ffn[norm=%.1f router=%.1f topk=%.1f pack=%.1f experts=%.1f shared=%.1f]\n",
                    n_tokens,
                    detail_ssm_param_ms, detail_ssm_conv_ms, detail_ssm_norm_ms,
                    detail_ssm_scan_ms, detail_ssm_gate_ms,
                    detail_attn_prep_ms, detail_attn_core_ms, detail_attn_post_ms,
                    detail_ffn_norm_ms, detail_ffn_router_ms, detail_ffn_topk_ms,
                    detail_ffn_pack_ms, detail_ffn_experts_ms, detail_ffn_shared_ms);
            fprintf(stderr,
                    "cuda_llm: expert detail (%d tok): gather=%.1f dequant=%.1f gemm=%.1f act=%.1f scatter=%.1f\n",
                    n_tokens, detail_exp_gather_ms, detail_exp_dequant_ms,
                    detail_exp_gemm_ms, detail_exp_act_ms, detail_exp_scatter_ms);
        }
    }

    {
        CUdeviceptr d_last = d_batch_x + (size_t)(n_tokens - 1) * n_embd * sizeof(float);
        cuMemcpyDtoDAsync(r->d_x, d_last, (size_t)n_embd * sizeof(float), r->stream);
        launch_rmsnorm(r, r->d_x, r->d_x, r->d_output_norm, n_embd, eps);
        cuMemcpyDtoHAsync(r->h_output, r->d_x, (size_t)n_embd * sizeof(float), r->stream);
        if (cuStreamSynchronize(r->stream) != CUDA_SUCCESS) {
            fprintf(stderr, "cuda_llm: prefill stream sync failed\n");
            return NULL;
        }
        result = r->h_output;
    }

    return result;
}

static float *cuda_llm_prefill_sequential(cuda_llm_runner *r, const int32_t *token_ids,
                                           const float *embeddings, int embd_stride,
                                           int n_tokens, int start_pos) {
    float *result = NULL;
    for (int t = 0; t < n_tokens; t++) {
        if (token_ids) {
            result = cuda_llm_forward(r, token_ids[t], start_pos + t);
        } else if (embeddings) {
            result = cuda_llm_forward_embd(r, embeddings + (size_t)t * embd_stride,
                                           embd_stride, start_pos + t);
        } else {
            return NULL;
        }
        if (!result) return NULL;
    }
    return result;
}

float *cuda_llm_prefill(cuda_llm_runner *r, const int32_t *token_ids,
                         const float *embeddings, int embd_stride,
                         int n_tokens, int start_pos) {
    if (!r || n_tokens <= 0) return NULL;
    if (cuda_llm_bind_context(r) != 0) return NULL;

    if (r->is_hybrid && !r->is_gemma4) {
        int exact_max = CLLM_PREFILL_EXACT_MAX_TOKENS_DEFAULT;
        const char *exact_env = getenv("CUDA_LLM_PREFILL_EXACT_MAX_TOKENS");
        if (exact_env) exact_max = atoi(exact_env);
        if (r->moe_prefill_backend == CLLM_MOE_PREFILL_EXACT ||
            (r->moe_prefill_backend == CLLM_MOE_PREFILL_AUTO &&
             exact_max > 0 && n_tokens <= exact_max)) {
            return cuda_llm_prefill_sequential(r, token_ids, embeddings, embd_stride,
                                               n_tokens, start_pos);
        }
        return cuda_llm_prefill_qwen35(r, token_ids, embeddings, embd_stride, n_tokens, start_pos);
    }

    /*
     * Generic batched prefill. For Gemma4 (LLM-only, text inputs) this is much
     * faster than running N sequential forwards. Set CUDA_LLM_BATCHED_PREFILL=0
     * to fall back to the single-token path.
     *
     * The VLM path uses vision embeddings interleaved with text tokens; the
     * batched path's numerical divergence on Gemma4 was the original reason
     * for always using sequential. We restrict the batched route to token-ids
     * input (LLM-only); vision-embedding prefill stays on sequential.
     */
    if (getenv("CUDA_LLM_BATCHED_PREFILL") && atoi(getenv("CUDA_LLM_BATCHED_PREFILL")) == 0) {
        return cuda_llm_prefill_sequential(r, token_ids, embeddings, embd_stride,
                                           n_tokens, start_pos);
    }
    if (embeddings != NULL) {
        /* VLM vision-embedding path. The batched body fully supports pre-computed
         * embeddings (uploads to d_batch_x, no embd-scale — see below), and the
         * "batched diverges on Gemma4" worry that originally forced sequential here
         * was a measurement artifact (it compared F16-batched vs the dp4a INT8
         * sequential path; vs an F32 oracle the batched path is ~3e-3, correct —
         * see cuda_llm_set_dp4a / the prefill F32-oracle check). Use batched for
         * gemma4 (≫ faster than N sequential forwards; ~3x on 16 vision tokens and
         * scales with image-token count). CUDA_LLM_VLM_SEQ_PREFILL=1 reverts.
         * PLE (2B) stays sequential. NOTE: batched prefill is F16 (vs the dp4a
         * sequential it replaces) so the KV cache differs by ~F16 precision; greedy
         * decode matches for many tokens then can diverge in wording on long
         * generations (near-tie flips), both valid — landmark/scene answers stable. */
        int vlm_seq = getenv("CUDA_LLM_VLM_SEQ_PREFILL") &&
                      atoi(getenv("CUDA_LLM_VLM_SEQ_PREFILL")) != 0;
        if (!r->is_gemma4 || vlm_seq ||
            (r->n_embd_per_layer > 0 && r->d_ple_combined)) {
            return cuda_llm_prefill_sequential(r, token_ids, embeddings, embd_stride,
                                               n_tokens, start_pos);
        }
        /* fall through to the batched body (handles embeddings at line ~16445) */
    }
    if (r->is_gemma4 && r->n_embd_per_layer > 0 && r->d_ple_combined) {
        /* Gemma4 with PLE (older 2B model): the per-layer embedding path is
         * still per-token in the batched code (if 0 && ...). Keep sequential. */
        return cuda_llm_prefill_sequential(r, token_ids, embeddings, embd_stride,
                                           n_tokens, start_pos);
    }
    if (!r->is_gemma4) {
        /* Non-Gemma4, non-hybrid: use sequential (defensive — keep correctness). */
        return cuda_llm_prefill_sequential(r, token_ids, embeddings, embd_stride,
                                           n_tokens, start_pos);
    }

    /* Chunked prefill for long context on limited VRAM. The batched path below
     * allocates O(n_tokens) F32 activation buffers (q/xb2/k/v/gate/up, ~2 GB at
     * 16384); alongside the 10.4 GB weights these don't fit on a 16 GB card, and
     * even an 8192 chunk leaves cuBLAS too little GEMM workspace (it falls off
     * the tensor-core path -> >10x slower). Process the tokens in blocks,
     * advancing start_pos, so each block's buffers stay bounded — every token's
     * forward pass only needs its own hidden state plus the KV cache of all
     * preceding tokens, which persists across blocks. The final block's
     * last-token hidden state is the result. Chunk size is a multiple of
     * swa_window so the circular SWA cache holds the window history in linear
     * order at block boundaries (the fast cross-chunk windowed-FA2 path).
     * Token-ids path only (the bench path); start_pos==0 entry only. */
    {
        int pchunk = 4096;
        const char *e = getenv("CUDA_LLM_PREFILL_CHUNK");
        if (e) { int v = atoi(e); if (v > 0) pchunk = v; }
        if (r->swa_window_size > 0 && pchunk % r->swa_window_size != 0)
            pchunk = (pchunk / r->swa_window_size + 1) * r->swa_window_size;
        if (token_ids && start_pos == 0 && n_tokens > pchunk) {
            float *last = NULL;
            for (int off = 0; off < n_tokens; off += pchunk) {
                int c = n_tokens - off;
                if (c > pchunk) c = pchunk;
                last = cuda_llm_prefill(r, token_ids + off, NULL, 0, c, off);
                if (!last) return NULL;
            }
            return last;
        }
    }

    int n_embd = r->n_embd;
    int n_heads = r->n_heads;
    int n_ff = r->n_ff;
    float eps = r->rms_norm_eps;

    /* Allocate batch buffers */
    size_t batch_embd = (size_t)n_tokens * n_embd * sizeof(float);
    int max_kv_dim = r->n_kv_heads * r->head_dim;
    if (r->is_gemma4) {
        int kv1 = r->n_kv_heads * r->head_dim_full;
        int kv2 = r->n_kv_heads * r->head_dim_swa;
        max_kv_dim = kv1 > kv2 ? kv1 : kv2;
    }
    int max_q_dim = n_heads * r->head_dim;
    if (r->is_gemma4) {
        int q1 = n_heads * r->head_dim_full;
        int q2 = n_heads * r->head_dim_swa;
        max_q_dim = q1 > q2 ? q1 : q2;
    }

    float *result = NULL;
    float *h_embd = NULL;
    CUdeviceptr d_batch_x = 0, d_batch_xb = 0, d_batch_q = 0, d_batch_k = 0, d_batch_v = 0;
    CUdeviceptr d_batch_xb2 = 0, d_batch_gate = 0, d_batch_up = 0;
#define CLLM_PREFILL_CU_OR_CLEANUP(call, label) do { \
        CUresult _ce = (call); \
        if (_ce != CUDA_SUCCESS) { \
            fprintf(stderr, "cuda_llm: generic prefill %s failed (err=%d)\n", (label), (int)_ce); \
            goto cleanup; \
        } \
    } while (0)
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemAlloc(&d_batch_x, batch_embd), "alloc batch_x");
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemAlloc(&d_batch_xb, batch_embd), "alloc batch_xb");
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemAlloc(&d_batch_q, (size_t)n_tokens * max_q_dim * sizeof(float)), "alloc batch_q");
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemAlloc(&d_batch_k, (size_t)n_tokens * max_kv_dim * sizeof(float)), "alloc batch_k");
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemAlloc(&d_batch_v, (size_t)n_tokens * max_kv_dim * sizeof(float)), "alloc batch_v");
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemAlloc(&d_batch_xb2, (size_t)n_tokens * max_q_dim * sizeof(float)), "alloc batch_xb2");
    /* FFN gate/up activations [n_tokens x n_ff]. Long-context prefill keeps these
     * bounded by processing the sequence in token-chunks (see the chunked-prefill
     * wrapper above) rather than chunking the FFN within a single pass — each
     * prefill chunk is <= the size that already fits in 16 GB (e.g. 8192). */
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemAlloc(&d_batch_gate, (size_t)n_tokens * n_ff * sizeof(float)), "alloc batch_gate");
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemAlloc(&d_batch_up, (size_t)n_tokens * n_ff * sizeof(float)), "alloc batch_up");

    /* Upload token embeddings */
    if (token_ids) {
        /* Embed each token directly into batch buffer (no host round-trip) */
        if (r->token_embd_type == GGML_TYPE_Q8_0 || r->token_embd_type == GGML_TYPE_Q4_0) {
            for (int t = 0; t < n_tokens; t++) {
                CUdeviceptr dst = d_batch_x + (size_t)t * n_embd * sizeof(float);
                if (r->token_embd_type == GGML_TYPE_Q8_0)
                    launch_embed_q8_0(r, dst, r->d_token_embd, token_ids[t], n_embd);
                else
                    launch_embed_q4_0(r, dst, r->d_token_embd, token_ids[t], n_embd);
                if (r->is_gemma4) {
                    void *args[] = { &dst, &r->embd_scale, &n_embd };
                    cuLaunchKernel(r->fn_scale_f32, (n_embd+255)/256, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
                }
            }
        } else {
            h_embd = (float *)malloc(batch_embd);
            if (!h_embd) { fprintf(stderr, "cuda_llm: generic prefill host embedding buffer alloc failed\n"); goto cleanup; }
            for (int t = 0; t < n_tokens; t++) {
                if (r->token_embd_type == GGML_TYPE_Q2_K) {
                    launch_embed_q2_K(r, r->d_x, r->d_token_embd, token_ids[t], n_embd);
                } else if (r->token_embd_type == GGML_TYPE_Q4_K) {
                    int32_t tok = token_ids[t];
                    void *ea[] = { &r->d_x, &r->d_token_embd, &tok, &n_embd };
                    cuLaunchKernel(r->fn_embed_q4_K, (n_embd+255)/256, 1, 1, 256, 1, 1, 0, r->stream, ea, NULL);
                } else {
                    launch_embed(r, r->d_x, r->d_token_embd, token_ids[t], n_embd);
                }
                if (r->is_gemma4) launch_scale(r, r->d_x, r->embd_scale, n_embd);
                CLLM_PREFILL_CU_OR_CLEANUP(cuMemcpyDtoHAsync(h_embd + t * n_embd, r->d_x,
                                                             n_embd * sizeof(float), r->stream),
                                           "embedding download");
            }
            CLLM_PREFILL_CU_OR_CLEANUP(cuStreamSynchronize(r->stream), "embedding sync");
            CLLM_PREFILL_CU_OR_CLEANUP(cuMemcpyHtoD(d_batch_x, h_embd, batch_embd),
                                       "embedding batch upload");
            free(h_embd);
            h_embd = NULL;
        }
    } else if (embeddings) {
        /* Pre-computed embeddings path */
        if (embd_stride == n_embd) {
            CLLM_PREFILL_CU_OR_CLEANUP(cuMemcpyHtoD(d_batch_x, embeddings, batch_embd),
                                       "embedding batch upload");
        } else {
            for (int t = 0; t < n_tokens; t++)
                CLLM_PREFILL_CU_OR_CLEANUP(cuMemcpyHtoDAsync(d_batch_x + (size_t)t * n_embd * sizeof(float),
                                                             embeddings + t * embd_stride,
                                                             n_embd * sizeof(float), r->stream),
                                           "embedding row upload");
        }
        /* Gemma4: scale embeddings only for token inputs, not vision embeddings */
    } else {
        goto cleanup;
    }

    /* Gemma4: stash token_id for PLE (use last token) */
    if (r->is_gemma4 && token_ids)
        r->current_token_id = token_ids[n_tokens - 1];

    /* Lazily create cuBLAS handle and allocate FA2 F16 buffers */
    if (r->use_cublas && !r->cublas) {
        int cublas_ret = cublasewCreate(&r->cublas, r->stream);
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_llm: cuBLAS handle creation: ret=%d\n", cublas_ret);
    }
    /* F16 input-conversion scratch for batched cuBLAS GEMM (Q6_K/Q4_K/F16 paths).
     * Every GEMM converts its [n_tokens x in_dim] F32 input into this buffer, so
     * it must hold n_tokens * max(in_dim) F16 elements. The largest in_dim is the
     * FFN-down (n_ff); the attention output projection (in_dim = n_heads*head_dim,
     * 8192 for gemma4 full-attn) also exceeds n_embd, so size to the true max. */
    if (r->use_cublas && r->cublas) {
        int q_dim_full = n_heads * (r->is_gemma4 ? r->head_dim_full : r->head_dim);
        size_t max_in = (size_t)n_ff;
        if ((size_t)q_dim_full > max_in) max_in = (size_t)q_dim_full;
        if ((size_t)n_embd   > max_in) max_in = (size_t)n_embd;
        size_t need = (size_t)n_tokens * max_in * sizeof(uint16_t);
        if (r->d_batch_f16_scratch && r->batch_f16_scratch_bytes < need) {
            cuMemFree(r->d_batch_f16_scratch);
            r->d_batch_f16_scratch = 0;
        }
        if (!r->d_batch_f16_scratch) {
            if (cuMemAlloc(&r->d_batch_f16_scratch, need) == CUDA_SUCCESS)
                r->batch_f16_scratch_bytes = need;
            else
                r->d_batch_f16_scratch = 0;
        }
    }
    /* Double-buffered weight-dequant overlap (scaffolding, default OFF). Measured
     * NO speedup on this GPU: dequant and the GEMM are both DRAM-bandwidth-bound,
     * so overlapping them just shares the same bandwidth. Enable for A/B with
     * CUDA_LLM_PREFILL_OVERLAP=1 (costs a 2nd 256MB weight buffer + stream). */
    if (r->use_cublas && r->cublas && !r->stream_dq) {
        const char *ov = getenv("CUDA_LLM_PREFILL_OVERLAP");
        if (ov && atoi(ov) != 0) {
            int ok = 1;
            if (!r->d_f16_scratch && cuMemAlloc(&r->d_f16_scratch, 256*1024*1024) != CUDA_SUCCESS) ok = 0;
            if (ok && cuMemAlloc(&r->d_f16_scratch2, 256*1024*1024) != CUDA_SUCCESS) ok = 0;
            if (ok && cuStreamCreate(&r->stream_dq, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) ok = 0;
            for (int i = 0; ok && i < 2; i++) {
                if (cuEventCreate(&r->dq_evt[i], CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) ok = 0;
                if (ok && cuEventCreate(&r->gemm_evt[i], CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) ok = 0;
                if (ok) cuEventRecord(r->gemm_evt[i], r->stream); /* start signaled */
            }
            r->prefill_overlap = ok;
            if (!ok && r->verbose >= 1)
                fprintf(stderr, "cuda_llm: prefill dequant/GEMM overlap setup failed; using single stream\n");
        }
    }
    r->dq_pp = 0;
    /* Each prefill starts with both weight buffers free. */
    if (r->prefill_overlap) {
        cuEventRecord(r->gemm_evt[0], r->stream);
        cuEventRecord(r->gemm_evt[1], r->stream);
    }
    if (r->fn_fa2_attn_256 && n_tokens > 0 &&
        (!r->d_fa2_q_f16 || n_tokens > r->fa2_buf_max_s)) {
        /* These buffers double as the cuBLAS-d512 attention scratch:
         *   d_fa2_q_f16      holds the F16 Q [n_tokens x q_dim] (also FA2's Q),
         *   d_fa2_o_f16      holds the F32 score matrix for ONE query-chunk (also
         *                    FA2's F16 output),
         *   d_fa2_scores_f16 holds the F16 copy of that score matrix.
         * cuBLAS-d512 chunks the query dim so the score matrix is
         * gqa*q_chunk*total_s (<= D512_SCORE_CAP_ELEMS), NOT gqa*n_tokens*total_s
         * which is 4 GB+ at n_tokens=8192. q_chunk is recomputed identically in
         * launch_cublas_d512_attention. An undersized buffer makes the GEMM
         * write out of bounds. */
        int max_hd = (n_heads * r->head_dim_full) > (n_heads * r->head_dim_swa) ?
                      (n_heads * r->head_dim_full) : (n_heads * r->head_dim_swa);
        int max_s = n_tokens > 128 ? n_tokens : 128;
        size_t total_s = (size_t)start_pos + max_s;
        size_t q_bytes = (size_t)max_s * max_hd * sizeof(uint16_t);   /* converted Q (full) */

        /* Free the old buffers first so their bytes count as free memory. */
        if (r->d_fa2_q_f16) { cuMemFree(r->d_fa2_q_f16); r->d_fa2_q_f16 = 0; }
        if (r->d_fa2_o_f16) { cuMemFree(r->d_fa2_o_f16); r->d_fa2_o_f16 = 0; }
        if (r->d_fa2_scores_f16) { cuMemFree(r->d_fa2_scores_f16); r->d_fa2_scores_f16 = 0; }
        r->fa2_buf_max_s = 0;

        /* Effective per-chunk score cap: the hard ceiling (env/default), further
         * clamped by free GPU memory. The score scratch is o(F32)+sc(F16) = 6
         * bytes/elem; we must leave a workspace margin free or cuBLAS's GEMMs fall
         * off the tensor-core path onto a pathological no-workspace algorithm
         * (measured: 8192-tok prefill 1483 -> <40 tok/s). q_chunk is recomputed
         * from r->d512_cap_elems in launch_cublas_d512_attention to match exactly. */
        size_t cap = D512_SCORE_CAP_ELEMS;
        size_t free_b = 0, total_b = 0;
        if (cuMemGetInfo(&free_b, &total_b) == CUDA_SUCCESS && free_b > 0) {
            /* Only shrink below the ceiling to keep the d512 scratch ALLOC itself
             * from failing (and disabling d512 -> slow fallback). The ceiling
             * already bounds the scratch small enough to leave cuBLAS its GEMM
             * workspace; this margin just covers buffers allocated later in the
             * layer loop (d_f16_scratch + d_batch_f16_scratch ~0.5 GB). Keep it
             * small so the common 8192 case stays at the ceiling (q_chunk~128)
             * rather than over-chunking to q_chunk=1. */
            const size_t margin = (size_t)512 * 1024 * 1024;
            size_t reserve = q_bytes + margin;
            size_t avail = free_b > reserve ? free_b - reserve : 0;
            size_t cap_mem = avail / 6; /* 4 (F32) + 2 (F16) bytes per score elem */
            if (cap_mem < cap) cap = cap_mem;
            if (cap < (size_t)n_heads * total_s) cap = (size_t)n_heads * total_s; /* >=1 query token/chunk */
        }
        r->d512_cap_elems = cap;

        int q_chunk = (int)(cap / ((size_t)n_heads * total_s));
        if (q_chunk < 1) q_chunk = 1;
        if (q_chunk > max_s) q_chunk = max_s;
        size_t score_elems = (size_t)n_heads * q_chunk * total_s;     /* one chunk, worst-case gqa=n_heads */
        size_t o_bytes = score_elems * sizeof(float);                 /* F32 scores (chunked) */
        if (q_bytes > o_bytes) o_bytes = q_bytes;                     /* FA2 reuses d_fa2_o_f16 as F16 output */
        size_t sc_bytes = score_elems * sizeof(uint16_t);             /* F16 scores (separate from Q) */
        if (cuMemAlloc(&r->d_fa2_q_f16, q_bytes) != CUDA_SUCCESS)
            r->d_fa2_q_f16 = 0;
        if (cuMemAlloc(&r->d_fa2_o_f16, o_bytes) != CUDA_SUCCESS)
            r->d_fa2_o_f16 = 0;
        if (cuMemAlloc(&r->d_fa2_scores_f16, sc_bytes) != CUDA_SUCCESS)
            r->d_fa2_scores_f16 = 0;
        if (r->d_fa2_q_f16 && r->d_fa2_o_f16 && r->d_fa2_scores_f16) r->fa2_buf_max_s = max_s;
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_llm: FA2 F16 buffers allocated (max_tok=%d, q=%.1fMB, o=%.1fMB, q_chunk=%d, %s)\n",
                    max_s, q_bytes/1048576.0, o_bytes/1048576.0, q_chunk,
                    (r->d_fa2_q_f16 && r->d_fa2_o_f16) ? "OK" : "FAIL");
    }
    /* Linear F16 K/V scratch for batched windowed SWA prefill. The circular SWA
     * cache can't hold all n_tokens at once, so attention reads a linear copy;
     * the circular cache is filled with the window tail afterward for decode.
     * For chunked prefill (start_pos>0, a multiple of swa_window) the buffer also
     * holds a `swa_window` history prefix copied from the circular cache so the
     * chunk's first tokens can attend across the chunk boundary — hence +window. */
    if (r->is_gemma4 && r->swa_window_size > 0 && n_tokens > r->swa_window_size &&
        (start_pos == 0 || start_pos % r->swa_window_size == 0) && r->fn_fa2_attn_256 &&
        (!r->d_swa_k_lin || n_tokens + r->swa_window_size > r->swa_lin_max_tokens)) {
        int max_kv = r->n_kv_heads * r->head_dim_swa;
        size_t cap_tok = (size_t)n_tokens + r->swa_window_size; /* +window history prefix */
        size_t bytes = cap_tok * max_kv * sizeof(uint16_t);
        if (r->d_swa_k_lin) { cuMemFree(r->d_swa_k_lin); r->d_swa_k_lin = 0; }
        if (r->d_swa_v_lin) { cuMemFree(r->d_swa_v_lin); r->d_swa_v_lin = 0; }
        r->swa_lin_max_tokens = 0;
        if (cuMemAlloc(&r->d_swa_k_lin, bytes) != CUDA_SUCCESS) r->d_swa_k_lin = 0;
        if (cuMemAlloc(&r->d_swa_v_lin, bytes) != CUDA_SUCCESS) r->d_swa_v_lin = 0;
        if (r->d_swa_k_lin && r->d_swa_v_lin) r->swa_lin_max_tokens = (int)cap_tok;
    }

    int n_layers = r->n_layers;
    if (r->max_layers > 0 && r->max_layers < n_layers) n_layers = r->max_layers;

    /* Process transformer blocks */
    double profile_layer_ms = 0;
    for (int l = 0; l < n_layers; l++) {
        double _tl0 = get_time_ms();
        cuda_layer *cl = &r->layers[l];
        if (cl->is_ssm) {
            fprintf(stderr,
                    "cuda_llm: unsupported generic SSM prefill route at layer %d; "
                    "hybrid SSM models should use cuda_llm_prefill_qwen35\n", l);
            goto cleanup;
        }

        int hd = cl->is_swa ? r->head_dim_swa : r->head_dim_full;
        int layer_kv_heads = cl->n_kv_heads;
        int local_kv_dim = layer_kv_heads * hd;
        int local_q_dim = n_heads * hd;

        /* 1. Pre-attention RMSNorm: batch_xb = rmsnorm(batch_x) */
        cuMemcpyDtoDAsync(d_batch_xb, d_batch_x, batch_embd, r->stream);
        { void *a[] = { &d_batch_xb, &cl->attn_norm_w, &n_embd, &eps, &n_tokens };
          cuLaunchKernel(r->fn_batch_rmsnorm_f32, n_tokens, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* 2. Q/K/V projections (batched) */
        if (r->debug_layers >= 3) cuStreamSynchronize(r->stream);
        double _proj_t0 = (r->debug_layers >= 3) ? get_time_ms() : 0;
        launch_batch_matvec(r, d_batch_q, cl->attn_q_w, 0, d_batch_xb,
                            cl->attn_q_rows, cl->attn_q_cols, n_tokens, cl->attn_q_type);
        if (cl->shared_kv_source < 0) {
            launch_batch_matvec(r, d_batch_k, cl->attn_k_w, 0, d_batch_xb,
                                cl->attn_k_rows, cl->attn_k_cols, n_tokens, cl->attn_k_type);
            if (cl->has_v_proj) {
                launch_batch_matvec(r, d_batch_v, cl->attn_v_w, 0, d_batch_xb,
                                    cl->attn_v_rows, cl->attn_v_cols, n_tokens, cl->attn_v_type);
            } else {
                cuMemcpyDtoDAsync(d_batch_v, d_batch_k, (size_t)n_tokens * local_kv_dim * sizeof(float), r->stream);
            }
        }

        /* 3. Batched ops for non-SWA layers; per-token for SWA (circular cache) */
        int kv_src = (cl->shared_kv_source >= 0) ? cl->shared_kv_source : l;
        float attn_scale = 1.0f;

        /* Batched windowed SWA: when the prefill wraps the circular window, store
         * K/V linearly and run a windowed FA2 instead of the slow per-token loop. */
        int swa_aligned = (start_pos == 0) || (start_pos % r->swa_window_size == 0);
        int swa_lin = cl->is_swa && cl->shared_kv_source < 0 && swa_aligned &&
                      n_tokens > r->swa_window_size && hd == 256 &&
                      r->fn_fa2_attn_256 && r->fn_batch_kv_store_swa_f16 &&
                      r->d_swa_k_lin && r->d_swa_v_lin && r->d_fa2_q_f16;

        if (!cl->is_swa || start_pos + n_tokens <= r->swa_window_size || swa_lin) {
            /* ----- Batched path for full-attention or non-wrapping SWA ----- */
            int total_q_heads = n_tokens * n_heads;
            int total_kv_heads = n_tokens * layer_kv_heads;

            if (cl->has_qk_norm) {
                int bd = 1; while (bd < hd) bd <<= 1;
                /* Q/K use the *weighted* per-head RMSNorm (qknorm_f32, 5 args), one
                 * block per (token,head). Using raw_rmsnorm_heads_f32 here was a bug:
                 * it takes 4 args so attn_q_norm_w was mis-marshalled into head_dim,
                 * indexing vec[h*total_heads] -> OOB + wrong hidden states. */
                if (cl->attn_q_norm_w) {
                    void *qa[] = { &d_batch_q, &cl->attn_q_norm_w, &total_q_heads, &hd, &eps };
                    cuLaunchKernel(r->fn_qknorm_f32, total_q_heads, 1, 1, bd, 1, 1, bd*sizeof(float), r->stream, qa, NULL);
                }
                if (cl->shared_kv_source < 0) {
                    if (cl->attn_k_norm_w) {
                        void *ka[] = { &d_batch_k, &cl->attn_k_norm_w, &total_kv_heads, &hd, &eps };
                        cuLaunchKernel(r->fn_qknorm_f32, total_kv_heads, 1, 1, bd, 1, 1, bd*sizeof(float), r->stream, ka, NULL);
                    }
                    /* V: raw (unweighted) per-head RMSNorm over all token*head rows */
                    void *va[] = { &d_batch_v, &total_kv_heads, &hd, &eps };
                    cuLaunchKernel(r->fn_raw_rmsnorm_heads_f32, total_kv_heads, 1, 1, bd, 1, 1, bd*sizeof(float), r->stream, va, NULL);
                }
            }

            if (cl->is_swa) {
                /* SWA: rotate ALL head_dim pairs (rope_dim = head_dim), matching the
                 * sequential path's zero_pairs=0. Passing r->n_rope_pairs (=full
                 * head_dim/2) here made rope_dim = 2*n_rope_pairs = full head_dim,
                 * i.e. half the correct rotation frequency on every SWA layer ->
                 * batched prefill diverged from sequential (garbage decode). */
                launch_batch_rope(r, d_batch_q, n_heads, n_tokens, hd, start_pos, r->rope_freq_base_swa, 0);
                if (cl->shared_kv_source < 0)
                    launch_batch_rope(r, d_batch_k, layer_kv_heads, n_tokens, hd, start_pos, r->rope_freq_base_swa, 0);
            } else {
                launch_batch_rope_with_factors(r, d_batch_q, n_heads, n_tokens, hd,
                                               start_pos, r->d_rope_inv_freq_full);
                if (cl->shared_kv_source < 0)
                    launch_batch_rope_with_factors(r, d_batch_k, layer_kv_heads, n_tokens, hd,
                                                   start_pos, r->d_rope_inv_freq_full);
            }

            if (swa_lin) {
                int win = r->swa_window_size;
                /* Chunked prefill (start_pos>0, a multiple of window): prepend a
                 * `window` history prefix so the chunk's first tokens can attend
                 * across the chunk boundary. Because start_pos is a multiple of
                 * window, the circular cache already holds the last `window`
                 * tokens in LINEAR position order, so a straight copy gives the
                 * history. start_pos==0 -> hist=0 (no prefix, original behaviour). */
                int hist = (start_pos > 0) ? win : 0;
                if (hist > 0) {
                    size_t hb = (size_t)win * local_kv_dim * sizeof(uint16_t);
                    cuMemcpyDtoDAsync(r->d_swa_k_lin, r->d_key_cache[l], hb, r->stream);
                    cuMemcpyDtoDAsync(r->d_swa_v_lin, r->d_value_cache[l], hb, r->stream);
                }
                /* Store this chunk's K/V LINEARLY after the prefix, at linear
                 * positions [hist, hist+n_tokens). */
                void *lin[] = { &r->d_swa_k_lin, &r->d_swa_v_lin, &d_batch_k, &d_batch_v,
                                &hist, &local_kv_dim, &n_tokens };
                cuLaunchKernel(r->fn_batch_kv_store_f16,
                               (local_kv_dim + 255) / 256, n_tokens, 1,
                               256, 1, 1, 0, r->stream, lin, NULL);
                /* Fill the circular cache with ONLY the last `window` tokens — each
                 * maps to a unique slot, so no write race (storing all n_tokens
                 * would race multiple positions onto the same slot). This leaves
                 * the cache identical to the per-token path for decode AND for the
                 * next chunk's history prefix. Use the ABSOLUTE tail position so
                 * the circular slots match (start_pos+tail_start)%window. */
                int tail = win;
                int tail_start = n_tokens - tail;          /* chunk-local */
                int tail_abs = start_pos + tail_start;     /* absolute first-tail position */
                CUdeviceptr k_tail = d_batch_k + (size_t)tail_start * local_kv_dim * sizeof(float);
                CUdeviceptr v_tail = d_batch_v + (size_t)tail_start * local_kv_dim * sizeof(float);
                void *circ[] = { &r->d_key_cache[l], &r->d_value_cache[l], &k_tail, &v_tail,
                                 &tail_abs, &local_kv_dim, &tail, &win };
                cuLaunchKernel(r->fn_batch_kv_store_swa_f16,
                               (local_kv_dim + 255) / 256, tail, 1,
                               256, 1, 1, 0, r->stream, circ, NULL);
                /* Windowed FA2 over [history ++ chunk]: query i sits at linear key
                 * position hist+i, so q_off=hist makes the kernel's causal+window
                 * mask attend exactly [hist+i-window+1, hist+i]. */
                launch_fa2_attention(r, d_batch_xb2, d_batch_q,
                                    r->d_swa_k_lin, r->d_swa_v_lin,
                                    n_tokens, hist, n_heads, layer_kv_heads,
                                    hd, local_kv_dim, attn_scale, win);
                goto swa_attn_done;
            }

            if (cl->shared_kv_source < 0) {
                void *kva[] = { &r->d_key_cache[l], &r->d_value_cache[l], &d_batch_k, &d_batch_v,
                                &start_pos, &local_kv_dim, &n_tokens };
                cuLaunchKernel(r->fn_batch_kv_store_f16,
                               (local_kv_dim + 255) / 256, n_tokens, 1,
                               256, 1, 1, 0, r->stream, kva, NULL);
            }

            if (r->debug_layers >= 3) cuStreamSynchronize(r->stream);
            double _attn_t0 = (r->debug_layers >= 3) ? get_time_ms() : 0;
            if (n_tokens >= 64 && hd == 256 && r->fn_fa2_attn_256 && r->d_fa2_q_f16) {
                /* window=0: this branch only runs when start_pos+n_tokens<=window,
                 * so causal == windowed and no extra masking is needed. */
                launch_fa2_attention(r, d_batch_xb2, d_batch_q,
                                    r->d_key_cache[kv_src], r->d_value_cache[kv_src],
                                    n_tokens, start_pos, n_heads, layer_kv_heads,
                                    hd, local_kv_dim, attn_scale, 0);
            } else if (n_tokens >= 64 && r->cublas && r->fn_causal_softmax_f32 && r->d_fa2_q_f16) {
                launch_cublas_d512_attention(r, d_batch_xb2, d_batch_q,
                                            r->d_key_cache[kv_src], r->d_value_cache[kv_src],
                                            n_tokens, start_pos, n_heads, layer_kv_heads,
                                            hd, local_kv_dim, attn_scale);
            } else if (n_tokens >= 64 && r->fn_batch_attn_all_tokens_f32) {
                int max_seq = start_pos + n_tokens;
                size_t smem = (size_t)n_tokens * max_seq * sizeof(float);
                void *a[] = { &d_batch_xb2, &d_batch_q,
                              &r->d_key_cache[kv_src], &r->d_value_cache[kv_src],
                              &n_tokens, &start_pos, &n_heads, &layer_kv_heads,
                              &hd, &local_kv_dim, &attn_scale };
                cuLaunchKernel(r->fn_batch_attn_all_tokens_f32,
                               n_heads, 1, 1, 256, 1, 1, smem, r->stream, a, NULL);
            } else {
                /* For smaller batches (<64 tokens), use batch_attn_causal_f32
                   which has better occupancy (n_heads × n_tokens blocks) */
                launch_batch_attention(r, d_batch_xb2, d_batch_q,
                                      r->d_key_cache[kv_src], r->d_value_cache[kv_src],
                                      n_tokens, start_pos, n_heads, layer_kv_heads,
                                      hd, local_kv_dim, attn_scale);
            }
            if (r->debug_layers >= 3) {
                cuStreamSynchronize(r->stream);
                double _attn_t1 = get_time_ms();
                fprintf(stderr, "    layer %d attn: %.2f ms\n", l, _attn_t1 - _attn_t0);
            }
        } else {
            /* ----- Per-token path for SWA layers (circular KV cache) ----- */
            for (int t = 0; t < n_tokens; t++) {
                int pos = start_pos + t;
                CUdeviceptr q_t = d_batch_q + (size_t)t * local_q_dim * sizeof(float);
                CUdeviceptr k_t = d_batch_k + (size_t)t * local_kv_dim * sizeof(float);
                CUdeviceptr v_t = d_batch_v + (size_t)t * local_kv_dim * sizeof(float);
                CUdeviceptr xb2_t = d_batch_xb2 + (size_t)t * local_q_dim * sizeof(float);

                if (cl->has_qk_norm) {
                    if (cl->attn_q_norm_w) launch_qknorm(r, q_t, cl->attn_q_norm_w, n_heads, hd, eps);
                    if (cl->shared_kv_source < 0) {
                        if (cl->attn_k_norm_w) launch_qknorm(r, k_t, cl->attn_k_norm_w, layer_kv_heads, hd, eps);
                        int bd = 1; while (bd < hd) bd <<= 1;
                        void *va[] = { &v_t, &layer_kv_heads, &hd, &eps };
                        cuLaunchKernel(r->fn_raw_rmsnorm_heads_f32, layer_kv_heads, 1, 1, bd, 1, 1, bd*sizeof(float), r->stream, va, NULL);
                    }
                }

                int half_swa = hd / 2, zero = 0;
                void *qa[] = { &q_t, &n_heads, &hd, &pos, &r->rope_freq_base_swa, &zero };
                cuLaunchKernel(r->fn_rope_neox_f32, n_heads, 1, 1, half_swa, 1, 1, 0, r->stream, qa, NULL);
                if (cl->shared_kv_source < 0) {
                    void *ka[] = { &k_t, &layer_kv_heads, &hd, &pos, &r->rope_freq_base_swa, &zero };
                    cuLaunchKernel(r->fn_rope_neox_f32, layer_kv_heads, 1, 1, half_swa, 1, 1, 0, r->stream, ka, NULL);
                }

                if (cl->shared_kv_source < 0) {
                    int slot = pos % r->swa_window_size;
                    launch_kv_store(r, r->d_key_cache[l], r->d_value_cache[l], k_t, v_t, slot, local_kv_dim);
                    if (r->kv_cache_q8) launch_kv_store_q8(r, l, k_t, v_t, slot, local_kv_dim);
                }

                launch_attention_swa(r, xb2_t, q_t,
                                    r->d_key_cache[kv_src], r->d_value_cache[kv_src],
                                    n_heads, layer_kv_heads, hd, local_kv_dim,
                                    pos, r->swa_window_size, attn_scale);
            }
        }
        swa_attn_done:;

        /* 4. Batched output projection */
        launch_batch_matvec(r, d_batch_xb, cl->attn_output_w, 0, d_batch_xb2,
                            cl->attn_output_rows, cl->attn_output_cols, n_tokens, cl->attn_output_type);

        /* 5. Batched post-attention norm */
        { void *a[] = { &d_batch_xb, &cl->post_attn_norm_w, &n_embd, &eps, &n_tokens };
          cuLaunchKernel(r->fn_batch_rmsnorm_f32, n_tokens, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* 6. Residual: batch_x += batch_xb */
        { int total = n_tokens * n_embd;
          void *a[] = { &d_batch_x, &d_batch_xb, &total };
          cuLaunchKernel(r->fn_vision_add, (total+255)/256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        if (r->debug_layers >= 3) {
            cuStreamSynchronize(r->stream);
            double _proj_t1 = get_time_ms();
            fprintf(stderr, "    layer %d proj+attn: %.2f ms\n", l, _proj_t1 - _proj_t0);
        }
        /* 7. Batched FFN */
        cuMemcpyDtoDAsync(d_batch_xb, d_batch_x, batch_embd, r->stream);
        { void *a[] = { &d_batch_xb, &cl->ffn_norm_w, &n_embd, &eps, &n_tokens };
          cuLaunchKernel(r->fn_batch_rmsnorm_f32, n_tokens, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Run gate and up matvecs in parallel on two streams when CUDA_LLM_PAR_FFN=1.
         * Both matvecs read the same input (d_batch_xb) and use the same per-layer
         * norm/dequant path; running them concurrently lets the GPU overlap the two
         * dequant writes and two cuBLAS GEMMs. Joined at GELU (both must complete). */
        if (getenv("CUDA_LLM_PAR_FFN") && r->stream_dq) {
            if (!r->gate_done_evt) cuEventCreate(&r->gate_done_evt, CU_EVENT_DISABLE_TIMING);
            launch_batch_matvec(r, d_batch_gate, cl->ffn_gate_w, 0, d_batch_xb,
                                cl->ffn_gate_rows, cl->ffn_gate_cols, n_tokens, cl->ffn_gate_type);
            cuEventRecord(r->gate_done_evt, r->stream);
            /* Run up matvec on the alt stream, wait for gate on main, then sync the main. */
            launch_batch_matvec_stream(r, d_batch_up, cl->ffn_up_w, 0, d_batch_xb,
                                       cl->ffn_up_rows, cl->ffn_up_cols, n_tokens, cl->ffn_up_type,
                                       r->stream_dq);
            cuStreamWaitEvent(r->stream, r->gate_done_evt, 0);
        } else {
            launch_batch_matvec(r, d_batch_gate, cl->ffn_gate_w, 0, d_batch_xb,
                                cl->ffn_gate_rows, cl->ffn_gate_cols, n_tokens, cl->ffn_gate_type);
            launch_batch_matvec(r, d_batch_up, cl->ffn_up_w, 0, d_batch_xb,
                                cl->ffn_up_rows, cl->ffn_up_cols, n_tokens, cl->ffn_up_type);
        }

        { int total = n_tokens * n_ff;
          void *a[] = { &d_batch_gate, &d_batch_up, &total };
          cuLaunchKernel(r->fn_gelu_mul_f32, (total+255)/256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        launch_batch_matvec(r, d_batch_xb, cl->ffn_down_w, 0, d_batch_gate,
                            cl->ffn_down_rows, cl->ffn_down_cols, n_tokens, cl->ffn_down_type);

        /* Post-FFN norm */
        { void *a[] = { &d_batch_xb, &cl->post_ffw_norm_w, &n_embd, &eps, &n_tokens };
          cuLaunchKernel(r->fn_batch_rmsnorm_f32, n_tokens, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Residual: batch_x += batch_xb */
        { int total = n_tokens * n_embd;
          void *a[] = { &d_batch_x, &d_batch_xb, &total };
          cuLaunchKernel(r->fn_vision_add, (total+255)/256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Per-layer embedding injection — skip in batched mode, fall back to sequential for PLE models */
        if (0 && r->d_ple_combined && r->n_embd_per_layer > 0) {
            int ple_dim = r->n_embd_per_layer;
            for (int t = 0; t < n_tokens; t++) {
                CUdeviceptr x_t = d_batch_x + (size_t)t * n_embd * sizeof(float);
                /* Need per-token PLE combined for this token */
                /* For now, precompute PLE for this token using r->d_x as scratch */
                /* Copy token hidden state to d_x */
                cuMemcpyDtoDAsync(r->d_x, x_t, n_embd * sizeof(float), r->stream);

                /* PLE combined must have been precomputed for each token.
                   Since we can't easily batch-precompute PLE for all tokens,
                   we need to do it here per token per layer. This is the PLE slice for layer l. */
                /* For token_ids path: PLE uses token_id to look up per_layer_token_embd */
                /* For embedding path: PLE uses pad token (0) */
                int tok_id = (token_ids && t < n_tokens) ? token_ids[t] : 0;

                /* Precompute PLE for this token at this layer */
                float *h_ple = (float *)alloca(ple_dim * sizeof(float));
                /* tok_ple = per_layer_token_embd[tok_id][l*ple_dim .. (l+1)*ple_dim-1] */
                {
                    size_t row_bytes = dequant_row_size(r->h_ple_token_embd.type, r->h_ple_token_embd.n_cols);
                    const unsigned char *row = (const unsigned char *)r->h_ple_token_embd.data + (size_t)tok_id * row_bytes;
                    float *full_ple = (float *)alloca(r->h_ple_token_embd.n_cols * sizeof(float));
                    dequant_row(r->h_ple_token_embd.type, row, full_ple, r->h_ple_token_embd.n_cols);
                    memcpy(h_ple, full_ple + l * ple_dim, ple_dim * sizeof(float));
                }
                /* model_proj @ x for this layer slice */
                {
                    float h_x[8192]; /* n_embd max */
                    cuStreamSynchronize(r->stream);
                    cuMemcpyDtoH(h_x, r->d_x, n_embd * sizeof(float));
                    size_t proj_row_bytes = dequant_row_size(r->h_ple_model_proj.type, r->h_ple_model_proj.n_cols);
                    for (int p = 0; p < ple_dim; p++) {
                        const unsigned char *prow = (const unsigned char *)r->h_ple_model_proj.data
                            + (size_t)(l * ple_dim + p) * proj_row_bytes;
                        float *pdata = (float *)alloca(n_embd * sizeof(float));
                        dequant_row(r->h_ple_model_proj.type, prow, pdata, n_embd);
                        float dot = 0;
                        for (int i = 0; i < n_embd; i++) dot += pdata[i] * h_x[i];
                        h_ple[p] += dot;
                    }
                }
                /* Normalize */
                if (r->h_ple_proj_norm.data) {
                    float *nw = (float *)alloca(ple_dim * n_layers * sizeof(float));
                    dequant_row(r->h_ple_proj_norm.type, r->h_ple_proj_norm.data, nw, ple_dim * n_layers);
                    for (int p = 0; p < ple_dim; p++) h_ple[p] *= nw[l * ple_dim + p];
                }
                cuMemcpyHtoD(r->d_ple_buf, h_ple, ple_dim * sizeof(float));

                /* GPU: inp_gate @ x */
                launch_matvec_auto(r, r->d_ple_proj, cl->ple_inp_gate_w, r->d_x,
                                  cl->ple_inp_gate_rows, cl->ple_inp_gate_cols, cl->ple_inp_gate_type);
                /* GELU(gate) * ple */
                launch_gelu_elementwise_mul(r, r->d_ple_buf, r->d_ple_proj, ple_dim);
                /* proj @ ple → [n_embd] */
                launch_matvec_auto(r, r->d_ple_proj, cl->ple_proj_w, r->d_ple_buf,
                                  cl->ple_proj_rows, cl->ple_proj_cols, cl->ple_proj_type);
                /* Post-norm */
                if (cl->ple_post_norm_w)
                    launch_rmsnorm(r, r->d_ple_proj, r->d_ple_proj, cl->ple_post_norm_w, n_embd, eps);
                /* Add to batch token */
                launch_add(r, x_t, r->d_ple_proj, n_embd);
            }
        }

        /* Layer output scaling */
        if (cl->layer_scale_val != 1.0f) {
            float sv = cl->layer_scale_val;
            int total = n_tokens * n_embd;
            void *a[] = { &d_batch_x, &sv, &total };
            cuLaunchKernel(r->fn_vision_scale, (total+255)/256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL);
        }

        if (r->debug_layers >= 2) {
            cuStreamSynchronize(r->stream);
            double _tl1 = get_time_ms();
            fprintf(stderr, "  prefill layer %d/%d: %.1f ms (hd=%d, swa=%d, kv_h=%d)\n",
                    l, n_layers, _tl1 - _tl0, hd, cl->is_swa, layer_kv_heads);
        }
    }
    if (r->debug_layers >= 2) {
        cuStreamSynchronize(r->stream);
        fprintf(stderr, "cuda_llm: prefill %d layers done\n", n_layers);
    }

    /* Final RMSNorm on last token */
    CUdeviceptr d_last = d_batch_x + (size_t)(n_tokens - 1) * n_embd * sizeof(float);
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemcpyDtoDAsync(r->d_x, d_last, n_embd * sizeof(float), r->stream),
                               "last token copy");
    launch_rmsnorm(r, r->d_x, r->d_x, r->d_output_norm, n_embd, eps);

    /* Sync and copy to host */
    CLLM_PREFILL_CU_OR_CLEANUP(cuStreamSynchronize(r->stream), "final norm sync");
    result = r->h_output;
    CLLM_PREFILL_CU_OR_CLEANUP(cuMemcpyDtoH(result, r->d_x, n_embd * sizeof(float)),
                               "hidden download");

cleanup:
    free(h_embd);
    /* Free batch buffers */
    if (d_batch_x) cuMemFree(d_batch_x);
    if (d_batch_xb) cuMemFree(d_batch_xb);
    if (d_batch_q) cuMemFree(d_batch_q);
    if (d_batch_k) cuMemFree(d_batch_k);
    if (d_batch_v) cuMemFree(d_batch_v);
    if (d_batch_xb2) cuMemFree(d_batch_xb2);
    if (d_batch_gate) cuMemFree(d_batch_gate);
    if (d_batch_up) cuMemFree(d_batch_up);
#undef CLLM_PREFILL_CU_OR_CLEANUP

    return result;
}

float *cuda_llm_prefill_logits(cuda_llm_runner *r, const int32_t *token_ids,
                                const float *embeddings, int embd_stride,
                                int n_tokens, int start_pos) {
    float *hidden = cuda_llm_prefill(r, token_ids, embeddings, embd_stride, n_tokens, start_pos);
    if (!hidden) return NULL;

    /* Run lm_head on the last token's hidden state (already in r->d_x) */
    launch_matvec_auto(r, r->d_logits, r->d_output_w, r->d_x,
                       r->n_vocab, r->n_embd, r->output_w_type);

    /* Logit softcapping (Gemma4) */
    if (r->is_gemma4 && r->final_logit_softcapping > 0) {
        void *a[] = { &r->d_logits, &r->n_vocab, &r->final_logit_softcapping };
        cuLaunchKernel(r->fn_logit_softcap_f32, (r->n_vocab+255)/256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL);
    }

    cuStreamSynchronize(r->stream);
    cuMemcpyDtoH(r->h_output, r->d_logits, (size_t)r->n_vocab * sizeof(float));
    return r->h_output;
}

/* ======================================================================== */
/* GPU Vision Encoder                                                       */
/* ======================================================================== */

/* Helper: upload F16 weight tensor to GPU. Dequant non-F16 types to F16. */
static CUdeviceptr vis_upload_f16(const qtensor *t) {
    if (!t->data) return 0;
    int n = t->n_rows * t->n_cols;
    size_t nbytes = (size_t)n * sizeof(uint16_t);

    if (t->type == GGML_TYPE_F16) {
        CUdeviceptr d; cuMemAlloc(&d, nbytes);
        cuMemcpyHtoD(d, t->data, nbytes);
        return d;
    }
    /* Dequant to F32, then F32→F16 */
    float *f32 = (float *)malloc((size_t)n * sizeof(float));
    dequant_row(t->type, t->data, f32, n);
    uint16_t *f16 = (uint16_t *)malloc(nbytes);
    for (int i = 0; i < n; i++) f16[i] = cllm_f32_to_f16(f32[i]);
    free(f32);
    CUdeviceptr d; cuMemAlloc(&d, nbytes);
    cuMemcpyHtoD(d, f16, nbytes);
    free(f16);
    return d;
}

/* Helper: upload F32 norm weight to GPU as F16 */
static CUdeviceptr vis_upload_norm_f16(const qtensor *t, int n) {
    if (!t->data) return 0;
    float *f32 = (float *)malloc((size_t)n * sizeof(float));
    dequant_row(t->type, t->data, f32, n);
    uint16_t *f16 = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
    for (int i = 0; i < n; i++) f16[i] = cllm_f32_to_f16(f32[i]);
    free(f32);
    CUdeviceptr d; cuMemAlloc(&d, (size_t)n * sizeof(uint16_t));
    cuMemcpyHtoD(d, f16, (size_t)n * sizeof(uint16_t));
    free(f16);
    return d;
}

/* Per-block GPU weights */
typedef struct {
    CUdeviceptr qw, kw, vw, outw;  /* [dim, dim] F16 */
    CUdeviceptr q_norm, k_norm;     /* [head_dim] F16 */
    CUdeviceptr attn_post_norm;     /* [dim] F16 */
    CUdeviceptr gate_w, up_w, down_w; /* FFN F16 */
    CUdeviceptr ffn_post_norm;      /* [dim] F16 */
    CUdeviceptr ln1_w, ln2_w;       /* [dim] F16 */
} vis_gpu_block;

float *cuda_llm_vision_encode(cuda_llm_runner *r, gguf_context *mmproj_gguf,
                               const uint8_t *image, int image_w, int image_h,
                               int *out_tokens, int *out_dim) {
    /* Load vision model config from GGUF */
    int dim = 768, n_heads = 12, ffn_dim = 3072, n_blocks = 16;
    int patch_size = 16, image_size = 224, proj_dim = 2560, spatial_merge = 3;
    float ln_eps = 1e-6f;
    {
        int idx;
        idx = gguf_find_key(mmproj_gguf, "clip.vision.embedding_length");
        if (idx >= 0) dim = mmproj_gguf->kv[idx].value.u32;
        idx = gguf_find_key(mmproj_gguf, "clip.vision.attention.head_count");
        if (idx >= 0) n_heads = mmproj_gguf->kv[idx].value.u32;
        idx = gguf_find_key(mmproj_gguf, "clip.vision.feed_forward_length");
        if (idx >= 0) ffn_dim = mmproj_gguf->kv[idx].value.u32;
        idx = gguf_find_key(mmproj_gguf, "clip.vision.block_count");
        if (idx >= 0) n_blocks = mmproj_gguf->kv[idx].value.u32;
        idx = gguf_find_key(mmproj_gguf, "clip.vision.patch_size");
        if (idx >= 0) patch_size = mmproj_gguf->kv[idx].value.u32;
        idx = gguf_find_key(mmproj_gguf, "clip.vision.image_size");
        if (idx >= 0) image_size = mmproj_gguf->kv[idx].value.u32;
        idx = gguf_find_key(mmproj_gguf, "clip.vision.projection_dim");
        if (idx >= 0) proj_dim = mmproj_gguf->kv[idx].value.u32;
        idx = gguf_find_key(mmproj_gguf, "clip.vision.attention.layer_norm_epsilon");
        if (idx >= 0) ln_eps = mmproj_gguf->kv[idx].value.f32;
    }
    int head_dim = dim / n_heads;
    int ph = image_size / patch_size, pw = ph;
    int n_patches = ph * pw;
    int n_merged = (ph / spatial_merge) * (pw / spatial_merge);
    if (image_w < image_size || image_h < image_size) {
        fprintf(stderr, "cuda_vision: image %dx%d smaller than model input %dx%d\n",
                image_w, image_h, image_size, image_size);
        return NULL;
    }

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_vision: dim=%d heads=%d blocks=%d patches=%d merged=%d proj=%d\n",
                dim, n_heads, n_blocks, n_patches, n_merged, proj_dim);

    /* --- 1. Patch embedding on CPU (fast, ~2% of total) --- */
    qtensor patch_embd_w = cllm_load_tensor(mmproj_gguf, "v.patch_embd.weight", 1);
    float *patches = (float *)calloc((size_t)n_patches * dim, sizeof(float));
    {
        /* Normalize image and extract patches */
        int ps = patch_size;
        float *filters = (float *)malloc((size_t)dim * ps * ps * 3 * sizeof(float));
        dequant_row(patch_embd_w.type, patch_embd_w.data, filters, dim * ps * ps * 3);
        int filter_size = ps * ps * 3;

        for (int py = 0; py < ph; py++) {
            for (int px = 0; px < pw; px++) {
                /* Extract and normalize patch in CHW format (channel-first) */
                float patch_buf[16*16*3]; /* max patch_size=16 */
                for (int c = 0; c < 3; c++) {
                    for (int dy = 0; dy < ps; dy++) {
                        for (int dx = 0; dx < ps; dx++) {
                            int iy = py * ps + dy, ix = px * ps + dx;
                            uint8_t val = image[(iy * image_w + ix) * 3 + c];
                            patch_buf[c * ps * ps + dy * ps + dx] = val / 255.0f * 2.0f - 1.0f;
                        }
                    }
                }
                /* Dot product with each filter */
                float *out = patches + (py * pw + px) * dim;
                for (int f = 0; f < dim; f++) {
                    float sum = 0;
                    const float *fw = filters + f * filter_size;
                    for (int k = 0; k < filter_size; k++) sum += fw[k] * patch_buf[k];
                    out[f] = sum;
                }
            }
        }
        free(filters);
    }

    /* --- 2. Upload patches + position embedding to GPU --- */
    CUdeviceptr d_tokens;
    cuMemAlloc(&d_tokens, (size_t)n_patches * dim * sizeof(float));
    cuMemcpyHtoD(d_tokens, patches, (size_t)n_patches * dim * sizeof(float));
    free(patches);

    /* Position embedding: add X and Y components */
    qtensor pos_embd = cllm_load_tensor(mmproj_gguf, "v.position_embd.weight", 1);
    int n_pos = (int)pos_embd.dims[1];
    CUdeviceptr d_pos_embd = vis_upload_f16(&pos_embd);

    /* Build position index arrays on host, upload */
    int *pos_x = (int *)malloc(n_patches * sizeof(int));
    int *pos_y = (int *)malloc(n_patches * sizeof(int));
    for (int y = 0; y < ph; y++)
        for (int x = 0; x < pw; x++) {
            pos_x[y * pw + x] = x;
            pos_y[y * pw + x] = y;
        }
    CUdeviceptr d_pos_x, d_pos_y;
    cuMemAlloc(&d_pos_x, n_patches * sizeof(int));
    cuMemAlloc(&d_pos_y, n_patches * sizeof(int));
    cuMemcpyHtoD(d_pos_x, pos_x, n_patches * sizeof(int));
    cuMemcpyHtoD(d_pos_y, pos_y, n_patches * sizeof(int));

    /* Add X position embeddings (first plane) */
    {
        void *args[] = { &d_tokens, &d_pos_embd, &d_pos_x, &dim, &n_patches };
        cuLaunchKernel(r->fn_vision_add_pos_embd, n_patches, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }
    /* Add Y position embeddings (second plane, offset by n_pos*dim) */
    {
        CUdeviceptr d_pos_y_table = d_pos_embd + (size_t)n_pos * dim * sizeof(uint16_t);
        void *args[] = { &d_tokens, &d_pos_y_table, &d_pos_y, &dim, &n_patches };
        cuLaunchKernel(r->fn_vision_add_pos_embd, n_patches, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }

    /* --- 3. Upload per-block weights --- */
    vis_gpu_block *gblocks = (vis_gpu_block *)calloc(n_blocks, sizeof(vis_gpu_block));
    for (int b = 0; b < n_blocks; b++) {
        char name[128];
        qtensor t;
        #define VIS_LOAD(field, suffix) \
            snprintf(name, sizeof(name), "v.blk.%d." suffix ".weight", b); \
            t = cllm_load_tensor(mmproj_gguf, name, 1); \
            gblocks[b].field = vis_upload_f16(&t);
        #define VIS_LOAD_NORM(field, suffix, sz) \
            snprintf(name, sizeof(name), "v.blk.%d." suffix ".weight", b); \
            t = cllm_load_tensor(mmproj_gguf, name, 1); \
            gblocks[b].field = vis_upload_norm_f16(&t, sz);

        VIS_LOAD(qw, "attn_q")
        VIS_LOAD(kw, "attn_k")
        VIS_LOAD(vw, "attn_v")
        VIS_LOAD(outw, "attn_out")
        VIS_LOAD_NORM(q_norm, "attn_q_norm", head_dim)
        VIS_LOAD_NORM(k_norm, "attn_k_norm", head_dim)
        VIS_LOAD_NORM(attn_post_norm, "attn_post_norm", dim)
        VIS_LOAD(gate_w, "ffn_gate")
        VIS_LOAD(up_w, "ffn_up")
        VIS_LOAD(down_w, "ffn_down")
        VIS_LOAD_NORM(ffn_post_norm, "ffn_post_norm", dim)
        VIS_LOAD_NORM(ln1_w, "ln1", dim)
        VIS_LOAD_NORM(ln2_w, "ln2", dim)
        #undef VIS_LOAD
        #undef VIS_LOAD_NORM
    }

    /* Upload projection weight */
    qtensor mm_proj_w = cllm_load_tensor(mmproj_gguf, "mm.input_projection.weight", 1);
    CUdeviceptr d_proj_w = vis_upload_f16(&mm_proj_w);

    /* Allocate work buffers */
    CUdeviceptr d_qkv, d_out, d_ffn_gate, d_ffn_up;
    size_t tok_buf = (size_t)n_patches * dim * sizeof(float);
    cuMemAlloc(&d_qkv, tok_buf * 3);  /* Q, K, V concatenated */
    cuMemAlloc(&d_out, tok_buf);
    cuMemAlloc(&d_ffn_gate, (size_t)n_patches * ffn_dim * sizeof(float));
    cuMemAlloc(&d_ffn_up, (size_t)n_patches * ffn_dim * sizeof(float));

    CUdeviceptr d_Q = d_qkv;
    CUdeviceptr d_K = d_qkv + tok_buf;
    CUdeviceptr d_V = d_qkv + tok_buf * 2;

    if (r->verbose >= 1) fprintf(stderr, "cuda_vision: weights uploaded, running %d blocks...\n", n_blocks);

    /* --- 4. Transformer blocks --- */
    for (int b = 0; b < n_blocks; b++) {
        vis_gpu_block *gb = &gblocks[b];

        /* Pre-attention RMSNorm → d_out */
        cuMemcpyDtoDAsync(d_out, d_tokens, tok_buf, r->stream);
        { void *a[] = { &d_out, &gb->ln1_w, &dim, &ln_eps, &n_patches };
          cuLaunchKernel(r->fn_vision_rmsnorm, n_patches, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Q/K/V projections */
        { void *a[] = { &d_Q, &gb->qw, &d_out, &dim, &dim, &n_patches };
          cuLaunchKernel(r->fn_vision_linear_f16, dim, n_patches, 1, 256, 1, 1, 0, r->stream, a, NULL); }
        { void *a[] = { &d_K, &gb->kw, &d_out, &dim, &dim, &n_patches };
          cuLaunchKernel(r->fn_vision_linear_f16, dim, n_patches, 1, 256, 1, 1, 0, r->stream, a, NULL); }
        { void *a[] = { &d_V, &gb->vw, &d_out, &dim, &dim, &n_patches };
          cuLaunchKernel(r->fn_vision_linear_f16, dim, n_patches, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Per-head Q/K RMSNorm */
        int hd_pow2 = 1; while (hd_pow2 < head_dim) hd_pow2 <<= 1;
        { void *a[] = { &d_Q, &gb->q_norm, &n_patches, &n_heads, &head_dim, &dim, &ln_eps };
          cuLaunchKernel(r->fn_vision_head_rmsnorm, n_patches, n_heads, 1, hd_pow2, 1, 1, 0, r->stream, a, NULL); }
        { void *a[] = { &d_K, &gb->k_norm, &n_patches, &n_heads, &head_dim, &dim, &ln_eps };
          cuLaunchKernel(r->fn_vision_head_rmsnorm, n_patches, n_heads, 1, hd_pow2, 1, 1, 0, r->stream, a, NULL); }

        /* 2D RoPE */
        float theta = 100.0f;
        int quarter = head_dim / 4;
        { void *a[] = { &d_Q, &d_pos_x, &d_pos_y, &n_patches, &n_heads, &head_dim, &dim, &theta };
          cuLaunchKernel(r->fn_vision_rope_2d, n_patches, n_heads, 1, quarter, 1, 1, 0, r->stream, a, NULL); }
        { void *a[] = { &d_K, &d_pos_x, &d_pos_y, &n_patches, &n_heads, &head_dim, &dim, &theta };
          cuLaunchKernel(r->fn_vision_rope_2d, n_patches, n_heads, 1, quarter, 1, 1, 0, r->stream, a, NULL); }

        /* Full NxN attention */
        size_t attn_smem = (size_t)n_patches * sizeof(float);
        { void *a[] = { &d_out, &d_Q, &d_K, &d_V, &n_patches, &n_heads, &head_dim, &dim };
          cuLaunchKernel(r->fn_vision_attention, n_heads, n_patches, 1, 256, 1, 1, attn_smem, r->stream, a, NULL); }

        /* Output projection */
        { void *a[] = { &d_Q, &gb->outw, &d_out, &dim, &dim, &n_patches }; /* reuse d_Q as temp */
          cuLaunchKernel(r->fn_vision_linear_f16, dim, n_patches, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Post-attention RMSNorm */
        { void *a[] = { &d_Q, &gb->attn_post_norm, &dim, &ln_eps, &n_patches };
          cuLaunchKernel(r->fn_vision_rmsnorm, n_patches, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Residual add: tokens += attn_output */
        int total_elem = n_patches * dim;
        { void *a[] = { &d_tokens, &d_Q, &total_elem };
          cuLaunchKernel(r->fn_vision_add, (total_elem + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Pre-FFN RMSNorm → d_out */
        cuMemcpyDtoDAsync(d_out, d_tokens, tok_buf, r->stream);
        { void *a[] = { &d_out, &gb->ln2_w, &dim, &ln_eps, &n_patches };
          cuLaunchKernel(r->fn_vision_rmsnorm, n_patches, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* FFN: gate + up projections */
        { void *a[] = { &d_ffn_gate, &gb->gate_w, &d_out, &ffn_dim, &dim, &n_patches };
          cuLaunchKernel(r->fn_vision_linear_f16, ffn_dim, n_patches, 1, 256, 1, 1, 0, r->stream, a, NULL); }
        { void *a[] = { &d_ffn_up, &gb->up_w, &d_out, &ffn_dim, &dim, &n_patches };
          cuLaunchKernel(r->fn_vision_linear_f16, ffn_dim, n_patches, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* GELU gate */
        int ffn_total = n_patches * ffn_dim;
        { void *a[] = { &d_ffn_gate, &d_ffn_up, &ffn_total };
          cuLaunchKernel(r->fn_vision_gelu_gate, (ffn_total + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Down projection → d_out */
        { void *a[] = { &d_out, &gb->down_w, &d_ffn_gate, &dim, &ffn_dim, &n_patches };
          cuLaunchKernel(r->fn_vision_linear_f16, dim, n_patches, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Post-FFN RMSNorm */
        { void *a[] = { &d_out, &gb->ffn_post_norm, &dim, &ln_eps, &n_patches };
          cuLaunchKernel(r->fn_vision_rmsnorm, n_patches, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

        /* Residual add: tokens += ffn_output */
        { void *a[] = { &d_tokens, &d_out, &total_elem };
          cuLaunchKernel(r->fn_vision_add, (total_elem + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

    }

    /* --- 5. Average pooling: [n_patches, dim] → [n_merged, dim] --- */
    cuStreamSynchronize(r->stream);
    float *h_tokens = (float *)malloc(tok_buf);
    cuMemcpyDtoH(h_tokens, d_tokens, tok_buf);

    float *pooled = (float *)calloc((size_t)n_merged * dim, sizeof(float));
    {
        int k = spatial_merge;
        int out_h = ph / k, out_w = pw / k;
        float inv_k2 = 1.0f / (k * k);
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float *dst = pooled + (oy * out_w + ox) * dim;
                for (int dy = 0; dy < k; dy++) {
                    for (int dx = 0; dx < k; dx++) {
                        int iy = oy * k + dy, ix = ox * k + dx;
                        if (iy < ph && ix < pw) {
                            const float *src = h_tokens + (iy * pw + ix) * dim;
                            for (int d = 0; d < dim; d++) dst[d] += src[d];
                        }
                    }
                }
                for (int d = 0; d < dim; d++) dst[d] *= inv_k2;
            }
        }
    }
    free(h_tokens);

    /* --- 6. Scale + Projection + Final norm on GPU --- */
    CUdeviceptr d_pooled;
    cuMemAlloc(&d_pooled, (size_t)n_merged * dim * sizeof(float));
    cuMemcpyHtoD(d_pooled, pooled, (size_t)n_merged * dim * sizeof(float));
    free(pooled);

    /* Scale by sqrt(dim) */
    float scale = sqrtf((float)dim);
    int pool_total = n_merged * dim;
    { void *a[] = { &d_pooled, &scale, &pool_total };
      cuLaunchKernel(r->fn_vision_scale, (pool_total + 255) / 256, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

    /* MM projection: [n_merged, dim] → [n_merged, proj_dim] */
    CUdeviceptr d_projected;
    cuMemAlloc(&d_projected, (size_t)n_merged * proj_dim * sizeof(float));
    { void *a[] = { &d_projected, &d_proj_w, &d_pooled, &proj_dim, &dim, &n_merged };
      cuLaunchKernel(r->fn_vision_linear_f16, proj_dim, n_merged, 1, 256, 1, 1, 0, r->stream, a, NULL); }

    /* Final RMSNorm (no learned weights) */
    CUdeviceptr null_weight = 0;
    { void *a[] = { &d_projected, &null_weight, &proj_dim, &ln_eps, &n_merged };
      cuLaunchKernel(r->fn_vision_rmsnorm, n_merged, 1, 1, 256, 1, 1, 0, r->stream, a, NULL); }

    /* --- 7. Download result --- */
    cuStreamSynchronize(r->stream);
    float *result = (float *)malloc((size_t)n_merged * proj_dim * sizeof(float));
    cuMemcpyDtoH(result, d_projected, (size_t)n_merged * proj_dim * sizeof(float));

    *out_tokens = n_merged;
    *out_dim = proj_dim;

    /* Cleanup GPU resources */
    cuMemFree(d_tokens); cuMemFree(d_pos_embd);
    cuMemFree(d_pos_x); cuMemFree(d_pos_y);
    cuMemFree(d_qkv); cuMemFree(d_out);
    cuMemFree(d_ffn_gate); cuMemFree(d_ffn_up);
    cuMemFree(d_pooled); cuMemFree(d_projected);
    cuMemFree(d_proj_w);
    for (int b = 0; b < n_blocks; b++) {
        vis_gpu_block *gb = &gblocks[b];
        cuMemFree(gb->qw); cuMemFree(gb->kw); cuMemFree(gb->vw); cuMemFree(gb->outw);
        cuMemFree(gb->q_norm); cuMemFree(gb->k_norm); cuMemFree(gb->attn_post_norm);
        cuMemFree(gb->gate_w); cuMemFree(gb->up_w); cuMemFree(gb->down_w);
        cuMemFree(gb->ffn_post_norm); cuMemFree(gb->ln1_w); cuMemFree(gb->ln2_w);
    }
    free(gblocks);
    free(pos_x); free(pos_y);

    return result;
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
    /* Gemma4 scratch */
    if (r->d_rope_inv_freq_full) cuMemFree(r->d_rope_inv_freq_full);
    if (r->d_rope_inv_freq_swa) cuMemFree(r->d_rope_inv_freq_swa);
    if (r->d_ple_combined) cuMemFree(r->d_ple_combined);
    if (r->d_ple_buf) cuMemFree(r->d_ple_buf);
    if (r->d_ple_proj) cuMemFree(r->d_ple_proj);
    free(r->swa_pattern);
    if (r->d_router_logits) { cuMemFree(r->d_router_logits); r->d_router_logits_entries = 0; }
    if (r->d_moe_accum)    cuMemFree(r->d_moe_accum);
    if (r->d_moe_f16w)     cuMemFree(r->d_moe_f16w);
    if (r->d_moe_f16w2)    cuMemFree(r->d_moe_f16w2);
    if (r->d_moe_f16w3)    cuMemFree(r->d_moe_f16w3);
    if (r->d_mmq_cxq8)     cuMemFree(r->d_mmq_cxq8);
    if (r->d_mmq_cxs)      cuMemFree(r->d_mmq_cxs);
    if (r->d_mmq_outg)     cuMemFree(r->d_mmq_outg);
    if (r->d_mmq_outu)     cuMemFree(r->d_mmq_outu);
    if (r->d_mmq_outd)     cuMemFree(r->d_mmq_outd);
    if (r->d_mmq_ebounds)  cuMemFree(r->d_mmq_ebounds);
    if (r->d_mmq_worklist) cuMemFree(r->d_mmq_worklist);
    if (r->d_pos_seq)    cuMemFree(r->d_pos_seq);
    if (r->d_graph_exec) cuGraphExecDestroy(r->d_graph_exec);
    if (r->d_graph)      cuGraphDestroy(r->d_graph);
    if (r->d_topk_idx)    cuMemFree(r->d_topk_idx);
    if (r->d_topk_wgt)    cuMemFree(r->d_topk_wgt);
    if (r->d_grid_ksigns) cuMemFree(r->d_grid_ksigns);
    if (r->d_grid_iq2s)   cuMemFree(r->d_grid_iq2s);
    if (r->d_grid_iq3s)   cuMemFree(r->d_grid_iq3s);
    if (r->d_grid_iq3)    cuMemFree(r->d_grid_iq3);
    if (r->moe_gpu_mod)   cuModuleUnload(r->moe_gpu_mod);
    if (r->d_hidden_snapshots) cuMemFree(r->d_hidden_snapshots);
    free(r->h_router_logits);
    free(r->h_stage);
    if (r->d_xb_q)     cuMemFree(r->d_xb_q);
    if (r->d_xb_scale)  cuMemFree(r->d_xb_scale);
    if (r->d_xb_q2)    cuMemFree(r->d_xb_q2);
    if (r->d_xb_scale2) cuMemFree(r->d_xb_scale2);
    if (r->d_xb_q81)    cuMemFree(r->d_xb_q81);
    if (r->d_xb_q81_2)  cuMemFree(r->d_xb_q81_2);

    /* Free KV cache (skip shared caches for Gemma4) */
    if (r->d_key_cache) {
        for (int l = 0; l < r->n_layers; l++) {
            if (r->is_gemma4 && r->layers && r->layers[l].shared_kv_source >= 0) continue;
            if (r->d_key_cache[l]) cuMemFree(r->d_key_cache[l]);
        }
        free(r->d_key_cache);
    }
    if (r->d_value_cache) {
        for (int l = 0; l < r->n_layers; l++) {
            if (r->is_gemma4 && r->layers && r->layers[l].shared_kv_source >= 0) continue;
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
            if (cl->attn_q_w_f16)  cuMemFree(cl->attn_q_w_f16);
            if (cl->attn_k_w)      cuMemFree(cl->attn_k_w);
            if (cl->attn_k_w_f16)  cuMemFree(cl->attn_k_w_f16);
            if (cl->attn_v_w)      cuMemFree(cl->attn_v_w);
            if (cl->attn_v_w_f16)  cuMemFree(cl->attn_v_w_f16);
            if (cl->attn_q_norm_w) cuMemFree(cl->attn_q_norm_w);
            if (cl->attn_k_norm_w) cuMemFree(cl->attn_k_norm_w);
            if (cl->attn_q_bias)   cuMemFree(cl->attn_q_bias);
            if (cl->attn_k_bias)   cuMemFree(cl->attn_k_bias);
            if (cl->attn_v_bias)   cuMemFree(cl->attn_v_bias);
            if (cl->attn_output_w) cuMemFree(cl->attn_output_w);
            if (cl->attn_output_w_f16) cuMemFree(cl->attn_output_w_f16);
            if (cl->ffn_norm_w)    cuMemFree(cl->ffn_norm_w);
            if (cl->ffn_gate_w)    cuMemFree(cl->ffn_gate_w);
            if (cl->ffn_gate_w_f16) cuMemFree(cl->ffn_gate_w_f16);
            if (cl->ffn_up_w)     cuMemFree(cl->ffn_up_w);
            if (cl->ffn_up_w_f16) cuMemFree(cl->ffn_up_w_f16);
            if (cl->ffn_down_w)   cuMemFree(cl->ffn_down_w);
            if (cl->ffn_down_w_f16) cuMemFree(cl->ffn_down_w_f16);
            /* SSM weights + state */
            if (cl->ssm_qkv_w)      cuMemFree(cl->ssm_qkv_w);
            if (cl->ssm_qkv_w_f16)  cuMemFree(cl->ssm_qkv_w_f16);
            if (cl->ssm_gate_w)     cuMemFree(cl->ssm_gate_w);
            if (cl->ssm_gate_w_f16) cuMemFree(cl->ssm_gate_w_f16);
            if (cl->ssm_alpha_w)    cuMemFree(cl->ssm_alpha_w);
            if (cl->ssm_beta_w)     cuMemFree(cl->ssm_beta_w);
            if (cl->ssm_out_w)      cuMemFree(cl->ssm_out_w);
            if (cl->ssm_out_w_f16)  cuMemFree(cl->ssm_out_w_f16);
            if (cl->ssm_a)          cuMemFree(cl->ssm_a);
            if (cl->ssm_dt_bias)    cuMemFree(cl->ssm_dt_bias);
            if (cl->ssm_conv1d_w)   cuMemFree(cl->ssm_conv1d_w);
            if (cl->ssm_norm_w)     cuMemFree(cl->ssm_norm_w);
            if (cl->d_conv_state)      cuMemFree(cl->d_conv_state);
            if (cl->d_recurrent_state) cuMemFree(cl->d_recurrent_state);
            /* Gemma4 per-layer weights */
            if (cl->post_attn_norm_w) cuMemFree(cl->post_attn_norm_w);
            if (cl->post_ffw_norm_w)  cuMemFree(cl->post_ffw_norm_w);
            if (cl->ple_inp_gate_w)   cuMemFree(cl->ple_inp_gate_w);
            if (cl->ple_proj_w)       cuMemFree(cl->ple_proj_w);
            if (cl->ple_post_norm_w)  cuMemFree(cl->ple_post_norm_w);
            /* MoE weights */
            if (cl->moe_gate_w)             cuMemFree(cl->moe_gate_w);
            if (cl->moe_gate_exps_w)        cuMemFree(cl->moe_gate_exps_w);
            if (cl->moe_gate_exps_w_f16)    cuMemFree(cl->moe_gate_exps_w_f16);
            if (cl->moe_up_exps_w)          cuMemFree(cl->moe_up_exps_w);
            if (cl->moe_up_exps_w_f16)      cuMemFree(cl->moe_up_exps_w_f16);
            if (cl->moe_down_exps_w)        cuMemFree(cl->moe_down_exps_w);
            if (cl->moe_down_exps_w_f16)    cuMemFree(cl->moe_down_exps_w_f16);
            if (cl->moe_shared_gate_w)      cuMemFree(cl->moe_shared_gate_w);
            if (cl->moe_shared_ffn_gate_w)  cuMemFree(cl->moe_shared_ffn_gate_w);
            if (cl->moe_shared_ffn_gate_w_f16) cuMemFree(cl->moe_shared_ffn_gate_w_f16);
            if (cl->moe_shared_ffn_up_w)    cuMemFree(cl->moe_shared_ffn_up_w);
            if (cl->moe_shared_ffn_up_w_f16) cuMemFree(cl->moe_shared_ffn_up_w_f16);
            if (cl->moe_shared_ffn_down_w)  cuMemFree(cl->moe_shared_ffn_down_w);
            if (cl->moe_shared_ffn_down_w_f16) cuMemFree(cl->moe_shared_ffn_down_w_f16);
        }
        free(r->layers);
    }

    /* Free global weights */
    if (r->d_token_embd) cuMemFree(r->d_token_embd);
    if (r->d_output_norm) cuMemFree(r->d_output_norm);
    if (r->d_output_w && r->d_output_w != r->d_token_embd) cuMemFree(r->d_output_w);
    if (r->d_logits) cuMemFree(r->d_logits);

    /* Free batch prefill buffers */
    if (r->d_batch_x)     cuMemFree(r->d_batch_x);
    if (r->d_batch_xb)    cuMemFree(r->d_batch_xb);
    if (r->d_batch_wide)  cuMemFree(r->d_batch_wide);
    if (r->d_batch_q)     cuMemFree(r->d_batch_q);
    if (r->d_batch_mid)   cuMemFree(r->d_batch_mid);
    if (r->d_batch_k)     cuMemFree(r->d_batch_k);
    if (r->d_batch_v)     cuMemFree(r->d_batch_v);
    if (r->d_batch_ff1)   cuMemFree(r->d_batch_ff1);
    if (r->d_batch_ff2)   cuMemFree(r->d_batch_ff2);
    if (r->d_batch_alpha) cuMemFree(r->d_batch_alpha);
    if (r->d_batch_beta)  cuMemFree(r->d_batch_beta);
    if (r->d_batch_token_ids) cuMemFree(r->d_batch_token_ids);
    if (r->d_batch_f16_scratch) cuMemFree(r->d_batch_f16_scratch);
    if (r->d_fa2_q_f16) cuMemFree(r->d_fa2_q_f16);
    if (r->d_fa2_o_f16) cuMemFree(r->d_fa2_o_f16);
    if (r->d_fa2_scores_f16) cuMemFree(r->d_fa2_scores_f16);
    if (r->d_swa_k_lin) cuMemFree(r->d_swa_k_lin);
    if (r->d_swa_v_lin) cuMemFree(r->d_swa_v_lin);
    if (r->d_f16_scratch2) cuMemFree(r->d_f16_scratch2);
    if (r->stream_dq) cuStreamDestroy(r->stream_dq);
    for (int i = 0; i < 2; i++) {
        if (r->dq_evt[i]) cuEventDestroy(r->dq_evt[i]);
        if (r->gemm_evt[i]) cuEventDestroy(r->gemm_evt[i]);
    }
    if (r->d_f16_scratch) cuMemFree(r->d_f16_scratch);

    /* Free modules */
    if (r->module) cuModuleUnload(r->module);
    if (r->fa2_mod) cuModuleUnload(r->fa2_mod);
    if (r->cublas) cublasewDestroy(r->cublas);

    /* Free host buffer */
    free(r->h_output);

    /* Destroy CUDA objects (keep primary context alive for other runners) */
    if (r->stream) cuStreamDestroy(r->stream);

    free(r);
}

/* ======================================================================== */
/* Public API: accessors                                                    */
/* ======================================================================== */

int cuda_llm_reset_state(cuda_llm_runner *r) {
    if (!r) return -1;
    if (cuda_llm_bind_context(r) != 0) return -1;

    int kv_dim = r->n_kv_heads * r->head_dim;
    if (r->d_key_cache && r->d_value_cache && r->max_seq_len > 0) {
        if (r->is_gemma4) {
            /* Invariant: shared_kv_source layers must alias the source's KV cache
             * pointer. If broken, the loop below would either (a) double-zero a
             * cache another layer reads from, or (b) leak unzeroed state into a
             * shared layer. Bail loudly rather than silently corrupt. */
            if (r->layers) {
                for (int l = 0; l < r->n_layers; l++) {
                    int src = r->layers[l].shared_kv_source;
                    if (src < 0) continue;
                    if (src >= r->n_layers ||
                        r->d_key_cache[l]   != r->d_key_cache[src] ||
                        r->d_value_cache[l] != r->d_value_cache[src]) {
                        fprintf(stderr,
                                "cuda_llm: FATAL gemma4 KV-share invariant broken at layer %d "
                                "(src=%d, key %llu vs %llu, val %llu vs %llu); reset aborted\n",
                                l, src,
                                (unsigned long long)r->d_key_cache[l],
                                (unsigned long long)(src >= 0 && src < r->n_layers ? r->d_key_cache[src] : 0),
                                (unsigned long long)r->d_value_cache[l],
                                (unsigned long long)(src >= 0 && src < r->n_layers ? r->d_value_cache[src] : 0));
                        return -1;
                    }
                }
            }
            for (int l = 0; l < r->n_layers; l++) {
                if (r->layers && r->layers[l].shared_kv_source >= 0) continue;
                int layer_kv_heads = r->layers ? r->layers[l].n_kv_heads : r->n_kv_heads;
                int hd = r->layers && r->layers[l].is_swa ? r->head_dim_swa : r->head_dim_full;
                int layer_kv_dim = layer_kv_heads * hd;
                int cache_len = (r->layers && r->layers[l].is_swa) ? r->swa_window_size : r->max_seq_len;
                size_t kv_cache_bytes = (size_t)cache_len * layer_kv_dim * sizeof(uint16_t);
                cu_async_zero(r->d_key_cache[l],   kv_cache_bytes, r->stream, "cuda_llm key_cache");
                cu_async_zero(r->d_value_cache[l], kv_cache_bytes, r->stream, "cuda_llm value_cache");
            }
        } else if (kv_dim > 0) {
            size_t kv_cache_bytes = (size_t)r->max_seq_len * kv_dim * sizeof(uint16_t);
            for (int l = 0; l < r->n_layers; l++) {
                cu_async_zero(r->d_key_cache[l],   kv_cache_bytes, r->stream, "cuda_llm key_cache");
                cu_async_zero(r->d_value_cache[l], kv_cache_bytes, r->stream, "cuda_llm value_cache");
            }
        }
    }
    if (r->d_hidden_snapshots && r->n_hidden_snapshots > 0 && r->n_embd > 0) {
        size_t snap_bytes = (size_t)r->n_hidden_snapshots * r->n_embd * sizeof(float);
        cu_async_zero(r->d_hidden_snapshots, snap_bytes, r->stream, "cuda_llm hidden_snapshots");
    }

    {
        int q_dim = r->n_heads * r->head_dim;
        int max_dim = r->n_embd;
        if (q_dim > max_dim) max_dim = q_dim;
        if (r->n_ff > max_dim) max_dim = r->n_ff;
        int xb2_dim = max_dim;
        if (r->is_hybrid) {
            if (r->ssm_qkv_dim > xb2_dim) xb2_dim = r->ssm_qkv_dim;
            if (2 * q_dim > xb2_dim) xb2_dim = 2 * q_dim;
        }
        if (r->is_moe) {
            int tc_down_dim = r->n_experts_used * r->n_embd;
            if (tc_down_dim > xb2_dim) xb2_dim = tc_down_dim;
        }
        int gate_dim = r->n_ff;
        if (r->is_moe && r->shared_expert_ff > gate_dim) gate_dim = r->shared_expert_ff;
        if (r->is_moe) {
            int tc_dim = r->n_experts_used * r->expert_ff;
            if (tc_dim > gate_dim) gate_dim = tc_dim;
        }

        cu_async_zero(r->d_x,   (size_t)max_dim * sizeof(float), r->stream, "cuda_llm x");
        cu_async_zero(r->d_xb,  (size_t)max_dim * sizeof(float), r->stream, "cuda_llm xb");
        cu_async_zero(r->d_xb2, (size_t)xb2_dim * sizeof(float), r->stream, "cuda_llm xb2");
        cu_async_zero(r->d_q,   (size_t)q_dim * sizeof(float), r->stream, "cuda_llm q");
        cu_async_zero(r->d_k,   (size_t)kv_dim * sizeof(float), r->stream, "cuda_llm k");
        cu_async_zero(r->d_v,   (size_t)kv_dim * sizeof(float), r->stream, "cuda_llm v");
        cu_async_zero(r->d_gate, (size_t)gate_dim * sizeof(float), r->stream, "cuda_llm gate");
        cu_async_zero(r->d_up,   (size_t)gate_dim * sizeof(float), r->stream, "cuda_llm up");
        cu_async_zero(r->d_xb_q,  (size_t)max_dim * sizeof(int8_t), r->stream, "cuda_llm xb_q");
        cu_async_zero(r->d_xb_q2, (size_t)max_dim * sizeof(int8_t), r->stream, "cuda_llm xb_q2");
        cu_async_zero(r->d_xb_scale,  sizeof(float), r->stream, "cuda_llm xb_scale");
        cu_async_zero(r->d_xb_scale2, sizeof(float), r->stream, "cuda_llm xb_scale2");
        {
            size_t q81_bytes = ((size_t)max_dim / 32 + 1) * 36;
            cu_async_zero(r->d_xb_q81, q81_bytes, r->stream, "cuda_llm xb_q81");
            cu_async_zero(r->d_xb_q81_2, q81_bytes, r->stream, "cuda_llm xb_q81_2");
        }
        if (r->is_moe) {
            cu_async_zero(r->d_router_logits,
                          (size_t)r->d_router_logits_entries * sizeof(float),
                          r->stream, "cuda_llm router_logits");
            cu_async_zero(r->d_moe_accum,
                          (size_t)r->n_embd * sizeof(float),
                          r->stream, "cuda_llm moe_accum");
            cu_async_zero(r->d_moe_f16w,
                          (size_t)r->expert_ff * r->n_embd * sizeof(uint16_t),
                          r->stream, "cuda_llm moe_f16w");
            cu_async_zero(r->d_moe_f16w2,
                          (size_t)r->expert_ff * r->n_embd * sizeof(uint16_t),
                          r->stream, "cuda_llm moe_f16w2");
            cu_async_zero(r->d_moe_f16w3,
                          (size_t)r->expert_ff * r->n_embd * sizeof(uint16_t),
                          r->stream, "cuda_llm moe_f16w3");
        }
    }

    if (r->is_hybrid) {
        cu_async_zero(r->d_ssm_qkv, (size_t)r->ssm_qkv_dim * sizeof(float),
                      r->stream, "cuda_llm ssm_qkv");
        cu_async_zero(r->d_ssm_z, (size_t)r->ssm_d_inner * sizeof(float),
                      r->stream, "cuda_llm ssm_z");
        cu_async_zero(r->d_ssm_alpha, (size_t)r->ssm_dt_rank * sizeof(float),
                      r->stream, "cuda_llm ssm_alpha");
        cu_async_zero(r->d_ssm_beta, (size_t)r->ssm_dt_rank * sizeof(float),
                      r->stream, "cuda_llm ssm_beta");
        cu_async_zero(r->d_ssm_Q_exp, (size_t)r->ssm_dt_rank * r->ssm_d_state * sizeof(float),
                      r->stream, "cuda_llm ssm_Q_exp");
        cu_async_zero(r->d_ssm_K_exp, (size_t)r->ssm_dt_rank * r->ssm_d_state * sizeof(float),
                      r->stream, "cuda_llm ssm_K_exp");
        cu_async_zero(r->d_ssm_out, (size_t)r->ssm_d_inner * sizeof(float),
                      r->stream, "cuda_llm ssm_out");
        cu_async_zero(r->d_ssm_conv_out, (size_t)r->ssm_qkv_dim * sizeof(float),
                      r->stream, "cuda_llm ssm_conv_out");
        cu_async_zero(r->d_attn_gate, (size_t)(r->n_heads * r->head_dim) * sizeof(float),
                      r->stream, "cuda_llm attn_gate");
        for (int l = 0; l < r->n_layers; l++) {
            cuda_layer *cl = &r->layers[l];
            if (!cl->is_ssm) continue;
            if (cl->d_conv_state) {
                size_t conv_bytes = (size_t)(r->ssm_conv_kernel - 1) * r->ssm_qkv_dim * sizeof(float);
                cu_async_zero(cl->d_conv_state, conv_bytes, r->stream, "cuda_llm conv_state");
            }
            if (cl->d_recurrent_state) {
                size_t rec_bytes = (size_t)r->ssm_dt_rank * r->ssm_d_state * r->ssm_d_state * sizeof(float);
                cu_async_zero(cl->d_recurrent_state, rec_bytes, r->stream, "cuda_llm recurrent_state");
            }
        }
    }
    {
        CUresult err = cuStreamSynchronize(r->stream);
        if (err != CUDA_SUCCESS) {
            const char *es = "?";
            cuGetErrorString(err, &es);
            fprintf(stderr, "cuda_llm: reset synchronize failed: %s (%d)\n",
                    es, (int)err);
            return -1;
        }
    }
    return 0;
}

int cuda_llm_inject_biases(cuda_llm_runner *r, const char *safetensors_path) {
    if (!r || !safetensors_path) return 0;

    st_context *st = safetensors_open(safetensors_path);
    if (!st) {
        fprintf(stderr, "cuda_llm: cannot open safetensors for biases: %s\n", safetensors_path);
        return 0;
    }

    int n_loaded = 0;
    for (int l = 0; l < r->n_layers; l++) {
        cuda_layer *cl = &r->layers[l];
        if (cl->is_ssm) continue;  /* SSM layers don't have attention biases */

        char wn[256];
        for (int qi = 0; qi < 3; qi++) {
            const char *proj = (qi == 0) ? "q" : (qi == 1) ? "k" : "v";
            snprintf(wn, sizeof(wn), "model.layers.%d.self_attn.%s_proj.bias", l, proj);
            int idx = safetensors_find(st, wn);
            if (idx < 0) continue;

            const uint64_t *shape = safetensors_shape(st, idx);
            int n = (int)shape[0];
            const char *dtype = safetensors_dtype(st, idx);
            const void *raw = safetensors_data(st, idx);

            /* Dequant to F32 */
            float *f32 = (float *)malloc((size_t)n * sizeof(float));
            if (strcmp(dtype, "BF16") == 0) {
                const uint16_t *bf = (const uint16_t *)raw;
                for (int i = 0; i < n; i++) {
                    uint32_t bits = (uint32_t)bf[i] << 16;
                    memcpy(&f32[i], &bits, 4);
                }
            } else if (strcmp(dtype, "F32") == 0) {
                memcpy(f32, raw, (size_t)n * sizeof(float));
            } else if (strcmp(dtype, "F16") == 0) {
                /* F16 to F32 — simple conversion */
                const uint16_t *fp16 = (const uint16_t *)raw;
                for (int i = 0; i < n; i++) {
                    uint32_t h = fp16[i];
                    uint32_t sign = (h >> 15) & 1;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    uint32_t f;
                    if (exp == 0) f = sign << 31;
                    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
                    else f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
                    memcpy(&f32[i], &f, 4);
                }
            }

            CUdeviceptr *dst = (qi == 0) ? &cl->attn_q_bias :
                               (qi == 1) ? &cl->attn_k_bias : &cl->attn_v_bias;
            size_t sz = (size_t)n * sizeof(float);
            if (*dst) cuMemFree(*dst);  /* free existing if any */
            cuMemAlloc(dst, sz);
            cuMemcpyHtoD(*dst, f32, sz);
            free(f32);
            n_loaded++;
        }
    }

    safetensors_close(st);
    if (r->verbose >= 1)
        fprintf(stderr, "cuda_llm: injected %d Q/K/V biases from %s\n", n_loaded, safetensors_path);
    return n_loaded;
}

int cuda_llm_n_embd(const cuda_llm_runner *r) { return r ? r->n_embd : 0; }
int cuda_llm_n_layers(const cuda_llm_runner *r) { return r ? r->n_layers : 0; }
int cuda_llm_n_vocab(const cuda_llm_runner *r) { return r ? r->n_vocab : 0; }
int cuda_llm_max_seq_len(const cuda_llm_runner *r) { return r ? r->max_seq_len : 0; }
int cuda_llm_uses_dp4a(const cuda_llm_runner *r) { return r ? r->use_dp4a : 0; }

/* Toggle the INT8 dp4a matvec path at runtime. Returns the previous setting.
 * Used by the test harness to take an F32 sequential reference (dp4a off) for a
 * meaningful batched-prefill comparison: batched is F16, so vs an F32 oracle it
 * is ~3e-3 (pure F16 precision); vs the dp4a path it is ~0.2-0.7 (int8 activation
 * quant divergence, NOT a batched bug). No effect if dp4a was never enabled. */
int cuda_llm_set_dp4a(cuda_llm_runner *r, int enable) {
    if (!r) return 0;
    int prev = r->use_dp4a;
    r->use_dp4a = enable ? 1 : 0;
    return prev;
}
