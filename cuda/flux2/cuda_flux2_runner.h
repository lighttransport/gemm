/*
 * cuda_flux2_runner.h - CUDA Flux.2 Klein text-to-image runner
 *
 * Uses NVRTC to compile kernels at runtime (no nvcc needed).
 * Runs the 5 double-stream + 20 single-stream DiT blocks on GPU.
 *
 * Weight format: FP8 E4M3 safetensors (dequantized to F16 on upload)
 *
 * Usage:
 *   cuda_flux2_runner *r = cuda_flux2_init(0, 1);
 *   cuda_flux2_load_dit(r, "flux-2-klein-4b.safetensors");
 *   cuda_flux2_load_vae(r, "flux2-vae.safetensors");
 *   cuda_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt, t, guidance, out);
 *   cuda_flux2_free(r);
 */
#ifndef CUDA_FLUX2_RUNNER_H
#define CUDA_FLUX2_RUNNER_H

#include <stdint.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_flux2_runner cuda_flux2_runner;

cuda_flux2_runner *cuda_flux2_init(int device_id, int verbose);
int  cuda_flux2_load_dit(cuda_flux2_runner *r, const char *safetensors_path);
int  cuda_flux2_load_vae(cuda_flux2_runner *r, const char *safetensors_path);
void cuda_flux2_free(cuda_flux2_runner *r);

/* Run single DiT denoising step on GPU.
 *   img_tokens: [n_img, patch_in_ch] patchified latent (CPU, F32)
 *   txt_tokens: [n_txt, txt_dim]    text hidden states (CPU, F32)
 *   timestep:   sigma × 1000
 *   guidance:   guidance scale (unused for distilled, pass 0)
 *   out:        [n_img, patch_in_ch] velocity (CPU, F32, pre-allocated)
 * Returns 0 on success. */
int cuda_flux2_dit_step(cuda_flux2_runner *r,
                        const float *img_tokens, int n_img,
                        const float *txt_tokens, int n_txt,
                        float timestep, float guidance, float *out);

/* VAE decode on GPU.
 *   latent:   [32, lat_h, lat_w] F32 (CPU)
 *   out_rgb:  [3, lat_h*8, lat_w*8] F32 (CPU, pre-allocated)
 * Returns 0 on success. */
int cuda_flux2_vae_decode(cuda_flux2_runner *r,
                          const float *latent, int lat_h, int lat_w,
                          float *out_rgb);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef CUDA_FLUX2_RUNNER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_kernels_common.h"
#include "../../common/safetensors.h"
#include "../../common/flux2_klein_dit.h"
#include "../../common/flux2_klein_vae.h"

/* ---- Flux.2 Klein CUDA kernels ---- */

static const char *flux2_kernel_src =
"\n/* ---- Flux.2 Klein specific kernels ---- */\n"

/* SwiGLU: [n_tok, 2*mlp_h] → [n_tok, mlp_h] */
"__global__ void flux2_swiglu(float *out, const float *in,\n"
"                              int n_tok, int mlp_h) {\n"
"    int tok = blockIdx.x, i = threadIdx.x + blockIdx.y * blockDim.x;\n"
"    if (tok >= n_tok || i >= mlp_h) return;\n"
"    float gate = in[(long)tok * 2 * mlp_h + i];\n"
"    float val  = in[(long)tok * 2 * mlp_h + mlp_h + i];\n"
"    out[(long)tok * mlp_h + i] = (gate / (1.0f + expf(-gate))) * val;\n"
"}\n"

/* Per-head RMSNorm: x[N, dim], w[head_dim], n_heads, head_dim */
"__global__ void rmsnorm_per_head_f32(float *__restrict__ x,\n"
"    const float *__restrict__ w, int N, int n_heads, int head_dim) {\n"
"    int tok = blockIdx.x, h = blockIdx.y;\n"
"    if (tok >= N || h >= n_heads) return;\n"
"    int dim = n_heads * head_dim;\n"
"    float *hd = x + (long)tok * dim + h * head_dim;\n"
"    float ss = 0.0f;\n"
"    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)\n"
"        ss += hd[i] * hd[i];\n"
"    for (int m = 16; m > 0; m >>= 1)\n"
"        ss += __shfl_xor_sync(0xFFFFFFFF, ss, m);\n"
"    float inv = rsqrtf(ss / (float)head_dim + 1e-6f);\n"
"    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)\n"
"        hd[i] *= inv * w[i];\n"
"}\n"

/* adaLN: out = LN(x) * (1 + scale) + shift. Grid: (N), Block: (256) */
"__global__ void adaln_modulate_f32(float *__restrict__ out,\n"
"    const float *__restrict__ x, const float *__restrict__ shift,\n"
"    const float *__restrict__ scale, int N, int dim) {\n"
"    int tok = blockIdx.x;\n"
"    if (tok >= N) return;\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    float s = 0;\n"
"    for (int i = tid; i < dim; i += blockDim.x) s += x[(long)tok*dim+i];\n"
"    sdata[tid] = s; __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)dim;\n"
"    __syncthreads();\n"
"    s = 0;\n"
"    for (int i = tid; i < dim; i += blockDim.x) { float d = x[(long)tok*dim+i] - mean; s += d*d; }\n"
"    sdata[tid] = s; __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)dim + 1e-6f);\n"
"    for (int i = tid; i < dim; i += blockDim.x)\n"
"        out[(long)tok*dim+i] = ((x[(long)tok*dim+i] - mean) * inv) * (1.0f + scale[i]) + shift[i];\n"
"}\n"

/* Gated residual: x += gate * proj. Gate broadcasts across tokens. */
"__global__ void gated_add_f32(float *__restrict__ x,\n"
"    const float *__restrict__ proj, const float *__restrict__ gate,\n"
"    int N, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= N * dim) return;\n"
"    x[i] += gate[i % dim] * proj[i];\n"
"}\n"

/* GroupNorm over channel groups on CHW tensor flattened as [C, spatial]. */
"__global__ void flux2_vae_groupnorm_f32(float *__restrict__ out,\n"
"    const float *__restrict__ in, const float *__restrict__ gamma,\n"
"    const float *__restrict__ beta, int C, int spatial, int num_groups) {\n"
"    int g = blockIdx.x;\n"
"    if (g >= num_groups) return;\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    int group_size = C / num_groups;\n"
"    int c0 = g * group_size;\n"
"    int n = group_size * spatial;\n"
"    float sum = 0.0f;\n"
"    for (int i = tid; i < n; i += blockDim.x) {\n"
"        int ch = c0 + i / spatial;\n"
"        int s = i % spatial;\n"
"        sum += in[(long)ch * spatial + s];\n"
"    }\n"
"    sdata[tid] = sum;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x / 2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid + r];\n"
"        __syncthreads();\n"
"    }\n"
"    float mean = sdata[0] / (float)n;\n"
"    __syncthreads();\n"
"    float var = 0.0f;\n"
"    for (int i = tid; i < n; i += blockDim.x) {\n"
"        int ch = c0 + i / spatial;\n"
"        int s = i % spatial;\n"
"        float d = in[(long)ch * spatial + s] - mean;\n"
"        var += d * d;\n"
"    }\n"
"    sdata[tid] = var;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x / 2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid + r];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv = rsqrtf(sdata[0] / (float)n + 1e-6f);\n"
"    for (int i = tid; i < n; i += blockDim.x) {\n"
"        int ch = c0 + i / spatial;\n"
"        int s = i % spatial;\n"
"        float v = (in[(long)ch * spatial + s] - mean) * inv;\n"
"        float gv = gamma ? gamma[ch] : 1.0f;\n"
"        float bv = beta ? beta[ch] : 0.0f;\n"
"        out[(long)ch * spatial + s] = v * gv + bv;\n"
"    }\n"
"}\n"

/* Conv2D on CHW tensor with same-padding semantics. */
"__global__ void flux2_vae_conv2d_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, const float *__restrict__ weight,\n"
"    const float *__restrict__ bias,\n"
"    int ci, int h, int w, int co, int kh, int kw, int pad) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = co * h * w;\n"
"    if (idx >= total) return;\n"
"    int oc = idx / (h * w);\n"
"    int rem = idx % (h * w);\n"
"    int oy = rem / w, ox = rem % w;\n"
"    int ph = pad, pw = pad;\n"
"    float sum = bias ? bias[oc] : 0.0f;\n"
"    for (int ic = 0; ic < ci; ic++) {\n"
"        for (int fy = 0; fy < kh; fy++) {\n"
"            int iy = oy + fy - ph;\n"
"            if (iy < 0 || iy >= h) continue;\n"
"            for (int fx = 0; fx < kw; fx++) {\n"
"                int ix = ox + fx - pw;\n"
"                if (ix < 0 || ix >= w) continue;\n"
"                sum += inp[(long)ic * h * w + iy * w + ix] *\n"
"                       weight[(((long)oc * ci + ic) * kh + fy) * kw + fx];\n"
"            }\n"
"        }\n"
"    }\n"
"    out[idx] = sum;\n"
"}\n"

/* ============================================================================
 * BF16 implicit-GEMM conv2d (3x3 stride=1 pad=1).
 * Input: F32 [ci, h, w]. Weight: BF16 [co, ci, 3, 3]. Output: F32 [co, h, w].
 * Uses mma.sync.m16n8k16.bf16.bf16.f32 (sm_80+, with sm_120 fragment swap).
 *
 * Implicit GEMM mapping:
 *   M = h*w (output spatial position, row-major)
 *   N = co
 *   K = ci * 9
 *
 * Block: 32 threads (1 warp). Grid: (ceil(N/8), ceil(M/16)).
 * Each warp computes a 16x8 output tile (one m16n8 N-tile).
 * ============================================================================ */
"#if __CUDA_ARCH__ >= 800\n"
"__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {\n"
"    unsigned int b; memcpy(&b, &f, 4);\n"
"    /* NaN -> 0x7FC0, otherwise round-to-nearest-even */\n"
"    if (((b >> 23) & 0xFF) == 0xFF && (b & 0x7FFFFF)) return 0x7FC0;\n"
"    unsigned int r = 0x7FFFu + ((b >> 16) & 1u);\n"
"    return (unsigned short)((b + r) >> 16);\n"
"}\n"
"__device__ __forceinline__ float bf16_to_f32(unsigned short h) {\n"
"    unsigned int b = ((unsigned int)h) << 16;\n"
"    float f; memcpy(&f, &b, 4); return f;\n"
"}\n"
"__device__ __forceinline__ float conv_in_at(\n"
"    const float *inp, int M, int K, int h, int w,\n"
"    int m_row, int k_col) {\n"
"    if (m_row >= M || k_col >= K) return 0.0f;\n"
"    int oy = m_row / w, ox = m_row - oy * w;\n"
"    int ic = k_col / 9, kf = k_col - ic * 9;\n"
"    int fy = kf / 3, fx = kf - fy * 3;\n"
"    int iy = oy + fy - 1, ix = ox + fx - 1;\n"
"    if (iy < 0 || iy >= h || ix < 0 || ix >= w) return 0.0f;\n"
"    return inp[((long)ic * h + iy) * w + ix];\n"
"}\n"

"#define CONV_NTILE 8\n"
"#define CONV_NWARPS 2\n"
"__global__ void conv2d_bf16_f32_3x3(\n"
"    float *__restrict__ out,\n"
"    const float *__restrict__ inp,         /* [ci, h, w] f32 */\n"
"    const unsigned short *__restrict__ wb, /* [co, ci, 9] bf16 */\n"
"    const float *__restrict__ bias,        /* [co] f32, may be NULL */\n"
"    int ci, int h, int w, int co) {\n"
"    int M = h * w;\n"
"    int K = ci * 9;\n"
"    int m_base = blockIdx.y * 16;\n"
"    int n_base = blockIdx.x * (8 * CONV_NTILE * CONV_NWARPS);\n"
"    int warp = threadIdx.x / 32;\n"
"    int lane = threadIdx.x & 31;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane & 3;\n"
"    int warp_n_base = n_base + warp * (8 * CONV_NTILE);\n"
"\n"
"    if (m_base >= M || n_base >= co) return;\n"
"\n"
"    float d0[CONV_NTILE], d1[CONV_NTILE], d2[CONV_NTILE], d3[CONV_NTILE];\n"
"#pragma unroll\n"
"    for (int i = 0; i < CONV_NTILE; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"\n"
"    /* Shared memory: F32 input tile [16 M rows × 16 K cols] + per-row (oy, ox) cache.\n"
"     * Stride padded to 17 to avoid 4-way bank conflicts when each warp lane reads\n"
"     * sm_in[gid*STRIDE + ...] (gid = lane>>2). */\n"
"    const int IN_STRIDE = 17;\n"
"    __shared__ float sm_in[16 * 17];\n"
"    __shared__ short sm_oy[16];\n"
"    __shared__ short sm_ox[16];\n"
"    if (threadIdx.x < 16) {\n"
"        int mr = threadIdx.x;\n"
"        int m_row = m_base + mr;\n"
"        if (m_row < M) {\n"
"            sm_oy[mr] = (short)(m_row / w);\n"
"            sm_ox[mr] = (short)(m_row - sm_oy[mr] * w);\n"
"        } else {\n"
"            sm_oy[mr] = -1;\n"
"            sm_ox[mr] = -1;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    for (int k_base = 0; k_base < K; k_base += 16) {\n"
"        /* Cooperative load of 16x16 F32 input tile into smem (stride 17 = padded). */\n"
"        for (int idx = threadIdx.x; idx < 256; idx += CONV_NWARPS * 32) {\n"
"            int mr = idx >> 4;        /* idx / 16 */\n"
"            int kc = idx & 15;        /* idx % 16 */\n"
"            int oy = sm_oy[mr], ox = sm_ox[mr];\n"
"            int k_col = k_base + kc;\n"
"            float v = 0.0f;\n"
"            if (oy >= 0 && k_col < K) {\n"
"                int ic = k_col / 9;\n"
"                int kf = k_col - ic * 9;\n"
"                int fy = kf / 3;\n"
"                int fx = kf - fy * 3;\n"
"                int iy = oy + fy - 1, ix = ox + fx - 1;\n"
"                if (iy >= 0 && iy < h && ix >= 0 && ix < w)\n"
"                    v = inp[((long)ic * h + iy) * w + ix];\n"
"            }\n"
"            sm_in[mr * IN_STRIDE + kc] = v;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* ----- A operand: each warp loads its frag from smem and converts to BF16 ----- */\n"
"        unsigned int a0, a1, a2, a3;\n"
"        a0 = (unsigned int)f32_to_bf16_rne(sm_in[gid * IN_STRIDE + tid4*2 + 0]) |\n"
"             ((unsigned int)f32_to_bf16_rne(sm_in[gid * IN_STRIDE + tid4*2 + 1]) << 16);\n"
"#if __CUDA_ARCH__ >= 1200\n"
"        a1 = (unsigned int)f32_to_bf16_rne(sm_in[(gid + 8) * IN_STRIDE + tid4*2 + 0]) |\n"
"             ((unsigned int)f32_to_bf16_rne(sm_in[(gid + 8) * IN_STRIDE + tid4*2 + 1]) << 16);\n"
"        a2 = (unsigned int)f32_to_bf16_rne(sm_in[gid * IN_STRIDE + tid4*2 + 8]) |\n"
"             ((unsigned int)f32_to_bf16_rne(sm_in[gid * IN_STRIDE + tid4*2 + 9]) << 16);\n"
"#else\n"
"        a1 = (unsigned int)f32_to_bf16_rne(sm_in[gid * IN_STRIDE + tid4*2 + 8]) |\n"
"             ((unsigned int)f32_to_bf16_rne(sm_in[gid * IN_STRIDE + tid4*2 + 9]) << 16);\n"
"        a2 = (unsigned int)f32_to_bf16_rne(sm_in[(gid + 8) * IN_STRIDE + tid4*2 + 0]) |\n"
"             ((unsigned int)f32_to_bf16_rne(sm_in[(gid + 8) * IN_STRIDE + tid4*2 + 1]) << 16);\n"
"#endif\n"
"        a3 = (unsigned int)f32_to_bf16_rne(sm_in[(gid + 8) * IN_STRIDE + tid4*2 + 8]) |\n"
"             ((unsigned int)f32_to_bf16_rne(sm_in[(gid + 8) * IN_STRIDE + tid4*2 + 9]) << 16);\n"
"\n"
"        /* ----- N-tile loop: CONV_NTILE m16n8 issues per warp ----- */\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < CONV_NTILE; nt++) {\n"
"            int b_col = warp_n_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (b_col < co) {\n"
"                const unsigned short *wp = wb + (size_t)b_col * K + k_base;\n"
"                int kk0 = tid4*2;\n"
"                int kk1 = tid4*2 + 8;\n"
"                unsigned short w0 = (k_base + kk0     < K) ? wp[kk0]     : 0;\n"
"                unsigned short w1 = (k_base + kk0 + 1 < K) ? wp[kk0 + 1] : 0;\n"
"                unsigned short w2 = (k_base + kk1     < K) ? wp[kk1]     : 0;\n"
"                unsigned short w3 = (k_base + kk1 + 1 < K) ? wp[kk1 + 1] : 0;\n"
"                b0 = (unsigned int)w0 | ((unsigned int)w1 << 16);\n"
"                b1 = (unsigned int)w2 | ((unsigned int)w3 << 16);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
"                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    /* ----- Write output: out[oc, m_row] -----\n"
"     * D-fragment per thread: rows {gid, gid+8} × cols {tid4*2, tid4*2+1} */\n"
"    int yr0 = m_base + gid;\n"
"    int yr1 = m_base + gid + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < CONV_NTILE; nt++) {\n"
"        int yc0 = warp_n_base + nt * 8 + tid4 * 2;\n"
"        int yc1 = yc0 + 1;\n"
"        float bv0 = (bias && yc0 < co) ? bias[yc0] : 0.0f;\n"
"        float bv1 = (bias && yc1 < co) ? bias[yc1] : 0.0f;\n"
"        if (yr0 < M && yc0 < co) out[(long)yc0 * M + yr0] = d0[nt] + bv0;\n"
"        if (yr0 < M && yc1 < co) out[(long)yc1 * M + yr0] = d1[nt] + bv1;\n"
"        if (yr1 < M && yc0 < co) out[(long)yc0 * M + yr1] = d2[nt] + bv0;\n"
"        if (yr1 < M && yc1 < co) out[(long)yc1 * M + yr1] = d3[nt] + bv1;\n"
"    }\n"
"}\n"
"#undef CONV_NTILE\n"
"#undef CONV_NWARPS\n"

/* ============================================================================
 * Conv2d v2: halo-cached BF16 conv2d (3x3 stride=1 pad=1).
 *
 * Each CTA processes 16 contiguous output positions in ONE h-row (assumes
 * w divisible by 16, which holds for Flux2 VAE: w in {64,128,256,512}).
 * Loads a [3 rows × 18 cols × IC_CHUNK channels] halo region into smem ONCE
 * per ic_chunk, then iterates K within the chunk reading from smem.
 *
 * This is the key optimization for memory bandwidth: input gmem reads become
 * coalesced (per-row), and the same input data is reused across all K-iters
 * within the chunk.
 *
 * Block: 64 threads (2 warps, NWARPS=2). Each warp = 8 N-tiles (CONV_NTILE=8).
 * Grid: (ceil(co/128), ceil(M/16)).
 * ============================================================================ */
"#define V2_NTILE 8\n"
"#define V2_NWARPS 2\n"
"#define V2_IC_CHUNK 16\n"
"#define V2_MTILE 2\n"
"#define V2_M_PER_CTA (16 * V2_MTILE)\n"
"#define V2_HALO_W (V2_M_PER_CTA + 2)\n"
"#define V2_SLICE (3 * V2_HALO_W)  /* per-ic slice size: 3 iy × halo_w ix */\n"
"__global__ void conv2d_bf16_f32_3x3_v2(\n"
"    float *__restrict__ out,\n"
"    const float *__restrict__ inp,\n"
"    const unsigned short *__restrict__ wb,\n"
"    const float *__restrict__ bias,\n"
"    int ci, int h, int w, int co) {\n"
"    int M = h * w;\n"
"    int K = ci * 9;\n"
"    int m_base = blockIdx.y * V2_M_PER_CTA;\n"
"    int n_base = blockIdx.x * (8 * V2_NTILE * V2_NWARPS);\n"
"    int warp = threadIdx.x / 32;\n"
"    int lane = threadIdx.x & 31;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane & 3;\n"
"    int warp_n_base = n_base + warp * (8 * V2_NTILE);\n"
"\n"
"    if (m_base >= M || n_base >= co) return;\n"
"\n"
"    /* This CTA's M-block must be inside one h-row (w divisible by M_PER_CTA=32). */\n"
"    int oy = m_base / w;\n"
"    int ox_base = m_base - oy * w;\n"
"\n"
"    /* Per-warp output frags: MTILE M-tiles × 8 N-tiles × 4 floats per thread per tile */\n"
"    float d0[V2_MTILE][V2_NTILE], d1[V2_MTILE][V2_NTILE];\n"
"    float d2[V2_MTILE][V2_NTILE], d3[V2_MTILE][V2_NTILE];\n"
"#pragma unroll\n"
"    for (int mt = 0; mt < V2_MTILE; mt++) {\n"
"#pragma unroll\n"
"        for (int i = 0; i < V2_NTILE; i++) {\n"
"            d0[mt][i]=0; d1[mt][i]=0; d2[mt][i]=0; d3[mt][i]=0;\n"
"        }\n"
"    }\n"
"\n"
"    /* Halo input cache: [IC_CHUNK][3 iy][HALO_W ix].\n"
"     * For MTILE=2: 3*34*16 = 1632 floats = 6.5KB.\n"
"     * Followed by K-position lookup table (144 shorts). */\n"
"    extern __shared__ float v2_smem[];\n"
"    float *sm_in = v2_smem;\n"
"    short *sm_kbase = (short*)(sm_in + V2_SLICE * V2_IC_CHUNK);\n"
"\n"
"    int total_threads = V2_NWARPS * 32;\n"
"\n"
"    for (int ic_base = 0; ic_base < ci; ic_base += V2_IC_CHUNK) {\n"
"        int ic_chunk = (ic_base + V2_IC_CHUNK <= ci) ? V2_IC_CHUNK : (ci - ic_base);\n"
"        int total_load = V2_SLICE * ic_chunk;\n"
"        for (int idx = threadIdx.x; idx < total_load; idx += total_threads) {\n"
"            int ic_off = idx / V2_SLICE;\n"
"            int rem = idx - ic_off * V2_SLICE;\n"
"            int iy_off = rem / V2_HALO_W;\n"
"            int ix_off = rem - iy_off * V2_HALO_W;\n"
"            int ic = ic_base + ic_off;\n"
"            int iy = oy + iy_off - 1;\n"
"            int ix = ox_base + ix_off - 1;\n"
"            float v = 0.0f;\n"
"            if (iy >= 0 && iy < h && ix >= 0 && ix < w)\n"
"                v = inp[((long)ic * h + iy) * w + ix];\n"
"            sm_in[idx] = v;\n"
"        }\n"
"        /* Precompute K-position -> smem-base lookup table.\n"
"         * sm_kbase[k] = ic_off*SLICE + fy*HALO_W + fx. */\n"
"        int K_in_chunk = ic_chunk * 9;\n"
"        for (int k = threadIdx.x; k < K_in_chunk; k += total_threads) {\n"
"            int io = k / 9;\n"
"            int kf = k - io * 9;\n"
"            int fy = kf / 3;\n"
"            int fx = kf - fy * 3;\n"
"            sm_kbase[k] = (short)(io * V2_SLICE + fy * V2_HALO_W + fx);\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* Process K within this ic_chunk in tiles of 16 */\n"
"        for (int k_within = 0; k_within < K_in_chunk; k_within += 16) {\n"
"            int kk0 = k_within + tid4*2;\n"
"            int kk0p1 = kk0 + 1;\n"
"            int kk8 = kk0 + 8;\n"
"            int kk8p1 = kk8 + 1;\n"
"            int b00 = (kk0   < K_in_chunk) ? (int)sm_kbase[kk0]   : -1;\n"
"            int b01 = (kk0p1 < K_in_chunk) ? (int)sm_kbase[kk0p1] : -1;\n"
"            int b80 = (kk8   < K_in_chunk) ? (int)sm_kbase[kk8]   : -1;\n"
"            int b81 = (kk8p1 < K_in_chunk) ? (int)sm_kbase[kk8p1] : -1;\n"
"\n"
"            /* Load A frags for each of the MTILE M-tiles */\n"
"            unsigned int a0_f[V2_MTILE], a1_f[V2_MTILE], a2_f[V2_MTILE], a3_f[V2_MTILE];\n"
"#pragma unroll\n"
"            for (int mt = 0; mt < V2_MTILE; mt++) {\n"
"                int mr = mt * 16;  /* m_off within CTA */\n"
"                float v_a0_lo = (b00 >= 0) ? sm_in[b00 + mr + gid    ] : 0.0f;\n"
"                float v_a0_hi = (b01 >= 0) ? sm_in[b01 + mr + gid    ] : 0.0f;\n"
"                float v_a3_lo = (b80 >= 0) ? sm_in[b80 + mr + gid + 8] : 0.0f;\n"
"                float v_a3_hi = (b81 >= 0) ? sm_in[b81 + mr + gid + 8] : 0.0f;\n"
"#if __CUDA_ARCH__ >= 1200\n"
"                float v_a1_lo = (b00 >= 0) ? sm_in[b00 + mr + gid + 8] : 0.0f;\n"
"                float v_a1_hi = (b01 >= 0) ? sm_in[b01 + mr + gid + 8] : 0.0f;\n"
"                float v_a2_lo = (b80 >= 0) ? sm_in[b80 + mr + gid    ] : 0.0f;\n"
"                float v_a2_hi = (b81 >= 0) ? sm_in[b81 + mr + gid    ] : 0.0f;\n"
"#else\n"
"                float v_a1_lo = (b80 >= 0) ? sm_in[b80 + mr + gid    ] : 0.0f;\n"
"                float v_a1_hi = (b81 >= 0) ? sm_in[b81 + mr + gid    ] : 0.0f;\n"
"                float v_a2_lo = (b00 >= 0) ? sm_in[b00 + mr + gid + 8] : 0.0f;\n"
"                float v_a2_hi = (b01 >= 0) ? sm_in[b01 + mr + gid + 8] : 0.0f;\n"
"#endif\n"
"                a0_f[mt] = (unsigned int)f32_to_bf16_rne(v_a0_lo) | ((unsigned int)f32_to_bf16_rne(v_a0_hi) << 16);\n"
"                a1_f[mt] = (unsigned int)f32_to_bf16_rne(v_a1_lo) | ((unsigned int)f32_to_bf16_rne(v_a1_hi) << 16);\n"
"                a2_f[mt] = (unsigned int)f32_to_bf16_rne(v_a2_lo) | ((unsigned int)f32_to_bf16_rne(v_a2_hi) << 16);\n"
"                a3_f[mt] = (unsigned int)f32_to_bf16_rne(v_a3_lo) | ((unsigned int)f32_to_bf16_rne(v_a3_hi) << 16);\n"
"            }\n"
"\n"
"            /* B operand load + mma issues */\n"
"            int k_abs = ic_base * 9 + k_within;\n"
"#pragma unroll\n"
"            for (int nt = 0; nt < V2_NTILE; nt++) {\n"
"                int b_col = warp_n_base + nt * 8 + gid;\n"
"                unsigned int b0 = 0, b1 = 0;\n"
"                if (b_col < co) {\n"
"                    const unsigned short *wp = wb + (size_t)b_col * K + k_abs;\n"
"                    int kk = tid4 * 2;\n"
"                    int kk2 = tid4 * 2 + 8;\n"
"                    unsigned short w0 = (k_abs + kk     < K) ? wp[kk]     : 0;\n"
"                    unsigned short w1 = (k_abs + kk + 1 < K) ? wp[kk + 1] : 0;\n"
"                    unsigned short w2 = (k_abs + kk2     < K) ? wp[kk2]     : 0;\n"
"                    unsigned short w3 = (k_abs + kk2 + 1 < K) ? wp[kk2 + 1] : 0;\n"
"                    b0 = (unsigned int)w0 | ((unsigned int)w1 << 16);\n"
"                    b1 = (unsigned int)w2 | ((unsigned int)w3 << 16);\n"
"                }\n"
"#pragma unroll\n"
"                for (int mt = 0; mt < V2_MTILE; mt++) {\n"
"                    asm volatile(\n"
"                        \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\\n\\t\"\n"
"                        \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                        : \"=f\"(d0[mt][nt]), \"=f\"(d1[mt][nt]), \"=f\"(d2[mt][nt]), \"=f\"(d3[mt][nt])\n"
"                        : \"r\"(a0_f[mt]), \"r\"(a1_f[mt]), \"r\"(a2_f[mt]), \"r\"(a3_f[mt]),\n"
"                          \"r\"(b0), \"r\"(b1),\n"
"                          \"f\"(d0[mt][nt]), \"f\"(d1[mt][nt]), \"f\"(d2[mt][nt]), \"f\"(d3[mt][nt])\n"
"                    );\n"
"                }\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    /* Write output */\n"
"#pragma unroll\n"
"    for (int mt = 0; mt < V2_MTILE; mt++) {\n"
"        int mr = mt * 16;\n"
"        int yr0 = m_base + mr + gid;\n"
"        int yr1 = m_base + mr + gid + 8;\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < V2_NTILE; nt++) {\n"
"            int yc0 = warp_n_base + nt * 8 + tid4 * 2;\n"
"            int yc1 = yc0 + 1;\n"
"            float bv0 = (bias && yc0 < co) ? bias[yc0] : 0.0f;\n"
"            float bv1 = (bias && yc1 < co) ? bias[yc1] : 0.0f;\n"
"            if (yr0 < M && yc0 < co) out[(long)yc0 * M + yr0] = d0[mt][nt] + bv0;\n"
"            if (yr0 < M && yc1 < co) out[(long)yc1 * M + yr0] = d1[mt][nt] + bv1;\n"
"            if (yr1 < M && yc0 < co) out[(long)yc0 * M + yr1] = d2[mt][nt] + bv0;\n"
"            if (yr1 < M && yc1 < co) out[(long)yc1 * M + yr1] = d3[mt][nt] + bv1;\n"
"        }\n"
"    }\n"
"}\n"
"#undef V2_NTILE\n"
"#undef V2_NWARPS\n"
"#undef V2_IC_CHUNK\n"
"#undef V2_MTILE\n"
"#undef V2_M_PER_CTA\n"
"#undef V2_HALO_W\n"
"#undef V2_SLICE\n"

/* ============================================================================
 * Generic BF16 × F32 -> F32 GEMM using mma.sync.m16n8k16.bf16.bf16.f32 (sm_80+).
 * Same signature as gemm_f32_f32: Y[n_tok, n_out] = X[n_tok, n_in] @ W[n_out, n_in]^T + bias
 * Used by VAE mid-block attention for QKV and output projections.
 *
 * Block: 128 threads (4 warps). Grid: (ceil(n_out/256), ceil(n_tok/16)).
 * Each warp computes 16 M × 64 N output via 8 m16n8k16 N-iters.
 * ============================================================================ */
"#define BG_NTILE 8\n"
"__global__ void gemm_bf16_f32(float *Y, const unsigned short *W, const float *X,\n"
"                               const float *bias, int n_out, int n_in, int n_tok) {\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int warp = threadIdx.x / 32;\n"
"    int lane = threadIdx.x & 31;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane & 3;\n"
"    int out_base = blockIdx.x * 256 + warp * 64;\n"
"\n"
"    if (tok_base >= n_tok) return;\n"
"\n"
"    float d0[BG_NTILE], d1[BG_NTILE], d2[BG_NTILE], d3[BG_NTILE];\n"
"#pragma unroll\n"
"    for (int i = 0; i < BG_NTILE; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"\n"
"    __shared__ float bg_sm_x[16 * 17];  /* stride 17 for bank conflict avoidance */\n"
"\n"
"    for (int k_base = 0; k_base < n_in; k_base += 16) {\n"
"        /* Cooperative load 16 × 16 X tile into smem (128 threads, 2 per thread) */\n"
"        for (int idx = threadIdx.x; idx < 256; idx += 128) {\n"
"            int mr = idx >> 4;\n"
"            int kc = idx & 15;\n"
"            int grow = tok_base + mr;\n"
"            float v = 0.0f;\n"
"            if (grow < n_tok && k_base + kc < n_in)\n"
"                v = X[(long)grow * n_in + k_base + kc];\n"
"            bg_sm_x[mr * 17 + kc] = v;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* Load A frag (same layout as conv2d_bf16 kernel) */\n"
"        unsigned int a0, a1, a2, a3;\n"
"        a0 = (unsigned int)f32_to_bf16_rne(bg_sm_x[gid * 17 + tid4*2 + 0]) |\n"
"             ((unsigned int)f32_to_bf16_rne(bg_sm_x[gid * 17 + tid4*2 + 1]) << 16);\n"
"#if __CUDA_ARCH__ >= 1200\n"
"        a1 = (unsigned int)f32_to_bf16_rne(bg_sm_x[(gid + 8) * 17 + tid4*2 + 0]) |\n"
"             ((unsigned int)f32_to_bf16_rne(bg_sm_x[(gid + 8) * 17 + tid4*2 + 1]) << 16);\n"
"        a2 = (unsigned int)f32_to_bf16_rne(bg_sm_x[gid * 17 + tid4*2 + 8]) |\n"
"             ((unsigned int)f32_to_bf16_rne(bg_sm_x[gid * 17 + tid4*2 + 9]) << 16);\n"
"#else\n"
"        a1 = (unsigned int)f32_to_bf16_rne(bg_sm_x[gid * 17 + tid4*2 + 8]) |\n"
"             ((unsigned int)f32_to_bf16_rne(bg_sm_x[gid * 17 + tid4*2 + 9]) << 16);\n"
"        a2 = (unsigned int)f32_to_bf16_rne(bg_sm_x[(gid + 8) * 17 + tid4*2 + 0]) |\n"
"             ((unsigned int)f32_to_bf16_rne(bg_sm_x[(gid + 8) * 17 + tid4*2 + 1]) << 16);\n"
"#endif\n"
"        a3 = (unsigned int)f32_to_bf16_rne(bg_sm_x[(gid + 8) * 17 + tid4*2 + 8]) |\n"
"             ((unsigned int)f32_to_bf16_rne(bg_sm_x[(gid + 8) * 17 + tid4*2 + 9]) << 16);\n"
"\n"
"        /* 8 N-tile mmas per warp */\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < BG_NTILE; nt++) {\n"
"            int bc = out_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < n_out) {\n"
"                const unsigned short *wp = W + (size_t)bc * n_in + k_base;\n"
"                unsigned short w0 = (k_base + tid4*2     < n_in) ? wp[tid4*2]     : 0;\n"
"                unsigned short w1 = (k_base + tid4*2 + 1 < n_in) ? wp[tid4*2 + 1] : 0;\n"
"                unsigned short w2 = (k_base + tid4*2 + 8 < n_in) ? wp[tid4*2 + 8] : 0;\n"
"                unsigned short w3 = (k_base + tid4*2 + 9 < n_in) ? wp[tid4*2 + 9] : 0;\n"
"                b0 = (unsigned int)w0 | ((unsigned int)w1 << 16);\n"
"                b1 = (unsigned int)w2 | ((unsigned int)w3 << 16);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
"                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    /* Write output [n_tok, n_out] row-major */\n"
"    int yr0 = tok_base + gid;\n"
"    int yr1 = tok_base + gid + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < BG_NTILE; nt++) {\n"
"        int yc0 = out_base + nt * 8 + tid4 * 2;\n"
"        int yc1 = yc0 + 1;\n"
"        float bv0 = (bias && yc0 < n_out) ? bias[yc0] : 0.0f;\n"
"        float bv1 = (bias && yc1 < n_out) ? bias[yc1] : 0.0f;\n"
"        if (yr0 < n_tok && yc0 < n_out) Y[(long)yr0 * n_out + yc0] = d0[nt] + bv0;\n"
"        if (yr0 < n_tok && yc1 < n_out) Y[(long)yr0 * n_out + yc1] = d1[nt] + bv1;\n"
"        if (yr1 < n_tok && yc0 < n_out) Y[(long)yr1 * n_out + yc0] = d2[nt] + bv0;\n"
"        if (yr1 < n_tok && yc1 < n_out) Y[(long)yr1 * n_out + yc1] = d3[nt] + bv1;\n"
"    }\n"
"}\n"
"#undef BG_NTILE\n"

/* ============================================================================
 * Support kernels for the BF16 VAE mid-block attention pipeline.
 * ============================================================================ */

/* F32 -> BF16 conversion (n elements), round-to-nearest-even. */
"__global__ void f32_to_bf16_kernel(unsigned short *out, const float *in, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    out[i] = f32_to_bf16_rne(in[i]);\n"
"}\n"

/* Transpose F32 -> BF16 in one pass. in[rows, cols] -> out[cols, rows] as bf16. */
"__global__ void transpose_f32_to_bf16(unsigned short *out, const float *in,\n"
"                                       int rows, int cols) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = rows * cols;\n"
"    if (idx >= total) return;\n"
"    int r = idx / cols, c = idx - r * cols;\n"
"    out[(long)c * rows + r] = f32_to_bf16_rne(in[idx]);\n"
"}\n"

/* Row-wise softmax with scale, in-place on F32 matrix [n_rows, n_cols].\n"
 * One block per row. Uses shared memory reduction. */
"__global__ void softmax_row_f32_inplace(float *mat, int n_rows, int n_cols,\n"
"                                         float scale) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= n_rows) return;\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    float *base = mat + (long)row * n_cols;\n"
"\n"
"    /* Pass 1: find max */\n"
"    float m_local = -1e30f;\n"
"    for (int i = tid; i < n_cols; i += nt) {\n"
"        float v = base[i] * scale;\n"
"        base[i] = v;  /* store scaled for next pass */\n"
"        if (v > m_local) m_local = v;\n"
"    }\n"
"    sdata[tid] = m_local;\n"
"    __syncthreads();\n"
"    for (int r = nt / 2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] = fmaxf(sdata[tid], sdata[tid + r]);\n"
"        __syncthreads();\n"
"    }\n"
"    float m_max = sdata[0];\n"
"    __syncthreads();\n"
"\n"
"    /* Pass 2: exp(x - max), sum */\n"
"    float s_local = 0.0f;\n"
"    for (int i = tid; i < n_cols; i += nt) {\n"
"        float e = expf(base[i] - m_max);\n"
"        base[i] = e;\n"
"        s_local += e;\n"
"    }\n"
"    sdata[tid] = s_local;\n"
"    __syncthreads();\n"
"    for (int r = nt / 2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid + r];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv = 1.0f / sdata[0];\n"
"    __syncthreads();\n"
"\n"
"    /* Pass 3: normalize */\n"
"    for (int i = tid; i < n_cols; i += nt)\n"
"        base[i] *= inv;\n"
"}\n"

"#endif\n"

/* Nearest-neighbor 2x upsample on CHW tensor. */
"__global__ void flux2_vae_upsample2x_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, int C, int H, int W) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int oH = H * 2, oW = W * 2;\n"
"    int total = C * oH * oW;\n"
"    if (idx >= total) return;\n"
"    int c = idx / (oH * oW);\n"
"    int rem = idx % (oH * oW);\n"
"    int oy = rem / oW, ox = rem % oW;\n"
"    int iy = oy / 2, ix = ox / 2;\n"
"    out[idx] = inp[(long)c * H * W + iy * W + ix];\n"
"}\n"

/* Patch-space latent affine using BN stats indexed by 2x2 patch position. */
"__global__ void flux2_vae_latent_bn_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, const float *__restrict__ mean,\n"
"    const float *__restrict__ std, int lc, int h, int w, int ps) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = lc * h * w;\n"
"    if (idx >= total) return;\n"
"    int ch = idx / (h * w);\n"
"    int rem = idx % (h * w);\n"
"    int y = rem / w, x = rem % w;\n"
"    int pr = y % ps, pc = x % ps;\n"
"    int bn_ch = ch * ps * ps + pr * ps + pc;\n"
"    out[idx] = inp[idx] * std[bn_ch] + mean[bn_ch];\n"
"}\n"

/* BF16 truncation: round F32 to BF16 precision in-place */
"__global__ void truncate_bf16_f32(float *__restrict__ x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    unsigned int bits;\n"
"    memcpy(&bits, &x[i], 4);\n"
"    unsigned int rounding = 0x7FFFu + ((bits >> 16) & 1u);\n"
"    bits += rounding;\n"
"    bits &= 0xFFFF0000u;\n"
"    memcpy(&x[i], &bits, 4);\n"
"}\n"

/* FP8 E4M3 → F32 device function (bias=7, 4-bit exponent, 3-bit mantissa) */
"__device__ __forceinline__ float fp8e4m3_to_f32(unsigned char b) {\n"
"    int sign = (b >> 7) & 1;\n"
"    int exp  = (b >> 3) & 0xF;\n"
"    int mant = b & 0x7;\n"
"    float v;\n"
"    if (exp == 0 && mant == 0) return 0.0f;\n"
"    if (exp == 15 && mant == 7) return 0.0f;\n"  /* NaN → 0 */
"    if (exp == 0) v = ((float)mant / 8.0f) * (1.0f / 64.0f);\n"  /* subnormal: 2^-6 */
"    else v = (1.0f + (float)mant / 8.0f) * exp2f((float)(exp - 7));\n"
"    return sign ? -v : v;\n"
"}\n"

/* Round F32 to BF16 precision (round-to-nearest-even, kept as F32 for math) */
"__device__ __forceinline__ float to_bf16(float f) {\n"
"    unsigned int b; memcpy(&b, &f, 4);\n"
"    unsigned int round = 0x7FFFu + ((b >> 16) & 1u);\n"
"    b += round;\n"
"    b &= 0xFFFF0000u;\n"
"    float r; memcpy(&r, &b, 4); return r;\n"
"}\n"

/* FP8 GEMM with BF16 activations — matches ComfyUI/PyTorch FP8 matmul.
 * Truncates X and weight to BF16 precision before multiply, F32 accumulation. */
"__global__ void gemm_tiled_fp8_bf16(float *Y, const unsigned char *W, const float *X,\n"
"    const float *bias, int n_out, int n_in, int n_tok, float w_scale) {\n"
"    __shared__ float smA[16][16];\n"
"    __shared__ float smB[16][16];\n"
"    int tx = threadIdx.x, ty = threadIdx.y;\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int out_base = blockIdx.x * 64;\n"
"    int row = tok_base + ty;\n"
"    /* Pre-scale: include w_scale in BF16 round so the effective weight matches BF16 cast */\n"
"    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        smA[ty][tx] = (tok_base+ty < n_tok && k+tx < n_in)\n"
"                      ? to_bf16(X[(long)(tok_base+ty)*n_in + k+tx]) : 0.f;\n"
"        __syncthreads();\n"
"        { int wo = out_base + tx;\n"
"          smB[ty][tx] = (wo < n_out && k+ty < n_in)\n"
"                        ? to_bf16(fp8e4m3_to_f32(W[(size_t)wo*n_in + k+ty]) * w_scale) : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc0 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int wo = out_base+16+tx;\n"
"          smB[ty][tx] = (wo < n_out && k+ty < n_in)\n"
"                        ? to_bf16(fp8e4m3_to_f32(W[(size_t)wo*n_in + k+ty]) * w_scale) : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc1 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int wo = out_base+32+tx;\n"
"          smB[ty][tx] = (wo < n_out && k+ty < n_in)\n"
"                        ? to_bf16(fp8e4m3_to_f32(W[(size_t)wo*n_in + k+ty]) * w_scale) : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc2 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int wo = out_base+48+tx;\n"
"          smB[ty][tx] = (wo < n_out && k+ty < n_in)\n"
"                        ? to_bf16(fp8e4m3_to_f32(W[(size_t)wo*n_in + k+ty]) * w_scale) : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc3 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < n_tok) {\n"
"        if (out_base+   tx < n_out) Y[(long)row*n_out+out_base+   tx] = to_bf16(acc0 + (bias?bias[out_base+   tx]:0.f));\n"
"        if (out_base+16+tx < n_out) Y[(long)row*n_out+out_base+16+tx] = to_bf16(acc1 + (bias?bias[out_base+16+tx]:0.f));\n"
"        if (out_base+32+tx < n_out) Y[(long)row*n_out+out_base+32+tx] = to_bf16(acc2 + (bias?bias[out_base+32+tx]:0.f));\n"
"        if (out_base+48+tx < n_out) Y[(long)row*n_out+out_base+48+tx] = to_bf16(acc3 + (bias?bias[out_base+48+tx]:0.f));\n"
"    }\n"
"}\n"

/* Tiled FP8 E4M3 GEMM: same grid/block as F16 tiled version.
 * W is uint8 FP8 E4M3, dequantized on-the-fly: Y = (X @ W^T) * w_scale + bias */
"__global__ void gemm_tiled_fp8_f32(float *Y, const unsigned char *W, const float *X,\n"
"    const float *bias, int n_out, int n_in, int n_tok, float w_scale) {\n"
"    __shared__ float smA[16][16];\n"
"    __shared__ float smB[16][16];\n"
"    int tx = threadIdx.x, ty = threadIdx.y;\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int out_base = blockIdx.x * 64;\n"
"    int row = tok_base + ty;\n"
"    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        smA[ty][tx] = (tok_base+ty < n_tok && k+tx < n_in)\n"
"                      ? X[(long)(tok_base+ty)*n_in + k+tx] : 0.f;\n"
"        __syncthreads();\n"
"        { int w_out = out_base + tx;\n"
"          smB[ty][tx] = (w_out < n_out && k+ty < n_in)\n"
"                        ? fp8e4m3_to_f32(W[(size_t)w_out*n_in + k+ty]) : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc0 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w_out = out_base+16+tx;\n"
"          smB[ty][tx] = (w_out < n_out && k+ty < n_in)\n"
"                        ? fp8e4m3_to_f32(W[(size_t)w_out*n_in + k+ty]) : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc1 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w_out = out_base+32+tx;\n"
"          smB[ty][tx] = (w_out < n_out && k+ty < n_in)\n"
"                        ? fp8e4m3_to_f32(W[(size_t)w_out*n_in + k+ty]) : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc2 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w_out = out_base+48+tx;\n"
"          smB[ty][tx] = (w_out < n_out && k+ty < n_in)\n"
"                        ? fp8e4m3_to_f32(W[(size_t)w_out*n_in + k+ty]) : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc3 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < n_tok) {\n"
"        if (out_base+   tx < n_out) Y[(long)row*n_out+out_base+   tx] = acc0*w_scale + (bias?bias[out_base+   tx]:0.f);\n"
"        if (out_base+16+tx < n_out) Y[(long)row*n_out+out_base+16+tx] = acc1*w_scale + (bias?bias[out_base+16+tx]:0.f);\n"
"        if (out_base+32+tx < n_out) Y[(long)row*n_out+out_base+32+tx] = acc2*w_scale + (bias?bias[out_base+32+tx]:0.f);\n"
"        if (out_base+48+tx < n_out) Y[(long)row*n_out+out_base+48+tx] = acc3*w_scale + (bias?bias[out_base+48+tx]:0.f);\n"
"    }\n"
"}\n"

/* ============================================================
 * MMA-based FP8 GEMM (m16n8k32, sm_89+) with per-tensor weight scale.
 * Grid: (ceil(n_out/256), ceil(n_tok/16)), Block: 128 threads (4 warps).
 * Shared mem: 16 * 32 * sizeof(float) = 2048 bytes for X tile.
 * Each warp computes 16 rows × 64 cols of Y via 8 mma.sync issues.
 * sm_120 (Blackwell) has the a1/a2 fragment swap (same as cuda_kernels_common.h).
 * ============================================================ */
"#if __CUDA_ARCH__ >= 890\n"
"#define FP8_GEMM_NTILE 8\n"
"#define FP8_GEMM_MTILE 2   /* number of m16n8 M-tiles per CTA */\n"
"#define FP8_GEMM_NWARPS 4  /* warps per CTA (4 × 64 N = 256 N cols per CTA) */\n"
"__global__ void gemm_fp8_scaled_f32(float *Y, const unsigned char *W, const float *X,\n"
"                                     const float *bias, int n_out, int n_in, int n_tok,\n"
"                                     float w_scale) {\n"
"    extern __shared__ float fp8_smem_x[];\n"
"    /* sm_x: [FP8_GEMM_MTILE*16 rows × 32 K cols] = 1024 floats per iter */\n"
"    int tok_base = blockIdx.y * (16 * FP8_GEMM_MTILE);\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int out_base = blockIdx.x * (64 * FP8_GEMM_NWARPS) + warp_id * 64;\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane % 4;\n"
"    int tid = threadIdx.x;\n"
"    if (tok_base >= n_tok) return;\n"
"\n"
"    /* 2 M-tiles × 8 N-tiles × 4 f32 per thread = 64 f32 per lane */\n"
"    float d0[FP8_GEMM_MTILE][FP8_GEMM_NTILE];\n"
"    float d1[FP8_GEMM_MTILE][FP8_GEMM_NTILE];\n"
"    float d2[FP8_GEMM_MTILE][FP8_GEMM_NTILE];\n"
"    float d3[FP8_GEMM_MTILE][FP8_GEMM_NTILE];\n"
"#pragma unroll\n"
"    for (int mt = 0; mt < FP8_GEMM_MTILE; mt++) {\n"
"#pragma unroll\n"
"        for (int i = 0; i < FP8_GEMM_NTILE; i++) {\n"
"            d0[mt][i]=0; d1[mt][i]=0; d2[mt][i]=0; d3[mt][i]=0;\n"
"        }\n"
"    }\n"
"\n"
"    int M_ROWS = 16 * FP8_GEMM_MTILE;\n"
"    for (int k = 0; k < n_in; k += 32) {\n"
"        /* Cooperative load M_ROWS × 32 K tile. 128 threads × 8 floats per iter = 1024 per pass.\n"
"         * For MTILE=2 (M_ROWS=32), one pass covers the full tile. */\n"
"        int srow = tid / 4;\n"
"        int scol = (tid & 3) * 8;\n"
"        if (srow < M_ROWS) {\n"
"            int grow = tok_base + srow;\n"
"#pragma unroll\n"
"            for (int j = 0; j < 8; j++) {\n"
"                float v = (grow < n_tok && k + scol + j < n_in)\n"
"                        ? X[(long)grow * n_in + k + scol + j] : 0.0f;\n"
"                fp8_smem_x[srow * 32 + scol + j] = v;\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"#define CVT_E4M3_PAIR(reg, r, c) \\\n"
"        { unsigned short lo, hi; \\\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(lo) \\\n"
"              : \"f\"(fp8_smem_x[(r)*32+(c)]), \"f\"(fp8_smem_x[(r)*32+(c)+1])); \\\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(hi) \\\n"
"              : \"f\"(fp8_smem_x[(r)*32+(c)+2]), \"f\"(fp8_smem_x[(r)*32+(c)+3])); \\\n"
"          reg = (unsigned int)lo | ((unsigned int)hi << 16); }\n"
"\n"
"        /* Load all A frags for each M-tile once (hoisted outside N-tile loop) */\n"
"        unsigned int a0[FP8_GEMM_MTILE], a1[FP8_GEMM_MTILE], a2[FP8_GEMM_MTILE], a3[FP8_GEMM_MTILE];\n"
"#pragma unroll\n"
"        for (int mt = 0; mt < FP8_GEMM_MTILE; mt++) {\n"
"            int mr = mt * 16;\n"
"            CVT_E4M3_PAIR(a0[mt], mr + gid,     tid4*4)\n"
"#if __CUDA_ARCH__ >= 1200\n"
"            CVT_E4M3_PAIR(a1[mt], mr + gid + 8, tid4*4)\n"
"            CVT_E4M3_PAIR(a2[mt], mr + gid,     tid4*4 + 16)\n"
"#else\n"
"            CVT_E4M3_PAIR(a1[mt], mr + gid,     tid4*4 + 16)\n"
"            CVT_E4M3_PAIR(a2[mt], mr + gid + 8, tid4*4)\n"
"#endif\n"
"            CVT_E4M3_PAIR(a3[mt], mr + gid + 8, tid4*4 + 16)\n"
"        }\n"
"#undef CVT_E4M3_PAIR\n"
"\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < FP8_GEMM_NTILE; nt++) {\n"
"            int bc = out_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < n_out && k + tid4*4 + 19 < n_in) {\n"
"                const unsigned char *wp = W + (size_t)bc * n_in + k;\n"
"                b0 = *(const unsigned int *)(wp + tid4 * 4);\n"
"                b1 = *(const unsigned int *)(wp + tid4 * 4 + 16);\n"
"            } else if (bc < n_out) {\n"
"                unsigned char tmp[8] = {0,0,0,0,0,0,0,0};\n"
"                const unsigned char *wp = W + (size_t)bc * n_in + k;\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    int kk = k + tid4*4 + j;\n"
"                    if (kk < n_in) tmp[j] = wp[tid4*4 + j];\n"
"                    int kk2 = k + tid4*4 + 16 + j;\n"
"                    if (kk2 < n_in) tmp[4+j] = wp[tid4*4 + 16 + j];\n"
"                }\n"
"                b0 = ((unsigned int)tmp[0]) | ((unsigned int)tmp[1] << 8) | ((unsigned int)tmp[2] << 16) | ((unsigned int)tmp[3] << 24);\n"
"                b1 = ((unsigned int)tmp[4]) | ((unsigned int)tmp[5] << 8) | ((unsigned int)tmp[6] << 16) | ((unsigned int)tmp[7] << 24);\n"
"            }\n"
"\n"
"            /* Issue mma for each M-tile with shared B */\n"
"#pragma unroll\n"
"            for (int mt = 0; mt < FP8_GEMM_MTILE; mt++) {\n"
"                asm volatile(\n"
"                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    : \"=f\"(d0[mt][nt]), \"=f\"(d1[mt][nt]), \"=f\"(d2[mt][nt]), \"=f\"(d3[mt][nt])\n"
"                    : \"r\"(a0[mt]), \"r\"(a1[mt]), \"r\"(a2[mt]), \"r\"(a3[mt]),\n"
"                      \"r\"(b0), \"r\"(b1),\n"
"                      \"f\"(d0[mt][nt]), \"f\"(d1[mt][nt]), \"f\"(d2[mt][nt]), \"f\"(d3[mt][nt])\n"
"                );\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"#pragma unroll\n"
"    for (int mt = 0; mt < FP8_GEMM_MTILE; mt++) {\n"
"        int mr = mt * 16;\n"
"        int yr0 = tok_base + mr + gid;\n"
"        int yr1 = tok_base + mr + gid + 8;\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < FP8_GEMM_NTILE; nt++) {\n"
"            int yc0 = out_base + nt * 8 + tid4 * 2;\n"
"            int yc1 = yc0 + 1;\n"
"            float bv0 = (bias && yc0 < n_out) ? bias[yc0] : 0.0f;\n"
"            float bv1 = (bias && yc1 < n_out) ? bias[yc1] : 0.0f;\n"
"            if (yr0 < n_tok && yc0 < n_out) Y[(long)yr0 * n_out + yc0] = d0[mt][nt] * w_scale + bv0;\n"
"            if (yr0 < n_tok && yc1 < n_out) Y[(long)yr0 * n_out + yc1] = d1[mt][nt] * w_scale + bv1;\n"
"            if (yr1 < n_tok && yc0 < n_out) Y[(long)yr1 * n_out + yc0] = d2[mt][nt] * w_scale + bv0;\n"
"            if (yr1 < n_tok && yc1 < n_out) Y[(long)yr1 * n_out + yc1] = d3[mt][nt] * w_scale + bv1;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* BF16-truncated output sibling. Pre-rounds X to BF16 before FP8 cast for ComfyUI matching. */\n"
"__global__ void gemm_fp8_scaled_bf16(float *Y, const unsigned char *W, const float *X,\n"
"                                      const float *bias, int n_out, int n_in, int n_tok,\n"
"                                      float w_scale) {\n"
"    extern __shared__ float fp8_smem_x[];\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int out_base = blockIdx.x * 256 + warp_id * 64;\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane % 4;\n"
"    int tid = threadIdx.x;\n"
"    if (tok_base >= n_tok) return;\n"
"\n"
"    float d0[FP8_GEMM_NTILE], d1[FP8_GEMM_NTILE], d2[FP8_GEMM_NTILE], d3[FP8_GEMM_NTILE];\n"
"#pragma unroll\n"
"    for (int i = 0; i < FP8_GEMM_NTILE; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"\n"
"    for (int k = 0; k < n_in; k += 32) {\n"
"        int srow = tid / 8, scol = (tid % 8) * 4;\n"
"        int grow = tok_base + srow;\n"
"        for (int j = 0; j < 4; j++) {\n"
"            int kk = k + scol + j;\n"
"            float v = (grow < n_tok && kk < n_in) ? X[(long)grow * n_in + kk] : 0.0f;\n"
"            fp8_smem_x[srow * 32 + scol + j] = to_bf16(v);\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        unsigned int a0, a1, a2, a3;\n"
"#define CVT_E4M3_PAIR(reg, r, c) \\\n"
"        { unsigned short lo, hi; \\\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(lo) \\\n"
"              : \"f\"(fp8_smem_x[(r)*32+(c)]), \"f\"(fp8_smem_x[(r)*32+(c)+1])); \\\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(hi) \\\n"
"              : \"f\"(fp8_smem_x[(r)*32+(c)+2]), \"f\"(fp8_smem_x[(r)*32+(c)+3])); \\\n"
"          reg = (unsigned int)lo | ((unsigned int)hi << 16); }\n"
"        CVT_E4M3_PAIR(a0, gid, tid4*4)\n"
"#if __CUDA_ARCH__ >= 1200\n"
"        CVT_E4M3_PAIR(a1, gid+8, tid4*4)\n"
"        CVT_E4M3_PAIR(a2, gid,   tid4*4+16)\n"
"#else\n"
"        CVT_E4M3_PAIR(a1, gid,   tid4*4+16)\n"
"        CVT_E4M3_PAIR(a2, gid+8, tid4*4)\n"
"#endif\n"
"        CVT_E4M3_PAIR(a3, gid+8, tid4*4+16)\n"
"#undef CVT_E4M3_PAIR\n"
"\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < FP8_GEMM_NTILE; nt++) {\n"
"            int bc = out_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < n_out && k + tid4*4 + 19 < n_in) {\n"
"                const unsigned char *wp = W + (size_t)bc * n_in + k;\n"
"                b0 = *(const unsigned int *)(wp + tid4 * 4);\n"
"                b1 = *(const unsigned int *)(wp + tid4 * 4 + 16);\n"
"            } else if (bc < n_out) {\n"
"                unsigned char tmp[8] = {0,0,0,0,0,0,0,0};\n"
"                const unsigned char *wp = W + (size_t)bc * n_in + k;\n"
"                for (int j = 0; j < 4; j++) {\n"
"                    if (k + tid4*4 + j      < n_in) tmp[j]   = wp[tid4*4 + j];\n"
"                    if (k + tid4*4 + 16 + j < n_in) tmp[4+j] = wp[tid4*4 + 16 + j];\n"
"                }\n"
"                b0 = ((unsigned int)tmp[0]) | ((unsigned int)tmp[1] << 8) | ((unsigned int)tmp[2] << 16) | ((unsigned int)tmp[3] << 24);\n"
"                b1 = ((unsigned int)tmp[4]) | ((unsigned int)tmp[5] << 8) | ((unsigned int)tmp[6] << 16) | ((unsigned int)tmp[7] << 24);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
"                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    int yr0 = tok_base + gid;\n"
"    int yr1 = tok_base + gid + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < FP8_GEMM_NTILE; nt++) {\n"
"        int yc0 = out_base + nt * 8 + tid4 * 2;\n"
"        int yc1 = yc0 + 1;\n"
"        float bv0 = (bias && yc0 < n_out) ? bias[yc0] : 0.0f;\n"
"        float bv1 = (bias && yc1 < n_out) ? bias[yc1] : 0.0f;\n"
"        if (yr0 < n_tok && yc0 < n_out) Y[(long)yr0 * n_out + yc0] = to_bf16(d0[nt] * w_scale + bv0);\n"
"        if (yr0 < n_tok && yc1 < n_out) Y[(long)yr0 * n_out + yc1] = to_bf16(d1[nt] * w_scale + bv1);\n"
"        if (yr1 < n_tok && yc0 < n_out) Y[(long)yr1 * n_out + yc0] = to_bf16(d2[nt] * w_scale + bv0);\n"
"        if (yr1 < n_tok && yc1 < n_out) Y[(long)yr1 * n_out + yc1] = to_bf16(d3[nt] * w_scale + bv1);\n"
"    }\n"
"}\n"
"#endif\n"

/* Transpose 2D: [rows, cols] → [cols, rows]. Grid: ceil(total/256) */
"__global__ void transpose_2d_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, int rows, int cols) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = rows * cols;\n"
"    if (idx >= total) return;\n"
"    int r = idx / cols, c = idx % cols;\n"
"    out[(long)c * rows + r] = inp[(long)r * cols + c];\n"
"}\n"

/* Transpose + residual: out[ch*S + s] = x[ch*S + s] + proj[s*C + ch] */
"__global__ void transpose_add_f32(float *__restrict__ out,\n"
"    const float *__restrict__ x, const float *__restrict__ proj,\n"
"    int C, int S) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = C * S;\n"
"    if (idx >= total) return;\n"
"    int ch = idx / S, s = idx % S;\n"
"    out[(long)ch * S + s] = x[(long)ch * S + s] + proj[(long)s * C + ch];\n"
"}\n"

/* Single-head flash attention for VAE mid-block (c=512 fixed).
 * Q,K,V: [N, 512], out: [N, 512]. Grid: (ceil(N/4)), Block: 128 (4 warps).
 * Online softmax FA2 style, BKV=8 KV tile size. */
"#define VAE_WARPS 4\n"
"#define VAE_BKV   8\n"
"#define VAE_EPT  16\n"
"__global__ void vae_attn_f32(float *__restrict__ out,\n"
"    const float *__restrict__ Q, const float *__restrict__ K,\n"
"    const float *__restrict__ V, int N, int dim, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    float *smK = smem;\n"
"    float *smV = smem + VAE_BKV * dim;\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int qi = blockIdx.x * VAE_WARPS + warp_id;\n"
"    float qr[VAE_EPT];\n"
"    if (qi < N) {\n"
"        for (int e = 0; e < VAE_EPT; e++)\n"
"            qr[e] = Q[(long)qi * dim + lane * VAE_EPT + e];\n"
"    }\n"
"    float m_prev = -1e30f, l_prev = 0.0f;\n"
"    float o[VAE_EPT];\n"
"    for (int e = 0; e < VAE_EPT; e++) o[e] = 0.0f;\n"
"    int n_tiles = (N + VAE_BKV - 1) / VAE_BKV;\n"
"    for (int t = 0; t < n_tiles; t++) {\n"
"        int kv_base = t * VAE_BKV;\n"
"        int tid = threadIdx.x, nt = blockDim.x;\n"
"        int total = VAE_BKV * dim;\n"
"        for (int idx = tid; idx < total; idx += nt) {\n"
"            int row = idx / dim, col = idx % dim;\n"
"            int gi = kv_base + row;\n"
"            smK[idx] = (gi < N) ? K[(long)gi * dim + col] : 0.0f;\n"
"            smV[idx] = (gi < N) ? V[(long)gi * dim + col] : 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        if (qi < N) {\n"
"            for (int ki = 0; ki < VAE_BKV; ki++) {\n"
"                if (kv_base + ki >= N) continue;\n"
"                float dot = 0.0f;\n"
"                for (int e = 0; e < VAE_EPT; e++)\n"
"                    dot += qr[e] * smK[ki * dim + lane * VAE_EPT + e];\n"
"                for (int m = 16; m > 0; m >>= 1)\n"
"                    dot += __shfl_xor_sync(0xFFFFFFFF, dot, m);\n"
"                dot *= scale;\n"
"                float m_new = fmaxf(m_prev, dot);\n"
"                float exp_prev = expf(m_prev - m_new);\n"
"                float exp_cur = expf(dot - m_new);\n"
"                float l_new = l_prev * exp_prev + exp_cur;\n"
"                for (int e = 0; e < VAE_EPT; e++)\n"
"                    o[e] = o[e] * exp_prev + exp_cur * smV[ki * dim + lane * VAE_EPT + e];\n"
"                m_prev = m_new;\n"
"                l_prev = l_new;\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (qi < N && l_prev > 0.0f) {\n"
"        float inv_l = 1.0f / l_prev;\n"
"        for (int e = 0; e < VAE_EPT; e++)\n"
"            out[(long)qi * dim + lane * VAE_EPT + e] = o[e] * inv_l;\n"
"    }\n"
"}\n"

/* F32 tiled GEMM: same grid as F16 version but uses float weights */
"__global__ void gemm_f32_f32(float *Y, const float *W, const float *X,\n"
"    const float *bias, int n_out, int n_in, int n_tok) {\n"
"    __shared__ float smA[16][16];\n"
"    __shared__ float smB[16][16];\n"
"    int tx = threadIdx.x, ty = threadIdx.y;\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int out_base = blockIdx.x * 64;\n"
"    int row = tok_base + ty;\n"
"    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        smA[ty][tx] = (tok_base+ty < n_tok && k+tx < n_in) ? X[(long)(tok_base+ty)*n_in + k+tx] : 0.f;\n"
"        __syncthreads();\n"
"        { int w = out_base + tx; smB[ty][tx] = (w < n_out && k+ty < n_in) ? W[(long)w*n_in+k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc0 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w = out_base+16+tx; smB[ty][tx] = (w < n_out && k+ty < n_in) ? W[(long)w*n_in+k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc1 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w = out_base+32+tx; smB[ty][tx] = (w < n_out && k+ty < n_in) ? W[(long)w*n_in+k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc2 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w = out_base+48+tx; smB[ty][tx] = (w < n_out && k+ty < n_in) ? W[(long)w*n_in+k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc3 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < n_tok) {\n"
"        if (out_base+   tx < n_out) Y[(long)row*n_out+out_base+   tx] = acc0 + (bias?bias[out_base+   tx]:0.f);\n"
"        if (out_base+16+tx < n_out) Y[(long)row*n_out+out_base+16+tx] = acc1 + (bias?bias[out_base+16+tx]:0.f);\n"
"        if (out_base+32+tx < n_out) Y[(long)row*n_out+out_base+32+tx] = acc2 + (bias?bias[out_base+32+tx]:0.f);\n"
"        if (out_base+48+tx < n_out) Y[(long)row*n_out+out_base+48+tx] = acc3 + (bias?bias[out_base+48+tx]:0.f);\n"
"    }\n"
"}\n"

/* RoPE for image tokens: axis 2 = row, axis 3 = col. theta=2000 */
"__global__ void flux2_rope_img_f32(float *x, int n_tok, int n_heads,\n"
"    int head_dim, int lat_w, float theta) {\n"
"    int tok = blockIdx.z, head = blockIdx.y;\n"
"    int pair = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (tok >= n_tok || head >= n_heads || pair >= head_dim / 2) return;\n"
"    int axis_dim = head_dim / 4;\n"
"    int n_pairs = axis_dim / 2;\n"
"    int ax = -1, p_in_ax = -1;\n"
"    if (pair >= n_pairs && pair < 2 * n_pairs) { ax = 1; p_in_ax = pair - n_pairs; }\n"
"    else if (pair >= 2 * n_pairs && pair < 3 * n_pairs) { ax = 2; p_in_ax = pair - 2 * n_pairs; }\n"
"    if (ax < 0) return;\n"
"    int row = tok / lat_w, col = tok % lat_w;\n"
"    float pos = (ax == 1) ? (float)row : (float)col;\n"
"    float freq = pos / powf(theta, (float)(2 * p_in_ax) / (float)axis_dim);\n"
"    float cos_f = cosf(freq), sin_f = sinf(freq);\n"
"    long idx = (long)tok * n_heads * head_dim + head * head_dim + pair * 2;\n"
"    float a = x[idx], b = x[idx + 1];\n"
"    x[idx]     = a * cos_f - b * sin_f;\n"
"    x[idx + 1] = a * sin_f + b * cos_f;\n"
"}\n"

/* RoPE for text tokens: only axis 3 = sequence position */
"__global__ void flux2_rope_txt_f32(float *x, int n_tok, int n_heads,\n"
"    int head_dim, float theta) {\n"
"    int tok = blockIdx.z, head = blockIdx.y;\n"
"    int pair = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (tok >= n_tok || head >= n_heads || pair >= head_dim / 2) return;\n"
"    int axis_dim = head_dim / 4;\n"
"    int n_pairs = axis_dim / 2;\n"
"    if (pair < 3 * n_pairs) return;\n"
"    int p_in_ax = pair - 3 * n_pairs;\n"
"    float freq = (float)tok / powf(theta, (float)(2 * p_in_ax) / (float)axis_dim);\n"
"    float cos_f = cosf(freq), sin_f = sinf(freq);\n"
"    long idx = (long)tok * n_heads * head_dim + head * head_dim + pair * 2;\n"
"    float a = x[idx], b = x[idx + 1];\n"
"    x[idx]     = a * cos_f - b * sin_f;\n"
"    x[idx + 1] = a * sin_f + b * cos_f;\n"
"}\n"

/* Flash attention: warp-cooperative, FA2 style. Q/K/V: [N, n_heads, head_dim] */
"#define FA2_WARPS   4\n"
"#define FA2_BKV    16\n"
"#define FA2_HD    128\n"
"#define FA2_EPT     4\n"
"\n"
"__global__ void flash_attn_f32(float *__restrict__ out,\n"
"    const float *__restrict__ Q, const float *__restrict__ K,\n"
"    const float *__restrict__ V,\n"
"    int N, int n_heads, int head_dim) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int dim = n_heads * head_dim;\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane    = threadIdx.x % 32;\n"
"    int qi = blockIdx.y * FA2_WARPS + warp_id;\n"
"    float scale = rsqrtf((float)head_dim);\n"
"    float q_r[FA2_EPT];\n"
"    for (int e = 0; e < FA2_EPT; e++) {\n"
"        int d = lane * FA2_EPT + e;\n"
"        q_r[e] = (qi < N && d < head_dim) ? Q[(long)qi * dim + h * head_dim + d] : 0.0f;\n"
"    }\n"
"    float m_i = -1e30f, l_i = 0.0f;\n"
"    float O_r[FA2_EPT];\n"
"    for (int e = 0; e < FA2_EPT; e++) O_r[e] = 0.0f;\n"
"    extern __shared__ float smem[];\n"
"    float *smK = smem, *smV = smem + FA2_BKV * FA2_HD;\n"
"    int n_tiles = (N + FA2_BKV - 1) / FA2_BKV;\n"
"    int n_threads = FA2_WARPS * 32;\n"
"    for (int tile = 0; tile < n_tiles; tile++) {\n"
"        int kv_base = tile * FA2_BKV;\n"
"        for (int idx = threadIdx.x; idx < FA2_BKV * FA2_HD; idx += n_threads) {\n"
"            int kj = idx / FA2_HD, d = idx % FA2_HD;\n"
"            int kv_tok = kv_base + kj;\n"
"            smK[idx] = (kv_tok < N) ? K[(long)kv_tok * dim + h * head_dim + d] : 0.0f;\n"
"            smV[idx] = (kv_tok < N) ? V[(long)kv_tok * dim + h * head_dim + d] : 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        for (int kj = 0; kj < FA2_BKV; kj++) {\n"
"            float dot = 0.0f;\n"
"            for (int e = 0; e < FA2_EPT; e++)\n"
"                dot += q_r[e] * smK[kj * FA2_HD + lane * FA2_EPT + e];\n"
"            for (int off = 16; off > 0; off >>= 1)\n"
"                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);\n"
"            float score = (kv_base + kj < N) ? dot * scale : -1e30f;\n"
"            float new_max = fmaxf(m_i, score);\n"
"            float alpha = expf(m_i - new_max);\n"
"            float p = expf(score - new_max);\n"
"            l_i = l_i * alpha + p;\n"
"            for (int e = 0; e < FA2_EPT; e++)\n"
"                O_r[e] = O_r[e] * alpha + p * smV[kj * FA2_HD + lane * FA2_EPT + e];\n"
"            m_i = new_max;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (qi < N) {\n"
"        float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;\n"
"        for (int e = 0; e < FA2_EPT; e++) {\n"
"            int d = lane * FA2_EPT + e;\n"
"            if (d < head_dim)\n"
"                out[(long)qi * dim + h * head_dim + d] = O_r[e] * inv_l;\n"
"        }\n"
"    }\n"
"}\n"

/* ============================================================================
 * FP8 Flash Attention helper: F32 -> e4m3 quantization with shared per-tensor
 * scale. Two-pass: caller pre-zeroes max_buf, then runs reduce_max_abs_f32 to
 * compute max, then quantize_to_fp8_e4m3 to write the FP8 buffer + scale.
 * ============================================================================ */
"__global__ void zero_f32(float *p, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) p[i] = 0.0f;\n"
"}\n"

"__global__ void reduce_max_abs_f32(float *out_max, const float *in, int n) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx >= n) return;\n"
"    float v = fabsf(in[idx]);\n"
"    /* warp reduce */\n"
"    for (int off = 16; off > 0; off >>= 1) {\n"
"        float other = __shfl_xor_sync(0xFFFFFFFF, v, off);\n"
"        if (other > v) v = other;\n"
"    }\n"
"    if ((threadIdx.x & 31) == 0 && v > 0.0f) {\n"
"        /* atomicMax on int interpretation works for non-negative IEEE f32 */\n"
"        int new_int; memcpy(&new_int, &v, 4);\n"
"        atomicMax((int*)out_max, new_int);\n"
"    }\n"
"}\n"

"__global__ void quantize_to_fp8_e4m3(unsigned char *out, float *out_scale,\n"
"                                      const float *in, const float *max_buf, int n) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    float max_abs = max_buf[0];\n"
"    float scale = (max_abs > 1e-30f) ? (max_abs / 448.0f) : 1.0f;\n"
"    if (idx == 0) *out_scale = scale;\n"
"    if (idx >= n) return;\n"
"    float v = in[idx] / scale;\n"
"    unsigned short pair;\n"
"    /* d=pair, a=0, b=v -> low byte = e4m3(v), high byte = e4m3(0) */\n"
"    asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\" : \"=h\"(pair) : \"f\"(0.0f), \"f\"(v));\n"
"    out[idx] = (unsigned char)(pair & 0xFF);\n"
"}\n"

/* ============================================================================
 * FP8 Flash Attention (mma.sync m16n8k32 e4m3, sm_89+).
 * 1 warp per CTA, 16 Q rows per CTA, BKV=32 KV tile.
 * Grid: (n_heads, ceil(N/16))
 * Block: 32 threads (1 warp)
 * Shared: K[32,128] + V_T[128,32] + P[16,32 f32] = 4KB+4KB+2KB = 10KB
 *
 * Q,K,V: [N, n_heads, head_dim] uint8 e4m3 with per-tensor scales sQ, sK, sV.
 * P scale = 1/448 (P after softmax is in (0,1], use full e4m3 range).
 * ============================================================================ */
/* Scalar reference: same FP8 buffers, scalar dot products. Used as a debug
 * cross-check for the MMA version. Same launch params. */
"__global__ void flash_attn_fp8_ref(\n"
"    float *__restrict__ out,\n"
"    const unsigned char *__restrict__ Qb,\n"
"    const unsigned char *__restrict__ Kb,\n"
"    const unsigned char *__restrict__ Vb,\n"
"    int N, int n_heads, int head_dim,\n"
"    float sQ, float sK, float sV) {\n"
"    int h = blockIdx.x;\n"
"    int qi = blockIdx.y * 32 + threadIdx.x;  /* one Q row per thread */\n"
"    if (qi >= N) return;\n"
"    int dim = n_heads * head_dim;\n"
"    /* Dequantize Q row to local f32 (head_dim=128) */\n"
"    float q_loc[128];\n"
"    for (int d = 0; d < head_dim; d++)\n"
"        q_loc[d] = (float)Qb[(long)qi * dim + h * head_dim + d] * 0.0f;\n"
"    /* Above is just a placeholder; do real dequant via cvt */\n"
"    for (int d = 0; d < head_dim; d++) {\n"
"        unsigned char b = Qb[(long)qi * dim + h * head_dim + d];\n"
"        q_loc[d] = fp8e4m3_to_f32(b) * sQ;\n"
"    }\n"
"    float scale = rsqrtf((float)head_dim);\n"
"    float m_i = -1e30f, l_i = 0.0f;\n"
"    float O_loc[128];\n"
"    for (int d = 0; d < head_dim; d++) O_loc[d] = 0.0f;\n"
"    for (int kj = 0; kj < N; kj++) {\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) {\n"
"            unsigned char b = Kb[(long)kj * dim + h * head_dim + d];\n"
"            dot += q_loc[d] * (fp8e4m3_to_f32(b) * sK);\n"
"        }\n"
"        float score = dot * scale;\n"
"        float new_m = fmaxf(m_i, score);\n"
"        float alpha = expf(m_i - new_m);\n"
"        float p = expf(score - new_m);\n"
"        l_i = l_i * alpha + p;\n"
"        for (int d = 0; d < head_dim; d++) {\n"
"            unsigned char b = Vb[(long)kj * dim + h * head_dim + d];\n"
"            O_loc[d] = O_loc[d] * alpha + p * (fp8e4m3_to_f32(b) * sV);\n"
"        }\n"
"        m_i = new_m;\n"
"    }\n"
"    float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;\n"
"    for (int d = 0; d < head_dim; d++)\n"
"        out[(long)qi * dim + h * head_dim + d] = O_loc[d] * inv_l;\n"
"}\n"

"#if __CUDA_ARCH__ >= 890\n"
"#define FAFP8_BKV    32\n"
"#define FAFP8_M      16\n"
"#define FAFP8_HD    128\n"
"#define FAFP8_NWARPS 4\n"
"__global__ void flash_attn_fp8(\n"
"    float *__restrict__ out,\n"
"    const unsigned char *__restrict__ Qb,\n"
"    const unsigned char *__restrict__ Kb,\n"
"    const unsigned char *__restrict__ Vb,\n"
"    int N, int n_heads, int head_dim,\n"
"    float sQ, float sK, float sV) {\n"
"    int h = blockIdx.x;\n"
"    int warp = threadIdx.x / 32;\n"
"    int lane = threadIdx.x & 31;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane & 3;\n"
"    /* Each warp owns its own 16 Q rows (block covers 64 Q rows total) */\n"
"    int q_base = blockIdx.y * (FAFP8_M * FAFP8_NWARPS) + warp * FAFP8_M;\n"
"    int dim = n_heads * head_dim;\n"
"\n"
"    /* ----- Load Q into A-fragment registers (4 K-chunks of 32) ----- */\n"
"    unsigned int q_frag[4][4];\n"
"#pragma unroll\n"
"    for (int kc = 0; kc < 4; kc++) {\n"
"        int k0 = kc * 32;\n"
"        int row0 = q_base + gid;\n"
"        int row1 = q_base + gid + 8;\n"
"        const unsigned char *qbase0 = (row0 < N) ? Qb + (long)row0 * dim + h * head_dim : (const unsigned char *)0;\n"
"        const unsigned char *qbase1 = (row1 < N) ? Qb + (long)row1 * dim + h * head_dim : (const unsigned char *)0;\n"
"        unsigned int z = 0;\n"
"        q_frag[kc][0] = qbase0 ? *(const unsigned int *)(qbase0 + k0 + tid4*4)      : z;\n"
"#if __CUDA_ARCH__ >= 1200\n"
"        q_frag[kc][1] = qbase1 ? *(const unsigned int *)(qbase1 + k0 + tid4*4)      : z;\n"
"        q_frag[kc][2] = qbase0 ? *(const unsigned int *)(qbase0 + k0 + tid4*4 + 16) : z;\n"
"#else\n"
"        q_frag[kc][1] = qbase0 ? *(const unsigned int *)(qbase0 + k0 + tid4*4 + 16) : z;\n"
"        q_frag[kc][2] = qbase1 ? *(const unsigned int *)(qbase1 + k0 + tid4*4)      : z;\n"
"#endif\n"
"        q_frag[kc][3] = qbase1 ? *(const unsigned int *)(qbase1 + k0 + tid4*4 + 16) : z;\n"
"    }\n"
"\n"
"    /* O accumulator: 16 N-iters (head_dim=128 / N=8) × 4 floats per thread per N-iter */\n"
"    float O[16][4];\n"
"#pragma unroll\n"
"    for (int i = 0; i < 16; i++) {\n"
"        O[i][0] = 0.0f; O[i][1] = 0.0f; O[i][2] = 0.0f; O[i][3] = 0.0f;\n"
"    }\n"
"    /* Per-row state (each thread has 2 row groups: gid, gid+8) */\n"
"    float m_state[2] = {-1e30f, -1e30f};\n"
"    float l_state[2] = {0.0f, 0.0f};\n"
"\n"
"    float qk_scale = sQ * sK * rsqrtf((float)head_dim);\n"
"\n"
"    extern __shared__ unsigned char fa_smem[];\n"
"    unsigned char *smK  = fa_smem;                                               /* [BKV, hd] */\n"
"    unsigned char *smVT = smK + FAFP8_BKV * FAFP8_HD;                            /* [hd, BKV] */\n"
"    float *smP_all = (float*)(smVT + FAFP8_HD * FAFP8_BKV);                      /* [NWARPS * 16 * BKV] */\n"
"    float *smP = smP_all + warp * 16 * FAFP8_BKV;                                /* this warp's slice */\n"
"\n"
"    int n_kv_tiles = (N + FAFP8_BKV - 1) / FAFP8_BKV;\n"
"    for (int kvt = 0; kvt < n_kv_tiles; kvt++) {\n"
"        int kv_base = kvt * FAFP8_BKV;\n"
"\n"
"        /* Cooperative load: K row-major [BKV, hd], V transposed [hd, BKV]. All 128 threads. */\n"
"        for (int idx = threadIdx.x; idx < FAFP8_BKV * FAFP8_HD; idx += FAFP8_NWARPS * 32) {\n"
"            int kr = idx / FAFP8_HD, kd = idx % FAFP8_HD;\n"
"            int row = kv_base + kr;\n"
"            unsigned char kk = (row < N) ? Kb[(long)row * dim + h * head_dim + kd] : 0;\n"
"            unsigned char vv = (row < N) ? Vb[(long)row * dim + h * head_dim + kd] : 0;\n"
"            smK[kr * FAFP8_HD + kd] = kk;\n"
"            smVT[kd * FAFP8_BKV + kr] = vv;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* S = Q · K^T : [16, BKV=32]. 4 N-iters of m16n8k32, 4 K-chunks per. */\n"
"        float S[4][4];\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < 4; ni++) {\n"
"            S[ni][0] = 0.0f; S[ni][1] = 0.0f; S[ni][2] = 0.0f; S[ni][3] = 0.0f;\n"
"        }\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < 4; ni++) {\n"
"            int n_base = ni * 8;\n"
"            int b_col = n_base + gid;  /* col of mma B = row of K in smK */\n"
"#pragma unroll\n"
"            for (int kc = 0; kc < 4; kc++) {\n"
"                int k_off = kc * 32;\n"
"                unsigned int b0 = 0, b1 = 0;\n"
"                if (b_col < FAFP8_BKV) {\n"
"                    const unsigned char *kp = smK + b_col * FAFP8_HD + k_off;\n"
"                    b0 = *(const unsigned int *)(kp + tid4 * 4);\n"
"                    b1 = *(const unsigned int *)(kp + tid4 * 4 + 16);\n"
"                }\n"
"                asm volatile(\n"
"                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    : \"=f\"(S[ni][0]), \"=f\"(S[ni][1]), \"=f\"(S[ni][2]), \"=f\"(S[ni][3])\n"
"                    : \"r\"(q_frag[kc][0]), \"r\"(q_frag[kc][1]), \"r\"(q_frag[kc][2]), \"r\"(q_frag[kc][3]),\n"
"                      \"r\"(b0), \"r\"(b1),\n"
"                      \"f\"(S[ni][0]), \"f\"(S[ni][1]), \"f\"(S[ni][2]), \"f\"(S[ni][3])\n"
"                );\n"
"            }\n"
"        }\n"
"\n"
"        /* Apply qk_scale and mask out-of-bounds KV cols. D-fragment col mapping:\n"
"         * d0=col tid4*2, d1=col tid4*2+1, d2=col tid4*2, d3=col tid4*2+1 (rows differ). */\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < 4; ni++) {\n"
"            int col0 = ni * 8 + tid4 * 2;\n"
"            int col1 = col0 + 1;\n"
"            int kvr0 = kv_base + col0;\n"
"            int kvr1 = kv_base + col1;\n"
"            S[ni][0] = (kvr0 < N) ? S[ni][0] * qk_scale : -1e30f;\n"
"            S[ni][1] = (kvr1 < N) ? S[ni][1] * qk_scale : -1e30f;\n"
"            S[ni][2] = (kvr0 < N) ? S[ni][2] * qk_scale : -1e30f;\n"
"            S[ni][3] = (kvr1 < N) ? S[ni][3] * qk_scale : -1e30f;\n"
"        }\n"
"\n"
"        /* Online softmax: per-row max */\n"
"        float new_m[2], alpha[2];\n"
"#pragma unroll\n"
"        for (int rp = 0; rp < 2; rp++) {\n"
"            float my = -1e30f;\n"
"#pragma unroll\n"
"            for (int ni = 0; ni < 4; ni++) {\n"
"                my = fmaxf(my, S[ni][rp*2 + 0]);\n"
"                my = fmaxf(my, S[ni][rp*2 + 1]);\n"
"            }\n"
"            /* Reduce across the 4 lanes that share gid (xor 1, 2 over tid4) */\n"
"            float o1 = __shfl_xor_sync(0xFFFFFFFF, my, 1); my = fmaxf(my, o1);\n"
"            float o2 = __shfl_xor_sync(0xFFFFFFFF, my, 2); my = fmaxf(my, o2);\n"
"            new_m[rp] = fmaxf(m_state[rp], my);\n"
"            alpha[rp] = expf(m_state[rp] - new_m[rp]);\n"
"        }\n"
"\n"
"        /* Rescale O and l */\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < 16; ni++) {\n"
"            O[ni][0] *= alpha[0]; O[ni][1] *= alpha[0];\n"
"            O[ni][2] *= alpha[1]; O[ni][3] *= alpha[1];\n"
"        }\n"
"        l_state[0] *= alpha[0]; l_state[1] *= alpha[1];\n"
"\n"
"        /* P = exp(S - new_m). Scale up by 448 so e4m3 conversion uses full range. */\n"
"        float P_data[4][4];\n"
"        float row_sum[2] = {0.0f, 0.0f};\n"
"#pragma unroll\n"
"        for (int rp = 0; rp < 2; rp++) {\n"
"#pragma unroll\n"
"            for (int ni = 0; ni < 4; ni++) {\n"
"                float p0 = expf(S[ni][rp*2 + 0] - new_m[rp]);\n"
"                float p1 = expf(S[ni][rp*2 + 1] - new_m[rp]);\n"
"                P_data[ni][rp*2 + 0] = p0;\n"
"                P_data[ni][rp*2 + 1] = p1;\n"
"                row_sum[rp] += p0 + p1;\n"
"            }\n"
"        }\n"
"        float r0a = __shfl_xor_sync(0xFFFFFFFF, row_sum[0], 1); row_sum[0] += r0a;\n"
"        float r0b = __shfl_xor_sync(0xFFFFFFFF, row_sum[0], 2); row_sum[0] += r0b;\n"
"        float r1a = __shfl_xor_sync(0xFFFFFFFF, row_sum[1], 1); row_sum[1] += r1a;\n"
"        float r1b = __shfl_xor_sync(0xFFFFFFFF, row_sum[1], 2); row_sum[1] += r1b;\n"
"        l_state[0] += row_sum[0]; l_state[1] += row_sum[1];\n"
"        m_state[0] = new_m[0]; m_state[1] = new_m[1];\n"
"\n"
"        /* Stash P (scaled by 448) into smP for re-read as A-fragment */\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < 4; ni++) {\n"
"            int col0 = ni * 8 + tid4 * 2;\n"
"            int col1 = col0 + 1;\n"
"            smP[gid * FAFP8_BKV + col0]     = P_data[ni][0] * 448.0f;\n"
"            smP[gid * FAFP8_BKV + col1]     = P_data[ni][1] * 448.0f;\n"
"            smP[(gid+8) * FAFP8_BKV + col0] = P_data[ni][2] * 448.0f;\n"
"            smP[(gid+8) * FAFP8_BKV + col1] = P_data[ni][3] * 448.0f;\n"
"        }\n"
"        __syncwarp();  /* smP is per-warp, only warp-local barrier needed */\n"
"\n"
"        /* Convert P to FP8 A-fragment (1 K-chunk: BKV=32 = K) */\n"
"        unsigned int p_frag[4];\n"
"        {\n"
"            int row0 = gid;\n"
"            int row1 = gid + 8;\n"
"#define CVTPF(reg, row, c) { \\\n"
"                unsigned short lo, hi; \\\n"
"                asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(lo) : \\\n"
"                    \"f\"(smP[(row)*FAFP8_BKV + (c)]), \"f\"(smP[(row)*FAFP8_BKV + (c)+1])); \\\n"
"                asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(hi) : \\\n"
"                    \"f\"(smP[(row)*FAFP8_BKV + (c)+2]), \"f\"(smP[(row)*FAFP8_BKV + (c)+3])); \\\n"
"                reg = (unsigned int)lo | ((unsigned int)hi << 16); }\n"
"            CVTPF(p_frag[0], row0, tid4*4)\n"
"#if __CUDA_ARCH__ >= 1200\n"
"            CVTPF(p_frag[1], row1, tid4*4)\n"
"            CVTPF(p_frag[2], row0, tid4*4 + 16)\n"
"#else\n"
"            CVTPF(p_frag[1], row0, tid4*4 + 16)\n"
"            CVTPF(p_frag[2], row1, tid4*4)\n"
"#endif\n"
"            CVTPF(p_frag[3], row1, tid4*4 + 16)\n"
"#undef CVTPF\n"
"        }\n"
"\n"
"        /* O += P · V via mma. 16 N-iters of m16n8 (head_dim=128/8). 1 K-iter (BKV=32). */\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < 16; ni++) {\n"
"            int n_base = ni * 8;\n"
"            int b_col = n_base + gid;  /* head_dim col */\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (b_col < head_dim) {\n"
"                const unsigned char *vp = smVT + b_col * FAFP8_BKV;\n"
"                b0 = *(const unsigned int *)(vp + tid4 * 4);\n"
"                b1 = *(const unsigned int *)(vp + tid4 * 4 + 16);\n"
"            }\n"
"            float d0 = O[ni][0], d1 = O[ni][1], d2 = O[ni][2], d3 = O[ni][3];\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0), \"=f\"(d1), \"=f\"(d2), \"=f\"(d3)\n"
"                : \"r\"(p_frag[0]), \"r\"(p_frag[1]), \"r\"(p_frag[2]), \"r\"(p_frag[3]),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0), \"f\"(d1), \"f\"(d2), \"f\"(d3)\n"
"            );\n"
"            O[ni][0] = d0; O[ni][1] = d1; O[ni][2] = d2; O[ni][3] = d3;\n"
"        }\n"
"        __syncthreads();  /* before next tile overwrites shared smK/smVT */\n"
"    }\n"
"\n"
"    /* Final write: descale by sV/448 (P scale was 1/448), divide by l_state */\n"
"    int row0 = q_base + gid;\n"
"    int row1 = q_base + gid + 8;\n"
"    float inv_l0 = (l_state[0] > 0.0f) ? 1.0f / l_state[0] : 0.0f;\n"
"    float inv_l1 = (l_state[1] > 0.0f) ? 1.0f / l_state[1] : 0.0f;\n"
"    float fac = sV * (1.0f / 448.0f);\n"
"#pragma unroll\n"
"    for (int ni = 0; ni < 16; ni++) {\n"
"        int col0 = ni * 8 + tid4 * 2;\n"
"        int col1 = col0 + 1;\n"
"        if (row0 < N && col0 < head_dim)\n"
"            out[(long)row0 * dim + h * head_dim + col0] = O[ni][0] * fac * inv_l0;\n"
"        if (row0 < N && col1 < head_dim)\n"
"            out[(long)row0 * dim + h * head_dim + col1] = O[ni][1] * fac * inv_l0;\n"
"        if (row1 < N && col0 < head_dim)\n"
"            out[(long)row1 * dim + h * head_dim + col0] = O[ni][2] * fac * inv_l1;\n"
"        if (row1 < N && col1 < head_dim)\n"
"            out[(long)row1 * dim + h * head_dim + col1] = O[ni][3] * fac * inv_l1;\n"
"    }\n"
"}\n"
"#endif\n"
"__device__ __forceinline__ unsigned short f32_to_bf16_bits(float f) {\n"
"    unsigned int b; memcpy(&b, &f, 4);\n"
"    if (((b >> 23) & 0xFF) == 0xFF && (b & 0x7FFFFF)) return 0x7FC0;\n"
"    unsigned int r = 0x7FFFu + ((b >> 16) & 1u);\n"
"    return (unsigned short)((b + r) >> 16);\n"
"}\n"
"#if __CUDA_ARCH__ >= 800\n"
"#define FABF16_BKV    32\n"
"#define FABF16_M      16\n"
"#define FABF16_HD    128\n"
"/* Pad SMEM row stride: (HDP/2) mod 32 = 4 → distinct quad of banks per row,\n"
" * eliminates 8-way LDS-32 bank conflicts on Q@K^T scalar B-frag and on the\n"
" * P@V ldmatrix.x4.trans source. Same trick that took cuda/fa v3 → v3.2 1.43x. */\n"
"#define FABF16_HDP   136\n"
"#define FABF16_NWARPS 4\n"
"#define FABF16_NI     (FABF16_BKV / 8)   /* Q.K^T N-tiles per outer iter */\n"
"#define FABF16_PV_KC  (FABF16_BKV / 16)  /* P.V K-chunks per N-iter */\n"
"__global__ void flash_attn_bf16(\n"
"    float *__restrict__ out,\n"
"    const unsigned short *__restrict__ Qb,\n"
"    const unsigned short *__restrict__ Kb,\n"
"    const unsigned short *__restrict__ Vb,\n"
"    int N, int n_heads, int head_dim) {\n"
"    int h = blockIdx.x;\n"
"    int warp = threadIdx.x / 32;\n"
"    int lane = threadIdx.x & 31;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane & 3;\n"
"    int q_base = blockIdx.y * (FABF16_M * FABF16_NWARPS) + warp * FABF16_M;\n"
"    int dim = n_heads * head_dim;\n"
"\n"
"    /* ----- Load Q into A-fragment registers. 8 K-chunks of k=16 BF16 each. ----- */\n"
"    /* Per lane per chunk: 4 u32 = 8 BF16 = 4 (row, col-pair) entries. */\n"
"    unsigned int q_frag[8][4];\n"
"#pragma unroll\n"
"    for (int kc = 0; kc < 8; kc++) {\n"
"        int k0 = kc * 16;\n"
"        int row0 = q_base + gid;\n"
"        int row1 = q_base + gid + 8;\n"
"        const unsigned short *qbase0 = (row0 < N) ? Qb + (long)row0 * dim + h * head_dim : (const unsigned short *)0;\n"
"        const unsigned short *qbase1 = (row1 < N) ? Qb + (long)row1 * dim + h * head_dim : (const unsigned short *)0;\n"
"        unsigned int z = 0;\n"
"        q_frag[kc][0] = qbase0 ? *(const unsigned int *)(qbase0 + k0 + tid4*2)     : z;\n"
"#if __CUDA_ARCH__ >= 1200\n"
"        q_frag[kc][1] = qbase1 ? *(const unsigned int *)(qbase1 + k0 + tid4*2)     : z;\n"
"        q_frag[kc][2] = qbase0 ? *(const unsigned int *)(qbase0 + k0 + tid4*2 + 8) : z;\n"
"#else\n"
"        q_frag[kc][1] = qbase0 ? *(const unsigned int *)(qbase0 + k0 + tid4*2 + 8) : z;\n"
"        q_frag[kc][2] = qbase1 ? *(const unsigned int *)(qbase1 + k0 + tid4*2)     : z;\n"
"#endif\n"
"        q_frag[kc][3] = qbase1 ? *(const unsigned int *)(qbase1 + k0 + tid4*2 + 8) : z;\n"
"    }\n"
"\n"
"    /* O accumulator: 16 N-iters (head_dim/8) x 4 floats per thread per N-iter */\n"
"    float O[16][4];\n"
"#pragma unroll\n"
"    for (int i = 0; i < 16; i++) {\n"
"        O[i][0] = 0.0f; O[i][1] = 0.0f; O[i][2] = 0.0f; O[i][3] = 0.0f;\n"
"    }\n"
"    float m_state[2] = {-1e30f, -1e30f};\n"
"    float l_state[2] = {0.0f, 0.0f};\n"
"\n"
"    float qk_scale = rsqrtf((float)head_dim);\n"
"\n"
"    extern __shared__ unsigned char fa_bf16_smem[];\n"
"    /* Double-buffered layout: [sK0][sK1][sV0][sV1] each 32 x HDP bf16 ~ 8.5 KB.\n"
"     * Total ~34 KB (under sm_120 default 48 KB dynamic SMEM cap). Stride padded\n"
"     * to 136 for bank-conflict-free LDS. P stays in per-lane registers. */\n"
"    unsigned short *sK_buf[2];\n"
"    unsigned short *sV_buf[2];\n"
"    sK_buf[0] = (unsigned short *)fa_bf16_smem;\n"
"    sK_buf[1] = sK_buf[0] + FABF16_BKV * FABF16_HDP;\n"
"    sV_buf[0] = sK_buf[1] + FABF16_BKV * FABF16_HDP;\n"
"    sV_buf[1] = sV_buf[0] + FABF16_BKV * FABF16_HDP;\n"
"\n"
"    int n_kv_tiles = (N + FABF16_BKV - 1) / FABF16_BKV;\n"
"\n"
"    /* Prologue: cp.async tile 0 into buffer 0. */\n"
"    {\n"
"        unsigned short *_sK = sK_buf[0];\n"
"        unsigned short *_sV = sV_buf[0];\n"
"#pragma unroll\n"
"        for (int _i = 0; _i < 4; _i++) {\n"
"            int _idx = (int)threadIdx.x + _i * 128;\n"
"            int _kr  = _idx >> 4;\n"
"            int _kd  = (_idx & 15) * 8;\n"
"            int _sr  = (_kr < N) ? _kr : 0;\n"
"            const unsigned short *_kp = Kb + (long)_sr * dim + h * head_dim + _kd;\n"
"            const unsigned short *_vp = Vb + (long)_sr * dim + h * head_dim + _kd;\n"
"            unsigned int _dk = (unsigned int)__cvta_generic_to_shared(_sK + _kr * FABF16_HDP + _kd);\n"
"            unsigned int _dv = (unsigned int)__cvta_generic_to_shared(_sV + _kr * FABF16_HDP + _kd);\n"
"            asm volatile(\"cp.async.ca.shared.global [%0], [%1], 16;\\n\" :: \"r\"(_dk), \"l\"(_kp));\n"
"            asm volatile(\"cp.async.ca.shared.global [%0], [%1], 16;\\n\" :: \"r\"(_dv), \"l\"(_vp));\n"
"        }\n"
"    }\n"
"    asm volatile(\"cp.async.commit_group;\\n\");\n"
"\n"
"    for (int kvt = 0; kvt < n_kv_tiles; kvt++) {\n"
"        int kv_base = kvt * FABF16_BKV;\n"
"        int cur = kvt & 1;\n"
"        int nxt = 1 - cur;\n"
"        unsigned short *smK = sK_buf[cur];\n"
"        unsigned short *smV = sV_buf[cur];\n"
"\n"
"        /* Issue cp.async for tile (kvt+1) into buffer nxt, then wait on current. */\n"
"        if (kvt + 1 < n_kv_tiles) {\n"
"            int kv_next = (kvt + 1) * FABF16_BKV;\n"
"            unsigned short *_sK = sK_buf[nxt];\n"
"            unsigned short *_sV = sV_buf[nxt];\n"
"#pragma unroll\n"
"            for (int _i = 0; _i < 4; _i++) {\n"
"                int _idx = (int)threadIdx.x + _i * 128;\n"
"                int _kr  = _idx >> 4;\n"
"                int _kd  = (_idx & 15) * 8;\n"
"                int _raw = kv_next + _kr;\n"
"                int _sr  = (_raw < N) ? _raw : 0;\n"
"                const unsigned short *_kp = Kb + (long)_sr * dim + h * head_dim + _kd;\n"
"                const unsigned short *_vp = Vb + (long)_sr * dim + h * head_dim + _kd;\n"
"                unsigned int _dk = (unsigned int)__cvta_generic_to_shared(_sK + _kr * FABF16_HDP + _kd);\n"
"                unsigned int _dv = (unsigned int)__cvta_generic_to_shared(_sV + _kr * FABF16_HDP + _kd);\n"
"                asm volatile(\"cp.async.ca.shared.global [%0], [%1], 16;\\n\" :: \"r\"(_dk), \"l\"(_kp));\n"
"                asm volatile(\"cp.async.ca.shared.global [%0], [%1], 16;\\n\" :: \"r\"(_dv), \"l\"(_vp));\n"
"            }\n"
"            asm volatile(\"cp.async.commit_group;\\n\");\n"
"            asm volatile(\"cp.async.wait_group 1;\\n\");\n"
"        } else {\n"
"            asm volatile(\"cp.async.wait_group 0;\\n\");\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* S = Q . K^T : [16, BKV]. NI N-iters of m16n8k16, 8 K-chunks per. */\n"
"        float S[FABF16_NI][4];\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < FABF16_NI; ni++) {\n"
"            S[ni][0] = 0.0f; S[ni][1] = 0.0f; S[ni][2] = 0.0f; S[ni][3] = 0.0f;\n"
"        }\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < FABF16_NI; ni++) {\n"
"            int n_base = ni * 8;\n"
"            int b_col = n_base + gid;  /* K row index in smK */\n"
"#pragma unroll\n"
"            for (int kc = 0; kc < 8; kc++) {\n"
"                int k_off = kc * 16;\n"
"                unsigned int b0 = 0, b1 = 0;\n"
"                if (b_col < FABF16_BKV) {\n"
"                    const unsigned short *kp = smK + b_col * FABF16_HDP + k_off;\n"
"                    b0 = *(const unsigned int *)(kp + tid4 * 2);\n"
"                    b1 = *(const unsigned int *)(kp + tid4 * 2 + 8);\n"
"                }\n"
"                asm volatile(\n"
"                    \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    : \"=f\"(S[ni][0]), \"=f\"(S[ni][1]), \"=f\"(S[ni][2]), \"=f\"(S[ni][3])\n"
"                    : \"r\"(q_frag[kc][0]), \"r\"(q_frag[kc][1]), \"r\"(q_frag[kc][2]), \"r\"(q_frag[kc][3]),\n"
"                      \"r\"(b0), \"r\"(b1),\n"
"                      \"f\"(S[ni][0]), \"f\"(S[ni][1]), \"f\"(S[ni][2]), \"f\"(S[ni][3])\n"
"                );\n"
"            }\n"
"        }\n"
"\n"
"        /* Apply qk_scale and mask out-of-bounds KV cols. */\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < FABF16_NI; ni++) {\n"
"            int col0 = ni * 8 + tid4 * 2;\n"
"            int col1 = col0 + 1;\n"
"            int kvr0 = kv_base + col0;\n"
"            int kvr1 = kv_base + col1;\n"
"            S[ni][0] = (kvr0 < N) ? S[ni][0] * qk_scale : -1e30f;\n"
"            S[ni][1] = (kvr1 < N) ? S[ni][1] * qk_scale : -1e30f;\n"
"            S[ni][2] = (kvr0 < N) ? S[ni][2] * qk_scale : -1e30f;\n"
"            S[ni][3] = (kvr1 < N) ? S[ni][3] * qk_scale : -1e30f;\n"
"        }\n"
"\n"
"        /* Online softmax: per-row max */\n"
"        float new_m[2], alpha[2];\n"
"#pragma unroll\n"
"        for (int rp = 0; rp < 2; rp++) {\n"
"            float my = -1e30f;\n"
"#pragma unroll\n"
"            for (int ni = 0; ni < FABF16_NI; ni++) {\n"
"                my = fmaxf(my, S[ni][rp*2 + 0]);\n"
"                my = fmaxf(my, S[ni][rp*2 + 1]);\n"
"            }\n"
"            float o1 = __shfl_xor_sync(0xFFFFFFFF, my, 1); my = fmaxf(my, o1);\n"
"            float o2 = __shfl_xor_sync(0xFFFFFFFF, my, 2); my = fmaxf(my, o2);\n"
"            new_m[rp] = fmaxf(m_state[rp], my);\n"
"            alpha[rp] = expf(m_state[rp] - new_m[rp]);\n"
"        }\n"
"\n"
"        /* Rescale O and l */\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < 16; ni++) {\n"
"            O[ni][0] *= alpha[0]; O[ni][1] *= alpha[0];\n"
"            O[ni][2] *= alpha[1]; O[ni][3] *= alpha[1];\n"
"        }\n"
"        l_state[0] *= alpha[0]; l_state[1] *= alpha[1];\n"
"\n"
"        /* P = exp(S - new_m). No 448 scaling -- BF16 has F32 dynamic range. */\n"
"        float P_data[FABF16_NI][4];\n"
"        float row_sum[2] = {0.0f, 0.0f};\n"
"#pragma unroll\n"
"        for (int rp = 0; rp < 2; rp++) {\n"
"#pragma unroll\n"
"            for (int ni = 0; ni < FABF16_NI; ni++) {\n"
"                float p0 = expf(S[ni][rp*2 + 0] - new_m[rp]);\n"
"                float p1 = expf(S[ni][rp*2 + 1] - new_m[rp]);\n"
"                P_data[ni][rp*2 + 0] = p0;\n"
"                P_data[ni][rp*2 + 1] = p1;\n"
"                row_sum[rp] += p0 + p1;\n"
"            }\n"
"        }\n"
"        float r0a = __shfl_xor_sync(0xFFFFFFFF, row_sum[0], 1); row_sum[0] += r0a;\n"
"        float r0b = __shfl_xor_sync(0xFFFFFFFF, row_sum[0], 2); row_sum[0] += r0b;\n"
"        float r1a = __shfl_xor_sync(0xFFFFFFFF, row_sum[1], 1); row_sum[1] += r1a;\n"
"        float r1b = __shfl_xor_sync(0xFFFFFFFF, row_sum[1], 2); row_sum[1] += r1b;\n"
"        l_state[0] += row_sum[0]; l_state[1] += row_sum[1];\n"
"        m_state[0] = new_m[0]; m_state[1] = new_m[1];\n"
"\n"
"        /* Pre-pack P (F32 -> BF16 bits) into per-lane u32 registers.\n"
"         * Each lane already holds all 4 ni's of P_data in registers, so the\n"
"         * mma A-operand can be built directly without a smP shared-memory\n"
"         * round-trip. Under sm_120 the a1<->a2 PTX slot swap applies, same as\n"
"         * the smP-based path we replaced.\n"
"         * P_pack[kc][slot] holds the 4 a-regs for each PV k-chunk. */\n"
"        unsigned int P_pack[FABF16_PV_KC][4];\n"
"#pragma unroll\n"
"        for (int kc = 0; kc < FABF16_PV_KC; kc++) {\n"
"            int ni_a = 2*kc;      /* ni for 'first half' of the 16-col span */\n"
"            int ni_b = 2*kc + 1;  /* ni for 'second half' */\n"
"            unsigned int q00 = (unsigned int)f32_to_bf16_bits(P_data[ni_a][0])\n"
"                             | ((unsigned int)f32_to_bf16_bits(P_data[ni_a][1]) << 16);\n"
"            unsigned int q01 = (unsigned int)f32_to_bf16_bits(P_data[ni_a][2])\n"
"                             | ((unsigned int)f32_to_bf16_bits(P_data[ni_a][3]) << 16);\n"
"            unsigned int q10 = (unsigned int)f32_to_bf16_bits(P_data[ni_b][0])\n"
"                             | ((unsigned int)f32_to_bf16_bits(P_data[ni_b][1]) << 16);\n"
"            unsigned int q11 = (unsigned int)f32_to_bf16_bits(P_data[ni_b][2])\n"
"                             | ((unsigned int)f32_to_bf16_bits(P_data[ni_b][3]) << 16);\n"
"            /* Standard PTX layout: a0=(gid,c), a1=(gid,c+8), a2=(gid+8,c), a3=(gid+8,c+8).\n"
"             * sm_120 swaps a1<->a2. */\n"
"            P_pack[kc][0] = q00;\n"
"#if __CUDA_ARCH__ >= 1200\n"
"            P_pack[kc][1] = q01;\n"
"            P_pack[kc][2] = q10;\n"
"#else\n"
"            P_pack[kc][1] = q10;\n"
"            P_pack[kc][2] = q01;\n"
"#endif\n"
"            P_pack[kc][3] = q11;\n"
"        }\n"
"\n"
"        /* O += P . V via mma. 16 N-iters of m16n8 (head_dim/8). Per ni we load a\n"
"         * 32-kv x 8-n block of V via one ldmatrix.x4.trans from row-major smV.\n"
"         * The 4 result regs correspond to mat0=kv[0..7], mat1=kv[8..15],\n"
"         * mat2=kv[16..23], mat3=kv[24..31] (8 cols of the same n-window each).\n"
"         * After .trans, each reg already has the {k0,k1} packed per-lane that\n"
"         * m16n8k16's B-operand wants, matching the former smVT col-major layout. */\n"
"#pragma unroll\n"
"        for (int ni = 0; ni < 16; ni++) {\n"
"            int n_base = ni * 8;\n"
"            unsigned int v_addr = (unsigned int)__cvta_generic_to_shared(\n"
"                smV + lane * FABF16_HDP + n_base);\n"
"            unsigned int vb0, vb1, vb2, vb3;\n"
"            asm(\"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"
"                : \"=r\"(vb0), \"=r\"(vb1), \"=r\"(vb2), \"=r\"(vb3)\n"
"                : \"r\"(v_addr));\n"
"#pragma unroll\n"
"            for (int kc = 0; kc < FABF16_PV_KC; kc++) {\n"
"                unsigned int p0 = P_pack[kc][0];\n"
"                unsigned int p1 = P_pack[kc][1];\n"
"                unsigned int p2 = P_pack[kc][2];\n"
"                unsigned int p3 = P_pack[kc][3];\n"
"                unsigned int b0 = (kc == 0) ? vb0 : vb2;\n"
"                unsigned int b1 = (kc == 0) ? vb1 : vb3;\n"
"                float d0 = O[ni][0], d1 = O[ni][1], d2 = O[ni][2], d3 = O[ni][3];\n"
"                asm volatile(\n"
"                    \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    : \"=f\"(d0), \"=f\"(d1), \"=f\"(d2), \"=f\"(d3)\n"
"                    : \"r\"(p0), \"r\"(p1), \"r\"(p2), \"r\"(p3),\n"
"                      \"r\"(b0), \"r\"(b1),\n"
"                      \"f\"(d0), \"f\"(d1), \"f\"(d2), \"f\"(d3)\n"
"                );\n"
"                O[ni][0] = d0; O[ni][1] = d1; O[ni][2] = d2; O[ni][3] = d3;\n"
"            }\n"
"        }\n"
"        __syncthreads();  /* before next tile overwrites smK/smV */\n"
"    }\n"
"\n"
"    /* Final write: divide by l_state. No descale -- no 448 scaling. */\n"
"    int row0 = q_base + gid;\n"
"    int row1 = q_base + gid + 8;\n"
"    float inv_l0 = (l_state[0] > 0.0f) ? 1.0f / l_state[0] : 0.0f;\n"
"    float inv_l1 = (l_state[1] > 0.0f) ? 1.0f / l_state[1] : 0.0f;\n"
"#pragma unroll\n"
"    for (int ni = 0; ni < 16; ni++) {\n"
"        int col0 = ni * 8 + tid4 * 2;\n"
"        int col1 = col0 + 1;\n"
"        if (row0 < N && col0 < head_dim)\n"
"            out[(long)row0 * dim + h * head_dim + col0] = O[ni][0] * inv_l0;\n"
"        if (row0 < N && col1 < head_dim)\n"
"            out[(long)row0 * dim + h * head_dim + col1] = O[ni][1] * inv_l0;\n"
"        if (row1 < N && col0 < head_dim)\n"
"            out[(long)row1 * dim + h * head_dim + col0] = O[ni][2] * inv_l1;\n"
"        if (row1 < N && col1 < head_dim)\n"
"            out[(long)row1 * dim + h * head_dim + col1] = O[ni][3] * inv_l1;\n"
"    }\n"
"}\n"
"\n"
"/* Bulk F32 -> BF16 cast kernel for converting Q/K/V before flash_attn_bf16. */\n"
"__global__ void cast_f32_to_bf16(const float *src, unsigned short *dst, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    dst[i] = f32_to_bf16_bits(src[i]);\n"
"}\n"
"#endif\n"

/* =========================================================================
 * gemm_fp8_v7: ported verbatim from cuda/gemm/cuda_gemm_ptx_kernels.h.
 * Computes Y[M,N] = X[M,K] @ W[N,K]^T where X and W are FP8 e4m3 row-major.
 * Y is raw F32 (no scaling, no bias). Caller must descale.
 * Constraint: K must be a multiple of 64.
 * Grid: (ceil(N/128), ceil(M/64), 1), block 256, smem 24 KiB.
 * ========================================================================= */
"typedef unsigned char fp8_raw;\n"
"extern \"C\" __global__ void gemm_fp8_v7(float *Y,\n"
"                                          const fp8_raw *X,\n"
"                                          const fp8_raw *W,\n"
"                                          int M, int N, int K) {\n"
"    extern __shared__ __align__(16) unsigned short smem_v7fp8[];\n"
"    unsigned short *sA0 = smem_v7fp8;\n"
"    unsigned short *sB0 = smem_v7fp8 + 2048;\n"
"    unsigned short *sA1 = smem_v7fp8 + 6144;\n"
"    unsigned short *sB1 = smem_v7fp8 + 8192;\n"
"    int tid = threadIdx.x;\n"
"    int wid = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int gid  = lane >> 2;\n"
"    int tid4 = lane & 3;\n"
"    int warp_m = wid >> 1;\n"
"    int warp_n = wid & 1;\n"
"    int blk_lin = blockIdx.y * gridDim.x + blockIdx.x;\n"
"    int npx_ = gridDim.x;\n"
"    int npy_ = gridDim.y;\n"
"    int panels_m_ = (npy_ + 3) >> 2;\n"
"    int per_panel_ = 16;\n"
"    int panel_id_ = blk_lin / per_panel_;\n"
"    int in_panel_ = blk_lin - panel_id_ * per_panel_;\n"
"    int panel_m_id_ = panel_id_ - (panel_id_ / panels_m_) * panels_m_;\n"
"    int panel_n_id_ = panel_id_ / panels_m_;\n"
"    int m_in_ = in_panel_ & 3;\n"
"    int n_in_ = in_panel_ >> 2;\n"
"    int tile_m_ = (panel_m_id_ << 2) + m_in_;\n"
"    int tile_n_ = (panel_n_id_ << 2) + n_in_;\n"
"    if (tile_m_ >= npy_ || tile_n_ >= npx_) return;\n"
"    int cta_m  = tile_m_ * 64;\n"
"    int cta_n  = tile_n_ * 128;\n"
"    int wm_row = warp_m * 16;\n"
"    int wn_col = warp_n * 64;\n"
"    if (cta_m >= M) return;\n"
"    float d0[8], d1[8], d2[8], d3[8];\n"
"    #pragma unroll\n"
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"    int row_a    = tid >> 2;\n"
"    int col_a_b  = (tid & 3) * 8;\n"
"    int col_a_g  = (tid & 3) * 16;\n"
"    int g_row_a  = cta_m + row_a;\n"
"    int sw_a     = (row_a & 3) * 8;\n"
"    int swz_a    = row_a * 32 + (col_a_b ^ sw_a);\n"
"    int row_b0   = tid >> 2;\n"
"    int col_b0_b = (tid & 3) * 8;\n"
"    int col_b0_g = (tid & 3) * 16;\n"
"    int g_row_b0 = cta_n + row_b0;\n"
"    int sw_b0    = (row_b0 & 3) * 8;\n"
"    int swz_b0   = row_b0 * 32 + (col_b0_b ^ sw_b0);\n"
"    int vid_b1   = tid + 256;\n"
"    int row_b1   = vid_b1 >> 2;\n"
"    int col_b1_b = (vid_b1 & 3) * 8;\n"
"    int col_b1_g = (vid_b1 & 3) * 16;\n"
"    int g_row_b1 = cta_n + row_b1;\n"
"    int sw_b1    = (row_b1 & 3) * 8;\n"
"    int swz_b1   = row_b1 * 32 + (col_b1_b ^ sw_b1);\n"
"    {\n"
"        unsigned int dA = __cvta_generic_to_shared(&sA0[swz_a]);\n"
"        if (g_row_a < M) {\n"
"            const fp8_raw *src = &X[(size_t)g_row_a * K + 0 + col_a_g];\n"
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"
"        unsigned int dB0 = __cvta_generic_to_shared(&sB0[swz_b0]);\n"
"        if (g_row_b0 < N) {\n"
"            const fp8_raw *src = &W[(size_t)g_row_b0 * K + 0 + col_b0_g];\n"
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"
"        unsigned int dB1 = __cvta_generic_to_shared(&sB0[swz_b1]);\n"
"        if (g_row_b1 < N) {\n"
"            const fp8_raw *src = &W[(size_t)g_row_b1 * K + 0 + col_b1_g];\n"
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"
"    }\n"
"    asm volatile(\"cp.async.commit_group;\\n\");\n"
"    int num_k = K >> 6;\n"
"    int la_row     = wm_row + (lane & 15);\n"
"    int la_col_off = (lane >> 4) * 8;\n"
"    int la_sw      = (la_row & 3) * 8;\n"
"    int lb_row     = lane & 15;\n"
"    int lb_col_off = (lane >> 4) * 8;\n"
"    for (int ki = 0; ki < num_k; ki++) {\n"
"        int stage = ki & 1;\n"
"        int next_stage = stage ^ 1;\n"
"        int next_k = (ki + 1) << 6;\n"
"        asm volatile(\"cp.async.wait_group 0;\\n\");\n"
"        __syncthreads();\n"
"        if (next_k < K) {\n"
"            unsigned short *sA_next = (next_stage == 0) ? sA0 : sA1;\n"
"            unsigned short *sB_next = (next_stage == 0) ? sB0 : sB1;\n"
"            unsigned int dA = __cvta_generic_to_shared(&sA_next[swz_a]);\n"
"            if (g_row_a < M) {\n"
"                const fp8_raw *src = &X[(size_t)g_row_a * K + next_k + col_a_g];\n"
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"
"            unsigned int dB0 = __cvta_generic_to_shared(&sB_next[swz_b0]);\n"
"            if (g_row_b0 < N) {\n"
"                const fp8_raw *src = &W[(size_t)g_row_b0 * K + next_k + col_b0_g];\n"
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"
"            unsigned int dB1 = __cvta_generic_to_shared(&sB_next[swz_b1]);\n"
"            if (g_row_b1 < N) {\n"
"                const fp8_raw *src = &W[(size_t)g_row_b1 * K + next_k + col_b1_g];\n"
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"
"        }\n"
"        asm volatile(\"cp.async.commit_group;\\n\");\n"
"        unsigned short *sAp = (stage == 0) ? sA0 : sA1;\n"
"        unsigned short *sBp = (stage == 0) ? sB0 : sB1;\n"
"        #pragma unroll\n"
"        for (int kg = 0; kg < 2; kg++) {\n"
"            int k_off = kg * 16;\n"
"            int a_off = la_row * 32 + ((k_off + la_col_off) ^ la_sw);\n"
"            unsigned int a0,a1,a2,a3;\n"
"            unsigned int p_a = __cvta_generic_to_shared(&sAp[a_off]);\n"
"            asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"
"                : \"=r\"(a0), \"=r\"(a1), \"=r\"(a2), \"=r\"(a3) : \"r\"(p_a));\n"
"            #pragma unroll\n"
"            for (int s = 0; s < 4; s++) {\n"
"                int sn_base = wn_col + s * 16;\n"
"                int b_brow  = sn_base + lb_row;\n"
"                int b_sw    = (b_brow & 3) * 8;\n"
"                int b_off   = b_brow * 32 + ((k_off + lb_col_off) ^ b_sw);\n"
"                unsigned int b0,b1,b2,b3;\n"
"                unsigned int p_b = __cvta_generic_to_shared(&sBp[b_off]);\n"
"                asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"
"                    : \"=r\"(b0), \"=r\"(b1), \"=r\"(b2), \"=r\"(b3) : \"r\"(p_b));\n"
"                int nt0 = s * 2 + 0;\n"
"                int nt1 = s * 2 + 1;\n"
"                asm volatile(\n"
"                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    : \"=f\"(d0[nt0]), \"=f\"(d1[nt0]), \"=f\"(d2[nt0]), \"=f\"(d3[nt0])\n"
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                      \"r\"(b0), \"r\"(b2),\n"
"                      \"f\"(d0[nt0]), \"f\"(d1[nt0]), \"f\"(d2[nt0]), \"f\"(d3[nt0])\n"
"                );\n"
"                asm volatile(\n"
"                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    : \"=f\"(d0[nt1]), \"=f\"(d1[nt1]), \"=f\"(d2[nt1]), \"=f\"(d3[nt1])\n"
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                      \"r\"(b1), \"r\"(b3),\n"
"                      \"f\"(d0[nt1]), \"f\"(d1[nt1]), \"f\"(d2[nt1]), \"f\"(d3[nt1])\n"
"                );\n"
"            }\n"
"        }\n"
"    }\n"
"    asm volatile(\"cp.async.wait_group 0;\\n\");\n"
"    int yr0 = cta_m + wm_row + gid;\n"
"    int yr1 = cta_m + wm_row + gid + 8;\n"
"    #pragma unroll\n"
"    for (int nt = 0; nt < 8; nt++) {\n"
"        int yc0 = cta_n + wn_col + nt*8 + tid4*2;\n"
"        int yc1 = yc0 + 1;\n"
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"
"    }\n"
"}\n"

/* Post-pass: Y[i,j] = Y[i,j] * (w_scale * x_scale[0]) + (bias ? bias[j] : 0).
 * x_scale_ptr points to the FP8 X tensor's per-tensor dequant scale (d_x_scale).
 * Pass bias=NULL to skip the add. */
"extern \"C\" __global__ void descale_add_bias_f32(\n"
"    float *Y, const float *bias, const float *x_scale_ptr,\n"
"    float w_scale, int rows, int cols, int has_bias) {\n"
"    int j = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int i = blockIdx.y;\n"
"    if (j >= cols || i >= rows) return;\n"
"    float xs = x_scale_ptr[0];\n"
"    float v = Y[(size_t)i * cols + j] * (w_scale * xs);\n"
"    if (has_bias) v += bias[j];\n"
"    Y[(size_t)i * cols + j] = v;\n"
"}\n"

"\n} /* extern \"C\" */\n"
;

/* ---- GPU weight structures ---- */

typedef struct {
    CUdeviceptr qkv_w;     /* [3H, H] — FP8 uint8 or F16/F32 depending on mode */
    CUdeviceptr proj_w;    /* [H, H] */
    CUdeviceptr mlp_up_w;  /* [2*n_ff, H] */
    CUdeviceptr mlp_dn_w;  /* [H, n_ff] */
    CUdeviceptr q_norm;    /* [head_dim] F32 */
    CUdeviceptr k_norm;    /* [head_dim] F32 */
    float qkv_scale, proj_scale, mlp_up_scale, mlp_dn_scale;  /* FP8 per-tensor scales */
} flux2_gpu_stream_t;

typedef struct {
    flux2_gpu_stream_t img, txt;
} flux2_gpu_dblk_t;

typedef struct {
    CUdeviceptr linear1_w;   /* [3H+2*n_ff, H] */
    CUdeviceptr l2_attn_w;   /* [H, H] — first H cols of linear2 */
    CUdeviceptr l2_mlp_w;    /* [H, n_ff] — last n_ff cols of linear2 */
    CUdeviceptr q_norm;      /* [head_dim] F32 */
    CUdeviceptr k_norm;      /* [head_dim] F32 */
    float linear1_scale, l2_attn_scale, l2_mlp_scale;  /* FP8 per-tensor scales */
} flux2_gpu_sblk_t;

/* ---- Forward decls ---- */
static int flux2_env_enabled(const char *name);

/* ---- Runner struct ---- */

struct cuda_flux2_runner {
    CUdevice   device;
    CUcontext  ctx;
    CUstream   stream;
    int        verbose;

    CUmodule   mod;
    CUfunction fn_gemm, fn_gemm_f16, fn_gemm_fp8, fn_gemm_fp8_bf16, fn_layernorm, fn_silu, fn_adaln, fn_gated_add;
    /* MMA m16n8k32 FP8 GEMM with per-tensor scale (sm_89+) */
    CUfunction fn_gemm_fp8_mma, fn_gemm_fp8_mma_bf16;
    int use_fp8_mma;     /* 1 = use MMA path; 0 = use tiled path */
    CUfunction fn_rmsnorm_ph, fn_swiglu, fn_flash_attn, fn_add;
    CUfunction fn_flash_attn_fp8, fn_flash_attn_fp8_ref, fn_quant_fp8, fn_reduce_max_abs, fn_zero_f32;
    CUfunction fn_vae_groupnorm, fn_vae_conv2d, fn_vae_upsample2x, fn_vae_latent_bn;
    CUfunction fn_vae_conv2d_bf16, fn_vae_conv2d_bf16_v2;
    CUfunction fn_gemm_bf16;
    CUfunction fn_f32_to_bf16, fn_transpose_f32_to_bf16, fn_softmax_row;
    int use_bf16_vae;
    int use_vae_v2;
    CUfunction fn_vae_attn, fn_transpose, fn_transpose_add;
    CUfunction fn_rope_img, fn_rope_txt, fn_bf16_trunc;

    int use_f16_gemm;   /* 1 = use gemm_tiled_f16_f32 with F16 weights */
    int use_fp8_gemm;   /* 1 = use gemm_tiled_fp8_f32 with native FP8 weights */
    int fp8_bf16_act;   /* 1 = BF16 activation truncation (matches ComfyUI/PyTorch) */

    /* CPU model (kept for arch params + VAE fallback) */
    flux2_dit_model *dit;
    flux2_vae_model *vae;

    int H, nH, hd, n_ff, pin, txt_dim, n_dbl, n_sgl;

    /* Global GPU weights */
    CUdeviceptr d_img_in_w, d_img_in_b, d_txt_in_w, d_txt_in_b;
    CUdeviceptr d_t_fc1_w, d_t_fc1_b, d_t_fc2_w, d_t_fc2_b;
    CUdeviceptr d_mod_img_w, d_mod_txt_w, d_mod_sgl_w;
    CUdeviceptr d_out_mod_w, d_out_proj_w;

    /* FP8 per-tensor scales for global weights (1.0 when not FP8) */
    float s_img_in, s_txt_in, s_t_fc1, s_t_fc2;
    float s_mod_img, s_mod_txt, s_mod_sgl;
    float s_out_mod, s_out_proj;

    flux2_gpu_dblk_t *gpu_dblk;
    flux2_gpu_sblk_t *gpu_sblk;

    /* Activation buffers */
    CUdeviceptr d_img, d_txt, d_joint;
    CUdeviceptr d_temb, d_temb_silu;
    CUdeviceptr d_mod_img_v, d_mod_txt_v, d_mod_sgl_v;
    CUdeviceptr d_q, d_k, d_v, d_attn_out;
    CUdeviceptr d_scratch1, d_scratch2, d_scratch3;
    CUdeviceptr d_img_in_buf, d_txt_in_buf;  /* input upload buffers */
    /* FP8 attention workspace */
    CUdeviceptr d_q_fp8, d_k_fp8, d_v_fp8;       /* uint8 e4m3 buffers */
    CUdeviceptr d_qkv_max;                       /* [3] f32 max-abs scratch */
    CUdeviceptr d_qkv_scales;                    /* [3] f32 sQ/sK/sV */
    size_t fp8_attn_buf_n;                       /* allocated count (n_tok*dim) */
    int use_fp8_attn;
    /* BF16 attention workspace (mma.sync, ported from cuda/fa v4) */
    CUfunction fn_flash_attn_bf16, fn_cast_f32_to_bf16;
    CUdeviceptr d_q_bf16, d_k_bf16, d_v_bf16;    /* uint16_t bf16 buffers */
    size_t bf16_attn_buf_n;                      /* allocated count (n_tok*dim halves) */
    int use_bf16_attn;
    /* FP8 v7 GEMM (ported from cuda/gemm) — quantize X to FP8 and use the
     * cp.async + ldmatrix + 4×4 panel-swizzle kernel. Workspace sized to the
     * largest activation across all GEMMs in the step. */
    CUfunction fn_gemm_fp8_v7, fn_descale_add_bias;
    CUdeviceptr d_x_fp8;          /* FP8 quantized activation buffer */
    CUdeviceptr d_x_max;           /* [1] f32 abs-max scratch */
    CUdeviceptr d_x_scale;         /* [1] f32 per-tensor scale */
    size_t x_fp8_buf_n;            /* allocated bytes */
    int use_fp8_v7;
    int max_tok;
    int gpu_loaded;
    int use_gpu_dbl_attn;
    int debug_dbl_attn;
    int profile_step;
};

/* ---- F32 to F16 conversion ---- */

static uint16_t f2h(float f) {
    unsigned int b;
    memcpy(&b, &f, 4);
    int s = (b >> 31), e = ((b >> 23) & 0xFF) - 127;
    unsigned m = b & 0x7FFFFF;
    if (e < -24) return (uint16_t)(s << 15);
    if (e < -14) { m |= 0x800000; int sh = -14 - e; return (uint16_t)((s << 15) | ((m + (1u << (sh + 12))) >> (sh + 13))); }
    if (e > 15) return (uint16_t)((s << 15) | 0x7C00);
    uint16_t hm = (uint16_t)((m + 0x1000) >> 13);
    if (hm >= 0x400) { hm = 0; e++; }
    if (e > 15) return (uint16_t)((s << 15) | 0x7C00);
    return (uint16_t)((s << 15) | ((e + 15) << 10) | hm);
}

/* Bulk F32→F16 conversion: AVX2 F16C when available, scalar fallback */
static void f32_to_f16_bulk(uint16_t *out, const float *in, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    /* AVX2 + F16C: convert 8 floats at a time using vcvtps2ph */
    int i = 0;
    int n8 = n & ~7;
    #pragma omp parallel for schedule(static) if(n > 100000)
    for (i = 0; i < n8; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(out + i), h);
    }
    /* Scalar tail */
    for (i = n8; i < n; i++) out[i] = f2h(in[i]);
#else
    #pragma omp parallel for schedule(static) if(n > 100000)
    for (int i = 0; i < n; i++) out[i] = f2h(in[i]);
#endif
}

/* Upload F32 array as F16 to GPU, returns device pointer */
static CUdeviceptr gpu_upload_f16(const float *data, int n) {
    uint16_t *tmp = (uint16_t *)malloc((size_t)n * 2);
    f32_to_f16_bulk(tmp, data, n);
    CUdeviceptr d = 0;
    CUresult err = cuMemAlloc(&d, (size_t)n * 2);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_flux2: gpu_upload_f16 alloc failed (%zu bytes, err=%d)\n",
                (size_t)n * 2, (int)err);
        free(tmp);
        return 0;
    }
    err = cuMemcpyHtoD(d, tmp, (size_t)n * 2);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_flux2: gpu_upload_f16 copy failed (%d)\n", (int)err);
        cuMemFree(d);
        free(tmp);
        return 0;
    }
    free(tmp);
    return d;
}

/* Upload F32 array as F32 to GPU */
static CUdeviceptr gpu_upload_f32(const float *data, int n) {
    CUdeviceptr d = 0;
    CUresult err = cuMemAlloc(&d, (size_t)n * sizeof(float));
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_flux2: gpu_upload_f32 alloc failed (%zu bytes, err=%d)\n",
                (size_t)n * sizeof(float), (int)err);
        return 0;
    }
    err = cuMemcpyHtoD(d, data, (size_t)n * sizeof(float));
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_flux2: gpu_upload_f32 copy failed (%d)\n", (int)err);
        cuMemFree(d);
        return 0;
    }
    return d;
}

static CUdeviceptr gpu_upload_f32_or0(const float *data, int n) {
    return data ? gpu_upload_f32(data, n) : 0;
}

/* Upload raw bytes to GPU */
static CUdeviceptr gpu_upload_bytes(const void *data, size_t n_bytes) {
    CUdeviceptr d = 0;
    CUresult err = cuMemAlloc(&d, n_bytes);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_flux2: gpu_upload_bytes alloc failed (%zu bytes, err=%d)\n",
                n_bytes, (int)err);
        return 0;
    }
    err = cuMemcpyHtoD(d, data, n_bytes);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_flux2: gpu_upload_bytes copy failed (%d)\n", (int)err);
        cuMemFree(d);
        return 0;
    }
    return d;
}

/* Upload flux2_mat (weight matrix) as F32 */
static CUdeviceptr gpu_upload_mat(const flux2_mat *m) {
    return gpu_upload_f32(m->w, m->rows * m->cols);
}

/* Upload flux2_mat (weight matrix) as F16 using round-to-nearest */
static CUdeviceptr gpu_upload_mat_f16(const flux2_mat *m) {
    return gpu_upload_f16(m->w, m->rows * m->cols);
}

/* Upload flux2_mat as F16 or F32 depending on flag */
static CUdeviceptr gpu_upload_mat_auto(const flux2_mat *m, int use_f16) {
    return use_f16 ? gpu_upload_mat_f16(m) : gpu_upload_mat(m);
}

/* Upload raw F32 array as F16 or F32 depending on flag */
static CUdeviceptr gpu_upload_f32_auto(const float *data, int n, int use_f16) {
    return use_f16 ? gpu_upload_f16(data, n) : gpu_upload_f32(data, n);
}

/* ---- Op functions ---- */

static void op_gemm(cuda_flux2_runner *r, CUdeviceptr Y, CUdeviceptr W,
                    CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    CUfunction fn = r->use_f16_gemm ? r->fn_gemm_f16 : r->fn_gemm;
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, r->stream, args, NULL);
}

/* Dispatch GEMM: picks F32/F16/FP8 based on runner mode + per-weight scale */
static int ensure_x_fp8_buf(cuda_flux2_runner *r, size_t bytes);

/* op_gemm_scaled dispatch: picks FP8 MMA / tiled / F32 path.
 *
 * Scale sentinel convention: w_scale > 0 means the weight buffer holds raw FP8
 * bytes and the FP8 path should be used with that per-tensor scale. w_scale < 0
 * (typically -1.0f) means the weight was not FP8 (F32/F16 fallback upload) and
 * the plain F32/F16 GEMM path must be used. This avoids the ambiguity of using
 * 1.0f as a sentinel when a genuine FP8 tensor happens to have scale=1.0. */
static void op_gemm_scaled(cuda_flux2_runner *r, CUdeviceptr Y, CUdeviceptr W,
                           CUdeviceptr X, CUdeviceptr bias,
                           int n_out, int n_in, int n_tok, float w_scale) {
    if (r->use_fp8_gemm && w_scale > 0.0f) {
        /* FP8 v7 path: quantize X to FP8, run gemm_fp8_v7, then descale+bias.
         * Constraints:
         *   K (= n_in) divisible by 64 (FP8 64-byte K-tile);
         *   M (= n_tok) >= 256 — v7's 4x4 CTA panel swizzle needs npy>=4 so
         *   that all 16 panel slots correspond to valid tiles (otherwise
         *   tile_m_>=npy_ early-return drops most of the N range). */
        if (r->use_fp8_v7 && r->fn_gemm_fp8_v7 && r->fn_descale_add_bias &&
            (n_in % 64) == 0 && n_tok >= 256) {
            size_t n_elems = (size_t)n_tok * (size_t)n_in;
            if (ensure_x_fp8_buf(r, n_elems) == 0) {
                /* 1) zero max scratch */
                int one = 1;
                cuMemsetD32Async(r->d_x_max, 0, 1, r->stream);
                /* 2) reduce max-abs of X */
                {
                    int n = (int)n_elems;
                    void *args[] = {&r->d_x_max, &X, &n};
                    cuLaunchKernel(r->fn_reduce_max_abs,
                                   (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1,
                                   0, r->stream, args, NULL);
                }
                /* 3) quantize X -> FP8 with per-tensor scale */
                {
                    int n = (int)n_elems;
                    void *args[] = {&r->d_x_fp8, &r->d_x_scale, &X, &r->d_x_max, &n};
                    cuLaunchKernel(r->fn_quant_fp8,
                                   (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1,
                                   0, r->stream, args, NULL);
                }
                /* 4) gemm_fp8_v7: Y[M=n_tok, N=n_out] = X_fp8[M,K] @ W_fp8[N,K]^T.
                 * Pad gx/gy to multiples of 4 so every CTA in every 4x4 panel is
                 * launched — the kernel's cta_m>=M / yr<M write guards safely
                 * handle the padded out-of-bounds tiles. Without padding, tiles
                 * with panel_id >= ceil(npy/4)*ceil(npx/4) are silently dropped. */
                {
                    int M = n_tok, N = n_out, K = n_in;
                    unsigned npx = (unsigned)((N + 127) / 128);
                    unsigned npy = (unsigned)((M + 63) / 64);
                    unsigned gx = (npx + 3u) & ~3u;
                    unsigned gy = (npy + 3u) & ~3u;
                    unsigned smem_bytes = 2u * (64u * 32u + 128u * 32u) * 2u;  /* 24 KiB */
                    void *args[] = {&Y, &r->d_x_fp8, &W, &M, &N, &K};
                    cuLaunchKernel(r->fn_gemm_fp8_v7, gx, gy, 1, 256, 1, 1,
                                   smem_bytes, r->stream, args, NULL);
                }
                /* 5) descale + bias */
                {
                    int has_bias = (bias != 0) ? 1 : 0;
                    int rows = n_tok, cols = n_out;
                    void *args[] = {&Y, &bias, &r->d_x_scale, &w_scale,
                                    &rows, &cols, &has_bias};
                    unsigned bx = 256;
                    unsigned gx = (unsigned)((cols + bx - 1) / bx);
                    unsigned gy = (unsigned)rows;
                    cuLaunchKernel(r->fn_descale_add_bias, gx, gy, 1, bx, 1, 1,
                                   0, r->stream, args, NULL);
                }
                (void)one;
                return;
            }
            /* OOM or unsupported — fall through to existing FP8 paths */
        }
        CUfunction mma_fn = r->fp8_bf16_act ? r->fn_gemm_fp8_mma_bf16 : r->fn_gemm_fp8_mma;
        if (r->use_fp8_mma && mma_fn) {
            /* MMA path: 4 warps × 64 N = 256 cols/block.
             * f32 variant: MTILE=2 → 32 rows/block.
             * bf16 variant: MTILE=1 → 16 rows/block. */
            int m_rows = r->fp8_bf16_act ? 16 : 32;
            unsigned gx = (unsigned)((n_out + 255) / 256);
            unsigned gy = (unsigned)((n_tok + m_rows - 1) / m_rows);
            unsigned smem_bytes = (unsigned)(m_rows * 32 * sizeof(float));
            void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &w_scale};
            cuLaunchKernel(mma_fn, gx, gy, 1, 128, 1, 1,
                           smem_bytes, r->stream, args, NULL);
            return;
        }
        /* Tiled fallback (16x16 block, 64 cols/block) */
        unsigned gx = (unsigned)((n_out + 63) / 64);
        unsigned gy = (unsigned)((n_tok + 15) / 16);
        CUfunction fn = r->fp8_bf16_act ? r->fn_gemm_fp8_bf16 : r->fn_gemm_fp8;
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &w_scale};
        cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, r->stream, args, NULL);
    } else {
        /* F32 or F16 (scale=1.0 means BF16→F32 fallback) */
        unsigned gx = (unsigned)((n_out + 63) / 64);
        unsigned gy = (unsigned)((n_tok + 15) / 16);
        CUfunction fn = r->use_f16_gemm ? r->fn_gemm_f16 : r->fn_gemm;
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, r->stream, args, NULL);
    }
}

/* FP8 GEMM: same grid but with per-tensor weight scale */
static void op_gemm_fp8(cuda_flux2_runner *r, CUdeviceptr Y, CUdeviceptr W,
                        CUdeviceptr X, CUdeviceptr bias,
                        int n_out, int n_in, int n_tok, float w_scale) {
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &w_scale};
    cuLaunchKernel(r->fn_gemm_fp8, gx, gy, 1, 16, 16, 1, 0, r->stream, args, NULL);
}

static void op_adaln(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr x,
                     CUdeviceptr shift, CUdeviceptr scale, int N, int dim) {
    void *args[] = {&out, &x, &shift, &scale, &N, &dim};
    cuLaunchKernel(r->fn_adaln, (unsigned)N, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static void op_silu(cuda_flux2_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->fn_silu, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void op_bf16_trunc(cuda_flux2_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->fn_bf16_trunc, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void op_rmsnorm_ph(cuda_flux2_runner *r, CUdeviceptr x, CUdeviceptr w,
                          int N, int n_heads, int head_dim) {
    void *args[] = {&x, &w, &N, &n_heads, &head_dim};
    cuLaunchKernel(r->fn_rmsnorm_ph, (unsigned)N, (unsigned)n_heads, 1,
                   32, 1, 1, 0, r->stream, args, NULL);
}

static void op_gated_add(cuda_flux2_runner *r, CUdeviceptr x, CUdeviceptr proj,
                         CUdeviceptr gate, int N, int dim) {
    int total = N * dim;
    void *args[] = {&x, &proj, &gate, &N, &dim};
    cuLaunchKernel(r->fn_gated_add, (unsigned)((total+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

/* Lazily allocate FP8 attention workspace for n bytes (FP8 buffers) + scales.
 * On OOM, sets the offending pointer(s) to 0 and returns -1. Callers must check
 * r->fp8_attn_buf_n after calling. */
static int ensure_fp8_attn_buf(cuda_flux2_runner *r, size_t n) {
    if (n <= r->fp8_attn_buf_n) return 0;
    if (r->d_q_fp8)      { cuMemFree(r->d_q_fp8);      r->d_q_fp8 = 0; }
    if (r->d_k_fp8)      { cuMemFree(r->d_k_fp8);      r->d_k_fp8 = 0; }
    if (r->d_v_fp8)      { cuMemFree(r->d_v_fp8);      r->d_v_fp8 = 0; }
    if (r->d_qkv_max)    { cuMemFree(r->d_qkv_max);    r->d_qkv_max = 0; }
    if (r->d_qkv_scales) { cuMemFree(r->d_qkv_scales); r->d_qkv_scales = 0; }
    r->fp8_attn_buf_n = 0;
    if (cuMemAlloc(&r->d_q_fp8, n) != CUDA_SUCCESS)                    { r->d_q_fp8 = 0; return -1; }
    if (cuMemAlloc(&r->d_k_fp8, n) != CUDA_SUCCESS)                    { r->d_k_fp8 = 0; return -1; }
    if (cuMemAlloc(&r->d_v_fp8, n) != CUDA_SUCCESS)                    { r->d_v_fp8 = 0; return -1; }
    if (cuMemAlloc(&r->d_qkv_max, 3 * sizeof(float)) != CUDA_SUCCESS)  { r->d_qkv_max = 0; return -1; }
    if (cuMemAlloc(&r->d_qkv_scales, 3 * sizeof(float)) != CUDA_SUCCESS) { r->d_qkv_scales = 0; return -1; }
    r->fp8_attn_buf_n = n;
    return 0;
}

/* Lazily allocate BF16 attention workspace: 3 buffers each n_elems * uint16_t. */
static int ensure_bf16_attn_buf(cuda_flux2_runner *r, size_t n_elems) {
    size_t need = n_elems * sizeof(unsigned short);
    if (n_elems <= r->bf16_attn_buf_n) return 0;
    if (r->d_q_bf16) { cuMemFree(r->d_q_bf16); r->d_q_bf16 = 0; }
    if (r->d_k_bf16) { cuMemFree(r->d_k_bf16); r->d_k_bf16 = 0; }
    if (r->d_v_bf16) { cuMemFree(r->d_v_bf16); r->d_v_bf16 = 0; }
    r->bf16_attn_buf_n = 0;
    if (cuMemAlloc(&r->d_q_bf16, need) != CUDA_SUCCESS) { r->d_q_bf16 = 0; return -1; }
    if (cuMemAlloc(&r->d_k_bf16, need) != CUDA_SUCCESS) { r->d_k_bf16 = 0; return -1; }
    if (cuMemAlloc(&r->d_v_bf16, need) != CUDA_SUCCESS) { r->d_v_bf16 = 0; return -1; }
    r->bf16_attn_buf_n = n_elems;
    return 0;
}

/* Lazily allocate FP8 v7 GEMM workspace (X quant buffer + max + scale).
 * `bytes` is the maximum activation tile size (n_tok * n_in) in FP8 bytes. */
static int ensure_x_fp8_buf(cuda_flux2_runner *r, size_t bytes) {
    if (bytes <= r->x_fp8_buf_n) return 0;
    if (r->d_x_fp8) { cuMemFree(r->d_x_fp8); r->d_x_fp8 = 0; }
    r->x_fp8_buf_n = 0;
    if (cuMemAlloc(&r->d_x_fp8, bytes) != CUDA_SUCCESS) { r->d_x_fp8 = 0; return -1; }
    if (!r->d_x_max) {
        if (cuMemAlloc(&r->d_x_max, sizeof(float)) != CUDA_SUCCESS) { r->d_x_max = 0; return -1; }
    }
    if (!r->d_x_scale) {
        if (cuMemAlloc(&r->d_x_scale, sizeof(float)) != CUDA_SUCCESS) { r->d_x_scale = 0; return -1; }
    }
    r->x_fp8_buf_n = bytes;
    return 0;
}

/* Cast n F32 values to BF16 in-place on r->stream. */
static void cast_buf_f32_to_bf16(cuda_flux2_runner *r, CUdeviceptr dst_bf16,
                                 CUdeviceptr src_f32, int n) {
    void *args[] = {&src_f32, &dst_bf16, &n};
    cuLaunchKernel(r->fn_cast_f32_to_bf16, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

/* Quantize a single F32 buffer to FP8 e4m3 with per-tensor scale.
 * max_slot/scale_slot point at one f32 each within d_qkv_max / d_qkv_scales. */
static void quantize_buf_fp8(cuda_flux2_runner *r, CUdeviceptr fp8_out,
                              CUdeviceptr scale_out, CUdeviceptr max_slot,
                              CUdeviceptr in, size_t n) {
    /* Clear max slot via async memset (cheaper than a kernel launch) */
    cuMemsetD32Async(max_slot, 0, 1, r->stream);
    /* reduce max abs */
    int n_int = (int)n;  /* kernels take int; caller ensures n <= INT_MAX */
    void *rargs[] = {&max_slot, &in, &n_int};
    cuLaunchKernel(r->fn_reduce_max_abs, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, rargs, NULL);
    /* quantize */
    void *qargs[] = {&fp8_out, &scale_out, &in, &max_slot, &n_int};
    cuLaunchKernel(r->fn_quant_fp8, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, qargs, NULL);
}

static void op_attn(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr q,
                    CUdeviceptr k, CUdeviceptr v, int n_tok, int n_heads, int head_dim) {
    /* BF16 MMA flash attention path (mma.sync.m16n8k16.bf16, cp.async double-
     * buffered, padded SMEM, ldmatrix.x4.trans P@V). Higher precision than FP8
     * (no per-tensor scale outlier crush) and faster than scalar F32 fallback. */
    if (r->use_bf16_attn && r->fn_flash_attn_bf16 && r->fn_cast_f32_to_bf16 &&
        head_dim == 128) {
        size_t n_elems = (size_t)n_tok * n_heads * head_dim;
        if (ensure_bf16_attn_buf(r, n_elems) == 0) {
            int n = (int)n_elems;
            cast_buf_f32_to_bf16(r, r->d_q_bf16, q, n);
            cast_buf_f32_to_bf16(r, r->d_k_bf16, k, n);
            cast_buf_f32_to_bf16(r, r->d_v_bf16, v, n);
            unsigned gy = (unsigned)((n_tok + 63) / 64);
            /* smem: 2x double-buffered (sK0+sK1+sV0+sV1) at HDP=136 ~34 KB. */
            size_t smem = (size_t)(4 * 32 * 136 * 2);
            void *args[] = {&out, &r->d_q_bf16, &r->d_k_bf16, &r->d_v_bf16,
                            &n_tok, &n_heads, &head_dim};
            cuLaunchKernel(r->fn_flash_attn_bf16,
                           (unsigned)n_heads, gy, 1,
                           128, 1, 1, smem, r->stream, args, NULL);
            return;
        }
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_flux2: WARN ensure_bf16_attn_buf(%zu) failed, falling back\n", n_elems);
    }
    if (r->use_fp8_attn && r->fn_flash_attn_fp8 && r->fn_quant_fp8 &&
        r->fn_reduce_max_abs && head_dim == 128) {
        size_t n_elems = (size_t)n_tok * n_heads * head_dim;
        if (ensure_fp8_attn_buf(r, n_elems) == 0) {
            CUdeviceptr max_q = r->d_qkv_max;
            CUdeviceptr max_k = r->d_qkv_max + sizeof(float);
            CUdeviceptr max_v = r->d_qkv_max + 2 * sizeof(float);
            CUdeviceptr s_q = r->d_qkv_scales;
            CUdeviceptr s_k = r->d_qkv_scales + sizeof(float);
            CUdeviceptr s_v = r->d_qkv_scales + 2 * sizeof(float);
            quantize_buf_fp8(r, r->d_q_fp8, s_q, max_q, q, n_elems);
            quantize_buf_fp8(r, r->d_k_fp8, s_k, max_k, k, n_elems);
            quantize_buf_fp8(r, r->d_v_fp8, s_v, max_v, v, n_elems);
            /* Read scales back to host. Synchronous DtoH implicitly syncs with r->stream
             * (blocks host until all prior stream work completes), so no explicit sync needed. */
            float scales[3];
            cuMemcpyDtoH(scales, r->d_qkv_scales, 3 * sizeof(float));
            if (flux2_env_enabled("FLUX2_FP8_ATTN_DBG")) {
                fprintf(stderr, "fp8_attn: n_tok=%d head_dim=%d sQ=%g sK=%g sV=%g\n",
                        n_tok, head_dim, scales[0], scales[1], scales[2]);
            }
            /* 4-warp CTA: 64 Q rows per block (4 warps × 16 rows each) */
            unsigned gy = (unsigned)((n_tok + 63) / 64);
            /* smK [BKV*HD] + smVT [HD*BKV] + smP [NWARPS * 16 * BKV f32] */
            size_t smem = (size_t)(32 * 128) + (size_t)(128 * 32) +
                          (size_t)(4 * 16 * 32 * sizeof(float));
            void *args[] = {&out, &r->d_q_fp8, &r->d_k_fp8, &r->d_v_fp8,
                            &n_tok, &n_heads, &head_dim,
                            &scales[0], &scales[1], &scales[2]};
            if (flux2_env_enabled("FLUX2_FP8_ATTN_REF") && r->fn_flash_attn_fp8_ref) {
                unsigned gyref = (unsigned)((n_tok + 31) / 32);
                cuLaunchKernel(r->fn_flash_attn_fp8_ref, (unsigned)n_heads, gyref, 1,
                               32, 1, 1, 0, r->stream, args, NULL);
            } else {
                cuLaunchKernel(r->fn_flash_attn_fp8, (unsigned)n_heads, gy, 1,
                               128, 1, 1, smem, r->stream, args, NULL);
            }
            return;
        }
        /* ensure_fp8_attn_buf failed (OOM); warn and fall through to scalar path */
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_flux2: WARN ensure_fp8_attn_buf(%zu) failed, falling back to scalar flash_attn_f32\n", n_elems);
    } else if (r->use_fp8_attn && r->verbose >= 2 && head_dim != 128) {
        /* Model with head_dim != 128 silently falls off the FP8 path (kernel is hardcoded) */
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "cuda_flux2: WARN FLUX2_FP8_ATTN=1 but head_dim=%d (kernel requires 128); using scalar flash_attn_f32\n", head_dim);
            warned = 1;
        }
    }
    int fa2_warps = 4, fa2_bkv = 16;
    unsigned gy = (unsigned)((n_tok + fa2_warps - 1) / fa2_warps);
    unsigned nt = (unsigned)(32 * fa2_warps);
    size_t smem = (size_t)2 * fa2_bkv * 128 * sizeof(float);
    void *args[] = {&out, &q, &k, &v, &n_tok, &n_heads, &head_dim};
    cuLaunchKernel(r->fn_flash_attn, (unsigned)n_heads, gy, 1,
                   nt, 1, 1, smem, r->stream, args, NULL);
}

static void op_swiglu(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in,
                      int n_tok, int mlp_h) {
    unsigned gx = (unsigned)((mlp_h + 255) / 256);
    void *args[] = {&out, &in, &n_tok, &mlp_h};
    cuLaunchKernel(r->fn_swiglu, (unsigned)n_tok, gx, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void op_rope_img(cuda_flux2_runner *r, CUdeviceptr x, int n_tok,
                        int n_heads, int hd, int lat_w, float theta) {
    unsigned gx = (unsigned)((hd/2 + 31) / 32);
    void *args[] = {&x, &n_tok, &n_heads, &hd, &lat_w, &theta};
    cuLaunchKernel(r->fn_rope_img, gx, (unsigned)n_heads, (unsigned)n_tok,
                   32, 1, 1, 0, r->stream, args, NULL);
}

static void op_rope_txt(cuda_flux2_runner *r, CUdeviceptr x, int n_tok,
                        int n_heads, int hd, float theta) {
    unsigned gx = (unsigned)((hd/2 + 31) / 32);
    void *args[] = {&x, &n_tok, &n_heads, &hd, &theta};
    cuLaunchKernel(r->fn_rope_txt, gx, (unsigned)n_heads, (unsigned)n_tok,
                   32, 1, 1, 0, r->stream, args, NULL);
}

static void op_transpose_2d(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in,
                            int rows, int cols) {
    int total = rows * cols;
    void *args[] = {&out, &in, &rows, &cols};
    cuLaunchKernel(r->fn_transpose, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void op_transpose_add(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr x,
                             CUdeviceptr proj, int C, int S) {
    int total = C * S;
    void *args[] = {&out, &x, &proj, &C, &S};
    cuLaunchKernel(r->fn_transpose_add, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

/* F32 GEMM op for VAE (VAE weights are always F32, not F16) */
static void op_gemm_f32(cuda_flux2_runner *r, CUdeviceptr Y, CUdeviceptr W,
                        CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    cuLaunchKernel(r->fn_gemm, gx, gy, 1, 16, 16, 1, 0, r->stream, args, NULL);
}

/* BF16-weight × F32-input GEMM using m16n8k16 mma. W must be BF16 (uint16). */
static void op_gemm_bf16(cuda_flux2_runner *r, CUdeviceptr Y, CUdeviceptr W_bf16,
                         CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    unsigned gx = (unsigned)((n_out + 255) / 256);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    void *args[] = {&Y, &W_bf16, &X, &bias, &n_out, &n_in, &n_tok};
    cuLaunchKernel(r->fn_gemm_bf16, gx, gy, 1, 128, 1, 1, 0, r->stream, args, NULL);
}

static void op_f32_to_bf16(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in, int n) {
    void *args[] = {&out, &in, &n};
    cuLaunchKernel(r->fn_f32_to_bf16, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void op_transpose_f32_to_bf16(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in,
                                      int rows, int cols) {
    int total = rows * cols;
    void *args[] = {&out, &in, &rows, &cols};
    cuLaunchKernel(r->fn_transpose_f32_to_bf16, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void op_softmax_row(cuda_flux2_runner *r, CUdeviceptr mat, int n_rows, int n_cols,
                            float scale) {
    void *args[] = {&mat, &n_rows, &n_cols, &scale};
    cuLaunchKernel(r->fn_softmax_row, (unsigned)n_rows, 1, 1,
                   256, 1, 1, 256 * sizeof(float), r->stream, args, NULL);
}

static void op_vae_attn(cuda_flux2_runner *r, CUdeviceptr out,
                        CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                        int N, int dim) {
    float scale = 1.0f / sqrtf((float)dim);
    int vae_warps = 4, vae_bkv = 8;
    unsigned gx = (unsigned)((N + vae_warps - 1) / vae_warps);
    unsigned nt = (unsigned)(32 * vae_warps);  /* 128 threads */
    size_t smem = (size_t)2 * vae_bkv * dim * sizeof(float);
    void *args[] = {&out, &Q, &K, &V, &N, &dim, &scale};
    cuLaunchKernel(r->fn_vae_attn, gx, 1, 1,
                   nt, 1, 1, smem, r->stream, args, NULL);
}

static void op_layernorm(cuda_flux2_runner *r, CUdeviceptr dst, CUdeviceptr src,
                         int n_tok, int dim) {
    CUdeviceptr null_ptr = 0;
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &null_ptr, &null_ptr, &dim, &eps};
    cuLaunchKernel(r->fn_layernorm, (unsigned)n_tok, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static void op_add(cuda_flux2_runner *r, CUdeviceptr dst, CUdeviceptr src, int n) {
    void *args[] = {&dst, &src, &n};
    cuLaunchKernel(r->fn_add, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void op_vae_groupnorm(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in,
                             CUdeviceptr gamma, CUdeviceptr beta,
                             int C, int spatial, int num_groups) {
    void *args[] = {&out, &in, &gamma, &beta, &C, &spatial, &num_groups};
    cuLaunchKernel(r->fn_vae_groupnorm, (unsigned)num_groups, 1, 1,
                   256, 1, 1, 256 * sizeof(float), r->stream, args, NULL);
}

/* AVX2 bulk F32 -> BF16 (round-to-nearest-even). Falls back to scalar tail. */
static void f32_to_bf16_bulk(uint16_t *out, const float *in, int n) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        __m256i bits = _mm256_castps_si256(v);
        __m256i mb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
        __m256i rnd = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), mb);
        bits = _mm256_add_epi32(bits, rnd);
        bits = _mm256_srli_epi32(bits, 16);
        /* Pack 32-bit lanes [a0..a7] -> 16-bit (each lane fits since values < 0x10000) */
        __m256i packed = _mm256_packus_epi32(bits, bits);
        __m128i lo = _mm256_castsi256_si128(packed);
        __m128i hi = _mm256_extracti128_si256(packed, 1);
        _mm_storel_epi64((__m128i*)(out + i), lo);
        _mm_storel_epi64((__m128i*)(out + i + 4), hi);
    }
#endif
    for (; i < n; i++) {
        unsigned int b;
        memcpy(&b, &in[i], 4);
        unsigned int round = 0x7FFFu + ((b >> 16) & 1u);
        out[i] = (uint16_t)((b + round) >> 16);
    }
}

/* Upload F32 weights as BF16 (half size). Used for VAE 3x3 convs. */
static CUdeviceptr gpu_upload_f32_as_bf16(const float *data, int n) {
    uint16_t *bf = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
    f32_to_bf16_bulk(bf, data, n);
    CUdeviceptr d;
    cuMemAlloc(&d, (size_t)n * sizeof(uint16_t));
    cuMemcpyHtoD(d, bf, (size_t)n * sizeof(uint16_t));
    free(bf);
    return d;
}

static void op_vae_conv2d(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in,
                          CUdeviceptr w, CUdeviceptr b,
                          int ci, int h, int wid, int co, int kh, int kw, int pad) {
    int total = co * h * wid;
    void *args[] = {&out, &in, &w, &b, &ci, &h, &wid, &co, &kh, &kw, &pad};
    cuLaunchKernel(r->fn_vae_conv2d, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

/* BF16 implicit-GEMM conv2d for 3x3 stride=1 pad=1.
 * w must already be BF16 (uploaded via gpu_upload_f32_as_bf16). */
static void op_vae_conv2d_bf16_3x3(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in,
                                    CUdeviceptr w_bf16, CUdeviceptr bias,
                                    int ci, int h, int wid, int co) {
    int M = h * wid;
    /* v2 (halo-cached, MTILE=2) processes 32 contiguous output positions per CTA.
     * Requires w divisible by 32 so the 32-M-row block stays within one h-row. */
    if (r->use_vae_v2 && r->fn_vae_conv2d_bf16_v2 && (wid % 32) == 0 && M >= 32) {
        /* v2 kernel processes M_PER_CTA=32 rows, N=128 cols per CTA (MTILE=2, NWARPS=2, NTILE=8) */
        unsigned gx = (unsigned)((co + 127) / 128);
        unsigned gy = (unsigned)((M + 31) / 32);
        /* sm_in [3*34*16 floats] + sm_kbase [144 shorts] */
        size_t smem = (size_t)(3 * 34 * 16) * sizeof(float) + 144 * sizeof(short);
        void *args[] = {&out, &in, &w_bf16, &bias, &ci, &h, &wid, &co};
        cuLaunchKernel(r->fn_vae_conv2d_bf16_v2, gx, gy, 1, 64, 1, 1,
                       (unsigned)smem, r->stream, args, NULL);
        return;
    }
    /* v1 fallback: 16 M × 128 N per CTA, 2 warps */
    unsigned gx = (unsigned)((co + 127) / 128);
    unsigned gy = (unsigned)((M + 15) / 16);
    void *args[] = {&out, &in, &w_bf16, &bias, &ci, &h, &wid, &co};
    cuLaunchKernel(r->fn_vae_conv2d_bf16, gx, gy, 1, 64, 1, 1, 0, r->stream, args, NULL);
}

/* Wrapper: upload + dispatch for a 3x3 stride=1 pad=1 conv. Uses BF16 path
 * when r->use_bf16_vae is set and the BF16 kernel is loaded. */
static CUdeviceptr vae_upload_w_3x3(cuda_flux2_runner *r, const float *data, int n) {
    if (r->use_bf16_vae && r->fn_vae_conv2d_bf16)
        return gpu_upload_f32_as_bf16(data, n);
    return gpu_upload_f32(data, n);
}

static void vae_conv_3x3(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in,
                         CUdeviceptr w, CUdeviceptr b,
                         int ci, int h, int wid, int co) {
    if (r->use_bf16_vae && r->fn_vae_conv2d_bf16) {
        op_vae_conv2d_bf16_3x3(r, out, in, w, b, ci, h, wid, co);
    } else {
        op_vae_conv2d(r, out, in, w, b, ci, h, wid, co, 3, 3, 1);
    }
}

static CUdeviceptr op_vae_upsample2x(cuda_flux2_runner *r, CUdeviceptr in, int c, int h, int w) {
    int oh = h * 2, ow = w * 2;
    CUdeviceptr out;
    cuMemAlloc(&out, (size_t)c * oh * ow * sizeof(float));
    int total = c * oh * ow;
    void *args[] = {&out, &in, &c, &h, &w};
    cuLaunchKernel(r->fn_vae_upsample2x, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
    return out;
}

static void op_vae_latent_bn(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in,
                             CUdeviceptr mean, CUdeviceptr std, int lc, int h, int w, int ps) {
    int total = lc * h * w;
    void *args[] = {&out, &in, &mean, &std, &lc, &h, &w, &ps};
    cuLaunchKernel(r->fn_vae_latent_bn, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

/* ---- FP8 native upload helpers ---- */

/* Upload a weight tensor from safetensors.
 *
 * Sentinel convention for *out_scale:
 *   > 0  : weight was uploaded as raw FP8 E4M3 bytes; dispatch FP8 GEMM path.
 *   < 0  : weight was uploaded as F32 floats (fallback); dispatch F32 GEMM path.
 *   == 0 : tensor not found.
 *
 * The F32 fallback branch must NOT return scale=1.0 because a genuine FP8 tensor
 * may legitimately have weight_scale==1.0; see op_gemm_scaled for dispatch. */
static CUdeviceptr gpu_upload_st_fp8(st_context *st, const char *wname, const char *sname,
                                      float *out_scale) {
    int widx = safetensors_find(st, wname);
    if (widx < 0) { *out_scale = 0.0f; return 0; }
    const char *dtype = safetensors_dtype(st, widx);
    const uint64_t *sh = safetensors_shape(st, widx);
    int nd = safetensors_ndims(st, widx);
    size_t n = (nd >= 2) ? (size_t)sh[0] * sh[1] : (size_t)sh[0];

    if (strcmp(dtype, "F8_E4M3") == 0) {
        /* Native FP8: upload raw bytes */
        const void *data = safetensors_data(st, widx);
        *out_scale = 1.0f;  /* FP8 dispatch, default scale 1.0 if no weight_scale tensor */
        if (sname) {
            int sidx = safetensors_find(st, sname);
            if (sidx >= 0) *out_scale = *(const float *)safetensors_data(st, sidx);
        }
        return gpu_upload_bytes(data, n);  /* 1 byte per element */
    } else {
        /* BF16/F32: dequant to F32, return sentinel -1.0 (F32 dispatch) */
        *out_scale = -1.0f;
        float *f32 = (float *)malloc(n * sizeof(float));
        if (strcmp(dtype, "BF16") == 0) {
            const uint16_t *src = (const uint16_t *)safetensors_data(st, widx);
            for (size_t i = 0; i < n; i++) {
                uint32_t b = (uint32_t)src[i] << 16;
                memcpy(&f32[i], &b, 4);
            }
        } else {
            memcpy(f32, safetensors_data(st, widx), n * sizeof(float));
        }
        CUdeviceptr d = gpu_upload_f32(f32, (int)n);
        free(f32);
        return d;
    }
}

/* Sinusoidal timestep embedding (CPU, small) */
static void flux2_ts_embed(float *out, float t, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i * logf(10000.0f) / (float)half);
        out[i] = cosf(t * freq);
        out[half + i] = sinf(t * freq);
    }
}

static int flux2_env_enabled(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    return !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0);
}

/* ---- Init ---- */

cuda_flux2_runner *cuda_flux2_init(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuda_flux2: failed to load CUDA driver\n");
        return NULL;
    }

    CUdevice device;
    CUcontext ctx;
    CU_CHECK_NULL(cuInit(0));
    CU_CHECK_NULL(cuDeviceGet(&device, device_id));
    CU_CHECK_NULL(cuDevicePrimaryCtxRetain(&ctx, device));
    CU_CHECK_NULL(cuCtxSetCurrent(ctx));

    char name[256];
    cuDeviceGetName(name, sizeof(name), device);
    if (verbose >= 1) fprintf(stderr, "cuda_flux2: device %d = %s\n", device_id, name);

    /* Compile kernels */
    size_t len1 = strlen(cuda_kernels_common_src), len2 = strlen(flux2_kernel_src);
    char *full = (char *)malloc(len1 + len2 + 16);
    memcpy(full, cuda_kernels_common_src, len1);
    memcpy(full + len1, flux2_kernel_src, len2 + 1);

    CUmodule mod;
    if (cu_compile_kernels(&mod, device, full, "flux2.cu", verbose, "cuda_flux2") < 0) {
        free(full);
        fprintf(stderr, "cuda_flux2: kernel compilation failed\n");
        return NULL;
    }
    free(full);

    cuda_flux2_runner *r = (cuda_flux2_runner *)calloc(1, sizeof(cuda_flux2_runner));
    r->device = device;
    r->ctx = ctx;
    r->verbose = verbose;
    r->mod = mod;
    r->use_gpu_dbl_attn = !flux2_env_enabled("FLUX2_CPU_DBL_ATTN");
    r->debug_dbl_attn = flux2_env_enabled("FLUX2_DEBUG_DBL_ATTN");
    r->use_f16_gemm = flux2_env_enabled("FLUX2_F16_GEMM");
    r->use_fp8_gemm = flux2_env_enabled("FLUX2_FP8_GEMM");
    r->fp8_bf16_act = flux2_env_enabled("FLUX2_FP8_BF16");
    if (r->fp8_bf16_act) r->use_fp8_gemm = 1;  /* BF16-act implies FP8 mode */
    if (r->use_fp8_gemm) r->use_f16_gemm = 0;  /* FP8 takes priority */
    /* Default ON: MMA m16n8k32 path. Set FLUX2_FP8_TILED=1 to fall back to scalar tile. */
    r->use_fp8_mma = r->use_fp8_gemm && !flux2_env_enabled("FLUX2_FP8_TILED");
    r->use_fp8_attn = flux2_env_enabled("FLUX2_FP8_ATTN");
    r->use_bf16_attn = flux2_env_enabled("FLUX2_BF16_ATTN");
    r->use_fp8_v7 = flux2_env_enabled("FLUX2_FP8_V7");
    r->d_x_fp8 = 0; r->d_x_max = 0; r->d_x_scale = 0; r->x_fp8_buf_n = 0;
    r->fn_gemm_fp8_v7 = NULL; r->fn_descale_add_bias = NULL;
    {
        const char *pv = getenv("FLUX2_PROFILE");
        r->profile_step = (pv && pv[0]) ? atoi(pv) : 0;
        if (pv && (!strcmp(pv,"true") || !strcmp(pv,"TRUE"))) r->profile_step = 1;
    }
    r->d_q_bf16 = 0; r->d_k_bf16 = 0; r->d_v_bf16 = 0; r->bf16_attn_buf_n = 0;
    /* VAE BF16 conv: default ON; set FLUX2_VAE_F32=1 to fall back. */
    r->use_bf16_vae = !flux2_env_enabled("FLUX2_VAE_F32");
    /* VAE conv v2 (halo-cached): default ON if available */
    r->use_vae_v2 = !flux2_env_enabled("FLUX2_VAE_V1");
    r->d_q_fp8 = 0; r->d_k_fp8 = 0; r->d_v_fp8 = 0;
    r->d_qkv_max = 0; r->d_qkv_scales = 0; r->fp8_attn_buf_n = 0;

    cuStreamCreate(&r->stream, 0);

    /* Get function handles */
    cuModuleGetFunction(&r->fn_gemm,        mod, "gemm_f32_f32");
    cuModuleGetFunction(&r->fn_gemm_f16,    mod, "gemm_tiled_f16_f32");
    cuModuleGetFunction(&r->fn_gemm_fp8,    mod, "gemm_tiled_fp8_f32");
    cuModuleGetFunction(&r->fn_gemm_fp8_bf16, mod, "gemm_tiled_fp8_bf16");
    /* MMA-based FP8 GEMMs (sm_89+); may fail to load on older arches */
    if (cuModuleGetFunction(&r->fn_gemm_fp8_mma, mod, "gemm_fp8_scaled_f32") != CUDA_SUCCESS)
        r->fn_gemm_fp8_mma = NULL;
    if (cuModuleGetFunction(&r->fn_gemm_fp8_mma_bf16, mod, "gemm_fp8_scaled_bf16") != CUDA_SUCCESS)
        r->fn_gemm_fp8_mma_bf16 = NULL;
    cuModuleGetFunction(&r->fn_layernorm,   mod, "layernorm_f32");
    cuModuleGetFunction(&r->fn_silu,        mod, "silu_f32");
    cuModuleGetFunction(&r->fn_adaln,       mod, "adaln_modulate_f32");
    cuModuleGetFunction(&r->fn_gated_add,   mod, "gated_add_f32");
    cuModuleGetFunction(&r->fn_rmsnorm_ph,  mod, "rmsnorm_per_head_f32");
    cuModuleGetFunction(&r->fn_swiglu,      mod, "flux2_swiglu");
    cuModuleGetFunction(&r->fn_flash_attn,  mod, "flash_attn_f32");
    if (cuModuleGetFunction(&r->fn_flash_attn_fp8, mod, "flash_attn_fp8") != CUDA_SUCCESS)
        r->fn_flash_attn_fp8 = NULL;
    if (cuModuleGetFunction(&r->fn_flash_attn_bf16, mod, "flash_attn_bf16") != CUDA_SUCCESS)
        r->fn_flash_attn_bf16 = NULL;
    if (cuModuleGetFunction(&r->fn_cast_f32_to_bf16, mod, "cast_f32_to_bf16") != CUDA_SUCCESS)
        r->fn_cast_f32_to_bf16 = NULL;
    if (r->use_bf16_attn && (!r->fn_flash_attn_bf16 || !r->fn_cast_f32_to_bf16)) {
        r->use_bf16_attn = 0;
        if (verbose >= 1)
            fprintf(stderr, "cuda_flux2: FLUX2_BF16_ATTN disabled (kernel(s) missing)\n");
    }
    if (cuModuleGetFunction(&r->fn_flash_attn_fp8_ref, mod, "flash_attn_fp8_ref") != CUDA_SUCCESS)
        r->fn_flash_attn_fp8_ref = NULL;
    if (cuModuleGetFunction(&r->fn_quant_fp8, mod, "quantize_to_fp8_e4m3") != CUDA_SUCCESS)
        r->fn_quant_fp8 = NULL;
    if (cuModuleGetFunction(&r->fn_reduce_max_abs, mod, "reduce_max_abs_f32") != CUDA_SUCCESS)
        r->fn_reduce_max_abs = NULL;
    if (cuModuleGetFunction(&r->fn_zero_f32, mod, "zero_f32") != CUDA_SUCCESS)
        r->fn_zero_f32 = NULL;
    if (cuModuleGetFunction(&r->fn_gemm_fp8_v7, mod, "gemm_fp8_v7") != CUDA_SUCCESS)
        r->fn_gemm_fp8_v7 = NULL;
    if (cuModuleGetFunction(&r->fn_descale_add_bias, mod, "descale_add_bias_f32") != CUDA_SUCCESS)
        r->fn_descale_add_bias = NULL;
    if (r->use_fp8_v7 && (!r->fn_gemm_fp8_v7 || !r->fn_descale_add_bias ||
                          !r->fn_quant_fp8 || !r->fn_reduce_max_abs || !r->fn_zero_f32)) {
        r->use_fp8_v7 = 0;
        if (verbose >= 1)
            fprintf(stderr, "cuda_flux2: FLUX2_FP8_V7 disabled (kernel(s) missing)\n");
    }
    cuModuleGetFunction(&r->fn_add,         mod, "add_f32");
    cuModuleGetFunction(&r->fn_vae_groupnorm, mod, "flux2_vae_groupnorm_f32");
    cuModuleGetFunction(&r->fn_vae_conv2d,    mod, "flux2_vae_conv2d_f32");
    if (cuModuleGetFunction(&r->fn_vae_conv2d_bf16, mod, "conv2d_bf16_f32_3x3") != CUDA_SUCCESS)
        r->fn_vae_conv2d_bf16 = NULL;
    if (cuModuleGetFunction(&r->fn_vae_conv2d_bf16_v2, mod, "conv2d_bf16_f32_3x3_v2") != CUDA_SUCCESS)
        r->fn_vae_conv2d_bf16_v2 = NULL;
    if (cuModuleGetFunction(&r->fn_gemm_bf16, mod, "gemm_bf16_f32") != CUDA_SUCCESS)
        r->fn_gemm_bf16 = NULL;
    if (cuModuleGetFunction(&r->fn_f32_to_bf16, mod, "f32_to_bf16_kernel") != CUDA_SUCCESS)
        r->fn_f32_to_bf16 = NULL;
    if (cuModuleGetFunction(&r->fn_transpose_f32_to_bf16, mod, "transpose_f32_to_bf16") != CUDA_SUCCESS)
        r->fn_transpose_f32_to_bf16 = NULL;
    if (cuModuleGetFunction(&r->fn_softmax_row, mod, "softmax_row_f32_inplace") != CUDA_SUCCESS)
        r->fn_softmax_row = NULL;
    cuModuleGetFunction(&r->fn_vae_upsample2x, mod, "flux2_vae_upsample2x_f32");
    cuModuleGetFunction(&r->fn_vae_latent_bn, mod, "flux2_vae_latent_bn_f32");
    cuModuleGetFunction(&r->fn_vae_attn,    mod, "vae_attn_f32");
    cuModuleGetFunction(&r->fn_transpose,   mod, "transpose_2d_f32");
    cuModuleGetFunction(&r->fn_transpose_add, mod, "transpose_add_f32");
    cuModuleGetFunction(&r->fn_rope_img,    mod, "flux2_rope_img_f32");
    cuModuleGetFunction(&r->fn_rope_txt,    mod, "flux2_rope_txt_f32");
    cuModuleGetFunction(&r->fn_bf16_trunc,  mod, "truncate_bf16_f32");

    if (verbose >= 1) {
        fprintf(stderr, "cuda_flux2: init OK");
        if (r->use_gpu_dbl_attn) fprintf(stderr, " [gpu-dbl-attn]");
        if (r->debug_dbl_attn) fprintf(stderr, " [debug-dbl-attn]");
        if (r->use_f16_gemm) fprintf(stderr, " [f16-gemm]");
        if (r->use_fp8_gemm) fprintf(stderr, r->fp8_bf16_act ? " [fp8-bf16act]" : " [fp8-gemm]");
        if (r->use_fp8_mma) fprintf(stderr, " [fp8-mma]");
        if (r->use_fp8_attn) fprintf(stderr, " [fp8-attn]");
        if (r->use_bf16_attn) fprintf(stderr, " [bf16-attn]");
        if (r->use_fp8_v7) fprintf(stderr, " [fp8-v7]");
        if (r->use_bf16_vae) fprintf(stderr, " [vae-bf16]");
        fprintf(stderr, "\n");
    }
    return r;
}

/* Upload stream block weights to GPU (F16 or F32 for GEMM weights) */
static void upload_stream(flux2_gpu_stream_t *gs, const flux2_stream_block *sb, int hd, int use_f16) {
    gs->qkv_w    = gpu_upload_mat_auto(&sb->qkv, use_f16);
    gs->proj_w   = gpu_upload_mat_auto(&sb->proj, use_f16);
    gs->mlp_up_w = gpu_upload_mat_auto(&sb->mlp_up, use_f16);
    gs->mlp_dn_w = gpu_upload_mat_auto(&sb->mlp_down, use_f16);
    gs->q_norm   = gpu_upload_f32(sb->q_norm, hd);
    gs->k_norm   = gpu_upload_f32(sb->k_norm, hd);
    /* -1.0f sentinel: weights are F32/F16, route via F32 GEMM path */
    gs->qkv_scale = gs->proj_scale = gs->mlp_up_scale = gs->mlp_dn_scale = -1.0f;
}

/* Upload stream block weights as native FP8 from safetensors */
static void upload_stream_fp8(flux2_gpu_stream_t *gs, st_context *st,
                               const char *prefix, int hd) {
    char wn[256], sn[256];
    snprintf(wn, sizeof(wn), "%s.qkv.weight", prefix);
    snprintf(sn, sizeof(sn), "%s.qkv.weight_scale", prefix);
    gs->qkv_w = gpu_upload_st_fp8(st, wn, sn, &gs->qkv_scale);

    snprintf(wn, sizeof(wn), "%s.proj.weight", prefix);
    snprintf(sn, sizeof(sn), "%s.proj.weight_scale", prefix);
    gs->proj_w = gpu_upload_st_fp8(st, wn, sn, &gs->proj_scale);

    snprintf(wn, sizeof(wn), "%s.mlp.0.weight", prefix);
    snprintf(sn, sizeof(sn), "%s.mlp.0.weight_scale", prefix);
    gs->mlp_up_w = gpu_upload_st_fp8(st, wn, sn, &gs->mlp_up_scale);

    snprintf(wn, sizeof(wn), "%s.mlp.2.weight", prefix);
    snprintf(sn, sizeof(sn), "%s.mlp.2.weight_scale", prefix);
    gs->mlp_dn_w = gpu_upload_st_fp8(st, wn, sn, &gs->mlp_dn_scale);

    /* Norm scales are always BF16 → F32, loaded from CPU model */
    snprintf(wn, sizeof(wn), "%s.norm.query_norm.scale", prefix);
    int idx = safetensors_find(st, wn);
    if (idx >= 0) {
        int nd = safetensors_ndims(st, idx);
        int n = (int)safetensors_shape(st, idx)[nd > 0 ? 0 : 0];
        float *tmp = (float *)malloc((size_t)n * sizeof(float));
        const uint16_t *src = (const uint16_t *)safetensors_data(st, idx);
        for (int i = 0; i < n; i++) { uint32_t b = (uint32_t)src[i] << 16; memcpy(&tmp[i], &b, 4); }
        gs->q_norm = gpu_upload_f32(tmp, n);
        free(tmp);
    } else {
        gs->q_norm = 0;
    }
    snprintf(wn, sizeof(wn), "%s.norm.key_norm.scale", prefix);
    idx = safetensors_find(st, wn);
    if (idx >= 0) {
        int n = (int)safetensors_shape(st, idx)[0];
        float *tmp = (float *)malloc((size_t)n * sizeof(float));
        const uint16_t *src = (const uint16_t *)safetensors_data(st, idx);
        for (int i = 0; i < n; i++) { uint32_t b = (uint32_t)src[i] << 16; memcpy(&tmp[i], &b, 4); }
        gs->k_norm = gpu_upload_f32(tmp, n);
        free(tmp);
    } else {
        gs->k_norm = 0;
    }
}

static void free_stream(flux2_gpu_stream_t *gs) {
    if (gs->qkv_w)    cuMemFree(gs->qkv_w);
    if (gs->proj_w)   cuMemFree(gs->proj_w);
    if (gs->mlp_up_w) cuMemFree(gs->mlp_up_w);
    if (gs->mlp_dn_w) cuMemFree(gs->mlp_dn_w);
    if (gs->q_norm)   cuMemFree(gs->q_norm);
    if (gs->k_norm)   cuMemFree(gs->k_norm);
}

int cuda_flux2_load_dit(cuda_flux2_runner *r, const char *path) {
    if (r->verbose >= 1) fprintf(stderr, "cuda_flux2: loading DiT %s\n", path);

    r->dit = flux2_dit_load_safetensors(path);
    if (!r->dit) return -1;

    flux2_dit_model *m = r->dit;
    r->H = m->hidden_dim;  r->nH = m->n_heads;  r->hd = m->head_dim;
    r->n_ff = m->n_ff;     r->pin = m->patch_in_channels;
    r->txt_dim = m->txt_dim;
    r->n_dbl = m->n_double_blocks;
    r->n_sgl = m->n_single_blocks;

    int f16 = r->use_f16_gemm;
    int fp8 = r->use_fp8_gemm;
    const char *wfmt = fp8 ? "FP8" : (f16 ? "F16" : "F32");

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_flux2: uploading weights to GPU (H=%d, %d+%d blocks, %s)...\n",
                r->H, r->n_dbl, r->n_sgl, wfmt);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* For FP8 native mode, re-open safetensors to get raw FP8 bytes */
    st_context *st_fp8 = NULL;
    if (fp8) {
        st_fp8 = safetensors_open(path);
        if (!st_fp8) { fprintf(stderr, "cuda_flux2: failed to reopen %s for FP8\n", path); return -1; }
    }

    /* Initialize global weight scales to 1.0 */
    r->s_img_in = r->s_txt_in = r->s_t_fc1 = r->s_t_fc2 = 1.0f;
    r->s_mod_img = r->s_mod_txt = r->s_mod_sgl = 1.0f;
    r->s_out_mod = r->s_out_proj = 1.0f;

    if (fp8) {
        /* Global weights: upload from safetensors as FP8 or BF16 */
        r->d_img_in_w  = gpu_upload_st_fp8(st_fp8, "img_in.weight", NULL, &r->s_img_in);
        r->d_txt_in_w  = gpu_upload_st_fp8(st_fp8, "txt_in.weight", NULL, &r->s_txt_in);
        r->d_t_fc1_w   = gpu_upload_st_fp8(st_fp8, "time_in.in_layer.weight", NULL, &r->s_t_fc1);
        r->d_t_fc2_w   = gpu_upload_st_fp8(st_fp8, "time_in.out_layer.weight", NULL, &r->s_t_fc2);
        r->d_mod_img_w = gpu_upload_st_fp8(st_fp8, "double_stream_modulation_img.lin.weight", NULL, &r->s_mod_img);
        r->d_mod_txt_w = gpu_upload_st_fp8(st_fp8, "double_stream_modulation_txt.lin.weight", NULL, &r->s_mod_txt);
        r->d_mod_sgl_w = gpu_upload_st_fp8(st_fp8, "single_stream_modulation.lin.weight", NULL, &r->s_mod_sgl);
        r->d_out_mod_w = gpu_upload_st_fp8(st_fp8, "final_layer.adaLN_modulation.1.weight", NULL, &r->s_out_mod);
        r->d_out_proj_w= gpu_upload_st_fp8(st_fp8, "final_layer.linear.weight", NULL, &r->s_out_proj);
    } else {
        /* F32 or F16 upload from CPU-dequantized model */
        r->d_img_in_w  = gpu_upload_mat_auto(&m->img_in, f16);
        r->d_txt_in_w  = gpu_upload_mat_auto(&m->txt_in, f16);
        r->d_t_fc1_w   = gpu_upload_mat_auto(&m->time_in_lin1, f16);
        r->d_t_fc2_w   = gpu_upload_mat_auto(&m->time_in_lin2, f16);
        r->d_mod_img_w = gpu_upload_mat_auto(&m->mod_img, f16);
        r->d_mod_txt_w = gpu_upload_mat_auto(&m->mod_txt, f16);
        r->d_mod_sgl_w = gpu_upload_mat_auto(&m->mod_sgl, f16);
        r->d_out_mod_w = gpu_upload_mat_auto(&m->out_mod, f16);
        r->d_out_proj_w= gpu_upload_mat_auto(&m->out_proj, f16);
    }
    /* Bias is always F32 regardless of weight format */
    r->d_img_in_b  = gpu_upload_f32_or0(m->img_in_b, r->H);
    r->d_txt_in_b  = gpu_upload_f32_or0(m->txt_in_b, r->H);
    r->d_t_fc1_b   = gpu_upload_f32_or0(m->time_in_lin1_b, r->H);
    r->d_t_fc2_b   = gpu_upload_f32_or0(m->time_in_lin2_b, r->H);

    /* Double-stream block weights */
    r->gpu_dblk = (flux2_gpu_dblk_t *)calloc((size_t)r->n_dbl, sizeof(flux2_gpu_dblk_t));
    for (int i = 0; i < r->n_dbl; i++) {
        if (fp8) {
            char prefix[128];
            snprintf(prefix, sizeof(prefix), "double_blocks.%d.img_attn", i);
            upload_stream_fp8(&r->gpu_dblk[i].img, st_fp8, prefix, r->hd);
            snprintf(prefix, sizeof(prefix), "double_blocks.%d.txt_attn", i);
            upload_stream_fp8(&r->gpu_dblk[i].txt, st_fp8, prefix, r->hd);
            /* MLP: need separate prefix since it's img_mlp/txt_mlp not img_attn/txt_attn */
            /* Actually the upload_stream_fp8 uses attn prefix for qkv/proj and mlp for mlp.0/mlp.2 */
            /* Fix: the mlp keys are double_blocks.N.img_mlp.0.weight, not double_blocks.N.img_attn.mlp.0 */
            /* Let me read the mlp weights separately */
            char wn[256], sn[256];
            snprintf(wn, sizeof(wn), "double_blocks.%d.img_mlp.0.weight", i);
            snprintf(sn, sizeof(sn), "double_blocks.%d.img_mlp.0.weight_scale", i);
            r->gpu_dblk[i].img.mlp_up_w = gpu_upload_st_fp8(st_fp8, wn, sn, &r->gpu_dblk[i].img.mlp_up_scale);
            snprintf(wn, sizeof(wn), "double_blocks.%d.img_mlp.2.weight", i);
            snprintf(sn, sizeof(sn), "double_blocks.%d.img_mlp.2.weight_scale", i);
            r->gpu_dblk[i].img.mlp_dn_w = gpu_upload_st_fp8(st_fp8, wn, sn, &r->gpu_dblk[i].img.mlp_dn_scale);
            snprintf(wn, sizeof(wn), "double_blocks.%d.txt_mlp.0.weight", i);
            snprintf(sn, sizeof(sn), "double_blocks.%d.txt_mlp.0.weight_scale", i);
            r->gpu_dblk[i].txt.mlp_up_w = gpu_upload_st_fp8(st_fp8, wn, sn, &r->gpu_dblk[i].txt.mlp_up_scale);
            snprintf(wn, sizeof(wn), "double_blocks.%d.txt_mlp.2.weight", i);
            snprintf(sn, sizeof(sn), "double_blocks.%d.txt_mlp.2.weight_scale", i);
            r->gpu_dblk[i].txt.mlp_dn_w = gpu_upload_st_fp8(st_fp8, wn, sn, &r->gpu_dblk[i].txt.mlp_dn_scale);
        } else {
            upload_stream(&r->gpu_dblk[i].img, &m->dblk[i].img, r->hd, f16);
            upload_stream(&r->gpu_dblk[i].txt, &m->dblk[i].txt, r->hd, f16);
        }
        if (r->verbose >= 2)
            fprintf(stderr, "\r  double block %d/%d", i+1, r->n_dbl);
    }

    /* Single-stream block weights */
    r->gpu_sblk = (flux2_gpu_sblk_t *)calloc((size_t)r->n_sgl, sizeof(flux2_gpu_sblk_t));
    for (int i = 0; i < r->n_sgl; i++) {
        if (fp8) {
            char wn[256], sn[256];
            snprintf(wn, sizeof(wn), "single_blocks.%d.linear1.weight", i);
            snprintf(sn, sizeof(sn), "single_blocks.%d.linear1.weight_scale", i);
            r->gpu_sblk[i].linear1_w = gpu_upload_st_fp8(st_fp8, wn, sn, &r->gpu_sblk[i].linear1_scale);

            /* linear2: for FP8, split columns from raw FP8 bytes */
            snprintf(wn, sizeof(wn), "single_blocks.%d.linear2.weight", i);
            snprintf(sn, sizeof(sn), "single_blocks.%d.linear2.weight_scale", i);
            float l2_scale = 1.0f;
            int widx = safetensors_find(st_fp8, wn);
            if (widx >= 0) {
                const char *dtype = safetensors_dtype(st_fp8, widx);
                int sidx = safetensors_find(st_fp8, sn);
                if (sidx >= 0) l2_scale = *(const float *)safetensors_data(st_fp8, sidx);
                int Hd = r->H, nf = r->n_ff;

                if (strcmp(dtype, "F8_E4M3") == 0) {
                    /* FP8: split columns byte-by-byte */
                    const uint8_t *raw = (const uint8_t *)safetensors_data(st_fp8, widx);
                    int l2_cols = Hd + nf;
                    uint8_t *attn = (uint8_t *)malloc((size_t)Hd * Hd);
                    uint8_t *mlp_ = (uint8_t *)malloc((size_t)Hd * nf);
                    for (int r2 = 0; r2 < Hd; r2++) {
                        memcpy(attn + (size_t)r2 * Hd, raw + (size_t)r2 * l2_cols, (size_t)Hd);
                        memcpy(mlp_ + (size_t)r2 * nf, raw + (size_t)r2 * l2_cols + Hd, (size_t)nf);
                    }
                    r->gpu_sblk[i].l2_attn_w = gpu_upload_bytes(attn, (size_t)Hd * Hd);
                    r->gpu_sblk[i].l2_mlp_w  = gpu_upload_bytes(mlp_, (size_t)Hd * nf);
                    free(attn); free(mlp_);
                } else {
                    /* BF16/F32: dequant + split as F32 */
                    const flux2_mat *l2 = &m->sblk[i].linear2;
                    int l2_in = l2->cols;
                    float *a = (float *)malloc((size_t)Hd * Hd * sizeof(float));
                    float *ml = (float *)malloc((size_t)Hd * nf * sizeof(float));
                    for (int r2 = 0; r2 < Hd; r2++) {
                        memcpy(a + (size_t)r2 * Hd, l2->w + (size_t)r2 * l2_in, (size_t)Hd * sizeof(float));
                        memcpy(ml + (size_t)r2 * nf, l2->w + (size_t)r2 * l2_in + Hd, (size_t)nf * sizeof(float));
                    }
                    r->gpu_sblk[i].l2_attn_w = gpu_upload_f32(a, Hd * Hd);
                    r->gpu_sblk[i].l2_mlp_w  = gpu_upload_f32(ml, Hd * nf);
                    free(a); free(ml);
                    l2_scale = -1.0f;  /* F32 path sentinel (dequantized, not FP8 bytes) */
                }
            }
            r->gpu_sblk[i].l2_attn_scale = l2_scale;
            r->gpu_sblk[i].l2_mlp_scale  = l2_scale;

            /* Norms from safetensors */
            snprintf(wn, sizeof(wn), "single_blocks.%d.norm.query_norm.scale", i);
            int nidx = safetensors_find(st_fp8, wn);
            if (nidx >= 0) {
                int n = (int)safetensors_shape(st_fp8, nidx)[0];
                float *tmp = (float *)malloc((size_t)n * sizeof(float));
                const uint16_t *src = (const uint16_t *)safetensors_data(st_fp8, nidx);
                for (int j = 0; j < n; j++) { uint32_t b = (uint32_t)src[j] << 16; memcpy(&tmp[j], &b, 4); }
                r->gpu_sblk[i].q_norm = gpu_upload_f32(tmp, n);
                free(tmp);
            }
            snprintf(wn, sizeof(wn), "single_blocks.%d.norm.key_norm.scale", i);
            nidx = safetensors_find(st_fp8, wn);
            if (nidx >= 0) {
                int n = (int)safetensors_shape(st_fp8, nidx)[0];
                float *tmp = (float *)malloc((size_t)n * sizeof(float));
                const uint16_t *src = (const uint16_t *)safetensors_data(st_fp8, nidx);
                for (int j = 0; j < n; j++) { uint32_t b = (uint32_t)src[j] << 16; memcpy(&tmp[j], &b, 4); }
                r->gpu_sblk[i].k_norm = gpu_upload_f32(tmp, n);
                free(tmp);
            }
        } else {
            r->gpu_sblk[i].linear1_w = gpu_upload_mat_auto(&m->sblk[i].linear1, f16);
            r->gpu_sblk[i].linear1_scale = -1.0f;  /* F32/F16 path sentinel */
            /* linear2 is [H, H+n_ff]. Split columns: first H → attn, last n_ff → mlp */
            {
                const flux2_mat *l2 = &m->sblk[i].linear2;
                int Hd = r->H, nf = r->n_ff;
                int l2_in = l2->cols;
                float *attn = (float *)malloc((size_t)Hd * Hd * sizeof(float));
                float *mlp_ = (float *)malloc((size_t)Hd * nf * sizeof(float));
                for (int r2 = 0; r2 < Hd; r2++) {
                    memcpy(attn + (size_t)r2 * Hd, l2->w + (size_t)r2 * l2_in, (size_t)Hd * sizeof(float));
                    memcpy(mlp_ + (size_t)r2 * nf, l2->w + (size_t)r2 * l2_in + Hd, (size_t)nf * sizeof(float));
                }
                r->gpu_sblk[i].l2_attn_w = gpu_upload_f32_auto(attn, Hd * Hd, f16);
                r->gpu_sblk[i].l2_mlp_w  = gpu_upload_f32_auto(mlp_, Hd * nf, f16);
                free(attn); free(mlp_);
            }
            r->gpu_sblk[i].l2_attn_scale = r->gpu_sblk[i].l2_mlp_scale = -1.0f;  /* F32/F16 path */
            r->gpu_sblk[i].q_norm = gpu_upload_f32(m->sblk[i].q_norm, r->hd);
            r->gpu_sblk[i].k_norm = gpu_upload_f32(m->sblk[i].k_norm, r->hd);
        }
        if (r->verbose >= 2)
            fprintf(stderr, "\r  single block %d/%d", i+1, r->n_sgl);
    }

    if (st_fp8) safetensors_close(st_fp8);

    cuCtxSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "\ncuda_flux2: weights uploaded in %.1f s\n", dt);

    r->gpu_loaded = 1;
    return 0;
}

int cuda_flux2_load_vae(cuda_flux2_runner *r, const char *path) {
    if (r->verbose >= 1) fprintf(stderr, "cuda_flux2: loading VAE %s\n", path);
    r->vae = flux2_vae_load(path);
    return r->vae ? 0 : -1;
}

/* ---- Activation buffer allocation ---- */

static int flux2_alloc_one(CUdeviceptr *dst, size_t bytes, const char *name) {
    *dst = 0;
    CUresult err = cuMemAlloc(dst, bytes);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_flux2: cuMemAlloc failed for %s (%zu bytes, err=%d)\n",
                name, bytes, (int)err);
        return -1;
    }
    return 0;
}

static int flux2_alloc_bufs(cuda_flux2_runner *r, int n_img, int n_txt) {
    int n_tot = n_img + n_txt;
    int H = r->H, n_ff = r->n_ff;

    if (r->max_tok >= n_tot) return 0;  /* already big enough */

    /* Free old buffers */
    if (r->d_img)        { cuMemFree(r->d_img); r->d_img = 0; }
    if (r->d_txt)        { cuMemFree(r->d_txt); r->d_txt = 0; }
    if (r->d_joint)      { cuMemFree(r->d_joint); r->d_joint = 0; }
    if (r->d_temb)       { cuMemFree(r->d_temb); r->d_temb = 0; }
    if (r->d_temb_silu)  { cuMemFree(r->d_temb_silu); r->d_temb_silu = 0; }
    if (r->d_mod_img_v)  { cuMemFree(r->d_mod_img_v); r->d_mod_img_v = 0; }
    if (r->d_mod_txt_v)  { cuMemFree(r->d_mod_txt_v); r->d_mod_txt_v = 0; }
    if (r->d_mod_sgl_v)  { cuMemFree(r->d_mod_sgl_v); r->d_mod_sgl_v = 0; }
    if (r->d_q)          { cuMemFree(r->d_q); r->d_q = 0; }
    if (r->d_k)          { cuMemFree(r->d_k); r->d_k = 0; }
    if (r->d_v)          { cuMemFree(r->d_v); r->d_v = 0; }
    if (r->d_attn_out)   { cuMemFree(r->d_attn_out); r->d_attn_out = 0; }
    if (r->d_scratch1)   { cuMemFree(r->d_scratch1); r->d_scratch1 = 0; }
    if (r->d_scratch2)   { cuMemFree(r->d_scratch2); r->d_scratch2 = 0; }
    if (r->d_scratch3)   { cuMemFree(r->d_scratch3); r->d_scratch3 = 0; }
    if (r->d_img_in_buf) { cuMemFree(r->d_img_in_buf); r->d_img_in_buf = 0; }
    if (r->d_txt_in_buf) { cuMemFree(r->d_txt_in_buf); r->d_txt_in_buf = 0; }

    size_t F = sizeof(float);
    if (flux2_alloc_one(&r->d_img,        (size_t)n_img * H * F, "d_img") != 0) return -1;
    if (flux2_alloc_one(&r->d_txt,        (size_t)n_txt * H * F, "d_txt") != 0) return -1;
    if (flux2_alloc_one(&r->d_joint,      (size_t)n_tot * H * F, "d_joint") != 0) return -1;
    if (flux2_alloc_one(&r->d_temb,       (size_t)H * F, "d_temb") != 0) return -1;
    if (flux2_alloc_one(&r->d_temb_silu,  (size_t)H * F, "d_temb_silu") != 0) return -1;
    if (flux2_alloc_one(&r->d_mod_img_v,  (size_t)6 * H * F, "d_mod_img_v") != 0) return -1;
    if (flux2_alloc_one(&r->d_mod_txt_v,  (size_t)6 * H * F, "d_mod_txt_v") != 0) return -1;
    if (flux2_alloc_one(&r->d_mod_sgl_v,  (size_t)3 * H * F, "d_mod_sgl_v") != 0) return -1;
    if (flux2_alloc_one(&r->d_q,          (size_t)n_tot * H * F, "d_q") != 0) return -1;
    if (flux2_alloc_one(&r->d_k,          (size_t)n_tot * H * F, "d_k") != 0) return -1;
    if (flux2_alloc_one(&r->d_v,          (size_t)n_tot * H * F, "d_v") != 0) return -1;
    if (flux2_alloc_one(&r->d_attn_out,   (size_t)n_tot * H * F, "d_attn_out") != 0) return -1;
    /* scratch1: for QKV/lin1 output [n_tot, max(3H, 3H+2*n_ff)] */
    int lin1_dim = 3 * H + 2 * n_ff;
    if (flux2_alloc_one(&r->d_scratch1,   (size_t)n_tot * lin1_dim * F, "d_scratch1") != 0) return -1;
    /* scratch2: for MLP gate_up [n_tot, 2*n_ff] */
    if (flux2_alloc_one(&r->d_scratch2,   (size_t)n_tot * 2 * n_ff * F, "d_scratch2") != 0) return -1;
    /* scratch3: for MLP out / proj [n_tot, max(n_ff, H)] */
    int s3 = n_ff > H ? n_ff : H;
    if (flux2_alloc_one(&r->d_scratch3,   (size_t)n_tot * s3 * F, "d_scratch3") != 0) return -1;
    if (flux2_alloc_one(&r->d_img_in_buf, (size_t)n_img * r->pin * F, "d_img_in_buf") != 0) return -1;
    if (flux2_alloc_one(&r->d_txt_in_buf, (size_t)n_txt * r->txt_dim * F, "d_txt_in_buf") != 0) return -1;

    r->max_tok = n_tot;
    if (r->verbose >= 1)
        fprintf(stderr, "cuda_flux2: allocated activation buffers for %d tokens\n", n_tot);
    return 0;
}

/* ---- GPU DiT forward pass ---- */

int cuda_flux2_dit_step(cuda_flux2_runner *r,
                        const float *img_tokens, int n_img,
                        const float *txt_tokens, int n_txt,
                        float timestep, float guidance, float *out) {
    (void)guidance;
    if (!r->dit || !r->gpu_loaded) {
        fprintf(stderr, "cuda_flux2: DiT not loaded\n");
        return -1;
    }

    int H = r->H, nH = r->nH, hd = r->hd, n_ff = r->n_ff;
    int n_tot = n_img + n_txt;
    size_t F = sizeof(float);
    size_t WE = r->use_fp8_gemm ? 1 : (r->use_f16_gemm ? 2 : 4);  /* weight element bytes */

    /* Macro: dispatch GEMM with per-weight scale (FP8-aware) */
    #define GEMM(Y, W, X, bias, nout, nin, ntok, scale) \
        op_gemm_scaled(r, Y, W, X, bias, nout, nin, ntok, scale)
    /* Macro: BF16 truncate buffer in-place when running BF16 inference mode */
    #define BF16(buf, n) do { if (r->fp8_bf16_act) op_bf16_trunc(r, buf, n); } while(0)

    if (flux2_alloc_bufs(r, n_img, n_txt) != 0) return -1;

    struct timespec _pt0={0}, _pt1={0}, _pt2={0}, _pt3={0}, _pt4={0};
    if (r->profile_step) { cuCtxSynchronize(); clock_gettime(CLOCK_MONOTONIC, &_pt0); }

    /* Upload inputs */
    cuMemcpyHtoD(r->d_img_in_buf, img_tokens, (size_t)n_img * r->pin * F);
    cuMemcpyHtoD(r->d_txt_in_buf, txt_tokens, (size_t)n_txt * r->txt_dim * F);

    /* 1. Timestep embedding (computed on CPU, uploaded) */
    float t_raw[256];
    flux2_ts_embed(t_raw, timestep * 1000.0f, 256);
    CUdeviceptr d_traw;
    cuMemAlloc(&d_traw, 256 * F);
    cuMemcpyHtoD(d_traw, t_raw, 256 * F);

    /* In BF16 mode, truncate the timestep embedding to BF16 to match ComfyUI */
    BF16(d_traw, 256);

    /* temb = SiLU(time_in_lin1 @ t_raw + b1) */
    GEMM(r->d_temb, r->d_t_fc1_w, d_traw, r->d_t_fc1_b, H, 256, 1, r->s_t_fc1);
    BF16(r->d_temb, H);
    op_silu(r, r->d_temb, H);
    BF16(r->d_temb, H);
    /* temb = time_in_lin2 @ temb + b2 (reuse d_temb_silu as temp) */
    GEMM(r->d_temb_silu, r->d_t_fc2_w, r->d_temb, r->d_t_fc2_b, H, H, 1, r->s_t_fc2);
    BF16(r->d_temb_silu, H);
    /* swap: temb_silu now has the final temb */
    CUdeviceptr d_temb_final = r->d_temb_silu;

    cuMemFree(d_traw);

    /* 2. Project img and txt tokens */
    BF16(r->d_img_in_buf, n_img * r->pin);
    BF16(r->d_txt_in_buf, n_txt * r->txt_dim);
    GEMM(r->d_img, r->d_img_in_w, r->d_img_in_buf, r->d_img_in_b,
         H, r->pin, n_img, r->s_img_in);
    BF16(r->d_img, n_img * H);
    GEMM(r->d_txt, r->d_txt_in_w, r->d_txt_in_buf, r->d_txt_in_b,
         H, r->txt_dim, n_txt, r->s_txt_in);
    BF16(r->d_txt, n_txt * H);

    /* 3. Compute global modulations: SiLU(temb) → mod vectors */
    cuMemcpyDtoD(r->d_temb, d_temb_final, (size_t)H * F);
    op_silu(r, r->d_temb, H);
    BF16(r->d_temb, H);
    /* d_temb now = SiLU(temb_final) */
    CUdeviceptr d0 = 0;
    GEMM(r->d_mod_img_v, r->d_mod_img_w, r->d_temb, d0, 6*H, H, 1, r->s_mod_img);
    GEMM(r->d_mod_txt_v, r->d_mod_txt_w, r->d_temb, d0, 6*H, H, 1, r->s_mod_txt);
    BF16(r->d_mod_img_v, 6*H);
    BF16(r->d_mod_txt_v, 6*H);
    GEMM(r->d_mod_sgl_v, r->d_mod_sgl_w, r->d_temb, d0, 3*H, H, 1, r->s_mod_sgl);
    BF16(r->d_mod_sgl_v, 3*H);

    /* Modulation layout: [shift_attn(H), scale_attn(H), gate_attn(H),
     *                     shift_ffn(H), scale_ffn(H), gate_ffn(H)] */
    CUdeviceptr mi_shift_a = r->d_mod_img_v;
    CUdeviceptr mi_scale_a = r->d_mod_img_v + (size_t)H * F;
    CUdeviceptr mi_gate_a  = r->d_mod_img_v + (size_t)2*H * F;
    CUdeviceptr mi_shift_f = r->d_mod_img_v + (size_t)3*H * F;
    CUdeviceptr mi_scale_f = r->d_mod_img_v + (size_t)4*H * F;
    CUdeviceptr mi_gate_f  = r->d_mod_img_v + (size_t)5*H * F;

    CUdeviceptr mt_shift_a = r->d_mod_txt_v;
    CUdeviceptr mt_scale_a = r->d_mod_txt_v + (size_t)H * F;
    CUdeviceptr mt_gate_a  = r->d_mod_txt_v + (size_t)2*H * F;
    CUdeviceptr mt_shift_f = r->d_mod_txt_v + (size_t)3*H * F;
    CUdeviceptr mt_scale_f = r->d_mod_txt_v + (size_t)4*H * F;
    CUdeviceptr mt_gate_f  = r->d_mod_txt_v + (size_t)5*H * F;

    CUdeviceptr ms_shift = r->d_mod_sgl_v;
    CUdeviceptr ms_scale = r->d_mod_sgl_v + (size_t)H * F;
    CUdeviceptr ms_gate  = r->d_mod_sgl_v + (size_t)2*H * F;

    /* Compute lat_w for RoPE (img tokens arranged as lat_h/ps × lat_w/ps grid) */
    int lat_w_p = 1;
    { /* infer from n_img = lat_h_p * lat_w_p (assume square) */
        lat_w_p = (int)sqrtf((float)n_img);
        if (lat_w_p * lat_w_p != n_img) lat_w_p = n_img; /* fallback to 1D */
    }
    float theta = FLUX2_ROPE_THETA;

    if (r->profile_step) { cuCtxSynchronize(); clock_gettime(CLOCK_MONOTONIC, &_pt1); }

    /* ---- Double-stream blocks ---- */
    for (int bi = 0; bi < r->n_dbl; bi++) {
        flux2_gpu_dblk_t *b = &r->gpu_dblk[bi];

        /* IMG stream: adaLN → QKV */
        op_adaln(r, r->d_scratch1, r->d_img, mi_shift_a, mi_scale_a, n_img, H);
        BF16(r->d_scratch1, n_img * H);

        /* Q,K,V projections from img stream (use weight slices of [3H, H]) */
        CUdeviceptr img_q_w = b->img.qkv_w;
        CUdeviceptr img_k_w = b->img.qkv_w + (size_t)H * H * WE;
        CUdeviceptr img_v_w = b->img.qkv_w + (size_t)2 * H * H * WE;
        GEMM(r->d_q, img_q_w, r->d_scratch1, d0, H, H, n_img, b->img.qkv_scale);
        GEMM(r->d_k, img_k_w, r->d_scratch1, d0, H, H, n_img, b->img.qkv_scale);
        GEMM(r->d_v, img_v_w, r->d_scratch1, d0, H, H, n_img, b->img.qkv_scale);

        /* TXT stream: adaLN → QKV */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_a, mt_scale_a, n_txt, H);
        BF16(r->d_scratch1, n_txt * H);
        CUdeviceptr txt_q_w = b->txt.qkv_w;
        CUdeviceptr txt_k_w = b->txt.qkv_w + (size_t)H * H * WE;
        CUdeviceptr txt_v_w = b->txt.qkv_w + (size_t)2 * H * H * WE;
        /* Write txt Q/K/V after img portion */
        CUdeviceptr tq = r->d_q + (size_t)n_img * H * F;
        CUdeviceptr tk = r->d_k + (size_t)n_img * H * F;
        CUdeviceptr tv = r->d_v + (size_t)n_img * H * F;
        GEMM(tq, txt_q_w, r->d_scratch1, d0, H, H, n_txt, b->txt.qkv_scale);
        GEMM(tk, txt_k_w, r->d_scratch1, d0, H, H, n_txt, b->txt.qkv_scale);
        GEMM(tv, txt_v_w, r->d_scratch1, d0, H, H, n_txt, b->txt.qkv_scale);

        /* Per-head RMSNorm on Q, K (separate norms for img/txt) */
        op_rmsnorm_ph(r, r->d_q, b->img.q_norm, n_img, nH, hd);
        op_rmsnorm_ph(r, r->d_k, b->img.k_norm, n_img, nH, hd);
        op_rmsnorm_ph(r, tq, b->txt.q_norm, n_txt, nH, hd);
        op_rmsnorm_ph(r, tk, b->txt.k_norm, n_txt, nH, hd);
        BF16(r->d_q, n_tot * H);
        BF16(r->d_k, n_tot * H);
        BF16(r->d_v, n_tot * H);

        /* RoPE: img=2D spatial, txt=1D sequence */
        op_rope_img(r, r->d_q, n_img, nH, hd, lat_w_p, theta);
        op_rope_img(r, r->d_k, n_img, nH, hd, lat_w_p, theta);
        op_rope_txt(r, tq, n_txt, nH, hd, theta);
        op_rope_txt(r, tk, n_txt, nH, hd, theta);
        BF16(r->d_q, n_tot * H);
        BF16(r->d_k, n_tot * H);
        /* Joint attention: GPU path is the default. Set FLUX2_CPU_DBL_ATTN=1
         * to force the previous CPU fallback for A/B checks. */
        if (r->use_gpu_dbl_attn) {
            op_attn(r, r->d_attn_out, r->d_q, r->d_k, r->d_v, n_tot, nH, hd);
        } else {
            size_t sz = (size_t)n_tot * H * F;
            float *cq = (float *)malloc(sz);
            float *ck = (float *)malloc(sz);
            float *cv = (float *)malloc(sz);
            float *co = (float *)malloc(sz);
            cuCtxSynchronize();
            cuMemcpyDtoH(cq, r->d_q, sz);
            cuMemcpyDtoH(ck, r->d_k, sz);
            cuMemcpyDtoH(cv, r->d_v, sz);
            flux2_mha(co, cq, ck, cv, n_tot, n_tot, nH, hd);
            cuMemcpyHtoD(r->d_attn_out, co, sz);
            free(cq);
            free(ck);
            free(cv);
            free(co);
        }

        if (r->debug_dbl_attn && bi == 0) {
            size_t sz = (size_t)n_tot * H * F;
            float *q_cpu = (float *)malloc(sz);
            float *k_cpu = (float *)malloc(sz);
            float *v_cpu = (float *)malloc(sz);
            float *attn_gpu = (float *)malloc(sz);
            float *attn_cpu = (float *)malloc(sz);
            cuCtxSynchronize();
            cuMemcpyDtoH(q_cpu, r->d_q, sz);
            cuMemcpyDtoH(k_cpu, r->d_k, sz);
            cuMemcpyDtoH(v_cpu, r->d_v, sz);
            cuMemcpyDtoH(attn_gpu, r->d_attn_out, sz);
            flux2_mha(attn_cpu, q_cpu, k_cpu, v_cpu, n_tot, n_tot, nH, hd);
            float max_diff = 0.0f, mean_diff = 0.0f;
            int max_idx = 0;
            int total = n_tot * H;
            for (int i = 0; i < total; i++) {
                float d = fabsf(attn_gpu[i] - attn_cpu[i]);
                mean_diff += d;
                if (d > max_diff) {
                    max_diff = d;
                    max_idx = i;
                }
            }
            fprintf(stderr,
                    "cuda_flux2: dbl_attn blk0 compare max_diff=%.6g mean_diff=%.6g idx=%d gpu=%.6g cpu=%.6g\n",
                    max_diff, mean_diff / (float)total, max_idx, attn_gpu[max_idx], attn_cpu[max_idx]);
            fprintf(stderr,
                    "cuda_flux2: dbl_attn blk0 q[0..3]=%.6g %.6g %.6g %.6g\n",
                    q_cpu[0], q_cpu[1], q_cpu[2], q_cpu[3]);
            fprintf(stderr,
                    "cuda_flux2: dbl_attn blk0 attn_gpu[0..3]=%.6g %.6g %.6g %.6g\n",
                    attn_gpu[0], attn_gpu[1], attn_gpu[2], attn_gpu[3]);
            fprintf(stderr,
                    "cuda_flux2: dbl_attn blk0 attn_cpu[0..3]=%.6g %.6g %.6g %.6g\n",
                    attn_cpu[0], attn_cpu[1], attn_cpu[2], attn_cpu[3]);
            free(q_cpu);
            free(k_cpu);
            free(v_cpu);
            free(attn_gpu);
            free(attn_cpu);
        }

        BF16(r->d_attn_out, n_tot * H);

        /* Output projections (img and txt use different weights) */
        GEMM(r->d_scratch1, b->img.proj_w, r->d_attn_out, d0, H, H, n_img, b->img.proj_scale);
        CUdeviceptr txt_attn = r->d_attn_out + (size_t)n_img * H * F;
        GEMM(r->d_scratch2, b->txt.proj_w, txt_attn, d0, H, H, n_txt, b->txt.proj_scale);

        /* Gated residual */
        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_a, n_img, H);
        op_gated_add(r, r->d_txt, r->d_scratch2, mt_gate_a, n_txt, H);
        BF16(r->d_img, n_img * H);
        BF16(r->d_txt, n_txt * H);

        /* FFN img */
        op_adaln(r, r->d_scratch1, r->d_img, mi_shift_f, mi_scale_f, n_img, H);
        BF16(r->d_scratch1, n_img * H);
        GEMM(r->d_scratch2, b->img.mlp_up_w, r->d_scratch1, d0, 2*n_ff, H, n_img, b->img.mlp_up_scale);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_img, n_ff);
        BF16(r->d_scratch3, n_img * n_ff);
        GEMM(r->d_scratch1, b->img.mlp_dn_w, r->d_scratch3, d0, H, n_ff, n_img, b->img.mlp_dn_scale);
        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_f, n_img, H);
        BF16(r->d_img, n_img * H);

        /* FFN txt */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_f, mt_scale_f, n_txt, H);
        BF16(r->d_scratch1, n_txt * H);
        GEMM(r->d_scratch2, b->txt.mlp_up_w, r->d_scratch1, d0, 2*n_ff, H, n_txt, b->txt.mlp_up_scale);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_txt, n_ff);
        BF16(r->d_scratch3, n_txt * n_ff);
        GEMM(r->d_scratch1, b->txt.mlp_dn_w, r->d_scratch3, d0, H, n_ff, n_txt, b->txt.mlp_dn_scale);
        op_gated_add(r, r->d_txt, r->d_scratch1, mt_gate_f, n_txt, H);
        BF16(r->d_txt, n_txt * H);
    }

    if (r->profile_step) { cuCtxSynchronize(); clock_gettime(CLOCK_MONOTONIC, &_pt2); }

    /* Deep profile accumulators (single-block per-stage) */
    double _ts_adaln = 0, _ts_lin1 = 0, _ts_norm = 0, _ts_attn = 0,
           _ts_swiglu = 0, _ts_lin2 = 0, _ts_resid = 0;
    struct timespec _ti0={0}, _ti1={0};
    #define _DEEP (r->profile_step >= 2)
    #define _TS_BEG() do { if (_DEEP) { cuCtxSynchronize(); clock_gettime(CLOCK_MONOTONIC, &_ti0); } } while(0)
    #define _TS_END(acc) do { if (_DEEP) { cuCtxSynchronize(); clock_gettime(CLOCK_MONOTONIC, &_ti1); \
        (acc) += (_ti1.tv_sec-_ti0.tv_sec)+(_ti1.tv_nsec-_ti0.tv_nsec)*1e-9; } } while(0)

    /* ---- Single-stream blocks ---- */
    /* Concatenate: joint = [txt, img] (txt first, then img) */
    cuMemcpyDtoD(r->d_joint, r->d_txt, (size_t)n_txt * H * F);
    cuMemcpyDtoD(r->d_joint + (size_t)n_txt * H * F, r->d_img, (size_t)n_img * H * F);

    int lin1_dim = 3 * H + 2 * n_ff;
    (void)lin1_dim;

    BF16(r->d_joint, n_tot * H);
    for (int bi = 0; bi < r->n_sgl; bi++) {
        flux2_gpu_sblk_t *b = &r->gpu_sblk[bi];

        /* adaLN */
        _TS_BEG();
        op_adaln(r, r->d_scratch1, r->d_joint, ms_shift, ms_scale, n_tot, H);
        BF16(r->d_scratch1, n_tot * H);
        _TS_END(_ts_adaln);

        /* linear1: compute Q, K, V, gate_up as separate GEMMs with weight slices */
        CUdeviceptr q_w = b->linear1_w;
        CUdeviceptr k_w = b->linear1_w + (size_t)H * H * WE;
        CUdeviceptr v_w = b->linear1_w + (size_t)2 * H * H * WE;
        CUdeviceptr gu_w = b->linear1_w + (size_t)3 * H * H * WE;

        _TS_BEG();
        GEMM(r->d_q, q_w, r->d_scratch1, d0, H, H, n_tot, b->linear1_scale);
        GEMM(r->d_k, k_w, r->d_scratch1, d0, H, H, n_tot, b->linear1_scale);
        GEMM(r->d_v, v_w, r->d_scratch1, d0, H, H, n_tot, b->linear1_scale);
        GEMM(r->d_scratch2, gu_w, r->d_scratch1, d0, 2*n_ff, H, n_tot, b->linear1_scale);
        _TS_END(_ts_lin1);

        /* Per-head Q/K norm + RoPE */
        _TS_BEG();
        op_rmsnorm_ph(r, r->d_q, b->q_norm, n_tot, nH, hd);
        op_rmsnorm_ph(r, r->d_k, b->k_norm, n_tot, nH, hd);
        BF16(r->d_q, n_tot * H);
        BF16(r->d_k, n_tot * H);
        BF16(r->d_v, n_tot * H);
        /* txt tokens at front [0..n_txt), img after */
        op_rope_txt(r, r->d_q, n_txt, nH, hd, theta);
        op_rope_txt(r, r->d_k, n_txt, nH, hd, theta);
        CUdeviceptr q_img = r->d_q + (size_t)n_txt * H * F;
        CUdeviceptr k_img = r->d_k + (size_t)n_txt * H * F;
        op_rope_img(r, q_img, n_img, nH, hd, lat_w_p, theta);
        op_rope_img(r, k_img, n_img, nH, hd, lat_w_p, theta);
        BF16(r->d_q, n_tot * H);
        BF16(r->d_k, n_tot * H);
        _TS_END(_ts_norm);

        /* Self-attention */
        _TS_BEG();
        op_attn(r, r->d_attn_out, r->d_q, r->d_k, r->d_v, n_tot, nH, hd);
        BF16(r->d_attn_out, n_tot * H);
        _TS_END(_ts_attn);

        /* Parallel MLP: SwiGLU on gate_up */
        _TS_BEG();
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_tot, n_ff);
        BF16(r->d_scratch3, n_tot * n_ff);
        _TS_END(_ts_swiglu);

        /* linear2: [H, H+n_ff] split into [H, H] (attn) + [H, n_ff] (mlp)
         * out = l2_attn @ attn_out + l2_mlp @ mlp_out */
        _TS_BEG();
        GEMM(r->d_scratch1, b->l2_attn_w, r->d_attn_out, d0, H, H, n_tot, b->l2_attn_scale);
        GEMM(r->d_scratch2, b->l2_mlp_w, r->d_scratch3, d0, H, n_ff, n_tot, b->l2_mlp_scale);
        _TS_END(_ts_lin2);

        /* Add the two halves: scratch1 += scratch2 */
        _TS_BEG();
        op_add(r, r->d_scratch1, r->d_scratch2, n_tot * H);

        /* Gated residual */
        op_gated_add(r, r->d_joint, r->d_scratch1, ms_gate, n_tot, H);
        BF16(r->d_joint, n_tot * H);
        _TS_END(_ts_resid);
    }

    /* ---- Output ---- */
    /* img portion = joint[n_txt:] */
    CUdeviceptr d_img_out = r->d_joint + (size_t)n_txt * H * F;

    /* Final adaLN: LN(img_out) * (1 + out_scale) + out_shift */
    if (r->profile_step) { cuCtxSynchronize(); clock_gettime(CLOCK_MONOTONIC, &_pt3); }
    #undef _DEEP
    #undef _TS_BEG
    #undef _TS_END

    GEMM(r->d_scratch1, r->d_out_mod_w, r->d_temb, d0, 2*H, H, 1, r->s_out_mod);
    BF16(r->d_scratch1, 2*H);
    CUdeviceptr out_shift = r->d_scratch1;
    CUdeviceptr out_scale = r->d_scratch1 + (size_t)H * F;

    op_adaln(r, r->d_scratch2, d_img_out, out_shift, out_scale, n_img, H);
    BF16(r->d_scratch2, n_img * H);

    /* Final linear: [pin, H] */
    GEMM(r->d_attn_out, r->d_out_proj_w, r->d_scratch2, d0, r->pin, H, n_img, r->s_out_proj);

    /* Download result */
    cuMemcpyDtoH(out, r->d_attn_out, (size_t)n_img * r->pin * F);
    cuCtxSynchronize();

    if (r->profile_step) {
        clock_gettime(CLOCK_MONOTONIC, &_pt4);
        #define _DT(a,b) (((b).tv_sec-(a).tv_sec)+((b).tv_nsec-(a).tv_nsec)*1e-9)
        double t_setup = _DT(_pt0,_pt1);
        double t_dbl   = _DT(_pt1,_pt2);
        double t_sgl   = _DT(_pt2,_pt3);
        double t_final = _DT(_pt3,_pt4);
        double t_tot   = _DT(_pt0,_pt4);
        fprintf(stderr,
            "[profile] setup %.3fs  double(%d) %.3fs  single(%d) %.3fs  final %.3fs  total %.3fs\n",
            t_setup, r->n_dbl, t_dbl, r->n_sgl, t_sgl, t_final, t_tot);
        if (r->profile_step >= 2) {
            fprintf(stderr,
                "[profile-deep single] adaln %.3f  lin1 %.3f  norm/rope %.3f  attn %.3f  swiglu %.3f  lin2 %.3f  resid %.3f  (sum %.3fs)\n",
                _ts_adaln, _ts_lin1, _ts_norm, _ts_attn, _ts_swiglu, _ts_lin2, _ts_resid,
                _ts_adaln+_ts_lin1+_ts_norm+_ts_attn+_ts_swiglu+_ts_lin2+_ts_resid);
        }
        #undef _DT
    }

    #undef GEMM
    #undef BF16
    return 0;
}

/* ---- Free ---- */

void cuda_flux2_free(cuda_flux2_runner *r) {
    if (!r) return;

    /* Free GPU block weights */
    if (r->gpu_dblk) {
        for (int i = 0; i < r->n_dbl; i++) {
            free_stream(&r->gpu_dblk[i].img);
            free_stream(&r->gpu_dblk[i].txt);
        }
        free(r->gpu_dblk);
    }
    if (r->gpu_sblk) {
        for (int i = 0; i < r->n_sgl; i++) {
            if (r->gpu_sblk[i].linear1_w)  cuMemFree(r->gpu_sblk[i].linear1_w);
            if (r->gpu_sblk[i].l2_attn_w)  cuMemFree(r->gpu_sblk[i].l2_attn_w);
            if (r->gpu_sblk[i].l2_mlp_w)   cuMemFree(r->gpu_sblk[i].l2_mlp_w);
            if (r->gpu_sblk[i].q_norm)     cuMemFree(r->gpu_sblk[i].q_norm);
            if (r->gpu_sblk[i].k_norm)     cuMemFree(r->gpu_sblk[i].k_norm);
        }
        free(r->gpu_sblk);
    }

    /* Free global weights */
    if (r->d_img_in_w)  cuMemFree(r->d_img_in_w);
    if (r->d_img_in_b)  cuMemFree(r->d_img_in_b);
    if (r->d_txt_in_w)  cuMemFree(r->d_txt_in_w);
    if (r->d_txt_in_b)  cuMemFree(r->d_txt_in_b);
    if (r->d_t_fc1_w)   cuMemFree(r->d_t_fc1_w);
    if (r->d_t_fc1_b)   cuMemFree(r->d_t_fc1_b);
    if (r->d_t_fc2_w)   cuMemFree(r->d_t_fc2_w);
    if (r->d_t_fc2_b)   cuMemFree(r->d_t_fc2_b);
    if (r->d_mod_img_w) cuMemFree(r->d_mod_img_w);
    if (r->d_mod_txt_w) cuMemFree(r->d_mod_txt_w);
    if (r->d_mod_sgl_w) cuMemFree(r->d_mod_sgl_w);
    if (r->d_out_mod_w) cuMemFree(r->d_out_mod_w);
    if (r->d_out_proj_w)cuMemFree(r->d_out_proj_w);

    /* Free activation buffers */
    if (r->d_img)       cuMemFree(r->d_img);
    if (r->d_txt)       cuMemFree(r->d_txt);
    if (r->d_joint)     cuMemFree(r->d_joint);
    if (r->d_temb)      cuMemFree(r->d_temb);
    if (r->d_temb_silu) cuMemFree(r->d_temb_silu);
    if (r->d_mod_img_v) cuMemFree(r->d_mod_img_v);
    if (r->d_mod_txt_v) cuMemFree(r->d_mod_txt_v);
    if (r->d_mod_sgl_v) cuMemFree(r->d_mod_sgl_v);
    if (r->d_q)         cuMemFree(r->d_q);
    if (r->d_k)         cuMemFree(r->d_k);
    if (r->d_v)         cuMemFree(r->d_v);
    if (r->d_attn_out)  cuMemFree(r->d_attn_out);
    if (r->d_scratch1)  cuMemFree(r->d_scratch1);
    if (r->d_scratch2)  cuMemFree(r->d_scratch2);
    if (r->d_scratch3)  cuMemFree(r->d_scratch3);
    if (r->d_img_in_buf)cuMemFree(r->d_img_in_buf);
    if (r->d_txt_in_buf)cuMemFree(r->d_txt_in_buf);

    /* FP8 attention workspace (lazily allocated by ensure_fp8_attn_buf) */
    if (r->d_q_fp8)      cuMemFree(r->d_q_fp8);
    if (r->d_k_fp8)      cuMemFree(r->d_k_fp8);
    if (r->d_v_fp8)      cuMemFree(r->d_v_fp8);
    if (r->d_qkv_max)    cuMemFree(r->d_qkv_max);
    if (r->d_qkv_scales) cuMemFree(r->d_qkv_scales);

    /* BF16 attention workspace (lazily allocated by ensure_bf16_attn_buf) */
    if (r->d_q_bf16) cuMemFree(r->d_q_bf16);
    if (r->d_k_bf16) cuMemFree(r->d_k_bf16);
    if (r->d_v_bf16) cuMemFree(r->d_v_bf16);

    /* FP8 v7 GEMM workspace (lazily allocated by ensure_x_fp8_buf) */
    if (r->d_x_fp8)   cuMemFree(r->d_x_fp8);
    if (r->d_x_max)   cuMemFree(r->d_x_max);
    if (r->d_x_scale) cuMemFree(r->d_x_scale);

    if (r->dit) flux2_dit_free(r->dit);
    if (r->vae) flux2_vae_free(r->vae);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->mod) cuModuleUnload(r->mod);
    cuDevicePrimaryCtxRelease(r->device);
    free(r);
}

/* ---- VAE decode ---- */

static CUdeviceptr flux2_vae_resblock_gpu(cuda_flux2_runner *r, CUdeviceptr x,
                                          const flux2_vae_resblock *rb,
                                          int h, int w, int num_groups) {
    int ci = rb->c_in, co = rb->c_out;
    int spatial = h * w;
    CUdeviceptr d_n1_w = gpu_upload_f32_or0(rb->norm1_w, ci);
    CUdeviceptr d_n1_b = gpu_upload_f32_or0(rb->norm1_b, ci);
    CUdeviceptr d_c1_w = vae_upload_w_3x3(r, rb->conv1_w, co * ci * 3 * 3);
    CUdeviceptr d_c1_b = gpu_upload_f32_or0(rb->conv1_b, co);
    CUdeviceptr d_n2_w = gpu_upload_f32_or0(rb->norm2_w, co);
    CUdeviceptr d_n2_b = gpu_upload_f32_or0(rb->norm2_b, co);
    CUdeviceptr d_c2_w = vae_upload_w_3x3(r, rb->conv2_w, co * co * 3 * 3);
    CUdeviceptr d_c2_b = gpu_upload_f32_or0(rb->conv2_b, co);
    CUdeviceptr d_sc_w = rb->skip_w ? gpu_upload_f32(rb->skip_w, co * ci) : 0;
    CUdeviceptr d_sc_b = gpu_upload_f32_or0(rb->skip_b, co);

    CUdeviceptr d_tmp1, d_tmp2, d_out;
    cuMemAlloc(&d_tmp1, (size_t)ci * spatial * sizeof(float));
    op_vae_groupnorm(r, d_tmp1, x, d_n1_w, d_n1_b, ci, spatial, num_groups);
    op_silu(r, d_tmp1, ci * spatial);

    cuMemAlloc(&d_tmp2, (size_t)co * spatial * sizeof(float));
    vae_conv_3x3(r, d_tmp2, d_tmp1, d_c1_w, d_c1_b, ci, h, w, co);
    cuMemFree(d_tmp1);

    cuMemAlloc(&d_tmp1, (size_t)co * spatial * sizeof(float));
    op_vae_groupnorm(r, d_tmp1, d_tmp2, d_n2_w, d_n2_b, co, spatial, num_groups);
    op_silu(r, d_tmp1, co * spatial);
    cuMemFree(d_tmp2);

    cuMemAlloc(&d_tmp2, (size_t)co * spatial * sizeof(float));
    vae_conv_3x3(r, d_tmp2, d_tmp1, d_c2_w, d_c2_b, co, h, w, co);
    cuMemFree(d_tmp1);

    cuMemAlloc(&d_out, (size_t)co * spatial * sizeof(float));
    if (d_sc_w) {
        op_vae_conv2d(r, d_out, x, d_sc_w, d_sc_b, ci, h, w, co, 1, 1, 0);
    } else {
        cuMemcpyDtoD(d_out, x, (size_t)co * spatial * sizeof(float));
    }
    op_add(r, d_out, d_tmp2, co * spatial);
    cuMemFree(d_tmp2);

    if (d_n1_w) cuMemFree(d_n1_w);
    if (d_n1_b) cuMemFree(d_n1_b);
    if (d_c1_w) cuMemFree(d_c1_w);
    if (d_c1_b) cuMemFree(d_c1_b);
    if (d_n2_w) cuMemFree(d_n2_w);
    if (d_n2_b) cuMemFree(d_n2_b);
    if (d_c2_w) cuMemFree(d_c2_w);
    if (d_c2_b) cuMemFree(d_c2_b);
    if (d_sc_w) cuMemFree(d_sc_w);
    if (d_sc_b) cuMemFree(d_sc_b);

    return d_out;
}

static CUdeviceptr flux2_vae_mid_attn_bridge(cuda_flux2_runner *r, CUdeviceptr x,
                                             const flux2_vae_attn *attn,
                                             int h, int w, int num_groups) {
    int c = attn->c;
    size_t n = (size_t)c * h * w;
    float *cpu_in = (float *)malloc(n * sizeof(float));
    float *cpu_out = (float *)malloc(n * sizeof(float));
    cuCtxSynchronize();
    cuMemcpyDtoH(cpu_in, x, n * sizeof(float));
    flux2_vae_mid_attn_forward(cpu_out, cpu_in, attn, h, w, num_groups);
    CUdeviceptr d_out = gpu_upload_f32(cpu_out, (int)n);
    free(cpu_in);
    free(cpu_out);
    return d_out;
}

/* All-GPU VAE mid-block attention: GroupNorm → transpose → QKV GEMM → attn → proj → transpose+residual */
static CUdeviceptr flux2_vae_mid_attn_gpu(cuda_flux2_runner *r, CUdeviceptr x,
                                           const flux2_vae_attn *attn,
                                           int h, int w, int num_groups) {
    int c = attn->c;
    int spatial = h * w;
    size_t feat_sz = (size_t)c * spatial;

    /* Decide BF16 vs F32 path once, up-front.
     * Requires: bf16 VAE enabled + all four helper kernels available (gemm_bf16,
     * f32_to_bf16, transpose_f32_to_bf16, softmax_row). Because all kernels are
     * loaded together from the same NVRTC module, they are either all present
     * or all absent, so this is effectively a single toggle. */
    int use_bf16 = (r->use_bf16_vae &&
                    r->fn_gemm_bf16 &&
                    r->fn_f32_to_bf16 &&
                    r->fn_transpose_f32_to_bf16 &&
                    r->fn_softmax_row);
    CUdeviceptr d_norm_w = gpu_upload_f32(attn->norm_w, c);
    CUdeviceptr d_norm_b = gpu_upload_f32(attn->norm_b, c);
    CUdeviceptr d_q_w = use_bf16 ? gpu_upload_f32_as_bf16(attn->q_w, c * c) : gpu_upload_f32(attn->q_w, c * c);
    CUdeviceptr d_k_w = use_bf16 ? gpu_upload_f32_as_bf16(attn->k_w, c * c) : gpu_upload_f32(attn->k_w, c * c);
    CUdeviceptr d_v_w = use_bf16 ? gpu_upload_f32_as_bf16(attn->v_w, c * c) : gpu_upload_f32(attn->v_w, c * c);
    CUdeviceptr d_out_w = use_bf16 ? gpu_upload_f32_as_bf16(attn->out_w, c * c) : gpu_upload_f32(attn->out_w, c * c);
    CUdeviceptr d_q_b = gpu_upload_f32_or0(attn->q_b, c);
    CUdeviceptr d_k_b = gpu_upload_f32_or0(attn->k_b, c);
    CUdeviceptr d_v_b = gpu_upload_f32_or0(attn->v_b, c);
    CUdeviceptr d_out_b = gpu_upload_f32_or0(attn->out_b, c);

    /* 1. GroupNorm: x [c, spatial] → normed [c, spatial] */
    CUdeviceptr d_normed;
    cuMemAlloc(&d_normed, feat_sz * sizeof(float));
    op_vae_groupnorm(r, d_normed, x, d_norm_w, d_norm_b, c, spatial, num_groups);
    cuMemFree(d_norm_w); cuMemFree(d_norm_b);

    /* 2. Transpose [c, spatial] → [spatial, c] */
    CUdeviceptr d_normed_t;
    cuMemAlloc(&d_normed_t, feat_sz * sizeof(float));
    op_transpose_2d(r, d_normed_t, d_normed, c, spatial);
    cuMemFree(d_normed);

    /* 3. Q, K, V projections: [spatial, c] × [c, c]^T → [spatial, c] */
    CUdeviceptr d_q, d_k, d_v;
    cuMemAlloc(&d_q, (size_t)spatial * c * sizeof(float));
    cuMemAlloc(&d_k, (size_t)spatial * c * sizeof(float));
    cuMemAlloc(&d_v, (size_t)spatial * c * sizeof(float));
    if (use_bf16) {
        op_gemm_bf16(r, d_q, d_q_w, d_normed_t, d_q_b, c, c, spatial);
        op_gemm_bf16(r, d_k, d_k_w, d_normed_t, d_k_b, c, c, spatial);
        op_gemm_bf16(r, d_v, d_v_w, d_normed_t, d_v_b, c, c, spatial);
    } else {
        op_gemm_f32(r, d_q, d_q_w, d_normed_t, d_q_b, c, c, spatial);
        op_gemm_f32(r, d_k, d_k_w, d_normed_t, d_k_b, c, c, spatial);
        op_gemm_f32(r, d_v, d_v_w, d_normed_t, d_v_b, c, c, spatial);
    }
    cuMemFree(d_normed_t);
    cuMemFree(d_q_w); cuMemFree(d_k_w); cuMemFree(d_v_w);
    if (d_q_b) cuMemFree(d_q_b);
    if (d_k_b) cuMemFree(d_k_b);
    if (d_v_b) cuMemFree(d_v_b);

    /* 4. Single-head attention via BF16 MMA 3-step: S = Q·K^T / sqrt(c), softmax, O = P·V */
    CUdeviceptr d_attn_out;
    cuMemAlloc(&d_attn_out, (size_t)spatial * c * sizeof(float));
    if (use_bf16) {
        /* gemm_bf16_f32 takes X as F32 and W as BF16. So only K and V need BF16 conversion. */
        CUdeviceptr d_K_bf, d_VT_bf, d_S;
        cuMemAlloc(&d_K_bf, (size_t)spatial * c * sizeof(uint16_t));
        cuMemAlloc(&d_VT_bf, (size_t)spatial * c * sizeof(uint16_t));
        cuMemAlloc(&d_S, (size_t)spatial * spatial * sizeof(float));
        /* K BF16 (as "weight" [spatial, c]) */
        op_f32_to_bf16(r, d_K_bf, d_k, spatial * c);
        /* V^T BF16 [c, spatial] (needed because gemm W layout is [n_out, n_in]) */
        op_transpose_f32_to_bf16(r, d_VT_bf, d_v, spatial, c);
        cuMemFree(d_k); cuMemFree(d_v);
        /* S = Q @ K^T:  Y[i,j] = Σ_k X[i,k] * W[j,k] = Σ_k Q[i,k] * K[j,k] */
        op_gemm_bf16(r, d_S, d_K_bf, d_q, 0, spatial, c, spatial);
        cuMemFree(d_q);
        /* Row-wise softmax with scale = 1/sqrt(c), in-place on F32 S */
        float attn_scale = 1.0f / sqrtf((float)c);
        op_softmax_row(r, d_S, spatial, spatial, attn_scale);
        /* O = P @ V:  Y[i,d] = Σ_k X[i,k] * W[d,k] = Σ_k P[i,k] * V_T[d,k] = Σ_k P[i,k] * V[k,d] */
        op_gemm_bf16(r, d_attn_out, d_VT_bf, d_S, 0, c, spatial, spatial);
        cuMemFree(d_K_bf); cuMemFree(d_VT_bf); cuMemFree(d_S);
    } else {
        op_vae_attn(r, d_attn_out, d_q, d_k, d_v, spatial, c);
        cuMemFree(d_q); cuMemFree(d_k); cuMemFree(d_v);
    }

    /* 5. Output projection: [spatial, c] × [c, c]^T → [spatial, c] */
    CUdeviceptr d_proj;
    cuMemAlloc(&d_proj, (size_t)spatial * c * sizeof(float));
    if (use_bf16)
        op_gemm_bf16(r, d_proj, d_out_w, d_attn_out, d_out_b, c, c, spatial);
    else
        op_gemm_f32(r, d_proj, d_out_w, d_attn_out, d_out_b, c, c, spatial);
    cuMemFree(d_attn_out);
    cuMemFree(d_out_w);
    if (d_out_b) cuMemFree(d_out_b);

    /* 6. Transpose [spatial, c] → [c, spatial] and add residual */
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, feat_sz * sizeof(float));
    op_transpose_add(r, d_out, x, d_proj, c, spatial);
    cuMemFree(d_proj);

    return d_out;
}

int cuda_flux2_vae_decode(cuda_flux2_runner *r,
                          const float *latent, int lat_h, int lat_w,
                          float *out_rgb) {
    if (!r->vae) {
        fprintf(stderr, "cuda_flux2: VAE not loaded\n");
        return -1;
    }

    flux2_vae_model *m = r->vae;
    int lc = m->latent_channels;
    int ng = m->num_groups;
    int h = lat_h, w = lat_w;
    int c = lc;
    int spatial = h * w;

    /* Optional per-phase timing (FLUX2_VAE_TIMING=1) */
    int vae_dbg = flux2_env_enabled("FLUX2_VAE_TIMING");
    CUevent ev_start = 0, ev_pqc = 0, ev_cin = 0, ev_mid = 0;
    CUevent ev_up[4] = {0};
    CUevent ev_tail = 0;
    if (vae_dbg) {
        cuEventCreate(&ev_start, 0);
        cuEventCreate(&ev_pqc, 0);
        cuEventCreate(&ev_cin, 0);
        cuEventCreate(&ev_mid, 0);
        for (int i = 0; i < 4; i++) cuEventCreate(&ev_up[i], 0);
        cuEventCreate(&ev_tail, 0);
        cuEventRecord(ev_start, r->stream);
    }

    CUdeviceptr d_x = gpu_upload_f32(latent, lc * h * w);

    /* NOTE: BN stats (bn.running_mean/var) are training artifacts — do NOT apply.
     * The DiT outputs latents in the correct space for the VAE decoder.
     * Applying BN denorm over-saturates and causes visible artifacts. */

    if (m->pqc_w) {
        CUdeviceptr d_pqc_w = gpu_upload_f32(m->pqc_w, lc * lc);
        CUdeviceptr d_pqc_b = gpu_upload_f32_or0(m->pqc_b, lc);
        CUdeviceptr d_tmp;
        cuMemAlloc(&d_tmp, (size_t)lc * h * w * sizeof(float));
        op_vae_conv2d(r, d_tmp, d_x, d_pqc_w, d_pqc_b, lc, h, w, lc, 1, 1, 0);
        cuMemFree(d_x);
        cuMemFree(d_pqc_w);
        if (d_pqc_b) cuMemFree(d_pqc_b);
        d_x = d_tmp;
    }
    if (vae_dbg) cuEventRecord(ev_pqc, r->stream);

    {
        int co = m->conv_in_out_ch;
        CUdeviceptr d_w = vae_upload_w_3x3(r, m->conv_in_w, co * lc * 3 * 3);
        CUdeviceptr d_b = gpu_upload_f32_or0(m->conv_in_b, co);
        CUdeviceptr d_tmp;
        cuMemAlloc(&d_tmp, (size_t)co * h * w * sizeof(float));
        vae_conv_3x3(r, d_tmp, d_x, d_w, d_b, lc, h, w, co);
        cuMemFree(d_x);
        cuMemFree(d_w);
        if (d_b) cuMemFree(d_b);
        d_x = d_tmp;
        c = co;
    }
    if (vae_dbg) cuEventRecord(ev_cin, r->stream);

    {
        CUdeviceptr d_tmp = flux2_vae_resblock_gpu(r, d_x, &m->mid_res0, h, w, ng);
        cuMemFree(d_x);
        d_x = d_tmp;
    }
    {
        CUdeviceptr d_tmp = flux2_vae_mid_attn_gpu(r, d_x, &m->mid_attn, h, w, ng);
        cuMemFree(d_x);
        d_x = d_tmp;
    }
    {
        CUdeviceptr d_tmp = flux2_vae_resblock_gpu(r, d_x, &m->mid_res1, h, w, ng);
        cuMemFree(d_x);
        d_x = d_tmp;
    }
    if (vae_dbg) cuEventRecord(ev_mid, r->stream);

    for (int bi = 0; bi < 4; bi++) {
        {
            CUdeviceptr d_tmp = flux2_vae_resblock_gpu(r, d_x, &m->up_res[bi][0], h, w, ng);
            cuMemFree(d_x);
            d_x = d_tmp;
            c = m->up_res[bi][0].c_out;
        }
        {
            CUdeviceptr d_tmp = flux2_vae_resblock_gpu(r, d_x, &m->up_res[bi][1], h, w, ng);
            cuMemFree(d_x);
            d_x = d_tmp;
            c = m->up_res[bi][1].c_out;
        }
        {
            CUdeviceptr d_tmp = flux2_vae_resblock_gpu(r, d_x, &m->up_res[bi][2], h, w, ng);
            cuMemFree(d_x);
            d_x = d_tmp;
            c = m->up_res[bi][2].c_out;
        }
        if (m->up_has_sample[bi]) {
            CUdeviceptr d_up = op_vae_upsample2x(r, d_x, c, h, w);
            CUdeviceptr d_w = vae_upload_w_3x3(r, m->up_sample[bi].conv_w, c * c * 3 * 3);
            CUdeviceptr d_b = gpu_upload_f32_or0(m->up_sample[bi].conv_b, c);
            CUdeviceptr d_tmp;
            h *= 2;
            w *= 2;
            cuMemAlloc(&d_tmp, (size_t)c * h * w * sizeof(float));
            vae_conv_3x3(r, d_tmp, d_up, d_w, d_b, c, h, w, c);
            cuMemFree(d_x);
            cuMemFree(d_up);
            cuMemFree(d_w);
            if (d_b) cuMemFree(d_b);
            d_x = d_tmp;
        }
        if (vae_dbg) cuEventRecord(ev_up[bi], r->stream);
    }

    spatial = h * w;
    {
        CUdeviceptr d_g = gpu_upload_f32_or0(m->norm_out_w, c);
        CUdeviceptr d_b = gpu_upload_f32_or0(m->norm_out_b, c);
        CUdeviceptr d_tmp;
        cuMemAlloc(&d_tmp, (size_t)c * spatial * sizeof(float));
        op_vae_groupnorm(r, d_tmp, d_x, d_g, d_b, c, spatial, ng);
        op_silu(r, d_tmp, c * spatial);
        cuMemFree(d_x);
        if (d_g) cuMemFree(d_g);
        if (d_b) cuMemFree(d_b);
        d_x = d_tmp;
    }
    {
        CUdeviceptr d_w = vae_upload_w_3x3(r, m->conv_out_w, 3 * c * 3 * 3);
        CUdeviceptr d_b = gpu_upload_f32_or0(m->conv_out_b, 3);
        CUdeviceptr d_rgb;
        cuMemAlloc(&d_rgb, (size_t)3 * spatial * sizeof(float));
        vae_conv_3x3(r, d_rgb, d_x, d_w, d_b, c, h, w, 3);
        cuMemFree(d_x);
        cuMemFree(d_w);
        if (d_b) cuMemFree(d_b);
        d_x = d_rgb;
    }

    if (vae_dbg) cuEventRecord(ev_tail, r->stream);
    cuCtxSynchronize();
    cuMemcpyDtoH(out_rgb, d_x, (size_t)3 * h * w * sizeof(float));
    cuMemFree(d_x);

    if (vae_dbg) {
        float t_pqc, t_cin, t_mid, t_tail, t_up[4];
        cuEventElapsedTime(&t_pqc, ev_start, ev_pqc);
        cuEventElapsedTime(&t_cin, ev_pqc, ev_cin);
        cuEventElapsedTime(&t_mid, ev_cin, ev_mid);
        for (int i = 0; i < 4; i++)
            cuEventElapsedTime(&t_up[i], i == 0 ? ev_mid : ev_up[i-1], ev_up[i]);
        cuEventElapsedTime(&t_tail, ev_up[3], ev_tail);
        fprintf(stderr, "VAE timing: pqc=%.1fms cin=%.1fms mid=%.1fms up0=%.1fms up1=%.1fms up2=%.1fms up3=%.1fms tail=%.1fms\n",
                t_pqc, t_cin, t_mid, t_up[0], t_up[1], t_up[2], t_up[3], t_tail);
        cuEventDestroy(ev_start); cuEventDestroy(ev_pqc); cuEventDestroy(ev_cin);
        cuEventDestroy(ev_mid);
        for (int i = 0; i < 4; i++) cuEventDestroy(ev_up[i]);
        cuEventDestroy(ev_tail);
    }
    return 0;
}

#endif /* CUDA_FLUX2_RUNNER_IMPLEMENTATION */
#endif /* CUDA_FLUX2_RUNNER_H */
