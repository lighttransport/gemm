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

/* ---- Runner struct ---- */

struct cuda_flux2_runner {
    CUdevice   device;
    CUcontext  ctx;
    CUstream   stream;
    int        verbose;

    CUmodule   mod;
    CUfunction fn_gemm, fn_gemm_f16, fn_gemm_fp8, fn_gemm_fp8_bf16, fn_layernorm, fn_silu, fn_adaln, fn_gated_add;
    CUfunction fn_rmsnorm_ph, fn_swiglu, fn_flash_attn, fn_add;
    CUfunction fn_vae_groupnorm, fn_vae_conv2d, fn_vae_upsample2x, fn_vae_latent_bn;
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
    int max_tok;
    int gpu_loaded;
    int use_gpu_dbl_attn;
    int debug_dbl_attn;
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
static void op_gemm_scaled(cuda_flux2_runner *r, CUdeviceptr Y, CUdeviceptr W,
                           CUdeviceptr X, CUdeviceptr bias,
                           int n_out, int n_in, int n_tok, float w_scale) {
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    if (r->use_fp8_gemm && w_scale != 1.0f) {
        /* FP8 weights with per-tensor scale; pick BF16-act variant if enabled */
        CUfunction fn = r->fp8_bf16_act ? r->fn_gemm_fp8_bf16 : r->fn_gemm_fp8;
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &w_scale};
        cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, r->stream, args, NULL);
    } else {
        /* F32 or F16 (scale=1.0 means BF16→F32 fallback) */
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

static void op_attn(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr q,
                    CUdeviceptr k, CUdeviceptr v, int n_tok, int n_heads, int head_dim) {
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

static void op_vae_conv2d(cuda_flux2_runner *r, CUdeviceptr out, CUdeviceptr in,
                          CUdeviceptr w, CUdeviceptr b,
                          int ci, int h, int wid, int co, int kh, int kw, int pad) {
    int total = co * h * wid;
    void *args[] = {&out, &in, &w, &b, &ci, &h, &wid, &co, &kh, &kw, &pad};
    cuLaunchKernel(r->fn_vae_conv2d, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
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

/* Upload a weight tensor from safetensors as raw FP8 bytes, return device ptr + scale.
 * If the tensor is BF16/F32 (not FP8), falls back to F32 upload with scale=1.0. */
static CUdeviceptr gpu_upload_st_fp8(st_context *st, const char *wname, const char *sname,
                                      float *out_scale) {
    int widx = safetensors_find(st, wname);
    if (widx < 0) { *out_scale = 1.0f; return 0; }
    const char *dtype = safetensors_dtype(st, widx);
    const uint64_t *sh = safetensors_shape(st, widx);
    int nd = safetensors_ndims(st, widx);
    size_t n = (nd >= 2) ? (size_t)sh[0] * sh[1] : (size_t)sh[0];

    if (strcmp(dtype, "F8_E4M3") == 0) {
        /* Native FP8: upload raw bytes */
        const void *data = safetensors_data(st, widx);
        *out_scale = 1.0f;
        if (sname) {
            int sidx = safetensors_find(st, sname);
            if (sidx >= 0) *out_scale = *(const float *)safetensors_data(st, sidx);
        }
        return gpu_upload_bytes(data, n);  /* 1 byte per element */
    } else {
        /* BF16/F32: dequant to F32, scale=1.0, use F32 upload */
        *out_scale = 1.0f;
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

    cuStreamCreate(&r->stream, 0);

    /* Get function handles */
    cuModuleGetFunction(&r->fn_gemm,        mod, "gemm_f32_f32");
    cuModuleGetFunction(&r->fn_gemm_f16,    mod, "gemm_tiled_f16_f32");
    cuModuleGetFunction(&r->fn_gemm_fp8,    mod, "gemm_tiled_fp8_f32");
    cuModuleGetFunction(&r->fn_gemm_fp8_bf16, mod, "gemm_tiled_fp8_bf16");
    cuModuleGetFunction(&r->fn_layernorm,   mod, "layernorm_f32");
    cuModuleGetFunction(&r->fn_silu,        mod, "silu_f32");
    cuModuleGetFunction(&r->fn_adaln,       mod, "adaln_modulate_f32");
    cuModuleGetFunction(&r->fn_gated_add,   mod, "gated_add_f32");
    cuModuleGetFunction(&r->fn_rmsnorm_ph,  mod, "rmsnorm_per_head_f32");
    cuModuleGetFunction(&r->fn_swiglu,      mod, "flux2_swiglu");
    cuModuleGetFunction(&r->fn_flash_attn,  mod, "flash_attn_f32");
    cuModuleGetFunction(&r->fn_add,         mod, "add_f32");
    cuModuleGetFunction(&r->fn_vae_groupnorm, mod, "flux2_vae_groupnorm_f32");
    cuModuleGetFunction(&r->fn_vae_conv2d,    mod, "flux2_vae_conv2d_f32");
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
    gs->qkv_scale = gs->proj_scale = gs->mlp_up_scale = gs->mlp_dn_scale = 1.0f;
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
                    l2_scale = 1.0f;  /* already dequantized */
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
            r->gpu_sblk[i].linear1_scale = 1.0f;
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
            r->gpu_sblk[i].l2_attn_scale = r->gpu_sblk[i].l2_mlp_scale = 1.0f;
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
        op_adaln(r, r->d_scratch1, r->d_joint, ms_shift, ms_scale, n_tot, H);
        BF16(r->d_scratch1, n_tot * H);

        /* linear1: compute Q, K, V, gate_up as separate GEMMs with weight slices */
        CUdeviceptr q_w = b->linear1_w;
        CUdeviceptr k_w = b->linear1_w + (size_t)H * H * WE;
        CUdeviceptr v_w = b->linear1_w + (size_t)2 * H * H * WE;
        CUdeviceptr gu_w = b->linear1_w + (size_t)3 * H * H * WE;

        GEMM(r->d_q, q_w, r->d_scratch1, d0, H, H, n_tot, b->linear1_scale);
        GEMM(r->d_k, k_w, r->d_scratch1, d0, H, H, n_tot, b->linear1_scale);
        GEMM(r->d_v, v_w, r->d_scratch1, d0, H, H, n_tot, b->linear1_scale);
        GEMM(r->d_scratch2, gu_w, r->d_scratch1, d0, 2*n_ff, H, n_tot, b->linear1_scale);

        /* Per-head Q/K norm + RoPE */
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

        /* Self-attention */
        op_attn(r, r->d_attn_out, r->d_q, r->d_k, r->d_v, n_tot, nH, hd);
        BF16(r->d_attn_out, n_tot * H);

        /* Parallel MLP: SwiGLU on gate_up */
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_tot, n_ff);
        BF16(r->d_scratch3, n_tot * n_ff);

        /* linear2: [H, H+n_ff] split into [H, H] (attn) + [H, n_ff] (mlp)
         * out = l2_attn @ attn_out + l2_mlp @ mlp_out */
        GEMM(r->d_scratch1, b->l2_attn_w, r->d_attn_out, d0, H, H, n_tot, b->l2_attn_scale);
        GEMM(r->d_scratch2, b->l2_mlp_w, r->d_scratch3, d0, H, n_ff, n_tot, b->l2_mlp_scale);

        /* Add the two halves: scratch1 += scratch2 */
        op_add(r, r->d_scratch1, r->d_scratch2, n_tot * H);

        /* Gated residual */
        op_gated_add(r, r->d_joint, r->d_scratch1, ms_gate, n_tot, H);
        BF16(r->d_joint, n_tot * H);
    }

    /* ---- Output ---- */
    /* img portion = joint[n_txt:] */
    CUdeviceptr d_img_out = r->d_joint + (size_t)n_txt * H * F;

    /* Final adaLN: LN(img_out) * (1 + out_scale) + out_shift */
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
    CUdeviceptr d_c1_w = gpu_upload_f32(rb->conv1_w, co * ci * 3 * 3);
    CUdeviceptr d_c1_b = gpu_upload_f32_or0(rb->conv1_b, co);
    CUdeviceptr d_n2_w = gpu_upload_f32_or0(rb->norm2_w, co);
    CUdeviceptr d_n2_b = gpu_upload_f32_or0(rb->norm2_b, co);
    CUdeviceptr d_c2_w = gpu_upload_f32(rb->conv2_w, co * co * 3 * 3);
    CUdeviceptr d_c2_b = gpu_upload_f32_or0(rb->conv2_b, co);
    CUdeviceptr d_sc_w = rb->skip_w ? gpu_upload_f32(rb->skip_w, co * ci) : 0;
    CUdeviceptr d_sc_b = gpu_upload_f32_or0(rb->skip_b, co);

    CUdeviceptr d_tmp1, d_tmp2, d_out;
    cuMemAlloc(&d_tmp1, (size_t)ci * spatial * sizeof(float));
    op_vae_groupnorm(r, d_tmp1, x, d_n1_w, d_n1_b, ci, spatial, num_groups);
    op_silu(r, d_tmp1, ci * spatial);

    cuMemAlloc(&d_tmp2, (size_t)co * spatial * sizeof(float));
    op_vae_conv2d(r, d_tmp2, d_tmp1, d_c1_w, d_c1_b, ci, h, w, co, 3, 3, 1);
    cuMemFree(d_tmp1);

    cuMemAlloc(&d_tmp1, (size_t)co * spatial * sizeof(float));
    op_vae_groupnorm(r, d_tmp1, d_tmp2, d_n2_w, d_n2_b, co, spatial, num_groups);
    op_silu(r, d_tmp1, co * spatial);
    cuMemFree(d_tmp2);

    cuMemAlloc(&d_tmp2, (size_t)co * spatial * sizeof(float));
    op_vae_conv2d(r, d_tmp2, d_tmp1, d_c2_w, d_c2_b, co, h, w, co, 3, 3, 1);
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

    /* Upload attention weights (F32, uploaded per-call since VAE is one-shot) */
    CUdeviceptr d_norm_w = gpu_upload_f32(attn->norm_w, c);
    CUdeviceptr d_norm_b = gpu_upload_f32(attn->norm_b, c);
    CUdeviceptr d_q_w = gpu_upload_f32(attn->q_w, c * c);
    CUdeviceptr d_k_w = gpu_upload_f32(attn->k_w, c * c);
    CUdeviceptr d_v_w = gpu_upload_f32(attn->v_w, c * c);
    CUdeviceptr d_out_w = gpu_upload_f32(attn->out_w, c * c);
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
    op_gemm_f32(r, d_q, d_q_w, d_normed_t, d_q_b, c, c, spatial);
    op_gemm_f32(r, d_k, d_k_w, d_normed_t, d_k_b, c, c, spatial);
    op_gemm_f32(r, d_v, d_v_w, d_normed_t, d_v_b, c, c, spatial);
    cuMemFree(d_normed_t);
    cuMemFree(d_q_w); cuMemFree(d_k_w); cuMemFree(d_v_w);
    if (d_q_b) cuMemFree(d_q_b);
    if (d_k_b) cuMemFree(d_k_b);
    if (d_v_b) cuMemFree(d_v_b);

    /* 4. Single-head flash attention: Q,K,V [spatial, c] → attn_out [spatial, c] */
    CUdeviceptr d_attn_out;
    cuMemAlloc(&d_attn_out, (size_t)spatial * c * sizeof(float));
    op_vae_attn(r, d_attn_out, d_q, d_k, d_v, spatial, c);
    cuMemFree(d_q); cuMemFree(d_k); cuMemFree(d_v);

    /* 5. Output projection: [spatial, c] × [c, c]^T → [spatial, c] */
    CUdeviceptr d_proj;
    cuMemAlloc(&d_proj, (size_t)spatial * c * sizeof(float));
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

    {
        int co = m->conv_in_out_ch;
        CUdeviceptr d_w = gpu_upload_f32(m->conv_in_w, co * lc * 3 * 3);
        CUdeviceptr d_b = gpu_upload_f32_or0(m->conv_in_b, co);
        CUdeviceptr d_tmp;
        cuMemAlloc(&d_tmp, (size_t)co * h * w * sizeof(float));
        op_vae_conv2d(r, d_tmp, d_x, d_w, d_b, lc, h, w, co, 3, 3, 1);
        cuMemFree(d_x);
        cuMemFree(d_w);
        if (d_b) cuMemFree(d_b);
        d_x = d_tmp;
        c = co;
    }

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
            CUdeviceptr d_w = gpu_upload_f32(m->up_sample[bi].conv_w, c * c * 3 * 3);
            CUdeviceptr d_b = gpu_upload_f32_or0(m->up_sample[bi].conv_b, c);
            CUdeviceptr d_tmp;
            h *= 2;
            w *= 2;
            cuMemAlloc(&d_tmp, (size_t)c * h * w * sizeof(float));
            op_vae_conv2d(r, d_tmp, d_up, d_w, d_b, c, h, w, c, 3, 3, 1);
            cuMemFree(d_x);
            cuMemFree(d_up);
            cuMemFree(d_w);
            if (d_b) cuMemFree(d_b);
            d_x = d_tmp;
        }
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
        CUdeviceptr d_w = gpu_upload_f32(m->conv_out_w, 3 * c * 3 * 3);
        CUdeviceptr d_b = gpu_upload_f32_or0(m->conv_out_b, 3);
        CUdeviceptr d_rgb;
        cuMemAlloc(&d_rgb, (size_t)3 * spatial * sizeof(float));
        op_vae_conv2d(r, d_rgb, d_x, d_w, d_b, c, h, w, 3, 3, 3, 1);
        cuMemFree(d_x);
        cuMemFree(d_w);
        if (d_b) cuMemFree(d_b);
        d_x = d_rgb;
    }

    cuCtxSynchronize();
    cuMemcpyDtoH(out_rgb, d_x, (size_t)3 * h * w * sizeof(float));
    cuMemFree(d_x);
    return 0;
}

#endif /* CUDA_FLUX2_RUNNER_IMPLEMENTATION */
#endif /* CUDA_FLUX2_RUNNER_H */
