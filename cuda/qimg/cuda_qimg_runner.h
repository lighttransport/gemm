/*
 * cuda_qimg_runner.h - CUDA Qwen-Image text-to-image runner
 *
 * Uses NVRTC to compile kernels at runtime. Processes the 60-block MMDiT
 * by loading one block at a time (dequant FP8→F16 on CPU, upload, compute,
 * free) to fit within 16GB VRAM.
 *
 * Weight format: FP8 E4M3 safetensors (from Comfy-Org/Qwen-Image_ComfyUI)
 *
 * Pipeline:
 *   1. Text encoder (CPU) → hidden states [N_txt, 3584]
 *   2. MMDiT denoising (GPU) × N_steps → latent [16, H/8, W/8]
 *   3. VAE decoder (GPU) → RGB image [3, H, W]
 *
 * Usage:
 *   cuda_qimg_runner *r = cuda_qimg_init(0, 1);
 *   cuda_qimg_load_dit(r, "qwen_image_fp8_e4m3fn.safetensors");
 *   cuda_qimg_load_vae(r, "qwen_image_vae.safetensors");
 *   cuda_qimg_dit_step(r, img_tok, n_img, txt_tok, n_txt, t, out);
 *   cuda_qimg_free(r);
 */
#ifndef CUDA_QIMG_RUNNER_H
#define CUDA_QIMG_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_qimg_runner cuda_qimg_runner;

cuda_qimg_runner *cuda_qimg_init(int device_id, int verbose);
int  cuda_qimg_load_dit(cuda_qimg_runner *r, const char *safetensors_path);
int  cuda_qimg_load_vae(cuda_qimg_runner *r, const char *safetensors_path);
void cuda_qimg_free(cuda_qimg_runner *r);

/* Run single DiT denoising step on GPU.
 *   img_tokens: [n_img, 64] patchified latent (CPU)
 *   txt_tokens: [n_txt, 3584] text hidden states (CPU)
 *   out: [n_img, 64] velocity prediction (CPU, pre-allocated)
 *   Returns 0 on success. */
int cuda_qimg_dit_step(cuda_qimg_runner *r,
                       const float *img_tokens, int n_img,
                       const float *txt_tokens, int n_txt,
                       float timestep, float *out);

/* VAE decode on GPU. Frees preloaded DiT blocks to make room.
 *   latent: [16, lat_h, lat_w] F32 (CPU)
 *   out_rgb: [3, lat_h*8, lat_w*8] F32 (CPU, pre-allocated)
 *   Returns 0 on success. */
int cuda_qimg_vae_decode(cuda_qimg_runner *r,
                         const float *latent, int lat_h, int lat_w,
                         float *out_rgb);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef CUDA_QIMG_RUNNER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_kernels_common.h"
#include "../../common/safetensors.h"

/* ---- FP8 E4M3 → F32 CPU conversion ---- */

static float fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0) {
        /* Subnormal: value = (-1)^s × 2^(1-7) × (0.mant) */
        f = ldexpf((float)mant / 8.0f, -6);
    } else if (exp == 15 && mant == 7) {
        return 0.0f;  /* NaN → 0 (safe for GEMM) */
    } else {
        /* Normal: value = (-1)^s × 2^(exp-7) × (1 + mant/8) */
        f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -f : f;
}

/* ---- Kernel source for Qwen-Image specific ops ---- */

static const char *qimg_kernel_src =
"\n/* ---- Qwen-Image specific kernels ---- */\n"

/* GPU-side FP8 E4M3 → F16 dequant via constant memory LUT */
"__device__ __constant__ unsigned short d_fp8_to_f16_lut[256];\n"
"\n"
/* Bulk dequant kernel: convert [n] FP8 bytes → [n] F16 values using LUT */
"__global__ void dequant_fp8_to_f16(const unsigned char *__restrict__ src,\n"
"    unsigned short *__restrict__ dst, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    dst[i] = d_fp8_to_f16_lut[src[i]];\n"
"}\n"
"\n"

/* Per-head RMSNorm: x[N, dim], w[head_dim], n_heads, head_dim */
"__global__ void rmsnorm_per_head_f32(float *__restrict__ x,\n"
"    const float *__restrict__ w, int N, int n_heads, int head_dim) {\n"
"    int tok = blockIdx.x;\n"
"    int h = blockIdx.y;\n"
"    if (tok >= N || h >= n_heads) return;\n"
"    int dim = n_heads * head_dim;\n"
"    float *hd = x + tok * dim + h * head_dim;\n"
"    float ss = 0.0f;\n"
"    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)\n"
"        ss += hd[i] * hd[i];\n"
"    /* Warp reduce */\n"
"    for (int m = 16; m > 0; m >>= 1)\n"
"        ss += __shfl_xor_sync(0xFFFFFFFF, ss, m);\n"
"    float inv = rsqrtf(ss / (float)head_dim + 1e-6f);\n"
"    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)\n"
"        hd[i] *= inv * w[i];\n"
"}\n"

/* adaLN modulate: out = LN(x) * (1+scale) + shift */
"__global__ void adaln_modulate_f32(float *__restrict__ out,\n"
"    const float *__restrict__ x, const float *__restrict__ shift,\n"
"    const float *__restrict__ scale, int N, int dim) {\n"
"    int tok = blockIdx.x;\n"
"    if (tok >= N) return;\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    /* Compute mean */\n"
"    float s = 0;\n"
"    for (int i = tid; i < dim; i += blockDim.x) s += x[tok*dim+i];\n"
"    sdata[tid] = s; __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)dim;\n"
"    __syncthreads();\n"
"    /* Compute var */\n"
"    s = 0;\n"
"    for (int i = tid; i < dim; i += blockDim.x) { float d = x[tok*dim+i] - mean; s += d*d; }\n"
"    sdata[tid] = s; __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)dim + 1e-6f);\n"
"    /* Apply modulation */\n"
"    for (int i = tid; i < dim; i += blockDim.x)\n"
"        out[tok*dim+i] = ((x[tok*dim+i] - mean) * inv) * (1.0f + scale[i]) + shift[i];\n"
"}\n"

/* Gated residual: x += gate * proj */
"__global__ void gated_add_f32(float *__restrict__ x,\n"
"    const float *__restrict__ proj, const float *__restrict__ gate,\n"
"    int N, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= N * dim) return;\n"
"    int col = i % dim;\n"
"    x[i] += gate[col] * proj[i];\n"
"}\n"

/* Patchify: latent [C, H, W] → tokens [H/ps*W/ps, C*ps*ps] */
"__global__ void patchify_f32(float *__restrict__ out,\n"
"    const float *__restrict__ latent, int C, int H, int W, int ps) {\n"
"    int tok = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int hp = H / ps, wp = W / ps;\n"
"    if (tok >= hp * wp) return;\n"
"    int py = tok / wp, px = tok % wp;\n"
"    int tok_dim = C * ps * ps;\n"
"    int idx = 0;\n"
"    for (int c = 0; c < C; c++)\n"
"        for (int dy = 0; dy < ps; dy++)\n"
"            for (int dx = 0; dx < ps; dx++)\n"
"                out[tok * tok_dim + idx++] = latent[c * H * W + (py*ps+dy) * W + (px*ps+dx)];\n"
"}\n"

/* Unpatchify: tokens [seq, C*ps*ps] → latent [C, H, W] */
"__global__ void unpatchify_f32(float *__restrict__ out,\n"
"    const float *__restrict__ tokens, int C, int H, int W, int ps) {\n"
"    int tok = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int hp = H / ps, wp = W / ps;\n"
"    if (tok >= hp * wp) return;\n"
"    int py = tok / wp, px = tok % wp;\n"
"    int tok_dim = C * ps * ps;\n"
"    int idx = 0;\n"
"    for (int c = 0; c < C; c++)\n"
"        for (int dy = 0; dy < ps; dy++)\n"
"            for (int dx = 0; dx < ps; dx++)\n"
"                out[c * H * W + (py*ps+dy) * W + (px*ps+dx)] = tokens[tok * tok_dim + idx++];\n"
"}\n"

/* Euler step: x = x + dt * v */
"__global__ void euler_step_f32(float *__restrict__ x,\n"
"    const float *__restrict__ v, float dt, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) x[i] += dt * v[i];\n"
"}\n"

/* Conv2D for VAE: replicate or zero padding, handles any spatial size.
 * Grid: (ceil(co*oh*ow / 256)), Block: (256) */
"__global__ void vae_conv2d_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, const float *__restrict__ weight,\n"
"    const float *__restrict__ bias,\n"
"    int ci, int h, int w, int co, int kh, int kw, int pad_replicate) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int oh = h, ow = w; /* same-padding */\n"
"    int total = co * oh * ow;\n"
"    if (idx >= total) return;\n"
"    int oc = idx / (oh * ow);\n"
"    int rem = idx % (oh * ow);\n"
"    int oy = rem / ow, ox = rem % ow;\n"
"    int ph = (kh - 1) / 2, pw = (kw - 1) / 2;\n"
"    float sum = bias ? bias[oc] : 0.0f;\n"
"    for (int ic = 0; ic < ci; ic++) {\n"
"        for (int fy = 0; fy < kh; fy++) {\n"
"            for (int fx = 0; fx < kw; fx++) {\n"
"                int iy = oy + fy - ph, ix = ox + fx - pw;\n"
"                if (pad_replicate) {\n"
"                    if (iy < 0) iy = 0; if (iy >= h) iy = h - 1;\n"
"                    if (ix < 0) ix = 0; if (ix >= w) ix = w - 1;\n"
"                } else {\n"
"                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;\n"
"                }\n"
"                sum += inp[ic * h * w + iy * w + ix] *\n"
"                       weight[((oc * ci + ic) * kh + fy) * kw + fx];\n"
"            }\n"
"        }\n"
"    }\n"
"    out[idx] = sum;\n"
"}\n"

/* GroupNorm for VAE: scale-only (no bias), 32 groups
 * Grid: (groups), Block: (min(spatial, 256)) */
"__global__ void vae_groupnorm_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, const float *__restrict__ gamma,\n"
"    int C, int spatial, int groups) {\n"
"    int g = blockIdx.x;\n"
"    if (g >= groups) return;\n"
"    int cpg = C / groups;\n"
"    int n = cpg * spatial;\n"
"    /* Two-pass: compute mean then variance */\n"
"    float mean = 0;\n"
"    for (int gc = 0; gc < cpg; gc++) {\n"
"        int ch = g * cpg + gc;\n"
"        for (int s = threadIdx.x; s < spatial; s += blockDim.x)\n"
"            mean += inp[ch * spatial + s];\n"
"    }\n"
"    for (int m = 16; m > 0; m >>= 1)\n"
"        mean += __shfl_xor_sync(0xFFFFFFFF, mean, m);\n"
"    mean /= (float)n;\n"
"    float var = 0;\n"
"    for (int gc = 0; gc < cpg; gc++) {\n"
"        int ch = g * cpg + gc;\n"
"        for (int s = threadIdx.x; s < spatial; s += blockDim.x) {\n"
"            float d = inp[ch * spatial + s] - mean;\n"
"            var += d * d;\n"
"        }\n"
"    }\n"
"    for (int m = 16; m > 0; m >>= 1)\n"
"        var += __shfl_xor_sync(0xFFFFFFFF, var, m);\n"
"    float inv = rsqrtf(var / (float)n + 1e-6f);\n"
"    for (int gc = 0; gc < cpg; gc++) {\n"
"        int ch = g * cpg + gc;\n"
"        float gv = gamma ? gamma[ch] : 1.0f;\n"
"        for (int s = threadIdx.x; s < spatial; s += blockDim.x)\n"
"            out[ch * spatial + s] = (inp[ch * spatial + s] - mean) * inv * gv;\n"
"    }\n"
"}\n"

/* gemm_fp8w_f32: FP8 weights dequanted via LUT in registers, F32 inputs+accumulation.
 * W is raw FP8 bytes [n_out, n_in], X is F32 [n_tok, n_in].
 * Uses the constant memory LUT d_fp8_to_f16_lut but converts to F32.
 * Grid: (ceil(n_out/64), ceil(n_tok/16)), Block: (16, 16) */
"__device__ __constant__ float d_fp8_to_f32_lut[256];\n"
"\n"
"__global__ void gemm_fp8w_f32(float *Y, const unsigned char *W, const float *X,\n"
"    const float *bias, int n_out, int n_in, int n_tok) {\n"
"    __shared__ float smA[16][16];\n"
"    __shared__ float smB[16][16];\n"
"    int tx = threadIdx.x, ty = threadIdx.y;\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int out_base = blockIdx.x * 64;\n"
"    int row = tok_base + ty;\n"
"    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        smA[ty][tx] = (tok_base+ty < n_tok && k+tx < n_in) ? X[(tok_base+ty)*n_in + k+tx] : 0.f;\n"
"        __syncthreads();\n"
"        { int w = out_base + tx;\n"
"          smB[ty][tx] = (w < n_out && k+ty < n_in) ? d_fp8_to_f32_lut[W[(size_t)w*n_in+k+ty]] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc0 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w = out_base+16+tx;\n"
"          smB[ty][tx] = (w < n_out && k+ty < n_in) ? d_fp8_to_f32_lut[W[(size_t)w*n_in+k+ty]] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc1 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w = out_base+32+tx;\n"
"          smB[ty][tx] = (w < n_out && k+ty < n_in) ? d_fp8_to_f32_lut[W[(size_t)w*n_in+k+ty]] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc2 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w = out_base+48+tx;\n"
"          smB[ty][tx] = (w < n_out && k+ty < n_in) ? d_fp8_to_f32_lut[W[(size_t)w*n_in+k+ty]] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc3 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < n_tok) {\n"
"        if (out_base+   tx < n_out) Y[row*n_out+out_base+   tx] = acc0 + (bias?bias[out_base+   tx]:0.f);\n"
"        if (out_base+16+tx < n_out) Y[row*n_out+out_base+16+tx] = acc1 + (bias?bias[out_base+16+tx]:0.f);\n"
"        if (out_base+32+tx < n_out) Y[row*n_out+out_base+32+tx] = acc2 + (bias?bias[out_base+32+tx]:0.f);\n"
"        if (out_base+48+tx < n_out) Y[row*n_out+out_base+48+tx] = acc3 + (bias?bias[out_base+48+tx]:0.f);\n"
"    }\n"
"}\n"
"\n"

/* gemm_f32_f32: pure F32 tiled GEMM (from HY3D kernels) */
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
"        smA[ty][tx] = (tok_base+ty < n_tok && k+tx < n_in) ? X[(tok_base+ty)*n_in + k+tx] : 0.f;\n"
"        __syncthreads();\n"
"        { int w = out_base + tx; smB[ty][tx] = (w < n_out && k+ty < n_in) ? W[(size_t)w*n_in+k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc0 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w = out_base+16+tx; smB[ty][tx] = (w < n_out && k+ty < n_in) ? W[(size_t)w*n_in+k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc1 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w = out_base+32+tx; smB[ty][tx] = (w < n_out && k+ty < n_in) ? W[(size_t)w*n_in+k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc2 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w = out_base+48+tx; smB[ty][tx] = (w < n_out && k+ty < n_in) ? W[(size_t)w*n_in+k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc3 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < n_tok) {\n"
"        if (out_base+   tx < n_out) Y[row*n_out+out_base+   tx] = acc0 + (bias?bias[out_base+   tx]:0.f);\n"
"        if (out_base+16+tx < n_out) Y[row*n_out+out_base+16+tx] = acc1 + (bias?bias[out_base+16+tx]:0.f);\n"
"        if (out_base+32+tx < n_out) Y[row*n_out+out_base+32+tx] = acc2 + (bias?bias[out_base+32+tx]:0.f);\n"
"        if (out_base+48+tx < n_out) Y[row*n_out+out_base+48+tx] = acc3 + (bias?bias[out_base+48+tx]:0.f);\n"
"    }\n"
"}\n"

/* SiLU for VAE: x = x / (1 + exp(-x)) */
"__global__ void vae_silu_f32(float *__restrict__ x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));\n"
"}\n"

/* Nearest-neighbor 2× upsample: [C, H, W] → [C, 2H, 2W] */
"__global__ void nn_upsample2x_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, int C, int H, int W) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int oH = H * 2, oW = W * 2;\n"
"    int total = C * oH * oW;\n"
"    if (idx >= total) return;\n"
"    int c = idx / (oH * oW);\n"
"    int rem = idx % (oH * oW);\n"
"    int oy = rem / oW, ox = rem % oW;\n"
"    int iy = oy / 2, ix = ox / 2;\n"
"    out[idx] = inp[c * H * W + iy * W + ix];\n"
"}\n"

/* Truncate F32 to BF16 precision in-place (simulate BF16 compute).
 * This matches the training precision (BF16) by rounding F32 intermediates
 * to 7 mantissa bits after each operation. Grid: ceil(n/256), Block: 256 */
"__global__ void truncate_bf16_f32(float *__restrict__ x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    unsigned int bits;\n"
"    memcpy(&bits, &x[i], 4);\n"
"    bits &= 0xFFFF0000u;  /* zero lower 16 bits (truncate to BF16) */\n"
"    memcpy(&x[i], &bits, 4);\n"
"}\n"

/* RMSNorm with weight: x[N, dim] *= rsqrt(mean(x^2)) * w[dim]
 * Grid: (N), Block: (256) */
"__global__ void rmsnorm_weighted_f32(float *__restrict__ x,\n"
"    const float *__restrict__ w, int N, int dim) {\n"
"    int tok = blockIdx.x;\n"
"    if (tok >= N) return;\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    float s = 0;\n"
"    for (int i = tid; i < dim; i += blockDim.x) {\n"
"        float v = x[tok * dim + i];\n"
"        s += v * v;\n"
"    }\n"
"    sdata[tid] = s; __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)dim + 1e-6f);\n"
"    for (int i = tid; i < dim; i += blockDim.x)\n"
"        x[tok * dim + i] *= inv * w[i];\n"
"}\n"

/* 2D RoPE for image tokens: apply height+width rotary embeddings
 * q,k: [n_tok, n_heads*head_dim], axes: t_dim, h_dim, w_dim
 * Grid: (n_tok), Block: (n_heads) */
"__global__ void rope_2d_f32(float *__restrict__ q, float *__restrict__ k,\n"
"    int n_tok, int n_heads, int head_dim, int w_patches,\n"
"    int t_dim, int h_dim, int w_dim, float theta) {\n"
"    int tok = blockIdx.x;\n"
"    int head = threadIdx.x;\n"
"    if (tok >= n_tok || head >= n_heads) return;\n"
"    int dim = n_heads * head_dim;\n"
"    int ph = tok / w_patches, pw = tok % w_patches;\n"
"    int off = head * head_dim;\n"
"    /* Height RoPE at offset t_dim */\n"
"    for (int i = 0; i < h_dim / 2; i++) {\n"
"        float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)h_dim);\n"
"        float angle = (float)ph * freq;\n"
"        float cs = cosf(angle), sn = sinf(angle);\n"
"        int idx = off + t_dim + 2 * i;\n"
"        float q0 = q[tok*dim+idx], q1 = q[tok*dim+idx+1];\n"
"        q[tok*dim+idx]   = q0*cs - q1*sn;\n"
"        q[tok*dim+idx+1] = q0*sn + q1*cs;\n"
"        float k0 = k[tok*dim+idx], k1 = k[tok*dim+idx+1];\n"
"        k[tok*dim+idx]   = k0*cs - k1*sn;\n"
"        k[tok*dim+idx+1] = k0*sn + k1*cs;\n"
"    }\n"
"    /* Width RoPE at offset t_dim + h_dim */\n"
"    for (int i = 0; i < w_dim / 2; i++) {\n"
"        float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)w_dim);\n"
"        float angle = (float)pw * freq;\n"
"        float cs = cosf(angle), sn = sinf(angle);\n"
"        int idx = off + t_dim + h_dim + 2 * i;\n"
"        float q0 = q[tok*dim+idx], q1 = q[tok*dim+idx+1];\n"
"        q[tok*dim+idx]   = q0*cs - q1*sn;\n"
"        q[tok*dim+idx+1] = q0*sn + q1*cs;\n"
"        float k0 = k[tok*dim+idx], k1 = k[tok*dim+idx+1];\n"
"        k[tok*dim+idx]   = k0*cs - k1*sn;\n"
"        k[tok*dim+idx+1] = k0*sn + k1*cs;\n"
"    }\n"
"}\n"

/* 1D RoPE for text tokens
 * Grid: (n_tok), Block: (n_heads) */
"__global__ void rope_1d_f32(float *__restrict__ q, float *__restrict__ k,\n"
"    int n_tok, int n_heads, int head_dim, float theta) {\n"
"    int tok = blockIdx.x;\n"
"    int head = threadIdx.x;\n"
"    if (tok >= n_tok || head >= n_heads) return;\n"
"    int dim = n_heads * head_dim;\n"
"    int off = head * head_dim;\n"
"    for (int i = 0; i < head_dim / 2; i++) {\n"
"        float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)head_dim);\n"
"        float angle = (float)tok * freq;\n"
"        float cs = cosf(angle), sn = sinf(angle);\n"
"        int idx = off + 2 * i;\n"
"        float q0 = q[tok*dim+idx], q1 = q[tok*dim+idx+1];\n"
"        q[tok*dim+idx]   = q0*cs - q1*sn;\n"
"        q[tok*dim+idx+1] = q0*sn + q1*cs;\n"
"        float k0 = k[tok*dim+idx], k1 = k[tok*dim+idx+1];\n"
"        k[tok*dim+idx]   = k0*cs - k1*sn;\n"
"        k[tok*dim+idx+1] = k0*sn + k1*cs;\n"
"    }\n"
"}\n"

/* Warp-cooperative flash attention for head_dim=128.
 * Q[N,dim], K[N,dim], V[N,dim] → out[N,dim], dim = n_heads * head_dim.
 *
 * Design: one warp (32 threads) per query. Each thread handles 4 elements
 * of head_dim=128 (128/32=4). Dot products via warp shuffle. K/V loaded
 * in tiles of FA2_BKV into shared memory for reuse across warps.
 *
 * Grid: (n_heads, ceil(N / FA2_WARPS))
 * Block: (32 * FA2_WARPS) threads
 * Shared mem: FA2_BKV * 128 * 2 * sizeof(float) for K+V tiles
 *
 * Register budget per thread: q[4] + O[4] + m + l + 8 misc ≈ 18 regs → high occupancy.
 */
"#define FA2_WARPS   4\n"  /* queries per block (warps per block) */
"#define FA2_BKV    16\n"  /* KV tile size */
"#define FA2_HD    128\n"  /* head_dim */
"#define FA2_EPT     4\n"  /* elements per thread (128/32) */
"\n"
"__global__ void flash_attn_f32(float *__restrict__ out,\n"
"    const float *__restrict__ Q, const float *__restrict__ K,\n"
"    const float *__restrict__ V,\n"
"    int N, int n_heads, int head_dim) {\n"
"    int h  = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int dim = n_heads * head_dim;\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane    = threadIdx.x % 32;\n"
"    int qi = blockIdx.y * FA2_WARPS + warp_id;\n"
"    float scale = rsqrtf((float)head_dim);\n"
"\n"
"    /* Load Q for this query into registers (4 elements per thread) */\n"
"    float q_r[FA2_EPT];\n"
"    for (int e = 0; e < FA2_EPT; e++) {\n"
"        int d = lane * FA2_EPT + e;\n"
"        q_r[e] = (qi < N && d < head_dim) ? Q[qi * dim + h * head_dim + d] : 0.0f;\n"
"    }\n"
"\n"
"    /* Online softmax accumulators */\n"
"    float m_i = -1e30f, l_i = 0.0f;\n"
"    float O_r[FA2_EPT];\n"
"    for (int e = 0; e < FA2_EPT; e++) O_r[e] = 0.0f;\n"
"\n"
"    /* Shared memory for K/V tiles: [FA2_BKV][128] × 2 */\n"
"    extern __shared__ float smem[];\n"
"    float *smK = smem;\n"
"    float *smV = smem + FA2_BKV * FA2_HD;\n"
"\n"
"    int n_tiles = (N + FA2_BKV - 1) / FA2_BKV;\n"
"    int n_threads = FA2_WARPS * 32;\n"
"\n"
"    for (int tile = 0; tile < n_tiles; tile++) {\n"
"        int kv_base = tile * FA2_BKV;\n"
"\n"
"        /* Cooperative load: all threads load K/V tile */\n"
"        for (int idx = threadIdx.x; idx < FA2_BKV * FA2_HD; idx += n_threads) {\n"
"            int kj = idx / FA2_HD, d = idx % FA2_HD;\n"
"            int kv_tok = kv_base + kj;\n"
"            smK[idx] = (kv_tok < N) ? K[kv_tok * dim + h * head_dim + d] : 0.0f;\n"
"            smV[idx] = (kv_tok < N) ? V[kv_tok * dim + h * head_dim + d] : 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* Compute attention scores for this tile (FA2_BKV keys) */\n"
"        for (int kj = 0; kj < FA2_BKV; kj++) {\n"
"            /* Dot product Q·K[kj] distributed across warp */\n"
"            float dot = 0.0f;\n"
"            for (int e = 0; e < FA2_EPT; e++)\n"
"                dot += q_r[e] * smK[kj * FA2_HD + lane * FA2_EPT + e];\n"
"            /* Warp shuffle reduction (5 rounds for 32 lanes) */\n"
"            for (int off = 16; off > 0; off >>= 1)\n"
"                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);\n"
"            /* dot is now the full Q·K score in all lanes */\n"
"            float score = (kv_base + kj < N) ? dot * scale : -1e30f;\n"
"\n"
"            /* Online softmax update */\n"
"            float new_max = fmaxf(m_i, score);\n"
"            float alpha = expf(m_i - new_max);\n"
"            float p = expf(score - new_max);\n"
"            l_i = l_i * alpha + p;\n"
"            for (int e = 0; e < FA2_EPT; e++) {\n"
"                O_r[e] = O_r[e] * alpha + p * smV[kj * FA2_HD + lane * FA2_EPT + e];\n"
"            }\n"
"            m_i = new_max;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    /* Write output: O / l */\n"
"    if (qi < N) {\n"
"        float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;\n"
"        for (int e = 0; e < FA2_EPT; e++) {\n"
"            int d = lane * FA2_EPT + e;\n"
"            if (d < head_dim)\n"
"                out[qi * dim + h * head_dim + d] = O_r[e] * inv_l;\n"
"        }\n"
"    }\n"
"}\n"

"}\n"; /* close extern "C" from cuda_kernels_common_src */


/* ---- Per-block GPU weight struct ---- */

typedef struct {
    CUdeviceptr attn_q_w, attn_q_b, attn_k_w, attn_k_b, attn_v_w, attn_v_b;
    CUdeviceptr attn_out_w, attn_out_b;
    CUdeviceptr attn_add_q_w, attn_add_q_b, attn_add_k_w, attn_add_k_b;
    CUdeviceptr attn_add_v_w, attn_add_v_b, attn_add_out_w, attn_add_out_b;
    CUdeviceptr norm_q_w, norm_k_w, norm_added_q_w, norm_added_k_w;
    CUdeviceptr img_mod_w, img_mod_b;
    CUdeviceptr img_mlp_fc1_w, img_mlp_fc1_b, img_mlp_fc2_w, img_mlp_fc2_b;
    CUdeviceptr txt_mod_w, txt_mod_b;
    CUdeviceptr txt_mlp_fc1_w, txt_mlp_fc1_b, txt_mlp_fc2_w, txt_mlp_fc2_b;
} qimg_block_gpu;


/* ---- Runner struct ---- */

struct cuda_qimg_runner {
    CUdevice device;
    CUcontext ctx;
    CUstream stream;
    CUmodule module;
    int sm_version;
    int verbose;

    /* Kernel handles */
    CUfunction gemm_f16_f32;
    CUfunction gemm_fp8_f32;     /* native FP8 GEMM (sm_89+) */
    CUfunction gemm_f32_f32;     /* pure F32 GEMM (highest precision) */
    CUfunction gemm_fp8w_f32;   /* FP8 weight (LUT dequant) × F32 input */
    CUfunction layernorm_f32;
    CUfunction gelu_f32;
    CUfunction silu_f32;
    CUfunction attn_prefill_f32;
    CUfunction add_bias_f32;
    CUfunction rmsnorm_per_head;
    CUfunction adaln_modulate;
    CUfunction dequant_fp8_to_f16;  /* GPU-side FP8→F16 via LUT */
    int use_fp8_gemm;  /* 1 if sm >= 89 and gemm_fp8_f32 available */
    int use_f16_gemm;  /* 1 to use F16 weights + gemm_f16_f32 (better precision) */
    CUfunction truncate_bf16;
    int use_bf16_trunc;  /* 1 to truncate intermediates to BF16 precision */
    CUfunction gated_add;
    CUfunction patchify;
    CUfunction unpatchify;
    CUfunction euler_step;
    CUfunction rope_2d;
    CUfunction rope_1d;
    CUfunction vae_conv2d;
    CUfunction vae_groupnorm;
    CUfunction vae_silu;
    CUfunction nn_upsample2x;
    CUfunction rmsnorm_weighted;

    /* DiT config */
    int dit_dim, dit_n_heads, dit_head_dim, dit_n_blocks;
    int dit_in_ch, dit_txt_dim, dit_mlp_h;

    /* Safetensors context (mmap'd) */
    void *dit_st;
    void *vae_st;

    /* Preloaded blocks on GPU (NULL if not preloaded, loaded on-demand) */
    qimg_block_gpu *gpu_blocks;  /* [dit_n_blocks] array, some may be zero */
    int n_preloaded;             /* how many blocks are resident on GPU */

    /* Persistent GPU: global weights (~50MB) */
    CUdeviceptr d_img_in_w, d_img_in_b;
    CUdeviceptr d_txt_in_w, d_txt_in_b;
    CUdeviceptr d_txt_norm_w;
    CUdeviceptr d_t_fc1_w, d_t_fc1_b;
    CUdeviceptr d_t_fc2_w, d_t_fc2_b;
    CUdeviceptr d_norm_out_w, d_norm_out_b;
    CUdeviceptr d_proj_out_w, d_proj_out_b;
};


/* ---- FP8 E4M3 → F16 LUT (256 entries, computed once) ---- */

static uint16_t qimg_fp8_to_f16_lut[256];
static int qimg_fp8_to_f16_lut_init = 0;

static void qimg_init_fp8_to_f16_lut(void) {
    if (qimg_fp8_to_f16_lut_init) return;
    for (int i = 0; i < 256; i++)
        qimg_fp8_to_f16_lut[i] = cu_f32_to_f16(fp8_e4m3_to_f32((uint8_t)i));
    qimg_fp8_to_f16_lut_init = 1;
}

/* ---- Safetensor FP8→F16 upload helpers ---- */

/* Upload safetensor as F16 weight using LUT for FP8→F16 conversion.
 * Safetensor shape [n_out, n_in] matches gemm_f16_f32 layout. */
static CUdeviceptr qimg_st_upload_f16(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return 0;
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];

    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    uint16_t *f16 = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        qimg_init_fp8_to_f16_lut();
        for (size_t i = 0; i < n; i++)
            f16[i] = qimg_fp8_to_f16_lut[src[i]];
    } else if (strcmp(dtype, "F16") == 0) {
        memcpy(f16, src, n * 2);
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *bf = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            float f; memcpy(&f, &bits, 4);
            f16[i] = cu_f32_to_f16(f);
        }
    } else if (strcmp(dtype, "F32") == 0) {
        const float *f32 = (const float *)src;
        for (size_t i = 0; i < n; i++)
            f16[i] = cu_f32_to_f16(f32[i]);
    } else {
        fprintf(stderr, "qimg: unsupported dtype '%s' for %s\n", dtype, name);
        free(f16);
        return 0;
    }

    CUdeviceptr d;
    cuMemAlloc(&d, n * sizeof(uint16_t));
    cuMemcpyHtoD(d, f16, n * sizeof(uint16_t));
    free(f16);
    return d;
}

/* ---- FP8 E4M3 → F32 LUT (256 entries) ---- */

static float qimg_cuda_fp8_to_f32_lut[256];
static int qimg_cuda_fp8_to_f32_lut_init = 0;

static void qimg_init_fp8_to_f32_lut(void) {
    if (qimg_cuda_fp8_to_f32_lut_init) return;
    for (int i = 0; i < 256; i++)
        qimg_cuda_fp8_to_f32_lut[i] = fp8_e4m3_to_f32((uint8_t)i);
    qimg_cuda_fp8_to_f32_lut_init = 1;
}

/* Checked cuMemAlloc — returns 0 on failure */
static CUdeviceptr checked_cuMemAlloc(size_t nbytes) {
    CUdeviceptr d = 0;
    CUresult err = cuMemAlloc(&d, nbytes);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_qimg: cuMemAlloc(%.1f MB) FAILED (err=%d)\n",
                (float)nbytes / (1 << 20), (int)err);
        return 0;
    }
    return d;
}

/* Upload safetensor as F32 (for biases, norms) */
static CUdeviceptr qimg_st_upload_f32(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return 0;
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];

    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    float *f32 = (float *)malloc(n * sizeof(float));
    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        qimg_init_fp8_to_f32_lut();
        for (size_t i = 0; i < n; i++)
            f32[i] = qimg_cuda_fp8_to_f32_lut[src[i]];
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(f32, src, n * 4);
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *bf = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else if (strcmp(dtype, "F16") == 0) {
        const uint16_t *f16 = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            /* F16→F32 */
            uint32_t bits = f16[i];
            uint32_t s = (bits >> 15) & 1;
            uint32_t e = (bits >> 10) & 0x1F;
            uint32_t m = bits & 0x3FF;
            uint32_t f;
            if (e == 0) f = s << 31;
            else if (e == 31) f = (s << 31) | (0xFF << 23) | (m << 13);
            else f = (s << 31) | ((e + 112) << 23) | (m << 13);
            memcpy(&f32[i], &f, 4);
        }
    }

    CUdeviceptr d = checked_cuMemAlloc(n * sizeof(float));
    if (!d) { free(f32); return 0; }
    cuMemcpyHtoD(d, f32, n * sizeof(float));
    free(f32);
    return d;
}


/* Upload raw FP8 bytes directly to GPU (zero-copy, no conversion) */
static CUdeviceptr qimg_st_upload_fp8_raw(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return 0;
    size_t nbytes = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);
    CUdeviceptr d = checked_cuMemAlloc(nbytes);
    if (!d) return 0;
    cuMemcpyHtoD(d, data, nbytes);
    return d;
}

/* ---- Load/free one DiT block ---- */

/* Upload block weights as raw FP8 (for native FP8 GEMM) or dequant to F16 */
static void qimg_free_block(qimg_block_gpu *b);  /* forward decl */

/* Returns 0 on success, -1 on OOM (partial allocs are freed) */
static int qimg_load_block(cuda_qimg_runner *r, int block_idx, qimg_block_gpu *b) {
    st_context *st = (st_context *)r->dit_st;
    char name[256];
    int ok = 1;

    /* Upload weight: FP8 raw for native GEMM, F16 for MMA, or F32 fallback */
    #define BLK_W(field, suffix) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        if (r->use_fp8_gemm) { \
            b->field = qimg_st_upload_fp8_raw(st, name); \
        } else if (r->use_f16_gemm) { \
            b->field = qimg_st_upload_f16(st, name); \
        } else { \
            b->field = qimg_st_upload_f32(st, name); \
        } \
        if (!b->field) ok = 0; \
    } } while(0)
    #define BLK_F32(field, suffix) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        b->field = qimg_st_upload_f32(st, name); \
        if (!b->field) ok = 0; \
    } } while(0)

    BLK_W(attn_q_w, "attn.to_q.weight"); BLK_F32(attn_q_b, "attn.to_q.bias");
    BLK_W(attn_k_w, "attn.to_k.weight"); BLK_F32(attn_k_b, "attn.to_k.bias");
    BLK_W(attn_v_w, "attn.to_v.weight"); BLK_F32(attn_v_b, "attn.to_v.bias");
    BLK_W(attn_out_w, "attn.to_out.0.weight"); BLK_F32(attn_out_b, "attn.to_out.0.bias");

    BLK_W(attn_add_q_w, "attn.add_q_proj.weight"); BLK_F32(attn_add_q_b, "attn.add_q_proj.bias");
    BLK_W(attn_add_k_w, "attn.add_k_proj.weight"); BLK_F32(attn_add_k_b, "attn.add_k_proj.bias");
    BLK_W(attn_add_v_w, "attn.add_v_proj.weight"); BLK_F32(attn_add_v_b, "attn.add_v_proj.bias");
    BLK_W(attn_add_out_w, "attn.to_add_out.weight"); BLK_F32(attn_add_out_b, "attn.to_add_out.bias");

    BLK_F32(norm_q_w, "attn.norm_q.weight");
    BLK_F32(norm_k_w, "attn.norm_k.weight");
    BLK_F32(norm_added_q_w, "attn.norm_added_q.weight");
    BLK_F32(norm_added_k_w, "attn.norm_added_k.weight");

    BLK_W(img_mod_w, "img_mod.1.weight"); BLK_F32(img_mod_b, "img_mod.1.bias");
    BLK_W(img_mlp_fc1_w, "img_mlp.net.0.proj.weight"); BLK_F32(img_mlp_fc1_b, "img_mlp.net.0.proj.bias");
    BLK_W(img_mlp_fc2_w, "img_mlp.net.2.weight"); BLK_F32(img_mlp_fc2_b, "img_mlp.net.2.bias");

    BLK_W(txt_mod_w, "txt_mod.1.weight"); BLK_F32(txt_mod_b, "txt_mod.1.bias");
    BLK_W(txt_mlp_fc1_w, "txt_mlp.net.0.proj.weight"); BLK_F32(txt_mlp_fc1_b, "txt_mlp.net.0.proj.bias");
    BLK_W(txt_mlp_fc2_w, "txt_mlp.net.2.weight"); BLK_F32(txt_mlp_fc2_b, "txt_mlp.net.2.bias");

    #undef BLK_W
    #undef BLK_F32

    if (!ok) {
        qimg_free_block(b);
        return -1;
    }
    return 0;
}

static void qimg_free_block(qimg_block_gpu *b) {
    CUdeviceptr *ptrs = (CUdeviceptr *)b;
    int n = sizeof(qimg_block_gpu) / sizeof(CUdeviceptr);
    for (int i = 0; i < n; i++) {
        if (ptrs[i]) { cuMemFree(ptrs[i]); ptrs[i] = 0; }
    }
}


/* ---- Op launch helpers ---- */
static void op_bf16_trunc(cuda_qimg_runner *r, CUdeviceptr x, int n);  /* forward decl */

static void op_gemm(cuda_qimg_runner *r, CUdeviceptr Y, CUdeviceptr W,
                    CUdeviceptr X, CUdeviceptr bias,
                    int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 255) / 256);
    unsigned gy = (unsigned)((n_tok + 15) / 16);

    if (r->use_f16_gemm) {
        /* gemm_f16_f32: F16 weights (MMA tensor cores), F32 inputs+accumulation.
         * Grid: (ceil(n_out/256), ceil(n_tok/16)), Block: (128) */
        cuLaunchKernel(r->gemm_f16_f32, gx, gy, 1, 128, 1, 1,
                       16 * 16 * sizeof(float), r->stream, args, NULL);
    } else if (r->use_fp8_gemm) {
        /* gemm_fp8w_f32: FP8 weights dequanted via LUT, F32 inputs.
         * Grid: (ceil(n_out/64), ceil(n_tok/16)), Block: (16, 16) */
        unsigned gx64 = (unsigned)((n_out + 63) / 64);
        cuLaunchKernel(r->gemm_fp8w_f32, gx64, gy, 1, 16, 16, 1,
                       0, r->stream, args, NULL);
    } else {
        /* gemm_f32_f32: pure F32 fallback.
         * Grid: (ceil(n_out/64), ceil(n_tok/16)), Block: (16, 16) */
        unsigned gx64 = (unsigned)((n_out + 63) / 64);
        cuLaunchKernel(r->gemm_f32_f32, gx64, gy, 1, 16, 16, 1,
                       0, r->stream, args, NULL);
    }
    /* Truncate GEMM output to BF16 precision (simulate BF16 training compute) */
    op_bf16_trunc(r, Y, n_out * n_tok);
}

/* Truncate buffer to BF16 precision (if enabled) */
static void op_bf16_trunc(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    if (!r->use_bf16_trunc) return;
    void *args[] = {&x, &n};
    cuLaunchKernel(r->truncate_bf16, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void op_silu(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->silu_f32, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
    op_bf16_trunc(r, x, n);
}

static void op_gelu(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->gelu_f32, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
    op_bf16_trunc(r, x, n);
}

static void op_adaln(cuda_qimg_runner *r, CUdeviceptr out, CUdeviceptr x,
                     CUdeviceptr shift, CUdeviceptr scale, int N, int dim) {
    void *args[] = {&out, &x, &shift, &scale, &N, &dim};
    cuLaunchKernel(r->adaln_modulate, (unsigned)N, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static void op_rmsnorm_ph(cuda_qimg_runner *r, CUdeviceptr x, CUdeviceptr w,
                          int N, int n_heads, int head_dim) {
    void *args[] = {&x, &w, &N, &n_heads, &head_dim};
    cuLaunchKernel(r->rmsnorm_per_head, (unsigned)N, (unsigned)n_heads, 1,
                   32, 1, 1, 0, r->stream, args, NULL);
}

static void op_gated_add(cuda_qimg_runner *r, CUdeviceptr x, CUdeviceptr proj,
                         CUdeviceptr gate, int N, int dim) {
    int total = N * dim;
    void *args[] = {&x, &proj, &gate, &N, &dim};
    cuLaunchKernel(r->gated_add, (unsigned)((total+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void op_attn(cuda_qimg_runner *r, CUdeviceptr d_out, CUdeviceptr d_q,
                    CUdeviceptr d_k, CUdeviceptr d_v,
                    int n_tok, int n_heads, int head_dim) {
    /* flash_attn_f32: warp-cooperative, FA2_WARPS queries per block.
     * Each warp handles one query, 32 threads × 4 elems = 128 head_dim.
     * Grid: (n_heads, ceil(n_tok/FA2_WARPS)), Block: (32*FA2_WARPS) */
    int fa2_warps = 4;  /* must match FA2_WARPS in kernel */
    int fa2_bkv = 16;   /* must match FA2_BKV in kernel */
    unsigned gy = (unsigned)((n_tok + fa2_warps - 1) / fa2_warps);
    unsigned n_threads = (unsigned)(32 * fa2_warps);
    /* Shared mem: 2 × BKV × 128 floats for K+V tiles */
    size_t smem = (size_t)2 * fa2_bkv * 128 * sizeof(float);
    void *args[] = {&d_out, &d_q, &d_k, &d_v, &n_tok, &n_heads, &head_dim};
    cuLaunchKernel(r->attn_prefill_f32,
                   (unsigned)n_heads, gy, 1,
                   n_threads, 1, 1,
                   smem, r->stream, args, NULL);
}


/* ---- Init ---- */

cuda_qimg_runner *cuda_qimg_init(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuda_qimg: cuewInit failed\n");
        return NULL;
    }
    CU_CHECK_NULL(cuInit(0));

    CUdevice dev;
    CU_CHECK_NULL(cuDeviceGet(&dev, device_id));

    int sm_major, sm_minor;
    cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    int sm = sm_major * 10 + sm_minor;

    CUcontext ctx;
    CU_CHECK_NULL(cuCtxCreate(&ctx, 0, dev));
    CUstream stream;
    CU_CHECK_NULL(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    if (verbose) {
        char name[256]; cuDeviceGetName(name, sizeof(name), dev);
        size_t mem; cuDeviceTotalMem(&mem, dev);
        fprintf(stderr, "cuda_qimg: %s (sm_%d, %.1f GB)\n", name, sm, (float)mem/(1<<30));
    }

    /* Compile kernels */
    size_t len1 = strlen(cuda_kernels_common_src);
    size_t len2 = strlen(qimg_kernel_src);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, cuda_kernels_common_src, len1);
    memcpy(full_src + len1, qimg_kernel_src, len2);
    full_src[len1 + len2] = '\0';

    CUmodule module;
    int rc = cu_compile_kernels(&module, dev, full_src, "qimg.cu", verbose, "cuda_qimg");
    free(full_src);
    if (rc < 0) return NULL;

    cuda_qimg_runner *r = (cuda_qimg_runner *)calloc(1, sizeof(*r));
    r->device = dev; r->ctx = ctx; r->stream = stream;
    r->module = module; r->sm_version = sm; r->verbose = verbose;

    #define GET(field, name) cuModuleGetFunction(&r->field, module, name)
    GET(gemm_f16_f32, "gemm_f16_f32");
    GET(gemm_f32_f32, "gemm_f32_f32");
    GET(gemm_fp8w_f32, "gemm_fp8w_f32");
    /* Try loading native FP8 GEMM (available on sm_89+) */
    r->use_fp8_gemm = 0;
    if (sm >= 89) {
        CUresult fp8_rc = cuModuleGetFunction(&r->gemm_fp8_f32, module, "gemm_fp8_f32");
        if (fp8_rc == CUDA_SUCCESS && r->gemm_fp8_f32) {
            r->use_fp8_gemm = 1;
            if (verbose)
                fprintf(stderr, "cuda_qimg: native FP8 GEMM available (sm_%d)\n", sm);
        }
    }
    GET(layernorm_f32, "layernorm_f32");
    GET(gelu_f32, "gelu_f32");
    GET(silu_f32, "silu_f32");
    GET(attn_prefill_f32, "flash_attn_f32");
    GET(add_bias_f32, "add_bias_f32");
    GET(rmsnorm_per_head, "rmsnorm_per_head_f32");
    GET(adaln_modulate, "adaln_modulate_f32");
    GET(gated_add, "gated_add_f32");
    GET(patchify, "patchify_f32");
    GET(unpatchify, "unpatchify_f32");
    GET(euler_step, "euler_step_f32");
    GET(rope_2d, "rope_2d_f32");
    GET(rope_1d, "rope_1d_f32");
    GET(vae_conv2d, "vae_conv2d_f32");
    GET(vae_groupnorm, "vae_groupnorm_f32");
    GET(vae_silu, "vae_silu_f32");
    GET(nn_upsample2x, "nn_upsample2x_f32");
    GET(rmsnorm_weighted, "rmsnorm_weighted_f32");
    GET(truncate_bf16, "truncate_bf16_f32");
    GET(dequant_fp8_to_f16, "dequant_fp8_to_f16");
    #undef GET

    /* Upload FP8→F16 LUT to GPU constant memory */
    {
        qimg_init_fp8_to_f16_lut();
        CUdeviceptr d_lut;
        size_t lut_size;
        CUresult lut_rc = cuModuleGetGlobal(&d_lut, &lut_size, module, "d_fp8_to_f16_lut");
        if (lut_rc == CUDA_SUCCESS && lut_size == 256 * sizeof(uint16_t)) {
            cuMemcpyHtoD(d_lut, qimg_fp8_to_f16_lut, 256 * sizeof(uint16_t));
            if (verbose)
                fprintf(stderr, "cuda_qimg: FP8→F16 LUT uploaded to GPU constant memory\n");
        }
    }

    /* Upload FP8→F32 LUT to GPU constant memory (for gemm_fp8w_f32) */
    {
        qimg_init_fp8_to_f32_lut();
        CUdeviceptr d_lut32;
        size_t lut_size32;
        CUresult lut_rc = cuModuleGetGlobal(&d_lut32, &lut_size32, module, "d_fp8_to_f32_lut");
        if (lut_rc == CUDA_SUCCESS && lut_size32 == 256 * sizeof(float)) {
            cuMemcpyHtoD(d_lut32, qimg_cuda_fp8_to_f32_lut, 256 * sizeof(float));
            if (verbose)
                fprintf(stderr, "cuda_qimg: FP8→F32 LUT uploaded\n");
        }
    }

    if (verbose) fprintf(stderr, "cuda_qimg: kernels compiled OK\n");
    return r;
}


/* Upload weight: raw FP8 for native GEMM, or upload FP8 + GPU dequant to F16 */
static CUdeviceptr qimg_upload_weight_auto(cuda_qimg_runner *r,
                                            st_context *st, const char *name) {
    if (r->use_fp8_gemm)
        return qimg_st_upload_fp8_raw(st, name);
    if (r->use_f16_gemm)
        return qimg_st_upload_f16(st, name);
    /* F32 path: dequant FP8→F32 on CPU, upload F32 (highest precision) */
    return qimg_st_upload_f32(st, name);
}

/* ---- Load DiT from FP8 safetensors ---- */

int cuda_qimg_load_dit(cuda_qimg_runner *r, const char *path) {
    fprintf(stderr, "cuda_qimg: loading DiT %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->dit_st = st;

    r->dit_dim = 3072; r->dit_n_heads = 24; r->dit_head_dim = 128;
    r->dit_in_ch = 64; r->dit_txt_dim = 3584; r->dit_mlp_h = 12288;

    /* Count blocks */
    r->dit_n_blocks = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *bp = strstr(nm, "transformer_blocks.");
        if (bp) {
            int blk = atoi(bp + 19);
            if (blk + 1 > r->dit_n_blocks) r->dit_n_blocks = blk + 1;
        }
    }

    /* Upload global weights — FP8 raw if native GEMM, else GPU dequant to F16 */
    #define GW(nm) qimg_upload_weight_auto(r, st, nm)
    /* Helper: upload + GPU dequant if not using native FP8 */
    r->d_img_in_w = GW("img_in.weight");
    r->d_img_in_b = qimg_st_upload_f32(st, "img_in.bias");
    r->d_txt_in_w = GW("txt_in.weight");
    r->d_txt_in_b = qimg_st_upload_f32(st, "txt_in.bias");
    r->d_txt_norm_w = qimg_st_upload_f32(st, "txt_norm.weight");
    r->d_t_fc1_w = GW("time_text_embed.timestep_embedder.linear_1.weight");
    r->d_t_fc1_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_1.bias");
    r->d_t_fc2_w = GW("time_text_embed.timestep_embedder.linear_2.weight");
    r->d_t_fc2_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_2.bias");
    r->d_norm_out_w = GW("norm_out.linear.weight");
    r->d_norm_out_b = qimg_st_upload_f32(st, "norm_out.linear.bias");
    r->d_proj_out_w = GW("proj_out.weight");
    r->d_proj_out_b = qimg_st_upload_f32(st, "proj_out.bias");
    #undef GW

    /* Preload as many blocks as fit in VRAM */
    {
        size_t free_mem = 0, total_mem = 0;
        cuMemGetInfo(&free_mem, &total_mem);
        /* Block size: FP8=1 byte, F32=4 bytes per element */
        /* Block size: FP8=1B, F16=2B, F32=4B per weight element (~324M params/block) */
        size_t block_bytes;
        if (r->use_fp8_gemm) block_bytes = 324 * 1024 * 1024;       /* FP8: 1 byte */
        else if (r->use_f16_gemm) block_bytes = 648ULL * 1024 * 1024; /* F16: 2 bytes */
        else block_bytes = 1296ULL * 1024 * 1024;                     /* F32: 4 bytes */
        size_t workspace = 2ULL * 1024 * 1024 * 1024; /* 2GB reserved for activations + scratch */
        int max_preload = (int)((free_mem - workspace) / block_bytes);
        if (max_preload > r->dit_n_blocks) max_preload = r->dit_n_blocks;
        if (max_preload < 0) max_preload = 0;

        r->gpu_blocks = (qimg_block_gpu *)calloc((size_t)r->dit_n_blocks,
                                                  sizeof(qimg_block_gpu));
        r->n_preloaded = max_preload;
        fprintf(stderr, "cuda_qimg: preloading %d/%d blocks to GPU "
                "(%.1f GB free, %.0f MB/block)\n",
                max_preload, r->dit_n_blocks,
                (float)free_mem / (1<<30), (float)block_bytes / (1<<20));

        for (int i = 0; i < max_preload; i++) {
            if (qimg_load_block(r, i, &r->gpu_blocks[i]) != 0) {
                fprintf(stderr, "cuda_qimg: stopped preloading at block %d (OOM)\n", i);
                r->n_preloaded = i;
                break;
            }
        }
        cuStreamSynchronize(r->stream);

        cuMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr, "cuda_qimg: after preload: %.1f GB free\n",
                (float)free_mem / (1<<30));
    }

    fprintf(stderr, "cuda_qimg: loaded %d blocks, dim=%d\n",
            r->dit_n_blocks, r->dit_dim);
    return 0;
}

int cuda_qimg_load_vae(cuda_qimg_runner *r, const char *path) {
    fprintf(stderr, "cuda_qimg: loading VAE %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->vae_st = st;
    fprintf(stderr, "cuda_qimg: VAE loaded (%d tensors)\n", st->n_tensors);
    return 0;
}

void cuda_qimg_free(cuda_qimg_runner *r) {
    if (!r) return;
    /* Free preloaded blocks */
    if (r->gpu_blocks) {
        for (int i = 0; i < r->n_preloaded; i++)
            qimg_free_block(&r->gpu_blocks[i]);
        free(r->gpu_blocks);
    }
    CUdeviceptr *globals = &r->d_img_in_w;
    for (int i = 0; i < 13; i++) { if (globals[i]) cuMemFree(globals[i]); }
    if (r->dit_st) safetensors_close((st_context *)r->dit_st);
    if (r->vae_st) safetensors_close((st_context *)r->vae_st);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->ctx) cuCtxDestroy(r->ctx);
    free(r);
}


/* ---- DiT single step (full 60-block forward on GPU) ---- */

int cuda_qimg_dit_step(cuda_qimg_runner *r,
                       const float *img_tokens, int n_img,
                       const float *txt_tokens, int n_txt,
                       float timestep, float *out) {
    int dim = r->dit_dim;
    int nh = r->dit_n_heads, hd = r->dit_head_dim;
    int in_ch = r->dit_in_ch, txt_dim = r->dit_txt_dim, mlp_h = r->dit_mlp_h;
    int n_total = n_img + n_txt;
    CUstream s = r->stream;

    /* Check GPU memory before allocations */
    {
        size_t free_mem, total_mem;
        cuMemGetInfo(&free_mem, &total_mem);
        size_t needed = (size_t)(n_img + n_txt + n_total*3 + n_total) * dim * sizeof(float)
                      + (size_t)(n_img > n_txt ? n_img : n_txt) * mlp_h * sizeof(float)
                      + (size_t)n_img * in_ch * sizeof(float) * 2
                      + (size_t)n_txt * txt_dim * sizeof(float)
                      + 6 * dim * sizeof(float) * 3;  /* modulation + t_emb */
        if (free_mem < needed + 50*1024*1024) {
            fprintf(stderr, "cuda_qimg: WARNING: low GPU memory! free=%.1fMB needed=%.1fMB\n",
                    (float)free_mem/(1<<20), (float)needed/(1<<20));
        }
    }

    /* Allocate GPU activation buffers */
    CUdeviceptr d_img, d_txt, d_t_emb;
    cuMemAlloc(&d_img, (size_t)n_img * dim * sizeof(float));
    cuMemAlloc(&d_txt, (size_t)n_txt * dim * sizeof(float));
    cuMemAlloc(&d_t_emb, (size_t)dim * sizeof(float));

    /* Upload inputs */
    CUdeviceptr d_img_in, d_txt_in;
    cuMemAlloc(&d_img_in, (size_t)n_img * in_ch * sizeof(float));
    cuMemcpyHtoD(d_img_in, img_tokens, (size_t)n_img * in_ch * sizeof(float));
    cuMemAlloc(&d_txt_in, (size_t)n_txt * txt_dim * sizeof(float));
    cuMemcpyHtoD(d_txt_in, txt_tokens, (size_t)n_txt * txt_dim * sizeof(float));

    /* 1. Timestep embedding: sinusoidal(256) → SiLU(GEMM) → GEMM */
    float t_sin[256];
    int half = 128;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i / (float)half * logf(10000.0f));
        float angle = timestep * freq;
        t_sin[i]        = cosf(angle);  /* cos first (flip_sin_to_cos=True) */
        t_sin[half + i] = sinf(angle);
    }
    CUdeviceptr d_t_sin;
    cuMemAlloc(&d_t_sin, 256 * sizeof(float));
    cuMemcpyHtoD(d_t_sin, t_sin, 256 * sizeof(float));

    op_gemm(r, d_t_emb, r->d_t_fc1_w, d_t_sin, r->d_t_fc1_b, dim, 256, 1);
    op_silu(r, d_t_emb, dim);
    CUdeviceptr d_t_emb2;
    cuMemAlloc(&d_t_emb2, (size_t)dim * sizeof(float));
    op_gemm(r, d_t_emb2, r->d_t_fc2_w, d_t_emb, r->d_t_fc2_b, dim, dim, 1);
    cuMemFree(d_t_emb); d_t_emb = d_t_emb2;
    cuMemFree(d_t_sin);

    /* 2. Text input: RMSNorm(txt_norm_w) → Linear(txt_in) */
    /* Note: ComfyUI's text encoder already outputs normalized hidden states.
     * The txt_norm in the DiT is applied to RAW encoder output.
     * When using ComfyUI pre-encoded hidden states, skip txt_norm since
     * ComfyUI's CLIP encoder already applies it internally. */
    if (r->d_txt_norm_w) {  /* Apply txt_norm RMSNorm (required by model) */
        void *rn_args[] = {&d_txt_in, &r->d_txt_norm_w, &n_txt, &txt_dim};
        cuLaunchKernel(r->rmsnorm_weighted, (unsigned)n_txt, 1, 1,
                       256, 1, 1, 256 * sizeof(float), s, rn_args, NULL);
    }
    op_gemm(r, d_txt, r->d_txt_in_w, d_txt_in, r->d_txt_in_b, dim, txt_dim, n_txt);
    cuMemFree(d_txt_in);

    /* 3. Image input: GEMM(64→3072) */
    op_gemm(r, d_img, r->d_img_in_w, d_img_in, r->d_img_in_b, dim, in_ch, n_img);
    cuMemFree(d_img_in);

    /* Scratch buffers */
    CUdeviceptr d_scratch1, d_scratch2, d_scratch3;
    size_t max_scratch = (size_t)n_total * dim * sizeof(float);
    cuMemAlloc(&d_scratch1, max_scratch);
    cuMemAlloc(&d_scratch2, max_scratch);
    size_t ffn_scratch = (size_t)(n_img > n_txt ? n_img : n_txt) * mlp_h * sizeof(float);
    cuMemAlloc(&d_scratch3, ffn_scratch);

    /* Debug: dump max value of a GPU buffer */
    #define DUMP_MAX(name, ptr, count) do { if (r->verbose >= 2) { \
        float *_tmp = (float*)malloc((count)*sizeof(float)); \
        cuStreamSynchronize(s); cuMemcpyDtoH(_tmp, ptr, (count)*sizeof(float)); \
        float _mx=0; int _nc=0; \
        for(int _i=0;_i<(count);_i++){if(_tmp[_i]!=_tmp[_i])_nc++;else if(fabsf(_tmp[_i])>_mx)_mx=fabsf(_tmp[_i]);} \
        fprintf(stderr, "    %s: max=%.4f nan=%d\n", name, _mx, _nc); \
        free(_tmp); } } while(0)

    /* Modulation scratch */
    CUdeviceptr d_mod;
    cuMemAlloc(&d_mod, (size_t)6 * dim * sizeof(float));

    /* Joint Q/K/V buffers */
    CUdeviceptr d_q, d_k, d_v, d_attn_out;
    cuMemAlloc(&d_q, (size_t)n_total * dim * sizeof(float));
    cuMemAlloc(&d_k, (size_t)n_total * dim * sizeof(float));
    cuMemAlloc(&d_v, (size_t)n_total * dim * sizeof(float));
    cuMemAlloc(&d_attn_out, (size_t)n_total * dim * sizeof(float));

    /* Compute image patch grid for RoPE */
    int hp_rope = (int)sqrtf((float)n_img);
    int wp_rope = n_img / hp_rope;
    float rope_theta = 10000.0f;
    int t_dim_rope = 16, h_dim_rope = 56, w_dim_rope = 56;

    /* 4. Process all blocks */
    for (int L = 0; L < r->dit_n_blocks; L++) {
        if (r->verbose && (L % 10 == 0 || L == r->dit_n_blocks - 1))
            fprintf(stderr, "\r  cuda_qimg: block %d/%d", L + 1, r->dit_n_blocks);

        /* Use preloaded block if available, otherwise load on-demand */
        qimg_block_gpu blk;
        int need_free = 0;
        if (L < r->n_preloaded && r->gpu_blocks[L].attn_q_w) {
            blk = r->gpu_blocks[L];
        } else {
            memset(&blk, 0, sizeof(blk));
            qimg_load_block(r, L, &blk);
            need_free = 1;
        }

        /* -- Image modulation: SiLU(t_emb) → Linear → 6×dim -- */
        CUdeviceptr d_t_silu;
        cuMemAlloc(&d_t_silu, (size_t)dim * sizeof(float));
        cuMemcpyDtoD(d_t_silu, d_t_emb, (size_t)dim * sizeof(float));
        op_silu(r, d_t_silu, dim);

        /* Image mod */
        CUdeviceptr d_img_mod;
        cuMemAlloc(&d_img_mod, (size_t)6 * dim * sizeof(float));
        op_gemm(r, d_img_mod, blk.img_mod_w, d_t_silu, blk.img_mod_b,
                6 * dim, dim, 1);

        /* Text mod */
        CUdeviceptr d_txt_mod;
        cuMemAlloc(&d_txt_mod, (size_t)6 * dim * sizeof(float));
        op_gemm(r, d_txt_mod, blk.txt_mod_w, d_t_silu, blk.txt_mod_b,
                6 * dim, dim, 1);
        cuMemFree(d_t_silu);

        /* -- adaLN + QKV projections -- */
        /* img_shift1 = d_img_mod[0..dim-1], img_scale1 = [dim..2*dim-1], etc. */
        CUdeviceptr img_sh1 = d_img_mod;
        CUdeviceptr img_sc1 = d_img_mod + (size_t)dim * sizeof(float);
        CUdeviceptr img_g1  = d_img_mod + (size_t)2 * dim * sizeof(float);
        CUdeviceptr img_sh2 = d_img_mod + (size_t)3 * dim * sizeof(float);
        CUdeviceptr img_sc2 = d_img_mod + (size_t)4 * dim * sizeof(float);
        CUdeviceptr img_g2  = d_img_mod + (size_t)5 * dim * sizeof(float);

        CUdeviceptr txt_sh1 = d_txt_mod;
        CUdeviceptr txt_sc1 = d_txt_mod + (size_t)dim * sizeof(float);
        CUdeviceptr txt_g1  = d_txt_mod + (size_t)2 * dim * sizeof(float);
        CUdeviceptr txt_sh2 = d_txt_mod + (size_t)3 * dim * sizeof(float);
        CUdeviceptr txt_sc2 = d_txt_mod + (size_t)4 * dim * sizeof(float);
        CUdeviceptr txt_g2  = d_txt_mod + (size_t)5 * dim * sizeof(float);

        /* adaLN image → d_scratch1 */
        if (L == 0) {
            DUMP_MAX("img_input", d_img, n_img*dim);
            DUMP_MAX("img_mod_shift1", img_sh1, dim);
            DUMP_MAX("img_mod_scale1", img_sc1, dim);
            DUMP_MAX("img_mod_gate1", img_g1, dim);
        }
        op_adaln(r, d_scratch1, d_img, img_sh1, img_sc1, n_img, dim);
        if (L == 0) { DUMP_MAX("img_adaln", d_scratch1, n_img*dim); }
        /* adaLN text → d_scratch2 */
        op_adaln(r, d_scratch2, d_txt, txt_sh1, txt_sc1, n_txt, dim);

        /* Image QKV → offset into joint buffers at [n_txt:] */
        CUdeviceptr d_img_q = d_q + (size_t)n_txt * dim * sizeof(float);
        CUdeviceptr d_img_k = d_k + (size_t)n_txt * dim * sizeof(float);
        CUdeviceptr d_img_v = d_v + (size_t)n_txt * dim * sizeof(float);
        op_gemm(r, d_img_q, blk.attn_q_w, d_scratch1, blk.attn_q_b, dim, dim, n_img);
        op_gemm(r, d_img_k, blk.attn_k_w, d_scratch1, blk.attn_k_b, dim, dim, n_img);
        op_gemm(r, d_img_v, blk.attn_v_w, d_scratch1, blk.attn_v_b, dim, dim, n_img);

        /* Text QKV → offset at [0:n_txt] */
        CUdeviceptr d_txt_q = d_q;
        CUdeviceptr d_txt_k = d_k;
        CUdeviceptr d_txt_v = d_v;
        op_gemm(r, d_txt_q, blk.attn_add_q_w, d_scratch2, blk.attn_add_q_b, dim, dim, n_txt);
        op_gemm(r, d_txt_k, blk.attn_add_k_w, d_scratch2, blk.attn_add_k_b, dim, dim, n_txt);
        op_gemm(r, d_txt_v, blk.attn_add_v_w, d_scratch2, blk.attn_add_v_b, dim, dim, n_txt);

        /* QK RMSNorm (per-head) */
        op_rmsnorm_ph(r, d_img_q, blk.norm_q_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_img_k, blk.norm_k_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_txt_q, blk.norm_added_q_w, n_txt, nh, hd);
        op_rmsnorm_ph(r, d_txt_k, blk.norm_added_k_w, n_txt, nh, hd);

        /* RoPE: 2D for image tokens, 1D for text tokens */
        {
            void *rope2d_args[] = {&d_img_q, &d_img_k,
                                   &n_img, &nh, &hd, &wp_rope,
                                   &t_dim_rope, &h_dim_rope, &w_dim_rope, &rope_theta};
            cuLaunchKernel(r->rope_2d, (unsigned)n_img, 1, 1,
                           (unsigned)nh, 1, 1, 0, s, rope2d_args, NULL);
            void *rope1d_args[] = {&d_txt_q, &d_txt_k,
                                   &n_txt, &nh, &hd, &rope_theta};
            cuLaunchKernel(r->rope_1d, (unsigned)n_txt, 1, 1,
                           (unsigned)nh, 1, 1, 0, s, rope1d_args, NULL);
        }

        /* Joint attention: Q/K/V already concatenated as [txt, img] */
        op_attn(r, d_attn_out, d_q, d_k, d_v, n_total, nh, hd);

        /* Output projections */
        CUdeviceptr d_img_attn = d_attn_out + (size_t)n_txt * dim * sizeof(float);
        CUdeviceptr d_txt_attn = d_attn_out;
        op_gemm(r, d_scratch1, blk.attn_out_w, d_img_attn, blk.attn_out_b, dim, dim, n_img);
        op_gemm(r, d_scratch2, blk.attn_add_out_w, d_txt_attn, blk.attn_add_out_b, dim, dim, n_txt);

        /* Gated residual */
        if (L == 0) { DUMP_MAX("img_attn_proj", d_scratch1, n_img*dim); }
        op_gated_add(r, d_img, d_scratch1, img_g1, n_img, dim);
        op_gated_add(r, d_txt, d_scratch2, txt_g1, n_txt, dim);
        if (L == 0) { DUMP_MAX("img_after_attn", d_img, n_img*dim); }

        /* -- MLP with adaLN -- */
        /* Image MLP */
        op_adaln(r, d_scratch1, d_img, img_sh2, img_sc2, n_img, dim);
        op_gemm(r, d_scratch3, blk.img_mlp_fc1_w, d_scratch1, blk.img_mlp_fc1_b,
                mlp_h, dim, n_img);
        op_gelu(r, d_scratch3, n_img * mlp_h);
        op_gemm(r, d_scratch1, blk.img_mlp_fc2_w, d_scratch3, blk.img_mlp_fc2_b,
                dim, mlp_h, n_img);
        op_gated_add(r, d_img, d_scratch1, img_g2, n_img, dim);
        if (L == 0) { DUMP_MAX("img_after_mlp", d_img, n_img*dim); }

        /* Text MLP */
        op_adaln(r, d_scratch2, d_txt, txt_sh2, txt_sc2, n_txt, dim);
        op_gemm(r, d_scratch3, blk.txt_mlp_fc1_w, d_scratch2, blk.txt_mlp_fc1_b,
                mlp_h, dim, n_txt);
        op_gelu(r, d_scratch3, n_txt * mlp_h);
        op_gemm(r, d_scratch2, blk.txt_mlp_fc2_w, d_scratch3, blk.txt_mlp_fc2_b,
                dim, mlp_h, n_txt);
        op_gated_add(r, d_txt, d_scratch2, txt_g2, n_txt, dim);

        /* Free modulation (always per-step), block weights only if loaded on-demand */
        cuMemFree(d_img_mod); cuMemFree(d_txt_mod);
        if (need_free) {
            CUdeviceptr *ptrs = (CUdeviceptr *)&blk;
            int np = sizeof(qimg_block_gpu) / sizeof(CUdeviceptr);
            for (int pi = 0; pi < np; pi++)
                if (ptrs[pi]) cuMemFree(ptrs[pi]);
        }

        cuStreamSynchronize(s);
    }
    if (r->verbose) fprintf(stderr, "\n");

    /* 5. Final output: adaLN → proj_out */
    {
        CUdeviceptr d_t_silu;
        cuMemAlloc(&d_t_silu, (size_t)dim * sizeof(float));
        cuMemcpyDtoD(d_t_silu, d_t_emb, (size_t)dim * sizeof(float));
        op_silu(r, d_t_silu, dim);
        CUdeviceptr d_final_mod;
        cuMemAlloc(&d_final_mod, (size_t)2 * dim * sizeof(float));
        op_gemm(r, d_final_mod, r->d_norm_out_w, d_t_silu, r->d_norm_out_b,
                2 * dim, dim, 1);
        cuMemFree(d_t_silu);

        /* LastLayer: scale, shift = chunk(emb, 2) — scale is FIRST half */
        CUdeviceptr f_scale = d_final_mod;
        CUdeviceptr f_shift = d_final_mod + (size_t)dim * sizeof(float);
        op_adaln(r, d_scratch1, d_img, f_shift, f_scale, n_img, dim);
        cuMemFree(d_final_mod);

        /* proj_out: [n_img, dim] → [n_img, in_ch] */
        CUdeviceptr d_out;
        cuMemAlloc(&d_out, (size_t)n_img * in_ch * sizeof(float));
        op_gemm(r, d_out, r->d_proj_out_w, d_scratch1, r->d_proj_out_b,
                in_ch, dim, n_img);

        /* Download result (sync stream first to ensure computation is complete) */
        cuStreamSynchronize(s);
        cuMemcpyDtoH(out, d_out, (size_t)n_img * in_ch * sizeof(float));
        cuMemFree(d_out);
    }

    /* Cleanup */
    cuMemFree(d_img); cuMemFree(d_txt); cuMemFree(d_t_emb);
    cuMemFree(d_scratch1); cuMemFree(d_scratch2); cuMemFree(d_scratch3);
    cuMemFree(d_mod);
    cuMemFree(d_q); cuMemFree(d_k); cuMemFree(d_v); cuMemFree(d_attn_out);

    return 0;
}

/* ---- CUDA VAE decode ---- */

/* Helper: upload BF16 safetensor as F32 on GPU */
static CUdeviceptr vae_upload_f32(st_context *st, const char *name, CUstream s) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return 0;
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];
    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    const char *dtype = safetensors_dtype(st, idx);
    float *f32 = (float *)malloc(n * sizeof(float));
    if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *bf = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(f32, src, n * 4);
    }
    CUdeviceptr d; cuMemAlloc(&d, n * sizeof(float));
    cuMemcpyHtoD(d, f32, n * sizeof(float));
    free(f32);
    (void)s;
    return d;
}

/* Helper: 3D conv weight [Co,Ci,kD,kH,kW] → 2D [Co,Ci,kH,kW] by summing temporal */
static CUdeviceptr vae_upload_conv3d(st_context *st, const char *name,
                                      int *out_co, int *out_ci, CUstream s) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return 0;
    const uint64_t *shape = safetensors_shape(st, idx);
    int co = (int)shape[0], ci = (int)shape[1], kd = (int)shape[2];
    int kh = (int)shape[3], kw = (int)shape[4];
    if (out_co) *out_co = co;
    if (out_ci) *out_ci = ci;
    size_t n3d = (size_t)co * ci * kd * kh * kw;
    const uint16_t *bf = (const uint16_t *)safetensors_data(st, idx);
    /* BF16 → F32 + sum temporal dim */
    size_t n2d = (size_t)co * ci * kh * kw;
    float *w2d = (float *)calloc(n2d, sizeof(float));
    for (int o = 0; o < co; o++)
        for (int i = 0; i < ci; i++)
            for (int d = 0; d < kd; d++)
                for (int h = 0; h < kh; h++)
                    for (int w = 0; w < kw; w++) {
                        size_t idx3 = ((((size_t)o*ci+i)*kd+d)*kh+h)*kw+w;
                        uint32_t bits = (uint32_t)bf[idx3] << 16;
                        float f; memcpy(&f, &bits, 4);
                        w2d[(((size_t)o*ci+i)*kh+h)*kw+w] += f;
                    }
    CUdeviceptr dp; cuMemAlloc(&dp, n2d * sizeof(float));
    cuMemcpyHtoD(dp, w2d, n2d * sizeof(float));
    free(w2d);
    (void)s; (void)n3d;
    return dp;
}

/* GPU VAE conv2d launch */
static void vae_op_conv2d(cuda_qimg_runner *r, CUdeviceptr out, CUdeviceptr inp,
                          CUdeviceptr w, CUdeviceptr b,
                          int ci, int h, int w_s, int co, int kh, int kw, int rep_pad) {
    int total = co * h * w_s;
    void *args[] = {&out, &inp, &w, &b, &ci, &h, &w_s, &co, &kh, &kw, &rep_pad};
    cuLaunchKernel(r->vae_conv2d, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

/* GPU VAE groupnorm launch */
static void vae_op_gn(cuda_qimg_runner *r, CUdeviceptr out, CUdeviceptr inp,
                      CUdeviceptr gamma, int C, int spatial) {
    int groups = 32;
    void *args[] = {&out, &inp, &gamma, &C, &spatial, &groups};
    cuLaunchKernel(r->vae_groupnorm, (unsigned)groups, 1, 1,
                   32, 1, 1, 0, r->stream, args, NULL);
}

/* GPU VAE silu launch */
static void vae_op_silu(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->vae_silu, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

/* GPU VAE NN upsample 2x */
static CUdeviceptr vae_op_upsample(cuda_qimg_runner *r, CUdeviceptr inp,
                                    int c, int h, int w) {
    int oh = h*2, ow = w*2;
    CUdeviceptr out; cuMemAlloc(&out, (size_t)c*oh*ow*sizeof(float));
    int total = c*oh*ow;
    void *args[] = {&out, &inp, &c, &h, &w};
    cuLaunchKernel(r->nn_upsample2x, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
    return out;
}

/* GPU VAE ResBlock: GroupNorm→SiLU→Conv→GroupNorm→SiLU→Conv + shortcut */
static CUdeviceptr vae_resblock_gpu(cuda_qimg_runner *r, CUdeviceptr x,
                                     CUdeviceptr n1_g, CUdeviceptr c1_w, CUdeviceptr c1_b,
                                     CUdeviceptr n2_g, CUdeviceptr c2_w, CUdeviceptr c2_b,
                                     CUdeviceptr sc_w, CUdeviceptr sc_b,
                                     int ci, int co, int h, int w) {
    int sp = h * w;
    CUdeviceptr tmp; cuMemAlloc(&tmp, (size_t)ci*sp*sizeof(float));
    vae_op_gn(r, tmp, x, n1_g, ci, sp);
    vae_op_silu(r, tmp, ci*sp);
    CUdeviceptr c1_out; cuMemAlloc(&c1_out, (size_t)co*sp*sizeof(float));
    vae_op_conv2d(r, c1_out, tmp, c1_w, c1_b, ci, h, w, co, 3, 3, 1);
    cuMemFree(tmp);

    tmp = (CUdeviceptr)0; cuMemAlloc(&tmp, (size_t)co*sp*sizeof(float));
    vae_op_gn(r, tmp, c1_out, n2_g, co, sp);
    vae_op_silu(r, tmp, co*sp);
    CUdeviceptr c2_out; cuMemAlloc(&c2_out, (size_t)co*sp*sizeof(float));
    vae_op_conv2d(r, c2_out, tmp, c2_w, c2_b, co, h, w, co, 3, 3, 1);
    cuMemFree(tmp); cuMemFree(c1_out);

    /* Shortcut + residual */
    CUdeviceptr out; cuMemAlloc(&out, (size_t)co*sp*sizeof(float));
    if (sc_w) {
        /* 1x1 conv shortcut (pointwise: treat as conv 1×1) */
        vae_op_conv2d(r, out, x, sc_w, sc_b, ci, h, w, co, 1, 1, 0);
        /* out += c2_out */
        int n = co * sp;
        void *args[] = {&out, &c2_out, &n};
        cuLaunchKernel(r->euler_step, (unsigned)((n+255)/256), 1, 1,
                       256, 1, 1, 0, r->stream, args, NULL);
        /* Hack: euler_step does x += dt*v, set dt=1 → x += v. Actually that kernel is
           euler_step_f32(x, v, dt, n) → x[i] += dt*v[i]. We need add. Let me just
           do a device-side add */
        /* Actually just do: for(i) out[i] += c2_out[i] via a generic kernel.
           euler_step with dt=1.0 works! */
        float one = 1.0f;
        void *add_args[] = {&out, &c2_out, &one, &n};
        cuLaunchKernel(r->euler_step, (unsigned)((n+255)/256), 1, 1,
                       256, 1, 1, 0, r->stream, add_args, NULL);
    } else {
        /* Same channels: just add residual */
        int n = co * sp;
        cuMemcpyDtoD(out, x, (size_t)n * sizeof(float));
        float one = 1.0f;
        void *add_args[] = {&out, &c2_out, &one, &n};
        cuLaunchKernel(r->euler_step, (unsigned)((n+255)/256), 1, 1,
                       256, 1, 1, 0, r->stream, add_args, NULL);
    }
    cuMemFree(c2_out);
    return out;
}

int cuda_qimg_vae_decode(cuda_qimg_runner *r,
                         const float *latent, int lat_h, int lat_w,
                         float *out_rgb) {
    st_context *st = (st_context *)r->vae_st;
    if (!st) { fprintf(stderr, "cuda_qimg: VAE not loaded\n"); return -1; }
    CUstream s = r->stream;

    /* Free preloaded DiT blocks to make room */
    if (r->gpu_blocks) {
        for (int i = 0; i < r->n_preloaded; i++)
            qimg_free_block(&r->gpu_blocks[i]);
        r->n_preloaded = 0;
    }
    cuStreamSynchronize(s);

    int h = lat_h, w = lat_w, c = 16;
    fprintf(stderr, "cuda_qimg_vae: decoding [%d, %d, %d] on GPU\n", c, h, w);

    /* Debug: dump GPU buffer stats */
    #define VAE_DUMP(label, ptr, count) do { \
        cuStreamSynchronize(s); \
        float *_t = (float*)malloc((count)*sizeof(float)); \
        cuMemcpyDtoH(_t, ptr, (count)*sizeof(float)); \
        float _mn=_t[0],_mx=_t[0],_s=0; int _nn=0; \
        for(int _i=0;_i<(count);_i++){ \
            if(_t[_i]!=_t[_i]){_nn++;}else{ \
                if(_t[_i]<_mn)_mn=_t[_i]; if(_t[_i]>_mx)_mx=_t[_i]; _s+=_t[_i];}} \
        fprintf(stderr, "  [vae] %s: min=%.4f max=%.4f mean=%.4f nan=%d/%d\n", \
                label, _mn, _mx, _s/((count)-_nn), _nn, (count)); \
        free(_t); } while(0)

    /* Upload latent */
    CUdeviceptr d_x;
    cuMemAlloc(&d_x, (size_t)c * h * w * sizeof(float));
    cuMemcpyHtoD(d_x, latent, (size_t)c * h * w * sizeof(float));
    if (r->verbose >= 2) VAE_DUMP("latent_input", d_x, c*h*w);

    /* post_quant_conv (conv2): 1×1×1 → effectively pointwise */
    CUdeviceptr d_pqc_w = vae_upload_f32(st, "conv2.weight", s);
    CUdeviceptr d_pqc_b = vae_upload_f32(st, "conv2.bias", s);
    if (d_pqc_w) {
        CUdeviceptr d_tmp; cuMemAlloc(&d_tmp, (size_t)c*h*w*sizeof(float));
        vae_op_conv2d(r, d_tmp, d_x, d_pqc_w, d_pqc_b, c, h, w, c, 1, 1, 0);
        cuMemFree(d_x); d_x = d_tmp;
        cuMemFree(d_pqc_w); cuMemFree(d_pqc_b);
    }
    if (r->verbose >= 2) VAE_DUMP("post_quant", d_x, c*h*w);

    /* decoder.conv1: 16→384, 3×3 (replicate pad) */
    int co_c1, ci_c1;
    CUdeviceptr d_c1_w = vae_upload_conv3d(st, "decoder.conv1.weight", &co_c1, &ci_c1, s);
    CUdeviceptr d_c1_b = vae_upload_f32(st, "decoder.conv1.bias", s);
    c = co_c1;
    {
        CUdeviceptr d_tmp; cuMemAlloc(&d_tmp, (size_t)c*h*w*sizeof(float));
        vae_op_conv2d(r, d_tmp, d_x, d_c1_w, d_c1_b, ci_c1, h, w, c, 3, 3, 1);
        cuMemFree(d_x); d_x = d_tmp;
        cuMemFree(d_c1_w); cuMemFree(d_c1_b);
    }
    fprintf(stderr, "  after conv1: [%d, %d, %d]\n", c, h, w);
    if (r->verbose >= 2) VAE_DUMP("conv1_out", d_x, c*h*w);

    /* Middle: ResBlock → skip attention for now → ResBlock */
    /* Load a resblock's weights given prefix string. Uses snprintf to build names. */
    #define LOAD_RB_NAMED(pfx_str, n1, c1w, c1b, n2, c2w, c2b, scw, scb) \
        CUdeviceptr n1, c1w, c1b, n2, c2w, c2b, scw = 0, scb = 0; \
        { char _nm[256]; \
          snprintf(_nm, sizeof(_nm), "%s.residual.0.gamma", pfx_str); n1 = vae_upload_f32(st, _nm, s); \
          snprintf(_nm, sizeof(_nm), "%s.residual.2.weight", pfx_str); { int _co, _ci; c1w = vae_upload_conv3d(st, _nm, &_co, &_ci, s); } \
          snprintf(_nm, sizeof(_nm), "%s.residual.2.bias", pfx_str); c1b = vae_upload_f32(st, _nm, s); \
          snprintf(_nm, sizeof(_nm), "%s.residual.3.gamma", pfx_str); n2 = vae_upload_f32(st, _nm, s); \
          snprintf(_nm, sizeof(_nm), "%s.residual.6.weight", pfx_str); { int _co2, _ci2; c2w = vae_upload_conv3d(st, _nm, &_co2, &_ci2, s); } \
          snprintf(_nm, sizeof(_nm), "%s.residual.6.bias", pfx_str); c2b = vae_upload_f32(st, _nm, s); \
          snprintf(_nm, sizeof(_nm), "%s.shortcut.weight", pfx_str); \
          if (safetensors_find(st, _nm) >= 0) { scw = vae_upload_f32(st, _nm, s); \
            snprintf(_nm, sizeof(_nm), "%s.shortcut.bias", pfx_str); scb = vae_upload_f32(st, _nm, s); } }

    /* mid.0 */
    { LOAD_RB_NAMED("decoder.middle.0", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
      CUdeviceptr d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b, scw, scb, c, c, h, w);
      cuMemFree(d_x); d_x = d_tmp;
      cuMemFree(n1); cuMemFree(c1w); cuMemFree(c1b); cuMemFree(n2); cuMemFree(c2w); cuMemFree(c2b);
      if (scw) cuMemFree(scw); if (scb) cuMemFree(scb); }

    /* Middle attention: GroupNorm → QKV → spatial self-attention → proj + residual */
    {
        int spatial = h * w;
        CUdeviceptr d_gn_g = vae_upload_f32(st, "decoder.middle.1.norm.gamma", s);
        CUdeviceptr d_qkv_w = vae_upload_f32(st, "decoder.middle.1.to_qkv.weight", s);
        CUdeviceptr d_qkv_b = vae_upload_f32(st, "decoder.middle.1.to_qkv.bias", s);
        CUdeviceptr d_proj_w = vae_upload_f32(st, "decoder.middle.1.proj.weight", s);
        CUdeviceptr d_proj_b = vae_upload_f32(st, "decoder.middle.1.proj.bias", s);

        /* GroupNorm */
        CUdeviceptr d_normed; cuMemAlloc(&d_normed, (size_t)c*spatial*sizeof(float));
        vae_op_gn(r, d_normed, d_x, d_gn_g, c, spatial);
        cuMemFree(d_gn_g);

        /* QKV: 1×1 conv = per-spatial-position linear: [3*C, C] @ [C, S] → [3*C, S]
         * Layout: data is [C, spatial] (CHW). QKV weight is [3*C, C, 1, 1].
         * Treat as GEMM: out[3C, S] = W[3C, C] @ inp[C, S] */
        CUdeviceptr d_qkv; cuMemAlloc(&d_qkv, (size_t)3*c*spatial*sizeof(float));
        /* We need a transposed GEMM since data is [C, S] not [S, C].
         * Simple approach: use conv2d with k=1 */
        vae_op_conv2d(r, d_qkv, d_normed, d_qkv_w, d_qkv_b, c, h, w, 3*c, 1, 1, 0);
        cuMemFree(d_normed); cuMemFree(d_qkv_w); cuMemFree(d_qkv_b);

        /* Attention: Q, K, V are each [C, spatial] in CHW layout.
         * We need to compute attention over spatial positions with C as feature dim.
         * Reshape to [spatial, C] (row per spatial position), then run attention with 1 head.
         * But our data is [C, spatial] (channel-first). We need to transpose. */

        /* For simplicity: download to CPU, run CPU attention, upload result */
        float *h_qkv = (float *)malloc((size_t)3 * c * spatial * sizeof(float));
        cuMemcpyDtoH(h_qkv, d_qkv, (size_t)3 * c * spatial * sizeof(float));
        cuMemFree(d_qkv);

        /* Transpose from [3C, spatial] to [spatial, 3C] for attention */
        float *h_qkv_t = (float *)malloc((size_t)3 * c * spatial * sizeof(float));
        for (int s_pos = 0; s_pos < spatial; s_pos++)
            for (int ch = 0; ch < 3 * c; ch++)
                h_qkv_t[s_pos * 3 * c + ch] = h_qkv[ch * spatial + s_pos];
        free(h_qkv);

        /* Split Q[spatial, C], K[spatial, C], V[spatial, C] */
        float *h_q = h_qkv_t;
        float *h_k = h_qkv_t + (size_t)spatial * c;
        float *h_v = h_qkv_t + (size_t)spatial * 2 * c;

        /* Run attention: 1 head with head_dim=C */
        float *h_attn = (float *)malloc((size_t)spatial * c * sizeof(float));
        float scale_at = 1.0f / sqrtf((float)c);
        for (int i = 0; i < spatial; i++) {
            float mx = -1e30f;
            for (int j = 0; j < spatial; j++) {
                float dot = 0;
                for (int d = 0; d < c; d++) dot += h_q[i*c+d] * h_k[j*c+d];
                dot *= scale_at;
                if (dot > mx) mx = dot;
            }
            float esum = 0;
            memset(h_attn + i*c, 0, (size_t)c * sizeof(float));
            for (int j = 0; j < spatial; j++) {
                float dot = 0;
                for (int d = 0; d < c; d++) dot += h_q[i*c+d] * h_k[j*c+d];
                float w_at = expf(dot * scale_at - mx);
                esum += w_at;
                for (int d = 0; d < c; d++) h_attn[i*c+d] += w_at * h_v[j*c+d];
            }
            float inv = 1.0f / esum;
            for (int d = 0; d < c; d++) h_attn[i*c+d] *= inv;
        }
        free(h_qkv_t);

        /* Output projection: [C, C] @ attn_out[C, spatial] + residual
         * First transpose attn back to [C, spatial] */
        float *h_attn_chw = (float *)malloc((size_t)c * spatial * sizeof(float));
        for (int s_pos = 0; s_pos < spatial; s_pos++)
            for (int ch = 0; ch < c; ch++)
                h_attn_chw[ch * spatial + s_pos] = h_attn[s_pos * c + ch];
        free(h_attn);

        /* Upload, conv 1×1, add residual */
        CUdeviceptr d_attn_chw; cuMemAlloc(&d_attn_chw, (size_t)c*spatial*sizeof(float));
        cuMemcpyHtoD(d_attn_chw, h_attn_chw, (size_t)c*spatial*sizeof(float));
        free(h_attn_chw);

        CUdeviceptr d_proj_out; cuMemAlloc(&d_proj_out, (size_t)c*spatial*sizeof(float));
        vae_op_conv2d(r, d_proj_out, d_attn_chw, d_proj_w, d_proj_b, c, h, w, c, 1, 1, 0);
        cuMemFree(d_attn_chw); cuMemFree(d_proj_w); cuMemFree(d_proj_b);

        /* Residual: d_x += d_proj_out */
        {
            int n = c * spatial;
            float one = 1.0f;
            void *add_args[] = {&d_x, &d_proj_out, &one, &n};
            cuLaunchKernel(r->euler_step, (unsigned)((n+255)/256), 1, 1,
                           256, 1, 1, 0, s, add_args, NULL);
        }
        cuMemFree(d_proj_out);
        cuStreamSynchronize(s);
    }

    /* mid.2 */
    { LOAD_RB_NAMED("decoder.middle.2", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
      CUdeviceptr d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b, scw, scb, c, c, h, w);
      cuMemFree(d_x); d_x = d_tmp;
      cuMemFree(n1); cuMemFree(c1w); cuMemFree(c1b); cuMemFree(n2); cuMemFree(c2w); cuMemFree(c2b);
      if (scw) cuMemFree(scw); if (scb) cuMemFree(scb); }
    fprintf(stderr, "  after middle: [%d, %d, %d]\n", c, h, w);
    if (r->verbose >= 2) VAE_DUMP("middle_out", d_x, c*h*w);

    /* Upsample blocks 0-14 */
    for (int i = 0; i < 15; i++) {
        char pfx[128];
        /* Check if this block has residual weights */
        snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.residual.2.weight", i);
        if (safetensors_find(st, pfx) >= 0) {
            char rb_pfx[128];
            snprintf(rb_pfx, sizeof(rb_pfx), "decoder.upsamples.%d", i);
            LOAD_RB_NAMED(rb_pfx, n1, c1w, c1b, n2, c2w, c2b, scw, scb);
            /* Detect ci/co from conv1 weight shape */
            snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.residual.2.weight", i);
            int _idx = safetensors_find(st, pfx);
            int new_co = (int)safetensors_shape(st, _idx)[0];
            int old_ci = c;
            CUdeviceptr d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b,
                                                  scw, scb, old_ci, new_co, h, w);
            cuMemFree(d_x); d_x = d_tmp;
            c = new_co;
            cuMemFree(n1); cuMemFree(c1w); cuMemFree(c1b);
            cuMemFree(n2); cuMemFree(c2w); cuMemFree(c2b);
            if (scw) cuMemFree(scw); if (scb) cuMemFree(scb);
        }

        /* Check for resample (spatial upsample) */
        snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.resample.1.weight", i);
        if (safetensors_find(st, pfx) >= 0) {
            CUdeviceptr rs_w = vae_upload_f32(st, pfx, s);
            snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.resample.1.bias", i);
            CUdeviceptr rs_b = vae_upload_f32(st, pfx, s);
            int rs_idx = safetensors_find(st, pfx);
            /* NN upsample 2x */
            CUdeviceptr d_up = vae_op_upsample(r, d_x, c, h, w);
            h *= 2; w *= 2;
            /* Conv2d (zero padding) */
            snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.resample.1.weight", i);
            rs_idx = safetensors_find(st, pfx);
            int new_c = (int)safetensors_shape(st, rs_idx)[0];
            CUdeviceptr d_tmp; cuMemAlloc(&d_tmp, (size_t)new_c*h*w*sizeof(float));
            vae_op_conv2d(r, d_tmp, d_up, rs_w, rs_b, c, h, w, new_c, 3, 3, 0);
            cuMemFree(d_up); cuMemFree(d_x);
            cuMemFree(rs_w); cuMemFree(rs_b);
            d_x = d_tmp; c = new_c;
            fprintf(stderr, "  upsample %d: [%d, %d, %d]\n", i, c, h, w);
        }

        /* Dump after each upsample iteration (verbose only) */
        if (r->verbose >= 2) {
            char _lbl[64]; snprintf(_lbl, sizeof(_lbl), "block_%d [%d,%d,%d]", i, c, h, w);
            VAE_DUMP(_lbl, d_x, c*h*w);
        }
    }
    #undef LOAD_RB_NAMED

    /* Head: GroupNorm → SiLU → Conv(96→3) */
    {
        if (r->verbose >= 2) VAE_DUMP("pre_head", d_x, c*h*w);
        CUdeviceptr d_gn = vae_upload_f32(st, "decoder.head.0.gamma", s);
        int spatial = h * w;
        CUdeviceptr d_tmp; cuMemAlloc(&d_tmp, (size_t)c*spatial*sizeof(float));
        vae_op_gn(r, d_tmp, d_x, d_gn, c, spatial);
        if (r->verbose >= 2) VAE_DUMP("head_gn", d_tmp, c*spatial);
        vae_op_silu(r, d_tmp, c * spatial);
        if (r->verbose >= 2) VAE_DUMP("head_silu", d_tmp, c*spatial);
        cuMemFree(d_gn);

        int head_co, head_ci;
        CUdeviceptr d_hw = vae_upload_conv3d(st, "decoder.head.2.weight", &head_co, &head_ci, s);
        CUdeviceptr d_hb = vae_upload_f32(st, "decoder.head.2.bias", s);
        CUdeviceptr d_rgb; cuMemAlloc(&d_rgb, (size_t)3*spatial*sizeof(float));
        vae_op_conv2d(r, d_rgb, d_tmp, d_hw, d_hb, c, h, w, 3, 3, 3, 1);
        if (r->verbose >= 2) VAE_DUMP("head_conv", d_rgb, 3*spatial);
        cuMemFree(d_tmp); cuMemFree(d_x); cuMemFree(d_hw); cuMemFree(d_hb);
        d_x = d_rgb;
        c = 3;
    }

    /* Download result (sync stream first to ensure all GPU ops complete) */
    cuStreamSynchronize(s);
    cuMemcpyDtoH(out_rgb, d_x, (size_t)3 * h * w * sizeof(float));
    cuMemFree(d_x);

    fprintf(stderr, "cuda_qimg_vae: decode complete [%d, %d, %d]\n", c, h, w);
    return 0;
}

#endif /* CUDA_QIMG_RUNNER_IMPLEMENTATION */
#endif /* CUDA_QIMG_RUNNER_H */
