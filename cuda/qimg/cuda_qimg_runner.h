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

/* CFG-batched DiT step. Runs cond and uncond through each block with ONE
 * block-weight load per block (vs two for cuda_qimg_dit_step called twice).
 *
 * Output layout: out_cond and out_uncond are each [n_img, 64] F32, caller
 * pre-allocated. img_tokens is shared (same noise for both).
 *
 * txt_cond has n_txt_cond tokens, txt_uncond has n_txt_uncond tokens — they
 * can differ (qwen-image: typ. 12 vs 6 at generation time). */
int cuda_qimg_dit_step_cfg(cuda_qimg_runner *r,
                           const float *img_tokens, int n_img,
                           const float *txt_cond,   int n_txt_cond,
                           const float *txt_uncond, int n_txt_uncond,
                           float timestep,
                           float *out_cond, float *out_uncond);

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
#include "../cuda_fp8_mma_kernels.h"
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

/* CFG combine: out = v_uncond + cfg_scale * (v_cond - v_uncond) */
"__global__ void cfg_combine_f32(float *__restrict__ out,\n"
"    const float *__restrict__ v_cond, const float *__restrict__ v_uncond,\n"
"    float cfg_scale, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) out[i] = v_uncond[i] + cfg_scale * (v_cond[i] - v_uncond[i]);\n"
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

/* RMS norm for VAE: F.normalize(x, dim=1) * sqrt(C) * gamma
 * = x / ||x||_2_per_spatial * sqrt(C) * gamma[c]
 * Input: [C, spatial] (CHW layout). Normalize along C at each spatial position.
 * Grid: (ceil(spatial/blockDim.x)), Block: (256) — one thread per spatial pos */
"__global__ void vae_rmsnorm_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, const float *__restrict__ gamma,\n"
"    int C, int spatial) {\n"
"    int s = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (s >= spatial) return;\n"
"    /* Compute L2 norm across channels at this spatial position */\n"
"    float sum_sq = 0.0f;\n"
"    for (int c = 0; c < C; c++) {\n"
"        float v = inp[c * spatial + s];\n"
"        sum_sq += v * v;\n"
"    }\n"
"    float inv_norm = rsqrtf(sum_sq + 1e-12f);\n"
"    float scale = sqrtf((float)C);\n"
"    for (int c = 0; c < C; c++) {\n"
"        float gv = gamma ? gamma[c] : 1.0f;\n"
"        out[c * spatial + s] = inp[c * spatial + s] * inv_norm * scale * gv;\n"
"    }\n"
"}\n"

/* gemm_fp8w_f32: FP8 weights dequanted via LUT in registers, F32 inputs+accumulation.
 * W is raw FP8 bytes [n_out, n_in], X is F32 [n_tok, n_in].
 * Uses the constant memory LUT d_fp8_to_f16_lut but converts to F32.
 * Grid: (ceil(n_out/64), ceil(n_tok/16)), Block: (16, 16) */
"__device__ __constant__ float d_fp8_to_f32_lut[256];\n"
"\n"
"/* FP8 LUT GEMM: dequant weights via constant memory LUT, F32 inputs+accum.\n"
" * Grid: (ceil(n_out/64), ceil(n_tok/16)), Block: (16, 16) */\n"
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

/* Round F32 to BF16 precision in-place (round-to-nearest-even).
 * Matches PyTorch's BF16 rounding, NOT simple truncation. */
/* Simulate FP8 E4M3 input quantization: F32 → FP8 → F32 (round-trip).
 * Clamp to [-448, 448], quantize to E4M3, dequant back to F32.
 * This matches ComfyUI's fp8_linear which casts input to FP8 before matmul. */
"__global__ void quantize_fp8_roundtrip_f32(float *__restrict__ x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float v = x[i];\n"
"    /* Clamp to FP8 E4M3 range */\n"
"    if (v > 448.0f) v = 448.0f;\n"
"    if (v < -448.0f) v = -448.0f;\n"
"    /* Quantize: F32 → FP8 E4M3 → F32 */\n"
"    unsigned int bits;\n"
"    memcpy(&bits, &v, 4);\n"
"    unsigned int sign = bits >> 31;\n"
"    int exp = (int)((bits >> 23) & 0xFF) - 127;  /* unbiased F32 exponent */\n"
"    unsigned int mant = bits & 0x7FFFFF;          /* F32 mantissa (23 bits) */\n"
"    /* E4M3: bias=7, max_exp=14 (exp=15 reserved for NaN except mant=7) */\n"
"    int fp8_exp = exp + 7;\n"
"    float result;\n"
"    if (exp < -9) { result = 0.0f; }  /* too small for subnormal */\n"
"    else if (fp8_exp <= 0) {\n"
"        /* Subnormal: round mantissa with implicit bit */\n"
"        unsigned int full_mant = mant | 0x800000;\n"
"        int shift = 1 - fp8_exp + 20;  /* shift to get 3-bit mantissa */\n"
"        if (shift >= 24) { result = 0.0f; }\n"
"        else {\n"
"            unsigned int fp8_mant = (full_mant + (1u << (shift-1))) >> shift;\n"
"            if (fp8_mant > 7) fp8_mant = 7;\n"
"            result = ldexpf((float)fp8_mant / 8.0f, -6);\n"
"            if (sign) result = -result;\n"
"        }\n"
"    } else if (fp8_exp >= 15) {\n"
"        result = sign ? -448.0f : 448.0f;\n"
"    } else {\n"
"        /* Normal: round 23-bit mantissa to 3 bits */\n"
"        unsigned int fp8_mant = (mant + (1u << 19)) >> 20;\n"
"        if (fp8_mant > 7) { fp8_mant = 0; fp8_exp++; }\n"
"        if (fp8_exp >= 15) { result = sign ? -448.0f : 448.0f; }\n"
"        else { result = ldexpf(1.0f + (float)fp8_mant / 8.0f, fp8_exp - 7);\n"
"               if (sign) result = -result; }\n"
"    }\n"
"    x[i] = result;\n"
"}\n"
"\n"
"__global__ void truncate_bf16_f32(float *__restrict__ x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    unsigned int bits;\n"
"    memcpy(&bits, &x[i], 4);\n"
"    /* Round to nearest even: add 0x7FFF + (bit 16) for unbiased rounding */\n"
"    unsigned int rounding = 0x7FFFu + ((bits >> 16) & 1u);\n"
"    bits += rounding;\n"
"    bits &= 0xFFFF0000u;\n"
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
"    int n_tok, int n_heads, int head_dim, int h_patches, int w_patches,\n"
"    int t_dim, int h_dim, int w_dim, float theta) {\n"
"    int tok = blockIdx.x;\n"
"    int head = threadIdx.x;\n"
"    if (tok >= n_tok || head >= n_heads) return;\n"
"    int dim = n_heads * head_dim;\n"
"    int ph = tok / w_patches, pw = tok % w_patches;\n"
"    /* Center positions around 0 (matches ComfyUI: pos - h_len//2) */\n"
"    float ph_f = (float)ph - (float)(h_patches / 2);\n"
"    float pw_f = (float)pw - (float)(w_patches / 2);\n"
"    int off = head * head_dim;\n"
"    /* Height RoPE at offset t_dim */\n"
"    for (int i = 0; i < h_dim / 2; i++) {\n"
"        float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)h_dim);\n"
"        float angle = ph_f * freq;\n"
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
"        float angle = pw_f * freq;\n"
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

/* 3-axis RoPE for text tokens: SAME structure as image RoPE but with
 * text-specific positions. ComfyUI uses txt_pos = txt_start + tok for
 * ALL 3 axes (temporal, height, width), with axes_dim=[16,56,56].
 * Grid: (n_tok), Block: (n_heads) */
"__global__ void rope_1d_f32(float *__restrict__ q, float *__restrict__ k,\n"
"    int n_tok, int n_heads, int head_dim, int txt_start,\n"
"    int t_dim, int h_dim, int w_dim, float theta) {\n"
"    int tok = blockIdx.x;\n"
"    int head = threadIdx.x;\n"
"    if (tok >= n_tok || head >= n_heads) return;\n"
"    int dim = n_heads * head_dim;\n"
"    int off = head * head_dim;\n"
"    float pos = (float)(txt_start + tok);\n"
"    /* Temporal RoPE (first t_dim/2 pairs) — same position for all text tokens */\n"
"    for (int i = 0; i < t_dim / 2; i++) {\n"
"        float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)t_dim);\n"
"        float angle = pos * freq;\n"
"        float cs = cosf(angle), sn = sinf(angle);\n"
"        int idx = off + 2 * i;\n"
"        float q0 = q[tok*dim+idx], q1 = q[tok*dim+idx+1];\n"
"        q[tok*dim+idx]   = q0*cs - q1*sn;\n"
"        q[tok*dim+idx+1] = q0*sn + q1*cs;\n"
"        float k0 = k[tok*dim+idx], k1 = k[tok*dim+idx+1];\n"
"        k[tok*dim+idx]   = k0*cs - k1*sn;\n"
"        k[tok*dim+idx+1] = k0*sn + k1*cs;\n"
"    }\n"
"    /* Height RoPE (next h_dim/2 pairs) */\n"
"    for (int i = 0; i < h_dim / 2; i++) {\n"
"        float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)h_dim);\n"
"        float angle = pos * freq;\n"
"        float cs = cosf(angle), sn = sinf(angle);\n"
"        int idx = off + t_dim + 2 * i;\n"
"        float q0 = q[tok*dim+idx], q1 = q[tok*dim+idx+1];\n"
"        q[tok*dim+idx]   = q0*cs - q1*sn;\n"
"        q[tok*dim+idx+1] = q0*sn + q1*cs;\n"
"        float k0 = k[tok*dim+idx], k1 = k[tok*dim+idx+1];\n"
"        k[tok*dim+idx]   = k0*cs - k1*sn;\n"
"        k[tok*dim+idx+1] = k0*sn + k1*cs;\n"
"    }\n"
"    /* Width RoPE (last w_dim/2 pairs) */\n"
"    for (int i = 0; i < w_dim / 2; i++) {\n"
"        float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)w_dim);\n"
"        float angle = pos * freq;\n"
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
/* BF16 truncation helper for attention scores.
 * Matches ComfyUI's BF16 compute precision for softmax. */
"__device__ __forceinline__ float to_bf16(float f) {\n"
"    unsigned int bits;\n"
"    memcpy(&bits, &f, 4);\n"
"    unsigned int rounding = 0x7FFFu + ((bits >> 16) & 1u);\n"
"    bits += rounding;\n"
"    bits &= 0xFFFF0000u;\n"
"    memcpy(&f, &bits, 4);\n"
"    return f;\n"
"}\n"
"\n"
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

/* Optimized GEMM: FP8 weights with LUT dequant, 128×128 tiles */
"__global__ void gemm_opt_fp8(float *Y, const unsigned char *W, const float *X,\n"
"                              const float *bias, int N, int K, int M) {\n"
"    const int BM = GEMM_OPT_BM, BN = GEMM_OPT_BN, BK = GEMM_OPT_BK;\n"
"    const int TM = GEMM_OPT_TM, TN = GEMM_OPT_TN;\n"
"    __shared__ float smA[BK][BM];\n"
"    __shared__ float smB[BK][BN];\n"
"    int tid = threadIdx.x;\n"
"    int bm = blockIdx.y * BM, bn = blockIdx.x * BN;\n"
"    int tr = tid / 16, tc = tid % 16;\n"
"    float acc[TM][TN];\n"
"    #pragma unroll\n"
"    for (int i = 0; i < TM; i++)\n"
"        #pragma unroll\n"
"        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;\n"
"    #define OFP8_LA(ko) do { \\\n"
"        for (int _i = tid; _i < BM*BK; _i += 256) { \\\n"
"            int _r = _i % BM, _c = _i / BM; \\\n"
"            int gr = bm+_r, gk = (ko)+_c; \\\n"
"            smA[_c][_r] = (gr<M&&gk<K) ? X[gr*K+gk] : 0.0f; \\\n"
"        } } while(0)\n"
"    #define OFP8_LB(ko) do { \\\n"
"        for (int _i = tid; _i < BN*BK; _i += 256) { \\\n"
"            int _r = _i % BN, _c = _i / BN; \\\n"
"            int gn = bn+_r, gk = (ko)+_c; \\\n"
"            smB[_c][_r] = (gn<N&&gk<K) ? d_fp8_to_f32_lut[W[(size_t)gn*K+gk]] : 0.0f; \\\n"
"        } } while(0)\n"
"    for (int k = 0; k < K; k += BK) {\n"
"        OFP8_LA(k); OFP8_LB(k);\n"
"        __syncthreads();\n"
"        #pragma unroll\n"
"        for (int kk = 0; kk < BK; kk++) {\n"
"            float af[TM], bf[TN];\n"
"            #pragma unroll\n"
"            for (int i=0;i<TM;i++) af[i] = smA[kk][tr*TM+i];\n"
"            #pragma unroll\n"
"            for (int j=0;j<TN;j++) bf[j] = smB[kk][tc*TN+j];\n"
"            #pragma unroll\n"
"            for (int i=0;i<TM;i++)\n"
"                #pragma unroll\n"
"                for (int j=0;j<TN;j++) acc[i][j] += af[i]*bf[j];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    #undef OFP8_LA\n"
"    #undef OFP8_LB\n"
"    #pragma unroll\n"
"    for (int i=0;i<TM;i++) {\n"
"        int gr = bm+tr*TM+i; if(gr>=M) continue;\n"
"        #pragma unroll\n"
"        for (int j=0;j<TN;j++) {\n"
"            int gn = bn+tc*TN+j;\n"
"            if(gn<N) {\n"
"                float val = acc[i][j]+(bias?bias[gn]:0.0f);\n"
"                unsigned int bits;\n"
"                asm(\"mov.b32 %0, %1;\" : \"=r\"(bits) : \"f\"(val));\n"
"                bits += 0x7FFFu+((bits>>16)&1u); bits &= 0xFFFF0000u;\n"
"                asm(\"mov.b32 %0, %1;\" : \"=f\"(val) : \"r\"(bits));\n"
"                Y[gr*N+gn] = val;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
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
    int verbose;  /* 0=silent, 1=progress(default), 2=stats, 3=dump .npy */

    /* Kernel handles */
    CUfunction gemm_f16_f32;
    CUfunction gemm_fp8_f32;     /* native FP8 GEMM (sm_89+) */
    CUfunction gemm_f32_f32;     /* pure F32 GEMM (highest precision) */
    CUfunction gemm_fp8w_f32;   /* FP8 weight (LUT dequant) × F32 input */
    CUfunction gemm_opt_f16;    /* optimized 128×128 tiled F16 GEMM */
    CUfunction gemm_opt_fp8;    /* optimized 128×128 tiled FP8 LUT GEMM */
    CUfunction gemm_fp8_mma;    /* mma.sync m16n8k32 FP8 tensor-core GEMM (sm_89+) */
    CUfunction gemm_fp8_mma_bf16; /* same, BF16 X pre-rounding + inline BF16 output */
    CUfunction gemm_fp8_mma_tc128; /* 128×128 tile, 16 warps, 8 MMAs/warp (sm_89+) */
    CUfunction flash_attn_fp8;  /* mma.sync m16n8k32 FP8 flash attention (sm_89+) */
    CUfunction quantize_fp8;    /* F32 → e4m3 with per-tensor scale */
    CUfunction reduce_max_abs;  /* atomic-max reduce to a single F32 slot */
    CUfunction fn_zero_f32;     /* memset(F32 slot, 0) helper */
    CUfunction layernorm_f32;
    CUfunction gelu_f32;
    CUfunction silu_f32;
    CUfunction attn_prefill_f32;
    CUfunction add_bias_f32;
    CUfunction rmsnorm_per_head;
    CUfunction adaln_modulate;
    CUfunction dequant_fp8_to_f16;  /* GPU-side FP8→F16 via LUT */
    int use_fp8_gemm;  /* 1 if sm >= 89 and gemm_fp8_f32 available */
    int use_fp8_mma;   /* 1 to use mma.sync FP8 tensor-core GEMM (env QIMG_FP8_MMA=1) */
    int use_fp8_mma_bf16; /* 1 to prefer BF16-X variant (env QIMG_FP8_MMA_BF16=0 opts out) */
    int use_fp8_mma_tc128; /* 1 to use 128×128-tile FP8 MMA (env QIMG_FP8_TC128=1) */
    int use_fp8_attn;  /* 1 to use mma.sync FP8 flash attention (env QIMG_FP8_ATTN=1) */
    /* Lazy FP8 attention workspace (re-used across blocks; sized to max n_tok*dim) */
    CUdeviceptr d_q_fp8, d_k_fp8, d_v_fp8;  /* [n_tok*dim] uint8 e4m3 */
    CUdeviceptr d_qkv_scales;               /* [3] float (sQ, sK, sV) */
    CUdeviceptr d_qkv_max;                  /* [1] float scratch */
    size_t      fp8_attn_buf_n;             /* current allocation size in bytes */
    int use_f16_gemm;  /* 1 to use F16 weights + gemm_f16_f32 (better precision) */
    CUfunction truncate_bf16;
    CUfunction quantize_fp8_rt;  /* FP8 roundtrip quantization */
    int use_bf16_trunc;  /* 1 to truncate intermediates to BF16 precision */
    int use_old_gemm;    /* 1 to use old 16×16 tiled GEMM (for A/B testing) */
    CUfunction gated_add;
    CUfunction patchify;
    CUfunction unpatchify;
    CUfunction euler_step;
    CUfunction cfg_combine;
    CUfunction rope_2d;
    CUfunction rope_1d;
    CUfunction vae_conv2d;
    CUfunction vae_rmsnorm;
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

    /* Block scratch slot for on-demand loads. Pre-allocated once at init so
     * qimg_load_block doesn't pay the ~30 cuMemAlloc + 30 cuMemFree driver
     * calls per block (measured ~86 ms/load at 512x512). The slot holds one
     * block at a time — blocks are loaded sequentially within a forward
     * and the next kernel reads from the slot. */
    qimg_block_gpu scratch_block;
    int scratch_block_ready;  /* non-zero if scratch_block buffers are allocated */

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

    CUdeviceptr d = 0;
    CUresult err = cuMemAlloc(&d, n * sizeof(uint16_t));
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_qimg: F16 upload cuMemAlloc(%.1f MB) FAILED (err=%d) for %s\n",
                (float)(n * 2) / (1 << 20), (int)err, name);
        free(f16);
        return 0;
    }
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

/* Async HtoD on r->stream — queues after pending compute kernels, stream
 * ordering serializes scratch-slot writes with prior compute so no explicit
 * cuStreamSynchronize is needed between block loads. */
static int qimg_st_upload_fp8_raw_async(st_context *st, const char *name,
                                         CUdeviceptr dst, size_t dst_nbytes,
                                         CUstream s) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return -1;
    size_t nbytes = safetensors_nbytes(st, idx);
    if (nbytes != dst_nbytes) return -1;
    cuMemcpyHtoDAsync(dst, safetensors_data(st, idx), nbytes, s);
    return 0;
}

static int qimg_st_upload_f32_async(st_context *st, const char *name,
                                    CUdeviceptr dst, size_t dst_nelem,
                                    CUstream s) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return -1;
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];
    if (n != dst_nelem) return -1;
    const char *dtype = safetensors_dtype(st, idx);
    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    if (strcmp(dtype, "F32") == 0) {
        cuMemcpyHtoDAsync(dst, src, n * sizeof(float), s);
        return 0;
    }
    /* Conversion path (BF16/F16/FP8) via a temporary host buffer. Rare.
     * Must sync so we can free the host buffer safely. */
    float *f32 = (float *)malloc(n * sizeof(float));
    if (!f32) return -1;
    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        qimg_init_fp8_to_f32_lut();
        for (size_t i = 0; i < n; i++) f32[i] = qimg_cuda_fp8_to_f32_lut[src[i]];
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *bf = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else {
        free(f32); return -1;
    }
    cuMemcpyHtoDAsync(dst, f32, n * sizeof(float), s);
    cuStreamSynchronize(s);
    free(f32);
    return 0;
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

/* Allocate the per-runner scratch block slot once. Reserves one block's
 * worth of device memory so on-demand loads can just H2D-copy into the
 * existing buffers instead of cuMemAlloc/cuMemFree per block. Assumes
 * use_fp8_gemm (FP8 raw bytes for *_w tensors). Returns 0 on success. */
static int qimg_alloc_scratch_block(cuda_qimg_runner *r) {
    if (r->scratch_block_ready) return 0;
    st_context *st = (st_context *)r->dit_st;
    if (!st || r->dit_n_blocks <= 0) return -1;
    int i0 = 0;  /* probe block 0 for tensor sizes */
    char name[256];
    qimg_block_gpu *b = &r->scratch_block;
    memset(b, 0, sizeof(*b));
    int ok = 1;
#define ALLOC_FROM_ST(field, suffix) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, i0); \
        int _ti = safetensors_find(st, name); \
        if (_ti < 0) { ok = 0; break; } \
        size_t _nb = safetensors_nbytes(st, _ti); \
        if (cuMemAlloc(&b->field, _nb) != CUDA_SUCCESS) { ok = 0; break; } \
    } } while(0)
#define ALLOC_F32(field, suffix) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, i0); \
        int _ti = safetensors_find(st, name); \
        if (_ti < 0) { ok = 0; break; } \
        const uint64_t *_sh = safetensors_shape(st, _ti); \
        int _nd = safetensors_ndims(st, _ti); \
        size_t _n = 1; for (int _d=0;_d<_nd;_d++) _n *= _sh[_d]; \
        if (cuMemAlloc(&b->field, _n * sizeof(float)) != CUDA_SUCCESS) { ok = 0; break; } \
    } } while(0)
    ALLOC_FROM_ST(attn_q_w, "attn.to_q.weight");
    ALLOC_F32(attn_q_b, "attn.to_q.bias");
    ALLOC_FROM_ST(attn_k_w, "attn.to_k.weight");
    ALLOC_F32(attn_k_b, "attn.to_k.bias");
    ALLOC_FROM_ST(attn_v_w, "attn.to_v.weight");
    ALLOC_F32(attn_v_b, "attn.to_v.bias");
    ALLOC_FROM_ST(attn_out_w, "attn.to_out.0.weight");
    ALLOC_F32(attn_out_b, "attn.to_out.0.bias");
    ALLOC_FROM_ST(attn_add_q_w, "attn.add_q_proj.weight");
    ALLOC_F32(attn_add_q_b, "attn.add_q_proj.bias");
    ALLOC_FROM_ST(attn_add_k_w, "attn.add_k_proj.weight");
    ALLOC_F32(attn_add_k_b, "attn.add_k_proj.bias");
    ALLOC_FROM_ST(attn_add_v_w, "attn.add_v_proj.weight");
    ALLOC_F32(attn_add_v_b, "attn.add_v_proj.bias");
    ALLOC_FROM_ST(attn_add_out_w, "attn.to_add_out.weight");
    ALLOC_F32(attn_add_out_b, "attn.to_add_out.bias");
    ALLOC_F32(norm_q_w, "attn.norm_q.weight");
    ALLOC_F32(norm_k_w, "attn.norm_k.weight");
    ALLOC_F32(norm_added_q_w, "attn.norm_added_q.weight");
    ALLOC_F32(norm_added_k_w, "attn.norm_added_k.weight");
    ALLOC_FROM_ST(img_mod_w, "img_mod.1.weight");
    ALLOC_F32(img_mod_b, "img_mod.1.bias");
    ALLOC_FROM_ST(img_mlp_fc1_w, "img_mlp.net.0.proj.weight");
    ALLOC_F32(img_mlp_fc1_b, "img_mlp.net.0.proj.bias");
    ALLOC_FROM_ST(img_mlp_fc2_w, "img_mlp.net.2.weight");
    ALLOC_F32(img_mlp_fc2_b, "img_mlp.net.2.bias");
    ALLOC_FROM_ST(txt_mod_w, "txt_mod.1.weight");
    ALLOC_F32(txt_mod_b, "txt_mod.1.bias");
    ALLOC_FROM_ST(txt_mlp_fc1_w, "txt_mlp.net.0.proj.weight");
    ALLOC_F32(txt_mlp_fc1_b, "txt_mlp.net.0.proj.bias");
    ALLOC_FROM_ST(txt_mlp_fc2_w, "txt_mlp.net.2.weight");
    ALLOC_F32(txt_mlp_fc2_b, "txt_mlp.net.2.bias");
#undef ALLOC_FROM_ST
#undef ALLOC_F32
    if (!ok) {
        qimg_free_block(b);
        return -1;
    }
    r->scratch_block_ready = 1;
    return 0;
}

/* Copy block i into the pre-allocated scratch slot. Caller gets a
 * qimg_block_gpu struct whose pointers alias the scratch slot.
 *
 * Must cuStreamSynchronize before reusing the slot — the slot is shared
 * across all on-demand blocks, and the previous block's kernels may still
 * be reading from it when we overwrite with the next block's weights. */
static int qimg_load_block_into_slot(cuda_qimg_runner *r, int block_idx,
                                     qimg_block_gpu *b) {
    if (!r->scratch_block_ready) {
        if (qimg_alloc_scratch_block(r) != 0) return -1;
    }
    st_context *st = (st_context *)r->dit_st;
    CUstream s = r->stream;
    *b = r->scratch_block;  /* copy pointers; stream ordering protects slot reuse */
    char name[256];
    int ok = 1;
#define UPLOAD_FP8(field, suffix, _bytes) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        if (qimg_st_upload_fp8_raw_async(st, name, b->field, _bytes, s) != 0) ok = 0; \
    } } while(0)
#define UPLOAD_F32(field, suffix, _nelem) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        if (qimg_st_upload_f32_async(st, name, b->field, _nelem, s) != 0) ok = 0; \
    } } while(0)
    int dim = r->dit_dim, mlp = r->dit_mlp_h;
    size_t sq = (size_t)dim * dim;               /* FP8 bytes */
    size_t s_mod = (size_t)6 * dim * dim;        /* img_mod, txt_mod: [6*dim, dim] */
    size_t s_fc1 = (size_t)mlp * dim;
    size_t s_fc2 = (size_t)dim * mlp;
    int hd = r->dit_head_dim;
    UPLOAD_FP8(attn_q_w, "attn.to_q.weight", sq);
    UPLOAD_F32(attn_q_b, "attn.to_q.bias", (size_t)dim);
    UPLOAD_FP8(attn_k_w, "attn.to_k.weight", sq);
    UPLOAD_F32(attn_k_b, "attn.to_k.bias", (size_t)dim);
    UPLOAD_FP8(attn_v_w, "attn.to_v.weight", sq);
    UPLOAD_F32(attn_v_b, "attn.to_v.bias", (size_t)dim);
    UPLOAD_FP8(attn_out_w, "attn.to_out.0.weight", sq);
    UPLOAD_F32(attn_out_b, "attn.to_out.0.bias", (size_t)dim);
    UPLOAD_FP8(attn_add_q_w, "attn.add_q_proj.weight", sq);
    UPLOAD_F32(attn_add_q_b, "attn.add_q_proj.bias", (size_t)dim);
    UPLOAD_FP8(attn_add_k_w, "attn.add_k_proj.weight", sq);
    UPLOAD_F32(attn_add_k_b, "attn.add_k_proj.bias", (size_t)dim);
    UPLOAD_FP8(attn_add_v_w, "attn.add_v_proj.weight", sq);
    UPLOAD_F32(attn_add_v_b, "attn.add_v_proj.bias", (size_t)dim);
    UPLOAD_FP8(attn_add_out_w, "attn.to_add_out.weight", sq);
    UPLOAD_F32(attn_add_out_b, "attn.to_add_out.bias", (size_t)dim);
    UPLOAD_F32(norm_q_w, "attn.norm_q.weight", (size_t)hd);
    UPLOAD_F32(norm_k_w, "attn.norm_k.weight", (size_t)hd);
    UPLOAD_F32(norm_added_q_w, "attn.norm_added_q.weight", (size_t)hd);
    UPLOAD_F32(norm_added_k_w, "attn.norm_added_k.weight", (size_t)hd);
    UPLOAD_FP8(img_mod_w, "img_mod.1.weight", s_mod);
    UPLOAD_F32(img_mod_b, "img_mod.1.bias", (size_t)6 * dim);
    UPLOAD_FP8(img_mlp_fc1_w, "img_mlp.net.0.proj.weight", s_fc1);
    UPLOAD_F32(img_mlp_fc1_b, "img_mlp.net.0.proj.bias", (size_t)mlp);
    UPLOAD_FP8(img_mlp_fc2_w, "img_mlp.net.2.weight", s_fc2);
    UPLOAD_F32(img_mlp_fc2_b, "img_mlp.net.2.bias", (size_t)dim);
    UPLOAD_FP8(txt_mod_w, "txt_mod.1.weight", s_mod);
    UPLOAD_F32(txt_mod_b, "txt_mod.1.bias", (size_t)6 * dim);
    UPLOAD_FP8(txt_mlp_fc1_w, "txt_mlp.net.0.proj.weight", s_fc1);
    UPLOAD_F32(txt_mlp_fc1_b, "txt_mlp.net.0.proj.bias", (size_t)mlp);
    UPLOAD_FP8(txt_mlp_fc2_w, "txt_mlp.net.2.weight", s_fc2);
    UPLOAD_F32(txt_mlp_fc2_b, "txt_mlp.net.2.bias", (size_t)dim);
#undef UPLOAD_FP8
#undef UPLOAD_F32
    return ok ? 0 : -1;
}


/* ---- Op launch helpers ---- */
static void op_bf16_trunc(cuda_qimg_runner *r, CUdeviceptr x, int n);  /* forward decl */

static void op_gemm(cuda_qimg_runner *r, CUdeviceptr Y, CUdeviceptr W,
                    CUdeviceptr X, CUdeviceptr bias,
                    int n_out, int n_in, int n_tok) {

    if (r->use_fp8_mma && r->gemm_fp8_mma && n_tok >= 16 && !r->use_old_gemm) {
        /* mma.sync m16n8k32 FP8 tensor-core GEMM with per-tensor weight scale.
         * qwen-image FP8 weights are raw e4m3 (no scale) → w_scale = 1.0f.
         * Grid: (ceil(n_out/256), ceil(n_tok/32)), Block: 128 threads (4 warps).
         * Shared mem: (16*MTILE) * 32 * sizeof(float) = 2048 bytes for X tile.
         * n_tok >= 16 gate: matches the MTILE=2 minimum-row assumption and
         * routes single-token ops (timestep embed, modulation) to the LUT path.
         *
         * Per-tensor input scaling: when X magnitudes exceed ±448 (e.g.
         * GELU output from mlp_fc1 can hit 600+), the in-kernel cvt.rn.
         * satfinite.e4m3 clamps and the output diverges. To match ComfyUI's
         * torch._scaled_mm, we pre-compute max(|X|) via reduce_max_abs_f32
         * and pass a device pointer so the kernel divides X by max/448 at
         * load and multiplies the accumulator by max/448 at writeback. */
        /* Always compute max(|X|) for the MMA GEMM. Initially tried gating
         * this on n_in >= 4096 (only mlp_fc2) but that broke correctness
         * (corr dropped from 0.999 to 0.951), indicating other GEMMs also
         * have inputs with outliers that exceed ±448. */
        CUdeviceptr x_max_ptr = 0;
        if (r->reduce_max_abs) {
            if (!r->d_qkv_max) cuMemAlloc(&r->d_qkv_max, sizeof(float));
            cuMemsetD32Async(r->d_qkv_max, 0, 1, r->stream);
            int n_x = n_tok * n_in;
            void *rargs[] = {&r->d_qkv_max, &X, &n_x};
            unsigned rb = (unsigned)((n_x + 255) / 256);
            cuLaunchKernel(r->reduce_max_abs, rb, 1, 1, 256, 1, 1,
                           0, r->stream, rargs, NULL);
            x_max_ptr = r->d_qkv_max;
        }
        float w_scale = 1.0f;
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &w_scale, &x_max_ptr};
        /* 128×128-tile variant: 16 warps (512 threads), 16 KB smem for X tile.
         * Requires n_tok >= 128 to fill a CTA row; smaller shapes fall through. */
        if (r->use_fp8_mma_tc128 && r->gemm_fp8_mma_tc128 && n_tok >= 128 && n_out >= 128) {
            unsigned gx128 = (unsigned)((n_out + 127) / 128);
            unsigned gy128 = (unsigned)((n_tok + 127) / 128);
            size_t smem128 = (size_t)128 * 32 * sizeof(float);
            cuLaunchKernel(r->gemm_fp8_mma_tc128, gx128, gy128, 1, 512, 1, 1,
                           smem128, r->stream, args, NULL);
            op_bf16_trunc(r, Y, n_out * n_tok);
            return;
        }
        unsigned gx = (unsigned)((n_out + 255) / 256);
        unsigned gy = (unsigned)((n_tok +  31) /  32);
        size_t smem = (size_t)(16 * 2) * 32 * sizeof(float);  /* MTILE=2 */
        /* Prefer the BF16-X sibling when available: pre-rounds X to BF16 before
         * the e4m3 cvt to match ComfyUI's BF16 inference dtype, and truncates
         * the output to BF16 inline so we skip the downstream op_bf16_trunc. */
        if (r->use_fp8_mma_bf16 && r->gemm_fp8_mma_bf16) {
            cuLaunchKernel(r->gemm_fp8_mma_bf16, gx, gy, 1, 128, 1, 1,
                           smem, r->stream, args, NULL);
            return;
        }
        cuLaunchKernel(r->gemm_fp8_mma, gx, gy, 1, 128, 1, 1,
                       smem, r->stream, args, NULL);
        /* Match the downstream BF16 precision of the tiled-FP8 path. */
        op_bf16_trunc(r, Y, n_out * n_tok);
        return;
    }

    if (r->use_f16_gemm && r->gemm_opt_f16 && n_tok >= 16 && !r->use_old_gemm) {
        /* Optimized 128×128 tiled GEMM with fused BF16 truncation.
         * Grid: (ceil(n_out/128), ceil(n_tok/128)), Block: (256, 1, 1) */
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        unsigned gx = (unsigned)((n_out + 127) / 128);
        unsigned gy = (unsigned)((n_tok + 127) / 128);
        cuLaunchKernel(r->gemm_opt_f16, gx, gy, 1, 256, 1, 1,
                       0, r->stream, args, NULL);
        return;  /* BF16 truncation fused into kernel */
    }

    if (r->use_fp8_gemm && r->gemm_opt_fp8 && n_tok >= 16 && !r->use_old_gemm) {
        /* Optimized 128×128 tiled GEMM with FP8 LUT dequant + fused BF16 truncation. */
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        unsigned gx = (unsigned)((n_out + 127) / 128);
        unsigned gy = (unsigned)((n_tok + 127) / 128);
        cuLaunchKernel(r->gemm_opt_fp8, gx, gy, 1, 256, 1, 1,
                       0, r->stream, args, NULL);
        return;  /* BF16 truncation fused into kernel */
    }

    /* Fallback: old 16×16 tiled kernels */
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gy = (unsigned)((n_tok + 15) / 16);

    if (r->use_f16_gemm) {
        unsigned gx64 = (unsigned)((n_out + 63) / 64);
        cuLaunchKernel(r->gemm_f16_f32, gx64, gy, 1, 16, 16, 1,
                       0, r->stream, args, NULL);
    } else if (r->use_fp8_gemm) {
        unsigned gx64 = (unsigned)((n_out + 63) / 64);
        cuLaunchKernel(r->gemm_fp8w_f32, gx64, gy, 1, 16, 16, 1,
                       0, r->stream, args, NULL);
    } else {
        unsigned gx64 = (unsigned)((n_out + 63) / 64);
        cuLaunchKernel(r->gemm_f32_f32, gx64, gy, 1, 16, 16, 1,
                       0, r->stream, args, NULL);
    }
    /* Truncate output to BF16: match ComfyUI's BF16 inference dtype. */
    op_bf16_trunc(r, Y, n_out * n_tok);
}

/* Truncate buffer to BF16 precision */
static void op_bf16_trunc(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    /* ComfyUI runs the DiT in BF16 inference mode — all activations are BF16.
     * We compute in F32 but truncate to BF16 precision to match. */
    void *args[] = {&x, &n};
    cuLaunchKernel(r->truncate_bf16, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void op_silu(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->silu_f32, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void op_gelu(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->gelu_f32, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
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

/* ---- FP8 flash attention workspace ---- */

/* Lazily (re)allocate the Q/K/V FP8 scratch buffers for attention.
 * Returns 0 on success, -1 on OOM (any partial allocs are freed). */
static int ensure_fp8_attn_buf(cuda_qimg_runner *r, size_t n_bytes) {
    if (n_bytes <= r->fp8_attn_buf_n) return 0;
    if (r->d_q_fp8)      { cuMemFree(r->d_q_fp8);      r->d_q_fp8 = 0; }
    if (r->d_k_fp8)      { cuMemFree(r->d_k_fp8);      r->d_k_fp8 = 0; }
    if (r->d_v_fp8)      { cuMemFree(r->d_v_fp8);      r->d_v_fp8 = 0; }
    if (r->d_qkv_scales) { cuMemFree(r->d_qkv_scales); r->d_qkv_scales = 0; }
    if (r->d_qkv_max)    { cuMemFree(r->d_qkv_max);    r->d_qkv_max = 0; }
    r->fp8_attn_buf_n = 0;
    if (cuMemAlloc(&r->d_q_fp8, n_bytes)          != CUDA_SUCCESS) { r->d_q_fp8 = 0; return -1; }
    if (cuMemAlloc(&r->d_k_fp8, n_bytes)          != CUDA_SUCCESS) { r->d_k_fp8 = 0; return -1; }
    if (cuMemAlloc(&r->d_v_fp8, n_bytes)          != CUDA_SUCCESS) { r->d_v_fp8 = 0; return -1; }
    if (cuMemAlloc(&r->d_qkv_scales, 3*sizeof(float)) != CUDA_SUCCESS) { r->d_qkv_scales = 0; return -1; }
    if (cuMemAlloc(&r->d_qkv_max, sizeof(float))      != CUDA_SUCCESS) { r->d_qkv_max = 0; return -1; }
    r->fp8_attn_buf_n = n_bytes;
    return 0;
}

/* Two-pass F32 → e4m3 quantization with per-tensor scale.
 * Writes out[n] uint8 e4m3 bytes and out_scale (1 float = max_abs/448). */
static void quantize_buf_fp8(cuda_qimg_runner *r,
                              CUdeviceptr fp8_out, CUdeviceptr scale_out,
                              CUdeviceptr in, int n) {
    /* Zero the shared max slot (works for fp32 since +0.0 bits == 0) */
    cuMemsetD32Async(r->d_qkv_max, 0, 1, r->stream);
    void *a0[] = {&r->d_qkv_max, &in, &n};
    unsigned blocks = (unsigned)((n + 255) / 256);
    cuLaunchKernel(r->reduce_max_abs, blocks, 1, 1, 256, 1, 1,
                   0, r->stream, a0, NULL);
    void *a1[] = {&fp8_out, &scale_out, &in, &r->d_qkv_max, &n};
    cuLaunchKernel(r->quantize_fp8, blocks, 1, 1, 256, 1, 1,
                   0, r->stream, a1, NULL);
}

static void op_attn(cuda_qimg_runner *r, CUdeviceptr d_out, CUdeviceptr d_q,
                    CUdeviceptr d_k, CUdeviceptr d_v,
                    int n_tok, int n_heads, int head_dim) {
    /* FP8 MMA flash attention path — per-tensor quantize Q/K/V, then mma.sync */
    if (r->use_fp8_attn && r->flash_attn_fp8 && r->quantize_fp8 && r->reduce_max_abs) {
        int dim = n_heads * head_dim;
        size_t need = (size_t)n_tok * dim;
        if (ensure_fp8_attn_buf(r, need) == 0) {
            CUdeviceptr s_q = r->d_qkv_scales + 0 * sizeof(float);
            CUdeviceptr s_k = r->d_qkv_scales + 1 * sizeof(float);
            CUdeviceptr s_v = r->d_qkv_scales + 2 * sizeof(float);
            int n = (int)need;
            quantize_buf_fp8(r, r->d_q_fp8, s_q, d_q, n);
            quantize_buf_fp8(r, r->d_k_fp8, s_k, d_k, n);
            quantize_buf_fp8(r, r->d_v_fp8, s_v, d_v, n);
            /* flash_attn_fp8 launch: grid=(n_heads, ceil(N/64)), block=128 (4 warps).
             * Scales are passed as a device pointer — kernel reads them directly,
             * so no DtoH sync per attention call. */
            unsigned gy = (unsigned)((n_tok + 63) / 64);
            size_t smem = (size_t)(32 * 128 + 128 * 32 + 4 * 16 * 32 * sizeof(float));  /* smK+smVT+smP */
            void *args[] = {&d_out, &r->d_q_fp8, &r->d_k_fp8, &r->d_v_fp8,
                            &n_tok, &n_heads, &head_dim,
                            &r->d_qkv_scales};
            cuLaunchKernel(r->flash_attn_fp8,
                           (unsigned)n_heads, gy, 1,
                           128, 1, 1, smem, r->stream, args, NULL);
            return;
        }
        /* ensure_fp8_attn_buf OOM → fall through to F32 path */
    }

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
    CU_CHECK_NULL(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK_NULL(cuCtxSetCurrent(ctx));
    CUstream stream;
    CU_CHECK_NULL(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    if (verbose) {
        char name[256]; cuDeviceGetName(name, sizeof(name), dev);
        size_t mem; cuDeviceTotalMem(&mem, dev);
        fprintf(stderr, "cuda_qimg: %s (sm_%d, %.1f GB)\n", name, sm, (float)mem/(1<<30));
    }

    /* Compile kernels. Concat order:
     *   cuda_kernels_common_src (opens extern "C")
     *   + qimg_kernel_src (closes extern "C" — provides to_bf16 device fn)
     *   + extern "C" { + fp8_mma_kernels_src + }
     * The FP8 MMA kernels live in a second extern "C" block so they're callable
     * by name via NVRTC, and can reference to_bf16 from the previous block. */
    const char *mma_open  = "\nextern \"C\" {\n";
    const char *mma_close = "\n} /* extern C (fp8_mma_kernels) */\n";
    size_t len1 = strlen(cuda_kernels_common_src);
    size_t len2 = strlen(qimg_kernel_src);
    size_t len3 = strlen(fp8_mma_kernels_src);
    size_t lo = strlen(mma_open);
    size_t lc = strlen(mma_close);
    char *full_src = (char *)malloc(len1 + len2 + lo + len3 + lc + 1);
    char *p = full_src;
    memcpy(p, cuda_kernels_common_src, len1); p += len1;
    memcpy(p, qimg_kernel_src, len2);         p += len2;
    memcpy(p, mma_open, lo);                  p += lo;
    memcpy(p, fp8_mma_kernels_src, len3);     p += len3;
    memcpy(p, mma_close, lc);                 p += lc;
    *p = '\0';

    CUmodule module;
    int rc = cu_compile_kernels(&module, dev, full_src, "qimg.cu", verbose, "cuda_qimg");
    free(full_src);
    if (rc < 0) return NULL;

    cuda_qimg_runner *r = (cuda_qimg_runner *)calloc(1, sizeof(*r));
    r->device = dev; r->ctx = ctx; r->stream = stream;
    r->module = module; r->sm_version = sm; r->verbose = verbose;

    #define GET(field, name) cuModuleGetFunction(&r->field, module, name)
    GET(gemm_f16_f32, "gemm_tiled_f16_f32");  /* Use tiled (correct) kernel, not MMA (has fragment mapping bug) */
    GET(gemm_f32_f32, "gemm_f32_f32");
    GET(gemm_fp8w_f32, "gemm_fp8w_f32");
    GET(gemm_opt_f16, "gemm_opt_f16");
    GET(gemm_opt_fp8, "gemm_opt_fp8");
    if (cuModuleGetFunction(&r->gemm_fp8_mma, module, "gemm_fp8_scaled_f32") != CUDA_SUCCESS)
        r->gemm_fp8_mma = NULL;
    if (cuModuleGetFunction(&r->gemm_fp8_mma_bf16, module, "gemm_fp8_scaled_bf16") != CUDA_SUCCESS)
        r->gemm_fp8_mma_bf16 = NULL;
    if (cuModuleGetFunction(&r->gemm_fp8_mma_tc128, module, "gemm_fp8_scaled_tc128_f32") != CUDA_SUCCESS)
        r->gemm_fp8_mma_tc128 = NULL;
    if (cuModuleGetFunction(&r->flash_attn_fp8, module, "flash_attn_fp8") != CUDA_SUCCESS)
        r->flash_attn_fp8 = NULL;
    if (cuModuleGetFunction(&r->quantize_fp8, module, "quantize_to_fp8_e4m3") != CUDA_SUCCESS)
        r->quantize_fp8 = NULL;
    if (cuModuleGetFunction(&r->reduce_max_abs, module, "reduce_max_abs_f32") != CUDA_SUCCESS)
        r->reduce_max_abs = NULL;
    if (cuModuleGetFunction(&r->fn_zero_f32, module, "zero_f32") != CUDA_SUCCESS)
        r->fn_zero_f32 = NULL;
    /* Enable FP8 weight format: LUT-based dequant works on any SM.
     * Use FP8 byte uploads + LUT GEMM to save 4× VRAM vs F32 — critical for 8GB GPUs. */
    r->use_fp8_gemm = 0;
    if (r->gemm_opt_fp8 || r->gemm_fp8w_f32) {
        r->use_fp8_gemm = 1;
        if (verbose)
            fprintf(stderr, "cuda_qimg: FP8 LUT GEMM enabled (sm_%d)\n", sm);
    }
    /* FP8 tensor-core MMA path — default ON when hardware supports it.
     *
     * Uses per-tensor input scaling (reduce_max_abs + in-kernel divide-at-load
     * and multiply-at-writeback) so GELU outputs that exceed ±448 (mlp_fc2
     * input peaks at ~650 in qwen-image) don't saturate the e4m3 cvt.
     * End-to-end vs ComfyUI at 256x256/10 steps: corr=0.9991, matches
     * LUT path's 0.9999 within FP8 quantization noise.
     * Opt out with QIMG_FP8_MMA=0 to A/B test against the LUT path. */
    r->use_fp8_mma = 0;
    if (r->gemm_fp8_mma && r->use_fp8_gemm && sm >= 89 && r->reduce_max_abs) {
        const char *env = getenv("QIMG_FP8_MMA");
        r->use_fp8_mma = (env && env[0] == '0') ? 0 : 1;
        if (verbose && r->use_fp8_mma)
            fprintf(stderr, "cuda_qimg: FP8 MMA tensor-core GEMM enabled (sm_%d)\n", sm);
    }
    /* BF16-X MMA variant: opt-in via QIMG_FP8_MMA_BF16=1. Measured 0.86 dB
     * WORSE than the F32-X variant vs ComfyUI (256x256/10steps/seed42,
     * 29.34 vs 30.20 dB), so default is off. Kept around for future A/B. */
    r->use_fp8_mma_bf16 = 0;
    if (r->use_fp8_mma && r->gemm_fp8_mma_bf16) {
        const char *env = getenv("QIMG_FP8_MMA_BF16");
        if (env && env[0] == '1') {
            r->use_fp8_mma_bf16 = 1;
            if (verbose)
                fprintf(stderr, "cuda_qimg: FP8 MMA BF16-X sibling enabled (opt-in)\n");
        }
    }
    /* 128×128-tile MMA variant: opt-in via QIMG_FP8_TC128=1 during bring-up.
     * Larger CTA tile (16 warps, 8 MMAs/warp) for higher throughput at typical
     * qwen-image DiT shapes (n_out/n_in ≥ 3072). */
    r->use_fp8_mma_tc128 = 0;
    if (r->use_fp8_mma && r->gemm_fp8_mma_tc128) {
        const char *env = getenv("QIMG_FP8_TC128");
        if (env && env[0] == '1') {
            r->use_fp8_mma_tc128 = 1;
            if (verbose)
                fprintf(stderr, "cuda_qimg: FP8 MMA tc128 (128×128 tile) enabled (opt-in)\n");
        }
    }
    /* FP8 flash attention: opt-in via QIMG_FP8_ATTN=1. */
    r->use_fp8_attn = 0;
    {
        const char *env = getenv("QIMG_FP8_ATTN");
        if (env && env[0] != '0' && r->flash_attn_fp8 && r->quantize_fp8 && r->reduce_max_abs && sm >= 89) {
            r->use_fp8_attn = 1;
            if (verbose)
                fprintf(stderr, "cuda_qimg: FP8 MMA flash attention enabled (sm_%d)\n", sm);
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
    GET(cfg_combine, "cfg_combine_f32");
    GET(rope_2d, "rope_2d_f32");
    GET(rope_1d, "rope_1d_f32");
    GET(vae_conv2d, "vae_conv2d_f32");
    GET(vae_rmsnorm, "vae_rmsnorm_f32");
    GET(vae_silu, "vae_silu_f32");
    GET(nn_upsample2x, "nn_upsample2x_f32");
    GET(rmsnorm_weighted, "rmsnorm_weighted_f32");
    GET(truncate_bf16, "truncate_bf16_f32");
    GET(quantize_fp8_rt, "quantize_fp8_roundtrip_f32");
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
        /* Scratch working set at 512x512 is ~300 MB (d_scratch1/2/3,
         * d_q/d_k/d_v/d_attn_out, d_img/d_txt, modulation). Also reserve
         * one block (324 MB) for the on-demand scratch slot that
         * qimg_load_block_into_slot uses — it eliminates the cuMemAlloc
         * cost per block load. */
        size_t workspace = (500ULL + 324ULL) * 1024 * 1024;
        int max_preload = (free_mem > workspace)
            ? (int)((free_mem - workspace) / block_bytes) : 0;
        if (max_preload > r->dit_n_blocks) max_preload = r->dit_n_blocks;

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

    /* Allocate the on-demand scratch slot once if we didn't preload every
     * block. Uses ~324 MB on top of the preloaded weights but eliminates
     * the 30 cuMemAlloc/cuMemFree per on-demand load. */
    if (r->n_preloaded < r->dit_n_blocks) {
        if (qimg_alloc_scratch_block(r) == 0) {
            if (r->verbose)
                fprintf(stderr, "cuda_qimg: on-demand scratch slot allocated\n");
        } else {
            fprintf(stderr, "cuda_qimg: scratch slot alloc failed — "
                            "falling back to per-block alloc\n");
        }
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
    if (r->scratch_block_ready) {
        qimg_free_block(&r->scratch_block);
        r->scratch_block_ready = 0;
    }
    CUdeviceptr *globals = &r->d_img_in_w;
    for (int i = 0; i < 13; i++) { if (globals[i]) cuMemFree(globals[i]); }
    /* FP8 attention workspace (lazily allocated by ensure_fp8_attn_buf) */
    if (r->d_q_fp8)      cuMemFree(r->d_q_fp8);
    if (r->d_k_fp8)      cuMemFree(r->d_k_fp8);
    if (r->d_v_fp8)      cuMemFree(r->d_v_fp8);
    if (r->d_qkv_scales) cuMemFree(r->d_qkv_scales);
    if (r->d_qkv_max)    cuMemFree(r->d_qkv_max);
    if (r->dit_st) safetensors_close((st_context *)r->dit_st);
    if (r->vae_st) safetensors_close((st_context *)r->vae_st);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->ctx) cuDevicePrimaryCtxRelease(r->device);
    free(r);
}


/* ---- Per-forward state used by qimg_forward_block ---- */
typedef struct {
    /* Token buffers owned by the caller of qimg_forward_block */
    CUdeviceptr d_img;   /* [n_img, dim] */
    CUdeviceptr d_txt;   /* [n_txt, dim] */
    /* Joint QKV / attention scratch: layout is [txt | img] along token axis */
    CUdeviceptr d_q, d_k, d_v, d_attn_out;
    /* Block-local scratch */
    CUdeviceptr d_scratch1, d_scratch2, d_scratch3;
    int n_img;
    int n_txt;
    int n_total;  /* = n_img + n_txt */
} qimg_fwd_state_t;

/* Run a single DiT block on the given state. Block weights (blk) and
 * modulation tensors (d_img_mod, d_txt_mod) are SHARED across passes when
 * CFG batching calls this twice per block for cond and uncond.
 *
 * Returns 0 on success. */
static int qimg_forward_block(cuda_qimg_runner *r, qimg_fwd_state_t *st,
                              const qimg_block_gpu *blk, int L,
                              CUdeviceptr d_img_mod, CUdeviceptr d_txt_mod,
                              int hp_rope, int wp_rope,
                              int t_dim_rope, int h_dim_rope, int w_dim_rope,
                              float rope_theta)
{
    CUstream s = r->stream;
    int dim = r->dit_dim, nh = r->dit_n_heads, hd = r->dit_head_dim, mlp_h = r->dit_mlp_h;
    int n_img = st->n_img, n_txt = st->n_txt, n_total = st->n_total;
    CUdeviceptr d_img = st->d_img, d_txt = st->d_txt;
    CUdeviceptr d_q = st->d_q, d_k = st->d_k, d_v = st->d_v;
    CUdeviceptr d_attn_out = st->d_attn_out;
    CUdeviceptr d_scratch1 = st->d_scratch1, d_scratch2 = st->d_scratch2, d_scratch3 = st->d_scratch3;

    /* -- adaLN + QKV projections -- */
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
    op_adaln(r, d_scratch1, d_img, img_sh1, img_sc1, n_img, dim);
    /* adaLN text → d_scratch2 */
    op_adaln(r, d_scratch2, d_txt, txt_sh1, txt_sc1, n_txt, dim);

    /* Image QKV → offset into joint buffers at [n_txt:] */
    CUdeviceptr d_img_q = d_q + (size_t)n_txt * dim * sizeof(float);
    CUdeviceptr d_img_k = d_k + (size_t)n_txt * dim * sizeof(float);
    CUdeviceptr d_img_v = d_v + (size_t)n_txt * dim * sizeof(float);
    op_gemm(r, d_img_q, blk->attn_q_w, d_scratch1, blk->attn_q_b, dim, dim, n_img);
    op_gemm(r, d_img_k, blk->attn_k_w, d_scratch1, blk->attn_k_b, dim, dim, n_img);
    op_gemm(r, d_img_v, blk->attn_v_w, d_scratch1, blk->attn_v_b, dim, dim, n_img);

    /* Text QKV → offset at [0:n_txt] */
    CUdeviceptr d_txt_q = d_q;
    CUdeviceptr d_txt_k = d_k;
    CUdeviceptr d_txt_v = d_v;
    op_gemm(r, d_txt_q, blk->attn_add_q_w, d_scratch2, blk->attn_add_q_b, dim, dim, n_txt);
    op_gemm(r, d_txt_k, blk->attn_add_k_w, d_scratch2, blk->attn_add_k_b, dim, dim, n_txt);
    op_gemm(r, d_txt_v, blk->attn_add_v_w, d_scratch2, blk->attn_add_v_b, dim, dim, n_txt);

    /* QK RMSNorm (per-head) */
    op_rmsnorm_ph(r, d_img_q, blk->norm_q_w, n_img, nh, hd);
    op_rmsnorm_ph(r, d_img_k, blk->norm_k_w, n_img, nh, hd);
    op_rmsnorm_ph(r, d_txt_q, blk->norm_added_q_w, n_txt, nh, hd);
    op_rmsnorm_ph(r, d_txt_k, blk->norm_added_k_w, n_txt, nh, hd);

    /* RoPE: image 2D, text 1D with txt_start offset */
    {
        void *rope2d_args[] = {&d_img_q, &d_img_k,
                               &n_img, &nh, &hd, &hp_rope, &wp_rope,
                               &t_dim_rope, &h_dim_rope, &w_dim_rope, &rope_theta};
        cuLaunchKernel(r->rope_2d, (unsigned)n_img, 1, 1,
                       (unsigned)nh, 1, 1, 0, s, rope2d_args, NULL);
        int txt_start = hp_rope > wp_rope ? hp_rope / 2 : wp_rope / 2;
        void *rope1d_args[] = {&d_txt_q, &d_txt_k,
                               &n_txt, &nh, &hd, &txt_start,
                               &t_dim_rope, &h_dim_rope, &w_dim_rope, &rope_theta};
        cuLaunchKernel(r->rope_1d, (unsigned)n_txt, 1, 1,
                       (unsigned)nh, 1, 1, 0, s, rope1d_args, NULL);
    }

    /* Joint attention: Q/K/V concatenated as [txt, img] */
    op_attn(r, d_attn_out, d_q, d_k, d_v, n_total, nh, hd);

    /* Output projections */
    CUdeviceptr d_img_attn = d_attn_out + (size_t)n_txt * dim * sizeof(float);
    CUdeviceptr d_txt_attn = d_attn_out;
    op_gemm(r, d_scratch1, blk->attn_out_w, d_img_attn, blk->attn_out_b, dim, dim, n_img);
    op_gemm(r, d_scratch2, blk->attn_add_out_w, d_txt_attn, blk->attn_add_out_b, dim, dim, n_txt);

    /* Gated residual */
    op_gated_add(r, d_img, d_scratch1, img_g1, n_img, dim);
    op_gated_add(r, d_txt, d_scratch2, txt_g1, n_txt, dim);

    /* Image MLP */
    op_adaln(r, d_scratch1, d_img, img_sh2, img_sc2, n_img, dim);
    op_gemm(r, d_scratch3, blk->img_mlp_fc1_w, d_scratch1, blk->img_mlp_fc1_b,
            mlp_h, dim, n_img);
    op_gelu(r, d_scratch3, n_img * mlp_h);
    op_gemm(r, d_scratch1, blk->img_mlp_fc2_w, d_scratch3, blk->img_mlp_fc2_b,
            dim, mlp_h, n_img);
    op_gated_add(r, d_img, d_scratch1, img_g2, n_img, dim);

    /* Text MLP */
    op_adaln(r, d_scratch2, d_txt, txt_sh2, txt_sc2, n_txt, dim);
    op_gemm(r, d_scratch3, blk->txt_mlp_fc1_w, d_scratch2, blk->txt_mlp_fc1_b,
            mlp_h, dim, n_txt);
    op_gelu(r, d_scratch3, n_txt * mlp_h);
    op_gemm(r, d_scratch2, blk->txt_mlp_fc2_w, d_scratch3, blk->txt_mlp_fc2_b,
            dim, mlp_h, n_txt);
    op_gated_add(r, d_txt, d_scratch2, txt_g2, n_txt, dim);

    /* Truncate block output to BF16 precision (match ComfyUI's BF16 compute) */
    op_bf16_trunc(r, d_img, n_img * dim);
    op_bf16_trunc(r, d_txt, n_txt * dim);

    (void)L;  /* unused in refactored body — verbose dumps moved to caller */
    return 0;
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
    /* Save raw patchified input for comparison */
    if (r->verbose >= 3) {
        FILE *_pf = fopen("cuda_dit_img_input.npy", "wb");
        if (_pf) {
            unsigned char _mag[] = {0x93,'N','U','M','P','Y',1,0};
            char _h[256];
            int _hl = snprintf(_h, sizeof(_h),
                "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", n_img, in_ch);
            int _pad = 64 - ((10 + _hl + 1) % 64); if (_pad == 64) _pad = 0;
            for (int _p = 0; _p < _pad; _p++) _h[_hl++] = ' ';
            _h[_hl++] = '\n';
            unsigned short _hs = (unsigned short)_hl;
            fwrite(_mag, 1, 8, _pf); fwrite(&_hs, 2, 1, _pf);
            fwrite(_h, 1, _hl, _pf);
            fwrite(img_tokens, sizeof(float), (size_t)n_img*in_ch, _pf);
            fclose(_pf);
            fprintf(stderr, "    Saved cuda_dit_img_input.npy [%d,%d]\n", n_img, in_ch);
        }
    }
    cuMemAlloc(&d_txt_in, (size_t)n_txt * txt_dim * sizeof(float));
    cuMemcpyHtoD(d_txt_in, txt_tokens, (size_t)n_txt * txt_dim * sizeof(float));

    /* Match ComfyUI: apply_model casts input to BF16 (inference dtype).
     * This is critical — BF16 truncation of activations changes the model output
     * significantly (std=0.44 vs 1.11 without it). */
    {
        void *bf_args[] = {&d_img_in, &(int){n_img * in_ch}};
        cuLaunchKernel(r->truncate_bf16, (unsigned)((n_img*in_ch+255)/256), 1, 1,
                       256, 1, 1, 0, s, bf_args, NULL);
        void *bf_args2[] = {&d_txt_in, &(int){n_txt * txt_dim}};
        cuLaunchKernel(r->truncate_bf16, (unsigned)((n_txt*txt_dim+255)/256), 1, 1,
                       256, 1, 1, 0, s, bf_args2, NULL);
    }

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

    /* Save temb for comparison */
    if (r->verbose >= 3) {
        float *_temb = (float*)malloc((size_t)dim*sizeof(float));
        cuStreamSynchronize(s); cuMemcpyDtoH(_temb, d_t_emb, (size_t)dim*sizeof(float));
        float _std = 0, _mean = 0;
        for (int i = 0; i < dim; i++) _mean += _temb[i];
        _mean /= dim;
        for (int i = 0; i < dim; i++) _std += (_temb[i]-_mean)*(_temb[i]-_mean);
        _std = sqrtf(_std / dim);
        fprintf(stderr, "    temb: std=%.6f\n", _std);
        free(_temb);
    }

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

    /* Helper: save GPU tensor as .npy [rows, cols] */
    #define DIT_SAVE_NPY(fname, ptr, rows, cols) do { \
        if (r->verbose >= 3) { \
            cuStreamSynchronize(s); \
            size_t _n = (size_t)(rows) * (cols); \
            float *_buf = (float*)malloc(_n * sizeof(float)); \
            cuMemcpyDtoH(_buf, ptr, _n * sizeof(float)); \
            FILE *_fp = fopen(fname, "wb"); \
            if (_fp) { \
                unsigned char _magic[] = {0x93,'N','U','M','P','Y',1,0}; \
                char _hdr[256]; \
                int _hl = snprintf(_hdr, sizeof(_hdr), \
                    "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", (rows), (cols)); \
                int _pad = 64 - ((10 + _hl + 1) % 64); \
                if (_pad == 64) _pad = 0; \
                for (int _p = 0; _p < _pad; _p++) _hdr[_hl++] = ' '; \
                _hdr[_hl++] = '\n'; \
                unsigned short _hs = (unsigned short)_hl; \
                fwrite(_magic, 1, 8, _fp); fwrite(&_hs, 2, 1, _fp); \
                fwrite(_hdr, 1, _hl, _fp); fwrite(_buf, sizeof(float), _n, _fp); \
                fclose(_fp); \
                fprintf(stderr, "    Saved %s [%d,%d]\n", fname, (rows), (cols)); \
            } \
            free(_buf); \
        } \
    } while(0)

    /* BF16 truncation after projection (matching ComfyUI BF16 inference) */
    op_bf16_trunc(r, d_img, n_img * dim);
    op_bf16_trunc(r, d_txt, n_txt * dim);
    op_bf16_trunc(r, d_t_emb, dim);

    DIT_SAVE_NPY("cuda_dit_img_projected.npy", d_img, n_img, dim);
    DIT_SAVE_NPY("cuda_dit_txt_projected.npy", d_txt, n_txt, dim);

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

    /* Per-block scratch buffers hoisted out of the loop — cuMemAlloc/cuMemFree
     * are synchronous driver calls, each costs ~0.5-1 ms. Allocating these
     * per-block adds ~90 ms/block of overhead on a 60-block, 2-CFG pipeline. */
    CUdeviceptr d_t_silu, d_img_mod, d_txt_mod;
    cuMemAlloc(&d_t_silu,  (size_t)dim * sizeof(float));
    cuMemAlloc(&d_img_mod, (size_t)6 * dim * sizeof(float));
    cuMemAlloc(&d_txt_mod, (size_t)6 * dim * sizeof(float));

    /* Compute image patch grid for RoPE */
    int hp_rope = (int)sqrtf((float)n_img);
    int wp_rope = n_img / hp_rope;
    float rope_theta = 10000.0f;
    int t_dim_rope = 16, h_dim_rope = 56, w_dim_rope = 56;

    /* 4. Process all blocks */
    for (int L = 0; L < r->dit_n_blocks; L++) {
        if (r->verbose && (L % 10 == 0 || L == r->dit_n_blocks - 1))
            fprintf(stderr, "\r  cuda_qimg: block %d/%d", L + 1, r->dit_n_blocks);

        /* Use preloaded block if available, else copy into the pre-allocated
         * scratch slot via async H2D. The slot is reused for every on-demand
         * block, so we pay the cuMemAlloc cost exactly once (at init) instead
         * of 30× per block. */
        qimg_block_gpu blk;
        int need_free = 0;
        if (L < r->n_preloaded && r->gpu_blocks[L].attn_q_w) {
            blk = r->gpu_blocks[L];
        } else {
            struct timespec _lt0, _lt1;
            if (L <= 15 && r->verbose >= 2) clock_gettime(CLOCK_MONOTONIC, &_lt0);
            if (qimg_load_block_into_slot(r, L, &blk) != 0) {
                fprintf(stderr, "cuda_qimg: block %d slot load failed\n", L);
                return -1;
            }
            /* No per-block free — slot is owned by the runner. */
            if (L <= 15 && r->verbose >= 2) {
                cuStreamSynchronize(s);
                clock_gettime(CLOCK_MONOTONIC, &_lt1);
                long _dns = (_lt1.tv_sec-_lt0.tv_sec)*1000000000L + (_lt1.tv_nsec-_lt0.tv_nsec);
                fprintf(stderr, "    block %d slot-load: %.2f ms\n", L, _dns/1e6);
            }
        }

        /* -- Image modulation: SiLU(t_emb) → Linear → 6×dim --
         * d_t_silu/d_img_mod/d_txt_mod are pre-allocated outside the loop. */
        cuMemcpyDtoDAsync(d_t_silu, d_t_emb, (size_t)dim * sizeof(float), s);
        op_silu(r, d_t_silu, dim);
        op_gemm(r, d_img_mod, blk.img_mod_w, d_t_silu, blk.img_mod_b, 6 * dim, dim, 1);
        op_gemm(r, d_txt_mod, blk.txt_mod_w, d_t_silu, blk.txt_mod_b, 6 * dim, dim, 1);

        /* Pack state and run the shared block forward */
        qimg_fwd_state_t _st;
        _st.d_img = d_img; _st.d_txt = d_txt;
        _st.d_q = d_q; _st.d_k = d_k; _st.d_v = d_v; _st.d_attn_out = d_attn_out;
        _st.d_scratch1 = d_scratch1; _st.d_scratch2 = d_scratch2; _st.d_scratch3 = d_scratch3;
        _st.n_img = n_img; _st.n_txt = n_txt; _st.n_total = n_total;
        qimg_forward_block(r, &_st, &blk, L, d_img_mod, d_txt_mod,
                           hp_rope, wp_rope, t_dim_rope, h_dim_rope, w_dim_rope, rope_theta);

        /* Save every block output for comparison */
        if (r->verbose >= 3) {
            char _fn[64];
            snprintf(_fn, sizeof(_fn), "cuda_dit_block%02d_img.npy", L);
            DIT_SAVE_NPY(_fn, d_img, n_img, dim);
            snprintf(_fn, sizeof(_fn), "cuda_dit_block%02d_txt.npy", L);
            DIT_SAVE_NPY(_fn, d_txt, n_txt, dim);
        }

        /* Block weights only freed if loaded on-demand (and not using slot). */
        if (need_free) {
            cuStreamSynchronize(s);
            CUdeviceptr *ptrs = (CUdeviceptr *)&blk;
            int np = sizeof(qimg_block_gpu) / sizeof(CUdeviceptr);
            for (int pi = 0; pi < np; pi++)
                if (ptrs[pi]) cuMemFree(ptrs[pi]);
        }
    }
    if (r->verbose) fprintf(stderr, "\n");

    /* 5. Final output: adaLN → proj_out */
    {
        CUdeviceptr d_t_silu;
        cuMemAlloc(&d_t_silu, (size_t)dim * sizeof(float));
        cuMemcpyDtoDAsync(d_t_silu, d_t_emb, (size_t)dim * sizeof(float), s);
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

        DIT_SAVE_NPY("cuda_dit_norm_out.npy", d_scratch1, n_img, dim);

        /* proj_out: [n_img, dim] → [n_img, in_ch] */
        CUdeviceptr d_out;
        cuMemAlloc(&d_out, (size_t)n_img * in_ch * sizeof(float));
        op_gemm(r, d_out, r->d_proj_out_w, d_scratch1, r->d_proj_out_b,
                in_ch, dim, n_img);

        DIT_SAVE_NPY("cuda_dit_proj_out.npy", d_out, n_img, in_ch);

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
    cuMemFree(d_t_silu); cuMemFree(d_img_mod); cuMemFree(d_txt_mod);

    return 0;
}

/* ---- CFG-batched DiT step ---- */

int cuda_qimg_dit_step_cfg(cuda_qimg_runner *r,
                           const float *img_tokens, int n_img,
                           const float *txt_cond,   int n_txt_cond,
                           const float *txt_uncond, int n_txt_uncond,
                           float timestep,
                           float *out_cond, float *out_uncond)
{
    int dim = r->dit_dim;
    int nh = r->dit_n_heads, hd = r->dit_head_dim;
    int in_ch = r->dit_in_ch, txt_dim = r->dit_txt_dim, mlp_h = r->dit_mlp_h;
    int n_total_cond   = n_img + n_txt_cond;
    int n_total_uncond = n_img + n_txt_uncond;
    int n_total_max = n_total_cond > n_total_uncond ? n_total_cond : n_total_uncond;
    CUstream s = r->stream;

    /* Allocate paired img/txt buffers. Both passes share the same img_tokens
     * at input (same noise) but the hidden states diverge after block 0 so
     * we need two independent d_img buffers. */
    CUdeviceptr d_img_c, d_img_u;
    CUdeviceptr d_txt_c, d_txt_u;
    cuMemAlloc(&d_img_c, (size_t)n_img * dim * sizeof(float));
    cuMemAlloc(&d_img_u, (size_t)n_img * dim * sizeof(float));
    cuMemAlloc(&d_txt_c, (size_t)n_txt_cond   * dim * sizeof(float));
    cuMemAlloc(&d_txt_u, (size_t)n_txt_uncond * dim * sizeof(float));

    /* Upload img (used twice) + both txt conditionings. */
    CUdeviceptr d_img_in;
    cuMemAlloc(&d_img_in, (size_t)n_img * in_ch * sizeof(float));
    cuMemcpyHtoD(d_img_in, img_tokens, (size_t)n_img * in_ch * sizeof(float));
    CUdeviceptr d_txt_in_c, d_txt_in_u;
    cuMemAlloc(&d_txt_in_c, (size_t)n_txt_cond   * txt_dim * sizeof(float));
    cuMemAlloc(&d_txt_in_u, (size_t)n_txt_uncond * txt_dim * sizeof(float));
    cuMemcpyHtoD(d_txt_in_c, txt_cond,   (size_t)n_txt_cond   * txt_dim * sizeof(float));
    cuMemcpyHtoD(d_txt_in_u, txt_uncond, (size_t)n_txt_uncond * txt_dim * sizeof(float));

    /* Timestep embedding — same for cond/uncond */
    CUdeviceptr d_t_emb;
    cuMemAlloc(&d_t_emb, (size_t)dim * sizeof(float));

    /* Scratch buffers sized for the LARGER of the two passes (since we
     * reuse them per-pass via the state struct). */
    size_t scratch_toks = (n_total_max > mlp_h / (sizeof(float))) ? n_total_max : n_total_max;
    (void)scratch_toks;
    size_t max_scratch  = (size_t)n_total_max * dim * sizeof(float);
    size_t ffn_scratch  = (size_t)n_total_max * mlp_h * sizeof(float);
    CUdeviceptr d_scratch1_c, d_scratch2_c, d_scratch3_c;
    CUdeviceptr d_scratch1_u, d_scratch2_u, d_scratch3_u;
    cuMemAlloc(&d_scratch1_c, max_scratch);
    cuMemAlloc(&d_scratch2_c, max_scratch);
    cuMemAlloc(&d_scratch3_c, ffn_scratch);
    cuMemAlloc(&d_scratch1_u, max_scratch);
    cuMemAlloc(&d_scratch2_u, max_scratch);
    cuMemAlloc(&d_scratch3_u, ffn_scratch);

    /* Joint QKV + attention scratch, sized per-pass since n_total differs. */
    CUdeviceptr d_q_c, d_k_c, d_v_c, d_attn_out_c;
    CUdeviceptr d_q_u, d_k_u, d_v_u, d_attn_out_u;
    cuMemAlloc(&d_q_c,        (size_t)n_total_cond   * dim * sizeof(float));
    cuMemAlloc(&d_k_c,        (size_t)n_total_cond   * dim * sizeof(float));
    cuMemAlloc(&d_v_c,        (size_t)n_total_cond   * dim * sizeof(float));
    cuMemAlloc(&d_attn_out_c, (size_t)n_total_cond   * dim * sizeof(float));
    cuMemAlloc(&d_q_u,        (size_t)n_total_uncond * dim * sizeof(float));
    cuMemAlloc(&d_k_u,        (size_t)n_total_uncond * dim * sizeof(float));
    cuMemAlloc(&d_v_u,        (size_t)n_total_uncond * dim * sizeof(float));
    cuMemAlloc(&d_attn_out_u, (size_t)n_total_uncond * dim * sizeof(float));

    /* Modulation buffers — same for cond and uncond since t_emb is shared */
    CUdeviceptr d_t_silu, d_img_mod, d_txt_mod;
    cuMemAlloc(&d_t_silu,  (size_t)dim * sizeof(float));
    cuMemAlloc(&d_img_mod, (size_t)6 * dim * sizeof(float));
    cuMemAlloc(&d_txt_mod, (size_t)6 * dim * sizeof(float));

    /* ---- 1. Timestep embedding: sinusoidal(256) → SiLU(GEMM) → GEMM ---- */
    float t_sin[256];
    int half = 128;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i / (float)half * logf(10000.0f));
        float angle = timestep * freq;
        t_sin[i]        = cosf(angle);
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

    /* ---- 2. Text input: RMSNorm → Linear (both passes) ---- */
    if (r->d_txt_norm_w) {
        void *rn_args[] = {&d_txt_in_c, &r->d_txt_norm_w, &n_txt_cond, &txt_dim};
        cuLaunchKernel(r->rmsnorm_weighted, (unsigned)n_txt_cond, 1, 1,
                       256, 1, 1, 256 * sizeof(float), s, rn_args, NULL);
        void *rn_args2[] = {&d_txt_in_u, &r->d_txt_norm_w, &n_txt_uncond, &txt_dim};
        cuLaunchKernel(r->rmsnorm_weighted, (unsigned)n_txt_uncond, 1, 1,
                       256, 1, 1, 256 * sizeof(float), s, rn_args2, NULL);
    }
    op_gemm(r, d_txt_c, r->d_txt_in_w, d_txt_in_c, r->d_txt_in_b, dim, txt_dim, n_txt_cond);
    op_gemm(r, d_txt_u, r->d_txt_in_w, d_txt_in_u, r->d_txt_in_b, dim, txt_dim, n_txt_uncond);
    cuMemFree(d_txt_in_c); cuMemFree(d_txt_in_u);

    /* ---- 3. Image input (same for cond/uncond) ---- */
    op_gemm(r, d_img_c, r->d_img_in_w, d_img_in, r->d_img_in_b, dim, in_ch, n_img);
    /* Copy to uncond buffer so both start identical */
    cuMemcpyDtoDAsync(d_img_u, d_img_c, (size_t)n_img * dim * sizeof(float), s);
    cuMemFree(d_img_in);

    /* Rope params (same for cond and uncond) */
    int hp_rope = (int)sqrtf((float)n_img);
    int wp_rope = n_img / hp_rope;
    float rope_theta = 10000.0f;
    int t_dim_rope = 16, h_dim_rope = 56, w_dim_rope = 56;

    /* ---- 4. Block loop: each block loaded ONCE, applied to both passes ---- */
    for (int L = 0; L < r->dit_n_blocks; L++) {
        if (r->verbose && (L % 10 == 0 || L == r->dit_n_blocks - 1))
            fprintf(stderr, "\r  cuda_qimg: block %d/%d", L + 1, r->dit_n_blocks);

        qimg_block_gpu blk;
        int need_free = 0;
        if (L < r->n_preloaded && r->gpu_blocks[L].attn_q_w) {
            blk = r->gpu_blocks[L];
        } else {
            if (qimg_load_block_into_slot(r, L, &blk) != 0) {
                fprintf(stderr, "cuda_qimg: block %d slot load failed\n", L);
                return -1;
            }
        }

        /* Modulation (shared between cond/uncond) */
        cuMemcpyDtoDAsync(d_t_silu, d_t_emb, (size_t)dim * sizeof(float), s);
        op_silu(r, d_t_silu, dim);
        op_gemm(r, d_img_mod, blk.img_mod_w, d_t_silu, blk.img_mod_b, 6 * dim, dim, 1);
        op_gemm(r, d_txt_mod, blk.txt_mod_w, d_t_silu, blk.txt_mod_b, 6 * dim, dim, 1);

        /* --- Cond pass --- */
        {
            qimg_fwd_state_t st_c;
            st_c.d_img = d_img_c; st_c.d_txt = d_txt_c;
            st_c.d_q = d_q_c; st_c.d_k = d_k_c; st_c.d_v = d_v_c; st_c.d_attn_out = d_attn_out_c;
            st_c.d_scratch1 = d_scratch1_c; st_c.d_scratch2 = d_scratch2_c; st_c.d_scratch3 = d_scratch3_c;
            st_c.n_img = n_img; st_c.n_txt = n_txt_cond; st_c.n_total = n_total_cond;
            qimg_forward_block(r, &st_c, &blk, L, d_img_mod, d_txt_mod,
                               hp_rope, wp_rope, t_dim_rope, h_dim_rope, w_dim_rope, rope_theta);
        }
        /* --- Uncond pass --- */
        {
            qimg_fwd_state_t st_u;
            st_u.d_img = d_img_u; st_u.d_txt = d_txt_u;
            st_u.d_q = d_q_u; st_u.d_k = d_k_u; st_u.d_v = d_v_u; st_u.d_attn_out = d_attn_out_u;
            st_u.d_scratch1 = d_scratch1_u; st_u.d_scratch2 = d_scratch2_u; st_u.d_scratch3 = d_scratch3_u;
            st_u.n_img = n_img; st_u.n_txt = n_txt_uncond; st_u.n_total = n_total_uncond;
            qimg_forward_block(r, &st_u, &blk, L, d_img_mod, d_txt_mod,
                               hp_rope, wp_rope, t_dim_rope, h_dim_rope, w_dim_rope, rope_theta);
        }

        if (need_free) {
            cuStreamSynchronize(s);
            CUdeviceptr *ptrs = (CUdeviceptr *)&blk;
            int np = sizeof(qimg_block_gpu) / sizeof(CUdeviceptr);
            for (int pi = 0; pi < np; pi++)
                if (ptrs[pi]) cuMemFree(ptrs[pi]);
        }
    }
    if (r->verbose) fprintf(stderr, "\n");

    /* ---- 5. Final output: adaLN → proj_out for BOTH cond and uncond ---- */
    for (int pass = 0; pass < 2; pass++) {
        CUdeviceptr d_img_p = pass ? d_img_u : d_img_c;
        CUdeviceptr d_scratch1_p = pass ? d_scratch1_u : d_scratch1_c;
        float *out_p = pass ? out_uncond : out_cond;

        CUdeviceptr d_t_silu2;
        cuMemAlloc(&d_t_silu2, (size_t)dim * sizeof(float));
        cuMemcpyDtoDAsync(d_t_silu2, d_t_emb, (size_t)dim * sizeof(float), s);
        op_silu(r, d_t_silu2, dim);
        CUdeviceptr d_final_mod;
        cuMemAlloc(&d_final_mod, (size_t)2 * dim * sizeof(float));
        op_gemm(r, d_final_mod, r->d_norm_out_w, d_t_silu2, r->d_norm_out_b,
                2 * dim, dim, 1);
        cuMemFree(d_t_silu2);

        CUdeviceptr f_scale = d_final_mod;
        CUdeviceptr f_shift = d_final_mod + (size_t)dim * sizeof(float);
        op_adaln(r, d_scratch1_p, d_img_p, f_shift, f_scale, n_img, dim);
        cuMemFree(d_final_mod);

        CUdeviceptr d_out;
        cuMemAlloc(&d_out, (size_t)n_img * in_ch * sizeof(float));
        op_gemm(r, d_out, r->d_proj_out_w, d_scratch1_p, r->d_proj_out_b,
                in_ch, dim, n_img);
        cuStreamSynchronize(s);
        cuMemcpyDtoH(out_p, d_out, (size_t)n_img * in_ch * sizeof(float));
        cuMemFree(d_out);
    }

    /* Cleanup */
    cuMemFree(d_img_c); cuMemFree(d_img_u);
    cuMemFree(d_txt_c); cuMemFree(d_txt_u);
    cuMemFree(d_t_emb);
    cuMemFree(d_scratch1_c); cuMemFree(d_scratch2_c); cuMemFree(d_scratch3_c);
    cuMemFree(d_scratch1_u); cuMemFree(d_scratch2_u); cuMemFree(d_scratch3_u);
    cuMemFree(d_q_c); cuMemFree(d_k_c); cuMemFree(d_v_c); cuMemFree(d_attn_out_c);
    cuMemFree(d_q_u); cuMemFree(d_k_u); cuMemFree(d_v_u); cuMemFree(d_attn_out_u);
    cuMemFree(d_t_silu); cuMemFree(d_img_mod); cuMemFree(d_txt_mod);
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
    /* BF16 → F32: take LAST temporal slice only (causal_zero for T=1 input) */
    size_t n2d = (size_t)co * ci * kh * kw;
    float *w2d = (float *)malloc(n2d * sizeof(float));
    int d_last = kd - 1;  /* causal: use last temporal position only */
    for (int o = 0; o < co; o++)
        for (int i = 0; i < ci; i++)
            for (int h = 0; h < kh; h++)
                for (int w = 0; w < kw; w++) {
                    size_t idx3 = ((((size_t)o*ci+i)*kd+d_last)*kh+h)*kw+w;
                    uint32_t bits = (uint32_t)bf[idx3] << 16;
                    float f; memcpy(&f, &bits, 4);
                    w2d[(((size_t)o*ci+i)*kh+h)*kw+w] = f;
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

/* GPU VAE RMS norm launch: L2-normalize along channels, scale by sqrt(C) * gamma */
static void vae_op_gn(cuda_qimg_runner *r, CUdeviceptr out, CUdeviceptr inp,
                      CUdeviceptr gamma, int C, int spatial) {
    void *args[] = {&out, &inp, &gamma, &C, &spatial};
    cuLaunchKernel(r->vae_rmsnorm, (unsigned)((spatial + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
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

/* BF16 truncation for VAE intermediates — DISABLED (ComfyUI F32=BF16). */
static void vae_bf16(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    (void)r; (void)x; (void)n;  /* no-op: F32 is correct */
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
    vae_bf16(r, tmp, ci*sp);
    vae_op_silu(r, tmp, ci*sp);
    vae_bf16(r, tmp, ci*sp);
    CUdeviceptr c1_out; cuMemAlloc(&c1_out, (size_t)co*sp*sizeof(float));
    vae_op_conv2d(r, c1_out, tmp, c1_w, c1_b, ci, h, w, co, 3, 3, 0);
    vae_bf16(r, c1_out, co*sp);
    cuMemFree(tmp);

    tmp = (CUdeviceptr)0; cuMemAlloc(&tmp, (size_t)co*sp*sizeof(float));
    vae_op_gn(r, tmp, c1_out, n2_g, co, sp);
    vae_bf16(r, tmp, co*sp);
    vae_op_silu(r, tmp, co*sp);
    vae_bf16(r, tmp, co*sp);
    CUdeviceptr c2_out; cuMemAlloc(&c2_out, (size_t)co*sp*sizeof(float));
    vae_op_conv2d(r, c2_out, tmp, c2_w, c2_b, co, h, w, co, 3, 3, 0);
    vae_bf16(r, c2_out, co*sp);
    cuMemFree(tmp); cuMemFree(c1_out);

    /* Shortcut + residual */
    CUdeviceptr out; cuMemAlloc(&out, (size_t)co*sp*sizeof(float));
    if (sc_w) {
        /* 1x1 conv shortcut + residual: out = shortcut(x) + c2_out */
        vae_op_conv2d(r, out, x, sc_w, sc_b, ci, h, w, co, 1, 1, 0);
        int n = co * sp;
        float one = 1.0f;
        void *add_args[] = {&out, &c2_out, &one, &n};
        cuLaunchKernel(r->euler_step, (unsigned)((n+255)/256), 1, 1,
                       256, 1, 1, 0, r->stream, add_args, NULL);
        vae_bf16(r, out, n);
    } else {
        /* Same channels: just add residual */
        int n = co * sp;
        cuMemcpyDtoD(out, x, (size_t)n * sizeof(float));
        float one = 1.0f;
        void *add_args[] = {&out, &c2_out, &one, &n};
        cuLaunchKernel(r->euler_step, (unsigned)((n+255)/256), 1, 1,
                       256, 1, 1, 0, r->stream, add_args, NULL);
        vae_bf16(r, out, n);
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

    /* Debug: dump GPU buffer stats (verbose >= 2) */
    #define VAE_DUMP(label, ptr, count) do { if (r->verbose >= 2) { \
        cuStreamSynchronize(s); \
        float *_t = (float*)malloc((count)*sizeof(float)); \
        cuMemcpyDtoH(_t, ptr, (count)*sizeof(float)); \
        float _mn=_t[0],_mx=_t[0],_s=0; int _nn=0; \
        for(int _i=0;_i<(count);_i++){ \
            if(_t[_i]!=_t[_i]){_nn++;}else{ \
                if(_t[_i]<_mn){_mn=_t[_i];} if(_t[_i]>_mx){_mx=_t[_i];} _s+=_t[_i];}} \
        fprintf(stderr, "  [vae] %s: min=%.4f max=%.4f mean=%.4f nan=%d/%d\n", \
                label, _mn, _mx, _s/((count)-_nn), _nn, (count)); \
        free(_t); } } while(0)

    /* Save GPU buffer as .npy file for comparison with ComfyUI */
    #define VAE_SAVE_NPY(fname, ptr, ndim, dims) do { \
        if (r->verbose >= 3) { \
            cuStreamSynchronize(s); \
            size_t _total = 1; \
            const int *_dims_p = (dims); \
            for (int _di = 0; _di < (ndim); _di++) _total *= _dims_p[_di]; \
            float *_buf = (float*)malloc(_total * sizeof(float)); \
            cuMemcpyDtoH(_buf, ptr, _total * sizeof(float)); \
            FILE *_fp = fopen(fname, "wb"); \
            if (_fp) { \
                /* NumPy .npy format header */ \
                unsigned char _magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0}; \
                char _hdr[256]; \
                int _hlen; \
                if ((ndim) == 1) _hlen = snprintf(_hdr, sizeof(_hdr), \
                    "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }", _dims_p[0]); \
                else if ((ndim) == 2) _hlen = snprintf(_hdr, sizeof(_hdr), \
                    "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", _dims_p[0], _dims_p[1]); \
                else _hlen = snprintf(_hdr, sizeof(_hdr), \
                    "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d), }", _dims_p[0], _dims_p[1], _dims_p[2]); \
                int _pad = 64 - ((10 + _hlen + 1) % 64); \
                if (_pad == 64) _pad = 0; \
                for (int _p = 0; _p < _pad; _p++) _hdr[_hlen++] = ' '; \
                _hdr[_hlen++] = '\n'; \
                unsigned short _hl = (unsigned short)_hlen; \
                fwrite(_magic, 1, 8, _fp); \
                fwrite(&_hl, 2, 1, _fp); \
                fwrite(_hdr, 1, _hlen, _fp); \
                fwrite(_buf, sizeof(float), _total, _fp); \
                fclose(_fp); \
                fprintf(stderr, "  [vae] saved %s (%zu floats)\n", fname, _total); \
            } \
            free(_buf); \
        } \
    } while(0)

    /* Upload latent */
    CUdeviceptr d_x;
    cuMemAlloc(&d_x, (size_t)c * h * w * sizeof(float));
    cuMemcpyHtoD(d_x, latent, (size_t)c * h * w * sizeof(float));
    VAE_DUMP("latent_input", d_x, c*h*w);

    /* post_quant_conv (conv2): 1×1×1 → effectively pointwise */
    CUdeviceptr d_pqc_w = vae_upload_f32(st, "conv2.weight", s);
    CUdeviceptr d_pqc_b = vae_upload_f32(st, "conv2.bias", s);
    if (d_pqc_w) {
        CUdeviceptr d_tmp; cuMemAlloc(&d_tmp, (size_t)c*h*w*sizeof(float));
        vae_op_conv2d(r, d_tmp, d_x, d_pqc_w, d_pqc_b, c, h, w, c, 1, 1, 0);
        vae_bf16(r, d_tmp, c*h*w);
        cuMemFree(d_x); d_x = d_tmp;
        cuMemFree(d_pqc_w); cuMemFree(d_pqc_b);
    }
    VAE_DUMP("post_quant", d_x, c*h*w);
    { int _d[] = {c, h, w}; VAE_SAVE_NPY("cuda_vae_post_quant.npy", d_x, 3, _d); }

    /* decoder.conv1: 16→384, 3×3 */
    int co_c1, ci_c1;
    CUdeviceptr d_c1_w = vae_upload_conv3d(st, "decoder.conv1.weight", &co_c1, &ci_c1, s);
    CUdeviceptr d_c1_b = vae_upload_f32(st, "decoder.conv1.bias", s);
    c = co_c1;
    {
        CUdeviceptr d_tmp; cuMemAlloc(&d_tmp, (size_t)c*h*w*sizeof(float));
        vae_op_conv2d(r, d_tmp, d_x, d_c1_w, d_c1_b, ci_c1, h, w, c, 3, 3, 0);
        vae_bf16(r, d_tmp, c*h*w);
        cuMemFree(d_x); d_x = d_tmp;
        cuMemFree(d_c1_w); cuMemFree(d_c1_b);
    }
    fprintf(stderr, "  after conv1: [%d, %d, %d]\n", c, h, w);
    VAE_DUMP("conv1_out", d_x, c*h*w);
    { int _d[] = {c, h, w}; VAE_SAVE_NPY("cuda_vae_conv1.npy", d_x, 3, _d); }

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
      if (scw) { cuMemFree(scw); } if (scb) { cuMemFree(scb); } }
    { int _d[] = {c, h, w}; VAE_SAVE_NPY("cuda_vae_middle_0.npy", d_x, 3, _d); }

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

        /* Split Q[spatial, C], K[spatial, C], V[spatial, C] from interleaved [spatial, 3C] */
        float *h_q = (float *)malloc((size_t)spatial * c * sizeof(float));
        float *h_k = (float *)malloc((size_t)spatial * c * sizeof(float));
        float *h_v = (float *)malloc((size_t)spatial * c * sizeof(float));
        for (int s_pos = 0; s_pos < spatial; s_pos++) {
            memcpy(h_q + (size_t)s_pos * c, h_qkv_t + (size_t)s_pos * 3 * c,           (size_t)c * sizeof(float));
            memcpy(h_k + (size_t)s_pos * c, h_qkv_t + (size_t)s_pos * 3 * c + c,       (size_t)c * sizeof(float));
            memcpy(h_v + (size_t)s_pos * c, h_qkv_t + (size_t)s_pos * 3 * c + 2 * c,   (size_t)c * sizeof(float));
        }

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
        free(h_q); free(h_k); free(h_v);

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
        vae_bf16(r, d_x, c * spatial);
        cuStreamSynchronize(s);
    }
    { int _d[] = {c, h, w}; VAE_SAVE_NPY("cuda_vae_middle_1.npy", d_x, 3, _d); }

    /* mid.2 */
    { LOAD_RB_NAMED("decoder.middle.2", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
      CUdeviceptr d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b, scw, scb, c, c, h, w);
      cuMemFree(d_x); d_x = d_tmp;
      cuMemFree(n1); cuMemFree(c1w); cuMemFree(c1b); cuMemFree(n2); cuMemFree(c2w); cuMemFree(c2b);
      if (scw) { cuMemFree(scw); } if (scb) { cuMemFree(scb); } }
    fprintf(stderr, "  after middle: [%d, %d, %d]\n", c, h, w);
    VAE_DUMP("middle_out", d_x, c*h*w);
    { int _d[] = {c, h, w}; VAE_SAVE_NPY("cuda_vae_middle_2.npy", d_x, 3, _d); }

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
            if (scw) { cuMemFree(scw); } if (scb) { cuMemFree(scb); }
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
            vae_bf16(r, d_x, c*h*w);
            fprintf(stderr, "  upsample %d: [%d, %d, %d]\n", i, c, h, w);
        }

        /* Dump after each upsample iteration */
        { char _lbl[64]; snprintf(_lbl, sizeof(_lbl), "block_%d [%d,%d,%d]", i, c, h, w);
          VAE_DUMP(_lbl, d_x, c*h*w); }
        { char _fn[64]; snprintf(_fn, sizeof(_fn), "cuda_vae_upsample_%d.npy", i);
          int _d[] = {c, h, w}; VAE_SAVE_NPY(_fn, d_x, 3, _d); }
    }
    #undef LOAD_RB_NAMED

    /* Head: GroupNorm → SiLU → Conv(96→3) */
    {
        VAE_DUMP("pre_head", d_x, c*h*w);
        CUdeviceptr d_gn = vae_upload_f32(st, "decoder.head.0.gamma", s);
        int spatial = h * w;
        CUdeviceptr d_tmp; cuMemAlloc(&d_tmp, (size_t)c*spatial*sizeof(float));
        vae_op_gn(r, d_tmp, d_x, d_gn, c, spatial);
        vae_bf16(r, d_tmp, c*spatial);
        VAE_DUMP("head_gn", d_tmp, c*spatial);
        vae_op_silu(r, d_tmp, c * spatial);
        vae_bf16(r, d_tmp, c*spatial);
        VAE_DUMP("head_silu", d_tmp, c*spatial);
        cuMemFree(d_gn);

        int head_co, head_ci;
        CUdeviceptr d_hw = vae_upload_conv3d(st, "decoder.head.2.weight", &head_co, &head_ci, s);
        CUdeviceptr d_hb = vae_upload_f32(st, "decoder.head.2.bias", s);
        CUdeviceptr d_rgb; cuMemAlloc(&d_rgb, (size_t)3*spatial*sizeof(float));
        vae_op_conv2d(r, d_rgb, d_tmp, d_hw, d_hb, c, h, w, 3, 3, 3, 0);  /* zero spatial pad */
        VAE_DUMP("head_conv", d_rgb, 3*spatial);
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
