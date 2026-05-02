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
#include "../cublasew.h"
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
"/* GPU-side FP8 E4M3 → BF16 dequant via constant memory LUT.\n"
" * Populated by qimg_init_fp8_to_bf16_lut(). Used by gemm_bf16_pipe_f32. */\n"
"__device__ __constant__ unsigned short d_fp8_to_bf16_lut[256];\n"
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

/* Gated residual: x += gate * proj, output BF16-truncated inline.
 *
 * Eliminates the trailing op_bf16_trunc(x) pass the caller used to do
 * separately: proj is already BF16 from the GEMM writeback, x is BF16
 * from the previous block end, so the F32 sum is rounded straight to
 * BF16 here instead of stored at F32 precision and rounded again later. */
"__global__ void gated_add_f32(float *__restrict__ x,\n"
"    const float *__restrict__ proj, const float *__restrict__ gate,\n"
"    int N, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= N * dim) return;\n"
"    int col = i % dim;\n"
"    float v = x[i] + gate[col] * proj[i];\n"
"    /* Inline BF16 rounding (round-to-nearest-even) to avoid depending on\n"
"     * a forward-declared helper. */\n"
"    unsigned int _b; memcpy(&_b, &v, 4);\n"
"    _b += 0x7FFFu + ((_b >> 16) & 1u);\n"
"    _b &= 0xFFFF0000u;\n"
"    memcpy(&v, &_b, 4);\n"
"    x[i] = v;\n"
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

/* On-device F32 -> FP8 e4m3 encoder for VAE conv weights. Flattens
 * [co, ci*kh*kw] to [pad_co, n_in_pad] with trailing rows / cols zeroed.
 * Grid: (ceil(n_in_pad/32), ceil(pad_co/8)); Block: (32, 8). */
"__global__ void vae_f32_to_fp8_padded(unsigned char *__restrict__ out,\n"
"    const float *__restrict__ inp, int co, int n_in, int pad_co, int n_in_pad) {\n"
"    int k = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int c = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (k >= n_in_pad || c >= pad_co) return;\n"
"    unsigned char bits = 0;\n"
"    if (c < co && k < n_in) {\n"
"        float f = inp[(long)c * n_in + k];\n"
"        /* satfinite cvt.e4m3 via PTX. */\n"
"        unsigned int u;\n"
"        asm(\"{\\n\\t\"\n"
"            \".reg .b16 h;\\n\\t\"\n"
"            \"cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, %1;\\n\\t\"\n"
"            \"cvt.u32.u16 %0, h;\\n\\t\"\n"
"            \"}\\n\"\n"
"            : \"=r\"(u) : \"f\"(f));\n"
"        bits = (unsigned char)(u & 0xFF);\n"
"    }\n"
"    out[(long)c * n_in_pad + k] = bits;\n"
"}\n"
"\n"
/* Tiled im2col for VAE conv2d->GEMM path. Input [ci, H, W] row-major,
 * output [chunk_n_tok, n_in_pad] row-major where n_in_pad = ceil(ci*kh*kw
 * / 32) * 32. The `tok_offset` arg lets the caller process a sub-range of
 * H*W tokens at a time so the im2col buffer stays bounded for very large
 * spatial dimensions (e.g. 1024x1024 with 96 ci would be 3.5 GB in one
 * shot; tiled in 64K-token chunks it peaks at ~220 MB). */
"__global__ void vae_im2col_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp,\n"
"    int ci, int h, int w, int kh, int kw, int pad_replicate,\n"
"    int n_in_pad, int tok_offset, int chunk_n_tok) {\n"
"    int tok_local = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    int k         = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (tok_local >= chunk_n_tok || k >= n_in_pad) return;\n"
"    int n_in = ci * kh * kw;\n"
"    if (k >= n_in) { out[(long)tok_local * n_in_pad + k] = 0.0f; return; }\n"
"    int ph = (kh - 1) / 2, pw = (kw - 1) / 2;\n"
"    int tok_global = tok_offset + tok_local;\n"
"    int oy = tok_global / w, ox = tok_global - oy * w;\n"
"    int ic = k / (kh * kw);\n"
"    int rem = k - ic * (kh * kw);\n"
"    int fy = rem / kw, fx = rem - fy * kw;\n"
"    int iy = oy + fy - ph, ix = ox + fx - pw;\n"
"    float v = 0.0f;\n"
"    if (pad_replicate) {\n"
"        if (iy < 0) iy = 0; else if (iy >= h) iy = h - 1;\n"
"        if (ix < 0) ix = 0; else if (ix >= w) ix = w - 1;\n"
"        v = inp[(long)ic * h * w + (long)iy * w + ix];\n"
"    } else {\n"
"        if (iy >= 0 && iy < h && ix >= 0 && ix < w)\n"
"            v = inp[(long)ic * h * w + (long)iy * w + ix];\n"
"    }\n"
"    out[(long)tok_local * n_in_pad + k] = v;\n"
"}\n"
"\n"
/* Crop + transpose output of vae_op_conv2d_mma's chunked GEMM: the GEMM
 * writes [chunk_n_tok, pad_co] row-major; the VAE wants [co, n_tok] row-
 * major. Each thread reads one (c, tok_local) and stores to
 * out[c * n_tok_full + (tok_offset + tok_local)] so chunks land in the
 * correct global spatial slot. */
"__global__ void vae_crop_transpose_add_bias_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, const float *__restrict__ bias,\n"
"    int co, int pad_co, int n_tok_full, int tok_offset, int chunk_n_tok) {\n"
"    int tok_local = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int c         = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (tok_local >= chunk_n_tok || c >= co) return;\n"
"    float v = inp[(long)tok_local * pad_co + c];\n"
"    if (bias) v += bias[c];\n"
"    out[(long)c * n_tok_full + tok_offset + tok_local] = v;\n"
"}\n"
"\n"
/* Transpose [C, spatial] -> [spatial, C] (F32). Used before VAE middle
 * self-attention so the attention kernel can do coalesced K/V row loads. */
"__global__ void vae_transpose_chw_to_sc_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, int C, int spatial) {\n"
"    int s = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int c = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (s >= spatial || c >= C) return;\n"
"    out[(long)s * C + c] = inp[(long)c * spatial + s];\n"
"}\n"
"\n"
/* Transpose back [spatial, C] -> [C, spatial] after the attention output. */
"__global__ void vae_transpose_sc_to_chw_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, int C, int spatial) {\n"
"    int s = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int c = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (s >= spatial || c >= C) return;\n"
"    out[(long)c * spatial + s] = inp[(long)s * C + c];\n"
"}\n"
"\n"
/* Single-head F32 self-attention for VAE middle block.
 * Q, K, V are each [n_tok, c] row-major (spatial-major). One CTA per query
 * row, 1 warp (32 threads) per CTA. Online softmax in registers, Q cached in
 * smem once at start, output accumulator in smem. K and V are streamed
 * one token at a time (row-contiguous reads -> fully coalesced). */
"__global__ void vae_attn_sc_f32(float *__restrict__ out,\n"
"    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,\n"
"    int n_tok, int c, float scale) {\n"
"    int q = blockIdx.x;\n"
"    if (q >= n_tok) return;\n"
"    int tid = threadIdx.x;  /* 0..31 */\n"
"    extern __shared__ float va_smem[];\n"
"    float *smQ = va_smem;           /* [c] */\n"
"    float *smO = va_smem + c;       /* [c] */\n"
"    for (int d = tid; d < c; d += 32) {\n"
"        smQ[d] = Q[(long)q * c + d];\n"
"        smO[d] = 0.0f;\n"
"    }\n"
"    __syncwarp();\n"
"\n"
"    float m_state = -1e30f;\n"
"    float l_state = 0.0f;\n"
"    for (int j = 0; j < n_tok; j++) {\n"
"        const float *kp = K + (long)j * c;\n"
"        float dot = 0.0f;\n"
"        for (int d = tid; d < c; d += 32) dot += smQ[d] * kp[d];\n"
"        /* warp reduce */\n"
"        for (int off = 16; off > 0; off >>= 1)\n"
"            dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);\n"
"        dot *= scale;\n"
"        float new_m = fmaxf(m_state, dot);\n"
"        float alpha = expf(m_state - new_m);\n"
"        float beta  = expf(dot - new_m);\n"
"        const float *vp = V + (long)j * c;\n"
"        for (int d = tid; d < c; d += 32) {\n"
"            smO[d] = smO[d] * alpha + beta * vp[d];\n"
"        }\n"
"        l_state = l_state * alpha + beta;\n"
"        m_state = new_m;\n"
"    }\n"
"    float inv_l = 1.0f / l_state;\n"
"    for (int d = tid; d < c; d += 32) {\n"
"        out[(long)q * c + d] = smO[d] * inv_l;\n"
"    }\n"
"}\n"
"\n"
/* Conv2D for VAE: replicate or zero padding, any spatial size.
 * One thread per output element -- not the fastest shape, but all VAE conv
 * layers combined run in ~1.5s at 512x512 so further tiling has diminishing
 * returns (the middle attention used to dwarf everything before it moved
 * to the GPU). Grid: (ceil(co*oh*ow/256)), Block: (256) */
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
/* ---- cuBLAS-LT FP8 helpers ----
 * compute_x_scale_from_max: takes a single device float (max(|X|)) and
 * writes x_scale = max/448 (clamped to >=1e-12) into d_x_scale.
 * quant_f32_to_fp8_e4m3: F32 X -> FP8 e4m3 X with that scale (X/scale,
 * round-to-nearest-even, satfinite via cvt.rn.satfinite.e4m3x2.f32). */
"__global__ void compute_x_scale_from_max(float *d_x_scale,\n"
"                                          const float *d_max_abs) {\n"
"    if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
"        float m = d_max_abs[0];\n"
"        float s = m * (1.0f / 448.0f);\n"
"        if (s < 1e-12f) s = 1e-12f;\n"
"        d_x_scale[0] = s;\n"
"    }\n"
"}\n"
"\n"
"__global__ void quant_f32_to_fp8_e4m3(unsigned char *Y, const float *X,\n"
"                                       const float *d_x_scale, int n) {\n"
"    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;\n"
"    if (i >= n) return;\n"
"    float inv_s = 1.0f / d_x_scale[0];\n"
"    float a = X[i] * inv_s;\n"
"    float b = (i + 1 < n) ? X[i + 1] * inv_s : 0.0f;\n"
"    unsigned short packed;\n"
"    asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\" : \"=h\"(packed) : \"f\"(b), \"f\"(a));\n"
"    Y[i] = (unsigned char)(packed & 0xff);\n"
"    if (i + 1 < n) Y[i + 1] = (unsigned char)((packed >> 8) & 0xff);\n"
"}\n"
"\n"
/* bf16_to_f32_add_bias: read BF16 Y_in, write F32 Y_out + optional bias.
 * Y_in: row-major [n_tok, n_out] bf16 (2B). bias: F32 [n_out] or NULL. */
"__global__ void bf16_to_f32_add_bias(float *Y, const unsigned short *Y_bf16,\n"
"                                      const float *bias, int n_out, int n_tok) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_out * n_tok;\n"
"    if (i >= total) return;\n"
"    unsigned int bits = ((unsigned int)Y_bf16[i]) << 16;\n"
"    float v;\n"
"    asm(\"mov.b32 %0, %1;\" : \"=f\"(v) : \"r\"(bits));\n"
"    if (bias) v += bias[i % n_out];\n"
"    Y[i] = v;\n"
"}\n"
"\n"
/* Fused MLP epilogue: BF16 → F32 + bias + GELU(tanh approx) in a single pass.
 * Replaces back-to-back bf16_to_f32_add_bias + gelu_f32 (saves one full
 * read-modify-write over the hidden buffer). */
"__global__ void bf16_add_bias_gelu_f32(float *Y, const unsigned short *Y_bf16,\n"
"                                        const float *bias, int n_out, int n_tok) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_out * n_tok;\n"
"    if (i >= total) return;\n"
"    unsigned int bits = ((unsigned int)Y_bf16[i]) << 16;\n"
"    float v;\n"
"    asm(\"mov.b32 %0, %1;\" : \"=f\"(v) : \"r\"(bits));\n"
"    if (bias) v += bias[i % n_out];\n"
"    /* gelu (tanh approx) */\n"
"    float g = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v*v*v)));\n"
"    Y[i] = g;\n"
"}\n"
"\n"
/* In-place BF16 fused epilogue: Y_bf16[i] = bf16(gelu(bf16_to_f32(Y_bf16[i]) + bias[i%n_out])).
 * Also computes per-tensor max(|.|) into d_max_out via warp+atomic reduction.
 * Used by op_mlp_fp8 to keep MLP hidden state in BF16 instead of F32 — halves
 * memory traffic for the 4096×12288 hidden tensor. */
"__global__ void bf16_add_bias_gelu_inplace_max(unsigned short *Y_bf16,\n"
"                                                const float *bias,\n"
"                                                float *d_max_out,\n"
"                                                int n_out, int n_tok) {\n"
"    int total = n_out * n_tok;\n"
"    float local_max = 0.0f;\n"
"    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int stride = gridDim.x * blockDim.x;\n"
"    for (int i = tid; i < total; i += stride) {\n"
"        unsigned int bits = ((unsigned int)Y_bf16[i]) << 16;\n"
"        float v;\n"
"        asm(\"mov.b32 %0, %1;\" : \"=f\"(v) : \"r\"(bits));\n"
"        if (bias) v += bias[i % n_out];\n"
"        float g = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v*v*v)));\n"
"        /* round to bf16 (truncate; matches mma output rounding) */\n"
"        unsigned int gb;\n"
"        asm(\"mov.b32 %0, %1;\" : \"=r\"(gb) : \"f\"(g));\n"
"        Y_bf16[i] = (unsigned short)((gb + 0x8000) >> 16);  /* RNE-ish */\n"
"        float ag = fabsf(g);\n"
"        if (ag > local_max) local_max = ag;\n"
"    }\n"
"    /* warp reduce */\n"
"    for (int off = 16; off > 0; off >>= 1) {\n"
"        float other = __shfl_xor_sync(0xffffffff, local_max, off);\n"
"        if (other > local_max) local_max = other;\n"
"    }\n"
"    if ((threadIdx.x & 31) == 0) {\n"
"        /* atomic max on float via int trick (positive only) */\n"
"        unsigned int *p = (unsigned int *)d_max_out;\n"
"        unsigned int v;\n"
"        asm(\"mov.b32 %0, %1;\" : \"=r\"(v) : \"f\"(local_max));\n"
"        atomicMax(p, v);\n"
"    }\n"
"}\n"
"\n"
/* Quantize BF16 → FP8 e4m3 using a per-tensor scale (1 elem / thread; uses
 * cvt.rn.satfinite.e4m3x2 like the F32 path but reads BF16 input). */
"__global__ void quant_bf16_to_fp8_e4m3(unsigned char *Y, const unsigned short *X,\n"
"                                        const float *d_scale, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int idx = i * 2;\n"
"    if (idx >= n) return;\n"
"    float inv = 1.0f / (*d_scale);\n"
"    unsigned int b0 = ((unsigned int)X[idx]) << 16;\n"
"    float a;\n"
"    asm(\"mov.b32 %0, %1;\" : \"=f\"(a) : \"r\"(b0));\n"
"    a *= inv;\n"
"    float b = 0.0f;\n"
"    if (idx + 1 < n) {\n"
"        unsigned int b1 = ((unsigned int)X[idx+1]) << 16;\n"
"        asm(\"mov.b32 %0, %1;\" : \"=f\"(b) : \"r\"(b1));\n"
"        b *= inv;\n"
"    }\n"
"    unsigned short packed;\n"
"    asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\" : \"=h\"(packed) : \"f\"(b), \"f\"(a));\n"
"    Y[idx] = (unsigned char)(packed & 0xff);\n"
"    if (idx + 1 < n) Y[idx+1] = (unsigned char)((packed >> 8) & 0xff);\n"
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
    CUfunction gemm_fp8_mma_pipe;  /* 2-stage cp.async pipelined W-in-smem variant (sm_80+) */
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
    int use_fp8_pipe;  /* 1 to use cp.async pipelined FP8 MMA GEMM (env QIMG_FP8_PIPE=1) */
    int use_bf16_mma;  /* 1 to use BF16 MMA GEMM with FP8→BF16 dequant (env QIMG_BF16_MMA=1) */
    CUfunction gemm_bf16_mma_pipe;
    /* Per-row FP8 MMA pipe — eliminates per-tensor outlier crush.
     * Drops pixel_mean from 1.81 (per-tensor) to ~1.0 (near gold) at the same
     * speed as the per-tensor FP8 pipe. Env: QIMG_FP8_PIPE_PERROW=1. */
    int use_fp8_pipe_perrow;
    CUfunction gemm_fp8_pipe_perrow;
    CUfunction gemm_fp8_pipe_perrow_mt4;  /* MTILE=4 variant for n_tok % 64 == 0 */
    CUfunction gemm_fp8_pipe_perrow_mt4_concat3;  /* fused QKV: 1 launch + shared row_max */
    CUfunction reduce_max_abs_per_row;
    CUdeviceptr d_row_max_buf;     /* [max_n_tok] f32, lazily allocated */
    size_t      row_max_buf_n;
    /* cuBLAS-LT FP8 path (env QIMG_CUBLASLT_FP8=1, default 1 if loaded). */
    int use_cublaslt_fp8;
    struct cublasew_context *cublaslt_ctx;
    CUfunction quant_f32_to_fp8_e4m3;
    CUfunction compute_x_scale_from_max;
    CUfunction bf16_to_f32_add_bias;
    CUfunction bf16_add_bias_gelu_f32;  /* fused MLP fc1 epilogue */
    CUfunction bf16_add_bias_gelu_inplace_max;  /* in-place BF16 MLP epilogue + max */
    CUfunction quant_bf16_to_fp8_e4m3;          /* BF16 → FP8 quant */
    int mlp_bf16_hidden;                        /* env QIMG_MLP_BF16_HIDDEN, default ON */
    CUdeviceptr d_x_fp8_scratch;   /* lazily grown to max_n_tok * max_n_in bytes */
    size_t      x_fp8_scratch_n;
    CUdeviceptr d_y_bf16_scratch;  /* lazily grown to max_n_tok * max_n_out * 2 B */
    size_t      y_bf16_scratch_n;
    CUdeviceptr d_x_scale_f32;     /* [1] float */
    CUdeviceptr d_w_scale_one;     /* [1] float, holds 1.0f */
    int use_fp8_attn;  /* 1 to use mma.sync FP8 flash attention (env QIMG_FP8_ATTN=1) */
    /* Lazy FP8 attention workspace (re-used across blocks; sized to max n_tok*dim) */
    CUdeviceptr d_q_fp8, d_k_fp8, d_v_fp8;  /* [n_tok*dim] uint8 e4m3 */
    CUdeviceptr d_qkv_scales;               /* [3] float (sQ, sK, sV) */
    CUdeviceptr d_qkv_max;                  /* [1] float scratch */
    size_t      fp8_attn_buf_n;             /* current allocation size in bytes */
    /* BF16 flash attention path — eliminates FP8 quantization error in attention.
     * Q/K/V are bulk-cast from F32 to BF16 (uint16_t) before the kernel runs. */
    int use_bf16_attn;            /* 1 to use mma.sync BF16 flash attention (env QIMG_BF16_ATTN=1) */
    CUfunction flash_attn_bf16;
    CUfunction cast_f32_to_bf16;  /* bulk F32→BF16 cast */
    CUdeviceptr d_q_bf16, d_k_bf16, d_v_bf16; /* [n_tok*dim] uint16 bf16 */
    size_t      bf16_attn_buf_n;
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
    CUfunction vae_f32_to_fp8_padded;
    CUfunction vae_im2col;
    CUfunction vae_crop_transpose_add_bias;
    CUfunction vae_transpose_chw_to_sc;
    CUfunction vae_transpose_sc_to_chw;
    CUfunction vae_attn_sc;
    CUfunction vae_rmsnorm;
    CUfunction vae_silu;
    CUfunction nn_upsample2x;
    CUfunction rmsnorm_weighted;

    /* DiT config */
    int dit_dim, dit_n_heads, dit_head_dim, dit_n_blocks;
    int dit_in_ch, dit_txt_dim, dit_mlp_h;

    /* Safetensors context (mmap'd) */
    void *dit_st;
    int dit_st_pinned;  /* 1 if cuMemHostRegister succeeded on the mmap region */
    /* Per-tensor pinned host buffers for streamed blocks.
     * dit_pinned_data[idx] = pinned ptr (NULL = use mmap source).
     * Backed by per-block pinned pools owned by dit_pinned_pools[]. */
    void **dit_pinned_data;
    int dit_pinned_n_tensors;
    void **dit_pinned_pools;     /* size = dit_n_blocks; one per streamed block */
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

    /* Double-buffered block scratch + copy stream for overlapping the
     * PCIe block load with the previous block's compute. Allocated lazily
     * together with scratch_block. Enabled when n_preloaded < n_blocks. */
    qimg_block_gpu scratch_block_b;  /* second slot (first is scratch_block) */
    int scratch_block_b_ready;
    CUstream    copy_stream;
    CUevent     slot_ready[2];  /* signal: slot has been fully HtoD'd */
    CUevent     slot_free[2];   /* signal: compute has released the slot */
    int         slot_initial_loaded;  /* 0 = nothing prefetched; 1+ = slots primed */

    /* Parallel CFG streams: when QIMG_CFG_STREAMS=1, dit_step_cfg runs the
     * cond pass on r->stream and the uncond pass on r->uncond_stream
     * concurrently within each block. Their computations can overlap on the
     * GPU because ops on different streams without explicit dependencies
     * run concurrently. */
    CUstream    uncond_stream;
    CUevent     block_barrier_cond;  /* cond-done barrier for next iter sync */
    CUevent     block_barrier_unc;   /* uncond-done barrier */
    CUevent     mod_ready;           /* modulation computed (both streams wait on this) */

    /* Persistent GPU: global weights (~50MB) */
    CUdeviceptr d_img_in_w, d_img_in_b;
    CUdeviceptr d_txt_in_w, d_txt_in_b;
    CUdeviceptr d_txt_norm_w;
    CUdeviceptr d_t_fc1_w, d_t_fc1_b;
    CUdeviceptr d_t_fc2_w, d_t_fc2_b;
    CUdeviceptr d_norm_out_w, d_norm_out_b;
    CUdeviceptr d_proj_out_w, d_proj_out_b;

    /* Per-stage block profiling (env QIMG_PROFILE_BLOCK=1).
     * When enabled, qimg_forward_block synchronizes between phases and
     * accumulates wall-clock time into these buckets; printed at end of
     * cuda_qimg_dit_step. */
    int profile_block;
    double prof_qkv_ms;       /* adaLN1 + QKV gemm (img+txt) */
    double prof_qknorm_ms;    /* QK rmsnorm + RoPE */
    double prof_attn_ms;      /* op_attn */
    double prof_attnout_ms;   /* attn out proj + gated_add */
    double prof_imgmlp_ms;    /* adaLN2 img + img MLP fc1+gelu+fc2 + gated_add */
    double prof_txtmlp_ms;    /* adaLN2 txt + txt MLP fc1+gelu+fc2 + gated_add */
    int prof_block_count;
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

/* ---- FP8 E4M3 → BF16 LUT for the BF16 MMA GEMM ---- */
static uint16_t qimg_fp8_to_bf16_lut[256];
static int qimg_fp8_to_bf16_lut_init = 0;

static uint16_t qimg_f32_to_bf16_bits(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    if (((b >> 23) & 0xFF) == 0xFF && (b & 0x7FFFFF)) return 0x7FC0;  /* NaN */
    uint32_t r = 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)((b + r) >> 16);
}

static void qimg_init_fp8_to_bf16_lut(void) {
    if (qimg_fp8_to_bf16_lut_init) return;
    for (int i = 0; i < 256; i++)
        qimg_fp8_to_bf16_lut[i] = qimg_f32_to_bf16_bits(fp8_e4m3_to_f32((uint8_t)i));
    qimg_fp8_to_bf16_lut_init = 1;
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
/* Forward decl: pinned host source override for streamed block tensors. */
static const void *qimg_tensor_host_src(struct cuda_qimg_runner *r, st_context *st, int idx);

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

/* Pinned-aware variant — used by streaming hot path. Passes runner so the
 * helper can substitute a pinned host source when one was pre-staged. */
static int qimg_st_upload_fp8_raw_async_r(struct cuda_qimg_runner *r,
                                           st_context *st, const char *name,
                                           CUdeviceptr dst, size_t dst_nbytes,
                                           CUstream s) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return -1;
    size_t nbytes = safetensors_nbytes(st, idx);
    if (nbytes != dst_nbytes) return -1;
    cuMemcpyHtoDAsync(dst, qimg_tensor_host_src(r, st, idx), nbytes, s);
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

/* Pinned-aware variant of qimg_st_upload_f32_async. Only the F32 fast-path
 * benefits from pinned (the conversion path goes through a temp buffer). */
static int qimg_st_upload_f32_async_r(struct cuda_qimg_runner *r,
                                      st_context *st, const char *name,
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
    if (strcmp(dtype, "F32") == 0) {
        cuMemcpyHtoDAsync(dst, qimg_tensor_host_src(r, st, idx),
                          n * sizeof(float), s);
        return 0;
    }
    /* Fall back to mmap-based path with conversion. */
    return qimg_st_upload_f32_async(st, name, dst, dst_nelem, s);
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

/* Same thing, for the second slot in the double-buffer pool. Also creates
 * the copy stream and slot events on first successful call. Returns 0 OK. */
static int qimg_alloc_scratch_block_b(cuda_qimg_runner *r) {
    if (r->scratch_block_b_ready) return 0;
    if (!r->scratch_block_ready) {
        if (qimg_alloc_scratch_block(r) != 0) return -1;
    }
    st_context *st = (st_context *)r->dit_st;
    if (!st || r->dit_n_blocks <= 0) return -1;
    int i0 = 0;
    char name[256];
    qimg_block_gpu *b = &r->scratch_block_b;
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
    /* Create copy stream + slot events. Use cuEventDisableTiming for speed. */
    if (!r->copy_stream) {
        if (cuStreamCreate(&r->copy_stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
            qimg_free_block(b);
            return -1;
        }
    }
    for (int i = 0; i < 2; i++) {
        if (!r->slot_ready[i])
            cuEventCreate(&r->slot_ready[i], CU_EVENT_DISABLE_TIMING);
        if (!r->slot_free[i])
            cuEventCreate(&r->slot_free[i], CU_EVENT_DISABLE_TIMING);
    }
    r->scratch_block_b_ready = 1;
    return 0;
}

/* Copy block i into a pre-allocated scratch slot. Caller gets a
 * qimg_block_gpu struct whose pointers alias the slot.
 *
 * Uses async H2D copies on the given stream. Stream ordering on the same
 * stream serializes with prior compute; to reuse a slot across streams,
 * the caller must use events. */
static int qimg_load_block_into_slot_ex(cuda_qimg_runner *r, int block_idx,
                                         qimg_block_gpu *slot,
                                         qimg_block_gpu *out_b, CUstream s) {
    st_context *st = (st_context *)r->dit_st;
    *out_b = *slot;  /* copy pointers (slot buffers stay owned by runner) */
    qimg_block_gpu *b = out_b;
    char name[256];
    int ok = 1;
#define UPLOAD_FP8(field, suffix, _bytes) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        if (qimg_st_upload_fp8_raw_async_r(r, st, name, b->field, _bytes, s) != 0) ok = 0; \
    } } while(0)
#define UPLOAD_F32(field, suffix, _nelem) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        if (qimg_st_upload_f32_async_r(r, st, name, b->field, _nelem, s) != 0) ok = 0; \
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

/* Returns pinned host source for the tensor if pre-staged, else mmap source. */
static const void *qimg_tensor_host_src(struct cuda_qimg_runner *r,
                                         st_context *st, int idx) {
    if (r && r->dit_pinned_data && idx >= 0 && idx < r->dit_pinned_n_tensors &&
        r->dit_pinned_data[idx]) {
        return r->dit_pinned_data[idx];
    }
    return safetensors_data(st, idx);
}

/* Stage one block's tensors into a pinned host pool.
 * Allocates ~324 MB pinned per call. Returns 0 on success, -1 on alloc fail. */
static int qimg_stage_block_pinned(cuda_qimg_runner *r, int block_idx) {
    st_context *st = (st_context *)r->dit_st;
    /* List of (suffix, expected_dtype_is_fp8) for all 30 tensors per block.
     * Order matches qimg_load_block_into_slot_ex. */
    static const char *suffixes[] = {
        "attn.to_q.weight", "attn.to_q.bias",
        "attn.to_k.weight", "attn.to_k.bias",
        "attn.to_v.weight", "attn.to_v.bias",
        "attn.to_out.0.weight", "attn.to_out.0.bias",
        "attn.add_q_proj.weight", "attn.add_q_proj.bias",
        "attn.add_k_proj.weight", "attn.add_k_proj.bias",
        "attn.add_v_proj.weight", "attn.add_v_proj.bias",
        "attn.to_add_out.weight", "attn.to_add_out.bias",
        "attn.norm_q.weight", "attn.norm_k.weight",
        "attn.norm_added_q.weight", "attn.norm_added_k.weight",
        "img_mod.1.weight", "img_mod.1.bias",
        "img_mlp.net.0.proj.weight", "img_mlp.net.0.proj.bias",
        "img_mlp.net.2.weight", "img_mlp.net.2.bias",
        "txt_mod.1.weight", "txt_mod.1.bias",
        "txt_mlp.net.0.proj.weight", "txt_mlp.net.0.proj.bias",
        "txt_mlp.net.2.weight", "txt_mlp.net.2.bias",
        NULL
    };
    /* First pass: total bytes (round up each to 256 B for cuMemAllocHost
     * alignment friendliness — minor padding overhead). */
    size_t total = 0;
    char name[256];
    int idxs[64]; size_t sizes[64], offs[64];
    int n = 0;
    for (int i = 0; suffixes[i]; i++) {
        snprintf(name, sizeof(name), "transformer_blocks.%d.%s", block_idx, suffixes[i]);
        int idx = safetensors_find(st, name);
        if (idx < 0) continue;
        idxs[n] = idx;
        sizes[n] = safetensors_nbytes(st, idx);
        offs[n] = total;
        total += (sizes[n] + 255) & ~(size_t)255;
        n++;
    }
    /* Allocate pinned pool. */
    void *pool = NULL;
    if (cuMemAllocHost(&pool, total) != CUDA_SUCCESS || !pool) return -1;
    r->dit_pinned_pools[block_idx] = pool;
    /* Second pass: memcpy + register pinned ptr in dit_pinned_data[]. */
    char *base = (char *)pool;
    for (int j = 0; j < n; j++) {
        memcpy(base + offs[j], safetensors_data(st, idxs[j]), sizes[j]);
        r->dit_pinned_data[idxs[j]] = base + offs[j];
    }
    return 0;
}

/* Single-slot compat wrapper: copies into r->scratch_block on r->stream. */
static int qimg_load_block_into_slot(cuda_qimg_runner *r, int block_idx,
                                     qimg_block_gpu *out_b) {
    if (!r->scratch_block_ready) {
        if (qimg_alloc_scratch_block(r) != 0) return -1;
    }
    return qimg_load_block_into_slot_ex(r, block_idx, &r->scratch_block,
                                         out_b, r->stream);
}


/* ---- Op launch helpers ---- */
static void op_bf16_trunc(cuda_qimg_runner *r, CUdeviceptr x, int n);  /* forward decl */

static void op_gemm(cuda_qimg_runner *r, CUdeviceptr Y, CUdeviceptr W,
                    CUdeviceptr X, CUdeviceptr bias,
                    int n_out, int n_in, int n_tok) {

    /* cuBLAS-LT FP8 e4m3 path — highest priority. Quantizes X to FP8 with a
     * per-tensor scale = max(|X|)/448, calls cublasLtMatmul (BF16 output),
     * then expands BF16 -> F32 with optional bias add. */
    if (r->use_cublaslt_fp8 && n_tok >= 8 && n_in >= 32 && n_out >= 16 &&
        !r->use_old_gemm) {
        size_t need_x = (size_t)n_tok * n_in;
        size_t need_y = (size_t)n_tok * n_out * sizeof(uint16_t);
        if (need_x > r->x_fp8_scratch_n) {
            if (r->d_x_fp8_scratch) cuMemFree(r->d_x_fp8_scratch);
            r->d_x_fp8_scratch = 0;
            if (cuMemAlloc(&r->d_x_fp8_scratch, need_x) == CUDA_SUCCESS) {
                r->x_fp8_scratch_n = need_x;
            } else { r->x_fp8_scratch_n = 0; }
        }
        if (need_y > r->y_bf16_scratch_n) {
            if (r->d_y_bf16_scratch) cuMemFree(r->d_y_bf16_scratch);
            r->d_y_bf16_scratch = 0;
            if (cuMemAlloc(&r->d_y_bf16_scratch, need_y) == CUDA_SUCCESS) {
                r->y_bf16_scratch_n = need_y;
            } else { r->y_bf16_scratch_n = 0; }
        }
        if (r->d_x_fp8_scratch && r->d_y_bf16_scratch) {
            /* 1. reduce_max_abs(X) -> d_qkv_max */
            if (!r->d_qkv_max) cuMemAlloc(&r->d_qkv_max, sizeof(float));
            cuMemsetD32Async(r->d_qkv_max, 0, 1, r->stream);
            int n_x = n_tok * n_in;
            void *r_args[] = {&r->d_qkv_max, &X, &n_x};
            unsigned rb = (unsigned)((n_x + 255) / 256);
            cuLaunchKernel(r->reduce_max_abs, rb, 1, 1, 256, 1, 1,
                           0, r->stream, r_args, NULL);
            /* 2. compute x_scale = max/448 (1 thread) */
            void *s_args[] = {&r->d_x_scale_f32, &r->d_qkv_max};
            cuLaunchKernel(r->compute_x_scale_from_max, 1, 1, 1, 1, 1, 1,
                           0, r->stream, s_args, NULL);
            /* 3. quant X -> FP8 using x_scale; 2 elems per thread */
            void *q_args[] = {&r->d_x_fp8_scratch, &X, &r->d_x_scale_f32, &n_x};
            unsigned qb = (unsigned)(((n_x + 1) / 2 + 255) / 256);
            cuLaunchKernel(r->quant_f32_to_fp8_e4m3, qb, 1, 1, 256, 1, 1,
                           0, r->stream, q_args, NULL);
            /* 4. cuBLAS-LT FP8 matmul -> Y_bf16 scratch */
            int rc = cublasew_gemm_fp8_e4m3_bf16out_rowmajor_nt(
                r->cublaslt_ctx,
                r->d_y_bf16_scratch,
                W, r->d_x_fp8_scratch,
                r->d_w_scale_one, r->d_x_scale_f32,
                0,  /* no LT bias epilogue; do it in expand */
                n_tok, n_out, n_in);
            if (rc == 0) {
                /* 5. expand BF16 -> F32 (+ optional bias) */
                void *e_args[] = {&Y, &r->d_y_bf16_scratch, &bias, &n_out, &n_tok};
                int total = n_out * n_tok;
                unsigned eb = (unsigned)((total + 255) / 256);
                cuLaunchKernel(r->bf16_to_f32_add_bias, eb, 1, 1, 256, 1, 1,
                               0, r->stream, e_args, NULL);
                return;
            }
            /* LT matmul failed at runtime — fall through to perrow path */
        }
    }

    /* Per-row FP8 MMA pipelined GEMM — top priority when enabled. Same speed
     * as the per-tensor FP8 pipe but drops pixel_mean blur from ~1.81 to ~1.0
     * by computing one X scale per output row instead of one scale for the
     * whole tensor. Cost: one extra reduce_max_abs_per_row launch per call. */
    if (r->use_fp8_pipe_perrow && r->gemm_fp8_pipe_perrow && r->reduce_max_abs_per_row &&
        n_tok >= 16 && (n_out % 256) == 0 && (n_in % 32) == 0 && !r->use_old_gemm) {
        /* Allocate / grow per-row max buffer as needed. */
        size_t need = (size_t)n_tok * sizeof(float);
        if (need > r->row_max_buf_n) {
            if (r->d_row_max_buf) { cuMemFree(r->d_row_max_buf); r->d_row_max_buf = 0; }
            if (cuMemAlloc(&r->d_row_max_buf, need) == CUDA_SUCCESS) {
                r->row_max_buf_n = need;
            } else {
                r->d_row_max_buf = 0; r->row_max_buf_n = 0;
            }
        }
        if (r->d_row_max_buf) {
            /* Launch per-row reduce: one CTA per row, 256 threads. */
            void *rargs[] = {&r->d_row_max_buf, &X, &n_tok, &n_in};
            cuLaunchKernel(r->reduce_max_abs_per_row, (unsigned)n_tok, 1, 1,
                           256, 1, 1, 0, r->stream, rargs, NULL);
            float w_scale = 1.0f;
            void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &w_scale, &r->d_row_max_buf};
            unsigned gx = (unsigned)((n_out + 255) / 256);
            /* MTILE=4 variant: 64 rows/CTA. Used when n_tok is a multiple of 64,
             * which holds for img stream at >=512 (n_img=1024+) and qkv shapes. */
            if (!getenv("QIMG_DISABLE_MT4") && r->gemm_fp8_pipe_perrow_mt4 && n_tok >= 64 && (n_tok % 64) == 0) {
                unsigned gy4 = (unsigned)((n_tok + 63) / 64);
                /* Pad grid to multiples of 4 in both dims for v7-style 4x4
                 * CTA panel swizzle inside the kernel (excess CTAs early-return). */
                unsigned gx4 = (gx + 3u) & ~3u;
                gy4 = (gy4 + 3u) & ~3u;
                /* smem: 2048 (smX 64x32) + 2*8192 (W) + 512 (64 inv + 64 fwd) = 18944 B */
                size_t smem_mt4 = 2048 + 8192 * 2 + 512;
                cuLaunchKernel(r->gemm_fp8_pipe_perrow_mt4, gx4, gy4, 1, 128, 1, 1,
                               smem_mt4, r->stream, args, NULL);
                return;
            }
            unsigned gy = (unsigned)((n_tok +  31) /  32);
            /* Pad to multiples of 4 for v7-style 4x4 CTA panel swizzle. */
            unsigned gx_pr = (gx + 3u) & ~3u;
            gy = (gy + 3u) & ~3u;
            /* smem: 1024 (smX FP8) + 2*8192 (W stages) + 256 (32 inv + 32 fwd scales) = 17664 B */
            size_t smem_pr = 1024 + 8192 * 2 + 256;
            cuLaunchKernel(r->gemm_fp8_pipe_perrow, gx_pr, gy, 1, 128, 1, 1,
                           smem_pr, r->stream, args, NULL);
            return;
        }
        /* OOM → fall through */
    }

    /* BF16 MMA pipelined GEMM — highest precision, slower. Reads FP8 weights
     * and decodes to BF16 inline at MMA time. Matches ComfyUI's BF16 reference. */
    if (r->use_bf16_mma && r->gemm_bf16_mma_pipe && n_tok >= 16 &&
        (n_out % 256) == 0 && (n_in % 32) == 0 && !r->use_old_gemm) {
        unsigned gx = (unsigned)((n_out + 255) / 256);
        unsigned gy = (unsigned)((n_tok +  31) /  32);
        /* Pad to multiples of 4 for v7-style 4x4 CTA panel swizzle. */
        gx = (gx + 3u) & ~3u;
        gy = (gy + 3u) & ~3u;
        /* smem: smX bf16 (2048) + smW_fp8 (8192) x 2 stages = 18432 B */
        size_t smem_bf16 = 2048 + 8192 * 2;
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        cuLaunchKernel(r->gemm_bf16_mma_pipe, gx, gy, 1, 128, 1, 1,
                       smem_bf16, r->stream, args, NULL);
        return;
    }

    if (r->use_fp8_mma && r->gemm_fp8_mma && n_tok >= 4 && !r->use_old_gemm) {
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
        /* Default kernel: MTILE=2 × NWARPS=4 × NTILE=8 × 8 = 256 N cols/CTA,
         * 32 M rows/CTA. BF16 truncation is fused into the writeback. */
        unsigned gx = (unsigned)((n_out + 255) / 256);
        unsigned gy = (unsigned)((n_tok +  31) /  32);
        size_t smem = (size_t)(16 * 2) * 32 * sizeof(float);  /* MTILE=2 */
        /* Pipelined variant: cp.async loads W into smem double-buffered so
         * the next k-tile's weight prefetch overlaps with the current tile's
         * MMAs. Requires divisible shapes (n_out%256==0, n_in%32==0) and
         * n_tok >= 16 — these hold for all qwen-image block linears. */
        if (r->use_fp8_pipe && r->gemm_fp8_mma_pipe && n_tok >= 16 &&
            (n_out % 256) == 0 && (n_in % 32) == 0) {
            size_t smem_pipe = 1024 + 8192 * 2;  /* FP8 X tile + 2 × W tile = 17408 B */
            /* Pad to multiples of 4 for v7-style 4x4 CTA panel swizzle. */
            unsigned gx_p = (gx + 3u) & ~3u;
            unsigned gy_p = (gy + 3u) & ~3u;
            cuLaunchKernel(r->gemm_fp8_mma_pipe, gx_p, gy_p, 1, 128, 1, 1,
                           smem_pipe, r->stream, args, NULL);
            return;
        }
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
        /* BF16 truncation is fused into the MMA kernel writeback (to_bf16 inline). */
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

/* Fused GEMM + bias + GELU. Equivalent to op_gemm(...) followed by op_gelu().
 * Saves one full read-modify-write pass over Y on the cuBLAS-LT FP8 fast path
 * by folding the gelu into the BF16→F32+bias expansion kernel. */
static void op_gemm_gelu(cuda_qimg_runner *r, CUdeviceptr Y, CUdeviceptr W,
                         CUdeviceptr X, CUdeviceptr bias,
                         int n_out, int n_in, int n_tok) {
    /* cuBLAS-LT FP8 path with fused bias+gelu epilogue. */
    if (r->use_cublaslt_fp8 && r->bf16_add_bias_gelu_f32 &&
        n_tok >= 8 && n_in >= 32 && n_out >= 16 && !r->use_old_gemm) {
        size_t need_x = (size_t)n_tok * n_in;
        size_t need_y = (size_t)n_tok * n_out * 2;
        if (need_x > r->x_fp8_scratch_n) {
            if (r->d_x_fp8_scratch) cuMemFree(r->d_x_fp8_scratch);
            r->d_x_fp8_scratch = 0;
            if (cuMemAlloc(&r->d_x_fp8_scratch, need_x) == CUDA_SUCCESS)
                r->x_fp8_scratch_n = need_x;
            else r->x_fp8_scratch_n = 0;
        }
        if (need_y > r->y_bf16_scratch_n) {
            if (r->d_y_bf16_scratch) cuMemFree(r->d_y_bf16_scratch);
            r->d_y_bf16_scratch = 0;
            if (cuMemAlloc(&r->d_y_bf16_scratch, need_y) == CUDA_SUCCESS)
                r->y_bf16_scratch_n = need_y;
            else r->y_bf16_scratch_n = 0;
        }
        if (r->d_x_fp8_scratch && r->d_y_bf16_scratch) {
            if (!r->d_qkv_max) cuMemAlloc(&r->d_qkv_max, sizeof(float));
            cuMemsetD32Async(r->d_qkv_max, 0, 1, r->stream);
            int n_x = n_tok * n_in;
            void *r_args[] = {&r->d_qkv_max, &X, &n_x};
            unsigned rb = (unsigned)((n_x + 255) / 256);
            cuLaunchKernel(r->reduce_max_abs, rb, 1, 1, 256, 1, 1,
                           0, r->stream, r_args, NULL);
            void *s_args[] = {&r->d_x_scale_f32, &r->d_qkv_max};
            cuLaunchKernel(r->compute_x_scale_from_max, 1, 1, 1, 1, 1, 1,
                           0, r->stream, s_args, NULL);
            void *q_args[] = {&r->d_x_fp8_scratch, &X, &r->d_x_scale_f32, &n_x};
            unsigned qb = (unsigned)(((n_x + 1) / 2 + 255) / 256);
            cuLaunchKernel(r->quant_f32_to_fp8_e4m3, qb, 1, 1, 256, 1, 1,
                           0, r->stream, q_args, NULL);
            int rc = cublasew_gemm_fp8_e4m3_bf16out_rowmajor_nt(
                r->cublaslt_ctx,
                r->d_y_bf16_scratch,
                W, r->d_x_fp8_scratch,
                r->d_w_scale_one, r->d_x_scale_f32,
                0, n_tok, n_out, n_in);
            if (rc == 0) {
                /* Fused bf16 -> f32 + bias + gelu (single pass). */
                void *e_args[] = {&Y, &r->d_y_bf16_scratch, &bias, &n_out, &n_tok};
                int total = n_out * n_tok;
                unsigned eb = (unsigned)((total + 255) / 256);
                cuLaunchKernel(r->bf16_add_bias_gelu_f32, eb, 1, 1, 256, 1, 1,
                               0, r->stream, e_args, NULL);
                return;
            }
        }
    }
    /* Fallback: separate GEMM + GELU (matches old behavior bit-exactly). */
    op_gemm(r, Y, W, X, bias, n_out, n_in, n_tok);
    op_gelu(r, Y, n_tok * n_out);
}

/* Fused two-GEMM MLP entirely on FP8/BF16: avoids materializing a F32 hidden.
 *   Y = fc2_W · gelu(fc1_W · X + fc1_b) + fc2_b
 * Returns 0 on success (LT path used end-to-end), -1 if any prerequisite is
 * missing — caller should fall back to op_gemm_gelu + op_gemm.
 *
 * Memory savings vs (op_gemm_gelu + op_gemm) for (n_tok=4096, mlp_h=12288):
 *   - skips writing 192 MB F32 hidden (gelu output)
 *   - skips reading 192 MB F32 hidden (fc2 X-quant)
 *   = ~384 MB saved per invocation. */
static int op_mlp_fp8(cuda_qimg_runner *r,
                      CUdeviceptr Y, CUdeviceptr X,
                      CUdeviceptr fc1_W, CUdeviceptr fc1_b,
                      CUdeviceptr fc2_W, CUdeviceptr fc2_b,
                      int dim, int mlp_h, int n_tok) {
    if (!r->mlp_bf16_hidden) return -1;
    if (!r->use_cublaslt_fp8 || !r->bf16_add_bias_gelu_inplace_max ||
        !r->quant_bf16_to_fp8_e4m3 || !r->bf16_to_f32_add_bias ||
        !r->reduce_max_abs || !r->compute_x_scale_from_max ||
        !r->quant_f32_to_fp8_e4m3 || r->use_old_gemm) return -1;
    if (n_tok < 1 || dim < 32 || mlp_h < 16) return -1;

    /* Scratch sizing: x_fp8 needs to fit max(n_tok*dim, n_tok*mlp_h);
     * y_bf16 needs n_tok*max(dim,mlp_h)*2B. Sized for fc1 first, grown for fc2. */
    size_t need_x = (size_t)n_tok * mlp_h;       /* fc2 input is largest */
    size_t need_y = (size_t)n_tok * mlp_h * 2;   /* fc1 output (BF16 hidden) */
    if (need_x > r->x_fp8_scratch_n) {
        if (r->d_x_fp8_scratch) cuMemFree(r->d_x_fp8_scratch);
        r->d_x_fp8_scratch = 0;
        if (cuMemAlloc(&r->d_x_fp8_scratch, need_x) != CUDA_SUCCESS) {
            r->x_fp8_scratch_n = 0; return -1;
        }
        r->x_fp8_scratch_n = need_x;
    }
    if (need_y > r->y_bf16_scratch_n) {
        if (r->d_y_bf16_scratch) cuMemFree(r->d_y_bf16_scratch);
        r->d_y_bf16_scratch = 0;
        if (cuMemAlloc(&r->d_y_bf16_scratch, need_y) != CUDA_SUCCESS) {
            r->y_bf16_scratch_n = 0; return -1;
        }
        r->y_bf16_scratch_n = need_y;
    }
    if (!r->d_qkv_max) {
        if (cuMemAlloc(&r->d_qkv_max, sizeof(float)) != CUDA_SUCCESS) return -1;
    }

    CUstream s = r->stream;

    /* --- Stage A: fc1 = LT FP8(X · fc1_W) → BF16 in d_y_bf16_scratch --- */
    /* A.1: per-tensor X→FP8 quant */
    cuMemsetD32Async(r->d_qkv_max, 0, 1, s);
    int n_x = n_tok * dim;
    {
        void *args[] = {&r->d_qkv_max, &X, &n_x};
        unsigned rb = (unsigned)((n_x + 255) / 256);
        cuLaunchKernel(r->reduce_max_abs, rb, 1, 1, 256, 1, 1, 0, s, args, NULL);
    }
    {
        void *args[] = {&r->d_x_scale_f32, &r->d_qkv_max};
        cuLaunchKernel(r->compute_x_scale_from_max, 1, 1, 1, 1, 1, 1, 0, s, args, NULL);
    }
    {
        void *args[] = {&r->d_x_fp8_scratch, &X, &r->d_x_scale_f32, &n_x};
        unsigned qb = (unsigned)(((n_x + 1) / 2 + 255) / 256);
        cuLaunchKernel(r->quant_f32_to_fp8_e4m3, qb, 1, 1, 256, 1, 1, 0, s, args, NULL);
    }
    /* A.2: LT fc1 — bias=0 (we add it in the in-place epilogue). */
    if (cublasew_gemm_fp8_e4m3_bf16out_rowmajor_nt(
            r->cublaslt_ctx, r->d_y_bf16_scratch,
            fc1_W, r->d_x_fp8_scratch,
            r->d_w_scale_one, r->d_x_scale_f32,
            0, n_tok, mlp_h, dim) != 0) {
        return -1;
    }

    /* --- Stage B: in-place +bias+gelu on BF16 hidden, accumulate per-tensor max --- */
    cuMemsetD32Async(r->d_qkv_max, 0, 1, s);
    {
        void *args[] = {&r->d_y_bf16_scratch, &fc1_b, &r->d_qkv_max, &mlp_h, &n_tok};
        /* persistent grid: 256 CTAs × 256 threads cover any size by stride loop */
        unsigned grid = 256;
        cuLaunchKernel(r->bf16_add_bias_gelu_inplace_max, grid, 1, 1,
                       256, 1, 1, 0, s, args, NULL);
    }
    /* compute fc2_x_scale = max/448 (reuse d_x_scale_f32) */
    {
        void *args[] = {&r->d_x_scale_f32, &r->d_qkv_max};
        cuLaunchKernel(r->compute_x_scale_from_max, 1, 1, 1, 1, 1, 1, 0, s, args, NULL);
    }

    /* --- Stage C: quant BF16 hidden → FP8 (fc2 input) --- */
    int n_h = n_tok * mlp_h;
    {
        void *args[] = {&r->d_x_fp8_scratch, &r->d_y_bf16_scratch, &r->d_x_scale_f32, &n_h};
        unsigned qb = (unsigned)(((n_h + 1) / 2 + 255) / 256);
        cuLaunchKernel(r->quant_bf16_to_fp8_e4m3, qb, 1, 1, 256, 1, 1, 0, s, args, NULL);
    }

    /* --- Stage D: LT fc2 — writes BACK to d_y_bf16_scratch (overwrites the
     *               consumed BF16 hidden, which is OK since stage C already
     *               read it). --- */
    if (cublasew_gemm_fp8_e4m3_bf16out_rowmajor_nt(
            r->cublaslt_ctx, r->d_y_bf16_scratch,
            fc2_W, r->d_x_fp8_scratch,
            r->d_w_scale_one, r->d_x_scale_f32,
            0, n_tok, dim, mlp_h) != 0) {
        return -1;
    }

    /* --- Stage E: expand BF16 → F32 + fc2_b → final Y --- */
    {
        int total = dim * n_tok;
        void *args[] = {&Y, &r->d_y_bf16_scratch, &fc2_b, &dim, &n_tok};
        unsigned eb = (unsigned)((total + 255) / 256);
        cuLaunchKernel(r->bf16_to_f32_add_bias, eb, 1, 1, 256, 1, 1, 0, s, args, NULL);
    }
    return 0;
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

/* ---- BF16 flash attention workspace ---- */

/* Lazily (re)allocate Q/K/V BF16 scratch buffers, sized in bytes (= n_tok*dim*2). */
static int ensure_bf16_attn_buf(cuda_qimg_runner *r, size_t n_bytes) {
    if (n_bytes <= r->bf16_attn_buf_n) return 0;
    if (r->d_q_bf16) { cuMemFree(r->d_q_bf16); r->d_q_bf16 = 0; }
    if (r->d_k_bf16) { cuMemFree(r->d_k_bf16); r->d_k_bf16 = 0; }
    if (r->d_v_bf16) { cuMemFree(r->d_v_bf16); r->d_v_bf16 = 0; }
    r->bf16_attn_buf_n = 0;
    if (cuMemAlloc(&r->d_q_bf16, n_bytes) != CUDA_SUCCESS) { r->d_q_bf16 = 0; return -1; }
    if (cuMemAlloc(&r->d_k_bf16, n_bytes) != CUDA_SUCCESS) { r->d_k_bf16 = 0; return -1; }
    if (cuMemAlloc(&r->d_v_bf16, n_bytes) != CUDA_SUCCESS) { r->d_v_bf16 = 0; return -1; }
    r->bf16_attn_buf_n = n_bytes;
    return 0;
}

/* Bulk F32 → BF16 cast launcher. n is the element count (not bytes). */
static void cast_buf_f32_to_bf16(cuda_qimg_runner *r, CUdeviceptr dst, CUdeviceptr src, int n) {
    void *args[] = {&src, &dst, &n};
    unsigned blocks = (unsigned)((n + 255) / 256);
    cuLaunchKernel(r->cast_f32_to_bf16, blocks, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
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

/* Fused QKV linear: Yq/Yk/Yv = X @ Wq/Wk/Wv^T + bq/bk/bv. Single launch with
 * shared row_max + L2 X reuse vs three op_gemm calls. Returns 0 on success;
 * caller falls back to 3x op_gemm on -1. Requires the perrow_mt4_concat3
 * kernel + n_tok>=64 + (n_tok%64)==0 + (dim%256)==0 + (n_in%32)==0. */
static int op_gemm_qkv_fused(cuda_qimg_runner *r,
                              CUdeviceptr Yq, CUdeviceptr Yk, CUdeviceptr Yv,
                              CUdeviceptr Wq, CUdeviceptr Wk, CUdeviceptr Wv,
                              CUdeviceptr X,
                              CUdeviceptr bq, CUdeviceptr bk, CUdeviceptr bv,
                              int dim, int n_in, int n_tok) {
    /* cuBLAS-LT FP8 path: 3 LT matmuls with a SHARED X→FP8 quant, vs the
     * old single perrow_mt4_concat3 launch. LT is fast enough that 3
     * separate calls beat one fused custom kernel; sharing the X quant
     * keeps overhead from tripling. */
    if (r->use_cublaslt_fp8 && n_tok >= 1 && n_in >= 32 && dim >= 16 &&
        !r->use_old_gemm) {
        size_t need_x = (size_t)n_tok * n_in;
        size_t need_y = (size_t)n_tok * dim * sizeof(uint16_t);
        if (need_x > r->x_fp8_scratch_n) {
            if (r->d_x_fp8_scratch) cuMemFree(r->d_x_fp8_scratch);
            r->d_x_fp8_scratch = 0;
            if (cuMemAlloc(&r->d_x_fp8_scratch, need_x) == CUDA_SUCCESS)
                r->x_fp8_scratch_n = need_x;
            else { r->x_fp8_scratch_n = 0; goto qkv_fallback; }
        }
        if (need_y > r->y_bf16_scratch_n) {
            if (r->d_y_bf16_scratch) cuMemFree(r->d_y_bf16_scratch);
            r->d_y_bf16_scratch = 0;
            if (cuMemAlloc(&r->d_y_bf16_scratch, need_y) == CUDA_SUCCESS)
                r->y_bf16_scratch_n = need_y;
            else { r->y_bf16_scratch_n = 0; goto qkv_fallback; }
        }
        if (!r->d_x_fp8_scratch || !r->d_y_bf16_scratch) goto qkv_fallback;

        /* 1. one reduce_max_abs(X) shared across Q/K/V */
        if (!r->d_qkv_max) cuMemAlloc(&r->d_qkv_max, sizeof(float));
        cuMemsetD32Async(r->d_qkv_max, 0, 1, r->stream);
        int n_x = n_tok * n_in;
        void *r_args[] = {&r->d_qkv_max, &X, &n_x};
        unsigned rb = (unsigned)((n_x + 255) / 256);
        cuLaunchKernel(r->reduce_max_abs, rb, 1, 1, 256, 1, 1,
                       0, r->stream, r_args, NULL);
        void *s_args[] = {&r->d_x_scale_f32, &r->d_qkv_max};
        cuLaunchKernel(r->compute_x_scale_from_max, 1, 1, 1, 1, 1, 1,
                       0, r->stream, s_args, NULL);
        /* 2. one quant X→FP8 shared across Q/K/V */
        void *q_args[] = {&r->d_x_fp8_scratch, &X, &r->d_x_scale_f32, &n_x};
        unsigned qb = (unsigned)(((n_x + 1) / 2 + 255) / 256);
        cuLaunchKernel(r->quant_f32_to_fp8_e4m3, qb, 1, 1, 256, 1, 1,
                       0, r->stream, q_args, NULL);

        /* 3. three LT matmuls into BF16 scratch + expand-with-bias.
         * y_bf16_scratch is reused after each expand. */
        CUdeviceptr Ws[3] = {Wq, Wk, Wv};
        CUdeviceptr bs[3] = {bq, bk, bv};
        CUdeviceptr Ys[3] = {Yq, Yk, Yv};
        for (int i = 0; i < 3; i++) {
            int rc = cublasew_gemm_fp8_e4m3_bf16out_rowmajor_nt(
                r->cublaslt_ctx,
                r->d_y_bf16_scratch,
                Ws[i], r->d_x_fp8_scratch,
                r->d_w_scale_one, r->d_x_scale_f32,
                0,
                n_tok, dim, n_in);
            if (rc != 0) goto qkv_fallback;
            CUdeviceptr Y_i = Ys[i], b_i = bs[i];
            void *e_args[] = {&Y_i, &r->d_y_bf16_scratch, &b_i, &dim, &n_tok};
            int total = dim * n_tok;
            unsigned eb = (unsigned)((total + 255) / 256);
            cuLaunchKernel(r->bf16_to_f32_add_bias, eb, 1, 1, 256, 1, 1,
                           0, r->stream, e_args, NULL);
        }
        return 0;
    }
qkv_fallback:
    if (!r->use_fp8_pipe_perrow || !r->gemm_fp8_pipe_perrow_mt4_concat3 ||
        !r->reduce_max_abs_per_row || r->use_old_gemm) return -1;
    if (n_tok < 64 || (n_tok % 64) != 0 || (dim % 256) != 0 || (n_in % 32) != 0)
        return -1;

    /* Grow shared row_max buf if needed (op_gemm uses the same buffer). */
    size_t need = (size_t)n_tok * sizeof(float);
    if (need > r->row_max_buf_n) {
        if (r->d_row_max_buf) { cuMemFree(r->d_row_max_buf); r->d_row_max_buf = 0; }
        if (cuMemAlloc(&r->d_row_max_buf, need) != CUDA_SUCCESS) {
            r->d_row_max_buf = 0; r->row_max_buf_n = 0;
            return -1;
        }
        r->row_max_buf_n = need;
    }

    /* One row_max launch shared across QKV. */
    void *rargs[] = {&r->d_row_max_buf, &X, &n_tok, &n_in};
    cuLaunchKernel(r->reduce_max_abs_per_row, (unsigned)n_tok, 1, 1,
                   256, 1, 1, 0, r->stream, rargs, NULL);

    /* Combined N-axis grid: 3*dim cols, 64 rows/CTA. Pad to mult of 4 for swizzle. */
    unsigned gx = (unsigned)((3 * dim + 255) / 256);
    unsigned gy = (unsigned)((n_tok + 63) / 64);
    gx = (gx + 3u) & ~3u;
    gy = (gy + 3u) & ~3u;
    size_t smem_c3 = 2048 + 8192 * 2 + 512;  /* same layout as perrow_mt4 */
    void *args[] = {&Yq, &Yk, &Yv, &Wq, &Wk, &Wv, &X,
                    &bq, &bk, &bv, &dim, &n_in, &n_tok, &r->d_row_max_buf};
    cuLaunchKernel(r->gemm_fp8_pipe_perrow_mt4_concat3, gx, gy, 1, 128, 1, 1,
                   smem_c3, r->stream, args, NULL);
    return 0;
}

static void op_attn(cuda_qimg_runner *r, CUdeviceptr d_out, CUdeviceptr d_q,
                    CUdeviceptr d_k, CUdeviceptr d_v,
                    int n_tok, int n_heads, int head_dim) {
    /* BF16 MMA flash attention path — bulk F32→BF16 cast Q/K/V, then mma.sync.
     * Has priority over the FP8 path because it matches ComfyUI's BF16 reference
     * exactly (no per-tensor scale outlier crush). */
    if (r->use_bf16_attn && r->flash_attn_bf16 && r->cast_f32_to_bf16) {
        int dim = n_heads * head_dim;
        int n_elem = n_tok * dim;
        size_t need = (size_t)n_elem * sizeof(unsigned short);
        if (ensure_bf16_attn_buf(r, need) == 0) {
            cast_buf_f32_to_bf16(r, r->d_q_bf16, d_q, n_elem);
            cast_buf_f32_to_bf16(r, r->d_k_bf16, d_k, n_elem);
            cast_buf_f32_to_bf16(r, r->d_v_bf16, d_v, n_elem);
            unsigned gy = (unsigned)((n_tok + 63) / 64);
            /* smem: 2× double-buffered (sK0+sK1+sV0+sV1) at HDP=136 (bank-
             * conflict-free padded stride) ~34 KB. P held in per-lane regs. */
            size_t smem = (size_t)(4 * 32 * 136 * 2);
            void *args[] = {&d_out, &r->d_q_bf16, &r->d_k_bf16, &r->d_v_bf16,
                            &n_tok, &n_heads, &head_dim};
            cuLaunchKernel(r->flash_attn_bf16,
                           (unsigned)n_heads, gy, 1,
                           128, 1, 1, smem, r->stream, args, NULL);
            return;
        }
        /* OOM → fall through */
    }

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
    if (cuModuleGetFunction(&r->gemm_fp8_mma_pipe, module, "gemm_fp8_scaled_f32_pipe") != CUDA_SUCCESS)
        r->gemm_fp8_mma_pipe = NULL;
    if (cuModuleGetFunction(&r->gemm_bf16_mma_pipe, module, "gemm_bf16_pipe_f32") != CUDA_SUCCESS)
        r->gemm_bf16_mma_pipe = NULL;
    if (cuModuleGetFunction(&r->gemm_fp8_pipe_perrow, module, "gemm_fp8_pipe_perrow_f32") != CUDA_SUCCESS)
        r->gemm_fp8_pipe_perrow = NULL;
    if (cuModuleGetFunction(&r->gemm_fp8_pipe_perrow_mt4, module, "gemm_fp8_pipe_perrow_mt4_f32") != CUDA_SUCCESS)
        r->gemm_fp8_pipe_perrow_mt4 = NULL;
    if (cuModuleGetFunction(&r->gemm_fp8_pipe_perrow_mt4_concat3, module, "gemm_fp8_pipe_perrow_mt4_concat3_f32") != CUDA_SUCCESS)
        r->gemm_fp8_pipe_perrow_mt4_concat3 = NULL;
    if (cuModuleGetFunction(&r->reduce_max_abs_per_row, module, "reduce_max_abs_per_row_f32") != CUDA_SUCCESS)
        r->reduce_max_abs_per_row = NULL;
    if (cuModuleGetFunction(&r->quant_f32_to_fp8_e4m3, module, "quant_f32_to_fp8_e4m3") != CUDA_SUCCESS)
        r->quant_f32_to_fp8_e4m3 = NULL;
    if (cuModuleGetFunction(&r->compute_x_scale_from_max, module, "compute_x_scale_from_max") != CUDA_SUCCESS)
        r->compute_x_scale_from_max = NULL;
    if (cuModuleGetFunction(&r->bf16_to_f32_add_bias, module, "bf16_to_f32_add_bias") != CUDA_SUCCESS)
        r->bf16_to_f32_add_bias = NULL;
    if (cuModuleGetFunction(&r->bf16_add_bias_gelu_f32, module, "bf16_add_bias_gelu_f32") != CUDA_SUCCESS)
        r->bf16_add_bias_gelu_f32 = NULL;
    if (cuModuleGetFunction(&r->bf16_add_bias_gelu_inplace_max, module, "bf16_add_bias_gelu_inplace_max") != CUDA_SUCCESS)
        r->bf16_add_bias_gelu_inplace_max = NULL;
    if (cuModuleGetFunction(&r->quant_bf16_to_fp8_e4m3, module, "quant_bf16_to_fp8_e4m3") != CUDA_SUCCESS)
        r->quant_bf16_to_fp8_e4m3 = NULL;
    if (cuModuleGetFunction(&r->flash_attn_fp8, module, "flash_attn_fp8") != CUDA_SUCCESS)
        r->flash_attn_fp8 = NULL;
    if (cuModuleGetFunction(&r->flash_attn_bf16, module, "flash_attn_bf16") != CUDA_SUCCESS)
        r->flash_attn_bf16 = NULL;
    if (cuModuleGetFunction(&r->cast_f32_to_bf16, module, "cast_f32_to_bf16") != CUDA_SUCCESS)
        r->cast_f32_to_bf16 = NULL;
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
    /* Pipelined cp.async W-in-smem variant: ON by default on sm_80+.
     * Prefetches each k-tile's W slab into smem while MMAs run on the
     * current tile, hiding operand-load latency. Measured +6% at 256x256
     * / +9% at 512x512 and bit-identical to the baseline MMA path.
     * Opt out with QIMG_FP8_PIPE=0 to A/B test. */
    r->use_fp8_pipe = 0;
    if (r->use_fp8_mma && r->gemm_fp8_mma_pipe && sm >= 80) {
        const char *env = getenv("QIMG_FP8_PIPE");
        r->use_fp8_pipe = (env && env[0] == '0') ? 0 : 1;
        if (verbose && r->use_fp8_pipe)
            fprintf(stderr, "cuda_qimg: FP8 MMA pipelined cp.async GEMM enabled\n");
    }
    /* Parallel CFG streams: opt-in via QIMG_CFG_STREAMS=1.
     * Creates a secondary compute stream + synchronization events so
     * dit_step_cfg can run cond and uncond passes concurrently. */
    if (getenv("QIMG_CFG_STREAMS")) {
        if (cuStreamCreate(&r->uncond_stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS) {
            cuEventCreate(&r->block_barrier_cond, CU_EVENT_DISABLE_TIMING);
            cuEventCreate(&r->block_barrier_unc,  CU_EVENT_DISABLE_TIMING);
            cuEventCreate(&r->mod_ready,          CU_EVENT_DISABLE_TIMING);
            if (verbose)
                fprintf(stderr, "cuda_qimg: parallel CFG streams enabled\n");
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
    /* BF16 flash attention: default ON (sm_80+). Higher precedence than FP8
     * attention — matches ComfyUI's BF16 reference exactly so apple_compare
     * mean_diff drops from ~5 to <2. Validated 2026-05-03 at 512² (1.95 vs
     * 2.50 s/step, ~22% faster than F32 attn). Set QIMG_BF16_ATTN=0 to opt out. */
    r->use_bf16_attn = 0;
    {
        const char *env = getenv("QIMG_BF16_ATTN");
        int want = (env == NULL) ? 1 : (env[0] != '0');
        if (want && r->flash_attn_bf16 && r->cast_f32_to_bf16 && sm >= 80) {
            r->use_bf16_attn = 1;
            if (verbose)
                fprintf(stderr, "cuda_qimg: BF16 MMA flash attention enabled (sm_%d)\n", sm);
        }
    }
    /* BF16 MMA GEMM: opt-in via QIMG_BF16_MMA=1. Reads FP8 weights, decodes to
     * BF16 in-kernel, runs mma.sync.m16n8k16.bf16. Matches ComfyUI precision
     * without sacrificing tensor-core speed. */
    r->use_bf16_mma = 0;
    {
        const char *env = getenv("QIMG_BF16_MMA");
        if (env && env[0] == '1' && r->gemm_bf16_mma_pipe && sm >= 80) {
            r->use_bf16_mma = 1;
            if (verbose)
                fprintf(stderr, "cuda_qimg: BF16 MMA pipelined GEMM enabled (sm_%d)\n", sm);
        }
    }
    /* Per-row FP8 MMA pipe: default ON (sm_80+). Eliminates per-tensor outlier
     * crush by using one X scale per row, and the cp.async + ldmatrix +
     * panel-swizzle pipeline beats the non-perrow path. Set
     * QIMG_FP8_PIPE_PERROW=0 to opt out. */
    r->use_fp8_pipe_perrow = 0;
    {
        const char *env = getenv("QIMG_FP8_PIPE_PERROW");
        int want = (env == NULL) ? 1 : (env[0] != '0');
        if (want && r->gemm_fp8_pipe_perrow && r->reduce_max_abs_per_row && sm >= 80) {
            r->use_fp8_pipe_perrow = 1;
            if (verbose)
                fprintf(stderr, "cuda_qimg: FP8 MMA per-row pipelined GEMM enabled (sm_%d)\n", sm);
        }
    }

    /* cuBLAS-LT FP8 path: highest priority when available. Reaches
     * 60-145 TFLOP/s on qimg shapes (RTX 5060 Ti) vs ~4-15 TFLOP/s for
     * the perrow kernel. Set QIMG_CUBLASLT_FP8=0 to opt out. */
    r->use_cublaslt_fp8 = 0;
    r->cublaslt_ctx = NULL;
    r->d_x_fp8_scratch = 0; r->x_fp8_scratch_n = 0;
    r->d_y_bf16_scratch = 0; r->y_bf16_scratch_n = 0;
    r->d_x_scale_f32 = 0; r->d_w_scale_one = 0;
    {
        const char *env = getenv("QIMG_CUBLASLT_FP8");
        int want = (env == NULL) ? 1 : (env[0] != '0');
        if (want && r->quant_f32_to_fp8_e4m3 && r->compute_x_scale_from_max &&
            r->bf16_to_f32_add_bias && r->reduce_max_abs && sm >= 89) {
            if (cublasewCreate(&r->cublaslt_ctx, r->stream) == 0 &&
                cublasew_lt_available(r->cublaslt_ctx) == 0) {
                if (cuMemAlloc(&r->d_x_scale_f32, sizeof(float)) == CUDA_SUCCESS &&
                    cuMemAlloc(&r->d_w_scale_one, sizeof(float)) == CUDA_SUCCESS) {
                    float one = 1.0f;
                    cuMemcpyHtoD(r->d_w_scale_one, &one, sizeof(float));
                    r->use_cublaslt_fp8 = 1;
                    if (verbose)
                        fprintf(stderr, "cuda_qimg: cuBLAS-LT FP8 e4m3 GEMM enabled (sm_%d)\n", sm);
                } else {
                    if (r->cublaslt_ctx) { cublasewDestroy(r->cublaslt_ctx); r->cublaslt_ctx = NULL; }
                }
            } else if (r->cublaslt_ctx) {
                cublasewDestroy(r->cublaslt_ctx);
                r->cublaslt_ctx = NULL;
            }
        }
    }
    /* Fused FP8 MLP with BF16 hidden state — default ON when LT-FP8 is active.
     * Skips materializing the F32 hidden tensor (~192 MB at 512², 4×bigger at
     * 1024²). Slight precision drift (~1% max, ~10% mean) from extra BF16
     * round-trip; bit-stable across runs. Set QIMG_MLP_BF16_HIDDEN=0 to opt out. */
    {
        const char *env = getenv("QIMG_MLP_BF16_HIDDEN");
        int want = (env == NULL || env[0] != '0');
        r->mlp_bf16_hidden = (want && r->use_cublaslt_fp8 &&
                              r->bf16_add_bias_gelu_inplace_max &&
                              r->quant_bf16_to_fp8_e4m3) ? 1 : 0;
        if (r->mlp_bf16_hidden && verbose)
            fprintf(stderr, "cuda_qimg: fused FP8 MLP with BF16 hidden enabled\n");
    }
    /* Per-stage block profiling */
    {
        const char *env = getenv("QIMG_PROFILE_BLOCK");
        r->profile_block = (env && env[0] != '0') ? 1 : 0;
        if (r->profile_block && verbose)
            fprintf(stderr, "cuda_qimg: per-stage block profiling enabled (host-sync)\n");
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
    if (cuModuleGetFunction(&r->vae_f32_to_fp8_padded, module, "vae_f32_to_fp8_padded") != CUDA_SUCCESS)
        r->vae_f32_to_fp8_padded = NULL;
    if (cuModuleGetFunction(&r->vae_im2col, module, "vae_im2col_f32") != CUDA_SUCCESS)
        r->vae_im2col = NULL;
    if (cuModuleGetFunction(&r->vae_crop_transpose_add_bias, module, "vae_crop_transpose_add_bias_f32") != CUDA_SUCCESS)
        r->vae_crop_transpose_add_bias = NULL;
    if (cuModuleGetFunction(&r->vae_transpose_chw_to_sc, module, "vae_transpose_chw_to_sc_f32") != CUDA_SUCCESS)
        r->vae_transpose_chw_to_sc = NULL;
    if (cuModuleGetFunction(&r->vae_transpose_sc_to_chw, module, "vae_transpose_sc_to_chw_f32") != CUDA_SUCCESS)
        r->vae_transpose_sc_to_chw = NULL;
    if (cuModuleGetFunction(&r->vae_attn_sc, module, "vae_attn_sc_f32") != CUDA_SUCCESS)
        r->vae_attn_sc = NULL;
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

    /* Upload FP8→BF16 LUT to GPU constant memory (for gemm_bf16_pipe_f32) */
    {
        qimg_init_fp8_to_bf16_lut();
        CUdeviceptr d_lut_bf;
        size_t lut_size_bf;
        CUresult lut_rc = cuModuleGetGlobal(&d_lut_bf, &lut_size_bf, module, "d_fp8_to_bf16_lut");
        if (lut_rc == CUDA_SUCCESS && lut_size_bf == 256 * sizeof(uint16_t)) {
            cuMemcpyHtoD(d_lut_bf, qimg_fp8_to_bf16_lut, 256 * sizeof(uint16_t));
            if (verbose)
                fprintf(stderr, "cuda_qimg: FP8→BF16 LUT uploaded\n");
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

    /* Pin the safetensors mmap region so cuMemcpyHtoDAsync can use the full
     * PCIe bandwidth and overlap with compute. Without this, the driver stages
     * pageable mmap pages through internal pinned bounce buffers serially and
     * QIMG_PIPELINE provides no benefit. Set QIMG_NO_PIN=1 to opt out. */
    if (!getenv("QIMG_NO_PIN") && cuMemHostRegister) {
        /* Try READ_ONLY first (lighter; CUDA 11.1+); fall back to default. */
        CUresult rr = cuMemHostRegister(st->map_base, st->map_size,
                                        CU_MEMHOSTREGISTER_READ_ONLY);
        if (rr != CUDA_SUCCESS) {
            rr = cuMemHostRegister(st->map_base, st->map_size, 0);
        }
        if (rr == CUDA_SUCCESS) {
            r->dit_st_pinned = 1;
            fprintf(stderr, "cuda_qimg: pinned %.2f GB DiT mmap region for fast H2D\n",
                    (double)st->map_size / (1ull << 30));
        } else {
            fprintf(stderr, "cuda_qimg: cuMemHostRegister failed (%d) — DiT H2D will be slow\n",
                    (int)rr);
        }
    }


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
        /* Scratch working set at 512x512 is ~300 MB. Reserve 324 MB for the
         * one on-demand scratch slot (+324 MB for the optional second slot
         * when QIMG_PIPELINE=1 — see dit_step block loop comment). */
        size_t workspace = (500ULL + 324ULL) * 1024 * 1024;
        if (getenv("QIMG_PIPELINE")) workspace += 324ULL * 1024 * 1024;
        int max_preload = (free_mem > workspace)
            ? (int)((free_mem - workspace) / block_bytes) : 0;
        if (max_preload > r->dit_n_blocks) max_preload = r->dit_n_blocks;
        /* QIMG_MAX_PRELOAD env var to cap preload count (for perf testing). */
        const char *_mp_env = getenv("QIMG_MAX_PRELOAD");
        if (_mp_env) {
            int cap = atoi(_mp_env);
            if (cap >= 0 && cap < max_preload) max_preload = cap;
        }

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

    /* Pre-stage streamed blocks into pinned host memory so cuMemcpyHtoDAsync
     * actually overlaps with compute and runs at full PCIe bandwidth (~13 GB/s
     * vs ~4 GB/s for pageable mmap). One pool of cuMemAllocHost per streamed
     * block; ~324 MB each, so 16 streamed blocks ≈ 5 GB pinned RAM. Skip with
     * QIMG_NO_BOUNCE=1. Falls back gracefully if pinned alloc fails. */
    if (r->n_preloaded < r->dit_n_blocks && !getenv("QIMG_NO_BOUNCE") &&
        !r->dit_st_pinned) {
        r->dit_pinned_n_tensors = st->n_tensors;
        r->dit_pinned_data = (void **)calloc((size_t)st->n_tensors, sizeof(void *));
        r->dit_pinned_pools = (void **)calloc((size_t)r->dit_n_blocks, sizeof(void *));
        if (r->dit_pinned_data && r->dit_pinned_pools) {
            int staged = 0;
            size_t pinned_bytes = 0;
            for (int b = r->n_preloaded; b < r->dit_n_blocks; b++) {
                if (qimg_stage_block_pinned(r, b) != 0) {
                    fprintf(stderr, "cuda_qimg: pinned bounce alloc failed at "
                                    "block %d (staged %d, %.1f GB pinned) — "
                                    "remaining blocks fall back to mmap\n",
                            b, staged, (float)pinned_bytes / (1<<30));
                    break;
                }
                staged++;
                /* Track pool bytes for log only; we don't store the size. */
                pinned_bytes += 324ull * 1024 * 1024;  /* approximate */
            }
            if (staged > 0)
                fprintf(stderr, "cuda_qimg: pre-staged %d streamed blocks in "
                        "pinned host memory (~%.1f GB) for fast H2D\n",
                        staged, (float)pinned_bytes / (1<<30));
        } else {
            free(r->dit_pinned_data); r->dit_pinned_data = NULL;
            free(r->dit_pinned_pools); r->dit_pinned_pools = NULL;
            r->dit_pinned_n_tensors = 0;
        }
    }

    /* Allocate the on-demand scratch slot(s) once if we didn't preload every
     * block. Uses ~324 MB per slot on top of the preloaded weights but
     * eliminates the 30 cuMemAlloc/cuMemFree per on-demand load. A second
     * slot + copy stream + events enables prefetching block L+1's weights
     * while block L's kernels are still running. */
    if (r->n_preloaded < r->dit_n_blocks) {
        if (qimg_alloc_scratch_block(r) == 0) {
            if (r->verbose)
                fprintf(stderr, "cuda_qimg: on-demand scratch slot allocated\n");
        } else {
            fprintf(stderr, "cuda_qimg: scratch slot alloc failed — "
                            "falling back to per-block alloc\n");
        }
        /* Only allocate the second scratch slot + copy stream when
         * QIMG_PIPELINE=1 (see dit_step block loop comment). Saves 324 MB. */
        if (getenv("QIMG_PIPELINE")) {
            if (qimg_alloc_scratch_block_b(r) == 0) {
                if (r->verbose)
                    fprintf(stderr, "cuda_qimg: double-buffered scratch enabled "
                            "(copy stream + 2 slots)\n");
            } else {
                fprintf(stderr, "cuda_qimg: 2nd scratch slot alloc failed — "
                                "single-slot mode\n");
            }
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
    if (r->scratch_block_b_ready) {
        qimg_free_block(&r->scratch_block_b);
        r->scratch_block_b_ready = 0;
    }
    for (int i = 0; i < 2; i++) {
        if (r->slot_ready[i]) cuEventDestroy(r->slot_ready[i]);
        if (r->slot_free[i])  cuEventDestroy(r->slot_free[i]);
    }
    if (r->copy_stream) cuStreamDestroy(r->copy_stream);
    if (r->uncond_stream) cuStreamDestroy(r->uncond_stream);
    if (r->block_barrier_cond) cuEventDestroy(r->block_barrier_cond);
    if (r->block_barrier_unc)  cuEventDestroy(r->block_barrier_unc);
    if (r->mod_ready)          cuEventDestroy(r->mod_ready);
    CUdeviceptr *globals = &r->d_img_in_w;
    for (int i = 0; i < 13; i++) { if (globals[i]) cuMemFree(globals[i]); }
    /* FP8 attention workspace (lazily allocated by ensure_fp8_attn_buf) */
    if (r->d_q_fp8)      cuMemFree(r->d_q_fp8);
    if (r->d_k_fp8)      cuMemFree(r->d_k_fp8);
    if (r->d_v_fp8)      cuMemFree(r->d_v_fp8);
    if (r->d_qkv_scales) cuMemFree(r->d_qkv_scales);
    if (r->d_qkv_max)    cuMemFree(r->d_qkv_max);
    /* cuBLAS-LT FP8 workspace */
    if (r->d_x_fp8_scratch) cuMemFree(r->d_x_fp8_scratch);
    if (r->d_y_bf16_scratch) cuMemFree(r->d_y_bf16_scratch);
    if (r->d_x_scale_f32) cuMemFree(r->d_x_scale_f32);
    if (r->d_w_scale_one) cuMemFree(r->d_w_scale_one);
    if (r->cublaslt_ctx) cublasewDestroy(r->cublaslt_ctx);
    /* BF16 attention workspace */
    if (r->d_q_bf16) cuMemFree(r->d_q_bf16);
    if (r->d_k_bf16) cuMemFree(r->d_k_bf16);
    if (r->d_v_bf16) cuMemFree(r->d_v_bf16);
    /* Per-row FP8 MMA scale buffer */
    if (r->d_row_max_buf) cuMemFree(r->d_row_max_buf);
    if (r->dit_pinned_pools) {
        for (int b = 0; b < r->dit_n_blocks; b++) {
            if (r->dit_pinned_pools[b]) cuMemFreeHost(r->dit_pinned_pools[b]);
        }
        free(r->dit_pinned_pools); r->dit_pinned_pools = NULL;
    }
    if (r->dit_pinned_data) {
        free(r->dit_pinned_data); r->dit_pinned_data = NULL;
    }
    r->dit_pinned_n_tensors = 0;
    if (r->dit_st) {
        if (r->dit_st_pinned && cuMemHostUnregister) {
            cuMemHostUnregister(((st_context *)r->dit_st)->map_base);
            r->dit_st_pinned = 0;
        }
        safetensors_close((st_context *)r->dit_st);
    }
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
 * When `img_mlp_in_external` is non-zero, the img adaLN2 output is written
 * there instead of d_scratch1, and the img MLP (fc1+gelu+fc2) plus the
 * post-MLP img gated_add are SKIPPED — the caller is expected to run a
 * batched version that processes cond+uncond together. Text MLP still runs
 * here since n_txt differs between passes. The caller is responsible for
 * the post-MLP img gated_add using the batched MLP output.
 *
 * Returns 0 on success. */
static int qimg_forward_block(cuda_qimg_runner *r, qimg_fwd_state_t *st,
                              const qimg_block_gpu *blk, int L,
                              CUdeviceptr d_img_mod, CUdeviceptr d_txt_mod,
                              int hp_rope, int wp_rope,
                              int t_dim_rope, int h_dim_rope, int w_dim_rope,
                              float rope_theta,
                              CUdeviceptr img_mlp_in_external)
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

    /* Profiling brackets — host-side sync, only active when profile_block=1.
     * Adds significant overhead, so opt-in via env QIMG_PROFILE_BLOCK=1. */
    double _prof_t0 = 0.0;
    #define PROF_BEGIN() do { if (r->profile_block) { cuStreamSynchronize(s); _prof_t0 = (double)clock()/CLOCKS_PER_SEC; } } while (0)
    #define PROF_END(bucket) do { if (r->profile_block) { cuStreamSynchronize(s); r->bucket += ((double)clock()/CLOCKS_PER_SEC - _prof_t0) * 1000.0; } } while (0)

    PROF_BEGIN();
    /* adaLN image → d_scratch1 */
    op_adaln(r, d_scratch1, d_img, img_sh1, img_sc1, n_img, dim);
    /* adaLN text → d_scratch2 */
    op_adaln(r, d_scratch2, d_txt, txt_sh1, txt_sc1, n_txt, dim);

    /* Image QKV → offset into joint buffers at [n_txt:] */
    CUdeviceptr d_img_q = d_q + (size_t)n_txt * dim * sizeof(float);
    CUdeviceptr d_img_k = d_k + (size_t)n_txt * dim * sizeof(float);
    CUdeviceptr d_img_v = d_v + (size_t)n_txt * dim * sizeof(float);
    if (op_gemm_qkv_fused(r, d_img_q, d_img_k, d_img_v,
                          blk->attn_q_w, blk->attn_k_w, blk->attn_v_w, d_scratch1,
                          blk->attn_q_b, blk->attn_k_b, blk->attn_v_b,
                          dim, dim, n_img) != 0) {
        op_gemm(r, d_img_q, blk->attn_q_w, d_scratch1, blk->attn_q_b, dim, dim, n_img);
        op_gemm(r, d_img_k, blk->attn_k_w, d_scratch1, blk->attn_k_b, dim, dim, n_img);
        op_gemm(r, d_img_v, blk->attn_v_w, d_scratch1, blk->attn_v_b, dim, dim, n_img);
    }

    /* Text QKV → offset at [0:n_txt] */
    CUdeviceptr d_txt_q = d_q;
    CUdeviceptr d_txt_k = d_k;
    CUdeviceptr d_txt_v = d_v;
    if (op_gemm_qkv_fused(r, d_txt_q, d_txt_k, d_txt_v,
                          blk->attn_add_q_w, blk->attn_add_k_w, blk->attn_add_v_w, d_scratch2,
                          blk->attn_add_q_b, blk->attn_add_k_b, blk->attn_add_v_b,
                          dim, dim, n_txt) != 0) {
        op_gemm(r, d_txt_q, blk->attn_add_q_w, d_scratch2, blk->attn_add_q_b, dim, dim, n_txt);
        op_gemm(r, d_txt_k, blk->attn_add_k_w, d_scratch2, blk->attn_add_k_b, dim, dim, n_txt);
        op_gemm(r, d_txt_v, blk->attn_add_v_w, d_scratch2, blk->attn_add_v_b, dim, dim, n_txt);
    }

    PROF_END(prof_qkv_ms);

    PROF_BEGIN();
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

    PROF_END(prof_qknorm_ms);

    PROF_BEGIN();
    /* Joint attention: Q/K/V concatenated as [txt, img] */
    op_attn(r, d_attn_out, d_q, d_k, d_v, n_total, nh, hd);
    PROF_END(prof_attn_ms);

    PROF_BEGIN();
    /* Output projections */
    CUdeviceptr d_img_attn = d_attn_out + (size_t)n_txt * dim * sizeof(float);
    CUdeviceptr d_txt_attn = d_attn_out;
    op_gemm(r, d_scratch1, blk->attn_out_w, d_img_attn, blk->attn_out_b, dim, dim, n_img);
    op_gemm(r, d_scratch2, blk->attn_add_out_w, d_txt_attn, blk->attn_add_out_b, dim, dim, n_txt);

    /* Gated residual */
    op_gated_add(r, d_img, d_scratch1, img_g1, n_img, dim);
    op_gated_add(r, d_txt, d_scratch2, txt_g1, n_txt, dim);
    PROF_END(prof_attnout_ms);

    PROF_BEGIN();
    /* Image MLP. When `img_mlp_in_external` is set, run adaLN2 into that
     * external buffer and skip the MLP + gated_add — the caller will do a
     * CFG-batched MLP across cond+uncond before running its own gated_add. */
    if (img_mlp_in_external) {
        op_adaln(r, img_mlp_in_external, d_img, img_sh2, img_sc2, n_img, dim);
    } else {
        op_adaln(r, d_scratch1, d_img, img_sh2, img_sc2, n_img, dim);
        /* Try fused FP8 MLP (BF16 hidden, no F32 intermediate). adaLN output is in
         * d_scratch1, so write fused MLP into d_scratch3 first, then move (or
         * use d_scratch3 as the output and pass it to gated_add). */
        if (op_mlp_fp8(r, d_scratch3, d_scratch1,
                       blk->img_mlp_fc1_w, blk->img_mlp_fc1_b,
                       blk->img_mlp_fc2_w, blk->img_mlp_fc2_b,
                       dim, mlp_h, n_img) == 0) {
            /* fused path: result already in d_scratch3 — alias for gated_add */
            op_gated_add(r, d_img, d_scratch3, img_g2, n_img, dim);
            goto img_mlp_done;
        }
        op_gemm_gelu(r, d_scratch3, blk->img_mlp_fc1_w, d_scratch1, blk->img_mlp_fc1_b,
                     mlp_h, dim, n_img);
        op_gemm(r, d_scratch1, blk->img_mlp_fc2_w, d_scratch3, blk->img_mlp_fc2_b,
                dim, mlp_h, n_img);
        op_gated_add(r, d_img, d_scratch1, img_g2, n_img, dim);
    }
    img_mlp_done:;
    PROF_END(prof_imgmlp_ms);

    PROF_BEGIN();
    /* Text MLP — try fused FP8 first (BF16 hidden, no F32 intermediate),
     * fall back to op_gemm_gelu+op_gemm if not available. */
    op_adaln(r, d_scratch2, d_txt, txt_sh2, txt_sc2, n_txt, dim);
    if (op_mlp_fp8(r, d_scratch3, d_scratch2,
                   blk->txt_mlp_fc1_w, blk->txt_mlp_fc1_b,
                   blk->txt_mlp_fc2_w, blk->txt_mlp_fc2_b,
                   dim, mlp_h, n_txt) == 0) {
        op_gated_add(r, d_txt, d_scratch3, txt_g2, n_txt, dim);
        goto txt_mlp_done;
    }
    op_gemm_gelu(r, d_scratch3, blk->txt_mlp_fc1_w, d_scratch2, blk->txt_mlp_fc1_b,
                 mlp_h, dim, n_txt);
    op_gemm(r, d_scratch2, blk->txt_mlp_fc2_w, d_scratch3, blk->txt_mlp_fc2_b,
            dim, mlp_h, n_txt);
    op_gated_add(r, d_txt, d_scratch2, txt_g2, n_txt, dim);
    /* gated_add writes BF16-rounded output — no trailing op_bf16_trunc needed. */
    txt_mlp_done:;
    PROF_END(prof_txtmlp_ms);
    if (r->profile_block) r->prof_block_count++;
    #undef PROF_BEGIN
    #undef PROF_END

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

    /* 4. Process all blocks with double-buffered on-demand loading.
     *
     * Pipeline: while block L's kernels run on r->stream, block L+1's
     * weights are copied into the OTHER scratch slot on r->copy_stream.
     * cuEvents coordinate slot reuse safely.
     *
     *  - slot_ready[i] : fires when copy_stream finished filling slot i
     *  - slot_free[i]  : fires when r->stream finished reading slot i
     */
    /* Double-buffered on-demand block load is OFF by default — measured
     * identical to the single-slot path (7.19 s/step at 512x512 with
     * max_preload=12 → 48 on-demand blocks, pipeline on vs off differs
     * by 10 ms). The copy engine on sm_89+ GPUs already runs concurrent
     * with SM compute even when both are queued on one stream, so manual
     * double-buffering adds event overhead without unlocking new parallelism.
     * Kept behind QIMG_PIPELINE=1 for future experimentation. */
    int use_pipeline = 0;
    if (getenv("QIMG_PIPELINE") && r->scratch_block_b_ready &&
        r->n_preloaded < r->dit_n_blocks) {
        use_pipeline = 1;
    }
    qimg_block_gpu *slot_ptrs[2] = { &r->scratch_block, &r->scratch_block_b };

    /* Prefetch initial 2 slots before the loop begins, if pipelining. */
    if (use_pipeline) {
        for (int i = 0; i < 2; i++) {
            int L0 = r->n_preloaded + i;
            if (L0 >= r->dit_n_blocks) break;
            qimg_block_gpu tmp;
            qimg_load_block_into_slot_ex(r, L0, slot_ptrs[i], &tmp, r->copy_stream);
            cuEventRecord(r->slot_ready[i], r->copy_stream);
        }
    }

    for (int L = 0; L < r->dit_n_blocks; L++) {
        if (r->verbose && (L % 10 == 0 || L == r->dit_n_blocks - 1))
            fprintf(stderr, "\r  cuda_qimg: block %d/%d", L + 1, r->dit_n_blocks);

        qimg_block_gpu blk;
        int slot_used = -1;  /* which double-buffer slot (0/1) this block consumed */
        if (L < r->n_preloaded && r->gpu_blocks[L].attn_q_w) {
            blk = r->gpu_blocks[L];
        } else if (use_pipeline) {
            int od = L - r->n_preloaded;
            slot_used = od % 2;
            /* Wait for the copy_stream to have finished filling this slot. */
            cuStreamWaitEvent(s, r->slot_ready[slot_used], 0);
            blk = *slot_ptrs[slot_used];
        } else {
            /* Single-slot fallback: synchronous copy on r->stream. */
            if (qimg_load_block_into_slot(r, L, &blk) != 0) {
                fprintf(stderr, "cuda_qimg: block %d slot load failed\n", L);
                return -1;
            }
        }

        /* Modulation + shared block forward */
        cuMemcpyDtoDAsync(d_t_silu, d_t_emb, (size_t)dim * sizeof(float), s);
        op_silu(r, d_t_silu, dim);
        op_gemm(r, d_img_mod, blk.img_mod_w, d_t_silu, blk.img_mod_b, 6 * dim, dim, 1);
        op_gemm(r, d_txt_mod, blk.txt_mod_w, d_t_silu, blk.txt_mod_b, 6 * dim, dim, 1);
        qimg_fwd_state_t _st;
        _st.d_img = d_img; _st.d_txt = d_txt;
        _st.d_q = d_q; _st.d_k = d_k; _st.d_v = d_v; _st.d_attn_out = d_attn_out;
        _st.d_scratch1 = d_scratch1; _st.d_scratch2 = d_scratch2; _st.d_scratch3 = d_scratch3;
        _st.n_img = n_img; _st.n_txt = n_txt; _st.n_total = n_total;
        qimg_forward_block(r, &_st, &blk, L, d_img_mod, d_txt_mod,
                           hp_rope, wp_rope, t_dim_rope, h_dim_rope, w_dim_rope, rope_theta,
                           (CUdeviceptr)0);

        /* Save every block output for comparison */
        if (r->verbose >= 3) {
            char _fn[64];
            snprintf(_fn, sizeof(_fn), "cuda_dit_block%02d_img.npy", L);
            DIT_SAVE_NPY(_fn, d_img, n_img, dim);
            snprintf(_fn, sizeof(_fn), "cuda_dit_block%02d_txt.npy", L);
            DIT_SAVE_NPY(_fn, d_txt, n_txt, dim);
        }

        /* If we just consumed a slot, release it and prefetch block L+2
         * into it on copy_stream. */
        if (use_pipeline && slot_used >= 0) {
            cuEventRecord(r->slot_free[slot_used], s);
            int next_L = L + 2;
            if (next_L < r->dit_n_blocks && next_L >= r->n_preloaded) {
                cuStreamWaitEvent(r->copy_stream, r->slot_free[slot_used], 0);
                qimg_block_gpu tmp;
                qimg_load_block_into_slot_ex(r, next_L, slot_ptrs[slot_used], &tmp,
                                             r->copy_stream);
                cuEventRecord(r->slot_ready[slot_used], r->copy_stream);
            }
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

    if (r->profile_block && r->prof_block_count > 0) {
        double n = (double)r->prof_block_count;
        double tot = r->prof_qkv_ms + r->prof_qknorm_ms + r->prof_attn_ms
                   + r->prof_attnout_ms + r->prof_imgmlp_ms + r->prof_txtmlp_ms;
        fprintf(stderr,
            "cuda_qimg: per-block profile (%d block-passes, ms/block)\n"
            "  adaLN1+QKV    : %7.3f  (%5.1f%%)\n"
            "  QK-norm+RoPE  : %7.3f  (%5.1f%%)\n"
            "  attention     : %7.3f  (%5.1f%%)\n"
            "  attn-out+gate : %7.3f  (%5.1f%%)\n"
            "  img-MLP       : %7.3f  (%5.1f%%)\n"
            "  txt-MLP       : %7.3f  (%5.1f%%)\n"
            "  total/block   : %7.3f  (%5.3f s sum across blocks, includes sync overhead)\n",
            r->prof_block_count,
            r->prof_qkv_ms/n,    100.0 * r->prof_qkv_ms    / (tot > 0 ? tot : 1),
            r->prof_qknorm_ms/n, 100.0 * r->prof_qknorm_ms / (tot > 0 ? tot : 1),
            r->prof_attn_ms/n,   100.0 * r->prof_attn_ms   / (tot > 0 ? tot : 1),
            r->prof_attnout_ms/n,100.0 * r->prof_attnout_ms/ (tot > 0 ? tot : 1),
            r->prof_imgmlp_ms/n, 100.0 * r->prof_imgmlp_ms / (tot > 0 ? tot : 1),
            r->prof_txtmlp_ms/n, 100.0 * r->prof_txtmlp_ms / (tot > 0 ? tot : 1),
            tot/n, tot/1000.0);
        r->prof_qkv_ms = r->prof_qknorm_ms = r->prof_attn_ms = 0;
        r->prof_attnout_ms = r->prof_imgmlp_ms = r->prof_txtmlp_ms = 0;
        r->prof_block_count = 0;
    }
    return 0;
}

/* Evict preloaded DiT blocks from the end until `need_bytes` of VRAM is
 * free (plus a safety margin). Freed blocks fall back to the on-demand
 * loader via scratch_block / scratch_block_b. Called at dit_step entry so
 * high-resolution activation buffers always have room. Returns how many
 * blocks were evicted. */
static int qimg_evict_preloaded_until_free(cuda_qimg_runner *r,
                                            size_t need_bytes,
                                            size_t safety_margin_bytes) {
    size_t free_mem = 0, total_mem = 0;
    cuMemGetInfo(&free_mem, &total_mem);
    size_t want = need_bytes + safety_margin_bytes;
    if (free_mem >= want) return 0;

    int evicted = 0;
    /* Free from the tail so prefetch logic stays happy (low-index blocks
     * remain resident; high-index ones use the on-demand scratch slot). */
    while (r->n_preloaded > 0 && free_mem < want) {
        int idx = r->n_preloaded - 1;
        if (r->gpu_blocks[idx].attn_q_w) {
            qimg_free_block(&r->gpu_blocks[idx]);
        }
        r->n_preloaded--;
        evicted++;
        cuMemGetInfo(&free_mem, &total_mem);
    }
    if (evicted > 0 && r->verbose) {
        fprintf(stderr, "cuda_qimg: evicted %d preloaded blocks for activation "
                        "workspace (now %d preloaded, %.1f GB free)\n",
                evicted, r->n_preloaded, (float)free_mem / (1<<30));
    }
    /* If we freed all preloaded blocks, make sure the on-demand scratch
     * slot exists so the loader has somewhere to stage. */
    if (r->n_preloaded < r->dit_n_blocks &&
        r->scratch_block.attn_q_w == 0) {
        if (qimg_alloc_scratch_block(r) == 0 && r->verbose)
            fprintf(stderr, "cuda_qimg: on-demand scratch slot allocated\n");
    }
    return evicted;
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

    /* Evict preloaded DiT blocks if the transient activation working set
     * would leave us short of VRAM. This protects 1024x1024 and larger
     * runs where the static preload sized for 512x512 no longer fits
     * the 2*n_img MLP buffers + per-pass QKV/attn_out/scratch. Numbers
     * mirror the cuMemAlloc calls below. */
    {
        size_t dim_b  = (size_t)dim   * sizeof(float);
        size_t mlp_b  = (size_t)mlp_h * sizeof(float);
        size_t need = 0;
        need += (size_t)n_img          * dim_b * 2;                          /* d_img_c,u */
        need += (size_t)(n_txt_cond + n_txt_uncond) * dim_b;                 /* d_txt_c,u */
        need += (size_t)n_img          * in_ch * sizeof(float);              /* d_img_in */
        need += (size_t)(n_txt_cond + n_txt_uncond) * txt_dim * sizeof(float);
        need += (size_t)n_img          * dim_b;                              /* d_scratch1_c */
        need += (size_t)n_txt_cond     * dim_b;                              /* d_scratch2_c */
        need += (size_t)n_txt_cond     * mlp_b;                              /* d_scratch3_c */
        need += (size_t)n_img          * dim_b;                              /* d_scratch1_u */
        need += (size_t)n_txt_uncond   * dim_b;                              /* d_scratch2_u */
        need += (size_t)n_txt_uncond   * mlp_b;                              /* d_scratch3_u */
        need += (size_t)n_total_cond   * dim_b * 4;                          /* d_q/k/v/attn_out cond */
        need += (size_t)n_total_uncond * dim_b * 4;                          /* d_q/k/v/attn_out uncond */
        need += (size_t)2 * n_img * dim_b;                                   /* d_img_mlp_in */
        need += (size_t)2 * n_img * mlp_b;                                   /* d_img_mlp_h */
        need += (size_t)2 * n_img * dim_b;                                   /* d_img_mlp_out */
        need += 32 * sizeof(float) * 6;  /* modulation buffers, misc */
        /* Safety margin of 256 MB. Empirically, when free VRAM after the
         * working-set alloc drops below ~100 MB the CUDA driver starts
         * doing heavy per-launch bookkeeping (~7x per-kernel slowdown on
         * first step at 1328x1328). Keeping >=256 MB headroom avoids it. */
        qimg_evict_preloaded_until_free(r, need, (size_t)256 << 20);
    }

    int alloc_timing = (getenv("QIMG_ALLOC_TIMING") != NULL);
    double alloc_t0 = 0;
    if (alloc_timing) { cuStreamSynchronize(s); alloc_t0 = (double)clock()/CLOCKS_PER_SEC; }

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

    /* Scratch buffers. scratch1 only ever holds n_img rows (img adaLN output
     * / attn_out). scratch2 only ever holds n_txt rows (text adaLN output /
     * attn_out / MLP output). scratch3 only holds the text MLP intermediate
     * of size (n_txt * mlp_h) — the img MLP now runs CFG-batched outside
     * forward_block so it no longer needs scratch3. */
    int n_txt_max = n_txt_cond > n_txt_uncond ? n_txt_cond : n_txt_uncond;
    size_t img_scratch = (size_t)n_img     * dim   * sizeof(float);
    size_t txt_scratch = (size_t)n_txt_max * dim   * sizeof(float);
    size_t txt_ffn     = (size_t)n_txt_max * mlp_h * sizeof(float);
    CUdeviceptr d_scratch1_c, d_scratch2_c, d_scratch3_c;
    CUdeviceptr d_scratch1_u, d_scratch2_u, d_scratch3_u;
    cuMemAlloc(&d_scratch1_c, img_scratch);
    cuMemAlloc(&d_scratch2_c, txt_scratch);
    cuMemAlloc(&d_scratch3_c, txt_ffn);
    cuMemAlloc(&d_scratch1_u, img_scratch);
    cuMemAlloc(&d_scratch2_u, txt_scratch);
    cuMemAlloc(&d_scratch3_u, txt_ffn);
    (void)n_total_max;

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

    /* CFG-batched img-MLP buffers: cond/uncond adaLN2 output is packed into
     * the first / second half of d_img_mlp_in, then one batched fc1+gelu+fc2
     * runs on 2*n_img rows, halving W traffic for the two heaviest GEMMs in
     * the block (mlp_h=14336 x dim=3072 each direction). Post-MLP gated_add
     * is then dispatched per-pass by reading from d_img_mlp_out halves. */
    CUdeviceptr d_img_mlp_in, d_img_mlp_h, d_img_mlp_out;
    cuMemAlloc(&d_img_mlp_in,  (size_t)2 * n_img * dim   * sizeof(float));
    cuMemAlloc(&d_img_mlp_h,   (size_t)2 * n_img * mlp_h * sizeof(float));
    cuMemAlloc(&d_img_mlp_out, (size_t)2 * n_img * dim   * sizeof(float));
    CUdeviceptr d_img_mlp_in_c  = d_img_mlp_in;
    CUdeviceptr d_img_mlp_in_u  = d_img_mlp_in  + (size_t)n_img * dim * sizeof(float);
    CUdeviceptr d_img_mlp_out_c = d_img_mlp_out;
    CUdeviceptr d_img_mlp_out_u = d_img_mlp_out + (size_t)n_img * dim * sizeof(float);

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

    if (alloc_timing) {
        cuStreamSynchronize(s);
        double dt = (double)clock()/CLOCKS_PER_SEC - alloc_t0;
        size_t fm = 0, tm = 0; cuMemGetInfo(&fm, &tm);
        fprintf(stderr, "  [alloc] %6.3fs (free now: %.2f GB)\n", dt, (float)fm/(1<<30));
    }

    /* ---- 4. Block loop with optional double-buffered on-demand loading.
     *       OFF by default — see dit_step comment above. ---- */
    int use_pipeline = 0;
    if (getenv("QIMG_PIPELINE") && r->scratch_block_b_ready &&
        r->n_preloaded < r->dit_n_blocks) {
        use_pipeline = 1;
    }
    qimg_block_gpu *slot_ptrs[2] = { &r->scratch_block, &r->scratch_block_b };
    if (use_pipeline) {
        for (int i = 0; i < 2; i++) {
            int L0 = r->n_preloaded + i;
            if (L0 >= r->dit_n_blocks) break;
            qimg_block_gpu tmp;
            qimg_load_block_into_slot_ex(r, L0, slot_ptrs[i], &tmp, r->copy_stream);
            cuEventRecord(r->slot_ready[i], r->copy_stream);
        }
    }

    int vae_blk_timing = (getenv("QIMG_BLK_TIMING") != NULL);
    double blk_t0 = 0;
    for (int L = 0; L < r->dit_n_blocks; L++) {
        if (vae_blk_timing) { cuStreamSynchronize(s); blk_t0 = (double)clock()/CLOCKS_PER_SEC; }
        if (r->verbose && (L % 10 == 0 || L == r->dit_n_blocks - 1))
            fprintf(stderr, "\r  cuda_qimg: block %d/%d", L + 1, r->dit_n_blocks);

        qimg_block_gpu blk;
        int slot_used = -1;
        if (L < r->n_preloaded && r->gpu_blocks[L].attn_q_w) {
            blk = r->gpu_blocks[L];
        } else if (use_pipeline) {
            int od = L - r->n_preloaded;
            slot_used = od % 2;
            cuStreamWaitEvent(s, r->slot_ready[slot_used], 0);
            blk = *slot_ptrs[slot_used];
        } else {
            if (qimg_load_block_into_slot(r, L, &blk) != 0) {
                fprintf(stderr, "cuda_qimg: block %d slot load failed\n", L);
                return -1;
            }
        }

        /* Modulation (shared between cond/uncond) — always on r->stream */
        cuMemcpyDtoDAsync(d_t_silu, d_t_emb, (size_t)dim * sizeof(float), s);
        op_silu(r, d_t_silu, dim);
        op_gemm(r, d_img_mod, blk.img_mod_w, d_t_silu, blk.img_mod_b, 6 * dim, dim, 1);
        op_gemm(r, d_txt_mod, blk.txt_mod_w, d_t_silu, blk.txt_mod_b, 6 * dim, dim, 1);

        /* --- Cond pass (on r->stream). Skips img MLP; adaLN2 output goes
         * into the first half of d_img_mlp_in for the batched MLP below. --- */
        {
            qimg_fwd_state_t st_c;
            st_c.d_img = d_img_c; st_c.d_txt = d_txt_c;
            st_c.d_q = d_q_c; st_c.d_k = d_k_c; st_c.d_v = d_v_c; st_c.d_attn_out = d_attn_out_c;
            st_c.d_scratch1 = d_scratch1_c; st_c.d_scratch2 = d_scratch2_c; st_c.d_scratch3 = d_scratch3_c;
            st_c.n_img = n_img; st_c.n_txt = n_txt_cond; st_c.n_total = n_total_cond;
            qimg_forward_block(r, &st_c, &blk, L, d_img_mod, d_txt_mod,
                               hp_rope, wp_rope, t_dim_rope, h_dim_rope, w_dim_rope, rope_theta,
                               d_img_mlp_in_c);
        }
        /* --- Uncond pass. Same deal, into the second half of d_img_mlp_in. --- */
        int use_parallel = (r->uncond_stream != NULL);
        CUstream saved_stream = r->stream;
        if (use_parallel) {
            cuEventRecord(r->mod_ready, s);
            cuStreamWaitEvent(r->uncond_stream, r->mod_ready, 0);
            r->stream = r->uncond_stream;
        }
        {
            qimg_fwd_state_t st_u;
            st_u.d_img = d_img_u; st_u.d_txt = d_txt_u;
            st_u.d_q = d_q_u; st_u.d_k = d_k_u; st_u.d_v = d_v_u; st_u.d_attn_out = d_attn_out_u;
            st_u.d_scratch1 = d_scratch1_u; st_u.d_scratch2 = d_scratch2_u; st_u.d_scratch3 = d_scratch3_u;
            st_u.n_img = n_img; st_u.n_txt = n_txt_uncond; st_u.n_total = n_total_uncond;
            qimg_forward_block(r, &st_u, &blk, L, d_img_mod, d_txt_mod,
                               hp_rope, wp_rope, t_dim_rope, h_dim_rope, w_dim_rope, rope_theta,
                               d_img_mlp_in_u);
        }
        if (use_parallel) {
            /* Rejoin: next block's modulation depends on shared d_img_mod/
             * d_txt_mod, which is on saved_stream. Make the saved stream
             * wait for uncond_stream to finish this block's uncond work. */
            cuEventRecord(r->block_barrier_unc, r->uncond_stream);
            r->stream = saved_stream;
            cuStreamWaitEvent(r->stream, r->block_barrier_unc, 0);
        }

        /* --- CFG-batched img MLP. W is loaded once for both cond and uncond. --- */
        {
            int two_n_img = 2 * n_img;
            op_gemm_gelu(r, d_img_mlp_h, blk.img_mlp_fc1_w, d_img_mlp_in, blk.img_mlp_fc1_b,
                         mlp_h, dim, two_n_img);
            op_gemm(r, d_img_mlp_out, blk.img_mlp_fc2_w, d_img_mlp_h, blk.img_mlp_fc2_b,
                    dim, mlp_h, two_n_img);
            CUdeviceptr img_g2 = d_img_mod + (size_t)5 * dim * sizeof(float);
            op_gated_add(r, d_img_c, d_img_mlp_out_c, img_g2, n_img, dim);
            op_gated_add(r, d_img_u, d_img_mlp_out_u, img_g2, n_img, dim);
        }

        /* If we consumed a double-buffer slot, release it and prefetch L+2. */
        if (use_pipeline && slot_used >= 0) {
            cuEventRecord(r->slot_free[slot_used], s);
            int next_L = L + 2;
            if (next_L < r->dit_n_blocks && next_L >= r->n_preloaded) {
                cuStreamWaitEvent(r->copy_stream, r->slot_free[slot_used], 0);
                qimg_block_gpu tmp;
                qimg_load_block_into_slot_ex(r, next_L, slot_ptrs[slot_used], &tmp,
                                             r->copy_stream);
                cuEventRecord(r->slot_ready[slot_used], r->copy_stream);
            }
        }
        if (vae_blk_timing) {
            cuStreamSynchronize(s);
            double dt = (double)clock()/CLOCKS_PER_SEC - blk_t0;
            fprintf(stderr, "  [blk %2d] %.3fs%s\n", L, dt,
                    L < r->n_preloaded ? "" : " (on-demand)");
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
    cuMemFree(d_img_mlp_in); cuMemFree(d_img_mlp_h); cuMemFree(d_img_mlp_out);
    cuMemFree(d_t_silu); cuMemFree(d_img_mod); cuMemFree(d_txt_mod);

    if (r->profile_block && r->prof_block_count > 0) {
        double n = (double)r->prof_block_count;
        double tot = r->prof_qkv_ms + r->prof_qknorm_ms + r->prof_attn_ms
                   + r->prof_attnout_ms + r->prof_imgmlp_ms + r->prof_txtmlp_ms;
        fprintf(stderr,
            "cuda_qimg(cfg): per-block profile (%d block-passes, ms/block)\n"
            "  adaLN1+QKV    : %7.3f  (%5.1f%%)\n"
            "  QK-norm+RoPE  : %7.3f  (%5.1f%%)\n"
            "  attention     : %7.3f  (%5.1f%%)\n"
            "  attn-out+gate : %7.3f  (%5.1f%%)\n"
            "  img-MLP       : %7.3f  (%5.1f%%)\n"
            "  txt-MLP       : %7.3f  (%5.1f%%)\n"
            "  total/block   : %7.3f  (%5.3f s sum across blocks, includes sync overhead)\n",
            r->prof_block_count,
            r->prof_qkv_ms/n,    100.0 * r->prof_qkv_ms    / (tot > 0 ? tot : 1),
            r->prof_qknorm_ms/n, 100.0 * r->prof_qknorm_ms / (tot > 0 ? tot : 1),
            r->prof_attn_ms/n,   100.0 * r->prof_attn_ms   / (tot > 0 ? tot : 1),
            r->prof_attnout_ms/n,100.0 * r->prof_attnout_ms/ (tot > 0 ? tot : 1),
            r->prof_imgmlp_ms/n, 100.0 * r->prof_imgmlp_ms / (tot > 0 ? tot : 1),
            r->prof_txtmlp_ms/n, 100.0 * r->prof_txtmlp_ms / (tot > 0 ? tot : 1),
            tot/n, tot/1000.0);
        r->prof_qkv_ms = r->prof_qknorm_ms = r->prof_attn_ms = 0;
        r->prof_attnout_ms = r->prof_imgmlp_ms = r->prof_txtmlp_ms = 0;
        r->prof_block_count = 0;
    }
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

/* GPU VAE conv2d launch. When shapes match the FP8 MMA GEMM's constraints
 * (kh==kw>=2, ci*kh*kw % 32 == 0, n_tok >= 16) we route through the tensor-
 * core im2col + GEMM + transpose path. Otherwise fall back to the naive
 * per-output-thread kernel. */
static void vae_op_conv2d_mma(cuda_qimg_runner *r,
                              CUdeviceptr out, CUdeviceptr inp,
                              CUdeviceptr w_fp8, CUdeviceptr bias,
                              int ci, int h, int w_s, int co, int kh, int kw,
                              int rep_pad, int pad_co, int n_in_pad);
static void vae_op_conv2d(cuda_qimg_runner *r, CUdeviceptr out, CUdeviceptr inp,
                          CUdeviceptr w, CUdeviceptr b,
                          int ci, int h, int w_s, int co, int kh, int kw, int rep_pad) {
    int n_tok = h * w_s;
    int n_in = ci * kh * kw;
    int n_in_pad = (n_in + 31) / 32 * 32;
    int pad_co = (co + 255) / 256 * 256;
    if (r->vae_im2col && r->vae_f32_to_fp8_padded && r->vae_crop_transpose_add_bias &&
        kh >= 2 && kw >= 2 && (n_in % 32) == 0 && n_tok >= 16 &&
        r->gemm_fp8_pipe_perrow && r->use_fp8_pipe_perrow) {
        /* Quantize the (already uploaded) F32 weight to padded FP8 on the fly. */
        CUdeviceptr d_w_fp8;
        if (cuMemAlloc(&d_w_fp8, (size_t)pad_co * n_in_pad) == CUDA_SUCCESS) {
            unsigned bx = 32, by = 8;
            unsigned gx = (unsigned)((n_in_pad + bx - 1) / bx);
            unsigned gy = (unsigned)((pad_co   + by - 1) / by);
            void *args[] = {&d_w_fp8, &w, &co, &n_in, &pad_co, &n_in_pad};
            cuLaunchKernel(r->vae_f32_to_fp8_padded, gx, gy, 1, bx, by, 1,
                           0, r->stream, args, NULL);
            vae_op_conv2d_mma(r, out, inp, d_w_fp8, b, ci, h, w_s, co,
                              kh, kw, rep_pad, pad_co, n_in_pad);
            cuMemFree(d_w_fp8);
            return;
        }
    }
    int total = co * h * w_s;
    void *args[] = {&out, &inp, &w, &b, &ci, &h, &w_s, &co, &kh, &kw, &rep_pad};
    cuLaunchKernel(r->vae_conv2d, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

/* Tensor-core conv2d path: tiled im2col the F32 input, run the existing
 * FP8 per-row MMA GEMM on pre-quantized FP8 weights, then crop/transpose
 * + bias-add each chunk into the [co, H*W] CHW output. Chunking the n_tok
 * dimension keeps peak VRAM bounded so 1024x1024 (1 M tokens) fits in a
 * 16 GB (and usually 8 GB) budget. */
static void vae_op_conv2d_mma(cuda_qimg_runner *r,
                              CUdeviceptr out, CUdeviceptr inp,
                              CUdeviceptr w_fp8, CUdeviceptr bias,
                              int ci, int h, int w_s, int co, int kh, int kw,
                              int rep_pad, int pad_co, int n_in_pad) {
    int n_tok = h * w_s;
    CUstream s = r->stream;

    /* Cap each chunk so the unfold buffer stays under ~256 MB.
     * unfold_bytes = chunk_n_tok * n_in_pad * sizeof(float) */
    int chunk_n_tok = n_tok;
    const size_t UNFOLD_CAP = (size_t)256 << 20;  /* 256 MB per chunk */
    if (n_in_pad > 0) {
        size_t max_chunk = UNFOLD_CAP / ((size_t)n_in_pad * sizeof(float));
        if (max_chunk < 1) max_chunk = 1;
        /* Round chunk_n_tok down to a multiple of 64 so MT4 GEMM applies
         * inside the inner dispatch. Last chunk can be smaller. */
        size_t aligned = max_chunk & ~(size_t)63;
        if (aligned == 0) aligned = max_chunk;
        if (aligned < (size_t)n_tok) chunk_n_tok = (int)aligned;
    }
    if (chunk_n_tok < 1) chunk_n_tok = 1;

    size_t unfold_bytes = (size_t)chunk_n_tok * n_in_pad * sizeof(float);
    size_t gemm_out_bytes = (size_t)chunk_n_tok * pad_co   * sizeof(float);

    CUdeviceptr d_unfold, d_gemm_out;
    if (cuMemAlloc(&d_unfold,   unfold_bytes)   != CUDA_SUCCESS ||
        cuMemAlloc(&d_gemm_out, gemm_out_bytes) != CUDA_SUCCESS) {
        /* Fall back to naive conv2d on OOM. */
        fprintf(stderr, "cuda_qimg_vae: mma conv OOM (chunk=%d, n_in_pad=%d), "
                        "falling back to naive\n", chunk_n_tok, n_in_pad);
        int total = co * n_tok;
        void *args[] = {&out, &inp, &w_fp8, &bias, &ci, &h, &w_s, &co, &kh, &kw, &rep_pad};
        cuLaunchKernel(r->vae_conv2d, (unsigned)((total + 255) / 256), 1, 1,
                       256, 1, 1, 0, s, args, NULL);
        return;
    }

    for (int tok0 = 0; tok0 < n_tok; tok0 += chunk_n_tok) {
        int tok1 = tok0 + chunk_n_tok;
        if (tok1 > n_tok) tok1 = n_tok;
        int cnk = tok1 - tok0;

        /* im2col for [tok0, tok1) -> d_unfold[0:cnk, :] */
        {
            unsigned bx = 32, by = 8;
            unsigned gx = (unsigned)((n_in_pad + bx - 1) / bx);
            unsigned gy = (unsigned)((cnk      + by - 1) / by);
            void *args[] = {&d_unfold, &inp, &ci, &h, &w_s, &kh, &kw, &rep_pad,
                            &n_in_pad, &tok0, &cnk};
            cuLaunchKernel(r->vae_im2col, gx, gy, 1, bx, by, 1, 0, s, args, NULL);
        }

        /* GEMM: [cnk, pad_co] = unfold[cnk, n_in_pad] * W_fp8[pad_co, n_in_pad]^T
         * (op_gemm semantics). */
        op_gemm(r, d_gemm_out, w_fp8, d_unfold, (CUdeviceptr)0, pad_co, n_in_pad, cnk);

        /* Crop + transpose into out[c, tok0..tok1). */
        {
            unsigned bx = 32, by = 8;
            unsigned gx = (unsigned)((cnk + bx - 1) / bx);
            unsigned gy = (unsigned)((co  + by - 1) / by);
            void *args[] = {&out, &d_gemm_out, &bias, &co, &pad_co, &n_tok, &tok0, &cnk};
            cuLaunchKernel(r->vae_crop_transpose_add_bias, gx, gy, 1, bx, by, 1,
                           0, s, args, NULL);
        }
    }

    cuMemFree(d_unfold);
    cuMemFree(d_gemm_out);
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

    /* Per-phase wall-clock timer driven by QIMG_VAE_TIMING=1 */
    int vae_timing = (getenv("QIMG_VAE_TIMING") != NULL);
    double vae_phase_t0 = 0;
    #define VAE_PHASE_BEGIN() do { if (vae_timing) { cuStreamSynchronize(s); vae_phase_t0 = (double)clock()/CLOCKS_PER_SEC; } } while (0)
    #define VAE_PHASE_END(label) do { if (vae_timing) { cuStreamSynchronize(s); \
        double _dt = (double)clock()/CLOCKS_PER_SEC - vae_phase_t0; \
        fprintf(stderr, "  [vae] %-24s %6.3fs\n", label, _dt); } } while (0)

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
    VAE_PHASE_BEGIN();
    {
        CUdeviceptr d_tmp; cuMemAlloc(&d_tmp, (size_t)c*h*w*sizeof(float));
        vae_op_conv2d(r, d_tmp, d_x, d_c1_w, d_c1_b, ci_c1, h, w, c, 3, 3, 0);
        vae_bf16(r, d_tmp, c*h*w);
        cuMemFree(d_x); d_x = d_tmp;
        cuMemFree(d_c1_w); cuMemFree(d_c1_b);
    }
    VAE_PHASE_END("conv1");
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
    VAE_PHASE_BEGIN();
    { LOAD_RB_NAMED("decoder.middle.0", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
      CUdeviceptr d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b, scw, scb, c, c, h, w);
      cuMemFree(d_x); d_x = d_tmp;
      cuMemFree(n1); cuMemFree(c1w); cuMemFree(c1b); cuMemFree(n2); cuMemFree(c2w); cuMemFree(c2b);
      if (scw) { cuMemFree(scw); } if (scb) { cuMemFree(scb); } }
    VAE_PHASE_END("mid.0 resblock");
    { int _d[] = {c, h, w}; VAE_SAVE_NPY("cuda_vae_middle_0.npy", d_x, 3, _d); }

    /* Middle attention: GroupNorm -> QKV conv1x1 -> spatial self-attention -> proj + residual.
     * Entirely on GPU: transposes [c,S] -> [S,c] for coalesced K/V row loads,
     * then runs the warp-per-query online-softmax kernel. */
    VAE_PHASE_BEGIN();
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

        /* QKV 1x1 conv: [3c, c] W times [c, spatial] x -> [3c, spatial] out (CHW). */
        CUdeviceptr d_qkv; cuMemAlloc(&d_qkv, (size_t)3*c*spatial*sizeof(float));
        vae_op_conv2d(r, d_qkv, d_normed, d_qkv_w, d_qkv_b, c, h, w, 3*c, 1, 1, 0);
        cuMemFree(d_normed); cuMemFree(d_qkv_w); cuMemFree(d_qkv_b);

        /* Transpose each [c, spatial] slice -> [spatial, c] (row-major). */
        CUdeviceptr d_Q_sc, d_K_sc, d_V_sc;
        cuMemAlloc(&d_Q_sc, (size_t)spatial*c*sizeof(float));
        cuMemAlloc(&d_K_sc, (size_t)spatial*c*sizeof(float));
        cuMemAlloc(&d_V_sc, (size_t)spatial*c*sizeof(float));
        {
            CUdeviceptr d_Q_chw = d_qkv;
            CUdeviceptr d_K_chw = d_qkv + (size_t)c * spatial * sizeof(float);
            CUdeviceptr d_V_chw = d_qkv + (size_t)2 * c * spatial * sizeof(float);
            unsigned bx = 16, by = 16;
            unsigned gx = (unsigned)((spatial + 15) / 16);
            unsigned gy = (unsigned)((c + 15) / 16);
            int sp = spatial;
            void *tq[] = {&d_Q_sc, &d_Q_chw, &c, &sp};
            void *tk[] = {&d_K_sc, &d_K_chw, &c, &sp};
            void *tv[] = {&d_V_sc, &d_V_chw, &c, &sp};
            cuLaunchKernel(r->vae_transpose_chw_to_sc, gx, gy, 1, bx, by, 1, 0, s, tq, NULL);
            cuLaunchKernel(r->vae_transpose_chw_to_sc, gx, gy, 1, bx, by, 1, 0, s, tk, NULL);
            cuLaunchKernel(r->vae_transpose_chw_to_sc, gx, gy, 1, bx, by, 1, 0, s, tv, NULL);
        }
        cuMemFree(d_qkv);

        /* Self-attention: one CTA per query, 1 warp, online softmax. */
        CUdeviceptr d_attn_sc; cuMemAlloc(&d_attn_sc, (size_t)spatial*c*sizeof(float));
        {
            float scale_at = 1.0f / sqrtf((float)c);
            int sp = spatial;
            size_t smem_bytes = (size_t)2 * c * sizeof(float);  /* smQ + smO */
            void *args[] = {&d_attn_sc, &d_Q_sc, &d_K_sc, &d_V_sc, &sp, &c, &scale_at};
            cuLaunchKernel(r->vae_attn_sc, (unsigned)spatial, 1, 1,
                           32, 1, 1, smem_bytes, s, args, NULL);
        }
        cuMemFree(d_Q_sc); cuMemFree(d_K_sc); cuMemFree(d_V_sc);

        /* Transpose attn output [spatial, c] -> [c, spatial] for the 1x1 proj. */
        CUdeviceptr d_attn_chw; cuMemAlloc(&d_attn_chw, (size_t)c*spatial*sizeof(float));
        {
            unsigned bx = 16, by = 16;
            unsigned gx = (unsigned)((spatial + 15) / 16);
            unsigned gy = (unsigned)((c + 15) / 16);
            int sp = spatial;
            void *args[] = {&d_attn_chw, &d_attn_sc, &c, &sp};
            cuLaunchKernel(r->vae_transpose_sc_to_chw, gx, gy, 1, bx, by, 1, 0, s, args, NULL);
        }
        cuMemFree(d_attn_sc);

        /* proj 1x1 conv + residual add. */
        CUdeviceptr d_proj_out; cuMemAlloc(&d_proj_out, (size_t)c*spatial*sizeof(float));
        vae_op_conv2d(r, d_proj_out, d_attn_chw, d_proj_w, d_proj_b, c, h, w, c, 1, 1, 0);
        cuMemFree(d_attn_chw); cuMemFree(d_proj_w); cuMemFree(d_proj_b);

        {
            int n = c * spatial;
            float one = 1.0f;
            void *add_args[] = {&d_x, &d_proj_out, &one, &n};
            cuLaunchKernel(r->euler_step, (unsigned)((n+255)/256), 1, 1,
                           256, 1, 1, 0, s, add_args, NULL);
        }
        cuMemFree(d_proj_out);
        vae_bf16(r, d_x, c * spatial);
    }
    VAE_PHASE_END("mid.1 attention(GPU)");
    { int _d[] = {c, h, w}; VAE_SAVE_NPY("cuda_vae_middle_1.npy", d_x, 3, _d); }

    /* mid.2 */
    VAE_PHASE_BEGIN();
    { LOAD_RB_NAMED("decoder.middle.2", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
      CUdeviceptr d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b, scw, scb, c, c, h, w);
      cuMemFree(d_x); d_x = d_tmp;
      cuMemFree(n1); cuMemFree(c1w); cuMemFree(c1b); cuMemFree(n2); cuMemFree(c2w); cuMemFree(c2b);
      if (scw) { cuMemFree(scw); } if (scb) { cuMemFree(scb); } }
    VAE_PHASE_END("mid.2 resblock");
    fprintf(stderr, "  after middle: [%d, %d, %d]\n", c, h, w);
    VAE_DUMP("middle_out", d_x, c*h*w);
    { int _d[] = {c, h, w}; VAE_SAVE_NPY("cuda_vae_middle_2.npy", d_x, 3, _d); }

    /* Upsample blocks 0-14 */
    VAE_PHASE_BEGIN();
    int last_res_h = h, last_res_w = w;
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

        /* Per-stage timing: resolution changed means we started a new stage */
        if (vae_timing && (h != last_res_h || w != last_res_w)) {
            char _lbl2[32];
            snprintf(_lbl2, sizeof(_lbl2), "upsamples @ %dx%d", last_res_h, last_res_w);
            VAE_PHASE_END(_lbl2);
            VAE_PHASE_BEGIN();
            last_res_h = h; last_res_w = w;
        }
    }
    if (vae_timing) {
        char _lbl2[32];
        snprintf(_lbl2, sizeof(_lbl2), "upsamples @ %dx%d", last_res_h, last_res_w);
        VAE_PHASE_END(_lbl2);
    }
    #undef LOAD_RB_NAMED

    /* Head: GroupNorm → SiLU → Conv(96→3) */
    VAE_PHASE_BEGIN();
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

    VAE_PHASE_END("head");

    /* Download result (sync stream first to ensure all GPU ops complete) */
    cuStreamSynchronize(s);
    cuMemcpyDtoH(out_rgb, d_x, (size_t)3 * h * w * sizeof(float));
    cuMemFree(d_x);

    fprintf(stderr, "cuda_qimg_vae: decode complete [%d, %d, %d]\n", c, h, w);
    return 0;
    #undef VAE_PHASE_BEGIN
    #undef VAE_PHASE_END
}

#endif /* CUDA_QIMG_RUNNER_IMPLEMENTATION */
#endif /* CUDA_QIMG_RUNNER_H */
