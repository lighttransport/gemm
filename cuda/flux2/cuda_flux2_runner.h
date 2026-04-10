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
    CUdeviceptr qkv_w;     /* [3H, H] F16 */
    CUdeviceptr proj_w;    /* [H, H] F16 */
    CUdeviceptr mlp_up_w;  /* [2*n_ff, H] F16 */
    CUdeviceptr mlp_dn_w;  /* [H, n_ff] F16 */
    CUdeviceptr q_norm;    /* [head_dim] F32 */
    CUdeviceptr k_norm;    /* [head_dim] F32 */
} flux2_gpu_stream_t;

typedef struct {
    flux2_gpu_stream_t img, txt;
} flux2_gpu_dblk_t;

typedef struct {
    CUdeviceptr linear1_w;   /* [3H+2*n_ff, H] F16 */
    CUdeviceptr l2_attn_w;   /* [H, H] F16 — first H cols of linear2 */
    CUdeviceptr l2_mlp_w;    /* [H, n_ff] F16 — last n_ff cols of linear2 */
    CUdeviceptr q_norm;      /* [head_dim] F32 */
    CUdeviceptr k_norm;      /* [head_dim] F32 */
} flux2_gpu_sblk_t;

/* ---- Runner struct ---- */

struct cuda_flux2_runner {
    CUdevice   device;
    CUcontext  ctx;
    CUstream   stream;
    int        verbose;

    CUmodule   mod;
    CUfunction fn_gemm, fn_layernorm, fn_silu, fn_adaln, fn_gated_add;
    CUfunction fn_rmsnorm_ph, fn_swiglu, fn_flash_attn, fn_add;
    CUfunction fn_vae_groupnorm, fn_vae_conv2d, fn_vae_upsample2x, fn_vae_latent_bn;
    CUfunction fn_rope_img, fn_rope_txt, fn_bf16_trunc;

    /* CPU model (kept for arch params + VAE fallback) */
    flux2_dit_model *dit;
    flux2_vae_model *vae;

    int H, nH, hd, n_ff, pin, txt_dim, n_dbl, n_sgl;

    /* Global GPU weights */
    CUdeviceptr d_img_in_w, d_img_in_b, d_txt_in_w, d_txt_in_b;
    CUdeviceptr d_t_fc1_w, d_t_fc1_b, d_t_fc2_w, d_t_fc2_b;
    CUdeviceptr d_mod_img_w, d_mod_txt_w, d_mod_sgl_w;
    CUdeviceptr d_out_mod_w, d_out_proj_w;

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

/* Upload F32 array as F16 to GPU, returns device pointer */
static CUdeviceptr gpu_upload_f16(const float *data, int n) {
    uint16_t *tmp = (uint16_t *)malloc((size_t)n * 2);
    for (int i = 0; i < n; i++) tmp[i] = f2h(data[i]);
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

/* Upload flux2_mat (weight matrix) as F32 */
static CUdeviceptr gpu_upload_mat(const flux2_mat *m) {
    return gpu_upload_f32(m->w, m->rows * m->cols);
}

/* ---- Op functions ---- */

static void op_gemm(cuda_flux2_runner *r, CUdeviceptr Y, CUdeviceptr W,
                    CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    cuLaunchKernel(r->fn_gemm, gx, gy, 1, 16, 16, 1, 0, r->stream, args, NULL);
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

    cuStreamCreate(&r->stream, 0);

    /* Get function handles */
    cuModuleGetFunction(&r->fn_gemm,        mod, "gemm_f32_f32");
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
    cuModuleGetFunction(&r->fn_rope_img,    mod, "flux2_rope_img_f32");
    cuModuleGetFunction(&r->fn_rope_txt,    mod, "flux2_rope_txt_f32");
    cuModuleGetFunction(&r->fn_bf16_trunc,  mod, "truncate_bf16_f32");

    if (verbose >= 1) {
        fprintf(stderr, "cuda_flux2: init OK");
        if (r->use_gpu_dbl_attn) fprintf(stderr, " [gpu-dbl-attn]");
        if (r->debug_dbl_attn) fprintf(stderr, " [debug-dbl-attn]");
        fprintf(stderr, "\n");
    }
    return r;
}

/* Upload stream block weights to GPU */
static void upload_stream(flux2_gpu_stream_t *gs, const flux2_stream_block *sb, int hd) {
    gs->qkv_w    = gpu_upload_mat(&sb->qkv);
    gs->proj_w   = gpu_upload_mat(&sb->proj);
    gs->mlp_up_w = gpu_upload_mat(&sb->mlp_up);
    gs->mlp_dn_w = gpu_upload_mat(&sb->mlp_down);
    gs->q_norm   = gpu_upload_f32(sb->q_norm, hd);
    gs->k_norm   = gpu_upload_f32(sb->k_norm, hd);
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

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_flux2: uploading weights to GPU (H=%d, %d+%d blocks)...\n",
                r->H, r->n_dbl, r->n_sgl);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Global weights */
    r->d_img_in_w  = gpu_upload_mat(&m->img_in);
    r->d_img_in_b  = gpu_upload_f32_or0(m->img_in_b, r->H);
    r->d_txt_in_w  = gpu_upload_mat(&m->txt_in);
    r->d_txt_in_b  = gpu_upload_f32_or0(m->txt_in_b, r->H);
    r->d_t_fc1_w   = gpu_upload_mat(&m->time_in_lin1);
    r->d_t_fc1_b   = gpu_upload_f32_or0(m->time_in_lin1_b, r->H);
    r->d_t_fc2_w   = gpu_upload_mat(&m->time_in_lin2);
    r->d_t_fc2_b   = gpu_upload_f32_or0(m->time_in_lin2_b, r->H);
    r->d_mod_img_w = gpu_upload_mat(&m->mod_img);
    r->d_mod_txt_w = gpu_upload_mat(&m->mod_txt);
    r->d_mod_sgl_w = gpu_upload_mat(&m->mod_sgl);
    r->d_out_mod_w = gpu_upload_mat(&m->out_mod);
    r->d_out_proj_w= gpu_upload_mat(&m->out_proj);

    /* Double-stream block weights */
    r->gpu_dblk = (flux2_gpu_dblk_t *)calloc((size_t)r->n_dbl, sizeof(flux2_gpu_dblk_t));
    for (int i = 0; i < r->n_dbl; i++) {
        upload_stream(&r->gpu_dblk[i].img, &m->dblk[i].img, r->hd);
        upload_stream(&r->gpu_dblk[i].txt, &m->dblk[i].txt, r->hd);
        if (r->verbose >= 2)
            fprintf(stderr, "\r  double block %d/%d", i+1, r->n_dbl);
    }

    /* Single-stream block weights */
    r->gpu_sblk = (flux2_gpu_sblk_t *)calloc((size_t)r->n_sgl, sizeof(flux2_gpu_sblk_t));
    for (int i = 0; i < r->n_sgl; i++) {
        r->gpu_sblk[i].linear1_w = gpu_upload_mat(&m->sblk[i].linear1);
        /* linear2 is [H, H+n_ff]. Split columns: first H → attn, last n_ff → mlp */
        {
            const flux2_mat *l2 = &m->sblk[i].linear2;
            int Hd = r->H, nf = r->n_ff;
            int l2_in = l2->cols;  /* H + n_ff */
            float *attn = (float *)malloc((size_t)Hd * Hd * sizeof(float));
            float *mlp_ = (float *)malloc((size_t)Hd * nf * sizeof(float));
            for (int r2 = 0; r2 < Hd; r2++) {
                memcpy(attn + (size_t)r2 * Hd, l2->w + (size_t)r2 * l2_in, (size_t)Hd * sizeof(float));
                memcpy(mlp_ + (size_t)r2 * nf, l2->w + (size_t)r2 * l2_in + Hd, (size_t)nf * sizeof(float));
            }
            r->gpu_sblk[i].l2_attn_w = gpu_upload_f32(attn, Hd * Hd);
            r->gpu_sblk[i].l2_mlp_w  = gpu_upload_f32(mlp_, Hd * nf);
            free(attn); free(mlp_);
        }
        r->gpu_sblk[i].q_norm = gpu_upload_f32(m->sblk[i].q_norm, r->hd);
        r->gpu_sblk[i].k_norm = gpu_upload_f32(m->sblk[i].k_norm, r->hd);
        if (r->verbose >= 2)
            fprintf(stderr, "\r  single block %d/%d", i+1, r->n_sgl);
    }

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

    /* temb = SiLU(time_in_lin1 @ t_raw + b1) */
    op_gemm(r, r->d_temb, r->d_t_fc1_w, d_traw, r->d_t_fc1_b, H, 256, 1);
    op_silu(r, r->d_temb, H);
    /* temb = time_in_lin2 @ temb + b2 (reuse d_temb_silu as temp) */
    op_gemm(r, r->d_temb_silu, r->d_t_fc2_w, r->d_temb, r->d_t_fc2_b, H, H, 1);
    /* swap: temb_silu now has the final temb */
    CUdeviceptr d_temb_final = r->d_temb_silu;

    cuMemFree(d_traw);

    /* 2. Project img and txt tokens */
    op_gemm(r, r->d_img, r->d_img_in_w, r->d_img_in_buf, r->d_img_in_b,
            H, r->pin, n_img);
    op_gemm(r, r->d_txt, r->d_txt_in_w, r->d_txt_in_buf, r->d_txt_in_b,
            H, r->txt_dim, n_txt);

    /* 3. Compute global modulations: SiLU(temb) → mod vectors */
    cuMemcpyDtoD(r->d_temb, d_temb_final, (size_t)H * F);
    op_silu(r, r->d_temb, H);
    /* d_temb now = SiLU(temb_final) */
    CUdeviceptr d0 = 0;
    op_gemm(r, r->d_mod_img_v, r->d_mod_img_w, r->d_temb, d0, 6*H, H, 1);
    op_gemm(r, r->d_mod_txt_v, r->d_mod_txt_w, r->d_temb, d0, 6*H, H, 1);
    op_gemm(r, r->d_mod_sgl_v, r->d_mod_sgl_w, r->d_temb, d0, 3*H, H, 1);

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

        /* Q,K,V projections from img stream (use weight slices of [3H, H]) */
        CUdeviceptr img_q_w = b->img.qkv_w;
        CUdeviceptr img_k_w = b->img.qkv_w + (size_t)H * H * 4;  /* F32: 4 bytes */
        CUdeviceptr img_v_w = b->img.qkv_w + (size_t)2 * H * H * 4;
        op_gemm(r, r->d_q, img_q_w, r->d_scratch1, d0, H, H, n_img);
        op_gemm(r, r->d_k, img_k_w, r->d_scratch1, d0, H, H, n_img);
        op_gemm(r, r->d_v, img_v_w, r->d_scratch1, d0, H, H, n_img);

        /* TXT stream: adaLN → QKV */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_a, mt_scale_a, n_txt, H);
        CUdeviceptr txt_q_w = b->txt.qkv_w;
        CUdeviceptr txt_k_w = b->txt.qkv_w + (size_t)H * H * 4;
        CUdeviceptr txt_v_w = b->txt.qkv_w + (size_t)2 * H * H * 4;
        /* Write txt Q/K/V after img portion */
        CUdeviceptr tq = r->d_q + (size_t)n_img * H * F;
        CUdeviceptr tk = r->d_k + (size_t)n_img * H * F;
        CUdeviceptr tv = r->d_v + (size_t)n_img * H * F;
        op_gemm(r, tq, txt_q_w, r->d_scratch1, d0, H, H, n_txt);
        op_gemm(r, tk, txt_k_w, r->d_scratch1, d0, H, H, n_txt);
        op_gemm(r, tv, txt_v_w, r->d_scratch1, d0, H, H, n_txt);

        /* Per-head RMSNorm on Q, K (separate norms for img/txt) */
        op_rmsnorm_ph(r, r->d_q, b->img.q_norm, n_img, nH, hd);
        op_rmsnorm_ph(r, r->d_k, b->img.k_norm, n_img, nH, hd);
        op_rmsnorm_ph(r, tq, b->txt.q_norm, n_txt, nH, hd);
        op_rmsnorm_ph(r, tk, b->txt.k_norm, n_txt, nH, hd);

        /* RoPE: img=2D spatial, txt=1D sequence */
        op_rope_img(r, r->d_q, n_img, nH, hd, lat_w_p, theta);
        op_rope_img(r, r->d_k, n_img, nH, hd, lat_w_p, theta);
        op_rope_txt(r, tq, n_txt, nH, hd, theta);
        op_rope_txt(r, tk, n_txt, nH, hd, theta);
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

        /* Output projections (img and txt use different weights) */
        op_gemm(r, r->d_scratch1, b->img.proj_w, r->d_attn_out, d0, H, H, n_img);
        CUdeviceptr txt_attn = r->d_attn_out + (size_t)n_img * H * F;
        op_gemm(r, r->d_scratch2, b->txt.proj_w, txt_attn, d0, H, H, n_txt);

        /* Gated residual */
        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_a, n_img, H);
        op_gated_add(r, r->d_txt, r->d_scratch2, mt_gate_a, n_txt, H);

        /* FFN img */
        op_adaln(r, r->d_scratch1, r->d_img, mi_shift_f, mi_scale_f, n_img, H);
        op_gemm(r, r->d_scratch2, b->img.mlp_up_w, r->d_scratch1, d0, 2*n_ff, H, n_img);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_img, n_ff);
        op_gemm(r, r->d_scratch1, b->img.mlp_dn_w, r->d_scratch3, d0, H, n_ff, n_img);
        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_f, n_img, H);

        /* FFN txt */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_f, mt_scale_f, n_txt, H);
        op_gemm(r, r->d_scratch2, b->txt.mlp_up_w, r->d_scratch1, d0, 2*n_ff, H, n_txt);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_txt, n_ff);
        op_gemm(r, r->d_scratch1, b->txt.mlp_dn_w, r->d_scratch3, d0, H, n_ff, n_txt);
        op_gated_add(r, r->d_txt, r->d_scratch1, mt_gate_f, n_txt, H);
    }

    /* ---- Single-stream blocks ---- */
    /* Concatenate: joint = [txt, img] (txt first, then img) */
    cuMemcpyDtoD(r->d_joint, r->d_txt, (size_t)n_txt * H * F);
    cuMemcpyDtoD(r->d_joint + (size_t)n_txt * H * F, r->d_img, (size_t)n_img * H * F);

    int lin1_dim = 3 * H + 2 * n_ff;
    (void)lin1_dim;

    for (int bi = 0; bi < r->n_sgl; bi++) {
        flux2_gpu_sblk_t *b = &r->gpu_sblk[bi];

        /* adaLN */
        op_adaln(r, r->d_scratch1, r->d_joint, ms_shift, ms_scale, n_tot, H);

        /* linear1: compute Q, K, V, gate_up as separate GEMMs with weight slices */
        CUdeviceptr q_w = b->linear1_w;
        CUdeviceptr k_w = b->linear1_w + (size_t)H * H * 4;
        CUdeviceptr v_w = b->linear1_w + (size_t)2 * H * H * 4;
        CUdeviceptr gu_w = b->linear1_w + (size_t)3 * H * H * 4;

        op_gemm(r, r->d_q, q_w, r->d_scratch1, d0, H, H, n_tot);
        op_gemm(r, r->d_k, k_w, r->d_scratch1, d0, H, H, n_tot);
        op_gemm(r, r->d_v, v_w, r->d_scratch1, d0, H, H, n_tot);
        op_gemm(r, r->d_scratch2, gu_w, r->d_scratch1, d0, 2*n_ff, H, n_tot);

        /* Per-head Q/K norm + RoPE */
        op_rmsnorm_ph(r, r->d_q, b->q_norm, n_tot, nH, hd);
        op_rmsnorm_ph(r, r->d_k, b->k_norm, n_tot, nH, hd);
        /* txt tokens at front [0..n_txt), img after */
        op_rope_txt(r, r->d_q, n_txt, nH, hd, theta);
        op_rope_txt(r, r->d_k, n_txt, nH, hd, theta);
        CUdeviceptr q_img = r->d_q + (size_t)n_txt * H * F;
        CUdeviceptr k_img = r->d_k + (size_t)n_txt * H * F;
        op_rope_img(r, q_img, n_img, nH, hd, lat_w_p, theta);
        op_rope_img(r, k_img, n_img, nH, hd, lat_w_p, theta);

        /* Self-attention */
        op_attn(r, r->d_attn_out, r->d_q, r->d_k, r->d_v, n_tot, nH, hd);

        /* Parallel MLP: SwiGLU on gate_up */
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_tot, n_ff);

        /* linear2: [H, H+n_ff] split into [H, H] (attn) + [H, n_ff] (mlp)
         * out = l2_attn @ attn_out + l2_mlp @ mlp_out */
        op_gemm(r, r->d_scratch1, b->l2_attn_w, r->d_attn_out, d0, H, H, n_tot);
        op_gemm(r, r->d_scratch2, b->l2_mlp_w, r->d_scratch3, d0, H, n_ff, n_tot);

        /* Add the two halves: scratch1 += scratch2 */
        op_add(r, r->d_scratch1, r->d_scratch2, n_tot * H);

        /* Gated residual */
        op_gated_add(r, r->d_joint, r->d_scratch1, ms_gate, n_tot, H);
    }

    /* ---- Output ---- */
    /* img portion = joint[n_txt:] */
    CUdeviceptr d_img_out = r->d_joint + (size_t)n_txt * H * F;

    /* Final adaLN: LN(img_out) * (1 + out_scale) + out_shift */
    /* Compute out_mod: SiLU(temb) → [2H] via out_mod_w */
    /* d_temb already contains SiLU(temb_final) from step 3 */
    op_gemm(r, r->d_scratch1, r->d_out_mod_w, r->d_temb, d0, 2*H, H, 1);
    CUdeviceptr out_shift = r->d_scratch1;
    CUdeviceptr out_scale = r->d_scratch1 + (size_t)H * F;

    op_adaln(r, r->d_scratch2, d_img_out, out_shift, out_scale, n_img, H);

    /* Final linear: [pin, H] */
    op_gemm(r, r->d_attn_out, r->d_out_proj_w, r->d_scratch2, d0, r->pin, H, n_img);

    /* Download result */
    cuMemcpyDtoH(out, r->d_attn_out, (size_t)n_img * r->pin * F);
    cuCtxSynchronize();

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
    int ps = 2;
    int spatial = h * w;

    CUdeviceptr d_x = gpu_upload_f32(latent, lc * h * w);

    if (m->bn_mean && m->bn_var) {
        float bn_std[128];
        for (int i = 0; i < 128; i++) bn_std[i] = sqrtf(m->bn_var[i] + m->bn_eps);
        CUdeviceptr d_bn_mean = gpu_upload_f32(m->bn_mean, 128);
        CUdeviceptr d_bn_std = gpu_upload_f32(bn_std, 128);
        CUdeviceptr d_tmp;
        cuMemAlloc(&d_tmp, (size_t)lc * h * w * sizeof(float));
        op_vae_latent_bn(r, d_tmp, d_x, d_bn_mean, d_bn_std, lc, h, w, ps);
        cuMemFree(d_x);
        cuMemFree(d_bn_mean);
        cuMemFree(d_bn_std);
        d_x = d_tmp;
    }

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
        CUdeviceptr d_tmp = flux2_vae_mid_attn_bridge(r, d_x, &m->mid_attn, h, w, ng);
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
