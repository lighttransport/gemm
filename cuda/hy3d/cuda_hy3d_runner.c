/*
 * cuda_hy3d_runner.c - CUDA Hunyuan3D-2.1 via NVRTC-compiled kernels
 *
 * Pipeline: DINOv2 encoder → DiT diffusion (flow matching) → ShapeVAE → MC mesh
 * Compiles with plain gcc (no nvcc). Uses cuew for dynamic CUDA/NVRTC loading.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define MARCHING_CUBES_IMPLEMENTATION
#include "../../common/marching_cubes.h"
#include "cuda_hy3d_runner.h"
#include "../cuew.h"
#include "../cuda_kernels_common.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ======================================================================== */
/* HY3D-specific CUDA kernel source (compiled at runtime via NVRTC)         */
/* Shared kernels are in cuda_kernels_common.h (cuda_kernels_common_src)    */
/* ======================================================================== */

static const char cuda_hy3d_specific_kernels[] =
"\n"
"/* ---- rms_norm_f32: per-head RMSNorm (for QK normalization) ---- */\n"
"/* One thread per (token, head). Normalizes head_dim elements in-place. */\n"
"__global__ void rms_norm_f32(float *data, const float *w,\n"
"                              int n_tok, int n_heads, int head_dim,\n"
"                              int stride, float eps) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * n_heads;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / n_heads;\n"
"    int h = idx % n_heads;\n"
"    float *base = data + tok * stride + h * head_dim;\n"
"    float sum_sq = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) sum_sq += base[i] * base[i];\n"
"    float inv = rsqrtf(sum_sq / (float)head_dim + eps);\n"
"    for (int i = 0; i < head_dim; i++)\n"
"        base[i] = base[i] * inv * w[i];\n"
"}\n"
"\n"
"/* ---- qk_layernorm_f32: per-head LayerNorm (for ShapeVAE QK norm) ---- */\n"
"__global__ void qk_layernorm_f32(float *data, const float *w, const float *b,\n"
"                                   int n_tok, int n_heads, int head_dim,\n"
"                                   int stride, float eps) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * n_heads;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / n_heads;\n"
"    int h = idx % n_heads;\n"
"    float *base = data + tok * stride + h * head_dim;\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) sum += base[i];\n"
"    float mean = sum / (float)head_dim;\n"
"    float var_sum = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) { float d = base[i] - mean; var_sum += d*d; }\n"
"    float inv = rsqrtf(var_sum / (float)head_dim + eps);\n"
"    for (int i = 0; i < head_dim; i++)\n"
"        base[i] = (base[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);\n"
"}\n"
"\n"
"/* ---- layerscale_add_f32: LayerScale residual for DINOv2 ---- */\n"
"/* dst[i] += src[i] * scale[i % dim] */\n"
"__global__ void layerscale_add_f32(float *dst, const float *src,\n"
"                                     const float *scale, int n, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    dst[i] += src[i] * scale[i % dim];\n"
"}\n"
"\n"
"/* ---- cross_attn_f32: cross-attention with different Q/KV lengths ---- */\n"
"/* One block per (head, query_token). Handles Q_len != KV_len.          */\n"
"__global__ void cross_attn_f32(float *out,\n"
"                                const float *Q, const float *K, const float *V,\n"
"                                int q_len, int kv_len, int dim,\n"
"                                int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qi = blockIdx.y;\n"
"    if (h >= n_heads || qi >= q_len) return;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    const float *q = Q + qi * dim + h * head_dim;\n"
"    extern __shared__ float sdata[];\n"
"    float *scores = sdata;\n"
"    float *rbuf = sdata + kv_len;\n"
"\n"
"    /* Phase 1: Q @ K^T */\n"
"    for (int ki = tid; ki < kv_len; ki += nt) {\n"
"        const float *k = K + ki * dim + h * head_dim;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) dot += q[d] * k[d];\n"
"        scores[ki] = dot * scale;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 2: Softmax — find max */\n"
"    float lmax = -1e30f;\n"
"    for (int ki = tid; ki < kv_len; ki += nt)\n"
"        lmax = fmaxf(lmax, scores[ki]);\n"
"    rbuf[tid] = lmax;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) rbuf[tid] = fmaxf(rbuf[tid], rbuf[tid+r]);\n"
"        __syncthreads();\n"
"    }\n"
"    float mx = rbuf[0];\n"
"    __syncthreads();\n"
"\n"
"    /* Exp + sum */\n"
"    float lsum = 0.0f;\n"
"    for (int ki = tid; ki < kv_len; ki += nt) {\n"
"        scores[ki] = expf(scores[ki] - mx);\n"
"        lsum += scores[ki];\n"
"    }\n"
"    rbuf[tid] = lsum;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) rbuf[tid] += rbuf[tid+r];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv_sum = (rbuf[0] > 0) ? 1.0f / rbuf[0] : 0.0f;\n"
"    for (int ki = tid; ki < kv_len; ki += nt)\n"
"        scores[ki] *= inv_sum;\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 3: Weighted sum of V */\n"
"    for (int d = tid; d < head_dim; d += nt) {\n"
"        float acc = 0.0f;\n"
"        for (int ki = 0; ki < kv_len; ki++)\n"
"            acc += scores[ki] * V[ki * dim + h * head_dim + d];\n"
"        out[qi * dim + h * head_dim + d] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* ---- fourier_embed_3d_f32: 3D coordinate Fourier embedding ---- */\n"
"/* input: [N, 3], output: [N, out_dim] where out_dim = 3*(2*nf+1)  */\n"
"/* Layout: [x,y,z, sin(f0*x),...sin(fn*x), sin(f0*y),..., cos(...)...] */\n"
"__global__ void fourier_embed_3d_f32(float *out, const float *coords,\n"
"                                      const float *freqs, int N,\n"
"                                      int num_freqs, int out_dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx >= N) return;\n"
"    const float *in = coords + idx * 3;\n"
"    float *o = out + idx * out_dim;\n"
"    int p = 0;\n"
"    /* Include raw input */\n"
"    o[p++] = in[0]; o[p++] = in[1]; o[p++] = in[2];\n"
"    /* Sin embeddings */\n"
"    for (int d = 0; d < 3; d++)\n"
"        for (int f = 0; f < num_freqs; f++)\n"
"            o[p++] = sinf(in[d] * freqs[f]);\n"
"    /* Cos embeddings */\n"
"    for (int d = 0; d < 3; d++)\n"
"        for (int f = 0; f < num_freqs; f++)\n"
"            o[p++] = cosf(in[d] * freqs[f]);\n"
"}\n"
"\n"
"/* ---- timestep_embed_f32: sinusoidal timestep embedding ---- */\n"
"/* Similar to transformer PE but for scalar timestep */\n"
"__global__ void timestep_embed_f32(float *out, float t, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int half = dim / 2;\n"
"    if (i >= half) return;\n"
"    float log_ts = logf(10000.0f) / (float)(half - 1);\n"
"    float emb = expf(-log_ts * (float)i) * t * 1000.0f;\n"
"    out[i] = sinf(emb);\n"
"    out[half + i] = cosf(emb);\n"
"}\n"
"\n"
"/* ---- euler_step_f32: x_new = x - dt * v ---- */\n"
"__global__ void euler_step_f32(float *x, const float *v, float dt, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    x[i] = x[i] - dt * v[i];\n"
"}\n"
"\n"
"/* ---- cfg_combine_f32: out = uncond + scale * (cond - uncond) ---- */\n"
"__global__ void cfg_combine_f32(float *out, const float *cond,\n"
"                                  const float *uncond,\n"
"                                  float scale, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    out[i] = uncond[i] + scale * (cond[i] - uncond[i]);\n"
"}\n"
"\n"
"/* ---- split_qkv_interleaved_f32: split [N, 3*W] with interleaved heads ---- */\n"
"/* Input:  [N, H, 3, HD] (3*W = H*3*HD) */\n"
"/* Output: Q[N, W], K[N, W], V[N, W] (W = H*HD) */\n"
"__global__ void split_qkv_interleaved_f32(\n"
"    float *Q, float *K, float *V,\n"
"    const float *qkv, int N, int H, int HD) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int W = H * HD;\n"
"    int total = N * W;\n"
"    if (idx >= total) return;\n"
"    int n = idx / W;\n"
"    int rem = idx % W;\n"
"    int h = rem / HD;\n"
"    int d = rem % HD;\n"
"    int base = n * 3 * W + h * 3 * HD;\n"
"    Q[idx] = qkv[base + d];\n"
"    K[idx] = qkv[base + HD + d];\n"
"    V[idx] = qkv[base + 2*HD + d];\n"
"}\n"
"\n"
"/* ---- split_kv_interleaved_f32: split [M, 2*W] with interleaved heads ---- */\n"
"/* Input:  [M, H, 2, HD] (2*W = H*2*HD) */\n"
"/* Output: K[M, W], V[M, W] (W = H*HD) */\n"
"__global__ void split_kv_interleaved_f32(\n"
"    float *K, float *V,\n"
"    const float *kv, int M, int H, int HD) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int W = H * HD;\n"
"    int total = M * W;\n"
"    if (idx >= total) return;\n"
"    int m = idx / W;\n"
"    int rem = idx % W;\n"
"    int h = rem / HD;\n"
"    int d = rem % HD;\n"
"    int base = m * 2 * W + h * 2 * HD;\n"
"    K[idx] = kv[base + d];\n"
"    V[idx] = kv[base + HD + d];\n"
"}\n"
"\n";

/* ======================================================================== */
/* Model constants                                                          */
/* ======================================================================== */

/* DINOv2-L config */
#define DINO_HIDDEN     1024
#define DINO_HEADS      16
#define DINO_HEAD_DIM   64
#define DINO_LAYERS     24
#define DINO_FFN        4096
#define DINO_PATCH      14
#define DINO_IMG_SIZE   518
#define DINO_NUM_PATCHES ((DINO_IMG_SIZE/DINO_PATCH)*(DINO_IMG_SIZE/DINO_PATCH)) /* 1369 */
#define DINO_SEQ_LEN    (DINO_NUM_PATCHES + 1)  /* 1370 */

/* DiT config */
#define DIT_INPUT_SIZE  4096
#define DIT_IN_CHANNELS 64
#define DIT_HIDDEN      2048
#define DIT_CONTEXT_DIM 1024
#define DIT_DEPTH       21
#define DIT_HEADS       16
#define DIT_HEAD_DIM    128

/* ShapeVAE config */
#define VAE_NUM_LATENTS  4096
#define VAE_EMBED_DIM    64
#define VAE_WIDTH        1024
#define VAE_HEADS        16
#define VAE_HEAD_DIM     64
#define VAE_DEC_LAYERS   16
#define VAE_NUM_FREQS    8
#define VAE_FOURIER_DIM  51   /* 3*(2*8+1) */

/* ======================================================================== */
/* Internal structures                                                      */
/* ======================================================================== */

/* DINOv2 per-layer weights (on GPU, F16) */
typedef struct {
    CUdeviceptr ln1_w, ln1_b;      /* LayerNorm 1 */
    CUdeviceptr q_w, q_b;          /* Query projection */
    CUdeviceptr k_w, k_b;          /* Key projection */
    CUdeviceptr v_w, v_b;          /* Value projection */
    CUdeviceptr out_w, out_b;      /* Output projection */
    CUdeviceptr ls1;               /* LayerScale 1 */
    CUdeviceptr ln2_w, ln2_b;      /* LayerNorm 2 */
    CUdeviceptr fc1_w, fc1_b;      /* MLP FC1 */
    CUdeviceptr fc2_w, fc2_b;      /* MLP FC2 */
    CUdeviceptr ls2;               /* LayerScale 2 */
} dino_layer_gpu;

/* DiT per-block weights (on GPU, F16) */
typedef struct {
    /* Self-attention */
    CUdeviceptr norm1_w, norm1_b;
    CUdeviceptr sa_q_w, sa_k_w, sa_v_w;
    CUdeviceptr sa_out_w, sa_out_b;
    CUdeviceptr sa_q_norm_w, sa_k_norm_w;  /* RMSNorm for QK */
    /* Cross-attention */
    CUdeviceptr norm2_w, norm2_b;
    CUdeviceptr ca_q_w, ca_k_w, ca_v_w;
    CUdeviceptr ca_out_w, ca_out_b;
    CUdeviceptr ca_q_norm_w, ca_k_norm_w;
    /* MLP */
    CUdeviceptr norm3_w, norm3_b;
    CUdeviceptr mlp_fc1_w, mlp_fc1_b;
    CUdeviceptr mlp_fc2_w, mlp_fc2_b;
} dit_block_gpu;

/* ShapeVAE transformer block weights (on GPU, F32) */
typedef struct {
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr qkv_w;             /* Fused QKV [3*W, W] */
    CUdeviceptr proj_w, proj_b;
    CUdeviceptr q_norm_w, q_norm_b;
    CUdeviceptr k_norm_w, k_norm_b;
    int use_qk_norm;
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr mlp_fc_w, mlp_fc_b;
    CUdeviceptr mlp_proj_w, mlp_proj_b;
} vae_block_gpu;

/* ShapeVAE geometry decoder weights */
typedef struct {
    CUdeviceptr query_proj_w, query_proj_b;
    CUdeviceptr ln1_w, ln1_b;     /* Query LN */
    CUdeviceptr ln2_w, ln2_b;     /* Key/Value LN */
    CUdeviceptr c_q_w;            /* Cross-attn Q proj */
    CUdeviceptr c_kv_w;           /* Cross-attn KV proj [2*W, W] */
    CUdeviceptr c_proj_w, c_proj_b;
    CUdeviceptr q_norm_w, q_norm_b;
    CUdeviceptr k_norm_w, k_norm_b;
    int use_qk_norm;
    CUdeviceptr ln3_w, ln3_b;
    CUdeviceptr mlp_fc_w, mlp_fc_b;
    CUdeviceptr mlp_proj_w, mlp_proj_b;
    CUdeviceptr ln_post_w, ln_post_b;
    CUdeviceptr output_w, output_b;
} vae_geo_decoder_gpu;

struct cuda_hy3d_runner {
    /* CUDA context */
    CUdevice device;
    CUcontext ctx;
    CUstream stream;
    CUmodule module;
    int sm_version;
    int verbose;

    /* Compiled kernel functions */
    CUfunction fn_layernorm;
    CUfunction fn_gemm;
    CUfunction fn_gemm_tiled;
    CUfunction fn_gelu;
    CUfunction fn_add;
    CUfunction fn_silu;
    CUfunction fn_resize_normalize;
    CUfunction fn_patch_embed;
    CUfunction fn_cls_pos_embed;
    CUfunction fn_attn_prefill;
    CUfunction fn_flash_attn_tiled;
    CUfunction fn_kv_transpose;
    CUfunction fn_bilinear_upsample;
    /* HY3D-specific */
    CUfunction fn_rms_norm;
    CUfunction fn_qk_layernorm;
    CUfunction fn_layerscale_add;
    CUfunction fn_cross_attn;
    CUfunction fn_fourier_embed;
    CUfunction fn_timestep_embed;
    CUfunction fn_euler_step;
    CUfunction fn_cfg_combine;
    CUfunction fn_split_qkv;
    CUfunction fn_split_kv;

    /* DINOv2 weights */
    CUdeviceptr dino_patch_w, dino_patch_b;
    CUdeviceptr dino_pos_emb;
    CUdeviceptr dino_cls_token;
    CUdeviceptr dino_final_ln_w, dino_final_ln_b;
    dino_layer_gpu dino_layers[DINO_LAYERS];

    /* DiT weights */
    CUdeviceptr dit_x_emb_w, dit_x_emb_b;
    CUdeviceptr dit_t_mlp0_w, dit_t_mlp0_b;
    CUdeviceptr dit_t_mlp2_w, dit_t_mlp2_b;
    CUdeviceptr dit_final_ln_w, dit_final_ln_b;
    CUdeviceptr dit_final_linear_w, dit_final_linear_b;
    dit_block_gpu dit_blocks[DIT_DEPTH];

    /* ShapeVAE weights */
    CUdeviceptr vae_post_kl_w, vae_post_kl_b;
    vae_block_gpu vae_blocks[VAE_DEC_LAYERS];
    vae_geo_decoder_gpu vae_geo;
    CUdeviceptr vae_fourier_freqs;  /* [num_freqs] precomputed */

    /* Scratch buffers (GPU) */
    CUdeviceptr scratch[8];        /* General purpose scratch buffers */
    size_t scratch_size[8];

    /* Pre-computed cross-attention K,V for DiT (constant across diffusion steps) */
    CUdeviceptr dit_ca_K[DIT_DEPTH];   /* [DINO_SEQ_LEN, DIT_HIDDEN] */
    CUdeviceptr dit_ca_V[DIT_DEPTH];   /* [DINO_SEQ_LEN, DIT_HIDDEN] */
    int ca_kv_precomputed;

    /* Load status */
    int dino_loaded, dit_loaded, vae_loaded;
};

/* ======================================================================== */
/* Kernel compilation                                                       */
/* ======================================================================== */

static int hy3d_compile_kernels(cuda_hy3d_runner *r) {
    /* Concatenate common + HY3D-specific kernel source */
    size_t len1 = strlen(cuda_kernels_common_src);
    size_t len2 = strlen(cuda_hy3d_specific_kernels);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, cuda_kernels_common_src, len1);
    memcpy(full_src + len1, cuda_hy3d_specific_kernels, len2);
    full_src[len1 + len2] = '\0';

    r->sm_version = cu_compile_kernels(&r->module, r->device, full_src,
                                        "hy3d_kernels", r->verbose, "HY3D");
    free(full_src);
    if (r->sm_version < 0) return -1;

    /* Get common kernel functions */
    #define GET_FN(name, var) \
        if (cuModuleGetFunction(&r->var, r->module, name) != CUDA_SUCCESS) { \
            fprintf(stderr, "HY3D: failed to get kernel '%s'\n", name); return -1; }

    GET_FN("layernorm_f32",          fn_layernorm);
    GET_FN("gemm_f16_f32",           fn_gemm);
    GET_FN("gemm_tiled_f16_f32",     fn_gemm_tiled);
    GET_FN("gelu_f32",               fn_gelu);
    GET_FN("add_f32",                fn_add);
    GET_FN("silu_f32",               fn_silu);
    GET_FN("resize_normalize",       fn_resize_normalize);
    GET_FN("patch_embed_conv2d",     fn_patch_embed);
    GET_FN("cls_pos_embed",          fn_cls_pos_embed);
    GET_FN("kv_transpose",           fn_kv_transpose);

    /* Use MMA attention if sm_70+ */
    if (r->sm_version >= 70) {
        GET_FN("attn_prefill_f32", fn_attn_prefill);
    } else {
        GET_FN("flash_attn_tiled_f32", fn_attn_prefill);
    }
    GET_FN("flash_attn_tiled_f32",   fn_flash_attn_tiled);
    GET_FN("bilinear_upsample_f32",  fn_bilinear_upsample);

    /* HY3D-specific functions */
    GET_FN("rms_norm_f32",           fn_rms_norm);
    GET_FN("qk_layernorm_f32",       fn_qk_layernorm);
    GET_FN("layerscale_add_f32",     fn_layerscale_add);
    GET_FN("cross_attn_f32",         fn_cross_attn);
    GET_FN("fourier_embed_3d_f32",   fn_fourier_embed);
    GET_FN("timestep_embed_f32",     fn_timestep_embed);
    GET_FN("euler_step_f32",         fn_euler_step);
    GET_FN("cfg_combine_f32",        fn_cfg_combine);
    GET_FN("split_qkv_interleaved_f32", fn_split_qkv);
    GET_FN("split_kv_interleaved_f32",  fn_split_kv);

    #undef GET_FN
    return 0;
}

/* ======================================================================== */
/* Upload helpers                                                           */
/* ======================================================================== */

/* Upload safetensors tensor to GPU as F16 (converting F32->F16 if needed) */
static CUdeviceptr st_upload_f16(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [WARN] tensor '%s' not found\n", name);
        return 0;
    }
    void *data = safetensors_data(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    if (strcmp(dtype, "F16") == 0 || strcmp(dtype, "BF16") == 0) {
        return cu_upload_raw(data, nbytes);
    } else if (strcmp(dtype, "F32") == 0) {
        /* Convert F32 to F16 on CPU then upload */
        size_t n = nbytes / sizeof(float);
        uint16_t *f16 = (uint16_t *)malloc(n * sizeof(uint16_t));
        const float *f32 = (const float *)data;
        for (size_t i = 0; i < n; i++) {
            f16[i] = cu_f32_to_f16(f32[i]);
        }
        CUdeviceptr d = cu_upload_raw(f16, n * sizeof(uint16_t));
        free(f16);
        return d;
    }
    if (verbose) fprintf(stderr, "  [WARN] tensor '%s' has unsupported dtype '%s'\n", name, dtype);
    return 0;
}

/* Upload safetensors tensor to GPU as F32 (converting F16->F32 if needed) */
static CUdeviceptr st_upload_f32(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [WARN] tensor '%s' not found\n", name);
        return 0;
    }
    void *data = safetensors_data(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    if (strcmp(dtype, "F32") == 0) {
        return cu_upload_raw(data, nbytes);
    } else if (strcmp(dtype, "F16") == 0) {
        size_t n = nbytes / sizeof(uint16_t);
        float *f32 = (float *)malloc(n * sizeof(float));
        const uint16_t *f16 = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            /* Simple F16 to F32 conversion */
            uint32_t sign = (f16[i] >> 15) & 0x1;
            uint32_t exp = (f16[i] >> 10) & 0x1f;
            uint32_t mant = f16[i] & 0x3ff;
            uint32_t f;
            if (exp == 0) {
                f = sign << 31;
            } else if (exp == 31) {
                f = (sign << 31) | 0x7f800000 | (mant << 13);
            } else {
                f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            }
            memcpy(&f32[i], &f, sizeof(float));
        }
        CUdeviceptr d = cu_upload_raw(f32, n * sizeof(float));
        free(f32);
        return d;
    }
    return 0;
}

/* Allocate GPU buffer */
static CUdeviceptr gpu_alloc(size_t bytes) {
    CUdeviceptr d = 0;
    if (cuMemAlloc(&d, bytes) != CUDA_SUCCESS) {
        fprintf(stderr, "HY3D: GPU alloc failed for %zu bytes\n", bytes);
        return 0;
    }
    return d;
}

/* ======================================================================== */
/* Kernel launch wrappers                                                   */
/* ======================================================================== */

static void kl_layernorm(cuda_hy3d_runner *r, CUdeviceptr dst, CUdeviceptr src,
                          CUdeviceptr w, CUdeviceptr b, int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    cuLaunchKernel(r->fn_layernorm, (unsigned)n_tok, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_gemm(cuda_hy3d_runner *r, CUdeviceptr Y, CUdeviceptr W,
                     CUdeviceptr X, CUdeviceptr bias,
                     int n_out, int n_in, int n_tok) {
    CUfunction fn = (r->sm_version >= 70) ? r->fn_gemm : r->fn_gemm_tiled;
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    int bx = (r->sm_version >= 70) ? 32 : 256;
    int gy = (n_tok + 15) / 16;
    cuLaunchKernel(fn, (unsigned)((n_out+15)/16), (unsigned)gy, 1,
                   bx, 1, 1, 0, r->stream, args, NULL);
}

static void kl_gelu(cuda_hy3d_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->fn_gelu, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_silu(cuda_hy3d_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->fn_silu, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_add(cuda_hy3d_runner *r, CUdeviceptr dst, CUdeviceptr src, int n) {
    void *args[] = {&dst, &src, &n};
    cuLaunchKernel(r->fn_add, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_layerscale_add(cuda_hy3d_runner *r, CUdeviceptr dst,
                               CUdeviceptr src, CUdeviceptr scale,
                               int n, int dim) {
    void *args[] = {&dst, &src, &scale, &n, &dim};
    cuLaunchKernel(r->fn_layerscale_add, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_rms_norm(cuda_hy3d_runner *r, CUdeviceptr data, CUdeviceptr w,
                         int n_tok, int n_heads, int head_dim, int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data, &w, &n_tok, &n_heads, &head_dim, &stride, &eps};
    cuLaunchKernel(r->fn_rms_norm, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_qk_layernorm(cuda_hy3d_runner *r, CUdeviceptr data,
                              CUdeviceptr w, CUdeviceptr b,
                              int n_tok, int n_heads, int head_dim, int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data, &w, &b, &n_tok, &n_heads, &head_dim, &stride, &eps};
    cuLaunchKernel(r->fn_qk_layernorm, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_cross_attn(cuda_hy3d_runner *r, CUdeviceptr out,
                           CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                           int q_len, int kv_len, int dim,
                           int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int nt = 128;  /* threads per block */
    size_t smem = (size_t)(kv_len + nt) * sizeof(float);
    void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim,
                    &n_heads, &head_dim, &scale};
    cuLaunchKernel(r->fn_cross_attn, (unsigned)n_heads, (unsigned)q_len, 1,
                   (unsigned)nt, 1, 1, smem, r->stream, args, NULL);
}

static void kl_self_attn(cuda_hy3d_runner *r, CUdeviceptr out,
                          CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                          int n_tok, int dim, int n_heads, int head_dim) {
    /* Reuse cross_attn with q_len == kv_len */
    kl_cross_attn(r, out, Q, K, V, n_tok, n_tok, dim, n_heads, head_dim);
}

static void kl_fourier_embed(cuda_hy3d_runner *r, CUdeviceptr out,
                              CUdeviceptr coords, CUdeviceptr freqs,
                              int N, int num_freqs, int out_dim) {
    void *args[] = {&out, &coords, &freqs, &N, &num_freqs, &out_dim};
    cuLaunchKernel(r->fn_fourier_embed, (unsigned)((N+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_timestep_embed(cuda_hy3d_runner *r, CUdeviceptr out,
                               float t, int dim) {
    int half = dim / 2;
    void *args[] = {&out, &t, &dim};
    cuLaunchKernel(r->fn_timestep_embed, (unsigned)((half+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_euler_step(cuda_hy3d_runner *r, CUdeviceptr x, CUdeviceptr v,
                           float dt, int n) {
    void *args[] = {&x, &v, &dt, &n};
    cuLaunchKernel(r->fn_euler_step, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_cfg_combine(cuda_hy3d_runner *r, CUdeviceptr out,
                            CUdeviceptr cond, CUdeviceptr uncond,
                            float scale, int n) {
    void *args[] = {&out, &cond, &uncond, &scale, &n};
    cuLaunchKernel(r->fn_cfg_combine, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_split_qkv(cuda_hy3d_runner *r, CUdeviceptr Q, CUdeviceptr K,
                           CUdeviceptr V, CUdeviceptr qkv,
                           int N, int H, int HD) {
    int total = N * H * HD;
    void *args[] = {&Q, &K, &V, &qkv, &N, &H, &HD};
    cuLaunchKernel(r->fn_split_qkv, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void kl_split_kv(cuda_hy3d_runner *r, CUdeviceptr K, CUdeviceptr V,
                          CUdeviceptr kv, int M, int H, int HD) {
    int total = M * H * HD;
    void *args[] = {&K, &V, &kv, &M, &H, &HD};
    cuLaunchKernel(r->fn_split_kv, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

/* ======================================================================== */
/* Scratch buffer management                                                */
/* ======================================================================== */

static void ensure_scratch(cuda_hy3d_runner *r, int idx, size_t bytes) {
    if (r->scratch_size[idx] < bytes) {
        if (r->scratch[idx]) cuMemFree(r->scratch[idx]);
        r->scratch[idx] = gpu_alloc(bytes);
        r->scratch_size[idx] = bytes;
    }
}

/* ======================================================================== */
/* Weight loading                                                           */
/* ======================================================================== */

static int load_dino_weights(cuda_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "HY3D: cannot open DINOv2 weights: %s\n", path);
        return -1;
    }
    if (r->verbose) fprintf(stderr, "HY3D: loading DINOv2 from %s (%d tensors)\n",
                             path, st->n_tensors);

    /* Embeddings */
    r->dino_patch_w = st_upload_f32(st, "main_image_encoder.model.embeddings.patch_embeddings.projection.weight", r->verbose);
    r->dino_patch_b = st_upload_f32(st, "main_image_encoder.model.embeddings.patch_embeddings.projection.bias", r->verbose);
    r->dino_pos_emb = st_upload_f32(st, "main_image_encoder.model.embeddings.position_embeddings", r->verbose);
    r->dino_cls_token = st_upload_f32(st, "main_image_encoder.model.embeddings.cls_token", r->verbose);

    /* Encoder layers */
    for (int i = 0; i < DINO_LAYERS; i++) {
        char name[256];
        dino_layer_gpu *l = &r->dino_layers[i];
        #define DINO_T(field, suffix) \
            snprintf(name, sizeof(name), "main_image_encoder.model.encoder.layer.%d.%s", i, suffix); \
            l->field = st_upload_f16(st, name, r->verbose);

        DINO_T(ln1_w, "norm1.weight");
        DINO_T(ln1_b, "norm1.bias");
        DINO_T(q_w,   "attention.attention.query.weight");
        DINO_T(q_b,   "attention.attention.query.bias");
        DINO_T(k_w,   "attention.attention.key.weight");
        DINO_T(k_b,   "attention.attention.key.bias");
        DINO_T(v_w,   "attention.attention.value.weight");
        DINO_T(v_b,   "attention.attention.value.bias");
        DINO_T(out_w, "attention.output.dense.weight");
        DINO_T(out_b, "attention.output.dense.bias");
        DINO_T(ls1,   "layer_scale1.lambda1");
        DINO_T(ln2_w, "norm2.weight");
        DINO_T(ln2_b, "norm2.bias");
        DINO_T(fc1_w, "mlp.fc1.weight");
        DINO_T(fc1_b, "mlp.fc1.bias");
        DINO_T(fc2_w, "mlp.fc2.weight");
        DINO_T(fc2_b, "mlp.fc2.bias");
        DINO_T(ls2,   "layer_scale2.lambda1");
        #undef DINO_T
    }

    /* Final LN */
    r->dino_final_ln_w = st_upload_f16(st, "main_image_encoder.model.layernorm.weight", r->verbose);
    r->dino_final_ln_b = st_upload_f16(st, "main_image_encoder.model.layernorm.bias", r->verbose);
    if (!r->dino_final_ln_w) {
        r->dino_final_ln_w = st_upload_f16(st, "main_image_encoder.model.norm.weight", r->verbose);
        r->dino_final_ln_b = st_upload_f16(st, "main_image_encoder.model.norm.bias", r->verbose);
    }

    safetensors_close(st);
    r->dino_loaded = 1;
    if (r->verbose) fprintf(stderr, "HY3D: DINOv2 weights loaded\n");
    return 0;
}

static int load_dit_weights(cuda_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "HY3D: cannot open DiT weights: %s\n", path);
        return -1;
    }
    if (r->verbose) fprintf(stderr, "HY3D: loading DiT from %s (%d tensors)\n",
                             path, st->n_tensors);

    /* x_embedder */
    r->dit_x_emb_w = st_upload_f16(st, "x_embedder.weight", r->verbose);
    r->dit_x_emb_b = st_upload_f16(st, "x_embedder.bias", r->verbose);

    /* t_embedder */
    r->dit_t_mlp0_w = st_upload_f16(st, "t_embedder.mlp.0.weight", r->verbose);
    r->dit_t_mlp0_b = st_upload_f16(st, "t_embedder.mlp.0.bias", r->verbose);
    r->dit_t_mlp2_w = st_upload_f16(st, "t_embedder.mlp.2.weight", r->verbose);
    r->dit_t_mlp2_b = st_upload_f16(st, "t_embedder.mlp.2.bias", r->verbose);

    /* Blocks */
    for (int i = 0; i < DIT_DEPTH; i++) {
        char name[256];
        dit_block_gpu *b = &r->dit_blocks[i];
        #define DIT_T(field, suffix) \
            snprintf(name, sizeof(name), "blocks.%d.%s", i, suffix); \
            b->field = st_upload_f16(st, name, r->verbose);

        DIT_T(norm1_w,     "norm1.weight");
        DIT_T(norm1_b,     "norm1.bias");
        DIT_T(sa_q_w,      "attn1.to_q.weight");
        DIT_T(sa_k_w,      "attn1.to_k.weight");
        DIT_T(sa_v_w,      "attn1.to_v.weight");
        DIT_T(sa_out_w,    "attn1.out_proj.weight");
        DIT_T(sa_out_b,    "attn1.out_proj.bias");
        DIT_T(sa_q_norm_w, "attn1.q_norm.weight");
        DIT_T(sa_k_norm_w, "attn1.k_norm.weight");

        DIT_T(norm2_w,     "norm2.weight");
        DIT_T(norm2_b,     "norm2.bias");
        DIT_T(ca_q_w,      "attn2.to_q.weight");
        DIT_T(ca_k_w,      "attn2.to_k.weight");
        DIT_T(ca_v_w,      "attn2.to_v.weight");
        DIT_T(ca_out_w,    "attn2.out_proj.weight");
        DIT_T(ca_out_b,    "attn2.out_proj.bias");
        DIT_T(ca_q_norm_w, "attn2.q_norm.weight");
        DIT_T(ca_k_norm_w, "attn2.k_norm.weight");

        /* norm3 before MLP (may or may not exist) */
        DIT_T(norm3_w,     "norm3.weight");
        DIT_T(norm3_b,     "norm3.bias");

        DIT_T(mlp_fc1_w,   "mlp.fc1.weight");
        DIT_T(mlp_fc1_b,   "mlp.fc1.bias");
        DIT_T(mlp_fc2_w,   "mlp.fc2.weight");
        DIT_T(mlp_fc2_b,   "mlp.fc2.bias");
        #undef DIT_T
    }

    /* Final layer */
    r->dit_final_ln_w = st_upload_f16(st, "final_layer.norm_final.weight", r->verbose);
    r->dit_final_ln_b = st_upload_f16(st, "final_layer.norm_final.bias", r->verbose);
    r->dit_final_linear_w = st_upload_f16(st, "final_layer.linear.weight", r->verbose);
    r->dit_final_linear_b = st_upload_f16(st, "final_layer.linear.bias", r->verbose);

    safetensors_close(st);
    r->dit_loaded = 1;
    if (r->verbose) fprintf(stderr, "HY3D: DiT weights loaded\n");
    return 0;
}

static int load_vae_weights(cuda_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "HY3D: cannot open VAE weights: %s\n", path);
        return -1;
    }
    if (r->verbose) fprintf(stderr, "HY3D: loading ShapeVAE from %s (%d tensors)\n",
                             path, st->n_tensors);

    /* Post-KL projection */
    r->vae_post_kl_w = st_upload_f32(st, "post_kl.weight", r->verbose);
    r->vae_post_kl_b = st_upload_f32(st, "post_kl.bias", r->verbose);

    /* Transformer decoder blocks */
    for (int i = 0; i < VAE_DEC_LAYERS; i++) {
        char name[256];
        vae_block_gpu *b = &r->vae_blocks[i];
        #define VAE_T(field, suffix) \
            snprintf(name, sizeof(name), "decoder.layers.%d.%s", i, suffix); \
            b->field = st_upload_f32(st, name, r->verbose);

        VAE_T(ln1_w,        "ln1.weight");
        VAE_T(ln1_b,        "ln1.bias");
        VAE_T(qkv_w,        "attn.qkv.weight");
        VAE_T(proj_w,       "attn.proj.weight");
        VAE_T(proj_b,       "attn.proj.bias");
        VAE_T(q_norm_w,     "attn.q_norm.weight");
        VAE_T(q_norm_b,     "attn.q_norm.bias");
        VAE_T(k_norm_w,     "attn.k_norm.weight");
        VAE_T(k_norm_b,     "attn.k_norm.bias");
        VAE_T(ln2_w,        "ln2.weight");
        VAE_T(ln2_b,        "ln2.bias");
        VAE_T(mlp_fc_w,     "mlp.fc.weight");
        VAE_T(mlp_fc_b,     "mlp.fc.bias");
        VAE_T(mlp_proj_w,   "mlp.proj.weight");
        VAE_T(mlp_proj_b,   "mlp.proj.bias");
        #undef VAE_T

        b->use_qk_norm = (b->q_norm_w != 0);
    }

    /* Geometry decoder */
    vae_geo_decoder_gpu *g = &r->vae_geo;
    #define GEO_T(field, suffix) \
        g->field = st_upload_f32(st, "geo_decoder." suffix, r->verbose);

    GEO_T(query_proj_w, "query_proj.weight");
    GEO_T(query_proj_b, "query_proj.bias");
    GEO_T(ln1_w,        "cross_attn_block.ln1.weight");
    GEO_T(ln1_b,        "cross_attn_block.ln1.bias");
    GEO_T(ln2_w,        "cross_attn_block.ln2.weight");
    GEO_T(ln2_b,        "cross_attn_block.ln2.bias");
    GEO_T(c_q_w,        "cross_attn_block.attn.q.weight");
    GEO_T(c_kv_w,       "cross_attn_block.attn.kv.weight");
    GEO_T(c_proj_w,     "cross_attn_block.attn.proj.weight");
    GEO_T(c_proj_b,     "cross_attn_block.attn.proj.bias");
    GEO_T(q_norm_w,     "cross_attn_block.attn.q_norm.weight");
    GEO_T(q_norm_b,     "cross_attn_block.attn.q_norm.bias");
    GEO_T(k_norm_w,     "cross_attn_block.attn.k_norm.weight");
    GEO_T(k_norm_b,     "cross_attn_block.attn.k_norm.bias");
    GEO_T(ln3_w,        "cross_attn_block.ln3.weight");
    GEO_T(ln3_b,        "cross_attn_block.ln3.bias");
    GEO_T(mlp_fc_w,     "cross_attn_block.mlp.fc.weight");
    GEO_T(mlp_fc_b,     "cross_attn_block.mlp.fc.bias");
    GEO_T(mlp_proj_w,   "cross_attn_block.mlp.proj.weight");
    GEO_T(mlp_proj_b,   "cross_attn_block.mlp.proj.bias");
    GEO_T(ln_post_w,    "ln_post.weight");
    GEO_T(ln_post_b,    "ln_post.bias");
    GEO_T(output_w,     "output_proj.weight");
    GEO_T(output_b,     "output_proj.bias");
    #undef GEO_T

    g->use_qk_norm = (g->q_norm_w != 0);

    /* Pre-compute Fourier frequencies on GPU */
    float freqs[VAE_NUM_FREQS];
    const float pi = 3.141592653589793f;
    for (int i = 0; i < VAE_NUM_FREQS; i++) {
        freqs[i] = powf(2.0f, (float)i) * pi;
    }
    r->vae_fourier_freqs = cu_upload_raw(freqs, sizeof(freqs));

    safetensors_close(st);
    r->vae_loaded = 1;
    if (r->verbose) fprintf(stderr, "HY3D: ShapeVAE weights loaded\n");
    return 0;
}

/* ======================================================================== */
/* Pipeline stages                                                          */
/* ======================================================================== */

/* Stage 1: DINOv2 forward pass
 * Input:  d_image [3, 518, 518] F32 (normalized)
 * Output: d_out [1370, 1024] F32 */
static void run_dinov2(cuda_hy3d_runner *r, CUdeviceptr d_image, CUdeviceptr d_out) {
    const int seq = DINO_SEQ_LEN;
    const int dim = DINO_HIDDEN;
    const int heads = DINO_HEADS;
    const int hd = DINO_HEAD_DIM;
    const int ffn = DINO_FFN;
    const int ps = DINO_PATCH;
    const int gw = DINO_IMG_SIZE / ps;  /* 37 */

    /* Scratch: 0=hidden[seq*dim], 1=qkv[3*seq*dim], 2=attn_out[seq*dim],
     *          3=mlp[seq*ffn], 4=normed[seq*dim], 5=K_t+V_t[2*heads*seq*hd] */
    ensure_scratch(r, 0, (size_t)seq * dim * sizeof(float));
    ensure_scratch(r, 1, (size_t)3 * seq * dim * sizeof(float));
    ensure_scratch(r, 2, (size_t)seq * dim * sizeof(float));
    ensure_scratch(r, 3, (size_t)seq * ffn * sizeof(float));
    ensure_scratch(r, 4, (size_t)seq * dim * sizeof(float));
    ensure_scratch(r, 5, (size_t)2 * seq * dim * sizeof(float));

    CUdeviceptr d_hidden = r->scratch[0];
    CUdeviceptr d_qkv    = r->scratch[1];
    CUdeviceptr d_attn   = r->scratch[2];
    CUdeviceptr d_mlp    = r->scratch[3];
    CUdeviceptr d_normed = r->scratch[4];
    CUdeviceptr d_Kt     = r->scratch[5];
    (void)d_Kt; /* KV transpose used by flash attention path */

    /* 1. Patch embedding: conv2d → [num_patches, dim] */
    {
        int gw2 = gw;
        int dim2 = dim, ps2 = ps, img_w = DINO_IMG_SIZE;
        CUdeviceptr pw = r->dino_patch_w, pb = r->dino_patch_b;
        void *args[] = {&d_hidden, &d_image, &pw, &pb, &gw2, &dim2, &ps2, &img_w};
        cuLaunchKernel(r->fn_patch_embed,
                       (unsigned)((dim+15)/16), (unsigned)(gw*gw), 1,
                       16, 1, 1, 0, r->stream, args, NULL);
    }

    /* 2. CLS token + position embeddings */
    {
        int n_tok = seq, d2 = dim;
        void *args[] = {(void*)&d_hidden, (void*)&r->dino_cls_token,
                        (void*)&r->dino_pos_emb, &n_tok, &d2};
        cuLaunchKernel(r->fn_cls_pos_embed,
                       (unsigned)((seq*dim+255)/256), 1, 1,
                       256, 1, 1, 0, r->stream, args, NULL);
    }

    /* 3. Encoder layers */
    for (int li = 0; li < DINO_LAYERS; li++) {
        dino_layer_gpu *l = &r->dino_layers[li];

        /* LN1 → Q,K,V → Attention → LayerScale + residual */
        kl_layernorm(r, d_normed, d_hidden, l->ln1_w, l->ln1_b, seq, dim);

        /* Q, K, V projections */
        CUdeviceptr d_Q = d_qkv;
        CUdeviceptr d_K = d_qkv + (size_t)seq * dim * sizeof(float);
        CUdeviceptr d_V = d_qkv + (size_t)2 * seq * dim * sizeof(float);
        kl_gemm(r, d_Q, l->q_w, d_normed, l->q_b, dim, dim, seq);
        kl_gemm(r, d_K, l->k_w, d_normed, l->k_b, dim, dim, seq);
        kl_gemm(r, d_V, l->v_w, d_normed, l->v_b, dim, dim, seq);

        /* Self-attention */
        kl_self_attn(r, d_attn, d_Q, d_K, d_V, seq, dim, heads, hd);

        /* Output projection */
        kl_gemm(r, d_normed, l->out_w, d_attn, l->out_b, dim, dim, seq);

        /* LayerScale 1 + residual */
        if (l->ls1)
            kl_layerscale_add(r, d_hidden, d_normed, l->ls1, seq * dim, dim);
        else
            kl_add(r, d_hidden, d_normed, seq * dim);

        /* LN2 → MLP → LayerScale + residual */
        kl_layernorm(r, d_normed, d_hidden, l->ln2_w, l->ln2_b, seq, dim);
        kl_gemm(r, d_mlp, l->fc1_w, d_normed, l->fc1_b, ffn, dim, seq);
        kl_gelu(r, d_mlp, seq * ffn);
        kl_gemm(r, d_normed, l->fc2_w, d_mlp, l->fc2_b, dim, ffn, seq);

        if (l->ls2)
            kl_layerscale_add(r, d_hidden, d_normed, l->ls2, seq * dim, dim);
        else
            kl_add(r, d_hidden, d_normed, seq * dim);
    }

    /* 4. Final LN */
    if (r->dino_final_ln_w) {
        kl_layernorm(r, d_out, d_hidden, r->dino_final_ln_w, r->dino_final_ln_b, seq, dim);
    } else {
        cuMemcpyDtoDAsync(d_out, d_hidden, (size_t)seq * dim * sizeof(float), r->stream);
    }
}

/* Pre-compute DiT cross-attention K,V for all blocks (constant across diffusion steps) */
static void precompute_dit_ca_kv(cuda_hy3d_runner *r, CUdeviceptr d_context) {
    for (int i = 0; i < DIT_DEPTH; i++) {
        dit_block_gpu *b = &r->dit_blocks[i];

        /* Allocate K,V buffers if not already done */
        if (!r->dit_ca_K[i]) {
            r->dit_ca_K[i] = gpu_alloc((size_t)DINO_SEQ_LEN * DIT_HIDDEN * sizeof(float));
            r->dit_ca_V[i] = gpu_alloc((size_t)DINO_SEQ_LEN * DIT_HIDDEN * sizeof(float));
        }

        /* K = context @ ca_k_w.T  (no bias) */
        kl_gemm(r, r->dit_ca_K[i], b->ca_k_w, d_context, 0,
                DIT_HIDDEN, DIT_CONTEXT_DIM, DINO_SEQ_LEN);
        /* V = context @ ca_v_w.T  (no bias) */
        kl_gemm(r, r->dit_ca_V[i], b->ca_v_w, d_context, 0,
                DIT_HIDDEN, DIT_CONTEXT_DIM, DINO_SEQ_LEN);

        /* Apply RMS norm to K for QK normalization */
        if (b->ca_k_norm_w) {
            kl_rms_norm(r, r->dit_ca_K[i], b->ca_k_norm_w,
                        DINO_SEQ_LEN, DIT_HEADS, DIT_HEAD_DIM, DIT_HIDDEN);
        }
    }
    r->ca_kv_precomputed = 1;
}

/* Stage 2: Single DiT forward pass
 * Input:  d_latents [4096, 64] F32, timestep (scalar), d_context [1370, 1024] F32
 * Output: d_output [4096, 64] F32 */
static void run_dit_forward(cuda_hy3d_runner *r, CUdeviceptr d_latents,
                             float timestep, CUdeviceptr d_context __attribute__((unused)),
                             CUdeviceptr d_output) {
    const int N = DIT_INPUT_SIZE;
    const int C = DIT_IN_CHANNELS;
    const int H_dim = DIT_HIDDEN;
    const int heads = DIT_HEADS;
    const int hd = DIT_HEAD_DIM;
    const int ffn = H_dim * 4;
    const int ctx_len = DINO_SEQ_LEN;

    /* Scratch layout:
     * 0: hidden [N, H_dim]
     * 1: Q/K/V [3 * N * H_dim]
     * 2: attn_out [N, H_dim]
     * 3: mlp [N, ffn]
     * 4: normed [N, H_dim]
     * 5: t_emb [H_dim] + t_mlp [ffn]
     * 6: cross_Q [N, H_dim]
     */
    ensure_scratch(r, 0, (size_t)N * H_dim * sizeof(float));
    ensure_scratch(r, 1, (size_t)3 * N * H_dim * sizeof(float));
    ensure_scratch(r, 2, (size_t)N * H_dim * sizeof(float));
    ensure_scratch(r, 3, (size_t)N * ffn * sizeof(float));
    ensure_scratch(r, 4, (size_t)N * H_dim * sizeof(float));
    ensure_scratch(r, 5, (size_t)(H_dim + ffn) * sizeof(float));
    ensure_scratch(r, 6, (size_t)N * H_dim * sizeof(float));

    CUdeviceptr d_hidden = r->scratch[0];
    CUdeviceptr d_qkv    = r->scratch[1];
    CUdeviceptr d_attn   = r->scratch[2];
    CUdeviceptr d_mlp    = r->scratch[3];
    CUdeviceptr d_normed = r->scratch[4];
    CUdeviceptr d_temb   = r->scratch[5];
    CUdeviceptr d_tmlp   = d_temb + (size_t)H_dim * sizeof(float);
    CUdeviceptr d_cross_Q = r->scratch[6];

    /* 1. Embed latents: [N, C] → [N, H_dim] */
    kl_gemm(r, d_hidden, r->dit_x_emb_w, d_latents, r->dit_x_emb_b,
            H_dim, C, N);

    /* 2. Timestep embedding: sinusoidal → MLP(SiLU) → add to all tokens */
    kl_timestep_embed(r, d_temb, timestep, H_dim);
    kl_gemm(r, d_tmlp, r->dit_t_mlp0_w, d_temb, r->dit_t_mlp0_b,
            ffn, H_dim, 1);
    kl_silu(r, d_tmlp, ffn);
    kl_gemm(r, d_temb, r->dit_t_mlp2_w, d_tmlp, r->dit_t_mlp2_b,
            H_dim, ffn, 1);

    /* Broadcast-add timestep embedding to all N tokens */
    /* We'll use a simple loop: replicate d_temb N times and add */
    /* For efficiency, use the bilinear_upsample as a broadcast or just do N adds */
    /* Simple approach: do it per-row via GPU kernel. Use add_f32 with tiling. */
    /* Actually, let's just iterate on CPU side — N is fixed at 4096 */
    for (int tok = 0; tok < N; tok++) {
        CUdeviceptr dst = d_hidden + (size_t)tok * H_dim * sizeof(float);
        kl_add(r, dst, d_temb, H_dim);
    }

    /* 3. Transformer blocks */
    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        dit_block_gpu *blk = &r->dit_blocks[bi];

        /* === Self-attention === */
        kl_layernorm(r, d_normed, d_hidden, blk->norm1_w, blk->norm1_b, N, H_dim);

        CUdeviceptr d_Q = d_qkv;
        CUdeviceptr d_K = d_qkv + (size_t)N * H_dim * sizeof(float);
        CUdeviceptr d_V = d_qkv + (size_t)2 * N * H_dim * sizeof(float);

        kl_gemm(r, d_Q, blk->sa_q_w, d_normed, 0, H_dim, H_dim, N);
        kl_gemm(r, d_K, blk->sa_k_w, d_normed, 0, H_dim, H_dim, N);
        kl_gemm(r, d_V, blk->sa_v_w, d_normed, 0, H_dim, H_dim, N);

        /* QK RMSNorm */
        if (blk->sa_q_norm_w)
            kl_rms_norm(r, d_Q, blk->sa_q_norm_w, N, heads, hd, H_dim);
        if (blk->sa_k_norm_w)
            kl_rms_norm(r, d_K, blk->sa_k_norm_w, N, heads, hd, H_dim);

        kl_self_attn(r, d_attn, d_Q, d_K, d_V, N, H_dim, heads, hd);

        kl_gemm(r, d_normed, blk->sa_out_w, d_attn, blk->sa_out_b, H_dim, H_dim, N);
        kl_add(r, d_hidden, d_normed, N * H_dim);

        /* === Cross-attention === */
        kl_layernorm(r, d_normed, d_hidden, blk->norm2_w, blk->norm2_b, N, H_dim);

        /* Q from hidden */
        kl_gemm(r, d_cross_Q, blk->ca_q_w, d_normed, 0, H_dim, H_dim, N);
        if (blk->ca_q_norm_w)
            kl_rms_norm(r, d_cross_Q, blk->ca_q_norm_w, N, heads, hd, H_dim);

        /* K, V pre-computed from context */
        kl_cross_attn(r, d_attn, d_cross_Q, r->dit_ca_K[bi], r->dit_ca_V[bi],
                      N, ctx_len, H_dim, heads, hd);

        kl_gemm(r, d_normed, blk->ca_out_w, d_attn, blk->ca_out_b, H_dim, H_dim, N);
        kl_add(r, d_hidden, d_normed, N * H_dim);

        /* === MLP === */
        if (blk->norm3_w) {
            kl_layernorm(r, d_normed, d_hidden, blk->norm3_w, blk->norm3_b, N, H_dim);
        } else {
            cuMemcpyDtoDAsync(d_normed, d_hidden, (size_t)N * H_dim * sizeof(float), r->stream);
        }

        kl_gemm(r, d_mlp, blk->mlp_fc1_w, d_normed, blk->mlp_fc1_b, ffn, H_dim, N);
        kl_gelu(r, d_mlp, N * ffn);
        kl_gemm(r, d_normed, blk->mlp_fc2_w, d_mlp, blk->mlp_fc2_b, H_dim, ffn, N);
        kl_add(r, d_hidden, d_normed, N * H_dim);
    }

    /* 4. Final layer: LN → Linear → output [N, C] */
    kl_layernorm(r, d_normed, d_hidden, r->dit_final_ln_w, r->dit_final_ln_b, N, H_dim);
    kl_gemm(r, d_output, r->dit_final_linear_w, d_normed, r->dit_final_linear_b,
            C, H_dim, N);
}

/* Stage 3: ShapeVAE single transformer block */
static void run_vae_block(cuda_hy3d_runner *r, vae_block_gpu *b,
                           CUdeviceptr d_in, CUdeviceptr d_out,
                           CUdeviceptr d_scratch) {
    const int N = VAE_NUM_LATENTS;
    const int W = VAE_WIDTH;
    const int H = VAE_HEADS;
    const int HD = VAE_HEAD_DIM;
    const int MLP = 4 * W;

    /* Scratch layout within d_scratch:
     * ln1_out [N*W], qkv [N*3*W], Q [N*W], K [N*W], V [N*W],
     * attn_out [N*W], proj [N*W], res1 [N*W],
     * ln2_out [N*W], mlp_h [N*MLP], mlp_out [N*W] */
    size_t off = 0;
    CUdeviceptr d_ln1   = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_qkv   = d_scratch + off; off += (size_t)N * 3 * W * sizeof(float);
    CUdeviceptr d_Q     = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_K     = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_V     = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_aout  = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_proj  = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_res1  = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_ln2   = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_mlph  = d_scratch + off; off += (size_t)N * MLP * sizeof(float);
    CUdeviceptr d_mlpo  = d_scratch + off;

    /* LN1 */
    kl_layernorm(r, d_ln1, d_in, b->ln1_w, b->ln1_b, N, W);

    /* Fused QKV projection */
    kl_gemm(r, d_qkv, b->qkv_w, d_ln1, 0, 3 * W, W, N);

    /* Split interleaved QKV */
    kl_split_qkv(r, d_Q, d_K, d_V, d_qkv, N, H, HD);

    /* QK normalization */
    if (b->use_qk_norm) {
        kl_qk_layernorm(r, d_Q, b->q_norm_w, b->q_norm_b, N, H, HD, W);
        kl_qk_layernorm(r, d_K, b->k_norm_w, b->k_norm_b, N, H, HD, W);
    }

    /* Self-attention */
    kl_self_attn(r, d_aout, d_Q, d_K, d_V, N, W, H, HD);

    /* Output projection */
    kl_gemm(r, d_proj, b->proj_w, d_aout, b->proj_b, W, W, N);

    /* Residual 1: res1 = input + proj */
    cuMemcpyDtoDAsync(d_res1, d_in, (size_t)N * W * sizeof(float), r->stream);
    kl_add(r, d_res1, d_proj, N * W);

    /* LN2 → MLP → Residual 2 */
    kl_layernorm(r, d_ln2, d_res1, b->ln2_w, b->ln2_b, N, W);
    kl_gemm(r, d_mlph, b->mlp_fc_w, d_ln2, b->mlp_fc_b, MLP, W, N);
    kl_gelu(r, d_mlph, N * MLP);
    kl_gemm(r, d_mlpo, b->mlp_proj_w, d_mlph, b->mlp_proj_b, W, MLP, N);

    /* Output = res1 + mlp_out */
    cuMemcpyDtoDAsync(d_out, d_res1, (size_t)N * W * sizeof(float), r->stream);
    kl_add(r, d_out, d_mlpo, N * W);
}

/* Stage 3: ShapeVAE decode + SDF query
 * Input:  d_latents [4096, 64] F32
 * Output: sdf_grid [grid_res^3] F32 (on CPU) */
static void run_shapevae(cuda_hy3d_runner *r, CUdeviceptr d_latents,
                          int grid_res, float *sdf_out) {
    const int N = VAE_NUM_LATENTS;
    const int E = VAE_EMBED_DIM;
    const int W = VAE_WIDTH;

    /* Allocate decoder buffers */
    CUdeviceptr d_dec_a = gpu_alloc((size_t)N * W * sizeof(float));
    CUdeviceptr d_dec_b = gpu_alloc((size_t)N * W * sizeof(float));

    /* VAE block scratch size (generous) */
    size_t block_scratch = (size_t)N * (W * 12 + 4 * W * 4) * sizeof(float);
    CUdeviceptr d_block_scratch = gpu_alloc(block_scratch);

    /* Post-KL projection: [N, E] → [N, W] */
    kl_gemm(r, d_dec_a, r->vae_post_kl_w, d_latents, r->vae_post_kl_b, W, E, N);

    /* Run transformer blocks */
    CUdeviceptr d_cur = d_dec_a;
    CUdeviceptr d_next = d_dec_b;
    for (int i = 0; i < VAE_DEC_LAYERS; i++) {
        run_vae_block(r, &r->vae_blocks[i], d_cur, d_next, d_block_scratch);
        CUdeviceptr tmp = d_cur; d_cur = d_next; d_next = tmp;
    }

    /* d_cur now contains decoded latents [N, W] */
    /* Query SDF at grid points in batches */
    int total_points = grid_res * grid_res * grid_res;
    int batch_size = 65536;  /* 64K points per batch */
    float bounds[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    /* Allocate GPU buffers for SDF query */
    CUdeviceptr d_coords = gpu_alloc((size_t)batch_size * 3 * sizeof(float));
    CUdeviceptr d_fourier = gpu_alloc((size_t)batch_size * VAE_FOURIER_DIM * sizeof(float));
    CUdeviceptr d_query_proj = gpu_alloc((size_t)batch_size * W * sizeof(float));
    CUdeviceptr d_sdf_out = gpu_alloc((size_t)batch_size * sizeof(float));

    /* Geometry decoder scratch */
    size_t geo_scratch_sz = (size_t)batch_size * (W * 10 + 4 * W * 4) * sizeof(float);
    CUdeviceptr d_geo_scratch = gpu_alloc(geo_scratch_sz);

    float dx = (bounds[3] - bounds[0]) / (float)(grid_res - 1);
    float dy = (bounds[4] - bounds[1]) / (float)(grid_res - 1);
    float dz = (bounds[5] - bounds[2]) / (float)(grid_res - 1);

    for (int start = 0; start < total_points; start += batch_size) {
        int count = (start + batch_size <= total_points) ? batch_size : (total_points - start);

        /* Generate 3D coordinates on CPU and upload */
        float *coords = (float *)malloc((size_t)count * 3 * sizeof(float));
        for (int i = 0; i < count; i++) {
            int idx = start + i;
            int iz = idx % grid_res;
            int iy = (idx / grid_res) % grid_res;
            int ix = idx / (grid_res * grid_res);
            coords[i*3+0] = bounds[0] + ix * dx;
            coords[i*3+1] = bounds[1] + iy * dy;
            coords[i*3+2] = bounds[2] + iz * dz;
        }
        cuMemcpyHtoDAsync(d_coords, coords, (size_t)count * 3 * sizeof(float), r->stream);
        free(coords);

        /* Fourier embedding */
        kl_fourier_embed(r, d_fourier, d_coords, r->vae_fourier_freqs,
                         count, VAE_NUM_FREQS, VAE_FOURIER_DIM);

        /* Query projection: [count, 51] → [count, W] */
        kl_gemm(r, d_query_proj, r->vae_geo.query_proj_w, d_fourier,
                r->vae_geo.query_proj_b, W, VAE_FOURIER_DIM, count);

        /* Cross-attention with decoded latents */
        vae_geo_decoder_gpu *g = &r->vae_geo;

        /* LN on queries and latents */
        size_t geo_off = 0;
        CUdeviceptr d_g_ln1  = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_ln2  = d_geo_scratch + geo_off; geo_off += (size_t)N * W * sizeof(float);
        CUdeviceptr d_g_Q    = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_KV   = d_geo_scratch + geo_off; geo_off += (size_t)N * 2 * W * sizeof(float);
        CUdeviceptr d_g_K    = d_geo_scratch + geo_off; geo_off += (size_t)N * W * sizeof(float);
        CUdeviceptr d_g_V    = d_geo_scratch + geo_off; geo_off += (size_t)N * W * sizeof(float);
        CUdeviceptr d_g_aout = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_proj = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_res  = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_ln3  = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_mlph = d_geo_scratch + geo_off; geo_off += (size_t)count * 4 * W * sizeof(float);
        CUdeviceptr d_g_mlpo = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_post = d_geo_scratch + geo_off;

        kl_layernorm(r, d_g_ln1, d_query_proj, g->ln1_w, g->ln1_b, count, W);
        kl_layernorm(r, d_g_ln2, d_cur, g->ln2_w, g->ln2_b, N, W);

        /* Q from queries, KV from latents */
        kl_gemm(r, d_g_Q, g->c_q_w, d_g_ln1, 0, W, W, count);
        kl_gemm(r, d_g_KV, g->c_kv_w, d_g_ln2, 0, 2 * W, W, N);

        /* Split KV */
        kl_split_kv(r, d_g_K, d_g_V, d_g_KV, N, VAE_HEADS, VAE_HEAD_DIM);

        /* QK norm */
        if (g->use_qk_norm) {
            kl_qk_layernorm(r, d_g_Q, g->q_norm_w, g->q_norm_b,
                            count, VAE_HEADS, VAE_HEAD_DIM, W);
            kl_qk_layernorm(r, d_g_K, g->k_norm_w, g->k_norm_b,
                            N, VAE_HEADS, VAE_HEAD_DIM, W);
        }

        /* Cross-attention */
        kl_cross_attn(r, d_g_aout, d_g_Q, d_g_K, d_g_V,
                      count, N, W, VAE_HEADS, VAE_HEAD_DIM);

        /* Output projection + residual */
        kl_gemm(r, d_g_proj, g->c_proj_w, d_g_aout, g->c_proj_b, W, W, count);
        cuMemcpyDtoDAsync(d_g_res, d_query_proj, (size_t)count * W * sizeof(float), r->stream);
        kl_add(r, d_g_res, d_g_proj, count * W);

        /* MLP block */
        kl_layernorm(r, d_g_ln3, d_g_res, g->ln3_w, g->ln3_b, count, W);
        kl_gemm(r, d_g_mlph, g->mlp_fc_w, d_g_ln3, g->mlp_fc_b, 4 * W, W, count);
        kl_gelu(r, d_g_mlph, count * 4 * W);
        kl_gemm(r, d_g_mlpo, g->mlp_proj_w, d_g_mlph, g->mlp_proj_b, W, 4 * W, count);
        cuMemcpyDtoDAsync(d_g_post, d_g_res, (size_t)count * W * sizeof(float), r->stream);
        kl_add(r, d_g_post, d_g_mlpo, count * W);

        /* Post LN */
        if (g->ln_post_w) {
            kl_layernorm(r, d_g_ln1, d_g_post, g->ln_post_w, g->ln_post_b, count, W);
        } else {
            cuMemcpyDtoDAsync(d_g_ln1, d_g_post, (size_t)count * W * sizeof(float), r->stream);
        }

        /* Final output projection: [count, W] → [count, 1] */
        kl_gemm(r, d_sdf_out, g->output_w, d_g_ln1, g->output_b, 1, W, count);

        /* Download SDF values */
        cuMemcpyDtoHAsync(sdf_out + start, d_sdf_out,
                          (size_t)count * sizeof(float), r->stream);
    }

    cuStreamSynchronize(r->stream);

    /* Cleanup */
    cuMemFree(d_dec_a);
    cuMemFree(d_dec_b);
    cuMemFree(d_block_scratch);
    cuMemFree(d_coords);
    cuMemFree(d_fourier);
    cuMemFree(d_query_proj);
    cuMemFree(d_sdf_out);
    cuMemFree(d_geo_scratch);
}

/* ======================================================================== */
/* Random noise generation (CPU, Box-Muller)                                */
/* ======================================================================== */

static void generate_randn(float *buf, int n, uint32_t seed) {
    /* Simple xorshift128+ PRNG + Box-Muller */
    uint64_t s0 = seed ? seed : (uint64_t)time(NULL);
    uint64_t s1 = s0 ^ 0x6c62272e07bb0142ULL;

    for (int i = 0; i < n; i += 2) {
        /* xorshift128+ */
        uint64_t x = s0;
        uint64_t y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t r1 = s1 + y;

        x = s0; y = s1; s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t r2 = s1 + y;

        /* Box-Muller */
        double u1 = ((r1 >> 11) + 0.5) / 2097152.0;
        double u2 = ((r2 >> 11) + 0.5) / 2097152.0;
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * 3.141592653589793 * u2;
        buf[i] = (float)(r * cos(theta));
        if (i + 1 < n) buf[i+1] = (float)(r * sin(theta));
    }
}

/* ======================================================================== */
/* Public API                                                               */
/* ======================================================================== */

cuda_hy3d_runner *cuda_hy3d_init(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "HY3D: cuew init failed\n");
        return NULL;
    }
    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "HY3D: cuInit failed\n");
        return NULL;
    }

    cuda_hy3d_runner *r = (cuda_hy3d_runner *)calloc(1, sizeof(cuda_hy3d_runner));
    r->verbose = verbose;

    CU_CHECK_NULL(cuDeviceGet(&r->device, device_id));
    CU_CHECK_NULL(cuCtxCreate(&r->ctx, 0, r->device));
    CU_CHECK_NULL(cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING));

    if (verbose) {
        char name[256];
        cuDeviceGetName(name, sizeof(name), r->device);
        fprintf(stderr, "HY3D: using GPU %d: %s\n", device_id, name);
    }

    if (hy3d_compile_kernels(r) != 0) {
        fprintf(stderr, "HY3D: kernel compilation failed\n");
        free(r);
        return NULL;
    }

    return r;
}

int cuda_hy3d_load_weights(cuda_hy3d_runner *r,
                           const char *conditioner_path,
                           const char *model_path,
                           const char *vae_path) {
    if (!r) return -1;

    if (conditioner_path && load_dino_weights(r, conditioner_path) != 0)
        return -1;
    if (model_path && load_dit_weights(r, model_path) != 0)
        return -1;
    if (vae_path && load_vae_weights(r, vae_path) != 0)
        return -1;

    return 0;
}

hy3d_mesh cuda_hy3d_predict(cuda_hy3d_runner *r,
                            const uint8_t *rgb, int w, int h,
                            int n_steps, float guidance_scale,
                            int grid_res, uint32_t seed) {
    hy3d_mesh result = {0};
    if (!r || !r->dino_loaded || !r->dit_loaded || !r->vae_loaded) {
        fprintf(stderr, "HY3D: runner not fully initialized\n");
        return result;
    }

    if (n_steps <= 0) n_steps = 30;
    if (guidance_scale <= 0.0f) guidance_scale = 7.5f;
    if (grid_res <= 0) grid_res = 256;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* ---- Stage 1: DINOv2 image encoding ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 1 - DINOv2 encoding...\n");

    /* Upload and preprocess image: resize to 518x518, normalize with ImageNet stats */
    CUdeviceptr d_rgb = gpu_alloc((size_t)w * h * 3);
    cuMemcpyHtoDAsync(d_rgb, rgb, (size_t)w * h * 3, r->stream);

    CUdeviceptr d_image = gpu_alloc((size_t)3 * DINO_IMG_SIZE * DINO_IMG_SIZE * sizeof(float));
    {
        int dw = DINO_IMG_SIZE, dh = DINO_IMG_SIZE;
        float mean0 = 0.485f, mean1 = 0.456f, mean2 = 0.406f;
        float istd0 = 1.0f/0.229f, istd1 = 1.0f/0.224f, istd2 = 1.0f/0.225f;
        void *args[] = {&d_image, &d_rgb, &w, &h, &dw, &dh,
                        &mean0, &mean1, &mean2, &istd0, &istd1, &istd2};
        cuLaunchKernel(r->fn_resize_normalize,
                       (unsigned)((dw*dh+255)/256), 1, 1,
                       256, 1, 1, 0, r->stream, args, NULL);
    }

    /* DINOv2 forward */
    CUdeviceptr d_dino_out = gpu_alloc((size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
    run_dinov2(r, d_image, d_dino_out);
    cuMemFree(d_rgb);
    cuMemFree(d_image);

    /* ---- Stage 2: DiT diffusion with flow matching ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 2 - DiT diffusion (%d steps)...\n", n_steps);

    /* Pre-compute cross-attention K,V */
    precompute_dit_ca_kv(r, d_dino_out);

    /* Create initial noise */
    int latent_size = DIT_INPUT_SIZE * DIT_IN_CHANNELS;
    float *noise_cpu = (float *)malloc((size_t)latent_size * sizeof(float));
    generate_randn(noise_cpu, latent_size, seed);

    CUdeviceptr d_latents = gpu_alloc((size_t)latent_size * sizeof(float));
    cuMemcpyHtoDAsync(d_latents, noise_cpu, (size_t)latent_size * sizeof(float), r->stream);
    free(noise_cpu);

    CUdeviceptr d_pred_cond = gpu_alloc((size_t)latent_size * sizeof(float));
    CUdeviceptr d_pred_uncond = gpu_alloc((size_t)latent_size * sizeof(float));
    CUdeviceptr d_pred_combined = gpu_alloc((size_t)latent_size * sizeof(float));

    /* Create zero context for unconditional pass */
    CUdeviceptr d_uncond_ctx = gpu_alloc((size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
    cuMemsetD8Async(d_uncond_ctx, 0, (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float), r->stream);

    /* Pre-compute unconditional cross-attention K,V */
    /* We need separate K,V arrays for unconditional pass.
     * For now, we'll do both conditional and unconditional passes per step,
     * recomputing K,V each time (since precompute_dit_ca_kv overwrites).
     * TODO: optimize by storing both sets. */

    /* Flow matching: timestep schedule from 1.0 → 0.0 */
    for (int step = 0; step < n_steps; step++) {
        float t_current = 1.0f - (float)step / (float)n_steps;
        float t_next = 1.0f - (float)(step + 1) / (float)n_steps;
        float dt = t_current - t_next;

        if (r->verbose && (step % 5 == 0 || step == n_steps - 1))
            fprintf(stderr, "  step %d/%d (t=%.3f)\n", step+1, n_steps, t_current);

        /* Conditional pass */
        precompute_dit_ca_kv(r, d_dino_out);
        run_dit_forward(r, d_latents, t_current, d_dino_out, d_pred_cond);

        /* Unconditional pass */
        precompute_dit_ca_kv(r, d_uncond_ctx);
        run_dit_forward(r, d_latents, t_current, d_uncond_ctx, d_pred_uncond);

        /* CFG combination */
        kl_cfg_combine(r, d_pred_combined, d_pred_cond, d_pred_uncond,
                       guidance_scale, latent_size);

        /* Euler step: x_{t-dt} = x_t - dt * v */
        kl_euler_step(r, d_latents, d_pred_combined, dt, latent_size);
    }

    cuMemFree(d_pred_cond);
    cuMemFree(d_pred_uncond);
    cuMemFree(d_pred_combined);
    cuMemFree(d_uncond_ctx);
    cuMemFree(d_dino_out);

    /* ---- Stage 3: ShapeVAE decode + SDF query ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 3 - ShapeVAE decode (grid %d^3)...\n", grid_res);

    int total_pts = grid_res * grid_res * grid_res;
    float *sdf_grid = (float *)malloc((size_t)total_pts * sizeof(float));
    run_shapevae(r, d_latents, grid_res, sdf_grid);
    cuMemFree(d_latents);

    /* ---- Stage 4: Marching cubes ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 4 - Marching cubes...\n");

    mc_mesh mc = mc_marching_cubes(sdf_grid, grid_res, grid_res, grid_res, 0.0f, NULL);
    free(sdf_grid);

    result.vertices = mc.vertices;
    result.triangles = mc.triangles;
    result.n_verts = mc.n_verts;
    result.n_tris = mc.n_tris;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (r->verbose)
        fprintf(stderr, "HY3D: done in %.2fs (%d verts, %d tris)\n",
                elapsed, result.n_verts, result.n_tris);

    return result;
}

void cuda_hy3d_free(cuda_hy3d_runner *r) {
    if (!r) return;

    /* Free scratch buffers */
    for (int i = 0; i < 8; i++) {
        if (r->scratch[i]) cuMemFree(r->scratch[i]);
    }

    /* Free pre-computed K,V */
    for (int i = 0; i < DIT_DEPTH; i++) {
        if (r->dit_ca_K[i]) cuMemFree(r->dit_ca_K[i]);
        if (r->dit_ca_V[i]) cuMemFree(r->dit_ca_V[i]);
    }

    /* Free Fourier frequencies */
    if (r->vae_fourier_freqs) cuMemFree(r->vae_fourier_freqs);

    /* Note: individual weight buffers are GPU allocations that should also be freed.
     * For a full cleanup, iterate all CUdeviceptr fields. For now, destroying
     * the CUDA context reclaims all GPU memory. */

    if (r->module) cuModuleUnload(r->module);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->ctx) cuCtxDestroy(r->ctx);

    free(r);
}
