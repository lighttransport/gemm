/*
 * cuda_hy3d_ops.h - Modular kernel-launch wrappers for Hunyuan3D-2.1 CUDA runner
 *
 * All ML operations as static inline functions operating on hy3d_ops context.
 * Each op takes (hy3d_ops *ops, CUstream stream, ...) and launches the
 * appropriate kernel.  The runner struct holds a single `hy3d_ops ops;` field.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#ifndef CUDA_HY3D_OPS_H
#define CUDA_HY3D_OPS_H

#include <math.h>
#include <stdio.h>
#include <string.h>

/* This header requires CUDA driver API types.
   Include cuew.h (or cuda.h) BEFORE including this header. */
#ifndef __CUEW_H__
#error "cuda_hy3d_ops.h requires cuew.h to be included first"
#endif

/* ======================================================================== */
/* Ops context: holds all compiled kernel function pointers                 */
/* ======================================================================== */

typedef struct {
    /* Common kernels (from cuda_kernels_common.h) */
    CUfunction layernorm;
    CUfunction gemm;
    CUfunction gemm_tiled;
    CUfunction gelu;
    CUfunction add;
    CUfunction silu;
    CUfunction resize_normalize;
    CUfunction patch_embed;
    CUfunction cls_pos_embed;
    CUfunction attn_prefill;
    CUfunction flash_attn_tiled;
    CUfunction kv_transpose;
    CUfunction bilinear_upsample;

    /* HY3D-specific kernels */
    CUfunction rms_norm;
    CUfunction qk_layernorm;
    CUfunction layerscale_add;
    CUfunction cross_attn;
    CUfunction fourier_embed;
    CUfunction timestep_embed;
    CUfunction euler_step;
    CUfunction cfg_combine;
    CUfunction split_qkv;
    CUfunction split_kv;
    CUfunction broadcast_add;
    CUfunction concat_first;     /* prepend token */
    CUfunction strip_first;      /* drop first token */
    CUfunction concat_last_dim;  /* concat along last dim */
    CUfunction gemm_f32;         /* pure F32 GEMM (no F16 conversion) */

    int sm_version;
    int use_f32_gemm;            /* 0=F16 weights (default), 1=F32 weights */
} hy3d_ops;

/* ======================================================================== */
/* Load all kernel functions from a compiled module                         */
/* ======================================================================== */

static int hy3d_ops_load(hy3d_ops *ops, CUmodule module, int sm_version) {
    memset(ops, 0, sizeof(*ops));
    ops->sm_version = sm_version;

    #define GET_FN(name, field) \
        if (cuModuleGetFunction(&ops->field, module, name) != CUDA_SUCCESS) { \
            fprintf(stderr, "HY3D ops: failed to get kernel '%s'\n", name); \
            return -1; \
        }

    /* Common kernels */
    GET_FN("layernorm_f32",          layernorm);
    GET_FN("gemm_f16_f32",           gemm);
    GET_FN("gemm_tiled_f16_f32",     gemm_tiled);
    /* DINOv2 + DiT + ShapeVAE all use torch nn.GELU() with exact erf form.
     * The tanh-approx gelu_f32 accumulates ~1e-2 mean err over 24 blocks. */
    GET_FN("gelu_exact_f32",         gelu);
    GET_FN("add_f32",                add);
    GET_FN("silu_f32",               silu);
    GET_FN("resize_normalize",       resize_normalize);
    GET_FN("patch_embed_conv2d",     patch_embed);
    GET_FN("cls_pos_embed",          cls_pos_embed);
    GET_FN("kv_transpose",           kv_transpose);

    /* Use prefill attention on sm_70+, fallback to tiled on older */
    if (sm_version >= 70) {
        GET_FN("attn_prefill_f32",   attn_prefill);
    } else {
        GET_FN("flash_attn_tiled_f32", attn_prefill);
    }
    GET_FN("flash_attn_tiled_f32",   flash_attn_tiled);
    GET_FN("bilinear_upsample_f32",  bilinear_upsample);

    /* HY3D-specific kernels */
    GET_FN("rms_norm_f32",                rms_norm);
    GET_FN("qk_layernorm_f32",            qk_layernorm);
    GET_FN("layerscale_add_f32",          layerscale_add);
    GET_FN("cross_attn_f32",              cross_attn);
    GET_FN("fourier_embed_3d_f32",        fourier_embed);
    GET_FN("timestep_embed_f32",          timestep_embed);
    GET_FN("euler_step_f32",              euler_step);
    GET_FN("cfg_combine_f32",             cfg_combine);
    GET_FN("split_qkv_interleaved_f32",   split_qkv);
    GET_FN("split_kv_interleaved_f32",    split_kv);
    GET_FN("broadcast_add_f32",           broadcast_add);
    GET_FN("concat_token_f32",             concat_first);
    GET_FN("strip_first_token_f32",       strip_first);
    GET_FN("concat_f32",                  concat_last_dim);
    GET_FN("gemm_f32_f32",               gemm_f32);

    #undef GET_FN
    return 0;
}

/* ======================================================================== */
/* Kernel launch wrappers                                                   */
/* ======================================================================== */

/* ---- LayerNorm: dst = LN(src) with weight w, bias b ---- */
static inline void op_layernorm(hy3d_ops *ops, CUstream stream,
                                CUdeviceptr dst, CUdeviceptr src,
                                CUdeviceptr w, CUdeviceptr b,
                                int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    cuLaunchKernel(ops->layernorm, (unsigned)n_tok, 1, 1,
                   256, 1, 1, 256 * sizeof(float), stream, args, NULL);
}

/* ---- GEMM: Y = W @ X + bias ---- */
/* When use_f32_gemm=0: W is F16 (half_raw), bias/X/Y are F32.
 * When use_f32_gemm=1: all F32.
 * Both use same grid/block config: (ceil(n_out/64), ceil(n_tok/16)), blockDim=(16,16) */
static inline void op_gemm(hy3d_ops *ops, CUstream stream,
                           CUdeviceptr Y, CUdeviceptr W,
                           CUdeviceptr X, CUdeviceptr bias,
                           int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    CUfunction fn = ops->use_f32_gemm ? ops->gemm_f32 : ops->gemm_tiled;
    cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, stream, args, NULL);
}

/* Force the pure F32 GEMM backend for a single call. */
/* ---- GELU activation in-place ---- */
static inline void op_gelu(hy3d_ops *ops, CUstream stream,
                           CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(ops->gelu, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- SiLU activation in-place ---- */
static inline void op_silu(hy3d_ops *ops, CUstream stream,
                           CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(ops->silu, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Element-wise add: dst += src ---- */
static inline void op_add(hy3d_ops *ops, CUstream stream,
                          CUdeviceptr dst, CUdeviceptr src, int n) {
    void *args[] = {&dst, &src, &n};
    cuLaunchKernel(ops->add, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- LayerScale residual: dst[i] += src[i] * scale[i % dim] ---- */
static inline void op_layerscale_add(hy3d_ops *ops, CUstream stream,
                                     CUdeviceptr dst, CUdeviceptr src,
                                     CUdeviceptr scale, int n, int dim) {
    void *args[] = {&dst, &src, &scale, &n, &dim};
    cuLaunchKernel(ops->layerscale_add, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- RMSNorm: per-head RMSNorm (for DiT QK normalization) ---- */
/* data: [n_tok, stride], w: [head_dim] */
static inline void op_rms_norm(hy3d_ops *ops, CUstream stream,
                               CUdeviceptr data, CUdeviceptr w,
                               int n_tok, int n_heads, int head_dim,
                               int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data, &w, &n_tok, &n_heads, &head_dim, &stride, &eps};
    cuLaunchKernel(ops->rms_norm, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- QK LayerNorm: per-head LayerNorm (for ShapeVAE QK normalization) ---- */
/* data: [n_tok, stride], w: [head_dim], b: [head_dim] (b may be 0/NULL) */
static inline void op_qk_layernorm(hy3d_ops *ops, CUstream stream,
                                   CUdeviceptr data,
                                   CUdeviceptr w, CUdeviceptr b,
                                   int n_tok, int n_heads, int head_dim,
                                   int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data, &w, &b, &n_tok, &n_heads, &head_dim, &stride, &eps};
    cuLaunchKernel(ops->qk_layernorm, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Cross-attention: Q/K/V may have different sequence lengths ---- */
/* Q: [q_len, dim], K: [kv_len, dim], V: [kv_len, dim], out: [q_len, dim] */
/* dim = n_heads * head_dim */
static inline void op_cross_attn(hy3d_ops *ops, CUstream stream,
                                 CUdeviceptr out,
                                 CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                                 int q_len, int kv_len, int dim,
                                 int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int nt = 128;  /* threads per block */
    size_t smem = (size_t)(kv_len + nt) * sizeof(float);
    void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim,
                    &n_heads, &head_dim, &scale};
    cuLaunchKernel(ops->cross_attn, (unsigned)n_heads, (unsigned)q_len, 1,
                   (unsigned)nt, 1, 1, smem, stream, args, NULL);
}

/* ---- Self-attention: wrapper over cross_attn with q_len == kv_len ---- */
static inline void op_self_attn(hy3d_ops *ops, CUstream stream,
                                CUdeviceptr out,
                                CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                                int n_tok, int dim,
                                int n_heads, int head_dim) {
    op_cross_attn(ops, stream, out, Q, K, V,
                  n_tok, n_tok, dim, n_heads, head_dim);
}

/* ---- 3D Fourier coordinate embedding ---- */
/* coords: [N, 3], freqs: [num_freqs], out: [N, out_dim] */
/* out_dim = 3 * (2 * num_freqs + 1) */
static inline void op_fourier_embed(hy3d_ops *ops, CUstream stream,
                                    CUdeviceptr out, CUdeviceptr coords,
                                    CUdeviceptr freqs,
                                    int N, int num_freqs, int out_dim) {
    void *args[] = {&out, &coords, &freqs, &N, &num_freqs, &out_dim};
    cuLaunchKernel(ops->fourier_embed, (unsigned)((N + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Sinusoidal timestep embedding ---- */
/* out: [dim], t: scalar timestep value */
static inline void op_timestep_embed(hy3d_ops *ops, CUstream stream,
                                     CUdeviceptr out, float t, int dim) {
    int half = dim / 2;
    void *args[] = {&out, &t, &dim};
    cuLaunchKernel(ops->timestep_embed, (unsigned)((half + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Euler step: x = x - dt * v ---- */
static inline void op_euler_step(hy3d_ops *ops, CUstream stream,
                                 CUdeviceptr x, CUdeviceptr v,
                                 float dt, int n) {
    void *args[] = {&x, &v, &dt, &n};
    cuLaunchKernel(ops->euler_step, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- CFG combine: out = uncond + scale * (cond - uncond) ---- */
static inline void op_cfg_combine(hy3d_ops *ops, CUstream stream,
                                  CUdeviceptr out, CUdeviceptr cond,
                                  CUdeviceptr uncond,
                                  float scale, int n) {
    void *args[] = {&out, &cond, &uncond, &scale, &n};
    cuLaunchKernel(ops->cfg_combine, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Split interleaved QKV: [N, H, 3, HD] -> Q[N, W], K[N, W], V[N, W] ---- */
/* W = H * HD */
static inline void op_split_qkv(hy3d_ops *ops, CUstream stream,
                                CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                                CUdeviceptr qkv,
                                int N, int H, int HD) {
    int total = N * H * HD;
    void *args[] = {&Q, &K, &V, &qkv, &N, &H, &HD};
    cuLaunchKernel(ops->split_qkv, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Split interleaved KV: [M, H, 2, HD] -> K[M, W], V[M, W] ---- */
/* W = H * HD */
static inline void op_split_kv(hy3d_ops *ops, CUstream stream,
                               CUdeviceptr K, CUdeviceptr V,
                               CUdeviceptr kv,
                               int M, int H, int HD) {
    int total = M * H * HD;
    void *args[] = {&K, &V, &kv, &M, &H, &HD};
    cuLaunchKernel(ops->split_kv, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Broadcast add: dst[i] += src[i % src_len] ---- */
/* Adds a vector of length src_len to each row of dst [n_rows, src_len] */
static inline void op_broadcast_add(hy3d_ops *ops, CUstream stream,
                                    CUdeviceptr dst, CUdeviceptr src,
                                    int n, int src_len) {
    void *args[] = {&dst, &src, &n, &src_len};
    cuLaunchKernel(ops->broadcast_add, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Concat first: prepend a token row to a sequence ---- */
/* token: [1, dim], seq: [seq_len, dim] -> out: [seq_len+1, dim] */
/* out[0, :] = token, out[1:, :] = seq */
static inline void op_concat_first(hy3d_ops *ops, CUstream stream,
                                   CUdeviceptr out, CUdeviceptr token,
                                   CUdeviceptr seq,
                                   int seq_len, int dim) {
    int total = (seq_len + 1) * dim;
    void *args[] = {&out, &token, &seq, &seq_len, &dim};
    cuLaunchKernel(ops->concat_first, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Strip first: drop the first token from a sequence ---- */
/* src: [seq_len, dim] -> dst: [seq_len-1, dim] */
/* dst[i, :] = src[i+1, :] */
static inline void op_strip_first(hy3d_ops *ops, CUstream stream,
                                  CUdeviceptr dst, CUdeviceptr src,
                                  int seq_len, int dim) {
    int total = (seq_len - 1) * dim;
    void *args[] = {&dst, &src, &seq_len, &dim};
    cuLaunchKernel(ops->strip_first, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Concat last dim: concatenate two tensors along the last dimension ---- */
/* a: [N, dim], b: [N, dim] -> out: [N, 2*dim] */
static inline void op_concat_last_dim(hy3d_ops *ops, CUstream stream,
                                      CUdeviceptr out,
                                      CUdeviceptr a, CUdeviceptr b,
                                      int N, int dim) {
    int total = N * 2 * dim;
    void *args[] = {&out, &a, &b, &N, &dim};
    cuLaunchKernel(ops->concat_last_dim, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

#endif /* CUDA_HY3D_OPS_H */
