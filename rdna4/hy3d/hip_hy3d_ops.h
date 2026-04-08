/*
 * hip_hy3d_ops.h - Modular kernel-launch wrappers for Hunyuan3D-2.1 HIP runner
 *
 * All ML operations as static inline functions operating on hy3d_ops context.
 * Each op takes (hy3d_ops *ops, hipStream_t stream, ...) and launches the
 * appropriate kernel.  The runner struct holds a single `hy3d_ops ops;` field.
 *
 * Port of cuda_hy3d_ops.h for AMD ROCm/HIP (RDNA4, gfx1201).
 * Key differences:
 *   - CUfunction -> hipFunction_t
 *   - CUstream -> hipStream_t
 *   - CUdeviceptr -> void *
 *   - cuLaunchKernel -> hipModuleLaunchKernel
 *   - cuModuleGetFunction -> hipModuleGetFunction
 *   - No gemm_f16_f32 (MMA) -- RDNA4 has no MMA
 *   - attn_prefill maps to flash_attn_tiled_f32
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#ifndef HIP_HY3D_OPS_H
#define HIP_HY3D_OPS_H

#include <math.h>
#include <stdio.h>
#include <string.h>

/* This header requires HIP types via rocew.h.
   Include rocew.h BEFORE including this header. */
#ifndef ROCEW_H_
#error "hip_hy3d_ops.h requires rocew.h to be included first"
#endif

/* ======================================================================== */
/* Ops context: holds all compiled kernel function pointers                 */
/* ======================================================================== */

typedef struct {
    /* Common kernels (from hip_kernels_common.h) */
    hipFunction_t layernorm;
    hipFunction_t gemm;              /* unused on RDNA4 (no MMA) */
    hipFunction_t gemm_tiled;
    hipFunction_t gelu;
    hipFunction_t add;
    hipFunction_t silu;
    hipFunction_t resize_normalize;
    hipFunction_t patch_embed;
    hipFunction_t cls_pos_embed;
    hipFunction_t attn_prefill;
    hipFunction_t flash_attn_tiled;
    hipFunction_t kv_transpose;
    hipFunction_t bilinear_upsample;

    /* HY3D-specific kernels */
    hipFunction_t rms_norm;
    hipFunction_t qk_layernorm;
    hipFunction_t layerscale_add;
    hipFunction_t cross_attn;
    hipFunction_t fourier_embed;
    hipFunction_t timestep_embed;
    hipFunction_t euler_step;
    hipFunction_t cfg_combine;
    hipFunction_t split_qkv;
    hipFunction_t split_kv;
    hipFunction_t broadcast_add;
    hipFunction_t concat_first;     /* prepend token */
    hipFunction_t strip_first;      /* drop first token */
    hipFunction_t concat_last_dim;  /* concat along last dim */
    hipFunction_t gemm_f32;         /* pure F32 GEMM (no F16 conversion) */

    int sm_version;
    int use_f32_gemm;            /* 0=F16 weights (default), 1=F32 weights */
} hy3d_ops;

/* ======================================================================== */
/* Load all kernel functions from a compiled module                         */
/* ======================================================================== */

static int hy3d_ops_load(hy3d_ops *ops, hipModule_t module, int sm_version) {
    memset(ops, 0, sizeof(*ops));
    ops->sm_version = sm_version;

    #define GET_FN(name, field) \
        if (hipModuleGetFunction(&ops->field, module, name) != hipSuccess) { \
            fprintf(stderr, "HY3D ops: failed to get kernel '%s'\n", name); \
            return -1; \
        }

    /* Common kernels */
    GET_FN("layernorm_f32",          layernorm);
    /* No gemm_f16_f32 (MMA) on RDNA4 -- skip ops->gemm */
    GET_FN("gemm_tiled_f16_f32",     gemm_tiled);
    GET_FN("gelu_f32",               gelu);
    GET_FN("add_f32",                add);
    GET_FN("silu_f32",               silu);
    GET_FN("resize_normalize",       resize_normalize);
    GET_FN("patch_embed_conv2d",     patch_embed);
    GET_FN("cls_pos_embed",          cls_pos_embed);
    GET_FN("kv_transpose",           kv_transpose);

    /* RDNA4 has no MMA -- always use flash_attn_tiled_f32 for both */
    GET_FN("flash_attn_tiled_f32",   attn_prefill);
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
static inline void op_layernorm(hy3d_ops *ops, hipStream_t stream,
                                void *dst, void *src,
                                void *w, void *b,
                                int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    hipModuleLaunchKernel(ops->layernorm, (unsigned)n_tok, 1, 1,
                   256, 1, 1, 256 * sizeof(float), stream, args, NULL);
}

/* ---- GEMM: Y = W @ X + bias ---- */
/* When use_f32_gemm=0: W is F16 (half_raw), bias/X/Y are F32.
 * When use_f32_gemm=1: all F32.
 * Both use same grid/block config: (ceil(n_out/64), ceil(n_tok/16)), blockDim=(16,16) */
static inline void op_gemm(hy3d_ops *ops, hipStream_t stream,
                           void *Y, void *W,
                           void *X, void *bias,
                           int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipFunction_t fn = ops->use_f32_gemm ? ops->gemm_f32 : ops->gemm_tiled;
    hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, stream, args, NULL);
}

/* ---- GELU activation in-place ---- */
static inline void op_gelu(hy3d_ops *ops, hipStream_t stream,
                           void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(ops->gelu, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- SiLU activation in-place ---- */
static inline void op_silu(hy3d_ops *ops, hipStream_t stream,
                           void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(ops->silu, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Element-wise add: dst += src ---- */
static inline void op_add(hy3d_ops *ops, hipStream_t stream,
                          void *dst, void *src, int n) {
    void *args[] = {&dst, &src, &n};
    hipModuleLaunchKernel(ops->add, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- LayerScale residual: dst[i] += src[i] * scale[i % dim] ---- */
static inline void op_layerscale_add(hy3d_ops *ops, hipStream_t stream,
                                     void *dst, void *src,
                                     void *scale, int n, int dim) {
    void *args[] = {&dst, &src, &scale, &n, &dim};
    hipModuleLaunchKernel(ops->layerscale_add, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- RMSNorm: per-head RMSNorm (for DiT QK normalization) ---- */
/* data: [n_tok, stride], w: [head_dim] */
static inline void op_rms_norm(hy3d_ops *ops, hipStream_t stream,
                               void *data, void *w,
                               int n_tok, int n_heads, int head_dim,
                               int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data, &w, &n_tok, &n_heads, &head_dim, &stride, &eps};
    hipModuleLaunchKernel(ops->rms_norm, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- QK LayerNorm: per-head LayerNorm (for ShapeVAE QK normalization) ---- */
/* data: [n_tok, stride], w: [head_dim], b: [head_dim] (b may be 0/NULL) */
static inline void op_qk_layernorm(hy3d_ops *ops, hipStream_t stream,
                                   void *data,
                                   void *w, void *b,
                                   int n_tok, int n_heads, int head_dim,
                                   int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data, &w, &b, &n_tok, &n_heads, &head_dim, &stride, &eps};
    hipModuleLaunchKernel(ops->qk_layernorm, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Cross-attention: Q/K/V may have different sequence lengths ---- */
/* Q: [q_len, dim], K: [kv_len, dim], V: [kv_len, dim], out: [q_len, dim] */
/* dim = n_heads * head_dim */
static inline void op_cross_attn(hy3d_ops *ops, hipStream_t stream,
                                 void *out,
                                 void *Q, void *K, void *V,
                                 int q_len, int kv_len, int dim,
                                 int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int nt = 128;  /* threads per block */
    size_t smem = (size_t)(kv_len + nt) * sizeof(float);
    void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim,
                    &n_heads, &head_dim, &scale};
    hipModuleLaunchKernel(ops->cross_attn, (unsigned)n_heads, (unsigned)q_len, 1,
                   (unsigned)nt, 1, 1, smem, stream, args, NULL);
}

/* ---- Self-attention: wrapper over cross_attn with q_len == kv_len ---- */
static inline void op_self_attn(hy3d_ops *ops, hipStream_t stream,
                                void *out,
                                void *Q, void *K, void *V,
                                int n_tok, int dim,
                                int n_heads, int head_dim) {
    op_cross_attn(ops, stream, out, Q, K, V,
                  n_tok, n_tok, dim, n_heads, head_dim);
}

/* ---- 3D Fourier coordinate embedding ---- */
/* coords: [N, 3], freqs: [num_freqs], out: [N, out_dim] */
/* out_dim = 3 * (2 * num_freqs + 1) */
static inline void op_fourier_embed(hy3d_ops *ops, hipStream_t stream,
                                    void *out, void *coords,
                                    void *freqs,
                                    int N, int num_freqs, int out_dim) {
    void *args[] = {&out, &coords, &freqs, &N, &num_freqs, &out_dim};
    hipModuleLaunchKernel(ops->fourier_embed, (unsigned)((N + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Sinusoidal timestep embedding ---- */
/* out: [dim], t: scalar timestep value */
static inline void op_timestep_embed(hy3d_ops *ops, hipStream_t stream,
                                     void *out, float t, int dim) {
    int half = dim / 2;
    void *args[] = {&out, &t, &dim};
    hipModuleLaunchKernel(ops->timestep_embed, (unsigned)((half + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Euler step: x = x - dt * v ---- */
static inline void op_euler_step(hy3d_ops *ops, hipStream_t stream,
                                 void *x, void *v,
                                 float dt, int n) {
    void *args[] = {&x, &v, &dt, &n};
    hipModuleLaunchKernel(ops->euler_step, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- CFG combine: out = uncond + scale * (cond - uncond) ---- */
static inline void op_cfg_combine(hy3d_ops *ops, hipStream_t stream,
                                  void *out, void *cond,
                                  void *uncond,
                                  float scale, int n) {
    void *args[] = {&out, &cond, &uncond, &scale, &n};
    hipModuleLaunchKernel(ops->cfg_combine, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Split interleaved QKV: [N, H, 3, HD] -> Q[N, W], K[N, W], V[N, W] ---- */
/* W = H * HD */
static inline void op_split_qkv(hy3d_ops *ops, hipStream_t stream,
                                void *Q, void *K, void *V,
                                void *qkv,
                                int N, int H, int HD) {
    int total = N * H * HD;
    void *args[] = {&Q, &K, &V, &qkv, &N, &H, &HD};
    hipModuleLaunchKernel(ops->split_qkv, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Split interleaved KV: [M, H, 2, HD] -> K[M, W], V[M, W] ---- */
/* W = H * HD */
static inline void op_split_kv(hy3d_ops *ops, hipStream_t stream,
                               void *K, void *V,
                               void *kv,
                               int M, int H, int HD) {
    int total = M * H * HD;
    void *args[] = {&K, &V, &kv, &M, &H, &HD};
    hipModuleLaunchKernel(ops->split_kv, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Broadcast add: dst[i] += src[i % src_len] ---- */
/* Adds a vector of length src_len to each row of dst [n_rows, src_len] */
static inline void op_broadcast_add(hy3d_ops *ops, hipStream_t stream,
                                    void *dst, void *src,
                                    int n, int src_len) {
    void *args[] = {&dst, &src, &n, &src_len};
    hipModuleLaunchKernel(ops->broadcast_add, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Concat first: prepend a token row to a sequence ---- */
/* token: [1, dim], seq: [seq_len, dim] -> out: [seq_len+1, dim] */
/* out[0, :] = token, out[1:, :] = seq */
static inline void op_concat_first(hy3d_ops *ops, hipStream_t stream,
                                   void *out, void *token,
                                   void *seq,
                                   int seq_len, int dim) {
    int total = (seq_len + 1) * dim;
    void *args[] = {&out, &token, &seq, &seq_len, &dim};
    hipModuleLaunchKernel(ops->concat_first, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Strip first: drop the first token from a sequence ---- */
/* src: [seq_len, dim] -> dst: [seq_len-1, dim] */
/* dst[i, :] = src[i+1, :] */
static inline void op_strip_first(hy3d_ops *ops, hipStream_t stream,
                                  void *dst, void *src,
                                  int seq_len, int dim) {
    int total = (seq_len - 1) * dim;
    void *args[] = {&dst, &src, &seq_len, &dim};
    hipModuleLaunchKernel(ops->strip_first, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Concat last dim: concatenate two tensors along the last dimension ---- */
/* a: [N, dim], b: [N, dim] -> out: [N, 2*dim] */
static inline void op_concat_last_dim(hy3d_ops *ops, hipStream_t stream,
                                      void *out,
                                      void *a, void *b,
                                      int N, int dim) {
    int total = N * 2 * dim;
    void *args[] = {&out, &a, &b, &N, &dim};
    hipModuleLaunchKernel(ops->concat_last_dim, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

#endif /* HIP_HY3D_OPS_H */
