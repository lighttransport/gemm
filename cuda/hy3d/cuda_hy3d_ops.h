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

    /* ---- Tensor-core GEMM + flash attention (shared with qimg/flux2) ---- */
    /* Per-row FP8 MMA GEMMs (m16n8k32, sm_89+). Row-max scaling is computed
     * online from F32 X via reduce_max_abs_per_row. */
    CUfunction gemm_fp8_perrow;          /* MTILE=2 (32 rows / CTA) */
    CUfunction gemm_fp8_perrow_mt4;      /* MTILE=4 (64 rows / CTA), n_tok % 64 == 0 */
    CUfunction gemm_fp8_perrow_concat3;  /* fused QKV: 3 weight banks, shared row_max */
    /* BF16 X / FP8 W MMA pipeline (m16n8k16, sm_80+). Inline FP8->BF16 dequant. */
    CUfunction gemm_bf16_pipe;
    /* Same as above + per-tensor weight scale multiplied at writeback.
     * Used when FP8 weights store round(w/scale) (hy3d online prequant). */
    CUfunction gemm_bf16_pipe_scaled;
    /* FP8 quantizer + reductions */
    CUfunction reduce_max_abs;           /* per-tensor max|x| -> single F32 */
    CUfunction reduce_max_abs_per_row;   /* per-row max|x|   -> [n_rows] F32 */
    CUfunction quantize_fp8;             /* F32 -> e4m3 with scale writeback */
    CUfunction cast_f32_to_bf16;         /* F32 -> BF16 (uint16) cast */
    /* Flash attention */
    CUfunction flash_attn_bf16;          /* head_dim=128 BF16 acts */
    CUfunction flash_attn_bf16_xq;       /* head_dim=128 BF16 acts, ragged Q (q_len != kv_len) */
    CUfunction flash_attn_bf16_hd64;     /* head_dim=64 BF16 acts (DINOv2 / ShapeVAE) */
    CUfunction flash_attn_bf16_hd64_xq;  /* head_dim=64 BF16 acts, ragged Q */
    CUfunction flash_attn_fp8;           /* head_dim=128 FP8 acts (sm_89+) */
    /* MoE on-device kernels (added by hy3d) */
    CUfunction moe_topk_softmax;         /* gate logits -> top-k indices + weights */
    CUfunction moe_scaled_accumulate;    /* acc[t,h] += w[t] * out[t,h] when idx[t,*] == eid */
    /* F16 -> F32 buffer cast (used for online FP8 weight quant at load) */
    CUfunction f16_to_f32_buf;

    int sm_version;
    int use_f32_gemm;            /* 0=F16 weights (default), 1=F32 weights */
    /* Tensor-core dispatch flags. Defaults set by cuda_hy3d_create() from env. */
    int use_bf16_attn;           /* HY3D_BF16_ATTN, default ON if sm>=80 */
    int use_fp8_attn;            /* HY3D_FP8_ATTN, default OFF (head_dim=128 only) */
    int use_fp8_gemm;            /* HY3D_FP8_GEMM, default ON if sm>=89 — MoE only */
    int use_fp8_gemm_attn_mlp;   /* HY3D_FP8_GEMM_ATTN_MLP, default OFF — extends FP8 to DiT attention QKV/out + MLP fc1/fc2; quality regression at all step counts (verified 30s) */
    int use_bf16_gemm;           /* HY3D_BF16_GEMM, default OFF (precision fallback) */
    int disable_mt4;             /* HY3D_DISABLE_MT4, default OFF */

    /* Lazy scratch buffers for tensor-core attention. Owned by the ops struct,
     * grown on first use, freed by hy3d_ops_destroy(). One contiguous BF16
     * buffer holds Q,K,V back-to-back (each n_tok*dim halfs). */
    CUdeviceptr d_qkv_bf16;
    size_t      qkv_bf16_n;      /* total bytes allocated */
    CUdeviceptr d_qkv_fp8;
    size_t      qkv_fp8_n;
    CUdeviceptr d_qkv_scales;    /* 3 floats: sQ, sK, sV (FP8 attn) */
    /* Per-row max(|X|) buffer for FP8 perrow GEMM. Grown on first use. */
    CUdeviceptr d_row_max;
    size_t      row_max_n;
    /* Per-tensor weight-scale cache: each prequantized weight has a 1-float
     * scale on device; the FP8 dispatcher used to DtoH it on every GEMM call
     * (~10k syncs/mesh). Cache it host-side at first use. */
    CUdeviceptr scale_cache_d[512];
    float       scale_cache_h[512];
    int         scale_cache_n;
} hy3d_ops;

/* Free scratch buffers attached to the ops context. */
static inline void hy3d_ops_destroy(hy3d_ops *ops) {
    if (ops->d_qkv_bf16)   { cuMemFree(ops->d_qkv_bf16);   ops->d_qkv_bf16 = 0;   ops->qkv_bf16_n = 0; }
    if (ops->d_qkv_fp8)    { cuMemFree(ops->d_qkv_fp8);    ops->d_qkv_fp8 = 0;    ops->qkv_fp8_n = 0; }
    if (ops->d_qkv_scales) { cuMemFree(ops->d_qkv_scales); ops->d_qkv_scales = 0; }
    if (ops->d_row_max)    { cuMemFree(ops->d_row_max);    ops->d_row_max = 0;    ops->row_max_n = 0; }
}

/* Ensure ops->d_row_max is at least n_tok floats. */
static inline int hy3d_ops_ensure_row_max(hy3d_ops *ops, int n_tok) {
    size_t need = (size_t)n_tok * sizeof(float);
    if (ops->row_max_n >= need) return 0;
    if (ops->d_row_max) { cuMemFree(ops->d_row_max); ops->d_row_max = 0; ops->row_max_n = 0; }
    if (cuMemAlloc(&ops->d_row_max, need) != CUDA_SUCCESS) {
        ops->d_row_max = 0; ops->row_max_n = 0;
        return -1;
    }
    ops->row_max_n = need;
    return 0;
}

/* Look up the host-side float for a per-tensor scale device pointer, doing a
 * one-time DtoH on cache miss. The set of scale pointers is fixed at load
 * (252 DiT GEMMs + 96 MoE buffers); 512 entries is safely above. */
static inline float hy3d_ops_scale_get(hy3d_ops *ops, CUdeviceptr d_scale) {
    for (int i = 0; i < ops->scale_cache_n; i++) {
        if (ops->scale_cache_d[i] == d_scale) return ops->scale_cache_h[i];
    }
    float h = 1.0f;
    cuMemcpyDtoH(&h, d_scale, sizeof(float));
    if (ops->scale_cache_n < (int)(sizeof(ops->scale_cache_d) /
                                   sizeof(ops->scale_cache_d[0]))) {
        int i = ops->scale_cache_n++;
        ops->scale_cache_d[i] = d_scale;
        ops->scale_cache_h[i] = h;
    }
    return h;
}

/* Ensure ops->d_qkv_bf16 is at least `bytes` large. Returns 0 on success, -1 on OOM. */
static inline int hy3d_ops_ensure_qkv_bf16(hy3d_ops *ops, size_t bytes) {
    if (bytes <= ops->qkv_bf16_n) return 0;
    if (ops->d_qkv_bf16) cuMemFree(ops->d_qkv_bf16);
    ops->d_qkv_bf16 = 0; ops->qkv_bf16_n = 0;
    if (cuMemAlloc(&ops->d_qkv_bf16, bytes) != CUDA_SUCCESS) return -1;
    ops->qkv_bf16_n = bytes;
    return 0;
}

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

    /* ---- Optional tensor-core kernels: failure to find is non-fatal ---- */
    /* All of these come from cuda/cuda_fp8_mma_kernels.h (shared with qimg/flux2)
     * plus moe_* added by hy3d-specific kernels. If a symbol is missing the
     * runtime falls back to the existing scalar/tiled path. */
    #define GET_OPT(name, field) \
        if (cuModuleGetFunction(&ops->field, module, name) != CUDA_SUCCESS) \
            ops->field = NULL;

    GET_OPT("gemm_fp8_pipe_perrow_f32",              gemm_fp8_perrow);
    GET_OPT("gemm_fp8_pipe_perrow_mt4_f32",          gemm_fp8_perrow_mt4);
    GET_OPT("gemm_fp8_pipe_perrow_mt4_concat3_f32",  gemm_fp8_perrow_concat3);
    GET_OPT("gemm_bf16_pipe_f32",                    gemm_bf16_pipe);
    GET_OPT("gemm_bf16_pipe_scaled_f32",             gemm_bf16_pipe_scaled);
    GET_OPT("reduce_max_abs_f32",                    reduce_max_abs);
    GET_OPT("reduce_max_abs_per_row_f32",            reduce_max_abs_per_row);
    GET_OPT("quantize_to_fp8_e4m3",                  quantize_fp8);
    GET_OPT("cast_f32_to_bf16",                      cast_f32_to_bf16);
    GET_OPT("flash_attn_bf16",                       flash_attn_bf16);
    GET_OPT("flash_attn_bf16_xq",                    flash_attn_bf16_xq);
    GET_OPT("flash_attn_bf16_hd64",                  flash_attn_bf16_hd64);
    GET_OPT("flash_attn_bf16_hd64_xq",               flash_attn_bf16_hd64_xq);
    GET_OPT("flash_attn_fp8",                        flash_attn_fp8);
    GET_OPT("moe_topk_softmax_f32",                  moe_topk_softmax);
    GET_OPT("moe_scaled_accumulate_f32",             moe_scaled_accumulate);
    GET_OPT("f16_to_f32_buf",                        f16_to_f32_buf);

    #undef GET_OPT
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

/* GEMM with optional FP8 weight bundle. Dispatches:
 *   1. F32 escape hatch (use_f32_gemm) — calls op_gemm.
 *   2. Per-row FP8 MMA when:
 *        use_fp8_gemm AND w_fp8 != 0 AND w_scale != 0 AND
 *        n_in % 32 == 0 AND n_out % 64 == 0 AND n_tok >= 16
 *      Uses MTILE=4 variant when n_tok % 64 == 0 and !disable_mt4.
 *   3. Otherwise: falls back to op_gemm (existing F16 tiled path).
 *
 * Caller passes both the F16 weight pointer (for fallback) and the FP8/scale
 * mirror pair (NULL/0 if not pre-quantized). */
static inline void op_gemm_qw(hy3d_ops *ops, CUstream stream,
                              CUdeviceptr Y,
                              CUdeviceptr w_f16,
                              CUdeviceptr w_fp8, CUdeviceptr w_scale,
                              CUdeviceptr X, CUdeviceptr bias,
                              int n_out, int n_in, int n_tok) {
    if (ops->use_f32_gemm || !ops->use_fp8_gemm || !w_fp8 || !w_scale ||
        !ops->gemm_fp8_perrow || !ops->reduce_max_abs_per_row ||
        n_tok < 16 || (n_in % 32) != 0 || (n_out % 64) != 0) {
        op_gemm(ops, stream, Y, w_f16, X, bias, n_out, n_in, n_tok);
        return;
    }
    if (hy3d_ops_ensure_row_max(ops, n_tok) != 0) {
        op_gemm(ops, stream, Y, w_f16, X, bias, n_out, n_in, n_tok);
        return;
    }
    /* Per-row reduce: one CTA per row, 256 threads. */
    {
        void *rargs[] = {&ops->d_row_max, &X, &n_tok, &n_in};
        cuLaunchKernel(ops->reduce_max_abs_per_row, (unsigned)n_tok, 1, 1,
                       256, 1, 1, 0, stream, rargs, NULL);
    }
    /* w_scale is a 1-float device pointer; kernel takes float-by-value and
     * qimg passes 1.0f (pre-baked weights). We prequantized at load time and
     * stored scale device-side; cache the host value to avoid sync DtoH
     * per call (5760 syncs/mesh on MoE alone). */
    float w_scale_h = hy3d_ops_scale_get(ops, w_scale);
    void *args[] = {&Y, &w_fp8, &X, &bias, &n_out, &n_in, &n_tok,
                    &w_scale_h, &ops->d_row_max};
    unsigned gx = (unsigned)((n_out + 255) / 256);
    if (!ops->disable_mt4 && ops->gemm_fp8_perrow_mt4 &&
        n_tok >= 64 && (n_tok % 64) == 0) {
        unsigned gy4 = (unsigned)((n_tok + 63) / 64);
        unsigned gx4 = (gx + 3u) & ~3u;
        gy4 = (gy4 + 3u) & ~3u;
        size_t smem_mt4 = 2048 + 8192 * 2 + 512;
        cuLaunchKernel(ops->gemm_fp8_perrow_mt4, gx4, gy4, 1, 128, 1, 1,
                       smem_mt4, stream, args, NULL);
        return;
    }
    unsigned gy = (unsigned)((n_tok + 31) / 32);
    unsigned gx_pr = (gx + 3u) & ~3u;
    gy = (gy + 3u) & ~3u;
    size_t smem_pr = 1024 + 8192 * 2 + 256;
    cuLaunchKernel(ops->gemm_fp8_perrow, gx_pr, gy, 1, 128, 1, 1,
                   smem_pr, stream, args, NULL);
}

/* DiT attention/MLP scope dispatcher.
 *   Priority:
 *     1. F32 escape (use_f32_gemm)                          -> op_gemm
 *     2. BF16 X * FP8 W pipe (use_bf16_gemm)                -> gemm_bf16_pipe_scaled
 *        (BF16 accumulate avoids per-tensor weight scale drift seen in pure FP8)
 *     3. Per-row FP8 MMA (use_fp8_gemm_attn_mlp, opt-in)    -> op_gemm_qw
 *     4. F16 tiled fallback                                  -> op_gemm
 *
 * MoE call sites use op_gemm_qw directly and stay on FP8 (where the trajectory
 * is robust). The gating differs only here. */
static inline void op_gemm_qw_dit(hy3d_ops *ops, CUstream stream,
                                  CUdeviceptr Y,
                                  CUdeviceptr w_f16,
                                  CUdeviceptr w_fp8, CUdeviceptr w_scale,
                                  CUdeviceptr X, CUdeviceptr bias,
                                  int n_out, int n_in, int n_tok) {
    if (!ops->use_f32_gemm && ops->use_bf16_gemm && w_fp8 && w_scale &&
        ops->gemm_bf16_pipe_scaled &&
        n_tok >= 16 && (n_in % 32) == 0 && (n_out % 256) == 0) {
        void *args[] = {&Y, &w_fp8, &X, &bias, &n_out, &n_in, &n_tok, &w_scale};
        unsigned gx = (unsigned)((n_out + 255) / 256);
        unsigned gy = (unsigned)((n_tok +  31) /  32);
        gx = (gx + 3u) & ~3u;
        gy = (gy + 3u) & ~3u;
        size_t smem_bf16 = 2048 + 8192 * 2;
        cuLaunchKernel(ops->gemm_bf16_pipe_scaled, gx, gy, 1, 128, 1, 1,
                       smem_bf16, stream, args, NULL);
        return;
    }
    if (!ops->use_fp8_gemm_attn_mlp) { w_fp8 = 0; w_scale = 0; }
    op_gemm_qw(ops, stream, Y, w_f16, w_fp8, w_scale, X, bias,
               n_out, n_in, n_tok);
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
static inline void op_cast_f32_to_bf16(hy3d_ops *ops, CUstream stream,
                                       CUdeviceptr dst_u16, CUdeviceptr src_f32,
                                       int n);

static inline void op_cross_attn(hy3d_ops *ops, CUstream stream,
                                 CUdeviceptr out,
                                 CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                                 int q_len, int kv_len, int dim,
                                 int n_heads, int head_dim) {
    /* BF16 ragged-Q flash fast path: head_dim==128 or head_dim==64, sm>=80. */
    CUfunction xq_fn = 0;
    size_t xq_smem = 0;
    if (ops->use_bf16_attn && ops->cast_f32_to_bf16 && q_len > 0 && kv_len > 0) {
        if (head_dim == 128 && ops->flash_attn_bf16_xq) {
            xq_fn = ops->flash_attn_bf16_xq;
            xq_smem = (size_t)(4 * 32 * 136 * 2);
        } else if (head_dim == 64 && ops->flash_attn_bf16_hd64_xq) {
            xq_fn = ops->flash_attn_bf16_hd64_xq;
            xq_smem = (size_t)(4 * 32 * 72 * 2);
        }
    }
    if (xq_fn) {
        int n_q_elem  = q_len  * dim;
        int n_kv_elem = kv_len * dim;
        size_t need = (size_t)(n_q_elem + 2 * n_kv_elem) * sizeof(unsigned short);
        if (hy3d_ops_ensure_qkv_bf16(ops, need) == 0) {
            CUdeviceptr d_q = ops->d_qkv_bf16;
            CUdeviceptr d_k = ops->d_qkv_bf16 + (CUdeviceptr)(n_q_elem * sizeof(unsigned short));
            CUdeviceptr d_v = ops->d_qkv_bf16 + (CUdeviceptr)((n_q_elem + n_kv_elem) * sizeof(unsigned short));
            op_cast_f32_to_bf16(ops, stream, d_q, Q, n_q_elem);
            op_cast_f32_to_bf16(ops, stream, d_k, K, n_kv_elem);
            op_cast_f32_to_bf16(ops, stream, d_v, V, n_kv_elem);
            unsigned gy = (unsigned)((q_len + 63) / 64);
            void *args[] = {&out, &d_q, &d_k, &d_v, &q_len, &kv_len, &n_heads, &head_dim};
            cuLaunchKernel(xq_fn,
                           (unsigned)n_heads, gy, 1,
                           128, 1, 1, xq_smem, stream, args, NULL);
            return;
        }
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    int nt = 128;  /* threads per block */
    size_t smem = (size_t)(kv_len + nt) * sizeof(float);
    void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim,
                    &n_heads, &head_dim, &scale};
    cuLaunchKernel(ops->cross_attn, (unsigned)n_heads, (unsigned)q_len, 1,
                   (unsigned)nt, 1, 1, smem, stream, args, NULL);
}

/* Internal: launch cast_f32_to_bf16 over n contiguous F32 elements. */
static inline void op_cast_f32_to_bf16(hy3d_ops *ops, CUstream stream,
                                       CUdeviceptr dst_u16, CUdeviceptr src_f32,
                                       int n) {
    void *args[] = {&src_f32, &dst_u16, &n};
    cuLaunchKernel(ops->cast_f32_to_bf16,
                   (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Self-attention: BF16 flash attn fast path, F32 fallback ---- */
/* Only the BF16 flash kernel from cuda_fp8_mma_kernels.h is wired here today.
 * It is head_dim=128 only, so DINOv2-L/ShapeVAE (head_dim=64) automatically
 * fall through to the existing cross_attn_f32 kernel. */
static inline void op_self_attn(hy3d_ops *ops, CUstream stream,
                                CUdeviceptr out,
                                CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                                int n_tok, int dim,
                                int n_heads, int head_dim) {
    /* BF16 fast path: head_dim==128 or head_dim==64, sm>=80. */
    CUfunction sa_fn = 0;
    size_t sa_smem = 0;
    if (ops->use_bf16_attn && ops->cast_f32_to_bf16 && n_tok > 0) {
        if (head_dim == 128 && ops->flash_attn_bf16) {
            sa_fn = ops->flash_attn_bf16;
            sa_smem = (size_t)(4 * 32 * 136 * 2);
        } else if (head_dim == 64 && ops->flash_attn_bf16_hd64) {
            sa_fn = ops->flash_attn_bf16_hd64;
            sa_smem = (size_t)(4 * 32 * 72 * 2);
        }
    }
    if (sa_fn) {
        int n_elem = n_tok * dim;
        size_t need = (size_t)3 * n_elem * sizeof(unsigned short);
        if (hy3d_ops_ensure_qkv_bf16(ops, need) == 0) {
            CUdeviceptr d_q = ops->d_qkv_bf16;
            CUdeviceptr d_k = ops->d_qkv_bf16 + (CUdeviceptr)(n_elem * sizeof(unsigned short));
            CUdeviceptr d_v = ops->d_qkv_bf16 + (CUdeviceptr)(2 * n_elem * sizeof(unsigned short));
            op_cast_f32_to_bf16(ops, stream, d_q, Q, n_elem);
            op_cast_f32_to_bf16(ops, stream, d_k, K, n_elem);
            op_cast_f32_to_bf16(ops, stream, d_v, V, n_elem);
            unsigned gy = (unsigned)((n_tok + 63) / 64);
            void *args[] = {&out, &d_q, &d_k, &d_v, &n_tok, &n_heads, &head_dim};
            cuLaunchKernel(sa_fn,
                           (unsigned)n_heads, gy, 1,
                           128, 1, 1, sa_smem, stream, args, NULL);
            return;
        }
        /* OOM -> fall through */
    }

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
