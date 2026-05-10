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

#ifdef HY3D_HIPBLASLT_ENABLED
#include "mm_blaslt_bridge.h"
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
    hipFunction_t gelu_exact;
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
    hipFunction_t qk_layernorm_f16;
    hipFunction_t layerscale_add;
    hipFunction_t layerscale_add_f16;
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
    hipFunction_t gemm_wmma;        /* F16w + BF16a WMMA (gfx12 only; NULL on older) */

    /* F16 residual-stream kernels (HY3D_DIT_FP16 path) */
    hipFunction_t layernorm_f16;
    hipFunction_t rms_norm_f16;
    hipFunction_t gelu_exact_f16;
    hipFunction_t add_f16;
    hipFunction_t cast_f32_to_f16;
    hipFunction_t cast_f16_to_f32;
    hipFunction_t flash_attn_sa_f16_wmma;   /* gfx12 only; NULL on older */
    hipFunction_t cross_attn_f16_wmma;      /* gfx12 only; NULL on older */
    hipFunction_t flash_attn_sa_f16_wmma_hd64; /* hd=64 variant for ShapeVAE */
    hipFunction_t cross_attn_f16_wmma_hd64;

    /* GPU-resident MoE gating + dispatch */
    hipFunction_t moe_gate_softmax_topk;
    hipFunction_t moe_weighted_add;
    hipFunction_t moe_dispatch_build;
    hipFunction_t moe_gather_f16;
    hipFunction_t moe_scatter_add_f16_to_f32;

    /* F16 layout helpers (split / concat / strip) */
    hipFunction_t split_qkv_f16;
    hipFunction_t split_kv_f16;
    hipFunction_t concat_first_f16;
    hipFunction_t strip_first_f16;
    hipFunction_t concat_last_dim_f16;

    int sm_version;
    int use_f32_gemm;            /* 0=F16 weights (default), 1=F32 weights */
    int use_wmma;                /* 1 if gemm_wmma is available (gfx1200/1201) */
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
    GET_FN("gelu_exact_f32",         gelu_exact);
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
    if (hipModuleGetFunction(&ops->qk_layernorm_f16, module,
                             "qk_layernorm_f16") != hipSuccess) {
        ops->qk_layernorm_f16 = NULL;
    }
    GET_FN("layerscale_add_f32",          layerscale_add);
    if (hipModuleGetFunction(&ops->layerscale_add_f16, module,
                             "layerscale_add_f16") != hipSuccess) {
        ops->layerscale_add_f16 = NULL;
    }
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

    /* F16 residual-stream kernels (used when HY3D_DIT_FP16 is enabled) */
    GET_FN("layernorm_f16",               layernorm_f16);
    GET_FN("rms_norm_f16",                rms_norm_f16);
    GET_FN("gelu_exact_f16",              gelu_exact_f16);
    GET_FN("add_f16",                     add_f16);
    GET_FN("cast_f32_to_f16",             cast_f32_to_f16);
    GET_FN("cast_f16_to_f32",             cast_f16_to_f32);

    /* GPU MoE kernels */
    GET_FN("moe_gate_softmax_topk_f32",   moe_gate_softmax_topk);
    GET_FN("moe_weighted_add_f32",        moe_weighted_add);
    GET_FN("moe_dispatch_build_f32",      moe_dispatch_build);
    GET_FN("moe_gather_f16",              moe_gather_f16);
    GET_FN("moe_scatter_weighted_add_f16_to_f32", moe_scatter_add_f16_to_f32);

    GET_FN("split_qkv_interleaved_f16",   split_qkv_f16);
    GET_FN("split_kv_interleaved_f16",    split_kv_f16);
    GET_FN("concat_token_f16",            concat_first_f16);
    GET_FN("strip_first_token_f16",       strip_first_f16);
    GET_FN("concat_last_dim_f16",         concat_last_dim_f16);

    /* Optional: WMMA kernel is only valid on gfx1200/1201. hipModuleGetFunction
     * returns failure on older archs when the compiled PTX lacked the symbol.
     * Treat failure here as "WMMA not available" and fall back to tiled GEMM.
     * Default is OFF — set env HIP_HY3D_WMMA=1 to enable (A/B testing). */
    /* Optional fp16 WMMA flash-attention (gfx12 only).  Failure to find these
     * symbols means the GPU lacks WMMA support; the f16 DiT path will fall
     * back to op_self_attn / op_cross_attn (f32) until a non-WMMA fp16 FA
     * is available. */
    if (hipModuleGetFunction(&ops->flash_attn_sa_f16_wmma, module,
                             "flash_attn_sa_f16_wmma") != hipSuccess) {
        ops->flash_attn_sa_f16_wmma = NULL;
    }
    if (hipModuleGetFunction(&ops->cross_attn_f16_wmma, module,
                             "cross_attn_f16_wmma") != hipSuccess) {
        ops->cross_attn_f16_wmma = NULL;
    }
    if (hipModuleGetFunction(&ops->flash_attn_sa_f16_wmma_hd64, module,
                             "flash_attn_sa_f16_wmma_hd64") != hipSuccess) {
        ops->flash_attn_sa_f16_wmma_hd64 = NULL;
    }
    if (hipModuleGetFunction(&ops->cross_attn_f16_wmma_hd64, module,
                             "cross_attn_f16_wmma_hd64") != hipSuccess) {
        ops->cross_attn_f16_wmma_hd64 = NULL;
    }

    if (hipModuleGetFunction(&ops->gemm_wmma, module, "gemm_f16w_bf16a_wmma_t") == hipSuccess) {
        const char *wenv = getenv("HIP_HY3D_WMMA");
        ops->use_wmma = (wenv && *wenv && wenv[0] != '0') ? 1 : 0;
    } else {
        ops->gemm_wmma = NULL;
        ops->use_wmma = 0;
    }

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
    /* WMMA path: F16 weights only, n_in must be multiple of 16. Tile is 128x128
     * with in-kernel bounds checks, so ragged M/N shapes are fine. Skip tiny
     * problems where the scalar tile is already cheap. */
    if (!ops->use_f32_gemm && ops->use_wmma && ops->gemm_wmma &&
        (n_in & 15) == 0 && n_out >= 64 && n_tok >= 16) {
        unsigned gx = (unsigned)((n_out + 127) / 128);
        unsigned gy = (unsigned)((n_tok + 127) / 128);
        hipModuleLaunchKernel(ops->gemm_wmma, gx, gy, 1, 256, 1, 1, 0, stream, args, NULL);
        return;
    }
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipFunction_t fn = ops->use_f32_gemm ? ops->gemm_f32 : ops->gemm_tiled;
    hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, stream, args, NULL);
}

#ifdef HY3D_HIPBLASLT_ENABLED
/* ---- F16 GEMM via hipBLASLt (Y = X @ W^T, F32 accumulator) ----
 * W: F16 [n_out, n_in], X: F16 [n_tok, n_in], Y: F32 [n_tok, n_out].
 * Bridge convention: M=n_tok (rows of Y), N=n_out (cols of Y), K=n_in. */
static inline void op_gemm_f16(hy3d_ops *ops, hipStream_t stream,
                               void *Y_f32, void *W_f16, void *X_f16,
                               int n_out, int n_in, int n_tok) {
    (void)ops;
    mm_blaslt_run_f16(Y_f32, W_f16, X_f16, n_tok, n_out, n_in, (void *)stream);
}

/* Y[F32] = X*W^T + bias.  bias may be NULL. */
static inline void op_gemm_f16_bias(hy3d_ops *ops, hipStream_t stream,
                                    void *Y_f32, void *W_f16, void *X_f16,
                                    void *bias_f32,
                                    int n_out, int n_in, int n_tok) {
    (void)ops;
    mm_blaslt_run_f16_bias(Y_f32, W_f16, X_f16, bias_f32,
                           n_tok, n_out, n_in, (void *)stream);
}

/* Y[F32] += X*W^T + bias  (in-place residual via beta=1, C=Y). */
static inline void op_gemm_f16_bias_residual(hy3d_ops *ops, hipStream_t stream,
                                             void *Y_f32, void *C_f32,
                                             void *W_f16, void *X_f16,
                                             void *bias_f32,
                                             int n_out, int n_in, int n_tok) {
    (void)ops;
    mm_blaslt_run_f16_bias_residual(Y_f32, C_f32, W_f16, X_f16, bias_f32,
                                    n_tok, n_out, n_in, (void *)stream);
}

/* Y[F16] = X*W^T + bias  (BF16/F16 D output for chained GEMMs). */
static inline void op_gemm_f16_bias_f16d(hy3d_ops *ops, hipStream_t stream,
                                         void *Y_f16, void *W_f16, void *X_f16,
                                         void *bias_f32,
                                         int n_out, int n_in, int n_tok) {
    (void)ops;
    mm_blaslt_run_f16_bias_f16d(Y_f16, W_f16, X_f16, bias_f32,
                                n_tok, n_out, n_in, (void *)stream);
}

/* Y[F16] = GELU(X*W^T + bias)  (fused GELU+downcast for MLP fc1). */
static inline void op_gemm_f16_bias_gelu_f16d(hy3d_ops *ops, hipStream_t stream,
                                              void *Y_f16, void *W_f16, void *X_f16,
                                              void *bias_f32,
                                              int n_out, int n_in, int n_tok) {
    (void)ops;
    mm_blaslt_run_f16_bias_gelu_f16d(Y_f16, W_f16, X_f16, bias_f32,
                                     n_tok, n_out, n_in, (void *)stream);
}
#endif  /* HY3D_HIPBLASLT_ENABLED */

/* ======================================================================== */
/* F16 residual-stream ops (HY3D_DIT_FP16 path)                              */
/* ======================================================================== */

/* In-place exact GELU on f16 buffer. */
static inline void op_gelu_exact_f16(hy3d_ops *ops, hipStream_t stream,
                                     void *x_f16, int n) {
    void *args[] = {&x_f16, &n};
    hipModuleLaunchKernel(ops->gelu_exact_f16, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* dst[f16] += src[f16]. */
static inline void op_add_f16(hy3d_ops *ops, hipStream_t stream,
                              void *dst_f16, void *src_f16, int n) {
    void *args[] = {&dst_f16, &src_f16, &n};
    hipModuleLaunchKernel(ops->add_f16, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* Per-head RMSNorm on f16 data; w stays f32 (per-head_dim, tiny). */
static inline void op_rms_norm_f16(hy3d_ops *ops, hipStream_t stream,
                                   void *data_f16, void *w_f32,
                                   int n_tok, int n_heads, int head_dim,
                                   int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data_f16, &w_f32, &n_tok, &n_heads, &head_dim, &stride, &eps};
    /* One warp per row; 256-thread block = 8 rows/block. */
    hipModuleLaunchKernel(ops->rms_norm_f16, (unsigned)((total + 7) / 8), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* LayerNorm on f16; w/b are f32 [dim].  Launch one block per token. */
static inline void op_layernorm_f16(hy3d_ops *ops, hipStream_t stream,
                                    void *dst_f16, void *src_f16,
                                    void *w_f32, void *b_f32,
                                    int n_tok, int dim) {
    float eps = 1e-5f;
    int nt = 256;
    size_t smem = (size_t)nt * sizeof(float);
    void *args[] = {&dst_f16, &src_f16, &w_f32, &b_f32, &dim, &eps};
    hipModuleLaunchKernel(ops->layernorm_f16, (unsigned)n_tok, 1, 1,
                   (unsigned)nt, 1, 1, smem, stream, args, NULL);
}

/* Cast helpers used at residual-stream boundaries (verify-mode + entry/exit). */
static inline void op_cast_f32_to_f16(hy3d_ops *ops, hipStream_t stream,
                                      void *dst_f16, void *src_f32, int n) {
    void *args[] = {&dst_f16, &src_f32, &n};
    hipModuleLaunchKernel(ops->cast_f32_to_f16, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

static inline void op_cast_f16_to_f32(hy3d_ops *ops, hipStream_t stream,
                                      void *dst_f32, void *src_f16, int n) {
    void *args[] = {&dst_f32, &src_f16, &n};
    hipModuleLaunchKernel(ops->cast_f16_to_f32, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Flash-attention (fp16, WMMA) ----
 * Self-attn: q_len = kv_len = n_tok; head_dim must be 128.
 * Returns 0 on success, -1 if WMMA path unavailable (caller should fall back). */
static inline int op_self_attn_f16(hy3d_ops *ops, hipStream_t stream,
                                   void *out_f16, void *Q_f16, void *K_f16,
                                   void *V_f16,
                                   int n_tok, int dim, int n_heads, int head_dim) {
    (void)dim;
    hipFunction_t fn = NULL;
    if (head_dim == 128) {
        fn = ops->flash_attn_sa_f16_wmma;
    } else if (head_dim == 64) {
        fn = ops->flash_attn_sa_f16_wmma_hd64;
    }
    if (!fn) return -1;
    void *args[] = {&out_f16, &Q_f16, &K_f16, &V_f16,
                    &n_tok, &n_heads, &head_dim};
    unsigned gy = (unsigned)((n_tok + 63) / 64);
    return hipModuleLaunchKernel(fn, (unsigned)n_heads, gy, 1,
                                 128, 1, 1, 0, stream, args, NULL) == hipSuccess ? 0 : -1;
}

/* Cross-attn: asymmetric q_len/kv_len, head_dim must be 128. */
static inline int op_cross_attn_f16(hy3d_ops *ops, hipStream_t stream,
                                    void *out_f16, void *Q_f16, void *K_f16,
                                    void *V_f16,
                                    int q_len, int kv_len, int dim,
                                    int n_heads, int head_dim) {
    hipFunction_t fn = NULL;
    if (head_dim == 128) fn = ops->cross_attn_f16_wmma;
    else if (head_dim == 64) fn = ops->cross_attn_f16_wmma_hd64;
    if (!fn) return -1;
    float scale = 1.0f / sqrtf((float)head_dim);
    void *args[] = {&out_f16, &Q_f16, &K_f16, &V_f16,
                    &q_len, &kv_len, &dim, &n_heads, &head_dim, &scale};
    unsigned gy = (unsigned)((q_len + 63) / 64);
    return hipModuleLaunchKernel(fn, (unsigned)n_heads, gy, 1,
                                 128, 1, 1, 0, stream, args, NULL) == hipSuccess ? 0 : -1;
}

/* ---- MoE gating: softmax + top-K mask, GPU-resident ----
 * logits: f32 [N_tok, n_experts] (raw gate output).
 * weights: f32 [N_tok, n_experts]; only top-K entries non-zero on output. */
static inline void op_moe_gate_softmax_topk(hy3d_ops *ops, hipStream_t stream,
                                            void *logits, void *weights,
                                            int N_tok, int n_experts, int top_k) {
    void *args[] = {&logits, &weights, &N_tok, &n_experts, &top_k};
    hipModuleLaunchKernel(ops->moe_gate_softmax_topk,
                          (unsigned)((N_tok + 127) / 128), 1, 1,
                          128, 1, 1, 0, stream, args, NULL);
}

/* acc[t,j] += weights[t, expert_idx] * src[t,j], for one expert column. */
static inline void op_moe_weighted_add(hy3d_ops *ops, hipStream_t stream,
                                       void *acc, void *src, void *weights,
                                       int expert_idx, int n_experts,
                                       int N_tok, int H_dim) {
    void *args[] = {&acc, &src, &weights, &expert_idx, &n_experts,
                    &N_tok, &H_dim};
    long total = (long)N_tok * H_dim;
    unsigned grid = (unsigned)((total + 255) / 256);
    hipModuleLaunchKernel(ops->moe_weighted_add, grid, 1, 1,
                          256, 1, 1, 0, stream, args, NULL);
}

/* Build per-expert dispatch lists.  counts[e] starts zeroed externally. */
static inline void op_moe_dispatch_build(hy3d_ops *ops, hipStream_t stream,
                                          void *weights, void *counts, void *perm,
                                          int N_tok, int n_experts) {
    void *args[] = {&weights, &counts, &perm, &N_tok, &n_experts};
    hipModuleLaunchKernel(ops->moe_dispatch_build,
                          (unsigned)((N_tok + 127) / 128), 1, 1,
                          128, 1, 1, 0, stream, args, NULL);
}

/* Gather rows by index: dst[k,:] = src[perm[k],:]. */
static inline void op_moe_gather_f16(hy3d_ops *ops, hipStream_t stream,
                                      void *dst, void *src, void *perm,
                                      int n_rows, int dim) {
    void *args[] = {&dst, &src, &perm, &n_rows, &dim};
    hipModuleLaunchKernel(ops->moe_gather_f16,
                          (unsigned)((dim + 255) / 256), (unsigned)n_rows, 1,
                          256, 1, 1, 0, stream, args, NULL);
}

/* Scatter+weight: acc[perm[k], :] += weights[perm[k], expert_idx] * src[k, :]. */
static inline void op_moe_scatter_add_f16_to_f32(hy3d_ops *ops, hipStream_t stream,
                                                  void *acc, void *src, void *perm,
                                                  void *weights, int expert_idx,
                                                  int n_experts, int n_rows, int H_dim) {
    void *args[] = {&acc, &src, &perm, &weights, &expert_idx,
                    &n_experts, &n_rows, &H_dim};
    hipModuleLaunchKernel(ops->moe_scatter_add_f16_to_f32,
                          (unsigned)((H_dim + 255) / 256), (unsigned)n_rows, 1,
                          256, 1, 1, 0, stream, args, NULL);
}

/* ---- GELU activation in-place ---- */
static inline void op_gelu(hy3d_ops *ops, hipStream_t stream,
                           void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(ops->gelu, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

/* ---- Exact GELU (erf) — matches nn.GELU default approximate='none' ---- */
static inline void op_gelu_exact(hy3d_ops *ops, hipStream_t stream,
                                 void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(ops->gelu_exact, (unsigned)((n + 255) / 256), 1, 1,
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

/* ---- LayerScale residual (f16): dst[i] += src[i] * scale[i % dim] ---- */
static inline void op_layerscale_add_f16(hy3d_ops *ops, hipStream_t stream,
                                         void *dst, void *src,
                                         void *scale, int n, int dim) {
    void *args[] = {&dst, &src, &scale, &n, &dim};
    hipModuleLaunchKernel(ops->layerscale_add_f16, (unsigned)((n + 255) / 256), 1, 1,
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

/* f16 variant: same op, f16 data — eliminates the cast f16->f32->LN->f16
 * round-trip used at VAE-fp16 sites.  Fallback to f32 path is the caller's
 * responsibility (op returns -1 if kernel missing).  */
static inline int op_qk_layernorm_f16(hy3d_ops *ops, hipStream_t stream,
                                      void *data_f16,
                                      void *w, void *b,
                                      int n_tok, int n_heads, int head_dim,
                                      int stride) {
    if (!ops->qk_layernorm_f16) return -1;
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data_f16, &w, &b, &n_tok, &n_heads, &head_dim, &stride, &eps};
    return hipModuleLaunchKernel(ops->qk_layernorm_f16,
                                 (unsigned)((total + 255) / 256), 1, 1,
                                 256, 1, 1, 0, stream, args, NULL) == hipSuccess ? 0 : -1;
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

/* ======================================================================== */
/* F16 layout helpers: split / concat / strip on half_raw buffers           */
/* ======================================================================== */

static inline void op_split_qkv_f16(hy3d_ops *ops, hipStream_t stream,
                                    void *Q, void *K, void *V, void *qkv,
                                    int N, int H, int HD) {
    int total = N * H * HD;
    void *args[] = {&Q, &K, &V, &qkv, &N, &H, &HD};
    hipModuleLaunchKernel(ops->split_qkv_f16, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

static inline void op_split_kv_f16(hy3d_ops *ops, hipStream_t stream,
                                   void *K, void *V, void *kv,
                                   int M, int H, int HD) {
    int total = M * H * HD;
    void *args[] = {&K, &V, &kv, &M, &H, &HD};
    hipModuleLaunchKernel(ops->split_kv_f16, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

static inline void op_concat_first_f16(hy3d_ops *ops, hipStream_t stream,
                                       void *out, void *token, void *seq,
                                       int seq_len, int dim) {
    int total = (seq_len + 1) * dim;
    void *args[] = {&out, &token, &seq, &seq_len, &dim};
    hipModuleLaunchKernel(ops->concat_first_f16, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

static inline void op_strip_first_f16(hy3d_ops *ops, hipStream_t stream,
                                      void *dst, void *src,
                                      int seq_len, int dim) {
    int total = (seq_len - 1) * dim;
    void *args[] = {&dst, &src, &seq_len, &dim};
    hipModuleLaunchKernel(ops->strip_first_f16, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

static inline void op_concat_last_dim_f16(hy3d_ops *ops, hipStream_t stream,
                                          void *out, void *a, void *b,
                                          int N, int dim) {
    int total = N * 2 * dim;
    void *args[] = {&out, &a, &b, &N, &dim};
    hipModuleLaunchKernel(ops->concat_last_dim_f16, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, stream, args, NULL);
}

#endif /* HIP_HY3D_OPS_H */
