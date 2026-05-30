/*
 * cuda_trellis2_ops.h - Kernel launch wrappers for TRELLIS.2 CUDA runner
 *
 * Requires cuew.h included before this header.
 */
#ifndef CUDA_TRELLIS2_OPS_H
#define CUDA_TRELLIS2_OPS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef __CUEW_H__
#error "cuda_trellis2_ops.h requires cuew.h to be included first"
#endif

#include "../cublasew.h"

typedef struct {
    int M[27];
    CUdeviceptr src_idx[27];
    CUdeviceptr dst_idx[27];
    CUdeviceptr src_storage;
    CUdeviceptr dst_storage;
} t2_sparse_conv_pack;

typedef struct {
    /* Common kernels */
    CUfunction layernorm;
    CUfunction layernorm_sqrt;
    CUfunction layernorm_cpuavx;
    CUfunction layernorm_serial;
    CUfunction layernorm_welford4;
    CUfunction gemm_tiled;
    CUfunction gemm_f32;
    CUfunction gemm_f32_cpuavx;
    CUfunction gemm_f32_cpuavx16;
    CUfunction gemm_f32_scalar;
    CUfunction gemm_f32_bias_init;
    CUfunction gemm_f32_pair32;
    CUfunction gemm_f32_group;
    CUfunction gemm_f32_group_fma;
    CUfunction gemm_f32_group_biasinit;
    CUfunction gemm_f32_group_biasinit_fma;
    CUfunction gemm_f32_group_mode;
    CUfunction gemm_f32_group_mode_fma;
    CUfunction gemm_f32_group_tree;
    CUfunction gemm_f32_group_tree_fma;
    CUfunction gemm_f32_double;
    CUfunction gelu;
    CUfunction silu;
    CUfunction add;
    CUfunction attn_prefill;
    CUfunction cross_attn;
    CUfunction timestep_embed;
    CUfunction euler_step;
    CUfunction cfg_combine;
    CUfunction split_qkv;
    CUfunction split_kv;
    CUfunction broadcast_add;
    CUfunction add_bias;

    /* TRELLIS.2-specific kernels */
    CUfunction adaln;
    CUfunction gated_add;
    CUfunction modulation;
    CUfunction rope_3d;
    CUfunction rms_norm_perhead;
    CUfunction conv3d_k3;
    CUfunction groupnorm_3d;
    CUfunction silu_inplace;
    CUfunction pixel_shuffle_3d;
    CUfunction layernorm_noaffine;
    CUfunction layernorm_noaffine_sqrt;
    CUfunction layernorm_noaffine_cpuavx;
    CUfunction layernorm_noaffine_serial;
    CUfunction layernorm_noaffine_welford4;
    CUfunction split_qkv_chunk;
    CUfunction split_kv_chunk;
    CUfunction timestep_embed_cossin;
    CUfunction channel_layernorm_3d;
    CUfunction rope_2d_dinov3;
    CUfunction dinov3_patch_embed;
    CUfunction dinov3_prepend_tokens;
    CUfunction layerscale_add;
    CUfunction cross_attn_tiled;

    CUfunction gemm_f16_mma;

    /* MMA attention for head_dim=128 */
    CUfunction attn_mma_hd128;

    /* Sparse conv kernels */
    CUfunction sparse_build_gather_map;
    CUfunction sparse_gather;
    CUfunction sparse_pack_from_gather_map;
    CUfunction sparse_pack_rows;
    CUfunction scatter_add_rows;
    CUfunction scatter_add;
    CUfunction broadcast_bias;
    CUfunction sparse_conv_direct;
    CUfunction residual_add;
    CUfunction c2s_gather;
    CUfunction c2s_residual_repeat;

    int sm_version;
    int use_f32_gemm;   /* 0=F16 weights, 1=F32 weights */
    int use_mma_gemm;   /* 1=use MMA tensor core GEMM (sm_70+, F16 weights) */
    int use_cublas_f32; /* 1=route F32 GEMMs through cuBLAS when available */
    int use_cublas_pedantic;
    int use_tf32_gemm;  /* 1=route F32 GEMMs through TF32 tensor cores (DiT;
                         * f32 exponent range so no fp16 clipping, ~1e-3 rel err) */
    int use_bf16_gemm;  /* 1=cast F32 W/X to bf16 and matmul (f32 accumulate) to
                         * match PyTorch's bf16 DiT reference trajectory (NEGATIVE
                         * result: GEMM-input casting alone barely moves cosine —
                         * superseded by bf16_round, default off scaffolding) */
    int bf16_round;     /* 1=round each DiT-block op output to bf16 precision (kept
                         * in f32 buffers) so the block runs the bf16 trajectory;
                         * paired with use_tf32_gemm (TF32 on bf16-valued inputs ==
                         * a bf16 matmul). Matches PyTorch's bf16 DiT blocks. */
    int use_packed_sparse_conv; /* 1=pack valid rows before sparse conv GEMM */
    cublasew_context *cublas;
    /* Scratch for the bf16 GEMM path (grown on demand; bf16 = 2 bytes/elem). */
    CUfunction cast_f32_to_bf16;
    CUfunction round_bf16;  /* in-place f32 -> bf16-precision -> f32 (bf16_round) */
    CUdeviceptr bf16_w, bf16_x;
    size_t bf16_w_cap, bf16_x_cap;  /* capacity in elements */
} t2_ops;

static int t2_ops_load(t2_ops *ops, CUmodule module, int sm_version) {
    memset(ops, 0, sizeof(*ops));
    ops->sm_version = sm_version;

    #define GET_FN(name, field) \
        if (cuModuleGetFunction(&ops->field, module, name) != CUDA_SUCCESS) { \
            fprintf(stderr, "T2 ops: failed to get kernel '%s'\n", name); \
            return -1; \
        }

    /* Common */
    GET_FN("layernorm_f32",           layernorm);
    GET_FN("layernorm_f32_sqrt",      layernorm_sqrt);
    GET_FN("layernorm_f32_cpuavx",    layernorm_cpuavx);
    GET_FN("layernorm_f32_serial",    layernorm_serial);
    GET_FN("layernorm_welford4_f32",  layernorm_welford4);
    GET_FN("gemm_tiled_f16_f32",      gemm_tiled);
    GET_FN("gemm_f32_f32",            gemm_f32);
    GET_FN("gemm_f32_cpuavx_f32",     gemm_f32_cpuavx);
    GET_FN("gemm_f32_cpuavx16_f32",   gemm_f32_cpuavx16);
    GET_FN("gemm_f32_scalar_f32",     gemm_f32_scalar);
    GET_FN("gemm_f32_bias_init_f32",  gemm_f32_bias_init);
    GET_FN("gemm_f32_pair32_f32",     gemm_f32_pair32);
    GET_FN("gemm_f32_group_f32",      gemm_f32_group);
    GET_FN("gemm_f32_group_fma_f32",  gemm_f32_group_fma);
    GET_FN("gemm_f32_group_biasinit_f32", gemm_f32_group_biasinit);
    GET_FN("gemm_f32_group_biasinit_fma_f32", gemm_f32_group_biasinit_fma);
    GET_FN("gemm_f32_group_mode_f32", gemm_f32_group_mode);
    GET_FN("gemm_f32_group_mode_fma_f32", gemm_f32_group_mode_fma);
    GET_FN("gemm_f32_group_tree_f32", gemm_f32_group_tree);
    GET_FN("gemm_f32_group_tree_fma_f32", gemm_f32_group_tree_fma);
    GET_FN("gemm_f32_double_f32",     gemm_f32_double);
    GET_FN("gelu_f32",                gelu);
    GET_FN("silu_f32",                silu);
    GET_FN("add_f32",                 add);
    GET_FN("cross_attn_f32",          cross_attn);
    GET_FN("timestep_embed_f32",      timestep_embed);
    GET_FN("euler_step_f32",          euler_step);
    GET_FN("cfg_combine_f32",         cfg_combine);
    GET_FN("split_qkv_interleaved_f32", split_qkv);
    GET_FN("split_kv_interleaved_f32",  split_kv);
    GET_FN("broadcast_add_f32",       broadcast_add);
    GET_FN("add_bias_f32",            add_bias);

    /* Optional bf16 cast/round kernels (for the bf16 DiT paths). Tolerant lookup:
     * if absent (e.g. an older cached module) the bf16 paths stay disabled. */
    if (cuModuleGetFunction(&ops->cast_f32_to_bf16, module, "t2_cast_f32_to_bf16")
            != CUDA_SUCCESS)
        ops->cast_f32_to_bf16 = NULL;
    if (cuModuleGetFunction(&ops->round_bf16, module, "t2_round_f32_bf16")
            != CUDA_SUCCESS)
        ops->round_bf16 = NULL;

    if (sm_version >= 70) {
        GET_FN("attn_prefill_f32",    attn_prefill);
    } else {
        GET_FN("flash_attn_tiled_f32", attn_prefill);
    }

    /* TRELLIS.2-specific */
    GET_FN("adaln_f32",               adaln);
    GET_FN("gated_add_f32",           gated_add);
    GET_FN("modulation_f32",          modulation);
    GET_FN("rope_3d_f32",             rope_3d);
    GET_FN("rms_norm_perhead_f32",    rms_norm_perhead);
    GET_FN("conv3d_k3_f32",           conv3d_k3);
    GET_FN("groupnorm_3d_f32",        groupnorm_3d);
    GET_FN("silu_inplace_f32",        silu_inplace);
    GET_FN("pixel_shuffle_3d_f32",    pixel_shuffle_3d);
    GET_FN("layernorm_noaffine_f32",  layernorm_noaffine);
    GET_FN("layernorm_noaffine_f32_sqrt", layernorm_noaffine_sqrt);
    GET_FN("layernorm_noaffine_f32_cpuavx", layernorm_noaffine_cpuavx);
    GET_FN("layernorm_noaffine_serial_f32", layernorm_noaffine_serial);
    GET_FN("layernorm_noaffine_welford4_f32", layernorm_noaffine_welford4);
    GET_FN("split_qkv_chunk_f32",    split_qkv_chunk);
    GET_FN("split_kv_chunk_f32",     split_kv_chunk);
    GET_FN("timestep_embed_cossin_f32", timestep_embed_cossin);
    GET_FN("channel_layernorm_3d_f32", channel_layernorm_3d);
    GET_FN("rope_2d_dinov3_f32",      rope_2d_dinov3);
    GET_FN("dinov3_patch_embed_f32",   dinov3_patch_embed);
    GET_FN("dinov3_prepend_tokens_f32", dinov3_prepend_tokens);
    GET_FN("layerscale_add_f32",       layerscale_add);
    GET_FN("cross_attn_tiled_f32",    cross_attn_tiled);

    if (sm_version >= 70) {
        GET_FN("gemm_f16_f32",        gemm_f16_mma);
        GET_FN("attn_mma_hd128_f32",  attn_mma_hd128);
    }

    /* Sparse conv kernels */
    GET_FN("sparse_build_gather_map_f32", sparse_build_gather_map);
    GET_FN("sparse_gather_f32",           sparse_gather);
    GET_FN("sparse_pack_from_gather_map_f32", sparse_pack_from_gather_map);
    GET_FN("sparse_pack_rows_f32",        sparse_pack_rows);
    GET_FN("scatter_add_rows_f32",        scatter_add_rows);
    GET_FN("scatter_add_f32",             scatter_add);
    GET_FN("broadcast_bias_f32",          broadcast_bias);
    GET_FN("sparse_conv3d_direct_f32",    sparse_conv_direct);
    GET_FN("residual_add_f32",            residual_add);
    GET_FN("c2s_gather_f32",              c2s_gather);
    GET_FN("c2s_residual_repeat_f32",     c2s_residual_repeat);

    #undef GET_FN
    return 0;
}

/* ======================================================================== */
/* Launch wrappers                                                          */
/* ======================================================================== */

/* In-place round a buffer to bf16 precision (kept as f32). No-op if the kernel
 * is missing or n<=0. Used by the bf16-block DiT path (ops->bf16_round). */
static inline void t2_op_round_bf16(t2_ops *ops, CUstream s,
                                    CUdeviceptr x, long n) {
    if (!ops->round_bf16 || n <= 0) return;
    void *args[] = {&x, &n};
    cuLaunchKernel(ops->round_bf16, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_gemm(t2_ops *ops, CUstream s,
                                CUdeviceptr Y, CUdeviceptr W,
                                CUdeviceptr X, CUdeviceptr bias,
                                int n_out, int n_in, int n_tok) {
    if (ops->use_f32_gemm && getenv("T2_SCVAE_SCALAR_GEMM")) {
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        cuLaunchKernel(ops->gemm_f32_scalar,
                       (unsigned)(((size_t)n_out * n_tok + 255) / 256), 1, 1,
                       256, 1, 1, 0, s, args, NULL);
        return;
    }
    if (ops->use_f32_gemm && getenv("T2_SCVAE_CPUAVX_GEMM")) {
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        cuLaunchKernel(ops->gemm_f32_cpuavx,
                       (unsigned)(((size_t)n_out * n_tok + 255) / 256), 1, 1,
                       256, 1, 1, 0, s, args, NULL);
        return;
    }
    if (ops->use_f32_gemm && bias && getenv("T2_SCVAE_CPUAVX16_GEMM")) {
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        cuLaunchKernel(ops->gemm_f32_cpuavx16,
                       (unsigned)(((size_t)n_out * n_tok + 255) / 256), 1, 1,
                       256, 1, 1, 0, s, args, NULL);
        return;
    }
    if (ops->use_f32_gemm && bias && getenv("T2_SCVAE_BIAS_INIT_GEMM")) {
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        unsigned gx = (unsigned)((n_out + 63) / 64);
        unsigned gy = (unsigned)((n_tok + 15) / 16);
        cuLaunchKernel(ops->gemm_f32_bias_init, gx, gy, 1, 16, 16, 1, 0, s,
                       args, NULL);
        return;
    }
    if (ops->use_f32_gemm && ops->gemm_f32_double &&
        getenv("T2_SCVAE_DOUBLE_GEMM") &&
        (bias || getenv("T2_SCVAE_DOUBLE_GEMM_ALL"))) {
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        cuLaunchKernel(ops->gemm_f32_double,
                       (unsigned)(((size_t)n_out * n_tok + 255) / 256),
                       1, 1, 256, 1, 1, 0, s, args, NULL);
        return;
    }
    {
        const char *group_env = getenv("T2_SCVAE_GROUP_BIASINIT_GEMM");
        int group = group_env && group_env[0] ? atoi(group_env) : 0;
        if (ops->use_f32_gemm && bias && group > 0) {
            CUfunction fn = getenv("T2_SCVAE_GROUP_BIASINIT_GEMM_FMA")
                ? ops->gemm_f32_group_biasinit_fma
                : ops->gemm_f32_group_biasinit;
            if (fn) {
                void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &group};
                cuLaunchKernel(fn,
                               (unsigned)(((size_t)n_out * n_tok + 255) / 256),
                               1, 1, 256, 1, 1, 0, s, args, NULL);
                return;
            }
        }
    }
    if (ops->use_f32_gemm && ops->cublas && bias &&
        getenv("T2_SCVAE_CUBLAS_BETA1_GEMM")) {
        int rc = -1;
        int total = n_tok * n_out;
        void *bargs[] = {&Y, &bias, &n_tok, &n_out};
        cuLaunchKernel(ops->broadcast_bias,
                       (unsigned)((total + 255) / 256), 1, 1,
                       256, 1, 1, 0, s, bargs, NULL);
        if (cublasewSetStream(ops->cublas, s) == 0) {
            rc = cublasew_gemm_f32_rowmajor_nt_beta1(ops->cublas, Y, W, X,
                                                     n_tok, n_out, n_in);
        }
        if (getenv("T2_SCVAE_DEBUG_GEMM"))
            fprintf(stderr, "T2: cuBLAS beta=1 biased GEMM %s for [%d,%d]x[%d,%d]\n",
                    rc == 0 ? "ok" : "fallback", n_tok, n_in, n_out, n_in);
        if (rc == 0) return;
    }
    {
        const char *group_env = getenv("T2_SCVAE_GROUP_GEMM");
        int group = group_env && group_env[0] ? atoi(group_env) : 0;
        if (ops->use_f32_gemm && group > 0 &&
            (bias || getenv("T2_SCVAE_GROUP_GEMM_ALL"))) {
            CUfunction fn = getenv("T2_SCVAE_GROUP_GEMM_FMA")
                ? ops->gemm_f32_group_fma
                : ops->gemm_f32_group;
            if (fn) {
                void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &group};
                cuLaunchKernel(fn,
                               (unsigned)(((size_t)n_out * n_tok + 255) / 256),
                               1, 1, 256, 1, 1, 0, s, args, NULL);
                return;
            }
        }
    }
    if (ops->use_f32_gemm && ops->cublas && bias &&
        getenv("T2_SCVAE_CUBLASLT_BIAS_GEMM")) {
        int rc = -1;
        if (cublasewSetStream(ops->cublas, s) == 0) {
            rc = cublasew_gemm_f32_lt_bias_rowmajor_nt(ops->cublas, Y, W, X,
                                                       bias, n_tok, n_out, n_in);
        }
        if (getenv("T2_SCVAE_DEBUG_GEMM"))
            fprintf(stderr, "T2: cuBLASLt bias GEMM %s for [%d,%d]x[%d,%d]\n",
                    rc == 0 ? "ok" : "fallback", n_tok, n_in, n_out, n_in);
        if (rc == 0) return;
    }
    if (ops->use_f32_gemm && ops->cublas && !bias &&
        getenv("T2_SCVAE_CUBLASLT_GEMM")) {
        int rc = -1;
        if (cublasewSetStream(ops->cublas, s) == 0) {
            rc = cublasew_gemm_f32_lt_rowmajor_nt(ops->cublas, Y, W, X,
                                                  n_tok, n_out, n_in);
        }
        if (getenv("T2_SCVAE_DEBUG_GEMM"))
            fprintf(stderr, "T2: cuBLASLt GEMM %s for [%d,%d]x[%d,%d]\n",
                    rc == 0 ? "ok" : "fallback", n_tok, n_in, n_out, n_in);
        if (rc == 0) return;
    }
    const char *cublas_bias_only = getenv("T2_SCVAE_CUBLAS_BIAS_ONLY");
    int use_cublas_bias_only = cublas_bias_only && atoi(cublas_bias_only);
    if (ops->use_f32_gemm && ops->cublas &&
        (ops->use_cublas_f32 || (use_cublas_bias_only && bias))) {
        int ok = ops->use_cublas_pedantic
            ? cublasew_gemm_f32_pedantic_rowmajor_nt(ops->cublas, Y, W, X,
                                                     n_tok, n_out, n_in)
            : cublasew_gemm_f32_rowmajor_nt(ops->cublas, Y, W, X,
                                            n_tok, n_out, n_in);
        if (ok == 0) {
            if (bias) {
                void *bargs[] = {&Y, &bias, &n_out, &n_tok};
                cuLaunchKernel(ops->add_bias,
                               (unsigned)((n_out * n_tok + 255) / 256), 1, 1,
                               256, 1, 1, 0, s, bargs, NULL);
            }
            return;
        }
    }
    if (ops->use_f32_gemm && ops->use_bf16_gemm && ops->cublas && ops->cast_f32_to_bf16) {
        /* bf16 matmul to match PyTorch's bf16 DiT reference: cast W (f32, == exact
         * bf16 upcast) and X (f32 activations) to bf16, then bf16-in/f32-accumulate
         * GEMM. Scratch grown on demand (W re-cast each call — fine for accuracy
         * testing; a resident bf16 weight cache would remove the W re-cast). */
        size_t wn = (size_t)n_out * n_in, xn = (size_t)n_tok * n_in;
        if (ops->bf16_w_cap < wn) {
            if (ops->bf16_w) cuMemFree(ops->bf16_w);
            ops->bf16_w_cap = (cuMemAlloc(&ops->bf16_w, wn * 2) == CUDA_SUCCESS) ? wn : 0;
        }
        if (ops->bf16_x_cap < xn) {
            if (ops->bf16_x) cuMemFree(ops->bf16_x);
            ops->bf16_x_cap = (cuMemAlloc(&ops->bf16_x, xn * 2) == CUDA_SUCCESS) ? xn : 0;
        }
        if (ops->bf16_w_cap >= wn && ops->bf16_x_cap >= xn) {
            long wn_l = (long)wn, xn_l = (long)xn;
            void *wa[] = {&W, &ops->bf16_w, &wn_l};
            cuLaunchKernel(ops->cast_f32_to_bf16, (unsigned)((wn + 255) / 256), 1, 1, 256, 1, 1, 0, s, wa, NULL);
            void *xa[] = {&X, &ops->bf16_x, &xn_l};
            cuLaunchKernel(ops->cast_f32_to_bf16, (unsigned)((xn + 255) / 256), 1, 1, 256, 1, 1, 0, s, xa, NULL);
            if (cublasewSetStream(ops->cublas, s) == 0 &&
                cublasew_gemm_bf16_bf16_f32_rowmajor_nt(ops->cublas, Y, ops->bf16_w, ops->bf16_x,
                                                        n_tok, n_out, n_in) == 0) {
                if (bias) {
                    void *bargs[] = {&Y, &bias, &n_out, &n_tok};
                    cuLaunchKernel(ops->add_bias,
                                   (unsigned)(((size_t)n_out * n_tok + 255) / 256), 1, 1,
                                   256, 1, 1, 0, s, bargs, NULL);
                }
                return;
            }
        }
    }
    if (ops->use_f32_gemm && ops->use_tf32_gemm && ops->cublas) {
        /* DiT default: TF32 tensor-core GEMM. F32 weights/activations in & out,
         * f32 exponent range (no fp16 clipping of hot intermediates), TF32-reduced
         * mantissa (~1e-3 rel err — more precise than PyTorch's bf16). Far faster
         * than the plain tiled f32 kernel on the dense 4096-token Stage-1 grid. */
        if (cublasewSetStream(ops->cublas, s) == 0 &&
            cublasew_gemm_f32_tf32_rowmajor_nt(ops->cublas, Y, W, X,
                                               n_tok, n_out, n_in) == 0) {
            if (bias) {
                void *bargs[] = {&Y, &bias, &n_out, &n_tok};
                cuLaunchKernel(ops->add_bias,
                               (unsigned)(((size_t)n_out * n_tok + 255) / 256), 1, 1,
                               256, 1, 1, 0, s, bargs, NULL);
            }
            return;
        }
    }
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    if (ops->use_f32_gemm) {
        /* F32 weights: use tiled F32 GEMM */
        unsigned gx = (unsigned)((n_out + 63) / 64);
        unsigned gy = (unsigned)((n_tok + 15) / 16);
        cuLaunchKernel(ops->gemm_f32, gx, gy, 1, 16, 16, 1, 0, s, args, NULL);
    } else if (ops->use_mma_gemm && ops->gemm_f16_mma) {
        /* F16 weights + MMA tensor cores (sm_70+)
         * Block: 256 threads (8 warps), each warp handles 64 output cols.
         * blockIdx.x * 256 with overlapping warps — grid uses 256 stride. */
        unsigned gx = (unsigned)((n_out + 255) / 256);
        unsigned gy = (unsigned)((n_tok + 15) / 16);
        size_t smem = 32 * 16 * sizeof(float);  /* smem_x: 256 threads/8=32 rows × 16 cols */
        cuLaunchKernel(ops->gemm_f16_mma, gx, gy, 1,
                       256, 1, 1, smem, s, args, NULL);
    } else {
        /* F16 weights + tiled GEMM (fallback) */
        unsigned gx = (unsigned)((n_out + 63) / 64);
        unsigned gy = (unsigned)((n_tok + 15) / 16);
        cuLaunchKernel(ops->gemm_tiled, gx, gy, 1, 16, 16, 1, 0, s, args, NULL);
    }
}

/* GEMM with explicitly F32 weights (for timestep MLP, modulation) */
static inline void t2_op_gemm_f32(t2_ops *ops, CUstream s,
                                CUdeviceptr Y, CUdeviceptr W,
                                CUdeviceptr X, CUdeviceptr bias,
                                int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    cuLaunchKernel(ops->gemm_f32, gx, gy, 1, 16, 16, 1, 0, s, args, NULL);
}

static inline int t2_op_gemm_f32_lt_bias(t2_ops *ops, CUstream s,
                                           CUdeviceptr Y, CUdeviceptr W,
                                           CUdeviceptr X, CUdeviceptr bias,
                                           int n_out, int n_in, int n_tok) {
    if (!ops->use_f32_gemm || !ops->cublas || !bias) return -1;
    if (cublasewSetStream(ops->cublas, s) != 0) return -1;
    int rc = cublasew_gemm_f32_lt_bias_rowmajor_nt(ops->cublas, Y, W, X, bias,
                                                   n_tok, n_out, n_in);
    if (getenv("T2_SCVAE_DEBUG_GEMM"))
        fprintf(stderr, "T2: cuBLASLt bias GEMM %s for [%d,%d]x[%d,%d]\n",
                rc == 0 ? "ok" : "fallback", n_tok, n_in, n_out, n_in);
    return rc;
}

static inline int t2_op_gemm_f32_lt(t2_ops *ops, CUstream s,
                                      CUdeviceptr Y, CUdeviceptr W,
                                      CUdeviceptr X,
                                      int n_out, int n_in, int n_tok) {
    if (!ops->use_f32_gemm || !ops->cublas) return -1;
    if (cublasewSetStream(ops->cublas, s) != 0) return -1;
    int rc = cublasew_gemm_f32_lt_rowmajor_nt(ops->cublas, Y, W, X,
                                              n_tok, n_out, n_in);
    if (getenv("T2_SCVAE_DEBUG_GEMM"))
        fprintf(stderr, "T2: cuBLASLt GEMM %s for [%d,%d]x[%d,%d]\n",
                rc == 0 ? "ok" : "fallback", n_tok, n_in, n_out, n_in);
    return rc;
}

static inline int t2_op_gemm_f32_bias_init(t2_ops *ops, CUstream s,
                                             CUdeviceptr Y, CUdeviceptr W,
                                             CUdeviceptr X, CUdeviceptr bias,
                                             int n_out, int n_in, int n_tok) {
    if (!ops->use_f32_gemm || !bias || !ops->gemm_f32_bias_init) return -1;
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    cuLaunchKernel(ops->gemm_f32_bias_init, gx, gy, 1, 16, 16, 1, 0, s,
                   args, NULL);
    return 0;
}

static inline int t2_op_gemm_f32_cublas_beta1(t2_ops *ops, CUstream s,
                                                CUdeviceptr Y, CUdeviceptr W,
                                                CUdeviceptr X, int n_out,
                                                int n_in, int n_tok) {
    if (!ops->use_f32_gemm || !ops->cublas) return -1;
    if (cublasewSetStream(ops->cublas, s) != 0) return -1;
    if (getenv("T2_SCVAE_CUBLASLT_BETA1")) {
        int rc = cublasew_gemm_f32_lt_rowmajor_nt_beta1(ops->cublas, Y, W, X,
                                                        n_tok, n_out, n_in);
        if (getenv("T2_SCVAE_DEBUG_GEMM"))
            fprintf(stderr, "T2: cuBLASLt beta=1 GEMM %s for [%d,%d]x[%d,%d]\n",
                    rc == 0 ? "ok" : "fallback", n_tok, n_in, n_in, n_out);
        if (rc == 0) return 0;
    }
    return cublasew_gemm_f32_rowmajor_nt_beta1(ops->cublas, Y, W, X,
                                              n_tok, n_out, n_in);
}

static inline int t2_op_gemm_f32_pair32(t2_ops *ops, CUstream s,
                                          CUdeviceptr Y, CUdeviceptr W,
                                          CUdeviceptr X, CUdeviceptr bias,
                                          int n_out, int n_in, int n_tok) {
    if (!ops->use_f32_gemm || !ops->gemm_f32_pair32) return -1;
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    cuLaunchKernel(ops->gemm_f32_pair32,
                   (unsigned)(((size_t)n_out * n_tok + 255) / 256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
    return 0;
}

static inline int t2_op_gemm_f32_group(t2_ops *ops, CUstream s,
                                         CUdeviceptr Y, CUdeviceptr W,
                                         CUdeviceptr X, CUdeviceptr bias,
                                         int n_out, int n_in, int n_tok,
                                         int group) {
    if (!ops->use_f32_gemm || group <= 0) return -1;
    int mode = 0;
    const char *mode_env = getenv("T2_SCVAE_OUTPUT_GROUP_MODE");
    if (mode_env && mode_env[0]) mode = atoi(mode_env);
    int use_fma = getenv("T2_SCVAE_OUTPUT_GROUP_FMA") ? 1 : 0;
    int biasinit = getenv("T2_SCVAE_OUTPUT_GROUP_BIASINIT") ? 1 : 0;
    int tree = getenv("T2_SCVAE_OUTPUT_GROUP_TREE") ? 1 : 0;
    if (tree) {
        CUfunction fn = use_fma ? ops->gemm_f32_group_tree_fma : ops->gemm_f32_group_tree;
        if (!fn) return -1;
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &group};
        cuLaunchKernel(fn,
                       (unsigned)(((size_t)n_out * n_tok + 255) / 256), 1, 1,
                       256, 1, 1, 0, s, args, NULL);
        return 0;
    }
    if (mode) {
        CUfunction fn = use_fma ? ops->gemm_f32_group_mode_fma : ops->gemm_f32_group_mode;
        if (!fn) return -1;
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &group, &mode};
        cuLaunchKernel(fn,
                       (unsigned)(((size_t)n_out * n_tok + 255) / 256), 1, 1,
                       256, 1, 1, 0, s, args, NULL);
        return 0;
    }
    CUfunction fn = biasinit
        ? (use_fma ? ops->gemm_f32_group_biasinit_fma : ops->gemm_f32_group_biasinit)
        : (use_fma ? ops->gemm_f32_group_fma : ops->gemm_f32_group);
    if (!fn) return -1;
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &group};
    cuLaunchKernel(fn,
                   (unsigned)(((size_t)n_out * n_tok + 255) / 256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
    return 0;
}

static inline void t2_op_layernorm(t2_ops *ops, CUstream s,
                                     CUdeviceptr dst, CUdeviceptr src,
                                     CUdeviceptr w, CUdeviceptr b,
                                     int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    const char *serial_ln = getenv("T2_SCVAE_SERIAL_LN");
    if (serial_ln && serial_ln[0]) {
        int mode = atoi(serial_ln);
        if (mode <= 0) mode = 1;
        void *sargs[] = {&dst, &src, &w, &b, &dim, &eps, &mode};
        cuLaunchKernel(ops->layernorm_serial, (unsigned)n_tok, 1, 1,
                       1, 1, 1, 0, s, sargs, NULL);
    } else if (getenv("T2_SCVAE_WELFORD_AFFINE_LN") && (dim % 4) == 0) {
        cuLaunchKernel(ops->layernorm_welford4, (unsigned)n_tok, 1, 1,
                       32, 8, 1, 12 * sizeof(float), s, args, NULL);
    } else if (getenv("T2_SCVAE_CPUAVX_LN")) {
        cuLaunchKernel(ops->layernorm_cpuavx, (unsigned)n_tok, 1, 1,
                       1, 1, 1, 0, s, args, NULL);
    } else {
        CUfunction fn = getenv("T2_SCVAE_LN_SQRT") ? ops->layernorm_sqrt : ops->layernorm;
        cuLaunchKernel(fn, (unsigned)n_tok, 1, 1,
                       256, 1, 1, 256 * sizeof(float), s, args, NULL);
    }
}

static inline void t2_op_layernorm_welford_eps(t2_ops *ops, CUstream s,
                                                 CUdeviceptr dst,
                                                 CUdeviceptr src,
                                                 CUdeviceptr w,
                                                 CUdeviceptr b,
                                                 int n_tok, int dim,
                                                 float eps) {
    if (!ops->layernorm_welford4 || (dim % 4) != 0) {
        t2_op_layernorm(ops, s, dst, src, w, b, n_tok, dim);
        return;
    }
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    cuLaunchKernel(ops->layernorm_welford4, (unsigned)n_tok, 1, 1,
                   32, 8, 1, 12 * sizeof(float), s, args, NULL);
}

static inline void t2_op_layernorm_noaffine_eps(t2_ops *ops, CUstream s,
                                                  CUdeviceptr dst,
                                                  CUdeviceptr src,
                                                  int n_tok, int dim,
    float eps) {
    void *args[] = {&dst, &src, &dim, &eps};
    const char *serial_ln = getenv("T2_SCVAE_SERIAL_LN");
    if (serial_ln && serial_ln[0]) {
        int mode = atoi(serial_ln);
        if (mode <= 0) mode = 1;
        void *sargs[] = {&dst, &src, &dim, &eps, &mode};
        cuLaunchKernel(ops->layernorm_noaffine_serial, (unsigned)n_tok, 1, 1,
                       1, 1, 1, 0, s, sargs, NULL);
    } else if (getenv("T2_SCVAE_WELFORD_LN") && (dim % 4) == 0) {
        cuLaunchKernel(ops->layernorm_noaffine_welford4, (unsigned)n_tok, 1, 1,
                       32, 8, 1, 12 * sizeof(float), s, args, NULL);
    } else if (getenv("T2_SCVAE_CPUAVX_LN")) {
        cuLaunchKernel(ops->layernorm_noaffine_cpuavx, (unsigned)n_tok, 1, 1,
                       1, 1, 1, 0, s, args, NULL);
    } else {
        CUfunction fn = getenv("T2_SCVAE_LN_SQRT")
            ? ops->layernorm_noaffine_sqrt
            : ops->layernorm_noaffine;
        cuLaunchKernel(fn, (unsigned)n_tok, 1, 1,
                       256, 1, 1, 512 * sizeof(float), s, args, NULL);
    }
}

static inline void t2_op_layernorm_noaffine_serial_eps(t2_ops *ops, CUstream s,
                                                         CUdeviceptr dst,
                                                         CUdeviceptr src,
                                                         int n_tok, int dim,
                                                         float eps, int mode) {
    void *args[] = {&dst, &src, &dim, &eps, &mode};
    cuLaunchKernel(ops->layernorm_noaffine_serial, (unsigned)n_tok, 1, 1,
                   1, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_layernorm_noaffine_welford_eps(t2_ops *ops, CUstream s,
                                                          CUdeviceptr dst,
                                                          CUdeviceptr src,
                                                          int n_tok, int dim,
                                                          float eps) {
    if (!ops->layernorm_noaffine_welford4 || (dim % 4) != 0) {
        t2_op_layernorm_noaffine_eps(ops, s, dst, src, n_tok, dim, eps);
        return;
    }
    void *args[] = {&dst, &src, &dim, &eps};
    cuLaunchKernel(ops->layernorm_noaffine_welford4, (unsigned)n_tok, 1, 1,
                   32, 8, 1, 12 * sizeof(float), s, args, NULL);
}

static inline void t2_op_adaln(t2_ops *ops, CUstream s,
                                 CUdeviceptr dst, CUdeviceptr src,
                                 CUdeviceptr shift, CUdeviceptr scale,
                                 int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &shift, &scale, &dim, &eps};
    cuLaunchKernel(ops->adaln, (unsigned)n_tok, 1, 1,
                   256, 1, 1, 512 * sizeof(float), s, args, NULL);
}

static inline void t2_op_gated_add(t2_ops *ops, CUstream s,
                                     CUdeviceptr dst, CUdeviceptr src,
                                     CUdeviceptr gate, int n, int dim) {
    void *args[] = {&dst, &src, &gate, &n, &dim};
    cuLaunchKernel(ops->gated_add, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_modulation(t2_ops *ops, CUstream s,
                                      CUdeviceptr out, CUdeviceptr t_emb,
                                      CUdeviceptr mod_w, CUdeviceptr mod_b,
                                      CUdeviceptr blk_bias,
                                      int dim, int out_dim) {
    void *args[] = {&out, &t_emb, &mod_w, &mod_b, &blk_bias, &dim, &out_dim};
    /* Warp-per-row: 256 threads/block = 8 warps = 8 output rows per block. Grid
     * covers all out_dim rows so the whole GPU is used (was a single block). */
    unsigned warps_per_block = 256u / 32u;
    unsigned grid = (unsigned)(((size_t)out_dim + warps_per_block - 1) / warps_per_block);
    cuLaunchKernel(ops->modulation, grid, 1, 1,
                   256, 1, 1, (size_t)dim * sizeof(float), s, args, NULL);
}

static inline void t2_op_rope_3d(t2_ops *ops, CUstream s,
                                    CUdeviceptr data,
                                    CUdeviceptr rope_cos, CUdeviceptr rope_sin,
                                    int N, int dim, int n_heads, int head_dim,
                                    int n_freqs, int axis_dim) {
    void *args[] = {&data, &rope_cos, &rope_sin, &N, &dim, &n_heads,
                    &head_dim, &n_freqs, &axis_dim};
    cuLaunchKernel(ops->rope_3d, (unsigned)N, 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_rms_norm_perhead(t2_ops *ops, CUstream s,
                                            CUdeviceptr data, CUdeviceptr gamma,
                                            int n_tok, int n_heads, int head_dim,
                                            int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data, &gamma, &n_tok, &n_heads, &head_dim, &stride, &eps};
    cuLaunchKernel(ops->rms_norm_perhead, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_gelu(t2_ops *ops, CUstream s, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(ops->gelu, (unsigned)((n+255)/256), 1, 1, 256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_add(t2_ops *ops, CUstream s, CUdeviceptr dst, CUdeviceptr src, int n) {
    void *args[] = {&dst, &src, &n};
    cuLaunchKernel(ops->add, (unsigned)((n+255)/256), 1, 1, 256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_cross_attn(t2_ops *ops, CUstream s,
                                      CUdeviceptr out,
                                      CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                                      int q_len, int kv_len, int dim,
                                      int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Use MMA attention for head_dim=128 on sm_70+ */
    if (head_dim == 128 && ops->attn_mma_hd128) {
        void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim,
                        &n_heads, &head_dim, &scale};
        unsigned gy = (unsigned)((q_len + 63) / 64);
        cuLaunchKernel(ops->attn_mma_hd128, (unsigned)n_heads, gy, 1,
                       128, 1, 1, 0, s, args, NULL);
        return;
    }

    /* Fallback: scalar attention kernels */
    size_t smem = (size_t)(kv_len + 128) * sizeof(float);
    /* sm_120 supports up to 228KB dynamic smem. Use smem-based kernel up to 200KB
     * (kv_len ~50K). Beyond that, fall back to tiled kernel. */
    size_t smem_limit = (ops->sm_version >= 120) ? 200 * 1024 :
                        (ops->sm_version >= 80)  ? 160 * 1024 :
                        (ops->sm_version >= 70)  ? 96 * 1024 : 48 * 1024;
    if (smem <= smem_limit) {
        /* Set max dynamic smem for this kernel if needed */
        if (smem > 48 * 1024) {
            cuFuncSetAttribute(ops->cross_attn,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem);
        }
        int nt = 128;
        void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim,
                        &n_heads, &head_dim, &scale};
        cuLaunchKernel(ops->cross_attn, (unsigned)n_heads, (unsigned)q_len, 1,
                       (unsigned)nt, 1, 1, smem, s, args, NULL);
    } else {
        /* Very large N: tiled online-softmax (1 thread per head×query) */
        void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim,
                        &n_heads, &head_dim, &scale};
        cuLaunchKernel(ops->cross_attn_tiled, (unsigned)n_heads, (unsigned)q_len, 1,
                       1, 1, 1, 0, s, args, NULL);
    }
}

static inline void t2_op_self_attn(t2_ops *ops, CUstream s,
                                     CUdeviceptr out,
                                     CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                                     int n_tok, int dim, int n_heads, int head_dim) {
    t2_op_cross_attn(ops, s, out, Q, K, V, n_tok, n_tok, dim, n_heads, head_dim);
}

static inline void t2_op_timestep_embed(t2_ops *ops, CUstream s,
                                          CUdeviceptr out, float t, int dim) {
    void *args[] = {&out, &t, &dim};
    cuLaunchKernel(ops->timestep_embed, (unsigned)((dim/2 + 255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_euler_step(t2_ops *ops, CUstream s,
                                      CUdeviceptr x, CUdeviceptr v, float dt, int n) {
    void *args[] = {&x, &v, &dt, &n};
    cuLaunchKernel(ops->euler_step, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_cfg_combine(t2_ops *ops, CUstream s,
                                       CUdeviceptr out, CUdeviceptr cond,
                                       CUdeviceptr uncond, float scale, int n) {
    void *args[] = {&out, &cond, &uncond, &scale, &n};
    cuLaunchKernel(ops->cfg_combine, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_split_qkv(t2_ops *ops, CUstream s,
                                      CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                                      CUdeviceptr qkv, int N, int H, int HD) {
    int total = N * H * HD;
    void *args[] = {&Q, &K, &V, &qkv, &N, &H, &HD};
    cuLaunchKernel(ops->split_qkv, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_split_kv(t2_ops *ops, CUstream s,
                                     CUdeviceptr K, CUdeviceptr V,
                                     CUdeviceptr kv, int M, int H, int HD) {
    int total = M * H * HD;
    void *args[] = {&K, &V, &kv, &M, &H, &HD};
    cuLaunchKernel(ops->split_kv, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

/* Non-interleaved chunk splits (for TRELLIS.2 standard torch.chunk(3)) */
static inline void t2_op_split_qkv_chunk(t2_ops *ops, CUstream s,
                                            CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V,
                                            CUdeviceptr qkv, int N, int W) {
    int total = N * W;
    void *args[] = {&Q, &K, &V, &qkv, &N, &W};
    cuLaunchKernel(ops->split_qkv_chunk, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_split_kv_chunk(t2_ops *ops, CUstream s,
                                           CUdeviceptr K, CUdeviceptr V,
                                           CUdeviceptr kv, int M, int W) {
    int total = M * W;
    void *args[] = {&K, &V, &kv, &M, &W};
    cuLaunchKernel(ops->split_kv_chunk, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_conv3d(t2_ops *ops, CUstream s,
                                  CUdeviceptr out, CUdeviceptr inp,
                                  CUdeviceptr W, CUdeviceptr bias,
                                  int Ci, int Co, int D, int H, int Wi) {
    int spatial = D * H * Wi;
    void *args[] = {&out, &inp, &W, &bias, &Ci, &Co, &D, &H, &Wi};
    cuLaunchKernel(ops->conv3d_k3, (unsigned)Co, (unsigned)((spatial+63)/64), 1,
                   64, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_groupnorm_3d(t2_ops *ops, CUstream s,
                                        CUdeviceptr dst, CUdeviceptr src,
                                        CUdeviceptr w, CUdeviceptr b,
                                        int C, int spatial, int G) {
    void *args[] = {&dst, &src, &w, &b, &C, &spatial, &G};
    cuLaunchKernel(ops->groupnorm_3d, (unsigned)G, 1, 1,
                   256, 1, 1, 512 * sizeof(float), s, args, NULL);
}

static inline void t2_op_silu_inplace(t2_ops *ops, CUstream s,
                                        CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(ops->silu_inplace, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_channel_layernorm_3d(t2_ops *ops, CUstream s,
                                                CUdeviceptr dst, CUdeviceptr src,
                                                CUdeviceptr w, CUdeviceptr b,
                                                int C, int spatial) {
    void *args[] = {&dst, &src, &w, &b, &C, &spatial};
    cuLaunchKernel(ops->channel_layernorm_3d, (unsigned)((spatial+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_pixel_shuffle_3d(t2_ops *ops, CUstream s,
                                            CUdeviceptr dst, CUdeviceptr src,
                                            int C, int D, int H, int W) {
    int total = C * 2*D * 2*H * 2*W;
    void *args[] = {&dst, &src, &C, &D, &H, &W};
    cuLaunchKernel(ops->pixel_shuffle_3d, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

/* ======================================================================== */
/* Sparse convolution ops                                                    */
/* ======================================================================== */

/* Build gather map: for each voxel, find neighbor indices via hash table.
 * out_map: [N, 27] int32 on GPU */
static inline void t2_op_sparse_build_gather_map(t2_ops *ops, CUstream s,
                                                   CUdeviceptr out_map,
                                                   CUdeviceptr coords, int N,
                                                   CUdeviceptr hash_keys,
                                                   CUdeviceptr hash_vals,
                                                   int hash_cap) {
    void *args[] = {&out_map, &coords, &N, &hash_keys, &hash_vals, &hash_cap};
    cuLaunchKernel(ops->sparse_build_gather_map, (unsigned)((N+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

/* Gather features for kernel position k_idx.
 * gathered: [N, C], feats: [N, C], gather_map: [N, 27] */
static inline void t2_op_sparse_gather(t2_ops *ops, CUstream s,
                                         CUdeviceptr gathered,
                                         CUdeviceptr feats,
                                         CUdeviceptr gather_map,
                                         int N, int C, int k_idx) {
    int total = N * C;
    void *args[] = {&gathered, &feats, &gather_map, &N, &C, &k_idx};
    cuLaunchKernel(ops->sparse_gather, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

/* Build packed sparse-conv row lists from an existing gather map.
 * src_all/dst_all: [27, N] int32, counts: [27] int32 */
static inline void t2_op_sparse_pack_from_gather_map(t2_ops *ops, CUstream s,
                                                       CUdeviceptr src_all,
                                                       CUdeviceptr dst_all,
                                                       CUdeviceptr counts,
                                                       CUdeviceptr gather_map,
                                                       int N) {
    int total = N * 27;
    void *args[] = {&src_all, &dst_all, &counts, &gather_map, &N};
    cuLaunchKernel(ops->sparse_pack_from_gather_map,
                   (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

/* Broadcast bias: dst[i*C + c] = bias[c] */
static inline void t2_op_broadcast_bias(t2_ops *ops, CUstream s,
                                          CUdeviceptr dst, CUdeviceptr bias,
                                          int N, int C) {
    int total = N * C;
    void *args[] = {&dst, &bias, &N, &C};
    cuLaunchKernel(ops->broadcast_bias, (unsigned)((total+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

/* Scatter add: dst += src */
static inline void t2_op_scatter_add(t2_ops *ops, CUstream s,
                                       CUdeviceptr dst, CUdeviceptr src, int n) {
    void *args[] = {&dst, &src, &n};
    cuLaunchKernel(ops->scatter_add, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

/* Residual add: dst += src */
static inline void t2_op_residual_add(t2_ops *ops, CUstream s,
                                        CUdeviceptr dst, CUdeviceptr src, int n) {
    void *args[] = {&dst, &src, &n};
    cuLaunchKernel(ops->residual_add, (unsigned)((n+255)/256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_c2s_gather(t2_ops *ops, CUstream s,
                                      CUdeviceptr h_fine,
                                      CUdeviceptr x_fine,
                                      CUdeviceptr h_coarse,
                                      CUdeviceptr x_coarse,
                                      CUdeviceptr idx,
                                      CUdeviceptr subidx,
                                      int N_fine,
                                      int C_out,
                                      int C_in8) {
    int mx = C_out > C_in8 ? C_out : C_in8;
    void *args[] = {&h_fine, &x_fine, &h_coarse, &x_coarse,
                    &idx, &subidx, &C_out, &C_in8};
    cuLaunchKernel(ops->c2s_gather, (unsigned)N_fine,
                   (unsigned)((mx + 255) / 256), 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_c2s_residual_repeat(t2_ops *ops, CUstream s,
                                               CUdeviceptr h,
                                               CUdeviceptr x,
                                               int N, int C_out, int C_in8) {
    void *args[] = {&h, &x, &N, &C_out, &C_in8};
    cuLaunchKernel(ops->c2s_residual_repeat, (unsigned)N, 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_sparse_pack_rows(t2_ops *ops, CUstream s,
                                            CUdeviceptr packed,
                                            CUdeviceptr feats,
                                            CUdeviceptr src_idx,
                                            int M, int C) {
    int total = M * C;
    void *args[] = {&packed, &feats, &src_idx, &M, &C};
    cuLaunchKernel(ops->sparse_pack_rows, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_scatter_add_rows(t2_ops *ops, CUstream s,
                                            CUdeviceptr dst,
                                            CUdeviceptr src,
                                            CUdeviceptr dst_idx,
                                            int M, int C) {
    int total = M * C;
    void *args[] = {&dst, &src, &dst_idx, &M, &C};
    cuLaunchKernel(ops->scatter_add_rows, (unsigned)((total + 255) / 256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

static inline void t2_op_sparse_conv3d_packed(t2_ops *ops, CUstream s,
                                                CUdeviceptr dst,
                                                CUdeviceptr feats,
                                                CUdeviceptr weight,
                                                CUdeviceptr bias,
                                                const t2_sparse_conv_pack *pack,
                                                CUdeviceptr scratch,
                                                CUdeviceptr scratch2,
                                                int in_C, int out_C,
                                                int N_total,
                                                int use_lt) {
    t2_op_broadcast_bias(ops, s, dst, bias, N_total, out_C);
    for (int k = 0; k < 27; k++) {
        int M = pack->M[k];
        if (M <= 0) continue;
        CUdeviceptr weight_k = weight + (size_t)k * out_C * in_C * sizeof(float);
        t2_op_sparse_pack_rows(ops, s, scratch, feats, pack->src_idx[k], M, in_C);
        if (!use_lt ||
            t2_op_gemm_f32_lt(ops, s, scratch2, weight_k, scratch,
                              out_C, in_C, M) != 0) {
            t2_op_gemm(ops, s, scratch2, weight_k, scratch, 0, out_C, in_C, M);
        }
        t2_op_scatter_add_rows(ops, s, dst, scratch2, pack->dst_idx[k], M, out_C);
    }
}

static inline void t2_op_sparse_conv3d_direct(t2_ops *ops, CUstream s,
                                                CUdeviceptr dst,
                                                CUdeviceptr feats,
                                                CUdeviceptr weight,
                                                CUdeviceptr bias,
                                                CUdeviceptr gather_map,
                                                int N, int in_C, int out_C,
                                                int mode) {
    void *args[] = {&dst, &feats, &weight, &bias, &gather_map,
                    &N, &in_C, &out_C, &mode};
    cuLaunchKernel(ops->sparse_conv_direct,
                   (unsigned)(((size_t)N * out_C + 255) / 256), 1, 1,
                   256, 1, 1, 0, s, args, NULL);
}

/* ---- Sparse Conv3D using gather-GEMM pattern ----
 * For each of 27 kernel positions:
 *   1. Gather neighbor features → dense [N, in_C]
 *   2. GEMM: weight_k[out_C, in_C] × gathered[N, in_C]^T → partial[N, out_C]
 *   3. Accumulate partial into output
 *
 * dst: [N, out_C], feats: [N, in_C], weight: [out_C, 27, in_C] (F16 or F32)
 * gather_map: [N, 27] precomputed neighbor indices
 * scratch: temp buffer >= N*max(in_C, out_C)*sizeof(float)
 * scratch2: temp buffer >= N*out_C*sizeof(float)
 */
static inline void t2_op_sparse_conv3d(t2_ops *ops, CUstream s,
                                         CUdeviceptr dst,
                                         CUdeviceptr feats,
                                         CUdeviceptr weight,
                                         CUdeviceptr bias,
                                         CUdeviceptr gather_map,
                                         CUdeviceptr scratch,   /* [N, in_C] gathered features */
                                         CUdeviceptr scratch2,  /* [N, out_C] partial GEMM result */
                                         int N, int in_C, int out_C,
                                         int use_lt) {
    /* Initialize output with bias */
    t2_op_broadcast_bias(ops, s, dst, bias, N, out_C);

    /* For each kernel position, gather + GEMM + accumulate */
    for (int k = 0; k < 27; k++) {
        /* Gather neighbor features for position k */
        t2_op_sparse_gather(ops, s, scratch, feats, gather_map, N, in_C, k);

        /* GEMM: scratch2[N, out_C] = weight_k[out_C, in_C] @ scratch[N, in_C]^T
         * weight_k starts at weight + k * out_C * in_C (but layout is [out_C, 27, in_C])
         * So for kernel position k, weight for output o is at: weight[(o*27 + k) * in_C]
         * This is NOT contiguous in out_C — we need weight transposed to [27, out_C, in_C].
         *
         * Actually the CPU layout is [out_C, 27, in_C]. For GEMM we need [out_C, in_C]
         * for each k. The stride between consecutive out_C for same k is 27*in_C.
         * This requires a strided GEMM or weight reshuffling.
         *
         * Simpler: assume weight is pre-transposed to [27, out_C, in_C] at load time.
         * Then weight_k = weight + k * out_C * in_C.
         */
        CUdeviceptr weight_k = weight + (size_t)k * out_C * in_C * sizeof(float);
        /* Note: if weights are F16, the stride is half */
        if (!ops->use_f32_gemm) {
            weight_k = weight + (size_t)k * out_C * in_C * sizeof(unsigned short);
        }

        /* GEMM: scratch2 = weight_k @ scratch^T + 0 (no bias, added once) */
        CUdeviceptr null_bias = 0;
        if (!use_lt ||
            t2_op_gemm_f32_lt(ops, s, scratch2, weight_k, scratch,
                              out_C, in_C, N) != 0) {
            t2_op_gemm(ops, s, scratch2, weight_k, scratch, null_bias,
                       out_C, in_C, N);
        }

        /* Accumulate: dst += scratch2 */
        t2_op_scatter_add(ops, s, dst, scratch2, N * out_C);
    }
}

#endif /* CUDA_TRELLIS2_OPS_H */
