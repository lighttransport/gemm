/*
 * cuda_trellis2_ops.h - Kernel launch wrappers for TRELLIS.2 CUDA runner
 *
 * Requires cuew.h included before this header.
 */
#ifndef CUDA_TRELLIS2_OPS_H
#define CUDA_TRELLIS2_OPS_H

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef __CUEW_H__
#error "cuda_trellis2_ops.h requires cuew.h to be included first"
#endif

typedef struct {
    /* Common kernels */
    CUfunction layernorm;
    CUfunction gemm_tiled;
    CUfunction gemm_f32;
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

    /* Sparse conv kernels */
    CUfunction sparse_build_gather_map;
    CUfunction sparse_gather;
    CUfunction scatter_add;
    CUfunction broadcast_bias;
    CUfunction residual_add;

    int sm_version;
    int use_f32_gemm;   /* 0=F16 weights, 1=F32 weights */
    int use_mma_gemm;   /* 1=use MMA tensor core GEMM (sm_70+, F16 weights) */
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
    GET_FN("gemm_tiled_f16_f32",      gemm_tiled);
    GET_FN("gemm_f32_f32",            gemm_f32);
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
    }

    /* Sparse conv kernels */
    GET_FN("sparse_build_gather_map_f32", sparse_build_gather_map);
    GET_FN("sparse_gather_f32",           sparse_gather);
    GET_FN("scatter_add_f32",             scatter_add);
    GET_FN("broadcast_bias_f32",          broadcast_bias);
    GET_FN("residual_add_f32",            residual_add);

    #undef GET_FN
    return 0;
}

/* ======================================================================== */
/* Launch wrappers                                                          */
/* ======================================================================== */

static inline void t2_op_gemm(t2_ops *ops, CUstream s,
                                CUdeviceptr Y, CUdeviceptr W,
                                CUdeviceptr X, CUdeviceptr bias,
                                int n_out, int n_in, int n_tok) {
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

static inline void t2_op_layernorm(t2_ops *ops, CUstream s,
                                     CUdeviceptr dst, CUdeviceptr src,
                                     CUdeviceptr w, CUdeviceptr b,
                                     int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    cuLaunchKernel(ops->layernorm, (unsigned)n_tok, 1, 1,
                   256, 1, 1, 256 * sizeof(float), s, args, NULL);
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
    cuLaunchKernel(ops->modulation, 1, 1, 1,
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
                                         int N, int in_C, int out_C) {
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
        t2_op_gemm(ops, s, scratch2, weight_k, scratch, null_bias, out_C, in_C, N);

        /* Accumulate: dst += scratch2 */
        t2_op_scatter_add(ops, s, dst, scratch2, N * out_C);
    }
}

#endif /* CUDA_TRELLIS2_OPS_H */
