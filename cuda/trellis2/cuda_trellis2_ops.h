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

    int sm_version;
    int use_f32_gemm;
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
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    CUfunction fn = ops->use_f32_gemm ? ops->gemm_f32 : ops->gemm_tiled;
    cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, s, args, NULL);
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
    int nt = 128;
    size_t smem = (size_t)(kv_len + nt) * sizeof(float);
    void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim,
                    &n_heads, &head_dim, &scale};
    cuLaunchKernel(ops->cross_attn, (unsigned)n_heads, (unsigned)q_len, 1,
                   (unsigned)nt, 1, 1, smem, s, args, NULL);
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

#endif /* CUDA_TRELLIS2_OPS_H */
