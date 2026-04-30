/* SS Flow DiT GPU forward driver — Phase 2c.11.
 *
 * Single-header. Composes the kernels validated in Phase 2c.0–2c.9
 * into a reusable per-block forward call. Mirrors the body of the
 * CPU `ssdit_block_forward` in common/sam3d_ss_flow_dit.h, but
 * operates entirely on device pointers from `hip_sam3d_ssdit_gpu.h`.
 *
 * Define HIP_SAM3D_SSDIT_FORWARD_IMPLEMENTATION in exactly one TU.
 *
 * Out-of-scope here (handled at a higher driver level later):
 *   - Timestep / shortcut embedding (`ssdit_time_mlp` analog).
 *   - adaLN_modulation gemm (silu(t_emb) → 6*D split into mod6).
 *   - latent_mapping input/output projections.
 *   - 24-block outer loop.
 *
 * Caller responsibilities:
 *   - Allocate scratch via cs3d_ssdit_block_ws_alloc with the largest
 *     N_s/N_p/N_c that will be passed.
 *   - Provide `mod6` (device ptr, [6*dim] floats — already produced
 *     externally as silu(t_emb) @ adaln_w + adaln_b).
 *   - x_shape/x_pose are updated **in-place** (residual streams).
 */

#ifndef HIP_SAM3D_SSDIT_FORWARD_H_
#define HIP_SAM3D_SSDIT_FORWARD_H_

#include <stddef.h>
#include "../rocew.h"
#include "hip_sam3d_ssdit_gpu.h"  /* cs3d_ssdit_block_w, cs3d_ssdit_block_stream_w */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    hipFunction_t gemm;        /* gemm_f32_bias */
    hipFunction_t mod_ln;      /* modulated_ln_f32 */
    hipFunction_t ln;          /* layernorm_token_f32 */
    hipFunction_t qkv_split;   /* qkv_split_f32 */
    hipFunction_t kv_split;    /* kv_split_f32 */
    hipFunction_t mhrms;       /* multi_head_rmsnorm_f32 */
    hipFunction_t sdpa;        /* sdpa_f32 */
    hipFunction_t gelu;        /* gelu_tanh_inplace_f32 */
    hipFunction_t gated;       /* gated_residual_add_f32 */
    hipFunction_t resadd;      /* residual_add_f32 */
} cs3d_ssdit_fns;

int cs3d_ssdit_fns_lookup(cs3d_ssdit_fns *fns, hipModule_t mod);

/* Per-block scratch workspace. Sized to handle the maximum N_s/N_p/N_c
 * passed at allocation time. Reusable across blocks. */
typedef struct {
    hipDeviceptr_t h_s, h_p;                  /* [N_s, D], [N_p, D]  — norm output */
    hipDeviceptr_t qkvs, qkvp;                /* [N_s, 3D], [N_p, 3D] */
    hipDeviceptr_t q_s, k_s, v_s;             /* [N_s, D]  per */
    hipDeviceptr_t q_p, k_p, v_p;             /* [N_p, D]  per */
    hipDeviceptr_t kpkv, vpkv;                /* [N_p+N_s, D]  pose attends to concat([pose;shape]) */
    hipDeviceptr_t sdpa_s, sdpa_p;            /* [N_s, D], [N_p, D] */
    hipDeviceptr_t t_s, t_p;                  /* [N_s, D], [N_p, D]  — sub-block accumulator */
    hipDeviceptr_t kvs_x, kvp_x;              /* [N_c, 2D] each */
    hipDeviceptr_t Ks, Vs, Kp, Vp;            /* [N_c, D] each */
    hipDeviceptr_t qx_s, qx_p;                /* [N_s, D], [N_p, D] */
    hipDeviceptr_t o_s, o_p;                  /* [N_s, D], [N_p, D] */
    hipDeviceptr_t m1s, m1p;                  /* [N_s, mlp_h], [N_p, mlp_h] */
    int N_s_max, N_p_max, N_c_max;
    int dim, mlp_h;
    size_t total_bytes;
} cs3d_ssdit_block_ws;

int  cs3d_ssdit_block_ws_alloc(cs3d_ssdit_block_ws *ws,
                               int N_s_max, int N_p_max, int N_c_max,
                               int dim, int mlp_h);
void cs3d_ssdit_block_ws_free (cs3d_ssdit_block_ws *ws);

/* Run one MOT block end-to-end. Updates d_xs/d_xp in-place.
 *   d_mod6 = [shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp]
 *            (6 contiguous [dim]-sized vectors).
 *   block points at one entry of cs3d_ssdit_gpu::blocks. */
int  cs3d_ssdit_block_forward(const cs3d_ssdit_fns *fns,
                              cs3d_ssdit_block_ws  *ws,
                              const cs3d_ssdit_block_w *block,
                              hipDeviceptr_t d_mod6,
                              hipDeviceptr_t d_xs, int N_s,
                              hipDeviceptr_t d_xp, int N_p,
                              hipDeviceptr_t d_cond, int N_c,
                              int dim, int n_heads, int head_dim, int mlp_h,
                              float ln_eps);

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_SSDIT_FORWARD_H_ */

/* ============================ implementation ============================ */
#ifdef HIP_SAM3D_SSDIT_FORWARD_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cs3d_ssdit_fns_lookup(cs3d_ssdit_fns *f, hipModule_t mod)
{
    if (hipModuleGetFunction(&f->gemm,      mod, "gemm_f32_bias")          != hipSuccess) return -1;
    if (hipModuleGetFunction(&f->mod_ln,    mod, "modulated_ln_f32")       != hipSuccess) return -1;
    if (hipModuleGetFunction(&f->ln,        mod, "layernorm_token_f32")    != hipSuccess) return -1;
    if (hipModuleGetFunction(&f->qkv_split, mod, "qkv_split_f32")          != hipSuccess) return -1;
    if (hipModuleGetFunction(&f->kv_split,  mod, "kv_split_f32")           != hipSuccess) return -1;
    if (hipModuleGetFunction(&f->mhrms,     mod, "multi_head_rmsnorm_f32") != hipSuccess) return -1;
    if (hipModuleGetFunction(&f->sdpa,      mod, "sdpa_f32")               != hipSuccess) return -1;
    if (hipModuleGetFunction(&f->gelu,      mod, "gelu_tanh_inplace_f32")  != hipSuccess) return -1;
    if (hipModuleGetFunction(&f->gated,     mod, "gated_residual_add_f32") != hipSuccess) return -1;
    if (hipModuleGetFunction(&f->resadd,    mod, "residual_add_f32")       != hipSuccess) return -1;
    return 0;
}

static int cs3d_ssdit_alloc_(hipDeviceptr_t *out, size_t bytes, size_t *tot) {
    if (hipMalloc(out, bytes) != hipSuccess) return -1;
    *tot += bytes;
    return 0;
}

int cs3d_ssdit_block_ws_alloc(cs3d_ssdit_block_ws *ws,
                              int N_s_max, int N_p_max, int N_c_max,
                              int dim, int mlp_h)
{
    if (!ws) return -1;
    memset(ws, 0, sizeof(*ws));
    ws->N_s_max = N_s_max; ws->N_p_max = N_p_max; ws->N_c_max = N_c_max;
    ws->dim = dim; ws->mlp_h = mlp_h;
    int N_kv_max = N_p_max + N_s_max;
    int D2 = 2 * dim, D3 = 3 * dim;
    size_t f = sizeof(float);
    size_t tot = 0;
#define A(field, n) if (cs3d_ssdit_alloc_(&ws->field, (size_t)(n) * f, &tot) < 0) goto fail
    A(h_s, (size_t)N_s_max * dim);
    A(h_p, (size_t)N_p_max * dim);
    A(qkvs, (size_t)N_s_max * D3);
    A(qkvp, (size_t)N_p_max * D3);
    A(q_s,  (size_t)N_s_max * dim); A(k_s,  (size_t)N_s_max * dim); A(v_s, (size_t)N_s_max * dim);
    A(q_p,  (size_t)N_p_max * dim); A(k_p,  (size_t)N_p_max * dim); A(v_p, (size_t)N_p_max * dim);
    A(kpkv, (size_t)N_kv_max * dim); A(vpkv, (size_t)N_kv_max * dim);
    A(sdpa_s, (size_t)N_s_max * dim); A(sdpa_p, (size_t)N_p_max * dim);
    A(t_s,  (size_t)N_s_max * dim); A(t_p, (size_t)N_p_max * dim);
    A(kvs_x, (size_t)N_c_max * D2); A(kvp_x, (size_t)N_c_max * D2);
    A(Ks, (size_t)N_c_max * dim); A(Vs, (size_t)N_c_max * dim);
    A(Kp, (size_t)N_c_max * dim); A(Vp, (size_t)N_c_max * dim);
    A(qx_s, (size_t)N_s_max * dim); A(qx_p, (size_t)N_p_max * dim);
    A(o_s,  (size_t)N_s_max * dim); A(o_p, (size_t)N_p_max * dim);
    A(m1s,  (size_t)N_s_max * mlp_h);
    A(m1p,  (size_t)N_p_max * mlp_h);
#undef A
    ws->total_bytes = tot;
    return 0;
fail:
    cs3d_ssdit_block_ws_free(ws);
    return -1;
}

void cs3d_ssdit_block_ws_free(cs3d_ssdit_block_ws *ws)
{
    if (!ws) return;
    hipDeviceptr_t *all[] = {
        &ws->h_s, &ws->h_p, &ws->qkvs, &ws->qkvp,
        &ws->q_s, &ws->k_s, &ws->v_s, &ws->q_p, &ws->k_p, &ws->v_p,
        &ws->kpkv, &ws->vpkv, &ws->sdpa_s, &ws->sdpa_p,
        &ws->t_s, &ws->t_p, &ws->kvs_x, &ws->kvp_x,
        &ws->Ks, &ws->Vs, &ws->Kp, &ws->Vp,
        &ws->qx_s, &ws->qx_p, &ws->o_s, &ws->o_p,
        &ws->m1s, &ws->m1p,
    };
    for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); i++) {
        if (*all[i]) { hipFree(*all[i]); *all[i] = 0; }
    }
    memset(ws, 0, sizeof(*ws));
}

/* ─── kernel launchers ─── */

static int sa_qkv_path(const cs3d_ssdit_fns *f,
                       const cs3d_ssdit_block_stream_w *sw,
                       hipDeviceptr_t d_x, int N, int dim, int H, int D_h,
                       hipDeviceptr_t d_qkv,
                       hipDeviceptr_t d_q, hipDeviceptr_t d_k, hipDeviceptr_t d_v)
{
    int D3 = 3 * dim;
    {
        unsigned gx = (N + 15) / 16, gy = (D3 + 15) / 16;
        void *args[] = { &d_qkv, &d_x, (void *)&sw->sa_qkv_w, (void *)&sw->sa_qkv_b,
                         &N, &dim, &D3 };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned grid = (unsigned)((N * dim + 255) / 256);
        void *args[] = { &d_q, &d_k, &d_v, &d_qkv, &N, &dim };
        if (hipModuleLaunchKernel(f->qkv_split, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    unsigned threads = 64;
    size_t smem = threads * sizeof(float);
    int stride = dim;
    {
        void *args[] = { &d_q, (void *)&sw->sa_q_rms_gamma, &N, &H, &D_h, &stride };
        if (hipModuleLaunchKernel(f->mhrms, (unsigned)H, (unsigned)N, 1, threads, 1, 1,
                           (unsigned)smem, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        void *args[] = { &d_k, (void *)&sw->sa_k_rms_gamma, &N, &H, &D_h, &stride };
        if (hipModuleLaunchKernel(f->mhrms, (unsigned)H, (unsigned)N, 1, threads, 1, 1,
                           (unsigned)smem, 0, args, NULL) != hipSuccess) return -1;
    }
    return 0;
}

int cs3d_ssdit_block_forward(const cs3d_ssdit_fns *f,
                             cs3d_ssdit_block_ws  *ws,
                             const cs3d_ssdit_block_w *block,
                             hipDeviceptr_t d_mod6,
                             hipDeviceptr_t d_xs, int N_s,
                             hipDeviceptr_t d_xp, int N_p,
                             hipDeviceptr_t d_cond, int N_c,
                             int dim, int H, int D_h, int mlp_h,
                             float eps)
{
    const cs3d_ssdit_block_stream_w *swsh = &block->stream[SAM3D_SS_STREAM_SHAPE];
    const cs3d_ssdit_block_stream_w *swp  = &block->stream[SAM3D_SS_STREAM_POSE];

    /* mod6 = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]. */
    hipDeviceptr_t d_smsa = d_mod6 + (size_t)0 * dim * sizeof(float);
    hipDeviceptr_t d_cmsa = d_mod6 + (size_t)1 * dim * sizeof(float);
    hipDeviceptr_t d_gmsa = d_mod6 + (size_t)2 * dim * sizeof(float);
    hipDeviceptr_t d_smlp = d_mod6 + (size_t)3 * dim * sizeof(float);
    hipDeviceptr_t d_cmlp = d_mod6 + (size_t)4 * dim * sizeof(float);
    hipDeviceptr_t d_gmlp = d_mod6 + (size_t)5 * dim * sizeof(float);

    float scale = 1.0f / sqrtf((float)D_h);
    int D2 = 2 * dim;
    int N_kv = N_p + N_s;
    int affine_yes = 1;

    /* === norm1 + adaLN_msa === */
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *as[] = { &ws->h_s, &d_xs, &d_smsa, &d_cmsa, &N_s, &dim, &eps };
        if (hipModuleLaunchKernel(f->mod_ln, (unsigned)N_s, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, as, NULL) != hipSuccess) return -1;
        void *ap[] = { &ws->h_p, &d_xp, &d_smsa, &d_cmsa, &N_p, &dim, &eps };
        if (hipModuleLaunchKernel(f->mod_ln, (unsigned)N_p, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, ap, NULL) != hipSuccess) return -1;
    }

    /* === MOT self-attn === */
    if (sa_qkv_path(f, swsh, ws->h_s, N_s, dim, H, D_h, ws->qkvs, ws->q_s, ws->k_s, ws->v_s) < 0) return -1;
    if (sa_qkv_path(f, swp,  ws->h_p, N_p, dim, H, D_h, ws->qkvp, ws->q_p, ws->k_p, ws->v_p) < 0) return -1;
    /* shape attends self only. */
    {
        unsigned threads = 256; size_t sm = (threads + (size_t)N_s) * sizeof(float);
        void *args[] = { &ws->sdpa_s, &ws->q_s, &ws->k_s, &ws->v_s, &N_s, &N_s, &H, &D_h, &scale };
        if (hipModuleLaunchKernel(f->sdpa, (unsigned)N_s, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, args, NULL) != hipSuccess) return -1;
    }
    /* pose KV concat = [pose; shape]. */
    hipMemcpyDtoD(ws->kpkv,                                       ws->k_p, (size_t)N_p * dim * sizeof(float));
    hipMemcpyDtoD(ws->kpkv + (size_t)N_p * dim * sizeof(float),   ws->k_s, (size_t)N_s * dim * sizeof(float));
    hipMemcpyDtoD(ws->vpkv,                                       ws->v_p, (size_t)N_p * dim * sizeof(float));
    hipMemcpyDtoD(ws->vpkv + (size_t)N_p * dim * sizeof(float),   ws->v_s, (size_t)N_s * dim * sizeof(float));
    {
        unsigned threads = 256; size_t sm = (threads + (size_t)N_kv) * sizeof(float);
        void *args[] = { &ws->sdpa_p, &ws->q_p, &ws->kpkv, &ws->vpkv, &N_p, &N_kv, &H, &D_h, &scale };
        if (hipModuleLaunchKernel(f->sdpa, (unsigned)N_p, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, args, NULL) != hipSuccess) return -1;
    }
    /* sa_out per stream. */
    {
        unsigned gx = (N_s + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &ws->t_s, &ws->sdpa_s, (void *)&swsh->sa_out_w, (void *)&swsh->sa_out_b,
                         &N_s, &dim, &dim };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &ws->t_p, &ws->sdpa_p, (void *)&swp->sa_out_w, (void *)&swp->sa_out_b,
                         &N_p, &dim, &dim };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    /* gated residual_msa */
    {
        unsigned grid = (unsigned)((N_s * dim + 255) / 256);
        void *args[] = { &d_xs, &ws->t_s, &d_gmsa, &N_s, &dim };
        if (hipModuleLaunchKernel(f->gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned grid = (unsigned)((N_p * dim + 255) / 256);
        void *args[] = { &d_xp, &ws->t_p, &d_gmsa, &N_p, &dim };
        if (hipModuleLaunchKernel(f->gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }

    /* === norm2 (affine) === */
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *as[] = { &ws->h_s, &d_xs, (void *)&swsh->norm2_w, (void *)&swsh->norm2_b,
                       &N_s, &dim, &eps, &affine_yes };
        if (hipModuleLaunchKernel(f->ln, (unsigned)N_s, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, as, NULL) != hipSuccess) return -1;
        void *ap[] = { &ws->h_p, &d_xp, (void *)&swp->norm2_w, (void *)&swp->norm2_b,
                       &N_p, &dim, &eps, &affine_yes };
        if (hipModuleLaunchKernel(f->ln, (unsigned)N_p, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, ap, NULL) != hipSuccess) return -1;
    }
    /* === cross-attn === */
    {
        unsigned gx = (N_s + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &ws->qx_s, &ws->h_s, (void *)&swsh->xa_q_w, (void *)&swsh->xa_q_b,
                         &N_s, &dim, &dim };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &ws->qx_p, &ws->h_p, (void *)&swp->xa_q_w, (void *)&swp->xa_q_b,
                         &N_p, &dim, &dim };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned gx = (N_c + 15) / 16, gy = (D2 + 15) / 16;
        void *args[] = { &ws->kvs_x, &d_cond, (void *)&swsh->xa_kv_w, (void *)&swsh->xa_kv_b,
                         &N_c, &dim, &D2 };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
        void *ap[] = { &ws->kvp_x, &d_cond, (void *)&swp->xa_kv_w, (void *)&swp->xa_kv_b,
                       &N_c, &dim, &D2 };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, ap, NULL) != hipSuccess) return -1;
    }
    {
        unsigned grid = (unsigned)((N_c * dim + 255) / 256);
        void *as[] = { &ws->Ks, &ws->Vs, &ws->kvs_x, &N_c, &dim };
        if (hipModuleLaunchKernel(f->kv_split, grid, 1, 1, 256, 1, 1, 0, 0, as, NULL) != hipSuccess) return -1;
        void *ap[] = { &ws->Kp, &ws->Vp, &ws->kvp_x, &N_c, &dim };
        if (hipModuleLaunchKernel(f->kv_split, grid, 1, 1, 256, 1, 1, 0, 0, ap, NULL) != hipSuccess) return -1;
    }
    {
        unsigned threads = 256; size_t sm = (threads + (size_t)N_c) * sizeof(float);
        void *as[] = { &ws->o_s, &ws->qx_s, &ws->Ks, &ws->Vs, &N_s, &N_c, &H, &D_h, &scale };
        if (hipModuleLaunchKernel(f->sdpa, (unsigned)N_s, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, as, NULL) != hipSuccess) return -1;
        void *ap[] = { &ws->o_p, &ws->qx_p, &ws->Kp, &ws->Vp, &N_p, &N_c, &H, &D_h, &scale };
        if (hipModuleLaunchKernel(f->sdpa, (unsigned)N_p, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, ap, NULL) != hipSuccess) return -1;
    }
    {
        unsigned gx = (N_s + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &ws->t_s, &ws->o_s, (void *)&swsh->xa_out_w, (void *)&swsh->xa_out_b,
                         &N_s, &dim, &dim };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &ws->t_p, &ws->o_p, (void *)&swp->xa_out_w, (void *)&swp->xa_out_b,
                         &N_p, &dim, &dim };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int total = N_s * dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_xs, &ws->t_s, &total };
        if (hipModuleLaunchKernel(f->resadd, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int total = N_p * dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_xp, &ws->t_p, &total };
        if (hipModuleLaunchKernel(f->resadd, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }

    /* === norm3 + adaLN_mlp + FFN + gated_mlp === */
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *as[] = { &ws->h_s, &d_xs, &d_smlp, &d_cmlp, &N_s, &dim, &eps };
        if (hipModuleLaunchKernel(f->mod_ln, (unsigned)N_s, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, as, NULL) != hipSuccess) return -1;
        void *ap[] = { &ws->h_p, &d_xp, &d_smlp, &d_cmlp, &N_p, &dim, &eps };
        if (hipModuleLaunchKernel(f->mod_ln, (unsigned)N_p, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, ap, NULL) != hipSuccess) return -1;
    }
    {
        unsigned gx = (N_s + 15) / 16, gy = (mlp_h + 15) / 16;
        void *args[] = { &ws->m1s, &ws->h_s, (void *)&swsh->mlp_fc1_w, (void *)&swsh->mlp_fc1_b,
                         &N_s, &dim, &mlp_h };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (mlp_h + 15) / 16;
        void *args[] = { &ws->m1p, &ws->h_p, (void *)&swp->mlp_fc1_w, (void *)&swp->mlp_fc1_b,
                         &N_p, &dim, &mlp_h };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int total = N_s * mlp_h;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &ws->m1s, &total };
        if (hipModuleLaunchKernel(f->gelu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int total = N_p * mlp_h;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &ws->m1p, &total };
        if (hipModuleLaunchKernel(f->gelu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned gx = (N_s + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &ws->t_s, &ws->m1s, (void *)&swsh->mlp_fc2_w, (void *)&swsh->mlp_fc2_b,
                         &N_s, &mlp_h, &dim };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned gx = (N_p + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &ws->t_p, &ws->m1p, (void *)&swp->mlp_fc2_w, (void *)&swp->mlp_fc2_b,
                         &N_p, &mlp_h, &dim };
        if (hipModuleLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned grid = (unsigned)((N_s * dim + 255) / 256);
        void *args[] = { &d_xs, &ws->t_s, &d_gmlp, &N_s, &dim };
        if (hipModuleLaunchKernel(f->gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned grid = (unsigned)((N_p * dim + 255) / 256);
        void *args[] = { &d_xp, &ws->t_p, &d_gmlp, &N_p, &dim };
        if (hipModuleLaunchKernel(f->gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    return 0;
}

#endif /* HIP_SAM3D_SSDIT_FORWARD_IMPLEMENTATION */
