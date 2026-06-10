/* SS Flow DiT GPU forward driver — Phase 2c.11.
 *
 * Single-header. Composes the kernels validated in Phase 2c.0–2c.9
 * into a reusable per-block forward call. Mirrors the body of the
 * CPU `ssdit_block_forward` in common/sam3d_ss_flow_dit.h, but
 * operates entirely on device pointers from `cuda_sam3d_ssdit_gpu.h`.
 *
 * Define CUDA_SAM3D_SSDIT_FORWARD_IMPLEMENTATION in exactly one TU.
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

#ifndef CUDA_SAM3D_SSDIT_FORWARD_H_
#define CUDA_SAM3D_SSDIT_FORWARD_H_

#include <stddef.h>
#include "../cuew.h"
#include "../cublasew.h"
#include "cuda_sam3d_ssdit_gpu.h"  /* cs3d_ssdit_block_w, cs3d_ssdit_block_stream_w */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    CUfunction gemm;        /* gemm_f32_bias */
    CUfunction mod_ln;      /* modulated_ln_f32 */
    CUfunction ln;          /* layernorm_token_f32 */
    CUfunction qkv_split;   /* qkv_split_f32 */
    CUfunction kv_split;    /* kv_split_f32 */
    CUfunction mhrms;       /* multi_head_rmsnorm_f32 */
    CUfunction sdpa;        /* sdpa_f32 */
    CUfunction flash_attn;  /* flash_attn_sep_hd64_f32 */
    CUfunction cast_f16;    /* cast_f32_to_f16_sam3d */
    CUfunction cast_bf16;   /* cast_f32_to_bf16_sam3d */
    CUfunction add_bias;    /* add_bias_rows_f32 */
    CUfunction gelu;        /* gelu_tanh_inplace_f32 */
    CUfunction gated;       /* gated_residual_add_f32 */
    CUfunction resadd;      /* residual_add_f32 */
    cublasew_context *cublas;
    CUdeviceptr gemm_x16;
    CUdeviceptr gemm_w16;
    size_t gemm_x16_cap;
    size_t gemm_w16_cap;
    int use_cublas_gemm;
    int use_flash_attn;
    int mma_precision;      /* 0=f32, 1=fp16, 2=bf16 */
} cs3d_ssdit_fns;

int cs3d_ssdit_fns_lookup(cs3d_ssdit_fns *fns, CUmodule mod);
void cs3d_ssdit_fns_set_precision(cs3d_ssdit_fns *fns, const char *precision);
void cs3d_ssdit_fns_free(cs3d_ssdit_fns *fns);

/* Per-block scratch workspace. Sized to handle the maximum N_s/N_p/N_c
 * passed at allocation time. Reusable across blocks. */
typedef struct {
    CUdeviceptr h_s, h_p;                  /* [N_s, D], [N_p, D]  — norm output */
    CUdeviceptr qkvs, qkvp;                /* [N_s, 3D], [N_p, 3D] */
    CUdeviceptr q_s, k_s, v_s;             /* [N_s, D]  per */
    CUdeviceptr q_p, k_p, v_p;             /* [N_p, D]  per */
    CUdeviceptr kpkv, vpkv;                /* [N_p+N_s, D]  pose attends to concat([pose;shape]) */
    CUdeviceptr sdpa_s, sdpa_p;            /* [N_s, D], [N_p, D] */
    CUdeviceptr t_s, t_p;                  /* [N_s, D], [N_p, D]  — sub-block accumulator */
    CUdeviceptr kvs_x, kvp_x;              /* [N_c, 2D] each */
    CUdeviceptr Ks, Vs, Kp, Vp;            /* [N_c, D] each */
    CUdeviceptr qx_s, qx_p;                /* [N_s, D], [N_p, D] */
    CUdeviceptr o_s, o_p;                  /* [N_s, D], [N_p, D] */
    CUdeviceptr m1s, m1p;                  /* [N_s, mlp_h], [N_p, mlp_h] */
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
                              CUdeviceptr d_mod6,
                              CUdeviceptr d_xs, int N_s,
                              CUdeviceptr d_xp, int N_p,
                              CUdeviceptr d_cond, int N_c,
                              int dim, int n_heads, int head_dim, int mlp_h,
                              float ln_eps);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SAM3D_SSDIT_FORWARD_H_ */

/* ============================ implementation ============================ */
#ifdef CUDA_SAM3D_SSDIT_FORWARD_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cs3d_ssdit_fns_lookup(cs3d_ssdit_fns *f, CUmodule mod)
{
    memset(f, 0, sizeof(*f));
    if (cuModuleGetFunction(&f->gemm,      mod, "gemm_f32_bias")          != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->mod_ln,    mod, "modulated_ln_f32")       != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->ln,        mod, "layernorm_token_f32")    != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->qkv_split, mod, "qkv_split_f32")          != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->kv_split,  mod, "kv_split_f32")           != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->mhrms,     mod, "multi_head_rmsnorm_f32") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->sdpa,      mod, "sdpa_f32")               != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->flash_attn, mod, "flash_attn_sep_hd64_f32") == CUDA_SUCCESS) {
        const char *fa_env = getenv("SAM3D_SSDIT_FLASH_ATTN");
        f->use_flash_attn = (!fa_env || fa_env[0] != '0');
    }
    if (cuModuleGetFunction(&f->cast_f16,  mod, "cast_f32_to_f16_sam3d")  != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->cast_bf16, mod, "cast_f32_to_bf16_sam3d") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->add_bias,  mod, "add_bias_rows_f32")      != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->gelu,      mod, "gelu_tanh_inplace_f32")  != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->gated,     mod, "gated_residual_add_f32") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->resadd,    mod, "residual_add_f32")       != CUDA_SUCCESS) return -1;
    const char *env = getenv("SAM3D_CUBLAS_GEMM");
    if (!env || env[0] != '0') {
        if (cublasewCreate(&f->cublas, 0) == 0) {
            f->use_cublas_gemm = 1;
        }
    }
    return 0;
}

void cs3d_ssdit_fns_set_precision(cs3d_ssdit_fns *f, const char *precision)
{
    if (!f) return;
    const char *env = getenv("SAM3D_MMA_GEMM");
    if (env && env[0] == '0') {
        f->mma_precision = 0;
        return;
    }
    const char *p = precision && precision[0] ? precision : "fp16";
    if (!strcmp(p, "bf16")) f->mma_precision = 2;
    else if (!strcmp(p, "fp16")) f->mma_precision = 1;
    else f->mma_precision = 0;
}

void cs3d_ssdit_fns_free(cs3d_ssdit_fns *f)
{
    if (!f) return;
    if (f->cublas) cublasewDestroy(f->cublas);
    if (f->gemm_x16) cuMemFree(f->gemm_x16);
    if (f->gemm_w16) cuMemFree(f->gemm_w16);
    f->cublas = NULL;
    f->gemm_x16 = 0;
    f->gemm_w16 = 0;
    f->gemm_x16_cap = 0;
    f->gemm_w16_cap = 0;
    f->use_cublas_gemm = 0;
    f->use_flash_attn = 0;
    f->mma_precision = 0;
}

static int cs3d_ssdit_ensure_u16(CUdeviceptr *ptr, size_t *cap, size_t n_elem)
{
    size_t bytes = n_elem * sizeof(unsigned short);
    if (*ptr && *cap >= bytes) return 0;
    if (*ptr) cuMemFree(*ptr);
    *ptr = 0;
    *cap = 0;
    if (cuMemAlloc(ptr, bytes) != CUDA_SUCCESS) return -1;
    *cap = bytes;
    return 0;
}

static int cs3d_ssdit_cast_u16(const cs3d_ssdit_fns *f,
                               CUdeviceptr d_src, CUdeviceptr d_dst, int n)
{
    CUfunction fn = (f->mma_precision == 2) ? f->cast_bf16 : f->cast_f16;
    void *args[] = { &d_src, &d_dst, &n };
    return (cuLaunchKernel(fn, (unsigned)((n + 255) / 256), 1, 1,
                           256, 1, 1, 0, 0, args, NULL) == CUDA_SUCCESS) ? 0 : -1;
}

static int cs3d_ssdit_add_bias(const cs3d_ssdit_fns *f,
                               CUdeviceptr d_out, CUdeviceptr d_b, int N, int M)
{
    int total = N * M;
    void *args[] = { &d_out, &d_b, &N, &M };
    return (cuLaunchKernel(f->add_bias, (unsigned)((total + 255) / 256), 1, 1,
                           256, 1, 1, 0, 0, args, NULL) == CUDA_SUCCESS) ? 0 : -1;
}

static int cs3d_ssdit_gemm(const cs3d_ssdit_fns *f,
                           CUdeviceptr d_out, CUdeviceptr d_in,
                           CUdeviceptr d_w, CUdeviceptr d_w16, CUdeviceptr d_b,
                           int N, int K, int M)
{
    if (f->use_cublas_gemm && f->cublas && f->mma_precision &&
        (d_w16 || N >= 128) && K >= 64 && M >= 64 && (K % 8) == 0 && (M % 8) == 0) {
        cs3d_ssdit_fns *mf = (cs3d_ssdit_fns *)f;
        size_t xn = (size_t)N * (size_t)K;
        size_t wn = (size_t)M * (size_t)K;
        CUdeviceptr use_w16 = d_w16;
        int have_w16 = use_w16 != 0;
        if (cs3d_ssdit_ensure_u16(&mf->gemm_x16, &mf->gemm_x16_cap, xn) == 0 &&
            (have_w16 || cs3d_ssdit_ensure_u16(&mf->gemm_w16, &mf->gemm_w16_cap, wn) == 0) &&
            cs3d_ssdit_cast_u16(f, d_in, mf->gemm_x16, (int)xn) == 0 &&
            (have_w16 || cs3d_ssdit_cast_u16(f, d_w,  mf->gemm_w16, (int)wn) == 0)) {
            if (!have_w16) use_w16 = mf->gemm_w16;
            int ok = (f->mma_precision == 2)
                ? cublasew_gemm_bf16_bf16_f32_rowmajor_nt(
                      f->cublas, d_out, use_w16, mf->gemm_x16, N, M, K)
                : cublasew_gemm_f16_f16_f32_rowmajor_nt(
                      f->cublas, d_out, use_w16, mf->gemm_x16, N, M, K);
            if (ok == 0) {
                if (d_b && cs3d_ssdit_add_bias(f, d_out, d_b, N, M) < 0) return -1;
                return 0;
            }
        }
    }
    if (!d_w) return -1;
    if (f->use_cublas_gemm && f->cublas) {
        int ok = d_b
            ? cublasew_gemm_f32_lt_bias_rowmajor_nt(f->cublas, d_out, d_w, d_in,
                                                    d_b, N, M, K)
            : cublasew_gemm_f32_lt_rowmajor_nt(f->cublas, d_out, d_w, d_in,
                                               N, M, K);
        if (ok == 0) return 0;
    }
    unsigned gx = (N + 15) / 16, gy = (M + 15) / 16;
    void *args[] = { &d_out, &d_in, &d_w, &d_b, &N, &K, &M };
    return (cuLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0,
                           args, NULL) == CUDA_SUCCESS) ? 0 : -1;
}

static int cs3d_ssdit_attn(const cs3d_ssdit_fns *f,
                           CUdeviceptr d_out, CUdeviceptr d_q,
                           CUdeviceptr d_k, CUdeviceptr d_v,
                           int N_q, int N_k, int H, int D_h, float scale)
{
    if (f->use_flash_attn && f->flash_attn && D_h == 64 && N_q >= 128) {
        unsigned warps = 4;
        unsigned bkv = 32;
        unsigned threads = warps * 32;
        unsigned gx = (unsigned)H;
        unsigned gy = (unsigned)((N_q + (int)warps - 1) / (int)warps);
        unsigned smem = (unsigned)(2 * bkv * D_h * sizeof(float));
        void *args[] = { &d_out, &d_q, &d_k, &d_v, &N_q, &N_k, &H, &D_h, &scale };
        if (cuLaunchKernel(f->flash_attn, gx, gy, 1, threads, 1, 1,
                           smem, 0, args, NULL) == CUDA_SUCCESS) return 0;
    }
    unsigned threads = 256;
    size_t sm = (threads + (size_t)N_k) * sizeof(float);
    void *args[] = { &d_out, &d_q, &d_k, &d_v, &N_q, &N_k, &H, &D_h, &scale };
    return (cuLaunchKernel(f->sdpa, (unsigned)N_q, (unsigned)H, 1,
                           threads, 1, 1, (unsigned)sm, 0, args, NULL) == CUDA_SUCCESS) ? 0 : -1;
}

static int cs3d_ssdit_alloc_(CUdeviceptr *out, size_t bytes, size_t *tot) {
    if (cuMemAlloc(out, bytes) != CUDA_SUCCESS) return -1;
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
    CUdeviceptr *all[] = {
        &ws->h_s, &ws->h_p, &ws->qkvs, &ws->qkvp,
        &ws->q_s, &ws->k_s, &ws->v_s, &ws->q_p, &ws->k_p, &ws->v_p,
        &ws->kpkv, &ws->vpkv, &ws->sdpa_s, &ws->sdpa_p,
        &ws->t_s, &ws->t_p, &ws->kvs_x, &ws->kvp_x,
        &ws->Ks, &ws->Vs, &ws->Kp, &ws->Vp,
        &ws->qx_s, &ws->qx_p, &ws->o_s, &ws->o_p,
        &ws->m1s, &ws->m1p,
    };
    for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); i++) {
        if (*all[i]) { cuMemFree(*all[i]); *all[i] = 0; }
    }
    memset(ws, 0, sizeof(*ws));
}

/* ─── kernel launchers ─── */

static int sa_qkv_path(const cs3d_ssdit_fns *f,
                       const cs3d_ssdit_block_stream_w *sw,
                       CUdeviceptr d_x, int N, int dim, int H, int D_h,
                       CUdeviceptr d_qkv,
                       CUdeviceptr d_q, CUdeviceptr d_k, CUdeviceptr d_v)
{
    int D3 = 3 * dim;
    {
        if (cs3d_ssdit_gemm(f, d_qkv, d_x, sw->sa_qkv_w, sw->sa_qkv_w16, sw->sa_qkv_b,
                       N, dim, D3) < 0) return -1;
    }
    {
        unsigned grid = (unsigned)((N * dim + 255) / 256);
        void *args[] = { &d_q, &d_k, &d_v, &d_qkv, &N, &dim };
        if (cuLaunchKernel(f->qkv_split, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    unsigned threads = 64;
    size_t smem = threads * sizeof(float);
    int stride = dim;
    {
        void *args[] = { &d_q, (void *)&sw->sa_q_rms_gamma, &N, &H, &D_h, &stride };
        if (cuLaunchKernel(f->mhrms, (unsigned)H, (unsigned)N, 1, threads, 1, 1,
                           (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        void *args[] = { &d_k, (void *)&sw->sa_k_rms_gamma, &N, &H, &D_h, &stride };
        if (cuLaunchKernel(f->mhrms, (unsigned)H, (unsigned)N, 1, threads, 1, 1,
                           (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    return 0;
}

int cs3d_ssdit_block_forward(const cs3d_ssdit_fns *f,
                             cs3d_ssdit_block_ws  *ws,
                             const cs3d_ssdit_block_w *block,
                             CUdeviceptr d_mod6,
                             CUdeviceptr d_xs, int N_s,
                             CUdeviceptr d_xp, int N_p,
                             CUdeviceptr d_cond, int N_c,
                             int dim, int H, int D_h, int mlp_h,
                             float eps)
{
    const cs3d_ssdit_block_stream_w *swsh = &block->stream[SAM3D_SS_STREAM_SHAPE];
    const cs3d_ssdit_block_stream_w *swp  = &block->stream[SAM3D_SS_STREAM_POSE];

    /* mod6 = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]. */
    CUdeviceptr d_smsa = d_mod6 + (size_t)0 * dim * sizeof(float);
    CUdeviceptr d_cmsa = d_mod6 + (size_t)1 * dim * sizeof(float);
    CUdeviceptr d_gmsa = d_mod6 + (size_t)2 * dim * sizeof(float);
    CUdeviceptr d_smlp = d_mod6 + (size_t)3 * dim * sizeof(float);
    CUdeviceptr d_cmlp = d_mod6 + (size_t)4 * dim * sizeof(float);
    CUdeviceptr d_gmlp = d_mod6 + (size_t)5 * dim * sizeof(float);

    float scale = 1.0f / sqrtf((float)D_h);
    int D2 = 2 * dim;
    int N_kv = N_p + N_s;
    int affine_yes = 1;

    /* === norm1 + adaLN_msa === */
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *as[] = { &ws->h_s, &d_xs, &d_smsa, &d_cmsa, &N_s, &dim, &eps };
        if (cuLaunchKernel(f->mod_ln, (unsigned)N_s, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, as, NULL) != CUDA_SUCCESS) return -1;
        void *ap[] = { &ws->h_p, &d_xp, &d_smsa, &d_cmsa, &N_p, &dim, &eps };
        if (cuLaunchKernel(f->mod_ln, (unsigned)N_p, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, ap, NULL) != CUDA_SUCCESS) return -1;
    }

    /* === MOT self-attn === */
    if (sa_qkv_path(f, swsh, ws->h_s, N_s, dim, H, D_h, ws->qkvs, ws->q_s, ws->k_s, ws->v_s) < 0) return -1;
    if (sa_qkv_path(f, swp,  ws->h_p, N_p, dim, H, D_h, ws->qkvp, ws->q_p, ws->k_p, ws->v_p) < 0) return -1;
    /* shape attends self only. */
    if (cs3d_ssdit_attn(f, ws->sdpa_s, ws->q_s, ws->k_s, ws->v_s,
                        N_s, N_s, H, D_h, scale) < 0) return -1;
    /* pose KV concat = [pose; shape]. */
    cuMemcpyDtoD(ws->kpkv,                                       ws->k_p, (size_t)N_p * dim * sizeof(float));
    cuMemcpyDtoD(ws->kpkv + (size_t)N_p * dim * sizeof(float),   ws->k_s, (size_t)N_s * dim * sizeof(float));
    cuMemcpyDtoD(ws->vpkv,                                       ws->v_p, (size_t)N_p * dim * sizeof(float));
    cuMemcpyDtoD(ws->vpkv + (size_t)N_p * dim * sizeof(float),   ws->v_s, (size_t)N_s * dim * sizeof(float));
    if (cs3d_ssdit_attn(f, ws->sdpa_p, ws->q_p, ws->kpkv, ws->vpkv,
                        N_p, N_kv, H, D_h, scale) < 0) return -1;
    /* sa_out per stream. */
    {
        if (cs3d_ssdit_gemm(f, ws->t_s, ws->sdpa_s, swsh->sa_out_w, swsh->sa_out_w16,
                       swsh->sa_out_b, N_s, dim, dim) < 0) return -1;
    }
    {
        if (cs3d_ssdit_gemm(f, ws->t_p, ws->sdpa_p, swp->sa_out_w, swp->sa_out_w16,
                       swp->sa_out_b, N_p, dim, dim) < 0) return -1;
    }
    /* gated residual_msa */
    {
        unsigned grid = (unsigned)((N_s * dim + 255) / 256);
        void *args[] = { &d_xs, &ws->t_s, &d_gmsa, &N_s, &dim };
        if (cuLaunchKernel(f->gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        unsigned grid = (unsigned)((N_p * dim + 255) / 256);
        void *args[] = { &d_xp, &ws->t_p, &d_gmsa, &N_p, &dim };
        if (cuLaunchKernel(f->gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }

    /* === norm2 (affine) === */
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *as[] = { &ws->h_s, &d_xs, (void *)&swsh->norm2_w, (void *)&swsh->norm2_b,
                       &N_s, &dim, &eps, &affine_yes };
        if (cuLaunchKernel(f->ln, (unsigned)N_s, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, as, NULL) != CUDA_SUCCESS) return -1;
        void *ap[] = { &ws->h_p, &d_xp, (void *)&swp->norm2_w, (void *)&swp->norm2_b,
                       &N_p, &dim, &eps, &affine_yes };
        if (cuLaunchKernel(f->ln, (unsigned)N_p, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, ap, NULL) != CUDA_SUCCESS) return -1;
    }
    /* === cross-attn === */
    {
        if (cs3d_ssdit_gemm(f, ws->qx_s, ws->h_s, swsh->xa_q_w, swsh->xa_q_w16,
                       swsh->xa_q_b, N_s, dim, dim) < 0) return -1;
    }
    {
        if (cs3d_ssdit_gemm(f, ws->qx_p, ws->h_p, swp->xa_q_w, swp->xa_q_w16,
                       swp->xa_q_b, N_p, dim, dim) < 0) return -1;
    }
    {
        if (cs3d_ssdit_gemm(f, ws->kvs_x, d_cond, swsh->xa_kv_w, swsh->xa_kv_w16,
                       swsh->xa_kv_b, N_c, dim, D2) < 0) return -1;
        if (cs3d_ssdit_gemm(f, ws->kvp_x, d_cond, swp->xa_kv_w, swp->xa_kv_w16,
                       swp->xa_kv_b, N_c, dim, D2) < 0) return -1;
    }
    {
        unsigned grid = (unsigned)((N_c * dim + 255) / 256);
        void *as[] = { &ws->Ks, &ws->Vs, &ws->kvs_x, &N_c, &dim };
        if (cuLaunchKernel(f->kv_split, grid, 1, 1, 256, 1, 1, 0, 0, as, NULL) != CUDA_SUCCESS) return -1;
        void *ap[] = { &ws->Kp, &ws->Vp, &ws->kvp_x, &N_c, &dim };
        if (cuLaunchKernel(f->kv_split, grid, 1, 1, 256, 1, 1, 0, 0, ap, NULL) != CUDA_SUCCESS) return -1;
    }
    if (cs3d_ssdit_attn(f, ws->o_s, ws->qx_s, ws->Ks, ws->Vs,
                        N_s, N_c, H, D_h, scale) < 0) return -1;
    if (cs3d_ssdit_attn(f, ws->o_p, ws->qx_p, ws->Kp, ws->Vp,
                        N_p, N_c, H, D_h, scale) < 0) return -1;
    {
        if (cs3d_ssdit_gemm(f, ws->t_s, ws->o_s, swsh->xa_out_w, swsh->xa_out_w16,
                       swsh->xa_out_b, N_s, dim, dim) < 0) return -1;
    }
    {
        if (cs3d_ssdit_gemm(f, ws->t_p, ws->o_p, swp->xa_out_w, swp->xa_out_w16,
                       swp->xa_out_b, N_p, dim, dim) < 0) return -1;
    }
    {
        int total = N_s * dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_xs, &ws->t_s, &total };
        if (cuLaunchKernel(f->resadd, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int total = N_p * dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_xp, &ws->t_p, &total };
        if (cuLaunchKernel(f->resadd, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }

    /* === norm3 + adaLN_mlp + FFN + gated_mlp === */
    {
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *as[] = { &ws->h_s, &d_xs, &d_smlp, &d_cmlp, &N_s, &dim, &eps };
        if (cuLaunchKernel(f->mod_ln, (unsigned)N_s, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, as, NULL) != CUDA_SUCCESS) return -1;
        void *ap[] = { &ws->h_p, &d_xp, &d_smlp, &d_cmlp, &N_p, &dim, &eps };
        if (cuLaunchKernel(f->mod_ln, (unsigned)N_p, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, ap, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        if (cs3d_ssdit_gemm(f, ws->m1s, ws->h_s, swsh->mlp_fc1_w, swsh->mlp_fc1_w16,
                       swsh->mlp_fc1_b, N_s, dim, mlp_h) < 0) return -1;
    }
    {
        if (cs3d_ssdit_gemm(f, ws->m1p, ws->h_p, swp->mlp_fc1_w, swp->mlp_fc1_w16,
                       swp->mlp_fc1_b, N_p, dim, mlp_h) < 0) return -1;
    }
    {
        int total = N_s * mlp_h;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &ws->m1s, &total };
        if (cuLaunchKernel(f->gelu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int total = N_p * mlp_h;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &ws->m1p, &total };
        if (cuLaunchKernel(f->gelu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        if (cs3d_ssdit_gemm(f, ws->t_s, ws->m1s, swsh->mlp_fc2_w, swsh->mlp_fc2_w16,
                       swsh->mlp_fc2_b, N_s, mlp_h, dim) < 0) return -1;
    }
    {
        if (cs3d_ssdit_gemm(f, ws->t_p, ws->m1p, swp->mlp_fc2_w, swp->mlp_fc2_w16,
                       swp->mlp_fc2_b, N_p, mlp_h, dim) < 0) return -1;
    }
    {
        unsigned grid = (unsigned)((N_s * dim + 255) / 256);
        void *args[] = { &d_xs, &ws->t_s, &d_gmlp, &N_s, &dim };
        if (cuLaunchKernel(f->gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        unsigned grid = (unsigned)((N_p * dim + 255) / 256);
        void *args[] = { &d_xp, &ws->t_p, &d_gmlp, &N_p, &dim };
        if (cuLaunchKernel(f->gated, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    return 0;
}

#endif /* CUDA_SAM3D_SSDIT_FORWARD_IMPLEMENTATION */
