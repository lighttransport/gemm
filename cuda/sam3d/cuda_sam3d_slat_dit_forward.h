/* SLAT Flow DiT transformer forward driver - Phase 5b.15. */
#ifndef CUDA_SAM3D_SLAT_DIT_FORWARD_H_
#define CUDA_SAM3D_SLAT_DIT_FORWARD_H_

#include <stddef.h>
#include "../cuew.h"
#include "../cublasew.h"
#include "cuda_sam3d_slat_dit_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    CUfunction gemm, mod_ln, ln, qkv_split, kv_split, mhrms, sdpa;
    CUfunction gelu, gated, resadd, silu;
    CUfunction cast_f16, cast_bf16, add_bias;
    cublasew_context *cublas;
    CUdeviceptr gemm_x16, gemm_w16;
    size_t gemm_x16_cap, gemm_w16_cap;
    int use_cublas_gemm;
    int mma_precision;      /* 0=f32, 1=fp16, 2=bf16 */
} cs3d_slatdit_fns;

typedef struct {
    CUdeviceptr t_silu, mod6, h, qkv, q, k, v, sa, proj;
    CUdeviceptr kv, K, V, xa, mh, mh2;
    int N_max, Nc_max, dim, hidden;
    size_t total_bytes;
} cs3d_slatdit_block_ws;

int  cs3d_slatdit_fns_lookup(cs3d_slatdit_fns *fns, CUmodule mod);
void cs3d_slatdit_fns_set_precision(cs3d_slatdit_fns *fns, const char *precision);
void cs3d_slatdit_fns_free(cs3d_slatdit_fns *fns);
int  cs3d_slatdit_block_ws_alloc(cs3d_slatdit_block_ws *ws, int N_max, int Nc_max, int dim, int hidden);
void cs3d_slatdit_block_ws_free(cs3d_slatdit_block_ws *ws);
int  cs3d_slatdit_block_forward(const cs3d_slatdit_fns *f, cs3d_slatdit_block_ws *ws,
                                const cs3d_slatdit_block_w *bd, CUdeviceptr d_t_emb,
                                CUdeviceptr d_x, int N, CUdeviceptr d_cond, int Nc,
                                int dim, int H, int D_h, int hidden, float eps);
int  cs3d_slatdit_stack_forward(const cs3d_slatdit_fns *f, cs3d_slatdit_block_ws *ws,
                                const cs3d_slatdit_gpu *g, int first_block, int n_blocks,
                                CUdeviceptr d_t_emb, CUdeviceptr d_x, int N,
                                CUdeviceptr d_cond, int Nc);

#ifdef __cplusplus
}
#endif
#endif

#ifdef CUDA_SAM3D_SLAT_DIT_FORWARD_IMPLEMENTATION
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cs3d_slatdit_fns_lookup(cs3d_slatdit_fns *f, CUmodule mod)
{
    memset(f, 0, sizeof(*f));
    if (cuModuleGetFunction(&f->gemm,      mod, "gemm_f32_bias")          != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->mod_ln,    mod, "modulated_ln_f32")       != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->ln,        mod, "layernorm_token_f32")    != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->qkv_split, mod, "qkv_split_f32")          != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->kv_split,  mod, "kv_split_f32")           != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->mhrms,     mod, "multi_head_rmsnorm_f32") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->sdpa,      mod, "sdpa_f32")               != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->gelu,      mod, "gelu_tanh_inplace_f32")  != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->gated,     mod, "gated_residual_add_f32") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->resadd,    mod, "residual_add_f32")       != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->silu,      mod, "silu_inplace_f32")       != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->cast_f16,  mod, "cast_f32_to_f16_sam3d")  != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->cast_bf16, mod, "cast_f32_to_bf16_sam3d") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&f->add_bias,  mod, "add_bias_rows_f32")      != CUDA_SUCCESS) return -1;
    const char *env = getenv("SAM3D_CUBLAS_GEMM");
    if (!env || env[0] != '0') {
        if (cublasewCreate(&f->cublas, 0) == 0) {
            f->use_cublas_gemm = 1;
        }
    }
    return 0;
}

void cs3d_slatdit_fns_set_precision(cs3d_slatdit_fns *f, const char *precision)
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

void cs3d_slatdit_fns_free(cs3d_slatdit_fns *f)
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
    f->mma_precision = 0;
}

static int cs3d_slatdit_ensure_u16(CUdeviceptr *ptr, size_t *cap, size_t n_elem)
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

static int cs3d_slatdit_cast_u16(const cs3d_slatdit_fns *f,
                                 CUdeviceptr d_src, CUdeviceptr d_dst, int n)
{
    CUfunction fn = (f->mma_precision == 2) ? f->cast_bf16 : f->cast_f16;
    void *args[] = { &d_src, &d_dst, &n };
    return (cuLaunchKernel(fn, (unsigned)((n + 255) / 256), 1, 1,
                           256, 1, 1, 0, 0, args, NULL) == CUDA_SUCCESS) ? 0 : -1;
}

static int cs3d_slatdit_add_bias(const cs3d_slatdit_fns *f,
                                 CUdeviceptr d_out, CUdeviceptr d_b, int N, int M)
{
    int total = N * M;
    void *args[] = { &d_out, &d_b, &N, &M };
    return (cuLaunchKernel(f->add_bias, (unsigned)((total + 255) / 256), 1, 1,
                           256, 1, 1, 0, 0, args, NULL) == CUDA_SUCCESS) ? 0 : -1;
}

static int cs3d_slatdit_gemm(const cs3d_slatdit_fns *f,
                             CUdeviceptr d_out, CUdeviceptr d_in,
                             CUdeviceptr d_w, CUdeviceptr d_w16, CUdeviceptr d_b,
                             int N, int K, int M)
{
    if (f->use_cublas_gemm && f->cublas && f->mma_precision &&
        (d_w16 || N >= 128) && K >= 64 && M >= 64 && (K % 8) == 0 && (M % 8) == 0) {
        cs3d_slatdit_fns *mf = (cs3d_slatdit_fns *)f;
        size_t xn = (size_t)N * (size_t)K;
        size_t wn = (size_t)M * (size_t)K;
        CUdeviceptr use_w16 = d_w16;
        int have_w16 = use_w16 != 0;
        if (cs3d_slatdit_ensure_u16(&mf->gemm_x16, &mf->gemm_x16_cap, xn) == 0 &&
            (have_w16 || cs3d_slatdit_ensure_u16(&mf->gemm_w16, &mf->gemm_w16_cap, wn) == 0) &&
            cs3d_slatdit_cast_u16(f, d_in, mf->gemm_x16, (int)xn) == 0 &&
            (have_w16 || cs3d_slatdit_cast_u16(f, d_w,  mf->gemm_w16, (int)wn) == 0)) {
            if (!have_w16) use_w16 = mf->gemm_w16;
            int ok = (f->mma_precision == 2)
                ? cublasew_gemm_bf16_bf16_f32_rowmajor_nt(
                      f->cublas, d_out, use_w16, mf->gemm_x16, N, M, K)
                : cublasew_gemm_f16_f16_f32_rowmajor_nt(
                      f->cublas, d_out, use_w16, mf->gemm_x16, N, M, K);
            if (ok == 0) {
                if (d_b && cs3d_slatdit_add_bias(f, d_out, d_b, N, M) < 0) return -1;
                return 0;
            }
        }
    }
    if (!d_w) return -1;
    if (f->use_cublas_gemm && f->cublas) {
        int ok = d_b
            ? cublasew_gemm_f32_lt_bias_rowmajor_nt(f->cublas, d_out, d_w,
                                                    d_in, d_b, N, M, K)
            : cublasew_gemm_f32_lt_rowmajor_nt(f->cublas, d_out, d_w,
                                               d_in, N, M, K);
        if (ok == 0) return 0;
    }
    unsigned gx = (N + 15) / 16, gy = (M + 15) / 16;
    void *a[] = { &d_out, &d_in, &d_w, &d_b, &N, &K, &M };
    return (cuLaunchKernel(f->gemm, gx, gy, 1, 16, 16, 1, 0, 0,
                           a, NULL) == CUDA_SUCCESS) ? 0 : -1;
}

static int cs3d_slatdit_alloc_(CUdeviceptr *out, size_t bytes, size_t *tot)
{ if (cuMemAlloc(out, bytes) != CUDA_SUCCESS) return -1; *tot += bytes; return 0; }

int cs3d_slatdit_block_ws_alloc(cs3d_slatdit_block_ws *ws, int N_max, int Nc_max, int dim, int hidden)
{
    if (!ws || N_max <= 0 || Nc_max <= 0 || dim <= 0 || hidden <= 0) return -1;
    memset(ws, 0, sizeof(*ws));
    ws->N_max = N_max; ws->Nc_max = Nc_max; ws->dim = dim; ws->hidden = hidden;
    size_t f = sizeof(float), tot = 0;
#define A(field, n) if (cs3d_slatdit_alloc_(&ws->field, (size_t)(n) * f, &tot) < 0) goto fail
    A(t_silu, dim); A(mod6, 6 * dim); A(h, (size_t)N_max * dim);
    A(qkv, (size_t)N_max * 3 * dim); A(q, (size_t)N_max * dim); A(k, (size_t)N_max * dim);
    A(v, (size_t)N_max * dim); A(sa, (size_t)N_max * dim); A(proj, (size_t)N_max * dim);
    A(kv, (size_t)Nc_max * 2 * dim); A(K, (size_t)Nc_max * dim); A(V, (size_t)Nc_max * dim);
    A(xa, (size_t)N_max * dim); A(mh, (size_t)N_max * hidden); A(mh2, (size_t)N_max * dim);
#undef A
    ws->total_bytes = tot; return 0;
fail:
    cs3d_slatdit_block_ws_free(ws); return -1;
}

void cs3d_slatdit_block_ws_free(cs3d_slatdit_block_ws *ws)
{
    if (!ws) return;
    CUdeviceptr *all[] = { &ws->t_silu, &ws->mod6, &ws->h, &ws->qkv, &ws->q, &ws->k, &ws->v,
        &ws->sa, &ws->proj, &ws->kv, &ws->K, &ws->V, &ws->xa, &ws->mh, &ws->mh2 };
    for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); i++) if (*all[i]) { cuMemFree(*all[i]); *all[i] = 0; }
    memset(ws, 0, sizeof(*ws));
}

int cs3d_slatdit_block_forward(const cs3d_slatdit_fns *f, cs3d_slatdit_block_ws *ws,
                               const cs3d_slatdit_block_w *bd, CUdeviceptr d_t_emb,
                               CUdeviceptr d_x, int N, CUdeviceptr d_cond, int Nc,
                               int dim, int H, int D_h, int hidden, float eps)
{
    if (!f || !ws || !bd || N > ws->N_max || Nc > ws->Nc_max) return -1;
    int qkv_dim = 3 * dim, kv_dim = 2 * dim;
    float attn_scale = 1.0f / sqrtf((float)D_h);
    int affine = 1, one = 1, six_dim = 6 * dim, n_elem = N * dim, n_mh = N * hidden, n_cd = Nc * dim;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float), rms_smem = 64 * sizeof(float);
    size_t sdpa_self_smem = (size_t)(256 + N) * sizeof(float);
    size_t sdpa_cross_smem = (size_t)(256 + Nc) * sizeof(float);

    if (cuMemcpyDtoD(ws->t_silu, d_t_emb, (size_t)dim * sizeof(float)) != CUDA_SUCCESS) return -1;
    { void *a[] = { &ws->t_silu, &dim }; if (cuLaunchKernel(f->silu, (dim+255)/256,1,1,256,1,1,0,0,a,NULL) != CUDA_SUCCESS) return -1; }
    if (cs3d_slatdit_gemm(f, ws->mod6, ws->t_silu, bd->adaln_w, bd->adaln_w16,
                          bd->adaln_b, one, dim, six_dim) < 0) return -1;
    CUdeviceptr d_shift_msa = ws->mod6 + (size_t)0 * dim * sizeof(float);
    CUdeviceptr d_scale_msa = ws->mod6 + (size_t)1 * dim * sizeof(float);
    CUdeviceptr d_gate_msa  = ws->mod6 + (size_t)2 * dim * sizeof(float);
    CUdeviceptr d_shift_mlp = ws->mod6 + (size_t)3 * dim * sizeof(float);
    CUdeviceptr d_scale_mlp = ws->mod6 + (size_t)4 * dim * sizeof(float);
    CUdeviceptr d_gate_mlp  = ws->mod6 + (size_t)5 * dim * sizeof(float);

    { void *a[] = { &ws->h, &d_x, &d_shift_msa, &d_scale_msa, &N, &dim, &eps };
      if (cuLaunchKernel(f->mod_ln, N,1,1,threads,1,1,(unsigned)ln_smem,0,a,NULL) != CUDA_SUCCESS) return -1; }
    if (cs3d_slatdit_gemm(f, ws->qkv, ws->h, bd->sa_qkv_w, bd->sa_qkv_w16,
                          bd->sa_qkv_b, N, dim, qkv_dim) < 0) return -1;
    { void *a[] = { &ws->qkv, (void *)&bd->sa_q_rms_gamma, &N, &H, &D_h, &qkv_dim };
      if (cuLaunchKernel(f->mhrms, H,N,1,64,1,1,(unsigned)rms_smem,0,a,NULL) != CUDA_SUCCESS) return -1;
      CUdeviceptr kptr = ws->qkv + (size_t)dim * sizeof(float); void *ak[] = { &kptr, (void *)&bd->sa_k_rms_gamma, &N, &H, &D_h, &qkv_dim };
      if (cuLaunchKernel(f->mhrms, H,N,1,64,1,1,(unsigned)rms_smem,0,ak,NULL) != CUDA_SUCCESS) return -1; }
    { void *a[] = { &ws->q, &ws->k, &ws->v, &ws->qkv, &N, &dim };
      if (cuLaunchKernel(f->qkv_split, (n_elem+255)/256,1,1,256,1,1,0,0,a,NULL) != CUDA_SUCCESS) return -1; }
    { void *a[] = { &ws->sa, &ws->q, &ws->k, &ws->v, &N, &N, &H, &D_h, &attn_scale };
      if (cuLaunchKernel(f->sdpa, N,H,1,256,1,1,(unsigned)sdpa_self_smem,0,a,NULL) != CUDA_SUCCESS) return -1; }
    if (cs3d_slatdit_gemm(f, ws->proj, ws->sa, bd->sa_out_w, bd->sa_out_w16,
                          bd->sa_out_b, N, dim, dim) < 0) return -1;
    { void *a[] = { &d_x, &ws->proj, &d_gate_msa, &N, &dim };
      if (cuLaunchKernel(f->gated, (n_elem+255)/256,1,1,256,1,1,0,0,a,NULL) != CUDA_SUCCESS) return -1; }

    { void *a[] = { &ws->h, &d_x, (void *)&bd->norm2_w, (void *)&bd->norm2_b, &N, &dim, &eps, &affine };
      if (cuLaunchKernel(f->ln, N,1,1,threads,1,1,(unsigned)ln_smem,0,a,NULL) != CUDA_SUCCESS) return -1; }
    if (cs3d_slatdit_gemm(f, ws->q, ws->h, bd->xa_q_w, bd->xa_q_w16,
                          bd->xa_q_b, N, dim, dim) < 0) return -1;
    if (cs3d_slatdit_gemm(f, ws->kv, d_cond, bd->xa_kv_w, bd->xa_kv_w16,
                          bd->xa_kv_b, Nc, dim, kv_dim) < 0) return -1;
    { void *a[] = { &ws->K, &ws->V, &ws->kv, &Nc, &dim };
      if (cuLaunchKernel(f->kv_split, (n_cd+255)/256,1,1,256,1,1,0,0,a,NULL) != CUDA_SUCCESS) return -1; }
    { void *a[] = { &ws->xa, &ws->q, &ws->K, &ws->V, &N, &Nc, &H, &D_h, &attn_scale };
      if (cuLaunchKernel(f->sdpa, N,H,1,256,1,1,(unsigned)sdpa_cross_smem,0,a,NULL) != CUDA_SUCCESS) return -1; }
    if (cs3d_slatdit_gemm(f, ws->proj, ws->xa, bd->xa_out_w, bd->xa_out_w16,
                          bd->xa_out_b, N, dim, dim) < 0) return -1;
    { void *a[] = { &d_x, &ws->proj, &n_elem };
      if (cuLaunchKernel(f->resadd, (n_elem+255)/256,1,1,256,1,1,0,0,a,NULL) != CUDA_SUCCESS) return -1; }

    { void *a[] = { &ws->h, &d_x, &d_shift_mlp, &d_scale_mlp, &N, &dim, &eps };
      if (cuLaunchKernel(f->mod_ln, N,1,1,threads,1,1,(unsigned)ln_smem,0,a,NULL) != CUDA_SUCCESS) return -1; }
    if (cs3d_slatdit_gemm(f, ws->mh, ws->h, bd->mlp_fc1_w, bd->mlp_fc1_w16,
                          bd->mlp_fc1_b, N, dim, hidden) < 0) return -1;
    { void *a[] = { &ws->mh, &n_mh };
      if (cuLaunchKernel(f->gelu, (n_mh+255)/256,1,1,256,1,1,0,0,a,NULL) != CUDA_SUCCESS) return -1; }
    if (cs3d_slatdit_gemm(f, ws->mh2, ws->mh, bd->mlp_fc2_w, bd->mlp_fc2_w16,
                          bd->mlp_fc2_b, N, hidden, dim) < 0) return -1;
    { void *a[] = { &d_x, &ws->mh2, &d_gate_mlp, &N, &dim };
      if (cuLaunchKernel(f->gated, (n_elem+255)/256,1,1,256,1,1,0,0,a,NULL) != CUDA_SUCCESS) return -1; }
    return 0;
}

int cs3d_slatdit_stack_forward(const cs3d_slatdit_fns *f, cs3d_slatdit_block_ws *ws,
                               const cs3d_slatdit_gpu *g, int first_block, int n_blocks,
                               CUdeviceptr d_t_emb, CUdeviceptr d_x, int N,
                               CUdeviceptr d_cond, int Nc)
{
    if (!g || !g->loaded || first_block < 0 || n_blocks < 0 || first_block + n_blocks > g->n_blocks) return -1;
    for (int i = 0; i < n_blocks; i++) {
        if (cs3d_slatdit_block_forward(f, ws, &g->blocks[first_block + i], d_t_emb,
                                       d_x, N, d_cond, Nc, g->dim, g->n_heads,
                                       g->head_dim, g->mlp_hidden, g->ln_eps) != 0)
            return -1;
    }
    return 0;
}

#endif /* CUDA_SAM3D_SLAT_DIT_FORWARD_IMPLEMENTATION */
