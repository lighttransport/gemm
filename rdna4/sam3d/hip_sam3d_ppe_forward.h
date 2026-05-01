/* Composed PointPatchEmbed forward on the device — Phase 2b.8b.
 *
 * Wraps the Phase 2b.{0,3,4,6a,6b,7} kernels into a single
 * pmap → win_cls graph callable from the runner:
 *
 *   ppe_linear3_invalid → ppe_window_pack →
 *     LN → qkv gemm+bias → split → sdpa_batched →
 *     proj+bias → resid → LN → fc1+bias → GELU →
 *     fc2+bias → resid →
 *   ppe_cls_pos_extract
 *
 * No LayerScale — PPE has none, unlike the DINOv2 blocks.
 *
 * Bilinear pos_embed-resize path is omitted: sam3d always runs at
 * input_size 256 → pos_embed grid (32×32) already matches Np×Np.
 *
 * Single-header. Validated end-to-end by `test_ppe_full`.
 */

#ifndef HIP_SAM3D_PPE_FORWARD_H_
#define HIP_SAM3D_PPE_FORWARD_H_

#include "../rocew.h"

#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    hipFunction_t ppe_linear3_invalid_f32;
    hipFunction_t ppe_window_pack_f32;
    hipFunction_t layernorm_token_f32;
    hipFunction_t gemm_f32_bias;
    hipFunction_t qkv_split_f32;
    hipFunction_t sdpa_batched_f32;
    hipFunction_t residual_add_f32;
    hipFunction_t gelu_inplace_f32;
    hipFunction_t ppe_cls_pos_extract_f32;
} cs3d_ppe_fns;

static inline int cs3d_ppe_fns_lookup(cs3d_ppe_fns *f, hipModule_t mod)
{
#define LOOKUP_(name) \
    if (hipModuleGetFunction(&f->name, mod, #name) != hipSuccess) { \
        fprintf(stderr, "cs3d_ppe_fns: lookup %s failed\n", #name); \
        return -1; \
    }
    LOOKUP_(ppe_linear3_invalid_f32);
    LOOKUP_(ppe_window_pack_f32);
    LOOKUP_(layernorm_token_f32);
    LOOKUP_(gemm_f32_bias);
    LOOKUP_(qkv_split_f32);
    LOOKUP_(sdpa_batched_f32);
    LOOKUP_(residual_add_f32);
    LOOKUP_(gelu_inplace_f32);
    LOOKUP_(ppe_cls_pos_extract_f32);
#undef LOOKUP_
    return 0;
}

/* Device weight pointers. Decoupled from the loader so the forward can
 * be unit-tested with random uploads (test_ppe_full). */
typedef struct {
    hipDeviceptr_t point_proj_w;       /* [D, 3]      */
    hipDeviceptr_t point_proj_b;       /* [D]         */
    hipDeviceptr_t invalid_xyz_token;  /* [D]         */
    hipDeviceptr_t cls_token;          /* [D]         */
    hipDeviceptr_t pos_embed_window;   /* [WL, D]     */
    hipDeviceptr_t pos_embed;          /* [D, Np, Np] */
    hipDeviceptr_t ln1_w, ln1_b;       /* [D]         */
    hipDeviceptr_t qkv_w, qkv_b;       /* [3D, D], [3D] */
    hipDeviceptr_t proj_w, proj_b;     /* [D, D], [D]   */
    hipDeviceptr_t ln2_w, ln2_b;       /* [D]           */
    hipDeviceptr_t fc1_w, fc1_b;       /* [ffn, D], [ffn] */
    hipDeviceptr_t fc2_w, fc2_b;       /* [D, ffn], [D]   */
} cs3d_ppe_w;

/* Per-call workspace. Sized for one PPE encode at (Np, P, D, ffn). */
typedef struct {
    int Np, P, D, ffn;
    int S, WL, Nwin, n_tok;
    hipDeviceptr_t xflat;     /* [S*S, D]              */
    hipDeviceptr_t tok;       /* [Nwin, WL, D]         */
    hipDeviceptr_t ln_buf;    /* [n_tok, D]            */
    hipDeviceptr_t qkv;       /* [n_tok, 3*D]          */
    hipDeviceptr_t Q, K, V;   /* [n_tok, D] each       */
    hipDeviceptr_t attn_out;  /* [n_tok, D]            */
    hipDeviceptr_t proj_out;  /* [n_tok, D]            */
    hipDeviceptr_t ffn_buf;   /* [n_tok, ffn]          */
} cs3d_ppe_ws;

static inline int cs3d_ppe_ws_alloc(cs3d_ppe_ws *w, int Np, int P, int D, int ffn)
{
    w->Np = Np; w->P = P; w->D = D; w->ffn = ffn;
    w->S    = Np * P;
    w->WL   = 1 + P * P;
    w->Nwin = Np * Np;
    w->n_tok = w->Nwin * w->WL;
    size_t nx = (size_t)w->S * w->S * D * sizeof(float);
    size_t nt = (size_t)w->n_tok * D * sizeof(float);
    size_t nq = (size_t)w->n_tok * 3 * D * sizeof(float);
    size_t nf = (size_t)w->n_tok * ffn * sizeof(float);
    if (hipMalloc(&w->xflat,    nx) != hipSuccess) return -1;
    if (hipMalloc(&w->tok,      nt) != hipSuccess) return -1;
    if (hipMalloc(&w->ln_buf,   nt) != hipSuccess) return -1;
    if (hipMalloc(&w->qkv,      nq) != hipSuccess) return -1;
    if (hipMalloc(&w->Q,        nt) != hipSuccess) return -1;
    if (hipMalloc(&w->K,        nt) != hipSuccess) return -1;
    if (hipMalloc(&w->V,        nt) != hipSuccess) return -1;
    if (hipMalloc(&w->attn_out, nt) != hipSuccess) return -1;
    if (hipMalloc(&w->proj_out, nt) != hipSuccess) return -1;
    if (hipMalloc(&w->ffn_buf,  nf) != hipSuccess) return -1;
    return 0;
}

static inline void cs3d_ppe_ws_free(cs3d_ppe_ws *w)
{
    if (w->xflat)    hipFree(w->xflat);
    if (w->tok)      hipFree(w->tok);
    if (w->ln_buf)   hipFree(w->ln_buf);
    if (w->qkv)      hipFree(w->qkv);
    if (w->Q)        hipFree(w->Q);
    if (w->K)        hipFree(w->K);
    if (w->V)        hipFree(w->V);
    if (w->attn_out) hipFree(w->attn_out);
    if (w->proj_out) hipFree(w->proj_out);
    if (w->ffn_buf)  hipFree(w->ffn_buf);
}

/* Run the full PPE encode on `pmap` ([S*S, 3] = [S, S, 3] f32) →
 * `out` ([Nwin, D]). Geometry derived from `ws`. Caller drives sync. */
static inline int cs3d_ppe_forward(
    const cs3d_ppe_fns *f, cs3d_ppe_ws *ws, const cs3d_ppe_w *bw,
    hipDeviceptr_t pmap, hipDeviceptr_t out, int n_heads, float ln_eps)
{
    int D = ws->D, ffn = ws->ffn, Np = ws->Np, P = ws->P;
    int S = ws->S, WL = ws->WL, Nwin = ws->Nwin, n_tok = ws->n_tok;
    int D_h = D / n_heads;
    int affine = 1;
    unsigned ln_shmem = (unsigned)(2 * 256 * sizeof(float));

    /* 1. ppe_linear3_invalid: per-pixel linear, NaN→inv_tok. */
    {
        int n_pix = S * S;
        void *args[] = { &ws->xflat, &pmap, &bw->point_proj_w, &bw->point_proj_b,
                         &bw->invalid_xyz_token, &n_pix, &D };
        if (hipModuleLaunchKernel(f->ppe_linear3_invalid_f32,
                           n_pix, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    /* 2. ppe_window_pack: reshape + CLS prepend + pos_embed_window add. */
    {
        int blocks = Nwin * WL;
        void *args[] = { &ws->tok, &ws->xflat, &bw->cls_token, &bw->pos_embed_window,
                         &Np, &P, &D };
        if (hipModuleLaunchKernel(f->ppe_window_pack_f32,
                           blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    /* 3. PPE block: LN1. */
    {
        void *args[] = { &ws->ln_buf, &ws->tok, &bw->ln1_w, &bw->ln1_b,
                         &n_tok, &D, &ln_eps, &affine };
        if (hipModuleLaunchKernel(f->layernorm_token_f32,
                           n_tok, 1, 1, 256, 1, 1, ln_shmem, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int Dout = 3 * D;
        int gx = (n_tok + 15) / 16, gy = (Dout + 15) / 16;
        void *args[] = { &ws->qkv, &ws->ln_buf, &bw->qkv_w, &bw->qkv_b,
                         &n_tok, &D, &Dout };
        if (hipModuleLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int total = n_tok * D, blocks = (total + 255) / 256;
        void *args[] = { &ws->Q, &ws->K, &ws->V, &ws->qkv, &n_tok, &D };
        if (hipModuleLaunchKernel(f->qkv_split_f32, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        unsigned shmem = (unsigned)((256 + WL) * sizeof(float));
        float scale = 1.0f / sqrtf((float)D_h);
        void *args[] = { &ws->attn_out, &ws->Q, &ws->K, &ws->V,
                         &WL, &WL, &n_heads, &D_h, &scale };
        if (hipModuleLaunchKernel(f->sdpa_batched_f32,
                           WL, n_heads, Nwin, 256, 1, 1, shmem, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int Dout = D;
        int gx = (n_tok + 15) / 16, gy = (Dout + 15) / 16;
        void *args[] = { &ws->proj_out, &ws->attn_out, &bw->proj_w, &bw->proj_b,
                         &n_tok, &D, &Dout };
        if (hipModuleLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int n = n_tok * D, blocks = (n + 255) / 256;
        void *args[] = { &ws->tok, &ws->proj_out, &n };
        if (hipModuleLaunchKernel(f->residual_add_f32, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        void *args[] = { &ws->ln_buf, &ws->tok, &bw->ln2_w, &bw->ln2_b,
                         &n_tok, &D, &ln_eps, &affine };
        if (hipModuleLaunchKernel(f->layernorm_token_f32,
                           n_tok, 1, 1, 256, 1, 1, ln_shmem, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int Dout = ffn;
        int gx = (n_tok + 15) / 16, gy = (Dout + 15) / 16;
        void *args[] = { &ws->ffn_buf, &ws->ln_buf, &bw->fc1_w, &bw->fc1_b,
                         &n_tok, &D, &Dout };
        if (hipModuleLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int n = n_tok * ffn, blocks = (n + 255) / 256;
        void *args[] = { &ws->ffn_buf, &n };
        if (hipModuleLaunchKernel(f->gelu_inplace_f32, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int Dout = D;
        int gx = (n_tok + 15) / 16, gy = (Dout + 15) / 16;
        void *args[] = { &ws->proj_out, &ws->ffn_buf, &bw->fc2_w, &bw->fc2_b,
                         &n_tok, &ffn, &Dout };
        if (hipModuleLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        int n = n_tok * D, blocks = (n + 255) / 256;
        void *args[] = { &ws->tok, &ws->proj_out, &n };
        if (hipModuleLaunchKernel(f->residual_add_f32, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    /* 4. CLS extract + pos. */
    {
        void *args[] = { &out, &ws->tok, &bw->pos_embed, &Np, &WL, &D };
        if (hipModuleLaunchKernel(f->ppe_cls_pos_extract_f32,
                           Nwin, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_PPE_FORWARD_H_ */
