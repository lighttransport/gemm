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

#ifndef CUDA_SAM3D_PPE_FORWARD_H_
#define CUDA_SAM3D_PPE_FORWARD_H_

#include "../cuew.h"
#include "../cuda_hip_compat.h"
#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    CUfunction ppe_linear3_invalid_f32;
    CUfunction ppe_window_pack_f32;
    CUfunction layernorm_token_f32;
    CUfunction gemm_f32_bias;
    CUfunction qkv_split_f32;
    CUfunction sdpa_batched_f32;
    CUfunction residual_add_f32;
    CUfunction gelu_inplace_f32;
    CUfunction ppe_cls_pos_extract_f32;
} cs3d_ppe_fns;

static inline int cs3d_ppe_fns_lookup(cs3d_ppe_fns *f, CUmodule mod)
{
#define LOOKUP_(name) \
    if (cuModuleGetFunction(&f->name, mod, #name) != CUDA_SUCCESS) { \
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
    CUdeviceptr point_proj_w;       /* [D, 3]      */
    CUdeviceptr point_proj_b;       /* [D]         */
    CUdeviceptr invalid_xyz_token;  /* [D]         */
    CUdeviceptr cls_token;          /* [D]         */
    CUdeviceptr pos_embed_window;   /* [WL, D]     */
    CUdeviceptr pos_embed;          /* [D, Np, Np] */
    CUdeviceptr ln1_w, ln1_b;       /* [D]         */
    CUdeviceptr qkv_w, qkv_b;       /* [3D, D], [3D] */
    CUdeviceptr proj_w, proj_b;     /* [D, D], [D]   */
    CUdeviceptr ln2_w, ln2_b;       /* [D]           */
    CUdeviceptr fc1_w, fc1_b;       /* [ffn, D], [ffn] */
    CUdeviceptr fc2_w, fc2_b;       /* [D, ffn], [D]   */
} cs3d_ppe_w;

/* Per-call workspace. Sized for one PPE encode at (Np, P, D, ffn). */
typedef struct {
    int Np, P, D, ffn;
    int S, WL, Nwin, n_tok;
    CUdeviceptr xflat;     /* [S*S, D]              */
    CUdeviceptr tok;       /* [Nwin, WL, D]         */
    CUdeviceptr ln_buf;    /* [n_tok, D]            */
    CUdeviceptr qkv;       /* [n_tok, 3*D]          */
    CUdeviceptr Q, K, V;   /* [n_tok, D] each       */
    CUdeviceptr attn_out;  /* [n_tok, D]            */
    CUdeviceptr proj_out;  /* [n_tok, D]            */
    CUdeviceptr ffn_buf;   /* [n_tok, ffn]          */
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
    if (cuMemAlloc(&w->xflat,    nx) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->tok,      nt) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->ln_buf,   nt) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->qkv,      nq) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->Q,        nt) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->K,        nt) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->V,        nt) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->attn_out, nt) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->proj_out, nt) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->ffn_buf,  nf) != CUDA_SUCCESS) return -1;
    return 0;
}

static inline void cs3d_ppe_ws_free(cs3d_ppe_ws *w)
{
    if (w->xflat)    cuMemFree(w->xflat);
    if (w->tok)      cuMemFree(w->tok);
    if (w->ln_buf)   cuMemFree(w->ln_buf);
    if (w->qkv)      cuMemFree(w->qkv);
    if (w->Q)        cuMemFree(w->Q);
    if (w->K)        cuMemFree(w->K);
    if (w->V)        cuMemFree(w->V);
    if (w->attn_out) cuMemFree(w->attn_out);
    if (w->proj_out) cuMemFree(w->proj_out);
    if (w->ffn_buf)  cuMemFree(w->ffn_buf);
}

/* Run the full PPE encode on `pmap` ([S*S, 3] = [S, S, 3] f32) →
 * `out` ([Nwin, D]). Geometry derived from `ws`. Caller drives sync. */
static inline int cs3d_ppe_forward(
    const cs3d_ppe_fns *f, cs3d_ppe_ws *ws, const cs3d_ppe_w *bw,
    CUdeviceptr pmap, CUdeviceptr out, int n_heads, float ln_eps)
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
        if (cuLaunchKernel(f->ppe_linear3_invalid_f32,
                           n_pix, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 2. ppe_window_pack: reshape + CLS prepend + pos_embed_window add. */
    {
        int blocks = Nwin * WL;
        void *args[] = { &ws->tok, &ws->xflat, &bw->cls_token, &bw->pos_embed_window,
                         &Np, &P, &D };
        if (cuLaunchKernel(f->ppe_window_pack_f32,
                           blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 3. PPE block: LN1. */
    {
        void *args[] = { &ws->ln_buf, &ws->tok, &bw->ln1_w, &bw->ln1_b,
                         &n_tok, &D, &ln_eps, &affine };
        if (cuLaunchKernel(f->layernorm_token_f32,
                           n_tok, 1, 1, 256, 1, 1, ln_shmem, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int Dout = 3 * D;
        int gx = (n_tok + 15) / 16, gy = (Dout + 15) / 16;
        void *args[] = { &ws->qkv, &ws->ln_buf, &bw->qkv_w, &bw->qkv_b,
                         &n_tok, &D, &Dout };
        if (cuLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int total = n_tok * D, blocks = (total + 255) / 256;
        void *args[] = { &ws->Q, &ws->K, &ws->V, &ws->qkv, &n_tok, &D };
        if (cuLaunchKernel(f->qkv_split_f32, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        unsigned shmem = (unsigned)((256 + WL) * sizeof(float));
        float scale = 1.0f / sqrtf((float)D_h);
        void *args[] = { &ws->attn_out, &ws->Q, &ws->K, &ws->V,
                         &WL, &WL, &n_heads, &D_h, &scale };
        if (cuLaunchKernel(f->sdpa_batched_f32,
                           WL, n_heads, Nwin, 256, 1, 1, shmem, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int Dout = D;
        int gx = (n_tok + 15) / 16, gy = (Dout + 15) / 16;
        void *args[] = { &ws->proj_out, &ws->attn_out, &bw->proj_w, &bw->proj_b,
                         &n_tok, &D, &Dout };
        if (cuLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int n = n_tok * D, blocks = (n + 255) / 256;
        void *args[] = { &ws->tok, &ws->proj_out, &n };
        if (cuLaunchKernel(f->residual_add_f32, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        void *args[] = { &ws->ln_buf, &ws->tok, &bw->ln2_w, &bw->ln2_b,
                         &n_tok, &D, &ln_eps, &affine };
        if (cuLaunchKernel(f->layernorm_token_f32,
                           n_tok, 1, 1, 256, 1, 1, ln_shmem, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int Dout = ffn;
        int gx = (n_tok + 15) / 16, gy = (Dout + 15) / 16;
        void *args[] = { &ws->ffn_buf, &ws->ln_buf, &bw->fc1_w, &bw->fc1_b,
                         &n_tok, &D, &Dout };
        if (cuLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int n = n_tok * ffn, blocks = (n + 255) / 256;
        void *args[] = { &ws->ffn_buf, &n };
        if (cuLaunchKernel(f->gelu_inplace_f32, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int Dout = D;
        int gx = (n_tok + 15) / 16, gy = (Dout + 15) / 16;
        void *args[] = { &ws->proj_out, &ws->ffn_buf, &bw->fc2_w, &bw->fc2_b,
                         &n_tok, &ffn, &Dout };
        if (cuLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    {
        int n = n_tok * D, blocks = (n + 255) / 256;
        void *args[] = { &ws->tok, &ws->proj_out, &n };
        if (cuLaunchKernel(f->residual_add_f32, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 4. CLS extract + pos. */
    {
        void *args[] = { &out, &ws->tok, &bw->pos_embed, &Np, &WL, &D };
        if (cuLaunchKernel(f->ppe_cls_pos_extract_f32,
                           Nwin, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SAM3D_PPE_FORWARD_H_ */
