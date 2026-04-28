/* Per-modality CondEmbedderFuser projection on the device — Phase 2b.8e.
 *
 * Wraps the layernorm/gemm/silu_mul kernels into a single
 *   in[N,Di] → LN → w1 @ ln → w3 @ ln → silu(h1)*h3 → w2 @ h1 + idx_emb
 * graph. The idx_emb broadcast add is folded into the final w2 gemm via
 * gemm_f32_bias, so no extra kernel is needed.
 *
 * Single-header; weight loader lives in cuda_sam3d_fuser_gpu.h.
 */

#ifndef CUDA_SAM3D_FUSER_FORWARD_H_
#define CUDA_SAM3D_FUSER_FORWARD_H_

#include "../cuew.h"
#include "../cuda_hip_compat.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    CUfunction layernorm_token_f32;
    CUfunction gemm_f32_bias;
    CUfunction silu_mul_f32;
} cs3d_fuser_fns;

static inline int cs3d_fuser_fns_lookup(cs3d_fuser_fns *f, CUmodule mod)
{
#define LOOKUP_(name) \
    if (cuModuleGetFunction(&f->name, mod, #name) != CUDA_SUCCESS) { \
        fprintf(stderr, "cs3d_fuser_fns: lookup %s failed\n", #name); \
        return -1; \
    }
    LOOKUP_(layernorm_token_f32);
    LOOKUP_(gemm_f32_bias);
    LOOKUP_(silu_mul_f32);
#undef LOOKUP_
    return 0;
}

/* Per-modality device weights. */
typedef struct {
    int Di, Dh, Do;
    CUdeviceptr ln_w, ln_b;   /* [Di] */
    CUdeviceptr w1;           /* [Dh, Di] */
    CUdeviceptr w3;           /* [Dh, Di] */
    CUdeviceptr w2;           /* [Do, Dh] */
} cs3d_fuser_proj_w;

/* Per-call scratch sized to the maximum of all modalities the runner
 * touches. ln[N_max * Di_max], h1/h3[N_max * Dh_max]. Output is
 * caller-allocated. */
typedef struct {
    int N_max, Di_max, Dh_max;
    CUdeviceptr ln_buf;
    CUdeviceptr h1;
    CUdeviceptr h3;
} cs3d_fuser_ws;

static inline int cs3d_fuser_ws_alloc(cs3d_fuser_ws *ws,
                                      int N_max, int Di_max, int Dh_max)
{
    ws->N_max = N_max; ws->Di_max = Di_max; ws->Dh_max = Dh_max;
    ws->ln_buf = ws->h1 = ws->h3 = 0;
    if (cuMemAlloc(&ws->ln_buf, (size_t)N_max * Di_max * sizeof(float)) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&ws->h1,     (size_t)N_max * Dh_max * sizeof(float)) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&ws->h3,     (size_t)N_max * Dh_max * sizeof(float)) != CUDA_SUCCESS) goto fail;
    return 0;
fail:
    if (ws->ln_buf) cuMemFree(ws->ln_buf);
    if (ws->h1)     cuMemFree(ws->h1);
    if (ws->h3)     cuMemFree(ws->h3);
    ws->ln_buf = ws->h1 = ws->h3 = 0;
    return -1;
}

static inline void cs3d_fuser_ws_free(cs3d_fuser_ws *ws)
{
    if (!ws) return;
    if (ws->ln_buf) cuMemFree(ws->ln_buf);
    if (ws->h1)     cuMemFree(ws->h1);
    if (ws->h3)     cuMemFree(ws->h3);
    ws->ln_buf = ws->h1 = ws->h3 = 0;
}

/* Forward one modality. in/out are device pointers; idx_emb_row may be 0
 * (no pos add). ln_eps mirrors the CPU path (1e-5). */
static inline int cs3d_fuser_project_forward(const cs3d_fuser_fns *f,
                                             cs3d_fuser_ws *ws,
                                             const cs3d_fuser_proj_w *w,
                                             CUdeviceptr in, CUdeviceptr out,
                                             int N,
                                             CUdeviceptr idx_emb_row,
                                             float ln_eps)
{
    if (N > ws->N_max || w->Di > ws->Di_max || w->Dh > ws->Dh_max) {
        fprintf(stderr, "cs3d_fuser_project_forward: ws too small\n");
        return -1;
    }
    int affine = 1;
    int Di = w->Di, Dh = w->Dh, Do = w->Do;

    /* LN_token: grid (N,), block (TH,), shmem 2*TH*sizeof(float). */
    {
        int TH = 256;
        size_t shm = (size_t)2 * TH * sizeof(float);
        void *args[] = { &ws->ln_buf, &in, &w->ln_w, &w->ln_b,
                         &N, &Di, &ln_eps, &affine };
        if (cuLaunchKernel(f->layernorm_token_f32, N, 1, 1, TH, 1, 1,
                           (unsigned int)shm, 0, args, NULL) != CUDA_SUCCESS) return -2;
    }

    /* h1 = w1 @ ln; h3 = w3 @ ln. gemm_f32_bias grid (ceil(N/16), ceil(Dh/16)), block (16,16). */
    {
        unsigned gx = (unsigned)((N + 15) / 16);
        unsigned gy = (unsigned)((Dh + 15) / 16);
        CUdeviceptr null_b = 0;
        void *args1[] = { &ws->h1, &ws->ln_buf, &w->w1, &null_b, &N, &Di, &Dh };
        void *args3[] = { &ws->h3, &ws->ln_buf, &w->w3, &null_b, &N, &Di, &Dh };
        if (cuLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1,
                           0, 0, args1, NULL) != CUDA_SUCCESS) return -3;
        if (cuLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1,
                           0, 0, args3, NULL) != CUDA_SUCCESS) return -3;
    }

    /* silu_mul: h1 = silu(h1) * h3, total = N * Dh. */
    {
        int total = N * Dh;
        int TH = 256;
        unsigned grid = (unsigned)((total + TH - 1) / TH);
        void *args[] = { &ws->h1, &ws->h3, &ws->h1, &total };
        if (cuLaunchKernel(f->silu_mul_f32, grid, 1, 1, TH, 1, 1,
                           0, 0, args, NULL) != CUDA_SUCCESS) return -4;
    }

    /* out = w2 @ h1 + idx_emb_row (broadcast bias, may be 0). */
    {
        unsigned gx = (unsigned)((N + 15) / 16);
        unsigned gy = (unsigned)((Do + 15) / 16);
        void *args[] = { &out, &ws->h1, &w->w2, &idx_emb_row, &N, &Dh, &Do };
        if (cuLaunchKernel(f->gemm_f32_bias, gx, gy, 1, 16, 16, 1,
                           0, 0, args, NULL) != CUDA_SUCCESS) return -5;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SAM3D_FUSER_FORWARD_H_ */
