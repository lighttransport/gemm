/* SLAT Flow DiT transformer forward driver - Phase 5b.15. */
#ifndef HIP_SAM3D_SLAT_DIT_FORWARD_H_
#define HIP_SAM3D_SLAT_DIT_FORWARD_H_

#include <stddef.h>
#include "../rocew.h"
#include "hip_sam3d_slat_dit_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    hipFunction_t gemm, mod_ln, ln, qkv_split, kv_split, mhrms, sdpa;
    hipFunction_t gelu, gated, resadd, silu;
} cs3d_slatdit_fns;

typedef struct {
    hipDeviceptr_t t_silu, mod6, h, qkv, q, k, v, sa, proj;
    hipDeviceptr_t kv, K, V, xa, mh, mh2;
    int N_max, Nc_max, dim, hidden;
    size_t total_bytes;
} cs3d_slatdit_block_ws;

int  cs3d_slatdit_fns_lookup(cs3d_slatdit_fns *fns, hipModule_t mod);
int  cs3d_slatdit_block_ws_alloc(cs3d_slatdit_block_ws *ws, int N_max, int Nc_max, int dim, int hidden);
void cs3d_slatdit_block_ws_free(cs3d_slatdit_block_ws *ws);
int  cs3d_slatdit_block_forward(const cs3d_slatdit_fns *f, cs3d_slatdit_block_ws *ws,
                                const cs3d_slatdit_block_w *bd, hipDeviceptr_t d_t_emb,
                                hipDeviceptr_t d_x, int N, hipDeviceptr_t d_cond, int Nc,
                                int dim, int H, int D_h, int hidden, float eps);
int  cs3d_slatdit_stack_forward(const cs3d_slatdit_fns *f, cs3d_slatdit_block_ws *ws,
                                const cs3d_slatdit_gpu *g, int first_block, int n_blocks,
                                hipDeviceptr_t d_t_emb, hipDeviceptr_t d_x, int N,
                                hipDeviceptr_t d_cond, int Nc);

#ifdef __cplusplus
}
#endif
#endif

#ifdef HIP_SAM3D_SLAT_DIT_FORWARD_IMPLEMENTATION
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cs3d_slatdit_fns_lookup(cs3d_slatdit_fns *f, hipModule_t mod)
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
    if (hipModuleGetFunction(&f->silu,      mod, "silu_inplace_f32")       != hipSuccess) return -1;
    return 0;
}

static int cs3d_slatdit_alloc_(hipDeviceptr_t *out, size_t bytes, size_t *tot)
{ if (hipMalloc(out, bytes) != hipSuccess) return -1; *tot += bytes; return 0; }

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
    hipDeviceptr_t *all[] = { &ws->t_silu, &ws->mod6, &ws->h, &ws->qkv, &ws->q, &ws->k, &ws->v,
        &ws->sa, &ws->proj, &ws->kv, &ws->K, &ws->V, &ws->xa, &ws->mh, &ws->mh2 };
    for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); i++) if (*all[i]) { hipFree(*all[i]); *all[i] = 0; }
    memset(ws, 0, sizeof(*ws));
}

int cs3d_slatdit_block_forward(const cs3d_slatdit_fns *f, cs3d_slatdit_block_ws *ws,
                               const cs3d_slatdit_block_w *bd, hipDeviceptr_t d_t_emb,
                               hipDeviceptr_t d_x, int N, hipDeviceptr_t d_cond, int Nc,
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

    if (hipMemcpyDtoD(ws->t_silu, d_t_emb, (size_t)dim * sizeof(float)) != hipSuccess) return -1;
    { void *a[] = { &ws->t_silu, &dim }; if (hipModuleLaunchKernel(f->silu, (dim+255)/256,1,1,256,1,1,0,0,a,NULL) != hipSuccess) return -1; }
    { unsigned gx=1, gy=(six_dim+15)/16; void *a[] = { &ws->mod6, &ws->t_silu, (void *)&bd->adaln_w, (void *)&bd->adaln_b, &one, &dim, &six_dim };
      if (hipModuleLaunchKernel(f->gemm, gx,gy,1,16,16,1,0,0,a,NULL) != hipSuccess) return -1; }
    hipDeviceptr_t d_shift_msa = ws->mod6 + (size_t)0 * dim * sizeof(float);
    hipDeviceptr_t d_scale_msa = ws->mod6 + (size_t)1 * dim * sizeof(float);
    hipDeviceptr_t d_gate_msa  = ws->mod6 + (size_t)2 * dim * sizeof(float);
    hipDeviceptr_t d_shift_mlp = ws->mod6 + (size_t)3 * dim * sizeof(float);
    hipDeviceptr_t d_scale_mlp = ws->mod6 + (size_t)4 * dim * sizeof(float);
    hipDeviceptr_t d_gate_mlp  = ws->mod6 + (size_t)5 * dim * sizeof(float);

    { void *a[] = { &ws->h, &d_x, &d_shift_msa, &d_scale_msa, &N, &dim, &eps };
      if (hipModuleLaunchKernel(f->mod_ln, N,1,1,threads,1,1,(unsigned)ln_smem,0,a,NULL) != hipSuccess) return -1; }
    { unsigned gx=(N+15)/16, gy=(qkv_dim+15)/16; void *a[] = { &ws->qkv, &ws->h, (void *)&bd->sa_qkv_w, (void *)&bd->sa_qkv_b, &N, &dim, &qkv_dim };
      if (hipModuleLaunchKernel(f->gemm, gx,gy,1,16,16,1,0,0,a,NULL) != hipSuccess) return -1; }
    { void *a[] = { &ws->qkv, (void *)&bd->sa_q_rms_gamma, &N, &H, &D_h, &qkv_dim };
      if (hipModuleLaunchKernel(f->mhrms, H,N,1,64,1,1,(unsigned)rms_smem,0,a,NULL) != hipSuccess) return -1;
      hipDeviceptr_t kptr = ws->qkv + (size_t)dim * sizeof(float); void *ak[] = { &kptr, (void *)&bd->sa_k_rms_gamma, &N, &H, &D_h, &qkv_dim };
      if (hipModuleLaunchKernel(f->mhrms, H,N,1,64,1,1,(unsigned)rms_smem,0,ak,NULL) != hipSuccess) return -1; }
    { void *a[] = { &ws->q, &ws->k, &ws->v, &ws->qkv, &N, &dim };
      if (hipModuleLaunchKernel(f->qkv_split, (n_elem+255)/256,1,1,256,1,1,0,0,a,NULL) != hipSuccess) return -1; }
    { void *a[] = { &ws->sa, &ws->q, &ws->k, &ws->v, &N, &N, &H, &D_h, &attn_scale };
      if (hipModuleLaunchKernel(f->sdpa, N,H,1,256,1,1,(unsigned)sdpa_self_smem,0,a,NULL) != hipSuccess) return -1; }
    { unsigned gx=(N+15)/16, gy=(dim+15)/16; void *a[] = { &ws->proj, &ws->sa, (void *)&bd->sa_out_w, (void *)&bd->sa_out_b, &N, &dim, &dim };
      if (hipModuleLaunchKernel(f->gemm, gx,gy,1,16,16,1,0,0,a,NULL) != hipSuccess) return -1; }
    { void *a[] = { &d_x, &ws->proj, &d_gate_msa, &N, &dim };
      if (hipModuleLaunchKernel(f->gated, (n_elem+255)/256,1,1,256,1,1,0,0,a,NULL) != hipSuccess) return -1; }

    { void *a[] = { &ws->h, &d_x, (void *)&bd->norm2_w, (void *)&bd->norm2_b, &N, &dim, &eps, &affine };
      if (hipModuleLaunchKernel(f->ln, N,1,1,threads,1,1,(unsigned)ln_smem,0,a,NULL) != hipSuccess) return -1; }
    { unsigned gx=(N+15)/16, gy=(dim+15)/16; void *a[] = { &ws->q, &ws->h, (void *)&bd->xa_q_w, (void *)&bd->xa_q_b, &N, &dim, &dim };
      if (hipModuleLaunchKernel(f->gemm, gx,gy,1,16,16,1,0,0,a,NULL) != hipSuccess) return -1; }
    { unsigned gx=(Nc+15)/16, gy=(kv_dim+15)/16; void *a[] = { &ws->kv, &d_cond, (void *)&bd->xa_kv_w, (void *)&bd->xa_kv_b, &Nc, &dim, &kv_dim };
      if (hipModuleLaunchKernel(f->gemm, gx,gy,1,16,16,1,0,0,a,NULL) != hipSuccess) return -1; }
    { void *a[] = { &ws->K, &ws->V, &ws->kv, &Nc, &dim };
      if (hipModuleLaunchKernel(f->kv_split, (n_cd+255)/256,1,1,256,1,1,0,0,a,NULL) != hipSuccess) return -1; }
    { void *a[] = { &ws->xa, &ws->q, &ws->K, &ws->V, &N, &Nc, &H, &D_h, &attn_scale };
      if (hipModuleLaunchKernel(f->sdpa, N,H,1,256,1,1,(unsigned)sdpa_cross_smem,0,a,NULL) != hipSuccess) return -1; }
    { unsigned gx=(N+15)/16, gy=(dim+15)/16; void *a[] = { &ws->proj, &ws->xa, (void *)&bd->xa_out_w, (void *)&bd->xa_out_b, &N, &dim, &dim };
      if (hipModuleLaunchKernel(f->gemm, gx,gy,1,16,16,1,0,0,a,NULL) != hipSuccess) return -1; }
    { void *a[] = { &d_x, &ws->proj, &n_elem };
      if (hipModuleLaunchKernel(f->resadd, (n_elem+255)/256,1,1,256,1,1,0,0,a,NULL) != hipSuccess) return -1; }

    { void *a[] = { &ws->h, &d_x, &d_shift_mlp, &d_scale_mlp, &N, &dim, &eps };
      if (hipModuleLaunchKernel(f->mod_ln, N,1,1,threads,1,1,(unsigned)ln_smem,0,a,NULL) != hipSuccess) return -1; }
    { unsigned gx=(N+15)/16, gy=(hidden+15)/16; void *a[] = { &ws->mh, &ws->h, (void *)&bd->mlp_fc1_w, (void *)&bd->mlp_fc1_b, &N, &dim, &hidden };
      if (hipModuleLaunchKernel(f->gemm, gx,gy,1,16,16,1,0,0,a,NULL) != hipSuccess) return -1; }
    { void *a[] = { &ws->mh, &n_mh };
      if (hipModuleLaunchKernel(f->gelu, (n_mh+255)/256,1,1,256,1,1,0,0,a,NULL) != hipSuccess) return -1; }
    { unsigned gx=(N+15)/16, gy=(dim+15)/16; void *a[] = { &ws->mh2, &ws->mh, (void *)&bd->mlp_fc2_w, (void *)&bd->mlp_fc2_b, &N, &hidden, &dim };
      if (hipModuleLaunchKernel(f->gemm, gx,gy,1,16,16,1,0,0,a,NULL) != hipSuccess) return -1; }
    { void *a[] = { &d_x, &ws->mh2, &d_gate_mlp, &N, &dim };
      if (hipModuleLaunchKernel(f->gated, (n_elem+255)/256,1,1,256,1,1,0,0,a,NULL) != hipSuccess) return -1; }
    return 0;
}

int cs3d_slatdit_stack_forward(const cs3d_slatdit_fns *f, cs3d_slatdit_block_ws *ws,
                               const cs3d_slatdit_gpu *g, int first_block, int n_blocks,
                               hipDeviceptr_t d_t_emb, hipDeviceptr_t d_x, int N,
                               hipDeviceptr_t d_cond, int Nc)
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

#endif /* HIP_SAM3D_SLAT_DIT_FORWARD_IMPLEMENTATION */
