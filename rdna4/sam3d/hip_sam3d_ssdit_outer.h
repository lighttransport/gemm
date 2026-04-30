/* SS Flow DiT outer GPU forward — Phase 2c.12.
 *
 * Wraps the per-block driver from `hip_sam3d_ssdit_forward.h` with:
 *   - timestep + (optional) shortcut embedder
 *   - 5-modality input projection + pos_emb add
 *   - pose-stream concat (4 modalities → 4-token stream)
 *   - 24-block adaLN_modulation gemm + block_forward loop
 *   - per-modality output projection (LN no-affine + gemm to in_channels)
 *
 * Inputs/outputs are host f32 arrays — internal upload/download is done
 * here. Mirrors the contract of CPU `sam3d_ss_flow_dit_forward` so it
 * can drop into `hip_sam3d_debug_ss_dit_forward`.
 *
 * Define HIP_SAM3D_SSDIT_OUTER_IMPLEMENTATION in exactly one TU.
 */

#ifndef HIP_SAM3D_SSDIT_OUTER_H_
#define HIP_SAM3D_SSDIT_OUTER_H_

#include <stddef.h>
#include "../rocew.h"
#include "hip_sam3d_ssdit_gpu.h"
#include "hip_sam3d_ssdit_forward.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    cs3d_ssdit_fns block;     /* kernel handles for block_forward */
    hipFunction_t ts_cossin;     /* timestep_embed_cossin_f32 */
    hipFunction_t silu;          /* silu_inplace_f32 */
} cs3d_ssdit_outer_fns;

int cs3d_ssdit_outer_fns_lookup(cs3d_ssdit_outer_fns *fns, hipModule_t mod);

typedef struct {
    /* Top-level scratch. Allocated to fit production geometry: latent
     * shape stream (max token_len) + pose stream (sum of pose token_lens). */
    hipDeviceptr_t xs;          /* [N_s, dim]  shape stream */
    hipDeviceptr_t xp;          /* [N_p_total, dim]  pose stream (concat) */
    hipDeviceptr_t cond;        /* [N_c_max, dim] */
    hipDeviceptr_t t_emb;       /* [dim] */
    hipDeviceptr_t d_emb;       /* [dim] (only allocated if shortcut) */
    hipDeviceptr_t t_silu;      /* [dim] scratch for silu(t_emb) per block */
    hipDeviceptr_t mod6;        /* [6 * dim]  per block */
    hipDeviceptr_t freq_buf;    /* [freq_dim] sinusoidal embed scratch */
    hipDeviceptr_t in_proj_tmp; /* [token_len_max, dim]  per-modality scratch */
    hipDeviceptr_t ln_tmp;      /* [token_len_max, dim]  output LN scratch */
    int N_s_cap, N_p_cap, N_c_cap;
    int dim;
    size_t total_bytes;
} cs3d_ssdit_outer_ws;

int  cs3d_ssdit_outer_ws_alloc(cs3d_ssdit_outer_ws *ws,
                               const cs3d_ssdit_gpu *g, int N_c_cap);
void cs3d_ssdit_outer_ws_free (cs3d_ssdit_outer_ws *ws);

/* One-shot host→device cond upload. Use this once per ODE run, then pass
 * cond=NULL to `cs3d_ssdit_outer_forward` so each shortcut step skips the
 * redundant 11 MiB upload. */
int  cs3d_ssdit_outer_upload_cond(cs3d_ssdit_outer_ws *ws,
                                  const float *cond, int n_cond, int dim);

/* Run the SS Flow DiT forward end-to-end on the GPU.
 *
 *   latents_in[5]/latents_out[5] follow the CPU forward contract — host
 *   f32 buffers sized by sam3d_cpu_ss_dit_lat_elts(i).
 *   `cond` is host [n_cond, cond_channels].
 *   `t`, `d` are scalars (d ignored if !is_shortcut).
 *
 * Returns 0 on success. */
int cs3d_ssdit_outer_forward(const cs3d_ssdit_gpu     *g,
                             const cs3d_ssdit_outer_fns *fns,
                             cs3d_ssdit_block_ws      *ws_block,
                             cs3d_ssdit_outer_ws      *ws_outer,
                             const float *const *latents_in,
                             float *const *latents_out,
                             const float *cond, int n_cond,
                             float t, float d);

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_SSDIT_OUTER_H_ */

/* ============================ implementation ============================ */
#ifdef HIP_SAM3D_SSDIT_OUTER_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hip_runner_common.h"

int cs3d_ssdit_outer_fns_lookup(cs3d_ssdit_outer_fns *fns, hipModule_t mod)
{
    if (cs3d_ssdit_fns_lookup(&fns->block, mod) < 0) return -1;
    if (hipModuleGetFunction(&fns->ts_cossin, mod, "timestep_embed_cossin_f32") != hipSuccess) return -1;
    if (hipModuleGetFunction(&fns->silu,      mod, "silu_inplace_f32")          != hipSuccess) return -1;
    return 0;
}

int cs3d_ssdit_outer_ws_alloc(cs3d_ssdit_outer_ws *ws,
                              const cs3d_ssdit_gpu *g, int N_c_cap)
{
    if (!ws || !g) return -1;
    memset(ws, 0, sizeof(*ws));
    ws->dim = g->dim;
    ws->N_c_cap = N_c_cap;
    int N_s = g->latent[SAM3D_SS_LAT_SHAPE].token_len;
    int N_p = 0, max_tl = N_s;
    for (int i = SAM3D_SS_LAT_6DROT; i <= SAM3D_SS_LAT_TRANSLATION_SCALE; i++) {
        N_p += g->latent[i].token_len;
        if (g->latent[i].token_len > max_tl) max_tl = g->latent[i].token_len;
    }
    ws->N_s_cap = N_s; ws->N_p_cap = N_p;
    int dim = g->dim;
    size_t f = sizeof(float);
    size_t tot = 0;
#define A(field, n) do { \
        if (hipMalloc(&ws->field, (size_t)(n) * f) != hipSuccess) goto fail; \
        tot += (size_t)(n) * f; \
    } while (0)
    A(xs, (size_t)N_s * dim);
    A(xp, (size_t)N_p * dim);
    A(cond, (size_t)N_c_cap * dim);
    A(t_emb, dim);
    if (g->is_shortcut) A(d_emb, dim);
    A(t_silu, dim);
    A(mod6, (size_t)6 * dim);
    A(freq_buf, g->freq_dim);
    A(in_proj_tmp, (size_t)max_tl * dim);
    A(ln_tmp, (size_t)N_s * dim);
#undef A
    ws->total_bytes = tot;
    return 0;
fail:
    cs3d_ssdit_outer_ws_free(ws);
    return -1;
}

int cs3d_ssdit_outer_upload_cond(cs3d_ssdit_outer_ws *ws,
                                 const float *cond, int n_cond, int dim)
{
    if (!ws || !cond) return -1;
    if (n_cond > ws->N_c_cap) {
        fprintf(stderr, "ssdit_outer: cond %d > cap %d\n", n_cond, ws->N_c_cap);
        return -1;
    }
    return hipMemcpyHtoD(ws->cond, cond, (size_t)n_cond * dim * sizeof(float))
           == hipSuccess ? 0 : -1;
}

void cs3d_ssdit_outer_ws_free(cs3d_ssdit_outer_ws *ws)
{
    if (!ws) return;
    hipDeviceptr_t *all[] = {
        &ws->xs, &ws->xp, &ws->cond, &ws->t_emb, &ws->d_emb,
        &ws->t_silu, &ws->mod6, &ws->freq_buf, &ws->in_proj_tmp, &ws->ln_tmp,
    };
    for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); i++) {
        if (*all[i]) { hipFree(*all[i]); *all[i] = 0; }
    }
    memset(ws, 0, sizeof(*ws));
}

/* GEMM helper: 16×16 tiled gemm_f32_bias kernel. */
static int outer_gemm(hipFunction_t k, hipDeviceptr_t d_out, hipDeviceptr_t d_in,
                      hipDeviceptr_t d_w, hipDeviceptr_t d_b,
                      int N, int K, int M)
{
    unsigned gx = (N + 15) / 16, gy = (M + 15) / 16;
    void *args[] = { &d_out, &d_in, &d_w, &d_b, &N, &K, &M };
    return (hipModuleLaunchKernel(k, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) == hipSuccess) ? 0 : -1;
}

/* Build t_emb on device: timestep_cossin → fc1 → silu → fc2. Output [dim]. */
static int build_t_emb(const cs3d_ssdit_outer_fns *fns,
                       const cs3d_ssdit_gpu *g,
                       cs3d_ssdit_outer_ws *ws,
                       hipDeviceptr_t d_out, float t,
                       hipDeviceptr_t fc1_w, hipDeviceptr_t fc1_b,
                       hipDeviceptr_t fc2_w, hipDeviceptr_t fc2_b)
{
    int half = g->freq_dim / 2;
    int dim  = g->dim;
    {
        unsigned grid = (unsigned)((half + 255) / 256);
        void *args[] = { &ws->freq_buf, &t, (void *)&g->freq_dim };
        if (hipModuleLaunchKernel(fns->ts_cossin, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    /* h1 = fc1(emb) [1, dim]. Reuse t_silu as h1 scratch. */
    int N1 = 1;
    if (outer_gemm(fns->block.gemm, ws->t_silu, ws->freq_buf, fc1_w, fc1_b, N1, g->freq_dim, dim) < 0) return -1;
    {
        unsigned grid = (unsigned)((dim + 255) / 256);
        void *args[] = { &ws->t_silu, &dim };
        if (hipModuleLaunchKernel(fns->silu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    if (outer_gemm(fns->block.gemm, d_out, ws->t_silu, fc2_w, fc2_b, N1, dim, dim) < 0) return -1;
    return 0;
}

int cs3d_ssdit_outer_forward(const cs3d_ssdit_gpu       *g,
                             const cs3d_ssdit_outer_fns *fns,
                             cs3d_ssdit_block_ws        *ws_block,
                             cs3d_ssdit_outer_ws        *ws,
                             const float *const *latents_in,
                             float *const *latents_out,
                             const float *cond, int n_cond,
                             float t, float d)
{
    if (!g || !fns || !ws_block || !ws || !latents_in || !latents_out) return -1;
    int dim = g->dim;
    int N_s = g->latent[SAM3D_SS_LAT_SHAPE].token_len;
    int N_p = 0;
    for (int i = SAM3D_SS_LAT_6DROT; i <= SAM3D_SS_LAT_TRANSLATION_SCALE; i++)
        N_p += g->latent[i].token_len;
    if (n_cond > ws->N_c_cap) {
        fprintf(stderr, "ssdit_outer: cond %d > cap %d\n", n_cond, ws->N_c_cap);
        return -1;
    }
    /* `cond == NULL` ⇒ caller has pre-uploaded into ws->cond via
     * cs3d_ssdit_outer_upload_cond (saves a redundant HtoD per ODE step). */
    if (cond) {
        if (hipMemcpyHtoD(ws->cond, cond, (size_t)n_cond * dim * sizeof(float)) != hipSuccess) return -1;
    }

    /* === input projection per modality + pos_emb add === */
    {
        /* shape stream goes directly into ws->xs. */
        const cs3d_ssdit_latent_w *L = &g->latent[SAM3D_SS_LAT_SHAPE];
        if (hipMemcpyHtoD(ws->in_proj_tmp, latents_in[SAM3D_SS_LAT_SHAPE],
                         (size_t)L->token_len * L->in_channels * sizeof(float)) != hipSuccess) return -1;
        if (outer_gemm(fns->block.gemm, ws->xs, ws->in_proj_tmp,
                       L->input_w, L->input_b, L->token_len, L->in_channels, dim) < 0) return -1;
        int total = L->token_len * dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &ws->xs, &L->pos_emb, &total };
        if (hipModuleLaunchKernel(fns->block.resadd, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }
    {
        /* Pose stream = concat of 4 modalities into ws->xp. */
        size_t off = 0;
        for (int i = SAM3D_SS_LAT_6DROT; i <= SAM3D_SS_LAT_TRANSLATION_SCALE; i++) {
            const cs3d_ssdit_latent_w *L = &g->latent[i];
            if (hipMemcpyHtoD(ws->in_proj_tmp, latents_in[i],
                             (size_t)L->token_len * L->in_channels * sizeof(float)) != hipSuccess) return -1;
            hipDeviceptr_t d_dst = ws->xp + off * sizeof(float);
            if (outer_gemm(fns->block.gemm, d_dst, ws->in_proj_tmp,
                           L->input_w, L->input_b, L->token_len, L->in_channels, dim) < 0) return -1;
            int total = L->token_len * dim;
            unsigned grid = (unsigned)((total + 255) / 256);
            void *args[] = { &d_dst, &L->pos_emb, &total };
            if (hipModuleLaunchKernel(fns->block.resadd, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
            off += (size_t)L->token_len * dim;
        }
    }

    /* === t/d embedder === */
    if (build_t_emb(fns, g, ws, ws->t_emb, t,
                    g->t_emb_fc1_w, g->t_emb_fc1_b,
                    g->t_emb_fc2_w, g->t_emb_fc2_b) < 0) return -1;
    if (g->is_shortcut) {
        if (build_t_emb(fns, g, ws, ws->d_emb, d,
                        g->d_emb_fc1_w, g->d_emb_fc1_b,
                        g->d_emb_fc2_w, g->d_emb_fc2_b) < 0) return -1;
        int total = dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &ws->t_emb, &ws->d_emb, &total };
        if (hipModuleLaunchKernel(fns->block.resadd, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
    }

    /* === 24-block loop === */
    for (int b = 0; b < g->n_blocks; b++) {
        /* t_silu = silu(t_emb). */
        hipMemcpyDtoD(ws->t_silu, ws->t_emb, (size_t)dim * sizeof(float));
        {
            unsigned grid = (unsigned)((dim + 255) / 256);
            void *args[] = { &ws->t_silu, &dim };
            if (hipModuleLaunchKernel(fns->silu, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != hipSuccess) return -1;
        }
        /* mod6 = adaln_w @ t_silu + adaln_b   shape [1, 6*dim]. */
        int six_d = 6 * dim;
        if (outer_gemm(fns->block.gemm, ws->mod6, ws->t_silu,
                       g->blocks[b].adaln_w, g->blocks[b].adaln_b,
                       1, dim, six_d) < 0) return -1;
        if (cs3d_ssdit_block_forward(&fns->block, ws_block, &g->blocks[b],
                                     ws->mod6,
                                     ws->xs, N_s, ws->xp, N_p,
                                     ws->cond, n_cond,
                                     dim, g->n_heads, g->head_dim, g->mlp_hidden,
                                     g->ln_eps) < 0) return -1;
    }

    /* === per-modality output projection (LN no-affine + gemm) === */
    int affine_no = 0;
    {
        const cs3d_ssdit_latent_w *L = &g->latent[SAM3D_SS_LAT_SHAPE];
        unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
        void *args[] = { &ws->ln_tmp, &ws->xs, &L->input_w /*ignored*/, &L->input_b /*ignored*/,
                         (void *)&L->token_len, &dim, &g->ln_eps, &affine_no };
        if (hipModuleLaunchKernel(fns->block.ln, (unsigned)L->token_len, 1, 1,
                           threads, 1, 1, (unsigned)smem, 0, args, NULL) != hipSuccess) return -1;
        /* gemm_f32_bias: out[T, in_ch] = ln_tmp @ out_w + out_b */
        int total = L->token_len * L->in_channels;
        hipDeviceptr_t d_proj_out;
        if (hipMalloc(&d_proj_out, (size_t)total * sizeof(float)) != hipSuccess) return -1;
        if (outer_gemm(fns->block.gemm, d_proj_out, ws->ln_tmp,
                       L->out_w, L->out_b, L->token_len, dim, L->in_channels) < 0) {
            hipFree(d_proj_out); return -1;
        }
        hipDeviceSynchronize();
        hipMemcpyDtoH(latents_out[SAM3D_SS_LAT_SHAPE], d_proj_out,
                     (size_t)total * sizeof(float));
        hipFree(d_proj_out);
    }
    {
        size_t off = 0;
        for (int i = SAM3D_SS_LAT_6DROT; i <= SAM3D_SS_LAT_TRANSLATION_SCALE; i++) {
            const cs3d_ssdit_latent_w *L = &g->latent[i];
            hipDeviceptr_t d_src = ws->xp + off * sizeof(float);
            unsigned threads = 256; size_t smem = 2 * threads * sizeof(float);
            void *args[] = { &ws->ln_tmp, &d_src, &L->input_w, &L->input_b,
                             (void *)&L->token_len, &dim, &g->ln_eps, &affine_no };
            if (hipModuleLaunchKernel(fns->block.ln, (unsigned)L->token_len, 1, 1,
                               threads, 1, 1, (unsigned)smem, 0, args, NULL) != hipSuccess) return -1;
            int total = L->token_len * L->in_channels;
            hipDeviceptr_t d_proj_out;
            if (hipMalloc(&d_proj_out, (size_t)total * sizeof(float)) != hipSuccess) return -1;
            if (outer_gemm(fns->block.gemm, d_proj_out, ws->ln_tmp,
                           L->out_w, L->out_b, L->token_len, dim, L->in_channels) < 0) {
                hipFree(d_proj_out); return -1;
            }
            hipDeviceSynchronize();
            hipMemcpyDtoH(latents_out[i], d_proj_out, (size_t)total * sizeof(float));
            hipFree(d_proj_out);
            off += (size_t)L->token_len * dim;
        }
    }
    return 0;
}

#endif /* HIP_SAM3D_SSDIT_OUTER_IMPLEMENTATION */
