/* SS Flow DiT device-side weight buffers — Phase 2c.10.
 *
 * Single-header. Define HIP_SAM3D_SSDIT_GPU_IMPLEMENTATION in exactly
 * one translation unit before including. Mirrors the shape of
 * hip_sam3d_ppe_gpu.h but for the much larger SS Flow DiT.
 *
 * Loads from an already-parsed `sam3d_ss_flow_dit_model` (CPU side
 * keeps the mmap and qtensor metadata). Each weight is dequantized to
 * host F32 then uploaded to the device — handles F32 / F16 / Q8_0 /
 * Q4_K / Q6_K transparently via qt_dequant.
 *
 * Layout (per-block):
 *   - shared modulation: adaln_w/b
 *   - per-stream (× SAM3D_SS_DIT_N_STREAMS = 2):
 *       norm2_w/b,
 *       sa_qkv_w/b, sa_out_w/b, sa_q_rms_gamma, sa_k_rms_gamma,
 *       xa_q_w/b, xa_kv_w/b, xa_out_w/b,
 *       mlp_fc1_w/b, mlp_fc2_w/b
 *
 * Top-level: t_embedder fc1/2, d_embedder fc1/2 (shortcut only),
 * 5 latent_mapping entries (input_w/b, out_w/b, pos_emb).
 */

#ifndef HIP_SAM3D_SSDIT_GPU_H_
#define HIP_SAM3D_SSDIT_GPU_H_

#include <stddef.h>
#include "../rocew.h"
#include "../../common/sam3d_ss_flow_dit.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-stream device pointers (one set per transformer stream). */
typedef struct {
    hipDeviceptr_t norm2_w, norm2_b;
    hipDeviceptr_t sa_qkv_w, sa_qkv_b;
    hipDeviceptr_t sa_out_w, sa_out_b;
    hipDeviceptr_t sa_q_rms_gamma, sa_k_rms_gamma;
    hipDeviceptr_t xa_q_w,  xa_q_b;
    hipDeviceptr_t xa_kv_w, xa_kv_b;
    hipDeviceptr_t xa_out_w, xa_out_b;
    hipDeviceptr_t mlp_fc1_w, mlp_fc1_b;
    hipDeviceptr_t mlp_fc2_w, mlp_fc2_b;
} cs3d_ssdit_block_stream_w;

typedef struct {
    hipDeviceptr_t adaln_w, adaln_b;                /* [6D, D], [6D] */
    cs3d_ssdit_block_stream_w stream[SAM3D_SS_DIT_N_STREAMS];
} cs3d_ssdit_block_w;

typedef struct {
    hipDeviceptr_t input_w, input_b;
    hipDeviceptr_t out_w,   out_b;
    hipDeviceptr_t pos_emb;
    int in_channels;
    int token_len;
} cs3d_ssdit_latent_w;

typedef struct {
    /* Geometry mirrored from sam3d_ss_flow_dit_model. */
    int dim, n_heads, head_dim, n_blocks, mlp_hidden, cond_channels;
    int freq_dim, is_shortcut;
    float ln_eps, time_scale, ss_resolution;

    /* Top-level embedders. */
    hipDeviceptr_t t_emb_fc1_w, t_emb_fc1_b;
    hipDeviceptr_t t_emb_fc2_w, t_emb_fc2_b;
    hipDeviceptr_t d_emb_fc1_w, d_emb_fc1_b;        /* 0 if !is_shortcut */
    hipDeviceptr_t d_emb_fc2_w, d_emb_fc2_b;

    cs3d_ssdit_latent_w latent[SAM3D_SS_DIT_N_LATENTS];
    cs3d_ssdit_block_w *blocks;                  /* n_blocks entries */

    size_t total_bytes;
    int    loaded;
} cs3d_ssdit_gpu;

/* Upload all SS DiT weights to the device. Returns 0 on success, <0 on
 * failure (any partial allocations are released). `verbose` ≥ 1 prints
 * geometry + total bytes. */
int  cs3d_ssdit_gpu_load(cs3d_ssdit_gpu *g,
                         const sam3d_ss_flow_dit_model *m, int verbose);

/* Release every hipDeviceptr_t held by `g` and zero its fields. Safe on
 * an unloaded (zeroed) struct. */
void cs3d_ssdit_gpu_free(cs3d_ssdit_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_SSDIT_GPU_H_ */

/* ============================ implementation ============================ */
#ifdef HIP_SAM3D_SSDIT_GPU_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hip_runner_common.h"

static int cs3d_ssditg_upload(const qtensor *t, const char *name,
                              hipDeviceptr_t *out_d, size_t *out_bytes)
{
    if (!t || !t->data) {
        fprintf(stderr, "cs3d_ssdit_gpu: missing tensor %s\n", name);
        return -1;
    }
    float *buf = qt_dequant(t);
    if (!buf) {
        fprintf(stderr, "cs3d_ssdit_gpu: dequant %s failed\n", name);
        return -1;
    }
    int n = qt_numel(t);
    size_t nb = (size_t)n * sizeof(float);
    hipDeviceptr_t d = hip_upload_raw(buf, nb);
    free(buf);
    if (!d) {
        fprintf(stderr, "cs3d_ssdit_gpu: hipMalloc failed for %s (%zu bytes)\n",
                name, nb);
        return -1;
    }
    *out_d = d;
    *out_bytes += nb;
    return 0;
}

int cs3d_ssdit_gpu_load(cs3d_ssdit_gpu *g,
                        const sam3d_ss_flow_dit_model *m, int verbose)
{
    if (!g || !m) return -1;
    memset(g, 0, sizeof(*g));
    g->dim           = m->dim;
    g->n_heads       = m->n_heads;
    g->head_dim      = m->head_dim;
    g->n_blocks      = m->n_blocks;
    g->mlp_hidden    = m->mlp_hidden;
    g->cond_channels = m->cond_channels;
    g->freq_dim      = m->freq_dim;
    g->is_shortcut   = m->is_shortcut;
    g->ln_eps        = m->ln_eps;
    g->time_scale    = m->time_scale;
    g->ss_resolution = m->ss_resolution;

    size_t tot = 0;

#define UP_(field, qt) \
    if (cs3d_ssditg_upload(&(m->qt), #qt, &g->field, &tot) < 0) goto fail
    UP_(t_emb_fc1_w, t_emb_fc1_w); UP_(t_emb_fc1_b, t_emb_fc1_b);
    UP_(t_emb_fc2_w, t_emb_fc2_w); UP_(t_emb_fc2_b, t_emb_fc2_b);
#undef UP_
    if (m->is_shortcut) {
#define UP_(field, qt) \
        if (cs3d_ssditg_upload(&(m->qt), #qt, &g->field, &tot) < 0) goto fail
        UP_(d_emb_fc1_w, d_emb_fc1_w); UP_(d_emb_fc1_b, d_emb_fc1_b);
        UP_(d_emb_fc2_w, d_emb_fc2_w); UP_(d_emb_fc2_b, d_emb_fc2_b);
#undef UP_
    }

    /* 5 latent_mapping entries. */
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        const sam3d_ss_latent_map *L = &m->latent[i];
        cs3d_ssdit_latent_w *G = &g->latent[i];
        G->in_channels = L->in_channels;
        G->token_len   = L->token_len;
#define UP_(field) \
        if (cs3d_ssditg_upload(&L->field, "latent." #field, &G->field, &tot) < 0) goto fail
        UP_(input_w); UP_(input_b);
        UP_(out_w);   UP_(out_b);
        UP_(pos_emb);
#undef UP_
    }

    /* Per-block. */
    g->blocks = (cs3d_ssdit_block_w *)calloc((size_t)m->n_blocks,
                                             sizeof(cs3d_ssdit_block_w));
    if (!g->blocks) goto fail;
    for (int b = 0; b < m->n_blocks; b++) {
        const sam3d_ss_block *B = &m->blocks[b];
        cs3d_ssdit_block_w  *D = &g->blocks[b];
        if (cs3d_ssditg_upload(&B->adaln_w, "adaln_w", &D->adaln_w, &tot) < 0) goto fail;
        if (cs3d_ssditg_upload(&B->adaln_b, "adaln_b", &D->adaln_b, &tot) < 0) goto fail;
        for (int s = 0; s < SAM3D_SS_DIT_N_STREAMS; s++) {
            const sam3d_ss_block_stream *S = &B->stream[s];
            cs3d_ssdit_block_stream_w   *T = &D->stream[s];
#define UP_(field) \
            if (cs3d_ssditg_upload(&S->field, #field, &T->field, &tot) < 0) goto fail
            UP_(norm2_w); UP_(norm2_b);
            UP_(sa_qkv_w); UP_(sa_qkv_b);
            UP_(sa_out_w); UP_(sa_out_b);
            UP_(sa_q_rms_gamma); UP_(sa_k_rms_gamma);
            UP_(xa_q_w);   UP_(xa_q_b);
            UP_(xa_kv_w);  UP_(xa_kv_b);
            UP_(xa_out_w); UP_(xa_out_b);
            UP_(mlp_fc1_w); UP_(mlp_fc1_b);
            UP_(mlp_fc2_w); UP_(mlp_fc2_b);
#undef UP_
        }
    }

    g->total_bytes = tot;
    g->loaded = 1;
    if (verbose) {
        fprintf(stderr,
                "cs3d_ssdit_gpu: loaded n_blocks=%d D=%d H=%d D_h=%d mlp_h=%d "
                "cond=%d shortcut=%d  %.1f MiB on device\n",
                g->n_blocks, g->dim, g->n_heads, g->head_dim, g->mlp_hidden,
                g->cond_channels, g->is_shortcut,
                (double)tot / (1024.0 * 1024.0));
    }
    return 0;

fail:
    cs3d_ssdit_gpu_free(g);
    return -1;
}

void cs3d_ssdit_gpu_free(cs3d_ssdit_gpu *g)
{
    if (!g) return;
    hipDeviceptr_t *top[] = {
        &g->t_emb_fc1_w, &g->t_emb_fc1_b, &g->t_emb_fc2_w, &g->t_emb_fc2_b,
        &g->d_emb_fc1_w, &g->d_emb_fc1_b, &g->d_emb_fc2_w, &g->d_emb_fc2_b,
    };
    for (size_t i = 0; i < sizeof(top)/sizeof(top[0]); i++) {
        if (*top[i]) { hipFree(*top[i]); *top[i] = 0; }
    }
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        cs3d_ssdit_latent_w *L = &g->latent[i];
        hipDeviceptr_t *all[] = { &L->input_w, &L->input_b, &L->out_w, &L->out_b, &L->pos_emb };
        for (size_t k = 0; k < sizeof(all)/sizeof(all[0]); k++) {
            if (*all[k]) { hipFree(*all[k]); *all[k] = 0; }
        }
    }
    if (g->blocks) {
        for (int b = 0; b < g->n_blocks; b++) {
            cs3d_ssdit_block_w *D = &g->blocks[b];
            hipDeviceptr_t *shared[] = { &D->adaln_w, &D->adaln_b };
            for (size_t k = 0; k < sizeof(shared)/sizeof(shared[0]); k++) {
                if (*shared[k]) { hipFree(*shared[k]); *shared[k] = 0; }
            }
            for (int s = 0; s < SAM3D_SS_DIT_N_STREAMS; s++) {
                cs3d_ssdit_block_stream_w *T = &D->stream[s];
                hipDeviceptr_t *all[] = {
                    &T->norm2_w, &T->norm2_b,
                    &T->sa_qkv_w, &T->sa_qkv_b, &T->sa_out_w, &T->sa_out_b,
                    &T->sa_q_rms_gamma, &T->sa_k_rms_gamma,
                    &T->xa_q_w, &T->xa_q_b, &T->xa_kv_w, &T->xa_kv_b,
                    &T->xa_out_w, &T->xa_out_b,
                    &T->mlp_fc1_w, &T->mlp_fc1_b, &T->mlp_fc2_w, &T->mlp_fc2_b,
                };
                for (size_t k = 0; k < sizeof(all)/sizeof(all[0]); k++) {
                    if (*all[k]) { hipFree(*all[k]); *all[k] = 0; }
                }
            }
        }
        free(g->blocks);
        g->blocks = NULL;
    }
    memset(g, 0, sizeof(*g));
}

#endif /* HIP_SAM3D_SSDIT_GPU_IMPLEMENTATION */
