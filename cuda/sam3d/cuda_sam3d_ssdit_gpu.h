/* SS Flow DiT device-side weight buffers — Phase 2c.10.
 *
 * Single-header. Define CUDA_SAM3D_SSDIT_GPU_IMPLEMENTATION in exactly
 * one translation unit before including. Mirrors the shape of
 * cuda_sam3d_ppe_gpu.h but for the much larger SS Flow DiT.
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

#ifndef CUDA_SAM3D_SSDIT_GPU_H_
#define CUDA_SAM3D_SSDIT_GPU_H_

#include <stddef.h>
#include "../cuew.h"
#include "../../common/sam3d_ss_flow_dit.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-stream device pointers (one set per transformer stream). */
typedef struct {
    CUdeviceptr norm2_w, norm2_b;
    CUdeviceptr sa_qkv_w, sa_qkv_b;
    CUdeviceptr sa_qkv_w16;
    CUdeviceptr sa_out_w, sa_out_b;
    CUdeviceptr sa_out_w16;
    CUdeviceptr sa_q_rms_gamma, sa_k_rms_gamma;
    CUdeviceptr xa_q_w,  xa_q_b;
    CUdeviceptr xa_q_w16;
    CUdeviceptr xa_kv_w, xa_kv_b;
    CUdeviceptr xa_kv_w16;
    CUdeviceptr xa_out_w, xa_out_b;
    CUdeviceptr xa_out_w16;
    CUdeviceptr mlp_fc1_w, mlp_fc1_b;
    CUdeviceptr mlp_fc1_w16;
    CUdeviceptr mlp_fc2_w, mlp_fc2_b;
    CUdeviceptr mlp_fc2_w16;
} cs3d_ssdit_block_stream_w;

typedef struct {
    CUdeviceptr adaln_w, adaln_b;                /* [6D, D], [6D] */
    CUdeviceptr adaln_w16;
    cs3d_ssdit_block_stream_w stream[SAM3D_SS_DIT_N_STREAMS];
} cs3d_ssdit_block_w;

typedef struct {
    CUdeviceptr input_w, input_b;
    CUdeviceptr input_w16;
    CUdeviceptr out_w,   out_b;
    CUdeviceptr out_w16;
    CUdeviceptr pos_emb;
    int in_channels;
    int token_len;
} cs3d_ssdit_latent_w;

typedef struct {
    /* Geometry mirrored from sam3d_ss_flow_dit_model. */
    int dim, n_heads, head_dim, n_blocks, mlp_hidden, cond_channels;
    int freq_dim, is_shortcut;
    float ln_eps, time_scale, ss_resolution;

    /* Top-level embedders. */
    CUdeviceptr t_emb_fc1_w, t_emb_fc1_b;
    CUdeviceptr t_emb_fc1_w16;
    CUdeviceptr t_emb_fc2_w, t_emb_fc2_b;
    CUdeviceptr t_emb_fc2_w16;
    CUdeviceptr d_emb_fc1_w, d_emb_fc1_b;        /* 0 if !is_shortcut */
    CUdeviceptr d_emb_fc1_w16;
    CUdeviceptr d_emb_fc2_w, d_emb_fc2_b;
    CUdeviceptr d_emb_fc2_w16;

    cs3d_ssdit_latent_w latent[SAM3D_SS_DIT_N_LATENTS];
    cs3d_ssdit_block_w *blocks;                  /* n_blocks entries */

    size_t total_bytes;
    int    loaded;
} cs3d_ssdit_gpu;

/* Upload all SS DiT weights to the device. Returns 0 on success, <0 on
 * failure (any partial allocations are released). `verbose` ≥ 1 prints
 * geometry + total bytes. */
int  cs3d_ssdit_gpu_load(cs3d_ssdit_gpu *g,
                         const sam3d_ss_flow_dit_model *m,
                         const char *cache_dir,
                         const char *precision,
                         int drop_mma_f32,
                         int verbose);

/* Release every CUdeviceptr held by `g` and zero its fields. Safe on
 * an unloaded (zeroed) struct. */
void cs3d_ssdit_gpu_free(cs3d_ssdit_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SAM3D_SSDIT_GPU_H_ */

/* ============================ implementation ============================ */
#ifdef CUDA_SAM3D_SSDIT_GPU_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../cuda_runner_common.h"
#include "cuda_sam3d_mma_cache.h"

static int cs3d_ssditg_upload(const qtensor *t, const char *name,
                              const char *cache_dir, const char *precision,
                              CUdeviceptr *out_d, CUdeviceptr *out_d16,
                              int drop_f32_if_u16,
                              size_t *out_bytes, size_t *out_bytes16,
                              int verbose)
{
    if (!t || !t->data) {
        fprintf(stderr, "cs3d_ssdit_gpu: missing tensor %s\n", name);
        return -1;
    }
    int n = qt_numel(t);
    if (out_d) *out_d = 0;
    if (out_d16) *out_d16 = 0;
    if (out_d16 && drop_f32_if_u16 &&
        cs3d_mma_upload_weight_u16(cache_dir, name, precision, NULL, n,
                                   out_d16, out_bytes16, verbose) == 0) {
        return 0;
    }

    float *buf = qt_dequant(t);
    if (!buf) {
        fprintf(stderr, "cs3d_ssdit_gpu: dequant %s failed\n", name);
        return -1;
    }
    size_t nb = (size_t)n * sizeof(float);
    CUdeviceptr d = 0;
    if (!drop_f32_if_u16 || !out_d16) {
        d = cu_upload_raw(buf, nb);
        if (!d) {
            fprintf(stderr, "cs3d_ssdit_gpu: cuMemAlloc failed for %s (%zu bytes)\n",
                    name, nb);
            free(buf);
            return -1;
        }
    }
    if (out_d16 &&
        cs3d_mma_upload_weight_u16(cache_dir, name, precision, buf, n,
                                   out_d16, out_bytes16, verbose) < 0) {
        if (d) cuMemFree(d);
        free(buf);
        return -1;
    }
    if (out_d) *out_d = d;
    if (d && out_bytes) *out_bytes += nb;
    free(buf);
    return 0;
}

int cs3d_ssdit_gpu_load(cs3d_ssdit_gpu *g,
                        const sam3d_ss_flow_dit_model *m,
                        const char *cache_dir,
                        const char *precision,
                        int drop_mma_f32,
                        int verbose)
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

    size_t tot = 0, tot16 = 0;

#define UPF_(field, qt) \
    if (cs3d_ssditg_upload(&(m->qt), "ssdit." #qt, cache_dir, precision, \
                           &g->field, NULL, 0, &tot, &tot16, verbose) < 0) goto fail
    UPF_(t_emb_fc1_w, t_emb_fc1_w); UPF_(t_emb_fc1_b, t_emb_fc1_b);
    UPF_(t_emb_fc2_w, t_emb_fc2_w); UPF_(t_emb_fc2_b, t_emb_fc2_b);
#undef UPF_
    if (m->is_shortcut) {
#define UPF_(field, qt) \
        if (cs3d_ssditg_upload(&(m->qt), "ssdit." #qt, cache_dir, precision, \
                               &g->field, NULL, 0, &tot, &tot16, verbose) < 0) goto fail
        UPF_(d_emb_fc1_w, d_emb_fc1_w); UPF_(d_emb_fc1_b, d_emb_fc1_b);
        UPF_(d_emb_fc2_w, d_emb_fc2_w); UPF_(d_emb_fc2_b, d_emb_fc2_b);
#undef UPF_
    }

    /* 5 latent_mapping entries. */
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        const sam3d_ss_latent_map *L = &m->latent[i];
        cs3d_ssdit_latent_w *G = &g->latent[i];
        G->in_channels = L->in_channels;
        G->token_len   = L->token_len;
#define UPW_(field) \
        do { \
            char tag[128]; \
            snprintf(tag, sizeof(tag), "ssdit.latent%d." #field, i); \
            if (cs3d_ssditg_upload(&L->field, tag, cache_dir, precision, \
                                   &G->field, &G->field ## 16, drop_mma_f32, &tot, &tot16, verbose) < 0) goto fail; \
        } while (0)
#define UPF_(field) \
        do { \
            char tag[128]; \
            snprintf(tag, sizeof(tag), "ssdit.latent%d." #field, i); \
            if (cs3d_ssditg_upload(&L->field, tag, cache_dir, precision, \
                                   &G->field, NULL, 0, &tot, &tot16, verbose) < 0) goto fail; \
        } while (0)
        UPF_(input_w); UPF_(input_b);
        UPF_(out_w);   UPF_(out_b);
        UPF_(pos_emb);
#undef UPW_
#undef UPF_
    }

    /* Per-block. */
    g->blocks = (cs3d_ssdit_block_w *)calloc((size_t)m->n_blocks,
                                             sizeof(cs3d_ssdit_block_w));
    if (!g->blocks) goto fail;
    for (int b = 0; b < m->n_blocks; b++) {
        const sam3d_ss_block *B = &m->blocks[b];
        cs3d_ssdit_block_w  *D = &g->blocks[b];
        {
            char tag[128];
            snprintf(tag, sizeof(tag), "ssdit.block%02d.adaln_w", b);
            if (cs3d_ssditg_upload(&B->adaln_w, tag, cache_dir, precision,
                                   &D->adaln_w, NULL, 0, &tot, &tot16, verbose) < 0) goto fail;
            snprintf(tag, sizeof(tag), "ssdit.block%02d.adaln_b", b);
            if (cs3d_ssditg_upload(&B->adaln_b, tag, cache_dir, precision,
                                   &D->adaln_b, NULL, 0, &tot, &tot16, verbose) < 0) goto fail;
        }
        for (int s = 0; s < SAM3D_SS_DIT_N_STREAMS; s++) {
            const sam3d_ss_block_stream *S = &B->stream[s];
            cs3d_ssdit_block_stream_w   *T = &D->stream[s];
#define UPW_(field) \
            do { \
                char tag[128]; \
                snprintf(tag, sizeof(tag), "ssdit.block%02d.stream%d." #field, b, s); \
                if (cs3d_ssditg_upload(&S->field, tag, cache_dir, precision, \
                                       &T->field, &T->field ## 16, drop_mma_f32, &tot, &tot16, verbose) < 0) goto fail; \
            } while (0)
#define UPF_(field) \
            do { \
                char tag[128]; \
                snprintf(tag, sizeof(tag), "ssdit.block%02d.stream%d." #field, b, s); \
                if (cs3d_ssditg_upload(&S->field, tag, cache_dir, precision, \
                                       &T->field, NULL, 0, &tot, &tot16, verbose) < 0) goto fail; \
            } while (0)
            UPF_(norm2_w); UPF_(norm2_b);
            if (s == SAM3D_SS_STREAM_SHAPE) { UPW_(sa_qkv_w); } else { UPF_(sa_qkv_w); } UPF_(sa_qkv_b);
            if (s == SAM3D_SS_STREAM_SHAPE) { UPW_(sa_out_w); } else { UPF_(sa_out_w); } UPF_(sa_out_b);
            UPF_(sa_q_rms_gamma); UPF_(sa_k_rms_gamma);
            if (s == SAM3D_SS_STREAM_SHAPE) { UPW_(xa_q_w); } else { UPF_(xa_q_w); } UPF_(xa_q_b);
            UPW_(xa_kv_w);  UPF_(xa_kv_b);
            if (s == SAM3D_SS_STREAM_SHAPE) { UPW_(xa_out_w); } else { UPF_(xa_out_w); } UPF_(xa_out_b);
            if (s == SAM3D_SS_STREAM_SHAPE) { UPW_(mlp_fc1_w); } else { UPF_(mlp_fc1_w); } UPF_(mlp_fc1_b);
            if (s == SAM3D_SS_STREAM_SHAPE) { UPW_(mlp_fc2_w); } else { UPF_(mlp_fc2_w); } UPF_(mlp_fc2_b);
#undef UPW_
#undef UPF_
        }
    }

    g->total_bytes = tot;
    g->loaded = 1;
    if (verbose) {
        fprintf(stderr,
                "cs3d_ssdit_gpu: loaded n_blocks=%d D=%d H=%d D_h=%d mlp_h=%d "
                "cond=%d shortcut=%d  %.1f MiB f32 + %.1f MiB u16 on device\n",
                g->n_blocks, g->dim, g->n_heads, g->head_dim, g->mlp_hidden,
                g->cond_channels, g->is_shortcut,
                (double)tot / (1024.0 * 1024.0),
                (double)tot16 / (1024.0 * 1024.0));
    }
    return 0;

fail:
    cs3d_ssdit_gpu_free(g);
    return -1;
}

void cs3d_ssdit_gpu_free(cs3d_ssdit_gpu *g)
{
    if (!g) return;
    CUdeviceptr *top[] = {
        &g->t_emb_fc1_w, &g->t_emb_fc1_b, &g->t_emb_fc1_w16,
        &g->t_emb_fc2_w, &g->t_emb_fc2_b, &g->t_emb_fc2_w16,
        &g->d_emb_fc1_w, &g->d_emb_fc1_b, &g->d_emb_fc1_w16,
        &g->d_emb_fc2_w, &g->d_emb_fc2_b, &g->d_emb_fc2_w16,
    };
    for (size_t i = 0; i < sizeof(top)/sizeof(top[0]); i++) {
        if (*top[i]) { cuMemFree(*top[i]); *top[i] = 0; }
    }
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        cs3d_ssdit_latent_w *L = &g->latent[i];
        CUdeviceptr *all[] = {
            &L->input_w, &L->input_b, &L->input_w16,
            &L->out_w, &L->out_b, &L->out_w16, &L->pos_emb
        };
        for (size_t k = 0; k < sizeof(all)/sizeof(all[0]); k++) {
            if (*all[k]) { cuMemFree(*all[k]); *all[k] = 0; }
        }
    }
    if (g->blocks) {
        for (int b = 0; b < g->n_blocks; b++) {
            cs3d_ssdit_block_w *D = &g->blocks[b];
            CUdeviceptr *shared[] = { &D->adaln_w, &D->adaln_b, &D->adaln_w16 };
            for (size_t k = 0; k < sizeof(shared)/sizeof(shared[0]); k++) {
                if (*shared[k]) { cuMemFree(*shared[k]); *shared[k] = 0; }
            }
            for (int s = 0; s < SAM3D_SS_DIT_N_STREAMS; s++) {
                cs3d_ssdit_block_stream_w *T = &D->stream[s];
                CUdeviceptr *all[] = {
                    &T->norm2_w, &T->norm2_b,
                    &T->sa_qkv_w, &T->sa_qkv_b, &T->sa_qkv_w16,
                    &T->sa_out_w, &T->sa_out_b, &T->sa_out_w16,
                    &T->sa_q_rms_gamma, &T->sa_k_rms_gamma,
                    &T->xa_q_w, &T->xa_q_b, &T->xa_q_w16,
                    &T->xa_kv_w, &T->xa_kv_b, &T->xa_kv_w16,
                    &T->xa_out_w, &T->xa_out_b, &T->xa_out_w16,
                    &T->mlp_fc1_w, &T->mlp_fc1_b, &T->mlp_fc1_w16,
                    &T->mlp_fc2_w, &T->mlp_fc2_b, &T->mlp_fc2_w16,
                };
                for (size_t k = 0; k < sizeof(all)/sizeof(all[0]); k++) {
                    if (*all[k]) { cuMemFree(*all[k]); *all[k] = 0; }
                }
            }
        }
        free(g->blocks);
        g->blocks = NULL;
    }
    memset(g, 0, sizeof(*g));
}

#endif /* CUDA_SAM3D_SSDIT_GPU_IMPLEMENTATION */
