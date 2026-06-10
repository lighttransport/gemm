/* SLAT Flow DiT device-side transformer-block weights — Phase 5b.14.
 *
 * Single-header. Define CUDA_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION in exactly
 * one translation unit before including. This is the persistent GPU weight
 * container for the verified SLAT transformer stack: AdaLN modulation,
 * self-attn, cross-attn, and MLP weights for all checkpoint blocks.
 *
 * It intentionally starts at the transformer-stack boundary verified by
 * `verify_slat_transformer_block_realw --stack`: input/out sparse resblocks
 * and top-level SLAT ODE wiring will be hoisted in later phases.
 */

#ifndef CUDA_SAM3D_SLAT_DIT_GPU_H_
#define CUDA_SAM3D_SLAT_DIT_GPU_H_

#include <stddef.h>
#include "../cuew.h"
#include "../../common/sam3d_slat_dit.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    CUdeviceptr adaln_w, adaln_b;               /* [6D,D], [6D] */
    CUdeviceptr adaln_w16;
    CUdeviceptr norm2_w, norm2_b;               /* [D], [D] */
    CUdeviceptr sa_qkv_w, sa_qkv_b;             /* [3D,D], [3D] */
    CUdeviceptr sa_qkv_w16;
    CUdeviceptr sa_q_rms_gamma, sa_k_rms_gamma; /* [D], [D] */
    CUdeviceptr sa_out_w, sa_out_b;             /* [D,D], [D] */
    CUdeviceptr sa_out_w16;
    CUdeviceptr xa_q_w, xa_q_b;                 /* [D,D], [D] */
    CUdeviceptr xa_q_w16;
    CUdeviceptr xa_kv_w, xa_kv_b;               /* [2D,D], [2D] */
    CUdeviceptr xa_kv_w16;
    CUdeviceptr xa_out_w, xa_out_b;             /* [D,D], [D] */
    CUdeviceptr xa_out_w16;
    CUdeviceptr mlp_fc1_w, mlp_fc1_b;           /* [4D,D], [4D] */
    CUdeviceptr mlp_fc1_w16;
    CUdeviceptr mlp_fc2_w, mlp_fc2_b;           /* [D,4D], [D] */
    CUdeviceptr mlp_fc2_w16;
} cs3d_slatdit_block_w;

typedef struct {
    int dim;
    int n_heads;
    int head_dim;
    int n_blocks;
    int mlp_hidden;
    int cond_channels;
    float ln_eps;

    cs3d_slatdit_block_w *blocks;               /* [n_blocks] */
    size_t total_bytes;
    int loaded;
} cs3d_slatdit_gpu;

int  cs3d_slatdit_gpu_load_transformer(cs3d_slatdit_gpu *g,
                                       const sam3d_slat_dit_model *m,
                                       const char *cache_dir,
                                       const char *precision,
                                       int drop_mma_f32,
                                       int verbose);
void cs3d_slatdit_gpu_free(cs3d_slatdit_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SAM3D_SLAT_DIT_GPU_H_ */

/* ============================ implementation ============================ */
#ifdef CUDA_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../cuda_runner_common.h"
#include "cuda_sam3d_mma_cache.h"

static int cs3d_slatditg_upload(const qtensor *t, const char *name,
                                const char *cache_dir, const char *precision,
                                CUdeviceptr *out_d, CUdeviceptr *out_d16,
                                int drop_f32_if_u16,
                                size_t *out_bytes, size_t *out_bytes16,
                                int verbose)
{
    if (!t || !t->data) {
        fprintf(stderr, "cs3d_slatdit_gpu: missing tensor %s\n", name);
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
        fprintf(stderr, "cs3d_slatdit_gpu: dequant %s failed\n", name);
        return -1;
    }
    size_t nb = (size_t)n * sizeof(float);
    CUdeviceptr d = 0;
    if (!drop_f32_if_u16 || !out_d16) {
        d = cu_upload_raw(buf, nb);
        if (!d) {
            fprintf(stderr, "cs3d_slatdit_gpu: cuMemAlloc failed for %s (%zu bytes)\n",
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

int cs3d_slatdit_gpu_load_transformer(cs3d_slatdit_gpu *g,
                                      const sam3d_slat_dit_model *m,
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
    g->mlp_hidden    = (int)(m->mlp_ratio * (float)m->dim + 0.5f);
    g->cond_channels = m->cond_channels;
    g->ln_eps        = m->ln_eps;

    g->blocks = (cs3d_slatdit_block_w *)calloc((size_t)m->n_blocks,
                                                sizeof(cs3d_slatdit_block_w));
    if (!g->blocks) return -1;

    size_t tot = 0, tot16 = 0;
    for (int b = 0; b < m->n_blocks; b++) {
        const sam3d_slat_block *B = &m->blocks[b];
        cs3d_slatdit_block_w *D = &g->blocks[b];
#define UPW_(field) \
        do { \
            char tag[128]; \
            snprintf(tag, sizeof(tag), "slatdit.block%02d." #field, b); \
            if (cs3d_slatditg_upload(&B->field, tag, cache_dir, precision, \
                                     &D->field, &D->field ## 16, drop_mma_f32, &tot, &tot16, verbose) < 0) goto fail; \
        } while (0)
#define UPF_(field) \
        do { \
            char tag[128]; \
            snprintf(tag, sizeof(tag), "slatdit.block%02d." #field, b); \
            if (cs3d_slatditg_upload(&B->field, tag, cache_dir, precision, \
                                     &D->field, NULL, 0, &tot, &tot16, verbose) < 0) goto fail; \
        } while (0)
        UPF_(adaln_w); UPF_(adaln_b);
        UPF_(norm2_w); UPF_(norm2_b);
        UPW_(sa_qkv_w); UPF_(sa_qkv_b);
        UPF_(sa_q_rms_gamma); UPF_(sa_k_rms_gamma);
        UPW_(sa_out_w); UPF_(sa_out_b);
        UPW_(xa_q_w); UPF_(xa_q_b);
        UPW_(xa_kv_w); UPF_(xa_kv_b);
        UPW_(xa_out_w); UPF_(xa_out_b);
        UPW_(mlp_fc1_w); UPF_(mlp_fc1_b);
        UPW_(mlp_fc2_w); UPF_(mlp_fc2_b);
#undef UPW_
#undef UPF_
    }

    g->total_bytes = tot;
    g->loaded = 1;
    if (verbose) {
        fprintf(stderr,
                "cs3d_slatdit_gpu: loaded transformer blocks=%d D=%d H=%d D_h=%d "
                "mlp_h=%d cond=%d %.1f MiB f32 + %.1f MiB u16 on device\n",
                g->n_blocks, g->dim, g->n_heads, g->head_dim,
                g->mlp_hidden, g->cond_channels,
                (double)tot / (1024.0 * 1024.0),
                (double)tot16 / (1024.0 * 1024.0));
    }
    return 0;

fail:
    cs3d_slatdit_gpu_free(g);
    return -1;
}

void cs3d_slatdit_gpu_free(cs3d_slatdit_gpu *g)
{
    if (!g) return;
    if (g->blocks) {
        for (int b = 0; b < g->n_blocks; b++) {
            cs3d_slatdit_block_w *D = &g->blocks[b];
            CUdeviceptr *all[] = {
                &D->adaln_w, &D->adaln_b, &D->adaln_w16,
                &D->norm2_w, &D->norm2_b,
                &D->sa_qkv_w, &D->sa_qkv_b, &D->sa_qkv_w16,
                &D->sa_q_rms_gamma, &D->sa_k_rms_gamma,
                &D->sa_out_w, &D->sa_out_b, &D->sa_out_w16,
                &D->xa_q_w, &D->xa_q_b, &D->xa_q_w16,
                &D->xa_kv_w, &D->xa_kv_b, &D->xa_kv_w16,
                &D->xa_out_w, &D->xa_out_b, &D->xa_out_w16,
                &D->mlp_fc1_w, &D->mlp_fc1_b, &D->mlp_fc1_w16,
                &D->mlp_fc2_w, &D->mlp_fc2_b, &D->mlp_fc2_w16,
            };
            for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); i++) {
                if (*all[i]) { cuMemFree(*all[i]); *all[i] = 0; }
            }
        }
        free(g->blocks);
    }
    memset(g, 0, sizeof(*g));
}

#endif /* CUDA_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION */
