/* SLAT Flow DiT device-side transformer-block weights — Phase 5b.14.
 *
 * Single-header. Define HIP_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION in exactly
 * one translation unit before including. This is the persistent GPU weight
 * container for the verified SLAT transformer stack: AdaLN modulation,
 * self-attn, cross-attn, and MLP weights for all checkpoint blocks.
 *
 * It intentionally starts at the transformer-stack boundary verified by
 * `verify_slat_transformer_block_realw --stack`: input/out sparse resblocks
 * and top-level SLAT ODE wiring will be hoisted in later phases.
 */

#ifndef HIP_SAM3D_SLAT_DIT_GPU_H_
#define HIP_SAM3D_SLAT_DIT_GPU_H_

#include <stddef.h>
#include "../rocew.h"
#include "../../common/sam3d_slat_dit.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    hipDeviceptr_t adaln_w, adaln_b;               /* [6D,D], [6D] */
    hipDeviceptr_t norm2_w, norm2_b;               /* [D], [D] */
    hipDeviceptr_t sa_qkv_w, sa_qkv_b;             /* [3D,D], [3D] */
    hipDeviceptr_t sa_q_rms_gamma, sa_k_rms_gamma; /* [D], [D] */
    hipDeviceptr_t sa_out_w, sa_out_b;             /* [D,D], [D] */
    hipDeviceptr_t xa_q_w, xa_q_b;                 /* [D,D], [D] */
    hipDeviceptr_t xa_kv_w, xa_kv_b;               /* [2D,D], [2D] */
    hipDeviceptr_t xa_out_w, xa_out_b;             /* [D,D], [D] */
    hipDeviceptr_t mlp_fc1_w, mlp_fc1_b;           /* [4D,D], [4D] */
    hipDeviceptr_t mlp_fc2_w, mlp_fc2_b;           /* [D,4D], [D] */
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
                                       int verbose);
void cs3d_slatdit_gpu_free(cs3d_slatdit_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_SLAT_DIT_GPU_H_ */

/* ============================ implementation ============================ */
#ifdef HIP_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hip_runner_common.h"

static int cs3d_slatditg_upload(const qtensor *t, const char *name,
                                hipDeviceptr_t *out_d, size_t *out_bytes)
{
    if (!t || !t->data) {
        fprintf(stderr, "cs3d_slatdit_gpu: missing tensor %s\n", name);
        return -1;
    }
    float *buf = qt_dequant(t);
    if (!buf) {
        fprintf(stderr, "cs3d_slatdit_gpu: dequant %s failed\n", name);
        return -1;
    }
    int n = qt_numel(t);
    size_t nb = (size_t)n * sizeof(float);
    hipDeviceptr_t d = hip_upload_raw(buf, nb);
    free(buf);
    if (!d) {
        fprintf(stderr, "cs3d_slatdit_gpu: hipMalloc failed for %s (%zu bytes)\n",
                name, nb);
        return -1;
    }
    *out_d = d;
    *out_bytes += nb;
    return 0;
}

int cs3d_slatdit_gpu_load_transformer(cs3d_slatdit_gpu *g,
                                      const sam3d_slat_dit_model *m,
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

    size_t tot = 0;
    for (int b = 0; b < m->n_blocks; b++) {
        const sam3d_slat_block *B = &m->blocks[b];
        cs3d_slatdit_block_w *D = &g->blocks[b];
#define UP_(field) \
        if (cs3d_slatditg_upload(&B->field, #field, &D->field, &tot) < 0) goto fail
        UP_(adaln_w); UP_(adaln_b);
        UP_(norm2_w); UP_(norm2_b);
        UP_(sa_qkv_w); UP_(sa_qkv_b);
        UP_(sa_q_rms_gamma); UP_(sa_k_rms_gamma);
        UP_(sa_out_w); UP_(sa_out_b);
        UP_(xa_q_w); UP_(xa_q_b);
        UP_(xa_kv_w); UP_(xa_kv_b);
        UP_(xa_out_w); UP_(xa_out_b);
        UP_(mlp_fc1_w); UP_(mlp_fc1_b);
        UP_(mlp_fc2_w); UP_(mlp_fc2_b);
#undef UP_
    }

    g->total_bytes = tot;
    g->loaded = 1;
    if (verbose) {
        fprintf(stderr,
                "cs3d_slatdit_gpu: loaded transformer blocks=%d D=%d H=%d D_h=%d "
                "mlp_h=%d cond=%d %.1f MiB on device\n",
                g->n_blocks, g->dim, g->n_heads, g->head_dim,
                g->mlp_hidden, g->cond_channels,
                (double)tot / (1024.0 * 1024.0));
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
            hipDeviceptr_t *all[] = {
                &D->adaln_w, &D->adaln_b,
                &D->norm2_w, &D->norm2_b,
                &D->sa_qkv_w, &D->sa_qkv_b,
                &D->sa_q_rms_gamma, &D->sa_k_rms_gamma,
                &D->sa_out_w, &D->sa_out_b,
                &D->xa_q_w, &D->xa_q_b,
                &D->xa_kv_w, &D->xa_kv_b,
                &D->xa_out_w, &D->xa_out_b,
                &D->mlp_fc1_w, &D->mlp_fc1_b,
                &D->mlp_fc2_w, &D->mlp_fc2_b,
            };
            for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); i++) {
                if (*all[i]) { hipFree(*all[i]); *all[i] = 0; }
            }
        }
        free(g->blocks);
    }
    memset(g, 0, sizeof(*g));
}

#endif /* HIP_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION */
