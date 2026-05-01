/* PointPatchEmbed device-side weight buffers — Phase 2b.8c.
 *
 * Single-header. Define HIP_SAM3D_PPE_GPU_IMPLEMENTATION in exactly
 * one translation unit before including. Mirrors the shape of
 * hip_sam3d_dinov2_gpu.h but for the small PPE module.
 *
 * Loads from an already-parsed `sam3d_ppe_model` (CPU side keeps the
 * mmap and qtensor metadata). Each weight is dequantized to host F32
 * and then uploaded to the device — handles F32 / F16 / Q8_0 / Q4_K /
 * Q6_K transparently via qt_dequant.
 */

#ifndef HIP_SAM3D_PPE_GPU_H_
#define HIP_SAM3D_PPE_GPU_H_

#include <stddef.h>
#include "../rocew.h"
#include "../../common/sam3d_cond_fuser.h"  /* sam3d_ppe_model */
#include "hip_sam3d_ppe_forward.h"          /* cs3d_ppe_w     */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Geometry, copied from sam3d_ppe_model. */
    int Np, P, D, ffn, n_heads, head_dim;
    float ln_eps;

    cs3d_ppe_w w;          /* device weight pointers (used by cs3d_ppe_forward) */

    size_t total_bytes;
    int    loaded;
} cs3d_ppe_gpu;

/* Upload all PPE weights to the device. Returns 0 on success, <0 on
 * failure (any partial allocations are released). `verbose` ≥ 1 prints
 * geometry + total bytes. */
int  cs3d_ppe_gpu_load(cs3d_ppe_gpu *g, const sam3d_ppe_model *m, int verbose);

/* Release every hipDeviceptr_t held by `g` and zero its fields. Safe on
 * an unloaded (zeroed) struct. */
void cs3d_ppe_gpu_free(cs3d_ppe_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_PPE_GPU_H_ */

/* ============================ implementation ============================ */
#ifdef HIP_SAM3D_PPE_GPU_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hip_runner_common.h"

static int cs3d_ppeg_upload(const qtensor *t, const char *name,
                            hipDeviceptr_t *out_d, size_t *out_bytes)
{
    if (!t || !t->data) {
        fprintf(stderr, "cs3d_ppe_gpu: missing tensor %s\n", name);
        return -1;
    }
    float *buf = qt_dequant(t);
    if (!buf) {
        fprintf(stderr, "cs3d_ppe_gpu: dequant %s failed\n", name);
        return -1;
    }
    int n = qt_numel(t);
    size_t nb = (size_t)n * sizeof(float);
    hipDeviceptr_t d = hip_upload_raw(buf, nb);
    free(buf);
    if (!d) {
        fprintf(stderr, "cs3d_ppe_gpu: hipMalloc failed for %s (%zu bytes)\n",
                name, nb);
        return -1;
    }
    *out_d = d;
    *out_bytes += nb;
    return 0;
}

int cs3d_ppe_gpu_load(cs3d_ppe_gpu *g, const sam3d_ppe_model *m, int verbose)
{
    if (!g || !m) return -1;
    memset(g, 0, sizeof(*g));
    g->Np      = m->num_patches;
    g->P       = m->patch_size;
    g->D       = m->embed_dim;
    g->ffn     = m->ffn_hidden;
    g->n_heads = m->n_heads;
    g->head_dim= m->head_dim;
    g->ln_eps  = m->ln_eps;

    cs3d_ppe_w *w = &g->w;
    size_t tot = 0;

#define UP_(field, qt) \
    if (cs3d_ppeg_upload(&(m->qt), #qt, &w->field, &tot) < 0) goto fail
    UP_(point_proj_w,      point_proj_w);
    UP_(point_proj_b,      point_proj_b);
    UP_(invalid_xyz_token, invalid_xyz_token);
    UP_(cls_token,         cls_token);
    UP_(pos_embed_window,  pos_embed_window);
    UP_(pos_embed,         pos_embed);
    UP_(ln1_w,             ln1_w);
    UP_(ln1_b,             ln1_b);
    UP_(qkv_w,             attn_qkv_w);
    UP_(qkv_b,             attn_qkv_b);
    UP_(proj_w,            attn_proj_w);
    UP_(proj_b,            attn_proj_b);
    UP_(ln2_w,             ln2_w);
    UP_(ln2_b,             ln2_b);
    UP_(fc1_w,             mlp_fc1_w);
    UP_(fc1_b,             mlp_fc1_b);
    UP_(fc2_w,             mlp_fc2_w);
    UP_(fc2_b,             mlp_fc2_b);
#undef UP_

    g->total_bytes = tot;
    g->loaded = 1;
    if (verbose) {
        fprintf(stderr, "cs3d_ppe_gpu: loaded Np=%d P=%d D=%d ffn=%d H=%d D_h=%d  "
                        "%.1f MiB on device\n",
                g->Np, g->P, g->D, g->ffn, g->n_heads, g->head_dim,
                (double)tot / (1024.0 * 1024.0));
    }
    return 0;

fail:
    cs3d_ppe_gpu_free(g);
    return -1;
}

void cs3d_ppe_gpu_free(cs3d_ppe_gpu *g)
{
    if (!g) return;
    cs3d_ppe_w *w = &g->w;
    hipDeviceptr_t *all[] = {
        &w->point_proj_w, &w->point_proj_b, &w->invalid_xyz_token,
        &w->cls_token, &w->pos_embed_window, &w->pos_embed,
        &w->ln1_w, &w->ln1_b, &w->qkv_w, &w->qkv_b,
        &w->proj_w, &w->proj_b, &w->ln2_w, &w->ln2_b,
        &w->fc1_w, &w->fc1_b, &w->fc2_w, &w->fc2_b,
    };
    for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); i++) {
        if (*all[i]) { hipFree(*all[i]); *all[i] = 0; }
    }
    memset(g, 0, sizeof(*g));
}

#endif /* HIP_SAM3D_PPE_GPU_IMPLEMENTATION */
