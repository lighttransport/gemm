/* Fuser projection device-side weight buffers — Phase 2b.8e.
 *
 * Single-header. Define HIP_SAM3D_FUSER_GPU_IMPLEMENTATION in exactly
 * one TU before including. Mirrors the hip_sam3d_ppe_gpu.h shape but
 * stores N modalities × {ln, w1, w2, w3} + idx_emb.
 *
 * Each weight is dequantized to host F32 then uploaded to the device,
 * so any quant type the parent fuser_model holds (F32/F16/Q8_0/Q4_K/...)
 * is handled transparently via qt_dequant.
 */

#ifndef HIP_SAM3D_FUSER_GPU_H_
#define HIP_SAM3D_FUSER_GPU_H_

#include <stddef.h>
#include "../rocew.h"
#include "../../common/sam3d_cond_fuser.h"
#include "hip_sam3d_fuser_forward.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CS3D_FUSER_MAX_MODALITIES 4

typedef struct {
    int n_modalities;
    int Do;
    cs3d_fuser_proj_w projs[CS3D_FUSER_MAX_MODALITIES];

    /* Per-pos-group rows of the CPU's idx_emb_f32 cache, uploaded once
     * each. NULL row → no positional add. */
    int n_pos;
    hipDeviceptr_t idx_emb_rows[8];

    size_t total_bytes;
    int    loaded;
} cs3d_fuser_gpu;

int  cs3d_fuser_gpu_load(cs3d_fuser_gpu *g, const sam3d_fuser_model *m, int verbose);
void cs3d_fuser_gpu_free(cs3d_fuser_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_FUSER_GPU_H_ */

/* ============================ implementation ============================ */
#ifdef HIP_SAM3D_FUSER_GPU_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hip_runner_common.h"

static int cs3d_fuserg_upload(const qtensor *t, const char *name,
                              hipDeviceptr_t *out_d, size_t *out_bytes)
{
    if (!t || !t->data) {
        fprintf(stderr, "cs3d_fuser_gpu: missing tensor %s\n", name);
        return -1;
    }
    float *buf = qt_dequant(t);
    if (!buf) { fprintf(stderr, "cs3d_fuser_gpu: dequant %s failed\n", name); return -1; }
    int n = qt_numel(t);
    size_t nb = (size_t)n * sizeof(float);
    hipDeviceptr_t d = hip_upload_raw(buf, nb);
    free(buf);
    if (!d) { fprintf(stderr, "cs3d_fuser_gpu: alloc %s failed\n", name); return -1; }
    *out_d = d; *out_bytes += nb;
    return 0;
}

int cs3d_fuser_gpu_load(cs3d_fuser_gpu *g, const sam3d_fuser_model *m, int verbose)
{
    if (!g || !m) return -1;
    memset(g, 0, sizeof(*g));
    g->n_modalities = m->n_modalities;
    g->Do = m->embed_dim_out;
    if (g->n_modalities > CS3D_FUSER_MAX_MODALITIES) return -1;
    size_t tot = 0;

    for (int i = 0; i < g->n_modalities; i++) {
        const sam3d_fuser_projection *p = &m->projs[i];
        cs3d_fuser_proj_w *gp = &g->projs[i];
        gp->Di = p->embed_dim_in;
        gp->Dh = p->ffn_hidden;
        gp->Do = p->embed_dim_out;
        char name[64];
#define UP_(field, qt) \
        snprintf(name, sizeof(name), "mod%d.%s", i, #qt); \
        if (cs3d_fuserg_upload(&p->qt, name, &gp->field, &tot) < 0) goto fail
        UP_(ln_w, ln_w);
        UP_(ln_b, ln_b);
        UP_(w1,   w1);
        UP_(w2,   w2);
        UP_(w3,   w3);
#undef UP_
    }

    /* Upload idx_emb rows individually so callers can pass exactly the
     * row pointer to gemm_f32_bias's bias arg. */
    int n_pos = (int)m->idx_emb.dims[0];
    if (n_pos > (int)(sizeof(g->idx_emb_rows) / sizeof(g->idx_emb_rows[0]))) {
        fprintf(stderr, "cs3d_fuser_gpu: too many pos groups (%d)\n", n_pos);
        goto fail;
    }
    g->n_pos = n_pos;
    for (int p = 0; p < n_pos; p++) {
        const float *row = m->idx_emb_f32 + (size_t)p * g->Do;
        size_t nb = (size_t)g->Do * sizeof(float);
        hipDeviceptr_t d = hip_upload_raw(row, nb);
        if (!d) { fprintf(stderr, "cs3d_fuser_gpu: alloc idx_emb row %d\n", p); goto fail; }
        g->idx_emb_rows[p] = d;
        tot += nb;
    }

    g->total_bytes = tot;
    g->loaded = 1;
    if (verbose) {
        fprintf(stderr, "cs3d_fuser_gpu: loaded n_mod=%d Do=%d n_pos=%d  %.1f MiB on device\n",
                g->n_modalities, g->Do, g->n_pos,
                (double)tot / (1024.0 * 1024.0));
    }
    return 0;

fail:
    cs3d_fuser_gpu_free(g);
    return -1;
}

void cs3d_fuser_gpu_free(cs3d_fuser_gpu *g)
{
    if (!g) return;
    for (int i = 0; i < g->n_modalities; i++) {
        cs3d_fuser_proj_w *gp = &g->projs[i];
        if (gp->ln_w) hipFree(gp->ln_w);
        if (gp->ln_b) hipFree(gp->ln_b);
        if (gp->w1)   hipFree(gp->w1);
        if (gp->w2)   hipFree(gp->w2);
        if (gp->w3)   hipFree(gp->w3);
    }
    for (int p = 0; p < g->n_pos; p++) {
        if (g->idx_emb_rows[p]) hipFree(g->idx_emb_rows[p]);
    }
    memset(g, 0, sizeof(*g));
}

#endif /* HIP_SAM3D_FUSER_GPU_IMPLEMENTATION */
