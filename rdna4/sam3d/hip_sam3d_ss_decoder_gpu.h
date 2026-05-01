/* SS-VAE decoder device-side weight buffers — Phase 4b.
 *
 * Uploads the TRELLIS.2-compatible sparse-structure decoder weights from the
 * already-loaded CPU model. We keep the CPU model for checkpoint ownership and
 * debug fallback, while this struct owns only dequantized F32 device buffers.
 */

#ifndef HIP_SAM3D_SS_DECODER_GPU_H_
#define HIP_SAM3D_SS_DECODER_GPU_H_

#include <stddef.h>
#include "../rocew.h"
#include "../../common/trellis2_ss_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    hipDeviceptr_t gn1_w, gn1_b;
    hipDeviceptr_t conv1_w, conv1_b;
    hipDeviceptr_t gn2_w, gn2_b;
    hipDeviceptr_t conv2_w, conv2_b;
} cs3d_ssdec_resblock_gpu;

typedef struct {
    hipDeviceptr_t conv_in_w, conv_in_b;
    cs3d_ssdec_resblock_gpu middle[2];
    cs3d_ssdec_resblock_gpu res_16[2];
    hipDeviceptr_t up1_conv_w, up1_conv_b;
    cs3d_ssdec_resblock_gpu res_32[2];
    hipDeviceptr_t up2_conv_w, up2_conv_b;
    cs3d_ssdec_resblock_gpu res_64[2];
    hipDeviceptr_t out_gn_w, out_gn_b;
    hipDeviceptr_t out_conv_w, out_conv_b;
    size_t total_bytes;
    int loaded;
} cs3d_ssdec_gpu;

int  cs3d_ssdec_gpu_load(cs3d_ssdec_gpu *g, const t2_ss_dec *m, int verbose);
void cs3d_ssdec_gpu_free(cs3d_ssdec_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_SS_DECODER_GPU_H_ */

#ifdef HIP_SAM3D_SS_DECODER_GPU_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hip_runner_common.h"

static int cs3d_ssdec_upload(const qtensor *t, const char *name,
                             hipDeviceptr_t *out_d, size_t *out_bytes)
{
    if (!t || !t->data) {
        fprintf(stderr, "cs3d_ssdec_gpu: missing tensor %s\n", name);
        return -1;
    }
    int n = t->n_rows * t->n_cols; /* Works for 5D conv weights. */
    if (n <= 0) {
        fprintf(stderr, "cs3d_ssdec_gpu: bad tensor size for %s\n", name);
        return -1;
    }
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return -1;
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        free(buf);
        fprintf(stderr, "cs3d_ssdec_gpu: unsupported tensor type for %s\n", name);
        return -1;
    }
    size_t nb = (size_t)n * sizeof(float);
    hipDeviceptr_t d = hip_upload_raw(buf, nb);
    free(buf);
    if (!d) {
        fprintf(stderr, "cs3d_ssdec_gpu: alloc %s failed (%zu bytes)\n", name, nb);
        return -1;
    }
    *out_d = d;
    *out_bytes += nb;
    return 0;
}

static int cs3d_ssdec_upload_block(cs3d_ssdec_resblock_gpu *g,
                                   const t2_ss_dec_resblock *b,
                                   const char *name, size_t *tot)
{
    char nm[128];
#define UP_(field) do { \
    snprintf(nm, sizeof(nm), "%s.%s", name, #field); \
    if (cs3d_ssdec_upload(&b->field, nm, &g->field, tot) < 0) return -1; \
} while (0)
    UP_(gn1_w);   UP_(gn1_b);
    UP_(conv1_w); UP_(conv1_b);
    UP_(gn2_w);   UP_(gn2_b);
    UP_(conv2_w); UP_(conv2_b);
#undef UP_
    return 0;
}

int cs3d_ssdec_gpu_load(cs3d_ssdec_gpu *g, const t2_ss_dec *m, int verbose)
{
    if (!g || !m) return -1;
    memset(g, 0, sizeof(*g));
    size_t tot = 0;

#define UP_(field) \
    if (cs3d_ssdec_upload(&m->field, #field, &g->field, &tot) < 0) goto fail
    UP_(conv_in_w);  UP_(conv_in_b);
    if (cs3d_ssdec_upload_block(&g->middle[0], &m->middle[0], "middle.0", &tot) < 0) goto fail;
    if (cs3d_ssdec_upload_block(&g->middle[1], &m->middle[1], "middle.1", &tot) < 0) goto fail;
    if (cs3d_ssdec_upload_block(&g->res_16[0], &m->res_16[0], "res16.0", &tot) < 0) goto fail;
    if (cs3d_ssdec_upload_block(&g->res_16[1], &m->res_16[1], "res16.1", &tot) < 0) goto fail;
    UP_(up1_conv_w); UP_(up1_conv_b);
    if (cs3d_ssdec_upload_block(&g->res_32[0], &m->res_32[0], "res32.0", &tot) < 0) goto fail;
    if (cs3d_ssdec_upload_block(&g->res_32[1], &m->res_32[1], "res32.1", &tot) < 0) goto fail;
    UP_(up2_conv_w); UP_(up2_conv_b);
    if (cs3d_ssdec_upload_block(&g->res_64[0], &m->res_64[0], "res64.0", &tot) < 0) goto fail;
    if (cs3d_ssdec_upload_block(&g->res_64[1], &m->res_64[1], "res64.1", &tot) < 0) goto fail;
    UP_(out_gn_w);   UP_(out_gn_b);
    UP_(out_conv_w); UP_(out_conv_b);
#undef UP_

    g->total_bytes = tot;
    g->loaded = 1;
    if (verbose) {
        fprintf(stderr, "cs3d_ssdec_gpu: loaded %.1f MiB on device\n",
                (double)tot / (1024.0 * 1024.0));
    }
    return 0;

fail:
    cs3d_ssdec_gpu_free(g);
    return -1;
}

static void cs3d_ssdec_free_block(cs3d_ssdec_resblock_gpu *b)
{
    hipDeviceptr_t *ptrs[] = {
        &b->gn1_w, &b->gn1_b, &b->conv1_w, &b->conv1_b,
        &b->gn2_w, &b->gn2_b, &b->conv2_w, &b->conv2_b,
    };
    for (size_t i = 0; i < sizeof(ptrs) / sizeof(ptrs[0]); i++) {
        if (*ptrs[i]) { hipFree(*ptrs[i]); *ptrs[i] = 0; }
    }
}

void cs3d_ssdec_gpu_free(cs3d_ssdec_gpu *g)
{
    if (!g) return;
    hipDeviceptr_t *top[] = {
        &g->conv_in_w, &g->conv_in_b, &g->up1_conv_w, &g->up1_conv_b,
        &g->up2_conv_w, &g->up2_conv_b, &g->out_gn_w, &g->out_gn_b,
        &g->out_conv_w, &g->out_conv_b,
    };
    for (size_t i = 0; i < sizeof(top) / sizeof(top[0]); i++) {
        if (*top[i]) { hipFree(*top[i]); *top[i] = 0; }
    }
    for (int i = 0; i < 2; i++) {
        cs3d_ssdec_free_block(&g->middle[i]);
        cs3d_ssdec_free_block(&g->res_16[i]);
        cs3d_ssdec_free_block(&g->res_32[i]);
        cs3d_ssdec_free_block(&g->res_64[i]);
    }
    memset(g, 0, sizeof(*g));
}

#endif /* HIP_SAM3D_SS_DECODER_GPU_IMPLEMENTATION */
