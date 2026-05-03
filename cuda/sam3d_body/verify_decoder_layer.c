/*
 * verify_decoder_layer (CUDA) — diff each TransformerDecoderLayer's
 * forward against /tmp/sam3d_body_ref/decoder_layer{i}_*.npy. Mirrors
 * the CPU verify_decoder_layer.c at cpu/sam3d_body/.
 */

#include "cuda_sam3d_body_runner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/npy_io.h"

static float *load_named(const char *refdir, const char *name,
                         int want_ndim, const int *want_dims)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    int nd = 0, dims[8] = {0};
    float *p = (float *)npy_load(path, &nd, dims, NULL);
    if (!p) {
        fprintf(stderr, "[cuda verify_decoder_layer] missing %s\n", path);
        return NULL;
    }
    if (nd != want_ndim) {
        fprintf(stderr, "[cuda verify_decoder_layer] %s: rank=%d want %d\n",
                name, nd, want_ndim);
        free(p); return NULL;
    }
    for (int i = 0; i < nd; i++) {
        if (dims[i] != want_dims[i]) {
            fprintf(stderr, "[cuda verify_decoder_layer] %s: dim[%d]=%d want %d\n",
                    name, i, dims[i], want_dims[i]);
            free(p); return NULL;
        }
    }
    return p;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    /* CPU port hits ~5e-4 max_abs vs PyTorch on layer 5. Allow CUDA the
     * same 2e-3 budget — float-vs-float SDPA accumulates similarly. */
    float threshold = 2e-3f;
    float mean_threshold = 3e-4f;
    int one_layer = -1, device = 0, verbose = 0;
    const char *precision = "bf16";
    cuda_sam3d_body_backbone_t backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--mean-threshold") && i+1 < argc) mean_threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--layer") && i+1 < argc) one_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--backbone") && i+1 < argc) {
            const char *v = argv[++i];
            if      (!strcmp(v, "dinov3")) backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;
            else if (!strcmp(v, "vith"))   backbone = CUDA_SAM3D_BODY_BACKBONE_VITH;
            else {
                fprintf(stderr, "unknown --backbone %s (use dinov3|vith)\n", v);
                return 2;
            }
        }
        else if (!strcmp(argv[i], "--device") && i+1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc) precision = argv[++i];
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir DIR --refdir DIR "
                        "[--threshold F] [--mean-threshold F] [--layer N] "
                        "[--backbone dinov3|vith] [--device N] "
                        "[--precision bf16|fp16] [-v]\n",
                argv[0]);
        return 2;
    }

    cuda_sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .image_size      = 512,
        .device_ordinal  = device,
        .verbose         = verbose,
        .precision       = precision,
        .backbone        = backbone,
    };
    cuda_sam3d_body_ctx *ctx = cuda_sam3d_body_create(&cfg);
    if (!ctx) { fprintf(stderr, "create failed\n"); return 5; }

    const int N_Q = 145, D = 1024, DC = 1280;
    int nd0 = 0, dims0[8] = {0};
    char p0[1024];
    snprintf(p0, sizeof(p0), "%s/decoder_layer0_in__context.npy", refdir);
    void *tmp0 = npy_load(p0, &nd0, dims0, NULL);
    if (!tmp0 || nd0 != 3 || dims0[0] != 1 || dims0[2] != DC) {
        fprintf(stderr, "[cuda verify_decoder_layer] bad %s\n", p0);
        free(tmp0);
        cuda_sam3d_body_destroy(ctx);
        return 6;
    }
    const int N_C = dims0[1];
    free(tmp0);

    float *x_out = (float *)malloc((size_t)N_Q * D * sizeof(float));
    if (!x_out) { cuda_sam3d_body_destroy(ctx); return 6; }

    int rc_total = 0;
    int lo = (one_layer >= 0) ? one_layer : 0;
    int hi = (one_layer >= 0) ? one_layer + 1 : 6;

    for (int li = lo; li < hi; li++) {
        char name[64];
        int xdims[3] = {1, N_Q, D};
        int cdims[3] = {1, N_C, DC};

        snprintf(name, sizeof(name), "decoder_layer%d_in__x", li);
        float *x_in = load_named(refdir, name, 3, xdims);
        snprintf(name, sizeof(name), "decoder_layer%d_in__context", li);
        float *c_in = load_named(refdir, name, 3, cdims);
        snprintf(name, sizeof(name), "decoder_layer%d_in__x_pe", li);
        float *xpe = load_named(refdir, name, 3, xdims);
        snprintf(name, sizeof(name), "decoder_layer%d_in__context_pe", li);
        float *cpe = load_named(refdir, name, 3, cdims);
        snprintf(name, sizeof(name), "decoder_layer%d_out__tokens", li);
        float *x_ref = load_named(refdir, name, 3, xdims);
        if (!x_in || !c_in || !xpe || !cpe || !x_ref) {
            free(x_in); free(c_in); free(xpe); free(cpe); free(x_ref);
            rc_total = 3; break;
        }

        int rc = cuda_sam3d_body_debug_run_decoder_layer(
            ctx, li, x_in, c_in, xpe, cpe, N_Q, N_C, x_out);
        if (rc != 0) {
            fprintf(stderr, "[cuda verify_decoder_layer] layer %d: rc=%d\n", li, rc);
            free(x_in); free(c_in); free(xpe); free(cpe); free(x_ref);
            rc_total = 7; break;
        }

        double sum = 0.0; float mx = 0.0f; size_t mx_i = 0;
        size_t nel = (size_t)N_Q * D;
        for (size_t i = 0; i < nel; i++) {
            float d = fabsf(x_out[i] - x_ref[i]);
            if (d > mx) { mx = d; mx_i = i; }
            sum += d;
        }
        double mean = sum / (double)nel;
        int tok = (int)(mx_i / D);
        int dim = (int)(mx_i % D);
        int layer_fail = (mx >= threshold || mean >= mean_threshold);
        fprintf(stderr, "[cuda verify_decoder_layer] layer %d  "
                        "max_abs=%.6e (tok=%d dim=%d)  mean_abs=%.6e  "
                        "(max_gate=%.1e mean_gate=%.1e) %s\n",
                li, mx, tok, dim, mean, threshold, mean_threshold,
                layer_fail ? "FAIL" : "ok");
        if (layer_fail) rc_total = 1;

        free(x_in); free(c_in); free(xpe); free(cpe); free(x_ref);
    }

    free(x_out);
    cuda_sam3d_body_destroy(ctx);
    return rc_total;
}
