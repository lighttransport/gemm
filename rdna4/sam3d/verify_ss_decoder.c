/*
 * verify_ss_decoder (CUDA) — diff SS-VAE 3D-conv decoder output (64³
 * occupancy logits) against /tmp/sam3d_ref/ss_dec_out.npy. ss_latent
 * is injected via hip_sam3d_debug_override_ss_latent so this stage's
 * drift is isolated from upstream DiT drift.
 *
 * Usage:
 *   verify_ss_decoder --safetensors-dir DIR --refdir /tmp/sam3d_ref
 *                     [--threshold F] [-v]
 */

#include "hip_sam3d_runner.h"
#include "../../common/npy_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *yaml = NULL, *refdir = NULL;
    int verbose = 0;
    float threshold = 1e-3f;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir   = argv[++i];
        else if (!strcmp(a, "--pipeline-yaml")   && i+1 < argc) yaml      = argv[++i];
        else if (!strcmp(a, "--refdir")          && i+1 < argc) refdir    = argv[++i];
        else if (!strcmp(a, "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                              verbose   = 1;
        else if (!strcmp(a, "--use-ref-inputs"))                { /* implied */ }
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if ((!sft_dir && !yaml) || !refdir) {
        fprintf(stderr,
                "Usage: %s (--safetensors-dir DIR | --pipeline-yaml YAML) "
                "--refdir DIR [--threshold F] [-v]\n", argv[0]);
        return 2;
    }

    /* ss_dec_in.npy: accept [8,16,16,16], [16,16,16,8], or [4096,8]. */
    char path[1024];
    snprintf(path, sizeof(path), "%s/ss_dec_in.npy", refdir);
    int nd = 0, dims[8] = {0}, is_f32 = 0;
    float *in = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!in || !is_f32) {
        fprintf(stderr, "missing/bad %s\n", path);
        free(in); return 4;
    }
    int in_n = 1;
    for (int i = 0; i < nd; i++) in_n *= dims[i];
    if (in_n != 8 * 16 * 16 * 16) {
        fprintf(stderr, "bad ss_dec_in numel=%d (want 32768)\n", in_n);
        free(in); return 4;
    }
    float *nchw = (float *)malloc((size_t)in_n * sizeof(float));
    if (!nchw) { free(in); return 5; }
    if (nd == 2 && dims[0] == 4096 && dims[1] == 8) {
        for (int n = 0; n < 4096; n++)
            for (int c = 0; c < 8; c++)
                nchw[c * 4096 + n] = in[n * 8 + c];
    } else if (nd == 4 && dims[0] == 16 && dims[1] == 16 && dims[2] == 16 && dims[3] == 8) {
        for (int n = 0; n < 4096; n++)
            for (int c = 0; c < 8; c++)
                nchw[c * 4096 + n] = in[n * 8 + c];
    } else {
        memcpy(nchw, in, (size_t)in_n * sizeof(float));
    }
    free(in);

    /* ss_dec_out.npy: 64³ occupancy logits. */
    snprintf(path, sizeof(path), "%s/ss_dec_out.npy", refdir);
    int rnd = 0, rdims[8] = {0}, rf32 = 0;
    float *ref = (float *)npy_load(path, &rnd, rdims, &rf32);
    if (!ref || !rf32) {
        fprintf(stderr, "missing/bad %s\n", path);
        free(nchw); free(ref); return 4;
    }
    int ref_n = 1;
    for (int i = 0; i < rnd; i++) ref_n *= rdims[i];
    if (ref_n != 64 * 64 * 64) {
        fprintf(stderr, "bad ss_dec_out numel=%d (want 262144)\n", ref_n);
        free(nchw); free(ref); return 4;
    }

    hip_sam3d_config cfg = {0};
    cfg.safetensors_dir = sft_dir;
    cfg.pipeline_yaml   = yaml;
    cfg.verbose         = verbose;
    cfg.precision       = "fp16";
    cfg.seed            = 42;
    hip_sam3d_ctx *ctx = hip_sam3d_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "hip_sam3d_create failed\n");
        free(nchw); free(ref); return 5;
    }

    const int lat_dims[4] = {8, 16, 16, 16};
    if (hip_sam3d_debug_override_ss_latent(ctx, nchw, lat_dims) != 0) {
        fprintf(stderr, "debug_override_ss_latent failed\n");
        hip_sam3d_destroy(ctx); free(nchw); free(ref); return 5;
    }
    free(nchw);

    if (hip_sam3d_run_ss_decode(ctx) != 0) {
        fprintf(stderr, "hip_sam3d_run_ss_decode failed\n");
        hip_sam3d_destroy(ctx); free(ref); return 6;
    }

    int out_dims[3] = {0};
    hip_sam3d_get_occupancy(ctx, NULL, out_dims);
    size_t on = (size_t)out_dims[0] * out_dims[1] * out_dims[2];
    float *out = (float *)malloc(on * sizeof(float));
    if (!out) { hip_sam3d_destroy(ctx); free(ref); return 5; }
    hip_sam3d_get_occupancy(ctx, out, out_dims);
    fprintf(stderr, "[verify_ss_decoder.cuda] occupancy: [%d,%d,%d]\n",
            out_dims[0], out_dims[1], out_dims[2]);

    int rc = 0;
    if ((int)on != ref_n) {
        fprintf(stderr, "size mismatch: ours=%zu ref=%d\n", on, ref_n);
        rc = 7;
    } else {
        double mean_abs = 0.0;
        float mx = npy_max_abs_f32(out, ref, ref_n, &mean_abs);
        fprintf(stderr, "[verify_ss_decoder.cuda] 64³  max_abs=%.6e  mean_abs=%.6e  "
                        "(threshold %.1e)\n",
                (double)mx, mean_abs, threshold);
        rc = (mx < threshold) ? 0 : 1;
    }

    free(out); free(ref);
    hip_sam3d_destroy(ctx);
    return rc;
}
