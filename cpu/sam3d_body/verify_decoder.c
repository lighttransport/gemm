/*
 * verify_decoder — diff our promptable decoder + MHR head (regressor)
 * output against the pytorch reference dump.
 *
 * Usage:
 *   verify_decoder --safetensors-dir <dir> --refdir /tmp/sam3d_body_ref \
 *                  [--use-ref-inputs] [--threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR:
 *   dinov3_tokens.npy  — upstream stage output (for --use-ref-inputs)
 *   mhr_params.npy     — decoder/MHR-head regression output (N floats)
 *
 * Scaffold: decoder not yet ported; exits 0 with a banner.
 */

#include "sam3d_body_runner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npy_io.h"

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    float threshold = 1e-3f;
    int n_threads = 1, verbose = 0, use_ref_inputs = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--use-ref-inputs")) use_ref_inputs = 1;
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--use-ref-inputs] [--threshold F] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/mhr_params.npy", refdir);
    int nd = 0, dims[8] = {0}, is_f32 = 0;
    float *ref = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!ref || !is_f32) {
        fprintf(stderr, "[verify_decoder] scaffold — decoder not yet ported, "
                        "no ref %s\n", path);
        free(ref);
        return 0;
    }
    int ref_n = 1;
    for (int i = 0; i < nd; i++) ref_n *= dims[i];
    fprintf(stderr, "[verify_decoder] ref mhr_params: n=%d\n", ref_n);

    sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .backbone        = SAM3D_BODY_BACKBONE_DINOV3,
        .seed            = 42,
        .n_threads       = n_threads,
        .verbose         = verbose,
    };
    sam3d_body_ctx *ctx = sam3d_body_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_body_create failed\n"); free(ref); return 5; }

    if (use_ref_inputs) {
        int t_nd = 0, t_dims[8] = {0}, t_f32 = 0;
        snprintf(path, sizeof(path), "%s/dinov3_tokens.npy", refdir);
        float *toks = (float *)npy_load(path, &t_nd, t_dims, &t_f32);
        if (toks && t_f32 && t_nd >= 2) {
            int nt = t_dims[t_nd == 3 ? 1 : 0];
            int dm = t_dims[t_nd == 3 ? 2 : 1];
            sam3d_body_debug_override_encoder(ctx, toks, nt, dm);
        }
        free(toks);
    }

    int rc = sam3d_body_run_decoder(ctx);
    if (rc != 0) {
        fprintf(stderr, "[verify_decoder] scaffold — decoder rc=%d (not yet "
                        "ported); ref n=%d\n", rc, ref_n);
        sam3d_body_destroy(ctx); free(ref);
        return 0;
    }

    int out_n = 0;
    sam3d_body_get_mhr_params(ctx, NULL, &out_n);
    float *ours = (float *)malloc((size_t)out_n * sizeof(float));
    sam3d_body_get_mhr_params(ctx, ours, &out_n);

    int rc_out = 0;
    if (out_n != ref_n) {
        fprintf(stderr, "[verify_decoder] size mismatch: ours=%d ref=%d\n",
                out_n, ref_n);
        rc_out = 7;
    } else {
        double sum = 0.0; float mx = 0.0f;
        for (int i = 0; i < out_n; i++) {
            float d = fabsf(ours[i] - ref[i]);
            if (d > mx) mx = d;
            sum += d;
        }
        double mean_abs = sum / out_n;
        float mean_gate = threshold * 1e-1f;
        fprintf(stderr, "[verify_decoder] n=%d  max_abs=%.6e mean_abs=%.6e "
                        "(max=%.1e mean=%.1e)\n",
                out_n, mx, mean_abs, threshold, mean_gate);
        rc_out = (mx < threshold && mean_abs < mean_gate) ? 0 : 1;
    }

    free(ours); free(ref);
    sam3d_body_destroy(ctx);
    return rc_out;
}
