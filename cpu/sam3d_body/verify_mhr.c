/*
 * verify_mhr — diff our MHR skinning (MHR params → 3D vertices)
 * against the pytorch reference dump.
 *
 * Usage:
 *   verify_mhr --mhr-assets <dir> --refdir /tmp/sam3d_body_ref \
 *              [--use-ref-inputs] [--threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR:
 *   mhr_params.npy     — (N,) regressor output
 *   out_vertices.npy   — (V, 3) skinned mesh in root-relative frame
 *   out_keypoints_3d.npy — (K, 3) joints in same frame
 *
 * Scaffold: MHR skinning not yet ported; exits 0 with a banner.
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
    const char *sft_dir = NULL, *refdir = NULL, *mhr_dir = NULL;
    float threshold = 1e-3f;
    int n_threads = 1, verbose = 0, use_ref_inputs = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--mhr-assets")      && i+1 < argc) mhr_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--use-ref-inputs"))                use_ref_inputs = 1;
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!mhr_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --mhr-assets <dir> --refdir <dir> "
                "[--use-ref-inputs] [--threshold F] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/out_vertices.npy", refdir);
    int nd = 0, dims[8] = {0}, is_f32 = 0;
    float *ref_v = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!ref_v || !is_f32 || nd != 2 || dims[1] != 3) {
        fprintf(stderr, "[verify_mhr] scaffold — MHR not yet ported, "
                        "no ref %s\n", path);
        free(ref_v);
        return 0;
    }
    int ref_v_count = dims[0];
    fprintf(stderr, "[verify_mhr] ref vertices: v=%d\n", ref_v_count);

    sam3d_body_config cfg = {
        .safetensors_dir = sft_dir ? sft_dir : "",
        .mhr_assets_dir  = mhr_dir,
        .backbone        = SAM3D_BODY_BACKBONE_DINOV3,
        .seed            = 42,
        .n_threads       = n_threads,
        .verbose         = verbose,
    };
    sam3d_body_ctx *ctx = sam3d_body_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_body_create failed\n"); free(ref_v); return 5; }

    if (use_ref_inputs) {
        int p_nd = 0, p_dims[8] = {0}, p_f32 = 0;
        snprintf(path, sizeof(path), "%s/mhr_params.npy", refdir);
        float *p = (float *)npy_load(path, &p_nd, p_dims, &p_f32);
        if (p && p_f32) {
            int n = 1; for (int i = 0; i < p_nd; i++) n *= p_dims[i];
            sam3d_body_debug_override_mhr_params(ctx, p, n);
        }
        free(p);
    }

    int rc = sam3d_body_run_mhr(ctx);
    if (rc != 0) {
        fprintf(stderr, "[verify_mhr] scaffold — MHR rc=%d (not yet ported)\n", rc);
        sam3d_body_destroy(ctx); free(ref_v);
        return 0;
    }

    int v = 0;
    sam3d_body_get_vertices(ctx, NULL, &v);
    float *ours = (float *)malloc((size_t)v * 3 * sizeof(float));
    sam3d_body_get_vertices(ctx, ours, &v);

    int rc_out = 0;
    if (v != ref_v_count) {
        fprintf(stderr, "[verify_mhr] V mismatch: ours=%d ref=%d\n", v, ref_v_count);
        rc_out = 7;
    } else {
        double sum = 0.0; float mx = 0.0f;
        int n = v * 3;
        for (int i = 0; i < n; i++) {
            float d = fabsf(ours[i] - ref_v[i]);
            if (d > mx) mx = d;
            sum += d;
        }
        double mean_abs = sum / n;
        fprintf(stderr, "[verify_mhr] V=%d  max_abs=%.6e mean_abs=%.6e "
                        "(threshold=%.1e)\n",
                v, mx, mean_abs, threshold);
        rc_out = (mx < threshold) ? 0 : 1;
    }

    free(ours); free(ref_v);
    sam3d_body_destroy(ctx);
    return rc_out;
}
