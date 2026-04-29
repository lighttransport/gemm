/*
 * verify_slat_dit (CUDA) — diff a single SLAT Flow DiT forward call against
 * /tmp/sam3d_ref/slat_dit_out_feats.npy. Inputs (coords, feats, cond, t)
 * are loaded directly from refs and passed via
 * cuda_sam3d_debug_slat_dit_forward, isolating per-call DiT drift from
 * upstream encoder/fuser/decoder drift.
 *
 * Usage:
 *   verify_slat_dit --safetensors-dir DIR --refdir /tmp/sam3d_ref
 *                   [--threshold F] [-v]
 */

#include "cuda_sam3d_runner.h"
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
    float threshold = 5e-2f;  /* bf16 inference-path floor */

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

    cuda_sam3d_config cfg = {0};
    cfg.safetensors_dir = sft_dir;
    cfg.pipeline_yaml   = yaml;
    cfg.verbose         = verbose;
    cfg.precision       = "fp16";
    cfg.seed            = 42;
    cuda_sam3d_ctx *ctx = cuda_sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "cuda_sam3d_create failed\n"); return 5; }

    int in_ch = 0, out_ch = 0, cond_ch = 0;
    if (cuda_sam3d_slat_dit_info(ctx, &in_ch, &out_ch, &cond_ch) != 0) {
        fprintf(stderr, "cuda_sam3d_slat_dit_info failed\n");
        cuda_sam3d_destroy(ctx); return 3;
    }
    fprintf(stderr, "[verify_slat_dit.cuda] OK: in_ch=%d out_ch=%d cond_ch=%d\n",
            in_ch, out_ch, cond_ch);

    char path[1024];
    int nd = 0, dims[8] = {0}, is_f32 = 0;

    snprintf(path, sizeof(path), "%s/slat_dit_in_coords.npy", refdir);
    int32_t *coords = (int32_t *)npy_load(path, &nd, dims, &is_f32);
    if (!coords || nd != 2 || dims[1] != 4) {
        fprintf(stderr, "[verify_slat_dit.cuda] bad %s\n", path);
        free(coords); cuda_sam3d_destroy(ctx); return 4;
    }
    int N = dims[0];

    snprintf(path, sizeof(path), "%s/slat_dit_in_feats.npy", refdir);
    float *feats = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!feats || !is_f32 || nd != 2 || dims[0] != N || dims[1] != in_ch) {
        fprintf(stderr, "[verify_slat_dit.cuda] bad %s\n", path);
        free(coords); free(feats); cuda_sam3d_destroy(ctx); return 4;
    }

    snprintf(path, sizeof(path), "%s/slat_dit_cond.npy", refdir);
    float *cond = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!cond || !is_f32 || nd != 3 || dims[2] != cond_ch) {
        fprintf(stderr, "[verify_slat_dit.cuda] bad %s\n", path);
        free(coords); free(feats); free(cond);
        cuda_sam3d_destroy(ctx); return 4;
    }
    int n_cond = dims[1];

    snprintf(path, sizeof(path), "%s/slat_dit_t.npy", refdir);
    int t_nd = 0, t_dims[8] = {0}, t_is_f32 = 0;
    float *tnp = (float *)npy_load(path, &t_nd, t_dims, &t_is_f32);
    if (!tnp || !t_is_f32) {
        fprintf(stderr, "[verify_slat_dit.cuda] bad %s\n", path);
        free(coords); free(feats); free(cond); free(tnp);
        cuda_sam3d_destroy(ctx); return 4;
    }
    float t_val = tnp[0];
    free(tnp);

    snprintf(path, sizeof(path), "%s/slat_dit_out_feats.npy", refdir);
    float *ref_out = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!ref_out || !is_f32 || nd != 2 || dims[0] != N || dims[1] != out_ch) {
        fprintf(stderr, "[verify_slat_dit.cuda] bad %s\n", path);
        free(coords); free(feats); free(cond); free(ref_out);
        cuda_sam3d_destroy(ctx); return 4;
    }

    fprintf(stderr,
            "[verify_slat_dit.cuda] forward: N=%d n_cond=%d t=%.4f\n",
            N, n_cond, (double)t_val);

    float *out = (float *)calloc((size_t)N * out_ch, sizeof(float));
    if (!out) {
        free(coords); free(feats); free(cond); free(ref_out);
        cuda_sam3d_destroy(ctx); return 5;
    }

    int rc = 0;
    if (cuda_sam3d_debug_slat_dit_forward(ctx, coords, feats, N, t_val,
                                          cond, n_cond, out) != 0) {
        fprintf(stderr, "cuda_sam3d_debug_slat_dit_forward failed\n");
        rc = 6;
        goto cleanup;
    }

    double mean_abs = 0.0;
    float mx = npy_max_abs_f32(out, ref_out, N * out_ch, &mean_abs);
    int ok = (mx < threshold);
    fprintf(stderr,
            "[verify_slat_dit.cuda] out_feats max_abs=%.6e mean_abs=%.6e n=%d  %s "
            "(threshold %.1e)\n",
            (double)mx, mean_abs, N * out_ch, ok ? "OK" : "FAIL",
            (double)threshold);
    if (!ok) rc = 1;

cleanup:
    free(out); free(coords); free(feats); free(cond); free(ref_out);
    cuda_sam3d_destroy(ctx);
    return rc;
}
