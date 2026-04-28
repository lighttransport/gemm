/*
 * verify_slat_gs (CUDA) — diff a single SLAT GS decoder forward (transformer
 * + to_representation) against /tmp/sam3d_ref/slat_gs_*.npy. Inputs (sparse
 * coords + feats) are loaded directly from refs and passed via
 * cuda_sam3d_debug_slat_gs_*, isolating per-call decoder drift from
 * upstream stages.
 *
 * Usage:
 *   verify_slat_gs --safetensors-dir DIR --refdir /tmp/sam3d_ref
 *                  [--threshold F] [-v]
 */

#include "cuda_sam3d_runner.h"
#include "../../common/npy_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int diff_and_report(const char *label, const float *a, const float *b,
                           int n, float threshold)
{
    double mean = 0.0;
    float mx = npy_max_abs_f32(a, b, n, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr, "[verify_slat_gs.cuda] %-30s max_abs=%.4g mean_abs=%.4g n=%d %s\n",
            label, (double)mx, mean, n, ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *yaml = NULL, *refdir = NULL;
    int verbose = 0;
    float threshold = 5e-3f;

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

    int in_ch = 0, out_ch = 0, G = 0;
    if (cuda_sam3d_slat_gs_info(ctx, &in_ch, &out_ch, &G) != 0) {
        fprintf(stderr, "cuda_sam3d_slat_gs_info failed\n");
        cuda_sam3d_destroy(ctx); return 3;
    }
    fprintf(stderr,
            "[verify_slat_gs.cuda] OK: in_ch=%d out_ch=%d num_gaussians=%d\n",
            in_ch, out_ch, G);

    char path[1024];
    int nd = 0, dims[8] = {0}, is_f32 = 0;

    snprintf(path, sizeof(path), "%s/slat_gs_in_coords.npy", refdir);
    int32_t *coords = (int32_t *)npy_load(path, &nd, dims, &is_f32);
    if (!coords || nd != 2 || dims[1] != 4) {
        fprintf(stderr, "[verify_slat_gs.cuda] bad %s\n", path);
        free(coords); cuda_sam3d_destroy(ctx); return 4;
    }
    int N = dims[0];

    snprintf(path, sizeof(path), "%s/slat_gs_in_feats.npy", refdir);
    float *feats = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!feats || !is_f32 || nd != 2 || dims[0] != N || dims[1] != in_ch) {
        fprintf(stderr, "[verify_slat_gs.cuda] bad %s\n", path);
        free(coords); free(feats); cuda_sam3d_destroy(ctx); return 4;
    }

    fprintf(stderr, "[verify_slat_gs.cuda] forward: N=%d in_ch=%d\n", N, in_ch);

    int rc = 0;
    float *out_feats = NULL;
    int    out_c     = 0;
    if (cuda_sam3d_debug_slat_gs_transformer(ctx, coords, feats, N,
                                             &out_feats, &out_c) != 0) {
        fprintf(stderr, "cuda_sam3d_debug_slat_gs_transformer failed\n");
        rc = 6;
        goto cleanup_io;
    }
    if (out_c != out_ch) {
        fprintf(stderr, "out_c mismatch: got %d want %d\n", out_c, out_ch);
        rc = 7;
        goto cleanup_out;
    }

    snprintf(path, sizeof(path), "%s/slat_gs_out_feats.npy", refdir);
    float *ref_out = (float *)npy_load(path, &nd, dims, &is_f32);
    if (ref_out && is_f32 && nd == 2 && dims[0] == N && dims[1] == out_c) {
        if (!diff_and_report("transformer out_feats", out_feats, ref_out,
                             N * out_c, threshold)) rc = 9;
    } else {
        fprintf(stderr, "[verify_slat_gs.cuda] cannot read %s\n", path);
    }
    free(ref_out);

    int total = N * G;
    float *xyz = (float *)malloc((size_t)total * 3 * sizeof(float));
    float *dc  = (float *)malloc((size_t)total * 3 * sizeof(float));
    float *scl = (float *)malloc((size_t)total * 3 * sizeof(float));
    float *rot = (float *)malloc((size_t)total * 4 * sizeof(float));
    float *op  = (float *)malloc((size_t)total     * sizeof(float));
    if (!xyz || !dc || !scl || !rot || !op) {
        free(xyz); free(dc); free(scl); free(rot); free(op);
        rc = 5; goto cleanup_out;
    }
    if (cuda_sam3d_debug_slat_gs_to_representation(ctx, coords, out_feats, N,
                                                   xyz, dc, scl, rot, op) != 0) {
        fprintf(stderr, "to_representation failed\n");
        free(xyz); free(dc); free(scl); free(rot); free(op);
        rc = 8; goto cleanup_out;
    }

    struct { const char *name; const char *label; const float *buf; int elts; } diffs[] = {
        {"slat_gs_rep_xyz.npy",      "rep_xyz",      xyz, total * 3},
        {"slat_gs_rep_dc.npy",       "rep_dc",       dc,  total * 3},
        {"slat_gs_rep_scaling.npy",  "rep_scaling",  scl, total * 3},
        {"slat_gs_rep_rotation.npy", "rep_rotation", rot, total * 4},
        {"slat_gs_rep_opacity.npy",  "rep_opacity",  op,  total},
    };
    for (size_t i = 0; i < sizeof(diffs)/sizeof(diffs[0]); i++) {
        snprintf(path, sizeof(path), "%s/%s", refdir, diffs[i].name);
        float *ref_buf = (float *)npy_load(path, &nd, dims, &is_f32);
        if (!ref_buf || !is_f32) {
            fprintf(stderr, "[verify_slat_gs.cuda] cannot read %s\n", path);
            free(ref_buf);
            continue;
        }
        if (!diff_and_report(diffs[i].label, diffs[i].buf, ref_buf,
                             diffs[i].elts, threshold)) rc = 9;
        free(ref_buf);
    }

    free(xyz); free(dc); free(scl); free(rot); free(op);

cleanup_out:
    free(out_feats);
cleanup_io:
    free(coords); free(feats);
    cuda_sam3d_destroy(ctx);
    return rc;
}
