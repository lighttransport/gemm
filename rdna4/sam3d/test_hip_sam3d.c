/* test_hip_sam3d — CLI for the CUDA SAM 3D Objects runner.
 *
 * Phase 7a: end-to-end image → splat.ply. Mirrors the CPU runner CLI in
 * cpu/sam3d/test_sam3d.c so a single fujisan.jpg + pointmap.npy
 * invocation drives the full six-stage pipeline (currently CPU
 * fallback under the hood; NVRTC kernels swap in per stage without
 * touching this file).
 *
 * Usage:
 *   test_hip_sam3d [--safetensors-dir DIR] [--pipeline-yaml YAML]
 *                   <image.png> <mask.png>
 *                   [--pointmap pmap.npy] [--slat-ref <dir>]
 *                   [--seed N] [--steps N] [--slat-steps N]
 *                   [--cfg F] [-o splat.ply]
 *                   [--device N] [--precision fp16|bf16|fp32] [-v|-vv]
 *
 * --slat-ref bypasses upstream stages by loading
 * slat_dit_out_{coords,feats}.npy (or slat_gs_in_*) and runs only
 * SLAT-GS-decode + PLY write.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#define GS_PLY_WRITER_IMPLEMENTATION
#include "../../common/gs_ply_writer.h"

#include "../../common/npy_io.h"
#include "hip_sam3d_runner.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    hip_sam3d_config cfg = {0};
    cfg.device_ordinal = 0;
    cfg.verbose        = 0;
    cfg.precision      = "fp16";
    cfg.seed           = 42;
    cfg.ss_steps       = 2;
    cfg.slat_steps     = 12;
    cfg.cfg_scale      = 2.0f;

    const char *image_path    = NULL;
    const char *mask_path     = NULL;
    const char *pointmap_path = NULL;
    const char *slat_ref_dir  = NULL;
    const char *out_path      = "splat.ply";

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) cfg.safetensors_dir = argv[++i];
        else if (!strcmp(a, "--pipeline-yaml")   && i+1 < argc) cfg.pipeline_yaml   = argv[++i];
        else if (!strcmp(a, "--device")          && i+1 < argc) cfg.device_ordinal  = atoi(argv[++i]);
        else if (!strcmp(a, "--precision")       && i+1 < argc) cfg.precision       = argv[++i];
        else if (!strcmp(a, "--pointmap")        && i+1 < argc) pointmap_path       = argv[++i];
        else if (!strcmp(a, "--slat-ref")        && i+1 < argc) slat_ref_dir        = argv[++i];
        else if (!strcmp(a, "--seed")            && i+1 < argc) cfg.seed            = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--steps")           && i+1 < argc) cfg.ss_steps        = atoi(argv[++i]);
        else if (!strcmp(a, "--slat-steps")      && i+1 < argc) cfg.slat_steps      = atoi(argv[++i]);
        else if (!strcmp(a, "--cfg")             && i+1 < argc) cfg.cfg_scale       = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-o")                && i+1 < argc) out_path            = argv[++i];
        else if (!strcmp(a, "-v"))                              cfg.verbose         = 1;
        else if (!strcmp(a, "-vv"))                             cfg.verbose         = 2;
        else if (a[0] != '-') {
            if      (positional == 0) image_path = a;
            else if (positional == 1) mask_path  = a;
            positional++;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            fprintf(stderr,
                    "Usage: %s [--safetensors-dir DIR | --pipeline-yaml YAML] "
                    "<image.png> <mask.png> "
                    "[--pointmap pmap.npy] [--slat-ref <dir>] "
                    "[--seed N] [--steps N] [--slat-steps N] [--cfg F] "
                    "[-o splat.ply] [--device N] [--precision fp16|bf16|fp32] "
                    "[-v|-vv]\n",
                    argv[0]);
            return 2;
        }
    }

    if (!cfg.safetensors_dir && !cfg.pipeline_yaml) {
        fprintf(stderr,
                "[test_hip_sam3d] need --safetensors-dir or --pipeline-yaml.\n"
                "  Default layout: $MODELS/sam3d/safetensors/ next to "
                "$MODELS/sam3d/checkpoints/pipeline.yaml\n");
        return 2;
    }
    if (!slat_ref_dir && (!image_path || !mask_path)) {
        fprintf(stderr,
                "[test_hip_sam3d] need <image.png> <mask.png> "
                "(or --slat-ref <dir> to bypass upstream).\n");
        return 2;
    }

    /* Load image (force 4 channels, RGBA). */
    int iw = 0, ih = 0, ichan = 0;
    uint8_t *pixels = NULL;
    if (image_path) {
        pixels = stbi_load(image_path, &iw, &ih, &ichan, 4);
        if (!pixels) { fprintf(stderr, "cannot decode %s\n", image_path); return 3; }
    }

    /* Load mask (grayscale). */
    int mw = 0, mh = 0, mchan = 0;
    uint8_t *mpix = NULL;
    if (mask_path) {
        mpix = stbi_load(mask_path, &mw, &mh, &mchan, 1);
        if (!mpix) {
            fprintf(stderr, "cannot decode %s\n", mask_path);
            stbi_image_free(pixels); return 3;
        }
    }

    /* Optional pointmap (required in v1 pipeline until MoGe is ported). */
    float *pmap = NULL;
    int pmap_dims[8] = {0}, pmap_ndim = 0, pmap_is_f32 = 0;
    if (pointmap_path) {
        pmap = (float *)npy_load(pointmap_path, &pmap_ndim, pmap_dims, &pmap_is_f32);
        if (!pmap || !pmap_is_f32 || pmap_ndim != 3 || pmap_dims[2] != 3) {
            fprintf(stderr, "pointmap must be (H, W, 3) f32; got ndim=%d\n", pmap_ndim);
            free(pmap); stbi_image_free(pixels); stbi_image_free(mpix);
            return 4;
        }
    }

    hip_sam3d_ctx *ctx = hip_sam3d_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "[test_hip_sam3d] hip_sam3d_create failed\n");
        stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }

    if (pixels && hip_sam3d_set_image_rgba(ctx, pixels, iw, ih) != 0) {
        fprintf(stderr, "set image failed\n");
        hip_sam3d_destroy(ctx); stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }
    if (mpix && hip_sam3d_set_mask(ctx, mpix, mw, mh) != 0) {
        fprintf(stderr, "set mask failed\n");
        hip_sam3d_destroy(ctx); stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }
    if (pmap && hip_sam3d_set_pointmap(ctx, pmap, pmap_dims[1], pmap_dims[0]) != 0) {
        fprintf(stderr, "set pointmap failed\n");
        hip_sam3d_destroy(ctx); stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }

    int rc = 0;
    int32_t *slat_coords = NULL;
    float   *slat_feats  = NULL;

    if (slat_ref_dir) {
        char pbuf[1536];
        int nd = 0, dims[8] = {0}, is_f32 = 0;
        const char *names_c[2] = { "slat_gs_in_coords.npy", "slat_dit_out_coords.npy" };
        const char *names_f[2] = { "slat_gs_in_feats.npy",  "slat_dit_out_feats.npy"  };
        for (int k = 0; k < 2 && !slat_coords; k++) {
            snprintf(pbuf, sizeof(pbuf), "%s/%s", slat_ref_dir, names_c[k]);
            slat_coords = (int32_t *)npy_load(pbuf, &nd, dims, &is_f32);
        }
        if (!slat_coords || nd != 2 || dims[1] != 4) {
            fprintf(stderr, "--slat-ref: cannot read slat_*_coords.npy from %s\n",
                    slat_ref_dir);
            rc = 6; goto cleanup;
        }
        int N = dims[0];
        for (int k = 0; k < 2 && !slat_feats; k++) {
            snprintf(pbuf, sizeof(pbuf), "%s/%s", slat_ref_dir, names_f[k]);
            slat_feats = (float *)npy_load(pbuf, &nd, dims, &is_f32);
        }
        if (!slat_feats || !is_f32 || nd != 2 || dims[0] != N) {
            fprintf(stderr, "--slat-ref: cannot read slat_*_feats.npy from %s\n",
                    slat_ref_dir);
            rc = 6; goto cleanup;
        }
        int C = dims[1];
        if (hip_sam3d_debug_override_slat(ctx, slat_feats, slat_coords, N, C) != 0) {
            fprintf(stderr, "--slat-ref: override failed\n");
            rc = 7; goto cleanup;
        }
        fprintf(stderr, "[test_hip_sam3d] --slat-ref: loaded N=%d C=%d\n", N, C);
    } else {
        if ((rc = hip_sam3d_run_dinov2(ctx))     != 0) { fprintf(stderr, "dinov2 rc=%d\n",     rc); goto cleanup; }
        if ((rc = hip_sam3d_run_cond_fuser(ctx)) != 0) { fprintf(stderr, "cond_fuser rc=%d\n", rc); goto cleanup; }
        if ((rc = hip_sam3d_run_ss_dit(ctx))     != 0) { fprintf(stderr, "ss_dit rc=%d\n",     rc); goto cleanup; }
        if ((rc = hip_sam3d_run_ss_decode(ctx))  != 0) { fprintf(stderr, "ss_decode rc=%d\n",  rc); goto cleanup; }
        if ((rc = hip_sam3d_run_slat_dit(ctx))   != 0) { fprintf(stderr, "slat_dit rc=%d\n",   rc); goto cleanup; }
    }

    if ((rc = hip_sam3d_run_slat_gs_decode(ctx)) != 0) {
        fprintf(stderr, "slat_gs_decode rc=%d\n", rc); goto cleanup;
    }

    int n_gauss = 0;
    if (hip_sam3d_get_gaussians(ctx, NULL, &n_gauss) != 0 || n_gauss <= 0) {
        fprintf(stderr, "no gaussians produced\n"); rc = 8; goto cleanup;
    }
    float *gaussians = (float *)malloc((size_t)n_gauss * HIP_SAM3D_GS_STRIDE * sizeof(float));
    if (!gaussians || hip_sam3d_get_gaussians(ctx, gaussians, NULL) != 0) {
        fprintf(stderr, "get_gaussians failed\n"); free(gaussians); rc = 8; goto cleanup;
    }

    /* INRIA-PLY rows: x y z  nx ny nz  f_dc(3)  opacity_logit
     *                 scale_log(3)  rot(4) — slice into per-channel
     * arrays for gs_ply_write. */
    float *xyz  = (float *)malloc((size_t)n_gauss * 3 * sizeof(float));
    float *f_dc = (float *)malloc((size_t)n_gauss * 3 * sizeof(float));
    float *op   = (float *)malloc((size_t)n_gauss     * sizeof(float));
    float *scl  = (float *)malloc((size_t)n_gauss * 3 * sizeof(float));
    float *rot  = (float *)malloc((size_t)n_gauss * 4 * sizeof(float));
    for (int i = 0; i < n_gauss; i++) {
        const float *row = gaussians + (size_t)i * HIP_SAM3D_GS_STRIDE;
        xyz [i*3+0] = row[0];  xyz [i*3+1] = row[1];  xyz [i*3+2] = row[2];
        f_dc[i*3+0] = row[6];  f_dc[i*3+1] = row[7];  f_dc[i*3+2] = row[8];
        op[i] = row[9];
        scl [i*3+0] = row[10]; scl [i*3+1] = row[11]; scl [i*3+2] = row[12];
        rot [i*4+0] = row[13]; rot [i*4+1] = row[14];
        rot [i*4+2] = row[15]; rot [i*4+3] = row[16];
    }
    if (gs_ply_write(out_path, n_gauss, xyz, NULL, f_dc, op, scl, rot) != 0) {
        fprintf(stderr, "gs_ply_write failed\n"); rc = 9;
    } else {
        fprintf(stderr, "[test_hip_sam3d] wrote %d gaussians to %s\n", n_gauss, out_path);
    }
    free(xyz); free(f_dc); free(op); free(scl); free(rot);
    free(gaussians);

cleanup:
    hip_sam3d_destroy(ctx);
    stbi_image_free(pixels);
    stbi_image_free(mpix);
    free(pmap);
    free(slat_coords); free(slat_feats);
    return rc;
}
