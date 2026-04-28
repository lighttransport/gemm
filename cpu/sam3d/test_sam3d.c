/*
 * test_sam3d — CLI for the SAM 3D Objects CPU runner.
 *
 * Usage:
 *   test_sam3d <pipeline.yaml> <image.png> <mask.png> \
 *              [--pointmap pmap.npy] [--seed N] [--steps N] \
 *              [--slat-steps N] [--cfg F] [-o splat.ply] [-v]
 *
 * v1 scope: requires --pointmap (MoGe is deferred). Writes a 3D
 * Gaussian splat to splat.ply on success.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#define GS_PLY_WRITER_IMPLEMENTATION
#include "../../common/gs_ply_writer.h"

#include "sam3d_runner.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

#define STAGE(name, call)                                                       \
    do {                                                                        \
        fprintf(stderr, "[stage] %-12s ...\n", name); fflush(stderr);           \
        double _t0 = now_ms();                                                  \
        rc = (call);                                                            \
        double _dt = now_ms() - _t0;                                            \
        if (rc != 0) {                                                          \
            fprintf(stderr, "[stage] %-12s FAIL rc=%d (%.1f ms)\n",             \
                    name, rc, _dt);                                             \
            goto cleanup;                                                       \
        }                                                                       \
        fprintf(stderr, "[stage] %-12s OK (%.1f ms)\n", name, _dt);             \
        fflush(stderr);                                                         \
    } while (0)

/* Minimal .npy reader — little-endian f32 or i4, any rank.
 * `want_i32` selects dtype ('f4' if 0, 'i4' if 1). Returns a malloc'd
 * buffer of ndim*dims element count; caller frees. */
static void *read_npy_raw(const char *path, int *ndim, int *dims, int want_i32) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return NULL; }
    fseek(f, 8, SEEK_SET);
    uint16_t hl = 0;
    if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = 0;
    *ndim = 0;
    char *sp = strstr(hdr, "shape");
    if (sp) {
        sp = strchr(sp, '(');
        if (sp) {
            sp++;
            while (*sp && *sp != ')') {
                while (*sp == ' ' || *sp == ',') sp++;
                if (*sp == ')') break;
                dims[(*ndim)++] = (int)strtol(sp, &sp, 10);
                if (*ndim >= 8) break;
            }
        }
    }
    const char *want = want_i32 ? "i4" : "f4";
    if (!strstr(hdr, want)) {
        fprintf(stderr, "npy %s: want %s, header=%s\n", path, want, hdr);
        free(hdr); fclose(f); return NULL;
    }
    size_t n = 1;
    for (int i = 0; i < *ndim; i++) n *= (size_t)dims[i];
    void *d = malloc(n * 4);
    size_t got = fread(d, 4, n, f);
    fclose(f); free(hdr);
    if (got != n) { free(d); return NULL; }
    return d;
}
static float *read_npy_f32(const char *path, int *ndim, int *dims) {
    return (float *)read_npy_raw(path, ndim, dims, 0);
}
static int32_t *read_npy_i32(const char *path, int *ndim, int *dims) {
    return (int32_t *)read_npy_raw(path, ndim, dims, 1);
}

int main(int argc, char **argv)
{
    const char *pipeline_yaml = NULL;
    const char *image_path    = NULL;
    const char *mask_path     = NULL;
    const char *pointmap_path = NULL;
    const char *slat_ref_dir  = NULL;
    const char *out_path      = "splat.ply";
    uint64_t seed       = 42;
    int      ss_steps   = 2;   /* shortcut ODE: pipeline default is 2 */
    int      slat_steps = 12;  /* slat flow: pipeline default is 12 */
    float    cfg_scale  = 2.0f;
    int      verbose    = 0;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--pointmap")    && i+1 < argc) pointmap_path = argv[++i];
        else if (!strcmp(a, "--slat-ref")    && i+1 < argc) slat_ref_dir  = argv[++i];
        else if (!strcmp(a, "--seed")        && i+1 < argc) seed          = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--steps")       && i+1 < argc) ss_steps      = atoi(argv[++i]);
        else if (!strcmp(a, "--slat-steps")  && i+1 < argc) slat_steps    = atoi(argv[++i]);
        else if (!strcmp(a, "--cfg")         && i+1 < argc) cfg_scale     = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-o")            && i+1 < argc) out_path      = argv[++i];
        else if (!strcmp(a, "-v"))                          verbose       = 1;
        else if (a[0] != '-') {
            if      (positional == 0) pipeline_yaml = a;
            else if (positional == 1) image_path    = a;
            else if (positional == 2) mask_path     = a;
            positional++;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            return 2;
        }
    }

    if (!pipeline_yaml || (!slat_ref_dir && (!image_path || !mask_path))) {
        fprintf(stderr,
                "Usage: %s <pipeline.yaml> <image.png> <mask.png> "
                "[--pointmap pmap.npy] [--slat-ref <dir>] [--seed N] "
                "[--steps N] [--slat-steps N] [--cfg F] "
                "[-o splat.ply] [-v]\n"
                "  --slat-ref <dir>: bypass upstream stages by loading "
                "slat_dit_out_{coords,feats}.npy and run only "
                "SLAT-GS-decode + PLY write. image/mask become optional.\n",
                argv[0]);
        return 2;
    }

    /* Load image (RGB or RGBA). */
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
    int pmap_dims[8] = {0}, pmap_ndim = 0;
    if (pointmap_path) {
        pmap = read_npy_f32(pointmap_path, &pmap_ndim, pmap_dims);
        if (!pmap || pmap_ndim != 3 || pmap_dims[2] != 3) {
            fprintf(stderr, "pointmap must be (H, W, 3) f32; got ndim=%d\n", pmap_ndim);
            free(pmap); stbi_image_free(pixels); stbi_image_free(mpix);
            return 4;
        }
    }

    sam3d_config cfg = {
        .pipeline_yaml = pipeline_yaml,
        .seed          = seed,
        .ss_steps      = ss_steps,
        .slat_steps    = slat_steps,
        .cfg_scale     = cfg_scale,
        .verbose       = verbose,
    };
    sam3d_ctx *ctx = sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_create failed\n"); return 5; }

    if (pixels && sam3d_set_image_rgba(ctx, pixels, iw, ih) != 0) {
        fprintf(stderr, "set image failed\n");
        sam3d_destroy(ctx); stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }
    if (mpix && sam3d_set_mask(ctx, mpix, mw, mh) != 0) {
        fprintf(stderr, "set mask failed\n");
        sam3d_destroy(ctx); stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }
    if (pmap && sam3d_set_pointmap(ctx, pmap, pmap_dims[1], pmap_dims[0]) != 0) {
        fprintf(stderr, "set pointmap failed\n");
        sam3d_destroy(ctx); stbi_image_free(pixels); stbi_image_free(mpix); free(pmap);
        return 5;
    }

    int rc = 0;
    int32_t *slat_coords = NULL;
    float   *slat_feats  = NULL;

    if (slat_ref_dir) {
        /* Bypass: load SLAT tokens directly from a ref dump and jump
         * straight to SLAT-GS decode. Tokens can come from either
         * ref/sam3d/dump_slat_gs_io.py (filename slat_gs_in_*) or a
         * SLAT-DiT end-of-forward dump (slat_dit_out_*). */
        char pbuf[1536];
        int nd = 0, dims[8] = {0};
        const char *names_c[2] = { "slat_gs_in_coords.npy", "slat_dit_out_coords.npy" };
        const char *names_f[2] = { "slat_gs_in_feats.npy",  "slat_dit_out_feats.npy"  };
        for (int k = 0; k < 2 && !slat_coords; k++) {
            snprintf(pbuf, sizeof(pbuf), "%s/%s", slat_ref_dir, names_c[k]);
            slat_coords = read_npy_i32(pbuf, &nd, dims);
        }
        if (!slat_coords || nd != 2 || dims[1] != 4) {
            fprintf(stderr, "--slat-ref: cannot read slat_*_coords.npy from %s\n",
                    slat_ref_dir);
            rc = 6; goto cleanup;
        }
        int N = dims[0];
        for (int k = 0; k < 2 && !slat_feats; k++) {
            snprintf(pbuf, sizeof(pbuf), "%s/%s", slat_ref_dir, names_f[k]);
            slat_feats = read_npy_f32(pbuf, &nd, dims);
        }
        if (!slat_feats || nd != 2 || dims[0] != N) {
            fprintf(stderr, "--slat-ref: cannot read slat_*_feats.npy from %s\n",
                    slat_ref_dir);
            rc = 6; goto cleanup;
        }
        int C = dims[1];
        if (sam3d_debug_override_slat(ctx, slat_feats, slat_coords, N, C) != 0) {
            fprintf(stderr, "--slat-ref: override failed\n");
            rc = 7; goto cleanup;
        }
        fprintf(stderr, "[test_sam3d] --slat-ref: loaded N=%d C=%d\n", N, C);
    } else {
        STAGE("dinov2",     sam3d_run_dinov2(ctx));
        STAGE("cond_fuser", sam3d_run_cond_fuser(ctx));
        STAGE("ss_dit",     sam3d_run_ss_dit(ctx));
        STAGE("ss_decode",  sam3d_run_ss_decode(ctx));
        STAGE("slat_dit",   sam3d_run_slat_dit(ctx));
    }

    STAGE("gs_decode", sam3d_run_slat_gs_decode(ctx));

    int n_gauss = 0;
    if (sam3d_get_gaussians(ctx, NULL, &n_gauss) != 0 || n_gauss <= 0) {
        fprintf(stderr, "no gaussians produced\n"); rc = 8; goto cleanup;
    }
    float *gaussians = (float *)malloc((size_t)n_gauss * SAM3D_GS_STRIDE * sizeof(float));
    if (!gaussians || sam3d_get_gaussians(ctx, gaussians, NULL) != 0) {
        fprintf(stderr, "get_gaussians failed\n"); free(gaussians); rc = 8; goto cleanup;
    }

    /* ctx->gaussians rows: x y z  nx ny nz  f_dc(3)  opacity_logit
     *                     scale_log(3)  rot(4)    — already in INRIA
     * storage convention. Hand the slices straight to the PLY writer. */
    float *xyz = (float *)malloc((size_t)n_gauss * 3 * sizeof(float));
    float *f_dc = (float *)malloc((size_t)n_gauss * 3 * sizeof(float));
    float *op = (float *)malloc((size_t)n_gauss * sizeof(float));
    float *scl = (float *)malloc((size_t)n_gauss * 3 * sizeof(float));
    float *rot = (float *)malloc((size_t)n_gauss * 4 * sizeof(float));
    for (int i = 0; i < n_gauss; i++) {
        const float *row = gaussians + (size_t)i * SAM3D_GS_STRIDE;
        xyz[i*3+0] = row[0]; xyz[i*3+1] = row[1]; xyz[i*3+2] = row[2];
        f_dc[i*3+0] = row[6]; f_dc[i*3+1] = row[7]; f_dc[i*3+2] = row[8];
        op[i] = row[9];
        scl[i*3+0] = row[10]; scl[i*3+1] = row[11]; scl[i*3+2] = row[12];
        rot[i*4+0] = row[13]; rot[i*4+1] = row[14];
        rot[i*4+2] = row[15]; rot[i*4+3] = row[16];
    }
    if (gs_ply_write(out_path, n_gauss, xyz, NULL, f_dc, op, scl, rot) != 0) {
        fprintf(stderr, "gs_ply_write failed\n"); rc = 9;
    } else {
        fprintf(stderr, "[test_sam3d] wrote %d gaussians to %s\n", n_gauss, out_path);
    }
    free(xyz); free(f_dc); free(op); free(scl); free(rot);
    free(gaussians);

cleanup:
    sam3d_destroy(ctx);
    stbi_image_free(pixels);
    stbi_image_free(mpix);
    free(pmap);
    free(slat_coords); free(slat_feats);
    return rc;
}
