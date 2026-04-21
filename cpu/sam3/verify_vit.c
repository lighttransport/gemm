/* Diff sam3_runner's ViT output against the PyTorch reference dumps.
 *
 * Supported --target values:
 *   block0       → ref: vit_block00.npy  (1, 72, 72, 1024) — after layer 0
 *   block31      → ref: vit_block31.npy  (1, 72, 72, 1024) — after layer 31
 *   final        → ref: vision_encoder.npy (1, 5184, 1024)  — same as block31,
 *                  reshaped; used when vit_block31 isn't dumped.
 *
 * Usage:
 *   verify_vit --ckpt <sam3.model.safetensors> --image <img>
 *              [--refdir /tmp/sam3_ref_cat] [--target block0|block31|final]
 */
#include "sam3_runner.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_npy_f32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET);
    uint16_t hl; if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = '\0'; *ndim = 0;
    if (strstr(hdr, "fortran_order': True") ||
        strstr(hdr, "fortran_order':True")) {
        fprintf(stderr, "%s: fortran_order=True not supported\n", path);
        free(hdr); fclose(f); return NULL;
    }
    char *sp = strstr(hdr, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') {
            while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break;
            dims[(*ndim)++] = (int)strtol(sp, &sp, 10);
            if (*ndim >= 8) break;
        } } }
    size_t n = 1; for (int i = 0; i < *ndim; i++) n *= (size_t)dims[i];
    float *d = (float *)malloc(n * sizeof(float));
    size_t got = fread(d, sizeof(float), n, f);
    fclose(f); free(hdr);
    if (got != n) { free(d); return NULL; }
    return d;
}

int main(int argc, char **argv)
{
    const char *ckpt = NULL, *img_path = NULL;
    const char *refdir = "/tmp/sam3_ref_cat";
    const char *target = "block0";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt = argv[++i];
        else if (!strcmp(argv[i], "--image")  && i+1 < argc) img_path = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--target") && i+1 < argc) target = argv[++i];
    }
    if (!ckpt || !img_path) {
        fprintf(stderr, "--ckpt and --image required\n"); return 1;
    }

    int stop_at = 0;
    const char *ref_name = NULL;
    if (!strcmp(target, "block0"))       { stop_at = 0;  ref_name = "vit_block00.npy"; }
    else if (!strcmp(target, "block31")) { stop_at = 31; ref_name = "vit_block31.npy"; }
    else if (!strcmp(target, "final"))   { stop_at = 31; ref_name = "vision_encoder.npy"; }
    else { fprintf(stderr, "unknown --target '%s'\n", target); return 1; }

    int H, W, C;
    unsigned char *rgb = stbi_load(img_path, &W, &H, &C, 3);
    if (!rgb) { fprintf(stderr, "can't load %s\n", img_path); return 2; }

    sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1008, .num_threads = 0 };
    sam3_ctx *ctx = sam3_create(&cfg);
    if (!ctx) return 3;

    char path[512];
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    int pxnd, pxd[8];
    float *ref_px = read_npy_f32(path, &pxnd, pxd);
    if (ref_px) {
        fprintf(stderr, "using ref pixel_values from %s\n", path);
        if (sam3_set_pixel_values(ctx, ref_px) != 0) return 4;
        free(ref_px);
    } else {
        if (sam3_set_image(ctx, rgb, H, W) != 0) return 4;
    }

    fprintf(stderr, "running ViT through block %d (target=%s) ...\n",
            stop_at, target);
    if (sam3_run_vit(ctx, stop_at) != 0) { fprintf(stderr, "run_vit failed\n"); return 5; }

    int n_tok, dim;
    const float *ours = sam3_get_vit_output(ctx, &n_tok, &dim);
    size_t n = (size_t)n_tok * dim;

    snprintf(path, sizeof(path), "%s/%s", refdir, ref_name);
    int nd, d[8]; float *ref = read_npy_f32(path, &nd, d);
    if (!ref) { fprintf(stderr, "cannot read %s\n", path); return 6; }
    size_t rn = 1; for (int i = 0; i < nd; i++) rn *= (size_t)d[i];
    if (rn != n) {
        fprintf(stderr, "shape mismatch: ref=%zu ours=%zu\n", rn, n); return 7;
    }

    double sd = 0, sa = 0, sb = 0;
    size_t cnt_gt_0_1 = 0, cnt_gt_1 = 0;
    for (size_t i = 0; i < n; i++) {
        double dd = fabs((double)ref[i] - (double)ours[i]);
        sa += ref[i]; sb += ours[i]; sd += dd;
        if (dd > 0.1) cnt_gt_0_1++;
        if (dd > 1.0) cnt_gt_1++;
    }
    /* Find the top 5 max-abs positions. */
    double top_diff[5] = {0};
    size_t top_idx[5] = {0};
    for (size_t i = 0; i < n; i++) {
        double dd = fabs((double)ref[i] - (double)ours[i]);
        for (int k = 0; k < 5; k++) {
            if (dd > top_diff[k]) {
                for (int j = 4; j > k; j--) {
                    top_diff[j] = top_diff[j-1]; top_idx[j] = top_idx[j-1];
                }
                top_diff[k] = dd; top_idx[k] = i; break;
            }
        }
    }
    fprintf(stderr,
            "vit_%s (%d,%d): max_abs=%.4e mean_abs=%.4e "
            "(ref mean=%.4e ours mean=%.4e) >0.1=%zu  >1.0=%zu\n",
            target, n_tok, dim, top_diff[0], sd / (double)n,
            sa / (double)n, sb / (double)n, cnt_gt_0_1, cnt_gt_1);
    for (int k = 0; k < 5; k++) {
        size_t i = top_idx[k];
        int tok = (int)(i / dim); int d_ = (int)(i % dim);
        int ty = tok / 72, tx = tok % 72;
        fprintf(stderr, "  #%d idx=%zu tok=(y=%d,x=%d) d=%d  ref=%.4e  ours=%.4e  diff=%.4e\n",
                k, i, ty, tx, d_, ref[i], ours[i], top_diff[k]);
    }
    free(ref);
    sam3_destroy(ctx);
    stbi_image_free(rgb);
    return 0;
}
