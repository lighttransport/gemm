/* Diff HIP SAM 3 ViT output against PyTorch reference dumps.
 *
 * Targets:
 *   block0   → vit_block00.npy (after block 0, windowed)
 *   block31  → vit_block31.npy (after block 31, global)
 *   final    → vision_encoder.npy (same shape, flat)
 */
#include "cuda_sam3_1_runner.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

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
    if (strstr(hdr, "fortran_order': True")) {
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
    if (!ckpt) { fprintf(stderr, "--ckpt required\n"); return 1; }

    int stop_at = 0;
    const char *ref_name = NULL;
    if (!strcmp(target, "block0"))       { stop_at = 0;  ref_name = "vit_block00.npy"; }
    else if (!strcmp(target, "block31")) { stop_at = 31; ref_name = "vit_block31.npy"; }
    else if (!strcmp(target, "final"))   { stop_at = 31; ref_name = "vision_encoder.npy"; }
    else { fprintf(stderr, "unknown --target '%s'\n", target); return 1; }

    cuda_sam3_1_config cfg = { .ckpt_path = ckpt, .image_size = 1008,
                             .device_ordinal = 0, .verbose = 1 };
    cuda_sam3_1_ctx *ctx = cuda_sam3_1_create(&cfg);
    if (!ctx) return 2;

    char path[512];
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    int pxnd, pxd[8];
    float *ref_px = read_npy_f32(path, &pxnd, pxd);
    if (ref_px) {
        fprintf(stderr, "using ref pixel_values from %s\n", path);
        if (cuda_sam3_1_set_pixel_values(ctx, ref_px) != 0) return 3;
        free(ref_px);
    } else if (img_path) {
        int H, W, C;
        unsigned char *rgb = stbi_load(img_path, &W, &H, &C, 3);
        if (!rgb) { fprintf(stderr, "can't load %s\n", img_path); return 3; }
        if (cuda_sam3_1_set_image(ctx, rgb, H, W) != 0) return 3;
        stbi_image_free(rgb);
    } else {
        fprintf(stderr, "no ref pixel_values and no --image\n"); return 3;
    }

    fprintf(stderr, "running HIP ViT through block %d (target=%s) ...\n",
            stop_at, target);
    if (cuda_sam3_1_run_vit(ctx, stop_at) != 0) {
        fprintf(stderr, "cuda_sam3_1_run_vit failed\n"); return 4;
    }

    int n_tok, dim;
    float *ours = (float *)malloc((size_t)5184 * 1024 * 4);
    if (cuda_sam3_1_get_vit_embed(ctx, ours, &n_tok, &dim) != 0) return 5;
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
    double max_abs = 0;
    for (size_t i = 0; i < n; i++) {
        double dd = fabs((double)ref[i] - (double)ours[i]);
        sa += ref[i]; sb += ours[i]; sd += dd;
        if (dd > max_abs) max_abs = dd;
        if (dd > 0.1) cnt_gt_0_1++;
        if (dd > 1.0) cnt_gt_1++;
    }
    fprintf(stderr,
            "hip vit_%s (%d,%d): max_abs=%.4e mean_abs=%.4e "
            "(ref mean=%.4e ours mean=%.4e) >0.1=%zu >1.0=%zu\n",
            target, n_tok, dim, max_abs, sd / (double)n,
            sa / (double)n, sb / (double)n, cnt_gt_0_1, cnt_gt_1);
    free(ref); free(ours);
    cuda_sam3_1_destroy(ctx);
    return 0;
}
