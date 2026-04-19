/* Diff sam3_runner's DETR encoder output vs detr_enc.npy.
 *
 * Usage: verify_detr --ckpt <sam3.model.safetensors> --image <img>
 *                    [--refdir /tmp/sam3_ref_cat]
 *
 * Runs ViT + FPN + text + DETR encoder, diffing (5184, 256) vs ref.
 */
#include "sam3_runner.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_npy(const char *path, int *ndim, int *dims, size_t *esz) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET);
    uint16_t hl; if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = '\0'; *ndim = 0; *esz = 0;
    if (strstr(hdr, "<f4")) *esz = 4;
    else if (strstr(hdr, "<i8")) *esz = 8;
    else if (strstr(hdr, "<i4")) *esz = 4;
    char *sp = strstr(hdr, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') {
            while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break;
            dims[(*ndim)++] = (int)strtol(sp, &sp, 10);
            if (*ndim >= 8) break;
        } } }
    size_t n = 1; for (int i = 0; i < *ndim; i++) n *= (size_t)dims[i];
    void *d = malloc(n * *esz);
    size_t got = fread(d, *esz, n, f);
    fclose(f); free(hdr);
    if (got != n) { free(d); return NULL; }
    return d;
}

int main(int argc, char **argv)
{
    const char *ckpt = NULL, *img_path = NULL;
    const char *refdir = "/tmp/sam3_ref_cat";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt = argv[++i];
        else if (!strcmp(argv[i], "--image")  && i+1 < argc) img_path = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
    }
    if (!ckpt || !img_path) {
        fprintf(stderr, "--ckpt and --image required\n"); return 1;
    }
    int H, W, C;
    unsigned char *rgb = stbi_load(img_path, &W, &H, &C, 3);
    if (!rgb) { fprintf(stderr, "bad image\n"); return 2; }

    sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1008, .num_threads = 0 };
    sam3_ctx *ctx = sam3_create(&cfg);
    if (!ctx) return 3;

    char path[512];
    int nd, d[8]; size_t esz;
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    float *px = (float *)read_npy(path, &nd, d, &esz);
    if (px) { sam3_set_pixel_values(ctx, px); free(px); }
    else sam3_set_image(ctx, rgb, H, W);

    snprintf(path, sizeof(path), "%s/input_input_ids.npy", refdir);
    void *idr = read_npy(path, &nd, d, &esz);
    int32_t ids[32], mask[32];
    int64_t *ai = (int64_t *)idr;
    for (int t = 0; t < 32; t++) ids[t] = (int32_t)ai[t];
    free(idr);
    snprintf(path, sizeof(path), "%s/input_attention_mask.npy", refdir);
    void *mr = read_npy(path, &nd, d, &esz);
    int64_t *am = (int64_t *)mr;
    for (int t = 0; t < 32; t++) mask[t] = (int32_t)am[t];
    free(mr);

    fprintf(stderr, "running ViT + FPN + text + DETR enc ...\n");
    if (sam3_run_vit(ctx, 31) != 0) return 4;
    if (sam3_run_fpn(ctx) != 0) return 5;
    if (sam3_set_input_ids(ctx, ids, mask) != 0) return 6;
    if (sam3_run_text(ctx) != 0) return 7;
    if (sam3_run_detr_enc(ctx) != 0) return 8;

    int n, dd;
    const float *ours = sam3_get_detr_enc(ctx, &n, &dd);
    fprintf(stderr, "detr_enc: (%d, %d)\n", n, dd);

    snprintf(path, sizeof(path), "%s/detr_enc.npy", refdir);
    float *ref = (float *)read_npy(path, &nd, d, &esz);
    if (!ref) { fprintf(stderr, "cannot read %s\n", path); return 9; }
    size_t sz = (size_t)n * dd;
    double sd = 0, sa = 0, sb = 0, mx = 0; size_t imx = 0;
    for (size_t i = 0; i < sz; i++) {
        double diff = fabs((double)ref[i] - (double)ours[i]);
        if (diff > mx) { mx = diff; imx = i; }
        sa += ref[i]; sb += ours[i]; sd += diff;
    }
    fprintf(stderr, "detr_enc: max_abs=%.4e mean_abs=%.4e "
            "(ref mean=%.4e ours mean=%.4e) at idx=%zu\n",
            mx, sd / sz, sa / sz, sb / sz, imx);
    free(ref);
    sam3_destroy(ctx);
    stbi_image_free(rgb);
    return 0;
}
