/* End-to-end verify: runs full sam3 pipeline + post-process, computes IoU
 * vs ref final_masks.npy. Also prints box + score comparison.
 *
 * Usage: verify_final --ckpt <ckpt> --image <img> [--refdir /tmp/sam3_ref_cat]
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
    else if (strstr(hdr, "|u1")) *esz = 1;
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
        if (!strcmp(argv[i], "--ckpt") && i+1 < argc) ckpt = argv[++i];
        else if (!strcmp(argv[i], "--image") && i+1 < argc) img_path = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
    }
    if (!ckpt || !img_path) { fprintf(stderr, "--ckpt and --image required\n"); return 1; }
    int H, W, C; unsigned char *rgb = stbi_load(img_path, &W, &H, &C, 3);
    if (!rgb) return 2;

    sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1008, .num_threads = 0 };
    sam3_ctx *ctx = sam3_create(&cfg);
    if (!ctx) return 3;

    char path[512]; int nd, d[8]; size_t esz;
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    float *px = (float *)read_npy(path, &nd, d, &esz);
    if (px) { sam3_set_pixel_values(ctx, px); free(px); }
    else sam3_set_image(ctx, rgb, H, W);

    snprintf(path, sizeof(path), "%s/input_input_ids.npy", refdir);
    int64_t *ai = (int64_t *)read_npy(path, &nd, d, &esz);
    snprintf(path, sizeof(path), "%s/input_attention_mask.npy", refdir);
    int64_t *am = (int64_t *)read_npy(path, &nd, d, &esz);
    int32_t ids[32], mask[32];
    for (int t = 0; t < 32; t++) { ids[t] = (int32_t)ai[t]; mask[t] = (int32_t)am[t]; }
    free(ai); free(am);

    /* Target size from input_original_sizes.npy (H, W). */
    int target_h = 512, target_w = 512;
    snprintf(path, sizeof(path), "%s/input_original_sizes.npy", refdir);
    int64_t *osz = (int64_t *)read_npy(path, &nd, d, &esz);
    if (osz) { target_h = (int)osz[0]; target_w = (int)osz[1]; free(osz); }

    fprintf(stderr, "running full pipeline (target %dx%d) ...\n", target_h, target_w);
    if (sam3_run_vit(ctx, 31)) return 4;
    if (sam3_run_fpn(ctx)) return 5;
    if (sam3_set_input_ids(ctx, ids, mask)) return 6;
    if (sam3_run_text(ctx)) return 7;
    if (sam3_run_detr_enc(ctx)) return 8;
    if (sam3_run_detr_dec(ctx)) return 9;
    if (sam3_run_dot_score(ctx)) return 10;
    if (sam3_run_mask_dec(ctx)) return 11;
    if (sam3_run_postprocess(ctx, target_h, target_w, 0.3f, 0.5f)) return 12;

    int nk, oh, ow;
    const float   *scores = sam3_get_final_scores(ctx, &nk);
    const float   *boxes  = sam3_get_final_boxes(ctx, &nk);
    const uint8_t *masks  = sam3_get_final_masks(ctx, &nk, &oh, &ow);

    fprintf(stderr, "kept %d instances, target size (%d,%d)\n", nk, oh, ow);
    for (int i = 0; i < nk && i < 10; i++) {
        fprintf(stderr, "  [%d] score=%.4f box=(%.2f, %.2f, %.2f, %.2f)\n", i,
                scores[i], boxes[i*4], boxes[i*4+1],
                boxes[i*4+2], boxes[i*4+3]);
    }

    /* Ref. */
    snprintf(path, sizeof(path), "%s/final_scores.npy", refdir);
    float *refs = (float *)read_npy(path, &nd, d, &esz);
    snprintf(path, sizeof(path), "%s/final_boxes.npy", refdir);
    float *refb = (float *)read_npy(path, &nd, d, &esz);
    snprintf(path, sizeof(path), "%s/final_masks.npy", refdir);
    uint8_t *refm = (uint8_t *)read_npy(path, &nd, d, &esz);
    int ref_n = refs ? d[0] : 0;
    if (refs && refb) {
        fprintf(stderr, "ref %d instances:\n", ref_n);
        for (int i = 0; i < ref_n; i++) {
            fprintf(stderr, "  [%d] score=%.4f box=(%.2f, %.2f, %.2f, %.2f)\n", i,
                    refs[i], refb[i*4], refb[i*4+1], refb[i*4+2], refb[i*4+3]);
        }
    }
    if (refm && nk >= 1 && ref_n >= 1) {
        /* IoU of first ours vs first ref. */
        size_t hw = (size_t)oh * ow;
        size_t inter = 0, uni = 0;
        for (size_t i = 0; i < hw; i++) {
            int a = masks[i], b = refm[i];
            inter += (a && b); uni += (a || b);
        }
        double iou = uni ? (double)inter / (double)uni : 0.0;
        fprintf(stderr, "mask IoU (ours[0] vs ref[0]) = %.4f\n", iou);
    }
    free(refs); free(refb); free(refm);

    sam3_destroy(ctx); stbi_image_free(rgb);
    return 0;
}
