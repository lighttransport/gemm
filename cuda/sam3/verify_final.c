/* Diff HIP SAM 3 post-processed final outputs vs reference. */
#include "cuda_sam3_runner.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_npy(const char *path, int *ndim, int *dims, int *itemsz_out) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET);
    uint16_t hl; if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = '\0'; *ndim = 0;
    char *sp = strstr(hdr, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') {
            while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break;
            dims[(*ndim)++] = (int)strtol(sp, &sp, 10);
            if (*ndim >= 8) break;
        } } }
    int itemsz = strstr(hdr, "u1") ? 1 :
                 strstr(hdr, "f4") ? 4 :
                 strstr(hdr, "i8") ? 8 : 4;
    if (itemsz_out) *itemsz_out = itemsz;
    size_t n = 1; for (int i = 0; i < *ndim; i++) n *= (size_t)dims[i];
    void *d = malloc(n * itemsz);
    size_t got = fread(d, itemsz, n, f);
    fclose(f); free(hdr);
    if (got != n) { free(d); return NULL; }
    return d;
}

int main(int argc, char **argv)
{
    const char *ckpt = NULL;
    const char *refdir = "/tmp/sam3_ref_cat";
    int target_h = 426, target_w = 640;
    float score_th = 0.01f, mask_th = 0.5f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ckpt") && i+1 < argc) ckpt = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--th") && i+1 < argc) target_h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tw") && i+1 < argc) target_w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--score") && i+1 < argc) score_th = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--mask")  && i+1 < argc) mask_th  = (float)atof(argv[++i]);
    }
    if (!ckpt) { fprintf(stderr, "--ckpt required\n"); return 1; }

    char path[512]; int nd, d[8], isz;
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    float *px = (float *)read_npy(path, &nd, d, &isz); if (!px) return 2;
    snprintf(path, sizeof(path), "%s/input_input_ids.npy", refdir);
    int64_t *ids64 = (int64_t *)read_npy(path, &nd, d, &isz);
    int32_t ids[32]; for (int i = 0; i < 32; i++) ids[i] = (int32_t)ids64[i];
    free(ids64);
    snprintf(path, sizeof(path), "%s/input_attention_mask.npy", refdir);
    int64_t *am64 = (int64_t *)read_npy(path, &nd, d, &isz);
    int32_t amask[32]; const int32_t *amask_p = NULL;
    if (am64) { for (int i = 0; i < 32; i++) amask[i] = (int32_t)am64[i];
                free(am64); amask_p = amask; }

    cuda_sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1008,
                             .device_ordinal = 0, .verbose = 1 };
    cuda_sam3_ctx *ctx = cuda_sam3_create(&cfg);
    if (!ctx) return 3;
    if (cuda_sam3_set_pixel_values(ctx, px) != 0) return 4;
    free(px);
    if (cuda_sam3_run_vit(ctx, 31)) return 5;
    if (cuda_sam3_run_fpn(ctx)) return 6;
    if (cuda_sam3_set_input_ids(ctx, ids, amask_p)) return 7;
    if (cuda_sam3_run_text(ctx)) return 8;
    if (cuda_sam3_run_detr_enc(ctx)) return 9;
    if (cuda_sam3_run_detr_dec(ctx)) return 10;
    if (cuda_sam3_run_dot_score(ctx)) return 11;
    if (cuda_sam3_run_mask_dec(ctx)) return 12;
    if (cuda_sam3_run_postprocess(ctx, target_h, target_w, score_th, mask_th)) return 13;

    int n_ours = 0;
    const float *scores = cuda_sam3_get_final_scores(ctx, &n_ours);
    int bn; const float *boxes = cuda_sam3_get_final_boxes(ctx, &bn);
    int mn, mh, mw; const uint8_t *masks = cuda_sam3_get_final_masks(ctx, &mn, &mh, &mw);
    fprintf(stderr, "ours: n=%d target=(%d,%d) scores[0]=%.4f\n",
            n_ours, mh, mw, n_ours ? scores[0] : 0.0f);

    snprintf(path, sizeof(path), "%s/final_scores.npy", refdir);
    float *rs = (float *)read_npy(path, &nd, d, &isz);
    int n_ref = rs ? d[0] : 0;
    if (rs) { fprintf(stderr, "ref scores[%d]: ", n_ref);
        for (int i = 0; i < n_ref; i++) fprintf(stderr, "%.4f ", rs[i]);
        fprintf(stderr, "\n"); }

    snprintf(path, sizeof(path), "%s/final_boxes.npy", refdir);
    float *rb = (float *)read_npy(path, &nd, d, &isz);
    if (rb && n_ref) { fprintf(stderr, "ref boxes[0]=[%.2f,%.2f,%.2f,%.2f]\n",
        rb[0], rb[1], rb[2], rb[3]);
        if (n_ours) fprintf(stderr, "our boxes[0]=[%.2f,%.2f,%.2f,%.2f]\n",
            boxes[0], boxes[1], boxes[2], boxes[3]); }

    snprintf(path, sizeof(path), "%s/final_masks.npy", refdir);
    uint8_t *rm = (uint8_t *)read_npy(path, &nd, d, &isz);
    if (rm && n_ref && n_ours) {
        size_t tot = (size_t)d[1] * d[2];
        size_t inter = 0, uni = 0;
        for (size_t i = 0; i < tot; i++) {
            int a = rm[i] ? 1 : 0, b = masks[i] ? 1 : 0;
            inter += (a & b); uni += (a | b);
        }
        double iou = uni ? (double)inter / (double)uni : 0.0;
        fprintf(stderr, "mask IoU (ref #0 vs ours #0): %.4f (inter=%zu uni=%zu)\n",
                iou, inter, uni);
    }
    free(rs); free(rb); free(rm);
    cuda_sam3_destroy(ctx);
    return 0;
}
