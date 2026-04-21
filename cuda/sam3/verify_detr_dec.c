/* Diff HIP SAM 3 DETR decoder pred_boxes / presence against reference. */
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
    int itemsz = strstr(hdr, "f4") || strstr(hdr, "<f4") ? 4
               : strstr(hdr, "i8") || strstr(hdr, "<i8") ? 8
               : 4;
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
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
    }
    if (!ckpt) { fprintf(stderr, "--ckpt required\n"); return 1; }

    char path[512]; int nd, d[8], isz;
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    float *px = (float *)read_npy(path, &nd, d, &isz);
    if (!px) return 2;
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
    if (cuda_sam3_run_vit(ctx, 31) != 0) return 5;
    if (cuda_sam3_run_fpn(ctx) != 0) return 6;
    if (cuda_sam3_set_input_ids(ctx, ids, amask_p) != 0) return 7;
    if (cuda_sam3_run_text(ctx) != 0) return 8;
    if (cuda_sam3_run_detr_enc(ctx) != 0) return 9;
    fprintf(stderr, "running DETR decoder (6 layers, 200 queries) ...\n");
    if (cuda_sam3_run_detr_dec(ctx) != 0) return 10;

    float ours_boxes[200 * 4], ours_pres[6];
    cuda_sam3_get_detr_dec_boxes(ctx, ours_boxes);
    cuda_sam3_get_detr_dec_presence(ctx, ours_pres);

    /* pred_boxes (1, 200, 4). */
    snprintf(path, sizeof(path), "%s/pred_boxes.npy", refdir);
    int rnd, rd[8];
    float *ref_boxes = (float *)read_npy(path, &rnd, rd, &isz);
    if (ref_boxes) {
        double sd = 0, maxd = 0;
        for (int i = 0; i < 200 * 4; i++) {
            double dd = fabs(ref_boxes[i] - ours_boxes[i]);
            sd += dd; if (dd > maxd) maxd = dd;
        }
        fprintf(stderr, "pred_boxes (200,4): max_abs=%.4e mean_abs=%.4e\n",
                maxd, sd / (200.0 * 4));
        free(ref_boxes);
    }

    /* presence_logits (1,1): compare vs last layer. */
    snprintf(path, sizeof(path), "%s/presence_logits.npy", refdir);
    float *ref_pres = (float *)read_npy(path, &rnd, rd, &isz);
    if (ref_pres) {
        fprintf(stderr, "presence (last layer): ref=%.4f ours=%.4f |diff|=%.4e\n",
                ref_pres[0], ours_pres[5], fabsf(ref_pres[0] - ours_pres[5]));
        free(ref_pres);
    }
    cuda_sam3_destroy(ctx);
    return 0;
}
