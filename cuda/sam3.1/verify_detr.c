/* Diff HIP SAM 3 DETR encoder output against PyTorch reference. */
#include "cuda_sam3_1_runner.h"
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
               : strstr(hdr, "i4") || strstr(hdr, "<i4") ? 4
               : 0;
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
    /* Pixel values. */
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    float *px = (float *)read_npy(path, &nd, d, &isz);
    if (!px) { fprintf(stderr, "need %s\n", path); return 2; }
    /* Input ids. */
    snprintf(path, sizeof(path), "%s/input_input_ids.npy", refdir);
    int64_t *ids64 = (int64_t *)read_npy(path, &nd, d, &isz);
    if (!ids64) { fprintf(stderr, "need %s\n", path); return 3; }
    int32_t ids[32]; for (int i = 0; i < 32; i++) ids[i] = (int32_t)ids64[i];
    free(ids64);
    /* Attention mask (optional). */
    snprintf(path, sizeof(path), "%s/input_attention_mask.npy", refdir);
    int64_t *am64 = (int64_t *)read_npy(path, &nd, d, &isz);
    int32_t amask[32]; const int32_t *amask_p = NULL;
    if (am64) {
        for (int i = 0; i < 32; i++) amask[i] = (int32_t)am64[i];
        free(am64); amask_p = amask;
    }

    cuda_sam3_1_config cfg = { .ckpt_path = ckpt, .image_size = 1008,
                             .device_ordinal = 0, .verbose = 1 };
    cuda_sam3_1_ctx *ctx = cuda_sam3_1_create(&cfg);
    if (!ctx) return 4;

    if (cuda_sam3_1_set_pixel_values(ctx, px) != 0) return 5;
    free(px);
    fprintf(stderr, "running ViT (32) ...\n");
    if (cuda_sam3_1_run_vit(ctx, 31) != 0) return 6;
    fprintf(stderr, "running FPN ...\n");
    if (cuda_sam3_1_run_fpn(ctx) != 0) return 7;
    if (cuda_sam3_1_set_input_ids(ctx, ids, amask_p) != 0) return 8;
    fprintf(stderr, "running CLIP text ...\n");
    if (cuda_sam3_1_run_text(ctx) != 0) return 9;
    fprintf(stderr, "running DETR encoder (6 layers) ...\n");
    if (cuda_sam3_1_run_detr_enc(ctx) != 0) return 10;

    int on, od;
    size_t cap = (size_t)5184 * 256;
    float *ours = (float *)malloc(cap * 4);
    cuda_sam3_1_get_detr_enc(ctx, ours, &on, &od);
    size_t n = (size_t)on * od;

    snprintf(path, sizeof(path), "%s/detr_enc.npy", refdir);
    int rnd, rd[8];
    float *ref = (float *)read_npy(path, &rnd, rd, &isz);
    if (!ref) {
        snprintf(path, sizeof(path), "%s/detr_encoder.npy", refdir);
        ref = (float *)read_npy(path, &rnd, rd, &isz);
    }
    if (!ref) { fprintf(stderr, "no detr_enc.npy\n"); return 11; }
    size_t rn = 1; for (int i = 0; i < rnd; i++) rn *= (size_t)rd[i];
    if (rn != n) {
        fprintf(stderr, "shape mismatch ref=%zu ours=%zu\n", rn, n);
        free(ref); return 12;
    }
    double sd = 0, maxd = 0, sa = 0, sb = 0;
    for (size_t i = 0; i < n; i++) {
        double dd = fabs((double)ref[i] - (double)ours[i]);
        sd += dd; sa += ref[i]; sb += ours[i];
        if (dd > maxd) maxd = dd;
    }
    fprintf(stderr, "hip detr_enc (%d,%d): max_abs=%.4e mean_abs=%.4e "
            "(ref mean=%.4e ours mean=%.4e)\n",
            on, od, maxd, sd / (double)n, sa / (double)n, sb / (double)n);
    free(ref); free(ours);
    cuda_sam3_1_destroy(ctx);
    return 0;
}
