/* Diff HIP SAM 3 dot_product_scoring against reference. */
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
    int itemsz = strstr(hdr, "f4") ? 4 : strstr(hdr, "i8") ? 8 : 4;
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
        if (!strcmp(argv[i], "--ckpt") && i+1 < argc) ckpt = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
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
    if (cuda_sam3_run_vit(ctx, 31) != 0) return 5;
    if (cuda_sam3_run_fpn(ctx) != 0) return 6;
    if (cuda_sam3_set_input_ids(ctx, ids, amask_p) != 0) return 7;
    if (cuda_sam3_run_text(ctx) != 0) return 8;
    if (cuda_sam3_run_detr_enc(ctx) != 0) return 9;
    if (cuda_sam3_run_detr_dec(ctx) != 0) return 10;
    if (cuda_sam3_run_dot_score(ctx) != 0) return 11;

    float ours[6 * 200];
    cuda_sam3_get_dot_scores(ctx, ours);

    snprintf(path, sizeof(path), "%s/dot_product_scoring.npy", refdir);
    float *ref = (float *)read_npy(path, &nd, d, &isz);
    if (ref) {
        for (int li = 0; li < 6; li++) {
            double mx = 0, sd = 0;
            for (int q = 0; q < 200; q++) {
                double dv = fabs((double)ref[li*200 + q] - (double)ours[li*200 + q]);
                if (dv > mx) mx = dv; sd += dv;
            }
            fprintf(stderr, "dot_score L%d: max_abs=%.4e mean_abs=%.4e\n",
                    li, mx, sd / 200.0);
        }
        free(ref);
    }

    snprintf(path, sizeof(path), "%s/pred_logits.npy", refdir);
    float *refp = (float *)read_npy(path, &nd, d, &isz);
    if (refp) {
        double mx = 0, sd = 0, sa = 0, sb = 0;
        for (int q = 0; q < 200; q++) {
            double dv = fabs((double)refp[q] - (double)ours[5*200 + q]);
            if (dv > mx) mx = dv;
            sa += refp[q]; sb += ours[5*200 + q]; sd += dv;
        }
        fprintf(stderr, "pred_logits (last): max_abs=%.4e mean_abs=%.4e "
                "(ref mean=%.4e ours mean=%.4e)\n",
                mx, sd/200.0, sa/200.0, sb/200.0);
        free(refp);
    }
    cuda_sam3_destroy(ctx);
    return 0;
}
