/* Diff HIP SAM 3 mask decoder (pred_masks + semantic_seg) against reference. */
#include "hip_sam3_runner.h"
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

    hip_sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1008,
                             .device_ordinal = 0, .verbose = 1 };
    hip_sam3_ctx *ctx = hip_sam3_create(&cfg);
    if (!ctx) return 3;
    if (hip_sam3_set_pixel_values(ctx, px) != 0) return 4;
    free(px);
    if (hip_sam3_run_vit(ctx, 31)) return 5;
    if (hip_sam3_run_fpn(ctx)) return 6;
    if (hip_sam3_set_input_ids(ctx, ids, amask_p)) return 7;
    if (hip_sam3_run_text(ctx)) return 8;
    if (hip_sam3_run_detr_enc(ctx)) return 9;
    if (hip_sam3_run_detr_dec(ctx)) return 10;
    if (hip_sam3_run_mask_dec(ctx)) return 11;

    const int Nq = 200, H0 = 288, W0 = 288;
    float *ours = (float *)malloc((size_t)Nq * H0 * W0 * sizeof(float));
    int oq, oh, ow;
    hip_sam3_get_pred_masks(ctx, ours, &oq, &oh, &ow);

    snprintf(path, sizeof(path), "%s/pred_masks.npy", refdir);
    float *ref = (float *)read_npy(path, &nd, d, &isz);
    if (ref) {
        double mx = 0, sd = 0, sa = 0, sb = 0;
        size_t tot = (size_t)Nq * H0 * W0;
        for (size_t i = 0; i < tot; i++) {
            double dv = fabs((double)ref[i] - (double)ours[i]);
            if (dv > mx) mx = dv;
            sd += dv; sa += ref[i]; sb += ours[i];
        }
        fprintf(stderr, "pred_masks (200,288,288): max_abs=%.4e mean_abs=%.4e "
                "(ref mean=%.4e ours mean=%.4e)\n",
                mx, sd / (double)tot, sa / (double)tot, sb / (double)tot);
        free(ref);
    }

    float *osem = (float *)malloc((size_t)H0 * W0 * sizeof(float));
    hip_sam3_get_semantic_seg(ctx, osem, &oh, &ow);
    snprintf(path, sizeof(path), "%s/semantic_seg.npy", refdir);
    float *refs = (float *)read_npy(path, &nd, d, &isz);
    if (refs) {
        double mx = 0, sd = 0, sa = 0, sb = 0;
        size_t tot = (size_t)H0 * W0;
        for (size_t i = 0; i < tot; i++) {
            double dv = fabs((double)refs[i] - (double)osem[i]);
            if (dv > mx) mx = dv;
            sd += dv; sa += refs[i]; sb += osem[i];
        }
        fprintf(stderr, "semantic_seg (288,288): max_abs=%.4e mean_abs=%.4e "
                "(ref mean=%.4e ours mean=%.4e)\n",
                mx, sd / (double)tot, sa / (double)tot, sb / (double)tot);
        free(refs);
    }

    free(ours); free(osem);
    hip_sam3_destroy(ctx);
    return 0;
}
