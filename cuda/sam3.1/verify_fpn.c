/* Diff HIP SAM 3 FPN outputs against PyTorch reference dumps. */
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
    const char *ckpt = NULL;
    const char *refdir = "/tmp/sam3_ref_cat";
    int level = -1;  /* -1 = all */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--level")  && i+1 < argc) level = atoi(argv[++i]);
    }
    if (!ckpt) { fprintf(stderr, "--ckpt required\n"); return 1; }

    cuda_sam3_1_config cfg = { .ckpt_path = ckpt, .image_size = 1008,
                             .device_ordinal = 0, .verbose = 1 };
    cuda_sam3_1_ctx *ctx = cuda_sam3_1_create(&cfg);
    if (!ctx) return 2;

    char path[512];
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    int pxnd, pxd[8];
    float *ref_px = read_npy_f32(path, &pxnd, pxd);
    if (!ref_px) { fprintf(stderr, "need %s\n", path); return 3; }
    if (cuda_sam3_1_set_pixel_values(ctx, ref_px) != 0) return 3;
    free(ref_px);

    fprintf(stderr, "running ViT through block 31 ...\n");
    if (cuda_sam3_1_run_vit(ctx, 31) != 0) return 4;
    fprintf(stderr, "running FPN neck ...\n");
    if (cuda_sam3_1_run_fpn(ctx) != 0) return 5;

    int lo = (level < 0) ? 0 : level;
    int hi = (level < 0) ? 4 : level + 1;
    for (int li = lo; li < hi; li++) {
        int oc, oh, ow;
        size_t cap = (size_t)256 * 288 * 288;
        float *ours = (float *)malloc(cap * 4);
        cuda_sam3_1_get_fpn(ctx, li, ours, &oc, &oh, &ow);
        size_t n = (size_t)oc * oh * ow;

        snprintf(path, sizeof(path), "%s/vit_fpn%d.npy", refdir, li);
        int nd, d[8]; float *ref = read_npy_f32(path, &nd, d);
        if (!ref) { fprintf(stderr, "cannot read %s\n", path); free(ours); continue; }
        size_t rn = 1; for (int i = 0; i < nd; i++) rn *= (size_t)d[i];
        if (rn != n) {
            fprintf(stderr, "fpn%d shape mismatch: ref=%zu ours=%zu\n", li, rn, n);
            free(ref); free(ours); continue;
        }
        double sd = 0, sa = 0, sb = 0, maxd = 0;
        size_t gt01 = 0, gt1 = 0;
        for (size_t i = 0; i < n; i++) {
            double dd = fabs((double)ref[i] - (double)ours[i]);
            sa += ref[i]; sb += ours[i]; sd += dd;
            if (dd > maxd) maxd = dd;
            if (dd > 0.1) gt01++;
            if (dd > 1.0) gt1++;
        }
        fprintf(stderr,
                "fpn%d (%d,%d,%d): max_abs=%.4e mean_abs=%.4e "
                "(ref mean=%.4e ours mean=%.4e) >0.1=%zu >1.0=%zu\n",
                li, oc, oh, ow, maxd, sd / (double)n,
                sa / (double)n, sb / (double)n, gt01, gt1);
        free(ref); free(ours);
    }
    cuda_sam3_1_destroy(ctx);
    return 0;
}
