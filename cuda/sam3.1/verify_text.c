/* Diff HIP SAM 3 CLIP text encoder output against PyTorch reference. */
#include "cuda_sam3_1_runner.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_npy(const char *path, const char *want_dtype,
                       int *ndim, int *dims) {
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
               : strstr(hdr, "i4") || strstr(hdr, "<i4") ? 4
               : strstr(hdr, "i8") || strstr(hdr, "<i8") ? 8
               : 0;
    (void)want_dtype;
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

    char path[512];
    /* Ref dumps input_ids as int64. */
    snprintf(path, sizeof(path), "%s/input_input_ids.npy", refdir);
    int nd, d[8];
    int64_t *ids64 = (int64_t *)read_npy(path, "i8", &nd, d);
    if (!ids64) { fprintf(stderr, "read %s failed\n", path); return 2; }
    size_t ntot = 1; for (int i = 0; i < nd; i++) ntot *= (size_t)d[i];
    if (ntot != 32) {
        fprintf(stderr, "expected 32 input_ids, got %zu\n", ntot); return 2;
    }
    int32_t ids[32];
    for (int i = 0; i < 32; i++) ids[i] = (int32_t)ids64[i];
    free(ids64);

    cuda_sam3_1_config cfg = { .ckpt_path = ckpt, .image_size = 1008,
                             .device_ordinal = 0, .verbose = 1 };
    cuda_sam3_1_ctx *ctx = cuda_sam3_1_create(&cfg);
    if (!ctx) return 3;

    if (cuda_sam3_1_set_input_ids(ctx, ids, NULL) != 0) return 4;
    fprintf(stderr, "running HIP CLIP text encoder (24 layers) ...\n");
    if (cuda_sam3_1_run_text(ctx) != 0) return 5;

    float ours[32 * 1024];
    int out_len, out_dim;
    cuda_sam3_1_get_text_output(ctx, ours, &out_len, &out_dim);
    size_t n = (size_t)out_len * out_dim;

    snprintf(path, sizeof(path), "%s/text_encoder.npy", refdir);
    int rnd, rd[8];
    float *ref = (float *)read_npy(path, "f4", &rnd, rd);
    if (!ref) {
        snprintf(path, sizeof(path), "%s/text_enc.npy", refdir);
        ref = (float *)read_npy(path, "f4", &rnd, rd);
    }
    if (!ref) { fprintf(stderr, "no text_encoder.npy/text_enc.npy\n"); return 6; }
    size_t rn = 1; for (int i = 0; i < rnd; i++) rn *= (size_t)rd[i];
    if (rn != n) {
        fprintf(stderr, "shape mismatch ref=%zu ours=%zu\n", rn, n);
        free(ref); return 7;
    }

    /* Find the number of valid (non-pad) tokens: id != 49407 (EOS/pad). */
    int valid = 0;
    for (int i = 0; i < 32; i++) {
        if (ids[i] == 49407) { valid = i + 1; break; }  /* include EOS */
    }
    if (valid == 0) valid = 32;

    /* Stats over valid tokens only and over all 32. */
    for (int pass = 0; pass < 2; pass++) {
        size_t lo = 0, hi = pass ? n : (size_t)valid * out_dim;
        double sd = 0, maxd = 0;
        for (size_t i = lo; i < hi; i++) {
            double dd = fabs((double)ref[i] - (double)ours[i]);
            sd += dd;
            if (dd > maxd) maxd = dd;
        }
        fprintf(stderr, "hip text_enc (%s): max_abs=%.4e mean_abs=%.4e\n",
                pass ? "all 32" : "valid t<=EOS",
                maxd, sd / (double)(hi - lo));
    }
    free(ref);
    cuda_sam3_1_destroy(ctx);
    return 0;
}
