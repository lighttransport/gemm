/* Diff sam3_runner's CLIP text encoder output vs text_enc.npy.
 *
 * Usage: verify_text --ckpt <sam3.model.safetensors> --refdir <dir>
 *
 * Loads input_input_ids.npy (int64 (1,32)) + input_attention_mask.npy
 * (int64 (1,32)) from --refdir and runs sam3_run_text, diffing against
 * text_enc.npy (fp32 (1,32,1024)).
 */
#include "sam3_runner.h"

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
    hdr[hl] = '\0'; *ndim = 0;
    /* dtype detect. */
    *esz = 0;
    if (strstr(hdr, "descr': '<f4") || strstr(hdr, "descr':'<f4")) *esz = 4;
    else if (strstr(hdr, "descr': '<i8") || strstr(hdr, "descr':'<i8")) *esz = 8;
    else if (strstr(hdr, "descr': '<i4") || strstr(hdr, "descr':'<i4")) *esz = 4;
    if (!*esz) {
        fprintf(stderr, "%s: unsupported dtype\n", path);
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
    void *d = malloc(n * *esz);
    size_t got = fread(d, *esz, n, f);
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
    int nd, d[8]; size_t esz;

    snprintf(path, sizeof(path), "%s/input_input_ids.npy", refdir);
    void *ids_raw = read_npy(path, &nd, d, &esz);
    if (!ids_raw) { fprintf(stderr, "cannot read %s\n", path); return 2; }
    int T = d[nd - 1];
    if (T != 32) { fprintf(stderr, "ctx_len=%d, want 32\n", T); return 3; }
    int32_t ids[32]; int32_t mask[32];
    if (esz == 8) {
        int64_t *a = (int64_t *)ids_raw;
        for (int t = 0; t < 32; t++) ids[t] = (int32_t)a[t];
    } else {
        int32_t *a = (int32_t *)ids_raw;
        for (int t = 0; t < 32; t++) ids[t] = a[t];
    }
    free(ids_raw);

    snprintf(path, sizeof(path), "%s/input_attention_mask.npy", refdir);
    void *msk_raw = read_npy(path, &nd, d, &esz);
    if (msk_raw) {
        if (esz == 8) {
            int64_t *a = (int64_t *)msk_raw;
            for (int t = 0; t < 32; t++) mask[t] = (int32_t)a[t];
        } else {
            int32_t *a = (int32_t *)msk_raw;
            for (int t = 0; t < 32; t++) mask[t] = a[t];
        }
        free(msk_raw);
    } else {
        for (int t = 0; t < 32; t++) mask[t] = 1;
    }

    sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1008, .num_threads = 0 };
    sam3_ctx *ctx = sam3_create(&cfg);
    if (!ctx) return 4;

    fprintf(stderr, "running CLIP text encoder ...\n");
    if (sam3_set_input_ids(ctx, ids, mask) != 0) return 5;
    if (sam3_run_text(ctx) != 0) return 6;

    int tl, td;
    const float *ours = sam3_get_text_output(ctx, &tl, &td);
    fprintf(stderr, "text out: (%d, %d)\n", tl, td);

    snprintf(path, sizeof(path), "%s/text_enc.npy", refdir);
    float *ref = (float *)read_npy(path, &nd, d, &esz);
    if (!ref || esz != 4) { fprintf(stderr, "cannot read %s\n", path); return 7; }
    size_t n = (size_t)tl * td;

    /* Find first PAD: positions 0..n_valid-1 should bit-close. */
    int n_valid = 0;
    for (int t = 0; t < 32; t++) if (mask[t]) n_valid = t + 1;

    double sd_v = 0, mx_v = 0; int imx_v = 0;
    double sd_a = 0, mx_a = 0;
    for (int t = 0; t < tl; t++) {
        for (int d0 = 0; d0 < td; d0++) {
            size_t i = (size_t)t * td + d0;
            double dd = fabs((double)ref[i] - (double)ours[i]);
            sd_a += dd; if (dd > mx_a) mx_a = dd;
            if (t < n_valid) {
                sd_v += dd; if (dd > mx_v) { mx_v = dd; imx_v = (int)i; }
            }
        }
    }
    fprintf(stderr, "text_enc (valid t<%d): max_abs=%.4e mean_abs=%.4e at idx=%d\n",
            n_valid, mx_v, sd_v / (n_valid * td), imx_v);
    fprintf(stderr, "text_enc (all 32)    : max_abs=%.4e mean_abs=%.4e\n",
            mx_a, sd_a / n);
    free(ref);
    sam3_destroy(ctx);
    return 0;
}
