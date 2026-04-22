/* Compare HIP Phase 1 (preprocess + patch_embed + pos_embed + pre-LN)
 * against /tmp/sam3_ref_<img>/vit_embed.npy. Uses the ref dump's
 * input_pixel_values.npy for bit-close comparison (bypasses stb resize
 * vs PIL BILINEAR mismatch).
 *
 * Usage: verify_patch_embed --ckpt <...> --refdir /tmp/sam3_ref_cat
 */
#include "cuda_sam3_1_runner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *read_npy(const char *path, int *nd, int *dims, size_t *esz)
{
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET);
    uint16_t hl; if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = '\0'; *nd = 0; *esz = 4;
    char *sp = strstr(hdr, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') {
            while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break;
            dims[(*nd)++] = (int)strtol(sp, &sp, 10);
            if (*nd >= 8) break;
        } } }
    size_t n = 1; for (int i = 0; i < *nd; i++) n *= (size_t)dims[i];
    void *d = malloc(n * *esz);
    if (fread(d, *esz, n, f) != n) { free(d); fclose(f); free(hdr); return NULL; }
    fclose(f); free(hdr);
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
    if (!ckpt) {
        fprintf(stderr, "usage: %s --ckpt <sam3.model.safetensors> [--refdir <dir>]\n", argv[0]);
        return 1;
    }

    cuda_sam3_1_config cfg = { .ckpt_path = ckpt, .image_size = 1008,
                             .device_ordinal = 0, .verbose = 1 };
    cuda_sam3_1_ctx *ctx = cuda_sam3_1_create(&cfg);
    if (!ctx) return 2;

    char path[512]; int nd, dims[8]; size_t esz;
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    float *px = (float *)read_npy(path, &nd, dims, &esz);
    if (!px) {
        fprintf(stderr, "couldn't read %s\n", path);
        cuda_sam3_1_destroy(ctx); return 3;
    }
    if (cuda_sam3_1_set_pixel_values(ctx, px)) {
        fprintf(stderr, "set_pixel_values failed\n"); free(px);
        cuda_sam3_1_destroy(ctx); return 4;
    }
    free(px);

    int n_tok = 0, D = 0;
    float *ours = (float *)malloc((size_t)5184 * 1024 * sizeof(float));
    if (cuda_sam3_1_get_vit_embed(ctx, ours, &n_tok, &D)) {
        fprintf(stderr, "get_vit_embed failed\n"); free(ours);
        cuda_sam3_1_destroy(ctx); return 5;
    }
    fprintf(stderr, "ours: (%d, %d)\n", n_tok, D);

    snprintf(path, sizeof(path), "%s/vit_embed.npy", refdir);
    float *ref = (float *)read_npy(path, &nd, dims, &esz);
    if (!ref) {
        fprintf(stderr, "no ref %s — dumping ours[0][:8] only\n", path);
        for (int i = 0; i < 8; i++) fprintf(stderr, "  %.6f\n", ours[i]);
    } else {
        double mx = 0, sum = 0;
        size_t n = (size_t)n_tok * D;
        for (size_t i = 0; i < n; i++) {
            double d = fabs((double)ours[i] - (double)ref[i]);
            if (d > mx) mx = d;
            sum += d;
        }
        fprintf(stderr, "vit_embed: max_abs=%.3e mean_abs=%.3e\n",
                mx, sum / (double)n);
        free(ref);
    }

    free(ours);
    cuda_sam3_1_destroy(ctx);
    return 0;
}
