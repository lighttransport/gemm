/* Diff dot-product scoring logits vs reference.
 *
 * Usage: verify_dot_score --ckpt <sam3.model.safetensors> --image <img>
 *                         [--refdir /tmp/sam3_ref_cat]
 *
 * Compares all 6 layers against dot_product_scoring.npy (6,1,200,1)
 * and last layer against pred_logits.npy (1,200).
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
    if (!rgb) { fprintf(stderr, "bad image\n"); return 2; }

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

    fprintf(stderr, "running ViT + FPN + text + DETR + dot_score ...\n");
    if (sam3_run_vit(ctx, 31)) return 4;
    if (sam3_run_fpn(ctx)) return 5;
    if (sam3_set_input_ids(ctx, ids, mask)) return 6;
    if (sam3_run_text(ctx)) return 7;
    if (sam3_run_detr_enc(ctx)) return 8;
    if (sam3_run_detr_dec(ctx)) return 9;
    if (sam3_run_dot_score(ctx)) return 10;

    const float *ours = sam3_get_dot_scores(ctx);

    /* Compare full (6,200) vs dot_product_scoring.npy (6,1,200,1). */
    snprintf(path, sizeof(path), "%s/dot_product_scoring.npy", refdir);
    float *ref = (float *)read_npy(path, &nd, d, &esz);
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

    /* Last layer vs pred_logits.npy. */
    snprintf(path, sizeof(path), "%s/pred_logits.npy", refdir);
    float *refp = (float *)read_npy(path, &nd, d, &esz);
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

    sam3_destroy(ctx); stbi_image_free(rgb);
    return 0;
}
