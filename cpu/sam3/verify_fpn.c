/* Diff sam3_runner's FPN level N output against vit_fpn{N}.npy.
 *
 * Usage: verify_fpn --ckpt <sam3.model.safetensors> --image <img>
 *                   [--refdir /tmp/sam3_ref_cat] [--level 0|1|2|3]
 */
#include "sam3_runner.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
    if (strstr(hdr, "fortran_order': True") ||
        strstr(hdr, "fortran_order':True")) {
        fprintf(stderr, "%s: fortran_order=True not supported\n", path);
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
    float *d = (float *)malloc(n * sizeof(float));
    size_t got = fread(d, sizeof(float), n, f);
    fclose(f); free(hdr);
    if (got != n) { free(d); return NULL; }
    return d;
}

int main(int argc, char **argv)
{
    const char *ckpt = NULL, *img_path = NULL;
    const char *refdir = "/tmp/sam3_ref_cat";
    int level = 2;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt = argv[++i];
        else if (!strcmp(argv[i], "--image")  && i+1 < argc) img_path = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--level")  && i+1 < argc) level = atoi(argv[++i]);
    }
    if (!ckpt || !img_path) {
        fprintf(stderr, "--ckpt and --image required\n"); return 1;
    }

    int H, W, C;
    unsigned char *rgb = stbi_load(img_path, &W, &H, &C, 3);
    if (!rgb) { fprintf(stderr, "can't load %s\n", img_path); return 2; }

    sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1008, .num_threads = 0 };
    sam3_ctx *ctx = sam3_create(&cfg);
    if (!ctx) return 3;

    char path[512];
    snprintf(path, sizeof(path), "%s/input_pixel_values.npy", refdir);
    int pxnd, pxd[8];
    float *ref_px = read_npy_f32(path, &pxnd, pxd);
    if (ref_px) {
        fprintf(stderr, "using ref pixel_values\n");
        if (sam3_set_pixel_values(ctx, ref_px) != 0) return 4;
        free(ref_px);
    } else {
        if (sam3_set_image(ctx, rgb, H, W) != 0) return 4;
    }

    fprintf(stderr, "running ViT + FPN ...\n");
    if (sam3_run_vit(ctx, 31) != 0) return 5;
    if (sam3_run_fpn(ctx) != 0) return 6;

    int c, h, w;
    const float *ours = sam3_get_fpn(ctx, level, &c, &h, &w);
    size_t n = (size_t)c * h * w;
    fprintf(stderr, "fpn level %d: (%d,%d,%d)\n", level, c, h, w);

    snprintf(path, sizeof(path), "%s/vit_fpn%d.npy", refdir, level);
    int nd, d[8]; float *ref = read_npy_f32(path, &nd, d);
    if (!ref) { fprintf(stderr, "cannot read %s\n", path); return 7; }
    size_t rn = 1; for (int i = 0; i < nd; i++) rn *= (size_t)d[i];
    if (rn != n) {
        fprintf(stderr, "shape mismatch: ref=%zu ours=%zu\n", rn, n); return 8;
    }

    double sd = 0, sa = 0, sb = 0, mx = 0; size_t imx = 0;
    for (size_t i = 0; i < n; i++) {
        double dd = fabs((double)ref[i] - (double)ours[i]);
        if (dd > mx) { mx = dd; imx = i; }
        sa += ref[i]; sb += ours[i]; sd += dd;
    }
    fprintf(stderr, "fpn%d: max_abs=%.4e mean_abs=%.4e "
            "(ref mean=%.4e ours mean=%.4e) at idx=%zu\n",
            level, mx, sd / n, sa / n, sb / n, imx);
    free(ref);
    sam3_destroy(ctx);
    stbi_image_free(rgb);
    return 0;
}
