#include "cuda_sam2_runner.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_npy_u8(const char *path, const uint8_t *data, int n, int h, int w) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    char hdr[256];
    int L = snprintf(hdr, sizeof(hdr),
                     "{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d, %d), }",
                     n, h, w);
    while ((L + 10) % 16 != 0) hdr[L++] = ' ';
    hdr[L++] = '\n';
    uint8_t magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};
    uint16_t hl = (uint16_t)L;
    memcpy(magic + 8, &hl, 2);
    fwrite(magic, 1, 10, f);
    fwrite(hdr, 1, L, f);
    fwrite(data, 1, (size_t)n * h * w, f);
    fclose(f);
}

static void usage(const char *p) {
    fprintf(stderr,
            "Usage: %s <sam2 model dir|ckpt> <image>\n"
            "       [--point <x> <y> <label>]... [--box <x0> <y0> <x1> <y1>] [-o out.npy]\n",
            p);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    const char *ckpt = argv[1];
    const char *img_path = argv[2];
    float points[64];
    int32_t labels[32];
    int n_points = 0;
    int has_box = 0;
    float box[4] = {0, 0, 0, 0};
    const char *out_path = NULL;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--point") && i + 3 < argc) {
            if (n_points >= 32) {
                fprintf(stderr, "too many points (max 32)\n");
                return 1;
            }
            points[n_points * 2 + 0] = (float)atof(argv[++i]);
            points[n_points * 2 + 1] = (float)atof(argv[++i]);
            labels[n_points] = (int32_t)atoi(argv[++i]);
            n_points++;
        } else if (!strcmp(argv[i], "--box") && i + 4 < argc) {
            box[0] = (float)atof(argv[++i]);
            box[1] = (float)atof(argv[++i]);
            box[2] = (float)atof(argv[++i]);
            box[3] = (float)atof(argv[++i]);
            has_box = 1;
        } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
            out_path = argv[++i];
        }
        else { usage(argv[0]); return 1; }
    }

    int w, h, c;
    uint8_t *rgb = stbi_load(img_path, &w, &h, &c, 3);
    if (!rgb) {
        fprintf(stderr, "failed to load image: %s\n", img_path);
        return 2;
    }

    cuda_sam2_config cfg = {
        .ckpt_path = ckpt,
        .image_size = 1024,
        .device_ordinal = 0,
        .verbose = 1,
    };
    cuda_sam2_ctx *ctx = cuda_sam2_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "cuda_sam2_create failed\n");
        stbi_image_free(rgb);
        return 3;
    }

    if (cuda_sam2_set_image(ctx, rgb, h, w) != 0) {
        fprintf(stderr, "cuda_sam2_set_image failed\n");
        cuda_sam2_destroy(ctx);
        stbi_image_free(rgb);
        return 4;
    }

    if (n_points == 0 && !has_box) {
        /* default prompt: image center foreground */
        points[0] = 0.5f * (float)w;
        points[1] = 0.5f * (float)h;
        labels[0] = 1;
        n_points = 1;
    }

    if (n_points > 0) {
        if (cuda_sam2_set_points(ctx, points, labels, n_points) != 0) {
            fprintf(stderr, "cuda_sam2_set_points failed\n");
            cuda_sam2_destroy(ctx);
            stbi_image_free(rgb);
            return 5;
        }
    }
    if (has_box) {
        if (cuda_sam2_set_box(ctx, box[0], box[1], box[2], box[3]) != 0) {
            fprintf(stderr, "cuda_sam2_set_box failed\n");
            cuda_sam2_destroy(ctx);
            stbi_image_free(rgb);
            return 5;
        }
    }

    if (cuda_sam2_run(ctx) != 0) {
        fprintf(stderr, "cuda_sam2_run failed\n");
        cuda_sam2_destroy(ctx);
        stbi_image_free(rgb);
        return 6;
    }

    int n = 0, mh = 0, mw = 0;
    const float *scores = cuda_sam2_get_scores(ctx, &n);
    const uint8_t *masks = cuda_sam2_get_masks(ctx, &n, &mh, &mw);
    fprintf(stderr, "sam2 result: n=%d mask=%dx%d\n", n, mh, mw);
    for (int i = 0; i < n && i < 8; i++) {
        fprintf(stderr, "  [%d] score=%.6f\n", i, scores ? scores[i] : -1.0f);
    }
    if (out_path && masks && n > 0 && mh > 0 && mw > 0) {
        write_npy_u8(out_path, masks, n, mh, mw);
        fprintf(stderr, "wrote %s (%d, %d, %d)\n", out_path, n, mh, mw);
    }

    cuda_sam2_destroy(ctx);
    stbi_image_free(rgb);
    return 0;
}
