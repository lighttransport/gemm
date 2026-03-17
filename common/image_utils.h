/*
 * image_utils.h — Image loading, resizing, and preprocessing utilities
 *
 * Single-header library for loading images (via stb_image), resizing with
 * align_corners=True bilinear interpolation (matching PyTorch F.interpolate),
 * and ImageNet normalization for DA3/ViT models.
 *
 * Usage:
 *   // In ONE .c file:
 *   #define IMAGE_UTILS_IMPLEMENTATION
 *   #include "image_utils.h"
 *
 *   // In other files:
 *   #include "image_utils.h"
 *
 * Dependencies:
 *   stb_image.h (for loading JPEG/PNG/BMP/etc.)
 *   stb_image_resize2.h (optional, for general-purpose resizing)
 *
 * API:
 *   img_load(path, &w, &h)         → uint8 RGB, NULL on failure
 *   img_resize_ac(src, sw, sh, dw, dh) → uint8 RGB, bilinear align_corners=True
 *   img_preprocess_da3(rgb, w, h, target, mean[3], std[3]) → float CHW normalized
 *   img_write_ppm(path, rgb, w, h) → write P6 PPM
 *   img_free(ptr)                  → free loaded/resized image
 */
#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Load image from any supported format (JPEG, PNG, BMP, PPM, TGA, etc.)
 * Returns RGB uint8 array (w * h * 3), or NULL on failure.
 * Caller must free with img_free(). */
uint8_t *img_load(const char *path, int *width, int *height);

/* Bilinear resize with align_corners=True (matches PyTorch F.interpolate).
 * Input/output: RGB uint8 [h][w][3].
 * Caller must free result with img_free(). */
uint8_t *img_resize_ac(const uint8_t *src, int src_w, int src_h,
                        int dst_w, int dst_h);

/* Bilinear resize with align_corners=True, float32 CHW input/output.
 * src: [channels, src_h, src_w], dst: [channels, dst_h, dst_w].
 * Caller must free result. */
float *img_resize_ac_f32(const float *src, int channels,
                         int src_w, int src_h, int dst_w, int dst_h);

/* Preprocess image for DA3/ViT inference:
 *   1. Bilinear resize (align_corners=True) to target_size x target_size
 *   2. Convert to float32, normalize: (pixel/255 - mean) / std
 *   3. Output layout: CHW [3, target_size, target_size]
 *
 * mean/std: ImageNet defaults if NULL ([0.485,0.456,0.406], [0.229,0.224,0.225])
 * Caller must free result. */
float *img_preprocess_da3(const uint8_t *rgb, int width, int height,
                          int target_size, const float *mean, const float *std);

/* Write 8-bit RGB as P6 PPM. Returns 0 on success. */
int img_write_ppm(const char *path, const uint8_t *rgb, int width, int height);

/* Write 16-bit PGM (P5) from float array (auto min/max normalize). */
void img_write_pgm16(const char *path, const float *data, int width, int height);

/* Free image memory allocated by img_load / img_resize_ac / img_preprocess_da3. */
void img_free(void *ptr);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
/* Implementation                                                           */
/* ======================================================================== */

#ifdef IMAGE_UTILS_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- stb_image loading ---- */

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

uint8_t *img_load(const char *path, int *width, int *height) {
    int w, h, channels;
    uint8_t *data = stbi_load(path, &w, &h, &channels, 3); /* force RGB */
    if (!data) {
        fprintf(stderr, "img_load: failed to load %s: %s\n",
                path, stbi_failure_reason());
        return NULL;
    }
    *width = w;
    *height = h;
    return data;
}

void img_free(void *ptr) {
    free(ptr);  /* stbi_load uses malloc, so free() works */
}

/* ---- Bilinear resize with align_corners=True ----
 *
 * PyTorch's F.interpolate(mode='bilinear', align_corners=True) uses:
 *   src_coord = dst_coord * (src_size - 1) / (dst_size - 1)
 *
 * This maps the first pixel center to 0 and the last pixel center to
 * src_size-1, ensuring exact corner alignment.
 */

uint8_t *img_resize_ac(const uint8_t *src, int sw, int sh,
                        int dw, int dh) {
    uint8_t *dst = (uint8_t *)malloc((size_t)dw * dh * 3);
    if (!dst) return NULL;

    for (int dy = 0; dy < dh; dy++) {
        float fy = (dh > 1) ? (float)dy * (sh - 1) / (dh - 1) : 0.0f;
        int y0 = (int)fy;
        int y1 = y0 + 1 < sh ? y0 + 1 : y0;
        float wy = fy - y0;

        for (int dx = 0; dx < dw; dx++) {
            float fx = (dw > 1) ? (float)dx * (sw - 1) / (dw - 1) : 0.0f;
            int x0 = (int)fx;
            int x1 = x0 + 1 < sw ? x0 + 1 : x0;
            float wx = fx - x0;

            for (int c = 0; c < 3; c++) {
                float v = (float)src[(y0 * sw + x0) * 3 + c] * (1 - wy) * (1 - wx)
                        + (float)src[(y0 * sw + x1) * 3 + c] * (1 - wy) * wx
                        + (float)src[(y1 * sw + x0) * 3 + c] * wy * (1 - wx)
                        + (float)src[(y1 * sw + x1) * 3 + c] * wy * wx;
                int iv = (int)(v + 0.5f);
                dst[(dy * dw + dx) * 3 + c] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
            }
        }
    }
    return dst;
}

float *img_resize_ac_f32(const float *src, int channels,
                         int sw, int sh, int dw, int dh) {
    float *dst = (float *)malloc((size_t)channels * dw * dh * sizeof(float));
    if (!dst) return NULL;

    for (int c = 0; c < channels; c++) {
        for (int dy = 0; dy < dh; dy++) {
            float fy = (dh > 1) ? (float)dy * (sh - 1) / (dh - 1) : 0.0f;
            int y0 = (int)fy;
            int y1 = y0 + 1 < sh ? y0 + 1 : y0;
            float wy = fy - y0;

            for (int dx = 0; dx < dw; dx++) {
                float fx = (dw > 1) ? (float)dx * (sw - 1) / (dw - 1) : 0.0f;
                int x0 = (int)fx;
                int x1 = x0 + 1 < sw ? x0 + 1 : x0;
                float wx = fx - x0;

                float v = src[c * sh * sw + y0 * sw + x0] * (1 - wy) * (1 - wx)
                        + src[c * sh * sw + y0 * sw + x1] * (1 - wy) * wx
                        + src[c * sh * sw + y1 * sw + x0] * wy * (1 - wx)
                        + src[c * sh * sw + y1 * sw + x1] * wy * wx;
                dst[c * dh * dw + dy * dw + dx] = v;
            }
        }
    }
    return dst;
}

/* ---- DA3 preprocessing: resize + normalize ---- */

static const float img_imagenet_mean[3] = {0.485f, 0.456f, 0.406f};
static const float img_imagenet_std[3]  = {0.229f, 0.224f, 0.225f};

float *img_preprocess_da3(const uint8_t *rgb, int width, int height,
                          int target_size, const float *mean, const float *std) {
    if (!mean) mean = img_imagenet_mean;
    if (!std)  std  = img_imagenet_std;

    int ts = target_size;
    float *out = (float *)malloc((size_t)3 * ts * ts * sizeof(float));
    if (!out) return NULL;

    /* Combined bilinear resize + float conversion + normalization */
    for (int c = 0; c < 3; c++) {
        for (int dy = 0; dy < ts; dy++) {
            float fy = (ts > 1) ? (float)dy * (height - 1) / (ts - 1) : 0.0f;
            int y0 = (int)fy;
            int y1 = y0 + 1 < height ? y0 + 1 : y0;
            float wy = fy - y0;

            for (int dx = 0; dx < ts; dx++) {
                float fx = (ts > 1) ? (float)dx * (width - 1) / (ts - 1) : 0.0f;
                int x0 = (int)fx;
                int x1 = x0 + 1 < width ? x0 + 1 : x0;
                float wx = fx - x0;

                float v = (float)rgb[(y0 * width + x0) * 3 + c] * (1 - wy) * (1 - wx)
                        + (float)rgb[(y0 * width + x1) * 3 + c] * (1 - wy) * wx
                        + (float)rgb[(y1 * width + x0) * 3 + c] * wy * (1 - wx)
                        + (float)rgb[(y1 * width + x1) * 3 + c] * wy * wx;
                out[c * ts * ts + dy * ts + dx] = (v / 255.0f - mean[c]) / std[c];
            }
        }
    }
    return out;
}

/* ---- PPM writer ---- */

int img_write_ppm(const char *path, const uint8_t *rgb, int width, int height) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(rgb, 1, (size_t)width * height * 3, f);
    fclose(f);
    return 0;
}

/* ---- PGM writer (16-bit, normalized) ---- */

void img_write_pgm16(const char *path, const float *data, int width, int height) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(f, "P5\n%d %d\n65535\n", width, height);

    int n = width * height;
    float mn = data[0], mx = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    float range = mx - mn;
    if (range < 1e-6f) range = 1.0f;

    for (int i = 0; i < n; i++) {
        float v = (data[i] - mn) / range * 65535.0f;
        if (v < 0) v = 0;
        uint16_t u = (v > 65535.0f) ? 65535 : (uint16_t)v;
        uint8_t hi = (uint8_t)(u >> 8);
        uint8_t lo = (uint8_t)(u & 0xFF);
        fwrite(&hi, 1, 1, f);
        fwrite(&lo, 1, 1, f);
    }
    fclose(f);
    fprintf(stderr, "Wrote %s (%dx%d)\n", path, width, height);
}

#endif /* IMAGE_UTILS_IMPLEMENTATION */
#endif /* IMAGE_UTILS_H */
