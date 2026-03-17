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
 *   stb_image.h        (loading JPEG/PNG/BMP/etc.)
 *   stb_image_write.h  (writing JPEG/PNG/BMP/HDR)
 *   stb_image_resize2.h (optional, general-purpose resizing)
 *   tinyexr.h          (optional, for EXR output — impl must be compiled as C++)
 *
 * API — Loading:
 *   img_load(path, &w, &h)         → uint8 RGB, NULL on failure
 *
 * API — Resizing:
 *   img_resize_ac(src, sw, sh, dw, dh) → uint8 RGB, bilinear align_corners=True
 *   img_resize_ac_f32(src, ch, sw, sh, dw, dh) → float32 CHW version
 *
 * API — Preprocessing:
 *   img_preprocess_da3(rgb, w, h, target, mean, std) → float CHW normalized
 *
 * API — Writing (uint8 RGB):
 *   img_write_ppm(path, rgb, w, h)           → P6 PPM
 *   img_write_png(path, rgb, w, h)           → PNG
 *   img_write_jpg(path, rgb, w, h, quality)  → JPEG (quality 1-100)
 *   img_write_bmp(path, rgb, w, h)           → BMP
 *
 * API — Writing (float):
 *   img_write_pgm16(path, data, w, h)            → 16-bit PGM (auto normalize)
 *   img_write_hdr(path, rgb_f32, w, h)            → Radiance HDR (3-ch float)
 *   img_write_exr(path, data, w, h, n_ch, names)  → OpenEXR (N-ch float32)
 *
 * API — Utility:
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

/* ---- Writers: uint8 RGB ---- */

/* Write 8-bit RGB as P6 PPM. Returns 0 on success. */
int img_write_ppm(const char *path, const uint8_t *rgb, int width, int height);

/* Write 8-bit RGB as PNG. Returns 0 on success. */
int img_write_png(const char *path, const uint8_t *rgb, int width, int height);

/* Write 8-bit RGB as JPEG. quality: 1 (worst) to 100 (best). Returns 0 on success. */
int img_write_jpg(const char *path, const uint8_t *rgb, int width, int height,
                  int quality);

/* Write 8-bit RGB as BMP. Returns 0 on success. */
int img_write_bmp(const char *path, const uint8_t *rgb, int width, int height);

/* ---- Writers: float32 ---- */

/* Write 16-bit PGM (P5) from float array (auto min/max normalize to 0-65535). */
void img_write_pgm16(const char *path, const float *data, int width, int height);

/* Write Radiance HDR from float32 RGB [h][w][3] (HWC layout). Returns 0 on success. */
int img_write_hdr(const char *path, const float *rgb_f32, int width, int height);

/* Write OpenEXR with N float32 channels. Each channel is [height * width] floats.
 * channel_data: array of N pointers to float arrays (one per channel).
 * channel_names: array of N null-terminated name strings.
 * Requires tinyexr.h. Returns 0 on success.
 * NOTE: tinyexr implementation must be compiled as C++ (tinyexr_impl.cc). */
int img_write_exr(const char *path, const float **channel_data,
                  int width, int height, int n_channels,
                  const char **channel_names);

/* Convenience: write single-channel float32 as EXR (e.g., depth map). */
int img_write_exr_1ch(const char *path, const float *data,
                      int width, int height, const char *channel_name);

/* Convenience: write 3-channel float32 as EXR (e.g., RGB). */
int img_write_exr_3ch(const char *path, const float *ch0, const float *ch1,
                      const float *ch2, int width, int height,
                      const char *name0, const char *name1, const char *name2);

/* ---- Utility ---- */

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

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image_write.h"

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

/* ---- PNG / JPEG / BMP writers (via stb_image_write) ---- */

int img_write_png(const char *path, const uint8_t *rgb, int width, int height) {
    int ret = stbi_write_png(path, width, height, 3, rgb, width * 3);
    if (ret) fprintf(stderr, "Wrote %s (%dx%d, PNG)\n", path, width, height);
    else     fprintf(stderr, "img_write_png: failed to write %s\n", path);
    return ret ? 0 : -1;
}

int img_write_jpg(const char *path, const uint8_t *rgb, int width, int height,
                  int quality) {
    int ret = stbi_write_jpg(path, width, height, 3, rgb, quality);
    if (ret) fprintf(stderr, "Wrote %s (%dx%d, JPEG q=%d)\n", path, width, height, quality);
    else     fprintf(stderr, "img_write_jpg: failed to write %s\n", path);
    return ret ? 0 : -1;
}

int img_write_bmp(const char *path, const uint8_t *rgb, int width, int height) {
    int ret = stbi_write_bmp(path, width, height, 3, rgb);
    if (ret) fprintf(stderr, "Wrote %s (%dx%d, BMP)\n", path, width, height);
    else     fprintf(stderr, "img_write_bmp: failed to write %s\n", path);
    return ret ? 0 : -1;
}

/* ---- HDR writer (via stb_image_write) ---- */

int img_write_hdr(const char *path, const float *rgb_f32, int width, int height) {
    /* stbi_write_hdr expects float RGB [h][w][3] in HWC layout */
    int ret = stbi_write_hdr(path, width, height, 3, rgb_f32);
    if (ret) fprintf(stderr, "Wrote %s (%dx%d, HDR)\n", path, width, height);
    else     fprintf(stderr, "img_write_hdr: failed to write %s\n", path);
    return ret ? 0 : -1;
}

/* ---- EXR writer (via tinyexr) ---- */

#ifdef TINYEXR_H_  /* Only compile if tinyexr.h was included before this header */

int img_write_exr(const char *path, const float **channel_data,
                  int width, int height, int n_channels,
                  const char **channel_names) {
    if (n_channels < 1 || n_channels > 128) return -1;

    /* Sort channels alphabetically (EXR requirement) */
    int *order = (int *)malloc((size_t)n_channels * sizeof(int));
    for (int i = 0; i < n_channels; i++) order[i] = i;
    /* Insertion sort by name */
    for (int i = 1; i < n_channels; i++) {
        int key = order[i];
        int j = i - 1;
        while (j >= 0 && strcmp(channel_names[order[j]], channel_names[key]) > 0) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = n_channels;
    image.width = width;
    image.height = height;

    /* Build sorted channel pointer array */
    const float **sorted_ptrs = (const float **)malloc(
        (size_t)n_channels * sizeof(const float *));
    for (int i = 0; i < n_channels; i++)
        sorted_ptrs[i] = channel_data[order[i]];
    image.images = (unsigned char **)sorted_ptrs;

    header.num_channels = n_channels;
    header.channels = (EXRChannelInfo *)malloc(
        sizeof(EXRChannelInfo) * (size_t)n_channels);
    header.pixel_types = (int *)malloc(sizeof(int) * (size_t)n_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * (size_t)n_channels);

    for (int i = 0; i < n_channels; i++) {
        strncpy(header.channels[i].name, channel_names[order[i]], 255);
        header.channels[i].name[255] = '\0';
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    const char *err = NULL;
    int ret = SaveEXRImageToFile(&image, &header, path, &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "img_write_exr: %s: %s\n", path, err ? err : "unknown error");
        FreeEXRErrorMessage(err);
    } else {
        fprintf(stderr, "Wrote %s (%dx%d, %d-ch EXR)\n", path, width, height, n_channels);
    }

    free(order);
    free(sorted_ptrs);
    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
    return ret == TINYEXR_SUCCESS ? 0 : -1;
}

int img_write_exr_1ch(const char *path, const float *data,
                      int width, int height, const char *channel_name) {
    const float *ptrs[1] = {data};
    const char *names[1] = {channel_name};
    return img_write_exr(path, ptrs, width, height, 1, names);
}

int img_write_exr_3ch(const char *path, const float *ch0, const float *ch1,
                      const float *ch2, int width, int height,
                      const char *name0, const char *name1, const char *name2) {
    const float *ptrs[3] = {ch0, ch1, ch2};
    const char *names[3] = {name0, name1, name2};
    return img_write_exr(path, ptrs, width, height, 3, names);
}

#endif /* TINYEXR_H_ */

#endif /* IMAGE_UTILS_IMPLEMENTATION */
#endif /* IMAGE_UTILS_H */
