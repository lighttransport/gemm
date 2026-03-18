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

/* Load image from any supported format (JPEG, PNG, BMP, PPM, TGA, HDR, etc.)
 * HDR/EXR are auto-tonemapped to uint8 RGB.
 * Returns RGB uint8 array (w * h * 3), or NULL on failure.
 * Caller must free with img_free(). */
uint8_t *img_load(const char *path, int *width, int *height);

/* Load image as float32 RGB [h][w][3] (HWC layout).
 * For LDR formats (JPEG/PNG): converts uint8 to [0,1] float.
 * For HDR/EXR: loads native float values.
 * Caller must free with img_free(). */
float *img_load_f32(const char *path, int *width, int *height);

/* Load and optionally resize in one call. resize_mode:
 *   NULL or "none"      — no resize
 *   "WxH"  (e.g. "640x480") — resize to exact pixel dimensions
 *   "N%"   (e.g. "50%")     — scale by percentage
 *   "Nt"   (e.g. "1369t")   — resize so total patches = N (patch_size=14)
 * Returns RGB uint8, or NULL on failure. Caller must free. */
uint8_t *img_load_resize(const char *path, int *width, int *height,
                         const char *resize_mode);

/* Compute best resolution for a target number of ViT tokens.
 * Preserves aspect ratio, rounds to patch_size grid.
 * out_w, out_h: output pixel dimensions. */
void img_calc_token_size(int src_w, int src_h, int target_tokens,
                         int patch_size, int *out_w, int *out_h);

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

/* ---- Depth visualization ---- */

/* Convert float depth map to Turbo-colormap RGB (uint8).
 * Normalizes depth to [0,1] using min/max. Near=blue, far=red.
 * Caller must free result with img_free(). */
uint8_t *img_depth_to_falsecolor(const float *depth, int width, int height);

/* Convenience: write depth as falsecolor PNG. Returns 0 on success. */
int img_write_depth_png(const char *path, const float *depth, int width, int height);

/* Convenience: write depth as fp16 single-channel EXR (raw, no normalization).
 * Requires tinyexr.h. Returns 0 on success. */
int img_write_depth_exr(const char *path, const float *depth, int width, int height);

/* Auto-export depth: writes both falsecolor PNG and fp16 EXR.
 * base_path: path without extension (e.g. "output/depth").
 * Writes {base_path}.png (falsecolor) and {base_path}.exr (fp16). */
void img_export_depth(const char *base_path, const float *depth, int width, int height);

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
    /* Try EXR first if tinyexr is available */
#ifdef TINYEXR_H_
    {
        size_t plen = strlen(path);
        if (plen > 4 && strcmp(path + plen - 4, ".exr") == 0) {
            float *fdata;
            int w, h;
            const char *err = NULL;
            int ret = LoadEXR(&fdata, &w, &h, path, &err);
            if (ret == TINYEXR_SUCCESS) {
                /* EXR loads as RGBA float; convert to uint8 RGB with tonemap */
                uint8_t *rgb = (uint8_t *)malloc((size_t)w * h * 3);
                for (int i = 0; i < w * h; i++) {
                    for (int c = 0; c < 3; c++) {
                        float v = fdata[i * 4 + c];
                        /* Simple Reinhard tonemap + gamma */
                        v = v / (1.0f + v);
                        v = powf(v > 0 ? v : 0, 1.0f / 2.2f) * 255.0f + 0.5f;
                        rgb[i * 3 + c] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                    }
                }
                free(fdata);
                *width = w; *height = h;
                return rgb;
            }
            if (err) FreeEXRErrorMessage(err);
        }
    }
#endif
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

float *img_load_f32(const char *path, int *width, int *height) {
    /* Try EXR first */
#ifdef TINYEXR_H_
    {
        size_t plen = strlen(path);
        if (plen > 4 && strcmp(path + plen - 4, ".exr") == 0) {
            float *fdata;
            int w, h;
            const char *err = NULL;
            int ret = LoadEXR(&fdata, &w, &h, path, &err);
            if (ret == TINYEXR_SUCCESS) {
                /* EXR loads as RGBA; extract RGB HWC */
                float *rgb = (float *)malloc((size_t)w * h * 3 * sizeof(float));
                for (int i = 0; i < w * h; i++) {
                    rgb[i * 3 + 0] = fdata[i * 4 + 0];
                    rgb[i * 3 + 1] = fdata[i * 4 + 1];
                    rgb[i * 3 + 2] = fdata[i * 4 + 2];
                }
                free(fdata);
                *width = w; *height = h;
                return rgb;
            }
            if (err) FreeEXRErrorMessage(err);
        }
    }
#endif
    /* stbi_loadf handles HDR natively, converts LDR to float [0,1] */
    int w, h, channels;
    float *data = stbi_loadf(path, &w, &h, &channels, 3);
    if (!data) {
        fprintf(stderr, "img_load_f32: failed to load %s: %s\n",
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

/* ---- Token-based resize calculation ---- */

void img_calc_token_size(int src_w, int src_h, int target_tokens,
                         int patch_size, int *out_w, int *out_h) {
    /* Find grid dimensions that give ~target_tokens patches while preserving
     * aspect ratio. grid_w * grid_h ≈ target_tokens. */
    float aspect = (float)src_w / (float)src_h;
    /* grid_h * (grid_h * aspect) = target_tokens → grid_h = sqrt(target_tokens / aspect) */
    float gh = sqrtf((float)target_tokens / aspect);
    float gw = gh * aspect;
    int grid_h = (int)(gh + 0.5f);
    int grid_w = (int)(gw + 0.5f);
    if (grid_h < 1) grid_h = 1;
    if (grid_w < 1) grid_w = 1;
    /* Adjust to get closer to target */
    while (grid_w * grid_h < target_tokens && grid_w * grid_h < target_tokens * 2) {
        if ((float)(grid_w + 1) / grid_h < aspect * 1.1f) grid_w++;
        else grid_h++;
        if (grid_w * grid_h >= target_tokens) break;
    }
    *out_w = grid_w * patch_size;
    *out_h = grid_h * patch_size;
}

/* ---- Load + resize ---- */

uint8_t *img_load_resize(const char *path, int *width, int *height,
                         const char *resize_mode) {
    int w, h;
    uint8_t *img = img_load(path, &w, &h);
    if (!img) return NULL;

    if (!resize_mode || strcmp(resize_mode, "none") == 0) {
        *width = w; *height = h;
        return img;
    }

    int dst_w = w, dst_h = h;

    /* Parse resize mode */
    int len = (int)strlen(resize_mode);
    if (len > 1 && resize_mode[len - 1] == '%') {
        /* Percentage: "50%" */
        float pct = (float)atof(resize_mode) / 100.0f;
        dst_w = (int)(w * pct + 0.5f);
        dst_h = (int)(h * pct + 0.5f);
    } else if (len > 1 && (resize_mode[len - 1] == 't' || resize_mode[len - 1] == 'T')) {
        /* Token count: "1369t" → find best resolution for N tokens */
        int tokens = atoi(resize_mode);
        img_calc_token_size(w, h, tokens, 14, &dst_w, &dst_h);
    } else if (strchr(resize_mode, 'x') || strchr(resize_mode, 'X')) {
        /* Pixel dimensions: "640x480" */
        if (sscanf(resize_mode, "%dx%d", &dst_w, &dst_h) != 2 &&
            sscanf(resize_mode, "%dX%d", &dst_w, &dst_h) != 2) {
            fprintf(stderr, "img_load_resize: invalid resize mode: %s\n", resize_mode);
            *width = w; *height = h;
            return img;
        }
    } else {
        fprintf(stderr, "img_load_resize: invalid resize mode: %s\n"
                "  Use: WxH (e.g. 640x480), N%% (e.g. 50%%), or Nt (e.g. 1369t)\n",
                resize_mode);
        *width = w; *height = h;
        return img;
    }

    if (dst_w < 1) dst_w = 1;
    if (dst_h < 1) dst_h = 1;
    if (dst_w == w && dst_h == h) {
        *width = w; *height = h;
        return img;
    }

    fprintf(stderr, "Resizing: %dx%d → %dx%d\n", w, h, dst_w, dst_h);
    uint8_t *resized = img_resize_ac(img, w, h, dst_w, dst_h);
    free(img);
    if (!resized) return NULL;
    *width = dst_w;
    *height = dst_h;
    return resized;
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

/* ---- Depth falsecolor (matching DA3 PyTorch: Spectral cmap, inverse depth,
 *      2nd/98th percentile normalization) ---- */

/* Spectral colormap 256-entry LUT (matplotlib "Spectral") */
static const uint8_t img_spectral_lut[256][3] = {
    {158,  1, 66},{160,  3, 67},{162,  6, 67},{164,  8, 68},{167, 11, 68},{169, 13, 69},{171, 15, 69},{173, 18, 70},
    {175, 20, 70},{177, 23, 71},{180, 25, 71},{182, 27, 72},{184, 30, 72},{186, 32, 73},{188, 34, 73},{190, 37, 74},
    {193, 39, 74},{195, 42, 75},{197, 44, 75},{199, 46, 76},{201, 49, 76},{203, 51, 77},{205, 54, 77},{208, 56, 78},
    {210, 58, 78},{212, 61, 79},{214, 63, 79},{215, 65, 78},{216, 67, 78},{217, 68, 77},{218, 70, 77},{220, 72, 76},
    {221, 74, 76},{222, 76, 75},{223, 78, 75},{225, 80, 75},{226, 81, 74},{227, 83, 74},{228, 85, 73},{229, 87, 73},
    {231, 89, 72},{232, 91, 72},{233, 92, 71},{234, 94, 71},{235, 96, 70},{237, 98, 70},{238,100, 69},{239,102, 69},
    {240,103, 68},{242,105, 68},{243,107, 67},{244,109, 67},{244,112, 68},{245,114, 69},{245,117, 71},{245,119, 72},
    {246,122, 73},{246,124, 74},{246,127, 75},{247,129, 76},{247,132, 78},{248,134, 79},{248,137, 80},{248,140, 81},
    {249,142, 82},{249,145, 83},{249,147, 85},{250,150, 86},{250,152, 87},{250,155, 88},{251,157, 89},{251,160, 91},
    {251,163, 92},{252,165, 93},{252,168, 94},{252,170, 95},{253,173, 96},{253,175, 98},{253,177, 99},{253,179,101},
    {253,181,103},{253,183,104},{253,185,106},{253,187,108},{253,189,109},{253,191,111},{253,193,113},{253,195,114},
    {253,197,116},{253,199,118},{254,200,119},{254,202,121},{254,204,123},{254,206,124},{254,208,126},{254,210,127},
    {254,212,129},{254,214,131},{254,216,132},{254,218,134},{254,220,136},{254,222,137},{254,224,139},{254,225,141},
    {254,226,143},{254,228,145},{254,229,147},{254,230,149},{254,231,151},{254,233,153},{254,234,155},{254,235,157},
    {254,236,159},{254,237,161},{254,239,163},{255,240,166},{255,241,168},{255,242,170},{255,243,172},{255,245,174},
    {255,246,176},{255,247,178},{255,248,180},{255,250,182},{255,251,184},{255,252,186},{255,253,188},{255,254,190},
    {255,255,190},{254,254,189},{253,254,187},{252,254,186},{251,253,184},{250,253,183},{249,252,181},{248,252,180},
    {247,252,178},{246,251,176},{245,251,175},{244,250,173},{243,250,172},{242,250,170},{241,249,169},{240,249,167},
    {239,249,166},{238,248,164},{237,248,163},{236,247,161},{235,247,160},{234,247,158},{233,246,157},{232,246,155},
    {231,245,154},{230,245,152},{228,244,152},{225,243,153},{223,242,153},{221,241,154},{218,240,154},{216,239,155},
    {214,238,155},{211,237,156},{209,237,156},{207,236,157},{205,235,157},{202,234,158},{200,233,158},{198,232,159},
    {195,231,159},{193,230,160},{191,229,160},{188,228,160},{186,227,161},{184,226,161},{181,225,162},{179,224,162},
    {177,223,163},{174,222,163},{172,221,164},{170,220,164},{167,219,164},{164,218,164},{162,217,164},{159,216,164},
    {156,215,164},{153,214,164},{151,213,164},{148,212,164},{145,211,164},{143,210,164},{140,209,164},{137,208,164},
    {134,207,165},{132,206,165},{129,205,165},{126,204,165},{124,202,165},{121,201,165},{118,200,165},{116,199,165},
    {113,198,165},{110,197,165},{107,196,165},{105,195,165},{102,194,165},{100,192,166},{ 98,189,167},{ 96,187,168},
    { 94,185,169},{ 92,183,170},{ 90,180,171},{ 88,178,172},{ 86,176,173},{ 84,174,173},{ 82,171,174},{ 80,169,175},
    { 78,167,176},{ 75,164,177},{ 73,162,178},{ 71,160,179},{ 69,158,180},{ 67,155,181},{ 65,153,182},{ 63,151,183},
    { 61,149,184},{ 59,146,185},{ 57,144,186},{ 55,142,187},{ 53,139,188},{ 51,137,189},{ 51,135,188},{ 53,133,187},
    { 54,130,186},{ 56,128,185},{ 58,126,184},{ 59,124,183},{ 61,121,182},{ 63,119,181},{ 65,117,180},{ 66,115,179},
    { 68,113,178},{ 70,110,177},{ 72,108,176},{ 73,106,175},{ 75,104,174},{ 77,101,173},{ 78, 99,172},{ 80, 97,170},
    { 82, 95,169},{ 84, 92,168},{ 85, 90,167},{ 87, 88,166},{ 89, 86,165},{ 91, 83,164},{ 92, 81,163},{ 94, 79,162}
};

/* Helper: compare floats for qsort */
static int img__cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

uint8_t *img_depth_to_falsecolor(const float *depth, int width, int height) {
    int npix = width * height;
    uint8_t *rgb = (uint8_t *)malloc((size_t)npix * 3);
    if (!rgb) return NULL;

    /* Step 1: compute inverse depth (1/depth where depth > 0) */
    float *inv = (float *)malloc((size_t)npix * sizeof(float));
    int n_valid = 0;
    for (int i = 0; i < npix; i++) {
        if (depth[i] > 0.0f) {
            inv[i] = 1.0f / depth[i];
            n_valid++;
        } else {
            inv[i] = 0.0f;
        }
    }

    /* Step 2: percentile-based normalization (2nd and 98th percentile) */
    float d_min = 0.0f, d_max = 1.0f;
    if (n_valid > 10) {
        /* Collect valid values and sort for percentile computation */
        float *valid = (float *)malloc((size_t)n_valid * sizeof(float));
        int vi = 0;
        for (int i = 0; i < npix; i++)
            if (depth[i] > 0.0f)
                valid[vi++] = inv[i];
        qsort(valid, (size_t)n_valid, sizeof(float), img__cmp_float);
        /* 2nd and 98th percentile */
        int p2_idx  = (int)(n_valid * 0.02f);
        int p98_idx = (int)(n_valid * 0.98f);
        if (p98_idx >= n_valid) p98_idx = n_valid - 1;
        d_min = valid[p2_idx];
        d_max = valid[p98_idx];
        free(valid);
    }
    if (d_max - d_min < 1e-6f) { d_min -= 1e-6f; d_max += 1e-6f; }

    /* Step 3: normalize, flip (1-x), apply Spectral colormap */
    float inv_range = 1.0f / (d_max - d_min);
    for (int i = 0; i < npix; i++) {
        float t = (inv[i] - d_min) * inv_range;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        t = 1.0f - t;  /* flip: near=warm(red), far=cool(blue) */
        int idx = (int)(t * 255.0f + 0.5f);
        if (idx < 0) idx = 0;
        if (idx > 255) idx = 255;
        rgb[i * 3 + 0] = img_spectral_lut[idx][0];
        rgb[i * 3 + 1] = img_spectral_lut[idx][1];
        rgb[i * 3 + 2] = img_spectral_lut[idx][2];
    }

    free(inv);
    return rgb;
}

int img_write_depth_png(const char *path, const float *depth, int width, int height) {
    uint8_t *rgb = img_depth_to_falsecolor(depth, width, height);
    if (!rgb) return -1;
    int ret = img_write_png(path, rgb, width, height);
    free(rgb);
    return ret;
}

#ifdef TINYEXR_H_
int img_write_depth_exr(const char *path, const float *depth, int width, int height) {
    /* Write as single-channel fp16 EXR */
    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 1;
    image.width = width;
    image.height = height;
    const float *ptrs[1] = {depth};
    image.images = (unsigned char **)ptrs;

    header.num_channels = 1;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo));
    strncpy(header.channels[0].name, "depth", 255);
    header.pixel_types = (int *)malloc(sizeof(int));
    header.requested_pixel_types = (int *)malloc(sizeof(int));
    header.pixel_types[0] = TINYEXR_PIXELTYPE_FLOAT;
    header.requested_pixel_types[0] = TINYEXR_PIXELTYPE_HALF; /* fp16 output */

    const char *err = NULL;
    int ret = SaveEXRImageToFile(&image, &header, path, &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "img_write_depth_exr: %s: %s\n", path, err ? err : "unknown");
        FreeEXRErrorMessage(err);
    } else {
        fprintf(stderr, "Wrote %s (%dx%d, fp16 depth EXR)\n", path, width, height);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
    return ret == TINYEXR_SUCCESS ? 0 : -1;
}
#endif /* TINYEXR_H_ */

void img_export_depth(const char *base_path, const float *depth, int width, int height) {
    char path[512];
    snprintf(path, sizeof(path), "%s.png", base_path);
    img_write_depth_png(path, depth, width, height);
#ifdef TINYEXR_H_
    snprintf(path, sizeof(path), "%s.exr", base_path);
    img_write_depth_exr(path, depth, width, height);
#endif
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
