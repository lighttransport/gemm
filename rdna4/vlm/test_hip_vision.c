/*
 * test_hip_vision.c - RDNA4/HIP Qwen vision encoder test and benchmark
 *
 * Usage:
 *   ./test_hip_vision <mmproj.gguf> [--image path] [--image-size N]
 *       [--ref embd.bin] [--warmup N] [--iters N] [--f32|--f16|--bf16] [--verbose]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#include "../../common/transformer.h"

#define VISION_ENCODER_IMPLEMENTATION
#include "../../common/vision_encoder.h"

#include "hip_vision_encoder.h"

typedef struct {
    int n_tokens;
    int embd_dim;
    int image_w;
    int image_h;
    float *embd;
} ref_embd;

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static uint8_t *generate_checkerboard(int width, int height, int block_size) {
    uint8_t *rgb = (uint8_t *)malloc((size_t)width * height * 3);
    if (!rgb) return NULL;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int white = ((x / block_size) + (y / block_size)) & 1;
            uint8_t val = white ? 220 : 35;
            rgb[(y * width + x) * 3 + 0] = val;
            rgb[(y * width + x) * 3 + 1] = (uint8_t)(val * 0.8f);
            rgb[(y * width + x) * 3 + 2] = (uint8_t)(val * 0.6f);
        }
    }
    return rgb;
}

static uint8_t *bilinear_resize(const uint8_t *src, int sw, int sh, int dw, int dh) {
    uint8_t *dst = (uint8_t *)malloc((size_t)dw * dh * 3);
    if (!dst) return NULL;
    for (int y = 0; y < dh; y++) {
        float fy = (dh > 1) ? (float)y * (sh - 1) / (dh - 1) : 0.0f;
        int y0 = (int)fy;
        int y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
        float wy = fy - y0;
        for (int x = 0; x < dw; x++) {
            float fx = (dw > 1) ? (float)x * (sw - 1) / (dw - 1) : 0.0f;
            int x0 = (int)fx;
            int x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                float v = src[(y0 * sw + x0) * 3 + c] * (1.0f - wy) * (1.0f - wx)
                        + src[(y0 * sw + x1) * 3 + c] * (1.0f - wy) * wx
                        + src[(y1 * sw + x0) * 3 + c] * wy * (1.0f - wx)
                        + src[(y1 * sw + x1) * 3 + c] * wy * wx;
                dst[(y * dw + x) * 3 + c] = (uint8_t)(v + 0.5f);
            }
        }
    }
    return dst;
}

static int gguf_get_u32_default(const gguf_context *g, const char *key, int def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_UINT32) return (int)g->kv[idx].value.u32;
    if (g->kv[idx].type == GGUF_TYPE_INT32) return g->kv[idx].value.i32;
    return def;
}

static void calc_size_preserved_ratio(int orig_w, int orig_h, int align_size,
                                      int min_pixels, int max_pixels,
                                      int *out_w, int *out_h) {
    int w = (int)((orig_w + align_size / 2) / align_size) * align_size;
    int h = (int)((orig_h + align_size / 2) / align_size) * align_size;
    if (w < align_size) w = align_size;
    if (h < align_size) h = align_size;

    if ((long long)w * h > max_pixels) {
        float beta = sqrtf((float)orig_w * orig_h / (float)max_pixels);
        w = (int)(orig_w / beta / align_size) * align_size;
        h = (int)(orig_h / beta / align_size) * align_size;
        if (w < align_size) w = align_size;
        if (h < align_size) h = align_size;
    } else if ((long long)w * h < min_pixels) {
        float beta = sqrtf((float)min_pixels / ((float)orig_w * orig_h));
        w = ((int)ceilf((float)orig_w * beta / align_size)) * align_size;
        h = ((int)ceilf((float)orig_h * beta / align_size)) * align_size;
    }

    *out_w = w;
    *out_h = h;
}

static int load_ref(const char *path, ref_embd *ref) {
    memset(ref, 0, sizeof(*ref));
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int32_t hdr[4];
    if (fread(hdr, sizeof(int32_t), 4, f) != 4) {
        fclose(f);
        return -1;
    }
    ref->n_tokens = hdr[0];
    ref->embd_dim = hdr[1];
    ref->image_w = hdr[2];
    ref->image_h = hdr[3];
    int n = ref->n_tokens * ref->embd_dim;
    ref->embd = (float *)malloc((size_t)n * sizeof(float));
    if (!ref->embd || fread(ref->embd, sizeof(float), (size_t)n, f) != (size_t)n) {
        free(ref->embd);
        memset(ref, 0, sizeof(*ref));
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

static int compare_ref(const float *gpu, int gpu_tokens, int gpu_dim, const ref_embd *ref) {
    int n_tokens = gpu_tokens < ref->n_tokens ? gpu_tokens : ref->n_tokens;
    int dim = gpu_dim < ref->embd_dim ? gpu_dim : ref->embd_dim;
    int n = n_tokens * dim;
    double sum_diff2 = 0.0;
    double sum_ref2 = 0.0;
    double dot = 0.0;
    double sum_gpu2 = 0.0;
    float max_abs = 0.0f;
    int max_idx = 0;
    int nan_count = 0;
    int inf_count = 0;

    for (int t = 0; t < n_tokens; t++) {
        for (int d = 0; d < dim; d++) {
            int gi = t * gpu_dim + d;
            int ri = t * ref->embd_dim + d;
            float a = gpu[gi];
            float b = ref->embd[ri];
            if (isnan(a) || isnan(b)) { nan_count++; continue; }
            if (isinf(a) || isinf(b)) { inf_count++; continue; }
            float diff = a - b;
            sum_diff2 += (double)diff * diff;
            sum_ref2 += (double)b * b;
            sum_gpu2 += (double)a * a;
            dot += (double)a * b;
            if (fabsf(diff) > max_abs) {
                max_abs = fabsf(diff);
                max_idx = gi;
            }
        }
    }

    double rms = n > 0 ? sqrt(sum_diff2 / n) : 0.0;
    double rel_l2 = sum_ref2 > 0.0 ? sqrt(sum_diff2 / sum_ref2) : sqrt(sum_diff2);
    double cosine = dot / (sqrt(sum_gpu2) * sqrt(sum_ref2) + 1e-30);
    int pass = (gpu_tokens == ref->n_tokens) && (gpu_dim == ref->embd_dim) &&
               nan_count == 0 && inf_count == 0 &&
               rel_l2 < 2e-2 && cosine >= 0.999;

    printf("\n=== llama.cpp reference compare ===\n");
    printf("  HIP:       %d tokens x %d dim\n", gpu_tokens, gpu_dim);
    printf("  reference: %d tokens x %d dim, image %dx%d\n",
           ref->n_tokens, ref->embd_dim, ref->image_w, ref->image_h);
    printf("  compared:  %d floats\n", n);
    printf("  rel_l2:    %.6e\n", rel_l2);
    printf("  rms:       %.6e\n", rms);
    printf("  max_abs:   %.6e at %d\n", max_abs, max_idx);
    printf("  cosine:    %.8f\n", cosine);
    printf("  nan/inf:   %d / %d\n", nan_count, inf_count);
    printf("  result:    %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mmproj.gguf> [--image path] [--image-size N] [--ref embd.bin] [--warmup N] [--iters N] [--verbose]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *image_path = NULL;
    const char *ref_path = NULL;
    int image_size = 0;
    int warmup = 1;
    int iters = 3;
    int verbose = 1;
    int vision_precision = -1;  /* -1=auto, 0=F32, 1=F16, 2=BF16 */

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) image_path = argv[++i];
        else if (strcmp(argv[i], "--image-size") == 0 && i + 1 < argc) image_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ref") == 0 && i + 1 < argc) ref_path = argv[++i];
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--f32") == 0) vision_precision = 0;
        else if (strcmp(argv[i], "--f16") == 0) vision_precision = 1;
        else if (strcmp(argv[i], "--bf16") == 0) vision_precision = 2;
        else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) verbose = 2;
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }
    if (warmup < 0) warmup = 0;
    if (iters < 1) iters = 1;

    ref_embd ref;
    int have_ref = 0;
    if (ref_path) {
        if (load_ref(ref_path, &ref) != 0) {
            fprintf(stderr, "Failed to load reference embedding: %s\n", ref_path);
            return 1;
        }
        have_ref = 1;
    }

    gguf_context *gguf = gguf_open(model_path, 1);
    if (!gguf) {
        fprintf(stderr, "Failed to load GGUF: %s\n", model_path);
        return 1;
    }
    if (vision_precision < 0) vision_precision = hip_vision_infer_precision(gguf);

    printf("=== HIP Vision Encoder %s Test ===\n",
           vision_precision == 1 ? "F16" : vision_precision == 2 ? "BF16" : "FP32");
    printf("Model: %s\n", model_path);

    vision_model *vm = vision_load(gguf);
    if (!vm) {
        fprintf(stderr, "Failed to load vision metadata\n");
        gguf_close(gguf);
        return 1;
    }

    int align = vm->patch_size * vm->spatial_merge;
    int target_w = 0, target_h = 0;
    uint8_t *rgb = NULL;
    int src_w = 0, src_h = 0, src_c = 0;

    if (image_path) {
        rgb = stbi_load(image_path, &src_w, &src_h, &src_c, 3);
        if (!rgb) {
            fprintf(stderr, "Failed to load image: %s\n", image_path);
            vision_free(vm);
            gguf_close(gguf);
            return 1;
        }
        if (have_ref) {
            target_w = ref.image_w;
            target_h = ref.image_h;
        } else if (image_size > 0) {
            target_w = (image_size / align) * align;
            target_h = target_w;
        } else {
            int min_pixels = gguf_get_u32_default(gguf, "clip.vision.image_min_pixels", 8192);
            int max_pixels = gguf_get_u32_default(gguf, "clip.vision.image_max_pixels", 4194304);
            calc_size_preserved_ratio(src_w, src_h, align, min_pixels, max_pixels, &target_w, &target_h);
        }
        if (target_w != src_w || target_h != src_h) {
            uint8_t *resized = bilinear_resize(rgb, src_w, src_h, target_w, target_h);
            stbi_image_free(rgb);
            rgb = resized;
            if (!rgb) {
                fprintf(stderr, "Resize failed\n");
                vision_free(vm);
                gguf_close(gguf);
                return 1;
            }
        }
        printf("Image: %s, %dx%d -> %dx%d\n", image_path, src_w, src_h, target_w, target_h);
    } else {
        target_w = image_size > 0 ? image_size : vm->image_size;
        target_w = (target_w / align) * align;
        if (target_w < align) target_w = align;
        target_h = target_w;
        rgb = generate_checkerboard(target_w, target_h, 32);
        if (!rgb) {
            fprintf(stderr, "Failed to generate checkerboard\n");
            vision_free(vm);
            gguf_close(gguf);
            return 1;
        }
        printf("Image: synthetic checkerboard %dx%d\n", target_w, target_h);
    }

    float *rgb_norm = vision_normalize_image(vm, rgb, target_w, target_h);
    if (image_path) stbi_image_free(rgb);
    else free(rgb);

    hip_vision_runner *hip = hip_vision_init(0, verbose, vision_precision);
    if (!hip) {
        fprintf(stderr, "HIP vision init failed\n");
        free(rgb_norm);
        vision_free(vm);
        gguf_close(gguf);
        return 1;
    }
    hip_vision_set_max_pixels(hip, target_w * target_h);
    if (hip_vision_load_weights(hip, gguf) != 0) {
        fprintf(stderr, "HIP vision weight load failed\n");
        hip_vision_free(hip);
        free(rgb_norm);
        vision_free(vm);
        gguf_close(gguf);
        return 1;
    }

    for (int i = 0; i < warmup; i++) {
        float *tmp = hip_vision_encode(hip, rgb_norm, target_w, target_h);
        free(tmp);
    }

    float *last = NULL;
    double sum_ms = 0.0;
    double min_ms = 1e30;
    double max_ms = 0.0;
    for (int i = 0; i < iters; i++) {
        double t0 = get_time_ms();
        float *out = hip_vision_encode(hip, rgb_norm, target_w, target_h);
        double t1 = get_time_ms();
        if (!out) {
            fprintf(stderr, "HIP vision encode failed\n");
            free(last);
            hip_vision_free(hip);
            free(rgb_norm);
            vision_free(vm);
            gguf_close(gguf);
            return 1;
        }
        free(last);
        last = out;
        double ms = t1 - t0;
        sum_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    int patches_w = target_w / vm->patch_size;
    int patches_h = target_h / vm->patch_size;
    int n_tokens = (patches_w / vm->spatial_merge) * (patches_h / vm->spatial_merge);
    int total_embd = hip_vision_total_embd(hip);
    double mean_ms = sum_ms / iters;

    printf("\n=== Benchmark ===\n");
    printf("  output:    %d tokens x %d dim\n", n_tokens, total_embd);
    printf("  warmup:    %d\n", warmup);
    printf("  iters:     %d\n", iters);
    printf("  mean:      %.3f ms\n", mean_ms);
    printf("  min/max:   %.3f / %.3f ms\n", min_ms, max_ms);
    printf("  tokens/s:  %.2f\n", 1000.0 * n_tokens / mean_ms);

    int rc = 0;
    if (have_ref) rc = compare_ref(last, n_tokens, total_embd, &ref);

    free(last);
    if (have_ref) free(ref.embd);
    hip_vision_free(hip);
    free(rgb_norm);
    vision_free(vm);
    gguf_close(gguf);

    return rc;
}
