/*
 * test_da3.c - Test harness for Depth-Anything-3 monocular depth estimation
 *
 * Usage:
 *   ./test_da3 <da3.gguf|model.safetensors> [-t threads] [-i input.ppm] [-o output.pgm|.exr]
 *   ./test_da3 model.safetensors -i photo.ppm --full -t 4
 *
 * Supports both GGUF and safetensors model files.
 * With --full, runs all available output modalities (depth, pose, rays, gaussians).
 *
 * Build:
 *   make              (uses Makefile)
 */

#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#define DEPTH_ANYTHING3_IMPLEMENTATION
#include "depth_anything3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Generate a synthetic gradient image as uint8 RGB [h][w][3] */
static uint8_t *generate_gradient(int width, int height) {
    uint8_t *img = (uint8_t *)malloc((size_t)width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            img[idx + 0] = (uint8_t)(x * 255 / (width > 1 ? width - 1 : 1));
            img[idx + 1] = (uint8_t)(y * 255 / (height > 1 ? height - 1 : 1));
            img[idx + 2] = (uint8_t)((x + y) * 255 / (width + height > 2 ? width + height - 2 : 1));
        }
    }
    return img;
}

/* Load a PPM (P6) file */
static uint8_t *load_ppm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) { fclose(f); return NULL; }
    int maxval;
    if (fscanf(f, "%d %d %d", w, h, &maxval) != 3) { fclose(f); return NULL; }
    fgetc(f);
    size_t sz = (size_t)(*w) * (*h) * 3;
    uint8_t *img = (uint8_t *)malloc(sz);
    if (fread(img, 1, sz, f) != sz) { free(img); fclose(f); return NULL; }
    fclose(f);
    return img;
}

/* Write 16-bit PGM (P5) */
static void write_pgm16(const char *path, const float *data, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(f, "P5\n%d %d\n65535\n", w, h);
    for (int i = 0; i < w * h; i++) {
        float v = data[i];
        if (v < 0) v = 0;
        uint16_t u = (v > 65535.0f) ? 65535 : (uint16_t)v;
        uint8_t hi = (uint8_t)(u >> 8);
        uint8_t lo = (uint8_t)(u & 0xFF);
        fwrite(&hi, 1, 1, f);
        fwrite(&lo, 1, 1, f);
    }
    fclose(f);
    fprintf(stderr, "Wrote %s (%dx%d)\n", path, w, h);
}

static void print_stats(const char *name, const float *data, int n) {
    if (!data || n == 0) return;
    float mn = data[0], mx = data[0], sum = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
    }
    fprintf(stderr, "  %-16s min=%.4f  max=%.4f  mean=%.4f\n", name, mn, mx, sum / n);
}

static int ends_with(const char *str, const char *suffix) {
    size_t sl = strlen(str), xl = strlen(suffix);
    return sl >= xl && strcmp(str + sl - xl, suffix) == 0;
}

/* Try to find config.json next to safetensors file */
static char *find_config_json(const char *st_path) {
    /* Look for config.json in the same directory */
    const char *slash = strrchr(st_path, '/');
    size_t dir_len = slash ? (size_t)(slash - st_path + 1) : 0;
    char *cfg = (char *)malloc(dir_len + 16);
    if (dir_len > 0) memcpy(cfg, st_path, dir_len);
    strcpy(cfg + dir_len, "config.json");
    FILE *f = fopen(cfg, "r");
    if (f) { fclose(f); return cfg; }
    free(cfg);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <da3.gguf|model.safetensors> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -t <threads>    Number of threads (default: 4)\n");
        fprintf(stderr, "  -i <input.ppm>  Input image (default: synthetic gradient)\n");
        fprintf(stderr, "  -o <output>     Output file (.pgm or .exr)\n");
        fprintf(stderr, "  -c <config>     config.json path (auto-detected if next to .safetensors)\n");
        fprintf(stderr, "  --full          Run all output modalities (depth+pose+rays+gaussians)\n");
        fprintf(stderr, "  --pose          Enable pose output\n");
        fprintf(stderr, "  --rays          Enable rays output\n");
        fprintf(stderr, "  --gaussians     Enable gaussians output\n");
        return 1;
    }

    const char *model_path = argv[1];
    int n_threads = 4;
    const char *input_ppm = NULL;
    const char *output_path = NULL;
    const char *config_path = NULL;
    int output_flags = DA3_OUTPUT_DEPTH;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) input_ppm = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) config_path = argv[++i];
        else if (strcmp(argv[i], "--full") == 0) output_flags = DA3_OUTPUT_ALL;
        else if (strcmp(argv[i], "--pose") == 0) output_flags |= DA3_OUTPUT_POSE;
        else if (strcmp(argv[i], "--rays") == 0) output_flags |= DA3_OUTPUT_RAYS;
        else if (strcmp(argv[i], "--gaussians") == 0) output_flags |= DA3_OUTPUT_GAUSSIANS;
    }

    int is_safetensors = ends_with(model_path, ".safetensors");

    /* Load model */
    da3_model *model = NULL;
    gguf_context *gguf = NULL;

    if (is_safetensors) {
        fprintf(stderr, "Loading safetensors: %s\n", model_path);
        char *auto_config = NULL;
        if (!config_path) {
            auto_config = find_config_json(model_path);
            if (auto_config) {
                config_path = auto_config;
                fprintf(stderr, "Found config: %s\n", config_path);
            }
        }
        model = da3_load_safetensors(model_path, config_path);
        free(auto_config);
        if (!model) { fprintf(stderr, "Failed to load safetensors model\n"); return 1; }
    } else {
        fprintf(stderr, "Loading GGUF: %s\n", model_path);
        gguf = gguf_open(model_path, 1);
        if (!gguf) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
        fprintf(stderr, "GGUF: %llu tensors, %llu KV pairs\n",
                (unsigned long long)gguf->n_tensors, (unsigned long long)gguf->n_kv);
        model = da3_load(gguf);
        if (!model) { fprintf(stderr, "Failed to load model\n"); gguf_close(gguf); return 1; }
    }

    /* Prepare input image */
    int img_w, img_h;
    uint8_t *img;
    if (input_ppm) {
        img = load_ppm(input_ppm, &img_w, &img_h);
        if (!img) { fprintf(stderr, "Failed to load %s\n", input_ppm); da3_free(model); if (gguf) gguf_close(gguf); return 1; }
        fprintf(stderr, "Input: %s (%dx%d)\n", input_ppm, img_w, img_h);
    } else {
        img_w = 518; img_h = 518;
        img = generate_gradient(img_w, img_h);
        fprintf(stderr, "Input: synthetic gradient (%dx%d)\n", img_w, img_h);
    }

    /* Run inference */
    int use_full = (output_flags != DA3_OUTPUT_DEPTH) ||
                   (is_safetensors && (model->has_cam_dec || model->has_aux || model->has_gsdpt));
    fprintf(stderr, "Running DA3 inference (threads=%d, full=%d) ...\n", n_threads, use_full);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (use_full) {
        da3_full_result full = da3_predict_full(model, img, img_w, img_h, n_threads, output_flags);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "Inference time: %.3f s\n", elapsed);

        int npix = full.width * full.height;
        fprintf(stderr, "Output: %dx%d\n", full.width, full.height);
        print_stats("depth", full.depth, npix);
        print_stats("confidence", full.confidence, npix);

        if (full.has_pose) {
            fprintf(stderr, "  pose: t=[%.4f, %.4f, %.4f] q=[%.4f, %.4f, %.4f, %.4f] fov=[%.4f, %.4f]\n",
                    full.pose[0], full.pose[1], full.pose[2],
                    full.pose[3], full.pose[4], full.pose[5], full.pose[6],
                    full.pose[7], full.pose[8]);
        }
        if (full.has_rays) {
            print_stats("rays", full.rays, 6 * npix);
            print_stats("ray_confidence", full.ray_confidence, npix);
        }
        if (full.has_gaussians) {
            int gs_oc = model->has_gsdpt ? model->gsdpt.gs_out_channels : 38;
            print_stats("gaussians", full.gaussians, gs_oc * npix);
        }

        /* Write output */
        if (output_path && full.depth) {
            if (ends_with(output_path, ".pgm")) {
                float mn = full.depth[0], mx = full.depth[0];
                for (int i = 1; i < npix; i++) {
                    if (full.depth[i] < mn) mn = full.depth[i];
                    if (full.depth[i] > mx) mx = full.depth[i];
                }
                float range = mx - mn;
                if (range < 1e-6f) range = 1.0f;
                float *normalized = (float *)malloc((size_t)npix * sizeof(float));
                for (int i = 0; i < npix; i++)
                    normalized[i] = (full.depth[i] - mn) / range * 65535.0f;
                write_pgm16(output_path, normalized, full.width, full.height);
                free(normalized);
            } else {
                fprintf(stderr, "Output format not supported (use .pgm): %s\n", output_path);
            }
        }

        da3_full_result_free(&full);
    } else {
        da3_result result = da3_predict(model, img, img_w, img_h, n_threads);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "Inference time: %.3f s\n", elapsed);

        int npix = result.width * result.height;
        fprintf(stderr, "Output: %dx%d\n", result.width, result.height);
        print_stats("depth", result.depth, npix);
        print_stats("confidence", result.confidence, npix);

        if (result.depth) {
            fprintf(stderr, "Sample depth values (corners + center):\n");
            int cx = result.width / 2, cy = result.height / 2;
            fprintf(stderr, "  TL=%.4f  TR=%.4f  C=%.4f  BL=%.4f  BR=%.4f\n",
                    result.depth[0], result.depth[result.width - 1],
                    result.depth[cy * result.width + cx],
                    result.depth[(result.height - 1) * result.width],
                    result.depth[(result.height - 1) * result.width + result.width - 1]);
        }

        if (output_path && result.depth) {
            float mn = result.depth[0], mx = result.depth[0];
            for (int i = 1; i < npix; i++) {
                if (result.depth[i] < mn) mn = result.depth[i];
                if (result.depth[i] > mx) mx = result.depth[i];
            }
            float range = mx - mn;
            if (range < 1e-6f) range = 1.0f;
            float *normalized = (float *)malloc((size_t)npix * sizeof(float));
            for (int i = 0; i < npix; i++)
                normalized[i] = (result.depth[i] - mn) / range * 65535.0f;
            write_pgm16(output_path, normalized, result.width, result.height);
            free(normalized);
        }

        da3_result_free(&result);
    }

    /* Cleanup */
    free(img);
    da3_free(model);
    if (gguf) gguf_close(gguf);

    fprintf(stderr, "Done.\n");
    return 0;
}
