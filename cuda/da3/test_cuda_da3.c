/*
 * test_cuda_da3.c - Test harness for CUDA DA3 depth estimation runner
 *
 * Loads a DA3 model, runs CUDA inference, prints statistics for all outputs.
 * Supports full DA3NESTED-GIANT-LARGE-1.1 outputs: depth, pose, rays, gaussians.
 *
 * Usage: ./test_cuda_da3 <da3.gguf|model.safetensors> [-i input.ppm] [-o output.pgm]
 *        [--full] [--pose] [--rays] [--gaussians] [-d device_id]
 *
 * Compile with gcc (no nvcc needed).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* Write a float32 array as NumPy .npy format (v1.0) — 2D (h, w) */
static void write_npy_f32(const char *path, const float *data, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    /* NumPy v1.0 format: magic + version + header */
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    /* Header: dict with descr, fortran_order, shape */
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", h, w);
    /* Pad header to multiple of 64 bytes (including 10-byte preamble) */
    int total = 10 + hlen + 1;  /* +1 for newline */
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    /* Data */
    fwrite(data, sizeof(float), (size_t)w * h, f);
    fclose(f);
    fprintf(stderr, "Wrote %s (%dx%d, float32)\n", path, w, h);
}

/* Write a float32 array as NumPy .npy format — 3D (c, h, w) */
static void write_npy_f32_3d(const char *path, const float *data, int c, int h, int w) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d), }", c, h, w);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), (size_t)c * h * w, f);
    fclose(f);
    fprintf(stderr, "Wrote %s (%dx%dx%d, float32)\n", path, c, h, w);
}

/* Write a float32 1D array as NumPy .npy */
static void write_npy_f32_1d(const char *path, const float *data, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }", n);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), (size_t)n, f);
    fclose(f);
    fprintf(stderr, "Wrote %s (%d, float32)\n", path, n);
}

/* GGUF loader */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

/* EXR output support (tinyexr implementation compiled separately as C++) */
#include "../../common/tinyexr.h"
#define EXR_WRITER_IMPLEMENTATION
#include "../../common/exr_writer.h"

/* Image utilities (falsecolor depth, PNG/EXR export) */
#define IMAGE_UTILS_IMPLEMENTATION
#include "../../common/image_utils.h"

/* CUDA DA3 runner */
#include "cuda_da3_runner.h"

/* ---- Helpers ---- */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

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
    /* Skip comments */
    int c2;
    while ((c2 = fgetc(f)) == '#') { while (fgetc(f) != '\n'); }
    ungetc(c2, f);
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
    fprintf(stderr, "  %-12s min=%.4f  max=%.4f  mean=%.4f\n", name, mn, mx, sum / n);
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <da3.gguf|model.safetensors> [-i input.ppm] [-o output.pgm|output.exr]\n"
                        "       [--npy path.npy] [--full] [--pose] [--rays] [--gaussians] [-d device_id] [-v verbosity]\n"
                        "  .exr output: writes raw float channels (depth, confidence, rays, gaussians)\n"
                        "  .pgm output: writes normalized 16-bit depth only\n"
                        "  --npy:       writes raw float32 depth as NumPy .npy file\n",
                argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *input_path = NULL;
    const char *output_pgm = NULL;
    const char *npy_path = NULL;
    const char *npy_dir = NULL;
    const char *resize_mode = NULL;
    int device_id = 0;
    int verbose = 1;
    int output_flags = DA3_OUTPUT_DEPTH;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) input_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output_pgm = argv[++i];
        else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) device_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) verbose = atoi(argv[++i]);
        else if (strcmp(argv[i], "--npy") == 0 && i + 1 < argc) npy_path = argv[++i];
        else if (strcmp(argv[i], "--npy-dir") == 0 && i + 1 < argc) npy_dir = argv[++i];
        else if (strcmp(argv[i], "--resize") == 0 && i + 1 < argc) resize_mode = argv[++i];
        else if (strcmp(argv[i], "--full") == 0) output_flags = DA3_OUTPUT_ALL;
        else if (strcmp(argv[i], "--pose") == 0) output_flags |= DA3_OUTPUT_POSE;
        else if (strcmp(argv[i], "--rays") == 0) output_flags |= DA3_OUTPUT_RAYS;
        else if (strcmp(argv[i], "--gaussians") == 0) output_flags |= DA3_OUTPUT_GAUSSIANS;
    }

    /* Detect file format */
    int is_safetensors = 0;
    {
        size_t len = strlen(model_path);
        if (len > 12 && strcmp(model_path + len - 12, ".safetensors") == 0)
            is_safetensors = 1;
    }

    /* Initialize CUDA runner */
    fprintf(stderr, "\n=== Initializing CUDA runner (device %d) ===\n", device_id);
    cuda_da3_runner *gpu = cuda_da3_init(device_id, verbose);
    if (!gpu) {
        fprintf(stderr, "Failed to init CUDA runner\n");
        return 1;
    }

    gguf_context *gguf = NULL;

    /* Load weights */
    fprintf(stderr, "\n=== Loading weights to GPU ===\n");
    if (is_safetensors) {
        fprintf(stderr, "Loading safetensors: %s\n", model_path);
        /* Try config.json in same directory */
        char config_path[512];
        const char *slash = strrchr(model_path, '/');
        if (slash) {
            int dir_len = (int)(slash - model_path);
            snprintf(config_path, sizeof(config_path), "%.*s/config.json", dir_len, model_path);
        } else {
            snprintf(config_path, sizeof(config_path), "config.json");
        }
        if (cuda_da3_load_safetensors(gpu, model_path, config_path) != 0) {
            fprintf(stderr, "Failed to load safetensors weights\n");
            cuda_da3_free(gpu);
            return 1;
        }
    } else {
        fprintf(stderr, "Loading GGUF: %s\n", model_path);
        gguf = gguf_open(model_path, 1);
        if (!gguf) { fprintf(stderr, "Failed to open GGUF\n"); cuda_da3_free(gpu); return 1; }
        fprintf(stderr, "GGUF: %llu tensors, %llu KV pairs\n",
                (unsigned long long)gguf->n_tensors, (unsigned long long)gguf->n_kv);
        if (cuda_da3_load_weights(gpu, gguf) != 0) {
            fprintf(stderr, "Failed to load weights\n");
            cuda_da3_free(gpu);
            gguf_close(gguf);
            return 1;
        }
    }

    /* Prepare input image */
    int img_w, img_h;
    uint8_t *img;
    if (input_path) {
        img = img_load_resize(input_path, &img_w, &img_h, resize_mode);
        if (!img) {
            fprintf(stderr, "Failed to load %s\n", input_path);
            cuda_da3_free(gpu);
            if (gguf) gguf_close(gguf);
            return 1;
        }
        fprintf(stderr, "\nInput: %s (%dx%d)\n", input_path, img_w, img_h);
    } else {
        img_w = 518; img_h = 518;
        img = generate_gradient(img_w, img_h);
        fprintf(stderr, "\nInput: synthetic gradient (%dx%d)\n", img_w, img_h);
    }

    /* Run inference */
    fprintf(stderr, "\n=== Running CUDA DA3 inference (flags=0x%02x) ===\n", output_flags);
    double t0 = get_time_ms();

    da3_full_result result = cuda_da3_predict_full(gpu, img, img_w, img_h, output_flags, NULL);

    double elapsed = get_time_ms() - t0;
    fprintf(stderr, "\nTotal inference time: %.1f ms\n", elapsed);

    /* Print statistics */
    int npix = result.width * result.height;
    if (result.depth && npix > 0) {
        fprintf(stderr, "Output: %dx%d\n", result.width, result.height);
        print_stats("depth", result.depth, npix);
        print_stats("confidence", result.confidence, npix);

        /* Sample depth values */
        fprintf(stderr, "Sample depth (corners + center):\n");
        int cx = result.width / 2, cy = result.height / 2;
        fprintf(stderr, "  TL=%.4f  TR=%.4f  C=%.4f  BL=%.4f  BR=%.4f\n",
                result.depth[0],
                result.depth[result.width - 1],
                result.depth[cy * result.width + cx],
                result.depth[(result.height - 1) * result.width],
                result.depth[(result.height - 1) * result.width + result.width - 1]);

        /* Depth sanity checks */
        int pass = 1;
        float dmin = result.depth[0], dmax = result.depth[0];
        float cmin = result.confidence ? result.confidence[0] : 0;
        float cmax = result.confidence ? result.confidence[0] : 0;
        for (int i = 0; i < npix; i++) {
            if (result.depth[i] < dmin) dmin = result.depth[i];
            if (result.depth[i] > dmax) dmax = result.depth[i];
            if (result.confidence) {
                if (result.confidence[i] < cmin) cmin = result.confidence[i];
                if (result.confidence[i] > cmax) cmax = result.confidence[i];
            }
        }
        if (dmin == dmax) {
            fprintf(stderr, "WARNING: all depth values are identical (%.4f)\n", dmin);
            pass = 0;
        }
        if (dmin < 0) {
            fprintf(stderr, "WARNING: negative depth values (min=%.4f)\n", dmin);
            pass = 0;
        }
        if (result.confidence && cmin < 1.0f - 1e-5f) {
            fprintf(stderr, "WARNING: confidence below 1.0 (min=%.4f) - expp1 should be >= 1.0\n", cmin);
            pass = 0;
        }
        fprintf(stderr, "\nDepth sanity: %s\n", pass ? "PASS" : "FAIL (see warnings above)");

        /* Write output */
        if (output_pgm && result.depth) {
            size_t olen = strlen(output_pgm);
            int is_exr = (olen > 4 && strcmp(output_pgm + olen - 4, ".exr") == 0);
            int is_png = (olen > 4 && strcmp(output_pgm + olen - 4, ".png") == 0);
            if (is_exr) {
                if (output_flags != DA3_OUTPUT_DEPTH) {
                    write_exr_full(output_pgm, &result, result.width, result.height);
                } else {
                    write_exr_depth(output_pgm, result.depth, result.confidence,
                                    result.width, result.height);
                }
            } else if (is_png) {
                img_write_depth_png(output_pgm, result.depth, result.width, result.height);
            } else {
                /* PGM: normalized 16-bit depth */
                img_write_pgm16(output_pgm, result.depth, result.width, result.height);
            }
            /* Auto-export: falsecolor PNG + fp16 depth EXR alongside */
            {
                char base[512];
                strncpy(base, output_pgm, sizeof(base) - 1);
                base[sizeof(base) - 1] = '\0';
                char *dot = strrchr(base, '.');
                if (dot) *dot = '\0';
                if (!is_png) {
                    char fpath[512];
                    snprintf(fpath, sizeof(fpath), "%s_falsecolor.png", base);
                    img_write_depth_png(fpath, result.depth, result.width, result.height);
                }
                if (!is_exr) {
                    char fpath[512];
                    snprintf(fpath, sizeof(fpath), "%s_depth.exr", base);
                    img_write_depth_exr(fpath, result.depth, result.width, result.height);
                }
            }
        }
        if (npy_path && result.depth)
            write_npy_f32(npy_path, result.depth, result.width, result.height);
    } else {
        fprintf(stderr, "No depth output produced (depth is NULL)\n");
    }

    /* Print pose results */
    if (result.has_pose) {
        fprintf(stderr, "\n--- Pose (CameraDec) ---\n");
        fprintf(stderr, "  translation: [%.6f, %.6f, %.6f]\n",
                result.pose[0], result.pose[1], result.pose[2]);
        fprintf(stderr, "  quaternion:  [%.6f, %.6f, %.6f, %.6f]\n",
                result.pose[3], result.pose[4], result.pose[5], result.pose[6]);
        fprintf(stderr, "  fov:         [%.6f, %.6f]\n",
                result.pose[7], result.pose[8]);
        /* Sanity: check for NaN */
        int pose_ok = 1;
        for (int i = 0; i < 9; i++) {
            if (isnan(result.pose[i]) || isinf(result.pose[i])) {
                fprintf(stderr, "  WARNING: pose[%d] is %s\n", i,
                        isnan(result.pose[i]) ? "NaN" : "Inf");
                pose_ok = 0;
            }
        }
        fprintf(stderr, "  Pose sanity: %s\n", pose_ok ? "PASS" : "FAIL");
    }

    /* Print ray results */
    if (result.has_rays && result.rays) {
        fprintf(stderr, "\n--- Rays (Aux DPT) ---\n");
        for (int c = 0; c < 6; c++) {
            char label[32];
            snprintf(label, sizeof(label), "ray_ch%d", c);
            print_stats(label, result.rays + (size_t)c * npix, npix);
        }
        if (result.ray_confidence)
            print_stats("ray_conf", result.ray_confidence, npix);
        if (result.sky_seg)
            print_stats("sky_seg", result.sky_seg, npix);
    }

    /* Print gaussian results */
    if (result.has_gaussians && result.gaussians) {
        fprintf(stderr, "\n--- Gaussians (GSDPT, 38 channels) ---\n");
        const char *gs_names[] = {
            "offset_x", "offset_y",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
            NULL /* rest are SH + offset_depth + opacity */
        };
        for (int c = 0; c < 38; c++) {
            char label[32];
            if (c < 9 && gs_names[c])
                snprintf(label, sizeof(label), "%s", gs_names[c]);
            else if (c < 36)
                snprintf(label, sizeof(label), "sh_%d", c - 9);
            else if (c == 36)
                snprintf(label, sizeof(label), "off_depth");
            else
                snprintf(label, sizeof(label), "opacity");
            print_stats(label, result.gaussians + (size_t)c * npix, npix);
        }
    }

    /* Print metric depth results */
    if (result.has_metric && result.metric_depth) {
        fprintf(stderr, "\n--- Metric Depth ---\n");
        print_stats("metric", result.metric_depth, npix);
    }

    /* Save all outputs as npy if --npy-dir specified */
    if (npy_dir && npix > 0) {
        char path[512];
        if (result.depth) {
            snprintf(path, sizeof(path), "%s/depth.npy", npy_dir);
            write_npy_f32(path, result.depth, result.width, result.height);
        }
        if (result.confidence) {
            snprintf(path, sizeof(path), "%s/confidence.npy", npy_dir);
            write_npy_f32(path, result.confidence, result.width, result.height);
        }
        if (result.has_pose) {
            snprintf(path, sizeof(path), "%s/pose.npy", npy_dir);
            write_npy_f32_1d(path, result.pose, 9);
        }
        if (result.has_rays && result.rays) {
            snprintf(path, sizeof(path), "%s/rays.npy", npy_dir);
            write_npy_f32_3d(path, result.rays, 6, result.height, result.width);
            if (result.ray_confidence) {
                snprintf(path, sizeof(path), "%s/ray_confidence.npy", npy_dir);
                write_npy_f32(path, result.ray_confidence, result.width, result.height);
            }
        }
        if (result.has_gaussians && result.gaussians) {
            int gs_oc = 38;
            snprintf(path, sizeof(path), "%s/gaussians.npy", npy_dir);
            write_npy_f32_3d(path, result.gaussians, gs_oc, result.height, result.width);
        }
    }

    /* Cleanup */
    free(img);
    da3_full_result_free(&result);
    cuda_da3_free(gpu);
    if (gguf) gguf_close(gguf);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
