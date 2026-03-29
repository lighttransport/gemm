/*
 * test_vulkan_da3.cc - Test harness for Vulkan DA3 depth estimation
 *
 * Usage: ./test_vulkan_da3 <model.safetensors> [-i input.jpg] [-o output.png]
 *        [--npy path.npy] [-d device_id] [-v verbosity]
 *        [--shader-dir path] [--full] [--pose] [--rays] [--gaussians]
 *
 * Matches the CLI of test_hip_da3.c for easy comparison.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ctime>

/* stb_image for JPEG/PNG loading (implementation in stb_impl.c) */
#include "../../common/stb_image.h"
#include "../../common/stb_image_write.h"
#include "../../common/image_utils.h"

/* Vulkan DA3 runner */
#include "vulkan_da3_runner.h"

/* ---- Helpers ---- */

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Write float32 array as NumPy .npy (2D or 3D) */
static void write_npy_f32(const char *path, const float *data, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", h, w);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), (size_t)w * h, f);
    fclose(f);
    fprintf(stderr, "Wrote %s (%dx%d, float32)\n", path, w, h);
}

/* Write 3D float array as NumPy .npy */
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
    int total_hdr = 10 + hlen + 1;
    int pad = ((total_hdr + 63) / 64) * 64 - total_hdr;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), (size_t)c * h * w, f);
    fclose(f);
    fprintf(stderr, "Wrote %s (%dx%dx%d, float32)\n", path, c, h, w);
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
        fprintf(stderr, "Usage: %s <model.safetensors> [-i input.jpg] [-o output.png]\n"
                        "       [--npy path.npy] [-d device_id] [-v verbosity]\n"
                        "       [--shader-dir path] [--full] [--pose] [--rays] [--gaussians]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *input_path = nullptr;
    const char *output_path = nullptr;
    const char *npy_path = nullptr;
    const char *shader_dir = nullptr;
    int device_id = 0;
    int verbose = 1;
    int output_flags = DA3_OUTPUT_DEPTH;
    bool flag_full = false;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) input_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) device_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) verbose = atoi(argv[++i]);
        else if (strcmp(argv[i], "--npy") == 0 && i + 1 < argc) npy_path = argv[++i];
        else if (strcmp(argv[i], "--shader-dir") == 0 && i + 1 < argc) shader_dir = argv[++i];
        else if (strcmp(argv[i], "--full") == 0) { flag_full = true; output_flags = DA3_OUTPUT_ALL; }
        else if (strcmp(argv[i], "--pose") == 0) output_flags |= DA3_OUTPUT_POSE;
        else if (strcmp(argv[i], "--rays") == 0) output_flags |= DA3_OUTPUT_RAYS;
        else if (strcmp(argv[i], "--gaussians") == 0) output_flags |= DA3_OUTPUT_GAUSSIANS;
    }

    /* Derive config.json path */
    char config_path[512];
    const char *slash = strrchr(model_path, '/');
    if (slash) {
        int dir_len = (int)(slash - model_path);
        snprintf(config_path, sizeof(config_path), "%.*s/config.json", dir_len, model_path);
    } else {
        snprintf(config_path, sizeof(config_path), "config.json");
    }

    /* Initialize Vulkan runner */
    fprintf(stderr, "\n=== Initializing Vulkan DA3 runner (device %d) ===\n", device_id);
    vulkan_da3_runner *gpu = vulkan_da3_init(device_id, verbose, shader_dir);
    if (!gpu) {
        fprintf(stderr, "Failed to init Vulkan runner\n");
        return 1;
    }

    /* Load weights */
    fprintf(stderr, "\n=== Loading weights ===\n");
    if (vulkan_da3_load_safetensors(gpu, model_path, config_path) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        vulkan_da3_free(gpu);
        return 1;
    }

    /* Load input image */
    int img_w, img_h, img_c;
    uint8_t *img;
    if (input_path) {
        img = stbi_load(input_path, &img_w, &img_h, &img_c, 3);
        if (!img) {
            fprintf(stderr, "Failed to load %s\n", input_path);
            vulkan_da3_free(gpu);
            return 1;
        }
        fprintf(stderr, "\nInput: %s (%dx%d)\n", input_path, img_w, img_h);
    } else {
        /* Generate synthetic gradient */
        img_w = 518; img_h = 518;
        img = (uint8_t *)malloc((size_t)img_w * img_h * 3);
        for (int y = 0; y < img_h; y++)
            for (int x = 0; x < img_w; x++) {
                int idx = (y * img_w + x) * 3;
                img[idx + 0] = (uint8_t)(x * 255 / (img_w - 1));
                img[idx + 1] = (uint8_t)(y * 255 / (img_h - 1));
                img[idx + 2] = (uint8_t)((x + y) * 255 / (img_w + img_h - 2));
            }
        fprintf(stderr, "\nInput: synthetic gradient (%dx%d)\n", img_w, img_h);
    }

    /* Run inference */
    fprintf(stderr, "\n=== Running Vulkan DA3 inference (flags=0x%02x) ===\n", output_flags);
    double t0 = get_time_ms();
    da3_full_result result = vulkan_da3_predict_full(gpu, img, img_w, img_h, output_flags, NULL);
    double elapsed = get_time_ms() - t0;
    fprintf(stderr, "\nTotal inference time: %.1f ms\n", elapsed);

    /* Print results */
    int npix = result.width * result.height;
    if (result.depth && npix > 0) {
        fprintf(stderr, "Output: %dx%d\n", result.width, result.height);
        print_stats("depth", result.depth, npix);
        print_stats("confidence", result.confidence, npix);

        /* Sanity check */
        float dmin = result.depth[0], dmax = result.depth[0];
        for (int i = 1; i < npix; i++) {
            if (result.depth[i] < dmin) dmin = result.depth[i];
            if (result.depth[i] > dmax) dmax = result.depth[i];
        }
        if (dmin == dmax) fprintf(stderr, "WARNING: all depth values identical\n");
        if (dmin < 0) fprintf(stderr, "WARNING: negative depth values\n");

        /* Save outputs */
        if (output_path && result.depth) {
            img_write_depth_png(output_path, result.depth, result.width, result.height);
        }
        if (npy_path && result.depth) {
            write_npy_f32(npy_path, result.depth, result.width, result.height);
        }
    } else {
        fprintf(stderr, "No depth output produced\n");
    }

    /* Pose output */
    if (result.has_pose) {
        fprintf(stderr, "\nPose estimation:\n");
        fprintf(stderr, "  translation: [%.6f, %.6f, %.6f]\n",
                result.pose[0], result.pose[1], result.pose[2]);
        fprintf(stderr, "  quaternion:  [%.6f, %.6f, %.6f, %.6f]\n",
                result.pose[3], result.pose[4], result.pose[5], result.pose[6]);
        fprintf(stderr, "  fov:         [%.6f, %.6f]\n",
                result.pose[7], result.pose[8]);
    }

    /* Rays output */
    if (result.has_rays && result.rays) {
        fprintf(stderr, "\nRays output:\n");
        print_stats("rays", result.rays, 6 * npix);
        print_stats("ray_conf", result.ray_confidence, npix);
    }

    /* Gaussians output */
    if (result.has_gaussians && result.gaussians) {
        fprintf(stderr, "\nGaussians output:\n");
        /* Print stats for first channel as proxy (full array is gs_oc * npix) */
        print_stats("gauss[0]", result.gaussians, npix);
        if (npy_path) {
            char gs_path[512];
            snprintf(gs_path, sizeof(gs_path), "%s.gaussians.npy", npy_path);
            /* Assume 38 output channels (standard DA3 gaussian representation) */
            write_npy_f32_3d(gs_path, result.gaussians, 38, result.height, result.width);
        }
    }

    /* Cleanup */
    if (input_path) stbi_image_free(img); else free(img);
    da3_full_result_free(&result);
    vulkan_da3_free(gpu);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
