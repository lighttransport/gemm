/*
 * compare_cpu_gpu.c - Compare HIP DA3 GPU output vs CPU reference
 *
 * Usage: ./compare_cpu_gpu <model.safetensors> [-i image.jpg]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* stb_image for JPEG/PNG loading */
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

/* GGUF loader */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

/* Dequant */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

/* Safetensors — must be included BEFORE depth_anything3.h for da3_load_safetensors */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

/* CPU compute */
#define CPU_COMPUTE_IMPLEMENTATION
#include "../../common/cpu_compute.h"

/* CPU DA3 reference */
#define DEPTH_ANYTHING3_IMPLEMENTATION
#include "../../common/depth_anything3.h"

/* HIP DA3 runner — only include header, not the runner .c itself */
#include "hip_da3_runner.h"

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void print_stats(const char *label, const float *data, int n) {
    if (!data || n == 0) { fprintf(stderr, "  %-12s (null)\n", label); return; }
    float mn = data[0], mx = data[0], sum = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
    }
    fprintf(stderr, "  %-12s min=%.6f  max=%.6f  mean=%.6f\n", label, mn, mx, sum / n);
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static float rel_l2_error(const float *a, const float *b, int n) {
    float diff_sq = 0, ref_sq = 0;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        diff_sq += d * d;
        ref_sq += b[i] * b[i];
    }
    if (ref_sq < 1e-12f) return sqrtf(diff_sq);
    return sqrtf(diff_sq / ref_sq);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.safetensors> [-i image.jpg]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *image_path = "../../common/fujisan.jpg";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) image_path = argv[++i];
    }

    /* Config path: same directory as model */
    char config_path[512];
    const char *slash = strrchr(model_path, '/');
    if (slash) {
        int dir_len = (int)(slash - model_path);
        snprintf(config_path, sizeof(config_path), "%.*s/config.json", dir_len, model_path);
    } else {
        snprintf(config_path, sizeof(config_path), "config.json");
    }

    /* Load image */
    fprintf(stderr, "Loading image: %s\n", image_path);
    int img_w, img_h, img_c;
    uint8_t *img = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!img) { fprintf(stderr, "Failed to load image\n"); return 1; }
    fprintf(stderr, "Image: %dx%d\n\n", img_w, img_h);

    /* ---- CPU reference ---- */
    fprintf(stderr, "=== CPU Reference ===\n");
    double t0 = get_time_ms();
    da3_model *cpu = da3_load_safetensors(model_path, config_path);
    double t1 = get_time_ms();
    if (!cpu) { fprintf(stderr, "CPU model load failed\n"); stbi_image_free(img); return 1; }
    fprintf(stderr, "CPU load: %.1f ms\n", t1 - t0);

    t0 = get_time_ms();
    da3_result cpu_res = da3_predict(cpu, img, img_w, img_h, 4);
    t1 = get_time_ms();
    fprintf(stderr, "CPU inference: %.1f ms\n", t1 - t0);
    int npix = cpu_res.width * cpu_res.height;
    print_stats("cpu_depth", cpu_res.depth, npix);
    print_stats("cpu_conf", cpu_res.confidence, npix);

    /* ---- GPU (HIP) ---- */
    fprintf(stderr, "\n=== GPU (HIP/RDNA4) ===\n");
    t0 = get_time_ms();
    hip_da3_runner *gpu = hip_da3_init(0, 1);
    if (!gpu) { fprintf(stderr, "GPU init failed\n"); return 1; }
    if (hip_da3_load_safetensors(gpu, model_path, config_path) != 0) {
        fprintf(stderr, "GPU load failed\n"); return 1;
    }
    t1 = get_time_ms();
    fprintf(stderr, "GPU load: %.1f ms\n", t1 - t0);

    t0 = get_time_ms();
    da3_full_result gpu_res = hip_da3_predict_full(gpu, img, img_w, img_h, DA3_OUTPUT_DEPTH, NULL);
    t1 = get_time_ms();
    fprintf(stderr, "GPU inference: %.1f ms\n", t1 - t0);
    print_stats("gpu_depth", gpu_res.depth, npix);
    print_stats("gpu_conf", gpu_res.confidence, npix);

    /* ---- Compare ---- */
    fprintf(stderr, "\n=== Comparison ===\n");
    if (!cpu_res.depth || !gpu_res.depth) {
        fprintf(stderr, "ERROR: one of the results is NULL\n");
        return 1;
    }

    if (cpu_res.width != gpu_res.width || cpu_res.height != gpu_res.height) {
        fprintf(stderr, "ERROR: size mismatch: CPU=%dx%d GPU=%dx%d\n",
                cpu_res.width, cpu_res.height, gpu_res.width, gpu_res.height);
        return 1;
    }

    float depth_max_diff = max_abs_diff(gpu_res.depth, cpu_res.depth, npix);
    float depth_rel_l2 = rel_l2_error(gpu_res.depth, cpu_res.depth, npix);
    fprintf(stderr, "Depth:      max_abs_diff=%.6f  rel_L2=%.6f\n", depth_max_diff, depth_rel_l2);

    if (cpu_res.confidence && gpu_res.confidence) {
        float conf_max_diff = max_abs_diff(gpu_res.confidence, cpu_res.confidence, npix);
        float conf_rel_l2 = rel_l2_error(gpu_res.confidence, cpu_res.confidence, npix);
        fprintf(stderr, "Confidence: max_abs_diff=%.6f  rel_L2=%.6f\n", conf_max_diff, conf_rel_l2);
    }

    /* Print worst-case pixels */
    {
        int worst_i = 0;
        float worst_d = 0;
        for (int i = 0; i < npix; i++) {
            float d = fabsf(gpu_res.depth[i] - cpu_res.depth[i]);
            if (d > worst_d) { worst_d = d; worst_i = i; }
        }
        int wy = worst_i / gpu_res.width, wx = worst_i % gpu_res.width;
        fprintf(stderr, "\nWorst pixel: (%d,%d) cpu=%.6f gpu=%.6f diff=%.6f\n",
                wx, wy, cpu_res.depth[worst_i], gpu_res.depth[worst_i], worst_d);
    }

    /* Sample comparison at corners + center */
    fprintf(stderr, "\nSample comparison (cpu vs gpu):\n");
    int cx = gpu_res.width / 2, cy = gpu_res.height / 2;
    int samples[] = { 0, gpu_res.width - 1,
                      cy * gpu_res.width + cx,
                      (gpu_res.height - 1) * gpu_res.width,
                      (gpu_res.height - 1) * gpu_res.width + gpu_res.width - 1 };
    const char *names[] = { "TL", "TR", "Center", "BL", "BR" };
    for (int s = 0; s < 5; s++) {
        int i = samples[s];
        fprintf(stderr, "  %-8s cpu=%.6f  gpu=%.6f  diff=%.6f\n",
                names[s], cpu_res.depth[i], gpu_res.depth[i],
                fabsf(cpu_res.depth[i] - gpu_res.depth[i]));
    }

    /* Pass/fail */
    int pass = (depth_rel_l2 < 0.01f);  /* 1% relative L2 */
    fprintf(stderr, "\nResult: %s (rel_L2=%.6f, threshold=0.01)\n",
            pass ? "PASS" : "FAIL", depth_rel_l2);

    /* Cleanup */
    da3_result_free(&cpu_res);
    da3_full_result_free(&gpu_res);
    da3_free(cpu);
    hip_da3_free(gpu);
    stbi_image_free(img);

    return pass ? 0 : 1;
}
