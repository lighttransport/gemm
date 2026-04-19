/*
 * verify_qimg_dit.c — single-step CPU↔HIP DiT comparison for Qwen-Image.
 *
 * Loads the FP8 DiT into both the CPU reference (qimg_dit_forward) and the
 * HIP runner (hip_qimg_dit_step), runs one forward pass with deterministic
 * random img/text inputs at a small latent size, and reports max_diff /
 * mean_diff / cosine. Used to gate the FP8 LUT and BF16 WMMA dispatch
 * paths against the CPU gold.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define QIMG_DIT_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/gguf_loader.h"
#include "../../common/qwen_image_dit.h"
#include "hip_qimg_runner.h"

/* Deterministic LCG → uniform → no Box-Muller (we want bounded values that
 * fit in FP8 E4M3 cleanly so quant noise on activations is not the issue). */
static uint64_t rng_state = 12345;
static float randf_unit(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((rng_state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}

int main(int argc, char **argv) {
    const char *dit_path = NULL;
    int device_id = 0;
    int lat = 8;             /* latent side -> (lat/2)^2 img tokens */
    int n_txt = 16;
    int n_threads = 16;
    float timestep = 500.0f; /* qimg uses raw 0..1000 */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--dit") && i+1 < argc) dit_path = argv[++i];
        else if (!strcmp(argv[i], "-d") && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lat") && i+1 < argc) lat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ntxt") && i+1 < argc) n_txt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--t") && i+1 < argc) timestep = (float)atof(argv[++i]);
    }
    if (!dit_path) {
        fprintf(stderr, "Usage: %s --dit <dit.safetensors> [--lat N] [--ntxt N] [--t T] [-d dev]\n", argv[0]);
        return 1;
    }

    /* Qwen-Image DiT shape (matches hip_qimg_runner defaults) */
    int lat_h = lat, lat_w = lat;
    int n_img = (lat_h / 2) * (lat_w / 2);
    int in_ch = 64;       /* 16 latent channels × 2×2 patch */
    int txt_dim = 3584;

    fprintf(stderr, "=== Qwen-Image DiT Verification ===\n");
    fprintf(stderr, "lat=%dx%d -> n_img=%d (in_ch=%d), n_txt=%d (txt_dim=%d), t=%.1f\n",
            lat_h, lat_w, n_img, in_ch, n_txt, txt_dim, timestep);

    size_t img_n = (size_t)n_img * in_ch;
    size_t txt_n = (size_t)n_txt * txt_dim;
    float *img = (float *)malloc(img_n * sizeof(float));
    float *txt = (float *)malloc(txt_n * sizeof(float));
    float *out_cpu = (float *)calloc(img_n, sizeof(float));
    float *out_gpu = (float *)calloc(img_n, sizeof(float));
    for (size_t i = 0; i < img_n; i++) img[i] = randf_unit() * 0.5f;
    for (size_t i = 0; i < txt_n; i++) txt[i] = randf_unit() * 0.1f;

    fprintf(stderr, "\nLoading CPU DiT...\n");
    qimg_dit_model *cpu = qimg_dit_load_safetensors(dit_path);
    if (!cpu) { fprintf(stderr, "CPU load failed\n"); return 1; }

    fprintf(stderr, "CPU forward (threads=%d)...\n", n_threads);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    qimg_dit_forward(out_cpu, img, n_img, txt, n_txt, timestep, cpu, n_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "CPU DiT step: %.2f s\n", cpu_time);
    qimg_dit_free(cpu);

    fprintf(stderr, "\nInit HIP runner...\n");
    hip_qimg_runner *r = hip_qimg_init(device_id, 1);
    if (!r) return 1;
    if (hip_qimg_load_dit(r, dit_path) < 0) {
        fprintf(stderr, "GPU load failed\n");
        hip_qimg_free(r);
        return 1;
    }

    fprintf(stderr, "GPU forward...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (hip_qimg_dit_step(r, img, n_img, txt, n_txt, timestep, out_gpu) != 0) {
        fprintf(stderr, "GPU step failed\n");
        hip_qimg_free(r);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "GPU DiT step: %.2f s\n", gpu_time);
    hip_qimg_free(r);

    /* Compare */
    double max_diff = 0, sum_diff = 0, sum_abs = 0;
    double dot = 0, nc = 0, ng = 0;
    int max_idx = 0, n_large = 0;
    for (size_t i = 0; i < img_n; i++) {
        double c = out_cpu[i], g = out_gpu[i];
        double d = fabs(c - g);
        if (d > max_diff) { max_diff = d; max_idx = (int)i; }
        sum_diff += d;
        sum_abs  += fabs(c);
        dot += c * g;
        nc  += c * c;
        ng  += g * g;
        if (d > 0.01) n_large++;
    }
    double mean_diff = sum_diff / (double)img_n;
    double mean_abs  = sum_abs  / (double)img_n;
    double corr      = dot / (sqrt(nc) * sqrt(ng) + 1e-30);

    fprintf(stderr, "\n=== Comparison ===\n");
    fprintf(stderr, "Elements:    %zu\n", img_n);
    fprintf(stderr, "Mean |CPU|:  %.6f\n", mean_abs);
    fprintf(stderr, "Mean |diff|: %.6f\n", mean_diff);
    fprintf(stderr, "Max |diff|:  %.6f @ %d (CPU=%.6f GPU=%.6f)\n",
            max_diff, max_idx, out_cpu[max_idx], out_gpu[max_idx]);
    fprintf(stderr, "Diffs>0.01:  %d / %zu (%.2f%%)\n",
            n_large, img_n, 100.0 * n_large / (double)img_n);
    fprintf(stderr, "Correlation: %.6f\n", corr);
    fprintf(stderr, "Relative:    %.4f%%\n",
            (mean_abs > 0) ? 100.0 * mean_diff / mean_abs : 0.0);

    fprintf(stderr, "\nFirst 8 values:\n  CPU: ");
    for (int i = 0; i < 8 && i < (int)img_n; i++) fprintf(stderr, "%9.5f ", out_cpu[i]);
    fprintf(stderr, "\n  GPU: ");
    for (int i = 0; i < 8 && i < (int)img_n; i++) fprintf(stderr, "%9.5f ", out_gpu[i]);
    fprintf(stderr, "\n");

    /* Looser threshold than F32 path: BF16 WMMA introduces ~5% per-element
     * residuals that average out to ~1% mean diff. */
    int pass = (max_diff < 1e-3 && corr > 0.999);
    int wmma_pass = (corr > 0.999);
    fprintf(stderr, "\n%s   (strict: max_diff<1e-3 && corr>0.999)\n",
            pass ? "PASS" : "FAIL");
    if (!pass)
        fprintf(stderr, "%s   (relaxed for BF16 WMMA: corr>0.999)\n",
                wmma_pass ? "PASS" : "FAIL");

    free(img); free(txt); free(out_cpu); free(out_gpu);
    return pass ? 0 : (wmma_pass ? 0 : 1);
}
