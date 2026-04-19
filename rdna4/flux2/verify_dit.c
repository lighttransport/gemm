/*
 * verify_dit.c - Compare GPU DiT single-step output against CPU reference
 *
 * Loads DiT weights on both CPU and HIP GPU, runs one forward step with
 * deterministic random img/txt tokens at a small size, reports diffs.
 *
 * Build: see Makefile target 'verify_dit'.
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/flux2_klein_dit.h"
#include "hip_flux2_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Deterministic PRNG */
static uint64_t rng_state = 12345;
static float randf_unit(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((rng_state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}

int main(int argc, char **argv) {
    const char *dit_path = NULL;
    int device_id = 0;
    int lat = 8;           /* latent side -> lat*lat/4 tokens after 2x2 patch */
    int n_txt = 16;
    int n_threads = 8;
    float timestep = 0.5f; /* sigma in [0,1] */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dit") && i+1 < argc) dit_path = argv[++i];
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

    /* Derive token counts */
    int lat_h = lat, lat_w = lat;
    int n_img = (lat_h / 2) * (lat_w / 2);
    int patch_in_ch = 32 * 2 * 2;   /* 128 */
    int txt_dim   = 7680;

    fprintf(stderr, "=== DiT Verification ===\n");
    fprintf(stderr, "lat=%dx%d -> n_img=%d (pin=%d), n_txt=%d (txt_dim=%d), t=%.3f\n",
            lat_h, lat_w, n_img, patch_in_ch, n_txt, txt_dim, timestep);

    /* Generate deterministic inputs */
    size_t img_n = (size_t)n_img * patch_in_ch;
    size_t txt_n = (size_t)n_txt * txt_dim;
    float *img = (float *)malloc(img_n * sizeof(float));
    float *txt = (float *)malloc(txt_n * sizeof(float));
    float *out_cpu = (float *)calloc(img_n, sizeof(float));
    float *out_gpu = (float *)calloc(img_n, sizeof(float));
    for (size_t i = 0; i < img_n; i++) img[i] = randf_unit() * 0.5f;
    for (size_t i = 0; i < txt_n; i++) txt[i] = randf_unit() * 0.1f;

    /* ---- CPU reference ---- */
    fprintf(stderr, "\nLoading CPU DiT...\n");
    flux2_dit_model *cpu = flux2_dit_load_safetensors(dit_path);
    if (!cpu) { fprintf(stderr, "CPU load failed\n"); return 1; }

    fprintf(stderr, "CPU forward (threads=%d)...\n", n_threads);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    flux2_dit_forward(out_cpu, img, n_img, txt, n_txt, timestep, cpu, n_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "CPU DiT step: %.2f s\n", cpu_time);
    flux2_dit_free(cpu);

    /* ---- HIP GPU ---- */
    fprintf(stderr, "\nInit HIP runner...\n");
    hip_flux2_runner *r = hip_flux2_init(device_id, 1);
    if (!r) return 1;
    if (hip_flux2_load_dit(r, dit_path) < 0) {
        fprintf(stderr, "GPU load failed\n"); hip_flux2_free(r); return 1;
    }

    fprintf(stderr, "GPU forward...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (hip_flux2_dit_step(r, img, n_img, txt, n_txt, timestep, 0.0f, out_gpu) != 0) {
        fprintf(stderr, "GPU step failed\n"); hip_flux2_free(r); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "GPU DiT step: %.2f s\n", gpu_time);

    hip_flux2_free(r);

    /* ---- Compare ---- */
    double max_diff = 0, sum_diff = 0, sum_abs = 0;
    double dot = 0, nc = 0, ng = 0;
    int max_idx = 0;
    int n_large = 0;
    for (size_t i = 0; i < img_n; i++) {
        double c = out_cpu[i], g = out_gpu[i];
        double d = fabs(c - g);
        if (d > max_diff) { max_diff = d; max_idx = (int)i; }
        sum_diff += d;
        sum_abs += fabs(c);
        dot += c * g;
        nc  += c * c;
        ng  += g * g;
        if (d > 0.01) n_large++;
    }
    double mean_diff = sum_diff / (double)img_n;
    double mean_abs  = sum_abs  / (double)img_n;
    double corr      = dot / (sqrt(nc) * sqrt(ng) + 1e-30);

    fprintf(stderr, "\n=== Comparison ===\n");
    fprintf(stderr, "Elements:   %zu\n", img_n);
    fprintf(stderr, "Mean |CPU|: %.6f\n", mean_abs);
    fprintf(stderr, "Mean |diff|:%.6f\n", mean_diff);
    fprintf(stderr, "Max |diff|: %.6f @ %d (CPU=%.6f GPU=%.6f)\n",
            max_diff, max_idx, out_cpu[max_idx], out_gpu[max_idx]);
    fprintf(stderr, "Diffs>0.01: %d / %zu (%.2f%%)\n",
            n_large, img_n, 100.0 * n_large / (double)img_n);
    fprintf(stderr, "Correlation:%.6f\n", corr);
    fprintf(stderr, "Relative :  %.4f%%\n",
            (mean_abs > 0) ? 100.0 * mean_diff / mean_abs : 0.0);

    fprintf(stderr, "\nFirst 8 values:\n");
    fprintf(stderr, "  CPU: ");
    for (int i = 0; i < 8 && i < (int)img_n; i++) fprintf(stderr, "%9.5f ", out_cpu[i]);
    fprintf(stderr, "\n  GPU: ");
    for (int i = 0; i < 8 && i < (int)img_n; i++) fprintf(stderr, "%9.5f ", out_gpu[i]);
    fprintf(stderr, "\n");

    int pass = (max_diff < 1e-3 && corr > 0.999);
    fprintf(stderr, "\n%s (max_diff<1e-3 && corr>0.999)\n", pass ? "PASS" : "FAIL");

    free(img); free(txt); free(out_cpu); free(out_gpu);
    return pass ? 0 : 1;
}
