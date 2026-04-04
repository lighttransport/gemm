/*
 * verify_vae.c - Compare GPU VAE decode output against CPU reference
 *
 * Loads VAE weights, generates deterministic latent input,
 * runs both CPU and GPU decode, compares element-wise.
 *
 * Build: gcc -O2 -I.. -o verify_vae verify_vae.c hip_flux2_runner.c ../rocew.c -ldl -lm -lpthread
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/qwen_image_scheduler.h"
#include "hip_flux2_runner.h"

/* CPU VAE functions come from hip_flux2_runner.c (FLUX2_VAE_IMPLEMENTATION) */
#include "../../common/flux2_klein_vae.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Deterministic PRNG */
static uint64_t rng_state = 12345;
static float randf(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((rng_state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}

int main(int argc, char **argv) {
    const char *dit_path = NULL;
    const char *vae_path = NULL;
    int device_id = 0;
    int lat_h = 8, lat_w = 8;  /* small for fast test: 8x8 latent = 64x64 output */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dit") && i+1 < argc) dit_path = argv[++i];
        else if (!strcmp(argv[i], "--vae") && i+1 < argc) vae_path = argv[++i];
        else if (!strcmp(argv[i], "-d") && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lat") && i+1 < argc) { lat_h = lat_w = atoi(argv[++i]); }
    }

    if (!dit_path || !vae_path) {
        fprintf(stderr, "Usage: %s --dit <dit.safetensors> --vae <vae.safetensors> [--lat N] [-d dev]\n", argv[0]);
        return 1;
    }

    /* Init GPU runner (need DiT loaded for kernel compilation) */
    hip_flux2_runner *r = hip_flux2_init(device_id, 1);
    if (!r) return 1;
    if (hip_flux2_load_dit(r, dit_path) < 0) return 1;
    if (hip_flux2_load_vae(r, vae_path) < 0) return 1;

    /* Also load VAE on CPU separately for reference */
    flux2_vae_model *vae_cpu = flux2_vae_load(vae_path);
    if (!vae_cpu) { fprintf(stderr, "CPU VAE load failed\n"); return 1; }
    vae_cpu->n_threads = 4;

    int lc = 32;
    int out_h = lat_h * 8, out_w = lat_w * 8;
    size_t lat_sz = (size_t)lc * lat_h * lat_w;
    size_t rgb_sz = (size_t)3 * out_h * out_w;

    fprintf(stderr, "\n=== VAE Verification ===\n");
    fprintf(stderr, "Latent: [%d, %d, %d] -> RGB: [3, %d, %d]\n", lc, lat_h, lat_w, out_h, out_w);

    /* Generate deterministic latent */
    float *latent = (float *)malloc(lat_sz * sizeof(float));
    for (size_t i = 0; i < lat_sz; i++) latent[i] = randf() * 0.5f;

    /* CPU decode */
    float *rgb_cpu = (float *)calloc(rgb_sz, sizeof(float));
    fprintf(stderr, "Running CPU VAE decode...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    flux2_vae_decode(rgb_cpu, latent, lat_h, lat_w, vae_cpu);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "CPU VAE: %.3f s\n", cpu_time);

    /* GPU decode */
    float *rgb_gpu = (float *)calloc(rgb_sz, sizeof(float));
    fprintf(stderr, "Running GPU VAE decode...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    hip_flux2_vae_decode(r, latent, lat_h, lat_w, rgb_gpu);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "GPU VAE: %.3f s\n", gpu_time);

    /* Compare */
    double max_diff = 0, sum_diff = 0, sum_abs = 0;
    int max_idx = 0;
    int n_large = 0;  /* diffs > 0.01 */
    for (size_t i = 0; i < rgb_sz; i++) {
        double d = fabs((double)rgb_gpu[i] - (double)rgb_cpu[i]);
        if (d > max_diff) { max_diff = d; max_idx = (int)i; }
        sum_diff += d;
        sum_abs += fabs((double)rgb_cpu[i]);
        if (d > 0.01) n_large++;
    }
    double mean_diff = sum_diff / (double)rgb_sz;
    double mean_abs = sum_abs / (double)rgb_sz;

    int mc = max_idx / (out_h * out_w);
    int mr = (max_idx % (out_h * out_w)) / out_w;
    int mx = max_idx % out_w;

    fprintf(stderr, "\n=== Comparison ===\n");
    fprintf(stderr, "Total pixels: %zu (3 x %d x %d)\n", rgb_sz, out_h, out_w);
    fprintf(stderr, "Mean |CPU|:   %.6f\n", mean_abs);
    fprintf(stderr, "Mean |diff|:  %.6f\n", mean_diff);
    fprintf(stderr, "Max |diff|:   %.6f at [c=%d, y=%d, x=%d] (CPU=%.6f, GPU=%.6f)\n",
            max_diff, mc, mr, mx, rgb_cpu[max_idx], rgb_gpu[max_idx]);
    fprintf(stderr, "Diffs > 0.01: %d / %zu (%.2f%%)\n", n_large, rgb_sz, 100.0*n_large/rgb_sz);
    fprintf(stderr, "Relative err: %.4f%%\n", (mean_abs > 0) ? 100.0 * mean_diff / mean_abs : 0.0);

    /* Sample values */
    fprintf(stderr, "\nSample values (first 8 of channel 0):\n");
    fprintf(stderr, "  CPU: ");
    for (int i = 0; i < 8 && i < (int)rgb_sz; i++) fprintf(stderr, "%8.4f ", rgb_cpu[i]);
    fprintf(stderr, "\n  GPU: ");
    for (int i = 0; i < 8 && i < (int)rgb_sz; i++) fprintf(stderr, "%8.4f ", rgb_gpu[i]);
    fprintf(stderr, "\n");

    int pass = (max_diff < 0.05 && mean_diff < 0.001);
    fprintf(stderr, "\n%s (max_diff < 0.05 && mean_diff < 0.001)\n",
            pass ? "PASS" : "FAIL");

    free(latent); free(rgb_cpu); free(rgb_gpu);
    flux2_vae_free(vae_cpu);
    hip_flux2_free(r);
    return pass ? 0 : 1;
}
