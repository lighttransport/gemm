/*
 * verify_decoder.c - Compare HIP decoder vs CPU reference
 *
 * Loads TRELLIS.2 Stage 1 decoder weights on both CPU (trellis2_ss_decoder.h)
 * and GPU (hip_trellis2_runner), runs forward on a deterministic latent,
 * reports diffs.
 *
 * Pass threshold: max_diff < 1e-3 && corr > 0.9999
 *
 * Build: see Makefile target 'verify_decoder'
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define T2_SS_DEC_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/trellis2_ss_decoder.h"
#include "hip_trellis2_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* Deterministic PRNG */
static uint64_t rng_state = 0xABCDEF9876543210ULL;
static float randn_lcg(void) {
    float u1, u2;
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    u1 = (float)((rng_state >> 11) + 1) / (float)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    u2 = (float)(rng_state >> 11) / (float)(1ULL << 53);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

static void print_stats(const char *label, const float *a, int n) {
    float mn = a[0], mx = a[0]; double s = 0;
    for (int i = 0; i < n; i++) { if (a[i]<mn) mn=a[i]; if (a[i]>mx) mx=a[i]; s+=a[i]; }
    fprintf(stderr, "  %s: min=%.4f max=%.4f mean=%.6f first=[%.4f %.4f %.4f %.4f]\n",
            label, mn, mx, s/n, a[0], a[1], a[2], a[3]);
}

int main(int argc, char **argv) {
    const char *dec_path = NULL;
    int device_id = 0;
    int n_threads  = 8;
    int use_zeros  = 0;   /* --zeros: use all-zero latent instead of random */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--decoder") && i+1 < argc) dec_path  = argv[++i];
        else if (!strcmp(argv[i], "-d")   && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--zeros"))  use_zeros = 1;
    }

    if (!dec_path) {
        fprintf(stderr, "Usage: %s --decoder <decoder.safetensors> [--zeros] [--threads N] [-d dev]\n",
                argv[0]);
        return 1;
    }

    /* latent: [8, 16, 16, 16] = 32768 elements */
    const int LATENT_N = 8 * 16 * 16 * 16;
    /* occupancy: [64, 64, 64] = 262144 elements */
    const int OCC_N    = 64 * 64 * 64;

    fprintf(stderr, "=== TRELLIS.2 Decoder Verification ===\n");
    fprintf(stderr, "latent=[8,16,16,16]  output=[64,64,64]  use_zeros=%d\n", use_zeros);

    /* --- Generate deterministic latent --- */
    float *latent   = (float *)malloc(LATENT_N * sizeof(float));
    float *out_cpu  = (float *)calloc(OCC_N, sizeof(float));
    float *out_gpu  = (float *)calloc(OCC_N, sizeof(float));

    if (use_zeros) {
        memset(latent, 0, LATENT_N * sizeof(float));
    } else {
        for (int i = 0; i < LATENT_N; i++) latent[i] = randn_lcg() * 0.5f;
    }

    /* --- CPU reference --- */
    fprintf(stderr, "\nLoading CPU decoder (threads=%d)...\n", n_threads);
    t2_ss_dec *cpu_d = t2_ss_dec_load(dec_path);
    if (!cpu_d) { fprintf(stderr, "CPU decoder load failed\n"); return 1; }

    fprintf(stderr, "CPU forward...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    float *cpu_result = t2_ss_dec_forward(cpu_d, latent, n_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) * 1000.0;
    fprintf(stderr, "CPU decoder: %.1f ms\n", cpu_ms);

    if (!cpu_result) { fprintf(stderr, "CPU forward returned NULL\n"); return 1; }
    memcpy(out_cpu, cpu_result, OCC_N * sizeof(float));
    free(cpu_result);
    print_stats("CPU out", out_cpu, OCC_N);
    t2_ss_dec_free(cpu_d);

    /* --- GPU (HIP) --- */
    fprintf(stderr, "\nInit HIP runner (device=%d)...\n", device_id);
    hip_trellis2_runner *r = hip_trellis2_init(device_id, 1);
    if (!r) { fprintf(stderr, "HIP init failed\n"); return 1; }

    fprintf(stderr, "Loading decoder weights on GPU...\n");
    if (hip_trellis2_load_decoder(r, dec_path) != 0) {
        fprintf(stderr, "GPU decoder load failed\n");
        hip_trellis2_free(r); return 1;
    }

    fprintf(stderr, "GPU forward...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (hip_trellis2_decode(r, latent, out_gpu) != 0) {
        fprintf(stderr, "GPU forward failed\n");
        hip_trellis2_free(r); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_ms = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) * 1000.0;
    fprintf(stderr, "GPU decoder: %.1f ms\n", gpu_ms);
    print_stats("GPU out", out_gpu, OCC_N);

    hip_trellis2_free(r);

    /* --- Compare --- */
    double max_diff = 0, sum_diff = 0;
    double dot = 0, nc = 0, ng = 0;
    int max_idx = 0, n_large = 0;

    for (int i = 0; i < OCC_N; i++) {
        double c = out_cpu[i], g = out_gpu[i];
        double d = fabs(c - g);
        if (d > max_diff) { max_diff = d; max_idx = i; }
        sum_diff += d;
        dot += c * g;
        nc += c * c;
        ng += g * g;
        if (d > 1e-3) n_large++;
    }
    double mean_diff = sum_diff / OCC_N;
    double corr = dot / (sqrt(nc) * sqrt(ng) + 1e-30);

    /* Threshold fraction */
    float thresh = 0.0f;
    int   n_above = 0;
    for (int i = 0; i < OCC_N; i++) if (out_cpu[i] > thresh) n_above++;
    fprintf(stderr, "\nOccupancy stats:\n");
    fprintf(stderr, "  CPU above %.1f: %d / %d (%.2f%%)\n",
            thresh, n_above, OCC_N, 100.0 * n_above / OCC_N);
    n_above = 0;
    for (int i = 0; i < OCC_N; i++) if (out_gpu[i] > thresh) n_above++;
    fprintf(stderr, "  GPU above %.1f: %d / %d (%.2f%%)\n",
            thresh, n_above, OCC_N, 100.0 * n_above / OCC_N);

    fprintf(stderr, "\n=== Comparison ===\n");
    fprintf(stderr, "Elements:    %d\n", OCC_N);
    fprintf(stderr, "Mean |diff|: %.6e\n", mean_diff);
    fprintf(stderr, "Max |diff|:  %.6e @ [%d] (CPU=%.6f GPU=%.6f)\n",
            max_diff, max_idx, out_cpu[max_idx], out_gpu[max_idx]);
    fprintf(stderr, "Diffs>1e-3:  %d / %d (%.2f%%)\n",
            n_large, OCC_N, 100.0 * n_large / OCC_N);
    fprintf(stderr, "Correlation: %.8f\n", corr);
    fprintf(stderr, "Speedup:     %.1fx (CPU %.1f ms / GPU %.1f ms)\n",
            cpu_ms / gpu_ms, cpu_ms, gpu_ms);

    fprintf(stderr, "\nFirst 8 values:\n  CPU: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%9.5f ", out_cpu[i]);
    fprintf(stderr, "\n  GPU: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%9.5f ", out_gpu[i]);
    fprintf(stderr, "\n");

    int pass = (max_diff < 1e-3 && corr > 0.9999);
    fprintf(stderr, "\n%s (threshold: max_diff<1e-3 && corr>0.9999)\n",
            pass ? "PASS" : "FAIL");

    free(latent); free(out_cpu); free(out_gpu);
    return pass ? 0 : 1;
}
