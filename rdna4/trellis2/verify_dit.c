/*
 * verify_dit.c - Compare HIP DiT single-step vs CPU reference
 *
 * Loads TRELLIS.2 Stage 1 DiT weights on both CPU (trellis2_dit.h) and GPU
 * (hip_trellis2_runner), runs one forward step with deterministic random
 * noise and features, reports diffs.
 *
 * Pass threshold: max_diff < 1e-3 && corr > 0.9999
 *
 * Build: see Makefile target 'verify_dit'
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define T2DIT_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/trellis2_dit.h"
#include "hip_trellis2_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* Deterministic PRNG (LCG) */
static uint64_t rng_state = 12345;
static float randf_unit(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((rng_state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}

/* Box-Muller for Gaussian noise */
static float randn_bm(void) {
    float u1, u2;
    do { u1 = (float)((rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL, rng_state >> 11)) / (float)(1ULL << 53); } while (u1 <= 0.0f);
    u2 = (float)((rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL, rng_state >> 11)) / (float)(1ULL << 53);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

static void print_stats(const char *label, const float *a, int n) {
    float mn = a[0], mx = a[0]; double s = 0;
    for (int i = 0; i < n; i++) { if (a[i]<mn) mn=a[i]; if (a[i]>mx) mx=a[i]; s+=a[i]; }
    fprintf(stderr, "  %s: min=%.4f max=%.4f mean=%.6f first=[%.4f %.4f %.4f %.4f]\n",
            label, mn, mx, s/n, a[0], a[1], a[2], a[3]);
}

int main(int argc, char **argv) {
    const char *dit_path = NULL;
    int device_id = 0;
    int n_threads  = 8;
    float t_val    = 500.0f;  /* timestep in [0, 1000] */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dit")     && i+1 < argc) dit_path  = argv[++i];
        else if (!strcmp(argv[i], "-d")   && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--t")  && i+1 < argc) t_val     = (float)atof(argv[++i]);
    }

    if (!dit_path) {
        fprintf(stderr, "Usage: %s --dit <stage1.safetensors> [--t T] [--threads N] [-d dev]\n",
                argv[0]);
        return 1;
    }

    /* --- Shapes --- */
    /* noise_flat: [4096, 8]  (x_t layout for HIP runner) */
    /* features:  [1029, 1024] (DINOv3 output)            */
    const int N_TOK  = 4096;
    const int IN_CH  = 8;
    const int N_COND = 1029;
    const int COND_DIM = 1024;

    fprintf(stderr, "=== TRELLIS.2 DiT Verification ===\n");
    fprintf(stderr, "noise=[%d,%d]  features=[%d,%d]  t=%.1f\n",
            N_TOK, IN_CH, N_COND, COND_DIM, t_val);

    /* --- Generate deterministic inputs --- */
    rng_state = 0xDEADBEEF12345678ULL;
    float *noise    = (float *)malloc((size_t)N_TOK * IN_CH * sizeof(float));
    float *features = (float *)malloc((size_t)N_COND * COND_DIM * sizeof(float));
    float *out_cpu  = (float *)calloc((size_t)N_TOK * IN_CH, sizeof(float));
    float *out_gpu  = (float *)calloc((size_t)N_TOK * IN_CH, sizeof(float));

    for (int i = 0; i < N_TOK * IN_CH; i++)    noise[i]    = randn_bm();
    for (int i = 0; i < N_COND * COND_DIM; i++) features[i] = randf_unit() * 0.1f;

    /* --- CPU reference --- */
    fprintf(stderr, "\nLoading CPU DiT (threads=%d)...\n", n_threads);
    t2dit_model *cpu_m = t2dit_load_safetensors(dit_path);
    if (!cpu_m) { fprintf(stderr, "CPU load failed\n"); return 1; }

    fprintf(stderr, "Precomputing CPU cross-attn KV cache...\n");
    float *cond_kv = t2dit_precompute_cond_kv(features, N_COND, cpu_m, n_threads);
    if (!cond_kv) { fprintf(stderr, "KV precompute failed\n"); return 1; }

    fprintf(stderr, "CPU forward...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    t2dit_forward(out_cpu, noise, t_val, cond_kv, cpu_m, n_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) * 1000.0;
    fprintf(stderr, "CPU DiT: %.1f ms\n", cpu_ms);
    print_stats("CPU out", out_cpu, N_TOK * IN_CH);

    free(cond_kv);
    t2dit_free(cpu_m);

    /* --- GPU (HIP) --- */
    fprintf(stderr, "\nInit HIP runner (device=%d)...\n", device_id);
    hip_trellis2_runner *r = hip_trellis2_init(device_id, 1);
    if (!r) { fprintf(stderr, "HIP init failed\n"); return 1; }

    fprintf(stderr, "Loading DiT weights on GPU...\n");
    if (hip_trellis2_load_dit(r, dit_path) != 0) {
        fprintf(stderr, "GPU DiT load failed\n");
        hip_trellis2_free(r); return 1;
    }

    fprintf(stderr, "GPU forward...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (hip_trellis2_dit_step(r, noise, features, t_val, out_gpu) != 0) {
        fprintf(stderr, "GPU forward failed\n");
        hip_trellis2_free(r); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_ms = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) * 1000.0;
    fprintf(stderr, "GPU DiT: %.1f ms\n", gpu_ms);
    print_stats("GPU out", out_gpu, N_TOK * IN_CH);

    hip_trellis2_free(r);

    /* --- Compare --- */
    int n = N_TOK * IN_CH;
    double max_diff = 0, sum_diff = 0;
    double dot = 0, nc = 0, ng = 0;
    int max_idx = 0, n_large = 0;

    for (int i = 0; i < n; i++) {
        double c = out_cpu[i], g = out_gpu[i];
        double d = fabs(c - g);
        if (d > max_diff) { max_diff = d; max_idx = i; }
        sum_diff += d;
        dot += c * g;
        nc += c * c;
        ng += g * g;
        if (d > 1e-3) n_large++;
    }
    double mean_diff = sum_diff / n;
    double corr = dot / (sqrt(nc) * sqrt(ng) + 1e-30);

    fprintf(stderr, "\n=== Comparison ===\n");
    fprintf(stderr, "Elements:    %d\n", n);
    fprintf(stderr, "Mean |diff|: %.6e\n", mean_diff);
    fprintf(stderr, "Max |diff|:  %.6e @ [%d] (CPU=%.6f GPU=%.6f)\n",
            max_diff, max_idx, out_cpu[max_idx], out_gpu[max_idx]);
    fprintf(stderr, "Diffs>1e-3:  %d / %d (%.2f%%)\n",
            n_large, n, 100.0 * n_large / n);
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

    free(noise); free(features); free(out_cpu); free(out_gpu);
    return pass ? 0 : 1;
}
