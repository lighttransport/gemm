/*
 * Benchmark for optimized two-pass fused exp2 + GEMM
 *
 * Scenario: head_dim = 128 (or larger), split into two V tiles
 *   Pass 1: exp2 + GEMM(V[:,0:64]) + store exp2 to cache
 *   Pass 2: Load cached exp2 + GEMM(V[:,64:128])
 *
 * Expected: Pass1 ~70% + Pass2 ~80% = Combined ~75%
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Two-pass kernels */
extern void exp2_gemm_pass1(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o);

extern void gemm_pass2(
    const float* P, const float* V, float* O,
    int Nc, int ld_p, int ld_v, int ld_o);

/* Simple fused kernel with store */
extern void exp2_gemm_store(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o);

/* Baseline */
extern void exp2_rows(
    const int32_t* S, float* P, int M, int Nc,
    float scale, float max_val, int ld_s, int ld_p);

extern void gemm_fp32_4x4(
    const float* A, const float* B, float* O,
    int K, int ld_a, int ld_b, int ld_o);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static void ref_exp2_gemm(const int32_t* S, const float* V, float* O,
                          int M, int Nc, int D,
                          float scale, float max_val) {
    for (int i = 0; i < M * D; i++) O[i] = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < Nc; k++) {
            float p = exp2f((float)S[i * Nc + k] * scale - max_val);
            for (int j = 0; j < D; j++) {
                O[i * D + j] += p * V[k * D + j];
            }
        }
    }
}

static int verify(const float* ref, const float* test, int n, float* max_err) {
    *max_err = 0.0f;
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - test[i]);
        float rel = (fabsf(ref[i]) > 1e-6f) ? err / fabsf(ref[i]) : err;
        if (rel > *max_err) *max_err = rel;
        if (rel > 0.01f && err > 1e-4f) errors++;
    }
    return errors;
}

int main(int argc, char** argv) {
    int Nc = 64;
    int D = 128;  // head_dim = 128, split into two tiles
    int iterations = 1000;

    if (argc > 1) Nc = atoi(argv[1]);
    if (argc > 2) D = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);

    Nc = (Nc / 16) * 16;
    if (Nc < 16) Nc = 16;
    D = (D / 64) * 64;
    if (D < 64) D = 64;

    int M = 4;
    int D_tile = 64;  // Each tile processes 64 columns
    int n_tiles = D / D_tile;

    printf("=== Optimized Two-Pass Fused exp2 + GEMM ===\n");
    printf("M=%d, Nc=%d, D=%d (split into %d tiles of %d)\n", M, Nc, D, n_tiles, D_tile);
    printf("Iterations: %d\n\n", iterations);

    /* Allocate */
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O_ref = aligned_alloc(64, M * D * sizeof(float));
    float* O_test = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));

    srand(42);
    for (int i = 0; i < M * Nc; i++) S[i] = (rand() % 201) - 100;
    for (int i = 0; i < Nc * D; i++) V[i] = ((float)(rand() % 1000) - 500) / 500.0f;

    float scale = 1.0f / 64.0f;
    float max_val = 1.5f;

    int ld_v = D * sizeof(float);
    int ld_o = D * sizeof(float);
    int ld_p = Nc * sizeof(float);

    /* Reference */
    ref_exp2_gemm(S, V, O_ref, M, Nc, D, scale, max_val);

    double timer_freq = 100e6;
    double cpu_freq = 2e9;
    double fp32_peak = 128.0;
    long long total_fmla_ops = (long long)M * Nc * D * 2;

    volatile uint64_t start, end;
    double elapsed, cycles, gflops;
    float max_err;
    int errors;

    /*========================================================================
     * Test 1: Optimized two-pass (Pass1 + Pass2)
     *========================================================================*/
    printf("=== Optimized Two-Pass ===\n");
    memset(O_test, 0, M * D * sizeof(float));

    /* Pass 1: exp2 + GEMM for first tile + store exp2 */
    exp2_gemm_pass1(S, V, O_test, P, Nc, scale, max_val, Nc, ld_v, ld_o);

    /* Pass 2+: Load cached exp2 + GEMM for remaining tiles */
    for (int tile = 1; tile < n_tiles; tile++) {
        float* V_tile = V + tile * D_tile;
        float* O_tile = O_test + tile * D_tile;
        gemm_pass2(P, V_tile, O_tile, Nc, ld_p, ld_v, ld_o);
    }

    errors = verify(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: error=%.4f%% %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    /* Benchmark */
    for (int i = 0; i < 10; i++) {
        exp2_gemm_pass1(S, V, O_test, P, Nc, scale, max_val, Nc, ld_v, ld_o);
        for (int tile = 1; tile < n_tiles; tile++) {
            float* V_tile = V + tile * D_tile;
            float* O_tile = O_test + tile * D_tile;
            gemm_pass2(P, V_tile, O_tile, Nc, ld_p, ld_v, ld_o);
        }
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_gemm_pass1(S, V, O_test, P, Nc, scale, max_val, Nc, ld_v, ld_o);
        for (int tile = 1; tile < n_tiles; tile++) {
            float* V_tile = V + tile * D_tile;
            float* O_tile = O_test + tile * D_tile;
            gemm_pass2(P, V_tile, O_tile, Nc, ld_p, ld_v, ld_o);
        }
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)total_fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Total cycles: %.1f\n", cycles);
    printf("  GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    /* Measure Pass1 and Pass2 separately */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_gemm_pass1(S, V, O_test, P, Nc, scale, max_val, Nc, ld_v, ld_o);
    }
    end = get_cycles();
    double pass1_cycles = (end - start) * (cpu_freq / timer_freq) / iterations;
    long long pass1_ops = (long long)M * Nc * D_tile * 2;
    double pass1_gflops = (double)pass1_ops * iterations / ((end - start) / timer_freq) / 1e9;
    printf("  Pass1 (exp2+GEMM+store): %.1f cycles, %.2f GFLOPS (%.1f%%)\n",
           pass1_cycles, pass1_gflops, pass1_gflops / fp32_peak * 100);

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        gemm_pass2(P, V + D_tile, O_test + D_tile, Nc, ld_p, ld_v, ld_o);
    }
    end = get_cycles();
    double pass2_cycles = (end - start) * (cpu_freq / timer_freq) / iterations;
    double pass2_gflops = (double)pass1_ops * iterations / ((end - start) / timer_freq) / 1e9;
    printf("  Pass2 (load+GEMM): %.1f cycles, %.2f GFLOPS (%.1f%%)\n",
           pass2_cycles, pass2_gflops, pass2_gflops / fp32_peak * 100);

    /*========================================================================
     * Test 1b: Simple fused (per-K loop with store)
     *========================================================================*/
    printf("\n=== Simple Fused (per-K with store) ===\n");
    memset(O_test, 0, M * D * sizeof(float));

    /* Pass 1: exp2 + GEMM + store */
    exp2_gemm_store(S, V, O_test, P, Nc, scale, max_val, Nc, ld_v, ld_o);

    /* Pass 2+: Load cached exp2 + GEMM */
    for (int tile = 1; tile < n_tiles; tile++) {
        float* V_tile = V + tile * D_tile;
        float* O_tile = O_test + tile * D_tile;
        gemm_pass2(P, V_tile, O_tile, Nc, ld_p, ld_v, ld_o);
    }

    errors = verify(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: error=%.4f%% %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    for (int i = 0; i < 10; i++) {
        exp2_gemm_store(S, V, O_test, P, Nc, scale, max_val, Nc, ld_v, ld_o);
        for (int tile = 1; tile < n_tiles; tile++) {
            float* V_tile = V + tile * D_tile;
            float* O_tile = O_test + tile * D_tile;
            gemm_pass2(P, V_tile, O_tile, Nc, ld_p, ld_v, ld_o);
        }
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_gemm_store(S, V, O_test, P, Nc, scale, max_val, Nc, ld_v, ld_o);
        for (int tile = 1; tile < n_tiles; tile++) {
            float* V_tile = V + tile * D_tile;
            float* O_tile = O_test + tile * D_tile;
            gemm_pass2(P, V_tile, O_tile, Nc, ld_p, ld_v, ld_o);
        }
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)total_fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Total cycles: %.1f\n", cycles);
    printf("  GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    /* Measure Pass1 separately for simple version */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_gemm_store(S, V, O_test, P, Nc, scale, max_val, Nc, ld_v, ld_o);
    }
    end = get_cycles();
    double simple_pass1_cycles = (end - start) * (cpu_freq / timer_freq) / iterations;
    double simple_pass1_gflops = (double)pass1_ops * iterations / ((end - start) / timer_freq) / 1e9;
    printf("  Pass1 (simple): %.1f cycles, %.2f GFLOPS (%.1f%%)\n",
           simple_pass1_cycles, simple_pass1_gflops, simple_pass1_gflops / fp32_peak * 100);

    /*========================================================================
     * Test 2: Baseline separate exp2 + GEMM
     *========================================================================*/
    printf("\n=== Baseline (separate exp2 + GEMM) ===\n");
    memset(O_test, 0, M * D * sizeof(float));

    exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
    for (int tile = 0; tile < n_tiles; tile++) {
        float* V_tile = V + tile * D_tile;
        float* O_tile = O_test + tile * D_tile;
        gemm_fp32_4x4(P, V_tile, O_tile, Nc, Nc * sizeof(float), ld_v, ld_o);
    }

    errors = verify(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: error=%.4f%% %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    for (int i = 0; i < 10; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * D_tile;
            float* O_tile = O_test + tile * D_tile;
            gemm_fp32_4x4(P, V_tile, O_tile, Nc, Nc * sizeof(float), ld_v, ld_o);
        }
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * D_tile;
            float* O_tile = O_test + tile * D_tile;
            gemm_fp32_4x4(P, V_tile, O_tile, Nc, Nc * sizeof(float), ld_v, ld_o);
        }
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)total_fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Total cycles: %.1f\n", cycles);
    printf("  GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    /*========================================================================
     * Test 3: Pure GEMM (no exp2)
     *========================================================================*/
    printf("\n=== Pure GEMM (no exp2) ===\n");
    for (int i = 0; i < M * Nc; i++) P[i] = 1.0f;

    for (int i = 0; i < 10; i++) {
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * D_tile;
            float* O_tile = O_test + tile * D_tile;
            gemm_fp32_4x4(P, V_tile, O_tile, Nc, Nc * sizeof(float), ld_v, ld_o);
        }
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * D_tile;
            float* O_tile = O_test + tile * D_tile;
            gemm_fp32_4x4(P, V_tile, O_tile, Nc, Nc * sizeof(float), ld_v, ld_o);
        }
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    double pure_gemm_cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)total_fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Total cycles: %.1f\n", pure_gemm_cycles);
    printf("  GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    /*========================================================================
     * Analysis
     *========================================================================*/
    printf("\n=== Analysis ===\n");
    printf("Pure GEMM: %.1f cycles (reference)\n", pure_gemm_cycles);
    printf("exp2 overhead in optimized two-pass:\n");
    printf("  Pass1 extra: %.1f cycles (vs %.1f for pure GEMM/tile)\n",
           pass1_cycles - pure_gemm_cycles / n_tiles,
           pure_gemm_cycles / n_tiles);
    printf("  exp2 as %% of GEMM: %.1f%%\n",
           (pass1_cycles - pure_gemm_cycles / n_tiles) / (pure_gemm_cycles / n_tiles) * 100);

    free(S);
    free(V);
    free(O_ref);
    free(O_test);
    free(P);

    return 0;
}
