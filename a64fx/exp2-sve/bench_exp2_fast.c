/*
 * Benchmark for fast exp2 implementations
 * Target: exp2 overhead < 5% of GEMM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Fast exp2 */
extern void exp2_fast_rows(
    const int32_t* S, float* P, int M, int Nc,
    float scale, float neg_max, int ld_s, int ld_p);

/* Ultra fast exp2 (2^floor only) */
extern void exp2_ultra_rows(
    const int32_t* S, float* P, int M, int Nc,
    float scale, float neg_max, int ld_s, int ld_p);

/* Original exp2 */
extern void exp2_rows(
    const int32_t* S, float* P, int M, int Nc,
    float scale, float max_val, int ld_s, int ld_p);

/* GEMM */
extern void gemm_fp32_4x4(
    const float* A, const float* B, float* O,
    int K, int ld_a, int ld_b, int ld_o);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static void ref_exp2(const int32_t* S, float* P, int M, int Nc,
                     float scale, float max_val) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Nc; j++) {
            P[i * Nc + j] = exp2f((float)S[i * Nc + j] * scale - max_val);
        }
    }
}

static int verify(const float* ref, const float* test, int n,
                  float* max_err, float* avg_err) {
    *max_err = 0.0f;
    *avg_err = 0.0f;
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - test[i]);
        float rel = (fabsf(ref[i]) > 1e-10f) ? err / fabsf(ref[i]) : err;
        if (rel > *max_err) *max_err = rel;
        *avg_err += rel;
        if (rel > 0.10f) errors++;  // 10% tolerance for fast version
    }
    *avg_err /= n;
    return errors;
}

int main(int argc, char** argv) {
    int Nc = 64;
    int iterations = 10000;

    if (argc > 1) Nc = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    Nc = (Nc / 16) * 16;
    if (Nc < 16) Nc = 16;

    int M = 4, D = 64;

    printf("=== Fast exp2 Benchmark ===\n");
    printf("M=%d, Nc=%d\n", M, Nc);
    printf("Iterations: %d\n\n", iterations);

    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* P_ref = aligned_alloc(64, M * Nc * sizeof(float));
    float* P_orig = aligned_alloc(64, M * Nc * sizeof(float));
    float* P_fast = aligned_alloc(64, M * Nc * sizeof(float));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));

    srand(42);
    for (int i = 0; i < M * Nc; i++) S[i] = (rand() % 201) - 100;
    for (int i = 0; i < Nc * D; i++) V[i] = ((float)(rand() % 1000) - 500) / 500.0f;

    float scale = 1.0f / 64.0f;
    float max_val = 1.5f;
    float neg_max = -max_val;

    /* Reference */
    ref_exp2(S, P_ref, M, Nc, scale, max_val);

    /* Test original exp2_rows */
    printf("=== Original exp2_rows ===\n");
    exp2_rows(S, P_orig, M, Nc, scale, max_val, Nc, Nc);
    float max_err, avg_err;
    int errors = verify(P_ref, P_orig, M * Nc, &max_err, &avg_err);
    printf("  Max error: %.4f%%, Avg error: %.4f%%\n", max_err * 100, avg_err * 100);

    /* Test fast exp2 */
    printf("\n=== Fast exp2_fast_rows ===\n");
    exp2_fast_rows(S, P_fast, M, Nc, scale, neg_max, Nc, Nc);
    errors = verify(P_ref, P_fast, M * Nc, &max_err, &avg_err);
    printf("  Max error: %.4f%%, Avg error: %.4f%%\n", max_err * 100, avg_err * 100);
    printf("  (Higher error is OK for softmax since it normalizes)\n");

    /* Test ultra-fast exp2 */
    float* P_ultra = aligned_alloc(64, M * Nc * sizeof(float));
    printf("\n=== Ultra-fast exp2_ultra_rows (2^floor only) ===\n");
    exp2_ultra_rows(S, P_ultra, M, Nc, scale, neg_max, Nc, Nc);
    errors = verify(P_ref, P_ultra, M * Nc, &max_err, &avg_err);
    printf("  Max error: %.4f%%, Avg error: %.4f%%\n", max_err * 100, avg_err * 100);
    printf("  (Approximation: 2^floor(x), ignores fractional part)\n");

    /* Benchmark */
    double timer_freq = 100e6;
    double cpu_freq = 2e9;
    volatile uint64_t start, end;
    double elapsed, cycles;

    printf("\n=== Performance ===\n");

    /* Original exp2 */
    for (int i = 0; i < 10; i++)
        exp2_rows(S, P_orig, M, Nc, scale, max_val, Nc, Nc);

    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        exp2_rows(S, P_orig, M, Nc, scale, max_val, Nc, Nc);
    end = get_cycles();

    elapsed = (double)(end - start);
    double orig_cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    printf("Original exp2: %.1f cycles (%.2f cycles/element)\n",
           orig_cycles, orig_cycles / (M * Nc));

    /* Fast exp2 */
    for (int i = 0; i < 10; i++)
        exp2_fast_rows(S, P_fast, M, Nc, scale, neg_max, Nc, Nc);

    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        exp2_fast_rows(S, P_fast, M, Nc, scale, neg_max, Nc, Nc);
    end = get_cycles();

    elapsed = (double)(end - start);
    double fast_cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    printf("Fast exp2: %.1f cycles (%.2f cycles/element)\n",
           fast_cycles, fast_cycles / (M * Nc));

    /* Ultra-fast exp2 */
    for (int i = 0; i < 10; i++)
        exp2_ultra_rows(S, P_ultra, M, Nc, scale, neg_max, Nc, Nc);

    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        exp2_ultra_rows(S, P_ultra, M, Nc, scale, neg_max, Nc, Nc);
    end = get_cycles();

    elapsed = (double)(end - start);
    double ultra_cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    printf("Ultra exp2: %.1f cycles (%.2f cycles/element)\n",
           ultra_cycles, ultra_cycles / (M * Nc));

    /* Pure GEMM */
    for (int i = 0; i < M * Nc; i++) P_orig[i] = 1.0f;

    for (int i = 0; i < 10; i++)
        gemm_fp32_4x4(P_orig, V, O, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));

    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        gemm_fp32_4x4(P_orig, V, O, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    end = get_cycles();

    elapsed = (double)(end - start);
    double gemm_cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    printf("Pure GEMM: %.1f cycles\n", gemm_cycles);

    /* Analysis */
    printf("\n=== Analysis ===\n");
    printf("GEMM cycles: %.1f\n", gemm_cycles);
    printf("Original exp2 overhead: %.1f%% of GEMM\n", orig_cycles / gemm_cycles * 100);
    printf("Fast exp2 overhead: %.1f%% of GEMM\n", fast_cycles / gemm_cycles * 100);
    printf("Ultra exp2 overhead: %.1f%% of GEMM\n", ultra_cycles / gemm_cycles * 100);
    printf("\nTarget: < 5%% = %.1f cycles\n", gemm_cycles * 0.05);
    printf("Speedup fast: %.2fx, ultra: %.2fx\n",
           orig_cycles / fast_cycles, orig_cycles / ultra_cycles);

    /* End-to-end comparison */
    printf("\n=== End-to-End (exp2 + GEMM) ===\n");

    /* Original two-pass */
    for (int i = 0; i < 10; i++) {
        exp2_rows(S, P_orig, M, Nc, scale, max_val, Nc, Nc);
        gemm_fp32_4x4(P_orig, V, O, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_rows(S, P_orig, M, Nc, scale, max_val, Nc, Nc);
        gemm_fp32_4x4(P_orig, V, O, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    double orig_total = elapsed * (cpu_freq / timer_freq) / iterations;
    long long fmla_ops = (long long)M * Nc * D * 2;
    double gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("Original: %.1f cycles, %.2f GFLOPS (%.1f%%)\n",
           orig_total, gflops, gflops / 128.0 * 100);

    /* Fast two-pass */
    for (int i = 0; i < 10; i++) {
        exp2_fast_rows(S, P_fast, M, Nc, scale, neg_max, Nc, Nc);
        gemm_fp32_4x4(P_fast, V, O, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_fast_rows(S, P_fast, M, Nc, scale, neg_max, Nc, Nc);
        gemm_fp32_4x4(P_fast, V, O, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    double fast_total = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("Fast: %.1f cycles, %.2f GFLOPS (%.1f%%)\n",
           fast_total, gflops, gflops / 128.0 * 100);

    /* Ultra-fast two-pass */
    for (int i = 0; i < 10; i++) {
        exp2_ultra_rows(S, P_ultra, M, Nc, scale, neg_max, Nc, Nc);
        gemm_fp32_4x4(P_ultra, V, O, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_ultra_rows(S, P_ultra, M, Nc, scale, neg_max, Nc, Nc);
        gemm_fp32_4x4(P_ultra, V, O, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    double ultra_total = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("Ultra: %.1f cycles, %.2f GFLOPS (%.1f%%)\n",
           ultra_total, gflops, gflops / 128.0 * 100);

    free(P_ultra);

    free(S);
    free(P_ref);
    free(P_orig);
    free(P_fast);
    free(V);
    free(O);

    return 0;
}
