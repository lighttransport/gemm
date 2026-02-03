/*
 * Benchmark for K-blocked fused exp2 + FMLA kernel
 * Target: exp2 overhead < 5% of GEMM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* K-blocked kernel */
extern void exp2_fmla_kblock16(
    const int32_t* S, const float* V, float* O,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o);

/* Baseline kernels */
extern void gemm_fp32_4x4(
    const float* A, const float* B, float* O,
    int K, int ld_a, int ld_b, int ld_o);

extern void exp2_rows(
    const int32_t* S, float* P, int M, int Nc,
    float scale, float max_val, int ld_s, int ld_p);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static void ref_exp2_fmla(const int32_t* S, const float* V, float* O,
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
        if (rel > 0.02f && err > 1e-4f) errors++;
    }
    return errors;
}

int main(int argc, char** argv) {
    int Nc = 64;
    int iterations = 1000;

    if (argc > 1) Nc = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    /* Nc must be multiple of 16 for kblock kernel */
    Nc = (Nc / 16) * 16;
    if (Nc < 16) Nc = 16;

    int M = 4, D = 64;

    printf("=== K-Blocked Fused exp2 + FMLA Benchmark ===\n");
    printf("M=%d, Nc=%d, D=%d\n", M, Nc, D);
    printf("Iterations: %d\n\n", iterations);

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

    ref_exp2_fmla(S, V, O_ref, M, Nc, D, scale, max_val);

    double timer_freq = 100e6;
    double cpu_freq = 2e9;
    double fp32_peak = 128.0;
    long long fmla_ops = (long long)M * Nc * D * 2;

    volatile uint64_t start, end;
    double elapsed, cycles, gflops;

    /* Test K-blocked fused kernel */
    printf("=== K-Blocked Fused (K=16 blocks) ===\n");
    memset(O_test, 0, M * D * sizeof(float));
    exp2_fmla_kblock16(S, V, O_test, Nc, scale, max_val, Nc, ld_v, ld_o);

    float max_err;
    int errors = verify(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: error=%.4f%% %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    for (int i = 0; i < 10; i++)
        exp2_fmla_kblock16(S, V, O_test, Nc, scale, max_val, Nc, ld_v, ld_o);

    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        exp2_fmla_kblock16(S, V, O_test, Nc, scale, max_val, Nc, ld_v, ld_o);
    end = get_cycles();

    elapsed = (double)(end - start);
    cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Cycles: %.1f, per K: %.2f\n", cycles, cycles / Nc);
    printf("  GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    /* Two-pass baseline */
    printf("\n=== Two-Pass (exp2_rows + gemm) ===\n");
    memset(O_test, 0, M * D * sizeof(float));
    exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
    gemm_fp32_4x4(P, V, O_test, Nc, Nc * sizeof(float), ld_v, ld_o);

    errors = verify(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: error=%.4f%% %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    for (int i = 0; i < 10; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
        gemm_fp32_4x4(P, V, O_test, Nc, Nc * sizeof(float), ld_v, ld_o);
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
        gemm_fp32_4x4(P, V, O_test, Nc, Nc * sizeof(float), ld_v, ld_o);
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    double twopass_cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Cycles: %.1f, per K: %.2f\n", twopass_cycles, twopass_cycles / Nc);
    printf("  GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    /* Pure GEMM baseline */
    printf("\n=== Pure GEMM (no exp2) ===\n");
    for (int i = 0; i < M * Nc; i++) P[i] = 1.0f;

    for (int i = 0; i < 10; i++)
        gemm_fp32_4x4(P, V, O_test, Nc, Nc * sizeof(float), ld_v, ld_o);

    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        gemm_fp32_4x4(P, V, O_test, Nc, Nc * sizeof(float), ld_v, ld_o);
    end = get_cycles();

    elapsed = (double)(end - start);
    double gemm_cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Cycles: %.1f, per K: %.2f\n", gemm_cycles, gemm_cycles / Nc);
    printf("  GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    /* Analysis */
    printf("\n=== Analysis ===\n");
    double exp2_overhead_twopass = twopass_cycles - gemm_cycles;
    double exp2_overhead_kblock = cycles - gemm_cycles;
    printf("Pure GEMM cycles: %.1f\n", gemm_cycles);
    printf("Two-pass exp2 overhead: %.1f cycles (%.1f%% of GEMM)\n",
           exp2_overhead_twopass, exp2_overhead_twopass / gemm_cycles * 100);
    printf("K-block exp2 overhead: %.1f cycles (%.1f%% of GEMM)\n",
           exp2_overhead_kblock, exp2_overhead_kblock / gemm_cycles * 100);
    printf("\nTarget: < 5%% of GEMM = %.1f cycles for exp2\n", gemm_cycles * 0.05);

    free(S);
    free(V);
    free(O_ref);
    free(O_test);
    free(P);

    return 0;
}
