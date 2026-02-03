/*
 * Clean benchmark for two-pass FlashAttention: exp2 + 8x3 GEMM
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* External kernels */
extern void exp2_fast_rows(const int32_t* S, float* P, int M, int Nc,
                           float scale, float neg_max, int ld_s, int ld_p);
extern void exp2_colmajor_8row(const int32_t* S, float* P, int Nc,
                                float scale, float neg_max);
/* Pack row-major P[M][Nc] to column-major P_packed[Nc][M] */
static void pack_p_colmajor(const float* P, float* P_packed, int M, int Nc) {
    for (int k = 0; k < Nc; k++) {
        for (int m = 0; m < M; m++) {
            P_packed[k * M + m] = P[m * Nc + k];
        }
    }
}
extern void micro_kernel_fp32_8x3_unroll4(const float* A, const float* B,
                                           float* C, int K, int alpha_flag, int ldc);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

/* Reference exp2 for verification */
static void exp2_reference(const int32_t* S, float* P, int M, int Nc,
                           float scale, float neg_max, int ld_s, int ld_p) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < Nc; n++) {
            float x = (float)S[m * ld_s + n] * scale + neg_max;
            P[m * ld_p + n] = exp2f(x);
        }
    }
}

/* Verify results */
static int verify(const float* ref, const float* test, int n, double* max_err) {
    int errors = 0;
    *max_err = 0;
    for (int i = 0; i < n; i++) {
        double err = fabs((double)test[i] - (double)ref[i]);
        double rel = (ref[i] != 0) ? err / fabs(ref[i]) : err;
        if (rel > *max_err) *max_err = rel;
        if (rel > 0.01) errors++;
    }
    return errors;
}

int main(int argc, char* argv[]) {
    int Nc = (argc > 1) ? atoi(argv[1]) : 256;
    int D = (argc > 2) ? atoi(argv[2]) : 48;
    int iterations = (argc > 3) ? atoi(argv[3]) : 1000;

    const int M = 8;
    const int n_tiles = (D + 47) / 48;
    const int D_padded = n_tiles * 48;

    printf("=== Two-Pass exp2 + 8x3 GEMM Benchmark ===\n");
    printf("M=%d, Nc=%d (K), D=%d (%d tiles)\n", M, Nc, D, n_tiles);
    printf("Iterations: %d\n\n", iterations);

    /* Allocate aligned buffers */
    int32_t* S = aligned_alloc(256, M * Nc * sizeof(int32_t));
    float* P = aligned_alloc(256, M * Nc * sizeof(float));
    float* P_packed = aligned_alloc(256, Nc * M * sizeof(float));
    float* V = aligned_alloc(256, Nc * D_padded * sizeof(float));
    float* O_ref = aligned_alloc(256, M * D_padded * sizeof(float));
    float* O_test = aligned_alloc(256, M * D_padded * sizeof(float));

    /* Initialize with realistic values */
    float scale = 1.0f / 8.0f;  /* 1/sqrt(d) for d=64 */
    float neg_max = -5.0f;

    srand(42);
    for (int i = 0; i < M * Nc; i++) {
        S[i] = (rand() % 100) - 50;
    }
    for (int i = 0; i < Nc * D_padded; i++) {
        V[i] = (float)(rand() % 100 - 50) / 100.0f;
    }

    /* Compute reference */
    memset(O_ref, 0, M * D_padded * sizeof(float));
    exp2_reference(S, P, M, Nc, scale, neg_max, Nc, Nc);
    pack_p_colmajor(P, P_packed, M, Nc);
    int ld_c = D_padded;
    for (int tile = 0; tile < n_tiles; tile++) {
        float* V_tile = V + tile * 48;
        float* O_tile = O_ref + tile * 48;
        micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
    }

    /* Timer parameters */
    double timer_freq = 100e6;
    double cpu_freq = 2.0e9;
    double fp32_peak = 128.0;  /* GFLOPS per core */

    uint64_t total_fmla_ops = (uint64_t)M * Nc * D * 2;
    uint64_t start, end;
    double elapsed, cycles, gflops;
    int errors;
    double max_err;

    /*========================================================================
     * Test 1: exp2_fast + pack + GEMM (current best approach)
     *========================================================================*/
    printf("=== Method 1: exp2_fast + pack + GEMM ===\n");

    /* Verify correctness */
    memset(O_test, 0, M * D_padded * sizeof(float));
    exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
    pack_p_colmajor(P, P_packed, M, Nc);
    for (int tile = 0; tile < n_tiles; tile++) {
        float* V_tile = V + tile * 48;
        float* O_tile = O_test + tile * 48;
        micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
    }
    errors = verify(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: %s (max_err=%.4f%%)\n", errors == 0 ? "PASS" : "FAIL", max_err * 100);

    /* Warmup */
    for (int i = 0; i < 100; i++) {
        exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
        pack_p_colmajor(P, P_packed, M, Nc);
        for (int tile = 0; tile < n_tiles; tile++) {
            micro_kernel_fp32_8x3_unroll4(P_packed, V + tile * 48, O_test + tile * 48, Nc, 0, ld_c);
        }
    }

    /* Benchmark */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
        pack_p_colmajor(P, P_packed, M, Nc);
        for (int tile = 0; tile < n_tiles; tile++) {
            micro_kernel_fp32_8x3_unroll4(P_packed, V + tile * 48, O_test + tile * 48, Nc, 0, ld_c);
        }
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)total_fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Total cycles: %.1f\n", cycles);
    printf("  GFLOPS: %.2f (%.1f%% of FP32 peak)\n", gflops, gflops / fp32_peak * 100);
    printf("  Cycles/K: %.2f\n\n", cycles / Nc);

    double method1_cycles = cycles;

    /*========================================================================
     * Test 2: exp2_colmajor + GEMM (no separate pack)
     *========================================================================*/
    printf("=== Method 2: exp2_colmajor + GEMM (no pack) ===\n");

    /* Verify correctness */
    memset(O_test, 0, M * D_padded * sizeof(float));
    exp2_colmajor_8row(S, P_packed, Nc, scale, neg_max);
    for (int tile = 0; tile < n_tiles; tile++) {
        float* V_tile = V + tile * 48;
        float* O_tile = O_test + tile * 48;
        micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
    }
    errors = verify(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: %s (max_err=%.4f%%)\n", errors == 0 ? "PASS" : "FAIL", max_err * 100);

    /* Warmup */
    for (int i = 0; i < 100; i++) {
        exp2_colmajor_8row(S, P_packed, Nc, scale, neg_max);
        for (int tile = 0; tile < n_tiles; tile++) {
            micro_kernel_fp32_8x3_unroll4(P_packed, V + tile * 48, O_test + tile * 48, Nc, 0, ld_c);
        }
    }

    /* Benchmark */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_colmajor_8row(S, P_packed, Nc, scale, neg_max);
        for (int tile = 0; tile < n_tiles; tile++) {
            micro_kernel_fp32_8x3_unroll4(P_packed, V + tile * 48, O_test + tile * 48, Nc, 0, ld_c);
        }
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)total_fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Total cycles: %.1f\n", cycles);
    printf("  GFLOPS: %.2f (%.1f%% of FP32 peak)\n", gflops, gflops / fp32_peak * 100);
    printf("  Cycles/K: %.2f\n\n", cycles / Nc);

    double method2_cycles = cycles;

    /*========================================================================
     * Test 3: Measure individual phases
     *========================================================================*/
    printf("=== Phase Breakdown ===\n");

    /* exp2_fast */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
    }
    end = get_cycles();
    double exp2_fast_cycles = (double)(end - start) * (cpu_freq / timer_freq) / iterations;
    printf("  exp2_fast:     %.1f cycles (%.2f cycles/elem)\n",
           exp2_fast_cycles, exp2_fast_cycles / (M * Nc));

    /* exp2_colmajor */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_colmajor_8row(S, P_packed, Nc, scale, neg_max);
    }
    end = get_cycles();
    double exp2_cm_cycles = (double)(end - start) * (cpu_freq / timer_freq) / iterations;
    printf("  exp2_colmajor: %.1f cycles (%.2f cycles/elem)\n",
           exp2_cm_cycles, exp2_cm_cycles / (M * Nc));

    /* pack */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        pack_p_colmajor(P, P_packed, M, Nc);
    }
    end = get_cycles();
    double pack_cycles = (double)(end - start) * (cpu_freq / timer_freq) / iterations;
    printf("  pack:          %.1f cycles (%.2f cycles/elem)\n",
           pack_cycles, pack_cycles / (M * Nc));

    /* GEMM only */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        for (int tile = 0; tile < n_tiles; tile++) {
            micro_kernel_fp32_8x3_unroll4(P_packed, V + tile * 48, O_test + tile * 48, Nc, 0, ld_c);
        }
    }
    end = get_cycles();
    double gemm_cycles = (double)(end - start) * (cpu_freq / timer_freq) / iterations;
    double gemm_gflops = (double)total_fmla_ops * iterations / ((end - start) / timer_freq) / 1e9;
    printf("  GEMM:          %.1f cycles (%.2f GFLOPS, %.1f%% peak)\n\n",
           gemm_cycles, gemm_gflops, gemm_gflops / fp32_peak * 100);

    /*========================================================================
     * Summary
     *========================================================================*/
    printf("=== Summary ===\n");
    printf("Method 1 (exp2+pack+GEMM): %.1f cycles\n", method1_cycles);
    printf("Method 2 (exp2_cm+GEMM):   %.1f cycles\n", method2_cycles);
    printf("Speedup: %.2fx\n\n", method1_cycles / method2_cycles);

    printf("Phase breakdown (Method 1):\n");
    printf("  exp2_fast: %.1f cycles (%.1f%%)\n", exp2_fast_cycles, exp2_fast_cycles / method1_cycles * 100);
    printf("  pack:      %.1f cycles (%.1f%%)\n", pack_cycles, pack_cycles / method1_cycles * 100);
    printf("  GEMM:      %.1f cycles (%.1f%%)\n", gemm_cycles, gemm_cycles / method1_cycles * 100);
    printf("  Sum:       %.1f cycles\n\n", exp2_fast_cycles + pack_cycles + gemm_cycles);

    printf("Method 2 saves: %.1f cycles (pack elimination)\n",
           (exp2_fast_cycles + pack_cycles) - exp2_cm_cycles);

    free(S);
    free(P);
    free(P_packed);
    free(V);
    free(O_ref);
    free(O_test);

    return 0;
}
