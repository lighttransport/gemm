/*
 * Benchmark for fused exp2 + FMLA GEMM kernels
 * Flash Attention Stage 2: O = exp2(S) @ V
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "exp2_fmla_fused.h"

/* Timer */
static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

/* Reference implementation */
static void ref_exp2_fmla_fp32(
    const int32_t* S, const float* V, float* O,
    int M, int Nc, int D,
    float scale, float max_val,
    int ld_s, int ld_v, int ld_o)
{
    /* Zero output */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            O[i * (ld_o / sizeof(float)) + j] = 0.0f;
        }
    }

    /* O = exp2(S * scale - max) @ V */
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < Nc; k++) {
            float s = (float)S[i * ld_s + k];
            float p = exp2f(s * scale - max_val);
            for (int j = 0; j < D; j++) {
                O[i * (ld_o / sizeof(float)) + j] += p * V[k * (ld_v / sizeof(float)) + j];
            }
        }
    }
}

/* Verify results */
static int verify_fp32(const float* ref, const float* test, int n,
                       float* max_err, float* avg_err) {
    *max_err = 0.0f;
    *avg_err = 0.0f;
    int errors = 0;

    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - test[i]);
        float rel = (fabsf(ref[i]) > 1e-6f) ? err / fabsf(ref[i]) : err;

        if (rel > *max_err) *max_err = rel;
        *avg_err += rel;

        if (rel > 0.02f && err > 1e-4f) {
            if (errors < 5) {
                printf("  Error[%d]: ref=%.6f test=%.6f rel=%.2f%%\n",
                       i, ref[i], test[i], rel * 100.0f);
            }
            errors++;
        }
    }
    *avg_err /= n;
    return errors;
}

int main(int argc, char** argv) {
    int M = 4;              /* Output rows */
    int Nc = 64;            /* Inner dimension (sequence length) */
    int D = 64;             /* Output cols (head dim, 4 SVE vectors) */
    int iterations = 1000;

    if (argc > 1) Nc = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    printf("=== Fused exp2 + FMLA Benchmark ===\n");
    printf("M=%d, Nc=%d, D=%d (4 fp32 vectors)\n", M, Nc, D);
    printf("Iterations: %d\n\n", iterations);

    /* Allocate */
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O_ref = aligned_alloc(64, M * D * sizeof(float));
    float* O_test = aligned_alloc(64, M * D * sizeof(float));

    if (!S || !V || !O_ref || !O_test) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Initialize */
    srand(42);
    for (int i = 0; i < M * Nc; i++) {
        S[i] = (rand() % 201) - 100;  /* -100 to 100 */
    }
    for (int i = 0; i < Nc * D; i++) {
        V[i] = ((float)(rand() % 1000) - 500) / 500.0f;  /* -1 to 1 */
    }

    float scale = 1.0f / 64.0f;
    float max_val = 1.5f;

    int ld_s = Nc;                      /* S is [M][Nc] */
    int ld_v = D * sizeof(float);       /* V is [Nc][D] */
    int ld_o = D * sizeof(float);       /* O is [M][D] */

    /* Reference */
    printf("Computing reference...\n");
    ref_exp2_fmla_fp32(S, V, O_ref, M, Nc, D, scale, max_val, ld_s, ld_v, ld_o);
    printf("Reference O[0:4]: %.6f %.6f %.6f %.6f\n",
           O_ref[0], O_ref[1], O_ref[2], O_ref[3]);

    /* Test FP32 kernel */
    printf("\nTesting exp2_fmla_fp32_4x4...\n");
    memset(O_test, 0, M * D * sizeof(float));
    exp2_fmla_fp32_4x4(S, V, O_test, Nc, scale, max_val, ld_s, ld_v, ld_o);

    printf("Test O[0:4]: %.6f %.6f %.6f %.6f\n",
           O_test[0], O_test[1], O_test[2], O_test[3]);

    float max_err, avg_err;
    int errors = verify_fp32(O_ref, O_test, M * D, &max_err, &avg_err);
    printf("  Max error: %.4f%%, Avg error: %.4f%%\n", max_err * 100, avg_err * 100);
    printf("  Result: %s\n", errors == 0 ? "PASS" : "FAIL");

    /* Benchmark */
    printf("\n=== Performance ===\n");

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        exp2_fmla_fp32_4x4(S, V, O_test, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    volatile uint64_t start, end;
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_fmla_fp32_4x4(S, V, O_test, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    end = get_cycles();

    double timer_freq = 100e6;
    double cpu_freq = 2e9;
    double elapsed = (double)(end - start);
    double total_cycles = elapsed * (cpu_freq / timer_freq);
    double cycles_per_call = total_cycles / iterations;

    /* Operations:
     * - exp2: Nc * M * 14 ops
     * - FMLA: M * Nc * D * 2 FLOPs (multiply-add)
     */
    long long exp2_ops = (long long)Nc * M * 14;
    long long fmla_ops = (long long)M * Nc * D * 2;
    long long total_ops = exp2_ops + fmla_ops;

    double gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;

    printf("exp2_fmla_fp32_4x4:\n");
    printf("  Cycles per call: %.1f\n", cycles_per_call);
    printf("  Cycles per Nc iteration: %.2f\n", cycles_per_call / Nc);
    printf("  FMLA GFLOPS: %.2f\n", gflops);

    /* Peak analysis */
    double fp32_peak = 128.0;  /* 2 pipes * 2 GHz * 16 elem * 2 FLOP */
    printf("  FP32 FMLA efficiency: %.1f%% (peak=%.0f GFLOPS)\n",
           gflops / fp32_peak * 100, fp32_peak);

    /* Breakdown */
    printf("\nBreakdown per Nc iteration:\n");
    printf("  exp2: ~14 scalar ops (for 4 rows)\n");
    printf("  FMLA: 16 ops (4 rows × 4 vectors)\n");
    printf("  Loads: 4 V vectors\n");
    printf("  Theoretical: (16 FMLA + 4 LD) / 2 pipes = 10 cycles\n");

    /* Test vectorized version */
    printf("\n=== Testing exp2_fmla_fp32_vec (16x64 tile) ===\n");

    int M_vec = 16;
    int32_t* S_vec = aligned_alloc(64, M_vec * Nc * sizeof(int32_t));
    float* O_ref_vec = aligned_alloc(64, M_vec * D * sizeof(float));
    float* O_test_vec = aligned_alloc(64, M_vec * D * sizeof(float));

    for (int i = 0; i < M_vec * Nc; i++) {
        S_vec[i] = (rand() % 201) - 100;
    }

    int ld_s_vec = Nc;
    int ld_o_vec = D * sizeof(float);

    ref_exp2_fmla_fp32(S_vec, V, O_ref_vec, M_vec, Nc, D, scale, max_val,
                       ld_s_vec, ld_v, ld_o_vec);

    memset(O_test_vec, 0, M_vec * D * sizeof(float));
    exp2_fmla_fp32_vec(S_vec, V, O_test_vec, Nc, scale, max_val,
                       ld_s_vec, ld_v, ld_o_vec);

    errors = verify_fp32(O_ref_vec, O_test_vec, M_vec * D, &max_err, &avg_err);
    printf("  Max error: %.4f%%, Result: %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    /* Benchmark vec version */
    for (int i = 0; i < 10; i++) {
        exp2_fmla_fp32_vec(S_vec, V, O_test_vec, Nc, scale, max_val,
                           ld_s_vec, ld_v, ld_o_vec);
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_fmla_fp32_vec(S_vec, V, O_test_vec, Nc, scale, max_val,
                           ld_s_vec, ld_v, ld_o_vec);
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    total_cycles = elapsed * (cpu_freq / timer_freq);
    cycles_per_call = total_cycles / iterations;

    fmla_ops = (long long)M_vec * Nc * D * 2;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;

    printf("exp2_fmla_fp32_vec (16x64):\n");
    printf("  Cycles per call: %.1f\n", cycles_per_call);
    printf("  FMLA GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    /* Test two-pass approach: exp2_rows + gemm_fp32_4x4 */
    printf("\n=== Testing Two-Pass: exp2_rows + gemm_fp32 ===\n");

    float* P = aligned_alloc(64, M * Nc * sizeof(float));
    float* O_two = aligned_alloc(64, M * D * sizeof(float));

    /* Pass 1: exp2 */
    exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);

    /* Pass 2: GEMM */
    memset(O_two, 0, M * D * sizeof(float));
    gemm_fp32_4x4(P, V, O_two, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));

    errors = verify_fp32(O_ref, O_two, M * D, &max_err, &avg_err);
    printf("  Max error: %.4f%%, Result: %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    /* Benchmark two-pass */
    for (int i = 0; i < 10; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
        gemm_fp32_4x4(P, V, O_two, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
        gemm_fp32_4x4(P, V, O_two, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    total_cycles = elapsed * (cpu_freq / timer_freq);
    cycles_per_call = total_cycles / iterations;

    fmla_ops = (long long)M * Nc * D * 2;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;

    printf("Two-pass (4x64):\n");
    printf("  Cycles per call: %.1f\n", cycles_per_call);
    printf("  FMLA GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    /* Benchmark pure GEMM (no exp2) for comparison */
    printf("\n=== Pure GEMM baseline (no exp2) ===\n");

    /* Initialize P with all 1s to skip exp2 */
    for (int i = 0; i < M * Nc; i++) P[i] = 1.0f;

    for (int i = 0; i < 10; i++) {
        gemm_fp32_4x4(P, V, O_two, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        gemm_fp32_4x4(P, V, O_two, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    total_cycles = elapsed * (cpu_freq / timer_freq);
    cycles_per_call = total_cycles / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;

    printf("Pure GEMM (4x64):\n");
    printf("  Cycles per call: %.1f\n", cycles_per_call);
    printf("  Cycles per K iteration: %.2f\n", cycles_per_call / Nc);
    printf("  FMLA GFLOPS: %.2f (%.1f%% of peak)\n", gflops, gflops / fp32_peak * 100);

    free(S);
    free(V);
    free(O_ref);
    free(O_test);
    free(S_vec);
    free(O_ref_vec);
    free(O_test_vec);
    free(P);
    free(O_two);

    return errors > 0 ? 1 : 0;
}
