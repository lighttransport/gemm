/*
 * Benchmark for fused GEMM + exp2 kernel
 *
 * Tests:
 * 1. Correctness vs reference implementation
 * 2. Performance measurement
 * 3. Comparison: separate GEMM+exp2 vs fused
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <arm_sve.h>

#include "gemm_exp2_fused.h"

/* Debug GEMM function */
extern void gemm_only_4x4(const int8_t* A, const int8_t* B, int32_t* C,
                          int K, int ldc);

/* Timer using cntvct_el0 (100 MHz on A64FX) */
static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

/* Reference GEMM (int8 -> int32) */
static void ref_gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
                          int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* Reference exp2 */
static void ref_exp2_softmax(const int32_t* in, float* out, int n,
                             float scale, float max_val) {
    for (int i = 0; i < n; i++) {
        float x = (float)in[i] * scale - max_val;
        out[i] = exp2f(x);
    }
}

/* Pack A matrix for GEMM: [M][K] -> [K/4][M][4] */
static void pack_A(const int8_t* A, int8_t* Apack, int M, int K) {
    for (int k = 0; k < K; k += 4) {
        for (int m = 0; m < M; m++) {
            for (int kk = 0; kk < 4; kk++) {
                Apack[(k/4) * M * 4 + m * 4 + kk] = A[m * K + k + kk];
            }
        }
    }
}

/* Pack B matrix for GEMM: [K][N] -> [K/4][N][4]
 * SDOT expects: for each output element j, 4 consecutive K values
 * Layout: Bpack[k_block][n][4] where the inner 4 are K values
 */
static void pack_B(const int8_t* B, int8_t* Bpack, int K, int N) {
    for (int k = 0; k < K; k += 4) {
        for (int n = 0; n < N; n++) {
            for (int kk = 0; kk < 4; kk++) {
                Bpack[(k/4) * N * 4 + n * 4 + kk] = B[(k + kk) * N + n];
            }
        }
    }
}

/* Verify results */
static int verify_results(const float* ref, const float* test, int n,
                          float* max_err, float* avg_err) {
    *max_err = 0.0f;
    *avg_err = 0.0f;
    int errors = 0;

    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - test[i]);
        float rel_err = (ref[i] != 0.0f) ? err / fabsf(ref[i]) : err;

        if (rel_err > *max_err) *max_err = rel_err;
        *avg_err += rel_err;

        /* Allow 2% relative error for FEXPA-based exp2 */
        if (rel_err > 0.02f && err > 1e-6f) {
            if (errors < 5) {
                printf("  Error at [%d]: ref=%.6f test=%.6f err=%.2f%%\n",
                       i, ref[i], test[i], rel_err * 100.0f);
            }
            errors++;
        }
    }

    *avg_err /= n;
    return errors;
}

int main(int argc, char** argv) {
    /* Default parameters */
    int M = 4;          /* Tile rows */
    int N = 64;         /* Tile cols (1 SVE vector width for fp32) */
    int K = 64;         /* Reduction dimension */
    int iterations = 1000;

    if (argc > 1) K = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    printf("=== Fused GEMM + exp2 Benchmark ===\n");
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("Iterations: %d\n\n", iterations);

    /* Allocate matrices */
    int8_t* A = aligned_alloc(64, M * K * sizeof(int8_t));
    int8_t* B = aligned_alloc(64, K * N * sizeof(int8_t));
    int8_t* Apack = aligned_alloc(64, (K/4) * M * 4 * sizeof(int8_t));
    int8_t* Bpack = aligned_alloc(64, (K/4) * 4 * N * sizeof(int8_t));
    int32_t* C_gemm = aligned_alloc(64, M * N * sizeof(int32_t));
    float* C_ref = aligned_alloc(64, M * N * sizeof(float));
    float* C_fused = aligned_alloc(64, M * N * sizeof(float));
    float* C_inter = aligned_alloc(64, K * N * sizeof(float));  /* For interleaved */

    if (!A || !B || !Apack || !Bpack || !C_gemm || !C_ref || !C_fused || !C_inter) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Initialize with small random values */
    srand(42);
    for (int i = 0; i < M * K; i++) {
        A[i] = (int8_t)((rand() % 21) - 10);  /* -10 to 10 */
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (int8_t)((rand() % 21) - 10);
    }

    /* Pack matrices */
    pack_A(A, Apack, M, K);
    pack_B(B, Bpack, K, N);

    /* Softmax parameters */
    float scale = 1.0f / 64.0f;  /* Typical attention scale: 1/sqrt(d) */
    float max_val = 5.0f;        /* Approximate max for stability */

    /* ===== Compute reference ===== */
    printf("Computing reference...\n");
    ref_gemm_int8(A, B, C_gemm, M, N, K);
    ref_exp2_softmax(C_gemm, C_ref, M * N, scale, max_val);

    /* Find actual max for better stability */
    float actual_max = -1e30f;
    for (int i = 0; i < M * N; i++) {
        float x = (float)C_gemm[i] * scale;
        if (x > actual_max) actual_max = x;
    }
    printf("Actual max value: %.2f (using %.2f)\n", actual_max, max_val);

    /* Brief reference output */
    printf("Reference GEMM[0]: %d, exp2[0]: %.6f\n", C_gemm[0], C_ref[0]);

    /* ===== Test fused kernel ===== */
    printf("\nTesting gemm_exp2_fused_4x4...\n");
    memset(C_fused, 0, M * N * sizeof(float));

    gemm_exp2_fused_4x4(Apack, Bpack, C_fused, K, scale, max_val, N * sizeof(float));

    float max_err, avg_err;
    int errors = verify_results(C_ref, C_fused, M * N, &max_err, &avg_err);
    printf("  Max relative error: %.4f%%\n", max_err * 100.0f);
    printf("  Avg relative error: %.4f%%\n", avg_err * 100.0f);
    printf("  Result: %s\n", errors == 0 ? "PASS" : "FAIL");

    /* ===== Test interleaved kernel ===== */
    printf("\nTesting gemm_exp2_interleaved...\n");
    memset(C_inter, 0, M * N * sizeof(float));

    gemm_exp2_interleaved(Apack, Bpack, C_inter, K, scale, max_val, N * sizeof(float));

    int inter_errors = verify_results(C_ref, C_inter, M * N, &max_err, &avg_err);
    printf("  Max relative error: %.4f%%\n", max_err * 100.0f);
    printf("  Result: %s\n", inter_errors == 0 ? "PASS" : "FAIL");

    /* ===== Performance benchmark ===== */
    printf("\n=== Performance Benchmark ===\n");

    volatile uint64_t start, end;
    volatile double elapsed;

    /* Warm up */
    for (int i = 0; i < 10; i++) {
        gemm_exp2_fused_4x4(Apack, Bpack, C_fused, K, scale, max_val, N * sizeof(float));
    }

    /* Benchmark fused kernel */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        gemm_exp2_fused_4x4(Apack, Bpack, C_fused, K, scale, max_val, N * sizeof(float));
    }
    end = get_cycles();
    volatile uint64_t fused_elapsed = end - start;

    /* Benchmark interleaved kernel */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        gemm_exp2_interleaved(Apack, Bpack, C_inter, K, scale, max_val, N * sizeof(float));
    }
    end = get_cycles();
    volatile uint64_t inter_elapsed = end - start;

    elapsed = (double)fused_elapsed;
    double timer_freq = 100e6;  /* 100 MHz timer */
    double cpu_freq = 2e9;      /* 2 GHz CPU */
    double cycles_per_timer = cpu_freq / timer_freq;

    double total_cycles = elapsed * cycles_per_timer;
    double cycles_per_call = total_cycles / iterations;

    /* Operations count */
    long long gemm_ops = 2LL * M * N * K;  /* GEMM: 2*M*N*K int8 ops */
    long long exp2_ops = M * N * 14;       /* exp2: ~14 ops per element */
    long long total_ops = gemm_ops + exp2_ops;

    double ops_per_cycle = (double)total_ops / cycles_per_call;
    double gops = (double)total_ops * iterations / (elapsed / timer_freq) / 1e9;

    printf("\ngemm_exp2_fused_4x4:\n");
    printf("  Cycles per call: %.1f\n", cycles_per_call);
    printf("  Throughput: %.2f GOPS\n", gops);

    /* Interleaved kernel stats */
    double inter_total_cycles = (double)inter_elapsed * cycles_per_timer;
    double inter_cycles_per_call = inter_total_cycles / iterations;
    double inter_gops = (double)total_ops * iterations / ((double)inter_elapsed / timer_freq) / 1e9;

    printf("\ngemm_exp2_interleaved:\n");
    printf("  Cycles per call: %.1f\n", inter_cycles_per_call);
    printf("  Throughput: %.2f GOPS\n", inter_gops);
    printf("  Speedup vs fused: %.2fx\n", cycles_per_call / inter_cycles_per_call);

    /* Breakdown */
    double cycles_per_k_iter = cycles_per_call / (K / 4);
    printf("  Cycles per K/4 iteration: %.1f\n", cycles_per_k_iter);

    /* Compare with separate GEMM + exp2 estimate */
    double gemm_cycles_est = (double)(K / 4) * 16 / 2;  /* 16 SDOT, 2/cycle */
    double exp2_cycles_est = M * N * 14.0 / 16 / 2;     /* 14 instr/vec, 2 pipes */
    printf("\nEstimated separate execution:\n");
    printf("  GEMM: %.1f cycles\n", gemm_cycles_est);
    printf("  exp2: %.1f cycles\n", exp2_cycles_est);
    printf("  Total: %.1f cycles\n", gemm_cycles_est + exp2_cycles_est);
    printf("  Fused speedup: %.2fx\n",
           (gemm_cycles_est + exp2_cycles_est) / cycles_per_call);

    /* ===== Instruction breakdown ===== */
    printf("\n=== Instruction Analysis ===\n");
    printf("GEMM (4x4 tile, K=%d):\n", K);
    printf("  SDOT per K/4: 16 (4 rows x 4 cols)\n");
    printf("  Total SDOT: %d\n", (K/4) * 16);
    printf("  Loads per K/4: 4 (A) + 4 (B) = 8\n");
    printf("\nexp2 (16 vectors):\n");
    printf("  Instructions per vector: 14 (fast FEXPA)\n");
    printf("  Total exp2 instructions: %d\n", 16 * 14);
    printf("\nInterleaving potential:\n");
    printf("  SDOT latency: 9 cycles -> can hide 8 exp2 ops\n");
    printf("  exp2 FP latency: 9 cycles -> can hide 8 SDOT ops\n");

    /* Cleanup */
    free(A);
    free(B);
    free(Apack);
    free(Bpack);
    free(C_gemm);
    free(C_ref);
    free(C_fused);
    free(C_inter);

    return errors > 0 ? 1 : 0;
}
