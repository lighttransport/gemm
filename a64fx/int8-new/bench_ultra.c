// bench_ultra.c
// Benchmark for ultra-optimized INT8 SDOT GEMM kernel
// Tests correctness and measures performance vs theoretical peak

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_pack_opt.h"

// Timer functions
static inline uint64_t read_timer(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

// Reference GEMM for correctness check
static void gemm_ref(const int8_t* A, int lda,
                     const int8_t* B, int ldb,
                     int32_t* C, int ldc,
                     int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * lda + k] * (int32_t)B[n * ldb + k];
            }
            C[m * ldc + n] = sum;
        }
    }
}

// Test correctness of the ultra kernel
static int test_correctness(int M, int N, int K) {
    printf("Testing correctness: M=%d, N=%d, K=%d... ", M, N, K);
    fflush(stdout);

    int8_t* A = (int8_t*)malloc((size_t)M * K);
    int8_t* B = (int8_t*)malloc((size_t)N * K);
    int32_t* C_ref = (int32_t*)malloc((size_t)M * N * sizeof(int32_t));
    int32_t* C_opt = (int32_t*)malloc((size_t)M * N * sizeof(int32_t));

    // Initialize with small values to avoid overflow
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 7) - 3;  // -3 to 3
    for (int i = 0; i < N * K; i++) B[i] = (rand() % 7) - 3;
    memset(C_ref, 0, (size_t)M * N * sizeof(int32_t));
    memset(C_opt, 0, (size_t)M * N * sizeof(int32_t));

    // Compute reference
    gemm_ref(A, K, B, K, C_ref, N, M, N, K);

    // Compute optimized
    gemm_opt_driver(A, K, B, K, C_opt, N, M, N, K);

    // Compare
    int errors = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (C_ref[m * N + n] != C_opt[m * N + n]) {
                if (errors < 10) {
                    printf("\n  Mismatch at [%d,%d]: ref=%d, opt=%d",
                           m, n, C_ref[m * N + n], C_opt[m * N + n]);
                }
                errors++;
            }
        }
    }

    free(A);
    free(B);
    free(C_ref);
    free(C_opt);

    if (errors == 0) {
        printf("PASSED\n");
        return 0;
    } else {
        printf("\n  FAILED: %d errors\n", errors);
        return 1;
    }
}

// Benchmark single GEMM with both kernels
static void bench_single(int M, int N, int K, int iterations) {
    printf("\nBenchmark: M=%d, N=%d, K=%d\n", M, N, K);

    int8_t* A = (int8_t*)malloc((size_t)M * K);
    int8_t* B = (int8_t*)malloc((size_t)N * K);
    int32_t* C = (int32_t*)malloc((size_t)M * N * sizeof(int32_t));

    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = 1;
    for (int i = 0; i < N * K; i++) B[i] = 1;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    uint64_t freq = get_timer_freq();
    double ops = 2.0 * (double)M * (double)N * (double)K * iterations;

    // Benchmark ULTRA kernel
    gemm_opt_driver(A, K, B, K, C, N, M, N, K);  // warmup
    uint64_t t0 = read_timer();
    for (int i = 0; i < iterations; i++) {
        gemm_opt_driver(A, K, B, K, C, N, M, N, K);
    }
    uint64_t t1 = read_timer();

    double time_ultra = (double)(t1 - t0) / (double)freq;
    double gops_ultra = ops / time_ultra / 1e9;
    double eff_ultra = 100.0 * gops_ultra / 512.0;

    // Benchmark PIPE kernel
    memset(C, 0, (size_t)M * N * sizeof(int32_t));
    gemm_pipe_driver(A, K, B, K, C, N, M, N, K);  // warmup
    t0 = read_timer();
    for (int i = 0; i < iterations; i++) {
        gemm_pipe_driver(A, K, B, K, C, N, M, N, K);
    }
    t1 = read_timer();

    double time_pipe = (double)(t1 - t0) / (double)freq;
    double gops_pipe = ops / time_pipe / 1e9;
    double eff_pipe = 100.0 * gops_pipe / 512.0;

    printf("  Ultra: %.2f GOPS (%.1f%%), %.3f ms/call\n",
           gops_ultra, eff_ultra, time_ultra*1000.0/iterations);
    printf("  Pipe:  %.2f GOPS (%.1f%%), %.3f ms/call\n",
           gops_pipe, eff_pipe, time_pipe*1000.0/iterations);

    free(A);
    free(B);
    free(C);
}

// Benchmark batched GEMM (simulating multiple attention heads)
static void bench_batch(int M, int N, int K, int batch, int iterations) {
    printf("\nBatch benchmark: M=%d, N=%d, K=%d, batch=%d (heads)\n", M, N, K, batch);

    size_t A_size = (size_t)M * K * batch;
    size_t B_size = (size_t)N * K * batch;
    size_t C_size = (size_t)M * N * batch;

    int8_t* A = (int8_t*)malloc(A_size);
    int8_t* B = (int8_t*)malloc(B_size);
    int32_t* C = (int32_t*)malloc(C_size * sizeof(int32_t));

    // Initialize
    for (size_t i = 0; i < A_size; i++) A[i] = 1;
    for (size_t i = 0; i < B_size; i++) B[i] = 1;
    memset(C, 0, C_size * sizeof(int32_t));

    int64_t strideA = M * K;
    int64_t strideB = N * K;
    int64_t strideC = M * N;

    uint64_t freq = get_timer_freq();

    // Warmup
    gemm_batch_opt(A, K, strideA, B, K, strideB, C, N, strideC, M, N, K, batch);

    // Benchmark
    uint64_t t0 = read_timer();
    for (int i = 0; i < iterations; i++) {
        gemm_batch_opt(A, K, strideA, B, K, strideB, C, N, strideC, M, N, K, batch);
    }
    uint64_t t1 = read_timer();

    double time = (double)(t1 - t0) / (double)freq;
    double ops = 2.0 * (double)M * (double)N * (double)K * batch * iterations;
    double gops = ops / time / 1e9;
    double efficiency = 100.0 * gops / 512.0;
    double time_per_call = time * 1000.0 / iterations;

    printf("  Time: %.3f ms/call (all %d heads)\n", time_per_call, batch);
    printf("  Performance: %.2f GOPS (%.1f%% of 512 GOPS peak)\n", gops, efficiency);

    free(A);
    free(B);
    free(C);
}

int main(int argc, char** argv) {
    printf("=== INT8 SDOT GEMM Ultra Benchmark ===\n");
    printf("Target: 95%% of 512 GOPS peak = 486 GOPS\n\n");

    // Correctness tests
    printf("--- Correctness Tests ---\n");
    int errors = 0;
    errors += test_correctness(6, 64, 256);     // Single tile
    errors += test_correctness(12, 128, 256);   // 2x2 tiles
    errors += test_correctness(100, 200, 256);  // Edge cases
    errors += test_correctness(256, 256, 256);  // Square

    if (errors > 0) {
        printf("\nCorrectness tests failed. Aborting benchmark.\n");
        return 1;
    }

    printf("\nAll correctness tests passed!\n");

    // Performance benchmarks
    printf("\n--- Performance Benchmarks ---\n");

    // Standard attention sizes
    bench_single(4096, 4096, 256, 10);
    bench_single(8192, 8192, 256, 5);

    // Batched GEMM (simulating multi-head attention)
    // Typical: 32 heads, seq_len varies
    bench_batch(128, 128, 256, 32, 100);   // Short sequences
    bench_batch(512, 512, 256, 32, 20);    // Medium sequences
    bench_batch(2048, 2048, 256, 8, 10);   // Long sequences, fewer heads

    return 0;
}
