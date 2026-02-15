#include "gemm_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Timer functions (A64FX CNT registers)
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

// ============================================================================
// Naive reference implementation: C = A × B^T
// ============================================================================
void gemm_ref_i8s32(const int8_t* A, int lda,
                    const int8_t* B, int ldb,
                    int32_t* C, int ldc,
                    int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                // C[m,n] = sum(A[m,k] * B[n,k])  -- B accessed as B^T
                sum += (int32_t)A[m * lda + k] * (int32_t)B[n * ldb + k];
            }
            C[m * ldc + n] = sum;
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================
static void init_matrix_i8(int8_t* mat, int rows, int cols, int ld, int seed) {
    srand(seed);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Random values in [-10, 10]
            mat[i * ld + j] = (int8_t)((rand() % 21) - 10);
        }
    }
}

static void zero_matrix_i32(int32_t* mat, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * ld + j] = 0;
        }
    }
}

static int compare_matrices_i32(const int32_t* A, const int32_t* B,
                                int rows, int cols, int ld,
                                const char* name_a, const char* name_b) {
    int errors = 0;
    int max_print = 10;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int32_t a_val = A[i * ld + j];
            int32_t b_val = B[i * ld + j];
            if (a_val != b_val) {
                if (errors < max_print) {
                    printf("  ERROR at [%d,%d]: %s=%d, %s=%d, diff=%d\n",
                           i, j, name_a, a_val, name_b, b_val, a_val - b_val);
                }
                errors++;
            }
        }
    }

    if (errors > 0) {
        printf("  Total errors: %d / %d elements\n", errors, rows * cols);
        return -1;
    }
    return 0;
}

// ============================================================================
// Correctness tests
// ============================================================================
#ifdef USE_5X4
static int test_correctness_5x4(void) {
    printf("=== Testing 5x4 Kernel Correctness ===\n");

    int test_cases[][3] = {
        {10, 128, 256},   // Small M, moderate N
        {15, 192, 256},   // Edge case: M % 5 != 0, N % 64 != 0
        {25, 256, 256},   // Multiple tiles
        {100, 512, 256},  // Larger problem
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;

    for (int t = 0; t < num_tests; t++) {
        int M = test_cases[t][0];
        int N = test_cases[t][1];
        int K = test_cases[t][2];

        printf("\nTest %d: M=%d, N=%d, K=%d\n", t + 1, M, N, K);

        // Allocate matrices
        int8_t* A = (int8_t*)malloc(M * K * sizeof(int8_t));
        int8_t* B = (int8_t*)malloc(N * K * sizeof(int8_t));
        int32_t* C_ref = (int32_t*)malloc(M * N * sizeof(int32_t));
        int32_t* C_opt = (int32_t*)malloc(M * N * sizeof(int32_t));

        // Initialize
        init_matrix_i8(A, M, K, K, 42 + t);
        init_matrix_i8(B, N, K, K, 99 + t);
        zero_matrix_i32(C_ref, M, N, N);
        zero_matrix_i32(C_opt, M, N, N);

        // Compute reference
        gemm_ref_i8s32(A, K, B, K, C_ref, N, M, N, K);

        // Compute with 5x4 kernel
        gemm_5x4_driver(A, K, B, K, C_opt, N, M, N, K);

        // Compare
        int result = compare_matrices_i32(C_ref, C_opt, M, N, N, "ref", "5x4");

        if (result == 0) {
            printf("  PASS\n");
            passed++;
        } else {
            printf("  FAIL\n");
        }

        free(A);
        free(B);
        free(C_ref);
        free(C_opt);
    }

    printf("\n=== 5x4 Correctness: %d / %d tests passed ===\n\n", passed, num_tests);
    return (passed == num_tests) ? 0 : -1;
}
#endif

#ifdef USE_6X4
static int test_correctness_6x4(void) {
    printf("=== Testing 6x4 Kernel Correctness ===\n");

    int test_cases[][3] = {
        {12, 128, 256},
        {18, 192, 256},   // Edge case: M % 6 != 0, N % 64 != 0
        {30, 256, 256},
        {100, 512, 256},
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;

    for (int t = 0; t < num_tests; t++) {
        int M = test_cases[t][0];
        int N = test_cases[t][1];
        int K = test_cases[t][2];

        printf("\nTest %d: M=%d, N=%d, K=%d\n", t + 1, M, N, K);

        int8_t* A = (int8_t*)malloc(M * K * sizeof(int8_t));
        int8_t* B = (int8_t*)malloc(N * K * sizeof(int8_t));
        int32_t* C_ref = (int32_t*)malloc(M * N * sizeof(int32_t));
        int32_t* C_opt = (int32_t*)malloc(M * N * sizeof(int32_t));

        init_matrix_i8(A, M, K, K, 42 + t);
        init_matrix_i8(B, N, K, K, 99 + t);
        zero_matrix_i32(C_ref, M, N, N);
        zero_matrix_i32(C_opt, M, N, N);

        gemm_ref_i8s32(A, K, B, K, C_ref, N, M, N, K);
        gemm_6x4_driver(A, K, B, K, C_opt, N, M, N, K);

        int result = compare_matrices_i32(C_ref, C_opt, M, N, N, "ref", "6x4");

        if (result == 0) {
            printf("  PASS\n");
            passed++;
        } else {
            printf("  FAIL\n");
        }

        free(A);
        free(B);
        free(C_ref);
        free(C_opt);
    }

    printf("\n=== 6x4 Correctness: %d / %d tests passed ===\n\n", passed, num_tests);
    return (passed == num_tests) ? 0 : -1;
}
#endif

// ============================================================================
// Performance benchmarks
// ============================================================================
static void bench_kernel(const char* name,
                        void (*gemm_func)(const int8_t*, int, const int8_t*, int,
                                         int32_t*, int, int, int, int),
                        int M, int N, int K, int warmup, int iterations) {
    // Allocate matrices
    int8_t* A = (int8_t*)malloc(M * K * sizeof(int8_t));
    int8_t* B = (int8_t*)malloc(N * K * sizeof(int8_t));
    int32_t* C = (int32_t*)malloc(M * N * sizeof(int32_t));

    // Initialize
    init_matrix_i8(A, M, K, K, 42);
    init_matrix_i8(B, N, K, K, 99);
    zero_matrix_i32(C, M, N, N);

    uint64_t freq = get_timer_freq();

    // Warmup
    for (int i = 0; i < warmup; i++) {
        gemm_func(A, K, B, K, C, N, M, N, K);
    }

    // Benchmark
    uint64_t t0 = read_timer();
    for (int i = 0; i < iterations; i++) {
        gemm_func(A, K, B, K, C, N, M, N, K);
    }
    uint64_t t1 = read_timer();

    // Calculate performance
    double elapsed = (double)(t1 - t0) / freq / iterations;
    double ops = 2.0 * M * N * K;  // MACs count as 2 ops
    double gops = ops / elapsed / 1e9;

    // A64FX single core INT8 peak: ~8 GOPS (2 FPU × 2 ops/cycle × 2 GHz)
    double peak_gops = 8.0;
    double efficiency = 100.0 * gops / peak_gops;

    printf("%-10s M=%4d N=%4d K=%d: %7.2f GOPS (%5.1f%% peak) [%.3f ms]\n",
           name, M, N, K, gops, efficiency, elapsed * 1000.0);

    free(A);
    free(B);
    free(C);
}

static void run_benchmarks(void) {
    printf("\n=== Performance Benchmarks ===\n");
    printf("Single-threaded performance on A64FX\n");
    printf("Target peak: ~8 GOPS/core for INT8 SDOT\n\n");

    int M_sizes[] = {512, 1024, 2048, 4096};
    int N_sizes[] = {1024, 2048, 4096};
    int K = 256;
    int warmup = 3;
    int iterations = 10;

    for (int i = 0; i < sizeof(M_sizes) / sizeof(M_sizes[0]); i++) {
        for (int j = 0; j < sizeof(N_sizes) / sizeof(N_sizes[0]); j++) {
            int M = M_sizes[i];
            int N = N_sizes[j];

#ifdef USE_5X4
            bench_kernel("5x4", gemm_5x4_driver, M, N, K, warmup, iterations);
#endif
#ifdef USE_6X4
            bench_kernel("6x4", gemm_6x4_driver, M, N, K, warmup, iterations);
#endif
        }
        printf("\n");
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    printf("INT8 GEMM Benchmark for A64FX SVE SDOT\n");
    printf("========================================\n\n");

    // Run correctness tests
#ifdef USE_5X4
    printf("Testing 5x4 microkernel...\n");
    if (test_correctness_5x4() != 0) {
        printf("ERROR: 5x4 correctness tests failed!\n");
        return 1;
    }
#endif

#ifdef USE_6X4
    printf("Testing 6x4 microkernel...\n");
    if (test_correctness_6x4() != 0) {
        printf("ERROR: 6x4 correctness tests failed!\n");
        return 1;
    }
#endif

    printf("All correctness tests passed!\n");

    // Run performance benchmarks
    run_benchmarks();

    return 0;
}
