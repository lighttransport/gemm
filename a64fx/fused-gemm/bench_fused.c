// Benchmark for Fused GEMM (A@B)@C on A64FX
// Tests correctness and performance

#include "fused_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Get current time in seconds
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Initialize matrix with random values
static void init_random(float* M, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M[i * ld + j] = (float)(rand() % 1000 - 500) / 500.0f;
        }
    }
}

// Reference GEMM (naive)
static void gemm_reference(
    int M, int K, int N,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = sum;
        }
    }
}

// Reference fused GEMM: E = (A @ B) @ C
static void fused_gemm_reference(
    int M, int K1, int K2, int N,
    const float* A, int lda,
    const float* B, int ldb,
    const float* C, int ldc,
    float* E, int lde
) {
    // Allocate temporary D
    float* D = (float*)malloc((size_t)M * K2 * sizeof(float));

    // D = A @ B
    gemm_reference(M, K1, K2, A, lda, B, ldb, D, K2);

    // E = D @ C
    gemm_reference(M, K2, N, D, K2, C, ldc, E, lde);

    free(D);
}

// Check correctness
static int check_result(
    const float* result, const float* reference,
    int M, int N, int ld_result, int ld_ref,
    float tolerance
) {
    float max_err = 0.0f;
    float max_rel_err = 0.0f;
    int errors = 0;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float r = result[m * ld_result + n];
            float ref = reference[m * ld_ref + n];
            float err = fabsf(r - ref);
            float rel_err = err / (fabsf(ref) + 1e-6f);

            if (err > max_err) max_err = err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;

            if (rel_err > tolerance) {
                if (errors < 10) {
                    printf("  Error at [%d,%d]: got %f, expected %f (err=%e, rel=%e)\n",
                           m, n, r, ref, err, rel_err);
                }
                errors++;
            }
        }
    }

    printf("  Max absolute error: %e\n", max_err);
    printf("  Max relative error: %e\n", max_rel_err);

    return errors;
}

// Test single GEMM kernel
static void test_single_gemm(int M, int K, int N) {
    printf("\n=== Single GEMM Test: M=%d, K=%d, N=%d ===\n", M, K, N);

    float* A = (float*)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float* B = (float*)aligned_alloc(64, (size_t)K * N * sizeof(float));
    float* C = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));
    float* C_ref = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    init_random(A, M, K, K);
    init_random(B, K, N, N);
    memset(C, 0, (size_t)M * N * sizeof(float));
    memset(C_ref, 0, (size_t)M * N * sizeof(float));

    // Reference
    printf("Computing reference...\n");
    gemm_reference(M, K, N, A, K, B, N, C_ref, N);

    // Optimized
    printf("Computing optimized...\n");
    gemm_fp32(M, K, N, A, K, B, N, C, N);

    // Check
    printf("Checking correctness...\n");
    int errors = check_result(C, C_ref, M, N, N, N, 1e-4f);
    printf("  Errors: %d\n", errors);

    // Benchmark
    printf("Benchmarking...\n");
    int warmup = 3;
    int iters = 10;

    for (int i = 0; i < warmup; i++) {
        gemm_fp32(M, K, N, A, K, B, N, C, N);
    }

    volatile double t0 = get_time();
    for (int i = 0; i < iters; i++) {
        gemm_fp32(M, K, N, A, K, B, N, C, N);
    }
    volatile double t1 = get_time();

    double elapsed = (double)t1 - (double)t0;
    double time_per_iter = elapsed / iters;
    double flops = 2.0 * M * K * N;
    double gflops = flops / time_per_iter / 1e9;

    printf("  Time: %.3f ms\n", time_per_iter * 1000);
    printf("  Performance: %.2f GFLOPS\n", gflops);

    // Theoretical peak (A64FX @ 2.2GHz, single core)
    // 2 FMA units * 16 FP32/vec * 2 FLOP/FMA * 2.2 GHz = 140.8 GFLOPS
    double peak = 140.8;
    #ifdef _OPENMP
    peak *= omp_get_max_threads();
    printf("  Threads: %d\n", omp_get_max_threads());
    #endif
    printf("  Efficiency: %.1f%% of %.1f GFLOPS peak\n", 100.0 * gflops / peak, peak);

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

// Test fused GEMM
static void test_fused_gemm(int M, int K1, int K2, int N) {
    printf("\n=== Fused GEMM Test: M=%d, K1=%d, K2=%d, N=%d ===\n", M, K1, K2, N);
    printf("  Computation: E[%d,%d] = (A[%d,%d] @ B[%d,%d]) @ C[%d,%d]\n",
           M, N, M, K1, K1, K2, K2, N);

    float* A = (float*)aligned_alloc(64, (size_t)M * K1 * sizeof(float));
    float* B = (float*)aligned_alloc(64, (size_t)K1 * K2 * sizeof(float));
    float* C = (float*)aligned_alloc(64, (size_t)K2 * N * sizeof(float));
    float* E = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));
    float* E_ref = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    init_random(A, M, K1, K1);
    init_random(B, K1, K2, K2);
    init_random(C, K2, N, N);
    memset(E, 0, (size_t)M * N * sizeof(float));
    memset(E_ref, 0, (size_t)M * N * sizeof(float));

    // Memory sizes
    size_t A_size = (size_t)M * K1 * sizeof(float);
    size_t B_size = (size_t)K1 * K2 * sizeof(float);
    size_t C_size = (size_t)K2 * N * sizeof(float);
    size_t D_size = (size_t)M * K2 * sizeof(float);  // Intermediate
    size_t E_size = (size_t)M * N * sizeof(float);

    printf("  Memory: A=%.1f MB, B=%.1f MB, C=%.1f MB, D=%.1f MB, E=%.1f MB\n",
           A_size / 1e6, B_size / 1e6, C_size / 1e6, D_size / 1e6, E_size / 1e6);
    printf("  Intermediate D fits in L2: %s (8 MB L2 per CMG)\n",
           D_size <= 8 * 1024 * 1024 ? "YES" : "NO");

    // Reference
    printf("Computing reference...\n");
    fused_gemm_reference(M, K1, K2, N, A, K1, B, K2, C, N, E_ref, N);

    // Optimized
    printf("Computing optimized...\n");
    fused_gemm_abc(M, K1, K2, N, A, K1, B, K2, C, N, E, N);

    // Check
    printf("Checking correctness...\n");
    int errors = check_result(E, E_ref, M, N, N, N, 1e-3f);
    printf("  Errors: %d\n", errors);

    // Benchmark
    printf("Benchmarking...\n");
    int warmup = 3;
    int iters = 10;

    for (int i = 0; i < warmup; i++) {
        fused_gemm_abc(M, K1, K2, N, A, K1, B, K2, C, N, E, N);
    }

    volatile double t0 = get_time();
    for (int i = 0; i < iters; i++) {
        fused_gemm_abc(M, K1, K2, N, A, K1, B, K2, C, N, E, N);
    }
    volatile double t1 = get_time();

    double elapsed = (double)t1 - (double)t0;
    double time_per_iter = elapsed / iters;

    // FLOP count for fused GEMM
    double flops_pass1 = 2.0 * M * K1 * K2;
    double flops_pass2 = 2.0 * M * K2 * N;
    double total_flops = flops_pass1 + flops_pass2;
    double gflops = total_flops / time_per_iter / 1e9;

    printf("  Pass 1 FLOPs: %.2f GFLOP\n", flops_pass1 / 1e9);
    printf("  Pass 2 FLOPs: %.2f GFLOP\n", flops_pass2 / 1e9);
    printf("  Total FLOPs: %.2f GFLOP\n", total_flops / 1e9);
    printf("  Time: %.3f ms\n", time_per_iter * 1000);
    printf("  Performance: %.2f GFLOPS\n", gflops);

    double peak = 140.8;
    #ifdef _OPENMP
    peak *= omp_get_max_threads();
    printf("  Threads: %d\n", omp_get_max_threads());
    #endif
    printf("  Efficiency: %.1f%% of %.1f GFLOPS peak\n", 100.0 * gflops / peak, peak);

    free(A);
    free(B);
    free(C);
    free(E);
    free(E_ref);
}

// Compare fused vs unfused performance
static void benchmark_fused_vs_unfused(int M, int K1, int K2, int N) {
    printf("\n=== Fused vs Unfused Comparison: M=%d, K1=%d, K2=%d, N=%d ===\n",
           M, K1, K2, N);

    float* A = (float*)aligned_alloc(64, (size_t)M * K1 * sizeof(float));
    float* B = (float*)aligned_alloc(64, (size_t)K1 * K2 * sizeof(float));
    float* C = (float*)aligned_alloc(64, (size_t)K2 * N * sizeof(float));
    float* D = (float*)aligned_alloc(64, (size_t)M * K2 * sizeof(float));
    float* E = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    init_random(A, M, K1, K1);
    init_random(B, K1, K2, K2);
    init_random(C, K2, N, N);

    int warmup = 3;
    int iters = 10;

    // Unfused: two separate GEMM calls
    printf("Unfused (two separate GEMMs):\n");
    for (int i = 0; i < warmup; i++) {
        gemm_fp32(M, K1, K2, A, K1, B, K2, D, K2);
        gemm_fp32(M, K2, N, D, K2, C, N, E, N);
    }

    volatile double t0 = get_time();
    for (int i = 0; i < iters; i++) {
        gemm_fp32(M, K1, K2, A, K1, B, K2, D, K2);
        gemm_fp32(M, K2, N, D, K2, C, N, E, N);
    }
    volatile double t1 = get_time();
    double unfused_time = ((double)t1 - (double)t0) / iters;

    double total_flops = 2.0 * M * K1 * K2 + 2.0 * M * K2 * N;
    printf("  Time: %.3f ms\n", unfused_time * 1000);
    printf("  GFLOPS: %.2f\n", total_flops / unfused_time / 1e9);

    // Fused
    printf("Fused:\n");
    for (int i = 0; i < warmup; i++) {
        fused_gemm_abc(M, K1, K2, N, A, K1, B, K2, C, N, E, N);
    }

    volatile double fused_t0 = get_time();
    printf("  fused_t0 = %f\n", fused_t0);
    for (int i = 0; i < iters; i++) {
        fused_gemm_abc(M, K1, K2, N, A, K1, B, K2, C, N, E, N);
    }
    volatile double fused_t1 = get_time();
    printf("  fused_t1 = %f\n", fused_t1);
    double fused_elapsed = (double)fused_t1 - (double)fused_t0;
    printf("  fused_elapsed = %f\n", fused_elapsed);
    double fused_time = fused_elapsed / (double)iters;
    printf("  fused_time = %f\n", fused_time);

    double fused_gflops = total_flops / fused_time / 1e9;
    printf("  Time: %.3f ms\n", fused_time * 1000);
    printf("  GFLOPS: %.2f\n", fused_gflops);

    printf("Speedup: %.2fx\n", unfused_time / fused_time);

    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
}

int main(int argc, char** argv) {
    printf("Fused GEMM Benchmark for A64FX\n");
    printf("==============================\n");

    #ifdef _OPENMP
    printf("OpenMP enabled, max threads: %d\n", omp_get_max_threads());
    #else
    printf("OpenMP disabled (single-threaded)\n");
    #endif

    srand(42);

    // Small test for correctness
    printf("\n--- Small Correctness Tests ---\n");
    test_single_gemm(64, 64, 48);
    test_fused_gemm(64, 64, 96, 48);

    // Medium test
    printf("\n--- Medium Size Tests ---\n");
    test_single_gemm(256, 256, 240);
    test_fused_gemm(256, 256, 768, 240);

    // MLP-like configuration (target from plan)
    printf("\n--- MLP-like Configuration ---\n");
    test_fused_gemm(512, 768, 3072, 768);

    // Compare fused vs unfused
    printf("\n--- Fused vs Unfused Comparison ---\n");
    benchmark_fused_vs_unfused(512, 768, 3072, 768);

    // Various sizes
    printf("\n--- Various Sizes ---\n");
    test_fused_gemm(128, 512, 2048, 512);
    test_fused_gemm(256, 1024, 4096, 1024);

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
