// Simplified benchmark
#include "fused_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("Simple Fused GEMM Benchmark\n");

    #ifdef _OPENMP
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    #endif

    // MLP-like config
    int M = 512;
    int K1 = 768;
    int K2 = 3072;
    int N = 768;

    printf("Config: E[%d,%d] = (A[%d,%d] @ B[%d,%d]) @ C[%d,%d]\n",
           M, N, M, K1, K1, K2, K2, N);

    // Allocate
    float* A = (float*)aligned_alloc(64, (size_t)M * K1 * sizeof(float));
    float* B = (float*)aligned_alloc(64, (size_t)K1 * K2 * sizeof(float));
    float* C = (float*)aligned_alloc(64, (size_t)K2 * N * sizeof(float));
    float* E = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    // Initialize
    srand(42);
    for (size_t i = 0; i < M * K1; i++) A[i] = (float)(rand() % 1000 - 500) / 5000.0f;
    for (size_t i = 0; i < K1 * K2; i++) B[i] = (float)(rand() % 1000 - 500) / 5000.0f;
    for (size_t i = 0; i < K2 * N; i++) C[i] = (float)(rand() % 1000 - 500) / 5000.0f;
    memset(E, 0, (size_t)M * N * sizeof(float));

    // Warmup
    printf("Warmup...\n");
    for (int i = 0; i < 3; i++) {
        fused_gemm_abc(M, K1, K2, N, A, K1, B, K2, C, N, E, N);
    }

    // Benchmark
    printf("Benchmarking...\n");
    int iters = 10;

    volatile double t0 = get_time();
    printf("t0 = %f\n", t0);

    for (int i = 0; i < iters; i++) {
        fused_gemm_abc(M, K1, K2, N, A, K1, B, K2, C, N, E, N);
    }

    volatile double t1 = get_time();
    printf("t1 = %f\n", t1);

    double elapsed = (double)t1 - (double)t0;
    printf("elapsed = %f seconds\n", elapsed);

    double time_per_iter = elapsed / iters;
    printf("time_per_iter = %f seconds\n", time_per_iter);

    // FLOPs
    double flops_pass1 = 2.0 * M * K1 * K2;
    double flops_pass2 = 2.0 * M * K2 * N;
    double total_flops = flops_pass1 + flops_pass2;

    printf("Total FLOPs: %.2f GFLOP\n", total_flops / 1e9);
    printf("Time: %.3f ms\n", time_per_iter * 1000);
    printf("Performance: %.2f GFLOPS\n", total_flops / time_per_iter / 1e9);

    // Cleanup
    free(A);
    free(B);
    free(C);
    free(E);

    return 0;
}
