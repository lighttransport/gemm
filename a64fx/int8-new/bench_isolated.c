// bench_isolated.c - Isolated benchmark for each kernel
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_pack_opt.h"

int main(int argc, char** argv) {
    printf("=== Isolated Kernel Benchmark ===\n\n");

    // Which kernel to test (0=ultra, 1=pipe)
    int kernel = 0;
    if (argc > 1) {
        kernel = atoi(argv[1]);
    }

    printf("Testing kernel: %s\n", kernel == 0 ? "Ultra" : "Pipe");

    int M = 4096, N = 4096, K = 256;
    int iterations = 10;

    // Allocate
    int8_t* A = malloc((size_t)M * K);
    int8_t* B = malloc((size_t)N * K);
    int32_t* C = malloc((size_t)M * N * sizeof(int32_t));

    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    // Get frequency
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    // Warmup
    printf("Warmup... ");
    fflush(stdout);
    if (kernel == 0) {
        gemm_opt_driver(A, K, B, K, C, N, M, N, K);
    } else {
        gemm_pipe_driver(A, K, B, K, C, N, M, N, K);
    }
    printf("done\n");

    // Checksum to verify work was done
    int64_t sum = 0;
    for (int i = 0; i < 10; i++) sum += C[i * M * N / 10];
    printf("Warmup checksum: %ld\n", sum);

    // Benchmark
    printf("Running %d iterations... ", iterations);
    fflush(stdout);

    uint64_t t0, t1;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t0));

    for (int i = 0; i < iterations; i++) {
        if (kernel == 0) {
            gemm_opt_driver(A, K, B, K, C, N, M, N, K);
        } else {
            gemm_pipe_driver(A, K, B, K, C, N, M, N, K);
        }
    }

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t1));
    printf("done\n");

    // Final checksum
    sum = 0;
    for (int i = 0; i < 10; i++) sum += C[i * M * N / 10];
    printf("Final checksum: %ld\n", sum);

    // Compute stats
    uint64_t cycles = t1 - t0;
    double time_s = (double)cycles / (double)freq;
    double ops = 2.0 * M * N * K * iterations;
    double gops = ops / time_s / 1e9;

    printf("\nResults:\n");
    printf("  Cycles: %lu\n", (unsigned long)cycles);
    printf("  Time: %.6f seconds (%.3f ms/call)\n", time_s, time_s * 1000.0 / iterations);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

    free(A);
    free(B);
    free(C);

    return 0;
}
