// bench_prepacked2.c - Simplified pre-packed GEMM benchmark
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_prepacked.h"

int main() {
    printf("=== Pre-packed GEMM Benchmark V2 ===\n\n");

    // Get timer frequency ONCE at the start
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    int M = 4096, N = 4096, K = 256;
    int iterations = 10;

    // Allocate original matrices
    int8_t* A = malloc((size_t)M * K);
    int8_t* B = malloc((size_t)N * K);
    int32_t* C = malloc((size_t)M * N * sizeof(int32_t));

    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;

    printf("M=%d, N=%d, K=%d, iterations=%d\n\n", M, N, K, iterations);

    // Pre-pack matrices
    printf("Pre-packing matrices...\n");
    packed_matrix_t* Apack = pack_A_prepacked(A, K, M, K);
    packed_matrix_t* Bpack = pack_B_prepacked(B, K, N, K);
    printf("Packing done.\n");

    if (!Apack || !Bpack) {
        printf("Packing failed!\n");
        return 1;
    }

    // Warmup
    memset(C, 0, (size_t)M * N * sizeof(int32_t));
    gemm_prepacked(Apack, Bpack, C, N);
    printf("Warmup C[0]=%d (expected 256)\n\n", C[0]);

    // Benchmark
    printf("Benchmarking...\n");
    uint64_t t0, t1;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t0));

    for (int i = 0; i < iterations; i++) {
        gemm_prepacked(Apack, Bpack, C, N);
    }

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t1));

    // Verify result
    int64_t checksum = 0;
    for (int i = 0; i < 10; i++) {
        checksum += C[i * M * N / 10];
    }
    printf("Final checksum: %ld\n", checksum);

    // Calculate stats
    uint64_t cycles = t1 - t0;
    printf("\nRaw measurements:\n");
    printf("  t0 = %lu\n", (unsigned long)t0);
    printf("  t1 = %lu\n", (unsigned long)t1);
    printf("  cycles = %lu\n", (unsigned long)cycles);
    printf("  freq = %lu\n", (unsigned long)freq);

    // Use double arithmetic carefully
    double cycles_d = (double)cycles;
    double freq_d = (double)freq;
    double time_s = cycles_d / freq_d;
    double ops = 2.0 * (double)M * (double)N * (double)K * (double)iterations;
    double gops = ops / time_s / 1e9;

    printf("\nCalculated:\n");
    printf("  Time: %.6f seconds (%.3f ms/call)\n", time_s, time_s * 1000.0 / iterations);
    printf("  Ops: %.0f\n", ops);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

    // Cleanup
    free_packed_matrix(Apack);
    free_packed_matrix(Bpack);
    free(A);
    free(B);
    free(C);

    return 0;
}
