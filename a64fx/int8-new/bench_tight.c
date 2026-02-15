// bench_tight.c - Tight loop benchmark with minimal overhead
// Use sizes that are exact multiples of tile sizes to eliminate edge cases
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_prepacked.h"

// Tight GEMM loop - assumes M is multiple of 6, N is multiple of 64
static void gemm_tight(const packed_matrix_t* Apack,
                       const packed_matrix_t* Bpack,
                       int32_t* C, int ldc) {
    int M = Apack->rows;
    int N = Bpack->rows;
    int K = 256;

    int m_tiles = M / 6;
    int n_tiles = N / 64;
    int ldc_bytes = ldc * sizeof(int32_t);

    int8_t* Aptr = Apack->data;
    for (int mt = 0; mt < m_tiles; mt++) {
        int8_t* Bptr = Bpack->data;
        int32_t* Crow = C + mt * 6 * ldc;

        for (int nt = 0; nt < n_tiles; nt++) {
            kernel_6x4_ultra(Aptr, Bptr, Crow + nt * 64, ldc_bytes);
            Bptr += 64 * K;
        }
        Aptr += 6 * K;
    }
}

int main() {
    printf("=== Tight Loop GEMM Benchmark ===\n\n");

    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    // Use sizes that are exact multiples of tile sizes
    // 4032 = 6 * 672 (no edge cases for M)
    // 4096 = 64 * 64 (no edge cases for N)
    int M = 4032, N = 4096, K = 256;
    int iterations = 10;

    printf("M=%d (6×%d), N=%d (64×%d), K=%d\n\n", M, M/6, N, N/64, K);

    // Allocate
    int8_t* A = malloc((size_t)M * K);
    int8_t* B = malloc((size_t)N * K);
    int32_t* C = malloc((size_t)M * N * sizeof(int32_t));

    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    // Pre-pack
    packed_matrix_t* Apack = pack_A_prepacked(A, K, M, K);
    packed_matrix_t* Bpack = pack_B_prepacked(B, K, N, K);

    // Warmup
    gemm_tight(Apack, Bpack, C, N);
    printf("Warmup C[0]=%d (expected 256)\n\n", C[0]);

    // Benchmark tight loop
    printf("Benchmarking tight loop (%d iterations)...\n", iterations);
    uint64_t t0, t1;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t0));

    for (int i = 0; i < iterations; i++) {
        gemm_tight(Apack, Bpack, C, N);
    }

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t1));

    uint64_t cycles = t1 - t0;
    double time_s = (double)cycles / (double)freq;
    double ops = 2.0 * M * N * K * iterations;
    double gops = ops / time_s / 1e9;

    printf("\nTight loop results:\n");
    printf("  Cycles: %lu\n", (unsigned long)cycles);
    printf("  Time: %.3f ms/call\n", time_s * 1000.0 / iterations);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

    // Compare to standard prepacked
    printf("\nBenchmarking standard prepacked (%d iterations)...\n", iterations);
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t0));

    for (int i = 0; i < iterations; i++) {
        gemm_prepacked(Apack, Bpack, C, N);
    }

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t1));

    cycles = t1 - t0;
    time_s = (double)cycles / (double)freq;
    gops = ops / time_s / 1e9;

    printf("\nStandard prepacked results:\n");
    printf("  Time: %.3f ms/call\n", time_s * 1000.0 / iterations);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

    // Theoretical analysis
    int n_kernels = (M / 6) * (N / 64);
    printf("\nAnalysis:\n");
    printf("  Total kernel calls: %d\n", n_kernels);
    printf("  Cycles per kernel call: %.1f\n",
           (double)cycles / iterations / n_kernels);

    free_packed_matrix(Apack);
    free_packed_matrix(Bpack);
    free(A);
    free(B);
    free(C);

    return 0;
}
