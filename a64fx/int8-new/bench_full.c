// bench_full.c - Full GEMM benchmark with unrolled kernel
// Compare: kernel-only, tight loop, full driver
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_prepacked.h"

// Declare fully unrolled kernel
extern void kernel_6x4_unroll(const int8_t* Apack, const int8_t* Bpack,
                               int32_t* C, int ldc);

static inline uint64_t rdtimer(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t rdfreq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

// Tight GEMM using unrolled kernel - no branches for full tiles
static void gemm_unroll_tight(const packed_matrix_t* Apack,
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
            kernel_6x4_unroll(Aptr, Bptr, Crow + nt * 64, ldc_bytes);
            Bptr += 64 * K;
        }
        Aptr += 6 * K;
    }
}

int main() {
    printf("=== Full GEMM Benchmark with Unrolled Kernel ===\n\n");

    volatile uint64_t freq = rdfreq();
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    // Test 1: Kernel-only benchmark
    printf("\n--- TEST 1: Kernel-only benchmark ---\n");
    {
        size_t A_pack_size = 6 * 256;
        size_t B_pack_size = 64 * 256;

        int8_t* Apack = NULL;
        int8_t* Bpack = NULL;
        int32_t* C = NULL;

        posix_memalign((void**)&Apack, 256, A_pack_size);
        posix_memalign((void**)&Bpack, 256, B_pack_size);
        posix_memalign((void**)&C, 256, 6 * 64 * sizeof(int32_t));

        for (size_t i = 0; i < A_pack_size; i++) Apack[i] = 1;
        for (size_t i = 0; i < B_pack_size; i++) Bpack[i] = 1;
        memset(C, 0, 6 * 64 * sizeof(int32_t));

        // Warmup
        kernel_6x4_unroll(Apack, Bpack, C, 64 * sizeof(int32_t));
        printf("Correctness: C[0]=%d (expected 256)\n", C[0]);

        int iterations = 1000000;
        double ops_per_call = 2.0 * 6 * 64 * 256;

        uint64_t t0 = rdtimer();
        for (int i = 0; i < iterations; i++) {
            kernel_6x4_unroll(Apack, Bpack, C, 64 * sizeof(int32_t));
        }
        uint64_t t1 = rdtimer();

        uint64_t cycles = t1 - t0;
        double time_s = (double)cycles / (double)freq;
        double gops = ops_per_call * iterations / time_s / 1e9;
        double cycles_per_call = (double)cycles / iterations;

        printf("Iterations: %d\n", iterations);
        printf("Cycles per call: %.1f (timer) / %.0f (CPU @ 2GHz)\n",
               cycles_per_call, cycles_per_call * 20);
        printf("GOPS: %.2f\n", gops);
        printf("Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

        free(Apack);
        free(Bpack);
        free(C);
    }

    // Test 2: Tight loop GEMM benchmark (multiple tiles, exact fit)
    printf("\n--- TEST 2: Tight loop GEMM benchmark ---\n");
    {
        int M = 4032, N = 4096, K = 256;  // Exact multiples of tile sizes
        int iterations = 10;

        printf("M=%d (6x%d), N=%d (64x%d), K=%d\n", M, M/6, N, N/64, K);

        int8_t* A = malloc((size_t)M * K);
        int8_t* B = malloc((size_t)N * K);
        int32_t* C = malloc((size_t)M * N * sizeof(int32_t));

        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
        for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;
        memset(C, 0, (size_t)M * N * sizeof(int32_t));

        packed_matrix_t* Apack = pack_A_prepacked(A, K, M, K);
        packed_matrix_t* Bpack = pack_B_prepacked(B, K, N, K);

        // Warmup
        gemm_unroll_tight(Apack, Bpack, C, N);
        printf("Correctness: C[0]=%d (expected 256)\n", C[0]);

        double ops = 2.0 * M * N * K;

        uint64_t t0 = rdtimer();
        for (int i = 0; i < iterations; i++) {
            gemm_unroll_tight(Apack, Bpack, C, N);
        }
        uint64_t t1 = rdtimer();

        uint64_t cycles = t1 - t0;
        double time_s = (double)cycles / (double)freq;
        double gops = ops * iterations / time_s / 1e9;

        int n_kernels = (M / 6) * (N / 64);
        double cycles_per_kernel = (double)cycles / iterations / n_kernels;

        printf("Iterations: %d, kernel calls per iter: %d\n", iterations, n_kernels);
        printf("Time: %.3f ms/call\n", time_s * 1000.0 / iterations);
        printf("Cycles per kernel: %.1f (timer) / %.0f (CPU @ 2GHz)\n",
               cycles_per_kernel, cycles_per_kernel * 20);
        printf("GOPS: %.2f\n", gops);
        printf("Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

        free_packed_matrix(Apack);
        free_packed_matrix(Bpack);
        free(A);
        free(B);
        free(C);
    }

    // Test 3: Larger problem with cache effects
    printf("\n--- TEST 3: Large GEMM (8192x8192x256) ---\n");
    {
        int M = 8190, N = 8192, K = 256;  // N exact, M has edge case
        int iterations = 3;

        printf("M=%d, N=%d (64x%d), K=%d\n", M, N, N/64, K);

        int8_t* A = malloc((size_t)M * K);
        int8_t* B = malloc((size_t)N * K);
        int32_t* C = malloc((size_t)M * N * sizeof(int32_t));

        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
        for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;
        memset(C, 0, (size_t)M * N * sizeof(int32_t));

        packed_matrix_t* Apack = pack_A_prepacked(A, K, M, K);
        packed_matrix_t* Bpack = pack_B_prepacked(B, K, N, K);

        // Warmup
        gemm_prepacked(Apack, Bpack, C, N);
        printf("Correctness: C[0]=%d (expected 256)\n", C[0]);

        double ops = 2.0 * M * N * K;

        uint64_t t0 = rdtimer();
        for (int i = 0; i < iterations; i++) {
            gemm_prepacked(Apack, Bpack, C, N);
        }
        uint64_t t1 = rdtimer();

        uint64_t cycles = t1 - t0;
        double time_s = (double)cycles / (double)freq;
        double gops = ops * iterations / time_s / 1e9;

        printf("Time: %.3f ms/call\n", time_s * 1000.0 / iterations);
        printf("GOPS: %.2f\n", gops);
        printf("Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

        free_packed_matrix(Apack);
        free_packed_matrix(Bpack);
        free(A);
        free(B);
        free(C);
    }

    // Analysis
    printf("\n=== Analysis ===\n");
    printf("Theoretical peak: 512 GOPS\n");
    printf("Target 95%%: 486 GOPS\n");
    printf("Target 90%%: 461 GOPS\n");
    printf("Target 85%%: 435 GOPS\n");
    printf("\n");
    printf("Single kernel: 196,608 ops / 768 min cycles = 256 ops/cycle\n");
    printf("A64FX 2 SDOT pipes × 16 lanes × 4 ops × 2 GHz = 256 ops/cycle = 512 GOPS\n");

    return 0;
}
