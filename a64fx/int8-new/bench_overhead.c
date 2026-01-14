// bench_overhead.c - Analyze GEMM overhead sources
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_prepacked.h"

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

// Version 1: Function call in tight loop
static void gemm_v1_tight(const int8_t* Apack, int A_stride,
                           const int8_t* Bpack, int B_stride,
                           int32_t* C, int ldc_bytes,
                           int m_tiles, int n_tiles) {
    for (int mt = 0; mt < m_tiles; mt++) {
        for (int nt = 0; nt < n_tiles; nt++) {
            kernel_6x4_unroll(Apack, Bpack + nt * B_stride,
                              C + nt * 64, ldc_bytes);
        }
        Apack += A_stride;
        C += 6 * (ldc_bytes / 4);  // ldc_bytes is in bytes, C is int32_t*
    }
}

// Version 2: Unroll inner loop 4x
static void gemm_v2_unroll4(const int8_t* Apack, int A_stride,
                             const int8_t* Bpack, int B_stride,
                             int32_t* C, int ldc_bytes,
                             int m_tiles, int n_tiles) {
    int ldc = ldc_bytes / 4;
    for (int mt = 0; mt < m_tiles; mt++) {
        int32_t* Crow = C + mt * 6 * ldc;
        const int8_t* Bptr = Bpack;

        int nt = 0;
        for (; nt + 4 <= n_tiles; nt += 4) {
            kernel_6x4_unroll(Apack, Bptr + 0 * B_stride, Crow + 0 * 64, ldc_bytes);
            kernel_6x4_unroll(Apack, Bptr + 1 * B_stride, Crow + 1 * 64, ldc_bytes);
            kernel_6x4_unroll(Apack, Bptr + 2 * B_stride, Crow + 2 * 64, ldc_bytes);
            kernel_6x4_unroll(Apack, Bptr + 3 * B_stride, Crow + 3 * 64, ldc_bytes);
            Bptr += 4 * B_stride;
            Crow += 4 * 64;
        }
        for (; nt < n_tiles; nt++) {
            kernel_6x4_unroll(Apack, Bptr, Crow, ldc_bytes);
            Bptr += B_stride;
            Crow += 64;
        }

        Apack += A_stride;
    }
}

int main() {
    printf("=== GEMM Overhead Analysis ===\n\n");

    volatile uint64_t freq = rdfreq();
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    // Problem size: 4032x4096x256 (exact multiples)
    int M = 4032, N = 4096, K = 256;
    int m_tiles = M / 6;    // 672
    int n_tiles = N / 64;   // 64
    int total_kernels = m_tiles * n_tiles;  // 43008

    printf("M=%d (%d tiles), N=%d (%d tiles), K=%d\n", M, m_tiles, N, n_tiles, K);
    printf("Total kernel calls per GEMM: %d\n\n", total_kernels);

    // Allocate and initialize
    int8_t* A = malloc((size_t)M * K);
    int8_t* B = malloc((size_t)N * K);
    int32_t* C = malloc((size_t)M * N * sizeof(int32_t));

    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    packed_matrix_t* Apack = pack_A_prepacked(A, K, M, K);
    packed_matrix_t* Bpack = pack_B_prepacked(B, K, N, K);

    int A_stride = 6 * K;
    int B_stride = 64 * K;
    int ldc_bytes = N * sizeof(int32_t);

    // Warmup
    gemm_v1_tight(Apack->data, A_stride, Bpack->data, B_stride, C, ldc_bytes, m_tiles, n_tiles);
    printf("Warmup: C[0]=%d (expected 256)\n\n", C[0]);

    // Operations
    double ops_per_gemm = 2.0 * (double)M * (double)N * (double)K;
    double ops_per_kernel = 2.0 * 6.0 * 64.0 * 256.0;
    int iterations = 5;

    // Test 1: Baseline tight loop
    printf("=== V1: Baseline tight loop ===\n");
    {
        memset(C, 0, (size_t)M * N * sizeof(int32_t));
        uint64_t t0 = rdtimer();
        for (int i = 0; i < iterations; i++) {
            gemm_v1_tight(Apack->data, A_stride, Bpack->data, B_stride, C, ldc_bytes, m_tiles, n_tiles);
        }
        uint64_t t1 = rdtimer();

        double cycles = (double)(t1 - t0);
        double time_s = cycles / (double)freq;
        double cycles_per_kernel = cycles / iterations / total_kernels;
        double gops = ops_per_gemm * iterations / time_s / 1e9;

        printf("  Time: %.3f ms/GEMM\n", time_s * 1000.0 / iterations);
        printf("  Cycles/kernel: %.1f timer (%.0f CPU @ 2GHz)\n",
               cycles_per_kernel, cycles_per_kernel * 20);
        printf("  GOPS: %.2f (%.1f%% of 512 peak)\n", gops, 100.0 * gops / 512.0);
    }

    // Test 2: Unrolled inner loop
    printf("\n=== V2: Inner loop unrolled 4x ===\n");
    {
        memset(C, 0, (size_t)M * N * sizeof(int32_t));
        uint64_t t0 = rdtimer();
        for (int i = 0; i < iterations; i++) {
            gemm_v2_unroll4(Apack->data, A_stride, Bpack->data, B_stride, C, ldc_bytes, m_tiles, n_tiles);
        }
        uint64_t t1 = rdtimer();

        double cycles = (double)(t1 - t0);
        double time_s = cycles / (double)freq;
        double cycles_per_kernel = cycles / iterations / total_kernels;
        double gops = ops_per_gemm * iterations / time_s / 1e9;

        printf("  Time: %.3f ms/GEMM\n", time_s * 1000.0 / iterations);
        printf("  Cycles/kernel: %.1f timer (%.0f CPU @ 2GHz)\n",
               cycles_per_kernel, cycles_per_kernel * 20);
        printf("  GOPS: %.2f (%.1f%% of 512 peak)\n", gops, 100.0 * gops / 512.0);
    }

    // Kernel-only baseline
    printf("\n=== Kernel-only baseline ===\n");
    {
        int8_t* Aptr = Apack->data;
        int8_t* Bptr = Bpack->data;
        int32_t* Cptr = C;

        memset(C, 0, (size_t)M * N * sizeof(int32_t));
        int kernel_iters = 100000;
        uint64_t t0 = rdtimer();
        for (int i = 0; i < kernel_iters; i++) {
            kernel_6x4_unroll(Aptr, Bptr, Cptr, ldc_bytes);
        }
        uint64_t t1 = rdtimer();

        double cycles = (double)(t1 - t0);
        double time_s = cycles / (double)freq;
        double cycles_per_kernel = cycles / kernel_iters;
        double gops = ops_per_kernel * kernel_iters / time_s / 1e9;

        printf("  Cycles/kernel: %.1f timer (%.0f CPU @ 2GHz)\n",
               cycles_per_kernel, cycles_per_kernel * 20);
        printf("  GOPS: %.2f (%.1f%% of 512 peak)\n", gops, 100.0 * gops / 512.0);
    }

    // Analysis
    printf("\n=== Analysis ===\n");
    printf("Theoretical minimum: 768 CPU cycles (1536 SDOTs / 2 pipes)\n");
    printf("Target 95%%: 808 CPU cycles\n");
    printf("Target 90%%: 853 CPU cycles\n");
    printf("Isolated kernel: ~869 CPU cycles (88.4%%)\n");
    printf("\n");
    printf("Overhead sources:\n");
    printf("  - Function call: ~50 cycles (save/restore regs)\n");
    printf("  - Loop control: ~10 cycles (increment, compare, branch)\n");
    printf("  - Memory latency: variable (depends on cache state)\n");

    free_packed_matrix(Apack);
    free_packed_matrix(Bpack);
    free(A);
    free(B);
    free(C);

    return 0;
}
