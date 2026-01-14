#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "gemm_driver.h"

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

int main() {
    int M = 4096, N = 4096, K = 256;

    printf("Allocating matrices M=%d, N=%d, K=%d...\n", M, N, K);
    int8_t* A = (int8_t*)malloc(M * K);
    int8_t* B = (int8_t*)malloc(N * K);
    int32_t* C = (int32_t*)malloc(M * N * sizeof(int32_t));

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize
    printf("Initializing matrices...\n");
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 21) - 10;
    for (int i = 0; i < N * K; i++) B[i] = (rand() % 21) - 10;
    for (int i = 0; i < M * N; i++) C[i] = 0;

    uint64_t freq = get_timer_freq();
    printf("Timer frequency: %lu Hz\n\n", freq);

    // Test baseline 6x4 kernel
    printf("=== Baseline 6x4 Kernel ===\n");
    printf("Warmup...\n");
    gemm_6x4_driver(A, K, B, K, C, N, M, N, K);
    gemm_6x4_driver(A, K, B, K, C, N, M, N, K);

    printf("Benchmarking (5 iterations)...\n");
    uint64_t t0 = read_timer();
    for (int i = 0; i < 5; i++) {
        gemm_6x4_driver(A, K, B, K, C, N, M, N, K);
    }
    uint64_t t1 = read_timer();

    uint64_t cycles = t1 - t0;
    double elapsed = (double)cycles / freq / 5.0;
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double gops = ops / elapsed / 1e9;

    printf("Cycles: %lu\n", cycles);
    printf("Elapsed: %.6f seconds\n", elapsed);
    printf("GOPS: %.2f\n", gops);
    printf("Efficiency: %.1f%% (of 512 GOPS peak)\n\n", 100.0 * gops / 512.0);

    // Test optimized 6x4 kernel
    printf("=== Optimized 6x4 Kernel (K-loop unrolled 8×) ===\n");
    printf("Warmup...\n");
    for (int i = 0; i < M * N; i++) C[i] = 0;
    gemm_6x4_opt_driver(A, K, B, K, C, N, M, N, K);
    gemm_6x4_opt_driver(A, K, B, K, C, N, M, N, K);

    printf("Benchmarking (5 iterations)...\n");
    t0 = read_timer();
    for (int i = 0; i < 5; i++) {
        gemm_6x4_opt_driver(A, K, B, K, C, N, M, N, K);
    }
    t1 = read_timer();

    cycles = t1 - t0;
    elapsed = (double)cycles / freq / 5.0;
    gops = ops / elapsed / 1e9;

    printf("Cycles: %lu\n", cycles);
    printf("Elapsed: %.6f seconds\n", elapsed);
    printf("GOPS: %.2f\n", gops);
    printf("Efficiency: %.1f%% (of 512 GOPS peak)\n", 100.0 * gops / 512.0);

    if (gops >= 486.0) {
        printf("\n*** TARGET ACHIEVED! >= 95%% efficiency ***\n");
    } else {
        printf("\nTarget: 486 GOPS (95%%), Gap: %.1f GOPS\n", 486.0 - gops);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
