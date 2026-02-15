// bench_v3.c - Minimal benchmark with detailed diagnostics
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_pack_opt.h"

int main() {
    printf("=== INT8 SDOT GEMM Benchmark V3 ===\n\n");

    // Get timer info upfront and make it volatile
    volatile uint64_t freq = 0;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    // Test sizes
    int M = 4096, N = 4096, K = 256;

    // Allocate
    int8_t* A = (int8_t*)malloc((size_t)M * K);
    int8_t* B = (int8_t*)malloc((size_t)N * K);
    int32_t* C = (int32_t*)malloc((size_t)M * N * sizeof(int32_t));

    if (!A || !B || !C) {
        printf("Allocation failed!\n");
        return 1;
    }

    // Initialize
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    printf("Allocated M=%d, N=%d, K=%d\n", M, N, K);
    printf("freq before warmup: %lu\n", (unsigned long)freq);

    // Warmup
    gemm_opt_driver(A, K, B, K, C, N, M, N, K);
    printf("freq after warmup: %lu\n", (unsigned long)freq);

    // Simple checksum to prevent optimization
    int64_t checksum = 0;
    for (int i = 0; i < 10; i++) {
        checksum += C[i * M * N / 10];
    }
    printf("Warmup checksum: %ld\n", checksum);

    // Get timer values
    uint64_t t0, t1;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t0));

    gemm_opt_driver(A, K, B, K, C, N, M, N, K);

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t1));

    printf("\nTiming results:\n");
    printf("  t0 = %lu\n", (unsigned long)t0);
    printf("  t1 = %lu\n", (unsigned long)t1);
    printf("  cycles = %lu\n", (unsigned long)(t1 - t0));
    printf("  freq = %lu\n", (unsigned long)freq);

    // Compute stats manually
    double cycles_d = (double)(t1 - t0);
    double freq_d = (double)freq;
    double time_s = cycles_d / freq_d;
    double ops = 2.0 * M * N * K;
    double gops = ops / time_s / 1e9;

    printf("\nManual calculation:\n");
    printf("  cycles_d = %.0f\n", cycles_d);
    printf("  freq_d = %.0f\n", freq_d);
    printf("  time_s = %.6f seconds\n", time_s);
    printf("  ops = %.0f\n", ops);
    printf("  gops = %.2f\n", gops);
    printf("  efficiency = %.1f%% of 512 GOPS\n", 100.0 * gops / 512.0);

    // Checksum after
    checksum = 0;
    for (int i = 0; i < 10; i++) {
        checksum += C[i * M * N / 10];
    }
    printf("\nResult checksum: %ld (expected: %ld)\n", checksum, (int64_t)10 * 256);

    free(A);
    free(B);
    free(C);

    return 0;
}
