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
    int M = 512, N = 1024, K = 256;

    int8_t* A = (int8_t*)malloc(M * K);
    int8_t* B = (int8_t*)malloc(N * K);
    int32_t* C = (int32_t*)malloc(M * N * sizeof(int32_t));

    // Initialize with ones
    for (int i = 0; i < M * K; i++) A[i] = 1;
    for (int i = 0; i < N * K; i++) B[i] = 1;
    for (int i = 0; i < M * N; i++) C[i] = 0;

    uint64_t freq = get_timer_freq();
    printf("Timer frequency: %lu Hz\n", freq);

    // Warmup
    gemm_5x4_driver(A, K, B, K, C, N, M, N, K);

    // Benchmark
    uint64_t t0 = read_timer();
    gemm_5x4_driver(A, K, B, K, C, N, M, N, K);
    uint64_t t1 = read_timer();

    uint64_t cycles = t1 - t0;
    double elapsed = (double)cycles / freq;
    double ops = 2.0 * M * N * K;
    double gops = ops / elapsed / 1e9;

    printf("M=%d N=%d K=%d\n", M, N, K);
    printf("Cycles: %lu\n", cycles);
    printf("Elapsed: %.6f seconds\n", elapsed);
    printf("Operations: %.0f\n", ops);
    printf("GOPS: %.2f\n", gops);
    printf("Peak efficiency: %.1f%%\n", 100.0 * gops / 8.0);

    // Verify result
    printf("\nSample outputs (expected 256 each):\n");
    for (int i = 0; i < 10; i++) {
        printf("  C[0,%d] = %d\n", i, C[i]);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
