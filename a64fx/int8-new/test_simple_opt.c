#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "gemm_driver.h"

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

    int8_t* A = (int8_t*)malloc((size_t)M * K);
    int8_t* B = (int8_t*)malloc((size_t)N * K);
    int32_t* C = (int32_t*)malloc((size_t)M * N * sizeof(int32_t));

    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = 0;

    uint64_t freq = get_timer_freq();

    // Baseline
    gemm_6x4_driver(A, K, B, K, C, N, M, N, K);
    uint64_t t0 = read_timer();
    gemm_6x4_driver(A, K, B, K, C, N, M, N, K);
    uint64_t t1 = read_timer();
    uint64_t cycles_base = t1 - t0;

    // Optimized
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = 0;
    gemm_6x4_opt_driver(A, K, B, K, C, N, M, N, K);
    t0 = read_timer();
    gemm_6x4_opt_driver(A, K, B, K, C, N, M, N, K);
    t1 = read_timer();
    uint64_t cycles_opt = t1 - t0;

    double time_base = (double)cycles_base / (double)freq;
    double time_opt = (double)cycles_opt / (double)freq;
    double ops = 2.0 * 4096.0 * 4096.0 * 256.0;
    double gops_base = ops / time_base / 1e9;
    double gops_opt = ops / time_opt / 1e9;

    printf("M=N=%d, K=%d\n", M, K);
    printf("Baseline:  %.2f GOPS (%.1f%% of 512), %.3f ms\n", 
           gops_base, 100.0*gops_base/512.0, time_base*1000.0);
    printf("Optimized: %.2f GOPS (%.1f%% of 512), %.3f ms\n", 
           gops_opt, 100.0*gops_opt/512.0, time_opt*1000.0);
    printf("Speedup: %.2fx\n", gops_opt/gops_base);

    free(A); free(B); free(C);
    return 0;
}
