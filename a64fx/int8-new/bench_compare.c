// bench_compare.c - Compare Ultra vs Pipe kernels
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_pack_opt.h"

static inline uint64_t timer_read(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t timer_freq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

void bench_single(const char* name, int M, int N, int K, int iters,
                  void (*gemm_func)(const int8_t*, int, const int8_t*, int,
                                    int32_t*, int, int, int, int)) {
    int8_t* A = malloc((size_t)M * K);
    int8_t* B = malloc((size_t)N * K);
    int32_t* C = malloc((size_t)M * N * sizeof(int32_t));

    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    // Warmup
    gemm_func(A, K, B, K, C, N, M, N, K);

    uint64_t freq = timer_freq();
    uint64_t t0 = timer_read();
    for (int i = 0; i < iters; i++) {
        gemm_func(A, K, B, K, C, N, M, N, K);
    }
    uint64_t t1 = timer_read();

    double time_s = (double)(t1 - t0) / (double)freq;
    double ops = 2.0 * M * N * K * iters;
    double gops = ops / time_s / 1e9;
    double ms_per_call = time_s * 1000.0 / iters;

    printf("  %s: %.2f GOPS (%.1f%%), %.3f ms/call\n",
           name, gops, 100.0 * gops / 512.0, ms_per_call);

    free(A);
    free(B);
    free(C);
}

int main() {
    printf("=== Ultra vs Pipe Kernel Comparison ===\n");
    printf("Target: 95%% of 512 GOPS = 486 GOPS\n\n");

    // Test various sizes
    printf("M=4096, N=4096, K=256 (10 iters):\n");
    bench_single("Ultra", 4096, 4096, 256, 10, gemm_opt_driver);
    bench_single("Pipe ", 4096, 4096, 256, 10, gemm_pipe_driver);

    printf("\nM=8192, N=8192, K=256 (5 iters):\n");
    bench_single("Ultra", 8192, 8192, 256, 5, gemm_opt_driver);
    bench_single("Pipe ", 8192, 8192, 256, 5, gemm_pipe_driver);

    printf("\nM=2048, N=2048, K=256 (20 iters):\n");
    bench_single("Ultra", 2048, 2048, 256, 20, gemm_opt_driver);
    bench_single("Pipe ", 2048, 2048, 256, 20, gemm_pipe_driver);

    printf("\nM=1024, N=1024, K=256 (50 iters):\n");
    bench_single("Ultra", 1024, 1024, 256, 50, gemm_opt_driver);
    bench_single("Pipe ", 1024, 1024, 256, 50, gemm_pipe_driver);

    return 0;
}
