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

// Naive reference
void gemm_ref(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * K + k] * (int32_t)B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

void bench_kernel(const char* name,
                  void (*gemm_func)(const int8_t*, int, const int8_t*, int,
                                   int32_t*, int, int, int, int),
                  int M, int N, int K, int warmup, int iterations) {
    int8_t* A = (int8_t*)malloc(M * K);
    int8_t* B = (int8_t*)malloc(N * K);
    int32_t* C = (int32_t*)malloc(M * N * sizeof(int32_t));

    // Initialize with random values
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 21) - 10;
    for (int i = 0; i < N * K; i++) B[i] = (rand() % 21) - 10;
    for (int i = 0; i < M * N; i++) C[i] = 0;

    uint64_t freq = get_timer_freq();

    // Warmup
    for (int i = 0; i < warmup; i++) {
        gemm_func(A, K, B, K, C, N, M, N, K);
    }

    // Benchmark
    uint64_t t0 = read_timer();
    for (int i = 0; i < iterations; i++) {
        gemm_func(A, K, B, K, C, N, M, N, K);
    }
    uint64_t t1 = read_timer();

    // Calculate performance
    double elapsed = (double)(t1 - t0) / freq / iterations;
    double ops = 2.0 * M * N * K;  // MACs
    double gops = ops / elapsed / 1e9;

    // A64FX INT8 SDOT peak: 512 GOPS (256 ops/cycle × 2 GHz)
    double peak_gops = 512.0;
    double efficiency = 100.0 * gops / peak_gops;

    printf("%-15s M=%5d N=%5d K=%d: %7.2f GOPS (%5.1f%% peak) [%.3f ms]\n",
           name, M, N, K, gops, efficiency, elapsed * 1000.0);

    free(A);
    free(B);
    free(C);
}

int main() {
    printf("========================================\n");
    printf("INT8 GEMM Optimized Kernel Benchmark\n");
    printf("Target: 95%% of 512 GOPS peak = 486+ GOPS\n");
    printf("========================================\n\n");

    int K = 256;
    int warmup = 3;
    int iterations = 20;

    printf("Correctness test (M=100, N=512, K=256):\n");
    {
        int M = 100, N = 512;
        int8_t* A = (int8_t*)malloc(M * K);
        int8_t* B = (int8_t*)malloc(N * K);
        int32_t* C_ref = (int32_t*)calloc(M * N, sizeof(int32_t));
        int32_t* C_opt = (int32_t*)calloc(M * N, sizeof(int32_t));

        srand(42);
        for (int i = 0; i < M * K; i++) A[i] = (rand() % 21) - 10;
        for (int i = 0; i < N * K; i++) B[i] = (rand() % 21) - 10;

        gemm_ref(A, B, C_ref, M, N, K);
        gemm_6x4_opt_driver(A, K, B, K, C_opt, N, M, N, K);

        int errors = 0;
        for (int i = 0; i < M * N; i++) {
            if (C_ref[i] != C_opt[i]) {
                if (errors < 5) {
                    printf("ERROR [%d]: ref=%d, opt=%d\n", i, C_ref[i], C_opt[i]);
                }
                errors++;
            }
        }

        if (errors == 0) {
            printf("  PASS: All %d elements match!\n\n", M * N);
        } else {
            printf("  FAIL: %d / %d elements wrong\n\n", errors, M * N);
            free(A); free(B); free(C_ref); free(C_opt);
            return 1;
        }

        free(A); free(B); free(C_ref); free(C_opt);
    }

    printf("Performance benchmarks:\n");
    printf("%-15s %6s %6s %4s    %s\n", "Kernel", "M", "N", "K", "Performance");
    printf("----------------------------------------------------------------------\n");

    // Comparison: baseline vs optimized
    int test_sizes[][2] = {
        {512, 1024},
        {1024, 2048},
        {2048, 4096},
        {4096, 4096},
        {4096, 8192},
        {8192, 8192}
    };

    for (int i = 0; i < sizeof(test_sizes) / sizeof(test_sizes[0]); i++) {
        int M = test_sizes[i][0];
        int N = test_sizes[i][1];

        // Baseline 6x4 kernel (no unrolling)
        bench_kernel("6x4", gemm_6x4_driver, M, N, K, warmup, iterations);

        // Optimized 6x4 kernel (K-loop unrolled 8×)
        bench_kernel("6x4_opt", gemm_6x4_opt_driver, M, N, K, warmup, iterations);

        printf("\n");
    }

    return 0;
}
