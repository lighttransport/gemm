// test_pipe.c - Diagnostic test for pipelined kernel
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_pack_opt.h"

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

// Reference
static void gemm_ref(const int8_t* A, int lda, const int8_t* B, int ldb,
                     int32_t* C, int ldc, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * lda + k] * (int32_t)B[n * ldb + k];
            }
            C[m * ldc + n] = sum;
        }
    }
}

int main() {
    printf("=== Pipelined Kernel Diagnostic Test ===\n\n");

    int M = 6, N = 64, K = 256;

    int8_t* A = (int8_t*)malloc(M * K);
    int8_t* B = (int8_t*)malloc(N * K);
    int32_t* C_ref = (int32_t*)malloc(M * N * sizeof(int32_t));
    int32_t* C_ultra = (int32_t*)malloc(M * N * sizeof(int32_t));
    int32_t* C_pipe = (int32_t*)malloc(M * N * sizeof(int32_t));

    // Initialize
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 7) - 3;
    for (int i = 0; i < N * K; i++) B[i] = (rand() % 7) - 3;
    memset(C_ref, 0, M * N * sizeof(int32_t));
    memset(C_ultra, 0, M * N * sizeof(int32_t));
    memset(C_pipe, 0, M * N * sizeof(int32_t));

    printf("Testing single 6x64 tile...\n");

    // Reference
    printf("  Computing reference... ");
    fflush(stdout);
    gemm_ref(A, K, B, K, C_ref, N, M, N, K);
    printf("done\n");

    // Ultra kernel
    printf("  Computing ultra kernel... ");
    fflush(stdout);
    gemm_opt_driver(A, K, B, K, C_ultra, N, M, N, K);
    printf("done\n");

    // Pipe kernel
    printf("  Computing pipe kernel... ");
    fflush(stdout);
    gemm_pipe_driver(A, K, B, K, C_pipe, N, M, N, K);
    printf("done\n");

    // Compare
    int ultra_errors = 0, pipe_errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (C_ref[i] != C_ultra[i]) ultra_errors++;
        if (C_ref[i] != C_pipe[i]) pipe_errors++;
    }

    printf("\nCorrectness:\n");
    printf("  Ultra errors: %d\n", ultra_errors);
    printf("  Pipe errors:  %d\n", pipe_errors);

    if (pipe_errors > 0) {
        printf("\nFirst 10 pipe mismatches:\n");
        int shown = 0;
        for (int i = 0; i < M * N && shown < 10; i++) {
            if (C_ref[i] != C_pipe[i]) {
                int m = i / N, n = i % N;
                printf("  [%d,%d]: ref=%d, pipe=%d\n", m, n, C_ref[i], C_pipe[i]);
                shown++;
            }
        }
    }

    // Performance
    printf("\nPerformance test (100 calls each)...\n");
    uint64_t freq = get_timer_freq();

    // Ultra
    uint64_t t0 = read_timer();
    for (int i = 0; i < 100; i++) {
        gemm_opt_driver(A, K, B, K, C_ultra, N, M, N, K);
    }
    uint64_t t1 = read_timer();
    double time_ultra = (double)(t1 - t0) / freq * 1000.0 / 100.0;

    // Pipe
    t0 = read_timer();
    for (int i = 0; i < 100; i++) {
        gemm_pipe_driver(A, K, B, K, C_pipe, N, M, N, K);
    }
    t1 = read_timer();
    double time_pipe = (double)(t1 - t0) / freq * 1000.0 / 100.0;

    printf("  Ultra: %.4f ms/call\n", time_ultra);
    printf("  Pipe:  %.4f ms/call\n", time_pipe);

    // Larger test
    printf("\nTesting 4096x4096...\n");
    M = 4096; N = 4096; K = 256;

    free(A); free(B); free(C_ref); free(C_ultra); free(C_pipe);
    A = (int8_t*)malloc((size_t)M * K);
    B = (int8_t*)malloc((size_t)N * K);
    C_ultra = (int32_t*)malloc((size_t)M * N * sizeof(int32_t));
    C_pipe = (int32_t*)malloc((size_t)M * N * sizeof(int32_t));

    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;

    printf("  Warming up ultra... ");
    fflush(stdout);
    memset(C_ultra, 0, (size_t)M * N * sizeof(int32_t));
    gemm_opt_driver(A, K, B, K, C_ultra, N, M, N, K);
    printf("done\n");

    printf("  Warming up pipe... ");
    fflush(stdout);
    memset(C_pipe, 0, (size_t)M * N * sizeof(int32_t));
    gemm_pipe_driver(A, K, B, K, C_pipe, N, M, N, K);
    printf("done\n");

    // Benchmark
    t0 = read_timer();
    gemm_opt_driver(A, K, B, K, C_ultra, N, M, N, K);
    t1 = read_timer();
    double time_ultra_large = (double)(t1 - t0) / freq;
    double gops_ultra = 2.0 * M * N * K / time_ultra_large / 1e9;

    t0 = read_timer();
    gemm_pipe_driver(A, K, B, K, C_pipe, N, M, N, K);
    t1 = read_timer();
    double time_pipe_large = (double)(t1 - t0) / freq;
    double gops_pipe = 2.0 * M * N * K / time_pipe_large / 1e9;

    printf("\n4096x4096x256 Performance:\n");
    printf("  Ultra: %.2f GOPS (%.1f%% peak), %.3f ms\n",
           gops_ultra, 100*gops_ultra/512, time_ultra_large*1000);
    printf("  Pipe:  %.2f GOPS (%.1f%% peak), %.3f ms\n",
           gops_pipe, 100*gops_pipe/512, time_pipe_large*1000);

    free(A); free(B); free(C_ultra); free(C_pipe);
    return 0;
}
