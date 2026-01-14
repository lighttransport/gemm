// bench_v2.c - Fixed benchmark with optimization prevention
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

// Prevent compiler from optimizing away the result
static volatile int32_t sink = 0;

static void use_result(int32_t* C, size_t size) {
    int32_t sum = 0;
    for (size_t i = 0; i < size; i += 1000) {
        sum += C[i];
    }
    sink = sum;
}

// Reference GEMM
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
    printf("=== INT8 SDOT GEMM Benchmark V2 ===\n");
    printf("Target: 95%% of 512 GOPS peak = 486 GOPS\n\n");

    uint64_t freq = get_timer_freq();
    printf("Timer frequency: %lu Hz\n", freq);

    // Quick correctness test
    printf("\n--- Quick Correctness Test ---\n");
    {
        int M = 6, N = 64, K = 256;
        int8_t* A = malloc(M * K);
        int8_t* B = malloc(N * K);
        int32_t* C_ref = malloc(M * N * sizeof(int32_t));
        int32_t* C_opt = malloc(M * N * sizeof(int32_t));
        int32_t* C_pipe = malloc(M * N * sizeof(int32_t));

        srand(42);
        for (int i = 0; i < M * K; i++) A[i] = (rand() % 7) - 3;
        for (int i = 0; i < N * K; i++) B[i] = (rand() % 7) - 3;

        gemm_ref(A, K, B, K, C_ref, N, M, N, K);
        gemm_opt_driver(A, K, B, K, C_opt, N, M, N, K);
        gemm_pipe_driver(A, K, B, K, C_pipe, N, M, N, K);

        int err_opt = 0, err_pipe = 0;
        for (int i = 0; i < M * N; i++) {
            if (C_ref[i] != C_opt[i]) err_opt++;
            if (C_ref[i] != C_pipe[i]) err_pipe++;
        }

        printf("Ultra kernel: %s (%d errors)\n", err_opt == 0 ? "PASS" : "FAIL", err_opt);
        printf("Pipe kernel:  %s (%d errors)\n", err_pipe == 0 ? "PASS" : "FAIL", err_pipe);

        free(A); free(B); free(C_ref); free(C_opt); free(C_pipe);
    }

    // Benchmark different sizes
    printf("\n--- Performance Benchmarks ---\n");

    int sizes[][2] = {{4096, 4096}, {8192, 8192}};
    int num_sizes = 2;

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s][0];
        int N = sizes[s][1];
        int K = 256;
        int iterations = (M == 4096) ? 10 : 5;

        printf("\nM=%d, N=%d, K=%d (%d iterations):\n", M, N, K, iterations);

        int8_t* A = malloc((size_t)M * K);
        int8_t* B = malloc((size_t)N * K);
        int32_t* C = malloc((size_t)M * N * sizeof(int32_t));

        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
        for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;

        double ops = 2.0 * M * N * K;

        // Ultra kernel benchmark
        memset(C, 0, (size_t)M * N * sizeof(int32_t));
        gemm_opt_driver(A, K, B, K, C, N, M, N, K);  // warmup
        use_result(C, (size_t)M * N);

        uint64_t t0 = read_timer();
        for (int i = 0; i < iterations; i++) {
            gemm_opt_driver(A, K, B, K, C, N, M, N, K);
        }
        uint64_t t1 = read_timer();
        use_result(C, (size_t)M * N);

        uint64_t cycles_ultra = t1 - t0;
        double time_ultra = (double)cycles_ultra / (double)freq;
        double gops_ultra = (ops * iterations) / time_ultra / 1e9;

        printf("  Ultra: cycles=%lu, time=%.3f ms, %.2f GOPS (%.1f%%)\n",
               cycles_ultra, time_ultra * 1000.0 / iterations,
               gops_ultra, 100.0 * gops_ultra / 512.0);

        // Pipe kernel benchmark
        memset(C, 0, (size_t)M * N * sizeof(int32_t));
        gemm_pipe_driver(A, K, B, K, C, N, M, N, K);  // warmup
        use_result(C, (size_t)M * N);

        t0 = read_timer();
        for (int i = 0; i < iterations; i++) {
            gemm_pipe_driver(A, K, B, K, C, N, M, N, K);
        }
        t1 = read_timer();
        use_result(C, (size_t)M * N);

        uint64_t cycles_pipe = t1 - t0;
        double time_pipe = (double)cycles_pipe / (double)freq;
        double gops_pipe = (ops * iterations) / time_pipe / 1e9;

        printf("  Pipe:  cycles=%lu, time=%.3f ms, %.2f GOPS (%.1f%%)\n",
               cycles_pipe, time_pipe * 1000.0 / iterations,
               gops_pipe, 100.0 * gops_pipe / 512.0);

        free(A); free(B); free(C);
    }

    // Batch benchmark (for attention heads)
    printf("\n--- Batch Benchmarks (for attention) ---\n");

    int batch_configs[][3] = {
        {128, 128, 32},   // short seq, 32 heads
        {512, 512, 32},   // medium seq, 32 heads
        {2048, 2048, 8},  // long seq, 8 heads
    };
    int num_batches = 3;

    for (int b = 0; b < num_batches; b++) {
        int M = batch_configs[b][0];
        int N = batch_configs[b][1];
        int batch = batch_configs[b][2];
        int K = 256;
        int iterations = 10;

        printf("\nM=%d, N=%d, K=%d, batch=%d:\n", M, N, K, batch);

        size_t A_size = (size_t)M * K * batch;
        size_t B_size = (size_t)N * K * batch;
        size_t C_size = (size_t)M * N * batch;

        int8_t* A = malloc(A_size);
        int8_t* B = malloc(B_size);
        int32_t* C = malloc(C_size * sizeof(int32_t));

        for (size_t i = 0; i < A_size; i++) A[i] = 1;
        for (size_t i = 0; i < B_size; i++) B[i] = 1;

        int64_t strideA = M * K;
        int64_t strideB = N * K;
        int64_t strideC = M * N;

        double ops = 2.0 * M * N * K * batch;

        // Ultra batch
        memset(C, 0, C_size * sizeof(int32_t));
        gemm_batch_opt(A, K, strideA, B, K, strideB, C, N, strideC, M, N, K, batch);
        use_result(C, C_size);

        uint64_t t0 = read_timer();
        for (int i = 0; i < iterations; i++) {
            gemm_batch_opt(A, K, strideA, B, K, strideB, C, N, strideC, M, N, K, batch);
        }
        uint64_t t1 = read_timer();
        use_result(C, C_size);

        double time_ultra = (double)(t1 - t0) / (double)freq;
        double gops_ultra = (ops * iterations) / time_ultra / 1e9;

        printf("  Ultra: %.2f GOPS (%.1f%%), %.3f ms/batch\n",
               gops_ultra, 100.0 * gops_ultra / 512.0, time_ultra * 1000.0 / iterations);

        // Pipe batch
        memset(C, 0, C_size * sizeof(int32_t));
        gemm_batch_pipe(A, K, strideA, B, K, strideB, C, N, strideC, M, N, K, batch);
        use_result(C, C_size);

        t0 = read_timer();
        for (int i = 0; i < iterations; i++) {
            gemm_batch_pipe(A, K, strideA, B, K, strideB, C, N, strideC, M, N, K, batch);
        }
        t1 = read_timer();
        use_result(C, C_size);

        double time_pipe = (double)(t1 - t0) / (double)freq;
        double gops_pipe = (ops * iterations) / time_pipe / 1e9;

        printf("  Pipe:  %.2f GOPS (%.1f%%), %.3f ms/batch\n",
               gops_pipe, 100.0 * gops_pipe / 512.0, time_pipe * 1000.0 / iterations);

        free(A); free(B); free(C);
    }

    printf("\nDone.\n");
    return 0;
}
