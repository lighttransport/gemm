// bench_prepacked.c - Benchmark pre-packed GEMM
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_prepacked.h"

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

int main() {
    printf("=== Pre-packed GEMM Benchmark ===\n\n");

    uint64_t freq = timer_freq();
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    int M = 4096, N = 4096, K = 256;

    // Allocate original matrices
    int8_t* A = malloc((size_t)M * K);
    int8_t* B = malloc((size_t)N * K);
    int32_t* C = malloc((size_t)M * N * sizeof(int32_t));

    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 1;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = 1;

    printf("M=%d, N=%d, K=%d\n\n", M, N, K);

    // Pre-pack both matrices
    printf("Pre-packing A... ");
    fflush(stdout);
    uint64_t t0 = timer_read();
    packed_matrix_t* Apack = pack_A_prepacked(A, K, M, K);
    uint64_t t1 = timer_read();
    double pack_a_time = (double)(t1 - t0) / freq * 1000.0;
    printf("done (%.2f ms)\n", pack_a_time);

    printf("Pre-packing B... ");
    fflush(stdout);
    t0 = timer_read();
    packed_matrix_t* Bpack = pack_B_prepacked(B, K, N, K);
    t1 = timer_read();
    double pack_b_time = (double)(t1 - t0) / freq * 1000.0;
    printf("done (%.2f ms)\n", pack_b_time);

    // Warmup
    memset(C, 0, (size_t)M * N * sizeof(int32_t));
    gemm_prepacked(Apack, Bpack, C, N);
    printf("Warmup result C[0]=%d (expected 256)\n\n", C[0]);

    // Benchmark with pre-packed matrices
    int iterations = 10;
    printf("Running %d iterations with pre-packed data...\n", iterations);

    t0 = timer_read();
    for (int i = 0; i < iterations; i++) {
        gemm_prepacked(Apack, Bpack, C, N);
    }
    t1 = timer_read();

    double time_s = (double)(t1 - t0) / freq;
    double ops = 2.0 * M * N * K * iterations;
    double gops = ops / time_s / 1e9;
    double ms_per_call = time_s * 1000.0 / iterations;

    printf("\nPre-packed GEMM results:\n");
    printf("  Time: %.3f ms/call\n", ms_per_call);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

    // Compare to overhead from packing
    printf("\nOverhead analysis:\n");
    printf("  Pack A time: %.2f ms\n", pack_a_time);
    printf("  Pack B time: %.2f ms\n", pack_b_time);
    printf("  Total pack time: %.2f ms\n", pack_a_time + pack_b_time);
    printf("  GEMM time: %.2f ms\n", ms_per_call);
    printf("  If B pre-packed, amortized time: %.2f ms\n",
           ms_per_call + pack_a_time);

    // Clean up
    free_packed_matrix(Apack);
    free_packed_matrix(Bpack);
    free(A);
    free(B);
    free(C);

    return 0;
}
