// bench_kernel_only.c
// Benchmark Q@K^T kernels only (no softmax)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "attention_d256_interleaved.h"

static inline uint64_t rdtsc(void) {
    uint64_t t;
    __asm__ volatile("mrs %0, CNTVCT_EL0" : "=r"(t));
    return t;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t f;
    __asm__ volatile("mrs %0, CNTFRQ_EL0" : "=r"(f));
    return f;
}

// Kernel declarations
extern void kernel_qkt_d256_interleaved(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t);
extern void kernel_qkt_d256_4x4(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t);
extern void kernel_qkt_d256_2n(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t);

void interleave_K_chunk(const int8_t* K, int8_t* K_int, int N) {
    for (int d_group = 0; d_group < D_GROUPS; d_group++) {
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < 4; i++) {
                K_int[d_group * N * 4 + n * 4 + i] = K[n * D256 + d_group * 4 + i];
            }
        }
    }
}

int main() {
    printf("================================================================\n");
    printf("Q@K^T Kernel Only Benchmark (D=256)\n");
    printf("================================================================\n\n");

    uint64_t timer_freq = get_timer_freq();

    int M = 1536, N = 64, D = D256;  // Single N-chunk

    // Allocate
    int8_t* Q = aligned_alloc(64, M * D);
    int8_t* K = aligned_alloc(64, N * D);
    int8_t* K_int = aligned_alloc(64, D_GROUPS * N * 4);
    int32_t* S_6row = aligned_alloc(64, 6 * N * sizeof(int32_t));
    int32_t* S_4x4 = aligned_alloc(64, 4 * N * sizeof(int32_t));
    int32_t* S_ref = aligned_alloc(64, 6 * N * sizeof(int32_t));

    // Initialize
    for (int i = 0; i < M * D; i++) Q[i] = (i % 7) - 3;
    for (int i = 0; i < N * D; i++) K[i] = (i % 11) - 5;
    interleave_K_chunk(K, K_int, N);

    // Compute reference for first 6 rows using scalar
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int d = 0; d < D; d++) {
                sum += (int32_t)Q[m * D + d] * (int32_t)K[n * D + d];
            }
            S_ref[m * N + n] = sum;
        }
    }

    // Test 6-row interleaved kernel
    memset(S_6row, 0, 6 * N * sizeof(int32_t));
    kernel_qkt_d256_interleaved(Q, K_int, S_6row, D, N);

    printf("6-row interleaved kernel verification:\n");
    int32_t max_diff_6row = 0;
    for (int i = 0; i < 6 * N; i++) {
        int32_t diff = S_6row[i] - S_ref[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff_6row) max_diff_6row = diff;
    }
    printf("  Max diff from scalar ref: %d\n", max_diff_6row);

    // Test 4x4 kernel
    memset(S_4x4, 0, 4 * N * sizeof(int32_t));
    kernel_qkt_d256_4x4(Q, K_int, S_4x4, D, N);

    printf("\n4x4 kernel verification:\n");
    int32_t max_diff_4x4 = 0;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < N; n++) {
            int32_t diff = S_4x4[m * N + n] - S_ref[m * N + n];
            if (diff < 0) diff = -diff;
            if (diff > max_diff_4x4) max_diff_4x4 = diff;
        }
    }
    printf("  Max diff from scalar ref: %d\n", max_diff_4x4);

    // Debug: print first few values
    printf("\n  Sample values (first 4 elements of row 0):\n");
    printf("    Scalar: %d %d %d %d\n", S_ref[0], S_ref[1], S_ref[2], S_ref[3]);
    printf("    6-row:  %d %d %d %d\n", S_6row[0], S_6row[1], S_6row[2], S_6row[3]);
    printf("    4x4:    %d %d %d %d\n", S_4x4[0], S_4x4[1], S_4x4[2], S_4x4[3]);

    // Benchmark kernels
    printf("\n--- Kernel Performance ---\n");
    int M_tiles_6 = M / 6;
    int M_tiles_4 = M / 4;
    int iters = 100;

    // 6-row kernel
    uint64_t t0 = rdtsc();
    for (int iter = 0; iter < iters; iter++) {
        for (int mt = 0; mt < M_tiles_6; mt++) {
            kernel_qkt_d256_interleaved(Q + mt * 6 * D, K_int, S_6row, D, N);
        }
    }
    uint64_t t1 = rdtsc();
    double cycles_6row = (double)(t1 - t0) / iters / timer_freq * 2.0e9;

    // 4x4 kernel
    uint64_t t2 = rdtsc();
    for (int iter = 0; iter < iters; iter++) {
        for (int mt = 0; mt < M_tiles_4; mt++) {
            kernel_qkt_d256_4x4(Q + mt * 4 * D, K_int, S_4x4, D, N);
        }
    }
    uint64_t t3 = rdtsc();
    double cycles_4x4 = (double)(t3 - t2) / iters / timer_freq * 2.0e9;

    // Compute GFLOPS
    double ops_6row = 2.0 * M_tiles_6 * 6 * N * D;  // 2 ops per MAC (mul + add)
    double ops_4x4 = 2.0 * M_tiles_4 * 4 * N * D;

    printf("\n%-15s %12s %12s %10s\n", "Kernel", "Cycles", "GFLOPS", "Eff%");
    printf("%-15s %12s %12s %10s\n", "---------------", "------------", "------------", "----------");
    printf("%-15s %12.0f %12.1f %9.1f%%\n", "6-row interleave",
           cycles_6row, ops_6row / (cycles_6row / 2e9) / 1e9,
           100.0 * ops_6row / (cycles_6row / 2e9) / 1e9 / 512.0);
    printf("%-15s %12.0f %12.1f %9.1f%%\n", "4x4 tile",
           cycles_4x4, ops_4x4 / (cycles_4x4 / 2e9) / 1e9,
           100.0 * ops_4x4 / (cycles_4x4 / 2e9) / 1e9 / 512.0);

    free(Q); free(K); free(K_int);
    free(S_6row); free(S_4x4); free(S_ref);

    printf("\n================================================================\n");
    return 0;
}
