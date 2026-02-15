// Test multiple GEMM calls (like benchmark warmup+iterations)
#include "fused_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void init_random(float* M, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M[i * ld + j] = (float)(rand() % 1000 - 500) / 500.0f;
        }
    }
}

int main() {
    int M = 64;
    int K = 64;
    int N = 48;

    srand(42);

    printf("Test: M=%d, K=%d, N=%d\n", M, K, N);

    float* A = (float*)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float* B = (float*)aligned_alloc(64, (size_t)K * N * sizeof(float));
    float* C = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    printf("Allocated: A=%p B=%p C=%p\n", A, B, C);

    init_random(A, M, K, K);
    init_random(B, K, N, N);

    // Multiple calls like benchmark
    int warmup = 3;
    int iters = 10;

    printf("Warmup (%d iterations)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        memset(C, 0, (size_t)M * N * sizeof(float));
        gemm_fp32(M, K, N, A, K, B, N, C, N);
        printf("  Iteration %d: C[0]=%f\n", i, C[0]);
    }

    printf("Benchmark (%d iterations)...\n", iters);
    for (int i = 0; i < iters; i++) {
        memset(C, 0, (size_t)M * N * sizeof(float));
        gemm_fp32(M, K, N, A, K, B, N, C, N);
        printf("  Iteration %d: C[0]=%f\n", i, C[0]);
    }

    printf("Freeing...\n");
    free(A);
    free(B);
    free(C);

    printf("Done!\n");
    return 0;
}
