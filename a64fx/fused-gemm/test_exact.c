// Exact replication of test_single_gemm
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

static void gemm_reference(
    int M, int K, int N,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = sum;
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
    float* C_ref = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    printf("Allocated: A=%p B=%p C=%p C_ref=%p\n", A, B, C, C_ref);

    init_random(A, M, K, K);
    init_random(B, K, N, N);
    memset(C, 0, (size_t)M * N * sizeof(float));
    memset(C_ref, 0, (size_t)M * N * sizeof(float));

    printf("Computing reference...\n");
    gemm_reference(M, K, N, A, K, B, N, C_ref, N);

    printf("Computing optimized...\n");
    gemm_fp32(M, K, N, A, K, B, N, C, N);

    printf("Checking...\n");
    int errors = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float got = C[m * N + n];
            float ref = C_ref[m * N + n];
            float err = fabsf(got - ref) / (fabsf(ref) + 1e-6f);
            if (err > 1e-4f) errors++;
        }
    }
    printf("Errors: %d\n", errors);

    printf("Freeing A...\n");
    free(A);
    printf("Freeing B...\n");
    free(B);
    printf("Freeing C...\n");
    free(C);
    printf("Freeing C_ref...\n");
    free(C_ref);

    printf("Done!\n");
    return 0;
}
