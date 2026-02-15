// Tiled test for debugging
#include "fused_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Test with multiple tiles: 16x64 @ 64x96 = 16x96
    // This gives: 2 M-blocks, 2 N-blocks
    int M = 16;
    int K = 64;
    int N = 96;

    printf("Test dimensions: M=%d, K=%d, N=%d\n", M, K, N);
    printf("Tiles: M_blocks=%d, N_blocks=%d\n", (M + MR - 1) / MR, (N + NR - 1) / NR);

    // Allocate matrices
    float* A = (float*)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float* B = (float*)aligned_alloc(64, (size_t)K * N * sizeof(float));
    float* C = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));
    float* C_ref = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    printf("A=%p, B=%p, C=%p\n", A, B, C);

    // Initialize with simple values
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            A[m * K + k] = 1.0f;
        }
    }
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            B[k * N + n] = 1.0f;
        }
    }

    // Reference GEMM
    printf("Computing reference...\n");
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C_ref[m * N + n] = sum;
        }
    }
    printf("Reference C[0,0]=%f (expected %f)\n", C_ref[0], (float)K);

    // Test optimized GEMM
    printf("Computing optimized...\n");
    memset(C, 0, (size_t)M * N * sizeof(float));
    gemm_fp32(M, K, N, A, K, B, N, C, N);

    printf("Optimized C[0,0]=%f (expected %f)\n", C[0], (float)K);

    // Check result
    int errors = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float expected = C_ref[m * N + n];
            float got = C[m * N + n];
            float err = got - expected;
            if (err < -0.001f || err > 0.001f) {
                if (errors < 10) {
                    printf("Error at [%d,%d]: got %f, expected %f\n",
                           m, n, got, expected);
                }
                errors++;
            }
        }
    }
    printf("Errors: %d\n", errors);

    printf("Freeing matrices...\n");
    free(A);
    free(B);
    free(C);
    free(C_ref);

    printf("Test complete!\n");
    return errors > 0 ? 1 : 0;
}
