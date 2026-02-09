// Minimal benchmark test
#include "fused_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    // Test with first benchmark size: M=64, K=64, N=48
    int M = 64;
    int K = 64;
    int N = 48;

    printf("Test: M=%d, K=%d, N=%d\n", M, K, N);
    printf("MR=%d, NR=%d\n", MR, NR);
    printf("M_blocks=%d, N_blocks=%d\n", (M + MR - 1) / MR, (N + NR - 1) / NR);

    size_t A_bytes = (size_t)M * K * sizeof(float);
    size_t B_bytes = (size_t)K * N * sizeof(float);
    size_t C_bytes = (size_t)M * N * sizeof(float);

    printf("Allocating: A=%zu, B=%zu, C=%zu bytes\n", A_bytes, B_bytes, C_bytes);

    float* A = (float*)aligned_alloc(64, A_bytes);
    float* B = (float*)aligned_alloc(64, B_bytes);
    float* C = (float*)aligned_alloc(64, C_bytes);
    float* C_ref = (float*)aligned_alloc(64, C_bytes);

    printf("A=%p, B=%p, C=%p, C_ref=%p\n", A, B, C, C_ref);

    if (!A || !B || !C || !C_ref) {
        printf("Allocation failed!\n");
        return 1;
    }

    // Initialize
    printf("Initializing...\n");
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 10) / 10.0f;
    memset(C, 0, C_bytes);
    memset(C_ref, 0, C_bytes);

    // Reference
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
    printf("Reference C_ref[0]=%f, C_ref[last]=%f\n", C_ref[0], C_ref[M*N-1]);

    // Optimized
    printf("Computing optimized...\n");
    gemm_fp32(M, K, N, A, K, B, N, C, N);
    printf("Optimized C[0]=%f, C[last]=%f\n", C[0], C[M*N-1]);

    // Check
    printf("Checking...\n");
    int errors = 0;
    float max_err = 0.0f;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float got = C[m * N + n];
            float ref = C_ref[m * N + n];
            float err = fabsf(got - ref);
            if (err > max_err) max_err = err;
            if (err > 0.01f) {
                if (errors < 5) {
                    printf("  Error at [%d,%d]: got=%f, ref=%f, err=%f\n",
                           m, n, got, ref, err);
                }
                errors++;
            }
        }
    }
    printf("Max error: %e, Errors: %d\n", max_err, errors);

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
