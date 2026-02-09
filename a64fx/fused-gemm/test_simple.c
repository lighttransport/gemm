// Simple test for debugging
#include "fused_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void micro_kernel_fp32_8x3_unroll4(
    const float* A_packed,
    const float* B_packed,
    float* C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes
);

extern size_t packed_A_size(int M, int K);
extern size_t packed_B_size(int K, int N);

int main() {
    // Test with minimal size: 8x4 @ 4x48 = 8x48
    int M = 8;
    int K = 4;
    int N = 48;

    printf("Test dimensions: M=%d, K=%d, N=%d\n", M, K, N);

    // Allocate matrices
    float* A = (float*)aligned_alloc(64, M * K * sizeof(float));
    float* B = (float*)aligned_alloc(64, K * N * sizeof(float));
    float* C = (float*)aligned_alloc(64, M * N * sizeof(float));

    printf("A=%p, B=%p, C=%p\n", A, B, C);

    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;
    memset(C, 0, M * N * sizeof(float));

    // Calculate packed sizes
    size_t A_size = packed_A_size(M, K);
    size_t B_size = packed_B_size(K, N);
    printf("Packed sizes: A=%zu, B=%zu\n", A_size, B_size);

    // Allocate packed buffers
    float* A_packed = (float*)aligned_alloc(64, A_size);
    float* B_packed = (float*)aligned_alloc(64, B_size);
    printf("A_packed=%p, B_packed=%p\n", A_packed, B_packed);

    if (!A_packed || !B_packed) {
        printf("Allocation failed!\n");
        return 1;
    }

    // Pack matrices
    printf("Packing A...\n");
    pack_A_fp32(M, K, A, K, A_packed);
    printf("Packing B...\n");
    pack_B_fp32(K, N, B, N, B_packed);

    printf("Packed A values:\n");
    for (int i = 0; i < 32 && i < M * K; i++) {
        printf("%.1f ", A_packed[i]);
    }
    printf("\n");

    printf("Packed B values:\n");
    for (int i = 0; i < 64 && i < K * N; i++) {
        printf("%.1f ", B_packed[i]);
    }
    printf("\n");

    // Call microkernel
    printf("Calling microkernel...\n");
    int64_t ldc_bytes = N * sizeof(float);
    micro_kernel_fp32_8x3_unroll4(A_packed, B_packed, C, K, 0, ldc_bytes);

    printf("Done! C[0,0]=%f (expected %f)\n", C[0], (float)K);

    // Verify result
    int errors = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float expected = (float)K;  // 1.0 * 1.0 * K = K
            if (C[m * N + n] != expected) {
                if (errors < 10) {
                    printf("Error at [%d,%d]: got %f, expected %f\n",
                           m, n, C[m * N + n], expected);
                }
                errors++;
            }
        }
    }
    printf("Errors: %d\n", errors);

    printf("Freeing A...\n");
    free(A);
    printf("Freeing B...\n");
    free(B);
    printf("Freeing C...\n");
    free(C);
    printf("Freeing A_packed...\n");
    free(A_packed);
    printf("Freeing B_packed...\n");
    free(B_packed);

    printf("Test complete!\n");
    return errors > 0 ? 1 : 0;
}
