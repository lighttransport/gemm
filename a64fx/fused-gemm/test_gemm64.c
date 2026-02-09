// Test GEMM with exact benchmark dimensions
#include "fused_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern size_t packed_A_size(int M, int K);
extern size_t packed_B_size(int K, int N);

extern void micro_kernel_fp32_8x3_unroll4(
    const float* A_packed,
    const float* B_packed,
    float* C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes
);

static inline int round_up(int x, int mult) {
    return ((x + mult - 1) / mult) * mult;
}

int main() {
    int M = 64;
    int K = 64;
    int N = 48;

    printf("Test GEMM: M=%d, K=%d, N=%d\n", M, K, N);

    int K_rounded = round_up(K, 4);
    printf("K_rounded = %d\n", K_rounded);

    size_t A_packed_size = packed_A_size(M, K);
    size_t B_packed_size = packed_B_size(K, N);
    printf("Packed sizes: A=%zu, B=%zu\n", A_packed_size, B_packed_size);

    // Allocate
    float* A = (float*)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float* B = (float*)aligned_alloc(64, (size_t)K * N * sizeof(float));
    float* C = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));
    float* A_packed = (float*)aligned_alloc(64, A_packed_size);
    float* B_packed = (float*)aligned_alloc(64, B_packed_size);

    printf("Allocated:\n");
    printf("  A=%p\n  B=%p\n  C=%p\n  A_packed=%p\n  B_packed=%p\n",
           A, B, C, A_packed, B_packed);

    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;
    memset(C, 0, (size_t)M * N * sizeof(float));

    // Pack
    printf("Packing A...\n");
    pack_A_fp32(M, K, A, K, A_packed);
    printf("Packing B...\n");
    pack_B_fp32(K, N, B, N, B_packed);

    // Tile loop
    int M_blocks = (M + MR - 1) / MR;
    int N_blocks = (N + NR - 1) / NR;
    printf("Blocks: M=%d, N=%d\n", M_blocks, N_blocks);

    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks; nb++) {
            int m_start = mb * MR;
            int n_start = nb * NR;

            const float* A_tile = A_packed + mb * K_rounded * MR;
            const float* B_tile = B_packed + nb * K_rounded * NR;
            float* C_tile = C + m_start * N + n_start;

            printf("Tile [%d,%d]: A_tile=%p, B_tile=%p, C_tile=%p\n",
                   mb, nb, A_tile, B_tile, C_tile);

            // Check bounds
            size_t A_offset = (size_t)((const char*)A_tile - (const char*)A_packed);
            size_t B_offset = (size_t)((const char*)B_tile - (const char*)B_packed);
            size_t A_end = A_offset + K_rounded * MR * sizeof(float);
            size_t B_end = B_offset + K_rounded * NR * sizeof(float);

            printf("  A: offset=%zu, end=%zu, size=%zu\n", A_offset, A_end, A_packed_size);
            printf("  B: offset=%zu, end=%zu, size=%zu\n", B_offset, B_end, B_packed_size);

            if (A_end > A_packed_size) {
                printf("  ERROR: A access out of bounds!\n");
            }
            if (B_end > B_packed_size) {
                printf("  ERROR: B access out of bounds!\n");
            }

            micro_kernel_fp32_8x3_unroll4(
                A_tile,
                B_tile,
                C_tile,
                K_rounded,
                0,
                (int64_t)N * sizeof(float)
            );
        }
    }

    printf("C[0,0] = %f (expected %f)\n", C[0], (float)K);

    // Free
    printf("Freeing...\n");
    free(A);
    printf("  A freed\n");
    free(B);
    printf("  B freed\n");
    free(C);
    printf("  C freed\n");
    printf("  About to free A_packed at %p\n", A_packed);
    free(A_packed);
    printf("  A_packed freed\n");
    printf("  About to free B_packed at %p\n", B_packed);
    free(B_packed);
    printf("  B_packed freed\n");

    printf("Done!\n");
    return 0;
}
