// Fused GEMM (A@B)@C implementation for A64FX SVE FP32
// Two-pass approach with L2-resident intermediate

#include "fused_gemm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// External assembly kernels
extern void micro_kernel_fp32_8x3_unroll4(
    const float* A_packed,
    const float* B_packed,
    float* C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes
);

extern void micro_kernel_fp32_8x3_unroll4_acc(
    const float* A_packed,
    const float* B_packed,
    float* C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes
);

// External packing functions (defined in pack_matrices.c)
extern size_t packed_A_size(int M, int K);
extern size_t packed_B_size(int K, int N);

// Forward declarations
static void gemm_fp32_acc(
    int M, int K, int N,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
);

// Round up to multiple
static inline int round_up(int x, int mult) {
    return ((x + mult - 1) / mult) * mult;
}

// Single GEMM: C = A @ B
// Uses tiled algorithm with 8x3 microkernel
void gemm_fp32(
    int M, int K, int N,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
) {
    // Round dimensions for packing
    int K_rounded = round_up(K, 4);

    // Allocate packed buffers
    size_t A_packed_size = packed_A_size(M, K);
    size_t B_packed_size = packed_B_size(K, N);

    float* A_packed = (float*)aligned_alloc(64, A_packed_size);
    float* B_packed = (float*)aligned_alloc(64, B_packed_size);

    if (!A_packed || !B_packed) {
        fprintf(stderr, "Failed to allocate packing buffers\n");
        free(A_packed);
        free(B_packed);
        return;
    }

    // Pack matrices
    pack_A_fp32(M, K, A, lda, A_packed);
    pack_B_fp32(K, N, B, ldb, B_packed);

    // Initialize output to zero
    for (int m = 0; m < M; m++) {
        memset(C + m * ldc, 0, N * sizeof(float));
    }

    // Tiling parameters
    int M_blocks = (M + MR - 1) / MR;
    int N_blocks = (N + NR - 1) / NR;

    // Allocate temporary tile buffer for edge cases
    float* tile_buf = NULL;
    int need_edge_handling = (N % NR != 0) || (M % MR != 0);
    if (need_edge_handling) {
        tile_buf = (float*)aligned_alloc(64, MR * NR * sizeof(float));
    }

    // Process tiles
    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks; nb++) {
            int m_start = mb * MR;
            int n_start = nb * NR;
            int m_count = (m_start + MR <= M) ? MR : M - m_start;
            int n_count = (n_start + NR <= N) ? NR : N - n_start;

            // Pointers to packed data for this tile
            const float* A_tile = A_packed + mb * K_rounded * MR;
            const float* B_tile = B_packed + nb * K_rounded * NR;

            // Check if this is an edge tile that needs special handling
            if (m_count == MR && n_count == NR) {
                // Full tile - write directly to output
                float* C_tile = C + m_start * ldc + n_start;
                micro_kernel_fp32_8x3_unroll4(
                    A_tile,
                    B_tile,
                    C_tile,
                    K_rounded,
                    0,
                    (int64_t)ldc * sizeof(float)
                );
            } else {
                // Edge tile - write to temp buffer then copy
                float local_buf[MR * NR] __attribute__((aligned(64)));
                micro_kernel_fp32_8x3_unroll4(
                    A_tile,
                    B_tile,
                    local_buf,
                    K_rounded,
                    0,
                    (int64_t)NR * sizeof(float)
                );
                // Copy valid portion to output
                for (int m = 0; m < m_count; m++) {
                    for (int n = 0; n < n_count; n++) {
                        C[(m_start + m) * ldc + n_start + n] = local_buf[m * NR + n];
                    }
                }
            }
        }
    }

    if (tile_buf) free(tile_buf);
    free(A_packed);
    free(B_packed);
}

// Fused GEMM: E = (A @ B) @ C
// Two-pass implementation:
//   Pass 1: D[M,K2] = A[M,K1] @ B[K1,K2]
//   Pass 2: E[M,N] = D[M,K2] @ C[K2,N]
void fused_gemm_abc(
    int M, int K1, int K2, int N,
    const float* A, int lda,
    const float* B, int ldb,
    const float* C, int ldc,
    float* E, int lde
) {
    // Allocate intermediate D (L2-resident target)
    // D[M,K2] stores result of A @ B
    float* D = (float*)aligned_alloc(64, (size_t)M * K2 * sizeof(float));
    if (!D) {
        fprintf(stderr, "Failed to allocate intermediate buffer D\n");
        return;
    }

    // Pass 1: D = A @ B
    gemm_fp32(M, K1, K2, A, lda, B, ldb, D, K2);

    // Pass 2: E = D @ C
    gemm_fp32(M, K2, N, D, K2, C, ldc, E, lde);

    free(D);
}

// Workspace size for fused GEMM (intermediate D buffer)
size_t fused_gemm_workspace_size(int M, int K1, int K2, int N) {
    // D[M,K2] intermediate
    size_t D_size = (size_t)M * K2 * sizeof(float);

    // Packing buffers (reused between passes)
    size_t A_pack = packed_A_size(M, K1 > K2 ? K1 : K2);
    size_t B_pack = packed_B_size(K1 > K2 ? K1 : K2, K2 > N ? K2 : N);

    return D_size + A_pack + B_pack;
}

// Optimized fused GEMM with pre-allocated workspace
// Avoids repeated allocations for batched operations
void fused_gemm_abc_workspace(
    int M, int K1, int K2, int N,
    const float* A, int lda,
    const float* B, int ldb,
    const float* C, int ldc,
    float* E, int lde,
    void* workspace
) {
    float* D = (float*)workspace;

    // Round dimensions
    int K1_rounded = round_up(K1, 4);
    int K2_rounded = round_up(K2, 4);

    // Workspace layout: D, A_packed, B_packed
    float* A_packed = D + M * K2;
    float* B_packed = A_packed + ((M + MR - 1) / MR) * K1_rounded * MR;

    // ═══════════════════════════════════════════════════════════════════
    // Pass 1: D[M,K2] = A[M,K1] @ B[K1,K2]
    // ═══════════════════════════════════════════════════════════════════

    // Pack A for pass 1
    pack_A_fp32(M, K1, A, lda, A_packed);

    // Pack B for pass 1
    pack_B_fp32(K1, K2, B, ldb, B_packed);

    // Initialize D to zero
    memset(D, 0, (size_t)M * K2 * sizeof(float));

    int M_blocks = (M + MR - 1) / MR;
    int N_blocks_p1 = (K2 + NR - 1) / NR;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks_p1; nb++) {
            int m_start = mb * MR;
            int n_start = nb * NR;

            const float* A_tile = A_packed + mb * K1_rounded * MR;
            const float* B_tile = B_packed + nb * K1_rounded * NR;
            float* D_tile = D + m_start * K2 + n_start;

            micro_kernel_fp32_8x3_unroll4(
                A_tile,
                B_tile,
                D_tile,
                K1_rounded,
                0,
                (int64_t)K2 * sizeof(float)
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pass 2: E[M,N] = D[M,K2] @ C[K2,N]
    // ═══════════════════════════════════════════════════════════════════

    // Reuse A_packed buffer for D packing
    pack_A_fp32(M, K2, D, K2, A_packed);

    // Pack C into B_packed buffer
    pack_B_fp32(K2, N, C, ldc, B_packed);

    // Initialize E to zero
    for (int m = 0; m < M; m++) {
        memset(E + m * lde, 0, N * sizeof(float));
    }

    int N_blocks_p2 = (N + NR - 1) / NR;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks_p2; nb++) {
            int m_start = mb * MR;
            int n_start = nb * NR;

            const float* D_tile = A_packed + mb * K2_rounded * MR;
            const float* C_tile = B_packed + nb * K2_rounded * NR;
            float* E_tile = E + m_start * lde + n_start;

            micro_kernel_fp32_8x3_unroll4(
                D_tile,
                C_tile,
                E_tile,
                K2_rounded,
                0,
                (int64_t)lde * sizeof(float)
            );
        }
    }
}

// Tiled fused GEMM for very large matrices
// Uses cache blocking to keep intermediate in L2
void fused_gemm_abc_tiled(
    int M, int K1, int K2, int N,
    const float* A, int lda,
    const float* B, int ldb,
    const float* C, int ldc,
    float* E, int lde
) {
    // Block sizes chosen for L2 residency (8MB L2 on A64FX)
    // Target: intermediate block fits in ~4MB to leave room for A, B, C blocks
    const int MB = 256;   // M block size
    const int K2B = 768;  // K2 block size (intermediate width)
    const int NB = 768;   // N block size

    // Allocate intermediate block D[MB, K2B]
    float* D_block = (float*)aligned_alloc(64, (size_t)MB * K2B * sizeof(float));
    if (!D_block) {
        // Fall back to simple two-pass
        fused_gemm_abc(M, K1, K2, N, A, lda, B, ldb, C, ldc, E, lde);
        return;
    }

    // Initialize output
    for (int m = 0; m < M; m++) {
        memset(E + m * lde, 0, N * sizeof(float));
    }

    // Tile over M
    for (int m0 = 0; m0 < M; m0 += MB) {
        int mb = (m0 + MB <= M) ? MB : M - m0;

        // Tile over K2 (intermediate dimension)
        for (int k2_0 = 0; k2_0 < K2; k2_0 += K2B) {
            int k2b = (k2_0 + K2B <= K2) ? K2B : K2 - k2_0;

            // Pass 1 block: D_block[mb, k2b] = A[m0:m0+mb, :] @ B[:, k2_0:k2_0+k2b]
            const float* A_block = A + m0 * lda;
            const float* B_block = B + k2_0;  // B[0, k2_0]

            gemm_fp32(mb, K1, k2b, A_block, lda, B_block, ldb, D_block, k2b);

            // Tile over N
            for (int n0 = 0; n0 < N; n0 += NB) {
                int nb = (n0 + NB <= N) ? NB : N - n0;

                // Pass 2 block: E[m0:m0+mb, n0:n0+nb] += D_block @ C[k2_0:k2_0+k2b, n0:n0+nb]
                const float* C_block = C + k2_0 * ldc + n0;
                float* E_block = E + m0 * lde + n0;

                // Accumulate into E (need accumulating GEMM for k2_0 > 0)
                if (k2_0 == 0) {
                    gemm_fp32(mb, k2b, nb, D_block, k2b, C_block, ldc, E_block, lde);
                } else {
                    // Accumulating GEMM: E_block += D_block @ C_block
                    gemm_fp32_acc(mb, k2b, nb, D_block, k2b, C_block, ldc, E_block, lde);
                }
            }
        }
    }

    free(D_block);
}

// Accumulating GEMM: C += A @ B
static void gemm_fp32_acc(
    int M, int K, int N,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
) {
    int K_rounded = round_up(K, 4);

    size_t A_packed_size = packed_A_size(M, K);
    size_t B_packed_size = packed_B_size(K, N);

    float* A_packed = (float*)aligned_alloc(64, A_packed_size);
    float* B_packed = (float*)aligned_alloc(64, B_packed_size);

    if (!A_packed || !B_packed) {
        free(A_packed);
        free(B_packed);
        return;
    }

    pack_A_fp32(M, K, A, lda, A_packed);
    pack_B_fp32(K, N, B, ldb, B_packed);

    int M_blocks = (M + MR - 1) / MR;
    int N_blocks = (N + NR - 1) / NR;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks; nb++) {
            int m_start = mb * MR;
            int n_start = nb * NR;
            int m_count = (m_start + MR <= M) ? MR : M - m_start;
            int n_count = (n_start + NR <= N) ? NR : N - n_start;

            const float* A_tile = A_packed + mb * K_rounded * MR;
            const float* B_tile = B_packed + nb * K_rounded * NR;

            if (m_count == MR && n_count == NR) {
                // Full tile
                float* C_tile = C + m_start * ldc + n_start;
                micro_kernel_fp32_8x3_unroll4_acc(
                    A_tile,
                    B_tile,
                    C_tile,
                    K_rounded,
                    0,
                    (int64_t)ldc * sizeof(float)
                );
            } else {
                // Edge tile - need to load, compute, then store partial
                float local_buf[MR * NR] __attribute__((aligned(64)));
                // Load existing values
                for (int m = 0; m < m_count; m++) {
                    for (int n = 0; n < n_count; n++) {
                        local_buf[m * NR + n] = C[(m_start + m) * ldc + n_start + n];
                    }
                    for (int n = n_count; n < NR; n++) {
                        local_buf[m * NR + n] = 0.0f;
                    }
                }
                for (int m = m_count; m < MR; m++) {
                    for (int n = 0; n < NR; n++) {
                        local_buf[m * NR + n] = 0.0f;
                    }
                }
                // Accumulate
                micro_kernel_fp32_8x3_unroll4_acc(
                    A_tile,
                    B_tile,
                    local_buf,
                    K_rounded,
                    0,
                    (int64_t)NR * sizeof(float)
                );
                // Store back
                for (int m = 0; m < m_count; m++) {
                    for (int n = 0; n < n_count; n++) {
                        C[(m_start + m) * ldc + n_start + n] = local_buf[m * NR + n];
                    }
                }
            }
        }
    }

    free(A_packed);
    free(B_packed);
}
