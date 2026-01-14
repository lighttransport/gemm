#include "gemm_pack_opt.h"
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Interleaved A Packing for optimal ld1rw loading
// ============================================================================
// Layout: [K/4][MR][4] - each K-group has MR consecutive 4-byte blocks
// This allows sequential ld1rw z.s, [ptr, #offset] with single pointer update

void pack_A_interleaved(const int8_t* A, int lda, int8_t* Apack, int M) {
    const int MR = MR_OPT;
    const int K = K_FIXED;

    int mr = (M >= MR) ? MR : M;

    // Pack K in groups of 4
    for (int k = 0; k < K; k += 4) {
        // For each row in the tile
        for (int m = 0; m < MR; m++) {
            if (m < mr) {
                // Copy 4 bytes from A[m][k:k+3]
                Apack[0] = A[m * lda + k + 0];
                Apack[1] = A[m * lda + k + 1];
                Apack[2] = A[m * lda + k + 2];
                Apack[3] = A[m * lda + k + 3];
            } else {
                // Zero-pad for incomplete tiles
                Apack[0] = 0;
                Apack[1] = 0;
                Apack[2] = 0;
                Apack[3] = 0;
            }
            Apack += 4;
        }
    }
}

// ============================================================================
// B Packing for SDOT (optimized version)
// ============================================================================
// Layout: [K/4][4 vectors][16 lanes][4 bytes]
// Each K-group produces 256 bytes (4 SVE vectors worth)

void pack_B_sdot(const int8_t* B, int ldb, int8_t* Bpack, int N) {
    const int NR = NR_OPT;
    const int K = K_FIXED;

    int nr = (N >= NR) ? NR : N;

    // Pack K in groups of 4
    for (int k = 0; k < K; k += 4) {
        // Pack 4 SVE vectors (64 columns total)
        for (int vec = 0; vec < 4; vec++) {
            // Each vector has 16 lanes
            for (int lane = 0; lane < 16; lane++) {
                int n = vec * 16 + lane;

                // Pack 4 consecutive K values for this column
                if (n < nr) {
                    Bpack[0] = B[n * ldb + k + 0];
                    Bpack[1] = B[n * ldb + k + 1];
                    Bpack[2] = B[n * ldb + k + 2];
                    Bpack[3] = B[n * ldb + k + 3];
                } else {
                    Bpack[0] = 0;
                    Bpack[1] = 0;
                    Bpack[2] = 0;
                    Bpack[3] = 0;
                }
                Bpack += 4;
            }
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

static inline void* aligned_alloc_helper(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

static inline int min_int(int a, int b) {
    return (a < b) ? a : b;
}

// Edge case handler using unpacked matrices
static void gemm_edge_case_opt(const int8_t* A, int lda,
                               const int8_t* B, int ldb,
                               int32_t* C, int ldc,
                               int mr, int nr, int K) {
    for (int m = 0; m < mr; m++) {
        for (int n = 0; n < nr; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * lda + k] * (int32_t)B[n * ldb + k];
            }
            C[m * ldc + n] = sum;
        }
    }
}

// ============================================================================
// Optimized Single GEMM Driver
// ============================================================================

void gemm_opt_driver(const int8_t* A, int lda,
                     const int8_t* B, int ldb,
                     int32_t* C, int ldc,
                     int M, int N, int K) {
    const int MR = MR_OPT;
    const int NR = NR_OPT;

    if (K != K_FIXED) return;

    // Allocate aligned packing buffers
    int8_t* Apack = (int8_t*)aligned_alloc_helper(256, MR * K);
    int8_t* Bpack = (int8_t*)aligned_alloc_helper(256, ((N + NR - 1) / NR) * NR * K);

    if (!Apack || !Bpack) {
        free(Apack);
        free(Bpack);
        return;
    }

    // Pack all of B once (reused across all M tiles)
    int8_t* Bpack_ptr = Bpack;
    for (int n0 = 0; n0 < N; n0 += NR) {
        int nr = min_int(NR, N - n0);
        pack_B_sdot(B + n0 * ldb, ldb, Bpack_ptr, nr);
        Bpack_ptr += NR * K;
    }

    // Process M in tiles of MR
    for (int m0 = 0; m0 < M; m0 += MR) {
        int mr = min_int(MR, M - m0);

        // Pack current A tile with interleaved layout
        pack_A_interleaved(A + m0 * lda, lda, Apack, mr);

        // Process N in tiles of NR
        Bpack_ptr = Bpack;
        for (int n0 = 0; n0 < N; n0 += NR) {
            int nr = min_int(NR, N - n0);

            int32_t* C_tile = C + m0 * ldc + n0;

            if (mr == MR && nr == NR) {
                // Full tile - call optimized kernel
                kernel_6x4_ultra(Apack, Bpack_ptr, C_tile, ldc * sizeof(int32_t));
            } else {
                // Edge case
                gemm_edge_case_opt(A + m0 * lda, lda, B + n0 * ldb, ldb,
                                   C_tile, ldc, mr, nr, K);
            }

            Bpack_ptr += NR * K;
        }
    }

    free(Apack);
    free(Bpack);
}

// ============================================================================
// Batched GEMM for Multiple Heads
// ============================================================================

void gemm_batch_opt(const int8_t* A, int lda, int64_t strideA,
                    const int8_t* B, int ldb, int64_t strideB,
                    int32_t* C, int ldc, int64_t strideC,
                    int M, int N, int K, int batch) {
    // Process each batch element
    for (int b = 0; b < batch; b++) {
        gemm_opt_driver(A + b * strideA, lda,
                        B + b * strideB, ldb,
                        C + b * strideC, ldc,
                        M, N, K);
    }
}

// ============================================================================
// Pipelined GEMM Driver (uses kernel_6x4_pipe)
// ============================================================================

void gemm_pipe_driver(const int8_t* A, int lda,
                      const int8_t* B, int ldb,
                      int32_t* C, int ldc,
                      int M, int N, int K) {
    const int MR = MR_OPT;
    const int NR = NR_OPT;

    if (K != K_FIXED) return;

    // Allocate aligned packing buffers
    int8_t* Apack = (int8_t*)aligned_alloc_helper(256, MR * K);
    int8_t* Bpack = (int8_t*)aligned_alloc_helper(256, ((N + NR - 1) / NR) * NR * K);

    if (!Apack || !Bpack) {
        free(Apack);
        free(Bpack);
        return;
    }

    // Pack all of B once (reused across all M tiles)
    int8_t* Bpack_ptr = Bpack;
    for (int n0 = 0; n0 < N; n0 += NR) {
        int nr = min_int(NR, N - n0);
        pack_B_sdot(B + n0 * ldb, ldb, Bpack_ptr, nr);
        Bpack_ptr += NR * K;
    }

    // Process M in tiles of MR
    for (int m0 = 0; m0 < M; m0 += MR) {
        int mr = min_int(MR, M - m0);

        // Pack current A tile with interleaved layout
        pack_A_interleaved(A + m0 * lda, lda, Apack, mr);

        // Process N in tiles of NR
        Bpack_ptr = Bpack;
        for (int n0 = 0; n0 < N; n0 += NR) {
            int nr = min_int(NR, N - n0);

            int32_t* C_tile = C + m0 * ldc + n0;

            if (mr == MR && nr == NR) {
                // Full tile - call pipelined kernel
                kernel_6x4_pipe(Apack, Bpack_ptr, C_tile, ldc * sizeof(int32_t));
            } else {
                // Edge case
                gemm_edge_case_opt(A + m0 * lda, lda, B + n0 * ldb, ldb,
                                   C_tile, ldc, mr, nr, K);
            }

            Bpack_ptr += NR * K;
        }
    }

    free(Apack);
    free(Bpack);
}

// ============================================================================
// Batched GEMM with Pipelined Kernel
// ============================================================================

void gemm_batch_pipe(const int8_t* A, int lda, int64_t strideA,
                     const int8_t* B, int ldb, int64_t strideB,
                     int32_t* C, int ldc, int64_t strideC,
                     int M, int N, int K, int batch) {
    for (int b = 0; b < batch; b++) {
        gemm_pipe_driver(A + b * strideA, lda,
                         B + b * strideB, ldb,
                         C + b * strideC, ldc,
                         M, N, K);
    }
}
