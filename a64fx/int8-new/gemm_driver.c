#include "gemm_driver.h"
#include "gemm_pack.h"
#include <stdlib.h>
#include <string.h>

#define ALIGN 256
#define K_FIXED 256

// Helper: aligned allocation
static inline void* aligned_alloc_helper(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

// Helper: min function
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

// Edge case handler (simple scalar fallback for incomplete tiles)
// Works with unpacked A and B to avoid complex packed layout indexing
static void gemm_edge_case(const int8_t* A, int lda,
                          const int8_t* B, int ldb,
                          int32_t* C, int ldc,
                          int mr, int nr, int K) {
    // Simple triple-loop for edge cases using unpacked matrices
    // C[m,n] = sum_k A[m,k] * B[n,k]  (B accessed as B^T)
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
// 5x4 Driver: C[M×N] = A[M×K] × B[N×K]^T with MR=5, NR=64, K=256
// ============================================================================
void gemm_5x4_driver(const int8_t* A, int lda,
                     const int8_t* B, int ldb,
                     int32_t* C, int ldc,
                     int M, int N, int K) {
    const int MR = 5;
    const int NR = 64;

    // Validate K
    if (K != K_FIXED) {
        // K must be 256 for this implementation
        return;
    }

    // Allocate aligned packing buffers
    // Apack: max Mr rows × K bytes
    // Bpack: all N cols (in 64-col tiles) × K bytes
    int8_t* Apack = (int8_t*)aligned_alloc_helper(ALIGN, MR * K);
    int8_t* Bpack = (int8_t*)aligned_alloc_helper(ALIGN, N * K);

    if (!Apack || !Bpack) {
        free(Apack);
        free(Bpack);
        return;
    }

    // Pack B once (reused across all M tiles)
    int8_t* Bpack_ptr = Bpack;
    for (int n0 = 0; n0 < N; n0 += NR) {
        pack_B_64x256(B + n0 * ldb, ldb, Bpack_ptr, N - n0);
        Bpack_ptr += NR * K;
    }

    // Outer M loop: process MR rows at a time
    for (int m0 = 0; m0 < M; m0 += MR) {
        int mr = min(MR, M - m0);

        // Pack current A tile
        pack_A_5x256(A + m0 * lda, lda, Apack, mr);

        // Inner N loop: process NR columns at a time
        Bpack_ptr = Bpack;
        for (int n0 = 0; n0 < N; n0 += NR) {
            int nr = min(NR, N - n0);

            int32_t* C_tile = C + m0 * ldc + n0;

            if (mr == MR && nr == NR) {
                // Full tile - call optimized kernel
                kernel_5x4_256(Apack, Bpack_ptr, C_tile, ldc * sizeof(int32_t));
            } else {
                // Edge case - use scalar fallback with unpacked matrices
                gemm_edge_case(A + m0 * lda, lda, B + n0 * ldb, ldb, C_tile, ldc, mr, nr, K);
            }

            Bpack_ptr += NR * K;
        }
    }

    free(Apack);
    free(Bpack);
}

// ============================================================================
// 6x4 Driver: C[M×N] = A[M×K] × B[N×K]^T with MR=6, NR=64, K=256
// ============================================================================
void gemm_6x4_driver(const int8_t* A, int lda,
                     const int8_t* B, int ldb,
                     int32_t* C, int ldc,
                     int M, int N, int K) {
    const int MR = 6;
    const int NR = 64;

    // Validate K
    if (K != K_FIXED) {
        return;
    }

    // Allocate aligned packing buffers
    int8_t* Apack = (int8_t*)aligned_alloc_helper(ALIGN, MR * K);
    int8_t* Bpack = (int8_t*)aligned_alloc_helper(ALIGN, N * K);

    if (!Apack || !Bpack) {
        free(Apack);
        free(Bpack);
        return;
    }

    // Pack B once
    int8_t* Bpack_ptr = Bpack;
    for (int n0 = 0; n0 < N; n0 += NR) {
        pack_B_64x256(B + n0 * ldb, ldb, Bpack_ptr, N - n0);
        Bpack_ptr += NR * K;
    }

    // Outer M loop
    for (int m0 = 0; m0 < M; m0 += MR) {
        int mr = min(MR, M - m0);

        // Pack current A tile
        pack_A_6x256(A + m0 * lda, lda, Apack, mr);

        // Inner N loop
        Bpack_ptr = Bpack;
        for (int n0 = 0; n0 < N; n0 += NR) {
            int nr = min(NR, N - n0);

            int32_t* C_tile = C + m0 * ldc + n0;

            if (mr == MR && nr == NR) {
                // Full tile - call optimized kernel
                kernel_6x4_256(Apack, Bpack_ptr, C_tile, ldc * sizeof(int32_t));
            } else {
                // Edge case - use scalar fallback with unpacked matrices
                gemm_edge_case(A + m0 * lda, lda, B + n0 * ldb, ldb, C_tile, ldc, mr, nr, K);
            }

            Bpack_ptr += NR * K;
        }
    }

    free(Apack);
    free(Bpack);
}

// ============================================================================
// 6x4 Optimized Driver: C[M×N] = A[M×K] × B[N×K]^T with MR=6, NR=64, K=256
// Uses optimized kernel with K-loop unrolled 8×
// ============================================================================
void gemm_6x4_opt_driver(const int8_t* A, int lda,
                         const int8_t* B, int ldb,
                         int32_t* C, int ldc,
                         int M, int N, int K) {
    const int MR = 6;
    const int NR = 64;

    // Validate K
    if (K != K_FIXED) {
        return;
    }

    // Allocate aligned packing buffers
    int8_t* Apack = (int8_t*)aligned_alloc_helper(ALIGN, MR * K);
    int8_t* Bpack = (int8_t*)aligned_alloc_helper(ALIGN, N * K);

    if (!Apack || !Bpack) {
        free(Apack);
        free(Bpack);
        return;
    }

    // Pack B once
    int8_t* Bpack_ptr = Bpack;
    for (int n0 = 0; n0 < N; n0 += NR) {
        pack_B_64x256(B + n0 * ldb, ldb, Bpack_ptr, N - n0);
        Bpack_ptr += NR * K;
    }

    // Outer M loop
    for (int m0 = 0; m0 < M; m0 += MR) {
        int mr = min(MR, M - m0);

        // Pack current A tile
        pack_A_6x256(A + m0 * lda, lda, Apack, mr);

        // Inner N loop
        Bpack_ptr = Bpack;
        for (int n0 = 0; n0 < N; n0 += NR) {
            int nr = min(NR, N - n0);

            int32_t* C_tile = C + m0 * ldc + n0;

            if (mr == MR && nr == NR) {
                // Full tile - call optimized kernel
                kernel_6x4_opt_256(Apack, Bpack_ptr, C_tile, ldc * sizeof(int32_t));
            } else {
                // Edge case - use scalar fallback with unpacked matrices
                gemm_edge_case(A + m0 * lda, lda, B + n0 * ldb, ldb, C_tile, ldc, mr, nr, K);
            }

            Bpack_ptr += NR * K;
        }
    }

    free(Apack);
    free(Bpack);
}
