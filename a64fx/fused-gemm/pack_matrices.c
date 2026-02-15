// Matrix packing functions for FP32 GEMM with 8x3 microkernel
// Optimized for A64FX SVE 512-bit vectors

#include "fused_gemm.h"
#include <string.h>

// Pack A matrix for broadcast access
// Input:  A[M,K] row-major with leading dimension lda
// Output: A_packed in format [K][MR] for each M-block
//         The microkernel loads A[k][m] via ld1rw at offsets 0,4,8,...,28
//
// Memory layout per M-block: for k in 0..K-1: A_packed[k*MR + m] = A[m_start+m, k]
// The kernel processes 4 K iterations per loop, loading 8 floats each time
void pack_A_fp32(int M, int K, const float* A, int lda, float* A_packed) {
    const int M_blocks = (M + MR - 1) / MR;
    const int K_rounded = ((K + 3) / 4) * 4;

    for (int mb = 0; mb < M_blocks; mb++) {
        int m_start = mb * MR;
        int m_end = (m_start + MR < M) ? m_start + MR : M;
        int m_count = m_end - m_start;

        float* dst = A_packed + (size_t)mb * K_rounded * MR;

        // Pack K dimension, storing MR elements per k
        for (int k = 0; k < K; k++) {
            for (int m = 0; m < m_count; m++) {
                dst[k * MR + m] = A[(m_start + m) * lda + k];
            }
            // Zero-pad if M-block is partial
            for (int m = m_count; m < MR; m++) {
                dst[k * MR + m] = 0.0f;
            }
        }
        // Zero-pad remaining K to rounded value
        for (int k = K; k < K_rounded; k++) {
            for (int m = 0; m < MR; m++) {
                dst[k * MR + m] = 0.0f;
            }
        }
    }
}

// Pack B matrix for vector access
// Input:  B[K,N] row-major with leading dimension ldb
// Output: B_packed in format [K][NR] for each N-block
//         The microkernel loads B[k][0:47] via ld1w at offsets 0, 1*vl, 2*vl
//
// Memory layout per N-block: for k in 0..K-1: B_packed[k*NR + n] = B[k, n_start+n]
// The kernel processes 4 K iterations per loop, advancing B pointer by 4*NR*4 bytes
void pack_B_fp32(int K, int N, const float* B, int ldb, float* B_packed) {
    const int N_blocks = (N + NR - 1) / NR;
    const int K_rounded = ((K + 3) / 4) * 4;

    for (int nb = 0; nb < N_blocks; nb++) {
        int n_start = nb * NR;
        int n_end = (n_start + NR < N) ? n_start + NR : N;
        int n_count = n_end - n_start;

        float* dst = B_packed + (size_t)nb * K_rounded * NR;

        // Pack K dimension, storing NR elements per k
        for (int k = 0; k < K; k++) {
            // Copy NR elements from row k
            for (int n = 0; n < n_count; n++) {
                dst[k * NR + n] = B[k * ldb + n_start + n];
            }
            // Zero-pad if N-block is partial
            for (int n = n_count; n < NR; n++) {
                dst[k * NR + n] = 0.0f;
            }
        }
        // Zero-pad remaining K to rounded value
        for (int k = K; k < K_rounded; k++) {
            for (int n = 0; n < NR; n++) {
                dst[k * NR + n] = 0.0f;
            }
        }
    }
}

// Alternative: Pack A in transposed format for better cache access
// Input:  A[M,K] row-major
// Output: A_packed[K,M] stored as [M_blocks][K_rounded][MR]
void pack_A_fp32_v2(int M, int K, const float* A, int lda, float* A_packed) {
    const int M_blocks = (M + MR - 1) / MR;
    const int K_rounded = ((K + 3) / 4) * 4;

    for (int mb = 0; mb < M_blocks; mb++) {
        int m_start = mb * MR;
        int m_end = (m_start + MR < M) ? m_start + MR : M;
        int m_count = m_end - m_start;

        float* dst = A_packed + mb * K_rounded * MR;

        for (int k = 0; k < K; k++) {
            for (int m = 0; m < m_count; m++) {
                dst[k * MR + m] = A[(m_start + m) * lda + k];
            }
            for (int m = m_count; m < MR; m++) {
                dst[k * MR + m] = 0.0f;
            }
        }
        // Zero-pad remaining K
        for (int k = K; k < K_rounded; k++) {
            for (int m = 0; m < MR; m++) {
                dst[k * MR + m] = 0.0f;
            }
        }
    }
}

// Pack B with K-major ordering optimized for streaming
void pack_B_fp32_v2(int K, int N, const float* B, int ldb, float* B_packed) {
    const int N_blocks = (N + NR - 1) / NR;
    const int K_rounded = ((K + 3) / 4) * 4;

    for (int nb = 0; nb < N_blocks; nb++) {
        int n_start = nb * NR;
        int n_end = (n_start + NR < N) ? n_start + NR : N;
        int n_count = n_end - n_start;

        float* dst = B_packed + nb * K_rounded * NR;

        for (int k = 0; k < K; k++) {
            for (int n = 0; n < n_count; n++) {
                dst[k * NR + n] = B[k * ldb + n_start + n];
            }
            for (int n = n_count; n < NR; n++) {
                dst[k * NR + n] = 0.0f;
            }
        }
        // Zero-pad remaining K
        for (int k = K; k < K_rounded; k++) {
            for (int n = 0; n < NR; n++) {
                dst[k * NR + n] = 0.0f;
            }
        }
    }
}

// Unpack result from tile storage back to row-major
// Used when output is in tiled format
void unpack_C_fp32(int M, int N, const float* C_packed, float* C, int ldc) {
    const int M_blocks = (M + MR - 1) / MR;
    const int N_blocks = (N + NR - 1) / NR;

    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks; nb++) {
            int m_start = mb * MR;
            int n_start = nb * NR;
            int m_end = (m_start + MR < M) ? m_start + MR : M;
            int n_end = (n_start + NR < N) ? n_start + NR : N;

            const float* src = C_packed + (mb * N_blocks + nb) * MR * NR;

            for (int m = m_start; m < m_end; m++) {
                for (int n = n_start; n < n_end; n++) {
                    C[m * ldc + n] = src[(m - m_start) * NR + (n - n_start)];
                }
            }
        }
    }
}

// Calculate packed buffer sizes
size_t packed_A_size(int M, int K) {
    int M_blocks = (M + MR - 1) / MR;
    int K_rounded = ((K + 3) / 4) * 4;
    return (size_t)M_blocks * K_rounded * MR * sizeof(float);
}

size_t packed_B_size(int K, int N) {
    int N_blocks = (N + NR - 1) / NR;
    int K_rounded = ((K + 3) / 4) * 4;
    return (size_t)N_blocks * K_rounded * NR * sizeof(float);
}
