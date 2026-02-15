#include "gemm_pack.h"

#define K_FIXED 256

// Pack A matrix for 5x4 microkernel
// Layout: [M_tiles][K][5] where each tile has Mr=5 rows
// Memory: Apack[m*K + k] = A[(m0+m)*lda + k] for m in [0, Mr), k in [0, K)
void pack_A_5x256(const int8_t* A, int lda, int8_t* Apack, int M) {
    const int MR = 5;
    const int K = K_FIXED;

    for (int m0 = 0; m0 < M; m0 += MR) {
        int mr = (m0 + MR <= M) ? MR : (M - m0);

        // Pack mr rows (may be less than MR for last tile)
        // Each row stored contiguously: row-major per row
        for (int m = 0; m < MR; m++) {
            if (m < mr) {
                // Copy actual row data
                for (int k = 0; k < K; k++) {
                    *Apack++ = A[(m0 + m) * lda + k];
                }
            } else {
                // Zero-pad incomplete tiles
                for (int k = 0; k < K; k++) {
                    *Apack++ = 0;
                }
            }
        }
    }
}

// Pack A matrix for 6x4 microkernel
// Layout: [M_tiles][K][6] where each tile has Mr=6 rows
// Memory: Apack[m*K + k] = A[(m0+m)*lda + k] for m in [0, Mr), k in [0, K)
void pack_A_6x256(const int8_t* A, int lda, int8_t* Apack, int M) {
    const int MR = 6;
    const int K = K_FIXED;

    for (int m0 = 0; m0 < M; m0 += MR) {
        int mr = (m0 + MR <= M) ? MR : (M - m0);

        // Pack mr rows (may be less than MR for last tile)
        // Each row stored contiguously: row-major per row
        for (int m = 0; m < MR; m++) {
            if (m < mr) {
                // Copy actual row data
                for (int k = 0; k < K; k++) {
                    *Apack++ = A[(m0 + m) * lda + k];
                }
            } else {
                // Zero-pad incomplete tiles
                for (int k = 0; k < K; k++) {
                    *Apack++ = 0;
                }
            }
        }
    }
}

// Pack B matrix for both 5x4 and 6x4 microkernels
// Layout optimized for SDOT instruction
//
// SDOT operates on groups of 4 bytes: for lane i, it computes:
//   result[i] += input1[4i:4i+3] ⋅ input2[4i:4i+3]
//
// For C[m,n] = sum_k A[m,k] * B[n,k], lane i needs:
//   A bytes [4i:4i+3] = [A[m,k], A[m,k+1], A[m,k+2], A[m,k+3]]
//   B bytes [4i:4i+3] = [B[n_i,k], B[n_i,k+1], B[n_i,k+2], B[n_i,k+3]]
//
// Since we have 64 columns (4 vectors × 16 lanes), for each k-group (k, k+1, k+2, k+3):
//   Vector 0 handles columns 0-15
//   Vector 1 handles columns 16-31
//   Vector 2 handles columns 32-47
//   Vector 3 handles columns 48-63
//
// Layout: [K/4][4 vectors][16 columns * 4 bytes]
void pack_B_64x256(const int8_t* B, int ldb, int8_t* Bpack, int N) {
    const int NR = 64;  // 4 SVE vectors × 16 int32 lanes
    const int K = K_FIXED;

    for (int n0 = 0; n0 < N; n0 += NR) {
        int nr = (n0 + NR <= N) ? NR : (N - n0);

        // Pack in k-groups of 4
        for (int k = 0; k < K; k += 4) {
            // Pack 4 vectors (for 64 columns)
            for (int vec = 0; vec < 4; vec++) {
                // Each vector handles 16 columns
                for (int col = 0; col < 16; col++) {
                    int n = n0 + vec * 16 + col;
                    // Pack 4 consecutive k values for this column
                    for (int kk = 0; kk < 4; kk++) {
                        if (n < N && k + kk < K) {
                            *Bpack++ = B[n * ldb + k + kk];
                        } else {
                            *Bpack++ = 0;
                        }
                    }
                }
            }
        }
    }
}
