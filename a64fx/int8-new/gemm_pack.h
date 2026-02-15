#ifndef GEMM_PACK_H
#define GEMM_PACK_H

#include <stdint.h>

// Pack A matrix for 5x4 microkernel
// Input: A[M×K] int8 row-major with leading dimension lda
// Output: Apack[M_tiles][K][5] int8 with Mr=5 rows per tile, zero-padded if M % 5 != 0
// Layout: For each 5-row tile, store K bytes per row contiguously: Apack[m][k] = A[m0+m][k]
void pack_A_5x256(const int8_t* A, int lda, int8_t* Apack, int M);

// Pack A matrix for 6x4 microkernel
// Input: A[M×K] int8 row-major with leading dimension lda
// Output: Apack[M_tiles][K][6] int8 with Mr=6 rows per tile, zero-padded if M % 6 != 0
// Layout: Same as 5x4 but with 6 rows per tile
void pack_A_6x256(const int8_t* A, int lda, int8_t* Apack, int M);

// Pack B matrix for both 5x4 and 6x4 microkernels
// Input: B[N×K] int8 row-major with leading dimension ldb (accessed as B^T in GEMM)
// Output: Bpack[N_tiles][K][64] int8 with Nr=64 columns per tile, K-major order
// Layout: Bpack[k*64 + lane] = B[(n0+lane)*ldb + k] for k=0..K-1, lane=0..63
//         Enables sequential ld1b loads in K-loop: each load gets 64 bytes for all lanes
void pack_B_64x256(const int8_t* B, int ldb, int8_t* Bpack, int N);

#endif // GEMM_PACK_H
