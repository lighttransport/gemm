#ifndef GEMM_PACK_H
#define GEMM_PACK_H

#include "gemm.h"

/*
 * Pack a panel of A[mc × K] into column-panel format.
 * A is row-major with leading dimension lda.
 * A_pack layout: for each k in [0,K), stores MR floats contiguously.
 *   A_pack[k * MR + m] = A[m * lda + k]
 * mc <= MR; if mc < MR, remaining rows are zero-padded.
 */
void pack_a(const float *A, int lda, float *A_pack, int mc, int K);

/*
 * Pack a panel of B[nc × K] into row-panel format.
 * B is row-major with leading dimension ldb (B is N×K, i.e., each row is a K-vector).
 * B_pack layout: for each k in [0,K), stores NR floats contiguously.
 *   B_pack[k * NR + n] = B[n * ldb + k]
 * nc <= NR; if nc < NR, remaining cols are zero-padded.
 */
void pack_b(const float *B, int ldb, float *B_pack, int nc, int K);

#endif /* GEMM_PACK_H */
