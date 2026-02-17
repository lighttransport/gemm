#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

/* Microkernel tile dimensions */
#define MR 6
#define NR 16

/* Cache blocking parameters (from BLIS zen2 config) */
#define MC_DEFAULT 144   /* 24 * MR, A panel = MC*KC*4 bytes */
#define NC_DEFAULT 4080  /* 255 * NR */
#define KC_DEFAULT 256   /* BLIS default KC */

/* Assembly microkernel: C = A * B (overwrite) */
void gemm_kernel_6x16(const float *A_packed, const float *B_packed,
                       float *C, int64_t K, int64_t ldc_bytes);

/* Assembly microkernel: C += A * B (accumulate) */
void gemm_kernel_6x16_accum(const float *A_packed, const float *B_packed,
                             float *C, int64_t K, int64_t ldc_bytes);

/* Assembly microkernel with next-tile A prefetch: C = A * B (overwrite) */
void gemm_kernel_6x16_pf(const float *A_packed, const float *B_packed,
                          float *C, int64_t K, int64_t ldc_bytes,
                          const float *A_next);

/* AVX2 edge kernels for mr < MR: C = A * B (overwrite) */
void gemm_kernel_4x16(const float *A_packed, const float *B_packed,
                       float *C, int64_t K, int64_t ldc_bytes);
void gemm_kernel_2x16(const float *A_packed, const float *B_packed,
                       float *C, int64_t K, int64_t ldc_bytes);

/* Full GEMM driver:
 *   C[M×N] = A[M×K] × B[N×K]^T   (B stored row-major as N×K)
 *   All matrices row-major.
 */
void gemm_fp32(const float *A, int lda,
               const float *B, int ldb,
               float *C, int ldc,
               int M, int N, int K);

/* Free persistent internal pack buffers (optional cleanup). */
void gemm_cleanup(void);

#endif /* GEMM_H */
