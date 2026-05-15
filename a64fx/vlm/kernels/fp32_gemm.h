/* fp32_gemm.h — FP32 storage GEMM with pre-packed B (BTP layout). */
#ifndef FP32_GEMM_H
#define FP32_GEMM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Same layout/bytes as pack_B_fp32 (also exposed as packed_B_size). */
size_t packed_B_fp32_size(int K, int N);

/* C[M,N] = A[M,K] @ B[K,N], where B is supplied pre-packed via pack_B_fp32. */
void gemm_fp32_BTP(int M, int K, int N,
                   const float *A,   int lda,
                   const float *BTP,
                   float       *C,   int ldc);

/* CMG-aware twin: each OMP thread reads its CMG-local BTP replica. */
void gemm_fp32_BTP_cmg(int M, int K, int N,
                       const float *A, int lda,
                       const float * const BTP_repl[],
                       int n_cmgs,
                       float       *C, int ldc);

#ifdef __cplusplus
}
#endif

#endif /* FP32_GEMM_H */
