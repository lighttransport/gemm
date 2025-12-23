/*
 * GEMM kernels optimized for AMD Zen2 (Ryzen 9 3950X)
 * Using AVX2 + FMA3 intrinsics
 *
 * Target architecture:
 *   - 256-bit AVX2 vectors (8 floats per YMM register)
 *   - FMA3 fused multiply-add instructions
 *   - 16 YMM registers available
 *   - L1D: 32KB, L2: 512KB per core
 */

#ifndef GEMM_AVX2_H
#define GEMM_AVX2_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Single-precision GEMM: C = alpha * A * B + beta * C
 *
 * A: M x K matrix (row-major)
 * B: K x N matrix (row-major)
 * C: M x N matrix (row-major)
 *
 * lda, ldb, ldc: leading dimensions (stride between rows)
 */
void sgemm_avx2(
    size_t M, size_t N, size_t K,
    float alpha,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    float beta,
    float *C, size_t ldc
);

/*
 * Optimized micro-kernel for 6x16 tile
 * Computes a 6x16 block of C += A * B
 *
 * A_panel: 6 x K (packed)
 * B_panel: K x 16 (packed)
 * C: 6 x 16 output block
 */
void sgemm_kernel_6x16(
    size_t K,
    const float *A_panel,
    const float *B_panel,
    float *C, size_t ldc,
    float alpha, float beta
);

/*
 * Reference naive implementation for correctness checking
 */
void sgemm_naive(
    size_t M, size_t N, size_t K,
    float alpha,
    const float *A, size_t lda,
    const float *B, size_t ldb,
    float beta,
    float *C, size_t ldc
);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_AVX2_H */
