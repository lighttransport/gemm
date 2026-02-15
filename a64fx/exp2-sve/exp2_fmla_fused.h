#ifndef EXP2_FMLA_FUSED_H
#define EXP2_FMLA_FUSED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * exp2_fmla_fp32_4x4 - Fused exp2 + FP32 FMLA for Flash Attention Stage 2
 *
 * Computes: O = exp2(S * scale - max) @ V
 *
 * @S:       Attention scores [4][Nc] (int32, row-major)
 * @V:       Value matrix [Nc][64] (fp32, row-major)
 * @O:       Output [4][64] (fp32, row-major)
 * @Nc:      Inner dimension (sequence length for attention)
 * @scale:   Softmax scale factor (typically 1/sqrt(d))
 * @max_val: Max value for numerical stability
 * @ld_s:    Leading dimension of S in elements
 * @ld_v:    Leading dimension of V in bytes
 * @ld_o:    Leading dimension of O in bytes
 */
void exp2_fmla_fp32_4x4(
    const int32_t* S,
    const float* V,
    float* O,
    int Nc,
    float scale,
    float max_val,
    int ld_s,
    int ld_v,
    int ld_o
);

/**
 * exp2_fmla_fp32_vec - Vectorized version for 16x64 output tile
 */
void exp2_fmla_fp32_vec(
    const int32_t* S,
    const float* V,
    float* O,
    int Nc,
    float scale,
    float max_val,
    int ld_s,
    int ld_v,
    int ld_o
);

/**
 * exp2_fmla_fp16_4x4 - FP16 version for Flash Attention Stage 2
 *
 * exp2 computed in fp32 for accuracy, then converted to fp16 for FMLA
 *
 * @V:  Value matrix [Nc][128] (fp16, 4 SVE vectors of 32 elements)
 * @O:  Output [4][128] (fp16)
 */
void exp2_fmla_fp16_4x4(
    const int32_t* S,
    const _Float16* V,
    _Float16* O,
    int Nc,
    float scale,
    float max_val,
    int ld_s,
    int ld_v,
    int ld_o
);

/**
 * Two-pass optimized: exp2 then GEMM
 * Pass 1: exp2_rows - compute exp2 for M rows of S
 * Pass 2: gemm_fp32_4x4 - standard GEMM
 */

/**
 * exp2_rows - Compute exp2 for M×Nc matrix (row-major)
 * Uses vectorized exp2_softmax_fast internally
 */
void exp2_rows(
    const int32_t* S,  // [M][Nc] input
    float* P,          // [M][Nc] output
    int M,
    int Nc,
    float scale,
    float max_val,
    int ld_s,          // S leading dim in elements
    int ld_p           // P leading dim in elements
);

/**
 * gemm_fp32_4x4 - FP32 GEMM: O += A @ B
 * A: [4][K] row-major
 * B: [K][64] row-major (4 SVE vectors)
 * O: [4][64] row-major
 */
void gemm_fp32_4x4(
    const float* A,    // [4][K]
    const float* B,    // [K][64]
    float* O,          // [4][64]
    int K,
    int ld_a,          // A leading dim in bytes
    int ld_b,          // B leading dim in bytes
    int ld_o           // O leading dim in bytes
);

#ifdef __cplusplus
}
#endif

#endif /* EXP2_FMLA_FUSED_H */
