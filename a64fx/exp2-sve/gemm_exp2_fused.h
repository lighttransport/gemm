#ifndef GEMM_EXP2_FUSED_H
#define GEMM_EXP2_FUSED_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * gemm_exp2_fused_4x4 - Fused GEMM + exp2 for Flash Attention
 *
 * Computes: C = exp2((A @ B) * scale - max_val)
 *
 * @A:       Packed int8 matrix [K/4][4][4]
 * @B:       Packed int8 matrix [K/4][4][64]
 * @C:       Output fp32 matrix [4][64]
 * @K:       Reduction dimension (must be multiple of 4)
 * @scale:   Scale factor for softmax
 * @max_val: Max value subtracted for numerical stability
 * @ldc:     Leading dimension of C in bytes
 *
 * Performance: GEMM phase followed by exp2 phase
 * Use gemm_exp2_interleaved for pipelined version.
 */
void gemm_exp2_fused_4x4(
    const int8_t* A,
    const int8_t* B,
    float* C,
    int K,
    float scale,
    float max_val,
    int ldc
);

/**
 * gemm_exp2_interleaved - Deeply interleaved GEMM + exp2
 *
 * Pipelined version that processes exp2 on tile N-1 while
 * computing SDOT for tile N. Achieves higher throughput by
 * overlapping SDOT and exp2 FP operations.
 *
 * Same parameters as gemm_exp2_fused_4x4.
 *
 * Note: Output is written row by row as tiles complete.
 * C should have space for (K/4) rows of 64 elements each.
 */
void gemm_exp2_interleaved(
    const int8_t* A,
    const int8_t* B,
    float* C,
    int K,
    float scale,
    float max_val,
    int ldc
);

#ifdef __cplusplus
}
#endif

#endif /* GEMM_EXP2_FUSED_H */
