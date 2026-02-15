/*
 * FlashAttention-style Tiled exp2 + GEMM
 * Uses LD1RW instead of DUP for broadcast (LD/ST pipe, not FLA)
 */

#ifndef EXP2_FLASH_TILED_H
#define EXP2_FLASH_TILED_H

#include <stdint.h>

/*
 * exp2_flash_tiled_4x4 - Fused exp2 + GEMM with LD1RW broadcast
 *
 * Computes: O[4][64] = exp2(S * scale - max) @ V
 *
 * Key: Uses LD1RW instead of DUP for broadcast
 *      LD1RW runs on LD/ST pipe (parallel with FMLA on FLA pipe)
 *
 * Parameters:
 *   S:       [4][Nc] int32 attention scores
 *   V:       [Nc][64] float32 values
 *   O:       [4][64] float32 output
 *   P:       [4][Nc] float32 exp2 buffer (for cache/reuse)
 *   Nc:      sequence length (K dimension)
 *   scale:   softmax scale factor
 *   max_val: max value for numerical stability
 *   ld_s:    S row stride in elements
 *   ld_v:    V row stride in bytes (64*4=256)
 *   ld_o:    O row stride in bytes
 */
void exp2_flash_tiled_4x4(
    const int32_t* S,
    const float* V,
    float* O,
    float* P,
    int Nc,
    float scale,
    float max_val,
    int ld_s,
    int ld_v,
    int ld_o
);

/*
 * exp2_flash_ld1rw_4x4 - Optimized version with store-to-load forwarding
 *
 * Same interface as exp2_flash_tiled_4x4
 * Uses store followed by immediate LD1RW from same address
 * This should hit the store buffer for minimal latency
 */
void exp2_flash_ld1rw_4x4(
    const int32_t* S,
    const float* V,
    float* O,
    float* P,
    int Nc,
    float scale,
    float max_val,
    int ld_s,
    int ld_v,
    int ld_o
);

#endif /* EXP2_FLASH_TILED_H */
