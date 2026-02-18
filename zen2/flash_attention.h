#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

/*
 * FlashAttention-1 with online softmax for Zen2 AVX2.
 *
 * Fused Q×K^T softmax and P×V using 6×16 GEMM microkernels.
 * O(L) memory — never materializes the full L×L attention matrix.
 *
 * Tile sizes: Br=48 (8×MR), Bc=64 (4×NR).
 */

/* Flash attention: O = softmax(Q × K^T × scale) × V
 *
 * Q, K, V: [L, d] row-major float32
 * O:       [L, d] row-major float32 (output)
 * scale:   typically 1/sqrt(d)
 *
 * Constraints:
 *   d must be a multiple of 16 (NR)
 *   d must be a multiple of 8 (AVX2 width)
 */
void flash_attention_fp32(
    const float *Q, const float *K, const float *V,
    float *O,
    int L, int d, float scale);

/* Free internal working buffers (optional cleanup). */
void flash_attention_cleanup(void);

#endif /* FLASH_ATTENTION_H */
