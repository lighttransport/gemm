/*
 * Flash Attention for AMD Zen2 (Ryzen 9 3950X)
 * 2-pass memory-efficient attention using AVX2 + FMA3
 *
 * Algorithm (Flash Attention style):
 *   Pass 1: S = Q @ K^T, compute row_max
 *   Pass 2: O = softmax(S) @ V (online softmax)
 *
 * Tile parameters optimized for Zen2 cache:
 *   BR = 4   (query block rows)
 *   BC = 64  (key/value block columns)
 *   D  = 64  (head dimension)
 */

#ifndef FLASH_ATTN_AVX2_H
#define FLASH_ATTN_AVX2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tile parameters */
#define FA_BR 4     /* Query block rows */
#define FA_BC 64    /* Key/Value block cols */
#define FA_D  64    /* Head dimension */
#define FA_VL 8     /* AVX2 vector length (floats) */

/*
 * Single-head attention for one tile
 *
 * Q: [BR, D] query block
 * K: [BC, D] key block (row-major, each row is a key vector)
 * V: [BC, D] value block
 * O: [BR, D] output block
 * S_scratch: [BR, BC] scratch for attention scores
 * m: [BR] row max values
 * l: [BR] row sum values
 */
void flash_attention_tile_avx2(
    const float *Q,         /* [BR, D] */
    const float *K,         /* [BC, D] */
    const float *V,         /* [BC, D] */
    float *O,               /* [BR, D] */
    float *S_scratch,       /* [BR, BC] */
    float *m,               /* [BR] */
    float *l                /* [BR] */
);

/*
 * Multi-block attention with online softmax
 * Processes multiple key/value blocks with running max/sum
 *
 * Q: [BR, D] query block
 * K: [seq_len, D] full key sequence
 * V: [seq_len, D] full value sequence
 * O: [BR, D] output block
 * seq_len: total sequence length (must be multiple of BC)
 */
void flash_attention_avx2(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    size_t seq_len
);

/*
 * Full multi-head attention
 *
 * Q: [batch, num_heads, seq_q, D]
 * K: [batch, num_heads, seq_kv, D]
 * V: [batch, num_heads, seq_kv, D]
 * O: [batch, num_heads, seq_q, D]
 */
void multi_head_attention_avx2(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    size_t batch,
    size_t num_heads,
    size_t seq_q,
    size_t seq_kv,
    size_t head_dim
);

/*
 * Reference implementation for correctness verification
 */
void flash_attention_ref(
    const float *Q,
    const float *K,
    const float *V,
    float *O
);

/*
 * AVX2 vectorized exp function
 * Uses polynomial approximation (relative error < 1e-6)
 */
void exp_avx2(const float *input, float *output, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_ATTN_AVX2_H */
