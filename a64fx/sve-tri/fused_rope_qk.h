#ifndef FUSED_ROPE_QK_H
#define FUSED_ROPE_QK_H

#include <stdint.h>
#include <stddef.h>

/*
 * Fused projection + RoPE for Q and K
 *
 * Single-pass kernel: Q = RoPE(Wq·X), K = RoPE(Wk·X)
 * Shares X loads between Q and K projections.
 *
 * Micro-kernel tile: MR=4 tokens × NR=32 output dims (2 SVE vectors).
 * The C driver tiles over tokens (blocks of 4) and output dims (blocks of 32).
 */

#define FUSED_MR 4
#define FUSED_NR 32   /* 2 × 16 (SVE VL=512 → 16 fp32/vector) */

/* Assembly micro-kernel */
void micro_fused_proj_rope_4x2(
    const float *X_packed,   /* [d_model × MR] packed */
    const float *Wq,         /* [d_model × 32] packed */
    const float *Wk,         /* [d_model × 32] packed */
    float *Q_out,            /* output row pointer */
    float *K_out,            /* output row pointer */
    int64_t d_model,         /* inner dimension */
    const float *theta,      /* [16] RoPE angles for this dim block */
    int64_t ldc_bytes        /* output row stride in bytes */
);

/*
 * High-level API
 *
 * Computes:
 *   Q[i, :] = RoPE(Wq · X[i, :], theta, i)   for i=0..n_tokens-1
 *   K[i, :] = RoPE(Wk · X[i, :], theta, i)   for i=0..n_tokens-1
 *
 * X:     [n_tokens × d_model] row-major input
 * Wq,Wk: [d_model × d_head] row-major weights (will be packed internally)
 * Q,K:   [n_tokens × d_head] row-major output
 * theta: [d_head/2] base RoPE angles (e.g. 1/10000^(2i/d_head))
 *
 * Position offsets: Q/K for token t uses angles theta[j] * t.
 * The pos array [n_tokens] gives the position index for each token.
 */
void fused_proj_rope_f32(
    const float *X,
    const float *Wq,
    const float *Wk,
    float *Q, float *K,
    const float *theta,
    const int *pos,
    int n_tokens, int d_model, int d_head);

/* Returns allocation size (bytes) for packed weight with padding */
size_t packed_weight_size(int d_model, int d_head);

/* Pack weight matrix from row-major [d_model × d_head]
 * to blocked [d_head/32 × (d_model+1) × 32] layout with padding */
void pack_weight_f32(const float *W_rm, float *W_packed,
                     int d_model, int d_head);

/*
 * Optimized path using assembly micro-kernel.
 *
 * theta_scaled must be pre-multiplied by position:
 *   theta_scaled[j] = theta_base[j] * position
 *
 * All tokens in each MR=4 block get the same RoPE angles.
 * Correct for: decode (1 token), or same-position batches.
 * Wq_packed, Wk_packed: output of pack_weight_f32.
 */
void fused_proj_rope_asm_f32(
    const float *X,
    const float *Wq_packed,
    const float *Wk_packed,
    float *Q, float *K,
    const float *theta_scaled,
    int n_tokens, int d_model, int d_head);

#endif /* FUSED_ROPE_QK_H */
