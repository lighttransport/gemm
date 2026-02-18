#ifndef FFN_H
#define FFN_H

/*
 * Feed-Forward Network (FFN) layers for transformer inference — Zen2 AVX2.
 *
 * Built on top of gemm_fp32() and AVX2 activation functions.
 *
 * Weight layout: all weight matrices are stored row-major.
 *   W_gate[D_ff × D], W_up[D_ff × D], W_down[D × D_ff]
 *
 * GEMM convention (from gemm.h):
 *   C[M×N] = A[M×K] × B[N×K]^T   (B stored row-major as N×K)
 */

/* Activation type for standard FFN */
typedef enum {
    FFN_ACT_RELU,
    FFN_ACT_GELU,
    FFN_ACT_SILU
} ffn_activation_t;

/*
 * SwiGLU FFN: output = (SiLU(X @ W_gate^T) * (X @ W_up^T)) @ W_down^T
 *
 *   X:      [M, D]      — input activations
 *   W_gate: [D_ff, D]   — gate projection weights
 *   W_up:   [D_ff, D]   — up projection weights
 *   W_down: [D, D_ff]   — down projection weights
 *   output: [M, D]      — output activations
 *
 * Total FLOPs: 6 * M * D * D_ff (activation is negligible).
 */
void ffn_swiglu_fp32(
    const float *X,
    const float *W_gate,
    const float *W_up,
    const float *W_down,
    float *output,
    int M, int D, int D_ff);

/*
 * Standard FFN: output = Activation(X @ W1^T) @ W2^T
 *
 *   X:      [M, D]      — input activations
 *   W1:     [D_ff, D]   — first projection weights
 *   W2:     [D, D_ff]   — second projection weights
 *   output: [M, D]      — output activations
 *
 * Total FLOPs: 4 * M * D * D_ff (activation is negligible).
 */
void ffn_standard_fp32(
    const float *X,
    const float *W1,
    const float *W2,
    float *output,
    int M, int D, int D_ff,
    ffn_activation_t activation);

/* Free persistent internal buffers (optional cleanup). */
void ffn_cleanup(void);

#endif /* FFN_H */
