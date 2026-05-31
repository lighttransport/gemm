/*
 * Neural Network Operations for AMD Zen2 (Ryzen 9 3950X)
 * Optimized with AVX2 + FMA3 intrinsics
 *
 * Includes:
 *   - RMSNorm (Root Mean Square Normalization)
 *   - LayerNorm (Layer Normalization)
 *   - FFN (Feed-Forward Network) with various activations
 *   - Activation functions: GELU, SiLU, QuickGELU, SwiGLU
 */

#ifndef NN_OPS_AVX2_H
#define NN_OPS_AVX2_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================
 * RMSNorm: output[i] = input[i] * gamma[i] / sqrt(mean(input^2) + eps)
 * ============================================ */

void rmsnorm_f32_avx2(
    const float *input,
    const float *gamma,
    float *output,
    size_t dim,
    float eps
);

/* Batched version */
void rmsnorm_f32_batch_avx2(
    const float *input,     /* [batch, dim] */
    const float *gamma,     /* [dim] */
    float *output,          /* [batch, dim] */
    size_t batch,
    size_t dim,
    float eps
);

/* ============================================
 * LayerNorm: output[i] = (input[i] - mean) * gamma[i] / sqrt(var + eps) + beta[i]
 * ============================================ */

void layernorm_f32_avx2(
    const float *input,
    const float *gamma,
    const float *beta,
    float *output,
    size_t dim,
    float eps
);

/* Batched version */
void layernorm_f32_batch_avx2(
    const float *input,     /* [batch, dim] */
    const float *gamma,     /* [dim] */
    const float *beta,      /* [dim] */
    float *output,          /* [batch, dim] */
    size_t batch,
    size_t dim,
    float eps
);

/* ============================================
 * Activation Functions
 * ============================================ */

/* GELU exact: 0.5 * x * (1 + erf(x / sqrt(2))) */
void gelu_f32_avx2(const float *input, float *output, size_t n);

/* GELU tanh approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
void gelu_tanh_f32_avx2(const float *input, float *output, size_t n);

/* SiLU (Swish): x * sigmoid(x) */
void silu_f32_avx2(const float *input, float *output, size_t n);

/* QuickGELU: x * sigmoid(1.702 * x) */
void quickgelu_f32_avx2(const float *input, float *output, size_t n);

/* Sigmoid: 1 / (1 + exp(-x)) */
void sigmoid_f32_avx2(const float *input, float *output, size_t n);

/* ReLU: max(0, x) */
void relu_f32_avx2(const float *input, float *output, size_t n);

/* ============================================
 * FFN (Feed-Forward Network)
 * ============================================
 * Standard FFN: output = W2 @ activation(W1 @ input + b1) + b2
 * Gated FFN (SwiGLU): output = W2 @ (SiLU(W_gate @ x) * (W_up @ x)) + b2
 */

/* Standard FFN with configurable activation */
typedef enum {
    FFN_ACT_GELU,
    FFN_ACT_GELU_TANH,
    FFN_ACT_SILU,
    FFN_ACT_RELU
} ffn_activation_t;

void ffn_f32_avx2(
    const float *input,     /* [batch, in_dim] */
    const float *W1,        /* [in_dim, hidden_dim] */
    const float *b1,        /* [hidden_dim] or NULL */
    const float *W2,        /* [hidden_dim, out_dim] */
    const float *b2,        /* [out_dim] or NULL */
    float *output,          /* [batch, out_dim] */
    float *hidden_scratch,  /* [batch, hidden_dim] */
    size_t batch,
    size_t in_dim,
    size_t hidden_dim,
    size_t out_dim,
    ffn_activation_t activation
);

/* SwiGLU FFN (used in LLaMA, Mistral, etc.)
 * output = W_down @ (SiLU(W_gate @ x) * (W_up @ x))
 */
void ffn_swiglu_f32_avx2(
    const float *input,     /* [batch, in_dim] */
    const float *W_gate,    /* [in_dim, hidden_dim] */
    const float *W_up,      /* [in_dim, hidden_dim] */
    const float *W_down,    /* [hidden_dim, out_dim] */
    float *output,          /* [batch, out_dim] */
    float *gate_scratch,    /* [batch, hidden_dim] */
    float *up_scratch,      /* [batch, hidden_dim] */
    size_t batch,
    size_t in_dim,
    size_t hidden_dim,
    size_t out_dim
);

/* ============================================
 * Reference implementations
 * ============================================ */

void rmsnorm_f32_ref(
    const float *input,
    const float *gamma,
    float *output,
    size_t dim,
    float eps
);

void layernorm_f32_ref(
    const float *input,
    const float *gamma,
    const float *beta,
    float *output,
    size_t dim,
    float eps
);

void gelu_f32_ref(const float *input, float *output, size_t n);
void silu_f32_ref(const float *input, float *output, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* NN_OPS_AVX2_H */
