#ifndef ACTIVATION_H
#define ACTIVATION_H

/*
 * AVX2-vectorized activation functions for Zen2.
 *
 * All functions support:
 *   - In-place operation (output == input)
 *   - Arbitrary n (scalar tail for n % 8 != 0)
 *   - No alignment requirement on input/output pointers
 */

/* SiLU(x) = x * sigmoid(x) */
void silu_avx2(const float *input, float *output, int n);

/* GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
void gelu_avx2(const float *input, float *output, int n);

/* ReLU(x) = max(0, x) */
void relu_avx2(const float *input, float *output, int n);

/* SwiGLU: output[i] = SiLU(gate[i]) * up[i] */
void swiglu_avx2(const float *gate, const float *up, float *output, int n);

#endif /* ACTIVATION_H */
