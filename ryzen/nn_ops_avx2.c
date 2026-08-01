/*
 * Neural Network Operations for AMD Zen2 using AVX2 + FMA3
 *
 * Optimizations:
 *   - AVX2 vectorized operations (8 floats per YMM register)
 *   - FMA3 fused multiply-add instructions
 *   - Fast rsqrt using Newton-Raphson refinement
 *   - Polynomial approximations for transcendental functions
 */

#include "nn_ops_avx2.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ============================================
 * Helper: Horizontal sum of __m256
 * ============================================ */

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 sum2 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128 sum1 = _mm_add_ps(sum2, _mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(sum1);
}

/* ============================================
 * Helper: Fast rsqrt with Newton-Raphson
 * x' = 0.5 * x * (3 - a * x * x)
 * ============================================ */

static inline __m256 rsqrt_nr(__m256 a) {
    __m256 x = _mm256_rsqrt_ps(a);
    /* One Newton-Raphson iteration for better precision */
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 three = _mm256_set1_ps(3.0f);
    __m256 ax2 = _mm256_mul_ps(a, _mm256_mul_ps(x, x));
    __m256 factor = _mm256_sub_ps(three, ax2);
    return _mm256_mul_ps(_mm256_mul_ps(half, x), factor);
}

/* Two Newton-Raphson iterations for higher precision */
static inline __m256 rsqrt_nr2(__m256 a) {
    __m256 x = _mm256_rsqrt_ps(a);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 three = _mm256_set1_ps(3.0f);
    /* First iteration */
    __m256 ax2 = _mm256_mul_ps(a, _mm256_mul_ps(x, x));
    x = _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_sub_ps(three, ax2));
    /* Second iteration */
    ax2 = _mm256_mul_ps(a, _mm256_mul_ps(x, x));
    return _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_sub_ps(three, ax2));
}

/* ============================================
 * Helper: Fast exp approximation
 * ============================================ */

static inline __m256 exp256_ps(__m256 x) {
    const float log2e = 1.44269504088896341f;

    /* Clamp input */
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.33654474f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.72283905f));

    /* exp(x) = 2^(x * log2(e)) */
    __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(log2e));
    __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(t, n);

    /* Polynomial for 2^f on [-0.5, 0.5] */
    __m256 p = _mm256_set1_ps(0.0013333558146428443f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.00967812610747928f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.05550410866482158f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.2402265069591007f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.6931471805599453f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));

    /* Scale by 2^n */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(p, pow2n);
}

/* ============================================
 * Helper: Sigmoid
 * sigmoid(x) = 1 / (1 + exp(-x))
 * ============================================ */

static inline __m256 sigmoid256_ps(__m256 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg_x = exp256_ps(neg_x);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 denom = _mm256_add_ps(one, exp_neg_x);
    /* Fast reciprocal with Newton-Raphson */
    __m256 rcp = _mm256_rcp_ps(denom);
    /* One NR iteration: rcp = rcp * (2 - denom * rcp) */
    __m256 two = _mm256_set1_ps(2.0f);
    rcp = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));
    return rcp;
}

/* ============================================
 * Helper: tanh
 * tanh(x) = 2 * sigmoid(2x) - 1
 * ============================================ */

static inline __m256 tanh256_ps(__m256 x) {
    __m256 two_x = _mm256_add_ps(x, x);
    __m256 sig = sigmoid256_ps(two_x);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_fmsub_ps(two, sig, one);
}

/* ============================================
 * Helper: erf approximation (Abramowitz & Stegun)
 * ============================================ */

static inline __m256 erf256_ps(__m256 x) {
    const float p = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    /* Sign and absolute value */
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 sign = _mm256_and_ps(x, sign_mask);
    __m256 ax = _mm256_andnot_ps(sign_mask, x);

    /* t = 1 / (1 + p * |x|) */
    __m256 denom = _mm256_fmadd_ps(ax, _mm256_set1_ps(p), _mm256_set1_ps(1.0f));
    __m256 t = _mm256_rcp_ps(denom);
    t = _mm256_mul_ps(t, _mm256_fnmadd_ps(denom, t, _mm256_set1_ps(2.0f)));

    /* Polynomial */
    __m256 poly = _mm256_set1_ps(a5);
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(a4));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(a3));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(a2));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(a1));
    poly = _mm256_mul_ps(poly, t);

    /* exp(-x^2) */
    __m256 x2 = _mm256_mul_ps(ax, ax);
    __m256 neg_x2 = _mm256_sub_ps(_mm256_setzero_ps(), x2);
    __m256 exp_neg_x2 = exp256_ps(neg_x2);

    /* erf = sign * (1 - poly * exp(-x^2)) */
    __m256 result = _mm256_fnmadd_ps(poly, exp_neg_x2, _mm256_set1_ps(1.0f));
    return _mm256_or_ps(result, sign);
}

/* ============================================
 * RMSNorm
 * output[i] = input[i] * gamma[i] / sqrt(mean(input^2) + eps)
 * ============================================ */

void rmsnorm_f32_avx2(
    const float *input,
    const float *gamma,
    float *output,
    size_t dim,
    float eps
) {
    /* Step 1: Compute sum of squares */
    __m256 sum_sq = _mm256_setzero_ps();
    size_t i = 0;

    /* Unroll by 4 for better throughput */
    for (; i + 32 <= dim; i += 32) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        __m256 x2 = _mm256_loadu_ps(input + i + 16);
        __m256 x3 = _mm256_loadu_ps(input + i + 24);

        sum_sq = _mm256_fmadd_ps(x0, x0, sum_sq);
        sum_sq = _mm256_fmadd_ps(x1, x1, sum_sq);
        sum_sq = _mm256_fmadd_ps(x2, x2, sum_sq);
        sum_sq = _mm256_fmadd_ps(x3, x3, sum_sq);
    }

    for (; i + 8 <= dim; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        sum_sq = _mm256_fmadd_ps(x, x, sum_sq);
    }

    /* Handle remainder */
    float total_sq = hsum256(sum_sq);
    for (; i < dim; i++) {
        total_sq += input[i] * input[i];
    }

    /* Step 2: Compute rsqrt(mean_sq + eps) */
    float mean_sq = total_sq / (float)dim;
    float inv_std = 1.0f / sqrtf(mean_sq + eps);
    __m256 vinv_std = _mm256_set1_ps(inv_std);

    /* Step 3: Normalize and scale */
    i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        __m256 x2 = _mm256_loadu_ps(input + i + 16);
        __m256 x3 = _mm256_loadu_ps(input + i + 24);

        __m256 g0 = _mm256_loadu_ps(gamma + i);
        __m256 g1 = _mm256_loadu_ps(gamma + i + 8);
        __m256 g2 = _mm256_loadu_ps(gamma + i + 16);
        __m256 g3 = _mm256_loadu_ps(gamma + i + 24);

        __m256 y0 = _mm256_mul_ps(_mm256_mul_ps(x0, vinv_std), g0);
        __m256 y1 = _mm256_mul_ps(_mm256_mul_ps(x1, vinv_std), g1);
        __m256 y2 = _mm256_mul_ps(_mm256_mul_ps(x2, vinv_std), g2);
        __m256 y3 = _mm256_mul_ps(_mm256_mul_ps(x3, vinv_std), g3);

        _mm256_storeu_ps(output + i, y0);
        _mm256_storeu_ps(output + i + 8, y1);
        _mm256_storeu_ps(output + i + 16, y2);
        _mm256_storeu_ps(output + i + 24, y3);
    }

    for (; i + 8 <= dim; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 g = _mm256_loadu_ps(gamma + i);
        __m256 y = _mm256_mul_ps(_mm256_mul_ps(x, vinv_std), g);
        _mm256_storeu_ps(output + i, y);
    }

    for (; i < dim; i++) {
        output[i] = input[i] * inv_std * gamma[i];
    }
}

void rmsnorm_f32_batch_avx2(
    const float *input,
    const float *gamma,
    float *output,
    size_t batch,
    size_t dim,
    float eps
) {
    for (size_t b = 0; b < batch; b++) {
        rmsnorm_f32_avx2(
            input + b * dim,
            gamma,
            output + b * dim,
            dim,
            eps
        );
    }
}

/* ============================================
 * LayerNorm
 * output[i] = (input[i] - mean) * gamma[i] / sqrt(var + eps) + beta[i]
 * ============================================ */

void layernorm_f32_avx2(
    const float *input,
    const float *gamma,
    const float *beta,
    float *output,
    size_t dim,
    float eps
) {
    /* Step 1: Compute mean */
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 32 <= dim; i += 32) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        __m256 x2 = _mm256_loadu_ps(input + i + 16);
        __m256 x3 = _mm256_loadu_ps(input + i + 24);
        vsum = _mm256_add_ps(vsum, x0);
        vsum = _mm256_add_ps(vsum, x1);
        vsum = _mm256_add_ps(vsum, x2);
        vsum = _mm256_add_ps(vsum, x3);
    }

    for (; i + 8 <= dim; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        vsum = _mm256_add_ps(vsum, x);
    }

    float total = hsum256(vsum);
    for (; i < dim; i++) {
        total += input[i];
    }

    float mean = total / (float)dim;
    __m256 vmean = _mm256_set1_ps(mean);

    /* Step 2: Compute variance */
    __m256 vvar = _mm256_setzero_ps();
    i = 0;

    for (; i + 32 <= dim; i += 32) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        __m256 x2 = _mm256_loadu_ps(input + i + 16);
        __m256 x3 = _mm256_loadu_ps(input + i + 24);

        __m256 d0 = _mm256_sub_ps(x0, vmean);
        __m256 d1 = _mm256_sub_ps(x1, vmean);
        __m256 d2 = _mm256_sub_ps(x2, vmean);
        __m256 d3 = _mm256_sub_ps(x3, vmean);

        vvar = _mm256_fmadd_ps(d0, d0, vvar);
        vvar = _mm256_fmadd_ps(d1, d1, vvar);
        vvar = _mm256_fmadd_ps(d2, d2, vvar);
        vvar = _mm256_fmadd_ps(d3, d3, vvar);
    }

    for (; i + 8 <= dim; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 d = _mm256_sub_ps(x, vmean);
        vvar = _mm256_fmadd_ps(d, d, vvar);
    }

    float var_sum = hsum256(vvar);
    for (; i < dim; i++) {
        float d = input[i] - mean;
        var_sum += d * d;
    }

    float variance = var_sum / (float)dim;
    float inv_std = 1.0f / sqrtf(variance + eps);
    __m256 vinv_std = _mm256_set1_ps(inv_std);

    /* Step 3: Normalize, scale, and shift */
    i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 g = _mm256_loadu_ps(gamma + i);
        __m256 b = _mm256_loadu_ps(beta + i);

        __m256 d = _mm256_sub_ps(x, vmean);
        __m256 normed = _mm256_mul_ps(d, vinv_std);
        __m256 y = _mm256_fmadd_ps(normed, g, b);

        _mm256_storeu_ps(output + i, y);
    }

    for (; i < dim; i++) {
        float normed = (input[i] - mean) * inv_std;
        output[i] = normed * gamma[i] + beta[i];
    }
}

void layernorm_f32_batch_avx2(
    const float *input,
    const float *gamma,
    const float *beta,
    float *output,
    size_t batch,
    size_t dim,
    float eps
) {
    for (size_t b = 0; b < batch; b++) {
        layernorm_f32_avx2(
            input + b * dim,
            gamma,
            beta,
            output + b * dim,
            dim,
            eps
        );
    }
}

/* ============================================
 * GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
 * ============================================ */

void gelu_f32_avx2(const float *input, float *output, size_t n) {
    const float inv_sqrt2 = 0.7071067811865476f;
    __m256 v_inv_sqrt2 = _mm256_set1_ps(inv_sqrt2);
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_one = _mm256_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 x_scaled = _mm256_mul_ps(x, v_inv_sqrt2);
        __m256 erf_val = erf256_ps(x_scaled);
        __m256 one_plus_erf = _mm256_add_ps(v_one, erf_val);
        __m256 result = _mm256_mul_ps(x, _mm256_mul_ps(v_half, one_plus_erf));
        _mm256_storeu_ps(output + i, result);
    }

    for (; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt2));
    }
}

/* ============================================
 * GELU tanh approx
 * ============================================ */

void gelu_tanh_f32_avx2(const float *input, float *output, size_t n) {
    const float sqrt_2_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    __m256 v_sqrt_2_pi = _mm256_set1_ps(sqrt_2_pi);
    __m256 v_coef = _mm256_set1_ps(coef);
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_one = _mm256_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);

        /* x + 0.044715 * x^3 */
        __m256 inner = _mm256_fmadd_ps(x3, v_coef, x);
        /* sqrt(2/pi) * inner */
        inner = _mm256_mul_ps(inner, v_sqrt_2_pi);
        /* tanh */
        __m256 tanh_val = tanh256_ps(inner);
        /* 0.5 * x * (1 + tanh) */
        __m256 one_plus_tanh = _mm256_add_ps(v_one, tanh_val);
        __m256 result = _mm256_mul_ps(x, _mm256_mul_ps(v_half, one_plus_tanh));

        _mm256_storeu_ps(output + i, result);
    }

    for (; i < n; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_pi * (x + coef * x3);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/* ============================================
 * SiLU (Swish): x * sigmoid(x)
 * ============================================ */

void silu_f32_avx2(const float *input, float *output, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 sig = sigmoid256_ps(x);
        __m256 result = _mm256_mul_ps(x, sig);
        _mm256_storeu_ps(output + i, result);
    }

    for (; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

/* ============================================
 * QuickGELU: x * sigmoid(1.702 * x)
 * ============================================ */

void quickgelu_f32_avx2(const float *input, float *output, size_t n) {
    const float alpha = 1.702f;
    __m256 v_alpha = _mm256_set1_ps(alpha);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 scaled = _mm256_mul_ps(x, v_alpha);
        __m256 sig = sigmoid256_ps(scaled);
        __m256 result = _mm256_mul_ps(x, sig);
        _mm256_storeu_ps(output + i, result);
    }

    for (; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-alpha * x));
    }
}

/* ============================================
 * Sigmoid: 1 / (1 + exp(-x))
 * ============================================ */

void sigmoid_f32_avx2(const float *input, float *output, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 result = sigmoid256_ps(x);
        _mm256_storeu_ps(output + i, result);
    }

    for (; i < n; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

/* ============================================
 * ReLU: max(0, x)
 * ============================================ */

void relu_f32_avx2(const float *input, float *output, size_t n) {
    __m256 zero = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 result = _mm256_max_ps(zero, x);
        _mm256_storeu_ps(output + i, result);
    }

    for (; i < n; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

/* ============================================
 * Simple GEMV for FFN: y = A @ x + b
 * A: [M, K], x: [K], y: [M]
 * ============================================ */

static void gemv_f32_avx2(
    const float *A,
    const float *x,
    const float *bias,  /* can be NULL */
    float *y,
    size_t M,
    size_t K
) {
    for (size_t i = 0; i < M; i++) {
        __m256 sum = _mm256_setzero_ps();
        const float *row = A + i * K;

        size_t k = 0;
        for (; k + 8 <= K; k += 8) {
            __m256 a = _mm256_loadu_ps(row + k);
            __m256 xv = _mm256_loadu_ps(x + k);
            sum = _mm256_fmadd_ps(a, xv, sum);
        }

        float result = hsum256(sum);
        for (; k < K; k++) {
            result += row[k] * x[k];
        }

        y[i] = bias ? result + bias[i] : result;
    }
}

/* ============================================
 * FFN with configurable activation
 * ============================================ */

void ffn_f32_avx2(
    const float *input,
    const float *W1,
    const float *b1,
    const float *W2,
    const float *b2,
    float *output,
    float *hidden,
    size_t batch,
    size_t in_dim,
    size_t hidden_dim,
    size_t out_dim,
    ffn_activation_t activation
) {
    for (size_t b = 0; b < batch; b++) {
        const float *x = input + b * in_dim;
        float *h = hidden + b * hidden_dim;
        float *y = output + b * out_dim;

        /* hidden = W1 @ x + b1 */
        gemv_f32_avx2(W1, x, b1, h, hidden_dim, in_dim);

        /* Apply activation */
        switch (activation) {
            case FFN_ACT_GELU:
                gelu_f32_avx2(h, h, hidden_dim);
                break;
            case FFN_ACT_GELU_TANH:
                gelu_tanh_f32_avx2(h, h, hidden_dim);
                break;
            case FFN_ACT_SILU:
                silu_f32_avx2(h, h, hidden_dim);
                break;
            case FFN_ACT_RELU:
                relu_f32_avx2(h, h, hidden_dim);
                break;
        }

        /* output = W2 @ hidden + b2 */
        gemv_f32_avx2(W2, h, b2, y, out_dim, hidden_dim);
    }
}

/* ============================================
 * SwiGLU FFN: output = W_down @ (SiLU(W_gate @ x) * (W_up @ x))
 * ============================================ */

void ffn_swiglu_f32_avx2(
    const float *input,
    const float *W_gate,
    const float *W_up,
    const float *W_down,
    float *output,
    float *gate_scratch,
    float *up_scratch,
    size_t batch,
    size_t in_dim,
    size_t hidden_dim,
    size_t out_dim
) {
    for (size_t b = 0; b < batch; b++) {
        const float *x = input + b * in_dim;
        float *gate = gate_scratch + b * hidden_dim;
        float *up = up_scratch + b * hidden_dim;
        float *y = output + b * out_dim;

        /* gate = W_gate @ x */
        gemv_f32_avx2(W_gate, x, NULL, gate, hidden_dim, in_dim);

        /* up = W_up @ x */
        gemv_f32_avx2(W_up, x, NULL, up, hidden_dim, in_dim);

        /* gate = SiLU(gate) * up */
        size_t i = 0;
        for (; i + 8 <= hidden_dim; i += 8) {
            __m256 g = _mm256_loadu_ps(gate + i);
            __m256 u = _mm256_loadu_ps(up + i);
            __m256 sig = sigmoid256_ps(g);
            __m256 silu_g = _mm256_mul_ps(g, sig);
            __m256 result = _mm256_mul_ps(silu_g, u);
            _mm256_storeu_ps(gate + i, result);
        }
        for (; i < hidden_dim; i++) {
            float g = gate[i];
            float sig = 1.0f / (1.0f + expf(-g));
            gate[i] = g * sig * up[i];
        }

        /* output = W_down @ gate */
        gemv_f32_avx2(W_down, gate, NULL, y, out_dim, hidden_dim);
    }
}

/* ============================================
 * Reference implementations
 * ============================================ */

void rmsnorm_f32_ref(
    const float *input,
    const float *gamma,
    float *output,
    size_t dim,
    float eps
) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum_sq += input[i] * input[i];
    }
    float mean_sq = sum_sq / (float)dim;
    float inv_std = 1.0f / sqrtf(mean_sq + eps);
    for (size_t i = 0; i < dim; i++) {
        output[i] = input[i] * inv_std * gamma[i];
    }
}

void layernorm_f32_ref(
    const float *input,
    const float *gamma,
    const float *beta,
    float *output,
    size_t dim,
    float eps
) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum += input[i];
    }
    float mean = sum / (float)dim;

    float var_sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = input[i] - mean;
        var_sum += d * d;
    }
    float variance = var_sum / (float)dim;
    float inv_std = 1.0f / sqrtf(variance + eps);

    for (size_t i = 0; i < dim; i++) {
        float normed = (input[i] - mean) * inv_std;
        output[i] = normed * gamma[i] + beta[i];
    }
}

void gelu_f32_ref(const float *input, float *output, size_t n) {
    const float inv_sqrt2 = 0.7071067811865476f;
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt2));
    }
}

void silu_f32_ref(const float *input, float *output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}
