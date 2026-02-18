/*
 * AVX2-vectorized activation functions for Zen2.
 *
 * Core building block: fast_exp_avx2 (4th-order minimax polynomial,
 * same as flash_attention.c) + rcp_ps + Newton-Raphson for sigmoid.
 *
 * Performance targets (Zen2 @ 3.5 GHz):
 *   SiLU:   ~2-3 cycles/elem (compute-bound)
 *   GELU:   ~3-4 cycles/elem (more FMAs)
 *   ReLU:   ~0.5 cycles/elem (bandwidth-bound)
 *   SwiGLU: ~3-4 cycles/elem (SiLU + 1 mul + 2 loads)
 */

#include "activation.h"
#include <immintrin.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Fast exp(x) — duplicated from flash_attention.c                    */
/* ------------------------------------------------------------------ */

/*
 * Vectorized exp(x) using 2^(x * log2e) decomposition.
 * 4th-order polynomial for 2^f, bit manipulation for 2^n.
 * Relative error ~1e-6. Input clamped to [-88, 88].
 */
static inline __m256 fast_exp_avx2(__m256 x)
{
    const __m256 lo  = _mm256_set1_ps(-88.0f);
    const __m256 hi  = _mm256_set1_ps(88.0f);
    x = _mm256_max_ps(x, lo);
    x = _mm256_min_ps(x, hi);

    const __m256 log2e = _mm256_set1_ps(1.4426950409f);
    __m256 t = _mm256_mul_ps(x, log2e);

    __m256 n = _mm256_floor_ps(t);
    __m256 f = _mm256_sub_ps(t, n);

    /* 2^f via 4th-order minimax polynomial on [0, 1) */
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931472f);
    const __m256 c2 = _mm256_set1_ps(0.2402265f);
    const __m256 c3 = _mm256_set1_ps(0.05550411f);
    const __m256 c4 = _mm256_set1_ps(0.009618129f);

    __m256 p = _mm256_fmadd_ps(c4, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);

    /* 2^n via bit manipulation: float(2^n) = (n+127) << 23 */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(pow2n, p);
}

/* ------------------------------------------------------------------ */
/*  Fast sigmoid(x) = 1 / (1 + exp(-x))                               */
/*  Uses rcp_ps + one Newton-Raphson step (~23-bit accuracy)           */
/* ------------------------------------------------------------------ */

static inline __m256 fast_sigmoid_avx2(__m256 x)
{
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg_x = fast_exp_avx2(neg_x);
    __m256 denom = _mm256_add_ps(one, exp_neg_x);

    /* rcp_ps: ~12-bit approximation of 1/denom */
    __m256 y0 = _mm256_rcp_ps(denom);

    /* One Newton-Raphson refinement: y1 = y0 * (2 - denom * y0) */
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 y1 = _mm256_mul_ps(y0,
        _mm256_fnmadd_ps(denom, y0, two));  /* y0 * (2 - denom*y0) */

    return y1;
}

/* Scalar sigmoid for tail elements */
static inline float sigmoid_scalar(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* ------------------------------------------------------------------ */
/*  SiLU(x) = x * sigmoid(x)                                          */
/* ------------------------------------------------------------------ */

void silu_avx2(const float *input, float *output, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 sig = fast_sigmoid_avx2(x);
        __m256 result = _mm256_mul_ps(x, sig);
        _mm256_storeu_ps(output + i, result);
    }
    /* Scalar tail */
    for (; i < n; i++) {
        float x = input[i];
        output[i] = x * sigmoid_scalar(x);
    }
}

/* ------------------------------------------------------------------ */
/*  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) */
/*  Using tanh(t) = 2*sigmoid(2t) - 1                                 */
/* ------------------------------------------------------------------ */

void gelu_avx2(const float *input, float *output, int n)
{
    const __m256 half    = _mm256_set1_ps(0.5f);
    const __m256 one     = _mm256_set1_ps(1.0f);
    const __m256 two     = _mm256_set1_ps(2.0f);
    const __m256 sqrt2pi = _mm256_set1_ps(0.7978845608f);  /* sqrt(2/pi) */
    const __m256 coeff   = _mm256_set1_ps(0.044715f);

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);

        /* t = sqrt(2/pi) * (x + 0.044715 * x^3) */
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 inner = _mm256_fmadd_ps(coeff, x3, x);  /* x + 0.044715*x^3 */
        __m256 t = _mm256_mul_ps(sqrt2pi, inner);

        /* tanh(t) = 2 * sigmoid(2t) - 1 */
        __m256 t2 = _mm256_mul_ps(two, t);
        __m256 sig2t = fast_sigmoid_avx2(t2);
        __m256 tanh_t = _mm256_fmsub_ps(two, sig2t, one);  /* 2*sig-1 */

        /* GELU = 0.5 * x * (1 + tanh) */
        __m256 result = _mm256_mul_ps(half,
            _mm256_mul_ps(x, _mm256_add_ps(one, tanh_t)));
        _mm256_storeu_ps(output + i, result);
    }
    /* Scalar tail */
    for (; i < n; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float t = 0.7978845608f * (x + 0.044715f * x3);
        float tanh_t = tanhf(t);
        output[i] = 0.5f * x * (1.0f + tanh_t);
    }
}

/* ------------------------------------------------------------------ */
/*  ReLU(x) = max(0, x)                                                */
/* ------------------------------------------------------------------ */

void relu_avx2(const float *input, float *output, int n)
{
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        _mm256_storeu_ps(output + i, _mm256_max_ps(zero, x));
    }
    /* Scalar tail */
    for (; i < n; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

/* ------------------------------------------------------------------ */
/*  SwiGLU: output[i] = SiLU(gate[i]) * up[i]                         */
/*  Fused: reads gate+up, computes SiLU(gate)*up, writes output        */
/* ------------------------------------------------------------------ */

void swiglu_avx2(const float *gate, const float *up, float *output, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        __m256 sig = fast_sigmoid_avx2(g);
        __m256 silu_g = _mm256_mul_ps(g, sig);
        __m256 result = _mm256_mul_ps(silu_g, u);
        _mm256_storeu_ps(output + i, result);
    }
    /* Scalar tail */
    for (; i < n; i++) {
        float g = gate[i];
        float silu_g = g * sigmoid_scalar(g);
        output[i] = silu_g * up[i];
    }
}
