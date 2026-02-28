/*
 * ggml_dequant.h - Dequantization routines for common GGML formats
 *
 * Usage:
 *   #define GGML_DEQUANT_IMPLEMENTATION
 *   #include "ggml_dequant.h"
 *
 * Dependencies: gguf_loader.h (for ggml_dtype enum)
 *
 * API:
 *   void dequantize_row_q2_K(const void *src, float *dst, int n);
 *   void dequantize_row_q3_K(const void *src, float *dst, int n);
 *   void dequantize_row_q8_0(const void *src, float *dst, int n);
 *   void dequantize_row_q4_K(const void *src, float *dst, int n);
 *   void dequantize_row_q5_K(const void *src, float *dst, int n);
 *   void dequantize_row_q6_K(const void *src, float *dst, int n);
 *   int  dequant_row(uint32_t ggml_type, const void *src, float *dst, int n);
 */
#ifndef GGML_DEQUANT_H
#define GGML_DEQUANT_H

#include <stdint.h>
#include <stddef.h>
#include "gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Q8_0 block: 32 elements, 34 bytes */
typedef struct {
    uint16_t d;       /* block scale (fp16) */
    int8_t   qs[32];  /* quantized values */
} block_q8_0;

/* Q2_K block: 256 elements, 84 bytes */
typedef struct {
    uint8_t scales[16]; /* packed low/high 4-bit scales/mins */
    uint8_t qs[64];     /* 2-bit quants, 4 values per byte */
    uint16_t d;         /* super-block scale (fp16) */
    uint16_t dmin;      /* super-block min scale (fp16) */
} block_q2_K;

/* Q3_K block: 256 elements, 110 bytes */
typedef struct {
    uint8_t hmask[32];  /* quant high-bit mask planes */
    uint8_t qs[64];     /* quant low 2 bits */
    uint8_t scales[12]; /* packed 6-bit scales */
    uint16_t d;         /* super-block scale (fp16) */
} block_q3_K;

/* Q4_K block: 256 elements, 144 bytes */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint16_t dmin;      /* super-block min (fp16) */
    uint8_t scales[12]; /* 6-bit scales and mins for 8 sub-blocks, packed */
    uint8_t qs[128];    /* 4-bit quants, 2 per byte */
} block_q4_K;

/* Q5_K block: 256 elements, 176 bytes */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint16_t dmin;      /* super-block min (fp16) */
    uint8_t scales[12]; /* 6-bit scales and mins for 8 sub-blocks, packed */
    uint8_t qh[32];     /* quant high bits */
    uint8_t qs[128];    /* quant low 4 bits */
} block_q5_K;

/* Q6_K block: 256 elements, 210 bytes */
typedef struct {
    uint8_t ql[128];    /* lower 4 bits of quants */
    uint8_t qh[64];     /* upper 2 bits of quants */
    int8_t  scales[16]; /* scales for 16 sub-blocks of 16 */
    uint16_t d;         /* super-block scale (fp16) */
} block_q6_K;

static inline float ggml_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            /* subnormal */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    __builtin_memcpy(&result, &f, 4);
    return result;
}

void dequantize_row_q2_K(const void *src, float *dst, int n);
void dequantize_row_q3_K(const void *src, float *dst, int n);
void dequantize_row_q8_0(const void *src, float *dst, int n);
void dequantize_row_q4_K(const void *src, float *dst, int n);
void dequantize_row_q5_K(const void *src, float *dst, int n);
void dequantize_row_q6_K(const void *src, float *dst, int n);

/* Dequantize a row. Returns 0 on success, -1 if type unsupported. */
int dequant_row(uint32_t ggml_type, const void *src, float *dst, int n);

/* Get the byte size of one row of n elements in a given GGML type. */
static inline size_t dequant_row_size(uint32_t type, int n) {
    int bs = 1, ts = 4;
    switch (type) {
        case GGML_TYPE_Q2_K: bs = 256; ts = 84;  break;
        case GGML_TYPE_Q3_K: bs = 256; ts = 110; break;
        case GGML_TYPE_Q8_0: bs = 32;  ts = 34;  break;
        case GGML_TYPE_Q4_K: bs = 256; ts = 144; break;
        case GGML_TYPE_Q5_K: bs = 256; ts = 176; break;
        case GGML_TYPE_Q6_K: bs = 256; ts = 210; break;
        case GGML_TYPE_F32:  bs = 1;   ts = 4;   break;
        case GGML_TYPE_F16:  bs = 1;   ts = 2;   break;
        default: return 0;
    }
    return (size_t)((n + bs - 1) / bs) * ts;
}

/* Fused F16·F32 dot product: sum(a_f16[i] * b_f32[i]) for i in [0, n).
 * Uses AVX2+F16C+FMA when available, scalar fallback otherwise. */
static inline float vec_dot_f16_f32(const uint16_t *a, const float *b, int n);

/* Multi-row F16 matvec: compute 6/4/2 output rows simultaneously.
 * dst[0..R-1] = dot(w0..wR-1, x) where each wi is n F16 values, x is n F32 values. */
static inline void matvec_f16_6row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const uint16_t *w2, const uint16_t *w3,
                                    const uint16_t *w4, const uint16_t *w5,
                                    const float *x, int n);
static inline void matvec_f16_4row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const uint16_t *w2, const uint16_t *w3,
                                    const float *x, int n);
static inline void matvec_f16_2row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const float *x, int n);

/* Batched GEMM: Y[n_rows, N] = W[n_rows, K] (F16) × X[N, K]^T (F32)
 * X is row-major: X[token][k], stride = X_stride (>= K).
 * Y is row-major: Y[row][token], stride = Y_stride (>= N).
 * For each output row r and token t: Y[r][t] = dot(W[r,:], X[t,:]) */
static inline void gemm_f16_f32(float *Y, const uint16_t *W, const float *X,
                                 int n_rows, int K, int N, int Y_stride, int X_stride);

/* Token-major GEMM: Y[N, n_rows] = (W[n_rows, K] × X[N, K]^T)^T
 * Output is token-major: Y[tok][row], stride = Y_stride (>= n_rows).
 * For each token t and row r: Y[t * Y_stride + r] = dot(W[r,:], X[t,:]) */
static inline void gemm_f16_f32_tokmajor(float *Y, const uint16_t *W, const float *X,
                                          int n_rows, int K, int N, int Y_stride, int X_stride);

/* Q8_0 fused dequant+dot: sum(q8_row[i] * x[i]) for i in [0, K) */
static inline float vec_dot_q8_0_f32(const void *q8_row, const float *x, int K);

/* Q8_0 single-row matvec wrapper */
static inline void matvec_q8_0_f32(float *dst, const void *q8_row, const float *x, int K);

/* Q8_0 token-major GEMM: Y[tok * Y_stride + row] = dot(W[row,:], X[tok,:])
 * W is Q8_0 packed: each row is K/32 blocks of 34 bytes. */
static inline void gemm_q8_0_f32_tokmajor(float *Y, const void *W, const float *X,
                                            int n_rows, int K, int N, int Y_stride, int X_stride);

#ifdef __cplusplus
}
#endif

/* ---- Inline SIMD dot product implementation ---- */

#if defined(__F16C__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

static inline float vec_dot_f16_f32(const uint16_t *a, const float *b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int i = 0;
    /* Process 32 elements per iteration (4 accumulators × 8) */
    for (; i + 31 < n; i += 32) {
        __m256 va0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i)));
        __m256 va1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i + 8)));
        __m256 va2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i + 16)));
        __m256 va3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i + 24)));
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
    }
    /* Process 8 elements at a time for remainder */
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i)));
        __m256 vb = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }

    /* Reduce 4 accumulators */
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    /* Horizontal sum of 8 floats */
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
    float result = _mm_cvtss_f32(sum128);

    /* Scalar tail */
    for (; i < n; i++) {
        result += ggml_fp16_to_fp32(a[i]) * b[i];
    }
    return result;
}

/* Multi-row F16 matvec: 6 rows, 12 accumulators, K×2 unrolled */
static inline void matvec_f16_6row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const uint16_t *w2, const uint16_t *w3,
                                    const uint16_t *w4, const uint16_t *w5,
                                    const float *x, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
    __m256 b4 = _mm256_setzero_ps(), b5 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);

        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w0 + i))), x0, a0);
        b0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w0 + i + 8))), x1, b0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w1 + i))), x0, a1);
        b1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w1 + i + 8))), x1, b1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w2 + i))), x0, a2);
        b2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w2 + i + 8))), x1, b2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w3 + i))), x0, a3);
        b3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w3 + i + 8))), x1, b3);
        a4 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w4 + i))), x0, a4);
        b4 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w4 + i + 8))), x1, b4);
        a5 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w5 + i))), x0, a5);
        b5 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w5 + i + 8))), x1, b5);
    }
    /* Merge K+0 and K+8 accumulators */
    a0 = _mm256_add_ps(a0, b0); a1 = _mm256_add_ps(a1, b1);
    a2 = _mm256_add_ps(a2, b2); a3 = _mm256_add_ps(a3, b3);
    a4 = _mm256_add_ps(a4, b4); a5 = _mm256_add_ps(a5, b5);

    /* Process remaining 8-element chunks */
    for (; i + 7 < n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w0 + i))), xv, a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w1 + i))), xv, a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w2 + i))), xv, a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w3 + i))), xv, a3);
        a4 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w4 + i))), xv, a4);
        a5 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w5 + i))), xv, a5);
    }

    /* Horizontal reduce each accumulator */
    #define HSUM256(acc) do { \
        __m128 _hi = _mm256_extractf128_ps(acc, 1); \
        __m128 _lo = _mm256_castps256_ps128(acc); \
        __m128 _s = _mm_add_ps(_lo, _hi); \
        _s = _mm_add_ps(_s, _mm_movehl_ps(_s, _s)); \
        _s = _mm_add_ss(_s, _mm_movehdup_ps(_s)); \
        dst[_idx++] = _mm_cvtss_f32(_s); \
    } while(0)

    int _idx = 0;
    HSUM256(a0); HSUM256(a1); HSUM256(a2);
    HSUM256(a3); HSUM256(a4); HSUM256(a5);
    #undef HSUM256

    /* Scalar tail */
    for (; i < n; i++) {
        float xv = x[i];
        dst[0] += ggml_fp16_to_fp32(w0[i]) * xv;
        dst[1] += ggml_fp16_to_fp32(w1[i]) * xv;
        dst[2] += ggml_fp16_to_fp32(w2[i]) * xv;
        dst[3] += ggml_fp16_to_fp32(w3[i]) * xv;
        dst[4] += ggml_fp16_to_fp32(w4[i]) * xv;
        dst[5] += ggml_fp16_to_fp32(w5[i]) * xv;
    }
}

/* Multi-row F16 matvec: 4 rows */
static inline void matvec_f16_4row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const uint16_t *w2, const uint16_t *w3,
                                    const float *x, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w0 + i))), x0, a0);
        b0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w0 + i + 8))), x1, b0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w1 + i))), x0, a1);
        b1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w1 + i + 8))), x1, b1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w2 + i))), x0, a2);
        b2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w2 + i + 8))), x1, b2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w3 + i))), x0, a3);
        b3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w3 + i + 8))), x1, b3);
    }
    a0 = _mm256_add_ps(a0, b0); a1 = _mm256_add_ps(a1, b1);
    a2 = _mm256_add_ps(a2, b2); a3 = _mm256_add_ps(a3, b3);

    for (; i + 7 < n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w0 + i))), xv, a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w1 + i))), xv, a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w2 + i))), xv, a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w3 + i))), xv, a3);
    }

    #define HSUM256(acc) do { \
        __m128 _hi = _mm256_extractf128_ps(acc, 1); \
        __m128 _lo = _mm256_castps256_ps128(acc); \
        __m128 _s = _mm_add_ps(_lo, _hi); \
        _s = _mm_add_ps(_s, _mm_movehl_ps(_s, _s)); \
        _s = _mm_add_ss(_s, _mm_movehdup_ps(_s)); \
        dst[_idx++] = _mm_cvtss_f32(_s); \
    } while(0)

    int _idx = 0;
    HSUM256(a0); HSUM256(a1); HSUM256(a2); HSUM256(a3);
    #undef HSUM256

    for (; i < n; i++) {
        float xv = x[i];
        dst[0] += ggml_fp16_to_fp32(w0[i]) * xv;
        dst[1] += ggml_fp16_to_fp32(w1[i]) * xv;
        dst[2] += ggml_fp16_to_fp32(w2[i]) * xv;
        dst[3] += ggml_fp16_to_fp32(w3[i]) * xv;
    }
}

/* Multi-row F16 matvec: 2 rows */
static inline void matvec_f16_2row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const float *x, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w0 + i))), x0, a0);
        b0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w0 + i + 8))), x1, b0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w1 + i))), x0, a1);
        b1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w1 + i + 8))), x1, b1);
    }
    a0 = _mm256_add_ps(a0, b0); a1 = _mm256_add_ps(a1, b1);

    for (; i + 7 < n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w0 + i))), xv, a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w1 + i))), xv, a1);
    }

    __m128 hi0 = _mm256_extractf128_ps(a0, 1), lo0 = _mm256_castps256_ps128(a0);
    __m128 s0 = _mm_add_ps(lo0, hi0);
    s0 = _mm_add_ps(s0, _mm_movehl_ps(s0, s0));
    s0 = _mm_add_ss(s0, _mm_movehdup_ps(s0));
    dst[0] = _mm_cvtss_f32(s0);

    __m128 hi1 = _mm256_extractf128_ps(a1, 1), lo1 = _mm256_castps256_ps128(a1);
    __m128 s1 = _mm_add_ps(lo1, hi1);
    s1 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
    s1 = _mm_add_ss(s1, _mm_movehdup_ps(s1));
    dst[1] = _mm_cvtss_f32(s1);

    for (; i < n; i++) {
        float xv = x[i];
        dst[0] += ggml_fp16_to_fp32(w0[i]) * xv;
        dst[1] += ggml_fp16_to_fp32(w1[i]) * xv;
    }
}

/* Batched GEMM: Y[n_rows, N] = W[n_rows, K] (F16) × X[N, K]^T (F32)
 * Tiles: 6 weight rows × 8 tokens = 48 accumulators (near AVX2 register limit).
 * Falls back to vec_dot_f16_f32 for remainder rows/tokens. */
static inline void gemm_f16_f32(float *Y, const uint16_t *W, const float *X,
                                 int n_rows, int K, int N, int Y_stride, int X_stride) {
    int r = 0;
    /* Process 6 rows at a time */
    for (; r + 5 < n_rows; r += 6) {
        const uint16_t *w0 = W + (size_t)r * K;
        const uint16_t *w1 = W + (size_t)(r+1) * K;
        const uint16_t *w2 = W + (size_t)(r+2) * K;
        const uint16_t *w3 = W + (size_t)(r+3) * K;
        const uint16_t *w4 = W + (size_t)(r+4) * K;
        const uint16_t *w5 = W + (size_t)(r+5) * K;

        int t = 0;
        /* Process 8 tokens at a time: 6 rows × 8 tokens = 48 accumulators */
        for (; t + 7 < N; t += 8) {
            __m256 a00=_mm256_setzero_ps(), a01=_mm256_setzero_ps();
            __m256 a10=_mm256_setzero_ps(), a11=_mm256_setzero_ps();
            __m256 a20=_mm256_setzero_ps(), a21=_mm256_setzero_ps();
            __m256 a30=_mm256_setzero_ps(), a31=_mm256_setzero_ps();
            __m256 a40=_mm256_setzero_ps(), a41=_mm256_setzero_ps();
            __m256 a50=_mm256_setzero_ps(), a51=_mm256_setzero_ps();

            /* Gather 8 token pointers */
            const float *x0 = X + (size_t)(t+0)*X_stride;
            const float *x1 = X + (size_t)(t+1)*X_stride;
            const float *x2 = X + (size_t)(t+2)*X_stride;
            const float *x3 = X + (size_t)(t+3)*X_stride;
            const float *x4 = X + (size_t)(t+4)*X_stride;
            const float *x5 = X + (size_t)(t+5)*X_stride;
            const float *x6 = X + (size_t)(t+6)*X_stride;
            const float *x7 = X + (size_t)(t+7)*X_stride;

            for (int k = 0; k + 7 < K; k += 8) {
                /* Load 8 weight elements for each of 6 rows */
                __m256 vw0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0+k)));
                __m256 vw1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1+k)));
                __m256 vw2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2+k)));
                __m256 vw3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3+k)));
                __m256 vw4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w4+k)));
                __m256 vw5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w5+k)));

                /* Load 8 elements from each of 8 tokens, accumulate dot products.
                 * We pack 2 tokens per accumulator pair to stay in 16 registers.
                 * Actually we use separate accumulators per row, interleaving tokens. */
                /* Token 0 */
                __m256 vx = _mm256_loadu_ps(x0+k);
                a00 = _mm256_fmadd_ps(vw0, vx, a00); a10 = _mm256_fmadd_ps(vw1, vx, a10);
                a20 = _mm256_fmadd_ps(vw2, vx, a20); a30 = _mm256_fmadd_ps(vw3, vx, a30);
                a40 = _mm256_fmadd_ps(vw4, vx, a40); a50 = _mm256_fmadd_ps(vw5, vx, a50);
                /* Token 1 */
                vx = _mm256_loadu_ps(x1+k);
                a01 = _mm256_fmadd_ps(vw0, vx, a01); a11 = _mm256_fmadd_ps(vw1, vx, a11);
                a21 = _mm256_fmadd_ps(vw2, vx, a21); a31 = _mm256_fmadd_ps(vw3, vx, a31);
                a41 = _mm256_fmadd_ps(vw4, vx, a41); a51 = _mm256_fmadd_ps(vw5, vx, a51);
            }
            /* Horizontal reduce and store the 6×2 partial results, then repeat for tokens 2-7 */
            /* We only have 12 accumulators above for tokens 0-1 (to fit registers).
             * For a true 6×8 tile we'd need 48 accumulators which exceeds 16 YMM regs.
             * So we process 2 tokens at a time through the K dimension, storing results. */
            #define HSUM_STORE(acc, row, tok) do { \
                __m128 _hi = _mm256_extractf128_ps(acc, 1); \
                __m128 _lo = _mm256_castps256_ps128(acc); \
                __m128 _s = _mm_add_ps(_lo, _hi); \
                _s = _mm_add_ps(_s, _mm_movehl_ps(_s, _s)); \
                _s = _mm_add_ss(_s, _mm_movehdup_ps(_s)); \
                Y[(row)*Y_stride + (tok)] = _mm_cvtss_f32(_s); \
            } while(0)
            HSUM_STORE(a00, r+0, t+0); HSUM_STORE(a01, r+0, t+1);
            HSUM_STORE(a10, r+1, t+0); HSUM_STORE(a11, r+1, t+1);
            HSUM_STORE(a20, r+2, t+0); HSUM_STORE(a21, r+2, t+1);
            HSUM_STORE(a30, r+3, t+0); HSUM_STORE(a31, r+3, t+1);
            HSUM_STORE(a40, r+4, t+0); HSUM_STORE(a41, r+4, t+1);
            HSUM_STORE(a50, r+5, t+0); HSUM_STORE(a51, r+5, t+1);

            /* Tokens 2-3 */
            a00=_mm256_setzero_ps(); a01=_mm256_setzero_ps();
            a10=_mm256_setzero_ps(); a11=_mm256_setzero_ps();
            a20=_mm256_setzero_ps(); a21=_mm256_setzero_ps();
            a30=_mm256_setzero_ps(); a31=_mm256_setzero_ps();
            a40=_mm256_setzero_ps(); a41=_mm256_setzero_ps();
            a50=_mm256_setzero_ps(); a51=_mm256_setzero_ps();
            for (int k = 0; k + 7 < K; k += 8) {
                __m256 vw0_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0+k)));
                __m256 vw1_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1+k)));
                __m256 vw2_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2+k)));
                __m256 vw3_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3+k)));
                __m256 vw4_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w4+k)));
                __m256 vw5_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w5+k)));
                __m256 vx = _mm256_loadu_ps(x2+k);
                a00 = _mm256_fmadd_ps(vw0_, vx, a00); a10 = _mm256_fmadd_ps(vw1_, vx, a10);
                a20 = _mm256_fmadd_ps(vw2_, vx, a20); a30 = _mm256_fmadd_ps(vw3_, vx, a30);
                a40 = _mm256_fmadd_ps(vw4_, vx, a40); a50 = _mm256_fmadd_ps(vw5_, vx, a50);
                vx = _mm256_loadu_ps(x3+k);
                a01 = _mm256_fmadd_ps(vw0_, vx, a01); a11 = _mm256_fmadd_ps(vw1_, vx, a11);
                a21 = _mm256_fmadd_ps(vw2_, vx, a21); a31 = _mm256_fmadd_ps(vw3_, vx, a31);
                a41 = _mm256_fmadd_ps(vw4_, vx, a41); a51 = _mm256_fmadd_ps(vw5_, vx, a51);
            }
            HSUM_STORE(a00, r+0, t+2); HSUM_STORE(a01, r+0, t+3);
            HSUM_STORE(a10, r+1, t+2); HSUM_STORE(a11, r+1, t+3);
            HSUM_STORE(a20, r+2, t+2); HSUM_STORE(a21, r+2, t+3);
            HSUM_STORE(a30, r+3, t+2); HSUM_STORE(a31, r+3, t+3);
            HSUM_STORE(a40, r+4, t+2); HSUM_STORE(a41, r+4, t+3);
            HSUM_STORE(a50, r+5, t+2); HSUM_STORE(a51, r+5, t+3);

            /* Tokens 4-5 */
            a00=_mm256_setzero_ps(); a01=_mm256_setzero_ps();
            a10=_mm256_setzero_ps(); a11=_mm256_setzero_ps();
            a20=_mm256_setzero_ps(); a21=_mm256_setzero_ps();
            a30=_mm256_setzero_ps(); a31=_mm256_setzero_ps();
            a40=_mm256_setzero_ps(); a41=_mm256_setzero_ps();
            a50=_mm256_setzero_ps(); a51=_mm256_setzero_ps();
            for (int k = 0; k + 7 < K; k += 8) {
                __m256 vw0_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0+k)));
                __m256 vw1_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1+k)));
                __m256 vw2_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2+k)));
                __m256 vw3_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3+k)));
                __m256 vw4_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w4+k)));
                __m256 vw5_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w5+k)));
                __m256 vx = _mm256_loadu_ps(x4+k);
                a00 = _mm256_fmadd_ps(vw0_, vx, a00); a10 = _mm256_fmadd_ps(vw1_, vx, a10);
                a20 = _mm256_fmadd_ps(vw2_, vx, a20); a30 = _mm256_fmadd_ps(vw3_, vx, a30);
                a40 = _mm256_fmadd_ps(vw4_, vx, a40); a50 = _mm256_fmadd_ps(vw5_, vx, a50);
                vx = _mm256_loadu_ps(x5+k);
                a01 = _mm256_fmadd_ps(vw0_, vx, a01); a11 = _mm256_fmadd_ps(vw1_, vx, a11);
                a21 = _mm256_fmadd_ps(vw2_, vx, a21); a31 = _mm256_fmadd_ps(vw3_, vx, a31);
                a41 = _mm256_fmadd_ps(vw4_, vx, a41); a51 = _mm256_fmadd_ps(vw5_, vx, a51);
            }
            HSUM_STORE(a00, r+0, t+4); HSUM_STORE(a01, r+0, t+5);
            HSUM_STORE(a10, r+1, t+4); HSUM_STORE(a11, r+1, t+5);
            HSUM_STORE(a20, r+2, t+4); HSUM_STORE(a21, r+2, t+5);
            HSUM_STORE(a30, r+3, t+4); HSUM_STORE(a31, r+3, t+5);
            HSUM_STORE(a40, r+4, t+4); HSUM_STORE(a41, r+4, t+5);
            HSUM_STORE(a50, r+5, t+4); HSUM_STORE(a51, r+5, t+5);

            /* Tokens 6-7 */
            a00=_mm256_setzero_ps(); a01=_mm256_setzero_ps();
            a10=_mm256_setzero_ps(); a11=_mm256_setzero_ps();
            a20=_mm256_setzero_ps(); a21=_mm256_setzero_ps();
            a30=_mm256_setzero_ps(); a31=_mm256_setzero_ps();
            a40=_mm256_setzero_ps(); a41=_mm256_setzero_ps();
            a50=_mm256_setzero_ps(); a51=_mm256_setzero_ps();
            for (int k = 0; k + 7 < K; k += 8) {
                __m256 vw0_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0+k)));
                __m256 vw1_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1+k)));
                __m256 vw2_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2+k)));
                __m256 vw3_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3+k)));
                __m256 vw4_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w4+k)));
                __m256 vw5_ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w5+k)));
                __m256 vx = _mm256_loadu_ps(x6+k);
                a00 = _mm256_fmadd_ps(vw0_, vx, a00); a10 = _mm256_fmadd_ps(vw1_, vx, a10);
                a20 = _mm256_fmadd_ps(vw2_, vx, a20); a30 = _mm256_fmadd_ps(vw3_, vx, a30);
                a40 = _mm256_fmadd_ps(vw4_, vx, a40); a50 = _mm256_fmadd_ps(vw5_, vx, a50);
                vx = _mm256_loadu_ps(x7+k);
                a01 = _mm256_fmadd_ps(vw0_, vx, a01); a11 = _mm256_fmadd_ps(vw1_, vx, a11);
                a21 = _mm256_fmadd_ps(vw2_, vx, a21); a31 = _mm256_fmadd_ps(vw3_, vx, a31);
                a41 = _mm256_fmadd_ps(vw4_, vx, a41); a51 = _mm256_fmadd_ps(vw5_, vx, a51);
            }
            HSUM_STORE(a00, r+0, t+6); HSUM_STORE(a01, r+0, t+7);
            HSUM_STORE(a10, r+1, t+6); HSUM_STORE(a11, r+1, t+7);
            HSUM_STORE(a20, r+2, t+6); HSUM_STORE(a21, r+2, t+7);
            HSUM_STORE(a30, r+3, t+6); HSUM_STORE(a31, r+3, t+7);
            HSUM_STORE(a40, r+4, t+6); HSUM_STORE(a41, r+4, t+7);
            HSUM_STORE(a50, r+5, t+6); HSUM_STORE(a51, r+5, t+7);
            #undef HSUM_STORE
        }
        /* Remainder tokens */
        for (; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            float d0 = vec_dot_f16_f32(w0, xt, K);
            float d1 = vec_dot_f16_f32(w1, xt, K);
            float d2 = vec_dot_f16_f32(w2, xt, K);
            float d3 = vec_dot_f16_f32(w3, xt, K);
            float d4 = vec_dot_f16_f32(w4, xt, K);
            float d5 = vec_dot_f16_f32(w5, xt, K);
            Y[(r+0)*Y_stride + t] = d0; Y[(r+1)*Y_stride + t] = d1;
            Y[(r+2)*Y_stride + t] = d2; Y[(r+3)*Y_stride + t] = d3;
            Y[(r+4)*Y_stride + t] = d4; Y[(r+5)*Y_stride + t] = d5;
        }
    }
    /* Remainder rows: 1 at a time */
    for (; r < n_rows; r++) {
        const uint16_t *wr = W + (size_t)r * K;
        for (int t = 0; t < N; t++) {
            Y[r * Y_stride + t] = vec_dot_f16_f32(wr, X + (size_t)t * X_stride, K);
        }
    }
}

/* Token-major GEMM: Y[tok * Y_stride + row] = dot(W[row,:], X[tok,:])
 *
 * 4-row × 4-token register tile = 16 accumulators. With 4 weight regs,
 * that's 20 YMM regs — compiler spills ~4 to stack (L1), acceptable.
 * Weights re-read per 4-token group but stay warm in L1 (4 rows × K × 2B
 * = 16KB for K=2048, fits L1 32KB). */
static inline void gemm_f16_f32_tokmajor(float *Y, const uint16_t *W, const float *X,
                                          int n_rows, int K, int N, int Y_stride, int X_stride) {
    #define TM_HSUM_STORE(acc, row, tok) do { \
        __m128 _hi = _mm256_extractf128_ps(acc, 1); \
        __m128 _lo = _mm256_castps256_ps128(acc); \
        __m128 _s = _mm_add_ps(_lo, _hi); \
        _s = _mm_add_ps(_s, _mm_movehl_ps(_s, _s)); \
        _s = _mm_add_ss(_s, _mm_movehdup_ps(_s)); \
        Y[(tok)*Y_stride + (row)] = _mm_cvtss_f32(_s); \
    } while(0)

    /* 3-row × 4-token tile: 12 accumulators + 3 weight + 1 input = 16 YMM regs.
     * No register spills. Weights (6KB/tile) stay in L1; input served from L3. */
    int r = 0;
    for (; r + 2 < n_rows; r += 3) {
        const uint16_t *w0 = W + (size_t)(r+0) * K;
        const uint16_t *w1 = W + (size_t)(r+1) * K;
        const uint16_t *w2 = W + (size_t)(r+2) * K;

        int t = 0;
        for (; t + 3 < N; t += 4) {
            const float *x0 = X + (size_t)(t+0) * X_stride;
            const float *x1 = X + (size_t)(t+1) * X_stride;
            const float *x2 = X + (size_t)(t+2) * X_stride;
            const float *x3 = X + (size_t)(t+3) * X_stride;

            __m256 a00=_mm256_setzero_ps(), a01=_mm256_setzero_ps();
            __m256 a02=_mm256_setzero_ps(), a03=_mm256_setzero_ps();
            __m256 a10=_mm256_setzero_ps(), a11=_mm256_setzero_ps();
            __m256 a12=_mm256_setzero_ps(), a13=_mm256_setzero_ps();
            __m256 a20=_mm256_setzero_ps(), a21=_mm256_setzero_ps();
            __m256 a22=_mm256_setzero_ps(), a23=_mm256_setzero_ps();

            for (int k = 0; k + 7 < K; k += 8) {
                __m256 vw0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0+k)));
                __m256 vw1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1+k)));
                __m256 vw2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2+k)));

                __m256 vx = _mm256_loadu_ps(x0+k);
                a00=_mm256_fmadd_ps(vw0,vx,a00); a10=_mm256_fmadd_ps(vw1,vx,a10);
                a20=_mm256_fmadd_ps(vw2,vx,a20);

                vx = _mm256_loadu_ps(x1+k);
                a01=_mm256_fmadd_ps(vw0,vx,a01); a11=_mm256_fmadd_ps(vw1,vx,a11);
                a21=_mm256_fmadd_ps(vw2,vx,a21);

                vx = _mm256_loadu_ps(x2+k);
                a02=_mm256_fmadd_ps(vw0,vx,a02); a12=_mm256_fmadd_ps(vw1,vx,a12);
                a22=_mm256_fmadd_ps(vw2,vx,a22);

                vx = _mm256_loadu_ps(x3+k);
                a03=_mm256_fmadd_ps(vw0,vx,a03); a13=_mm256_fmadd_ps(vw1,vx,a13);
                a23=_mm256_fmadd_ps(vw2,vx,a23);
            }

            TM_HSUM_STORE(a00,r+0,t+0); TM_HSUM_STORE(a01,r+0,t+1);
            TM_HSUM_STORE(a02,r+0,t+2); TM_HSUM_STORE(a03,r+0,t+3);
            TM_HSUM_STORE(a10,r+1,t+0); TM_HSUM_STORE(a11,r+1,t+1);
            TM_HSUM_STORE(a12,r+1,t+2); TM_HSUM_STORE(a13,r+1,t+3);
            TM_HSUM_STORE(a20,r+2,t+0); TM_HSUM_STORE(a21,r+2,t+1);
            TM_HSUM_STORE(a22,r+2,t+2); TM_HSUM_STORE(a23,r+2,t+3);
        }
        for (; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            Y[t*Y_stride+r+0] = vec_dot_f16_f32(w0, xt, K);
            Y[t*Y_stride+r+1] = vec_dot_f16_f32(w1, xt, K);
            Y[t*Y_stride+r+2] = vec_dot_f16_f32(w2, xt, K);
        }
    }
    /* Remainder rows: 1 at a time */
    for (; r < n_rows; r++) {
        const uint16_t *wr = W + (size_t)r * K;
        for (int t = 0; t < N; t++) {
            Y[t * Y_stride + r] = vec_dot_f16_f32(wr, X + (size_t)t * X_stride, K);
        }
    }
    #undef TM_HSUM_STORE
}

/* Fused dual-matrix GEMM: compute Y1[tok,row]=dot(W1[row,:],X[tok,:]) and
 * Y2[tok,row]=dot(W2[row,:],X[tok,:]) simultaneously, reading X once.
 * Both matrices must have same n_rows and K. Tile: 2 matrices × 2 rows × 4 tokens = 16 accumulators. */
static inline void gemm_f16_f32_tokmajor_fused2(
        float *Y1, const uint16_t *W1,
        float *Y2, const uint16_t *W2,
        const float *X, int n_rows, int K, int N,
        int Y1_stride, int Y2_stride, int X_stride) {

    #define FUSED_HSUM_STORE(acc, Y, row, tok, stride) do { \
        __m128 _hi = _mm256_extractf128_ps(acc, 1); \
        __m128 _lo = _mm256_castps256_ps128(acc); \
        __m128 _s = _mm_add_ps(_lo, _hi); \
        _s = _mm_add_ps(_s, _mm_movehl_ps(_s, _s)); \
        _s = _mm_add_ss(_s, _mm_movehdup_ps(_s)); \
        Y[(tok)*(stride) + (row)] = _mm_cvtss_f32(_s); \
    } while(0)

    int r = 0;
    for (; r + 1 < n_rows; r += 2) {
        const uint16_t *w1a = W1 + (size_t)(r+0) * K;
        const uint16_t *w1b = W1 + (size_t)(r+1) * K;
        const uint16_t *w2a = W2 + (size_t)(r+0) * K;
        const uint16_t *w2b = W2 + (size_t)(r+1) * K;

        int t = 0;
        for (; t + 3 < N; t += 4) {
            const float *x0 = X + (size_t)(t+0) * X_stride;
            const float *x1 = X + (size_t)(t+1) * X_stride;
            const float *x2 = X + (size_t)(t+2) * X_stride;
            const float *x3 = X + (size_t)(t+3) * X_stride;

            /* 2 matrices × 2 rows × 4 tokens = 16 accumulators */
            __m256 a00=_mm256_setzero_ps(), a01=_mm256_setzero_ps();
            __m256 a02=_mm256_setzero_ps(), a03=_mm256_setzero_ps();
            __m256 a10=_mm256_setzero_ps(), a11=_mm256_setzero_ps();
            __m256 a12=_mm256_setzero_ps(), a13=_mm256_setzero_ps();
            __m256 b00=_mm256_setzero_ps(), b01=_mm256_setzero_ps();
            __m256 b02=_mm256_setzero_ps(), b03=_mm256_setzero_ps();
            __m256 b10=_mm256_setzero_ps(), b11=_mm256_setzero_ps();
            __m256 b12=_mm256_setzero_ps(), b13=_mm256_setzero_ps();

            for (int k = 0; k + 7 < K; k += 8) {
                __m256 vx0 = _mm256_loadu_ps(x0+k);
                __m256 vx1 = _mm256_loadu_ps(x1+k);
                __m256 vx2 = _mm256_loadu_ps(x2+k);
                __m256 vx3 = _mm256_loadu_ps(x3+k);

                __m256 vw;
                vw = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1a+k)));
                a00=_mm256_fmadd_ps(vw,vx0,a00); a01=_mm256_fmadd_ps(vw,vx1,a01);
                a02=_mm256_fmadd_ps(vw,vx2,a02); a03=_mm256_fmadd_ps(vw,vx3,a03);

                vw = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1b+k)));
                a10=_mm256_fmadd_ps(vw,vx0,a10); a11=_mm256_fmadd_ps(vw,vx1,a11);
                a12=_mm256_fmadd_ps(vw,vx2,a12); a13=_mm256_fmadd_ps(vw,vx3,a13);

                vw = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2a+k)));
                b00=_mm256_fmadd_ps(vw,vx0,b00); b01=_mm256_fmadd_ps(vw,vx1,b01);
                b02=_mm256_fmadd_ps(vw,vx2,b02); b03=_mm256_fmadd_ps(vw,vx3,b03);

                vw = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2b+k)));
                b10=_mm256_fmadd_ps(vw,vx0,b10); b11=_mm256_fmadd_ps(vw,vx1,b11);
                b12=_mm256_fmadd_ps(vw,vx2,b12); b13=_mm256_fmadd_ps(vw,vx3,b13);
            }

            FUSED_HSUM_STORE(a00,Y1,r+0,t+0,Y1_stride); FUSED_HSUM_STORE(a01,Y1,r+0,t+1,Y1_stride);
            FUSED_HSUM_STORE(a02,Y1,r+0,t+2,Y1_stride); FUSED_HSUM_STORE(a03,Y1,r+0,t+3,Y1_stride);
            FUSED_HSUM_STORE(a10,Y1,r+1,t+0,Y1_stride); FUSED_HSUM_STORE(a11,Y1,r+1,t+1,Y1_stride);
            FUSED_HSUM_STORE(a12,Y1,r+1,t+2,Y1_stride); FUSED_HSUM_STORE(a13,Y1,r+1,t+3,Y1_stride);
            FUSED_HSUM_STORE(b00,Y2,r+0,t+0,Y2_stride); FUSED_HSUM_STORE(b01,Y2,r+0,t+1,Y2_stride);
            FUSED_HSUM_STORE(b02,Y2,r+0,t+2,Y2_stride); FUSED_HSUM_STORE(b03,Y2,r+0,t+3,Y2_stride);
            FUSED_HSUM_STORE(b10,Y2,r+1,t+0,Y2_stride); FUSED_HSUM_STORE(b11,Y2,r+1,t+1,Y2_stride);
            FUSED_HSUM_STORE(b12,Y2,r+1,t+2,Y2_stride); FUSED_HSUM_STORE(b13,Y2,r+1,t+3,Y2_stride);
        }
        for (; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            Y1[t*Y1_stride+r+0] = vec_dot_f16_f32(w1a, xt, K);
            Y1[t*Y1_stride+r+1] = vec_dot_f16_f32(w1b, xt, K);
            Y2[t*Y2_stride+r+0] = vec_dot_f16_f32(w2a, xt, K);
            Y2[t*Y2_stride+r+1] = vec_dot_f16_f32(w2b, xt, K);
        }
    }
    /* Remainder: odd last row */
    if (r < n_rows) {
        const uint16_t *w1r = W1 + (size_t)r * K;
        const uint16_t *w2r = W2 + (size_t)r * K;
        for (int t = 0; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            Y1[t * Y1_stride + r] = vec_dot_f16_f32(w1r, xt, K);
            Y2[t * Y2_stride + r] = vec_dot_f16_f32(w2r, xt, K);
        }
    }
    #undef FUSED_HSUM_STORE
}

/* ---- Q8_0 SIMD kernels ---- */

/* Fused dequant+dot for one Q8_0 row × F32 vector.
 * Processes one block (32 int8 + fp16 scale = 34 bytes) per iteration.
 * 4 groups of 8 int8 → cvtepi8_epi32 → cvtepi32_ps → mul(scale) → FMA. */
static inline float vec_dot_q8_0_f32(const void *q8_row, const float *x, int K) {
    const block_q8_0 *blocks = (const block_q8_0 *)q8_row;
    int nb = K / 32;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        float scale = ggml_fp16_to_fp32(blocks[b].d);
        __m256 vscale = _mm256_set1_ps(scale);
        const int8_t *qs = blocks[b].qs;
        const float *xp = x + b * 32;

        /* Group 0: qs[0..7] */
        __m128i bytes0 = _mm_loadl_epi64((const __m128i *)(qs));
        __m256i i32_0 = _mm256_cvtepi8_epi32(bytes0);
        __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_0), vscale);
        acc0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(xp), acc0);

        /* Group 1: qs[8..15] */
        __m128i bytes1 = _mm_loadl_epi64((const __m128i *)(qs + 8));
        __m256i i32_1 = _mm256_cvtepi8_epi32(bytes1);
        __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_1), vscale);
        acc1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xp + 8), acc1);

        /* Group 2: qs[16..23] */
        __m128i bytes2 = _mm_loadl_epi64((const __m128i *)(qs + 16));
        __m256i i32_2 = _mm256_cvtepi8_epi32(bytes2);
        __m256 f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_2), vscale);
        acc0 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(xp + 16), acc0);

        /* Group 3: qs[24..31] */
        __m128i bytes3 = _mm_loadl_epi64((const __m128i *)(qs + 24));
        __m256i i32_3 = _mm256_cvtepi8_epi32(bytes3);
        __m256 f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_3), vscale);
        acc1 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(xp + 24), acc1);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/* Single-row Q8_0 matvec wrapper */
static inline void matvec_q8_0_f32(float *dst, const void *q8_row, const float *x, int K) {
    *dst = vec_dot_q8_0_f32(q8_row, x, K);
}

/* Q8_0 token-major GEMM: 3-row × 4-token tile, 12 accumulators.
 * Per block per row: broadcast scale, 4 groups of 8 int8→float FMA.
 * Weight dequant amortized across 4 tokens. */
static inline void gemm_q8_0_f32_tokmajor(float *Y, const void *W, const float *X,
                                            int n_rows, int K, int N, int Y_stride, int X_stride) {
    int nb = K / 32;
    size_t row_bytes = (size_t)nb * sizeof(block_q8_0);

    #define Q8_HSUM_STORE(acc, row, tok) do { \
        __m128 _hi = _mm256_extractf128_ps(acc, 1); \
        __m128 _lo = _mm256_castps256_ps128(acc); \
        __m128 _s = _mm_add_ps(_lo, _hi); \
        _s = _mm_add_ps(_s, _mm_movehl_ps(_s, _s)); \
        _s = _mm_add_ss(_s, _mm_movehdup_ps(_s)); \
        Y[(tok)*Y_stride + (row)] = _mm_cvtss_f32(_s); \
    } while(0)

    int r = 0;
    for (; r + 2 < n_rows; r += 3) {
        const block_q8_0 *w0 = (const block_q8_0 *)((const uint8_t *)W + (size_t)(r+0) * row_bytes);
        const block_q8_0 *w1 = (const block_q8_0 *)((const uint8_t *)W + (size_t)(r+1) * row_bytes);
        const block_q8_0 *w2 = (const block_q8_0 *)((const uint8_t *)W + (size_t)(r+2) * row_bytes);

        int t = 0;
        for (; t + 3 < N; t += 4) {
            const float *x0 = X + (size_t)(t+0) * X_stride;
            const float *x1 = X + (size_t)(t+1) * X_stride;
            const float *x2 = X + (size_t)(t+2) * X_stride;
            const float *x3 = X + (size_t)(t+3) * X_stride;

            /* 3 rows × 4 tokens = 12 accumulators */
            __m256 a00=_mm256_setzero_ps(), a01=_mm256_setzero_ps();
            __m256 a02=_mm256_setzero_ps(), a03=_mm256_setzero_ps();
            __m256 a10=_mm256_setzero_ps(), a11=_mm256_setzero_ps();
            __m256 a12=_mm256_setzero_ps(), a13=_mm256_setzero_ps();
            __m256 a20=_mm256_setzero_ps(), a21=_mm256_setzero_ps();
            __m256 a22=_mm256_setzero_ps(), a23=_mm256_setzero_ps();

            for (int b = 0; b < nb; b++) {
                int base_k = b * 32;

                /* Dequant 3 rows × 4 groups, FMA with 4 tokens */
                for (int g = 0; g < 4; g++) {
                    int off = base_k + g * 8;
                    __m256 vx0 = _mm256_loadu_ps(x0 + off);
                    __m256 vx1 = _mm256_loadu_ps(x1 + off);
                    __m256 vx2 = _mm256_loadu_ps(x2 + off);
                    __m256 vx3 = _mm256_loadu_ps(x3 + off);

                    /* Row 0 */
                    {
                        __m256 vscale = _mm256_set1_ps(ggml_fp16_to_fp32(w0[b].d));
                        __m128i bytes = _mm_loadl_epi64((const __m128i *)(w0[b].qs + g * 8));
                        __m256 vw = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)), vscale);
                        a00 = _mm256_fmadd_ps(vw, vx0, a00);
                        a01 = _mm256_fmadd_ps(vw, vx1, a01);
                        a02 = _mm256_fmadd_ps(vw, vx2, a02);
                        a03 = _mm256_fmadd_ps(vw, vx3, a03);
                    }
                    /* Row 1 */
                    {
                        __m256 vscale = _mm256_set1_ps(ggml_fp16_to_fp32(w1[b].d));
                        __m128i bytes = _mm_loadl_epi64((const __m128i *)(w1[b].qs + g * 8));
                        __m256 vw = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)), vscale);
                        a10 = _mm256_fmadd_ps(vw, vx0, a10);
                        a11 = _mm256_fmadd_ps(vw, vx1, a11);
                        a12 = _mm256_fmadd_ps(vw, vx2, a12);
                        a13 = _mm256_fmadd_ps(vw, vx3, a13);
                    }
                    /* Row 2 */
                    {
                        __m256 vscale = _mm256_set1_ps(ggml_fp16_to_fp32(w2[b].d));
                        __m128i bytes = _mm_loadl_epi64((const __m128i *)(w2[b].qs + g * 8));
                        __m256 vw = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)), vscale);
                        a20 = _mm256_fmadd_ps(vw, vx0, a20);
                        a21 = _mm256_fmadd_ps(vw, vx1, a21);
                        a22 = _mm256_fmadd_ps(vw, vx2, a22);
                        a23 = _mm256_fmadd_ps(vw, vx3, a23);
                    }
                }
            }

            Q8_HSUM_STORE(a00,r+0,t+0); Q8_HSUM_STORE(a01,r+0,t+1);
            Q8_HSUM_STORE(a02,r+0,t+2); Q8_HSUM_STORE(a03,r+0,t+3);
            Q8_HSUM_STORE(a10,r+1,t+0); Q8_HSUM_STORE(a11,r+1,t+1);
            Q8_HSUM_STORE(a12,r+1,t+2); Q8_HSUM_STORE(a13,r+1,t+3);
            Q8_HSUM_STORE(a20,r+2,t+0); Q8_HSUM_STORE(a21,r+2,t+1);
            Q8_HSUM_STORE(a22,r+2,t+2); Q8_HSUM_STORE(a23,r+2,t+3);
        }
        /* Remainder tokens */
        for (; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            Y[t*Y_stride+r+0] = vec_dot_q8_0_f32(w0, xt, K);
            Y[t*Y_stride+r+1] = vec_dot_q8_0_f32(w1, xt, K);
            Y[t*Y_stride+r+2] = vec_dot_q8_0_f32(w2, xt, K);
        }
    }
    /* Remainder rows */
    for (; r < n_rows; r++) {
        const void *wr = (const uint8_t *)W + (size_t)r * row_bytes;
        for (int t = 0; t < N; t++) {
            Y[t * Y_stride + r] = vec_dot_q8_0_f32(wr, X + (size_t)t * X_stride, K);
        }
    }
    #undef Q8_HSUM_STORE
}

#else
/* Scalar fallback */
static inline float vec_dot_f16_f32(const uint16_t *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += ggml_fp16_to_fp32(a[i]) * b[i];
    }
    return sum;
}
/* Scalar fallback multi-row kernels */
static inline void matvec_f16_6row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const uint16_t *w2, const uint16_t *w3,
                                    const uint16_t *w4, const uint16_t *w5,
                                    const float *x, int n) {
    dst[0] = vec_dot_f16_f32(w0, x, n); dst[1] = vec_dot_f16_f32(w1, x, n);
    dst[2] = vec_dot_f16_f32(w2, x, n); dst[3] = vec_dot_f16_f32(w3, x, n);
    dst[4] = vec_dot_f16_f32(w4, x, n); dst[5] = vec_dot_f16_f32(w5, x, n);
}
static inline void matvec_f16_4row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const uint16_t *w2, const uint16_t *w3,
                                    const float *x, int n) {
    dst[0] = vec_dot_f16_f32(w0, x, n); dst[1] = vec_dot_f16_f32(w1, x, n);
    dst[2] = vec_dot_f16_f32(w2, x, n); dst[3] = vec_dot_f16_f32(w3, x, n);
}
static inline void matvec_f16_2row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const float *x, int n) {
    dst[0] = vec_dot_f16_f32(w0, x, n); dst[1] = vec_dot_f16_f32(w1, x, n);
}
static inline void gemm_f16_f32(float *Y, const uint16_t *W, const float *X,
                                 int n_rows, int K, int N, int Y_stride, int X_stride) {
    for (int r = 0; r < n_rows; r++) {
        const uint16_t *wr = W + (size_t)r * K;
        for (int t = 0; t < N; t++) {
            Y[r * Y_stride + t] = vec_dot_f16_f32(wr, X + (size_t)t * X_stride, K);
        }
    }
}
static inline void gemm_f16_f32_tokmajor(float *Y, const uint16_t *W, const float *X,
                                          int n_rows, int K, int N, int Y_stride, int X_stride) {
    for (int r = 0; r < n_rows; r++) {
        const uint16_t *wr = W + (size_t)r * K;
        for (int t = 0; t < N; t++) {
            Y[t * Y_stride + r] = vec_dot_f16_f32(wr, X + (size_t)t * X_stride, K);
        }
    }
}
static inline void gemm_f16_f32_tokmajor_fused2(
        float *Y1, const uint16_t *W1, float *Y2, const uint16_t *W2,
        const float *X, int n_rows, int K, int N,
        int Y1_stride, int Y2_stride, int X_stride) {
    for (int r = 0; r < n_rows; r++) {
        const uint16_t *w1r = W1 + (size_t)r * K;
        const uint16_t *w2r = W2 + (size_t)r * K;
        for (int t = 0; t < N; t++) {
            const float *xt = X + (size_t)t * X_stride;
            Y1[t * Y1_stride + r] = vec_dot_f16_f32(w1r, xt, K);
            Y2[t * Y2_stride + r] = vec_dot_f16_f32(w2r, xt, K);
        }
    }
}
/* Scalar Q8_0 fallbacks */
static inline float vec_dot_q8_0_f32(const void *q8_row, const float *x, int K) {
    const block_q8_0 *blocks = (const block_q8_0 *)q8_row;
    int nb = K / 32;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        float scale = ggml_fp16_to_fp32(blocks[b].d);
        for (int j = 0; j < 32; j++) sum += scale * blocks[b].qs[j] * x[b * 32 + j];
    }
    return sum;
}
static inline void matvec_q8_0_f32(float *dst, const void *q8_row, const float *x, int K) {
    *dst = vec_dot_q8_0_f32(q8_row, x, K);
}
static inline void gemm_q8_0_f32_tokmajor(float *Y, const void *W, const float *X,
                                            int n_rows, int K, int N, int Y_stride, int X_stride) {
    int nb = K / 32;
    size_t row_bytes = (size_t)nb * sizeof(block_q8_0);
    for (int r = 0; r < n_rows; r++) {
        const void *wr = (const uint8_t *)W + (size_t)r * row_bytes;
        for (int t = 0; t < N; t++)
            Y[t * Y_stride + r] = vec_dot_q8_0_f32(wr, X + (size_t)t * X_stride, K);
    }
}
#endif /* __F16C__ && __AVX2__ && __FMA__ */

/* ======================================================================== */
#ifdef GGML_DEQUANT_IMPLEMENTATION

#include <string.h>
#include <math.h>

void dequantize_row_q8_0(const void *src, float *dst, int n) {
    const int nb = n / 32;
    const block_q8_0 *blocks = (const block_q8_0 *)src;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            dst[i * 32 + j] = d * blocks[i].qs[j];
        }
    }
}

void dequantize_row_q2_K(const void *src, float *dst, int n) {
    const int nb = n / 256;
    const block_q2_K *blocks = (const block_q2_K *)src;

    for (int i = 0; i < nb; i++) {
        const float d   = ggml_fp16_to_fp32(blocks[i].d);
        const float min = ggml_fp16_to_fp32(blocks[i].dmin);
        const uint8_t *q = blocks[i].qs;
        float *y = dst + i * 256;

        int is = 0;
        for (int n0 = 0; n0 < 256; n0 += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = blocks[i].scales[is++];
                const float dl = d * (sc & 0xF);
                const float ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l] >> shift) & 3) - ml;

                sc = blocks[i].scales[is++];
                const float dl2 = d * (sc & 0xF);
                const float ml2 = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl2 * ((q[l + 16] >> shift) & 3) - ml2;

                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_row_q3_K(const void *src, float *dst, int n) {
    const int nb = n / 256;
    const block_q3_K *blocks = (const block_q3_K *)src;
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;

    for (int i = 0; i < nb; i++) {
        const float d_all = ggml_fp16_to_fp32(blocks[i].d);
        const uint8_t *q = blocks[i].qs;
        const uint8_t *hm = blocks[i].hmask;
        float *y = dst + i * 256;
        uint8_t m = 1;

        uint32_t aux[4] = {0, 0, 0, 0};
        memcpy(aux, blocks[i].scales, 12);
        {
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        }
        const int8_t *scales = (const int8_t *)aux;

        int is = 0;
        for (int n0 = 0; n0 < 256; n0 += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                const float dl1 = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl1 * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }

                const float dl2 = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl2 * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_row_q4_K(const void *src, float *dst, int n) {
    const int nb = n / 256;
    const block_q4_K *blocks = (const block_q4_K *)src;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        const float d = ggml_fp16_to_fp32(b->d);
        const float dmin = ggml_fp16_to_fp32(b->dmin);
        float *y = dst + i * 256;

        const uint8_t *q = b->qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m_val;
            get_scale_min_k4(is + 0, b->scales, &sc, &m_val);
            const float d1 = d * sc;
            const float m1 = dmin * m_val;
            get_scale_min_k4(is + 1, b->scales, &sc, &m_val);
            const float d2 = d * sc;
            const float m2 = dmin * m_val;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void dequantize_row_q5_K(const void *src, float *dst, int n) {
    const int nb = n / 256;
    const block_q5_K *blocks = (const block_q5_K *)src;

    for (int i = 0; i < nb; i++) {
        const uint8_t *ql = blocks[i].qs;
        const uint8_t *qh = blocks[i].qh;
        const float d = ggml_fp16_to_fp32(blocks[i].d);
        const float min = ggml_fp16_to_fp32(blocks[i].dmin);
        float *y = dst + i * 256;

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dequantize_row_q6_K(const void *src, float *dst, int n) {
    const int nb = n / 256;
    const block_q6_K *blocks = (const block_q6_K *)src;

    for (int i = 0; i < nb; i++) {
        const block_q6_K *b = &blocks[i];
        const float d = ggml_fp16_to_fp32(b->d);
        float *y = dst + i * 256;

        /* 256 elements processed in two halves of 128.
         * Each half uses 64 bytes ql, 32 bytes qh, 8 scale values.
         * Within each half, 32 iterations produce 4 outputs each. */
        const uint8_t *ql = b->ql;
        const uint8_t *qh = b->qh;
        const int8_t  *sc = b->scales;

        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

int dequant_row(uint32_t type, const void *src, float *dst, int n) {
    switch (type) {
        case GGML_TYPE_Q2_K:
            dequantize_row_q2_K(src, dst, n);
            return 0;
        case GGML_TYPE_Q3_K:
            dequantize_row_q3_K(src, dst, n);
            return 0;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0(src, dst, n);
            return 0;
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K(src, dst, n);
            return 0;
        case GGML_TYPE_Q5_K:
            dequantize_row_q5_K(src, dst, n);
            return 0;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K(src, dst, n);
            return 0;
        case GGML_TYPE_F32:
            memcpy(dst, src, n * sizeof(float));
            return 0;
        case GGML_TYPE_F16: {
            const uint16_t *s = (const uint16_t *)src;
            for (int i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(s[i]);
            return 0;
        }
        default:
            return -1;
    }
}

#endif /* GGML_DEQUANT_IMPLEMENTATION */
#endif /* GGML_DEQUANT_H */
