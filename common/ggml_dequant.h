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

/* IQ2_XXS block: 256 elements, 66 bytes (2.0625 bpw) */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint16_t qs[32];    /* packed grid indices + signs + sub-block scales */
} block_iq2_xxs;

/* Q4_0 block: 32 elements, 18 bytes */
typedef struct {
    uint16_t d;         /* block scale (fp16) */
    uint8_t  qs[16];    /* nibbles / quants */
} block_q4_0;

/* Q4_1 block: 32 elements, 20 bytes */
typedef struct {
    uint16_t d;         /* delta (fp16) */
    uint16_t m;         /* min (fp16) */
    uint8_t  qs[16];    /* nibbles / quants */
} block_q4_1;

/* Q5_0 block: 32 elements, 22 bytes */
typedef struct {
    uint16_t d;         /* block scale (fp16) */
    uint8_t  qh[4];     /* 5-th bit of quants */
    uint8_t  qs[16];    /* nibbles / quants */
} block_q5_0;

/* Q5_1 block: 32 elements, 24 bytes */
typedef struct {
    uint16_t d;         /* delta (fp16) */
    uint16_t m;         /* min (fp16) */
    uint8_t  qh[4];     /* 5-th bit of quants */
    uint8_t  qs[16];    /* nibbles / quants */
} block_q5_1;

/* IQ2_XS block: 256 elements, 74 bytes (2.3125 bpw) */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint16_t qs[32];    /* grid indices + signs */
    uint8_t  scales[8]; /* sub-block scales */
} block_iq2_xs;

/* IQ2_S block: 256 elements, 82 bytes (2.5625 bpw) */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint8_t  qs[64];    /* grid indices (low 8 bits) */
    uint8_t  qh[8];     /* grid indices (high bits) */
    uint8_t  scales[8]; /* sub-block scales */
} block_iq2_s;

/* IQ3_XXS block: 256 elements, 98 bytes (3.0625 bpw) */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint8_t  qs[96];    /* 3*QK_K/8 = quants + scales_and_signs */
} block_iq3_xxs;

/* IQ3_S block: 256 elements, 110 bytes (3.4375 bpw) */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint8_t  qs[64];    /* quants low 8 bits */
    uint8_t  qh[8];     /* quants high bit */
    uint8_t  signs[32]; /* sign bits */
    uint8_t  scales[4]; /* sub-block scales */
} block_iq3_s;

/* IQ4_NL block: 32 elements, 18 bytes (non-linear 4-bit) */
typedef struct {
    uint16_t d;         /* block scale (fp16) */
    uint8_t  qs[16];    /* nibbles / quants */
} block_iq4_nl;

/* IQ4_XS block: 256 elements, 136 bytes */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint16_t scales_h;  /* high 2 bits of scales */
    uint8_t  scales_l[4]; /* low 4 bits of scales */
    uint8_t  qs[128];   /* nibbles / quants */
} block_iq4_xs;

/* IQ1_S block: 256 elements, 50 bytes (1.5625 bpw) */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint8_t  qs[32];    /* grid index, low 8 bits */
    uint16_t qh[8];     /* grid index high bits + scale + delta */
} block_iq1_s;

/* IQ1_M block: 256 elements, 56 bytes (1.75 bpw) */
typedef struct {
    uint8_t  qs[32];    /* grid index, low 8 bits */
    uint8_t  qh[16];    /* grid index high 3 bits + shift bit */
    uint8_t  scales[8]; /* 3-bit block scales (packing d in high bits) */
} block_iq1_m;

/* IQ1_M scale union */
typedef union {
    uint16_t f16;
    uint16_t u16;
} iq1m_scale_t;

/* TQ1_0 block: 256 elements, 54 bytes (1.6875 bpw) */
typedef struct {
    uint8_t  qs[48];    /* 5 trits per byte (3^5=243 < 256) */
    uint8_t  qh[4];     /* 4 trits per byte */
    uint16_t d;         /* block scale (fp16) */
} block_tq1_0;

/* TQ2_0 block: 256 elements, 66 bytes (2.0625 bpw) */
typedef struct {
    uint8_t  qs[64];    /* 2 bits per element */
    uint16_t d;         /* block scale (fp16) */
} block_tq2_0;

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
void dequantize_row_iq2_xxs(const void *src, float *dst, int n);
void dequantize_row_q4_0(const void *src, float *dst, int n);
void dequantize_row_q4_1(const void *src, float *dst, int n);
void dequantize_row_q5_0(const void *src, float *dst, int n);
void dequantize_row_q5_1(const void *src, float *dst, int n);
void dequantize_row_iq2_xs(const void *src, float *dst, int n);
void dequantize_row_iq2_s(const void *src, float *dst, int n);
void dequantize_row_iq3_xxs(const void *src, float *dst, int n);
void dequantize_row_iq3_s(const void *src, float *dst, int n);
void dequantize_row_iq4_nl(const void *src, float *dst, int n);
void dequantize_row_iq4_xs(const void *src, float *dst, int n);
void dequantize_row_iq1_s(const void *src, float *dst, int n);
void dequantize_row_iq1_m(const void *src, float *dst, int n);
void dequantize_row_tq1_0(const void *src, float *dst, int n);
void dequantize_row_tq2_0(const void *src, float *dst, int n);
void dequantize_row_bf16(const void *src, float *dst, int n);

/* Dequantize a row. Returns 0 on success, -1 if type unsupported. */
int dequant_row(uint32_t ggml_type, const void *src, float *dst, int n);

/* Get the byte size of one row of n elements in a given GGML type. */
static inline size_t dequant_row_size(uint32_t type, int n) {
    int bs = 1, ts = 4;
    switch (type) {
        case GGML_TYPE_Q2_K:    bs = 256; ts = 84;  break;
        case GGML_TYPE_Q3_K:    bs = 256; ts = 110; break;
        case GGML_TYPE_Q8_0:    bs = 32;  ts = 34;  break;
        case GGML_TYPE_Q4_K:    bs = 256; ts = 144; break;
        case GGML_TYPE_Q5_K:    bs = 256; ts = 176; break;
        case GGML_TYPE_Q6_K:    bs = 256; ts = 210; break;
        case GGML_TYPE_IQ2_XXS: bs = 256; ts = 66;  break;
        case GGML_TYPE_Q4_0:    bs = 32;  ts = 18;  break;
        case GGML_TYPE_Q4_1:    bs = 32;  ts = 20;  break;
        case GGML_TYPE_Q5_0:    bs = 32;  ts = 22;  break;
        case GGML_TYPE_Q5_1:    bs = 32;  ts = 24;  break;
        case GGML_TYPE_IQ2_XS:  bs = 256; ts = 74;  break;
        case GGML_TYPE_IQ2_S:   bs = 256; ts = 82;  break;
        case GGML_TYPE_IQ3_XXS: bs = 256; ts = 98;  break;
        case GGML_TYPE_IQ3_S:   bs = 256; ts = 110; break;
        case GGML_TYPE_IQ4_NL:  bs = 32;  ts = 18;  break;
        case GGML_TYPE_IQ4_XS:  bs = 256; ts = 136; break;
        case GGML_TYPE_IQ1_S:   bs = 256; ts = 50;  break;
        case GGML_TYPE_IQ1_M:   bs = 256; ts = 56;  break;
        case GGML_TYPE_TQ1_0:   bs = 256; ts = 54;  break;
        case GGML_TYPE_TQ2_0:   bs = 256; ts = 66;  break;
        case GGML_TYPE_F32:     bs = 1;   ts = 4;   break;
        case GGML_TYPE_F16:     bs = 1;   ts = 2;   break;
        case GGML_TYPE_BF16:    bs = 1;   ts = 2;   break;
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

void dequantize_row_bf16(const void *src, float *dst, int n) {
    const uint16_t *s = (const uint16_t *)src;
    for (int i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)s[i] << 16;  /* BF16 → F32: pad mantissa with zeros */
        memcpy(&dst[i], &bits, sizeof(float));
    }
}

/* IQ2_XXS lookup tables (from ggml-common.h) */
static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint8_t kmask_iq2xs[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static const uint32_t iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

static const uint64_t iq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

static const uint64_t iq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

static const uint64_t iq1s_grid[2048] = {
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
    0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
    0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000,
    0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
    0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101,
    0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
    0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00,
    0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
    0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100,
    0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
    0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000,
    0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
    0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101,
    0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
    0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100,
    0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
    0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000,
    0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
    0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001,
    0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
    0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00,
    0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
    0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff,
    0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
    0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff,
    0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
    0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101,
    0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
    0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000,
    0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
    0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01,
    0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
    0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00,
    0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
    0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff,
    0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
    0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff,
    0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
    0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101,
    0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
    0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff,
    0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
    0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001,
    0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
    0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff,
    0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
    0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000,
    0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
    0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01,
    0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
    0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000,
    0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
    0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000,
    0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
    0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff,
    0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
    0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff,
    0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
    0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000,
    0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
    0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00,
    0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
    0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100,
    0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
    0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00,
    0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
    0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000,
    0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
    0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff,
    0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
    0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000,
    0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
    0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100,
    0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
    0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101,
    0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
    0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000,
    0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
    0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01,
    0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
    0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000,
    0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
    0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101,
    0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
    0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff,
    0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
    0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001,
    0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
    0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000,
    0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
    0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00,
    0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
    0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01,
    0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
    0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff,
    0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
    0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00,
    0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
    0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff,
    0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
    0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000,
    0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
    0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff,
    0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
    0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00,
    0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
    0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001,
    0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
    0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100,
    0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
    0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101,
    0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
    0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00,
    0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
    0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000,
    0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
    0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff,
    0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
    0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101,
    0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
    0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100,
    0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
    0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01,
    0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
    0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00,
    0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
    0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101,
    0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
    0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001,
    0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
    0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00,
    0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
    0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100,
    0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
    0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01,
    0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
    0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101,
    0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
    0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100,
    0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
    0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff,
    0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
    0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000,
    0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
    0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000,
    0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
    0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101,
    0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
    0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01,
    0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
    0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001,
    0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
    0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff,
    0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
    0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000,
    0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
    0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff,
    0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
    0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000,
    0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
    0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00,
    0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
    0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000,
    0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
    0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff,
    0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
    0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000,
    0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
    0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000,
    0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
    0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000,
    0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
    0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff,
    0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
    0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101,
    0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
    0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff,
    0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
    0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff,
    0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
    0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff,
    0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
    0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff,
    0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
    0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001,
    0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
    0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01,
    0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
    0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100,
    0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
    0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff,
    0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
    0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01,
    0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
    0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100,
    0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
    0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000,
    0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
    0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000,
    0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
    0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001,
    0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
    0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00,
    0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
    0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00,
    0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
    0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff,
    0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
    0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101,
    0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
    0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff,
    0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
    0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff,
    0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
    0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000,
    0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
    0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100,
    0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
    0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100,
    0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
    0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00,
    0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
    0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff,
    0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
    0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff,
    0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
    0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00,
    0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00,
    0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
    0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff,
    0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
    0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001,
    0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
    0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101,
    0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
    0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000,
    0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
    0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000,
    0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
    0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000,
    0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
    0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101,
    0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101,
    0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
    0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff,
    0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
    0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff,
    0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
    0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100,
    0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101,
    0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
    0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01,
    0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
    0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
    0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00,
    0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
    0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff,
    0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
    0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001,
    0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
    0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff,
    0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
    0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff,
    0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
    0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000,
    0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
    0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100,
    0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
    0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff,
    0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
    0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000,
    0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
    0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01,
    0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
    0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100,
    0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
    0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101,
    0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
    0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01,
    0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
    0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff,
    0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
    0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff,
    0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
    0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001,
    0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
    0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff,
    0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
    0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00,
    0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
    0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100,
    0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
    0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001,
    0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
    0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff,
    0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
    0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01,
    0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
    0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001,
    0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
    0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101,
    0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
    0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00,
    0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
    0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001,
    0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
    0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000,
    0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
    0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001,
    0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
    0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000,
    0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
    0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00,
    0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
    0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff,
    0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
    0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001,
    0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
    0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100,
    0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
    0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00,
    0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
    0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001,
    0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
    0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff,
    0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
    0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100,
    0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
    0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101,
    0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
    0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff,
    0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
    0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00,
    0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
    0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff,
    0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
    0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100,
    0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
    0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00,
    0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
    0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff,
    0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
    0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001,
    0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
    0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff,
    0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
    0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff,
    0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
    0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001,
    0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
    0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101,
    0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
    0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001,
    0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
    0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101,
    0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
    0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff,
    0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
    0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101,
    0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
    0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff,
    0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
    0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff,
    0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
    0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000,
    0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
    0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff,
    0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
    0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff,
    0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
    0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100,
    0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
    0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100,
    0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
    0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001,
    0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
    0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000,
    0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
    0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00,
    0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
    0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001,
    0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
    0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101,
    0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
    0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000,
    0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
    0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000,
    0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
    0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff,
    0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
    0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000,
    0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
    0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff,
    0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
    0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01,
    0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
    0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000,
    0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
    0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff,
    0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
    0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff,
    0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
    0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00,
    0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
    0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000,
    0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
    0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001,
    0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
    0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000,
    0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
    0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000,
    0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
    0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000,
    0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
    0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001,
    0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
    0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001,
    0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
    0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000,
    0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
    0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000,
    0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
    0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001,
    0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
    0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001,
    0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
    0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000,
    0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
    0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100,
    0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
    0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001,
    0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
    0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff,
    0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
    0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00,
    0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
    0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100,
    0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
    0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff,
    0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
    0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff,
    0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
    0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff,
    0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
    0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff,
    0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
    0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001,
    0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
    0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000,
    0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
    0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff,
    0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
    0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001,
    0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
    0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000,
    0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
    0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff,
    0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
    0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000,
    0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00,
    0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
    0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff,
    0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
    0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00,
    0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
    0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000,
    0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
    0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff,
    0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
    0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff,
    0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
    0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100,
    0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
    0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101,
    0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
    0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff,
    0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
    0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000,
    0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
    0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00,
    0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
    0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff,
    0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
    0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00,
    0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
    0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff,
    0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
    0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00,
    0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
    0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01,
    0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
    0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000,
    0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
    0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff,
    0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
    0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001,
    0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
    0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01,
    0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
    0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00,
    0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
    0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001,
    0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
    0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101,
    0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
    0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000,
    0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
};

void dequantize_row_iq2_xxs(const void *src, float *dst, int n) {
    const int nb = n / 256;
    const block_iq2_xxs *blocks = (const block_iq2_xxs *)src;
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(blocks[i].d);
        float *y = dst + i * 256;

        for (int ib32 = 0; ib32 < 8; ++ib32) {
            memcpy(aux32, blocks[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

void dequantize_row_q4_0(const void *restrict vx, float *restrict y, int k) {
    const block_q4_0 *restrict x = (const block_q4_0 *)vx;
    const int nb = k / 32;
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < 16; j++) {
            const int v0 = (x[i].qs[j] & 0x0F) - 8;
            const int v1 = (x[i].qs[j] >>    4) - 8;
            y[i*32 + j +  0] = v0 * d;
            y[i*32 + j + 16] = v1 * d;
        }
    }
}

void dequantize_row_q4_1(const void *restrict vx, float *restrict y, int k) {
    const block_q4_1 *restrict x = (const block_q4_1 *)vx;
    const int nb = k / 32;
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const float m = ggml_fp16_to_fp32(x[i].m);
        for (int j = 0; j < 16; j++) {
            const int v0 = (x[i].qs[j] & 0x0F);
            const int v1 = (x[i].qs[j] >>    4);
            y[i*32 + j +  0] = v0 * d + m;
            y[i*32 + j + 16] = v1 * d + m;
        }
    }
}

void dequantize_row_q5_0(const void *restrict vx, float *restrict y, int k) {
    const block_q5_0 *restrict x = (const block_q5_0 *)vx;
    const int nb = k / 32;
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < 16; j++) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>    4) | xh_1) - 16;
            y[i*32 + j +  0] = x0 * d;
            y[i*32 + j + 16] = x1 * d;
        }
    }
}

void dequantize_row_q5_1(const void *restrict vx, float *restrict y, int k) {
    const block_q5_1 *restrict x = (const block_q5_1 *)vx;
    const int nb = k / 32;
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const float m = ggml_fp16_to_fp32(x[i].m);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < 16; j++) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int32_t x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int32_t x1 = (x[i].qs[j] >>    4) | xh_1;
            y[i*32 + j +  0] = x0 * d + m;
            y[i*32 + j + 16] = x1 * d + m;
        }
    }
}

void dequantize_row_iq3_xxs(const void *restrict vx, float *restrict y, int k) {
    const block_iq3_xxs *restrict x = (const block_iq3_xxs *)vx;
    const int nb = k / 256;
    uint32_t aux32;
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *scales_and_signs = qs + 256/4;
        for (int ib32 = 0; ib32 < 256/32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

void dequantize_row_iq4_nl(const void *restrict vx, float *restrict y, int k) {
    const block_iq4_nl *restrict x = (const block_iq4_nl *)vx;
    const int nb = k / 32;
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = x[i].qs;
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < 16; ++j) {
            y[j +  0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = d * kvalues_iq4nl[qs[j] >>  4];
        }
        y  += 32;
    }
}

void dequantize_row_iq4_xs(const void *restrict vx, float *restrict y, int k) {
    const block_iq4_xs *restrict x = (const block_iq4_xs *)vx;
    const int nb = k / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = x[i].qs;
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int ib = 0; ib < 256/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

void dequantize_row_iq2_xs(const void *restrict vx, float *restrict y, int k) {
    const block_iq2_xs *restrict x = (const block_iq2_xs *)vx;
    const int nb = k / 256;
    float db[2];
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int ib32 = 0; ib32 < 256/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

void dequantize_row_iq2_s(const void *restrict vx, float *restrict y, int k) {
    const block_iq2_s *restrict x = (const block_iq2_s *)vx;
    const int nb = k / 256;
    float db[2];
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + 256/8;
        for (int ib32 = 0; ib32 < 256/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l/2];
                const uint8_t *grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

void dequantize_row_iq3_s(const void *restrict vx, float *restrict y, int k) {
    const block_iq3_s *restrict x = (const block_iq3_s *)vx;
    const int nb = k / 256;
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = x[i].signs;
        for (int ib32 = 0; ib32 < 256/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

#define IQ1S_DELTA 0.125f

void dequantize_row_iq1_s(const void *restrict vx, float *restrict y, int k) {
    const block_iq1_s *restrict x = (const block_iq1_s *)vx;
    const int nb = k / 256;
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const uint8_t  *qs = x[i].qs;
        const uint16_t *qh = x[i].qh;
        for (int ib = 0; ib < 256/32; ++ib) {
            const float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
            const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t *grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (grid[j] + delta);
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

void dequantize_row_iq1_m(const void *restrict vx, float *restrict y, int k) {
    const block_iq1_m *restrict x = (const block_iq1_m *)vx;
    const int nb = k / 256;
    float delta[4];
    uint16_t idx[4];
    iq1m_scale_t scale;
    for (int i = 0; i < nb; i++) {
        const uint16_t *sc = (const uint16_t *)x[i].scales;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = ggml_fp16_to_fp32(scale.f16);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        for (int ib = 0; ib < 256/32; ++ib) {
            const float dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
            const float dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);
            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 2; ++l) {
                const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl1 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t *grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl2 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

void dequantize_row_tq1_0(const void *restrict vx, float *restrict y, int k) {
    const block_tq1_0 *restrict x = (const block_tq1_0 *)vx;
    const int nb = k / 256;
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    for (int64_t i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }
        for (size_t n = 0; n < 4; ++n) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (float) (xi - 1) * d;
            }
        }
    }
}

void dequantize_row_tq2_0(const void *restrict vx, float *restrict y, int k) {
    const block_tq2_0 *restrict x = (const block_tq2_0 *)vx;
    const int nb = k / 256;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (float) (q - 1) * d;
                }
            }
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
        case GGML_TYPE_IQ2_XXS:
            dequantize_row_iq2_xxs(src, dst, n);
            return 0;
        case GGML_TYPE_Q4_0:
            dequantize_row_q4_0(src, dst, n);
            return 0;
        case GGML_TYPE_Q4_1:
            dequantize_row_q4_1(src, dst, n);
            return 0;
        case GGML_TYPE_Q5_0:
            dequantize_row_q5_0(src, dst, n);
            return 0;
        case GGML_TYPE_Q5_1:
            dequantize_row_q5_1(src, dst, n);
            return 0;
        case GGML_TYPE_IQ2_XS:
            dequantize_row_iq2_xs(src, dst, n);
            return 0;
        case GGML_TYPE_IQ2_S:
            dequantize_row_iq2_s(src, dst, n);
            return 0;
        case GGML_TYPE_IQ3_XXS:
            dequantize_row_iq3_xxs(src, dst, n);
            return 0;
        case GGML_TYPE_IQ3_S:
            dequantize_row_iq3_s(src, dst, n);
            return 0;
        case GGML_TYPE_IQ4_NL:
            dequantize_row_iq4_nl(src, dst, n);
            return 0;
        case GGML_TYPE_IQ4_XS:
            dequantize_row_iq4_xs(src, dst, n);
            return 0;
        case GGML_TYPE_IQ1_S:
            dequantize_row_iq1_s(src, dst, n);
            return 0;
        case GGML_TYPE_IQ1_M:
            dequantize_row_iq1_m(src, dst, n);
            return 0;
        case GGML_TYPE_TQ1_0:
            dequantize_row_tq1_0(src, dst, n);
            return 0;
        case GGML_TYPE_TQ2_0:
            dequantize_row_tq2_0(src, dst, n);
            return 0;
        case GGML_TYPE_F32:
            memcpy(dst, src, n * sizeof(float));
            return 0;
        case GGML_TYPE_F16: {
            const uint16_t *s = (const uint16_t *)src;
            for (int i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(s[i]);
            return 0;
        }
        case GGML_TYPE_BF16:
            dequantize_row_bf16(src, dst, n);
            return 0;
        default:
            return -1;
    }
}

#endif /* GGML_DEQUANT_IMPLEMENTATION */
#endif /* GGML_DEQUANT_H */
