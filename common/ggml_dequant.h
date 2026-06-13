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
#include <stdlib.h>
#include <string.h>
#include "gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Q8_0 block: 32 elements, 34 bytes */
typedef struct {
    uint16_t d;       /* block scale (fp16) */
    int8_t   qs[32];  /* quantized values */
} block_q8_0;

/* Q1_0 block: 128 elements, 18 bytes */
typedef struct {
    uint16_t d;       /* block scale (fp16) */
    uint8_t  qs[16];  /* bits / quants */
} block_q1_0;

/* MXFP4 block: 32 elements, 17 bytes */
typedef struct {
    uint8_t e;        /* E8M0 scale */
    uint8_t qs[16];   /* packed 4-bit E2M1 values */
} block_mxfp4;

/* NVFP4 block: 64 elements, 36 bytes */
typedef struct {
    uint8_t d[4];     /* UE4M3 scales, one per 16-element sub-block */
    uint8_t qs[32];   /* packed 4-bit E2M1 values */
} block_nvfp4;

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

/* Round-to-nearest f32 -> f16 (IEEE half). Used to encode per-block quant
 * scales, which are always small positive values (absmax/127), so the
 * subnormal/overflow branches below are defensive rather than hot. */
static inline uint16_t ggml_fp32_to_fp16(float f) {
    union { float f; uint32_t u; } u; u.f = f;
    uint32_t bits = u.u;
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0)       return (uint16_t)sign;
    if (exp >= 31)      return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

void dequantize_row_q2_K(const void *src, float *dst, int n);
void dequantize_row_q3_K(const void *src, float *dst, int n);
void dequantize_row_q8_0(const void *src, float *dst, int n);
void dequantize_row_q1_0(const void *src, float *dst, int n);
void dequantize_row_mxfp4(const void *src, float *dst, int n);
void dequantize_row_nvfp4(const void *src, float *dst, int n);
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
        case GGML_TYPE_Q1_0:    bs = 128; ts = 18;  break;
        case GGML_TYPE_MXFP4:   bs = 32;  ts = 17;  break;
        case GGML_TYPE_NVFP4:   bs = 64;  ts = 36;  break;
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

/* BF16·F32 dot product: sum(a_bf16[i] * b_f32[i]) for i in [0, n).
 * Uses SVE on aarch64, AVX2 on x86, scalar fallback otherwise.
 * BF16 → F32 conversion: (uint32_t)bf16 << 16 (zero-pad mantissa). */
static inline float vec_dot_bf16_f32(const uint16_t *a, const float *b, int n);

/* Multi-row BF16 matvec: compute 4 output rows simultaneously. */
static inline void matvec_bf16_4row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                     const uint16_t *w2, const uint16_t *w3,
                                     const float *x, int n);

/* BF16 token-major GEMM: Y[tok * Y_stride + row] = dot(W[row,:], X[tok,:])
 * W is BF16 packed: each row is K * 2 bytes. */
static inline void gemm_bf16_f32_tokmajor(float *Y, const uint16_t *W, const float *X,
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

#elif defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

/* SVE F16→F32 conversion helper: load 16 FP16 half-words zero-extended into
 * S lanes (low 16 bits hold the FP16 datum), then FCVT.S.H to FP32.
 * Mirrors the LD1H{Z.S}+FCVT pattern used by a64fx/vlm's FP16 microkernel. */
#define SVE_F16_TO_F32(pg, ptr) \
    svcvt_f32_f16_x((pg), svreinterpret_f16( \
        svreinterpret_u16(svld1uh_u32((pg), (const uint16_t *)(ptr)))))

static inline float vec_dot_f16_f32(const uint16_t *a, const float *b, int n) {
    int vl = (int)svcntw();
    int stride4 = 4 * vl;
    svfloat32_t acc0 = svdup_f32(0.0f), acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f), acc3 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int i = 0;
    for (; i + stride4 - 1 < n; i += stride4) {
        svfloat32_t va0 = SVE_F16_TO_F32(pg, &a[i]);
        svfloat32_t va1 = SVE_F16_TO_F32(pg, &a[i + vl]);
        svfloat32_t va2 = SVE_F16_TO_F32(pg, &a[i + 2 * vl]);
        svfloat32_t va3 = SVE_F16_TO_F32(pg, &a[i + 3 * vl]);
        svfloat32_t vb0 = svld1(pg, &b[i]);
        svfloat32_t vb1 = svld1(pg, &b[i + vl]);
        svfloat32_t vb2 = svld1(pg, &b[i + 2 * vl]);
        svfloat32_t vb3 = svld1(pg, &b[i + 3 * vl]);
        acc0 = svmla_x(pg, acc0, va0, vb0);
        acc1 = svmla_x(pg, acc1, va1, vb1);
        acc2 = svmla_x(pg, acc2, va2, vb2);
        acc3 = svmla_x(pg, acc3, va3, vb3);
    }
    for (; i < n; i += vl) {
        svbool_t ptail = svwhilelt_b32(i, n);
        acc0 = svmla_m(ptail, acc0, SVE_F16_TO_F32(ptail, &a[i]), svld1(ptail, &b[i]));
    }
    return svaddv(pg, svadd_x(pg, svadd_x(pg, acc0, acc1), svadd_x(pg, acc2, acc3)));
}

/* 6-row F16 matvec: 6 weight streams share one activation vector.
 * 6 FMAs per activation load → strong reuse of L1 activation cache. */
static inline void matvec_f16_6row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const uint16_t *w2, const uint16_t *w3,
                                    const uint16_t *w4, const uint16_t *w5,
                                    const float *x, int n) {
    int vl = (int)svcntw();
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svfloat32_t a4 = svdup_f32(0.0f), a5 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int i = 0;
    for (; i + vl - 1 < n; i += vl) {
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, SVE_F16_TO_F32(pg, &w0[i]), vx);
        a1 = svmla_x(pg, a1, SVE_F16_TO_F32(pg, &w1[i]), vx);
        a2 = svmla_x(pg, a2, SVE_F16_TO_F32(pg, &w2[i]), vx);
        a3 = svmla_x(pg, a3, SVE_F16_TO_F32(pg, &w3[i]), vx);
        a4 = svmla_x(pg, a4, SVE_F16_TO_F32(pg, &w4[i]), vx);
        a5 = svmla_x(pg, a5, SVE_F16_TO_F32(pg, &w5[i]), vx);
    }
    if (i < n) {
        svbool_t ptail = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1(ptail, &x[i]);
        a0 = svmla_m(ptail, a0, SVE_F16_TO_F32(ptail, &w0[i]), vx);
        a1 = svmla_m(ptail, a1, SVE_F16_TO_F32(ptail, &w1[i]), vx);
        a2 = svmla_m(ptail, a2, SVE_F16_TO_F32(ptail, &w2[i]), vx);
        a3 = svmla_m(ptail, a3, SVE_F16_TO_F32(ptail, &w3[i]), vx);
        a4 = svmla_m(ptail, a4, SVE_F16_TO_F32(ptail, &w4[i]), vx);
        a5 = svmla_m(ptail, a5, SVE_F16_TO_F32(ptail, &w5[i]), vx);
    }
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
}

static inline void matvec_f16_4row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const uint16_t *w2, const uint16_t *w3,
                                    const float *x, int n) {
    int vl = (int)svcntw();
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int i = 0;
    for (; i + vl - 1 < n; i += vl) {
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, SVE_F16_TO_F32(pg, &w0[i]), vx);
        a1 = svmla_x(pg, a1, SVE_F16_TO_F32(pg, &w1[i]), vx);
        a2 = svmla_x(pg, a2, SVE_F16_TO_F32(pg, &w2[i]), vx);
        a3 = svmla_x(pg, a3, SVE_F16_TO_F32(pg, &w3[i]), vx);
    }
    if (i < n) {
        svbool_t ptail = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1(ptail, &x[i]);
        a0 = svmla_m(ptail, a0, SVE_F16_TO_F32(ptail, &w0[i]), vx);
        a1 = svmla_m(ptail, a1, SVE_F16_TO_F32(ptail, &w1[i]), vx);
        a2 = svmla_m(ptail, a2, SVE_F16_TO_F32(ptail, &w2[i]), vx);
        a3 = svmla_m(ptail, a3, SVE_F16_TO_F32(ptail, &w3[i]), vx);
    }
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
}

static inline void matvec_f16_2row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                    const float *x, int n) {
    int vl = (int)svcntw();
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    int i = 0;
    for (; i + vl - 1 < n; i += vl) {
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, SVE_F16_TO_F32(pg, &w0[i]), vx);
        a1 = svmla_x(pg, a1, SVE_F16_TO_F32(pg, &w1[i]), vx);
    }
    if (i < n) {
        svbool_t ptail = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1(ptail, &x[i]);
        a0 = svmla_m(ptail, a0, SVE_F16_TO_F32(ptail, &w0[i]), vx);
        a1 = svmla_m(ptail, a1, SVE_F16_TO_F32(ptail, &w1[i]), vx);
    }
    dst[0] = svaddv(pg, a0);
    dst[1] = svaddv(pg, a1);
}

/* SVE Q8_0 dot: 32 int8 quants per block × FP16 scale. */
static inline float vec_dot_q8_0_f32(const void *q8_row, const float *x, int K) {
    const block_q8_0 *blocks = (const block_q8_0 *)q8_row;
    int nb = K / 32;
    svbool_t pg = svptrue_b32();
    svfloat32_t acc = svdup_f32(0.0f);
    int vl = (int)svcntw();  /* 16 on A64FX */
    /* 32 quants per block; on A64FX vl=16 → exactly 2 vector loads per block */
    for (int b = 0; b < nb; b++) {
        float scale = ggml_fp16_to_fp32(blocks[b].d);
        svfloat32_t vscale = svdup_f32(scale);
        const float *xb = x + b * 32;
        int j = 0;
        for (; j + vl - 1 < 32; j += vl) {
            svint32_t qi = svld1sb_s32(pg, (const int8_t *)(blocks[b].qs + j));
            svfloat32_t qf = svcvt_f32_s32_x(pg, qi);
            svfloat32_t vb = svld1(pg, xb + j);
            acc = svmla_x(pg, acc, svmul_x(pg, qf, vscale), vb);
        }
        for (; j < 32; j++) {
            float v = scale * (float)blocks[b].qs[j] * xb[j];
            acc = svadd_n_f32_x(pg, acc, v);
        }
    }
    return svaddv(pg, acc);
}

static inline void matvec_q8_0_f32(float *dst, const void *q8_row, const float *x, int K) {
    *dst = vec_dot_q8_0_f32(q8_row, x, K);
}

/* Generic GEMM wrappers: loop over rows/tokens calling the SIMD dot. */
static inline void gemm_f16_f32(float *Y, const uint16_t *W, const float *X,
                                 int n_rows, int K, int N, int Y_stride, int X_stride) {
    int r = 0;
    for (; r + 5 < n_rows; r += 6) {
        const uint16_t *w0 = W + (size_t)r * K, *w1 = W + (size_t)(r+1) * K;
        const uint16_t *w2 = W + (size_t)(r+2) * K, *w3 = W + (size_t)(r+3) * K;
        const uint16_t *w4 = W + (size_t)(r+4) * K, *w5 = W + (size_t)(r+5) * K;
        for (int t = 0; t < N; t++) {
            float tmp[6];
            matvec_f16_6row(tmp, w0, w1, w2, w3, w4, w5, X + (size_t)t * X_stride, K);
            Y[(r+0) * Y_stride + t] = tmp[0]; Y[(r+1) * Y_stride + t] = tmp[1];
            Y[(r+2) * Y_stride + t] = tmp[2]; Y[(r+3) * Y_stride + t] = tmp[3];
            Y[(r+4) * Y_stride + t] = tmp[4]; Y[(r+5) * Y_stride + t] = tmp[5];
        }
    }
    for (; r < n_rows; r++) {
        const uint16_t *wr = W + (size_t)r * K;
        for (int t = 0; t < N; t++)
            Y[r * Y_stride + t] = vec_dot_f16_f32(wr, X + (size_t)t * X_stride, K);
    }
}

static inline void gemm_f16_f32_tokmajor(float *Y, const uint16_t *W, const float *X,
                                          int n_rows, int K, int N, int Y_stride, int X_stride) {
    int r = 0;
    for (; r + 5 < n_rows; r += 6) {
        const uint16_t *w0 = W + (size_t)r * K, *w1 = W + (size_t)(r+1) * K;
        const uint16_t *w2 = W + (size_t)(r+2) * K, *w3 = W + (size_t)(r+3) * K;
        const uint16_t *w4 = W + (size_t)(r+4) * K, *w5 = W + (size_t)(r+5) * K;
        for (int t = 0; t < N; t++) {
            float tmp[6];
            matvec_f16_6row(tmp, w0, w1, w2, w3, w4, w5, X + (size_t)t * X_stride, K);
            Y[t * Y_stride + r + 0] = tmp[0]; Y[t * Y_stride + r + 1] = tmp[1];
            Y[t * Y_stride + r + 2] = tmp[2]; Y[t * Y_stride + r + 3] = tmp[3];
            Y[t * Y_stride + r + 4] = tmp[4]; Y[t * Y_stride + r + 5] = tmp[5];
        }
    }
    for (; r < n_rows; r++) {
        const uint16_t *wr = W + (size_t)r * K;
        for (int t = 0; t < N; t++)
            Y[t * Y_stride + r] = vec_dot_f16_f32(wr, X + (size_t)t * X_stride, K);
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

#undef SVE_F16_TO_F32

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

/* ---- BF16 dot product / matvec kernels ---- */
/* BF16 → FP32: zero-pad lower 16 mantissa bits, i.e. (uint32_t)bf16 << 16 */

static inline float bf16_to_f32_scalar(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

/* SVE BF16→FP32 conversion helper: load BF16 half-words, shift to FP32 */
#define SVE_BF16_TO_F32(pg, ptr) svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, ptr), 16))

static inline float vec_dot_bf16_f32(const uint16_t *a, const float *b, int n) {
    int i = 0;
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);
    int vl = (int)svcntw();  /* 16 FP32 elements on A64FX (512-bit SVE) */
    int stride4 = 4 * vl;
    svbool_t pg = svptrue_b32();

    /* Software-pipelined loop: prefetch 2 iterations ahead.
     * A64FX L2 latency ~40 cycles; prefetch distance = 2 * 4*vl * 2 bytes = 256 bytes */
    for (; i + stride4 - 1 < n; i += stride4) {
        /* Prefetch BF16 weights and FP32 activations 2 iterations ahead */
        /* No SW prefetch: A64FX HW stream prefetcher handles sequential patterns */
        svfloat32_t va0 = SVE_BF16_TO_F32(pg, &a[i]);
        svfloat32_t va1 = SVE_BF16_TO_F32(pg, &a[i + vl]);
        svfloat32_t va2 = SVE_BF16_TO_F32(pg, &a[i + 2 * vl]);
        svfloat32_t va3 = SVE_BF16_TO_F32(pg, &a[i + 3 * vl]);
        svfloat32_t vb0 = svld1(pg, &b[i]);
        svfloat32_t vb1 = svld1(pg, &b[i + vl]);
        svfloat32_t vb2 = svld1(pg, &b[i + 2 * vl]);
        svfloat32_t vb3 = svld1(pg, &b[i + 3 * vl]);
        acc0 = svmla_x(pg, acc0, va0, vb0);
        acc1 = svmla_x(pg, acc1, va1, vb1);
        acc2 = svmla_x(pg, acc2, va2, vb2);
        acc3 = svmla_x(pg, acc3, va3, vb3);
    }
    for (; i < n; i += vl) {
        svbool_t ptail = svwhilelt_b32(i, n);
        acc0 = svmla_m(ptail, acc0, SVE_BF16_TO_F32(ptail, &a[i]), svld1(ptail, &b[i]));
    }
    return svaddv(pg, svadd_x(pg, svadd_x(pg, acc0, acc1), svadd_x(pg, acc2, acc3)));
}

/* 4-row BF16 matvec: compute 4 dot products with shared activation vector.
 * Uses SVE prefetch to hide L2 latency for weight streams. */
static inline void matvec_bf16_4row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                     const uint16_t *w2, const uint16_t *w3,
                                     const float *x, int n) {
    int i = 0;
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    int vl = (int)svcntw();
    svbool_t pg = svptrue_b32();

    for (; i + vl - 1 < n; i += vl) {
        /* Prefetch 4 weight rows and activation 2 iterations ahead */
        /* No SW prefetch: A64FX HW stream prefetcher handles sequential patterns */
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, SVE_BF16_TO_F32(pg, &w0[i]), vx);
        a1 = svmla_x(pg, a1, SVE_BF16_TO_F32(pg, &w1[i]), vx);
        a2 = svmla_x(pg, a2, SVE_BF16_TO_F32(pg, &w2[i]), vx);
        a3 = svmla_x(pg, a3, SVE_BF16_TO_F32(pg, &w3[i]), vx);
    }
    if (i < n) {
        svbool_t ptail = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1(ptail, &x[i]);
        a0 = svmla_m(ptail, a0, SVE_BF16_TO_F32(ptail, &w0[i]), vx);
        a1 = svmla_m(ptail, a1, SVE_BF16_TO_F32(ptail, &w1[i]), vx);
        a2 = svmla_m(ptail, a2, SVE_BF16_TO_F32(ptail, &w2[i]), vx);
        a3 = svmla_m(ptail, a3, SVE_BF16_TO_F32(ptail, &w3[i]), vx);
    }
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
}

/* 8-row BF16 matvec: compute 8 dot products with shared activation vector.
 * Reads activation once from L1, amortizes across 8 weight streams from L2/HBM2.
 * This doubles the compute-to-load ratio vs 4-row: 8 FMAs per activation load. */
static inline void matvec_bf16_8row(float *dst,
                                     const uint16_t *w0, const uint16_t *w1,
                                     const uint16_t *w2, const uint16_t *w3,
                                     const uint16_t *w4, const uint16_t *w5,
                                     const uint16_t *w6, const uint16_t *w7,
                                     const float *x, int n) {
    int i = 0;
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svfloat32_t a4 = svdup_f32(0.0f), a5 = svdup_f32(0.0f);
    svfloat32_t a6 = svdup_f32(0.0f), a7 = svdup_f32(0.0f);
    int vl = (int)svcntw();
    svbool_t pg = svptrue_b32();

    for (; i + vl - 1 < n; i += vl) {
        /* No SW prefetch: A64FX HW stream prefetcher handles sequential patterns */
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, SVE_BF16_TO_F32(pg, &w0[i]), vx);
        a1 = svmla_x(pg, a1, SVE_BF16_TO_F32(pg, &w1[i]), vx);
        a2 = svmla_x(pg, a2, SVE_BF16_TO_F32(pg, &w2[i]), vx);
        a3 = svmla_x(pg, a3, SVE_BF16_TO_F32(pg, &w3[i]), vx);
        a4 = svmla_x(pg, a4, SVE_BF16_TO_F32(pg, &w4[i]), vx);
        a5 = svmla_x(pg, a5, SVE_BF16_TO_F32(pg, &w5[i]), vx);
        a6 = svmla_x(pg, a6, SVE_BF16_TO_F32(pg, &w6[i]), vx);
        a7 = svmla_x(pg, a7, SVE_BF16_TO_F32(pg, &w7[i]), vx);
    }
    if (i < n) {
        svbool_t ptail = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1(ptail, &x[i]);
        a0 = svmla_m(ptail, a0, SVE_BF16_TO_F32(ptail, &w0[i]), vx);
        a1 = svmla_m(ptail, a1, SVE_BF16_TO_F32(ptail, &w1[i]), vx);
        a2 = svmla_m(ptail, a2, SVE_BF16_TO_F32(ptail, &w2[i]), vx);
        a3 = svmla_m(ptail, a3, SVE_BF16_TO_F32(ptail, &w3[i]), vx);
        a4 = svmla_m(ptail, a4, SVE_BF16_TO_F32(ptail, &w4[i]), vx);
        a5 = svmla_m(ptail, a5, SVE_BF16_TO_F32(ptail, &w5[i]), vx);
        a6 = svmla_m(ptail, a6, SVE_BF16_TO_F32(ptail, &w6[i]), vx);
        a7 = svmla_m(ptail, a7, SVE_BF16_TO_F32(ptail, &w7[i]), vx);
    }
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* 8-row BF16 matvec, p_odd predicated-load variant.
 * Inputs are 4 pair buffers, each holding 2 adjacent rows interleaved at
 * bf16 granularity per 16-element chunk. Two ld1h.h with p_odd at offsets
 * -2 / 0 bytes extract both rows as FP32 (bf16 lands in upper half of .s
 * lanes), eliminating the LSL the lsl variant needs.
 *
 * Pair buf layout (per 16-element chunk = 64 bytes = 32 halfwords):
 *   HW 0  = rA[c*16+0]   HW 1  = rB[c*16+0]
 *   HW 2  = rA[c*16+1]   HW 3  = rB[c*16+1]
 *   ... up to HW 30/31
 *
 * Constraints: n_cols % vl (= 16 on A64FX) == 0, n_rows % 8 == 0.
 * Caller must have packed weights via pack_bf16_rows_to_pv first. */
static inline void matvec_bf16_8row_pv(float *dst,
                                        const uint16_t *pAB, const uint16_t *pCD,
                                        const uint16_t *pEF, const uint16_t *pGH,
                                        const float *x, int n) {
    svbool_t pg = svptrue_b32();
    svbool_t p_all_h = svptrue_b16();
    svuint16_t idx_h = svindex_u16(0, 1);
    svbool_t p_odd = svcmpne_n_u16(p_all_h,
                                    svand_n_u16_x(p_all_h, idx_h, 1), 0);

    int vl = (int)svcntw();
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svfloat32_t a4 = svdup_f32(0.0f), a5 = svdup_f32(0.0f);
    svfloat32_t a6 = svdup_f32(0.0f), a7 = svdup_f32(0.0f);
    int i = 0;
    /* Lighter SW prefetch: distance 8 iters = 2 × 256 B = 2 cachelines per
     * pair stream, to L2 (locality=2). With ~12 threads/CMG × 4 pair streams
     * the HW prefetcher's ~16 slots/CMG are oversubscribed; an L2-only hint
     * provides headroom without blowing the L1. Guarded by TF_BF16PV_PREFETCH=1. */
    static int pf_env_done = 0, pf_on = 0;
    if (__builtin_expect(!pf_env_done, 0)) {
        const char *e = getenv("TF_BF16PV_PREFETCH");
        pf_on = (e && *e && *e != '0');
        pf_env_done = 1;
    }
    const int PFD_HW = 8 * 32;  /* halfwords = 2 × 256-B cachelines */
    for (; i + vl - 1 < n; i += vl) {
        /* pair[hw_base = 2*i] points at chunk c = i/vl */
        const uint16_t *ab = pAB + 2 * i;
        const uint16_t *cd = pCD + 2 * i;
        const uint16_t *ef = pEF + 2 * i;
        const uint16_t *gh = pGH + 2 * i;
        if (pf_on) {
            __builtin_prefetch(ab + PFD_HW, 0, 2);
            __builtin_prefetch(cd + PFD_HW, 0, 2);
            __builtin_prefetch(ef + PFD_HW, 0, 2);
            __builtin_prefetch(gh + PFD_HW, 0, 2);
        }
        svuint16_t vA = svld1_u16(p_odd, ab - 1);
        svuint16_t vB = svld1_u16(p_odd, ab);
        svuint16_t vC = svld1_u16(p_odd, cd - 1);
        svuint16_t vD = svld1_u16(p_odd, cd);
        svuint16_t vE = svld1_u16(p_odd, ef - 1);
        svuint16_t vF = svld1_u16(p_odd, ef);
        svuint16_t vG = svld1_u16(p_odd, gh - 1);
        svuint16_t vH = svld1_u16(p_odd, gh);
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, svreinterpret_f32(vA), vx);
        a1 = svmla_x(pg, a1, svreinterpret_f32(vB), vx);
        a2 = svmla_x(pg, a2, svreinterpret_f32(vC), vx);
        a3 = svmla_x(pg, a3, svreinterpret_f32(vD), vx);
        a4 = svmla_x(pg, a4, svreinterpret_f32(vE), vx);
        a5 = svmla_x(pg, a5, svreinterpret_f32(vF), vx);
        a6 = svmla_x(pg, a6, svreinterpret_f32(vG), vx);
        a7 = svmla_x(pg, a7, svreinterpret_f32(vH), vx);
    }
    /* n is required to be a multiple of vl by the panel build constraint;
     * no tail handling needed in the kernel. */
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* Accumulating K-tile variant of matvec_bf16_8row_pv, for the batched (M>1)
 * prefill GEMM. ADDS the 8-row partial dot products over a K-segment [0,n) into
 * acc[0..7] (caller zeroes acc, then sweeps K tiles for one token). The pair
 * pointers must already be advanced to the tile's column base (pAB + 2*k0, etc.)
 * and x to x + k0. Same p_odd predicated-load layout/constraints (n % 16 == 0)
 * as matvec_bf16_8row_pv; no SW prefetch (the GEMM keeps the weight tile hot in
 * L1 across all M tokens, so HW streaming suffices). The per-tile svaddv reorders
 * the K reduction vs the single-shot matvec -> result is bit-SIMILAR, not
 * bit-identical (validated to ~1e-4 in ds4f_gemm_test.c). */
static inline void matvec_bf16_8row_pv_acc(float *acc,
                                            const uint16_t *pAB, const uint16_t *pCD,
                                            const uint16_t *pEF, const uint16_t *pGH,
                                            const float *x, int n) {
    svbool_t pg = svptrue_b32();
    svbool_t p_all_h = svptrue_b16();
    svuint16_t idx_h = svindex_u16(0, 1);
    svbool_t p_odd = svcmpne_n_u16(p_all_h, svand_n_u16_x(p_all_h, idx_h, 1), 0);
    int vl = (int)svcntw();
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svfloat32_t a4 = svdup_f32(0.0f), a5 = svdup_f32(0.0f);
    svfloat32_t a6 = svdup_f32(0.0f), a7 = svdup_f32(0.0f);
    int i = 0;
    for (; i + vl - 1 < n; i += vl) {
        const uint16_t *ab = pAB + 2 * i, *cd = pCD + 2 * i;
        const uint16_t *ef = pEF + 2 * i, *gh = pGH + 2 * i;
        svuint16_t vA = svld1_u16(p_odd, ab - 1), vB = svld1_u16(p_odd, ab);
        svuint16_t vC = svld1_u16(p_odd, cd - 1), vD = svld1_u16(p_odd, cd);
        svuint16_t vE = svld1_u16(p_odd, ef - 1), vF = svld1_u16(p_odd, ef);
        svuint16_t vG = svld1_u16(p_odd, gh - 1), vH = svld1_u16(p_odd, gh);
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, svreinterpret_f32(vA), vx);
        a1 = svmla_x(pg, a1, svreinterpret_f32(vB), vx);
        a2 = svmla_x(pg, a2, svreinterpret_f32(vC), vx);
        a3 = svmla_x(pg, a3, svreinterpret_f32(vD), vx);
        a4 = svmla_x(pg, a4, svreinterpret_f32(vE), vx);
        a5 = svmla_x(pg, a5, svreinterpret_f32(vF), vx);
        a6 = svmla_x(pg, a6, svreinterpret_f32(vG), vx);
        a7 = svmla_x(pg, a7, svreinterpret_f32(vH), vx);
    }
    acc[0] += svaddv(pg, a0); acc[1] += svaddv(pg, a1);
    acc[2] += svaddv(pg, a2); acc[3] += svaddv(pg, a3);
    acc[4] += svaddv(pg, a4); acc[5] += svaddv(pg, a5);
    acc[6] += svaddv(pg, a6); acc[7] += svaddv(pg, a7);
}

/* Register-blocked 8-row x 3-token accumulating pv GEMM microkernel.
 *
 * matvec_bf16_8row_pv_acc replayed per token loads the 8 weight-row vectors from
 * L1 ONCE PER TOKEN (8 loads + 1 x-load for only 8 FMAs) -> the FLA pipes starve on
 * L1 load ports, capping the GEMM near ~17% of peak. This variant loads each of the
 * 8 weight rows ONCE per K-step (same p_odd dual-load pv layout) and issues it
 * against 3 token columns: 24 FMAs per 11 loads (8 weight + 3 x). 24 live f32 lane-
 * accumulators + 3 x vectors + a transient weight reg fit in the 32 SVE z-registers
 * (the same 8x3 blocking the vlm micro_kernel_bf16B_8x3_pv.S uses). Row->acc mapping
 * (ab-1,ab,cd-1,cd,ef-1,ef,gh-1,gh -> rows 0..7) is byte-identical to the single-
 * token kernel, so results match it (bit-similar via the per-tile svaddv reorder).
 * acc0/acc1/acc2 are each [8]; caller zeroes them and sweeps K tiles. n % 16 == 0. */
static inline void matvec_bf16_8x3_pv_acc(float *acc0, float *acc1, float *acc2,
                                          const uint16_t *pAB, const uint16_t *pCD,
                                          const uint16_t *pEF, const uint16_t *pGH,
                                          const float *x0, const float *x1, const float *x2,
                                          int n) {
    svbool_t pg = svptrue_b32();
    svbool_t p_all_h = svptrue_b16();
    svuint16_t idx_h = svindex_u16(0, 1);
    svbool_t p_odd = svcmpne_n_u16(p_all_h, svand_n_u16_x(p_all_h, idx_h, 1), 0);
    int vl = (int)svcntw();
    svfloat32_t a00=svdup_f32(0),a10=svdup_f32(0),a20=svdup_f32(0),a30=svdup_f32(0);
    svfloat32_t a40=svdup_f32(0),a50=svdup_f32(0),a60=svdup_f32(0),a70=svdup_f32(0);
    svfloat32_t a01=svdup_f32(0),a11=svdup_f32(0),a21=svdup_f32(0),a31=svdup_f32(0);
    svfloat32_t a41=svdup_f32(0),a51=svdup_f32(0),a61=svdup_f32(0),a71=svdup_f32(0);
    svfloat32_t a02=svdup_f32(0),a12=svdup_f32(0),a22=svdup_f32(0),a32=svdup_f32(0);
    svfloat32_t a42=svdup_f32(0),a52=svdup_f32(0),a62=svdup_f32(0),a72=svdup_f32(0);
    int i = 0;
    for (; i + vl - 1 < n; i += vl) {
        const uint16_t *ab = pAB + 2*i, *cd = pCD + 2*i, *ef = pEF + 2*i, *gh = pGH + 2*i;
        svfloat32_t vx0 = svld1(pg, &x0[i]);
        svfloat32_t vx1 = svld1(pg, &x1[i]);
        svfloat32_t vx2 = svld1(pg, &x2[i]);
        svfloat32_t w;
        w = svreinterpret_f32(svld1_u16(p_odd, ab-1)); a00=svmla_x(pg,a00,w,vx0); a01=svmla_x(pg,a01,w,vx1); a02=svmla_x(pg,a02,w,vx2);
        w = svreinterpret_f32(svld1_u16(p_odd, ab  )); a10=svmla_x(pg,a10,w,vx0); a11=svmla_x(pg,a11,w,vx1); a12=svmla_x(pg,a12,w,vx2);
        w = svreinterpret_f32(svld1_u16(p_odd, cd-1)); a20=svmla_x(pg,a20,w,vx0); a21=svmla_x(pg,a21,w,vx1); a22=svmla_x(pg,a22,w,vx2);
        w = svreinterpret_f32(svld1_u16(p_odd, cd  )); a30=svmla_x(pg,a30,w,vx0); a31=svmla_x(pg,a31,w,vx1); a32=svmla_x(pg,a32,w,vx2);
        w = svreinterpret_f32(svld1_u16(p_odd, ef-1)); a40=svmla_x(pg,a40,w,vx0); a41=svmla_x(pg,a41,w,vx1); a42=svmla_x(pg,a42,w,vx2);
        w = svreinterpret_f32(svld1_u16(p_odd, ef  )); a50=svmla_x(pg,a50,w,vx0); a51=svmla_x(pg,a51,w,vx1); a52=svmla_x(pg,a52,w,vx2);
        w = svreinterpret_f32(svld1_u16(p_odd, gh-1)); a60=svmla_x(pg,a60,w,vx0); a61=svmla_x(pg,a61,w,vx1); a62=svmla_x(pg,a62,w,vx2);
        w = svreinterpret_f32(svld1_u16(p_odd, gh  )); a70=svmla_x(pg,a70,w,vx0); a71=svmla_x(pg,a71,w,vx1); a72=svmla_x(pg,a72,w,vx2);
    }
    acc0[0]+=svaddv(pg,a00); acc0[1]+=svaddv(pg,a10); acc0[2]+=svaddv(pg,a20); acc0[3]+=svaddv(pg,a30);
    acc0[4]+=svaddv(pg,a40); acc0[5]+=svaddv(pg,a50); acc0[6]+=svaddv(pg,a60); acc0[7]+=svaddv(pg,a70);
    acc1[0]+=svaddv(pg,a01); acc1[1]+=svaddv(pg,a11); acc1[2]+=svaddv(pg,a21); acc1[3]+=svaddv(pg,a31);
    acc1[4]+=svaddv(pg,a41); acc1[5]+=svaddv(pg,a51); acc1[6]+=svaddv(pg,a61); acc1[7]+=svaddv(pg,a71);
    acc2[0]+=svaddv(pg,a02); acc2[1]+=svaddv(pg,a12); acc2[2]+=svaddv(pg,a22); acc2[3]+=svaddv(pg,a32);
    acc2[4]+=svaddv(pg,a42); acc2[5]+=svaddv(pg,a52); acc2[6]+=svaddv(pg,a62); acc2[7]+=svaddv(pg,a72);
}

/* int8 svdot W8A8 8-row matvec for the q8_pv "group" layout.
 *
 * group layout (per 8 rows × K cols, K % 64 == 0):
 *   nb = K / 64 blocks, each 528 bytes:
 *     bytes [0..16)    -> 8 fp16 row-scales (row 0..7)
 *     bytes [16..528)  -> 8 rows × 64 int8 quants, row-major
 *
 * X is pre-quantized once per matvec into int8 + per-64-block fp16 scale
 * (see tf_quant_x_sdot). Per row per 64-block: one svdot_s32 (64 int8 MACs
 * -> 16 i32 lanes) + svcvt + svmla by the scalar (w_scale × x_scale) into a
 * float lane-accumulator. Block scales fold via the svmla, so there is only
 * ONE horizontal reduction (svaddv) per row, at the very end. Weight DRAM is
 * ~1.03 B/elem (528/512) vs bf16_8row_pv's 2.0 — and unlike the scalar-scale
 * Q8 kernel this stays HBM-BW-bound (svdot keeps the FLA pipe from saturating)
 * giving ~2× at the model's bandwidth regime. */
static inline void matvec_sdot_8row(float *dst,
                                     const uint8_t *group, /* points at block 0 */
                                     const int8_t *xq, const float *xscale,
                                     int K) {
    svbool_t pg = svptrue_b32();
    svbool_t pb = svptrue_b8();
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svfloat32_t a4 = svdup_f32(0.0f), a5 = svdup_f32(0.0f);
    svfloat32_t a6 = svdup_f32(0.0f), a7 = svdup_f32(0.0f);
    int nb = K / 64;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = group + (size_t)b * 528;
        const uint16_t *scl = (const uint16_t *)blk;
        const int8_t *qs    = (const int8_t *)(blk + 16);
        float xs = xscale[b];   /* fp32 activation scale (WS6: fp16 overflowed to Inf for amax>~8.3e6 -> 0*Inf=NaN) */
        svint8_t xv = svld1_s8(pb, xq + (size_t)b * 64);
        #define SDOT_ROW(R, ACC)                                              \
            do {                                                              \
                svint8_t wv = svld1_s8(pb, qs + (size_t)(R) * 64);            \
                svint32_t d = svdot_s32(svdup_s32(0), wv, xv);                \
                svfloat32_t df = svcvt_f32_s32_x(pg, d);                      \
                float sc = ggml_fp16_to_fp32(scl[R]) * xs;                    \
                ACC = svmla_x(pg, ACC, df, svdup_f32(sc));                    \
            } while (0)
        SDOT_ROW(0, a0); SDOT_ROW(1, a1); SDOT_ROW(2, a2); SDOT_ROW(3, a3);
        SDOT_ROW(4, a4); SDOT_ROW(5, a5); SDOT_ROW(6, a6); SDOT_ROW(7, a7);
        #undef SDOT_ROW
    }
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* 8-row x 3-token int8 svdot (prefill weight reuse): same q8_pv group layout as
 * matvec_sdot_8row, but each 8-row weight block is loaded ONCE and dotted against
 * 3 tokens' int8 (xq0/xq1/xq2 with per-64-block scales xs0/xs1/xs2), amortizing
 * the weight L1->reg loads across the triple. 24 float lane-accumulators (8 rows
 * x 3 tokens) — register-tight (fcc may spill 1-2, as the bf16 8x3 does), still a
 * net win when the loop is svdot-issue-bound. Per-(row,token) the K reduction is
 * the SAME order as the single-token kernel => results match matvec_sdot_8row. */
static inline void matvec_sdot_8row_3x(float *dst0, float *dst1, float *dst2,
                                       const uint8_t *group,
                                       const int8_t *xq0, const float *xs0,
                                       const int8_t *xq1, const float *xs1,
                                       const int8_t *xq2, const float *xs2,
                                       int K) {
    svbool_t pg = svptrue_b32();
    svbool_t pb = svptrue_b8();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f),
                a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    svfloat32_t b0=svdup_f32(0.f),b1=svdup_f32(0.f),b2=svdup_f32(0.f),b3=svdup_f32(0.f),
                b4=svdup_f32(0.f),b5=svdup_f32(0.f),b6=svdup_f32(0.f),b7=svdup_f32(0.f);
    svfloat32_t c0=svdup_f32(0.f),c1=svdup_f32(0.f),c2=svdup_f32(0.f),c3=svdup_f32(0.f),
                c4=svdup_f32(0.f),c5=svdup_f32(0.f),c6=svdup_f32(0.f),c7=svdup_f32(0.f);
    int nb = K / 64;
    for (int blk = 0; blk < nb; blk++) {
        const uint8_t *bp = group + (size_t)blk * 528;
        const uint16_t *scl = (const uint16_t *)bp;
        const int8_t *qs    = (const int8_t *)(bp + 16);
        float s0 = xs0[blk];   /* fp32 activation scale (WS6 overflow fix) */
        float s1 = xs1[blk];
        float s2 = xs2[blk];
        svint8_t xv0 = svld1_s8(pb, xq0 + (size_t)blk*64);
        svint8_t xv1 = svld1_s8(pb, xq1 + (size_t)blk*64);
        svint8_t xv2 = svld1_s8(pb, xq2 + (size_t)blk*64);
        #define SDOT3_ROW(R, AA, BB, CC)                                       \
            do {                                                               \
                svint8_t wv = svld1_s8(pb, qs + (size_t)(R)*64);               \
                float ws = ggml_fp16_to_fp32(scl[R]);                          \
                AA = svmla_x(pg, AA, svcvt_f32_s32_x(pg, svdot_s32(svdup_s32(0), wv, xv0)), svdup_f32(ws*s0)); \
                BB = svmla_x(pg, BB, svcvt_f32_s32_x(pg, svdot_s32(svdup_s32(0), wv, xv1)), svdup_f32(ws*s1)); \
                CC = svmla_x(pg, CC, svcvt_f32_s32_x(pg, svdot_s32(svdup_s32(0), wv, xv2)), svdup_f32(ws*s2)); \
            } while (0)
        SDOT3_ROW(0,a0,b0,c0); SDOT3_ROW(1,a1,b1,c1); SDOT3_ROW(2,a2,b2,c2); SDOT3_ROW(3,a3,b3,c3);
        SDOT3_ROW(4,a4,b4,c4); SDOT3_ROW(5,a5,b5,c5); SDOT3_ROW(6,a6,b6,c6); SDOT3_ROW(7,a7,b7,c7);
        #undef SDOT3_ROW
    }
    dst0[0]=svaddv(pg,a0);dst0[1]=svaddv(pg,a1);dst0[2]=svaddv(pg,a2);dst0[3]=svaddv(pg,a3);
    dst0[4]=svaddv(pg,a4);dst0[5]=svaddv(pg,a5);dst0[6]=svaddv(pg,a6);dst0[7]=svaddv(pg,a7);
    dst1[0]=svaddv(pg,b0);dst1[1]=svaddv(pg,b1);dst1[2]=svaddv(pg,b2);dst1[3]=svaddv(pg,b3);
    dst1[4]=svaddv(pg,b4);dst1[5]=svaddv(pg,b5);dst1[6]=svaddv(pg,b6);dst1[7]=svaddv(pg,b7);
    dst2[0]=svaddv(pg,c0);dst2[1]=svaddv(pg,c1);dst2[2]=svaddv(pg,c2);dst2[3]=svaddv(pg,c3);
    dst2[4]=svaddv(pg,c4);dst2[5]=svaddv(pg,c5);dst2[6]=svaddv(pg,c6);dst2[7]=svaddv(pg,c7);
}

/* ========================================================================
 * DeepSeek-V4-Flash on-demand dequant matvecs (split-layout FP8 + MXFP4).
 *
 * These live in the always-available (non-IMPLEMENTATION) region because they
 * are static-inline kernels emitted into every TU that includes this header
 * (ds4f.h / ds4f_runner.c). They keep weights quantized in HBM and dequant a
 * panel into registers per token, in the proven 8-accumulator + final-svaddv
 * shape. M=1 decode is HBM-BW-bound, so DRAM B/elem (FP8 ~1.0, MXFP4 ~0.53)
 * is what matters; the f32 unpack overlaps HBM latency across 48 threads.
 * ======================================================================== */

/* Standard OCP MX E8M0 scale: biased exponent, value = 2^(x - 127).
 * x==0xff is NaN (avoid in synthetic fill); x==0 maps to +0 here (OCP spec
 * value 2^-127, irrelevant for the harness). NOTE: distinct from the GGML
 * ggml_e8m0_to_fp32_half() which folds an extra ×0.5 — do NOT use that one. */
static inline float ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits = (uint32_t)x << 23;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* MXFP4 e2m1 code -> value, as f32 (svtbl_f32 table form of kvalues_mxfp4). */
static const float ds4f_kvalues_mxfp4_f32[16] = {
    0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f,
    0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f,
};

/* FP8 E4M3 (bias 7, exp==15 => NaN, no Inf) -> f32 bit pattern. Scalar form
 * lifted from a64fx/fp8-conv/fp8_convert.h, used to build the 256-entry LUT
 * the matvec gathers from (LUT gather ~0.70 cyc/elem, the validated winner). */
static inline uint32_t ds4f_fp8_e4m3_to_fp32_bits(uint8_t x) {
    uint8_t sign = (x >> 7) & 1;
    uint8_t exp  = (x >> 3) & 0xF;
    uint8_t mant = x & 0x7;
    if (exp == 0) {
        if (mant == 0) return (uint32_t)sign << 31;
        int shift = 0;
        while ((mant & 0x4) == 0) { mant <<= 1; shift++; }
        mant &= 0x3;
        exp = 127 - 7 - shift;
        return ((uint32_t)sign << 31) | ((uint32_t)exp << 23) | ((uint32_t)mant << 20);
    }
    if (exp == 15) /* NaN */
        return ((uint32_t)sign << 31) | (0xFFu << 23) | ((uint32_t)mant << 20);
    uint32_t new_exp = exp + (127 - 7);
    return ((uint32_t)sign << 31) | (new_exp << 23) | ((uint32_t)mant << 20);
}

/* Fill a caller-provided 256-entry LUT (call once at model init). */
static inline void ds4f_init_fp8_e4m3_lut(uint32_t *lut) {
    for (int i = 0; i < 256; i++) lut[i] = ds4f_fp8_e4m3_to_fp32_bits((uint8_t)i);
}

#if defined(__ARM_FEATURE_SVE)
/* FP8 E4M3 dense matvec, 8 rows, on-demand dequant via L1-resident LUT gather.
 * Weight = row-major FP8 bytes (1 B/elem); scale = per 128×128 block E8M0.
 * The 8 rows are one row-block (caller keeps groups 8-aligned, and 128 is a
 * multiple of 8 so a group never straddles a 128-row boundary), so for each
 * 128-col tile a single E8M0 scalar applies to all 8 rows -> fold it into x
 * once and reuse across the 8 gather+FMA streams. K must be a multiple of 128.
 *   escale[cb] = E8M0 scale for (this row-block, col-block cb), length K/128. */
static inline void matvec_fp8e4m3_8row(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const uint32_t *lut,
        const float *x, int K) {
    svbool_t pg = svptrue_b32();
    svfloat32_t a0 = svdup_f32(0.f), a1 = svdup_f32(0.f);
    svfloat32_t a2 = svdup_f32(0.f), a3 = svdup_f32(0.f);
    svfloat32_t a4 = svdup_f32(0.f), a5 = svdup_f32(0.f);
    svfloat32_t a6 = svdup_f32(0.f), a7 = svdup_f32(0.f);
    int vl = (int)svcntw();
    for (int c0 = 0; c0 < K; c0 += 128) {
        float s = ggml_e8m0_to_fp32(escale[c0 >> 7]);
        svfloat32_t vs = svdup_f32(s);
        for (int c = c0; c < c0 + 128; c += vl) {
            svfloat32_t vxs = svmul_x(pg, svld1(pg, &x[c]), vs);
            #define FP8_ROW(W, ACC) do {                                       \
                svuint32_t idx  = svld1ub_u32(pg, (W) + c);                     \
                svuint32_t bits = svld1_gather_u32index_u32(pg, lut, idx);      \
                ACC = svmla_x(pg, ACC, svreinterpret_f32_u32(bits), vxs);       \
            } while (0)
            FP8_ROW(w0, a0); FP8_ROW(w1, a1); FP8_ROW(w2, a2); FP8_ROW(w3, a3);
            FP8_ROW(w4, a4); FP8_ROW(w5, a5); FP8_ROW(w6, a6); FP8_ROW(w7, a7);
            #undef FP8_ROW
        }
    }
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* Enable flush-to-zero (FPCR.FZ, bit 24) on the CURRENT thread. The magic FP8
 * decode below builds a transient f32 subnormal (the multiply input) for E4M3
 * subnormals; A64FX penalizes denormal operands by ~3.5x, so FTZ (flush those
 * to 0) is REQUIRED for the magic path to win. Acceptable for the synthetic
 * harness (values meaningless; E4M3 subnormals -> 0). Idempotent + per-thread. */
static inline void ds4f_set_ftz(void) {
    uint64_t fpcr; __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ull << 24); __asm__ __volatile__("msr fpcr, %0" :: "r"(fpcr));
}

/* FP8 E4M3 -> f32 via the DENORMAL MAGIC-MULTIPLY (no LUT, no gather, no select):
 * build a tiny f32 from the 7 low bits, one fmul by 2^120 renormalizes BOTH
 * normals and subnormals in hardware, then re-apply sign. ~6 ops/lane. REQUIRES
 * FTZ (ds4f_set_ftz) on every calling thread or the subnormal multiply input
 * triggers A64FX denormal microcode. exp==15 maps to a finite <=480 (not NaN),
 * which is harmless / desirable for the harness. */
static inline svfloat32_t ds4f_fp8_decode_magic_u32(svbool_t pg, svuint32_t b) {
    svuint32_t sign = svlsl_n_u32_x(pg, svand_n_u32_x(pg, b, 0x80u), 24);
    svuint32_t mag  = svlsl_n_u32_x(pg, svand_n_u32_x(pg, b, 0x7Fu), 20);
    svfloat32_t f   = svmul_n_f32_x(pg, svreinterpret_f32_u32(mag), 0x1.0p+120f);
    return svreinterpret_f32_u32(svorr_u32_x(pg, sign, svreinterpret_u32_f32(f)));
}

/* FP8 E4M3 dense matvec, 8 rows, magic-decode variant. Same args/contract as
 * matvec_fp8e4m3_8row EXCEPT no LUT (decode is arithmetic). Packed 64-byte
 * svld1_u8 load + svunpk to four u32 sub-vectors, magic-decode each, FMA against
 * the per-128-block-scaled x. Faster than the gather variant in the HBM-stream
 * (M=1 decode) regime: holds 75-140 Gmac/s vs the gather's 73-118 (+2..18%) and
 * ~1.7-3x over bf16-predequant. Callers MUST ds4f_set_ftz() on each thread. */
static inline void matvec_fp8e4m3_8row_magic(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const float *x, int K) {
    svbool_t pg = svptrue_b32();
    svbool_t pb = svptrue_b8();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    int vlb = (int)svcntb();   /* 64 bytes / iter on A64FX */
    for (int c0 = 0; c0 < K; c0 += 128) {
        float s = ggml_e8m0_to_fp32(escale[c0 >> 7]);
        svfloat32_t vs = svdup_f32(s);
        for (int c = c0; c < c0 + 128; c += vlb) {
            svfloat32_t vx0 = svmul_x(pg, svld1(pg,&x[c]),    vs);
            svfloat32_t vx1 = svmul_x(pg, svld1(pg,&x[c+16]), vs);
            svfloat32_t vx2 = svmul_x(pg, svld1(pg,&x[c+32]), vs);
            svfloat32_t vx3 = svmul_x(pg, svld1(pg,&x[c+48]), vs);
            #define FP8_MAGIC_ROW(W, ACC) do {                                  \
                svuint8_t  b   = svld1_u8(pb, (W) + c);                          \
                svuint16_t l16 = svunpklo_u16(b), h16 = svunpkhi_u16(b);         \
                svfloat32_t f0 = ds4f_fp8_decode_magic_u32(pg, svunpklo_u32(l16)); \
                svfloat32_t f1 = ds4f_fp8_decode_magic_u32(pg, svunpkhi_u32(l16)); \
                svfloat32_t f2 = ds4f_fp8_decode_magic_u32(pg, svunpklo_u32(h16)); \
                svfloat32_t f3 = ds4f_fp8_decode_magic_u32(pg, svunpkhi_u32(h16)); \
                ACC = svmla_x(pg, ACC, f0, vx0);                                 \
                ACC = svmla_x(pg, ACC, f1, vx1);                                 \
                ACC = svmla_x(pg, ACC, f2, vx2);                                 \
                ACC = svmla_x(pg, ACC, f3, vx3);                                 \
            } while (0)
            FP8_MAGIC_ROW(w0,a0); FP8_MAGIC_ROW(w1,a1); FP8_MAGIC_ROW(w2,a2); FP8_MAGIC_ROW(w3,a3);
            FP8_MAGIC_ROW(w4,a4); FP8_MAGIC_ROW(w5,a5); FP8_MAGIC_ROW(w6,a6); FP8_MAGIC_ROW(w7,a7);
            #undef FP8_MAGIC_ROW
        }
    }
    dst[0]=svaddv(pg,a0); dst[1]=svaddv(pg,a1); dst[2]=svaddv(pg,a2); dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4); dst[5]=svaddv(pg,a5); dst[6]=svaddv(pg,a6); dst[7]=svaddv(pg,a7);
}

/* Split-layout MXFP4 (e2m1) expert matvec, 8 rows, W4A16 f32 (svtbl unpack).
 * Weight = row-major packed nibbles (K/2 B/row); scale = per-32-block E8M0
 * (K/32 B/row, per-row, unlike FP8's shared block). DRAM ~0.53 B/elem.
 * Layout per 32-block of 16 bytes (matches dequantize_row_mxfp4): byte j low
 * nibble -> element j, high nibble -> element j+16. Unpack both halves to f32
 * via svtbl_f32 over the 16-entry kvalues table, accumulate the unscaled block
 * dot, fold the per-row E8M0 scalar via one svmla. K must be a multiple of 32.
 *   wr[r]: K/2 nibble bytes;  sr[r]: K/32 E8M0 bytes. */
static inline void matvec_mxfp4_8row(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const uint8_t *s4, const uint8_t *s5, const uint8_t *s6, const uint8_t *s7,
        const float *x, int K) {
    svbool_t pg = svptrue_b32();
    svfloat32_t kv = svld1(pg, ds4f_kvalues_mxfp4_f32);
    svfloat32_t a0 = svdup_f32(0.f), a1 = svdup_f32(0.f);
    svfloat32_t a2 = svdup_f32(0.f), a3 = svdup_f32(0.f);
    svfloat32_t a4 = svdup_f32(0.f), a5 = svdup_f32(0.f);
    svfloat32_t a6 = svdup_f32(0.f), a7 = svdup_f32(0.f);
    int nb = K / 32;
    for (int b = 0; b < nb; b++) {
        svfloat32_t vxlo = svld1(pg, &x[b * 32]);
        svfloat32_t vxhi = svld1(pg, &x[b * 32 + 16]);
        #define MXFP4_ROW(W, S, ACC) do {                                      \
            svuint32_t braw = svld1ub_u32(pg, (W) + (size_t)b * 16);            \
            svuint32_t lo = svand_n_u32_x(pg, braw, 0xf);                       \
            svuint32_t hi = svand_n_u32_x(pg, svlsr_n_u32_x(pg, braw, 4), 0xf); \
            svfloat32_t wlo = svtbl_f32(kv, lo);                               \
            svfloat32_t whi = svtbl_f32(kv, hi);                               \
            svfloat32_t p = svmul_x(pg, wlo, vxlo);                            \
            p = svmla_x(pg, p, whi, vxhi);                                     \
            float sc = ggml_e8m0_to_fp32((S)[b]);                             \
            ACC = svmla_x(pg, ACC, p, svdup_f32(sc));                          \
        } while (0)
        MXFP4_ROW(w0, s0, a0); MXFP4_ROW(w1, s1, a1);
        MXFP4_ROW(w2, s2, a2); MXFP4_ROW(w3, s3, a3);
        MXFP4_ROW(w4, s4, a4); MXFP4_ROW(w5, s5, a5);
        MXFP4_ROW(w6, s6, a6); MXFP4_ROW(w7, s7, a7);
        #undef MXFP4_ROW
    }
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* Two-token register-blocked MXFP4 matvec for batched (M>1) expert GEMM.
 * Same per-row svtbl dequant as matvec_mxfp4_8row, but each block's dequantized
 * weight (wlo/whi) is reused across TWO x-vectors before being discarded, so the
 * weight nibbles read from L1 once feed both tokens -> when the caller keeps an
 * 8-row group L1-resident across its token pairs, the weight is read from HBM
 * once and amortized over all M tokens. Each token's accumulation order (over
 * blocks) is identical to matvec_mxfp4_8row, so dst0/dst1 are BIT-IDENTICAL to
 * two separate single-token calls (no cross-token reassociation here). */
static inline void matvec_mxfp4_8row_2x(float *dst0, float *dst1,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const uint8_t *s4, const uint8_t *s5, const uint8_t *s6, const uint8_t *s7,
        const float *x0, const float *x1, int K) {
    svbool_t pg = svptrue_b32();
    svfloat32_t kv = svld1(pg, ds4f_kvalues_mxfp4_f32);
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    svfloat32_t b0=svdup_f32(0.f),b1=svdup_f32(0.f),b2=svdup_f32(0.f),b3=svdup_f32(0.f);
    svfloat32_t b4=svdup_f32(0.f),b5=svdup_f32(0.f),b6=svdup_f32(0.f),b7=svdup_f32(0.f);
    int nb = K / 32;
    for (int b = 0; b < nb; b++) {
        svfloat32_t vxlo0 = svld1(pg, &x0[b*32]), vxhi0 = svld1(pg, &x0[b*32+16]);
        svfloat32_t vxlo1 = svld1(pg, &x1[b*32]), vxhi1 = svld1(pg, &x1[b*32+16]);
        #define MXFP4_ROW2(W, S, ACCA, ACCB) do {                              \
            svuint32_t braw = svld1ub_u32(pg, (W) + (size_t)b * 16);           \
            svuint32_t lo = svand_n_u32_x(pg, braw, 0xf);                      \
            svuint32_t hi = svand_n_u32_x(pg, svlsr_n_u32_x(pg, braw, 4), 0xf);\
            svfloat32_t wlo = svtbl_f32(kv, lo);                              \
            svfloat32_t whi = svtbl_f32(kv, hi);                              \
            svfloat32_t vsc = svdup_f32(ggml_e8m0_to_fp32((S)[b]));           \
            svfloat32_t pa = svmul_x(pg, wlo, vxlo0);                         \
            pa = svmla_x(pg, pa, whi, vxhi0);                                 \
            ACCA = svmla_x(pg, ACCA, pa, vsc);                                \
            svfloat32_t pbv = svmul_x(pg, wlo, vxlo1);                        \
            pbv = svmla_x(pg, pbv, whi, vxhi1);                               \
            ACCB = svmla_x(pg, ACCB, pbv, vsc);                               \
        } while (0)
        MXFP4_ROW2(w0,s0,a0,b0); MXFP4_ROW2(w1,s1,a1,b1);
        MXFP4_ROW2(w2,s2,a2,b2); MXFP4_ROW2(w3,s3,a3,b3);
        MXFP4_ROW2(w4,s4,a4,b4); MXFP4_ROW2(w5,s5,a5,b5);
        MXFP4_ROW2(w6,s6,a6,b6); MXFP4_ROW2(w7,s7,a7,b7);
        #undef MXFP4_ROW2
    }
    dst0[0]=svaddv(pg,a0); dst0[1]=svaddv(pg,a1); dst0[2]=svaddv(pg,a2); dst0[3]=svaddv(pg,a3);
    dst0[4]=svaddv(pg,a4); dst0[5]=svaddv(pg,a5); dst0[6]=svaddv(pg,a6); dst0[7]=svaddv(pg,a7);
    dst1[0]=svaddv(pg,b0); dst1[1]=svaddv(pg,b1); dst1[2]=svaddv(pg,b2); dst1[3]=svaddv(pg,b3);
    dst1[4]=svaddv(pg,b4); dst1[5]=svaddv(pg,b5); dst1[6]=svaddv(pg,b6); dst1[7]=svaddv(pg,b7);
}
#endif /* __ARM_FEATURE_SVE */

/* Tiled BF16 GEMM: Y[tok, row] = W[row, K] · X[tok, K]^T
 * Token-major output: Y[t * Y_stride + r].
 * Processes 4 rows × N tokens with column-tiled inner loop for L1 reuse. */
static inline void gemm_bf16_f32_tokmajor(float *Y, const uint16_t *W, const float *X,
                                           int n_rows, int K, int N, int Y_stride, int X_stride) {
    int r;
    /* 4-row blocks: amortize weight loads across multiple tokens */
    for (r = 0; r + 3 < n_rows; r += 4) {
        const uint16_t *w0 = W + (size_t)(r)     * K;
        const uint16_t *w1 = W + (size_t)(r + 1) * K;
        const uint16_t *w2 = W + (size_t)(r + 2) * K;
        const uint16_t *w3 = W + (size_t)(r + 3) * K;

        if (N == 1) {
            /* Single token: direct 4-row matvec */
            matvec_bf16_4row(&Y[r], w0, w1, w2, w3, X, K);
        } else {
            /* Multi-token: tile over K to reuse weight cache lines across tokens.
             * Tile size chosen to fit 4 weight rows in L1D (~256KB per tile). */
            int TILE_K = 512;  /* 4 rows × 512 × 2B = 4KB per row, fits L1 */
            if (TILE_K > K) TILE_K = K;

            /* Initialize output */
            for (int t = 0; t < N; t++) {
                Y[t * Y_stride + r]     = 0.0f;
                Y[t * Y_stride + r + 1] = 0.0f;
                Y[t * Y_stride + r + 2] = 0.0f;
                Y[t * Y_stride + r + 3] = 0.0f;
            }

            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int klen = K - k0 < TILE_K ? K - k0 : TILE_K;
                const uint16_t *tw0 = w0 + k0, *tw1 = w1 + k0;
                const uint16_t *tw2 = w2 + k0, *tw3 = w3 + k0;
                for (int t = 0; t < N; t++) {
                    float tmp[4];
                    matvec_bf16_4row(tmp, tw0, tw1, tw2, tw3,
                                     X + (size_t)t * X_stride + k0, klen);
                    Y[t * Y_stride + r]     += tmp[0];
                    Y[t * Y_stride + r + 1] += tmp[1];
                    Y[t * Y_stride + r + 2] += tmp[2];
                    Y[t * Y_stride + r + 3] += tmp[3];
                }
            }
        }
    }
    /* Remainder rows */
    for (; r < n_rows; r++) {
        const uint16_t *wr = W + (size_t)r * K;
        for (int t = 0; t < N; t++)
            Y[t * Y_stride + r] = vec_dot_bf16_f32(wr, X + (size_t)t * X_stride, K);
    }
}

#undef SVE_BF16_TO_F32

#elif defined(__F16C__) && defined(__AVX2__) && defined(__FMA__)

/* AVX2 BF16 dot product: load u16, zero-extend to u32, shift left 16, reinterpret as f32 */
static inline float vec_dot_bf16_f32(const uint16_t *a, const float *b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        /* Load 8 BF16 at a time, convert to F32 via shift */
        __m128i raw0 = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i raw1 = _mm_loadu_si128((const __m128i *)(a + i + 8));
        __m256i wide0 = _mm256_cvtepu16_epi32(raw0);
        __m256i wide1 = _mm256_cvtepu16_epi32(raw1);
        __m256 va0 = _mm256_castsi256_ps(_mm256_slli_epi32(wide0, 16));
        __m256 va1 = _mm256_castsi256_ps(_mm256_slli_epi32(wide1, 16));
        acc0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 7 < n; i += 8) {
        __m128i raw = _mm_loadu_si128((const __m128i *)(a + i));
        __m256i wide = _mm256_cvtepu16_epi32(raw);
        __m256 va = _mm256_castsi256_ps(_mm256_slli_epi32(wide, 16));
        acc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b + i), acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    float result = _mm_cvtss_f32(s);
    for (; i < n; i++) result += bf16_to_f32_scalar(a[i]) * b[i];
    return result;
}

static inline void matvec_bf16_4row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                     const uint16_t *w2, const uint16_t *w3,
                                     const float *x, int n) {
    dst[0] = vec_dot_bf16_f32(w0, x, n); dst[1] = vec_dot_bf16_f32(w1, x, n);
    dst[2] = vec_dot_bf16_f32(w2, x, n); dst[3] = vec_dot_bf16_f32(w3, x, n);
}

static inline void gemm_bf16_f32_tokmajor(float *Y, const uint16_t *W, const float *X,
                                           int n_rows, int K, int N, int Y_stride, int X_stride) {
    for (int r = 0; r < n_rows; r++) {
        const uint16_t *wr = W + (size_t)r * K;
        for (int t = 0; t < N; t++)
            Y[t * Y_stride + r] = vec_dot_bf16_f32(wr, X + (size_t)t * X_stride, K);
    }
}

#else
/* Scalar BF16 fallback */
static inline float vec_dot_bf16_f32(const uint16_t *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += bf16_to_f32_scalar(a[i]) * b[i];
    return sum;
}
static inline void matvec_bf16_4row(float *dst, const uint16_t *w0, const uint16_t *w1,
                                     const uint16_t *w2, const uint16_t *w3,
                                     const float *x, int n) {
    dst[0] = vec_dot_bf16_f32(w0, x, n); dst[1] = vec_dot_bf16_f32(w1, x, n);
    dst[2] = vec_dot_bf16_f32(w2, x, n); dst[3] = vec_dot_bf16_f32(w3, x, n);
}
static inline void gemm_bf16_f32_tokmajor(float *Y, const uint16_t *W, const float *X,
                                           int n_rows, int K, int N, int Y_stride, int X_stride) {
    for (int r = 0; r < n_rows; r++) {
        const uint16_t *wr = W + (size_t)r * K;
        for (int t = 0; t < N; t++)
            Y[t * Y_stride + r] = vec_dot_bf16_f32(wr, X + (size_t)t * X_stride, K);
    }
}
#endif /* BF16 kernels */

/* ======================================================================== */
#ifdef GGML_DEQUANT_IMPLEMENTATION

#include "ggml_dequant_impl.inc"

#endif /* GGML_DEQUANT_IMPLEMENTATION */
#endif /* GGML_DEQUANT_H */
