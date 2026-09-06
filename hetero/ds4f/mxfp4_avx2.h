/* mxfp4_avx2.h — AVX2 MXFP4 (e2m1) expert matvec kernels for DeepSeek-V4-Flash.
 *
 * x86 counterpart of the SVE matvec_mxfp4_8row in common/ggml_dequant.h
 * (:1966). Same weight layout, same accumulation order, so results match the
 * A64FX path to within f32 reassociation of the horizontal reduction only.
 *
 * Layout (identical to the SVE kernel):
 *   w[r] : row-major packed nibbles, K/2 bytes per row. Within each 32-element
 *          block of 16 bytes, byte j's LOW nibble is element j and its HIGH
 *          nibble is element j+16.
 *   s[r] : per-32-block E8M0 exponent byte, K/32 bytes per row.
 *
 * The e2m1 code table ds4f_kvalues_mxfp4_f32 is {0,1,2,3,4,6,8,12} and negatives
 * -- all small integers -- so the nibble->value map is a single _mm_shuffle_epi8
 * over an int8 table, then a widening convert to f32. No gather, no LUT loads.
 *
 * This header is standalone (no ggml_dequant.h dependency) so the S0 roofline
 * benchmark can build without pulling in the SVE-guarded model headers. It will
 * move into common/ggml_dequant.h as the __AVX2__ branch in S1.
 */
#ifndef DS4F_MXFP4_AVX2_H
#define DS4F_MXFP4_AVX2_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

/* E8M0 exponent byte -> f32 (x<<23 reinterpreted). Matches ggml_e8m0_to_fp32. */
static inline float ds4f_e8m0_to_f32(uint8_t x) {
    uint32_t bits = (uint32_t)x << 23;
    float r; memcpy(&r, &bits, sizeof(r)); return r;
}

/* Signed-int8 form of ds4f_kvalues_mxfp4_f32, duplicated into both 128-bit
 * lanes so one _mm256_shuffle_epi8 could serve 32 bytes. The 16-byte block
 * kernel below only needs the low lane. */
static const int8_t ds4f_mxfp4_i8_tbl[32] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

/* Unpack one 16-byte MXFP4 block into 32 f32 values, in SVE element order:
 * out[0..15] are the low nibbles of bytes 0..15, out[16..31] the high nibbles. */
static inline void ds4f_mxfp4_unpack_block(const uint8_t *w, __m256 out[4]) {
    const __m128i tbl  = _mm_loadu_si128((const __m128i *)ds4f_mxfp4_i8_tbl);
    const __m128i mask = _mm_set1_epi8(0x0f);
    __m128i raw = _mm_loadu_si128((const __m128i *)w);
    __m128i lo  = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, mask));
    __m128i hi  = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw, 4), mask));
    /* int8 -> int32 -> f32, 8 at a time */
    out[0] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
    out[1] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
    out[2] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
    out[3] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
}

static inline float ds4f_hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

/* MXFP4 matvec, R rows at a time (R rows share the activation load).
 * dst[r] = sum_k w[r][k] * x[k]. K must be a multiple of 32.
 *
 * AVX2 has 16 ymm registers; 4 activation vectors per block plus 4 unpacked
 * weight vectors leaves room for 4 row accumulators (single ymm each,
 * reduced at the end). R=4 is the register-clean shape here, unlike SVE's 8.
 */
static inline void ds4f_matvec_mxfp4_4row(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const float *x, int K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    const int nb = K / 32;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + (size_t)b * 32;
        __m256 x0 = _mm256_loadu_ps(xb);
        __m256 x1 = _mm256_loadu_ps(xb + 8);
        __m256 x2 = _mm256_loadu_ps(xb + 16);
        __m256 x3 = _mm256_loadu_ps(xb + 24);
        #define MXFP4_ROW(W, S, ACC) do {                                     \
            __m256 wv[4];                                                     \
            ds4f_mxfp4_unpack_block((W) + (size_t)b * 16, wv);                \
            __m256 p = _mm256_mul_ps(wv[0], x0);                              \
            p = _mm256_fmadd_ps(wv[1], x1, p);                                \
            p = _mm256_fmadd_ps(wv[2], x2, p);                                \
            p = _mm256_fmadd_ps(wv[3], x3, p);                                \
            ACC = _mm256_fmadd_ps(p, _mm256_set1_ps(ds4f_e8m0_to_f32((S)[b])), ACC); \
        } while (0)
        MXFP4_ROW(w0, s0, a0); MXFP4_ROW(w1, s1, a1);
        MXFP4_ROW(w2, s2, a2); MXFP4_ROW(w3, s3, a3);
        #undef MXFP4_ROW
    }
    dst[0] = ds4f_hsum256(a0); dst[1] = ds4f_hsum256(a1);
    dst[2] = ds4f_hsum256(a2); dst[3] = ds4f_hsum256(a3);
}

/* v2: same math, restructured for Zen1 instruction-level parallelism.
 *
 * v1's per-row inner sequence is a 5-deep serial FMA chain
 * (p=mul; p=fma; p=fma; p=fma; acc=fma), and at ~5-cycle FMA latency that
 * alone costs 25 cycles per block-row with only 4 chains (one per row) to
 * hide it. Here the E8M0 scale is folded into the four unpacked weight
 * vectors instead of the product, and each row keeps TWO accumulators, so
 * the dependency depth per block drops from 5 to 2 and there are 8
 * independent chains in flight. Costs 3 extra vmulps per block-row, which
 * Zen1 has the FP throughput to absorb.
 *
 * Bit-exactness note: this changes the summation order relative to v1 and to
 * the SVE kernel (partial sums are split across two accumulators), so results
 * differ in the last f32 ulps. Verified against the scalar reference below.
 */
static inline void ds4f_matvec_mxfp4_4row_v2(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const float *x, int K) {
    __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
    const int nb = K / 32;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + (size_t)b * 32;
        __m256 x0 = _mm256_loadu_ps(xb);
        __m256 x1 = _mm256_loadu_ps(xb + 8);
        __m256 x2 = _mm256_loadu_ps(xb + 16);
        __m256 x3 = _mm256_loadu_ps(xb + 24);
        #define MXFP4_ROW2(W, S, ACCA, ACCB) do {                             \
            __m256 wv[4];                                                     \
            ds4f_mxfp4_unpack_block((W) + (size_t)b * 16, wv);                \
            __m256 sc = _mm256_castsi256_ps(                                  \
                _mm256_set1_epi32((int)(uint32_t)(S)[b] << 23));              \
            ACCA = _mm256_fmadd_ps(_mm256_mul_ps(wv[0], sc), x0, ACCA);       \
            ACCB = _mm256_fmadd_ps(_mm256_mul_ps(wv[1], sc), x1, ACCB);       \
            ACCA = _mm256_fmadd_ps(_mm256_mul_ps(wv[2], sc), x2, ACCA);       \
            ACCB = _mm256_fmadd_ps(_mm256_mul_ps(wv[3], sc), x3, ACCB);       \
        } while (0)
        MXFP4_ROW2(w0, s0, a0, b0); MXFP4_ROW2(w1, s1, a1, b1);
        MXFP4_ROW2(w2, s2, a2, b2); MXFP4_ROW2(w3, s3, a3, b3);
        #undef MXFP4_ROW2
    }
    dst[0] = ds4f_hsum256(_mm256_add_ps(a0, b0));
    dst[1] = ds4f_hsum256(_mm256_add_ps(a1, b1));
    dst[2] = ds4f_hsum256(_mm256_add_ps(a2, b2));
    dst[3] = ds4f_hsum256(_mm256_add_ps(a3, b3));
}

/* 8-row wrapper, API-compatible with the SVE matvec_mxfp4_8row so the shared
 * ds4f_impl.h call sites need no change. Two 4-row passes; the activation is
 * L1-resident across both so the second pass costs no extra DRAM traffic. */
static inline void ds4f_matvec_mxfp4_8row(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const uint8_t *s4, const uint8_t *s5, const uint8_t *s6, const uint8_t *s7,
        const float *x, int K) {
#ifdef DS4F_MXFP4_V1
    ds4f_matvec_mxfp4_4row(dst,     w0, w1, w2, w3, s0, s1, s2, s3, x, K);
    ds4f_matvec_mxfp4_4row(dst + 4, w4, w5, w6, w7, s4, s5, s6, s7, x, K);
#else
    ds4f_matvec_mxfp4_4row_v2(dst,     w0, w1, w2, w3, s0, s1, s2, s3, x, K);
    ds4f_matvec_mxfp4_4row_v2(dst + 4, w4, w5, w6, w7, s4, s5, s6, s7, x, K);
#endif
}

/* ===================== W4A8 integer path =====================
 *
 * The f32 kernels above are throughput-bound, not bandwidth-bound: on Zen1
 * every 256-bit AVX2 op is split into two 128-bit uops, and unpacking 32
 * nibbles to f32 costs 4x vpmovsxbd + 4x vcvtdq2ps + the nibble shuffles --
 * ~19 instructions of overhead for 4 FMAs of useful work. Measured 23.8 GB/s
 * against a 48 GB/s read ceiling.
 *
 * This variant keeps the whole dot product in integers, so the nibbles feed
 * vpmaddubsw directly and no float conversion happens per element:
 *
 *   sum_k w_k x_k  =  e8m0 * xscale_b * ( sum_k wu_k xq_k  -  12 * sum_k xq_k )
 *
 * where wu = value + 12 is the UNSIGNED biased weight (range 0..24, since the
 * e2m1 code table is {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}), and xq is the
 * activation quantized to int8 per 32-element block -- the same block
 * granularity the weight's E8M0 scale already uses. The bias is removable
 * exactly because the -12*sum(xq) correction depends only on the block, not
 * the row, so it is precomputed once per matvec and folded in with one scalar
 * FMA per block.
 *
 * vpmaddubsw needs (u8, i8): wu is the unsigned operand, xq the signed one.
 * Overflow is safe: 24 * 127 * 2 = 6096, well inside int16.
 *
 * Everything is 128-bit wide, which is full rate on Zen1 (1 uop per
 * instruction instead of 2).
 *
 * ACCURACY: this quantizes the ACTIVATION to int8 per 32 values (~0.4%
 * relative per block), unlike the f32-activation kernels above. This is the
 * same trade ggml makes for every K-quant, and the DS4F A64FX path already
 * has an int8-activation dense mode (DS4F_Q8_PV). Weights stay exact MXFP4.
 */

/* wu = kvalues + 12, as u8. Index is the raw 4-bit code. */
static const uint8_t ds4f_mxfp4_u8_tbl[16] = {
    12, 13, 14, 15, 16, 18, 20, 24,     /*  0  1  2  3  4  6  8 12 */
    12, 11, 10,  9,  8,  6,  4,  0,     /* -0 -1 -2 -3 -4 -6 -8 -12 */
};

/* Per-32-block int8 quantization of the activation.
 *   xq   [K]      int8 values, same element order as the weight nibbles
 *   xs   [K/32]   f32 scale per block
 *   xcorr[K/32]   f32, 12 * sum(xq) per block -- the debias term
 */
static inline void ds4f_quant_act_i8(const float *x, int K,
                                     int8_t *xq, float *xs, float *xcorr) {
    for (int b = 0; b < K / 32; b++) {
        const float *xb = x + (size_t)b * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; j++) { float a = fabsf(xb[j]); if (a > amax) amax = a; }
        float s = amax / 127.0f;
        float inv = (amax > 0.f) ? 127.0f / amax : 0.f;
        int sum = 0;
        for (int j = 0; j < 32; j++) {
            int v = (int)lrintf(xb[j] * inv);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            xq[(size_t)b * 32 + j] = (int8_t)v;
            sum += v;
        }
        xs[b] = s;
        xcorr[b] = 12.0f * (float)sum;
    }
}

/* W4A8 MXFP4 matvec, 4 rows sharing the quantized activation. */
static inline void ds4f_matvec_mxfp4_4row_i8(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const int8_t *xq, const float *xs, const float *xcorr, int K) {
    const __m128i tbl  = _mm_loadu_si128((const __m128i *)ds4f_mxfp4_u8_tbl);
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    __m128 f0 = _mm_setzero_ps(), f1 = _mm_setzero_ps();
    __m128 f2 = _mm_setzero_ps(), f3 = _mm_setzero_ps();
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    const int nb = K / 32;
    for (int b = 0; b < nb; b++) {
        __m128i xlo = _mm_loadu_si128((const __m128i *)(xq + (size_t)b * 32));
        __m128i xhi = _mm_loadu_si128((const __m128i *)(xq + (size_t)b * 32 + 16));
        const float xsb = xs[b], xcb = xcorr[b];
        #define MXFP4_ROW_I8(W, S, FACC, CACC) do {                            \
            __m128i raw = _mm_loadu_si128((const __m128i *)((W) + (size_t)b * 16)); \
            __m128i wlo = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, mask));      \
            __m128i whi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw, 4), mask)); \
            __m128i p   = _mm_madd_epi16(_mm_maddubs_epi16(wlo, xlo), ones);    \
            p = _mm_add_epi32(p, _mm_madd_epi16(_mm_maddubs_epi16(whi, xhi), ones)); \
            float sc = ds4f_e8m0_to_f32((S)[b]) * xsb;                          \
            FACC = _mm_add_ps(FACC, _mm_mul_ps(_mm_cvtepi32_ps(p), _mm_set1_ps(sc))); \
            CACC += sc * xcb;                                                   \
        } while (0)
        MXFP4_ROW_I8(w0, s0, f0, c0); MXFP4_ROW_I8(w1, s1, f1, c1);
        MXFP4_ROW_I8(w2, s2, f2, c2); MXFP4_ROW_I8(w3, s3, f3, c3);
        #undef MXFP4_ROW_I8
    }
    #define HSUM128(V) ({ __m128 t = _mm_add_ps(V, _mm_movehl_ps(V, V)); \
                          t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)); _mm_cvtss_f32(t); })
    dst[0] = HSUM128(f0) - c0; dst[1] = HSUM128(f1) - c1;
    dst[2] = HSUM128(f2) - c2; dst[3] = HSUM128(f3) - c3;
    #undef HSUM128
}

static inline void ds4f_matvec_mxfp4_8row_i8(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const uint8_t *s4, const uint8_t *s5, const uint8_t *s6, const uint8_t *s7,
        const int8_t *xq, const float *xs, const float *xcorr, int K) {
    ds4f_matvec_mxfp4_4row_i8(dst,     w0, w1, w2, w3, s0, s1, s2, s3, xq, xs, xcorr, K);
    ds4f_matvec_mxfp4_4row_i8(dst + 4, w4, w5, w6, w7, s4, s5, s6, s7, xq, xs, xcorr, K);
}

/* ---- W4A8, 256-bit, two blocks per iteration ----
 *
 * A 256-bit vpshufb is two independent 128-bit lane shuffles, which is exactly
 * the shape we want: one ymm load covers TWO consecutive 32-element blocks
 * (32 weight bytes), and after vpmaddubsw + vpmaddwd the low lane holds block
 * 2p's four int32 partials and the high lane holds block 2p+1's. Same uop count
 * as the 128-bit version but half the instructions, which matters because Zen1
 * decodes 4 instructions but dispatches 6 uops per cycle.
 *
 * For this to work the activation must be stored so that one ymm load yields
 * "block 2p elements 0..15" then "block 2p+1 elements 0..15". That is the
 * permuted layout ds4f_quant_act_i8_pairs writes:
 *
 *   [ b(2p).lo16 | b(2p+1).lo16 | b(2p).hi16 | b(2p+1).hi16 ]  per 64 bytes
 *
 * K must be a multiple of 64 (two blocks). HIDDEN=4096 and MOE_INTER=2048 both are.
 */
static inline void ds4f_quant_act_i8_pairs(const float *x, int K,
                                           int8_t *xq, float *xs, float *xcorr) {
    for (int b = 0; b < K / 32; b++) {
        const float *xb = x + (size_t)b * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; j++) { float a = fabsf(xb[j]); if (a > amax) amax = a; }
        float inv = (amax > 0.f) ? 127.0f / amax : 0.f;
        /* destination of this block's low/high halves in the paired layout */
        int p = b >> 1, odd = b & 1;
        int8_t *dlo = xq + (size_t)p * 64 + (size_t)odd * 16;
        int8_t *dhi = dlo + 32;
        int sum = 0;
        for (int j = 0; j < 32; j++) {
            int v = (int)lrintf(xb[j] * inv);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            (j < 16 ? dlo : dhi)[j & 15] = (int8_t)v;
            sum += v;
        }
        xs[b] = amax / 127.0f;
        xcorr[b] = 12.0f * (float)sum;
    }
}

static inline void ds4f_matvec_mxfp4_4row_i8x2(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const int8_t *xq, const float *xs, const float *xcorr, int K) {
    const __m256i tbl = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)ds4f_mxfp4_u8_tbl));
    const __m256i mask = _mm256_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 f0 = _mm256_setzero_ps(), f1 = _mm256_setzero_ps();
    __m256 f2 = _mm256_setzero_ps(), f3 = _mm256_setzero_ps();
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    const int npair = K / 64;
    for (int p = 0; p < npair; p++) {
        __m256i xlo = _mm256_loadu_si256((const __m256i *)(xq + (size_t)p * 64));
        __m256i xhi = _mm256_loadu_si256((const __m256i *)(xq + (size_t)p * 64 + 32));
        const float xa = xs[2 * p], xb_ = xs[2 * p + 1];
        const float ca = xcorr[2 * p], cb = xcorr[2 * p + 1];
        #define MXFP4_ROW_I8X2(W, S, FACC, CACC) do {                          \
            __m256i raw = _mm256_loadu_si256((const __m256i *)((W) + (size_t)p * 32)); \
            __m256i wlo = _mm256_shuffle_epi8(tbl, _mm256_and_si256(raw, mask));\
            __m256i whi = _mm256_shuffle_epi8(tbl,                             \
                _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask));            \
            __m256i acc = _mm256_add_epi32(                                    \
                _mm256_madd_epi16(_mm256_maddubs_epi16(wlo, xlo), ones),       \
                _mm256_madd_epi16(_mm256_maddubs_epi16(whi, xhi), ones));      \
            float sa = ds4f_e8m0_to_f32((S)[2 * p])     * xa;                  \
            float sb = ds4f_e8m0_to_f32((S)[2 * p + 1]) * xb_;                 \
            __m256 sv = _mm256_insertf128_ps(_mm256_castps128_ps256(           \
                _mm_set1_ps(sa)), _mm_set1_ps(sb), 1);                         \
            FACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), sv, FACC);         \
            CACC += sa * ca + sb * cb;                                         \
        } while (0)
        MXFP4_ROW_I8X2(w0, s0, f0, c0); MXFP4_ROW_I8X2(w1, s1, f1, c1);
        MXFP4_ROW_I8X2(w2, s2, f2, c2); MXFP4_ROW_I8X2(w3, s3, f3, c3);
        #undef MXFP4_ROW_I8X2
    }
    dst[0] = ds4f_hsum256(f0) - c0; dst[1] = ds4f_hsum256(f1) - c1;
    dst[2] = ds4f_hsum256(f2) - c2; dst[3] = ds4f_hsum256(f3) - c3;
}

static inline void ds4f_matvec_mxfp4_8row_i8x2(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const uint8_t *s4, const uint8_t *s5, const uint8_t *s6, const uint8_t *s7,
        const int8_t *xq, const float *xs, const float *xcorr, int K) {
    ds4f_matvec_mxfp4_4row_i8x2(dst,     w0, w1, w2, w3, s0, s1, s2, s3, xq, xs, xcorr, K);
    ds4f_matvec_mxfp4_4row_i8x2(dst + 4, w4, w5, w6, w7, s4, s5, s6, s7, xq, xs, xcorr, K);
}

/* ---- W4A8 with a hoisted scale prologue ----
 *
 * In ds4f_matvec_mxfp4_4row_i8 roughly 40% of the inner-loop instructions are
 * scale bookkeeping, and the worst part is per block-row:
 *
 *     movzbl s[b] ; shl 23 ; vmovd -> xmm ; vmulss xs[b] ; vbroadcastss
 *
 * -- two GPR<->XMM crossings, which Zen1 handles poorly. None of it depends on
 * the weight nibbles, so it is hoisted: for a whole 8-row group, convert all
 * 8*(K/32) E8M0 bytes to combined f32 scales (e8m0 * xscale) up front into a
 * small L1-resident buffer, and reduce the debias correction to one scalar per
 * row at the same time. The inner loop then reads a scale with a single
 * vbroadcastss from memory and folds it in with one FMA.
 *
 * K <= 4096 keeps the scratch at 8*128 floats = 4 KiB.
 */
#define DS4F_MXFP4_MAXBLK 128

static inline void ds4f_matvec_mxfp4_8row_i8h(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const uint8_t *s4, const uint8_t *s5, const uint8_t *s6, const uint8_t *s7,
        const int8_t *xq, const float *xs, const float *xcorr, int K) {
    const int nb = K / 32;
    const uint8_t *srow[8] = { s0, s1, s2, s3, s4, s5, s6, s7 };
    const uint8_t *wrow[8] = { w0, w1, w2, w3, w4, w5, w6, w7 };
    float sc[8][DS4F_MXFP4_MAXBLK];
    float corr[8];

    /* prologue: sc[r][b] = e8m0(s[r][b]) * xs[b];  corr[r] = sum_b sc[r][b]*xcorr[b] */
    for (int r = 0; r < 8; r++) {
        __m256 cacc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b += 8) {
            /* 8 E8M0 bytes -> 8 f32 powers of two, by placing the byte in the
             * f32 exponent field. No table, no conversion instruction. */
            __m256i e = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)(srow[r] + b)));
            __m256 v = _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));
            v = _mm256_mul_ps(v, _mm256_loadu_ps(xs + b));
            _mm256_storeu_ps(&sc[r][b], v);
            cacc = _mm256_fmadd_ps(v, _mm256_loadu_ps(xcorr + b), cacc);
        }
        corr[r] = ds4f_hsum256(cacc);
    }

    const __m128i tbl  = _mm_loadu_si128((const __m128i *)ds4f_mxfp4_u8_tbl);
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    for (int rg = 0; rg < 8; rg += 4) {
        __m128 f0 = _mm_setzero_ps(), f1 = _mm_setzero_ps();
        __m128 f2 = _mm_setzero_ps(), f3 = _mm_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m128i xlo = _mm_loadu_si128((const __m128i *)(xq + (size_t)b * 32));
            __m128i xhi = _mm_loadu_si128((const __m128i *)(xq + (size_t)b * 32 + 16));
            #define MXFP4_ROW_I8H(R, FACC) do {                                  \
                __m128i raw = _mm_loadu_si128(                                   \
                    (const __m128i *)(wrow[rg + (R)] + (size_t)b * 16));          \
                __m128i wl = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, mask));     \
                __m128i wh = _mm_shuffle_epi8(tbl,                               \
                    _mm_and_si128(_mm_srli_epi16(raw, 4), mask));                \
                __m128i p = _mm_add_epi32(                                       \
                    _mm_madd_epi16(_mm_maddubs_epi16(wl, xlo), ones),            \
                    _mm_madd_epi16(_mm_maddubs_epi16(wh, xhi), ones));           \
                FACC = _mm_fmadd_ps(_mm_cvtepi32_ps(p),                          \
                                    _mm_broadcast_ss(&sc[rg + (R)][b]), FACC);   \
            } while (0)
            MXFP4_ROW_I8H(0, f0); MXFP4_ROW_I8H(1, f1);
            MXFP4_ROW_I8H(2, f2); MXFP4_ROW_I8H(3, f3);
            #undef MXFP4_ROW_I8H
        }
        #define HS(V) ({ __m128 t = _mm_add_ps(V, _mm_movehl_ps(V, V)); \
                         t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)); _mm_cvtss_f32(t); })
        dst[rg + 0] = HS(f0) - corr[rg + 0]; dst[rg + 1] = HS(f1) - corr[rg + 1];
        dst[rg + 2] = HS(f2) - corr[rg + 2]; dst[rg + 3] = HS(f3) - corr[rg + 3];
        #undef HS
    }
}

/* ---- W4A8, one row at a time, strictly sequential ----
 *
 * The 8-row grouping is inherited from the SVE kernel, where it amortizes an
 * f32 activation (16 KiB for K=4096) across eight rows. Under W4A8 the
 * activation is int8 -- 4 KiB, permanently L1-resident -- so there is nothing
 * left to amortize, and processing eight rows at once instead makes each
 * thread drive EIGHT concurrent memory streams 2 KiB apart. With 16 threads
 * that is 128 streams plus 128 scale streams, far past what Zen1's L2
 * prefetchers and 64-entry miss queue can track.
 *
 * This variant walks one row start-to-finish, so each thread issues one purely
 * sequential stream over its slice of the weight matrix -- the pattern the
 * hardware prefetcher is built for. Four accumulators keep the FMA chains
 * independent.
 */
static inline void ds4f_matvec_mxfp4_1row_i8(float *dst,
        const uint8_t *w, const uint8_t *s,
        const int8_t *xq, const float *xs, const float *xcorr, int K) {
    const __m128i tbl  = _mm_loadu_si128((const __m128i *)ds4f_mxfp4_u8_tbl);
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    __m128 f0 = _mm_setzero_ps(), f1 = _mm_setzero_ps();
    float c0 = 0.f, c1 = 0.f;
    const int nb = K / 32;
    for (int b = 0; b < nb; b += 2) {
        #define MXFP4_BLK(BB, XOFF, FACC, CACC) do {                          \
            __m128i raw = _mm_loadu_si128((const __m128i *)(w + (size_t)(BB) * 16)); \
            __m128i wl = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, mask));      \
            __m128i wh = _mm_shuffle_epi8(tbl,                                \
                _mm_and_si128(_mm_srli_epi16(raw, 4), mask));                 \
            __m128i p = _mm_add_epi32(                                        \
                _mm_madd_epi16(_mm_maddubs_epi16(wl,                          \
                    _mm_loadu_si128((const __m128i *)(xq + (XOFF)))), ones),  \
                _mm_madd_epi16(_mm_maddubs_epi16(wh,                          \
                    _mm_loadu_si128((const __m128i *)(xq + (XOFF) + 16))), ones)); \
            float sc = ds4f_e8m0_to_f32(s[BB]) * xs[BB];                      \
            FACC = _mm_fmadd_ps(_mm_cvtepi32_ps(p), _mm_set1_ps(sc), FACC);   \
            CACC += sc * xcorr[BB];                                           \
        } while (0)
        MXFP4_BLK(b,     (size_t)b * 32,       f0, c0);
        MXFP4_BLK(b + 1, (size_t)(b + 1) * 32, f1, c1);
        #undef MXFP4_BLK
    }
    __m128 f = _mm_add_ps(f0, f1);
    __m128 t = _mm_add_ps(f, _mm_movehl_ps(f, f));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55));
    *dst = _mm_cvtss_f32(t) - (c0 + c1);
}

static inline void ds4f_matvec_mxfp4_8row_i8seq(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const uint8_t *s4, const uint8_t *s5, const uint8_t *s6, const uint8_t *s7,
        const int8_t *xq, const float *xs, const float *xcorr, int K) {
    (void)w1; (void)w2; (void)w3; (void)w4; (void)w5; (void)w6; (void)w7;
    (void)s1; (void)s2; (void)s3; (void)s4; (void)s5; (void)s6; (void)s7;
    /* rows are contiguous, so the eight rows of this group form one run */
    const size_t rw = (size_t)K / 2, rs = (size_t)K / 32;
    for (int r = 0; r < 8; r++)
        ds4f_matvec_mxfp4_1row_i8(dst + r, w0 + (size_t)r * rw, s0 + (size_t)r * rs,
                                  xq, xs, xcorr, K);
}

/* ---- exact-activation (f32) kernel, sequential single-row ----
 * Same streaming fix as ds4f_matvec_mxfp4_1row_i8, but keeping f32 activations
 * so the result is numerically the exact MXFP4 dot product (no activation
 * quantization). Kept for the accuracy/speed comparison. */
static inline void ds4f_matvec_mxfp4_1row_f32(float *dst, const uint8_t *w,
                                              const uint8_t *s, const float *x, int K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    const int nb = K / 32;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + (size_t)b * 32;
        __m256 wv[4];
        ds4f_mxfp4_unpack_block(w + (size_t)b * 16, wv);
        __m256 sc = _mm256_castsi256_ps(_mm256_set1_epi32((int)(uint32_t)s[b] << 23));
        __m256 p0 = _mm256_mul_ps(wv[0], _mm256_loadu_ps(xb));
        p0 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xb + 8), p0);
        __m256 p1 = _mm256_mul_ps(wv[2], _mm256_loadu_ps(xb + 16));
        p1 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xb + 24), p1);
        a0 = _mm256_fmadd_ps(p0, sc, a0);
        a1 = _mm256_fmadd_ps(p1, sc, a1);
    }
    *dst = ds4f_hsum256(_mm256_add_ps(a0, a1));
}

static inline void ds4f_matvec_mxfp4_8row_f32seq(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const uint8_t *s4, const uint8_t *s5, const uint8_t *s6, const uint8_t *s7,
        const float *x, int K) {
    (void)w1; (void)w2; (void)w3; (void)w4; (void)w5; (void)w6; (void)w7;
    (void)s1; (void)s2; (void)s3; (void)s4; (void)s5; (void)s6; (void)s7;
    const size_t rw = (size_t)K / 2, rs = (size_t)K / 32;
    for (int r = 0; r < 8; r++)
        ds4f_matvec_mxfp4_1row_f32(dst + r, w0 + (size_t)r * rw, s0 + (size_t)r * rs, x, K);
}

/* Scalar reference for verification. */
static inline void ds4f_matvec_mxfp4_ref(float *dst, const uint8_t *w,
                                         const uint8_t *s, const float *x, int K) {
    static const float kv[16] = { 0.f,1.f,2.f,3.f,4.f,6.f,8.f,12.f,
                                  0.f,-1.f,-2.f,-3.f,-4.f,-6.f,-8.f,-12.f };
    float acc = 0.f;
    for (int b = 0; b < K / 32; b++) {
        float p = 0.f;
        for (int j = 0; j < 16; j++) {
            uint8_t byte = w[(size_t)b * 16 + j];
            p += kv[byte & 0xf] * x[b * 32 + j];
            p += kv[byte >> 4]  * x[b * 32 + j + 16];
        }
        acc += p * ds4f_e8m0_to_f32(s[b]);
    }
    *dst = acc;
}

#endif /* __AVX2__ && __FMA__ */
#endif /* DS4F_MXFP4_AVX2_H */
