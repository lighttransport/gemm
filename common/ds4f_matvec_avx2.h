/* ds4f_matvec_avx2.h — AVX2 counterparts of the DS4F 8-row decode matvec
 * kernels, for the x86 side of hetero/ds4f.
 *
 * Included by ggml_dequant.h when __ARM_FEATURE_SVE is absent. Each kernel here
 * mirrors the signature and weight layout of the SVE original so ds4f_impl.h's
 * ds4f_mv_worker needs no per-target call-site changes.
 *
 * Layouts (unchanged from the SVE versions, see ggml_dequant.h):
 *   BF16       row-major uint16, one pointer per row
 *   BF16_PV    pair-interleaved: pAB holds [a0,b0,a1,b1,...] for rows A and B
 *   Q8_PV      K/64 blocks of 528 B = 8 fp16 row scales (16 B) + 8 rows x 64 int8
 *   FP8        row-major uint8 e4m3, one E8M0 scale per 128 columns per 128 rows
 *   MXFP4      row-major packed nibbles, K/2 B per row; within each 16-byte
 *              block, byte j low nibble = element j, high nibble = element j+16;
 *              per-32-block E8M0 scale, K/32 B per row
 *
 * PERFORMANCE NOTE (measured, see hetero/ds4f/README.md): the 8-row grouping is
 * an A64FX shape. On Zen1 it makes each thread drive eight concurrent memory
 * streams and costs ~25% of achievable bandwidth on the MXFP4 expert path, so
 * the MXFP4 kernels below walk one row at a time instead. The 8-row entry point
 * is kept only for API compatibility; it loops rows sequentially.
 */
#ifndef DS4F_MATVEC_AVX2_H
#define DS4F_MATVEC_AVX2_H

#if defined(__AVX2__) && defined(__FMA__)
#include <math.h>
#include <immintrin.h>
#include <stdint.h>

/* x86 has no FPCR; the FP8 "magic" decode's FTZ requirement is handled by not
 * using the magic decode at all on this target (see matvec_fp8e4m3_8row_magic
 * below, which forwards to the lossless LUT path). Kept as a no-op so
 * ds4f_mv_worker's call site stays common. */
static inline void ds4f_set_ftz(void) { }

static inline float ds4f_avx2_hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

static inline float ds4f_avx2_hsum128(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 0x55));
    return _mm_cvtss_f32(v);
}

/* Widen 8 bf16 halfwords to f32 by shifting them into the exponent/mantissa
 * position of an f32 -- the same reinterpretation the SVE version does. */
static inline __m256 ds4f_avx2_bf16x8(const uint16_t *p) {
    __m128i h = _mm_loadu_si128((const __m128i *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

/* Attention uses the same BF16 latent layout as the dense matvecs, but only
 * needs one dot/AXPY at a time. Keeping these helpers beside the existing
 * BF16 widening primitive removes the scalar per-element decode from the
 * long-context window path. */
static inline float ds4f_avx2_dot_bf16(const float *x, const uint16_t *k, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),
                             ds4f_avx2_bf16x8(k + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8),
                             ds4f_avx2_bf16x8(k + i + 8), a1);
    }
    for (; i + 7 < n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),
                             ds4f_avx2_bf16x8(k + i), a0);
    float sum = ds4f_avx2_hsum(_mm256_add_ps(a0, a1));
    for (; i < n; ++i) {
        uint32_t bits = (uint32_t)k[i] << 16;
        float v;
        memcpy(&v, &bits, sizeof(v));
        sum += x[i] * v;
    }
    return sum;
}

static inline void ds4f_avx2_axpy_bf16(float *out, const uint16_t *k,
                                       float weight, int n) {
    __m256 w = _mm256_set1_ps(weight);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 y = _mm256_loadu_ps(out + i);
        y = _mm256_fmadd_ps(ds4f_avx2_bf16x8(k + i), w, y);
        _mm256_storeu_ps(out + i, y);
    }
    for (; i < n; ++i) {
        uint32_t bits = (uint32_t)k[i] << 16;
        float v;
        memcpy(&v, &bits, sizeof(v));
        out[i] += weight * v;
    }
}

/* Prefill attention variant: score/accumulate eight heads while widening each
 * BF16 KV element only once. The query and output streams remain independent,
 * but the latent KV row is shared across the head block. */
static inline void ds4f_avx2_score8_bf16(float s[8], const float *q, int qs,
                                          const uint16_t *k, int n) {
    __m256 a0[8], a1[8];
    for (int h = 0; h < 8; ++h) {
        a0[h] = _mm256_setzero_ps();
        a1[h] = _mm256_setzero_ps();
    }
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 kv0 = ds4f_avx2_bf16x8(k + i);
        __m256 kv1 = ds4f_avx2_bf16x8(k + i + 8);
        for (int h = 0; h < 8; ++h) {
            const float *qh = q + (size_t)h * qs;
            a0[h] = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), kv0, a0[h]);
            a1[h] = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i + 8), kv1, a1[h]);
        }
    }
    for (; i + 7 < n; i += 8) {
        __m256 kv = ds4f_avx2_bf16x8(k + i);
        for (int h = 0; h < 8; ++h)
            a0[h] = _mm256_fmadd_ps(_mm256_loadu_ps(q + (size_t)h * qs + i), kv, a0[h]);
    }
    for (int h = 0; h < 8; ++h) {
        s[h] = ds4f_avx2_hsum(_mm256_add_ps(a0[h], a1[h]));
        for (int j = i; j < n; ++j) {
            uint32_t bits = (uint32_t)k[j] << 16;
            float v; memcpy(&v, &bits, sizeof(v));
            s[h] += q[(size_t)h * qs + j] * v;
        }
    }
}

static inline void ds4f_avx2_axpy8_bf16(float *out, int os,
                                        const uint16_t *k, const float w[8], int n) {
    __m256 ww[8];
    for (int h = 0; h < 8; ++h) ww[h] = _mm256_set1_ps(w[h]);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 kv = ds4f_avx2_bf16x8(k + i);
        for (int h = 0; h < 8; ++h) {
            float *o = out + (size_t)h * os + i;
            _mm256_storeu_ps(o, _mm256_fmadd_ps(kv, ww[h], _mm256_loadu_ps(o)));
        }
    }
    for (; i < n; ++i) {
        uint32_t bits = (uint32_t)k[i] << 16;
        float v; memcpy(&v, &bits, sizeof(v));
        for (int h = 0; h < 8; ++h)
            out[(size_t)h * os + i] += v * w[h];
    }
}

/* ---------------- BF16, row-major ---------------- */
static inline void matvec_bf16_8row(float *dst,
        const uint16_t *w0, const uint16_t *w1, const uint16_t *w2, const uint16_t *w3,
        const uint16_t *w4, const uint16_t *w5, const uint16_t *w6, const uint16_t *w7,
        const float *x, int n) {
    const uint16_t *w[8] = { w0, w1, w2, w3, w4, w5, w6, w7 };
    for (int r = 0; r < 8; r++) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        const uint16_t *wr = w[r];
        int i = 0;
        for (; i + 15 < n; i += 16) {
            a0 = _mm256_fmadd_ps(ds4f_avx2_bf16x8(wr + i),     _mm256_loadu_ps(x + i),     a0);
            a1 = _mm256_fmadd_ps(ds4f_avx2_bf16x8(wr + i + 8), _mm256_loadu_ps(x + i + 8), a1);
        }
        for (; i + 7 < n; i += 8)
            a0 = _mm256_fmadd_ps(ds4f_avx2_bf16x8(wr + i), _mm256_loadu_ps(x + i), a0);
        float acc = ds4f_avx2_hsum(_mm256_add_ps(a0, a1));
        for (; i < n; i++) {
            uint32_t b = (uint32_t)wr[i] << 16; float f;
            memcpy(&f, &b, sizeof(f));
            acc += f * x[i];
        }
        dst[r] = acc;
    }
}

/* Two-token BF16 GEMM microkernel.  Prefill walks an 8-row weight group across
 * several token vectors; loading each BF16 weight once for two tokens removes
 * the dominant weight-stream duplication without changing either token's
 * reduction order. */
static inline void matvec_bf16_8row_2x(
        float *dst0, float *dst1,
        const uint16_t *w0, const uint16_t *w1, const uint16_t *w2, const uint16_t *w3,
        const uint16_t *w4, const uint16_t *w5, const uint16_t *w6, const uint16_t *w7,
        const float *x0, const float *x1, int n) {
    const uint16_t *w[8] = { w0, w1, w2, w3, w4, w5, w6, w7 };
    for (int r = 0; r < 8; r++) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        const uint16_t *wr = w[r];
        int i = 0;
        for (; i + 15 < n; i += 16) {
            __m256 wv0 = ds4f_avx2_bf16x8(wr + i);
            __m256 wv1 = ds4f_avx2_bf16x8(wr + i + 8);
            a0 = _mm256_fmadd_ps(wv0, _mm256_loadu_ps(x0 + i), a0);
            a1 = _mm256_fmadd_ps(wv1, _mm256_loadu_ps(x0 + i + 8), a1);
            b0 = _mm256_fmadd_ps(wv0, _mm256_loadu_ps(x1 + i), b0);
            b1 = _mm256_fmadd_ps(wv1, _mm256_loadu_ps(x1 + i + 8), b1);
        }
        for (; i + 7 < n; i += 8) {
            __m256 wv = ds4f_avx2_bf16x8(wr + i);
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + i), a0);
            b0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + i), b0);
        }
        dst0[r] = ds4f_avx2_hsum(_mm256_add_ps(a0, a1));
        dst1[r] = ds4f_avx2_hsum(_mm256_add_ps(b0, b1));
        for (; i < n; i++) {
            uint32_t bits = (uint32_t)wr[i] << 16; float f;
            memcpy(&f, &bits, sizeof(f));
            dst0[r] += f * x0[i];
            dst1[r] += f * x1[i];
        }
    }
}

/* ---------------- BF16 pair-interleaved (PV) ----------------
 * pAB[2j] is row A element j, pAB[2j+1] is row B element j. The SVE kernel gets
 * the bf16->f32 widen for free from its odd-lane predicated loads; here the two
 * rows are separated with a shuffle and then shifted into place. */
static inline void matvec_bf16_8row_pv(float *dst,
        const uint16_t *pAB, const uint16_t *pCD,
        const uint16_t *pEF, const uint16_t *pGH,
        const float *x, int n) {
    const uint16_t *pair[4] = { pAB, pCD, pEF, pGH };
    for (int p = 0; p < 4; p++) {
        const uint16_t *pp = pair[p];
        __m256 aa = _mm256_setzero_ps(), ab = _mm256_setzero_ps();
        for (int i = 0; i < n; i += 8) {
            /* 16 halfwords = 8 interleaved (a,b) pairs */
            __m256i v = _mm256_loadu_si256((const __m256i *)(pp + 2 * i));
            /* even halfwords -> row A in the low 16 bits of each 32-bit lane,
             * odd halfwords -> row B; shift each into the f32 upper half */
            __m256 fa = _mm256_castsi256_ps(_mm256_slli_epi32(v, 16));
            __m256 fb = _mm256_castsi256_ps(_mm256_and_si256(v, _mm256_set1_epi32((int)0xFFFF0000u)));
            __m256 vx = _mm256_loadu_ps(x + i);
            aa = _mm256_fmadd_ps(fa, vx, aa);
            ab = _mm256_fmadd_ps(fb, vx, ab);
        }
        dst[2 * p]     = ds4f_avx2_hsum(aa);
        dst[2 * p + 1] = ds4f_avx2_hsum(ab);
    }
}

/* ---------------- Q8_PV (W8A8) ----------------
 * Zen1 has no VNNI, so the int8 dot is vpmaddubsw + vpmaddwd. Both operands are
 * signed here, so the weight's sign is moved onto the activation with vpsignb
 * and the weight is made unsigned via its absolute value -- the standard ggml
 * q8 trick, exact for values in [-127,127]. */
static inline void matvec_sdot_8row(float *dst, const uint8_t *group,
                                    const int8_t *xq, const float *xscale, int K) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[8];
    for (int r = 0; r < 8; r++) acc[r] = _mm256_setzero_ps();
    int nb = K / 64;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = group + (size_t)b * 528;
        const uint16_t *scl = (const uint16_t *)blk;
        const int8_t *qs = (const int8_t *)(blk + 16);
        float xs = xscale[b];
        for (int r = 0; r < 8; r++) {
            const int8_t *wr = qs + (size_t)r * 64;
            __m256i sum = _mm256_setzero_si256();
            for (int c = 0; c < 64; c += 32) {
                __m256i wv = _mm256_loadu_si256((const __m256i *)(wr + c));
                __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + (size_t)b * 64 + c));
                __m256i wabs = _mm256_sign_epi8(wv, wv);        /* |w| as u8 */
                __m256i xsig = _mm256_sign_epi8(xv, wv);        /* x * sign(w) */
                sum = _mm256_add_epi32(sum,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(wabs, xsig), ones));
            }
            float sc = ggml_fp16_to_fp32(scl[r]) * xs;
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), _mm256_set1_ps(sc), acc[r]);
        }
    }
    for (int r = 0; r < 8; r++) dst[r] = ds4f_avx2_hsum(acc[r]);
}

/* ---------------- FP8 E4M3, 128-column E8M0 blocks ---------------- */
static inline void matvec_fp8e4m3_8row(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const uint32_t *lut, const float *x, int K) {
    const uint8_t *w[8] = { w0, w1, w2, w3, w4, w5, w6, w7 };
    for (int r = 0; r < 8; r++) {
        const uint8_t *wr = w[r];
        __m256 a0 = _mm256_setzero_ps();
        for (int c0 = 0; c0 < K; c0 += 128) {
            __m256 vs = _mm256_set1_ps(ggml_e8m0_to_fp32(escale[c0 >> 7]));
            __m256 blk = _mm256_setzero_ps();
            for (int c = c0; c < c0 + 128; c += 8) {
                __m256i idx = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64((const __m128i *)(wr + c)));
                __m256 wv = _mm256_castsi256_ps(
                    _mm256_i32gather_epi32((const int *)lut, idx, 4));
                blk = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + c), blk);
            }
            a0 = _mm256_fmadd_ps(blk, vs, a0);
        }
        dst[r] = ds4f_avx2_hsum(a0);
    }
}

/* Two-token FP8 GEMM microkernel.  The FP8 LUT gather is shared by the two
 * activation vectors; this is particularly effective for the large BF16 head
 * and FP8 o-projections during batched prefill. */
static inline void matvec_fp8e4m3_8row_2x(
        float *dst0, float *dst1,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const uint32_t *lut,
        const float *x0, const float *x1, int K) {
    const uint8_t *w[8] = { w0, w1, w2, w3, w4, w5, w6, w7 };
    for (int r = 0; r < 8; r++) {
        const uint8_t *wr = w[r];
        __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
        for (int c0 = 0; c0 < K; c0 += 128) {
            __m256 vs = _mm256_set1_ps(ggml_e8m0_to_fp32(escale[c0 >> 7]));
            __m256 blk0 = _mm256_setzero_ps(), blk1 = _mm256_setzero_ps();
            for (int c = c0; c < c0 + 128; c += 8) {
                __m256i idx = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64((const __m128i *)(wr + c)));
                __m256 wv = _mm256_castsi256_ps(
                    _mm256_i32gather_epi32((const int *)lut, idx, 4));
                blk0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + c), blk0);
                blk1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + c), blk1);
            }
            a0 = _mm256_fmadd_ps(blk0, vs, a0);
            b0 = _mm256_fmadd_ps(blk1, vs, b0);
        }
        dst0[r] = ds4f_avx2_hsum(a0);
        dst1[r] = ds4f_avx2_hsum(b0);
    }
}

/* The SVE "magic" decode avoids the LUT gather at the cost of flushing FP8
 * subnormals (it requires FTZ), and its win is specific to A64FX's gather cost.
 * It is not implemented here; ds4f_load_real forces m->fp8_magic = 0 on targets
 * without SVE, so this is unreachable. It exists so the call site in
 * ds4f_mv_worker stays target-independent. */
static inline void matvec_fp8e4m3_8row_magic(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const float *x, int K) {
    (void)dst; (void)w0; (void)w1; (void)w2; (void)w3;
    (void)w4; (void)w5; (void)w6; (void)w7; (void)escale; (void)x; (void)K;
    fprintf(stderr, "matvec_fp8e4m3_8row_magic: SVE-only; expected fp8_magic=0\n");
    abort();
}

/* ---------------- MXFP4 ----------------
 * See the note at the top of this file: one row at a time, not eight. */
static inline void ds4f_mxfp4_unpack16(const uint8_t *w, __m256 out[4]) {
    static const int8_t tbl16[16] = { 0, 1, 2, 3, 4, 6, 8, 12,
                                      0, -1, -2, -3, -4, -6, -8, -12 };
    const __m128i tbl  = _mm_loadu_si128((const __m128i *)tbl16);
    const __m128i mask = _mm_set1_epi8(0x0f);
    __m128i raw = _mm_loadu_si128((const __m128i *)w);
    __m128i lo = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, mask));
    __m128i hi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw, 4), mask));
    out[0] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
    out[1] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
    out[2] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
    out[3] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
}

/* Exact-activation MXFP4 row dot. */
static inline void matvec_mxfp4_1row(float *dst, const uint8_t *w, const uint8_t *s,
                                     const float *x, int K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int b = 0; b < K / 32; b++) {
        const float *xb = x + (size_t)b * 32;
        __m256 wv[4];
        ds4f_mxfp4_unpack16(w + (size_t)b * 16, wv);
        __m256 sc = _mm256_castsi256_ps(_mm256_set1_epi32((int)(uint32_t)s[b] << 23));
        __m256 p0 = _mm256_mul_ps(wv[0], _mm256_loadu_ps(xb));
        p0 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xb + 8), p0);
        __m256 p1 = _mm256_mul_ps(wv[2], _mm256_loadu_ps(xb + 16));
        p1 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xb + 24), p1);
        a0 = _mm256_fmadd_ps(p0, sc, a0);
        a1 = _mm256_fmadd_ps(p1, sc, a1);
    }
    *dst = ds4f_avx2_hsum(_mm256_add_ps(a0, a1));
}

static inline void matvec_mxfp4_8row(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, const uint8_t *s3,
        const uint8_t *s4, const uint8_t *s5, const uint8_t *s6, const uint8_t *s7,
        const float *x, int K) {
    const uint8_t *w[8] = { w0, w1, w2, w3, w4, w5, w6, w7 };
    const uint8_t *s[8] = { s0, s1, s2, s3, s4, s5, s6, s7 };
    for (int r = 0; r < 8; r++) matvec_mxfp4_1row(dst + r, w[r], s[r], x, K);
}

/* ---- MXFP4 W4A8: the throughput path for the routed experts ----
 *
 *   sum_k w_k x_k = e8m0 * xscale_b * ( sum_k wu_k xq_k - 12 * sum_k xq_k )
 *
 * with wu = value + 12 unsigned (range 0..24) so it can be vpmaddubsw's u8
 * operand, and the debias term depending only on the block -- so it is computed
 * once per matvec, not once per row. Overflow is safe: 24*127*2 = 6096 < 32767.
 *
 * This quantizes the ACTIVATION to int8 per 32 values; weights stay exact
 * MXFP4. It measured 43.3 GB/s vs 31.0 for the exact-f32 kernel on a 1950X
 * (hetero/ds4f/README.md), which is the difference between missing and clearing
 * the 10 tok/s target. Gated at the call site by DS4F_MXFP4_W4A8.
 */
static const uint8_t ds4f_mxfp4_u8_tbl16[16] = {
    12, 13, 14, 15, 16, 18, 20, 24,
    12, 11, 10,  9,  8,  6,  4,  0,
};

/* xq[K] int8, xs[K/32] block scale, xc[K/32] = 12 * sum(xq) per block. */
static inline void ds4f_mxfp4_quant_act(const float *x, int K,
                                        int8_t *xq, float *xs, float *xc) {
    for (int b = 0; b < K / 32; b++) {
        const float *xb = x + (size_t)b * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; j++) { float a = fabsf(xb[j]); if (a > amax) amax = a; }
        float inv = (amax > 0.f) ? 127.0f / amax : 0.f;
        int sum = 0;
        for (int j = 0; j < 32; j++) {
            int v = (int)lrintf(xb[j] * inv);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            xq[(size_t)b * 32 + j] = (int8_t)v;
            sum += v;
        }
        xs[b] = amax / 127.0f;
        xc[b] = 12.0f * (float)sum;
    }
}

static inline void matvec_mxfp4_1row_i8(float *dst, const uint8_t *w, const uint8_t *s,
                                        const int8_t *xq, const float *xs,
                                        const float *xc, int K) {
    const __m128i tbl  = _mm_loadu_si128((const __m128i *)ds4f_mxfp4_u8_tbl16);
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    __m128 f0 = _mm_setzero_ps(), f1 = _mm_setzero_ps();
    float c0 = 0.f, c1 = 0.f;
    const int nb = K / 32;
    for (int b = 0; b < nb; b += 2) {
        #define DS4F_MXFP4_BLK(BB, FACC, CACC) do {                              \
            __m128i raw = _mm_loadu_si128((const __m128i *)(w + (size_t)(BB) * 16)); \
            __m128i wl = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, mask));        \
            __m128i wh = _mm_shuffle_epi8(tbl,                                   \
                _mm_and_si128(_mm_srli_epi16(raw, 4), mask));                    \
            const int8_t *xb = xq + (size_t)(BB) * 32;                           \
            __m128i p = _mm_add_epi32(                                           \
                _mm_madd_epi16(_mm_maddubs_epi16(wl,                             \
                    _mm_loadu_si128((const __m128i *)xb)), ones),                \
                _mm_madd_epi16(_mm_maddubs_epi16(wh,                             \
                    _mm_loadu_si128((const __m128i *)(xb + 16))), ones));        \
            float sc = ggml_e8m0_to_fp32(s[BB]) * xs[BB];                        \
            FACC = _mm_fmadd_ps(_mm_cvtepi32_ps(p), _mm_set1_ps(sc), FACC);      \
            CACC += sc * xc[BB];                                                 \
        } while (0)
        DS4F_MXFP4_BLK(b, f0, c0);
        if (b + 1 < nb) DS4F_MXFP4_BLK(b + 1, f1, c1);   /* nb is even for DS4F's K=4096/2048 */
        #undef DS4F_MXFP4_BLK
    }
    *dst = ds4f_avx2_hsum128(_mm_add_ps(f0, f1)) - (c0 + c1);
}

/* ---- RAW (on-disk) MXFP4 layout, for zero-copy expert weights ----
 *
 * ds4f_copy_worker normally REPACKS MXFP4 experts while loading them into the
 * arena, doing two things:
 *   1. nibble permutation -- on disk, byte j holds elements 2j (low nibble) and
 *      2j+1 (high); the repacked form wants byte j to hold elements j and j+16.
 *   2. scale adjustment   -- e8m0 byte e becomes (e ? e-1 : 0), i.e. x0.5,
 *      because the e2m1 code table ds4f_kvalues_mxfp4_f32 is the 2x form.
 *
 * That repack is why the experts have to be materialized in a 155 GB anonymous
 * arena, which does not fit on this host alongside the page cache for the
 * source. The kernels below consume the ON-DISK bytes instead, so expert
 * tensors can point straight into the mapped safetensors shards -- clean,
 * evictable, file-backed pages, and no copy at all.
 *
 * Both differences are absorbed without touching the weights:
 *   1. the ACTIVATION is permuted once per matvec into [evens | odds] per
 *      32-element block, which is exactly the order the on-disk nibbles unpack
 *      to (pshufb yields elements 0,2,4,...,30 then 1,3,5,...,31).
 *   2. the scale byte is decremented in the kernel, reproducing the repack's
 *      arithmetic EXACTLY -- including its flush of e<=1 to +0.0, which
 *      ggml_e8m0_to_fp32_half does NOT do (it keeps a denormal there).
 */
static inline float ds4f_mxfp4_raw_scale(uint8_t e) {
    return ggml_e8m0_to_fp32(e ? (uint8_t)(e - 1) : (uint8_t)0);
}

/* f32 activation permuted per 32-block into [evens(16) | odds(16)]. */
static inline void ds4f_mxfp4_perm_act_f32(const float *x, int K, float *xp) {
    for (int b = 0; b < K / 32; b++) {
        const float *xb = x + (size_t)b * 32;
        float *dp = xp + (size_t)b * 32;
        for (int j = 0; j < 16; j++) { dp[j] = xb[2 * j]; dp[j + 16] = xb[2 * j + 1]; }
    }
}

/* int8 activation quantization writing the same permuted order. Block scale and
 * debias are unaffected by the permutation (both are order-independent sums). */
static inline void ds4f_mxfp4_quant_act_raw(const float *x, int K,
                                            int8_t *xq, float *xs, float *xc) {
    for (int b = 0; b < K / 32; b++) {
        const float *xb = x + (size_t)b * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; j++) { float a = fabsf(xb[j]); if (a > amax) amax = a; }
        float inv = (amax > 0.f) ? 127.0f / amax : 0.f;
        int8_t *dq = xq + (size_t)b * 32;
        int sum = 0;
        for (int j = 0; j < 32; j++) {
            int v = (int)lrintf(xb[j] * inv);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            dq[(j & 1) ? (16 + (j >> 1)) : (j >> 1)] = (int8_t)v;   /* evens | odds */
            sum += v;
        }
        xs[b] = amax / 127.0f;
        xc[b] = 12.0f * (float)sum;
    }
}

static inline void matvec_mxfp4_1row_f32_raw(float *dst, const uint8_t *w,
                                             const uint8_t *s, const float *xp, int K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int b = 0; b < K / 32; b++) {
        const float *xb = xp + (size_t)b * 32;
        __m256 wv[4];
        ds4f_mxfp4_unpack16(w + (size_t)b * 16, wv);
        __m256 sc = _mm256_set1_ps(ds4f_mxfp4_raw_scale(s[b]));
        __m256 p0 = _mm256_mul_ps(wv[0], _mm256_loadu_ps(xb));
        p0 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xb + 8), p0);
        __m256 p1 = _mm256_mul_ps(wv[2], _mm256_loadu_ps(xb + 16));
        p1 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xb + 24), p1);
        a0 = _mm256_fmadd_ps(p0, sc, a0);
        a1 = _mm256_fmadd_ps(p1, sc, a1);
    }
    *dst = ds4f_avx2_hsum(_mm256_add_ps(a0, a1));
}

/* Four-token form of matvec_mxfp4_1row_f32_raw.
 *
 * ds4f_mxfp4_unpack16 is about fifteen ops (mask, shift, two pshufb, four
 * sign-extends and four int-to-float converts) to produce one block's 32
 * weights, against six ops to consume them for one token.  The 1-row kernel
 * repeats that decode for every token, so a prefill tile of M tokens unpacks
 * each expert weight M times.  Here each block is unpacked once and reused by
 * four tokens.
 *
 * Every token keeps its own a0/a1 pair and accumulates in exactly the order
 * the 1-row kernel uses, so each output is bit-identical to the corresponding
 * matvec_mxfp4_1row_f32_raw call.  xp is [4, K] with stride K; results go to
 * dst[0], dst[Ys], dst[2*Ys], dst[3*Ys]. */
static inline void matvec_mxfp4_1row_f32_raw_4x(float *dst, int Ys,
                                                const uint8_t *w, const uint8_t *s,
                                                const float *xp, int K) {
    __m256 a0[4], a1[4];
    for (int t = 0; t < 4; t++) {
        a0[t] = _mm256_setzero_ps();
        a1[t] = _mm256_setzero_ps();
    }
    const size_t xstride = (size_t)K;
    for (int b = 0; b < K / 32; b++) {
        __m256 wv[4];
        ds4f_mxfp4_unpack16(w + (size_t)b * 16, wv);
        __m256 sc = _mm256_set1_ps(ds4f_mxfp4_raw_scale(s[b]));
        for (int t = 0; t < 4; t++) {
            const float *xb = xp + (size_t)t * xstride + (size_t)b * 32;
            __m256 p0 = _mm256_mul_ps(wv[0], _mm256_loadu_ps(xb));
            p0 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xb + 8), p0);
            __m256 p1 = _mm256_mul_ps(wv[2], _mm256_loadu_ps(xb + 16));
            p1 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xb + 24), p1);
            a0[t] = _mm256_fmadd_ps(p0, sc, a0[t]);
            a1[t] = _mm256_fmadd_ps(p1, sc, a1[t]);
        }
    }
    for (int t = 0; t < 4; t++)
        dst[(size_t)t * Ys] = ds4f_avx2_hsum(_mm256_add_ps(a0[t], a1[t]));
}

/* Decode a row once for reuse across a complete prefill tile. MXFP4 values
 * times E8M0 scales are exactly representable in f32; this keeps activation
 * quality identical to the exact path (only dot-product reassociation differs). */
static inline void ds4f_mxfp4_dequant_row_f32(float *dst, const uint8_t *w,
                                               const uint8_t *s, int K) {
    for (int b = 0; b < K / 32; ++b) {
        __m256 wv[4];
        ds4f_mxfp4_unpack16(w + (size_t)b * 16, wv);
        __m256 sc = _mm256_set1_ps(ds4f_mxfp4_raw_scale(s[b]));
        _mm256_storeu_ps(dst + (size_t)b * 32,      _mm256_mul_ps(wv[0], sc));
        _mm256_storeu_ps(dst + (size_t)b * 32 + 8,  _mm256_mul_ps(wv[1], sc));
        _mm256_storeu_ps(dst + (size_t)b * 32 + 16, _mm256_mul_ps(wv[2], sc));
        _mm256_storeu_ps(dst + (size_t)b * 32 + 24, _mm256_mul_ps(wv[3], sc));
    }
}

static inline void ds4f_mxfp4_decoded_row_4x(float *dst, int Ys,
                                              const float *w, const float *x,
                                              int Xs, int K) {
    __m256 a0[4], a1[4];
    for (int t = 0; t < 4; ++t) { a0[t] = _mm256_setzero_ps(); a1[t] = _mm256_setzero_ps(); }
    for (int b = 0; b < K / 32; ++b) {
        const float *wb = w + (size_t)b * 32;
        for (int t = 0; t < 4; ++t) {
            const float *xb = x + (size_t)t * Xs + (size_t)b * 32;
            __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(wb), _mm256_loadu_ps(xb));
            p0 = _mm256_fmadd_ps(_mm256_loadu_ps(wb + 8), _mm256_loadu_ps(xb + 8), p0);
            __m256 p1 = _mm256_mul_ps(_mm256_loadu_ps(wb + 16), _mm256_loadu_ps(xb + 16));
            p1 = _mm256_fmadd_ps(_mm256_loadu_ps(wb + 24), _mm256_loadu_ps(xb + 24), p1);
            if (b & 1) { a1[t] = _mm256_add_ps(a1[t], p0); a1[t] = _mm256_add_ps(a1[t], p1); }
            else       { a0[t] = _mm256_add_ps(a0[t], p0); a0[t] = _mm256_add_ps(a0[t], p1); }
        }
    }
    for (int t = 0; t < 4; ++t)
        dst[(size_t)t * Ys] = ds4f_avx2_hsum(_mm256_add_ps(a0[t], a1[t]));
}

static inline void matvec_mxfp4_1row_i8_raw(float *dst, const uint8_t *w, const uint8_t *s,
                                            const int8_t *xq, const float *xs,
                                            const float *xc, int K) {
    const __m128i tbl  = _mm_loadu_si128((const __m128i *)ds4f_mxfp4_u8_tbl16);
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    __m128 f0 = _mm_setzero_ps(), f1 = _mm_setzero_ps();
    float c0 = 0.f, c1 = 0.f;
    const int nb = K / 32;
    for (int b = 0; b < nb; b += 2) {
        #define DS4F_MXFP4_BLK_RAW(BB, FACC, CACC) do {                          \
            __m128i raw = _mm_loadu_si128((const __m128i *)(w + (size_t)(BB) * 16)); \
            __m128i wl = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, mask));        \
            __m128i wh = _mm_shuffle_epi8(tbl,                                   \
                _mm_and_si128(_mm_srli_epi16(raw, 4), mask));                    \
            const int8_t *xb = xq + (size_t)(BB) * 32;                           \
            __m128i p = _mm_add_epi32(                                           \
                _mm_madd_epi16(_mm_maddubs_epi16(wl,                             \
                    _mm_loadu_si128((const __m128i *)xb)), ones),                \
                _mm_madd_epi16(_mm_maddubs_epi16(wh,                             \
                    _mm_loadu_si128((const __m128i *)(xb + 16))), ones));        \
            float sc = ds4f_mxfp4_raw_scale(s[BB]) * xs[BB];                     \
            FACC = _mm_fmadd_ps(_mm_cvtepi32_ps(p), _mm_set1_ps(sc), FACC);      \
            CACC += sc * xc[BB];                                                 \
        } while (0)
        DS4F_MXFP4_BLK_RAW(b, f0, c0);
        if (b + 1 < nb) DS4F_MXFP4_BLK_RAW(b + 1, f1, c1);
        #undef DS4F_MXFP4_BLK_RAW
    }
    *dst = ds4f_avx2_hsum128(_mm_add_ps(f0, f1)) - (c0 + c1);
}

/* Four-token form of matvec_mxfp4_1row_i8_raw.
 *
 * The nibble decode -- the 16-byte load, the mask, the shift and the two
 * pshufb that feed it -- is about half the per-block work and does not depend
 * on the activation.  The 1-row kernel repeats it once per token, so a
 * prefill tile of M tokens decodes every expert weight M times.  Here it is
 * done once per block and reused by four tokens.
 *
 * Each token keeps its own pair of accumulators, laid out and reduced exactly
 * as the 1-row kernel does (even blocks into f0/c0, odd into f1/c1, then
 * hsum(f0+f1) - (c0+c1)), so each output is bit-identical to the
 * corresponding matvec_mxfp4_1row_i8_raw call.
 *
 * xq is [4, K] with stride K; xs and xc are [4, K/32] with stride K/32.
 * The four results are written to dst[0], dst[Ys], dst[2*Ys], dst[3*Ys]. */
static inline void matvec_mxfp4_1row_i8_raw_4x(float *dst, int Ys,
                                               const uint8_t *w, const uint8_t *s,
                                               const int8_t *xq, const float *xs,
                                               const float *xc, int K) {
    const __m128i tbl  = _mm_loadu_si128((const __m128i *)ds4f_mxfp4_u8_tbl16);
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    __m128 f0[4], f1[4];
    float c0[4], c1[4];
    for (int t = 0; t < 4; t++) {
        f0[t] = _mm_setzero_ps(); f1[t] = _mm_setzero_ps();
        c0[t] = 0.f; c1[t] = 0.f;
    }
    const size_t qs = (size_t)K, ss = (size_t)K / 32;
    const int nb = K / 32;
    for (int b = 0; b < nb; b += 2) {
        #define DS4F_MXFP4_BLK_RAW_4X(BB, FACC, CACC) do {                       \
            __m128i raw = _mm_loadu_si128((const __m128i *)(w + (size_t)(BB) * 16)); \
            __m128i wl = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, mask));        \
            __m128i wh = _mm_shuffle_epi8(tbl,                                   \
                _mm_and_si128(_mm_srli_epi16(raw, 4), mask));                    \
            float ws = ds4f_mxfp4_raw_scale(s[BB]);                              \
            for (int t = 0; t < 4; t++) {                                        \
                const int8_t *xb = xq + (size_t)t * qs + (size_t)(BB) * 32;      \
                __m128i p = _mm_add_epi32(                                       \
                    _mm_madd_epi16(_mm_maddubs_epi16(wl,                         \
                        _mm_loadu_si128((const __m128i *)xb)), ones),            \
                    _mm_madd_epi16(_mm_maddubs_epi16(wh,                         \
                        _mm_loadu_si128((const __m128i *)(xb + 16))), ones));    \
                float sc = ws * xs[(size_t)t * ss + (BB)];                       \
                FACC[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(p),                       \
                                       _mm_set1_ps(sc), FACC[t]);                \
                CACC[t] += sc * xc[(size_t)t * ss + (BB)];                       \
            }                                                                    \
        } while (0)
        DS4F_MXFP4_BLK_RAW_4X(b, f0, c0);
        if (b + 1 < nb) DS4F_MXFP4_BLK_RAW_4X(b + 1, f1, c1);
        #undef DS4F_MXFP4_BLK_RAW_4X
    }
    for (int t = 0; t < 4; t++)
        dst[(size_t)t * Ys] =
            ds4f_avx2_hsum128(_mm_add_ps(f0[t], f1[t])) - (c0[t] + c1[t]);
}

#endif /* __AVX2__ && __FMA__ */
#endif /* DS4F_MATVEC_AVX2_H */
