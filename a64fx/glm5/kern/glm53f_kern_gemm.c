/* Prefill GEMM on 16-row weight panels: Y[t][r] = sum_k X[t][k] W[r][k].
 *
 * Micro-tile: 4 panels (64 output rows, one SVE vector each) x 6 tokens =
 * 24 int32 accumulators.  Per 4-value k-group: 4 weight vector loads, 6
 * LD1RW token broadcasts, 24 SDOT (the 6x4 shape of a64fx/int8-cmg, which
 * reaches 94% of SDOT peak).  After every scale block the int32 sums are
 * converted and accumulated in f32 with the row scale times token scale.
 *
 * Two weight formats share this code, differing only in the scale block:
 *   SB = 32:  Q8_0R16 panels (lossless GGUF Q8_0; 576 B per 16x32 block).
 *             Structural ceiling ~67% of SDOT peak: per block the epilogue
 *             is 24 x (zero, SCVTF, FMUL, FMLA) = 96 FP ops per 192 SDOTs.
 *   SB = 128: int8 with one f32 scale per row per 128 columns (the
 *             FP8-derived 128x128-block INT8 path; lossy relative to GGUF,
 *             quality-gated): 16*128 + 64 = 2112 B per block, ceiling ~89%.
 *   SB = 256: same layout with 256-column scale blocks, ceiling ~94%.
 * Activations: int8 X[t][k] (row stride ldx bytes) and f32 scales
 * xs[t][k / SB] (row stride ldxs floats), i.e. per-token Q8 blocks of SB. */
#include "glm53f_kern.h"
#include <arm_sve.h>

static inline svint8_t gk_gbc4(const int8_t *x) {
    int32_t v;
    __builtin_memcpy(&v, x, 4);
    return svreinterpret_s8_s32(svdup_n_s32(v));
}

#define GEMM_KGROUP(J) do {                                                        \
    const svint8_t w0 = svld1_s8(p8, wp0 + 64 * (J)), w1 = svld1_s8(p8, wp1 + 64 * (J)); \
    const svint8_t w2 = svld1_s8(p8, wp2 + 64 * (J)), w3 = svld1_s8(p8, wp3 + 64 * (J)); \
    const int ko = 4 * (J);                                                        \
    svint8_t xb;                                                                   \
    xb = gk_gbc4(x0 + ko); i00 = svdot_s32(i00, w0, xb); i01 = svdot_s32(i01, w1, xb); \
                           i02 = svdot_s32(i02, w2, xb); i03 = svdot_s32(i03, w3, xb); \
    xb = gk_gbc4(x1 + ko); i10 = svdot_s32(i10, w0, xb); i11 = svdot_s32(i11, w1, xb); \
                           i12 = svdot_s32(i12, w2, xb); i13 = svdot_s32(i13, w3, xb); \
    xb = gk_gbc4(x2 + ko); i20 = svdot_s32(i20, w0, xb); i21 = svdot_s32(i21, w1, xb); \
                           i22 = svdot_s32(i22, w2, xb); i23 = svdot_s32(i23, w3, xb); \
    xb = gk_gbc4(x3 + ko); i30 = svdot_s32(i30, w0, xb); i31 = svdot_s32(i31, w1, xb); \
                           i32 = svdot_s32(i32, w2, xb); i33 = svdot_s32(i33, w3, xb); \
    xb = gk_gbc4(x4 + ko); i40 = svdot_s32(i40, w0, xb); i41 = svdot_s32(i41, w1, xb); \
                           i42 = svdot_s32(i42, w2, xb); i43 = svdot_s32(i43, w3, xb); \
    xb = gk_gbc4(x5 + ko); i50 = svdot_s32(i50, w0, xb); i51 = svdot_s32(i51, w1, xb); \
                           i52 = svdot_s32(i52, w2, xb); i53 = svdot_s32(i53, w3, xb); \
} while (0)

/* Epilogue for token T: y[T][r..r+63] += cvt(i) * (row scale * token scale).
 * The f32 accumulators live in the (L1-resident) output tile: 24 int32
 * accumulators + 4 weight vectors + a broadcast leave no registers for 24
 * f32 accumulators. */
#define GEMM_EPI_T(T) do {                                                         \
    const float s = xs##T[b];                                                      \
    float *yt = y + (size_t)(T) * ldy + r;                                         \
    svst1_f32(p32, yt, svmla_f32_x(p32, svld1_f32(p32, yt),                        \
              svcvt_f32_s32_x(p32, i##T##0), svmul_n_f32_x(p32, d0, s)));          \
    svst1_f32(p32, yt + 16, svmla_f32_x(p32, svld1_f32(p32, yt + 16),              \
              svcvt_f32_s32_x(p32, i##T##1), svmul_n_f32_x(p32, d1, s)));          \
    svst1_f32(p32, yt + 32, svmla_f32_x(p32, svld1_f32(p32, yt + 32),              \
              svcvt_f32_s32_x(p32, i##T##2), svmul_n_f32_x(p32, d2, s)));          \
    svst1_f32(p32, yt + 48, svmla_f32_x(p32, svld1_f32(p32, yt + 48),              \
              svcvt_f32_s32_x(p32, i##T##3), svmul_n_f32_x(p32, d3, s)));          \
} while (0)

#define GEMM_ZERO_T(T) i##T##0 = i##T##1 = i##T##2 = i##T##3 = svdup_s32(0)

/* Scale block sb in {32, 128, 256}: bytes per 16 x sb block = 16*sb + 64. */
static inline size_t gemm_blk(int sb) { return (size_t)16 * sb + 64; }

static inline void gemm_tile(int sb, const uint8_t *w, size_t panel_bytes, int K,
                             const int8_t *x, size_t ldx, const float *xs, size_t ldxs,
                             float *y, size_t ldy, int r) {
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const size_t blk = gemm_blk(sb);
    const int8_t *wp0 = (const int8_t *)w, *wp1 = wp0 + panel_bytes;
    const int8_t *wp2 = wp1 + panel_bytes, *wp3 = wp2 + panel_bytes;
    const int8_t *x0 = x, *x1 = x + ldx, *x2 = x + 2 * ldx, *x3 = x + 3 * ldx;
    const int8_t *x4 = x + 4 * ldx, *x5 = x + 5 * ldx;
    const float *xs0 = xs, *xs1 = xs + ldxs, *xs2 = xs + 2 * ldxs, *xs3 = xs + 3 * ldxs;
    const float *xs4 = xs + 4 * ldxs, *xs5 = xs + 5 * ldxs;
    for (int t = 0; t < 6; ++t)
        for (int q = 0; q < 64; q += 16) svst1_f32(p32, y + (size_t)t * ldy + r + q, svdup_f32(0.0f));
    for (int b = 0; b < K / sb; ++b) {
        svint32_t i00, i01, i02, i03, i10, i11, i12, i13, i20, i21, i22, i23;
        svint32_t i30, i31, i32, i33, i40, i41, i42, i43, i50, i51, i52, i53;
        GEMM_ZERO_T(0); GEMM_ZERO_T(1); GEMM_ZERO_T(2);
        GEMM_ZERO_T(3); GEMM_ZERO_T(4); GEMM_ZERO_T(5);
#pragma clang loop unroll_count(2)
        for (int j = 0; j < sb / 4; ++j) GEMM_KGROUP(j);
        const svfloat32_t d0 = svld1_f32(p32, (const float *)(wp0 + 16 * sb));
        const svfloat32_t d1 = svld1_f32(p32, (const float *)(wp1 + 16 * sb));
        const svfloat32_t d2 = svld1_f32(p32, (const float *)(wp2 + 16 * sb));
        const svfloat32_t d3 = svld1_f32(p32, (const float *)(wp3 + 16 * sb));
        GEMM_EPI_T(0); GEMM_EPI_T(1); GEMM_EPI_T(2);
        GEMM_EPI_T(3); GEMM_EPI_T(4); GEMM_EPI_T(5);
        wp0 += blk; wp1 += blk; wp2 += blk; wp3 += blk;
        x0 += sb; x1 += sb; x2 += sb; x3 += sb; x4 += sb; x5 += sb;
    }
}

size_t gk_panel_bytes_gemm(int sb, int columns) { return (size_t)(columns / sb) * gemm_blk(sb); }

void gk_pack_panel16(int sb, uint8_t *dst, const int8_t *q, const float *scale,
                     int nrows, int columns) {
    /* q: nrows x columns int8 (row-major), scale: nrows x (columns/sb) f32. */
    for (int b = 0; b < columns / sb; ++b) {
        uint8_t *blk = dst + (size_t)b * gemm_blk(sb);
        float *dw = (float *)(blk + 16 * sb);
        for (int r = 0; r < 16; ++r) {
            dw[r] = r < nrows ? scale[(size_t)r * (columns / sb) + b] : 0.0f;
            for (int j = 0; j < sb / 4; ++j)
                for (int i = 0; i < 4; ++i)
                    blk[j * 64 + r * 4 + i] = r < nrows ? (uint8_t)q[(size_t)r * columns + sb * b + 4 * j + i] : 0;
        }
    }
}

void gk_gemm_tile6x4_asm(const uint8_t *w, size_t panel_bytes, long nblocks, long kgroups4,
                         const int8_t *x, size_t ldx, const float *xs, size_t ldxs_bytes,
                         float *y, size_t ldy_bytes);

/* Assembly micro-kernel path (glm53f_kern_gemm_asm.S): no register spills. */
void gk_gemm_panel16_asm(int sb, const uint8_t *w, int K, int r0, int r1, int t0, int t1,
                         const int8_t *x, size_t ldx, const float *xs, size_t ldxs,
                         float *y, size_t ldy) {
    const size_t pb = gk_panel_bytes_gemm(sb, K);
    for (int r = r0; r < r1; r += 64)
        for (int t = t0; t + 6 <= t1; t += 6) {
            for (int u = 0; u < 6; ++u) __builtin_memset(y + (size_t)(t + u) * ldy + r, 0, 64 * 4);
            gk_gemm_tile6x4_asm(w + (size_t)(r / 16) * pb, pb, K / sb, sb / 16,
                                x + (size_t)t * ldx, ldx, xs + (size_t)t * ldxs, ldxs * 4,
                                y + (size_t)t * ldy + r, ldy * 4);
        }
}

void gk_gemm_panel16(int sb, const uint8_t *w, int K, int r0, int r1, int t0, int t1,
                     const int8_t *x, size_t ldx, const float *xs, size_t ldxs,
                     float *y, size_t ldy) {
    const size_t pb = gk_panel_bytes_gemm(sb, K);
    for (int r = r0; r < r1; r += 64)
        for (int t = t0; t + 6 <= t1; t += 6)
            gemm_tile(sb, w + (size_t)(r / 16) * pb, pb, K, x + (size_t)t * ldx, ldx,
                      xs + (size_t)t * ldxs, ldxs, y + (size_t)t * ldy, ldy, r);
}

/* ---- fully packed variant (glm53f_kern_gemm_asm.S gk_gemm_tile6x4p_asm) ---- */
void gk_gemm_tile6x4p_asm(const int8_t *w, long nblocks, long kgroups4, const int8_t *xp,
                          const float *xsp, float *y, size_t ldy_bytes);

size_t gk_panel64_bytes(int sb, int columns) { return (size_t)(columns / sb) * (64 * (size_t)sb + 256); }

/* q: 64 x columns int8 row-major (rows >= nrows are zero), scale: 64 x (columns/sb). */
void gk_pack_panel64(int sb, uint8_t *dst, const int8_t *q, const float *scale, int nrows, int columns) {
    for (int b = 0; b < columns / sb; ++b) {
        uint8_t *blk = dst + (size_t)b * (64 * (size_t)sb + 256);
        for (int j = 0; j < sb / 4; ++j)
            for (int p = 0; p < 4; ++p)
                for (int r = 0; r < 16; ++r)
                    for (int i = 0; i < 4; ++i) {
                        const int row = 16 * p + r;
                        blk[(size_t)j * 256 + p * 64 + r * 4 + i] =
                            row < nrows ? (uint8_t)q[(size_t)row * columns + sb * b + 4 * j + i] : 0;
                    }
        float *d = (float *)(blk + 64 * (size_t)sb);
        for (int row = 0; row < 64; ++row)
            d[row] = row < nrows ? scale[(size_t)row * (columns / sb) + b] : 0.0f;
    }
}

/* Pack 6 tokens: xp[k/4][t][4], xsp[b][t]. */
void gk_pack_act6(int sb, int8_t *xp, float *xsp, const int8_t *x, size_t ldx,
                  const float *xs, size_t ldxs, int K) {
    for (int g = 0; g < K / 4; ++g)
        for (int t = 0; t < 6; ++t)
            for (int i = 0; i < 4; ++i) xp[(size_t)g * 24 + t * 4 + i] = x[(size_t)t * ldx + 4 * g + i];
    for (int b = 0; b < K / sb; ++b)
        for (int t = 0; t < 6; ++t) xsp[(size_t)b * 6 + t] = xs[(size_t)t * ldxs + b];
}

/* w: 64-row super-panels (gk_panel64_bytes each); xp/xsp: packed per 6-token
 * group (K bytes x 6 and K/sb x 6 floats per group). */
/* K chunk (columns) of gk_gemm_panel64; a multiple of the scale block. */
int gk_gemm_kchunk = 512;

void gk_gemm_panel64(int sb, const uint8_t *w, int K, int r0, int r1, int t0, int t1,
                     const int8_t *xp, const float *xsp, float *y, size_t ldy) {
    /* K is processed in chunks of up to 512 columns so a 64-row weight chunk
     * (32-64 KiB) stays in L1 while every 6-token tile reuses it; the f32
     * output tile accumulates across chunks. */
    const size_t pb = gk_panel64_bytes(sb, K), blk = 64 * (size_t)sb + 256;
    const int kc = gk_gemm_kchunk > sb ? gk_gemm_kchunk / sb * sb : sb;
    for (int r = r0; r < r1; r += 64) {
        const uint8_t *wr = w + (size_t)(r / 64) * pb;
        for (int t = t0; t + 6 <= t1; t += 6)
            for (int u = 0; u < 6; ++u) __builtin_memset(y + (size_t)(t + u) * ldy + r, 0, 64 * 4);
        for (int k0 = 0; k0 < K; k0 += kc) {
            const int kn = K - k0 < kc ? K - k0 : kc;
            for (int t = t0; t + 6 <= t1; t += 6)
                gk_gemm_tile6x4p_asm((const int8_t *)(wr + (size_t)(k0 / sb) * blk), kn / sb, sb / 16,
                                     xp + (size_t)(t / 6) * 6 * K + (size_t)(k0 / 4) * 24,
                                     xsp + (size_t)(t / 6) * 6 * (K / sb) + (size_t)(k0 / sb) * 6,
                                     y + (size_t)t * ldy + r, ldy * 4);
        }
    }
}
