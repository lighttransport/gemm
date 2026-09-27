/* Q5_KP16: lossless 16-row panel repack of GGUF Q5_K for decode GEMV (v1).
 *
 * GGUF Q5_K super-block: d, dmin (fp16), 6-bit scale/min per 32-weight
 * sub-block, 4-bit qs (same order as Q4_K) and qh[32]: the fifth bit of
 * sub-block j's weight l is (qh[l] >> j) & 1.  w = d*sc_j*q - dmin*m_j,
 * q in 0..31.
 *
 * Panel per 16 rows and super-block (2944 bytes, 184 per row, +4.5%):
 *   [0, 2048)    nibble plane, identical to Q4_KP16 (k-group t of sub-block
 *                j is the low nibble of vector t/2 for even t, high for odd);
 *   [2048, 2560) high-bit plane, one 64-byte vector per sub-block: bit t of
 *                byte r*4+i is the fifth bit of W[r][32j + 4t + i];
 *   [2560, 2688) sc[j][r], [2688, 2816) m[j][r]   uint8;
 *   [2816, 2880) d[r],     [2880, 2944) dmin[r]   f32.
 * sum q*x = sum nib*x + 16 * sum hbit*x, both exact int32 SDOT sums, so the
 * high part is a separate SDOT on (plane >> t) & 1 and is scaled by 16 in
 * the integer epilogue instead of being merged into each nibble byte. */
#include "glm53f_kern.h"
#include <arm_sve.h>

#define Q5KP16_SB_BYTES 2944

size_t gk_row_bytes_q5_kp16(int columns) { return (size_t)(columns / 256) * 184; }

static inline float gk_h2f5(uint16_t h) {
    __fp16 v;
    __builtin_memcpy(&v, &h, 2);
    return (float)v;
}
static inline void gk_k4_5(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
/* Full 5-bit value of weight k (0..255) of a GGUF Q5_K block. */
static inline int q5_value(const gk_block_q5_K *b, int k) {
    const int j = k / 32, l = k % 32, g = j / 2;
    const int nib = (j & 1) ? b->qs[32 * g + l] >> 4 : b->qs[32 * g + l] & 15;
    return nib | (((b->qh[l] >> j) & 1) << 4);
}

void gk_pack_q5_kp16(uint8_t *dst, const uint8_t *rows, size_t row_bytes,
                     int nrows, int columns) {
    for (int b = 0; b < columns / 256; ++b) {
        uint8_t *sb = dst + (size_t)b * Q5KP16_SB_BYTES;
        uint8_t *hp = sb + 2048, *sc = sb + 2560, *mn = sb + 2688;
        float *dd = (float *)(sb + 2816), *dm = (float *)(sb + 2880);
        for (int r = 0; r < 16; ++r) {
            const gk_block_q5_K *blk = r < nrows
                ? (const gk_block_q5_K *)(rows + (size_t)r * row_bytes) + b : 0;
            dd[r] = blk ? gk_h2f5(blk->d) : 0.0f;
            dm[r] = blk ? gk_h2f5(blk->dmin) : 0.0f;
            for (int j = 0; j < 8; ++j) {
                uint8_t s = 0, m = 0;
                if (blk) gk_k4_5(j, blk->scales, &s, &m);
                sc[j * 16 + r] = s;
                mn[j * 16 + r] = m;
                for (int i = 0; i < 4; ++i) {
                    uint8_t bits = 0;
                    for (int t = 0; t < 8; ++t) {
                        const int q = blk ? q5_value(blk, 32 * j + 4 * t + i) : 0;
                        bits |= (uint8_t)((q >> 4) << t);
                    }
                    hp[j * 64 + r * 4 + i] = bits;
                }
                for (int v = 0; v < 4; ++v)
                    for (int i = 0; i < 4; ++i) {
                        const int lo = blk ? q5_value(blk, 32 * j + 8 * v + i) & 15 : 0;
                        const int hi = blk ? q5_value(blk, 32 * j + 8 * v + 4 + i) & 15 : 0;
                        sb[j * 256 + v * 64 + r * 4 + i] = (uint8_t)(lo | (hi << 4));
                    }
            }
        }
    }
}

static inline svint8_t gk_bc4_5(const int8_t *x) {
    int32_t v;
    __builtin_memcpy(&v, x, 4);
    return svreinterpret_s8_s32(svdup_n_s32(v));
}

#define Q5KP16_SUB(J) do {                                                        \
    const uint8_t *qj = q + (J) * 256;                                            \
    const svuint8_t hb = svld1_u8(p8, hp + 64 * (J));                             \
    const int8_t *xj = x + 32 * (J);                                              \
    svint32_t ia = svdup_s32(0), ib = svdup_s32(0);                               \
    svint32_t ha = svdup_s32(0), hc = svdup_s32(0);                               \
    for (int v = 0; v < 4; ++v) {                                                 \
        const svuint8_t pk = svld1_u8(p8, qj + 64 * v);                           \
        const svint8_t x0 = gk_bc4_5(xj + 8 * v), x1 = gk_bc4_5(xj + 8 * v + 4); \
        ia = svdot_s32(ia, svreinterpret_s8_u8(svand_n_u8_x(p8, pk, 15)), x0);    \
        ib = svdot_s32(ib, svreinterpret_s8_u8(svlsr_n_u8_x(p8, pk, 4)), x1);     \
        ha = svdot_s32(ha, svreinterpret_s8_u8(svand_n_u8_x(p8,                  \
                 svlsr_n_u8_x(p8, hb, 2 * v), 1)), x0);                           \
        hc = svdot_s32(hc, svreinterpret_s8_u8(svand_n_u8_x(p8,                  \
                 svlsr_n_u8_x(p8, hb, 2 * v + 1), 1)), x1);                       \
    }                                                                             \
    const svint32_t tot = svadd_s32_x(p32, svadd_s32_x(p32, ia, ib),              \
        svlsl_n_s32_x(p32, svadd_s32_x(p32, ha, hc), 4));                         \
    sacc = svmla_s32_x(p32, sacc, tot, svld1ub_s32(p32, sc + 16 * (J)));          \
    macc = svmla_s32_x(p32, macc, svld1ub_s32(p32, mn + 16 * (J)),               \
                       svdup_n_s32(bs[J]));                                       \
} while (0)

static inline void gk_q5_kp16_body(const gk_mv *m, int r0, int r1, int pf) {
    const int nsb = m->columns / 256;
    const size_t pb = (size_t)nsb * Q5KP16_SB_BYTES;
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    for (int r = r0; r < r1; r += 16) {
        const uint8_t *panel = m->w + (size_t)(r / 16) * pb;
        svfloat32_t f0 = svdup_f32(0.0f), f1 = f0;
        for (int b = 0; b < nsb; ++b) {
            const uint8_t *q = panel + (size_t)b * Q5KP16_SB_BYTES;
            const uint8_t *hp = q + 2048, *sc = q + 2560, *mn = q + 2688;
            if (pf)
                for (int l = 0; l < Q5KP16_SB_BYTES; l += 256)
                    __builtin_prefetch(q + pf + l, 0, 2);
            const int8_t *x = m->a->q8k[b].q;
            const int32_t *bs = m->a->q8k_bsum32 + 8 * b;
            const float dx = m->a->q8k[b].d;
            svint32_t sacc = svdup_s32(0), macc = svdup_s32(0);
#pragma clang loop unroll_count(2)
            for (int j = 0; j < 8; ++j) Q5KP16_SUB(j);
            const svfloat32_t dd = svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(q + 2816)), dx);
            const svfloat32_t dm = svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(q + 2880)), dx);
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, sacc), dd);
            f1 = svmls_f32_x(p32, f1, svcvt_f32_s32_x(p32, macc), dm);
        }
        svst1_f32(svwhilelt_b32(r, r1), m->y + r, svadd_f32_x(p32, f0, f1));
    }
}

void gk_q5_kp16_v1(const gk_mv *m, int r0, int r1) { gk_q5_kp16_body(m, r0, r1, 0); }
void gk_q5_kp16_v1pf(const gk_mv *m, int r0, int r1) { gk_q5_kp16_body(m, r0, r1, 16384); }

/* v2: each sub-block's 16 SDOTs spread over 8 two-deep chains (v parity
 * splits the nibble and high-bit chains) to shorten the per-sub-block
 * critical path that limits v1 once the 128-entry ROB holds ~2 sub-blocks. */
#define Q5KP16_V2(V, IA, IB, HA, HC) do {                                        \
    const svuint8_t pk = svld1_u8(p8, qj + 64 * (V));                             \
    const svint8_t x0 = gk_bc4_5(xj + 8 * (V)), x1 = gk_bc4_5(xj + 8 * (V) + 4); \
    IA = svdot_s32(IA, svreinterpret_s8_u8(svand_n_u8_x(p8, pk, 15)), x0);        \
    IB = svdot_s32(IB, svreinterpret_s8_u8(svlsr_n_u8_x(p8, pk, 4)), x1);         \
    HA = svdot_s32(HA, svreinterpret_s8_u8(svand_n_u8_x(p8,                      \
             svlsr_n_u8_x(p8, hb, 2 * (V)), 1)), x0);                             \
    HC = svdot_s32(HC, svreinterpret_s8_u8(svand_n_u8_x(p8,                      \
             svlsr_n_u8_x(p8, hb, 2 * (V) + 1), 1)), x1);                         \
} while (0)
#define Q5KP16_SUB2(J) do {                                                       \
    const uint8_t *qj = q + (J) * 256;                                            \
    const svuint8_t hb = svld1_u8(p8, hp + 64 * (J));                             \
    const int8_t *xj = x + 32 * (J);                                              \
    svint32_t ia0 = svdup_s32(0), ia1 = ia0, ib0 = ia0, ib1 = ia0;                \
    svint32_t ha0 = ia0, ha1 = ia0, hc0 = ia0, hc1 = ia0;                         \
    Q5KP16_V2(0, ia0, ib0, ha0, hc0);                                             \
    Q5KP16_V2(1, ia1, ib1, ha1, hc1);                                             \
    Q5KP16_V2(2, ia0, ib0, ha0, hc0);                                             \
    Q5KP16_V2(3, ia1, ib1, ha1, hc1);                                             \
    const svint32_t lo = svadd_s32_x(p32, svadd_s32_x(p32, ia0, ia1),             \
                                     svadd_s32_x(p32, ib0, ib1));                 \
    const svint32_t hi = svadd_s32_x(p32, svadd_s32_x(p32, ha0, ha1),             \
                                     svadd_s32_x(p32, hc0, hc1));                 \
    sacc = svmla_s32_x(p32, sacc, svadd_s32_x(p32, lo, svlsl_n_s32_x(p32, hi, 4)), \
                       svld1ub_s32(p32, sc + 16 * (J)));                          \
    macc = svmla_s32_x(p32, macc, svld1ub_s32(p32, mn + 16 * (J)),               \
                       svdup_n_s32(bs[J]));                                       \
} while (0)

void gk_q5_kp16_v2pf(const gk_mv *m, int r0, int r1) {
    const int nsb = m->columns / 256;
    const size_t pb = (size_t)nsb * Q5KP16_SB_BYTES;
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    for (int r = r0; r < r1; r += 16) {
        const uint8_t *panel = m->w + (size_t)(r / 16) * pb;
        svfloat32_t f0 = svdup_f32(0.0f), f1 = f0;
        for (int b = 0; b < nsb; ++b) {
            const uint8_t *q = panel + (size_t)b * Q5KP16_SB_BYTES;
            const uint8_t *hp = q + 2048, *sc = q + 2560, *mn = q + 2688;
            for (int l = 0; l < Q5KP16_SB_BYTES; l += 256)
                __builtin_prefetch(q + 16384 + l, 0, 2);
            const int8_t *x = m->a->q8k[b].q;
            const int32_t *bs = m->a->q8k_bsum32 + 8 * b;
            const float dx = m->a->q8k[b].d;
            svint32_t sacc = svdup_s32(0), macc = svdup_s32(0);
#pragma clang loop unroll_count(2)
            for (int j = 0; j < 8; ++j) Q5KP16_SUB2(j);
            const svfloat32_t dd = svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(q + 2816)), dx);
            const svfloat32_t dm = svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(q + 2880)), dx);
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, sacc), dd);
            f1 = svmls_f32_x(p32, f1, svcvt_f32_s32_x(p32, macc), dm);
        }
        svst1_f32(svwhilelt_b32(r, r1), m->y + r, svadd_f32_x(p32, f0, f1));
    }
}

/* v3: two 16-row panels per iteration sharing the activation broadcasts,
 * sums and loop work (see gk_q4_kp16_v3pf: the panel kernels are bound by
 * how much work fits in the 128-entry ROB). */
#define Q5KP16_PAIR(J) do {                                                        \
    const uint8_t *qa = q + (J) * 256, *qb = qB + (J) * 256;                       \
    const svuint8_t ha = svld1_u8(p8, q + 2048 + 64 * (J));                        \
    const svuint8_t hb2 = svld1_u8(p8, qB + 2048 + 64 * (J));                      \
    const int8_t *xj = x + 32 * (J);                                               \
    svint32_t la = svdup_s32(0), lb = la, xa = la, xb = la;                        \
    for (int v = 0; v < 4; ++v) {                                                  \
        const svint8_t x0 = gk_bc4_5(xj + 8 * v), x1 = gk_bc4_5(xj + 8 * v + 4);   \
        const svuint8_t pa = svld1_u8(p8, qa + 64 * v), pb2 = svld1_u8(p8, qb + 64 * v); \
        la = svdot_s32(la, svreinterpret_s8_u8(svand_n_u8_x(p8, pa, 15)), x0);     \
        la = svdot_s32(la, svreinterpret_s8_u8(svlsr_n_u8_x(p8, pa, 4)), x1);      \
        lb = svdot_s32(lb, svreinterpret_s8_u8(svand_n_u8_x(p8, pb2, 15)), x0);    \
        lb = svdot_s32(lb, svreinterpret_s8_u8(svlsr_n_u8_x(p8, pb2, 4)), x1);     \
        xa = svdot_s32(xa, svreinterpret_s8_u8(svand_n_u8_x(p8, svlsr_n_u8_x(p8, ha, 2 * v), 1)), x0);      \
        xa = svdot_s32(xa, svreinterpret_s8_u8(svand_n_u8_x(p8, svlsr_n_u8_x(p8, ha, 2 * v + 1), 1)), x1);  \
        xb = svdot_s32(xb, svreinterpret_s8_u8(svand_n_u8_x(p8, svlsr_n_u8_x(p8, hb2, 2 * v), 1)), x0);     \
        xb = svdot_s32(xb, svreinterpret_s8_u8(svand_n_u8_x(p8, svlsr_n_u8_x(p8, hb2, 2 * v + 1), 1)), x1); \
    }                                                                              \
    const svint32_t bsv = svdup_n_s32(bs[J]);                                      \
    sA = svmla_s32_x(p32, sA, svadd_s32_x(p32, la, svlsl_n_s32_x(p32, xa, 4)),     \
                     svld1ub_s32(p32, q + 2560 + 16 * (J)));                       \
    mA = svmla_s32_x(p32, mA, svld1ub_s32(p32, q + 2688 + 16 * (J)), bsv);         \
    sB = svmla_s32_x(p32, sB, svadd_s32_x(p32, lb, svlsl_n_s32_x(p32, xb, 4)),     \
                     svld1ub_s32(p32, qB + 2560 + 16 * (J)));                      \
    mB = svmla_s32_x(p32, mB, svld1ub_s32(p32, qB + 2688 + 16 * (J)), bsv);        \
} while (0)

void gk_q5_kp16_v3pf(const gk_mv *m, int r0, int r1) {
    const int nsb = m->columns / 256;
    const size_t pb = (size_t)nsb * Q5KP16_SB_BYTES;
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    int r = r0;
    for (; r + 16 < r1; r += 32) {
        const uint8_t *panelA = m->w + (size_t)(r / 16) * pb, *panelB = panelA + pb;
        svfloat32_t fA = svdup_f32(0.0f), gA = fA, fB = fA, gB = fA;
        for (int b = 0; b < nsb; ++b) {
            const uint8_t *q = panelA + (size_t)b * Q5KP16_SB_BYTES;
            const uint8_t *qB = panelB + (size_t)b * Q5KP16_SB_BYTES;
            for (int l = 0; l < Q5KP16_SB_BYTES; l += 256) {
                __builtin_prefetch(q + 16384 + l, 0, 2);
                __builtin_prefetch(qB + 16384 + l, 0, 2);
            }
            const int8_t *x = m->a->q8k[b].q;
            const int32_t *bs = m->a->q8k_bsum32 + 8 * b;
            const float dx = m->a->q8k[b].d;
            svint32_t sA = svdup_s32(0), mA = sA, sB = sA, mB = sA;
#pragma clang loop unroll(disable)
            for (int j = 0; j < 8; ++j) Q5KP16_PAIR(j);
            fA = svmla_f32_x(p32, fA, svcvt_f32_s32_x(p32, sA),
                             svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(q + 2816)), dx));
            gA = svmls_f32_x(p32, gA, svcvt_f32_s32_x(p32, mA),
                             svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(q + 2880)), dx));
            fB = svmla_f32_x(p32, fB, svcvt_f32_s32_x(p32, sB),
                             svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(qB + 2816)), dx));
            gB = svmls_f32_x(p32, gB, svcvt_f32_s32_x(p32, mB),
                             svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(qB + 2880)), dx));
        }
        svst1_f32(p32, m->y + r, svadd_f32_x(p32, fA, gA));
        svst1_f32(svwhilelt_b32(r + 16, r1), m->y + r + 16, svadd_f32_x(p32, fB, gB));
    }
    if (r < r1) gk_q5_kp16_body(m, r, r1, 16384);
}
