/* Q8_0R16: lossless 16-row panel repack of Q8_0 for decode GEMV (v1).
 *
 * Panel = 16 rows.  For every 32-column block b the panel stores
 *   8 x 64 bytes  q[j][r*4+i] = W[r][32b + 4j + i]   (j = 0..7, r = 0..15)
 *   16 x f32      dw[r]      = row r's Q8_0 scale for block b
 * i.e. 576 bytes per block, exactly 1.125 bytes per weight like Q8_0R.
 *
 * One 32-bit SDOT lane is one output row, so a block is 8 indexed SDOTs
 * against the activation replicated per 128-bit segment (LD1RQB), followed
 * by one SCVTF, FMUL and FMLA for all 16 rows.  Rows land in lanes: no
 * horizontal reduction, and the 16 outputs are one vector store.  The
 * integer products are identical to Q8_0; only the float summation order
 * differs from the v0 row kernel (per-block FMLA instead of per-pair lanes
 * reduced by FADDV). */
#include "glm53f_kern.h"
#include <arm_sve.h>

size_t gk_panel_bytes_q8_0r16(int columns) {
    return (size_t)16 * gk_row_bytes_q8_0r(columns);
}

void gk_pack_q8_0r16(uint8_t *dst, const uint8_t *rows, size_t row_bytes,
                     int nrows, int columns) {
    for (int b = 0; b < columns / 32; ++b) {
        uint8_t *blk = dst + (size_t)b * 576;
        float *dw = (float *)(blk + 512);
        for (int r = 0; r < 16; ++r) {
            const uint8_t *src = rows + (size_t)r * row_bytes;
            float d = 0.0f;
            if (r < nrows) __builtin_memcpy(&d, src + columns + 4 * b, 4);
            dw[r] = d;
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 4; ++i)
                    blk[j * 64 + r * 4 + i] = r < nrows ? src[32 * b + 4 * j + i] : 0;
        }
    }
}

/* One block: 8 indexed SDOTs into a fresh accumulator. */
#define Q8R16_BLOCK(IACC, W, XA, XB) do {                                   \
    IACC = svdot_lane_s32(svdup_s32(0), svld1_s8(p8, (W)), XA, 0);          \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 64), XA, 1);             \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 128), XA, 2);            \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 192), XA, 3);            \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 256), XB, 0);            \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 320), XB, 1);            \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 384), XB, 2);            \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 448), XB, 3);            \
} while (0)

void gk_q8_0r16_v1(const gk_mv *m, int r0, int r1) {
    const int nb = m->columns / 32;
    const size_t pb = gk_panel_bytes_q8_0r16(m->columns);
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const int8_t *xq = m->a->xq;
    const float *xd = m->a->xd;
    for (int r = r0; r < r1; r += 16) {
        const int8_t *w = (const int8_t *)(m->w + (size_t)(r / 16) * pb);
        svfloat32_t f0 = svdup_f32(0.0f), f1 = f0;
        for (int b = 0; b < nb; b += 4, w += 4 * 576) {
            const svfloat32_t xs = svld1rq_f32(p32, xd + b);
            svint32_t i0, i1, i2, i3;
            Q8R16_BLOCK(i0, w, svld1rq_s8(p8, xq + 32 * b), svld1rq_s8(p8, xq + 32 * b + 16));
            Q8R16_BLOCK(i1, w + 576, svld1rq_s8(p8, xq + 32 * b + 32), svld1rq_s8(p8, xq + 32 * b + 48));
            Q8R16_BLOCK(i2, w + 1152, svld1rq_s8(p8, xq + 32 * b + 64), svld1rq_s8(p8, xq + 32 * b + 80));
            Q8R16_BLOCK(i3, w + 1728, svld1rq_s8(p8, xq + 32 * b + 96), svld1rq_s8(p8, xq + 32 * b + 112));
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, i0),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 512)), xs, 0));
            f1 = svmla_f32_x(p32, f1, svcvt_f32_s32_x(p32, i1),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 576 + 512)), xs, 1));
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, i2),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 1152 + 512)), xs, 2));
            f1 = svmla_f32_x(p32, f1, svcvt_f32_s32_x(p32, i3),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 1728 + 512)), xs, 3));
        }
        svst1_f32(svwhilelt_b32(r, r1), m->y + r, svadd_f32_x(p32, f0, f1));
    }
}

/* v2: same layout and integer products as v1, but each block's 8 SDOTs are
 * split into two 4-deep chains (lanes 0-3 of each 16-byte activation half)
 * that are added before the SCVTF.  v1 is latency-bound: one 8-deep chain of
 * 9-cycle SDOTs per block, and the 2x20-entry FP reservation stations cannot
 * hold enough of the next iteration to hide it. */
#define Q8R16_HALF(IACC, W, X) do {                                         \
    IACC = svdot_lane_s32(svdup_s32(0), svld1_s8(p8, (W)), X, 0);           \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 64), X, 1);              \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 128), X, 2);             \
    IACC = svdot_lane_s32(IACC, svld1_s8(p8, (W) + 192), X, 3);             \
} while (0)

void gk_q8_0r16_v2(const gk_mv *m, int r0, int r1) {
    const int nb = m->columns / 32;
    const size_t pb = gk_panel_bytes_q8_0r16(m->columns);
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const int8_t *xq = m->a->xq;
    const float *xd = m->a->xd;
    for (int r = r0; r < r1; r += 16) {
        const int8_t *w = (const int8_t *)(m->w + (size_t)(r / 16) * pb);
        svfloat32_t f0 = svdup_f32(0.0f), f1 = f0;
        for (int b = 0; b < nb; b += 4, w += 4 * 576) {
            const int8_t *x = xq + 32 * b;
            const svfloat32_t xs = svld1rq_f32(p32, xd + b);
            svint32_t a0, b0, a1, b1, a2, b2, a3, b3;
            Q8R16_HALF(a0, w, svld1rq_s8(p8, x));
            Q8R16_HALF(b0, w + 256, svld1rq_s8(p8, x + 16));
            Q8R16_HALF(a1, w + 576, svld1rq_s8(p8, x + 32));
            Q8R16_HALF(b1, w + 576 + 256, svld1rq_s8(p8, x + 48));
            Q8R16_HALF(a2, w + 1152, svld1rq_s8(p8, x + 64));
            Q8R16_HALF(b2, w + 1152 + 256, svld1rq_s8(p8, x + 80));
            Q8R16_HALF(a3, w + 1728, svld1rq_s8(p8, x + 96));
            Q8R16_HALF(b3, w + 1728 + 256, svld1rq_s8(p8, x + 112));
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a0, b0)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 512)), xs, 0));
            f1 = svmla_f32_x(p32, f1, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a1, b1)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 576 + 512)), xs, 1));
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a2, b2)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 1152 + 512)), xs, 2));
            f1 = svmla_f32_x(p32, f1, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a3, b3)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 1728 + 512)), xs, 3));
        }
        svst1_f32(svwhilelt_b32(r, r1), m->y + r, svadd_f32_x(p32, f0, f1));
    }
}

/* v3: v2's two 4-deep chains per block, but with plain (1-uop) SDOT against
 * a 32-bit load-and-replicate (LD1RW) of the activation group instead of the
 * 2-uop indexed SDOT.  The broadcast moves from the FP pipes to the load
 * pipes: per block 8 SDOT + 6 FP ops versus 16 + 6 uops for v1/v2. */
static inline svint8_t gk_bcast4(const int8_t *x) {
    int32_t v;
    __builtin_memcpy(&v, x, 4);
    return svreinterpret_s8_s32(svdup_n_s32(v));
}
#define Q8R16_HALF3(IACC, W, X) do {                                        \
    IACC = svdot_s32(svdup_s32(0), svld1_s8(p8, (W)), gk_bcast4((X)));      \
    IACC = svdot_s32(IACC, svld1_s8(p8, (W) + 64), gk_bcast4((X) + 4));     \
    IACC = svdot_s32(IACC, svld1_s8(p8, (W) + 128), gk_bcast4((X) + 8));    \
    IACC = svdot_s32(IACC, svld1_s8(p8, (W) + 192), gk_bcast4((X) + 12));   \
} while (0)

void gk_q8_0r16_v3(const gk_mv *m, int r0, int r1) {
    const int nb = m->columns / 32;
    const size_t pb = gk_panel_bytes_q8_0r16(m->columns);
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const int8_t *xq = m->a->xq;
    const float *xd = m->a->xd;
    for (int r = r0; r < r1; r += 16) {
        const int8_t *w = (const int8_t *)(m->w + (size_t)(r / 16) * pb);
        svfloat32_t f0 = svdup_f32(0.0f), f1 = f0;
        for (int b = 0; b < nb; b += 4, w += 4 * 576) {
            const int8_t *x = xq + 32 * b;
            const svfloat32_t xs = svld1rq_f32(p32, xd + b);
            svint32_t a0, b0, a1, b1, a2, b2, a3, b3;
            Q8R16_HALF3(a0, w, x);
            Q8R16_HALF3(b0, w + 256, x + 16);
            Q8R16_HALF3(a1, w + 576, x + 32);
            Q8R16_HALF3(b1, w + 576 + 256, x + 48);
            Q8R16_HALF3(a2, w + 1152, x + 64);
            Q8R16_HALF3(b2, w + 1152 + 256, x + 80);
            Q8R16_HALF3(a3, w + 1728, x + 96);
            Q8R16_HALF3(b3, w + 1728 + 256, x + 112);
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a0, b0)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 512)), xs, 0));
            f1 = svmla_f32_x(p32, f1, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a1, b1)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 576 + 512)), xs, 1));
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a2, b2)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 1152 + 512)), xs, 2));
            f1 = svmla_f32_x(p32, f1, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a3, b3)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 1728 + 512)), xs, 3));
        }
        svst1_f32(svwhilelt_b32(r, r1), m->y + r, svadd_f32_x(p32, f0, f1));
    }
}

/* v3 plus explicit L2 prefetch of the weight stream `pf` bytes ahead (one
 * PRFM per 256-byte A64FX line of the current 4-block step).  HBM-streaming
 * decode GEMVs are memory-parallelism bound, not issue bound, once the
 * kernel exceeds a core's share of CMG bandwidth. */
void gk_q8_0r16_v3pf(const gk_mv *m, int r0, int r1, int pf) {
    const int nb = m->columns / 32;
    const size_t pb = gk_panel_bytes_q8_0r16(m->columns);
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const int8_t *xq = m->a->xq;
    const float *xd = m->a->xd;
    for (int r = r0; r < r1; r += 16) {
        const int8_t *w = (const int8_t *)(m->w + (size_t)(r / 16) * pb);
        svfloat32_t f0 = svdup_f32(0.0f), f1 = f0;
        for (int b = 0; b < nb; b += 4, w += 4 * 576) {
            for (int l = 0; l < 4 * 576; l += 256)
                __builtin_prefetch(w + pf + l, 0, 2);
            const int8_t *x = xq + 32 * b;
            const svfloat32_t xs = svld1rq_f32(p32, xd + b);
            svint32_t a0, b0, a1, b1, a2, b2, a3, b3;
            Q8R16_HALF3(a0, w, x);
            Q8R16_HALF3(b0, w + 256, x + 16);
            Q8R16_HALF3(a1, w + 576, x + 32);
            Q8R16_HALF3(b1, w + 576 + 256, x + 48);
            Q8R16_HALF3(a2, w + 1152, x + 64);
            Q8R16_HALF3(b2, w + 1152 + 256, x + 80);
            Q8R16_HALF3(a3, w + 1728, x + 96);
            Q8R16_HALF3(b3, w + 1728 + 256, x + 112);
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a0, b0)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 512)), xs, 0));
            f1 = svmla_f32_x(p32, f1, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a1, b1)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 576 + 512)), xs, 1));
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a2, b2)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 1152 + 512)), xs, 2));
            f1 = svmla_f32_x(p32, f1, svcvt_f32_s32_x(p32, svadd_s32_x(p32, a3, b3)),
                svmul_lane_f32(svld1_f32(p32, (const float *)(w + 1728 + 512)), xs, 3));
        }
        svst1_f32(svwhilelt_b32(r, r1), m->y + r, svadd_f32_x(p32, f0, f1));
    }
}
void gk_q8_0r16_v3pf4k(const gk_mv *m, int r0, int r1) { gk_q8_0r16_v3pf(m, r0, r1, 4096); }
void gk_q8_0r16_v3pf16k(const gk_mv *m, int r0, int r1) { gk_q8_0r16_v3pf(m, r0, r1, 16384); }
void gk_q8_0r16_v3pf64k(const gk_mv *m, int r0, int r1) { gk_q8_0r16_v3pf(m, r0, r1, 65536); }
