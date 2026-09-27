/* Q4_KP16: lossless 16-row panel repack of GGUF Q4_K for decode GEMV (v1).
 *
 * GGUF Q4_K row super-block (256 weights): d, dmin (fp16), 6-bit scale and
 * min for 8 sub-blocks of 32, and 4-bit q.  w = d*sc_j*q - dmin*m_j.
 * Against a Q8_K activation block (dx, x[256]) the dot product is
 *   dx * ( d * sum_j sc_j * (sum_l q*x)  -  dmin * sum_j m_j * (sum_l x) ).
 *
 * Panel layout, per 16 rows and super-block (2432 bytes, 152 per row):
 *   q[j][v][r*4+i], j=0..7, v=0..3 (4 x 64 B per sub-block): the low nibble
 *     is W[r][32j + 8v + i], the high nibble W[r][32j + 8v + 4 + i];
 *   sc[j][r], m[j][r]       uint8 (8 x 16 B each; exact 6-bit values);
 *   d[r], dmin[r]           f32 (16 x 4 B each; exact fp16 values).
 * One SDOT lane is one output row.  Integer products and the per-row
 * sub-block scale/min sums are exact int32 (|sum| < 2^31 for 8-bit x); only
 * the final float combine differs in rounding order from the v0 row kernel.
 * The byte count is 5.6% above Q4_K because scales are stored unpacked. */
#include "glm53f_kern.h"
#include <arm_sve.h>

#define Q4KP16_SB_BYTES 2432

size_t gk_row_bytes_q4_kp16(int columns) { return (size_t)(columns / 256) * 152; }

static inline float gk_h2f(uint16_t h) {
    __fp16 v;
    __builtin_memcpy(&v, &h, 2);
    return (float)v;
}
static inline void gk_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void gk_pack_q4_kp16(uint8_t *dst, const uint8_t *rows, size_t row_bytes,
                     int nrows, int columns) {
    for (int b = 0; b < columns / 256; ++b) {
        uint8_t *sb = dst + (size_t)b * Q4KP16_SB_BYTES;
        uint8_t *sc = sb + 2048, *mn = sb + 2176;
        float *dd = (float *)(sb + 2304), *dm = (float *)(sb + 2368);
        for (int r = 0; r < 16; ++r) {
            const gk_block_q4_K *blk = r < nrows
                ? (const gk_block_q4_K *)(rows + (size_t)r * row_bytes) + b : 0;
            dd[r] = blk ? gk_h2f(blk->d) : 0.0f;
            dm[r] = blk ? gk_h2f(blk->dmin) : 0.0f;
            for (int j = 0; j < 8; ++j) {
                uint8_t s = 0, m = 0;
                if (blk) gk_k4(j, blk->scales, &s, &m);
                sc[j * 16 + r] = s;
                mn[j * 16 + r] = m;
                for (int v = 0; v < 4; ++v)
                    for (int i = 0; i < 4; ++i) {
                        uint8_t lo = 0, hi = 0;
                        if (blk) {
                            /* GGUF: qs[32g + l], low nibble k = 64g + l,
                             * high nibble k = 64g + 32 + l. */
                            const int k0 = 32 * j + 8 * v + i, k1 = k0 + 4;
                            const int g0 = k0 / 64, l0 = k0 % 64, g1 = k1 / 64, l1 = k1 % 64;
                            lo = l0 < 32 ? blk->qs[32 * g0 + l0] & 15 : blk->qs[32 * g0 + l0 - 32] >> 4;
                            hi = l1 < 32 ? blk->qs[32 * g1 + l1] & 15 : blk->qs[32 * g1 + l1 - 32] >> 4;
                        }
                        sb[j * 256 + v * 64 + r * 4 + i] = (uint8_t)(lo | (hi << 4));
                    }
            }
        }
    }
}

static inline svint8_t gk_bc4(const int8_t *x) {
    int32_t v;
    __builtin_memcpy(&v, x, 4);
    return svreinterpret_s8_s32(svdup_n_s32(v));
}

/* One sub-block: 4 nibble vectors -> 8 SDOTs in two 4-deep chains. */
#define Q4KP16_SUB(J) do {                                                       \
    const uint8_t *qj = q + (J) * 256;                                           \
    const int8_t *xj = x + 32 * (J);                                             \
    svint32_t ia = svdup_s32(0), ib = svdup_s32(0);                              \
    for (int v = 0; v < 4; ++v) {                                                \
        const svuint8_t pk = svld1_u8(p8, qj + 64 * v);                          \
        ia = svdot_s32(ia, svreinterpret_s8_u8(svand_n_u8_x(p8, pk, 15)),        \
                       gk_bc4(xj + 8 * v));                                      \
        ib = svdot_s32(ib, svreinterpret_s8_u8(svlsr_n_u8_x(p8, pk, 4)),         \
                       gk_bc4(xj + 8 * v + 4));                                  \
    }                                                                            \
    sacc = svmla_s32_x(p32, sacc, svadd_s32_x(p32, ia, ib),                      \
                       svld1ub_s32(p32, sc + 16 * (J)));                         \
    macc = svmla_s32_x(p32, macc, svld1ub_s32(p32, mn + 16 * (J)),              \
                       svdup_n_s32(bs[J]));                                      \
} while (0)

static inline void gk_q4_kp16_body(const gk_mv *m, int r0, int r1, int pf) {
    const int nsb = m->columns / 256;
    const size_t pb = (size_t)nsb * Q4KP16_SB_BYTES;
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    for (int r = r0; r < r1; r += 16) {
        const uint8_t *panel = m->w + (size_t)(r / 16) * pb;
        svfloat32_t f0 = svdup_f32(0.0f), f1 = f0;
        for (int b = 0; b < nsb; ++b) {
            const uint8_t *q = panel + (size_t)b * Q4KP16_SB_BYTES;
            const uint8_t *sc = q + 2048, *mn = q + 2176;
            if (pf)
                for (int l = 0; l < Q4KP16_SB_BYTES; l += 256)
                    __builtin_prefetch(q + pf + l, 0, 2);
            const int8_t *x = m->a->q8k[b].q;
            const int32_t *bs = m->a->q8k_bsum32 + 8 * b;
            const float dx = m->a->q8k[b].d;
            svint32_t sacc = svdup_s32(0), macc = svdup_s32(0);
            /* A full 8x unroll spills Z registers (~80 STR/LDR Z per
             * super-block); two sub-blocks per iteration fit in registers. */
#pragma clang loop unroll_count(2)
            for (int j = 0; j < 8; ++j) Q4KP16_SUB(j);
            const svfloat32_t dd = svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(q + 2304)), dx);
            const svfloat32_t dm = svmul_n_f32_x(p32, svld1_f32(p32, (const float *)(q + 2368)), dx);
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, sacc), dd);
            f1 = svmls_f32_x(p32, f1, svcvt_f32_s32_x(p32, macc), dm);
        }
        svst1_f32(svwhilelt_b32(r, r1), m->y + r, svadd_f32_x(p32, f0, f1));
    }
}

void gk_q4_kp16_v1(const gk_mv *m, int r0, int r1) { gk_q4_kp16_body(m, r0, r1, 0); }
/* v1 plus L2 software prefetch of the panel stream 16 KiB ahead. */
void gk_q4_kp16_v1pf(const gk_mv *m, int r0, int r1) { gk_q4_kp16_body(m, r0, r1, 16384); }
