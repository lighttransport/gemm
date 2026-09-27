/* Q6_KP16: lossless 16-row panel repack of GGUF Q6_K for decode GEMV (v1).
 *
 * GGUF Q6_K super-block (256 weights, 210 bytes): 4 low bits in ql, 2 high
 * bits in qh, int8 scale per 16 weights, fp16 d.  w = d * sc[k/16] * (q-32),
 * q in 0..63 (the layout below is decoded by q6_value()).
 *
 * Panel per 16 rows and super-block (3392 bytes, 212 per row, +1%):
 *   for each 16-weight sub-block s (208 bytes):
 *     2 x 64 B nibbles: vector v, byte r*4+i: low nibble = W[r][16s + 8v + i],
 *                       high nibble = W[r][16s + 8v + 4 + i]  (k-groups 2v, 2v+1)
 *     1 x 64 B high bits: bits 2g..2g+1 of byte r*4+i = q[r][16s + 4g + i] >> 4
 *     16 B int8 sc[s][r]
 *   then 16 x f32 d[r].
 * sum (q-32)*x = sum nib*x + 16 * sum h*x - 32 * sum x: exact int32 SDOTs,
 * with per-16 activation sums supplied in gk_act.q8k_bsum16. */
#include "glm53f_kern.h"
#include <arm_sve.h>

#define Q6KP16_SUB_BYTES 208
#define Q6KP16_SB_BYTES (16 * Q6KP16_SUB_BYTES + 64)

size_t gk_row_bytes_q6_kp16(int columns) { return (size_t)(columns / 256) * 212; }

static inline float gk_h2f6(uint16_t h) {
    __fp16 v;
    __builtin_memcpy(&v, &h, 2);
    return (float)v;
}
/* 6-bit value (0..63) of weight k (0..255) of a GGUF Q6_K block. */
static inline int q6_value(const gk_block_q6_K *b, int k) {
    const int n = k / 128, rem = k % 128, jq = rem / 32, l = rem % 32;
    const uint8_t *ql = b->ql + 64 * n, *qh = b->qh + 32 * n;
    const int lo = (jq & 1) ? ql[l + 32] : ql[l];
    const int nib = (jq & 2) ? lo >> 4 : lo & 15;
    return nib | (((qh[l] >> (2 * jq)) & 3) << 4);
}

void gk_pack_q6_kp16(uint8_t *dst, const uint8_t *rows, size_t row_bytes,
                     int nrows, int columns) {
    for (int b = 0; b < columns / 256; ++b) {
        uint8_t *sb = dst + (size_t)b * Q6KP16_SB_BYTES;
        float *dd = (float *)(sb + 16 * Q6KP16_SUB_BYTES);
        for (int r = 0; r < 16; ++r) {
            const gk_block_q6_K *blk = r < nrows
                ? (const gk_block_q6_K *)(rows + (size_t)r * row_bytes) + b : 0;
            dd[r] = blk ? gk_h2f6(blk->d) : 0.0f;
            for (int s = 0; s < 16; ++s) {
                uint8_t *sub = sb + s * Q6KP16_SUB_BYTES;
                sub[192 + r] = blk ? (uint8_t)blk->scales[s] : 0;
                for (int i = 0; i < 4; ++i) {
                    uint8_t hbits = 0;
                    for (int g = 0; g < 4; ++g) {
                        const int q = blk ? q6_value(blk, 16 * s + 4 * g + i) : 0;
                        hbits |= (uint8_t)((q >> 4) << (2 * g));
                    }
                    sub[128 + r * 4 + i] = hbits;
                    for (int v = 0; v < 2; ++v) {
                        const int lo = blk ? q6_value(blk, 16 * s + 8 * v + i) & 15 : 0;
                        const int hi = blk ? q6_value(blk, 16 * s + 8 * v + 4 + i) & 15 : 0;
                        sub[v * 64 + r * 4 + i] = (uint8_t)(lo | (hi << 4));
                    }
                }
            }
        }
    }
}

static inline svint8_t gk_bc4_6(const int8_t *x) {
    int32_t v;
    __builtin_memcpy(&v, x, 4);
    return svreinterpret_s8_s32(svdup_n_s32(v));
}

#define Q6KP16_SUB(S) do {                                                        \
    const uint8_t *sub = q + (S) * Q6KP16_SUB_BYTES;                              \
    const svuint8_t p0 = svld1_u8(p8, sub), p1 = svld1_u8(p8, sub + 64);          \
    const svuint8_t hq = svld1_u8(p8, sub + 128);                                 \
    const int8_t *xs = x + 16 * (S);                                              \
    const svint8_t x0 = gk_bc4_6(xs), x1 = gk_bc4_6(xs + 4);                      \
    const svint8_t x2 = gk_bc4_6(xs + 8), x3 = gk_bc4_6(xs + 12);                 \
    svint32_t ia = svdot_s32(svdup_s32(0),                                        \
                             svreinterpret_s8_u8(svand_n_u8_x(p8, p0, 15)), x0);  \
    svint32_t ib = svdot_s32(svdup_s32(0),                                        \
                             svreinterpret_s8_u8(svlsr_n_u8_x(p8, p0, 4)), x1);   \
    ia = svdot_s32(ia, svreinterpret_s8_u8(svand_n_u8_x(p8, p1, 15)), x2);        \
    ib = svdot_s32(ib, svreinterpret_s8_u8(svlsr_n_u8_x(p8, p1, 4)), x3);         \
    svint32_t ha = svdot_s32(svdup_s32(0),                                        \
                             svreinterpret_s8_u8(svand_n_u8_x(p8, hq, 3)), x0);   \
    svint32_t hb = svdot_s32(svdup_s32(0), svreinterpret_s8_u8(                   \
                             svand_n_u8_x(p8, svlsr_n_u8_x(p8, hq, 2), 3)), x1);  \
    ha = svdot_s32(ha, svreinterpret_s8_u8(                                       \
                   svand_n_u8_x(p8, svlsr_n_u8_x(p8, hq, 4), 3)), x2);            \
    hb = svdot_s32(hb, svreinterpret_s8_u8(svlsr_n_u8_x(p8, hq, 6)), x3);         \
    const svint32_t tot = svadd_s32_x(p32, svadd_s32_x(p32, ia, ib),              \
        svlsl_n_s32_x(p32, svadd_s32_x(p32, ha, hb), 4));                         \
    const svint32_t scv = svld1sb_s32(p32, (const int8_t *)sub + 192);            \
    sacc = svmla_s32_x(p32, sacc, tot, scv);                                      \
    oacc = svmla_s32_x(p32, oacc, scv, svdup_n_s32(bs[S]));                       \
} while (0)

static inline void gk_q6_kp16_body(const gk_mv *m, int r0, int r1, int pf) {
    const int nsb = m->columns / 256;
    const size_t pb = (size_t)nsb * Q6KP16_SB_BYTES;
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    for (int r = r0; r < r1; r += 16) {
        const uint8_t *panel = m->w + (size_t)(r / 16) * pb;
        svfloat32_t f0 = svdup_f32(0.0f);
        for (int b = 0; b < nsb; ++b) {
            const uint8_t *q = panel + (size_t)b * Q6KP16_SB_BYTES;
            if (pf)
                for (int l = 0; l < Q6KP16_SB_BYTES; l += 256)
                    __builtin_prefetch(q + pf + l, 0, 2);
            const int8_t *x = m->a->q8k[b].q;
            const int32_t *bs = m->a->q8k_bsum16 + 16 * b;
            svint32_t sacc = svdup_s32(0), oacc = svdup_s32(0);
#pragma clang loop unroll_count(2)
            for (int s = 0; s < 16; ++s) Q6KP16_SUB(s);
            /* sum sc*(q-32)*x = sacc - 32*oacc, exact in int32. */
            const svint32_t tot = svsub_s32_x(p32, sacc, svlsl_n_s32_x(p32, oacc, 5));
            const svfloat32_t dd = svmul_n_f32_x(p32,
                svld1_f32(p32, (const float *)(q + 16 * Q6KP16_SUB_BYTES)), m->a->q8k[b].d);
            f0 = svmla_f32_x(p32, f0, svcvt_f32_s32_x(p32, tot), dd);
        }
        svst1_f32(svwhilelt_b32(r, r1), m->y + r, f0);
    }
}

void gk_q6_kp16_v1(const gk_mv *m, int r0, int r1) { gk_q6_kp16_body(m, r0, r1, 0); }
void gk_q6_kp16_v1pf(const gk_mv *m, int r0, int r1) { gk_q6_kp16_body(m, r0, r1, 16384); }
