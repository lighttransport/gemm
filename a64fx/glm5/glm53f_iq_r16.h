/* Row-interleaved ("R16") repack of native GGUF Q4_K / Q5_K matrices for the
 * A64FX decode matvec (rows in SVE lanes, no horizontal reductions).
 *
 * Layout: rows are grouped in tiles of 16 (rows % 16 == 0). For tile T and
 * 256-column super-block b, one tile-block holds
 *   float d[16], dmin[16]            per-row super-block scales
 *   int8  sc[8][16], mn[8][16]       6-bit sub-block scales/mins, unpacked
 *   uint8 qs[8][4][64]               per 32-column group g and quarter h:
 *                                    lane L (row) byte i = lo nibble col 8h+i,
 *                                    hi nibble col 8h+4+i (i < 4)
 *   uint8 qh[8][64]  (Q5_K only)     lane L byte i: bit 2h = 5th bit of the
 *                                    lo nibble of quarter h, bit 2h+1 of hi
 * Tile-blocks are stored tile-major then block (T * blocks + b).
 *
 * Kernel: per group, indexed SDOT against the activation broadcast in 128-bit
 * segments; integer sums are scaled by sc in int32 across the 8 groups, the
 * mins term is accumulated in int32 from per-group activation sums, and one
 * float conversion per super-block applies d/dmin and the activation scale.
 * Integer arithmetic is exact; only float rounding differs from the GGUF
 * reference (weights and activations are unchanged). */
#ifndef GLM53F_IQ_R16_H
#define GLM53F_IQ_R16_H
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum { GLM53F_R16_ROWS = 16, GLM53F_R16_HEAD = 2 * 16 * 4 + 2 * 8 * 16 };
static inline size_t glm53f_r16_block_bytes(int q5) { return GLM53F_R16_HEAD + 8 * 4 * 64 + (q5 ? 8 * 64 : 0); }
/* Bytes of a repacked rows x (blocks * 256) matrix. */
static inline size_t glm53f_r16_bytes(int rows, int blocks, int q5) {
    return (size_t)(rows / GLM53F_R16_ROWS) * blocks * glm53f_r16_block_bytes(q5);
}

/* Activation for one 256-column super-block: natural-order int8 values,
 * per-32 sums and the shared scale (same quantiser as glm5_iq_quant_q8). */
typedef struct {
    _Alignas(64) int8_t q[256];
    int32_t s[8];
    float d;
} glm53f_r16_act;

static inline float glm53f_r16_half(uint16_t h) { __fp16 v; memcpy(&v, &h, 2); return (float)v; }

/* Decode one GGUF Q4_K (144 B) / Q5_K (176 B) super-block row to values
 * 0..31, 6-bit scales/mins and d/dmin (ggml get_scale_min_k4 order). */
static inline void glm53f_r16_decode(const uint8_t *blk, int q5, uint8_t q[256], uint8_t sc[8], uint8_t mn[8],
                                     float *d, float *dmin) {
    uint16_t h;
    memcpy(&h, blk, 2); *d = glm53f_r16_half(h);
    memcpy(&h, blk + 2, 2); *dmin = glm53f_r16_half(h);
    const uint8_t *s = blk + 4, *qh = blk + 16, *qs = blk + (q5 ? 48 : 16);
    for (int j = 0; j < 8; ++j) {
        if (j < 4) { sc[j] = s[j] & 63; mn[j] = s[j + 4] & 63; }
        else {
            sc[j] = (uint8_t)((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
            mn[j] = (uint8_t)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
        }
    }
    for (int c = 0; c < 4; ++c)          /* 64-column chunks */
        for (int l = 0; l < 32; ++l) {
            uint8_t lo = qs[c * 32 + l] & 15, hi = qs[c * 32 + l] >> 4;
            if (q5) {
                lo |= ((qh[l] >> (2 * c)) & 1) << 4;
                hi |= ((qh[l] >> (2 * c + 1)) & 1) << 4;
            }
            q[c * 64 + l] = lo;
            q[c * 64 + 32 + l] = hi;
        }
}

/* Repack rows x (blocks*256) GGUF rows (row stride rb) into dst (layout above). */
static inline void glm53f_r16_repack(uint8_t *dst, const uint8_t *src, size_t rb, int rows, int blocks, int q5) {
    const size_t bsz = q5 ? 176 : 144, tb = glm53f_r16_block_bytes(q5);
    for (int t = 0; t < rows / GLM53F_R16_ROWS; ++t)
        for (int b = 0; b < blocks; ++b) {
            uint8_t *o = dst + ((size_t)t * blocks + b) * tb;
            float *d = (float *)o, *dm = d + 16;
            int8_t *sc = (int8_t *)(o + 128), *mn = sc + 128;
            uint8_t *qs = o + GLM53F_R16_HEAD, *qh = qs + 8 * 4 * 64;
            if (q5) memset(qh, 0, 8 * 64);
            for (int L = 0; L < 16; ++L) {
                uint8_t q[256], s[8], m[8];
                glm53f_r16_decode(src + (size_t)(t * 16 + L) * rb + (size_t)b * bsz, q5, q, s, m, d + L, dm + L);
                for (int g = 0; g < 8; ++g) {
                    sc[g * 16 + L] = (int8_t)s[g]; mn[g * 16 + L] = (int8_t)m[g];
                    for (int hq = 0; hq < 4; ++hq)
                        for (int i = 0; i < 4; ++i) {
                            const uint8_t lo = q[g * 32 + 8 * hq + i], hi = q[g * 32 + 8 * hq + 4 + i];
                            qs[(g * 4 + hq) * 64 + 4 * L + i] = (uint8_t)((lo & 15) | (hi << 4));
                            if (q5) qh[g * 64 + 4 * L + i] |= (uint8_t)(((lo >> 4) << (2 * hq)) | ((hi >> 4) << (2 * hq + 1)));
                        }
                }
            }
        }
}

/* Scalar model of the kernel's arithmetic (for tests): exact integer sums,
 * then the same float operation order as glm53f_r16_tile. */
static inline void glm53f_r16_tile_ref(float out[16], const uint8_t *tile, int blocks, int q5, const glm53f_r16_act *a) {
    const size_t tb = glm53f_r16_block_bytes(q5);
    float acc[16] = {0};
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *o = tile + (size_t)b * tb;
        const float *d = (const float *)o, *dm = d + 16;
        const int8_t *sc = (const int8_t *)(o + 128), *mn = sc + 128;
        const uint8_t *qs = o + GLM53F_R16_HEAD, *qh = qs + 8 * 4 * 64;
        for (int L = 0; L < 16; ++L) {
            int32_t tot = 0, mins = 0;
            for (int g = 0; g < 8; ++g) {
                int32_t dot = 0;
                for (int hq = 0; hq < 4; ++hq)
                    for (int i = 0; i < 4; ++i) {
                        const uint8_t byte = qs[(g * 4 + hq) * 64 + 4 * L + i];
                        int lo = byte & 15, hi = byte >> 4;
                        if (q5) {
                            const uint8_t hb = qh[g * 64 + 4 * L + i];
                            lo |= ((hb >> (2 * hq)) & 1) << 4; hi |= ((hb >> (2 * hq + 1)) & 1) << 4;
                        }
                        dot += lo * a[b].q[g * 32 + 8 * hq + i] + hi * a[b].q[g * 32 + 8 * hq + 4 + i];
                    }
                tot += dot * sc[g * 16 + L];
                mins += mn[g * 16 + L] * a[b].s[g];
            }
            acc[L] = acc[L] + d[L] * a[b].d * (float)tot;
            acc[L] = acc[L] - dm[L] * a[b].d * (float)mins;
        }
    }
    memcpy(out, acc, sizeof(acc));
}

/* One 16-row tile over `blocks` super-blocks; out[0..15]. SVE vector length 512. */
static inline __attribute__((always_inline)) void glm53f_r16_tile(float *out, const uint8_t *tile, int blocks, int q5,
                                                                  const glm53f_r16_act *a) {
    const size_t tb = glm53f_r16_block_bytes(q5);
    const svbool_t p32 = svptrue_b32(), p8 = svptrue_b8();
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *o = tile + (size_t)b * tb;
        const int8_t *sc = (const int8_t *)(o + 128), *mn = sc + 128;
        const uint8_t *qs = o + GLM53F_R16_HEAD, *qh = qs + 8 * 4 * 64;
        const int8_t *xq = a[b].q;
        svint32_t tot = svdup_s32(0), mins = svdup_s32(0);
        for (int g = 0; g < 8; ++g) {
            const svint8_t xa = svld1rq_s8(p8, xq + g * 32), xb = svld1rq_s8(p8, xq + g * 32 + 16);
            const uint8_t *v = qs + g * 4 * 64;
            svuint8_t w0 = svld1_u8(p8, v), w1 = svld1_u8(p8, v + 64), w2 = svld1_u8(p8, v + 128), w3 = svld1_u8(p8, v + 192);
            svuint8_t l0 = svand_n_u8_x(p8, w0, 15), h0 = svlsr_n_u8_x(p8, w0, 4);
            svuint8_t l1 = svand_n_u8_x(p8, w1, 15), h1 = svlsr_n_u8_x(p8, w1, 4);
            svuint8_t l2 = svand_n_u8_x(p8, w2, 15), h2 = svlsr_n_u8_x(p8, w2, 4);
            svuint8_t l3 = svand_n_u8_x(p8, w3, 15), h3 = svlsr_n_u8_x(p8, w3, 4);
            if (q5) {
                const svuint8_t hb = svld1_u8(p8, qh + g * 64);
/* bit s of hb moves to bit 4 (value 16) of the s-th nibble vector */
                const svuint8_t b0 = svand_n_u8_x(p8, svlsl_n_u8_x(p8, hb, 4), 16), b1 = svand_n_u8_x(p8, svlsl_n_u8_x(p8, hb, 3), 16);
                const svuint8_t b2 = svand_n_u8_x(p8, svlsl_n_u8_x(p8, hb, 2), 16), b3 = svand_n_u8_x(p8, svlsl_n_u8_x(p8, hb, 1), 16);
                const svuint8_t b4 = svand_n_u8_x(p8, hb, 16), b5 = svand_n_u8_x(p8, svlsr_n_u8_x(p8, hb, 1), 16);
                const svuint8_t b6 = svand_n_u8_x(p8, svlsr_n_u8_x(p8, hb, 2), 16), b7 = svand_n_u8_x(p8, svlsr_n_u8_x(p8, hb, 3), 16);
                l0 = svorr_u8_x(p8, l0, b0); h0 = svorr_u8_x(p8, h0, b1); l1 = svorr_u8_x(p8, l1, b2); h1 = svorr_u8_x(p8, h1, b3);
                l2 = svorr_u8_x(p8, l2, b4); h2 = svorr_u8_x(p8, h2, b5); l3 = svorr_u8_x(p8, l3, b6); h3 = svorr_u8_x(p8, h3, b7);
            }
            svint32_t dot = svdup_s32(0);
            dot = svdot_lane_s32(dot, svreinterpret_s8_u8(l0), xa, 0);
            dot = svdot_lane_s32(dot, svreinterpret_s8_u8(h0), xa, 1);
            dot = svdot_lane_s32(dot, svreinterpret_s8_u8(l1), xa, 2);
            dot = svdot_lane_s32(dot, svreinterpret_s8_u8(h1), xa, 3);
            dot = svdot_lane_s32(dot, svreinterpret_s8_u8(l2), xb, 0);
            dot = svdot_lane_s32(dot, svreinterpret_s8_u8(h2), xb, 1);
            dot = svdot_lane_s32(dot, svreinterpret_s8_u8(l3), xb, 2);
            dot = svdot_lane_s32(dot, svreinterpret_s8_u8(h3), xb, 3);
            tot = svmla_s32_x(p32, tot, dot, svld1sb_s32(p32, sc + g * 16));
            mins = svmla_n_s32_x(p32, mins, svld1sb_s32(p32, mn + g * 16), a[b].s[g]);
        }
        const svfloat32_t d = svld1_f32(p32, (const float *)o), dm = svld1_f32(p32, (const float *)o + 16);
        acc = svmla_f32_x(p32, acc, svmul_n_f32_x(p32, d, a[b].d), svcvt_f32_s32_x(p32, tot));
        acc = svmls_f32_x(p32, acc, svmul_n_f32_x(p32, dm, a[b].d), svcvt_f32_s32_x(p32, mins));
    }
    svst1_f32(p32, out, acc);
}

/* out[t*16 .. t*16+15] for tiles [t0, t1) of a repacked matrix. */
static inline void glm53f_r16_tiles(float *out, const uint8_t *w, int t0, int t1, int blocks, int q5,
                                    const glm53f_r16_act *a) {
    const size_t stride = (size_t)blocks * glm53f_r16_block_bytes(q5);
    for (int t = t0; t < t1; ++t) glm53f_r16_tile(out + (size_t)t * 16, w + (size_t)t * stride, blocks, q5, a);
}

/* Quantize 256-column blocks of x (glm5_iq_quant_q8 rule: d = amax/127,
 * round-to-nearest) and record per-32 sums. */
static inline void glm53f_r16_quant(glm53f_r16_act *a, const float *x, int blocks) {
    for (int b = 0; b < blocks; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 256; ++i) { const float v = x[b * 256 + i] < 0 ? -x[b * 256 + i] : x[b * 256 + i]; if (v > amax) amax = v; }
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        for (int i = 0; i < 256; ++i) {
            float v = x[b * 256 + i] * id;
            int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            a[b].q[i] = (int8_t)(q > 127 ? 127 : q < -127 ? -127 : q);
        }
        for (int g = 0; g < 8; ++g) { int32_t s = 0; for (int i = 0; i < 32; ++i) s += a[b].q[g * 32 + i]; a[b].s[g] = s; }
        a[b].d = d;
    }
}
#endif
