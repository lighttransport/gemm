/* Native Q4_K/Q5_K scale/minimum extraction. GPR masks combine four bytes at a time;
 * two SVE unpack operations widen each packed scale/minimum vector. No sidecar
 * is read and SDOT/FMA/reduction order remains unchanged. */
#ifndef GLM53F_IQ_SCALE_WORDS_H
#define GLM53F_IQ_SCALE_WORDS_H
#include "glm53f_iq_fast.h"
#include <string.h>
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Native GGUF scale words require little-endian A64FX"
#endif
/* Like iqf_rows:512-bit SVE, with iqf_init() completed before dispatch. */
#define IQFW_DECL_K \
    const svbool_t p32 = svptrue_b32(), p8 = svptrue_b8(), pg8 = svwhilelt_b32(0, 8), pb32 = svwhilelt_b8(0, 32); \
    const svuint32_t t0 = svld1_u32(p32, iqf_sel0), t1 = svld1_u32(p32, iqf_sel1), \
        t2 = svld1_u32(p32, iqf_sel2), t3 = svld1_u32(p32, iqf_sel3); \
    const svuint8_t vrep = svld1_u8(p8, iqf_rep), vs0 = svld1_u8(p8, iqf_sh0), vs1 = svld1_u8(p8, iqf_sh1), \
        vs2 = svld1_u8(p8, iqf_sh2), vs3 = svld1_u8(p8, iqf_sh3)
static inline __attribute__((always_inline)) svfloat32_t iqfw_block(svfloat32_t acc,
        const uint8_t *blk, const iqf_act *ab, int q5,
        svbool_t p32, svbool_t p8, svbool_t pg8, svbool_t pb32,
        svuint32_t t0, svuint32_t t1, svuint32_t t2, svuint32_t t3,
        svuint8_t vrep, svuint8_t vs0, svuint8_t vs1, svuint8_t vs2, svuint8_t vs3) {
    __builtin_prefetch(blk + IQF_PF_BYTES, 0, IQF_PF_LVL);
    uint32_t s0, s1, s2;
    memcpy(&s0, blk + 4, 4); memcpy(&s1, blk + 8, 4); memcpy(&s2, blk + 12, 4);
    uint32_t sc0 = s0 & 0x3f3f3f3fu;
    uint32_t sc1 = (s2 & 0x0f0f0f0fu) | ((s0 >> 2) & 0x30303030u);
    uint32_t mn0 = s1 & 0x3f3f3f3fu;
    uint32_t mn1 = ((s2 >> 4) & 0x0f0f0f0fu) | ((s1 >> 2) & 0x30303030u);
    uint64_t scale_bytes = (uint64_t)sc0 | ((uint64_t)sc1 << 32);
    uint64_t minimum_bytes = (uint64_t)mn0 | ((uint64_t)mn1 << 32);
    svuint32_t sc = svunpklo_u32(svunpklo_u16(svreinterpret_u8_u64(svdup_u64(scale_bytes))));
    svuint32_t mn = svunpklo_u32(svunpklo_u16(svreinterpret_u8_u64(svdup_u64(minimum_bytes))));
    const uint8_t *qs = blk + (q5 ? 48 : 16);
    const svuint8_t qa = svld1_u8(p8, qs), qb = svld1_u8(p8, qs + 64);
    svuint8_t l0 = svand_n_u8_x(p8, qa, 15), h0 = svlsr_n_u8_x(p8, qa, 4);
    svuint8_t l1 = svand_n_u8_x(p8, qb, 15), h1 = svlsr_n_u8_x(p8, qb, 4);
    if (q5) {
        const svuint8_t hv = svtbl_u8(svld1_u8(pb32, blk + 16), vrep);
        l0 = svadd_u8_x(p8, l0, svlsl_n_u8_x(p8, svand_n_u8_x(p8, svlsr_u8_x(p8, hv, vs0), 1), 4));
        h0 = svadd_u8_x(p8, h0, svlsl_n_u8_x(p8, svand_n_u8_x(p8, svlsr_u8_x(p8, hv, vs1), 1), 4));
        l1 = svadd_u8_x(p8, l1, svlsl_n_u8_x(p8, svand_n_u8_x(p8, svlsr_u8_x(p8, hv, vs2), 1), 4));
        h1 = svadd_u8_x(p8, h1, svlsl_n_u8_x(p8, svand_n_u8_x(p8, svlsr_u8_x(p8, hv, vs3), 1), 4));
    }
    return iqf_accumulate(acc, ab, (float)*(const __fp16 *)blk, (float)*(const __fp16 *)(blk + 2), sc, mn, l0, h0, l1, h1, p32, p8, pg8, t0, t1, t2, t3);
}
#define IQFW_CALL(acc, blk, ab) iqfw_block(acc, blk, ab, q5, p32, p8, pg8, pb32, t0, t1, t2, t3, vrep, vs0, vs1, vs2, vs3)
static inline void iqfw_rows(float *out, const uint8_t *row0, size_t rb, int rows,
        const iqf_act *activation, int blocks, int q5) {
    IQFW_DECL_K;
    const size_t bsz = q5 ? 176 : 144;
    int r = 0;
    if (blocks == 1)
        for (; r + 3 < rows; r += 4) {
            const uint8_t *w = row0 + (size_t)r * rb;
            svfloat32_t a0 = IQFW_CALL(svdup_f32(0), w, activation);
            svfloat32_t a1 = IQFW_CALL(svdup_f32(0), w + rb, activation);
            svfloat32_t a2 = IQFW_CALL(svdup_f32(0), w + 2 * rb, activation);
            svfloat32_t a3 = IQFW_CALL(svdup_f32(0), w + 3 * rb, activation);
            iqf_reduce_four(out + r, a0, a1, a2, a3);
        }
    for (; r + 1 < rows; r += 2) {
        const uint8_t *wa = row0 + (size_t)r * rb, *wb = wa + rb;
        svfloat32_t aa = svdup_f32(0), ab = aa;
        for (int b = 0; b < blocks; ++b) {
            aa = IQFW_CALL(aa, wa + (size_t)b * bsz, activation + b);
            ab = IQFW_CALL(ab, wb + (size_t)b * bsz, activation + b);
        }
        out[r] = svaddv_f32(p32, aa); out[r + 1] = svaddv_f32(p32, ab);
    }
    for (; r < rows; ++r) {
        svfloat32_t a = svdup_f32(0);
        for (int b = 0; b < blocks; ++b)
            a = IQFW_CALL(a, row0 + (size_t)r * rb + (size_t)b * bsz,
                        activation + b);
        out[r] = svaddv_f32(p32, a);
    }
}
#undef IQFW_CALL
#undef IQFW_DECL_K
#endif
