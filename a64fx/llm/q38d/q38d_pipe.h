/* Software-pipelined q38d group kernels (A64FX).
 *
 * A pair's dependency path (load, decode, SDOT, combine, SCVTF, FMLA) is
 * about 55 cycles, longer than the in-order-retire window can hide. The
 * loops below are modulo scheduled: one iteration handles two pairs and
 * performs, in program order,
 *   S5 FMLA      for pairs of iteration i-3
 *   S4 combine   for pairs of iteration i-2 (adds, SCVTF)
 *   S3 SDOT      for pairs of iteration i-1
 *   S2 decode    for pairs of iteration i   (and, lsr, tbl; scale)
 *   S1 load      for pairs of iteration i+1
 * Each stage reads values produced one iteration earlier, before the
 * producing stage overwrites them, so one register set per pair slot is
 * enough. Loads may run up to two pairs past the code stream (still inside
 * the group's scale stream); those values are never accumulated. */
#ifndef Q38D_PIPE_H
#define Q38D_PIPE_H
#include "q38d_kern.h"

#ifdef __ARM_FEATURE_SVE
#define Q38D_AI static inline __attribute__((always_inline))

/* Opaque constants: keep AND on the vector form (both FP pipes) and zero
 * initialisation as MOVPRFX (fused), instead of FLA-only immediates/MOVI. */
Q38D_AI svuint8_t q38d_opaque_u8(uint8_t v) {
    svuint8_t r = svdup_n_u8(v); __asm__("" : "+w"(r)); return r;
}
Q38D_AI svint32_t q38d_opaque_zero(void) {
    svint32_t r = svdup_n_s32(0); __asm__("" : "+w"(r)); return r;
}
Q38D_AI svint8_t q38d_r8(const int8_t *p) {
    int64_t v; memcpy(&v, p, 8); return svreinterpret_s8_s64(svdup_n_s64(v));
}
Q38D_AI svfloat32_t q38d_r2(const float *p) {
    uint64_t v; memcpy(&v, p, 8); return svreinterpret_f32_u64(svdup_n_u64(v));
}

/* ---- stage macros; X is the pair-slot suffix (e or o) ---- */
#define P_S1(X, q) do {                                                             \
    z0##X = svld1_u8(pb, code + (q) * 128); z1##X = svld1_u8(pb, code + (q) * 128 + 64); \
    if (F6) { y0##X = svld1_u8(p32, high + (q) * 64); y1##X = svld1_u8(p32, high + (q) * 64 + 32); } \
    if (PF2) svprfb(pb, code + (q) * 128 + PF2, SV_PLDL2KEEP);                       \
} while (0)

#define P_DEC6(Z, Y, L, H) do {                                                      \
    svuint8_t t_ = svzip1_u8(svlsl_n_u8_x(pb, Y, 4), Y);                             \
    L = svtbl_s8(lut, svorr_u8_x(pb, svand_u8_x(pb, Z, m0f), svand_u8_x(pb, t_, m30))); \
    H = svtbl_s8(lut, svorr_u8_x(pb, svlsr_n_u8_x(pb, Z, 4),                         \
                                 svlsr_n_u8_x(pb, svand_u8_x(pb, t_, mc0), 2)));  \
} while (0)

#define P_S2(X, q) do {                                                             \
    if (F6) { P_DEC6(z0##X, y0##X, l0##X, h0##X); P_DEC6(z1##X, y1##X, l1##X, h1##X); } \
    else {                                                                          \
        l0##X = svtbl_s8(lut, svand_u8_x(pb, z0##X, m0f));                          \
        h0##X = svtbl_s8(lut, svlsr_n_u8_x(pb, z0##X, 4));                           \
        l1##X = svtbl_s8(lut, svand_u8_x(pb, z1##X, m0f));                          \
        h1##X = svtbl_s8(lut, svlsr_n_u8_x(pb, z1##X, 4));                           \
    }                                                                               \
} while (0)

#define P_S3(X, q) do {                                                             \
    const int8_t *aq_ = aqb + (q) * AQS;                                             \
    if (!A16) {                                                                     \
        da##X = svdot_s32(zero, l0##X, q38d_r8(aq_));                                \
        db##X = svdot_s32(zero, h0##X, q38d_r8(aq_ + 8));                            \
        dc##X = svdot_s32(zero, l1##X, q38d_r8(aq_ + 16));                           \
        dd##X = svdot_s32(zero, h1##X, q38d_r8(aq_ + 24));                           \
    } else {                                                                        \
        da##X = svdot_s32(svdot_s32(zero, l0##X, q38d_r8(aq_)), l1##X, q38d_r8(aq_ + 16)); \
        db##X = svdot_s32(svdot_s32(zero, h0##X, q38d_r8(aq_ + 8)), h1##X, q38d_r8(aq_ + 24)); \
        dc##X = svdot_s32(svdot_s32(zero, l0##X, q38d_r8(aq_ + 32)), l1##X, q38d_r8(aq_ + 48)); \
        dd##X = svdot_s32(svdot_s32(zero, h0##X, q38d_r8(aq_ + 40)), h1##X, q38d_r8(aq_ + 56)); \
    }                                                                               \
    if (F6) su##X = svld1ub_u32(pl8, sc + (q) * 8); else su##X = svld1ub_u32(pf, sc + (q) * 16); \
} while (0)

#define P_S4(X, q) do {                                                                \
    svint32_t lo_ = svadd_s32_x(pf, da##X, db##X), hi_ = svadd_s32_x(pf, dc##X, dd##X); \
    if (A16) hi_ = svlsl_n_s32_x(pf, hi_, 8);                                        \
    fv##X = svcvt_f32_s32_x(pf, svadd_s32_x(pf, lo_, hi_));                          \
    if (F6) sh##X = svmul_f32_x(pf, svreinterpret_f32_u32(svlsl_n_u32_x(pf,         \
                        svzip1_u32(su##X, su##X), 23)), q38d_r2(asc + 2 * (q)));    \
    else sh##X = svmul_f32_x(pf, svreinterpret_f32_u32(svlsl_n_u32_x(pf, su##X, 20)), \
                             q38d_r2(asc + 2 * (q)));                                \
} while (0)

#define P_S5(X) do { acc##X = svmla_f32_x(pf, acc##X, fv##X, sh##X); } while (0)

#define P_DECL(X)                                                                   \
    svuint8_t z0##X = svdup_n_u8(0), z1##X = z0##X, y0##X = z0##X, y1##X = z0##X;    \
    svuint32_t su##X = svdup_n_u32(0);                                              \
    svint8_t l0##X = svdup_n_s8(0), h0##X = l0##X, l1##X = l0##X, h1##X = l0##X;    \
    svfloat32_t sh##X = svdup_n_f32(0), fv##X = sh##X, acc##X = sh##X; \
    svint32_t da##X = svdup_n_s32(0), db##X = da##X, dc##X = da##X, dd##X = da##X

/* One 8-row group; np must be even and >= 8. */
Q38D_AI void q38d_group_pipe(float *out, const uint8_t *g, const int F6, const int A16,
                             const q38d_act *a, int mode, int nrows, const int PF2) {
    const svbool_t pb = svptrue_b8(), pf = svptrue_b32(), pl8 = svptrue_pat_b32(SV_VL8);
    const svbool_t p32 = svwhilelt_b8((uint32_t)0, (uint32_t)32);
    const size_t np = (size_t)a->cols / 32, AQS = A16 ? 64 : 32;
    const svint8_t lut = svld1_s8(pb, F6 ? q38d_lut_f6 : q38d_lut_f4);
    const svint32_t zero = q38d_opaque_zero();
    const svuint8_t m0f = q38d_opaque_u8(0x0f), m30 = q38d_opaque_u8(0x30), mc0 = q38d_opaque_u8(0xc0);
    const uint8_t *code = g, *high = g + np * 128;
    const uint8_t *sc = g + np * (F6 ? 192 : 128);
    const int8_t *aqb = a->q;
    const float *asc = a->sc;
    P_DECL(e); P_DECL(o);
    const size_t ni = np / 2;
    /* prologue: iterations -3..-1 fill the pipeline */
    P_S1(e, 0); P_S1(o, 1);
    P_S2(e, 0); P_S2(o, 1); P_S1(e, 2); P_S1(o, 3);
    P_S3(e, 0); P_S3(o, 1); P_S2(e, 2); P_S2(o, 3); P_S1(e, 4); P_S1(o, 5);
    P_S4(e, 0); P_S4(o, 1); P_S3(e, 2); P_S3(o, 3); P_S2(e, 4); P_S2(o, 5); P_S1(e, 6); P_S1(o, 7);
    /* steady state: iteration i finishes pairs 2i-6.. and loads 2i+2.. */
    for (size_t i = 4; i < ni; i++) {
        size_t q = 2 * i;
        P_S5(e); P_S5(o);
        P_S4(e, q - 6); P_S4(o, q - 5);
        P_S3(e, q - 4); P_S3(o, q - 3);
        P_S2(e, q - 2); P_S2(o, q - 1);
        P_S1(e, q); P_S1(o, q + 1);
    }
    /* epilogue: drain the last four iterations */
    {
        size_t q = 2 * ni;
        P_S5(e); P_S5(o); P_S4(e, q - 6); P_S4(o, q - 5); P_S3(e, q - 4); P_S3(o, q - 3); P_S2(e, q - 2); P_S2(o, q - 1);
        P_S5(e); P_S5(o); P_S4(e, q - 4); P_S4(o, q - 3); P_S3(e, q - 2); P_S3(o, q - 1);
        P_S5(e); P_S5(o); P_S4(e, q - 2); P_S4(o, q - 1);
        P_S5(e); P_S5(o);
    }
    svfloat32_t acc = svadd_f32_x(pf, acce, acco);
    svfloat32_t r = svadd_f32_x(pf, svuzp1_f32(acc, acc), svuzp2_f32(acc, acc));
    r = svmul_n_f32_x(pf, r, F6 ? 0x1p-67f : 0x1p45f);
    svbool_t p8 = svwhilelt_b32((uint32_t)0, (uint32_t)nrows);
    if (mode) r = svadd_f32_x(p8, r, svld1_f32(p8, out));
    svst1_f32(p8, out, r);
}
#endif
#endif
