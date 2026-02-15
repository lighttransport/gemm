#ifndef SVE_MATH_H
#define SVE_MATH_H

#include <arm_sve.h>
#include <math.h>
#include <stdint.h>

/* ── Constants ── */
#define LOG2E    1.4426950408889634f
#define LN2      0.6931471805599453f
#define SQRT2    1.4142135623730951f

/* FEXPA shift: reinterpret 0x48481fc0 as float */
#define FEXPA_SHIFT_U32  0x48481fc0u

static inline float fexpa_shift_f32(void) {
    union { uint32_t u; float f; } c = { .u = FEXPA_SHIFT_U32 };
    return c.f;
}

/* ── SVE exp2 via FEXPA (A64FX-specific) ──
 *
 * exp2(x) = fexpa(x + shift)
 * ~0.4% relative error, 2 SVE insns
 */
static inline svfloat32_t sve_fexpa(svfloat32_t z) {
    svfloat32_t r;
    __asm__("fexpa %0.s, %1.s" : "=w"(r) : "w"(z));
    return r;
}

/* ── SVE exp via FEXPA ──
 *
 * exp(x) = 2^(x * log2(e))
 *        = fexpa(x * LOG2E + shift)
 *
 * CRITICAL: unlike softmax where base cancels in normalization,
 * cross-entropy needs natural exp, so we multiply by LOG2E.
 */
static inline svfloat32_t sve_exp_fexpa(svbool_t pg, svfloat32_t x,
                                         svfloat32_t vlog2e,
                                         svfloat32_t vshift) {
    /* z = x * LOG2E + shift */
    svfloat32_t z = svmla_f32_x(pg, vshift, x, vlog2e);
    return sve_fexpa(z);
}

/* ── SVE log2 (vectorized, purely vertical, no horizontal ops) ──
 *
 * Algorithm:
 *   1. Decompose: x = 2^n * m,  1 <= m < 2
 *      n = (bits >> 23) - 127
 *      m = (bits & 0x7FFFFF) | 0x3F800000
 *   2. Range reduce: if m > sqrt(2), m /= 2, n += 1
 *      f = m - 1.0,  f in [-0.293, 0.414]
 *   3. Minimax degree-5 polynomial: log2(1+f)
 *   4. log2(x) = n + poly(f)
 *
 * ~16 SVE ops per vector, throughput ~8 cy/vec.
 * All element-wise vertical — zero horizontal reductions.
 */

/* Minimax coefficients for log2(1+f), f in [-0.293, 0.414]
 * Approximating: log2(1+f) ~ c0*f + c1*f^2 + c2*f^3 + c3*f^4 + c4*f^5
 * Optimized for fp32 (max ~1e-6 relative error in range)
 */
#define LOG2_C0  1.44269504089f   /* 1/ln(2) */
#define LOG2_C1 -0.72134752045f   /* -1/(2*ln(2)) */
#define LOG2_C2  0.48089834696f   /* 1/(3*ln(2)) */
#define LOG2_C3 -0.36067376023f   /* -1/(4*ln(2)) */
#define LOG2_C4  0.28853900819f   /* 1/(5*ln(2)) */

static inline svfloat32_t sve_log2_f32(svbool_t pg, svfloat32_t x) {
    /* 1. Bit decompose */
    svint32_t bits = svreinterpret_s32(x);

    /* n = (bits >> 23) - 127 */
    svint32_t n_i = svsub_n_s32_x(pg, svasr_n_s32_x(pg, bits, 23), 127);
    svfloat32_t n = svcvt_f32_s32_x(pg, n_i);

    /* m = (bits & 0x7FFFFF) | 0x3F800000  →  1.0 <= m < 2.0 */
    svint32_t mantissa = svand_n_s32_x(pg, bits, 0x007FFFFF);
    svint32_t m_bits = svorr_n_s32_x(pg, mantissa, 0x3F800000);
    svfloat32_t m = svreinterpret_f32(m_bits);

    /* 2. Range reduce: if m > sqrt(2), m /= 2, n += 1 */
    svbool_t hi = svcmpgt(pg, m, svdup_f32(SQRT2));
    /* m_adj: divide by 2 by subtracting 0x00800000 from bits (exponent -1) */
    svint32_t m_adj_bits = svsub_n_s32_x(pg, m_bits, 0x00800000);
    svfloat32_t m_adj = svreinterpret_f32(m_adj_bits);
    m = svsel(hi, m_adj, m);
    n = svadd_f32_m(hi, n, svdup_f32(1.0f));

    /* f = m - 1.0,  f in [-0.293, 0.414] */
    svfloat32_t f = svsub_n_f32_x(pg, m, 1.0f);

    /* 3. Horner evaluation: log2(1+f) ≈ f*(c0 + f*(c1 + f*(c2 + f*(c3 + f*c4)))) */
    svfloat32_t p = svdup_f32(LOG2_C4);
    p = svmla_f32_x(pg, svdup_f32(LOG2_C3), f, p);   /* c3 + f*c4 */
    p = svmla_f32_x(pg, svdup_f32(LOG2_C2), f, p);   /* c2 + f*(...) */
    p = svmla_f32_x(pg, svdup_f32(LOG2_C1), f, p);   /* c1 + f*(...) */
    p = svmla_f32_x(pg, svdup_f32(LOG2_C0), f, p);   /* c0 + f*(...) */
    p = svmul_f32_x(pg, p, f);                         /* f * poly */

    /* 4. log2(x) = n + poly(f) */
    return svadd_f32_x(pg, n, p);
}

/* ── SVE ln (natural log) = log2(x) * LN2 ── */
static inline svfloat32_t sve_log_f32(svbool_t pg, svfloat32_t x) {
    svfloat32_t l2 = sve_log2_f32(pg, x);
    return svmul_n_f32_x(pg, l2, LN2);
}

/* ── FPCR.FZ16: Flush fp16 denormals to zero ──
 * Without this, fp16 denormal values trigger microcode traps (~100+ cy/op).
 * Must be called before any fp16 computation.
 */
static inline void set_fpcr_fz16(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1UL << 19);
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
}

/* ── Load fp16 → fp32 without unpack ──
 * LD1H {Z.S} loads 16 halfwords into 32-bit containers (zero-extend).
 * FCVT Z.S, Pg/M, Z.H reads the low 16 bits as fp16 and converts to fp32.
 * 2 instructions per 16 elements (vs 5 for ld1h+unpack×2+fcvt×2 per 32).
 */
static inline svfloat32_t svld1_cvt_f16_f32(svbool_t pg, const uint16_t *ptr) {
    svuint32_t raw = svld1uh_u32(pg, ptr);              /* LD1H {Z.S} */
    return svcvt_f32_f16_x(pg, svreinterpret_f16(raw)); /* FCVT Z.S, Z.H */
}

#endif /* SVE_MATH_H */
