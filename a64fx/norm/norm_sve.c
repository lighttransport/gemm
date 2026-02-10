#include "norm_sve.h"
#include "sve_math.h"
#include <arm_sve.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Prefetch distance in cache lines (256 bytes each on A64FX).
 * 8 lines = 2KB ahead. */
#ifndef PF_DIST_LINES
#define PF_DIST_LINES 8
#endif
#define PF_DIST_BYTES (PF_DIST_LINES * 256)

static inline void prefetch_l1(const void *base, int byte_offset) {
    __builtin_prefetch((const char *)base + byte_offset, 0, 3);
}

/* ── SVE rsqrt: FRSQRTE + 2x Newton-Raphson ──
 * Newton-Raphson step: x_{n+1} = x_n * (3 - d * x_n^2) / 2
 * After 2 NR steps: ~1e-7 relative error (full FP32 precision).
 */
static inline svfloat32_t sve_rsqrt_f32(svbool_t pg, svfloat32_t d) {
    svfloat32_t x = svrsqrte_f32(d);                          /* FRSQRTE ~8-bit */
    /* NR step 1 */
    svfloat32_t dx2 = svmul_f32_x(pg, d, x);                 /* d * x */
    svfloat32_t step1 = svrsqrts_f32(dx2, x);                 /* (3 - d*x*x) / 2 */
    x = svmul_f32_x(pg, x, step1);
    /* NR step 2 */
    dx2 = svmul_f32_x(pg, d, x);
    svfloat32_t step2 = svrsqrts_f32(dx2, x);
    x = svmul_f32_x(pg, x, step2);
    return x;
}

/* ── FP32 → FP16 store ── */
static inline void svst1_cvt_f32_f16(svbool_t pg, uint16_t *ptr, svfloat32_t v) {
    svfloat16_t h = svcvt_f16_f32_x(pg, v);                   /* FCVT Z.H, Z.S */
    svst1h_u32(pg, ptr, svreinterpret_u32(h));                 /* ST1H {Z.S} */
}

/* ── Horizontal sum via store + scalar loop ── */
static inline float hsum_f32(svbool_t pg, svfloat32_t v) {
    float buf[16] __attribute__((aligned(256)));
    svst1_f32(pg, buf, v);
    int VL = (int)svcntw();
    float s = 0.0f;
    for (int k = 0; k < VL; k++)
        s += buf[k];
    return s;
}


/* ════════════════════════════════════════════════════════════════
 * RMSNorm Forward FP32
 *
 * Pass 1: sum_sq = sum(x[i]^2) — 8x unrolled FMLA
 * Compute: inv_rms = 1/sqrt(sum_sq/N + eps) via FRSQRTE + 2 NR
 * Pass 2: y[i] = gamma[i] * x[i] * inv_rms — 8x unrolled
 * ════════════════════════════════════════════════════════════════ */

void rmsnorm_fwd_f32(const float *x, float *y, const float *gamma,
                     float eps, int N) {
    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int N8 = N & ~(8 * VL - 1);
    const int N1 = N & ~(VL - 1);

    /* ── Pass 1: sum(x^2), 8x unrolled ── */
    svfloat32_t sq0 = svdup_f32(0.0f), sq1 = sq0, sq2 = sq0, sq3 = sq0;
    svfloat32_t sq4 = sq0, sq5 = sq0, sq6 = sq0, sq7 = sq0;

    int i = 0;
    for (; i < N8; i += 8 * VL) {
        prefetch_l1(x + i, PF_DIST_BYTES);
        prefetch_l1(x + i, PF_DIST_BYTES + 256);
        svfloat32_t v0 = svld1_f32(pg, x + i + 0 * VL);
        svfloat32_t v1 = svld1_f32(pg, x + i + 1 * VL);
        svfloat32_t v2 = svld1_f32(pg, x + i + 2 * VL);
        svfloat32_t v3 = svld1_f32(pg, x + i + 3 * VL);
        svfloat32_t v4 = svld1_f32(pg, x + i + 4 * VL);
        svfloat32_t v5 = svld1_f32(pg, x + i + 5 * VL);
        svfloat32_t v6 = svld1_f32(pg, x + i + 6 * VL);
        svfloat32_t v7 = svld1_f32(pg, x + i + 7 * VL);
        sq0 = svmla_f32_x(pg, sq0, v0, v0);
        sq1 = svmla_f32_x(pg, sq1, v1, v1);
        sq2 = svmla_f32_x(pg, sq2, v2, v2);
        sq3 = svmla_f32_x(pg, sq3, v3, v3);
        sq4 = svmla_f32_x(pg, sq4, v4, v4);
        sq5 = svmla_f32_x(pg, sq5, v5, v5);
        sq6 = svmla_f32_x(pg, sq6, v6, v6);
        sq7 = svmla_f32_x(pg, sq7, v7, v7);
    }
    for (; i < N1; i += VL) {
        svfloat32_t v = svld1_f32(pg, x + i);
        sq0 = svmla_f32_x(pg, sq0, v, v);
    }
    if (N1 < N) {
        svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
        svfloat32_t v = svld1_f32(ptail, x + N1);
        sq0 = svmla_f32_m(ptail, sq0, v, v);
    }

    /* Tree reduce 8→1 */
    sq0 = svadd_f32_x(pg, sq0, sq1);
    sq2 = svadd_f32_x(pg, sq2, sq3);
    sq4 = svadd_f32_x(pg, sq4, sq5);
    sq6 = svadd_f32_x(pg, sq6, sq7);
    sq0 = svadd_f32_x(pg, sq0, sq2);
    sq4 = svadd_f32_x(pg, sq4, sq6);
    svfloat32_t vsum_sq = svadd_f32_x(pg, sq0, sq4);

    float sum_sq = hsum_f32(pg, vsum_sq);

    /* inv_rms = 1/sqrt(sum_sq/N + eps) */
    float var = sum_sq / (float)N + eps;
    svfloat32_t vvar = svdup_f32(var);
    svfloat32_t vinv_rms = sve_rsqrt_f32(pg, vvar);

    /* Extract scalar inv_rms */
    float ibuf[16] __attribute__((aligned(256)));
    svst1_f32(pg, ibuf, vinv_rms);
    float inv_rms = ibuf[0];
    svfloat32_t vinv = svdup_f32(inv_rms);

    /* ── Pass 2: y[i] = gamma[i] * x[i] * inv_rms, 8x unrolled ── */
    i = 0;
    if (gamma) {
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x + i, PF_DIST_BYTES);
            prefetch_l1(gamma + i, PF_DIST_BYTES);
            svfloat32_t v0 = svmul_f32_x(pg, svld1_f32(pg, x + i + 0 * VL), vinv);
            svfloat32_t v1 = svmul_f32_x(pg, svld1_f32(pg, x + i + 1 * VL), vinv);
            svfloat32_t v2 = svmul_f32_x(pg, svld1_f32(pg, x + i + 2 * VL), vinv);
            svfloat32_t v3 = svmul_f32_x(pg, svld1_f32(pg, x + i + 3 * VL), vinv);
            svfloat32_t v4 = svmul_f32_x(pg, svld1_f32(pg, x + i + 4 * VL), vinv);
            svfloat32_t v5 = svmul_f32_x(pg, svld1_f32(pg, x + i + 5 * VL), vinv);
            svfloat32_t v6 = svmul_f32_x(pg, svld1_f32(pg, x + i + 6 * VL), vinv);
            svfloat32_t v7 = svmul_f32_x(pg, svld1_f32(pg, x + i + 7 * VL), vinv);
            svst1_f32(pg, y + i + 0 * VL, svmul_f32_x(pg, v0, svld1_f32(pg, gamma + i + 0 * VL)));
            svst1_f32(pg, y + i + 1 * VL, svmul_f32_x(pg, v1, svld1_f32(pg, gamma + i + 1 * VL)));
            svst1_f32(pg, y + i + 2 * VL, svmul_f32_x(pg, v2, svld1_f32(pg, gamma + i + 2 * VL)));
            svst1_f32(pg, y + i + 3 * VL, svmul_f32_x(pg, v3, svld1_f32(pg, gamma + i + 3 * VL)));
            svst1_f32(pg, y + i + 4 * VL, svmul_f32_x(pg, v4, svld1_f32(pg, gamma + i + 4 * VL)));
            svst1_f32(pg, y + i + 5 * VL, svmul_f32_x(pg, v5, svld1_f32(pg, gamma + i + 5 * VL)));
            svst1_f32(pg, y + i + 6 * VL, svmul_f32_x(pg, v6, svld1_f32(pg, gamma + i + 6 * VL)));
            svst1_f32(pg, y + i + 7 * VL, svmul_f32_x(pg, v7, svld1_f32(pg, gamma + i + 7 * VL)));
        }
        for (; i < N1; i += VL) {
            svfloat32_t v = svmul_f32_x(pg, svld1_f32(pg, x + i), vinv);
            svst1_f32(pg, y + i, svmul_f32_x(pg, v, svld1_f32(pg, gamma + i)));
        }
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svfloat32_t v = svmul_f32_x(pg, svld1_f32(ptail, x + N1), vinv);
            svst1_f32(ptail, y + N1, svmul_f32_x(pg, v, svld1_f32(ptail, gamma + N1)));
        }
    } else {
        /* No gamma: y = x * inv_rms */
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x + i, PF_DIST_BYTES);
            svst1_f32(pg, y + i + 0 * VL, svmul_f32_x(pg, svld1_f32(pg, x + i + 0 * VL), vinv));
            svst1_f32(pg, y + i + 1 * VL, svmul_f32_x(pg, svld1_f32(pg, x + i + 1 * VL), vinv));
            svst1_f32(pg, y + i + 2 * VL, svmul_f32_x(pg, svld1_f32(pg, x + i + 2 * VL), vinv));
            svst1_f32(pg, y + i + 3 * VL, svmul_f32_x(pg, svld1_f32(pg, x + i + 3 * VL), vinv));
            svst1_f32(pg, y + i + 4 * VL, svmul_f32_x(pg, svld1_f32(pg, x + i + 4 * VL), vinv));
            svst1_f32(pg, y + i + 5 * VL, svmul_f32_x(pg, svld1_f32(pg, x + i + 5 * VL), vinv));
            svst1_f32(pg, y + i + 6 * VL, svmul_f32_x(pg, svld1_f32(pg, x + i + 6 * VL), vinv));
            svst1_f32(pg, y + i + 7 * VL, svmul_f32_x(pg, svld1_f32(pg, x + i + 7 * VL), vinv));
        }
        for (; i < N1; i += VL)
            svst1_f32(pg, y + i, svmul_f32_x(pg, svld1_f32(pg, x + i), vinv));
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svst1_f32(ptail, y + N1, svmul_f32_x(pg, svld1_f32(ptail, x + N1), vinv));
        }
    }
}


/* ════════════════════════════════════════════════════════════════
 * RMSNorm Forward FP16
 *
 * FP16 input/output, FP32 compute. Uses LD1H{Z.S} for fp16→fp32.
 * ════════════════════════════════════════════════════════════════ */

void rmsnorm_fwd_f16(const uint16_t *x_f16, uint16_t *y_f16,
                     const float *gamma, float eps, int N) {
    set_fpcr_fz16();

    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int N8 = N & ~(8 * VL - 1);
    const int N1 = N & ~(VL - 1);

    /* ── Pass 1: sum(x^2), 8x unrolled ── */
    svfloat32_t sq0 = svdup_f32(0.0f), sq1 = sq0, sq2 = sq0, sq3 = sq0;
    svfloat32_t sq4 = sq0, sq5 = sq0, sq6 = sq0, sq7 = sq0;

    int i = 0;
    for (; i < N8; i += 8 * VL) {
        prefetch_l1(x_f16 + i, PF_DIST_BYTES);
        svfloat32_t v0 = svld1_cvt_f16_f32(pg, x_f16 + i + 0 * VL);
        svfloat32_t v1 = svld1_cvt_f16_f32(pg, x_f16 + i + 1 * VL);
        svfloat32_t v2 = svld1_cvt_f16_f32(pg, x_f16 + i + 2 * VL);
        svfloat32_t v3 = svld1_cvt_f16_f32(pg, x_f16 + i + 3 * VL);
        svfloat32_t v4 = svld1_cvt_f16_f32(pg, x_f16 + i + 4 * VL);
        svfloat32_t v5 = svld1_cvt_f16_f32(pg, x_f16 + i + 5 * VL);
        svfloat32_t v6 = svld1_cvt_f16_f32(pg, x_f16 + i + 6 * VL);
        svfloat32_t v7 = svld1_cvt_f16_f32(pg, x_f16 + i + 7 * VL);
        sq0 = svmla_f32_x(pg, sq0, v0, v0);
        sq1 = svmla_f32_x(pg, sq1, v1, v1);
        sq2 = svmla_f32_x(pg, sq2, v2, v2);
        sq3 = svmla_f32_x(pg, sq3, v3, v3);
        sq4 = svmla_f32_x(pg, sq4, v4, v4);
        sq5 = svmla_f32_x(pg, sq5, v5, v5);
        sq6 = svmla_f32_x(pg, sq6, v6, v6);
        sq7 = svmla_f32_x(pg, sq7, v7, v7);
    }
    for (; i < N1; i += VL) {
        svfloat32_t v = svld1_cvt_f16_f32(pg, x_f16 + i);
        sq0 = svmla_f32_x(pg, sq0, v, v);
    }
    if (N1 < N) {
        svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
        svfloat32_t v = svld1_cvt_f16_f32(ptail, x_f16 + N1);
        sq0 = svmla_f32_m(ptail, sq0, v, v);
    }

    /* Tree reduce */
    sq0 = svadd_f32_x(pg, sq0, sq1);
    sq2 = svadd_f32_x(pg, sq2, sq3);
    sq4 = svadd_f32_x(pg, sq4, sq5);
    sq6 = svadd_f32_x(pg, sq6, sq7);
    sq0 = svadd_f32_x(pg, sq0, sq2);
    sq4 = svadd_f32_x(pg, sq4, sq6);
    svfloat32_t vsum_sq = svadd_f32_x(pg, sq0, sq4);

    float sum_sq = hsum_f32(pg, vsum_sq);
    float var = sum_sq / (float)N + eps;
    svfloat32_t vinv_rms = sve_rsqrt_f32(pg, svdup_f32(var));
    float ibuf[16] __attribute__((aligned(256)));
    svst1_f32(pg, ibuf, vinv_rms);
    float inv_rms = ibuf[0];
    svfloat32_t vinv = svdup_f32(inv_rms);

    /* ── Pass 2: y = gamma * x * inv_rms, convert back to fp16 ── */
    i = 0;
    if (gamma) {
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x_f16 + i, PF_DIST_BYTES);
            prefetch_l1(gamma + i, PF_DIST_BYTES);
            for (int u = 0; u < 8; u++) {
                svfloat32_t v = svmul_f32_x(pg, svld1_cvt_f16_f32(pg, x_f16 + i + u * VL), vinv);
                v = svmul_f32_x(pg, v, svld1_f32(pg, gamma + i + u * VL));
                svst1_cvt_f32_f16(pg, y_f16 + i + u * VL, v);
            }
        }
        for (; i < N1; i += VL) {
            svfloat32_t v = svmul_f32_x(pg, svld1_cvt_f16_f32(pg, x_f16 + i), vinv);
            v = svmul_f32_x(pg, v, svld1_f32(pg, gamma + i));
            svst1_cvt_f32_f16(pg, y_f16 + i, v);
        }
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svfloat32_t v = svmul_f32_x(pg, svld1_cvt_f16_f32(ptail, x_f16 + N1), vinv);
            v = svmul_f32_x(pg, v, svld1_f32(ptail, gamma + N1));
            svst1_cvt_f32_f16(ptail, y_f16 + N1, v);
        }
    } else {
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x_f16 + i, PF_DIST_BYTES);
            for (int u = 0; u < 8; u++) {
                svfloat32_t v = svmul_f32_x(pg, svld1_cvt_f16_f32(pg, x_f16 + i + u * VL), vinv);
                svst1_cvt_f32_f16(pg, y_f16 + i + u * VL, v);
            }
        }
        for (; i < N1; i += VL) {
            svfloat32_t v = svmul_f32_x(pg, svld1_cvt_f16_f32(pg, x_f16 + i), vinv);
            svst1_cvt_f32_f16(pg, y_f16 + i, v);
        }
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svfloat32_t v = svmul_f32_x(pg, svld1_cvt_f16_f32(ptail, x_f16 + N1), vinv);
            svst1_cvt_f32_f16(ptail, y_f16 + N1, v);
        }
    }
}


/* ════════════════════════════════════════════════════════════════
 * LayerNorm Forward FP32
 *
 * Pass 1: Simultaneously accumulate sum(x) and sum(x^2) — dual accum
 *         4x unrolled (8 accumulator vectors: 4 for sum, 4 for sum_sq)
 * Compute: mean, var = E[x^2] - E[x]^2, inv_std via FRSQRTE + 2 NR
 * Pass 2: y[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i]
 * ════════════════════════════════════════════════════════════════ */

void layernorm_fwd_f32(const float *x, float *y, const float *gamma,
                       const float *beta, float eps, int N) {
    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int N8 = N & ~(8 * VL - 1);
    const int N1 = N & ~(VL - 1);

    /* ── Pass 1: sum(x) and sum(x^2), 8x unrolled ── */
    svfloat32_t s0 = svdup_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    svfloat32_t s4 = s0, s5 = s0, s6 = s0, s7 = s0;
    svfloat32_t q0 = svdup_f32(0.0f), q1 = q0, q2 = q0, q3 = q0;
    svfloat32_t q4 = q0, q5 = q0, q6 = q0, q7 = q0;

    int i = 0;
    for (; i < N8; i += 8 * VL) {
        prefetch_l1(x + i, PF_DIST_BYTES);
        prefetch_l1(x + i, PF_DIST_BYTES + 256);
        svfloat32_t v0 = svld1_f32(pg, x + i + 0 * VL);
        svfloat32_t v1 = svld1_f32(pg, x + i + 1 * VL);
        svfloat32_t v2 = svld1_f32(pg, x + i + 2 * VL);
        svfloat32_t v3 = svld1_f32(pg, x + i + 3 * VL);
        svfloat32_t v4 = svld1_f32(pg, x + i + 4 * VL);
        svfloat32_t v5 = svld1_f32(pg, x + i + 5 * VL);
        svfloat32_t v6 = svld1_f32(pg, x + i + 6 * VL);
        svfloat32_t v7 = svld1_f32(pg, x + i + 7 * VL);
        s0 = svadd_f32_x(pg, s0, v0);
        s1 = svadd_f32_x(pg, s1, v1);
        s2 = svadd_f32_x(pg, s2, v2);
        s3 = svadd_f32_x(pg, s3, v3);
        s4 = svadd_f32_x(pg, s4, v4);
        s5 = svadd_f32_x(pg, s5, v5);
        s6 = svadd_f32_x(pg, s6, v6);
        s7 = svadd_f32_x(pg, s7, v7);
        q0 = svmla_f32_x(pg, q0, v0, v0);
        q1 = svmla_f32_x(pg, q1, v1, v1);
        q2 = svmla_f32_x(pg, q2, v2, v2);
        q3 = svmla_f32_x(pg, q3, v3, v3);
        q4 = svmla_f32_x(pg, q4, v4, v4);
        q5 = svmla_f32_x(pg, q5, v5, v5);
        q6 = svmla_f32_x(pg, q6, v6, v6);
        q7 = svmla_f32_x(pg, q7, v7, v7);
    }
    for (; i < N1; i += VL) {
        svfloat32_t v = svld1_f32(pg, x + i);
        s0 = svadd_f32_x(pg, s0, v);
        q0 = svmla_f32_x(pg, q0, v, v);
    }
    if (N1 < N) {
        svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
        svfloat32_t v = svld1_f32(ptail, x + N1);
        s0 = svadd_f32_m(ptail, s0, v);
        q0 = svmla_f32_m(ptail, q0, v, v);
    }

    /* Tree reduce sum */
    s0 = svadd_f32_x(pg, s0, s1);
    s2 = svadd_f32_x(pg, s2, s3);
    s4 = svadd_f32_x(pg, s4, s5);
    s6 = svadd_f32_x(pg, s6, s7);
    s0 = svadd_f32_x(pg, s0, s2);
    s4 = svadd_f32_x(pg, s4, s6);
    svfloat32_t vsum = svadd_f32_x(pg, s0, s4);

    /* Tree reduce sum_sq */
    q0 = svadd_f32_x(pg, q0, q1);
    q2 = svadd_f32_x(pg, q2, q3);
    q4 = svadd_f32_x(pg, q4, q5);
    q6 = svadd_f32_x(pg, q6, q7);
    q0 = svadd_f32_x(pg, q0, q2);
    q4 = svadd_f32_x(pg, q4, q6);
    svfloat32_t vsum_sq = svadd_f32_x(pg, q0, q4);

    float sum = hsum_f32(pg, vsum);
    float sum_sq = hsum_f32(pg, vsum_sq);

    float mean = sum / (float)N;
    float var = sum_sq / (float)N - mean * mean;
    float inv_std_val;
    {
        svfloat32_t vv = svdup_f32(var + eps);
        svfloat32_t vr = sve_rsqrt_f32(pg, vv);
        float rbuf[16] __attribute__((aligned(256)));
        svst1_f32(pg, rbuf, vr);
        inv_std_val = rbuf[0];
    }
    svfloat32_t vmean = svdup_f32(mean);
    svfloat32_t vinv_std = svdup_f32(inv_std_val);

    /* ── Pass 2: y = gamma * (x - mean) * inv_std + beta ── */
    i = 0;
    if (gamma && beta) {
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x + i, PF_DIST_BYTES);
            prefetch_l1(gamma + i, PF_DIST_BYTES);
            for (int u = 0; u < 8; u++) {
                svfloat32_t v = svld1_f32(pg, x + i + u * VL);
                svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
                svfloat32_t g = svld1_f32(pg, gamma + i + u * VL);
                svfloat32_t b = svld1_f32(pg, beta + i + u * VL);
                svst1_f32(pg, y + i + u * VL, svmla_f32_x(pg, b, xhat, g));
            }
        }
        for (; i < N1; i += VL) {
            svfloat32_t v = svld1_f32(pg, x + i);
            svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
            svfloat32_t g = svld1_f32(pg, gamma + i);
            svfloat32_t b = svld1_f32(pg, beta + i);
            svst1_f32(pg, y + i, svmla_f32_x(pg, b, xhat, g));
        }
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svfloat32_t v = svld1_f32(ptail, x + N1);
            svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
            svfloat32_t g = svld1_f32(ptail, gamma + N1);
            svfloat32_t b = svld1_f32(ptail, beta + N1);
            svst1_f32(ptail, y + N1, svmla_f32_x(pg, b, xhat, g));
        }
    } else if (gamma) {
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x + i, PF_DIST_BYTES);
            prefetch_l1(gamma + i, PF_DIST_BYTES);
            for (int u = 0; u < 8; u++) {
                svfloat32_t v = svld1_f32(pg, x + i + u * VL);
                svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
                svst1_f32(pg, y + i + u * VL, svmul_f32_x(pg, xhat, svld1_f32(pg, gamma + i + u * VL)));
            }
        }
        for (; i < N1; i += VL) {
            svfloat32_t v = svld1_f32(pg, x + i);
            svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
            svst1_f32(pg, y + i, svmul_f32_x(pg, xhat, svld1_f32(pg, gamma + i)));
        }
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svfloat32_t v = svld1_f32(ptail, x + N1);
            svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
            svst1_f32(ptail, y + N1, svmul_f32_x(pg, xhat, svld1_f32(ptail, gamma + N1)));
        }
    } else {
        /* No gamma/beta: y = (x - mean) * inv_std */
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x + i, PF_DIST_BYTES);
            for (int u = 0; u < 8; u++) {
                svfloat32_t v = svld1_f32(pg, x + i + u * VL);
                svst1_f32(pg, y + i + u * VL,
                          svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std));
            }
        }
        for (; i < N1; i += VL) {
            svfloat32_t v = svld1_f32(pg, x + i);
            svst1_f32(pg, y + i, svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std));
        }
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svfloat32_t v = svld1_f32(ptail, x + N1);
            svst1_f32(ptail, y + N1,
                      svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std));
        }
    }
}


/* ════════════════════════════════════════════════════════════════
 * LayerNorm Forward FP16
 * ════════════════════════════════════════════════════════════════ */

void layernorm_fwd_f16(const uint16_t *x_f16, uint16_t *y_f16,
                       const float *gamma, const float *beta,
                       float eps, int N) {
    set_fpcr_fz16();

    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int N8 = N & ~(8 * VL - 1);
    const int N1 = N & ~(VL - 1);

    /* ── Pass 1: sum(x) and sum(x^2), 8x unrolled ── */
    svfloat32_t s0 = svdup_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    svfloat32_t s4 = s0, s5 = s0, s6 = s0, s7 = s0;
    svfloat32_t q0 = svdup_f32(0.0f), q1 = q0, q2 = q0, q3 = q0;
    svfloat32_t q4 = q0, q5 = q0, q6 = q0, q7 = q0;

    int i = 0;
    for (; i < N8; i += 8 * VL) {
        prefetch_l1(x_f16 + i, PF_DIST_BYTES);
        svfloat32_t v0 = svld1_cvt_f16_f32(pg, x_f16 + i + 0 * VL);
        svfloat32_t v1 = svld1_cvt_f16_f32(pg, x_f16 + i + 1 * VL);
        svfloat32_t v2 = svld1_cvt_f16_f32(pg, x_f16 + i + 2 * VL);
        svfloat32_t v3 = svld1_cvt_f16_f32(pg, x_f16 + i + 3 * VL);
        svfloat32_t v4 = svld1_cvt_f16_f32(pg, x_f16 + i + 4 * VL);
        svfloat32_t v5 = svld1_cvt_f16_f32(pg, x_f16 + i + 5 * VL);
        svfloat32_t v6 = svld1_cvt_f16_f32(pg, x_f16 + i + 6 * VL);
        svfloat32_t v7 = svld1_cvt_f16_f32(pg, x_f16 + i + 7 * VL);
        s0 = svadd_f32_x(pg, s0, v0); q0 = svmla_f32_x(pg, q0, v0, v0);
        s1 = svadd_f32_x(pg, s1, v1); q1 = svmla_f32_x(pg, q1, v1, v1);
        s2 = svadd_f32_x(pg, s2, v2); q2 = svmla_f32_x(pg, q2, v2, v2);
        s3 = svadd_f32_x(pg, s3, v3); q3 = svmla_f32_x(pg, q3, v3, v3);
        s4 = svadd_f32_x(pg, s4, v4); q4 = svmla_f32_x(pg, q4, v4, v4);
        s5 = svadd_f32_x(pg, s5, v5); q5 = svmla_f32_x(pg, q5, v5, v5);
        s6 = svadd_f32_x(pg, s6, v6); q6 = svmla_f32_x(pg, q6, v6, v6);
        s7 = svadd_f32_x(pg, s7, v7); q7 = svmla_f32_x(pg, q7, v7, v7);
    }
    for (; i < N1; i += VL) {
        svfloat32_t v = svld1_cvt_f16_f32(pg, x_f16 + i);
        s0 = svadd_f32_x(pg, s0, v);
        q0 = svmla_f32_x(pg, q0, v, v);
    }
    if (N1 < N) {
        svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
        svfloat32_t v = svld1_cvt_f16_f32(ptail, x_f16 + N1);
        s0 = svadd_f32_m(ptail, s0, v);
        q0 = svmla_f32_m(ptail, q0, v, v);
    }

    /* Tree reduce */
    s0 = svadd_f32_x(pg, s0, s1); q0 = svadd_f32_x(pg, q0, q1);
    s2 = svadd_f32_x(pg, s2, s3); q2 = svadd_f32_x(pg, q2, q3);
    s4 = svadd_f32_x(pg, s4, s5); q4 = svadd_f32_x(pg, q4, q5);
    s6 = svadd_f32_x(pg, s6, s7); q6 = svadd_f32_x(pg, q6, q7);
    s0 = svadd_f32_x(pg, s0, s2); q0 = svadd_f32_x(pg, q0, q2);
    s4 = svadd_f32_x(pg, s4, s6); q4 = svadd_f32_x(pg, q4, q6);
    svfloat32_t vsum = svadd_f32_x(pg, s0, s4);
    svfloat32_t vsum_sq = svadd_f32_x(pg, q0, q4);

    float sum = hsum_f32(pg, vsum);
    float sum_sq = hsum_f32(pg, vsum_sq);

    float mean = sum / (float)N;
    float var = sum_sq / (float)N - mean * mean;
    float inv_std_val;
    {
        svfloat32_t vv = svdup_f32(var + eps);
        svfloat32_t vr = sve_rsqrt_f32(pg, vv);
        float rbuf[16] __attribute__((aligned(256)));
        svst1_f32(pg, rbuf, vr);
        inv_std_val = rbuf[0];
    }
    svfloat32_t vmean = svdup_f32(mean);
    svfloat32_t vinv_std = svdup_f32(inv_std_val);

    /* ── Pass 2: y = gamma * (x - mean) * inv_std + beta ── */
    i = 0;
    if (gamma && beta) {
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x_f16 + i, PF_DIST_BYTES);
            prefetch_l1(gamma + i, PF_DIST_BYTES);
            for (int u = 0; u < 8; u++) {
                svfloat32_t v = svld1_cvt_f16_f32(pg, x_f16 + i + u * VL);
                svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
                svfloat32_t g = svld1_f32(pg, gamma + i + u * VL);
                svfloat32_t b = svld1_f32(pg, beta + i + u * VL);
                svst1_cvt_f32_f16(pg, y_f16 + i + u * VL, svmla_f32_x(pg, b, xhat, g));
            }
        }
        for (; i < N1; i += VL) {
            svfloat32_t v = svld1_cvt_f16_f32(pg, x_f16 + i);
            svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
            svfloat32_t g = svld1_f32(pg, gamma + i);
            svfloat32_t b = svld1_f32(pg, beta + i);
            svst1_cvt_f32_f16(pg, y_f16 + i, svmla_f32_x(pg, b, xhat, g));
        }
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svfloat32_t v = svld1_cvt_f16_f32(ptail, x_f16 + N1);
            svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
            svfloat32_t g = svld1_f32(ptail, gamma + N1);
            svfloat32_t b = svld1_f32(ptail, beta + N1);
            svst1_cvt_f32_f16(ptail, y_f16 + N1, svmla_f32_x(pg, b, xhat, g));
        }
    } else if (gamma) {
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x_f16 + i, PF_DIST_BYTES);
            for (int u = 0; u < 8; u++) {
                svfloat32_t v = svld1_cvt_f16_f32(pg, x_f16 + i + u * VL);
                svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
                svst1_cvt_f32_f16(pg, y_f16 + i + u * VL,
                                  svmul_f32_x(pg, xhat, svld1_f32(pg, gamma + i + u * VL)));
            }
        }
        for (; i < N1; i += VL) {
            svfloat32_t v = svld1_cvt_f16_f32(pg, x_f16 + i);
            svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
            svst1_cvt_f32_f16(pg, y_f16 + i,
                              svmul_f32_x(pg, xhat, svld1_f32(pg, gamma + i)));
        }
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svfloat32_t v = svld1_cvt_f16_f32(ptail, x_f16 + N1);
            svfloat32_t xhat = svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std);
            svst1_cvt_f32_f16(ptail, y_f16 + N1,
                              svmul_f32_x(pg, xhat, svld1_f32(ptail, gamma + N1)));
        }
    } else {
        for (; i < N8; i += 8 * VL) {
            prefetch_l1(x_f16 + i, PF_DIST_BYTES);
            for (int u = 0; u < 8; u++) {
                svfloat32_t v = svld1_cvt_f16_f32(pg, x_f16 + i + u * VL);
                svst1_cvt_f32_f16(pg, y_f16 + i + u * VL,
                                  svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std));
            }
        }
        for (; i < N1; i += VL) {
            svfloat32_t v = svld1_cvt_f16_f32(pg, x_f16 + i);
            svst1_cvt_f32_f16(pg, y_f16 + i,
                              svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std));
        }
        if (N1 < N) {
            svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
            svfloat32_t v = svld1_cvt_f16_f32(ptail, x_f16 + N1);
            svst1_cvt_f32_f16(ptail, y_f16 + N1,
                              svmul_f32_x(pg, svsub_f32_x(pg, v, vmean), vinv_std));
        }
    }
}


/* ════════════════════════════════════════════════════════════════
 * RMSNorm Backward FP32
 *
 * Pass 1: ds = sum(dy[i] * gamma[i] * x[i]) / N
 * Pass 2: dx[i] = (dy[i]*gamma[i] - x[i]*ds) * inv_rms
 *          dgamma[i] += dy[i] * x[i] * inv_rms
 * ════════════════════════════════════════════════════════════════ */

void rmsnorm_bwd_f32(const float *dy, const float *x, float *dx,
                     const float *gamma, float *dgamma,
                     float inv_rms, int N) {
    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int N8 = N & ~(8 * VL - 1);
    const int N1 = N & ~(VL - 1);
    svfloat32_t vinv = svdup_f32(inv_rms);

    /* ── Pass 1: ds = sum(dy * gamma * x) / N ── */
    svfloat32_t d0 = svdup_f32(0.0f), d1 = d0, d2 = d0, d3 = d0;
    svfloat32_t d4 = d0, d5 = d0, d6 = d0, d7 = d0;

    int i = 0;
    for (; i < N8; i += 8 * VL) {
        prefetch_l1(dy + i, PF_DIST_BYTES);
        prefetch_l1(x + i, PF_DIST_BYTES);
        prefetch_l1(gamma + i, PF_DIST_BYTES);
        for (int u = 0; u < 8; u++) {
            svfloat32_t dv = svld1_f32(pg, dy + i + u * VL);
            svfloat32_t xv = svld1_f32(pg, x + i + u * VL);
            svfloat32_t gv = svld1_f32(pg, gamma + i + u * VL);
            svfloat32_t dgx = svmul_f32_x(pg, dv, gv);
            svfloat32_t *acc;
            switch (u) {
                case 0: d0 = svmla_f32_x(pg, d0, dgx, xv); break;
                case 1: d1 = svmla_f32_x(pg, d1, dgx, xv); break;
                case 2: d2 = svmla_f32_x(pg, d2, dgx, xv); break;
                case 3: d3 = svmla_f32_x(pg, d3, dgx, xv); break;
                case 4: d4 = svmla_f32_x(pg, d4, dgx, xv); break;
                case 5: d5 = svmla_f32_x(pg, d5, dgx, xv); break;
                case 6: d6 = svmla_f32_x(pg, d6, dgx, xv); break;
                case 7: d7 = svmla_f32_x(pg, d7, dgx, xv); break;
            }
        }
    }
    for (; i < N1; i += VL) {
        svfloat32_t dv = svld1_f32(pg, dy + i);
        svfloat32_t xv = svld1_f32(pg, x + i);
        svfloat32_t gv = svld1_f32(pg, gamma + i);
        d0 = svmla_f32_x(pg, d0, svmul_f32_x(pg, dv, gv), xv);
    }
    if (N1 < N) {
        svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
        svfloat32_t dv = svld1_f32(ptail, dy + N1);
        svfloat32_t xv = svld1_f32(ptail, x + N1);
        svfloat32_t gv = svld1_f32(ptail, gamma + N1);
        d0 = svmla_f32_m(ptail, d0, svmul_f32_x(pg, dv, gv), xv);
    }

    /* Tree reduce */
    d0 = svadd_f32_x(pg, d0, d1);
    d2 = svadd_f32_x(pg, d2, d3);
    d4 = svadd_f32_x(pg, d4, d5);
    d6 = svadd_f32_x(pg, d6, d7);
    d0 = svadd_f32_x(pg, d0, d2);
    d4 = svadd_f32_x(pg, d4, d6);
    svfloat32_t vds = svadd_f32_x(pg, d0, d4);
    float ds = hsum_f32(pg, vds) / (float)N;

    /* ds_scaled = ds * inv_rms^2 = ds * inv_rms * inv_rms */
    float ds_inv_rms2 = ds * inv_rms * inv_rms;
    svfloat32_t vds_ir2 = svdup_f32(ds_inv_rms2);

    /* ── Pass 2: dx[i] = (dy[i]*gamma[i] - x[i]*ds*inv_rms^2) * inv_rms
     *            dx[i] = dy[i]*gamma[i]*inv_rms - x[i]*ds_inv_rms2
     *            dgamma[i] += dy[i] * x[i] * inv_rms
     */
    i = 0;
    for (; i < N8; i += 8 * VL) {
        prefetch_l1(dy + i, PF_DIST_BYTES);
        prefetch_l1(x + i, PF_DIST_BYTES);
        prefetch_l1(gamma + i, PF_DIST_BYTES);
        for (int u = 0; u < 8; u++) {
            svfloat32_t dv = svld1_f32(pg, dy + i + u * VL);
            svfloat32_t xv = svld1_f32(pg, x + i + u * VL);
            svfloat32_t gv = svld1_f32(pg, gamma + i + u * VL);
            /* dx = dy*gamma*inv_rms - x*ds_inv_rms2 */
            svfloat32_t dg_ir = svmul_f32_x(pg, svmul_f32_x(pg, dv, gv), vinv);
            svfloat32_t dx_val = svmsb_f32_x(pg, xv, vds_ir2, dg_ir);
            svst1_f32(pg, dx + i + u * VL, dx_val);
            /* dgamma += dy * x * inv_rms */
            svfloat32_t dg_cur = svld1_f32(pg, dgamma + i + u * VL);
            dg_cur = svmla_f32_x(pg, dg_cur, svmul_f32_x(pg, dv, xv), vinv);
            svst1_f32(pg, dgamma + i + u * VL, dg_cur);
        }
    }
    for (; i < N1; i += VL) {
        svfloat32_t dv = svld1_f32(pg, dy + i);
        svfloat32_t xv = svld1_f32(pg, x + i);
        svfloat32_t gv = svld1_f32(pg, gamma + i);
        svfloat32_t dg_ir = svmul_f32_x(pg, svmul_f32_x(pg, dv, gv), vinv);
        svst1_f32(pg, dx + i, svmsb_f32_x(pg, xv, vds_ir2, dg_ir));
        svfloat32_t dg_cur = svld1_f32(pg, dgamma + i);
        dg_cur = svmla_f32_x(pg, dg_cur, svmul_f32_x(pg, dv, xv), vinv);
        svst1_f32(pg, dgamma + i, dg_cur);
    }
    if (N1 < N) {
        svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
        svfloat32_t dv = svld1_f32(ptail, dy + N1);
        svfloat32_t xv = svld1_f32(ptail, x + N1);
        svfloat32_t gv = svld1_f32(ptail, gamma + N1);
        svfloat32_t dg_ir = svmul_f32_x(pg, svmul_f32_x(pg, dv, gv), vinv);
        svst1_f32(ptail, dx + N1, svmsb_f32_x(pg, xv, vds_ir2, dg_ir));
        svfloat32_t dg_cur = svld1_f32(ptail, dgamma + N1);
        dg_cur = svmla_f32_x(pg, dg_cur, svmul_f32_x(pg, dv, xv), vinv);
        svst1_f32(ptail, dgamma + N1, dg_cur);
    }
}


/* ════════════════════════════════════════════════════════════════
 * LayerNorm Backward FP32
 *
 * Pass 1: ds = sum(dy[i]*gamma[i]*xhat[i]), db = sum(dy[i]*gamma[i])
 *         where xhat[i] = (x[i] - mean) * inv_std
 * Pass 2: dx[i] = inv_std * (dy[i]*gamma[i] - (db + ds*xhat[i])/N)
 *         dgamma[i] += dy[i] * xhat[i]
 *         dbeta[i]  += dy[i]
 * ════════════════════════════════════════════════════════════════ */

void layernorm_bwd_f32(const float *dy, const float *x, float *dx,
                       const float *gamma, float *dgamma, float *dbeta,
                       float mean, float inv_std, int N) {
    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int N8 = N & ~(8 * VL - 1);
    const int N1 = N & ~(VL - 1);

    svfloat32_t vmean = svdup_f32(mean);
    svfloat32_t vinv_std = svdup_f32(inv_std);

    /* ── Pass 1: ds = sum(dy*gamma*xhat), db = sum(dy*gamma) ── */
    svfloat32_t ds0 = svdup_f32(0.0f), ds1 = ds0, ds2 = ds0, ds3 = ds0;
    svfloat32_t ds4 = ds0, ds5 = ds0, ds6 = ds0, ds7 = ds0;
    svfloat32_t db0 = svdup_f32(0.0f), db1 = db0, db2 = db0, db3 = db0;
    svfloat32_t db4 = db0, db5 = db0, db6 = db0, db7 = db0;

    int i = 0;
    for (; i < N8; i += 8 * VL) {
        prefetch_l1(dy + i, PF_DIST_BYTES);
        prefetch_l1(x + i, PF_DIST_BYTES);
        prefetch_l1(gamma + i, PF_DIST_BYTES);
        svfloat32_t dv0 = svld1_f32(pg, dy + i + 0 * VL);
        svfloat32_t dv1 = svld1_f32(pg, dy + i + 1 * VL);
        svfloat32_t dv2 = svld1_f32(pg, dy + i + 2 * VL);
        svfloat32_t dv3 = svld1_f32(pg, dy + i + 3 * VL);
        svfloat32_t dv4 = svld1_f32(pg, dy + i + 4 * VL);
        svfloat32_t dv5 = svld1_f32(pg, dy + i + 5 * VL);
        svfloat32_t dv6 = svld1_f32(pg, dy + i + 6 * VL);
        svfloat32_t dv7 = svld1_f32(pg, dy + i + 7 * VL);
        svfloat32_t g0 = svld1_f32(pg, gamma + i + 0 * VL);
        svfloat32_t g1 = svld1_f32(pg, gamma + i + 1 * VL);
        svfloat32_t g2 = svld1_f32(pg, gamma + i + 2 * VL);
        svfloat32_t g3 = svld1_f32(pg, gamma + i + 3 * VL);
        svfloat32_t g4 = svld1_f32(pg, gamma + i + 4 * VL);
        svfloat32_t g5 = svld1_f32(pg, gamma + i + 5 * VL);
        svfloat32_t g6 = svld1_f32(pg, gamma + i + 6 * VL);
        svfloat32_t g7 = svld1_f32(pg, gamma + i + 7 * VL);
        svfloat32_t dg0 = svmul_f32_x(pg, dv0, g0);
        svfloat32_t dg1 = svmul_f32_x(pg, dv1, g1);
        svfloat32_t dg2 = svmul_f32_x(pg, dv2, g2);
        svfloat32_t dg3 = svmul_f32_x(pg, dv3, g3);
        svfloat32_t dg4 = svmul_f32_x(pg, dv4, g4);
        svfloat32_t dg5 = svmul_f32_x(pg, dv5, g5);
        svfloat32_t dg6 = svmul_f32_x(pg, dv6, g6);
        svfloat32_t dg7 = svmul_f32_x(pg, dv7, g7);
        /* xhat = (x - mean) * inv_std */
        svfloat32_t xh0 = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, x + i + 0 * VL), vmean), vinv_std);
        svfloat32_t xh1 = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, x + i + 1 * VL), vmean), vinv_std);
        svfloat32_t xh2 = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, x + i + 2 * VL), vmean), vinv_std);
        svfloat32_t xh3 = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, x + i + 3 * VL), vmean), vinv_std);
        svfloat32_t xh4 = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, x + i + 4 * VL), vmean), vinv_std);
        svfloat32_t xh5 = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, x + i + 5 * VL), vmean), vinv_std);
        svfloat32_t xh6 = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, x + i + 6 * VL), vmean), vinv_std);
        svfloat32_t xh7 = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, x + i + 7 * VL), vmean), vinv_std);
        /* Accumulate ds = sum(dy*gamma*xhat), db = sum(dy*gamma) */
        ds0 = svmla_f32_x(pg, ds0, dg0, xh0); db0 = svadd_f32_x(pg, db0, dg0);
        ds1 = svmla_f32_x(pg, ds1, dg1, xh1); db1 = svadd_f32_x(pg, db1, dg1);
        ds2 = svmla_f32_x(pg, ds2, dg2, xh2); db2 = svadd_f32_x(pg, db2, dg2);
        ds3 = svmla_f32_x(pg, ds3, dg3, xh3); db3 = svadd_f32_x(pg, db3, dg3);
        ds4 = svmla_f32_x(pg, ds4, dg4, xh4); db4 = svadd_f32_x(pg, db4, dg4);
        ds5 = svmla_f32_x(pg, ds5, dg5, xh5); db5 = svadd_f32_x(pg, db5, dg5);
        ds6 = svmla_f32_x(pg, ds6, dg6, xh6); db6 = svadd_f32_x(pg, db6, dg6);
        ds7 = svmla_f32_x(pg, ds7, dg7, xh7); db7 = svadd_f32_x(pg, db7, dg7);
    }
    for (; i < N1; i += VL) {
        svfloat32_t dv = svld1_f32(pg, dy + i);
        svfloat32_t gv = svld1_f32(pg, gamma + i);
        svfloat32_t dg = svmul_f32_x(pg, dv, gv);
        svfloat32_t xh = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, x + i), vmean), vinv_std);
        ds0 = svmla_f32_x(pg, ds0, dg, xh);
        db0 = svadd_f32_x(pg, db0, dg);
    }
    if (N1 < N) {
        svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
        svfloat32_t dv = svld1_f32(ptail, dy + N1);
        svfloat32_t gv = svld1_f32(ptail, gamma + N1);
        svfloat32_t dg = svmul_f32_x(pg, dv, gv);
        svfloat32_t xh = svmul_f32_x(pg, svsub_f32_x(pg, svld1_f32(ptail, x + N1), vmean), vinv_std);
        ds0 = svmla_f32_m(ptail, ds0, dg, xh);
        db0 = svadd_f32_m(ptail, db0, dg);
    }

    /* Tree reduce ds */
    ds0 = svadd_f32_x(pg, ds0, ds1); ds2 = svadd_f32_x(pg, ds2, ds3);
    ds4 = svadd_f32_x(pg, ds4, ds5); ds6 = svadd_f32_x(pg, ds6, ds7);
    ds0 = svadd_f32_x(pg, ds0, ds2); ds4 = svadd_f32_x(pg, ds4, ds6);
    float ds_val = hsum_f32(pg, svadd_f32_x(pg, ds0, ds4)) / (float)N;

    /* Tree reduce db */
    db0 = svadd_f32_x(pg, db0, db1); db2 = svadd_f32_x(pg, db2, db3);
    db4 = svadd_f32_x(pg, db4, db5); db6 = svadd_f32_x(pg, db6, db7);
    db0 = svadd_f32_x(pg, db0, db2); db4 = svadd_f32_x(pg, db4, db6);
    float db_val = hsum_f32(pg, svadd_f32_x(pg, db0, db4)) / (float)N;

    svfloat32_t vds_n = svdup_f32(ds_val);
    svfloat32_t vdb_n = svdup_f32(db_val);

    /* ── Pass 2: dx, dgamma, dbeta ── */
    i = 0;
    for (; i < N8; i += 8 * VL) {
        prefetch_l1(dy + i, PF_DIST_BYTES);
        prefetch_l1(x + i, PF_DIST_BYTES);
        prefetch_l1(gamma + i, PF_DIST_BYTES);
        for (int u = 0; u < 8; u++) {
            svfloat32_t dv = svld1_f32(pg, dy + i + u * VL);
            svfloat32_t xv = svld1_f32(pg, x + i + u * VL);
            svfloat32_t gv = svld1_f32(pg, gamma + i + u * VL);
            svfloat32_t xh = svmul_f32_x(pg, svsub_f32_x(pg, xv, vmean), vinv_std);
            /* dx = inv_std * (dy*gamma - db/N - ds/N * xhat) */
            svfloat32_t dg = svmul_f32_x(pg, dv, gv);
            svfloat32_t inner = svsub_f32_x(pg, dg, vdb_n);
            inner = svmsb_f32_x(pg, xh, vds_n, inner);
            svst1_f32(pg, dx + i + u * VL, svmul_f32_x(pg, inner, vinv_std));
            /* dgamma += dy * xhat */
            svfloat32_t dg_cur = svld1_f32(pg, dgamma + i + u * VL);
            svst1_f32(pg, dgamma + i + u * VL, svmla_f32_x(pg, dg_cur, dv, xh));
            /* dbeta += dy */
            if (dbeta) {
                svfloat32_t db_cur = svld1_f32(pg, dbeta + i + u * VL);
                svst1_f32(pg, dbeta + i + u * VL, svadd_f32_x(pg, db_cur, dv));
            }
        }
    }
    for (; i < N1; i += VL) {
        svfloat32_t dv = svld1_f32(pg, dy + i);
        svfloat32_t xv = svld1_f32(pg, x + i);
        svfloat32_t gv = svld1_f32(pg, gamma + i);
        svfloat32_t xh = svmul_f32_x(pg, svsub_f32_x(pg, xv, vmean), vinv_std);
        svfloat32_t dg = svmul_f32_x(pg, dv, gv);
        svfloat32_t inner = svsub_f32_x(pg, dg, vdb_n);
        inner = svmsb_f32_x(pg, xh, vds_n, inner);
        svst1_f32(pg, dx + i, svmul_f32_x(pg, inner, vinv_std));
        svfloat32_t dg_cur = svld1_f32(pg, dgamma + i);
        svst1_f32(pg, dgamma + i, svmla_f32_x(pg, dg_cur, dv, xh));
        if (dbeta) {
            svfloat32_t db_cur = svld1_f32(pg, dbeta + i);
            svst1_f32(pg, dbeta + i, svadd_f32_x(pg, db_cur, dv));
        }
    }
    if (N1 < N) {
        svbool_t ptail = svwhilelt_b32((uint32_t)N1, (uint32_t)N);
        svfloat32_t dv = svld1_f32(ptail, dy + N1);
        svfloat32_t xv = svld1_f32(ptail, x + N1);
        svfloat32_t gv = svld1_f32(ptail, gamma + N1);
        svfloat32_t xh = svmul_f32_x(pg, svsub_f32_x(pg, xv, vmean), vinv_std);
        svfloat32_t dg = svmul_f32_x(pg, dv, gv);
        svfloat32_t inner = svsub_f32_x(pg, dg, vdb_n);
        inner = svmsb_f32_x(pg, xh, vds_n, inner);
        svst1_f32(ptail, dx + N1, svmul_f32_x(pg, inner, vinv_std));
        svfloat32_t dg_cur = svld1_f32(ptail, dgamma + N1);
        svst1_f32(ptail, dgamma + N1, svmla_f32_x(pg, dg_cur, dv, xh));
        if (dbeta) {
            svfloat32_t db_cur = svld1_f32(ptail, dbeta + N1);
            svst1_f32(ptail, dbeta + N1, svadd_f32_x(pg, db_cur, dv));
        }
    }
}


/* ════════════════════════════════════════════════════════════════
 * Batch Forward with OpenMP
 * ════════════════════════════════════════════════════════════════ */

void rmsnorm_batch_fwd_f32(const float *x, float *y, const float *gamma,
                           float eps, int M, int N) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++)
        rmsnorm_fwd_f32(x + (int64_t)m * N, y + (int64_t)m * N,
                        gamma, eps, N);
}

void rmsnorm_batch_fwd_f16(const uint16_t *x_f16, uint16_t *y_f16,
                           const float *gamma, float eps, int M, int N) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++)
        rmsnorm_fwd_f16(x_f16 + (int64_t)m * N, y_f16 + (int64_t)m * N,
                        gamma, eps, N);
}

void layernorm_batch_fwd_f32(const float *x, float *y, const float *gamma,
                             const float *beta, float eps, int M, int N) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++)
        layernorm_fwd_f32(x + (int64_t)m * N, y + (int64_t)m * N,
                          gamma, beta, eps, N);
}

void layernorm_batch_fwd_f16(const uint16_t *x_f16, uint16_t *y_f16,
                             const float *gamma, const float *beta,
                             float eps, int M, int N) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++)
        layernorm_fwd_f16(x_f16 + (int64_t)m * N, y_f16 + (int64_t)m * N,
                          gamma, beta, eps, N);
}


/* ════════════════════════════════════════════════════════════════
 * Batch Backward with OpenMP
 *
 * Per-thread dgamma/dbeta accumulation to avoid atomics.
 * ════════════════════════════════════════════════════════════════ */

void rmsnorm_batch_bwd_f32(const float *dy, const float *x, float *dx,
                           const float *gamma, float *dgamma,
                           const float *inv_rms, int M, int N) {
    /* Zero dgamma */
    memset(dgamma, 0, (size_t)N * sizeof(float));

#ifdef _OPENMP
    int nthreads;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    /* Per-thread dgamma buffers */
    float *thr_dgamma = (float *)aligned_alloc(256, (size_t)nthreads * N * sizeof(float));
    memset(thr_dgamma, 0, (size_t)nthreads * N * sizeof(float));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *my_dgamma = thr_dgamma + (int64_t)tid * N;
        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            rmsnorm_bwd_f32(dy + (int64_t)m * N, x + (int64_t)m * N,
                           dx + (int64_t)m * N, gamma, my_dgamma,
                           inv_rms[m], N);
        }
    }

    /* Reduce per-thread dgamma */
    for (int t = 0; t < nthreads; t++) {
        const float *src = thr_dgamma + (int64_t)t * N;
        for (int j = 0; j < N; j++)
            dgamma[j] += src[j];
    }
    free(thr_dgamma);
#else
    for (int m = 0; m < M; m++) {
        rmsnorm_bwd_f32(dy + (int64_t)m * N, x + (int64_t)m * N,
                       dx + (int64_t)m * N, gamma, dgamma,
                       inv_rms[m], N);
    }
#endif
}

void layernorm_batch_bwd_f32(const float *dy, const float *x, float *dx,
                             const float *gamma, float *dgamma, float *dbeta,
                             const float *mean, const float *inv_std,
                             int M, int N) {
    memset(dgamma, 0, (size_t)N * sizeof(float));
    if (dbeta) memset(dbeta, 0, (size_t)N * sizeof(float));

#ifdef _OPENMP
    int nthreads;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    float *thr_dgamma = (float *)aligned_alloc(256, (size_t)nthreads * N * sizeof(float));
    float *thr_dbeta  = dbeta ? (float *)aligned_alloc(256, (size_t)nthreads * N * sizeof(float)) : NULL;
    memset(thr_dgamma, 0, (size_t)nthreads * N * sizeof(float));
    if (thr_dbeta) memset(thr_dbeta, 0, (size_t)nthreads * N * sizeof(float));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *my_dgamma = thr_dgamma + (int64_t)tid * N;
        float *my_dbeta  = thr_dbeta ? thr_dbeta + (int64_t)tid * N : NULL;
        #pragma omp for schedule(static)
        for (int m = 0; m < M; m++) {
            layernorm_bwd_f32(dy + (int64_t)m * N, x + (int64_t)m * N,
                             dx + (int64_t)m * N, gamma, my_dgamma, my_dbeta,
                             mean[m], inv_std[m], N);
        }
    }

    /* Reduce per-thread buffers */
    for (int t = 0; t < nthreads; t++) {
        const float *src_g = thr_dgamma + (int64_t)t * N;
        for (int j = 0; j < N; j++)
            dgamma[j] += src_g[j];
        if (dbeta && thr_dbeta) {
            const float *src_b = thr_dbeta + (int64_t)t * N;
            for (int j = 0; j < N; j++)
                dbeta[j] += src_b[j];
        }
    }
    free(thr_dgamma);
    free(thr_dbeta);
#else
    for (int m = 0; m < M; m++) {
        layernorm_bwd_f32(dy + (int64_t)m * N, x + (int64_t)m * N,
                         dx + (int64_t)m * N, gamma, dgamma, dbeta,
                         mean[m], inv_std[m], N);
    }
#endif
}


/* ════════════════════════════════════════════════════════════════
 * Reference implementations (double precision)
 * ════════════════════════════════════════════════════════════════ */

void rmsnorm_ref_f64(const float *x, float *y, const float *gamma,
                     float eps, int N) {
    double sum_sq = 0.0;
    for (int i = 0; i < N; i++)
        sum_sq += (double)x[i] * (double)x[i];
    double inv_rms = 1.0 / sqrt(sum_sq / N + (double)eps);
    for (int i = 0; i < N; i++) {
        double val = (double)x[i] * inv_rms;
        if (gamma) val *= (double)gamma[i];
        y[i] = (float)val;
    }
}

void layernorm_ref_f64(const float *x, float *y, const float *gamma,
                       const float *beta, float eps, int N) {
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < N; i++) {
        sum += (double)x[i];
        sum_sq += (double)x[i] * (double)x[i];
    }
    double mean = sum / N;
    double var = sum_sq / N - mean * mean;
    double inv_std = 1.0 / sqrt(var + (double)eps);
    for (int i = 0; i < N; i++) {
        double xhat = ((double)x[i] - mean) * inv_std;
        double val = xhat;
        if (gamma) val *= (double)gamma[i];
        if (beta) val += (double)beta[i];
        y[i] = (float)val;
    }
}


/* ════════════════════════════════════════════════════════════════
 * Scalar baselines
 * ════════════════════════════════════════════════════════════════ */

void rmsnorm_scalar_f32(const float *x, float *y, const float *gamma,
                        float eps, int N) {
    float sum_sq = 0.0f;
    for (int i = 0; i < N; i++)
        sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)N + eps);
    for (int i = 0; i < N; i++) {
        float val = x[i] * inv_rms;
        if (gamma) val *= gamma[i];
        y[i] = val;
    }
}

void layernorm_scalar_f32(const float *x, float *y, const float *gamma,
                          const float *beta, float eps, int N) {
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += x[i];
        sum_sq += x[i] * x[i];
    }
    float mean = sum / (float)N;
    float var = sum_sq / (float)N - mean * mean;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < N; i++) {
        float xhat = (x[i] - mean) * inv_std;
        float val = xhat;
        if (gamma) val *= gamma[i];
        if (beta) val += beta[i];
        y[i] = val;
    }
}
