/* v0 kernel cores: the production GLM-5.3-Flash decode inner loops,
 * unchanged apart from the row-range interface.  Sources:
 *   gk_q6_k_v0/gk_q5_k_v0/gk_q4_k_v0  <- glm53f_iq_bridge.c q{6,5,4}_k_q8_row
 *   gk_q8_0r_v0                       <- glm53f_iq_bridge.c q8_0r_rows
 *   gk_f32_v0                         <- glm53f_target_head_12n.c dot_f32
 * Keep these byte-for-byte equivalent in arithmetic so the harness baseline
 * is the production baseline; optimized variants live in other files. */
#include "glm53f_kern.h"
#include <arm_sve.h>

static inline float gk_fp16(uint16_t h) {
    __fp16 v;
    __builtin_memcpy(&v, &h, sizeof(v));
    return (float)v;
}

static inline void gk_scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                   uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

size_t gk_row_bytes_q8_0r(int c) { return (size_t)c + (size_t)(c / 32) * 4; }
size_t gk_row_bytes_q4_k(int c) { return (size_t)(c / 256) * sizeof(gk_block_q4_K); }
size_t gk_row_bytes_q5_k(int c) { return (size_t)(c / 256) * sizeof(gk_block_q5_K); }
size_t gk_row_bytes_q6_k(int c) { return (size_t)(c / 256) * sizeof(gk_block_q6_K); }
size_t gk_row_bytes_f32(int c) { return (size_t)c * 4; }

static float q6_k_q8_row(const gk_block_q6_K *w, const gk_q8k_block *x,
                         int blocks) {
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svbool_t p4 = svwhilelt_b32(0, 4), p32bytes = svwhilelt_b8(0, 32);
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const float ds = gk_fp16(w[b].d) * x[b].d;
        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql = w[b].ql + half * 64;
            const svuint8_t qh = svld1_u8(p32bytes, w[b].qh + half * 32);
            const int8_t *sc = w[b].scales + half * 8;
            for (int g = 0; g < 4; ++g) {
                svuint8_t q = svld1_u8(p32bytes, ql + ((g & 1) ? 32 : 0));
                q = (g >= 2) ? svlsr_n_u8_x(p8, q, 4) : svand_n_u8_x(p8, q, 15);
                svuint8_t hi = svand_n_u8_x(p8, svlsr_n_u8_x(p8, qh, 2 * g), 3);
                svint8_t qs = svreinterpret_s8_u8(svsub_n_u8_x(
                    p8, svorr_u8_x(p8, q, svlsl_n_u8_x(p8, hi, 4)), 32));
                const svint8_t xv = svld1_s8(p32bytes,
                    x[b].q + half * 128 + g * 32);
                const svint32_t dot = svdot_s32(svdup_s32(0), qs, xv);
                const svint32_t scale = svsel_s32(p4, svdup_s32(sc[2 * g]),
                                                       svdup_s32(sc[2 * g + 1]));
                acc = svmla_n_f32_x(p32, acc,
                    svcvt_f32_s32_x(p32, svmul_s32_x(p32, dot, scale)), ds);
            }
        }
    }
    return svaddv_f32(p32, acc);
}

static float q5_k_q8_row(const gk_block_q5_K *w, const gk_q8k_block *x,
                         int blocks) {
    const svbool_t p8 = svptrue_b8();
    const svbool_t p32 = svptrue_b32();
    const svbool_t p32bytes = svwhilelt_b8(0, 32);
    const svbool_t lo8 = svwhilelt_b32(0, 8);
    const svint8_t ones = svdup_s8(1);
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const float d = gk_fp16(w[b].d);
        const float dm = gk_fp16(w[b].dmin);
        svint32_t dots = svdup_s32(0), mins = svdup_s32(0);
        int is = 0;
        uint8_t bit0 = 1, bit1 = 2;
        for (int g = 0; g < 256; g += 64, is += 2, bit0 <<= 2, bit1 <<= 2) {
            uint8_t sc0, min0, sc1, min1;
            gk_scale_min_k4(is, w[b].scales, &sc0, &min0);
            gk_scale_min_k4(is + 1, w[b].scales, &sc1, &min1);
            const svuint8_t packed = svld1_u8(p32bytes, w[b].qs + (g >> 1));
            const svuint8_t high = svld1_u8(p32bytes, w[b].qh);
            svuint8_t uq0 = svand_n_u8_x(p8, packed, 15);
            svuint8_t uq1 = svlsr_n_u8_x(p8, packed, 4);
            uq0 = svadd_u8_x(p8, uq0, svsel_u8(svcmpne_n_u8(p32bytes,
                svand_n_u8_x(p8, high, bit0), 0), svdup_u8(16), svdup_u8(0)));
            uq1 = svadd_u8_x(p8, uq1, svsel_u8(svcmpne_n_u8(p32bytes,
                svand_n_u8_x(p8, high, bit1), 0), svdup_u8(16), svdup_u8(0)));
            const svint8_t qv = svreinterpret_s8_u8(svsplice_u8(p32bytes, uq0, uq1));
            const svint8_t xv = svld1_s8(p8, x[b].q + g);
            const svint32_t dot = svdot_s32(svdup_s32(0), qv, xv);
            const svint32_t sx = svdot_s32(svdup_s32(0), ones, xv);
            const svint32_t scv = svsel_s32(lo8, svdup_s32(sc0), svdup_s32(sc1));
            const svint32_t mnv = svsel_s32(lo8, svdup_s32(min0), svdup_s32(min1));
            dots = svmla_s32_x(p32, dots, dot, scv);
            mins = svmla_s32_x(p32, mins, sx, mnv);
        }
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, dots), x[b].d * d);
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, mins), -x[b].d * dm);
    }
    return svaddv_f32(p32, acc);
}

static float q4_k_q8_row(const gk_block_q4_K *w, const gk_q8k_block *x,
                         int blocks) {
    const svbool_t p8 = svptrue_b8();
    const svbool_t p32 = svptrue_b32();
    const svint8_t ones = svdup_s8(1);
    const svbool_t p32bytes = svwhilelt_b8(0, 32);
    const svbool_t lo8 = svwhilelt_b32(0, 8);
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const float d = gk_fp16(w[b].d);
        const float dm = gk_fp16(w[b].dmin);
        svint32_t dots = svdup_s32(0), mins = svdup_s32(0);
        int is = 0;
        for (int g = 0; g < 256; g += 64, is += 2) {
            uint8_t sc0, min0, sc1, min1;
            gk_scale_min_k4(is, w[b].scales, &sc0, &min0);
            gk_scale_min_k4(is + 1, w[b].scales, &sc1, &min1);
            const svuint8_t packed = svld1_u8(p32bytes, w[b].qs + (g >> 1));
            const svint8_t q0 = svreinterpret_s8_u8(svand_n_u8_x(p8, packed, 15));
            const svint8_t q1 = svreinterpret_s8_u8(svlsr_n_u8_x(p8, packed, 4));
            const svint8_t qv = svsplice_s8(p32bytes, q0, q1);
            const svint8_t xv = svld1_s8(p8, x[b].q + g);
            const svint32_t dot = svdot_s32(svdup_s32(0), qv, xv);
            const svint32_t sx = svdot_s32(svdup_s32(0), ones, xv);
            const svint32_t scv = svsel_s32(lo8, svdup_s32(sc0), svdup_s32(sc1));
            const svint32_t mnv = svsel_s32(lo8, svdup_s32(min0), svdup_s32(min1));
            dots = svmla_s32_x(p32, dots, dot, scv);
            mins = svmla_s32_x(p32, mins, sx, mnv);
        }
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, dots), x[b].d * d);
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, mins), -x[b].d * dm);
    }
    return svaddv_f32(p32, acc);
}

void gk_q4_k_v0(const gk_mv *m, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        m->y[r] = q4_k_q8_row((const gk_block_q4_K *)(m->w + (size_t)r * m->row_bytes),
                              m->a->q8k, m->columns / 256);
}
void gk_q5_k_v0(const gk_mv *m, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        m->y[r] = q5_k_q8_row((const gk_block_q5_K *)(m->w + (size_t)r * m->row_bytes),
                              m->a->q8k, m->columns / 256);
}
void gk_q6_k_v0(const gk_mv *m, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        m->y[r] = q6_k_q8_row((const gk_block_q6_K *)(m->w + (size_t)r * m->row_bytes),
                              m->a->q8k, m->columns / 256);
}

/* Up to four repacked rows share each activation vector. */
static inline void q8_0r_rows(float *out, const uint8_t *w, size_t rb,
                              int nrows, const gk_act *a) {
    const int columns = a->columns, pairs = columns / 64;
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svbool_t lo8 = svptrue_pat_b32(SV_VL8);
    const int8_t *q[4];
    const float *d[4];
    for (int r = 0; r < 4; ++r) {
        const int rr = r < nrows ? r : 0;
        q[r] = (const int8_t *)(w + (size_t)rr * rb);
        d[r] = (const float *)(w + (size_t)rr * rb + columns);
    }
    svfloat32_t acc0 = svdup_f32(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (int k = 0; k < pairs; ++k) {
        const svint8_t xv = svld1_s8(p8, a->xq + 64 * k);
        const svfloat32_t xs = svld1_f32(p32, a->xpat + 16 * k);
#define GLM53F_Q8R_ROW(R, ACC) do { \
        const svint8_t wv = svld1_s8(p8, q[R] + 64 * k); \
        const svfloat32_t ws = svsel_f32(lo8, svdup_f32(d[R][2 * k]), \
                                         svdup_f32(d[R][2 * k + 1])); \
        ACC = svmla_f32_m(p32, ACC, \
            svcvt_f32_s32_x(p32, svdot_s32(svdup_s32(0), wv, xv)), \
            svmul_f32_x(p32, ws, xs)); \
    } while (0)
        GLM53F_Q8R_ROW(0, acc0);
        if (nrows > 1) GLM53F_Q8R_ROW(1, acc1);
        if (nrows > 2) GLM53F_Q8R_ROW(2, acc2);
        if (nrows > 3) GLM53F_Q8R_ROW(3, acc3);
#undef GLM53F_Q8R_ROW
    }
    out[0] = svaddv_f32(p32, acc0);
    if (nrows > 1) out[1] = svaddv_f32(p32, acc1);
    if (nrows > 2) out[2] = svaddv_f32(p32, acc2);
    if (nrows > 3) out[3] = svaddv_f32(p32, acc3);
}

void gk_q8_0r_v0(const gk_mv *m, int r0, int r1) {
    for (int r = r0; r < r1; r += 4) {
        const int n = r1 - r < 4 ? r1 - r : 4;
        q8_0r_rows(m->y + r, m->w + (size_t)r * m->row_bytes, m->row_bytes, n,
                   m->a);
    }
}

static inline float dot_f32(const float *w, const float *x, int n) {
    svfloat32_t a = svdup_f32(0);
    int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        svbool_t p = svwhilelt_b32(i, n);
        a = svmla_x(p, a, svld1(p, w + i), svld1(p, x + i));
    }
    return svaddv_f32(svptrue_b32(), a);
}

void gk_f32_v0(const gk_mv *m, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        m->y[r] = dot_f32((const float *)(m->w + (size_t)r * m->row_bytes),
                          m->a->xf, m->columns);
}
