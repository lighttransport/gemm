#ifndef GLM53F_BF16_ROWS_H
#define GLM53F_BF16_ROWS_H
/* Small-batch BF16 row dots shared by verify-batch decode paths. */
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

/* Four bf16 rows against K token inputs. Each (row, token) keeps the eight-row b16dot8/router_dot8
 * lane-wise FMA chain and final reduction, so outputs are bit-identical; the
 * weight vector is loaded once per K tokens. y[j * ldy + r], x[j * ldx + i]. */
static inline __attribute__((always_inline)) void glm53f_bf16_dot4xk(float *y, size_t ldy,
        const uint16_t *w, const float *x, size_t ldx, int n, const int K) {
    svfloat32_t a00 = svdup_f32(0), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00,
                a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    const int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        svbool_t p = svwhilelt_b32(i, n);
        svfloat32_t x0 = svld1(p, x + i), x1 = x0, x2 = x0, x3 = x0;
        if (K > 1) x1 = svld1(p, x + ldx + i);
        if (K > 2) x2 = svld1(p, x + 2 * ldx + i);
        if (K > 3) x3 = svld1(p, x + 3 * ldx + i);
#define R(N, A0, A1, A2, A3) do { svfloat32_t z = svreinterpret_f32_u32(svlsl_n_u32_x(p, \
            svld1uh_u32(p, w + (size_t)(N) * n + i), 16)); A0 = svmla_x(p, A0, z, x0); \
            if (K > 1) A1 = svmla_x(p, A1, z, x1); if (K > 2) A2 = svmla_x(p, A2, z, x2); \
            if (K > 3) A3 = svmla_x(p, A3, z, x3); } while (0)
        R(0, a00, a01, a02, a03); R(1, a10, a11, a12, a13);
        R(2, a20, a21, a22, a23); R(3, a30, a31, a32, a33);
#undef R
    }
    svbool_t p = svptrue_b32();
#define S(J, B0, B1, B2, B3) do { float *o = y + (size_t)(J) * ldy; o[0] = svaddv_f32(p, B0); \
        o[1] = svaddv_f32(p, B1); o[2] = svaddv_f32(p, B2); o[3] = svaddv_f32(p, B3); } while (0)
    S(0, a00, a10, a20, a30);
    if (K > 1) S(1, a01, a11, a21, a31);
    if (K > 2) S(2, a02, a12, a22, a32);
    if (K > 3) S(3, a03, a13, a23, a33);
#undef S
}
static inline void glm53f_bf16_dot4x(float *y, size_t ldy, const uint16_t *w, const float *x,
                     size_t ldx, int n, int k) {
    switch (k) {
    case 1: glm53f_bf16_dot4xk(y, ldy, w, x, ldx, n, 1); break;
    case 2: glm53f_bf16_dot4xk(y, ldy, w, x, ldx, n, 2); break;
    case 3: glm53f_bf16_dot4xk(y, ldy, w, x, ldx, n, 3); break;
    default: glm53f_bf16_dot4xk(y, ldy, w, x, ldx, n, 4); break;
    }
}
#endif
