#ifndef Q38D_REDUCE_H
#define Q38D_REDUCE_H
#include <arm_sve.h>
#include <math.h>
/* Like the scalar comparison loop, ignore NaNs and return -Inf for n=0.
 * Max needs no reassociation of floating-point sums, so finite values retain
 * the same softmax shift. Zero ties may select a different sign of zero. */
static inline float q38d_score_max(const float *s, int n, int vector) {
    if (!vector) {
        float m = -INFINITY;
        for (int i = 0; i < n; i++) if (s[i] > m) m = s[i];
        return m;
    }
    svfloat32_t m = svdup_n_f32(-INFINITY);
    for (int i = 0; i < n; i += svcntw()) {
        svbool_t p = svwhilelt_b32(i, n);
        m = svmaxnm_f32_m(p, m, svld1_f32(p, s + i));
    }
    return svmaxnmv_f32(svptrue_b32(), m);
}
#endif
