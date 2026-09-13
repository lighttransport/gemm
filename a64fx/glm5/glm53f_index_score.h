/* Exact index scores: SIMD lanes are heads, never reduction dimensions.
 * Transpose the 32x128 query once per decode step. Each head still visits
 * dimensions 0..127 in order, followed by the original head-order reduction.
 */
#ifndef GLM53F_INDEX_SCORE_H
#define GLM53F_INDEX_SCORE_H
#include <math.h>
#include <stddef.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

static inline void glm53f_index_query_transpose(double *qt, const float *q) {
    for (int d = 0; d < 128; ++d)
        for (int h = 0; h < 32; ++h) qt[d * 32 + h] = q[h * 128 + d];
}

static inline float glm53f_index_score_transposed(const double *qt,
        const float *head_weight, const float *key) {
    double dot[32] = {0};
#if defined(__ARM_FEATURE_SVE)
    if (svcntd() == 8) {
        svfloat64_t a = svdup_f64(0), b = svdup_f64(0);
        svfloat64_t c = svdup_f64(0), e = svdup_f64(0);
        svbool_t p = svptrue_b64();
        for (int d = 0; d < 128; ++d) {
            const double *q = qt + (size_t)d * 32;
            double k = key[d];
            a = svmla_n_f64_x(p, a, svld1(p, q), k);
            b = svmla_n_f64_x(p, b, svld1(p, q + 8), k);
            c = svmla_n_f64_x(p, c, svld1(p, q + 16), k);
            e = svmla_n_f64_x(p, e, svld1(p, q + 24), k);
        }
        svst1(p, dot, a); svst1(p, dot + 8, b);
        svst1(p, dot + 16, c); svst1(p, dot + 24, e);
    } else
#endif
    {
        for (int d = 0; d < 128; ++d)
            for (int h = 0; h < 32; ++h) dot[h] += qt[d * 32 + h] * (double)key[d];
    }
    float score = 0;
    for (int h = 0; h < 32; ++h)
        if (dot[h] > 0) score += head_weight[h] *
            (float)(dot[h] / sqrt(128.0)) / sqrtf(32.0f);
    return score;
}
#endif
