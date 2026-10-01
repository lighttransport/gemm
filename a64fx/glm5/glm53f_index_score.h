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

/* Long-context decode uses one FP32 SVE accumulator per head. Interleave
 * eight heads to hide FMLA latency while reusing each loaded key vector;
 * preserve lane accumulation, FADDV, and the final head-order sum. */
static inline float glm53f_index_score_f32_heads(const float *q,
        const float *head_weight, const float *key) {
    float dot[32];
#if defined(__ARM_FEATURE_SVE)
    const int vl = (int)svcntw();
    for (int h = 0; h < 32; h += 8) {
        svfloat32_t a = svdup_f32(0), b = a, c = a, d = a;
        svfloat32_t e = a, f = a, g = a, j = a;
        for (int i = 0; i < 128; i += vl) {
            svbool_t p = svwhilelt_b32(i, 128);
            svfloat32_t k = svld1_f32(p, key + i);
#define INDEX_ACC(N, A) A = svmla_f32_x(p, A, svld1_f32(p, q + (size_t)(h + N) * 128 + i), k)
            INDEX_ACC(0, a); INDEX_ACC(1, b); INDEX_ACC(2, c); INDEX_ACC(3, d);
            INDEX_ACC(4, e); INDEX_ACC(5, f); INDEX_ACC(6, g); INDEX_ACC(7, j);
#undef INDEX_ACC
        }
        svbool_t p = svptrue_b32();
        dot[h] = svaddv_f32(p, a); dot[h + 1] = svaddv_f32(p, b);
        dot[h + 2] = svaddv_f32(p, c); dot[h + 3] = svaddv_f32(p, d);
        dot[h + 4] = svaddv_f32(p, e); dot[h + 5] = svaddv_f32(p, f);
        dot[h + 6] = svaddv_f32(p, g); dot[h + 7] = svaddv_f32(p, j);
    }
#else
    for (int h = 0; h < 32; ++h) {
        dot[h] = 0;
        for (int i = 0; i < 128; ++i) dot[h] = fmaf(q[(size_t)h * 128 + i], key[i], dot[h]);
    }
#endif
    float score = 0;
    for (int h = 0; h < 32; ++h)
        if (dot[h] > 0) score += head_weight[h] * dot[h] / sqrtf(128.f * 32.f);
    return score;
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
