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

#if defined(__ARM_FEATURE_SVE)
/* Keep each score's final reduction independent of the four-key outer loop.
 * Fast-math must not vectorize across keys and change its head reduction. */
__attribute__((noinline)) static float glm53f_index_score_finish_f32(
        const float *dot, const float *weight) {
    float score = 0;
    for (int h = 0; h < 32; ++h)
        if (dot[h] > 0) score += weight[h] * dot[h] / sqrtf(128.f * 32.f);
    return score;
}
#endif

/* Four independent pool keys share query loads. Accumulation lanes, FADDV
 * and the final head-order sum are identical to the one-key kernel. */
static inline void glm53f_index_score_f32_keys4(float *out, const float *q,
        const float *weight, const float *keys, size_t stride, int count) {
#if defined(__ARM_FEATURE_SVE)
    float dot[4][32];
    const float *k0 = keys, *k1 = keys + (count > 1 ? stride : 0);
    const float *k2 = keys + (count > 2 ? 2 * stride : 0);
    const float *k3 = keys + (count > 3 ? 3 * stride : 0);
    const int vl = (int)svcntw();
    for (int h = 0; h < 32; h += 4) {
        svfloat32_t a00 = svdup_f32(0), a01 = svdup_f32(0), a02 = svdup_f32(0), a03 = svdup_f32(0);
        svfloat32_t a10 = svdup_f32(0), a11 = svdup_f32(0), a12 = svdup_f32(0), a13 = svdup_f32(0);
        svfloat32_t a20 = svdup_f32(0), a21 = svdup_f32(0), a22 = svdup_f32(0), a23 = svdup_f32(0);
        svfloat32_t a30 = svdup_f32(0), a31 = svdup_f32(0), a32 = svdup_f32(0), a33 = svdup_f32(0);
        for (int i = 0; i < 128; i += vl) {
            svbool_t p = svwhilelt_b32(i, 128);
            svfloat32_t v0 = svld1_f32(p, k0 + i), v1 = svld1_f32(p, k1 + i);
            svfloat32_t v2 = svld1_f32(p, k2 + i), v3 = svld1_f32(p, k3 + i);
            svfloat32_t q0 = svld1_f32(p, q + (size_t)(h + 0) * 128 + i);
            a00 = svmla_f32_x(p, a00, q0, v0);
            a10 = svmla_f32_x(p, a10, q0, v1);
            a20 = svmla_f32_x(p, a20, q0, v2);
            a30 = svmla_f32_x(p, a30, q0, v3);
            svfloat32_t q1 = svld1_f32(p, q + (size_t)(h + 1) * 128 + i);
            a01 = svmla_f32_x(p, a01, q1, v0);
            a11 = svmla_f32_x(p, a11, q1, v1);
            a21 = svmla_f32_x(p, a21, q1, v2);
            a31 = svmla_f32_x(p, a31, q1, v3);
            svfloat32_t q2 = svld1_f32(p, q + (size_t)(h + 2) * 128 + i);
            a02 = svmla_f32_x(p, a02, q2, v0);
            a12 = svmla_f32_x(p, a12, q2, v1);
            a22 = svmla_f32_x(p, a22, q2, v2);
            a32 = svmla_f32_x(p, a32, q2, v3);
            svfloat32_t q3 = svld1_f32(p, q + (size_t)(h + 3) * 128 + i);
            a03 = svmla_f32_x(p, a03, q3, v0);
            a13 = svmla_f32_x(p, a13, q3, v1);
            a23 = svmla_f32_x(p, a23, q3, v2);
            a33 = svmla_f32_x(p, a33, q3, v3);
        }
        svbool_t p = svptrue_b32();
        dot[0][h + 0] = svaddv_f32(p, a00);
        dot[0][h + 1] = svaddv_f32(p, a01);
        dot[0][h + 2] = svaddv_f32(p, a02);
        dot[0][h + 3] = svaddv_f32(p, a03);
        dot[1][h + 0] = svaddv_f32(p, a10);
        dot[1][h + 1] = svaddv_f32(p, a11);
        dot[1][h + 2] = svaddv_f32(p, a12);
        dot[1][h + 3] = svaddv_f32(p, a13);
        dot[2][h + 0] = svaddv_f32(p, a20);
        dot[2][h + 1] = svaddv_f32(p, a21);
        dot[2][h + 2] = svaddv_f32(p, a22);
        dot[2][h + 3] = svaddv_f32(p, a23);
        dot[3][h + 0] = svaddv_f32(p, a30);
        dot[3][h + 1] = svaddv_f32(p, a31);
        dot[3][h + 2] = svaddv_f32(p, a32);
        dot[3][h + 3] = svaddv_f32(p, a33);
    }
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (int k = 0; k < count; ++k)
        out[k] = glm53f_index_score_finish_f32(dot[k], weight);
#else
    for (int k = 0; k < count; ++k)
        out[k] = glm53f_index_score_f32_heads(q, weight, keys + k * stride);
#endif
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
