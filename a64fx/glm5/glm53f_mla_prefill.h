#ifndef GLM53F_MLA_PREFILL_H
#define GLM53F_MLA_PREFILL_H
#include <arm_sve.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Same per-lane FMA order, SVE horizontal dot and eight-shard value tree as
 * the established MLA paths. Register blocking changes only independent
 * columns/rows, not the selected-key order or any nonlinear operation. */
static inline int glm53f_mla_prefill_one(float *out, const float *q,
        const float *cache, const uint16_t *weight, const int *selected,
        int count, int shards) {
    enum { D = 512, Q = 256, V = 256 };
    if (count < 1 || count > 2051 || (shards != 1 && shards != 8)) return -1;
    float ql[D], va[D], score[2052], part[8 * D];
    int vl = (int)svcntw();
    svbool_t pg = svptrue_b32();
    /* A64FX has sixteen FP32 lanes. Fall back at the caller on other SVE VL. */
    if (vl != 16) return -1;
    for (int d = 0; d < D; d += 64) {
        svfloat32_t a = svdup_f32(0), b = a, c = a, e = a;
        for (int j = 0; j < Q; ++j) {
            const uint16_t *w = weight + (size_t)j * D + d;
            float x = q[j] / sqrtf((float)Q);
#define MLA_ABSORB(A, O) A = svmla_n_f32_x(pg, A, svreinterpret_f32_u32( \
            svlsl_n_u32_x(pg, svld1uh_u32(pg, w + O), 16)), x)
            MLA_ABSORB(a, 0); MLA_ABSORB(b, 16);
            MLA_ABSORB(c, 32); MLA_ABSORB(e, 48);
#undef MLA_ABSORB
        }
        svst1(pg, ql + d, a); svst1(pg, ql + d + 16, b);
        svst1(pg, ql + d + 32, c); svst1(pg, ql + d + 48, e);
    }
    float maximum = -INFINITY, sum = 0;
    for (int t = 0; t < count; ++t) {
        const float *z = cache + (size_t)selected[t] * D;
        svfloat32_t a = svdup_f32(0);
        for (int d = 0; d < D; d += vl)
            a = svmla_f32_x(pg, a, svld1(pg, ql + d), svld1(pg, z + d));
        score[t] = svaddv_f32(pg, a);
        if (score[t] > maximum) maximum = score[t];
    }
    for (int t = 0; t < count; ++t) {
        score[t] = expf(score[t] - maximum); sum += score[t];
    }
    for (int t = 0; t < count; ++t) score[t] /= sum;
    for (int s = 0; s < shards; ++s) {
        int begin = count * s / shards, end = count * (s + 1) / shards;
        for (int d = 0; d < D; d += 64) {
            svfloat32_t a = svdup_f32(0), b = a, c = a, e = a;
            for (int t = begin; t < end; ++t) {
                const float *z = cache + (size_t)selected[t] * D + d;
                float x = score[t];
                a = svmla_n_f32_x(pg, a, svld1(pg, z), x);
                b = svmla_n_f32_x(pg, b, svld1(pg, z + 16), x);
                c = svmla_n_f32_x(pg, c, svld1(pg, z + 32), x);
                e = svmla_n_f32_x(pg, e, svld1(pg, z + 48), x);
            }
            float *v = shards == 1 ? va + d : part + s * D + d;
            svst1(pg, v, a); svst1(pg, v + 16, b);
            svst1(pg, v + 32, c); svst1(pg, v + 48, e);
        }
    }
    if (shards == 8)
        for (int d = 0; d < D; ++d) {
            float sum_value = 0;
            for (int s = 0; s < 8; ++s) sum_value += part[s * D + d];
            va[d] = sum_value;
        }
    const uint16_t *wv = weight + (size_t)Q * D;
    for (int j = 0; j < V; j += 8) {
        svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
        svfloat32_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        for (int d = 0; d < D; d += vl) {
            svfloat32_t x = svld1(pg, va + d);
#define MLA_OUTPUT(N) a##N = svmla_f32_x(pg, a##N, svreinterpret_f32_u32( \
                svlsl_n_u32_x(pg, svld1uh_u32(pg, wv + (size_t)(j + N) * D + d), 16)), x)
            MLA_OUTPUT(0); MLA_OUTPUT(1); MLA_OUTPUT(2); MLA_OUTPUT(3);
            MLA_OUTPUT(4); MLA_OUTPUT(5); MLA_OUTPUT(6); MLA_OUTPUT(7);
#undef MLA_OUTPUT
        }
        out[j] = svaddv_f32(pg, a0); out[j+1] = svaddv_f32(pg, a1);
        out[j+2] = svaddv_f32(pg, a2); out[j+3] = svaddv_f32(pg, a3);
        out[j+4] = svaddv_f32(pg, a4); out[j+5] = svaddv_f32(pg, a5);
        out[j+6] = svaddv_f32(pg, a6); out[j+7] = svaddv_f32(pg, a7);
    }
    return 0;
}
#endif
