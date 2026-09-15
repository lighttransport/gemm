#ifndef GLM53F_KDA_PREFILL_H
#define GLM53F_KDA_PREFILL_H
#include <stddef.h>
#include <math.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/* One task owns a complete chronological trajectory of sixteen value columns.
 * Packed state is [key128][value16]; each task begins on a 256-byte boundary.
 * q/k and exp(log_decay) have already been computed once per token/head.
 * Canonical state is packed/restored by the caller, never exposed to decode. */
static inline void glm53f_kda_column_tile(float *state, float *output,
        const float *q, const float *k, const float *v, const float *decay,
        const float *beta, int tokens, int qstride, int bstride) {
    const float scale = 1.0f / sqrtf(128.0f);
    for (int t = 0; t < tokens; ++t) {
#if defined(__ARM_FEATURE_SVE)
        for (int j = 0; j < 16; j += (int)svcntw()) {
            svbool_t pg = svwhilelt_b32(j, 16);
            svfloat32_t work = svdup_f32(0.0f), out = svdup_f32(0.0f);
            for (int d = 0; d < 128; ++d) {
                float *row = state + d * 16 + j;
                svfloat32_t s = svmul_n_f32_x(pg, svld1(pg, row), decay[d]);
                svst1(pg, row, s);
                work = svmla_n_f32_x(pg, work, s, k[d]);
            }
            work = svmul_n_f32_x(pg, svsub_f32_x(pg, svld1(pg, v + j), work), *beta);
            for (int d = 0; d < 128; ++d) {
                float *row = state + d * 16 + j;
                svfloat32_t s = svmla_n_f32_x(pg, svld1(pg, row), work, k[d]);
                svst1(pg, row, s);
                out = svmla_n_f32_x(pg, out, s, q[d] * scale);
            }
            svst1(pg, output + j, out);
        }
#else
        float work[16] = {0}, out[16] = {0};
        for (int d = 0; d < 128; ++d)
            for (int j = 0; j < 16; ++j) {
                state[d * 16 + j] *= decay[d];
                work[j] += state[d * 16 + j] * k[d];
            }
        for (int j = 0; j < 16; ++j) work[j] = (v[j] - work[j]) * *beta;
        for (int d = 0; d < 128; ++d)
            for (int j = 0; j < 16; ++j) {
                state[d * 16 + j] += k[d] * work[j];
                out[j] += state[d * 16 + j] * (q[d] * scale);
            }
        for (int j = 0; j < 16; ++j) output[j] = out[j];
#endif
        q += qstride; k += qstride; v += qstride; decay += qstride;
        output += qstride; beta += bstride;
    }
}
#endif
