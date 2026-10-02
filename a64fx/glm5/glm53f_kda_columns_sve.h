#ifndef GLM53F_KDA_COLUMNS_SVE_H
#define GLM53F_KDA_COLUMNS_SVE_H
#include <arm_sve.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
/* Isolate the same scalar libm call used by the original streamed recurrence.
 * A vector exp replacement would require a separate numerical contract. */
static __attribute__((noinline)) float glm53f_kda_scalar_factor(float x) { return expf(x); }
/* Canonical [key128][value128] state; own one aligned64-value half.
 * Four lane accumulators keep each chronological FMA chain unchanged.
 * The caller prepares factors with its existing scalar/decode or prefill
 * recipe, so this kernel changes neither exponentials nor normalization. */
static __attribute__((noinline)) void glm53f_kda_columns64_sve(float *state, float *out,
        const float *q, const float *k, const float *v, const float *factor, float beta) {
    if (svcntw() != 16) abort();
    const svbool_t pg = svptrue_b32();
    const float scale = 1.0f / sqrtf(128.0f);
    svfloat32_t w0 = svdup_f32(0), w1 = w0, w2 = w0, w3 = w0;
    for (int d = 0; d < 128; ++d) {
        float *row = state + (size_t)d * 128;
#define DECAY(N) do { \
        svfloat32_t s = svmul_n_f32_x(pg, svld1_f32(pg, row + 16 * N), factor[d]); \
        svst1_f32(pg, row + 16 * N, s); \
        w##N = svmla_n_f32_x(pg, w##N, s, k[d]); \
    } while (0)
        DECAY(0); DECAY(1); DECAY(2); DECAY(3);
#undef DECAY
    }
#define DELTA(N) w##N = svmul_n_f32_x(pg, svsub_f32_x(pg, svld1_f32(pg, v + 16 * N), w##N), beta)
    DELTA(0); DELTA(1); DELTA(2); DELTA(3);
#undef DELTA
    svfloat32_t y0 = svdup_f32(0), y1 = y0, y2 = y0, y3 = y0;
    for (int d = 0; d < 128; ++d) {
        float *row = state + (size_t)d * 128;
        const float kd = k[d], qd = q[d] * scale;
#define UPDATE(N) do { \
        svfloat32_t s = svmla_n_f32_x(pg, svld1_f32(pg, row + 16 * N), w##N, kd); \
        svst1_f32(pg, row + 16 * N, s); \
        y##N = svmla_n_f32_x(pg, y##N, s, qd); \
    } while (0)
        UPDATE(0); UPDATE(1); UPDATE(2); UPDATE(3);
#undef UPDATE
    }
    svst1_f32(pg, out, y0); svst1_f32(pg, out + 16, y1);
    svst1_f32(pg, out + 32, y2); svst1_f32(pg, out + 48, y3);
}

#endif
