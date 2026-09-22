#ifndef A64FX_KV_F16_SVE_H
#define A64FX_KV_F16_SVE_H

#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

/* Exact widening for finite IEEE half values, including signed subnormals.
 * Keep the conversion opaque to loop vectorization: QK's scalar reduction
 * must retain its existing accumulation order. No FPCR flags are changed. */
static inline float tf_kv_f16_to_f32_sve(uint16_t bits)
{
    __fp16 half;
    float result;
    memcpy(&half, &bits, sizeof(half));
    __asm__("fcvt %s0, %h1" : "=w"(result) : "w"(half));
    return result;
}

static inline float tf_kv_fma_f32_sve(float accum, float q, float k)
{
    __asm__("fmadd %s0, %s1, %s2, %s0"
            : "+w"(accum) : "w"(q), "w"(k));
    return accum;
}

/* Interleave independent positions, never dimensions of the same reduction.
 * Explicit scalar FMAs keep each score in ascending dimension order. */
static inline void tf_kv_f16_dot4_sve(float *scores, const float *query,
                                     const uint16_t *keys, int stride, int n)
{
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int d = 0; d < n; d++) {
        float q = query[d];
        float k0 = tf_kv_f16_to_f32_sve(keys[d]);
        float k1 = tf_kv_f16_to_f32_sve(keys[(size_t)stride + d]);
        float k2 = tf_kv_f16_to_f32_sve(keys[(size_t)2 * stride + d]);
        float k3 = tf_kv_f16_to_f32_sve(keys[(size_t)3 * stride + d]);
        s0 = tf_kv_fma_f32_sve(s0, q, k0);
        s1 = tf_kv_fma_f32_sve(s1, q, k1);
        s2 = tf_kv_fma_f32_sve(s2, q, k2);
        s3 = tf_kv_fma_f32_sve(s3, q, k3);
    }
    scores[0] = s0;
    scores[1] = s1;
    scores[2] = s2;
    scores[3] = s3;
}

/* Vectorize independent PV output dimensions, retaining sequence-order FMA
 * accumulation in each dimension. Predication handles partial vectors. */
static inline void tf_kv_f16_axpy_sve(float *dst, const uint16_t *values,
                                     float scale, int n)
{
    for (int d = 0; d < n; d += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(d, n);
        svuint32_t bits = svld1uh_u32(pg, values + d);
        svfloat32_t value = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(bits));
        svfloat32_t accum = svld1(pg, dst + d);
        svst1(pg, dst + d, svmla_n_f32_m(pg, accum, value, scale));
    }
}

#endif
