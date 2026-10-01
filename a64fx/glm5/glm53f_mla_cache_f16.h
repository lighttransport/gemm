#ifndef GLM53F_MLA_CACHE_F16_H
#define GLM53F_MLA_CACHE_F16_H
#include <arm_sve.h>
#include <stdint.h>

/* Native MLA already rounds every loaded latent to FP16. Store that exact
 * derived view once, preserving the FP32 cache used by snapshots/reference. */
static inline void glm53f_mla_cache_f16_store(uint16_t *out,
        const float *input, int count) {
    for (int i = 0; i < count; i += (int)svcntw()) {
        svbool_t p = svwhilelt_b32(i, count);
        svfloat16_t half = svcvt_f16_f32_x(p, svld1_f32(p, input + i));
        svst1h_u32(p, out + i, svreinterpret_u32_f16(half));
    }
}
static inline svfloat32_t glm53f_mla_cache_f16_load(svbool_t p,
        const uint16_t *input) {
    return svcvt_f32_f16_x(p, svreinterpret_f16_u32(svld1uh_u32(p, input)));
}
#endif
