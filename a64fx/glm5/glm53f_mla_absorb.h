#ifndef GLM53F_MLA_ABSORB_H
#define GLM53F_MLA_ABSORB_H
#include <arm_sve.h>
#include "glm53f_mla_cache_f16.h"
#include <stddef.h>
#include <stdint.h>

/* One 64-column tile of the BF16 absorbed query. Keep each lane's j-order;
 * four register accumulators replace the repeated scratch load/store.
 * Caller requires sixteen FP32 lanes and distinct output/query storage. */
static inline void glm53f_mla_absorb64(float *out, const uint16_t *weight,
        const float *query, int begin) {
    svbool_t p = svptrue_b32();
    svfloat32_t a = svdup_f32(0), b = a, c = a, d = a;
    for (int j = 0; j < 256; ++j) {
        const uint16_t *w = weight + (size_t)j * 512 + begin;
        float x = query[j] / 16.0f;
#define MLA_ABSORB_TILE(A, OFFSET) A = svmla_n_f32_x(p, A, svreinterpret_f32_u32( \
        svlsl_n_u32_x(p, svld1uh_u32(p, w + OFFSET), 16)), x)
        MLA_ABSORB_TILE(a, 0); MLA_ABSORB_TILE(b, 16);
        MLA_ABSORB_TILE(c, 32); MLA_ABSORB_TILE(d, 48);
#undef MLA_ABSORB_TILE
    }
    svst1_f32(p, out + begin, a); svst1_f32(p, out + begin + 16, b);
    svst1_f32(p, out + begin + 32, c); svst1_f32(p, out + begin + 48, d);
}
/* Retain selected-key order while keeping the value tile in registers.
 * Packed decode latents are already FP16-rounded; indexed input also supports
 * the reference path. Caller requires the A64FX sixteen-lane vector length. */
static inline void glm53f_mla_value64(float *out, const float *latent,
        const float *probability, const float *sum, const int *selected,
        int count, int begin) {
    svbool_t p = svptrue_b32();
    svfloat32_t a = svdup_f32(0), b = a, c = a, d = a;
    for (int t = 0; t < count; ++t) {
        int row = selected ? selected[t] : t;
        const float *z = latent + (size_t)row * 512 + begin;
        /* The scratch loop can alias sum, so its fast-math division is not
         * hoisted into a reciprocal. Preserve that rounding here too. */
        float x = probability[t] / *(const volatile float *)sum;
        a = svmla_n_f32_x(p, a, svld1_f32(p, z), x);
        b = svmla_n_f32_x(p, b, svld1_f32(p, z + 16), x);
        c = svmla_n_f32_x(p, c, svld1_f32(p, z + 32), x);
        d = svmla_n_f32_x(p, d, svld1_f32(p, z + 48), x);
    }
    svst1_f32(p, out + begin, a); svst1_f32(p, out + begin + 16, b);
    svst1_f32(p, out + begin + 32, c); svst1_f32(p, out + begin + 48, d);
}
/* Identical lane/key order over the derived half cache. */
static inline void glm53f_mla_value64_f16(float *out, const uint16_t *latent,
        const float *probability, const float *sum, const int *selected,
        int count, int begin) {
    svbool_t p = svptrue_b32();
    svfloat32_t a = svdup_f32(0), b = a, c = a, d = a;
    for (int t = 0; t < count; ++t) {
        int row = selected ? selected[t] : t;
        const uint16_t *z = latent + (size_t)row * 512 + begin;
        /* The scratch loop can alias sum, so its fast-math division is not
         * hoisted into a reciprocal. Preserve that rounding here too. */
        float x = probability[t] / *(const volatile float *)sum;
        a = svmla_n_f32_x(p, a, glm53f_mla_cache_f16_load(p, z), x);
        b = svmla_n_f32_x(p, b, glm53f_mla_cache_f16_load(p, z + 16), x);
        c = svmla_n_f32_x(p, c, glm53f_mla_cache_f16_load(p, z + 32), x);
        d = svmla_n_f32_x(p, d, glm53f_mla_cache_f16_load(p, z + 48), x);
    }
    svst1_f32(p, out + begin, a); svst1_f32(p, out + begin + 16, b);
    svst1_f32(p, out + begin + 32, c); svst1_f32(p, out + begin + 48, d);
}
#endif
