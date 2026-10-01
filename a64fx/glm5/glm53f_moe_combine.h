#ifndef GLM53F_MOE_COMBINE_H
#define GLM53F_MOE_COMBINE_H
#include <arm_sve.h>
/* Each position retains top-k order and adds the shared expert last. Hoist
 * route validity out of the hidden-dimension loop; no reduction reassociation. */
static inline void glm53f_moe_combine_rows(float *shared, const float *const route[8],
        const float weight[8], int begin, int end) {
    for (int i = begin; i < end; i += (int)svcntw()) {
        svbool_t p = svwhilelt_b32(i, end);
        svfloat32_t routed = svdup_f32(0);
        for (int k = 0; k < 8; ++k)
            if (route[k]) routed = svmla_n_f32_x(p, routed, svld1_f32(p, route[k] + i), weight[k]);
        svst1_f32(p, shared + i, svadd_f32_x(p, svld1_f32(p, shared + i), routed));
    }
}
#endif
