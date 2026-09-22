#ifndef A64FX_SWIGLU_SVE_H
#define A64FX_SWIGLU_SVE_H
#include <arm_sve.h>

/* FEXPA with a cubic residual correction and two reciprocal refinements.
 * This is an explicit approximation, including clamping exp2 input to +/-80. */
static inline void tf_swiglu_approx_sve(float *out, const float *gate,
                                       const float *up, int n)
{
    for (int i = 0; i < n; i += (int)svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t g = svld1(pg, gate + i);
        svfloat32_t x = svmul_n_f32_x(pg, g, -1.4426950408889634f);
        x = svmax_n_f32_x(pg, svmin_n_f32_x(pg, x, 80.0f), -80.0f);
        svfloat32_t z = svadd_n_f32_x(pg, x, 204927.0f);
        svfloat32_t r = svsub_f32_x(pg, x, svsub_n_f32_x(pg, z, 204927.0f));
        svfloat32_t corr = svmla_n_f32_x(pg, svdup_f32(0.2402265069591007f), r, 0.0555041086648216f);
        corr = svmla_f32_x(pg, svdup_f32(0.6931471805599453f), r, corr);
        corr = svmla_f32_x(pg, svdup_f32(1.0f), r, corr);
        svfloat32_t ex = svmul_f32_x(pg, svexpa_f32(svreinterpret_u32_f32(z)), corr);
        svfloat32_t den = svadd_n_f32_x(pg, ex, 1.0f);
        svfloat32_t inv = svrecpe_f32(den);
        inv = svmul_f32_x(pg, inv, svrecps_f32(den, inv));
        inv = svmul_f32_x(pg, inv, svrecps_f32(den, inv));
        svfloat32_t y = svmul_f32_x(pg, g, inv);
        if (up) y = svmul_f32_x(pg, y, svld1(pg, up + i));
        svst1(pg, out + i, y);
    }
}
#endif
