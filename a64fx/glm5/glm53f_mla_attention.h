#ifndef GLM53F_MLA_ATTENTION_H
#define GLM53F_MLA_ATTENTION_H
#include "glm53f_mla_cache_f16.h"
#include "glm53f_moe_grouped_native.h"
#include <stddef.h>

/* Same selected-key and lane accumulation order for the canonical FP32
 * cache and its derived FP16 view. The caller requires sixteen FP32 lanes. */
enum { GLM53F_MLA_ATTENTION_LAT = 512, GLM53F_MLA_ATTENTION_SLOTS = 2052 };
static inline svfloat32_t mlb_r16(svbool_t p, svfloat32_t v) {
    return svcvt_f32_f16_x(p, svcvt_f16_f32_x(p, v));
}
static inline __attribute__((always_inline)) svfloat32_t mlb_cache_load(
        svbool_t p, const float *cache, const uint16_t *half, size_t offset) {
    return half ? glm53f_mla_cache_f16_load(p, half + offset)
                : mlb_r16(p, svld1_f32(p, cache + offset));
}

/* logits[h][t] = f32dot(ql[h], round16(cache[sel[t]])) for NH heads, 4 keys at a time. */
static inline __attribute__((always_inline)) void mlb_logits(float *lg, const float *ql, const float *cache, const uint16_t *half,
        const int *sel, int nt, const int NH) {
    const svbool_t pg = svptrue_b32();
    int t = 0;
    for (; t + 4 <= nt; t += 4) {
#define MLB_L_DECL(H) svfloat32_t l0_##H = svdup_f32(0), l1_##H = l0_##H, l2_##H = l0_##H, l3_##H = l0_##H
        MLB_L_DECL(0); MLB_L_DECL(1); MLB_L_DECL(2); MLB_L_DECL(3); MLB_L_DECL(4); MLB_L_DECL(5);
#undef MLB_L_DECL
        for (int d = 0; d < GLM53F_MLA_ATTENTION_LAT; d += 16) {
            const svfloat32_t v0 = mlb_cache_load(pg, cache, half, (size_t)sel[t + 0] * GLM53F_MLA_ATTENTION_LAT + d), v1 = mlb_cache_load(pg, cache, half, (size_t)sel[t + 1] * GLM53F_MLA_ATTENTION_LAT + d);
            const svfloat32_t v2 = mlb_cache_load(pg, cache, half, (size_t)sel[t + 2] * GLM53F_MLA_ATTENTION_LAT + d), v3 = mlb_cache_load(pg, cache, half, (size_t)sel[t + 3] * GLM53F_MLA_ATTENTION_LAT + d);
#define MLB_L_STEP(H) if ((H) < NH) { const svfloat32_t q = svld1(pg, ql + (size_t)(H) * GLM53F_MLA_ATTENTION_LAT + d); \
            l0_##H = svmla_f32_x(pg, l0_##H, q, v0); l1_##H = svmla_f32_x(pg, l1_##H, q, v1); \
            l2_##H = svmla_f32_x(pg, l2_##H, q, v2); l3_##H = svmla_f32_x(pg, l3_##H, q, v3); }
            MLB_L_STEP(0) MLB_L_STEP(1) MLB_L_STEP(2) MLB_L_STEP(3) MLB_L_STEP(4) MLB_L_STEP(5)
#undef MLB_L_STEP
        }
#define MLB_L_ST(H) if ((H) < NH) { float *o = lg + (size_t)(H) * GLM53F_MLA_ATTENTION_SLOTS + t; \
        o[0] = svaddv_f32(pg, l0_##H); o[1] = svaddv_f32(pg, l1_##H); o[2] = svaddv_f32(pg, l2_##H); o[3] = svaddv_f32(pg, l3_##H); }
        MLB_L_ST(0) MLB_L_ST(1) MLB_L_ST(2) MLB_L_ST(3) MLB_L_ST(4) MLB_L_ST(5)
#undef MLB_L_ST
    }
    for (; t < nt; ++t) {
#define MLB_T_DECL(H) svfloat32_t s##H = svdup_f32(0)
        MLB_T_DECL(0); MLB_T_DECL(1); MLB_T_DECL(2); MLB_T_DECL(3); MLB_T_DECL(4); MLB_T_DECL(5);
#undef MLB_T_DECL
        for (int d = 0; d < GLM53F_MLA_ATTENTION_LAT; d += 16) {
            const svfloat32_t v0 = mlb_cache_load(pg, cache, half, (size_t)sel[t + 0] * GLM53F_MLA_ATTENTION_LAT + d);
#define MLB_T_STEP(H) if ((H) < NH) s##H = svmla_f32_x(pg, s##H, svld1(pg, ql + (size_t)(H) * GLM53F_MLA_ATTENTION_LAT + d), v0);
            MLB_T_STEP(0) MLB_T_STEP(1) MLB_T_STEP(2) MLB_T_STEP(3) MLB_T_STEP(4) MLB_T_STEP(5)
#undef MLB_T_STEP
        }
#define MLB_T_ST(H) if ((H) < NH) lg[(size_t)(H) * GLM53F_MLA_ATTENTION_SLOTS + t] = svaddv_f32(pg, s##H);
        MLB_T_ST(0) MLB_T_ST(1) MLB_T_ST(2) MLB_T_ST(3) MLB_T_ST(4) MLB_T_ST(5)
#undef MLB_T_ST
    }
}

/* va[h][d] = sum_t p[h][t] * round16(cache[sel[t]][d]) in key order, 64 columns at a time. va rows: va + h*va_stride. */
static inline __attribute__((always_inline)) void mlb_values(float *va, size_t va_stride, const float *lg,
        const float *cache, const uint16_t *half, const int *sel, int nt, const int NH) {
    const svbool_t pg = svptrue_b32();
    for (int db = 0; db < GLM53F_MLA_ATTENTION_LAT; db += 64) {
#define MLB_V_DECL(H) svfloat32_t v##H##0 = svdup_f32(0), v##H##1 = v##H##0, v##H##2 = v##H##0, v##H##3 = v##H##0
        MLB_V_DECL(0); MLB_V_DECL(1); MLB_V_DECL(2); MLB_V_DECL(3); MLB_V_DECL(4); MLB_V_DECL(5);
#undef MLB_V_DECL
        for (int t = 0; t < nt; ++t) {
            const svfloat32_t z0 = mlb_cache_load(pg, cache, half, (size_t)sel[t] * GLM53F_MLA_ATTENTION_LAT + db), z1 = mlb_cache_load(pg, cache, half, (size_t)sel[t] * GLM53F_MLA_ATTENTION_LAT + db + 16);
            const svfloat32_t z2 = mlb_cache_load(pg, cache, half, (size_t)sel[t] * GLM53F_MLA_ATTENTION_LAT + db + 32), z3 = mlb_cache_load(pg, cache, half, (size_t)sel[t] * GLM53F_MLA_ATTENTION_LAT + db + 48);
#define MLB_V_STEP(H) if ((H) < NH) { const float x = lg[(size_t)(H) * GLM53F_MLA_ATTENTION_SLOTS + t]; \
            v##H##0 = svmla_n_f32_x(pg, v##H##0, z0, x); v##H##1 = svmla_n_f32_x(pg, v##H##1, z1, x); \
            v##H##2 = svmla_n_f32_x(pg, v##H##2, z2, x); v##H##3 = svmla_n_f32_x(pg, v##H##3, z3, x); }
            MLB_V_STEP(0) MLB_V_STEP(1) MLB_V_STEP(2) MLB_V_STEP(3) MLB_V_STEP(4) MLB_V_STEP(5)
#undef MLB_V_STEP
        }
#define MLB_V_ST(H) if ((H) < NH) { float *o = va + (size_t)(H) * va_stride + db; \
        svst1(pg, o, v##H##0); svst1(pg, o + 16, v##H##1); svst1(pg, o + 32, v##H##2); svst1(pg, o + 48, v##H##3); }
        MLB_V_ST(0) MLB_V_ST(1) MLB_V_ST(2) MLB_V_ST(3) MLB_V_ST(4) MLB_V_ST(5)
#undef MLB_V_ST
    }
}

static void mlb_token(float *va, size_t va_stride, float *lg, const float *ql, const float *cache, const uint16_t *half,
        const int *sel, int nt, int NH) {
    switch (NH) {
#define MLB_CASE(N) case N: mlb_logits(lg, ql, cache, half, sel, nt, N); break;
    MLB_CASE(1) MLB_CASE(2) MLB_CASE(3) MLB_CASE(4) MLB_CASE(5) MLB_CASE(6)
#undef MLB_CASE
    }
    for (int h = 0; h < NH; ++h) {
        float *l = lg + (size_t)h * GLM53F_MLA_ATTENTION_SLOTS;
        const svbool_t pt = svptrue_b32();
        svfloat32_t vmx = svdup_f32(-INFINITY);
        int t = 0;
        for (; t + 16 <= nt; t += 16) vmx = svmax_f32_x(pt, vmx, svld1_f32(pt, l + t));
        float mx = svmaxv_f32(pt, vmx);
        for (; t < nt; ++t) if (l[t] > mx) mx = l[t];
        /* vector exp (rel. error ~1e-7) replaces 12k scalar expf calls per token */
        svfloat32_t vsum = svdup_f32(0);
        for (t = 0; t < nt; t += 16) {
            const svbool_t p = svwhilelt_b32(t, nt);
            svfloat32_t e = gmn_expf(p, svsub_n_f32_x(p, svld1_f32(p, l + t), mx));
            svst1_f32(p, l + t, e);
            vsum = svadd_f32_m(p, vsum, e);
        }
        const float sum = svaddv_f32(pt, vsum);
        for (t = 0; t < nt; t += 16) {
            const svbool_t p = svwhilelt_b32(t, nt);
            svst1_f32(p, l + t, svdiv_n_f32_x(p, svld1_f32(p, l + t), sum));
        }
    }
    switch (NH) {
#define MLB_CASE(N) case N: mlb_values(va, va_stride, lg, cache, half, sel, nt, N); break;
    MLB_CASE(1) MLB_CASE(2) MLB_CASE(3) MLB_CASE(4) MLB_CASE(5) MLB_CASE(6)
#undef MLB_CASE
    }
}

#endif
