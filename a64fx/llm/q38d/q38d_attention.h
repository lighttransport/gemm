#ifndef Q38D_ATTENTION_H
#define Q38D_ATTENTION_H
#include <arm_sve.h>
#include <stddef.h>
#include <string.h>
/* Time blocking reuses V in cache across head pairs, with the original
 * two-head/128-column register tile. Each output retains its original FMA
 * order over positions; spilling a completed tile to FP32 does not round it
 * again. HD must be 256, as in the Qwen3.8 decode engine. */
#define PV8(X) svfloat32_t X##0, X##1, X##2, X##3, X##4, X##5, X##6, X##7
#define PV8_EACH(F) F(0); F(1); F(2); F(3); F(4); F(5); F(6); F(7)
static void q38d_attn_pv6_range(const float *V, int t0, int t1, const float *p, int nt, float *out, int ostride, int block, int reset) {
    const svbool_t pf = svptrue_b32();
    if (block <= 0) block = t1 > t0 ? t1 - t0 : 1;
    if (t1 <= t0) {
        for (int h = 0; h < 6; h++) memset(out + (size_t)h * ostride, 0, HD * sizeof(float));
        return;
    }
    for (int b0 = t0; b0 < t1;) {
        int b1 = t1 - b0 < block ? t1 : b0 + block;
        for (int hg = 0; hg < 6; hg += 2)
            for (int d0 = 0; d0 < HD; d0 += 128) {
                PV8(x); PV8(y);
#define PZ(j) x##j = b0 == t0 && reset ? svdup_n_f32(0) : svld1_f32(pf, out + (size_t)hg * ostride + d0 + 16 * j); \
              y##j = b0 == t0 && reset ? svdup_n_f32(0) : svld1_f32(pf, out + (size_t)(hg + 1) * ostride + d0 + 16 * j)
                PV8_EACH(PZ);
#undef PZ
                const float *pa = p + hg * nt, *pb = pa + nt;
                const float *v = V + (size_t)b0 * HD + d0;
#pragma clang loop unroll(disable)
                for (int t = b0; t < b1; t++, v += HD) {
                    svfloat32_t wa = svdup_n_f32(pa[t - t0]), wb = svdup_n_f32(pb[t - t0]);
#define PF2(j) { svfloat32_t v_ = svld1_f32(pf, v + 16 * j); x##j = svmla_f32_x(pf, x##j, v_, wa); y##j = svmla_f32_x(pf, y##j, v_, wb); }
                    PV8_EACH(PF2);
#undef PF2
                }
                float *oa = out + (size_t)hg * ostride + d0, *ob = oa + ostride;
#define PST(j) svst1_f32(pf, oa + 16 * j, x##j); svst1_f32(pf, ob + 16 * j, y##j)
                PV8_EACH(PST);
#undef PST
            }
        b0 = b1;
    }
}
static void q38d_attn_pv6_blocked(const float *V, int t0, int t1, const float *p, int nt, float *out, int ostride, int block) {
    q38d_attn_pv6_range(V, t0, t1, p, nt, out, ostride, block, 1);
}
#undef PV8
#undef PV8_EACH
#endif
