#include "cross_entropy_sve.h"
#include "sve_math.h"
#include <arm_sve.h>
#include <math.h>
#include <float.h>


/* Prefetch distance in cache lines (256 bytes each on A64FX).
 * 8 lines = 2KB ahead. Tunable via PF_DIST_LINES. */
#ifndef PF_DIST_LINES
#define PF_DIST_LINES 8
#endif
#define PF_DIST_BYTES (PF_DIST_LINES * 256)

/* Prefetch helper: pldl1keep at byte offset ahead */
static inline void prefetch_l1(const void *base, int byte_offset) {
    __builtin_prefetch((const char *)base + byte_offset, 0, 3);
}

/* ════════════════════════════════════════════════════════════════
 * Forward pass: loss = -logits[target] + max + log(sum(exp(x-max)))
 *
 * 2-pass, 8× unrolled, SW prefetch, FEXPA exp approximation.
 * ════════════════════════════════════════════════════════════════ */

float cross_entropy_fwd_f32(const float *logits, int target, int V) {
    const int VL = (int)svcntw();          /* 16 for A64FX */
    const svbool_t pg = svptrue_b32();
    const int V8 = V & ~(8 * VL - 1);     /* 8×VL aligned count */
    const int V1 = V & ~(VL - 1);         /* 1×VL aligned count */

    /* ── Pass 1: find max (8× unrolled + SW prefetch) ── */
    svfloat32_t vm0 = svdup_f32(-FLT_MAX), vm1 = vm0, vm2 = vm0, vm3 = vm0;
    svfloat32_t vm4 = vm0, vm5 = vm0, vm6 = vm0, vm7 = vm0;

    int i = 0;
    for (; i < V8; i += 8 * VL) {
        prefetch_l1(logits + i, PF_DIST_BYTES);
        prefetch_l1(logits + i, PF_DIST_BYTES + 256);
        vm0 = svmax_f32_x(pg, vm0, svld1_f32(pg, logits + i + 0 * VL));
        vm1 = svmax_f32_x(pg, vm1, svld1_f32(pg, logits + i + 1 * VL));
        vm2 = svmax_f32_x(pg, vm2, svld1_f32(pg, logits + i + 2 * VL));
        vm3 = svmax_f32_x(pg, vm3, svld1_f32(pg, logits + i + 3 * VL));
        vm4 = svmax_f32_x(pg, vm4, svld1_f32(pg, logits + i + 4 * VL));
        vm5 = svmax_f32_x(pg, vm5, svld1_f32(pg, logits + i + 5 * VL));
        vm6 = svmax_f32_x(pg, vm6, svld1_f32(pg, logits + i + 6 * VL));
        vm7 = svmax_f32_x(pg, vm7, svld1_f32(pg, logits + i + 7 * VL));
    }
    for (; i < V1; i += VL)
        vm0 = svmax_f32_x(pg, vm0, svld1_f32(pg, logits + i));
    if (V1 < V) {
        svbool_t ptail = svwhilelt_b32((uint32_t)V1, (uint32_t)V);
        svfloat32_t tail = svsel_f32(ptail,
            svld1_f32(ptail, logits + V1), svdup_f32(-FLT_MAX));
        vm0 = svmax_f32_x(pg, vm0, tail);
    }

    /* Tree reduce 8→1 */
    vm0 = svmax_f32_x(pg, vm0, vm1);
    vm2 = svmax_f32_x(pg, vm2, vm3);
    vm4 = svmax_f32_x(pg, vm4, vm5);
    vm6 = svmax_f32_x(pg, vm6, vm7);
    vm0 = svmax_f32_x(pg, vm0, vm2);
    vm4 = svmax_f32_x(pg, vm4, vm6);
    svfloat32_t vmax = svmax_f32_x(pg, vm0, vm4);

    float mbuf[16] __attribute__((aligned(256)));
    svst1_f32(pg, mbuf, vmax);
    float max_val = mbuf[0];
    for (int k = 1; k < VL; k++)
        max_val = fmaxf(max_val, mbuf[k]);

    /* ── Pass 2: sum(exp(x - max)) via FEXPA, 8× unrolled + SW prefetch ── */
    float shift_f = fexpa_shift_f32();
    float bias = -max_val * LOG2E + shift_f;
    svfloat32_t vbias   = svdup_f32(bias);
    svfloat32_t vlog2e  = svdup_f32(LOG2E);

    svfloat32_t vs0 = svdup_f32(0.0f), vs1 = vs0, vs2 = vs0, vs3 = vs0;
    svfloat32_t vs4 = vs0, vs5 = vs0, vs6 = vs0, vs7 = vs0;

    i = 0;
    for (; i < V8; i += 8 * VL) {
        prefetch_l1(logits + i, PF_DIST_BYTES);
        prefetch_l1(logits + i, PF_DIST_BYTES + 256);
        svfloat32_t z0 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 0 * VL), vlog2e);
        svfloat32_t z1 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 1 * VL), vlog2e);
        svfloat32_t z2 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 2 * VL), vlog2e);
        svfloat32_t z3 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 3 * VL), vlog2e);
        svfloat32_t z4 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 4 * VL), vlog2e);
        svfloat32_t z5 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 5 * VL), vlog2e);
        svfloat32_t z6 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 6 * VL), vlog2e);
        svfloat32_t z7 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 7 * VL), vlog2e);
        vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z0));
        vs1 = svadd_f32_x(pg, vs1, sve_fexpa(z1));
        vs2 = svadd_f32_x(pg, vs2, sve_fexpa(z2));
        vs3 = svadd_f32_x(pg, vs3, sve_fexpa(z3));
        vs4 = svadd_f32_x(pg, vs4, sve_fexpa(z4));
        vs5 = svadd_f32_x(pg, vs5, sve_fexpa(z5));
        vs6 = svadd_f32_x(pg, vs6, sve_fexpa(z6));
        vs7 = svadd_f32_x(pg, vs7, sve_fexpa(z7));
    }
    for (; i < V1; i += VL) {
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i), vlog2e);
        vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z));
    }
    if (V1 < V) {
        svbool_t ptail = svwhilelt_b32((uint32_t)V1, (uint32_t)V);
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(ptail, logits + V1), vlog2e);
        svfloat32_t e = svsel_f32(ptail, sve_fexpa(z), svdup_f32(0.0f));
        vs0 = svadd_f32_x(pg, vs0, e);
    }

    /* Tree reduce 8→1 + scalar extract */
    vs0 = svadd_f32_x(pg, vs0, vs1);
    vs2 = svadd_f32_x(pg, vs2, vs3);
    vs4 = svadd_f32_x(pg, vs4, vs5);
    vs6 = svadd_f32_x(pg, vs6, vs7);
    vs0 = svadd_f32_x(pg, vs0, vs2);
    vs4 = svadd_f32_x(pg, vs4, vs6);
    svfloat32_t vsum = svadd_f32_x(pg, vs0, vs4);

    float sbuf[16] __attribute__((aligned(256)));
    svst1_f32(pg, sbuf, vsum);
    float sum_exp = 0.0f;
    for (int k = 0; k < VL; k++)
        sum_exp += sbuf[k];

    return -logits[target] + max_val + logf(sum_exp);
}

/* ════════════════════════════════════════════════════════════════
 * Forward with FP16 input → FP32 compute
 *
 * Fixed: explicit 4× fp16-vec unroll (no switch statement).
 * Loads 4×32 fp16 = 128 fp16 per iteration, unpacks to 8 fp32 vecs.
 * ════════════════════════════════════════════════════════════════ */

float cross_entropy_fwd_f16(const uint16_t *logits_f16, int target, int V) {
    const int VL = (int)svcntw();          /* 16 fp32 lanes */
    const int VL16 = 2 * VL;              /* 32 fp16 lanes per load */
    const svbool_t pg = svptrue_b32();
    const svbool_t pg16 = svptrue_b16();
    const int V4 = V & ~(4 * VL16 - 1);  /* 4 fp16-vec aligned */
    const int V1 = V & ~(VL16 - 1);

    /* ── Pass 1: max (4× fp16-vec unroll = 8 fp32 accumulators) ── */
    svfloat32_t vm0 = svdup_f32(-FLT_MAX), vm1 = vm0, vm2 = vm0, vm3 = vm0;
    svfloat32_t vm4 = vm0, vm5 = vm0, vm6 = vm0, vm7 = vm0;

    int i = 0;
    for (; i < V4; i += 4 * VL16) {
        prefetch_l1(logits_f16 + i, PF_DIST_BYTES);
        svfloat16_t h0 = svld1_f16(pg16, (const float16_t *)(logits_f16 + i + 0 * VL16));
        svfloat16_t h1 = svld1_f16(pg16, (const float16_t *)(logits_f16 + i + 1 * VL16));
        svfloat16_t h2 = svld1_f16(pg16, (const float16_t *)(logits_f16 + i + 2 * VL16));
        svfloat16_t h3 = svld1_f16(pg16, (const float16_t *)(logits_f16 + i + 3 * VL16));

        svfloat32_t lo0 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h0))));
        svfloat32_t hi0 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h0))));
        svfloat32_t lo1 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h1))));
        svfloat32_t hi1 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h1))));
        svfloat32_t lo2 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h2))));
        svfloat32_t hi2 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h2))));
        svfloat32_t lo3 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h3))));
        svfloat32_t hi3 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h3))));

        vm0 = svmax_f32_x(pg, vm0, lo0);
        vm1 = svmax_f32_x(pg, vm1, hi0);
        vm2 = svmax_f32_x(pg, vm2, lo1);
        vm3 = svmax_f32_x(pg, vm3, hi1);
        vm4 = svmax_f32_x(pg, vm4, lo2);
        vm5 = svmax_f32_x(pg, vm5, hi2);
        vm6 = svmax_f32_x(pg, vm6, lo3);
        vm7 = svmax_f32_x(pg, vm7, hi3);
    }
    for (; i < V1; i += VL16) {
        svfloat16_t h = svld1_f16(pg16, (const float16_t *)(logits_f16 + i));
        svfloat32_t lo = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h))));
        svfloat32_t hi = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h))));
        vm0 = svmax_f32_x(pg, vm0, lo);
        vm1 = svmax_f32_x(pg, vm1, hi);
    }
    /* Scalar tail */
    float tail_max = -FLT_MAX;
    for (int j = i; j < V; j++) {
        union { uint16_t u; _Float16 f; } cv = { .u = logits_f16[j] };
        float v = (float)cv.f;
        if (v > tail_max) tail_max = v;
    }

    /* Tree reduce */
    vm0 = svmax_f32_x(pg, vm0, vm1);
    vm2 = svmax_f32_x(pg, vm2, vm3);
    vm4 = svmax_f32_x(pg, vm4, vm5);
    vm6 = svmax_f32_x(pg, vm6, vm7);
    vm0 = svmax_f32_x(pg, vm0, vm2);
    vm4 = svmax_f32_x(pg, vm4, vm6);
    svfloat32_t vmax_vec = svmax_f32_x(pg, vm0, vm4);

    float mbuf[16] __attribute__((aligned(256)));
    svst1_f32(pg, mbuf, vmax_vec);
    float max_val = mbuf[0];
    for (int k = 1; k < VL; k++)
        max_val = fmaxf(max_val, mbuf[k]);
    max_val = fmaxf(max_val, tail_max);

    /* ── Pass 2: sum(exp(x - max)), 4× fp16-vec unrolled + SW prefetch ── */
    float shift_f = fexpa_shift_f32();
    float bias = -max_val * LOG2E + shift_f;
    svfloat32_t vbias  = svdup_f32(bias);
    svfloat32_t vlog2e = svdup_f32(LOG2E);

    svfloat32_t vs0 = svdup_f32(0.0f), vs1 = vs0, vs2 = vs0, vs3 = vs0;
    svfloat32_t vs4 = vs0, vs5 = vs0, vs6 = vs0, vs7 = vs0;

    i = 0;
    for (; i < V4; i += 4 * VL16) {
        prefetch_l1(logits_f16 + i, PF_DIST_BYTES);
        svfloat16_t h0 = svld1_f16(pg16, (const float16_t *)(logits_f16 + i + 0 * VL16));
        svfloat16_t h1 = svld1_f16(pg16, (const float16_t *)(logits_f16 + i + 1 * VL16));
        svfloat16_t h2 = svld1_f16(pg16, (const float16_t *)(logits_f16 + i + 2 * VL16));
        svfloat16_t h3 = svld1_f16(pg16, (const float16_t *)(logits_f16 + i + 3 * VL16));

        svfloat32_t lo0 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h0))));
        svfloat32_t hi0 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h0))));
        svfloat32_t lo1 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h1))));
        svfloat32_t hi1 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h1))));
        svfloat32_t lo2 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h2))));
        svfloat32_t hi2 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h2))));
        svfloat32_t lo3 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h3))));
        svfloat32_t hi3 = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h3))));

        svfloat32_t z0 = svmla_f32_x(pg, vbias, lo0, vlog2e);
        svfloat32_t z1 = svmla_f32_x(pg, vbias, hi0, vlog2e);
        svfloat32_t z2 = svmla_f32_x(pg, vbias, lo1, vlog2e);
        svfloat32_t z3 = svmla_f32_x(pg, vbias, hi1, vlog2e);
        svfloat32_t z4 = svmla_f32_x(pg, vbias, lo2, vlog2e);
        svfloat32_t z5 = svmla_f32_x(pg, vbias, hi2, vlog2e);
        svfloat32_t z6 = svmla_f32_x(pg, vbias, lo3, vlog2e);
        svfloat32_t z7 = svmla_f32_x(pg, vbias, hi3, vlog2e);

        vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z0));
        vs1 = svadd_f32_x(pg, vs1, sve_fexpa(z1));
        vs2 = svadd_f32_x(pg, vs2, sve_fexpa(z2));
        vs3 = svadd_f32_x(pg, vs3, sve_fexpa(z3));
        vs4 = svadd_f32_x(pg, vs4, sve_fexpa(z4));
        vs5 = svadd_f32_x(pg, vs5, sve_fexpa(z5));
        vs6 = svadd_f32_x(pg, vs6, sve_fexpa(z6));
        vs7 = svadd_f32_x(pg, vs7, sve_fexpa(z7));
    }
    for (; i < V1; i += VL16) {
        svfloat16_t h = svld1_f16(pg16, (const float16_t *)(logits_f16 + i));
        svfloat32_t lo = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo(svreinterpret_u16(h))));
        svfloat32_t hi = svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi(svreinterpret_u16(h))));
        svfloat32_t z_lo = svmla_f32_x(pg, vbias, lo, vlog2e);
        svfloat32_t z_hi = svmla_f32_x(pg, vbias, hi, vlog2e);
        vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z_lo));
        vs1 = svadd_f32_x(pg, vs1, sve_fexpa(z_hi));
    }
    /* Scalar tail */
    float tail_sum = 0.0f;
    for (int j = i; j < V; j++) {
        union { uint16_t u; _Float16 f; } cv = { .u = logits_f16[j] };
        tail_sum += expf((float)cv.f - max_val);
    }

    /* Tree reduce 8→1 */
    vs0 = svadd_f32_x(pg, vs0, vs1);
    vs2 = svadd_f32_x(pg, vs2, vs3);
    vs4 = svadd_f32_x(pg, vs4, vs5);
    vs6 = svadd_f32_x(pg, vs6, vs7);
    vs0 = svadd_f32_x(pg, vs0, vs2);
    vs4 = svadd_f32_x(pg, vs4, vs6);
    svfloat32_t vsum = svadd_f32_x(pg, vs0, vs4);

    float sbuf[16] __attribute__((aligned(256)));
    svst1_f32(pg, sbuf, vsum);
    float sum_exp = 0.0f;
    for (int k = 0; k < VL; k++)
        sum_exp += sbuf[k];
    sum_exp += tail_sum;

    /* Target logit in fp32 */
    union { uint16_t u; _Float16 f; } tgt_cv = { .u = logits_f16[target] };
    float target_logit = (float)tgt_cv.f;

    return -target_logit + max_val + logf(sum_exp);
}

/* ════════════════════════════════════════════════════════════════
 * Backward: grad[i] = softmax(x)_i - 1{i==target}
 *
 * 8× unrolled + SW prefetch for both read and write streams.
 * ════════════════════════════════════════════════════════════════ */

void cross_entropy_bwd_f32(const float *logits, int target, int V,
                           float max_val, float sum_exp, float *grad) {
    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int V8 = V & ~(8 * VL - 1);
    const int V1 = V & ~(VL - 1);

    float shift_f = fexpa_shift_f32();
    float bias = -max_val * LOG2E + shift_f;
    svfloat32_t vbias   = svdup_f32(bias);
    svfloat32_t vlog2e  = svdup_f32(LOG2E);
    svfloat32_t vinv    = svdup_f32(1.0f / sum_exp);

    int i = 0;
    for (; i < V8; i += 8 * VL) {
        prefetch_l1(logits + i, PF_DIST_BYTES);
        prefetch_l1(logits + i, PF_DIST_BYTES + 256);
        svfloat32_t z0 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 0 * VL), vlog2e);
        svfloat32_t z1 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 1 * VL), vlog2e);
        svfloat32_t z2 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 2 * VL), vlog2e);
        svfloat32_t z3 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 3 * VL), vlog2e);
        svfloat32_t z4 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 4 * VL), vlog2e);
        svfloat32_t z5 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 5 * VL), vlog2e);
        svfloat32_t z6 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 6 * VL), vlog2e);
        svfloat32_t z7 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 7 * VL), vlog2e);
        svst1_f32(pg, grad + i + 0 * VL, svmul_f32_x(pg, sve_fexpa(z0), vinv));
        svst1_f32(pg, grad + i + 1 * VL, svmul_f32_x(pg, sve_fexpa(z1), vinv));
        svst1_f32(pg, grad + i + 2 * VL, svmul_f32_x(pg, sve_fexpa(z2), vinv));
        svst1_f32(pg, grad + i + 3 * VL, svmul_f32_x(pg, sve_fexpa(z3), vinv));
        svst1_f32(pg, grad + i + 4 * VL, svmul_f32_x(pg, sve_fexpa(z4), vinv));
        svst1_f32(pg, grad + i + 5 * VL, svmul_f32_x(pg, sve_fexpa(z5), vinv));
        svst1_f32(pg, grad + i + 6 * VL, svmul_f32_x(pg, sve_fexpa(z6), vinv));
        svst1_f32(pg, grad + i + 7 * VL, svmul_f32_x(pg, sve_fexpa(z7), vinv));
    }
    for (; i < V1; i += VL) {
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i), vlog2e);
        svst1_f32(pg, grad + i, svmul_f32_x(pg, sve_fexpa(z), vinv));
    }
    if (V1 < V) {
        svbool_t ptail = svwhilelt_b32((uint32_t)V1, (uint32_t)V);
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(ptail, logits + V1), vlog2e);
        svst1_f32(ptail, grad + V1, svmul_f32_x(pg, sve_fexpa(z), vinv));
    }

    grad[target] -= 1.0f;
}

/* ════════════════════════════════════════════════════════════════
 * Combined forward + backward (3 passes, 8× unrolled + SW prefetch)
 * ════════════════════════════════════════════════════════════════ */

float cross_entropy_fwd_bwd_f32(const float *logits, int target, int V,
                                float *grad) {
    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int V8 = V & ~(8 * VL - 1);
    const int V1 = V & ~(VL - 1);

    /* ── Pass 1: max (8× unrolled + SW prefetch) ── */
    svfloat32_t vm0 = svdup_f32(-FLT_MAX), vm1 = vm0, vm2 = vm0, vm3 = vm0;
    svfloat32_t vm4 = vm0, vm5 = vm0, vm6 = vm0, vm7 = vm0;

    int i = 0;
    for (; i < V8; i += 8 * VL) {
        prefetch_l1(logits + i, PF_DIST_BYTES);
        prefetch_l1(logits + i, PF_DIST_BYTES + 256);
        vm0 = svmax_f32_x(pg, vm0, svld1_f32(pg, logits + i + 0 * VL));
        vm1 = svmax_f32_x(pg, vm1, svld1_f32(pg, logits + i + 1 * VL));
        vm2 = svmax_f32_x(pg, vm2, svld1_f32(pg, logits + i + 2 * VL));
        vm3 = svmax_f32_x(pg, vm3, svld1_f32(pg, logits + i + 3 * VL));
        vm4 = svmax_f32_x(pg, vm4, svld1_f32(pg, logits + i + 4 * VL));
        vm5 = svmax_f32_x(pg, vm5, svld1_f32(pg, logits + i + 5 * VL));
        vm6 = svmax_f32_x(pg, vm6, svld1_f32(pg, logits + i + 6 * VL));
        vm7 = svmax_f32_x(pg, vm7, svld1_f32(pg, logits + i + 7 * VL));
    }
    for (; i < V1; i += VL)
        vm0 = svmax_f32_x(pg, vm0, svld1_f32(pg, logits + i));
    if (V1 < V) {
        svbool_t ptail = svwhilelt_b32((uint32_t)V1, (uint32_t)V);
        svfloat32_t tail = svsel_f32(ptail,
            svld1_f32(ptail, logits + V1), svdup_f32(-FLT_MAX));
        vm0 = svmax_f32_x(pg, vm0, tail);
    }

    vm0 = svmax_f32_x(pg, vm0, vm1);
    vm2 = svmax_f32_x(pg, vm2, vm3);
    vm4 = svmax_f32_x(pg, vm4, vm5);
    vm6 = svmax_f32_x(pg, vm6, vm7);
    vm0 = svmax_f32_x(pg, vm0, vm2);
    vm4 = svmax_f32_x(pg, vm4, vm6);
    svfloat32_t vmax = svmax_f32_x(pg, vm0, vm4);

    float mbuf[16] __attribute__((aligned(256)));
    svst1_f32(pg, mbuf, vmax);
    float max_val = mbuf[0];
    for (int k = 1; k < VL; k++)
        max_val = fmaxf(max_val, mbuf[k]);

    /* ── Pass 2: sum(exp(x - max)), 8× unrolled + SW prefetch ── */
    float shift_f = fexpa_shift_f32();
    float bias = -max_val * LOG2E + shift_f;
    svfloat32_t vbias  = svdup_f32(bias);
    svfloat32_t vlog2e = svdup_f32(LOG2E);

    svfloat32_t vs0 = svdup_f32(0.0f), vs1 = vs0, vs2 = vs0, vs3 = vs0;
    svfloat32_t vs4 = vs0, vs5 = vs0, vs6 = vs0, vs7 = vs0;

    i = 0;
    for (; i < V8; i += 8 * VL) {
        prefetch_l1(logits + i, PF_DIST_BYTES);
        prefetch_l1(logits + i, PF_DIST_BYTES + 256);
        svfloat32_t z0 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 0 * VL), vlog2e);
        svfloat32_t z1 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 1 * VL), vlog2e);
        svfloat32_t z2 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 2 * VL), vlog2e);
        svfloat32_t z3 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 3 * VL), vlog2e);
        svfloat32_t z4 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 4 * VL), vlog2e);
        svfloat32_t z5 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 5 * VL), vlog2e);
        svfloat32_t z6 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 6 * VL), vlog2e);
        svfloat32_t z7 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 7 * VL), vlog2e);
        vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z0));
        vs1 = svadd_f32_x(pg, vs1, sve_fexpa(z1));
        vs2 = svadd_f32_x(pg, vs2, sve_fexpa(z2));
        vs3 = svadd_f32_x(pg, vs3, sve_fexpa(z3));
        vs4 = svadd_f32_x(pg, vs4, sve_fexpa(z4));
        vs5 = svadd_f32_x(pg, vs5, sve_fexpa(z5));
        vs6 = svadd_f32_x(pg, vs6, sve_fexpa(z6));
        vs7 = svadd_f32_x(pg, vs7, sve_fexpa(z7));
    }
    for (; i < V1; i += VL) {
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i), vlog2e);
        vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z));
    }
    if (V1 < V) {
        svbool_t ptail = svwhilelt_b32((uint32_t)V1, (uint32_t)V);
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(ptail, logits + V1), vlog2e);
        svfloat32_t e = svsel_f32(ptail, sve_fexpa(z), svdup_f32(0.0f));
        vs0 = svadd_f32_x(pg, vs0, e);
    }

    vs0 = svadd_f32_x(pg, vs0, vs1);
    vs2 = svadd_f32_x(pg, vs2, vs3);
    vs4 = svadd_f32_x(pg, vs4, vs5);
    vs6 = svadd_f32_x(pg, vs6, vs7);
    vs0 = svadd_f32_x(pg, vs0, vs2);
    vs4 = svadd_f32_x(pg, vs4, vs6);
    svfloat32_t vsum = svadd_f32_x(pg, vs0, vs4);

    float sbuf[16] __attribute__((aligned(256)));
    svst1_f32(pg, sbuf, vsum);
    float sum_exp = 0.0f;
    for (int k = 0; k < VL; k++)
        sum_exp += sbuf[k];

    float loss = -logits[target] + max_val + logf(sum_exp);

    /* ── Pass 3: backward grad[i] = softmax_i - 1{i==target}, 8× + prefetch ── */
    svfloat32_t vinv = svdup_f32(1.0f / sum_exp);

    i = 0;
    for (; i < V8; i += 8 * VL) {
        prefetch_l1(logits + i, PF_DIST_BYTES);
        prefetch_l1(logits + i, PF_DIST_BYTES + 256);
        svfloat32_t z0 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 0 * VL), vlog2e);
        svfloat32_t z1 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 1 * VL), vlog2e);
        svfloat32_t z2 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 2 * VL), vlog2e);
        svfloat32_t z3 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 3 * VL), vlog2e);
        svfloat32_t z4 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 4 * VL), vlog2e);
        svfloat32_t z5 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 5 * VL), vlog2e);
        svfloat32_t z6 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 6 * VL), vlog2e);
        svfloat32_t z7 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 7 * VL), vlog2e);
        svst1_f32(pg, grad + i + 0 * VL, svmul_f32_x(pg, sve_fexpa(z0), vinv));
        svst1_f32(pg, grad + i + 1 * VL, svmul_f32_x(pg, sve_fexpa(z1), vinv));
        svst1_f32(pg, grad + i + 2 * VL, svmul_f32_x(pg, sve_fexpa(z2), vinv));
        svst1_f32(pg, grad + i + 3 * VL, svmul_f32_x(pg, sve_fexpa(z3), vinv));
        svst1_f32(pg, grad + i + 4 * VL, svmul_f32_x(pg, sve_fexpa(z4), vinv));
        svst1_f32(pg, grad + i + 5 * VL, svmul_f32_x(pg, sve_fexpa(z5), vinv));
        svst1_f32(pg, grad + i + 6 * VL, svmul_f32_x(pg, sve_fexpa(z6), vinv));
        svst1_f32(pg, grad + i + 7 * VL, svmul_f32_x(pg, sve_fexpa(z7), vinv));
    }
    for (; i < V1; i += VL) {
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i), vlog2e);
        svst1_f32(pg, grad + i, svmul_f32_x(pg, sve_fexpa(z), vinv));
    }
    if (V1 < V) {
        svbool_t ptail = svwhilelt_b32((uint32_t)V1, (uint32_t)V);
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(ptail, logits + V1), vlog2e);
        svst1_f32(ptail, grad + V1, svmul_f32_x(pg, sve_fexpa(z), vinv));
    }

    grad[target] -= 1.0f;
    return loss;
}

/* ════════════════════════════════════════════════════════════════
 * Blocked forward: fuse max + exp+sum into 1 L2 pass
 *
 * For each block (32KB = 8192 floats):
 *   1. Stream from L2, find block_max → data now warm in L1
 *   2. Update global_max; if changed, correct running_sum
 *   3. Re-read block from L1, accumulate exp(x - global_max)
 *
 * Total: 1 L2 read + 1 L1 read per element (vs 2 L2 reads in 2-pass)
 * ════════════════════════════════════════════════════════════════ */

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8192  /* floats per block (32 KB = half L1D) */
#endif

float cross_entropy_fwd_blocked_f32(const float *logits, int target, int V) {
    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int V8 = 8 * VL;  /* elements per 8-wide iteration */

    float shift_f = fexpa_shift_f32();
    float global_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int b = 0; b < V; b += BLOCK_SIZE) {
        int blen = (b + BLOCK_SIZE <= V) ? BLOCK_SIZE : (V - b);
        const float *bp = logits + b;
        int blen8 = blen & ~(V8 - 1);
        int blen1 = blen & ~(VL - 1);

        /* ── Sub-pass 1a: find block max (L2→L1) ── */
        svfloat32_t vm0 = svdup_f32(-FLT_MAX), vm1 = vm0, vm2 = vm0, vm3 = vm0;
        svfloat32_t vm4 = vm0, vm5 = vm0, vm6 = vm0, vm7 = vm0;

        int j = 0;
        for (; j < blen8; j += V8) {
            prefetch_l1(bp + j, PF_DIST_BYTES);
            prefetch_l1(bp + j, PF_DIST_BYTES + 256);
            vm0 = svmax_f32_x(pg, vm0, svld1_f32(pg, bp + j + 0 * VL));
            vm1 = svmax_f32_x(pg, vm1, svld1_f32(pg, bp + j + 1 * VL));
            vm2 = svmax_f32_x(pg, vm2, svld1_f32(pg, bp + j + 2 * VL));
            vm3 = svmax_f32_x(pg, vm3, svld1_f32(pg, bp + j + 3 * VL));
            vm4 = svmax_f32_x(pg, vm4, svld1_f32(pg, bp + j + 4 * VL));
            vm5 = svmax_f32_x(pg, vm5, svld1_f32(pg, bp + j + 5 * VL));
            vm6 = svmax_f32_x(pg, vm6, svld1_f32(pg, bp + j + 6 * VL));
            vm7 = svmax_f32_x(pg, vm7, svld1_f32(pg, bp + j + 7 * VL));
        }
        for (; j < blen1; j += VL)
            vm0 = svmax_f32_x(pg, vm0, svld1_f32(pg, bp + j));
        if (blen1 < blen) {
            svbool_t ptail = svwhilelt_b32((uint32_t)blen1, (uint32_t)blen);
            svfloat32_t tail = svsel_f32(ptail,
                svld1_f32(ptail, bp + blen1), svdup_f32(-FLT_MAX));
            vm0 = svmax_f32_x(pg, vm0, tail);
        }

        vm0 = svmax_f32_x(pg, vm0, vm1);
        vm2 = svmax_f32_x(pg, vm2, vm3);
        vm4 = svmax_f32_x(pg, vm4, vm5);
        vm6 = svmax_f32_x(pg, vm6, vm7);
        vm0 = svmax_f32_x(pg, vm0, vm2);
        vm4 = svmax_f32_x(pg, vm4, vm6);
        svfloat32_t vmax_blk = svmax_f32_x(pg, vm0, vm4);

        float mbuf[16] __attribute__((aligned(256)));
        svst1_f32(pg, mbuf, vmax_blk);
        float block_max = mbuf[0];
        for (int k = 1; k < VL; k++)
            block_max = fmaxf(block_max, mbuf[k]);

        /* ── Update global max, correct running_sum if needed ── */
        if (block_max > global_max) {
            if (running_sum > 0.0f) {
                /* Correct: sum *= exp(old_max - new_max)
                 *        = fexpa((old_max - new_max) * LOG2E + shift) */
                float correction_arg = (global_max - block_max) * LOG2E + shift_f;
                /* Scalar FEXPA via SVE (single element) */
                svfloat32_t vc = svdup_f32(correction_arg);
                svfloat32_t ve = sve_fexpa(vc);
                float cbuf[16] __attribute__((aligned(256)));
                svst1_f32(pg, cbuf, ve);
                running_sum *= cbuf[0];
            }
            global_max = block_max;
        }

        /* ── Sub-pass 1b: accumulate exp(x - global_max) from L1 ── */
        float bias = -global_max * LOG2E + shift_f;
        svfloat32_t vbias  = svdup_f32(bias);
        svfloat32_t vlog2e = svdup_f32(LOG2E);

        svfloat32_t vs0 = svdup_f32(0.0f), vs1 = vs0, vs2 = vs0, vs3 = vs0;
        svfloat32_t vs4 = vs0, vs5 = vs0, vs6 = vs0, vs7 = vs0;

        j = 0;
        for (; j < blen8; j += V8) {
            /* Data should be in L1 from pass 1a — no prefetch needed */
            svfloat32_t z0 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 0 * VL), vlog2e);
            svfloat32_t z1 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 1 * VL), vlog2e);
            svfloat32_t z2 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 2 * VL), vlog2e);
            svfloat32_t z3 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 3 * VL), vlog2e);
            svfloat32_t z4 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 4 * VL), vlog2e);
            svfloat32_t z5 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 5 * VL), vlog2e);
            svfloat32_t z6 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 6 * VL), vlog2e);
            svfloat32_t z7 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 7 * VL), vlog2e);
            vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z0));
            vs1 = svadd_f32_x(pg, vs1, sve_fexpa(z1));
            vs2 = svadd_f32_x(pg, vs2, sve_fexpa(z2));
            vs3 = svadd_f32_x(pg, vs3, sve_fexpa(z3));
            vs4 = svadd_f32_x(pg, vs4, sve_fexpa(z4));
            vs5 = svadd_f32_x(pg, vs5, sve_fexpa(z5));
            vs6 = svadd_f32_x(pg, vs6, sve_fexpa(z6));
            vs7 = svadd_f32_x(pg, vs7, sve_fexpa(z7));
        }
        for (; j < blen1; j += VL) {
            svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j), vlog2e);
            vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z));
        }
        if (blen1 < blen) {
            svbool_t ptail = svwhilelt_b32((uint32_t)blen1, (uint32_t)blen);
            svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(ptail, bp + blen1), vlog2e);
            svfloat32_t e = svsel_f32(ptail, sve_fexpa(z), svdup_f32(0.0f));
            vs0 = svadd_f32_x(pg, vs0, e);
        }

        /* Reduce block sum */
        vs0 = svadd_f32_x(pg, vs0, vs1);
        vs2 = svadd_f32_x(pg, vs2, vs3);
        vs4 = svadd_f32_x(pg, vs4, vs5);
        vs6 = svadd_f32_x(pg, vs6, vs7);
        vs0 = svadd_f32_x(pg, vs0, vs2);
        vs4 = svadd_f32_x(pg, vs4, vs6);
        svfloat32_t vblk_sum = svadd_f32_x(pg, vs0, vs4);

        float sbuf[16] __attribute__((aligned(256)));
        svst1_f32(pg, sbuf, vblk_sum);
        float block_sum = 0.0f;
        for (int k = 0; k < VL; k++)
            block_sum += sbuf[k];

        running_sum += block_sum;
    }

    return -logits[target] + global_max + logf(running_sum);
}

/* ════════════════════════════════════════════════════════════════
 * Blocked forward + backward
 *
 * Phase 1: Blocked max+exp+sum (1 L2 pass via blocking)
 * Phase 2: Stream logits, compute grad = exp(x-max)/sum, write (1 L2 read + 1 write)
 * Total: 2 L2 reads + 1 write (vs 3 reads + 1 write in 3-pass)
 * ════════════════════════════════════════════════════════════════ */

float cross_entropy_fwd_bwd_blocked_f32(const float *logits, int target, int V,
                                         float *grad) {
    const int VL = (int)svcntw();
    const svbool_t pg = svptrue_b32();
    const int V8 = 8 * VL;

    float shift_f = fexpa_shift_f32();
    float global_max = -FLT_MAX;
    float running_sum = 0.0f;

    /* ── Phase 1: Blocked max + exp + sum (1 L2 pass) ── */
    for (int b = 0; b < V; b += BLOCK_SIZE) {
        int blen = (b + BLOCK_SIZE <= V) ? BLOCK_SIZE : (V - b);
        const float *bp = logits + b;
        int blen8 = blen & ~(V8 - 1);
        int blen1 = blen & ~(VL - 1);

        /* Sub-pass 1a: block max (L2→L1) */
        svfloat32_t vm0 = svdup_f32(-FLT_MAX), vm1 = vm0, vm2 = vm0, vm3 = vm0;
        svfloat32_t vm4 = vm0, vm5 = vm0, vm6 = vm0, vm7 = vm0;

        int j = 0;
        for (; j < blen8; j += V8) {
            prefetch_l1(bp + j, PF_DIST_BYTES);
            prefetch_l1(bp + j, PF_DIST_BYTES + 256);
            vm0 = svmax_f32_x(pg, vm0, svld1_f32(pg, bp + j + 0 * VL));
            vm1 = svmax_f32_x(pg, vm1, svld1_f32(pg, bp + j + 1 * VL));
            vm2 = svmax_f32_x(pg, vm2, svld1_f32(pg, bp + j + 2 * VL));
            vm3 = svmax_f32_x(pg, vm3, svld1_f32(pg, bp + j + 3 * VL));
            vm4 = svmax_f32_x(pg, vm4, svld1_f32(pg, bp + j + 4 * VL));
            vm5 = svmax_f32_x(pg, vm5, svld1_f32(pg, bp + j + 5 * VL));
            vm6 = svmax_f32_x(pg, vm6, svld1_f32(pg, bp + j + 6 * VL));
            vm7 = svmax_f32_x(pg, vm7, svld1_f32(pg, bp + j + 7 * VL));
        }
        for (; j < blen1; j += VL)
            vm0 = svmax_f32_x(pg, vm0, svld1_f32(pg, bp + j));
        if (blen1 < blen) {
            svbool_t ptail = svwhilelt_b32((uint32_t)blen1, (uint32_t)blen);
            svfloat32_t tail = svsel_f32(ptail,
                svld1_f32(ptail, bp + blen1), svdup_f32(-FLT_MAX));
            vm0 = svmax_f32_x(pg, vm0, tail);
        }

        vm0 = svmax_f32_x(pg, vm0, vm1);
        vm2 = svmax_f32_x(pg, vm2, vm3);
        vm4 = svmax_f32_x(pg, vm4, vm5);
        vm6 = svmax_f32_x(pg, vm6, vm7);
        vm0 = svmax_f32_x(pg, vm0, vm2);
        vm4 = svmax_f32_x(pg, vm4, vm6);
        svfloat32_t vmax_blk = svmax_f32_x(pg, vm0, vm4);

        float mbuf[16] __attribute__((aligned(256)));
        svst1_f32(pg, mbuf, vmax_blk);
        float block_max = mbuf[0];
        for (int k = 1; k < VL; k++)
            block_max = fmaxf(block_max, mbuf[k]);

        /* Update global max, correct running_sum */
        if (block_max > global_max) {
            if (running_sum > 0.0f) {
                float correction_arg = (global_max - block_max) * LOG2E + shift_f;
                svfloat32_t vc = svdup_f32(correction_arg);
                svfloat32_t ve = sve_fexpa(vc);
                float cbuf[16] __attribute__((aligned(256)));
                svst1_f32(pg, cbuf, ve);
                running_sum *= cbuf[0];
            }
            global_max = block_max;
        }

        /* Sub-pass 1b: accumulate exp from L1 */
        float bias = -global_max * LOG2E + shift_f;
        svfloat32_t vbias  = svdup_f32(bias);
        svfloat32_t vlog2e = svdup_f32(LOG2E);

        svfloat32_t vs0 = svdup_f32(0.0f), vs1 = vs0, vs2 = vs0, vs3 = vs0;
        svfloat32_t vs4 = vs0, vs5 = vs0, vs6 = vs0, vs7 = vs0;

        j = 0;
        for (; j < blen8; j += V8) {
            svfloat32_t z0 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 0 * VL), vlog2e);
            svfloat32_t z1 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 1 * VL), vlog2e);
            svfloat32_t z2 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 2 * VL), vlog2e);
            svfloat32_t z3 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 3 * VL), vlog2e);
            svfloat32_t z4 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 4 * VL), vlog2e);
            svfloat32_t z5 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 5 * VL), vlog2e);
            svfloat32_t z6 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 6 * VL), vlog2e);
            svfloat32_t z7 = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j + 7 * VL), vlog2e);
            vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z0));
            vs1 = svadd_f32_x(pg, vs1, sve_fexpa(z1));
            vs2 = svadd_f32_x(pg, vs2, sve_fexpa(z2));
            vs3 = svadd_f32_x(pg, vs3, sve_fexpa(z3));
            vs4 = svadd_f32_x(pg, vs4, sve_fexpa(z4));
            vs5 = svadd_f32_x(pg, vs5, sve_fexpa(z5));
            vs6 = svadd_f32_x(pg, vs6, sve_fexpa(z6));
            vs7 = svadd_f32_x(pg, vs7, sve_fexpa(z7));
        }
        for (; j < blen1; j += VL) {
            svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(pg, bp + j), vlog2e);
            vs0 = svadd_f32_x(pg, vs0, sve_fexpa(z));
        }
        if (blen1 < blen) {
            svbool_t ptail = svwhilelt_b32((uint32_t)blen1, (uint32_t)blen);
            svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(ptail, bp + blen1), vlog2e);
            svfloat32_t e = svsel_f32(ptail, sve_fexpa(z), svdup_f32(0.0f));
            vs0 = svadd_f32_x(pg, vs0, e);
        }

        vs0 = svadd_f32_x(pg, vs0, vs1);
        vs2 = svadd_f32_x(pg, vs2, vs3);
        vs4 = svadd_f32_x(pg, vs4, vs5);
        vs6 = svadd_f32_x(pg, vs6, vs7);
        vs0 = svadd_f32_x(pg, vs0, vs2);
        vs4 = svadd_f32_x(pg, vs4, vs6);
        svfloat32_t vblk_sum = svadd_f32_x(pg, vs0, vs4);

        float sbuf[16] __attribute__((aligned(256)));
        svst1_f32(pg, sbuf, vblk_sum);
        float block_sum = 0.0f;
        for (int k = 0; k < VL; k++)
            block_sum += sbuf[k];

        running_sum += block_sum;
    }

    float loss = -logits[target] + global_max + logf(running_sum);

    /* ── Phase 2: gradient pass (1 L2 read + 1 write) ── */
    float bias = -global_max * LOG2E + shift_f;
    svfloat32_t vbias  = svdup_f32(bias);
    svfloat32_t vlog2e = svdup_f32(LOG2E);
    svfloat32_t vinv   = svdup_f32(1.0f / running_sum);

    int V8a = V & ~(V8 - 1);
    int V1  = V & ~(VL - 1);

    int i = 0;
    for (; i < V8a; i += V8) {
        prefetch_l1(logits + i, PF_DIST_BYTES);
        prefetch_l1(logits + i, PF_DIST_BYTES + 256);
        svfloat32_t z0 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 0 * VL), vlog2e);
        svfloat32_t z1 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 1 * VL), vlog2e);
        svfloat32_t z2 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 2 * VL), vlog2e);
        svfloat32_t z3 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 3 * VL), vlog2e);
        svfloat32_t z4 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 4 * VL), vlog2e);
        svfloat32_t z5 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 5 * VL), vlog2e);
        svfloat32_t z6 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 6 * VL), vlog2e);
        svfloat32_t z7 = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i + 7 * VL), vlog2e);
        svst1_f32(pg, grad + i + 0 * VL, svmul_f32_x(pg, sve_fexpa(z0), vinv));
        svst1_f32(pg, grad + i + 1 * VL, svmul_f32_x(pg, sve_fexpa(z1), vinv));
        svst1_f32(pg, grad + i + 2 * VL, svmul_f32_x(pg, sve_fexpa(z2), vinv));
        svst1_f32(pg, grad + i + 3 * VL, svmul_f32_x(pg, sve_fexpa(z3), vinv));
        svst1_f32(pg, grad + i + 4 * VL, svmul_f32_x(pg, sve_fexpa(z4), vinv));
        svst1_f32(pg, grad + i + 5 * VL, svmul_f32_x(pg, sve_fexpa(z5), vinv));
        svst1_f32(pg, grad + i + 6 * VL, svmul_f32_x(pg, sve_fexpa(z6), vinv));
        svst1_f32(pg, grad + i + 7 * VL, svmul_f32_x(pg, sve_fexpa(z7), vinv));
    }
    for (; i < V1; i += VL) {
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(pg, logits + i), vlog2e);
        svst1_f32(pg, grad + i, svmul_f32_x(pg, sve_fexpa(z), vinv));
    }
    if (V1 < V) {
        svbool_t ptail = svwhilelt_b32((uint32_t)V1, (uint32_t)V);
        svfloat32_t z = svmla_f32_x(pg, vbias, svld1_f32(ptail, logits + V1), vlog2e);
        svst1_f32(ptail, grad + V1, svmul_f32_x(pg, sve_fexpa(z), vinv));
    }

    grad[target] -= 1.0f;
    return loss;
}

/* ════════════════════════════════════════════════════════════════
 * Batch forward with OpenMP
 * ════════════════════════════════════════════════════════════════ */

void cross_entropy_batch_f32(const float *logits, const int *targets,
                             float *losses, int batch_tokens, int V) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int t = 0; t < batch_tokens; t++) {
        losses[t] = cross_entropy_fwd_f32(logits + (int64_t)t * V,
                                           targets[t], V);
    }
}

/* ════════════════════════════════════════════════════════════════
 * Scalar baseline (for accuracy comparison)
 * ════════════════════════════════════════════════════════════════ */

float cross_entropy_scalar_f32(const float *logits, int target, int V) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < V; i++)
        max_val = fmaxf(max_val, logits[i]);

    float sum_exp = 0.0f;
    for (int i = 0; i < V; i++)
        sum_exp += expf(logits[i] - max_val);

    return -logits[target] + max_val + logf(sum_exp);
}

/* ════════════════════════════════════════════════════════════════
 * Double-precision reference
 * ════════════════════════════════════════════════════════════════ */

double cross_entropy_ref_f64(const float *logits, int target, int V) {
    double max_val = -1e30;
    for (int i = 0; i < V; i++)
        if ((double)logits[i] > max_val) max_val = (double)logits[i];

    double sum_exp = 0.0;
    for (int i = 0; i < V; i++)
        sum_exp += exp((double)logits[i] - max_val);

    return -(double)logits[target] + max_val + log(sum_exp);
}
