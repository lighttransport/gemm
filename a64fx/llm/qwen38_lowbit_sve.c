#include "qwen38_lowbit.h"
#include <arm_sve.h>
#include <float.h>
#include <math.h>

static const int8_t fp4_lut[64] = {
    0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
};
/* Exact E2M3 values multiplied by eight. No rounding in fused dequant. */
static const int8_t fp6_lut[64] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,
    0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,
    -16,-18,-20,-22,-24,-26,-28,-30,-32,-36,-40,-44,-48,-52,-56,-60
};

#include "qwen38_lowbit_scale.inc"

int q38_lowbit_prepare_sve(q38_lowbit_act *out, size_t count,
                          const float *x, int cols, int arithmetic) {
    if (svcntb() != 64 || !out || !x || cols <= 0 ||
        count < ((size_t)cols + 15) / 16 ||
        (arithmetic != Q38_LB_A8 && arithmetic != Q38_LB_A16)) return 0;
    svbool_t pb = svptrue_b8(), pf = svptrue_b32(), pd = svptrue_b64();
    for (int k = 0; k < cols; k += 16) {
        svbool_t pg = svwhilelt_b32(k, cols);
        svfloat32_t v = svld1_f32(pg, x + k);
        if (svptest_any(pg, svcmpuo_f32(pg, v, v)) ||
            svptest_any(pg, svcmpge_n_f32(pg, svabs_f32_x(pg, v), INFINITY))) return 0;
    }
    /* Select each low byte from eight int64 lanes, then replicate that row
     * eight times directly into the SDOT operand. No scalar byte scatter. */
    svuint8_t index = svlsl_n_u8_x(pb, svand_n_u8_x(pb, svindex_u8(0, 1), 7), 3);
    int limit = arithmetic == Q38_LB_A8 ? 127 : 32639;
    for (int k = 0; k < cols; k += 32) {
        int n = cols - k < 32 ? cols - k : 32;
        svfloat32_t a = svabs_f32_x(pf, svld1_f32(svwhilelt_b32(0, n), x + k));
        if (n > 16) a = svmax_f32_x(pf, a,
            svabs_f32_x(pf, svld1_f32(svwhilelt_b32(16, n), x + k + 16)));
        float amax = svmaxv_f32(pf, a);
        float scale = amax ? (float)((double)amax / limit) : 1.0f;
        if (!scale) scale = FLT_TRUE_MIN;
        double inverse = 1.0 / (double)scale, midpoint = .5 * (double)scale;
        for (int s = 0; s < (n + 15) / 16; s++) {
            q38_lowbit_act *dst = out + k / 16 + s;
            dst->scale = scale;
            for (int half = 0; half < 2; half++) {
                int first = k + s * 16 + half * 8;
                /* LD1W into D lanes places each FP32 value in the low half
                 * of a D lane, precisely the widening FCVT source layout. */
                svuint64_t bits = svdup_u64(0);
                if (first < cols) bits = svld1uw_u64(svwhilelt_b64(first, cols),
                                                    (const uint32_t *)(x + first));
                svfloat64_t v = svcvt_f64_f32_x(pd, svreinterpret_f32_u64(bits));
                svfloat64_t rounded = svrintn_f64_x(pd, svmul_n_f64_x(pd, v, inverse));
                svint64_t q = svcvt_s64_f64_x(pd, rounded);
                /* The reciprocal estimate can straddle an exact half tie.
                 * Correct it using x-q*scale, not the rounded quotient.
                 * FP32 inputs/scales and |q|<=32639 make this residual exact
                 * in FP64. This preserves scalar ties-to-even while sharing
                 * one division across 32 activations. */
                svfloat64_t residual = svmls_n_f64_x(pd, v, rounded, (double)scale);
                svbool_t odd = svcmpne_n_s64(pd, svand_n_s64_x(pd, q, 1), 0);
                svbool_t up = svorr_b_z(pd, svcmpgt_n_f64(pd, residual, midpoint),
                    svand_b_z(pd, odd, svcmpeq_n_f64(pd, residual, midpoint)));
                svbool_t down = svorr_b_z(pd, svcmplt_n_f64(pd, residual, -midpoint),
                    svand_b_z(pd, odd, svcmpeq_n_f64(pd, residual, -midpoint)));
                q = svadd_n_s64_m(up, q, 1);
                q = svsub_n_s64_m(down, q, 1);
                q = svmax_n_s64_x(pd, svmin_n_s64_x(pd, q, limit), -limit);
                svint8_t lo = svtbl_s8(svreinterpret_s8_s64(q), index);
                svint8_t hi = svdup_s8(0);
                if (arithmetic == Q38_LB_A16) {
                    svint64_t high = svasr_n_s64_x(pd, svadd_n_s64_x(pd, q, 128), 8);
                    hi = svtbl_s8(svreinterpret_s8_s64(high), index);
                }
                svst1_s8(pb, dst->lo[half], lo);
                svst1_s8(pb, dst->hi[half], hi);
            }
        }
    }
    return 1;
}

int q38_lowbit_sve_f32(float *out, const void *weights, int format,
                     const float *x, int rows, int cols) {
    if (svcntb() != 64 || !out || !weights || !x ||
        !q38_lowbit_bytes(format, rows, cols)) return 0;
    svbool_t pb = svptrue_b8(), pf = svptrue_b32();
    svbool_t p8 = svwhilelt_b8((uint64_t)0, (uint64_t)8);
    svbool_t p4 = svwhilelt_b8((uint64_t)0, (uint64_t)4);
    svint8_t lut = svld1_s8(pb, format == Q38_LB_NVFP4 ? fp4_lut : fp6_lut);
    size_t nb = ((size_t)cols + 63) / 64;
    for (int r = 0; r < rows; r++) {
        svfloat32_t sum = svdup_f32(0);
        for (int k = 0; k < cols; k += 16) {
            size_t b = (size_t)(r / 8) * nb + k / 64;
            int s = k % 64 / 16, rr = r % 8;
            const q38_fp4_tile *w4 = (const q38_fp4_tile *)weights + b;
            const q38_fp6_tile *w6 = (const q38_fp6_tile *)weights + b;
            svuint8_t codes = svld1_u8(p8, format == Q38_LB_NVFP4 ?
                w4->codes[s] + rr * 8 : w6->low[s] + rr * 8);
            svuint8_t lo = svand_n_u8_x(pb, codes, 15), hi = svlsr_n_u8_x(pb, codes, 4);
            float scale;
            if (format == Q38_LB_FP6_E2M3) {
                svuint8_t h = svld1_u8(p4, w6->high[s] + rr * 4);
                h = svzip1_u8(svand_n_u8_x(pb, h, 15), svlsr_n_u8_x(pb, h, 4));
                lo = svorr_u8_x(pb, lo, svlsl_n_u8_x(pb, svand_n_u8_x(pb, h, 3), 4));
                hi = svorr_u8_x(pb, hi, svlsl_n_u8_x(pb, svlsr_n_u8_x(pb, h, 2), 4));
                scale = fp6_scales[w6->scale[s / 2][rr]];
            } else scale = fp4_scales[w4->scale[s][rr]];
            svint8_t q = svsplice_s8(p8, svtbl_s8(lut, lo), svtbl_s8(lut, hi));
            svfloat32_t w = svcvt_f32_s32_x(pf, svunpklo_s32(svunpklo_s16(q)));
            w = svmul_n_f32_x(pf, w, scale);
            svbool_t tail = svwhilelt_b32(k, cols);
            sum = svmla_f32_m(tail, sum, w, svld1_f32(tail, x + k));
        }
        out[r] = svaddv_f32(pf, sum);
    }
    return 1;
}
static inline svfloat32_t load_scale(const uint8_t *src, int format) {
    svbool_t p8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
    svuint32_t indices = svld1ub_u32(p8, src);
    svfloat32_t scale = svld1_gather_u32index_f32(p8,
        format == Q38_LB_NVFP4 ? fp4_scales : fp6_scales, indices);
    return svzip1_f32(scale, scale);
}

static inline void kernel(float *out, const void *weights, int format,
                          const q38_lowbit_act *act, int arithmetic,
                          int rows, int cols) {
    const svbool_t pb = svptrue_b8(), pf = svptrue_b32();
    const svbool_t p32 = svwhilelt_b8((uint64_t)0, (uint64_t)32);
    const svint8_t lut = svld1_s8(pb, format == Q38_LB_NVFP4 ? fp4_lut : fp6_lut);
    const size_t nb = ((size_t)cols + 63) / 64;
    for (int row = 0; row < rows; row += 8) {
        svfloat32_t acc0 = svdup_f32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        for (size_t ib = 0; ib < nb; ib++) {
            size_t index = (size_t)(row / 8) * nb + ib;
            const q38_fp4_tile *w4 = (const q38_fp4_tile *)weights + index;
            const q38_fp6_tile *w6 = (const q38_fp6_tile *)weights + index;
            int step = format == Q38_LB_NVFP4 ? 1 : 2;
            for (int first = 0; first < 4; first += step) {
                if (ib * 64 + (size_t)first * 16 >= (size_t)cols) break;
                svint32_t dot = svdup_s32(0);
                for (int s = first; s < first + step; s++) {
                    if (ib * 64 + (size_t)s * 16 >= (size_t)cols) break;
                    const q38_lowbit_act *a = act + ib * 4 + s;
                    svuint8_t z = svld1_u8(pb, format == Q38_LB_NVFP4 ? w4->codes[s] : w6->low[s]);
                    svuint8_t il = svand_n_u8_x(pb, z, 15), ih = svlsr_n_u8_x(pb, z, 4);
                    if (format == Q38_LB_FP6_E2M3) {
                        svuint8_t h = svld1_u8(p32, w6->high[s]);
                        h = svzip1_u8(svand_n_u8_x(pb, h, 15), svlsr_n_u8_x(pb, h, 4));
                        il = svorr_u8_x(pb, il, svlsl_n_u8_x(pb, svand_n_u8_x(pb, h, 3), 4));
                        ih = svorr_u8_x(pb, ih, svlsl_n_u8_x(pb, svlsr_n_u8_x(pb, h, 2), 4));
                    }
                    svint8_t lo = svtbl_s8(lut, il), hi = svtbl_s8(lut, ih);
                    svint32_t part = svdot_s32(svdup_s32(0), lo, svld1_s8(pb, a->lo[0]));
                    part = svdot_s32(part, hi, svld1_s8(pb, a->lo[1]));
                    if (arithmetic == Q38_LB_A16) {
                        svint32_t high = svdot_s32(svdup_s32(0), lo, svld1_s8(pb, a->hi[0]));
                        high = svdot_s32(high, hi, svld1_s8(pb, a->hi[1]));
                        part = svadd_s32_x(pf, part, svmul_n_s32_x(pf, high, 256));
                    }
                    dot = svadd_s32_x(pf, dot, part);
                }
                /* Each lane holds at most 16 products, each <=60*32639.
                 * Reduction per 32-column block bounds INT32 at 31,333,440,
                 * independently of K. FP4's bound is smaller. */
                svfloat32_t scale = load_scale(format == Q38_LB_NVFP4 ?
                    w4->scale[first] : w6->scale[first / 2], format);
                scale = svmul_n_f32_x(pf, scale, act[ib * 4 + first].scale);
                svfloat32_t value = svcvt_f32_s32_x(pf, dot);
                switch (first) {
                case 0: acc0 = svmla_f32_x(pf, acc0, value, scale); break;
                case 1: acc1 = svmla_f32_x(pf, acc1, value, scale); break;
                case 2: acc2 = svmla_f32_x(pf, acc2, value, scale); break;
                default: acc3 = svmla_f32_x(pf, acc3, value, scale); break;
                }
            }
        }
        svfloat32_t sum = svadd_f32_x(pf, svadd_f32_x(pf, acc0, acc1), svadd_f32_x(pf, acc2, acc3));
        svbool_t tail = svwhilelt_b32((uint64_t)0, (uint64_t)(rows - row < 8 ? rows - row : 8));
        svst1_f32(tail, out + row, svadd_f32_x(tail, svuzp1_f32(sum, sum), svuzp2_f32(sum, sum)));
    }
}

int q38_lowbit_sve(float *out, const void *weights, int format,
                   const q38_lowbit_act *act, int arithmetic, int rows, int cols) {
    if (svcntb() != 64 || !out || !weights || !act ||
        !q38_lowbit_bytes(format, rows, cols)) return 0;
    if (format == Q38_LB_NVFP4) {
        if (arithmetic == Q38_LB_A8) kernel(out, weights, Q38_LB_NVFP4, act, Q38_LB_A8, rows, cols);
        else if (arithmetic == Q38_LB_A16) kernel(out, weights, Q38_LB_NVFP4, act, Q38_LB_A16, rows, cols);
        else return 0;
    } else {
        if (arithmetic == Q38_LB_A8) kernel(out, weights, Q38_LB_FP6_E2M3, act, Q38_LB_A8, rows, cols);
        else if (arithmetic == Q38_LB_A16) kernel(out, weights, Q38_LB_FP6_E2M3, act, Q38_LB_A16, rows, cols);
        else return 0;
    }
    return 1;
}
