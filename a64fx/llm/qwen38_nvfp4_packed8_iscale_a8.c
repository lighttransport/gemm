/* Experimental W4A8 single-token GEMV with one FP32 scale per 64-column tile. */
#include <arm_sve.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct { float d[8]; uint8_t qs[64]; } q38_source_subblock;
typedef struct { q38_source_subblock s[4]; } q38_source_block;
typedef struct { int8_t lo[64], hi[64]; } q38_a8_act;
typedef struct {
    uint8_t codes[4][64];
    int8_t multipliers[4][16];
    float base[8];
} q38_iscale_block;
_Static_assert(sizeof(q38_iscale_block) == 352, "integer-scale tile width");

int q38_nvfp4_packed8_iscale_repack(void *dst, const void *source,
                                     int blocks) {
    if (!dst || !source || blocks <= 0) return 0;
    q38_iscale_block *out = dst;
    const q38_source_block *in = source;
    for (int b = 0; b < blocks; b++) {
        for (int row = 0; row < 8; row++) {
            float base = INFINITY;
            for (int s = 0; s < 4; s++) {
                float d = in[b].s[s].d[row];
                if (!isfinite(d) || d < 0.0f) return 0;
                if (d > 0.0f && d < base) base = d;
            }
            if (!isfinite(base)) base = 1.0f;
            out[b].base[row] = base;
            for (int s = 0; s < 4; s++) {
                long q = lrintf(in[b].s[s].d[row] / base);
                if (q < 0 || q > 127) return 0;
                out[b].multipliers[s][2 * row] = (int8_t)q;
                out[b].multipliers[s][2 * row + 1] = (int8_t)q;
            }
        }
        for (int s = 0; s < 4; s++)
            memcpy(out[b].codes[s], in[b].s[s].qs, 64);
    }
    return 1;
}

int q38_nvfp4_packed8_iscale_a8_rows(float *y, const void *weights,
                                       const q38_a8_act *act,
                                       float act_scale, int rows, int cols) {
    if (svcntb() != 64 || !y || !weights || !act || rows <= 0 ||
        rows % 8 || cols <= 0 || cols % 64) return 0;
    static const int8_t code[64] = {
        0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
    };
    const svbool_t pb = svptrue_b8(), pf = svptrue_b32();
    const svbool_t p8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
    const svint8_t lut = svld1_s8(pb, code);
    const q38_iscale_block *base = weights;
    const int nb = cols / 64;
    for (int row = 0; row < rows; row += 8) {
        svfloat32_t sum = svdup_f32(0);
        const q38_iscale_block *w = base + (size_t)(row / 8) * nb;
        for (int ib = 0; ib < nb; ib++) {
            svint32_t a0=svdup_s32(0),a1=a0,a2=a0,a3=a0;
            for (int s = 0; s < 4; s++) {
                const q38_a8_act *xq = &act[ib * 4 + s];
                svuint8_t z = svld1_u8(pb, w[ib].codes[s]);
                svint8_t lo = svtbl_s8(lut, svand_n_u8_x(pb, z, 15));
                svint8_t hi = svtbl_s8(lut, svlsr_n_u8_x(pb, z, 4));
                svint8_t xl = svld1_s8(pb, xq->lo);
                svint8_t xh = svld1_s8(pb, xq->hi);
                svint32_t part = svadd_s32_x(pf,
                    svdot_s32(svdup_s32(0), lo, xl),
                    svdot_s32(svdup_s32(0), hi, xh));
                svint32_t scale = svld1sb_s32(pf, w[ib].multipliers[s]);
                switch (s) {
                case 0: a0 = svmla_s32_x(pf, a0, part, scale); break;
                case 1: a1 = svmla_s32_x(pf, a1, part, scale); break;
                case 2: a2 = svmla_s32_x(pf, a2, part, scale); break;
                default:a3 = svmla_s32_x(pf, a3, part, scale); break;
                }
            }
            svint32_t tile = svadd_s32_x(pf,
                svadd_s32_x(pf, a0, a1), svadd_s32_x(pf, a2, a3));
            svfloat32_t tile_scale = svzip1_f32(
                svld1_f32(p8, w[ib].base), svld1_f32(p8, w[ib].base));
            sum = svmla_f32_x(pf, sum, svcvt_f32_s32_x(pf, tile), tile_scale);
        }
        sum = svmul_n_f32_x(pf, sum, act_scale);
        svst1_f32(p8, y + row,
            svadd_f32_x(p8, svuzp1_f32(sum, sum), svuzp2_f32(sum, sum)));
    }
    return 1;
}
