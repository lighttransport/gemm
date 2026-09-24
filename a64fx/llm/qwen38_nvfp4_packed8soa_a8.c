/* Experimental same-size SoA repack for single-token packed NVFP4 W4A8. */
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct { float d[8]; uint8_t qs[64]; } q38_a8_source_subblock;
typedef struct { q38_a8_source_subblock s[4]; } q38_a8_source_block;
typedef struct { int8_t lo[64], hi[64]; } q38_a8_act;
typedef struct {
    uint8_t codes[4][64];
    float scales[4][8];
} q38_a8_soa_block;
_Static_assert(sizeof(q38_a8_soa_block) == 384, "SoA tile width");

int q38_nvfp4_packed8soa_repack(void *dst, const void *source, int blocks) {
    if (!dst || !source || blocks <= 0) return 0;
    q38_a8_soa_block *out = dst;
    const q38_a8_source_block *in = source;
    for (int b = 0; b < blocks; b++)
        for (int s = 0; s < 4; s++) {
            memcpy(out[b].codes[s], in[b].s[s].qs, 64);
            memcpy(out[b].scales[s], in[b].s[s].d, 8 * sizeof(float));
        }
    return 1;
}

int q38_nvfp4_packed8soa_a8_rows(float *y, const void *weights,
                                   const q38_a8_act *act, float act_scale,
                                   int rows, int cols) {
    if (svcntb() != 64 || rows <= 0 || rows % 8 ||
        cols <= 0 || cols % 64) return 0;
    static const int8_t code[64] = {
        0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
    };
    const svbool_t pb = svptrue_b8(), pf = svptrue_b32();
    const svbool_t p8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
    const svint8_t lut = svld1_s8(pb, code);
    const q38_a8_soa_block *base = weights;
    int nb = cols / 64;
    for (int row = 0; row < rows; row += 8) {
        svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0;
        const q38_a8_soa_block *w = base + (size_t)(row / 8) * nb;
        for (int ib = 0; ib < nb; ib++)
            for (int s = 0; s < 4; s++) {
                const q38_a8_act *xq = &act[ib * 4 + s];
                svuint8_t z = svld1_u8(pb,w[ib].codes[s]);
                svint8_t lo = svtbl_s8(lut,svand_n_u8_x(pb,z,15));
                svint8_t hi = svtbl_s8(lut,svlsr_n_u8_x(pb,z,4));
                svint8_t xl = svld1_s8(pb,xq->lo);
                svint8_t xh = svld1_s8(pb,xq->hi);
                svint32_t d0=svdot_s32(svdup_s32(0),lo,xl);
                svint32_t d1=svdot_s32(svdup_s32(0),hi,xh);
                svfloat32_t ds=svzip1_f32(
                    svld1(p8,w[ib].scales[s]),
                    svld1(p8,w[ib].scales[s]));
                svfloat32_t v=svmul_f32_x(pf,svcvt_f32_s32_x(pf,
                    svadd_s32_x(pf,d0,d1)),ds);
                switch (s) {
                case 0: a0=svadd_f32_x(pf,a0,v); break;
                case 1: a1=svadd_f32_x(pf,a1,v); break;
                case 2: a2=svadd_f32_x(pf,a2,v); break;
                default:a3=svadd_f32_x(pf,a3,v); break;
                }
            }
        svfloat32_t total=svmul_n_f32_x(pf,
            svadd_f32_x(pf,svadd_f32_x(pf,a0,a1),
                            svadd_f32_x(pf,a2,a3)),act_scale);
        svst1(p8,y+row,svadd_f32_x(p8,svuzp1_f32(total,total),
                                     svuzp2_f32(total,total)));
    }
    return 1;
}
