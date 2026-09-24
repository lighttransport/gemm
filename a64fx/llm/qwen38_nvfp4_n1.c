/* Exact single-token packed NVFP4 GEMV on 512-bit A64FX SVE. */
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

typedef struct { float d[8]; uint8_t qs[64]; } q38_n1_subblock;
typedef struct { q38_n1_subblock s[4]; } q38_n1_block;
_Static_assert(sizeof(q38_n1_block) == 384, "packed tile width");
static const float q38_n1_code[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

int q38_nvfp4_packed_n1_rows(float *y, const void *weights, const float *x,
                              int rows, int cols) {
    if (svcntw() != 16 || rows % 8 || cols % 64) return 0;
    const svbool_t pg = svptrue_b32();
    const svbool_t p8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
    const svuint32_t repeat8 = svand_n_u32_x(pg, svindex_u32(0, 1), 7);
    const svfloat32_t lut = svld1(pg, q38_n1_code);
    const q38_n1_block *base = (const q38_n1_block *)weights;
    const int nb = cols / 64;
    for (int row = 0; row < rows; row += 8) {
        svfloat32_t a0=svdup_f32(0), a1=a0, a2=a0, a3=a0;
        const q38_n1_block *w = base + (size_t)(row / 8) * nb;
        for (int ib = 0; ib < nb; ib++)
            for (int s = 0; s < 4; s++) {
                const q38_n1_subblock *p = &w[ib].s[s];
                const float *xp = x + ib * 64 + s * 16;
                svfloat32_t xl = svtbl_f32(svld1(p8, xp), repeat8);
                svfloat32_t xh = svtbl_f32(svld1(p8, xp + 8), repeat8);
#define Q38_N1_PAIR(R,A) do { \
                    int rr = 2 * (R); \
                    svuint32_t z = svld1ub_u32(pg, p->qs + rr * 8); \
                    svfloat32_t d = svsel_f32(p8, svdup_f32(p->d[rr]), \
                                                    svdup_f32(p->d[rr + 1])); \
                    svfloat32_t lo = svmul_x(pg, \
                        svtbl_f32(lut, svand_n_u32_x(pg, z, 15)), d); \
                    svfloat32_t hi = svmul_x(pg, \
                        svtbl_f32(lut, svlsr_n_u32_x(pg, z, 4)), d); \
                    (A) = svmla_x(pg, (A), lo, xl); \
                    (A) = svmla_x(pg, (A), hi, xh); \
                } while (0)
                Q38_N1_PAIR(0,a0);
                Q38_N1_PAIR(1,a1);
                Q38_N1_PAIR(2,a2);
                Q38_N1_PAIR(3,a3);
#undef Q38_N1_PAIR
            }
        y[row] = svaddv_f32(p8, a0);
        y[row+1] = svaddv_f32(p8, svext_f32(a0, a0, 8));
        y[row+2] = svaddv_f32(p8, a1);
        y[row+3] = svaddv_f32(p8, svext_f32(a1, a1, 8));
        y[row+4] = svaddv_f32(p8, a2);
        y[row+5] = svaddv_f32(p8, svext_f32(a2, a2, 8));
        y[row+6] = svaddv_f32(p8, a3);
        y[row+7] = svaddv_f32(p8, svext_f32(a3, a3, 8));
    }
    return 1;
}
