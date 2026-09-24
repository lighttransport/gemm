/* Experimental 5/6-bit execution layouts for the integer-scale W4A8 path.
 * Repacking preserves the integer-scale representation, including its existing
 * scale rounding. It does not introduce another weight quantization step. */
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct { int8_t lo[64], hi[64]; } q38_a8_act;
typedef struct {
    uint8_t codes[4][64];
    int8_t multipliers[4][16];
    float base[8];
} q38_iscale_block;
typedef struct {
    uint8_t nibbles[4][64];
    uint8_t signs[4][2][8];
    int8_t multipliers[4][16];
    float base[8];
} q38_packed5_block;
typedef struct {
    int8_t expanded[2][2][64];
    uint8_t codes[2][64];
    int8_t multipliers[4][16];
    float base[8];
} q38_packed6_block;
_Static_assert(sizeof(q38_packed5_block) == 416, "5-bit execution tile");
_Static_assert(sizeof(q38_packed6_block) == 480, "6-bit execution tile");
extern int q38_nvfp4_packed8_iscale_repack(void *, const void *, int);
static const int8_t fp4_code[64] = {
    0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
};

int q38_nvfp4_packed5_repack(void *dst, const void *source, int blocks) {
    if (!dst || !source || blocks <= 0) return 0;
    q38_packed5_block *out = dst;
    for (int b = 0; b < blocks; b++) {
        q38_iscale_block w;
        if (!q38_nvfp4_packed8_iscale_repack(&w,
                (const uint8_t *)source + (size_t)b * 384, 1)) return 0;
        memset(out[b].signs, 0, sizeof(out[b].signs));
        memcpy(out[b].multipliers, w.multipliers, sizeof(w.multipliers));
        memcpy(out[b].base, w.base, sizeof(w.base));
        for (int s = 0; s < 4; s++)
            for (int j = 0; j < 64; j++) {
                int lo = fp4_code[w.codes[s][j] & 15];
                int hi = fp4_code[w.codes[s][j] >> 4];
                out[b].nibbles[s][j] = (uint8_t)((lo & 15) | ((hi & 15) << 4));
                if (lo < 0) out[b].signs[s][0][j / 8] |= (uint8_t)(1u << (j % 8));
                if (hi < 0) out[b].signs[s][1][j / 8] |= (uint8_t)(1u << (j % 8));
            }
    }
    return 1;
}

int q38_nvfp4_packed6_repack(void *dst, const void *source, int blocks) {
    if (!dst || !source || blocks <= 0) return 0;
    q38_packed6_block *out = dst;
    for (int b = 0; b < blocks; b++) {
        q38_iscale_block w;
        if (!q38_nvfp4_packed8_iscale_repack(&w,
                (const uint8_t *)source + (size_t)b * 384, 1)) return 0;
        memcpy(out[b].multipliers, w.multipliers, sizeof(w.multipliers));
        memcpy(out[b].base, w.base, sizeof(w.base));
        memcpy(out[b].codes, w.codes[2], sizeof(out[b].codes));
        for (int s = 0; s < 2; s++)
            for (int j = 0; j < 64; j++) {
                out[b].expanded[s][0][j] = fp4_code[w.codes[s][j] & 15];
                out[b].expanded[s][1][j] = fp4_code[w.codes[s][j] >> 4];
            }
    }
    return 1;
}

static inline svbool_t load_signs(const uint8_t *bits) {
    svbool_t signs;
    __asm__("ldr %0, [%1]" : "=Upa"(signs) : "r"(bits) : "memory");
    return signs;
}

/* Keep each subblock's dot product independent before the scale multiply. */
#define Q38_ACCUMULATE() \
    svint32_t part = svadd_s32_x(pf, \
        svdot_s32(svdup_s32(0), lo, svld1_s8(pb, xq->lo)), \
        svdot_s32(svdup_s32(0), hi, svld1_s8(pb, xq->hi))); \
    svint32_t scale = svld1sb_s32(pf, w[ib].multipliers[s]); \
    switch (s) { \
    case 0: a0 = svmul_s32_x(pf, part, scale); break; \
    case 1: a1 = svmul_s32_x(pf, part, scale); break; \
    case 2: a2 = svmul_s32_x(pf, part, scale); break; \
    default:a3 = svmul_s32_x(pf, part, scale); break; \
    }

#define Q38_FINISH_TILE() \
    svint32_t tile = svadd_s32_x(pf, \
        svadd_s32_x(pf, a0, a1), svadd_s32_x(pf, a2, a3)); \
    svfloat32_t scale = svld1_f32(p8, w[ib].base); \
    sum = svmla_f32_x(pf, sum, svcvt_f32_s32_x(pf, tile), \
                      svzip1_f32(scale, scale))

#define Q38_FINISH_ROW() \
    sum = svmul_n_f32_x(pf, sum, act_scale); \
    svst1_f32(p8, y + row, \
        svadd_f32_x(p8, svuzp1_f32(sum, sum), svuzp2_f32(sum, sum)))

int q38_nvfp4_packed5_a8_rows(float *y, const void *weights,
        const q38_a8_act *act, float act_scale, int rows, int cols) {
    if (svcntb() != 64 || !y || !weights || !act || rows <= 0 ||
        rows % 8 || cols <= 0 || cols % 64) return 0;
    const svbool_t pb = svptrue_b8(), pf = svptrue_b32();
    const svbool_t p8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
    const int nb = cols / 64;
    for (int row = 0; row < rows; row += 8) {
        const q38_packed5_block *w = (const q38_packed5_block *)weights +
                                   (size_t)(row / 8) * nb;
        svfloat32_t sum = svdup_f32(0);
        for (int ib = 0; ib < nb; ib++) {
            svint32_t a0 = svdup_s32(0), a1 = a0, a2 = a0, a3 = a0;
            for (int s = 0; s < 4; s++) {
                const q38_a8_act *xq = &act[ib * 4 + s];
                svuint8_t z = svld1_u8(pb, w[ib].nibbles[s]);
                svint8_t lo = svreinterpret_s8_u8(svand_n_u8_x(pb, z, 15));
                svint8_t hi = svreinterpret_s8_u8(svlsr_n_u8_x(pb, z, 4));
                lo = svsub_n_s8_m(load_signs(w[ib].signs[s][0]), lo, 16);
                hi = svsub_n_s8_m(load_signs(w[ib].signs[s][1]), hi, 16);
                Q38_ACCUMULATE();
            }
            Q38_FINISH_TILE();
        }
        Q38_FINISH_ROW();
    }
    return 1;
}

int q38_nvfp4_packed6_a8_rows(float *y, const void *weights,
        const q38_a8_act *act, float act_scale, int rows, int cols) {
    if (svcntb() != 64 || !y || !weights || !act || rows <= 0 ||
        rows % 8 || cols <= 0 || cols % 64) return 0;
    const svbool_t pb = svptrue_b8(), pf = svptrue_b32();
    const svbool_t p8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
    const svint8_t lut = svld1_s8(pb, fp4_code);
    const int nb = cols / 64;
    for (int row = 0; row < rows; row += 8) {
        const q38_packed6_block *w = (const q38_packed6_block *)weights +
                                   (size_t)(row / 8) * nb;
        svfloat32_t sum = svdup_f32(0);
        for (int ib = 0; ib < nb; ib++) {
            svint32_t a0 = svdup_s32(0), a1 = a0, a2 = a0, a3 = a0;
            for (int s = 0; s < 4; s++) {
                const q38_a8_act *xq = &act[ib * 4 + s];
                svint8_t lo, hi;
                if (s < 2) {
                    lo = svld1_s8(pb, w[ib].expanded[s][0]);
                    hi = svld1_s8(pb, w[ib].expanded[s][1]);
                } else {
                    svuint8_t z = svld1_u8(pb, w[ib].codes[s - 2]);
                    lo = svtbl_s8(lut, svand_n_u8_x(pb, z, 15));
                    hi = svtbl_s8(lut, svlsr_n_u8_x(pb, z, 4));
                }
                Q38_ACCUMULATE();
            }
            Q38_FINISH_TILE();
        }
        Q38_FINISH_ROW();
    }
    return 1;
}
