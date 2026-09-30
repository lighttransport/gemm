/* Full-width SVE decode row kernels for native GGUF Q4_K / Q5_K weights (no repack).
 *
 * The reference row loops in glm53f_iq_bridge.c work on 32-byte halves of the 64-byte vector, rebuild every
 * sub-block scale with scalar code plus GPR->vector moves, and issue a second SDOT per group just for the
 * activation sums (~130 cycles per 256 columns).  Here one 512-bit SDOT covers two 32-column sub-blocks:
 *
 *   qs[0..63]   : lo nibbles = cols [0,32) | [64,96)     hi nibbles = cols [32,64) | [96,128)
 *   qs[64..127] : lo nibbles = cols [128,160) | [192,224) hi nibbles = cols [160,192) | [224,256)
 *
 * so the activation block is permuted once per input vector (iqf_prepare) and the four SDOTs per super-block
 * multiply by scale vectors {sc[a] x8 | sc[b] x8} built with TBL from the packed 12-byte scale field.  The
 * min term uses precomputed per-32 activation sums.  Integer sums are exact; only the float lane order of the
 * final reduction differs from the reference loops. */
#ifndef GLM53F_IQ_FAST_H
#define GLM53F_IQ_FAST_H

#include <arm_sve.h>
#include <stdint.h>
#include <string.h>

typedef struct { float d; int8_t q[256]; } iqf_src_block;   /* == glm5_iq_q8_block */

typedef struct {
    _Alignas(64) int8_t v[4][64];   /* lo0, hi0, lo1, hi1 (see file comment) */
    int32_t s[16];                  /* sums of the 8 32-column sub-blocks in column order (lanes 8..15 = 0) */
    float d;
    float pad[15];
} iqf_act;

static inline void iqf_prepare(iqf_act *o, const iqf_src_block *x, int blocks) {
    for (int b = 0; b < blocks; ++b) {
        const int8_t *q = x[b].q;
        for (int h = 0; h < 2; ++h) {
            memcpy(o[b].v[2 * h] + 0, q + h * 128 + 0, 32);
            memcpy(o[b].v[2 * h] + 32, q + h * 128 + 64, 32);
            memcpy(o[b].v[2 * h + 1] + 0, q + h * 128 + 32, 32);
            memcpy(o[b].v[2 * h + 1] + 32, q + h * 128 + 96, 32);
        }
        for (int g = 0; g < 8; ++g) {
            int32_t sum = 0;
            for (int i = 0; i < 32; ++i) sum += q[g * 32 + i];
            o[b].s[g] = sum;
        }
        for (int g = 8; g < 16; ++g) o[b].s[g] = 0;
        o[b].d = x[b].d;
    }
}

static const uint32_t iqf_idxA[16] = {0, 1, 2, 3, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint32_t iqf_idxB[16] = {0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint32_t iqf_idxC[16] = {4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint32_t iqf_idxD[16] = {0, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint32_t iqf_maskA[16] = {63, 63, 63, 63, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint32_t iqf_maskH[16] = {0, 0, 0, 0, 0xC0, 0xC0, 0xC0, 0xC0, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint32_t iqf_maskC[16] = {63, 63, 63, 63, 0xF0, 0xF0, 0xF0, 0xF0, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint32_t iqf_shiftC[16] = {0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint32_t iqf_sel0[16] = {0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2};
static const uint32_t iqf_sel1[16] = {1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3};
static const uint32_t iqf_sel2[16] = {4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6};
static const uint32_t iqf_sel3[16] = {5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7};

#ifndef IQF_PF_BYTES
#define IQF_PF_BYTES 3072
#endif
#define IQF_DECL_K \
    const svbool_t iqf_p32 = svptrue_b32(), iqf_p8 = svptrue_b8(), iqf_pg8 = svwhilelt_b32(0, 8), iqf_pb32 = svwhilelt_b8(0, 32); \
    const svuint32_t k_iA = svld1_u32(iqf_p32, iqf_idxA), k_iB = svld1_u32(iqf_p32, iqf_idxB), k_iC = svld1_u32(iqf_p32, iqf_idxC), \
        k_iD = svld1_u32(iqf_p32, iqf_idxD), k_mA = svld1_u32(iqf_p32, iqf_maskA), k_mH = svld1_u32(iqf_p32, iqf_maskH), \
        k_mC = svld1_u32(iqf_p32, iqf_maskC), k_sC = svld1_u32(iqf_p32, iqf_shiftC), k_t0 = svld1_u32(iqf_p32, iqf_sel0), \
        k_t1 = svld1_u32(iqf_p32, iqf_sel1), k_t2 = svld1_u32(iqf_p32, iqf_sel2), k_t3 = svld1_u32(iqf_p32, iqf_sel3); \
    const svuint8_t k_vrep = svld1_u8(iqf_p8, iqf_rep), k_vs0 = svld1_u8(iqf_p8, iqf_sh0), k_vs1 = svld1_u8(iqf_p8, iqf_sh1), \
        k_vs2 = svld1_u8(iqf_p8, iqf_sh2), k_vs3 = svld1_u8(iqf_p8, iqf_sh3);

static uint8_t iqf_rep[64], iqf_sh0[64], iqf_sh1[64], iqf_sh2[64], iqf_sh3[64];
static inline void iqf_init(void) {
    for (int i = 0; i < 64; ++i) {
        iqf_rep[i] = (uint8_t)(i & 31);
        const int half = i >> 5;
        iqf_sh0[i] = (uint8_t)(0 + 2 * half); iqf_sh1[i] = (uint8_t)(1 + 2 * half);
        iqf_sh2[i] = (uint8_t)(4 + 2 * half); iqf_sh3[i] = (uint8_t)(5 + 2 * half);
    }
}

static inline __attribute__((always_inline)) svfloat32_t iqf_block(svfloat32_t acc, const uint8_t *blk, const iqf_act *ab, int q5,
    svbool_t p32, svbool_t p8, svbool_t pg8, svbool_t pb32,
    svuint32_t iA, svuint32_t iB, svuint32_t iC, svuint32_t iD, svuint32_t mA, svuint32_t mH, svuint32_t mC, svuint32_t sC,
    svuint32_t t0, svuint32_t t1, svuint32_t t2, svuint32_t t3, svuint8_t vrep, svuint8_t vs0, svuint8_t vs1, svuint8_t vs2, svuint8_t vs3) {
    __builtin_prefetch(blk + IQF_PF_BYTES, 0, 1);
    const float dd = (float)*(const __fp16 *)blk, dm = (float)*(const __fp16 *)(blk + 2);
    const svuint32_t Q = svld1ub_u32(p32, blk + 4);
    const svuint32_t va = svtbl_u32(Q, iA), vb = svtbl_u32(Q, iB), vc = svtbl_u32(Q, iC), vd = svtbl_u32(Q, iD);
    const svuint32_t sc = svorr_u32_x(p32, svand_u32_x(p32, va, mA),
                                      svlsr_n_u32_x(p32, svand_u32_x(p32, vb, mH), 2));
    const svuint32_t mn = svorr_u32_x(p32, svlsr_u32_x(p32, svand_u32_x(p32, vc, mC), sC),
                                      svlsr_n_u32_x(p32, svand_u32_x(p32, vd, mH), 2));
    const uint8_t *qs = blk + (q5 ? 48 : 16);
    const svuint8_t qa = svld1_u8(p8, qs), qb = svld1_u8(p8, qs + 64);
    svuint8_t l0 = svand_n_u8_x(p8, qa, 15), h0 = svlsr_n_u8_x(p8, qa, 4);
    svuint8_t l1 = svand_n_u8_x(p8, qb, 15), h1 = svlsr_n_u8_x(p8, qb, 4);
    if (q5) {
        const svuint8_t hv = svtbl_u8(svld1_u8(pb32, blk + 16), vrep);
        l0 = svadd_u8_x(p8, l0, svlsl_n_u8_x(p8, svand_n_u8_x(p8, svlsr_u8_x(p8, hv, vs0), 1), 4));
        h0 = svadd_u8_x(p8, h0, svlsl_n_u8_x(p8, svand_n_u8_x(p8, svlsr_u8_x(p8, hv, vs1), 1), 4));
        l1 = svadd_u8_x(p8, l1, svlsl_n_u8_x(p8, svand_n_u8_x(p8, svlsr_u8_x(p8, hv, vs2), 1), 4));
        h1 = svadd_u8_x(p8, h1, svlsl_n_u8_x(p8, svand_n_u8_x(p8, svlsr_u8_x(p8, hv, vs3), 1), 4));
    }
    const svint32_t z = svdup_n_s32(0);
    const svint32_t d0 = svdot_s32(z, svreinterpret_s8_u8(l0), svld1_s8(p8, ab->v[0]));
    const svint32_t d1 = svdot_s32(z, svreinterpret_s8_u8(h0), svld1_s8(p8, ab->v[1]));
    const svint32_t d2 = svdot_s32(z, svreinterpret_s8_u8(l1), svld1_s8(p8, ab->v[2]));
    const svint32_t d3 = svdot_s32(z, svreinterpret_s8_u8(h1), svld1_s8(p8, ab->v[3]));
    svint32_t dots = svmul_s32_x(p32, d0, svreinterpret_s32_u32(svtbl_u32(sc, t0)));
    dots = svmla_s32_x(p32, dots, d1, svreinterpret_s32_u32(svtbl_u32(sc, t1)));
    dots = svmla_s32_x(p32, dots, d2, svreinterpret_s32_u32(svtbl_u32(sc, t2)));
    dots = svmla_s32_x(p32, dots, d3, svreinterpret_s32_u32(svtbl_u32(sc, t3)));
    const svint32_t msum = svmul_s32_z(pg8, svreinterpret_s32_u32(mn), svld1_s32(pg8, ab->s));
    acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, dots), ab->d * dd);
    return svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, msum), -ab->d * dm);
}

#define IQF_CALL(acc, blk, ab) iqf_block(acc, blk, ab, q5, iqf_p32, iqf_p8, iqf_pg8, iqf_pb32, k_iA, k_iB, k_iC, k_iD, k_mA, k_mH, k_mC, k_sC, k_t0, k_t1, k_t2, k_t3, k_vrep, k_vs0, k_vs1, k_vs2, k_vs3)

/* out[i] = dot(row0 + i*rb, activation) for i in [0, nrows).  q5 = 0: Q4_K (144 B/block), 1: Q5_K (176 B/block).
 * Two rows are interleaved so the TBL/SDOT latency chains of one row hide behind the other. iqf_init() must
 * have run once. */
static inline void iqf_rows(float *out, const uint8_t *row0, size_t rb, int nrows, const iqf_act *a, int blocks,
                            int q5) {
    IQF_DECL_K
    const size_t bsz = q5 ? 176 : 144;
    int r = 0;
    for (; r + 1 < nrows; r += 2) {
        const uint8_t *rowa = row0 + (size_t)r * rb, *rowb = rowa + rb;
        svfloat32_t acca = svdup_f32(0.0f), accb = acca;
        for (int b = 0; b < blocks; ++b) {
            acca = IQF_CALL(acca, rowa + (size_t)b * bsz, a + b);
            accb = IQF_CALL(accb, rowb + (size_t)b * bsz, a + b);
        }
        out[r] = svaddv_f32(iqf_p32, acca);
        out[r + 1] = svaddv_f32(iqf_p32, accb);
    }
    for (; r < nrows; ++r) {
        const uint8_t *row = row0 + (size_t)r * rb;
        svfloat32_t acc = svdup_f32(0.0f);
        for (int b = 0; b < blocks; ++b) acc = IQF_CALL(acc, row + (size_t)b * bsz, a + b);
        out[r] = svaddv_f32(iqf_p32, acc);
    }
}

#endif
