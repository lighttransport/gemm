/* Grouped routed-expert prefill kernels that read NATIVE GGUF Q4_K / Q5_K rows.
 *
 * A 64-row x 256-column superblock is expanded with SVE gathers straight from
 * the row-major GGUF blocks into the int8 "panel64" tile layout consumed by
 * gk_gemm_tile6x4p_asm (kern/glm53f_kern_gemm_asm.S).  No repacked copy of the
 * weights exists, so decode keeps reading the identical blob.
 *
 * Q4_K/Q5_K value:  w = d*sc_b*q - dmin*m_b   (b = 32-column sub-block)
 *   y[t][r] = sum_b d*sc_b * xs[t][b] * dot(q, xq)   (int8 SDOT, tile kernel)
 *           - sum_b dmin*m_b * xs[t][b] * sum(xq[t][b]) (fp32 pass, "min correction")
 * Activations are int8 with one fp32 scale per 32 columns (sb = 32).
 *
 * Header-only (static inline) so tests, the benchmark and the runtime share it.
 * Requires 512-bit SVE (A64FX). */
#ifndef GLM53F_MOE_GROUPED_NATIVE_H
#define GLM53F_MOE_GROUPED_NATIVE_H
#include <arm_sve.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void gk_gemm_tile6x4p_asm(const int8_t *w, long nblocks, long kgroups4, const int8_t *xp,
                          const float *xsp, float *y, size_t ldy_bytes);

enum {
    GMN_SB = 32,                 /* scale block (columns) */
    GMN_BLK = 64 * GMN_SB + 256, /* one panel64 block: int8 data + 64 fp32 row scales */
    GMN_KC = 512,                /* K columns expanded at a time (16 blocks, 36 KiB) */
    GMN_Q4K_BYTES = 144,
    GMN_Q5K_BYTES = 176,
    GMN_TYPE_Q4K = 0,
    GMN_TYPE_Q5K = 1
};

static inline size_t gmn_sblk_bytes(int type) { return type == GMN_TYPE_Q5K ? GMN_Q5K_BYTES : GMN_Q4K_BYTES; }
static inline size_t gmn_row_bytes(int type, int K) { return (size_t)(K / 256) * gmn_sblk_bytes(type); }

/* ---- scale / min extraction (get_scale_min_k4) on 16 rows at once -------- */
#define GMN_WORD(k) (((k) >> 2) == 0 ? s0 : ((k) >> 2) == 1 ? s1 : s2)
#define GMN_BYTE(pg, k) svand_n_u32_x(pg, svlsr_n_u32_x(pg, GMN_WORD(k), 8 * ((k) & 3)), 0xFF)

static inline void gmn_scale_min(svbool_t pg, int j, svuint32_t s0, svuint32_t s1, svuint32_t s2,
                                 svuint32_t *sc, svuint32_t *m) {
    /* j is a compile-time constant after unrolling */
    switch (j) {
#define GMN_CASE_LO(J) case J: *sc = svand_n_u32_x(pg, GMN_BYTE(pg, J), 63); *m = svand_n_u32_x(pg, GMN_BYTE(pg, (J) + 4), 63); return;
#define GMN_CASE_HI(J) case J: \
        *sc = svorr_u32_x(pg, svand_n_u32_x(pg, GMN_BYTE(pg, (J) + 4), 0xF), svlsl_n_u32_x(pg, svlsr_n_u32_x(pg, GMN_BYTE(pg, (J) - 4), 6), 4)); \
        *m = svorr_u32_x(pg, svlsr_n_u32_x(pg, GMN_BYTE(pg, (J) + 4), 4), svlsl_n_u32_x(pg, svlsr_n_u32_x(pg, GMN_BYTE(pg, J), 6), 4)); return;
    GMN_CASE_LO(0) GMN_CASE_LO(1) GMN_CASE_LO(2) GMN_CASE_LO(3)
    GMN_CASE_HI(4) GMN_CASE_HI(5) GMN_CASE_HI(6) GMN_CASE_HI(7)
#undef GMN_CASE_LO
#undef GMN_CASE_HI
    }
}

/* Expand superblock `sbi` of the 64 rows starting at `rows` (row stride `stride` bytes) into 8 panel64 blocks
 * in cb (8 * GMN_BLK bytes) and dmin*m terms into mina[8][64]. */
static inline void gmn_expand_sblk(int type, const uint8_t *rows, size_t stride, int sbi, uint8_t *cb, float *mina) {
    const svbool_t pg = svptrue_b32();
    const svuint32_t off = svmul_n_u32_x(pg, svindex_u32(0, 1), (uint32_t)stride);
    const size_t sbytes = gmn_sblk_bytes(type);
    for (int p = 0; p < 4; ++p) {
        const uint8_t *bp = rows + (size_t)(16 * p) * stride + (size_t)sbi * sbytes;
        const svuint32_t dd = svld1_gather_u32offset_u32(pg, (const uint32_t *)bp, off);
        const svuint32_t s0 = svld1_gather_u32offset_u32(pg, (const uint32_t *)(bp + 4), off);
        const svuint32_t s1 = svld1_gather_u32offset_u32(pg, (const uint32_t *)(bp + 8), off);
        const svuint32_t s2 = svld1_gather_u32offset_u32(pg, (const uint32_t *)(bp + 12), off);
        const svfloat32_t d32 = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(dd));
        const svfloat32_t dm32 = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(svlsr_n_u32_x(pg, dd, 16)));
#define GMN_SCALE(J) do { svuint32_t sc_, m_; gmn_scale_min(pg, J, s0, s1, s2, &sc_, &m_); \
            svst1_f32(pg, (float *)(cb + (size_t)(J) * GMN_BLK + 64 * GMN_SB) + 16 * p, svmul_f32_x(pg, d32, svcvt_f32_u32_x(pg, sc_))); \
            svst1_f32(pg, mina + (J) * 64 + 16 * p, svmul_f32_x(pg, dm32, svcvt_f32_u32_x(pg, m_))); } while (0)
        GMN_SCALE(0); GMN_SCALE(1); GMN_SCALE(2); GMN_SCALE(3); GMN_SCALE(4); GMN_SCALE(5); GMN_SCALE(6); GMN_SCALE(7);
#undef GMN_SCALE
        if (type == GMN_TYPE_Q4K) {
            for (int g = 0; g < 4; ++g)
                for (int jq = 0; jq < 8; ++jq) {
                    const svuint32_t w = svld1_gather_u32offset_u32(pg, (const uint32_t *)(bp + 16 + g * 32 + jq * 4), off);
                    svst1_u32(pg, (uint32_t *)(cb + (size_t)(2 * g) * GMN_BLK + jq * 256 + p * 64), svand_n_u32_x(pg, w, 0x0F0F0F0F));
                    svst1_u32(pg, (uint32_t *)(cb + (size_t)(2 * g + 1) * GMN_BLK + jq * 256 + p * 64),
                              svand_n_u32_x(pg, svlsr_n_u32_x(pg, w, 4), 0x0F0F0F0F));
                }
        } else {
            for (int jq = 0; jq < 8; ++jq) {
                const svuint32_t qh = svld1_gather_u32offset_u32(pg, (const uint32_t *)(bp + 16 + jq * 4), off);
#define GMN_Q5(G) do { \
                const svuint32_t w = svld1_gather_u32offset_u32(pg, (const uint32_t *)(bp + 48 + (G) * 32 + jq * 4), off); \
                const svuint32_t h0 = svlsl_n_u32_x(pg, svand_n_u32_x(pg, svlsr_n_u32_x(pg, qh, 2 * (G)), 0x01010101), 4); \
                const svuint32_t h1 = svlsl_n_u32_x(pg, svand_n_u32_x(pg, svlsr_n_u32_x(pg, qh, 2 * (G) + 1), 0x01010101), 4); \
                svst1_u32(pg, (uint32_t *)(cb + (size_t)(2 * (G)) * GMN_BLK + jq * 256 + p * 64), \
                          svorr_u32_x(pg, svand_n_u32_x(pg, w, 0x0F0F0F0F), h0)); \
                svst1_u32(pg, (uint32_t *)(cb + (size_t)(2 * (G) + 1) * GMN_BLK + jq * 256 + p * 64), \
                          svorr_u32_x(pg, svand_n_u32_x(pg, svlsr_n_u32_x(pg, w, 4), 0x0F0F0F0F), h1)); } while (0)
                GMN_Q5(0); GMN_Q5(1); GMN_Q5(2); GMN_Q5(3);
#undef GMN_Q5
            }
        }
    }
}

/* Activation packing: 6 token rows -> xp32[g4*6 + t], xsp[b*6 + t] (tile-kernel layout). */
static inline void gmn_pack6(int8_t *xp, float *xsp, const int8_t *const rows[6], const float *const xs[6], int K) {
    uint32_t *dst = (uint32_t *)xp;
    const uint32_t *r0 = (const uint32_t *)rows[0], *r1 = (const uint32_t *)rows[1], *r2 = (const uint32_t *)rows[2];
    const uint32_t *r3 = (const uint32_t *)rows[3], *r4 = (const uint32_t *)rows[4], *r5 = (const uint32_t *)rows[5];
    for (int g = 0; g < K / 4; ++g) {
        dst[g * 6 + 0] = r0[g]; dst[g * 6 + 1] = r1[g]; dst[g * 6 + 2] = r2[g];
        dst[g * 6 + 3] = r3[g]; dst[g * 6 + 4] = r4[g]; dst[g * 6 + 5] = r5[g];
    }
    for (int b = 0; b < K / GMN_SB; ++b)
        for (int t = 0; t < 6; ++t) xsp[b * 6 + t] = xs[t][b];
}

/* y[t][r0..] = W(rows x K, native type) . xq[t]  for mpad tokens (multiple of 6), rows multiple of 64.
 * W points at row 0; consecutive rows are `stride` bytes apart. cbuf: (GMN_KC/32)*GMN_BLK bytes (256-aligned);
 * mina: (K/32)*64 floats.  Bg[t][b] = xs[t][b]*sum(xq[t][b*32..]).  y is [mpad][ldy]. */
static inline void gmn_gemm(int type, const uint8_t *W, size_t stride, int K, int rows, int mpad,
                            const int8_t *xp, const float *xsp, const float *Bg,
                            float *y, size_t ldy, uint8_t *cbuf, float *mina, int prefetch) {
    const int nb = K / 32;
    const svbool_t pg = svptrue_b32();
    const size_t sbytes = gmn_sblk_bytes(type);
    for (int r = 0; r < rows; r += 64) {
        const uint8_t *wp = W + (size_t)r * stride;
        for (int t = 0; t < mpad; ++t) memset(y + (size_t)t * ldy + r, 0, 64 * 4);
        for (int k0 = 0; k0 < K; k0 += GMN_KC) {
            const int kn = K - k0 < GMN_KC ? K - k0 : GMN_KC, nblk = kn / 32;
            if (prefetch) {
                /* next chunk of this panel, or first chunk of the next panel */
                if (k0 + kn < K) {
                    for (int rr = 0; rr < 64; ++rr)
                        for (int s = 0; s < kn / 256; ++s) {
                            const char *a = (const char *)wp + (size_t)rr * stride + (size_t)(k0 / 256 + kn / 256 + s) * sbytes;
                            __builtin_prefetch(a, 0, 2); __builtin_prefetch(a + sbytes - 1, 0, 2);
                        }
                } else if (r + 64 < rows) {
                    for (int rr = 0; rr < 64; ++rr) {
                        const char *a = (const char *)wp + (size_t)(64 + rr) * stride;
                        for (int s = 0; s < kn / 256; ++s) { __builtin_prefetch(a + s * sbytes, 0, 2); __builtin_prefetch(a + (s + 1) * sbytes - 1, 0, 2); }
                    }
                }
            }
            for (int s = 0; s < kn / 256; ++s)
                gmn_expand_sblk(type, wp, stride, k0 / 256 + s, cbuf + (size_t)s * 8 * GMN_BLK, mina + (size_t)(k0 / 32 + s * 8) * 64);
            for (int t = 0; t < mpad; t += 6)
                gk_gemm_tile6x4p_asm((const int8_t *)cbuf, nblk, GMN_SB / 16, xp + (size_t)(t / 6) * 6 * K + (size_t)(k0 / 4) * 24,
                                     xsp + (size_t)(t / 6) * 6 * nb + (size_t)(k0 / 32) * 6, y + (size_t)t * ldy + r, ldy * 4);
            /* min correction for this chunk: 4 tokens x 4 row-vectors of accumulators */
            const int b0 = k0 / 32;
            for (int t = 0; t < mpad; t += 4) {
#define GMN_ACC(u) svfloat32_t a##u##0 = svdup_f32(0), a##u##1 = a##u##0, a##u##2 = a##u##0, a##u##3 = a##u##0
                GMN_ACC(0); GMN_ACC(1); GMN_ACC(2); GMN_ACC(3);
#undef GMN_ACC
                const float *bg0 = Bg + (size_t)t * nb, *bg1 = Bg + (size_t)(t + 1 < mpad ? t + 1 : mpad - 1) * nb;
                const float *bg2 = Bg + (size_t)(t + 2 < mpad ? t + 2 : mpad - 1) * nb, *bg3 = Bg + (size_t)(t + 3 < mpad ? t + 3 : mpad - 1) * nb;
                for (int b = b0; b < b0 + nblk; ++b) {
                    const svfloat32_t m0 = svld1_f32(pg, mina + (size_t)b * 64), m1 = svld1_f32(pg, mina + (size_t)b * 64 + 16);
                    const svfloat32_t m2 = svld1_f32(pg, mina + (size_t)b * 64 + 32), m3 = svld1_f32(pg, mina + (size_t)b * 64 + 48);
#define GMN_STEP(u) do { const float bv = bg##u[b]; \
                    a##u##0 = svmla_n_f32_x(pg, a##u##0, m0, bv); a##u##1 = svmla_n_f32_x(pg, a##u##1, m1, bv); \
                    a##u##2 = svmla_n_f32_x(pg, a##u##2, m2, bv); a##u##3 = svmla_n_f32_x(pg, a##u##3, m3, bv); } while (0)
                    GMN_STEP(0); GMN_STEP(1); GMN_STEP(2); GMN_STEP(3);
#undef GMN_STEP
                }
#define GMN_FIN(u) do { if (t + (u) < mpad) { float *yt = y + (size_t)(t + (u)) * ldy + r; \
                    svst1_f32(pg, yt, svsub_f32_x(pg, svld1_f32(pg, yt), a##u##0)); \
                    svst1_f32(pg, yt + 16, svsub_f32_x(pg, svld1_f32(pg, yt + 16), a##u##1)); \
                    svst1_f32(pg, yt + 32, svsub_f32_x(pg, svld1_f32(pg, yt + 32), a##u##2)); \
                    svst1_f32(pg, yt + 48, svsub_f32_x(pg, svld1_f32(pg, yt + 48), a##u##3)); } } while (0)
                GMN_FIN(0); GMN_FIN(1); GMN_FIN(2); GMN_FIN(3);
#undef GMN_FIN
            }
        }
    }
}

/* ---- vector math / quantization helpers -------------------------------- */
static inline svfloat32_t gmn_expf(svbool_t pg, svfloat32_t x) {
    x = svmax_n_f32_x(pg, svmin_n_f32_x(pg, x, 88.f), -87.f);
    svfloat32_t n = svrintn_f32_x(pg, svmul_n_f32_x(pg, x, 1.4426950408889634f));
    svfloat32_t r = svmls_n_f32_x(pg, x, n, 0.693145751953125f);
    r = svmls_n_f32_x(pg, r, n, 1.428606765330187e-06f);
    svfloat32_t q = svdup_n_f32(1.0f / 720.f);
    q = svmad_n_f32_x(pg, q, r, 1.0f / 120.f); q = svmad_n_f32_x(pg, q, r, 1.0f / 24.f);
    q = svmad_n_f32_x(pg, q, r, 1.0f / 6.f);   q = svmad_n_f32_x(pg, q, r, 0.5f);
    q = svmad_n_f32_x(pg, q, r, 1.0f);         q = svmad_n_f32_x(pg, q, r, 1.0f);
    return svscale_f32_x(pg, q, svcvt_s32_f32_x(pg, n));
}

/* Quantize 32 floats (two vectors) to int8 with a per-32 scale; returns scale, writes q, *bsum = scale*sum(q). */
static inline float gmn_quant32(svbool_t pg, svfloat32_t a0, svfloat32_t a1, int8_t *q, float *bsum) {
    const float mx = svmaxv_f32(pg, svmax_f32_x(pg, svabs_f32_x(pg, a0), svabs_f32_x(pg, a1)));
    const float sc = mx / 127.f, inv = sc > 0 ? 1.f / sc : 0.f;
    svint32_t q0 = svcvt_s32_f32_x(pg, svrintn_f32_x(pg, svmul_n_f32_x(pg, a0, inv)));
    svint32_t q1 = svcvt_s32_f32_x(pg, svrintn_f32_x(pg, svmul_n_f32_x(pg, a1, inv)));
    q0 = svmax_n_s32_x(pg, svmin_n_s32_x(pg, q0, 127), -127);
    q1 = svmax_n_s32_x(pg, svmin_n_s32_x(pg, q1, 127), -127);
    svst1b_s32(pg, q, q0); svst1b_s32(pg, q + 16, q1);
    *bsum = sc * (float)(svaddv_s32(pg, q0) + svaddv_s32(pg, q1));
    return sc;
}

/* x[K] float -> xq[K] int8, xs[K/32], bt[K/32] (bt = xs*sum(xq)). */
static inline void gmn_quant_row(const float *x, int K, int8_t *xq, float *xs, float *bt) {
    const svbool_t pg = svptrue_b32();
    for (int b = 0; b < K / 32; ++b)
        xs[b] = gmn_quant32(pg, svld1_f32(pg, x + b * 32), svld1_f32(pg, x + b * 32 + 16), xq + b * 32, &bt[b]);
}

/* SwiGLU (gate g, up u in ygu row: [0,inter) gate, [inter,2*inter) up, clamped like the decode path) + int8 requant.
 * ygu: [m][ldg]; a8 [mpad][inter]; as/bd [mpad][inter/32]; rows m..mpad-1 zeroed. */
static inline void gmn_swiglu_quant(const float *ygu, size_t ldg, int inter, int m, int mpad, int8_t *a8, float *as, float *bd) {
    const svbool_t pg = svptrue_b32();
    const int nb = inter / 32;
    memset(a8 + (size_t)m * inter, 0, (size_t)(mpad - m) * inter);
    memset(as + (size_t)m * nb, 0, (size_t)(mpad - m) * nb * 4);
    memset(bd + (size_t)m * nb, 0, (size_t)(mpad - m) * nb * 4);
    for (int t = 0; t < m; ++t) {
        const float *g = ygu + (size_t)t * ldg, *u = g + inter;
        for (int b = 0; b < nb; ++b) {
            svfloat32_t a0, a1;
            {
                svfloat32_t gv = svmax_n_f32_x(pg, svmin_n_f32_x(pg, svld1_f32(pg, g + b * 32), 10.f), -100.f);
                svfloat32_t uv = svmax_n_f32_x(pg, svmin_n_f32_x(pg, svld1_f32(pg, u + b * 32), 10.f), -10.f);
                a0 = svmul_f32_x(pg, svdiv_f32_x(pg, gv, svadd_n_f32_x(pg, gmn_expf(pg, svneg_f32_x(pg, gv)), 1.f)), uv);
            }
            {
                svfloat32_t gv = svmax_n_f32_x(pg, svmin_n_f32_x(pg, svld1_f32(pg, g + b * 32 + 16), 10.f), -100.f);
                svfloat32_t uv = svmax_n_f32_x(pg, svmin_n_f32_x(pg, svld1_f32(pg, u + b * 32 + 16), 10.f), -10.f);
                a1 = svmul_f32_x(pg, svdiv_f32_x(pg, gv, svadd_n_f32_x(pg, gmn_expf(pg, svneg_f32_x(pg, gv)), 1.f)), uv);
            }
            as[(size_t)t * nb + b] = gmn_quant32(pg, a0, a1, a8 + (size_t)t * inter + b * 32, &bd[(size_t)t * nb + b]);
        }
    }
}


/* ---- Q6_K down projection (sb = 16, no min term) ----------------------------------------------------------
 * value = d * sc[col/16] * (q6 - 32), q6 = 4 low bits from ql plus 2 high bits from qh (ggml layout).  Blocks of 16
 * columns use the same panel64 layout with kgroups4 = 1.  Rows are 210 B per 256 columns, i.e. only 2-byte aligned. */
enum { GMN_TYPE_Q6K = 2, GMN_Q6K_BYTES = 210, GMN_SB16 = 16, GMN_BLK16 = 64 * 16 + 256 };

static inline void gmn_expand_q6_sblk(const uint8_t *rows, size_t stride, int sbi, uint8_t *cb) {
    const svbool_t pg = svptrue_b32(), pg8 = svptrue_b8();
    const svuint32_t off = svmul_n_u32_x(pg, svindex_u32(0, 1), (uint32_t)stride);
    for (int p = 0; p < 4; ++p) {
        const uint8_t *bp = rows + (size_t)(16 * p) * stride + (size_t)sbi * GMN_Q6K_BYTES;
        const svfloat32_t d32 = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(svld1uh_gather_u32offset_u32(pg, (const uint16_t *)(bp + 208), off)));
        for (int i = 0; i < 8; ++i) {
            const svuint32_t hv = svld1uh_gather_u32offset_u32(pg, (const uint16_t *)(bp + 192 + 2 * i), off);
            const svfloat32_t s0 = svcvt_f32_s32_x(pg, svextb_s32_x(pg, svreinterpret_s32_u32(hv)));
            const svfloat32_t s1 = svcvt_f32_s32_x(pg, svextb_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, hv, 8))));
            svst1_f32(pg, (float *)(cb + (size_t)(2 * i) * GMN_BLK16 + 64 * GMN_SB16) + 16 * p, svmul_f32_x(pg, d32, s0));
            svst1_f32(pg, (float *)(cb + (size_t)(2 * i + 1) * GMN_BLK16 + 64 * GMN_SB16) + 16 * p, svmul_f32_x(pg, d32, s1));
        }
        for (int half = 0; half < 2; ++half)
            for (int m = 0; m < 8; ++m) {
                const svuint32_t w0 = svld1_gather_u32offset_u32(pg, (const uint32_t *)(bp + 64 * half + 4 * m), off);
                const svuint32_t w1 = svld1_gather_u32offset_u32(pg, (const uint32_t *)(bp + 64 * half + 32 + 4 * m), off);
                const svuint32_t wq = svld1_gather_u32offset_u32(pg, (const uint32_t *)(bp + 128 + 32 * half + 4 * m), off);
#define GMN_Q6(G, LOW) do { \
                svuint32_t lo = (G) & 1 ? w1 : w0; \
                lo = (G) >= 2 ? svlsr_n_u32_x(pg, lo, 4) : lo; \
                lo = svand_n_u32_x(pg, lo, 0x0F0F0F0F); \
                svuint32_t hi = svlsl_n_u32_x(pg, svand_n_u32_x(pg, svlsr_n_u32_x(pg, wq, 2 * (G)), 0x03030303), 4); \
                svuint32_t q = svreinterpret_u32_u8(svsub_n_u8_x(pg8, svreinterpret_u8_u32(svorr_u32_x(pg, lo, hi)), 32)); \
                const int quad = 32 * half + 8 * (G) + m; \
                svst1_u32(pg, (uint32_t *)(cb + (size_t)(quad / 4) * GMN_BLK16 + (quad % 4) * 256 + p * 64), q); } while (0)
                GMN_Q6(0, 0); GMN_Q6(1, 0); GMN_Q6(2, 0); GMN_Q6(3, 0);
#undef GMN_Q6
            }
    }
}

/* y[t][r] = sum_k W6[r][k] * xq[t][k]; xp/xsp packed with 16-column scale blocks. */
static inline void gmn_gemm_q6(const uint8_t *W, size_t stride, int K, int rows, int mpad,
                               const int8_t *xp, const float *xsp, float *y, size_t ldy, uint8_t *cbuf) {
    const int nb16 = K / 16;
    for (int r = 0; r < rows; r += 64) {
        const uint8_t *wp = W + (size_t)r * stride;
        for (int t = 0; t < mpad; ++t) memset(y + (size_t)t * ldy + r, 0, 64 * 4);
        if (r + 64 < rows)
            for (size_t o = 0; o < 64 * stride; o += 256) __builtin_prefetch((const char *)wp + 64 * stride + o, 0, 2);
        for (int sb = 0; sb < K / 256; ++sb) {
            gmn_expand_q6_sblk(wp, stride, sb, cbuf);
            for (int t = 0; t < mpad; t += 6)
                gk_gemm_tile6x4p_asm((const int8_t *)cbuf, 16, 1, xp + (size_t)(t / 6) * 6 * K + (size_t)(sb * 256 / 4) * 24,
                                     xsp + (size_t)(t / 6) * 6 * nb16 + (size_t)sb * 16 * 6, y + (size_t)t * ldy + r, ldy * 4);
        }
    }
}

/* SwiGLU + requant with 16-column scale blocks (one vector per block). */
static inline void gmn_swiglu_quant16(const float *ygu, size_t ldg, int inter, int m, int mpad, int8_t *a8, float *as) {
    const svbool_t pg = svptrue_b32();
    const int nb = inter / 16;
    memset(a8 + (size_t)m * inter, 0, (size_t)(mpad - m) * inter);
    memset(as + (size_t)m * nb, 0, (size_t)(mpad - m) * nb * 4);
    for (int t = 0; t < m; ++t) {
        const float *g = ygu + (size_t)t * ldg, *u = g + inter;
        for (int b = 0; b < nb; ++b) {
            svfloat32_t gv = svmax_n_f32_x(pg, svmin_n_f32_x(pg, svld1_f32(pg, g + b * 16), 10.f), -100.f);
            svfloat32_t uv = svmax_n_f32_x(pg, svmin_n_f32_x(pg, svld1_f32(pg, u + b * 16), 10.f), -10.f);
            svfloat32_t a = svmul_f32_x(pg, svdiv_f32_x(pg, gv, svadd_n_f32_x(pg, gmn_expf(pg, svneg_f32_x(pg, gv)), 1.f)), uv);
            const float mx = svmaxv_f32(pg, svabs_f32_x(pg, a));
            const float sc = mx / 127.f, inv = sc > 0 ? 1.f / sc : 0.f;
            svint32_t q = svcvt_s32_f32_x(pg, svrintn_f32_x(pg, svmul_n_f32_x(pg, a, inv)));
            q = svmax_n_s32_x(pg, svmin_n_s32_x(pg, q, 127), -127);
            svst1b_s32(pg, a8 + (size_t)t * inter + b * 16, q);
            as[(size_t)t * nb + b] = sc;
        }
    }
}

/* Generic 6-token packing with an sb-column scale block. */
static inline void gmn_pack6_sb(int8_t *xp, float *xsp, const int8_t *const rows[6], const float *const xs[6], int K, int sb) {
    uint32_t *dst = (uint32_t *)xp;
    for (int t = 0; t < 6; ++t) {
        const uint32_t *r = (const uint32_t *)rows[t];
        for (int g = 0; g < K / 4; ++g) dst[g * 6 + t] = r[g];
    }
    for (int b = 0; b < K / sb; ++b)
        for (int t = 0; t < 6; ++t) xsp[b * 6 + t] = xs[t][b];
}

/* ---- shared expert (FP8 e4m3 block-128 weights) as a bf16 K-major tile GEMM --------------------------------
 * e4m3 values are exact in bf16.  Weights [rows][cols] are re-tiled to wt[tile][k][32] (tile = 32 consecutive rows,
 * which never straddle a 128-row scale block).  y[t][tile*32+e] = sum_k wt * (x[t][k] * scale[rowblock][k/128]):
 * the block scale is folded into the broadcast activation, so weights are never rescaled. */
static inline uint16_t gmn_fp8_to_bf16(uint8_t q) {
    const uint32_t sign = (uint32_t)(q & 0x80) << 24, e = (q >> 3) & 15, m = q & 7;
    float f;
    if (e) { uint32_t bits = sign | ((e + 120) << 23) | (m << 20); memcpy(&f, &bits, 4); }
    else { f = (float)m * (1.0f / 512.0f); if (q & 0x80) f = -f; }
    uint32_t b; memcpy(&b, &f, 4);
    return (uint16_t)(b >> 16);
}

static inline void gmn_fp8_pack_tiles(uint16_t *dst, const uint8_t *w, int rows, int cols) {
    for (int tile = 0; tile < rows / 32; ++tile)
        for (int k = 0; k < cols; ++k)
            for (int e = 0; e < 32; ++e) dst[((size_t)tile * cols + k) * 32 + e] = gmn_fp8_to_bf16(w[(size_t)(tile * 32 + e) * cols + k]);
}

static inline __attribute__((always_inline)) void gmn_fp8g_block(float *y, size_t ldy, const uint16_t *wt,
        const float *scale_row, const float *x, size_t ldx, int K, const int n) {
    const svbool_t pg = svptrue_b32();
#define GMN_F_DECL(U) svfloat32_t f##U##0 = svdup_f32(0), f##U##1 = f##U##0
    GMN_F_DECL(0); GMN_F_DECL(1); GMN_F_DECL(2); GMN_F_DECL(3); GMN_F_DECL(4); GMN_F_DECL(5);
#undef GMN_F_DECL
    for (int kb = 0; kb * 128 < K; ++kb) {
        const float sc = scale_row[kb];
        const int k1 = (kb + 1) * 128 < K ? (kb + 1) * 128 : K;
        for (int k = kb * 128; k < k1; ++k) {
            const uint16_t *wp = wt + (size_t)k * 32;
            const svfloat32_t w0 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp), 16));
            const svfloat32_t w1 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp + 16), 16));
#define GMN_F_STEP(U) if ((U) < n) { const float xv = x[(size_t)(U) * ldx + k] * sc; \
            f##U##0 = svmla_n_f32_x(pg, f##U##0, w0, xv); f##U##1 = svmla_n_f32_x(pg, f##U##1, w1, xv); }
            GMN_F_STEP(0) GMN_F_STEP(1) GMN_F_STEP(2) GMN_F_STEP(3) GMN_F_STEP(4) GMN_F_STEP(5)
#undef GMN_F_STEP
        }
    }
#define GMN_F_ST(U) if ((U) < n) { svst1_f32(pg, y + (size_t)(U) * ldy, f##U##0); svst1_f32(pg, y + (size_t)(U) * ldy + 16, f##U##1); }
    GMN_F_ST(0) GMN_F_ST(1) GMN_F_ST(2) GMN_F_ST(3) GMN_F_ST(4) GMN_F_ST(5)
#undef GMN_F_ST
}

static inline void gmn_fp8g_run(float *y, size_t ldy, const uint16_t *wt, const float *scale_row,
        const float *x, size_t ldx, int K, int n) {
    switch (n) {
#define GMN_F_CASE(N) case N: gmn_fp8g_block(y, ldy, wt, scale_row, x, ldx, K, N); break;
    GMN_F_CASE(1) GMN_F_CASE(2) GMN_F_CASE(3) GMN_F_CASE(4) GMN_F_CASE(5) GMN_F_CASE(6)
#undef GMN_F_CASE
    }
}

/* ---- router logits GEMM ------------------------------------------------------------------------------------
 * out[t][e] = sum_k bf16 W[e][k] * x[t][k] for the 288 routed experts.  Weights are pre-tiled into 6 tiles of 48
 * experts: wt[tile][k][48] (bf16), so a tile streams contiguously.  Accumulation is sequential over k per output
 * (independent chains: 6 tokens x 3 vectors), unlike the lane-wise dot of the per-token path, so the logits can
 * differ in the last bits; near-tie top-8 flips are possible and are counted by the verify mode. */
enum { GMN_NEXP = 288, GMN_RTILE = 48, GMN_RTILES = GMN_NEXP / GMN_RTILE };

static inline void gmn_router_pack(uint16_t *wt, const uint16_t *w, int K) {
    for (int tile = 0; tile < GMN_RTILES; ++tile)
        for (int k = 0; k < K; ++k)
            for (int e = 0; e < GMN_RTILE; ++e)
                wt[((size_t)tile * K + k) * GMN_RTILE + e] = w[(size_t)(tile * GMN_RTILE + e) * K + k];
}

static inline __attribute__((always_inline)) void gmn_router_block(float *out, size_t ldo, const uint16_t *wt,
        const float *x, size_t ldx, int K, const int n) {
    const svbool_t pg = svptrue_b32();
#define GMN_R_DECL(U) svfloat32_t r##U##0 = svdup_f32(0), r##U##1 = r##U##0, r##U##2 = r##U##0
    GMN_R_DECL(0); GMN_R_DECL(1); GMN_R_DECL(2); GMN_R_DECL(3); GMN_R_DECL(4); GMN_R_DECL(5);
#undef GMN_R_DECL
    for (int k = 0; k < K; ++k) {
        const uint16_t *wp = wt + (size_t)k * GMN_RTILE;
        const svfloat32_t w0 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp), 16));
        const svfloat32_t w1 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp + 16), 16));
        const svfloat32_t w2 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp + 32), 16));
#define GMN_R_STEP(U) if ((U) < n) { const float xv = x[(size_t)(U) * ldx + k]; \
        r##U##0 = svmla_n_f32_x(pg, r##U##0, w0, xv); r##U##1 = svmla_n_f32_x(pg, r##U##1, w1, xv); \
        r##U##2 = svmla_n_f32_x(pg, r##U##2, w2, xv); }
        GMN_R_STEP(0) GMN_R_STEP(1) GMN_R_STEP(2) GMN_R_STEP(3) GMN_R_STEP(4) GMN_R_STEP(5)
#undef GMN_R_STEP
    }
#define GMN_R_ST(U) if ((U) < n) { float *o = out + (size_t)(U) * ldo; \
    svst1_f32(pg, o, r##U##0); svst1_f32(pg, o + 16, r##U##1); svst1_f32(pg, o + 32, r##U##2); }
    GMN_R_ST(0) GMN_R_ST(1) GMN_R_ST(2) GMN_R_ST(3) GMN_R_ST(4) GMN_R_ST(5)
#undef GMN_R_ST
}

/* n tokens (1..6) of tile `tile`: out points at out[t0][tile*48]. */
static inline void gmn_router_run(float *out, size_t ldo, const uint16_t *wt_layer, int tile,
        const float *x, size_t ldx, int K, int n) {
    const uint16_t *wt = wt_layer + (size_t)tile * K * GMN_RTILE;
    switch (n) {
#define GMN_R_CASE(N) case N: gmn_router_block(out, ldo, wt, x, ldx, K, N); break;
    GMN_R_CASE(1) GMN_R_CASE(2) GMN_R_CASE(3) GMN_R_CASE(4) GMN_R_CASE(5) GMN_R_CASE(6)
#undef GMN_R_CASE
    }
}

/* Twelve tokens x16 experts reuse packed weights without the36 accumulators
 * of a12x48 tile. Each output retains the original sequential key FMAs. */
static inline __attribute__((always_inline)) void gmn_router_block12(float *out,
        size_t ldo, const uint16_t *wt, const float *x, size_t ldx, int K, const int n) {
    const svbool_t pg = svptrue_b32();
#define GR12_DECL(U) svfloat32_t r##U = svdup_f32(0)
    GR12_DECL(0); GR12_DECL(1); GR12_DECL(2); GR12_DECL(3);
    GR12_DECL(4); GR12_DECL(5); GR12_DECL(6); GR12_DECL(7);
    GR12_DECL(8); GR12_DECL(9); GR12_DECL(10); GR12_DECL(11);
#undef GR12_DECL
    for (int k = 0; k < K; ++k) {
        const svfloat32_t w = svreinterpret_f32_u32(svlsl_n_u32_x(pg,
            svld1uh_u32(pg, wt + (size_t)k * GMN_RTILE), 16));
#define GR12_STEP(U) if ((U) < n) r##U = svmla_n_f32_x(pg, r##U, w, x[(size_t)(U) * ldx + k])
        GR12_STEP(0); GR12_STEP(1); GR12_STEP(2); GR12_STEP(3);
        GR12_STEP(4); GR12_STEP(5); GR12_STEP(6); GR12_STEP(7);
        GR12_STEP(8); GR12_STEP(9); GR12_STEP(10); GR12_STEP(11);
#undef GR12_STEP
    }
#define GR12_STORE(U) if ((U) < n) svst1_f32(pg, out + (size_t)(U) * ldo, r##U)
    GR12_STORE(0); GR12_STORE(1); GR12_STORE(2); GR12_STORE(3);
    GR12_STORE(4); GR12_STORE(5); GR12_STORE(6); GR12_STORE(7);
    GR12_STORE(8); GR12_STORE(9); GR12_STORE(10); GR12_STORE(11);
#undef GR12_STORE
}
static inline void gmn_router_run12(float *out, size_t ldo,
        const uint16_t *wt_layer, int tile, const float *x, size_t ldx, int K, int n) {
    const uint16_t *wt = wt_layer + (size_t)(tile / 3) * K * GMN_RTILE + (tile % 3) * 16;
    switch (n) {
#define GR12_CASE(N) case N: gmn_router_block12(out, ldo, wt, x, ldx, K, N); break
        GR12_CASE(1); GR12_CASE(2); GR12_CASE(3); GR12_CASE(4);
        GR12_CASE(5); GR12_CASE(6); GR12_CASE(7); GR12_CASE(8);
        GR12_CASE(9); GR12_CASE(10); GR12_CASE(11); GR12_CASE(12);
#undef GR12_CASE
    }
}
/* Eight tokens x32 experts. Packed48-column tiles can straddle the two
 * halves, so compute their packed pointers separately; do not repack weights. */
static inline __attribute__((always_inline)) void gmn_router_block8(float *out,
        size_t ldo, const uint16_t *w0, const uint16_t *w1, const float *x,
        size_t ldx, int K, const int n) {
    const svbool_t pg = svptrue_b32();
#define GR8_DECL(U) svfloat32_t a##U = svdup_f32(0), b##U = a##U
    GR8_DECL(0); GR8_DECL(1); GR8_DECL(2); GR8_DECL(3);
    GR8_DECL(4); GR8_DECL(5); GR8_DECL(6); GR8_DECL(7);
#undef GR8_DECL
    for (int k = 0; k < K; ++k) {
        const svfloat32_t v0 = svreinterpret_f32_u32(svlsl_n_u32_x(pg,
            svld1uh_u32(pg, w0 + (size_t)k * GMN_RTILE), 16));
        const svfloat32_t v1 = svreinterpret_f32_u32(svlsl_n_u32_x(pg,
            svld1uh_u32(pg, w1 + (size_t)k * GMN_RTILE), 16));
#define GR8_STEP(U) if ((U) < n) { const float v = x[(size_t)(U) * ldx + k]; \
        a##U = svmla_n_f32_x(pg, a##U, v0, v); b##U = svmla_n_f32_x(pg, b##U, v1, v); }
        GR8_STEP(0); GR8_STEP(1); GR8_STEP(2); GR8_STEP(3);
        GR8_STEP(4); GR8_STEP(5); GR8_STEP(6); GR8_STEP(7);
#undef GR8_STEP
    }
#define GR8_STORE(U) if ((U) < n) { svst1_f32(pg, out + (size_t)(U) * ldo, a##U); \
        svst1_f32(pg, out + (size_t)(U) * ldo + 16, b##U); }
    GR8_STORE(0); GR8_STORE(1); GR8_STORE(2); GR8_STORE(3);
    GR8_STORE(4); GR8_STORE(5); GR8_STORE(6); GR8_STORE(7);
#undef GR8_STORE
}
static inline void gmn_router_run8(float *out, size_t ldo,
        const uint16_t *wt, int tile, const float *x, size_t ldx, int K, int n) {
    const int r0 = tile * 32, r1 = r0 + 16;
    const uint16_t *w0 = wt + (size_t)(r0 / GMN_RTILE) * K * GMN_RTILE + r0 % GMN_RTILE;
    const uint16_t *w1 = wt + (size_t)(r1 / GMN_RTILE) * K * GMN_RTILE + r1 % GMN_RTILE;
    switch (n) {
#define GR8_CASE(N) case N: gmn_router_block8(out, ldo, w0, w1, x, ldx, K, N); break
        GR8_CASE(1); GR8_CASE(2); GR8_CASE(3); GR8_CASE(4);
        GR8_CASE(5); GR8_CASE(6); GR8_CASE(7); GR8_CASE(8);
#undef GR8_CASE
    }
}
static inline __attribute__((always_inline)) void gmn_router_block_unroll1(float *out, size_t ldo, const uint16_t *wt,
        const float *x, size_t ldx, int K, const int n) {
    const svbool_t pg = svptrue_b32();
#define GMN_R_DECL(U) svfloat32_t r##U##0 = svdup_f32(0), r##U##1 = r##U##0, r##U##2 = r##U##0
    GMN_R_DECL(0); GMN_R_DECL(1); GMN_R_DECL(2); GMN_R_DECL(3); GMN_R_DECL(4); GMN_R_DECL(5);
#undef GMN_R_DECL
    /* FCC key-loop unrolling otherwise extends broadcasts across keys. */
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int k = 0; k < K; ++k) {
        const uint16_t *wp = wt + (size_t)k * GMN_RTILE;
        const svfloat32_t w0 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp), 16));
        const svfloat32_t w1 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp + 16), 16));
        const svfloat32_t w2 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, wp + 32), 16));
#define GMN_R_STEP(U) if ((U) < n) { const float xv = x[(size_t)(U) * ldx + k]; \
        r##U##0 = svmla_n_f32_x(pg, r##U##0, w0, xv); r##U##1 = svmla_n_f32_x(pg, r##U##1, w1, xv); \
        r##U##2 = svmla_n_f32_x(pg, r##U##2, w2, xv); }
        GMN_R_STEP(0) GMN_R_STEP(1) GMN_R_STEP(2) GMN_R_STEP(3) GMN_R_STEP(4) GMN_R_STEP(5)
#undef GMN_R_STEP
    }
#define GMN_R_ST(U) if ((U) < n) { float *o = out + (size_t)(U) * ldo; \
    svst1_f32(pg, o, r##U##0); svst1_f32(pg, o + 16, r##U##1); svst1_f32(pg, o + 32, r##U##2); }
    GMN_R_ST(0) GMN_R_ST(1) GMN_R_ST(2) GMN_R_ST(3) GMN_R_ST(4) GMN_R_ST(5)
#undef GMN_R_ST
}

/* n tokens (1..6) of tile `tile`: out points at out[t0][tile*48]. */
static inline void gmn_router_run_unroll1(float *out, size_t ldo, const uint16_t *wt_layer, int tile,
        const float *x, size_t ldx, int K, int n) {
    const uint16_t *wt = wt_layer + (size_t)tile * K * GMN_RTILE;
    switch (n) {
#define GMN_R_CASE(N) case N: gmn_router_block_unroll1(out, ldo, wt, x, ldx, K, N); break;
    GMN_R_CASE(1) GMN_R_CASE(2) GMN_R_CASE(3) GMN_R_CASE(4) GMN_R_CASE(5) GMN_R_CASE(6)
#undef GMN_R_CASE
    }
}


/* Collective OpenMP entry, also used by the strong native fixture. */
static inline void gmn_router_prefill(float *out, const uint16_t *wt,
        const float *x, int tokens, int K, int tile_mode) {
    const int width = tile_mode == 2 ? 8 : tile_mode == 1 ? 12 : 6;
    const int chunk = tile_mode == 2 ? 3 : tile_mode == 1 ? 2 : 4;
    const int groups = (tokens + width - 1) / width, chunks = (groups + chunk - 1) / chunk;
    const int tiles = tile_mode == 2 ? GMN_NEXP / 32 : tile_mode == 1 ? GMN_NEXP / 16 : GMN_RTILES;
#pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0; task < tiles * chunks; ++task) {
        const int tile = task / chunks, ch = task % chunks;
        for (int g = ch * chunk; g < groups && g < (ch + 1) * chunk; ++g) {
            const int t = g * width, n = tokens - t < width ? tokens - t : width;
            if (tile_mode == 2) gmn_router_run8(out + (size_t)t * GMN_NEXP + tile * 32,
                GMN_NEXP, wt, tile, x + (size_t)t * K, K, K, n);
            else if (tile_mode == 3) gmn_router_run_unroll1(out + (size_t)t * GMN_NEXP + tile * GMN_RTILE,
                GMN_NEXP, wt, tile, x + (size_t)t * K, K, K, n);
            else if (tile_mode == 1) gmn_router_run12(out + (size_t)t * GMN_NEXP + tile * 16,
                GMN_NEXP, wt, tile, x + (size_t)t * K, K, K, n);
            else gmn_router_run(out + (size_t)t * GMN_NEXP + tile * GMN_RTILE,
                GMN_NEXP, wt, tile, x + (size_t)t * K, K, K, n);
        }
    }
}

#ifdef __cplusplus
}
#endif
#endif
