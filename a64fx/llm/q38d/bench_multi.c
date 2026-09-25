/* Single-core microbenchmark: FP4 A16 group kernel for N activation
 * vectors (speculative-decoding verification) vs the N=1 kernels.
 *
 *   bench_multi [groups] [reps]
 *
 * groups*8 rows x 5120 columns of random F4 weights; the working set decides
 * whether the run is L2-resident (e.g. 16 groups) or streams from HBM. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "q38d_kern.h"

static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0,cntvct_el0" : "=r"(v)); return v; }
static double tick_hz(void) { uint64_t v; __asm__ volatile("mrs %0,cntfrq_el0" : "=r"(v)); return (double)v; }

/* N-token F4 A16 group: out[t][8] for t < N. Two pairs per iteration; each
 * token's hi/lo digit dots run as two independent SDOT chains per pair so
 * the dependency depth per pair is 2 SDOTs + combine. */
#define MT_DOT2(t, A, P, L0, H0, L1, H1, WS)                                                 \
    do {                                                                                     \
        const int8_t *aq_ = a[t].q + (P) * 64, *ah_ = aq_ + 32;                              \
        svint32_t x_ = svdot_s32(svdup_n_s32(0), L0, q38d_rep8(ah_));                        \
        svint32_t y_ = svdot_s32(svdup_n_s32(0), H0, q38d_rep8(ah_ + 8));                    \
        svint32_t u_ = svdot_s32(svdup_n_s32(0), L0, q38d_rep8(aq_));                        \
        svint32_t v_ = svdot_s32(svdup_n_s32(0), H0, q38d_rep8(aq_ + 8));                    \
        x_ = svdot_s32(x_, L1, q38d_rep8(ah_ + 16));                                         \
        y_ = svdot_s32(y_, H1, q38d_rep8(ah_ + 24));                                         \
        u_ = svdot_s32(u_, L1, q38d_rep8(aq_ + 16));                                         \
        v_ = svdot_s32(v_, H1, q38d_rep8(aq_ + 24));                                         \
        svint32_t hi_ = svadd_s32_x(pf, x_, y_), lo_ = svadd_s32_x(pf, u_, v_);              \
        svint32_t d_ = svadd_s32_x(pf, svlsl_n_s32_x(pf, hi_, 8), lo_);                      \
        A = svmla_f32_x(pf, A, svcvt_f32_s32_x(pf, d_), svmul_f32_x(pf, WS, q38d_rep2f(a[t].sc + 2 * (P)))); \
    } while (0)
#define MT_DEC(P, L0, H0, L1, H1, WS)                                                        \
    svuint8_t z0##P = svld1_u8(pb, g + (P) * 128), z1##P = svld1_u8(pb, g + (P) * 128 + 64); \
    svint8_t L0 = svtbl_s8(lut, svand_n_u8_x(pb, z0##P, 15));                               \
    svint8_t H0 = svtbl_s8(lut, svlsr_n_u8_x(pb, z0##P, 4));                                \
    svint8_t L1 = svtbl_s8(lut, svand_n_u8_x(pb, z1##P, 15));                               \
    svint8_t H1 = svtbl_s8(lut, svlsr_n_u8_x(pb, z1##P, 4));                                \
    svfloat32_t WS = svreinterpret_f32_u32(svlsl_n_u32_x(pf, svld1ub_u32(pf, sc + (P) * 16), 20))
#define MT_OUT(t, A, B) do { svfloat32_t s_ = svadd_f32_x(pf, A, B);                        \
    svst1_f32(svptrue_pat_b32(SV_VL8), out + 8 * (t),                                        \
              svmul_n_f32_x(pf, svadd_f32_x(pf, svuzp1_f32(s_, s_), svuzp2_f32(s_, s_)), 0x1p45f)); } while (0)
#define MT_KERNEL(N)                                                                         \
static void group_f4_a16_n##N(float *out, const uint8_t *g, const q38d_act *a) {            \
    const svbool_t pf = svptrue_b32(), pb = svptrue_b8();                                    \
    const size_t np = (size_t)a[0].cols / 32;                                                \
    const svint8_t lut = svld1_s8(pb, q38d_lut_f4);                                          \
    const uint8_t *sc = g + np * 128;                                                        \
    svfloat32_t a0 = svdup_n_f32(0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0; \
    for (size_t p = 0; p < np; p += 2) {                                                     \
        MT_DEC(p, l0, h0, l1, h1, ws);                                                       \
        size_t q = p + 1;                                                                    \
        MT_DEC(q, m0, k0, m1, k1, wt);                                                       \
        MT_DOT2(0, a0, p, l0, h0, l1, h1, ws); MT_DOT2(0, b0, q, m0, k0, m1, k1, wt);        \
        if (N > 1) { MT_DOT2(1, a1, p, l0, h0, l1, h1, ws); MT_DOT2(1, b1, q, m0, k0, m1, k1, wt); } \
        if (N > 2) { MT_DOT2(2, a2, p, l0, h0, l1, h1, ws); MT_DOT2(2, b2, q, m0, k0, m1, k1, wt); } \
        if (N > 3) { MT_DOT2(3, a3, p, l0, h0, l1, h1, ws); MT_DOT2(3, b3, q, m0, k0, m1, k1, wt); } \
    }                                                                                        \
    MT_OUT(0, a0, b0); if (N > 1) MT_OUT(1, a1, b1); if (N > 2) MT_OUT(2, a2, b2); if (N > 3) MT_OUT(3, a3, b3); \
}
MT_KERNEL(1)
MT_KERNEL(2)
MT_KERNEL(3)
MT_KERNEL(4)

/* R16 layout: 16-row groups, tile t = 16 columns (one NVFP4 unit) for all
 * 16 rows. Code vector h (64 B): lane r bytes b=0..3, low nibble column
 * 4h+b, high nibble column 8+4h+b. Weight vectors W[e], e = 2h+n, are
 * multiplied by activation element e (4 bytes, columns 8n+4h..+3) of the
 * 16-byte digit record replicated to every 128-bit segment (LD1RQ).
 * Scales: 16 bytes per tile (row r in lane r, E5M3).
 * Activation per token and tile: lo digits [16], hi digits [16] (element
 * order e), scale float. */
typedef struct { const int8_t *q; const float *sc; } r16_act;
#define R16_TOK(t, A)                                                                        \
    do {                                                                                     \
        const int8_t *aq_ = a[t].q + (size_t)T * 32;                                         \
        svint8_t al_ = svld1rq_s8(pb, aq_), ah_ = svld1rq_s8(pb, aq_ + 16);                  \
        svint32_t x_ = svdot_lane_s32(svdup_n_s32(0), W0, ah_, 0);                           \
        svint32_t y_ = svdot_lane_s32(svdup_n_s32(0), W1, ah_, 1);                           \
        svint32_t u_ = svdot_lane_s32(svdup_n_s32(0), W0, al_, 0);                           \
        svint32_t v_ = svdot_lane_s32(svdup_n_s32(0), W1, al_, 1);                           \
        x_ = svdot_lane_s32(x_, W2, ah_, 2); y_ = svdot_lane_s32(y_, W3, ah_, 3);             \
        u_ = svdot_lane_s32(u_, W2, al_, 2); v_ = svdot_lane_s32(v_, W3, al_, 3);             \
        svint32_t d_ = svadd_s32_x(pf, svlsl_n_s32_x(pf, svadd_s32_x(pf, x_, y_), 8), svadd_s32_x(pf, u_, v_)); \
        A = svmla_f32_x(pf, A, svcvt_f32_s32_x(pf, d_), svmul_n_f32_x(pf, WS, a[t].sc[T]));  \
    } while (0)
#define R16_KERNEL(N)                                                                        \
static void r16_f4_a16_n##N(float *out, const uint8_t *g, const r16_act *a, int ntile) {    \
    const svbool_t pf = svptrue_b32(), pb = svptrue_b8();                                    \
    const svint8_t lut = svld1_s8(pb, q38d_lut_f4);                                          \
    const uint8_t *sc = g + (size_t)ntile * 128;                                             \
    svfloat32_t a0 = svdup_n_f32(0), a1 = a0, a2 = a0, a3 = a0;                              \
    for (int T = 0; T < ntile; T++) {                                                        \
        svuint8_t z0 = svld1_u8(pb, g + (size_t)T * 128), z1 = svld1_u8(pb, g + (size_t)T * 128 + 64); \
        svint8_t W0 = svtbl_s8(lut, svand_n_u8_x(pb, z0, 15));                               \
        svint8_t W1 = svtbl_s8(lut, svlsr_n_u8_x(pb, z0, 4));                                \
        svint8_t W2 = svtbl_s8(lut, svand_n_u8_x(pb, z1, 15));                               \
        svint8_t W3 = svtbl_s8(lut, svlsr_n_u8_x(pb, z1, 4));                                \
        svfloat32_t WS = svreinterpret_f32_u32(svlsl_n_u32_x(pf, svld1ub_u32(pf, sc + (size_t)T * 16), 20)); \
        R16_TOK(0, a0); if (N > 1) R16_TOK(1, a1); if (N > 2) R16_TOK(2, a2); if (N > 3) R16_TOK(3, a3); \
    }                                                                                        \
    svst1_f32(pf, out, a0); if (N > 1) svst1_f32(pf, out + 16, a1);                          \
    if (N > 2) svst1_f32(pf, out + 32, a2); if (N > 3) svst1_f32(pf, out + 48, a3);          \
}
R16_KERNEL(1)
R16_KERNEL(2)
R16_KERNEL(4)

void q38d_asmn4_f4_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                       const float *, long, float *, const int8_t *);
/* interleave four per-token activations: q[np][4][64], sc[np][4][2] */
static void interleave4(int8_t *q, float *sc, const q38d_act *a) {
    int np = a[0].cols / 32;
    for (int p = 0; p < np; p++)
        for (int t = 0; t < 4; t++) {
            memcpy(q + ((size_t)p * 4 + t) * 64, a[t].q + (size_t)p * 64, 64);
            memcpy(sc + ((size_t)p * 4 + t) * 2, a[t].sc + (size_t)p * 2, 8);
        }
}
void q38d_asmnb4_f4_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                        const float *, long, float *, const int8_t *);
void q38d_asmn4c_f4_a16(const uint8_t *, const uint8_t *, const uint8_t *, const int8_t *,
                        const float *, long, float *, const int8_t *);
static int n4_variant = 0;
static void group_n4_asm(float *out, const uint8_t *g, const int8_t *q, const float *sc, int np) {
    float acc[64] __attribute__((aligned(256)));
    (n4_variant == 2 ? q38d_asmn4c_f4_a16 : n4_variant ? q38d_asmnb4_f4_a16 : q38d_asmn4_f4_a16)(g, NULL, g + (size_t)np * 128, q, sc, np, acc, q38d_lut_f4);
    for (int t = 0; t < 4; t++)
        for (int r = 0; r < 8; r++) out[8 * t + r] = (acc[16 * t + 2 * r] + acc[16 * t + 2 * r + 1]) * 0x1p45f;
}

#define G2DECL(n) void q38d_asmg2n##n##_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *, \
                                             const float *, long, float *, const int8_t *);
G2DECL(1) G2DECL(2) G2DECL(3)
#define G2CDECL(n) void q38d_asmg2c##n##_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *, \
                                              const float *, long, float *, const int8_t *);
G2CDECL(1) G2CDECL(2) G2CDECL(4)
#define G2SDECL(n) void q38d_asmg2s##n##_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *, \
                                              const float *, long, float *, const int8_t *);
G2SDECL(2) G2SDECL(4)
static int g2_kind = 0;
/* interleave nt per-token activations: q[np][nt][64], sc[np][nt][2] */
static void interleave_n(int8_t *q, float *sc, const q38d_act *a, int nt) {
    int np = a[0].cols / 32;
    for (int p = 0; p < np; p++)
        for (int t = 0; t < nt; t++) {
            memcpy(q + ((size_t)p * nt + t) * 64, a[t].q + (size_t)p * 64, 64);
            memcpy(sc + ((size_t)p * nt + t) * 2, a[t].sc + (size_t)p * 2, 8);
        }
}
/* out[g][t][8] for two groups at g and g + gb */
static void g2_asm(int nt, float *out, const uint8_t *g, size_t gb, const int8_t *q, const float *sc, int np) {
    float acc[2 * 4 * 16] __attribute__((aligned(256)));
    void (*f)(const uint8_t *, long, const uint8_t *, const int8_t *, const float *, long, float *, const int8_t *) =
        g2_kind == 2 ? (nt == 2 ? q38d_asmg2s2_f4_a16 : q38d_asmg2s4_f4_a16) :
        g2_kind ? (nt == 1 ? q38d_asmg2c1_f4_a16 : nt == 2 ? q38d_asmg2c2_f4_a16 : q38d_asmg2c4_f4_a16)
                : (nt == 1 ? q38d_asmg2n1_f4_a16 : nt == 2 ? q38d_asmg2n2_f4_a16 : q38d_asmg2n3_f4_a16);
    f(g, (long)gb, g + (size_t)np * 128, q, sc, np, acc, q38d_lut_f4);
    for (int k = 0; k < 2 * nt; k++)
        for (int r = 0; r < 8; r++) out[8 * k + r] = (acc[16 * k + 2 * r] + acc[16 * k + 2 * r + 1]) * 0x1p45f;
}

#define G2PDECL(n) void q38d_asmg2p##n##_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, \
                                              const float *const *, long, float *, const int8_t *);
G2PDECL(2) G2PDECL(3) G2PDECL(4)
#define G2IDECL(n) void q38d_asmg2i##n##_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, \
                                              const float *const *, long, float *, const int8_t *);
G2IDECL(3) G2IDECL(4)
#define G2JDECL(n) void q38d_asmg2j##n##_f4_a16(const uint8_t *, long, const uint8_t *, const int8_t *const *, \
                                              const float *const *, long, float *, const int8_t *);
G2JDECL(3) G2JDECL(4)

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2 - 1; }

int main(int argc, char **argv) {
    int groups = argc > 1 ? atoi(argv[1]) : 16, reps = argc > 2 ? atoi(argv[2]) : 200;
    const int cols = 5120;
    size_t gb = q38d_group_bytes(Q38D_F4, cols);
    uint8_t *w = aligned_alloc(256, gb * groups);
    for (size_t i = 0; i < gb * groups; i++) w[i] = (uint8_t)rand();
    for (int g = 0; g < groups; g++)   /* sane scales: E5M3 exponents near bias */
        for (size_t i = 0; i < (size_t)cols / 32 * 16; i++) w[g * gb + (size_t)cols / 32 * 128 + i] = (uint8_t)(150 + rand() % 16);
    q38d_act a[4];
    float *x = malloc(sizeof(float) * cols);
    for (int t = 0; t < 4; t++) {
        for (int i = 0; i < cols; i++) x[i] = frand();
        a[t].cols = cols; a[t].arith = Q38D_A16;
        a[t].q = aligned_alloc(256, q38d_act_qbytes(cols, Q38D_A16));
        a[t].sc = aligned_alloc(256, cols / 16 * 4 + 256);
        a[t].sum = aligned_alloc(256, cols / 16 * 4 + 256);
        q38d_prepare_ref(&a[t], x);
    }
    float *out = aligned_alloc(256, sizeof(float) * 64 * groups), *ref = malloc(sizeof(float) * 8);
    /* correctness of the N-token kernel against the reference group kernel */
    double maxrel = 0;
    for (int g = 0; g < groups && g < 4; g++) {
        group_f4_a16_n4(out, w + g * gb, a);
        for (int t = 0; t < 4; t++) {
            q38d_group_sve(ref, w + g * gb, Q38D_F4, Q38D_A16, &a[t], 0, 8);
            for (int r = 0; r < 8; r++) {
                double e = fabs(out[8 * t + r] - ref[r]) / (fabs(ref[r]) + 1e-30);
                if (e > maxrel) maxrel = e;
            }
        }
    }
    printf("check: max rel diff N=4 vs group_sve = %.3g\n", maxrel);
    int np = cols / 32;
    int8_t *qi = aligned_alloc(256, (size_t)np * 256);
    float *sci = aligned_alloc(256, (size_t)np * 32);
    interleave4(qi, sci, a);
    for (n4_variant = 0; n4_variant < 3; n4_variant++) {
    maxrel = 0;
    for (int g = 0; g < groups && g < 4; g++) {
        group_n4_asm(out, w + g * gb, qi, sci, np);
        for (int t = 0; t < 4; t++) {
            q38d_group_sve(ref, w + g * gb, Q38D_F4, Q38D_A16, &a[t], 0, 8);
            for (int r = 0; r < 8; r++) {
                double e = fabs(out[8 * t + r] - ref[r]) / (fabs(ref[r]) + 1e-30);
                if (e > maxrel) maxrel = e;
            }
        }
    }
    printf("check: max rel diff asm N=4 vs group_sve = %.3g\n", maxrel);
    {
        uint64_t t0 = ticks();
        for (int r = 0; r < reps; r++)
            for (int g = 0; g < groups; g++) group_n4_asm(out + 32 * g, w + g * gb, qi, sci, np);
        __asm__ volatile("" ::: "memory");
        double s = (double)(ticks() - t0) / tick_hz(), pr = (double)groups * np * reps, cyc = s * 2.0e9 / pr;
        printf("asmN4%c N=4: %.2f cycles/pair, %.2f cycles/pair/token, %.1f GB/s weights\n", "abc"[n4_variant], cyc, cyc / 4, pr * 144 / s / 1e9);
    }
    }
    double hz = tick_hz(), pairs = (double)groups * (cols / 32) * reps;
    for (int N = 0; N <= 4; N++) {
        uint64_t t0 = ticks();
        for (int r = 0; r < reps; r++)
            for (int g = 0; g < groups; g++) {
                const uint8_t *gp = w + g * gb;
                switch (N) {
                case 0: q38d_group_asm(out + 8 * g, gp, Q38D_F4, Q38D_A16, &a[0], 0, 8); break;
                case 1: group_f4_a16_n1(out + 32 * g, gp, a); break;
                case 2: group_f4_a16_n2(out + 32 * g, gp, a); break;
                case 3: group_f4_a16_n3(out + 32 * g, gp, a); break;
                case 4: group_f4_a16_n4(out + 32 * g, gp, a); break;
                }
            }
        __asm__ volatile("" ::: "memory");
        double s = (double)(ticks() - t0) / hz, cyc = s * 2.0e9 / pairs;
        double cks = 0;
        for (int i = 0; i < 32 * groups; i++) cks += out[i];
        printf("[%.3g] %s N=%d: %.2f cycles/pair, %.2f cycles/pair/token, %.1f GB/s weights\n",
               cks, N ? "intrin" : "asm   ", N ? N : 1, cyc, cyc / (N ? N : 1), pairs * 144 / s / 1e9);
        (void)cks;
    }
    for (g2_kind = 0; g2_kind < 3; g2_kind++)
    for (int nt = 1; nt <= (g2_kind ? 4 : 3); nt++) {
        if (g2_kind && nt == 3) continue;
        if (g2_kind == 2 && nt == 1) continue;
        int np = cols / 32;
        int8_t *qn = aligned_alloc(256, (size_t)np * 64 * nt);
        float *scn = aligned_alloc(256, (size_t)np * 8 * nt);
        interleave_n(qn, scn, a, nt);
        double mr = 0;
        for (int g = 0; g + 1 < groups && g < 4; g += 2) {
            g2_asm(nt, out, w + g * gb, gb, qn, scn, np);
            for (int gg = 0; gg < 2; gg++)
                for (int t = 0; t < nt; t++) {
                    q38d_group_sve(ref, w + (g + gg) * gb, Q38D_F4, Q38D_A16, &a[t], 0, 8);
                    for (int r = 0; r < 8; r++) {
                        double e = fabs(out[8 * (gg * nt + t) + r] - ref[r]) / (fabs(ref[r]) + 1e-30);
                        if (e > mr) mr = e;
                    }
                }
        }
        uint64_t t0 = ticks();
        for (int r = 0; r < reps; r++)
            for (int g = 0; g + 1 < groups; g += 2) g2_asm(nt, out + 32 * g, w + g * gb, gb, qn, scn, np);
        __asm__ volatile("" ::: "memory");
        double s = (double)(ticks() - t0) / hz, pr = (double)(groups / 2 * 2) * np * reps, cyc = s * 2.0e9 / pr;
        printf("g2%s   N=%d: %.2f cycles/pair, %.2f cycles/pair/token, %.1f GB/s weights (check %.3g)\n", g2_kind == 2 ? "s" : g2_kind ? "c" : "n", nt, cyc, cyc / nt, pr * 144 / s / 1e9, mr);
        (void)0;
    }
    for (int vi = 0; vi < 7; vi++) {
        int nt = vi < 3 ? vi + 2 : vi < 5 ? vi : vi - 2;   /* g2p 2,3,4; g2i 3,4; g2j 3,4 */
        int np = cols / 32;
        const int8_t *aqp[4] = {a[0].q, a[1].q, a[2].q, a[3].q};
        const float *asp[4] = {a[0].sc, a[1].sc, a[2].sc, a[3].sc};
        void (*f)(const uint8_t *, long, const uint8_t *, const int8_t *const *, const float *const *, long, float *, const int8_t *) =
            vi == 5 ? q38d_asmg2j3_f4_a16 : vi == 6 ? q38d_asmg2j4_f4_a16 :
            vi == 3 ? q38d_asmg2i3_f4_a16 : vi == 4 ? q38d_asmg2i4_f4_a16 :
            nt == 2 ? q38d_asmg2p2_f4_a16 : nt == 3 ? q38d_asmg2p3_f4_a16 : q38d_asmg2p4_f4_a16;
        float acc[2 * 4 * 16] __attribute__((aligned(256)));
        double mr = 0;
        for (int g = 0; g + 1 < groups && g < 4; g += 2) {
            f(w + g * gb, (long)gb, w + g * gb + (size_t)np * 128, aqp, asp, np, acc, q38d_lut_f4);
            for (int gg = 0; gg < 2; gg++)
                for (int t = 0; t < nt; t++) {
                    q38d_group_sve(ref, w + (g + gg) * gb, Q38D_F4, Q38D_A16, &a[t], 0, 8);
                    for (int r = 0; r < 8; r++) {
                        float v = (acc[16 * (gg * nt + t) + 2 * r] + acc[16 * (gg * nt + t) + 2 * r + 1]) * 0x1p45f;
                        double e = fabs(v - ref[r]) / (fabs(ref[r]) + 1e-30);
                        if (e > mr) mr = e;
                    }
                }
        }
        uint64_t t0 = ticks();
        for (int r = 0; r < reps; r++)
            for (int g = 0; g + 1 < groups; g += 2) f(w + g * gb, (long)gb, w + g * gb + (size_t)np * 128, aqp, asp, np, out + 32 * g, q38d_lut_f4);
        __asm__ volatile("" ::: "memory");
        double s = (double)(ticks() - t0) / hz, pr = (double)(groups / 2 * 2) * np * reps, cyc = s * 2.0e9 / pr;
        printf("%s   N=%d: %.2f cycles/pair, %.2f cycles/pair/token (check %.3g)\n", vi < 3 ? "g2p" : vi < 5 ? "g2i" : "g2j", nt, cyc, cyc / nt, mr);
    }
    /* R16: same bytes, 16-row groups (groups/2 of them), 320 tiles each */
    {
        int ntile = cols / 16, ng = groups / 2;
        r16_act ra[4];
        for (int t = 0; t < 4; t++) {
            int8_t *q = aligned_alloc(256, (size_t)ntile * 32);
            float *sc = aligned_alloc(256, (size_t)ntile * 4);
            for (int i = 0; i < ntile * 32; i++) q[i] = (int8_t)(rand() % 255 - 127);
            for (int i = 0; i < ntile; i++) sc[i] = 0x1p64f * 1e-3f * (1 + i % 7);
            ra[t].q = q; ra[t].sc = sc;
        }
        size_t g16 = (size_t)ntile * 144;
        for (int g = 0; g < ng; g++)
            for (int i = 0; i < ntile * 16; i++) w[g * g16 + (size_t)ntile * 128 + i] = (uint8_t)(150 + rand() % 16);
        for (int N = 1; N <= 4; N *= 2) {
            uint64_t t0 = ticks();
            for (int r = 0; r < reps; r++)
                for (int g = 0; g < ng; g++) {
                    const uint8_t *gp = w + g * g16;
                    if (N == 1) r16_f4_a16_n1(out + 64 * g, gp, ra, ntile);
                    else if (N == 2) r16_f4_a16_n2(out + 64 * g, gp, ra, ntile);
                    else r16_f4_a16_n4(out + 64 * g, gp, ra, ntile);
                }
            __asm__ volatile("" ::: "memory");
            double s = (double)(ticks() - t0) / hz, pr = (double)ng * ntile * reps / 2; /* 256-weight units */
            double cyc = s * 2.0e9 / pr;
            printf("r16    N=%d: %.2f cycles/256w, %.2f cycles/256w/token, %.1f GB/s weights\n", N, cyc, cyc / N, pr * 144 / s / 1e9);
        }
    }
    return 0;
}
