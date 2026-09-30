/* Single-node microbenchmark: grouped routed-expert prefill GEMM, stage 2.
 * Weights stay 4-bit (Q4_K-style: 4-bit q, 6-bit scale and min per 32 columns,
 * fp32 d/dmin per 256 columns) and are expanded on the fly, one 512-column
 * K chunk at a time, into an L1-resident int8 panel64 tile that the existing
 * gk_gemm_tile6x4p_asm micro-kernel consumes.  The Q4_K min term
 *     y[t][r] -= sum_b dmin_r*m_rb * xs[t][b]*sum(xq[t][b])
 * is applied by a small fp32 pass per 64-row panel.
 *
 * Nibble layout of one 64-row x 32-column block (1024 B): [jj 0..3][p 0..3][64 B]
 * with byte (r*4+i) of group (jj,p) = lo: q[16p+r][4jj+i], hi: q[16p+r][4(jj+4)+i],
 * so ONE 64-byte load gives the panel64 groups j=jj (and 0xF) and j=jj+4 (lsr 4).
 *
 * env: GK_DOWN=nib|i8 (down projection storage; real model is Q5_K/Q6_K)
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I. -Ikern \
 *          bench_glm53f_moe_q4.c kern/glm53f_kern_gemm.c kern/glm53f_kern_gemm_asm.S -lm
 */
#define _GNU_SOURCE
#include <arm_sve.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kern/glm53f_kern.h"

void gk_gemm_tile6x4p_asm(const int8_t *w, long nblocks, long kgroups4, const int8_t *xp,
                          const float *xsp, float *y, size_t ldy_bytes);

enum { H = 4096, INTER = 256, GU_ROWS = 512, NEXP = 288, TOPK = 8, NPART_PER_EXP = 8,
       RANKS = 12, SB = 32, MAXM = 192, LDY_GU = 512, LDY_DN = 4096 + 64,
       KC = 512, /* K chunk columns */
       BLK_I8 = 64 * SB + 256, /* panel64 block bytes */
       SBLK_BYTES = 8 * 1024 + 8 * 64 + 8 * 64 + 256 + 256 /* 9728 */ };

static uint64_t rng_s = 88172645463325252ull;
static uint32_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t)(rng_s >> 11); }
static double now(void) { return omp_get_wtime(); }
static inline uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v)); return v; }
static double tick_hz(void) { uint64_t f; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); return (double)f; }
static void *al(size_t n) { void *p = NULL; if (posix_memalign(&p, 256, n ? n : 256)) { perror("alloc"); exit(1); } return p; }

/* ---- Q4 panel format --------------------------------------------------- */
static size_t q4_panel_bytes(int K) { return (size_t)(K / 256) * SBLK_BYTES; }

typedef struct { uint8_t *q; uint8_t *sc, *m; float *d, *dmin; int K; } q4_src; /* q:64xK (0..15), sc/m: 64x(K/32), d/dmin: 64x(K/256) */

static q4_src q4_random_src(int K) {
    q4_src s = { al(64 * (size_t)K), al(64 * (size_t)(K / 32)), al(64 * (size_t)(K / 32)), al(64 * (size_t)(K / 256) * 4), al(64 * (size_t)(K / 256) * 4), K };
    for (int i = 0; i < 64 * K; ++i) s.q[i] = rnd() & 15;
    for (int i = 0; i < 64 * (K / 32); ++i) { s.sc[i] = 1 + rnd() % 63; s.m[i] = rnd() % 64; }
    for (int i = 0; i < 64 * (K / 256); ++i) { s.d[i] = (0.5f + (rnd() % 1000) * 1e-3f) * 4e-5f; s.dmin[i] = (0.5f + (rnd() % 1000) * 1e-3f) * 4e-5f; }
    return s;
}

static void q4_pack(uint8_t *dst, const q4_src *s) {
    const int K = s->K, nsb = K / 256;
    for (int sbi = 0; sbi < nsb; ++sbi) {
        uint8_t *sp = dst + (size_t)sbi * SBLK_BYTES;
        uint8_t *nib = sp, *sc = sp + 8 * 1024, *mm = sc + 8 * 64;
        float *d = (float *)(mm + 8 * 64), *dm = d + 64;
        for (int bi = 0; bi < 8; ++bi) {
            const int b = sbi * 8 + bi;
            for (int jj = 0; jj < 4; ++jj)
                for (int p = 0; p < 4; ++p)
                    for (int r = 0; r < 16; ++r)
                        for (int i = 0; i < 4; ++i) {
                            const int row = 16 * p + r;
                            uint8_t lo = s->q[(size_t)row * K + b * 32 + 4 * jj + i];
                            uint8_t hi = s->q[(size_t)row * K + b * 32 + 4 * (jj + 4) + i];
                            nib[(size_t)bi * 1024 + (jj * 4 + p) * 64 + r * 4 + i] = lo | (uint8_t)(hi << 4);
                        }
            for (int row = 0; row < 64; ++row) { sc[bi * 64 + row] = s->sc[(size_t)row * (K / 32) + b]; mm[bi * 64 + row] = s->m[(size_t)row * (K / 32) + b]; }
        }
        for (int row = 0; row < 64; ++row) { d[row] = s->d[(size_t)row * nsb + sbi]; dm[row] = s->dmin[(size_t)row * nsb + sbi]; }
    }
}

/* Expand one superblock (8 blocks) into 8 panel64 int8 blocks + scale floats,
 * and the min terms dmin*m into mina[bi*64 + row]. */
static inline void q4_expand_sblk(const uint8_t *sp, uint8_t *cb, float *mina) {
    const svbool_t pg8 = svptrue_b8(), pg32 = svptrue_b32();
    const uint8_t *nib = sp, *sc = sp + 8 * 1024, *mm = sc + 8 * 64;
    const float *d = (const float *)(mm + 8 * 64), *dm = d + 64;
    for (int bi = 0; bi < 8; ++bi) {
        uint8_t *dst = cb + (size_t)bi * BLK_I8;
        const uint8_t *src = nib + (size_t)bi * 1024;
        for (int jj = 0; jj < 4; ++jj)
            for (int p = 0; p < 4; ++p) {
                svuint8_t v = svld1_u8(pg8, src + (jj * 4 + p) * 64);
                svst1_u8(pg8, dst + jj * 256 + p * 64, svand_n_u8_x(pg8, v, 15));
                svst1_u8(pg8, dst + (jj + 4) * 256 + p * 64, svlsr_n_u8_x(pg8, v, 4));
            }
        float *sf = (float *)(dst + 64 * SB);
        for (int v = 0; v < 4; ++v) {
            svfloat32_t dd = svld1_f32(pg32, d + v * 16), mmv = svld1_f32(pg32, dm + v * 16);
            svfloat32_t s32 = svcvt_f32_u32_x(pg32, svld1ub_u32(pg32, sc + bi * 64 + v * 16));
            svfloat32_t m32 = svcvt_f32_u32_x(pg32, svld1ub_u32(pg32, mm + bi * 64 + v * 16));
            svst1_f32(pg32, sf + v * 16, svmul_f32_x(pg32, s32, dd));
            svst1_f32(pg32, mina + bi * 64 + v * 16, svmul_f32_x(pg32, m32, mmv));
        }
    }
}

typedef struct { double expand, gemm, corr; } q4_time;

/* rows x K GEMM for mpad tokens (multiple of 6) using nibble weights W. */
static int getenv_pf = 1;
static void gemm_q4(const uint8_t *W, int K, int rows, int mpad, const int8_t *xp, const float *xsp,
                    const float *Bg /* [mpad][K/32] */, float *y, size_t ldy, uint8_t *cbuf, float *mina, q4_time *tm) {
    const size_t pbytes = q4_panel_bytes(K);
    const int nb = K / 32;
    for (int r = 0; r < rows; r += 64) {
        const uint8_t *wp = W + (size_t)(r / 64) * pbytes;
        for (int t = 0; t < mpad; ++t) memset(y + (size_t)t * ldy + r, 0, 64 * 4);
        for (int k0 = 0; k0 < K; k0 += KC) {
            const int kn = K - k0 < KC ? K - k0 : KC, nblk = kn / 32;
            uint64_t a0 = ticks();
            if (getenv_pf) {
                const char *nx = (const char *)wp + (size_t)(k0 / 256 + kn / 256) * SBLK_BYTES; /* next chunk (spills into next panel: contiguous) */
                for (size_t o = 0; o < (size_t)(kn / 256) * SBLK_BYTES; o += 256) __builtin_prefetch(nx + o, 0, 2);
            }
            for (int s = 0; s < kn / 256; ++s)
                q4_expand_sblk(wp + (size_t)(k0 / 256 + s) * SBLK_BYTES, cbuf + (size_t)s * 8 * BLK_I8, mina + (size_t)(k0 / 32 + s * 8) * 64);
            uint64_t a1 = ticks();
            for (int t = 0; t < mpad; t += 6)
                gk_gemm_tile6x4p_asm((const int8_t *)cbuf, nblk, SB / 16, xp + (size_t)(t / 6) * 6 * K + (size_t)(k0 / 4) * 24,
                                     xsp + (size_t)(t / 6) * 6 * nb + (size_t)(k0 / 32) * 6, y + (size_t)t * ldy + r, ldy * 4);
            uint64_t a2 = ticks();
            uint64_t c0 = ticks();
            {
                const svbool_t pg = svptrue_b32();
                const int b0 = k0 / 32;
                for (int t = 0; t < mpad; t += 4) {
#define ACC(u) svfloat32_t a##u##0 = svdup_f32(0), a##u##1 = a##u##0, a##u##2 = a##u##0, a##u##3 = a##u##0
                    ACC(0); ACC(1); ACC(2); ACC(3);
#undef ACC
                    const float *bg0 = Bg + (size_t)t * nb, *bg1 = Bg + (size_t)(t + 1 < mpad ? t + 1 : mpad - 1) * nb;
                    const float *bg2 = Bg + (size_t)(t + 2 < mpad ? t + 2 : mpad - 1) * nb, *bg3 = Bg + (size_t)(t + 3 < mpad ? t + 3 : mpad - 1) * nb;
                    for (int b = b0; b < b0 + nblk; ++b) {
                        svfloat32_t m0 = svld1_f32(pg, mina + (size_t)b * 64), m1 = svld1_f32(pg, mina + (size_t)b * 64 + 16);
                        svfloat32_t m2 = svld1_f32(pg, mina + (size_t)b * 64 + 32), m3 = svld1_f32(pg, mina + (size_t)b * 64 + 48);
#define STEP(u) do { float bv = bg##u[b]; \
                        a##u##0 = svmla_n_f32_x(pg, a##u##0, m0, bv); a##u##1 = svmla_n_f32_x(pg, a##u##1, m1, bv); \
                        a##u##2 = svmla_n_f32_x(pg, a##u##2, m2, bv); a##u##3 = svmla_n_f32_x(pg, a##u##3, m3, bv); } while (0)
                        STEP(0); STEP(1); STEP(2); STEP(3);
#undef STEP
                    }
#define FIN(u) do { if (t + u < mpad) { float *yt = y + (size_t)(t + u) * ldy + r; \
                    svst1_f32(pg, yt, svsub_f32_x(pg, svld1_f32(pg, yt), a##u##0)); \
                    svst1_f32(pg, yt + 16, svsub_f32_x(pg, svld1_f32(pg, yt + 16), a##u##1)); \
                    svst1_f32(pg, yt + 32, svsub_f32_x(pg, svld1_f32(pg, yt + 32), a##u##2)); \
                    svst1_f32(pg, yt + 48, svsub_f32_x(pg, svld1_f32(pg, yt + 48), a##u##3)); } } while (0)
                    FIN(0); FIN(1); FIN(2); FIN(3);
#undef FIN
                }
            }
            tm->corr += (double)(ticks() - c0);
            tm->expand += (double)(a1 - a0); tm->gemm += (double)(a2 - a1);
        }
    }
}

/* int8 panel64 path for the down projection when GK_DOWN=i8 */
typedef struct { int pi, m; } task_t;
typedef struct { int8_t *xg, *xp, *a8, *xp2; float *xsg, *xsp, *ygu, *as, *asp, *Bg; uint8_t *cbuf; float *mina; } tbuf_t;

static void swiglu_quant(const float *ygu, int m, int mpad, int8_t *a8, float *as) {
    memset(a8, 0, (size_t)mpad * INTER);
    memset(as, 0, (size_t)mpad * (INTER / SB) * sizeof(float));
    for (int t = 0; t < m; ++t) {
        float a[INTER];
        for (int j = 0; j < INTER; ++j) {
            float g = ygu[(size_t)t * LDY_GU + j], u = ygu[(size_t)t * LDY_GU + INTER + j];
            g = g > 10.f ? 10.f : g;
            u = u > 10.f ? 10.f : u < -10.f ? -10.f : u;
            a[j] = g / (1.f + expf(-g)) * u;
        }
        for (int b = 0; b < INTER / SB; ++b) {
            float mx = 0;
            for (int i = 0; i < SB; ++i) { float v = fabsf(a[b * SB + i]); if (v > mx) mx = v; }
            float sc = mx / 127.f, inv = sc > 0 ? 1.f / sc : 0.f;
            as[(size_t)t * (INTER / SB) + b] = sc;
            for (int i = 0; i < SB; ++i) {
                float q = a[b * SB + i] * inv;
                a8[(size_t)t * INTER + b * SB + i] = (int8_t)(q < -127 ? -127 : q > 127 ? 127 : lrintf(q));
            }
        }
    }
}


/* ---- vectorized helpers ------------------------------------------------ */
static inline svfloat32_t sv_expf(svbool_t pg, svfloat32_t x) {
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

/* SwiGLU + per-32 int8 requant + bsum term (Bd[t][b] = scale*sum(q)) for m real tokens, zero pad to mpad. */
static void swiglu_quant_fast(const float *ygu, int m, int mpad, int8_t *a8, float *as, float *Bd) {
    const svbool_t pg = svptrue_b32();
    memset(a8 + (size_t)m * INTER, 0, (size_t)(mpad - m) * INTER);
    memset(as + (size_t)m * (INTER / SB), 0, (size_t)(mpad - m) * (INTER / SB) * 4);
    memset(Bd + (size_t)m * (INTER / SB), 0, (size_t)(mpad - m) * (INTER / SB) * 4);
    for (int t = 0; t < m; ++t) {
        const float *g = ygu + (size_t)t * LDY_GU, *u = g + INTER;
        for (int b = 0; b < INTER / SB; ++b) {
            svfloat32_t a0, a1;
            {
                svfloat32_t gv = svmin_n_f32_x(pg, svld1_f32(pg, g + b * SB), 10.f);
                svfloat32_t uv = svmax_n_f32_x(pg, svmin_n_f32_x(pg, svld1_f32(pg, u + b * SB), 10.f), -10.f);
                svfloat32_t e = sv_expf(pg, svneg_f32_x(pg, gv));
                a0 = svmul_f32_x(pg, svdiv_f32_x(pg, gv, svadd_n_f32_x(pg, e, 1.f)), uv);
            }
            {
                svfloat32_t gv = svmin_n_f32_x(pg, svld1_f32(pg, g + b * SB + 16), 10.f);
                svfloat32_t uv = svmax_n_f32_x(pg, svmin_n_f32_x(pg, svld1_f32(pg, u + b * SB + 16), 10.f), -10.f);
                svfloat32_t e = sv_expf(pg, svneg_f32_x(pg, gv));
                a1 = svmul_f32_x(pg, svdiv_f32_x(pg, gv, svadd_n_f32_x(pg, e, 1.f)), uv);
            }
            float mx = svmaxv_f32(pg, svmax_f32_x(pg, svabs_f32_x(pg, a0), svabs_f32_x(pg, a1)));
            float sc = mx / 127.f, inv = sc > 0 ? 1.f / sc : 0.f;
            svint32_t q0 = svcvt_s32_f32_x(pg, svrintn_f32_x(pg, svmul_n_f32_x(pg, a0, inv)));
            svint32_t q1 = svcvt_s32_f32_x(pg, svrintn_f32_x(pg, svmul_n_f32_x(pg, a1, inv)));
            q0 = svmax_n_s32_x(pg, svmin_n_s32_x(pg, q0, 127), -127); q1 = svmax_n_s32_x(pg, svmin_n_s32_x(pg, q1, 127), -127);
            svst1b_s32(pg, a8 + (size_t)t * INTER + b * SB, q0);
            svst1b_s32(pg, a8 + (size_t)t * INTER + b * SB + 16, q1);
            as[(size_t)t * (INTER / SB) + b] = sc;
            Bd[(size_t)t * (INTER / SB) + b] = sc * (float)(svaddv_s32(pg, q0) + svaddv_s32(pg, q1));
        }
    }
}

/* Pack 6 token rows: xp32[g4*6 + t] = row_t[g4] (32-bit quads), xsp[b*6 + t] = xs_t[b]. */
static void pack6_fast(int8_t *xp, float *xsp, const int8_t *const rows[6], const float *const xs[6], int K) {
    uint32_t *dst = (uint32_t *)xp;
    const uint32_t *r0 = (const uint32_t *)rows[0], *r1 = (const uint32_t *)rows[1], *r2 = (const uint32_t *)rows[2];
    const uint32_t *r3 = (const uint32_t *)rows[3], *r4 = (const uint32_t *)rows[4], *r5 = (const uint32_t *)rows[5];
    for (int g = 0; g < K / 4; ++g) {
        dst[g * 6 + 0] = r0[g]; dst[g * 6 + 1] = r1[g]; dst[g * 6 + 2] = r2[g];
        dst[g * 6 + 3] = r3[g]; dst[g * 6 + 4] = r4[g]; dst[g * 6 + 5] = r5[g];
    }
    for (int b = 0; b < K / SB; ++b) for (int t = 0; t < 6; ++t) xsp[b * 6 + t] = xs[t][b];
}

/* Bg[t][b] = xs[t][b] * sum(xq[t][b*32..]) for the min correction */
static void compute_bt(const int8_t *x, const float *xs, int m, int nb, int ldx, float *out) {
    for (int t = 0; t < m; ++t)
        for (int b = 0; b < nb; ++b) {
            int s = 0;
            for (int i = 0; i < 32; ++i) s += x[(size_t)t * ldx + b * 32 + i];
            out[(size_t)t * nb + b] = xs[(size_t)t * nb + b] * (float)s;
        }
}

static float ref_w(const q4_src *s, int row, int k) { /* dequantized weight */
    const int b = k / 32, sbi = k / 256, nsb = s->K / 256;
    return s->d[(size_t)row * nsb + sbi] * s->sc[(size_t)row * (s->K / 32) + b] * s->q[(size_t)row * s->K + k]
         - s->dmin[(size_t)row * nsb + sbi] * s->m[(size_t)row * (s->K / 32) + b];
}

int main(int argc, char **argv) {
    const int C = argc > 1 ? atoi(argv[1]) : 512;
    const int iters = argc > 2 ? atoi(argv[2]) : 5;
    const int down_i8 = getenv("GK_DOWN") && !strcmp(getenv("GK_DOWN"), "i8");
    if (getenv("GK_PF")) getenv_pf = atoi(getenv("GK_PF"));
    const int nparts = NEXP * NPART_PER_EXP / RANKS, nthr = omp_get_max_threads();
    printf("C=%d iters=%d threads=%d parts=%d down=%s\n", C, iters, nthr, nparts, down_i8 ? "i8" : "q4");

    int *ptok = al((size_t)nparts * MAXM * sizeof(int)), *pcnt = calloc(nparts, sizeof(int));
    int *cpi = al((size_t)C * 16 * sizeof(int)), *cslot = al((size_t)C * 16 * sizeof(int)), *ccnt = calloc(C, sizeof(int));
    float *cw = al((size_t)C * 16 * sizeof(float));
    for (int t = 0; t < C; ++t) {
        int ex[NEXP];
        for (int i = 0; i < NEXP; ++i) ex[i] = i;
        for (int k = 0; k < TOPK; ++k) {
            int j = k + rnd() % (NEXP - k), e = ex[j]; ex[j] = ex[k]; ex[k] = e;
            for (int p = 0; p < NPART_PER_EXP; ++p) {
                int g = e * NPART_PER_EXP + p;
                if (g % RANKS) continue;
                int pi = g / RANKS;
                if (pcnt[pi] >= MAXM) { fprintf(stderr, "MAXM exceeded\n"); return 1; }
                ptok[(size_t)pi * MAXM + pcnt[pi]] = t;
                cpi[t * 16 + ccnt[t]] = pi; cslot[t * 16 + ccnt[t]] = pcnt[pi]; cw[t * 16 + ccnt[t]] = 1.0f / TOPK;
                pcnt[pi]++; ccnt[t]++;
            }
        }
    }
    long real_assign = 0, padded_assign = 0;
    for (int i = 0; i < nparts; ++i) { real_assign += pcnt[i]; padded_assign += (pcnt[i] + 5) / 6 * 6; }
    printf("assignments real=%ld padded=%ld (%.1f%% tile fill) mean M=%.2f\n", real_assign, padded_assign, 100.0 * real_assign / padded_assign, (double)real_assign / nparts);

    int8_t *X = al((size_t)C * H); float *XS = al((size_t)C * (H / SB) * sizeof(float)), *BT = al((size_t)C * (H / SB) * sizeof(float));
    for (size_t i = 0; i < (size_t)C * H; ++i) X[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (size_t i = 0; i < (size_t)C * (H / SB); ++i) XS[i] = 0.001f + (rnd() % 1000) * 1e-6f;
    compute_bt(X, XS, C, H / SB, H, BT);
    int8_t *ZROW = al(H); float *ZXS = al((H / SB) * 4); memset(ZROW, 0, H); memset(ZXS, 0, (H / SB) * 4);

    /* weights */
    const size_t gub = q4_panel_bytes(H), gu_part = (GU_ROWS / 64) * gub;
    const size_t dnb = down_i8 ? gk_panel64_bytes(SB, INTER) : q4_panel_bytes(INTER), dn_part = (H / 64) * dnb;
    q4_src gsrc = q4_random_src(H), dsrc = q4_random_src(INTER);
    uint8_t *gu_t = al(gub), *dn_t = al(dnb);
    q4_pack(gu_t, &gsrc);
    int8_t *dq = al(64 * INTER); float *dscale = al(64 * (INTER / SB) * 4); /* i8 down source, for the reference */
    if (down_i8) {
        for (int i = 0; i < 64 * INTER; ++i) dq[i] = (int8_t)((int)(rnd() % 255) - 127);
        for (int i = 0; i < 64 * (INTER / SB); ++i) dscale[i] = 0.0005f + (rnd() % 1000) * 1e-6f;
        gk_pack_panel64(SB, dn_t, dq, dscale, 64, INTER);
    } else q4_pack(dn_t, &dsrc);
    uint8_t *WGU = al((size_t)nparts * gu_part), *WDN = al((size_t)nparts * dn_part);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < nparts; ++p) {
        for (int b = 0; b < GU_ROWS / 64; ++b) memcpy(WGU + (size_t)p * gu_part + b * gub, gu_t, gub);
        for (int b = 0; b < H / 64; ++b) memcpy(WDN + (size_t)p * dn_part + b * dnb, dn_t, dnb);
    }
    printf("weights: %.1f MiB (gate/up q4 %.1f MiB, down %s %.1f MiB)\n", (double)nparts * (gu_part + dn_part) / 1048576.0,
           (double)nparts * gu_part / 1048576.0, down_i8 ? "i8" : "q4", (double)nparts * dn_part / 1048576.0);

    float *YD = al((size_t)nparts * MAXM * LDY_DN * sizeof(float)), *OUT = al((size_t)C * H * sizeof(float));
    task_t *tasks = malloc(nparts * sizeof(task_t));
    for (int i = 0; i < nparts; ++i) tasks[i] = (task_t){i, pcnt[i]};
    for (int i = 0; i < nparts; ++i) for (int j = i + 1; j < nparts; ++j) if (tasks[j].m > tasks[i].m) { task_t x = tasks[i]; tasks[i] = tasks[j]; tasks[j] = x; }

    tbuf_t *tb = calloc(nthr, sizeof(tbuf_t));
    #pragma omp parallel
    {
        tbuf_t *B = &tb[omp_get_thread_num()];
        B->xg = al((size_t)MAXM * H); B->xp = al((size_t)MAXM * H); B->a8 = al((size_t)MAXM * INTER); B->xp2 = al((size_t)MAXM * INTER);
        B->xsg = al((size_t)MAXM * (H / SB) * 4); B->xsp = al((size_t)MAXM * (H / SB) * 4); B->Bg = al((size_t)MAXM * (H / SB) * 4);
        B->ygu = al((size_t)MAXM * LDY_GU * 4); B->as = al((size_t)MAXM * (INTER / SB) * 4); B->asp = al((size_t)MAXM * (INTER / SB) * 4);
        B->cbuf = al((size_t)(KC / 32) * BLK_I8); B->mina = al((size_t)(H / 32) * 64 * 4);
        memset(B->xg, 0, (size_t)MAXM * H); memset(B->xp, 0, (size_t)MAXM * H); memset(B->a8, 0, (size_t)MAXM * INTER); memset(B->xp2, 0, (size_t)MAXM * INTER);
        memset(B->xsg, 0, (size_t)MAXM * (H / SB) * 4); memset(B->xsp, 0, (size_t)MAXM * (H / SB) * 4); memset(B->Bg, 0, (size_t)MAXM * (H / SB) * 4);
        memset(B->ygu, 0, (size_t)MAXM * LDY_GU * 4); memset(B->as, 0, (size_t)MAXM * (INTER / SB) * 4); memset(B->asp, 0, (size_t)MAXM * (INTER / SB) * 4);
        memset(B->cbuf, 0, (size_t)(KC / 32) * BLK_I8); memset(B->mina, 0, (size_t)(H / 32) * 64 * 4);
    }
    memset(YD, 0, (size_t)nparts * MAXM * LDY_DN * sizeof(float));

    q4_time *ttm = calloc(nthr * 2, sizeof(q4_time)); /* [tid*2+0]=gu, [tid*2+1]=down */
    double *tacc = calloc((size_t)nthr * 8, sizeof(double));
    int next_task = 0;
    double best = 1e9, sum = 0, best_comb = 0;
    for (int it = -2; it < iters; ++it) {
        memset(tacc, 0, (size_t)nthr * 8 * sizeof(double)); memset(ttm, 0, (size_t)nthr * 2 * sizeof(q4_time));
        next_task = 0;
        double t0 = now();
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            tbuf_t *B = &tb[tid];
            double *ta = tacc + (size_t)tid * 8;
            for (;;) {
                int ti = __atomic_fetch_add(&next_task, 1, __ATOMIC_RELAXED);
                if (ti >= nparts) break;
                const int pi = tasks[ti].pi, m = tasks[ti].m, mpad = (m + 5) / 6 * 6;
                if (!m) continue;
                double a0 = now();
                static const int8_t zrow_dummy = 0; (void)zrow_dummy;
                for (int g = 0; g < mpad / 6; ++g) {
                    const int8_t *rows[6]; const float *xsr[6];
                    for (int u = 0; u < 6; ++u) {
                        int sl = g * 6 + u;
                        if (sl < m) { int t = ptok[(size_t)pi * MAXM + sl]; rows[u] = X + (size_t)t * H; xsr[u] = XS + (size_t)t * (H / SB); }
                        else { rows[u] = ZROW; xsr[u] = ZXS; }
                    }
                    pack6_fast(B->xp + (size_t)g * 6 * H, B->xsp + (size_t)g * 6 * (H / SB), rows, xsr, H);
                }
                for (int sl = 0; sl < mpad; ++sl) {
                    if (sl < m) memcpy(B->Bg + (size_t)sl * (H / SB), BT + (size_t)ptok[(size_t)pi * MAXM + sl] * (H / SB), (H / SB) * 4);
                    else memset(B->Bg + (size_t)sl * (H / SB), 0, (H / SB) * 4);
                }
                double a1 = now();
                gemm_q4(WGU + (size_t)pi * gu_part, H, GU_ROWS, mpad, B->xp, B->xsp, B->Bg, B->ygu, LDY_GU, B->cbuf, B->mina, &ttm[tid * 2]);
                double a2 = now();
                float *Bd = B->Bg; /* min-correction input of the activation (gate/up Bg no longer needed) */
                swiglu_quant_fast(B->ygu, m, mpad, B->a8, B->as, Bd);
                for (int g = 0; g < mpad / 6; ++g) {
                    const int8_t *rows[6]; const float *xsr[6];
                    for (int u = 0; u < 6; ++u) { rows[u] = B->a8 + (size_t)(g * 6 + u) * INTER; xsr[u] = B->as + (size_t)(g * 6 + u) * (INTER / SB); }
                    pack6_fast(B->xp2 + (size_t)g * 6 * INTER, B->asp + (size_t)g * 6 * (INTER / SB), rows, xsr, INTER);
                }
                double a3 = now();
                if (down_i8)
                    gk_gemm_panel64(SB, WDN + (size_t)pi * dn_part, INTER, 0, H, 0, mpad, B->xp2, B->asp, YD + (size_t)pi * MAXM * LDY_DN, LDY_DN);
                else
                    gemm_q4(WDN + (size_t)pi * dn_part, INTER, H, mpad, B->xp2, B->asp, Bd, YD + (size_t)pi * MAXM * LDY_DN, LDY_DN, B->cbuf, B->mina, &ttm[tid * 2 + 1]);
                double a4 = now();
                ta[0] += a1 - a0; ta[1] += a2 - a1; ta[2] += a3 - a2; ta[3] += a4 - a3;
            }
        }
        double t1 = now();
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < C; ++t) {
            float *o = OUT + (size_t)t * H;
            memset(o, 0, H * 4);
            for (int c = 0; c < ccnt[t]; ++c) {
                const float *y = YD + ((size_t)cpi[t * 16 + c] * MAXM + cslot[t * 16 + c]) * LDY_DN;
                const float w = cw[t * 16 + c];
                for (int i = 0; i < H; ++i) o[i] += w * y[i];
            }
        }
        double t2 = now();
        if (it >= 0) { double tot = t2 - t0; sum += tot; if (tot < best) { best = tot; best_comb = t2 - t1; } }
    }
    double ph[4] = {0}, gu[3] = {0}, dn[3] = {0};
    for (int t = 0; t < nthr; ++t) {
        for (int k = 0; k < 4; ++k) ph[k] += tacc[t * 8 + k];
        gu[0] += ttm[t * 2].expand; gu[1] += ttm[t * 2].gemm; gu[2] += ttm[t * 2].corr;
        dn[0] += ttm[t * 2 + 1].expand; dn[1] += ttm[t * 2 + 1].gemm; dn[2] += ttm[t * 2 + 1].corr;
    }
    { double f = tick_hz(); for (int k = 0; k < 3; ++k) { gu[k] /= f; dn[k] /= f; } }
    const double macs_real = (double)real_assign * ((double)GU_ROWS * H + (double)H * INTER);
    const double macs_pad = (double)padded_assign * ((double)GU_ROWS * H + (double)H * INTER);
    const double peak = 2.0e9 * 128.0 * nthr;
    printf("best %.3f ms (mean %.3f) combine %.3f ms\n", best * 1e3, sum / iters * 1e3, best_comb * 1e3);
    printf("per-thread-avg ms: gather+pack %.3f | gu total %.3f (expand %.3f gemm %.3f corr %.3f) | swiglu+pack+bt %.3f | down total %.3f (expand %.3f gemm %.3f corr %.3f)\n",
           ph[0] / nthr * 1e3, ph[1] / nthr * 1e3, gu[0] / nthr * 1e3, gu[1] / nthr * 1e3, gu[2] / nthr * 1e3,
           ph[2] / nthr * 1e3, ph[3] / nthr * 1e3, dn[0] / nthr * 1e3, dn[1] / nthr * 1e3, dn[2] / nthr * 1e3);
    printf("%.2f GMAC real, %.2f GMAC padded -> %.2f TMAC/s = %.1f%% of SDOT peak\n", macs_real / 1e9, macs_pad / 1e9, macs_pad / best / 1e12, 100.0 * macs_pad / best / peak);
    printf("=> %.3f ms per chunk-layer; 42 layers = %.1f ms/chunk = %.0f tok/s (routed-expert local compute only)\n", best * 1e3, best * 42e3, C / (best * 42));

    /* correctness: largest part vs dequantized-float reference (double accumulation) */
    {
        int bi = 0;
        for (int i = 1; i < nparts; ++i) if (pcnt[i] > pcnt[bi]) bi = i;
        int pi = bi, m = pcnt[pi], mpad = (m + 5) / 6 * 6;
        float *ref_gu = al((size_t)MAXM * LDY_GU * 4);
        memset(ref_gu, 0, (size_t)MAXM * LDY_GU * 4);
        for (int s = 0; s < m; ++s) {
            int t = ptok[(size_t)pi * MAXM + s];
            for (int r = 0; r < GU_ROWS; ++r) {
                double acc = 0;
                for (int b = 0; b < H / SB; ++b) {
                    double blk = 0;
                    for (int i = 0; i < SB; ++i) blk += (double)ref_w(&gsrc, r % 64, b * SB + i) * (double)X[(size_t)t * H + b * SB + i];
                    acc += blk * XS[(size_t)t * (H / SB) + b];
                }
                ref_gu[(size_t)s * LDY_GU + r] = (float)acc;
            }
        }
        int8_t *a8 = al((size_t)MAXM * INTER); float *as = al((size_t)MAXM * 8 * 4);
        swiglu_quant(ref_gu, m, mpad, a8, as);
        double se = 0, sr = 0;
        for (int s = 0; s < m; ++s)
            for (int r = 0; r < H; ++r) {
                double acc = 0;
                for (int b = 0; b < INTER / SB; ++b) {
                    double blk = 0;
                    for (int i = 0; i < SB; ++i) {
                        double w = down_i8 ? (double)dq[(size_t)(r % 64) * INTER + b * SB + i] * dscale[(size_t)(r % 64) * (INTER / SB) + b]
                                           : (double)ref_w(&dsrc, r % 64, b * SB + i);
                        blk += w * (down_i8 ? (double)a8[(size_t)s * INTER + b * SB + i] * 1.0 : (double)a8[(size_t)s * INTER + b * SB + i]);
                    }
                    acc += (down_i8 ? blk : blk) * as[(size_t)s * (INTER / SB) + b];
                }
                double got = YD[((size_t)pi * MAXM + s) * LDY_DN + r], df = got - acc;
                se += df * df; sr += acc * acc;
            }
        double rel = sqrt(se / (sr + 1e-30));
        printf("correctness (part %d, M=%d): rel L2 err vs dequantized reference = %.3e %s\n", pi, m, rel, rel < 1e-3 ? "PASS" : "FAIL");
    }
    return 0;
}
