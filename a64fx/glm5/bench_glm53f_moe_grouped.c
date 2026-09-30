/* Single-node microbenchmark: grouped routed-expert prefill GEMM (P2, stage 1).
 *
 * Models the per-rank routed-expert work of one MoE layer for a chunk of C
 * prefill tokens on GLM-5.3F EP12: 288 experts x 8 column parts = 2304
 * (expert,part) pairs, rank owns 192 of them (about 14 tokens each at C=512).
 * Each part: gate/up GEMM (512 rows x K=4096), SwiGLU + int8 requant,
 * down GEMM (4096 rows x K=256), then a token-ordered combine.
 *
 * Weights are int8 sb=32 panel64 (Q8_0-equivalent = ceiling for Q4_K which
 * needs on-the-fly expansion, stage 2).  Uses gk_gemm_panel64 unchanged.
 *
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I. -Ikern \
 *          bench_glm53f_moe_grouped.c kern/glm53f_kern_gemm.c kern/glm53f_kern_gemm_asm.S -lm
 * run:   OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores ./a.out [C=512] [iters=5] [gran=0|1]
 */
#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kern/glm53f_kern.h"

enum { H = 4096, INTER = 256, GU_ROWS = 512, NEXP = 288, TOPK = 8, NPART_PER_EXP = 8,
       RANKS = 12, SB = 32, MAXM = 192, LDY_GU = 512, LDY_DN = 4096 + 64 };

static uint64_t rng_s = 88172645463325252ull;
static uint32_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t)(rng_s >> 11); }
static double now(void) { return omp_get_wtime(); }
static void *al(size_t n) { void *p = NULL; if (posix_memalign(&p, 256, n ? n : 256)) { perror("alloc"); exit(1); } return p; }

typedef struct { int pi, m; } task_t;
typedef struct { int8_t *xg, *xp, *a8, *xp2; float *xsg, *xsp, *ygu, *as, *asp; } tbuf_t;

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

int main(int argc, char **argv) {
    const int C = argc > 1 ? atoi(argv[1]) : 512;
    const int iters = argc > 2 ? atoi(argv[2]) : 5;
    const int gran = argc > 3 ? atoi(argv[3]) : 0; /* 0: task=part, 1: task=(part,{gu,down split by row half}) */
    const int nparts = NEXP * NPART_PER_EXP / RANKS;
    if (getenv("GK_L1PF")) gk_gemm_l1pf = atoi(getenv("GK_L1PF"));
    if (getenv("GK_KCHUNK")) gk_gemm_kchunk = atoi(getenv("GK_KCHUNK"));
    if (getenv("GK_CHUNKPF")) gk_gemm_chunkpf = atoi(getenv("GK_CHUNKPF"));
    if (getenv("GK_YPF")) gk_gemm_ypf = atoi(getenv("GK_YPF"));
    printf("C=%d iters=%d gran=%d threads=%d parts=%d\n", C, iters, gran, omp_get_max_threads(), nparts);

    /* routing */
    int *ptok = al((size_t)nparts * MAXM * sizeof(int));
    int *pcnt = calloc(nparts, sizeof(int));
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
    long real_assign = 0, padded_assign = 0; int mmax = 0;
    for (int i = 0; i < nparts; ++i) { real_assign += pcnt[i]; padded_assign += (pcnt[i] + 5) / 6 * 6; if (pcnt[i] > mmax) mmax = pcnt[i]; }
    printf("assignments real=%ld padded=%ld (%.1f%% tile fill) mean M=%.2f max M=%d\n", real_assign, padded_assign,
           100.0 * real_assign / padded_assign, (double)real_assign / nparts, mmax);

    /* activations */
    int8_t *X = al((size_t)C * H); float *XS = al((size_t)C * (H / SB) * sizeof(float));
    for (size_t i = 0; i < (size_t)C * H; ++i) X[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (size_t i = 0; i < (size_t)C * (H / SB); ++i) XS[i] = 0.001f + (rnd() % 1000) * 1e-6f;

    /* weights: template panels replicated per part (distinct memory) */
    const size_t gub = gk_panel64_bytes(SB, H), dnb = gk_panel64_bytes(SB, INTER);
    const size_t gu_part = (GU_ROWS / 64) * gub, dn_part = (H / 64) * dnb;
    int8_t *tq = al(64 * H); float *tsc = al(64 * (H / SB) * sizeof(float));
    uint8_t *gu_t = al(gub), *dn_t = al(dnb);
    for (int i = 0; i < 64 * H; ++i) tq[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int i = 0; i < 64 * (H / SB); ++i) tsc[i] = 0.0005f + (rnd() % 1000) * 1e-6f;
    gk_pack_panel64(SB, gu_t, tq, tsc, 64, H);
    int8_t *tq2 = al(64 * INTER); float *tsc2 = al(64 * (INTER / SB) * sizeof(float));
    for (int i = 0; i < 64 * INTER; ++i) tq2[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int i = 0; i < 64 * (INTER / SB); ++i) tsc2[i] = 0.0005f + (rnd() % 1000) * 1e-6f;
    gk_pack_panel64(SB, dn_t, tq2, tsc2, 64, INTER);
    uint8_t *WGU = al((size_t)nparts * gu_part), *WDN = al((size_t)nparts * dn_part);
    const int ncmg = getenv("GK_NCMG") ? atoi(getenv("GK_NCMG")) : 4;
    const int nthr0 = omp_get_max_threads();
    const int per_cmg = (nthr0 + ncmg - 1) / ncmg;
    /* part -> CMG by LPT on padded M (weighted by thread count of the CMG) */
    int *pcmg = malloc(nparts * sizeof(int));
    {
        int *ord = malloc(nparts * sizeof(int));
        for (int i = 0; i < nparts; ++i) ord[i] = i;
        for (int i = 0; i < nparts; ++i) for (int j = i + 1; j < nparts; ++j)
            if (pcnt[ord[j]] > pcnt[ord[i]]) { int x = ord[i]; ord[i] = ord[j]; ord[j] = x; }
        double load[16] = {0}; int nth[16];
        for (int c = 0; c < ncmg; ++c) { int lo = c * per_cmg, hi = lo + per_cmg > nthr0 ? nthr0 : lo + per_cmg; nth[c] = hi > lo ? hi - lo : 1; }
        for (int i = 0; i < nparts; ++i) {
            int best_c = 0;
            for (int c = 1; c < ncmg; ++c) if (load[c] / nth[c] < load[best_c] / nth[best_c]) best_c = c;
            pcmg[ord[i]] = best_c; load[best_c] += (pcnt[ord[i]] + 5) / 6 * 6 + 2; /* +2: fixed per-part cost */
        }
        free(ord);
    }
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num(), c = tid / per_cmg, lane = tid % per_cmg;
        int seq = 0;
        for (int p = 0; p < nparts; ++p) {
            if (pcmg[p] != c) continue;
            if ((seq++ % per_cmg) != lane) continue;
            for (int b = 0; b < GU_ROWS / 64; ++b) memcpy(WGU + (size_t)p * gu_part + b * gub, gu_t, gub);
            for (int b = 0; b < H / 64; ++b) memcpy(WDN + (size_t)p * dn_part + b * dnb, dn_t, dnb);
        }
    }
    printf("weights: %.1f MiB int8 (Q4_K would be ~%.1f MiB)\n", (double)nparts * (gu_part + dn_part) / 1048576.0,
           (double)nparts * (gu_part + dn_part) / 1048576.0 * 0.5625 / 1.125);

    /* outputs */
    float *YD = al((size_t)nparts * MAXM * LDY_DN * sizeof(float));
    float *OUT = al((size_t)C * H * sizeof(float));

    /* per-CMG task lists ordered by M descending */
    task_t *tasks = malloc((size_t)ncmg * nparts * sizeof(task_t));
    int ntasks[16] = {0}; int next_task[16 * 16] = {0}; /* padded to separate cache lines */
    for (int i = 0; i < nparts; ++i) { int c = pcmg[i]; tasks[(size_t)c * nparts + ntasks[c]++] = (task_t){i, pcnt[i]}; }
    for (int c = 0; c < ncmg; ++c) {
        task_t *tl = tasks + (size_t)c * nparts;
        for (int i = 0; i < ntasks[c]; ++i) for (int j = i + 1; j < ntasks[c]; ++j) if (tl[j].m > tl[i].m) { task_t x = tl[i]; tl[i] = tl[j]; tl[j] = x; }
    }
    task_t big = tasks[0];
    for (int c = 0; c < ncmg; ++c) for (int i = 0; i < ntasks[c]; ++i) if (tasks[(size_t)c * nparts + i].m > big.m) big = tasks[(size_t)c * nparts + i];

    const int nthr = omp_get_max_threads();
    double *tacc = calloc((size_t)nthr * 8, sizeof(double));
    tbuf_t *tb = calloc(nthr, sizeof(tbuf_t));
    #pragma omp parallel
    {
        tbuf_t *B = &tb[omp_get_thread_num()];
        B->xg = al((size_t)MAXM * H); B->xp = al((size_t)MAXM * H); B->a8 = al((size_t)MAXM * INTER); B->xp2 = al((size_t)MAXM * INTER);
        B->xsg = al((size_t)MAXM * (H / SB) * 4); B->xsp = al((size_t)MAXM * (H / SB) * 4);
        B->ygu = al((size_t)MAXM * LDY_GU * 4); B->as = al((size_t)MAXM * (INTER / SB) * 4); B->asp = al((size_t)MAXM * (INTER / SB) * 4);
        memset(B->xg, 0, (size_t)MAXM * H); memset(B->xp, 0, (size_t)MAXM * H); memset(B->a8, 0, (size_t)MAXM * INTER); memset(B->xp2, 0, (size_t)MAXM * INTER);
        memset(B->xsg, 0, (size_t)MAXM * (H / SB) * 4); memset(B->xsp, 0, (size_t)MAXM * (H / SB) * 4);
        memset(B->ygu, 0, (size_t)MAXM * LDY_GU * 4); memset(B->as, 0, (size_t)MAXM * (INTER / SB) * 4); memset(B->asp, 0, (size_t)MAXM * (INTER / SB) * 4);
    }
    memset(YD, 0, (size_t)nparts * MAXM * LDY_DN * sizeof(float));
    double best = 1e9, sum = 0, best_comb = 0;
    for (int it = -2; it < iters; ++it) {
        memset(tacc, 0, (size_t)nthr * 8 * sizeof(double));
        for (int c = 0; c < 16; ++c) next_task[c * 16] = 0;
        double t0 = now();
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            tbuf_t *B = &tb[tid];
            int8_t *xg = B->xg, *xp = B->xp, *a8 = B->a8, *xp2 = B->xp2;
            float *xsg = B->xsg, *xsp = B->xsp, *ygu = B->ygu, *as = B->as, *asp = B->asp;
            double *ta = tacc + (size_t)tid * 8;
            const int mycmg = tid / per_cmg;
            for (;;) {
                int ti = __atomic_fetch_add(&next_task[mycmg * 16], 1, __ATOMIC_RELAXED);
                if (ti >= ntasks[mycmg]) break;
                const task_t tk = tasks[(size_t)mycmg * nparts + ti];
                const int pi = tk.pi, m = tk.m, mpad = (m + 5) / 6 * 6;
                if (!m) continue;
                double a0 = now();
                for (int s = 0; s < mpad; ++s) {
                    if (s < m) {
                        int t = ptok[(size_t)pi * MAXM + s];
                        memcpy(xg + (size_t)s * H, X + (size_t)t * H, H);
                        memcpy(xsg + (size_t)s * (H / SB), XS + (size_t)t * (H / SB), (H / SB) * 4);
                    } else { memset(xg + (size_t)s * H, 0, H); memset(xsg + (size_t)s * (H / SB), 0, (H / SB) * 4); }
                }
                for (int g = 0; g < mpad / 6; ++g)
                    gk_pack_act6(SB, xp + (size_t)g * 6 * H, xsp + (size_t)g * 6 * (H / SB), xg + (size_t)g * 6 * H, H,
                                 xsg + (size_t)g * 6 * (H / SB), H / SB, H);
                double a1 = now();
                gk_gemm_panel64(SB, WGU + (size_t)pi * gu_part, H, 0, GU_ROWS, 0, mpad, xp, xsp, ygu, LDY_GU);
                double a2 = now();
                swiglu_quant(ygu, m, mpad, a8, as);
                for (int g = 0; g < mpad / 6; ++g)
                    gk_pack_act6(SB, xp2 + (size_t)g * 6 * INTER, asp + (size_t)g * 6 * (INTER / SB), a8 + (size_t)g * 6 * INTER, INTER,
                                 as + (size_t)g * 6 * (INTER / SB), INTER / SB, INTER);
                double a3 = now();
                gk_gemm_panel64(SB, WDN + (size_t)pi * dn_part, INTER, 0, H, 0, mpad, xp2, asp, YD + (size_t)pi * MAXM * LDY_DN, LDY_DN);
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
    double ph[4] = {0, 0, 0, 0};
    for (int t = 0; t < nthr; ++t) for (int k = 0; k < 4; ++k) ph[k] += tacc[t * 8 + k];
    const double macs_real = (double)real_assign * ((double)GU_ROWS * H + (double)H * INTER);
    const double macs_pad = (double)padded_assign * ((double)GU_ROWS * H + (double)H * INTER);
    const double peak = 2.0e9 * 128.0 * nthr; /* MAC/s: 2 SDOT pipes x 64 MAC, 2.0 GHz */
    printf("best %.3f ms (mean %.3f) combine %.3f ms | per-thread-avg ms: gather+pack %.3f gu %.3f swiglu+pack %.3f down %.3f (1 iter, last)\n",
           best * 1e3, sum / iters * 1e3, best_comb * 1e3, ph[0] / nthr * 1e3, ph[1] / nthr * 1e3, ph[2] / nthr * 1e3, ph[3] / nthr * 1e3);
    printf("MoE-local only: %.2f GMAC real, %.2f GMAC padded -> %.2f TMAC/s padded = %.1f%% of SDOT peak; ideal(100%% tile fill,50%% peak)=%.3f ms\n",
           macs_real / 1e9, macs_pad / 1e9, macs_pad / best / 1e12, 100.0 * macs_pad / best / peak, macs_real / (0.5 * peak) * 1e3);
    printf("=> %.3f ms per chunk-layer; 42 layers = %.1f ms/chunk = %.0f tok/s (routed-expert local compute only)\n",
           best * 1e3, best * 42e3, C / (best * 42));

    /* correctness: part with most tokens vs scalar reference */
    {
        int pi = big.pi, m = big.m, mpad = (m + 5) / 6 * 6;
        float *ref_gu = al((size_t)MAXM * LDY_GU * 4);
        memset(ref_gu, 0, (size_t)MAXM * LDY_GU * 4);
        for (int s = 0; s < m; ++s) {
            int t = ptok[(size_t)pi * MAXM + s];
            for (int r = 0; r < GU_ROWS; ++r) {
                float acc = 0;
                for (int b = 0; b < H / SB; ++b) {
                    int d = 0;
                    for (int i = 0; i < SB; ++i) d += (int)tq[(size_t)(r % 64) * H + b * SB + i] * (int)X[(size_t)t * H + b * SB + i];
                    acc += (float)d * tsc[(size_t)(r % 64) * (H / SB) + b] * XS[(size_t)t * (H / SB) + b];
                }
                ref_gu[(size_t)s * LDY_GU + r] = acc;
            }
        }
        int8_t *a8 = al((size_t)MAXM * INTER); float *as = al((size_t)MAXM * 8 * 4);
        swiglu_quant(ref_gu, m, mpad, a8, as);
        double se = 0, sr = 0;
        for (int s = 0; s < m; ++s)
            for (int r = 0; r < H; ++r) {
                float acc = 0;
                for (int b = 0; b < INTER / SB; ++b) {
                    int d = 0;
                    for (int i = 0; i < SB; ++i) d += (int)tq2[(size_t)(r % 64) * INTER + b * SB + i] * (int)a8[(size_t)s * INTER + b * SB + i];
                    acc += (float)d * tsc2[(size_t)(r % 64) * (INTER / SB) + b] * as[(size_t)s * (INTER / SB) + b];
                }
                double got = YD[((size_t)pi * MAXM + s) * LDY_DN + r], df = got - acc;
                se += df * df; sr += (double)acc * acc;
            }
        printf("correctness (part %d, M=%d): rel L2 err vs scalar ref = %.3e %s\n", pi, m, sqrt(se / (sr + 1e-30)),
               sqrt(se / (sr + 1e-30)) < 1e-3 ? "PASS" : "FAIL");
    }
    return 0;
}
