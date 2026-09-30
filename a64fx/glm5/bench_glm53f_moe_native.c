/* Single-node benchmark + correctness test of the grouped routed-expert prefill path that reads NATIVE GGUF
 * blocks (gate/up = Q4_K, down = Q5_K) via glm53f_moe_grouped_native.h, i.e. exactly the code the runtime uses.
 * Reference: common/ggml_dequant.h dequantize_row_q4_K / q5_K + float dot with dequantized int8 activations.
 *
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I. -I../../common \
 *          bench_glm53f_moe_native.c kern/glm53f_kern_gemm.c kern/glm53f_kern_gemm_asm.S -lm
 * run:   OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores XOS_MMM_L_PAGING_POLICY=demand:demand:demand ./a.out [C=512] [iters=8]
 */
#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "glm53f_moe_grouped_native.h"
#include "kern/glm53f_kern.h"

enum { H = 4096, INTER = 256, GU_ROWS = 512, NEXP = 288, TOPK = 8, NPART_PER_EXP = 8, RANKS = 12, MAXM = 192,
       LDY_GU = 512, LDY_DN = 4096 + 64 };
static uint64_t rng_s = 88172645463325252ull;
static uint32_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t)(rng_s >> 11); }
static void *al(size_t n) { void *p = NULL; if (posix_memalign(&p, 256, n ? n : 256)) { perror("alloc"); exit(1); } return p; }
static uint16_t f2h(float f) { /* small positive floats only */
    uint32_t b; memcpy(&b, &f, 4);
    int e = ((b >> 23) & 0xFF) - 127 + 15; uint32_t m = (b >> 13) & 0x3FF;
    return (uint16_t)((e << 10) | m);
}
/* random valid native rows */
static void fill_q4k(uint8_t *dst, int rows, int K) {
    for (int r = 0; r < rows; ++r)
        for (int s = 0; s < K / 256; ++s) {
            uint8_t *b = dst + ((size_t)r * (K / 256) + s) * 144;
            uint16_t d = f2h((0.5f + (rnd() % 1000) * 1e-3f) * 2e-3f), dm = f2h((0.5f + (rnd() % 1000) * 1e-3f) * 2e-3f);
            memcpy(b, &d, 2); memcpy(b + 2, &dm, 2);
            for (int i = 4; i < 144; ++i) b[i] = (uint8_t)rnd();
        }
}
static void fill_q5k(uint8_t *dst, int rows, int K) {
    for (int r = 0; r < rows; ++r)
        for (int s = 0; s < K / 256; ++s) {
            uint8_t *b = dst + ((size_t)r * (K / 256) + s) * 176;
            uint16_t d = f2h((0.5f + (rnd() % 1000) * 1e-3f) * 1e-3f), dm = f2h((0.5f + (rnd() % 1000) * 1e-3f) * 1e-3f);
            memcpy(b, &d, 2); memcpy(b + 2, &dm, 2);
            for (int i = 4; i < 176; ++i) b[i] = (uint8_t)rnd();
        }
}

typedef struct { int pi, m; } task_t;
typedef struct { int8_t *xp, *xp2, *a8; float *xsp, *Bg, *ygu, *as, *asp, *mina; uint8_t *cbuf; } tbuf_t;

int main(int argc, char **argv) {
    const int C = argc > 1 ? atoi(argv[1]) : 512, iters = argc > 2 ? atoi(argv[2]) : 8;
    const int nparts = NEXP * NPART_PER_EXP / RANKS, nthr = omp_get_max_threads();
    const int prefetch = getenv("GK_PF") ? atoi(getenv("GK_PF")) : 1;
    printf("C=%d iters=%d threads=%d parts=%d prefetch=%d (native Q4_K gate/up + Q5_K down)\n", C, iters, nthr, nparts, prefetch);
    int *ptok = al((size_t)nparts * MAXM * sizeof(int)), *pcnt = calloc(nparts, sizeof(int));
    int *cpi = al((size_t)C * 16 * sizeof(int)), *cslot = al((size_t)C * 16 * sizeof(int)), *ccnt = calloc(C, sizeof(int));
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
                ptok[(size_t)pi * MAXM + pcnt[pi]] = t; cpi[t * 16 + ccnt[t]] = pi; cslot[t * 16 + ccnt[t]] = pcnt[pi];
                pcnt[pi]++; ccnt[t]++;
            }
        }
    }
    long real_assign = 0, padded_assign = 0;
    for (int i = 0; i < nparts; ++i) { real_assign += pcnt[i]; padded_assign += (pcnt[i] + 5) / 6 * 6; }
    printf("assignments real=%ld padded=%ld (%.1f%% tile fill) mean M=%.2f\n", real_assign, padded_assign, 100.0 * real_assign / padded_assign, (double)real_assign / nparts);

    /* float hidden states -> int8 activations (the runtime quantizes the same way) */
    float *XF = al((size_t)C * H * 4);
    for (size_t i = 0; i < (size_t)C * H; ++i) XF[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
    int8_t *X = al((size_t)C * H); float *XS = al((size_t)C * (H / 32) * 4), *BT = al((size_t)C * (H / 32) * 4);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < C; ++t) gmn_quant_row(XF + (size_t)t * H, H, X + (size_t)t * H, XS + (size_t)t * (H / 32), BT + (size_t)t * (H / 32));

    /* native weights: one template per part shape, replicated into distinct memory */
    const size_t gu_rb = gmn_row_bytes(GMN_TYPE_Q4K, H), dn_rb = gmn_row_bytes(GMN_TYPE_Q5K, INTER);
    const size_t gu_part = (size_t)GU_ROWS * gu_rb, dn_part = (size_t)H * dn_rb;
    uint8_t *gu_t = al(gu_part), *dn_t = al(dn_part);
    fill_q4k(gu_t, GU_ROWS, H); fill_q5k(dn_t, H, INTER);
    uint8_t *WGU = al((size_t)nparts * gu_part), *WDN = al((size_t)nparts * dn_part);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < nparts; ++p) { memcpy(WGU + (size_t)p * gu_part, gu_t, gu_part); memcpy(WDN + (size_t)p * dn_part, dn_t, dn_part); }
    printf("weights: %.1f MiB native (gate/up Q4_K %.1f, down Q5_K %.1f)\n", (double)nparts * (gu_part + dn_part) / 1048576.0,
           (double)nparts * gu_part / 1048576.0, (double)nparts * dn_part / 1048576.0);

    float *YD = al((size_t)nparts * MAXM * LDY_DN * 4);
    task_t *tasks = malloc(nparts * sizeof(task_t));
    for (int i = 0; i < nparts; ++i) tasks[i] = (task_t){i, pcnt[i]};
    for (int i = 0; i < nparts; ++i) for (int j = i + 1; j < nparts; ++j) if (tasks[j].m > tasks[i].m) { task_t x = tasks[i]; tasks[i] = tasks[j]; tasks[j] = x; }
    int8_t *ZROW = al(H); float *ZXS = al((H / 32) * 4); memset(ZROW, 0, H); memset(ZXS, 0, (H / 32) * 4);

    tbuf_t *tb = calloc(nthr, sizeof(tbuf_t));
    #pragma omp parallel
    {
        tbuf_t *B = &tb[omp_get_thread_num()];
        B->xp = al((size_t)MAXM * H); B->xsp = al((size_t)MAXM * (H / 32) * 4); B->Bg = al((size_t)MAXM * (H / 32) * 4);
        B->xp2 = al((size_t)MAXM * INTER); B->a8 = al((size_t)MAXM * INTER); B->as = al((size_t)MAXM * 8 * 4); B->asp = al((size_t)MAXM * 8 * 4);
        B->ygu = al((size_t)MAXM * LDY_GU * 4); B->mina = al((size_t)(H / 32) * 64 * 4); B->cbuf = al((size_t)(GMN_KC / 32) * GMN_BLK);
        memset(B->xp, 0, (size_t)MAXM * H); memset(B->xsp, 0, (size_t)MAXM * (H / 32) * 4); memset(B->Bg, 0, (size_t)MAXM * (H / 32) * 4);
        memset(B->xp2, 0, (size_t)MAXM * INTER); memset(B->a8, 0, (size_t)MAXM * INTER); memset(B->as, 0, (size_t)MAXM * 32); memset(B->asp, 0, (size_t)MAXM * 32);
        memset(B->ygu, 0, (size_t)MAXM * LDY_GU * 4); memset(B->mina, 0, (size_t)(H / 32) * 64 * 4); memset(B->cbuf, 0, (size_t)(GMN_KC / 32) * GMN_BLK);
    }
    memset(YD, 0, (size_t)nparts * MAXM * LDY_DN * 4);

    double *tacc = calloc((size_t)nthr * 8, sizeof(double));
    int next_task = 0;
    double best = 1e9, sum = 0;
    for (int it = -2; it < iters; ++it) {
        memset(tacc, 0, (size_t)nthr * 8 * sizeof(double)); next_task = 0;
        double t0 = omp_get_wtime();
        #pragma omp parallel
        {
            tbuf_t *B = &tb[omp_get_thread_num()];
            double *ta = tacc + (size_t)omp_get_thread_num() * 8;
            for (;;) {
                int ti = __atomic_fetch_add(&next_task, 1, __ATOMIC_RELAXED);
                if (ti >= nparts) break;
                const int pi = tasks[ti].pi, m = tasks[ti].m, mpad = (m + 5) / 6 * 6;
                if (!m) continue;
                double a0 = omp_get_wtime();
                for (int g = 0; g < mpad / 6; ++g) {
                    const int8_t *rows[6]; const float *xsr[6];
                    for (int u = 0; u < 6; ++u) {
                        int sl = g * 6 + u;
                        if (sl < m) { int t = ptok[(size_t)pi * MAXM + sl]; rows[u] = X + (size_t)t * H; xsr[u] = XS + (size_t)t * (H / 32); }
                        else { rows[u] = ZROW; xsr[u] = ZXS; }
                    }
                    gmn_pack6(B->xp + (size_t)g * 6 * H, B->xsp + (size_t)g * 6 * (H / 32), rows, xsr, H);
                }
                for (int sl = 0; sl < mpad; ++sl) {
                    if (sl < m) memcpy(B->Bg + (size_t)sl * (H / 32), BT + (size_t)ptok[(size_t)pi * MAXM + sl] * (H / 32), (H / 32) * 4);
                    else memset(B->Bg + (size_t)sl * (H / 32), 0, (H / 32) * 4);
                }
                double a1 = omp_get_wtime();
                gmn_gemm(GMN_TYPE_Q4K, WGU + (size_t)pi * gu_part, gu_rb, H, GU_ROWS, mpad, B->xp, B->xsp, B->Bg, B->ygu, LDY_GU, B->cbuf, B->mina, prefetch);
                double a2 = omp_get_wtime();
                float *Bd = B->Bg;
                gmn_swiglu_quant(B->ygu, LDY_GU, INTER, m, mpad, B->a8, B->as, Bd);
                for (int g = 0; g < mpad / 6; ++g) {
                    const int8_t *rows[6]; const float *xsr[6];
                    for (int u = 0; u < 6; ++u) { rows[u] = B->a8 + (size_t)(g * 6 + u) * INTER; xsr[u] = B->as + (size_t)(g * 6 + u) * 8; }
                    gmn_pack6(B->xp2 + (size_t)g * 6 * INTER, B->asp + (size_t)g * 6 * 8, rows, xsr, INTER);
                }
                double a3 = omp_get_wtime();
                gmn_gemm(GMN_TYPE_Q5K, WDN + (size_t)pi * dn_part, dn_rb, INTER, H, mpad, B->xp2, B->asp, Bd, YD + (size_t)pi * MAXM * LDY_DN, LDY_DN, B->cbuf, B->mina, prefetch);
                double a4 = omp_get_wtime();
                ta[0] += a1 - a0; ta[1] += a2 - a1; ta[2] += a3 - a2; ta[3] += a4 - a3;
            }
        }
        double t1 = omp_get_wtime();
        if (it >= 0) { sum += t1 - t0; if (t1 - t0 < best) best = t1 - t0; }
    }
    double ph[4] = {0};
    for (int t = 0; t < nthr; ++t) for (int k = 0; k < 4; ++k) ph[k] += tacc[t * 8 + k];
    const double macs_pad = (double)padded_assign * ((double)GU_ROWS * H + (double)H * INTER);
    printf("best %.3f ms (mean %.3f), no combine\n", best * 1e3, sum / iters * 1e3);
    printf("per-thread-avg ms: gather+pack %.3f | gate/up %.3f | swiglu+pack %.3f | down %.3f\n", ph[0] / nthr * 1e3, ph[1] / nthr * 1e3, ph[2] / nthr * 1e3, ph[3] / nthr * 1e3);
    printf("%.2f TMAC/s padded = %.1f%% of SDOT peak; %.3f ms per chunk-layer (+~0.6 ms combine)\n", macs_pad / best / 1e12, 100.0 * macs_pad / best / (2.0e9 * 128.0 * nthr), best * 1e3);

    /* correctness: three parts (largest, a mid one, one with M small) vs dequantized-float reference */
    int bi = 0;
    for (int i = 1; i < nparts; ++i) if (pcnt[i] > pcnt[bi]) bi = i;
    int check[3] = {bi, nparts / 2, nparts - 1}; int ok_all = 1;
    for (int ci = 0; ci < 3; ++ci) {
        int pi = check[ci], m = pcnt[pi];
        if (!m) continue;
        /* dequantize all weight rows once */
        float *Wg = al((size_t)GU_ROWS * H * 4), *Wd = al((size_t)H * INTER * 4);
        for (int r = 0; r < GU_ROWS; ++r) dequantize_row_q4_K(WGU + (size_t)pi * gu_part + (size_t)r * gu_rb, Wg + (size_t)r * H, H);
        for (int r = 0; r < H; ++r) dequantize_row_q5_K(WDN + (size_t)pi * dn_part + (size_t)r * dn_rb, Wd + (size_t)r * INTER, INTER);
        double se = 0, sr = 0, se_gu = 0, sr_gu = 0;
        int mpad = (m + 5) / 6 * 6;
        float *gu_ref = al((size_t)mpad * LDY_GU * 4); memset(gu_ref, 0, (size_t)mpad * LDY_GU * 4);
        for (int s = 0; s < m; ++s) {
            int t = ptok[(size_t)pi * MAXM + s];
            float xd[H];
            for (int k = 0; k < H; ++k) xd[k] = (float)X[(size_t)t * H + k] * XS[(size_t)t * (H / 32) + k / 32];
            for (int r = 0; r < GU_ROWS; ++r) { double a = 0; for (int k = 0; k < H; ++k) a += (double)Wg[(size_t)r * H + k] * xd[k]; gu_ref[(size_t)s * LDY_GU + r] = (float)a; }
        }
        /* rerun the production path for this part on this thread's buffers to get its gate/up + down outputs */
        tbuf_t *B = &tb[0];
        int mp = mpad;
        for (int g = 0; g < mp / 6; ++g) {
            const int8_t *rows[6]; const float *xsr[6];
            for (int u = 0; u < 6; ++u) { int sl = g * 6 + u; if (sl < m) { int t = ptok[(size_t)pi * MAXM + sl]; rows[u] = X + (size_t)t * H; xsr[u] = XS + (size_t)t * (H / 32); } else { rows[u] = ZROW; xsr[u] = ZXS; } }
            gmn_pack6(B->xp + (size_t)g * 6 * H, B->xsp + (size_t)g * 6 * (H / 32), rows, xsr, H);
        }
        for (int sl = 0; sl < mp; ++sl) { if (sl < m) memcpy(B->Bg + (size_t)sl * (H / 32), BT + (size_t)ptok[(size_t)pi * MAXM + sl] * (H / 32), (H / 32) * 4); else memset(B->Bg + (size_t)sl * (H / 32), 0, (H / 32) * 4); }
        gmn_gemm(GMN_TYPE_Q4K, WGU + (size_t)pi * gu_part, gu_rb, H, GU_ROWS, mp, B->xp, B->xsp, B->Bg, B->ygu, LDY_GU, B->cbuf, B->mina, 0);
        for (int s = 0; s < m; ++s) for (int r = 0; r < GU_ROWS; ++r) { double d = B->ygu[(size_t)s * LDY_GU + r] - gu_ref[(size_t)s * LDY_GU + r]; se_gu += d * d; sr_gu += (double)gu_ref[(size_t)s * LDY_GU + r] * gu_ref[(size_t)s * LDY_GU + r]; }
        /* down: reference from the KERNEL's own quantized activation (isolates the down expansion) */
        float *Bd = al((size_t)mp * 8 * 4);
        gmn_swiglu_quant(B->ygu, LDY_GU, INTER, m, mp, B->a8, B->as, Bd);
        for (int g = 0; g < mp / 6; ++g) {
            const int8_t *rows[6]; const float *xsr[6];
            for (int u = 0; u < 6; ++u) { rows[u] = B->a8 + (size_t)(g * 6 + u) * INTER; xsr[u] = B->as + (size_t)(g * 6 + u) * 8; }
            gmn_pack6(B->xp2 + (size_t)g * 6 * INTER, B->asp + (size_t)g * 6 * 8, rows, xsr, INTER);
        }
        float *yd = al((size_t)mp * LDY_DN * 4);
        gmn_gemm(GMN_TYPE_Q5K, WDN + (size_t)pi * dn_part, dn_rb, INTER, H, mp, B->xp2, B->asp, Bd, yd, LDY_DN, B->cbuf, B->mina, 0);
        for (int s = 0; s < m; ++s) {
            float ad[INTER];
            for (int k = 0; k < INTER; ++k) ad[k] = (float)B->a8[(size_t)s * INTER + k] * B->as[(size_t)s * 8 + k / 32];
            for (int r = 0; r < H; ++r) { double a = 0; for (int k = 0; k < INTER; ++k) a += (double)Wd[(size_t)r * INTER + k] * ad[k]; double d = yd[(size_t)s * LDY_DN + r] - a; se += d * d; sr += a * a; }
        }
        double rg = sqrt(se_gu / (sr_gu + 1e-30)), rd = sqrt(se / (sr + 1e-30));
        int ok = rg < 2e-4 && rd < 2e-4; ok_all &= ok;
        printf("correctness part %d (M=%d): gate/up Q4_K rel L2 %.3e | down Q5_K rel L2 %.3e %s\n", pi, m, rg, rd, ok ? "PASS" : "FAIL");
    }
    printf("%s\n", ok_all ? "ALL PASS" : "SOME FAIL");
    return ok_all ? 0 : 1;
}
