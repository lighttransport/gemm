/* mv_bench.c - single-core F16 matvec microbench for A64FX.
 *
 * Goal: optimize W*x where W is [N][K] fp16, x is [K] fp16, dst is [N] fp32.
 * This is memory-bound (~1 FLOP/byte). Established facts from prior runs:
 *   - single-core BW ceiling (KBW, 4 indep accumulators) ~= 41 GB/s on 64KiB pages
 *   - naive row-major + svaddv horizontal reduce  ~= 7 GB/s
 *   - panel layout Wp[N/VL][K][VL], 4 k-accumulators, NO horizontal op ~= 21 GB/s
 *
 * Open question: K7 (panel, 4 acc) hits 21 GB/s vs 41 GB/s BW ceiling.
 * Hypothesis: insufficient ILP - svmla_f16 latency ~9c, 4 accumulators can't
 * hide it. Test panel kernel with 4/8/12/16 k-accumulators here.
 *
 * Panel layout: Wp[blk][k][lane] = W[blk*VL+lane][k], VL = svcnth() = 32.
 * Each SVE lane accumulates a different output row -> result store is a plain
 * svst1 of widened fp16->fp32, no svaddv anywhere.
 *
 * Build (normal 64KiB pages):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *       -o mv_bench mv_bench.c -lm
 * Build (2MiB large pages via libmpg, W goes through malloc >=128MiB):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *       -o mv_bench_lp mv_bench.c \
 *       -Wl,-T/opt/FJSVxos/mmm/util/bss-2mb.lds \
 *       -L/opt/FJSVxos/mmm/lib64 -lmpg -lc -lpthread -no-pie -lm
 * Run: OMP_NUM_THREADS=1 ./mv_bench
 */
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static double mono_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* W buffer goes through plain malloc so libmpg can intercept it
 * (>=128MiB allocations -> mmap with 2MiB pages when linked with -lmpg). */
static void *walloc(size_t bytes) {
    void *p = malloc(bytes);
    if (p) memset(p, 0, bytes);
    return p;
}

/* ---- panel kernels: single sequential stream over Wp, N k-accumulators ----
 * SVE types are sizeless (no arrays), so accumulators are explicit. The
 * MLA/MA/ST macros keep the bodies compact. */

#define MLA(i) a##i = svmla_x(ph, a##i, \
    svld1_f16(ph, w + (size_t)(k + i) * vlh), svdup_f16(xh[k + i]))
#define STORE_PANEL(s)                                                         \
    do {                                                                       \
        svuint16_t u = svreinterpret_u16(s);                                    \
        svst1_f32(pg, &dst[blk],                                                \
            svcvt_f32_f16_x(pg, svreinterpret_f16(svunpklo_u32(u))));            \
        svst1_f32(pg, &dst[blk + vlh / 2],                                       \
            svcvt_f32_f16_x(pg, svreinterpret_f16(svunpkhi_u32(u))));            \
    } while (0)
#define TAIL                                                                   \
    for (; k < K; k++)                                                          \
        a0 = svmla_x(ph, a0, svld1_f16(ph, w + (size_t)k * vlh), svdup_f16(xh[k]))

static void panel4(float *dst, const float16_t *wp, const float16_t *xh,
                   int N, int K) {
    svbool_t ph = svptrue_b16(), pg = svptrue_b32();
    int vlh = (int)svcnth();
    for (int blk = 0; blk + vlh - 1 < N; blk += vlh) {
        const float16_t *w = wp + (size_t)blk * K;
        svfloat16_t a0=svdup_f16(0),a1=svdup_f16(0),a2=svdup_f16(0),a3=svdup_f16(0);
        int k = 0;
        for (; k + 3 < K; k += 4) { MLA(0);MLA(1);MLA(2);MLA(3); }
        TAIL;
        STORE_PANEL(svadd_x(ph, svadd_x(ph,a0,a1), svadd_x(ph,a2,a3)));
    }
}

static void panel8(float *dst, const float16_t *wp, const float16_t *xh,
                   int N, int K) {
    svbool_t ph = svptrue_b16(), pg = svptrue_b32();
    int vlh = (int)svcnth();
    for (int blk = 0; blk + vlh - 1 < N; blk += vlh) {
        const float16_t *w = wp + (size_t)blk * K;
        svfloat16_t a0=svdup_f16(0),a1=svdup_f16(0),a2=svdup_f16(0),a3=svdup_f16(0);
        svfloat16_t a4=svdup_f16(0),a5=svdup_f16(0),a6=svdup_f16(0),a7=svdup_f16(0);
        int k = 0;
        for (; k + 7 < K; k += 8) {
            MLA(0);MLA(1);MLA(2);MLA(3);MLA(4);MLA(5);MLA(6);MLA(7);
        }
        TAIL;
        svfloat16_t s = svadd_x(ph, svadd_x(ph,svadd_x(ph,a0,a1),svadd_x(ph,a2,a3)),
                                    svadd_x(ph,svadd_x(ph,a4,a5),svadd_x(ph,a6,a7)));
        STORE_PANEL(s);
    }
}

static void panel12(float *dst, const float16_t *wp, const float16_t *xh,
                    int N, int K) {
    svbool_t ph = svptrue_b16(), pg = svptrue_b32();
    int vlh = (int)svcnth();
    for (int blk = 0; blk + vlh - 1 < N; blk += vlh) {
        const float16_t *w = wp + (size_t)blk * K;
        svfloat16_t a0=svdup_f16(0),a1=svdup_f16(0),a2=svdup_f16(0),a3=svdup_f16(0);
        svfloat16_t a4=svdup_f16(0),a5=svdup_f16(0),a6=svdup_f16(0),a7=svdup_f16(0);
        svfloat16_t a8=svdup_f16(0),a9=svdup_f16(0),a10=svdup_f16(0),a11=svdup_f16(0);
        int k = 0;
        for (; k + 11 < K; k += 12) {
            MLA(0);MLA(1);MLA(2);MLA(3);MLA(4);MLA(5);
            MLA(6);MLA(7);MLA(8);MLA(9);MLA(10);MLA(11);
        }
        TAIL;
        svfloat16_t s = svadd_x(ph,
            svadd_x(ph, svadd_x(ph,svadd_x(ph,a0,a1),svadd_x(ph,a2,a3)),
                        svadd_x(ph,svadd_x(ph,a4,a5),svadd_x(ph,a6,a7))),
            svadd_x(ph, svadd_x(ph,a8,a9), svadd_x(ph,a10,a11)));
        STORE_PANEL(s);
    }
}

static void panel16(float *dst, const float16_t *wp, const float16_t *xh,
                    int N, int K) {
    svbool_t ph = svptrue_b16(), pg = svptrue_b32();
    int vlh = (int)svcnth();
    for (int blk = 0; blk + vlh - 1 < N; blk += vlh) {
        const float16_t *w = wp + (size_t)blk * K;
        svfloat16_t a0=svdup_f16(0),a1=svdup_f16(0),a2=svdup_f16(0),a3=svdup_f16(0);
        svfloat16_t a4=svdup_f16(0),a5=svdup_f16(0),a6=svdup_f16(0),a7=svdup_f16(0);
        svfloat16_t a8=svdup_f16(0),a9=svdup_f16(0),a10=svdup_f16(0),a11=svdup_f16(0);
        svfloat16_t a12=svdup_f16(0),a13=svdup_f16(0),a14=svdup_f16(0),a15=svdup_f16(0);
        int k = 0;
        for (; k + 15 < K; k += 16) {
            MLA(0);MLA(1);MLA(2);MLA(3);MLA(4);MLA(5);MLA(6);MLA(7);
            MLA(8);MLA(9);MLA(10);MLA(11);MLA(12);MLA(13);MLA(14);MLA(15);
        }
        TAIL;
        svfloat16_t s = svadd_x(ph,
            svadd_x(ph, svadd_x(ph,svadd_x(ph,a0,a1),svadd_x(ph,a2,a3)),
                        svadd_x(ph,svadd_x(ph,a4,a5),svadd_x(ph,a6,a7))),
            svadd_x(ph, svadd_x(ph,svadd_x(ph,a8,a9),svadd_x(ph,a10,a11)),
                        svadd_x(ph,svadd_x(ph,a12,a13),svadd_x(ph,a14,a15))));
        STORE_PANEL(s);
    }
}

/* threaded panel8: parallelize the block loop (each thread = disjoint
 * Wp sub-stream, still sequential within a thread). */
static void panel8_mt(float *dst, const float16_t *wp, const float16_t *xh,
                      int N, int K) {
    int vlh = (int)svcnth();
    int nblk = N / vlh;
#pragma omp parallel for schedule(static)
    for (int b = 0; b < nblk; b++) {
        svbool_t ph = svptrue_b16(), pg = svptrue_b32();
        int blk = b * vlh;
        const float16_t *w = wp + (size_t)blk * K;
        svfloat16_t a0=svdup_f16(0),a1=svdup_f16(0),a2=svdup_f16(0),a3=svdup_f16(0);
        svfloat16_t a4=svdup_f16(0),a5=svdup_f16(0),a6=svdup_f16(0),a7=svdup_f16(0);
        int k = 0;
        for (; k + 7 < K; k += 8) {
            MLA(0);MLA(1);MLA(2);MLA(3);MLA(4);MLA(5);MLA(6);MLA(7);
        }
        TAIL;
        svfloat16_t s = svadd_x(ph, svadd_x(ph,svadd_x(ph,a0,a1),svadd_x(ph,a2,a3)),
                                    svadd_x(ph,svadd_x(ph,a4,a5),svadd_x(ph,a6,a7)));
        STORE_PANEL(s);
    }
}

/* pure bandwidth probe: just sum the whole Wp stream, 8 indep accumulators */
static float bw_probe(const float16_t *wp, size_t nelem) {
    svbool_t ph = svptrue_b16();
    int vlh = (int)svcnth();
    svfloat16_t a0=svdup_f16(0),a1=svdup_f16(0),a2=svdup_f16(0),a3=svdup_f16(0);
    svfloat16_t a4=svdup_f16(0),a5=svdup_f16(0),a6=svdup_f16(0),a7=svdup_f16(0);
    size_t i = 0;
    for (; i + 8 * (size_t)vlh <= nelem; i += 8 * (size_t)vlh) {
        a0 = svadd_x(ph, a0, svld1_f16(ph, wp + i + 0 * vlh));
        a1 = svadd_x(ph, a1, svld1_f16(ph, wp + i + 1 * vlh));
        a2 = svadd_x(ph, a2, svld1_f16(ph, wp + i + 2 * vlh));
        a3 = svadd_x(ph, a3, svld1_f16(ph, wp + i + 3 * vlh));
        a4 = svadd_x(ph, a4, svld1_f16(ph, wp + i + 4 * vlh));
        a5 = svadd_x(ph, a5, svld1_f16(ph, wp + i + 5 * vlh));
        a6 = svadd_x(ph, a6, svld1_f16(ph, wp + i + 6 * vlh));
        a7 = svadd_x(ph, a7, svld1_f16(ph, wp + i + 7 * vlh));
    }
    svfloat16_t s = svadd_x(ph, svadd_x(ph, svadd_x(ph,a0,a1), svadd_x(ph,a2,a3)),
                                svadd_x(ph, svadd_x(ph,a4,a5), svadd_x(ph,a6,a7)));
    return svaddv_f32(svptrue_b32(),
        svcvt_f32_f16_x(svptrue_b32(),
            svreinterpret_f16(svunpklo_u32(svreinterpret_u16(s)))));
}

typedef void (*kfn)(float *, const float16_t *, const float16_t *, int, int);

int main(int argc, char **argv) {
    int N = 248320, K = 1024;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    int vlh = (int)svcnth();
    N = (N / vlh) * vlh;  /* whole panels */

    size_t welem = (size_t)N * K;
    size_t wbytes = welem * sizeof(float16_t);
    printf("N=%d K=%d  VL(fp16)=%d  W=%.1f MB\n", N, K, vlh, wbytes / 1048576.0);

    float16_t *wp = (float16_t *)walloc(wbytes);
    float16_t *xh = (float16_t *)walloc((size_t)K * sizeof(float16_t));
    float     *dst = (float *)walloc((size_t)N * sizeof(float));
    if (!wp || !xh || !dst) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (int k = 0; k < K; k++) xh[k] = (float16_t)(0.5f + 0.001f * k);
    /* fill Wp; parallel so first-touch matches panel8_mt's static block
     * schedule -> each CMG owns the pages its threads will read (NUMA). */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < welem; i++)
        wp[i] = (float16_t)(((i * 1103515245u + 12345u) >> 16 & 0xff) * 0.01f);

    struct { const char *name; kfn fn; } ks[] = {
        {"panel4  (4 k-acc)",  panel4},
        {"panel8  (8 k-acc)",  panel8},
        {"panel12 (12 k-acc)", panel12},
        {"panel16 (16 k-acc)", panel16},
    };
    int nk = sizeof(ks) / sizeof(ks[0]);

    const int reps = 5;
    for (int t = 0; t < nk; t++) {
        ks[t].fn(dst, wp, xh, N, K);  /* warmup */
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = mono_sec();
            ks[t].fn(dst, wp, xh, N, K);
            double dt = mono_sec() - t0;
            if (dt < best) best = dt;
        }
        double gbs = wbytes / best / 1e9;
        double gflops = 2.0 * welem / best / 1e9;
        printf("  %-22s  %.3f ms  %6.2f GB/s  %6.2f GFLOP/s  dst[0]=%.3f\n",
               ks[t].name, best * 1e3, gbs, gflops, dst[0]);
    }

    /* multi-thread scaling of panel8 (set OMP_NUM_THREADS / OMP_PROC_BIND) */
    {
        int nt = omp_get_max_threads();
        panel8_mt(dst, wp, xh, N, K);  /* warmup */
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = mono_sec();
            panel8_mt(dst, wp, xh, N, K);
            double dt = mono_sec() - t0;
            if (dt < best) best = dt;
        }
        printf("  %-18s nt=%-3d  %.3f ms  %6.2f GB/s  %7.2f GFLOP/s  dst[0]=%.3f\n",
               "panel8_mt", nt, best * 1e3, wbytes / best / 1e9,
               2.0 * welem / best / 1e9, dst[0]);
    }

    /* bandwidth ceiling */
    {
        volatile float sink = bw_probe(wp, welem);  /* warmup */
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = mono_sec();
            sink = bw_probe(wp, welem);
            double dt = mono_sec() - t0;
            if (dt < best) best = dt;
        }
        (void)sink;
        printf("  %-22s  %.3f ms  %6.2f GB/s  (pure load ceiling)\n",
               "bw_probe", best * 1e3, wbytes / best / 1e9);
    }
    return 0;
}
