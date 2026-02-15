/*
 * L1D Set Partitioning — Multi-Stream Way Conflict Benchmark
 *
 * Kernel: acc[j] += src[0][k*N+j] + src[1][k*N+j] + ... + src[n-1][k*N+j]
 *
 *   acc: N_ACC doubles (16KB = 1 way = 64 CL over all 64 sets)
 *        persistent across K_ITER iterations (temporal reuse, read-modify-write)
 *   src[s]: K_ITER * N_ACC doubles each (streaming, one row per k)
 *
 * At each j position all streams hit the SAME cache set (when ALIGNED).
 * With n_src read streams + 1 acc (rw) = (n_src+1) total concurrent ways:
 *
 *   n_src=3 → 4 ways needed → exact L1 fit, 0 spare
 *   n_src=5 → 6 ways needed → exceeds 4-way by 2 → acc thrashing
 *   n_src=7 → 8 ways needed → exceeds 4-way by 4 → severe thrashing
 *
 * STAGGER separates each stream into its own set → 1 way/set → no conflict.
 *
 * L1D: 64KB, 4-way, 256B line, 64 sets. Way = 16KB.
 * Set index = addr[13:8].
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#ifdef USE_FAPP
#include "fj_tool/fapp.h"
#endif

/* L1D cache parameters */
#define L1_SIZE     (64 * 1024)
#define L1_WAYS     4
#define L1_LINE     256
#define L1_SETS     64
#define WAY_SIZE    (16 * 1024)     /* 16KB = L1_SIZE / L1_WAYS */

/* Multi-stream benchmark parameters */
#define N_ACC       2048    /* acc doubles = 16KB = WAY_SIZE = 64 CL */
#define K_ITER      64      /* outer iterations (acc reuse count) */
#define MAX_SRC     7       /* maximum read streams */
#define NUM_ITERS   100     /* timed repetitions */
#define WARMUP_ITERS 3

/* Offset configurations */
typedef struct {
    const char *name;
    int stagger_cl;
} offset_config_t;

static const offset_config_t configs[] = {
    { "ALIGNED",     0 },   /* all streams same set → max conflict */
    { "STAGGER_1",   1 },   /* consecutive sets */
    { "STAGGER_16", 16 },   /* gcd(16,64)=16 → only 4 unique sets */
    { "STAGGER_21", 21 },   /* gcd(21,64)=1  → all unique sets */
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))

/* Read-stream counts to test */
static const int stream_counts[] = { 3, 5, 7 };
#define NUM_SCOUNTS (sizeof(stream_counts) / sizeof(stream_counts[0]))

/* ------------------------------------------------------------------ */
/*  Utility functions                                                  */
/* ------------------------------------------------------------------ */

static inline uint64_t read_cycle(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_freq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

static inline void memory_fence(void) {
    __asm__ volatile("dmb ish" ::: "memory");
}

static void flush_cache(void *ptr, size_t size) {
    char *p = (char *)ptr;
    for (size_t i = 0; i < size; i += L1_LINE) {
        __asm__ volatile("dc civac, %0" :: "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
}

/* ------------------------------------------------------------------ */
/*  Kernels                                                            */
/* ------------------------------------------------------------------ */

/*
 * SVE multi-stream accumulate.
 * Explicit svld1/svst1 forces acc load+store every k step,
 * creating the maximum way pressure at each set.
 */
static void __attribute__((noinline))
kernel_sve(double *acc, double **src, int n_src, int n_acc, int k_iter)
{
    svbool_t pg = svptrue_b64();
    int vl = (int)svcntd();
    for (int k = 0; k < k_iter; k++) {
        size_t off = (size_t)k * n_acc;
        for (int j = 0; j < n_acc; j += vl) {
            svfloat64_t va = svld1_f64(pg, &acc[j]);
            for (int s = 0; s < n_src; s++)
                va = svadd_f64_x(pg, va,
                     svld1_f64(pg, &src[s][off + j]));
            svst1_f64(pg, &acc[j], va);
        }
    }
}

static void __attribute__((noinline))
kernel_scalar(double *acc, double **src, int n_src, int n_acc, int k_iter)
{
    for (int k = 0; k < k_iter; k++) {
        size_t off = (size_t)k * n_acc;
        for (int j = 0; j < n_acc; j++) {
            double v = acc[j];
            for (int s = 0; s < n_src; s++)
                v += src[s][off + j];
            acc[j] = v;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Diagnostics                                                        */
/* ------------------------------------------------------------------ */

static void print_ptr_set(const char *label, const void *ptr) {
    uint64_t addr = (uint64_t)ptr;
    printf("  %-4s: %p  set=%2d  way_off=0x%04lx\n",
           label, ptr,
           (int)((addr >> 8) & 0x3F),
           (unsigned long)(addr & (WAY_SIZE - 1)));
}

/* ------------------------------------------------------------------ */
/*  Benchmark runner                                                   */
/* ------------------------------------------------------------------ */

typedef void (*multi_fn_t)(double *, double **, int, int, int);

static void run_benchmark(const char *cfg_name, int n_src,
                          double *acc, double **src,
                          multi_fn_t kernel, const char *kname)
{
    size_t acc_bytes = (size_t)N_ACC * sizeof(double);
    size_t src_bytes = (size_t)K_ITER * N_ACC * sizeof(double);
    uint64_t freq = get_freq();

    /* Zero acc, flush all arrays */
    memset(acc, 0, acc_bytes);
    flush_cache(acc, acc_bytes);
    for (int s = 0; s < n_src; s++)
        flush_cache(src[s], src_bytes);
    memory_fence();

    /* Warmup */
    for (int w = 0; w < WARMUP_ITERS; w++)
        kernel(acc, src, n_src, N_ACC, K_ITER);
    memory_fence();

    /* Re-zero acc, re-flush for clean timed run */
    memset(acc, 0, acc_bytes);
    flush_cache(acc, acc_bytes);
    for (int s = 0; s < n_src; s++)
        flush_cache(src[s], src_bytes);
    memory_fence();

#ifdef USE_FAPP
    char region[64];
    snprintf(region, sizeof(region), "%s_%dsrc_%s", cfg_name, n_src, kname);
    fapp_start(region, 1, 0);
#endif

    uint64_t t0 = read_cycle();
    for (int it = 0; it < NUM_ITERS; it++)
        kernel(acc, src, n_src, N_ACC, K_ITER);
    uint64_t t1 = read_cycle();
    memory_fence();

#ifdef USE_FAPP
    fapp_stop(region, 1, 0);
#endif

    uint64_t cycles = t1 - t0;
    /* n_src adds per acc element per k step */
    double flops = (double)n_src * N_ACC * K_ITER * NUM_ITERS;
    double seconds = (double)cycles / (double)freq;
    double gflops = flops / seconds / 1e9;
    /* cycles per acc-element update (one k step) */
    double cyc_per_upd = (double)cycles / ((double)N_ACC * K_ITER * NUM_ITERS);

    printf("    %-8s %12lu cyc  %7.2f GFLOPS  %5.2f cyc/upd\n",
           kname, (unsigned long)cycles, gflops, cyc_per_upd);
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    printf("=== L1D Set Partitioning — Multi-Stream Way Conflict ===\n\n");

    uint64_t freq = get_freq();
    size_t acc_bytes = (size_t)N_ACC * sizeof(double);   /* 16KB */
    size_t src_bytes = (size_t)K_ITER * N_ACC * sizeof(double); /* 1MB */

    printf("Timer freq  : %lu Hz\n", (unsigned long)freq);
    printf("L1D         : %dKB, %d-way, %dB line, %d sets\n",
           L1_SIZE / 1024, L1_WAYS, L1_LINE, L1_SETS);
    printf("Way size    : %dKB\n", WAY_SIZE / 1024);
    printf("SVE VL      : %lu bits (%lu fp64/vec)\n",
           (unsigned long)(svcntd() * 64), (unsigned long)svcntd());
    printf("\nacc         : %d doubles = %luKB = 1 way = %d CL (all %d sets)\n",
           N_ACC, (unsigned long)(acc_bytes / 1024),
           (int)(acc_bytes / L1_LINE), L1_SETS);
    printf("Each src    : %d rows x %d cols = %luKB\n",
           K_ITER, N_ACC, (unsigned long)(src_bytes / 1024));
    printf("acc reuse   : %d (across k)\n", K_ITER);
    printf("Timed iters : %d\n\n", NUM_ITERS);

    /* Allocate buffer for worst case: acc + MAX_SRC srcs + stagger */
    int max_stagger = configs[NUM_CONFIGS - 1].stagger_cl;
    size_t buf_size = acc_bytes + (size_t)MAX_SRC * src_bytes
                    + (size_t)(MAX_SRC + 1) * max_stagger * L1_LINE
                    + WAY_SIZE;

    char *buf = NULL;
    if (posix_memalign((void **)&buf, WAY_SIZE, buf_size) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }

    /* Initialize with small values */
    for (size_t i = 0; i < buf_size / sizeof(double); i++)
        ((double *)buf)[i] = ((double)((int)(i % 997) - 498)) / 997.0;

    for (size_t sc = 0; sc < NUM_SCOUNTS; sc++) {
        int n_src = stream_counts[sc];
        int total = n_src + 1;

#ifdef USE_FAPP
        /* fapp: only profile the 8-stream (7+1) case */
        if (n_src != 7) continue;
#endif

        printf("============================================================\n");
        printf("%d read + 1 acc(rw) = %d total streams\n", n_src, total);
        if (total <= L1_WAYS)
            printf("Ways needed: %d = L1 ways (%d) → exact fit, 0 spare\n",
                   total, L1_WAYS);
        else
            printf("Ways needed: %d > L1 ways (%d) → EXCEEDS by %d\n",
                   total, L1_WAYS, total - L1_WAYS);
        printf("============================================================\n\n");

        for (size_t ci = 0; ci < NUM_CONFIGS; ci++) {
            int S = configs[ci].stagger_cl;
            size_t sb = (size_t)S * L1_LINE;

#ifdef USE_FAPP
            /* fapp: only ALIGNED vs STAGGER_21 */
            if (S != 0 && S != 21) continue;
#endif

            /*
             * Layout (acc_bytes and src_bytes are multiples of WAY_SIZE):
             *   acc    = buf + 0
             *   src[s] = buf + acc_bytes + s*src_bytes + (s+1)*sb
             *
             * ALIGNED (sb=0): all start at set 0 → full conflict.
             * STAGGER_21:     stream s starts at set (s+1)*21 mod 64 → unique.
             */
            double *acc = (double *)buf;
            double *src_ptrs[MAX_SRC];
            for (int s = 0; s < n_src; s++)
                src_ptrs[s] = (double *)(buf + acc_bytes
                              + (size_t)s * src_bytes
                              + (size_t)(s + 1) * sb);

            printf("--- %s (stagger=%d CL, %lu B) ---\n",
                   configs[ci].name, S, (unsigned long)sb);
            print_ptr_set("acc", acc);
            for (int s = 0; s < n_src; s++) {
                char lbl[8];
                snprintf(lbl, sizeof(lbl), "s%d", s);
                print_ptr_set(lbl, src_ptrs[s]);
            }

            run_benchmark(configs[ci].name, n_src, acc, src_ptrs,
                          kernel_sve, "sve");
#ifndef USE_FAPP
            run_benchmark(configs[ci].name, n_src, acc, src_ptrs,
                          kernel_scalar, "scalar");
#endif
            printf("\n");
        }
    }

    free(buf);
    printf("Done.\n");
    return 0;
}
