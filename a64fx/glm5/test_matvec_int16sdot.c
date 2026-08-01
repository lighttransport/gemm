#define _GNU_SOURCE
/*
 * Honest large-working-set benchmark for glm5_matvec_int16sdot_8row.
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *       -I../../common test_matvec_int16sdot.c -lm -o test_matvec_int16sdot
 * Run:
 *   OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=47 \
 *       ./test_matvec_int16sdot
 * Profile one region:
 *   fcc ... -DGLM5_MATVEC_FAPP test_matvec_int16sdot.c -lm -o test_matvec_int16sdot_fapp
 *   MODE=baseline REPS=1 fapp -C -d prof -Hevent_raw=... ./test_matvec_int16sdot_fapp
 *
 * The default 2 GiB weight allocation is intentional: smaller/best-of-N tests can
 * report L2/page-table warmth rather than HBM behavior. Every output is stored and
 * included in g_sink to prevent dead-code elimination.
 */
#include <arm_sve.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "glm5_int8.h"

#ifdef GLM5_MATVEC_FAPP
extern void fapp_start(const char *, int, int);
extern void fapp_stop(const char *, int, int);
#else
static inline void fapp_start(const char *name, int number, int level) {
    (void)name; (void)number; (void)level;
}
static inline void fapp_stop(const char *name, int number, int level) {
    (void)name; (void)number; (void)level;
}
#endif

enum { BENCH_COLS = 6144, BENCH_ROWS = 2048, BENCH_GS = 64 };
static volatile double g_sink;

typedef void (*kernel_fn)(float *,
        const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *,
        const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *,
        const float *, const float *, const float *, const float *,
        const float *, const float *, const float *, const float *,
        int, int, const int16_t *, const int64_t *, int);

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

static size_t env_size(const char *name, size_t def) {
    const char *s = getenv(name);
    if (!s || !*s) return def;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 0);
    return errno || end == s || *end ? def : (size_t)v;
}

static int env_int(const char *name, int def) {
    const char *s = getenv(name);
    if (!s || !*s) return def;
    char *end = NULL;
    long v = strtol(s, &end, 0);
    return end == s || *end ? def : (int)v;
}

static uint32_t lcg32(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

__attribute__((noinline))
static void kernel_baseline(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const float *s0, const float *s1, const float *s2, const float *s3,
        const float *s4, const float *s5, const float *s6, const float *s7,
        int gs, int qg0, const int16_t *xq, const int64_t *xgsum, int cols) {
    glm5_matvec_int16sdot_8row(dst, w0, w1, w2, w3, w4, w5, w6, w7,
            s0, s1, s2, s3, s4, s5, s6, s7, gs, qg0, xq, xgsum, cols);
}

#define CANDIDATE_ARGS float *restrict dst, \
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3, \
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7, \
        const float *s0, const float *s1, const float *s2, const float *s3, \
        const float *s4, const float *s5, const float *s6, const float *s7, \
        int gs, int qg0, const int16_t *xq, const int64_t *restrict xgsum, int cols

#define LOAD_DOT_2() do { \
    svint16_t v0 = svreinterpret_s16_u16(svld1ub_u16(pg, w0 + c)); \
    svint16_t v1 = svreinterpret_s16_u16(svld1ub_u16(pg, w1 + c)); \
    d0 = svdot_s64(d0, v0, xv); d1 = svdot_s64(d1, v1, xv); \
    __asm__ volatile("" ::: "memory"); \
    svint16_t v2 = svreinterpret_s16_u16(svld1ub_u16(pg, w2 + c)); \
    svint16_t v3 = svreinterpret_s16_u16(svld1ub_u16(pg, w3 + c)); \
    d2 = svdot_s64(d2, v2, xv); d3 = svdot_s64(d3, v3, xv); \
    __asm__ volatile("" ::: "memory"); \
    svint16_t v4 = svreinterpret_s16_u16(svld1ub_u16(pg, w4 + c)); \
    svint16_t v5 = svreinterpret_s16_u16(svld1ub_u16(pg, w5 + c)); \
    d4 = svdot_s64(d4, v4, xv); d5 = svdot_s64(d5, v5, xv); \
    __asm__ volatile("" ::: "memory"); \
    svint16_t v6 = svreinterpret_s16_u16(svld1ub_u16(pg, w6 + c)); \
    svint16_t v7 = svreinterpret_s16_u16(svld1ub_u16(pg, w7 + c)); \
    d6 = svdot_s64(d6, v6, xv); d7 = svdot_s64(d7, v7, xv); \
} while (0)

#define LOAD_DOT_4() do { \
    svint16_t v0 = svreinterpret_s16_u16(svld1ub_u16(pg, w0 + c)); \
    svint16_t v1 = svreinterpret_s16_u16(svld1ub_u16(pg, w1 + c)); \
    svint16_t v2 = svreinterpret_s16_u16(svld1ub_u16(pg, w2 + c)); \
    svint16_t v3 = svreinterpret_s16_u16(svld1ub_u16(pg, w3 + c)); \
    d0 = svdot_s64(d0, v0, xv); d1 = svdot_s64(d1, v1, xv); \
    d2 = svdot_s64(d2, v2, xv); d3 = svdot_s64(d3, v3, xv); \
    __asm__ volatile("" ::: "memory"); \
    svint16_t v4 = svreinterpret_s16_u16(svld1ub_u16(pg, w4 + c)); \
    svint16_t v5 = svreinterpret_s16_u16(svld1ub_u16(pg, w5 + c)); \
    svint16_t v6 = svreinterpret_s16_u16(svld1ub_u16(pg, w6 + c)); \
    svint16_t v7 = svreinterpret_s16_u16(svld1ub_u16(pg, w7 + c)); \
    d4 = svdot_s64(d4, v4, xv); d5 = svdot_s64(d5, v5, xv); \
    d6 = svdot_s64(d6, v6, xv); d7 = svdot_s64(d7, v7, xv); \
} while (0)

#define LOAD_DOT_8() do { \
    svint16_t v0 = svreinterpret_s16_u16(svld1ub_u16(pg, w0 + c)); \
    svint16_t v1 = svreinterpret_s16_u16(svld1ub_u16(pg, w1 + c)); \
    svint16_t v2 = svreinterpret_s16_u16(svld1ub_u16(pg, w2 + c)); \
    svint16_t v3 = svreinterpret_s16_u16(svld1ub_u16(pg, w3 + c)); \
    svint16_t v4 = svreinterpret_s16_u16(svld1ub_u16(pg, w4 + c)); \
    svint16_t v5 = svreinterpret_s16_u16(svld1ub_u16(pg, w5 + c)); \
    svint16_t v6 = svreinterpret_s16_u16(svld1ub_u16(pg, w6 + c)); \
    svint16_t v7 = svreinterpret_s16_u16(svld1ub_u16(pg, w7 + c)); \
    d0 = svdot_s64(d0, v0, xv); d1 = svdot_s64(d1, v1, xv); \
    d2 = svdot_s64(d2, v2, xv); d3 = svdot_s64(d3, v3, xv); \
    d4 = svdot_s64(d4, v4, xv); d5 = svdot_s64(d5, v5, xv); \
    d6 = svdot_s64(d6, v6, xv); d7 = svdot_s64(d7, v7, xv); \
} while (0)

#define DEFINE_CANDIDATE(NAME, LOAD_DOT) \
__attribute__((noinline)) \
static void NAME(CANDIDATE_ARGS) { \
    if (gs != 64 || (qg0 & 63) != 0 || (cols & 63) != 0) { \
        kernel_baseline(dst, w0, w1, w2, w3, w4, w5, w6, w7, \
                s0, s1, s2, s3, s4, s5, s6, s7, gs, qg0, xq, xgsum, cols); \
        return; \
    } \
    const svbool_t pd = svptrue_b64(); \
    const svbool_t pg = svptrue_b16(); \
    const int vh = (int)svcnth(); \
    svfloat64_t a0 = svdup_f64(0); \
    svfloat64_t a1 = svdup_f64(0), a2 = svdup_f64(0), a3 = svdup_f64(0); \
    svfloat64_t a4 = svdup_f64(0), a5 = svdup_f64(0), a6 = svdup_f64(0), a7 = svdup_f64(0); \
    double bc0 = 0, bc1 = 0, bc2 = 0, bc3 = 0, bc4 = 0, bc5 = 0, bc6 = 0, bc7 = 0; \
    for (int b = 0; b < cols; b += 64) { \
        svint64_t d0 = svdup_s64(0), d1 = svdup_s64(0), d2 = svdup_s64(0), d3 = svdup_s64(0); \
        svint64_t d4 = svdup_s64(0), d5 = svdup_s64(0), d6 = svdup_s64(0), d7 = svdup_s64(0); \
        for (int c = b; c < b + 64; c += vh) { \
            svint16_t xv = svld1_s16(pg, xq + c); \
            LOAD_DOT(); \
        } \
        const int blk = (qg0 + b) >> 6; \
        double sc0 = s0[blk], sc1 = s1[blk], sc2 = s2[blk], sc3 = s3[blk]; \
        double sc4 = s4[blk], sc5 = s5[blk], sc6 = s6[blk], sc7 = s7[blk]; \
        double xg = (double)xgsum[blk]; \
        a0 = svmla_n_f64_x(pd, a0, svcvt_f64_s64_x(pd, d0), sc0); \
        a1 = svmla_n_f64_x(pd, a1, svcvt_f64_s64_x(pd, d1), sc1); \
        a2 = svmla_n_f64_x(pd, a2, svcvt_f64_s64_x(pd, d2), sc2); \
        a3 = svmla_n_f64_x(pd, a3, svcvt_f64_s64_x(pd, d3), sc3); \
        a4 = svmla_n_f64_x(pd, a4, svcvt_f64_s64_x(pd, d4), sc4); \
        a5 = svmla_n_f64_x(pd, a5, svcvt_f64_s64_x(pd, d5), sc5); \
        a6 = svmla_n_f64_x(pd, a6, svcvt_f64_s64_x(pd, d6), sc6); \
        a7 = svmla_n_f64_x(pd, a7, svcvt_f64_s64_x(pd, d7), sc7); \
        bc0 += sc0 * xg; bc1 += sc1 * xg; bc2 += sc2 * xg; bc3 += sc3 * xg; \
        bc4 += sc4 * xg; bc5 += sc5 * xg; bc6 += sc6 * xg; bc7 += sc7 * xg; \
    } \
    dst[0] = (float)(svaddv_f64(pd, a0) - 128.0 * bc0); \
    dst[1] = (float)(svaddv_f64(pd, a1) - 128.0 * bc1); \
    dst[2] = (float)(svaddv_f64(pd, a2) - 128.0 * bc2); \
    dst[3] = (float)(svaddv_f64(pd, a3) - 128.0 * bc3); \
    dst[4] = (float)(svaddv_f64(pd, a4) - 128.0 * bc4); \
    dst[5] = (float)(svaddv_f64(pd, a5) - 128.0 * bc5); \
    dst[6] = (float)(svaddv_f64(pd, a6) - 128.0 * bc6); \
    dst[7] = (float)(svaddv_f64(pd, a7) - 128.0 * bc7); \
}

static inline __attribute__((always_inline))
void kernel_prefetch_impl(CANDIDATE_ARGS, int distance, int interval) {
    if (gs != 64 || (qg0 & 63) != 0 || (cols & 63) != 0) {
        kernel_baseline(dst, w0, w1, w2, w3, w4, w5, w6, w7,
                s0, s1, s2, s3, s4, s5, s6, s7, gs, qg0, xq, xgsum, cols);
        return;
    }
    svbool_t pf = svptrue_b64();
    int vh = (int)svcnth();
    svfloat64_t a0 = svdup_f64(0), a1 = svdup_f64(0), a2 = svdup_f64(0), a3 = svdup_f64(0);
    svfloat64_t a4 = svdup_f64(0), a5 = svdup_f64(0), a6 = svdup_f64(0), a7 = svdup_f64(0);
    double bc0 = 0, bc1 = 0, bc2 = 0, bc3 = 0, bc4 = 0, bc5 = 0, bc6 = 0, bc7 = 0;
    for (int b = 0; b < cols;) {
        int blk = (qg0 + b) >> 6;
        int bend = ((blk + 1) << 6) - qg0;
        if (bend > cols) bend = cols;
        int c = b;
        /* interval=256 issues one hint per A64FX cache line instead of four
         * duplicate hints at the 64-byte quant-group cadence. */
        if (distance > 0 && (b & (interval - 1)) == 0 && b + distance < cols) {
            __builtin_prefetch(w0 + b + distance); __builtin_prefetch(w1 + b + distance);
            __builtin_prefetch(w2 + b + distance); __builtin_prefetch(w3 + b + distance);
            __builtin_prefetch(w4 + b + distance); __builtin_prefetch(w5 + b + distance);
            __builtin_prefetch(w6 + b + distance); __builtin_prefetch(w7 + b + distance);
        }
        svint64_t d0 = svdup_s64(0), d1 = svdup_s64(0), d2 = svdup_s64(0), d3 = svdup_s64(0);
        svint64_t d4 = svdup_s64(0), d5 = svdup_s64(0), d6 = svdup_s64(0), d7 = svdup_s64(0);
#define FAST_SD16(WP, D) do { \
            svint16_t wv = svreinterpret_s16_u16(svld1ub_u16(pg, &(WP)[c])); \
            D = svdot_s64(D, wv, xv); \
        } while (0)
        for (; c < bend; c += vh) {
            svbool_t pg = svwhilelt_b16((uint32_t)c, (uint32_t)bend);
            svint16_t xv = svld1_s16(pg, &xq[c]);
            FAST_SD16(w0, d0); FAST_SD16(w1, d1); FAST_SD16(w2, d2); FAST_SD16(w3, d3);
            FAST_SD16(w4, d4); FAST_SD16(w5, d5); FAST_SD16(w6, d6); FAST_SD16(w7, d7);
        }
#undef FAST_SD16
        double sc0 = s0[blk], sc1 = s1[blk], sc2 = s2[blk], sc3 = s3[blk];
        double sc4 = s4[blk], sc5 = s5[blk], sc6 = s6[blk], sc7 = s7[blk];
        double xg = (double)xgsum[blk];
        a0 = svmla_n_f64_x(pf, a0, svcvt_f64_s64_x(pf, d0), sc0);
        a1 = svmla_n_f64_x(pf, a1, svcvt_f64_s64_x(pf, d1), sc1);
        a2 = svmla_n_f64_x(pf, a2, svcvt_f64_s64_x(pf, d2), sc2);
        a3 = svmla_n_f64_x(pf, a3, svcvt_f64_s64_x(pf, d3), sc3);
        a4 = svmla_n_f64_x(pf, a4, svcvt_f64_s64_x(pf, d4), sc4);
        a5 = svmla_n_f64_x(pf, a5, svcvt_f64_s64_x(pf, d5), sc5);
        a6 = svmla_n_f64_x(pf, a6, svcvt_f64_s64_x(pf, d6), sc6);
        a7 = svmla_n_f64_x(pf, a7, svcvt_f64_s64_x(pf, d7), sc7);
        bc0 += sc0 * xg; bc1 += sc1 * xg; bc2 += sc2 * xg; bc3 += sc3 * xg;
        bc4 += sc4 * xg; bc5 += sc5 * xg; bc6 += sc6 * xg; bc7 += sc7 * xg;
        b = bend;
    }
    dst[0] = (float)(svaddv_f64(pf, a0) - 128.0 * bc0);
    dst[1] = (float)(svaddv_f64(pf, a1) - 128.0 * bc1);
    dst[2] = (float)(svaddv_f64(pf, a2) - 128.0 * bc2);
    dst[3] = (float)(svaddv_f64(pf, a3) - 128.0 * bc3);
    dst[4] = (float)(svaddv_f64(pf, a4) - 128.0 * bc4);
    dst[5] = (float)(svaddv_f64(pf, a5) - 128.0 * bc5);
    dst[6] = (float)(svaddv_f64(pf, a6) - 128.0 * bc6);
    dst[7] = (float)(svaddv_f64(pf, a7) - 128.0 * bc7);
}

#define PREFETCH_CALL(DIST, INTERVAL) \
    kernel_prefetch_impl(dst, w0, w1, w2, w3, w4, w5, w6, w7, \
            s0, s1, s2, s3, s4, s5, s6, s7, gs, qg0, xq, xgsum, cols, DIST, INTERVAL)
#define DEFINE_PREFETCH_CANDIDATE(NAME, DIST, INTERVAL) \
    __attribute__((noinline)) static void NAME(CANDIDATE_ARGS) { PREFETCH_CALL(DIST, INTERVAL); }

DEFINE_PREFETCH_CANDIDATE(kernel_fast1, 0, 64)
DEFINE_PREFETCH_CANDIDATE(kernel_pf256, 256, 64)
DEFINE_PREFETCH_CANDIDATE(kernel_pf512, 512, 64)
DEFINE_PREFETCH_CANDIDATE(kernel_pf1024, 1024, 64)
DEFINE_PREFETCH_CANDIDATE(kernel_pf2048, 2048, 64)
DEFINE_PREFETCH_CANDIDATE(kernel_pf4096, 4096, 64)
DEFINE_PREFETCH_CANDIDATE(kernel_pf512_i128, 512, 128)
DEFINE_PREFETCH_CANDIDATE(kernel_pf512_i256, 512, 256)
DEFINE_PREFETCH_CANDIDATE(kernel_pf512_i512, 512, 512)
DEFINE_PREFETCH_CANDIDATE(kernel_pf768_i256, 768, 256)

#undef DEFINE_PREFETCH_CANDIDATE
#undef PREFETCH_CALL

/* Four streams at a time: reload the hot activation for rows 4..7 in exchange
 * for halving the live integer-dot state and load-stream fan-out. */
__attribute__((noinline))
static void kernel_4phase(CANDIDATE_ARGS) {
    if (gs != 64 || (qg0 & 63) != 0 || (cols & 63) != 0) {
        kernel_baseline(dst, w0, w1, w2, w3, w4, w5, w6, w7,
                s0, s1, s2, s3, s4, s5, s6, s7, gs, qg0, xq, xgsum, cols);
        return;
    }
    const svbool_t pd = svptrue_b64();
    const svbool_t ph = svptrue_b16();
    svfloat64_t a0 = svdup_f64(0), a1 = svdup_f64(0), a2 = svdup_f64(0), a3 = svdup_f64(0);
    svfloat64_t a4 = svdup_f64(0), a5 = svdup_f64(0), a6 = svdup_f64(0), a7 = svdup_f64(0);
    double bc0 = 0, bc1 = 0, bc2 = 0, bc3 = 0, bc4 = 0, bc5 = 0, bc6 = 0, bc7 = 0;
    for (int b = 0; b < cols; b += 64) {
        if ((b & 255) == 0 && b + 512 < cols) {
            __builtin_prefetch(w0 + b + 512); __builtin_prefetch(w1 + b + 512);
            __builtin_prefetch(w2 + b + 512); __builtin_prefetch(w3 + b + 512);
            __builtin_prefetch(w4 + b + 512); __builtin_prefetch(w5 + b + 512);
            __builtin_prefetch(w6 + b + 512); __builtin_prefetch(w7 + b + 512);
        }
        svint64_t d0 = svdup_s64(0), d1 = svdup_s64(0), d2 = svdup_s64(0), d3 = svdup_s64(0);
        for (int c = b; c < b + 64; c += 32) {
            svint16_t xv = svld1_s16(ph, xq + c);
            d0 = svdot_s64(d0, svreinterpret_s16_u16(svld1ub_u16(ph, w0 + c)), xv);
            d1 = svdot_s64(d1, svreinterpret_s16_u16(svld1ub_u16(ph, w1 + c)), xv);
            d2 = svdot_s64(d2, svreinterpret_s16_u16(svld1ub_u16(ph, w2 + c)), xv);
            d3 = svdot_s64(d3, svreinterpret_s16_u16(svld1ub_u16(ph, w3 + c)), xv);
        }
        const int blk = (qg0 + b) >> 6;
        const double sc0 = s0[blk], sc1 = s1[blk], sc2 = s2[blk], sc3 = s3[blk];
        const double xg = (double)xgsum[blk];
        a0 = svmla_n_f64_x(pd, a0, svcvt_f64_s64_x(pd, d0), sc0);
        a1 = svmla_n_f64_x(pd, a1, svcvt_f64_s64_x(pd, d1), sc1);
        a2 = svmla_n_f64_x(pd, a2, svcvt_f64_s64_x(pd, d2), sc2);
        a3 = svmla_n_f64_x(pd, a3, svcvt_f64_s64_x(pd, d3), sc3);
        bc0 += sc0 * xg; bc1 += sc1 * xg; bc2 += sc2 * xg; bc3 += sc3 * xg;

        d0 = svdup_s64(0); d1 = svdup_s64(0); d2 = svdup_s64(0); d3 = svdup_s64(0);
        for (int c = b; c < b + 64; c += 32) {
            svint16_t xv = svld1_s16(ph, xq + c);
            d0 = svdot_s64(d0, svreinterpret_s16_u16(svld1ub_u16(ph, w4 + c)), xv);
            d1 = svdot_s64(d1, svreinterpret_s16_u16(svld1ub_u16(ph, w5 + c)), xv);
            d2 = svdot_s64(d2, svreinterpret_s16_u16(svld1ub_u16(ph, w6 + c)), xv);
            d3 = svdot_s64(d3, svreinterpret_s16_u16(svld1ub_u16(ph, w7 + c)), xv);
        }
        const double sc4 = s4[blk], sc5 = s5[blk], sc6 = s6[blk], sc7 = s7[blk];
        a4 = svmla_n_f64_x(pd, a4, svcvt_f64_s64_x(pd, d0), sc4);
        a5 = svmla_n_f64_x(pd, a5, svcvt_f64_s64_x(pd, d1), sc5);
        a6 = svmla_n_f64_x(pd, a6, svcvt_f64_s64_x(pd, d2), sc6);
        a7 = svmla_n_f64_x(pd, a7, svcvt_f64_s64_x(pd, d3), sc7);
        bc4 += sc4 * xg; bc5 += sc5 * xg; bc6 += sc6 * xg; bc7 += sc7 * xg;
    }
    dst[0] = (float)(svaddv_f64(pd, a0) - 128.0 * bc0);
    dst[1] = (float)(svaddv_f64(pd, a1) - 128.0 * bc1);
    dst[2] = (float)(svaddv_f64(pd, a2) - 128.0 * bc2);
    dst[3] = (float)(svaddv_f64(pd, a3) - 128.0 * bc3);
    dst[4] = (float)(svaddv_f64(pd, a4) - 128.0 * bc4);
    dst[5] = (float)(svaddv_f64(pd, a5) - 128.0 * bc5);
    dst[6] = (float)(svaddv_f64(pd, a6) - 128.0 * bc6);
    dst[7] = (float)(svaddv_f64(pd, a7) - 128.0 * bc7);
}

/* Two independent column-panel dots per row remove the only repeated SDOT
 * destination within a 64-column group; the final integer add is exact. */
__attribute__((noinline))
static void kernel_panel2(CANDIDATE_ARGS) {
    if (gs != 64 || (qg0 & 63) != 0 || (cols & 63) != 0) {
        kernel_baseline(dst, w0, w1, w2, w3, w4, w5, w6, w7,
                s0, s1, s2, s3, s4, s5, s6, s7, gs, qg0, xq, xgsum, cols);
        return;
    }
    const svbool_t pd = svptrue_b64();
    const svbool_t ph = svptrue_b16();
    svfloat64_t a0 = svdup_f64(0), a1 = svdup_f64(0), a2 = svdup_f64(0), a3 = svdup_f64(0);
    svfloat64_t a4 = svdup_f64(0), a5 = svdup_f64(0), a6 = svdup_f64(0), a7 = svdup_f64(0);
    double bc0 = 0, bc1 = 0, bc2 = 0, bc3 = 0, bc4 = 0, bc5 = 0, bc6 = 0, bc7 = 0;
    for (int b = 0; b < cols; b += 64) {
        if ((b & 255) == 0 && b + 512 < cols) {
            __builtin_prefetch(w0 + b + 512); __builtin_prefetch(w1 + b + 512);
            __builtin_prefetch(w2 + b + 512); __builtin_prefetch(w3 + b + 512);
            __builtin_prefetch(w4 + b + 512); __builtin_prefetch(w5 + b + 512);
            __builtin_prefetch(w6 + b + 512); __builtin_prefetch(w7 + b + 512);
        }
        const svint16_t xl = svld1_s16(ph, xq + b);
        const svint16_t xh = svld1_s16(ph, xq + b + 32);
#define PANEL_DOT(WP, LO, HI) do { \
            LO = svdot_s64(svdup_s64(0), svreinterpret_s16_u16(svld1ub_u16(ph, (WP) + b)), xl); \
            HI = svdot_s64(svdup_s64(0), svreinterpret_s16_u16(svld1ub_u16(ph, (WP) + b + 32)), xh); \
        } while (0)
        svint64_t l0, l1, l2, l3, l4, l5, l6, l7;
        svint64_t h0, h1, h2, h3, h4, h5, h6, h7;
        PANEL_DOT(w0, l0, h0); PANEL_DOT(w1, l1, h1);
        PANEL_DOT(w2, l2, h2); PANEL_DOT(w3, l3, h3);
        PANEL_DOT(w4, l4, h4); PANEL_DOT(w5, l5, h5);
        PANEL_DOT(w6, l6, h6); PANEL_DOT(w7, l7, h7);
#undef PANEL_DOT
        l0 = svadd_s64_x(pd, l0, h0); l1 = svadd_s64_x(pd, l1, h1);
        l2 = svadd_s64_x(pd, l2, h2); l3 = svadd_s64_x(pd, l3, h3);
        l4 = svadd_s64_x(pd, l4, h4); l5 = svadd_s64_x(pd, l5, h5);
        l6 = svadd_s64_x(pd, l6, h6); l7 = svadd_s64_x(pd, l7, h7);
        const int blk = (qg0 + b) >> 6;
        const double sc0 = s0[blk], sc1 = s1[blk], sc2 = s2[blk], sc3 = s3[blk];
        const double sc4 = s4[blk], sc5 = s5[blk], sc6 = s6[blk], sc7 = s7[blk];
        const double xg = (double)xgsum[blk];
        a0 = svmla_n_f64_x(pd, a0, svcvt_f64_s64_x(pd, l0), sc0);
        a1 = svmla_n_f64_x(pd, a1, svcvt_f64_s64_x(pd, l1), sc1);
        a2 = svmla_n_f64_x(pd, a2, svcvt_f64_s64_x(pd, l2), sc2);
        a3 = svmla_n_f64_x(pd, a3, svcvt_f64_s64_x(pd, l3), sc3);
        a4 = svmla_n_f64_x(pd, a4, svcvt_f64_s64_x(pd, l4), sc4);
        a5 = svmla_n_f64_x(pd, a5, svcvt_f64_s64_x(pd, l5), sc5);
        a6 = svmla_n_f64_x(pd, a6, svcvt_f64_s64_x(pd, l6), sc6);
        a7 = svmla_n_f64_x(pd, a7, svcvt_f64_s64_x(pd, l7), sc7);
        bc0 += sc0 * xg; bc1 += sc1 * xg; bc2 += sc2 * xg; bc3 += sc3 * xg;
        bc4 += sc4 * xg; bc5 += sc5 * xg; bc6 += sc6 * xg; bc7 += sc7 * xg;
    }
    dst[0] = (float)(svaddv_f64(pd, a0) - 128.0 * bc0);
    dst[1] = (float)(svaddv_f64(pd, a1) - 128.0 * bc1);
    dst[2] = (float)(svaddv_f64(pd, a2) - 128.0 * bc2);
    dst[3] = (float)(svaddv_f64(pd, a3) - 128.0 * bc3);
    dst[4] = (float)(svaddv_f64(pd, a4) - 128.0 * bc4);
    dst[5] = (float)(svaddv_f64(pd, a5) - 128.0 * bc5);
    dst[6] = (float)(svaddv_f64(pd, a6) - 128.0 * bc6);
    dst[7] = (float)(svaddv_f64(pd, a7) - 128.0 * bc7);
}

/* Packed block layout: for each 32-column panel, store 32 bytes from rows
 * 0..7 consecutively. One 8-row panel is exactly one 256-byte cache line. */
__attribute__((noinline))
static void kernel_packed(CANDIDATE_ARGS) {
    (void)w1; (void)w2; (void)w3; (void)w4; (void)w5; (void)w6; (void)w7;
    if (gs != 64 || (qg0 & 63) != 0 || (cols & 63) != 0) {
        memset(dst, 0, 8 * sizeof(*dst));
        return;
    }
    const svbool_t pd = svptrue_b64();
    const svbool_t ph = svptrue_b16();
    svfloat64_t a0 = svdup_f64(0), a1 = svdup_f64(0), a2 = svdup_f64(0), a3 = svdup_f64(0);
    svfloat64_t a4 = svdup_f64(0), a5 = svdup_f64(0), a6 = svdup_f64(0), a7 = svdup_f64(0);
    double bc0 = 0, bc1 = 0, bc2 = 0, bc3 = 0, bc4 = 0, bc5 = 0, bc6 = 0, bc7 = 0;
    for (int b = 0; b < cols; b += 64) {
        if (b + 128 < cols) __builtin_prefetch(w0 + (size_t)(b + 128) * 8);
        svint64_t d0 = svdup_s64(0), d1 = svdup_s64(0), d2 = svdup_s64(0), d3 = svdup_s64(0);
        svint64_t d4 = svdup_s64(0), d5 = svdup_s64(0), d6 = svdup_s64(0), d7 = svdup_s64(0);
        for (int c = b; c < b + 64; c += 32) {
            const uint8_t *p = w0 + (size_t)c * 8;
            svint16_t xv = svld1_s16(ph, xq + c);
#define PACKED_DOT(ROW, D) do { \
                svint16_t wv = svreinterpret_s16_u16(svld1ub_u16(ph, p + (ROW) * 32)); \
                D = svdot_s64(D, wv, xv); \
                __asm__ volatile("" ::: "memory"); \
            } while (0)
            PACKED_DOT(0, d0); PACKED_DOT(1, d1); PACKED_DOT(2, d2); PACKED_DOT(3, d3);
            PACKED_DOT(4, d4); PACKED_DOT(5, d5); PACKED_DOT(6, d6); PACKED_DOT(7, d7);
#undef PACKED_DOT
        }
        const int blk = (qg0 + b) >> 6;
        const double sc0 = s0[blk], sc1 = s1[blk], sc2 = s2[blk], sc3 = s3[blk];
        const double sc4 = s4[blk], sc5 = s5[blk], sc6 = s6[blk], sc7 = s7[blk];
        const double xg = (double)xgsum[blk];
        a0 = svmla_n_f64_x(pd, a0, svcvt_f64_s64_x(pd, d0), sc0);
        a1 = svmla_n_f64_x(pd, a1, svcvt_f64_s64_x(pd, d1), sc1);
        a2 = svmla_n_f64_x(pd, a2, svcvt_f64_s64_x(pd, d2), sc2);
        a3 = svmla_n_f64_x(pd, a3, svcvt_f64_s64_x(pd, d3), sc3);
        a4 = svmla_n_f64_x(pd, a4, svcvt_f64_s64_x(pd, d4), sc4);
        a5 = svmla_n_f64_x(pd, a5, svcvt_f64_s64_x(pd, d5), sc5);
        a6 = svmla_n_f64_x(pd, a6, svcvt_f64_s64_x(pd, d6), sc6);
        a7 = svmla_n_f64_x(pd, a7, svcvt_f64_s64_x(pd, d7), sc7);
        bc0 += sc0 * xg; bc1 += sc1 * xg; bc2 += sc2 * xg; bc3 += sc3 * xg;
        bc4 += sc4 * xg; bc5 += sc5 * xg; bc6 += sc6 * xg; bc7 += sc7 * xg;
    }
    dst[0] = (float)(svaddv_f64(pd, a0) - 128.0 * bc0);
    dst[1] = (float)(svaddv_f64(pd, a1) - 128.0 * bc1);
    dst[2] = (float)(svaddv_f64(pd, a2) - 128.0 * bc2);
    dst[3] = (float)(svaddv_f64(pd, a3) - 128.0 * bc3);
    dst[4] = (float)(svaddv_f64(pd, a4) - 128.0 * bc4);
    dst[5] = (float)(svaddv_f64(pd, a5) - 128.0 * bc5);
    dst[6] = (float)(svaddv_f64(pd, a6) - 128.0 * bc6);
    dst[7] = (float)(svaddv_f64(pd, a7) - 128.0 * bc7);
}

__attribute__((noinline))
static void kernel_packed_wide(CANDIDATE_ARGS) {
    (void)w1; (void)w2; (void)w3; (void)w4; (void)w5; (void)w6; (void)w7;
    if (gs != 64 || (qg0 & 63) != 0 || (cols & 63) != 0) {
        memset(dst, 0, 8 * sizeof(*dst));
        return;
    }
    const svbool_t pd = svptrue_b64();
    const svbool_t ph = svptrue_b16();
    const svbool_t pb = svptrue_b8();
    svfloat64_t a0 = svdup_f64(0), a1 = svdup_f64(0), a2 = svdup_f64(0), a3 = svdup_f64(0);
    svfloat64_t a4 = svdup_f64(0), a5 = svdup_f64(0), a6 = svdup_f64(0), a7 = svdup_f64(0);
    double bc0 = 0, bc1 = 0, bc2 = 0, bc3 = 0, bc4 = 0, bc5 = 0, bc6 = 0, bc7 = 0;
    for (int b = 0; b < cols; b += 64) {
        if (b + 128 < cols) __builtin_prefetch(w0 + (size_t)(b + 128) * 8);
        svint64_t d0 = svdup_s64(0), d1 = svdup_s64(0), d2 = svdup_s64(0), d3 = svdup_s64(0);
        svint64_t d4 = svdup_s64(0), d5 = svdup_s64(0), d6 = svdup_s64(0), d7 = svdup_s64(0);
        for (int c = b; c < b + 64; c += 32) {
            const uint8_t *p = w0 + (size_t)c * 8;
            const svint16_t xv = svld1_s16(ph, xq + c);
#define PACKED_PAIR(OFF, DA, DB) do { \
                svuint8_t pair = svld1_u8(pb, p + (OFF)); \
                DA = svdot_s64(DA, svreinterpret_s16_u16(svunpklo_u16(pair)), xv); \
                DB = svdot_s64(DB, svreinterpret_s16_u16(svunpkhi_u16(pair)), xv); \
            } while (0)
            PACKED_PAIR(0, d0, d1); PACKED_PAIR(64, d2, d3);
            PACKED_PAIR(128, d4, d5); PACKED_PAIR(192, d6, d7);
#undef PACKED_PAIR
        }
        const int blk = (qg0 + b) >> 6;
        const double sc0 = s0[blk], sc1 = s1[blk], sc2 = s2[blk], sc3 = s3[blk];
        const double sc4 = s4[blk], sc5 = s5[blk], sc6 = s6[blk], sc7 = s7[blk];
        const double xg = (double)xgsum[blk];
        a0 = svmla_n_f64_x(pd, a0, svcvt_f64_s64_x(pd, d0), sc0);
        a1 = svmla_n_f64_x(pd, a1, svcvt_f64_s64_x(pd, d1), sc1);
        a2 = svmla_n_f64_x(pd, a2, svcvt_f64_s64_x(pd, d2), sc2);
        a3 = svmla_n_f64_x(pd, a3, svcvt_f64_s64_x(pd, d3), sc3);
        a4 = svmla_n_f64_x(pd, a4, svcvt_f64_s64_x(pd, d4), sc4);
        a5 = svmla_n_f64_x(pd, a5, svcvt_f64_s64_x(pd, d5), sc5);
        a6 = svmla_n_f64_x(pd, a6, svcvt_f64_s64_x(pd, d6), sc6);
        a7 = svmla_n_f64_x(pd, a7, svcvt_f64_s64_x(pd, d7), sc7);
        bc0 += sc0 * xg; bc1 += sc1 * xg; bc2 += sc2 * xg; bc3 += sc3 * xg;
        bc4 += sc4 * xg; bc5 += sc5 * xg; bc6 += sc6 * xg; bc7 += sc7 * xg;
    }
    dst[0] = (float)(svaddv_f64(pd, a0) - 128.0 * bc0);
    dst[1] = (float)(svaddv_f64(pd, a1) - 128.0 * bc1);
    dst[2] = (float)(svaddv_f64(pd, a2) - 128.0 * bc2);
    dst[3] = (float)(svaddv_f64(pd, a3) - 128.0 * bc3);
    dst[4] = (float)(svaddv_f64(pd, a4) - 128.0 * bc4);
    dst[5] = (float)(svaddv_f64(pd, a5) - 128.0 * bc5);
    dst[6] = (float)(svaddv_f64(pd, a6) - 128.0 * bc6);
    dst[7] = (float)(svaddv_f64(pd, a7) - 128.0 * bc7);
}

typedef struct {
    const uint8_t *w;
    const float *s[8];
    const int16_t *xq;
    const int64_t *xg;
    int cols;
} packed_asm_job;

extern void glm5_matvec_int16sdot_packed_asm(float *dst, const packed_asm_job *job);

__attribute__((noinline))
static void kernel_packed_asm(CANDIDATE_ARGS) {
    (void)w1; (void)w2; (void)w3; (void)w4; (void)w5; (void)w6; (void)w7;
    if (gs != 64 || (qg0 & 63) != 0 || (cols & 63) != 0) {
        memset(dst, 0, 8 * sizeof(*dst));
        return;
    }
    const int blk0 = qg0 >> 6;
    const packed_asm_job job = {
        w0,
        {s0 + blk0, s1 + blk0, s2 + blk0, s3 + blk0,
         s4 + blk0, s5 + blk0, s6 + blk0, s7 + blk0},
        xq, xgsum + blk0, cols
    };
    glm5_matvec_int16sdot_packed_asm(dst, &job);
}

__attribute__((noinline))
static void kernel_wide64(CANDIDATE_ARGS) {
    if (gs != 64 || (qg0 & 63) != 0 || (cols & 63) != 0) {
        kernel_baseline(dst, w0, w1, w2, w3, w4, w5, w6, w7,
                s0, s1, s2, s3, s4, s5, s6, s7, gs, qg0, xq, xgsum, cols);
        return;
    }
    svbool_t pd = svptrue_b64();
    svbool_t ph = svptrue_b16();
    svbool_t pb = svptrue_b8();
    svfloat64_t a0 = svdup_f64(0), a1 = svdup_f64(0), a2 = svdup_f64(0), a3 = svdup_f64(0);
    svfloat64_t a4 = svdup_f64(0), a5 = svdup_f64(0), a6 = svdup_f64(0), a7 = svdup_f64(0);
    double bc0 = 0, bc1 = 0, bc2 = 0, bc3 = 0, bc4 = 0, bc5 = 0, bc6 = 0, bc7 = 0;
    for (int b = 0; b < cols; b += 64) {
        svint16_t xl = svld1_s16(ph, xq + b);
        svint16_t xh = svld1_s16(ph, xq + b + 32);
        svint64_t d0 = svdup_s64(0), d1 = svdup_s64(0), d2 = svdup_s64(0), d3 = svdup_s64(0);
        svint64_t d4 = svdup_s64(0), d5 = svdup_s64(0), d6 = svdup_s64(0), d7 = svdup_s64(0);
#define WIDE_DOT(WP, D) do { \
            svuint8_t wb = svld1_u8(pb, (WP) + b); \
            svint16_t wl = svreinterpret_s16_u16(svunpklo_u16(wb)); \
            svint16_t wh = svreinterpret_s16_u16(svunpkhi_u16(wb)); \
            D = svdot_s64(D, wl, xl); \
            D = svdot_s64(D, wh, xh); \
            __asm__ volatile("" ::: "memory"); \
        } while (0)
        WIDE_DOT(w0, d0); WIDE_DOT(w1, d1); WIDE_DOT(w2, d2); WIDE_DOT(w3, d3);
        WIDE_DOT(w4, d4); WIDE_DOT(w5, d5); WIDE_DOT(w6, d6); WIDE_DOT(w7, d7);
#undef WIDE_DOT
        int blk = (qg0 + b) >> 6;
        double sc0 = s0[blk], sc1 = s1[blk], sc2 = s2[blk], sc3 = s3[blk];
        double sc4 = s4[blk], sc5 = s5[blk], sc6 = s6[blk], sc7 = s7[blk];
        double xg = (double)xgsum[blk];
        a0 = svmla_n_f64_x(pd, a0, svcvt_f64_s64_x(pd, d0), sc0);
        a1 = svmla_n_f64_x(pd, a1, svcvt_f64_s64_x(pd, d1), sc1);
        a2 = svmla_n_f64_x(pd, a2, svcvt_f64_s64_x(pd, d2), sc2);
        a3 = svmla_n_f64_x(pd, a3, svcvt_f64_s64_x(pd, d3), sc3);
        a4 = svmla_n_f64_x(pd, a4, svcvt_f64_s64_x(pd, d4), sc4);
        a5 = svmla_n_f64_x(pd, a5, svcvt_f64_s64_x(pd, d5), sc5);
        a6 = svmla_n_f64_x(pd, a6, svcvt_f64_s64_x(pd, d6), sc6);
        a7 = svmla_n_f64_x(pd, a7, svcvt_f64_s64_x(pd, d7), sc7);
        bc0 += sc0 * xg; bc1 += sc1 * xg; bc2 += sc2 * xg; bc3 += sc3 * xg;
        bc4 += sc4 * xg; bc5 += sc5 * xg; bc6 += sc6 * xg; bc7 += sc7 * xg;
    }
    dst[0] = (float)(svaddv_f64(pd, a0) - 128.0 * bc0);
    dst[1] = (float)(svaddv_f64(pd, a1) - 128.0 * bc1);
    dst[2] = (float)(svaddv_f64(pd, a2) - 128.0 * bc2);
    dst[3] = (float)(svaddv_f64(pd, a3) - 128.0 * bc3);
    dst[4] = (float)(svaddv_f64(pd, a4) - 128.0 * bc4);
    dst[5] = (float)(svaddv_f64(pd, a5) - 128.0 * bc5);
    dst[6] = (float)(svaddv_f64(pd, a6) - 128.0 * bc6);
    dst[7] = (float)(svaddv_f64(pd, a7) - 128.0 * bc7);
}

DEFINE_CANDIDATE(kernel_load2, LOAD_DOT_2)
DEFINE_CANDIDATE(kernel_load4, LOAD_DOT_4)
DEFINE_CANDIDATE(kernel_load8, LOAD_DOT_8)

__attribute__((noinline))
static void kernel_sdot_only(CANDIDATE_ARGS) {
    (void)s0; (void)s1; (void)s2; (void)s3; (void)s4; (void)s5; (void)s6; (void)s7;
    (void)gs; (void)qg0; (void)xgsum;
    const svbool_t pd = svptrue_b64();
    const int vh = (int)svcnth();
    svint64_t d0 = svdup_s64(0), d1 = svdup_s64(0), d2 = svdup_s64(0), d3 = svdup_s64(0);
    svint64_t d4 = svdup_s64(0), d5 = svdup_s64(0), d6 = svdup_s64(0), d7 = svdup_s64(0);
    for (int c = 0; c < cols; c += vh) {
        svbool_t pg = svwhilelt_b16((uint32_t)c, (uint32_t)cols);
        svint16_t xv = svld1_s16(pg, xq + c);
        svint16_t wv;
#define SDOT_ONE(W, D) do { wv = svreinterpret_s16_u16(svld1ub_u16(pg, (W) + c)); D = svdot_s64(D, wv, xv); } while (0)
        SDOT_ONE(w0, d0); SDOT_ONE(w1, d1); SDOT_ONE(w2, d2); SDOT_ONE(w3, d3);
        SDOT_ONE(w4, d4); SDOT_ONE(w5, d5); SDOT_ONE(w6, d6); SDOT_ONE(w7, d7);
#undef SDOT_ONE
    }
    dst[0] = (float)svaddv_s64(pd, d0); dst[1] = (float)svaddv_s64(pd, d1);
    dst[2] = (float)svaddv_s64(pd, d2); dst[3] = (float)svaddv_s64(pd, d3);
    dst[4] = (float)svaddv_s64(pd, d4); dst[5] = (float)svaddv_s64(pd, d5);
    dst[6] = (float)svaddv_s64(pd, d6); dst[7] = (float)svaddv_s64(pd, d7);
}

__attribute__((noinline))
static void kernel_raw8(CANDIDATE_ARGS) {
    (void)s0; (void)s1; (void)s2; (void)s3; (void)s4; (void)s5; (void)s6; (void)s7;
    (void)gs; (void)qg0; (void)xq; (void)xgsum;
    const svbool_t pg = svptrue_b64();
    const int vb = (int)svcntb();
    svuint64_t a0 = svdup_u64(0), a1 = svdup_u64(0), a2 = svdup_u64(0), a3 = svdup_u64(0);
    svuint64_t a4 = svdup_u64(0), a5 = svdup_u64(0), a6 = svdup_u64(0), a7 = svdup_u64(0);
    for (int c = 0; c < cols; c += vb) {
        a0 = svadd_u64_x(pg, a0, svld1_u64(pg, (const uint64_t *)(w0 + c)));
        a1 = svadd_u64_x(pg, a1, svld1_u64(pg, (const uint64_t *)(w1 + c)));
        a2 = svadd_u64_x(pg, a2, svld1_u64(pg, (const uint64_t *)(w2 + c)));
        a3 = svadd_u64_x(pg, a3, svld1_u64(pg, (const uint64_t *)(w3 + c)));
        a4 = svadd_u64_x(pg, a4, svld1_u64(pg, (const uint64_t *)(w4 + c)));
        a5 = svadd_u64_x(pg, a5, svld1_u64(pg, (const uint64_t *)(w5 + c)));
        a6 = svadd_u64_x(pg, a6, svld1_u64(pg, (const uint64_t *)(w6 + c)));
        a7 = svadd_u64_x(pg, a7, svld1_u64(pg, (const uint64_t *)(w7 + c)));
    }
    dst[0] = (float)svaddv_u64(pg, a0); dst[1] = (float)svaddv_u64(pg, a1);
    dst[2] = (float)svaddv_u64(pg, a2); dst[3] = (float)svaddv_u64(pg, a3);
    dst[4] = (float)svaddv_u64(pg, a4); dst[5] = (float)svaddv_u64(pg, a5);
    dst[6] = (float)svaddv_u64(pg, a6); dst[7] = (float)svaddv_u64(pg, a7);
}

__attribute__((noinline))
static void kernel_stream1(CANDIDATE_ARGS) {
    (void)w1; (void)w2; (void)w3; (void)w4; (void)w5; (void)w6; (void)w7;
    (void)s0; (void)s1; (void)s2; (void)s3; (void)s4; (void)s5; (void)s6; (void)s7;
    (void)gs; (void)qg0; (void)xq; (void)xgsum;
    const svbool_t pg = svptrue_b64();
    const int vb = (int)svcntb();
    svuint64_t a0 = svdup_u64(0), a1 = svdup_u64(0), a2 = svdup_u64(0), a3 = svdup_u64(0);
    svuint64_t a4 = svdup_u64(0), a5 = svdup_u64(0), a6 = svdup_u64(0), a7 = svdup_u64(0);
    for (int c = 0; c < 8 * cols; c += 8 * vb) {
        a0 = svadd_u64_x(pg, a0, svld1_u64(pg, (const uint64_t *)(w0 + c + 0 * vb)));
        a1 = svadd_u64_x(pg, a1, svld1_u64(pg, (const uint64_t *)(w0 + c + 1 * vb)));
        a2 = svadd_u64_x(pg, a2, svld1_u64(pg, (const uint64_t *)(w0 + c + 2 * vb)));
        a3 = svadd_u64_x(pg, a3, svld1_u64(pg, (const uint64_t *)(w0 + c + 3 * vb)));
        a4 = svadd_u64_x(pg, a4, svld1_u64(pg, (const uint64_t *)(w0 + c + 4 * vb)));
        a5 = svadd_u64_x(pg, a5, svld1_u64(pg, (const uint64_t *)(w0 + c + 5 * vb)));
        a6 = svadd_u64_x(pg, a6, svld1_u64(pg, (const uint64_t *)(w0 + c + 6 * vb)));
        a7 = svadd_u64_x(pg, a7, svld1_u64(pg, (const uint64_t *)(w0 + c + 7 * vb)));
    }
    dst[0] = (float)svaddv_u64(pg, a0); dst[1] = (float)svaddv_u64(pg, a1);
    dst[2] = (float)svaddv_u64(pg, a2); dst[3] = (float)svaddv_u64(pg, a3);
    dst[4] = (float)svaddv_u64(pg, a4); dst[5] = (float)svaddv_u64(pg, a5);
    dst[6] = (float)svaddv_u64(pg, a6); dst[7] = (float)svaddv_u64(pg, a7);
}

typedef struct {
    const char *name;
    kernel_fn fn;
    int packed;
} variant;

static int check_candidates(void) {
    enum { MAX_COLS = BENCH_COLS, MAX_SB = BENCH_COLS / BENCH_GS + 4 };
    uint8_t *w = mmap(NULL, (size_t)8 * MAX_COLS, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint8_t *wp = mmap(NULL, (size_t)8 * MAX_COLS, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *s = mmap(NULL, (size_t)8 * MAX_SB * sizeof(float), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int16_t *xq = mmap(NULL, (size_t)MAX_COLS * sizeof(int16_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int64_t *xg = mmap(NULL, (size_t)MAX_SB * sizeof(int64_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (w == MAP_FAILED || wp == MAP_FAILED || s == MAP_FAILED ||
            xq == MAP_FAILED || xg == MAP_FAILED) return 2;

    kernel_fn candidates[] = {
        kernel_fast1, kernel_pf256, kernel_pf512, kernel_pf1024, kernel_pf2048, kernel_pf4096,
        kernel_pf512_i128, kernel_pf512_i256, kernel_pf512_i512, kernel_pf768_i256,
        kernel_4phase, kernel_panel2,
        kernel_wide64, kernel_load2, kernel_load4, kernel_load8
    };
    kernel_fn packed_candidates[] = {kernel_packed, kernel_packed_wide, kernel_packed_asm};
    const int cols_cases[] = {64, 128, BENCH_COLS};
    const int qg_cases[] = {0, 64};
    uint32_t rng = 1;
    int errors = 0;
    for (size_t ci = 0; ci < sizeof(cols_cases) / sizeof(cols_cases[0]); ci++) {
        int cols = cols_cases[ci];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < cols; c++) w[(size_t)r * MAX_COLS + c] = (uint8_t)(lcg32(&rng) >> 24);
            for (int b = 0; b < MAX_SB; b++) s[(size_t)r * MAX_SB + b] = (float)(1 + (lcg32(&rng) & 1023)) * 1.0e-6f;
        }
        for (int c = 0; c < cols; c += 32) {
            for (int r = 0; r < 8; r++)
                memcpy(wp + (size_t)c * 8 + r * 32, w + (size_t)r * MAX_COLS + c, 32);
        }
        for (int trial = 0; trial < 16; trial++) {
            for (int c = 0; c < cols; c++) {
                if (trial == 0) xq[c] = (int16_t)((c & 1) ? 32767 : -32767);
                else xq[c] = (int16_t)((int)(lcg32(&rng) % 4001) - 2000);
            }
            for (size_t qi = 0; qi < sizeof(qg_cases) / sizeof(qg_cases[0]); qi++) {
                int qg0 = qg_cases[qi];
                memset(xg, 0, (size_t)MAX_SB * sizeof(*xg));
                for (int b = 0; b < cols; b += 64) {
                    int64_t sum = 0;
                    for (int c = b; c < b + 64; c++) sum += xq[c];
                    xg[(qg0 + b) / 64] = sum;
                }
                float ref[8], got[8];
                kernel_baseline(ref,
                        w, w + MAX_COLS, w + 2 * MAX_COLS, w + 3 * MAX_COLS,
                        w + 4 * MAX_COLS, w + 5 * MAX_COLS, w + 6 * MAX_COLS, w + 7 * MAX_COLS,
                        s, s + MAX_SB, s + 2 * MAX_SB, s + 3 * MAX_SB,
                        s + 4 * MAX_SB, s + 5 * MAX_SB, s + 6 * MAX_SB, s + 7 * MAX_SB,
                        64, qg0, xq, xg, cols);
                for (size_t k = 0; k < sizeof(candidates) / sizeof(candidates[0]); k++) {
                    candidates[k](got,
                            w, w + MAX_COLS, w + 2 * MAX_COLS, w + 3 * MAX_COLS,
                            w + 4 * MAX_COLS, w + 5 * MAX_COLS, w + 6 * MAX_COLS, w + 7 * MAX_COLS,
                            s, s + MAX_SB, s + 2 * MAX_SB, s + 3 * MAX_SB,
                            s + 4 * MAX_SB, s + 5 * MAX_SB, s + 6 * MAX_SB, s + 7 * MAX_SB,
                            64, qg0, xq, xg, cols);
                    if (memcmp(ref, got, sizeof(ref)) != 0) {
                        errors++;
                        fprintf(stderr, "CHECK_FAIL candidate=%zu cols=%d qg0=%d trial=%d\n", k, cols, qg0, trial);
                    }
                }
                for (size_t k = 0; k < sizeof(packed_candidates) / sizeof(packed_candidates[0]); k++) {
                    packed_candidates[k](got,
                            wp, wp + MAX_COLS, wp + 2 * MAX_COLS, wp + 3 * MAX_COLS,
                            wp + 4 * MAX_COLS, wp + 5 * MAX_COLS, wp + 6 * MAX_COLS, wp + 7 * MAX_COLS,
                            s, s + MAX_SB, s + 2 * MAX_SB, s + 3 * MAX_SB,
                            s + 4 * MAX_SB, s + 5 * MAX_SB, s + 6 * MAX_SB, s + 7 * MAX_SB,
                            64, qg0, xq, xg, cols);
                    if (memcmp(ref, got, sizeof(ref)) != 0) {
                        if (errors == 0) {
                            fprintf(stderr, "CHECK_DETAIL ref=%g,%g got=%g,%g\n",
                                    ref[0], ref[1], got[0], got[1]);
                        }
                        errors++;
                        fprintf(stderr, "CHECK_FAIL packed=%zu cols=%d qg0=%d trial=%d\n", k, cols, qg0, trial);
                    }
                }
            }
        }
    }
    munmap(w, (size_t)8 * MAX_COLS);
    munmap(wp, (size_t)8 * MAX_COLS);
    munmap(s, (size_t)8 * MAX_SB * sizeof(float));
    munmap(xq, (size_t)MAX_COLS * sizeof(int16_t));
    munmap(xg, (size_t)MAX_SB * sizeof(int64_t));
    printf("CHECK bit_exact_cases=%zu errors=%d\n",
            sizeof(cols_cases) / sizeof(cols_cases[0]) *
            sizeof(qg_cases) / sizeof(qg_cases[0]) * 16u *
            (sizeof(candidates) / sizeof(candidates[0]) +
             sizeof(packed_candidates) / sizeof(packed_candidates[0])), errors);
    return errors ? 1 : 0;
}

static double run_variant(const variant *v, uint8_t *w, uint8_t *wp, float *s, float *y,
        const int16_t *xq, const int64_t *xg, size_t ntens, int rows,
        int sb, int reps, size_t weight_bytes) {
    double best = 1.0e100, total = 0;
    for (int it = 0; it < reps; it++) {
        double sink = 0;
        double t0 = now_sec();
        fapp_start(v->name, 1, 0);
#pragma omp parallel reduction(+:sink)
        {
            int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
            int nb = BENCH_ROWS / 8;
            int per = (nb + nthr - 1) / nthr;
            int b0 = tid * per;
            int b1 = b0 + per > nb ? nb : b0 + per;
            for (size_t t = 0; t < ntens; t++) {
                uint8_t *wt = (v->packed ? wp : w) + t * (size_t)BENCH_ROWS * BENCH_COLS;
                float *st = s + t * (size_t)BENCH_ROWS * sb;
                float *yt = y + t * (size_t)BENCH_ROWS;
                for (int bi = b0; bi < b1; bi++) {
                    int r = bi * 8;
                    const uint8_t *wr = wt + (size_t)r * BENCH_COLS;
                    const float *sr = st + (size_t)r * sb;
                    float *dst = yt + r;
                    v->fn(dst,
                            wr, wr + BENCH_COLS, wr + 2 * BENCH_COLS, wr + 3 * BENCH_COLS,
                            wr + 4 * BENCH_COLS, wr + 5 * BENCH_COLS, wr + 6 * BENCH_COLS, wr + 7 * BENCH_COLS,
                            sr, sr + sb, sr + 2 * sb, sr + 3 * sb,
                            sr + 4 * sb, sr + 5 * sb, sr + 6 * sb, sr + 7 * sb,
                            BENCH_GS, 0, xq, xg, BENCH_COLS);
                    sink += dst[0] + dst[1] + dst[2] + dst[3] + dst[4] + dst[5] + dst[6] + dst[7];
                }
            }
        }
        fapp_stop(v->name, 1, 0);
        double dt = now_sec() - t0;
        g_sink += sink;
        if (dt < best) best = dt;
        total += dt;
        printf("RESULT %-9s rep=%d seconds=%.6f GB/s=%.1f sink=%.9g\n",
                v->name, it, dt, (double)weight_bytes / dt / 1.0e9, sink);
    }
    printf("SUMMARY %-9s best_GB/s=%.1f mean_GB/s=%.1f rows=%d bytes=%zu reps=%d\n",
            v->name, (double)weight_bytes / best / 1.0e9,
            (double)weight_bytes / (total / reps) / 1.0e9, rows, weight_bytes, reps);
    return best;
}

int main(void) {
    int check = check_candidates();
    if (check) return check;
    const char *mode = getenv("MODE");
    if (mode && strcmp(mode, "check") == 0) return 0;

    size_t target = env_size("WEIGHT_BYTES", (size_t)2 << 30);
    int reps = env_int("REPS", 3);
    size_t tensor_bytes = (size_t)BENCH_ROWS * BENCH_COLS;
    size_t ntens = target / tensor_bytes;
    if (ntens < 1) ntens = 1;
    int rows = (int)(ntens * BENCH_ROWS);
    size_t weight_bytes = ntens * tensor_bytes;
    int sb = BENCH_COLS / BENCH_GS;
    size_t scale_count = (size_t)rows * sb;

    uint8_t *w = mmap(NULL, weight_bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint8_t *wp = mmap(NULL, weight_bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *s = mmap(NULL, scale_count * sizeof(float), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *y = mmap(NULL, (size_t)rows * sizeof(float), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int16_t *xq = mmap(NULL, (size_t)BENCH_COLS * sizeof(int16_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int64_t *xg = mmap(NULL, (size_t)sb * sizeof(int64_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (w == MAP_FAILED || wp == MAP_FAILED || s == MAP_FAILED || y == MAP_FAILED ||
            xq == MAP_FAILED || xg == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        return 2;
    }

    for (int c = 0; c < BENCH_COLS; c++) xq[c] = (int16_t)((c * 37) % 4001 - 2000);
    for (int b = 0; b < sb; b++) {
        int64_t sum = 0;
        for (int c = b * BENCH_GS; c < (b + 1) * BENCH_GS; c++) sum += xq[c];
        xg[b] = sum;
    }
    const int global_touch = env_int("GLOBAL_TOUCH", 0);
    if (global_touch) {
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            uint32_t rng = 1u + (uint32_t)r * 2654435761u;
            uint8_t *wr = w + (size_t)r * BENCH_COLS;
            float *sr = s + (size_t)r * sb;
            for (int c = 0; c < BENCH_COLS; c++) wr[c] = (uint8_t)(lcg32(&rng) >> 24);
            for (int b = 0; b < sb; b++) sr[b] = (float)(1 + (lcg32(&rng) & 1023)) * 1.0e-6f;
        }
    } else {
        /* Match production ownership: each tensor is first-touched by the same
         * 8-row-block partition that reads it in glm5_i16_worker. */
#pragma omp parallel
        {
            int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
            int nb = BENCH_ROWS / 8;
            int per = (nb + nthr - 1) / nthr;
            int b0 = tid * per;
            int b1 = b0 + per > nb ? nb : b0 + per;
            for (size_t t = 0; t < ntens; t++) {
                for (int bi = b0; bi < b1; bi++) {
                    for (int j = 0; j < 8; j++) {
                        int r = (int)(t * BENCH_ROWS) + bi * 8 + j;
                        uint32_t rng = 1u + (uint32_t)r * 2654435761u;
                        uint8_t *wr = w + (size_t)r * BENCH_COLS;
                        float *sr = s + (size_t)r * sb;
                        for (int c = 0; c < BENCH_COLS; c++) wr[c] = (uint8_t)(lcg32(&rng) >> 24);
                        for (int b = 0; b < sb; b++) sr[b] = (float)(1 + (lcg32(&rng) & 1023)) * 1.0e-6f;
                    }
                }
            }
        }
    }

    /* Repack after row initialization. The same worker that consumes an
     * 8-row block first-touches its packed copy. */
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
        int nb = BENCH_ROWS / 8;
        int per = (nb + nthr - 1) / nthr;
        int b0 = tid * per;
        int b1 = b0 + per > nb ? nb : b0 + per;
        for (size_t t = 0; t < ntens; t++) {
            for (int bi = b0; bi < b1; bi++) {
                const int r0 = (int)(t * BENCH_ROWS) + bi * 8;
                uint8_t *pb = wp + (size_t)r0 * BENCH_COLS;
                for (int c = 0; c < BENCH_COLS; c += 32) {
                    for (int j = 0; j < 8; j++)
                        memcpy(pb + (size_t)c * 8 + j * 32,
                                w + (size_t)(r0 + j) * BENCH_COLS + c, 32);
                }
            }
        }
    }

    variant variants[] = {
        {"stream1", kernel_stream1, 0},
        {"raw8", kernel_raw8, 0},
        {"sdot8", kernel_sdot_only, 0},
        {"baseline", kernel_baseline, 0},
        {"fast1", kernel_fast1, 0},
        {"pf256", kernel_pf256, 0},
        {"pf512", kernel_pf512, 0},
        {"pf1024", kernel_pf1024, 0},
        {"pf2048", kernel_pf2048, 0},
        {"pf4096", kernel_pf4096, 0},
        {"pf512i128", kernel_pf512_i128, 0},
        {"pf512i256", kernel_pf512_i256, 0},
        {"pf512i512", kernel_pf512_i512, 0},
        {"pf768i256", kernel_pf768_i256, 0},
        {"phase4", kernel_4phase, 0},
        {"panel2", kernel_panel2, 0},
        {"packedraw", kernel_stream1, 1},
        {"packed", kernel_packed, 1},
        {"packedwide", kernel_packed_wide, 1},
        {"packedasm", kernel_packed_asm, 1},
        {"wide64", kernel_wide64, 0},
        {"load2", kernel_load2, 0},
        {"load4", kernel_load4, 0},
        {"load8", kernel_load8, 0},
    };
    printf("CONFIG cpus=%d threads=%d ntens=%zu rows=%d cols=%d gs=%d weights=%.3f_GiB reps=%d mode=%s touch=%s\n",
            omp_get_num_procs(), omp_get_max_threads(), ntens, rows, BENCH_COLS, BENCH_GS,
            (double)weight_bytes / (double)((size_t)1 << 30), reps, mode ? mode : "all",
            global_touch ? "global" : "reader-matched");
    for (size_t i = 0; i < sizeof(variants) / sizeof(variants[0]); i++) {
        if (mode && strcmp(mode, "all") != 0 && strcmp(mode, variants[i].name) != 0) continue;
        run_variant(&variants[i], w, wp, s, y, xq, xg, ntens, rows, sb, reps, weight_bytes);
    }
    printf("DONE g_sink=%.17g\n", g_sink);
    return 0;
}
