/* Kernel-structure variants for F4/F6 A8/A16 (development benchmark).
 * usage: bench_var VARIANT FMT ARITH ROWS COLS CORES PASSES
 *   variant 1: baseline q38d_group_sve
 *   variant 2: one group, split SDOT chains
 *   variant 3: two groups interleaved, shared activation loads, split chains
 *   variant 4: variant 3 + L1 prefetch of the weight streams */
#define _GNU_SOURCE
#include "q38d_pipe.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define AI static inline __attribute__((always_inline))
static int nosync, pf1 = 1024, pf2 = 0;

AI svint8_t rep8(const int8_t *p) { int64_t v; memcpy(&v, p, 8); return svreinterpret_s8_s64(svdup_n_s64(v)); }
AI svfloat32_t rep2(const float *p) { uint64_t v; memcpy(&v, p, 8); return svreinterpret_f32_u64(svdup_n_u64(v)); }

/* Decode one code vector to (L, H) int8 weights. */
AI void dec(int fmt, svint8_t lut, const uint8_t *code, const uint8_t *high, svint8_t *L, svint8_t *H) {
    const svbool_t pb = svptrue_b8();
    svuint8_t z = svld1_u8(pb, code);
    if (fmt == Q38D_F4) {
        *L = svtbl_s8(lut, svand_n_u8_x(pb, z, 15));
        *H = svtbl_s8(lut, svlsr_n_u8_x(pb, z, 4));
    } else {
        svuint8_t hp = svld1_u8(svwhilelt_b8((uint32_t)0, (uint32_t)32), high);
        svuint8_t t = svzip1_u8(svlsl_n_u8_x(pb, hp, 4), hp);
        svuint8_t il = svorr_u8_x(pb, svand_n_u8_x(pb, z, 15), svand_n_u8_x(pb, t, 0x30));
        svuint8_t ih = svorr_u8_x(pb, svlsr_n_u8_x(pb, z, 4), svlsr_n_u8_x(pb, svand_n_u8_x(pb, t, 0xc0), 2));
        *L = svtbl_s8(lut, il);
        *H = svtbl_s8(lut, ih);
    }
}

/* Integer pair dot with split chains. */
AI svint32_t pdot(int a16, svint8_t l0, svint8_t h0, svint8_t l1, svint8_t h1,
                  svint8_t A0, svint8_t A1, svint8_t A2, svint8_t A3,
                  svint8_t B0, svint8_t B1, svint8_t B2, svint8_t B3) {
    const svbool_t pf = svptrue_b32();
    svint32_t z = svdup_n_s32(0);
    svint32_t dA = svdot_s32(svdot_s32(z, l0, A0), h0, A1);
    svint32_t dB = svdot_s32(svdot_s32(z, l1, A2), h1, A3);
    svint32_t d = svadd_s32_x(pf, dA, dB);
    if (a16) {
        svint32_t eA = svdot_s32(svdot_s32(z, l0, B0), h0, B1);
        svint32_t eB = svdot_s32(svdot_s32(z, l1, B2), h1, B3);
        d = svadd_s32_x(pf, d, svlsl_n_s32_x(pf, svadd_s32_x(pf, eA, eB), 8));
    }
    return d;
}

AI svfloat32_t pscale(int fmt, const uint8_t *sc, size_t p, svfloat32_t asc) {
    const svbool_t pf = svptrue_b32();
    if (fmt == Q38D_F4) {
        svuint32_t u = svld1ub_u32(pf, sc + p * 16);
        return svmul_f32_x(pf, svreinterpret_f32_u32(svlsl_n_u32_x(pf, u, 20)), asc);
    }
    svuint32_t u = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc + p * 8);
    u = svzip1_u32(u, u);
    return svmul_f32_x(pf, svreinterpret_f32_u32(svlsl_n_u32_x(pf, u, 23)), asc);
}

AI void finish(float *out, svfloat32_t acc, int fmt) {
    const svbool_t pf = svptrue_b32();
    svfloat32_t r = svadd_f32_x(pf, svuzp1_f32(acc, acc), svuzp2_f32(acc, acc));
    svst1_f32(svptrue_pat_b32(SV_VL8), out, svmul_n_f32_x(pf, r, q38d_out_scale(fmt)));
}

/* variant 2: one group, split chains, unroll 2 pairs */
AI void v2_group(float *out, const uint8_t *g, int fmt, int a16, const q38d_act *a) {
    const svbool_t pf = svptrue_b32(), pb = svptrue_b8();
    const size_t np = (size_t)a->cols / 32, aqs = a16 ? 64 : 32;
    const svint8_t lut = svld1_s8(pb, fmt == Q38D_F4 ? q38d_lut_f4 : q38d_lut_f6);
    const uint8_t *high = g + np * 128, *sc = q38d_scale_stream((uint8_t *)g, fmt, a->cols);
    svfloat32_t acc0 = svdup_n_f32(0), acc1 = acc0;
    for (size_t p = 0; p < np; p += 2) {
        for (int k = 0; k < 2; k++) {
            size_t q = p + k;
            const int8_t *aq = a->q + q * aqs;
            svint8_t l0, h0, l1, h1;
            dec(fmt, lut, g + q * 128, high + q * 64, &l0, &h0);
            dec(fmt, lut, g + q * 128 + 64, high + q * 64 + 32, &l1, &h1);
            svint32_t d = pdot(a16, l0, h0, l1, h1, rep8(aq), rep8(aq + 8), rep8(aq + 16), rep8(aq + 24),
                               rep8(aq + 32), rep8(aq + 40), rep8(aq + 48), rep8(aq + 56));
            svfloat32_t s = pscale(fmt, sc, q, rep2(a->sc + 2 * q));
            if (k) acc1 = svmla_f32_x(pf, acc1, svcvt_f32_s32_x(pf, d), s);
            else acc0 = svmla_f32_x(pf, acc0, svcvt_f32_s32_x(pf, d), s);
        }
    }
    finish(out, svadd_f32_x(pf, acc0, acc1), fmt);
}

/* variant 3/4: two groups interleaved */
AI void v3_groups(float *out, const uint8_t *g0, const uint8_t *g1, int fmt, int a16,
                  const q38d_act *a, int pf_dist) {
    const svbool_t pf = svptrue_b32(), pb = svptrue_b8();
    const size_t np = (size_t)a->cols / 32, aqs = a16 ? 64 : 32;
    const svint8_t lut = svld1_s8(pb, fmt == Q38D_F4 ? q38d_lut_f4 : q38d_lut_f6);
    const uint8_t *hi0 = g0 + np * 128, *hi1 = g1 + np * 128;
    const uint8_t *sc0 = q38d_scale_stream((uint8_t *)g0, fmt, a->cols);
    const uint8_t *sc1 = q38d_scale_stream((uint8_t *)g1, fmt, a->cols);
    svfloat32_t a00 = svdup_n_f32(0), a01 = a00, a10 = a00, a11 = a00;
    for (size_t p = 0; p < np; p += 2) {
        if (pf2) {
            svprfb(pb, g0 + p * 128 + pf2, SV_PLDL2KEEP);
            svprfb(pb, g1 + p * 128 + pf2, SV_PLDL2KEEP);
        }
        if (pf_dist) {
            svprfb(pb, g0 + p * 128 + pf_dist, SV_PLDL1KEEP);
            svprfb(pb, g1 + p * 128 + pf_dist, SV_PLDL1KEEP);
            if (fmt != Q38D_F4) {
                svprfb(pb, hi0 + p * 64 + pf_dist / 2, SV_PLDL1KEEP);
                svprfb(pb, hi1 + p * 64 + pf_dist / 2, SV_PLDL1KEEP);
            }
        }
        for (int k = 0; k < 2; k++) {
            size_t q = p + k;
            const int8_t *aq = a->q + q * aqs;
            svint8_t A0 = rep8(aq), A1 = rep8(aq + 8), A2 = rep8(aq + 16), A3 = rep8(aq + 24);
            svint8_t B0 = A0, B1 = A1, B2 = A2, B3 = A3;
            if (a16) { B0 = rep8(aq + 32); B1 = rep8(aq + 40); B2 = rep8(aq + 48); B3 = rep8(aq + 56); }
            svfloat32_t asc = rep2(a->sc + 2 * q);
            svint8_t l0, h0, l1, h1, m0, j0, m1, j1;
            dec(fmt, lut, g0 + q * 128, hi0 + q * 64, &l0, &h0);
            dec(fmt, lut, g0 + q * 128 + 64, hi0 + q * 64 + 32, &l1, &h1);
            dec(fmt, lut, g1 + q * 128, hi1 + q * 64, &m0, &j0);
            dec(fmt, lut, g1 + q * 128 + 64, hi1 + q * 64 + 32, &m1, &j1);
            svint32_t d0 = pdot(a16, l0, h0, l1, h1, A0, A1, A2, A3, B0, B1, B2, B3);
            svint32_t d1 = pdot(a16, m0, j0, m1, j1, A0, A1, A2, A3, B0, B1, B2, B3);
            svfloat32_t s0 = pscale(fmt, sc0, q, asc), s1 = pscale(fmt, sc1, q, asc);
            if (k) {
                a01 = svmla_f32_x(pf, a01, svcvt_f32_s32_x(pf, d0), s0);
                a11 = svmla_f32_x(pf, a11, svcvt_f32_s32_x(pf, d1), s1);
            } else {
                a00 = svmla_f32_x(pf, a00, svcvt_f32_s32_x(pf, d0), s0);
                a10 = svmla_f32_x(pf, a10, svcvt_f32_s32_x(pf, d1), s1);
            }
        }
    }
    finish(out, svadd_f32_x(pf, a00, a01), fmt);
    finish(out + 8, svadd_f32_x(pf, a10, a11), fmt);
}


static int variant, fmt, arith, rows, cols, cores, passes;
static size_t gb;
static uint8_t *seg[4];
static int seg_groups[4];
static q38d_act act[4];
static _Atomic int bar_count, bar_sense;
typedef struct { int id; uint64_t t0, t1; float sink; } worker;
static worker W[48];
static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0,cntvct_el0" : "=r"(v)); return v; }
static uint64_t freq(void) { uint64_t v; __asm__ volatile("mrs %0,cntfrq_el0" : "=r"(v)); return v; }
static void barrier(int *local) {
    int s = !*local; *local = s;
    if (atomic_fetch_add_explicit(&bar_count, 1, memory_order_acq_rel) == cores - 1) {
        atomic_store_explicit(&bar_count, 0, memory_order_relaxed);
        atomic_store_explicit(&bar_sense, s, memory_order_release);
    } else while (atomic_load_explicit(&bar_sense, memory_order_acquire) != s) __asm__ volatile("yield");
}
static void pin(int cpu) { cpu_set_t m; CPU_ZERO(&m); CPU_SET(cpu, &m); sched_setaffinity(0, sizeof(m), &m); }

static void run_range(float *y, const uint8_t *base, const q38d_act *a, int g0, int g1) {
    int a16 = arith == 16;
    for (int g = g0; g < g1; ) {
        const uint8_t *w = base + (size_t)g * gb;
        if (variant >= 3 && variant <= 4 && g + 1 < g1) {
            int pd = variant == 4 ? pf1 : 0;
            if (fmt == Q38D_F4) { if (a16) v3_groups(y + g * 8, w, w + gb, Q38D_F4, 1, a, pd); else v3_groups(y + g * 8, w, w + gb, Q38D_F4, 0, a, pd); }
            else { if (a16) v3_groups(y + g * 8, w, w + gb, Q38D_F6, 1, a, pd); else v3_groups(y + g * 8, w, w + gb, Q38D_F6, 0, a, pd); }
            g += 2; continue;
        }
        if (variant >= 6) {
            q38d_asm_variant = variant == 8 ? 4 : variant == 7 ? 2 : variant == 9 ? 5 : variant == 10 ? 6 : variant == 11 ? 7 : 1;
            q38d_group_asm(y + g * 8, w, fmt, arith, a, 0, 8);
            g++; continue;
        }
        if (variant == 5) {
            if (fmt == Q38D_F4) { if (a16) q38d_group_pipe(y + g * 8, w, 0, 1, a, 0, 8, pf2); else q38d_group_pipe(y + g * 8, w, 0, 0, a, 0, 8, pf2); }
            else { if (a16) q38d_group_pipe(y + g * 8, w, 1, 1, a, 0, 8, pf2); else q38d_group_pipe(y + g * 8, w, 1, 0, a, 0, 8, pf2); }
            g++; continue;
        }
        if (variant >= 2) {
            if (fmt == Q38D_F4) { if (a16) v2_group(y + g * 8, w, Q38D_F4, 1, a); else v2_group(y + g * 8, w, Q38D_F4, 0, a); }
            else { if (a16) v2_group(y + g * 8, w, Q38D_F6, 1, a); else v2_group(y + g * 8, w, Q38D_F6, 0, a); }
        } else q38d_gemv(y + g * 8, w, fmt, a, 0, 1, 8, 0);
        g++;
    }
}

static void *run(void *arg) {
    worker *w = arg;
    int per = cores >= 12 ? 12 : cores, c = w->id / per, lane = w->id % per;
    pin(12 + c * 12 + lane);
    int G = seg_groups[c], g0 = G * lane / per, g1 = G * (lane + 1) / per;
    float *y = aligned_alloc(256, (size_t)(G + 2) * 8 * sizeof(float));
    int local = 0;
    for (int rep = -2; rep < passes; rep++) {
        if (rep == 0) { barrier(&local); w->t0 = ticks(); }
        run_range(y, seg[c], &act[c], g0, g1);
        if (!nosync || rep == passes - 1) barrier(&local);
    }
    w->t1 = ticks();
    w->sink = y[0];
    /* correctness spot check against the baseline kernel */
    if (lane == 0 && c == 0) {
        float ref[16];
        q38d_gemv(ref, seg[c], fmt, &act[c], g0, g0 + 2, (g0 + 2) * 8, 0);
        for (int i = 0; i < 16; i++)
            if (fabsf(ref[i] - y[g0 * 8 + i]) > 1e-4f * (1 + fabsf(ref[i])))
                { fprintf(stderr, "MISMATCH row %d: %g vs %g\n", i, y[g0 * 8 + i], ref[i]); break; }
    }
    free(y);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s VAR FMT ARITH ROWS COLS CORES PASSES\n", argv[0]); return 2; }
    nosync = getenv("NOSYNC") ? atoi(getenv("NOSYNC")) : 0;
    if (getenv("PF1")) pf1 = atoi(getenv("PF1"));
    if (getenv("PF2")) pf2 = atoi(getenv("PF2"));
    variant = atoi(argv[1]); fmt = atoi(argv[2]); arith = atoi(argv[3]); rows = atoi(argv[4]);
    cols = atoi(argv[5]); cores = atoi(argv[6]); passes = atoi(argv[7]);
    int cmgs = cores > 12 ? 4 : 1, groups = rows / 8;
    gb = q38d_group_bytes(fmt, cols);
    float *x = malloc((size_t)cols * 4);
    for (int k = 0; k < cols; k++) x[k] = (float)((k * 37) % 101 - 50) * 0.01f;
    for (int c = 0; c < cmgs; c++) {
        pin(12 + 12 * c);
        seg_groups[c] = groups * (c + 1) / cmgs - groups * c / cmgs;
        size_t bytes = (size_t)seg_groups[c] * gb, alloc = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
        seg[c] = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        unsigned long node = 1ul << (4 + c);
        if (syscall(SYS_mbind, seg[c], alloc, 2, &node, 64ul, 0ul)) perror("mbind");
        uint64_t r = 0x1234567 + c;
        for (size_t i = 0; i < bytes; i += 8) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; memcpy(seg[c] + i, &r, 8); }
        size_t off = (size_t)(q38d_scale_stream(seg[c], fmt, cols) - seg[c]);
        for (int g = 0; g < seg_groups[c]; g++) {
            uint8_t *s = seg[c] + (size_t)g * gb + off;
            for (size_t i = 0; i < gb - off; i++) s[i] = fmt == Q38D_F4 ? 40 + (s[i] & 31) : 118 + (s[i] & 7);
        }
        act[c].cols = cols; act[c].arith = arith;
        act[c].q = aligned_alloc(256, q38d_act_qbytes(cols, arith) + 256);
        act[c].sc = aligned_alloc(256, (size_t)cols / 16 * 4 + 256);
        act[c].sum = aligned_alloc(256, (size_t)cols / 16 * 4 + 256);
        q38d_prepare_sve(&act[c], x, 1.f, NULL);
    }
    pthread_t t[48];
    for (int i = 0; i < cores; i++) W[i].id = i;
    for (int i = 1; i < cores; i++) pthread_create(&t[i], NULL, run, &W[i]);
    run(&W[0]);
    for (int i = 1; i < cores; i++) pthread_join(t[i], NULL);
    uint64_t t0 = UINT64_MAX, t1 = 0;
    for (int i = 0; i < cores; i++) { if (W[i].t0 < t0) t0 = W[i].t0; if (W[i].t1 > t1) t1 = W[i].t1; }
    double sec = (double)(t1 - t0) / freq();
    double bytes = (double)(cores < 12 ? seg_groups[0] : groups) * gb * passes;
    printf("VAR%d fmt=%d a%d rows=%d cols=%d cores=%d us/pass=%.2f GBps=%.1f cyc/pair/core=%.1f\n",
           variant, fmt, arith, rows, cols, cores, sec / passes * 1e6, bytes / sec * 1e-9,
           sec * 2e9 * (cores < 12 ? cores : 12) / (bytes / q38d_pair_bytes(fmt) / (cmgs)));
    return 0;
}
