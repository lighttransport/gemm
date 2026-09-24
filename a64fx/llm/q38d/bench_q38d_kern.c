/* Native bandwidth benchmark for q38d pair-interleaved kernels.
 * usage: bench FMT(1=F4,2=F6,3=Q6K,4=Q4K,0=scan) ARITH(8|16) ROWS COLS CORES PASSES [sync]
 * Each CMG c owns rows [c/4, (c+1)/4) of the matrix, allocated on NUMA node
 * 4+c by the first thread of the CMG. With sync=1 all workers meet at a
 * spin barrier after every pass, as in decode. */
#define _GNU_SOURCE
#include "q38d_kern.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0,cntvct_el0" : "=r"(v)); return v; }
static uint64_t freq(void) { uint64_t v; __asm__ volatile("mrs %0,cntfrq_el0" : "=r"(v)); return v; }

static int fmt, arith, rows, cols, cores, passes, sync_mode;
static size_t gb;
static uint8_t *seg[4];
static int seg_groups[4];
static q38d_act act[4];
static float *xin;
static _Atomic int bar_count;
static _Atomic int bar_sense;
typedef struct { int id; uint64_t t0, t1; float sink; } worker;
static worker W[48];

static void barrier(int *local) {
    int s = !*local;
    *local = s;
    if (atomic_fetch_add_explicit(&bar_count, 1, memory_order_acq_rel) == cores - 1) {
        atomic_store_explicit(&bar_count, 0, memory_order_relaxed);
        atomic_store_explicit(&bar_sense, s, memory_order_release);
    } else
        while (atomic_load_explicit(&bar_sense, memory_order_acquire) != s) __asm__ volatile("yield");
}
static void pin(int cpu) { cpu_set_t m; CPU_ZERO(&m); CPU_SET(cpu, &m); sched_setaffinity(0, sizeof(m), &m); }

static void *run(void *arg) {
    worker *w = arg;
    int per = cores >= 12 ? 12 : cores;
    int c = w->id / per, lane = w->id % per;
    pin(12 + c * 12 + lane);
    int G = seg_groups[c];
    int g0 = G * lane / per, g1 = G * (lane + 1) / per;
    float *y = aligned_alloc(256, (size_t)(G + 1) * 8 * sizeof(float));
    int local = 0;
    svuint8_t s = svdup_n_u8(0);
    for (int rep = -2; rep < passes; rep++) {
        if (rep == 0) { barrier(&local); w->t0 = ticks(); }
        if (fmt) q38d_gemv(y, seg[c], fmt, &act[c], g0, g1, G * 8, 0);
        else {
            const uint8_t *p = seg[c] + (size_t)g0 * gb, *e = seg[c] + (size_t)g1 * gb;
            for (; p + 256 <= e; p += 256) {
                s = sveor_u8_x(svptrue_b8(), s, svld1_u8(svptrue_b8(), p));
                s = sveor_u8_x(svptrue_b8(), s, svld1_u8(svptrue_b8(), p + 64));
                s = sveor_u8_x(svptrue_b8(), s, svld1_u8(svptrue_b8(), p + 128));
                s = sveor_u8_x(svptrue_b8(), s, svld1_u8(svptrue_b8(), p + 192));
            }
        }
        if (sync_mode) barrier(&local);
    }
    w->t1 = ticks();
    w->sink = y[0] + (float)svorv_u8(svptrue_b8(), s);
    free(y);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s FMT ARITH ROWS COLS CORES PASSES [sync]\n", argv[0]);
        return 2;
    }
    fmt = atoi(argv[1]); arith = atoi(argv[2]); rows = atoi(argv[3]); cols = atoi(argv[4]);
    cores = atoi(argv[5]); passes = atoi(argv[6]); sync_mode = argc > 7 ? atoi(argv[7]) : 1;
    int cmgs = cores > 12 ? 4 : 1;
    gb = q38d_group_bytes(fmt ? fmt : Q38D_F4, cols);
    int groups = rows / 8;
    xin = malloc((size_t)cols * 4);
    for (int k = 0; k < cols; k++) xin[k] = (float)((k * 37) % 101 - 50) * 0.01f;
    for (int c = 0; c < cmgs; c++) {
        pin(12 + 12 * c);
        seg_groups[c] = groups * (c + 1) / cmgs - groups * c / cmgs;
        size_t bytes = (size_t)seg_groups[c] * gb, alloc = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
        seg[c] = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        unsigned long node = 1ul << (4 + c);
        if (syscall(SYS_mbind, seg[c], alloc, 2, &node, 64ul, 0ul)) perror("mbind");
        uint64_t r = 0x1234567 + c;
        for (size_t i = 0; i < bytes; i += 8) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; memcpy(seg[c] + i, &r, 8); }
        if (fmt) {
            uint8_t *sc0 = q38d_scale_stream(seg[c], fmt, cols);
            size_t off = (size_t)(sc0 - seg[c]);
            for (int g = 0; g < seg_groups[c]; g++) {
                uint8_t *s = seg[c] + (size_t)g * gb + off;
                size_t n = gb - off;
                if (fmt == Q38D_F4) for (size_t i = 0; i < n; i++) s[i] = 40 + (s[i] & 31);
                if (fmt == Q38D_F6) for (size_t i = 0; i < n; i++) s[i] = 118 + (s[i] & 7);
                if (fmt == Q38D_Q6K) {
                    float *d = (float *)(s + (size_t)cols / 32 * 16);
                    for (int i = 0; i < cols / 256 * 8; i++) d[i] = 1e-3f;
                }
                if (fmt == Q38D_Q4K) { float *f = (float *)s; for (size_t i = 0; i < n / 4; i++) f[i] = 1e-2f; }
            }
            act[c].cols = cols; act[c].arith = arith;
            act[c].q = aligned_alloc(256, q38d_act_qbytes(cols, arith) + 256);
            act[c].sc = aligned_alloc(256, (size_t)cols / 16 * 4 + 256);
            act[c].sum = aligned_alloc(256, (size_t)cols / 16 * 4 + 256);
            q38d_prepare_sve(&act[c], xin, 1.f, NULL);
        }
        int node_seen = -1;
        syscall(SYS_get_mempolicy, &node_seen, NULL, 0ul, seg[c], 3ul);
        if (node_seen != 4 + c) fprintf(stderr, "placement: cmg %d on node %d\n", c, node_seen);
    }
    pthread_t t[48];
    for (int i = 0; i < cores; i++) W[i].id = i;
    for (int i = 1; i < cores; i++) pthread_create(&t[i], NULL, run, &W[i]);
    run(&W[0]);
    for (int i = 1; i < cores; i++) pthread_join(t[i], NULL);
    uint64_t t0 = UINT64_MAX, t1 = 0;
    for (int i = 0; i < cores; i++) { if (W[i].t0 < t0) t0 = W[i].t0; if (W[i].t1 > t1) t1 = W[i].t1; }
    double sec = (double)(t1 - t0) / freq();
    double bytes = (double)groups * gb * passes;
    if (cores < 12) bytes = (double)seg_groups[0] * gb * passes;
    printf("Q38D fmt=%d arith=%d rows=%d cols=%d cores=%d passes=%d sync=%d us_per_pass=%.2f GBps=%.1f\n",
           fmt, arith, rows, cols, cores, passes, sync_mode, sec / passes * 1e6, bytes / sec * 1e-9);
    return 0;
}
