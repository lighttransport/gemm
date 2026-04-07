#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#define BUF_SIZE (256 * 1024 * 1024)
#define N_ITERS 5

typedef struct {
    void *buf_a, *buf_b;
    size_t size;
    int tid;
    double read_bw, write_bw, copy_bw;
} bw_task;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void stream_read(const void *buf, size_t size) {
#ifdef __ARM_FEATURE_SVE
    const uint8_t *p = (const uint8_t *)buf;
    svbool_t pg = svptrue_b8();
    svuint8_t sink = svdup_u8(0);
    size_t vl = svcntb();
    for (size_t i = 0; i < size; i += vl * 4) {
        sink = svorr_x(pg, sink, svld1(pg, p + i));
        sink = svorr_x(pg, sink, svld1(pg, p + i + vl));
        sink = svorr_x(pg, sink, svld1(pg, p + i + vl * 2));
        sink = svorr_x(pg, sink, svld1(pg, p + i + vl * 3));
    }
    volatile uint8_t v = svlastb(pg, sink); (void)v;
#else
    volatile uint64_t s = 0;
    const uint64_t *p = (const uint64_t *)buf;
    for (size_t i = 0; i < size/8; i += 4) s += p[i]+p[i+1]+p[i+2]+p[i+3];
#endif
}

static void stream_write(void *buf, size_t size) {
#ifdef __ARM_FEATURE_SVE
    uint8_t *p = (uint8_t *)buf;
    svbool_t pg = svptrue_b8();
    svuint8_t zero = svdup_u8(0);
    size_t vl = svcntb();
    for (size_t i = 0; i < size; i += vl * 4) {
        svst1(pg, p + i, zero);
        svst1(pg, p + i + vl, zero);
        svst1(pg, p + i + vl * 2, zero);
        svst1(pg, p + i + vl * 3, zero);
    }
#else
    memset(buf, 0, size);
#endif
}

static void stream_copy(void *dst, const void *src, size_t size) {
#ifdef __ARM_FEATURE_SVE
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    svbool_t pg = svptrue_b8();
    size_t vl = svcntb();
    for (size_t i = 0; i < size; i += vl * 4) {
        svst1(pg, d+i,       svld1(pg, s+i));
        svst1(pg, d+i+vl,   svld1(pg, s+i+vl));
        svst1(pg, d+i+vl*2, svld1(pg, s+i+vl*2));
        svst1(pg, d+i+vl*3, svld1(pg, s+i+vl*3));
    }
#else
    memcpy(dst, src, size);
#endif
}

static void *bw_worker(void *arg) {
    bw_task *t = (bw_task *)arg;
    size_t sz = t->size;

    /* NUMA first-touch: each thread touches its OWN buffers */
    memset(t->buf_a, 1, sz);
    memset(t->buf_b, 0, sz);

    /* Warmup */
    stream_read(t->buf_a, sz);
    stream_write(t->buf_b, sz);

    double t0 = now_sec();
    for (int i = 0; i < N_ITERS; i++) stream_read(t->buf_a, sz);
    double t1 = now_sec();
    t->read_bw = (double)sz * N_ITERS / (t1-t0) / 1e9;

    t0 = now_sec();
    for (int i = 0; i < N_ITERS; i++) stream_write(t->buf_b, sz);
    t1 = now_sec();
    t->write_bw = (double)sz * N_ITERS / (t1-t0) / 1e9;

    t0 = now_sec();
    for (int i = 0; i < N_ITERS; i++) stream_copy(t->buf_b, t->buf_a, sz);
    t1 = now_sec();
    t->copy_bw = (double)sz * 2 * N_ITERS / (t1-t0) / 1e9;

    return NULL;
}

int main(int argc, char **argv) {
    int nt = (argc >= 2) ? atoi(argv[1]) : 1;
    fprintf(stderr, "A64FX HBM2 BW (NUMA-local): %d threads, %d MB/thread\n\n", nt, BUF_SIZE/(1024*1024));

    pthread_t *thr = malloc(nt * sizeof(pthread_t));
    bw_task *tasks = malloc(nt * sizeof(bw_task));
    for (int t = 0; t < nt; t++) {
        /* Allocate but DON'T touch — let worker thread first-touch for NUMA */
        posix_memalign(&tasks[t].buf_a, 2*1024*1024, BUF_SIZE);
        posix_memalign(&tasks[t].buf_b, 2*1024*1024, BUF_SIZE);
        tasks[t].size = BUF_SIZE; tasks[t].tid = t;
    }
    for (int t = 0; t < nt; t++) pthread_create(&thr[t], NULL, bw_worker, &tasks[t]);
    for (int t = 0; t < nt; t++) pthread_join(thr[t], NULL);

    double tr=0, tw=0, tc=0;
    for (int t = 0; t < nt; t++) { tr+=tasks[t].read_bw; tw+=tasks[t].write_bw; tc+=tasks[t].copy_bw; }
    fprintf(stderr, "Aggregate  Read: %.1f GB/s  Write: %.1f GB/s  Copy: %.1f GB/s\n", tr, tw, tc);
    int n_cmgs = 4;
    if (nt < n_cmgs) n_cmgs = nt;
    fprintf(stderr, "Per-CMG    Read: %.1f GB/s  Write: %.1f GB/s  Copy: %.1f GB/s\n", tr/n_cmgs, tw/n_cmgs, tc/n_cmgs);

    for (int t = 0; t < nt; t++) { free(tasks[t].buf_a); free(tasks[t].buf_b); }
    free(thr); free(tasks);
    return 0;
}
