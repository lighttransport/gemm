/* PMU accounting while 12 workers of CMG 0 stream a large matrix (worker 0 reports).
 * usage: pmu_stream FMT(1|2) ARITH VARIANT ROWS COLS */
#define _GNU_SOURCE
#include "q38d_kern.h"
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
static int fmt, arith, rows, cols;
static uint8_t *W; static size_t gb; static q38d_act A;
static pthread_barrier_t bar;
static unsigned long ev[8] = {0x11, 0x08, 0x18a, 0x184, 0x182, 0x180, 0x190, 0x1a4};
static const char *nm[8] = {"cycles", "inst", "FL_COMP_WAIT", "LD_COMP_WAIT", "LD_WAIT_L1MISS", "LD_WAIT_L2MISS", "0INST_COMMIT", "FLA_VAL"};
static int open_raw(unsigned long cfg, int group) {
    struct perf_event_attr a; memset(&a, 0, sizeof a);
    a.size = sizeof a; a.type = PERF_TYPE_RAW; a.config = cfg;
    a.disabled = group < 0; a.exclude_kernel = 1; a.read_format = PERF_FORMAT_GROUP;
    return (int)syscall(SYS_perf_event_open, &a, 0, -1, group, 0);
}
static void *run(void *arg) {
    int id = (int)(intptr_t)arg;
    cpu_set_t m; CPU_ZERO(&m); CPU_SET(12 + id, &m); sched_setaffinity(0, sizeof m, &m);
    int G = rows / 8, g0 = G * id / 12, g1 = G * (id + 1) / 12;
    float *y = aligned_alloc(256, (size_t)(G + 1) * 32);
    int fd[8], lead = -1;
    if (!id) for (int i = 0; i < 8; i++) { fd[i] = open_raw(ev[i], lead); if (!i) lead = fd[i]; }
    for (int rep = 0; rep < 2; rep++) q38d_gemv_any(y, W, fmt, &A, g0, g1, rows, 0);
    pthread_barrier_wait(&bar);
    if (!id) { ioctl(lead, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP); ioctl(lead, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP); }
    int reps = 5;
    for (int rep = 0; rep < reps; rep++) q38d_gemv_any(y, W, fmt, &A, g0, g1, rows, 0);
    if (!id) {
        ioctl(lead, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        uint64_t buf[9]; if (read(lead, buf, sizeof buf) < 0) return NULL;
        double pairs = (double)reps * (g1 - g0) * (cols / 32);
        printf("stream fmt=%d a%d per pair:", fmt, arith);
        for (int i = 0; i < 8; i++) printf(" %s=%.2f", nm[i], buf[1 + i] / pairs);
        printf("  => %.1f GB/s/core\n", (double)q38d_pair_bytes(fmt) * 2e9 / (buf[1] / pairs) * 1e-9);
    }
    return NULL;
}
int main(int argc, char **argv) {
    fmt = atoi(argv[1]); arith = atoi(argv[2]); q38d_asm_variant = atoi(argv[3]); rows = atoi(argv[4]); cols = atoi(argv[5]);
    gb = q38d_group_bytes(fmt, cols);
    size_t bytes = (size_t)(rows / 8) * gb, alloc = (bytes + (2u << 20)) & ~(size_t)((2u << 20) - 1);
    cpu_set_t m; CPU_ZERO(&m); CPU_SET(12, &m); sched_setaffinity(0, sizeof m, &m);
    W = mmap(NULL, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    unsigned long node = 1ul << 4; syscall(SYS_mbind, W, alloc, 2, &node, 64ul, 0ul);
    uint64_t r = 88172645463325252ull;
    for (size_t i = 0; i < bytes; i += 8) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; memcpy(W + i, &r, 8); }
    size_t off = (size_t)(q38d_scale_stream(W, fmt, cols) - W);
    for (int g = 0; g < rows / 8; g++) for (size_t i = off; i < gb; i++) W[(size_t)g * gb + i] = fmt == 1 ? 100 + (W[(size_t)g * gb + i] & 31) : 118 + (W[(size_t)g * gb + i] & 7);
    float *x = malloc(cols * 4); for (int i = 0; i < cols; i++) x[i] = (float)((i * 37) % 101 - 50) * 0.01f;
    A = (q38d_act){cols, arith, aligned_alloc(256, q38d_act_qbytes(cols, 16)), aligned_alloc(256, cols / 16 * 4), aligned_alloc(256, cols / 16 * 4), x};
    q38d_prepare_sve(&A, x, 1.f, NULL);
    pthread_barrier_init(&bar, NULL, 12);
    pthread_t t[12];
    for (int i = 1; i < 12; i++) pthread_create(&t[i], NULL, run, (void *)(intptr_t)i);
    run((void *)0);
    for (int i = 1; i < 12; i++) pthread_join(t[i], NULL);
    return 0;
}
