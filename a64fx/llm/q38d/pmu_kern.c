/* PMU cycle accounting for the q38d assembly kernels (one core, L1-resident).
 * usage: pmu_kern FMT(1|2) ARITH(8|16) VARIANT(1|2|4) */
#define _GNU_SOURCE
#include "q38d_kern.h"
#include <linux/perf_event.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

static int open_raw(unsigned long cfg, int group) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a; a.type = PERF_TYPE_RAW; a.config = cfg;
    a.disabled = group < 0; a.exclude_kernel = 1; a.exclude_hv = 1;
    a.read_format = PERF_FORMAT_GROUP;
    return (int)syscall(SYS_perf_event_open, &a, 0, -1, group, 0);
}
int main(int argc, char **argv) {
    int fmt = argc > 1 ? atoi(argv[1]) : 1, arith = argc > 2 ? atoi(argv[2]) : 16;
    q38d_asm_variant = argc > 3 ? atoi(argv[3]) : 2;
    cpu_set_t m; CPU_ZERO(&m); CPU_SET(12, &m); sched_setaffinity(0, sizeof m, &m);
    const int cols = 5120, groups = 2;
    size_t gb = q38d_group_bytes(fmt, cols);
    uint8_t *g = aligned_alloc(256, gb * groups);
    for (size_t i = 0; i < gb * groups; i++) g[i] = (uint8_t)(i * 2654435761u >> 13);
    size_t off = (size_t)(q38d_scale_stream(g, fmt, cols) - g);
    for (int k = 0; k < groups; k++) for (size_t i = off; i < gb; i++) g[k * gb + i] = fmt == 1 ? 100 + (g[k * gb + i] & 31) : fmt == 2 ? 118 + (g[k * gb + i] & 7) : 60 + (g[k * gb + i] & 3);
    float *x = malloc(cols * 4);
    for (int i = 0; i < cols; i++) x[i] = (float)((i * 37) % 101 - 50) * 0.01f;
    q38d_act a = {cols, arith, aligned_alloc(256, q38d_act_qbytes(cols, 16)), aligned_alloc(256, cols / 16 * 4),
                  aligned_alloc(256, cols / 16 * 4), x};
    q38d_prepare_sve(&a, x, 1.f, NULL);
    unsigned long ev[8] = {0x11, 0x08, 0x1a4, 0x1a5, 0x18a, 0x184, 0x190, 0x194};
    const char *nm[8] = {"cycles", "inst", "FLA_VAL", "FLB_VAL", "FL_COMP_WAIT", "LD_COMP_WAIT", "0INST_COMMIT", "4INST_COMMIT"};
    int fd[8], lead = -1;
    for (int i = 0; i < 8; i++) {
        fd[i] = open_raw(ev[i], lead);
        if (fd[i] < 0) { perror("perf_event_open"); return 1; }
        if (i == 0) lead = fd[i];
    }
    float out[16];
    long reps = 20000;
    for (int r = 0; r < 100; r++) q38d_group_asm(out, g, fmt, arith, &a, 0, 8);
    ioctl(lead, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(lead, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    int dual = getenv("PMU_DUAL") && atoi(getenv("PMU_DUAL"));
    for (long r = 0; r < reps; r++) {
        if (dual) { float o2[16]; q38d_gemv_dual_f4(o2, o2 + 8, g, g + gb, &a, 0, 1); out[0] += o2[0]; }
        else for (int k = 0; k < groups; k++) q38d_group_asm(out + 8 * (k & 1), g + k * gb, fmt, arith, &a, 0, 8);
    }
    ioctl(lead, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    uint64_t buf[9];
    if (read(lead, buf, sizeof buf) < 0) { perror("read"); return 1; }
    double pairs = (double)reps * groups * (cols / 32);
    printf("fmt=%d a%d variant=%d per pair:", fmt, arith, q38d_asm_variant);
    for (int i = 0; i < 8; i++) printf(" %s=%.2f", nm[i], buf[1 + i] / pairs);
    printf("\n");
    return (int)out[0] & 0;
}
