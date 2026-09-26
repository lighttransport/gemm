/* Test Q38D's uTofu transport independently of model sharding and kernels.
 * Launch one rank per node after creating TOPO_PATH with tofu_topo_helper:
 *   Q38D_TP=9 mpiexec -n 9 ./test_q38d_tp_collective 9 5120
 */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <utofu.h>

#include "q38d_tp.h"

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 9;
    int count = argc > 2 ? atoi(argv[2]) : 5120;
    if (n < 2 || n > Q38D_TP_MAXN || count < 16 || count % 16) {
        fprintf(stderr, "usage: %s ranks count(positive multiple of 16)\n", argv[0]);
        return 2;
    }
    q38d_tp_init(n, count);
    float *part = aligned_alloc(256, (size_t)count * sizeof(*part));
    float *residual = aligned_alloc(256, (size_t)count * sizeof(*residual));
    if (!part || !residual) {
        fprintf(stderr, "rank %d: allocation failed\n", tp_r);
        return 1;
    }
    const int iterations = 100;
    const float expected = (float)(n * (n + 1) / 2) * 0.125f;
    double t0 = tp_ar_now();
    for (int it = 0; it < iterations; it++) {
        for (int i = 0; i < count; i++) {
            part[i] = (float)(tp_r + 1) * 0.125f;
            residual[i] = 0.0f;
        }
        q38d_tp_sum_add(part, residual, count);
        for (int i = 0; i < count; i++) {
            if (residual[i] != expected) {
                fprintf(stderr, "rank %d iteration %d index %d: got %.9g expected %.9g\n",
                        tp_r, it, i, residual[i], expected);
                return 1;
            }
        }
    }
    double usec = (tp_ar_now() - t0) * 1e6 / iterations;
    if (tp_r == 0)
        printf("q38d_tp_collective: ranks=%d count=%d iterations=%d avg_us=%.2f PASS\n",
               n, count, iterations, usec);
    return 0;
}
