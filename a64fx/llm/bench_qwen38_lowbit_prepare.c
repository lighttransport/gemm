/* Isolate serial activation preparation from the projection/barriers. */
#define _POSIX_C_SOURCE 200809L
#include "qwen38_lowbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    const int sizes[] = {5120, 6144, 17408};
    for (int si = 0; si < 3; si++) for (int arith = 8; arith <= 16; arith += 8) {
        int cols = sizes[si], passes = 1000;
        size_t count = ((size_t)cols + 15) / 16;
        float *x = malloc((size_t)cols * sizeof(*x));
        q38_lowbit_act *a = malloc(count * sizeof(*a));
        if (!x || !a) return 1;
        for (int k = 0; k < cols; k++) x[k] = (float)((k * 17) % 101 - 50) * .015625f;
        if (!q38_lowbit_prepare(a, count, x, cols, arith)) return 1;
        int variants = 1;
#ifdef __ARM_FEATURE_SVE
        variants = 2;
#endif
        for (int v = 0; v < variants; v++) {
            double t = seconds();
            for (int p = 0; p < passes; p++) {
                int ok;
#ifdef __ARM_FEATURE_SVE
                if (v) ok = q38_lowbit_prepare_sve(a, count, x, cols, arith);
                else
#endif
                    ok = q38_lowbit_prepare(a, count, x, cols, arith);
                if (!ok) return 1;
            }
            double elapsed = seconds() - t;
            printf("PREPARE variant=%s cols=%d arithmetic=%d passes=%d us=%.3f bytes=%zu checksum=%d\n",
                   v ? "sve" : "scalar", cols, arith, passes, elapsed * 1e6 / passes,
                   count * sizeof(*a), a[0].lo[0][0]);
        }
        free(a); free(x);
    }
    return 0;
}
