#define _POSIX_C_SOURCE 200809L
#include "q38d_reduce.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9;
}
int main(void) {
    float *x = malloc(32769 * sizeof(float));
    uint32_t r = 123;
    for (int i = 0; i < 32769; i++) { r = r * 1664525 + 1013904223; x[i] = (int32_t)r * 0x1p-24f; }
    for (int n = 0; n < 257; n++) assert(q38d_score_max(x, n, 0) == q38d_score_max(x, n, 1));
    for (int n = 16383; n <= 32769; n += 8193) assert(q38d_score_max(x, n, 0) == q38d_score_max(x, n, 1));
    for (int i = 0; i < 32769; i++) x[i] = NAN;
    x[1] = -INFINITY; x[32768] = 5;
    assert(q38d_score_max(x, 32769, 1) == 5);
    x[32768] = INFINITY; assert(q38d_score_max(x, 32769, 1) == INFINITY);
    x[32768] = NAN; assert(q38d_score_max(x, 32769, 1) == -INFINITY);
    for (int i = 0; i < 32769; i++) x[i] = (float)(i % 123);
    volatile float sink = 0;
    for (int v = 0; v <= 1; v++) {
        double t = now();
        for (int rep = 0; rep < 10000; rep++) { x[rep % 32769] = (float)(rep % 123); sink += q38d_score_max(x, 32769, v); }
        printf("score_max vector=%d ns/element=%.3f\n", v, (now() - t) * 1e9 / (10000.0 * 32769));
    }
    printf("PASS score maxima, tail lengths, NaNs and infinities (sink=%g)\n", sink);
    free(x); return 0;
}
