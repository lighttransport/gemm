#define _POSIX_C_SOURCE 200809L
#define HD 256
#include "q38d_attention.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9;
}
static uint32_t rng = 123;
static float random_value(void) { rng = rng * 1664525 + 1013904223; return (int32_t)rng * 0x1p-31f; }
void q38p_qk6x4(const float *, const float *, float *, long, float);
static void test_qk(void) {
    float q[6 * HD], k[4 * HD], sc[6 * 9];
    for (int rep = 0; rep < 32; rep++) {
        for (int i = 0; i < 6 * HD; i++) q[i] = random_value();
        for (int i = 0; i < 4 * HD; i++) k[i] = random_value();
        for (int i = 0; i < 6 * 9; i++) sc[i] = -123;
        q38p_qk6x4(q, k, sc, 9, 0.0625f);
        for (int h = 0; h < 6; h++) for (int t = 0; t < 4; t++) {
            float lanes[16] = {0};
            for (int d = 0; d < HD; d++) lanes[d % 16] = fmaf(q[h * HD + d], k[t * HD + d], lanes[d % 16]);
            for (int n = 16; n > 1; n /= 2)
                for (int j = 0; j < n / 2; j++) lanes[j] = lanes[2 * j] + lanes[2 * j + 1];
            float ref = lanes[0] * 0.0625f;
            assert(!memcmp(&ref, &sc[h * 9 + t], sizeof(float)));
        }
        for (int h = 0; h < 6; h++) for (int t = 4; t < 9; t++) assert(sc[h * 9 + t] == -123);
    }
    puts("PASS: six-head QK bit-exact lane-FMA and reduction reference, strided scores");
}
int main(void) {
    test_qk();
    const int maxn = 32768, stride = HD + 2;
    float *v = malloc((size_t)(maxn + 7) * HD * 4), *p = malloc((size_t)6 * maxn * 4);
    float a[6 * (HD + 2)], b[6 * (HD + 2)];
    for (int i = 0; i < (maxn + 7) * HD; i++) v[i] = random_value();
    for (int i = 0; i < 6 * maxn; i++) p[i] = fabsf(random_value());
    int ns[] = {0, 1, 3, 17, 257, 1031, 4097, maxn};
    for (int off = 0; off <= 7; off += 7) for (int ni = 0; ni < 8; ni++) {
        int n = ns[ni];
        memset(a, 0xa5, sizeof(a));
        q38d_attn_pv6_blocked(v, off, off + n, p, n, a, stride, 0);
        for (int block = 128; block <= 512; block *= 2) {
            memset(b, 0xa5, sizeof(b));
            q38d_attn_pv6_blocked(v, off, off + n, p, n, b, stride, block);
            assert(!memcmp(a, b, sizeof(a)));
        }
        if (n > 0) {
            memset(b, 0xa5, sizeof(b));
            for (int t = 0; t < n; t += 256) {
                int end = t + 256 < n ? t + 256 : n;
                q38d_attn_pv6_range(v, off + t, off + end, p + t, n, b, stride, 256, t == 0);
            }
            assert(!memcmp(a, b, sizeof(a)));
        }
        if (n <= 1031) for (int h = 0; h < 6; h++) for (int d = 0; d < HD; d++) {
            float ref = 0;
            for (int t = 0; t < n; t++) ref = fmaf(p[h * n + t], v[(size_t)(off + t) * HD + d], ref);
            assert(a[h * stride + d] == ref);
        }
    }
    volatile float sink = 0;
    int blocks[] = {0, 128, 256, 512};
    for (int j = 0; j < 4; j++) {
        double t = now();
        for (int rep = 0; rep < 16; rep++) {
            q38d_attn_pv6_blocked(v, 0, maxn, p, maxn, a, stride, blocks[j]); sink += a[rep];
        }
        printf("pv6 block=%d ms=%.3f\n", blocks[j], (now() - t) * 1e3 / 16);
    }
    printf("PASS: bit-exact PV tails, offset ranges, strided output; scalar FMA reference (sink=%g)\n", sink);
    free(v); free(p); return 0;
}
