#define _POSIX_C_SOURCE 200809L
#include "glm53f_mla_softmax.h"
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HEADS = 6, STRIDE = 2052, GUARD = 16, ELEMENTS = HEADS * STRIDE + GUARD };
static void legacy(float *a, float *sum, int heads, int tokens) {
#pragma omp for schedule(static)
    for (int h = 0; h < heads; ++h) {
        float *l = a + (size_t)h * STRIDE, mx = -INFINITY, s = 0;
        for (int t = 0; t < tokens; ++t) if (l[t] > mx) mx = l[t];
        for (int t = 0; t < tokens; ++t) { l[t] = expf(l[t] - mx); s += l[t]; }
        sum[h] = s;
    }
}
static uint32_t next(uint32_t *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return *seed;
}
static void fill(float *a, int distribution) {
    uint32_t seed = 71 + distribution;
    for (int i = 0; i < ELEMENTS; ++i) {
        const uint32_t r = next(&seed);
        const float x = ((int)(r % 20001) - 10000) * .0001f;
        switch (distribution) {
        case 0: a[i] = x; break;
        case 1: a[i] = x * 100; break;
        case 2: a[i] = x * .00001f; break;
        case 3: a[i] = 0; break;
        case 4: a[i] = (i % 17) ? -100 : 1; break;
        case 5: a[i] = -1000 + x; break;
        case 6: a[i] = 1000 + x; break;
        case 7: a[i] = (float)(i % 7); break;
        default: a[i] = -0.0f; break;
        }
    }
}
static int finite_bits(const float *a, int count) {
    for (int i = 0; i < count; ++i) {
        uint32_t bits; memcpy(&bits, a + i, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) return 0;
    }
    return 1;
}
static void restore(float *out, const float *src) {
#pragma omp for schedule(static)
    for (int i = 0; i < ELEMENTS; ++i) out[i] = src[i];
}
static double measure(int mode, int heads, float *a, const float *src,
        float *sum, float *mx) {
    double begin = 0, end = 0;
#pragma omp parallel shared(begin, end)
    {
#pragma omp master
        begin = omp_get_wtime();
#pragma omp barrier
        for (int rep = 0; rep < 90; ++rep) {
            restore(a, src);
            if (mode) glm53f_mla_softmax_parallel(a, sum, mx, heads, 2048, STRIDE);
            else legacy(a, sum, heads, 2048);
        }
#pragma omp master
        end = omp_get_wtime();
    }
    return (end - begin) * 1e6 / 90;
}
static double median(double *a) {
    for (int i = 1; i < 7; ++i) {
        double x = a[i]; int j = i;
        while (j && a[j - 1] > x) { a[j] = a[j - 1]; --j; }
        a[j] = x;
    }
    return a[3];
}
int main(int argc, char **argv) {
    float *src = NULL, *a = NULL, *b = NULL;
    if (posix_memalign((void **)&src, 256, ELEMENTS * sizeof(float)) ||
        posix_memalign((void **)&a, 256, ELEMENTS * sizeof(float)) ||
        posix_memalign((void **)&b, 256, ELEMENTS * sizeof(float))) return 2;
    const int counts[] = {1,2,3,7,15,16,17,31,32,33,47,48,49,63,64,65,
        127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048,2049,2051,2052};
    int bad = 0, cases = 0;
    float sa[HEADS + GUARD], sb[HEADS + GUARD], maxima[HEADS + GUARD];
    for (int dist = 0; dist < 9; ++dist) {
        fill(src, dist);
        for (int heads = 1; heads <= HEADS; ++heads)
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ++ci) {
                memcpy(a, src, ELEMENTS * sizeof(float)); memcpy(b, src, ELEMENTS * sizeof(float));
                memset(sa, 0xa5, sizeof(sa)); memset(sb, 0xa5, sizeof(sb));
                memset(maxima, 0xa5, sizeof(maxima));
#pragma omp parallel
                {
                    legacy(a, sa, heads, counts[ci]);
                    glm53f_mla_softmax_parallel(b, sb, maxima, heads, counts[ci], STRIDE);
                }
                const int expbad = memcmp(a, b, ELEMENTS * sizeof(float)) != 0;
                const int sumbad = memcmp(sa, sb, sizeof(sa)) != 0;
                unsigned char guard[GUARD * sizeof(float)]; memset(guard, 0xa5, sizeof(guard));
                const int guardbad = memcmp(maxima + heads, guard, sizeof(guard)) != 0;
                const int finite = finite_bits(sb, heads);
                if ((expbad || sumbad || guardbad || !finite) && bad < 12)
                    printf("MLA_SOFTMAX_MISMATCH dist=%d heads=%d tokens=%d exp=%d sum=%d guard=%d finite=%d\n",
                        dist, heads, counts[ci], expbad, sumbad, guardbad, finite);
                bad += expbad || sumbad || guardbad || !finite;
                ++cases;
            }
    }
    printf("MLA_SOFTMAX_EXACT cases=%d threads=%d mismatches=%d %s\n",
        cases, omp_get_max_threads(), bad, bad ? "FAIL" : "PASS");
    if (!bad && argc > 1 && !strcmp(argv[1], "--bench")) {
        fill(src, 0);
        for (int heads = 5; heads <= 6; ++heads) {
            double times[2][7];
            for (int trial = -1; trial < 7; ++trial)
                for (int order = 0; order < 2; ++order) {
                    int mode = (trial + 1 + order) & 1;
                    const double us = measure(mode, heads, mode ? b : a, src,
                        mode ? sb : sa, maxima);
                    if (trial >= 0) times[mode][trial] = us;
                }
            double old = median(times[0]), candidate = median(times[1]);
            bad |= memcmp(a, b, ELEMENTS * sizeof(float)) || memcmp(sa, sb, heads * sizeof(float));
            printf("MLA_SOFTMAX_TIMING heads=%d threads=%d legacy_us=%.6f parallel_us=%.6f ratio=%.6f %s\n",
                heads, omp_get_max_threads(), old, candidate, old / candidate, bad ? "FAIL" : "PASS");
        }
    }
    free(b); free(a); free(src);
    return bad ? 1 : 0;
}
