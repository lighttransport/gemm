#define _POSIX_C_SOURCE 200809L
#include "glm53f_index_score.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    enum { POOLS = 2053, STRIDE = 12 };
    float q[4096], weight[32];
    float *keys = malloc((size_t)POOLS * STRIDE * 128 * sizeof(float));
    float *expected = malloc(POOLS * sizeof(float));
    float *actual = malloc((POOLS + 4) * sizeof(float));
    uint32_t seed = 73;
    if (!keys || !expected || !actual) return 2;
    for (int i = 0; i < 4096; ++i) {
        seed = seed * 1664525u + 1013904223u;
        q[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f;
    }
    for (int h = 0; h < 32; ++h) weight[h] = (h - 11) * .01f;
    for (int i = 0; i < POOLS * STRIDE * 128; ++i) {
        seed = seed * 1664525u + 1013904223u;
        keys[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f;
    }
    memset(keys, 0, 128 * sizeof(float));
    int bad = 0, cases = 0;
    for (int stride = 1; stride <= STRIDE; stride += STRIDE - 1) {
        size_t step = (size_t)stride * 128;
        /* Exercise every partial group and verify exclusive output bounds. */
        for (int n = 1; n <= 4; ++n) {
            float result[5] = {0, 0, 0, 0, 123.25f};
            for (int i = n; i < 5; ++i) result[i] = 123.25f;
            glm53f_index_score_f32_keys4(result, q, weight, keys, step, n);
            for (int k = 0; k < n; ++k) {
                float ref = glm53f_index_score_f32_heads(q, weight, keys + k * step);
                bad |= memcmp(&ref, result + k, sizeof(float)) != 0;
            }
            for (int k = n; k < 5; ++k) bad |= result[k] != 123.25f;
            ++cases;
        }
        double old_time = 1e30, new_time = 1e30;
        for (int rep = -1; rep < 3; ++rep) {
            double begin = seconds();
#pragma omp parallel for schedule(static)
            for (int p = 0; p < POOLS; ++p)
                expected[p] = glm53f_index_score_f32_heads(q, weight, keys + p * step);
            double dt = seconds() - begin;
            if (rep >= 0 && dt < old_time) old_time = dt;
            begin = seconds();
#pragma omp parallel for schedule(static)
            for (int group = 0; group < (POOLS + 3) / 4; ++group) {
                int first = group * 4, n = POOLS - first;
                if (n > 4) n = 4;
                glm53f_index_score_f32_keys4(actual + first, q, weight,
                    keys + first * step, step, n);
            }
            dt = seconds() - begin;
            if (rep >= 0 && dt < new_time) new_time = dt;
            bad |= memcmp(expected, actual, POOLS * sizeof(float)) != 0;
        }
        printf("INDEX_KEYS4 stride=%d pools=%d heads_ms=%.6f keys4_ms=%.6f speedup=%.3f %s\n",
            stride, POOLS, old_time * 1e3, new_time * 1e3, old_time / new_time,
            bad ? "FAIL" : "PASS");
    }
    printf("INDEX_KEYS4_BOUNDARY cases=%d %s\n", cases, bad ? "FAIL" : "PASS");
    free(actual); free(expected); free(keys);
    return bad;
}
