#define _POSIX_C_SOURCE 200809L
#include "glm53f_mla_absorb.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
/* Match the existing scratch-update loop, including its pointer-based sum. */
__attribute__((noinline)) static void reference(float *out, const float *z,
        const float *prob, const float *sum, const int *selected, int count, int begin) {
    for (int d = begin; d < begin + 64; ++d) out[d] = 0;
    for (int t = 0; t < count; ++t) {
        int row = selected ? selected[t] : t;
        float x = prob[t] / *sum;
        for (int d = begin; d < begin + 64; d += (int)svcntw()) {
            svbool_t p = svwhilelt_b32(d, 512);
            svst1_f32(p, out + d, svmla_n_f32_x(p, svld1_f32(p, out + d),
                svld1_f32(p, z + (size_t)row * 512 + d), x));
        }
    }
}
int main(void) {
    enum { HEADS = 6, SLOTS = 2052 };
    float *z = malloc((size_t)SLOTS * 512 * sizeof(float));
    uint16_t *half = malloc((size_t)SLOTS * 512 * sizeof(uint16_t));
    float *prob = malloc((size_t)HEADS * SLOTS * sizeof(float));
    float sum[HEADS], expected[HEADS * 512], actual[HEADS * 512];
    int selected[SLOTS];
    if (!z || !prob || !half || svcntw() != 16) return 2;
    uint32_t seed = 113;
    for (int i = 0; i < SLOTS * 512; ++i) {
        seed = seed * 1664525u + 1013904223u;
        z[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f;
    }
    glm53f_mla_cache_f16_store(half, z, SLOTS * 512);
    for (int i = 0; i < SLOTS * 512; ++i) z[i] = (float)(_Float16)z[i];
    for (int i = 0; i < HEADS * SLOTS; ++i) {
        seed = seed * 1664525u + 1013904223u;
        prob[i] = (seed % 10001 + 1) * .0001f;
    }
    for (int t = 0; t < SLOTS; ++t) selected[t] = SLOTS - 1 - t;
    const int counts[] = {1, 7, 128, 2048, 2051};
    int bad = 0;
    for (int index = 0; index < 2; ++index) for (int ci = 0; ci < 5; ++ci) {
        int count = counts[ci];
        for (int h = 0; h < HEADS; ++h) {
            sum[h] = 0;
            for (int t = 0; t < count; ++t) sum[h] += prob[h * SLOTS + t];
        }
        double old_time = 1e30, new_time = 1e30;
        for (int rep = -1; rep < 3; ++rep) {
            double start = seconds();
#pragma omp parallel for schedule(static)
            for (int task = 0; task < HEADS * 8; ++task) {
                int h = task / 8, begin = task % 8 * 64;
                reference(expected + h * 512, z, prob + h * SLOTS,
                    sum + h, index ? selected : NULL, count, begin);
            }
            double dt = seconds() - start;
            if (rep >= 0 && dt < old_time) old_time = dt;
            start = seconds();
#pragma omp parallel for schedule(static)
            for (int task = 0; task < HEADS * 8; ++task) {
                int h = task / 8, begin = task % 8 * 64;
                glm53f_mla_value64(actual + h * 512, z, prob + h * SLOTS,
                    sum + h, index ? selected : NULL, count, begin);
            }
            dt = seconds() - start;
            if (rep >= 0 && dt < new_time) new_time = dt;
            bad |= memcmp(expected, actual, sizeof(actual)) != 0;
#pragma omp parallel for schedule(static)
            for (int task = 0; task < HEADS * 8; ++task) {
                int h = task / 8, begin = task % 8 * 64;
                glm53f_mla_value64_f16(actual + h * 512, half, prob + h * SLOTS,
                    sum + h, index ? selected : NULL, count, begin);
            }
            bad |= memcmp(expected, actual, sizeof(actual)) != 0;
        }
        printf("MLA_VALUE indexed=%d count=%d scratch_us=%.6f registers_us=%.6f speedup=%.3f %s\n",
            index, count, old_time * 1e6, new_time * 1e6, old_time / new_time,
            bad ? "FAIL" : "PASS");
    }
    free(prob); free(half); free(z);
    return bad;
}
