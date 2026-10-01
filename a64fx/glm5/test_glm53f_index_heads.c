#define _POSIX_C_SOURCE 200809L
#include "glm53f_index_score.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
static float reference(const float *q, const float *weight, const float *key) {
    float score = 0;
    for (int h = 0; h < 32; ++h) {
        svfloat32_t a = svdup_f32(0);
        for (int i = 0; i < 128; i += (int)svcntw()) {
            svbool_t p = svwhilelt_b32(i, 128);
            a = svmla_f32_x(p, a, svld1_f32(p, q + h * 128 + i), svld1_f32(p, key + i));
        }
        float dot = svaddv_f32(svptrue_b32(), a);
        if (dot > 0) score += weight[h] * dot / sqrtf(128.f * 32.f);
    }
    return score;
}
static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
int main(void) {
    enum { POOLS = 8193 };
    float q[4096], weight[32], *keys = malloc((size_t)POOLS * 128 * sizeof(float));
    float *expected = malloc(POOLS * sizeof(float)), *actual = malloc(POOLS * sizeof(float));
    uint32_t seed = 73;
    if (!keys || !expected || !actual) return 1;
    for (int i = 0; i < 4096; ++i) { seed = seed * 1664525u + 1013904223u; q[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f; }
    for (int h = 0; h < 32; ++h) weight[h] = (h - 11) * .01f;
    for (int i = 0; i < POOLS * 128; ++i) { seed = seed * 1664525u + 1013904223u; keys[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f; }
    memset(keys, 0, 128 * sizeof(float));
    double old_time = 1e30, new_time = 1e30;
    int bad = 0;
    for (int rep = -2; rep < 5; ++rep) {
        double begin = seconds();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < POOLS; ++i) expected[i] = reference(q, weight, keys + (size_t)i * 128);
        double dt = seconds() - begin;
        if (rep >= 0 && dt < old_time) old_time = dt;
        begin = seconds();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < POOLS; ++i) actual[i] = glm53f_index_score_f32_heads(q, weight, keys + (size_t)i * 128);
        dt = seconds() - begin;
        if (rep >= 0 && dt < new_time) new_time = dt;
        bad |= memcmp(expected, actual, POOLS * sizeof(float)) != 0;
    }
    printf("INDEX_HEADS %s pools=%d reference_ms=%.6f candidate_ms=%.6f speedup=%.3f\n", bad ? "FAIL" : "PASS", POOLS, old_time * 1e3, new_time * 1e3, old_time / new_time);
    free(actual); free(expected); free(keys);
    return bad;
}
