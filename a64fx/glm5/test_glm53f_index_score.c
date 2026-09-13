#define _POSIX_C_SOURCE 200809L
#include "glm53f_index_score.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static float reference(const float *q, const float *hw, const float *pk) {
    float score = 0;
    for (int h = 0; h < 32; ++h) {
        double dot = 0;
        for (int d = 0; d < 128; ++d) dot += (double)q[h * 128 + d] * pk[d];
        if (dot > 0) score += hw[h] * (float)(dot / sqrt(128.0)) / sqrtf(32.0f);
    }
    return score;
}
static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
int main(void) {
    enum { POOLS = 10923 };
    float q[4096], hw[32], *keys = malloc((size_t)POOLS * 128 * 4);
    float *expected = malloc(POOLS * 4), *actual = malloc(POOLS * 4);
    double qt[4096]; uint32_t seed = 73;
    if (!keys || !expected || !actual) return 1;
    for (int i = 0; i < 4096; ++i) { seed = seed * 1664525u + 1013904223u; q[i] = ((int)(seed >> 8) % 20001 - 10000) * 0.0001f; }
    for (int h = 0; h < 32; ++h) hw[h] = (h - 11) * 0.01f;
    for (int i = 0; i < POOLS * 128; ++i) { seed = seed * 1664525u + 1013904223u; keys[i] = ((int)(seed >> 8) % 20001 - 10000) * 0.0001f; }
    memset(keys, 0, 128 * 4);
    glm53f_index_query_transpose(qt, q);
    double ref_time = 1e30, candidate_time = 1e30;
    /* Warm both paths and the OpenMP team; report matched steady-state minima. */
    for (int rep = -2; rep < 8; ++rep) {
    double begin = seconds();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < POOLS; ++i) expected[i] = reference(q, hw, keys + (size_t)i * 128);
    double elapsed = seconds() - begin;
    if (rep >= 0 && elapsed < ref_time) ref_time = elapsed;
    begin = seconds();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < POOLS; ++i) actual[i] = glm53f_index_score_transposed(qt, hw, keys + (size_t)i * 128);
    elapsed = seconds() - begin;
    if (rep >= 0 && elapsed < candidate_time) candidate_time = elapsed;
    }
    int mismatches = 0; double checksum = 0;
    for (int i = 0; i < POOLS; ++i) {
        mismatches += memcmp(expected + i, actual + i, sizeof(float)) != 0;
        checksum += actual[i];
    }
    printf("INDEX_SCORE pools=%d bit_mismatches=%d reference_ms=%.6f candidate_ms=%.6f checksum=%.12g %s\n",
        POOLS, mismatches, ref_time * 1e3, candidate_time * 1e3, checksum, mismatches ? "FAIL" : "PASS");
    free(actual); free(expected); free(keys);
    return mismatches != 0;
}
