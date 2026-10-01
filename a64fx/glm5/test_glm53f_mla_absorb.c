/* Native absorbed-query arithmetic and warm timing, no model/MPI required. */
#include "glm53f_mla_absorb.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HEADS = 6, D = 512, Q = 256, REPEATS = 100 };
static uint16_t weight[HEADS * Q * D] __attribute__((aligned(256)));
static float query[HEADS * Q], ref[HEADS * D], got[HEADS * D];
static void legacy(float *out, const uint16_t *w, const float *q, int begin) {
    svbool_t p = svptrue_b32();
    for (int d = begin; d < begin + 64; ++d) out[d] = 0;
    for (int j = 0; j < Q; ++j) {
        float x = q[j] / sqrtf((float)Q);
        for (int d = begin; d < begin + 64; d += 16) {
            svuint32_t b = svlsl_n_u32_x(p, svld1uh_u32(p, w + (size_t)j * D + d), 16);
            svst1_f32(p, out + d, svmla_n_f32_x(p, svld1_f32(p, out + d), svreinterpret_f32_u32(b), x));
        }
    }
}
static void run(int candidate) {
#pragma omp parallel for schedule(static)
    for (int b = 0; b < HEADS * D / 64; ++b) {
        int h = b / 8, begin = b % 8 * 64;
        if (candidate) glm53f_mla_absorb64(got + h * D, weight + (size_t)h * Q * D, query + h * Q, begin);
        else legacy(ref + h * D, weight + (size_t)h * Q * D, query + h * Q, begin);
    }
}
int main(void) {
    if (svcntw() != 16) return 2;
    unsigned state = 12345;
    for (int i = 0; i < HEADS * Q * D; ++i) {
        state = state * 1664525u + 1013904223u;
        /* Bounded signed BF16, including cancellation and tiny values. */
        weight[i] = (uint16_t)((state >> 16 & 0x807f) | ((118 + (state % 16)) << 7));
    }
    for (int i = 0; i < HEADS * Q; ++i) query[i] = (float)((i * 73) % 257 - 128) / 127;
    run(0); run(1);
    if (memcmp(ref, got, sizeof(ref))) { puts("GLM53F_MLA_ABSORB FAIL"); return 1; }
    double old[5], next[5];
    for (int trial = 0; trial < 5; ++trial) {
        /* Matched warm trials with alternating order. */
        for (int order = 0; order < 2; ++order) {
            int candidate = (order + trial) % 2;
            run(candidate);
            double begin = omp_get_wtime();
            for (int r = 0; r < REPEATS; ++r) run(candidate);
            (candidate ? next : old)[trial] = (omp_get_wtime() - begin) / REPEATS;
        }
        if (memcmp(ref, got, sizeof(ref))) return 1;
    }
    for (int i = 1; i < 5; ++i) {
        for (int j = i; j > 0 && old[j] < old[j-1]; --j) { double v = old[j]; old[j] = old[j-1]; old[j-1] = v; }
        for (int j = i; j > 0 && next[j] < next[j-1]; --j) { double v = next[j]; next[j] = next[j-1]; next[j-1] = v; }
    }
    printf("GLM53F_MLA_ABSORB threads=%d legacy_us=%.6f candidate_us=%.6f speedup=%.6f BIT_EXACT PASS\n",
        omp_get_max_threads(), old[2] * 1e6, next[2] * 1e6, old[2] / next[2]);
    return 0;
}
