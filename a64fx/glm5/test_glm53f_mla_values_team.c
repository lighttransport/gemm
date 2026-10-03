#include "glm53f_mla_absorb.h"
#include "glm53f_mla_attention.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

enum { S = 2052, D = 512, H = 6, GUARD = 16 };
/* Diagnostic only: modes3/4 divide each probability once into bounded scratch
 * before2×32 or3×64 shared value-cache tiles. Each division result and key FMA retains
 * legacy rounding. Other modes retain the volatile division and ascending selected-key
 * FMAs of value64_f16; share the half-cache loads across independent heads. */
static inline __attribute__((always_inline)) void grouped_tile(float *out,
        const uint16_t *cache, const float *prob, const float *sum,
        const int *selected, int nt, int begin, const int nh, const int nc, const int normalized) {
    const svbool_t pg = svptrue_b32();
    svfloat32_t a0 = svdup_f32(0), b0 = a0;
    svfloat32_t a1 = a0, b1 = a0, a2 = a0, b2 = a0;
    svfloat32_t c0 = a0, d0 = a0, c1 = a0, d1 = a0, c2 = a0, d2 = a0;
    for (int t = 0; t < nt; ++t) {
        const uint16_t *z = cache + (size_t)selected[t] * D + begin;
        svfloat32_t z0 = glm53f_mla_cache_f16_load(pg, z);
        svfloat32_t z1 = z0;
        if (nc >= 2) z1 = glm53f_mla_cache_f16_load(pg, z + 16);
        svfloat32_t z2 = z0, z3 = z0;
        if (nc == 4) { z2 = glm53f_mla_cache_f16_load(pg, z + 32); z3 = glm53f_mla_cache_f16_load(pg, z + 48); }
#define STEP(N) if ((N) < nh) { \
        float x = prob[(size_t)(N) * S + t]; \
        if (!normalized) x /= *((const volatile float *)sum + (N)); \
        a##N = svmla_n_f32_x(pg, a##N, z0, x); \
        if (nc >= 2) b##N = svmla_n_f32_x(pg, b##N, z1, x); \
        if (nc == 4) { c##N = svmla_n_f32_x(pg, c##N, z2, x); d##N = svmla_n_f32_x(pg, d##N, z3, x); } }
        STEP(0); STEP(1); STEP(2);
#undef STEP
    }
#define STORE(N) if ((N) < nh) { \
    svst1_f32(pg, out + (size_t)(N) * D + begin, a##N); \
    if (nc >= 2) svst1_f32(pg, out + (size_t)(N) * D + begin + 16, b##N); \
    if (nc == 4) { svst1_f32(pg, out + (size_t)(N) * D + begin + 32, c##N); svst1_f32(pg, out + (size_t)(N) * D + begin + 48, d##N); } }
    STORE(0); STORE(1); STORE(2);
#undef STORE
}
static void values_team(float *out, const uint16_t *cache, const float *prob,
        const float *sum, const int *selected, int nt, int heads, int mode, float *normalized) {
    if (mode == 3 && heads == 6) {
        mlb_values_normalized2_team(out, prob, normalized, cache, sum, selected, nt);
        return;
    }
    if (mode >= 3) {
#pragma omp for schedule(static)
        for (int w = 0; w < heads * nt; ++w) {
            const int h = w / nt, t = w % nt;
            normalized[(size_t)h * S + t] = prob[(size_t)h * S + t] /
                *((const volatile float *)sum + h);
        }
    }
    if (!mode) {
#pragma omp for schedule(static)
        for (int w = 0; w < heads * 8; ++w)
            glm53f_mla_value64_f16(out + (size_t)(w / 8) * D, cache,
                prob + (size_t)(w / 8) * S, sum + w / 8, selected, nt, (w % 8) * 64);
    } else {
        const int group = mode == 1 || mode == 3 ? 2 : 3;
        const int columns = mode == 2 ? 16 : mode == 4 ? 64 : 32;
        const int blocks = D / columns;
#pragma omp for schedule(static)
        for (int w = 0; w < ((heads + group - 1) / group) * blocks; ++w) {
            int h = (w / blocks) * group, begin = (w % blocks) * columns;
            int n = heads - h < group ? heads - h : group;
            float *o = out + (size_t)h * D;
            const float *p = (mode >= 3 ? normalized : prob) + (size_t)h * S, *s = sum + h;
            if (mode == 4) {
                switch (n) {
                case 1: grouped_tile(o, cache, p, s, selected, nt, begin, 1, 4, 1); break;
                case 2: grouped_tile(o, cache, p, s, selected, nt, begin, 2, 4, 1); break;
                case 3: grouped_tile(o, cache, p, s, selected, nt, begin, 3, 4, 1); break;
                }
            } else if (mode != 2) {
                if (n == 1) grouped_tile(o, cache, p, s, selected, nt, begin, 1, 2, mode == 3);
                else grouped_tile(o, cache, p, s, selected, nt, begin, 2, 2, mode == 3);
            } else {
                switch (n) {
                case 1: grouped_tile(o, cache, p, s, selected, nt, begin, 1, 1, 0); break;
                case 2: grouped_tile(o, cache, p, s, selected, nt, begin, 2, 1, 0); break;
                case 3: grouped_tile(o, cache, p, s, selected, nt, begin, 3, 1, 0); break;
                }
            }
        }
    }
}
static float input(uint32_t *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    uint32_t bits = (*seed & UINT32_C(0x807fffff)) |
        ((106u + ((*seed >> 24) & 31u)) << 23);
    float x; memcpy(&x, &bits, sizeof(x)); return x;
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    float *cache = malloc((size_t)S * D * sizeof(float));
    uint16_t *half = malloc((size_t)S * D * sizeof(uint16_t));
    float prob[H * S], normalized[H * S + GUARD], sums[H], expected[H * D + GUARD], actual[H * D + GUARD];
    int selected[S], bad = 0, cases = 0;
    if (!cache || !half || svcntw() != 16) MPI_Abort(MPI_COMM_WORLD, 2);
    uint32_t seed = 173u + (uint32_t)rank * 71u;
    for (int i = 0; i < S * D; ++i) cache[i] = input(&seed);
    glm53f_mla_cache_f16_store(half, cache, S * D);
    for (int h = 0; h < H; ++h) {
        sums[h] = 1.3f + 0.71f * h;
        for (int t = 0; t < S; ++t) {
            prob[h * S + t] = fabsf(input(&seed)) * 0.0031f;
            if (t % 17 == 0) prob[h * S + t] = 0.0f;
            else if (t % 19 == 0) {
                uint32_t tiny = (seed & UINT32_C(0x007fffff)) | 1u;
                memcpy(prob + h * S + t, &tiny, sizeof(tiny));
            }
        }
    }
    const int counts[] = {1, 3, 4, 7, 15, 16, 17, 127, 128, 511, 512, 513, 2048, 2051, 2052};
    for (int order = 0; order < 3; ++order) {
        for (int i = 0; i < S; ++i)
            selected[i] = order == 0 ? i : order == 1 ? S - 1 - i : (i * 37) % S;
        for (int heads = 1; heads <= H; ++heads)
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ++ci) {
                int nt = counts[ci];
                memset(expected, 0xa5, sizeof(expected));
#pragma omp parallel
                { values_team(expected, half, prob, sums, selected, nt, heads, 0, normalized); }
                for (int mode = 1; mode <= 4; ++mode) {
                    memset(actual, 0xa5, sizeof(actual));
                    memset(normalized, 0xa5, sizeof(normalized));
#pragma omp parallel
                    { values_team(actual, half, prob, sums, selected, nt, heads, mode, normalized); }
                    bad |= memcmp(expected, actual, sizeof(actual)) != 0;
                    for (int i = 0; i < heads * D; ++i) {
                        uint32_t bits; memcpy(&bits, actual + i, 4);
                        bad |= (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000);
                    }
                    if (mode >= 3) {
                        for (int i = 0; i < H * S + GUARD; ++i)
                            if (i >= heads * S || i % S >= nt) {
                                uint32_t bits; memcpy(&bits, normalized + i, 4);
                                bad |= bits != UINT32_C(0xa5a5a5a5);
                            }
                        memcpy(normalized, prob, sizeof(prob));
                        memset(normalized + H * S, 0xa5, GUARD * sizeof(float));
                        memset(actual, 0xa5, sizeof(actual));
#pragma omp parallel
                        { values_team(actual, half, normalized, sums, selected, nt, heads, mode, normalized); }
                        bad |= memcmp(expected, actual, sizeof(actual)) != 0;
                        for (int i = 0; i < H * S + GUARD; ++i)
                            if (i >= heads * S || i % S >= nt) {
                                uint32_t bits, original = UINT32_C(0xa5a5a5a5);
                                memcpy(&bits, normalized + i, 4);
                                if (i < H * S) memcpy(&original, prob + i, 4);
                                bad |= bits != original;
                            }
                        ++cases;
                    }
                    ++cases;
                }
            }
    }
    int global; MPI_Allreduce(&bad, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) printf("MLA_VALUES_TEAM cases=%d finite_values_guards=%s %s\n", cases,
        global ? "FAIL" : "BIT_EXACT", global ? "FAIL" : "PASS");
    if (global) MPI_Abort(MPI_COMM_WORLD, 3);
    for (int i = 0; i < S; ++i) selected[i] = (i * 37) % S;
    for (int heads = 5; heads <= H; ++heads)
        for (int trial = 0; trial < 7; ++trial)
            for (int step = 0; step < 5; ++step) {
                int mode = (trial + step) % 5;
                MPI_Barrier(MPI_COMM_WORLD);
                double start = MPI_Wtime();
#pragma omp parallel
                {
                    for (int repeat = 0; repeat < 100; ++repeat)
                        values_team(actual, half, prob, sums, selected, S - repeat % 47, heads, mode, normalized);
                }
                double dt = MPI_Wtime() - start, maximum;
                MPI_Reduce(&dt, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (!rank) printf("MLA_VALUES_TEAM_TIME heads=%d trial=%d mode=%d seconds=%.9f\n",
                    heads, trial, mode, maximum);
            }
    free(cache); free(half); MPI_Finalize(); return 0;
}
