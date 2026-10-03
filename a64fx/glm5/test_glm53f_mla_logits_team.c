#include "glm53f_mla_attention.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void reference_team(float *lg, const float *q, const float *cache,
        const uint16_t *half, const int *selected, int nt, int heads) {
    const svbool_t pg = svptrue_b32();
#pragma omp for schedule(static)
    for (int w = 0; w < heads * nt; ++w) {
        const int h = w / nt, t = w % nt;
        svfloat32_t dot = svdup_f32(0);
        for (int d = 0; d < 512; d += 16)
            dot = svmla_f32_x(pg, dot, svld1_f32(pg, q + (size_t)h * 512 + d),
                mlb_cache_load(pg, cache, half, (size_t)selected[t] * 512 + d));
        lg[(size_t)h * 2052 + t] = svaddv_f32(pg, dot);
    }
}
static float input(uint32_t *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    uint32_t bits = (*seed & UINT32_C(0x807fffff)) |
        ((106u + ((*seed >> 24) & 31u)) << 23);
    float x; memcpy(&x, &bits, 4); return x;
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    enum { S = 2052, D = 512, H = 6, GUARD = 16 };
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    float *cache = malloc((size_t)S * D * 4);
    uint16_t *half = malloc((size_t)S * D * 2);
    float q[H * D], expected[H * S + GUARD], actual[H * S + GUARD];
    int selected[S], bad = 0, cases = 0;
    if (!cache || !half || svcntw() != 16) MPI_Abort(MPI_COMM_WORLD, 2);
    uint32_t seed = 173u + (uint32_t)rank * 71u;
    for (int i = 0; i < S * D; ++i) cache[i] = input(&seed);
    for (int i = 0; i < H * D; ++i) q[i] = input(&seed);
    glm53f_mla_cache_f16_store(half, cache, S * D);
    const int counts[] = {1, 3, 4, 7, 15, 16, 17, 127, 128, 2048, 2051, 2052};
    for (int view = 0; view < 2; ++view) for (int order = 0; order < 3; ++order) {
        for (int i = 0; i < S; ++i)
            selected[i] = order == 0 ? i : order == 1 ? S - 1 - i : (i * 37) % S;
        const uint16_t *hc = view ? half : NULL;
        for (int heads = 1; heads <= H; ++heads)
            for (size_t ci = 0; ci < sizeof(counts)/sizeof(counts[0]); ++ci) {
                const int nt = counts[ci];
                memset(expected, 0xa5, sizeof(expected));
                memset(actual, 0xa5, sizeof(actual));
#pragma omp parallel
                { reference_team(expected, q, cache, hc, selected, nt, heads); }
#pragma omp parallel
                { mlb_logits_heads3_team(actual, q, cache, hc, selected, nt, heads); }
                bad |= memcmp(expected, actual, sizeof(actual)) != 0;
                for (int h = 0; h < heads; ++h) for (int t = 0; t < nt; ++t) {
                    uint32_t bits; memcpy(&bits, actual + (size_t)h * S + t, 4);
                    bad |= (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000);
                }
                ++cases;
            }
    }
    int global; MPI_Allreduce(&bad, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) printf("MLA_LOGITS_TEAM cases=%d finite_logits_guards=%s %s\n", cases,
        global ? "FAIL" : "BIT_EXACT", global ? "FAIL" : "PASS");
    if (global) MPI_Abort(MPI_COMM_WORLD, 3);
    for (int i = 0; i < S; ++i) selected[i] = (i * 37) % S;
    for (int heads = 5; heads <= H; ++heads)
        for (int trial = 0; trial < 5; ++trial)
            for (int step = 0; step < 2; ++step) {
                int candidate = trial % 2 ? 1 - step : step;
                MPI_Barrier(MPI_COMM_WORLD);
                double start = MPI_Wtime();
#pragma omp parallel
                {
                    for (int repeat = 0; repeat < 100; ++repeat) {
                        if (candidate) mlb_logits_heads3_team(actual, q, cache, half, selected, S - repeat % 47, heads);
                        else reference_team(actual, q, cache, half, selected, S - repeat % 47, heads);
                    }
                }
                double dt = MPI_Wtime() - start, maximum;
                MPI_Reduce(&dt, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (!rank) printf("MLA_LOGITS_TEAM_TIME heads=%d trial=%d candidate=%d seconds=%.9f\n",
                    heads, trial, candidate, maximum);
            }
    free(cache); free(half); MPI_Finalize(); return 0;
}
