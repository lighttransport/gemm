#include "glm53f_mla_attention.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Diagnostic: split only independent heads; keep each head's lane/key order. */
static void tiled(float *va, int stride, float *lg, const float *q, const float *cache,
        const uint16_t *half, const int *selected, int nt, int heads, int tile) {
    if (heads == 6 && (tile == 3 || tile == 7)) {
        if (mlb_token_prefill(va, stride, lg, q, cache, half, selected, nt, heads, tile == 7 ? 2 : 1)) abort();
        return;
    }
    for (int h = 0; h < heads; h += tile) {
        int n = heads - h < tile ? heads - h : tile;
        mlb_token(va + (size_t)h * stride, stride, lg + (size_t)h * 2052,
            q + (size_t)h * 512, cache, half, selected, nt, n);
    }
}
static float input(uint32_t *seed, int query) {
    *seed = *seed * 1664525u + 1013904223u;
    uint32_t bits = (*seed & UINT32_C(0x807fffff)) |
        ((uint32_t)(query ? 119 : 124) << 23);
    float x; memcpy(&x, &bits, 4); return x;
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    enum { S = 2052, D = 512, H = 6, T = 47, VS = 768 };
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    float *cache = malloc((size_t)S * D * 4);
    uint16_t *half = malloc((size_t)S * D * 2);
    float *q = malloc((size_t)T * H * D * 4);
    float *va = malloc((size_t)T * H * VS * 4);
    float *lg = malloc((size_t)T * H * S * 4);
    float *refva = malloc((size_t)H * VS * 4), *reflg = malloc((size_t)H * S * 4);
    int selected[S], bad = 0, cases = 0;
    if (!cache || !half || !q || !va || !lg || !refva || !reflg || svcntw() != 16)
        MPI_Abort(MPI_COMM_WORLD, 2);
    uint32_t seed = 173 + (uint32_t)rank * 71;
    for (int i = 0; i < S * D; ++i) cache[i] = input(&seed, 0);
    for (int i = 0; i < T * H * D; ++i) q[i] = input(&seed, 1);
    glm53f_mla_cache_f16_store(half, cache, S * D);
    const int counts[] = {1, 3, 7, 15, 16, 17, 127, 128, 2048, 2051, 2052};
    for (int view = 0; view < 2; ++view) for (int order = 0; order < 3; ++order) {
        for (int i = 0; i < S; ++i)
            selected[i] = order == 0 ? i : order == 1 ? S - 1 - i : (i * 37) % S;
        for (int heads = 5; heads <= 6; ++heads)
            for (int stride = D; stride <= VS; stride += VS - D)
                for (size_t ci = 0; ci < sizeof(counts)/sizeof(counts[0]); ++ci) {
                    const uint16_t *hc = view ? half : NULL;
                    memset(refva, 0xa5, (size_t)H * VS * 4);
                    memset(reflg, 0xa5, (size_t)H * S * 4);
                    mlb_token(refva, stride, reflg, q, cache, hc, selected, counts[ci], heads);
                    for (int tile = 1; tile <= 7; ++tile) {
                        memset(va, 0xa5, (size_t)H * VS * 4);
                        memset(lg, 0xa5, (size_t)H * S * 4);
                        tiled(va, stride, lg, q, cache, hc, selected, counts[ci], heads, tile);
                        bad |= memcmp(va, refva, (size_t)H * VS * 4) != 0;
                        bad |= memcmp(lg, reflg, (size_t)H * S * 4) != 0;
                        for (int h = 0; h < heads; ++h) for (int d = 0; d < D; ++d) {
                            uint32_t bits; memcpy(&bits, va + (size_t)h * stride + d, 4);
                            bad |= (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000);
                        }
                        ++cases;
                    }
                }
    }
    int global; MPI_Allreduce(&bad, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) printf("MLA_HEAD_TILES cases=%d logits_values_guards=%s %s\n", cases,
        global ? "FAIL" : "BIT_EXACT", global ? "FAIL" : "PASS");
    if (global) MPI_Abort(MPI_COMM_WORLD, 3);
    for (int i = 0; i < S; ++i) selected[i] = (i * 37) % S;
    for (int heads = 5; heads <= 6; ++heads)
        for (int trial = 0; trial < 5; ++trial)
            for (int step = 0; step < 7; ++step) {
                int tile = trial % 2 ? 7 - step : step + 1;
                MPI_Barrier(MPI_COMM_WORLD);
                double start = MPI_Wtime();
#pragma omp parallel for schedule(dynamic, 1)
                for (int t = 0; t < T; ++t)
                    tiled(va + (size_t)t * H * D, D, lg + (size_t)t * H * S,
                        q + (size_t)t * H * D, cache, half, selected, S - t,
                        heads, tile);
                double dt = MPI_Wtime() - start, maximum;
                MPI_Reduce(&dt, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (!rank) printf("MLA_HEAD_TILE_TIME heads=%d trial=%d tile=%d seconds=%.9f\n",
                    heads, trial, tile, maximum);
            }
    free(cache); free(half); free(q); free(va); free(lg); free(refva); free(reflg);
    MPI_Finalize(); return 0;
}
