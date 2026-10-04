#define _GNU_SOURCE
#include <arm_sve.h>
#include <omp.h>
#include "glm53f_moe_grouped_native.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

static uint32_t seed = 1;
static float input(void) {
    seed = seed * 1664525u + 1013904223u;
    uint32_t bits = (seed & 0x807fffffu) | ((112u + (seed >> 27)) << 23);
    float x; memcpy(&x, &bits, sizeof(x)); return x;
}
static void run(float *out, const uint16_t *w, const float *x,
        int tokens, int rows, int cols, int mode) {
    if (mode) gmn_router_prefill(out, w, x, tokens, cols, 1);
    else {
        const int ngroups = (tokens + 5) / 6, nch = (ngroups + 3) / 4;
#pragma omp parallel for schedule(dynamic, 1)
        for (int task = 0; task < GMN_RTILES * nch; ++task) {
            const int tile = task / nch, ch = task % nch;
            for (int g = ch * 4; g < ngroups && g < (ch + 1) * 4; ++g) {
                const int t = g * 6, n = tokens - t < 6 ? tokens - t : 6;
                gmn_router_run(out + (size_t)t * rows + tile * GMN_RTILE,
                    rows, w, tile, x + (size_t)t * cols, cols, cols, n);
            }
        }
    }
}
static int check(int tokens, int rows, int cols, int rank) {
    size_t count = (size_t)tokens * rows;
    uint16_t *w = malloc((size_t)rows * cols * 2);
    uint16_t *packed = malloc((size_t)rows * cols * 2);
    float *x = malloc((size_t)tokens * cols * 4);
    float *a = malloc((count + 32) * 4), *b = malloc((count + 32) * 4);
    if (!w || !packed || !x || !a || !b) MPI_Abort(MPI_COMM_WORLD, 2);
    seed = 17u + (unsigned)rank;
    for (int i = 0; i < rows * cols; ++i) { float v = input(); uint32_t u; memcpy(&u, &v, 4); w[i] = (uint16_t)(u >> 16); }
    for (size_t i = 0; i < (size_t)tokens * cols; ++i) x[i] = i % 17 ? input() : 0;
    for (size_t i = 0; i < count + 32; ++i) a[i] = b[i] = -12345;
    gmn_router_pack(packed, w, cols);
    run(a + 16, packed, x, tokens, rows, cols, 0);
    run(b + 16, packed, x, tokens, rows, cols, 1);
    int bad = memcmp(a, b, (count + 32) * 4) != 0;
    for (size_t i = 16; i < count + 16; ++i) { uint32_t u; memcpy(&u, b + i, 4); bad |= (u & 0x7f800000u) == 0x7f800000u; }
    for (int i = 0; i < 16; ++i) bad |= a[i] != -12345 || a[count + 16 + i] != -12345;
    free(w); free(packed); free(x); free(a); free(b); return bad;
}
static void bench(int tokens, int rank) {
    enum { R = 288, C = 4096 };
    uint16_t *w = malloc((size_t)R * C * 2);
    float *x = malloc((size_t)tokens * C * 4), *y = malloc((size_t)tokens * R * 4);
    if (!w || !x || !y) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int i = 0; i < R * C; ++i) w[i] = 0x3f00 + (i % 127);
    for (size_t i = 0; i < (size_t)tokens * C; ++i) x[i] = (float)(i % 31) * .01f;
    for (int mode = 0; mode < 2; ++mode) run(y, w, x, tokens, R, C, mode);
    for (int pair = 0; pair < 7; ++pair)
        for (int turn = 0; turn < 2; ++turn) {
            int mode = turn ^ (pair & 1);
            MPI_Barrier(MPI_COMM_WORLD);
            double begin = MPI_Wtime();
            for (int rep = 0; rep < 2; ++rep) run(y, w, x, tokens, R, C, mode);
            double elapsed = (MPI_Wtime() - begin) / 2, maximum;
            MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (!rank) printf("ROUTER_TILES_TIME tokens=%d pair=%d mode=%d seconds=%.9f\n", tokens, pair, mode, maximum);
        }
    free(w); free(x); free(y);
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int tokens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24, 25, 47, 48, 49, 127, 128, 129};
    const int rows[] = {288}, cols[] = {16, 31, 64};
    int failed = 0, cases = 0;
    for (size_t t = 0; t < sizeof(tokens) / sizeof(tokens[0]); ++t)
        for (size_t r = 0; r < sizeof(rows) / sizeof(rows[0]); ++r)
            for (size_t c = 0; c < sizeof(cols) / sizeof(cols[0]); ++c) { failed |= check(tokens[t], rows[r], cols[c], rank); ++cases; }
    const int wide[] = {5, 47, 511, 512, 513, 4096};
    for (size_t t = 0; t < sizeof(wide) / sizeof(wide[0]); ++t) { failed |= check(wide[t], 288, 4096, rank); ++cases; }
    int any; MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) printf("ROUTER_TILES cases=%d bits_and_guards=%s %s\n", cases, any ? "FAIL" : "BIT_EXACT", any ? "FAIL" : "PASS");
    if (!any && omp_get_max_threads() > 1) { bench(512, rank); bench(4096, rank); }
    MPI_Finalize(); return any ? 1 : 0;
}
