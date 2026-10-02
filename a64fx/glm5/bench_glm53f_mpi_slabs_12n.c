/* Private MPI message-size probe; compare against the existing512-token
 * MoE reduction before timing. Synthetic parity is not a full-model gate. */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { WIDTH = 4096, TOKENS = 4096, NS = 7, TRIALS = 7 };
static const int slabs[NS] = {64, 128, 256, 512, 1024, 2048, 4096};
static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    x *= 0x846ca68bu; return x ^ (x >> 16);
}
static void fill(float *p, int n, int rank, int pattern) {
    for (int i = 0; i < n; ++i) {
        uint32_t u = mix((uint32_t)i + (uint32_t)rank * 0x9e3779b9u), bits;
        if (pattern == 4) { p[i] = (float)(rank + 1) + (float)(i % 7) * .25f; continue; }
        if (pattern == 1) {
            u = mix((uint32_t)i + (uint32_t)(rank / 2) * 0x9e3779b9u);
            bits = ((uint32_t)(rank & 1) << 31) | (126u << 23) | (u & 0x7fffffu);
            if (rank & 1) bits += (uint32_t)(i & 3);
        } else if (pattern == 2) bits = (u & 0x807fffffu) | ((80u + (u >> 24) % 60) << 23);
        else if (pattern == 3) bits = (u & 0x807fffffu) | ((1u + (u >> 24) % 8) << 23);
        else if (pattern == 5) bits = (127u << 23) | (mix((uint32_t)i) & 0x7ffff0u) | (uint32_t)rank;
        else bits = (u & 0x807fffffu) | ((123u + (u >> 24) % 8) << 23);
        memcpy(p + i, &bits, sizeof(bits));
    }
}
static void reduce(const float *in, float *out, int tokens, int slab) {
    for (int base = 0; base < tokens; base += slab) {
        int n = tokens - base < slab ? tokens - base : slab;
        if (MPI_Allreduce(in + (size_t)base * WIDTH, out + (size_t)base * WIDTH,
                n * WIDTH, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS)
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (ranks != 12) MPI_Abort(MPI_COMM_WORLD, 2);
    const size_t n = (size_t)TOKENS * WIDTH;
    float *in = malloc(n * 4), *out = malloc(n * 4), *ref = malloc(n * 4);
    if (!in || !out || !ref) MPI_Abort(MPI_COMM_WORLD, 2);
    int exact[NS]; for (int s = 0; s < NS; ++s) exact[s] = 1;
    const int lengths[] = {128, 3953, 4096};
    for (int pattern = 0; pattern < 6; ++pattern)
        for (unsigned l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) {
            int tokens = lengths[l]; fill(in, tokens * WIDTH, rank, pattern);
            reduce(in, ref, tokens, 512);
            for (int s = 0; s < NS; ++s) {
                memset(out, 0, (size_t)tokens * WIDTH * 4); reduce(in, out, tokens, slabs[s]);
                long mismatches = 0;
                for (size_t i = 0; i < (size_t)tokens * WIDTH; ++i)
                    mismatches += memcmp(ref + i, out + i, 4) != 0;
                long maximum; MPI_Allreduce(&mismatches, &maximum, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
                if (maximum) exact[s] = 0;
                if (!rank) printf("MPI_SLAB_GATE pattern=%d tokens=%d slab=%d maximum_bit_mismatches=%ld %s\n",
                    pattern, tokens, slabs[s], maximum, maximum ? "FAIL" : "PASS");
            }
        }
    fill(in, TOKENS * WIDTH, rank, 0);
    for (int trial = -1; trial < TRIALS; ++trial)
        for (int order = 0; order < NS; ++order) {
            int s = (order + trial + 1) % NS;
            if (!exact[s]) continue;
            MPI_Barrier(MPI_COMM_WORLD); double begin = MPI_Wtime();
            reduce(in, out, TOKENS, slabs[s]); double elapsed = MPI_Wtime() - begin, maximum;
            MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (!rank && trial >= 0) printf("MPI_SLAB_TIMING trial=%d tokens=%d slab=%d maximum_seconds=%.9f\n",
                trial, TOKENS, slabs[s], maximum);
        }
    if (!rank) {
        for (int s = 0; s < NS; ++s) printf("MPI_SLAB_ELIGIBLE slab=%d exact=%d\n", slabs[s], exact[s]);
        puts("MPI_SLAB_PROBE_COMPLETE");
    }
    free(in); free(out); free(ref); MPI_Finalize(); return 0;
}
