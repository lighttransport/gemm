/* 12-rank allreduce microbenchmark for the prefill collectives (the MoE combine and the sparse o_proj partial).
 * Measures the pure communication cost with a barrier before every iteration, so rank skew is excluded.
 *
 * run:  mpiexec -np 12 ./bench_glm53f_allreduce_12n [tokens=512] [iters=10]   (TOFU_TOPO_PATH must point at a topology file)
 * algorithms: 0 utofu (decode collective), 1 MPI reduce-scatter+allgatherv, 2 ring, 3 tree-rsag, 4 tree-packed, 5 multi-TNI rs+ag */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "glm53f_collective_12n.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int tokens = argc > 1 ? atoi(argv[1]) : 512, iters = argc > 2 ? atoi(argv[2]) : 10, W = 4096;
    const int maxtok = getenv("BENCH_MAXTOK") ? atoi(getenv("BENCH_MAXTOK")) : 32;
    if (glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"), maxtok * W)) {
        if (!rank) fprintf(stderr, "collective init failed (TOFU_TOPO_PATH=%s)\n", getenv("TOFU_TOPO_PATH"));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    float *in = malloc((size_t)tokens * W * 4), *out = malloc((size_t)tokens * W * 4);
    for (size_t i = 0; i < (size_t)tokens * W; ++i) in[i] = (float)(rank + 1) + (float)(i % 7) * 0.25f;
    if (argc > 3 && !strcmp(argv[3], "lat")) { /* small-message latency: decode-style 1..8 token allreduces, barrier */
        const int counts[] = {1024, 4096, 16384, 32768};
        for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ++ci) {
            const int cnt = counts[ci], reps = 2000;
            float *a = malloc((size_t)cnt * 4), *b = malloc((size_t)cnt * 4);
            for (int i = 0; i < cnt; ++i) a[i] = (float)rank;
            for (int kind = 0; kind < 2; ++kind) {
                for (int w = 0; w < 200; ++w) { if (kind == 0) glm53f_sum_allreduce_12n(a, b, cnt); else MPI_Allreduce(a, b, cnt, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); }
                MPI_Barrier(MPI_COMM_WORLD);
                double t0 = MPI_Wtime();
                for (int r = 0; r < reps; ++r) { if (kind == 0) glm53f_sum_allreduce_12n(a, b, cnt); else MPI_Allreduce(a, b, cnt, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); }
                double dt = (MPI_Wtime() - t0) / reps, mx;
                MPI_Allreduce(&dt, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                if (!rank) printf("latency %-12s count=%6d (%5.1f KiB): %.2f us per allreduce\n", kind == 0 ? "utofu-decode" : "MPI_Allreduce", cnt, cnt * 4 / 1024.0, mx * 1e6);
            }
        }
        double t0 = MPI_Wtime();
        for (int r = 0; r < 2000; ++r) MPI_Barrier(MPI_COMM_WORLD);
        double db = (MPI_Wtime() - t0) / 2000;
        if (!rank) printf("latency MPI_Barrier: %.2f us\n", db * 1e6);
        MPI_Finalize();
        return 0;
    }
    if (!rank) printf("tokens=%d width=%d payload=%.1f MiB ranks=%d\n", tokens, W, (double)tokens * W * 4 / 1048576.0, ranks);
    const int slabs[] = {1, 4, 8, 16, 32, 64, 128, 256, 512};
    for (int algo = 0; algo <= 5; ++algo) {
        if (glm53f_collective_prefill_algorithm_12n(algo)) { if (!rank) printf("algo %d: unsupported\n", algo); continue; }
        for (unsigned si = 0; si < sizeof(slabs) / sizeof(*slabs); ++si) {
            const int slab = slabs[si];
            double best = 1e9, sum = 0; int ok = 1;
            for (int it = -2; it < iters; ++it) {
                memset(out, 0, (size_t)tokens * W * 4);
                MPI_Barrier(MPI_COMM_WORLD);
                double t0 = MPI_Wtime();
                if (glm53f_sum_allreduce_slabs_12n(in, out, tokens, W, slab)) { ok = 0; break; }
                double t1 = MPI_Wtime(), dt = t1 - t0, mx;
                MPI_Allreduce(&dt, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                if (it >= 0) { sum += mx; if (mx < best) best = mx; }
            }
            double err = 0;
            for (size_t i = 0; i < (size_t)tokens * W; i += 977) {
                double want = 0;
                for (int r = 0; r < ranks; ++r) want += (double)(r + 1) + (double)(i % 7) * 0.25;
                err = fmax(err, fabs(out[i] - want) / want);
            }
            double gerr; MPI_Allreduce(&err, &gerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (!rank)
                printf("algo=%d slab=%2d : best %.3f ms mean %.3f ms -> %.2f us/token, %.2f GB/s payload %s\n", algo, slab,
                       best * 1e3, sum / iters * 1e3, best * 1e6 / tokens, (double)tokens * W * 4 / best / 1e9,
                       ok && gerr < 1e-5 ? "OK" : "FAIL");
        }
    }
    /* raw MPI collectives on large messages (independent of the uTofu wrapper) */
    const int big[] = {32, 64, 128, 256, 512};
    for (unsigned bi = 0; bi < sizeof(big) / sizeof(*big); ++bi) {
        const int tk = big[bi] < tokens ? big[bi] : tokens, cnt = tk * W;
        float *rs = malloc((size_t)cnt / ranks * 4 + 64), *tmp = malloc((size_t)cnt * 4);
        for (int kind = 0; kind < 3; ++kind) {
            double best = 1e9;
            for (int it = -2; it < iters; ++it) {
                MPI_Barrier(MPI_COMM_WORLD);
                double t0 = MPI_Wtime();
                for (int t = 0; t < tokens; t += tk) {
                    if (kind == 0) MPI_Allreduce(in + (size_t)t * W, out + (size_t)t * W, cnt, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                    else if (kind == 1) { /* reduce_scatter_block + allgather */
                        MPI_Reduce_scatter_block(in + (size_t)t * W, rs, cnt / ranks, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                        MPI_Allgather(rs, cnt / ranks, MPI_FLOAT, out + (size_t)t * W, cnt / ranks, MPI_FLOAT, MPI_COMM_WORLD);
                    } else { /* in-place allreduce */
                        memcpy(tmp, in + (size_t)t * W, (size_t)cnt * 4);
                        MPI_Allreduce(MPI_IN_PLACE, tmp, cnt, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                    }
                }
                double dt = MPI_Wtime() - t0, mx;
                MPI_Allreduce(&dt, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                if (it >= 0 && mx < best) best = mx;
            }
            if (!rank)
                printf("raw MPI %-22s call=%3d tokens: best %.3f ms per %d tokens -> %.2f us/token, %.2f GB/s\n",
                       kind == 0 ? "Allreduce" : kind == 1 ? "RSblock+Allgather" : "Allreduce(in-place)", tk, best * 1e3, tokens,
                       best * 1e6 / tokens, (double)tokens * W * 4 / best / 1e9);
        }
        free(rs); free(tmp);
    }
    MPI_Finalize();
    return 0;
}
