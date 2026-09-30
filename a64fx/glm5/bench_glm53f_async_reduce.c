/* Stress test of the async slab-reduction helper (glm53f_async_*): a producer publishes 32-token tiles while an OpenMP
 * team "computes"; the helper thread reduces them with the multi-TNI allreduce.  Reproduces/diagnoses the long-prompt hang.
 * run: mpiexec -np 12 ./a.out [iters=300] [mpi_every=17]   (TOFU_TOPO_PATH must be set; algorithm 5 = mtni) */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "glm53f_collective_12n.h"
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int iters = argc > 1 ? atoi(argv[1]) : 300, mpi_every = argc > 2 ? atoi(argv[2]) : 17, W = 4096, T = 512, TILE = 32;
    if (glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"), TILE * W)) { if (!rank) fprintf(stderr, "init failed\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    if (glm53f_collective_prefill_algorithm_12n(5) || !glm53f_async_available_12n()) { if (!rank) fprintf(stderr, "mtni/async unavailable\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
    const int jt = argc > 3 ? atoi(argv[3]) : 1; /* MPI_Allreduce tokens after each finish (512 = MoE combine) */
    float *in = aligned_alloc(256, (size_t)T * W * 4), *out = aligned_alloc(256, (size_t)T * W * 4), *junk = aligned_alloc(256, (size_t)jt * 4096 * 4), *jout = aligned_alloc(256, (size_t)jt * 4096 * 4);
    memset(junk, 0, (size_t)jt * 4096 * 4);
    volatile double sink = 0;
    for (int it = 0; it < iters; ++it) {
        const float base = (float)(rank + 1) + (it % 5);
        if (glm53f_async_begin_12n(in, out, T, W, TILE)) { fprintf(stderr, "rank %d begin failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 4); }
        for (int tile = 0; tile < T / TILE; ++tile) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < TILE * W; ++i) in[(size_t)tile * TILE * W + i] = base + (float)((tile + i) % 7) * 0.25f;
            double s = 0; for (int k = 0; k < 4000; ++k) s += sqrt((double)k + rank); sink += s; /* skewed "compute" */
            glm53f_async_ready_12n((tile + 1) * TILE);
        }
        if (glm53f_async_finish_12n()) { fprintf(stderr, "rank %d finish failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 5); }
        double err = 0;
        for (size_t i = 0; i < (size_t)T * W; i += 4099) {
            const int tile = (int)(i / ((size_t)TILE * W)), j = (int)(i % ((size_t)TILE * W));
            double want = 0; for (int r = 0; r < ranks; ++r) want += (double)((r + 1) + (it % 5)) + (double)((tile + j) % 7) * 0.25;
            err = fmax(err, fabs(out[i] - want) / want);
        }
        double gerr; MPI_Allreduce(&err, &gerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (gerr > 1e-5) { if (!rank) fprintf(stderr, "iter %d MISMATCH err %g\n", it, gerr); MPI_Abort(MPI_COMM_WORLD, 6); }
        if (mpi_every && it % mpi_every == 0) MPI_Allreduce(junk, jout, jt * 4096, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); /* like the MoE combine */
        if (!rank && it % 50 == 0) { printf("iter %d ok\n", it); fflush(stdout); }
    }
    if (!rank) printf("ASYNC_STRESS_DONE iters=%d\n", iters);
    MPI_Finalize();
    return 0;
}
