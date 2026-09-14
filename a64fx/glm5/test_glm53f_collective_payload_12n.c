/* Isolated payload/arrival-skew sweep. No weights; not an integrated-model
 * speed prediction. Registered capacity and arithmetic are checked explicitly. */
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm53f_collective_12n.h"

int main(int argc, char **argv) {
    enum { HIDDEN = 4096, TOKENS = 32, REPEATS = 24 };
    int rank, ranks, failed = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (ranks != 12) MPI_Abort(MPI_COMM_WORLD, 2);
    if (glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"), HIDDEN * TOKENS)) {
        if (!rank) fprintf(stderr, "COLLECTIVE_PAYLOAD init_failed max_count=%d\n", HIDDEN * TOKENS);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    float *input = malloc((size_t)HIDDEN * TOKENS * sizeof(float));
    float *output = malloc((size_t)HIDDEN * TOKENS * sizeof(float));
    if (!input || !output) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int i = 0; i < HIDDEN * TOKENS; ++i) input[i] = rank + 1 + (i % 31) * 0.0625f;
    for (int skew = 0; skew < 2; ++skew)
        for (int slab = 1; slab <= TOKENS; slab *= 2) {
            double elapsed = 0, collective = 0;
            for (int rep = -2; rep < REPEATS; ++rep) {
                MPI_Barrier(MPI_COMM_WORLD);
                double begin = MPI_Wtime(), calls = 0;
                for (int t = 0; t < TOKENS; t += slab) {
                    double deadline = MPI_Wtime() + (skew ? rank % 3 * 20e-6 : 0);
                    while (MPI_Wtime() < deadline) { }
                    double call = MPI_Wtime();
                    if (glm53f_sum_allreduce_12n(input + (size_t)t * HIDDEN,
                            output + (size_t)t * HIDDEN, slab * HIDDEN)) MPI_Abort(MPI_COMM_WORLD, 3);
                    calls += MPI_Wtime() - call;
                }
                if (rep >= 0) { elapsed += MPI_Wtime() - begin; collective += calls; }
            }
            for (int i = 0; i < HIDDEN * TOKENS; ++i) {
                float expected = ranks * (ranks + 1) / 2 + ranks * (i % 31) * 0.0625f;
                failed |= output[i] != expected || !isfinite(output[i]);
            }
            double maximum, call_min, call_max;
            MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&collective, &call_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&collective, &call_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            int any;
            MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (!rank) printf("COLLECTIVE_PAYLOAD slab=%d count=%d skew=%d "
                "wall_us_token=%.3f call_min_us_token=%.3f call_max_us_token=%.3f exact=%d\n",
                slab, slab * HIDDEN, skew, maximum * 1e6 / (REPEATS * TOKENS),
                call_min * 1e6 / (REPEATS * TOKENS), call_max * 1e6 / (REPEATS * TOKENS), !any);
            failed = any;
        }
    free(output); free(input);
    glm53f_collective_free_12n(); MPI_Finalize();
    return failed;
}
