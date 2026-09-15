/* Isolated payload/arrival-skew sweep. No weights; not an integrated-model
 * speed prediction. Registered capacity and arithmetic are checked explicitly. */
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
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
    int algorithm = argc > 1 ? atoi(argv[1]) : 0;
    if (glm53f_collective_prefill_algorithm_12n(algorithm)) MPI_Abort(MPI_COMM_WORLD, 2);
    float *input = malloc((size_t)HIDDEN * TOKENS * sizeof(float));
    float *output = malloc((size_t)HIDDEN * TOKENS * sizeof(float));
    if (!input || !output) MPI_Abort(MPI_COMM_WORLD, 2);
    float *reference = malloc((size_t)HIDDEN * TOKENS * sizeof(float));
    if (!reference) MPI_Abort(MPI_COMM_WORLD, 2);
    uint32_t random=1234567u+rank*104729u;
    for (int i=0;i<HIDDEN*TOKENS;++i) {
        random=random*1664525u+1013904223u;
        input[i]=ldexpf((float)(int32_t)random, -30-(i%12));
    }
    if (glm53f_sum_allreduce_12n(input,reference,HIDDEN*TOKENS) ||
        glm53f_sum_allreduce_slabs_12n(input,output,TOKENS,HIDDEN,32)) MPI_Abort(MPI_COMM_WORLD,3);
    double err=0,norm=0;
    for (int i=0;i<HIDDEN*TOKENS;++i) {
        double d=(double)output[i]-reference[i];
        err+=d*d; norm+=(double)reference[i]*reference[i];
    }
    int exact=memcmp(output,reference,(size_t)HIDDEN*TOKENS*sizeof(float))==0, all_exact;
    MPI_Allreduce(&exact,&all_exact,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if (!rank) printf("COLLECTIVE_TREE_REFERENCE algorithm=%d exact=%d rel_l2=%.9g\n",algorithm,all_exact,sqrt(err/norm));
    if (algorithm>=3) failed |= !all_exact;
    failed |= glm53f_collective_capacity_12n() != HIDDEN * TOKENS;
    failed |= glm53f_sum_allreduce_12n(input, output, HIDDEN * TOKENS + 1) != -1;
    failed |= glm53f_sum_allreduce_12n(input, output, 0) != -1;
    failed |= glm53f_sum_allreduce_slabs_12n(input, output, 1, HIDDEN * TOKENS + 1, 32) != -1;
    /* No barrier between varying-size prefill and scalar operations. Exercise
     * footer placement, slot reuse, and transition back into the old protocol. */
    const int widths[]={4096,513,2048,17}, tiles[]={32,31,7,1};
    for (int pass=0;pass<256;++pass) {
        int width=widths[pass%4], tokens=tiles[(pass/4)%4], count=width*tokens;
        for (int i=0;i<count;++i) input[i]=(float)(rank+1)+(i%31)*.0625f+(pass%7)*.125f;
        if (glm53f_sum_allreduce_slabs_12n(input,output,tokens,width,32)) MPI_Abort(MPI_COMM_WORLD,3);
        for (int i=0;i<count;++i) failed |= output[i] !=
            ranks*(ranks+1)/2+ranks*((i%31)*.0625f+(pass%7)*.125f);
    }
    for (int i = 0; i < HIDDEN * TOKENS; ++i) input[i] = rank + 1 + (i % 31) * 0.0625f;
    if (glm53f_sum_allreduce_slabs_12n(input, output, 31, HIDDEN, 8)) MPI_Abort(MPI_COMM_WORLD, 3);
    for (int i = 0; i < HIDDEN * 31; ++i)
        failed |= output[i] != ranks * (ranks + 1) / 2 + ranks * (i % 31) * 0.0625f;
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
                    if (glm53f_sum_allreduce_slabs_12n(input + (size_t)t * HIDDEN,
                            output + (size_t)t * HIDDEN, slab, HIDDEN, slab)) MPI_Abort(MPI_COMM_WORLD, 3);
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
            if (!rank) printf("COLLECTIVE_PAYLOAD algorithm=%d slab=%d count=%d skew=%d "
                "wall_us_token=%.3f call_min_us_token=%.3f call_max_us_token=%.3f exact=%d\n",
                algorithm, slab, slab * HIDDEN, skew, maximum * 1e6 / (REPEATS * TOKENS),
                call_min * 1e6 / (REPEATS * TOKENS), call_max * 1e6 / (REPEATS * TOKENS), !any);
            failed = any;
        }
    free(reference); free(output); free(input);
    glm53f_collective_free_12n(); MPI_Finalize();
    return failed;
}
