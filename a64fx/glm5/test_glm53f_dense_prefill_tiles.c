#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_dense_ffn_12n.h"

enum { WIDTH = 4096, LIMIT = 129 };
static int execute(glm53f_dense_ffn_context_12n *c, float *y, const float *x,
        int count, int tile) {
    for (int start = 0; start < count; start += tile) {
        int n = count - start; if (n > tile) n = tile;
        if (glm53f_dense_ffn_sublayer_batch_12n(c, y + (size_t)start * WIDTH,
                x + (size_t)start * WIDTH, n)) return -1;
    }
    return 0;
}
int main(int argc, char **argv) {
    int rank, ranks, failed = 0;
    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 2 || ranks != 12) MPI_Abort(MPI_COMM_WORLD, 2);
    if (setenv("GLM53F_Q2_DENSE_STAGE", argv[1], 1)) MPI_Abort(MPI_COMM_WORLD, 2);
    float *x = malloc((size_t)LIMIT * WIDTH * 4);
    float *a = malloc((size_t)LIMIT * WIDTH * 4);
    float *b = malloc((size_t)LIMIT * WIDTH * 4);
    if (!x || !a || !b) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int i = 0; i < LIMIT * WIDTH; ++i) x[i] = (float)((i * 17 + 3) % 251 - 125) / 125.0f;
    const int counts[] = {1,3,4,15,16,17,31,32,33,63,64,65,127,129};
    const int tiles[] = {16,32,64};
    for (int layer = 0; layer < 3; ++layer) {
        setenv("GLM53F_DENSE_PREFILL_TILE", "4", 1);
        glm53f_dense_ffn_context_12n *reference = glm53f_dense_ffn_create_12n(NULL, layer);
        if (!reference || glm53f_dense_ffn_batch_capacity_12n(reference) != 4) MPI_Abort(MPI_COMM_WORLD, 2);
        if (!glm53f_dense_ffn_set_batch_tile_12n(reference, 16) ||
            glm53f_dense_ffn_batch_capacity_12n(NULL) != 0) MPI_Abort(MPI_COMM_WORLD, 2);
        for (int tile_index = 0; tile_index < 3; ++tile_index) {
            char text[16]; int tile = tiles[tile_index]; snprintf(text,sizeof(text),"%d",tile);
            setenv("GLM53F_DENSE_PREFILL_TILE", text, 1);
            glm53f_dense_ffn_context_12n *candidate = glm53f_dense_ffn_create_12n(NULL, layer);
            if (!candidate || glm53f_dense_ffn_batch_capacity_12n(candidate) != tile) MPI_Abort(MPI_COMM_WORLD, 2);
            if (!glm53f_dense_ffn_set_batch_tile_12n(candidate, 65) ||
                glm53f_dense_ffn_set_batch_tile_12n(candidate, 4) ||
                glm53f_dense_ffn_batch_capacity_12n(candidate) != 4 ||
                glm53f_dense_ffn_set_batch_tile_12n(candidate, tile)) MPI_Abort(MPI_COMM_WORLD, 2);
            for (size_t ci = 0; ci < sizeof(counts)/sizeof(counts[0]); ++ci) {
                int count = counts[ci];
                if (execute(reference,a,x,count,4) || execute(candidate,b,x,count,tile)) MPI_Abort(MPI_COMM_WORLD, 2);
                uint64_t local = 0, maximum;
                for (int i = 0; i < count * WIDTH; ++i) {
                    uint32_t av,bv; memcpy(&av,a+i,4); memcpy(&bv,b+i,4);
                    local += av != bv || (av & 0x7f800000u) == 0x7f800000u || (bv & 0x7f800000u) == 0x7f800000u;
                }
                MPI_Allreduce(&local,&maximum,1,MPI_UINT64_T,MPI_MAX,MPI_COMM_WORLD);
                failed |= maximum != 0;
                if (!rank) printf("GLM53F_DENSE_TILE layer=%d tile=%d tokens=%d bit_mismatches=%" PRIu64 " %s\n",layer,tile,count,maximum,maximum?"FAIL":"PASS");
            }
            /* Paired warm timings include all four-token reductions. */
            for (int trial = 0; trial < 5; ++trial) {
                double times[2], max_times[2];
                for (int order = 0; order < 2; ++order) {
                    int which = order ^ (trial & 1);
                    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
                    for (int repeat = 0; repeat < 8; ++repeat)
                        if (execute(which?candidate:reference,which?b:a,x,128,which?tile:4)) MPI_Abort(MPI_COMM_WORLD,2);
                    times[which] = MPI_Wtime()-start;
                }
                MPI_Reduce(times,max_times,2,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
                if (!rank) printf("GLM53F_DENSE_TILE_TIME layer=%d tile=%d trial=%d baseline_seconds=%.9f candidate_seconds=%.9f ratio=%.6f\n",layer,tile,trial,max_times[0],max_times[1],max_times[0]/max_times[1]);
            }
            glm53f_dense_ffn_free_12n(candidate);
        }
        glm53f_dense_ffn_free_12n(reference);
    }
    free(b); free(a); free(x);
    if (!rank) printf("GLM53F_DENSE_TILES_%s\n",failed?"FAIL":"PASS");
    MPI_Finalize(); return failed?1:0;
}
