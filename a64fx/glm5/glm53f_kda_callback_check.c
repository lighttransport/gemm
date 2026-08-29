#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_kda_12n.h"

enum { HIDDEN = 4096, TOKENS = 3 };

int main(int argc, char **argv) {
    int rank, size, local_ok, ok;
    int layer = argc > 2 ? atoi(argv[2]) : 44;
    float x[TOKENS][HIDDEN], a[TOKENS][HIDDEN], b[TOKENS][HIDDEN];
    double seq_begin, seq_elapsed, batch_begin, batch_elapsed;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 2 || size != 12) MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_kda_context_12n *ca = glm53f_kda_create_12n(argv[1], layer);
    glm53f_kda_context_12n *cb = glm53f_kda_create_12n(argv[1], layer);
    if (!ca || !cb) MPI_Abort(MPI_COMM_WORLD, 2);
    size_t state_bytes = glm53f_kda_state_bytes_12n(ca);
    unsigned char *seq_state = malloc(TOKENS * state_bytes);
    unsigned char *batch_state = malloc(TOKENS * state_bytes);
    if (!seq_state || !batch_state) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int t = 0; t < TOKENS; ++t)
        for (int i = 0; i < HIDDEN; ++i)
            x[t][i] = (float)(((i * 29 + t * 17 + 7) % 257) - 128) / 128.0f;
    /* Remove OpenMP and page-touch first-use costs from both paths. */
    local_ok = !glm53f_kda_sublayer_12n(ca, a[0], x[0]) &&
               !glm53f_kda_sublayer_batch_12n(cb, b[0], x[0], 1);
    glm53f_kda_reset_12n(ca);
    glm53f_kda_reset_12n(cb);
    MPI_Barrier(MPI_COMM_WORLD);
    seq_begin = MPI_Wtime();
    local_ok = 1;
    for (int t = 0; t < TOKENS; ++t) {
        local_ok &= !glm53f_kda_sublayer_12n(ca, a[t], x[t]);
        local_ok &= !glm53f_kda_save_state_12n(
            ca, seq_state + (size_t)t * state_bytes, state_bytes);
    }
    seq_elapsed = MPI_Wtime() - seq_begin;
    MPI_Barrier(MPI_COMM_WORLD);
    batch_begin = MPI_Wtime();
    local_ok &= !glm53f_kda_sublayer_batch_capture_12n(
        cb, b[0], x[0], TOKENS, batch_state, state_bytes);
    batch_elapsed = MPI_Wtime() - batch_begin;
    double diff2 = 0.0, ref2 = 0.0;
    for (int t = 0; t < TOKENS; ++t)
        for (int i = 0; i < HIDDEN; ++i) {
            double d = (double)a[t][i] - b[t][i];
            diff2 += d * d;
            ref2 += (double)a[t][i] * a[t][i];
        }
    double rel_l2 = sqrt(diff2 / (ref2 + 1e-30));
    local_ok &= rel_l2 < 2e-6;
    int state_ok = !memcmp(seq_state, batch_state, TOKENS * state_bytes);
    local_ok &= state_ok;
    for (int t = 0; t < TOKENS; ++t)
        for (int i = 0; i < HIDDEN; ++i)
            local_ok &= isfinite(a[t][i]);
    int all_state_ok;
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&state_ok, &all_state_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    double seq_max, batch_max;
    MPI_Reduce(&seq_elapsed, &seq_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&batch_elapsed, &batch_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank)
        printf("GLM53F_KDA_CALLBACK layer=%d tokens=%d batch=%s state=%s "
               "rel_l2=%.9g seq_ms=%.3f batch_ms=%.3f speedup=%.3f %s\n", layer, TOKENS,
               ok ? "REL_L2_OK" : "FAIL", all_state_ok ? "BIT_EXACT" : "FAIL",
               rel_l2, seq_max * 1e3, batch_max * 1e3, seq_max / batch_max,
               ok ? "PASS" : "FAIL");
    glm53f_kda_free_12n(cb);
    glm53f_kda_free_12n(ca);
    free(batch_state); free(seq_state);
    MPI_Finalize();
    return ok ? 0 : 1;
}
