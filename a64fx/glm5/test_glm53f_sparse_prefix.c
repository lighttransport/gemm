/* Real-weight 12-rank boundary check for the replicated CP prefix path. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_sparse_12n.h"

int main(int argc, char **argv) {
    int rank, ranks, mismatch = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 2 || ranks != 12) MPI_Abort(MPI_COMM_WORLD, 2);
    unsetenv("GLM53F_CP_PREFIX_REPLICATED");
    glm53f_sparse_context_12n *reference = glm53f_sparse_create_format_12n(argv[1], 43, 65536, 1);
    setenv("GLM53F_CP_PREFIX_REPLICATED", "1", 1);
    glm53f_sparse_context_12n *candidate = glm53f_sparse_create_format_12n(argv[1], 43, 65536, 1);
    float x[4096], a[4096], b[4096];
    if (!reference || !candidate) MPI_Abort(MPI_COMM_WORLD, 2);
    double reference_seconds = 0, candidate_seconds = 0;
    for (int token = 0; token < 513; ++token) {
        for (int d = 0; d < 4096; ++d)
            x[d] = ((token * 17LL + d * 31) % 251 - 125) * 0.002f;
        double begin = MPI_Wtime();
        if (glm53f_sparse_sublayer_12n(reference, a, x)) MPI_Abort(MPI_COMM_WORLD, 3);
        reference_seconds += MPI_Wtime() - begin;
        begin = MPI_Wtime();
        if (glm53f_sparse_sublayer_12n(candidate, b, x)) MPI_Abort(MPI_COMM_WORLD, 3);
        candidate_seconds += MPI_Wtime() - begin;
        if (memcmp(a, b, sizeof(a))) {
            mismatch = token + 1;
            break;
        }
    }
    int any_mismatch;
    double local[2] = {reference_seconds, candidate_seconds}, maximum[2];
    MPI_Reduce(&mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, maximum, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("SPARSE_PREFIX positions=513 first_mismatch=%d reference_ms=%.3f candidate_ms=%.3f %s\n",
        any_mismatch, maximum[0] * 1e3, maximum[1] * 1e3, any_mismatch ? "FAIL" : "PASS");
    glm53f_sparse_free_12n(candidate);
    glm53f_sparse_free_12n(reference);
    MPI_Finalize();
    return mismatch != 0;
}
