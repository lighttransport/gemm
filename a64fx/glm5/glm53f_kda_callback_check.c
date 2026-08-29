#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_kda_12n.h"

enum { HIDDEN = 4096 };

int main(int argc, char **argv) {
    int rank, size, local_ok, ok;
    int layer = argc > 2 ? atoi(argv[2]) : 44;
    float x[HIDDEN], a[HIDDEN], b[HIDDEN];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 2 || size != 12) MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_kda_context_12n *ca = glm53f_kda_create_12n(argv[1], layer);
    glm53f_kda_context_12n *cb = glm53f_kda_create_12n(argv[1], layer);
    if (!ca || !cb) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int i = 0; i < HIDDEN; ++i)
        x[i] = (float)(((i * 29 + 7) % 257) - 128) / 128.0f;
    local_ok = !glm53f_kda_sublayer_12n(ca, a, x) &&
               !glm53f_kda_sublayer_12n(cb, b, x) &&
               !memcmp(a, b, sizeof(a));
    for (int i = 0; i < HIDDEN; ++i)
        local_ok &= isfinite(a[i]);
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!rank)
        printf("GLM53F_KDA_CALLBACK layer=%d independent_state=%s %s\n",
               layer, ok ? "BIT_EXACT" : "FAIL", ok ? "PASS" : "FAIL");
    glm53f_kda_free_12n(cb);
    glm53f_kda_free_12n(ca);
    MPI_Finalize();
    return ok ? 0 : 1;
}
