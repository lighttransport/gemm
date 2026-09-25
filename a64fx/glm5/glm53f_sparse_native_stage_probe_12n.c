#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "glm53f_sparse_12n.h"
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, failed = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int layer = 3; layer < 45; layer += 4)
        if (glm53f_sparse_native_stage_probe_12n(layer)) {
            fprintf(stderr, "rank=%d sparse native probe failed layer=%d\n", rank, layer);
            failed = 1;
            break;
        }
    if (!failed)
        printf("SENTINEL glm53f_sparse_native_stage_probe=PASS rank=%d\n", rank);
    MPI_Finalize();
    return failed;
}
