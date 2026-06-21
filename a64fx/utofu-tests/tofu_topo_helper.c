/*
 * tofu_topo_helper - external MPI helper that discovers per-node Tofu network
 * coordinates ONCE and writes them to a plain-text topology file.
 *
 * This is the only program in the demo that uses MPI. The actual uTofu
 * communication app (tofu_put_demo) reads the file this writes and makes zero
 * MPI calls. Node topology is assumed fixed for the duration of the job.
 *
 * Build:
 *   fccpx -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall \
 *         -o tofu_topo_helper tofu_topo_helper.c -lmpi -ltofucom
 *   (fallback: mpifccpx -Nclang ... -o tofu_topo_helper tofu_topo_helper.c -ltofucom)
 *
 * Run (inside a pjsub node=N allocation):
 *   mpiexec -np <N> ./tofu_topo_helper
 *
 * Output file format (one line per rank):
 *   # rank x y z a b c
 *   0 a b c d e f
 *   1 a b c d e f
 *   ...
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <utofu.h>

#include "tofu_demo.h"

int main(int argc, char **argv)
{
    int rank, nprocs, rc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Confirm a one-sided TNI exists; this also matches what tofu_put_demo
     * will use (index DEMO_TNI_INDEX). We don't store the tni in the file --
     * it is a fixed convention -- but we sanity check it is present. */
    utofu_tni_id_t *tni_ids = NULL;
    size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS || num_tnis <= DEMO_TNI_INDEX) {
        fprintf(stderr, "rank %d: utofu_get_onesided_tnis failed (rc=%d, ntni=%zu)\n",
                rank, rc, num_tnis);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    free(tni_ids);

    /* This node's 6D Tofu coordinates. */
    uint8_t coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(coords);
    if (rc != UTOFU_SUCCESS) {
        fprintf(stderr, "rank %d: utofu_query_my_coords failed (rc=%d)\n", rank, rc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Gather every rank's coordinates to rank 0. */
    uint8_t *all = NULL;
    if (rank == 0) {
        all = malloc((size_t)nprocs * TOFU_NCOORDS);
        if (!all) {
            fprintf(stderr, "rank 0: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Gather(coords, TOFU_NCOORDS, MPI_UINT8_T,
               all, TOFU_NCOORDS, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *f = fopen(TOPO_PATH, "w");
        if (!f) {
            perror("fopen " TOPO_PATH);
            free(all);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(f, "# rank x y z a b c\n");
        for (int r = 0; r < nprocs; r++) {
            const uint8_t *c = &all[(size_t)r * TOFU_NCOORDS];
            fprintf(f, "%d %u %u %u %u %u %u\n", r,
                    c[0], c[1], c[2], c[3], c[4], c[5]);
        }
        fclose(f);
        free(all);
        printf("wrote %s for %d node(s)\n", TOPO_PATH, nprocs);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
