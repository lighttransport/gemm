#define _GNU_SOURCE
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utofu.h>
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"
#include "glm53f_collective_12n.h"

enum { GLM53F_COLLECTIVE_MAX_NODES = 32 };
static tp_comm glm53f_comm;
static utofu_vcq_hdl_t glm53f_vcq;
static int glm53f_utofu_active;

static void glm53f_mpi_barrier(void) { MPI_Barrier(MPI_COMM_WORLD); }

static int glm53f_read_topology(const char *path,
        uint8_t coords[][TOFU_NCOORDS]) {
    FILE *file = fopen(path, "r");
    char line[256];
    int count = 0;
    if (!file) return -1;
    while (fgets(line, sizeof(line), file)) {
        unsigned rank, c[TOFU_NCOORDS];
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%u %u %u %u %u %u %u", &rank, &c[0], &c[1],
                &c[2], &c[3], &c[4], &c[5]) != 7 || rank != (unsigned)count ||
                count >= GLM53F_COLLECTIVE_MAX_NODES) {
            fclose(file);
            return -1;
        }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[count][k] = (uint8_t)c[k];
        count++;
    }
    fclose(file);
    return count;
}

int glm53f_collective_init_12n(const char *path, int max_count) {
    int rank, ranks, rc, topo_count, physical_rank = -1;
    uint8_t topology[GLM53F_COLLECTIVE_MAX_NODES][TOFU_NCOORDS];
    uint8_t mine[TOFU_NCOORDS];
    utofu_tni_id_t *tnis = NULL;
    utofu_tni_id_t tni;
    size_t ntni = 0;
    utofu_vcq_id_t peers[GLM53F_COLLECTIVE_MAX_NODES];
    if (glm53f_utofu_active) return 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (!path || ranks != 12 || max_count < 1 ||
            (topo_count = glm53f_read_topology(path, topology)) != ranks) {
        if (getenv("GLM53F_UTOFU_DEBUG"))
            fprintf(stderr, "GLM53F_UTOFU init preflight path=%s ranks=%d max=%d topo=%d\n",
                    path ? path : "(null)", ranks, max_count,
                    path ? topo_count : -1);
        return -1;
    }
    if (utofu_query_my_coords(mine) != UTOFU_SUCCESS) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU query coords failed rank=%d\n", rank);
        return -1;
    }
    for (int r = 0; r < ranks; r++)
        if (!memcmp(mine, topology[r], TOFU_NCOORDS)) physical_rank = r;
    if (physical_rank != rank) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU rank map failed rank=%d physical=%d\n", rank, physical_rank);
        return -1;
    }
    rc = utofu_get_onesided_tnis(&tnis, &ntni);
    if (rc != UTOFU_SUCCESS || ntni < 1) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU get tni failed rank=%d rc=%d ntni=%zu\n", rank, rc, ntni);
        free(tnis); return -1;
    }
    tni = tnis[0];
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &glm53f_vcq);
    free(tnis);
    if (rc != UTOFU_SUCCESS) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU create vcq failed rank=%d rc=%d\n", rank, rc);
        return -1;
    }
    for (int r = 0; r < ranks; r++) {
        rc = utofu_construct_vcq_id(topology[r], tni, DEMO_CQ_ID, DEMO_CMP_ID,
                                    &peers[r]);
        if (rc != UTOFU_SUCCESS) {
            if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU construct peer failed rank=%d peer=%d rc=%d\n", rank, r, rc);
            return -1;
        }
        utofu_set_vcq_id_path(&peers[r], NULL);
    }
    if (tp_comm_init(&glm53f_comm, glm53f_vcq, peers, rank, ranks,
                     max_count, glm53f_mpi_barrier)) {
        if (getenv("GLM53F_UTOFU_DEBUG")) fprintf(stderr, "GLM53F_UTOFU tp_comm_init failed rank=%d\n", rank);
        return -1;
    }
    glm53f_utofu_active = 1;
    if (!rank) fprintf(stderr, "GLM53F_COLLECTIVE mode=utofu max_count=%d\n", max_count);
    return 0;
}

void glm53f_collective_free_12n(void) {
    if (!glm53f_utofu_active) return;
    tp_comm_free(&glm53f_comm);
    utofu_free_vcq(glm53f_vcq);
    glm53f_utofu_active = 0;
}

int glm53f_collective_is_utofu_12n(void) { return glm53f_utofu_active; }

int glm53f_sum_allreduce_12n(const float *input, float *output, int count) {
    if (!input || !output || count < 1) return -1;
    if (!glm53f_utofu_active) {
        /* Optional payload-sharded experiment.  Reduce-scatter computes each
         * rank's disjoint output slice, then an allgatherv reconstructs the
         * full vector.  The default MPI_Allreduce path remains unchanged. */
        if (getenv("GLM53F_SPLIT_AR")) {
            int rank, nr, rc[GLM53F_COLLECTIVE_MAX_NODES], ds[GLM53F_COLLECTIVE_MAX_NODES];
            static float *slice;
            static int slice_cap;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &nr);
            if (nr > GLM53F_COLLECTIVE_MAX_NODES) return -1;
            int off = 0;
            for (int r = 0; r < nr; ++r) {
                rc[r] = count / nr + (r < count % nr);
                ds[r] = off;
                off += rc[r];
            }
            if (slice_cap < rc[rank]) {
                float *p = realloc(slice, (size_t)rc[rank] * sizeof(*slice));
                if (!p) return -1;
                slice = p;
                slice_cap = rc[rank];
            }
            if (MPI_Reduce_scatter(input, slice, rc, MPI_FLOAT, MPI_SUM,
                                   MPI_COMM_WORLD) != MPI_SUCCESS ||
                MPI_Allgatherv(slice, rc[rank], MPI_FLOAT, output, rc, ds,
                               MPI_FLOAT, MPI_COMM_WORLD) != MPI_SUCCESS)
                return -1;
            return 0;
        }
        return MPI_Allreduce(input, output, count, MPI_FLOAT, MPI_SUM,
                             MPI_COMM_WORLD) == MPI_SUCCESS ? 0 : -1;
    }
    if (input != output) memcpy(output, input, (size_t)count * sizeof(float));
    tp_allreduce_sum(&glm53f_comm, output, count);
    return 0;
}
