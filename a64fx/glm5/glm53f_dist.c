#include "glm53f_dist.h"
#include <string.h>

int glm53f_dist_init(glm53f_dist *d, MPI_Comm world, const glm53f_parallel_config *c) {
    if (!d || world == MPI_COMM_NULL) return -1;
    memset(d, 0, sizeof(*d));
    d->world = world; d->tp = d->pipeline = MPI_COMM_NULL;
    int rank, size, valid, all, thread_level, main_thread;
    if (MPI_Comm_rank(world, &rank) != MPI_SUCCESS || MPI_Comm_size(world, &size) != MPI_SUCCESS) return -1;
    if (MPI_Query_thread(&thread_level) != MPI_SUCCESS ||
        MPI_Is_thread_main(&main_thread) != MPI_SUCCESS) return -1;
    valid = thread_level >= MPI_THREAD_SERIALIZED && main_thread &&
        !glm53f_parallel_map_rank(c, rank, size, &d->map);
    if (MPI_Allreduce(&valid, &all, 1, MPI_INT, MPI_MIN, world) != MPI_SUCCESS || !all) return -1;
    int fields[4] = {(int)c->layout, c->microbatch, c->cuts[0], c->cuts[1]}, low[4], high[4];
    if (MPI_Allreduce(fields, low, 4, MPI_INT, MPI_MIN, world) != MPI_SUCCESS ||
        MPI_Allreduce(fields, high, 4, MPI_INT, MPI_MAX, world) != MPI_SUCCESS ||
        memcmp(low, high, sizeof(low))) return -1;
    d->config = *c;
    if (MPI_Comm_split(world, d->map.stage, d->map.tp_rank, &d->tp) != MPI_SUCCESS ||
        MPI_Comm_split(world, d->map.tp_rank, d->map.stage, &d->pipeline) != MPI_SUCCESS) {
        glm53f_dist_free(d); return -1;
    }
    if (MPI_Comm_set_errhandler(d->tp, MPI_ERRORS_RETURN) != MPI_SUCCESS ||
        MPI_Comm_set_errhandler(d->pipeline, MPI_ERRORS_RETURN) != MPI_SUCCESS) {
        glm53f_dist_free(d); return -1;
    }
    d->initialized = 1;
    return 0;
}
void glm53f_dist_free(glm53f_dist *d) {
    if (!d) return;
    if (d->pipeline != MPI_COMM_NULL) MPI_Comm_free(&d->pipeline);
    if (d->tp != MPI_COMM_NULL) MPI_Comm_free(&d->tp);
    d->initialized = 0;
}
int glm53f_dist_sum(const glm53f_dist *d, const float *input, float *output, int count) {
    if (!d || !d->initialized || !input || !output || count < 1) return -1;
    return MPI_Allreduce(input == output ? MPI_IN_PLACE : input, output,
        count, MPI_FLOAT, MPI_SUM, d->tp) == MPI_SUCCESS ? 0 : -1;
}
