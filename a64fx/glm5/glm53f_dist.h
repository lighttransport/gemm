#ifndef GLM53F_DIST_H
#define GLM53F_DIST_H
#include <mpi.h>
#include "glm53f_parallel.h"

typedef struct {
    glm53f_parallel_config config;
    glm53f_parallel_map map;
    MPI_Comm world, tp, pipeline;
    int initialized;
    /* Optional PP compact core path, borrowed alongside model images. */
    const char *core_stage;
} glm53f_dist;
/* Collective on world, on the MPI main/controller thread. Requires at least
 * MPI_THREAD_SERIALIZED. Communicators are owned; world is borrowed. The
 * context must outlive components borrowing it. Free after init (also after
 * a failed init), never on an uninitialized object. */
int glm53f_dist_init(glm53f_dist *d, MPI_Comm world, const glm53f_parallel_config *config);
void glm53f_dist_free(glm53f_dist *d);
int glm53f_dist_sum(const glm53f_dist *d, const float *input, float *output, int count);
#endif
