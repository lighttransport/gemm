#ifndef GLM53F_CLOCK_H
#define GLM53F_CLOCK_H
#include <time.h>
/* Profiling may run on compute workers while the communication owner is in
 * MPI. A local monotonic clock avoids even concurrent MPI_Wtime calls under
 * MPI_THREAD_SERIALIZED. Whole-run times are still reduced across ranks. */
static inline double glm53f_clock(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#endif
