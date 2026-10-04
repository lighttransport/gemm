/* Diagnostic synchronization profile for decode, linked only into benchmark
 * builds with -Wl,--wrap=glm53f_team_dispatch,--wrap=glm53f_sum_allreduce_12n,
 * --wrap=__kmpc_barrier. Inactive unless GLM53F_PROFILE_SYNC is set:
 *   1: count team dispatches, OpenMP barriers (thread 0) and decode allreduces
 *      with their wall time;
 *   2: additionally time a PMPI_Barrier before each allreduce, separating
 *      arrival skew from collective latency (adds one barrier per allreduce).
 * Production sources are unchanged; results are not throughput measurements. */
#define _GNU_SOURCE
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*glm53f_team_callback)(void *);
void __real_glm53f_team_dispatch(glm53f_team_callback fn, void *context);
int __real_glm53f_sum_allreduce_12n(const float *input, float *output, int count);
void __real___kmpc_barrier(void *loc, int32_t gtid);

static int sync_mode = -1;
static struct {
    uint64_t dispatches, barriers, allreduces, allreduce_floats;
    double dispatch_s, barrier_s, allreduce_s, skew_s;
} sync_stats;

static inline double sync_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static inline int sync_on(void) {
    if (sync_mode < 0) {
        const char *e = getenv("GLM53F_PROFILE_SYNC");
        sync_mode = e ? atoi(e) : 0;
    }
    return sync_mode;
}

void __wrap_glm53f_team_dispatch(glm53f_team_callback fn, void *context) {
    if (!sync_on()) { __real_glm53f_team_dispatch(fn, context); return; }
    double begin = sync_now();
    __real_glm53f_team_dispatch(fn, context);
    sync_stats.dispatch_s += sync_now() - begin;
    ++sync_stats.dispatches;
}

void __wrap___kmpc_barrier(void *loc, int32_t gtid) {
    if (!sync_on() || omp_get_thread_num() != 0) { __real___kmpc_barrier(loc, gtid); return; }
    double begin = sync_now();
    __real___kmpc_barrier(loc, gtid);
    sync_stats.barrier_s += sync_now() - begin;
    ++sync_stats.barriers;
}

int __wrap_glm53f_sum_allreduce_12n(const float *input, float *output, int count) {
    const int mode = sync_on();
    if (!mode) return __real_glm53f_sum_allreduce_12n(input, output, count);
    double begin = sync_now();
    if (mode >= 2) {
        PMPI_Barrier(MPI_COMM_WORLD);
        double ready = sync_now();
        sync_stats.skew_s += ready - begin;
        begin = ready;
    }
    int rc = __real_glm53f_sum_allreduce_12n(input, output, count);
    sync_stats.allreduce_s += sync_now() - begin;
    ++sync_stats.allreduces;
    sync_stats.allreduce_floats += (uint64_t)count;
    return rc;
}

void glm53f_sync_profile_reset(void) {
    sync_stats = (__typeof__(sync_stats)){0};
}

/* Rank 0 prints per-position averages; positions = decoded positions. */
void glm53f_sync_profile_report(const char *label, long positions) {
    int rank = 0;
    if (!sync_on() || positions <= 0) return;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank)
        printf("GLM53F_SYNC_PROFILE label=%s mode=%d positions=%ld dispatches=%.1f dispatch_ms=%.3f "
               "barriers=%.1f barrier_wait_ms=%.3f allreduces=%.1f allreduce_floats=%.0f "
               "allreduce_ms=%.3f skew_ms=%.3f per_pos\n",
               label, sync_mode, positions,
               (double)sync_stats.dispatches / positions, sync_stats.dispatch_s * 1e3 / positions,
               (double)sync_stats.barriers / positions, sync_stats.barrier_s * 1e3 / positions,
               (double)sync_stats.allreduces / positions,
               (double)sync_stats.allreduce_floats / (sync_stats.allreduces ? sync_stats.allreduces : 1),
               sync_stats.allreduce_s * 1e3 / positions, sync_stats.skew_s * 1e3 / positions);
    glm53f_sync_profile_reset();
}
