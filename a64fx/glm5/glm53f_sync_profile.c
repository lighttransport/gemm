/* Diagnostic synchronization profile for decode, linked only into benchmark
 * builds with -Wl,--wrap=glm53f_team_dispatch,--wrap=glm53f_sum_allreduce_12n,
 * --wrap=__kmpc_barrier. Inactive unless GLM53F_PROFILE_SYNC is set:
 *   1: count team dispatches, OpenMP barriers (thread 0) and decode allreduces
 *      with their wall time;
 *   2: additionally time a PMPI_Barrier before each allreduce, separating
 *      arrival skew from collective latency (adds one barrier per allreduce);
 *   3: mode 1 plus per-call-site barrier/dispatch counts and wait time
 *      (return addresses; resolve offline with addr2line -f -e BINARY);
 *   4: mode 3 plus per-site thread arrival spread (max - min arrival time of
 *      all team threads at each barrier), i.e. phase load imbalance. Approximate:
 *      a released thread may store its next arrival before thread 0 reads.
 * Production sources are unchanged; results are not throughput measurements. */
#define _GNU_SOURCE
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

enum { SITES = 512 };
static struct { uintptr_t pc; uint64_t calls; double seconds, spread; char kind; } sync_site[SITES];
static void sync_site_add(uintptr_t pc, double seconds, char kind, double spread) {
    unsigned h = (unsigned)((pc >> 2) * 2654435761u) % SITES;
    for (int i = 0; i < SITES; ++i, h = (h + 1) % SITES) {
        if (!sync_site[h].pc) { sync_site[h].pc = pc; sync_site[h].kind = kind; }
        if (sync_site[h].pc == pc) { ++sync_site[h].calls; sync_site[h].seconds += seconds; sync_site[h].spread += spread; return; }
    }
}
static double sync_arrive[256 * 8];   /* per-thread arrival times, one cache-line stride apart */
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
    const double t = sync_now() - begin;
    sync_stats.dispatch_s += t;
    ++sync_stats.dispatches;
    if (sync_mode >= 3) sync_site_add((uintptr_t)__builtin_return_address(0), t, 'D', 0.0);
}

void __wrap___kmpc_barrier(void *loc, int32_t gtid) {
    const int tid = omp_get_thread_num();
    if (!sync_on() || (tid != 0 && sync_mode < 4)) { __real___kmpc_barrier(loc, gtid); return; }
    double begin = sync_now();
    if (sync_mode >= 4 && tid < 256) sync_arrive[tid * 8] = begin;
    if (tid != 0) { __real___kmpc_barrier(loc, gtid); return; }
    __real___kmpc_barrier(loc, gtid);
    const double t = sync_now() - begin;
    sync_stats.barrier_s += t;
    ++sync_stats.barriers;
    double spread = 0.0;
    if (sync_mode >= 4) {   /* every thread stored its arrival before the barrier completed */
        const int nt = omp_get_num_threads() < 256 ? omp_get_num_threads() : 256;
        double lo = sync_arrive[0], hi = sync_arrive[0];
        for (int i = 1; i < nt; ++i) { const double a = sync_arrive[i * 8]; if (a < lo) lo = a; if (a > hi) hi = a; }
        spread = hi - lo;
    }
    if (sync_mode >= 3) sync_site_add((uintptr_t)__builtin_return_address(0), t, 'B', spread);
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
    memset(sync_site, 0, sizeof(sync_site));
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
    if (!rank && sync_mode >= 3)
        for (int i = 0; i < SITES; ++i)
            if (sync_site[i].pc)
                printf("GLM53F_SYNC_SITE label=%s kind=%c pc=%#lx per_pos=%.2f us_per_pos=%.2f spread_us_per_pos=%.2f\n",
                       label, sync_site[i].kind, (unsigned long)sync_site[i].pc, (double)sync_site[i].calls / positions,
                       sync_site[i].seconds * 1e6 / positions, sync_site[i].spread * 1e6 / positions);
    glm53f_sync_profile_reset();
}
