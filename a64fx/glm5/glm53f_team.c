/* Epoch publication with per-worker completion lines, inspired by Strata's
 * reusable CPU pool. No Strata source is copied. One controller, one live job. */
#include "glm53f_team.h"
#include <omp.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

enum { MAX_WORKERS = 128 };
static struct {
    _Alignas(256) _Atomic uint64_t epoch;
    glm53f_team_callback callback;
    void *context;
    int active, workers;
    struct { _Alignas(256) _Atomic uint64_t epoch; } completed[MAX_WORKERS];
} team;
static inline void relax(void) {
#ifdef __aarch64__
    __asm__ __volatile__("yield" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}
int glm53f_team_active(void) { return team.active; }
void glm53f_team_dispatch(glm53f_team_callback fn, void *context) {
    if (!team.active || !fn || omp_get_thread_num() != 0) abort();
    uint64_t epoch = atomic_load_explicit(&team.epoch, memory_order_relaxed) + 1;
    team.callback = fn; team.context = context;
    atomic_store_explicit(&team.epoch, epoch, memory_order_release);
    fn(context);
    for (int t = 1; t < team.workers; ++t)
        while (atomic_load_explicit(&team.completed[t].epoch, memory_order_acquire) != epoch) relax();
}
void glm53f_team_run(glm53f_team_callback controller, void *context) {
    if (!controller || omp_in_parallel() || team.active) abort();
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp master
        {
            team.workers = omp_get_num_threads();
            if (team.workers > MAX_WORKERS) abort();
            team.active = 1;
        }
#pragma omp barrier
        if (!tid) {
            controller(context);
            team.callback = NULL;
            atomic_fetch_add_explicit(&team.epoch, 1, memory_order_release);
        } else {
            uint64_t seen = 0;
            for (;;) {
                uint64_t epoch;
                while ((epoch = atomic_load_explicit(&team.epoch, memory_order_acquire)) == seen) relax();
                /* Capture the published epoch BEFORE invoking the callback.
                 * A delayed worker cannot skip publication or claim a new job
                 * using an old context after its completion is observed. */
                seen = epoch;
                glm53f_team_callback fn = team.callback;
                void *arg = team.context;
                if (!fn) break;
                fn(arg);
                atomic_store_explicit(&team.completed[tid].epoch, epoch, memory_order_release);
            }
        }
#pragma omp barrier
#pragma omp master
        {
            team.active = 0;
            atomic_store_explicit(&team.epoch, 0, memory_order_relaxed);
            for (int t = 0; t < team.workers; ++t)
                atomic_store_explicit(&team.completed[t].epoch, 0, memory_order_relaxed);
        }
    }
}
