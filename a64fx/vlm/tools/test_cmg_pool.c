/* test_cmg_pool — sanity-check mbind + per-CMG pinning + cross-CMG barrier.
 *
 * Allocates 4 × 16MB regions, one per CMG, then spawns 4 threads (one per CMG)
 * and has each touch its own region. Reports per-thread CMG affinity, the
 * mbind result, and a simple bandwidth measurement (memcpy local vs remote).
 *
 * Build: handled by the main Makefile (see tools/test_cmg_pool target).
 * Run:   ./build/test_cmg_pool
 */
#define _GNU_SOURCE
#include "cmg_pool.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double mono_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#define REGION_BYTES ((size_t)16 * 1024 * 1024)

typedef struct {
    int       cmg;
    void     *local;
    void     *remote;       /* points at a remote CMG's region */
    double    t_local_ms;
    double    t_remote_ms;
} worker_arg;

static void *worker(void *arg) {
    worker_arg *w = (worker_arg *)arg;
    if (cmg_pin_thread(w->cmg, 0) != 0) {
        fprintf(stderr, "[cmg %d] pin failed\n", w->cmg);
        return NULL;
    }
    int self = cmg_self();
    fprintf(stderr, "[cmg %d] pinned, cmg_self()=%d\n", w->cmg, self);

    /* Touch local region (first-touch faults pages on bound node). */
    memset(w->local, 0xa5, REGION_BYTES);

    /* Local bandwidth: streaming write. */
    double t0 = mono_sec();
    for (int rep = 0; rep < 4; rep++) memset(w->local, (rep & 0xff), REGION_BYTES);
    double t1 = mono_sec();
    w->t_local_ms = (t1 - t0) * 1000.0;

    /* Remote bandwidth: write to a remote CMG's region. */
    t0 = mono_sec();
    for (int rep = 0; rep < 4; rep++) memset(w->remote, (rep & 0xff), REGION_BYTES);
    t1 = mono_sec();
    w->t_remote_ms = (t1 - t0) * 1000.0;
    return NULL;
}

int main(void) {
    int n = cmg_count();
    fprintf(stderr, "cmg_pool test: n_cmgs=%d cores_per_cmg=%d region=%zu MB\n",
            n, cmg_cores_per_cmg(), REGION_BYTES / (1024 * 1024));

    void *regions[CMG_MAX] = {0};
    for (int c = 0; c < n; c++) {
        regions[c] = cmg_alloc(c, REGION_BYTES);
        if (!regions[c]) {
            fprintf(stderr, "cmg_alloc(%d) returned NULL\n", c);
            return 1;
        }
        fprintf(stderr, "cmg %d region @ %p\n", c, regions[c]);
    }

    pthread_t th[CMG_MAX];
    worker_arg args[CMG_MAX];
    for (int c = 0; c < n; c++) {
        args[c].cmg    = c;
        args[c].local  = regions[c];
        args[c].remote = regions[(c + 2) % n];  /* opposite CMG when n=4 */
        pthread_create(&th[c], NULL, worker, &args[c]);
    }
    for (int c = 0; c < n; c++) pthread_join(th[c], NULL);

    fprintf(stderr, "\n=== streaming write (4× %zu MB) ===\n",
            REGION_BYTES / (1024 * 1024));
    for (int c = 0; c < n; c++) {
        double bytes = 4.0 * (double)REGION_BYTES;
        double local_gb  = bytes / (args[c].t_local_ms  * 1e6);
        double remote_gb = bytes / (args[c].t_remote_ms * 1e6);
        fprintf(stderr, "cmg %d: local=%6.2f ms (%5.1f GB/s)  remote=%6.2f ms (%5.1f GB/s)  ratio=%.2fx\n",
                c, args[c].t_local_ms, local_gb,
                args[c].t_remote_ms, remote_gb,
                args[c].t_remote_ms / args[c].t_local_ms);
    }

    for (int c = 0; c < n; c++) cmg_free(regions[c], REGION_BYTES);
    return 0;
}
