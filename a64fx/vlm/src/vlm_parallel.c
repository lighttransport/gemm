/*
 * vlm_parallel.c - C11 thrd + OpenMP backends.
 */
#include "vlm_parallel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#if !defined(USE_OPENMP)
#include <threads.h>
#endif

#define VLM_MAX_THREADS 256

typedef struct {
    void  *ptr;
    size_t cap;
} vlm_scratch;

struct vlm_pool {
    int           nthreads;
    vlm_scratch   scratch[VLM_MAX_THREADS];

#if !defined(USE_OPENMP)
    /* C11 pool internals */
    thrd_t        threads[VLM_MAX_THREADS];
    mtx_t         mtx;
    cnd_t         go_cv;
    cnd_t         done_cv;
    int           pending;        /* threads still working on current job */
    int           gen;            /* job generation counter */
    int           gen_seen[VLM_MAX_THREADS];
    int           shutdown;
    /* Current job state */
    int           job_n;
    int           job_grain;
    vlm_body_fn   job_body;
    void         *job_arg;
#endif
};

static int env_int(const char *name, int def) {
    const char *s = getenv(name);
    if (!s || !*s) return def;
    int v = atoi(s);
    return v > 0 ? v : def;
}

static int resolve_nthreads(int requested) {
    if (requested > 0) return requested;
    int n = env_int("VLM_NUM_THREADS", 0);
    if (n > 0) return n;
#ifdef USE_OPENMP
    n = env_int("OMP_NUM_THREADS", 0);
    if (n > 0) return n;
    return omp_get_max_threads();
#else
    n = env_int("OMP_NUM_THREADS", 0);
    if (n > 0) return n;
    return 1;
#endif
}

void *vlm_pool_scratch(vlm_pool *p, int tid, size_t bytes) {
    if (!p || tid < 0 || tid >= VLM_MAX_THREADS) return NULL;
    vlm_scratch *s = &p->scratch[tid];
    if (s->cap >= bytes && s->ptr) return s->ptr;
    free(s->ptr);
    size_t cap = bytes < 4096 ? 4096 : bytes;
    s->ptr = aligned_alloc(64, (cap + 63) & ~(size_t)63);
    s->cap = s->ptr ? cap : 0;
    return s->ptr;
}

int vlm_pool_size(const vlm_pool *p) {
    return p ? p->nthreads : 1;
}

#ifdef USE_OPENMP

/* ────────── OpenMP backend ────────── */

vlm_pool *vlm_pool_init(int nthreads) {
    int n = resolve_nthreads(nthreads);
    if (n > VLM_MAX_THREADS) n = VLM_MAX_THREADS;
    vlm_pool *p = (vlm_pool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->nthreads = n;
    omp_set_num_threads(n);
    return p;
}

void vlm_pool_free(vlm_pool *p) {
    if (!p) return;
    for (int i = 0; i < VLM_MAX_THREADS; i++) free(p->scratch[i].ptr);
    free(p);
}

void vlm_parallel_for(vlm_pool *p, int n, int grain,
                      vlm_body_fn body, void *arg) {
    if (!body || n <= 0) return;
    int nth = p ? p->nthreads : omp_get_max_threads();
    if (nth < 1) nth = 1;
    if (grain < 1) grain = 1;
    int max_chunks = (n + grain - 1) / grain;
    if (max_chunks < nth) nth = max_chunks;

    int per = (n + nth - 1) / nth;
    /* round per up to grain multiple */
    per = ((per + grain - 1) / grain) * grain;

    #pragma omp parallel num_threads(nth)
    {
        int tid = omp_get_thread_num();
        int t0 = tid * per;
        int t1 = t0 + per;
        if (t0 > n) t0 = n;
        if (t1 > n) t1 = n;
        if (t1 > t0) body(tid, t0, t1, arg);
    }
}

#else

/* ────────── C11 thrd backend ────────── */

static int vlm_worker(void *arg_) {
    vlm_pool *p = (vlm_pool *)arg_;
    /* Lookup our tid (linear scan). Worker stores its tid via thread_local? */
    int tid = -1;
    thrd_t self = thrd_current();
    for (int i = 0; i < p->nthreads - 1; i++) {
        if (thrd_equal(p->threads[i], self)) { tid = i + 1; break; }
    }
    if (tid < 0) return -1;

    for (;;) {
        mtx_lock(&p->mtx);
        while (!p->shutdown && p->gen_seen[tid] == p->gen) cnd_wait(&p->go_cv, &p->mtx);
        if (p->shutdown) { mtx_unlock(&p->mtx); break; }
        int my_gen = p->gen;
        p->gen_seen[tid] = my_gen;
        int n = p->job_n, grain = p->job_grain;
        vlm_body_fn body = p->job_body;
        void *arg = p->job_arg;
        int nth = p->nthreads;
        mtx_unlock(&p->mtx);

        int per = (n + nth - 1) / nth;
        per = ((per + grain - 1) / grain) * grain;
        int t0 = tid * per, t1 = t0 + per;
        if (t0 > n) t0 = n;
        if (t1 > n) t1 = n;
        if (body && t1 > t0) body(tid, t0, t1, arg);

        mtx_lock(&p->mtx);
        if (--p->pending == 0) cnd_signal(&p->done_cv);
        mtx_unlock(&p->mtx);
    }
    return 0;
}

vlm_pool *vlm_pool_init(int nthreads) {
    int n = resolve_nthreads(nthreads);
    if (n < 1) n = 1;
    if (n > VLM_MAX_THREADS) n = VLM_MAX_THREADS;
    vlm_pool *p = (vlm_pool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->nthreads = n;
    if (mtx_init(&p->mtx, mtx_plain) != thrd_success) { free(p); return NULL; }
    if (cnd_init(&p->go_cv) != thrd_success) { mtx_destroy(&p->mtx); free(p); return NULL; }
    if (cnd_init(&p->done_cv) != thrd_success) {
        cnd_destroy(&p->go_cv); mtx_destroy(&p->mtx); free(p); return NULL;
    }
    /* Spawn n-1 workers; the calling thread acts as tid=0. */
    for (int i = 0; i < n - 1; i++) {
        p->gen_seen[i + 1] = 0;
        if (thrd_create(&p->threads[i], vlm_worker, p) != thrd_success) {
            fprintf(stderr, "vlm_pool: thrd_create failed at i=%d\n", i);
            p->nthreads = i + 1;  /* fall back to fewer threads */
            break;
        }
    }
    return p;
}

void vlm_pool_free(vlm_pool *p) {
    if (!p) return;
    mtx_lock(&p->mtx);
    p->shutdown = 1;
    p->gen++;
    for (int i = 1; i < p->nthreads; i++) p->gen_seen[i] = p->gen - 1;  /* force wake */
    cnd_broadcast(&p->go_cv);
    mtx_unlock(&p->mtx);
    for (int i = 0; i < p->nthreads - 1; i++) thrd_join(p->threads[i], NULL);
    cnd_destroy(&p->go_cv);
    cnd_destroy(&p->done_cv);
    mtx_destroy(&p->mtx);
    for (int i = 0; i < VLM_MAX_THREADS; i++) free(p->scratch[i].ptr);
    free(p);
}

void vlm_parallel_for(vlm_pool *p, int n, int grain,
                      vlm_body_fn body, void *arg) {
    if (!body || n <= 0) return;
    if (!p || p->nthreads <= 1) {
        body(0, 0, n, arg);
        return;
    }
    if (grain < 1) grain = 1;
    int max_chunks = (n + grain - 1) / grain;
    int nth = p->nthreads;
    if (max_chunks < nth) nth = max_chunks;

    mtx_lock(&p->mtx);
    p->job_n = n;
    p->job_grain = grain;
    p->job_body = body;
    p->job_arg = arg;
    p->gen++;
    p->pending = nth - 1;  /* workers; main thread handles tid=0 */
    cnd_broadcast(&p->go_cv);
    mtx_unlock(&p->mtx);

    /* main thread runs tid=0 chunk */
    int per = (n + nth - 1) / nth;
    per = ((per + grain - 1) / grain) * grain;
    int t0 = 0, t1 = per;
    if (t1 > n) t1 = n;
    if (t1 > t0) body(0, t0, t1, arg);

    if (nth > 1) {
        mtx_lock(&p->mtx);
        while (p->pending > 0) cnd_wait(&p->done_cv, &p->mtx);
        mtx_unlock(&p->mtx);
    }
}

#endif /* USE_OPENMP */
