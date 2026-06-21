/*
 * vlm_parallel.h - thin C11-thread / OpenMP abstraction for a64fx/vlm.
 *
 * Two backends, picked at compile time:
 *   default        : C11 <threads.h> persistent worker pool
 *   -DUSE_OPENMP=1 : OpenMP parallel for
 *
 * API is intentionally minimal — this is a runner, not a runtime.
 *
 *   vlm_pool *p = vlm_pool_init(0);      // 0 = auto (env: VLM_NUM_THREADS, OMP_NUM_THREADS, then 1)
 *   int n = vlm_pool_size(p);
 *
 *   vlm_parallel_for(p, total_iters, grain,
 *                    body_fn, user_arg);
 *   // body_fn signature: void body(int tid, int t0, int t1, void *arg);
 *
 *   void *buf = vlm_pool_scratch(p, tid, bytes);  // per-thread scratch
 *   vlm_pool_free(p);
 */
#ifndef VLM_PARALLEL_H
#define VLM_PARALLEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vlm_pool vlm_pool;
typedef void (*vlm_body_fn)(int tid, int t0, int t1, void *arg);

vlm_pool *vlm_pool_init(int nthreads);
void      vlm_pool_free(vlm_pool *p);
int       vlm_pool_size(const vlm_pool *p);

void      vlm_parallel_for(vlm_pool *p, int n, int grain,
                           vlm_body_fn body, void *arg);

/* Per-thread scratch; grows on demand. Returned pointer is valid until the
 * pool is destroyed or vlm_pool_scratch is called again with a larger size
 * for the same tid. Memory is 64-byte aligned. */
void     *vlm_pool_scratch(vlm_pool *p, int tid, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* VLM_PARALLEL_H */
