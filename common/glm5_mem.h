/* glm5_mem.h - GLM5 NUMA-aware aligned allocator (replaces raw malloc/calloc/free).
 *
 * A64FX has 4 CMGs each with its own HBM2 stack (NUMA node). The GLM5 EP runner uses one MPI
 * rank per node with OpenMP threads spread across all 4 CMGs (esp. for compute-bound prefill
 * at OMP_NUM_THREADS=48). Replicated weights / KV caches / scratch are read by threads on ALL
 * CMGs, so the backing pages should be INTERLEAVED across the 4 HBM nodes for balanced
 * bandwidth (a single-node placement bottlenecks 3 CMGs on one HBM stack).
 *
 * glm5_amalloc: 256-byte aligned (matches the ds4f BF16_PV / activation convention; SVE-friendly,
 * cache-line-pair aligned), and for large blocks (>= GLM5_NUMA_THRESH) applies a best-effort
 * MPOL_INTERLEAVE over all NUMA nodes via mbind. The mbind is NON-FATAL: on failure pages fall
 * back to first-touch, so correctness never depends on it (set GLM5_NO_NUMA=1 to disable).
 * Memory is plain aligned_alloc storage -> free()/glm5_afree() work normally (no size tracking).
 */
#ifndef GLM5_MEM_H
#define GLM5_MEM_H
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif

#define GLM5_ALIGN       256u          /* default alignment (bytes) */
#define GLM5_NUMA_THRESH (1u<<20)      /* interleave allocations >= 1 MiB */

#if defined(__linux__) && defined(SYS_mbind)
/* best-effort MPOL_INTERLEAVE(=3) across all NUMA nodes; page-granular, non-fatal. */
static inline void glm5_numa_interleave(void *p, size_t n){
    static int off=-1; if(off<0) off = getenv("GLM5_NO_NUMA") ? 1 : 0; if(off) return;
    unsigned long mask=~0UL;   /* all nodes; kernel intersects with this task's allowed set */
    (void)syscall(SYS_mbind, p, n, 3 /*MPOL_INTERLEAVE*/, &mask, (unsigned long)(sizeof(mask)*8), 0u);
}
#else
static inline void glm5_numa_interleave(void *p, size_t n){ (void)p; (void)n; }
#endif

/* 256-aligned allocation; size rounded up to the alignment (aligned_alloc requirement). */
static inline void *glm5_amalloc(size_t n){
    if(n==0) n=1;
    size_t a=(n + (size_t)GLM5_ALIGN - 1) & ~(size_t)(GLM5_ALIGN - 1);
    void *p=aligned_alloc(GLM5_ALIGN, a);
    if(p && a>=GLM5_NUMA_THRESH) glm5_numa_interleave(p, a);
    return p;
}
static inline void *glm5_acalloc(size_t cnt, size_t sz){
    size_t n=cnt*sz; void *p=glm5_amalloc(n); if(p) memset(p, 0, n); return p;
}
static inline void glm5_afree(void *p){ free(p); }

#endif /* GLM5_MEM_H */
