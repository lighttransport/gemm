/* Prefetch plan: weights that a LATER decode stage will read, prefetched into L2 by the compute threads while an earlier
 * latency-bound stage (the mHC regions) runs.  Decode keeps HBM almost idle, so this hides the cold-miss latency of the
 * next matvec.  The plan is a per-thread table of byte spans, built with the SAME static row partition the consuming
 * matvec uses, so every line lands in the L2 of the CMG that will read it.  Spans are consumed by glm53f_pf_run(tid)
 * (called at the start of a parallel region); the plan persists until the next glm53f_pf_clear(). */
#ifndef GLM53F_PF_PLAN_H
#define GLM53F_PF_PLAN_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include "glm53f_iq_bridge.h"

enum { GLM53F_PF_MAX_THREADS = 128, GLM53F_PF_MAX_ITEMS = 12 };
typedef struct { const char *p; size_t len; } glm53f_pf_span;
typedef struct { int n; glm53f_pf_span s[GLM53F_PF_MAX_THREADS][GLM53F_PF_MAX_ITEMS]; } glm53f_pf_table;
extern glm53f_pf_table glm53f_pf_tab;
__attribute__((weak)) glm53f_pf_table glm53f_pf_tab;

/* Set while glm53f_cmg_place.h collects row partitions for load-time page
 * placement: plans are built without enabling decode prefetch. */
extern int glm53f_pf_collecting;
__attribute__((weak)) int glm53f_pf_collecting;
static inline int glm53f_pf_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("GLM53F_PF_PLAN"); v = e && *e && atoi(e); } /* default off: net-neutral, see below */
    return v || glm53f_pf_collecting;
}
static inline size_t glm53f_pf_cap(void) {
    static long v = -1;
    if (v < 0) { const char *e = getenv("GLM53F_PF_CAP"); v = e && *e ? atol(e) : 16384; }
    return (size_t)v;
}
static inline void glm53f_pf_clear(void) { glm53f_pf_tab.n = 0; }
/* OpenMP schedule(static) without chunk: contiguous blocks, the first n%nt threads get one extra iteration */
static inline void glm53f_pf_static_range(long n, int nt, int t, long *lo, long *hi) {
    const long chunk = n / nt, rem = n % nt;
    *lo = t < rem ? t * (chunk + 1) : rem * (chunk + 1) + (t - rem) * chunk;
    *hi = *lo + (t < rem ? chunk + 1 : chunk);
}
/* n tasks of task_bytes each (e.g. 8-row blocks of a bf16 matrix) split like `omp for schedule(static)` */
static inline void glm53f_pf_add_tasks(const void *base, size_t task_bytes, long ntasks) {
    if (!base || glm53f_pf_tab.n >= GLM53F_PF_MAX_ITEMS || !glm53f_pf_enabled()) return;
    int nt = omp_get_max_threads();
    if (nt > GLM53F_PF_MAX_THREADS) nt = GLM53F_PF_MAX_THREADS;
    const int i = glm53f_pf_tab.n++;
    for (int t = 0; t < nt; ++t) {
        long lo, hi;
        glm53f_pf_static_range(ntasks, nt, t, &lo, &hi);
        glm53f_pf_tab.s[t][i].p = (const char *)base + (size_t)lo * task_bytes;
        glm53f_pf_tab.s[t][i].len = (size_t)(hi - lo) * task_bytes;
    }
}
/* the matrices of one glm53f_native_matvec_team call (same work items / groups / partition) */
static inline void glm53f_pf_add_matvec(const glm53f_native_matrix *m, int count) {
    if (glm53f_pf_tab.n + count > GLM53F_PF_MAX_ITEMS || !glm53f_pf_enabled()) return;
    int nt = omp_get_max_threads();
    if (nt > GLM53F_PF_MAX_THREADS) nt = GLM53F_PF_MAX_THREADS;
    long start[9], total = 0;
    size_t rb[8];
    int group[8];
    for (int i = 0; i < count; ++i) {
        rb[i] = glm53f_native_row_size(m[i].type, m[i].columns);
        group[i] = m[i].type == GLM53F_NATIVE_Q8_0R16 ? 16 : m[i].type == GLM53F_NATIVE_Q8_0R ? 4 : 1;
        start[i] = total;
        total += (m[i].rows + group[i] - 1) / group[i];
    }
    start[count] = total;
    const int base = glm53f_pf_tab.n;
    glm53f_pf_tab.n += count;
    for (int t = 0; t < nt; ++t) {
        long q0, q1;
        glm53f_pf_static_range(total, nt, t, &q0, &q1);
        for (int i = 0; i < count; ++i) {
            const long lo = q0 > start[i] ? q0 : start[i], hi = q1 < start[i + 1] ? q1 : start[i + 1];
            glm53f_pf_span *sp = &glm53f_pf_tab.s[t][base + i];
            if (hi > lo && m[i].weight) {
                sp->p = (const char *)m[i].weight + (size_t)(lo - start[i]) * group[i] * rb[i];
                sp->len = (size_t)(hi - lo) * group[i] * rb[i];
            } else { sp->p = NULL; sp->len = 0; }
        }
    }
}
static inline void glm53f_pf_run(int tid) {
    if (tid >= GLM53F_PF_MAX_THREADS) return;
    for (int i = 0; i < glm53f_pf_tab.n; ++i) {
        const glm53f_pf_span sp = glm53f_pf_tab.s[tid][i];
        /* Only the head of each span: issuing a whole slice (hundreds of KB) stalls the issuing core behind a full
         * prefetch queue; the consuming kernel's own look-ahead prefetch covers the rest. */
        const size_t cap = glm53f_pf_cap(), len = sp.len < cap ? sp.len : cap;
        for (size_t o = 0; o < len; o += 256) __builtin_prefetch(sp.p + o, 0, 2);
    }
}
#endif
