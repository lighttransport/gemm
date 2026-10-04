/* Load-time CMG-local weight placement for decode.
 *
 * A64FX decode matvecs partition rows with the static OpenMP schedule, so a
 * thread reads the same weight bytes every token. glm53f_pf_plan.h already
 * computes those per-thread byte spans; with glm53f_pf_collecting set, the
 * layer plan builders register them without enabling prefetch. This file
 * migrates each span's pages to the NUMA node of the CMG running that thread
 * (move_pages), so streaming reads stay CMG-local. Arithmetic is unchanged.
 * Pages shared by two threads go to the owner of their midpoint. */
#ifndef GLM53F_CMG_PLACE_H
#define GLM53F_CMG_PLACE_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "glm53f_pf_plan.h"
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef struct { long requested, moved, failed, already; int move_errno; } glm53f_cmg_place_stats;
enum { GLM53F_CMG_PLACE_BATCH = 1024 };

/* NUMA node of each OpenMP thread (same numbering as the decode team). */
static inline int glm53f_cmg_thread_nodes(int *node, int max_threads) {
    int nt = 0, bad = 0;
#pragma omp parallel reduction(|:bad)
    {
        unsigned cpu = 0, nd = 0;
        const int t = omp_get_thread_num();
#pragma omp single
        nt = omp_get_num_threads();
        if (syscall(SYS_getcpu, &cpu, &nd, NULL) != 0 || t >= max_threads) bad = 1;
        else node[t] = (int)nd;
    }
    return bad || nt > max_threads ? -1 : nt;
}

typedef struct glm53f_cmg_batch {
    void *pages[GLM53F_CMG_PLACE_BATCH];
    int nodes[GLM53F_CMG_PLACE_BATCH], status[GLM53F_CMG_PLACE_BATCH], n;
    glm53f_cmg_place_stats *st;
} glm53f_cmg_batch;

/* move_pages is refused on the A64FX compute nodes, so pages are re-homed by
 * first touch: copy the batch out to a staging buffer, drop the physical
 * pages (anonymous private memory; MADV_DONTNEED), then let threads of each
 * target CMG copy their pages back under a thread-local MPOL_LOCAL policy.
 * Each thread's previous policy is restored afterwards. Contents are
 * unchanged; placement is verified per page with get_mempolicy. */
static inline void glm53f_cmg_batch_flush(glm53f_cmg_batch *b) {
    static unsigned char *stage;
    const size_t ps = (size_t)sysconf(_SC_PAGESIZE);
    if (!b->n) return;
    if (!b->st->move_errno) {   /* record why migration is unavailable, once */
        int probe_status = 0, probe_node = b->nodes[0];
        void *probe_page = b->pages[0];
        long rc = syscall(SYS_move_pages, 0, 1UL, &probe_page, &probe_node, &probe_status, 2L /* MPOL_MF_MOVE */);
        b->st->move_errno = rc < 0 ? errno : -1 - (probe_status < 0 ? -probe_status : 0);
    }
    if (!stage && !(stage = aligned_alloc(ps, ps * GLM53F_CMG_PLACE_BATCH))) { b->st->failed += b->n; b->n = 0; return; }
    for (int i = 0; i < b->n; ++i) memcpy(stage + (size_t)i * ps, b->pages[i], ps);
    for (int i = 0; i < b->n; ++i)
        if (madvise(b->pages[i], ps, MADV_DONTNEED)) b->status[i] = -1; else b->status[i] = 0;
    const int n = b->n;
#pragma omp parallel
    {
        unsigned cpu = 0, mynode = 0;
        int mode = 0;
        unsigned long mask[4] = {0, 0, 0, 0};
        syscall(SYS_getcpu, &cpu, &mynode, NULL);
        const long have = syscall(SYS_get_mempolicy, &mode, mask, 256UL, NULL, 0UL);
        syscall(SYS_set_mempolicy, 4L /* MPOL_LOCAL */, NULL, 0UL);
        /* threads on the same node take every k-th page of that node */
        int rank_on_node = 0, on_node = 0;
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        static int thread_node[256];
        thread_node[tid] = (int)mynode;
#pragma omp barrier
        for (int t = 0; t < nt; ++t) if (thread_node[t] == (int)mynode) { if (t < tid) ++rank_on_node; ++on_node; }
        int seen = 0;
        for (int i = 0; i < n; ++i)
            if (b->nodes[i] == (int)mynode && (seen++ % on_node) == rank_on_node)
                memcpy(b->pages[i], stage + (size_t)i * ps, ps);
        if (have == 0) syscall(SYS_set_mempolicy, (long)mode, mask, 256UL);
#pragma omp barrier
    }
    for (int i = 0; i < n; ++i) {   /* pages whose target node has no thread are restored by the caller thread */
        int node = -1;
        if (syscall(SYS_get_mempolicy, &node, NULL, 0UL, b->pages[i], 3UL /* MPOL_F_NODE|MPOL_F_ADDR */) != 0) node = -1;
        if (node < 0) { memcpy(b->pages[i], stage + (size_t)i * ps, ps); ++b->st->failed; }
        else if (node == b->nodes[i]) ++b->st->moved;
        else ++b->st->already;
    }
    b->n = 0;
}
/* Only pages lying entirely inside [lo, hi) go to node: re-homing drops a
 * page's contents temporarily, so pages shared with other heap objects (the
 * edges of a weight buffer) must never be touched. */
static inline void glm53f_cmg_add_range(glm53f_cmg_batch *b, uintptr_t lo, uintptr_t hi, int node) {
    const uintptr_t ps = (uintptr_t)sysconf(_SC_PAGESIZE);
    if (hi <= lo) return;
    for (uintptr_t pg = (lo + ps - 1) & ~(ps - 1); pg + ps <= hi; pg += ps) {
        b->pages[b->n] = (void *)pg; b->nodes[b->n] = node; ++b->st->requested;
        if (++b->n == GLM53F_CMG_PLACE_BATCH) glm53f_cmg_batch_flush(b);
    }
}
/* Row split of iq_cmg_bounds (glm53f_iq_bridge.c): CMG c owns rows [bound[c], bound[c+1]). */
static inline void glm53f_cmg_row_bounds(int rows, int nt, int *bound) {
    int cum = 0;
    bound[0] = 0;
    for (int c = 0; c < 4; ++c) {
        int n = nt - c * 12; n = n < 0 ? 0 : (n > 12 ? 12 : n);
        cum += n;
        bound[c + 1] = c == 3 ? rows : (int)((long long)rows * cum / nt) & ~1;
    }
}
/* Rows of one matrix split by CMG; cmg_node[c] is the NUMA node of CMG c. */
static inline void glm53f_cmg_place_rows(glm53f_cmg_batch *b, const uint8_t *base, size_t row_bytes,
                                         int rows, int nt, const int *cmg_node) {
    int bound[5];
    if (!base || !row_bytes || rows < 1) return;
    glm53f_cmg_row_bounds(rows, nt, bound);
    for (int c = 0; c < 4; ++c)
        glm53f_cmg_add_range(b, (uintptr_t)(base + (size_t)bound[c] * row_bytes),
                             (uintptr_t)(base + (size_t)bound[c + 1] * row_bytes), cmg_node[c]);
}

/* Place the spans currently in glm53f_pf_tab (one plan = one decode stage). */
static inline void glm53f_cmg_place_table(glm53f_cmg_batch *b, const int *node, int nt) {
    if (nt > GLM53F_PF_MAX_THREADS) nt = GLM53F_PF_MAX_THREADS;
    for (int i = 0; i < glm53f_pf_tab.n; ++i)
        for (int t = 0; t < nt; ++t) {
            const glm53f_pf_span *sp = &glm53f_pf_tab.s[t][i];
            if (sp->p && sp->len)
                glm53f_cmg_add_range(b, (uintptr_t)sp->p, (uintptr_t)sp->p + sp->len, node[t]);
        }
}
#endif
