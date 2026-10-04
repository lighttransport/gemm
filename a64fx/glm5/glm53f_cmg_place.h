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
#include <sys/syscall.h>
#include <unistd.h>

typedef struct { long requested, moved, failed, already; } glm53f_cmg_place_stats;
enum { GLM53F_CMG_PLACE_BATCH = 4096 };

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

static inline void glm53f_cmg_batch_flush(glm53f_cmg_batch *b) {
    if (!b->n) return;
    long rc = syscall(SYS_move_pages, 0, (unsigned long)b->n, b->pages, b->nodes, b->status, 1L /* MPOL_MF_MOVE */);
    if (rc < 0) b->st->failed += b->n;
    else
        for (int i = 0; i < b->n; ++i) {
            if (b->status[i] == b->nodes[i]) ++b->st->moved;
            else if (b->status[i] >= 0) ++b->st->already;
            else ++b->st->failed;
        }
    b->n = 0;
}
/* Pages whose midpoint lies in [lo, hi) go to node; a range smaller than a
 * page keeps the page holding its start. */
static inline void glm53f_cmg_add_range(glm53f_cmg_batch *b, uintptr_t lo, uintptr_t hi, int node) {
    const uintptr_t ps = (uintptr_t)sysconf(_SC_PAGESIZE);
    if (hi <= lo) return;
    for (uintptr_t pg = lo & ~(ps - 1); pg < hi; pg += ps) {
        const uintptr_t mid = pg + ps / 2;
        if (!(mid >= lo && mid < hi) && !(pg <= lo && lo < pg + ps && hi - lo < ps)) continue;
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
