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

static inline int glm53f_cmg_flush(void **pages, int *nodes, int *status, int n,
                                   glm53f_cmg_place_stats *st) {
    if (!n) return 0;
    long rc = syscall(SYS_move_pages, 0, (unsigned long)n, pages, nodes, status, 1L /* MPOL_MF_MOVE */);
    if (rc < 0) { st->failed += n; return -1; }
    for (int i = 0; i < n; ++i) {
        if (status[i] == nodes[i]) ++st->moved;
        else if (status[i] >= 0) ++st->already;
        else ++st->failed;
    }
    return 0;
}

/* Place the spans currently in glm53f_pf_tab (one plan = one decode stage). */
static inline void glm53f_cmg_place_table(const int *node, int nt, glm53f_cmg_place_stats *st) {
    static void *pages[GLM53F_CMG_PLACE_BATCH];
    static int nodes[GLM53F_CMG_PLACE_BATCH], status[GLM53F_CMG_PLACE_BATCH];
    const uintptr_t ps = (uintptr_t)sysconf(_SC_PAGESIZE);
    int n = 0;
    if (nt > GLM53F_PF_MAX_THREADS) nt = GLM53F_PF_MAX_THREADS;
    for (int i = 0; i < glm53f_pf_tab.n; ++i)
        for (int t = 0; t < nt; ++t) {
            const glm53f_pf_span *sp = &glm53f_pf_tab.s[t][i];
            if (!sp->p || !sp->len) continue;
            const uintptr_t lo = (uintptr_t)sp->p, hi = lo + sp->len;
            for (uintptr_t pg = lo & ~(ps - 1); pg < hi; pg += ps) {
                const uintptr_t mid = pg + ps / 2;
                /* Midpoint ownership; spans smaller than a page keep their first page. */
                if (!(mid >= lo && mid < hi) && !(pg <= lo && lo < pg + ps && sp->len < ps)) continue;
                pages[n] = (void *)pg; nodes[n] = node[t]; ++st->requested;
                if (++n == GLM53F_CMG_PLACE_BATCH) { glm53f_cmg_flush(pages, nodes, status, n, st); n = 0; }
            }
        }
    glm53f_cmg_flush(pages, nodes, status, n, st);
}
#endif
