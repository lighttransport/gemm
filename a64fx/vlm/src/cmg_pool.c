/* cmg_pool.c — A64FX CMG NUMA infrastructure.
 *
 * Uses raw mbind() syscall (no libnuma dependency, matches Fugaku-tested
 * pattern from tools/clair/a64fx/a64fx-hugepages-malloc.md).
 *
 * mbind reference: man 2 mbind. MPOL_BIND with single-node mask guarantees
 * pages allocated for the region resolve from the bound HBM2 node.
 */
#define _GNU_SOURCE
#include "cmg_pool.h"

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif

/* mbind syscall wrapper; matches glibc signature. */
static long mbind_(void *addr, unsigned long len, int mode,
                   const unsigned long *nodemask, unsigned long maxnode,
                   unsigned flags)
{
    return syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
}

int cmg_count(void)         { return CMG_MAX; }
int cmg_cores_per_cmg(void) { return CMG_CORES_PER_CMG; }

void *cmg_alloc(int cmg, size_t size) {
    if (cmg < 0 || cmg >= CMG_MAX) return NULL;
    if (size == 0) return NULL;

    /* Round up to A64FX 64KB page granularity. */
    long pgsz = sysconf(_SC_PAGESIZE);
    if (pgsz <= 0) pgsz = 65536;
    size_t aligned = (size + (size_t)pgsz - 1) & ~(size_t)(pgsz - 1);

    void *p = mmap(NULL, aligned, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "cmg_alloc: mmap %zu failed: %s\n", aligned, strerror(errno));
        return NULL;
    }

    /* nodemask bit (CMG_NODE_BASE + cmg). maxnode in glibc mbind = highest
     * node index + 2 (mask is bits 0..maxnode-1). For node 7 we pass 8. */
    int node = CMG_NODE_BASE + cmg;
    unsigned long mask = 1UL << node;
    if (mbind_(p, aligned, MPOL_BIND, &mask, 64, 0) != 0) {
        /* Non-fatal: page placement may fall back to first-touch.
         * On Fugaku this normally succeeds without root. */
        fprintf(stderr, "cmg_alloc: mbind cmg=%d node=%d failed: %s (continuing)\n",
                cmg, node, strerror(errno));
    }
    return p;
}

void cmg_free(void *p, size_t size) {
    if (!p) return;
    long pgsz = sysconf(_SC_PAGESIZE);
    if (pgsz <= 0) pgsz = 65536;
    size_t aligned = (size + (size_t)pgsz - 1) & ~(size_t)(pgsz - 1);
    munmap(p, aligned);
}

int cmg_pin_thread(int cmg, int local_tid) {
    if (cmg < 0 || cmg >= CMG_MAX) return -1;
    if (local_tid < 0 || local_tid >= CMG_CORES_PER_CMG) return -1;
    int core = CMG_CORE_BASE + cmg * CMG_CORES_PER_CMG + local_tid;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        fprintf(stderr, "cmg_pin_thread: setaffinity core=%d failed: %s\n",
                core, strerror(errno));
        return -1;
    }
    return 0;
}

int cmg_self(void) {
    cpu_set_t set;
    if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) != 0) return -1;
    int found = -1;
    for (int cmg = 0; cmg < CMG_MAX; cmg++) {
        int lo = CMG_CORE_BASE + cmg * CMG_CORES_PER_CMG;
        int hi = lo + CMG_CORES_PER_CMG;
        int all_in = 1, any_in = 0;
        for (int c = 0; c < CPU_SETSIZE; c++) {
            if (CPU_ISSET(c, &set)) {
                if (c >= lo && c < hi) any_in = 1;
                else                    all_in = 0;
            }
        }
        if (any_in && all_in) {
            if (found < 0) found = cmg;
            else            return -1;  /* affinity spans multiple CMGs */
        }
    }
    return found;
}

/* ── Cross-CMG barrier (sense-reversal) ─────────────────────────────── */
void cmg_barrier_init(cmg_barrier_t *b, int n_cmgs) {
    if (!b) return;
    b->sense  = 0;
    b->count  = 0;
    b->n_cmgs = (n_cmgs > 0 && n_cmgs <= CMG_MAX) ? n_cmgs : CMG_MAX;
}

void cmg_barrier_wait(cmg_barrier_t *b) {
    if (!b) return;
    int local_sense = !b->sense;
    int c = __atomic_add_fetch(&b->count, 1, __ATOMIC_ACQ_REL);
    if (c == b->n_cmgs) {
        __atomic_store_n(&b->count, 0, __ATOMIC_RELEASE);
        __atomic_store_n(&b->sense, local_sense, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&b->sense, __ATOMIC_ACQUIRE) != local_sense) {
            __asm__ __volatile__ ("yield" ::: "memory");
        }
    }
}
