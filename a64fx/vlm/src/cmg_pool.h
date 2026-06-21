/* cmg_pool.h — A64FX CMG (Core Memory Group) NUMA infrastructure.
 *
 * A64FX has 4 CMGs, each with directly-attached HBM2 (~7 GB) and 12 compute
 * cores. NUMA topology:
 *   CMG0 = node 4, cores 12-23
 *   CMG1 = node 5, cores 24-35
 *   CMG2 = node 6, cores 36-47
 *   CMG3 = node 7, cores 48-59
 *
 * Cross-CMG bandwidth is ~80-100 GB/s (ring); local HBM is ~200 GB/s per CMG.
 * 4×CMG aggregate peak is ~756 GB/s (91% of 833 GB/s streaming peak).
 *
 * This module provides:
 *   - cmg_alloc / cmg_free       : mmap + mbind allocator pinned to one CMG
 *   - cmg_pin_thread             : pthread CPU affinity to a CMG-local core
 *   - cmg_barrier                : 4-way cross-CMG barrier (CMG leaders)
 *   - cmg_topo                   : query topology (n_cmgs, cores_per_cmg, etc.)
 */
#ifndef CMG_POOL_H
#define CMG_POOL_H

#include <stddef.h>
#include <stdint.h>

#define CMG_MAX            4
#define CMG_NODE_BASE      4        /* NUMA node 4 = CMG0 */
#define CMG_CORE_BASE      12       /* first compute core (cores 0-11 are OS) */
#define CMG_CORES_PER_CMG  12

#ifdef __cplusplus
extern "C" {
#endif

/* Topology / runtime query. Always returns CMG_MAX for now; future could probe. */
int cmg_count(void);
int cmg_cores_per_cmg(void);

/* mmap+mbind allocator: returns a region of `size` bytes pinned to CMG `cmg`.
 * `size` is rounded up to page granularity (64 KB on A64FX).
 * Pages are not faulted; first-touch from CMG-local threads will fault-in locally
 * (mbind guarantees they land on the bound node regardless of first-touch).
 * Returns NULL on failure. Free with cmg_free(p, size). */
void *cmg_alloc(int cmg, size_t size);
void  cmg_free(void *p, size_t size);

/* Pin the calling thread to core (CMG_CORE_BASE + cmg*CMG_CORES_PER_CMG + local_tid).
 * local_tid is [0, CMG_CORES_PER_CMG). Returns 0 on success, -1 on error. */
int cmg_pin_thread(int cmg, int local_tid);

/* Query which CMG the calling thread is bound to (based on its current affinity).
 * Returns 0..3 if pinned to a single CMG-local core, -1 otherwise. */
int cmg_self(void);

/* Lightweight 4-way barrier across CMG leaders. Initialize once, then call
 * cmg_barrier_wait from each CMG leader. */
typedef struct {
    volatile int sense;
    volatile int count;
    int          n_cmgs;
} cmg_barrier_t;

void cmg_barrier_init(cmg_barrier_t *b, int n_cmgs);
void cmg_barrier_wait(cmg_barrier_t *b);

#ifdef __cplusplus
}
#endif

#endif /* CMG_POOL_H */
