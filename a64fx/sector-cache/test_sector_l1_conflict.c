/*
 * A64FX Sector Cache L1 Way-Conflict Test (Pointer-Chase)
 *
 * Uses pointer chasing to directly expose L1 vs L2 latency difference
 * when sector cache prevents (or doesn't prevent) eviction.
 *
 * A64FX L1D: 64KB, 4-way set-associative, 256B line, 64 sets
 * One "way" = 64 sets × 256B = 16KB
 *
 * Memory layout (contiguous, 2MB-aligned):
 *   [0, 32KB)    = "keep" array → 128 cache lines, 2 ways per set
 *   [32KB, 128KB) = "evict" array → 384 cache lines, 6 ways per set
 *
 * Both arrays map to the same 64 cache sets (aliasing by design).
 *
 * Test pattern per iteration:
 *   1. Pointer-chase through "keep" (128 dependent loads, primes L1)
 *   2. Stream "evict" array (96KB, causes eviction pressure)
 *   3. Pointer-chase through "keep" again — measures L1 hit vs miss
 *
 * Without sector cache:
 *   Step 2 evicts keep data (6 ways needed, only 4 total)
 *   Step 3: ~128 × 40 cycles (L2 latency) = ~5120 cycles
 *
 * With sector cache (sec0=2, sec1=2):
 *   Evict goes to sector 1 (2 ways), keep stays in sector 0 (2 ways)
 *   Step 3: ~128 × 4 cycles (L1 latency) = ~512 cycles → ~10x faster
 *
 * Build:
 *   fcc -Nnoclang -O2 -Kocl,hpctag -o test_sector_l1_conflict test_sector_l1_conflict.c
 *
 * Build (fapp):
 *   fcc -Nnoclang -O2 -Kocl,hpctag -DUSE_FAPP -o test_sector_l1_conflict_fapp test_sector_l1_conflict.c
 *
 * Run:
 *   FLIB_SCCR_CNTL=TRUE FLIB_L1_SCCR_CNTL=TRUE ./test_sector_l1_conflict
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef USE_FAPP
#include <fj_tool/fapp.h>
#endif

#define L1_LINE    256
#define KEEP_SIZE  (32 * 1024)   /* 32KB = 2 ways */
#define EVICT_SIZE (96 * 1024)   /* 96KB = 6 ways (overflows 4-way L1) */
#define KEEP_LINES (KEEP_SIZE / L1_LINE)   /* 128 lines */
#define EVICT_ELEMS (EVICT_SIZE / sizeof(uint64_t))  /* 12288 elements */

#define NUM_ITERS  500
#define CHASE_REPS 4  /* repeat pointer chase per iteration for signal */

static inline uint64_t rdtsc(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static inline uint64_t rdfreq(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}
static inline void flush_line(void *p) {
    __asm__ volatile("dc civac, %0" :: "r"(p) : "memory");
}
static inline void full_barrier(void) {
    __asm__ volatile("dsb ish" ::: "memory");
}

/*
 * Build a random pointer-chase chain through the keep array.
 * Each cache line's first 8 bytes hold the BYTE OFFSET to the next line.
 * Chain visits all KEEP_LINES lines exactly once (Hamiltonian cycle).
 */
static void build_chase_chain(void *keep_base) {
    int order[KEEP_LINES];
    int i, j, tmp;
    uint64_t *p;

    /* Initialize sequential order */
    for (i = 0; i < KEEP_LINES; i++) order[i] = i;

    /* Fisher-Yates shuffle (simple deterministic seed) */
    srand(42);
    for (i = KEEP_LINES - 1; i > 0; i--) {
        j = rand() % (i + 1);
        tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    /* Write chain: line[order[i]] → line[order[i+1]] */
    for (i = 0; i < KEEP_LINES - 1; i++) {
        p = (uint64_t *)((char *)keep_base + (size_t)order[i] * L1_LINE);
        *p = (uint64_t)((size_t)order[i + 1] * L1_LINE);
    }
    /* Last → first (cycle) */
    p = (uint64_t *)((char *)keep_base + (size_t)order[KEEP_LINES - 1] * L1_LINE);
    *p = (uint64_t)((size_t)order[0] * L1_LINE);
}

/*
 * Pointer-chase through keep array. Returns accumulated checksum.
 * Each load depends on previous → fully serialized → exposes true latency.
 */
static inline uint64_t chase(void *keep_base, int lines) {
    uint64_t offset = 0;
    int i;
    for (i = 0; i < lines; i++) {
        offset = *(volatile uint64_t *)((char *)keep_base + offset);
    }
    return offset;
}

static void flush_range(void *base, size_t size) {
    size_t off;
    for (off = 0; off < size; off += L1_LINE)
        flush_line((char *)base + off);
    full_barrier();
}

/*
 * Baseline: no sector cache. Evict stream has no tag → same sector as keep.
 */
static void run_nohint(
    void *keep, uint64_t *evict,
    uint64_t *out_reload_ticks, uint64_t *out_total_ticks)
{
    volatile uint64_t sink = 0;
    uint64_t t0, t1, t2, t3;
    uint64_t chase_sum = 0;
    uint64_t reload_acc = 0;
    int iter, r, j;

    flush_range(keep, KEEP_SIZE);
    flush_range(evict, EVICT_SIZE);

#ifdef USE_FAPP
    fapp_start("nohint_all", 1, 0);
#endif
    t0 = rdtsc();

    for (iter = 0; iter < NUM_ITERS; iter++) {
        /* Step 1: Prime keep into L1 via pointer chase */
        for (r = 0; r < CHASE_REPS; r++)
            chase_sum += chase(keep, KEEP_LINES);

        /* Step 2: Stream evict (eviction pressure) */
        {
            uint64_t s = 0;
            for (j = 0; j < (int)EVICT_ELEMS; j++)
                s += evict[j];
            sink = s;
        }

        /* Step 3: Reload keep via pointer chase — measure this */
        t2 = rdtsc();
        for (r = 0; r < CHASE_REPS; r++)
            chase_sum += chase(keep, KEEP_LINES);
        t3 = rdtsc();
        reload_acc += t3 - t2;
    }

    t1 = rdtsc();
#ifdef USE_FAPP
    fapp_stop("nohint_all", 1, 0);
#endif
    sink += chase_sum;
    (void)sink;

    *out_reload_ticks = reload_acc;
    *out_total_ticks = t1 - t0;
}

/*
 * With sector cache: evict → tagged (sector 1), keep → untagged (sector 0).
 */
static void run_sector(
    void *keep, uint64_t *evict,
    uint64_t *out_reload_ticks, uint64_t *out_total_ticks)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign evict
    volatile uint64_t sink = 0;
    uint64_t t0, t1, t2, t3;
    uint64_t chase_sum = 0;
    uint64_t reload_acc = 0;
    int iter, r, j;

    flush_range(keep, KEEP_SIZE);
    flush_range(evict, EVICT_SIZE);

#ifdef USE_FAPP
    fapp_start("sector_all", 1, 0);
#endif
    t0 = rdtsc();

    for (iter = 0; iter < NUM_ITERS; iter++) {
        /* Step 1: Prime keep into L1 sector 0 via pointer chase */
        for (r = 0; r < CHASE_REPS; r++)
            chase_sum += chase(keep, KEEP_LINES);

        /* Step 2: Stream evict (tagged → sector 1, doesn't evict keep) */
        {
            uint64_t s = 0;
            for (j = 0; j < (int)EVICT_ELEMS; j++)
                s += evict[j];
            sink = s;
        }

        /* Step 3: Reload keep via pointer chase — should be L1 hit */
        t2 = rdtsc();
        for (r = 0; r < CHASE_REPS; r++)
            chase_sum += chase(keep, KEEP_LINES);
        t3 = rdtsc();
        reload_acc += t3 - t2;
    }

    t1 = rdtsc();
#ifdef USE_FAPP
    fapp_stop("sector_all", 1, 0);
#endif
    sink += chase_sum;
    (void)sink;

    *out_reload_ticks = reload_acc;
    *out_total_ticks = t1 - t0;
}

int main() {
    void *block;
    void *keep_base;
    uint64_t *evict_base;
    uint64_t freq;
    uint64_t reload_nohint, total_nohint;
    uint64_t reload_sector, total_sector;
    const char *sccr_env, *l1_env;

    freq = rdfreq();

    printf("=== Sector Cache L1 Way-Conflict Test (Pointer Chase) ===\n\n");
    printf("L1D           : 64KB, 4-way, 256B line, 64 sets\n");
    printf("Keep array    : %d KB (%d cache lines, %d ways worth)\n",
           KEEP_SIZE/1024, KEEP_LINES, KEEP_SIZE / (16*1024));
    printf("Evict array   : %d KB (%d ways worth)\n",
           EVICT_SIZE/1024, EVICT_SIZE / (16*1024));
    printf("Chase per step: %d reps × %d lines = %d dependent loads\n",
           CHASE_REPS, KEEP_LINES, CHASE_REPS * KEEP_LINES);
    printf("Iterations    : %d\n", NUM_ITERS);
    printf("Timer freq    : %lu MHz\n\n", freq / 1000000);

    sccr_env = getenv("FLIB_SCCR_CNTL");
    l1_env   = getenv("FLIB_L1_SCCR_CNTL");
    printf("FLIB_SCCR_CNTL    = %s\n", sccr_env ? sccr_env : "(not set)");
    printf("FLIB_L1_SCCR_CNTL = %s\n\n", l1_env ? l1_env : "(not set)");

    /* Allocate 2MB-aligned contiguous block: [keep | evict] */
    posix_memalign(&block, 2 * 1024 * 1024, KEEP_SIZE + EVICT_SIZE);
    if (!block) { fprintf(stderr, "alloc failed\n"); return 1; }
    memset(block, 0, KEEP_SIZE + EVICT_SIZE);

    keep_base = block;
    evict_base = (uint64_t *)((char *)block + KEEP_SIZE);

    printf("Addresses:\n");
    printf("  keep  = %p (sets 0-63, 2 ways)\n", keep_base);
    printf("  evict = %p (sets 0-63, 6 ways)\n", (void *)evict_base);
    printf("  Aliasing: keep set0 = 0x%02lx, evict set0 = 0x%02lx\n",
           ((uintptr_t)keep_base >> 8) & 0x3f,
           ((uintptr_t)evict_base >> 8) & 0x3f);
    printf("\n");

    /* Build pointer-chase chain in keep array */
    build_chase_chain(keep_base);

    /* Initialize evict array */
    {
        int j;
        for (j = 0; j < (int)EVICT_ELEMS; j++)
            evict_base[j] = (uint64_t)j;
    }

    /* Warmup */
    run_nohint(keep_base, evict_base, &reload_nohint, &total_nohint);
    run_sector(keep_base, evict_base, &reload_sector, &total_sector);

    /* Actual tests */
    run_nohint(keep_base, evict_base, &reload_nohint, &total_nohint);
    run_sector(keep_base, evict_base, &reload_sector, &total_sector);

    {
        double total_speedup = (double)total_nohint / total_sector;
        double reload_speedup = (double)reload_nohint / reload_sector;
        int chase_loads = CHASE_REPS * KEEP_LINES;
        int total_chase = chase_loads * NUM_ITERS;
        /* cntvct runs at 100MHz, CPU at 2000MHz → 1 tick = 20 cycles */
        double cyc_per_load_nohint = (double)reload_nohint * 20.0 / total_chase;
        double cyc_per_load_sector = (double)reload_sector * 20.0 / total_chase;

        printf("=== Results ===\n\n");

        printf("Total time (all 3 steps):\n");
        printf("  nohint : %10lu ticks  %6.2f ms\n",
               total_nohint, (double)total_nohint / freq * 1e3);
        printf("  sector : %10lu ticks  %6.2f ms  speedup=%.2fx\n",
               total_sector, (double)total_sector / freq * 1e3, total_speedup);

        printf("\nStep 3 only (reload chase after eviction pressure):\n");
        printf("  nohint : %10lu ticks  %6.2f ms  %.1f cyc/load\n",
               reload_nohint, (double)reload_nohint / freq * 1e3,
               cyc_per_load_nohint);
        printf("  sector : %10lu ticks  %6.2f ms  %.1f cyc/load  speedup=%.2fx\n",
               reload_sector, (double)reload_sector / freq * 1e3,
               cyc_per_load_sector, reload_speedup);

        printf("\nLatency interpretation:\n");
        printf("  %.1f cyc/load → %s\n", cyc_per_load_nohint,
               cyc_per_load_nohint > 15 ? "L2 (evicted from L1)" : "L1 (still cached)");
        printf("  %.1f cyc/load → %s\n", cyc_per_load_sector,
               cyc_per_load_sector > 15 ? "L2 (evicted from L1)" : "L1 (sector cache kept it)");

        printf("\nVerdict: ");
        if (reload_speedup > 2.0) {
            printf("STRONG PASS — step3 reload %.2fx faster with sector cache\n",
                   reload_speedup);
            printf("  → L1 partitioning CONFIRMED: keep data stays in L1 sector 0\n");
        } else if (total_speedup > 1.10) {
            printf("PASS — sector cache shows %.2fx total speedup\n", total_speedup);
            printf("  → L1 partitioning is ACTIVE and effective\n");
        } else if (total_speedup > 1.02) {
            printf("MARGINAL — %.2fx (small but measurable effect)\n", total_speedup);
        } else {
            printf("NO EFFECT — %.2fx\n", total_speedup);
        }
    }

    free(block);
    return 0;
}
