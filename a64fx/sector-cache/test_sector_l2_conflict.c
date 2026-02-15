/*
 * A64FX Sector Cache L2 Way-Conflict Test (Pointer-Chase)
 *
 * Same approach as test_sector_l1_conflict.c but targets L2 cache.
 * Uses pointer chasing to directly expose L2 vs Memory latency difference
 * when L2 sector cache prevents (or doesn't prevent) eviction.
 *
 * A64FX L2: 8MB, 16-way set-associative, 256B line
 * One "way" = 8MB / 16 = 512KB
 * Available to app: 14 ways = 7MB (2 ways reserved by OS/runtime)
 *
 * Memory layout (contiguous, 16MB-aligned):
 *   [0, KEEP_SIZE)               = "keep" array → pointer chase, L2 resident
 *   [KEEP_SIZE, KEEP+EVICT_SIZE) = "evict" array → streaming, eviction pressure
 *
 * Both arrays map to the same L2 cache sets (aliasing by design: contiguous
 * and sized as multiples of 512KB = 1 way).
 *
 * Test pattern per iteration:
 *   1. Pointer-chase through "keep" (primes L2)
 *   2. Stream "evict" array (causes L2 eviction pressure)
 *   3. Pointer-chase through "keep" again — measures L2 hit vs memory miss
 *
 * Without L2 sector cache:
 *   Step 2 evicts keep data (evict needs more than remaining L2 ways)
 *   Step 3: ~memory latency per load (~100+ cycles)
 *
 * With L2 sector cache:
 *   Evict goes to sector 1, keep stays in sector 0
 *   Step 3: ~L2 latency per load (~40 cycles)
 *
 * Build:
 *   fcc -Nnoclang -O2 -Kocl,hpctag -o test_sector_l2_conflict test_sector_l2_conflict.c
 *
 * Build (fapp):
 *   fcc -Nnoclang -O2 -Kocl,hpctag -DUSE_FAPP -o test_sector_l2_conflict_fapp test_sector_l2_conflict.c
 *
 * Run:
 *   FLIB_SCCR_CNTL=TRUE FLIB_L1_SCCR_CNTL=TRUE ./test_sector_l2_conflict
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef USE_FAPP
#include <fj_tool/fapp.h>
#endif

#define CACHE_LINE  256

/* L2: 8MB, 16-way, 14 ways available to app. 1 way = 512KB */
#define L2_WAY_SIZE   (512 * 1024)   /* 512KB per way */
#define L2_TOTAL_WAYS 14             /* available to app */

/*
 * We test multiple keep/evict size combinations.
 * keep_ways: how many L2 ways worth of data to keep
 * evict_ways: how many L2 ways worth of data to stream (pressure)
 *
 * For sector cache to help:
 *   keep_ways + evict_ways > 14 (total overflow)
 *   sec0 >= keep_ways (protect keep)
 *   sec1 >= 1 (allow evict)
 */

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
 * Build a random pointer-chase chain.
 * Each cache line's first 8 bytes hold the BYTE OFFSET to the next line.
 * Chain visits all lines exactly once (Hamiltonian cycle).
 */
static void build_chase_chain(void *base, int nlines) {
    int *order = (int *)malloc(nlines * sizeof(int));
    int i, j, tmp;
    uint64_t *p;

    for (i = 0; i < nlines; i++) order[i] = i;

    /* Fisher-Yates shuffle */
    srand(12345);
    for (i = nlines - 1; i > 0; i--) {
        j = rand() % (i + 1);
        tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    for (i = 0; i < nlines - 1; i++) {
        p = (uint64_t *)((char *)base + (size_t)order[i] * CACHE_LINE);
        *p = (uint64_t)((size_t)order[i + 1] * CACHE_LINE);
    }
    p = (uint64_t *)((char *)base + (size_t)order[nlines - 1] * CACHE_LINE);
    *p = (uint64_t)((size_t)order[0] * CACHE_LINE);

    free(order);
}

/*
 * Pointer-chase through array. Returns final offset (for dependency chain).
 */
static inline uint64_t chase(void *base, int nlines) {
    uint64_t offset = 0;
    int i;
    for (i = 0; i < nlines; i++) {
        offset = *(volatile uint64_t *)((char *)base + offset);
    }
    return offset;
}

static void flush_range(void *base, size_t size) {
    size_t off;
    for (off = 0; off < size; off += CACHE_LINE)
        flush_line((char *)base + off);
    full_barrier();
}

/*
 * Baseline: no sector cache hints. Everything in default sector.
 */
static void run_nohint(
    void *keep, size_t keep_size, int keep_lines,
    uint64_t *evict, size_t evict_size,
    int num_iters, int chase_reps,
    uint64_t *out_reload_ticks, uint64_t *out_total_ticks)
{
    volatile uint64_t sink = 0;
    uint64_t t0, t1, t2, t3;
    uint64_t chase_sum = 0;
    uint64_t reload_acc = 0;
    int iter, r;
    size_t evict_elems = evict_size / sizeof(uint64_t);
    size_t j;

    flush_range(keep, keep_size);
    flush_range(evict, evict_size);
    full_barrier();

#ifdef USE_FAPP
    fapp_start("nohint_all", 1, 0);
#endif
    t0 = rdtsc();

    for (iter = 0; iter < num_iters; iter++) {
        /* Step 1: Prime keep into L2 via pointer chase */
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);

        /* Step 2: Stream evict (eviction pressure on L2) */
        {
            uint64_t s = 0;
            for (j = 0; j < evict_elems; j++)
                s += evict[j];
            sink = s;
        }

        /* Step 3: Reload keep via pointer chase — measure this */
        t2 = rdtsc();
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);
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
 * With L2 sector cache: evict → tagged (sector 1), keep → untagged (sector 0).
 * FCC pragma assigns evict to sector 1 and sets L2 way partition.
 * L1 partition is also set (L1=2 default) but the test targets L2.
 */
static void run_sector_l2_5(
    void *keep, size_t keep_size, int keep_lines,
    uint64_t *evict, size_t evict_size,
    int num_iters, int chase_reps,
    uint64_t *out_reload_ticks, uint64_t *out_total_ticks)
{
/* L2=5 means 5 ways for sector1(evict), 14-5=9 ways for sector0(keep) */
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign evict
    volatile uint64_t sink = 0;
    uint64_t t0, t1, t2, t3;
    uint64_t chase_sum = 0;
    uint64_t reload_acc = 0;
    int iter, r;
    size_t evict_elems = evict_size / sizeof(uint64_t);
    size_t j;

    flush_range(keep, keep_size);
    flush_range(evict, evict_size);
    full_barrier();

#ifdef USE_FAPP
    fapp_start("sector_l2_5", 1, 0);
#endif
    t0 = rdtsc();

    for (iter = 0; iter < num_iters; iter++) {
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);

        {
            uint64_t s = 0;
            for (j = 0; j < evict_elems; j++)
                s += evict[j];
            sink = s;
        }

        t2 = rdtsc();
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);
        t3 = rdtsc();
        reload_acc += t3 - t2;
    }

    t1 = rdtsc();
#ifdef USE_FAPP
    fapp_stop("sector_l2_5", 1, 0);
#endif
    sink += chase_sum;
    (void)sink;

    *out_reload_ticks = reload_acc;
    *out_total_ticks = t1 - t0;
}

/* L2=3: 3 ways for evict, 14-3=11 ways for keep */
static void run_sector_l2_3(
    void *keep, size_t keep_size, int keep_lines,
    uint64_t *evict, size_t evict_size,
    int num_iters, int chase_reps,
    uint64_t *out_reload_ticks, uint64_t *out_total_ticks)
{
#pragma procedure scache_isolate_way L2=3 L1=2
#pragma procedure scache_isolate_assign evict
    volatile uint64_t sink = 0;
    uint64_t t0, t1, t2, t3;
    uint64_t chase_sum = 0;
    uint64_t reload_acc = 0;
    int iter, r;
    size_t evict_elems = evict_size / sizeof(uint64_t);
    size_t j;

    flush_range(keep, keep_size);
    flush_range(evict, evict_size);
    full_barrier();

#ifdef USE_FAPP
    fapp_start("sector_l2_3", 1, 0);
#endif
    t0 = rdtsc();

    for (iter = 0; iter < num_iters; iter++) {
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);

        {
            uint64_t s = 0;
            for (j = 0; j < evict_elems; j++)
                s += evict[j];
            sink = s;
        }

        t2 = rdtsc();
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);
        t3 = rdtsc();
        reload_acc += t3 - t2;
    }

    t1 = rdtsc();
#ifdef USE_FAPP
    fapp_stop("sector_l2_3", 1, 0);
#endif
    sink += chase_sum;
    (void)sink;

    *out_reload_ticks = reload_acc;
    *out_total_ticks = t1 - t0;
}

/* L2=7: 7 ways for evict, 14-7=7 ways for keep */
static void run_sector_l2_7(
    void *keep, size_t keep_size, int keep_lines,
    uint64_t *evict, size_t evict_size,
    int num_iters, int chase_reps,
    uint64_t *out_reload_ticks, uint64_t *out_total_ticks)
{
#pragma procedure scache_isolate_way L2=7 L1=2
#pragma procedure scache_isolate_assign evict
    volatile uint64_t sink = 0;
    uint64_t t0, t1, t2, t3;
    uint64_t chase_sum = 0;
    uint64_t reload_acc = 0;
    int iter, r;
    size_t evict_elems = evict_size / sizeof(uint64_t);
    size_t j;

    flush_range(keep, keep_size);
    flush_range(evict, evict_size);
    full_barrier();

#ifdef USE_FAPP
    fapp_start("sector_l2_7", 1, 0);
#endif
    t0 = rdtsc();

    for (iter = 0; iter < num_iters; iter++) {
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);

        {
            uint64_t s = 0;
            for (j = 0; j < evict_elems; j++)
                s += evict[j];
            sink = s;
        }

        t2 = rdtsc();
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);
        t3 = rdtsc();
        reload_acc += t3 - t2;
    }

    t1 = rdtsc();
#ifdef USE_FAPP
    fapp_stop("sector_l2_7", 1, 0);
#endif
    sink += chase_sum;
    (void)sink;

    *out_reload_ticks = reload_acc;
    *out_total_ticks = t1 - t0;
}

/*
 * Manual SCCR test: write L2 SCCR directly with arbitrary sec0/sec1 values.
 * Tag evict pointer with bit 56 manually (same as pragma does).
 */
static void run_manual_l2(
    void *keep, size_t keep_size, int keep_lines,
    uint64_t *evict, size_t evict_size,
    int num_iters, int chase_reps,
    unsigned sec0, unsigned sec1,
    uint64_t *out_reload_ticks, uint64_t *out_total_ticks)
{
/* Use pragma to init libsec (enable EL0 SCCR access), then override manually */
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign evict

    volatile uint64_t sink = 0;
    uint64_t t0, t1, t2, t3;
    uint64_t chase_sum = 0;
    uint64_t reload_acc = 0;
    int iter, r;
    size_t evict_elems = evict_size / sizeof(uint64_t);
    size_t j;

    /* Override L2 SCCR with custom values */
    uint64_t l2_val = ((uint64_t)(sec1 & 0x1F) << 8) | (sec0 & 0x1F);
    __asm__ volatile("msr S3_3_C15_C8_2, %0" :: "r"(l2_val));
    __asm__ volatile("isb" ::: "memory");

    /* Also set L1 to a reasonable value */
    uint64_t l1_val = 0x22; /* sec0=2, sec1=2 */
    __asm__ volatile("msr S3_3_C11_C8_2, %0" :: "r"(l1_val));
    __asm__ volatile("isb" ::: "memory");

    flush_range(keep, keep_size);
    flush_range(evict, evict_size);
    full_barrier();

    t0 = rdtsc();

    for (iter = 0; iter < num_iters; iter++) {
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);

        {
            uint64_t s = 0;
            for (j = 0; j < evict_elems; j++)
                s += evict[j];
            sink = s;
        }

        t2 = rdtsc();
        for (r = 0; r < chase_reps; r++)
            chase_sum += chase(keep, keep_lines);
        t3 = rdtsc();
        reload_acc += t3 - t2;
    }

    t1 = rdtsc();
    sink += chase_sum;
    (void)sink;

    *out_reload_ticks = reload_acc;
    *out_total_ticks = t1 - t0;
}

static void print_result(const char *label,
                         int keep_ways, int evict_ways,
                         int keep_lines, int chase_reps, int num_iters,
                         uint64_t reload_ticks, uint64_t total_ticks,
                         uint64_t freq,
                         uint64_t reload_baseline, uint64_t total_baseline)
{
    int total_chases = chase_reps * keep_lines * num_iters;
    /* cntvct at 100MHz, CPU at 2000MHz → 1 tick = 20 cycles */
    double cyc_per_load = (double)reload_ticks * 20.0 / total_chases;
    double reload_speedup = (reload_baseline > 0) ?
        (double)reload_baseline / reload_ticks : 1.0;
    double total_speedup = (total_baseline > 0) ?
        (double)total_baseline / total_ticks : 1.0;

    const char *level;
    if (cyc_per_load < 50)
        level = "L2";
    else if (cyc_per_load < 150)
        level = "Mem(near)";
    else
        level = "Mem(far)";

    printf("  %-28s %7.1f cyc/load (%s)  reload=%.2fx  total=%.2fx  [%.2f ms]\n",
           label, cyc_per_load, level, reload_speedup, total_speedup,
           (double)total_ticks / freq * 1e3);
}

int main() {
    uint64_t freq = rdfreq();
    const char *sccr_env = getenv("FLIB_SCCR_CNTL");
    const char *l1_env   = getenv("FLIB_L1_SCCR_CNTL");

    printf("=== Sector Cache L2 Way-Conflict Test (Pointer Chase) ===\n\n");
    printf("L2 cache      : 8MB, 16-way, 256B line, 1 way = 512KB\n");
    printf("Available ways: %d (14 of 16 available to app)\n", L2_TOTAL_WAYS);
    printf("Timer freq    : %lu MHz\n", freq / 1000000);
    printf("FLIB_SCCR_CNTL    = %s\n", sccr_env ? sccr_env : "(not set)");
    printf("FLIB_L1_SCCR_CNTL = %s\n\n", l1_env ? l1_env : "(not set)");

    /*
     * Test configurations:
     *   keep=2MB (4 ways), evict=6MB (12 ways) → total 16 > 14 → overflows
     *   keep=3.5MB (7 ways), evict=5MB (10 ways) → total 17 > 14 → overflows
     *   keep=1MB (2 ways), evict=7MB (14 ways) → total 16 > 14 → overflows
     */
    struct test_config {
        int keep_ways;
        int evict_ways;
    } configs[] = {
        { 2, 14 },  /* 1MB keep, 7MB evict (strong pressure) */
        { 4, 12 },  /* 2MB keep, 6MB evict */
        { 7, 10 },  /* 3.5MB keep, 5MB evict */
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

    for (int c = 0; c < nconfigs; c++) {
        int keep_ways = configs[c].keep_ways;
        int evict_ways = configs[c].evict_ways;
        size_t keep_size = (size_t)keep_ways * L2_WAY_SIZE;
        size_t evict_size = (size_t)evict_ways * L2_WAY_SIZE;
        int keep_lines = keep_size / CACHE_LINE;
        size_t total_size = keep_size + evict_size;

        /* Reduce iteration count for large datasets */
        int num_iters = 20;
        int chase_reps = 2;
        if (keep_size > 4 * 1024 * 1024) {
            num_iters = 10;
            chase_reps = 1;
        }

        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("Config: keep=%dMB (%d ways), evict=%dMB (%d ways), total=%dMB\n",
               (int)(keep_size / (1024*1024)), keep_ways,
               (int)(evict_size / (1024*1024)), evict_ways,
               (int)(total_size / (1024*1024)));
        printf("  keep_lines=%d, iters=%d, chase_reps=%d\n",
               keep_lines, num_iters, chase_reps);
        printf("  Overflow: %d + %d = %d ways needed vs %d available\n\n",
               keep_ways, evict_ways, keep_ways + evict_ways, L2_TOTAL_WAYS);

        /* Allocate 16MB-aligned contiguous block */
        void *block;
        posix_memalign(&block, 16 * 1024 * 1024, total_size);
        if (!block) {
            fprintf(stderr, "alloc failed for %zu bytes\n", total_size);
            continue;
        }
        memset(block, 0, total_size);

        void *keep_base = block;
        uint64_t *evict_base = (uint64_t *)((char *)block + keep_size);

        /* Build pointer-chase chain in keep array */
        build_chase_chain(keep_base, keep_lines);

        /* Initialize evict array */
        {
            size_t evict_elems = evict_size / sizeof(uint64_t);
            for (size_t j = 0; j < evict_elems; j++)
                evict_base[j] = (uint64_t)j;
        }

        uint64_t reload_nohint, total_nohint;
        uint64_t reload_sec, total_sec;

        /* Warmup */
        run_nohint(keep_base, keep_size, keep_lines,
                   evict_base, evict_size, num_iters/2, chase_reps,
                   &reload_nohint, &total_nohint);

        /* Baseline: no sector cache */
        run_nohint(keep_base, keep_size, keep_lines,
                   evict_base, evict_size, num_iters, chase_reps,
                   &reload_nohint, &total_nohint);

        print_result("nohint (baseline)", keep_ways, evict_ways,
                     keep_lines, chase_reps, num_iters,
                     reload_nohint, total_nohint, freq, 0, 0);

        /* Sector cache L2=5 (sec0=9, sec1=5) */
        run_sector_l2_5(keep_base, keep_size, keep_lines,
                        evict_base, evict_size, num_iters, chase_reps,
                        &reload_sec, &total_sec);
        print_result("sector L2=5 (sec0=9,sec1=5)", keep_ways, evict_ways,
                     keep_lines, chase_reps, num_iters,
                     reload_sec, total_sec, freq, reload_nohint, total_nohint);

        /* Sector cache L2=3 (sec0=11, sec1=3) */
        run_sector_l2_3(keep_base, keep_size, keep_lines,
                        evict_base, evict_size, num_iters, chase_reps,
                        &reload_sec, &total_sec);
        print_result("sector L2=3 (sec0=11,sec1=3)", keep_ways, evict_ways,
                     keep_lines, chase_reps, num_iters,
                     reload_sec, total_sec, freq, reload_nohint, total_nohint);

        /* Sector cache L2=7 (sec0=7, sec1=7) */
        run_sector_l2_7(keep_base, keep_size, keep_lines,
                        evict_base, evict_size, num_iters, chase_reps,
                        &reload_sec, &total_sec);
        print_result("sector L2=7 (sec0=7,sec1=7)", keep_ways, evict_ways,
                     keep_lines, chase_reps, num_iters,
                     reload_sec, total_sec, freq, reload_nohint, total_nohint);

        /* Manual L2 SCCR: probe boundary values */
        printf("\n  --- Manual L2 SCCR probing ---\n");

        struct {
            unsigned sec0;
            unsigned sec1;
            const char *desc;
        } probes[] = {
            { 14,  1, "sec0=14,sec1=1  (max keep)" },
            { 13,  1, "sec0=13,sec1=1" },
            { 10,  4, "sec0=10,sec1=4" },
            {  7,  7, "sec0=7,sec1=7   (equal split)" },
            {  4, 10, "sec0=4,sec1=10  (more evict)" },
            { 14, 14, "sec0=14,sec1=14 (oversubscribed)" },
            { 16, 16, "sec0=16,sec1=16 (beyond 14?)" },
            {  0,  0, "sec0=0,sec1=0   (no partition)" },
        };
        int nprobes = sizeof(probes) / sizeof(probes[0]);

        for (int p = 0; p < nprobes; p++) {
            char label[64];
            snprintf(label, sizeof(label), "%s", probes[p].desc);
            run_manual_l2(keep_base, keep_size, keep_lines,
                          evict_base, evict_size, num_iters, chase_reps,
                          probes[p].sec0, probes[p].sec1,
                          &reload_sec, &total_sec);
            print_result(label, keep_ways, evict_ways,
                         keep_lines, chase_reps, num_iters,
                         reload_sec, total_sec, freq,
                         reload_nohint, total_nohint);
        }

        printf("\n");
        free(block);
    }

    printf("=== Latency Reference ===\n");
    printf("  L1 hit:   ~4-6 cycles\n");
    printf("  L2 hit:   ~37-40 cycles\n");
    printf("  Memory:   ~100-200+ cycles\n");
    printf("  (cntvct runs at 100MHz, CPU at 2000MHz, 1 tick = 20 cycles)\n\n");

    printf("=== Interpretation ===\n");
    printf("  If nohint shows ~100+ cyc/load → memory (L2 evicted by pressure)\n");
    printf("  If sector shows ~40 cyc/load   → L2 hit (sector cache protected keep)\n");
    printf("  Speedup = nohint/sector should be 2-3x if L2 partitioning works\n");

    return 0;
}
