/*
 * Gather-Stream Sector Cache Benchmark
 *
 * Pattern: result += values[i] * table[indices[i]]
 *   - table[]:   medium, RANDOM access → keep in cache (sector 0)
 *   - values[]:  large, sequential    → stream through (sector 1)
 *   - indices[]: large, sequential    → stream through (sector 1)
 *
 * This is the SpMV inner loop: sequential val/col_idx stream,
 * random access into x vector. The stream evicts the table
 * under LRU. Sector cache protects the table.
 *
 * Build (single-thread):
 *   fcc -Nnoclang -O2 -Kocl,hpctag -o bench_gather_sector \
 *       bench_gather_sector.c -lm
 *
 * Build (multi-core):
 *   fcc -Nnoclang -O2 -Kocl,hpctag,openmp -o bench_gather_sector \
 *       bench_gather_sector.c -lm
 *
 * Run:
 *   FLIB_SCCR_CNTL=TRUE FLIB_L1_SCCR_CNTL=TRUE FLIB_L2_SCCR_CNTL_EX=TRUE \
 *   OMP_NUM_THREADS=12 numactl --cpunodebind=4 --membind=4 ./bench_gather_sector
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_FAPP
extern void fapp_start(const char *, int, int);
extern void fapp_stop(const char *, int, int);
#endif

static inline uint64_t rdtsc(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t rdfreq(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}
static inline void barrier(void) {
    __asm__ volatile("dsb ish" ::: "memory");
}
static inline void flush_line(void *p) {
    __asm__ volatile("dc civac, %0" :: "r"(p) : "memory");
}

#define APPLY_TAG(ptr, tag) \
    ((__typeof__(ptr))((uintptr_t)(ptr) | ((uint64_t)(tag) << 56)))

static volatile double g_sink;

/* ──────────────────────────────────────────────────────────────────
 * Gather-stream kernel: no hints (baseline)
 * ────────────────────────────────────────────────────────────────── */
double gather_nohint(
    const float *table, const float *values,
    const int *indices, long count)
{
    double sum = 0;
    long i;
    for (i = 0; i < count; i++) {
        sum += (double)values[i] * (double)table[indices[i]];
    }
    return sum;
}

/* ──────────────────────────────────────────────────────────────────
 * Gather-stream kernel: tagged pointers
 * values+indices → sector 1 (stream), table → sector 0 (keep)
 * ────────────────────────────────────────────────────────────────── */
double gather_tagged(
    const float *table, const float *values,
    const int *indices, long count,
    unsigned stream_tag)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign values
    const float *val_t = APPLY_TAG(values, stream_tag);
    const int *idx_t = APPLY_TAG(indices, stream_tag);
    __asm__ volatile("" : "+r"(val_t));
    __asm__ volatile("" : "+r"(idx_t));

    double sum = 0;
    long i;
    for (i = 0; i < count; i++) {
        int col = idx_t[i];
        __asm__ volatile("" : "+r"(col));
        sum += (double)val_t[i] * (double)table[col];
    }
    return sum;
}

/* ──────────────────────────────────────────────────────────────────
 * Gather-stream kernel: manual SCCR + tagged pointers
 * ────────────────────────────────────────────────────────────────── */
double gather_manual(
    const float *table, const float *values,
    const int *indices, long count,
    unsigned stream_tag,
    unsigned l1s0, unsigned l1s1, unsigned l2s0, unsigned l2s1)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign values
    uint64_t l1v = (l1s1 & 0x7) << 4 | (l1s0 & 0x7);
    uint64_t l2v = ((uint64_t)(l2s1 & 0x1F) << 8) | (l2s0 & 0x1F);
    __asm__ volatile("msr S3_3_C11_C8_2, %0" :: "r"(l1v));
    __asm__ volatile("msr S3_3_C15_C8_2, %0" :: "r"(l2v));
    __asm__ volatile("isb" ::: "memory");

    const float *val_t = APPLY_TAG(values, stream_tag);
    const int *idx_t = APPLY_TAG(indices, stream_tag);
    __asm__ volatile("" : "+r"(val_t));
    __asm__ volatile("" : "+r"(idx_t));

    double sum = 0;
    long i;
    for (i = 0; i < count; i++) {
        int col = idx_t[i];
        __asm__ volatile("" : "+r"(col));
        sum += (double)val_t[i] * (double)table[col];
    }
    return sum;
}

/* ──────────────────────────────────────────────────────────────────
 * OpenMP versions: each thread processes its own chunk
 * ────────────────────────────────────────────────────────────────── */
#ifdef _OPENMP

double gather_nohint_omp(
    const float *table, const float *values,
    const int *indices, long count)
{
    double total = 0;
#pragma omp parallel reduction(+:total)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        long chunk = count / nth;
        long start = tid * chunk;
        long end = (tid == nth - 1) ? count : start + chunk;
        double sum = 0;
        long i;
        for (i = start; i < end; i++) {
            sum += (double)values[i] * (double)table[indices[i]];
        }
        total += sum;
    }
    return total;
}

double gather_tagged_omp(
    const float *table, const float *values,
    const int *indices, long count,
    unsigned stream_tag)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign values
    double total = 0;
#pragma omp parallel reduction(+:total)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        long chunk = count / nth;
        long start = tid * chunk;
        long end = (tid == nth - 1) ? count : start + chunk;

        const float *val_t = APPLY_TAG(values + start, stream_tag);
        const int *idx_t = APPLY_TAG(indices + start, stream_tag);
        __asm__ volatile("" : "+r"(val_t));
        __asm__ volatile("" : "+r"(idx_t));

        double sum = 0;
        long len = end - start;
        long i;
        for (i = 0; i < len; i++) {
            int col = idx_t[i];
            __asm__ volatile("" : "+r"(col));
            sum += (double)val_t[i] * (double)table[col];
        }
        total += sum;
    }
    return total;
}

double gather_manual_omp(
    const float *table, const float *values,
    const int *indices, long count,
    unsigned stream_tag,
    unsigned l1s0, unsigned l1s1, unsigned l2s0, unsigned l2s1)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign values
    uint64_t l1v = (l1s1 & 0x7) << 4 | (l1s0 & 0x7);
    uint64_t l2v = ((uint64_t)(l2s1 & 0x1F) << 8) | (l2s0 & 0x1F);
    double total = 0;
#pragma omp parallel reduction(+:total)
    {
        __asm__ volatile("msr S3_3_C11_C8_2, %0" :: "r"(l1v));
        __asm__ volatile("msr S3_3_C15_C8_2, %0" :: "r"(l2v));
        __asm__ volatile("isb" ::: "memory");

        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        long chunk = count / nth;
        long start = tid * chunk;
        long end = (tid == nth - 1) ? count : start + chunk;

        const float *val_t = APPLY_TAG(values + start, stream_tag);
        const int *idx_t = APPLY_TAG(indices + start, stream_tag);
        __asm__ volatile("" : "+r"(val_t));
        __asm__ volatile("" : "+r"(idx_t));

        double sum = 0;
        long len = end - start;
        long i;
        for (i = 0; i < len; i++) {
            int col = idx_t[i];
            __asm__ volatile("" : "+r"(col));
            sum += (double)val_t[i] * (double)table[col];
        }
        total += sum;
    }
    return total;
}

#endif /* _OPENMP */

/* ──────────────────────────────────────────────────────────────────
 * Flush table from cache (force cold start)
 * ────────────────────────────────────────────────────────────────── */
static void flush_range(void *base, long size) {
    long off;
    for (off = 0; off < size; off += 256)
        flush_line((char *)base + off);
    barrier();
}

/* ──────────────────────────────────────────────────────────────────
 * Test runner
 * ────────────────────────────────────────────────────────────────── */
static void run_test(
    long table_elems, long stream_elems, int nthreads, const char *desc)
{
    uint64_t freq = rdfreq();
    long table_bytes = table_elems * 4;
    long val_bytes = stream_elems * 4;
    long idx_bytes = stream_elems * 4;
    long total_stream = val_bytes + idx_bytes;

    printf("--------------------------------------------------------------------\n");
    printf("%s (%d threads)\n", desc, nthreads);
    printf("  table : %ld KB = %ld elems (%.1f L2 ways) -- KEEP (random access)\n",
           table_bytes/1024, table_elems, (double)table_bytes/(512*1024));
    printf("  stream: %ld MB (values %ldMB + indices %ldMB) -- DISCARD\n",
           total_stream/(1024*1024), val_bytes/(1024*1024), idx_bytes/(1024*1024));
    printf("  ratio : %.0f:1 stream-to-table\n\n",
           (double)total_stream / table_bytes);

    /* Allocate */
    float *table   = (float *)aligned_alloc(256, table_bytes);
    float *values  = (float *)aligned_alloc(256, val_bytes);
    int   *indices = (int *)aligned_alloc(256, idx_bytes);
    if (!table || !values || !indices) {
        fprintf(stderr, "alloc failed\n");
        return;
    }

    /* Init table: random values */
    long i;
    srand(12345);
    for (i = 0; i < table_elems; i++)
        table[i] = (float)(rand() % 1000) / 1000.0f;

    /* Init stream: sequential values, random indices into table */
    for (i = 0; i < stream_elems; i++) {
        values[i] = (float)(i % 1000) / 1000.0f;
        indices[i] = rand() % table_elems;
    }

    /* Determine iteration count */
    int iters;
    double bytes_per_iter = (double)total_stream;
#ifdef USE_FAPP
    double target_bytes = 2e9;  /* fewer iters for fapp overhead */
#else
    double target_bytes = 20e9; /* ~20GB total to amortize overhead */
#endif
    iters = (int)(target_bytes / bytes_per_iter);
    if (iters < 2) iters = 2;
    if (iters > 200) iters = 200;

    double gf_base = 0;
    double bw, sec;
    uint64_t t0, t1, ticks;

    /* Helper macro for timing + optional fapp */
    #ifdef USE_FAPP
    #define FAPP_START(lbl) fapp_start(lbl, 1, 0)
    #define FAPP_STOP(lbl)  fapp_stop(lbl, 1, 0)
    #else
    #define FAPP_START(lbl)
    #define FAPP_STOP(lbl)
    #endif

    #define RUN_BENCH(label, fapp_id, call) do {                        \
        flush_range(table, table_bytes);                               \
        flush_range(values, val_bytes);                                \
        flush_range(indices, idx_bytes);                               \
        g_sink = call;   /* warmup */                                  \
        barrier();                                                     \
        barrier();                                                     \
        FAPP_START(fapp_id);                                           \
        t0 = rdtsc();                                                  \
        for (i = 0; i < iters; i++) {                                  \
            g_sink += call;                                            \
            __asm__ volatile("" ::: "memory");                         \
        }                                                              \
        barrier();                                                     \
        t1 = rdtsc();                                                  \
        FAPP_STOP(fapp_id);                                            \
        ticks = t1 - t0;                                               \
        sec = (double)ticks / freq;                                    \
        bw = (double)total_stream * iters / sec / 1e9;                 \
        if (gf_base == 0) gf_base = bw;                               \
        printf("  %-42s %6.2f GB/s  %10lu ticks  %.3fx\n",            \
               label, bw, ticks, bw / gf_base);                       \
    } while(0)

    if (nthreads == 1) {
        RUN_BENCH("nohint (baseline)", "nohint",
            gather_nohint(table, values, indices, stream_elems));

        RUN_BENCH("tagged B=0xA (SCE+sec2)", "tag_0xA",
            gather_tagged(table, values, indices, stream_elems, 0xA));

        RUN_BENCH("tagged B=0x2 (sec2, no SCE)", "tag_0x2",
            gather_tagged(table, values, indices, stream_elems, 0x2));

        RUN_BENCH("manual L1(3,1) L2(13,1) B=0xA", "manual_13_1",
            gather_manual(table, values, indices, stream_elems,
                          0xA, 3, 1, 13, 1));

        RUN_BENCH("manual L1(3,1) L2(12,2) B=0xA", "manual_12_2",
            gather_manual(table, values, indices, stream_elems,
                          0xA, 3, 1, 12, 2));

        RUN_BENCH("manual L1(2,2) L2(9,5) B=0xA", "manual_9_5",
            gather_manual(table, values, indices, stream_elems,
                          0xA, 2, 2, 9, 5));
    }
#ifdef _OPENMP
    else {
        RUN_BENCH("nohint (baseline)", "nohint",
            gather_nohint_omp(table, values, indices, stream_elems));

        RUN_BENCH("tagged B=0xA (SCE+sec2)", "tag_0xA",
            gather_tagged_omp(table, values, indices, stream_elems, 0xA));

        RUN_BENCH("tagged B=0x2 (sec2, no SCE)", "tag_0x2",
            gather_tagged_omp(table, values, indices, stream_elems, 0x2));

        RUN_BENCH("manual L1(3,1) L2(13,1) B=0xA", "manual_13_1",
            gather_manual_omp(table, values, indices, stream_elems,
                              0xA, 3, 1, 13, 1));

        RUN_BENCH("manual L1(3,1) L2(12,2) B=0xA", "manual_12_2",
            gather_manual_omp(table, values, indices, stream_elems,
                              0xA, 3, 1, 12, 2));

        RUN_BENCH("manual L1(2,2) L2(9,5) B=0xA", "manual_9_5",
            gather_manual_omp(table, values, indices, stream_elems,
                              0xA, 2, 2, 9, 5));
    }
#endif

    #undef RUN_BENCH

    printf("\n");
    free(table); free(values); free(indices);
}

int main(void) {
    uint64_t freq = rdfreq();
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    printf("=== Gather-Stream Sector Cache Benchmark ===\n\n");
    printf("Pattern: result += values[i] * table[indices[i]]\n");
    printf("  table   = random access (KEEP in cache, sector 0)\n");
    printf("  val+idx = sequential stream (DISCARD, sector 1)\n\n");
    printf("Threads    : %d\n", nthreads);
    printf("Timer freq : %lu MHz\n\n", freq / 1000000);

    const char *e1 = getenv("FLIB_SCCR_CNTL");
    const char *e2 = getenv("FLIB_L1_SCCR_CNTL");
    const char *e3 = getenv("FLIB_L2_SCCR_CNTL_EX");
    printf("FLIB_SCCR_CNTL       = %s\n", e1 ? e1 : "(not set)");
    printf("FLIB_L1_SCCR_CNTL    = %s\n", e2 ? e2 : "(not set)");
    printf("FLIB_L2_SCCR_CNTL_EX = %s\n\n", e3 ? e3 : "(not set)");

    /*
     * Table sizes vs L2 (8MB, 14 usable ways, 1 way=512KB):
     *   128K elems =  512KB = 1 way   → easily protected
     *   512K elems = 2048KB = 4 ways  → needs ~30% of L2
     *   1M elems   = 4096KB = 8 ways  → needs ~57% of L2
     *
     * Stream size: 16M elems = val(64MB) + idx(64MB) = 128MB
     *   >> L2, forces streaming from HBM
     */
    long stream = 16L * 1024 * 1024;  /* 16M elements */

#ifdef USE_FAPP
    /* Fapp mode: fewer table sizes, keep profile focused */
    run_test(512L * 1024, stream, nthreads,
             "table=2MB (4 ways), stream=128MB");

    run_test(1024L * 1024, stream, nthreads,
             "table=4MB (8 ways), stream=128MB");
#else
    /* table=512KB (1 L2 way): easy to protect */
    run_test(128L * 1024, stream, nthreads,
             "table=512KB (1 way), stream=128MB");

    /* table=2MB (4 L2 ways): significant L2 fraction */
    run_test(512L * 1024, stream, nthreads,
             "table=2MB (4 ways), stream=128MB");

    /* table=4MB (8 L2 ways): over half L2 */
    run_test(1024L * 1024, stream, nthreads,
             "table=4MB (8 ways), stream=128MB");

    /* table=6MB (12 L2 ways): nearly fills L2 */
    run_test(1536L * 1024, stream, nthreads,
             "table=6MB (12 ways), stream=128MB");
#endif

    printf("====================================================================\n");
    return 0;
}
