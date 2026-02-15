/*
 * A64FX Sector Cache L1 Reuse Test - FCC Pragma Version
 *
 * Uses FCC OCL pragmas (scache_isolate_way/scache_isolate_assign) to enable
 * sector cache via the Fujitsu runtime library (__jwe_xset_sccr).
 *
 * MUST be compiled with FCC traditional mode:
 *   fcc -Nnoclang -O2 -Kocl,hpctag -o test_sector_l1_pragma test_sector_l1_pragma.c -lm
 *
 * With fapp:
 *   fcc -Nnoclang -O2 -Kocl,hpctag -DUSE_FAPP -o test_sector_l1_pragma_fapp test_sector_l1_pragma.c -lm
 *
 * Run:
 *   export FLIB_SCCR_CNTL=TRUE
 *   export FLIB_L1_SCCR_CNTL=TRUE
 *   ./test_sector_l1_pragma
 *
 * Pattern per iteration (same as manual-tag version):
 *   Step 1: Load reuse data (24KB) into L1
 *   Step 2: Stream cold data (256KB) through L1  (eviction pressure!)
 *   Step 3: Reload reuse data -- L1 hit if sector cache partitioning works
 *
 * Two functions:
 *   run_nohint():  no pragma, no sector cache → streaming evicts reuse data
 *   run_sector():  #pragma procedure scache_isolate_way L2=5 L1=2
 *                  #pragma procedure scache_isolate_assign stream
 *                  → streaming goes to sector 1, reuse stays in sector 0
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef USE_FAPP
#include <fj_tool/fapp.h>
#endif

#define L1_LINE      256           /* A64FX L1 cache line = 256 bytes */
#define REUSE_SIZE   (24 * 1024)   /* 24KB - fits in one sector (32KB) */
#define STREAM_SIZE  (256 * 1024)  /* 256KB - much larger than L1 */
#define NUM_ITERS    100

static inline uint64_t read_cycle(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_freq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

static void flush_cache(void* ptr, size_t size) {
    char* p = (char*)ptr;
    size_t i;
    for (i = 0; i < size; i += L1_LINE) {
        __asm__ volatile("dc civac, %0" :: "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
}

static inline void memory_fence(void) {
    __asm__ volatile("dmb ish" ::: "memory");
}

/*
 * Baseline: no sector cache.
 * Streaming data evicts reuse data from L1 → step 3 causes L1 misses.
 */
static uint64_t run_nohint(
    float* reuse, float* stream,
    int reuse_n, int stream_n, int iters)
{
    volatile float sink = 0.0f;
    int iter, i;
    uint64_t start, end;

    flush_cache(reuse, reuse_n * sizeof(float));
    flush_cache(stream, stream_n * sizeof(float));
    memory_fence();

#ifdef USE_FAPP
    fapp_start("nohint", 1, 0);
#endif

    start = read_cycle();

    for (iter = 0; iter < iters; iter++) {
        float sum = 0.0f;
        /* Step 1: Load reuse data into L1 */
        for (i = 0; i < reuse_n; i++) sum += reuse[i];
        /* Step 2: Stream cold data (eviction pressure) */
        for (i = 0; i < stream_n; i++) sum += stream[i];
        /* Step 3: Reload reuse data -- L1 miss expected */
        for (i = 0; i < reuse_n; i++) sum += reuse[i];
        sink = sum;
    }

    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("nohint", 1, 0);
#endif

    memory_fence();
    (void)sink;
    return end - start;
}

/*
 * With sector cache: stream → sector 1, reuse → sector 0.
 * The compiler generates __jwe_xset_sccr() calls and tagged addresses.
 * SCCR L1 config: sector0=2ways (32KB), sector1=2ways (32KB).
 * Reuse data (24KB) should survive in sector 0 even while streaming.
 */
static uint64_t run_sector(
    float* reuse, float* stream,
    int reuse_n, int stream_n, int iters)
{
#pragma procedure scache_isolate_way L2=5 L1=2
#pragma procedure scache_isolate_assign stream
    volatile float sink = 0.0f;
    int iter, i;
    uint64_t start, end;

    flush_cache(reuse, reuse_n * sizeof(float));
    flush_cache(stream, stream_n * sizeof(float));
    memory_fence();

#ifdef USE_FAPP
    fapp_start("sector", 1, 0);
#endif

    start = read_cycle();

    for (iter = 0; iter < iters; iter++) {
        float sum = 0.0f;
        /* Step 1: Load reuse data into L1 (sector 0) */
        for (i = 0; i < reuse_n; i++) sum += reuse[i];
        /* Step 2: Stream cold data (sector 1 — should not evict reuse) */
        for (i = 0; i < stream_n; i++) sum += stream[i];
        /* Step 3: Reload reuse data -- L1 hit expected */
        for (i = 0; i < reuse_n; i++) sum += reuse[i];
        sink = sum;
    }

    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("sector", 1, 0);
#endif

    memory_fence();
    (void)sink;
    return end - start;
}

int main(int argc, char* argv[]) {
    uint64_t freq = get_freq();
    int reuse_n = REUSE_SIZE / sizeof(float);
    int stream_n = STREAM_SIZE / sizeof(float);
    const char* env_sccr;
    const char* env_l1;
    const char* env_l2;
    float* reuse;
    float* stream;
    int i;
    uint64_t cyc_nohint, cyc_sector;
    double total_bytes, speedup;

    printf("=== Sector Cache L1 Reuse - FCC Pragma Version ===\n\n");

    printf("Compiler mode   : FCC traditional (-Nnoclang -Kocl,hpctag)\n");
    printf("Reuse data      : %d KB\n", REUSE_SIZE / 1024);
    printf("Stream data     : %d KB\n", STREAM_SIZE / 1024);
    printf("Iterations      : %d\n", NUM_ITERS);
    printf("L1 sector split : sector0=2ways(32KB), sector1=2ways(32KB)\n");
    printf("L2 sector split : sector0=9ways, sector1=5ways\n");
    printf("\n");

    /* Check FLIB environment variables */
    env_sccr = getenv("FLIB_SCCR_CNTL");
    env_l1   = getenv("FLIB_L1_SCCR_CNTL");
    env_l2   = getenv("FLIB_L2_SECTOR_NWAYS_INIT");
    printf("=== FLIB Environment ===\n");
    printf("FLIB_SCCR_CNTL            = %s (default=TRUE)\n",
           env_sccr ? env_sccr : "(not set)");
    printf("FLIB_L1_SCCR_CNTL         = %s (default=TRUE)\n",
           env_l1   ? env_l1   : "(not set)");
    printf("FLIB_L2_SECTOR_NWAYS_INIT = %s\n",
           env_l2   ? env_l2   : "(not set)");
    printf("\n");

    /* Allocate aligned buffers */
    reuse = NULL;
    stream = NULL;
    posix_memalign((void**)&reuse, L1_LINE, REUSE_SIZE);
    posix_memalign((void**)&stream, L1_LINE, STREAM_SIZE);

    if (!reuse || !stream) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Initialize */
    for (i = 0; i < reuse_n; i++)
        reuse[i] = (float)(i % 100) / 10.0f;
    for (i = 0; i < stream_n; i++)
        stream[i] = (float)(i % 100) / 10.0f;

    /* Warmup */
    run_nohint(reuse, stream, reuse_n, stream_n, 3);
    run_sector(reuse, stream, reuse_n, stream_n, 3);

    /* Test 1: No sector cache (baseline) */
    cyc_nohint = run_nohint(reuse, stream, reuse_n, stream_n, NUM_ITERS);

    /* Test 2: With sector cache pragma */
    cyc_sector = run_sector(reuse, stream, reuse_n, stream_n, NUM_ITERS);

    /* Results */
    total_bytes = (double)(REUSE_SIZE * 2 + STREAM_SIZE) * NUM_ITERS;
    speedup = (double)cyc_nohint / cyc_sector;

    printf("=== Results ===\n");
    printf("  nohint : %10lu cycles (%6.2f GB/s) [baseline]\n",
           cyc_nohint, total_bytes / ((double)cyc_nohint / freq) / 1e9);
    printf("  sector : %10lu cycles (%6.2f GB/s) speedup=%.2fx\n",
           cyc_sector, total_bytes / ((double)cyc_sector / freq) / 1e9, speedup);
    printf("\n");

    if (speedup > 1.05) {
        printf("PASS: sector cache pragma shows %.2fx speedup\n", speedup);
        printf("  -> __jwe_xset_sccr() successfully programmed SCCR\n");
        printf("  -> L1 partitioning is ACTIVE on this system\n");
    } else if (speedup > 0.95) {
        printf("NO EFFECT: sector cache pragma shows %.2fx (no partitioning)\n", speedup);
        printf("  -> __jwe_xset_sccr() may have failed or kernel doesn't support SCCR\n");
        printf("  Check with: fcc -Nnoclang -O2 -Kocl,hpctag -S test_sector_l1_pragma.c\n");
        printf("  Look for: bl __jwe_xset_sccr in assembly\n");
    } else {
        printf("REGRESSION: sector cache pragma shows %.2fx (overhead)\n", speedup);
        printf("  -> __jwe_xset_sccr() overhead without benefit\n");
    }
    printf("\n");

    free(reuse);
    free(stream);
    return 0;
}
