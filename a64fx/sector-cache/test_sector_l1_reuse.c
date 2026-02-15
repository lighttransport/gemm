/*
 * A64FX Sector Cache L1 Reuse Test
 *
 * Proves that sector cache hints protect L1 reuse data from streaming eviction.
 *
 * Pattern per iteration:
 *   Step 1: Load reuse data (24KB) into L1
 *   Step 2: Stream cold data (256KB) through L1  (eviction pressure!)
 *   Step 3: Reload reuse data -- L1 hit if sector hints worked, miss if not
 *
 * Test variants:
 *   nohint:  reuse=0x0, stream=0x0  (no sector hints, no SCCR)
 *   sector:  reuse=0x9, stream=0xA  (strong sector 0 / sector 1) + SCCR(2,2)
 *   bypass:  reuse=0x0, stream=0xB  (streaming bypasses L1 entirely)
 *
 * SCCR programming:
 *   Before "sector" test: write IMP_SCCR_L1_EL0 = 0x00020002 (2 ways per sector)
 *   After "sector" test:  write IMP_SCCR_L1_EL0 = 0x00040000 (reset: 4 ways sector0)
 *
 * L1 layout with SCCR(2,2):
 *   64KB total, 4-way set associative, 256B lines
 *   Sector 0 = ways 0,1 (32KB)   ← reuse data (24KB) fits here
 *   Sector 1 = ways 2,3 (32KB)   ← streaming data goes here
 *
 * Expected L1D_CACHE_REFILL difference:
 *   nohint: step 3 always misses → ~96 extra cache line refills/iter × N iters
 *   sector: step 3 always hits  → ~0 extra refills
 *   bypass: step 3 always hits  → ~0 extra refills (streaming never entered L1)
 *
 * Build:  make l1_reuse          (cycle count only)
 *         make l1_reuse_fapp     (with fapp instrumentation)
 *
 * Run with env vars:
 *   export FLIB_SCCR_CNTL=TRUE
 *   export FLIB_L1_SCCR_CNTL=TRUE
 *   export FLIB_L2_SECTOR_NWAYS_INIT=4,10
 *   ./test_sector_l1_reuse
 *
 * fapp L1 events:
 *   fapp -C -d prof_l1 -Icpupa \
 *     -Hevent_raw=0x0003,0x0004,0x0011,0x0240,0x0241,0x02a0,0x02a1,0x0200,method=fast,mode=user \
 *     ./test_sector_l1_reuse_fapp
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <arm_sve.h>

#ifdef USE_FAPP
#include <fj_tool/fapp.h>
#endif

/* Sector cache tag values (bits [59:56] of virtual address) */
#define TAG_NORMAL    0x0ULL
#define TAG_SECTOR0_S 0x9ULL  /* Strong sector 0 (reuse) */
#define TAG_SECTOR1_S 0xAULL  /* Strong sector 1 (streaming) */
#define TAG_BYPASS_L1 0xBULL  /* Bypass L1 */

#define APPLY_TAG(ptr, tag) ((void*)((uint64_t)(ptr) | ((tag) << 56)))

#define FORCE_PTR(ptr) do { \
    void* volatile _tmp = (ptr); \
    (ptr) = _tmp; \
} while(0)

/* Cache parameters */
#define L1_LINE      256           /* A64FX L1 cache line = 256 bytes */
#define L1_SIZE      (64 * 1024)   /* 64KB L1D */

/* Test parameters */
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

static inline void memory_fence(void) {
    __asm__ volatile("dmb ish" ::: "memory");
}

static void flush_cache(void* ptr, size_t size) {
    char* p = (char*)ptr;
    for (size_t i = 0; i < size; i += L1_LINE) {
        __asm__ volatile("dc civac, %0" :: "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
}

/* ========== SCCR Register Programming ========== */

/*
 * IMP_SCCR_L1_EL0 = sys_reg(3, 3, 11, 8, 2)
 * Encoding: sector0_ways | (sector1_ways << 16)
 *
 * This is an EL0 register (user-space accessible IF the kernel enables it
 * via IMP_SCTLR_EL1.L1SECTORE).
 */

static volatile sig_atomic_t sccr_fault = 0;
static sigjmp_buf sccr_jmpbuf;

static void sccr_sigill_handler(int sig) {
    sccr_fault = 1;
    siglongjmp(sccr_jmpbuf, 1);
}

static inline uint64_t read_l1_sccr(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, S3_3_C11_C8_2" : "=r"(val));
    return val;
}

static inline void write_l1_sccr(uint32_t s0_ways, uint32_t s1_ways) {
    uint64_t val = (uint64_t)s0_ways | ((uint64_t)s1_ways << 16);
    __asm__ volatile("msr S3_3_C11_C8_2, %0" :: "r"(val));
    __asm__ volatile("isb" ::: "memory");
}

/*
 * Try to program L1 SCCR. Returns 1 on success, 0 on SIGILL trap.
 */
static int try_set_l1_sccr(uint32_t s0_ways, uint32_t s1_ways) {
    struct sigaction sa, old_sa;
    sa.sa_handler = sccr_sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &old_sa);

    sccr_fault = 0;
    if (sigsetjmp(sccr_jmpbuf, 1) == 0) {
        write_l1_sccr(s0_ways, s1_ways);
    }

    sigaction(SIGILL, &old_sa, NULL);
    return !sccr_fault;
}

static int try_read_l1_sccr(uint64_t *val_out) {
    struct sigaction sa, old_sa;
    sa.sa_handler = sccr_sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &old_sa);

    sccr_fault = 0;
    if (sigsetjmp(sccr_jmpbuf, 1) == 0) {
        *val_out = read_l1_sccr();
    }

    sigaction(SIGILL, &old_sa, NULL);
    return !sccr_fault;
}

/* ========== Mixed Workload Kernel ========== */

/*
 * Mixed workload: load reuse → stream cold → reload reuse.
 *
 * MUST be noinline so the compiler can't see that tagged and raw pointers
 * alias the same memory and strip the tag bits.
 */
static __attribute__((noinline)) uint64_t run_mixed_pattern(
    float* reuse_raw, float* reuse_tagged,
    float* stream_raw, float* stream_tagged,
    size_t reuse_n, size_t stream_n,
    int iters, const char* region_name)
{
    volatile float sink = 0.0f;
    svbool_t pg = svptrue_b32();
    uint64_t start, end;

    /* Compiler barriers: keep tagged pointers alive */
    __asm__ volatile("" : "+r"(reuse_tagged));
    __asm__ volatile("" : "+r"(stream_tagged));

    /* Flush everything so all variants start cold */
    flush_cache(reuse_raw, reuse_n * sizeof(float));
    flush_cache(stream_raw, stream_n * sizeof(float));
    memory_fence();

#ifdef USE_FAPP
    fapp_start(region_name, 1, 0);
#endif

    start = read_cycle();

    for (int iter = 0; iter < iters; iter++) {
        svfloat32_t vsum = svdup_f32(0.0f);

        /* Step 1: Load reuse data into L1 */
        for (size_t i = 0; i < reuse_n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &reuse_tagged[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }

        /* Step 2: Stream through cold data (eviction pressure) */
        for (size_t i = 0; i < stream_n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &stream_tagged[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }

        /* Step 3: Reload reuse data -- hits L1 if sector hints worked */
        for (size_t i = 0; i < reuse_n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &reuse_tagged[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }

        sink += svaddv_f32(pg, vsum);
    }

    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop(region_name, 1, 0);
#endif

    memory_fence();
    (void)sink;
    return end - start;
}

int main(int argc, char* argv[]) {
    printf("=== A64FX Sector Cache L1 Reuse Test ===\n\n");

    uint64_t freq = get_freq();
    size_t reuse_n = REUSE_SIZE / sizeof(float);
    size_t stream_n = STREAM_SIZE / sizeof(float);

    printf("Timer frequency : %lu Hz\n", freq);
    printf("Reuse data      : %d KB (fits in one sector = 32KB)\n",
           REUSE_SIZE / 1024);
    printf("Stream data     : %d KB (>> L1 = 64KB)\n",
           STREAM_SIZE / 1024);
    printf("Iterations      : %d\n", NUM_ITERS);
    printf("SVE vector len  : %lu bits\n", svcntb() * 8);
    printf("\n");

    /* Check FLIB environment variables */
    printf("=== FLIB Environment ===\n");
    const char* env_sccr = getenv("FLIB_SCCR_CNTL");
    const char* env_l1   = getenv("FLIB_L1_SCCR_CNTL");
    const char* env_l2   = getenv("FLIB_L2_SECTOR_NWAYS_INIT");
    printf("FLIB_SCCR_CNTL          = %s (default=TRUE)\n",  env_sccr ? env_sccr : "(not set)");
    printf("FLIB_L1_SCCR_CNTL       = %s (default=TRUE)\n",  env_l1   ? env_l1   : "(not set)");
    printf("FLIB_L2_SECTOR_NWAYS_INIT = %s\n", env_l2 ? env_l2 : "(not set)");
    printf("\n");

    /* Try reading current L1 SCCR value */
    printf("=== SCCR Register Probe ===\n");
    uint64_t sccr_val = 0;
    int sccr_readable = try_read_l1_sccr(&sccr_val);
    if (sccr_readable) {
        printf("IMP_SCCR_L1_EL0 read OK: 0x%016lx\n", sccr_val);
        printf("  sector0_ways = %lu, sector1_ways = %lu\n",
               sccr_val & 0xFFFF, (sccr_val >> 16) & 0xFFFF);
    } else {
        printf("IMP_SCCR_L1_EL0 read FAILED (SIGILL) -> kernel does not expose SCCR at EL0\n");
    }

    /* Try writing L1 SCCR */
    int sccr_writable = 0;
    if (sccr_readable) {
        sccr_writable = try_set_l1_sccr(2, 2);  /* test: 2/2 split */
        if (sccr_writable) {
            uint64_t readback = 0;
            try_read_l1_sccr(&readback);
            printf("IMP_SCCR_L1_EL0 write OK: wrote 0x00020002, readback=0x%016lx\n", readback);
            if ((readback & 0xFFFF) == 2 && ((readback >> 16) & 0xFFFF) == 2) {
                printf("  SCCR programming CONFIRMED (readback matches)\n");
            } else {
                printf("  SCCR write accepted but readback differs (hw may ignore)\n");
            }
            /* Reset to default */
            try_set_l1_sccr(4, 0);
        } else {
            printf("IMP_SCCR_L1_EL0 write FAILED (SIGILL) -> SCCR is read-only at EL0\n");
        }
    }
    printf("\n");

    printf("Pattern per iteration:\n");
    printf("  Step 1: Load reuse (24KB) into L1\n");
    printf("  Step 2: Stream cold (256KB) through L1  [eviction pressure]\n");
    printf("  Step 3: Reload reuse (24KB)  [L1 hit or miss?]\n");
    printf("\n");

    /* Allocate aligned buffers */
    float* reuse_data = NULL;
    float* stream_data = NULL;
    posix_memalign((void**)&reuse_data, L1_LINE, REUSE_SIZE);
    posix_memalign((void**)&stream_data, L1_LINE, STREAM_SIZE);

    if (!reuse_data || !stream_data) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Initialize */
    for (size_t i = 0; i < reuse_n; i++)
        reuse_data[i] = (float)(i % 100) / 10.0f;
    for (size_t i = 0; i < stream_n; i++)
        stream_data[i] = (float)(i % 100) / 10.0f;

    /* Create tagged pointers */
    float* reuse_s0  = (float*)APPLY_TAG(reuse_data,  TAG_SECTOR0_S);
    float* stream_s1 = (float*)APPLY_TAG(stream_data, TAG_SECTOR1_S);
    float* stream_bp = (float*)APPLY_TAG(stream_data, TAG_BYPASS_L1);
    FORCE_PTR(reuse_s0);
    FORCE_PTR(stream_s1);
    FORCE_PTR(stream_bp);

    /* Pointer verification */
    printf("=== Pointer Tag Verification ===\n");
    printf("reuse_data   = 0x%016lx  tag=0x%lX\n",
           (uint64_t)reuse_data, ((uint64_t)reuse_data >> 56) & 0xF);
    printf("reuse_s0     = 0x%016lx  tag=0x%lX  (sector 0 strong)\n",
           (uint64_t)reuse_s0, ((uint64_t)reuse_s0 >> 56) & 0xF);
    printf("stream_data  = 0x%016lx  tag=0x%lX\n",
           (uint64_t)stream_data, ((uint64_t)stream_data >> 56) & 0xF);
    printf("stream_s1    = 0x%016lx  tag=0x%lX  (sector 1 strong)\n",
           (uint64_t)stream_s1, ((uint64_t)stream_s1 >> 56) & 0xF);
    printf("stream_bp    = 0x%016lx  tag=0x%lX  (L1 bypass)\n",
           (uint64_t)stream_bp, ((uint64_t)stream_bp >> 56) & 0xF);
    printf("\n");

    /* Warmup */
    run_mixed_pattern(reuse_data, reuse_data, stream_data, stream_data,
                      reuse_n, stream_n, 3, "warmup");

    /* --- Test 1: No hints --- */
    uint64_t cyc_nohint = run_mixed_pattern(
        reuse_data, reuse_data,    /* reuse: tag 0x0 */
        stream_data, stream_data,  /* stream: tag 0x0 */
        reuse_n, stream_n, NUM_ITERS, "nohint");

    /* --- Test 2: Strong sector hints (0x9/0xA) + SCCR(2,2) --- */
    if (sccr_writable) {
        try_set_l1_sccr(2, 2);  /* L1: 2 ways sector 0, 2 ways sector 1 */
        printf("  [SCCR set to (2,2) for sector test]\n");
    }
    uint64_t cyc_sector = run_mixed_pattern(
        reuse_data, reuse_s0,      /* reuse: tag 0x9 (sector 0) */
        stream_data, stream_s1,    /* stream: tag 0xA (sector 1) */
        reuse_n, stream_n, NUM_ITERS, "sector");
    if (sccr_writable) {
        try_set_l1_sccr(4, 0);  /* reset to default */
        printf("  [SCCR reset to (4,0)]\n");
    }

    /* --- Test 3: L1 bypass for streaming --- */
    uint64_t cyc_bypass = run_mixed_pattern(
        reuse_data, reuse_data,    /* reuse: tag 0x0 (normal) */
        stream_data, stream_bp,    /* stream: tag 0xB (bypass L1) */
        reuse_n, stream_n, NUM_ITERS, "bypass");

    /* --- Results --- */
    double total_bytes = (double)(REUSE_SIZE * 2 + STREAM_SIZE) * NUM_ITERS;

    printf("=== Cycle Results ===\n");
    printf("  nohint : %10lu cycles (%6.2f GB/s) [baseline]\n",
           cyc_nohint, total_bytes / ((double)cyc_nohint / freq) / 1e9);
    printf("  sector : %10lu cycles (%6.2f GB/s) speedup=%.2fx\n",
           cyc_sector, total_bytes / ((double)cyc_sector / freq) / 1e9,
           (double)cyc_nohint / cyc_sector);
    printf("  bypass : %10lu cycles (%6.2f GB/s) speedup=%.2fx\n",
           cyc_bypass, total_bytes / ((double)cyc_bypass / freq) / 1e9,
           (double)cyc_nohint / cyc_bypass);
    printf("\n");

    /* Expected L1 miss analysis */
    int reuse_lines = REUSE_SIZE / L1_LINE;
    printf("=== Expected L1 Analysis ===\n");
    printf("Reuse data = %d cache lines (24KB / 256B)\n", reuse_lines);
    printf("\n");
    printf("Per iteration:\n");
    printf("  nohint: step 2 evicts reuse data from L1\n");
    printf("          step 3 -> ~%d L1 misses (reload from L2)\n", reuse_lines);
    printf("  sector: step 2 goes to sector 1, reuse stays in sector 0\n");
    printf("          step 3 -> ~0 L1 misses (still in L1!)\n");
    printf("  bypass: step 2 bypasses L1, reuse data untouched\n");
    printf("          step 3 -> ~0 L1 misses (still in L1!)\n");
    printf("\n");
    printf("Over %d iters: nohint should have ~%d extra L1 refills\n",
           NUM_ITERS, reuse_lines * NUM_ITERS);
    printf("\n");

    /* Interpretation */
    double speedup_sector = (double)cyc_nohint / cyc_sector;
    double speedup_bypass = (double)cyc_nohint / cyc_bypass;
    printf("=== Interpretation ===\n");
    printf("SCCR writable: %s\n", sccr_writable ? "YES" : "NO");
    if (sccr_writable) {
        if (speedup_sector > 1.05) {
            printf("PASS: sector+SCCR show %.2fx speedup -> L1 partitioning WORKS\n",
                   speedup_sector);
        } else {
            printf("FAIL: sector+SCCR show only %.2fx speedup\n", speedup_sector);
            printf("  SCCR register accepted writes but hardware ignores partitioning\n");
            printf("  IMP_SCTLR_EL1.L1SECTORE may not be enabled in kernel\n");
        }
    } else {
        if (speedup_sector > 1.05) {
            printf("PASS: sector tags alone show %.2fx speedup -> tags affect replacement\n",
                   speedup_sector);
        } else {
            printf("EXPECTED: sector tags without SCCR show %.2fx (no partitioning)\n",
                   speedup_sector);
        }
    }
    if (speedup_bypass > 1.05) {
        printf("PASS: L1 bypass shows %.2fx speedup -> streaming doesn't pollute L1\n",
               speedup_bypass);
    } else {
        printf("INCONCLUSIVE: L1 bypass shows only %.2fx speedup\n",
               speedup_bypass);
    }

    printf("\nRun with fapp to see L1D_CACHE_REFILL + demand miss counts:\n");
    printf("  export FLIB_SCCR_CNTL=TRUE\n");
    printf("  export FLIB_L1_SCCR_CNTL=TRUE\n");
    printf("  make l1_reuse_fapp\n");
    printf("  fapp -C -d prof_l1 -Icpupa \\\n");
    printf("    -Hevent_raw=0x0003,0x0004,0x0011,0x0240,0x0241,0x02a0,0x02a1,0x0200,"
           "method=fast,mode=user \\\n");
    printf("    ./test_sector_l1_reuse_fapp\n");
    printf("  fapppx -A -Icpupa -tcsv -o prof_l1.csv -d prof_l1\n");
    printf("\n");

    free(reuse_data);
    free(stream_data);
    return 0;
}
