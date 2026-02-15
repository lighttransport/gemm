/**
 * test_sector_cache.c - Standalone Sector Cache Test for A64FX
 *
 * Tests sector cache way partitioning using FCC OCL (Optimization Control Lines).
 * Must be compiled with FCC traditional mode (no -Nclang) since OCL pragmas
 * are only supported in traditional mode.
 *
 * Compile:
 *   fcc -Nnoclang -O2 -Kocl,hpctag -S -o test_sector_cache.s test_sector_cache.c
 *   fcc -Nnoclang -O2 -Kocl,hpctag    -o test_sector_cache   test_sector_cache.c
 *
 * OCL Sector Cache Directives (C syntax):
 *
 * Method 1: L1-only partitioning (#pragma statement)
 *   #pragma statement cache_sector_size(sector0_ways, sector1_ways)
 *   #pragma statement cache_subsector_assign(array_ptr)
 *   ...loop body...
 *   #pragma statement end_cache_subsector
 *   #pragma statement end_cache_sector_size
 *
 * Method 2: L1+L2 partitioning (#pragma procedure)
 *   #pragma procedure scache_isolate_way L2=N2 [L1=N1]
 *   #pragma procedure scache_isolate_assign ptr1 [,ptr2,...]
 *   ...function body...
 *
 * How it works:
 *   1. cache_sector_size(S0, S1) partitions L1D 4 ways into S0+S1=4
 *      scache_isolate_way L2=N L1=M partitions L1(4-way) and L2(16-way)
 *      - Sector 0 (default): untagged data (B, C in GEMM)
 *      - Sector 1 (tagged):  tagged data (streaming A in GEMM)
 *   2. cache_subsector_assign(ptr) / scache_isolate_assign marks loads
 *      - Compiler emits: ORR x_tagged, x_ptr, #0x0100000000000000
 *      - Bit 56 = sector 1 indicator in A64FX HPC tag address override
 *   3. Runtime calls __jwe_xset_sccr to program the sector cache control
 *      register (SCCR) with the way partition configuration
 *
 * SCCR struct layout:
 *   [+0] command: 256=start sector, 512=end sector
 *   [+4] L1 config: sector0_ways | (sector1_ways << 16)  (S0+S1=4)
 *   [+8] L2 config: sector0_ways | (sector1_ways << 16)
 *
 * A64FX Sector Cache Architecture:
 *   - L1D: 64KB, 4-way set-associative, 256B line
 *   - L2:  8MB, 16-way, shared per CMG (12-13 cores)
 *   - Sector cache splits ways into 2 sectors
 *   - L1 has 4 flat sectors; L2 has 2 groups of 2 sectors
 *   - FCC pragmas control 2 sectors (sector 0 = default, sector 1 = tagged)
 *   - Requires IMP_SCTLR_EL1.L1SECTORE to be enabled in kernel
 *
 * Reference: https://epub.ub.uni-muenchen.de/126883/1/3624062.3624198.pdf
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* A64FX constants */
#define L1_SIZE     (64 * 1024)       /* 64KB L1D */
#define L1_WAYS     4                 /* 4-way set-associative */
#define CACHE_LINE  256               /* 256B cache line */
#define L1_SETS     (L1_SIZE / (L1_WAYS * CACHE_LINE))  /* = 64 sets */

/* Timer: cntvct_el0 at 100 MHz, CPU at 2 GHz */
#define TIMER_FREQ  100
#define CPU_FREQ    2000

static inline uint64_t rdtimer(void) {
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

static inline void barrier(void) {
    __asm__ __volatile__("dsb sy\n\tisb" ::: "memory");
}

static inline uint64_t ticks_to_cycles(uint64_t ticks) {
    return ticks * (CPU_FREQ / TIMER_FREQ);
}

/* ================================================================
 * Test 1: Sector cache with simple streaming + resident pattern
 *
 * Sector 0 (3 ways = 48KB): B array (resident, reuse)
 * Sector 1 (1 way  = 16KB): A array (streaming, no reuse)
 *
 * Without sector cache, A streaming would evict B from cache.
 * With sector cache, A is confined to 1 way and B stays resident.
 * ================================================================ */
void test_sector_stream_vs_resident(double *A, double *B, double *C, int nA, int nB) {
    int i, iter;
    volatile double sum;

    /* Warm B into cache */
    sum = 0.0;
    for (i = 0; i < nB; i++) {
        sum += B[i];
    }

    /* Now stream through A while repeatedly accessing B */
    /* With sector cache: A confined to 1 way, B stays in 3 ways */
#pragma statement cache_sector_size(3,1)
#pragma statement cache_subsector_assign(A)
    for (iter = 0; iter < 10; iter++) {
        for (i = 0; i < nA; i++) {
            sum += A[i];
        }
        for (i = 0; i < nB; i++) {
            sum += B[i];
        }
    }
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size

    C[0] = sum;
}

/* Same test WITHOUT sector cache (baseline comparison) */
void test_nosector_stream_vs_resident(double *A, double *B, double *C, int nA, int nB) {
    int i, iter;
    volatile double sum;

    sum = 0.0;
    for (i = 0; i < nB; i++) {
        sum += B[i];
    }

    for (iter = 0; iter < 10; iter++) {
        for (i = 0; i < nA; i++) {
            sum += A[i];
        }
        for (i = 0; i < nB; i++) {
            sum += B[i];
        }
    }

    C[0] = sum;
}

/* ================================================================
 * Test 2: GEMM-like pattern with sector cache
 *
 * GEMM: C[M,N] += A[M,K] * B[K,N]
 * - A streams through K dimension → sector 1 (1 way, 16KB)
 * - B reused across M rows        → sector 0 (3 ways, 48KB)
 * - C accumulated                  → sector 0 (shared with B)
 * ================================================================ */
void test_gemm_sector(double *A, double *B, double *C, int M, int N, int K) {
    int i, j, k;

#pragma statement cache_sector_size(3,1)
#pragma statement cache_subsector_assign(A)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += sum;
        }
    }
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size
}

/* GEMM without sector cache (baseline) */
void test_gemm_nosector(double *A, double *B, double *C, int M, int N, int K) {
    int i, j, k;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += sum;
        }
    }
}

/* ================================================================
 * Test 3: Different way partition configurations
 *
 * Test (1,3), (2,2), (3,1) to see if sector cache is active
 * ================================================================ */
void test_sector_config_1_3(double *A, double *B, double *C, int n) {
    int i, iter;
    volatile double sum = 0.0;

#pragma statement cache_sector_size(1,3)
#pragma statement cache_subsector_assign(A)
    for (iter = 0; iter < 100; iter++) {
        for (i = 0; i < n; i++) {
            sum += A[i] + B[i];
        }
    }
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size

    C[0] = sum;
}

void test_sector_config_2_2(double *A, double *B, double *C, int n) {
    int i, iter;
    volatile double sum = 0.0;

#pragma statement cache_sector_size(2,2)
#pragma statement cache_subsector_assign(A)
    for (iter = 0; iter < 100; iter++) {
        for (i = 0; i < n; i++) {
            sum += A[i] + B[i];
        }
    }
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size

    C[0] = sum;
}

void test_sector_config_3_1(double *A, double *B, double *C, int n) {
    int i, iter;
    volatile double sum = 0.0;

#pragma statement cache_sector_size(3,1)
#pragma statement cache_subsector_assign(A)
    for (iter = 0; iter < 100; iter++) {
        for (i = 0; i < n; i++) {
            sum += A[i] + B[i];
        }
    }
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size

    C[0] = sum;
}

/* ================================================================
 * Test 4: L2 sector cache using scache_isolate_way
 *
 * #pragma procedure scache_isolate_way L2=N [L1=M]
 * #pragma procedure scache_isolate_assign ptr
 *
 * L2: 8MB, 16-way. Split: sector1=N ways (streaming), sector0=rest (reuse)
 * L1: 64KB, 4-way.  Split: sector1=M ways (streaming), sector0=rest (reuse)
 * ================================================================ */
void test_l2_sector(double *A, double *B, double *C, int nA, int nB) {
#pragma procedure scache_isolate_way L2=5 L1=1
#pragma procedure scache_isolate_assign A
    int i, iter;
    volatile double sum = 0.0;

    /* Warm B into cache */
    for (i = 0; i < nB; i++) {
        sum += B[i];
    }

    /* Stream through A (large) while accessing B (resident) */
    /* With L2 sector: A confined to 5 L2 ways, B gets 11 ways */
    for (iter = 0; iter < 5; iter++) {
        for (i = 0; i < nA; i++) {
            sum += A[i];
        }
        for (i = 0; i < nB; i++) {
            sum += B[i];
        }
    }

    C[0] = sum;
}

void test_l2_nosector(double *A, double *B, double *C, int nA, int nB) {
    int i, iter;
    volatile double sum = 0.0;

    for (i = 0; i < nB; i++) {
        sum += B[i];
    }

    for (iter = 0; iter < 5; iter++) {
        for (i = 0; i < nA; i++) {
            sum += A[i];
        }
        for (i = 0; i < nB; i++) {
            sum += B[i];
        }
    }

    C[0] = sum;
}

/* ================================================================
 * Benchmark driver
 * ================================================================ */

static void fill_random(double *buf, int n) {
    int i;
    for (i = 0; i < n; i++) {
        buf[i] = (double)(i % 1000) * 0.001;
    }
}

static void run_l2_sector_test(void);

static void run_streaming_test(void) {
    /* A = 128KB (streaming, larger than L1) */
    /* B = 32KB (resident, fits in 2 ways of L1) */
    int nA = 128 * 1024 / sizeof(double);  /* 16384 doubles */
    int nB = 32 * 1024 / sizeof(double);   /* 4096 doubles */
    double *A, *B, *C;
    uint64_t t0, t1;
    double cy_sector, cy_nosector;
    int warmup;

    A = (double *)aligned_alloc(256, nA * sizeof(double));
    B = (double *)aligned_alloc(256, nB * sizeof(double));
    C = (double *)aligned_alloc(256, sizeof(double));

    fill_random(A, nA);
    fill_random(B, nB);
    C[0] = 0.0;

    printf("\n=== Test 1: Streaming A (128KB) + Resident B (32KB) ===\n");
    printf("Sector config: sector0=3ways(48KB) for B, sector1=1way(16KB) for A\n\n");

    /* Warmup */
    for (warmup = 0; warmup < 3; warmup++) {
        test_nosector_stream_vs_resident(A, B, C, nA, nB);
        test_sector_stream_vs_resident(A, B, C, nA, nB);
    }

    /* Benchmark: no sector cache */
    barrier();
    t0 = rdtimer();
    barrier();
    test_nosector_stream_vs_resident(A, B, C, nA, nB);
    barrier();
    t1 = rdtimer();
    cy_nosector = (double)ticks_to_cycles(t1 - t0);

    /* Benchmark: with sector cache */
    barrier();
    t0 = rdtimer();
    barrier();
    test_sector_stream_vs_resident(A, B, C, nA, nB);
    barrier();
    t1 = rdtimer();
    cy_sector = (double)ticks_to_cycles(t1 - t0);

    printf("  No sector:   %.0f cycles\n", cy_nosector);
    printf("  With sector: %.0f cycles\n", cy_sector);
    printf("  Speedup:     %.2fx\n", cy_nosector / cy_sector);

    free(A);
    free(B);
    free(C);
}

static void run_gemm_test(void) {
    int M = 64, N = 64, K = 64;
    double *A, *B, *C1, *C2;
    uint64_t t0, t1;
    double cy_sector, cy_nosector;
    int warmup;
    double max_diff;
    int i;

    A  = (double *)aligned_alloc(256, M * K * sizeof(double));
    B  = (double *)aligned_alloc(256, K * N * sizeof(double));
    C1 = (double *)aligned_alloc(256, M * N * sizeof(double));
    C2 = (double *)aligned_alloc(256, M * N * sizeof(double));

    fill_random(A, M * K);
    fill_random(B, K * N);
    memset(C1, 0, M * N * sizeof(double));
    memset(C2, 0, M * N * sizeof(double));

    printf("\n=== Test 2: GEMM %dx%dx%d with Sector Cache ===\n", M, N, K);
    printf("A=%dKB(sector1,stream), B=%dKB(sector0,reuse), C=%dKB(sector0)\n",
           (int)(M * K * sizeof(double) / 1024),
           (int)(K * N * sizeof(double) / 1024),
           (int)(M * N * sizeof(double) / 1024));

    /* Verify correctness */
    test_gemm_nosector(A, B, C1, M, N, K);
    test_gemm_sector(A, B, C2, M, N, K);

    max_diff = 0.0;
    for (i = 0; i < M * N; i++) {
        double d = C1[i] - C2[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
    }
    printf("Correctness check: max_diff = %e %s\n\n",
           max_diff, max_diff < 1e-10 ? "(OK)" : "(MISMATCH!)");

    /* Warmup */
    for (warmup = 0; warmup < 3; warmup++) {
        memset(C1, 0, M * N * sizeof(double));
        test_gemm_nosector(A, B, C1, M, N, K);
        memset(C2, 0, M * N * sizeof(double));
        test_gemm_sector(A, B, C2, M, N, K);
    }

    /* Benchmark: no sector */
    memset(C1, 0, M * N * sizeof(double));
    barrier();
    t0 = rdtimer();
    barrier();
    test_gemm_nosector(A, B, C1, M, N, K);
    barrier();
    t1 = rdtimer();
    cy_nosector = (double)ticks_to_cycles(t1 - t0);

    /* Benchmark: with sector */
    memset(C2, 0, M * N * sizeof(double));
    barrier();
    t0 = rdtimer();
    barrier();
    test_gemm_sector(A, B, C2, M, N, K);
    barrier();
    t1 = rdtimer();
    cy_sector = (double)ticks_to_cycles(t1 - t0);

    printf("  No sector:   %.0f cycles (%.1f MFLOP/s at 2GHz)\n",
           cy_nosector, 2.0 * M * N * K / cy_nosector * 2000.0);
    printf("  With sector: %.0f cycles (%.1f MFLOP/s at 2GHz)\n",
           cy_sector, 2.0 * M * N * K / cy_sector * 2000.0);
    printf("  Speedup:     %.2fx\n", cy_nosector / cy_sector);

    free(A);
    free(B);
    free(C1);
    free(C2);
}

static void run_config_test(void) {
    /* Use 8KB data arrays - fits in 1 way of L1 (16KB/way) */
    int n = 8 * 1024 / sizeof(double);  /* 1024 doubles = 8KB */
    double *A, *B, *C;
    uint64_t t0, t1;
    double cy[3];
    int warmup;

    A = (double *)aligned_alloc(256, n * sizeof(double));
    B = (double *)aligned_alloc(256, n * sizeof(double));
    C = (double *)aligned_alloc(256, sizeof(double));

    fill_random(A, n);
    fill_random(B, n);

    printf("\n=== Test 3: Way Partition Configurations ===\n");
    printf("Data: A=%dKB, B=%dKB (both fit in L1)\n",
           (int)(n * sizeof(double) / 1024),
           (int)(n * sizeof(double) / 1024));
    printf("If sector cache is active, (3,1) should be faster than (1,3)\n");
    printf("because B gets more ways in sector 0.\n\n");

    /* Warmup all configs */
    for (warmup = 0; warmup < 3; warmup++) {
        test_sector_config_1_3(A, B, C, n);
        test_sector_config_2_2(A, B, C, n);
        test_sector_config_3_1(A, B, C, n);
    }

    /* (1,3): A gets 3 ways, B gets 1 way */
    barrier();
    t0 = rdtimer();
    barrier();
    test_sector_config_1_3(A, B, C, n);
    barrier();
    t1 = rdtimer();
    cy[0] = (double)ticks_to_cycles(t1 - t0);

    /* (2,2): balanced */
    barrier();
    t0 = rdtimer();
    barrier();
    test_sector_config_2_2(A, B, C, n);
    barrier();
    t1 = rdtimer();
    cy[1] = (double)ticks_to_cycles(t1 - t0);

    /* (3,1): A gets 1 way, B gets 3 ways */
    barrier();
    t0 = rdtimer();
    barrier();
    test_sector_config_3_1(A, B, C, n);
    barrier();
    t1 = rdtimer();
    cy[2] = (double)ticks_to_cycles(t1 - t0);

    printf("  Config (1,3): %.0f cycles  [A=3ways, B(default)=1way]\n", cy[0]);
    printf("  Config (2,2): %.0f cycles  [A=2ways, B(default)=2ways]\n", cy[1]);
    printf("  Config (3,1): %.0f cycles  [A=1way,  B(default)=3ways]\n", cy[2]);
    printf("\n");
    if (cy[0] != cy[1] || cy[1] != cy[2]) {
        printf("  Variation detected → sector cache may be active\n");
    } else {
        printf("  No variation → sector cache likely NOT active (IMP_SCTLR_EL1)\n");
    }

    free(A);
    free(B);
    free(C);
}

static void run_l2_sector_test(void) {
    /* A = 4MB (streaming, exceeds L2 per-core share) */
    /* B = 512KB (resident in L2) */
    int nA = 4 * 1024 * 1024 / sizeof(double);   /* 524288 doubles */
    int nB = 512 * 1024 / sizeof(double);          /* 65536 doubles */
    double *A, *B, *C;
    uint64_t t0, t1;
    double cy_sector, cy_nosector;
    int warmup;

    A = (double *)aligned_alloc(256, nA * sizeof(double));
    B = (double *)aligned_alloc(256, nB * sizeof(double));
    C = (double *)aligned_alloc(256, sizeof(double));

    fill_random(A, nA);
    fill_random(B, nB);

    printf("\n=== Test 4: L2 Sector Cache (scache_isolate_way) ===\n");
    printf("Pragma: #pragma procedure scache_isolate_way L2=5 L1=1\n");
    printf("A=%dMB(sector1,stream), B=%dKB(sector0,reuse)\n",
           (int)(nA * sizeof(double) / (1024*1024)),
           (int)(nB * sizeof(double) / 1024));
    printf("L2 split: sector0=11ways(~5.5MB), sector1=5ways(~2.5MB)\n");
    printf("L1 split: sector0=3ways(48KB),    sector1=1way(16KB)\n\n");

    /* Warmup */
    for (warmup = 0; warmup < 2; warmup++) {
        test_l2_nosector(A, B, C, nA, nB);
        test_l2_sector(A, B, C, nA, nB);
    }

    /* No sector */
    barrier();
    t0 = rdtimer();
    barrier();
    test_l2_nosector(A, B, C, nA, nB);
    barrier();
    t1 = rdtimer();
    cy_nosector = (double)ticks_to_cycles(t1 - t0);

    /* With sector */
    barrier();
    t0 = rdtimer();
    barrier();
    test_l2_sector(A, B, C, nA, nB);
    barrier();
    t1 = rdtimer();
    cy_sector = (double)ticks_to_cycles(t1 - t0);

    printf("  No sector:   %.0f cycles\n", cy_nosector);
    printf("  With sector: %.0f cycles\n", cy_sector);
    printf("  Speedup:     %.2fx\n", cy_nosector / cy_sector);

    free(A);
    free(B);
    free(C);
}

int main(void) {
    printf("========================================\n");
    printf("A64FX Sector Cache Test (FCC OCL Pragmas)\n");
    printf("========================================\n");
    printf("\n");
    printf("Compiler: FCC traditional mode (-Nnoclang)\n");
    printf("OCL pragma format:\n");
    printf("  #pragma statement cache_sector_size(sector0_ways, sector1_ways)\n");
    printf("  #pragma statement cache_subsector_assign(ptr)\n");
    printf("  ...loop body with ptr accesses tagged...\n");
    printf("  #pragma statement end_cache_subsector\n");
    printf("  #pragma statement end_cache_sector_size\n");
    printf("\n");
    printf("HPC tag: loads from ptr use ORR x, x, #0x0100000000000000\n");
    printf("         (bit 56 = sector 1 tag in top byte)\n");
    printf("\n");
    printf("Runtime: __jwe_xset_sccr() programs sector cache control register\n");
    printf("         SCCR config = sector0_ways | (sector1_ways << 16)\n");
    printf("\n");
    printf("L1D: 64KB, 4-way, 256B line, 64 sets\n");
    printf("  1 way = 16KB, 2 ways = 32KB, 3 ways = 48KB\n");

    run_streaming_test();
    run_gemm_test();
    run_config_test();
    run_l2_sector_test();

    printf("\n========================================\n");
    printf("Done.\n");

    return 0;
}
