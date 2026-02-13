/*
 * A64FX Hardware Barrier Test
 *
 * Measures OpenMP barrier latency to verify hardware barrier is used.
 * Fujitsu OpenMP runtime (-Nfjomplib) uses HW barrier by default.
 * HW barrier latency should be significantly lower than SW barrier.
 *
 * Build:
 *   fcc -Nnoclang -O2 -Kopenmp -Nfjomplib -o test_hwbarrier test_hwbarrier.c
 *
 * Environment:
 *   export FLIB_BARRIER=HARD   (hardware barrier - default with -Nfjomplib)
 *   export FLIB_BARRIER=SOFT   (software barrier - for comparison)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

/* Timer: cntvct_el0 always accessible from EL0 */
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

/*
 * Test 1: Pure barrier latency
 * Each thread executes N barriers, measure total time.
 */
static void test_barrier_latency(int nbarriers) {
    int nthreads;
    double t0, t1;
    uint64_t c0, c1;

    printf("=== Test 1: Pure Barrier Latency ===\n");
    printf("  Barriers: %d\n", nbarriers);

    /* Warm up */
    #pragma omp parallel
    {
        for (int i = 0; i < 100; i++) {
            #pragma omp barrier
        }
    }

    t0 = omp_get_wtime();
    c0 = rdtsc();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();

        for (int i = 0; i < nbarriers; i++) {
            #pragma omp barrier
        }
    }

    c1 = rdtsc();
    t1 = omp_get_wtime();

    uint64_t freq = rdfreq();
    double elapsed_s = t1 - t0;
    uint64_t ticks = c1 - c0;
    double ns_per_barrier = elapsed_s * 1e9 / nbarriers;
    double ticks_per_barrier = (double)ticks / nbarriers;

    printf("  Threads:  %d\n", nthreads);
    printf("  Total:    %.6f s (%lu ticks @ %lu Hz)\n", elapsed_s, ticks, freq);
    printf("  Per barrier: %.1f ns  (%.1f ticks)\n\n", ns_per_barrier, ticks_per_barrier);
}

/*
 * Test 2: Barrier with work (producer-consumer pattern)
 * Each thread does some computation between barriers.
 * This tests that HW barrier doesn't just optimize empty barriers.
 */
static void test_barrier_with_work(int nbarriers, int work_size) {
    int nthreads;

    printf("=== Test 2: Barrier + Work (ping-pong) ===\n");
    printf("  Barriers: %d, Work per phase: %d fp adds\n", nbarriers, work_size);

    volatile double *shared = NULL;

    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }

    shared = (volatile double*)calloc(nthreads, sizeof(double));

    /* Warm up */
    #pragma omp parallel
    {
        for (int i = 0; i < 100; i++) {
            #pragma omp barrier
        }
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        double local = (double)(tid + 1);

        for (int i = 0; i < nbarriers; i++) {
            /* Phase 1: each thread writes */
            for (int w = 0; w < work_size; w++) {
                local += 0.001;
            }
            shared[tid] = local;

            #pragma omp barrier

            /* Phase 2: each thread reads neighbor's result */
            int neighbor = (tid + 1) % nt;
            local += shared[neighbor] * 0.0001;

            #pragma omp barrier
        }

        shared[tid] = local;
    }

    double t1 = omp_get_wtime();
    double elapsed_s = t1 - t0;
    double ns_per_barrier = elapsed_s * 1e9 / (2 * nbarriers); /* 2 barriers per iteration */

    printf("  Threads:  %d\n", nthreads);
    printf("  Total:    %.6f s\n", elapsed_s);
    printf("  Per barrier: %.1f ns\n\n", ns_per_barrier);

    free((void*)shared);
}

/*
 * Test 3: Many rapid barriers (stress test)
 * Tests sustained barrier throughput.
 */
static void test_rapid_barriers(int nbarriers) {
    int nthreads;

    printf("=== Test 3: Rapid Barrier Stress Test ===\n");
    printf("  Barriers: %d\n", nbarriers);

    /* Warm up */
    #pragma omp parallel
    {
        for (int i = 0; i < 1000; i++) {
            #pragma omp barrier
        }
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        volatile int dummy = 0;

        for (int i = 0; i < nbarriers; i++) {
            dummy += tid;
            #pragma omp barrier
            dummy -= tid;
            #pragma omp barrier
        }
        (void)dummy;
    }

    double t1 = omp_get_wtime();
    double ns = (t1 - t0) * 1e9 / (2 * nbarriers);

    printf("  Threads:  %d\n", nthreads);
    printf("  Total:    %.6f s (%d double-barriers)\n", t1 - t0, nbarriers);
    printf("  Per barrier: %.1f ns\n\n", ns);
}

/*
 * Test 4: Critical section vs barrier
 * Compares barrier-based sync with critical section.
 */
static void test_barrier_vs_critical(int niters) {
    int nthreads;
    double sum_barrier = 0.0, sum_critical = 0.0;

    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }

    printf("=== Test 4: Barrier vs Critical (%d threads) ===\n", nthreads);
    printf("  Iterations: %d\n\n", niters);

    /* Barrier-based reduction */
    double *partial = (double*)calloc(nthreads, sizeof(double));

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double local = 0.0;
        for (int i = 0; i < niters; i++) {
            local += 1.0;
            #pragma omp barrier
        }
        partial[tid] = local;
    }
    double t1 = omp_get_wtime();
    for (int i = 0; i < nthreads; i++) sum_barrier += partial[i];
    double ns_barrier = (t1 - t0) * 1e9 / niters;

    /* Critical-based sync */
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        double local = 0.0;
        for (int i = 0; i < niters; i++) {
            local += 1.0;
            #pragma omp critical
            {
                sum_critical += local;
                local = 0.0;
            }
        }
    }
    t1 = omp_get_wtime();
    double ns_critical = (t1 - t0) * 1e9 / niters;

    printf("  %-20s  %10s  %10s\n", "Method", "ns/iter", "Ratio");
    printf("  %-20s  %10s  %10s\n", "------", "-------", "-----");
    printf("  %-20s  %10.1f  %9.2fx\n", "barrier", ns_barrier, 1.0);
    printf("  %-20s  %10.1f  %9.2fx\n", "critical", ns_critical, ns_critical / ns_barrier);
    printf("  (sum_barrier=%.0f, sum_critical=%.0f)\n\n", sum_barrier, sum_critical);

    free(partial);
}

int main(int argc, char *argv[]) {
    int nbarriers = 100000;

    if (argc > 1) nbarriers = atoi(argv[1]);

    printf("========================================\n");
    printf(" A64FX OpenMP Barrier Performance Test\n");
    printf("========================================\n\n");

    /* Check FLIB_BARRIER env var */
    const char *barrier_env = getenv("FLIB_BARRIER");
    printf("FLIB_BARRIER = %s\n", barrier_env ? barrier_env : "(not set, default=HARD with -Nfjomplib)");
    printf("OMP_NUM_THREADS = %s\n", getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "(not set)");
    printf("Max threads: %d\n\n", omp_get_max_threads());

    test_barrier_latency(nbarriers);
    test_barrier_with_work(nbarriers / 10, 100);
    test_rapid_barriers(nbarriers);
    test_barrier_vs_critical(nbarriers / 10);

    printf("========================================\n");
    printf(" Done.\n");
    printf("========================================\n");

    return 0;
}
