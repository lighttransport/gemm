// bench_peak.c
// Peak INT16 SDOT throughput test

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t read_cycle_counter(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

static void* aligned_alloc_wrapper(size_t align, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, align, size);
    return ptr;
}

// ASM kernels
extern void int16_sdot_peak_40(
    const int16_t* A,
    const int16_t* B,
    int64_t* C,
    int64_t iterations
);

int main(int argc, char** argv) {
    uint64_t timer_freq = get_timer_freq();
    int64_t iterations = 1000000;
    int warmup = 10000;

    if (argc > 1) iterations = atoll(argv[1]);

    printf("==============================================\n");
    printf("INT16 SDOT Peak Throughput Test\n");
    printf("==============================================\n\n");

    // Allocate
    int16_t* A = (int16_t*)aligned_alloc_wrapper(64, 128);
    int16_t* B = (int16_t*)aligned_alloc_wrapper(64, 256);
    int64_t* C = (int64_t*)aligned_alloc_wrapper(64, 64);

    for (int i = 0; i < 64; i++) A[i] = i % 5;
    for (int i = 0; i < 128; i++) B[i] = i % 3;
    memset(C, 0, 64);

    // 40 SDOT per iteration
    // Each SDOT: 8 lanes × 4 MACs = 32 ops (counting MACs as ops)
    // Total: 40 × 32 = 1280 ops per iteration
    double ops_per_iter = 40.0 * 8 * 4;
    double sdot_per_iter = 40.0;

    printf("Configuration:\n");
    printf("  SDOT per iteration: 40\n");
    printf("  Ops per iteration:  %.0f (counting MACs)\n", ops_per_iter);
    printf("  Iterations:         %ld\n\n", iterations);

    // Test: 40 SDOT kernel
    printf("--- 40 SDOT per iteration ---\n");

    for (int i = 0; i < warmup; i++) {
        int16_sdot_peak_40(A, B, C, 2);
    }
    __asm__ volatile("isb" ::: "memory");

    uint64_t t0 = read_cycle_counter();
    int16_sdot_peak_40(A, B, C, iterations);
    uint64_t t1 = read_cycle_counter();

    uint64_t delta = t1 - t0;
    double elapsed = (double)delta / (double)timer_freq;
    double total_ops = ops_per_iter * (double)iterations;
    double gops = total_ops / elapsed / 1e9;
    double ns_per_iter = elapsed / (double)iterations * 1e9;
    double cycles_per_iter = ns_per_iter * 2.0;
    double sdot_per_cycle = sdot_per_iter / cycles_per_iter;

    printf("  Time:           %.3f ms\n", elapsed * 1000);
    printf("  Per iteration:  %.2f ns (%.1f cycles @ 2.0GHz)\n", ns_per_iter, cycles_per_iter);
    printf("  Throughput:     %.2f GOPS (INT16)\n", gops);
    printf("  SDOT/cycle:     %.2f\n", sdot_per_cycle);
    printf("\n");

    // Analysis
    printf("--- Analysis ---\n");
    printf("If 2 SDOT/cycle: expect %.1f cycles, peak 128 GOPS\n", sdot_per_iter / 2.0);
    printf("If 4 SDOT/cycle: expect %.1f cycles, peak 256 GOPS\n", sdot_per_iter / 4.0);
    printf("Measured:        %.1f cycles, %.1f GOPS\n", cycles_per_iter, gops);
    printf("\n");

    if (sdot_per_cycle > 3.5) {
        printf("Result: A64FX achieves ~4 INT16 SDOT/cycle!\n");
    } else if (sdot_per_cycle > 1.8) {
        printf("Result: A64FX achieves ~2 INT16 SDOT/cycle (same as INT8)\n");
    } else {
        printf("Result: Suboptimal, possible bottleneck\n");
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
