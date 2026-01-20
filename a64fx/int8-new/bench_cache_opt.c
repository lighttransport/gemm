// Benchmark cache optimization strategies for D=512
// Compare: baseline, software prefetching, K-tiling

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

// External kernel functions
extern void gemm_6row_int8_d512(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

extern void gemm_6row_int8_d512_prefetch(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

extern void gemm_6row_int8_d512_ktile(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

// Timer frequency (A64FX)
#define TIMER_FREQ 100000000ULL  // 100 MHz = 0.1 GHz

// Get cycle counter
static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Allocate aligned memory
static void* aligned_alloc_portable(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

// Initialize INT8 matrix with random values
static void init_int8_matrix(int8_t* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (int8_t)((rand() % 256) - 128);
    }
}

// Benchmark a kernel
static void bench_kernel(
    const char* name,
    void (*kernel)(const int8_t*, const int8_t*, int32_t*, int, int, int),
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N,
    int warmup_iters, int bench_iters)
{
    printf("\n=== %s ===\n", name);

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        memset(C, 0, M * N * sizeof(int32_t));
        kernel(A, B, C, M, K, N);
    }

    // Benchmark
    uint64_t total_cycles = 0;
    for (int i = 0; i < bench_iters; i++) {
        memset(C, 0, M * N * sizeof(int32_t));

        uint64_t start = get_cycles();
        kernel(A, B, C, M, K, N);
        uint64_t end = get_cycles();

        total_cycles += (end - start);
    }

    // Calculate metrics
    double avg_cycles = (double)total_cycles / bench_iters;
    double cpu_cycles = avg_cycles * 20.0;  // Timer @ 100 MHz, CPU @ 2 GHz

    // INT8 operations
    uint64_t int8_ops = 2ULL * M * K * N;  // MACs count as 2 ops
    uint64_t int8_macs = (uint64_t)M * K * N;

    // Performance metrics
    double ops_per_cycle = (double)int8_ops / cpu_cycles;
    double gops = ops_per_cycle * 2.0;  // CPU @ 2 GHz
    double gmacs = gops / 2.0;

    // Efficiency (peak = 512 GOPS = 256 INT8 ops/cycle)
    double efficiency = (ops_per_cycle / 256.0) * 100.0;

    // SDOT analysis
    int64_t k_groups = K / 4;
    int64_t n_chunks = N / 64;
    int64_t sdots_per_call = M * k_groups * n_chunks * 4;  // 4 K-elements per SDOT
    double sdots_per_cycle = (double)sdots_per_call / cpu_cycles;

    printf("Configuration: M=%d, K=%d, N=%d\n", M, K, N);
    printf("Timer cycles: %.1f\n", avg_cycles);
    printf("CPU cycles: %.1f\n", cpu_cycles);
    printf("INT8 operations: %llu (or %llu MACs)\n",
           (unsigned long long)int8_ops, (unsigned long long)int8_macs);
    printf("INT8 ops/cycle: %.2f\n", ops_per_cycle);
    printf("GOPS: %.2f\n", gops);
    printf("GMAC/s: %.2f\n", gmacs);
    printf("Efficiency: %.2f%% of peak (512 GOPS)\n", efficiency);
    printf("Ops/cycle utilization: %.2f / 256 = %.2f%%\n",
           ops_per_cycle, efficiency);
    printf("\nSDOT analysis:\n");
    printf("SDOTs per call: %lld\n", (long long)sdots_per_call);
    printf("SDOTs/cycle: %.2f\n", sdots_per_cycle);
    printf("Theoretical max: 2 SDOTs/cycle (with 2 FPUs)\n");
    printf("SDOT utilization: %.2f%%\n", (sdots_per_cycle / 2.0) * 100.0);
}

int main(void) {
    printf("================================================================\n");
    printf("Cache Optimization Benchmark - D=512 Kernels\n");
    printf("A64FX L1: 11 cycles, L2: 27-36 cycles\n");
    printf("================================================================\n");
    printf("Timer frequency: %llu Hz (%.1f GHz)\n",
           (unsigned long long)TIMER_FREQ, TIMER_FREQ / 1e9);

    printf("\nA64FX INT8 Peak Performance:\n");
    printf("- 512 GOPS (INT8 operations per second)\n");
    printf("- 256 INT8 ops/cycle at 2 GHz\n");
    printf("- 2 FPUs × 2 SDOT/cycle × 64 INT8 MACs = 256 INT8 MACs/cycle\n");
    printf("\n");

    // Configuration
    const int M = 6;
    const int K = 512;
    const int N = 2048;
    const int warmup = 10;
    const int iters = 100;

    // Allocate matrices
    int8_t* A = (int8_t*)aligned_alloc_portable(64, M * K * sizeof(int8_t));
    int8_t* B = (int8_t*)aligned_alloc_portable(64, K * N * sizeof(int8_t));
    int32_t* C = (int32_t*)aligned_alloc_portable(64, M * N * sizeof(int32_t));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize
    srand(42);
    init_int8_matrix(A, M, K);
    init_int8_matrix(B, K, N);

    // Benchmark baseline
    bench_kernel("Baseline (6-row split loading)",
                 gemm_6row_int8_d512, A, B, C, M, K, N, warmup, iters);

    // Benchmark with software prefetching
    bench_kernel("Software Prefetching (2 K-groups ahead)",
                 gemm_6row_int8_d512_prefetch, A, B, C, M, K, N, warmup, iters);

    // Benchmark with K-tiling
    bench_kernel("K-Tiling (32 K-groups per tile)",
                 gemm_6row_int8_d512_ktile, A, B, C, M, K, N, warmup, iters);

    printf("\n================================================================\n");
    printf("Cache Optimization Analysis:\n");
    printf("================================================================\n");
    printf("\nBaseline:\n");
    printf("  - Process all 128 K-groups sequentially\n");
    printf("  - 8 KB working set per iteration\n");
    printf("  - L1 hit rate: ~75%% (estimated)\n");
    printf("  - L2 accesses: ~25%%\n");
    printf("\nSoftware Prefetching:\n");
    printf("  - Prefetch 2 K-groups ahead (16 KB distance)\n");
    printf("  - Hide L2 latency (27-36 cycles) with computation\n");
    printf("  - Expected: +5-10%% efficiency improvement\n");
    printf("\nK-Tiling:\n");
    printf("  - Process K in 4 tiles of 32 K-groups\n");
    printf("  - Each tile: 256 KB working set (fits L2 better)\n");
    printf("  - Better temporal locality across tiles\n");
    printf("  - Expected: +8-12%% efficiency improvement\n");
    printf("\n================================================================\n");

    // Cleanup
    free(A);
    free(B);
    free(C);

    return 0;
}
