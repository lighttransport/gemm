// Comprehensive cache optimization benchmark for D=256 and D=512
// Compare: baseline vs prefetching for both dimensions

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

// External kernel functions - D=256
extern void kernel_ffn_6row_gemm_d256(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

extern void gemm_6row_int8_d256_prefetch(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N);

// External kernel functions - D=512
extern void kernel_ffn_6row_gemm_d512(
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

// Benchmark result structure
typedef struct {
    double timer_cycles;
    double cpu_cycles;
    double ops_per_cycle;
    double gops;
    double efficiency;
    double sdots_per_cycle;
} bench_result_t;

// Benchmark a kernel
static bench_result_t bench_kernel(
    void (*kernel)(const int8_t*, const int8_t*, int32_t*, int, int, int),
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int K, int N,
    int warmup_iters, int bench_iters)
{
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
    bench_result_t result;
    result.timer_cycles = (double)total_cycles / bench_iters;
    result.cpu_cycles = result.timer_cycles * 20.0;  // Timer @ 100 MHz, CPU @ 2 GHz

    // INT8 operations
    uint64_t int8_ops = 2ULL * M * K * N;  // MACs count as 2 ops

    // Performance metrics
    result.ops_per_cycle = (double)int8_ops / result.cpu_cycles;
    result.gops = result.ops_per_cycle * 2.0;  // CPU @ 2 GHz

    // Efficiency (peak = 512 GOPS = 256 INT8 ops/cycle)
    result.efficiency = (result.ops_per_cycle / 256.0) * 100.0;

    // SDOT analysis
    int64_t k_groups = K / 4;
    int64_t n_chunks = N / 64;
    int64_t sdots_per_call = M * k_groups * n_chunks * 4;  // 4 K-elements per SDOT
    result.sdots_per_cycle = (double)sdots_per_call / result.cpu_cycles;

    return result;
}

// Print comparison
static void print_comparison(const char* name, bench_result_t baseline, bench_result_t opt) {
    double speedup = baseline.cpu_cycles / opt.cpu_cycles;
    double eff_gain = opt.efficiency - baseline.efficiency;
    double sdot_gain = ((opt.sdots_per_cycle / baseline.sdots_per_cycle) - 1.0) * 100.0;

    printf("\n%s:\n", name);
    printf("  Baseline:  %.1f cycles, %.2f GOPS, %.2f%% eff, %.2f SDOT/cyc\n",
           baseline.cpu_cycles, baseline.gops, baseline.efficiency, baseline.sdots_per_cycle);
    printf("  Optimized: %.1f cycles, %.2f GOPS, %.2f%% eff, %.2f SDOT/cyc\n",
           opt.cpu_cycles, opt.gops, opt.efficiency, opt.sdots_per_cycle);
    printf("  Speedup:   %.3fx (%.1f%% faster)\n", speedup, (speedup - 1.0) * 100.0);
    printf("  Eff gain:  %+.2f%% (%.2f%% -> %.2f%%)\n",
           eff_gain, baseline.efficiency, opt.efficiency);
    printf("  SDOT gain: %+.2f%%\n", sdot_gain);
}

int main(void) {
    printf("================================================================\n");
    printf("Comprehensive Cache Optimization Benchmark\n");
    printf("A64FX L1: 11 cycles, L2: 27-36 cycles\n");
    printf("================================================================\n");
    printf("Timer frequency: %llu Hz (%.1f GHz)\n",
           (unsigned long long)TIMER_FREQ, TIMER_FREQ / 1e9);

    printf("\nA64FX INT8 Peak Performance:\n");
    printf("- 512 GOPS (INT8 operations per second)\n");
    printf("- 256 INT8 ops/cycle at 2 GHz\n");
    printf("- 2 FPUs × 2 SDOT/cycle × 64 INT8 MACs = 256 INT8 MACs/cycle\n");
    printf("\n");

    const int warmup = 10;
    const int iters = 100;

    // ==================== D=256 Benchmarks ====================
    printf("================================================================\n");
    printf("D=256 Benchmarks (M=6, K=256, N=1024)\n");
    printf("================================================================\n");

    const int M256 = 6;
    const int K256 = 256;
    const int N256 = 1024;

    // Allocate D=256 matrices
    int8_t* A256 = (int8_t*)aligned_alloc_portable(64, M256 * K256 * sizeof(int8_t));
    int8_t* B256 = (int8_t*)aligned_alloc_portable(64, K256 * N256 * sizeof(int8_t));
    int32_t* C256 = (int32_t*)aligned_alloc_portable(64, M256 * N256 * sizeof(int32_t));

    if (!A256 || !B256 || !C256) {
        fprintf(stderr, "D=256 memory allocation failed\n");
        return 1;
    }

    srand(42);
    init_int8_matrix(A256, M256, K256);
    init_int8_matrix(B256, K256, N256);

    printf("\nBenchmarking D=256 kernels...\n");
    bench_result_t d256_baseline = bench_kernel(kernel_ffn_6row_gemm_d256,
        A256, B256, C256, M256, K256, N256, warmup, iters);

    bench_result_t d256_prefetch = bench_kernel(gemm_6row_int8_d256_prefetch,
        A256, B256, C256, M256, K256, N256, warmup, iters);

    print_comparison("D=256: Baseline vs Prefetching", d256_baseline, d256_prefetch);

    // ==================== D=512 Benchmarks ====================
    printf("\n================================================================\n");
    printf("D=512 Benchmarks (M=6, K=512, N=2048)\n");
    printf("================================================================\n");

    const int M512 = 6;
    const int K512 = 512;
    const int N512 = 2048;

    // Allocate D=512 matrices
    int8_t* A512 = (int8_t*)aligned_alloc_portable(64, M512 * K512 * sizeof(int8_t));
    int8_t* B512 = (int8_t*)aligned_alloc_portable(64, K512 * N512 * sizeof(int8_t));
    int32_t* C512 = (int32_t*)aligned_alloc_portable(64, M512 * N512 * sizeof(int32_t));

    if (!A512 || !B512 || !C512) {
        fprintf(stderr, "D=512 memory allocation failed\n");
        free(A256); free(B256); free(C256);
        return 1;
    }

    srand(42);
    init_int8_matrix(A512, M512, K512);
    init_int8_matrix(B512, K512, N512);

    printf("\nBenchmarking D=512 kernels...\n");
    bench_result_t d512_baseline = bench_kernel(kernel_ffn_6row_gemm_d512,
        A512, B512, C512, M512, K512, N512, warmup, iters);

    bench_result_t d512_prefetch = bench_kernel(gemm_6row_int8_d512_prefetch,
        A512, B512, C512, M512, K512, N512, warmup, iters);

    bench_result_t d512_ktile = bench_kernel(gemm_6row_int8_d512_ktile,
        A512, B512, C512, M512, K512, N512, warmup, iters);

    print_comparison("D=512: Baseline vs Prefetching", d512_baseline, d512_prefetch);
    print_comparison("D=512: Baseline vs K-Tiling", d512_baseline, d512_ktile);

    // ==================== Summary ====================
    printf("\n================================================================\n");
    printf("Summary\n");
    printf("================================================================\n");

    printf("\nD=256 Results:\n");
    printf("  Working set: 4 KB per K-group iteration\n");
    printf("  Baseline efficiency: %.2f%%\n", d256_baseline.efficiency);
    printf("  Prefetch efficiency: %.2f%% (%+.2f%%)\n",
           d256_prefetch.efficiency, d256_prefetch.efficiency - d256_baseline.efficiency);

    printf("\nD=512 Results:\n");
    printf("  Working set: 8 KB per K-group iteration\n");
    printf("  Baseline efficiency: %.2f%%\n", d512_baseline.efficiency);
    printf("  Prefetch efficiency: %.2f%% (%+.2f%%)\n",
           d512_prefetch.efficiency, d512_prefetch.efficiency - d512_baseline.efficiency);
    printf("  K-tiling efficiency: %.2f%% (%+.2f%%)\n",
           d512_ktile.efficiency, d512_ktile.efficiency - d512_baseline.efficiency);

    printf("\nKey Findings:\n");
    if (d256_prefetch.efficiency > d256_baseline.efficiency) {
        printf("  ✓ D=256 benefits from prefetching\n");
    } else {
        printf("  ✗ D=256 does not benefit from prefetching (already L1-bound)\n");
    }

    if (d512_prefetch.efficiency > d512_baseline.efficiency + 1.0) {
        printf("  ✓ D=512 benefits significantly from prefetching\n");
    } else {
        printf("  ⚠ D=512 prefetching shows modest improvement\n");
    }

    if (d512_ktile.efficiency > d512_baseline.efficiency + 1.0) {
        printf("  ✓ D=512 benefits significantly from K-tiling\n");
    } else {
        printf("  ⚠ D=512 K-tiling shows modest improvement\n");
    }

    printf("\n================================================================\n");

    // Cleanup
    free(A256); free(B256); free(C256);
    free(A512); free(B512); free(C512);

    return 0;
}
