/*
 * Benchmark for FP8 to FP16/FP32 Conversion Kernels
 *
 * Tests:
 * - LUT-based gather (FCVT-like, 9 cycles per gather)
 * - Bit arithmetic with ARM64 base instructions (EX* pipe)
 * - Bit arithmetic with SVE instructions (FL* pipe)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "fp8_convert.h"

#define ALIGN 256
#define CPU_FREQ_MHZ 2000  // A64FX runs at 2.0 GHz (normal mode)

static inline uint64_t get_timer_ticks(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

// Allocate aligned memory
static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

// Verify conversion correctness
static int verify_fp16(const uint16_t* result, const uint16_t* expected, int n, const char* name) {
    int errors = 0;
    for (int i = 0; i < n && errors < 10; i++) {
        if (result[i] != expected[i]) {
            printf("  %s MISMATCH at %d: got 0x%04x, expected 0x%04x\n",
                   name, i, result[i], expected[i]);
            errors++;
        }
    }
    return errors == 0;
}

static int verify_fp32(const uint32_t* result, const uint32_t* expected, int n, const char* name) {
    int errors = 0;
    for (int i = 0; i < n && errors < 10; i++) {
        if (result[i] != expected[i]) {
            printf("  %s MISMATCH at %d: got 0x%08x, expected 0x%08x\n",
                   name, i, result[i], expected[i]);
            errors++;
        }
    }
    return errors == 0;
}

// Benchmark function type
typedef void (*convert_fp8_to_fp16_fn)(const uint8_t*, uint16_t*, int);
typedef void (*convert_fp8_to_fp32_fn)(const uint8_t*, uint32_t*, int);

// Run benchmark - returns timer ticks per element
static double benchmark_fp16(convert_fp8_to_fp16_fn fn, const uint8_t* src, uint16_t* dst,
                              int n, int warmup, int iterations) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        fn(src, dst, n);
    }

    uint64_t start = get_timer_ticks();
    for (int i = 0; i < iterations; i++) {
        fn(src, dst, n);
    }
    uint64_t end = get_timer_ticks();

    return (double)(end - start) / (iterations * n);
}

static double benchmark_fp32(convert_fp8_to_fp32_fn fn, const uint8_t* src, uint32_t* dst,
                              int n, int warmup, int iterations) {
    for (int i = 0; i < warmup; i++) {
        fn(src, dst, n);
    }

    uint64_t start = get_timer_ticks();
    for (int i = 0; i < iterations; i++) {
        fn(src, dst, n);
    }
    uint64_t end = get_timer_ticks();

    return (double)(end - start) / (iterations * n);
}

int main(int argc, char** argv) {
    printf("=== FP8 to FP16/FP32 Conversion Benchmark ===\n\n");

    // Get timer frequency and calculate conversion factor
    uint64_t timer_freq = get_timer_freq();
    double ticks_to_cycles = (double)CPU_FREQ_MHZ * 1e6 / timer_freq;

    printf("Timer frequency: %lu Hz\n", timer_freq);
    printf("CPU frequency: %d MHz\n", CPU_FREQ_MHZ);
    printf("Ticks to cycles factor: %.1f\n", ticks_to_cycles);
    printf("SVE vector length: %d bits (%d x 32-bit, %d x 16-bit)\n\n",
           (int)svcntb() * 8, (int)svcntw(), (int)svcnth());

    // A64FX memory specs
    printf("A64FX L1 specs:\n");
    printf("  - Load bandwidth: 128 bytes/cycle (2 x 64B pipes)\n");
    printf("  - Load latency: 11 cycles\n");
    printf("  - Gather latency: ~9 cycles (per element, pipelined)\n\n");

    // Initialize LUTs
    printf("Initializing lookup tables...\n");
    init_fp8_luts();

    int n = 65536;  // Fixed size for detailed analysis
    int warmup = 10;
    int iterations = 100;

    // Allocate buffers
    uint8_t* src = (uint8_t*)aligned_alloc_wrapper(ALIGN, n);
    uint16_t* dst_fp16 = (uint16_t*)aligned_alloc_wrapper(ALIGN, n * sizeof(uint16_t));
    uint16_t* ref_fp16 = (uint16_t*)aligned_alloc_wrapper(ALIGN, n * sizeof(uint16_t));
    uint32_t* dst_fp32 = (uint32_t*)aligned_alloc_wrapper(ALIGN, n * sizeof(uint32_t));
    uint32_t* ref_fp32 = (uint32_t*)aligned_alloc_wrapper(ALIGN, n * sizeof(uint32_t));

    if (!src || !dst_fp16 || !ref_fp16 || !dst_fp32 || !ref_fp32) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize source with all possible FP8 values (cycling)
    for (int i = 0; i < n; i++) {
        src[i] = i & 0xFF;
    }

    int vl32 = svcntw();  // 32-bit vector length (16 for 512-bit SVE)
    int vl16 = svcnth();  // 16-bit vector length (32 for 512-bit SVE)

    printf("\n");
    printf("================================================================================\n");
    printf("Detailed Performance Analysis (N=%d)\n", n);
    printf("================================================================================\n\n");

    // E4M3 to FP16
    printf("--- E4M3 -> FP16 ---\n");
    for (int i = 0; i < n; i++) ref_fp16[i] = fp8_e4m3_to_fp16_scalar(src[i]);

    double ticks_gather = benchmark_fp16(fp8_e4m3_to_fp16_gather, src, dst_fp16, n, warmup, iterations);
    verify_fp16(dst_fp16, ref_fp16, n, "Gather");

    double ticks_base = benchmark_fp16(fp8_e4m3_to_fp16_base, src, dst_fp16, n, warmup, iterations);
    verify_fp16(dst_fp16, ref_fp16, n, "Base");

    double ticks_sve = benchmark_fp16(fp8_e4m3_to_fp16_sve, src, dst_fp16, n, warmup, iterations);
    verify_fp16(dst_fp16, ref_fp16, n, "SVE");

    double cyc_gather = ticks_gather * ticks_to_cycles;
    double cyc_base = ticks_base * ticks_to_cycles;
    double cyc_sve = ticks_sve * ticks_to_cycles;

    printf("  Method          | cyc/elem | cyc/16elem | cyc/32elem | Notes\n");
    printf("  ----------------|----------|------------|------------|------------------\n");
    printf("  Gather (LUT)    | %8.2f | %10.1f | %10.1f | 16 elem per SVE op\n",
           cyc_gather, cyc_gather * 16, cyc_gather * 32);
    printf("  Base (EX*)      | %8.2f | %10.1f | %10.1f | Scalar\n",
           cyc_base, cyc_base * 16, cyc_base * 32);
    printf("  SVE (FL*)       | %8.2f | %10.1f | %10.1f | 16 elem per SVE op\n",
           cyc_sve, cyc_sve * 16, cyc_sve * 32);
    printf("\n");

    // E4M3 to FP32
    printf("--- E4M3 -> FP32 ---\n");
    for (int i = 0; i < n; i++) ref_fp32[i] = fp8_e4m3_to_fp32_scalar(src[i]);

    ticks_gather = benchmark_fp32(fp8_e4m3_to_fp32_gather, src, dst_fp32, n, warmup, iterations);
    verify_fp32(dst_fp32, ref_fp32, n, "Gather");

    ticks_base = benchmark_fp32(fp8_e4m3_to_fp32_base, src, dst_fp32, n, warmup, iterations);
    verify_fp32(dst_fp32, ref_fp32, n, "Base");

    ticks_sve = benchmark_fp32(fp8_e4m3_to_fp32_sve, src, dst_fp32, n, warmup, iterations);
    verify_fp32(dst_fp32, ref_fp32, n, "SVE");

    cyc_gather = ticks_gather * ticks_to_cycles;
    cyc_base = ticks_base * ticks_to_cycles;
    cyc_sve = ticks_sve * ticks_to_cycles;

    printf("  Method          | cyc/elem | cyc/16elem | cyc/32elem | Notes\n");
    printf("  ----------------|----------|------------|------------|------------------\n");
    printf("  Gather (LUT)    | %8.2f | %10.1f | %10.1f | 16 elem per SVE op\n",
           cyc_gather, cyc_gather * 16, cyc_gather * 32);
    printf("  Base (EX*)      | %8.2f | %10.1f | %10.1f | Scalar\n",
           cyc_base, cyc_base * 16, cyc_base * 32);
    printf("  SVE (FL*)       | %8.2f | %10.1f | %10.1f | 16 elem per SVE op\n",
           cyc_sve, cyc_sve * 16, cyc_sve * 32);
    printf("\n");

    // E5M2 to FP16
    printf("--- E5M2 -> FP16 ---\n");
    for (int i = 0; i < n; i++) ref_fp16[i] = fp8_e5m2_to_fp16_scalar(src[i]);

    ticks_gather = benchmark_fp16(fp8_e5m2_to_fp16_gather, src, dst_fp16, n, warmup, iterations);
    verify_fp16(dst_fp16, ref_fp16, n, "Gather");

    ticks_base = benchmark_fp16(fp8_e5m2_to_fp16_base, src, dst_fp16, n, warmup, iterations);
    verify_fp16(dst_fp16, ref_fp16, n, "Base");

    ticks_sve = benchmark_fp16(fp8_e5m2_to_fp16_sve, src, dst_fp16, n, warmup, iterations);
    verify_fp16(dst_fp16, ref_fp16, n, "SVE");

    cyc_gather = ticks_gather * ticks_to_cycles;
    cyc_base = ticks_base * ticks_to_cycles;
    cyc_sve = ticks_sve * ticks_to_cycles;

    printf("  Method          | cyc/elem | cyc/16elem | cyc/32elem | Notes\n");
    printf("  ----------------|----------|------------|------------|------------------\n");
    printf("  Gather (LUT)    | %8.2f | %10.1f | %10.1f | 16 elem per SVE op\n",
           cyc_gather, cyc_gather * 16, cyc_gather * 32);
    printf("  Base (EX*)      | %8.2f | %10.1f | %10.1f | Scalar\n",
           cyc_base, cyc_base * 16, cyc_base * 32);
    printf("  SVE (FL*)       | %8.2f | %10.1f | %10.1f | 16 elem per SVE op\n",
           cyc_sve, cyc_sve * 16, cyc_sve * 32);
    printf("\n");

    // E5M2 to FP32
    printf("--- E5M2 -> FP32 ---\n");
    for (int i = 0; i < n; i++) ref_fp32[i] = fp8_e5m2_to_fp32_scalar(src[i]);

    ticks_gather = benchmark_fp32(fp8_e5m2_to_fp32_gather, src, dst_fp32, n, warmup, iterations);
    verify_fp32(dst_fp32, ref_fp32, n, "Gather");

    ticks_base = benchmark_fp32(fp8_e5m2_to_fp32_base, src, dst_fp32, n, warmup, iterations);
    verify_fp32(dst_fp32, ref_fp32, n, "Base");

    ticks_sve = benchmark_fp32(fp8_e5m2_to_fp32_sve, src, dst_fp32, n, warmup, iterations);
    verify_fp32(dst_fp32, ref_fp32, n, "SVE");

    cyc_gather = ticks_gather * ticks_to_cycles;
    cyc_base = ticks_base * ticks_to_cycles;
    cyc_sve = ticks_sve * ticks_to_cycles;

    printf("  Method          | cyc/elem | cyc/16elem | cyc/32elem | Notes\n");
    printf("  ----------------|----------|------------|------------|------------------\n");
    printf("  Gather (LUT)    | %8.2f | %10.1f | %10.1f | 16 elem per SVE op\n",
           cyc_gather, cyc_gather * 16, cyc_gather * 32);
    printf("  Base (EX*)      | %8.2f | %10.1f | %10.1f | Scalar\n",
           cyc_base, cyc_base * 16, cyc_base * 32);
    printf("  SVE (FL*)       | %8.2f | %10.1f | %10.1f | 16 elem per SVE op\n",
           cyc_sve, cyc_sve * 16, cyc_sve * 32);

    printf("\n");
    printf("================================================================================\n");
    printf("Analysis Notes:\n");
    printf("================================================================================\n");
    printf("- LUT size: 256 x 4B = 1KB (fits in L1, 16 cache lines)\n");
    printf("- Gather processes 16 x 32-bit elements per SVE instruction\n");
    printf("- For 32 FP16 elements: need 2 SVE ops (or pack from 32-bit)\n");
    printf("- L1 gather latency model: 11 + (N_loads - 1) = 11 + 15 = 26 cyc max\n");
    printf("  (if all 16 loads hit different cache lines, pipelined)\n");
    printf("- Observed ~12 cyc/16elem suggests good L1 hit rate and pipelining\n");
    printf("================================================================================\n");

    // Cleanup
    free(src);
    free(dst_fp16);
    free(ref_fp16);
    free(dst_fp32);
    free(ref_fp32);

    return 0;
}
