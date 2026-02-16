#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include "fp8_quant.h"

#define ALIGN 64

// Read cycle counter (A64FX)
static inline uint64_t read_cycles(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Get timer frequency
static inline uint64_t get_freq(void) {
    uint64_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

// Software FP32 to FP16
static uint16_t fp32_to_fp16_sw(float f) {
    union { float fv; uint32_t u; } conv = {f};
    uint32_t bits = conv.u;
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | mant;
}

int main(int argc, char** argv) {
    printf("=== FP8 Quantization Cycle Analysis for A64FX ===\n\n");

    uint64_t freq = get_freq();
    printf("Timer frequency: %lu Hz (%.2f MHz)\n", freq, freq / 1e6);
    printf("SVE Vector Length: %lu bits (%lu bytes)\n", svcntb() * 8, svcntb());
    printf("  - FP16 elements per vector: %lu\n", svcnth());
    printf("  - FP32 elements per vector: %lu\n\n", svcntw());

    // Test sizes - use sizes that fit in L1/L2 cache for accurate cycle measurement
    size_t n = 4096;  // Fits in L1 cache (64KB per core)
    int warmup = 100;
    int iterations = 1000;

    // Allocate
    uint16_t* fp16_data = aligned_alloc_wrapper(ALIGN, n * sizeof(uint16_t));
    float* fp32_data = aligned_alloc_wrapper(ALIGN, n * sizeof(float));
    uint8_t* fp8_out = aligned_alloc_wrapper(ALIGN, n);

    // Initialize with values in normal range (avoid subnormal overhead)
    srand(42);
    for (size_t i = 0; i < n; i++) {
        float f = 1.0f + (float)(rand() % 100) / 10.0f;  // 1.0 to 11.0
        if (rand() % 2) f = -f;
        fp32_data[i] = f;
        fp16_data[i] = fp32_to_fp16_sw(f);
    }

    printf("Test configuration:\n");
    printf("  - Elements: %zu\n", n);
    printf("  - Warmup iterations: %d\n", warmup);
    printf("  - Timed iterations: %d\n\n", iterations);

    // =========================================================================
    // FP16 -> E4M3
    // =========================================================================
    for (int w = 0; w < warmup; w++) {
        fp16_to_fp8_e4m3_sve(fp16_data, fp8_out, n);
    }

    uint64_t t0 = read_cycles();
    for (int it = 0; it < iterations; it++) {
        fp16_to_fp8_e4m3_sve(fp16_data, fp8_out, n);
    }
    uint64_t t1 = read_cycles();

    uint64_t cycles_fp16_e4m3 = (t1 - t0) / iterations;
    double cycles_per_elem_fp16_e4m3 = (double)cycles_fp16_e4m3 / n;

    // =========================================================================
    // FP16 -> E5M2
    // =========================================================================
    for (int w = 0; w < warmup; w++) {
        fp16_to_fp8_e5m2_sve(fp16_data, fp8_out, n);
    }

    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) {
        fp16_to_fp8_e5m2_sve(fp16_data, fp8_out, n);
    }
    t1 = read_cycles();

    uint64_t cycles_fp16_e5m2 = (t1 - t0) / iterations;
    double cycles_per_elem_fp16_e5m2 = (double)cycles_fp16_e5m2 / n;

    // =========================================================================
    // FP32 -> E4M3
    // =========================================================================
    for (int w = 0; w < warmup; w++) {
        fp32_to_fp8_e4m3_sve(fp32_data, fp8_out, n);
    }

    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) {
        fp32_to_fp8_e4m3_sve(fp32_data, fp8_out, n);
    }
    t1 = read_cycles();

    uint64_t cycles_fp32_e4m3 = (t1 - t0) / iterations;
    double cycles_per_elem_fp32_e4m3 = (double)cycles_fp32_e4m3 / n;

    // =========================================================================
    // FP32 -> E5M2
    // =========================================================================
    for (int w = 0; w < warmup; w++) {
        fp32_to_fp8_e5m2_sve(fp32_data, fp8_out, n);
    }

    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) {
        fp32_to_fp8_e5m2_sve(fp32_data, fp8_out, n);
    }
    t1 = read_cycles();

    uint64_t cycles_fp32_e5m2 = (t1 - t0) / iterations;
    double cycles_per_elem_fp32_e5m2 = (double)cycles_fp32_e5m2 / n;

    // =========================================================================
    // Results
    // =========================================================================
    printf("=== Cycle Count Results ===\n\n");
    printf("%-16s  %12s  %16s  %16s\n", "Kernel", "Total Cycles", "Cycles/Element", "Cycles/Vector");
    printf("%-16s  %12s  %16s  %16s\n", "------", "------------", "--------------", "-------------");

    printf("%-16s  %12lu  %16.2f  %16.2f (32 FP16)\n",
           "FP16 -> E4M3", cycles_fp16_e4m3, cycles_per_elem_fp16_e4m3,
           cycles_per_elem_fp16_e4m3 * 32);

    printf("%-16s  %12lu  %16.2f  %16.2f (32 FP16)\n",
           "FP16 -> E5M2", cycles_fp16_e5m2, cycles_per_elem_fp16_e5m2,
           cycles_per_elem_fp16_e5m2 * 32);

    printf("%-16s  %12lu  %16.2f  %16.2f (16 FP32)\n",
           "FP32 -> E4M3", cycles_fp32_e4m3, cycles_per_elem_fp32_e4m3,
           cycles_per_elem_fp32_e4m3 * 16);

    printf("%-16s  %12lu  %16.2f  %16.2f (16 FP32)\n",
           "FP32 -> E5M2", cycles_fp32_e5m2, cycles_per_elem_fp32_e5m2,
           cycles_per_elem_fp32_e5m2 * 16);

    printf("\n=== Throughput Analysis ===\n\n");

    // A64FX runs at 2.0 GHz (normal) or 2.2 GHz (boost)
    double freq_ghz = 2.0;

    printf("At %.1f GHz:\n", freq_ghz);
    printf("%-16s  Elements/cycle: %.2f, GB/s: %.2f\n",
           "FP16 -> E4M3", 1.0 / cycles_per_elem_fp16_e4m3,
           (n * 3.0) / cycles_fp16_e4m3 * freq_ghz);  // 2B in + 1B out
    printf("%-16s  Elements/cycle: %.2f, GB/s: %.2f\n",
           "FP16 -> E5M2", 1.0 / cycles_per_elem_fp16_e5m2,
           (n * 3.0) / cycles_fp16_e5m2 * freq_ghz);
    printf("%-16s  Elements/cycle: %.2f, GB/s: %.2f\n",
           "FP32 -> E4M3", 1.0 / cycles_per_elem_fp32_e4m3,
           (n * 5.0) / cycles_fp32_e4m3 * freq_ghz);  // 4B in + 1B out
    printf("%-16s  Elements/cycle: %.2f, GB/s: %.2f\n",
           "FP32 -> E5M2", 1.0 / cycles_per_elem_fp32_e5m2,
           (n * 5.0) / cycles_fp32_e5m2 * freq_ghz);

    // Cleanup
    free(fp16_data);
    free(fp32_data);
    free(fp8_out);

    printf("\nDone.\n");
    return 0;
}
