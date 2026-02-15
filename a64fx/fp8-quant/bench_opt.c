#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include "fp8_quant.h"
#include "fp8_quant_opt.h"

#define ALIGN 64

static inline uint64_t read_cycles(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

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
    printf("=== FP8 Quantization: Original vs Unrolled Optimization ===\n\n");
    printf("A64FX Latencies: LD=11, FCVT=9, MUL=9, ADD=5, BITOP=4 cycles\n");
    printf("SVE VL: %lu bits, Timer: 100MHz, CPU: 2.0GHz\n\n", svcntb() * 8);

    size_t n = 4096;  // L1 resident
    int warmup = 100;
    int iterations = 1000;

    uint16_t* fp16_data = aligned_alloc_wrapper(ALIGN, n * sizeof(uint16_t));
    float* fp32_data = aligned_alloc_wrapper(ALIGN, n * sizeof(float));
    uint8_t* out_orig = aligned_alloc_wrapper(ALIGN, n);
    uint8_t* out_opt = aligned_alloc_wrapper(ALIGN, n);

    // Initialize with normal range values
    srand(42);
    for (size_t i = 0; i < n; i++) {
        float f = 1.0f + (float)(rand() % 100) / 10.0f;
        if (rand() % 2) f = -f;
        fp32_data[i] = f;
        fp16_data[i] = fp32_to_fp16_sw(f);
    }

    printf("Test: %zu elements, %d warmup, %d iterations\n\n", n, warmup, iterations);

    printf("%-30s  %10s  %14s  %10s\n",
           "Kernel", "Ticks", "Cycles/Elem", "Speedup");
    printf("%-30s  %10s  %14s  %10s\n",
           "------", "-----", "-----------", "-------");

    uint64_t t0, t1;

    // =========================================================================
    // FP16 -> E5M2: Original
    // =========================================================================
    for (int w = 0; w < warmup; w++) fp16_to_fp8_e5m2_sve(fp16_data, out_orig, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp16_to_fp8_e5m2_sve(fp16_data, out_orig, n);
    t1 = read_cycles();
    uint64_t orig_fp16_e5m2 = (t1 - t0) / iterations;
    double cpe_orig_fp16 = (double)orig_fp16_e5m2 * 20.0 / n;
    printf("%-30s  %10lu  %14.2f  %10s\n", "FP16->E5M2 (original)", orig_fp16_e5m2, cpe_orig_fp16, "-");

    // FP16 -> E5M2: Unroll x4
    for (int w = 0; w < warmup; w++) fp16_to_fp8_e5m2_sve_unroll4(fp16_data, out_opt, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp16_to_fp8_e5m2_sve_unroll4(fp16_data, out_opt, n);
    t1 = read_cycles();
    uint64_t opt_fp16_e5m2 = (t1 - t0) / iterations;
    double cpe_opt_fp16 = (double)opt_fp16_e5m2 * 20.0 / n;
    double speedup_fp16 = (double)orig_fp16_e5m2 / opt_fp16_e5m2;
    printf("%-30s  %10lu  %14.2f  %10.2fx\n", "FP16->E5M2 (unroll x4)", opt_fp16_e5m2, cpe_opt_fp16, speedup_fp16);

    // Validate
    int errors = 0;
    for (size_t i = 0; i < n && errors < 5; i++) {
        if (out_orig[i] != out_opt[i]) {
            printf("  MISMATCH at %zu: orig=0x%02X opt=0x%02X\n", i, out_orig[i], out_opt[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  [Validation PASSED]\n");
    printf("\n");

    // =========================================================================
    // FP32 -> E5M2: Original
    // =========================================================================
    for (int w = 0; w < warmup; w++) fp32_to_fp8_e5m2_sve(fp32_data, out_orig, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp32_to_fp8_e5m2_sve(fp32_data, out_orig, n);
    t1 = read_cycles();
    uint64_t orig_fp32_e5m2 = (t1 - t0) / iterations;
    double cpe_orig_fp32 = (double)orig_fp32_e5m2 * 20.0 / n;
    printf("%-30s  %10lu  %14.2f  %10s\n", "FP32->E5M2 (original)", orig_fp32_e5m2, cpe_orig_fp32, "-");

    // FP32 -> E5M2: Unroll x4
    for (int w = 0; w < warmup; w++) fp32_to_fp8_e5m2_sve_unroll4(fp32_data, out_opt, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp32_to_fp8_e5m2_sve_unroll4(fp32_data, out_opt, n);
    t1 = read_cycles();
    uint64_t opt_fp32_e5m2 = (t1 - t0) / iterations;
    double cpe_opt_fp32 = (double)opt_fp32_e5m2 * 20.0 / n;
    double speedup_fp32 = (double)orig_fp32_e5m2 / opt_fp32_e5m2;
    printf("%-30s  %10lu  %14.2f  %10.2fx\n", "FP32->E5M2 (unroll x4)", opt_fp32_e5m2, cpe_opt_fp32, speedup_fp32);

    // Validate
    errors = 0;
    for (size_t i = 0; i < n && errors < 5; i++) {
        if (out_orig[i] != out_opt[i]) {
            printf("  MISMATCH at %zu: orig=0x%02X opt=0x%02X (src=%f)\n",
                   i, out_orig[i], out_opt[i], fp32_data[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  [Validation PASSED]\n");
    printf("\n");

    printf("\n");
    printf("=== Summary ===\n");
    printf("FP16->E5M2: %.2f -> %.2f cycles/elem (%.1fx speedup)\n",
           cpe_orig_fp16, cpe_opt_fp16, speedup_fp16);
    printf("FP32->E5M2: %.2f -> %.2f cycles/elem (%.1fx speedup with unroll)\n",
           cpe_orig_fp32, cpe_opt_fp32, speedup_fp32);

    free(fp16_data);
    free(fp32_data);
    free(out_orig);
    free(out_opt);

    return 0;
}
