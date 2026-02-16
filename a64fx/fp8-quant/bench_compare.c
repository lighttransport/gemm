#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include "fp8_quant.h"
#include "fp8_quant_opt.h"
#include "fp8_quant_opt8.h"

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
    printf("=== FP8 Quantization: Unroll Factor Comparison ===\n\n");
    printf("A64FX Latencies: LD=11, FCVT=9, MUL=9, ADD=5, BITOP=4 cycles\n");
    printf("SVE VL: %lu bits, Timer: 100MHz, CPU: 2.0GHz (20x ratio)\n\n", svcntb() * 8);

    size_t n = 8192;  // Multiple of 256 for x8 unroll (256 = 8 * 32)
    int warmup = 100;
    int iterations = 1000;

    uint16_t* fp16_data = aligned_alloc_wrapper(ALIGN, n * sizeof(uint16_t));
    float* fp32_data = aligned_alloc_wrapper(ALIGN, n * sizeof(float));
    uint8_t* out_ref = aligned_alloc_wrapper(ALIGN, n);
    uint8_t* out_test = aligned_alloc_wrapper(ALIGN, n);

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
    uint64_t baseline_fp16 = 0, baseline_fp32 = 0;

    // =========================================================================
    // FP16 -> E5M2: Compare unroll factors
    // =========================================================================
    printf("\n=== FP16 -> E5M2 ===\n");

    // Original (x1)
    for (int w = 0; w < warmup; w++) fp16_to_fp8_e5m2_sve(fp16_data, out_ref, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp16_to_fp8_e5m2_sve(fp16_data, out_ref, n);
    t1 = read_cycles();
    baseline_fp16 = (t1 - t0) / iterations;
    double cpe = (double)baseline_fp16 * 20.0 / n;
    printf("%-30s  %10lu  %14.2f  %10s\n", "Original (x1)", baseline_fp16, cpe, "-");

    // Unroll x4
    for (int w = 0; w < warmup; w++) fp16_to_fp8_e5m2_sve_unroll4(fp16_data, out_test, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp16_to_fp8_e5m2_sve_unroll4(fp16_data, out_test, n);
    t1 = read_cycles();
    uint64_t time_x4 = (t1 - t0) / iterations;
    cpe = (double)time_x4 * 20.0 / n;
    double speedup = (double)baseline_fp16 / time_x4;
    printf("%-30s  %10lu  %14.2f  %10.2fx\n", "Unroll x4", time_x4, cpe, speedup);

    // Validate x4
    int errors = 0;
    for (size_t i = 0; i < n && errors < 3; i++) {
        if (out_ref[i] != out_test[i]) {
            printf("  x4 MISMATCH at %zu: ref=0x%02X got=0x%02X\n", i, out_ref[i], out_test[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  [x4 Validation PASSED]\n");

    // Unroll x8
    for (int w = 0; w < warmup; w++) fp16_to_fp8_e5m2_sve_unroll8(fp16_data, out_test, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp16_to_fp8_e5m2_sve_unroll8(fp16_data, out_test, n);
    t1 = read_cycles();
    uint64_t time_x8 = (t1 - t0) / iterations;
    cpe = (double)time_x8 * 20.0 / n;
    speedup = (double)baseline_fp16 / time_x8;
    printf("%-30s  %10lu  %14.2f  %10.2fx\n", "Unroll x8", time_x8, cpe, speedup);

    // Validate x8
    errors = 0;
    for (size_t i = 0; i < n && errors < 3; i++) {
        if (out_ref[i] != out_test[i]) {
            printf("  x8 MISMATCH at %zu: ref=0x%02X got=0x%02X\n", i, out_ref[i], out_test[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  [x8 Validation PASSED]\n");

    // =========================================================================
    // FP32 -> E5M2: Compare unroll factors
    // =========================================================================
    printf("\n=== FP32 -> E5M2 ===\n");

    // Original (x1)
    for (int w = 0; w < warmup; w++) fp32_to_fp8_e5m2_sve(fp32_data, out_ref, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp32_to_fp8_e5m2_sve(fp32_data, out_ref, n);
    t1 = read_cycles();
    baseline_fp32 = (t1 - t0) / iterations;
    cpe = (double)baseline_fp32 * 20.0 / n;
    printf("%-30s  %10lu  %14.2f  %10s\n", "Original (x1)", baseline_fp32, cpe, "-");

    // Unroll x4
    for (int w = 0; w < warmup; w++) fp32_to_fp8_e5m2_sve_unroll4(fp32_data, out_test, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp32_to_fp8_e5m2_sve_unroll4(fp32_data, out_test, n);
    t1 = read_cycles();
    time_x4 = (t1 - t0) / iterations;
    cpe = (double)time_x4 * 20.0 / n;
    speedup = (double)baseline_fp32 / time_x4;
    printf("%-30s  %10lu  %14.2f  %10.2fx\n", "Unroll x4", time_x4, cpe, speedup);

    // Validate x4
    errors = 0;
    for (size_t i = 0; i < n && errors < 3; i++) {
        if (out_ref[i] != out_test[i]) {
            printf("  x4 MISMATCH at %zu: ref=0x%02X got=0x%02X (src=%f)\n",
                   i, out_ref[i], out_test[i], fp32_data[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  [x4 Validation PASSED]\n");

    // Unroll x8
    for (int w = 0; w < warmup; w++) fp32_to_fp8_e5m2_sve_unroll8(fp32_data, out_test, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp32_to_fp8_e5m2_sve_unroll8(fp32_data, out_test, n);
    t1 = read_cycles();
    time_x8 = (t1 - t0) / iterations;
    cpe = (double)time_x8 * 20.0 / n;
    speedup = (double)baseline_fp32 / time_x8;
    printf("%-30s  %10lu  %14.2f  %10.2fx\n", "Unroll x8", time_x8, cpe, speedup);

    // Validate x8
    errors = 0;
    for (size_t i = 0; i < n && errors < 3; i++) {
        if (out_ref[i] != out_test[i]) {
            printf("  x8 MISMATCH at %zu: ref=0x%02X got=0x%02X (src=%f)\n",
                   i, out_ref[i], out_test[i], fp32_data[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  [x8 Validation PASSED]\n");

    printf("\n=== Summary ===\n");
    printf("Higher unroll factors help hide the 11-cycle load latency.\n");
    printf("x8 should approach the memory bandwidth limit for these simple kernels.\n");

    free(fp16_data);
    free(fp32_data);
    free(out_ref);
    free(out_test);

    return 0;
}
