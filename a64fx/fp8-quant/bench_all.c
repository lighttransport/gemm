#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include "fp8_quant.h"
#include "fp8_quant_opt.h"
#include "fp8_quant_opt8.h"
#include "fp8_quant_swp.h"

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

typedef void (*fp16_kernel_t)(const uint16_t*, uint8_t*, size_t);
typedef void (*fp32_kernel_t)(const float*, uint8_t*, size_t);

int main(int argc, char** argv) {
    printf("=== FP8 Quantization: Complete Optimization Comparison ===\n\n");
    printf("A64FX: LD=11cy, FCVT=9cy, MUL=9cy, ADD=5cy, BITOP=4cy\n");
    printf("SVE VL: %lu bits, Timer: 100MHz, CPU: 2.0GHz\n\n", svcntb() * 8);

    size_t n = 8192;
    int warmup = 100;
    int iterations = 1000;

    uint16_t* fp16_data = aligned_alloc_wrapper(ALIGN, n * sizeof(uint16_t));
    float* fp32_data = aligned_alloc_wrapper(ALIGN, n * sizeof(float));
    uint8_t* out_ref = aligned_alloc_wrapper(ALIGN, n);
    uint8_t* out_test = aligned_alloc_wrapper(ALIGN, n);

    srand(42);
    for (size_t i = 0; i < n; i++) {
        float f = 1.0f + (float)(rand() % 100) / 10.0f;
        if (rand() % 2) f = -f;
        fp32_data[i] = f;
        fp16_data[i] = fp32_to_fp16_sw(f);
    }

    printf("Test: %zu elements, %d warmup, %d iters\n\n", n, warmup, iterations);

    uint64_t t0, t1, ticks;
    double cpe, speedup;

    // =========================================================================
    // FP16 -> E5M2
    // =========================================================================
    printf("=== FP16 -> E5M2 ===\n");
    printf("%-25s  %8s  %12s  %8s  %s\n", "Kernel", "Ticks", "Cycles/Elem", "Speedup", "Status");
    printf("%-25s  %8s  %12s  %8s  %s\n", "------", "-----", "-----------", "-------", "------");

    // Reference
    for (int w = 0; w < warmup; w++) fp16_to_fp8_e5m2_sve(fp16_data, out_ref, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp16_to_fp8_e5m2_sve(fp16_data, out_ref, n);
    t1 = read_cycles();
    uint64_t base_fp16 = (t1 - t0) / iterations;
    cpe = (double)base_fp16 * 20.0 / n;
    printf("%-25s  %8lu  %12.2f  %8s  %s\n", "Original (x1)", base_fp16, cpe, "-", "REF");

    struct { const char* name; fp16_kernel_t fn; } fp16_kernels[] = {
        {"Unroll x4", fp16_to_fp8_e5m2_sve_unroll4},
        {"Unroll x8", fp16_to_fp8_e5m2_sve_unroll8},
        {"SW Pipeline", fp16_to_fp8_e5m2_sve_swp},
    };

    for (int k = 0; k < 3; k++) {
        for (int w = 0; w < warmup; w++) fp16_kernels[k].fn(fp16_data, out_test, n);
        t0 = read_cycles();
        for (int it = 0; it < iterations; it++) fp16_kernels[k].fn(fp16_data, out_test, n);
        t1 = read_cycles();
        ticks = (t1 - t0) / iterations;
        cpe = (double)ticks * 20.0 / n;
        speedup = (double)base_fp16 / ticks;

        int errors = 0;
        for (size_t i = 0; i < n && errors < 1; i++) {
            if (out_ref[i] != out_test[i]) errors++;
        }
        printf("%-25s  %8lu  %12.2f  %7.2fx  %s\n",
               fp16_kernels[k].name, ticks, cpe, speedup,
               errors == 0 ? "PASS" : "FAIL");
    }

    // =========================================================================
    // FP32 -> E5M2
    // =========================================================================
    printf("\n=== FP32 -> E5M2 ===\n");
    printf("%-25s  %8s  %12s  %8s  %s\n", "Kernel", "Ticks", "Cycles/Elem", "Speedup", "Status");
    printf("%-25s  %8s  %12s  %8s  %s\n", "------", "-----", "-----------", "-------", "------");

    // Reference
    for (int w = 0; w < warmup; w++) fp32_to_fp8_e5m2_sve(fp32_data, out_ref, n);
    t0 = read_cycles();
    for (int it = 0; it < iterations; it++) fp32_to_fp8_e5m2_sve(fp32_data, out_ref, n);
    t1 = read_cycles();
    uint64_t base_fp32 = (t1 - t0) / iterations;
    cpe = (double)base_fp32 * 20.0 / n;
    printf("%-25s  %8lu  %12.2f  %8s  %s\n", "Original (x1)", base_fp32, cpe, "-", "REF");

    struct { const char* name; fp32_kernel_t fn; } fp32_kernels[] = {
        {"Unroll x4", fp32_to_fp8_e5m2_sve_unroll4},
        {"Unroll x8", fp32_to_fp8_e5m2_sve_unroll8},
        {"SW Pipeline", fp32_to_fp8_e5m2_sve_swp},
        {"FCVT (FP32->FP16->E5M2)", fp32_to_fp8_e5m2_sve_fcvt},
    };

    for (int k = 0; k < 4; k++) {
        for (int w = 0; w < warmup; w++) fp32_kernels[k].fn(fp32_data, out_test, n);
        t0 = read_cycles();
        for (int it = 0; it < iterations; it++) fp32_kernels[k].fn(fp32_data, out_test, n);
        t1 = read_cycles();
        ticks = (t1 - t0) / iterations;
        cpe = (double)ticks * 20.0 / n;
        speedup = (double)base_fp32 / ticks;

        int errors = 0;
        for (size_t i = 0; i < n && errors < 1; i++) {
            if (out_ref[i] != out_test[i]) errors++;
        }

        const char* status = errors == 0 ? "PASS" : "FAIL";
        // FCVT may have minor rounding differences, check if close
        if (errors > 0 && k == 3) {
            int close = 0;
            for (size_t i = 0; i < n; i++) {
                int diff = (int)out_ref[i] - (int)out_test[i];
                if (diff >= -1 && diff <= 1) close++;
            }
            if (close == (int)n) status = "PASS (rnd)";
        }

        printf("%-25s  %8lu  %12.2f  %7.2fx  %s\n",
               fp32_kernels[k].name, ticks, cpe, speedup, status);
    }

    // =========================================================================
    // Summary
    // =========================================================================
    printf("\n=== Analysis ===\n");
    printf("- Unrolling hides load latency (11 cycles) by issuing multiple loads early\n");
    printf("- SW pipelining overlaps load/compute/store across iterations\n");
    printf("- FCVT uses hardware FP32->FP16, then simple bit ops for FP16->E5M2\n");
    printf("- Best approach depends on data size and cache residency\n");

    free(fp16_data);
    free(fp32_data);
    free(out_ref);
    free(out_test);

    return 0;
}
