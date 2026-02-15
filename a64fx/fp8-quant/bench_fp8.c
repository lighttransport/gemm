#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_sve.h>
#include "fp8_quant.h"

#define ALIGN 64

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

// Software FP32 to FP16 conversion
static uint16_t fp32_to_fp16_sw(float f) {
    union { float fv; uint32_t u; } conv = {f};
    uint32_t bits = conv.u;

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;

    if (exp <= 0) {
        return sign;
    } else if (exp >= 31) {
        return sign | 0x7C00;
    }
    return sign | (exp << 10) | mant;
}

// Generate test FP16 values (stored as uint16_t)
static void generate_fp16_data(uint16_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        // Generate random FP16 values with various magnitudes
        float f = (float)(rand() % 1000) / 10.0f - 50.0f;  // -50 to 50
        // Scale some values
        if (rand() % 10 == 0) f *= 10.0f;  // Some larger values
        if (rand() % 10 == 0) f *= 0.01f;  // Some smaller values

        // Convert float to FP16 using software implementation
        data[i] = fp32_to_fp16_sw(f);
    }
}

// Generate test FP32 values
static void generate_fp32_data(float* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float f = (float)(rand() % 1000) / 10.0f - 50.0f;
        if (rand() % 10 == 0) f *= 10.0f;
        if (rand() % 10 == 0) f *= 0.01f;
        data[i] = f;
    }
}

// Validate FP16 -> FP8 E4M3
static int validate_fp16_to_e4m3(const uint16_t* src, const fp8_e4m3_t* dst, size_t n) {
    int errors = 0;
    for (size_t i = 0; i < n && errors < 10; i++) {
        fp8_e4m3_t expected = fp16_to_fp8_e4m3_scalar(src[i]);
        if (dst[i] != expected) {
            printf("  FP16->E4M3 mismatch at %zu: got 0x%02X, expected 0x%02X (src=0x%04X)\n",
                   i, dst[i], expected, src[i]);
            errors++;
        }
    }
    return errors;
}

// Validate FP16 -> FP8 E5M2
static int validate_fp16_to_e5m2(const uint16_t* src, const fp8_e5m2_t* dst, size_t n) {
    int errors = 0;
    for (size_t i = 0; i < n && errors < 10; i++) {
        fp8_e5m2_t expected = fp16_to_fp8_e5m2_scalar(src[i]);
        if (dst[i] != expected) {
            printf("  FP16->E5M2 mismatch at %zu: got 0x%02X, expected 0x%02X (src=0x%04X)\n",
                   i, dst[i], expected, src[i]);
            errors++;
        }
    }
    return errors;
}

// Validate FP32 -> FP8 E4M3
static int validate_fp32_to_e4m3(const float* src, const fp8_e4m3_t* dst, size_t n) {
    int errors = 0;
    for (size_t i = 0; i < n && errors < 10; i++) {
        fp8_e4m3_t expected = fp32_to_fp8_e4m3_scalar(src[i]);
        if (dst[i] != expected) {
            printf("  FP32->E4M3 mismatch at %zu: got 0x%02X, expected 0x%02X (src=%f)\n",
                   i, dst[i], expected, src[i]);
            errors++;
        }
    }
    return errors;
}

// Validate FP32 -> FP8 E5M2
static int validate_fp32_to_e5m2(const float* src, const fp8_e5m2_t* dst, size_t n) {
    int errors = 0;
    for (size_t i = 0; i < n && errors < 10; i++) {
        fp8_e5m2_t expected = fp32_to_fp8_e5m2_scalar(src[i]);
        if (dst[i] != expected) {
            printf("  FP32->E5M2 mismatch at %zu: got 0x%02X, expected 0x%02X (src=%f)\n",
                   i, dst[i], expected, src[i]);
            errors++;
        }
    }
    return errors;
}

int main(int argc, char** argv) {
    printf("=== FP8 Quantization Benchmark for A64FX ===\n\n");

    // Print SVE vector length
    printf("SVE Vector Length: %lu bits (%lu bytes)\n",
           svcntb() * 8, svcntb());
    printf("  - FP16 elements per vector: %lu\n", svcnth());
    printf("  - FP32 elements per vector: %lu\n\n", svcntw());

    // Test sizes
    size_t sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Allocate largest size
    size_t max_size = sizes[num_sizes - 1];
    uint16_t* fp16_data = aligned_alloc_wrapper(ALIGN, max_size * sizeof(uint16_t));
    float* fp32_data = aligned_alloc_wrapper(ALIGN, max_size * sizeof(float));
    fp8_e4m3_t* e4m3_out = aligned_alloc_wrapper(ALIGN, max_size);
    fp8_e5m2_t* e5m2_out = aligned_alloc_wrapper(ALIGN, max_size);

    srand(42);
    generate_fp16_data(fp16_data, max_size);
    generate_fp32_data(fp32_data, max_size);

    // =========================================================================
    // Correctness Tests
    // =========================================================================
    printf("=== Correctness Validation ===\n");

    size_t test_size = 4096;

    // Test FP16 -> E4M3
    memset(e4m3_out, 0, test_size);
    fp16_to_fp8_e4m3_sve(fp16_data, e4m3_out, test_size);
    int err1 = validate_fp16_to_e4m3(fp16_data, e4m3_out, test_size);
    printf("FP16 -> E4M3: %s (%d errors)\n", err1 == 0 ? "PASS" : "FAIL", err1);

    // Test FP16 -> E5M2
    memset(e5m2_out, 0, test_size);
    fp16_to_fp8_e5m2_sve(fp16_data, e5m2_out, test_size);
    int err2 = validate_fp16_to_e5m2(fp16_data, e5m2_out, test_size);
    printf("FP16 -> E5M2: %s (%d errors)\n", err2 == 0 ? "PASS" : "FAIL", err2);

    // Test FP32 -> E4M3
    memset(e4m3_out, 0, test_size);
    fp32_to_fp8_e4m3_sve(fp32_data, e4m3_out, test_size);
    int err3 = validate_fp32_to_e4m3(fp32_data, e4m3_out, test_size);
    printf("FP32 -> E4M3: %s (%d errors)\n", err3 == 0 ? "PASS" : "FAIL", err3);

    // Test FP32 -> E5M2
    memset(e5m2_out, 0, test_size);
    fp32_to_fp8_e5m2_sve(fp32_data, e5m2_out, test_size);
    int err4 = validate_fp32_to_e5m2(fp32_data, e5m2_out, test_size);
    printf("FP32 -> E5M2: %s (%d errors)\n", err4 == 0 ? "PASS" : "FAIL", err4);

    printf("\n");

    // =========================================================================
    // Edge Case Tests
    // =========================================================================
    printf("=== Edge Case Tests ===\n");

    // Test specific values
    float test_vals[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 448.0f, -448.0f,
                        1000.0f, 0.001f, 0.015625f};
    int n_test = sizeof(test_vals) / sizeof(test_vals[0]);

    printf("FP32 -> E4M3 edge cases:\n");
    for (int i = 0; i < n_test; i++) {
        fp8_e4m3_t e4m3 = fp32_to_fp8_e4m3_scalar(test_vals[i]);
        float back = fp8_e4m3_to_fp32_scalar(e4m3);
        printf("  %12.6f -> 0x%02X -> %12.6f\n", test_vals[i], e4m3, back);
    }

    printf("\nFP32 -> E5M2 edge cases:\n");
    for (int i = 0; i < n_test; i++) {
        fp8_e5m2_t e5m2 = fp32_to_fp8_e5m2_scalar(test_vals[i]);
        float back = fp8_e5m2_to_fp32_scalar(e5m2);
        printf("  %12.6f -> 0x%02X -> %12.6f\n", test_vals[i], e5m2, back);
    }

    printf("\n");

    // =========================================================================
    // Performance Benchmarks
    // =========================================================================
    printf("=== Performance Benchmarks ===\n\n");

    int warmup = 3;
    int iterations = 10;

    printf("%-12s  %-14s  %-14s  %-14s  %-14s\n",
           "Size", "FP16->E4M3", "FP16->E5M2", "FP32->E4M3", "FP32->E5M2");
    printf("%-12s  %-14s  %-14s  %-14s  %-14s\n",
           "", "(GB/s)", "(GB/s)", "(GB/s)", "(GB/s)");
    printf("------------------------------------------------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        // FP16 -> E4M3
        for (int w = 0; w < warmup; w++) {
            fp16_to_fp8_e4m3_sve(fp16_data, e4m3_out, n);
        }
        double t0 = get_time();
        for (int it = 0; it < iterations; it++) {
            fp16_to_fp8_e4m3_sve(fp16_data, e4m3_out, n);
        }
        double t1 = get_time();
        double bw_fp16_e4m3 = (n * (2 + 1) * iterations) / (t1 - t0) / 1e9;  // 2B in, 1B out

        // FP16 -> E5M2
        for (int w = 0; w < warmup; w++) {
            fp16_to_fp8_e5m2_sve(fp16_data, e5m2_out, n);
        }
        t0 = get_time();
        for (int it = 0; it < iterations; it++) {
            fp16_to_fp8_e5m2_sve(fp16_data, e5m2_out, n);
        }
        t1 = get_time();
        double bw_fp16_e5m2 = (n * (2 + 1) * iterations) / (t1 - t0) / 1e9;

        // FP32 -> E4M3
        for (int w = 0; w < warmup; w++) {
            fp32_to_fp8_e4m3_sve(fp32_data, e4m3_out, n);
        }
        t0 = get_time();
        for (int it = 0; it < iterations; it++) {
            fp32_to_fp8_e4m3_sve(fp32_data, e4m3_out, n);
        }
        t1 = get_time();
        double bw_fp32_e4m3 = (n * (4 + 1) * iterations) / (t1 - t0) / 1e9;  // 4B in, 1B out

        // FP32 -> E5M2
        for (int w = 0; w < warmup; w++) {
            fp32_to_fp8_e5m2_sve(fp32_data, e5m2_out, n);
        }
        t0 = get_time();
        for (int it = 0; it < iterations; it++) {
            fp32_to_fp8_e5m2_sve(fp32_data, e5m2_out, n);
        }
        t1 = get_time();
        double bw_fp32_e5m2 = (n * (4 + 1) * iterations) / (t1 - t0) / 1e9;

        printf("%-12zu  %-14.2f  %-14.2f  %-14.2f  %-14.2f\n",
               n, bw_fp16_e4m3, bw_fp16_e5m2, bw_fp32_e4m3, bw_fp32_e5m2);
    }

    printf("\n");

    // =========================================================================
    // Scalar vs SVE Speedup
    // =========================================================================
    printf("=== Scalar vs SVE Comparison (n=65536) ===\n");
    size_t bench_n = 65536;
    iterations = 100;

    // Scalar FP16 -> E4M3
    double t0 = get_time();
    for (int it = 0; it < iterations; it++) {
        for (size_t i = 0; i < bench_n; i++) {
            e4m3_out[i] = fp16_to_fp8_e4m3_scalar(fp16_data[i]);
        }
    }
    double t1 = get_time();
    double scalar_fp16_e4m3 = (t1 - t0) / iterations;

    // SVE FP16 -> E4M3
    t0 = get_time();
    for (int it = 0; it < iterations; it++) {
        fp16_to_fp8_e4m3_sve(fp16_data, e4m3_out, bench_n);
    }
    t1 = get_time();
    double sve_fp16_e4m3 = (t1 - t0) / iterations;

    // Scalar FP32 -> E4M3
    t0 = get_time();
    for (int it = 0; it < iterations; it++) {
        for (size_t i = 0; i < bench_n; i++) {
            e4m3_out[i] = fp32_to_fp8_e4m3_scalar(fp32_data[i]);
        }
    }
    t1 = get_time();
    double scalar_fp32_e4m3 = (t1 - t0) / iterations;

    // SVE FP32 -> E4M3
    t0 = get_time();
    for (int it = 0; it < iterations; it++) {
        fp32_to_fp8_e4m3_sve(fp32_data, e4m3_out, bench_n);
    }
    t1 = get_time();
    double sve_fp32_e4m3 = (t1 - t0) / iterations;

    printf("FP16 -> E4M3:  Scalar: %.3f ms, SVE: %.3f ms, Speedup: %.1fx\n",
           scalar_fp16_e4m3 * 1000, sve_fp16_e4m3 * 1000, scalar_fp16_e4m3 / sve_fp16_e4m3);
    printf("FP32 -> E4M3:  Scalar: %.3f ms, SVE: %.3f ms, Speedup: %.1fx\n",
           scalar_fp32_e4m3 * 1000, sve_fp32_e4m3 * 1000, scalar_fp32_e4m3 / sve_fp32_e4m3);

    // Cleanup
    free(fp16_data);
    free(fp32_data);
    free(e4m3_out);
    free(e5m2_out);

    printf("\nDone.\n");
    return 0;
}
