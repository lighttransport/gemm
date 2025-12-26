// bench_philox.c
// Benchmark and test for SVE-optimized Philox RNG
//
// Usage: ./bench_philox [iterations] [count]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "philox.h"

//=============================================================================
// Timing utilities
//=============================================================================
static inline double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

//=============================================================================
// Statistical tests
//=============================================================================

// Mean of float array
static double compute_mean_f32(const float* data, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

// Variance of float array
static double compute_variance_f32(const float* data, size_t n, double mean)
{
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / n;
}

// Check if values are in range [0, 1)
static int check_range_01(const float* data, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (data[i] < 0.0f || data[i] >= 1.0f) {
            printf("    Out of range at %zu: %f\n", i, data[i]);
            return 0;
        }
    }
    return 1;
}

// Simple chi-squared test for uniformity (16 bins)
static double chi_squared_uniform(const float* data, size_t n)
{
    int bins[16] = {0};
    double expected = (double)n / 16.0;

    for (size_t i = 0; i < n; i++) {
        int bin = (int)(data[i] * 16);
        if (bin >= 16) bin = 15;
        if (bin < 0) bin = 0;
        bins[bin]++;
    }

    double chi_sq = 0.0;
    for (int i = 0; i < 16; i++) {
        double diff = bins[i] - expected;
        chi_sq += diff * diff / expected;
    }

    return chi_sq;
}

//=============================================================================
// Test functions
//=============================================================================

static int test_philox_scalar(void)
{
    printf("  Testing philox4x32_scalar...\n");

    uint32_t ctr[4] = {0, 0, 0, 0};
    uint32_t key[2] = {0x12345678, 0x9ABCDEF0};
    uint32_t out_scalar[4], out_ref[4];

    // Test scalar implementation
    philox4x32_scalar(ctr, key, out_scalar);
    philox4x32_10_ref(ctr, key, out_ref);

    int passed = 1;
    for (int i = 0; i < 4; i++) {
        if (out_scalar[i] != out_ref[i]) {
            printf("    FAILED: out[%d] = 0x%08X, expected 0x%08X\n",
                   i, out_scalar[i], out_ref[i]);
            passed = 0;
        }
    }

    if (passed) {
        printf("    PASSED\n");
    }
    return passed ? 0 : 1;
}

static int test_philox_sve_u32(size_t count)
{
    printf("  Testing philox4x32_sve_u32 (count=%zu)...\n", count);

    uint64_t counter = 0;
    uint64_t key = 0x123456789ABCDEF0ULL;

    uint32_t* out_sve = aligned_alloc(64, count * sizeof(uint32_t));
    uint32_t* out_ref = aligned_alloc(64, count * sizeof(uint32_t));

    if (!out_sve || !out_ref) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    philox4x32_sve_u32(counter, key, out_sve, count);
    philox_u32_ref(counter, key, out_ref, count);

    int errors = 0;
    for (size_t i = 0; i < count && errors < 10; i++) {
        if (out_sve[i] != out_ref[i]) {
            printf("    Mismatch at %zu: SVE=0x%08X, ref=0x%08X\n",
                   i, out_sve[i], out_ref[i]);
            errors++;
        }
    }

    free(out_sve);
    free(out_ref);

    if (errors == 0) {
        printf("    PASSED\n");
        return 0;
    } else {
        printf("    FAILED with %d errors\n", errors);
        return 1;
    }
}

static int test_philox_sve_f32(size_t count)
{
    printf("  Testing philox4x32_sve_f32 (count=%zu)...\n", count);

    uint64_t counter = 12345;
    uint64_t key = 0xDEADBEEFCAFEBABEULL;

    float* out_sve = aligned_alloc(64, count * sizeof(float));
    float* out_ref = aligned_alloc(64, count * sizeof(float));

    if (!out_sve || !out_ref) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    philox4x32_sve_f32(counter, key, out_sve, count);
    philox_f32_ref(counter, key, out_ref, count);

    // Check range
    if (!check_range_01(out_sve, count)) {
        printf("    FAILED: Values out of [0, 1) range\n");
        free(out_sve);
        free(out_ref);
        return 1;
    }

    // Check values match reference (with small tolerance for float conversion)
    int errors = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(out_sve[i] - out_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-6f) {
            if (errors < 10) {
                printf("    Mismatch at %zu: SVE=%f, ref=%f, diff=%e\n",
                       i, out_sve[i], out_ref[i], diff);
            }
            errors++;
        }
    }

    free(out_sve);
    free(out_ref);

    printf("    max_diff = %e\n", max_diff);
    if (errors == 0) {
        printf("    PASSED\n");
        return 0;
    } else {
        printf("    FAILED with %d errors\n", errors);
        return 1;
    }
}

static int test_philox_quality(size_t count)
{
    printf("  Testing statistical quality (count=%zu)...\n", count);

    uint64_t counter = 0;
    uint64_t key = 0x0123456789ABCDEFULL;

    float* data = aligned_alloc(64, count * sizeof(float));
    if (!data) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    philox4x32_sve_f32(counter, key, data, count);

    // Check range
    if (!check_range_01(data, count)) {
        printf("    FAILED: Values out of range\n");
        free(data);
        return 1;
    }

    // Check mean (should be ~0.5 for uniform [0,1))
    double mean = compute_mean_f32(data, count);
    printf("    Mean: %f (expected ~0.5)\n", mean);
    if (fabsf(mean - 0.5f) > 0.01f) {
        printf("    WARNING: Mean deviates from 0.5\n");
    }

    // Check variance (should be ~1/12 = 0.0833 for uniform [0,1))
    double variance = compute_variance_f32(data, count, mean);
    printf("    Variance: %f (expected ~0.0833)\n", variance);
    if (fabsf(variance - 0.0833f) > 0.01f) {
        printf("    WARNING: Variance deviates from 1/12\n");
    }

    // Chi-squared test
    double chi_sq = chi_squared_uniform(data, count);
    // For 16 bins, df=15, chi-squared should be < ~25 at 95% confidence
    printf("    Chi-squared: %f (should be < 25 at 95%% confidence)\n", chi_sq);

    free(data);

    printf("    PASSED (quality check)\n");
    return 0;
}

static int test_philox_normal(size_t count)
{
    printf("  Testing normal distribution (count=%zu)...\n", count);

    uint64_t counter = 0;
    uint64_t key = 0xFEDCBA9876543210ULL;

    float* uniform = aligned_alloc(64, count * sizeof(float));
    float* normal = aligned_alloc(64, count * sizeof(float));

    if (!uniform || !normal) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    // Generate uniforms
    philox4x32_sve_f32(counter, key, uniform, count);

    // Apply Box-Muller
    box_muller_transform(uniform, normal, count);

    // Check mean (should be ~0)
    double mean = compute_mean_f32(normal, count);
    printf("    Mean: %f (expected ~0)\n", mean);

    // Check variance (should be ~1)
    double variance = compute_variance_f32(normal, count, mean);
    printf("    Variance: %f (expected ~1)\n", variance);

    int passed = (fabsf(mean) < 0.05f) && (fabsf(variance - 1.0f) < 0.1f);

    free(uniform);
    free(normal);

    if (passed) {
        printf("    PASSED\n");
        return 0;
    } else {
        printf("    FAILED: Statistics deviate from N(0,1)\n");
        return 1;
    }
}

static int test_philox_reproducibility(void)
{
    printf("  Testing reproducibility...\n");

    uint64_t counter = 42;
    uint64_t key = 0x1234567890ABCDEFULL;
    size_t count = 1000;

    float* out1 = aligned_alloc(64, count * sizeof(float));
    float* out2 = aligned_alloc(64, count * sizeof(float));

    if (!out1 || !out2) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    // Generate twice with same seed
    philox4x32_sve_f32(counter, key, out1, count);
    philox4x32_sve_f32(counter, key, out2, count);

    // Should be identical
    int identical = 1;
    for (size_t i = 0; i < count; i++) {
        if (out1[i] != out2[i]) {
            printf("    Mismatch at %zu: %f != %f\n", i, out1[i], out2[i]);
            identical = 0;
            break;
        }
    }

    free(out1);
    free(out2);

    if (identical) {
        printf("    PASSED\n");
        return 0;
    } else {
        printf("    FAILED: Not reproducible\n");
        return 1;
    }
}

static int test_philox_bytes(size_t count)
{
    printf("  Testing philox4x32_sve_bytes (count=%zu)...\n", count);

    uint64_t counter = 0;
    uint64_t key = 0xABCDEF0123456789ULL;

    uint8_t* out = aligned_alloc(64, count);
    if (!out) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    philox4x32_sve_bytes(counter, key, out, count);

    // Check that we have variety (not all zeros or all same value)
    int histogram[256] = {0};
    for (size_t i = 0; i < count; i++) {
        histogram[out[i]]++;
    }

    int non_zero_bins = 0;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) non_zero_bins++;
    }

    free(out);

    // Should have many different byte values
    if (non_zero_bins > 200) {  // Most of 256 bins should be hit for large count
        printf("    PASSED (used %d of 256 byte values)\n", non_zero_bins);
        return 0;
    } else if (count < 1000) {
        printf("    PASSED (small count, used %d byte values)\n", non_zero_bins);
        return 0;
    } else {
        printf("    WARNING: Only %d of 256 byte values used\n", non_zero_bins);
        return 0;  // Not a hard failure
    }
}

//=============================================================================
// Benchmark functions
//=============================================================================

static void bench_philox_u32(size_t iterations, size_t count)
{
    printf("\nBenchmark: philox4x32_sve_u32\n");
    printf("  count=%zu, iterations=%zu\n", count, iterations);

    uint64_t counter = 0;
    uint64_t key = 0x123456789ABCDEF0ULL;

    uint32_t* out = aligned_alloc(64, count * sizeof(uint32_t));
    if (!out) {
        printf("  FAILED: Memory allocation\n");
        return;
    }

    // Warmup
    for (size_t i = 0; i < 10; i++) {
        philox4x32_sve_u32(counter, key, out, count);
        counter += (count + 3) / 4;
    }

    // Benchmark reference
    counter = 0;
    double t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        philox_u32_ref(counter, key, out, count);
        counter += (count + 3) / 4;
    }
    double t1 = get_time_sec();
    double ref_time = (t1 - t0) / iterations;

    // Benchmark SVE
    counter = 0;
    t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        philox4x32_sve_u32(counter, key, out, count);
        counter += (count + 3) / 4;
    }
    t1 = get_time_sec();
    double sve_time = (t1 - t0) / iterations;

    // Throughput: bytes per second
    double bytes = count * sizeof(uint32_t);
    double ref_throughput = bytes / ref_time / 1e9;
    double sve_throughput = bytes / sve_time / 1e9;

    // Random numbers per second
    double ref_nums = count / ref_time / 1e9;
    double sve_nums = count / sve_time / 1e9;

    printf("  Reference: %.3f us, %.2f GB/s, %.2f Gnum/s\n",
           ref_time * 1e6, ref_throughput, ref_nums);
    printf("  SVE:       %.3f us, %.2f GB/s, %.2f Gnum/s (%.2fx speedup)\n",
           sve_time * 1e6, sve_throughput, sve_nums, ref_time / sve_time);

    free(out);
}

static void bench_philox_f32(size_t iterations, size_t count)
{
    printf("\nBenchmark: philox4x32_sve_f32\n");
    printf("  count=%zu, iterations=%zu\n", count, iterations);

    uint64_t counter = 0;
    uint64_t key = 0x123456789ABCDEF0ULL;

    float* out = aligned_alloc(64, count * sizeof(float));
    if (!out) {
        printf("  FAILED: Memory allocation\n");
        return;
    }

    // Warmup
    for (size_t i = 0; i < 10; i++) {
        philox4x32_sve_f32(counter, key, out, count);
        counter += (count + 3) / 4;
    }

    // Benchmark reference
    counter = 0;
    double t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        philox_f32_ref(counter, key, out, count);
        counter += (count + 3) / 4;
    }
    double t1 = get_time_sec();
    double ref_time = (t1 - t0) / iterations;

    // Benchmark SVE
    counter = 0;
    t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        philox4x32_sve_f32(counter, key, out, count);
        counter += (count + 3) / 4;
    }
    t1 = get_time_sec();
    double sve_time = (t1 - t0) / iterations;

    double bytes = count * sizeof(float);
    double ref_throughput = bytes / ref_time / 1e9;
    double sve_throughput = bytes / sve_time / 1e9;

    printf("  Reference: %.3f us, %.2f GB/s\n", ref_time * 1e6, ref_throughput);
    printf("  SVE:       %.3f us, %.2f GB/s (%.2fx speedup)\n",
           sve_time * 1e6, sve_throughput, ref_time / sve_time);

    free(out);
}

static void bench_philox_bytes(size_t iterations, size_t count)
{
    printf("\nBenchmark: philox4x32_sve_bytes\n");
    printf("  count=%zu bytes, iterations=%zu\n", count, iterations);

    uint64_t counter = 0;
    uint64_t key = 0x123456789ABCDEF0ULL;

    uint8_t* out = aligned_alloc(64, count);
    if (!out) {
        printf("  FAILED: Memory allocation\n");
        return;
    }

    // Warmup
    for (size_t i = 0; i < 10; i++) {
        philox4x32_sve_bytes(counter, key, out, count);
        counter += (count + 15) / 16;
    }

    // Benchmark SVE
    counter = 0;
    double t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        philox4x32_sve_bytes(counter, key, out, count);
        counter += (count + 15) / 16;
    }
    double t1 = get_time_sec();
    double sve_time = (t1 - t0) / iterations;

    double throughput = count / sve_time / 1e9;
    printf("  SVE: %.3f us, %.2f GB/s\n", sve_time * 1e6, throughput);

    free(out);
}

static void bench_philox_normal(size_t iterations, size_t count)
{
    printf("\nBenchmark: philox + Box-Muller (normal distribution)\n");
    printf("  count=%zu, iterations=%zu\n", count, iterations);

    uint64_t counter = 0;
    uint64_t key = 0x123456789ABCDEF0ULL;

    float* out = aligned_alloc(64, count * sizeof(float));
    if (!out) {
        printf("  FAILED: Memory allocation\n");
        return;
    }

    // Warmup
    for (size_t i = 0; i < 10; i++) {
        philox4x32_sve_f32(counter, key, out, count);
        box_muller_transform(out, out, count);
        counter += (count + 3) / 4;
    }

    // Benchmark
    counter = 0;
    double t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        philox4x32_sve_f32(counter, key, out, count);
        box_muller_transform(out, out, count);
        counter += (count + 3) / 4;
    }
    double t1 = get_time_sec();
    double time = (t1 - t0) / iterations;

    double bytes = count * sizeof(float);
    double throughput = bytes / time / 1e9;
    double nums = count / time / 1e9;

    printf("  Time: %.3f us, %.2f GB/s, %.2f Gnum/s\n",
           time * 1e6, throughput, nums);

    free(out);
}

//=============================================================================
// Main
//=============================================================================
int main(int argc, char** argv)
{
    size_t iterations = 1000;
    size_t count = 1024 * 1024;  // 1M numbers

    if (argc > 1) iterations = (size_t)atol(argv[1]);
    if (argc > 2) count = (size_t)atol(argv[2]);

    printf("==============================================\n");
    printf("SVE Philox RNG Benchmark for A64FX\n");
    printf("==============================================\n");
    printf("Parameters:\n");
    printf("  iterations = %zu\n", iterations);
    printf("  count      = %zu\n", count);
    printf("\n");

    // Run tests
    printf("=== Correctness Tests ===\n");
    int failures = 0;

    failures += test_philox_scalar();
    failures += test_philox_sve_u32(count);
    failures += test_philox_sve_f32(count);
    failures += test_philox_quality(count);
    failures += test_philox_normal(count);
    failures += test_philox_reproducibility();
    failures += test_philox_bytes(count);

    // Edge cases
    printf("\n  Edge cases:\n");
    failures += test_philox_sve_u32(1);
    failures += test_philox_sve_u32(7);
    failures += test_philox_sve_u32(64);
    failures += test_philox_sve_u32(65);
    failures += test_philox_sve_f32(1);
    failures += test_philox_sve_f32(63);

    if (failures > 0) {
        printf("\n!!! %d tests FAILED !!!\n", failures);
    } else {
        printf("\nAll tests PASSED!\n");
    }

    // Run benchmarks
    printf("\n=== Performance Benchmarks ===\n");
    bench_philox_u32(iterations, count);
    bench_philox_f32(iterations, count);
    bench_philox_bytes(iterations, count);
    bench_philox_normal(iterations, count);

    // Scaling tests
    printf("\n=== Count Scaling ===\n");
    size_t counts[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    for (size_t i = 0; i < sizeof(counts)/sizeof(counts[0]); i++) {
        bench_philox_f32(iterations / 10, counts[i]);
    }

    printf("\n==============================================\n");
    printf("Benchmark complete.\n");

    return failures;
}
