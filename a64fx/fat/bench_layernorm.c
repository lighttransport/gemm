// bench_layernorm.c
// Benchmark for LayerNorm and RMSNorm implementations
// Tests correctness and performance of intrinsics and assembly versions

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "pcg32.h"

// ============================================
// Assembly function declarations
// ============================================

// RMSNorm FP32
extern void rmsnorm_f32_asm(const float* input, const float* gamma,
                            float* output, size_t dim, float eps);
extern void rmsnorm_f32_fast_asm(const float* input, const float* gamma,
                                 float* output, size_t dim, float eps);

// RMSNorm FP64
extern void rmsnorm_f64_asm(const double* input, const double* gamma,
                            double* output, size_t dim, double eps);

// LayerNorm FP32
extern void layernorm_f32_asm(const float* input, const float* gamma,
                              const float* beta, float* output, size_t dim, float eps);

// LayerNorm FP64
extern void layernorm_f64_asm(const double* input, const double* gamma,
                              const double* beta, double* output, size_t dim, double eps);

// ============================================
// Intrinsics function declarations
// ============================================

extern void rmsnorm_f32_intrin(const float* input, const float* gamma,
                               float* output, size_t dim, float eps);
extern void rmsnorm_f32_unroll4_intrin(const float* input, const float* gamma,
                                       float* output, size_t dim, float eps);
extern void layernorm_f32_intrin(const float* input, const float* gamma,
                                 const float* beta, float* output, size_t dim, float eps);
extern void rmsnorm_f64_intrin(const double* input, const double* gamma,
                               double* output, size_t dim, double eps);
extern void layernorm_f64_intrin(const double* input, const double* gamma,
                                 const double* beta, double* output, size_t dim, double eps);

// ============================================
// Cycle counter (100 MHz on A64FX)
// ============================================
static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// ============================================
// Reference implementations
// ============================================

static void rmsnorm_f32_ref(const float* input, const float* gamma,
                            float* output, size_t dim, float eps) {
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum_sq += input[i] * input[i];
    }

    // mean_sq = sum_sq / dim
    float mean_sq = sum_sq / (float)dim;

    // inv_std = 1 / sqrt(mean_sq + eps)
    float inv_std = 1.0f / sqrtf(mean_sq + eps);

    // output = input * gamma * inv_std
    for (size_t i = 0; i < dim; i++) {
        output[i] = input[i] * gamma[i] * inv_std;
    }
}

static void rmsnorm_f64_ref(const double* input, const double* gamma,
                            double* output, size_t dim, double eps) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < dim; i++) {
        sum_sq += input[i] * input[i];
    }

    double mean_sq = sum_sq / (double)dim;
    double inv_std = 1.0 / sqrt(mean_sq + eps);

    for (size_t i = 0; i < dim; i++) {
        output[i] = input[i] * gamma[i] * inv_std;
    }
}

static void layernorm_f32_ref(const float* input, const float* gamma,
                              const float* beta, float* output, size_t dim, float eps) {
    // Compute mean
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum += input[i];
    }
    float mean = sum / (float)dim;

    // Compute variance
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = input[i] - mean;
        sum_sq += diff * diff;
    }
    float variance = sum_sq / (float)dim;

    // inv_std = 1 / sqrt(variance + eps)
    float inv_std = 1.0f / sqrtf(variance + eps);

    // output = (input - mean) * inv_std * gamma + beta
    for (size_t i = 0; i < dim; i++) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

static void layernorm_f64_ref(const double* input, const double* gamma,
                              const double* beta, double* output, size_t dim, double eps) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; i++) {
        sum += input[i];
    }
    double mean = sum / (double)dim;

    double sum_sq = 0.0;
    for (size_t i = 0; i < dim; i++) {
        double diff = input[i] - mean;
        sum_sq += diff * diff;
    }
    double variance = sum_sq / (double)dim;
    double inv_std = 1.0 / sqrt(variance + eps);

    for (size_t i = 0; i < dim; i++) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ============================================
// Correctness checking
// ============================================

static int check_rmsnorm_f32(const char* name,
                             void (*fn)(const float*, const float*, float*, size_t, float),
                             size_t dim, float eps, float tol) {
    float* input = aligned_alloc(64, dim * sizeof(float));
    float* gamma = aligned_alloc(64, dim * sizeof(float));
    float* output = aligned_alloc(64, dim * sizeof(float));
    float* ref = aligned_alloc(64, dim * sizeof(float));

    if (!input || !gamma || !output || !ref) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize with random values
    for (size_t i = 0; i < dim; i++) {
        input[i] = pcg32_float() * 2.0f - 1.0f;
        gamma[i] = pcg32_float() * 0.5f + 0.75f;  // [0.75, 1.25]
    }

    // Compute reference
    rmsnorm_f32_ref(input, gamma, ref, dim, eps);

    // Compute using test function
    fn(input, gamma, output, dim, eps);

    // Check results
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    int errors = 0;

    for (size_t i = 0; i < dim; i++) {
        float abs_err = fabsf(output[i] - ref[i]);
        float rel_err = (ref[i] != 0.0f) ? abs_err / fabsf(ref[i]) : abs_err;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;

        if (rel_err > tol && abs_err > 1e-6f) {
            if (errors < 5) {
                printf("    Error at %zu: expected=%.6f, got=%.6f, rel_err=%.2e\n",
                       i, ref[i], output[i], rel_err);
            }
            errors++;
        }
    }

    printf("  %-20s: max_rel=%.2e, max_abs=%.2e, errors=%d/%zu %s\n",
           name, max_rel_err, max_abs_err, errors, dim,
           (errors == 0) ? "[PASS]" : "[FAIL]");

    free(input);
    free(gamma);
    free(output);
    free(ref);

    return (errors == 0) ? 0 : -1;
}

static int check_rmsnorm_f64(const char* name,
                             void (*fn)(const double*, const double*, double*, size_t, double),
                             size_t dim, double eps, double tol) {
    double* input = aligned_alloc(64, dim * sizeof(double));
    double* gamma = aligned_alloc(64, dim * sizeof(double));
    double* output = aligned_alloc(64, dim * sizeof(double));
    double* ref = aligned_alloc(64, dim * sizeof(double));

    if (!input || !gamma || !output || !ref) {
        printf("Memory allocation failed\n");
        return -1;
    }

    for (size_t i = 0; i < dim; i++) {
        input[i] = pcg32_double() * 2.0 - 1.0;
        gamma[i] = pcg32_double() * 0.5 + 0.75;
    }

    rmsnorm_f64_ref(input, gamma, ref, dim, eps);
    fn(input, gamma, output, dim, eps);

    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    int errors = 0;

    for (size_t i = 0; i < dim; i++) {
        double abs_err = fabs(output[i] - ref[i]);
        double rel_err = (ref[i] != 0.0) ? abs_err / fabs(ref[i]) : abs_err;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;

        if (rel_err > tol && abs_err > 1e-12) {
            if (errors < 5) {
                printf("    Error at %zu: expected=%.10f, got=%.10f, rel_err=%.2e\n",
                       i, ref[i], output[i], rel_err);
            }
            errors++;
        }
    }

    printf("  %-20s: max_rel=%.2e, max_abs=%.2e, errors=%d/%zu %s\n",
           name, max_rel_err, max_abs_err, errors, dim,
           (errors == 0) ? "[PASS]" : "[FAIL]");

    free(input);
    free(gamma);
    free(output);
    free(ref);

    return (errors == 0) ? 0 : -1;
}

static int check_layernorm_f32(const char* name,
                               void (*fn)(const float*, const float*, const float*, float*, size_t, float),
                               size_t dim, float eps, float tol) {
    float* input = aligned_alloc(64, dim * sizeof(float));
    float* gamma = aligned_alloc(64, dim * sizeof(float));
    float* beta = aligned_alloc(64, dim * sizeof(float));
    float* output = aligned_alloc(64, dim * sizeof(float));
    float* ref = aligned_alloc(64, dim * sizeof(float));

    if (!input || !gamma || !beta || !output || !ref) {
        printf("Memory allocation failed\n");
        return -1;
    }

    for (size_t i = 0; i < dim; i++) {
        input[i] = pcg32_float() * 2.0f - 1.0f;
        gamma[i] = pcg32_float() * 0.5f + 0.75f;
        beta[i] = pcg32_float() * 0.2f - 0.1f;
    }

    layernorm_f32_ref(input, gamma, beta, ref, dim, eps);
    fn(input, gamma, beta, output, dim, eps);

    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    int errors = 0;

    for (size_t i = 0; i < dim; i++) {
        float abs_err = fabsf(output[i] - ref[i]);
        float rel_err = (ref[i] != 0.0f) ? abs_err / fabsf(ref[i]) : abs_err;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;

        if (rel_err > tol && abs_err > 1e-6f) {
            if (errors < 5) {
                printf("    Error at %zu: expected=%.6f, got=%.6f, rel_err=%.2e\n",
                       i, ref[i], output[i], rel_err);
            }
            errors++;
        }
    }

    printf("  %-20s: max_rel=%.2e, max_abs=%.2e, errors=%d/%zu %s\n",
           name, max_rel_err, max_abs_err, errors, dim,
           (errors == 0) ? "[PASS]" : "[FAIL]");

    free(input);
    free(gamma);
    free(beta);
    free(output);
    free(ref);

    return (errors == 0) ? 0 : -1;
}

static int check_layernorm_f64(const char* name,
                               void (*fn)(const double*, const double*, const double*, double*, size_t, double),
                               size_t dim, double eps, double tol) {
    double* input = aligned_alloc(64, dim * sizeof(double));
    double* gamma = aligned_alloc(64, dim * sizeof(double));
    double* beta = aligned_alloc(64, dim * sizeof(double));
    double* output = aligned_alloc(64, dim * sizeof(double));
    double* ref = aligned_alloc(64, dim * sizeof(double));

    if (!input || !gamma || !beta || !output || !ref) {
        printf("Memory allocation failed\n");
        return -1;
    }

    for (size_t i = 0; i < dim; i++) {
        input[i] = pcg32_double() * 2.0 - 1.0;
        gamma[i] = pcg32_double() * 0.5 + 0.75;
        beta[i] = pcg32_double() * 0.2 - 0.1;
    }

    layernorm_f64_ref(input, gamma, beta, ref, dim, eps);
    fn(input, gamma, beta, output, dim, eps);

    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    int errors = 0;

    for (size_t i = 0; i < dim; i++) {
        double abs_err = fabs(output[i] - ref[i]);
        double rel_err = (ref[i] != 0.0) ? abs_err / fabs(ref[i]) : abs_err;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;

        if (rel_err > tol && abs_err > 1e-12) {
            if (errors < 5) {
                printf("    Error at %zu: expected=%.10f, got=%.10f, rel_err=%.2e\n",
                       i, ref[i], output[i], rel_err);
            }
            errors++;
        }
    }

    printf("  %-20s: max_rel=%.2e, max_abs=%.2e, errors=%d/%zu %s\n",
           name, max_rel_err, max_abs_err, errors, dim,
           (errors == 0) ? "[PASS]" : "[FAIL]");

    free(input);
    free(gamma);
    free(beta);
    free(output);
    free(ref);

    return (errors == 0) ? 0 : -1;
}

// ============================================
// Benchmark functions
// ============================================

static void bench_rmsnorm_f32(const char* name,
                              void (*fn)(const float*, const float*, float*, size_t, float),
                              size_t dim, float eps, int iterations) {
    float* input = aligned_alloc(64, dim * sizeof(float));
    float* gamma = aligned_alloc(64, dim * sizeof(float));
    float* output = aligned_alloc(64, dim * sizeof(float));

    if (!input || !gamma || !output) {
        printf("Memory allocation failed\n");
        return;
    }

    for (size_t i = 0; i < dim; i++) {
        input[i] = pcg32_float() * 2.0f - 1.0f;
        gamma[i] = 1.0f;
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        fn(input, gamma, output, dim, eps);
    }

    // Timed runs
    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        fn(input, gamma, output, dim, eps);
    }
    uint64_t end = get_cycles();

    double cycles = (double)(end - start);
    double cycles_per_elem = cycles / (iterations * dim);
    double cpu_cycles_per_elem = cycles_per_elem * 22.0;  // 100MHz counter, 2.2GHz CPU

    // Throughput calculation
    // RMSNorm: ~3 passes over data (sum_sq, normalize*gamma)
    // Per element: 1 mul (x*x), 1 mul (x*inv_std), 1 mul (*gamma) = ~3 FLOPs
    double elapsed_sec = cycles / 100e6;
    double throughput = (iterations * dim) / elapsed_sec;
    double gops = throughput / 1e9;

    // Bandwidth: read input 2x, gamma 1x, write output 1x = 4 * sizeof(float) = 16 bytes
    double bytes_per_elem = 4 * sizeof(float);
    double bandwidth = (iterations * dim * bytes_per_elem) / elapsed_sec / 1e9;

    printf("  %-20s: %.2f cyc/elem, %.2f Gelem/s, %.1f GB/s\n",
           name, cpu_cycles_per_elem, gops, bandwidth);

    free(input);
    free(gamma);
    free(output);
}

static void bench_rmsnorm_f64(const char* name,
                              void (*fn)(const double*, const double*, double*, size_t, double),
                              size_t dim, double eps, int iterations) {
    double* input = aligned_alloc(64, dim * sizeof(double));
    double* gamma = aligned_alloc(64, dim * sizeof(double));
    double* output = aligned_alloc(64, dim * sizeof(double));

    if (!input || !gamma || !output) {
        printf("Memory allocation failed\n");
        return;
    }

    for (size_t i = 0; i < dim; i++) {
        input[i] = pcg32_double() * 2.0 - 1.0;
        gamma[i] = 1.0;
    }

    for (int i = 0; i < 10; i++) {
        fn(input, gamma, output, dim, eps);
    }

    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        fn(input, gamma, output, dim, eps);
    }
    uint64_t end = get_cycles();

    double cycles = (double)(end - start);
    double cycles_per_elem = cycles / (iterations * dim);
    double cpu_cycles_per_elem = cycles_per_elem * 22.0;

    double elapsed_sec = cycles / 100e6;
    double throughput = (iterations * dim) / elapsed_sec;
    double gops = throughput / 1e9;

    double bytes_per_elem = 4 * sizeof(double);
    double bandwidth = (iterations * dim * bytes_per_elem) / elapsed_sec / 1e9;

    printf("  %-20s: %.2f cyc/elem, %.2f Gelem/s, %.1f GB/s\n",
           name, cpu_cycles_per_elem, gops, bandwidth);

    free(input);
    free(gamma);
    free(output);
}

static void bench_layernorm_f32(const char* name,
                                void (*fn)(const float*, const float*, const float*, float*, size_t, float),
                                size_t dim, float eps, int iterations) {
    float* input = aligned_alloc(64, dim * sizeof(float));
    float* gamma = aligned_alloc(64, dim * sizeof(float));
    float* beta = aligned_alloc(64, dim * sizeof(float));
    float* output = aligned_alloc(64, dim * sizeof(float));

    if (!input || !gamma || !beta || !output) {
        printf("Memory allocation failed\n");
        return;
    }

    for (size_t i = 0; i < dim; i++) {
        input[i] = pcg32_float() * 2.0f - 1.0f;
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
    }

    for (int i = 0; i < 10; i++) {
        fn(input, gamma, beta, output, dim, eps);
    }

    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        fn(input, gamma, beta, output, dim, eps);
    }
    uint64_t end = get_cycles();

    double cycles = (double)(end - start);
    double cycles_per_elem = cycles / (iterations * dim);
    double cpu_cycles_per_elem = cycles_per_elem * 22.0;

    double elapsed_sec = cycles / 100e6;
    double throughput = (iterations * dim) / elapsed_sec;
    double gops = throughput / 1e9;

    // LayerNorm: read input 3x, gamma 1x, beta 1x, write output 1x = 6 * sizeof(float)
    double bytes_per_elem = 6 * sizeof(float);
    double bandwidth = (iterations * dim * bytes_per_elem) / elapsed_sec / 1e9;

    printf("  %-20s: %.2f cyc/elem, %.2f Gelem/s, %.1f GB/s\n",
           name, cpu_cycles_per_elem, gops, bandwidth);

    free(input);
    free(gamma);
    free(beta);
    free(output);
}

static void bench_layernorm_f64(const char* name,
                                void (*fn)(const double*, const double*, const double*, double*, size_t, double),
                                size_t dim, double eps, int iterations) {
    double* input = aligned_alloc(64, dim * sizeof(double));
    double* gamma = aligned_alloc(64, dim * sizeof(double));
    double* beta = aligned_alloc(64, dim * sizeof(double));
    double* output = aligned_alloc(64, dim * sizeof(double));

    if (!input || !gamma || !beta || !output) {
        printf("Memory allocation failed\n");
        return;
    }

    for (size_t i = 0; i < dim; i++) {
        input[i] = pcg32_double() * 2.0 - 1.0;
        gamma[i] = 1.0;
        beta[i] = 0.0;
    }

    for (int i = 0; i < 10; i++) {
        fn(input, gamma, beta, output, dim, eps);
    }

    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        fn(input, gamma, beta, output, dim, eps);
    }
    uint64_t end = get_cycles();

    double cycles = (double)(end - start);
    double cycles_per_elem = cycles / (iterations * dim);
    double cpu_cycles_per_elem = cycles_per_elem * 22.0;

    double elapsed_sec = cycles / 100e6;
    double throughput = (iterations * dim) / elapsed_sec;
    double gops = throughput / 1e9;

    double bytes_per_elem = 6 * sizeof(double);
    double bandwidth = (iterations * dim * bytes_per_elem) / elapsed_sec / 1e9;

    printf("  %-20s: %.2f cyc/elem, %.2f Gelem/s, %.1f GB/s\n",
           name, cpu_cycles_per_elem, gops, bandwidth);

    free(input);
    free(gamma);
    free(beta);
    free(output);
}

// ============================================
// Main
// ============================================

int main(int argc, char** argv) {
    int iterations = 10000;
    size_t dim = 4096;  // Typical hidden dimension

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) dim = atol(argv[2]);

    const float eps_f32 = 1e-5f;
    const double eps_f64 = 1e-5;

    printf("==============================================\n");
    printf("LayerNorm / RMSNorm SVE Benchmark (A64FX)\n");
    printf("==============================================\n");
    printf("Dimension: %zu\n", dim);
    printf("Iterations: %d\n", iterations);
    printf("FP32 data size: %.1f KB\n", dim * sizeof(float) / 1024.0);
    printf("FP64 data size: %.1f KB\n", dim * sizeof(double) / 1024.0);
    printf("\n");

    srand(42);

    // ========================================
    // FP32 Correctness Tests
    // ========================================
    printf("=== FP32 Correctness Tests ===\n");
    printf("RMSNorm FP32:\n");
    srand(42); check_rmsnorm_f32("intrin", rmsnorm_f32_intrin, dim, eps_f32, 1e-5f);
    srand(42); check_rmsnorm_f32("intrin_unroll4", rmsnorm_f32_unroll4_intrin, dim, eps_f32, 1e-5f);
    srand(42); check_rmsnorm_f32("asm", rmsnorm_f32_asm, dim, eps_f32, 1e-5f);
    srand(42); check_rmsnorm_f32("asm_fast (1NR)", rmsnorm_f32_fast_asm, dim, eps_f32, 5e-3f);  // Lower precision

    printf("\nLayerNorm FP32:\n");
    srand(42); check_layernorm_f32("intrin", layernorm_f32_intrin, dim, eps_f32, 1e-5f);
    srand(42); check_layernorm_f32("asm", layernorm_f32_asm, dim, eps_f32, 1e-5f);
    printf("\n");

    // ========================================
    // FP64 Correctness Tests
    // ========================================
    printf("=== FP64 Correctness Tests ===\n");
    printf("RMSNorm FP64:\n");
    srand(42); check_rmsnorm_f64("intrin", rmsnorm_f64_intrin, dim, eps_f64, 1e-10);
    srand(42); check_rmsnorm_f64("asm", rmsnorm_f64_asm, dim, eps_f64, 1e-10);

    printf("\nLayerNorm FP64:\n");
    srand(42); check_layernorm_f64("intrin", layernorm_f64_intrin, dim, eps_f64, 1e-10);
    srand(42); check_layernorm_f64("asm", layernorm_f64_asm, dim, eps_f64, 1e-10);
    printf("\n");

    // ========================================
    // FP32 Performance Benchmarks
    // ========================================
    printf("=== FP32 Performance Benchmarks ===\n");
    printf("RMSNorm FP32:\n");
    srand(42); bench_rmsnorm_f32("ref (scalar)", rmsnorm_f32_ref, dim, eps_f32, iterations);
    srand(42); bench_rmsnorm_f32("intrin", rmsnorm_f32_intrin, dim, eps_f32, iterations);
    srand(42); bench_rmsnorm_f32("intrin_unroll4", rmsnorm_f32_unroll4_intrin, dim, eps_f32, iterations);
    srand(42); bench_rmsnorm_f32("asm", rmsnorm_f32_asm, dim, eps_f32, iterations);
    srand(42); bench_rmsnorm_f32("asm_fast", rmsnorm_f32_fast_asm, dim, eps_f32, iterations);

    printf("\nLayerNorm FP32:\n");
    srand(42); bench_layernorm_f32("ref (scalar)", layernorm_f32_ref, dim, eps_f32, iterations);
    srand(42); bench_layernorm_f32("intrin", layernorm_f32_intrin, dim, eps_f32, iterations);
    srand(42); bench_layernorm_f32("asm", layernorm_f32_asm, dim, eps_f32, iterations);
    printf("\n");

    // ========================================
    // FP64 Performance Benchmarks
    // ========================================
    printf("=== FP64 Performance Benchmarks ===\n");
    printf("RMSNorm FP64:\n");
    srand(42); bench_rmsnorm_f64("ref (scalar)", rmsnorm_f64_ref, dim, eps_f64, iterations);
    srand(42); bench_rmsnorm_f64("intrin", rmsnorm_f64_intrin, dim, eps_f64, iterations);
    srand(42); bench_rmsnorm_f64("asm", rmsnorm_f64_asm, dim, eps_f64, iterations);

    printf("\nLayerNorm FP64:\n");
    srand(42); bench_layernorm_f64("ref (scalar)", layernorm_f64_ref, dim, eps_f64, iterations);
    srand(42); bench_layernorm_f64("intrin", layernorm_f64_intrin, dim, eps_f64, iterations);
    srand(42); bench_layernorm_f64("asm", layernorm_f64_asm, dim, eps_f64, iterations);
    printf("\n");

    // ========================================
    // Scaling Tests
    // ========================================
    printf("=== Scaling Tests (RMSNorm FP32 ASM) ===\n");
    size_t test_dims[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    for (int d = 0; d < 7; d++) {
        size_t test_dim = test_dims[d];
        char name[32];
        snprintf(name, sizeof(name), "dim=%zu", test_dim);
        srand(42);
        bench_rmsnorm_f32(name, rmsnorm_f32_asm, test_dim, eps_f32, iterations);
    }

    return 0;
}
