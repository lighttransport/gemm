// bench_activation.c
// Benchmark for activation functions: GELU, GELU(tanh), QuickGELU, SiLU
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

extern void silu_f32_asm(const float* input, float* output, size_t n);
extern void silu_f64_asm(const double* input, double* output, size_t n);
extern void quickgelu_f32_asm(const float* input, float* output, size_t n);
extern void quickgelu_f64_asm(const double* input, double* output, size_t n);
extern void gelu_tanh_f32_asm(const float* input, float* output, size_t n);
extern void gelu_tanh_f64_asm(const double* input, double* output, size_t n);
extern void swiglu_f32_asm(const float* x, const float* gate, float* output, size_t n);
extern void swiglu_f64_asm(const double* x, const double* gate, double* output, size_t n);

// ============================================
// Intrinsics function declarations
// ============================================

extern void gelu_f32_intrin(const float* input, float* output, size_t n);
extern void gelu_f64_intrin(const double* input, double* output, size_t n);
extern void gelu_tanh_f32_intrin(const float* input, float* output, size_t n);
extern void gelu_tanh_f64_intrin(const double* input, double* output, size_t n);
extern void quickgelu_f32_intrin(const float* input, float* output, size_t n);
extern void quickgelu_f64_intrin(const double* input, double* output, size_t n);
extern void silu_f32_intrin(const float* input, float* output, size_t n);
extern void silu_f64_intrin(const double* input, double* output, size_t n);
extern void silu_f32_unroll2_intrin(const float* input, float* output, size_t n);
extern void quickgelu_f32_unroll2_intrin(const float* input, float* output, size_t n);
extern void swiglu_f32_intrin(const float* x, const float* gate, float* output, size_t n);
extern void swiglu_f64_intrin(const double* x, const double* gate, double* output, size_t n);
extern void swiglu_f32_unroll2_intrin(const float* x, const float* gate, float* output, size_t n);

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

static float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static double sigmoid_f64(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
static void gelu_f32_ref(const float* input, float* output, size_t n) {
    const float inv_sqrt2 = 0.7071067811865476f;
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt2));
    }
}

static void gelu_f64_ref(const double* input, double* output, size_t n) {
    const double inv_sqrt2 = 0.7071067811865476;
    for (size_t i = 0; i < n; i++) {
        double x = input[i];
        output[i] = 0.5 * x * (1.0 + erf(x * inv_sqrt2));
    }
}

// GELU tanh approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static void gelu_tanh_f32_ref(const float* input, float* output, size_t n) {
    const float sqrt_2_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_pi * (x + coef * x3);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

static void gelu_tanh_f64_ref(const double* input, double* output, size_t n) {
    const double sqrt_2_pi = 0.7978845608028654;
    const double coef = 0.044715;
    for (size_t i = 0; i < n; i++) {
        double x = input[i];
        double x3 = x * x * x;
        double inner = sqrt_2_pi * (x + coef * x3);
        output[i] = 0.5 * x * (1.0 + tanh(inner));
    }
}

// QuickGELU: x * sigmoid(1.702 * x)
static void quickgelu_f32_ref(const float* input, float* output, size_t n) {
    const float alpha = 1.702f;
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x * sigmoid_f32(alpha * x);
    }
}

static void quickgelu_f64_ref(const double* input, double* output, size_t n) {
    const double alpha = 1.702;
    for (size_t i = 0; i < n; i++) {
        double x = input[i];
        output[i] = x * sigmoid_f64(alpha * x);
    }
}

// SiLU (Swish): x * sigmoid(x)
static void silu_f32_ref(const float* input, float* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x * sigmoid_f32(x);
    }
}

static void silu_f64_ref(const double* input, double* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double x = input[i];
        output[i] = x * sigmoid_f64(x);
    }
}

// SwiGLU: x * gate * sigmoid(gate) = x * SiLU(gate)
static void swiglu_f32_ref(const float* x, const float* gate, float* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float g = gate[i];
        float silu_g = g * sigmoid_f32(g);
        output[i] = x[i] * silu_g;
    }
}

static void swiglu_f64_ref(const double* x, const double* gate, double* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double g = gate[i];
        double silu_g = g * sigmoid_f64(g);
        output[i] = x[i] * silu_g;
    }
}

// ============================================
// Correctness checking
// ============================================

typedef void (*activation_f32_fn)(const float*, float*, size_t);
typedef void (*activation_f64_fn)(const double*, double*, size_t);

static int check_f32(const char* name, activation_f32_fn fn, activation_f32_fn ref_fn,
                     size_t n, float tol) {
    float* input = aligned_alloc(64, n * sizeof(float));
    float* output = aligned_alloc(64, n * sizeof(float));
    float* ref = aligned_alloc(64, n * sizeof(float));

    if (!input || !output || !ref) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize with values in range [-3, 3] to test all regions
    for (size_t i = 0; i < n; i++) {
        input[i] = pcg32_float() * 6.0f - 3.0f;
    }

    ref_fn(input, ref, n);
    fn(input, output, n);

    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    int errors = 0;

    for (size_t i = 0; i < n; i++) {
        float abs_err = fabsf(output[i] - ref[i]);
        float denom = fabsf(ref[i]);
        float rel_err = (denom > 1e-10f) ? (abs_err / denom) : abs_err;

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        if (rel_err > tol && abs_err > 1e-4f) {
            if (errors < 3) {
                printf("    Error at %zu: x=%.6f expected=%.6f got=%.6f rel_err=%.2e\n",
                       i, input[i], ref[i], output[i], rel_err);
            }
            errors++;
        }
    }

    printf("  %-24s: max_rel=%.2e, max_abs=%.2e, errors=%d/%zu %s\n",
           name, max_rel_err, max_abs_err, errors, n,
           (errors == 0) ? "[PASS]" : "[FAIL]");

    free(input);
    free(output);
    free(ref);

    return (errors == 0) ? 0 : -1;
}

static int check_f64(const char* name, activation_f64_fn fn, activation_f64_fn ref_fn,
                     size_t n, double tol) {
    double* input = aligned_alloc(64, n * sizeof(double));
    double* output = aligned_alloc(64, n * sizeof(double));
    double* ref = aligned_alloc(64, n * sizeof(double));

    if (!input || !output || !ref) {
        printf("Memory allocation failed\n");
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        input[i] = pcg32_double() * 6.0 - 3.0;
    }

    ref_fn(input, ref, n);
    fn(input, output, n);

    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    int errors = 0;

    for (size_t i = 0; i < n; i++) {
        double abs_err = fabs(output[i] - ref[i]);
        double denom = fabs(ref[i]);
        double rel_err = (denom > 1e-15) ? (abs_err / denom) : abs_err;

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        if (rel_err > tol && abs_err > 1e-5) {
            if (errors < 3) {
                printf("    Error at %zu: x=%.10f expected=%.10f got=%.10f rel_err=%.2e\n",
                       i, input[i], ref[i], output[i], rel_err);
            }
            errors++;
        }
    }

    printf("  %-24s: max_rel=%.2e, max_abs=%.2e, errors=%d/%zu %s\n",
           name, max_rel_err, max_abs_err, errors, n,
           (errors == 0) ? "[PASS]" : "[FAIL]");

    free(input);
    free(output);
    free(ref);

    return (errors == 0) ? 0 : -1;
}

// SwiGLU check function (2 inputs)
typedef void (*swiglu_f32_fn)(const float*, const float*, float*, size_t);
typedef void (*swiglu_f64_fn)(const double*, const double*, double*, size_t);

static int check_swiglu_f32(const char* name, swiglu_f32_fn fn,
                            size_t n, float tol) {
    float* x = aligned_alloc(64, n * sizeof(float));
    float* gate = aligned_alloc(64, n * sizeof(float));
    float* output = aligned_alloc(64, n * sizeof(float));
    float* ref = aligned_alloc(64, n * sizeof(float));

    if (!x || !gate || !output || !ref) {
        printf("Memory allocation failed\n");
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        x[i] = pcg32_float() * 6.0f - 3.0f;
        gate[i] = pcg32_float() * 6.0f - 3.0f;
    }

    swiglu_f32_ref(x, gate, ref, n);
    fn(x, gate, output, n);

    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    int errors = 0;

    for (size_t i = 0; i < n; i++) {
        float abs_err = fabsf(output[i] - ref[i]);
        float denom = fabsf(ref[i]);
        float rel_err = (denom > 1e-10f) ? (abs_err / denom) : abs_err;

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        if (rel_err > tol && abs_err > 1e-4f) {
            if (errors < 3) {
                printf("    Error at %zu: x=%.6f gate=%.6f expected=%.6f got=%.6f rel_err=%.2e\n",
                       i, x[i], gate[i], ref[i], output[i], rel_err);
            }
            errors++;
        }
    }

    printf("  %-24s: max_rel=%.2e, max_abs=%.2e, errors=%d/%zu %s\n",
           name, max_rel_err, max_abs_err, errors, n,
           (errors == 0) ? "[PASS]" : "[FAIL]");

    free(x);
    free(gate);
    free(output);
    free(ref);

    return (errors == 0) ? 0 : -1;
}

static int check_swiglu_f64(const char* name, swiglu_f64_fn fn,
                            size_t n, double tol) {
    double* x = aligned_alloc(64, n * sizeof(double));
    double* gate = aligned_alloc(64, n * sizeof(double));
    double* output = aligned_alloc(64, n * sizeof(double));
    double* ref = aligned_alloc(64, n * sizeof(double));

    if (!x || !gate || !output || !ref) {
        printf("Memory allocation failed\n");
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        x[i] = pcg32_double() * 6.0 - 3.0;
        gate[i] = pcg32_double() * 6.0 - 3.0;
    }

    swiglu_f64_ref(x, gate, ref, n);
    fn(x, gate, output, n);

    double max_rel_err = 0.0;
    double max_abs_err = 0.0;
    int errors = 0;

    for (size_t i = 0; i < n; i++) {
        double abs_err = fabs(output[i] - ref[i]);
        double denom = fabs(ref[i]);
        double rel_err = (denom > 1e-15) ? (abs_err / denom) : abs_err;

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        if (rel_err > tol && abs_err > 1e-5) {
            if (errors < 3) {
                printf("    Error at %zu: x=%.10f gate=%.10f expected=%.10f got=%.10f rel_err=%.2e\n",
                       i, x[i], gate[i], ref[i], output[i], rel_err);
            }
            errors++;
        }
    }

    printf("  %-24s: max_rel=%.2e, max_abs=%.2e, errors=%d/%zu %s\n",
           name, max_rel_err, max_abs_err, errors, n,
           (errors == 0) ? "[PASS]" : "[FAIL]");

    free(x);
    free(gate);
    free(output);
    free(ref);

    return (errors == 0) ? 0 : -1;
}

// ============================================
// Benchmark functions
// ============================================

static void bench_f32(const char* name, activation_f32_fn fn,
                      size_t n, int iterations) {
    float* input = aligned_alloc(64, n * sizeof(float));
    float* output = aligned_alloc(64, n * sizeof(float));

    if (!input || !output) {
        printf("Memory allocation failed\n");
        return;
    }

    for (size_t i = 0; i < n; i++) {
        input[i] = pcg32_float() * 6.0f - 3.0f;
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        fn(input, output, n);
    }

    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        fn(input, output, n);
    }
    uint64_t end = get_cycles();

    double cycles = (double)(end - start);
    double cycles_per_elem = cycles / (iterations * n);
    double cpu_cycles = cycles_per_elem * 22.0;  // 100MHz counter, 2.2GHz CPU

    double elapsed_sec = cycles / 100e6;
    double throughput = (iterations * n) / elapsed_sec / 1e9;

    // Bandwidth: read + write = 2 * sizeof(float) = 8 bytes per element
    double bandwidth = (iterations * n * 2 * sizeof(float)) / elapsed_sec / 1e9;

    printf("  %-24s: %.2f cyc/elem, %.2f Gelem/s, %.1f GB/s\n",
           name, cpu_cycles, throughput, bandwidth);

    free(input);
    free(output);
}

static void bench_f64(const char* name, activation_f64_fn fn,
                      size_t n, int iterations) {
    double* input = aligned_alloc(64, n * sizeof(double));
    double* output = aligned_alloc(64, n * sizeof(double));

    if (!input || !output) {
        printf("Memory allocation failed\n");
        return;
    }

    for (size_t i = 0; i < n; i++) {
        input[i] = pcg32_double() * 6.0 - 3.0;
    }

    for (int i = 0; i < 10; i++) {
        fn(input, output, n);
    }

    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        fn(input, output, n);
    }
    uint64_t end = get_cycles();

    double cycles = (double)(end - start);
    double cycles_per_elem = cycles / (iterations * n);
    double cpu_cycles = cycles_per_elem * 22.0;

    double elapsed_sec = cycles / 100e6;
    double throughput = (iterations * n) / elapsed_sec / 1e9;
    double bandwidth = (iterations * n * 2 * sizeof(double)) / elapsed_sec / 1e9;

    printf("  %-24s: %.2f cyc/elem, %.2f Gelem/s, %.1f GB/s\n",
           name, cpu_cycles, throughput, bandwidth);

    free(input);
    free(output);
}

// SwiGLU benchmark (2 inputs)
static void bench_swiglu_f32(const char* name, swiglu_f32_fn fn,
                              size_t n, int iterations) {
    float* x = aligned_alloc(64, n * sizeof(float));
    float* gate = aligned_alloc(64, n * sizeof(float));
    float* output = aligned_alloc(64, n * sizeof(float));

    if (!x || !gate || !output) {
        printf("Memory allocation failed\n");
        return;
    }

    for (size_t i = 0; i < n; i++) {
        x[i] = pcg32_float() * 6.0f - 3.0f;
        gate[i] = pcg32_float() * 6.0f - 3.0f;
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        fn(x, gate, output, n);
    }

    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        fn(x, gate, output, n);
    }
    uint64_t end = get_cycles();

    double cycles = (double)(end - start);
    double cycles_per_elem = cycles / (iterations * n);
    double cpu_cycles = cycles_per_elem * 22.0;

    double elapsed_sec = cycles / 100e6;
    double throughput = (iterations * n) / elapsed_sec / 1e9;

    // Bandwidth: 2 reads + 1 write = 3 * sizeof(float) = 12 bytes per element
    double bandwidth = (iterations * n * 3 * sizeof(float)) / elapsed_sec / 1e9;

    printf("  %-24s: %.2f cyc/elem, %.2f Gelem/s, %.1f GB/s\n",
           name, cpu_cycles, throughput, bandwidth);

    free(x);
    free(gate);
    free(output);
}

static void bench_swiglu_f64(const char* name, swiglu_f64_fn fn,
                              size_t n, int iterations) {
    double* x = aligned_alloc(64, n * sizeof(double));
    double* gate = aligned_alloc(64, n * sizeof(double));
    double* output = aligned_alloc(64, n * sizeof(double));

    if (!x || !gate || !output) {
        printf("Memory allocation failed\n");
        return;
    }

    for (size_t i = 0; i < n; i++) {
        x[i] = pcg32_double() * 6.0 - 3.0;
        gate[i] = pcg32_double() * 6.0 - 3.0;
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        fn(x, gate, output, n);
    }

    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        fn(x, gate, output, n);
    }
    uint64_t end = get_cycles();

    double cycles = (double)(end - start);
    double cycles_per_elem = cycles / (iterations * n);
    double cpu_cycles = cycles_per_elem * 22.0;

    double elapsed_sec = cycles / 100e6;
    double throughput = (iterations * n) / elapsed_sec / 1e9;

    // Bandwidth: 2 reads + 1 write = 3 * sizeof(double) = 24 bytes per element
    double bandwidth = (iterations * n * 3 * sizeof(double)) / elapsed_sec / 1e9;

    printf("  %-24s: %.2f cyc/elem, %.2f Gelem/s, %.1f GB/s\n",
           name, cpu_cycles, throughput, bandwidth);

    free(x);
    free(gate);
    free(output);
}

// ============================================
// Main
// ============================================

int main(int argc, char** argv) {
    int iterations = 10000;
    size_t n = 4096;

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) n = atol(argv[2]);

    printf("==============================================\n");
    printf("Activation Functions SVE Benchmark (A64FX)\n");
    printf("==============================================\n");
    printf("Size: %zu elements\n", n);
    printf("Iterations: %d\n", iterations);
    printf("FP32 data size: %.1f KB\n", n * sizeof(float) / 1024.0);
    printf("FP64 data size: %.1f KB\n", n * sizeof(double) / 1024.0);
    printf("\n");

    // ========================================
    // FP32 Correctness Tests
    // ========================================
    printf("=== FP32 Correctness Tests ===\n");

    printf("\nGELU (exact, erf-based):\n");
    srand(42); check_f32("intrin", gelu_f32_intrin, gelu_f32_ref, n, 1e-3f);

    printf("\nGELU (tanh approx):\n");
    srand(42); check_f32("intrin", gelu_tanh_f32_intrin, gelu_tanh_f32_ref, n, 1e-3f);
    srand(42); check_f32("asm", gelu_tanh_f32_asm, gelu_tanh_f32_ref, n, 1e-2f);

    printf("\nQuickGELU:\n");
    srand(42); check_f32("intrin", quickgelu_f32_intrin, quickgelu_f32_ref, n, 1e-3f);
    srand(42); check_f32("intrin_unroll2", quickgelu_f32_unroll2_intrin, quickgelu_f32_ref, n, 1e-3f);
    srand(42); check_f32("asm", quickgelu_f32_asm, quickgelu_f32_ref, n, 1e-2f);

    printf("\nSiLU (Swish):\n");
    srand(42); check_f32("intrin", silu_f32_intrin, silu_f32_ref, n, 1e-3f);
    srand(42); check_f32("intrin_unroll2", silu_f32_unroll2_intrin, silu_f32_ref, n, 1e-3f);
    srand(42); check_f32("asm", silu_f32_asm, silu_f32_ref, n, 1e-2f);

    printf("\nSwiGLU:\n");
    srand(42); check_swiglu_f32("intrin", swiglu_f32_intrin, n, 1e-3f);
    srand(42); check_swiglu_f32("intrin_unroll2", swiglu_f32_unroll2_intrin, n, 1e-3f);
    srand(42); check_swiglu_f32("asm", swiglu_f32_asm, n, 1e-2f);

    printf("\n");

    // ========================================
    // FP64 Correctness Tests
    // ========================================
    printf("=== FP64 Correctness Tests ===\n");

    printf("\nGELU (exact, erf-based):\n");
    srand(42); check_f64("intrin", gelu_f64_intrin, gelu_f64_ref, n, 1e-6);

    printf("\nGELU (tanh approx):\n");
    srand(42); check_f64("intrin", gelu_tanh_f64_intrin, gelu_tanh_f64_ref, n, 1e-6);
    srand(42); check_f64("asm", gelu_tanh_f64_asm, gelu_tanh_f64_ref, n, 1e-4);

    printf("\nQuickGELU:\n");
    srand(42); check_f64("intrin", quickgelu_f64_intrin, quickgelu_f64_ref, n, 1e-6);
    srand(42); check_f64("asm", quickgelu_f64_asm, quickgelu_f64_ref, n, 1e-4);

    printf("\nSiLU (Swish):\n");
    srand(42); check_f64("intrin", silu_f64_intrin, silu_f64_ref, n, 1e-6);
    srand(42); check_f64("asm", silu_f64_asm, silu_f64_ref, n, 1e-4);

    printf("\nSwiGLU:\n");
    srand(42); check_swiglu_f64("intrin", swiglu_f64_intrin, n, 1e-6);
    srand(42); check_swiglu_f64("asm", swiglu_f64_asm, n, 1e-4);

    printf("\n");

    // ========================================
    // FP32 Performance Benchmarks
    // ========================================
    printf("=== FP32 Performance Benchmarks ===\n");

    printf("\nGELU (exact):\n");
    srand(42); bench_f32("ref (scalar)", gelu_f32_ref, n, iterations);
    srand(42); bench_f32("intrin", gelu_f32_intrin, n, iterations);

    printf("\nGELU (tanh approx):\n");
    srand(42); bench_f32("ref (scalar)", gelu_tanh_f32_ref, n, iterations);
    srand(42); bench_f32("intrin", gelu_tanh_f32_intrin, n, iterations);
    srand(42); bench_f32("asm", gelu_tanh_f32_asm, n, iterations);

    printf("\nQuickGELU:\n");
    srand(42); bench_f32("ref (scalar)", quickgelu_f32_ref, n, iterations);
    srand(42); bench_f32("intrin", quickgelu_f32_intrin, n, iterations);
    srand(42); bench_f32("intrin_unroll2", quickgelu_f32_unroll2_intrin, n, iterations);
    srand(42); bench_f32("asm", quickgelu_f32_asm, n, iterations);

    printf("\nSiLU (Swish):\n");
    srand(42); bench_f32("ref (scalar)", silu_f32_ref, n, iterations);
    srand(42); bench_f32("intrin", silu_f32_intrin, n, iterations);
    srand(42); bench_f32("intrin_unroll2", silu_f32_unroll2_intrin, n, iterations);
    srand(42); bench_f32("asm", silu_f32_asm, n, iterations);

    printf("\nSwiGLU:\n");
    srand(42); bench_swiglu_f32("ref (scalar)", swiglu_f32_ref, n, iterations);
    srand(42); bench_swiglu_f32("intrin", swiglu_f32_intrin, n, iterations);
    srand(42); bench_swiglu_f32("intrin_unroll2", swiglu_f32_unroll2_intrin, n, iterations);
    srand(42); bench_swiglu_f32("asm", swiglu_f32_asm, n, iterations);

    printf("\n");

    // ========================================
    // FP64 Performance Benchmarks
    // ========================================
    printf("=== FP64 Performance Benchmarks ===\n");

    printf("\nGELU (exact):\n");
    srand(42); bench_f64("ref (scalar)", gelu_f64_ref, n, iterations);
    srand(42); bench_f64("intrin", gelu_f64_intrin, n, iterations);

    printf("\nGELU (tanh approx):\n");
    srand(42); bench_f64("ref (scalar)", gelu_tanh_f64_ref, n, iterations);
    srand(42); bench_f64("intrin", gelu_tanh_f64_intrin, n, iterations);
    srand(42); bench_f64("asm", gelu_tanh_f64_asm, n, iterations);

    printf("\nQuickGELU:\n");
    srand(42); bench_f64("ref (scalar)", quickgelu_f64_ref, n, iterations);
    srand(42); bench_f64("intrin", quickgelu_f64_intrin, n, iterations);
    srand(42); bench_f64("asm", quickgelu_f64_asm, n, iterations);

    printf("\nSiLU (Swish):\n");
    srand(42); bench_f64("ref (scalar)", silu_f64_ref, n, iterations);
    srand(42); bench_f64("intrin", silu_f64_intrin, n, iterations);
    srand(42); bench_f64("asm", silu_f64_asm, n, iterations);

    printf("\nSwiGLU:\n");
    srand(42); bench_swiglu_f64("ref (scalar)", swiglu_f64_ref, n, iterations);
    srand(42); bench_swiglu_f64("intrin", swiglu_f64_intrin, n, iterations);
    srand(42); bench_swiglu_f64("asm", swiglu_f64_asm, n, iterations);

    printf("\n");

    // ========================================
    // Scaling Tests
    // ========================================
    printf("=== Scaling Tests (SiLU FP32 intrin) ===\n");
    size_t test_sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    for (int t = 0; t < 7; t++) {
        size_t sz = test_sizes[t];
        char label[32];
        sprintf(label, "n=%zu", sz);
        srand(42);
        bench_f32(label, silu_f32_intrin, sz, iterations);
    }

    return 0;
}
