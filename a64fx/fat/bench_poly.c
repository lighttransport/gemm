// bench_poly.c
// Benchmark for SVE exp() with varying polynomial terms (1-5)
// Tests accuracy vs performance tradeoff

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// FP32 polynomial variants
extern void exp_f32_poly1(const float* input, float* output, size_t count);
extern void exp_f32_poly2(const float* input, float* output, size_t count);
extern void exp_f32_poly3(const float* input, float* output, size_t count);
extern void exp_f32_poly4(const float* input, float* output, size_t count);
extern void exp_f32_poly5(const float* input, float* output, size_t count);

// FP64 polynomial variants
extern void exp_f64_poly1(const double* input, double* output, size_t count);
extern void exp_f64_poly2(const double* input, double* output, size_t count);
extern void exp_f64_poly3(const double* input, double* output, size_t count);
extern void exp_f64_poly4(const double* input, double* output, size_t count);
extern void exp_f64_poly5(const double* input, double* output, size_t count);

// Cycle counter (100 MHz on A64FX)
static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Reference implementations
static void exp_ref_f32(const float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; i++) output[i] = expf(input[i]);
}

static void exp_ref_f64(const double* input, double* output, size_t count) {
    for (size_t i = 0; i < count; i++) output[i] = exp(input[i]);
}

// Test FP32 variant
static void test_f32(const char* name,
                     void (*fn)(const float*, float*, size_t),
                     const float* input, const float* ref,
                     float* output, size_t count, int iterations) {
    // Correctness
    fn(input, output, count);
    volatile float max_rel = 0;
    volatile float max_abs = 0;
    for (size_t i = 0; i < count; i++) {
        float out_val = output[i];
        float ref_val = ref[i];
        float err = fabsf(out_val - ref_val);
        float rel = (ref_val != 0.0f) ? err / fabsf(ref_val) : err;
        if (rel > max_rel) max_rel = rel;
        if (err > max_abs) max_abs = err;
    }

    // Performance
    for (int i = 0; i < 10; i++) fn(input, output, count);
    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) fn(input, output, count);
    uint64_t end = get_cycles();

    uint64_t elapsed = end - start;
    double total_ops = (double)iterations * (double)count;
    // cntvct runs at 100 MHz, CPU at 2.2 GHz, so multiply by 22
    double elapsed_d = (double)elapsed;
    double cycles = elapsed_d / total_ops * 22.0;
    double secs = elapsed_d / 100e6;
    double gops = total_ops / secs / 1e9;

    float max_rel_val = max_rel;
    float max_abs_val = max_abs;
    printf("  %s: rel=%.2e abs=%.2e el=%lu el_d=%.1f ops=%.0f cyc=%.2f\n",
           name, max_rel_val, max_abs_val, (unsigned long)elapsed, elapsed_d, total_ops, cycles);
}

// Test FP64 variant
static void test_f64(const char* name,
                     void (*fn)(const double*, double*, size_t),
                     const double* input, const double* ref,
                     double* output, size_t count, int iterations) {
    // Correctness
    fn(input, output, count);
    volatile double max_rel = 0;
    volatile double max_abs = 0;
    for (size_t i = 0; i < count; i++) {
        double out_val = output[i];
        double ref_val = ref[i];
        double err = fabs(out_val - ref_val);
        double rel = (ref_val != 0.0) ? err / fabs(ref_val) : err;
        if (rel > max_rel) max_rel = rel;
        if (err > max_abs) max_abs = err;
    }

    // Performance
    for (int i = 0; i < 10; i++) fn(input, output, count);
    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) fn(input, output, count);
    uint64_t end = get_cycles();

    uint64_t elapsed = end - start;
    double total_ops = (double)iterations * (double)count;
    double cycles = (double)elapsed / total_ops * 22.0;
    double secs = (double)elapsed / 100e6;
    double gops = total_ops / secs / 1e9;

    double max_rel_val = max_rel;
    double max_abs_val = max_abs;
    printf("  %s: max_rel=%.2e, max_abs=%.2e, %.2f cyc/elem, %.2f Gexp/s\n",
           name, max_rel_val, max_abs_val, cycles, gops);
}

int main(int argc, char** argv) {
    int iterations = 1000;
    size_t count = 16384;

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) count = atol(argv[2]);

    printf("=== SVE exp() Polynomial Term Comparison ===\n");
    printf("Count: %zu elements, Iterations: %d\n\n", count, iterations);

    srand(42);

    // FP32 tests
    {
        printf("=== FP32 (float) Results ===\n");
        printf("Terms | Max Rel Err | Max Abs Err | Cycles/elem | Gexp/s\n");
        printf("------+-------------+-------------+-------------+--------\n");

        // Add extra padding to detect buffer overflows
        float* input = aligned_alloc(64, (count + 256) * sizeof(float));
        float* output = aligned_alloc(64, (count + 256) * sizeof(float));
        float* ref = aligned_alloc(64, (count + 256) * sizeof(float));

        for (size_t i = 0; i < count; i++) {
            input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        }
        exp_ref_f32(input, ref, count);

        test_f32("poly1", exp_f32_poly1, input, ref, output, count, iterations);
        test_f32("poly2", exp_f32_poly2, input, ref, output, count, iterations);
        test_f32("poly3", exp_f32_poly3, input, ref, output, count, iterations);
        test_f32("poly4", exp_f32_poly4, input, ref, output, count, iterations);
        test_f32("poly5", exp_f32_poly5, input, ref, output, count, iterations);

        // Reference
        for (int i = 0; i < 10; i++) exp_ref_f32(input, output, count);
        uint64_t start = get_cycles();
        for (int i = 0; i < iterations; i++) exp_ref_f32(input, output, count);
        uint64_t end = get_cycles();
        uint64_t elapsed = end - start;
        double total_ops = (double)iterations * (double)count;
        double cycles = (double)elapsed / total_ops * 22.0;
        double secs = (double)elapsed / 100e6;
        double gops = total_ops / secs / 1e9;
        printf("  expf:  (reference)                    %.2f cyc/elem, %.2f Gexp/s\n", cycles, gops);

        free(input);
        free(output);
        free(ref);
    }

    printf("\n");

    // FP64 tests
    {
        printf("=== FP64 (double) Results ===\n");
        printf("Terms | Max Rel Err | Max Abs Err | Cycles/elem | Gexp/s\n");
        printf("------+-------------+-------------+-------------+--------\n");

        size_t count64 = count / 2;  // Same memory footprint
        double* input = aligned_alloc(64, count64 * sizeof(double));
        double* output = aligned_alloc(64, count64 * sizeof(double));
        double* ref = aligned_alloc(64, count64 * sizeof(double));

        for (size_t i = 0; i < count64; i++) {
            input[i] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
        }
        exp_ref_f64(input, ref, count64);

        test_f64("poly1", exp_f64_poly1, input, ref, output, count64, iterations);
        test_f64("poly2", exp_f64_poly2, input, ref, output, count64, iterations);
        test_f64("poly3", exp_f64_poly3, input, ref, output, count64, iterations);
        test_f64("poly4", exp_f64_poly4, input, ref, output, count64, iterations);
        test_f64("poly5", exp_f64_poly5, input, ref, output, count64, iterations);

        // Reference
        for (int i = 0; i < 10; i++) exp_ref_f64(input, output, count64);
        uint64_t start64 = get_cycles();
        for (int i = 0; i < iterations; i++) exp_ref_f64(input, output, count64);
        uint64_t end64 = get_cycles();
        uint64_t elapsed64 = end64 - start64;
        double total_ops64 = (double)iterations * (double)count64;
        double cycles64 = (double)elapsed64 / total_ops64 * 22.0;
        double secs64 = (double)elapsed64 / 100e6;
        double gops64 = total_ops64 / secs64 / 1e9;
        printf("  exp:   (reference)                    %.2f cyc/elem, %.2f Gexp/s\n", cycles64, gops64);

        free(input);
        free(output);
        free(ref);
    }

    printf("\n=== Summary ===\n");
    printf("Polynomial terms vs accuracy tradeoff:\n");
    printf("  1-term: exp(r) ≈ 1 + r                     (fastest, ~10%% error)\n");
    printf("  2-term: exp(r) ≈ 1 + r + r²/2              (~1%% error)\n");
    printf("  3-term: exp(r) ≈ 1 + r + r²/2 + r³/6       (~0.1%% error)\n");
    printf("  4-term: exp(r) ≈ 1 + r + ... + r⁴/24       (~0.01%% error)\n");
    printf("  5-term: exp(r) ≈ 1 + r + ... + r⁵/120      (~0.001%% error)\n");

    return 0;
}
