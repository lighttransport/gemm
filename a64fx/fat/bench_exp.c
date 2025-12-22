// bench_exp.c
// Benchmark for SVE exp() implementations
// Tests correctness and performance of exp_vec_unroll{4,5,8,10}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// Assembly function declarations - FP32
extern void exp_vec_unroll4(const float* input, float* output, size_t count);
extern void exp_vec_unroll5(const float* input, float* output, size_t count);
extern void exp_vec_unroll8(const float* input, float* output, size_t count);
extern void exp_vec_unroll10(const float* input, float* output, size_t count);
extern void test_exp_debug(const float* input, float* output);

// Assembly function declarations - FP64
extern void exp_vec_f64_unroll4(const double* input, double* output, size_t count);
extern void exp_vec_f64_unroll8(const double* input, double* output, size_t count);

// Assembly function declarations - FP16 (use __fp16 for ARM)
// Note: __fp16 is the ARM extension for half-precision
typedef __fp16 float16_t;
extern void exp_vec_f16_unroll4(const float16_t* input, float16_t* output, size_t count);
extern void exp_vec_f16_unroll8(const float16_t* input, float16_t* output, size_t count);
extern void test_fp16_simple(const float16_t* input, float16_t* output);
extern void test_fp16_exp_step(const float16_t* input, float16_t* output);
extern void test_fp16_const(float16_t* output);

// Read cycle counter (100 MHz on A64FX)
static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Reference exp using standard library - FP32
static void exp_ref(const float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        output[i] = expf(input[i]);
    }
}

// Reference exp - FP64
static void exp_ref_f64(const double* input, double* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        output[i] = exp(input[i]);
    }
}

// Manual FP16 to float conversion using bit manipulation
// FP16: 1 sign, 5 exponent (bias 15), 10 mantissa
// FP32: 1 sign, 8 exponent (bias 127), 23 mantissa
// NOTE: __attribute__((noinline)) to prevent FCC optimizer bug
__attribute__((noinline))
static float fp16_to_float(float16_t h) {
    union { float16_t h; uint16_t u; } hu;
    union { float f; uint32_t u; } fu;
    hu.h = h;
    uint16_t bits = hu.u;

    uint32_t sign = (bits >> 15) & 1;
    uint32_t exp = (bits >> 10) & 0x1f;
    uint32_t mant = bits & 0x3ff;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            fu.u = sign << 31;
        } else {
            // Subnormal: normalize
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            exp = (127 - 15) + exp;
            fu.u = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        fu.u = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        // Normal
        exp = exp - 15 + 127;
        fu.u = (sign << 31) | (exp << 23) | (mant << 13);
    }
    return fu.f;
}

static float16_t float_to_fp16(float f) {
    union { float f; uint32_t u; } fu;
    union { float16_t h; uint16_t u; } hu;
    fu.f = f;
    uint32_t bits = fu.u;

    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;

    if (exp <= 0) {
        // Underflow or subnormal
        if (exp < -10) {
            hu.u = sign << 15;  // Zero
        } else {
            // Subnormal
            mant = (mant | 0x400) >> (1 - exp);
            hu.u = (sign << 15) | mant;
        }
    } else if (exp >= 31) {
        // Overflow -> Inf
        hu.u = (sign << 15) | (31 << 10);
    } else {
        // Normal
        hu.u = (sign << 15) | (exp << 10) | mant;
    }
    return hu.h;
}

// Reference exp - FP16 (convert to float, compute, convert back)
static void exp_ref_f16(const float16_t* input, float16_t* output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        float x = fp16_to_float(input[i]);
        float y = expf(x);
        output[i] = float_to_fp16(y);
    }
}

// C implementation matching ASM algorithm for verification
static float approx_exp_c(float x) {
    // Range reduction: x = n * ln(2) + r
    const float log2e = 1.4426950408889634f;  // 1/ln(2)
    const float ln2 = 0.6931471805599453f;

    float n = roundf(x * log2e);
    float r = x - n * ln2;

    // Polynomial: 1 + r(1 + r(0.5 + r(1/6 + r/24)))
    float p = 1.0f/24.0f;
    p = p * r + 1.0f/6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    // 2^n via IEEE754
    int ni = (int)n;
    union { float f; int32_t i; } u;
    u.i = (ni + 127) << 23;

    return p * u.f;
}

// Debug version with print
static float approx_exp_c_debug(float x) {
    const float log2e = 1.4426950408889634f;
    const float ln2 = 0.6931471805599453f;

    float n = roundf(x * log2e);
    float r = x - n * ln2;

    float p = 1.0f/24.0f;
    printf("    x=%.4f, n=%.0f, r=%.6f, p_init=%.6f\n", x, n, r, p);

    p = p * r + 1.0f/6.0f;
    printf("    after c3: p=%.6f\n", p);
    p = p * r + 0.5f;
    printf("    after c2: p=%.6f\n", p);
    p = p * r + 1.0f;
    printf("    after c1: p=%.6f\n", p);
    p = p * r + 1.0f;
    printf("    after c0: p=%.6f\n", p);

    int ni = (int)n;
    union { float f; int32_t i; } u;
    u.i = (ni + 127) << 23;
    printf("    2^n = %.6f (n=%d)\n", u.f, ni);

    return p * u.f;
}

// Check correctness
static int check_correctness(const char* name,
                             void (*exp_fn)(const float*, float*, size_t),
                             size_t count) {
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output = aligned_alloc(64, count * sizeof(float));
    float* ref = aligned_alloc(64, count * sizeof(float));

    if (!input || !output || !ref) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize with values in valid range for exp (avoid overflow/underflow)
    for (size_t i = 0; i < count; i++) {
        // Range: -10 to 10 (exp ranges from ~4.5e-5 to ~22026)
        input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }

    // Compute reference
    exp_ref(input, ref, count);

    // Compute using ASM function
    exp_fn(input, output, count);

    // Check results
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    int errors = 0;

    for (size_t i = 0; i < count; i++) {
        float expected = ref[i];
        float actual = output[i];
        float abs_err = fabsf(actual - expected);
        float rel_err = (expected != 0.0f) ? abs_err / fabsf(expected) : abs_err;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;

        // Allow up to 1e-4 relative error (4-term polynomial approximation)
        if (rel_err > 1e-4f && abs_err > 1e-6f) {
            if (errors < 5) {
                printf("  Error at %zu: input=%f, expected=%f, got=%f, rel_err=%e\n",
                       i, input[i], expected, actual, rel_err);
            }
            errors++;
        }
    }

    printf("  %s: max_rel_err=%.2e, max_abs_err=%.2e, errors=%d/%zu\n",
           name, max_rel_err, max_abs_err, errors, count);

    free(input);
    free(output);
    free(ref);

    return (errors == 0) ? 0 : -1;
}

// Benchmark performance
static void benchmark(const char* name,
                      void (*exp_fn)(const float*, float*, size_t),
                      size_t count, int iterations) {
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output = aligned_alloc(64, count * sizeof(float));

    if (!input || !output) {
        printf("Memory allocation failed\n");
        return;
    }

    // Initialize
    for (size_t i = 0; i < count; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        exp_fn(input, output, count);
    }

    // Timed runs
    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp_fn(input, output, count);
    }
    uint64_t end = get_cycles();

    double cycles_total = (double)(end - start);
    double cycles_per_elem = cycles_total / (iterations * count);

    // A64FX cycle counter is at 100 MHz, CPU at 2.2 GHz
    // So actual CPU cycles = counter * 22
    double cpu_cycles_per_elem = cycles_per_elem * 22.0;

    // Throughput: elements per second
    // Timer is 100 MHz, so elapsed seconds = cycles_total / 100e6
    double elapsed_sec = cycles_total / 100e6;
    double throughput = (iterations * count) / elapsed_sec;
    double gops = throughput / 1e9;  // Giga-ops per second

    printf("  %s: %.2f CPU cycles/elem, %.2f Gexp/s\n",
           name, cpu_cycles_per_elem, gops);

    free(input);
    free(output);
}

int main(int argc, char** argv) {
    int iterations = 10000;
    size_t count = 16384;  // 64KB = 16K floats, fits in L1

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) count = atol(argv[2]);

    printf("=== SVE exp() Benchmark ===\n");
    printf("Count: %zu elements (%.1f KB)\n", count, count * sizeof(float) / 1024.0);
    printf("Iterations: %d\n\n", iterations);

    // Seed random
    srand(42);

    // First verify C approximation matches
    printf("Verifying C approx_exp vs expf:\n");
    float max_err = 0;
    for (int i = 0; i < 1000; i++) {
        float x = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        float ref_val = expf(x);
        float approx_val = approx_exp_c(x);
        float rel_err = fabsf(approx_val - ref_val) / fabsf(ref_val);
        if (rel_err > max_err) max_err = rel_err;
    }
    printf("  C approx_exp max relative error: %.2e\n\n", max_err);

    // Debug: test specific values with ASM (use 16 to fill one SVE vector)
    printf("Debug ASM exp for specific values:\n");
    float* test_vals = aligned_alloc(64, 16 * sizeof(float));
    float* test_out = aligned_alloc(64, 16 * sizeof(float));
    test_vals[0] = 0.0f; test_vals[1] = 1.0f; test_vals[2] = -1.0f;
    test_vals[3] = 2.0f; test_vals[4] = 3.0f; test_vals[5] = -3.0f;
    for (int i = 6; i < 16; i++) test_vals[i] = (float)(i - 8);
    exp_vec_unroll4(test_vals, test_out, 16);
    for (int i = 0; i < 6; i++) {
        printf("  exp(%.1f): expf=%.6f, approx_c=%.6f, asm=%.6f\n",
               test_vals[i], expf(test_vals[i]), approx_exp_c(test_vals[i]), test_out[i]);
    }
    free(test_vals);
    free(test_out);
    printf("\n");

    // Debug C algorithm for exp(1)
    printf("Debug C approx_exp for x=1.0:\n");
    approx_exp_c_debug(1.0f);
    printf("\n");

    // Debug ASM step by step
    printf("Debug ASM step by step for x=1.0:\n");
    float x_in = 1.0f;
    float dbg_out[7];
    test_exp_debug(&x_in, dbg_out);
    printf("  x      = %.6f\n", dbg_out[0]);
    printf("  n      = %.6f\n", dbg_out[1]);
    printf("  r      = %.6f\n", dbg_out[2]);
    printf("  n_int  = %.6f\n", dbg_out[3]);
    printf("  2^n    = %.6f\n", dbg_out[4]);
    printf("  poly   = %.6f\n", dbg_out[5]);
    printf("  result = %.6f\n", dbg_out[6]);
    printf("\n");

    // Correctness tests
    printf("=== Correctness Tests ===\n");
    srand(42);
    check_correctness("unroll4", exp_vec_unroll4, count);
    srand(42);
    check_correctness("unroll5", exp_vec_unroll5, count);
    srand(42);
    check_correctness("unroll8", exp_vec_unroll8, count);
    srand(42);
    check_correctness("unroll10", exp_vec_unroll10, count);
    printf("\n");

    // Performance benchmarks
    printf("=== Performance Benchmarks ===\n");
    srand(42);
    benchmark("unroll4 ", exp_vec_unroll4, count, iterations);
    srand(42);
    benchmark("unroll5 ", exp_vec_unroll5, count, iterations);
    srand(42);
    benchmark("unroll8 ", exp_vec_unroll8, count, iterations);
    srand(42);
    benchmark("unroll10", exp_vec_unroll10, count, iterations);
    printf("\n");

    // Reference performance
    printf("=== Reference (expf) ===\n");
    srand(42);
    benchmark("expf    ", exp_ref, count, iterations);

    // =====================================================
    // FP64 Tests
    // =====================================================
    printf("\n");
    printf("========================================\n");
    printf("FP64 (double precision) exp() Benchmark\n");
    printf("========================================\n");
    size_t count_f64 = count / 2;  // Same memory footprint
    printf("Count: %zu doubles (%.1f KB)\n\n", count_f64, count_f64 * sizeof(double) / 1024.0);

    // FP64 correctness
    {
        printf("=== FP64 Correctness Tests ===\n");
        double* input_f64 = aligned_alloc(64, count_f64 * sizeof(double));
        double* output_f64 = aligned_alloc(64, count_f64 * sizeof(double));
        double* ref_f64 = aligned_alloc(64, count_f64 * sizeof(double));

        srand(42);
        for (size_t i = 0; i < count_f64; i++) {
            input_f64[i] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
        }
        exp_ref_f64(input_f64, ref_f64, count_f64);

        // Test unroll4
        // Note: Using 5-term polynomial gives ~1e-6 precision, allow 1e-5 threshold
        exp_vec_f64_unroll4(input_f64, output_f64, count_f64);
        double max_rel_err = 0, max_abs_err = 0;
        int errors = 0;
        size_t max_idx = 0;
        for (size_t i = 0; i < count_f64; i++) {
            double abs_err = fabs(output_f64[i] - ref_f64[i]);
            double rel_err = (ref_f64[i] != 0.0) ? abs_err / fabs(ref_f64[i]) : abs_err;
            if (rel_err > max_rel_err) { max_rel_err = rel_err; max_idx = i; }
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > 1e-5) errors++;  // 1e-5 threshold for 5-term polynomial
        }
        printf("  f64_unroll4: max_rel_err=%.2e, max_abs_err=%.2e, errors=%d/%zu\n",
               max_rel_err, max_abs_err, errors, count_f64);
        if (errors > 0 && errors < 10) {
            printf("    worst at [%zu]: x=%.6f, ref=%.10f, asm=%.10f\n",
                   max_idx, input_f64[max_idx], ref_f64[max_idx], output_f64[max_idx]);
        }

        // Test unroll8
        exp_vec_f64_unroll8(input_f64, output_f64, count_f64);
        max_rel_err = 0; max_abs_err = 0; errors = 0;
        for (size_t i = 0; i < count_f64; i++) {
            double abs_err = fabs(output_f64[i] - ref_f64[i]);
            double rel_err = (ref_f64[i] != 0.0) ? abs_err / fabs(ref_f64[i]) : abs_err;
            if (rel_err > max_rel_err) { max_rel_err = rel_err; max_idx = i; }
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > 1e-5) errors++;
        }
        printf("  f64_unroll8: max_rel_err=%.2e, max_abs_err=%.2e, errors=%d/%zu\n",
               max_rel_err, max_abs_err, errors, count_f64);
        if (errors > 0 && errors < 10) {
            printf("    worst at [%zu]: x=%.6f, ref=%.10f, asm=%.10f\n",
                   max_idx, input_f64[max_idx], ref_f64[max_idx], output_f64[max_idx]);
        }

        free(input_f64);
        free(output_f64);
        free(ref_f64);
    }
    printf("\n");

    // FP64 performance
    {
        printf("=== FP64 Performance Benchmarks ===\n");
        double* input_f64 = aligned_alloc(64, count_f64 * sizeof(double));
        double* output_f64 = aligned_alloc(64, count_f64 * sizeof(double));

        srand(42);
        for (size_t i = 0; i < count_f64; i++) {
            input_f64[i] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
        }

        // Warmup and time unroll4
        for (int i = 0; i < 10; i++) exp_vec_f64_unroll4(input_f64, output_f64, count_f64);
        uint64_t start = get_cycles();
        for (int i = 0; i < iterations; i++) exp_vec_f64_unroll4(input_f64, output_f64, count_f64);
        uint64_t end = get_cycles();
        double cycles_per_elem = (double)(end - start) / (iterations * count_f64);
        double throughput = (iterations * count_f64) / ((end - start) / 100e6) / 1e9;
        printf("  f64_unroll4: %.2f CPU cycles/elem, %.2f Gexp/s\n", cycles_per_elem * 22, throughput);

        // Warmup and time unroll8
        for (int i = 0; i < 10; i++) exp_vec_f64_unroll8(input_f64, output_f64, count_f64);
        start = get_cycles();
        for (int i = 0; i < iterations; i++) exp_vec_f64_unroll8(input_f64, output_f64, count_f64);
        end = get_cycles();
        cycles_per_elem = (double)(end - start) / (iterations * count_f64);
        throughput = (iterations * count_f64) / ((end - start) / 100e6) / 1e9;
        printf("  f64_unroll8: %.2f CPU cycles/elem, %.2f Gexp/s\n", cycles_per_elem * 22, throughput);

        // Reference
        for (int i = 0; i < 10; i++) exp_ref_f64(input_f64, output_f64, count_f64);
        start = get_cycles();
        for (int i = 0; i < iterations; i++) exp_ref_f64(input_f64, output_f64, count_f64);
        end = get_cycles();
        cycles_per_elem = (double)(end - start) / (iterations * count_f64);
        throughput = (iterations * count_f64) / ((end - start) / 100e6) / 1e9;
        printf("  exp (ref):   %.2f CPU cycles/elem, %.2f Gexp/s\n", cycles_per_elem * 22, throughput);

        free(input_f64);
        free(output_f64);
    }

    // =====================================================
    // FP16 Tests
    // =====================================================
    printf("\n");
    printf("========================================\n");
    printf("FP16 (half precision) exp() Benchmark\n");
    printf("========================================\n");
    size_t count_f16 = count * 2;  // Same memory footprint
    printf("Count: %zu halfs (%.1f KB)\n\n", count_f16, count_f16 * sizeof(float16_t) / 1024.0);

    // FP16 correctness
    {
        printf("=== FP16 Correctness Tests ===\n");
        float16_t* input_f16 = aligned_alloc(64, count_f16 * sizeof(float16_t));
        float16_t* output_f16 = aligned_alloc(64, count_f16 * sizeof(float16_t));
        float16_t* ref_f16 = aligned_alloc(64, count_f16 * sizeof(float16_t));

        srand(42);
        for (size_t i = 0; i < count_f16; i++) {
            // FP16 range is limited: max ~65504, min subnormal ~5.96e-8
            // Use range -5 to 5 for exp (exp(5) ≈ 148, exp(-5) ≈ 0.0067)
            input_f16[i] = float_to_fp16(((float)rand() / RAND_MAX) * 10.0f - 5.0f);
        }
        exp_ref_f16(input_f16, ref_f16, count_f16);

        // Debug: check first few values
        printf("  Debug FP16 first values:\n");
        exp_vec_f16_unroll4(input_f16, output_f16, count_f16);
        for (int i = 0; i < 5; i++) {
            printf("    [%d] in=%.4f, ref=%.4f, asm=%.4f\n",
                   i, fp16_to_float(input_f16[i]), fp16_to_float(ref_f16[i]), fp16_to_float(output_f16[i]));
        }

        // Debug: test FP16 constant loading
        printf("  Debug FP16 constant loading:\n");
        float16_t const_out[4];
        test_fp16_const(const_out);
        printf("    fmov 1.0        = %.4f\n", fp16_to_float(const_out[0]));
        printf("    dup 0x3c00      = %.4f\n", fp16_to_float(const_out[1]));
        printf("    dup 0x3dc5      = %.4f\n", fp16_to_float(const_out[2]));
        printf("    1.0 * 0x3dc5    = %.4f\n", fp16_to_float(const_out[3]));

        // Debug: step-by-step FP16 exp for x=1.0
        printf("  Debug FP16 exp step-by-step for x=1.0:\n");
        float16_t dbg_in = float_to_fp16(1.0f);
        float16_t dbg_out[8];
        test_fp16_exp_step(&dbg_in, dbg_out);
        printf("    x       = %.4f\n", fp16_to_float(dbg_out[0]));
        printf("    x/ln2   = %.4f\n", fp16_to_float(dbg_out[1]));
        printf("    n       = %.4f\n", fp16_to_float(dbg_out[2]));
        printf("    r       = %.4f\n", fp16_to_float(dbg_out[3]));
        printf("    n_int   = %.4f\n", fp16_to_float(dbg_out[4]));
        printf("    n+15    = %.4f\n", fp16_to_float(dbg_out[5]));
        printf("    2^n     = %.4f\n", fp16_to_float(dbg_out[6]));
        printf("    result  = %.4f\n", fp16_to_float(dbg_out[7]));

        // Test unroll4
        exp_vec_f16_unroll4(input_f16, output_f16, count_f16);
        {
            float max_rel = 0.0f;
            float max_abs = 0.0f;
            size_t max_idx = 0;
            int errors = 0, inf_nan_count = 0;
            int first_big = 0;
            for (size_t i = 0; i < count_f16; i++) {
                float expected = fp16_to_float(ref_f16[i]);
                float actual = fp16_to_float(output_f16[i]);
                if (!isfinite(expected) || !isfinite(actual)) { inf_nan_count++; continue; }
                float abs_err = fabsf(actual - expected);
                float rel_err = (expected != 0.0f) ? abs_err / fabsf(expected) : abs_err;
                if (rel_err > max_rel) {
                    max_rel = rel_err;
                    max_idx = i;
                    // Print when we find a big error
                    if (rel_err > 0.5f && first_big < 3) {
                        printf("    BIG rel_err at [%zu]: rel=%.4f, x=%.6f, exp=%.6f, act=%.6f\n",
                               i, rel_err, fp16_to_float(input_f16[i]), expected, actual);
                        first_big++;
                    }
                }
                if (abs_err > max_abs) max_abs = abs_err;
                if (rel_err > 0.01f && abs_err > 1e-3f) errors++;
            }
            printf("  f16_unroll4: max_rel=%.4f, max_abs=%.4f, errors=%d/%zu (inf/nan=%d)\n",
                   max_rel, max_abs, errors, count_f16, inf_nan_count);
            // Print the worst case
            {
                float x = fp16_to_float(input_f16[max_idx]);
                float ref = fp16_to_float(ref_f16[max_idx]);
                float asm_v = fp16_to_float(output_f16[max_idx]);
                printf("    worst at [%zu]: x=%.6f, ref=%.6f, asm=%.6f\n", max_idx, x, ref, asm_v);
            }
        }

        // Test unroll8
        exp_vec_f16_unroll8(input_f16, output_f16, count_f16);
        {
            volatile float max_rel = 0.0f;
            volatile float max_abs = 0.0f;
            int errors = 0;
            for (size_t i = 0; i < count_f16; i++) {
                float expected = fp16_to_float(ref_f16[i]);
                float actual = fp16_to_float(output_f16[i]);
                if (!isfinite(expected) || !isfinite(actual)) continue;
                float abs_err = fabsf(actual - expected);
                float rel_err = (expected != 0.0f) ? abs_err / fabsf(expected) : abs_err;
                if (rel_err > max_rel) max_rel = rel_err;
                if (abs_err > max_abs) max_abs = abs_err;
                if (rel_err > 0.01f && abs_err > 1e-3f) errors++;
            }
            printf("  f16_unroll8: max_rel_err=%.2e, max_abs_err=%.2e, errors=%d/%zu\n",
                   (float)max_rel, (float)max_abs, errors, count_f16);
        }

        free(input_f16);
        free(output_f16);
        free(ref_f16);
    }
    printf("\n");

    // FP16 performance
    {
        printf("=== FP16 Performance Benchmarks ===\n");
        float16_t* input_f16 = aligned_alloc(64, count_f16 * sizeof(float16_t));
        float16_t* output_f16 = aligned_alloc(64, count_f16 * sizeof(float16_t));

        srand(42);
        for (size_t i = 0; i < count_f16; i++) {
            input_f16[i] = float_to_fp16(((float)rand() / RAND_MAX) * 10.0f - 5.0f);
        }

        // Warmup and time unroll4
        for (int i = 0; i < 10; i++) exp_vec_f16_unroll4(input_f16, output_f16, count_f16);
        uint64_t start = get_cycles();
        for (int i = 0; i < iterations; i++) exp_vec_f16_unroll4(input_f16, output_f16, count_f16);
        uint64_t end = get_cycles();
        double cycles_per_elem = (double)(end - start) / (iterations * count_f16);
        double throughput = (iterations * count_f16) / ((end - start) / 100e6) / 1e9;
        printf("  f16_unroll4: %.2f CPU cycles/elem, %.2f Gexp/s\n", cycles_per_elem * 22, throughput);

        // Warmup and time unroll8
        for (int i = 0; i < 10; i++) exp_vec_f16_unroll8(input_f16, output_f16, count_f16);
        start = get_cycles();
        for (int i = 0; i < iterations; i++) exp_vec_f16_unroll8(input_f16, output_f16, count_f16);
        end = get_cycles();
        cycles_per_elem = (double)(end - start) / (iterations * count_f16);
        throughput = (iterations * count_f16) / ((end - start) / 100e6) / 1e9;
        printf("  f16_unroll8: %.2f CPU cycles/elem, %.2f Gexp/s\n", cycles_per_elem * 22, throughput);

        // Reference (scalar via float conversion)
        for (int i = 0; i < 10; i++) exp_ref_f16(input_f16, output_f16, count_f16);
        start = get_cycles();
        for (int i = 0; i < iterations; i++) exp_ref_f16(input_f16, output_f16, count_f16);
        end = get_cycles();
        cycles_per_elem = (double)(end - start) / (iterations * count_f16);
        throughput = (iterations * count_f16) / ((end - start) / 100e6) / 1e9;
        printf("  expf (ref):  %.2f CPU cycles/elem, %.2f Gexp/s\n", cycles_per_elem * 22, throughput);

        free(input_f16);
        free(output_f16);
    }

    return 0;
}
