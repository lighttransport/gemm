// bench_opt.c
// Compare optimized intrinsics vs hand-written assembly

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// Hand-written assembly
extern void exp_f32_poly5(const float*, float*, size_t);
extern void exp_f64_poly5(const double*, double*, size_t);

// Intrinsics - simple
extern void exp_f32_poly5_intrin(const float*, float*, size_t);
extern void exp_f64_poly5_intrin(const double*, double*, size_t);

// Intrinsics - optimized (4x unrolled)
extern void exp_f32_poly5_intrin_opt(const float*, float*, size_t);
extern void exp_f64_poly5_intrin_opt(const double*, double*, size_t);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

int main(int argc, char** argv) {
    int iterations = 1000;
    size_t count = 16384;

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) count = atol(argv[2]);

    printf("=== Optimized Intrinsics vs Assembly (poly5 only) ===\n");
    printf("Count: %zu, Iterations: %d\n\n", count, iterations);

    srand(42);

    // FP32
    float* input_f32 = aligned_alloc(64, count * sizeof(float));
    float* output_f32 = aligned_alloc(64, count * sizeof(float));
    float* ref_f32 = aligned_alloc(64, count * sizeof(float));

    for (size_t i = 0; i < count; i++) {
        input_f32[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        ref_f32[i] = expf(input_f32[i]);
    }

    // FP64
    double* input_f64 = aligned_alloc(64, count * sizeof(double));
    double* output_f64 = aligned_alloc(64, count * sizeof(double));
    double* ref_f64 = aligned_alloc(64, count * sizeof(double));

    for (size_t i = 0; i < count; i++) {
        input_f64[i] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
        ref_f64[i] = exp(input_f64[i]);
    }

    printf("=== FP32 poly5 ===\n");
    printf("%-25s %-12s %-10s %s\n", "Version", "Max Rel Err", "Cyc/elem", "Gexp/s");
    printf("----------------------------------------------------------\n");

    // Benchmark each version
    typedef void (*f32_fn)(const float*, float*, size_t);
    f32_fn f32_funcs[] = {exp_f32_poly5, exp_f32_poly5_intrin, exp_f32_poly5_intrin_opt};
    const char* f32_names[] = {"asm (4x unrolled)", "intrin (simple)", "intrin (4x unrolled)"};

    for (int f = 0; f < 3; f++) {
        f32_funcs[f](input_f32, output_f32, count);
        float max_rel = 0;
        for (size_t i = 0; i < count; i++) {
            float err = fabsf(output_f32[i] - ref_f32[i]);
            float rel = (ref_f32[i] != 0) ? err / fabsf(ref_f32[i]) : err;
            if (rel > max_rel) max_rel = rel;
        }

        for (int w = 0; w < 10; w++) f32_funcs[f](input_f32, output_f32, count);
        uint64_t start = get_cycles();
        for (int it = 0; it < iterations; it++) f32_funcs[f](input_f32, output_f32, count);
        uint64_t end = get_cycles();

        double total = (double)iterations * count;
        double cyc = (double)(end - start) / total * 22.0;
        double gops = total / ((double)(end - start) / 100e6) / 1e9;

        printf("%-25s %.2e     %.2f       %.2f\n", f32_names[f], max_rel, cyc, gops);
    }

    printf("\n=== FP64 poly5 ===\n");
    printf("%-25s %-12s %-10s %s\n", "Version", "Max Rel Err", "Cyc/elem", "Gexp/s");
    printf("----------------------------------------------------------\n");

    typedef void (*f64_fn)(const double*, double*, size_t);
    f64_fn f64_funcs[] = {exp_f64_poly5, exp_f64_poly5_intrin, exp_f64_poly5_intrin_opt};
    const char* f64_names[] = {"asm (4x unrolled)", "intrin (simple)", "intrin (4x unrolled)"};

    for (int f = 0; f < 3; f++) {
        f64_funcs[f](input_f64, output_f64, count);
        double max_rel = 0;
        for (size_t i = 0; i < count; i++) {
            double err = fabs(output_f64[i] - ref_f64[i]);
            double rel = (ref_f64[i] != 0) ? err / fabs(ref_f64[i]) : err;
            if (rel > max_rel) max_rel = rel;
        }

        for (int w = 0; w < 10; w++) f64_funcs[f](input_f64, output_f64, count);
        uint64_t start = get_cycles();
        for (int it = 0; it < iterations; it++) f64_funcs[f](input_f64, output_f64, count);
        uint64_t end = get_cycles();

        double total = (double)iterations * count;
        double cyc = (double)(end - start) / total * 22.0;
        double gops = total / ((double)(end - start) / 100e6) / 1e9;

        printf("%-25s %.2e     %.2f       %.2f\n", f64_names[f], max_rel, cyc, gops);
    }

    free(input_f32);
    free(output_f32);
    free(ref_f32);
    free(input_f64);
    free(output_f64);
    free(ref_f64);

    return 0;
}
