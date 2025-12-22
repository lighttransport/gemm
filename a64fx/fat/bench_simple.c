// Simple benchmark for SVE exp() polynomials
// Tests each function in isolation to avoid register corruption

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);
extern void exp_f32_poly2(const float* input, float* output, size_t count);
extern void exp_f32_poly3(const float* input, float* output, size_t count);
extern void exp_f32_poly4(const float* input, float* output, size_t count);
extern void exp_f32_poly5(const float* input, float* output, size_t count);

extern void exp_f64_poly1(const double* input, double* output, size_t count);
extern void exp_f64_poly2(const double* input, double* output, size_t count);
extern void exp_f64_poly3(const double* input, double* output, size_t count);
extern void exp_f64_poly4(const double* input, double* output, size_t count);
extern void exp_f64_poly5(const double* input, double* output, size_t count);

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

    printf("=== SVE exp() Polynomial Benchmark ===\n");
    printf("Count: %zu, Iterations: %d\n\n", count, iterations);

    srand(42);

    // Allocate arrays
    float* input_f32 = aligned_alloc(64, count * sizeof(float));
    float* output_f32 = aligned_alloc(64, count * sizeof(float));
    float* ref_f32 = aligned_alloc(64, count * sizeof(float));

    double* input_f64 = aligned_alloc(64, count * sizeof(double));
    double* output_f64 = aligned_alloc(64, count * sizeof(double));
    double* ref_f64 = aligned_alloc(64, count * sizeof(double));

    // Initialize
    for (size_t i = 0; i < count; i++) {
        input_f32[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        input_f64[i] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
        ref_f32[i] = expf(input_f32[i]);
        ref_f64[i] = exp(input_f64[i]);
    }

    printf("=== FP32 Results ===\n");
    printf("Poly | Max Rel Err | Cycles/elem | Gexp/s\n");

    // Test each FP32 variant
    void (*f32_funcs[])(const float*, float*, size_t) = {
        exp_f32_poly1, exp_f32_poly2, exp_f32_poly3, exp_f32_poly4, exp_f32_poly5
    };
    const char* names[] = {"1", "2", "3", "4", "5"};

    for (int p = 0; p < 5; p++) {
        // Accuracy
        f32_funcs[p](input_f32, output_f32, count);
        float max_rel = 0;
        for (size_t i = 0; i < count; i++) {
            float err = fabsf(output_f32[i] - ref_f32[i]);
            float rel = (ref_f32[i] != 0) ? err / fabsf(ref_f32[i]) : err;
            if (rel > max_rel) max_rel = rel;
        }

        // Performance
        for (int i = 0; i < 10; i++) f32_funcs[p](input_f32, output_f32, count);
        uint64_t start = get_cycles();
        for (int i = 0; i < iterations; i++) f32_funcs[p](input_f32, output_f32, count);
        uint64_t end = get_cycles();

        uint64_t elapsed = end - start;
        double total = (double)iterations * count;
        double cyc = (double)elapsed / total * 22.0;
        double gops = total / ((double)elapsed / 100e6) / 1e9;

        printf("  %s  | %.2e    | %.2f        | %.2f\n", names[p], max_rel, cyc, gops);
    }

    printf("\n=== FP64 Results ===\n");
    printf("Poly | Max Rel Err | Cycles/elem | Gexp/s\n");

    // Test each FP64 variant
    void (*f64_funcs[])(const double*, double*, size_t) = {
        exp_f64_poly1, exp_f64_poly2, exp_f64_poly3, exp_f64_poly4, exp_f64_poly5
    };

    for (int p = 0; p < 5; p++) {
        // Accuracy
        f64_funcs[p](input_f64, output_f64, count);
        double max_rel = 0;
        for (size_t i = 0; i < count; i++) {
            double err = fabs(output_f64[i] - ref_f64[i]);
            double rel = (ref_f64[i] != 0) ? err / fabs(ref_f64[i]) : err;
            if (rel > max_rel) max_rel = rel;
        }

        // Performance
        for (int i = 0; i < 10; i++) f64_funcs[p](input_f64, output_f64, count);
        uint64_t start = get_cycles();
        for (int i = 0; i < iterations; i++) f64_funcs[p](input_f64, output_f64, count);
        uint64_t end = get_cycles();

        uint64_t elapsed = end - start;
        double total = (double)iterations * count;
        double cyc = (double)elapsed / total * 22.0;
        double gops = total / ((double)elapsed / 100e6) / 1e9;

        printf("  %s  | %.2e    | %.2f        | %.2f\n", names[p], max_rel, cyc, gops);
    }

    free(input_f32);
    free(output_f32);
    free(ref_f32);
    free(input_f64);
    free(output_f64);
    free(ref_f64);

    return 0;
}
