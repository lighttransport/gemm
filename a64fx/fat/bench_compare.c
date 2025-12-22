// bench_compare.c
// Compare SVE intrinsics vs hand-written assembly for exp()

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// Hand-written assembly versions
extern void exp_f32_poly1(const float*, float*, size_t);
extern void exp_f32_poly2(const float*, float*, size_t);
extern void exp_f32_poly3(const float*, float*, size_t);
extern void exp_f32_poly4(const float*, float*, size_t);
extern void exp_f32_poly5(const float*, float*, size_t);

extern void exp_f64_poly1(const double*, double*, size_t);
extern void exp_f64_poly2(const double*, double*, size_t);
extern void exp_f64_poly3(const double*, double*, size_t);
extern void exp_f64_poly4(const double*, double*, size_t);
extern void exp_f64_poly5(const double*, double*, size_t);

// Intrinsics versions
extern void exp_f32_poly1_intrin(const float*, float*, size_t);
extern void exp_f32_poly2_intrin(const float*, float*, size_t);
extern void exp_f32_poly3_intrin(const float*, float*, size_t);
extern void exp_f32_poly4_intrin(const float*, float*, size_t);
extern void exp_f32_poly5_intrin(const float*, float*, size_t);

extern void exp_f64_poly1_intrin(const double*, double*, size_t);
extern void exp_f64_poly2_intrin(const double*, double*, size_t);
extern void exp_f64_poly3_intrin(const double*, double*, size_t);
extern void exp_f64_poly4_intrin(const double*, double*, size_t);
extern void exp_f64_poly5_intrin(const double*, double*, size_t);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

typedef void (*exp_f32_fn)(const float*, float*, size_t);
typedef void (*exp_f64_fn)(const double*, double*, size_t);

static void bench_f32(const char* name, exp_f32_fn fn,
                      const float* input, const float* ref,
                      float* output, size_t count, int iterations) {
    // Accuracy
    fn(input, output, count);
    float max_rel = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max_rel) max_rel = rel;
    }

    // Warmup
    for (int i = 0; i < 10; i++) fn(input, output, count);

    // Timing
    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) fn(input, output, count);
    uint64_t end = get_cycles();

    uint64_t elapsed = end - start;
    double total = (double)iterations * count;
    double cyc = (double)elapsed / total * 22.0;
    double gops = total / ((double)elapsed / 100e6) / 1e9;

    printf("  %-20s rel=%.2e  cyc=%.2f  Gexp/s=%.2f\n", name, max_rel, cyc, gops);
}

static void bench_f64(const char* name, exp_f64_fn fn,
                      const double* input, const double* ref,
                      double* output, size_t count, int iterations) {
    // Accuracy
    fn(input, output, count);
    double max_rel = 0;
    for (size_t i = 0; i < count; i++) {
        double err = fabs(output[i] - ref[i]);
        double rel = (ref[i] != 0) ? err / fabs(ref[i]) : err;
        if (rel > max_rel) max_rel = rel;
    }

    // Warmup
    for (int i = 0; i < 10; i++) fn(input, output, count);

    // Timing
    uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++) fn(input, output, count);
    uint64_t end = get_cycles();

    uint64_t elapsed = end - start;
    double total = (double)iterations * count;
    double cyc = (double)elapsed / total * 22.0;
    double gops = total / ((double)elapsed / 100e6) / 1e9;

    printf("  %-20s rel=%.2e  cyc=%.2f  Gexp/s=%.2f\n", name, max_rel, cyc, gops);
}

int main(int argc, char** argv) {
    int iterations = 1000;
    size_t count = 16384;

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) count = atol(argv[2]);

    printf("=== SVE exp() Intrinsics vs Assembly Comparison ===\n");
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

    printf("=== FP32 Comparison ===\n");
    printf("%-22s %-12s %-8s %s\n", "Function", "Max Rel Err", "Cyc/elem", "Gexp/s");
    printf("------------------------------------------------------\n");

    printf("Poly1:\n");
    bench_f32("asm", exp_f32_poly1, input_f32, ref_f32, output_f32, count, iterations);
    bench_f32("intrinsics", exp_f32_poly1_intrin, input_f32, ref_f32, output_f32, count, iterations);

    printf("Poly2:\n");
    bench_f32("asm", exp_f32_poly2, input_f32, ref_f32, output_f32, count, iterations);
    bench_f32("intrinsics", exp_f32_poly2_intrin, input_f32, ref_f32, output_f32, count, iterations);

    printf("Poly3:\n");
    bench_f32("asm", exp_f32_poly3, input_f32, ref_f32, output_f32, count, iterations);
    bench_f32("intrinsics", exp_f32_poly3_intrin, input_f32, ref_f32, output_f32, count, iterations);

    printf("Poly4:\n");
    bench_f32("asm", exp_f32_poly4, input_f32, ref_f32, output_f32, count, iterations);
    bench_f32("intrinsics", exp_f32_poly4_intrin, input_f32, ref_f32, output_f32, count, iterations);

    printf("Poly5:\n");
    bench_f32("asm", exp_f32_poly5, input_f32, ref_f32, output_f32, count, iterations);
    bench_f32("intrinsics", exp_f32_poly5_intrin, input_f32, ref_f32, output_f32, count, iterations);

    printf("\n=== FP64 Comparison ===\n");
    printf("%-22s %-12s %-8s %s\n", "Function", "Max Rel Err", "Cyc/elem", "Gexp/s");
    printf("------------------------------------------------------\n");

    printf("Poly1:\n");
    bench_f64("asm", exp_f64_poly1, input_f64, ref_f64, output_f64, count, iterations);
    bench_f64("intrinsics", exp_f64_poly1_intrin, input_f64, ref_f64, output_f64, count, iterations);

    printf("Poly2:\n");
    bench_f64("asm", exp_f64_poly2, input_f64, ref_f64, output_f64, count, iterations);
    bench_f64("intrinsics", exp_f64_poly2_intrin, input_f64, ref_f64, output_f64, count, iterations);

    printf("Poly3:\n");
    bench_f64("asm", exp_f64_poly3, input_f64, ref_f64, output_f64, count, iterations);
    bench_f64("intrinsics", exp_f64_poly3_intrin, input_f64, ref_f64, output_f64, count, iterations);

    printf("Poly4:\n");
    bench_f64("asm", exp_f64_poly4, input_f64, ref_f64, output_f64, count, iterations);
    bench_f64("intrinsics", exp_f64_poly4_intrin, input_f64, ref_f64, output_f64, count, iterations);

    printf("Poly5:\n");
    bench_f64("asm", exp_f64_poly5, input_f64, ref_f64, output_f64, count, iterations);
    bench_f64("intrinsics", exp_f64_poly5_intrin, input_f64, ref_f64, output_f64, count, iterations);

    free(input_f32);
    free(output_f32);
    free(ref_f32);
    free(input_f64);
    free(output_f64);
    free(ref_f64);

    return 0;
}
