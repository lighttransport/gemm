/*
 * Benchmark comparing Horner vs Estrin polynomial evaluation for exp2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

#define LOG2E 1.4426950408889634f

/* External functions */
extern void exp2_poly_simple_v2(const float* in, float* out, int n);
extern void exp2_poly_softmax_v2(const int32_t* in, float* out, int n, float scale, int32_t max_val);
extern void exp2_estrin_simple(const float* in, float* out, int n);
extern void exp2_estrin_softmax(const int32_t* in, float* out, int n, float scale, int32_t max_val);

static int ulp_error(float a, float b) {
    if (isnan(a) || isnan(b)) return 0;
    if (isinf(a) || isinf(b)) return 0;
    if (a == b) return 0;
    union { float f; int32_t i; } ua, ub;
    ua.f = a; ub.f = b;
    return abs(ua.i - ub.i);
}

static void exp2_reference(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = exp2f(in[i]);
}

static void* aligned_alloc_safe(size_t align, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, align, size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

int main(int argc, char** argv) {
    int n = 1024 * 1024;
    int iters = 100;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    printf("=== Horner vs Estrin Polynomial Comparison ===\n");
    printf("Elements: %d, Iterations: %d\n\n", n, iters);

    float* in_f = aligned_alloc_safe(256, n * sizeof(float));
    int32_t* in_i = aligned_alloc_safe(256, n * sizeof(int32_t));
    float* out_horner = aligned_alloc_safe(256, n * sizeof(float));
    float* out_estrin = aligned_alloc_safe(256, n * sizeof(float));
    float* out_ref = aligned_alloc_safe(256, n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        in_f[i] = ((float)rand() / RAND_MAX * 100.0f) - 50.0f;
        in_i[i] = (rand() % 100000) - 50000;
    }

    /* Correctness check */
    printf("=== Correctness Check ===\n");
    exp2_reference(in_f, out_ref, n);
    exp2_poly_simple_v2(in_f, out_horner, n);
    exp2_estrin_simple(in_f, out_estrin, n);

    int max_ulp_horner = 0, max_ulp_estrin = 0;
    for (int i = 0; i < n; i++) {
        int ulp_h = ulp_error(out_ref[i], out_horner[i]);
        int ulp_e = ulp_error(out_ref[i], out_estrin[i]);
        if (ulp_h > max_ulp_horner) max_ulp_horner = ulp_h;
        if (ulp_e > max_ulp_estrin) max_ulp_estrin = ulp_e;
    }
    printf("Horner max ULP: %d\n", max_ulp_horner);
    printf("Estrin max ULP: %d\n", max_ulp_estrin);

    /* Sample check */
    printf("\nSample values:\n");
    printf("%-12s %-14s %-14s %-14s\n", "Input", "Reference", "Horner", "Estrin");
    for (int i = 0; i < 5; i++) {
        printf("%-12.4f %-14.6e %-14.6e %-14.6e\n",
               in_f[i], out_ref[i], out_horner[i], out_estrin[i]);
    }

    /* Performance comparison */
    printf("\n=== Performance: Simple exp2 ===\n");

    // Warmup
    for (int i = 0; i < 5; i++) {
        exp2_poly_simple_v2(in_f, out_horner, n);
        exp2_estrin_simple(in_f, out_estrin, n);
    }

    double t0, t1;

    // Horner
    t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_poly_simple_v2(in_f, out_horner, n);
    }
    t1 = omp_get_wtime();
    double horner_time = (t1 - t0) / iters;
    double horner_throughput = n / horner_time / 1e9;

    // Estrin
    t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_estrin_simple(in_f, out_estrin, n);
    }
    t1 = omp_get_wtime();
    double estrin_time = (t1 - t0) / iters;
    double estrin_throughput = n / estrin_time / 1e9;

    // Reference
    t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_reference(in_f, out_ref, n);
    }
    t1 = omp_get_wtime();
    double ref_time = (t1 - t0) / iters;

    printf("%-12s %10s %12s %12s\n", "Method", "Time(ms)", "Gelem/s", "Cycles/elem");
    printf("%-12s %10.3f %12.2f %12.2f\n", "Horner", horner_time*1000, horner_throughput, 2.0/horner_throughput);
    printf("%-12s %10.3f %12.2f %12.2f\n", "Estrin", estrin_time*1000, estrin_throughput, 2.0/estrin_throughput);
    printf("%-12s %10.3f %12.2f %12.2f\n", "libm", ref_time*1000, n/ref_time/1e9, 2.0/(n/ref_time/1e9));

    printf("\nSpeedup: Estrin vs Horner: %.2fx\n", horner_time / estrin_time);
    printf("Speedup: Estrin vs libm: %.2fx\n", ref_time / estrin_time);

    /* Softmax performance */
    printf("\n=== Performance: Softmax exp2 ===\n");
    float scale = 0.01f * LOG2E;
    int32_t max_val = 25000;

    // Warmup
    for (int i = 0; i < 5; i++) {
        exp2_poly_softmax_v2(in_i, out_horner, n, scale, max_val);
        exp2_estrin_softmax(in_i, out_estrin, n, scale, max_val);
    }

    t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_poly_softmax_v2(in_i, out_horner, n, scale, max_val);
    }
    t1 = omp_get_wtime();
    double horner_sm_time = (t1 - t0) / iters;

    t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_estrin_softmax(in_i, out_estrin, n, scale, max_val);
    }
    t1 = omp_get_wtime();
    double estrin_sm_time = (t1 - t0) / iters;

    printf("%-12s %10s %12s %12s\n", "Method", "Time(ms)", "Gelem/s", "Cycles/elem");
    printf("%-12s %10.3f %12.2f %12.2f\n", "Horner", horner_sm_time*1000, n/horner_sm_time/1e9, 2.0/(n/horner_sm_time/1e9));
    printf("%-12s %10.3f %12.2f %12.2f\n", "Estrin", estrin_sm_time*1000, n/estrin_sm_time/1e9, 2.0/(n/estrin_sm_time/1e9));

    printf("\nSpeedup: Estrin vs Horner (softmax): %.2fx\n", horner_sm_time / estrin_sm_time);

    free(in_f); free(in_i);
    free(out_horner); free(out_estrin); free(out_ref);
    return 0;
}
