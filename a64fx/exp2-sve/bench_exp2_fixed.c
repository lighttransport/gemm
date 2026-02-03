/*
 * Benchmark for fixed exp2 FEXPA kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>

#define LOG2E 1.4426950408889634f

/* External assembly functions (fixed version) */
extern void exp2_fexpa_simple_fixed(const float* in, float* out, int n);
extern void exp2_fexpa_softmax_fixed(const int32_t* in, float* out, int n,
                                      float scale, int32_t max_val);

/* ULP error calculation */
static int ulp_error(float a, float b) {
    if (isnan(a) || isnan(b)) return 0;
    if (isinf(a) || isinf(b)) return 0;
    if (a == b) return 0;

    union { float f; int32_t i; } ua, ub;
    ua.f = a;
    ub.f = b;

    int diff = abs(ua.i - ub.i);
    return diff;
}

/* Scalar reference */
static void exp2_reference(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = exp2f(in[i]);
    }
}

static void exp2_softmax_reference(const int32_t* in, float* out, int n,
                                    float scale, int32_t max_val) {
    for (int i = 0; i < n; i++) {
        float x = ((float)in[i] - (float)max_val) * scale;
        out[i] = exp2f(x);
    }
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

    printf("=== EXP2 FEXPA Kernel Benchmark (FIXED) ===\n");
    printf("Elements: %d (%.2f MB float)\n", n, n * sizeof(float) / 1e6);
    printf("Iterations: %d\n\n", iters);

    float* in_f = aligned_alloc_safe(256, n * sizeof(float));
    int32_t* in_i = aligned_alloc_safe(256, n * sizeof(int32_t));
    float* out_fexpa = aligned_alloc_safe(256, n * sizeof(float));
    float* out_ref = aligned_alloc_safe(256, n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        in_f[i] = ((float)rand() / RAND_MAX * 100.0f) - 50.0f;
        in_i[i] = (rand() % 100000) - 50000;
    }

    /* Test 1: Simple exp2 */
    printf("=== Test 1: Simple exp2 (float input) ===\n");

    exp2_reference(in_f, out_ref, n);
    exp2_fexpa_simple_fixed(in_f, out_fexpa, n);

    int max_ulp = 0;
    long long total_ulp = 0;
    int error_count = 0;
    int high_error_count = 0;

    for (int i = 0; i < n; i++) {
        int ulp = ulp_error(out_ref[i], out_fexpa[i]);
        if (ulp > max_ulp) max_ulp = ulp;
        total_ulp += ulp;
        if (ulp > 1000) error_count++;
        if (ulp > 100) high_error_count++;
    }

    double avg_ulp = (double)total_ulp / n;
    printf("Max ULP error: %d\n", max_ulp);
    printf("Avg ULP error: %.2f\n", avg_ulp);
    printf("High errors (>100 ULP): %d\n", high_error_count);
    printf("Large errors (>1000 ULP): %d\n", error_count);
    printf("Correctness: %s\n", max_ulp < 1000 ? "PASS" : "FAIL");

    /* Show a few samples */
    printf("\nSample values (first 5):\n");
    for (int i = 0; i < 5; i++) {
        printf("  in[%d]=%.4f: ref=%.6f, fexpa=%.6f, ulp=%d\n",
               i, in_f[i], out_ref[i], out_fexpa[i],
               ulp_error(out_ref[i], out_fexpa[i]));
    }
    printf("\n");

    /* Test 2: Softmax exp2 */
    printf("=== Test 2: Softmax exp2 (int32 input) ===\n");

    float scale = 0.01f * LOG2E;
    int32_t max_val = 25000;

    exp2_softmax_reference(in_i, out_ref, n, scale, max_val);
    exp2_fexpa_softmax_fixed(in_i, out_fexpa, n, scale, max_val);

    max_ulp = 0;
    total_ulp = 0;
    error_count = 0;
    high_error_count = 0;

    for (int i = 0; i < n; i++) {
        int ulp = ulp_error(out_ref[i], out_fexpa[i]);
        if (ulp > max_ulp) max_ulp = ulp;
        total_ulp += ulp;
        if (ulp > 1000) error_count++;
        if (ulp > 100) high_error_count++;
    }

    avg_ulp = (double)total_ulp / n;
    printf("Max ULP error: %d\n", max_ulp);
    printf("Avg ULP error: %.2f\n", avg_ulp);
    printf("High errors (>100 ULP): %d\n", high_error_count);
    printf("Large errors (>1000 ULP): %d\n", error_count);
    printf("Correctness: %s\n\n", max_ulp < 1000 ? "PASS" : "FAIL");

    /* Test 3: Performance */
    printf("=== Test 3: Performance ===\n");

    for (int i = 0; i < 5; i++) {
        exp2_fexpa_simple_fixed(in_f, out_fexpa, n);
    }

    double t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_fexpa_simple_fixed(in_f, out_fexpa, n);
    }
    double t1 = omp_get_wtime();
    double simple_time = (t1 - t0) / iters;
    double simple_throughput = n / simple_time / 1e9;

    printf("Simple exp2 (FEXPA fixed):\n");
    printf("  Time: %.3f ms\n", simple_time * 1000);
    printf("  Throughput: %.2f Gelem/s\n", simple_throughput);
    printf("  Cycles/elem: %.2f (at 2.0 GHz)\n", 2.0 / simple_throughput);

    t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_fexpa_softmax_fixed(in_i, out_fexpa, n, scale, max_val);
    }
    t1 = omp_get_wtime();
    double softmax_time = (t1 - t0) / iters;
    double softmax_throughput = n / softmax_time / 1e9;

    printf("\nSoftmax exp2 (FEXPA fixed):\n");
    printf("  Time: %.3f ms\n", softmax_time * 1000);
    printf("  Throughput: %.2f Gelem/s\n", softmax_throughput);
    printf("  Cycles/elem: %.2f (at 2.0 GHz)\n", 2.0 / softmax_throughput);

    t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_reference(in_f, out_ref, n);
    }
    t1 = omp_get_wtime();
    double ref_time = (t1 - t0) / iters;
    double ref_throughput = n / ref_time / 1e9;

    printf("\nReference (libm exp2f):\n");
    printf("  Time: %.3f ms\n", ref_time * 1000);
    printf("  Throughput: %.2f Gelem/s\n", ref_throughput);

    printf("\n=== Summary ===\n");
    printf("Speedup over libm: %.2fx\n", ref_time / simple_time);

    free(in_f);
    free(in_i);
    free(out_fexpa);
    free(out_ref);

    return 0;
}
