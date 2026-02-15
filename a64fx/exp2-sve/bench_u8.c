/*
 * Benchmark for 8x unrolled exp2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

extern void exp2_estrin_simple(const float* in, float* out, int n);
extern void exp2_estrin_u8(const float* in, float* out, int n);

static int ulp_error(float a, float b) {
    if (a == b) return 0;
    union { float f; int32_t i; } ua, ub;
    ua.f = a; ub.f = b;
    return abs(ua.i - ub.i);
}

static void* aligned_alloc_safe(size_t align, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, align, size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

int main(int argc, char** argv) {
    int n = 4 * 1024 * 1024;  // 4M elements
    int iters = 100;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    printf("=== 4x vs 8x Unroll Comparison ===\n");
    printf("Elements: %d (%.1f MB), Iterations: %d\n\n", n, n*4.0/1e6, iters);

    float* in = aligned_alloc_safe(256, n * sizeof(float));
    float* out_u4 = aligned_alloc_safe(256, n * sizeof(float));
    float* out_u8 = aligned_alloc_safe(256, n * sizeof(float));
    float* out_ref = aligned_alloc_safe(256, n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        in[i] = ((float)rand() / RAND_MAX * 100.0f) - 50.0f;
    }

    // Reference
    for (int i = 0; i < n; i++) out_ref[i] = exp2f(in[i]);

    // Correctness check
    exp2_estrin_simple(in, out_u4, n);
    exp2_estrin_u8(in, out_u8, n);

    int max_ulp_u4 = 0, max_ulp_u8 = 0;
    for (int i = 0; i < n; i++) {
        int u4 = ulp_error(out_ref[i], out_u4[i]);
        int u8 = ulp_error(out_ref[i], out_u8[i]);
        if (u4 > max_ulp_u4) max_ulp_u4 = u4;
        if (u8 > max_ulp_u8) max_ulp_u8 = u8;
    }
    printf("Correctness: 4x max ULP=%d, 8x max ULP=%d\n\n", max_ulp_u4, max_ulp_u8);

    // Warmup
    for (int i = 0; i < 5; i++) {
        exp2_estrin_simple(in, out_u4, n);
        exp2_estrin_u8(in, out_u8, n);
    }

    double t0, t1;

    // 4x unroll
    t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_estrin_simple(in, out_u4, n);
    }
    t1 = omp_get_wtime();
    double time_u4 = (t1 - t0) / iters;

    // 8x unroll
    t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        exp2_estrin_u8(in, out_u8, n);
    }
    t1 = omp_get_wtime();
    double time_u8 = (t1 - t0) / iters;

    printf("%-12s %10s %12s %12s\n", "Unroll", "Time(ms)", "Gelem/s", "Cycles/elem");
    printf("%-12s %10.3f %12.2f %12.3f\n", "4x (Estrin)", time_u4*1000, n/time_u4/1e9, 2.0/(n/time_u4/1e9));
    printf("%-12s %10.3f %12.2f %12.3f\n", "8x (Estrin)", time_u8*1000, n/time_u8/1e9, 2.0/(n/time_u8/1e9));

    printf("\nSpeedup 8x vs 4x: %.2fx\n", time_u4 / time_u8);

    free(in); free(out_u4); free(out_u8); free(out_ref);
    return 0;
}
