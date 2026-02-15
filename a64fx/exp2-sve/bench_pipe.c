/*
 * Benchmark comparing software-pipelined exp2 kernels
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

extern void exp2_estrin_simple(const float* in, float* out, int n);
extern void exp2_estrin_u8(const float* in, float* out, int n);
extern void exp2_estrin_pipe(const float* in, float* out, int n);
extern void exp2_estrin_pipe8(const float* in, float* out, int n);

static int ulp_error(float a, float b) {
    if (a == b) return 0;
    if (isnan(a) || isnan(b)) return 0;
    if (isinf(a) || isinf(b)) return 0;
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

typedef void (*exp2_func)(const float*, float*, int);

static void benchmark(const char* name, exp2_func fn, const float* in, float* out,
                      const float* ref, int n, int iters) {
    // Correctness check
    fn(in, out, n);
    int max_ulp = 0;
    for (int i = 0; i < n; i++) {
        int u = ulp_error(ref[i], out[i]);
        if (u > max_ulp) max_ulp = u;
    }

    // Warmup
    for (int i = 0; i < 5; i++) {
        fn(in, out, n);
    }

    // Benchmark
    double t0 = omp_get_wtime();
    for (int i = 0; i < iters; i++) {
        fn(in, out, n);
    }
    double t1 = omp_get_wtime();

    double time_ms = (t1 - t0) / iters * 1000;
    double gelem_s = n / (t1 - t0) * iters / 1e9;
    double cycles_elem = 2.0 / gelem_s;  // 2 GHz

    printf("%-20s %8.3f ms  %6.2f Gelem/s  %5.3f cyc/elem  ULP=%d\n",
           name, time_ms, gelem_s, cycles_elem, max_ulp);
}

int main(int argc, char** argv) {
    int n = 4 * 1024 * 1024;  // 4M elements
    int iters = 100;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    printf("=== Software Pipelining Comparison ===\n");
    printf("Elements: %d (%.1f MB), Iterations: %d\n", n, n*4.0/1e6, iters);
    printf("A64FX @ 2 GHz, SVE 512-bit (16 floats/vector)\n\n");

    float* in = aligned_alloc_safe(256, n * sizeof(float));
    float* out = aligned_alloc_safe(256, n * sizeof(float));
    float* ref = aligned_alloc_safe(256, n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        in[i] = ((float)rand() / RAND_MAX * 100.0f) - 50.0f;
    }

    // Reference
    for (int i = 0; i < n; i++) ref[i] = exp2f(in[i]);

    printf("%-20s %8s      %6s        %5s        %s\n",
           "Method", "Time", "Thput", "Cyc/el", "Max");

    benchmark("Estrin 4x", exp2_estrin_simple, in, out, ref, n, iters);
    benchmark("Estrin 8x", exp2_estrin_u8, in, out, ref, n, iters);
    benchmark("Pipe 4x", exp2_estrin_pipe, in, out, ref, n, iters);
    benchmark("Pipe 8x", exp2_estrin_pipe8, in, out, ref, n, iters);

    printf("\n");

    free(in); free(out); free(ref);
    return 0;
}
