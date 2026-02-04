/*
 * Benchmark different unroll factors for FEXPA exp2
 *
 * A64FX latencies: fadd/fsub/fmul/fmla = 9 cycles, fexpa = 4 cycles
 * Critical path: fadd→fsub→fsub→fmul→fmla = 45 cycles
 * Throughput: 6 FLA ops / 2 pipes = 3 cycles/vector
 * Need 45/3 = 15 vectors in flight for peak
 * Peak: 16 elem / 3 cyc * 2 GHz = 10.67 Gelem/s
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

extern void exp2_fexpa_opt(const float* in, float* out, int n);    // 4x unroll
extern void exp2_fexpa_u8(const float* in, float* out, int n);     // 8x unroll
extern void exp2_fexpa_u16(const float* in, float* out, int n);    // 16x unroll
extern void exp2_fexpa_pipe(const float* in, float* out, int n);   // 16x interleaved

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define PEAK_GELEM 10.67  // 16 elem / 3 cyc * 2 GHz

void test(const char* name, void (*fn)(const float*, float*, int),
          const float* in, float* out, const float* ref, int n, int check_ulp) {
    // Correctness check
    int max_ulp = 0;
    if (check_ulp) {
        fn(in, out, n);
        for (int i = 0; i < n; i++) {
            union { float f; int i; } ua, ub;
            ua.f = out[i]; ub.f = ref[i];
            int ulp = abs(ua.i - ub.i);
            if (ulp > max_ulp) max_ulp = ulp;
        }
    }

    // Warmup
    for (int i = 0; i < 10; i++) fn(in, out, n);

    // Benchmark
    double t0 = get_time();
    for (int i = 0; i < 200; i++) fn(in, out, n);
    double t1 = get_time();

    double gelem = n / (t1 - t0) * 200 / 1e9;
    double cyc_per_elem = 2.0 / gelem;
    double pct_peak = gelem / PEAK_GELEM * 100;

    if (check_ulp) {
        printf("%-18s %6.3f ms  %5.2f Gelem/s  %5.3f cyc/elem  %5.1f%%  ULP=%d\n",
               name, (t1 - t0) / 200 * 1000, gelem, cyc_per_elem, pct_peak, max_ulp);
    } else {
        printf("%-18s %6.3f ms  %5.2f Gelem/s  %5.3f cyc/elem  %5.1f%%\n",
               name, (t1 - t0) / 200 * 1000, gelem, cyc_per_elem, pct_peak);
    }
}

void test_size(const char* label, int n, int iters) {
    float* in = aligned_alloc(256, n * sizeof(float));
    float* out = aligned_alloc(256, n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++)
        in[i] = -8.0f * (float)rand() / RAND_MAX;

    printf("\n=== %s (n=%d, %.1f KB) ===\n", label, n, n * 4.0 / 1024);
    printf("%-18s %9s  %13s  %13s  %6s\n",
           "Method", "Time", "Throughput", "Cycles", "Peak%");

    // Warmup
    for (int i = 0; i < 20; i++) {
        exp2_fexpa_opt(in, out, n);
        exp2_fexpa_u8(in, out, n);
        exp2_fexpa_u16(in, out, n);
        exp2_fexpa_pipe(in, out, n);
    }

    // Benchmark each
    double t0, t1, gelem, cyc, pct;

    t0 = get_time();
    for (int i = 0; i < iters; i++) exp2_fexpa_opt(in, out, n);
    t1 = get_time();
    gelem = (double)n * iters / (t1 - t0) / 1e9;
    cyc = 2.0 / gelem;
    pct = gelem / PEAK_GELEM * 100;
    printf("%-18s %6.3f ms  %5.2f Gelem/s  %5.3f cyc/elem  %5.1f%%\n",
           "4x unroll", (t1 - t0) / iters * 1000, gelem, cyc, pct);

    t0 = get_time();
    for (int i = 0; i < iters; i++) exp2_fexpa_u8(in, out, n);
    t1 = get_time();
    gelem = (double)n * iters / (t1 - t0) / 1e9;
    cyc = 2.0 / gelem;
    pct = gelem / PEAK_GELEM * 100;
    printf("%-18s %6.3f ms  %5.2f Gelem/s  %5.3f cyc/elem  %5.1f%%\n",
           "8x unroll", (t1 - t0) / iters * 1000, gelem, cyc, pct);

    t0 = get_time();
    for (int i = 0; i < iters; i++) exp2_fexpa_u16(in, out, n);
    t1 = get_time();
    gelem = (double)n * iters / (t1 - t0) / 1e9;
    cyc = 2.0 / gelem;
    pct = gelem / PEAK_GELEM * 100;
    printf("%-18s %6.3f ms  %5.2f Gelem/s  %5.3f cyc/elem  %5.1f%%\n",
           "16x unroll", (t1 - t0) / iters * 1000, gelem, cyc, pct);

    t0 = get_time();
    for (int i = 0; i < iters; i++) exp2_fexpa_pipe(in, out, n);
    t1 = get_time();
    gelem = (double)n * iters / (t1 - t0) / 1e9;
    cyc = 2.0 / gelem;
    pct = gelem / PEAK_GELEM * 100;
    printf("%-18s %6.3f ms  %5.2f Gelem/s  %5.3f cyc/elem  %5.1f%%\n",
           "16x interleaved", (t1 - t0) / iters * 1000, gelem, cyc, pct);

    free(in);
    free(out);
}

int main(int argc, char** argv) {
    printf("=== FEXPA exp2 Unroll Benchmark ===\n");
    printf("\nA64FX: fadd/fsub/fmul/fmla = 9 cyc latency, 2 FLA pipes\n");
    printf("Critical path: 45 cycles, Throughput: 3 cyc/vec\n");
    printf("Peak: %.2f Gelem/s (need 15x unroll to saturate)\n", PEAK_GELEM);

    // Test correctness with large array
    int n = 4 * 1024 * 1024;
    float* in = aligned_alloc(256, n * sizeof(float));
    float* out = aligned_alloc(256, n * sizeof(float));
    float* ref = aligned_alloc(256, n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++)
        in[i] = -8.0f * (float)rand() / RAND_MAX;
    for (int i = 0; i < n; i++)
        ref[i] = exp2f(in[i]);

    printf("\n=== Correctness + Performance (n=%d, %.1f MB) ===\n", n, n * 4.0 / 1e6);
    printf("%-18s %9s  %13s  %13s  %6s  %s\n",
           "Method", "Time", "Throughput", "Cycles", "Peak%", "ULP");

    test("4x unroll", exp2_fexpa_opt, in, out, ref, n, 1);
    test("8x unroll", exp2_fexpa_u8, in, out, ref, n, 1);
    test("16x unroll", exp2_fexpa_u16, in, out, ref, n, 1);
    test("16x interleaved", exp2_fexpa_pipe, in, out, ref, n, 1);

    free(in); free(out); free(ref);

    // Test different sizes to identify memory vs compute bottleneck
    test_size("L1 resident", 8192, 100000);           // 32 KB < 64 KB L1
    test_size("L2 resident", 512 * 1024, 2000);       // 2 MB < 8 MB L2
    test_size("Memory bound", 4 * 1024 * 1024, 200);  // 16 MB > L2

    printf("\n=== Analysis ===\n");
    printf("If L1 >> Memory: memory bandwidth limited\n");
    printf("If L1 ≈ Memory: compute limited, need better scheduling\n");
    printf("\nMemory BW at peak: %.1f GB/s (read) + %.1f GB/s (write)\n",
           PEAK_GELEM * 4, PEAK_GELEM * 4);

    return 0;
}
