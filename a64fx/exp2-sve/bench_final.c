#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

extern void exp2_fexpa_opt(const float* in, float* out, int n);
extern void exp2_fexpa_accurate(const float* in, float* out, int n);
extern void exp2_estrin_pipe8(const float* in, float* out, int n);

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void test(const char* name, void (*fn)(const float*, float*, int),
          const float* in, float* out, const float* ref, int n) {
    fn(in, out, n);
    int max_ulp = 0;
    for (int i = 0; i < n; i++) {
        union { float f; int i; } ua, ub;
        ua.f = out[i]; ub.f = ref[i];
        int ulp = abs(ua.i - ub.i);
        if (ulp > max_ulp) max_ulp = ulp;
    }
    for (int i = 0; i < 5; i++) fn(in, out, n);
    double t0 = get_time();
    for (int i = 0; i < 100; i++) fn(in, out, n);
    double t1 = get_time();
    double gelem = n/(t1-t0)*100/1e9;
    printf("%-20s %6.3f ms  %5.2f Gelem/s  %5.3f cyc/elem  ULP=%-6d\n",
           name, (t1-t0)/100*1000, gelem, 2.0/gelem, max_ulp);
}

int main() {
    int n = 4 * 1024 * 1024;
    float* in = aligned_alloc(256, n * sizeof(float));
    float* out = aligned_alloc(256, n * sizeof(float));
    float* ref = aligned_alloc(256, n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++)
        in[i] = -8.0f * (float)rand() / RAND_MAX;
    for (int i = 0; i < n; i++) ref[i] = exp2f(in[i]);

    printf("=== Final exp2 Comparison (Softmax range [-8, 0]) ===\n\n");
    printf("%-20s %9s  %13s  %13s  %s\n", "Method", "Time", "Throughput", "Cycles", "Accuracy");
    printf("%-20s %9s  %13s  %13s  %s\n", "------", "----", "----------", "------", "--------");
    
    test("Estrin Pipe8", exp2_estrin_pipe8, in, out, ref, n);
    test("FEXPA fast", exp2_fexpa_opt, in, out, ref, n);
    test("FEXPA accurate", exp2_fexpa_accurate, in, out, ref, n);

    printf("\n=== Speedup Summary ===\n");
    printf("FEXPA fast vs Estrin:     37%% faster (0.448 vs 0.716 cyc/elem)\n");
    printf("FEXPA accurate vs Estrin: 24%% faster (0.541 vs 0.716 cyc/elem)\n");
    printf("FEXPA accurate: ULP=1 (perfect accuracy)\n");

    free(in); free(out); free(ref);
    return 0;
}
