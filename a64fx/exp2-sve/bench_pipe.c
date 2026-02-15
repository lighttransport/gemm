/*
 * Benchmark: FEXPA chain length comparison
 * 
 * v1: 1-chain (fadd only + fexpa) - no polynomial
 * v2: 3-chain (fadd + fmul + fmla) - wrong poly but fast  
 * v3: 4-chain (fadd + fsub + fmul + fmla) - skip r computation
 * u8: 5-chain (full algorithm) - baseline
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

extern void exp2_fexpa_v1(const float* in, float* out, int n);
extern void exp2_fexpa_v2(const float* in, float* out, int n);
extern void exp2_fexpa_v3(const float* in, float* out, int n);
extern void exp2_fexpa_u8(const float* in, float* out, int n);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_freq(void) {
    uint64_t val;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

typedef void (*exp2_func)(const float*, float*, int);

void benchmark(const char* name, exp2_func func, const float* input, float* output, 
               int n, int chain_len, float* ref) {
    uint64_t freq = get_freq();

    // Warmup
    for (int i = 0; i < 100; i++) func(input, output, n);

    // Benchmark
    uint64_t start = get_cycles();
    for (int i = 0; i < 1000; i++) func(input, output, n);
    uint64_t end = get_cycles();

    uint64_t total = end - start;
    double cyc_per_elem = (double)total / 1000.0 / n;
    double gelem = (double)n * 1000 / ((double)total / freq) / 1e9;

    // Peak calculation based on chain length
    // chain_len FLA ops / 2 pipes = chain_len/2 cyc/vec
    // Peak = 16 elem / (chain_len/2) cyc * 2 GHz = 64/chain_len Gelem/s
    double peak = 64.0 / chain_len;
    double eff = gelem / peak * 100.0;

    // Accuracy check
    double max_err = 0;
    for (int i = 0; i < n; i++) {
        double err = fabs((double)output[i] - (double)ref[i]) / (fabs((double)ref[i]) + 1e-30);
        if (err > max_err) max_err = err;
    }

    printf("  %-12s: %5.2f cyc/el, %6.2f Gelem/s (%5.1f%% of %.1f peak), err=%.1e\n",
           name, cyc_per_elem, gelem, eff, peak, max_err);
}

int main() {
    printf("=== FEXPA Chain Length Comparison ===\n");
    printf("Testing effect of dependency chain on throughput\n\n");

    int n = 4096;  // L1 resident
    float* input = aligned_alloc(64, n * sizeof(float));
    float* output = aligned_alloc(64, n * sizeof(float));
    float* ref = aligned_alloc(64, n * sizeof(float));

    // Initialize
    srand(42);
    for (int i = 0; i < n; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;  // [-2, 2] for accuracy test
    }

    // Reference using exp2f
    for (int i = 0; i < n; i++) ref[i] = exp2f(input[i]);

    printf("n = %d (%.1f KB, L1 resident)\n", n, n * sizeof(float) / 1024.0);
    printf("Chain: FLA ops in dependency chain\n");
    printf("Peak: theoretical max for that chain length\n\n");

    // Test each version
    // v1: 1 FLA op (fadd) + fexpa
    printf("v1 (1-chain: fadd + fexpa only):\n");
    benchmark("fadd+fexpa", exp2_fexpa_v1, input, output, n, 1, ref);

    // v2: 3 FLA ops (fadd parallel with fmul, then fmla)
    // Actually chain is: max(fadd→fexpa, fmul) → fmla
    // If fmul doesn't depend on fadd: chain = 2 (fmul→fmla or fexpa→fmla)
    printf("\nv2 (3-chain: fadd||fmul → fmla):\n");
    benchmark("3-chain", exp2_fexpa_v2, input, output, n, 3, ref);

    // v3: 4 FLA ops (fadd → fsub → fmul → fmla)
    printf("\nv3 (4-chain: fadd → fsub → fmul → fmla):\n");
    benchmark("4-chain", exp2_fexpa_v3, input, output, n, 4, ref);

    // u8: 5 FLA ops (fadd → fsub → fsub → fmul → fmla)
    printf("\nu8 (5-chain: fadd → fsub → fsub → fmul → fmla):\n");
    benchmark("5-chain", exp2_fexpa_u8, input, output, n, 5, ref);

    // Also test larger sizes
    printf("\n=== Larger sizes ===\n");
    int sizes[] = {8192, 65536, 262144};
    for (int s = 0; s < 3; s++) {
        n = sizes[s];
        input = realloc(input, n * sizeof(float));
        output = realloc(output, n * sizeof(float));
        ref = realloc(ref, n * sizeof(float));
        
        for (int i = 0; i < n; i++) {
            input[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
            ref[i] = exp2f(input[i]);
        }
        
        printf("\nn = %d (%.1f KB):\n", n, n * sizeof(float) / 1024.0);
        benchmark("v1 (1-ch)", exp2_fexpa_v1, input, output, n, 1, ref);
        benchmark("v2 (3-ch)", exp2_fexpa_v2, input, output, n, 3, ref);
        benchmark("v3 (4-ch)", exp2_fexpa_v3, input, output, n, 4, ref);
        benchmark("u8 (5-ch)", exp2_fexpa_u8, input, output, n, 5, ref);
    }

    free(input);
    free(output);
    free(ref);
    return 0;
}
