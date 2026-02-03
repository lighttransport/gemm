/*
 * Benchmark for M=16 fused exp2 + FMLA (using exp2_fmla_fp32_vec)
 * Test if larger M improves fused kernel efficiency
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

extern void exp2_fmla_fp32_vec(
    const int32_t* S, const float* V, float* O,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o);

extern void gemm_fp32_4x4(
    const float* A, const float* B, float* O,
    int K, int ld_a, int ld_b, int ld_o);

extern void exp2_rows(
    const int32_t* S, float* P, int M, int Nc,
    float scale, float max_val, int ld_s, int ld_p);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static void ref_exp2_fmla_fp32(
    const int32_t* S, const float* V, float* O,
    int M, int Nc, int D,
    float scale, float max_val,
    int ld_s, int ld_v, int ld_o)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            O[i * (ld_o / sizeof(float)) + j] = 0.0f;
        }
    }

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < Nc; k++) {
            float s = (float)S[i * ld_s + k];
            float p = exp2f(s * scale - max_val);
            for (int j = 0; j < D; j++) {
                O[i * (ld_o / sizeof(float)) + j] += p * V[k * (ld_v / sizeof(float)) + j];
            }
        }
    }
}

static int verify_fp32(const float* ref, const float* test, int n, float* max_err) {
    *max_err = 0.0f;
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - test[i]);
        float rel = (fabsf(ref[i]) > 1e-6f) ? err / fabsf(ref[i]) : err;
        if (rel > *max_err) *max_err = rel;
        if (rel > 0.02f && err > 1e-4f) errors++;
    }
    return errors;
}

/* Two-pass for M=16: 4 calls to gemm_fp32_4x4 */
static void two_pass_m16(const int32_t* S, const float* V, float* P, float* O,
                         int Nc, float scale, float max_val,
                         int ld_s, int ld_v, int ld_o) {
    exp2_rows(S, P, 16, Nc, scale, max_val, ld_s, Nc);

    /* GEMM for each 4-row chunk */
    for (int i = 0; i < 4; i++) {
        float* P_row = P + i * 4 * Nc;
        float* O_row = (float*)((char*)O + i * 4 * ld_o);
        gemm_fp32_4x4(P_row, V, O_row, Nc, Nc * sizeof(float), ld_v, ld_o);
    }
}

int main(int argc, char** argv) {
    int Nc = 64;
    int iterations = 1000;

    if (argc > 1) Nc = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    int M = 16;
    int D = 64;

    printf("=== M=16 Fused exp2 + FMLA Benchmark ===\n");
    printf("M=%d, Nc=%d, D=%d\n", M, Nc, D);
    printf("Iterations: %d\n\n", iterations);

    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O_ref = aligned_alloc(64, M * D * sizeof(float));
    float* O_test = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));

    srand(42);
    for (int i = 0; i < M * Nc; i++) S[i] = (rand() % 201) - 100;
    for (int i = 0; i < Nc * D; i++) V[i] = ((float)(rand() % 1000) - 500) / 500.0f;

    float scale = 1.0f / 64.0f;
    float max_val = 1.5f;
    int ld_s = Nc;
    int ld_v = D * sizeof(float);
    int ld_o = D * sizeof(float);

    ref_exp2_fmla_fp32(S, V, O_ref, M, Nc, D, scale, max_val, ld_s, ld_v, ld_o);

    double timer_freq = 100e6;
    double cpu_freq = 2e9;
    double fp32_peak = 128.0;
    long long fmla_ops = (long long)M * Nc * D * 2;

    /* Test fused M=16 kernel */
    printf("=== Fused exp2_fmla_fp32_vec (M=16) ===\n");
    memset(O_test, 0, M * D * sizeof(float));
    exp2_fmla_fp32_vec(S, V, O_test, Nc, scale, max_val, ld_s, ld_v, ld_o);

    float max_err;
    int errors = verify_fp32(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: error=%.4f%% %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    for (int i = 0; i < 10; i++)
        exp2_fmla_fp32_vec(S, V, O_test, Nc, scale, max_val, ld_s, ld_v, ld_o);

    volatile uint64_t start, end;
    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        exp2_fmla_fp32_vec(S, V, O_test, Nc, scale, max_val, ld_s, ld_v, ld_o);
    end = get_cycles();

    double elapsed = (double)(end - start);
    double total_cycles = elapsed * (cpu_freq / timer_freq);
    double cycles_per_call = total_cycles / iterations;
    double gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;

    printf("  Cycles: %.1f, per K: %.2f, GFLOPS: %.2f (%.1f%%)\n",
           cycles_per_call, cycles_per_call / Nc, gflops, gflops / fp32_peak * 100);

    /* Test two-pass M=16 */
    printf("\n=== Two-pass (M=16) ===\n");
    memset(O_test, 0, M * D * sizeof(float));
    two_pass_m16(S, V, P, O_test, Nc, scale, max_val, ld_s, ld_v, ld_o);

    errors = verify_fp32(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: error=%.4f%% %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    for (int i = 0; i < 10; i++)
        two_pass_m16(S, V, P, O_test, Nc, scale, max_val, ld_s, ld_v, ld_o);

    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        two_pass_m16(S, V, P, O_test, Nc, scale, max_val, ld_s, ld_v, ld_o);
    end = get_cycles();

    elapsed = (double)(end - start);
    total_cycles = elapsed * (cpu_freq / timer_freq);
    cycles_per_call = total_cycles / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;

    printf("  Cycles: %.1f, per K: %.2f, GFLOPS: %.2f (%.1f%%)\n",
           cycles_per_call, cycles_per_call / Nc, gflops, gflops / fp32_peak * 100);

    /* Pure GEMM baseline */
    printf("\n=== Pure GEMM (M=16) ===\n");
    for (int i = 0; i < M * Nc; i++) P[i] = 1.0f;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 4; j++) {
            float* P_row = P + j * 4 * Nc;
            float* O_row = (float*)((char*)O_test + j * 4 * ld_o);
            gemm_fp32_4x4(P_row, V, O_row, Nc, Nc * sizeof(float), ld_v, ld_o);
        }
    }

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 4; j++) {
            float* P_row = P + j * 4 * Nc;
            float* O_row = (float*)((char*)O_test + j * 4 * ld_o);
            gemm_fp32_4x4(P_row, V, O_row, Nc, Nc * sizeof(float), ld_v, ld_o);
        }
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    total_cycles = elapsed * (cpu_freq / timer_freq);
    cycles_per_call = total_cycles / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;

    printf("  Cycles: %.1f, per K: %.2f, GFLOPS: %.2f (%.1f%%)\n",
           cycles_per_call, cycles_per_call / Nc, gflops, gflops / fp32_peak * 100);

    free(S);
    free(V);
    free(O_ref);
    free(O_test);
    free(P);

    return 0;
}
