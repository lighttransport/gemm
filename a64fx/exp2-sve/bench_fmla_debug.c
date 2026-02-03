/*
 * Benchmark for fused exp2 + FMLA GEMM kernels
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "exp2_fmla_fused.h"

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

int main(int argc, char** argv) {
    int M = 4, Nc = 64, D = 64;
    int iterations = 1000;
    
    if (argc > 1) Nc = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);
    
    printf("=== Fused exp2 + FMLA Benchmark (Debug) ===\n");
    printf("M=%d, Nc=%d, D=%d\n", M, Nc, D);
    printf("Iterations: %d\n\n", iterations);
    
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));
    
    srand(42);
    for (int i = 0; i < M * Nc; i++) S[i] = (rand() % 201) - 100;
    for (int i = 0; i < Nc * D; i++) V[i] = ((float)(rand() % 1000) - 500) / 500.0f;
    
    float scale = 1.0f / 64.0f;
    float max_val = 1.5f;
    
    double timer_freq = 100e6, cpu_freq = 2e9;
    double fp32_peak = 128.0;
    
    printf("Testing exp2_fmla_fp32_4x4...\n");
    memset(O, 0, M * D * sizeof(float));
    exp2_fmla_fp32_4x4(S, V, O, Nc, scale, max_val, Nc, D*4, D*4);
    printf("O[0:4]: %.3f %.3f %.3f %.3f\n", O[0], O[1], O[2], O[3]);
    
    /* Benchmark fused version */
    for (int i = 0; i < 10; i++)
        exp2_fmla_fp32_4x4(S, V, O, Nc, scale, max_val, Nc, D*4, D*4);
    
    volatile uint64_t start = get_cycles();
    for (int i = 0; i < iterations; i++)
        exp2_fmla_fp32_4x4(S, V, O, Nc, scale, max_val, Nc, D*4, D*4);
    volatile uint64_t end = get_cycles();
    
    double elapsed = (double)(end - start);
    double total_cycles = elapsed * (cpu_freq / timer_freq);
    double cycles_per_call = total_cycles / iterations;
    long long fmla_ops = (long long)M * Nc * D * 2;
    double gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    
    printf("exp2_fmla_fp32_4x4:\n");
    printf("  Cycles: %.1f, GFLOPS: %.2f (%.1f%%)\n", cycles_per_call, gflops, gflops/fp32_peak*100);
    
    /* Test pure GEMM */
    printf("\nTesting gemm_fp32_4x4...\n");
    for (int i = 0; i < M * Nc; i++) P[i] = 1.0f;
    memset(O, 0, M * D * sizeof(float));
    
    for (int i = 0; i < 10; i++)
        gemm_fp32_4x4(P, V, O, Nc, Nc*4, D*4, D*4);
    
    start = get_cycles();
    for (int i = 0; i < iterations; i++)
        gemm_fp32_4x4(P, V, O, Nc, Nc*4, D*4, D*4);
    end = get_cycles();
    
    elapsed = (double)(end - start);
    total_cycles = elapsed * (cpu_freq / timer_freq);
    cycles_per_call = total_cycles / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    
    printf("gemm_fp32_4x4:\n");
    printf("  Cycles: %.1f per K: %.2f, GFLOPS: %.2f (%.1f%%)\n", 
           cycles_per_call, cycles_per_call/Nc, gflops, gflops/fp32_peak*100);
    
    /* Test two-pass */
    printf("\nTesting two-pass (exp2_rows + gemm)...\n");
    exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
    memset(O, 0, M * D * sizeof(float));
    gemm_fp32_4x4(P, V, O, Nc, Nc*4, D*4, D*4);
    
    for (int i = 0; i < 10; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
        gemm_fp32_4x4(P, V, O, Nc, Nc*4, D*4, D*4);
    }
    
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
        gemm_fp32_4x4(P, V, O, Nc, Nc*4, D*4, D*4);
    }
    end = get_cycles();
    
    elapsed = (double)(end - start);
    total_cycles = elapsed * (cpu_freq / timer_freq);
    cycles_per_call = total_cycles / iterations;
    gflops = (double)fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    
    printf("Two-pass:\n");
    printf("  Cycles: %.1f, GFLOPS: %.2f (%.1f%%)\n", cycles_per_call, gflops, gflops/fp32_peak*100);
    
    free(S); free(V); free(O); free(P);
    return 0;
}
