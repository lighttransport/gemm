#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "exp2_fmla_fused.h"

int main(int argc, char** argv) {
    int M = 4, Nc = 64, D = 64;
    int iterations = 1000;
    
    if (argc > 1) Nc = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);
    
    printf("=== Benchmark (Nc=%d, iter=%d) ===\n", Nc, iterations);
    
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));
    
    srand(42);
    for (int i = 0; i < M * Nc; i++) S[i] = (rand() % 201) - 100;
    for (int i = 0; i < Nc * D; i++) V[i] = ((float)(rand() % 1000) - 500) / 500.0f;
    
    float scale = 1.0f / 64.0f;
    float max_val = 1.5f;
    
    /* Test fused version */
    printf("Testing fused...\n"); fflush(stdout);
    memset(O, 0, M * D * sizeof(float));
    
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iterations; i++)
        exp2_fmla_fp32_4x4(S, V, O, Nc, scale, max_val, Nc, D*4, D*4);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    
    double fused_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    long long ops = (long long)M * Nc * D * 2 * iterations;
    printf("Fused: %.3f ms, %.2f GFLOPS\n", fused_time*1000, ops/fused_time/1e9);
    
    /* Test pure GEMM */
    printf("Testing GEMM...\n"); fflush(stdout);
    for (int i = 0; i < M * Nc; i++) P[i] = 1.0f;
    memset(O, 0, M * D * sizeof(float));
    
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iterations; i++)
        gemm_fp32_4x4(P, V, O, Nc, Nc*4, D*4, D*4);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    
    double gemm_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("GEMM:  %.3f ms, %.2f GFLOPS (%.1f%%)\n", gemm_time*1000, ops/gemm_time/1e9, ops/gemm_time/1e9/128*100);
    
    /* Test two-pass */
    printf("Testing two-pass...\n"); fflush(stdout);
    
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iterations; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
        gemm_fp32_4x4(P, V, O, Nc, Nc*4, D*4, D*4);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    
    double twopass_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("2-pass: %.3f ms, %.2f GFLOPS (%.1f%%)\n", twopass_time*1000, ops/twopass_time/1e9, ops/twopass_time/1e9/128*100);
    
    printf("Done!\n");
    free(S); free(V); free(O); free(P);
    return 0;
}
