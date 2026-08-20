// Single-thread int8 SDOT GEMM for the VLM's transformer shapes (K chunked to 256).
// Measures per-core GOPS to compare against the fp16 GEMM.
#include "gemm_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
// clock_gettime(CLOCK_MONOTONIC) is UNRELIABLE on this A64FX node after SVE asm
// (returns huge deltas); use the CNTVCT_EL0 hardware counter (100 MHz).
static double now(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return (double)v*1e-8; }

static void bench(const char *name, int M, int N, int K) {
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)N*K);
    int32_t *C = malloc((size_t)M*N*4);
    for (int i = 0; i < M*K; i++) A[i] = (int8_t)((i*7)%31 - 15);
    for (int i = 0; i < N*K; i++) B[i] = (int8_t)((i*13)%41 - 20);
    // warmup
    memset(C,0,(size_t)M*N*4);
    for (int k0=0;k0<K;k0+=256)
        gemm_6x4_opt_driver(A+k0, K, B+k0, K, C, N, M, N, 256);
    double best = 1e9;
    for (int it=0; it<5; it++) {
        memset(C,0,(size_t)M*N*4);
        double t0 = now();
        for (int k0=0;k0<K;k0+=256)
            gemm_6x4_opt_driver(A+k0, K, B+k0, K, C, N, M, N, 256);
        double t = now()-t0;
        if (t<best) best=t;
    }
    double gops = 2.0*M*N*K/best/1e9;
    printf("%-10s M=%d N=%d K=%d : %7.3f ms  %8.1f GOPS  (1 core, int8 SDOT)\n", name, M,N,K, best*1000, gops);
    free(A); free(B); free(C);
}
int main(void){
    printf("A64FX int8 SDOT peak = 512 GOPS/core (2 FPU x 2 SDOT x 64 MAC)\n");
    bench("ffn_up", 96, 4096, 1024);
    bench("ffn_down",96, 1024, 4096);
    bench("qkv",    96, 3072, 1024);
    return 0;
}
