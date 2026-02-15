#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define ALIGN 64

extern void fp8_gemm_kernel_asm(const float* Ap, const float* Bp,
                                float* C, int64_t ldc, int64_t K);

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

int main() {
    int64_t MR = 8, NR = 3, VL = 16, K = 512;
    int64_t N_tile = NR * VL; // 48
    int64_t M_panels = 48;
    
    float* Ap = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* Bp = aligned_alloc(ALIGN, K * N_tile * sizeof(float));
    float* C = aligned_alloc(ALIGN, MR * M_panels * N_tile * sizeof(float));
    
    // Initialize
    for (int i = 0; i < MR * K; i++) Ap[i] = 0.01f * (i % 100);
    for (int i = 0; i < K * N_tile; i++) Bp[i] = 0.01f * (i % 100);
    memset(C, 0, MR * M_panels * N_tile * sizeof(float));
    
    printf("Testing kernel...\n");
    printf("Ap=%p, Bp=%p, C=%p, ldc=%ld, K=%ld\n", Ap, Bp, C, N_tile, K);
    
    // Single call test
    fp8_gemm_kernel_asm(Ap, Bp, C, N_tile, K);
    printf("After 1 call: C[0]=%.4f, C[47]=%.4f\n", C[0], C[47]);
    
    // Check for reasonable values
    int bad = 0;
    for (int i = 0; i < MR * N_tile; i++) {
        if (C[i] < -1e10 || C[i] > 1e10 || C[i] != C[i]) bad++;
    }
    printf("Bad values in output: %d / %ld\n", bad, MR * N_tile);
    
    // Timing test
    printf("\nTiming test...\n");
    int iters = 100;
    
    uint64_t t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(Ap, Bp, C + p * MR * N_tile, N_tile, K);
        }
    }
    uint64_t t1 = get_ticks();
    
    printf("t0=%lu, t1=%lu, diff=%lu\n", t0, t1, t1 - t0);
    
    double ticks = (double)(t1 - t0) / iters;
    double cycles = ticks * 20.0; // 2GHz / 100MHz timer = 20
    double flops = 2.0 * MR * M_panels * N_tile * K;
    double gflops = flops * 2000.0 / (cycles * 1e3);
    
    printf("Ticks: %.0f\n", ticks);
    printf("Cycles: %.0f\n", cycles);
    printf("FLOPs: %.0f\n", flops);
    printf("GFLOPS: %.2f (%.1f%% of 128)\n", gflops, 100.0 * gflops / 128.0);
    
    free(Ap); free(Bp); free(C);
    return 0;
}
