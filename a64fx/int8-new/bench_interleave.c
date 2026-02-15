// bench_interleave.c - Compare baseline vs interleaved instruction scheduling
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t rdtsc(void) {
    uint64_t t;
    asm volatile("mrs %0, CNTVCT_EL0" : "=r"(t));
    return t;
}

extern void kernel_6x4_same_addr(const int8_t* A, const int8_t* B, int32_t* C, int K_iters, int ldc);
extern void kernel_6x4_interleave_same(const int8_t* A, const int8_t* B, int32_t* C, int K_iters, int ldc);
extern void kernel_6x4_kloop(const int8_t* A, const int8_t* B, int32_t* C, int K, int ldc);
extern void kernel_6x4_interleave(const int8_t* A, const int8_t* B, int32_t* C, int K, int ldc);

#define MR 6
#define NR 64

int main(void) {
    printf("=== Interleaved vs Baseline Instruction Scheduling ===\n");
    printf("Timer: 100MHz, CPU: 2GHz (ratio=20)\n");
    printf("Ideal: 12 CPU cycles per K-iteration\n\n");

    int8_t* A = aligned_alloc(256, 256);
    int8_t* B = aligned_alloc(256, 512);
    int32_t* C = aligned_alloc(256, 4096);

    for (int i = 0; i < 24; i++) A[i] = (i % 7) - 3;
    for (int i = 0; i < 256; i++) B[i] = ((i * 3) % 7) - 3;

    int k_iters = 10000000;

    printf("=== Same-Address Test (Pure L1D) ===\n");

    // Warmup
    kernel_6x4_same_addr(A, B, C, 100000, NR * 4);
    kernel_6x4_interleave_same(A, B, C, 100000, NR * 4);

    // Baseline
    memset(C, 0, MR * NR * 4);
    uint64_t t0 = rdtsc();
    kernel_6x4_same_addr(A, B, C, k_iters, NR * 4);
    uint64_t t1 = rdtsc();
    uint64_t ticks1 = t1 - t0;
    double cyc1 = (double)ticks1 * 20.0 / k_iters;
    double eff1 = 12.0 / cyc1 * 100.0;
    double giops1 = (double)k_iters * 24 * 128 / ((double)ticks1 / 100000000.0) / 1e9;
    printf("Baseline:    %.1f GIOPS, %.2f cyc/K, %.1f%% eff\n", giops1, cyc1, eff1);

    // Interleaved
    memset(C, 0, MR * NR * 4);
    uint64_t t0_inter = rdtsc();
    kernel_6x4_interleave_same(A, B, C, k_iters, NR * 4);
    uint64_t t1_inter = rdtsc();
    uint64_t interleaved_ticks = t1_inter - t0_inter;
    double interleaved_cyc = (double)interleaved_ticks * 20.0 / k_iters;
    double interleaved_eff = 12.0 / interleaved_cyc * 100.0;
    double interleaved_giops = (double)k_iters * 24 * 128 / ((double)interleaved_ticks / 100000000.0) / 1e9;
    printf("Interleaved: %.1f GIOPS, %.2f cyc/K, %.1f%% eff\n", interleaved_giops, interleaved_cyc, interleaved_eff);

    double improvement = (interleaved_giops - giops1) / giops1 * 100.0;
    printf("Improvement: %.1f%%\n", improvement);

    // Also test with real memory access (K=4096)
    printf("\n=== Streaming Memory Test (K=4096) ===\n");
    int K = 4096;
    size_t A_size = (K/4) * MR * 4;
    size_t B_size = (K/4) * NR * 4;

    int8_t* Ap = aligned_alloc(256, A_size + 256);
    int8_t* Bp = aligned_alloc(256, B_size + 256);

    for (size_t i = 0; i < A_size; i++) Ap[i] = (i % 7) - 3;
    for (size_t i = 0; i < B_size; i++) Bp[i] = ((i * 3) % 7) - 3;

    int iters = 100000;

    // Warmup
    for (int i = 0; i < 1000; i++) {
        kernel_6x4_kloop(Ap, Bp, C, K, NR * 4);
        kernel_6x4_interleave(Ap, Bp, C, K, NR * 4);
    }

    // Baseline
    t0 = rdtsc();
    for (int i = 0; i < iters; i++) {
        kernel_6x4_kloop(Ap, Bp, C, K, NR * 4);
        asm volatile("" ::: "memory");
    }
    t1 = rdtsc();
    ticks1 = t1 - t0;
    int total_k = (K/4) * iters;
    cyc1 = (double)ticks1 * 20.0 / total_k;
    eff1 = 12.0 / cyc1 * 100.0;
    giops1 = (double)total_k * 24 * 128 / ((double)ticks1 / 100000000.0) / 1e9;
    printf("Baseline:    %.1f GIOPS, %.2f cyc/K, %.1f%% eff\n", giops1, cyc1, eff1);

    // Interleaved
    t0 = rdtsc();
    for (int i = 0; i < iters; i++) {
        kernel_6x4_interleave(Ap, Bp, C, K, NR * 4);
        asm volatile("" ::: "memory");
    }
    t1 = rdtsc();
    uint64_t stream_ticks2 = t1 - t0;
    double stream_cyc2 = (double)stream_ticks2 * 20.0 / total_k;
    double stream_eff2 = 12.0 / stream_cyc2 * 100.0;
    double stream_giops2 = (double)total_k * 24 * 128 / ((double)stream_ticks2 / 100000000.0) / 1e9;
    printf("Interleaved: %.1f GIOPS, %.2f cyc/K, %.1f%% eff\n", stream_giops2, stream_cyc2, stream_eff2);

    improvement = (stream_giops2 - giops1) / giops1 * 100.0;
    printf("Improvement: %.1f%%\n", improvement);

    free(A); free(B); free(C); free(Ap); free(Bp);
    return 0;
}
