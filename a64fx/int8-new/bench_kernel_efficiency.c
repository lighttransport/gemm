// bench_kernel_efficiency.c - Measure INT8 GEMM kernel efficiency
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t rdtsc(void) {
    uint64_t t;
    __asm__ volatile("mrs %0, CNTVCT_EL0" : "=r"(t));
    return t;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t f;
    __asm__ volatile("mrs %0, CNTFRQ_EL0" : "=r"(f));
    return f;
}

extern void kernel_ffn_6row_gemm_d256(const int8_t* A, const int8_t* B, int32_t* C);
extern void kernel_ffn_6row_gemm_d512(const int8_t* A, const int8_t* B, int32_t* C);

void benchmark_d256_kernel(uint64_t freq) {
    const int M = 6;
    const int K = 256;
    const int N = 1024;
    const int K_groups = K / 4;

    printf("\n=== D=256 Kernel Efficiency ===\n");
    printf("Configuration: M=6, K=256, N=1024\n");

    int8_t* A = aligned_alloc(64, M * K);
    int8_t* B = aligned_alloc(64, K_groups * N * 4);
    int32_t* C = aligned_alloc(64, M * N * sizeof(int32_t));

    for (int i = 0; i < M * K; i++) A[i] = rand() % 256 - 128;
    for (int i = 0; i < K_groups * N * 4; i++) B[i] = rand() % 256 - 128;

    // Warmup
    for (int w = 0; w < 5; w++) {
        kernel_ffn_6row_gemm_d256(A, B, C);
    }

    // Benchmark
    int reps = 100;
    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++) {
        kernel_ffn_6row_gemm_d256(A, B, C);
    }
    uint64_t t1 = rdtsc();

    uint64_t cycles = (t1 - t0) / reps;
    double time_sec = (double)cycles / freq;

    // Calculate INT8 operations
    // GEMM: C[M,N] = A[M,K] @ B[K,N]
    // Each output element: K multiply-adds = 2K INT8 ops
    // Total: M * N * 2K INT8 ops
    uint64_t int8_ops = (uint64_t)M * N * K * 2;

    // Or count as MACs: M * N * K multiply-accumulates
    uint64_t int8_macs = (uint64_t)M * N * K;

    double gops = int8_ops / time_sec / 1e9;
    double gmacs = int8_macs / time_sec / 1e9;

    // Peak: 512 GOPS (INT8), 128 ops/cycle at 2 GHz
    double peak_gops = 512.0;
    double peak_ops_per_cycle = 128.0;
    double efficiency = (gops / peak_gops) * 100;

    printf("Cycles per call: %lu\n", cycles);
    printf("INT8 operations: %lu (or %lu MACs)\n", int8_ops, int8_macs);
    printf("INT8 ops/cycle: %.2f\n", (double)int8_ops / cycles);
    printf("GOPS: %.2f\n", gops);
    printf("GMAC/s: %.2f\n", gmacs);
    printf("Efficiency: %.2f%% of peak (512 GOPS)\n", efficiency);
    printf("Ops/cycle utilization: %.2f / 128 = %.2f%%\n",
           (double)int8_ops / cycles, ((double)int8_ops / cycles) / peak_ops_per_cycle * 100);

    // Calculate based on SDOT instructions
    // Each SDOT processes 16 lanes × 4 elements = 64 INT8 MACs (or 128 INT8 ops)
    int sdots_per_call = 24 * K_groups * (N / 64);  // 24 SDOTs per inner loop
    printf("\nSDOT analysis:\n");
    printf("SDOTs per call: %d\n", sdots_per_call);
    printf("SDOTs/cycle: %.2f\n", (double)sdots_per_call / cycles);
    printf("Theoretical max: 2 SDOTs/cycle (with 2 FPUs)\n");

    free(A); free(B); free(C);
}

void benchmark_d512_kernel(uint64_t freq) {
    const int M = 6;
    const int K = 512;
    const int N = 2048;
    const int K_groups = K / 4;

    printf("\n=== D=512 Kernel Efficiency ===\n");
    printf("Configuration: M=6, K=512, N=2048\n");

    int8_t* A = aligned_alloc(64, M * K);
    int8_t* B = aligned_alloc(64, K_groups * N * 4);
    int32_t* C = aligned_alloc(64, M * N * sizeof(int32_t));

    for (int i = 0; i < M * K; i++) A[i] = rand() % 256 - 128;
    for (int i = 0; i < K_groups * N * 4; i++) B[i] = rand() % 256 - 128;

    // Warmup
    for (int w = 0; w < 5; w++) {
        kernel_ffn_6row_gemm_d512(A, B, C);
    }

    // Benchmark
    int reps = 100;
    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++) {
        kernel_ffn_6row_gemm_d512(A, B, C);
    }
    uint64_t t1 = rdtsc();

    uint64_t cycles = (t1 - t0) / reps;
    double time_sec = (double)cycles / freq;

    uint64_t int8_ops = (uint64_t)M * N * K * 2;
    uint64_t int8_macs = (uint64_t)M * N * K;

    double gops = int8_ops / time_sec / 1e9;
    double gmacs = int8_macs / time_sec / 1e9;

    double peak_gops = 512.0;
    double peak_ops_per_cycle = 128.0;
    double efficiency = (gops / peak_gops) * 100;

    printf("Cycles per call: %lu\n", cycles);
    printf("INT8 operations: %lu (or %lu MACs)\n", int8_ops, int8_macs);
    printf("INT8 ops/cycle: %.2f\n", (double)int8_ops / cycles);
    printf("GOPS: %.2f\n", gops);
    printf("GMAC/s: %.2f\n", gmacs);
    printf("Efficiency: %.2f%% of peak (512 GOPS)\n", efficiency);
    printf("Ops/cycle utilization: %.2f / 128 = %.2f%%\n",
           (double)int8_ops / cycles, ((double)int8_ops / cycles) / peak_ops_per_cycle * 100);

    int sdots_per_call = 24 * K_groups * (N / 64);
    printf("\nSDOT analysis:\n");
    printf("SDOTs per call: %d\n", sdots_per_call);
    printf("SDOTs/cycle: %.2f\n", (double)sdots_per_call / cycles);
    printf("Theoretical max: 2 SDOTs/cycle (with 2 FPUs)\n");

    free(A); free(B); free(C);
}

int main() {
    printf("================================================================\n");
    printf("INT8 GEMM Kernel Efficiency Benchmark\n");
    printf("6-row split loading optimization\n");
    printf("================================================================\n");

    uint64_t freq = get_timer_freq();
    printf("Timer frequency: %lu Hz (%.1f GHz)\n", freq, freq / 1e9);

    printf("\nA64FX INT8 Peak Performance:\n");
    printf("- 512 GOPS (INT8 operations per second)\n");
    printf("- 128 INT8 ops/cycle at 2 GHz\n");
    printf("- 2 FPUs × 1 SDOT/cycle × 64 INT8 MACs = 128 INT8 MACs/cycle\n");
    printf("- Or: 2 FPUs × 2 GHz × 64 INT8 MACs = 256 GMAC/s\n");
    printf("- With 2 SDOTs/cycle: 256 INT8 MACs/cycle = 512 GMAC/s\n");

    benchmark_d256_kernel(freq);
    benchmark_d512_kernel(freq);

    printf("\n================================================================\n");
    printf("Summary:\n");
    printf("- 6-row processing: 24 SDOTs per K-group iteration\n");
    printf("- D=256: 64 K-groups × 16 N-chunks = 1,024 iterations\n");
    printf("- D=512: 128 K-groups × 32 N-chunks = 4,096 iterations\n");
    printf("- Each SDOT: 64 INT8 MACs (16 lanes × 4 MACs/lane)\n");
    printf("- Split loading: 24 SDOTs / 10 loads = 2.4 SDOT/load\n");
    printf("================================================================\n");

    return 0;
}
