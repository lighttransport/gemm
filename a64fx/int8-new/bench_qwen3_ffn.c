// bench_qwen3_ffn.c - Benchmark Qwen3-style SwiGLU FFN
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ffn_qwen3.h"

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

void benchmark_d256(int M, uint64_t freq) {
    const int D = 256;
    const int D_ff = 1024;

    printf("\n=== Qwen3 SwiGLU FFN D=256 Benchmark: M=%d ===\n", M);

    // Allocate
    int8_t* input = aligned_alloc(64, M * D);
    int8_t* W_gate = aligned_alloc(64, D * D_ff);
    int8_t* W_up = aligned_alloc(64, D * D_ff);
    int8_t* W_down = aligned_alloc(64, D_ff * D);
    int32_t* output = aligned_alloc(64, M * D * sizeof(int32_t));

    for (int i = 0; i < M * D; i++) input[i] = rand() % 256 - 128;
    for (int i = 0; i < D * D_ff; i++) W_gate[i] = rand() % 256 - 128;
    for (int i = 0; i < D * D_ff; i++) W_up[i] = rand() % 256 - 128;
    for (int i = 0; i < D_ff * D; i++) W_down[i] = rand() % 256 - 128;

    // Warmup
    for (int w = 0; w < 2; w++) {
        qwen3_ffn_forward_d256(input, W_gate, W_up, W_down, output, M);
    }

    // Benchmark
    int reps = 5;
    uint64_t best = UINT64_MAX;
    for (int r = 0; r < reps; r++) {
        uint64_t t0 = rdtsc();
        qwen3_ffn_forward_d256(input, W_gate, W_up, W_down, output, M);
        uint64_t t1 = rdtsc();
        if (t1 - t0 < best) best = t1 - t0;
    }

    // SwiGLU FLOPs: 2 GEMMs for gate/up, elementwise, 1 GEMM for down
    // = 2 * (2 * M * D * D_ff) + 2 * M * D_ff * D
    double flops = 2.0 * (2.0 * M * D * D_ff) + 2.0 * M * D_ff * D;
    double sec = (double)best / freq;
    double gflops = flops / sec / 1e9;
    double peak_gops = 128.0;
    double efficiency = gflops / peak_gops * 100;

    printf("Cycles: %lu\n", best);
    printf("Time: %.4f sec\n", sec);
    printf("GFLOPS: %.1f\n", gflops);
    printf("Efficiency: %.1f%%\n", efficiency);
    printf("Throughput: %.1f samples/sec\n", M / sec);

    free(input);
    free(W_gate);
    free(W_up);
    free(W_down);
    free(output);
}

void benchmark_d512(int M, uint64_t freq) {
    const int D = 512;
    const int D_ff = 2048;

    printf("\n=== Qwen3 SwiGLU FFN D=512 Benchmark: M=%d ===\n", M);

    // Allocate
    int8_t* input = aligned_alloc(64, M * D);
    int8_t* W_gate = aligned_alloc(64, D * D_ff);
    int8_t* W_up = aligned_alloc(64, D * D_ff);
    int8_t* W_down = aligned_alloc(64, D_ff * D);
    int32_t* output = aligned_alloc(64, M * D * sizeof(int32_t));

    for (int i = 0; i < M * D; i++) input[i] = rand() % 256 - 128;
    for (int i = 0; i < D * D_ff; i++) W_gate[i] = rand() % 256 - 128;
    for (int i = 0; i < D * D_ff; i++) W_up[i] = rand() % 256 - 128;
    for (int i = 0; i < D_ff * D; i++) W_down[i] = rand() % 256 - 128;

    // Warmup
    for (int w = 0; w < 2; w++) {
        qwen3_ffn_forward_d512(input, W_gate, W_up, W_down, output, M);
    }

    // Benchmark
    int reps = 5;
    uint64_t best = UINT64_MAX;
    for (int r = 0; r < reps; r++) {
        uint64_t t0 = rdtsc();
        qwen3_ffn_forward_d512(input, W_gate, W_up, W_down, output, M);
        uint64_t t1 = rdtsc();
        if (t1 - t0 < best) best = t1 - t0;
    }

    double flops = 2.0 * (2.0 * M * D * D_ff) + 2.0 * M * D_ff * D;
    double sec = (double)best / freq;
    double gflops = flops / sec / 1e9;
    double peak_gops = 128.0;
    double efficiency = gflops / peak_gops * 100;

    printf("Cycles: %lu\n", best);
    printf("Time: %.4f sec\n", sec);
    printf("GFLOPS: %.1f\n", gflops);
    printf("Efficiency: %.1f%%\n", efficiency);
    printf("Throughput: %.1f samples/sec\n", M / sec);

    free(input);
    free(W_gate);
    free(W_up);
    free(W_down);
    free(output);
}

int main() {
    printf("================================================================\n");
    printf("Qwen3-next Style SwiGLU FFN Benchmark\n");
    printf("6-row split loading optimization\n");
    printf("================================================================\n");

    uint64_t freq = get_timer_freq();
    printf("Timer frequency: %lu Hz\n", freq);

    // Print configurations
    printf("\nQwen3-next FFN Configurations:\n");
    for (int i = 0; i < 5; i++) {
        const qwen3_ffn_config* cfg = qwen3_get_config((qwen3_config_t)i);
        printf("  %s\n", cfg->name);
        printf("    D_model=%d, D_ff=%d, %.2fx expansion\n",
               cfg->D_model, cfg->D_ff, cfg->expansion);
    }

    // Batch sizes (must be divisible by 6)
    int M_values[] = {6, 30, 60, 120, 300, 600};
    int num_M = 6;

    // Benchmark D=256
    printf("\n--- D=256 SwiGLU FFN Benchmarks ---\n");
    for (int i = 0; i < num_M; i++) {
        benchmark_d256(M_values[i], freq);
    }

    // Benchmark D=512
    printf("\n--- D=512 SwiGLU FFN Benchmarks ---\n");
    for (int i = 0; i < num_M; i++) {
        benchmark_d512(M_values[i], freq);
    }

    printf("\n================================================================\n");
    printf("Summary:\n");
    printf("- SwiGLU: FFN(x) = (SiLU(x@W_gate) * (x@W_up)) @ W_down\n");
    printf("- 6-row processing with split loading (2.4 SDOT/load)\n");
    printf("- Optimized for 11-cycle A64FX load latency\n");
    printf("- D=256: 1024 FFN dim (4x), 64 K-groups\n");
    printf("- D=512: 2048 FFN dim (4x), 128 K-groups\n");
    printf("================================================================\n");

    return 0;
}
