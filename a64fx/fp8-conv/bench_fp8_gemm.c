/*
 * Benchmark for FP8 GEMM with FP32 Accumulation
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fp8_gemm.h"
#include "fp8_convert.h"

// LUT for kernel-only benchmark
extern uint32_t fp8_e4m3_to_fp32_lut[256];

#define CPU_FREQ_MHZ 2000
#define ALIGN 64

static inline uint64_t get_timer_ticks(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

// Reference GEMM in FP32
static void gemm_ref(const fp8_e4m3_t* A, int64_t lda,
                     const fp8_e4m3_t* B, int64_t ldb,
                     float* C, int64_t ldc,
                     int64_t M, int64_t N, int64_t K) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                // Convert FP8 to float via FP16
                uint16_t a_fp16 = fp8_e4m3_to_fp16_scalar(A[i * lda + k]);
                uint16_t b_fp16 = fp8_e4m3_to_fp16_scalar(B[k * ldb + j]);
                float a_f32 = *((__fp16*)&a_fp16);
                float b_f32 = *((__fp16*)&b_fp16);
                acc += a_f32 * b_f32;
            }
            C[i * ldc + j] = acc;
        }
    }
}

// Fill with random FP8 values (normal range only for simplicity)
static void fill_random_fp8(fp8_e4m3_t* buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        // Generate random normal FP8 E4M3 values (exp 1-14, mant 0-7)
        uint8_t sign = rand() & 1;
        uint8_t exp = (rand() % 14) + 1;  // 1-14 (avoid 0 and 15)
        uint8_t mant = rand() & 0x7;
        buf[i] = (sign << 7) | (exp << 3) | mant;
    }
}

static float max_rel_error(const float* a, const float* b, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float ref = fabsf(a[i]);
        float diff = fabsf(a[i] - b[i]);
        float rel = (ref > 1e-6f) ? diff / ref : diff;
        if (rel > max_err) max_err = rel;
    }
    return max_err;
}

int main(int argc, char** argv) {
    printf("=== FP8 GEMM Benchmark (FP8 -> FP16 -> FP32 accumulation) ===\n\n");

    // Timer setup
    uint64_t timer_freq = get_timer_freq();
    double ticks_to_cycles = (double)CPU_FREQ_MHZ * 1e6 / timer_freq;

    printf("Timer frequency: %lu Hz\n", timer_freq);
    printf("CPU frequency: %d MHz\n", CPU_FREQ_MHZ);
    printf("SVE vector length: %d bits\n", (int)svcntb() * 8);
    printf("Tile size: MR=%d, NR=%d, N_tile=%d\n\n",
           MR_FP8, NR_FP8, NR_FP8 * VL_FP32);

    // Initialize LUTs
    fp8_gemm_init();

    srand(42);

    // Test sizes
    int64_t M = 384;   // Multiple of MR_FP8=8
    int64_t N = NR_FP8 * VL_FP32;  // 48 = 3 * 16
    int64_t K = 512;

    if (argc > 1) M = atol(argv[1]);
    if (argc > 2) K = atol(argv[2]);

    printf("Problem size: M=%ld, N=%ld, K=%ld\n", M, N, K);
    printf("FLOPs: %.2f MFLOPs\n\n", 2.0 * M * N * K / 1e6);

    // Allocate matrices
    fp8_e4m3_t* A = aligned_alloc_wrapper(ALIGN, M * K);
    fp8_e4m3_t* B = aligned_alloc_wrapper(ALIGN, K * N);
    float* C = aligned_alloc_wrapper(ALIGN, M * N * sizeof(float));
    float* C_ref = aligned_alloc_wrapper(ALIGN, M * N * sizeof(float));

    if (!A || !B || !C || !C_ref) {
        printf("Allocation failed\n");
        return 1;
    }

    // Initialize
    fill_random_fp8(A, M * K);
    fill_random_fp8(B, K * N);
    memset(C, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    // Reference computation
    printf("Computing reference...\n");
    gemm_ref(A, K, B, N, C_ref, N, M, N, K);

    // Warmup
    printf("Warmup...\n");
    fp8_gemm(A, K, B, N, C, N, M, N, K);

    // Verify
    float err = max_rel_error(C_ref, C, M * N);
    printf("Max relative error: %.2e\n", err);
    if (err > 0.01f) {
        printf("WARNING: Error too large!\n");
        // Print first few differences
        int shown = 0;
        for (int64_t i = 0; i < M && shown < 5; i++) {
            for (int64_t j = 0; j < N && shown < 5; j++) {
                float ref = C_ref[i * N + j];
                float got = C[i * N + j];
                if (fabsf(ref - got) > 0.01f * fabsf(ref)) {
                    printf("  [%ld,%ld]: ref=%.6f, got=%.6f\n", i, j, ref, got);
                    shown++;
                }
            }
        }
    }

    // Benchmark
    printf("\nBenchmarking...\n");
    int iterations = 10;

    uint64_t start = get_timer_ticks();
    for (int iter = 0; iter < iterations; iter++) {
        fp8_gemm(A, K, B, N, C, N, M, N, K);
    }
    uint64_t end = get_timer_ticks();

    double ticks_per_call = (double)(end - start) / iterations;
    double cycles_per_call = ticks_per_call * ticks_to_cycles;
    double flops = 2.0 * M * N * K;
    // time_seconds = cycles / (CPU_FREQ_MHZ * 1e6)
    // GFLOPS = flops / 1e9 / time_seconds = flops * CPU_FREQ_MHZ / (cycles * 1e3)
    double gflops = flops * CPU_FREQ_MHZ / (cycles_per_call * 1e3);

    printf("\nResults:\n");
    printf("  Cycles per call: %.0f\n", cycles_per_call);
    printf("  GFLOPS: %.2f\n", gflops);
    printf("  Cycles per FLOP: %.3f\n", cycles_per_call / flops);

    // Calculate kernel-only metrics
    int64_t M_pad = ((M + MR_FP8 - 1) / MR_FP8) * MR_FP8;
    int64_t num_tiles = M_pad / MR_FP8;
    double cycles_per_tile = cycles_per_call / num_tiles;
    double cycles_per_k = cycles_per_tile / K;

    printf("\nPer-tile metrics (MR=%d, N=%d, K=%d):\n", MR_FP8, (int)N, (int)K);
    printf("  Cycles per tile: %.1f\n", cycles_per_tile);
    printf("  Cycles per K iteration: %.2f\n", cycles_per_k);

    // Theoretical analysis
    double peak_gflops = 2.0 * 16.0 * 2.0 * CPU_FREQ_MHZ / 1000.0; // 2 FMA units × 16 FP32 × 2 ops × freq
    printf("\nTheoretical analysis:\n");
    printf("  Peak FP32 GFLOPS (1 core): %.1f\n", peak_gflops);
    printf("  Achieved efficiency: %.1f%%\n", 100.0 * gflops / peak_gflops);

    // ===== Kernel-only benchmark =====
    printf("\n===== Kernel-only Benchmark (FP32 loads only) =====\n");

    // Pre-allocate and pre-convert data for kernel-only timing
    int64_t N_tile = NR_FP8 * VL_FP32;
    // M_pad already defined above
    int64_t M_panels = M_pad / MR_FP8;

    float* Ap_f32 = aligned_alloc_wrapper(ALIGN, M_pad * K * sizeof(float));
    float* Bp_f32 = aligned_alloc_wrapper(ALIGN, K * N_tile * sizeof(float));

    if (!Ap_f32 || !Bp_f32) {
        printf("Kernel benchmark allocation failed\n");
        free(A); free(B); free(C); free(C_ref);
        return 1;
    }

    // Pre-convert A: pack then convert
    for (int64_t ir = 0; ir < M_pad; ir += MR_FP8) {
        for (int64_t k = 0; k < K; k++) {
            for (int m = 0; m < MR_FP8; m++) {
                int64_t row = ir + m;
                uint8_t fp8_val = (row < M) ? A[row * K + k] : 0;
                uint32_t fp32_bits = fp8_e4m3_to_fp32_lut[fp8_val];
                int64_t idx = (ir / MR_FP8) * MR_FP8 * K + k * MR_FP8 + m;
                Ap_f32[idx] = *((float*)&fp32_bits);
            }
        }
    }

    // Pre-convert B: pack then convert
    for (int64_t k = 0; k < K; k++) {
        for (int64_t j = 0; j < N_tile; j++) {
            uint8_t fp8_val = (j < N) ? B[k * N + j] : 0;
            uint32_t fp32_bits = fp8_e4m3_to_fp32_lut[fp8_val];
            Bp_f32[k * N_tile + j] = *((float*)&fp32_bits);
        }
    }

    // Warmup kernel
    memset(C, 0, M * N * sizeof(float));
    for (int64_t ir = 0; ir < M_pad; ir += MR_FP8) {
        const float* Ap_tile = Ap_f32 + (ir / MR_FP8) * (MR_FP8 * K);
        float* C_tile = C + ir * N;
        fp8_gemm_kernel_fp32(Ap_tile, Bp_f32, C_tile, N, K);
    }

    // Kernel-only benchmark
    int kernel_iters = 100;
    start = get_timer_ticks();
    for (int iter = 0; iter < kernel_iters; iter++) {
        for (int64_t ir = 0; ir < M_pad; ir += MR_FP8) {
            const float* Ap_tile = Ap_f32 + (ir / MR_FP8) * (MR_FP8 * K);
            float* C_tile = C + ir * N;
            fp8_gemm_kernel_fp32(Ap_tile, Bp_f32, C_tile, N, K);
        }
    }
    end = get_timer_ticks();

    double kernel_ticks = (double)(end - start) / kernel_iters;
    double kernel_cycles = kernel_ticks * ticks_to_cycles;
    double kernel_gflops = flops * CPU_FREQ_MHZ / (kernel_cycles * 1e3);
    double cycles_per_tile_kernel = kernel_cycles / M_panels;
    double cycles_per_k_kernel = cycles_per_tile_kernel / K;

    printf("Kernel-only results:\n");
    printf("  Cycles per call: %.0f\n", kernel_cycles);
    printf("  GFLOPS: %.2f\n", kernel_gflops);
    printf("  Efficiency: %.1f%%\n", 100.0 * kernel_gflops / peak_gflops);
    printf("  Cycles per tile: %.1f\n", cycles_per_tile_kernel);
    printf("  Cycles per K iteration: %.2f\n", cycles_per_k_kernel);

    // Per K analysis
    int fmlas_per_k = MR_FP8 * NR_FP8;  // 24
    double fmla_throughput = 2.0;  // 2 FMA pipes
    double min_cycles_per_k = fmlas_per_k / fmla_throughput;  // 12 cycles
    printf("\n  Per K iteration analysis:\n");
    printf("    - FMLAs: %d\n", fmlas_per_k);
    printf("    - Theoretical minimum: %.1f cycles (FMLA limited)\n", min_cycles_per_k);
    printf("    - Observed: %.2f cycles\n", cycles_per_k_kernel);
    printf("    - FMLA utilization: %.1f%%\n", 100.0 * min_cycles_per_k / cycles_per_k_kernel);

    // ===== Assembly Kernel Benchmark =====
    printf("\n===== Assembly Kernel Benchmark =====\n");

    // Warmup
    memset(C, 0, M * N * sizeof(float));
    for (int64_t ir = 0; ir < M_pad; ir += MR_FP8) {
        const float* Ap_tile = Ap_f32 + (ir / MR_FP8) * (MR_FP8 * K);
        float* C_tile = C + ir * N;
        fp8_gemm_kernel_asm(Ap_tile, Bp_f32, C_tile, N, K);
    }

    // Verify assembly kernel
    float asm_err = max_rel_error(C_ref, C, M * N);
    printf("Max relative error (asm): %.2e\n", asm_err);

    // Benchmark
    int asm_iters = 100;
    start = get_timer_ticks();
    for (int iter = 0; iter < asm_iters; iter++) {
        for (int64_t ir = 0; ir < M_pad; ir += MR_FP8) {
            const float* Ap_tile = Ap_f32 + (ir / MR_FP8) * (MR_FP8 * K);
            float* C_tile = C + ir * N;
            fp8_gemm_kernel_asm(Ap_tile, Bp_f32, C_tile, N, K);
        }
    }
    end = get_timer_ticks();

    double asm_ticks = (double)(end - start) / asm_iters;
    double asm_cycles = asm_ticks * ticks_to_cycles;
    double asm_gflops = flops * CPU_FREQ_MHZ / (asm_cycles * 1e3);
    double asm_cycles_per_tile = asm_cycles / M_panels;
    double asm_cycles_per_k = asm_cycles_per_tile / K;

    printf("Assembly kernel results:\n");
    printf("  Cycles per call: %.0f\n", asm_cycles);
    printf("  GFLOPS: %.2f\n", asm_gflops);
    printf("  Efficiency: %.1f%% (%.2f / %.1f)\n", 100.0 * asm_gflops / peak_gflops, asm_gflops, peak_gflops);
    printf("  Cycles per tile: %.1f\n", asm_cycles_per_tile);
    printf("  Cycles per K iteration: %.2f\n", asm_cycles_per_tile / (double)K);
    printf("  FMLA utilization: %.1f%%\n", 100.0 * min_cycles_per_k / (asm_cycles_per_tile / (double)K));

    free(Ap_f32);
    free(Bp_f32);
    free(A);
    free(B);
    free(C);
    free(C_ref);

    return 0;
}
