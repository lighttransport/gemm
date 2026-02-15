/*
 * FP16 GEMM Optimization on A64FX
 *
 * Testing FP16 input with FP32 accumulation
 * Kernel: micro_kernel_fp16fp32_8x3
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include <arm_fp16.h>

#define ALIGN 64

typedef __fp16 fp16_t;

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    __asm__ volatile("isb" ::: "memory");
    return val;
}

// Convert FP16 array to FP32 (for pre-conversion approach)
void convert_fp16_to_fp32(const fp16_t* src, float* dst, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        dst[i] = (float)src[i];
    }
}

// Convert FP16 A panel to FP32 packed format [K][MR]
void convert_A_fp16_to_fp32(const fp16_t* A, float* A_fp32,
                            int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            A_fp32[k * MR + m] = (float)A[(panel*MR+m)*lda + k];
        }
    }
}

// Convert full A matrix from FP16 to FP32
void convert_A_full_fp16_to_fp32(const fp16_t* A, float* A_fp32,
                                  int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t M_panels = M / MR;
    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < MR; m++) {
                dst[k * MR + m] = (float)A[(p*MR+m)*lda + k];
            }
        }
    }
}

// Convert FP16 A panel to FP16 packed format [K][MR]
void pack_A_fp16(const fp16_t* A, fp16_t* A_packed,
                 int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            A_packed[k * MR + m] = A[(panel*MR+m)*lda + k];
        }
    }
}

// Pack full A matrix from row-major to packed FP16
void pack_A_full_fp16(const fp16_t* A, fp16_t* A_packed,
                      int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t M_panels = M / MR;
    for (int64_t p = 0; p < M_panels; p++) {
        fp16_t* dst = A_packed + p * MR * K;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < MR; m++) {
                dst[k * MR + m] = A[(p*MR+m)*lda + k];
            }
        }
    }
}

// Convert FP16 B to FP32
void convert_B_fp16_to_fp32(const fp16_t* B, float* B_fp32, int64_t K, int64_t N) {
    for (int64_t i = 0; i < K * N; i++) {
        B_fp32[i] = (float)B[i];
    }
}

// External kernels
extern void fp8_gemm_kernel_asm(const float* A, const float* B, float* C,
                                 int64_t ldc_elem, int64_t K);

extern void micro_kernel_fp16fp32_8x3(const fp16_t* A, const float* B, float* C,
                                       int64_t K, int64_t unused, int64_t ldc_bytes);

int main() {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║        FP16 GEMM Optimization on A64FX               ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512, N = NR * VL;
    int64_t M_panels = M / MR;
    int64_t ldc_bytes = N * sizeof(float);
    int64_t flops = 2L * M * N * K;
    int64_t ideal = flops / 1280;  // ticks at 128 GFLOPS
    int iters = 100;

    printf("Config: M=%ld, K=%ld, N=%ld (FLOPs=%.1fM)\n", M, K, N, flops/1e6);
    printf("Target: Ideal=%ld ticks, 90%%=%ld ticks\n\n", ideal, ideal*100/90);

    // Allocate
    fp16_t* A_fp16 = aligned_alloc(ALIGN, M * K * sizeof(fp16_t));
    fp16_t* B_fp16 = aligned_alloc(ALIGN, K * N * sizeof(fp16_t));
    fp16_t* A_fp16_packed = aligned_alloc(ALIGN, M * K * sizeof(fp16_t));
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* A_fp32 = aligned_alloc(ALIGN, M * K * sizeof(float));
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));

    // Initialize
    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A_fp16[i] = (fp16_t)(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
    for (int64_t i = 0; i < K * N; i++) {
        B_fp16[i] = (fp16_t)(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }

    // Pre-convert B to FP32
    convert_B_fp16_to_fp32(B_fp16, B_fp32, K, N);

    uint64_t t0, t1;

    printf("═══════════════════════════════════════════════════════\n");
    printf("                 BENCHMARK RESULTS\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    // 1. Pure FP16→FP32 kernel (A already packed)
    printf("1. Pure FP16→FP32 Kernel (A pre-packed):\n");
    pack_A_full_fp16(A_fp16, A_fp16_packed, M, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            micro_kernel_fp16fp32_8x3(A_fp16_packed + p * MR * K, B_fp32,
                                      C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    uint64_t pure_fp16 = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n\n", pure_fp16, ideal * 100 / pure_fp16);

    // 2. Pure FP32 kernel (for comparison)
    printf("2. Pure FP32 Kernel (A pre-converted to FP32):\n");
    convert_A_full_fp16_to_fp32(A_fp16, A_fp32, M, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32,
                               C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t pure_fp32 = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n\n", pure_fp32, ideal * 100 / pure_fp32);

    // 3. FP16 pack per-panel + FP16→FP32 kernel
    printf("3. Per-Panel Pack + FP16→FP32 Kernel:\n");
    fp16_t* A_panel = aligned_alloc(ALIGN, MR * K * sizeof(fp16_t));
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            pack_A_fp16(A_fp16, A_panel, p, K, K, MR);
            micro_kernel_fp16fp32_8x3(A_panel, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    uint64_t pack_fp16 = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n\n", pack_fp16, ideal * 100 / pack_fp16);

    // 4. FP16→FP32 convert per-panel + FP32 kernel
    printf("4. Per-Panel Convert to FP32 + FP32 Kernel:\n");
    float* A_fp32_panel = aligned_alloc(ALIGN, MR * K * sizeof(float));
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_fp16_to_fp32(A_fp16, A_fp32_panel, p, K, K, MR);
            fp8_gemm_kernel_asm(A_fp32_panel, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t conv_fp32 = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n\n", conv_fp32, ideal * 100 / conv_fp32);

    // 5. Measure packing cost
    printf("5. FP16 Packing Cost:\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        pack_A_full_fp16(A_fp16, A_fp16_packed, M, K, K, MR);
    }
    t1 = get_ticks();
    uint64_t pack_cost = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n\n", pack_cost, pack_cost * 20.0 / (M * K));

    // 6. Measure FP16→FP32 conversion cost
    printf("6. FP16→FP32 Conversion Cost:\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_full_fp16_to_fp32(A_fp16, A_fp32, M, K, K, MR);
    }
    t1 = get_ticks();
    uint64_t conv_cost = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n\n", conv_cost, conv_cost * 20.0 / (M * K));

    // Summary table
    printf("═══════════════════════════════════════════════════════\n");
    printf("                      SUMMARY\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    printf("┌────────────────────────────────┬─────────┬──────┐\n");
    printf("│ Approach                       │ Ticks   │ Eff%% │\n");
    printf("├────────────────────────────────┼─────────┼──────┤\n");
    printf("│ Pure FP16→FP32 kernel          │ %7lu │ %3lu%% │\n", pure_fp16, ideal*100/pure_fp16);
    printf("│ Pure FP32 kernel               │ %7lu │ %3lu%% │\n", pure_fp32, ideal*100/pure_fp32);
    printf("│ Per-panel pack + FP16 kernel   │ %7lu │ %3lu%% │\n", pack_fp16, ideal*100/pack_fp16);
    printf("│ Per-panel conv + FP32 kernel   │ %7lu │ %3lu%% │\n", conv_fp32, ideal*100/conv_fp32);
    printf("└────────────────────────────────┴─────────┴──────┘\n\n");

    printf("Component Costs:\n");
    printf("  FP16 packing:      %5lu ticks (%.2f cycles/elem)\n",
           pack_cost, pack_cost * 20.0 / (M * K));
    printf("  FP16→FP32 convert: %5lu ticks (%.2f cycles/elem)\n\n",
           conv_cost, conv_cost * 20.0 / (M * K));

    printf("Analysis:\n");
    printf("  FP16 kernel vs FP32: %.1fx slower\n", (double)pure_fp16 / pure_fp32);
    printf("  FCVT overhead: ~%ld ticks (%ld%%)\n",
           pure_fp16 - pure_fp32, (pure_fp16 - pure_fp32) * 100 / pure_fp32);

    free(A_fp16); free(B_fp16); free(A_fp16_packed);
    free(B_fp32); free(A_fp32); free(C);
    free(A_panel); free(A_fp32_panel);
    return 0;
}
