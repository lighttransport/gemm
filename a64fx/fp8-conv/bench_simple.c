/*
 * Simple timing test to isolate corruption
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#define ALIGN 64
#define CPU_FREQ_MHZ 2000

typedef uint8_t fp8_e4m3_t;
uint32_t fp8_to_fp32_lut[256] __attribute__((aligned(64)));

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

void init_lut(void) {
    for (int i = 0; i < 256; i++) {
        uint8_t sign = (i >> 7) & 1;
        uint8_t exp = (i >> 3) & 0xF;
        uint8_t mant = i & 0x7;
        uint32_t fp32;
        if (exp == 0) {
            if (mant == 0) fp32 = (uint32_t)sign << 31;
            else {
                int shift = 0;
                uint8_t m = mant;
                while ((m & 0x4) == 0) { m <<= 1; shift++; }
                fp32 = ((uint32_t)sign << 31) | ((127 - 6 - shift) << 23) | ((uint32_t)(m & 3) << 21);
            }
        } else if (exp == 15 && mant == 7) {
            fp32 = ((uint32_t)sign << 31) | 0x7FC00000;
        } else {
            fp32 = ((uint32_t)sign << 31) | ((exp + 120) << 23) | ((uint32_t)mant << 20);
        }
        fp8_to_fp32_lut[i] = fp32;
    }
}

extern void micro_kernel_fp32_8x3(const float* A, const float* B, float* C,
                                   int64_t K, int64_t unused, int64_t ldc);

extern void fp8_fused_kernel_8x3(const fp8_e4m3_t* A_fp8, const float* B_fp32,
                                  float* C, int64_t ldc, int64_t K, uint32_t* lut);

extern void fp8_fused_kernel_v2(const fp8_e4m3_t* A_fp8, const float* B_fp32,
                                 float* C, int64_t ldc, int64_t K, uint32_t* lut);

extern void fp8_fused_kernel_v3(const fp8_e4m3_t* A_fp8, const float* B_fp32,
                                 float* C, int64_t ldc, int64_t K, uint32_t* lut);

int main() {
    printf("=== Simple Timing Test ===\n\n");
    init_lut();

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t K = 512;
    int64_t N = NR * VL;
    int64_t ldc_bytes = N * sizeof(float);

    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, K * MR);
    float* A_fp32 = aligned_alloc(ALIGN, K * MR * sizeof(float));
    float* B = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* C = aligned_alloc(ALIGN, MR * N * sizeof(float));

    // Initialize
    for (int64_t i = 0; i < K * MR; i++) {
        A_fp8[i] = (8 << 3);  // 2.0
        A_fp32[i] = 2.0f;
    }
    for (int64_t i = 0; i < K * N; i++) {
        B[i] = 1.0f;
    }
    memset(C, 0, MR * N * sizeof(float));

    uint64_t t0, t1;

    // Test 1: Just timing without any kernel
    printf("Test 1: Empty timing loop\n");
    t0 = get_ticks();
    for (int i = 0; i < 1000; i++) {
        __asm__ volatile("" ::: "memory");
    }
    t1 = get_ticks();
    printf("  t0=%lu, t1=%lu, diff=%lu\n", t0, t1, t1-t0);

    // Test 2: FP32 kernel single call
    printf("\nTest 2: FP32 kernel single call\n");
    t0 = get_ticks();
    micro_kernel_fp32_8x3(A_fp32, B, C, K, 0, ldc_bytes);
    t1 = get_ticks();
    printf("  t0=%lu, t1=%lu, diff=%lu\n", t0, t1, t1-t0);
    printf("  C[0]=%f (expected=%f)\n", C[0], (float)(K * 2.0));

    // Test 3: Fused kernel single call
    printf("\nTest 3: Fused kernel single call\n");
    memset(C, 0, MR * N * sizeof(float));
    t0 = get_ticks();
    fp8_fused_kernel_8x3(A_fp8, B, C, ldc_bytes, K, fp8_to_fp32_lut);
    t1 = get_ticks();
    printf("  t0=%lu, t1=%lu, diff=%lu\n", t0, t1, t1-t0);
    printf("  C[0]=%f (expected=%f)\n", C[0], (float)(K * 2.0));

    // Test 4: Multiple FP32 calls
    printf("\nTest 4: FP32 kernel 100 calls\n");
    memset(C, 0, MR * N * sizeof(float));
    t0 = get_ticks();
    for (int i = 0; i < 100; i++) {
        micro_kernel_fp32_8x3(A_fp32, B, C, K, 0, ldc_bytes);
    }
    t1 = get_ticks();
    printf("  t0=%lu, t1=%lu, diff=%lu, per_call=%lu\n", t0, t1, t1-t0, (t1-t0)/100);

    // Test 5: Multiple fused calls
    printf("\nTest 5: Fused kernel v1 100 calls\n");
    memset(C, 0, MR * N * sizeof(float));
    t0 = get_ticks();
    for (int i = 0; i < 100; i++) {
        fp8_fused_kernel_8x3(A_fp8, B, C, ldc_bytes, K, fp8_to_fp32_lut);
    }
    t1 = get_ticks();
    printf("  t0=%lu, t1=%lu, diff=%lu, per_call=%lu\n", t0, t1, t1-t0, (t1-t0)/100);

    // Test 6: Fused v2 single call
    printf("\nTest 6: Fused kernel v2 single call\n");
    memset(C, 0, MR * N * sizeof(float));
    t0 = get_ticks();
    fp8_fused_kernel_v2(A_fp8, B, C, ldc_bytes, K, fp8_to_fp32_lut);
    t1 = get_ticks();
    printf("  t0=%lu, t1=%lu, diff=%lu\n", t0, t1, t1-t0);
    printf("  C[0]=%f (expected=%f)\n", C[0], (float)(K * 2.0));

    // Test 7: Fused v2 100 calls
    printf("\nTest 7: Fused kernel v2 100 calls\n");
    memset(C, 0, MR * N * sizeof(float));
    t0 = get_ticks();
    for (int i = 0; i < 100; i++) {
        fp8_fused_kernel_v2(A_fp8, B, C, ldc_bytes, K, fp8_to_fp32_lut);
    }
    t1 = get_ticks();
    printf("  t0=%lu, t1=%lu, diff=%lu, per_call=%lu\n", t0, t1, t1-t0, (t1-t0)/100);

    // Test 8: Fused v3 single call
    printf("\nTest 8: Fused kernel v3 single call\n");
    memset(C, 0, MR * N * sizeof(float));
    t0 = get_ticks();
    fp8_fused_kernel_v3(A_fp8, B, C, ldc_bytes, K, fp8_to_fp32_lut);
    t1 = get_ticks();
    printf("  t0=%lu, t1=%lu, diff=%lu\n", t0, t1, t1-t0);
    printf("  C[0]=%f (expected=%f)\n", C[0], (float)(K * 2.0));

    // Test 9: Fused v3 100 calls
    printf("\nTest 9: Fused kernel v3 100 calls\n");
    memset(C, 0, MR * N * sizeof(float));
    t0 = get_ticks();
    for (int i = 0; i < 100; i++) {
        fp8_fused_kernel_v3(A_fp8, B, C, ldc_bytes, K, fp8_to_fp32_lut);
    }
    t1 = get_ticks();
    printf("  t0=%lu, t1=%lu, diff=%lu, per_call=%lu\n", t0, t1, t1-t0, (t1-t0)/100);

    // Summary
    printf("\n=== Summary ===\n");
    printf("FLOPs per call: %ld\n", 2L * MR * N * K);
    printf("Ticks to cycles: ~80 (2000MHz / 25MHz counter)\n");

    free(A_fp8); free(A_fp32); free(B); free(C);
    return 0;
}
