/*
 * Simple test of fused FP8 kernel
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#define ALIGN 64

typedef uint8_t fp8_e4m3_t;
uint32_t fp8_to_fp32_lut[256] __attribute__((aligned(64)));

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

extern void fp8_fused_kernel_8x3(const fp8_e4m3_t* A_fp8, const float* B_fp32,
                                  float* C, int64_t ldc, int64_t K, uint32_t* lut);

int main() {
    printf("=== Simple Fused Kernel Test ===\n\n");
    init_lut();

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t K = 4;  // Small K for testing
    int64_t N = NR * VL;

    // A is [K][MR] FP8 packed
    fp8_e4m3_t* A = aligned_alloc(ALIGN, K * MR);
    float* B = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* C = aligned_alloc(ALIGN, MR * N * sizeof(float));

    // Initialize with simple values
    // A[k][m] = (k + 1) converted to FP8 (exponent 8 = +7 bias, mantissa 0 = 1.0)
    // FP8 E4M3: exp=8 gives float exp=120+8=128, so value = 2^(128-127) = 2.0
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            // Use exp=8, mant=0 -> gives 2.0
            A[k * MR + m] = (8 << 3) | 0;  // = 64 in decimal
        }
    }

    // B[k][n] = 1.0 for all
    for (int64_t i = 0; i < K * N; i++) {
        B[i] = 1.0f;
    }

    memset(C, 0, MR * N * sizeof(float));

    printf("Input A (FP8 packed [K][MR]):\n");
    for (int64_t k = 0; k < K; k++) {
        printf("  k=%ld: ", k);
        for (int64_t m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A[k * MR + m]];
            float val = *((float*)&bits);
            printf("%.1f ", val);
        }
        printf("\n");
    }

    printf("\nCalling fused kernel...\n");
    int64_t ldc_bytes = N * sizeof(float);
    fp8_fused_kernel_8x3(A, B, C, ldc_bytes, K, fp8_to_fp32_lut);

    printf("\nOutput C[MR][N]:\n");
    for (int64_t m = 0; m < MR; m++) {
        printf("  row %ld: ", m);
        for (int64_t n = 0; n < 8; n++) {  // Print first 8 columns
            printf("%.1f ", C[m * N + n]);
        }
        printf("...\n");
    }

    // Expected: C[m][n] = sum_k(A[k][m] * B[k][n]) = K * 2.0 * 1.0 = 8.0
    printf("\nExpected: %.1f (K * 2.0 * 1.0)\n", (float)(K * 2.0));

    free(A); free(B); free(C);
    return 0;
}
