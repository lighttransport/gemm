/*
 * Benchmark FP8 GEMM with full conversion cost included
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <arm_sve.h>

#define ALIGN 64
#define CPU_FREQ_MHZ 2000

typedef uint8_t fp8_e4m3_t;
uint32_t fp8_to_fp32_lut[256];

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_freq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
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

void convert_fp8_to_fp32_sve(const fp8_e4m3_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();
    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t indices = svld1ub_u32(pg, src + i);
        svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
        svst1(pg, dst + i, svreinterpret_f32(fp32_bits));
    }
}

extern void fp8_gemm_kernel_asm(const float* Ap, const float* Bp,
                                float* C, int64_t ldc, int64_t K);

int main() {
    printf("=== FP8 GEMM Full Cost Benchmark ===\n\n");
    init_lut();

    uint64_t freq = get_freq();
    double ticks_to_cycles = (double)CPU_FREQ_MHZ * 1e6 / freq;

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512;
    int64_t N = NR * VL;
    int64_t M_panels = M / MR;
    double flops = 2.0 * M * N * K;

    printf("M=%ld, N=%ld, K=%ld, FLOPs=%.2fM\n\n", M, N, K, flops/1e6);

    fp8_e4m3_t* A = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* B = aligned_alloc(ALIGN, K * N);
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));
    float* A_buf = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* B_buf = aligned_alloc(ALIGN, K * N * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) A[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    for (int64_t i = 0; i < K * N; i++) B[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);

    int iters = 100;
    volatile uint64_t start, end;
    double cycles;

    // B conversion
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t k = 0; k < K; k++)
            convert_fp8_to_fp32_sve(B + k * N, B_buf + k * N, N);
    }
    end = get_ticks();
    double b_conv_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("B conversion: %.0f cycles (%.2f cycles/elem)\n", b_conv_cycles, b_conv_cycles/(K*N));

    // A conversion
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            for (int64_t k = 0; k < K; k++) {
                for (int m = 0; m < MR; m++) {
                    uint32_t bits = fp8_to_fp32_lut[A[(p*MR+m)*K + k]];
                    A_buf[k * MR + m] = *((float*)&bits);
                }
            }
        }
    }
    end = get_ticks();
    double a_conv_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("A conversion: %.0f cycles (%.2f cycles/elem)\n", a_conv_cycles, a_conv_cycles/(M*K));

    // Pre-convert for kernel test
    for (int64_t k = 0; k < K; k++)
        convert_fp8_to_fp32_sve(B + k * N, B_buf + k * N, N);
    for (int64_t k = 0; k < K; k++) {
        for (int m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A[k]];
            A_buf[k * MR + m] = *((float*)&bits);
        }
    }

    // Kernel only
    printf("\nKernel only (timing %d calls)...\n", iters * (int)M_panels);
    memset(C, 0, M * N * sizeof(float));
    
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_buf, B_buf, C + p * MR * N, N, K);
        }
    }
    end = get_ticks();
    
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    double kernel_gflops = flops * CPU_FREQ_MHZ / (cycles * 1e3);
    printf("Kernel: %.0f cycles, %.2f GFLOPS (%.1f%% of 128)\n", 
           cycles, kernel_gflops, 100.0 * kernel_gflops / 128.0);
    printf("Cycles/K/tile: %.2f (ideal=12)\n", cycles / M_panels / K);

    // Full pass
    printf("\nFull 2-pass...\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t k = 0; k < K; k++)
            convert_fp8_to_fp32_sve(B + k * N, B_buf + k * N, N);
        for (int64_t p = 0; p < M_panels; p++) {
            for (int64_t k = 0; k < K; k++) {
                for (int m = 0; m < MR; m++) {
                    uint32_t bits = fp8_to_fp32_lut[A[(p*MR+m)*K + k]];
                    A_buf[k * MR + m] = *((float*)&bits);
                }
            }
            fp8_gemm_kernel_asm(A_buf, B_buf, C + p * MR * N, N, K);
        }
    }
    end = get_ticks();
    
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    double total_gflops = flops * CPU_FREQ_MHZ / (cycles * 1e3);
    printf("Total: %.0f cycles, %.2f GFLOPS (%.1f%% of 128)\n", 
           cycles, total_gflops, 100.0 * total_gflops / 128.0);

    printf("\n=== Breakdown ===\n");
    double sum = b_conv_cycles + a_conv_cycles + (cycles - b_conv_cycles - a_conv_cycles);
    printf("B convert: %5.1f%% (%.0f cycles)\n", 100.0*b_conv_cycles/cycles, b_conv_cycles);
    printf("A convert: %5.1f%% (%.0f cycles)\n", 100.0*a_conv_cycles/cycles, a_conv_cycles);
    double kernel_est = cycles - b_conv_cycles - a_conv_cycles;
    printf("Kernel:    %5.1f%% (%.0f cycles est)\n", 100.0*kernel_est/cycles, kernel_est);

    free(A); free(B); free(C); free(A_buf); free(B_buf);
    return 0;
}
