#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#define ALIGN 64
typedef uint8_t fp8_t;

uint32_t fp8_to_fp32_lut[256];

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
        if (exp == 0 && mant == 0) fp32 = (uint32_t)sign << 31;
        else if (exp == 0) {
            int shift = 0;
            uint8_t m = mant;
            while ((m & 4) == 0) { m <<= 1; shift++; }
            fp32 = ((uint32_t)sign << 31) | ((127-6-shift) << 23) | ((uint32_t)(m&3) << 21);
        } else if (exp == 15 && mant == 7) fp32 = ((uint32_t)sign << 31) | 0x7FC00000;
        else fp32 = ((uint32_t)sign << 31) | ((exp + 120) << 23) | ((uint32_t)mant << 20);
        fp8_to_fp32_lut[i] = fp32;
    }
}

void convert_fp8_to_fp32(const fp8_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();
    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t idx = svld1ub_u32(pg, src + i);
        svuint32_t bits = svld1_gather_index(pg, fp8_to_fp32_lut, idx);
        svst1(pg, dst + i, svreinterpret_f32(bits));
    }
}

extern void fp8_gemm_kernel_asm(const float* Ap, const float* Bp,
                                float* C, int64_t ldc, int64_t K);

int main() {
    init_lut();
    
    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512, N = NR * VL;
    int64_t M_panels = M / MR;
    double flops = 2.0 * M * N * K;
    
    printf("=== FP8 GEMM Full Cost ===\n");
    printf("M=%ld, N=%ld, K=%ld, FLOPs=%.2fM\n\n", M, N, K, flops/1e6);
    
    fp8_t* A = aligned_alloc(ALIGN, M * K);
    fp8_t* B = aligned_alloc(ALIGN, K * N);
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));
    float* A_buf = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* B_buf = aligned_alloc(ALIGN, K * N * sizeof(float));
    
    srand(42);
    for (int64_t i = 0; i < M * K; i++) A[i] = ((rand()%14+1)<<3) | (rand()&7);
    for (int64_t i = 0; i < K * N; i++) B[i] = ((rand()%14+1)<<3) | (rand()&7);
    
    int iters = 100;
    uint64_t t0, t1;
    
    // Conversion cost - B
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t k = 0; k < K; k++)
            convert_fp8_to_fp32(B + k*N, B_buf + k*N, N);
    }
    t1 = get_ticks();
    double b_ticks = (double)(t1 - t0) / iters;
    
    // Conversion cost - A (with packing)
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            for (int64_t k = 0; k < K; k++) {
                for (int m = 0; m < MR; m++) {
                    uint32_t bits = fp8_to_fp32_lut[A[(p*MR+m)*K + k]];
                    A_buf[k*MR + m] = *((float*)&bits);
                }
            }
        }
    }
    t1 = get_ticks();
    double a_ticks = (double)(t1 - t0) / iters;
    
    // Kernel only
    for (int64_t k = 0; k < K; k++) convert_fp8_to_fp32(B + k*N, B_buf + k*N, N);
    for (int64_t k = 0; k < K; k++) {
        for (int m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A[k]];
            A_buf[k*MR + m] = *((float*)&bits);
        }
    }
    
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++)
            fp8_gemm_kernel_asm(A_buf, B_buf, C + p*MR*N, N, K);
    }
    t1 = get_ticks();
    double kernel_ticks = (double)(t1 - t0) / iters;
    
    // Full 2-pass
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t k = 0; k < K; k++)
            convert_fp8_to_fp32(B + k*N, B_buf + k*N, N);
        
        for (int64_t p = 0; p < M_panels; p++) {
            for (int64_t k = 0; k < K; k++) {
                for (int m = 0; m < MR; m++) {
                    uint32_t bits = fp8_to_fp32_lut[A[(p*MR+m)*K + k]];
                    A_buf[k*MR + m] = *((float*)&bits);
                }
            }
            fp8_gemm_kernel_asm(A_buf, B_buf, C + p*MR*N, N, K);
        }
    }
    t1 = get_ticks();
    double full_ticks = (double)(t1 - t0) / iters;
    
    // Convert to cycles (timer=100MHz, CPU=2GHz)
    double conv = 20.0;
    double b_cycles = b_ticks * conv;
    double a_cycles = a_ticks * conv;
    double kernel_cycles = kernel_ticks * conv;
    double full_cycles = full_ticks * conv;
    
    double kernel_gflops = flops * 2000.0 / (kernel_cycles * 1e3);
    double full_gflops = flops * 2000.0 / (full_cycles * 1e3);
    
    printf("=== Results ===\n");
    printf("B conversion: %8.0f cycles (%.2f cyc/elem)\n", b_cycles, b_cycles/(K*N));
    printf("A conversion: %8.0f cycles (%.2f cyc/elem)\n", a_cycles, a_cycles/(M*K));
    printf("Kernel only:  %8.0f cycles -> %.1f GFLOPS (%.1f%% of 128)\n", 
           kernel_cycles, kernel_gflops, 100*kernel_gflops/128);
    printf("Full 2-pass:  %8.0f cycles -> %.1f GFLOPS (%.1f%% of 128)\n",
           full_cycles, full_gflops, 100*full_gflops/128);
    
    printf("\n=== Breakdown ===\n");
    printf("B convert: %5.1f%%\n", 100*b_cycles/full_cycles);
    printf("A convert: %5.1f%%\n", 100*a_cycles/full_cycles);
    printf("Kernel:    %5.1f%%\n", 100*kernel_cycles/full_cycles);
    
    free(A); free(B); free(C); free(A_buf); free(B_buf);
    return 0;
}
