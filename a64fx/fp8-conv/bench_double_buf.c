/*
 * FP8 GEMM with Double Buffering - overlap conversion with GEMM
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include <omp.h>

#define ALIGN 64

typedef uint8_t fp8_e4m3_t;
uint32_t fp8_to_fp32_lut[256] __attribute__((aligned(64)));

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    __asm__ volatile("isb" ::: "memory");
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

void convert_A_panel(const fp8_e4m3_t* A, float* A_fp32,
                     int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A[(panel*MR+m)*lda + k]];
            A_fp32[k * MR + m] = *((float*)&bits);
        }
    }
}

void convert_B_fp8_to_fp32(const fp8_e4m3_t* B, float* B_fp32, int64_t K, int64_t N) {
    int64_t vl = svcntw();
    for (int64_t k = 0; k < K; k++) {
        for (int64_t n = 0; n < N; n += vl) {
            svbool_t pg = svwhilelt_b32(n, N);
            svuint32_t indices = svld1ub_u32(pg, B + k * N + n);
            svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
            svst1(pg, B_fp32 + k * N + n, svreinterpret_f32(fp32_bits));
        }
    }
}

extern void fp8_gemm_kernel_asm(const float* A, const float* B, float* C,
                                 int64_t ldc_elem, int64_t K);

int main() {
    printf("=== FP8 GEMM Double Buffering ===\n");
    printf("Timer: 100 MHz × 20 = cycles\n\n");
    init_lut();

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512, N = NR * VL;
    int64_t M_panels = M / MR;
    int iters = 100;

    printf("M=%ld, N=%ld, K=%ld, panels=%ld\n", M, N, K, M_panels);
    printf("Ideal ticks/GEMM: 14750, 90%% target: 16389\n\n");

    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* B_fp8 = aligned_alloc(ALIGN, K * N);
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
    // Double buffer for A
    float* A_fp32_buf0 = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* A_fp32_buf1 = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) A_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    for (int64_t i = 0; i < K * N; i++) B_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);

    convert_B_fp8_to_fp32(B_fp8, B_fp32, K, N);

    uint64_t t0, t1;

    // Baseline: sequential convert + GEMM
    printf("1. Sequential (convert then GEMM):\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_panel(A_fp8, A_fp32_buf0, p, K, K, MR);
            fp8_gemm_kernel_asm(A_fp32_buf0, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    printf("   %lu ticks (%lu per GEMM)\n", t1-t0, (t1-t0)/iters);

    // Double buffering: convert next while computing current
    printf("2. Double buffering (software pipeline):\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        float* buf_curr = A_fp32_buf0;
        float* buf_next = A_fp32_buf1;

        // Prime the pipeline: convert first panel
        convert_A_panel(A_fp8, buf_curr, 0, K, K, MR);

        for (int64_t p = 0; p < M_panels - 1; p++) {
            // Start converting next panel (could be parallelized)
            convert_A_panel(A_fp8, buf_next, p + 1, K, K, MR);
            // Compute current panel
            fp8_gemm_kernel_asm(buf_curr, B_fp32, C + p * MR * N, N, K);
            // Swap buffers
            float* tmp = buf_curr; buf_curr = buf_next; buf_next = tmp;
        }
        // Process last panel
        fp8_gemm_kernel_asm(buf_curr, B_fp32, C + (M_panels-1) * MR * N, N, K);
    }
    t1 = get_ticks();
    printf("   %lu ticks (%lu per GEMM)\n", t1-t0, (t1-t0)/iters);

    // OpenMP parallel: one thread converts, one computes
    printf("3. OpenMP parallel (2 threads):\n");
    memset(C, 0, M * N * sizeof(float));
    omp_set_num_threads(2);

    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        // Convert first panel
        convert_A_panel(A_fp8, A_fp32_buf0, 0, K, K, MR);

        for (int64_t p = 0; p < M_panels - 1; p++) {
            float* buf_curr = (p % 2 == 0) ? A_fp32_buf0 : A_fp32_buf1;
            float* buf_next = (p % 2 == 0) ? A_fp32_buf1 : A_fp32_buf0;

            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    // Thread 0: convert next
                    convert_A_panel(A_fp8, buf_next, p + 1, K, K, MR);
                }
                #pragma omp section
                {
                    // Thread 1: compute current
                    fp8_gemm_kernel_asm(buf_curr, B_fp32, C + p * MR * N, N, K);
                }
            }
        }
        // Last panel
        float* buf_last = ((M_panels-1) % 2 == 0) ? A_fp32_buf0 : A_fp32_buf1;
        fp8_gemm_kernel_asm(buf_last, B_fp32, C + (M_panels-1) * MR * N, N, K);
    }
    t1 = get_ticks();
    printf("   %lu ticks (%lu per GEMM)\n\n", t1-t0, (t1-t0)/iters);

    // Analysis
    printf("=== Analysis ===\n");
    printf("Conversion: ~711 ticks/panel, Kernel: ~362 ticks/panel\n");
    printf("Theoretical overlap gain: max(711,362) = 711 vs 711+362 = 1073\n");
    printf("Expected speedup with perfect overlap: 1073/711 = 1.5x\n");

    free(A_fp8); free(B_fp8); free(B_fp32);
    free(A_fp32_buf0); free(A_fp32_buf1); free(C);
    return 0;
}
