/*
 * Benchmark A conversion - Optimized approaches
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

// Original scalar A conversion with packing
void convert_A_scalar(const fp8_e4m3_t* A, float* A_buf,
                      int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A[(panel*MR+m)*lda + k]];
            A_buf[k * MR + m] = *((float*)&bits);
        }
    }
}

// Optimized scalar: loop interchange (m outer, k inner for better cache)
void convert_A_scalar_opt(const fp8_e4m3_t* A, float* A_buf,
                          int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    int64_t base = panel * MR * lda;
    for (int64_t m = 0; m < MR; m++) {
        const fp8_e4m3_t* row = A + base + m * lda;
        for (int64_t k = 0; k < K; k++) {
            uint32_t bits = fp8_to_fp32_lut[row[k]];
            A_buf[k * MR + m] = *((float*)&bits);
        }
    }
}

// Optimized with unrolled k
void convert_A_scalar_u4(const fp8_e4m3_t* A, float* A_buf,
                         int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    int64_t base = panel * MR * lda;
    for (int64_t m = 0; m < MR; m++) {
        const fp8_e4m3_t* row = A + base + m * lda;
        int64_t k = 0;
        for (; k + 3 < K; k += 4) {
            uint32_t b0 = fp8_to_fp32_lut[row[k]];
            uint32_t b1 = fp8_to_fp32_lut[row[k+1]];
            uint32_t b2 = fp8_to_fp32_lut[row[k+2]];
            uint32_t b3 = fp8_to_fp32_lut[row[k+3]];
            A_buf[k * MR + m] = *((float*)&b0);
            A_buf[(k+1) * MR + m] = *((float*)&b1);
            A_buf[(k+2) * MR + m] = *((float*)&b2);
            A_buf[(k+3) * MR + m] = *((float*)&b3);
        }
        for (; k < K; k++) {
            uint32_t bits = fp8_to_fp32_lut[row[k]];
            A_buf[k * MR + m] = *((float*)&bits);
        }
    }
}

// SVE: Convert row by row, then scatter to packed format
void convert_A_sve_row(const fp8_e4m3_t* A, float* A_buf,
                       int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    int64_t base = panel * MR * lda;
    int64_t vl = svcntw();

    for (int64_t m = 0; m < MR; m++) {
        const fp8_e4m3_t* row = A + base + m * lda;
        for (int64_t k = 0; k < K; k += vl) {
            svbool_t pg = svwhilelt_b32(k, K);
            svuint32_t idx = svld1ub_u32(pg, row + k);
            svuint32_t fp32 = svld1_gather_index(pg, fp8_to_fp32_lut, idx);

            // Scatter to packed format [K][MR]
            uint32_t temp[16];
            svst1_u32(pg, temp, fp32);
            int64_t kmax = (k + vl < K) ? vl : (K - k);
            for (int64_t i = 0; i < kmax; i++) {
                A_buf[(k + i) * MR + m] = *((float*)&temp[i]);
            }
        }
    }
}

// SVE block (8 rows × 16 k) - simplified
void convert_A_sve_block(const fp8_e4m3_t* A, float* A_buf,
                         int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    int64_t base = panel * MR * lda;
    int64_t vl = svcntw();  // 16

    // Process K in chunks of 16
    for (int64_t k = 0; k < K; k += vl) {
        svbool_t pg = svwhilelt_b32(k, K);
        int64_t kmax = (k + vl < K) ? vl : (K - k);

        // Load and convert 8 rows, store to temp
        float temp[8][16] __attribute__((aligned(64)));
        for (int64_t m = 0; m < MR; m++) {
            const fp8_e4m3_t* row = A + base + m * lda + k;
            svuint32_t idx = svld1ub_u32(pg, row);
            svuint32_t fp32 = svld1_gather_index(pg, fp8_to_fp32_lut, idx);
            svst1_f32(pg, temp[m], svreinterpret_f32(fp32));
        }

        // Transpose and store
        for (int64_t i = 0; i < kmax; i++) {
            for (int64_t m = 0; m < MR; m++) {
                A_buf[(k + i) * MR + m] = temp[m][i];
            }
        }
    }
}

int main() {
    printf("=== A Conversion Optimization v3 ===\n\n");
    init_lut();

    uint64_t freq = get_freq();
    double ticks_to_cycles = (double)CPU_FREQ_MHZ * 1e6 / freq;

    const int64_t MR = 8;
    int64_t M = 384, K = 512;
    int64_t lda = K;
    int64_t M_panels = M / MR;
    int64_t total_elements = M * K;

    printf("M=%ld, K=%ld, MR=%ld, Panels=%ld\n", M, K, MR, M_panels);
    printf("Total elements: %ld\n\n", total_elements);

    fp8_e4m3_t* A = aligned_alloc(ALIGN, M * K);
    float* A_buf = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* ref = aligned_alloc(ALIGN, MR * K * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }

    int iters = 100;
    volatile uint64_t start, end;
    double cycles;

    // Reference
    convert_A_scalar(A, ref, 0, K, lda, MR);

    // Test 1: Original scalar
    printf("1. Scalar (original):\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_scalar(A, A_buf, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);

    // Test 2: Optimized scalar (loop interchange)
    printf("2. Scalar optimized (m outer):\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_scalar_opt(A, A_buf, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_scalar_opt(A, A_buf, 0, K, lda, MR);
    int errors = 0;
    for (int64_t i = 0; i < MR * K; i++) if (A_buf[i] != ref[i]) errors++;
    printf("   Errors: %d\n", errors);

    // Test 3: Scalar unrolled
    printf("3. Scalar unrolled x4:\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_scalar_u4(A, A_buf, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_scalar_u4(A, A_buf, 0, K, lda, MR);
    errors = 0;
    for (int64_t i = 0; i < MR * K; i++) if (A_buf[i] != ref[i]) errors++;
    printf("   Errors: %d\n", errors);

    // Test 4: SVE row-by-row
    printf("4. SVE row-by-row:\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_sve_row(A, A_buf, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_sve_row(A, A_buf, 0, K, lda, MR);
    errors = 0;
    for (int64_t i = 0; i < MR * K; i++) if (A_buf[i] != ref[i]) errors++;
    printf("   Errors: %d\n", errors);

    // Test 5: SVE block (8 rows × 16 k)
    printf("5. SVE block 8x16:\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_sve_block(A, A_buf, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_sve_block(A, A_buf, 0, K, lda, MR);
    errors = 0;
    for (int64_t i = 0; i < MR * K; i++) if (A_buf[i] != ref[i]) errors++;
    printf("   Errors: %d\n", errors);

    printf("\n=== Target: minimize total cycles/elem ===\n");
    printf("B sequential: 0.72 cycles/elem (baseline)\n");

    free(A); free(A_buf); free(ref);
    return 0;
}
