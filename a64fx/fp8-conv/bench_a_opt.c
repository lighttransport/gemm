/*
 * Benchmark optimized A conversion using SVE gather
 * Goal: Reduce A conversion from 3.7 cycles/elem to ~0.8 like B conversion
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

// SVE: Convert B-style (sequential) - reference for what we want to achieve
void convert_B_sve(const fp8_e4m3_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();
    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t indices = svld1ub_u32(pg, src + i);
        svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
        svst1(pg, dst + i, svreinterpret_f32(fp32_bits));
    }
}

// Approach 1: Pre-pack A rows to contiguous buffer, then convert like B
void convert_A_prepack_then_sve(const fp8_e4m3_t* A, float* A_buf,
                                 fp8_e4m3_t* temp,
                                 int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    // Step 1: Copy 8 rows to contiguous temp buffer (still FP8)
    // temp layout: [MR][K] row-major
    for (int64_t m = 0; m < MR; m++) {
        memcpy(temp + m * K, A + (panel*MR+m)*lda, K);
    }

    // Step 2: Convert each row with SVE, then pack
    int64_t vl = svcntw();
    for (int64_t m = 0; m < MR; m++) {
        const fp8_e4m3_t* src = temp + m * K;
        for (int64_t k = 0; k < K; k += vl) {
            svbool_t pg = svwhilelt_b32(k, K);
            svuint32_t indices = svld1ub_u32(pg, src + k);
            svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
            // Store to a temp location first, then scatter to packed format
            float temp_out[16] __attribute__((aligned(64)));
            svst1_f32(pg, temp_out, svreinterpret_f32(fp32_bits));
            // Scatter to packed [K][MR] format
            int64_t kmax = (k + vl < K) ? vl : (K - k);
            for (int64_t kk = 0; kk < kmax; kk++) {
                A_buf[(k + kk) * MR + m] = temp_out[kk];
            }
        }
    }
}

// Approach 2: Convert to temp FP32 row-major, then transpose
void convert_A_then_transpose(const fp8_e4m3_t* A, float* A_buf,
                              float* temp,
                              int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    int64_t vl = svcntw();

    // Step 1: Convert 8 rows to temp buffer (row-major FP32)
    // temp layout: [MR][K]
    for (int64_t m = 0; m < MR; m++) {
        const fp8_e4m3_t* src = A + (panel*MR+m)*lda;
        float* dst = temp + m * K;
        for (int64_t k = 0; k < K; k += vl) {
            svbool_t pg = svwhilelt_b32(k, K);
            svuint32_t indices = svld1ub_u32(pg, src + k);
            svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
            svst1_f32(pg, dst + k, svreinterpret_f32(fp32_bits));
        }
    }

    // Step 2: Transpose [MR][K] -> [K][MR]
    // For MR=8, can process 8 elements at a time
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            A_buf[k * MR + m] = temp[m * K + k];
        }
    }
}

// Approach 3: Convert to temp FP32 row-major, then transpose with SVE
void convert_A_then_transpose_sve(const fp8_e4m3_t* A, float* A_buf,
                                  float* temp,
                                  int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    int64_t vl = svcntw();

    // Step 1: Convert 8 rows to temp buffer (row-major FP32)
    for (int64_t m = 0; m < MR; m++) {
        const fp8_e4m3_t* src = A + (panel*MR+m)*lda;
        float* dst = temp + m * K;
        for (int64_t k = 0; k < K; k += vl) {
            svbool_t pg = svwhilelt_b32(k, K);
            svuint32_t indices = svld1ub_u32(pg, src + k);
            svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
            svst1_f32(pg, dst + k, svreinterpret_f32(fp32_bits));
        }
    }

    // Step 2: Transpose [MR][K] -> [K][MR] using SVE loads
    // Load 8 consecutive elements from each row (1 per row), store as column
    svbool_t pg8 = svwhilelt_b32((uint64_t)0, (uint64_t)MR);
    for (int64_t k = 0; k < K; k++) {
        // Load temp[m][k] for m=0..7
        float col[8];
        for (int64_t m = 0; m < MR; m++) {
            col[m] = temp[m * K + k];
        }
        svfloat32_t vcol = svld1_f32(pg8, col);
        svst1_f32(pg8, A_buf + k * MR, vcol);
    }
}

// Approach 4: Use scalar gather indices, SVE LUT gather
// Manually load 8 strided FP8 values, then SVE gather from LUT
void convert_A_hybrid(const fp8_e4m3_t* A, float* A_buf,
                      int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    svbool_t pg8 = svwhilelt_b32((uint64_t)0, (uint64_t)MR);

    for (int64_t k = 0; k < K; k++) {
        // Load 8 FP8 values with strided access (scalar loads)
        uint32_t indices[8];
        for (int64_t m = 0; m < MR; m++) {
            indices[m] = A[(panel*MR+m)*lda + k];
        }
        // SVE load of indices
        svuint32_t vindices = svld1_u32(pg8, indices);

        // Gather from LUT
        svuint32_t fp32_bits = svld1_gather_index(pg8, fp8_to_fp32_lut, vindices);

        // Store packed
        svst1_f32(pg8, A_buf + k * MR, svreinterpret_f32(fp32_bits));
    }
}

// Approach 5: Unroll K by 4 for better instruction-level parallelism
void convert_A_hybrid_u4(const fp8_e4m3_t* A, float* A_buf,
                         int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    svbool_t pg8 = svwhilelt_b32((uint64_t)0, (uint64_t)MR);
    int64_t base = panel * MR * lda;

    int64_t k = 0;
    for (; k + 3 < K; k += 4) {
        uint32_t idx0[8], idx1[8], idx2[8], idx3[8];
        for (int64_t m = 0; m < MR; m++) {
            int64_t row_base = m * lda;
            idx0[m] = A[base + row_base + k];
            idx1[m] = A[base + row_base + k + 1];
            idx2[m] = A[base + row_base + k + 2];
            idx3[m] = A[base + row_base + k + 3];
        }

        svuint32_t v0 = svld1_u32(pg8, idx0);
        svuint32_t v1 = svld1_u32(pg8, idx1);
        svuint32_t v2 = svld1_u32(pg8, idx2);
        svuint32_t v3 = svld1_u32(pg8, idx3);

        svuint32_t g0 = svld1_gather_index(pg8, fp8_to_fp32_lut, v0);
        svuint32_t g1 = svld1_gather_index(pg8, fp8_to_fp32_lut, v1);
        svuint32_t g2 = svld1_gather_index(pg8, fp8_to_fp32_lut, v2);
        svuint32_t g3 = svld1_gather_index(pg8, fp8_to_fp32_lut, v3);

        svst1_f32(pg8, A_buf + k * MR, svreinterpret_f32(g0));
        svst1_f32(pg8, A_buf + (k+1) * MR, svreinterpret_f32(g1));
        svst1_f32(pg8, A_buf + (k+2) * MR, svreinterpret_f32(g2));
        svst1_f32(pg8, A_buf + (k+3) * MR, svreinterpret_f32(g3));
    }

    // Remainder
    for (; k < K; k++) {
        uint32_t indices[8];
        for (int64_t m = 0; m < MR; m++) {
            indices[m] = A[base + m*lda + k];
        }
        svuint32_t vindices = svld1_u32(pg8, indices);
        svuint32_t fp32_bits = svld1_gather_index(pg8, fp8_to_fp32_lut, vindices);
        svst1_f32(pg8, A_buf + k * MR, svreinterpret_f32(fp32_bits));
    }
}

int main() {
    printf("=== A Conversion Optimization Benchmark ===\n\n");
    init_lut();

    uint64_t freq = get_freq();
    double ticks_to_cycles = (double)CPU_FREQ_MHZ * 1e6 / freq;

    const int64_t MR = 8;
    int64_t M = 384, K = 512;
    int64_t lda = K;  // A is [M][K]
    int64_t M_panels = M / MR;
    int64_t total_elements = M * K;

    printf("M=%ld, K=%ld, MR=%ld, Panels=%ld\n", M, K, MR, M_panels);
    printf("Total elements: %ld\n\n", total_elements);

    fp8_e4m3_t* A = aligned_alloc(ALIGN, M * K);
    float* A_buf = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* temp = aligned_alloc(ALIGN, MR * K * sizeof(float));
    fp8_e4m3_t* temp_fp8 = aligned_alloc(ALIGN, MR * K);
    float* ref = aligned_alloc(ALIGN, MR * K * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }

    int iters = 100;
    volatile uint64_t start, end;
    double cycles;

    // Generate reference with scalar
    convert_A_scalar(A, ref, 0, K, lda, MR);

    // Test 0: B-style SVE (for reference - sequential access)
    printf("0. B-style SVE (sequential, reference):\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        // Simulate sequential access (no packing needed for B)
        for (int64_t p = 0; p < M_panels; p++) {
            convert_B_sve(A + p*MR*K, temp, MR * K);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Total: %.0f cycles, %.2f cycles/elem\n\n", cycles, cycles / total_elements);

    // Test 1: Scalar (original)
    printf("1. Scalar (original):\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_scalar(A, A_buf, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Total: %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_scalar(A, A_buf, 0, K, lda, MR);
    int errors = 0;
    for (int64_t i = 0; i < MR * K && errors < 10; i++) {
        if (A_buf[i] != ref[i]) errors++;
    }
    printf("   Errors: %d\n\n", errors);

    // Test 2: Pre-pack then SVE
    printf("2. Pre-pack then SVE:\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_prepack_then_sve(A, A_buf, temp_fp8, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Total: %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_prepack_then_sve(A, A_buf, temp_fp8, 0, K, lda, MR);
    errors = 0;
    for (int64_t i = 0; i < MR * K && errors < 10; i++) {
        if (A_buf[i] != ref[i]) errors++;
    }
    printf("   Errors: %d\n\n", errors);

    // Test 3: Convert then transpose
    printf("3. Convert then transpose (scalar):\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_then_transpose(A, A_buf, temp, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Total: %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_then_transpose(A, A_buf, temp, 0, K, lda, MR);
    errors = 0;
    for (int64_t i = 0; i < MR * K && errors < 10; i++) {
        if (A_buf[i] != ref[i]) errors++;
    }
    printf("   Errors: %d\n\n", errors);

    // Test 4: Convert then transpose with SVE
    printf("4. Convert then transpose (SVE):\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_then_transpose_sve(A, A_buf, temp, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Total: %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_then_transpose_sve(A, A_buf, temp, 0, K, lda, MR);
    errors = 0;
    for (int64_t i = 0; i < MR * K && errors < 10; i++) {
        if (A_buf[i] != ref[i]) errors++;
    }
    printf("   Errors: %d\n\n", errors);

    // Test 5: Hybrid (scalar strided load + SVE LUT gather)
    printf("5. Hybrid (scalar load + SVE gather):\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_hybrid(A, A_buf, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Total: %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_hybrid(A, A_buf, 0, K, lda, MR);
    errors = 0;
    for (int64_t i = 0; i < MR * K && errors < 10; i++) {
        if (A_buf[i] != ref[i]) errors++;
    }
    printf("   Errors: %d\n\n", errors);

    // Test 6: Hybrid unrolled by 4
    printf("6. Hybrid unrolled x4:\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_hybrid_u4(A, A_buf, p, K, lda, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Total: %.0f cycles, %.2f cycles/elem\n", cycles, cycles / total_elements);
    convert_A_hybrid_u4(A, A_buf, 0, K, lda, MR);
    errors = 0;
    for (int64_t i = 0; i < MR * K && errors < 10; i++) {
        if (A_buf[i] != ref[i]) errors++;
    }
    printf("   Errors: %d\n\n", errors);

    printf("=== Target: ~0.8 cycles/elem (like B conversion) ===\n");
    printf("=== Current best scalar: ~3.7 cycles/elem ===\n");

    free(A); free(A_buf); free(temp); free(temp_fp8); free(ref);
    return 0;
}
