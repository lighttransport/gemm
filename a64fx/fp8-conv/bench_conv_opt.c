/*
 * Optimizing FP8->FP32 conversion for A matrix
 * Target: reduce conversion overhead to achieve 90% overall
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

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

// Original scalar conversion (k-outer, m-inner)
void convert_scalar_orig(const fp8_e4m3_t* A, float* A_fp32,
                         int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t M_panels = M / MR;
    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < MR; m++) {
                uint32_t bits = fp8_to_fp32_lut[A[(p*MR+m)*lda + k]];
                dst[k * MR + m] = *((float*)&bits);
            }
        }
    }
}

// Scalar with unrolled inner loop
void convert_scalar_unroll(const fp8_e4m3_t* A, float* A_fp32,
                           int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t M_panels = M / MR;
    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;
        const fp8_e4m3_t* src0 = A + (p*MR+0)*lda;
        const fp8_e4m3_t* src1 = A + (p*MR+1)*lda;
        const fp8_e4m3_t* src2 = A + (p*MR+2)*lda;
        const fp8_e4m3_t* src3 = A + (p*MR+3)*lda;
        const fp8_e4m3_t* src4 = A + (p*MR+4)*lda;
        const fp8_e4m3_t* src5 = A + (p*MR+5)*lda;
        const fp8_e4m3_t* src6 = A + (p*MR+6)*lda;
        const fp8_e4m3_t* src7 = A + (p*MR+7)*lda;

        for (int64_t k = 0; k < K; k++) {
            float* d = dst + k * 8;
            uint32_t b0 = fp8_to_fp32_lut[src0[k]];
            uint32_t b1 = fp8_to_fp32_lut[src1[k]];
            uint32_t b2 = fp8_to_fp32_lut[src2[k]];
            uint32_t b3 = fp8_to_fp32_lut[src3[k]];
            uint32_t b4 = fp8_to_fp32_lut[src4[k]];
            uint32_t b5 = fp8_to_fp32_lut[src5[k]];
            uint32_t b6 = fp8_to_fp32_lut[src6[k]];
            uint32_t b7 = fp8_to_fp32_lut[src7[k]];
            d[0] = *((float*)&b0);
            d[1] = *((float*)&b1);
            d[2] = *((float*)&b2);
            d[3] = *((float*)&b3);
            d[4] = *((float*)&b4);
            d[5] = *((float*)&b5);
            d[6] = *((float*)&b6);
            d[7] = *((float*)&b7);
        }
    }
}

// SVE gather-based conversion (read row-by-row from packed A)
void convert_sve_gather(const fp8_e4m3_t* A, float* A_fp32,
                        int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t vl = svcntw();  // 16 for A64FX
    int64_t M_panels = M / MR;

    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;

        // Process K in chunks of VL
        for (int64_t k = 0; k < K; k += vl) {
            svbool_t pg = svwhilelt_b32(k, K);

            // For each of the 8 rows in the panel
            for (int64_t m = 0; m < MR; m++) {
                const fp8_e4m3_t* src = A + (p*MR+m)*lda + k;
                // Load FP8 bytes, zero-extend to 32-bit
                svuint32_t indices = svld1ub_u32(pg, src);
                // Gather FP32 values from LUT
                svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
                // Store as float
                svst1(pg, dst + m, svreinterpret_f32(fp32_bits));
            }
            dst += MR * vl;
        }
    }
}

// SVE gather with k-unrolling
void convert_sve_gather_unroll(const fp8_e4m3_t* A, float* A_fp32,
                               int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t vl = svcntw();  // 16
    int64_t M_panels = M / MR;
    svbool_t pg = svptrue_b32();

    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;

        // Prepare row pointers
        const fp8_e4m3_t* rows[8];
        for (int m = 0; m < 8; m++) {
            rows[m] = A + (p*MR+m)*lda;
        }

        // Process K in chunks of 16
        for (int64_t k = 0; k < K; k += 16) {
            // Process all 8 rows for this K chunk
            for (int m = 0; m < 8; m++) {
                svuint32_t idx = svld1ub_u32(pg, rows[m] + k);
                svuint32_t fp32 = svld1_gather_index(pg, fp8_to_fp32_lut, idx);
                // Store interleaved: dst[k*8 + m], dst[(k+1)*8 + m], ...
                // Actually we need contiguous output for kernel
                // Output layout: dst[k*MR + m] for each k
                // But SVE loads 16 k values, so we need scatter store
                // This is getting complex - let's try simpler approach
            }
            // For now, just do row-by-row
            for (int m = 0; m < 8; m++) {
                svuint32_t idx = svld1ub_u32(pg, rows[m] + k);
                svuint32_t fp32 = svld1_gather_index(pg, fp8_to_fp32_lut, idx);
                // Store to temp, then transpose
                float temp[16];
                svst1(pg, (float*)temp, svreinterpret_f32(fp32));
                for (int i = 0; i < 16 && (k+i) < K; i++) {
                    dst[(k+i)*8 + m] = temp[i];
                }
            }
        }
    }
}

// Pre-transpose A to sequential layout, then SVE convert
void convert_pretranspose(const fp8_e4m3_t* A, float* A_fp32,
                          int64_t M, int64_t K, int64_t lda, int64_t MR,
                          fp8_e4m3_t* A_trans) {
    int64_t vl = svcntw();
    int64_t M_panels = M / MR;

    // First: transpose A to [panel][k][m] in FP8
    for (int64_t p = 0; p < M_panels; p++) {
        fp8_e4m3_t* trans = A_trans + p * K * MR;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < MR; m++) {
                trans[k * MR + m] = A[(p*MR+m)*lda + k];
            }
        }
    }

    // Second: SVE convert sequential data
    svbool_t pg = svptrue_b32();
    int64_t total = M * K;
    for (int64_t i = 0; i < total; i += vl) {
        svbool_t pg_i = svwhilelt_b32(i, total);
        svuint32_t idx = svld1ub_u32(pg_i, A_trans + i);
        svuint32_t fp32 = svld1_gather_index(pg_i, fp8_to_fp32_lut, idx);
        svst1(pg_i, A_fp32 + i, svreinterpret_f32(fp32));
    }
}

int main() {
    printf("=== FP8->FP32 Conversion Optimization ===\n\n");
    init_lut();

    int64_t M = 384, K = 512, MR = 8;
    int iters = 100;

    printf("Matrix: M=%ld, K=%ld, MR=%ld\n", M, K, MR);
    printf("Elements: %ld\n\n", M * K);

    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* A_trans = aligned_alloc(ALIGN, M * K);
    float* A_fp32 = aligned_alloc(ALIGN, M * K * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }

    uint64_t t0, t1;

    printf("Conversion Methods (%d iters):\n", iters);
    printf("────────────────────────────────────────\n");

    // 1. Original scalar
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_scalar_orig(A_fp8, A_fp32, M, K, K, MR);
    }
    t1 = get_ticks();
    printf("1. Scalar (orig):     %lu ticks (%lu/iter)\n", t1-t0, (t1-t0)/iters);

    // 2. Scalar unrolled
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_scalar_unroll(A_fp8, A_fp32, M, K, K, MR);
    }
    t1 = get_ticks();
    printf("2. Scalar (unroll):   %lu ticks (%lu/iter)\n", t1-t0, (t1-t0)/iters);

    // 3. SVE gather row-by-row
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_sve_gather(A_fp8, A_fp32, M, K, K, MR);
    }
    t1 = get_ticks();
    printf("3. SVE gather:        %lu ticks (%lu/iter)\n", t1-t0, (t1-t0)/iters);

    // 4. Pre-transpose + SVE
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_pretranspose(A_fp8, A_fp32, M, K, K, MR, A_trans);
    }
    t1 = get_ticks();
    printf("4. Pre-trans + SVE:   %lu ticks (%lu/iter)\n", t1-t0, (t1-t0)/iters);

    printf("\nTarget: < 700 ticks to reach 90%% with reuse=1\n");
    printf("(GEMM=15683 ticks, ideal=14745, 90%%=16383)\n");

    free(A_fp8); free(A_trans); free(A_fp32);
    return 0;
}
