/*
 * Benchmark A conversion - Approach: Pre-transpose A, then convert like B
 *
 * Key insight: B conversion is fast (0.72 cycles/elem) because it's sequential.
 * A is slow because we need strided access for packing [M][K] -> [K][MR].
 *
 * New approach: Pre-pack A to [K][M] layout (FP8), then convert sequentially.
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

// Fast sequential SVE conversion (like B matrix)
void convert_sve_sequential(const fp8_e4m3_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();
    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t indices = svld1ub_u32(pg, src + i);
        svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
        svst1(pg, dst + i, svreinterpret_f32(fp32_bits));
    }
}

// Transpose A from [M][K] to [K][M] (FP8 level, before conversion)
// This is done once for the entire matrix
void transpose_fp8(const fp8_e4m3_t* src, fp8_e4m3_t* dst, int64_t M, int64_t K) {
    for (int64_t m = 0; m < M; m++) {
        for (int64_t k = 0; k < K; k++) {
            dst[k * M + m] = src[m * K + k];
        }
    }
}

// Transpose with blocking for cache efficiency
void transpose_fp8_blocked(const fp8_e4m3_t* src, fp8_e4m3_t* dst, int64_t M, int64_t K) {
    const int64_t BLOCK = 64;
    for (int64_t mb = 0; mb < M; mb += BLOCK) {
        int64_t m_end = (mb + BLOCK < M) ? mb + BLOCK : M;
        for (int64_t kb = 0; kb < K; kb += BLOCK) {
            int64_t k_end = (kb + BLOCK < K) ? kb + BLOCK : K;
            for (int64_t m = mb; m < m_end; m++) {
                for (int64_t k = kb; k < k_end; k++) {
                    dst[k * M + m] = src[m * K + k];
                }
            }
        }
    }
}

// After pre-transpose: A_T is [K][M], extract panel and convert
void convert_A_from_transposed(const fp8_e4m3_t* A_T, float* A_buf,
                                int64_t panel, int64_t K, int64_t M, int64_t MR) {
    // A_T[k][panel*MR .. (panel+1)*MR-1] -> A_buf[k][0..MR-1]
    // For each K, copy MR consecutive bytes and convert
    int64_t vl = svcntw();

    for (int64_t k = 0; k < K; k++) {
        // Source: A_T[k * M + panel*MR] for MR elements
        // These are now consecutive!
        const fp8_e4m3_t* src = A_T + k * M + panel * MR;
        float* dst = A_buf + k * MR;

        // Since MR=8 < VL=16, use predicate
        svbool_t pg = svwhilelt_b32((uint64_t)0, (uint64_t)MR);
        svuint32_t indices = svld1ub_u32(pg, src);
        svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
        svst1_f32(pg, dst, svreinterpret_f32(fp32_bits));
    }
}

// After pre-transpose: Bulk convert entire A_T to FP32, then extract panels
void convert_A_bulk(const fp8_e4m3_t* A_T, float* A_buf_full, int64_t M, int64_t K) {
    // Convert all of A_T [K][M] to FP32
    convert_sve_sequential(A_T, A_buf_full, M * K);
}

int main() {
    printf("=== A Conversion Optimization v2 ===\n\n");
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

    fp8_e4m3_t* A = aligned_alloc(ALIGN, M * K);          // Original [M][K]
    fp8_e4m3_t* A_T = aligned_alloc(ALIGN, M * K);        // Transposed [K][M]
    float* A_buf = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* A_buf_full = aligned_alloc(ALIGN, M * K * sizeof(float));  // All panels
    float* ref = aligned_alloc(ALIGN, MR * K * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }

    int iters = 100;
    volatile uint64_t start, end;
    double cycles;

    // Generate reference
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
    printf("   Total: %.0f cycles, %.2f cycles/elem\n\n", cycles, cycles / total_elements);

    // Test 2: Pre-transpose FP8, then convert
    printf("2. Pre-transpose FP8, then convert:\n");

    // Measure transpose time
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        transpose_fp8(A, A_T, M, K);
    }
    end = get_ticks();
    double transpose_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Transpose: %.0f cycles (%.2f cycles/elem)\n", transpose_cycles, transpose_cycles / total_elements);

    // Pre-transpose once
    transpose_fp8(A, A_T, M, K);

    // Measure conversion from transposed layout
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_from_transposed(A_T, A_buf, p, K, M, MR);
        }
    }
    end = get_ticks();
    double convert_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Convert: %.0f cycles (%.2f cycles/elem)\n", convert_cycles, convert_cycles / total_elements);
    printf("   Total: %.0f cycles (%.2f cycles/elem)\n", transpose_cycles + convert_cycles,
           (transpose_cycles + convert_cycles) / total_elements);

    // Verify
    convert_A_from_transposed(A_T, A_buf, 0, K, M, MR);
    int errors = 0;
    for (int64_t i = 0; i < MR * K && errors < 10; i++) {
        if (A_buf[i] != ref[i]) errors++;
    }
    printf("   Errors: %d\n\n", errors);

    // Test 3: Blocked transpose
    printf("3. Blocked transpose, then convert:\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        transpose_fp8_blocked(A, A_T, M, K);
    }
    end = get_ticks();
    transpose_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Blocked transpose: %.0f cycles (%.2f cycles/elem)\n", transpose_cycles, transpose_cycles / total_elements);

    transpose_fp8_blocked(A, A_T, M, K);

    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_from_transposed(A_T, A_buf, p, K, M, MR);
        }
    }
    end = get_ticks();
    convert_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Convert: %.0f cycles (%.2f cycles/elem)\n", convert_cycles, convert_cycles / total_elements);
    printf("   Total: %.0f cycles (%.2f cycles/elem)\n\n", transpose_cycles + convert_cycles,
           (transpose_cycles + convert_cycles) / total_elements);

    // Test 4: Bulk convert entire transposed matrix, then extract panels
    printf("4. Bulk convert transposed matrix:\n");

    transpose_fp8_blocked(A, A_T, M, K);

    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        // Convert all of A_T to FP32 (sequential)
        convert_A_bulk(A_T, A_buf_full, M, K);
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Bulk convert: %.0f cycles (%.2f cycles/elem)\n", cycles, cycles / total_elements);
    printf("   + transpose: %.0f cycles (%.2f cycles/elem total)\n\n",
           transpose_cycles + cycles, (transpose_cycles + cycles) / total_elements);

    // Test 5: For comparison - if we store A pre-transposed
    printf("5. Assuming A already transposed [K][M]:\n");
    // Simulate by just converting panels from pre-transposed A_T
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_from_transposed(A_T, A_buf, p, K, M, MR);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("   Panel convert: %.0f cycles (%.2f cycles/elem)\n\n", cycles, cycles / total_elements);

    // Comparison summary
    printf("=== Summary ===\n");
    printf("B-style (sequential): 0.72 cycles/elem (reference)\n");
    printf("Original scalar with packing: ~1.87 cycles/elem\n");
    printf("Target: Minimize total (transpose + convert)\n");

    free(A); free(A_T); free(A_buf); free(A_buf_full); free(ref);
    return 0;
}
