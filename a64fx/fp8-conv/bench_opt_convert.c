/*
 * Optimized FP8 GEMM with vectorized A conversion
 * Target: Match B conversion efficiency (0.8 cycles/elem) for A
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
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

// Original scalar A conversion with packing
void convert_pack_A_scalar(const fp8_e4m3_t* A, float* A_packed,
                           int64_t M, int64_t K, int64_t MR) {
    int64_t M_panels = M / MR;
    for (int64_t p = 0; p < M_panels; p++) {
        for (int64_t k = 0; k < K; k++) {
            for (int m = 0; m < MR; m++) {
                uint32_t bits = fp8_to_fp32_lut[A[(p*MR+m)*K + k]];
                A_packed[p*K*MR + k*MR + m] = *((float*)&bits);
            }
        }
    }
}

// Vectorized A conversion - process multiple K values at once
// A is row-major [M][K], we want packed [M/MR][K][MR]
void convert_pack_A_sve(const fp8_e4m3_t* A, float* A_packed,
                        int64_t M, int64_t K, int64_t MR) {
    int64_t M_panels = M / MR;
    int64_t vl = svcntw();  // 16 for A64FX

    for (int64_t p = 0; p < M_panels; p++) {
        // For each row in this panel, convert K elements vectorized
        for (int64_t m = 0; m < MR; m++) {
            const fp8_e4m3_t* row_ptr = A + (p*MR + m)*K;

            for (int64_t k = 0; k < K; k += vl) {
                svbool_t pg = svwhilelt_b32(k, K);

                // Load bytes and zero-extend to 32-bit
                svuint32_t indices = svld1ub_u32(pg, row_ptr + k);

                // LUT lookup
                svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);

                // Scatter store to packed format
                // Target: A_packed[p*K*MR + (k:k+vl)*MR + m]
                // This is strided store with stride MR*4 bytes

                // Build store offsets: m*4, (MR+m)*4, (2*MR+m)*4, ...
                svuint32_t store_idx = svindex_u32(0, 1);  // 0,1,2,...,15
                svuint32_t store_offsets = svadd_x(pg,
                    svmul_x(pg, store_idx, (uint32_t)(MR * 4)),
                    (uint32_t)(m * 4));

                // Scatter store
                float* base = A_packed + p*K*MR + k*MR;
                svst1_scatter_offset(pg, base, store_offsets, svreinterpret_f32(fp32_bits));
            }
        }
    }
}

// Alternative: Process by K slices, load from multiple rows with gather
void convert_pack_A_sve_v2(const fp8_e4m3_t* A, float* A_packed,
                           int64_t M, int64_t K, int64_t MR) {
    int64_t M_panels = M / MR;

    // We'll process MR=8 elements at a time (one from each row of a panel)
    // This uses half the vector width but has better locality

    for (int64_t p = 0; p < M_panels; p++) {
        for (int64_t k = 0; k < K; k++) {
            // Load MR bytes from different rows using gather
            // A[(p*MR+0)*K + k], A[(p*MR+1)*K + k], ..., A[(p*MR+MR-1)*K + k]

            // Build load offsets: 0*K + k, 1*K + k, ..., (MR-1)*K + k (in bytes)
            // Offsets: p*MR*K + k, p*MR*K + K + k, p*MR*K + 2*K + k, ...

            svbool_t pg = svwhilelt_b32((int64_t)0, MR);
            svuint32_t row_idx = svindex_u32(0, 1);  // 0, 1, 2, ..., 7
            svuint32_t load_offsets = svmul_x(pg, row_idx, (uint32_t)K);
            // Offsets are now: 0, K, 2K, ..., 7K (byte offsets from panel start + k)

            const fp8_e4m3_t* base = A + p*MR*K + k;

            // Gather load bytes
            svuint32_t bytes = svld1ub_gather_offset_u32(pg, base, load_offsets);

            // LUT lookup
            svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, bytes);

            // Contiguous store (this is the packed format we want)
            svst1(pg, A_packed + p*K*MR + k*MR, svreinterpret_f32(fp32_bits));
        }
    }
}

// B conversion (contiguous, for comparison)
void convert_B_sve(const fp8_e4m3_t* B, float* B_fp32, int64_t K, int64_t N) {
    int64_t vl = svcntw();
    for (int64_t k = 0; k < K; k++) {
        for (int64_t n = 0; n < N; n += vl) {
            svbool_t pg = svwhilelt_b32(n, N);
            svuint32_t indices = svld1ub_u32(pg, B + k*N + n);
            svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
            svst1(pg, B_fp32 + k*N + n, svreinterpret_f32(fp32_bits));
        }
    }
}

extern void fp8_gemm_kernel_asm(const float* Ap, const float* Bp,
                                float* C, int64_t ldc, int64_t K);

int main() {
    printf("=== FP8 GEMM Optimized Conversion Benchmark ===\n\n");
    init_lut();

    uint64_t freq = get_freq();
    double ticks_to_cycles = (double)CPU_FREQ_MHZ * 1e6 / freq;
    printf("Timer freq: %lu Hz, conversion factor: %.2f\n\n", freq, ticks_to_cycles);

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512;
    int64_t N = NR * VL;  // 48
    int64_t M_panels = M / MR;
    double flops = 2.0 * M * N * K;

    printf("M=%ld, N=%ld, K=%ld, MR=%ld, M_panels=%ld\n", M, N, K, MR, M_panels);
    printf("FLOPs = %.2fM\n\n", flops/1e6);

    // Allocate
    fp8_e4m3_t* A = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* B = aligned_alloc(ALIGN, K * N);
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));
    float* A_packed = aligned_alloc(ALIGN, M * K * sizeof(float));  // Full packed A
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));

    // Initialize with valid FP8 values
    srand(42);
    for (int64_t i = 0; i < M * K; i++) A[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    for (int64_t i = 0; i < K * N; i++) B[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);

    int iters = 100;
    volatile uint64_t start, end;
    double cycles;

    // ========== Scalar A conversion ==========
    printf("=== A Conversion Comparison ===\n");

    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_pack_A_scalar(A, A_packed, M, K, MR);
    }
    end = get_ticks();
    double scalar_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("Scalar A: %.0f cycles (%.2f cycles/elem)\n", scalar_cycles, scalar_cycles/(M*K));

    // ========== SVE A conversion v1 (row-wise) ==========
    memset(A_packed, 0, M * K * sizeof(float));

    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_pack_A_sve(A, A_packed, M, K, MR);
    }
    end = get_ticks();
    double sve_v1_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("SVE v1 A:  %.0f cycles (%.2f cycles/elem)\n", sve_v1_cycles, sve_v1_cycles/(M*K));

    // Verify v1
    float* A_ref = aligned_alloc(ALIGN, M * K * sizeof(float));
    convert_pack_A_scalar(A, A_ref, M, K, MR);
    int v1_errs = 0;
    for (int64_t i = 0; i < M * K && v1_errs < 10; i++) {
        if (A_packed[i] != A_ref[i]) {
            if (v1_errs == 0) printf("  v1 mismatch at %ld: got %.6f, expected %.6f\n", i, A_packed[i], A_ref[i]);
            v1_errs++;
        }
    }
    if (v1_errs) printf("  v1 total errors: %d\n", v1_errs);
    else printf("  v1 verified OK\n");

    // ========== SVE A conversion v2 (column-wise gather) ==========
    memset(A_packed, 0, M * K * sizeof(float));

    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_pack_A_sve_v2(A, A_packed, M, K, MR);
    }
    end = get_ticks();
    double sve_v2_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("SVE v2 A:  %.0f cycles (%.2f cycles/elem)\n", sve_v2_cycles, sve_v2_cycles/(M*K));

    // Verify v2
    int v2_errs = 0;
    for (int64_t i = 0; i < M * K && v2_errs < 10; i++) {
        if (A_packed[i] != A_ref[i]) {
            if (v2_errs == 0) printf("  v2 mismatch at %ld: got %.6f, expected %.6f\n", i, A_packed[i], A_ref[i]);
            v2_errs++;
        }
    }
    if (v2_errs) printf("  v2 total errors: %d\n", v2_errs);
    else printf("  v2 verified OK\n");

    // ========== B conversion for reference ==========
    printf("\n=== B Conversion ===\n");
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_B_sve(B, B_fp32, K, N);
    }
    end = get_ticks();
    double b_cycles = (double)(end - start) / iters * ticks_to_cycles;
    printf("B convert: %.0f cycles (%.2f cycles/elem)\n", b_cycles, b_cycles/(K*N));

    // ========== Full GEMM with best A conversion ==========
    printf("\n=== Full GEMM (Pre-pack all A, convert B, then kernel) ===\n");

    // Use the faster A conversion
    double best_a_cycles = (sve_v1_cycles < sve_v2_cycles) ? sve_v1_cycles : sve_v2_cycles;
    const char* best_a_name = (sve_v1_cycles < sve_v2_cycles) ? "SVE v1" : "SVE v2";
    printf("Using %s for A (%.2f cycles/elem)\n", best_a_name, best_a_cycles/(M*K));

    // Pre-pack all A and convert B once
    if (sve_v1_cycles < sve_v2_cycles) {
        convert_pack_A_sve(A, A_packed, M, K, MR);
    } else {
        convert_pack_A_sve_v2(A, A_packed, M, K, MR);
    }
    convert_B_sve(B, B_fp32, K, N);

    // Kernel only
    memset(C, 0, M * N * sizeof(float));
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_packed + p*K*MR, B_fp32, C + p*MR*N, N, K);
        }
    }
    end = get_ticks();
    double kernel_cycles = (double)(end - start) / iters * ticks_to_cycles;
    double kernel_gflops = flops * CPU_FREQ_MHZ / (kernel_cycles * 1e3);
    printf("Kernel only: %.0f cycles, %.2f GFLOPS (%.1f%% of 128)\n",
           kernel_cycles, kernel_gflops, 100.0 * kernel_gflops / 128.0);

    // Full cost
    double total_cycles = best_a_cycles + b_cycles + kernel_cycles;
    double total_gflops = flops * CPU_FREQ_MHZ / (total_cycles * 1e3);

    printf("\n=== Summary ===\n");
    printf("A convert: %8.0f cycles (%5.1f%%)\n", best_a_cycles, 100*best_a_cycles/total_cycles);
    printf("B convert: %8.0f cycles (%5.1f%%)\n", b_cycles, 100*b_cycles/total_cycles);
    printf("Kernel:    %8.0f cycles (%5.1f%%)\n", kernel_cycles, 100*kernel_cycles/total_cycles);
    printf("Total:     %8.0f cycles\n", total_cycles);
    printf("Effective: %.2f GFLOPS (%.1f%% of 128 peak)\n", total_gflops, 100*total_gflops/128);

    // Speedup comparison
    printf("\n=== Speedup vs Scalar ===\n");
    printf("Scalar total: %.0f cycles\n", scalar_cycles + b_cycles + kernel_cycles);
    printf("Optimized:    %.0f cycles\n", total_cycles);
    printf("Speedup:      %.2fx\n", (scalar_cycles + b_cycles + kernel_cycles) / total_cycles);

    free(A); free(B); free(C); free(A_packed); free(A_ref); free(B_fp32);
    return 0;
}
