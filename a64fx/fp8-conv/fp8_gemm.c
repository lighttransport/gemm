/*
 * FP8 GEMM with FP32 Accumulation for A64FX (SVE1)
 *
 * High-performance approach:
 * 1. Pre-convert FP8 A -> FP32 (one-time cost)
 * 2. Pre-convert FP8 B -> FP32 (one-time cost, amortized over M tiles)
 * 3. Pure FP32 kernel: scalar A loads + vector B loads + FMLA
 *
 * This eliminates ALL gather operations from the inner loop.
 */

#include "fp8_gemm.h"
#include "fp8_convert.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// External LUTs from fp8_convert.c
extern uint32_t fp8_e4m3_to_fp16_lut32[256];
extern uint32_t fp8_e4m3_to_fp32_lut[256];

void fp8_gemm_init(void) {
    init_fp8_luts();
}

// Round up to multiple
static inline int64_t round_up(int64_t x, int64_t m) {
    return (x + m - 1) / m * m;
}

// Pack A: M×K row-major -> panels of MR×K
void pack_fp8_A(const fp8_e4m3_t* A, int64_t lda, fp8_e4m3_t* Ap,
                int64_t M, int64_t K) {
    int64_t M_pad = round_up(M, MR_FP8);
    for (int64_t ir = 0; ir < M_pad; ir += MR_FP8) {
        for (int64_t k = 0; k < K; k++) {
            for (int i = 0; i < MR_FP8; i++) {
                int64_t row = ir + i;
                fp8_e4m3_t v = (row < M) ? A[row * lda + k] : 0;
                *Ap++ = v;
            }
        }
    }
}

// Pack B: K×N row-major -> panels of K × (NR × VL_FP32)
// Each k has NR vectors of VL_FP32 FP8 elements
void pack_fp8_B(const fp8_e4m3_t* B, int64_t ldb, fp8_e4m3_t* Bp,
                int64_t K, int64_t N) {
    const int64_t panel_n = NR_FP8 * VL_FP32;  // 4 × 16 = 64 elements
    for (int64_t k = 0; k < K; k++) {
        for (int64_t j = 0; j < panel_n; j++) {
            fp8_e4m3_t v = (j < N) ? B[k * ldb + j] : 0;
            *Bp++ = v;
        }
    }
}

/*
 * Pure FP32 Micro-kernel: 4×6 tile using FMLA with lane index
 *
 * Key optimization: 4 A values fit exactly in one 128-bit segment
 * - Single 128-bit A load, replicated across all segments
 * - No need for multiple segment handling
 *
 * Per K iteration:
 * - 1 vector load for A (4 values, replicated)
 * - 6 vector loads for B (FP32)
 * - 24 FMLAs with lane indexing
 */
void fp8_gemm_kernel_fp32(const float* Ap_f32, const float* Bp_f32,
                          float* C, int64_t ldc, int64_t K) {
    // 24 FP32 accumulators: 4 rows × 6 vectors
    svfloat32_t acc00 = svdup_f32(0.0f), acc01 = svdup_f32(0.0f);
    svfloat32_t acc02 = svdup_f32(0.0f), acc03 = svdup_f32(0.0f);
    svfloat32_t acc04 = svdup_f32(0.0f), acc05 = svdup_f32(0.0f);
    svfloat32_t acc10 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f);
    svfloat32_t acc12 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f);
    svfloat32_t acc14 = svdup_f32(0.0f), acc15 = svdup_f32(0.0f);
    svfloat32_t acc20 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f);
    svfloat32_t acc22 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f);
    svfloat32_t acc24 = svdup_f32(0.0f), acc25 = svdup_f32(0.0f);
    svfloat32_t acc30 = svdup_f32(0.0f), acc31 = svdup_f32(0.0f);
    svfloat32_t acc32 = svdup_f32(0.0f), acc33 = svdup_f32(0.0f);
    svfloat32_t acc34 = svdup_f32(0.0f), acc35 = svdup_f32(0.0f);

    svbool_t pg = svptrue_b32();
    const int64_t A_stride = MR_FP8;            // 4 floats per k
    const int64_t B_stride = NR_FP8 * VL_FP32;  // 96 floats per k

    for (int64_t k = 0; k < K; k++) {
        const float* Ap_k = Ap_f32 + k * A_stride;
        const float* Bp_k = Bp_f32 + k * B_stride;

        // Load A (4 values = 128-bit) and replicate across all segments
        svfloat32_t a_raw = svld1_f32(svwhilelt_b32(0, 4), Ap_k);
        svfloat32_t a = svdupq_lane_f32(a_raw, 0);  // replicate segment 0

        // Load B (6 vectors)
        svfloat32_t b0 = svld1_f32(pg, Bp_k + 0*VL_FP32);
        svfloat32_t b1 = svld1_f32(pg, Bp_k + 1*VL_FP32);
        svfloat32_t b2 = svld1_f32(pg, Bp_k + 2*VL_FP32);
        svfloat32_t b3 = svld1_f32(pg, Bp_k + 3*VL_FP32);
        svfloat32_t b4 = svld1_f32(pg, Bp_k + 4*VL_FP32);
        svfloat32_t b5 = svld1_f32(pg, Bp_k + 5*VL_FP32);

        // 24 FMLAs with lane indexing: 4 rows × 6 vectors
        // Row 0: use a[0]
        acc00 = svmla_lane_f32(acc00, b0, a, 0);
        acc01 = svmla_lane_f32(acc01, b1, a, 0);
        acc02 = svmla_lane_f32(acc02, b2, a, 0);
        acc03 = svmla_lane_f32(acc03, b3, a, 0);
        acc04 = svmla_lane_f32(acc04, b4, a, 0);
        acc05 = svmla_lane_f32(acc05, b5, a, 0);

        // Row 1: use a[1]
        acc10 = svmla_lane_f32(acc10, b0, a, 1);
        acc11 = svmla_lane_f32(acc11, b1, a, 1);
        acc12 = svmla_lane_f32(acc12, b2, a, 1);
        acc13 = svmla_lane_f32(acc13, b3, a, 1);
        acc14 = svmla_lane_f32(acc14, b4, a, 1);
        acc15 = svmla_lane_f32(acc15, b5, a, 1);

        // Row 2: use a[2]
        acc20 = svmla_lane_f32(acc20, b0, a, 2);
        acc21 = svmla_lane_f32(acc21, b1, a, 2);
        acc22 = svmla_lane_f32(acc22, b2, a, 2);
        acc23 = svmla_lane_f32(acc23, b3, a, 2);
        acc24 = svmla_lane_f32(acc24, b4, a, 2);
        acc25 = svmla_lane_f32(acc25, b5, a, 2);

        // Row 3: use a[3]
        acc30 = svmla_lane_f32(acc30, b0, a, 3);
        acc31 = svmla_lane_f32(acc31, b1, a, 3);
        acc32 = svmla_lane_f32(acc32, b2, a, 3);
        acc33 = svmla_lane_f32(acc33, b3, a, 3);
        acc34 = svmla_lane_f32(acc34, b4, a, 3);
        acc35 = svmla_lane_f32(acc35, b5, a, 3);
    }

    // Store results (4 rows × 6 vectors)
    svst1_f32(pg, C + 0*ldc + 0*VL_FP32, acc00);
    svst1_f32(pg, C + 0*ldc + 1*VL_FP32, acc01);
    svst1_f32(pg, C + 0*ldc + 2*VL_FP32, acc02);
    svst1_f32(pg, C + 0*ldc + 3*VL_FP32, acc03);
    svst1_f32(pg, C + 0*ldc + 4*VL_FP32, acc04);
    svst1_f32(pg, C + 0*ldc + 5*VL_FP32, acc05);

    svst1_f32(pg, C + 1*ldc + 0*VL_FP32, acc10);
    svst1_f32(pg, C + 1*ldc + 1*VL_FP32, acc11);
    svst1_f32(pg, C + 1*ldc + 2*VL_FP32, acc12);
    svst1_f32(pg, C + 1*ldc + 3*VL_FP32, acc13);
    svst1_f32(pg, C + 1*ldc + 4*VL_FP32, acc14);
    svst1_f32(pg, C + 1*ldc + 5*VL_FP32, acc15);

    svst1_f32(pg, C + 2*ldc + 0*VL_FP32, acc20);
    svst1_f32(pg, C + 2*ldc + 1*VL_FP32, acc21);
    svst1_f32(pg, C + 2*ldc + 2*VL_FP32, acc22);
    svst1_f32(pg, C + 2*ldc + 3*VL_FP32, acc23);
    svst1_f32(pg, C + 2*ldc + 4*VL_FP32, acc24);
    svst1_f32(pg, C + 2*ldc + 5*VL_FP32, acc25);

    svst1_f32(pg, C + 3*ldc + 0*VL_FP32, acc30);
    svst1_f32(pg, C + 3*ldc + 1*VL_FP32, acc31);
    svst1_f32(pg, C + 3*ldc + 2*VL_FP32, acc32);
    svst1_f32(pg, C + 3*ldc + 3*VL_FP32, acc33);
    svst1_f32(pg, C + 3*ldc + 4*VL_FP32, acc34);
    svst1_f32(pg, C + 3*ldc + 5*VL_FP32, acc35);
}

// Original kernel for compatibility (uses gathers) - 4×6 tile
void fp8_gemm_kernel_3x4(const fp8_e4m3_t* Ap, const fp8_e4m3_t* Bp,
                          float* C, int64_t ldc, int64_t K) {
    svfloat32_t acc00 = svdup_f32(0.0f), acc01 = svdup_f32(0.0f);
    svfloat32_t acc02 = svdup_f32(0.0f), acc03 = svdup_f32(0.0f);
    svfloat32_t acc04 = svdup_f32(0.0f), acc05 = svdup_f32(0.0f);
    svfloat32_t acc10 = svdup_f32(0.0f), acc11 = svdup_f32(0.0f);
    svfloat32_t acc12 = svdup_f32(0.0f), acc13 = svdup_f32(0.0f);
    svfloat32_t acc14 = svdup_f32(0.0f), acc15 = svdup_f32(0.0f);
    svfloat32_t acc20 = svdup_f32(0.0f), acc21 = svdup_f32(0.0f);
    svfloat32_t acc22 = svdup_f32(0.0f), acc23 = svdup_f32(0.0f);
    svfloat32_t acc24 = svdup_f32(0.0f), acc25 = svdup_f32(0.0f);
    svfloat32_t acc30 = svdup_f32(0.0f), acc31 = svdup_f32(0.0f);
    svfloat32_t acc32 = svdup_f32(0.0f), acc33 = svdup_f32(0.0f);
    svfloat32_t acc34 = svdup_f32(0.0f), acc35 = svdup_f32(0.0f);

    svbool_t pg = svptrue_b32();
    const int64_t A_stride = MR_FP8;  // 4 bytes per k
    const int64_t B_stride = NR_FP8 * VL_FP32;  // 96 bytes per k

    for (int64_t k = 0; k < K; k++) {
        const fp8_e4m3_t* Bp_k = Bp + k * B_stride;
        const fp8_e4m3_t* Ap_k = Ap + k * A_stride;

        // Load B with gather (6 vectors × 16 elements)
        svuint8_t b0_bytes = svld1_u8(svwhilelt_b8(0, 16), Bp_k + 0);
        svuint8_t b1_bytes = svld1_u8(svwhilelt_b8(0, 16), Bp_k + 16);
        svuint8_t b2_bytes = svld1_u8(svwhilelt_b8(0, 16), Bp_k + 32);
        svuint8_t b3_bytes = svld1_u8(svwhilelt_b8(0, 16), Bp_k + 48);
        svuint8_t b4_bytes = svld1_u8(svwhilelt_b8(0, 16), Bp_k + 64);
        svuint8_t b5_bytes = svld1_u8(svwhilelt_b8(0, 16), Bp_k + 80);

        svuint32_t b0_off = svlsl_n_u32_x(pg, svunpklo_u32(svunpklo_u16(b0_bytes)), 2);
        svuint32_t b1_off = svlsl_n_u32_x(pg, svunpklo_u32(svunpklo_u16(b1_bytes)), 2);
        svuint32_t b2_off = svlsl_n_u32_x(pg, svunpklo_u32(svunpklo_u16(b2_bytes)), 2);
        svuint32_t b3_off = svlsl_n_u32_x(pg, svunpklo_u32(svunpklo_u16(b3_bytes)), 2);
        svuint32_t b4_off = svlsl_n_u32_x(pg, svunpklo_u32(svunpklo_u16(b4_bytes)), 2);
        svuint32_t b5_off = svlsl_n_u32_x(pg, svunpklo_u32(svunpklo_u16(b5_bytes)), 2);

        svfloat32_t b0 = svreinterpret_f32(svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp32_lut, b0_off));
        svfloat32_t b1 = svreinterpret_f32(svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp32_lut, b1_off));
        svfloat32_t b2 = svreinterpret_f32(svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp32_lut, b2_off));
        svfloat32_t b3 = svreinterpret_f32(svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp32_lut, b3_off));
        svfloat32_t b4 = svreinterpret_f32(svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp32_lut, b4_off));
        svfloat32_t b5 = svreinterpret_f32(svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp32_lut, b5_off));

        // Load A (4 scalars) via LUT
        svfloat32_t a0 = svdup_f32(*((float*)&fp8_e4m3_to_fp32_lut[Ap_k[0]]));
        svfloat32_t a1 = svdup_f32(*((float*)&fp8_e4m3_to_fp32_lut[Ap_k[1]]));
        svfloat32_t a2 = svdup_f32(*((float*)&fp8_e4m3_to_fp32_lut[Ap_k[2]]));
        svfloat32_t a3 = svdup_f32(*((float*)&fp8_e4m3_to_fp32_lut[Ap_k[3]]));

        // 24 FMLAs (4 rows × 6 vectors)
        acc00 = svmla_f32_x(pg, acc00, a0, b0); acc01 = svmla_f32_x(pg, acc01, a0, b1);
        acc02 = svmla_f32_x(pg, acc02, a0, b2); acc03 = svmla_f32_x(pg, acc03, a0, b3);
        acc04 = svmla_f32_x(pg, acc04, a0, b4); acc05 = svmla_f32_x(pg, acc05, a0, b5);
        acc10 = svmla_f32_x(pg, acc10, a1, b0); acc11 = svmla_f32_x(pg, acc11, a1, b1);
        acc12 = svmla_f32_x(pg, acc12, a1, b2); acc13 = svmla_f32_x(pg, acc13, a1, b3);
        acc14 = svmla_f32_x(pg, acc14, a1, b4); acc15 = svmla_f32_x(pg, acc15, a1, b5);
        acc20 = svmla_f32_x(pg, acc20, a2, b0); acc21 = svmla_f32_x(pg, acc21, a2, b1);
        acc22 = svmla_f32_x(pg, acc22, a2, b2); acc23 = svmla_f32_x(pg, acc23, a2, b3);
        acc24 = svmla_f32_x(pg, acc24, a2, b4); acc25 = svmla_f32_x(pg, acc25, a2, b5);
        acc30 = svmla_f32_x(pg, acc30, a3, b0); acc31 = svmla_f32_x(pg, acc31, a3, b1);
        acc32 = svmla_f32_x(pg, acc32, a3, b2); acc33 = svmla_f32_x(pg, acc33, a3, b3);
        acc34 = svmla_f32_x(pg, acc34, a3, b4); acc35 = svmla_f32_x(pg, acc35, a3, b5);
    }

    // Store (4 rows × 6 vectors)
    svst1_f32(pg, C + 0*ldc + 0*VL_FP32, acc00); svst1_f32(pg, C + 0*ldc + 1*VL_FP32, acc01);
    svst1_f32(pg, C + 0*ldc + 2*VL_FP32, acc02); svst1_f32(pg, C + 0*ldc + 3*VL_FP32, acc03);
    svst1_f32(pg, C + 0*ldc + 4*VL_FP32, acc04); svst1_f32(pg, C + 0*ldc + 5*VL_FP32, acc05);
    svst1_f32(pg, C + 1*ldc + 0*VL_FP32, acc10); svst1_f32(pg, C + 1*ldc + 1*VL_FP32, acc11);
    svst1_f32(pg, C + 1*ldc + 2*VL_FP32, acc12); svst1_f32(pg, C + 1*ldc + 3*VL_FP32, acc13);
    svst1_f32(pg, C + 1*ldc + 4*VL_FP32, acc14); svst1_f32(pg, C + 1*ldc + 5*VL_FP32, acc15);
    svst1_f32(pg, C + 2*ldc + 0*VL_FP32, acc20); svst1_f32(pg, C + 2*ldc + 1*VL_FP32, acc21);
    svst1_f32(pg, C + 2*ldc + 2*VL_FP32, acc22); svst1_f32(pg, C + 2*ldc + 3*VL_FP32, acc23);
    svst1_f32(pg, C + 2*ldc + 4*VL_FP32, acc24); svst1_f32(pg, C + 2*ldc + 5*VL_FP32, acc25);
    svst1_f32(pg, C + 3*ldc + 0*VL_FP32, acc30); svst1_f32(pg, C + 3*ldc + 1*VL_FP32, acc31);
    svst1_f32(pg, C + 3*ldc + 2*VL_FP32, acc32); svst1_f32(pg, C + 3*ldc + 3*VL_FP32, acc33);
    svst1_f32(pg, C + 3*ldc + 4*VL_FP32, acc34); svst1_f32(pg, C + 3*ldc + 5*VL_FP32, acc35);
}

// Convert packed FP8 B to FP32 using gather (one-time cost)
static void convert_fp8_B_to_fp32(const fp8_e4m3_t* Bp, float* Bp_f32,
                                   int64_t K, int64_t N_tile) {
    svbool_t pg = svptrue_b32();
    for (int64_t k = 0; k < K; k++) {
        for (int64_t j = 0; j < N_tile; j += VL_FP32) {
            const fp8_e4m3_t* src = Bp + k * N_tile + j;
            float* dst = Bp_f32 + k * N_tile + j;

            svuint8_t bytes = svld1_u8(svwhilelt_b8(0, 16), src);
            svuint32_t idx32 = svunpklo_u32(svunpklo_u16(bytes));
            svuint32_t off = svlsl_n_u32_x(pg, idx32, 2);
            svuint32_t fp32_u32 = svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp32_lut, off);
            svst1_f32(pg, dst, svreinterpret_f32(fp32_u32));
        }
    }
}

// Convert packed FP8 A to FP32 using scalar LUT lookups (one-time cost)
// Packed A: panels of MR × K (MR FP8 values per K)
static void convert_fp8_A_to_fp32(const fp8_e4m3_t* Ap, float* Ap_f32,
                                   int64_t M_panels, int64_t K) {
    for (int64_t panel = 0; panel < M_panels; panel++) {
        for (int64_t k = 0; k < K; k++) {
            for (int m = 0; m < MR_FP8; m++) {
                int64_t idx = panel * MR_FP8 * K + k * MR_FP8 + m;
                uint8_t fp8_val = Ap[idx];
                Ap_f32[idx] = *((float*)&fp8_e4m3_to_fp32_lut[fp8_val]);
            }
        }
    }
}

// Full GEMM with pre-converted A and B (all FP32 in hot loop)
void fp8_gemm(const fp8_e4m3_t* A, int64_t lda,
              const fp8_e4m3_t* B, int64_t ldb,
              float* C, int64_t ldc,
              int64_t M, int64_t N, int64_t K) {

    const int64_t N_tile = NR_FP8 * VL_FP32;  // 48 FP32 elements per tile
    int64_t M_pad = round_up(M, MR_FP8);
    int64_t M_panels = M_pad / MR_FP8;

    // Allocate packed buffers (FP8 for packing, FP32 for kernel)
    size_t Ap_fp8_size = (size_t)M_pad * K;
    size_t Ap_f32_size = (size_t)M_pad * K * sizeof(float);
    size_t Bp_fp8_size = (size_t)K * N_tile;
    size_t Bp_f32_size = (size_t)K * N_tile * sizeof(float);

    fp8_e4m3_t* Ap_fp8 = (fp8_e4m3_t*)aligned_alloc(64, Ap_fp8_size);
    float* Ap_f32 = (float*)aligned_alloc(64, Ap_f32_size);
    fp8_e4m3_t* Bp_fp8 = (fp8_e4m3_t*)aligned_alloc(64, Bp_fp8_size);
    float* Bp_f32 = (float*)aligned_alloc(64, Bp_f32_size);

    if (!Ap_fp8 || !Ap_f32 || !Bp_fp8 || !Bp_f32) {
        fprintf(stderr, "fp8_gemm: allocation failed\n");
        free(Ap_fp8);
        free(Ap_f32);
        free(Bp_fp8);
        free(Bp_f32);
        return;
    }

    // Pack FP8 matrices
    pack_fp8_A(A, lda, Ap_fp8, M, K);
    pack_fp8_B(B, ldb, Bp_fp8, K, N);

    // Pre-convert A and B to FP32 (one-time cost)
    convert_fp8_A_to_fp32(Ap_fp8, Ap_f32, M_panels, K);
    convert_fp8_B_to_fp32(Bp_fp8, Bp_f32, K, N_tile);

    // Zero output
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            C[i * ldc + j] = 0.0f;
        }
    }

    // Execute pure FP32 kernel for each tile (NO gathers in hot loop)
    for (int64_t ir = 0; ir < M_pad; ir += MR_FP8) {
        const float* Ap_tile = Ap_f32 + (ir / MR_FP8) * (MR_FP8 * K);
        float* C_tile = C + ir * ldc;
        fp8_gemm_kernel_fp32(Ap_tile, Bp_f32, C_tile, ldc, K);
    }

    free(Ap_fp8);
    free(Ap_f32);
    free(Bp_fp8);
    free(Bp_f32);
}
