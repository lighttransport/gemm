/*
 * FP8 GEMM with FP32 Accumulation for A64FX
 *
 * Flow: FP8 input -> Gather LUT -> FP16 -> FMLALB/FMLALT -> FP32 accumulator
 *
 * A64FX SVE is 512-bit:
 *   - 64 FP8 elements per vector
 *   - 32 FP16 elements per vector
 *   - 16 FP32 elements per vector
 */

#ifndef FP8_GEMM_H
#define FP8_GEMM_H

#include <stdint.h>
#include <arm_sve.h>

// FP8 types
typedef uint8_t fp8_e4m3_t;
typedef uint8_t fp8_e5m2_t;

// Vector lengths for 512-bit SVE
#define VL_BYTES  64
#define VL_FP8    64   // 64 FP8 per vector
#define VL_FP16   32   // 32 FP16 per vector
#define VL_FP32   16   // 16 FP32 per vector

// Tile sizes for micro-kernel
// With FP32 accumulators: MR rows × NR FP32-vectors
// 8×3 tile: 8 × 3 = 24 accumulator registers
// Better arithmetic intensity: 3.43 FLOPS/byte (vs 2.74 for 4×6)
// N per tile = NR × VL_FP32 = 3 × 16 = 48 elements
#define MR_FP8 8
#define NR_FP8 3

// LUT declarations (from fp8_convert.h)
extern uint32_t fp8_e4m3_to_fp16_lut32[256];
extern uint32_t fp8_e5m2_to_fp16_lut32[256];
extern void init_fp8_luts(void);

// Initialize FP8 GEMM (calls init_fp8_luts)
void fp8_gemm_init(void);

// Pack FP8 A matrix: M×K row-major -> packed panels (MR elements per K)
void pack_fp8_A(const fp8_e4m3_t* A, int64_t lda, fp8_e4m3_t* Ap,
                int64_t M, int64_t K);

// Pack FP8 B matrix: K×N row-major -> packed panels (NR vectors per K)
void pack_fp8_B(const fp8_e4m3_t* B, int64_t ldb, fp8_e4m3_t* Bp,
                int64_t K, int64_t N);

// Micro-kernel: FP8 inputs, FP32 accumulation
// Ap: packed A panel (MR × K)
// Bp: packed B panel (K × NR × VL_FP16)
// C: output FP32 matrix (MR × NR×VL_FP16)
// ldc: leading dimension of C in FP32 elements
// K: inner dimension
void fp8_gemm_kernel_3x4(const fp8_e4m3_t* Ap, const fp8_e4m3_t* Bp,
                          float* C, int64_t ldc, int64_t K);

// Full GEMM: C = A × B (FP8 inputs, FP32 output)
// A: M×K FP8 matrix (row-major)
// B: K×N FP8 matrix (row-major)
// C: M×N FP32 matrix (row-major)
void fp8_gemm(const fp8_e4m3_t* A, int64_t lda,
              const fp8_e4m3_t* B, int64_t ldb,
              float* C, int64_t ldc,
              int64_t M, int64_t N, int64_t K);

// Pure FP32 kernel (for kernel-only benchmarking)
// Ap_f32: packed A in FP32, MR floats per K (already converted)
// Bp_f32: packed B in FP32, NR×VL_FP32 floats per K (already converted)
void fp8_gemm_kernel_fp32(const float* Ap_f32, const float* Bp_f32,
                          float* C, int64_t ldc, int64_t K);

// Assembly kernel with software pipelining (highest performance)
extern void fp8_gemm_kernel_asm(const float* Ap_f32, const float* Bp_f32,
                                float* C, int64_t ldc, int64_t K);

#endif // FP8_GEMM_H
