// fused_gemm.h
// Fused GEMM: O = (A @ B^T) @ C
// All matrices [L, d] where d=256
// Uses SDOT for both stages with intermediate quantization

#ifndef FUSED_GEMM_H
#define FUSED_GEMM_H

#include <stdint.h>
#include <stddef.h>

// Tile sizes
#define FUSED_MR 6      // Output rows per tile
#define FUSED_LB 64     // L dimension block size (matches NR)
#define FUSED_D 256     // d dimension (fixed)

// Packed matrix for fused GEMM
typedef struct {
    int8_t* data;
    int L;           // Sequence length
    int d;           // Feature dimension (256)
    size_t size;     // Buffer size in bytes
} fused_matrix_t;

// Pack A matrix [L, d] for first GEMM (A @ B^T)
// Layout: [L/MR][d/4][MR][4] interleaved for ld1rw
fused_matrix_t* pack_A_fused(const int8_t* A, int L, int d);

// Pack B matrix [L, d] for first GEMM (transposed access)
// Layout: [L/LB][d/4][LB/16][4][16][4] SDOT layout
fused_matrix_t* pack_B_fused(const int8_t* B, int L, int d);

// Pack C matrix [L, d] for second GEMM
// Layout: [L/LB][d/4][LB/16][4][16][4] SDOT layout
fused_matrix_t* pack_C_fused(const int8_t* C, int L, int d);

// Free packed matrix
void free_fused_matrix(fused_matrix_t* m);

// Fused GEMM: O = (A @ B^T) @ C
// A, B, C: [L, d] packed matrices
// O: [L, d] output (int32)
// scale1: quantization scale for first GEMM output
// scale2: quantization scale for second GEMM output
void fused_gemm_ABtC(const fused_matrix_t* Apack,
                      const fused_matrix_t* Bpack,
                      const fused_matrix_t* Cpack,
                      int32_t* O, int ldo,
                      float scale1, float scale2);

// Non-fused reference for correctness testing
void ref_gemm_ABtC(const int8_t* A, const int8_t* B, const int8_t* C,
                    int32_t* O, int L, int d);

// Declare kernels
// First stage: [MR, LB] = [MR, d] @ [LB, d]^T, K=256
extern void kernel_6x4_unroll(const int8_t* Apack, const int8_t* Bpack,
                               int32_t* C, int ldc);

// Second stage kernel: [MR, d_tile] += [MR, LB] @ [LB, d_tile], K=64
// This uses quantized int8 intermediate
extern void kernel_fused_stage2(const int8_t* S_tile, const int8_t* Cpack,
                                 int32_t* O_acc, int ldo);

#endif // FUSED_GEMM_H
