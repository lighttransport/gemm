#ifndef GEMM_PACK_OPT_H
#define GEMM_PACK_OPT_H

#include <stdint.h>

// ============================================================================
// Optimized Packing for 95%+ Efficiency INT8 SDOT GEMM
// ============================================================================
//
// Key insight: A packing must be INTERLEAVED for efficient ld1rw loading
//
// Original A packing [MR][K]:
//   Row 0: [k0 k1 k2 k3 | k4 k5 k6 k7 | ...]
//   Row 1: [k0 k1 k2 k3 | k4 k5 k6 k7 | ...]
//   ...
//   Requires 6 scattered loads per K-group, 6 pointer updates
//
// Optimized A packing [K/4][MR][4]:
//   K-group 0: [R0:k0-3][R1:k0-3][R2:k0-3][R3:k0-3][R4:k0-3][R5:k0-3]
//   K-group 1: [R0:k4-7][R1:k4-7][R2:k4-7][R3:k4-7][R4:k4-7][R5:k4-7]
//   ...
//   Allows sequential ld1rw with single pointer, 1 pointer update per K-group
//
// B packing remains [K/4][NR][4] for SDOT (already optimal)

#define MR_OPT 6
#define NR_OPT 64
#define K_FIXED 256

// Pack A matrix with interleaved layout for ld1rw
// Input:  A[M][K] row-major
// Output: Apack[K/4][MR][4] interleaved
// Size:   MR × K bytes per tile
void pack_A_interleaved(const int8_t* A, int lda, int8_t* Apack, int M);

// Pack B matrix for SDOT (same as before)
// Input:  B[N][K] row-major (accessed as B^T)
// Output: Bpack[K/4][4][16][4] for 64 columns
// Size:   NR × K bytes per tile
void pack_B_sdot(const int8_t* B, int ldb, int8_t* Bpack, int N);

// Optimized microkernel: MR=6, NR=64, K=256
// Uses ld1rw for A loads, fully unrolled K-loop
void kernel_6x4_ultra(const int8_t* Apack, const int8_t* Bpack,
                      int32_t* C, int ldc);

// Software-pipelined microkernel: MR=6, NR=64, K=256
// Uses ld1rw for A loads, 4× K-loop unrolling, load/compute overlap
void kernel_6x4_pipe(const int8_t* Apack, const int8_t* Bpack,
                     int32_t* C, int ldc);

// Batched GEMM for multiple attention heads
// Computes C[b] = A[b] × B[b]^T for b in [0, batch)
void gemm_batch_opt(const int8_t* A, int lda, int64_t strideA,
                    const int8_t* B, int ldb, int64_t strideB,
                    int32_t* C, int ldc, int64_t strideC,
                    int M, int N, int K, int batch);

// Single GEMM optimized driver (uses kernel_6x4_ultra)
void gemm_opt_driver(const int8_t* A, int lda,
                     const int8_t* B, int ldb,
                     int32_t* C, int ldc,
                     int M, int N, int K);

// Single GEMM pipelined driver (uses kernel_6x4_pipe)
void gemm_pipe_driver(const int8_t* A, int lda,
                      const int8_t* B, int ldb,
                      int32_t* C, int ldc,
                      int M, int N, int K);

// Batched GEMM with pipelined kernel
void gemm_batch_pipe(const int8_t* A, int lda, int64_t strideA,
                     const int8_t* B, int ldb, int64_t strideB,
                     int32_t* C, int ldc, int64_t strideC,
                     int M, int N, int K, int batch);

#endif // GEMM_PACK_OPT_H
