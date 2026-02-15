// Fused GEMM (A@B)@C Kernel for A64FX SVE FP32
// Computes E[M,N] = (A[M,K1] @ B[K1,K2]) @ C[K2,N]
#ifndef FUSED_GEMM_H
#define FUSED_GEMM_H

#include <stdint.h>
#include <stddef.h>

// Tile sizes for 8x3 microkernel (SVE 512-bit)
#define MR 8       // Rows per micro-tile
#define NR 48      // Columns per micro-tile (3 * 16 FP32)
#define KU 4       // K-unroll factor

// Fused GEMM: E = (A @ B) @ C
// Two-pass implementation with L2-resident intermediate
void fused_gemm_abc(
    int M, int K1, int K2, int N,
    const float* A, int lda,    // A[M,K1] row-major
    const float* B, int ldb,    // B[K1,K2] row-major
    const float* C, int ldc,    // C[K2,N] row-major
    float* E, int lde           // E[M,N] row-major output
);

// Single GEMM: C = A @ B (for comparison)
void gemm_fp32(
    int M, int K, int N,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
);

// Assembly microkernel interface
// Computes C[8,48] += A_packed[K,8] @ B_packed[K,48]
// Arguments: x0=A_packed, x1=B_packed, x2=C, x3=K, x5=ldc (in bytes)
extern void micro_kernel_fp32_8x3_unroll4(
    const float* A_packed,
    const float* B_packed,
    float* C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes
);

// Packing functions
void pack_A_fp32(int M, int K, const float* A, int lda, float* A_packed);
void pack_B_fp32(int K, int N, const float* B, int ldb, float* B_packed);

// Workspace size calculations
size_t fused_gemm_workspace_size(int M, int K1, int K2, int N);

#endif // FUSED_GEMM_H
