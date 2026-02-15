#ifndef GEMM_DRIVER_H
#define GEMM_DRIVER_H

#include <stdint.h>

// Microkernel declarations (implemented in assembly)
// Both kernels compute C[Mr×64] += A[Mr×256] × B[64×256]^T
// with packed inputs and int32 output

// 5x4 microkernel: MR=5, NR=64 (4 SVE vectors)
void kernel_5x4_256(const int8_t* Apack, const int8_t* Bpack,
                    int32_t* C, int ldc);

// 6x4 microkernel: MR=6, NR=64 (4 SVE vectors)
void kernel_6x4_256(const int8_t* Apack, const int8_t* Bpack,
                    int32_t* C, int ldc);

// 6x4 optimized microkernel: MR=6, NR=64, K-loop unrolled 8×
void kernel_6x4_opt_256(const int8_t* Apack, const int8_t* Bpack,
                        int32_t* C, int ldc);

// Macro-kernel drivers
// Compute C[M×N] = A[M×K] × B[N×K]^T with K=256
// A: M×K int8 row-major with leading dimension lda
// B: N×K int8 row-major with leading dimension ldb (accessed as B^T)
// C: M×N int32 row-major with leading dimension ldc

// Driver for 5x4 microkernel
void gemm_5x4_driver(const int8_t* A, int lda,
                     const int8_t* B, int ldb,
                     int32_t* C, int ldc,
                     int M, int N, int K);

// Driver for 6x4 microkernel
void gemm_6x4_driver(const int8_t* A, int lda,
                     const int8_t* B, int ldb,
                     int32_t* C, int ldc,
                     int M, int N, int K);

// Driver for 6x4 optimized microkernel (K-loop unrolled 8×)
void gemm_6x4_opt_driver(const int8_t* A, int lda,
                         const int8_t* B, int ldb,
                         int32_t* C, int ldc,
                         int M, int N, int K);

#endif // GEMM_DRIVER_H
