// gemm_prepacked.h
// Pre-packed GEMM for maximum efficiency
// Eliminates packing overhead by reusing packed buffers

#ifndef GEMM_PREPACKED_H
#define GEMM_PREPACKED_H

#include <stdint.h>
#include <stddef.h>

#define MR_PACK 6
#define NR_PACK 64
#define K_PACK 256

// Opaque handle for pre-packed matrix
typedef struct {
    int8_t* data;      // Packed data
    int rows;          // Original rows (M or N)
    int K;             // K dimension (always 256)
    int tiles;         // Number of tiles
    size_t size;       // Total buffer size
} packed_matrix_t;

// Pack A matrix [M×K] for reuse
// Returns handle to packed matrix
packed_matrix_t* pack_A_prepacked(const int8_t* A, int lda, int M, int K);

// Pack B matrix [N×K] for reuse
packed_matrix_t* pack_B_prepacked(const int8_t* B, int ldb, int N, int K);

// Free packed matrix
void free_packed_matrix(packed_matrix_t* pm);

// GEMM with pre-packed matrices
// C = Apack × Bpack^T
void gemm_prepacked(const packed_matrix_t* Apack,
                    const packed_matrix_t* Bpack,
                    int32_t* C, int ldc);

// Batch GEMM with shared pre-packed B matrix
// Each batch uses different A, same B
// Useful for attention: Q × K^T where K is shared
void gemm_batch_shared_B(const int8_t* A, int lda, int64_t strideA,
                         const packed_matrix_t* Bpack,
                         int32_t* C, int ldc, int64_t strideC,
                         int M, int batch);

// Direct kernel call for benchmarking
void kernel_6x4_ultra(const int8_t* Apack, const int8_t* Bpack,
                      int32_t* C, int ldc);

#endif // GEMM_PREPACKED_H
