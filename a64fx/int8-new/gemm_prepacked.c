// gemm_prepacked.c
// Pre-packed GEMM implementation for maximum efficiency

#include "gemm_prepacked.h"
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Pack A matrix with interleaved layout [K/4][MR][4]
// ============================================================================

packed_matrix_t* pack_A_prepacked(const int8_t* A, int lda, int M, int K) {
    if (K != K_PACK) return NULL;

    packed_matrix_t* pm = malloc(sizeof(packed_matrix_t));
    if (!pm) return NULL;

    int tiles = (M + MR_PACK - 1) / MR_PACK;
    size_t tile_size = MR_PACK * K_PACK;
    pm->size = tiles * tile_size;
    pm->rows = M;
    pm->K = K;
    pm->tiles = tiles;

    if (posix_memalign((void**)&pm->data, 256, pm->size) != 0) {
        free(pm);
        return NULL;
    }

    // Pack each MR×K tile
    int8_t* dst = pm->data;
    for (int m0 = 0; m0 < M; m0 += MR_PACK) {
        int mr = (m0 + MR_PACK <= M) ? MR_PACK : (M - m0);

        // Interleaved layout: [K/4][MR][4]
        for (int k = 0; k < K; k += 4) {
            for (int m = 0; m < MR_PACK; m++) {
                if (m < mr) {
                    dst[0] = A[(m0 + m) * lda + k + 0];
                    dst[1] = A[(m0 + m) * lda + k + 1];
                    dst[2] = A[(m0 + m) * lda + k + 2];
                    dst[3] = A[(m0 + m) * lda + k + 3];
                } else {
                    dst[0] = dst[1] = dst[2] = dst[3] = 0;
                }
                dst += 4;
            }
        }
    }

    return pm;
}

// ============================================================================
// Pack B matrix for SDOT layout [K/4][4 vectors][16 lanes][4]
// ============================================================================

packed_matrix_t* pack_B_prepacked(const int8_t* B, int ldb, int N, int K) {
    if (K != K_PACK) return NULL;

    packed_matrix_t* pm = malloc(sizeof(packed_matrix_t));
    if (!pm) return NULL;

    int tiles = (N + NR_PACK - 1) / NR_PACK;
    size_t tile_size = NR_PACK * K_PACK;
    pm->size = tiles * tile_size;
    pm->rows = N;
    pm->K = K;
    pm->tiles = tiles;

    if (posix_memalign((void**)&pm->data, 256, pm->size) != 0) {
        free(pm);
        return NULL;
    }

    // Pack each 64×K tile
    int8_t* dst = pm->data;
    for (int n0 = 0; n0 < N; n0 += NR_PACK) {
        int nr = (n0 + NR_PACK <= N) ? NR_PACK : (N - n0);

        // SDOT layout: [K/4][4 vectors][16 lanes][4 bytes]
        for (int k = 0; k < K; k += 4) {
            for (int vec = 0; vec < 4; vec++) {
                for (int lane = 0; lane < 16; lane++) {
                    int n = vec * 16 + lane;
                    if (n < nr) {
                        dst[0] = B[(n0 + n) * ldb + k + 0];
                        dst[1] = B[(n0 + n) * ldb + k + 1];
                        dst[2] = B[(n0 + n) * ldb + k + 2];
                        dst[3] = B[(n0 + n) * ldb + k + 3];
                    } else {
                        dst[0] = dst[1] = dst[2] = dst[3] = 0;
                    }
                    dst += 4;
                }
            }
        }
    }

    return pm;
}

// ============================================================================
// Free packed matrix
// ============================================================================

void free_packed_matrix(packed_matrix_t* pm) {
    if (pm) {
        free(pm->data);
        free(pm);
    }
}

// ============================================================================
// Edge case handler
// ============================================================================

static void gemm_edge_scalar(const int8_t* A, int lda,
                             const int8_t* B, int ldb,
                             int32_t* C, int ldc,
                             int mr, int nr, int K) {
    for (int m = 0; m < mr; m++) {
        for (int n = 0; n < nr; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * lda + k] * (int32_t)B[n * ldb + k];
            }
            C[m * ldc + n] = sum;
        }
    }
}

// ============================================================================
// GEMM with pre-packed matrices
// ============================================================================

void gemm_prepacked(const packed_matrix_t* Apack,
                    const packed_matrix_t* Bpack,
                    int32_t* C, int ldc) {
    int M = Apack->rows;
    int N = Bpack->rows;
    int K = K_PACK;

    int8_t* Aptr = Apack->data;
    for (int m0 = 0; m0 < M; m0 += MR_PACK) {
        int mr = (m0 + MR_PACK <= M) ? MR_PACK : (M - m0);

        int8_t* Bptr = Bpack->data;
        for (int n0 = 0; n0 < N; n0 += NR_PACK) {
            int nr = (n0 + NR_PACK <= N) ? NR_PACK : (N - n0);

            int32_t* C_tile = C + m0 * ldc + n0;

            if (mr == MR_PACK && nr == NR_PACK) {
                // Full tile - call optimized kernel
                kernel_6x4_ultra(Aptr, Bptr, C_tile, ldc * sizeof(int32_t));
            } else {
                // Edge case - need to unpack and compute
                // For simplicity, use a small local buffer
                int32_t C_local[MR_PACK * NR_PACK] = {0};
                kernel_6x4_ultra(Aptr, Bptr, C_local, NR_PACK * sizeof(int32_t));

                // Copy valid portion to output
                for (int m = 0; m < mr; m++) {
                    for (int n = 0; n < nr; n++) {
                        C_tile[m * ldc + n] = C_local[m * NR_PACK + n];
                    }
                }
            }

            Bptr += NR_PACK * K;
        }

        Aptr += MR_PACK * K;
    }
}

// ============================================================================
// Batch GEMM with shared pre-packed B matrix
// ============================================================================

void gemm_batch_shared_B(const int8_t* A, int lda, int64_t strideA,
                         const packed_matrix_t* Bpack,
                         int32_t* C, int ldc, int64_t strideC,
                         int M, int batch) {
    int N = Bpack->rows;
    int K = K_PACK;

    // Allocate A packing buffer once (reused across batch)
    int8_t* Apack_buf = NULL;
    posix_memalign((void**)&Apack_buf, 256, MR_PACK * K);
    if (!Apack_buf) return;

    for (int b = 0; b < batch; b++) {
        const int8_t* A_batch = A + b * strideA;
        int32_t* C_batch = C + b * strideC;

        // Process M in tiles
        for (int m0 = 0; m0 < M; m0 += MR_PACK) {
            int mr = (m0 + MR_PACK <= M) ? MR_PACK : (M - m0);

            // Pack current A tile
            int8_t* dst = Apack_buf;
            for (int k = 0; k < K; k += 4) {
                for (int m = 0; m < MR_PACK; m++) {
                    if (m < mr) {
                        dst[0] = A_batch[(m0 + m) * lda + k + 0];
                        dst[1] = A_batch[(m0 + m) * lda + k + 1];
                        dst[2] = A_batch[(m0 + m) * lda + k + 2];
                        dst[3] = A_batch[(m0 + m) * lda + k + 3];
                    } else {
                        dst[0] = dst[1] = dst[2] = dst[3] = 0;
                    }
                    dst += 4;
                }
            }

            // Process N tiles using pre-packed B
            int8_t* Bptr = Bpack->data;
            for (int n0 = 0; n0 < N; n0 += NR_PACK) {
                int nr = (n0 + NR_PACK <= N) ? NR_PACK : (N - n0);

                int32_t* C_tile = C_batch + m0 * ldc + n0;

                if (mr == MR_PACK && nr == NR_PACK) {
                    kernel_6x4_ultra(Apack_buf, Bptr, C_tile, ldc * sizeof(int32_t));
                } else {
                    int32_t C_local[MR_PACK * NR_PACK] = {0};
                    kernel_6x4_ultra(Apack_buf, Bptr, C_local, NR_PACK * sizeof(int32_t));
                    for (int m = 0; m < mr; m++) {
                        for (int n = 0; n < nr; n++) {
                            C_tile[m * ldc + n] = C_local[m * NR_PACK + n];
                        }
                    }
                }

                Bptr += NR_PACK * K;
            }
        }
    }

    free(Apack_buf);
}
