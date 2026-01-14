// fused_gemm.c
// Fused GEMM: O = (A @ B^T) @ C
// All matrices [L, d] where d=256

#include "fused_gemm.h"
#include <stdlib.h>
#include <string.h>
#include <arm_sve.h>

// ============================================================================
// Packing functions
// ============================================================================

// Pack A matrix [L, d] for first GEMM
// Interleaved layout: [L/MR][d/4][MR][4]
fused_matrix_t* pack_A_fused(const int8_t* A, int L, int d) {
    fused_matrix_t* m = malloc(sizeof(fused_matrix_t));
    if (!m) return NULL;

    int m_tiles = (L + FUSED_MR - 1) / FUSED_MR;
    m->size = m_tiles * FUSED_MR * d;
    m->L = L;
    m->d = d;

    if (posix_memalign((void**)&m->data, 256, m->size) != 0) {
        free(m);
        return NULL;
    }

    int8_t* dst = m->data;
    for (int m0 = 0; m0 < L; m0 += FUSED_MR) {
        int mr = (m0 + FUSED_MR <= L) ? FUSED_MR : (L - m0);

        // Interleaved: [d/4][MR][4]
        for (int k = 0; k < d; k += 4) {
            for (int m = 0; m < FUSED_MR; m++) {
                if (m < mr) {
                    dst[0] = A[(m0 + m) * d + k + 0];
                    dst[1] = A[(m0 + m) * d + k + 1];
                    dst[2] = A[(m0 + m) * d + k + 2];
                    dst[3] = A[(m0 + m) * d + k + 3];
                } else {
                    dst[0] = dst[1] = dst[2] = dst[3] = 0;
                }
                dst += 4;
            }
        }
    }

    return m;
}

// Pack B matrix [L, d] for first GEMM (A @ B^T)
// SDOT layout: [L/LB][d/4][4 vectors][16 lanes][4 bytes]
fused_matrix_t* pack_B_fused(const int8_t* B, int L, int d) {
    fused_matrix_t* m = malloc(sizeof(fused_matrix_t));
    if (!m) return NULL;

    int l_tiles = (L + FUSED_LB - 1) / FUSED_LB;
    m->size = l_tiles * FUSED_LB * d;
    m->L = L;
    m->d = d;

    if (posix_memalign((void**)&m->data, 256, m->size) != 0) {
        free(m);
        return NULL;
    }

    int8_t* dst = m->data;
    for (int l0 = 0; l0 < L; l0 += FUSED_LB) {
        int lb = (l0 + FUSED_LB <= L) ? FUSED_LB : (L - l0);

        // SDOT layout: [d/4][4 vectors][16 lanes][4 bytes]
        for (int k = 0; k < d; k += 4) {
            for (int vec = 0; vec < 4; vec++) {
                for (int lane = 0; lane < 16; lane++) {
                    int l = vec * 16 + lane;
                    if (l < lb) {
                        dst[0] = B[(l0 + l) * d + k + 0];
                        dst[1] = B[(l0 + l) * d + k + 1];
                        dst[2] = B[(l0 + l) * d + k + 2];
                        dst[3] = B[(l0 + l) * d + k + 3];
                    } else {
                        dst[0] = dst[1] = dst[2] = dst[3] = 0;
                    }
                    dst += 4;
                }
            }
        }
    }

    return m;
}

// Pack C matrix [L, d] for second GEMM (S @ C)
// Layout: [L/LB][d/4][LB/16][4][16][4]
// For second stage: C is treated as [L, d] where we access blocks of [LB, d]
fused_matrix_t* pack_C_fused(const int8_t* C, int L, int d) {
    fused_matrix_t* m = malloc(sizeof(fused_matrix_t));
    if (!m) return NULL;

    int l_tiles = (L + FUSED_LB - 1) / FUSED_LB;
    // For stage 2, we need C packed per d-tile (64 columns at a time)
    // Layout: [L/LB][d/64][K/4][4 vectors][16 lanes][4]
    // where K=LB=64 for stage 2
    m->size = l_tiles * d * FUSED_LB;
    m->L = L;
    m->d = d;

    if (posix_memalign((void**)&m->data, 256, m->size) != 0) {
        free(m);
        return NULL;
    }

    int8_t* dst = m->data;
    for (int l0 = 0; l0 < L; l0 += FUSED_LB) {
        int lb = (l0 + FUSED_LB <= L) ? FUSED_LB : (L - l0);

        // For each d-tile of 64 columns
        for (int n0 = 0; n0 < d; n0 += 64) {
            // Pack [LB, 64] block in SDOT layout
            // K=LB=64, so [K/4][4 vectors][16 lanes][4]
            for (int k = 0; k < FUSED_LB; k += 4) {
                for (int vec = 0; vec < 4; vec++) {
                    for (int lane = 0; lane < 16; lane++) {
                        int n = n0 + vec * 16 + lane;
                        if (k < lb && n < d) {
                            dst[0] = C[(l0 + k + 0) * d + n];
                            dst[1] = C[(l0 + k + 1) * d + n];
                            dst[2] = C[(l0 + k + 2) * d + n];
                            dst[3] = C[(l0 + k + 3) * d + n];
                        } else {
                            dst[0] = dst[1] = dst[2] = dst[3] = 0;
                        }
                        dst += 4;
                    }
                }
            }
        }
    }

    return m;
}

void free_fused_matrix(fused_matrix_t* m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

// ============================================================================
// Quantization helper
// ============================================================================

// Quantize int32 S_tile to int8 for second stage
// Uses floating-point for accurate scaling
static inline void quantize_s_tile(const int32_t* src, int8_t* dst,
                                    int rows, int cols, float scale) {
    for (int i = 0; i < rows * cols; i++) {
        // Use float to handle small scales correctly
        float val = (float)src[i] * scale;
        // Round and clamp to int8 range
        int32_t ival = (int32_t)(val + (val >= 0 ? 0.5f : -0.5f));
        if (ival > 127) ival = 127;
        if (ival < -128) ival = -128;
        dst[i] = (int8_t)ival;
    }
}

// Pack quantized S_tile for stage 2
// Input: S_tile [MR, LB] row-major
// Output: S_pack [K/4][MR][4] where K=LB
static inline void pack_s_tile(const int8_t* src, int8_t* dst, int mr, int lb) {
    for (int k = 0; k < lb; k += 4) {
        for (int m = 0; m < FUSED_MR; m++) {
            if (m < mr) {
                dst[0] = src[m * lb + k + 0];
                dst[1] = src[m * lb + k + 1];
                dst[2] = src[m * lb + k + 2];
                dst[3] = src[m * lb + k + 3];
            } else {
                dst[0] = dst[1] = dst[2] = dst[3] = 0;
            }
            dst += 4;
        }
    }
}

// ============================================================================
// Fused GEMM implementation
// ============================================================================

// Declare assembly kernels
extern void kernel_6x4_unroll(const int8_t* Apack, const int8_t* Bpack,
                               int32_t* C, int ldc);
extern void kernel_stage2_k64(const int8_t* S_tile, const int8_t* Cpack,
                               int32_t* O_acc, int ldo);

void fused_gemm_ABtC(const fused_matrix_t* Apack,
                      const fused_matrix_t* Bpack,
                      const fused_matrix_t* Cpack,
                      int32_t* O, int ldo,
                      float scale1, float scale2) {
    int L = Apack->L;
    int d = Apack->d;

    // Temporary buffers for intermediate results
    int32_t S_tile_i32[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));
    int8_t S_tile_i8[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));
    int8_t S_pack[FUSED_MR * FUSED_LB] __attribute__((aligned(256)));

    // Per-output-tile accumulator
    int32_t O_tile[FUSED_MR * d] __attribute__((aligned(256)));

    int A_tile_stride = FUSED_MR * d;  // Bytes per A tile
    int B_tile_stride = FUSED_LB * d;  // Bytes per B tile
    int C_tile_stride = FUSED_LB * d;  // Bytes per C L-tile
    int C_d_stride = FUSED_LB * 64;    // Bytes per C d-tile within L-tile

    // Outer loop over M dimension (output rows)
    for (int m0 = 0; m0 < L; m0 += FUSED_MR) {
        int mr = (m0 + FUSED_MR <= L) ? FUSED_MR : (L - m0);

        const int8_t* Aptr = Apack->data + (m0 / FUSED_MR) * A_tile_stride;

        // Zero output accumulator for this M-tile
        memset(O_tile, 0, FUSED_MR * d * sizeof(int32_t));

        // Loop over L dimension (reduction)
        for (int l0 = 0; l0 < L; l0 += FUSED_LB) {
            int lb = (l0 + FUSED_LB <= L) ? FUSED_LB : (L - l0);

            const int8_t* Bptr = Bpack->data + (l0 / FUSED_LB) * B_tile_stride;
            const int8_t* Cptr = Cpack->data + (l0 / FUSED_LB) * C_tile_stride;

            // Stage 1: S_tile = A[m0:m0+MR, :] @ B[l0:l0+LB, :]^T
            // Result: [MR, LB] from [MR, d] @ [LB, d]^T, K=d=256
            memset(S_tile_i32, 0, sizeof(S_tile_i32));
            kernel_6x4_unroll(Aptr, Bptr, S_tile_i32, FUSED_LB * sizeof(int32_t));

            // Quantize S_tile from int32 to int8
            quantize_s_tile(S_tile_i32, S_tile_i8, FUSED_MR, FUSED_LB, scale1);

            // Pack S_tile for stage 2
            pack_s_tile(S_tile_i8, S_pack, mr, lb);

            // Stage 2: O_tile += S_tile @ C[l0:l0+LB, :]
            // For each d-tile of 64 columns
            for (int n0 = 0; n0 < d; n0 += 64) {
                const int8_t* C_block = Cptr + (n0 / 64) * C_d_stride;
                int32_t* O_block = O_tile + n0;

                // [MR, 64] += [MR, LB] @ [LB, 64], K=LB=64
                kernel_stage2_k64(S_pack, C_block, O_block, d * sizeof(int32_t));
            }
        }

        // Store O_tile to output
        for (int m = 0; m < mr; m++) {
            for (int n = 0; n < d; n++) {
                O[(m0 + m) * ldo + n] = O_tile[m * d + n];
            }
        }
    }
}

// ============================================================================
// Reference implementation
// ============================================================================

void ref_gemm_ABtC(const int8_t* A, const int8_t* B, const int8_t* C,
                    int32_t* O, int L, int d) {
    // Allocate intermediate S matrix
    int64_t* S = calloc((size_t)L * L, sizeof(int64_t));
    if (!S) return;

    // Stage 1: S = A @ B^T
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int64_t sum = 0;
            for (int k = 0; k < d; k++) {
                sum += (int32_t)A[i * d + k] * (int32_t)B[j * d + k];
            }
            S[i * L + j] = sum;
        }
    }

    // Stage 2: O = S @ C
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < d; j++) {
            int64_t sum = 0;
            for (int k = 0; k < L; k++) {
                sum += S[i * L + k] * (int64_t)C[k * d + j];
            }
            // Truncate to int32
            O[i * d + j] = (int32_t)sum;
        }
    }

    free(S);
}
