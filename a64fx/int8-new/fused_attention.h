// fused_attention.h
// Fused Attention: O = softmax(Q @ K^T / sqrt(d)) @ V
// All matrices [L, d] where d=256
// Uses INT8 SDOT for both GEMM stages with exp2 softmax

#ifndef FUSED_ATTENTION_H
#define FUSED_ATTENTION_H

#include <stdint.h>
#include <stddef.h>
#include "fused_gemm.h"
#include "softmax_exp2.h"

// Attention configuration
typedef struct {
    float scale;           // 1/sqrt(d) scaling factor
    int use_uint8_softmax; // 0 = int8, 1 = uint8 variant
} attention_config_t;

// ============================================================================
// Fused Attention: O = softmax(Q @ K^T / sqrt(d)) @ V
// ============================================================================
// Algorithm:
// 1. Compute S_tile = Q_tile @ K_tile^T (INT8 SDOT, K=d=256)
// 2. Apply softmax with exp2: P_tile = softmax(S_tile * scale)
// 3. Quantize P to int8 or uint8
// 4. Compute O_tile += P_tile @ V_tile (INT8 SDOT, K=LB=64)
//
// Two variants:
// - INT8: P in [-127, 127], V in [-128, 127], use SDOT
// - UINT8: P in [0, 255], V_biased = V + 128, use UDOT, correct bias

// INT8 variant
void fused_attention_int8(const fused_matrix_t* Qpack,    // [L, d] packed Q
                           const fused_matrix_t* Kpack,    // [L, d] packed K
                           const fused_matrix_t* Vpack,    // [L, d] packed V
                           int32_t* O,                      // [L, d] output
                           int ldo,
                           float scale);

// UINT8 variant with bias correction
void fused_attention_uint8(const fused_matrix_t* Qpack,
                            const fused_matrix_t* Kpack,
                            const fused_matrix_t* Vpack,
                            int32_t* O,
                            int ldo,
                            float scale);

// ============================================================================
// Online Softmax Attention (Correct FlashAttention-style)
// ============================================================================
// Uses online softmax normalization to correctly compute global softmax
// while processing in tiles. Maintains running max and sum per row.

void fused_attention_online(const fused_matrix_t* Qpack,
                             const fused_matrix_t* Kpack,
                             const int8_t* V,           // Unpacked V [L, d]
                             int32_t* O,
                             int ldo,
                             float scale);

// Reference implementation for correctness testing
void ref_attention(const int8_t* Q, const int8_t* K, const int8_t* V,
                    int32_t* O, int L, int d, float scale);

#endif // FUSED_ATTENTION_H
