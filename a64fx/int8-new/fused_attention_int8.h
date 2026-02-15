// fused_attention_int8.h
// Fused INT8 Attention: O = softmax(Q @ K^T / sqrt(d)) @ V
//
// Uses online softmax (FlashAttention style) for numerical stability
// Both Q@K^T and P@V use INT8 SDOT for maximum throughput

#ifndef FUSED_ATTENTION_INT8_H
#define FUSED_ATTENTION_INT8_H

#include <stdint.h>

// Fused attention parameters
typedef struct {
    int M;              // Query sequence length
    int N;              // Key/Value sequence length
    int D;              // Head dimension
    float scale;        // 1/sqrt(D) for attention scaling
    float qk_scale;     // Dequantization scale for Q@K^T
    float p_scale;      // Quantization scale for P (attention weights)
    float v_scale;      // Dequantization scale for V
    float o_scale;      // Output scale
} fused_attn_params_t;

// Main fused attention function
// Q: [M, D] INT8, packed for SDOT
// K: [N, D] INT8, packed for SDOT (will be transposed internally)
// V: [N, D] INT8, packed for SDOT
// O: [M, D] INT32 output (or FP32 if final)
void fused_attention_int8(
    const int8_t* Q,        // Query matrix [M, D]
    const int8_t* K,        // Key matrix [N, D]
    const int8_t* V,        // Value matrix [N, D]
    int32_t* O,             // Output matrix [M, D]
    const fused_attn_params_t* params
);

// Version with FP32 output
void fused_attention_int8_fp32(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O,
    const fused_attn_params_t* params
);

// Pack Q for Q@K^T: Q[M][D] -> Qp[M/6][D/4][6][4]
void pack_Q_fused(const int8_t* Q, int8_t* Qp, int M, int D);

// Pack K for Q@K^T (transposed): K[N][D] -> Kp[N/64][D/4][64][4]
void pack_K_fused(const int8_t* K, int8_t* Kp, int N, int D);

// Pack V for P@V: V[N][D] -> Vp[N/4][4][D]
void pack_V_fused(const int8_t* V, int8_t* Vp, int N, int D);

#endif // FUSED_ATTENTION_INT8_H
