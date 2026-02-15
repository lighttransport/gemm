// layernorm.h - INT8/INT16 LayerNorm and RMSNorm for A64FX
// LayerNorm: y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
// RMSNorm: y = x / sqrt(mean(x^2) + epsilon) * gamma

#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <stdint.h>
#include <stddef.h>

// ============================================================================
// INT8 LayerNorm
// ============================================================================

// INT8 LayerNorm with INT32 accumulation
// Input: x[N] in INT8
// Output: y[N] in INT8
// Gamma: gamma[N] in INT8 (scale factor)
// Beta: beta[N] in INT8 (bias)
// Scale: quantization scale for output
void layernorm_int8(
    const int8_t* input,    // [N] input tensor
    int8_t* output,         // [N] output tensor
    const int8_t* gamma,    // [N] scale weights
    const int8_t* beta,     // [N] bias
    int32_t epsilon,        // Small constant for numerical stability (Q8.24)
    float input_scale,      // Input quantization scale
    float output_scale,     // Output quantization scale
    size_t N                // Dimension
);

// INT8 LayerNorm without affine parameters (no gamma/beta)
void layernorm_int8_noaffine(
    const int8_t* input,
    int8_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N
);

// ============================================================================
// INT16 LayerNorm
// ============================================================================

// INT16 LayerNorm with INT32 accumulation
void layernorm_int16(
    const int16_t* input,   // [N] input tensor
    int16_t* output,        // [N] output tensor
    const int16_t* gamma,   // [N] scale weights
    const int16_t* beta,    // [N] bias
    int32_t epsilon,        // Small constant (Q16.16)
    float input_scale,
    float output_scale,
    size_t N
);

// INT16 LayerNorm without affine parameters
void layernorm_int16_noaffine(
    const int16_t* input,
    int16_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N
);

// ============================================================================
// INT8 RMSNorm
// ============================================================================

// INT8 RMSNorm with INT32 accumulation
// RMSNorm: y = x / RMS(x) * gamma, where RMS(x) = sqrt(mean(x^2) + epsilon)
void rmsnorm_int8(
    const int8_t* input,    // [N] input tensor
    int8_t* output,         // [N] output tensor
    const int8_t* gamma,    // [N] scale weights
    int32_t epsilon,        // Small constant (Q8.24)
    float input_scale,
    float output_scale,
    size_t N
);

// INT8 RMSNorm without affine parameters
void rmsnorm_int8_noaffine(
    const int8_t* input,
    int8_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N
);

// ============================================================================
// INT16 RMSNorm
// ============================================================================

// INT16 RMSNorm with INT32 accumulation
void rmsnorm_int16(
    const int16_t* input,   // [N] input tensor
    int16_t* output,        // [N] output tensor
    const int16_t* gamma,   // [N] scale weights
    int32_t epsilon,        // Small constant (Q16.16)
    float input_scale,
    float output_scale,
    size_t N
);

// INT16 RMSNorm without affine parameters
void rmsnorm_int16_noaffine(
    const int16_t* input,
    int16_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N
);

// ============================================================================
// Utility functions
// ============================================================================

// Fast inverse square root approximation using Newton-Raphson
// Input: x in INT32 (Q16.16 format)
// Output: 1/sqrt(x) in INT32 (Q16.16 format)
int32_t fast_invsqrt_int32(int32_t x);

// Integer square root
uint32_t isqrt_uint32(uint32_t x);

#endif // LAYERNORM_H
