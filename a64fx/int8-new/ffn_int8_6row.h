// ffn_int8_6row.h - INT8 FFN with 6-row split loading optimization
#ifndef FFN_INT8_6ROW_H
#define FFN_INT8_6ROW_H

#include <stdint.h>

// Activation functions
typedef enum {
    ACT_SQUARED_RELU,  // max(0, x)^2
    ACT_SHIFT_GELU,    // Approximate GELU with shift
    ACT_SILU           // Sigmoid Linear Unit (SiLU/Swish)
} activation_t;

// FFN forward pass with 6-row processing
// Input: [M, D_in] int8
// W1: [D_in, D_ff] int8 (packed for efficiency)
// W2: [D_ff, D_out] int8 (packed for efficiency)
// Output: [M, D_out] int32
// Computes: out = activation(x @ W1) @ W2
void ffn_int8_6row_forward(
    const int8_t* input,      // [M, D_in]
    const int8_t* W1,         // [D_in, D_ff] packed
    const int8_t* W2,         // [D_ff, D_out] packed
    int32_t* output,          // [M, D_out]
    int M,
    int D_in,
    int D_ff,
    int D_out,
    activation_t act
);

// Gated FFN: out = (act1(x @ W_gate) * act2(x @ W_up)) @ W_down
void ffn_int8_6row_gated(
    const int8_t* input,      // [M, D_in]
    const int8_t* W_gate,     // [D_in, D_ff] packed
    const int8_t* W_up,       // [D_in, D_ff] packed
    const int8_t* W_down,     // [D_ff, D_out] packed
    int32_t* output,          // [M, D_out]
    int M,
    int D_in,
    int D_ff,
    int D_out,
    activation_t gate_act,
    activation_t up_act
);

#endif // FFN_INT8_6ROW_H
