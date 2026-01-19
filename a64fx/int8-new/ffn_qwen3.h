// ffn_qwen3.h - Qwen3-next style FFN configurations
#ifndef FFN_QWEN3_H
#define FFN_QWEN3_H

#include <stdint.h>

// Qwen3-next uses SwiGLU activation
// FFN(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
//
// Where ⊙ is element-wise multiplication

typedef enum {
    QWEN3_CONFIG_256,   // D=256, D_ff=1024 (4x)
    QWEN3_CONFIG_512,   // D=512, D_ff=2048 (4x)
    QWEN3_CONFIG_896,   // D=896, D_ff=4864 (~5.4x, Qwen3-0.5B)
    QWEN3_CONFIG_1536,  // D=1536, D_ff=8960 (~5.8x, Qwen3-1.5B)
    QWEN3_CONFIG_2048   // D=2048, D_ff=10240 (5x, Qwen3-3B+)
} qwen3_config_t;

typedef struct {
    int D_model;        // Hidden dimension
    int D_ff;           // FFN expansion dimension
    float expansion;    // Expansion ratio
    const char* name;
} qwen3_ffn_config;

// Get configuration details
const qwen3_ffn_config* qwen3_get_config(qwen3_config_t config);

// Qwen3-style SwiGLU FFN
// out = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
void qwen3_ffn_forward_d256(
    const int8_t* input,      // [M, 256]
    const int8_t* W_gate,     // [256, 1024]
    const int8_t* W_up,       // [256, 1024]
    const int8_t* W_down,     // [1024, 256]
    int32_t* output,          // [M, 256]
    int M
);

void qwen3_ffn_forward_d512(
    const int8_t* input,      // [M, 512]
    const int8_t* W_gate,     // [512, 2048]
    const int8_t* W_up,       // [512, 2048]
    const int8_t* W_down,     // [2048, 512]
    int32_t* output,          // [M, 512]
    int M
);

// Generic SwiGLU for any dimension (slower, for testing)
void qwen3_ffn_forward_generic(
    const int8_t* input,
    const int8_t* W_gate,
    const int8_t* W_up,
    const int8_t* W_down,
    int32_t* output,
    int M,
    int D_model,
    int D_ff
);

#endif // FFN_QWEN3_H
