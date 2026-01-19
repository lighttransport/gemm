// ffn_int8_6row.c - INT8 FFN with 6-row split loading
#include "ffn_int8_6row.h"
#include <stdlib.h>
#include <string.h>

// External kernel and activation functions
extern void kernel_ffn_6row_gemm_d512(const int8_t* A, const int8_t* B, int32_t* C);
extern void squared_relu_int32_sve(const int32_t* input, int32_t* output, size_t length, int scale_shift);
extern void shift_gelu_int32_sve(const int32_t* input, int32_t* output, size_t length, int shift);
extern void silu_int32_sve(const int32_t* input, int32_t* output, size_t length);

// Helper: quantize INT32 back to INT8
static void quantize_int32_to_int8(const int32_t* input, int8_t* output, size_t length, int scale_shift) {
    for (size_t i = 0; i < length; i++) {
        int32_t val = input[i] >> scale_shift;
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[i] = (int8_t)val;
    }
}

// Pack weight matrix from [K, N] to [K/4, N, 4] layout
static void pack_weights_kn_to_k4n4(const int8_t* W, int8_t* W_packed, int K, int N) {
    int K_groups = K / 4;
    for (int kg = 0; kg < K_groups; kg++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < 4; k++) {
                W_packed[(kg * N + n) * 4 + k] = W[(kg * 4 + k) * N + n];
            }
        }
    }
}

// FFN forward pass: out = activation(x @ W1) @ W2
// Specialized for D=512, D_ff=2048
void ffn_int8_forward_d512(
    const int8_t* input,      // [M, 512]
    const int8_t* W1,         // [512, 2048] (row-major)
    const int8_t* W2,         // [2048, 512] (row-major)
    int32_t* output,          // [M, 512]
    int M,
    activation_t act
) {
    const int D_in = 512;
    const int D_ff = 2048;
    const int D_out = 512;

    // Pack W1 and W2 for efficient GEMM
    int8_t* W1_packed = aligned_alloc(64, (D_in/4) * D_ff * 4);
    int8_t* W2_packed = aligned_alloc(64, (D_ff/4) * D_out * 4);
    pack_weights_kn_to_k4n4(W1, W1_packed, D_in, D_ff);
    pack_weights_kn_to_k4n4(W2, W2_packed, D_ff, D_out);

    // Allocate intermediate buffers
    int32_t* hidden_int32 = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* hidden_act = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int8_t* hidden_int8 = aligned_alloc(64, M * D_ff);

    // First GEMM: hidden = input @ W1 (process 6 rows at a time)
    int M_tiles = M / 6;
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d512(
            input + mt * 6 * D_in,
            W1_packed,
            hidden_int32 + mt * 6 * D_ff
        );
    }

    // Handle remainder rows if M % 6 != 0
    // (simplified: assume M is divisible by 6 for now)

    // Apply activation function
    switch (act) {
        case ACT_SQUARED_RELU:
            squared_relu_int32_sve(hidden_int32, hidden_act, M * D_ff, 8);
            break;
        case ACT_SHIFT_GELU:
            shift_gelu_int32_sve(hidden_int32, hidden_act, M * D_ff, 0);
            break;
        case ACT_SILU:
            silu_int32_sve(hidden_int32, hidden_act, M * D_ff);
            break;
        default:
            memcpy(hidden_act, hidden_int32, M * D_ff * sizeof(int32_t));
    }

    // Quantize back to INT8
    quantize_int32_to_int8(hidden_act, hidden_int8, M * D_ff, 8);

    // Second GEMM: output = hidden_int8 @ W2
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d512(
            hidden_int8 + mt * 6 * D_ff,
            W2_packed,
            output + mt * 6 * D_out
        );
    }

    // Cleanup
    free(W1_packed);
    free(W2_packed);
    free(hidden_int32);
    free(hidden_act);
    free(hidden_int8);
}

// Gated FFN: out = (act1(x @ W_gate) * act2(x @ W_up)) @ W_down
void ffn_int8_gated_d512(
    const int8_t* input,      // [M, 512]
    const int8_t* W_gate,     // [512, 2048]
    const int8_t* W_up,       // [512, 2048]
    const int8_t* W_down,     // [2048, 512]
    int32_t* output,          // [M, 512]
    int M,
    activation_t gate_act,
    activation_t up_act
) {
    const int D_in = 512;
    const int D_ff = 2048;
    const int D_out = 512;

    // Pack weights
    int8_t* W_gate_packed = aligned_alloc(64, (D_in/4) * D_ff * 4);
    int8_t* W_up_packed = aligned_alloc(64, (D_in/4) * D_ff * 4);
    int8_t* W_down_packed = aligned_alloc(64, (D_ff/4) * D_out * 4);
    pack_weights_kn_to_k4n4(W_gate, W_gate_packed, D_in, D_ff);
    pack_weights_kn_to_k4n4(W_up, W_up_packed, D_in, D_ff);
    pack_weights_kn_to_k4n4(W_down, W_down_packed, D_ff, D_out);

    // Allocate buffers
    int32_t* gate_int32 = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* up_int32 = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* gate_act_buf = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* up_act_buf = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* hidden_mul = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int8_t* hidden_int8 = aligned_alloc(64, M * D_ff);

    int M_tiles = M / 6;

    // Compute gate = input @ W_gate
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d512(
            input + mt * 6 * D_in,
            W_gate_packed,
            gate_int32 + mt * 6 * D_ff
        );
    }

    // Compute up = input @ W_up
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d512(
            input + mt * 6 * D_in,
            W_up_packed,
            up_int32 + mt * 6 * D_ff
        );
    }

    // Apply activations
    switch (gate_act) {
        case ACT_SQUARED_RELU:
            squared_relu_int32_sve(gate_int32, gate_act_buf, M * D_ff, 8);
            break;
        case ACT_SHIFT_GELU:
            shift_gelu_int32_sve(gate_int32, gate_act_buf, M * D_ff, 0);
            break;
        case ACT_SILU:
            silu_int32_sve(gate_int32, gate_act_buf, M * D_ff);
            break;
        default:
            memcpy(gate_act_buf, gate_int32, M * D_ff * sizeof(int32_t));
    }

    switch (up_act) {
        case ACT_SQUARED_RELU:
            squared_relu_int32_sve(up_int32, up_act_buf, M * D_ff, 8);
            break;
        case ACT_SHIFT_GELU:
            shift_gelu_int32_sve(up_int32, up_act_buf, M * D_ff, 0);
            break;
        case ACT_SILU:
            silu_int32_sve(up_int32, up_act_buf, M * D_ff);
            break;
        default:
            memcpy(up_act_buf, up_int32, M * D_ff * sizeof(int32_t));
    }

    // Element-wise multiply: hidden = gate * up
    for (size_t i = 0; i < (size_t)M * D_ff; i++) {
        int64_t product = (int64_t)gate_act_buf[i] * up_act_buf[i];
        hidden_mul[i] = (int32_t)(product >> 16);  // Scale down
    }

    // Quantize to INT8
    quantize_int32_to_int8(hidden_mul, hidden_int8, M * D_ff, 8);

    // Final GEMM: output = hidden @ W_down
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d512(
            hidden_int8 + mt * 6 * D_ff,
            W_down_packed,
            output + mt * 6 * D_out
        );
    }

    // Cleanup
    free(W_gate_packed);
    free(W_up_packed);
    free(W_down_packed);
    free(gate_int32);
    free(up_int32);
    free(gate_act_buf);
    free(up_act_buf);
    free(hidden_mul);
    free(hidden_int8);
}
