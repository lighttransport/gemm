// ffn_qwen3.c - Qwen3-next style SwiGLU FFN implementation
#include "ffn_qwen3.h"
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

// External kernels
extern void kernel_ffn_6row_gemm_d256(const int8_t* A, const int8_t* B, int32_t* C);
extern void kernel_ffn_6row_gemm_d512(const int8_t* A, const int8_t* B, int32_t* C);

// External activations
extern void silu_int32_sve(const int32_t* input, int32_t* output, size_t length);

// Configuration table
static const qwen3_ffn_config configs[] = {
    {256, 1024, 4.0f, "Qwen3-256 (4x expansion)"},
    {512, 2048, 4.0f, "Qwen3-512 (4x expansion)"},
    {896, 4864, 5.43f, "Qwen3-0.5B"},
    {1536, 8960, 5.83f, "Qwen3-1.5B"},
    {2048, 10240, 5.0f, "Qwen3-3B+"}
};

const qwen3_ffn_config* qwen3_get_config(qwen3_config_t config) {
    if (config >= 0 && config < 5) {
        return &configs[config];
    }
    return NULL;
}

// Helper: pack weights
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

// Helper: quantize INT32 to INT8
static void quantize_int32_to_int8(const int32_t* input, int8_t* output, size_t length, int scale_shift) {
    for (size_t i = 0; i < length; i++) {
        int32_t val = input[i] >> scale_shift;
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[i] = (int8_t)val;
    }
}

// Qwen3 SwiGLU for D=256
// FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
void qwen3_ffn_forward_d256(
    const int8_t* input,
    const int8_t* W_gate,
    const int8_t* W_up,
    const int8_t* W_down,
    int32_t* output,
    int M
) {
    const int D = 256;
    const int D_ff = 1024;

    // Pack weights
    int8_t* W_gate_packed = aligned_alloc(64, (D/4) * D_ff * 4);
    int8_t* W_up_packed = aligned_alloc(64, (D/4) * D_ff * 4);
    int8_t* W_down_packed = aligned_alloc(64, (D_ff/4) * D * 4);
    pack_weights_kn_to_k4n4(W_gate, W_gate_packed, D, D_ff);
    pack_weights_kn_to_k4n4(W_up, W_up_packed, D, D_ff);
    pack_weights_kn_to_k4n4(W_down, W_down_packed, D_ff, D);

    // Allocate intermediate buffers
    int32_t* gate_int32 = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* up_int32 = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* gate_silu = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* hidden_mul = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int8_t* hidden_int8 = aligned_alloc(64, M * D_ff);

    int M_tiles = M / 6;

    // Compute gate = input @ W_gate
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d256(
            input + mt * 6 * D,
            W_gate_packed,
            gate_int32 + mt * 6 * D_ff
        );
    }

    // Compute up = input @ W_up
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d256(
            input + mt * 6 * D,
            W_up_packed,
            up_int32 + mt * 6 * D_ff
        );
    }

    // Apply SiLU to gate
    silu_int32_sve(gate_int32, gate_silu, M * D_ff);

    // Element-wise multiply: hidden = SiLU(gate) * up
    for (size_t i = 0; i < (size_t)M * D_ff; i++) {
        int64_t product = ((int64_t)gate_silu[i] * up_int32[i]) >> 16;
        hidden_mul[i] = (int32_t)product;
    }

    // Quantize to INT8
    quantize_int32_to_int8(hidden_mul, hidden_int8, M * D_ff, 8);

    // Final GEMM: output = hidden @ W_down
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d256(
            hidden_int8 + mt * 6 * D_ff,
            W_down_packed,
            output + mt * 6 * D
        );
    }

    // Cleanup
    free(W_gate_packed);
    free(W_up_packed);
    free(W_down_packed);
    free(gate_int32);
    free(up_int32);
    free(gate_silu);
    free(hidden_mul);
    free(hidden_int8);
}

// Qwen3 SwiGLU for D=512
void qwen3_ffn_forward_d512(
    const int8_t* input,
    const int8_t* W_gate,
    const int8_t* W_up,
    const int8_t* W_down,
    int32_t* output,
    int M
) {
    const int D = 512;
    const int D_ff = 2048;

    // Pack weights
    int8_t* W_gate_packed = aligned_alloc(64, (D/4) * D_ff * 4);
    int8_t* W_up_packed = aligned_alloc(64, (D/4) * D_ff * 4);
    int8_t* W_down_packed = aligned_alloc(64, (D_ff/4) * D * 4);
    pack_weights_kn_to_k4n4(W_gate, W_gate_packed, D, D_ff);
    pack_weights_kn_to_k4n4(W_up, W_up_packed, D, D_ff);
    pack_weights_kn_to_k4n4(W_down, W_down_packed, D_ff, D);

    // Allocate intermediate buffers
    int32_t* gate_int32 = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* up_int32 = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* gate_silu = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* hidden_mul = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int8_t* hidden_int8 = aligned_alloc(64, M * D_ff);

    int M_tiles = M / 6;

    // Compute gate = input @ W_gate
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d512(
            input + mt * 6 * D,
            W_gate_packed,
            gate_int32 + mt * 6 * D_ff
        );
    }

    // Compute up = input @ W_up
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d512(
            input + mt * 6 * D,
            W_up_packed,
            up_int32 + mt * 6 * D_ff
        );
    }

    // Apply SiLU to gate
    silu_int32_sve(gate_int32, gate_silu, M * D_ff);

    // Element-wise multiply: hidden = SiLU(gate) * up
    for (size_t i = 0; i < (size_t)M * D_ff; i++) {
        int64_t product = ((int64_t)gate_silu[i] * up_int32[i]) >> 16;
        hidden_mul[i] = (int32_t)product;
    }

    // Quantize to INT8
    quantize_int32_to_int8(hidden_mul, hidden_int8, M * D_ff, 8);

    // Final GEMM: output = hidden @ W_down
    for (int mt = 0; mt < M_tiles; mt++) {
        kernel_ffn_6row_gemm_d512(
            hidden_int8 + mt * 6 * D_ff,
            W_down_packed,
            output + mt * 6 * D
        );
    }

    // Cleanup
    free(W_gate_packed);
    free(W_up_packed);
    free(W_down_packed);
    free(gate_int32);
    free(up_int32);
    free(gate_silu);
    free(hidden_mul);
    free(hidden_int8);
}

// Generic implementation (for other sizes)
void qwen3_ffn_forward_generic(
    const int8_t* input,
    const int8_t* W_gate,
    const int8_t* W_up,
    const int8_t* W_down,
    int32_t* output,
    int M,
    int D_model,
    int D_ff
) {
    // Simple reference implementation (not optimized)
    // For production, implement optimized kernels for each size

    // Allocate buffers
    int32_t* gate_int32 = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* up_int32 = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* gate_silu = aligned_alloc(64, M * D_ff * sizeof(int32_t));
    int32_t* hidden_mul = aligned_alloc(64, M * D_ff * sizeof(int32_t));

    // Naive GEMM for gate
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < D_ff; n++) {
            int32_t sum = 0;
            for (int k = 0; k < D_model; k++) {
                sum += (int32_t)input[m * D_model + k] * W_gate[k * D_ff + n];
            }
            gate_int32[m * D_ff + n] = sum;
        }
    }

    // Naive GEMM for up
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < D_ff; n++) {
            int32_t sum = 0;
            for (int k = 0; k < D_model; k++) {
                sum += (int32_t)input[m * D_model + k] * W_up[k * D_ff + n];
            }
            up_int32[m * D_ff + n] = sum;
        }
    }

    // Apply SiLU to gate
    silu_int32_sve(gate_int32, gate_silu, M * D_ff);

    // Element-wise multiply
    for (size_t i = 0; i < (size_t)M * D_ff; i++) {
        int64_t product = ((int64_t)gate_silu[i] * up_int32[i]) >> 16;
        hidden_mul[i] = (int32_t)product;
    }

    // Quantize and final GEMM for down
    int8_t* hidden_int8 = aligned_alloc(64, M * D_ff);
    quantize_int32_to_int8(hidden_mul, hidden_int8, M * D_ff, 8);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < D_model; n++) {
            int32_t sum = 0;
            for (int k = 0; k < D_ff; k++) {
                sum += (int32_t)hidden_int8[m * D_ff + k] * W_down[k * D_model + n];
            }
            output[m * D_model + n] = sum;
        }
    }

    free(gate_int32);
    free(up_int32);
    free(gate_silu);
    free(hidden_mul);
    free(hidden_int8);
}
