// ffn_int8_v4.h
// Complete INT8 FFN with multiple activation paths for A64FX SVE
//
// Supports:
// 1. FP32 activation path: int32 -> fp32 -> activation -> int8 (stochastic rounding)
// 2. Pure INT32 activation path: int32 -> int32 activation -> int8
// 3. Both SiLU (SwiGLU) and GELU activations
//
// Pipeline: x -> GEMM1(gate) + GEMM2(up) -> activation -> GEMM3(down) -> output

#ifndef FFN_INT8_V4_H
#define FFN_INT8_V4_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "activation_int32.h"

//=============================================================================
// Configuration
//=============================================================================

typedef enum {
    FFN_ACTIVATION_SILU,      // SiLU (Swish) - used in LLaMA, Qwen
    FFN_ACTIVATION_GELU,      // GELU - used in GPT, BERT
    FFN_ACTIVATION_SILU_INT32, // Pure INT32 SiLU
    FFN_ACTIVATION_GELU_INT32  // Pure INT32 GELU
} ffn_activation_t;

typedef enum {
    FFN_QUANT_DETERMINISTIC,   // Standard rounding
    FFN_QUANT_STOCHASTIC       // Stochastic rounding with Philox RNG
} ffn_quantization_t;

typedef struct {
    int hidden_dim;            // Model hidden dimension (e.g., 4096)
    int intermediate_dim;      // FFN intermediate dimension (e.g., 11008)
    ffn_activation_t activation;
    ffn_quantization_t quantization;
    uint64_t rng_seed;         // Seed for stochastic rounding
    float input_scale;         // Scale for input quantization
    float weight_scale;        // Scale for weight quantization
} ffn_config_t;

//=============================================================================
// Weight structure (pre-packed for GEMM)
//=============================================================================

typedef struct {
    int8_t* W_gate;            // [intermediate_dim, hidden_dim] packed
    int8_t* W_up;              // [intermediate_dim, hidden_dim] packed
    int8_t* W_down;            // [hidden_dim, intermediate_dim] packed
    float scale_gate;
    float scale_up;
    float scale_down;
    int hidden_dim;
    int intermediate_dim;
} ffn_weights_t;

//=============================================================================
// External GEMM kernel declarations
//=============================================================================

// 6x4 kernel: C[6][64] += A[6][K] @ B[K][64]
extern void kernel_6x4_kloop(const int8_t* A, const int8_t* B, int32_t* C,
                              int K, int ldc);

//=============================================================================
// Packing functions
//=============================================================================

// Pack A matrix for GEMM: [M, K] row-major -> [M_tiles, K, 6] for kernel
static inline void pack_A_mr6(const int8_t* A, int8_t* Apack, int M, int K, int lda) {
    int m = 0;
    for (; m + 5 < M; m += 6) {
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < 6; i++) {
                Apack[(m/6) * K * 6 + k * 6 + i] = A[(m + i) * lda + k];
            }
        }
    }
    // Handle remainder
    if (m < M) {
        int rem = M - m;
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < 6; i++) {
                Apack[(m/6) * K * 6 + k * 6 + i] = (i < rem) ? A[(m + i) * lda + k] : 0;
            }
        }
    }
}

// Pack B matrix for GEMM: [K, N] col-major -> [N_tiles, K, 64] for kernel
static inline void pack_B_nr64(const int8_t* B, int8_t* Bpack, int K, int N, int ldb) {
    int n = 0;
    for (; n + 63 < N; n += 64) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < 64; j++) {
                Bpack[(n/64) * K * 64 + k * 64 + j] = B[k * ldb + n + j];
            }
        }
    }
    // Handle remainder
    if (n < N) {
        int rem = N - n;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < 64; j++) {
                Bpack[(n/64) * K * 64 + k * 64 + j] = (j < rem) ? B[k * ldb + n + j] : 0;
            }
        }
    }
}

//=============================================================================
// FP32 Activation Path
//=============================================================================

// SwiGLU: SiLU(gate) * up
// Input: gate[n], up[n] in int32 (from GEMM)
// Output: out[n] in int8 (quantized for next GEMM)
static inline void swiglu_fp32_path(
    const int32_t* gate, const int32_t* up, int8_t* out, int n,
    float gemm_scale, float quant_scale,
    ffn_quantization_t quant_mode, uint64_t seed, uint32_t stream)
{
    // Temporary float buffer
    float* temp = (float*)aligned_alloc(64, n * sizeof(float));

    // Convert to float, apply SiLU, multiply
    int i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);

        // Load gate and up
        svint32_t vgate_i = svld1_s32(pg, gate + i);
        svint32_t vup_i = svld1_s32(pg, up + i);

        // Convert to float and scale
        svfloat32_t vgate = svcvt_f32_s32_x(pg, vgate_i);
        svfloat32_t vup = svcvt_f32_s32_x(pg, vup_i);
        vgate = svmul_f32_x(pg, vgate, svdup_f32(gemm_scale));
        vup = svmul_f32_x(pg, vup, svdup_f32(gemm_scale));

        // SiLU using rational approximation (fast)
        svfloat32_t abs_gate = svabs_f32_x(pg, vgate);
        svfloat32_t denom = svadd_f32_x(pg, svdup_f32(1.0f), abs_gate);
        svfloat32_t frac = svdiv_f32_x(pg, vgate, denom);
        svfloat32_t sig = svmla_f32_x(pg, svdup_f32(0.5f), svdup_f32(0.5f), frac);
        svfloat32_t silu = svmul_f32_x(pg, vgate, sig);

        // Multiply: SiLU(gate) * up
        svfloat32_t result = svmul_f32_x(pg, silu, vup);

        svst1_f32(pg, temp + i, result);
        i += svcntw();
    }

    // Quantize to int8
    if (quant_mode == FFN_QUANT_STOCHASTIC) {
        quantize_f32_to_i8_stochastic(temp, out, n, quant_scale, seed, stream);
    } else {
        // Deterministic rounding
        for (int j = 0; j < n; j++) {
            float x = temp[j] * quant_scale;
            float rounded = (x >= 0) ? floorf(x + 0.5f) : ceilf(x - 0.5f);
            if (rounded > 127.0f) rounded = 127.0f;
            if (rounded < -128.0f) rounded = -128.0f;
            out[j] = (int8_t)rounded;
        }
    }

    free(temp);
}

// GELU activation path
static inline void gelu_fp32_path(
    const int32_t* in, int8_t* out, int n,
    float gemm_scale, float quant_scale,
    ffn_quantization_t quant_mode, uint64_t seed, uint32_t stream)
{
    float* temp = (float*)aligned_alloc(64, n * sizeof(float));

    // Apply GELU
    gelu_i32_to_f32(in, temp, n, gemm_scale);

    // Quantize
    if (quant_mode == FFN_QUANT_STOCHASTIC) {
        quantize_f32_to_i8_stochastic(temp, out, n, quant_scale, seed, stream);
    } else {
        for (int j = 0; j < n; j++) {
            float x = temp[j] * quant_scale;
            float rounded = (x >= 0) ? floorf(x + 0.5f) : ceilf(x - 0.5f);
            if (rounded > 127.0f) rounded = 127.0f;
            if (rounded < -128.0f) rounded = -128.0f;
            out[j] = (int8_t)rounded;
        }
    }

    free(temp);
}

//=============================================================================
// Pure INT32 Activation Path
//=============================================================================

// SwiGLU in pure INT32
// Avoids float conversion entirely for maximum throughput
static inline void swiglu_int32_path(
    const int32_t* gate, const int32_t* up, int8_t* out, int n,
    float input_scale, float output_scale)
{
    // Convert scales to Q16 fixed-point
    float scale_ratio = output_scale / input_scale;
    int32_t scale_q16 = (int32_t)(scale_ratio * 65536.0f);

    for (int i = 0; i < n; i++) {
        // Scale input to Q16 range
        int64_t gate_scaled = ((int64_t)gate[i] * scale_q16) >> 16;
        int64_t up_scaled = ((int64_t)up[i] * scale_q16) >> 16;

        // Clamp to sigmoid range
        if (gate_scaled > 524288) gate_scaled = 524288;
        if (gate_scaled < -524288) gate_scaled = -524288;
        if (up_scaled > 524288) up_scaled = 524288;
        if (up_scaled < -524288) up_scaled = -524288;

        // SiLU(gate) in Q16
        int32_t silu = silu_q16_lut((int32_t)gate_scaled);

        // Multiply: SiLU(gate) * up
        int64_t product = ((int64_t)silu * up_scaled) >> 16;

        // Scale for int8 output
        int32_t result = (int32_t)(product >> 8);  // Approximate scaling

        // Clamp to int8
        if (result > 127) result = 127;
        if (result < -128) result = -128;

        out[i] = (int8_t)result;
    }
}

// GELU in pure INT32
static inline void gelu_int32_path(
    const int32_t* in, int8_t* out, int n,
    float input_scale, float output_scale)
{
    float scale_ratio = output_scale / input_scale;
    int32_t scale_q16 = (int32_t)(scale_ratio * 65536.0f);

    for (int i = 0; i < n; i++) {
        int64_t x_scaled = ((int64_t)in[i] * scale_q16) >> 16;

        if (x_scaled > 524288) x_scaled = 524288;
        if (x_scaled < -524288) x_scaled = -524288;

        int32_t gelu_result = gelu_q16((int32_t)x_scaled);

        int32_t result = (int32_t)(gelu_result >> 8);
        if (result > 127) result = 127;
        if (result < -128) result = -128;

        out[i] = (int8_t)result;
    }
}

//=============================================================================
// Main FFN Forward Function
//=============================================================================

// FFN forward pass
// x: [batch, hidden_dim] int8 input
// out: [batch, hidden_dim] int32 output (before final dequantization)
static inline void ffn_forward_v4(
    const int8_t* x, int32_t* out,
    int batch, const ffn_weights_t* weights,
    const ffn_config_t* config, uint32_t call_id)
{
    int H = weights->hidden_dim;
    int I = weights->intermediate_dim;

    // Allocate intermediate buffers
    int8_t* x_packed = (int8_t*)aligned_alloc(256, batch * H);
    int32_t* gate_out = (int32_t*)aligned_alloc(256, batch * I * sizeof(int32_t));
    int32_t* up_out = (int32_t*)aligned_alloc(256, batch * I * sizeof(int32_t));
    int8_t* hidden = (int8_t*)aligned_alloc(256, batch * I);

    // Pack input
    pack_A_mr6(x, x_packed, batch, H, H);

    // Zero output buffers
    memset(gate_out, 0, batch * I * sizeof(int32_t));
    memset(up_out, 0, batch * I * sizeof(int32_t));

    // GEMM1: x @ W_gate^T -> gate_out [batch, intermediate]
    // GEMM2: x @ W_up^T -> up_out [batch, intermediate]
    // (In practice, these would use tiled GEMM with the 6x4 kernel)

    int m_tiles = (batch + 5) / 6;
    int n_tiles = (I + 63) / 64;

    for (int mt = 0; mt < m_tiles; mt++) {
        int m_start = mt * 6;
        int m_end = (m_start + 6 < batch) ? m_start + 6 : batch;

        for (int nt = 0; nt < n_tiles; nt++) {
            int n_start = nt * 64;

            // Gate GEMM
            kernel_6x4_kloop(
                x_packed + mt * H * 6,
                weights->W_gate + nt * H * 64,
                gate_out + m_start * I + n_start,
                H, I * sizeof(int32_t));

            // Up GEMM
            kernel_6x4_kloop(
                x_packed + mt * H * 6,
                weights->W_up + nt * H * 64,
                up_out + m_start * I + n_start,
                H, I * sizeof(int32_t));
        }
    }

    // GEMM output scale
    float gemm_scale = config->input_scale * config->weight_scale;

    // Activation + Quantization
    switch (config->activation) {
        case FFN_ACTIVATION_SILU:
            for (int b = 0; b < batch; b++) {
                swiglu_fp32_path(
                    gate_out + b * I, up_out + b * I, hidden + b * I, I,
                    gemm_scale, 127.0f / (gemm_scale * 10.0f),  // Heuristic quant scale
                    config->quantization, config->rng_seed, call_id * batch + b);
            }
            break;

        case FFN_ACTIVATION_GELU:
            for (int b = 0; b < batch; b++) {
                gelu_fp32_path(
                    gate_out + b * I, hidden + b * I, I,
                    gemm_scale, 127.0f / (gemm_scale * 5.0f),
                    config->quantization, config->rng_seed, call_id * batch + b);
            }
            break;

        case FFN_ACTIVATION_SILU_INT32:
            for (int b = 0; b < batch; b++) {
                swiglu_int32_path(
                    gate_out + b * I, up_out + b * I, hidden + b * I, I,
                    gemm_scale, 127.0f / (gemm_scale * 10.0f));
            }
            break;

        case FFN_ACTIVATION_GELU_INT32:
            for (int b = 0; b < batch; b++) {
                gelu_int32_path(
                    gate_out + b * I, hidden + b * I, I,
                    gemm_scale, 127.0f / (gemm_scale * 5.0f));
            }
            break;
    }

    // Pack hidden for final GEMM
    int8_t* hidden_packed = (int8_t*)aligned_alloc(256, batch * I);
    pack_A_mr6(hidden, hidden_packed, batch, I, I);

    // Zero final output
    memset(out, 0, batch * H * sizeof(int32_t));

    // GEMM3: hidden @ W_down^T -> out [batch, hidden]
    m_tiles = (batch + 5) / 6;
    n_tiles = (H + 63) / 64;

    for (int mt = 0; mt < m_tiles; mt++) {
        int m_start = mt * 6;

        for (int nt = 0; nt < n_tiles; nt++) {
            int n_start = nt * 64;

            kernel_6x4_kloop(
                hidden_packed + mt * I * 6,
                weights->W_down + nt * I * 64,
                out + m_start * H + n_start,
                I, H * sizeof(int32_t));
        }
    }

    // Cleanup
    free(x_packed);
    free(gate_out);
    free(up_out);
    free(hidden);
    free(hidden_packed);
}

//=============================================================================
// Utility Functions
//=============================================================================

// Initialize FFN weights (allocate and pack)
static inline ffn_weights_t* ffn_weights_create(
    const int8_t* W_gate_raw, const int8_t* W_up_raw, const int8_t* W_down_raw,
    int hidden_dim, int intermediate_dim,
    float scale_gate, float scale_up, float scale_down)
{
    ffn_weights_t* w = (ffn_weights_t*)malloc(sizeof(ffn_weights_t));

    w->hidden_dim = hidden_dim;
    w->intermediate_dim = intermediate_dim;
    w->scale_gate = scale_gate;
    w->scale_up = scale_up;
    w->scale_down = scale_down;

    // Allocate packed weights
    int gate_size = ((intermediate_dim + 63) / 64) * hidden_dim * 64;
    int up_size = gate_size;
    int down_size = ((hidden_dim + 63) / 64) * intermediate_dim * 64;

    w->W_gate = (int8_t*)aligned_alloc(256, gate_size);
    w->W_up = (int8_t*)aligned_alloc(256, up_size);
    w->W_down = (int8_t*)aligned_alloc(256, down_size);

    // Pack weights
    pack_B_nr64(W_gate_raw, w->W_gate, hidden_dim, intermediate_dim, intermediate_dim);
    pack_B_nr64(W_up_raw, w->W_up, hidden_dim, intermediate_dim, intermediate_dim);
    pack_B_nr64(W_down_raw, w->W_down, intermediate_dim, hidden_dim, hidden_dim);

    return w;
}

// Free FFN weights
static inline void ffn_weights_free(ffn_weights_t* w) {
    if (w) {
        free(w->W_gate);
        free(w->W_up);
        free(w->W_down);
        free(w);
    }
}

// Initialize config with defaults
static inline ffn_config_t ffn_config_default(int hidden_dim, int intermediate_dim) {
    ffn_config_t cfg;
    cfg.hidden_dim = hidden_dim;
    cfg.intermediate_dim = intermediate_dim;
    cfg.activation = FFN_ACTIVATION_SILU;
    cfg.quantization = FFN_QUANT_DETERMINISTIC;
    cfg.rng_seed = 0x12345678ULL;
    cfg.input_scale = 1.0f / 127.0f;
    cfg.weight_scale = 1.0f / 127.0f;
    return cfg;
}

#endif // FFN_INT8_V4_H
