// fused_attention_int8.c
// Fused INT8 Attention implementation

#include "fused_attention_int8.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_sve.h>

// External optimized kernels
// Q@K^T kernel: computes 6x64 tile of attention scores
// Output is 6 rows x 64 cols = 6 rows x 4 vectors
extern void kernel_qkt_6x4_2x(
    const int8_t* Q,    // Packed Q [D/4][6][4]
    const int8_t* K,    // Packed K [D/4][64][4]
    int32_t* S,         // Output scores [6][64]
    int64_t D,          // Head dimension
    int64_t S_stride    // Output stride in bytes
);

// P@V kernel: computes 6xD tile of output
extern void kernel_pv_int8_opt(
    const int8_t* P,    // Packed P [K/4][6][4]
    const int8_t* V,    // Packed V [K/4][4][D]
    int32_t* O,         // Output [6][D]
    int64_t D,
    int64_t O_stride    // Output stride in bytes
);

// Pack Q for Q@K^T: Q[M][D] -> Qp[M/6][D/4][6][4]
void pack_Q_fused(const int8_t* Q, int8_t* Qp, int M, int D) {
    int M_tiles = M / 6;
    for (int mt = 0; mt < M_tiles; mt++) {
        for (int dg = 0; dg < D / 4; dg++) {
            for (int m = 0; m < 6; m++) {
                for (int d = 0; d < 4; d++) {
                    Qp[mt * (D/4) * 6 * 4 + dg * 6 * 4 + m * 4 + d] =
                        Q[(mt * 6 + m) * D + dg * 4 + d];
                }
            }
        }
    }
}

// Pack K for Q@K^T: K[N][D] -> Kp[N/64][D/4][64][4]
void pack_K_fused(const int8_t* K, int8_t* Kp, int N, int D) {
    int N_chunks = N / 64;
    for (int nc = 0; nc < N_chunks; nc++) {
        for (int dg = 0; dg < D / 4; dg++) {
            for (int n = 0; n < 64; n++) {
                for (int d = 0; d < 4; d++) {
                    Kp[nc * (D/4) * 64 * 4 + dg * 64 * 4 + n * 4 + d] =
                        K[(nc * 64 + n) * D + dg * 4 + d];
                }
            }
        }
    }
}

// Pack V for P@V: V[N][D] -> Vp[D_tile][kg][vec][d16][k4] interleaved for SDOT
// SDOT needs bytes [d*4+k] = V[k][d] within each 64-byte vector
// Each K-group (4 K elements) has 4 vectors of 64 bytes covering 64 D elements
// For D>64: V is laid out as [D_tile][K_group][4][64] - each D tile contiguous
void pack_V_fused(const int8_t* V, int8_t* Vp, int N, int D) {
    int K_groups = N / 4;
    int D_tiles = (D + 63) / 64;
    int V_tile_stride = K_groups * 256;  // 4096 bytes for K=64

    for (int dt = 0; dt < D_tiles; dt++) {
        int8_t* Vp_tile = Vp + dt * V_tile_stride;
        int d_base = dt * 64;

        for (int kg = 0; kg < K_groups; kg++) {
            for (int vec = 0; vec < 4; vec++) {       // 4 vectors per K-group
                for (int d16 = 0; d16 < 16; d16++) {  // 16 D elements per vector
                    for (int k4 = 0; k4 < 4; k4++) {  // 4 K elements interleaved
                        int d = d_base + vec * 16 + d16;
                        int k = kg * 4 + k4;
                        int8_t val = (d < D) ? V[k * D + d] : 0;
                        Vp_tile[kg * 256 + vec * 64 + d16 * 4 + k4] = val;
                    }
                }
            }
        }
    }
}

// Pack P (attention weights) for P@V: P[6][64] -> Pp[16][6][4]
// P is already computed as 6 rows x 64 cols
static void pack_P_tile(const int8_t* P, int8_t* Pp) {
    // P[6][64] -> Pp[K/4][6][4] where K=64
    for (int kg = 0; kg < 16; kg++) {  // 64/4 = 16 K groups
        for (int m = 0; m < 6; m++) {
            for (int k = 0; k < 4; k++) {
                Pp[kg * 6 * 4 + m * 4 + k] = P[m * 64 + kg * 4 + k];
            }
        }
    }
}

// Online softmax for a single row
// Updates running max and sum, returns rescale factor
static inline float online_softmax_update(
    const int32_t* scores,  // Input scores (INT32)
    int n,                   // Number of scores
    float scale,            // Combined scale (qk_scale * attn_scale)
    float* row_max,         // Running max (updated)
    float* row_sum,         // Running sum (updated)
    float* softmax_out      // Output softmax values (FP32)
) {
    // Find max in this chunk
    float chunk_max = -1e30f;
    for (int i = 0; i < n; i++) {
        float val = (float)scores[i] * scale;
        if (val > chunk_max) chunk_max = val;
    }

    // Update running max
    float old_max = *row_max;
    float new_max = (chunk_max > old_max) ? chunk_max : old_max;

    // Compute rescale factor for previous sum
    float rescale = expf(old_max - new_max);

    // Compute exp and sum for this chunk
    float chunk_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float val = (float)scores[i] * scale;
        float exp_val = expf(val - new_max);
        softmax_out[i] = exp_val;
        chunk_sum += exp_val;
    }

    // Update running sum
    *row_sum = (*row_sum) * rescale + chunk_sum;
    *row_max = new_max;

    return rescale;
}

// Quantize softmax output to INT8
static inline void quantize_softmax(
    const float* softmax,   // FP32 softmax values
    int8_t* out,            // INT8 output
    int n,
    float row_sum,          // Sum for normalization
    float scale             // Quantization scale
) {
    float inv_sum = 1.0f / row_sum;
    for (int i = 0; i < n; i++) {
        float normalized = softmax[i] * inv_sum;
        int32_t quantized = (int32_t)roundf(normalized * scale);
        // Clamp to INT8 range
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        out[i] = (int8_t)quantized;
    }
}

// Main fused attention function
void fused_attention_int8(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    int32_t* O,
    const fused_attn_params_t* params
) {
    int M = params->M;
    int N = params->N;
    int D = params->D;
    float combined_scale = params->qk_scale * params->scale;
    float p_scale = params->p_scale;

    int M_tiles = M / 6;
    int N_chunks = N / 64;

    // Temporary buffers
    int32_t* S_tile = aligned_alloc(64, 6 * 64 * sizeof(int32_t));  // Attention scores
    float* softmax_buf = aligned_alloc(64, 6 * 64 * sizeof(float)); // Softmax output
    int8_t* P_tile = aligned_alloc(64, 6 * 64);                      // Quantized attention
    int8_t* Pp_tile = aligned_alloc(64, 16 * 6 * 4);                 // Packed P for P@V
    int32_t* O_tile = aligned_alloc(64, 6 * D * sizeof(int32_t));   // Tile output

    // Online softmax state per row
    float row_max[6];
    float row_sum[6];
    float* O_fp32 = aligned_alloc(64, 6 * D * sizeof(float));  // FP32 accumulator for rescaling

    for (int mt = 0; mt < M_tiles; mt++) {
        // Initialize output accumulator and softmax state
        memset(O_fp32, 0, 6 * D * sizeof(float));
        for (int r = 0; r < 6; r++) {
            row_max[r] = -1e30f;
            row_sum[r] = 0.0f;
        }

        const int8_t* Q_tile = Q + mt * (D/4) * 6 * 4;

        for (int nc = 0; nc < N_chunks; nc++) {
            const int8_t* K_chunk = K + nc * (D/4) * 64 * 4;
            const int8_t* V_chunk = V + nc * 16 * 4 * D;  // V packed as [N/4][4][D]

            // Stage 1: Q @ K^T -> S[6][64]
            kernel_qkt_6x4_2x(Q_tile, K_chunk, S_tile, D, 64 * sizeof(int32_t));

            // Online softmax per row
            float rescale[6];
            for (int r = 0; r < 6; r++) {
                rescale[r] = online_softmax_update(
                    S_tile + r * 64, 64,
                    combined_scale,
                    &row_max[r], &row_sum[r],
                    softmax_buf + r * 64
                );

                // Rescale previous O accumulator
                for (int d = 0; d < D; d++) {
                    O_fp32[r * D + d] *= rescale[r];
                }

                // Quantize softmax to INT8 (without final normalization yet)
                // We'll normalize at the end
                for (int i = 0; i < 64; i++) {
                    float val = softmax_buf[r * 64 + i];
                    int32_t q = (int32_t)roundf(val * p_scale);
                    if (q > 127) q = 127;
                    if (q < 0) q = 0;  // Softmax values are positive
                    P_tile[r * 64 + i] = (int8_t)q;
                }
            }

            // Pack P for P@V kernel
            pack_P_tile(P_tile, Pp_tile);

            // Stage 2: P @ V -> O_tile[6][D]
            memset(O_tile, 0, 6 * D * sizeof(int32_t));
            kernel_pv_int8_opt(Pp_tile, V_chunk, O_tile, D, D * sizeof(int32_t));

            // Accumulate to FP32 (with dequantization)
            float v_scale = params->v_scale;
            float pv_scale = 1.0f / (p_scale * v_scale);
            for (int r = 0; r < 6; r++) {
                for (int d = 0; d < D; d++) {
                    O_fp32[r * D + d] += (float)O_tile[r * D + d] * pv_scale;
                }
            }
        }

        // Final normalization and output
        for (int r = 0; r < 6; r++) {
            float inv_sum = 1.0f / row_sum[r];
            for (int d = 0; d < D; d++) {
                // Normalize and quantize to INT32 output
                float val = O_fp32[r * D + d] * inv_sum;
                O[(mt * 6 + r) * D + d] = (int32_t)roundf(val * params->o_scale);
            }
        }
    }

    free(S_tile);
    free(softmax_buf);
    free(P_tile);
    free(Pp_tile);
    free(O_tile);
    free(O_fp32);
}

// Version with FP32 output
void fused_attention_int8_fp32(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O,
    const fused_attn_params_t* params
) {
    int M = params->M;
    int N = params->N;
    int D = params->D;
    float combined_scale = params->qk_scale * params->scale;
    float p_scale = params->p_scale;

    int M_tiles = M / 6;
    int N_chunks = N / 64;

    // Temporary buffers
    int32_t* S_tile = aligned_alloc(64, 6 * 64 * sizeof(int32_t));
    float* softmax_buf = aligned_alloc(64, 6 * 64 * sizeof(float));
    int8_t* P_tile = aligned_alloc(64, 6 * 64);
    int8_t* Pp_tile = aligned_alloc(64, 16 * 6 * 4);
    int32_t* O_tile = aligned_alloc(64, 6 * D * sizeof(int32_t));

    float row_max[6];
    float row_sum[6];
    float* O_fp32 = aligned_alloc(64, 6 * D * sizeof(float));

    for (int mt = 0; mt < M_tiles; mt++) {
        memset(O_fp32, 0, 6 * D * sizeof(float));
        for (int r = 0; r < 6; r++) {
            row_max[r] = -1e30f;
            row_sum[r] = 0.0f;
        }

        const int8_t* Q_tile = Q + mt * (D/4) * 6 * 4;

        for (int nc = 0; nc < N_chunks; nc++) {
            const int8_t* K_chunk = K + nc * (D/4) * 64 * 4;
            const int8_t* V_chunk = V + nc * 16 * 4 * D;

            // Stage 1: Q @ K^T
            kernel_qkt_6x4_2x(Q_tile, K_chunk, S_tile, D, 64 * sizeof(int32_t));

            // Online softmax
            float rescale[6];
            for (int r = 0; r < 6; r++) {
                rescale[r] = online_softmax_update(
                    S_tile + r * 64, 64,
                    combined_scale,
                    &row_max[r], &row_sum[r],
                    softmax_buf + r * 64
                );

                for (int d = 0; d < D; d++) {
                    O_fp32[r * D + d] *= rescale[r];
                }

                for (int i = 0; i < 64; i++) {
                    float val = softmax_buf[r * 64 + i];
                    int32_t q = (int32_t)roundf(val * p_scale);
                    if (q > 127) q = 127;
                    if (q < 0) q = 0;
                    P_tile[r * 64 + i] = (int8_t)q;
                }
            }

            pack_P_tile(P_tile, Pp_tile);

            // Stage 2: P @ V
            memset(O_tile, 0, 6 * D * sizeof(int32_t));
            kernel_pv_int8_opt(Pp_tile, V_chunk, O_tile, D, D * sizeof(int32_t));

            float v_scale = params->v_scale;
            float pv_scale = 1.0f / (p_scale * v_scale);
            for (int r = 0; r < 6; r++) {
                for (int d = 0; d < D; d++) {
                    O_fp32[r * D + d] += (float)O_tile[r * D + d] * pv_scale;
                }
            }
        }

        // Final normalization
        for (int r = 0; r < 6; r++) {
            float inv_sum = 1.0f / row_sum[r];
            for (int d = 0; d < D; d++) {
                O[(mt * 6 + r) * D + d] = O_fp32[r * D + d] * inv_sum;
            }
        }
    }

    free(S_tile);
    free(softmax_buf);
    free(P_tile);
    free(Pp_tile);
    free(O_tile);
    free(O_fp32);
}
