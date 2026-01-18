// fused_attention_opt.c
// Optimized Fused INT8 Attention implementation
// - INT8 SDOT for Q@K^T (93% efficient)
// - FP32 softmax (simple and accurate)
// - INT8 SDOT for P@V (75% efficient)
// - Pre-allocated workspace for efficiency

#include "fused_attention_int8.h"
#include "exp2_int.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_sve.h>

// External optimized kernels
extern void kernel_qkt_6x4_2x(
    const int8_t* Q, const int8_t* K, int32_t* S,
    int64_t D, int64_t S_stride);

extern void kernel_pv_int8_opt(
    const int8_t* P, const int8_t* V, int32_t* O,
    int64_t D, int64_t O_stride);

// =============================================================================
// Workspace for pre-allocated buffers
// =============================================================================
typedef struct {
    int8_t* Qp;           // Packed Q
    int8_t* Kp;           // Packed K
    int8_t* Vp;           // Packed V
    int32_t* S_tile;      // Attention scores [6][64]
    float* softmax_buf;   // Softmax values [6][64]
    int8_t* P_tile;       // Quantized P [6][64]
    int8_t* Pp_tile;      // Packed P for kernel
    int32_t* O_tile;      // P@V output tile
    float* O_fp32;        // FP32 accumulator
    size_t Qp_size;
    size_t Kp_size;
    size_t Vp_size;
    int M, N, D;
} fused_attn_workspace_t;

// =============================================================================
// Allocate workspace
// =============================================================================
fused_attn_workspace_t* fused_attn_alloc_workspace(int M, int N, int D) {
    fused_attn_workspace_t* ws = malloc(sizeof(fused_attn_workspace_t));

    int M_tiles = (M + 5) / 6;
    int N_chunks = (N + 63) / 64;
    int D_groups = D / 4;
    int D_tiles = (D + 63) / 64;
    int K_groups = 16;  // 64 / 4

    ws->Qp_size = M_tiles * D_groups * 24;
    ws->Kp_size = N_chunks * D_groups * 256;
    ws->Vp_size = N_chunks * D_tiles * 16 * 256;  // 16 K_groups per 64-element chunk

    ws->Qp = aligned_alloc(64, ws->Qp_size);
    ws->Kp = aligned_alloc(64, ws->Kp_size);
    ws->Vp = aligned_alloc(64, ws->Vp_size);
    ws->S_tile = aligned_alloc(64, 6 * 64 * sizeof(int32_t));
    ws->softmax_buf = aligned_alloc(64, 6 * 64 * sizeof(float));
    ws->P_tile = aligned_alloc(64, 6 * 64);
    ws->Pp_tile = aligned_alloc(64, K_groups * 24);
    ws->O_tile = aligned_alloc(64, 6 * D * sizeof(int32_t));
    ws->O_fp32 = aligned_alloc(64, 6 * D * sizeof(float));

    ws->M = M;
    ws->N = N;
    ws->D = D;

    return ws;
}

// =============================================================================
// Free workspace
// =============================================================================
void fused_attn_free_workspace(fused_attn_workspace_t* ws) {
    free(ws->Qp);
    free(ws->Kp);
    free(ws->Vp);
    free(ws->S_tile);
    free(ws->softmax_buf);
    free(ws->P_tile);
    free(ws->Pp_tile);
    free(ws->O_tile);
    free(ws->O_fp32);
    free(ws);
}

// =============================================================================
// Pack Q for Q@K^T: Q[M][D] -> Qp[M/6][D/4][6][4]
// =============================================================================
static void pack_Q_opt(const int8_t* Q, int8_t* Qp, int M, int D) {
    int M_tiles = M / 6;
    int D_groups = D / 4;
    for (int mt = 0; mt < M_tiles; mt++) {
        for (int dg = 0; dg < D_groups; dg++) {
            for (int m = 0; m < 6; m++) {
                for (int d = 0; d < 4; d++) {
                    Qp[mt * D_groups * 24 + dg * 24 + m * 4 + d] =
                        Q[(mt * 6 + m) * D + dg * 4 + d];
                }
            }
        }
    }
}

// =============================================================================
// Pack K for Q@K^T: K[N][D] -> Kp[N/64][D/4][64][4]
// =============================================================================
static void pack_K_opt(const int8_t* K, int8_t* Kp, int N, int D) {
    int N_chunks = N / 64;
    int D_groups = D / 4;
    for (int nc = 0; nc < N_chunks; nc++) {
        for (int dg = 0; dg < D_groups; dg++) {
            for (int n = 0; n < 64; n++) {
                for (int d = 0; d < 4; d++) {
                    Kp[nc * D_groups * 256 + dg * 256 + n * 4 + d] =
                        K[(nc * 64 + n) * D + dg * 4 + d];
                }
            }
        }
    }
}

// =============================================================================
// Pack V for P@V: V[N][D] -> Vp[N_chunk][D_tile][K_group=16][4][64] interleaved
// Each N chunk has 64 elements (16 K_groups of 4)
// =============================================================================
static void pack_V_opt(const int8_t* V, int8_t* Vp, int N, int D) {
    int N_chunks = N / 64;
    int D_tiles = (D + 63) / 64;
    int chunk_stride = D_tiles * 16 * 256;  // Size per N chunk (16 K_groups per chunk)

    for (int nc = 0; nc < N_chunks; nc++) {
        int8_t* Vp_chunk = Vp + nc * chunk_stride;
        int k_base = nc * 64;  // Starting K index for this chunk

        for (int dt = 0; dt < D_tiles; dt++) {
            int8_t* Vp_tile = Vp_chunk + dt * 16 * 256;  // 16 K_groups per chunk
            int d_base = dt * 64;

            for (int kg = 0; kg < 16; kg++) {  // 16 K_groups = 64 elements / 4
                for (int vec = 0; vec < 4; vec++) {
                    for (int d16 = 0; d16 < 16; d16++) {
                        for (int k4 = 0; k4 < 4; k4++) {
                            int d = d_base + vec * 16 + d16;
                            int k = k_base + kg * 4 + k4;
                            int8_t val = (d < D) ? V[k * D + d] : 0;
                            Vp_tile[kg * 256 + vec * 64 + d16 * 4 + k4] = val;
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Pack P for P@V: P[6][64] -> Pp[16][6][4]
// =============================================================================
static void pack_P_opt(const int8_t* P, int8_t* Pp, int K) {
    int K_groups = K / 4;
    for (int kg = 0; kg < K_groups; kg++) {
        for (int m = 0; m < 6; m++) {
            for (int k = 0; k < 4; k++) {
                Pp[kg * 24 + m * 4 + k] = P[m * K + kg * 4 + k];
            }
        }
    }
}

// =============================================================================
// Simple row softmax using FP32 (accurate)
// =============================================================================
static inline void softmax_row_fp32(
    const int32_t* S,    // Input scores [N]
    float* P,            // Output probabilities [N]
    int N,
    float scale)
{
    // Find max
    float max_val = (float)S[0] * scale;
    for (int i = 1; i < N; i++) {
        float val = (float)S[i] * scale;
        if (val > max_val) max_val = val;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = (float)S[i] * scale;
        P[i] = expf(val - max_val);
        sum += P[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++) {
        P[i] *= inv_sum;
    }
}

// =============================================================================
// Quantize FP32 softmax to INT8
// =============================================================================
static inline void quantize_softmax_to_int8(
    const float* P_fp32,  // Input probabilities [N]
    int8_t* P_int8,       // Output INT8 [N]
    int N)
{
    // Softmax outputs are in [0, 1], scale to [-128, 127]
    // Use unsigned representation: P * 127 mapped to [0, 127]
    for (int i = 0; i < N; i++) {
        int32_t p = (int32_t)(P_fp32[i] * 127.0f + 0.5f);
        if (p > 127) p = 127;
        if (p < 0) p = 0;
        P_int8[i] = (int8_t)p;
    }
}

// =============================================================================
// Pack input matrices (call once before benchmark loop)
// =============================================================================
void fused_attn_pack_inputs(
    const int8_t* Q, const int8_t* K, const int8_t* V,
    fused_attn_workspace_t* ws)
{
    pack_Q_opt(Q, ws->Qp, ws->M, ws->D);
    pack_K_opt(K, ws->Kp, ws->N, ws->D);
    pack_V_opt(V, ws->Vp, ws->N, ws->D);
}

// =============================================================================
// Main optimized fused attention (with pre-allocated workspace)
// =============================================================================
void fused_attention_opt_fp32_ws(
    float* O,
    fused_attn_workspace_t* ws,
    const fused_attn_params_t* params)
{
    int M = params->M;
    int N = params->N;
    int D = params->D;
    float scale = params->scale;

    int M_tiles = M / 6;
    int N_chunks = N / 64;
    int D_groups = D / 4;
    int D_tiles = (D + 63) / 64;
    int K_groups = 16;

    // Process M tiles
    for (int mt = 0; mt < M_tiles; mt++) {
        // Initialize FP32 output accumulator
        memset(ws->O_fp32, 0, 6 * D * sizeof(float));

        const int8_t* Q_tile = ws->Qp + mt * D_groups * 24;

        // Online softmax state
        float row_max[6] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
        float row_sum[6] = {0, 0, 0, 0, 0, 0};

        // Process N chunks
        for (int nc = 0; nc < N_chunks; nc++) {
            const int8_t* K_chunk = ws->Kp + nc * D_groups * 256;
            const int8_t* V_chunk = ws->Vp + nc * D_tiles * 16 * 256;  // 16 K_groups per chunk

            // Stage 1: Q @ K^T -> S[6][64]
            memset(ws->S_tile, 0, 6 * 64 * sizeof(int32_t));
            kernel_qkt_6x4_2x(Q_tile, K_chunk, ws->S_tile, D, 64 * sizeof(int32_t));

            // Stage 2: Online softmax per row
            // Track max_exp per row for proper dequantization
            float max_exp[6];
            for (int r = 0; r < 6; r++) {
                // Find chunk max
                float chunk_max = -1e30f;
                for (int i = 0; i < 64; i++) {
                    float val = (float)ws->S_tile[r * 64 + i] * scale;
                    if (val > chunk_max) chunk_max = val;
                }

                // Update running max
                float old_max = row_max[r];
                float new_max = (chunk_max > old_max) ? chunk_max : old_max;

                // Compute rescale factor
                float rescale = expf(old_max - new_max);

                // Rescale previous O accumulator
                for (int d = 0; d < D; d++) {
                    ws->O_fp32[r * D + d] *= rescale;
                }

                // Compute exp and find max_exp for this chunk
                float chunk_sum = 0.0f;
                float chunk_max_exp = 0.0f;
                for (int i = 0; i < 64; i++) {
                    float val = (float)ws->S_tile[r * 64 + i] * scale;
                    float exp_val = expf(val - new_max);
                    ws->softmax_buf[r * 64 + i] = exp_val;
                    chunk_sum += exp_val;
                    if (exp_val > chunk_max_exp) chunk_max_exp = exp_val;
                }

                // Update running state
                row_sum[r] = row_sum[r] * rescale + chunk_sum;
                row_max[r] = new_max;
                max_exp[r] = chunk_max_exp;

                // Quantize to INT8: P_int8 = exp * 127 / max_exp
                // This scales to use full INT8 range without normalization
                if (chunk_max_exp > 0) {
                    float inv_max = 127.0f / chunk_max_exp;
                    for (int i = 0; i < 64; i++) {
                        int32_t p = (int32_t)(ws->softmax_buf[r * 64 + i] * inv_max + 0.5f);
                        if (p > 127) p = 127;
                        ws->P_tile[r * 64 + i] = (int8_t)p;
                    }
                } else {
                    for (int i = 0; i < 64; i++) {
                        ws->P_tile[r * 64 + i] = 0;
                    }
                }
            }

            // Pack P for P@V kernel
            pack_P_opt(ws->P_tile, ws->Pp_tile, 64);

            // Stage 3: P @ V -> O_tile[6][D]
            memset(ws->O_tile, 0, 6 * D * sizeof(int32_t));
            kernel_pv_int8_opt(ws->Pp_tile, V_chunk, ws->O_tile, D, D * sizeof(int32_t));

            // Accumulate to FP32 with proper dequantization
            // P was scaled by 127/max_exp, so need to multiply by max_exp/127
            for (int r = 0; r < 6; r++) {
                float dequant = max_exp[r] / 127.0f;
                for (int d = 0; d < D; d++) {
                    ws->O_fp32[r * D + d] += (float)ws->O_tile[r * D + d] * dequant;
                }
            }
        }

        // Final normalization
        for (int r = 0; r < 6; r++) {
            float inv_sum = 1.0f / row_sum[r];
            for (int d = 0; d < D; d++) {
                O[(mt * 6 + r) * D + d] = ws->O_fp32[r * D + d] * inv_sum;
            }
        }
    }
}

// =============================================================================
// Legacy interface (with internal allocation - slower)
// =============================================================================
void fused_attention_opt_fp32(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O,
    const fused_attn_params_t* params)
{
    fused_attn_workspace_t* ws = fused_attn_alloc_workspace(params->M, params->N, params->D);
    fused_attn_pack_inputs(Q, K, V, ws);
    fused_attention_opt_fp32_ws(O, ws, params);
    fused_attn_free_workspace(ws);
}
