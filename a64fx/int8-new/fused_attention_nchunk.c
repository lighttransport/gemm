// fused_attention_nchunk.c
// N-chunk major fused attention for better cache reuse
//
// Key insight: By processing N-chunk major, K and V data stays in cache
// while we iterate through all M tiles. This reduces memory traffic significantly.
//
// Current (M-tile major): K/V loaded M_tiles times each
// New (N-chunk major): K/V loaded once, Q loaded N_chunks times

#include "fused_attention_int8.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// External optimized kernels
extern void kernel_qkt_6x4_2x(
    const int8_t* Q, const int8_t* K, int32_t* S,
    int64_t D, int64_t S_stride);

extern void kernel_pv_int8_opt(
    const int8_t* P, const int8_t* V, int32_t* O,
    int64_t D, int64_t O_stride);

// =============================================================================
// N-chunk major workspace (larger to hold all M tiles' state)
// =============================================================================
typedef struct {
    int8_t* Qp;           // Packed Q
    int8_t* Kp;           // Packed K  
    int8_t* Vp;           // Packed V
    int32_t* S_tile;      // Attention scores [6][64]
    float* softmax_buf;   // Softmax values [6][64]
    int8_t* P_tile;       // Quantized P [6][64]
    int8_t* Pp_tile;      // Packed P for kernel
    int32_t* O_tile;      // P@V output tile [6][D]
    
    // Per M-tile state (allocated for ALL M tiles)
    float* O_all;         // FP32 accumulators [M_tiles][6][D]
    float* row_max_all;   // Running max [M_tiles][6]
    float* row_sum_all;   // Running sum [M_tiles][6]
    
    size_t Qp_size;
    size_t Kp_size;
    size_t Vp_size;
    int M, N, D;
    int M_tiles;
} nchunk_workspace_t;

// =============================================================================
// Allocate workspace
// =============================================================================
nchunk_workspace_t* nchunk_alloc_workspace(int M, int N, int D) {
    nchunk_workspace_t* ws = malloc(sizeof(nchunk_workspace_t));

    int M_tiles = (M + 5) / 6;
    int N_chunks = (N + 63) / 64;
    int D_groups = D / 4;
    int D_tiles = (D + 63) / 64;

    ws->Qp_size = M_tiles * D_groups * 24;
    ws->Kp_size = N_chunks * D_groups * 256;
    ws->Vp_size = N_chunks * D_tiles * 16 * 256;

    ws->Qp = aligned_alloc(64, ws->Qp_size);
    ws->Kp = aligned_alloc(64, ws->Kp_size);
    ws->Vp = aligned_alloc(64, ws->Vp_size);
    ws->S_tile = aligned_alloc(64, 6 * 64 * sizeof(int32_t));
    ws->softmax_buf = aligned_alloc(64, 6 * 64 * sizeof(float));
    ws->P_tile = aligned_alloc(64, 6 * 64);
    ws->Pp_tile = aligned_alloc(64, 16 * 24);
    ws->O_tile = aligned_alloc(64, 6 * D * sizeof(int32_t));

    // Allocate state for ALL M tiles
    ws->O_all = aligned_alloc(64, M_tiles * 6 * D * sizeof(float));
    ws->row_max_all = aligned_alloc(64, M_tiles * 6 * sizeof(float));
    ws->row_sum_all = aligned_alloc(64, M_tiles * 6 * sizeof(float));

    ws->M = M;
    ws->N = N;
    ws->D = D;
    ws->M_tiles = M_tiles;

    return ws;
}

// =============================================================================
// Free workspace
// =============================================================================
void nchunk_free_workspace(nchunk_workspace_t* ws) {
    free(ws->Qp);
    free(ws->Kp);
    free(ws->Vp);
    free(ws->S_tile);
    free(ws->softmax_buf);
    free(ws->P_tile);
    free(ws->Pp_tile);
    free(ws->O_tile);
    free(ws->O_all);
    free(ws->row_max_all);
    free(ws->row_sum_all);
    free(ws);
}

// =============================================================================
// Pack functions (same as before)
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

static void pack_V_opt(const int8_t* V, int8_t* Vp, int N, int D) {
    int N_chunks = N / 64;
    int D_tiles = (D + 63) / 64;
    int chunk_stride = D_tiles * 16 * 256;

    for (int nc = 0; nc < N_chunks; nc++) {
        int8_t* Vp_chunk = Vp + nc * chunk_stride;
        int k_base = nc * 64;

        for (int dt = 0; dt < D_tiles; dt++) {
            int8_t* Vp_tile = Vp_chunk + dt * 16 * 256;
            int d_base = dt * 64;

            for (int kg = 0; kg < 16; kg++) {
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
// Pack inputs
// =============================================================================
void nchunk_pack_inputs(
    const int8_t* Q, const int8_t* K, const int8_t* V,
    nchunk_workspace_t* ws)
{
    pack_Q_opt(Q, ws->Qp, ws->M, ws->D);
    pack_K_opt(K, ws->Kp, ws->N, ws->D);
    pack_V_opt(V, ws->Vp, ws->N, ws->D);
}

// =============================================================================
// N-chunk major fused attention
// =============================================================================
void fused_attention_nchunk(
    float* O,
    nchunk_workspace_t* ws,
    float scale)
{
    int M = ws->M;
    int N = ws->N;
    int D = ws->D;
    int M_tiles = ws->M_tiles;
    int N_chunks = N / 64;
    int D_groups = D / 4;
    int D_tiles = (D + 63) / 64;

    // Initialize all M tiles' state
    for (int mt = 0; mt < M_tiles; mt++) {
        for (int r = 0; r < 6; r++) {
            ws->row_max_all[mt * 6 + r] = -1e30f;
            ws->row_sum_all[mt * 6 + r] = 0.0f;
        }
        memset(ws->O_all + mt * 6 * D, 0, 6 * D * sizeof(float));
    }

    // N-chunk major loop: K/V stay in cache for all M tiles
    for (int nc = 0; nc < N_chunks; nc++) {
        const int8_t* K_chunk = ws->Kp + nc * D_groups * 256;
        const int8_t* V_chunk = ws->Vp + nc * D_tiles * 16 * 256;

        // Process all M tiles with this K/V chunk in cache
        for (int mt = 0; mt < M_tiles; mt++) {
            const int8_t* Q_tile = ws->Qp + mt * D_groups * 24;
            float* O_mt = ws->O_all + mt * 6 * D;
            float* row_max = ws->row_max_all + mt * 6;
            float* row_sum = ws->row_sum_all + mt * 6;

            // Stage 1: Q @ K^T -> S[6][64]
            memset(ws->S_tile, 0, 6 * 64 * sizeof(int32_t));
            kernel_qkt_6x4_2x(Q_tile, K_chunk, ws->S_tile, D, 64 * sizeof(int32_t));

            // Stage 2: Online softmax per row
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

                // Rescale previous O accumulator
                float rescale = expf(old_max - new_max);
                for (int d = 0; d < D; d++) {
                    O_mt[r * D + d] *= rescale;
                }

                // Compute exp and find max_exp
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

                // Quantize to INT8
                if (chunk_max_exp > 0) {
                    float inv_max = 127.0f / chunk_max_exp;
                    for (int i = 0; i < 64; i++) {
                        int32_t p = (int32_t)(ws->softmax_buf[r * 64 + i] * inv_max + 0.5f);
                        if (p > 127) p = 127;
                        ws->P_tile[r * 64 + i] = (int8_t)p;
                    }
                } else {
                    memset(ws->P_tile + r * 64, 0, 64);
                }
            }

            // Pack P for kernel
            pack_P_opt(ws->P_tile, ws->Pp_tile, 64);

            // Stage 3: P @ V -> O_tile[6][D]
            memset(ws->O_tile, 0, 6 * D * sizeof(int32_t));
            kernel_pv_int8_opt(ws->Pp_tile, V_chunk, ws->O_tile, D, D * sizeof(int32_t));

            // Accumulate to FP32
            for (int r = 0; r < 6; r++) {
                float dequant = max_exp[r] / 127.0f;
                for (int d = 0; d < D; d++) {
                    O_mt[r * D + d] += (float)ws->O_tile[r * D + d] * dequant;
                }
            }
        }
    }

    // Final normalization for all M tiles
    for (int mt = 0; mt < M_tiles; mt++) {
        float* O_mt = ws->O_all + mt * 6 * D;
        float* row_sum = ws->row_sum_all + mt * 6;
        
        for (int r = 0; r < 6; r++) {
            float inv_sum = 1.0f / row_sum[r];
            for (int d = 0; d < D; d++) {
                O[(mt * 6 + r) * D + d] = O_mt[r * D + d] * inv_sum;
            }
        }
    }
}
