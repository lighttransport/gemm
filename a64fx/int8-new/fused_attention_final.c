// fused_attention_final.c
// Final optimized fused attention with SVE softmax and fast pack

#include "fused_attention_int8.h"
#include "softmax_sve.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void kernel_qkt_6x4_2x(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t);
extern void kernel_pv_int8_opt(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t);

typedef struct {
    int8_t* Qp;
    int8_t* Kp;
    int8_t* Vp;
    int32_t* S_tile;
    int8_t* P_tile;
    int8_t* Pp_tile;
    int32_t* O_tile;
    float* O_fp32;
    int M, N, D;
} final_attn_workspace_t;

// Pack functions
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

// FAST pack_P using 32-bit loads (39x faster than byte-by-byte)
static inline void pack_P_fast(const int8_t* P, int8_t* Pp) {
    for (int kg = 0; kg < 16; kg++) {
        int8_t* out = Pp + kg * 24;
        int offset = kg * 4;
        *(uint32_t*)(out + 0) = *(uint32_t*)(P + 0 * 64 + offset);
        *(uint32_t*)(out + 4) = *(uint32_t*)(P + 1 * 64 + offset);
        *(uint32_t*)(out + 8) = *(uint32_t*)(P + 2 * 64 + offset);
        *(uint32_t*)(out + 12) = *(uint32_t*)(P + 3 * 64 + offset);
        *(uint32_t*)(out + 16) = *(uint32_t*)(P + 4 * 64 + offset);
        *(uint32_t*)(out + 20) = *(uint32_t*)(P + 5 * 64 + offset);
    }
}

final_attn_workspace_t* final_attn_alloc(int M, int N, int D) {
    final_attn_workspace_t* ws = malloc(sizeof(final_attn_workspace_t));
    int M_tiles = (M + 5) / 6;
    int N_chunks = (N + 63) / 64;
    int D_groups = D / 4;
    int D_tiles = (D + 63) / 64;
    
    ws->Qp = aligned_alloc(64, M_tiles * D_groups * 24);
    ws->Kp = aligned_alloc(64, N_chunks * D_groups * 256);
    ws->Vp = aligned_alloc(64, N_chunks * D_tiles * 16 * 256);
    ws->S_tile = aligned_alloc(64, 6 * 64 * sizeof(int32_t));
    ws->P_tile = aligned_alloc(64, 6 * 64);
    ws->Pp_tile = aligned_alloc(64, 16 * 24);
    ws->O_tile = aligned_alloc(64, 6 * D * sizeof(int32_t));
    ws->O_fp32 = aligned_alloc(64, 6 * D * sizeof(float));
    ws->M = M; ws->N = N; ws->D = D;
    return ws;
}

void final_attn_free(final_attn_workspace_t* ws) {
    free(ws->Qp); free(ws->Kp); free(ws->Vp);
    free(ws->S_tile); free(ws->P_tile); free(ws->Pp_tile);
    free(ws->O_tile); free(ws->O_fp32); free(ws);
}

void final_attn_pack(const int8_t* Q, const int8_t* K, const int8_t* V,
                     final_attn_workspace_t* ws) {
    pack_Q_opt(Q, ws->Qp, ws->M, ws->D);
    pack_K_opt(K, ws->Kp, ws->N, ws->D);
    pack_V_opt(V, ws->Vp, ws->N, ws->D);
}

void fused_attention_final(float* O, final_attn_workspace_t* ws, float scale) {
    int M = ws->M, N = ws->N, D = ws->D;
    int M_tiles = M / 6;
    int N_chunks = N / 64;
    int D_groups = D / 4;
    int D_tiles = (D + 63) / 64;
    
    for (int mt = 0; mt < M_tiles; mt++) {
        memset(ws->O_fp32, 0, 6 * D * sizeof(float));
        const int8_t* Q_tile = ws->Qp + mt * D_groups * 24;
        
        softmax_state_t state[6];
        softmax_state_init(state, 6);
        
        for (int nc = 0; nc < N_chunks; nc++) {
            const int8_t* K_chunk = ws->Kp + nc * D_groups * 256;
            const int8_t* V_chunk = ws->Vp + nc * D_tiles * 16 * 256;
            
            memset(ws->S_tile, 0, 6 * 64 * sizeof(int32_t));
            kernel_qkt_6x4_2x(Q_tile, K_chunk, ws->S_tile, D, 64 * sizeof(int32_t));
            
            float max_exp[6];
            softmax_tile_sve(ws->S_tile, scale, state, ws->O_fp32, D,
                            ws->P_tile, max_exp);
            
            pack_P_fast(ws->P_tile, ws->Pp_tile);  // FAST pack!
            
            memset(ws->O_tile, 0, 6 * D * sizeof(int32_t));
            kernel_pv_int8_opt(ws->Pp_tile, V_chunk, ws->O_tile, D, D * sizeof(int32_t));
            
            for (int r = 0; r < 6; r++) {
                float dequant = max_exp[r] / 127.0f;
                for (int d = 0; d < D; d++) {
                    ws->O_fp32[r * D + d] += (float)ws->O_tile[r * D + d] * dequant;
                }
            }
        }
        
        for (int r = 0; r < 6; r++) {
            float inv_sum = 1.0f / state[r].sum;
            for (int d = 0; d < D; d++) {
                O[(mt * 6 + r) * D + d] = ws->O_fp32[r * D + d] * inv_sum;
            }
        }
    }
}
