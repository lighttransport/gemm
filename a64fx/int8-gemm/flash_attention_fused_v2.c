#include "flash_attention_fused_v2.h"
#include "gqa_pack.h"
#include "exp2_int.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// External microkernel declarations
extern void micro_kernel_6x4_vec_init(
    const int8_t* A, const int8_t* Bp, int32_t* C_vec,
    int64_t K, int64_t lda, int64_t C_stride);

extern void micro_kernel_4x4_vec_init(
    const int8_t* A, const int8_t* Bp, int32_t* C_vec,
    int64_t K, int64_t lda, int64_t C_stride);

extern void micro_kernel_2x4_vec_init(
    const int8_t* A, const int8_t* Bp, int32_t* C_vec,
    int64_t K, int64_t lda, int64_t C_stride);

// =============================================================================
// Memory allocation
// =============================================================================

void* flash_aligned_alloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return NULL;
    }
    return ptr;
}

void flash_aligned_free(void* ptr) {
    free(ptr);
}

// =============================================================================
// K Packing (reuse existing gqa_pack_b)
// =============================================================================

void pack_k_for_flash_attention(
    const int8_t* K,
    int8_t* Kp,
    int64_t L,
    int64_t head_dim)
{
    // K is [L, head_dim], needs to be accessed as K^T for Q@K^T
    // Pack using existing gqa_pack_b
    gqa_pack_b(K, Kp, L, head_dim, head_dim);
}

// =============================================================================
// V Packing (transpose for column access in P@V)
// =============================================================================

void pack_v_for_flash_attention(
    const int8_t* V,
    int8_t* Vp,
    int64_t L,
    int64_t head_dim)
{
    // V is [L, head_dim], needs to be accessed column-wise in P@V
    // Pack as [head_dim/64][L/64][L/4][4][64]

    int64_t head_tiles = (head_dim + 63) >> 6;
    int64_t L_tiles = (L + 63) >> 6;
    int64_t L_groups = (L + 3) >> 2;

    for (int64_t ht = 0; ht < head_tiles; ht++) {
        for (int64_t lt = 0; lt < L_tiles; lt++) {
            for (int64_t lg = 0; lg < L_groups; lg++) {
                int8_t* Vp_block = Vp +
                    (ht * L_tiles * L_groups + lt * L_groups + lg) * 256;

                // Pack 4 consecutive L positions × 64 head_dim elements
                for (int k = 0; k < 4; k++) {
                    int64_t l_idx = (lt << 6) + (lg << 2) + k;
                    for (int d = 0; d < 64; d++) {
                        int64_t h_idx = (ht << 6) + d;
                        int8_t val = (l_idx < L && h_idx < head_dim) ?
                            V[l_idx * head_dim + h_idx] : 0;
                        Vp_block[k * 64 + d] = val;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Step 1: Q@K^T GEMM tile function
// =============================================================================

static void gemm_qk_tile(
    const int8_t* Q_tile,
    const int8_t* Kp_chunk,
    int32_t* S_chunk,
    int64_t Br,
    int64_t Bc,
    int64_t K)
{
    int64_t K_groups = (K + 3) >> 2;
    int64_t Bc_tiles = (Bc + 63) >> 6;
    int64_t S_stride = Bc_tiles * 4 * 16;  // Row stride in elements

    // Process 64-column chunks
    for (int64_t j = 0; j < Bc; j += 64) {
        int64_t j_tile = j >> 6;
        int64_t bc_chunk = (j + 64 <= Bc) ? 64 : (Bc - j);

        const int8_t* Kp_col = Kp_chunk + j_tile * (K_groups << 8);
        int32_t* S_col = S_chunk + j_tile * 64;

        // Process 6-row chunks
        int64_t i = 0;
        for (; i + 6 <= Br; i += 6) {
            micro_kernel_6x4_vec_init(
                Q_tile + i * K,
                Kp_col,
                S_col + i * S_stride,
                K, K, S_stride << 2);
        }

        // Tail: 4-row kernel
        if (i + 4 <= Br) {
            micro_kernel_4x4_vec_init(
                Q_tile + i * K,
                Kp_col,
                S_col + i * S_stride,
                K, K, S_stride << 2);
            i += 4;
        }

        // Tail: 2-row kernel
        if (i + 2 <= Br) {
            micro_kernel_2x4_vec_init(
                Q_tile + i * K,
                Kp_col,
                S_col + i * S_stride,
                K, K, S_stride << 2);
        }
    }
}

// =============================================================================
// Helper: Convert C_vec format to flat row-major
// =============================================================================

static void cvec_to_flat(
    const int32_t* S_chunk_vec,
    int32_t* S_chunk_flat,
    int64_t Br,
    int64_t Bc)
{
    int64_t Bc_tiles = (Bc + 63) >> 6;

    for (int64_t i = 0; i < Br; i++) {
        for (int64_t jt = 0; jt < Bc_tiles; jt++) {
            const int32_t* src = S_chunk_vec + i * Bc_tiles * 64 + jt * 64;
            int32_t* dst = S_chunk_flat + i * Bc + jt * 64;

            // Copy 64 elements (4 vectors × 16 lanes)
            for (int k = 0; k < 64 && (jt * 64 + k) < Bc; k++) {
                dst[k] = src[k];
            }
        }
    }
}

// =============================================================================
// Step 2: Online softmax with FP32 conversion
// =============================================================================

static void online_softmax_fused(
    int32_t* S_chunk,
    float* P_chunk,
    flash_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    int32_t score_scale)
{
    int32_t combined_scale = (score_scale * LOG2E_Q8) >> 8;

    for (int64_t i = 0; i < Br; i++) {
        int32_t* S_row = S_chunk + i * Bc;
        float* P_row = P_chunk + i * Bc;

        // 1. Find tile max
        int32_t tile_max = S_row[0];
        for (int64_t j = 1; j < Bc; j++) {
            if (S_row[j] > tile_max) tile_max = S_row[j];
        }
        int32_t tile_max_q8 = (int32_t)(((int64_t)tile_max * combined_scale) >> 16);

        // 2. Update running max
        int32_t prev_max = state[i].m;
        int32_t new_max = (tile_max_q8 > prev_max) ? tile_max_q8 : prev_max;

        // 3. Compute rescale factor
        int32_t rescale;
        if (prev_max <= -8388608) {  // First tile (prev_max = -inf)
            rescale = 65536;  // 1.0 in Q16.16
        } else {
            int32_t diff = prev_max - new_max;
            rescale = exp2_int32(diff);
        }

        state[i].l = (int64_t)(((int64_t)state[i].l * rescale) >> 16);
        state[i].rescale = rescale;
        state[i].m = new_max;

        // 4. Compute exp + convert to FP32
        int64_t tile_sum = 0;
        for (int64_t j = 0; j < Bc; j++) {
            // Scale score
            int32_t x_q8 = (int32_t)(((int64_t)S_row[j] * combined_scale) >> 16);

            // Subtract max
            int32_t diff = x_q8 - new_max;

            // Compute exp2
            int32_t exp_q16 = exp2_int32(diff);

            // Convert to FP32
            P_row[j] = (float)exp_q16 / 65536.0f;

            // Accumulate sum
            tile_sum += exp_q16;
        }
        state[i].l += tile_sum;
    }
}

// =============================================================================
// Step 3: P@V accumulation
// =============================================================================

static void accumulate_pv_fused(
    float* O_tile,
    const float* P_chunk,
    const int8_t* Vp_chunk,
    const flash_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    int64_t head_dim)
{
    int64_t head_tiles = (head_dim + 63) >> 6;
    int64_t Bc_tiles = (Bc + 63) >> 6;
    int64_t Bc_groups = (Bc + 3) >> 2;

    // 1. Rescale O_tile
    for (int64_t i = 0; i < Br; i++) {
        float rescale_fp32 = (float)state[i].rescale / 65536.0f;
        for (int64_t d = 0; d < head_dim; d++) {
            O_tile[i * head_dim + d] *= rescale_fp32;
        }
    }

    // 2. O += P @ V
    for (int64_t ht = 0; ht < head_tiles; ht++) {
        for (int64_t i = 0; i < Br; i++) {
            float* O_row = O_tile + i * head_dim + (ht << 6);
            const float* P_row = P_chunk + i * Bc;

            // Inner loop: 64 output dims × Bc inputs
            for (int64_t k = 0; k < Bc; k++) {
                float p_val = P_row[k];

                // Locate V[k, ht*64:(ht+1)*64] in packed format
                int64_t kt = k >> 6;
                int64_t kg = (k & 63) >> 2;
                int64_t koffset = k & 3;

                const int8_t* V_col = Vp_chunk +
                    (ht * Bc_tiles * Bc_groups + kt * Bc_groups + kg) * 256 +
                    koffset;

                // Accumulate 64 elements
                for (int64_t d = 0; d < 64 && (ht * 64 + d) < head_dim; d++) {
                    int8_t v_val = V_col[d * 4];  // 4-strided access
                    O_row[d] += p_val * (float)v_val;
                }
            }
        }
    }
}

// =============================================================================
// Final normalization
// =============================================================================

static void normalize_output(
    float* O_tile,
    const flash_softmax_state_t* state,
    float* logsumexp_tile,
    int64_t Br,
    int64_t head_dim)
{
    for (int64_t i = 0; i < Br; i++) {
        // Convert running sum from Q16.16 to FP32
        float sum_fp32 = (float)state[i].l / 65536.0f;

        if (logsumexp_tile != NULL) {
            // logsumexp = max + log(sum)
            float max_fp32 = (float)state[i].m / 256.0f;  // Q8.8 → FP32
            logsumexp_tile[i] = max_fp32 + logf(sum_fp32);
        }

        // Normalize: O /= sum
        float inv_sum = 1.0f / sum_fp32;
        for (int64_t d = 0; d < head_dim; d++) {
            O_tile[i * head_dim + d] *= inv_sum;
        }
    }
}

// =============================================================================
// Main forward pass (with pre-packed K/V)
// =============================================================================

void flash_attention_fused_forward_packed(
    const int8_t* Q,
    const int8_t* Kp,
    const int8_t* Vp,
    float* O,
    float* logsumexp,
    int64_t L,
    int64_t head_dim)
{
    // Select score scale based on head_dim
    int32_t score_scale = (head_dim == 128) ? SCORE_SCALE_128 : SCORE_SCALE_256;

    // Allocate workspace for tiles (thread-local in parallel version)
    size_t S_chunk_size = FA_TILE_BR * FA_TILE_BC * sizeof(int32_t);
    size_t P_chunk_size = FA_TILE_BR * FA_TILE_BC * sizeof(float);
    size_t O_tile_size = FA_TILE_BR * head_dim * sizeof(float);
    size_t state_size = FA_TILE_BR * sizeof(flash_softmax_state_t);

    int32_t* S_chunk = (int32_t*)flash_aligned_alloc(S_chunk_size);
    float* P_chunk = (float*)flash_aligned_alloc(P_chunk_size);
    float* O_tile = (float*)flash_aligned_alloc(O_tile_size);
    flash_softmax_state_t* state = (flash_softmax_state_t*)flash_aligned_alloc(state_size);

    // Process Q in TILE_BR-row chunks
    for (int64_t q_tile = 0; q_tile < L; q_tile += FA_TILE_BR) {
        int64_t Br = ((q_tile + FA_TILE_BR) <= L) ? FA_TILE_BR : (L - q_tile);
        const int8_t* Q_tile = Q + q_tile * head_dim;
        float* O_out = O + q_tile * head_dim;
        float* lse_out = (logsumexp != NULL) ? (logsumexp + q_tile) : NULL;

        // Initialize O_tile and state
        memset(O_tile, 0, Br * head_dim * sizeof(float));
        for (int64_t i = 0; i < Br; i++) {
            state[i].m = -8388608;  // -inf in Q8.8
            state[i].l = 0;
            state[i].rescale = 65536;  // 1.0 in Q16.16
        }

        // Stream through K/V in TILE_BC-column chunks
        for (int64_t k_tile = 0; k_tile < L; k_tile += FA_TILE_BC) {
            int64_t Bc = ((k_tile + FA_TILE_BC) <= L) ? FA_TILE_BC : (L - k_tile);

            int64_t kt_start = k_tile >> 6;
            int64_t K_groups = (head_dim + 3) >> 2;
            const int8_t* Kp_chunk = Kp + kt_start * (K_groups << 8);

            int64_t ht_count = (head_dim + 63) >> 6;
            int64_t Bc_groups = (Bc + 3) >> 2;
            const int8_t* Vp_chunk = Vp + 0;  // Will offset inside accumulate_pv

            // === FUSED KERNEL ===
            // Step 1: Q@K^T GEMM
            gemm_qk_tile(Q_tile, Kp_chunk, S_chunk, Br, Bc, head_dim);

            // Convert C_vec format to flat for softmax
            int32_t* S_flat = S_chunk;  // Reuse same buffer
            // Note: S_chunk is already in a usable format (row-major with padding)

            // Step 2: Online softmax
            online_softmax_fused(S_chunk, P_chunk, state, Br, Bc, score_scale);

            // Step 3: P@V accumulation
            accumulate_pv_fused(O_tile, P_chunk, Vp_chunk, state, Br, Bc, head_dim);
        }

        // Final normalization
        normalize_output(O_tile, state, lse_out, Br, head_dim);

        // Copy to output
        memcpy(O_out, O_tile, Br * head_dim * sizeof(float));
    }

    // Cleanup
    flash_aligned_free(S_chunk);
    flash_aligned_free(P_chunk);
    flash_aligned_free(O_tile);
    flash_aligned_free(state);
}

// =============================================================================
// Main forward pass (with unpacked K/V)
// =============================================================================

void flash_attention_fused_forward(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O,
    float* logsumexp,
    int64_t L,
    int64_t head_dim)
{
    // Pack K and V
    size_t Kp_size = flash_kp_size(L, head_dim);
    size_t Vp_size = flash_vp_size(L, head_dim);

    int8_t* Kp = (int8_t*)flash_aligned_alloc(Kp_size);
    int8_t* Vp = (int8_t*)flash_aligned_alloc(Vp_size);

    pack_k_for_flash_attention(K, Kp, L, head_dim);
    pack_v_for_flash_attention(V, Vp, L, head_dim);

    // Call packed version
    flash_attention_fused_forward_packed(Q, Kp, Vp, O, logsumexp, L, head_dim);

    // Cleanup
    flash_aligned_free(Kp);
    flash_aligned_free(Vp);
}
