#include "flash_attention_fused_v2.h"
#include "gqa_pack.h"
#include "exp2_int.h"
#include "exp2_sve_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Timing for profiling
static double prof_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Global timing accumulators (set PROFILE_FUSED=1 to enable)
#define PROFILE_FUSED 1
#if PROFILE_FUSED
static double t_gemm = 0, t_softmax = 0, t_pv = 0, t_other = 0;
static int prof_count = 0;
#endif
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#include "pv_kernel_opt.h"  // Optimized P@V kernel
#endif

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

static void online_softmax_fused_scalar(
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

#ifdef __ARM_FEATURE_SVE
static void online_softmax_fused_sve(
    int32_t* S_chunk,
    float* P_chunk,
    flash_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    int32_t score_scale)
{
    // SVE kernel only handles full 64-column tiles.
    if (Bc != 64) {
        online_softmax_fused_scalar(S_chunk, P_chunk, state, Br, Bc, score_scale);
        return;
    }

    const float inv_q16 = 1.0f / 65536.0f;
    svbool_t pg = svptrue_b32();

    for (int64_t i = 0; i < Br; i++) {
        int32_t* S_row = S_chunk + i * Bc;
        int32_t* P_q16 = S_row;  // reuse buffer
        float* P_row = P_chunk + i * Bc;

        int32_t prev_max = state[i].m;
        int32_t new_max = prev_max;
        int64_t tile_sum = 0;

        exp2_row_64_sve(S_row, P_q16, &new_max, &tile_sum, score_scale, prev_max);

        int32_t rescale;
        if (prev_max <= -8388608) {
            rescale = 65536;
        } else {
            int32_t diff = prev_max - new_max;
            rescale = exp2_int32(diff);
        }

        state[i].l = (int64_t)(((int64_t)state[i].l * rescale) >> 16);
        state[i].l += tile_sum;
        state[i].rescale = rescale;
        state[i].m = new_max;

        svfloat32_t inv = svdup_f32(inv_q16);
        svint32_t p0 = svld1_s32(pg, P_q16 + 0);
        svint32_t p1 = svld1_s32(pg, P_q16 + 16);
        svint32_t p2 = svld1_s32(pg, P_q16 + 32);
        svint32_t p3 = svld1_s32(pg, P_q16 + 48);

        svfloat32_t f0 = svmul_f32_x(pg, svcvt_f32_s32_z(pg, p0), inv);
        svfloat32_t f1 = svmul_f32_x(pg, svcvt_f32_s32_z(pg, p1), inv);
        svfloat32_t f2 = svmul_f32_x(pg, svcvt_f32_s32_z(pg, p2), inv);
        svfloat32_t f3 = svmul_f32_x(pg, svcvt_f32_s32_z(pg, p3), inv);

        svst1_f32(pg, P_row + 0, f0);
        svst1_f32(pg, P_row + 16, f1);
        svst1_f32(pg, P_row + 32, f2);
        svst1_f32(pg, P_row + 48, f3);
    }
}
#endif

static void online_softmax_fused(
    int32_t* S_chunk,
    float* P_chunk,
    flash_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    int32_t score_scale)
{
#ifdef __ARM_FEATURE_SVE
    online_softmax_fused_sve(S_chunk, P_chunk, state, Br, Bc, score_scale);
#else
    online_softmax_fused_scalar(S_chunk, P_chunk, state, Br, Bc, score_scale);
#endif
}

// =============================================================================
// Step 3: P@V accumulation
// =============================================================================

static void accumulate_pv_fused_scalar(
    float* O_tile,
    const float* P_chunk,
    const int8_t* Vp,
    const flash_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    int64_t head_dim,
    int64_t L,
    int64_t k_tile)
{
    int64_t head_tiles = (head_dim + 63) >> 6;
    int64_t L_tiles = (L + 63) >> 6;
    int64_t L_groups = (L + 3) >> 2;

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
                int64_t k_global = k_tile + k;
                float p_val = P_row[k];

                // Locate V[k, ht*64:(ht+1)*64] in packed format
                int64_t kt = k_global >> 6;
                int64_t kg = (k_global & 63) >> 2;
                int64_t koffset = k_global & 3;

                const int8_t* V_col = Vp +
                    (ht * L_tiles * L_groups + kt * L_groups + kg) * 256 +
                    koffset * 64;

                // Accumulate 64 elements
                for (int64_t d = 0; d < 64 && (ht * 64 + d) < head_dim; d++) {
                    int8_t v_val = V_col[d];
                    O_row[d] += p_val * (float)v_val;
                }
            }
        }
    }
}

// Optimized P@V using packed P and K-blocking (row-major V)
#ifdef __ARM_FEATURE_SVE
static void accumulate_pv_fused_sve_opt(
    float* O_tile,
    const float* P_chunk,
    float* P_packed,           // Workspace for packed P
    const int8_t* V,           // Row-major V [L][head_dim]
    const flash_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    int64_t head_dim,
    int64_t k_tile)
{
    // Extract rescale factors
    int32_t rescale_factors[FA_TILE_BR];
    for (int64_t i = 0; i < Br; i++) {
        rescale_factors[i] = state[i].rescale;
    }

    // Call optimized kernel
    accumulate_pv_opt(O_tile, P_chunk, P_packed, V, rescale_factors,
                      Br, Bc, head_dim, k_tile);
}
#endif

// Original SVE implementation (for packed V format)
#ifdef __ARM_FEATURE_SVE
static void accumulate_pv_fused_sve(
    float* O_tile,
    const float* P_chunk,
    const int8_t* Vp,
    const flash_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    int64_t head_dim,
    int64_t L,
    int64_t k_tile)
{
    int64_t head_tiles = (head_dim + 63) >> 6;
    int64_t L_tiles = (L + 63) >> 6;
    int64_t L_groups = (L + 3) >> 2;

    // 1. Rescale O_tile
    for (int64_t i = 0; i < Br; i++) {
        float rescale_fp32 = (float)state[i].rescale / 65536.0f;
        for (int64_t d = 0; d < head_dim; d++) {
            O_tile[i * head_dim + d] *= rescale_fp32;
        }
    }

    svbool_t pg8 = svptrue_b8();

    // 2. O += P @ V
    for (int64_t ht = 0; ht < head_tiles; ht++) {
        int64_t base = ht << 6;
        svbool_t pg0 = svwhilelt_b32((uint64_t)(base + 0), (uint64_t)head_dim);
        svbool_t pg1 = svwhilelt_b32((uint64_t)(base + 16), (uint64_t)head_dim);
        svbool_t pg2 = svwhilelt_b32((uint64_t)(base + 32), (uint64_t)head_dim);
        svbool_t pg3 = svwhilelt_b32((uint64_t)(base + 48), (uint64_t)head_dim);

        for (int64_t i = 0; i < Br; i++) {
            float* O_row = O_tile + i * head_dim + base;
            const float* P_row = P_chunk + i * Bc;

            for (int64_t k = 0; k < Bc; k++) {
                int64_t k_global = k_tile + k;
                float p_val = P_row[k];
                svfloat32_t p = svdup_f32(p_val);

                int64_t kt = k_global >> 6;
                int64_t kg = (k_global & 63) >> 2;
                int64_t koffset = k_global & 3;

                const int8_t* V_col = Vp +
                    (ht * L_tiles * L_groups + kt * L_groups + kg) * 256 +
                    koffset * 64;

                svint8_t v8 = svld1_s8(pg8, V_col);
                svint16_t v16_lo = svunpklo_s16(v8);
                svint16_t v16_hi = svunpkhi_s16(v8);

                svint32_t v32_0 = svunpklo_s32(v16_lo);
                svint32_t v32_1 = svunpkhi_s32(v16_lo);
                svint32_t v32_2 = svunpklo_s32(v16_hi);
                svint32_t v32_3 = svunpkhi_s32(v16_hi);

                svfloat32_t f0 = svcvt_f32_s32_z(pg0, v32_0);
                svfloat32_t f1 = svcvt_f32_s32_z(pg1, v32_1);
                svfloat32_t f2 = svcvt_f32_s32_z(pg2, v32_2);
                svfloat32_t f3 = svcvt_f32_s32_z(pg3, v32_3);

                svfloat32_t o0 = svld1_f32(pg0, O_row + 0);
                svfloat32_t o1 = svld1_f32(pg1, O_row + 16);
                svfloat32_t o2 = svld1_f32(pg2, O_row + 32);
                svfloat32_t o3 = svld1_f32(pg3, O_row + 48);

                o0 = svmla_f32_m(pg0, o0, f0, p);
                o1 = svmla_f32_m(pg1, o1, f1, p);
                o2 = svmla_f32_m(pg2, o2, f2, p);
                o3 = svmla_f32_m(pg3, o3, f3, p);

                svst1_f32(pg0, O_row + 0, o0);
                svst1_f32(pg1, O_row + 16, o1);
                svst1_f32(pg2, O_row + 32, o2);
                svst1_f32(pg3, O_row + 48, o3);
            }
        }
    }
}
#endif

static void accumulate_pv_fused(
    float* O_tile,
    const float* P_chunk,
    const int8_t* Vp,
    const flash_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    int64_t head_dim,
    int64_t L,
    int64_t k_tile)
{
#ifdef __ARM_FEATURE_SVE
    accumulate_pv_fused_sve(O_tile, P_chunk, Vp, state, Br, Bc, head_dim, L, k_tile);
#else
    accumulate_pv_fused_scalar(O_tile, P_chunk, Vp, state, Br, Bc, head_dim, L, k_tile);
#endif
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

            // === FUSED KERNEL ===
#if PROFILE_FUSED
            double t0 = prof_get_time_ms();
#endif
            // Step 1: Q@K^T GEMM
            gemm_qk_tile(Q_tile, Kp_chunk, S_chunk, Br, Bc, head_dim);

#if PROFILE_FUSED
            double t1 = prof_get_time_ms();
#endif
            // Convert C_vec format to flat for softmax
            int32_t* S_flat = S_chunk;  // Reuse same buffer
            // Note: S_chunk is already in a usable format (row-major with padding)

            // Step 2: Online softmax
            online_softmax_fused(S_chunk, P_chunk, state, Br, Bc, score_scale);

#if PROFILE_FUSED
            double t2 = prof_get_time_ms();
#endif
            // Step 3: P@V accumulation
            accumulate_pv_fused(O_tile, P_chunk, Vp, state, Br, Bc, head_dim, L, k_tile);

#if PROFILE_FUSED
            double t3 = prof_get_time_ms();
            t_gemm += t1 - t0;
            t_softmax += t2 - t1;
            t_pv += t3 - t2;
            prof_count++;
#endif
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

#if PROFILE_FUSED
    printf("[PROFILE packed] tiles=%d, gemm=%.3f ms, softmax=%.3f ms, pv=%.3f ms, total=%.3f ms\n",
           prof_count, t_gemm, t_softmax, t_pv, t_gemm + t_softmax + t_pv);
    t_gemm = t_softmax = t_pv = 0;
    prof_count = 0;
#endif
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
    if (!Kp || !Vp) {
        fprintf(stderr, "Failed to allocate packed K/V buffers\n");
        flash_aligned_free(Kp);
        flash_aligned_free(Vp);
        return;
    }

    pack_k_for_flash_attention(K, Kp, L, head_dim);
    pack_v_for_flash_attention(V, Vp, L, head_dim);

    // Call packed version
    flash_attention_fused_forward_packed(Q, Kp, Vp, O, logsumexp, L, head_dim);

    // Cleanup
    flash_aligned_free(Kp);
    flash_aligned_free(Vp);
}

// =============================================================================
// Optimized forward pass with unpacked V (uses optimized P@V kernel)
// =============================================================================

void flash_attention_fused_forward_opt(
    const int8_t* Q,
    const int8_t* Kp,      // Packed K
    const int8_t* V,       // Unpacked V [L][head_dim], row-major
    float* O,
    float* logsumexp,
    int64_t L,
    int64_t head_dim)
{
    int32_t score_scale = (head_dim == 128) ? SCORE_SCALE_128 : SCORE_SCALE_256;

    size_t S_chunk_size = FA_TILE_BR * FA_TILE_BC * sizeof(int32_t);
    size_t P_chunk_size = FA_TILE_BR * FA_TILE_BC * sizeof(float);
    size_t P_packed_size = FA_TILE_BR * FA_TILE_BC * sizeof(float);  // For packed P
    size_t O_tile_size = FA_TILE_BR * head_dim * sizeof(float);
    size_t state_size = FA_TILE_BR * sizeof(flash_softmax_state_t);

    int32_t* S_chunk = (int32_t*)flash_aligned_alloc(S_chunk_size);
    float* P_chunk = (float*)flash_aligned_alloc(P_chunk_size);
    float* P_packed = (float*)flash_aligned_alloc(P_packed_size);
    float* O_tile = (float*)flash_aligned_alloc(O_tile_size);
    flash_softmax_state_t* state = (flash_softmax_state_t*)flash_aligned_alloc(state_size);

    for (int64_t q_tile = 0; q_tile < L; q_tile += FA_TILE_BR) {
        int64_t Br = ((q_tile + FA_TILE_BR) <= L) ? FA_TILE_BR : (L - q_tile);
        const int8_t* Q_tile = Q + q_tile * head_dim;
        float* O_out = O + q_tile * head_dim;
        float* lse_out = (logsumexp != NULL) ? (logsumexp + q_tile) : NULL;

        memset(O_tile, 0, Br * head_dim * sizeof(float));
        for (int64_t i = 0; i < Br; i++) {
            state[i].m = -8388608;
            state[i].l = 0;
            state[i].rescale = 65536;
        }

        for (int64_t k_tile = 0; k_tile < L; k_tile += FA_TILE_BC) {
            int64_t Bc = ((k_tile + FA_TILE_BC) <= L) ? FA_TILE_BC : (L - k_tile);

            int64_t kt_start = k_tile >> 6;
            int64_t K_groups = (head_dim + 3) >> 2;
            const int8_t* Kp_chunk = Kp + kt_start * (K_groups << 8);

            // Step 1: Q@K^T GEMM
            gemm_qk_tile(Q_tile, Kp_chunk, S_chunk, Br, Bc, head_dim);

            // Step 2: Online softmax
            online_softmax_fused(S_chunk, P_chunk, state, Br, Bc, score_scale);

            // Step 3: P@V with optimized kernel (unpacked V)
#ifdef __ARM_FEATURE_SVE
            accumulate_pv_fused_sve_opt(O_tile, P_chunk, P_packed, V, state,
                                        Br, Bc, head_dim, k_tile);
#else
            // Scalar fallback
            for (int64_t i = 0; i < Br; i++) {
                float rescale_fp32 = (float)state[i].rescale / 65536.0f;
                for (int64_t d = 0; d < head_dim; d++) {
                    O_tile[i * head_dim + d] *= rescale_fp32;
                }
            }
            for (int64_t i = 0; i < Br; i++) {
                const float* P_row = P_chunk + i * Bc;
                float* O_row = O_tile + i * head_dim;
                for (int64_t k = 0; k < Bc; k++) {
                    float p_val = P_row[k];
                    const int8_t* V_row = V + (k_tile + k) * head_dim;
                    for (int64_t d = 0; d < head_dim; d++) {
                        O_row[d] += p_val * (float)V_row[d];
                    }
                }
            }
#endif
        }

        normalize_output(O_tile, state, lse_out, Br, head_dim);
        memcpy(O_out, O_tile, Br * head_dim * sizeof(float));
    }

    flash_aligned_free(S_chunk);
    flash_aligned_free(P_chunk);
    flash_aligned_free(P_packed);
    flash_aligned_free(O_tile);
    flash_aligned_free(state);
}

// =============================================================================
// Scalar reference implementation (same fixed-point algorithm)
// =============================================================================

void flash_attention_fused_reference(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O,
    float* logsumexp,
    int64_t L,
    int64_t head_dim)
{
    int32_t score_scale = (head_dim == 128) ? SCORE_SCALE_128 : SCORE_SCALE_256;

    size_t Kp_size = flash_kp_size(L, head_dim);
    size_t Vp_size = flash_vp_size(L, head_dim);
    int8_t* Kp = (int8_t*)flash_aligned_alloc(Kp_size);
    int8_t* Vp = (int8_t*)flash_aligned_alloc(Vp_size);
    if (!Kp || !Vp) {
        fprintf(stderr, "Failed to allocate packed K/V buffers for reference\n");
        flash_aligned_free(Kp);
        flash_aligned_free(Vp);
        return;
    }

    pack_k_for_flash_attention(K, Kp, L, head_dim);
    pack_v_for_flash_attention(V, Vp, L, head_dim);

    size_t S_chunk_size = FA_TILE_BR * FA_TILE_BC * sizeof(int32_t);
    size_t P_chunk_size = FA_TILE_BR * FA_TILE_BC * sizeof(float);
    size_t O_tile_size = FA_TILE_BR * head_dim * sizeof(float);
    size_t state_size = FA_TILE_BR * sizeof(flash_softmax_state_t);

    int32_t* S_chunk = (int32_t*)flash_aligned_alloc(S_chunk_size);
    float* P_chunk = (float*)flash_aligned_alloc(P_chunk_size);
    float* O_tile = (float*)flash_aligned_alloc(O_tile_size);
    flash_softmax_state_t* state = (flash_softmax_state_t*)flash_aligned_alloc(state_size);
    if (!S_chunk || !P_chunk || !O_tile || !state) {
        fprintf(stderr, "Failed to allocate reference workspace\n");
        flash_aligned_free(S_chunk);
        flash_aligned_free(P_chunk);
        flash_aligned_free(O_tile);
        flash_aligned_free(state);
        flash_aligned_free(Kp);
        flash_aligned_free(Vp);
        return;
    }

    for (int64_t q_tile = 0; q_tile < L; q_tile += FA_TILE_BR) {
        int64_t Br = ((q_tile + FA_TILE_BR) <= L) ? FA_TILE_BR : (L - q_tile);
        const int8_t* Q_tile = Q + q_tile * head_dim;
        float* O_out = O + q_tile * head_dim;
        float* lse_out = (logsumexp != NULL) ? (logsumexp + q_tile) : NULL;

        memset(O_tile, 0, Br * head_dim * sizeof(float));
        for (int64_t i = 0; i < Br; i++) {
            state[i].m = -8388608;
            state[i].l = 0;
            state[i].rescale = 65536;
        }

        for (int64_t k_tile = 0; k_tile < L; k_tile += FA_TILE_BC) {
            int64_t Bc = ((k_tile + FA_TILE_BC) <= L) ? FA_TILE_BC : (L - k_tile);

            for (int64_t i = 0; i < Br; i++) {
                const int8_t* q_row = Q_tile + i * head_dim;
                int32_t* s_row = S_chunk + i * Bc;
                for (int64_t j = 0; j < Bc; j++) {
                    const int8_t* k_row = K + (k_tile + j) * head_dim;
                    int32_t dot = 0;
                    for (int64_t k = 0; k < head_dim; k++) {
                        dot += (int32_t)q_row[k] * (int32_t)k_row[k];
                    }
                    s_row[j] = dot;
                }
            }

            online_softmax_fused(S_chunk, P_chunk, state, Br, Bc, score_scale);
            accumulate_pv_fused(O_tile, P_chunk, Vp, state, Br, Bc, head_dim, L, k_tile);
        }

        normalize_output(O_tile, state, lse_out, Br, head_dim);
        memcpy(O_out, O_tile, Br * head_dim * sizeof(float));
    }

    flash_aligned_free(S_chunk);
    flash_aligned_free(P_chunk);
    flash_aligned_free(O_tile);
    flash_aligned_free(state);
    flash_aligned_free(Kp);
    flash_aligned_free(Vp);
}
