// flash_attention_int16.c
// Flash Attention with INT16 attention weights for A64FX SVE
//
// Pipeline:
// 1. Q@K^T: INT8 SDOT -> INT32
// 2. Softmax: INT32 -> FP32 -> fast piecewise linear approximation
// 3. Quantize: FP32 -> INT16 (stochastic or deterministic)
// 4. P@V: INT16 × INT8(widened) -> INT32 using SVE SDOT

#include "flash_attention_int16.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

// =============================================================================
// Memory allocation
// =============================================================================

void* flash16_aligned_alloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 256, size) != 0) {
        return NULL;
    }
    return ptr;
}

void flash16_aligned_free(void* ptr) {
    free(ptr);
}

// =============================================================================
// Fast softmax approximation (piecewise linear, no division)
// =============================================================================

// Fast exp approximation for softmax
// exp(x) for x in [-inf, 0], output in [0, 1]
static inline float fast_exp_softmax(float x) {
    // Clamp to prevent overflow
    if (x < -10.0f) return 0.0f;
    if (x > 0.0f) x = 0.0f;

    // Polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6
    // For x in [-10, 0], use piecewise
    float x2 = x * x;
    float result = 1.0f + x * (1.0f + x * (0.5f + x * 0.166667f));
    return result > 0.0f ? result : 0.0f;
}

#ifdef __ARM_FEATURE_SVE
// SVE fast exp for softmax
static inline svfloat32_t fast_exp_sve_softmax(svbool_t pg, svfloat32_t x) {
    // Clamp x to [-10, 0]
    x = svmax_f32_x(pg, x, svdup_f32(-10.0f));
    x = svmin_f32_x(pg, x, svdup_f32(0.0f));

    // Polynomial: 1 + x + x²/2 + x³/6
    svfloat32_t x2 = svmul_f32_x(pg, x, x);
    svfloat32_t x3 = svmul_f32_x(pg, x2, x);

    svfloat32_t result = svdup_f32(1.0f);
    result = svadd_f32_x(pg, result, x);
    result = svmla_f32_x(pg, result, x2, svdup_f32(0.5f));
    result = svmla_f32_x(pg, result, x3, svdup_f32(0.166667f));

    // Clamp to [0, inf]
    return svmax_f32_x(pg, result, svdup_f32(0.0f));
}
#endif

// =============================================================================
// K Packing for Q@K^T (INT8 SDOT)
// =============================================================================

void pack_k_int16(
    const int8_t* K,
    int8_t* Kp,
    int64_t L,
    int64_t head_dim)
{
    // Pack K for INT8 SDOT: [L/64][head_dim/4][4][64]
    // Each 64-column tile is packed in K-major order

    int64_t L_tiles = (L + 63) / 64;
    int64_t K_groups = (head_dim + 3) / 4;

    for (int64_t lt = 0; lt < L_tiles; lt++) {
        for (int64_t kg = 0; kg < K_groups; kg++) {
            int8_t* Kp_block = Kp + (lt * K_groups + kg) * 256;

            for (int k = 0; k < 4; k++) {
                int64_t k_idx = kg * 4 + k;
                for (int l = 0; l < 64; l++) {
                    int64_t l_idx = lt * 64 + l;
                    int8_t val = (l_idx < L && k_idx < head_dim) ?
                        K[l_idx * head_dim + k_idx] : 0;
                    Kp_block[k * 64 + l] = val;
                }
            }
        }
    }
}

// =============================================================================
// V Packing for P@V (INT16 × INT8 widened)
// =============================================================================

void pack_v_int16(
    const int8_t* V,
    int8_t* Vp,
    int64_t L,
    int64_t head_dim)
{
    // Pack V for INT16 attention × INT8 value
    // Layout: [head_dim/32][L/4][4][32]
    // Each group of 4 L positions × 32 head_dim elements
    // V INT8 will be widened to INT16 during SDOT

    int64_t head_tiles = (head_dim + 31) / 32;  // 32 INT16 = 1 SVE vector
    int64_t L_groups = (L + 3) / 4;

    for (int64_t ht = 0; ht < head_tiles; ht++) {
        for (int64_t lg = 0; lg < L_groups; lg++) {
            int8_t* Vp_block = Vp + (ht * L_groups + lg) * 128;

            for (int k = 0; k < 4; k++) {
                int64_t l_idx = lg * 4 + k;
                for (int d = 0; d < 32; d++) {
                    int64_t h_idx = ht * 32 + d;
                    int8_t val = (l_idx < L && h_idx < head_dim) ?
                        V[l_idx * head_dim + h_idx] : 0;
                    Vp_block[k * 32 + d] = val;
                }
            }
        }
    }
}

// =============================================================================
// Step 1: Q@K^T using INT8 SDOT -> INT32
// =============================================================================

#ifdef __ARM_FEATURE_SVE
static void gemm_qk_int8_sve(
    const int8_t* Q_tile,   // [Br, K] INT8
    const int8_t* Kp,       // Packed K
    int32_t* S_tile,        // [Br, Bc] INT32 output
    int64_t Br,
    int64_t Bc,
    int64_t K,
    int64_t lda)            // Q row stride
{
    // Simple INT8 SDOT implementation
    // Each SDOT: 4 INT8 × 4 INT8 -> INT32 accumulate

    int64_t K_groups = (K + 3) / 4;

    for (int64_t i = 0; i < Br; i++) {
        const int8_t* Q_row = Q_tile + i * lda;

        for (int64_t j = 0; j < Bc; j += 16) {
            svbool_t pg = svwhilelt_b32((uint32_t)j, (uint32_t)Bc);
            svint32_t acc = svdup_s32(0);

            // Accumulate over K dimension
            for (int64_t kg = 0; kg < K_groups; kg++) {
                // Load 4 Q elements, broadcast to all lanes
                int8_t q0 = Q_row[kg * 4 + 0];
                int8_t q1 = Q_row[kg * 4 + 1];
                int8_t q2 = Q_row[kg * 4 + 2];
                int8_t q3 = Q_row[kg * 4 + 3];

                // Load K elements for this column chunk
                // K is packed as [L_tile][K_groups][4][64]
                int64_t l_tile = j / 64;
                int64_t l_offset = j % 64;
                const int8_t* Kp_ptr = Kp + (l_tile * K_groups + kg) * 256;

                // Manual dot product (SVE sdot operates on bytes grouped by 4)
                for (int lane = 0; lane < 16 && j + lane < Bc; lane++) {
                    int32_t dot = (int32_t)q0 * Kp_ptr[0 * 64 + l_offset + lane] +
                                  (int32_t)q1 * Kp_ptr[1 * 64 + l_offset + lane] +
                                  (int32_t)q2 * Kp_ptr[2 * 64 + l_offset + lane] +
                                  (int32_t)q3 * Kp_ptr[3 * 64 + l_offset + lane];
                    S_tile[i * Bc + j + lane] += dot;
                }
            }
        }
    }
}
#endif

// Scalar fallback for Q@K^T
static void gemm_qk_int8_scalar(
    const int8_t* Q_tile,
    const int8_t* K_tile,  // Unpacked K [Bc, K]
    int32_t* S_tile,
    int64_t Br,
    int64_t Bc,
    int64_t K,
    int64_t lda_q,
    int64_t lda_k)
{
    for (int64_t i = 0; i < Br; i++) {
        for (int64_t j = 0; j < Bc; j++) {
            int32_t sum = 0;
            for (int64_t k = 0; k < K; k++) {
                sum += (int32_t)Q_tile[i * lda_q + k] * K_tile[j * lda_k + k];
            }
            S_tile[i * Bc + j] = sum;
        }
    }
}

// =============================================================================
// Step 2: Softmax with FP32 approximation
// =============================================================================

#ifdef __ARM_FEATURE_SVE
static void softmax_fp32_sve(
    const int32_t* S_tile,  // [Br, Bc] INT32 scores
    float* P_tile,          // [Br, Bc] FP32 attention weights
    flash16_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    float scale)
{
    svbool_t pg = svptrue_b32();

    for (int64_t i = 0; i < Br; i++) {
        const int32_t* S_row = S_tile + i * Bc;
        float* P_row = P_tile + i * Bc;

        // 1. Find tile max (convert to FP32 first)
        svfloat32_t vmax = svdup_f32(-1e10f);
        for (int64_t j = 0; j < Bc; j += 16) {
            svbool_t pg_j = svwhilelt_b32((uint32_t)j, (uint32_t)Bc);
            svint32_t vs = svld1_s32(pg_j, S_row + j);
            svfloat32_t vf = svmul_f32_x(pg_j, svcvt_f32_s32_x(pg_j, vs), svdup_f32(scale));
            vmax = svmax_f32_m(pg_j, vmax, vf);
        }
        float tile_max = svmaxv_f32(pg, vmax);

        // 2. Update running max
        float prev_max = state[i].m;
        float new_max = (tile_max > prev_max) ? tile_max : prev_max;

        // 3. Compute rescale factor
        float rescale;
        if (prev_max < -1e9f) {
            rescale = 1.0f;  // First tile
        } else {
            rescale = fast_exp_softmax(prev_max - new_max);
        }

        state[i].l = state[i].l * rescale;
        state[i].rescale = rescale;
        state[i].m = new_max;

        // 4. Compute exp and sum
        float tile_sum = 0.0f;
        for (int64_t j = 0; j < Bc; j += 16) {
            svbool_t pg_j = svwhilelt_b32((uint32_t)j, (uint32_t)Bc);

            // Convert to FP32 and scale
            svint32_t vs = svld1_s32(pg_j, S_row + j);
            svfloat32_t vf = svmul_f32_x(pg_j, svcvt_f32_s32_x(pg_j, vs), svdup_f32(scale));

            // Subtract max
            vf = svsub_f32_x(pg_j, vf, svdup_f32(new_max));

            // Fast exp
            svfloat32_t vexp = fast_exp_sve_softmax(pg_j, vf);

            // Store
            svst1_f32(pg_j, P_row + j, vexp);

            // Accumulate sum
            tile_sum += svaddv_f32(pg_j, vexp);
        }
        state[i].l += tile_sum;
    }
}
#endif

// Scalar softmax fallback
static void softmax_fp32_scalar(
    const int32_t* S_tile,
    float* P_tile,
    flash16_softmax_state_t* state,
    int64_t Br,
    int64_t Bc,
    float scale)
{
    for (int64_t i = 0; i < Br; i++) {
        const int32_t* S_row = S_tile + i * Bc;
        float* P_row = P_tile + i * Bc;

        // Find max
        float tile_max = S_row[0] * scale;
        for (int64_t j = 1; j < Bc; j++) {
            float val = S_row[j] * scale;
            if (val > tile_max) tile_max = val;
        }

        // Update state
        float prev_max = state[i].m;
        float new_max = (tile_max > prev_max) ? tile_max : prev_max;
        float rescale = (prev_max < -1e9f) ? 1.0f : fast_exp_softmax(prev_max - new_max);

        state[i].l = state[i].l * rescale;
        state[i].rescale = rescale;
        state[i].m = new_max;

        // Compute exp
        float tile_sum = 0.0f;
        for (int64_t j = 0; j < Bc; j++) {
            float val = S_row[j] * scale - new_max;
            float exp_val = fast_exp_softmax(val);
            P_row[j] = exp_val;
            tile_sum += exp_val;
        }
        state[i].l += tile_sum;
    }
}

// =============================================================================
// Step 3: FP32 -> INT16 Quantization
// =============================================================================

// Note: We DON'T normalize by l here - that would break the online softmax algorithm.
// Instead, we scale by a fixed factor (32767) and divide by l only at the final output.
// P_int16 = P_fp32 * 32767, where P_fp32 = exp(S - max)
// Final output: O = O_acc / (l * 32767)

#ifdef __ARM_FEATURE_SVE
static void quantize_fp32_to_int16_sve(
    const float* P_tile,    // [Br, Bc] FP32 attention weights (unnormalized exp values)
    int16_t* P_int16,       // [Br, Bc] INT16 output
    int64_t Br,
    int64_t Bc)
{
    // Scale by 32767 (max INT16 value) - don't normalize by l yet
    svfloat32_t v_scale = svdup_f32(32767.0f);

    for (int64_t i = 0; i < Br; i++) {
        const float* P_row = P_tile + i * Bc;
        int16_t* P16_row = P_int16 + i * Bc;

        for (int64_t j = 0; j < Bc; j += 16) {
            svbool_t pg_j = svwhilelt_b32((uint32_t)j, (uint32_t)Bc);

            // Load FP32
            svfloat32_t vf = svld1_f32(pg_j, P_row + j);

            // Scale and convert to INT32
            vf = svmul_f32_x(pg_j, vf, v_scale);
            svint32_t vi32 = svcvt_s32_f32_x(pg_j, vf);

            // Saturate to INT16 range
            vi32 = svmax_s32_x(pg_j, vi32, svdup_s32(-32768));
            vi32 = svmin_s32_x(pg_j, vi32, svdup_s32(32767));

            // Store as INT16 (need to narrow)
            int32_t temp[16];
            svst1_s32(pg_j, temp, vi32);
            for (int k = 0; k < 16 && j + k < Bc; k++) {
                P16_row[j + k] = (int16_t)temp[k];
            }
        }
    }
}
#endif

static void quantize_fp32_to_int16_scalar(
    const float* P_tile,
    int16_t* P_int16,
    int64_t Br,
    int64_t Bc)
{
    // Scale by 32767 (max INT16 value) - don't normalize by l yet
    for (int64_t i = 0; i < Br; i++) {
        for (int64_t j = 0; j < Bc; j++) {
            float val = P_tile[i * Bc + j] * 32767.0f;
            int32_t ival = (int32_t)roundf(val);
            if (ival > 32767) ival = 32767;
            if (ival < -32768) ival = -32768;
            P_int16[i * Bc + j] = (int16_t)ival;
        }
    }
}

// =============================================================================
// Step 4: P@V using INT16 × INT8(widened) -> INT32
// =============================================================================

#ifdef __ARM_FEATURE_SVE
static void gemm_pv_int16_sve(
    const int16_t* P_tile,  // [Br, Bc] INT16 attention weights
    const int8_t* V_tile,   // [Bc, head_dim] INT8 values
    int32_t* O_acc,         // [Br, head_dim] INT32 accumulator
    const flash16_softmax_state_t* state,  // Per-row state with rescale factors
    int64_t Br,
    int64_t Bc,
    int64_t head_dim,
    int64_t lda_v)
{
    // P@V: [Br, Bc] × [Bc, head_dim] -> [Br, head_dim]
    // P is INT16, V is INT8 widened to INT16 during load

    for (int64_t i = 0; i < Br; i++) {
        const int16_t* P_row = P_tile + i * Bc;
        float rescale = state[i].rescale;

        // Rescale existing accumulator
        if (rescale < 0.999f || rescale > 1.001f) {
            svfloat32_t vrescale = svdup_f32(rescale);
            for (int64_t d = 0; d < head_dim; d += 16) {
                svbool_t pg_d = svwhilelt_b32((uint32_t)d, (uint32_t)head_dim);
                svint32_t vo = svld1_s32(pg_d, O_acc + i * head_dim + d);
                svfloat32_t vof = svcvt_f32_s32_x(pg_d, vo);
                vof = svmul_f32_x(pg_d, vof, vrescale);
                vo = svcvt_s32_f32_x(pg_d, vof);
                svst1_s32(pg_d, O_acc + i * head_dim + d, vo);
            }
        }

        // Accumulate P × V using SVE1 operations
        // For INT16 × INT8 -> INT32, we:
        // 1. Load V as INT8, widen to INT16, then widen to INT32
        // 2. Broadcast P value as INT32
        // 3. Multiply and accumulate as INT32
        for (int64_t k = 0; k < Bc; k++) {
            int16_t p_val = P_row[k];
            if (p_val == 0) continue;  // Skip zeros

            const int8_t* V_row = V_tile + k * lda_v;
            int32_t p_val32 = (int32_t)p_val;
            svint32_t vp32 = svdup_s32(p_val32);

            // Process 16 INT32 elements at a time (SVE 512-bit = 16 × 32-bit)
            for (int64_t d = 0; d < head_dim; d += 16) {
                svbool_t pg32 = svwhilelt_b32((uint32_t)d, (uint32_t)head_dim);

                // Load V as INT8 (16 elements for 16 outputs)
                // Note: We need to manually widen INT8 -> INT16 -> INT32
                int32_t v32_buf[16];
                for (int dd = 0; dd < 16 && d + dd < head_dim; dd++) {
                    v32_buf[dd] = (int32_t)V_row[d + dd];
                }
                svint32_t v32 = svld1_s32(pg32, v32_buf);

                // Load existing accumulator, multiply, accumulate
                svint32_t acc = svld1_s32(pg32, O_acc + i * head_dim + d);
                svint32_t prod = svmul_s32_x(pg32, vp32, v32);
                acc = svadd_s32_x(pg32, acc, prod);
                svst1_s32(pg32, O_acc + i * head_dim + d, acc);
            }
        }
    }
}
#endif

// Scalar P@V fallback with per-row rescale
static void gemm_pv_int16_scalar(
    const int16_t* P_tile,
    const int8_t* V_tile,
    int32_t* O_acc,
    const flash16_softmax_state_t* state,  // Per-row state with rescale factors
    int64_t Br,
    int64_t Bc,
    int64_t head_dim,
    int64_t lda_v)
{
    for (int64_t i = 0; i < Br; i++) {
        float rescale = state[i].rescale;

        // Rescale existing accumulator for this row
        if (rescale < 0.999f || rescale > 1.001f) {
            for (int64_t d = 0; d < head_dim; d++) {
                O_acc[i * head_dim + d] = (int32_t)(O_acc[i * head_dim + d] * rescale);
            }
        }

        // Accumulate P × V
        for (int64_t k = 0; k < Bc; k++) {
            int16_t p_val = P_tile[i * Bc + k];
            for (int64_t d = 0; d < head_dim; d++) {
                O_acc[i * head_dim + d] += (int32_t)p_val * (int32_t)V_tile[k * lda_v + d];
            }
        }
    }
}

// =============================================================================
// Main API
// =============================================================================

void flash_attention_int16_forward(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O,
    float* logsumexp,
    int64_t L,
    int64_t head_dim,
    float scale)
{
    // Allocate workspace
    int32_t* S_tile = (int32_t*)flash16_aligned_alloc(FA16_TILE_BR * FA16_TILE_BC * sizeof(int32_t));
    float* P_fp32 = (float*)flash16_aligned_alloc(FA16_TILE_BR * FA16_TILE_BC * sizeof(float));
    int16_t* P_int16 = (int16_t*)flash16_aligned_alloc(FA16_TILE_BR * FA16_TILE_BC * sizeof(int16_t));
    int32_t* O_acc = (int32_t*)flash16_aligned_alloc(FA16_TILE_BR * head_dim * sizeof(int32_t));
    flash16_softmax_state_t* state = (flash16_softmax_state_t*)flash16_aligned_alloc(
        FA16_TILE_BR * sizeof(flash16_softmax_state_t));

    // Process Q in tiles of BR rows
    for (int64_t i0 = 0; i0 < L; i0 += FA16_TILE_BR) {
        int64_t Br = (i0 + FA16_TILE_BR <= L) ? FA16_TILE_BR : (L - i0);

        // Initialize state for this Q tile
        for (int64_t i = 0; i < Br; i++) {
            state[i].m = -1e10f;
            state[i].l = 0.0f;
            state[i].rescale = 1.0f;
        }

        // Initialize O accumulator
        memset(O_acc, 0, Br * head_dim * sizeof(int32_t));

        // Process K/V in tiles of BC columns
        for (int64_t j0 = 0; j0 < L; j0 += FA16_TILE_BC) {
            int64_t Bc = (j0 + FA16_TILE_BC <= L) ? FA16_TILE_BC : (L - j0);

            // Clear S tile
            memset(S_tile, 0, Br * Bc * sizeof(int32_t));

            // Step 1: Q@K^T -> S (INT8 SDOT -> INT32)
            gemm_qk_int8_scalar(
                Q + i0 * head_dim,
                K + j0 * head_dim,
                S_tile,
                Br, Bc, head_dim,
                head_dim, head_dim);

            // Step 2: Softmax (INT32 -> FP32)
#ifdef __ARM_FEATURE_SVE
            softmax_fp32_sve(S_tile, P_fp32, state, Br, Bc, scale);
#else
            softmax_fp32_scalar(S_tile, P_fp32, state, Br, Bc, scale);
#endif

            // Step 3: Quantize to INT16 (don't normalize by l yet)
#ifdef __ARM_FEATURE_SVE
            quantize_fp32_to_int16_sve(P_fp32, P_int16, Br, Bc);
#else
            quantize_fp32_to_int16_scalar(P_fp32, P_int16, Br, Bc);
#endif

            // Step 4: P@V (INT16 × INT8 -> INT32)
            gemm_pv_int16_scalar(
                P_int16,
                V + j0 * head_dim,
                O_acc,
                state,  // Pass full state array for per-row rescale
                Br, Bc, head_dim,
                head_dim);
        }

        // Final normalization and convert to FP32
        // O_acc = sum(P_int16 * V) where P_int16 = exp(S - max) * 32767
        // We need: O = sum(P_normalized * V) where P_normalized = exp(S - max) / l
        // Therefore: O = O_acc / (l * 32767)
        for (int64_t i = 0; i < Br; i++) {
            float o_scale = 1.0f / (state[i].l * 32767.0f);

            for (int64_t d = 0; d < head_dim; d++) {
                O[(i0 + i) * head_dim + d] = O_acc[i * head_dim + d] * o_scale;
            }

            // Store logsumexp if requested
            if (logsumexp != NULL) {
                logsumexp[i0 + i] = state[i].m + logf(state[i].l);
            }
        }
    }

    // Free workspace
    flash16_aligned_free(S_tile);
    flash16_aligned_free(P_fp32);
    flash16_aligned_free(P_int16);
    flash16_aligned_free(O_acc);
    flash16_aligned_free(state);
}
