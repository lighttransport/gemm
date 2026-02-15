// softmax_sve.c - SVE Optimized Softmax Implementation
// Uses fast polynomial exp approximation with SVE vectors

#include "softmax_sve.h"
#include <arm_sve.h>
#include <math.h>

// =============================================================================
// Fast exp approximation using polynomial
// exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 for x in [-ln(2), 0]
// For general x: exp(x) = 2^n * exp(f) where x = n*ln(2) + f
// =============================================================================

// Constants for exp approximation
static const float LOG2E = 1.4426950408889634f;  // log2(e)
static const float LN2 = 0.6931471805599453f;    // ln(2)
static const float EXP_C1 = 1.0f;
static const float EXP_C2 = 0.6931472f;          // ln(2)
static const float EXP_C3 = 0.2402265f;          // ln(2)²/2
static const float EXP_C4 = 0.0555041f;          // ln(2)³/6

// SVE fast exp for 16 floats
// Uses 2^x = 2^floor(x) * 2^frac(x) with polynomial approximation
static inline svfloat32_t exp_approx_sve(svfloat32_t x, svbool_t pg) {
    // Convert to base-2: x * log2(e)
    svfloat32_t y = svmul_f32_x(pg, x, svdup_f32(LOG2E));
    
    // Split into integer and fractional parts
    // floor(y) - be careful with negative values
    svfloat32_t y_floor = svrintm_f32_x(pg, y);  // Floor
    svfloat32_t f = svsub_f32_x(pg, y, y_floor); // Fractional part [0, 1)
    
    // Compute 2^floor(y) using bit manipulation
    // 2^n = (127 + n) << 23 in IEEE 754
    svint32_t n = svcvt_s32_f32_x(pg, y_floor);
    svint32_t exp_bits = svadd_s32_x(pg, n, svdup_s32(127));
    exp_bits = svlsl_n_s32_x(pg, exp_bits, 23);
    svfloat32_t pow2_n = svreinterpret_f32_s32(exp_bits);
    
    // Clamp to avoid overflow/underflow
    pow2_n = svmax_f32_x(pg, pow2_n, svdup_f32(0.0f));
    
    // Compute 2^f using polynomial: 2^f ≈ 1 + f*ln(2) + f²*ln(2)²/2 + f³*ln(2)³/6
    // Horner's method: ((c4*f + c3)*f + c2)*f + c1
    svfloat32_t result = svdup_f32(EXP_C4);
    result = svmla_f32_x(pg, svdup_f32(EXP_C3), result, f);
    result = svmla_f32_x(pg, svdup_f32(EXP_C2), result, f);
    result = svmla_f32_x(pg, svdup_f32(EXP_C1), result, f);
    
    // Multiply by 2^n
    result = svmul_f32_x(pg, result, pow2_n);
    
    return result;
}

// =============================================================================
// SVE Softmax for 6 rows × 64 columns
// =============================================================================

void softmax_tile_sve(
    const int32_t* scores,
    float scale,
    softmax_state_t* state,
    float* O_accum,
    int D,
    int8_t* P_out,
    float* max_exp_out)
{
    svbool_t pg = svptrue_b32();
    svfloat32_t scale_vec = svdup_f32(scale);
    
    for (int r = 0; r < 6; r++) {
        const int32_t* row_scores = scores + r * 64;
        float* row_O = O_accum + r * D;
        int8_t* row_P = P_out + r * 64;
        
        // Step 1: Find chunk max using SVE
        svfloat32_t chunk_max_vec = svdup_f32(-1e30f);
        for (int i = 0; i < 64; i += 16) {
            svint32_t s = svld1_s32(pg, row_scores + i);
            svfloat32_t sf = svcvt_f32_s32_x(pg, s);
            sf = svmul_f32_x(pg, sf, scale_vec);
            chunk_max_vec = svmax_f32_x(pg, chunk_max_vec, sf);
        }
        float chunk_max = svmaxv_f32(pg, chunk_max_vec);
        
        // Step 2: Update running max and compute rescale factor
        float old_max = state[r].max;
        float new_max = (chunk_max > old_max) ? chunk_max : old_max;
        float rescale = expf(old_max - new_max);
        
        // Step 3: Rescale previous O accumulator
        svfloat32_t rescale_vec = svdup_f32(rescale);
        for (int d = 0; d < D; d += 16) {
            svbool_t pg_d = svwhilelt_b32(d, D);
            svfloat32_t o = svld1_f32(pg_d, row_O + d);
            o = svmul_f32_x(pg_d, o, rescale_vec);
            svst1_f32(pg_d, row_O + d, o);
        }
        
        // Step 4: Compute exp(score - new_max) and find max_exp, sum
        svfloat32_t new_max_vec = svdup_f32(new_max);
        svfloat32_t chunk_sum_vec = svdup_f32(0.0f);
        svfloat32_t max_exp_vec = svdup_f32(0.0f);
        float exp_buf[64] __attribute__((aligned(64)));
        
        for (int i = 0; i < 64; i += 16) {
            svint32_t s = svld1_s32(pg, row_scores + i);
            svfloat32_t sf = svcvt_f32_s32_x(pg, s);
            sf = svmul_f32_x(pg, sf, scale_vec);
            svfloat32_t x = svsub_f32_x(pg, sf, new_max_vec);
            
            // Fast exp approximation
            svfloat32_t exp_val = exp_approx_sve(x, pg);
            
            // Store for later quantization
            svst1_f32(pg, exp_buf + i, exp_val);
            
            // Accumulate sum and find max
            chunk_sum_vec = svadd_f32_x(pg, chunk_sum_vec, exp_val);
            max_exp_vec = svmax_f32_x(pg, max_exp_vec, exp_val);
        }
        
        float chunk_sum = svaddv_f32(pg, chunk_sum_vec);
        float chunk_max_exp = svmaxv_f32(pg, max_exp_vec);
        
        // Step 5: Update running state
        state[r].sum = state[r].sum * rescale + chunk_sum;
        state[r].max = new_max;
        max_exp_out[r] = chunk_max_exp;
        
        // Step 6: Quantize to INT8
        // P_int8 = exp * 127 / max_exp
        if (chunk_max_exp > 0.0f) {
            svfloat32_t inv_max = svdup_f32(127.0f / chunk_max_exp);
            svfloat32_t half = svdup_f32(0.5f);
            svfloat32_t zero = svdup_f32(0.0f);
            svfloat32_t max127 = svdup_f32(127.0f);
            
            for (int i = 0; i < 64; i += 16) {
                svfloat32_t exp_val = svld1_f32(pg, exp_buf + i);
                svfloat32_t p = svmul_f32_x(pg, exp_val, inv_max);
                p = svadd_f32_x(pg, p, half);  // Round
                p = svmax_f32_x(pg, p, zero);
                p = svmin_f32_x(pg, p, max127);
                
                svint32_t p_int = svcvt_s32_f32_x(pg, p);
                // Narrow to int8 (store lower 8 bits)
                int32_t temp[16] __attribute__((aligned(64)));
                svst1_s32(pg, temp, p_int);
                for (int j = 0; j < 16; j++) {
                    row_P[i + j] = (int8_t)temp[j];
                }
            }
        } else {
            for (int i = 0; i < 64; i++) {
                row_P[i] = 0;
            }
        }
    }
}

// =============================================================================
// SVE Softmax for 4 rows × 64 columns (for 4-row kernels)
// =============================================================================

void softmax_tile_4row_sve(
    const int32_t* scores,
    float scale,
    softmax_state_t* state,
    float* O_accum,
    int D,
    int8_t* P_out,
    float* max_exp_out)
{
    svbool_t pg = svptrue_b32();
    svfloat32_t scale_vec = svdup_f32(scale);

    for (int r = 0; r < 4; r++) {
        const int32_t* row_scores = scores + r * 64;
        float* row_O = O_accum + r * D;
        int8_t* row_P = P_out + r * 64;

        // Step 1: Find chunk max using SVE
        svfloat32_t chunk_max_vec = svdup_f32(-1e30f);
        for (int i = 0; i < 64; i += 16) {
            svint32_t s = svld1_s32(pg, row_scores + i);
            svfloat32_t sf = svcvt_f32_s32_x(pg, s);
            sf = svmul_f32_x(pg, sf, scale_vec);
            chunk_max_vec = svmax_f32_x(pg, chunk_max_vec, sf);
        }
        float chunk_max = svmaxv_f32(pg, chunk_max_vec);

        // Step 2: Update running max and compute rescale factor
        float old_max = state[r].max;
        float new_max = (chunk_max > old_max) ? chunk_max : old_max;
        float rescale = expf(old_max - new_max);

        // Step 3: Rescale previous O accumulator
        svfloat32_t rescale_vec = svdup_f32(rescale);
        for (int d = 0; d < D; d += 16) {
            svbool_t pg_d = svwhilelt_b32(d, D);
            svfloat32_t o = svld1_f32(pg_d, row_O + d);
            o = svmul_f32_x(pg_d, o, rescale_vec);
            svst1_f32(pg_d, row_O + d, o);
        }

        // Step 4: Compute exp(score - new_max) and find max_exp, sum
        svfloat32_t new_max_vec = svdup_f32(new_max);
        svfloat32_t chunk_sum_vec = svdup_f32(0.0f);
        svfloat32_t max_exp_vec = svdup_f32(0.0f);
        float exp_buf[64] __attribute__((aligned(64)));

        for (int i = 0; i < 64; i += 16) {
            svint32_t s = svld1_s32(pg, row_scores + i);
            svfloat32_t sf = svcvt_f32_s32_x(pg, s);
            sf = svmul_f32_x(pg, sf, scale_vec);
            svfloat32_t x = svsub_f32_x(pg, sf, new_max_vec);

            // Fast exp approximation
            svfloat32_t exp_val = exp_approx_sve(x, pg);

            // Store for later quantization
            svst1_f32(pg, exp_buf + i, exp_val);

            // Accumulate sum and find max
            chunk_sum_vec = svadd_f32_x(pg, chunk_sum_vec, exp_val);
            max_exp_vec = svmax_f32_x(pg, max_exp_vec, exp_val);
        }

        float chunk_sum = svaddv_f32(pg, chunk_sum_vec);
        float chunk_max_exp = svmaxv_f32(pg, max_exp_vec);

        // Step 5: Update running state
        state[r].sum = state[r].sum * rescale + chunk_sum;
        state[r].max = new_max;
        max_exp_out[r] = chunk_max_exp;

        // Step 6: Quantize to INT8
        if (chunk_max_exp > 0.0f) {
            svfloat32_t inv_max = svdup_f32(127.0f / chunk_max_exp);
            svfloat32_t half = svdup_f32(0.5f);
            svfloat32_t zero = svdup_f32(0.0f);
            svfloat32_t max127 = svdup_f32(127.0f);

            for (int i = 0; i < 64; i += 16) {
                svfloat32_t exp_val = svld1_f32(pg, exp_buf + i);
                svfloat32_t p = svmul_f32_x(pg, exp_val, inv_max);
                p = svadd_f32_x(pg, p, half);
                p = svmax_f32_x(pg, p, zero);
                p = svmin_f32_x(pg, p, max127);

                svint32_t p_int = svcvt_s32_f32_x(pg, p);
                int32_t temp[16] __attribute__((aligned(64)));
                svst1_s32(pg, temp, p_int);
                for (int j = 0; j < 16; j++) {
                    row_P[i + j] = (int8_t)temp[j];
                }
            }
        } else {
            for (int i = 0; i < 64; i++) {
                row_P[i] = 0;
            }
        }
    }
}

// =============================================================================
// Single-row version for compatibility
// =============================================================================

float softmax_chunk_sve(
    const int32_t* scores,
    float scale,
    softmax_state_t* state,
    float* O_accum,
    int D,
    int8_t* P_out)
{
    float max_exp;
    softmax_state_t temp_state = *state;
    
    softmax_tile_sve(scores, scale, &temp_state, O_accum, D, P_out, &max_exp);
    *state = temp_state;
    
    return max_exp;
}

// =============================================================================
// Final normalization
// =============================================================================

void softmax_finalize_sve(
    float* O,
    const float* O_accum,
    const softmax_state_t* state,
    int D)
{
    svbool_t pg = svptrue_b32();
    
    for (int r = 0; r < 6; r++) {
        float inv_sum = 1.0f / state[r].sum;
        svfloat32_t inv_sum_vec = svdup_f32(inv_sum);
        
        const float* src = O_accum + r * D;
        float* dst = O + r * D;
        
        for (int d = 0; d < D; d += 16) {
            svbool_t pg_d = svwhilelt_b32(d, D);
            svfloat32_t o = svld1_f32(pg_d, src + d);
            o = svmul_f32_x(pg_d, o, inv_sum_vec);
            svst1_f32(pg_d, dst + d, o);
        }
    }
}
