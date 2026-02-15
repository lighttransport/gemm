// softmax_exp2.c
// Fast softmax using exp2 approximation with SVE vectorization

#include "softmax_exp2.h"
#include <arm_sve.h>
#include <math.h>
#include <string.h>

// ============================================================================
// SVE fast exp2 approximation
// ============================================================================
// exp2(x) = 2^x using integer bit manipulation + polynomial refinement
//
// Algorithm:
// 1. Split x = floor(x) + frac(x) where frac in [0, 1)
// 2. 2^floor(x) via IEEE754 exponent manipulation
// 3. 2^frac(x) via polynomial approximation
// 4. Result = 2^floor(x) * 2^frac(x)

static inline svfloat32_t sve_exp2_f32(svfloat32_t x, svbool_t pg) {
    // Clamp to valid range [-126, 127]
    svfloat32_t min_val = svdup_f32(-126.0f);
    svfloat32_t max_val = svdup_f32(127.0f);
    x = svmax_f32_x(pg, x, min_val);
    x = svmin_f32_x(pg, x, max_val);

    // Shift to positive range for floor computation
    svfloat32_t bias = svdup_f32(126.0f);
    svfloat32_t x_biased = svadd_f32_x(pg, x, bias);

    // Floor via truncation (convert to int, back to float)
    svint32_t xi = svcvt_s32_f32_x(pg, x_biased);
    svfloat32_t xf = svcvt_f32_s32_x(pg, xi);

    // Fractional part: frac = x - (floor(x_biased) - 126)
    svfloat32_t floor_x = svsub_f32_x(pg, xf, bias);
    svfloat32_t frac = svsub_f32_x(pg, x, floor_x);

    // Polynomial for 2^frac, frac in [0, 1)
    // p(f) = 1 + f*(c1 + f*(c2 + f*c3))
    // Coefficients from minimax approximation
    svfloat32_t c1 = svdup_f32(0.6931472f);
    svfloat32_t c2 = svdup_f32(0.2402265f);
    svfloat32_t c3 = svdup_f32(0.0554913f);
    svfloat32_t one = svdup_f32(1.0f);

    // Horner's method
    svfloat32_t p = svmla_f32_x(pg, c2, frac, c3);     // c2 + f*c3
    p = svmla_f32_x(pg, c1, frac, p);                   // c1 + f*(c2 + f*c3)
    p = svmla_f32_x(pg, one, frac, p);                  // 1 + f*(c1 + f*(c2 + f*c3))

    // 2^floor(x) via exponent manipulation
    // IEEE754: float = 2^(exp-127), so set exp = floor(x) + 127
    svint32_t exp_bits = svadd_s32_x(pg, xi, svdup_s32(1));  // xi is floor(x)+126, add 1 for 127
    exp_bits = svlsl_n_s32_x(pg, exp_bits, 23);              // Shift to exponent position

    // Reinterpret as float
    svfloat32_t pow2_int = svreinterpret_f32_s32(exp_bits);

    // Result = 2^floor(x) * 2^frac(x)
    return svmul_f32_x(pg, pow2_int, p);
}

// ============================================================================
// INT8 Softmax with SVE
// ============================================================================
void softmax_int8_sve(const int32_t* S_tile, int8_t* P_tile,
                       int rows, int cols, float scale) {
    // Scale factor includes log2(e) for exp2
    float scale_log2e = scale * LOG2_E;

    for (int row = 0; row < rows; row++) {
        const int32_t* S_row = S_tile + row * cols;
        int8_t* P_row = P_tile + row * cols;

        // Pass 1: Find row maximum (for numerical stability)
        float row_max = -1e30f;
        svbool_t pg = svptrue_b32();
        svfloat32_t vmax = svdup_f32(-1e30f);

        int col = 0;
        while (col < cols) {
            svbool_t pg_col = svwhilelt_b32(col, cols);
            svint32_t vs = svld1_s32(pg_col, S_row + col);
            svfloat32_t vf = svcvt_f32_s32_x(pg_col, vs);
            vf = svmul_f32_x(pg_col, vf, svdup_f32(scale_log2e));
            vmax = svmax_f32_m(pg_col, vmax, vf);
            col += svcntw();
        }
        row_max = svmaxv_f32(pg, vmax);

        // Pass 2: Compute exp2(x - max) and sum
        float row_sum = 0.0f;
        svfloat32_t vsum = svdup_f32(0.0f);
        svfloat32_t vrow_max = svdup_f32(row_max);

        // Temporary buffer for exp values
        float exp_buf[cols] __attribute__((aligned(64)));

        col = 0;
        while (col < cols) {
            svbool_t pg_col = svwhilelt_b32(col, cols);
            svint32_t vs = svld1_s32(pg_col, S_row + col);
            svfloat32_t vf = svcvt_f32_s32_x(pg_col, vs);
            vf = svmul_f32_x(pg_col, vf, svdup_f32(scale_log2e));
            vf = svsub_f32_x(pg_col, vf, vrow_max);  // x - max

            // Fast exp2
            svfloat32_t vexp = sve_exp2_f32(vf, pg_col);
            svst1_f32(pg_col, exp_buf + col, vexp);
            vsum = svadd_f32_m(pg_col, vsum, vexp);

            col += svcntw();
        }
        row_sum = svaddv_f32(pg, vsum);

        // Pass 3: Normalize and quantize to int8
        float inv_sum = 127.0f / row_sum;  // Scale to [-127, 127]
        svfloat32_t vinv_sum = svdup_f32(inv_sum);

        col = 0;
        while (col < cols) {
            svbool_t pg_col = svwhilelt_b32(col, cols);
            svfloat32_t vexp = svld1_f32(pg_col, exp_buf + col);
            svfloat32_t vnorm = svmul_f32_x(pg_col, vexp, vinv_sum);

            // Round and convert to int8
            // Add 0.5 for rounding, then truncate
            svfloat32_t vhalf = svdup_f32(0.5f);
            vnorm = svadd_f32_x(pg_col, vnorm, vhalf);
            svint32_t vi32 = svcvt_s32_f32_x(pg_col, vnorm);

            // Clamp to int8 range
            vi32 = svmax_s32_x(pg_col, vi32, svdup_s32(-128));
            vi32 = svmin_s32_x(pg_col, vi32, svdup_s32(127));

            // Narrow to int8 and store
            // SVE doesn't have direct int32->int8, so we do it element by element
            // For better performance, could use uzp/trn instructions
            int32_t temp[16];
            svst1_s32(pg_col, temp, vi32);
            int cnt = svcntp_b32(pg_col, pg_col);
            for (int i = 0; i < cnt && (col + i) < cols; i++) {
                P_row[col + i] = (int8_t)temp[i];
            }

            col += svcntw();
        }
    }
}

// ============================================================================
// UINT8 Softmax with SVE (for UDOT with bias correction)
// ============================================================================
void softmax_uint8_sve(const int32_t* S_tile, uint8_t* P_tile,
                        int rows, int cols, float scale,
                        uint32_t* row_sums) {
    float scale_log2e = scale * LOG2_E;

    for (int row = 0; row < rows; row++) {
        const int32_t* S_row = S_tile + row * cols;
        uint8_t* P_row = P_tile + row * cols;

        // Pass 1: Find row maximum
        svbool_t pg = svptrue_b32();
        svfloat32_t vmax = svdup_f32(-1e30f);

        int col = 0;
        while (col < cols) {
            svbool_t pg_col = svwhilelt_b32(col, cols);
            svint32_t vs = svld1_s32(pg_col, S_row + col);
            svfloat32_t vf = svcvt_f32_s32_x(pg_col, vs);
            vf = svmul_f32_x(pg_col, vf, svdup_f32(scale_log2e));
            vmax = svmax_f32_m(pg_col, vmax, vf);
            col += svcntw();
        }
        float row_max = svmaxv_f32(pg, vmax);

        // Pass 2: Compute exp2(x - max) and sum
        svfloat32_t vsum = svdup_f32(0.0f);
        svfloat32_t vrow_max = svdup_f32(row_max);
        float exp_buf[cols] __attribute__((aligned(64)));

        col = 0;
        while (col < cols) {
            svbool_t pg_col = svwhilelt_b32(col, cols);
            svint32_t vs = svld1_s32(pg_col, S_row + col);
            svfloat32_t vf = svcvt_f32_s32_x(pg_col, vs);
            vf = svmul_f32_x(pg_col, vf, svdup_f32(scale_log2e));
            vf = svsub_f32_x(pg_col, vf, vrow_max);

            svfloat32_t vexp = sve_exp2_f32(vf, pg_col);
            svst1_f32(pg_col, exp_buf + col, vexp);
            vsum = svadd_f32_m(pg_col, vsum, vexp);

            col += svcntw();
        }
        float row_sum = svaddv_f32(pg, vsum);

        // Pass 3: Normalize and quantize to uint8 [0, 255]
        float inv_sum = 255.0f / row_sum;
        svfloat32_t vinv_sum = svdup_f32(inv_sum);
        uint32_t p_sum = 0;  // Sum of P values for bias correction

        col = 0;
        while (col < cols) {
            svbool_t pg_col = svwhilelt_b32(col, cols);
            svfloat32_t vexp = svld1_f32(pg_col, exp_buf + col);
            svfloat32_t vnorm = svmul_f32_x(pg_col, vexp, vinv_sum);

            // Round
            vnorm = svadd_f32_x(pg_col, vnorm, svdup_f32(0.5f));
            svint32_t vi32 = svcvt_s32_f32_x(pg_col, vnorm);

            // Clamp to uint8 range
            vi32 = svmax_s32_x(pg_col, vi32, svdup_s32(0));
            vi32 = svmin_s32_x(pg_col, vi32, svdup_s32(255));

            // Store and accumulate sum
            int32_t temp[16];
            svst1_s32(pg_col, temp, vi32);
            int cnt = svcntp_b32(pg_col, pg_col);
            for (int i = 0; i < cnt && (col + i) < cols; i++) {
                P_row[col + i] = (uint8_t)temp[i];
                p_sum += temp[i];
            }

            col += svcntw();
        }

        if (row_sums) {
            row_sums[row] = p_sum;
        }
    }
}

// ============================================================================
// Scalar fallback implementations
// ============================================================================
void softmax_int8(const int32_t* S_tile, int8_t* P_tile,
                   int rows, int cols, float scale, float* row_sums_out) {
    float scale_log2e = scale * LOG2_E;

    for (int row = 0; row < rows; row++) {
        const int32_t* S_row = S_tile + row * cols;
        int8_t* P_row = P_tile + row * cols;

        // Find max
        float row_max = -1e30f;
        for (int col = 0; col < cols; col++) {
            float val = (float)S_row[col] * scale_log2e;
            if (val > row_max) row_max = val;
        }

        // Compute exp and sum
        float row_sum = 0.0f;
        float exp_buf[cols];
        for (int col = 0; col < cols; col++) {
            float val = (float)S_row[col] * scale_log2e - row_max;
            float exp_val = fast_exp2f(val);
            exp_buf[col] = exp_val;
            row_sum += exp_val;
        }

        if (row_sums_out) {
            row_sums_out[row] = row_sum;
        }

        // Normalize and quantize
        float inv_sum = 127.0f / row_sum;
        for (int col = 0; col < cols; col++) {
            float norm = exp_buf[col] * inv_sum;
            int val = (int)(norm + 0.5f);
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            P_row[col] = (int8_t)val;
        }
    }
}

void softmax_uint8(const int32_t* S_tile, uint8_t* P_tile,
                    int rows, int cols, float scale, uint32_t* row_sums) {
    float scale_log2e = scale * LOG2_E;

    for (int row = 0; row < rows; row++) {
        const int32_t* S_row = S_tile + row * cols;
        uint8_t* P_row = P_tile + row * cols;

        // Find max
        float row_max = -1e30f;
        for (int col = 0; col < cols; col++) {
            float val = (float)S_row[col] * scale_log2e;
            if (val > row_max) row_max = val;
        }

        // Compute exp and sum
        float exp_sum = 0.0f;
        float exp_buf[cols];
        for (int col = 0; col < cols; col++) {
            float val = (float)S_row[col] * scale_log2e - row_max;
            float exp_val = fast_exp2f(val);
            exp_buf[col] = exp_val;
            exp_sum += exp_val;
        }

        // Normalize and quantize to uint8
        float inv_sum = 255.0f / exp_sum;
        uint32_t p_sum = 0;
        for (int col = 0; col < cols; col++) {
            float norm = exp_buf[col] * inv_sum;
            int val = (int)(norm + 0.5f);
            if (val > 255) val = 255;
            if (val < 0) val = 0;
            P_row[col] = (uint8_t)val;
            p_sum += val;
        }

        if (row_sums) {
            row_sums[row] = p_sum;
        }
    }
}

// ============================================================================
// Pack V with +128 bias for UDOT
// ============================================================================
void pack_V_uint8_biased(const int8_t* V, uint8_t* V_biased, int L, int d) {
    for (int i = 0; i < L * d; i++) {
        // Convert from int8 [-128, 127] to uint8 [0, 255] by adding 128
        V_biased[i] = (uint8_t)((int)V[i] + 128);
    }
}
