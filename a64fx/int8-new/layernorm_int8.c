// layernorm_int8.c - INT8 LayerNorm and RMSNorm using SVE intrinsics

#include "layernorm.h"
#include <arm_sve.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Helper: Fast inverse square root using integer approximation
int32_t fast_invsqrt_int32(int32_t x) {
    if (x <= 0) return 0x7FFFFFFF; // Max INT32

    // Use floating point for simplicity (can optimize later)
    float xf = (float)x / (1 << 16); // Q16.16 -> float
    float inv_sqrt = 1.0f / sqrtf(xf);
    return (int32_t)(inv_sqrt * (1 << 16)); // float -> Q16.16
}

// Integer square root
uint32_t isqrt_uint32(uint32_t x) {
    if (x == 0) return 0;

    uint32_t res = 0;
    uint32_t bit = 1U << 30; // Second-to-top bit set

    while (bit > x) bit >>= 2;

    while (bit != 0) {
        if (x >= res + bit) {
            x -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return res;
}

// ============================================================================
// INT8 LayerNorm
// ============================================================================

void layernorm_int8(
    const int8_t* input,
    int8_t* output,
    const int8_t* gamma,
    const int8_t* beta,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N)
{
    // Step 1: Compute mean using SVE
    int64_t sum = 0;
    size_t i = 0;

    svint32_t sum_vec = svdup_n_s32(0);

    // Process 64 elements at a time (64 bytes = full SVE vector)
    while (i + 64 <= N) {
        svbool_t pg = svptrue_b8();
        svint8_t v = svld1_s8(pg, &input[i]);

        // Extend INT8 to INT32 for accumulation
        // Split into 4 chunks of 16 elements each
        svint32_t v0 = svunpklo_s32(svunpklo_s16(v));
        svint32_t v1 = svunpkhi_s32(svunpklo_s16(v));
        svint32_t v2 = svunpklo_s32(svunpkhi_s16(v));
        svint32_t v3 = svunpkhi_s32(svunpkhi_s16(v));

        sum_vec = svadd_s32_x(pg, sum_vec, v0);
        sum_vec = svadd_s32_x(pg, sum_vec, v1);
        sum_vec = svadd_s32_x(pg, sum_vec, v2);
        sum_vec = svadd_s32_x(pg, sum_vec, v3);

        i += 64;
    }

    // Horizontal reduction
    svbool_t pg = svptrue_b32();
    sum += svaddv_s32(pg, sum_vec);

    // Process remaining elements
    while (i < N) {
        sum += input[i];
        i++;
    }

    // Mean in Q24.8 format (scaled by 256)
    int32_t mean = (sum << 8) / (int32_t)N;

    // Step 2: Compute variance = mean((x - mean)^2)
    int64_t var_sum = 0;
    i = 0;

    svint32_t var_vec = svdup_n_s32(0);
    svint32_t mean_vec = svdup_n_s32(mean >> 8); // Back to INT8 range

    while (i + 64 <= N) {
        svbool_t pg = svptrue_b8();
        svint8_t v = svld1_s8(pg, &input[i]);

        // Extend to INT32
        svint32_t v0 = svunpklo_s32(svunpklo_s16(v));
        svint32_t v1 = svunpkhi_s32(svunpklo_s16(v));
        svint32_t v2 = svunpklo_s32(svunpkhi_s16(v));
        svint32_t v3 = svunpkhi_s32(svunpkhi_s16(v));

        // Compute (x - mean)^2
        svint32_t d0 = svsub_s32_x(pg, v0, mean_vec);
        svint32_t d1 = svsub_s32_x(pg, v1, mean_vec);
        svint32_t d2 = svsub_s32_x(pg, v2, mean_vec);
        svint32_t d3 = svsub_s32_x(pg, v3, mean_vec);

        d0 = svmul_s32_x(pg, d0, d0);
        d1 = svmul_s32_x(pg, d1, d1);
        d2 = svmul_s32_x(pg, d2, d2);
        d3 = svmul_s32_x(pg, d3, d3);

        var_vec = svadd_s32_x(pg, var_vec, d0);
        var_vec = svadd_s32_x(pg, var_vec, d1);
        var_vec = svadd_s32_x(pg, var_vec, d2);
        var_vec = svadd_s32_x(pg, var_vec, d3);

        i += 64;
    }

    pg = svptrue_b32();
    var_sum += svaddv_s32(pg, var_vec);

    // Remaining elements
    int32_t mean_scalar = mean >> 8;
    while (i < N) {
        int32_t diff = input[i] - mean_scalar;
        var_sum += (int64_t)diff * diff;
        i++;
    }

    int32_t variance = (int32_t)(var_sum / N);

    // Step 3: Compute 1/sqrt(variance + epsilon)
    // Convert to Q16.16 for better precision
    int32_t var_q16 = variance << 8; // Q8 -> Q16.16
    int32_t inv_std = fast_invsqrt_int32(var_q16 + epsilon);

    // Step 4: Normalize and apply affine transformation
    // y = ((x - mean) * inv_std * gamma + beta)
    i = 0;

    svint32_t mean_v = svdup_n_s32(mean_scalar);
    svint32_t inv_std_v = svdup_n_s32(inv_std);

    while (i + 16 <= N) {
        svbool_t pg = svptrue_pat_b32(SV_VL16);

        // Load input (16 INT8 elements)
        svint8_t x_s8 = svld1_s8(pg, &input[i]);
        svint32_t x = svunpklo_s32(svunpklo_s16(x_s8));

        // Load gamma and beta
        svint8_t g_s8 = svld1_s8(pg, &gamma[i]);
        svint32_t g = svunpklo_s32(svunpklo_s16(g_s8));

        svint8_t b_s8 = svld1_s8(pg, &beta[i]);
        svint32_t b = svunpklo_s32(svunpklo_s16(b_s8));

        // Normalize: (x - mean) * inv_std
        svint32_t norm = svsub_s32_x(pg, x, mean_v);
        norm = svmul_s32_x(pg, norm, inv_std_v);
        norm = svasr_n_s32_x(pg, norm, 16); // Q16.16 -> Q0

        // Apply affine: norm * gamma + beta
        svint32_t y = svmul_s32_x(pg, norm, g);
        y = svasr_n_s32_x(pg, y, 7); // Scale down
        y = svadd_s32_x(pg, y, b);

        // Clamp to INT8 range
        y = svmax_n_s32_x(pg, y, -128);
        y = svmin_n_s32_x(pg, y, 127);

        // Store as INT8
        svint16_t y_s16 = svuzp1_s16(svreinterpret_s16_s32(y), svreinterpret_s16_s32(y));
        svint8_t y_s8 = svuzp1_s8(svreinterpret_s8_s16(y_s16), svreinterpret_s8_s16(y_s16));

        svst1_s8(pg, &output[i], y_s8);
        i += 16;
    }

    // Process remaining elements
    while (i < N) {
        int32_t x = input[i];
        int32_t norm = ((x - mean_scalar) * inv_std) >> 16;
        int32_t y = ((norm * gamma[i]) >> 7) + beta[i];

        // Clamp
        if (y < -128) y = -128;
        if (y > 127) y = 127;
        output[i] = (int8_t)y;
        i++;
    }
}

void layernorm_int8_noaffine(
    const int8_t* input,
    int8_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N)
{
    // Use gamma=1, beta=0 (identity transform after normalization)
    int8_t* gamma = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t* beta = (int8_t*)malloc(N * sizeof(int8_t));

    for (size_t i = 0; i < N; i++) {
        gamma[i] = 127; // ~1.0 in INT8
        beta[i] = 0;
    }

    layernorm_int8(input, output, gamma, beta, epsilon, input_scale, output_scale, N);

    free(gamma);
    free(beta);
}

// ============================================================================
// INT8 RMSNorm
// ============================================================================

void rmsnorm_int8(
    const int8_t* input,
    int8_t* output,
    const int8_t* gamma,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N)
{
    // Step 1: Compute RMS = sqrt(mean(x^2) + epsilon)
    int64_t sq_sum = 0;
    size_t i = 0;

    svint32_t sq_vec = svdup_n_s32(0);

    // Process 64 elements at a time
    while (i + 64 <= N) {
        svbool_t pg = svptrue_b8();
        svint8_t v = svld1_s8(pg, &input[i]);

        // Extend to INT32 and compute x^2
        svint32_t v0 = svunpklo_s32(svunpklo_s16(v));
        svint32_t v1 = svunpkhi_s32(svunpklo_s16(v));
        svint32_t v2 = svunpklo_s32(svunpkhi_s16(v));
        svint32_t v3 = svunpkhi_s32(svunpkhi_s16(v));

        v0 = svmul_s32_x(pg, v0, v0);
        v1 = svmul_s32_x(pg, v1, v1);
        v2 = svmul_s32_x(pg, v2, v2);
        v3 = svmul_s32_x(pg, v3, v3);

        sq_vec = svadd_s32_x(pg, sq_vec, v0);
        sq_vec = svadd_s32_x(pg, sq_vec, v1);
        sq_vec = svadd_s32_x(pg, sq_vec, v2);
        sq_vec = svadd_s32_x(pg, sq_vec, v3);

        i += 64;
    }

    // Horizontal reduction
    svbool_t pg = svptrue_b32();
    sq_sum += svaddv_s32(pg, sq_vec);

    // Remaining elements
    while (i < N) {
        int32_t x = input[i];
        sq_sum += (int64_t)x * x;
        i++;
    }

    // Mean squared value
    int32_t mean_sq = (int32_t)(sq_sum / N);

    // Compute 1/sqrt(mean_sq + epsilon)
    int32_t mean_sq_q16 = mean_sq << 8; // Q8 -> Q16.16
    int32_t inv_rms = fast_invsqrt_int32(mean_sq_q16 + epsilon);

    // Step 2: Normalize and apply gamma
    // y = (x * inv_rms * gamma)
    i = 0;

    svint32_t inv_rms_v = svdup_n_s32(inv_rms);

    while (i + 16 <= N) {
        svbool_t pg = svptrue_pat_b32(SV_VL16);

        // Load input
        svint8_t x_s8 = svld1_s8(pg, &input[i]);
        svint32_t x = svunpklo_s32(svunpklo_s16(x_s8));

        // Load gamma
        svint8_t g_s8 = svld1_s8(pg, &gamma[i]);
        svint32_t g = svunpklo_s32(svunpklo_s16(g_s8));

        // Normalize: x * inv_rms
        svint32_t norm = svmul_s32_x(pg, x, inv_rms_v);
        norm = svasr_n_s32_x(pg, norm, 16); // Q16.16 -> Q0

        // Apply gamma: norm * gamma
        svint32_t y = svmul_s32_x(pg, norm, g);
        y = svasr_n_s32_x(pg, y, 7); // Scale down

        // Clamp to INT8 range
        y = svmax_n_s32_x(pg, y, -128);
        y = svmin_n_s32_x(pg, y, 127);

        // Store as INT8
        svint16_t y_s16 = svuzp1_s16(svreinterpret_s16_s32(y), svreinterpret_s16_s32(y));
        svint8_t y_s8 = svuzp1_s8(svreinterpret_s8_s16(y_s16), svreinterpret_s8_s16(y_s16));

        svst1_s8(pg, &output[i], y_s8);
        i += 16;
    }

    // Remaining elements
    while (i < N) {
        int32_t x = input[i];
        int32_t norm = (x * inv_rms) >> 16;
        int32_t y = (norm * gamma[i]) >> 7;

        // Clamp
        if (y < -128) y = -128;
        if (y > 127) y = 127;
        output[i] = (int8_t)y;
        i++;
    }
}

void rmsnorm_int8_noaffine(
    const int8_t* input,
    int8_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N)
{
    // Use gamma=1 (identity scale)
    int8_t* gamma = (int8_t*)malloc(N * sizeof(int8_t));

    for (size_t i = 0; i < N; i++) {
        gamma[i] = 127; // ~1.0 in INT8
    }

    rmsnorm_int8(input, output, gamma, epsilon, input_scale, output_scale, N);

    free(gamma);
}
