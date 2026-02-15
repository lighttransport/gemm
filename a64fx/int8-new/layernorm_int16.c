// layernorm_int16.c - INT16 LayerNorm and RMSNorm using SVE intrinsics

#include "layernorm.h"
#include <arm_sve.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// INT16 LayerNorm
// ============================================================================

void layernorm_int16(
    const int16_t* input,
    int16_t* output,
    const int16_t* gamma,
    const int16_t* beta,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N)
{
    // Step 1: Compute mean using SVE
    int64_t sum = 0;
    size_t i = 0;

    svint32_t sum_vec = svdup_n_s32(0);

    // Process 32 elements at a time (64 bytes = full SVE vector for INT16)
    while (i + 32 <= N) {
        svbool_t pg = svptrue_b16();
        svint16_t v = svld1_s16(pg, &input[i]);

        // Extend INT16 to INT32 for accumulation
        svint32_t v0 = svunpklo_s32(v);
        svint32_t v1 = svunpkhi_s32(v);

        sum_vec = svadd_s32_x(pg, sum_vec, v0);
        sum_vec = svadd_s32_x(pg, sum_vec, v1);

        i += 32;
    }

    // Horizontal reduction
    svbool_t pg = svptrue_b32();
    sum += svaddv_s32(pg, sum_vec);

    // Process remaining elements
    while (i < N) {
        sum += input[i];
        i++;
    }

    // Mean in INT32 (Q16.16 format after scaling)
    int32_t mean = (int32_t)(sum / (int64_t)N);

    // Step 2: Compute variance = mean((x - mean)^2)
    int64_t var_sum = 0;
    i = 0;

    svint64_t var_vec = svdup_n_s64(0);
    svint32_t mean_vec = svdup_n_s32(mean);

    while (i + 32 <= N) {
        svbool_t pg = svptrue_b16();
        svint16_t v = svld1_s16(pg, &input[i]);

        // Extend to INT32
        svint32_t v0 = svunpklo_s32(v);
        svint32_t v1 = svunpkhi_s32(v);

        // Compute (x - mean)^2 with INT64 accumulation
        svint32_t d0 = svsub_s32_x(pg, v0, mean_vec);
        svint32_t d1 = svsub_s32_x(pg, v1, mean_vec);

        // Extend to INT64 for multiplication
        svint64_t d0_lo = svunpklo_s64(d0);
        svint64_t d0_hi = svunpkhi_s64(d0);
        svint64_t d1_lo = svunpklo_s64(d1);
        svint64_t d1_hi = svunpkhi_s64(d1);

        // Square
        d0_lo = svmul_s64_x(pg, d0_lo, d0_lo);
        d0_hi = svmul_s64_x(pg, d0_hi, d0_hi);
        d1_lo = svmul_s64_x(pg, d1_lo, d1_lo);
        d1_hi = svmul_s64_x(pg, d1_hi, d1_hi);

        var_vec = svadd_s64_x(pg, var_vec, d0_lo);
        var_vec = svadd_s64_x(pg, var_vec, d0_hi);
        var_vec = svadd_s64_x(pg, var_vec, d1_lo);
        var_vec = svadd_s64_x(pg, var_vec, d1_hi);

        i += 32;
    }

    pg = svptrue_b64();
    var_sum += svaddv_s64(pg, var_vec);

    // Remaining elements
    while (i < N) {
        int64_t diff = (int64_t)input[i] - mean;
        var_sum += diff * diff;
        i++;
    }

    int32_t variance = (int32_t)(var_sum / N);

    // Step 3: Compute 1/sqrt(variance + epsilon)
    // variance is in Q16 format, convert to Q16.16
    int32_t inv_std = fast_invsqrt_int32(variance + epsilon);

    // Step 4: Normalize and apply affine transformation
    i = 0;

    svint32_t mean_v = svdup_n_s32(mean);
    svint32_t inv_std_v = svdup_n_s32(inv_std);

    while (i + 16 <= N) {
        svbool_t pg = svptrue_pat_b32(SV_VL16);

        // Load input (16 INT16 elements)
        svint16_t x_s16 = svld1_s16(pg, &input[i]);
        svint32_t x = svunpklo_s32(x_s16);

        // Load gamma and beta
        svint16_t g_s16 = svld1_s16(pg, &gamma[i]);
        svint32_t g = svunpklo_s32(g_s16);

        svint16_t b_s16 = svld1_s16(pg, &beta[i]);
        svint32_t b = svunpklo_s32(b_s16);

        // Normalize: (x - mean) * inv_std
        svint32_t norm = svsub_s32_x(pg, x, mean_v);
        norm = svmul_s32_x(pg, norm, inv_std_v);
        norm = svasr_n_s32_x(pg, norm, 16); // Q16.16 -> Q0

        // Apply affine: norm * gamma + beta
        svint32_t y = svmul_s32_x(pg, norm, g);
        y = svasr_n_s32_x(pg, y, 14); // Scale down for INT16
        y = svadd_s32_x(pg, y, b);

        // Clamp to INT16 range
        y = svmax_n_s32_x(pg, y, -32768);
        y = svmin_n_s32_x(pg, y, 32767);

        // Store as INT16
        svint16_t y_s16 = svuzp1_s16(svreinterpret_s16_s32(y), svreinterpret_s16_s32(y));
        svst1_s16(pg, &output[i], y_s16);
        i += 16;
    }

    // Process remaining elements
    while (i < N) {
        int32_t x = input[i];
        int32_t norm = ((x - mean) * inv_std) >> 16;
        int32_t y = ((norm * gamma[i]) >> 14) + beta[i];

        // Clamp
        if (y < -32768) y = -32768;
        if (y > 32767) y = 32767;
        output[i] = (int16_t)y;
        i++;
    }
}

void layernorm_int16_noaffine(
    const int16_t* input,
    int16_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N)
{
    // Use gamma=1, beta=0
    int16_t* gamma = (int16_t*)malloc(N * sizeof(int16_t));
    int16_t* beta = (int16_t*)malloc(N * sizeof(int16_t));

    for (size_t i = 0; i < N; i++) {
        gamma[i] = 16384; // ~1.0 in Q14 format
        beta[i] = 0;
    }

    layernorm_int16(input, output, gamma, beta, epsilon, input_scale, output_scale, N);

    free(gamma);
    free(beta);
}

// ============================================================================
// INT16 RMSNorm
// ============================================================================

void rmsnorm_int16(
    const int16_t* input,
    int16_t* output,
    const int16_t* gamma,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N)
{
    // Step 1: Compute RMS = sqrt(mean(x^2) + epsilon)
    int64_t sq_sum = 0;
    size_t i = 0;

    svint64_t sq_vec = svdup_n_s64(0);

    // Process 32 elements at a time
    while (i + 32 <= N) {
        svbool_t pg = svptrue_b16();
        svint16_t v = svld1_s16(pg, &input[i]);

        // Extend to INT32
        svint32_t v0 = svunpklo_s32(v);
        svint32_t v1 = svunpkhi_s32(v);

        // Extend to INT64 and compute x^2
        svint64_t v0_lo = svunpklo_s64(v0);
        svint64_t v0_hi = svunpkhi_s64(v0);
        svint64_t v1_lo = svunpklo_s64(v1);
        svint64_t v1_hi = svunpkhi_s64(v1);

        v0_lo = svmul_s64_x(pg, v0_lo, v0_lo);
        v0_hi = svmul_s64_x(pg, v0_hi, v0_hi);
        v1_lo = svmul_s64_x(pg, v1_lo, v1_lo);
        v1_hi = svmul_s64_x(pg, v1_hi, v1_hi);

        sq_vec = svadd_s64_x(pg, sq_vec, v0_lo);
        sq_vec = svadd_s64_x(pg, sq_vec, v0_hi);
        sq_vec = svadd_s64_x(pg, sq_vec, v1_lo);
        sq_vec = svadd_s64_x(pg, sq_vec, v1_hi);

        i += 32;
    }

    // Horizontal reduction
    svbool_t pg = svptrue_b64();
    sq_sum += svaddv_s64(pg, sq_vec);

    // Remaining elements
    while (i < N) {
        int64_t x = input[i];
        sq_sum += x * x;
        i++;
    }

    // Mean squared value
    int32_t mean_sq = (int32_t)(sq_sum / N);

    // Compute 1/sqrt(mean_sq + epsilon)
    int32_t inv_rms = fast_invsqrt_int32(mean_sq + epsilon);

    // Step 2: Normalize and apply gamma
    i = 0;

    svint32_t inv_rms_v = svdup_n_s32(inv_rms);

    while (i + 16 <= N) {
        svbool_t pg = svptrue_pat_b32(SV_VL16);

        // Load input
        svint16_t x_s16 = svld1_s16(pg, &input[i]);
        svint32_t x = svunpklo_s32(x_s16);

        // Load gamma
        svint16_t g_s16 = svld1_s16(pg, &gamma[i]);
        svint32_t g = svunpklo_s32(g_s16);

        // Normalize: x * inv_rms
        svint32_t norm = svmul_s32_x(pg, x, inv_rms_v);
        norm = svasr_n_s32_x(pg, norm, 16); // Q16.16 -> Q0

        // Apply gamma: norm * gamma
        svint32_t y = svmul_s32_x(pg, norm, g);
        y = svasr_n_s32_x(pg, y, 14); // Scale down for INT16

        // Clamp to INT16 range
        y = svmax_n_s32_x(pg, y, -32768);
        y = svmin_n_s32_x(pg, y, 32767);

        // Store as INT16
        svint16_t y_s16 = svuzp1_s16(svreinterpret_s16_s32(y), svreinterpret_s16_s32(y));
        svst1_s16(pg, &output[i], y_s16);
        i += 16;
    }

    // Remaining elements
    while (i < N) {
        int32_t x = input[i];
        int32_t norm = (x * inv_rms) >> 16;
        int32_t y = (norm * gamma[i]) >> 14;

        // Clamp
        if (y < -32768) y = -32768;
        if (y > 32767) y = 32767;
        output[i] = (int16_t)y;
        i++;
    }
}

void rmsnorm_int16_noaffine(
    const int16_t* input,
    int16_t* output,
    int32_t epsilon,
    float input_scale,
    float output_scale,
    size_t N)
{
    // Use gamma=1
    int16_t* gamma = (int16_t*)malloc(N * sizeof(int16_t));

    for (size_t i = 0; i < N; i++) {
        gamma[i] = 16384; // ~1.0 in Q14 format
    }

    rmsnorm_int16(input, output, gamma, epsilon, input_scale, output_scale, N);

    free(gamma);
}
