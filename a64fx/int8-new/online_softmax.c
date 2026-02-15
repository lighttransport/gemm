// online_softmax.c
// Online softmax normalization for FlashAttention-style tiled computation

#include "online_softmax.h"
#include "softmax_exp2.h"
#include "fused_gemm.h"  // For FUSED_MR, FUSED_LB
#include <arm_sve.h>
#include <math.h>

// ============================================================================
// Scalar implementation of online softmax update
// ============================================================================
void online_softmax_update(online_softmax_state_t* state,
                            const int32_t* S_tile,
                            const int8_t* V_tile,
                            int lb, float scale) {
    int mr = state->mr;
    int d = state->d;
    float* row_max = state->row_max;
    float* row_sum = state->row_sum;
    float* O_acc = state->O_acc;

    float scale_log2e = scale * LOG2_E;

    // Temporary buffer for exp values of this tile
    // Use fixed max size instead of VLA to avoid undefined behavior with alignment
    float exp_tile[FUSED_MR * FUSED_LB];

    for (int m = 0; m < mr; m++) {
        const int32_t* S_row = S_tile + m * lb;
        float* exp_row = exp_tile + m * lb;

        // Step 1: Find local max for this row
        float local_max = -1e30f;
        for (int l = 0; l < lb; l++) {
            float val = (float)S_row[l] * scale_log2e;
            if (val > local_max) local_max = val;
        }

        // Step 2: Update global max
        float old_max = row_max[m];
        float new_max = (local_max > old_max) ? local_max : old_max;
        row_max[m] = new_max;

        // Step 3: Compute correction factor for old values
        // alpha = exp2(old_max - new_max)
        float alpha = (old_max > -1e20f) ? fast_exp2f(old_max - new_max) : 0.0f;

        // Step 4: Scale existing sum and output
        row_sum[m] *= alpha;
        float* O_row = O_acc + m * d;
        for (int n = 0; n < d; n++) {
            O_row[n] *= alpha;
        }

        // Step 5: Compute exp2(S - new_max) for new tile
        float local_sum = 0.0f;
        for (int l = 0; l < lb; l++) {
            float val = (float)S_row[l] * scale_log2e - new_max;
            float exp_val = fast_exp2f(val);
            exp_row[l] = exp_val;
            local_sum += exp_val;
        }

        // Step 6: Accumulate sum
        row_sum[m] += local_sum;

        // Step 7: Accumulate output: O += exp_tile @ V
        // V_tile is [lb, d] row-major (or packed)
        // For simplicity, assume V_tile is [lb, d] row-major here
        for (int n = 0; n < d; n++) {
            float sum = 0.0f;
            for (int l = 0; l < lb; l++) {
                sum += exp_row[l] * (float)V_tile[l * d + n];
            }
            O_row[n] += sum;
        }
    }
}

// ============================================================================
// SVE-optimized online softmax update
// ============================================================================
void online_softmax_update_sve(online_softmax_state_t* state,
                                const int32_t* S_tile,
                                const int8_t* V_tile,
                                int lb, float scale) {
    int mr = state->mr;
    int d = state->d;
    float* row_max = state->row_max;
    float* row_sum = state->row_sum;
    float* O_acc = state->O_acc;

    float scale_log2e = scale * LOG2_E;
    svbool_t pg = svptrue_b32();

    // Temporary buffer for exp values
    // Use fixed max size instead of VLA to avoid undefined behavior with alignment
    float exp_tile[FUSED_MR * FUSED_LB] __attribute__((aligned(64)));

    for (int m = 0; m < mr; m++) {
        const int32_t* S_row = S_tile + m * lb;
        float* exp_row = exp_tile + m * lb;

        // Step 1: Find local max using SVE
        svfloat32_t vmax = svdup_f32(-1e30f);
        int l = 0;
        while (l < lb) {
            svbool_t pg_l = svwhilelt_b32(l, lb);
            svint32_t vs = svld1_s32(pg_l, S_row + l);
            svfloat32_t vf = svcvt_f32_s32_x(pg_l, vs);
            vf = svmul_f32_x(pg_l, vf, svdup_f32(scale_log2e));
            vmax = svmax_f32_m(pg_l, vmax, vf);
            l += svcntw();
        }
        float local_max = svmaxv_f32(pg, vmax);

        // Step 2: Update global max
        float old_max = row_max[m];
        float new_max = fmaxf(local_max, old_max);
        row_max[m] = new_max;

        // Step 3: Compute correction factor
        float alpha = (old_max > -1e20f) ? fast_exp2f(old_max - new_max) : 0.0f;

        // Step 4: Scale existing sum and output with SVE
        row_sum[m] *= alpha;

        float* O_row = O_acc + m * d;
        svfloat32_t valpha = svdup_f32(alpha);
        int n = 0;
        while (n < d) {
            svbool_t pg_n = svwhilelt_b32(n, d);
            svfloat32_t vo = svld1_f32(pg_n, O_row + n);
            vo = svmul_f32_x(pg_n, vo, valpha);
            svst1_f32(pg_n, O_row + n, vo);
            n += svcntw();
        }

        // Step 5: Compute exp2(S - new_max) with SVE
        svfloat32_t vnew_max = svdup_f32(new_max);
        svfloat32_t vsum = svdup_f32(0.0f);

        l = 0;
        while (l < lb) {
            svbool_t pg_l = svwhilelt_b32(l, lb);
            svint32_t vs = svld1_s32(pg_l, S_row + l);
            svfloat32_t vf = svcvt_f32_s32_x(pg_l, vs);
            vf = svmul_f32_x(pg_l, vf, svdup_f32(scale_log2e));
            vf = svsub_f32_x(pg_l, vf, vnew_max);

            // Fast exp2 approximation
            // Clamp to valid range
            vf = svmax_f32_x(pg_l, vf, svdup_f32(-126.0f));
            vf = svmin_f32_x(pg_l, vf, svdup_f32(127.0f));

            // Split into integer and fractional parts
            svfloat32_t vbiased = svadd_f32_x(pg_l, vf, svdup_f32(126.0f));
            svint32_t vi = svcvt_s32_f32_x(pg_l, vbiased);
            svfloat32_t vfloor = svcvt_f32_s32_x(pg_l, vi);
            svfloat32_t vfrac = svsub_f32_x(pg_l, vf, svsub_f32_x(pg_l, vfloor, svdup_f32(126.0f)));

            // Polynomial for 2^frac
            svfloat32_t vp = svmla_f32_x(pg_l, svdup_f32(0.2402265f), vfrac, svdup_f32(0.0554913f));
            vp = svmla_f32_x(pg_l, svdup_f32(0.6931472f), vfrac, vp);
            vp = svmla_f32_x(pg_l, svdup_f32(1.0f), vfrac, vp);

            // 2^floor via exponent manipulation
            svint32_t vexp = svadd_s32_x(pg_l, vi, svdup_s32(1));
            vexp = svlsl_n_s32_x(pg_l, vexp, 23);
            svfloat32_t vpow2 = svreinterpret_f32_s32(vexp);

            // Combine
            svfloat32_t vexp_val = svmul_f32_x(pg_l, vpow2, vp);

            // Store and accumulate sum
            svst1_f32(pg_l, exp_row + l, vexp_val);
            vsum = svadd_f32_m(pg_l, vsum, vexp_val);

            l += svcntw();
        }

        row_sum[m] += svaddv_f32(pg, vsum);

        // Step 7: Accumulate output O += exp_tile @ V
        // V_tile[lb, d] - process d columns at a time
        for (n = 0; n < d; n++) {
            float sum = 0.0f;
            // Vectorize over lb dimension
            svfloat32_t vacc = svdup_f32(0.0f);
            l = 0;
            while (l < lb) {
                svbool_t pg_l = svwhilelt_b32(l, lb);

                // Load exp values
                svfloat32_t vexp = svld1_f32(pg_l, exp_row + l);

                // Load V values (int8 -> float)
                // V_tile is assumed [lb, d] row-major
                // We need V_tile[l:l+vl, n], which is strided access
                // For simplicity, use scalar gather or reorganize V layout
                // Here we do scalar for correctness
                int cnt = svcntp_b32(pg_l, pg_l);
                float v_vals[16];
                for (int i = 0; i < cnt && (l + i) < lb; i++) {
                    v_vals[i] = (float)V_tile[(l + i) * d + n];
                }
                svfloat32_t vv = svld1_f32(pg_l, v_vals);

                vacc = svmla_f32_m(pg_l, vacc, vexp, vv);
                l += svcntw();
            }
            O_row[n] += svaddv_f32(pg, vacc);
        }
    }
}

// ============================================================================
// Finalize: divide by row sums and convert to int32
// ============================================================================
void online_softmax_finalize(const online_softmax_state_t* state,
                              int32_t* O_out, int ldo) {
    int mr = state->mr;
    int d = state->d;
    const float* row_sum = state->row_sum;
    const float* O_acc = state->O_acc;

    for (int m = 0; m < mr; m++) {
        float inv_sum = 1.0f / row_sum[m];
        const float* O_row = O_acc + m * d;

        for (int n = 0; n < d; n++) {
            float val = O_row[n] * inv_sum;
            O_out[m * ldo + n] = (int32_t)(val + (val >= 0 ? 0.5f : -0.5f));
        }
    }
}

// ============================================================================
// Finalize with quantization to int8
// ============================================================================
void online_softmax_finalize_int8(const online_softmax_state_t* state,
                                   int8_t* O_out, int ldo, float out_scale) {
    int mr = state->mr;
    int d = state->d;
    const float* row_sum = state->row_sum;
    const float* O_acc = state->O_acc;

    for (int m = 0; m < mr; m++) {
        float inv_sum = out_scale / row_sum[m];
        const float* O_row = O_acc + m * d;

        for (int n = 0; n < d; n++) {
            float val = O_row[n] * inv_sum;
            int32_t ival = (int32_t)(val + (val >= 0 ? 0.5f : -0.5f));
            if (ival > 127) ival = 127;
            if (ival < -128) ival = -128;
            O_out[m * ldo + n] = (int8_t)ival;
        }
    }
}
