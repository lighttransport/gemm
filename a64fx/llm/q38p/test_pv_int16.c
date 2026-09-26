#define HD 256
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void q38p_mk_r4t5(const void *, const void *, const double *, double *, long, long);
#include "q38p_pv_int16.h"

static unsigned rng = 1;
static float random_float(void) {
    rng = rng * 1664525u + 1013904223u;
    return (float)((int)(rng >> 8) - 8388608) * (1.0f / 8388608.0f);
}

static void check_case(int first, int nq) {
    const int extent = first + nq;
    const int blocks = (extent + Q38I_BK - 1) / Q38I_BK;
    float *v = malloc((size_t)extent * HD * sizeof(float));
    float *p = calloc((size_t)nq * 6 * extent, sizeof(float));
    int16_t *w = aligned_alloc(256, (size_t)blocks * Q38I_DT * Q38I_WS * sizeof(int16_t));
    float *scale = malloc((size_t)blocks * Q38I_DT * sizeof(float));
    int16_t *wres = aligned_alloc(256, (size_t)blocks * Q38I_DT * Q38I_WS * sizeof(int16_t));
    float *rscale = malloc((size_t)blocks * Q38I_DT * sizeof(float));
    int16_t *a = aligned_alloc(256, Q38I_AS * sizeof(int16_t));
    int16_t *a_res = aligned_alloc(256, Q38I_AS * sizeof(int16_t));
    double *acc = aligned_alloc(256, (size_t)Q38I_DT * Q38I_OS * sizeof(double));
    float *out = malloc((size_t)nq * 6 * HD * sizeof(float));
    float *base = malloc((size_t)nq * 6 * HD * sizeof(float));
    float *p_only = malloc((size_t)nq * 6 * HD * sizeof(float));
    float *both = malloc((size_t)nq * 6 * HD * sizeof(float));
    assert(v && p && w && scale && wres && rscale && a && a_res && acc);
    assert(out && base && p_only && both);
    for (size_t i = 0; i < (size_t)extent * HD; i++) v[i] = random_float();
    for (int h = 0; h < nq * 6; h++)
        for (int k = 0; k <= first + h / 6; k++)
            p[(size_t)h * extent + k] = (random_float() + 1.0f) * 0.5f;
    for (int b = 0; b < blocks; b++)
        for (int d = 0; d < Q38I_DT; d++) {
            int valid = extent - b * Q38I_BK;
            if (valid > Q38I_BK) valid = Q38I_BK;
            q38p_pack_v16(v, b, valid, d,
                          w + ((size_t)b * Q38I_DT + d) * Q38I_WS,
                          scale + (size_t)b * Q38I_DT + d);
            q38p_pack_v16_residual(v, b, valid, d,
                                    scale[(size_t)b * Q38I_DT + d],
                                    wres + ((size_t)b * Q38I_DT + d) * Q38I_WS,
                                    rscale + (size_t)b * Q38I_DT + d);
        }
    q38p_pv_int16_eval(w, scale, NULL, NULL, p, extent, nq, first, base, a, NULL, acc);
    q38p_pv_int16_eval(w, scale, wres, rscale, p, extent, nq, first, out, a, NULL, acc);
    q38p_pv_int16_eval(w, scale, NULL, NULL, p, extent, nq, first, p_only, a, a_res, acc);
    q38p_pv_int16_eval(w, scale, wres, rscale, p, extent, nq, first, both, a, a_res, acc);
    double max_error = 0, max_base_error = 0, max_p_error = 0, max_both_error = 0, max_ref = 0;
    double max_fp32_rounding = 0, max_both_fp32_error = 0;
    for (int h = 0; h < nq * 6; h++)
        for (int d = 0; d < HD; d++) {
            double ref = 0;
            float fp32 = 0;
            for (int k = 0; k <= first + h / 6; k++) {
                ref += (double)p[(size_t)h * extent + k] * v[(size_t)k * HD + d];
                fp32 = fmaf(p[(size_t)h * extent + k], v[(size_t)k * HD + d], fp32);
            }
            double err = fabs((double)out[(size_t)h * HD + d] - ref);
            double base_err = fabs((double)base[(size_t)h * HD + d] - ref);
            double p_err = fabs((double)p_only[(size_t)h * HD + d] - ref);
            double both_err = fabs((double)both[(size_t)h * HD + d] - ref);
            if (err > max_error) max_error = err;
            if (base_err > max_base_error) max_base_error = base_err;
            if (p_err > max_p_error) max_p_error = p_err;
            if (both_err > max_both_error) max_both_error = both_err;
            if (fabs(ref) > max_ref) max_ref = fabs(ref);
            double round_err = fabs((double)fp32 - ref);
            double both_fp32_err = fabs((double)both[(size_t)h * HD + d] - fp32);
            if (round_err > max_fp32_rounding) max_fp32_rounding = round_err;
            if (both_fp32_err > max_both_fp32_error) max_both_fp32_error = both_fp32_err;
        }
    printf("pv_int16: first=%d nq=%d base=%.6g value_res=%.6g prob_res=%.6g both=%.6g fp32_round=%.6g both_vs_fp32=%.6g max_ref=%.6g\n",
           first, nq, max_base_error, max_error, max_p_error, max_both_error,
           max_fp32_rounding, max_both_fp32_error, max_ref);
    assert(max_base_error < 0.02);
    assert(max_error < 0.02);
    assert(max_p_error < 0.02);
    assert(max_both_error < 1e-5);
    free(both); free(p_only); free(base); free(out); free(acc); free(a_res); free(a);
    free(rscale); free(wres); free(scale); free(w); free(p); free(v);
}

int main(void) {
    check_case(0, 1);
    check_case(509, 7);
    check_case(1023, 8);
    check_case(1535, 3);
    return 0;
}
