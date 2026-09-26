#ifndef Q38P_PV_INT16_H
#define Q38P_PV_INT16_H
#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define Q38I_BK 512
#define Q38I_R 4
#define Q38I_T 5
#define Q38I_ST (Q38I_BK / 4)
#define Q38I_DT (HD / 32)
#define Q38I_WS (Q38I_ST * Q38I_R * 32)
#define Q38I_NGMAX ((16 * 6 + Q38I_T - 1) / Q38I_T)
#define Q38I_AS (Q38I_NGMAX * Q38I_ST * Q38I_T * 4)
#define Q38I_OS (Q38I_NGMAX * Q38I_R * Q38I_T * 8)

/* Cache one Q38I_BK-position × 32-column V panel in the existing int16 SDOT
 * weight layout. A partial panel is refreshed when new KV arrives. */
static void q38p_pack_v16(const float *v, int block, int valid, int d_tile,
                           int16_t *dst, float *vscale) {
    const svbool_t all = svptrue_b32();
    const int d0 = d_tile * 32;
    svfloat32_t vmax = svdup_n_f32(0);
    for (int k = 0; k < valid; k++) {
        const float *src = v + ((size_t)block * Q38I_BK + k) * HD + d0;
        vmax = svmax_f32_x(all, vmax, svabs_f32_x(all, svld1_f32(all, src)));
        vmax = svmax_f32_x(all, vmax, svabs_f32_x(all, svld1_f32(all, src + 16)));
    }
    float mx = svmaxv_f32(all, vmax);
    float scale = mx > 0 ? 32767.0f / mx : 1.0f;
    *vscale = scale;
    for (int st = 0; st < Q38I_ST; st++) {
        int16_t tmp[4][32] __attribute__((aligned(64)));
        for (int c = 0; c < 4; c++) {
            int k = st * 4 + c;
            if (k >= valid) { memset(tmp[c], 0, sizeof tmp[c]); continue; }
            const float *src = v + ((size_t)block * Q38I_BK + k) * HD + d0;
            for (int j = 0; j < 2; j++) {
                svfloat32_t x = svmul_n_f32_x(all, svld1_f32(all, src + 16 * j), scale);
                x = svmax_n_f32_x(all, svmin_n_f32_x(all, x, 32767.0f), -32767.0f);
                svint32_t q = svcvt_s32_f32_x(all, svrintn_f32_x(all, x));
                svst1h_s32(all, tmp[c] + 16 * j, q);
            }
        }
        for (int r = 0; r < 32; r++)
            for (int c = 0; c < 4; c++)
                dst[(st * Q38I_R + r / 8) * 32 + (r % 8) * 4 + c] = tmp[c][r];
    }
}

/* A second int16 panel captures the rounding residual of the first panel.
 * This is optional because it adds one dot-product pass per PV tile. */
static void q38p_pack_v16_residual(const float *v, int block, int valid,
                                    int d_tile, float base_scale, int16_t *dst,
                                    float *res_scale) {
    int d0 = d_tile * 32;
    float mx = 0.0f;
    for (int k = 0; k < valid; k++)
        for (int r = 0; r < 32; r++) {
            float x = v[((size_t)block * Q38I_BK + k) * HD + d0 + r];
            float q = nearbyintf(fmaxf(-32767.0f, fminf(32767.0f, x * base_scale)));
            float e = x - q / base_scale;
            float a = fabsf(e);
            if (a > mx) mx = a;
        }
    float scale = mx > 0 ? 32767.0f / mx : 1.0f;
    *res_scale = scale;
    for (int st = 0; st < Q38I_ST; st++)
        for (int r = 0; r < 32; r++)
            for (int c = 0; c < 4; c++) {
                int k = st * 4 + c;
                int16_t qres = 0;
                if (k < valid) {
                    float x = v[((size_t)block * Q38I_BK + k) * HD + d0 + r];
                    float q = nearbyintf(fmaxf(-32767.0f, fminf(32767.0f, x * base_scale)));
                    float e = x - q / base_scale;
                    qres = (int16_t)nearbyintf(fmaxf(-32767.0f, fminf(32767.0f, e * scale)));
                }
                dst[(st * Q38I_R + r / 8) * 32 + (r % 8) * 4 + c] = qres;
            }
}

/* P rows are query/head-major. Every fifth head occupies one SDOT token
 * group; mask future keys in the final causal panel. */
static void q38p_pack_p16(const float *p, int stride, int block, int nq,
                           int first_pos, int16_t *dst, float *pscale) {
    const svbool_t all = svptrue_b32();
    const int ng = (nq * 6 + Q38I_T - 1) / Q38I_T;
    for (int h = 0; h < ng * Q38I_T; h++) {
        int valid = h < nq * 6 ? first_pos + h / 6 + 1 - block * Q38I_BK : 0;
        if (valid > Q38I_BK) valid = Q38I_BK;
        if (valid <= 0) { pscale[h] = 1.0f; continue; }
        const float *src = p + (size_t)h * stride + block * Q38I_BK;
        svfloat32_t vmax = svdup_n_f32(0);
        for (int k = 0; k < valid; k += 16) {
            svbool_t pg = svwhilelt_b32(k, valid);
            vmax = svmax_f32_x(all, vmax, svld1_f32(pg, src + k));
        }
        float mx = svmaxv_f32(all, vmax);
        pscale[h] = mx > 1e-30f ? 32767.0f / mx : 1.0f;
    }
    for (int g = 0; g < ng; g++)
        for (int k16 = 0; k16 < Q38I_BK; k16 += 16) {
            int16_t tmp[Q38I_T][16] __attribute__((aligned(64)));
            for (int tt = 0; tt < Q38I_T; tt++) {
                int h = g * Q38I_T + tt;
                int valid = h < nq * 6 ? first_pos + h / 6 + 1 - block * Q38I_BK - k16 : 0;
                if (valid <= 0) { memset(tmp[tt], 0, sizeof tmp[tt]); continue; }
                svbool_t pg = svwhilelt_b32(0, valid < 16 ? valid : 16);
                const float *src = p + (size_t)h * stride + block * Q38I_BK + k16;
                svfloat32_t x = svld1_f32(pg, src);
                x = svmax_n_f32_x(all, svmin_n_f32_x(all, x, 1.0f), 0.0f);
                svint32_t q = svcvt_s32_f32_x(all, svrintn_f32_x(all, svmul_n_f32_x(all, x, pscale[h])));
                svst1h_s32(all, tmp[tt], q);
            }
            for (int u = 0; u < 4; u++)
                for (int tt = 0; tt < Q38I_T; tt++)
                    memcpy(dst + (((g * Q38I_ST + k16 / 4 + u) * Q38I_T + tt) * 4),
                           tmp[tt] + u * 4, 4 * sizeof(int16_t));
        }
}

/* Quantize the error left by the first probability panel. Its scale is
 * independent for every query/head in each key block. */
static void q38p_pack_p16_residual(const float *p, int stride, int block,
                                    int nq, int first_pos, const float *pscale,
                                    int16_t *dst, float *res_scale) {
    const svbool_t all = svptrue_b32();
    const int ng = (nq * 6 + Q38I_T - 1) / Q38I_T;
    for (int h = 0; h < ng * Q38I_T; h++) {
        int valid = h < nq * 6 ? first_pos + h / 6 + 1 - block * Q38I_BK : 0;
        if (valid > Q38I_BK) valid = Q38I_BK;
        /* The base round-off is bounded by about 0.5 / pscale. A fixed
         * second scale avoids another full scan of the probability row. */
        res_scale[h] = valid > 0 ? fminf(pscale[h] * 65534.0f, 1e30f) : 1.0f;
    }
    for (int g = 0; g < ng; g++)
        for (int k16 = 0; k16 < Q38I_BK; k16 += 16) {
            int16_t tmp[Q38I_T][16] __attribute__((aligned(64)));
            for (int tt = 0; tt < Q38I_T; tt++) {
                int h = g * Q38I_T + tt;
                int valid = h < nq * 6 ? first_pos + h / 6 + 1 - block * Q38I_BK - k16 : 0;
                if (valid <= 0) { memset(tmp[tt], 0, sizeof tmp[tt]); continue; }
                svbool_t pg = svwhilelt_b32(0, valid < 16 ? valid : 16);
                const float *src = p + (size_t)h * stride + block * Q38I_BK + k16;
                svfloat32_t x = svld1_f32(pg, src);
                x = svmax_n_f32_x(all, svmin_n_f32_x(all, x, 1.0f), 0.0f);
                svfloat32_t q = svrintn_f32_x(all, svmul_n_f32_x(all, x, pscale[h]));
                svfloat32_t e = svsub_f32_x(all, x, svmul_n_f32_x(all, q, 1.0f / pscale[h]));
                e = svmul_n_f32_x(all, e, res_scale[h]);
                e = svmax_n_f32_x(all, svmin_n_f32_x(all, e, 32767.0f), -32767.0f);
                svst1h_s32(all, tmp[tt], svcvt_s32_f32_x(all, svrintn_f32_x(all, e)));
            }
            for (int u = 0; u < 4; u++)
                for (int tt = 0; tt < Q38I_T; tt++)
                    memcpy(dst + (((g * Q38I_ST + k16 / 4 + u) * Q38I_T + tt) * 4),
                           tmp[tt] + u * 4, 4 * sizeof(int16_t));
        }
}

static void q38p_pv_int16_eval(const int16_t *wcache, const float *vscale,
                                const int16_t *wres, const float *res_scale,
                                const float *p, int stride, int nq, int first_pos,
                                float *out, int16_t *apack, int16_t *apack_res,
                                double *acc) {
    int ng = (nq * 6 + Q38I_T - 1) / Q38I_T;
    int blocks = (first_pos + nq + Q38I_BK - 1) / Q38I_BK;
    memset(acc, 0, (size_t)Q38I_DT * ng * Q38I_R * Q38I_T * 8 * sizeof(double));
    for (int b = 0; b < blocks; b++) {
        float pscale[Q38I_NGMAX * Q38I_T];
        q38p_pack_p16(p, stride, b, nq, first_pos, apack, pscale);
        float pres_scale[Q38I_NGMAX * Q38I_T];
        if (apack_res)
            q38p_pack_p16_residual(p, stride, b, nq, first_pos, pscale,
                                    apack_res, pres_scale);
        for (int d = 0; d < Q38I_DT; d++) {
            double s[Q38I_NGMAX * Q38I_T];
            double factor = 1.0 / vscale[(size_t)b * Q38I_DT + d];
            for (int i = 0; i < ng * Q38I_T; i++) s[i] = factor / pscale[i];
            q38p_mk_r4t5(wcache + ((size_t)b * Q38I_DT + d) * Q38I_WS,
                          apack, s, acc + (size_t)d * ng * Q38I_R * Q38I_T * 8,
                          ng, Q38I_ST);
            if (wres) {
                double factor_res = 1.0 / res_scale[(size_t)b * Q38I_DT + d];
                for (int i = 0; i < ng * Q38I_T; i++) s[i] = factor_res / pscale[i];
                q38p_mk_r4t5(wres + ((size_t)b * Q38I_DT + d) * Q38I_WS,
                              apack, s, acc + (size_t)d * ng * Q38I_R * Q38I_T * 8,
                              ng, Q38I_ST);
            }
            if (apack_res) {
                for (int i = 0; i < ng * Q38I_T; i++) s[i] = factor / pres_scale[i];
                q38p_mk_r4t5(wcache + ((size_t)b * Q38I_DT + d) * Q38I_WS,
                              apack_res, s, acc + (size_t)d * ng * Q38I_R * Q38I_T * 8,
                              ng, Q38I_ST);
                /* P1*V1 is second order in the two int16 round-offs. */
            }
        }
    }
    for (int h = 0; h < nq * 6; h++)
        for (int d = 0; d < HD; d++) {
            int tile = d / 32, row = (d % 32) / 8, lane = d % 8;
            int g = h / Q38I_T, tt = h % Q38I_T;
            out[(size_t)h * HD + d] = (float)acc[(size_t)tile * ng * Q38I_R * Q38I_T * 8 +
                                                   (((g * Q38I_R + row) * Q38I_T + tt) * 8 + lane)];
        }
}
#endif
