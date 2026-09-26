#ifndef Q38P_GEMM_ATTENTION_H
#define Q38P_GEMM_ATTENTION_H
#include <arm_sve.h>
#include <stddef.h>
#include <string.h>
#include "../q38d/q38d_attention.h"
/* A64FX packed FP32 tile; inputs have one padded row for software pipelining.
 * QK accumulates sequentially across HD (different rounding from the older
 * 16-lane reduction). PV/softmax and decode arithmetic are unchanged. */
void q38p_sgemm12x32(const float *a, const float *b, float *c, long k, long stride, long reset);

static void q38p_pack_queries(const float *q, int nq, int qstride, float *a) {
    for (int pair = 0; pair < (nq + 1) / 2; pair++) {
        float *dst = a + (size_t)pair * (HD + 1) * 12;
        for (int d = 0; d < HD; d++)
            for (int h = 0; h < 12; h++) {
                int t = 2 * pair + h / 6;
                dst[d * 12 + h] = t < nq ? q[(size_t)t * qstride + (h % 6) * HD + d] : 0;
            }
        memset(dst + HD * 12, 0, 12 * sizeof(float));
    }
}

static void q38p_pack_keys(const float *K, int kb, int nc, float *b) {
    const svbool_t all = svptrue_b32();
    const svuint32_t index = svmul_n_u32_x(all, svindex_u32(0, 1), HD);
    svbool_t p0 = svwhilelt_b32(0, nc), p1 = svwhilelt_b32(16, nc);
    for (int d = 0; d < HD; d++) {
        const float *src = K + (size_t)kb * HD + d;
        svst1_f32(all, b + d * 32, svld1_gather_u32index_f32(p0, src, index));
        svst1_f32(all, b + d * 32 + 16, svld1_gather_u32index_f32(p1, src + 16 * HD, index));
    }
    memset(b + HD * 32, 0, 32 * sizeof(float));
}

/* k0 must be a multiple of 32; cache panels are indexed from key zero. */
static void q38p_scores_gemm(const float *K, int k0, int k1, const float *a,
                             int nq, float *sc, int stride, float *b, float *ctmp, const float *cache) {
    const svbool_t all = svptrue_b32();
    for (int kb = k0; kb < k1; kb += 32) {
        int nc = k1 - kb < 32 ? k1 - kb : 32;
        svbool_t p0 = svwhilelt_b32(0, nc), p1 = svwhilelt_b32(16, nc);
        if (!cache) q38p_pack_keys(K, kb, nc, b);
        const float *panel = cache ? cache + (size_t)(kb / 32) * (HD + 1) * 32 : b;
        for (int pair = 0; pair < (nq + 1) / 2; pair++) {
            q38p_sgemm12x32(a + (size_t)pair * (HD + 1) * 12, panel, ctmp, HD, 32, 1);
            for (int h = 0; h < 12 && pair * 12 + h < nq * 6; h++) {
                float *dst = sc + (size_t)(pair * 12 + h) * stride + kb;
                svst1_f32(p0, dst, svmul_n_f32_x(all, svld1_f32(all, ctmp + h * 32), 0.0625f));
                svst1_f32(p1, dst + 16, svmul_n_f32_x(all, svld1_f32(all, ctmp + h * 32 + 16), 0.0625f));
            }
        }
    }
}

/* All nq queries share a time block. Mask future keys before packing P.
 * Accumulation keeps the original increasing-time FMA order. */
static void q38p_pv_gemm(const float *V, int k0, int k1, const float *sc,
                         int stride, int nq, int first_pos, float *out, float *a, float *b) {
    const svbool_t all = svptrue_b32();
    int nk = k1 - k0;
    for (int pair = 0; pair < (nq + 1) / 2; pair++) {
        float *dst = a + (size_t)pair * (nk + 1) * 12;
        for (int t = 0; t < nk; t++)
            for (int h = 0; h < 12; h++) {
                int q = pair * 2 + h / 6;
                dst[t * 12 + h] = q < nq && k0 + t <= first_pos + q ?
                    sc[(size_t)(pair * 12 + h) * stride + k0 + t] : 0;
            }
        memset(dst + nk * 12, 0, 12 * sizeof(float));
    }
    for (int d = 0; d < HD; d += 32) {
        for (int t = 0; t < nk; t++) {
            const float *src = V + (size_t)(k0 + t) * HD + d;
            svst1_f32(all, b + t * 32, svld1_f32(all, src));
            svst1_f32(all, b + t * 32 + 16, svld1_f32(all, src + 16));
        }
        memset(b + nk * 32, 0, 32 * sizeof(float));
        for (int pair = 0; pair < (nq + 1) / 2; pair++)
            q38p_sgemm12x32(a + (size_t)pair * (nk + 1) * 12, b,
                            out + (size_t)pair * 12 * HD + d, nk, HD, k0 == 0);
    }
}
#endif
