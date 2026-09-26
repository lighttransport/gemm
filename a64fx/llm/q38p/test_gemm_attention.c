#define _POSIX_C_SOURCE 200809L
#define HD 256
#include "q38p_gemm_attention.h"
#include "../q38d/q38d_attention.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void q38p_qk6x4(const float *, const float *, float *, long, float);
static uint32_t seed = 9;
static float rnd(void) { seed = seed * 1664525u + 1013904223u; return (int32_t)seed * 0x1p-31f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(void) {
    const int maxn = 32768, nqmax = 16, qstride = 24 * HD;
    float *q = malloc((size_t)nqmax * qstride * 4);
    float *kv = malloc((size_t)(maxn + 32) * HD * 4);
    float *sc = malloc((size_t)nqmax * 6 * maxn * 4);
    float *ref = malloc((size_t)nqmax * 6 * maxn * 4);
    float *a = malloc((size_t)8 * 257 * 12 * 4);
    float *b = malloc((size_t)1025 * 32 * 4), *c = malloc(12 * 32 * 4);
    float *out = malloc((size_t)nqmax * 6 * HD * sizeof(float));
    assert(q && kv && sc && ref && a && b && c && out);
    for (int i = 0; i < nqmax * qstride; i++) q[i] = rnd();
    for (int i = 0; i < (maxn + 32) * HD; i++) kv[i] = rnd();
    int ns[] = {1, 2, 3, 8, 15, 16};
    for (int ni = 0; ni < 6; ni++) {
        int nq = ns[ni], nt = 73;
        q38p_pack_queries(q + 6 * HD, nq, qstride, a);
        q38p_scores_gemm(kv, 0, nt, a, nq, ref, nt, b, c, NULL);
        float *kt = malloc((size_t)3 * (HD + 1) * 32 * sizeof(float));
        assert(kt);
        for (int k = 0; k < nt; k += 32)
            q38p_pack_keys(kv, k, nt - k < 32 ? nt - k : 32, kt + (size_t)(k / 32) * (HD + 1) * 32);
        q38p_scores_gemm(kv, 0, nt, a, nq, sc, nt, b, c, kt);
        assert(!memcmp(sc, ref, (size_t)nq * 6 * nt * sizeof(float)));
        free(kt);
        for (int t = 0; t < nq; t++) for (int h = 0; h < 6; h++) for (int k = 0; k < nt; k++) {
            float v = 0;
            for (int d = 0; d < HD; d++) v = fmaf(q[t * qstride + (6 + h) * HD + d], kv[k * HD + d], v);
            assert(sc[(t * 6 + h) * nt + k] == v * 0.0625f);
        }
        const int stride = nt + 16;
        float *prob = malloc((size_t)nq * 6 * stride * sizeof(float));
        assert(prob);
        for (int h = 0; h < nq * 6; h++)
            for (int k = 0; k < nt; k++) prob[(size_t)h * stride + k] = fabsf(rnd());
        for (int first = 0; first <= 57; first += 57) {
            for (int k0 = 0; k0 < nt; k0 += 32) {
                int k1 = k0 + 32 < nt ? k0 + 32 : nt;
                q38p_pv_gemm(kv, k0, k1, prob, stride, nq, first, out, a, b);
            }
            for (int t = 0; t < nq; t++) for (int h = 0; h < 6; h++) for (int d = 0; d < HD; d++) {
                float v = 0;
                for (int k = 0; k <= first + t && k < nt; k++)
                    v = fmaf(prob[((size_t)t * 6 + h) * stride + k], kv[(size_t)k * HD + d], v);
                assert(out[((size_t)t * 6 + h) * HD + d] == v);
            }
        }
        free(prob);
    }
    puts("PASS: packed/cached QK agrees with sequential-FMA reference for odd query counts and key tails");
    const int nq = 8, nt = maxn;
    q38p_pack_queries(q, nq, qstride, a);
    double t = now();
    for (int rep = 0; rep < 3; rep++)
        for (int k = 0; k < nt; k += 32) for (int i = 0; i < nq; i++) for (int j = k; j < k + 32; j += 4)
            q38p_qk6x4(q + i * qstride, kv + (size_t)j * HD, ref + (size_t)i * 6 * nt + j, nt, 0.0625f);
    printf("qk old ms=%.3f\n", (now() - t) * 1e3 / 3);
    t = now();
    for (int rep = 0; rep < 3; rep++) q38p_scores_gemm(kv, 0, nt, a, nq, sc, nt, b, c, NULL);
    printf("qk packed ms=%.3f\n", (now() - t) * 1e3 / 3);
    float *cache = malloc((size_t)(nt / 32) * (HD + 1) * 32 * sizeof(float));
    assert(cache);
    for (int k = 0; k < nt; k += 32) q38p_pack_keys(kv, k, 32, cache + (size_t)(k / 32) * (HD + 1) * 32);
    t = now();
    for (int rep = 0; rep < 3; rep++) q38p_scores_gemm(kv, 0, nt, a, nq, sc, nt, b, c, cache);
    printf("qk cached ms=%.3f\n", (now() - t) * 1e3 / 3);
    free(cache);
    double max_diff = 0;
    for (int i = 0; i < nq * 6 * nt; i++) { double d = fabs(sc[i] - ref[i]); if (d > max_diff) max_diff = d; sc[i] = fabsf(sc[i]); }
    printf("qk max_abs_difference=%.9g\n", max_diff);
    free(q); free(kv); free(sc); free(ref); free(a); free(b); free(c); free(out);
    return 0;
}
