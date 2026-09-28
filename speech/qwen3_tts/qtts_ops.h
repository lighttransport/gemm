/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * qtts_ops.h - CPU math for the Qwen3-TTS runner (AVX2/FMA + OpenMP).
 *
 * Everything is row-major, activations channels-last ([T, C]).
 *   - qt_pack_b / qt_sgemm: C[M,N] (+)= A[M,K] * W[N,K]^T with W pre-packed
 *     into 16-column K-major panels (6x16 FMA micro-kernel).
 *   - qt_gemv_bf16: Y[M,N] = X[M,K] * W[N,K]^T for BF16 weights read in place
 *     (decode path; memory bound).
 *   - small elementwise helpers (RMSNorm, LayerNorm, RoPE, SiLU, GELU, attention).
 *
 * Single-file header: define QTTS_OPS_IMPLEMENTATION in one translation unit.
 */
#ifndef QTTS_OPS_H
#define QTTS_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QT_NR 16
#define QT_MR 6

/* Packed B operand: panels of QT_NR columns, each K-major. */
typedef struct {
    float *p;   /* [ceil(N/16)][K][16] */
    int n, k;
} qt_packed;

/* get(W, n, k) accessors for the common source layouts. */
typedef float (*qt_get_fn)(const void *w, int n, int k, const void *ctx);

void  qt_pack_b(qt_packed *dst, int n, int k, qt_get_fn get, const void *w, const void *ctx);
void  qt_pack_b_f32_nk(qt_packed *dst, const float *w, int n, int k);    /* W[n][k] */
void  qt_pack_b_bf16_nk(qt_packed *dst, const uint16_t *w, int n, int k);/* W[n][k] */
void  qt_packed_free(qt_packed *p);

/* C[m*ldc + n] = (accum ? C : 0) + sum_k A[m*lda + k] * B(n, k) */
void  qt_sgemm(int m, const float *a, int lda, const qt_packed *b, float *c, int ldc, int accum);

/* Y[m*n_out + n] = sum_k X[m*k + kk] * W[n*k + kk], W bf16, M <= any (grouped by 4). */
void  qt_gemv_bf16(int m, const float *x, const uint16_t *w, int n, int k, float *y);

static inline float qt_bf16_to_f32(uint16_t v) {
    union { uint32_t u; float f; } c; c.u = (uint32_t)v << 16; return c.f;
}

void  qt_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void  qt_rmsnorm_bf16w(float *out, const float *x, const uint16_t *w, int n, float eps);
void  qt_layernorm(float *out, const float *x, const float *w, const float *b, int n, float eps);
/* HF rotate_half RoPE on one head vector of size hd at position pos. */
void  qt_rope_neox(float *v, int hd, int pos, float theta);
float qt_silu(float x);
float qt_gelu_erf(float x);

/* Causal attention for nq query rows at absolute positions pos0..pos0+nq-1 against
 * a KV cache holding positions [0, pos0+nq). window>0 limits keys to (q-window, q].
 * q: [nq][nh*hd], k/v cache: [pos][nkv*hd], out: [nq][nh*hd]. */
void  qt_attention(float *out, const float *q, const float *kc, const float *vc,
                   int nq, int pos0, int nh, int nkv, int hd, int window);

/* Minimal .npy writer (f4 or i4) for fixture dumps. */
int   qt_npy_save_f32(const char *path, const float *d, int ndim, const int *dims);
int   qt_npy_save_i32(const char *path, const int32_t *d, int ndim, const int *dims);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef QTTS_OPS_IMPLEMENTATION

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *qt_aligned_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes ? bytes : 64)) return NULL;
    return p;
}

void qt_pack_b(qt_packed *dst, int n, int k, qt_get_fn get, const void *w, const void *ctx) {
    int np = (n + QT_NR - 1) / QT_NR;
    dst->n = n; dst->k = k;
    dst->p = (float *)qt_aligned_alloc((size_t)np * k * QT_NR * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int pi = 0; pi < np; pi++) {
        float *pp = dst->p + (size_t)pi * k * QT_NR;
        for (int kk = 0; kk < k; kk++)
            for (int j = 0; j < QT_NR; j++) {
                int nn = pi * QT_NR + j;
                pp[(size_t)kk * QT_NR + j] = nn < n ? get(w, nn, kk, ctx) : 0.0f;
            }
    }
}

static float qt__get_f32_nk(const void *w, int n, int k, const void *ctx) {
    return ((const float *)w)[(size_t)n * (size_t)(intptr_t)ctx + k];
}
static float qt__get_bf16_nk(const void *w, int n, int k, const void *ctx) {
    return qt_bf16_to_f32(((const uint16_t *)w)[(size_t)n * (size_t)(intptr_t)ctx + k]);
}
void qt_pack_b_f32_nk(qt_packed *dst, const float *w, int n, int k) {
    qt_pack_b(dst, n, k, qt__get_f32_nk, w, (const void *)(intptr_t)k);
}
void qt_pack_b_bf16_nk(qt_packed *dst, const uint16_t *w, int n, int k) {
    qt_pack_b(dst, n, k, qt__get_bf16_nk, w, (const void *)(intptr_t)k);
}
void qt_packed_free(qt_packed *p) { free(p->p); p->p = NULL; }

/* 6x16 micro-kernel: acc = A[6 rows][K] * P[K][16]. */
static inline void qt__kernel_6x16(int k, const float *const *ar, const float *bp, float *tile) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    const float *a0 = ar[0], *a1 = ar[1], *a2 = ar[2], *a3 = ar[3], *a4 = ar[4], *a5 = ar[5];
    for (int kk = 0; kk < k; kk++) {
        __m256 b0 = _mm256_load_ps(bp), b1 = _mm256_load_ps(bp + 8);
        bp += QT_NR;
        __m256 a;
        a = _mm256_broadcast_ss(a0 + kk); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(a1 + kk); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(a2 + kk); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(a3 + kk); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(a4 + kk); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(a5 + kk); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }
    _mm256_storeu_ps(tile + 0, c00);  _mm256_storeu_ps(tile + 8, c01);
    _mm256_storeu_ps(tile + 16, c10); _mm256_storeu_ps(tile + 24, c11);
    _mm256_storeu_ps(tile + 32, c20); _mm256_storeu_ps(tile + 40, c21);
    _mm256_storeu_ps(tile + 48, c30); _mm256_storeu_ps(tile + 56, c31);
    _mm256_storeu_ps(tile + 64, c40); _mm256_storeu_ps(tile + 72, c41);
    _mm256_storeu_ps(tile + 80, c50); _mm256_storeu_ps(tile + 88, c51);
}

#define QT_KC 512

void qt_sgemm(int m, const float *a, int lda, const qt_packed *b, float *c, int ldc, int accum) {
    int n = b->n, k = b->k;
    int nb_m = (m + QT_MR - 1) / QT_MR;
    int nb_n = (n + QT_NR - 1) / QT_NR;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int bn = 0; bn < nb_n; bn++) {
        for (int bm = 0; bm < nb_m; bm++) {
            float tile[QT_MR * QT_NR], sum[QT_MR * QT_NR];
            int m0 = bm * QT_MR, mr = m - m0 < QT_MR ? m - m0 : QT_MR;
            int n0 = bn * QT_NR, nr = n - n0 < QT_NR ? n - n0 : QT_NR;
            const float *ar[QT_MR];
            memset(sum, 0, sizeof(sum));
            for (int k0 = 0; k0 < k; k0 += QT_KC) {
                int kc = k - k0 < QT_KC ? k - k0 : QT_KC;
                for (int i = 0; i < QT_MR; i++)
                    ar[i] = a + (size_t)(m0 + (i < mr ? i : 0)) * lda + k0;
                qt__kernel_6x16(kc, ar, b->p + ((size_t)bn * k + k0) * QT_NR, tile);
                for (int i = 0; i < QT_MR * QT_NR; i++) sum[i] += tile[i];
            }
            for (int i = 0; i < mr; i++) {
                float *cr = c + (size_t)(m0 + i) * ldc + n0;
                if (accum) for (int j = 0; j < nr; j++) cr[j] += sum[i * QT_NR + j];
                else       for (int j = 0; j < nr; j++) cr[j] = sum[i * QT_NR + j];
            }
        }
    }
}

static inline __m256 qt__ld_bf16x8(const uint16_t *p) {
    __m128i h = _mm_loadu_si128((const __m128i *)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

static inline float qt__hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

void qt_gemv_bf16(int m, const float *x, const uint16_t *w, int n, int k, float *y) {
    for (int m0 = 0; m0 < m; m0 += 4) {
        int mr = m - m0 < 4 ? m - m0 : 4;
        const float *x0 = x + (size_t)m0 * k;
        const float *x1 = x0 + (mr > 1 ? k : 0);
        const float *x2 = x0 + (mr > 2 ? 2 * (size_t)k : 0);
        const float *x3 = x0 + (mr > 3 ? 3 * (size_t)k : 0);
        #pragma omp parallel for schedule(static)
        for (int nn = 0; nn < n; nn++) {
            const uint16_t *wr = w + (size_t)nn * k;
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
            int kk = 0;
            for (; kk + 8 <= k; kk += 8) {
                __m256 wv = qt__ld_bf16x8(wr + kk);
                s0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + kk), s0);
                s1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + kk), s1);
                s2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2 + kk), s2);
                s3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3 + kk), s3);
            }
            float r0 = qt__hsum(s0), r1 = qt__hsum(s1), r2 = qt__hsum(s2), r3 = qt__hsum(s3);
            for (; kk < k; kk++) {
                float wv = qt_bf16_to_f32(wr[kk]);
                r0 += wv * x0[kk]; r1 += wv * x1[kk]; r2 += wv * x2[kk]; r3 += wv * x3[kk];
            }
            y[(size_t)m0 * n + nn] = r0;
            if (mr > 1) y[(size_t)(m0 + 1) * n + nn] = r1;
            if (mr > 2) y[(size_t)(m0 + 2) * n + nn] = r2;
            if (mr > 3) y[(size_t)(m0 + 3) * n + nn] = r3;
        }
    }
}

void qt_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * (w ? w[i] : 1.0f);
}

void qt_rmsnorm_bf16w(float *out, const float *x, const uint16_t *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * qt_bf16_to_f32(w[i]);
}

void qt_layernorm(float *out, const float *x, const float *w, const float *b, int n, float eps) {
    double mu = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= n;
    for (int i = 0; i < n; i++) { double d = x[i] - mu; var += d * d; }
    var /= n;
    float r = (float)(1.0 / sqrt(var + eps));
    for (int i = 0; i < n; i++) out[i] = (float)(x[i] - mu) * r * w[i] + b[i];
}

void qt_rope_neox(float *v, int hd, int pos, float theta) {
    int half = hd / 2;
    for (int i = 0; i < half; i++) {
        /* HF: inv_freq = 1 / theta^(2i/hd) computed in float32 */
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)hd);
        float ang = (float)pos * inv;
        float cs = cosf(ang), sn = sinf(ang);
        float a = v[i], b = v[i + half];
        v[i] = a * cs - b * sn;
        v[i + half] = b * cs + a * sn;
    }
}

float qt_silu(float x) { return x / (1.0f + expf(-x)); }
float qt_gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

void qt_attention(float *out, const float *q, const float *kc, const float *vc,
                  int nq, int pos0, int nh, int nkv, int hd, int window) {
    int grp = nh / nkv;
    float scale = 1.0f / sqrtf((float)hd);
    int kvs = nkv * hd;
    #pragma omp parallel for collapse(2) schedule(dynamic) if (nq * nh > 8)
    for (int i = 0; i < nq; i++) {
        for (int h = 0; h < nh; h++) {
            int qp = pos0 + i;
            int kv0 = window > 0 && qp - window + 1 > 0 ? qp - window + 1 : 0;
            int nk = qp + 1 - kv0;
            const float *qv = q + ((size_t)i * nh + h) * hd;
            const float *kb = kc + (size_t)(h / grp) * hd;
            const float *vb = vc + (size_t)(h / grp) * hd;
            float stack_s[1024];
            float *s = nk <= 1024 ? stack_s : (float *)malloc(sizeof(float) * nk);
            float mx = -INFINITY;
            for (int j = 0; j < nk; j++) {
                const float *kv = kb + (size_t)(kv0 + j) * kvs;
                float d = 0.0f;
                for (int t = 0; t < hd; t++) d += qv[t] * kv[t];
                s[j] = d * scale;
                if (s[j] > mx) mx = s[j];
            }
            double den = 0.0;
            for (int j = 0; j < nk; j++) { s[j] = expf(s[j] - mx); den += s[j]; }
            float inv = (float)(1.0 / den);
            float *o = out + ((size_t)i * nh + h) * hd;
            for (int t = 0; t < hd; t++) o[t] = 0.0f;
            for (int j = 0; j < nk; j++) {
                const float *vv = vb + (size_t)(kv0 + j) * kvs;
                float p = s[j] * inv;
                for (int t = 0; t < hd; t++) o[t] += p * vv[t];
            }
            if (s != stack_s) free(s);
        }
    }
}

static int qt__npy_save(const char *path, const void *d, size_t esz, const char *descr,
                        int ndim, const int *dims) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    char hdr[256], shape[128] = "";
    size_t n = 1, off = 0;
    for (int i = 0; i < ndim; i++) {
        off += (size_t)snprintf(shape + off, sizeof(shape) - off, "%d%s", dims[i],
                                ndim == 1 ? "," : (i + 1 < ndim ? ", " : ""));
        n *= (size_t)dims[i];
    }
    int hl = snprintf(hdr, sizeof(hdr), "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }",
                      descr, shape);
    int total = 10 + hl + 1;
    int pad = (64 - total % 64) % 64;
    uint16_t hlen = (uint16_t)(hl + pad + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    size_t w = fwrite(d, esz, n, f);
    fclose(f);
    return w == n ? 0 : -1;
}
int qt_npy_save_f32(const char *path, const float *d, int ndim, const int *dims) {
    return qt__npy_save(path, d, 4, "<f4", ndim, dims);
}
int qt_npy_save_i32(const char *path, const int32_t *d, int ndim, const int *dims) {
    return qt__npy_save(path, d, 4, "<i4", ndim, dims);
}

#endif /* QTTS_OPS_IMPLEMENTATION */
#endif /* QTTS_OPS_H */
