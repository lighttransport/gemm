/*
 * trellis2_shape_decoder.h - TRELLIS.2 Stage 2 Shape Decoder (SC-VAE)
 *
 * Usage:
 *   #define T2_SHAPE_DEC_IMPLEMENTATION
 *   #include "trellis2_shape_decoder.h"
 *
 * Dependencies: sparse3d.h, safetensors.h, ggml_dequant.h
 *
 * Decodes shape structured latent [N, 32] to per-voxel predictions [N', 7]
 * using sparse ConvNeXt blocks with channel-to-spatial upsampling.
 *
 * Architecture:
 *   from_latent: Linear(32 → 1024)
 *   blocks.0: 4× ConvNeXtBlock(1024) + C2S(1024→512)
 *   blocks.1: 16× ConvNeXtBlock(512) + C2S(512→256)
 *   blocks.2: 8× ConvNeXtBlock(256) + C2S(256→128)
 *   blocks.3: 4× ConvNeXtBlock(128) + C2S(128→64)
 *   output_layer: Linear(64 → 7)
 *
 * API:
 *   t2_shape_dec *t2_shape_dec_load(const char *st_path);
 *   void          t2_shape_dec_free(t2_shape_dec *d);
 *   t2_shape_dec_result t2_shape_dec_forward(t2_shape_dec *d,
 *       const sp3d_tensor *slat, int n_threads);
 */
#ifndef T2_SHAPE_DEC_H
#define T2_SHAPE_DEC_H

#include <stdint.h>
#include "sparse3d.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *feats;       /* [N, 7]: vertex_xyz(3), intersected(3), quad_lerp(1) */
    int32_t *coords;    /* [N, 4]: batch, z, y, x */
    int N;
} t2_shape_dec_result;

typedef struct t2_shape_dec t2_shape_dec;

t2_shape_dec *t2_shape_dec_load(const char *st_path);
void          t2_shape_dec_free(t2_shape_dec *d);
t2_shape_dec_result t2_shape_dec_forward(t2_shape_dec *d,
    const sp3d_tensor *slat, int n_threads);
void t2_shape_dec_result_free(t2_shape_dec_result *r);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef T2_SHAPE_DEC_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define T2SD_AVX2 1
#else
#define T2SD_AVX2 0
#endif

/* ---- ConvNeXtBlock weights ---- */
typedef struct {
    float *conv_w, *conv_b;   /* [C, 27, C], [C] */
    float *norm_w, *norm_b;   /* [C] */
    float *mlp0_w, *mlp0_b;  /* [4C, C] */
    float *mlp2_w, *mlp2_b;  /* [C, 4C] */
    int C;
} t2sd_convnext;

/* ---- Channel-to-Spatial block weights ---- */
typedef struct {
    float *norm1_w, *norm1_b;       /* [C_in] */
    float *conv1_w, *conv1_b;       /* [C_out*8, 27, C_in] */
    float *conv2_w, *conv2_b;       /* [C_out, 27, C_out] */
    float *to_subdiv_w, *to_subdiv_b; /* [8, C_in] */
    int C_in, C_out;
} t2sd_c2s;

/* ---- Decoder model ---- */
#define T2SD_MAX_BLOCKS 20
#define T2SD_MAX_STAGES 5

struct t2_shape_dec {
    float *from_latent_w, *from_latent_b;  /* [1024, 32] */
    float *output_w, *output_b;             /* [7, 64] */

    /* Stages: each has N convnext blocks + 1 C2S block */
    int n_stages;
    int n_convnext[T2SD_MAX_STAGES];
    int channels[T2SD_MAX_STAGES];  /* [1024, 512, 256, 128, 64] */
    t2sd_convnext convnext[T2SD_MAX_STAGES][T2SD_MAX_BLOCKS];
    t2sd_c2s c2s[T2SD_MAX_STAGES];

    void *st_ctx;
};

static double t2sd_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- AVX2/FMA dot product helper ---- */
static inline float t2sd_dot(const float *a, const float *b, int n) {
#if T2SD_AVX2
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int j = 0;
    for (; j + 15 < n; j += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j + 8), _mm256_loadu_ps(b + j + 8), sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    for (; j + 7 < n; j += 8)
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j), sum0);
    /* Horizontal sum */
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; j < n; j++) s += a[j] * b[j];
    return s;
#else
    float s = 0;
    for (int j = 0; j < n; j++) s += a[j] * b[j];
    return s;
#endif
}

/* ---- Multithreaded LayerNorm ---- */
typedef struct { float *dst; const float *src, *w, *b; int start, end, C; float eps; } t2sd_ln_task;
static void *t2sd_layernorm_worker(void *arg) {
    t2sd_ln_task *t = (t2sd_ln_task *)arg;
    int C = t->C; float eps = t->eps;
    for (int i = t->start; i < t->end; i++) {
        const float *xi = t->src + (size_t)i * C;
        float *yi = t->dst + (size_t)i * C;
#if T2SD_AVX2
        __m256 vsum = _mm256_setzero_ps();
        int j = 0;
        for (; j + 7 < C; j += 8) vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(xi + j));
        float mean = 0; float tmp[8]; _mm256_storeu_ps(tmp, vsum);
        for (int k = 0; k < 8; k++) mean += tmp[k];
        for (; j < C; j++) mean += xi[j];
        mean /= C;
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vvar = _mm256_setzero_ps();
        for (j = 0; j + 7 < C; j += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(xi + j), vmean);
            vvar = _mm256_fmadd_ps(d, d, vvar);
        }
        float var = 0; _mm256_storeu_ps(tmp, vvar);
        for (int k = 0; k < 8; k++) var += tmp[k];
        for (; j < C; j++) { float d = xi[j] - mean; var += d * d; }
        var /= C;
        float inv = 1.0f / sqrtf(var + eps);
        __m256 vinv = _mm256_set1_ps(inv);
        for (j = 0; j + 7 < C; j += 8) {
            __m256 val = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(xi + j), vmean), vinv);
            if (t->w) val = _mm256_mul_ps(val, _mm256_loadu_ps(t->w + j));
            if (t->b) val = _mm256_add_ps(val, _mm256_loadu_ps(t->b + j));
            _mm256_storeu_ps(yi + j, val);
        }
        for (; j < C; j++)
            yi[j] = (xi[j] - mean) * inv * (t->w ? t->w[j] : 1.0f) + (t->b ? t->b[j] : 0.0f);
#else
        float mean = 0;
        for (int j = 0; j < C; j++) mean += xi[j];
        mean /= C;
        float var = 0;
        for (int j = 0; j < C; j++) { float d = xi[j] - mean; var += d * d; }
        var /= C;
        float inv = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < C; j++)
            yi[j] = (xi[j] - mean) * inv * (t->w ? t->w[j] : 1.0f) + (t->b ? t->b[j] : 0.0f);
#endif
    }
    return NULL;
}

static void t2sd_layernorm(float *dst, const float *src, const float *w, const float *b,
                             int N, int C, float eps) {
    t2sd_ln_task task = {dst, src, w, b, 0, N, C, eps};
    t2sd_layernorm_worker(&task);
}

static void t2sd_layernorm_mt(float *dst, const float *src, const float *w, const float *b,
                                int N, int C, float eps, int n_threads) {
    if (n_threads <= 1) { t2sd_layernorm(dst, src, w, b, N, C, eps); return; }
    pthread_t *threads = (pthread_t *)alloca((size_t)n_threads * sizeof(pthread_t));
    t2sd_ln_task *tasks = (t2sd_ln_task *)alloca((size_t)n_threads * sizeof(t2sd_ln_task));
    for (int i = 0; i < n_threads; i++) {
        tasks[i] = (t2sd_ln_task){dst, src, w, b, i * N / n_threads, (i + 1) * N / n_threads, C, eps};
        pthread_create(&threads[i], NULL, t2sd_layernorm_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* ---- GELU (AVX2 approximation) ---- */
static void t2sd_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

/* ---- Multithreaded linear with AVX2 dot product ---- */
typedef struct {
    float *dst; const float *src, *W, *bias;
    int start, end, in_C, out_C;
} t2sd_linear_task;

static void *t2sd_linear_worker(void *arg) {
    t2sd_linear_task *t = (t2sd_linear_task *)arg;
    int in_C = t->in_C, out_C = t->out_C;
    for (int i = t->start; i < t->end; i++) {
        const float *xi = t->src + (size_t)i * in_C;
        float *yi = t->dst + (size_t)i * out_C;
        for (int o = 0; o < out_C; o++) {
            const float *wr = t->W + (size_t)o * in_C;
            yi[o] = t2sd_dot(wr, xi, in_C) + (t->bias ? t->bias[o] : 0.0f);
        }
    }
    return NULL;
}

static void t2sd_linear(float *dst, const float *src, int N,
                          const float *W, const float *bias,
                          int out_C, int in_C) {
    t2sd_linear_task task = {dst, src, W, bias, 0, N, in_C, out_C};
    t2sd_linear_worker(&task);
}

static void t2sd_linear_mt(float *dst, const float *src, int N,
                             const float *W, const float *bias,
                             int out_C, int in_C, int n_threads) {
    if (n_threads <= 1 || N < n_threads) {
        t2sd_linear(dst, src, N, W, bias, out_C, in_C); return;
    }
    pthread_t *threads = (pthread_t *)alloca((size_t)n_threads * sizeof(pthread_t));
    t2sd_linear_task *tasks = (t2sd_linear_task *)alloca((size_t)n_threads * sizeof(t2sd_linear_task));
    for (int i = 0; i < n_threads; i++) {
        tasks[i] = (t2sd_linear_task){dst, src, W, bias,
                                       i * N / n_threads, (i + 1) * N / n_threads,
                                       in_C, out_C};
        pthread_create(&threads[i], NULL, t2sd_linear_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* ---- Multithreaded sparse conv3d with AVX2 ---- */
typedef struct {
    float *dst; const sp3d_tensor *t; const float *weight, *bias;
    int start, end, in_C, out_C;
} t2sd_conv_task;

/* Matrix-vector multiply: dst[out_C] += W[out_C, in_C] @ x[in_C]
 * Process 4 output rows at once for better ILP */
static inline void t2sd_matvec_add(float *dst, const float *W, const float *x,
                                     int out_C, int in_C) {
#if T2SD_AVX2
    int o = 0;
    for (; o + 3 < out_C; o += 4) {
        const float *w0 = W + (size_t)(o + 0) * in_C;
        const float *w1 = W + (size_t)(o + 1) * in_C;
        const float *w2 = W + (size_t)(o + 2) * in_C;
        const float *w3 = W + (size_t)(o + 3) * in_C;
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 7 < in_C; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + j), xv, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + j), xv, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + j), xv, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + j), xv, s3);
        }
        /* Horizontal reduce */
        __m128 lo0 = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
        __m128 lo1 = _mm_add_ps(_mm256_castps256_ps128(s1), _mm256_extractf128_ps(s1, 1));
        __m128 lo2 = _mm_add_ps(_mm256_castps256_ps128(s2), _mm256_extractf128_ps(s2, 1));
        __m128 lo3 = _mm_add_ps(_mm256_castps256_ps128(s3), _mm256_extractf128_ps(s3, 1));
        /* Transpose 4×4 and sum */
        __m128 t01lo = _mm_unpacklo_ps(lo0, lo1);
        __m128 t01hi = _mm_unpackhi_ps(lo0, lo1);
        __m128 t23lo = _mm_unpacklo_ps(lo2, lo3);
        __m128 t23hi = _mm_unpackhi_ps(lo2, lo3);
        __m128 row0 = _mm_movelh_ps(t01lo, t23lo);
        __m128 row1 = _mm_movehl_ps(t23lo, t01lo);
        __m128 row2 = _mm_movelh_ps(t01hi, t23hi);
        __m128 row3 = _mm_movehl_ps(t23hi, t01hi);
        __m128 sum4 = _mm_add_ps(_mm_add_ps(row0, row1), _mm_add_ps(row2, row3));
        /* Handle tail elements */
        float tail[4] = {0, 0, 0, 0};
        for (; j < in_C; j++) {
            float xv = x[j];
            tail[0] += w0[j] * xv; tail[1] += w1[j] * xv;
            tail[2] += w2[j] * xv; tail[3] += w3[j] * xv;
        }
        sum4 = _mm_add_ps(sum4, _mm_loadu_ps(tail));
        /* Accumulate into dst */
        _mm_storeu_ps(dst + o, _mm_add_ps(_mm_loadu_ps(dst + o), sum4));
    }
    for (; o < out_C; o++) {
        dst[o] += t2sd_dot(W + (size_t)o * in_C, x, in_C);
    }
#else
    for (int o = 0; o < out_C; o++)
        dst[o] += t2sd_dot(W + (size_t)o * in_C, x, in_C);
#endif
}

static void *t2sd_sparse_conv_worker(void *arg) {
    t2sd_conv_task *tk = (t2sd_conv_task *)arg;
    const sp3d_tensor *t = tk->t;
    int in_C = tk->in_C, out_C = tk->out_C;

    for (int i = tk->start; i < tk->end; i++) {
        float *di = tk->dst + (size_t)i * out_C;
        /* Init with bias */
        if (tk->bias)
            memcpy(di, tk->bias, (size_t)out_C * sizeof(float));
        else
            memset(di, 0, (size_t)out_C * sizeof(float));

        int32_t bz = t->coords[i * 4];
        int32_t z  = t->coords[i * 4 + 1];
        int32_t y  = t->coords[i * 4 + 2];
        int32_t x  = t->coords[i * 4 + 3];

        for (int kd = 0; kd < 3; kd++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int nz = z + kd - 1, ny = y + kh - 1, nx = x + kw - 1;
                    int ni = sp3d_hash_lookup(t->hash, bz, nz, ny, nx);
                    if (ni < 0) continue;
                    int k_idx = kd * 9 + kh * 3 + kw;
                    const float *feat_n = t->feats + (size_t)ni * in_C;
                    /* Weight layout: [out_C, 27, in_C]
                     * For kernel position k_idx, weight for output o is at:
                     *   weight[(o * 27 + k_idx) * in_C]
                     * Row stride between consecutive o: 27 * in_C */
                    const float *W_k0 = tk->weight + (size_t)k_idx * in_C;
                    int w_stride = 27 * in_C;
#if T2SD_AVX2
                    int o = 0;
                    for (; o + 3 < out_C; o += 4) {
                        const float *w0 = W_k0 + (size_t)(o + 0) * w_stride;
                        const float *w1 = W_k0 + (size_t)(o + 1) * w_stride;
                        const float *w2 = W_k0 + (size_t)(o + 2) * w_stride;
                        const float *w3 = W_k0 + (size_t)(o + 3) * w_stride;
                        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                        int j = 0;
                        for (; j + 7 < in_C; j += 8) {
                            __m256 xv = _mm256_loadu_ps(feat_n + j);
                            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + j), xv, s0);
                            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + j), xv, s1);
                            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + j), xv, s2);
                            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + j), xv, s3);
                        }
                        __m128 lo0 = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
                        __m128 lo1 = _mm_add_ps(_mm256_castps256_ps128(s1), _mm256_extractf128_ps(s1, 1));
                        __m128 lo2 = _mm_add_ps(_mm256_castps256_ps128(s2), _mm256_extractf128_ps(s2, 1));
                        __m128 lo3 = _mm_add_ps(_mm256_castps256_ps128(s3), _mm256_extractf128_ps(s3, 1));
                        __m128 t01lo = _mm_unpacklo_ps(lo0, lo1);
                        __m128 t01hi = _mm_unpackhi_ps(lo0, lo1);
                        __m128 t23lo = _mm_unpacklo_ps(lo2, lo3);
                        __m128 t23hi = _mm_unpackhi_ps(lo2, lo3);
                        __m128 row0 = _mm_movelh_ps(t01lo, t23lo);
                        __m128 row1 = _mm_movehl_ps(t23lo, t01lo);
                        __m128 row2 = _mm_movelh_ps(t01hi, t23hi);
                        __m128 row3 = _mm_movehl_ps(t23hi, t01hi);
                        __m128 sum4 = _mm_add_ps(_mm_add_ps(row0, row1), _mm_add_ps(row2, row3));
                        float tail[4] = {0, 0, 0, 0};
                        for (; j < in_C; j++) {
                            float xv2 = feat_n[j];
                            tail[0] += w0[j]*xv2; tail[1] += w1[j]*xv2;
                            tail[2] += w2[j]*xv2; tail[3] += w3[j]*xv2;
                        }
                        sum4 = _mm_add_ps(sum4, _mm_loadu_ps(tail));
                        _mm_storeu_ps(di + o, _mm_add_ps(_mm_loadu_ps(di + o), sum4));
                    }
                    for (; o < out_C; o++)
                        di[o] += t2sd_dot(W_k0 + (size_t)o * w_stride, feat_n, in_C);
#else
                    for (int o = 0; o < out_C; o++) {
                        const float *kern = W_k0 + (size_t)o * w_stride;
                        di[o] += t2sd_dot(kern, feat_n, in_C);
                    }
#endif
                }
            }
        }
    }
    return NULL;
}

static void t2sd_sparse_conv(float *dst, const sp3d_tensor *t,
                               const float *weight, const float *bias,
                               int in_C, int out_C, int n_threads) {
    sp3d_ensure_hash((sp3d_tensor *)t);
    int N = t->N;
    if (n_threads <= 1 || N < n_threads) {
        t2sd_conv_task task = {dst, t, weight, bias, 0, N, in_C, out_C};
        t2sd_sparse_conv_worker(&task);
        return;
    }
    pthread_t *threads = (pthread_t *)alloca((size_t)n_threads * sizeof(pthread_t));
    t2sd_conv_task *tasks = (t2sd_conv_task *)alloca((size_t)n_threads * sizeof(t2sd_conv_task));
    for (int i = 0; i < n_threads; i++) {
        tasks[i] = (t2sd_conv_task){dst, t, weight, bias,
                                     i * N / n_threads, (i + 1) * N / n_threads,
                                     in_C, out_C};
        pthread_create(&threads[i], NULL, t2sd_sparse_conv_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* ---- Forward helpers ---- */

static void t2sd_convnext_forward(float *feats, int N, const t2sd_convnext *blk,
                                    sp3d_tensor *t, int n_threads) {
    int C = blk->C;
    float *tmp = (float *)malloc((size_t)N * C * sizeof(float));
    float *mlp_buf = (float *)malloc((size_t)N * 4 * C * sizeof(float));

    /* conv(feats) -> tmp */
    t2sd_sparse_conv(tmp, t, blk->conv_w, blk->conv_b, C, C, n_threads);

    /* layernorm(tmp) */
    t2sd_layernorm_mt(tmp, tmp, blk->norm_w, blk->norm_b, N, C, 1e-6f, n_threads);

    /* mlp: Linear(C, 4C) -> GELU -> Linear(4C, C) */
    t2sd_linear_mt(mlp_buf, tmp, N, blk->mlp0_w, blk->mlp0_b, 4 * C, C, n_threads);
    t2sd_gelu(mlp_buf, N * 4 * C);
    t2sd_linear_mt(tmp, mlp_buf, N, blk->mlp2_w, blk->mlp2_b, C, 4 * C, n_threads);

    /* residual: feats += tmp */
#if T2SD_AVX2
    int j = 0;
    for (; j + 7 < N * C; j += 8) {
        __m256 a = _mm256_loadu_ps(feats + j);
        __m256 b = _mm256_loadu_ps(tmp + j);
        _mm256_storeu_ps(feats + j, _mm256_add_ps(a, b));
    }
    for (; j < N * C; j++) feats[j] += tmp[j];
#else
    for (int i = 0; i < N * C; i++) feats[i] += tmp[i];
#endif

    free(tmp);
    free(mlp_buf);
}

static sp3d_tensor *t2sd_c2s_forward(sp3d_tensor *t, const t2sd_c2s *blk, int n_threads) {
    int N = t->N;
    int C_in = blk->C_in, C_out = blk->C_out;

    /* 1. Predict subdivision: which sub-voxels to activate */
    float *sub_logits = (float *)malloc((size_t)N * 8 * sizeof(float));
    if (blk->to_subdiv_w) {
        t2sd_linear_mt(sub_logits, t->feats, N, blk->to_subdiv_w, blk->to_subdiv_b, 8, C_in, n_threads);
    } else {
        /* No subdivision prediction — activate all 8 sub-voxels (for texture decoder) */
        for (int i = 0; i < N * 8; i++) sub_logits[i] = 1.0f;
    }

    /* Threshold: activate sub-voxels where logit > 0 */
    int total_sub = 0;
    for (int i = 0; i < N * 8; i++)
        if (sub_logits[i] > 0) total_sub++;

    fprintf(stderr, "    C2S %d->%d: %d voxels -> %d sub-voxels (%.1f avg)\n",
            C_in, C_out, N, total_sub, (float)total_sub / N);

    /* 2. LayerNorm input */
    float *normed = (float *)malloc((size_t)N * C_in * sizeof(float));
    t2sd_layernorm_mt(normed, t->feats, blk->norm1_w, blk->norm1_b, N, C_in, 1e-6f, n_threads);

    /* 3. conv1: [C_out*8, 27, C_in] -> [N, C_out*8] */
    float *expanded = (float *)malloc((size_t)N * C_out * 8 * sizeof(float));
    /* Need a temp tensor with normed features for conv */
    sp3d_tensor *t_normed = sp3d_replace_feats(t, normed, C_in);
    t2sd_sparse_conv(expanded, t_normed, blk->conv1_w, blk->conv1_b,
                         C_in, C_out * 8, n_threads);

    /* 4. Create sub-voxel coordinates and features */
    int32_t *sub_coords = (int32_t *)malloc((size_t)total_sub * 4 * sizeof(int32_t));
    float *sub_feats = (float *)malloc((size_t)total_sub * C_out * sizeof(float));
    int si = 0;
    for (int i = 0; i < N; i++) {
        int32_t bz = t->coords[i * 4 + 0];
        int32_t z  = t->coords[i * 4 + 1];
        int32_t y  = t->coords[i * 4 + 2];
        int32_t x  = t->coords[i * 4 + 3];
        for (int s = 0; s < 8; s++) {
            if (sub_logits[i * 8 + s] <= 0) continue;
            /* Sub-voxel offset: s = (dz*4 + dy*2 + dx) */
            int dz = (s >> 2) & 1, dy = (s >> 1) & 1, dx = s & 1;
            sub_coords[si * 4 + 0] = bz;
            sub_coords[si * 4 + 1] = z * 2 + dz;
            sub_coords[si * 4 + 2] = y * 2 + dy;
            sub_coords[si * 4 + 3] = x * 2 + dx;
            /* Feature: extract sub-channel from expanded[i, s*C_out : (s+1)*C_out] */
            memcpy(sub_feats + (size_t)si * C_out,
                   expanded + (size_t)i * C_out * 8 + (size_t)s * C_out,
                   (size_t)C_out * sizeof(float));
            si++;
        }
    }

    free(sub_logits); free(normed); free(expanded);
    sp3d_free(t_normed);

    /* 5. Create new sparse tensor at higher resolution */
    sp3d_tensor *t_sub = sp3d_create(sub_coords, sub_feats, total_sub, C_out, 1);
    free(sub_coords); free(sub_feats);

    /* 6. conv2: [C_out, 27, C_out] at new resolution */
    float *conv2_out = (float *)malloc((size_t)total_sub * C_out * sizeof(float));
    t2sd_sparse_conv(conv2_out, t_sub, blk->conv2_w, blk->conv2_b,
                         C_out, C_out, n_threads);
    /* Replace features */
    memcpy(t_sub->feats, conv2_out, (size_t)total_sub * C_out * sizeof(float));
    free(conv2_out);

    return t_sub;
}

/* ---- Full forward ---- */

t2_shape_dec_result t2_shape_dec_forward(t2_shape_dec *d,
    const sp3d_tensor *slat, int n_threads) {
    t2_shape_dec_result result = {0};
    double t0 = t2sd_time_ms();
    int N = slat->N;
    int C = d->channels[0]; /* 1024 */

    fprintf(stderr, "shape_dec: input N=%d, C=%d\n", N, slat->C);

    /* from_latent: Linear(32 -> 1024) */
    float *feats = (float *)malloc((size_t)N * C * sizeof(float));
    t2sd_linear(feats, slat->feats, N, d->from_latent_w, d->from_latent_b, C, slat->C);
    fprintf(stderr, "shape_dec: from_latent -> [%d, %d]\n", N, C);

    /* Create working sparse tensor */
    sp3d_tensor *t = sp3d_create(slat->coords, feats, N, C, 1);
    free(feats);

    /* Process stages */
    for (int stage = 0; stage < d->n_stages; stage++) {
        int nc = d->n_convnext[stage];
        int ch = d->channels[stage];

        fprintf(stderr, "shape_dec: stage %d: %d ConvNeXt(%d), N=%d\n",
                stage, nc, ch, t->N);
        double ts0 = t2sd_time_ms();

        /* ConvNeXt blocks */
        for (int b = 0; b < nc; b++) {
            t2sd_convnext_forward(t->feats, t->N, &d->convnext[stage][b], t, n_threads);
        }

        double ts1 = t2sd_time_ms();
        fprintf(stderr, "  convnext: %.1f s\n", (ts1 - ts0) / 1000.0);

        /* C2S upsample (if not last stage, or if c2s weights exist) */
        if (d->c2s[stage].conv1_w) {
            sp3d_tensor *t_new = t2sd_c2s_forward(t, &d->c2s[stage], n_threads);
            sp3d_free(t);
            t = t_new;
            fprintf(stderr, "  c2s: -> N=%d, C=%d (%.1f s)\n",
                    t->N, t->C, (t2sd_time_ms() - ts1) / 1000.0);
        }
    }

    /* output_layer: Linear(64 -> 7) */
    float *out_feats = (float *)malloc((size_t)t->N * 7 * sizeof(float));
    t2sd_linear(out_feats, t->feats, t->N, d->output_w, d->output_b, 7, t->C);

    result.feats = out_feats;
    result.coords = (int32_t *)malloc((size_t)t->N * 4 * sizeof(int32_t));
    memcpy(result.coords, t->coords, (size_t)t->N * 4 * sizeof(int32_t));
    result.N = t->N;

    sp3d_free(t);

    double t1 = t2sd_time_ms();
    fprintf(stderr, "shape_dec: done in %.1f s, output N=%d\n",
            (t1 - t0) / 1000.0, result.N);
    return result;
}

void t2_shape_dec_result_free(t2_shape_dec_result *r) {
    free(r->feats); free(r->coords);
    r->feats = NULL; r->coords = NULL; r->N = 0;
}

/* ---- Weight loading ---- */

#ifdef SAFETENSORS_H

static float *t2sd_load_f32(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    const char *dt = safetensors_dtype(st, idx);
    size_t nb = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);

    size_t n_elem = (strcmp(dt, "F32") == 0) ? nb / 4 :
                    (strcmp(dt, "F16") == 0 || strcmp(dt, "BF16") == 0) ? nb / 2 : 0;
    float *buf = (float *)malloc(n_elem * sizeof(float));

    if (strcmp(dt, "F32") == 0) {
        memcpy(buf, data, nb);
    } else if (strcmp(dt, "F16") == 0) {
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n_elem; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else if (strcmp(dt, "BF16") == 0) {
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&buf[i], &bits, 4);
        }
    }
    return buf;
}

t2_shape_dec *t2_shape_dec_load(const char *st_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) return NULL;
    fprintf(stderr, "shape_dec: loading from %s (%d tensors)\n", st_path, st->n_tensors);

    t2_shape_dec *d = (t2_shape_dec *)calloc(1, sizeof(t2_shape_dec));
    d->st_ctx = st;

    d->from_latent_w = t2sd_load_f32(st, "from_latent.weight");
    d->from_latent_b = t2sd_load_f32(st, "from_latent.bias");
    d->output_w = t2sd_load_f32(st, "output_layer.weight");
    d->output_b = t2sd_load_f32(st, "output_layer.bias");

    /* Detect stages and blocks from weight names */
    int channels[] = {1024, 512, 256, 128, 64};
    int n_convnext[] = {4, 16, 8, 4, 0};
    d->n_stages = 4;  /* stages with C2S blocks */
    memcpy(d->channels, channels, sizeof(channels));
    memcpy(d->n_convnext, n_convnext, sizeof(n_convnext));

    for (int s = 0; s < d->n_stages; s++) {
        int C = channels[s];
        char name[256];

        /* ConvNeXt blocks */
        for (int b = 0; b < n_convnext[s]; b++) {
            t2sd_convnext *blk = &d->convnext[s][b];
            blk->C = C;
            #define LOAD(field, suffix) do { \
                snprintf(name, sizeof(name), "blocks.%d.%d.%s", s, b, suffix); \
                blk->field = t2sd_load_f32(st, name); \
            } while(0)
            LOAD(conv_w, "conv.weight");
            LOAD(conv_b, "conv.bias");
            LOAD(norm_w, "norm.weight");
            LOAD(norm_b, "norm.bias");
            LOAD(mlp0_w, "mlp.0.weight");
            LOAD(mlp0_b, "mlp.0.bias");
            LOAD(mlp2_w, "mlp.2.weight");
            LOAD(mlp2_b, "mlp.2.bias");
            #undef LOAD
        }

        /* C2S block (last block in the stage) */
        int c2s_idx = n_convnext[s];
        t2sd_c2s *c = &d->c2s[s];
        c->C_in = C;
        c->C_out = (s + 1 < 5) ? channels[s + 1] : C;
        #define LOAD_C2S(field, suffix) do { \
            snprintf(name, sizeof(name), "blocks.%d.%d.%s", s, c2s_idx, suffix); \
            c->field = t2sd_load_f32(st, name); \
        } while(0)
        LOAD_C2S(norm1_w, "norm1.weight");
        LOAD_C2S(norm1_b, "norm1.bias");
        LOAD_C2S(conv1_w, "conv1.weight");
        LOAD_C2S(conv1_b, "conv1.bias");
        LOAD_C2S(conv2_w, "conv2.weight");
        LOAD_C2S(conv2_b, "conv2.bias");
        LOAD_C2S(to_subdiv_w, "to_subdiv.weight");
        LOAD_C2S(to_subdiv_b, "to_subdiv.bias");
        #undef LOAD_C2S

        fprintf(stderr, "  stage %d: %d ConvNeXt(%d) + C2S(%d->%d) %s\n",
                s, n_convnext[s], C, C, c->C_out,
                c->conv1_w ? "loaded" : "MISSING");
    }

    fprintf(stderr, "shape_dec: loaded\n");
    return d;
}

#endif /* SAFETENSORS_H */

void t2_shape_dec_free(t2_shape_dec *d) {
    if (!d) return;
    /* Free all weight buffers */
    free(d->from_latent_w); free(d->from_latent_b);
    free(d->output_w); free(d->output_b);
    for (int s = 0; s < d->n_stages; s++) {
        for (int b = 0; b < d->n_convnext[s]; b++) {
            t2sd_convnext *blk = &d->convnext[s][b];
            free(blk->conv_w); free(blk->conv_b);
            free(blk->norm_w); free(blk->norm_b);
            free(blk->mlp0_w); free(blk->mlp0_b);
            free(blk->mlp2_w); free(blk->mlp2_b);
        }
        t2sd_c2s *c = &d->c2s[s];
        free(c->norm1_w); free(c->norm1_b);
        free(c->conv1_w); free(c->conv1_b);
        free(c->conv2_w); free(c->conv2_b);
        free(c->to_subdiv_w); free(c->to_subdiv_b);
    }
#ifdef SAFETENSORS_H
    if (d->st_ctx) safetensors_close((st_context *)d->st_ctx);
#endif
    free(d);
}

#endif /* T2_SHAPE_DEC_IMPLEMENTATION */
#endif /* T2_SHAPE_DEC_H */
