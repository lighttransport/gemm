/*
 * cpu_compute.h - Shared CPU compute primitives for ViT inference
 *
 * Usage:
 *   #define CPU_COMPUTE_IMPLEMENTATION
 *   #include "cpu_compute.h"
 *
 * Dependencies: gemm_f16_f32_tokmajor() from ggml_dequant.h must be available
 *               before including with CPU_COMPUTE_IMPLEMENTATION defined.
 *
 * All functions operate on raw float* arrays (no qtensor dependency).
 */
#ifndef CPU_COMPUTE_H
#define CPU_COMPUTE_H

#include <stdint.h>

/* ======================================================================== */
#ifdef CPU_COMPUTE_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <float.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/* ---- Horizontal sum for AVX ---- */

#if defined(__AVX2__) && defined(__FMA__)
static inline float cpu_hsum_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

/* ---- LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b ---- */

static void cpu_layernorm(float *dst, const float *src, const float *w,
                           const float *b, int n_tok, int dim, float eps) {
    for (int t = 0; t < n_tok; t++) {
        const float *x = src + t * dim;
        float *y = dst + t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++)
            y[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/* ---- QK Normalization: per-head LayerNorm on Q/K with stride ---- */

static void cpu_qk_norm(float *vec, int n_tok, int n_heads, int head_dim,
                          int stride, const float *w, const float *b, float eps) {
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = vec + t * stride + h * head_dim;
            float mean = 0.0f;
            for (int i = 0; i < head_dim; i++) mean += v[i];
            mean /= head_dim;
            float var = 0.0f;
            for (int i = 0; i < head_dim; i++) { float d = v[i] - mean; var += d * d; }
            var /= head_dim;
            float s = 1.0f / sqrtf(var + eps);
            for (int i = 0; i < head_dim; i++)
                v[i] = (v[i] - mean) * s * w[i] + b[i];
        }
    }
}

/* ---- 2D RoPE: split head_dim in half for y/x positions ---- */

static void cpu_rope_2d(float *vec, int n_tok, int n_heads, int head_dim,
                          int stride, const int *pos_y, const int *pos_x,
                          float freq_base) {
    int half = head_dim / 2;
    int quarter = half / 2;
    for (int t = 0; t < n_tok; t++) {
        float py = (float)pos_y[t];
        float px = (float)pos_x[t];
        for (int h = 0; h < n_heads; h++) {
            float *v = vec + t * stride + h * head_dim;
            /* Y rotation: first half */
            for (int j = 0; j < quarter; j++) {
                float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)half);
                float theta = py * freq;
                float c = cosf(theta), s = sinf(theta);
                float v0 = v[j], v1 = v[j + quarter];
                v[j]           = v0 * c - v1 * s;
                v[j + quarter] = v0 * s + v1 * c;
            }
            /* X rotation: second half */
            for (int j = 0; j < quarter; j++) {
                float freq = 1.0f / powf(freq_base, (float)(2 * j) / (float)half);
                float theta = px * freq;
                float c = cosf(theta), s = sinf(theta);
                float v0 = v[half + j], v1 = v[half + j + quarter];
                v[half + j]           = v0 * c - v1 * s;
                v[half + j + quarter] = v0 * s + v1 * c;
            }
        }
    }
}

/* ---- Softmax ---- */

static void cpu_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ---- GELU (tanh approximation) ---- */

static void cpu_gelu(float *x, int n) {
    const float c = 0.7978845608f; /* sqrt(2/pi) */
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float t = tanhf(c * (v + 0.044715f * v * v * v));
        x[i] = 0.5f * v * (1.0f + t);
    }
}

/* ---- SiLU: x * sigmoid(x) ---- */

static void cpu_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* ---- Bilinear resize (CHW layout, align_corners=True) ---- */

static void cpu_bilinear_resize(float *dst, const float *src,
                                  int C, int Hi, int Wi, int Ho, int Wo) {
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < Ho; oh++) {
            float fy = (Ho > 1) ? (float)oh * (Hi - 1) / (Ho - 1) : 0;
            int y0 = (int)fy;
            int y1 = (y0 + 1 < Hi) ? y0 + 1 : y0;
            float dy = fy - y0;
            for (int ow = 0; ow < Wo; ow++) {
                float fx = (Wo > 1) ? (float)ow * (Wi - 1) / (Wo - 1) : 0;
                int x0 = (int)fx;
                int x1 = (x0 + 1 < Wi) ? x0 + 1 : x0;
                float dx = fx - x0;
                dst[c * Ho * Wo + oh * Wo + ow] =
                    src[c * Hi * Wi + y0 * Wi + x0] * (1-dy) * (1-dx) +
                    src[c * Hi * Wi + y0 * Wi + x1] * (1-dy) * dx +
                    src[c * Hi * Wi + y1 * Wi + x0] * dy * (1-dx) +
                    src[c * Hi * Wi + y1 * Wi + x1] * dy * dx;
            }
        }
    }
}

/* ---- Threaded F16 GEMM: Y[tok][out] = W[out,:] * X[tok,:] + bias[out] ---- */

typedef struct {
    float *dst;
    const uint16_t *W;
    const float *src;
    int n_out, n_in, n_tok;
    int r_start, r_end;
} cpu_gemm_task;

static void *cpu_gemm_worker(void *arg) {
    cpu_gemm_task *t = (cpu_gemm_task *)arg;
    int count = t->r_end - t->r_start;
    if (count <= 0) return NULL;
    gemm_f16_f32_tokmajor(t->dst + t->r_start,
                           t->W + (size_t)t->r_start * t->n_in,
                           t->src,
                           count, t->n_in, t->n_tok,
                           t->n_out, t->n_in);
    return NULL;
}

static void cpu_gemm_f16(float *dst, const uint16_t *W, const float *bias,
                           const float *src, int n_tok, int n_out, int n_in,
                           int n_threads) {
    if (n_threads > 1 && n_out >= n_threads * 3) {
        cpu_gemm_task *tasks = (cpu_gemm_task *)calloc((size_t)n_threads, sizeof(cpu_gemm_task));
        pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
        int rows_per = ((n_out / n_threads) / 3) * 3;
        if (rows_per < 3) rows_per = 3;
        int r = 0, actual = 0;
        for (int i = 0; i < n_threads && r < n_out; i++) {
            int end = (i == n_threads - 1) ? n_out : r + rows_per;
            if (end > n_out) end = n_out;
            tasks[i] = (cpu_gemm_task){dst, W, src, n_out, n_in, n_tok, r, end};
            pthread_create(&threads[i], NULL, cpu_gemm_worker, &tasks[i]);
            r = end;
            actual = i + 1;
        }
        for (int i = 0; i < actual; i++) pthread_join(threads[i], NULL);
        free(tasks); free(threads);
    } else {
        gemm_f16_f32_tokmajor(dst, W, src, n_out, n_in, n_tok, n_out, n_in);
    }
    if (bias) {
        for (int t = 0; t < n_tok; t++)
            for (int i = 0; i < n_out; i++)
                dst[t * n_out + i] += bias[i];
    }
}

/* ---- Flash Attention: tiled online softmax with K/V transpose ---- */

#define CPU_ATTN_TILE 64

typedef struct {
    const float *qkv;
    float *attn_out;
    int n_tok, dim, head_dim, n_heads;
    int h_start, h_end;
    float scale;
} cpu_attn_task;

#if defined(__AVX2__) && defined(__FMA__)

static void *cpu_attn_worker(void *arg) {
    cpu_attn_task *t = (cpu_attn_task *)arg;
    int N = t->n_tok, hd = t->head_dim, dim3 = 3 * t->dim;
    float scale = t->scale;

    /* Contiguous K/V buffers: stride hd instead of dim3 */
    float *K_buf = (float *)malloc((size_t)N * (size_t)hd * sizeof(float));
    float *V_buf = (float *)malloc((size_t)N * (size_t)hd * sizeof(float));
    float scores[CPU_ATTN_TILE];

    for (int h = t->h_start; h < t->h_end; h++) {
        /* Transpose K and V into contiguous per-head buffers */
        for (int ki = 0; ki < N; ki++) {
            const float *k_src = t->qkv + ki * dim3 + t->dim + h * hd;
            const float *v_src = t->qkv + ki * dim3 + 2 * t->dim + h * hd;
            float *k_dst = K_buf + ki * hd;
            float *v_dst = V_buf + ki * hd;
            for (int d = 0; d < hd; d += 8) {
                _mm256_storeu_ps(k_dst + d, _mm256_loadu_ps(k_src + d));
                _mm256_storeu_ps(v_dst + d, _mm256_loadu_ps(v_src + d));
            }
        }

        for (int qi = 0; qi < N; qi++) {
            const float *q_h = t->qkv + qi * dim3 + h * hd;

            /* Load Q into registers (head_dim=64 -> 8 __m256) */
            __m256 q0 = _mm256_loadu_ps(q_h);
            __m256 q1 = _mm256_loadu_ps(q_h + 8);
            __m256 q2 = _mm256_loadu_ps(q_h + 16);
            __m256 q3 = _mm256_loadu_ps(q_h + 24);
            __m256 q4 = _mm256_loadu_ps(q_h + 32);
            __m256 q5 = _mm256_loadu_ps(q_h + 40);
            __m256 q6 = _mm256_loadu_ps(q_h + 48);
            __m256 q7 = _mm256_loadu_ps(q_h + 56);

            /* Accumulators for weighted V sum */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();
            float running_max = -FLT_MAX;
            float running_sum = 0.0f;

            for (int ki_base = 0; ki_base < N; ki_base += CPU_ATTN_TILE) {
                int tile_end = ki_base + CPU_ATTN_TILE;
                if (tile_end > N) tile_end = N;
                int tile_len = tile_end - ki_base;

                /* Q @ K^T scores for this tile */
                float tile_max = -FLT_MAX;
                for (int j = 0; j < tile_len; j++) {
                    const float *k_j = K_buf + (ki_base + j) * hd;
                    __m256 dot = _mm256_mul_ps(q0, _mm256_loadu_ps(k_j));
                    dot = _mm256_fmadd_ps(q1, _mm256_loadu_ps(k_j + 8), dot);
                    dot = _mm256_fmadd_ps(q2, _mm256_loadu_ps(k_j + 16), dot);
                    dot = _mm256_fmadd_ps(q3, _mm256_loadu_ps(k_j + 24), dot);
                    dot = _mm256_fmadd_ps(q4, _mm256_loadu_ps(k_j + 32), dot);
                    dot = _mm256_fmadd_ps(q5, _mm256_loadu_ps(k_j + 40), dot);
                    dot = _mm256_fmadd_ps(q6, _mm256_loadu_ps(k_j + 48), dot);
                    dot = _mm256_fmadd_ps(q7, _mm256_loadu_ps(k_j + 56), dot);
                    float s = cpu_hsum_avx(dot) * scale;
                    scores[j] = s;
                    if (s > tile_max) tile_max = s;
                }

                /* Online softmax: rescale previous accumulator */
                float new_max = running_max > tile_max ? running_max : tile_max;
                float correction = expf(running_max - new_max);
                __m256 vc = _mm256_set1_ps(correction);
                running_sum *= correction;
                acc0 = _mm256_mul_ps(acc0, vc);
                acc1 = _mm256_mul_ps(acc1, vc);
                acc2 = _mm256_mul_ps(acc2, vc);
                acc3 = _mm256_mul_ps(acc3, vc);
                acc4 = _mm256_mul_ps(acc4, vc);
                acc5 = _mm256_mul_ps(acc5, vc);
                acc6 = _mm256_mul_ps(acc6, vc);
                acc7 = _mm256_mul_ps(acc7, vc);

                /* Accumulate weighted V for this tile */
                for (int j = 0; j < tile_len; j++) {
                    float w = expf(scores[j] - new_max);
                    running_sum += w;
                    __m256 vw = _mm256_set1_ps(w);
                    const float *v_j = V_buf + (ki_base + j) * hd;
                    acc0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_j), acc0);
                    acc1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_j + 8), acc1);
                    acc2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_j + 16), acc2);
                    acc3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_j + 24), acc3);
                    acc4 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_j + 32), acc4);
                    acc5 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_j + 40), acc5);
                    acc6 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_j + 48), acc6);
                    acc7 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_j + 56), acc7);
                }
                running_max = new_max;
            }

            /* Normalize and store */
            float inv_sum = 1.0f / running_sum;
            __m256 vinv = _mm256_set1_ps(inv_sum);
            float *out_h = t->attn_out + qi * t->dim + h * hd;
            _mm256_storeu_ps(out_h,      _mm256_mul_ps(acc0, vinv));
            _mm256_storeu_ps(out_h + 8,  _mm256_mul_ps(acc1, vinv));
            _mm256_storeu_ps(out_h + 16, _mm256_mul_ps(acc2, vinv));
            _mm256_storeu_ps(out_h + 24, _mm256_mul_ps(acc3, vinv));
            _mm256_storeu_ps(out_h + 32, _mm256_mul_ps(acc4, vinv));
            _mm256_storeu_ps(out_h + 40, _mm256_mul_ps(acc5, vinv));
            _mm256_storeu_ps(out_h + 48, _mm256_mul_ps(acc6, vinv));
            _mm256_storeu_ps(out_h + 56, _mm256_mul_ps(acc7, vinv));
        }
    }
    free(K_buf);
    free(V_buf);
    return NULL;
}

#else /* Scalar fallback */

static void *cpu_attn_worker(void *arg) {
    cpu_attn_task *t = (cpu_attn_task *)arg;
    int N = t->n_tok, hd = t->head_dim, dim3 = 3 * t->dim;
    float scale = t->scale;

    float *K_buf = (float *)malloc((size_t)N * (size_t)hd * sizeof(float));
    float *V_buf = (float *)malloc((size_t)N * (size_t)hd * sizeof(float));
    float scores[CPU_ATTN_TILE];

    for (int h = t->h_start; h < t->h_end; h++) {
        for (int ki = 0; ki < N; ki++) {
            memcpy(K_buf + ki * hd, t->qkv + ki * dim3 + t->dim + h * hd,
                   (size_t)hd * sizeof(float));
            memcpy(V_buf + ki * hd, t->qkv + ki * dim3 + 2 * t->dim + h * hd,
                   (size_t)hd * sizeof(float));
        }

        for (int qi = 0; qi < N; qi++) {
            const float *q_h = t->qkv + qi * dim3 + h * hd;
            float acc[64];
            memset(acc, 0, (size_t)hd * sizeof(float));
            float running_max = -FLT_MAX;
            float running_sum = 0.0f;

            for (int ki_base = 0; ki_base < N; ki_base += CPU_ATTN_TILE) {
                int tile_end = ki_base + CPU_ATTN_TILE;
                if (tile_end > N) tile_end = N;
                int tile_len = tile_end - ki_base;

                float tile_max = -FLT_MAX;
                for (int j = 0; j < tile_len; j++) {
                    const float *k_j = K_buf + (ki_base + j) * hd;
                    float dot = 0.0f;
                    for (int d = 0; d < hd; d++) dot += q_h[d] * k_j[d];
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > tile_max) tile_max = s;
                }

                float new_max = running_max > tile_max ? running_max : tile_max;
                float correction = expf(running_max - new_max);
                running_sum *= correction;
                for (int d = 0; d < hd; d++) acc[d] *= correction;

                for (int j = 0; j < tile_len; j++) {
                    float w = expf(scores[j] - new_max);
                    running_sum += w;
                    const float *v_j = V_buf + (ki_base + j) * hd;
                    for (int d = 0; d < hd; d++) acc[d] += w * v_j[d];
                }
                running_max = new_max;
            }

            float inv_sum = 1.0f / running_sum;
            float *out_h = t->attn_out + qi * t->dim + h * hd;
            for (int d = 0; d < hd; d++) out_h[d] = acc[d] * inv_sum;
        }
    }
    free(K_buf);
    free(V_buf);
    return NULL;
}

#endif /* AVX2+FMA */

static void cpu_attention(float *out, const float *qkv, int n_tok, int dim,
                           int n_heads, int head_dim, int n_threads) {
    float scale = 1.0f / sqrtf((float)head_dim);

    if (n_threads <= 1) {
        cpu_attn_task task = {qkv, out, n_tok, dim, head_dim, n_heads,
                              0, n_heads, scale};
        cpu_attn_worker(&task);
        return;
    }

    int nt = n_threads < n_heads ? n_threads : n_heads;
    cpu_attn_task *tasks = (cpu_attn_task *)calloc((size_t)nt, sizeof(cpu_attn_task));
    pthread_t *threads = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
    int heads_per = n_heads / nt;
    int extra = n_heads % nt;
    int h = 0;
    for (int i = 0; i < nt; i++) {
        int count = heads_per + (i < extra ? 1 : 0);
        tasks[i] = (cpu_attn_task){qkv, out, n_tok, dim, head_dim, n_heads,
                                    h, h + count, scale};
        h += count;
        pthread_create(&threads[i], NULL, cpu_attn_worker, &tasks[i]);
    }
    for (int i = 0; i < nt; i++) pthread_join(threads[i], NULL);
    free(tasks); free(threads);
}

#endif /* CPU_COMPUTE_IMPLEMENTATION */
#endif /* CPU_COMPUTE_H */
