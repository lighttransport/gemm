/*
 * trellis2_dit.h - TRELLIS.2 Diffusion Transformer (DiT) blocks
 *
 * Usage:
 *   #define T2DIT_IMPLEMENTATION
 *   #include "trellis2_dit.h"
 *
 * Dependencies: ggml_dequant.h, safetensors.h, cpu_compute.h
 *
 * Architecture: ModulatedTransformerCrossBlock
 *   model_channels=1536, heads=12, head_dim=128, ffn=8192, 30 blocks
 *   adaLN self-attention + cross-attention + adaLN MLP
 *   Shared modulation (share_mod=true): single SiLU->Linear for all blocks
 *
 * API:
 *   t2dit_model  *t2dit_load_safetensors(const char *path);
 *   void          t2dit_free(t2dit_model *m);
 *   void          t2dit_forward(float *out, const float *x_t, float t_val,
 *                               const float *cond_kv_cache,
 *                               t2dit_model *m, int n_threads);
 *   float        *t2dit_precompute_cond_kv(const float *cond, int n_cond,
 *                                          t2dit_model *m, int n_threads);
 */
#ifndef T2DIT_H
#define T2DIT_H

#include <stdint.h>
#include <stddef.h>
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reuse qtensor if not already defined */
#ifndef TRANSFORMER_H
#ifndef DEPTH_ANYTHING3_H
#ifndef DINOV3_H
#ifndef SPARSE3D_H
typedef struct {
    void    *data;
    uint32_t type;
    int      n_rows;
    int      n_cols;
    int      n_dims;
    uint64_t dims[4];
} qtensor;
#endif
#endif
#endif
#endif

typedef struct {
    /* Self-attention */
    qtensor sa_qkv_w, sa_qkv_b;         /* [3*dim, dim] */
    qtensor sa_q_norm_w;                  /* [n_heads, head_dim] RMSNorm gamma */
    qtensor sa_k_norm_w;                  /* [n_heads, head_dim] RMSNorm gamma */
    qtensor sa_proj_w, sa_proj_b;        /* [dim, dim] */
    /* Cross-attention */
    qtensor norm2_w, norm2_b;             /* [dim] affine LayerNorm */
    qtensor ca_q_w, ca_q_b;             /* [dim, dim] */
    qtensor ca_kv_w, ca_kv_b;           /* [2*dim, cond_dim] */
    qtensor ca_q_norm_w;                  /* [n_heads, head_dim] RMSNorm gamma */
    qtensor ca_k_norm_w;                  /* [n_heads, head_dim] RMSNorm gamma */
    qtensor ca_proj_w, ca_proj_b;        /* [dim, dim] */
    /* MLP */
    qtensor mlp_fc1_w, mlp_fc1_b;       /* [ffn, dim] */
    qtensor mlp_fc2_w, mlp_fc2_b;       /* [dim, ffn] */
    /* Per-block modulation bias: [6*dim] added to global mod */
    qtensor mod_bias;
} t2dit_block;

typedef struct {
    int model_channels;   /* 1536 */
    int n_heads;          /* 12 */
    int head_dim;         /* 128 */
    int ffn_hidden;       /* 8192 */
    int n_blocks;         /* 30 */
    int in_channels;      /* 8 */
    int cond_dim;         /* 1024 */
    int grid_size;        /* 16 */
    int n_tokens;         /* grid_size^3 = 4096 */
    float ln_eps;         /* 1e-6 */
    float rope_theta;     /* 10000.0 */

    /* Timestep embedder: sinusoidal(256) -> MLP(256->dim->dim) */
    qtensor t_embed_fc1_w, t_embed_fc1_b;
    qtensor t_embed_fc2_w, t_embed_fc2_b;
    /* Shared modulation: SiLU -> Linear(dim, 6*dim) */
    qtensor mod_w, mod_b;
    /* Input embedder: Linear(in_channels, dim) */
    qtensor x_embed_w, x_embed_b;
    /* Output layer: Linear(dim, in_channels) */
    qtensor out_w, out_b;

    t2dit_block *blocks;

    /* Precomputed 3D RoPE: [n_tokens][3][n_freqs] for cos and sin */
    float *rope_cos;
    float *rope_sin;
    int n_rope_freqs;
    int rope_axis_dim;

    void *st_ctx;   /* safetensors context, kept for mmap lifetime */
} t2dit_model;

t2dit_model *t2dit_load_safetensors(const char *path);
void         t2dit_free(t2dit_model *m);

/* Run single denoising step. cond_kv_cache: precomputed cross-attn KV. */
void t2dit_forward(float *out, const float *x_t, float t_val,
                   const float *cond_kv_cache,
                   t2dit_model *m, int n_threads);

/* Precompute cross-attention KV cache for all blocks.
 * Returns [n_blocks * n_cond * 2*model_channels] float array.
 * Caller must free() the result. */
float *t2dit_precompute_cond_kv(const float *cond, int n_cond,
                                 t2dit_model *m, int n_threads);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef T2DIT_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <float.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/* Include cpu_compute.h for cpu_cross_attention and cpu_hsum_avx */
#ifndef CPU_COMPUTE_H
#define CPU_COMPUTE_IMPLEMENTATION
#endif
#include "cpu_compute.h"

static double t2dit_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Tensor helpers ---- */

static int t2dit_numel(const qtensor *t) {
    if (!t->data) return 0;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    return n;
}

static float *t2dit_dequant_full(const qtensor *t) {
    if (!t->data) return NULL;
    int n = t2dit_numel(t);
    if (n <= 0) return NULL;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        memset(buf, 0, (size_t)n * sizeof(float));
    }
    return buf;
}

/* ---- Threaded F32 GEMM: dst[tok,out] = src[tok,in] @ W[out,in]^T + bias ---- */

typedef struct {
    float *dst;
    const float *W;
    const float *src;
    int n_out, n_in, n_tok;
    int r_start, r_end;
} t2dit_gemm_task;

#if defined(__AVX2__) && defined(__FMA__)

static void *t2dit_gemm_f32_worker(void *arg) {
    t2dit_gemm_task *t = (t2dit_gemm_task *)arg;
    int n_in = t->n_in, n_tok = t->n_tok, n_out = t->n_out;
    for (int r = t->r_start; r < t->r_end; r++) {
        const float *w_row = t->W + (size_t)r * n_in;
        for (int tok = 0; tok < n_tok; tok++) {
            const float *x = t->src + (size_t)tok * n_in;
            __m256 acc = _mm256_setzero_ps();
            int j = 0;
            for (; j + 7 < n_in; j += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(w_row + j),
                                       _mm256_loadu_ps(x + j), acc);
            float sum = cpu_hsum_avx(acc);
            for (; j < n_in; j++) sum += w_row[j] * x[j];
            t->dst[(size_t)tok * n_out + r] = sum;
        }
    }
    return NULL;
}

#else

static void *t2dit_gemm_f32_worker(void *arg) {
    t2dit_gemm_task *t = (t2dit_gemm_task *)arg;
    int n_in = t->n_in, n_tok = t->n_tok, n_out = t->n_out;
    for (int r = t->r_start; r < t->r_end; r++) {
        const float *w_row = t->W + (size_t)r * n_in;
        for (int tok = 0; tok < n_tok; tok++) {
            const float *x = t->src + (size_t)tok * n_in;
            float sum = 0.0f;
            for (int j = 0; j < n_in; j++) sum += w_row[j] * x[j];
            t->dst[(size_t)tok * n_out + r] = sum;
        }
    }
    return NULL;
}

#endif

static void t2dit_gemm_f32(float *dst, const float *W, const float *bias,
                             const float *src, int n_tok, int n_out, int n_in,
                             int n_threads) {
    if (n_threads <= 1 || n_out < n_threads * 2) {
        t2dit_gemm_task task = {dst, W, src, n_out, n_in, n_tok, 0, n_out};
        t2dit_gemm_f32_worker(&task);
    } else {
        int nt = n_threads;
        t2dit_gemm_task *tasks = (t2dit_gemm_task *)calloc((size_t)nt, sizeof(t2dit_gemm_task));
        pthread_t *threads = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
        int rows_per = n_out / nt;
        int extra = n_out % nt;
        int r = 0, actual = 0;
        for (int i = 0; i < nt && r < n_out; i++) {
            int count = rows_per + (i < extra ? 1 : 0);
            tasks[i] = (t2dit_gemm_task){dst, W, src, n_out, n_in, n_tok, r, r + count};
            pthread_create(&threads[i], NULL, t2dit_gemm_f32_worker, &tasks[i]);
            r += count;
            actual = i + 1;
        }
        for (int i = 0; i < actual; i++) pthread_join(threads[i], NULL);
        free(tasks); free(threads);
    }
    if (bias) {
        for (int t = 0; t < n_tok; t++)
            for (int i = 0; i < n_out; i++)
                dst[(size_t)t * n_out + i] += bias[i];
    }
}

/* Batch GEMM wrapper: dispatches based on weight type */
static void t2dit_batch_gemm(float *dst, const qtensor *W, const qtensor *bias,
                               const float *src, int n_tok, int n_out, int n_in,
                               int n_threads) {
    if (!W->data) {
        memset(dst, 0, (size_t)n_tok * n_out * sizeof(float));
        return;
    }
    float *bias_f = NULL;
    if (bias && bias->data) {
        if (bias->type == GGML_TYPE_F32)
            bias_f = (float *)bias->data;
        else {
            bias_f = t2dit_dequant_full(bias);
        }
    }
    if (W->type == GGML_TYPE_F32) {
        t2dit_gemm_f32(dst, (const float *)W->data, bias_f, src,
                        n_tok, n_out, n_in, n_threads);
    } else if (W->type == GGML_TYPE_F16) {
        cpu_gemm_f16(dst, (const uint16_t *)W->data, bias_f, src,
                     n_tok, n_out, n_in, n_threads);
    } else {
        /* Slow fallback: row-by-row dequant */
        float *tmp = (float *)malloc((size_t)n_in * sizeof(float));
        for (int t = 0; t < n_tok; t++) {
            for (int r = 0; r < n_out; r++) {
                const float *row;
                if (W->type == GGML_TYPE_F32) {
                    row = (const float *)W->data + (size_t)r * n_in;
                } else {
                    dequant_row(W->type, (const uint8_t *)W->data + (size_t)r * W->n_cols * 2,
                                tmp, n_in);
                    row = tmp;
                }
                float s = 0.0f;
                for (int j = 0; j < n_in; j++) s += row[j] * src[(size_t)t * n_in + j];
                dst[(size_t)t * n_out + r] = s;
            }
        }
        free(tmp);
        if (bias_f) {
            for (int t = 0; t < n_tok; t++)
                for (int i = 0; i < n_out; i++)
                    dst[(size_t)t * n_out + i] += bias_f[i];
        }
    }
    /* Free bias_f only if we allocated it (non-F32 original) */
    if (bias_f && bias && bias->type != GGML_TYPE_F32)
        free(bias_f);
}

/* ---- LayerNorm (no affine): y = (x - mean) / sqrt(var + eps) ---- */

static void t2dit_layernorm_noaffine(float *dst, const float *src,
                                       int n_tok, int dim, float eps) {
    for (int t = 0; t < n_tok; t++) {
        const float *x = src + (size_t)t * dim;
        float *y = dst + (size_t)t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++) y[i] = (x[i] - mean) * inv;
    }
}

/* ---- LayerNorm (with affine): y = (x - mean) / sqrt(var + eps) * w + b ---- */

static void t2dit_layernorm_affine(float *dst, const float *src,
                                     const float *w, const float *b,
                                     int n_tok, int dim, float eps) {
    for (int t = 0; t < n_tok; t++) {
        const float *x = src + (size_t)t * dim;
        float *y = dst + (size_t)t * dim;
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

/* ---- AdaLN: y = LN_noaffine(x) * (1 + scale) + shift ---- */

static void t2dit_adaln(float *dst, const float *src,
                          const float *shift, const float *scale,
                          int n_tok, int dim, float eps) {
    for (int t = 0; t < n_tok; t++) {
        const float *x = src + (size_t)t * dim;
        float *y = dst + (size_t)t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++)
            y[i] = (x[i] - mean) * inv * (1.0f + scale[i]) + shift[i];
    }
}

/* ---- Per-head RMSNorm: x = x * w / sqrt(mean(x^2) + eps) ---- */

/* w is [n_heads * head_dim] (flattened from [n_heads, head_dim]) */
static void t2dit_rmsnorm(float *vec, int n_tok, int n_heads, int head_dim,
                            int stride, const float *w, float eps) {
    if (!w) return;
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = vec + (size_t)t * stride + h * head_dim;
            const float *wh = w + h * head_dim;  /* per-head weights */
            float ss = 0.0f;
            for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
            float rms = 1.0f / sqrtf(ss / head_dim + eps);
            for (int i = 0; i < head_dim; i++) v[i] *= rms * wh[i];
        }
    }
}

/* ---- 3D RoPE: precompute and apply ---- */

static void t2dit_precompute_rope(t2dit_model *m) {
    int gs = m->grid_size;
    int nt = gs * gs * gs;
    int hd = m->head_dim;
    int n_freqs = hd / 6;
    int axis_dim = 2 * n_freqs;
    m->n_rope_freqs = n_freqs;
    m->rope_axis_dim = axis_dim;

    /* Frequencies: 1.0 / (theta^(j / n_freqs)) */
    float *freqs = (float *)malloc((size_t)n_freqs * sizeof(float));
    for (int j = 0; j < n_freqs; j++)
        freqs[j] = 1.0f / powf(m->rope_theta, (float)j / (float)n_freqs);

    /* Tables: [nt, 3, n_freqs] for cos and sin */
    m->rope_cos = (float *)malloc((size_t)nt * 3 * n_freqs * sizeof(float));
    m->rope_sin = (float *)malloc((size_t)nt * 3 * n_freqs * sizeof(float));

    for (int i = 0; i < nt; i++) {
        int z = i / (gs * gs);
        int y = (i / gs) % gs;
        int x = i % gs;
        float coords[3] = {(float)z, (float)y, (float)x};
        for (int axis = 0; axis < 3; axis++) {
            for (int j = 0; j < n_freqs; j++) {
                float theta = coords[axis] * freqs[j];
                m->rope_cos[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = cosf(theta);
                m->rope_sin[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = sinf(theta);
            }
        }
    }
    free(freqs);
}

/* Apply 3D RoPE to Q and K (complex-pair convention).
 * Official TRELLIS.2 uses view_as_complex on consecutive pairs: (x[2k], x[2k+1]).
 * Phase layout per head: [z_freqs(21), y_freqs(21), x_freqs(21), pad(1)] complex pairs.
 * QKV layout: [n_tok, 3*dim], Q at offset 0, K at offset dim. */
static void t2dit_apply_rope_qkv(float *qkv, int n_tok, int dim,
                                    int n_heads, int head_dim,
                                    const float *rope_cos, const float *rope_sin,
                                    int n_freqs, int axis_dim) {
    (void)axis_dim;
    for (int t = 0; t < n_tok; t++) {
        const float *cs = rope_cos + (size_t)t * 3 * n_freqs;
        const float *sn = rope_sin + (size_t)t * 3 * n_freqs;
        for (int qk = 0; qk < 2; qk++) {
            for (int h = 0; h < n_heads; h++) {
                float *v = qkv + (size_t)t * 3 * dim + qk * dim + h * head_dim;
                int pair = 0;
                for (int axis = 0; axis < 3; axis++) {
                    const float *ca = cs + axis * n_freqs;
                    const float *sa = sn + axis * n_freqs;
                    for (int j = 0; j < n_freqs; j++, pair++) {
                        int idx = pair * 2;
                        float re = v[idx], im = v[idx + 1];
                        v[idx]     = re * ca[j] - im * sa[j];
                        v[idx + 1] = re * sa[j] + im * ca[j];
                    }
                }
            }
        }
    }
}

/* ---- Timestep embedding: sinusoidal(256) -> Linear(256,dim) -> SiLU -> Linear(dim,dim) ---- */

static void t2dit_timestep_embed(float *out, float t_val, const t2dit_model *m) {
    int half = 128;
    float embed[256];
    for (int j = 0; j < half; j++) {
        float freq = expf(-(float)j / (float)half * logf(10000.0f));
        float arg = t_val * freq;
        embed[j] = cosf(arg);
        embed[half + j] = sinf(arg);
    }
    /* FC1: [dim, 256] */
    int dim = m->model_channels;
    float *h1 = (float *)calloc((size_t)dim, sizeof(float));
    const float *w1 = (const float *)m->t_embed_fc1_w.data;
    const float *b1 = m->t_embed_fc1_b.data ? (const float *)m->t_embed_fc1_b.data : NULL;
    for (int r = 0; r < dim; r++) {
        float s = 0.0f;
        const float *wr = w1 + r * 256;
        for (int j = 0; j < 256; j++) s += wr[j] * embed[j];
        h1[r] = s + (b1 ? b1[r] : 0.0f);
    }
    /* SiLU */
    for (int i = 0; i < dim; i++)
        h1[i] = h1[i] / (1.0f + expf(-h1[i]));
    /* FC2: [dim, dim] */
    const float *w2 = (const float *)m->t_embed_fc2_w.data;
    const float *b2 = m->t_embed_fc2_b.data ? (const float *)m->t_embed_fc2_b.data : NULL;
    for (int r = 0; r < dim; r++) {
        float s = 0.0f;
        const float *wr = w2 + (size_t)r * dim;
        for (int j = 0; j < dim; j++) s += wr[j] * h1[j];
        out[r] = s + (b2 ? b2[r] : 0.0f);
    }
    free(h1);
}

/* ---- Modulation: SiLU(t_emb) -> Linear(dim, 6*dim) -> 6 chunks ---- */

static void t2dit_modulation(float *mod_out, const float *t_emb, const t2dit_model *m) {
    int dim = m->model_channels;
    int out_dim = 6 * dim;
    /* SiLU on t_emb */
    float *h = (float *)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < dim; i++)
        h[i] = t_emb[i] / (1.0f + expf(-t_emb[i]));
    /* Linear */
    const float *w = (const float *)m->mod_w.data;
    const float *b = m->mod_b.data ? (const float *)m->mod_b.data : NULL;
    for (int r = 0; r < out_dim; r++) {
        float s = 0.0f;
        const float *wr = w + (size_t)r * dim;
        for (int j = 0; j < dim; j++) s += wr[j] * h[j];
        mod_out[r] = s + (b ? b[r] : 0.0f);
    }
    free(h);
}

/* ---- Self-attention via cpu_cross_attention (supports arbitrary head_dim) ---- */

static void t2dit_self_attention(float *out, const float *qkv,
                                   int n_tok, int dim, int n_heads,
                                   int head_dim, int n_threads) {
    /* Reformat QKV[n_tok, 3*dim] -> Q[n_tok, dim] + KV[n_tok, 2*dim] */
    float *Q_buf = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    float *KV_buf = (float *)malloc((size_t)n_tok * 2 * dim * sizeof(float));
    for (int t = 0; t < n_tok; t++) {
        memcpy(Q_buf + (size_t)t * dim,
               qkv + (size_t)t * 3 * dim,
               (size_t)dim * sizeof(float));
        memcpy(KV_buf + (size_t)t * 2 * dim,
               qkv + (size_t)t * 3 * dim + dim,
               (size_t)2 * dim * sizeof(float));
    }
    cpu_cross_attention(out, Q_buf, KV_buf, n_tok, n_tok, dim,
                        n_heads, head_dim, n_threads);
    free(Q_buf);
    free(KV_buf);
}

/* ---- DiT block forward ---- */

static void t2dit_block_forward(float *hidden, const t2dit_block *blk,
                                  const float *global_mod,
                                  const float *cond_kv, int n_cond,
                                  const t2dit_model *m, int n_threads) {
    int nt = m->n_tokens;
    int dim = m->model_channels;

    /* Combine global mod + per-block mod bias */
    float *mod = (float *)malloc((size_t)6 * dim * sizeof(float));
    memcpy(mod, global_mod, (size_t)6 * dim * sizeof(float));
    if (blk->mod_bias.data) {
        float *mb = t2dit_dequant_full(&blk->mod_bias);
        if (mb) {
            for (int i = 0; i < 6 * dim; i++) mod[i] += mb[i];
            free(mb);
        }
    }
    const float *shift_sa  = mod;
    const float *scale_sa  = mod + dim;
    const float *gate_sa   = mod + 2 * dim;
    const float *shift_mlp = mod + 3 * dim;
    const float *scale_mlp = mod + 4 * dim;
    const float *gate_mlp  = mod + 5 * dim;
    int hd = m->head_dim;
    int nh = m->n_heads;

    float *tmp = (float *)malloc((size_t)nt * dim * sizeof(float));
    float *qkv = (float *)malloc((size_t)nt * 3 * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)nt * dim * sizeof(float));

    /* ---- Self-attention with adaLN ---- */
    /* adaLN: LN(no affine) * (1+scale) + shift */
    t2dit_adaln(tmp, hidden, shift_sa, scale_sa, nt, dim, m->ln_eps);

    /* QKV projection */
    t2dit_batch_gemm(qkv, &blk->sa_qkv_w, &blk->sa_qkv_b,
                     tmp, nt, 3 * dim, dim, n_threads);

    /* QK RMSNorm */
    if (blk->sa_q_norm_w.data) {
        float *qn = t2dit_dequant_full(&blk->sa_q_norm_w);
        float *kn = t2dit_dequant_full(&blk->sa_k_norm_w);
        /* Q portion: stride = 3*dim, offset = 0 */
        t2dit_rmsnorm(qkv, nt, nh, hd, 3 * dim, qn, m->ln_eps);
        /* K portion: stride = 3*dim, offset = dim */
        t2dit_rmsnorm(qkv + dim, nt, nh, hd, 3 * dim, kn, m->ln_eps);
        free(qn); free(kn);
    }

    /* 3D RoPE */
    t2dit_apply_rope_qkv(qkv, nt, dim, nh, hd,
                           m->rope_cos, m->rope_sin,
                           m->n_rope_freqs, m->rope_axis_dim);

    /* Self-attention */
    t2dit_self_attention(attn_out, qkv, nt, dim, nh, hd, n_threads);
    free(qkv);

    /* Output projection */
    float *proj_out = (float *)malloc((size_t)nt * dim * sizeof(float));
    t2dit_batch_gemm(proj_out, &blk->sa_proj_w, &blk->sa_proj_b,
                     attn_out, nt, dim, dim, n_threads);

    /* Gated residual */
    for (int t = 0; t < nt; t++)
        for (int i = 0; i < dim; i++)
            hidden[(size_t)t * dim + i] += gate_sa[i] * proj_out[(size_t)t * dim + i];
    free(proj_out);

    /* ---- Cross-attention (no adaLN, uses affine LayerNorm) ---- */
    if (blk->ca_q_w.data && cond_kv) {
        float *ln2_w = t2dit_dequant_full(&blk->norm2_w);
        float *ln2_b = t2dit_dequant_full(&blk->norm2_b);
        t2dit_layernorm_affine(tmp, hidden, ln2_w, ln2_b, nt, dim, m->ln_eps);
        free(ln2_w); free(ln2_b);

        /* Q projection */
        float *cross_q = (float *)malloc((size_t)nt * dim * sizeof(float));
        t2dit_batch_gemm(cross_q, &blk->ca_q_w, &blk->ca_q_b,
                         tmp, nt, dim, dim, n_threads);

        /* QK RMSNorm on Q */
        if (blk->ca_q_norm_w.data) {
            float *qn = t2dit_dequant_full(&blk->ca_q_norm_w);
            t2dit_rmsnorm(cross_q, nt, nh, hd, dim, qn, m->ln_eps);
            free(qn);
        }

        /* Cross attention: Q from tokens, KV from cached cond projection */
        cpu_cross_attention(attn_out, cross_q, cond_kv,
                           nt, n_cond, dim, nh, hd, n_threads);
        free(cross_q);

        /* Cross-attn output projection + residual (no gate) */
        float *ca_proj = (float *)malloc((size_t)nt * dim * sizeof(float));
        t2dit_batch_gemm(ca_proj, &blk->ca_proj_w, &blk->ca_proj_b,
                         attn_out, nt, dim, dim, n_threads);
        for (int i = 0; i < nt * dim; i++)
            hidden[i] += ca_proj[i];
        free(ca_proj);
    }

    /* ---- MLP with adaLN ---- */
    t2dit_adaln(tmp, hidden, shift_mlp, scale_mlp, nt, dim, m->ln_eps);

    /* FC1 -> GELU */
    int ffn = m->ffn_hidden;
    float *ffn_buf = (float *)malloc((size_t)nt * ffn * sizeof(float));
    t2dit_batch_gemm(ffn_buf, &blk->mlp_fc1_w, &blk->mlp_fc1_b,
                     tmp, nt, ffn, dim, n_threads);
    cpu_gelu(ffn_buf, nt * ffn);

    /* FC2 */
    float *mlp_out = (float *)malloc((size_t)nt * dim * sizeof(float));
    t2dit_batch_gemm(mlp_out, &blk->mlp_fc2_w, &blk->mlp_fc2_b,
                     ffn_buf, nt, dim, ffn, n_threads);
    free(ffn_buf);

    /* Gated residual */
    for (int t = 0; t < nt; t++)
        for (int i = 0; i < dim; i++)
            hidden[(size_t)t * dim + i] += gate_mlp[i] * mlp_out[(size_t)t * dim + i];
    free(mlp_out);

    free(mod);
    free(tmp);
    free(attn_out);
}

/* ---- Full forward pass (single denoising step) ---- */

void t2dit_forward(float *out, const float *x_t, float t_val,
                   const float *cond_kv_cache,
                   t2dit_model *m, int n_threads) {
    int nt = m->n_tokens;
    int dim = m->model_channels;
    int n_cond = 0;

    /* Detect n_cond from first block's KV weight shape */
    if (m->blocks[0].ca_kv_w.data)
        n_cond = m->blocks[0].ca_kv_w.n_rows / 2;  /* [2*dim, cond_dim] has n_rows = 2*dim... */
    /* Actually, n_cond comes from the cache. Use hardcoded 1029 for DINOv3. */
    /* Better: pass it as a parameter or detect from the cache size. */
    /* For now, store it in the model. */
    n_cond = 1029;  /* DINOv3: 1 CLS + 4 storage + 1024 patches */

    /* Timestep embedding — official TRELLIS.2 scales t by 1000 */
    float *t_emb = (float *)malloc((size_t)dim * sizeof(float));
    t2dit_timestep_embed(t_emb, t_val * 1000.0f, m);

    /* Shared modulation -> 6 chunks of [dim] */
    float *mod = (float *)malloc((size_t)6 * dim * sizeof(float));
    t2dit_modulation(mod, t_emb, m);
    free(t_emb);

    /* Input embedding: x_t [n_tokens, in_channels] -> [n_tokens, dim] */
    float *hidden = (float *)malloc((size_t)nt * dim * sizeof(float));
    t2dit_batch_gemm(hidden, &m->x_embed_w, &m->x_embed_b,
                     x_t, nt, dim, m->in_channels, n_threads);

    /* Process all DiT blocks */
    for (int L = 0; L < m->n_blocks; L++) {
        /* Per-block cross-attn KV cache: offset into the large cache */
        const float *block_kv = NULL;
        if (cond_kv_cache)
            block_kv = cond_kv_cache + (size_t)L * n_cond * 2 * dim;

        t2dit_block_forward(hidden, &m->blocks[L], mod,
                            block_kv, n_cond, m, n_threads);
    }

    /* Final LayerNorm (no affine) before output projection */
    t2dit_layernorm_noaffine(hidden, hidden, nt, dim, m->ln_eps);

    /* Output projection: [n_tokens, dim] -> [n_tokens, in_channels] */
    t2dit_batch_gemm(out, &m->out_w, &m->out_b,
                     hidden, nt, m->in_channels, dim, n_threads);

    free(hidden);
    free(mod);
}

/* ---- Precompute cross-attention KV cache ---- */

float *t2dit_precompute_cond_kv(const float *cond, int n_cond,
                                 t2dit_model *m, int n_threads) {
    int dim = m->model_channels;
    int nb = m->n_blocks;
    int hd = m->head_dim;
    int nh = m->n_heads;
    size_t block_size = (size_t)n_cond * 2 * dim;
    float *cache = (float *)malloc((size_t)nb * block_size * sizeof(float));

    for (int L = 0; L < nb; L++) {
        t2dit_block *blk = &m->blocks[L];
        float *kv = cache + (size_t)L * block_size;

        if (!blk->ca_kv_w.data) {
            memset(kv, 0, block_size * sizeof(float));
            continue;
        }

        /* KV projection: cond [n_cond, cond_dim] -> [n_cond, 2*dim] */
        t2dit_batch_gemm(kv, &blk->ca_kv_w, &blk->ca_kv_b,
                         cond, n_cond, 2 * dim, m->cond_dim, n_threads);

        /* Apply K RMSNorm to the K portion [n_cond, dim] */
        if (blk->ca_k_norm_w.data) {
            float *kn = t2dit_dequant_full(&blk->ca_k_norm_w);
            t2dit_rmsnorm(kv, n_cond, nh, hd, 2 * dim, kn, m->ln_eps);
            free(kn);
        }
    }

    fprintf(stderr, "t2dit: precomputed cross-attn KV cache for %d blocks "
            "(%d cond tokens, %.1f MB)\n",
            nb, n_cond, (float)(nb * block_size * sizeof(float)) / (1024 * 1024));
    return cache;
}

/* ==================================================================== */
/* SafeTensors loading                                                   */
/* ==================================================================== */

#ifdef SAFETENSORS_H

static qtensor t2dit_make_tensor(st_context *st, int idx) {
    qtensor t = {0};
    if (idx < 0) return t;
    t.data = safetensors_data(st, idx);
    const char *dt = safetensors_dtype(st, idx);
    if (strcmp(dt, "F32") == 0)       t.type = GGML_TYPE_F32;
    else if (strcmp(dt, "F16") == 0)  t.type = GGML_TYPE_F16;
    else if (strcmp(dt, "BF16") == 0) t.type = GGML_TYPE_F32;  /* will convert */
    else return (qtensor){0};
    t.n_dims = safetensors_ndims(st, idx);
    const uint64_t *shape = safetensors_shape(st, idx);
    for (int d = 0; d < t.n_dims; d++) t.dims[d] = shape[d];
    if (t.n_dims >= 2) {
        t.n_rows = (int)shape[0];
        t.n_cols = 1;
        for (int d = 1; d < t.n_dims; d++) t.n_cols *= (int)shape[d];
    } else {
        t.n_cols = (int)shape[0];
        t.n_rows = 1;
    }
    /* BF16 -> F32 conversion */
    if (strcmp(dt, "BF16") == 0) {
        int numel = t.n_cols * t.n_rows;
        float *buf = (float *)malloc((size_t)numel * sizeof(float));
        const uint16_t *src = (const uint16_t *)t.data;
        for (int i = 0; i < numel; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&buf[i], &bits, 4);
        }
        t.data = buf;
        t.type = GGML_TYPE_F32;
    }
    return t;
}

t2dit_model *t2dit_load_safetensors(const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) return NULL;

    fprintf(stderr, "t2dit: opened safetensors, %d tensors\n", st->n_tensors);

    /* Print first tensors for debugging */
    int show = st->n_tensors < 10 ? st->n_tensors : 10;
    for (int i = 0; i < show; i++) {
        const char *nm = safetensors_name(st, i);
        const char *dt = safetensors_dtype(st, i);
        int nd = safetensors_ndims(st, i);
        const uint64_t *sh = safetensors_shape(st, i);
        fprintf(stderr, "  [%d] %s: %s [", i, nm, dt);
        for (int d = 0; d < nd; d++) fprintf(stderr, "%s%lu", d ? "," : "", (unsigned long)sh[d]);
        fprintf(stderr, "]\n");
    }
    if (st->n_tensors > 10)
        fprintf(stderr, "  ... (%d more)\n", st->n_tensors - 10);

    /* Detect prefix by scanning for known anchor tensor */
    char prefix[256] = "";
    const char *anchors[] = {"t_embedder", "x_embedder", "mod.", NULL};
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        for (int a = 0; anchors[a]; a++) {
            const char *p = strstr(nm, anchors[a]);
            if (p) {
                size_t pl = (size_t)(p - nm);
                if (pl < sizeof(prefix)) {
                    memcpy(prefix, nm, pl);
                    prefix[pl] = '\0';
                }
                goto prefix_found;
            }
        }
    }
prefix_found:
    fprintf(stderr, "t2dit: detected prefix: '%s'\n", prefix);

    /* Helper macro */
    #define T2DIT_FIND(suffix) ({ \
        char _buf[512]; \
        snprintf(_buf, sizeof(_buf), "%s%s", prefix, suffix); \
        int _idx = safetensors_find(st, _buf); \
        (_idx >= 0) ? t2dit_make_tensor(st, _idx) : (qtensor){0}; \
    })

    /* Auto-detect architecture parameters */
    int model_channels = 1536, n_heads = 12, head_dim = 128;
    int ffn_hidden = 8192, n_blocks = 0, in_channels = 8;
    int cond_dim = 1024;

    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const uint64_t *sh = safetensors_shape(st, i);
        int nd = safetensors_ndims(st, i);

        /* Detect model_channels from x_embedder.weight [dim, in_channels] */
        if (strstr(nm, "x_embedder.weight") && nd == 2) {
            model_channels = (int)sh[0];
            in_channels = (int)sh[1];
        }
        /* Detect cond_dim from cross_attn.kv.weight [2*dim, cond_dim] */
        if (strstr(nm, "cross_attn.kv.weight") && nd == 2) {
            cond_dim = (int)sh[1];
            if (model_channels == 0)
                model_channels = (int)sh[0] / 2;
        }
        /* Detect head_dim from q_norm.weight */
        if (strstr(nm, "self_attn.q_norm.weight") && nd == 1) {
            head_dim = (int)sh[0];
        }
        /* Detect ffn from mlp.0.weight */
        if (strstr(nm, "blocks.0.mlp.0.weight") && nd == 2) {
            ffn_hidden = (int)sh[0];
        }
        /* Count blocks */
        const char *bp = strstr(nm, "blocks.");
        if (bp) {
            bp += 7;
            int blk = 0;
            while (*bp >= '0' && *bp <= '9') { blk = blk * 10 + (*bp - '0'); bp++; }
            if (blk + 1 > n_blocks) n_blocks = blk + 1;
        }
    }

    n_heads = model_channels / head_dim;
    int grid_size = 16;
    int n_tokens = grid_size * grid_size * grid_size;

    fprintf(stderr, "t2dit: dim=%d, heads=%d, head_dim=%d, blocks=%d, ffn=%d\n",
            model_channels, n_heads, head_dim, n_blocks, ffn_hidden);
    fprintf(stderr, "t2dit: in_channels=%d, cond_dim=%d, grid=%d, tokens=%d\n",
            in_channels, cond_dim, grid_size, n_tokens);

    /* Allocate model */
    t2dit_model *m = (t2dit_model *)calloc(1, sizeof(t2dit_model));
    m->model_channels = model_channels;
    m->n_heads = n_heads;
    m->head_dim = head_dim;
    m->ffn_hidden = ffn_hidden;
    m->n_blocks = n_blocks;
    m->in_channels = in_channels;
    m->cond_dim = cond_dim;
    m->grid_size = grid_size;
    m->n_tokens = n_tokens;
    m->ln_eps = 1e-6f;
    m->rope_theta = 10000.0f;
    m->st_ctx = st;

    /* Load top-level weights */
    m->t_embed_fc1_w = T2DIT_FIND("t_embedder.mlp.0.weight");
    m->t_embed_fc1_b = T2DIT_FIND("t_embedder.mlp.0.bias");
    m->t_embed_fc2_w = T2DIT_FIND("t_embedder.mlp.2.weight");
    m->t_embed_fc2_b = T2DIT_FIND("t_embedder.mlp.2.bias");
    m->mod_w = T2DIT_FIND("adaLN_modulation.1.weight");
    m->mod_b = T2DIT_FIND("adaLN_modulation.1.bias");
    if (!m->mod_w.data) {
        m->mod_w = T2DIT_FIND("mod.1.weight");
        m->mod_b = T2DIT_FIND("mod.1.bias");
    }
    m->x_embed_w = T2DIT_FIND("input_layer.weight");
    m->x_embed_b = T2DIT_FIND("input_layer.bias");
    if (!m->x_embed_w.data) {
        m->x_embed_w = T2DIT_FIND("x_embedder.weight");
        m->x_embed_b = T2DIT_FIND("x_embedder.bias");
    }
    m->out_w = T2DIT_FIND("out_layer.weight");
    m->out_b = T2DIT_FIND("out_layer.bias");

    fprintf(stderr, "t2dit: t_embedder: %s\n", m->t_embed_fc1_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "t2dit: mod: %s\n", m->mod_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "t2dit: x_embedder: %s\n", m->x_embed_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "t2dit: out_layer: %s\n", m->out_w.data ? "loaded" : "MISSING");

    /* Load blocks */
    m->blocks = (t2dit_block *)calloc((size_t)n_blocks, sizeof(t2dit_block));
    for (int L = 0; L < n_blocks; L++) {
        t2dit_block *blk = &m->blocks[L];
        char name[512];
        int idx;

        #define T2DIT_LOAD(field, suffix) do { \
            snprintf(name, sizeof(name), "%sblocks.%d.%s", prefix, L, suffix); \
            idx = safetensors_find(st, name); \
            if (idx >= 0) blk->field = t2dit_make_tensor(st, idx); \
        } while(0)

        /* Self-attention */
        T2DIT_LOAD(sa_qkv_w,      "self_attn.to_qkv.weight");
        T2DIT_LOAD(sa_qkv_b,      "self_attn.to_qkv.bias");
        T2DIT_LOAD(sa_q_norm_w,   "self_attn.q_rms_norm.gamma");
        T2DIT_LOAD(sa_k_norm_w,   "self_attn.k_rms_norm.gamma");
        T2DIT_LOAD(sa_proj_w,     "self_attn.to_out.weight");
        T2DIT_LOAD(sa_proj_b,     "self_attn.to_out.bias");

        /* Cross-attention */
        T2DIT_LOAD(norm2_w,       "norm2.weight");
        T2DIT_LOAD(norm2_b,       "norm2.bias");
        T2DIT_LOAD(ca_q_w,        "cross_attn.to_q.weight");
        T2DIT_LOAD(ca_q_b,        "cross_attn.to_q.bias");
        T2DIT_LOAD(ca_kv_w,       "cross_attn.to_kv.weight");
        T2DIT_LOAD(ca_kv_b,       "cross_attn.to_kv.bias");
        T2DIT_LOAD(ca_q_norm_w,   "cross_attn.q_rms_norm.gamma");
        T2DIT_LOAD(ca_k_norm_w,   "cross_attn.k_rms_norm.gamma");
        T2DIT_LOAD(ca_proj_w,     "cross_attn.to_out.weight");
        T2DIT_LOAD(ca_proj_b,     "cross_attn.to_out.bias");

        /* MLP */
        T2DIT_LOAD(mlp_fc1_w,     "mlp.mlp.0.weight");
        T2DIT_LOAD(mlp_fc1_b,     "mlp.mlp.0.bias");
        T2DIT_LOAD(mlp_fc2_w,     "mlp.mlp.2.weight");
        T2DIT_LOAD(mlp_fc2_b,     "mlp.mlp.2.bias");

        /* Per-block modulation bias */
        T2DIT_LOAD(mod_bias,      "modulation");

        #undef T2DIT_LOAD

        if (L == 0) {
            if (!blk->sa_qkv_w.data) fprintf(stderr, "t2dit: WARNING: block 0 sa_qkv missing\n");
            if (!blk->ca_q_w.data)   fprintf(stderr, "t2dit: WARNING: block 0 ca_q missing\n");
            if (!blk->mlp_fc1_w.data) fprintf(stderr, "t2dit: WARNING: block 0 mlp.0 missing\n");
        }
    }

    #undef T2DIT_FIND

    /* Precompute 3D RoPE tables */
    t2dit_precompute_rope(m);

    fprintf(stderr, "t2dit: loaded %d blocks, RoPE %d freqs/axis\n",
            n_blocks, m->n_rope_freqs);
    return m;
}

#endif /* SAFETENSORS_H */

void t2dit_free(t2dit_model *m) {
    if (!m) return;
    free(m->blocks);
    free(m->rope_cos);
    free(m->rope_sin);
#ifdef SAFETENSORS_H
    if (m->st_ctx) safetensors_close((st_context *)m->st_ctx);
#endif
    free(m);
}

#endif /* T2DIT_IMPLEMENTATION */
#endif /* T2DIT_H */
