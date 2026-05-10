/*
 * trellis2_slat_dit.h — TRELLIS.2 SLAT (Sparse-Latent) Flow DiT.
 *
 * Sister of common/trellis2_dit.h (the SS / dense DiT). Same block
 * architecture (ModulatedTransformerCrossBlock with share_mod=true,
 * qk_rms_norm, single-stream sparse self-attn + cross-attn over a
 * dense conditioning sequence, MLP), but:
 *
 *   - tokens are sparse: variable N per call, indexed by coords [N,4]
 *     (b, z, y, x) with z,y,x in [0, resolution).
 *   - in_channels = 32 (vs SS DiT's 8).
 *   - pe_mode = "rope": per-call 3D RoPE computed from coords, NOT a
 *     precomputed dense grid. Frequency layout matches the SS DiT
 *     (axis-major: [z_freqs, y_freqs, x_freqs] complex pairs per head).
 *
 * Top-level keys (no prefix in the published safetensors):
 *   adaLN_modulation.1.{weight,bias}        [6*dim, dim] / [6*dim]
 *   input_layer.{weight,bias}               [dim, in_ch] / [dim]
 *   out_layer.{weight,bias}                 [in_ch, dim] / [in_ch]
 *   t_embedder.mlp.{0,2}.{weight,bias}      [dim,256]/[dim], [dim,dim]/[dim]
 *   blocks.{i}.modulation                   [6*dim] additive bias
 *   blocks.{i}.self_attn.to_qkv.{weight,bias}
 *   blocks.{i}.self_attn.{q,k}_rms_norm.gamma
 *   blocks.{i}.self_attn.to_out.{weight,bias}
 *   blocks.{i}.norm2.{weight,bias}
 *   blocks.{i}.cross_attn.{to_q,to_kv,to_out}.{weight,bias}
 *   blocks.{i}.cross_attn.{q,k}_rms_norm.gamma
 *   blocks.{i}.mlp.mlp.{0,2}.{weight,bias}
 *
 * Usage:
 *   #define T2SLATDIT_IMPLEMENTATION
 *   #include "trellis2_slat_dit.h"
 *
 * Single-step forward:
 *   t2slatdit_forward(out, x_t, coords, N, t, cond, n_cond, m, n_threads);
 *
 * Dependencies: ggml_dequant.h, safetensors.h, cpu_compute.h.
 */
#ifndef T2SLATDIT_H
#define T2SLATDIT_H

#include <stdint.h>
#include <stddef.h>
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "qtensor_utils.h"

typedef struct {
    /* Self-attention */
    qtensor sa_qkv_w, sa_qkv_b;
    qtensor sa_q_norm_w, sa_k_norm_w;
    qtensor sa_proj_w, sa_proj_b;
    /* Cross-attention */
    qtensor norm2_w, norm2_b;
    qtensor ca_q_w, ca_q_b;
    qtensor ca_kv_w, ca_kv_b;
    qtensor ca_q_norm_w, ca_k_norm_w;
    qtensor ca_proj_w, ca_proj_b;
    /* MLP */
    qtensor mlp_fc1_w, mlp_fc1_b;
    qtensor mlp_fc2_w, mlp_fc2_b;
    /* Per-block modulation bias [6*dim] */
    qtensor mod_bias;
} t2slatdit_block;

typedef struct {
    int model_channels;   /* 1536 */
    int n_heads;          /* 12 */
    int head_dim;         /* 128 */
    int ffn_hidden;       /* 8192 (≈ mlp_ratio 5.3334 * dim) */
    int n_blocks;         /* 30 */
    int in_channels;      /* 32 */
    int cond_dim;         /* 1024 */
    int resolution;       /* 32 */
    float ln_eps;         /* 1e-6 */
    float rope_theta;     /* 10000.0 */
    int n_rope_freqs;     /* head_dim / 6 */

    /* Top-level */
    qtensor t_embed_fc1_w, t_embed_fc1_b;
    qtensor t_embed_fc2_w, t_embed_fc2_b;
    qtensor mod_w, mod_b;
    qtensor x_embed_w, x_embed_b;
    qtensor out_w, out_b;

    t2slatdit_block *blocks;

    void *st_ctx;
} t2slatdit_model;

t2slatdit_model *t2slatdit_load_safetensors(const char *path);
void             t2slatdit_free(t2slatdit_model *m);

/* Single denoising step.
 *   out    : [N, in_channels] f32 (host, pre-allocated)
 *   x_t    : [N, in_channels] f32 (host)
 *   coords : [N, 4] i32  (b, z, y, x), z/y/x in [0, resolution)
 *   t      : scalar timestep (caller-space; this fn scales by 1000 for
 *            the sinusoidal embedder, matching the SS DiT)
 *   cond   : [n_cond, cond_dim] f32 (host) — drop the leading batch=1
 *            dim before passing
 */
void t2slatdit_forward(float *out,
                       const float *x_t,
                       const int32_t *coords, int N,
                       float t,
                       const float *cond, int n_cond,
                       t2slatdit_model *m, int n_threads);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef T2SLATDIT_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <float.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#ifndef CPU_COMPUTE_H
#define CPU_COMPUTE_IMPLEMENTATION
#endif
#include "cpu_compute.h"

/* ---- Tensor helpers ---- */

static int t2slatdit_numel(const qtensor *t) {
    if (!t->data) return 0;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    return n;
}

static float *t2slatdit_dequant_full(const qtensor *t) {
    if (!t->data) return NULL;
    int n = t2slatdit_numel(t);
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
} t2slatdit_gemm_task;

#if defined(__AVX2__) && defined(__FMA__)
static void *t2slatdit_gemm_f32_worker(void *arg) {
    t2slatdit_gemm_task *t = (t2slatdit_gemm_task *)arg;
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
static void *t2slatdit_gemm_f32_worker(void *arg) {
    t2slatdit_gemm_task *t = (t2slatdit_gemm_task *)arg;
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

static void t2slatdit_gemm_f32(float *dst, const float *W, const float *bias,
                               const float *src, int n_tok, int n_out, int n_in,
                               int n_threads) {
    if (n_threads <= 1 || n_out < n_threads * 2) {
        t2slatdit_gemm_task task = {dst, W, src, n_out, n_in, n_tok, 0, n_out};
        t2slatdit_gemm_f32_worker(&task);
    } else {
        int nt = n_threads;
        t2slatdit_gemm_task *tasks = (t2slatdit_gemm_task *)calloc((size_t)nt, sizeof(t2slatdit_gemm_task));
        pthread_t *threads = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
        int rows_per = n_out / nt;
        int extra = n_out % nt;
        int r = 0, actual = 0;
        for (int i = 0; i < nt && r < n_out; i++) {
            int count = rows_per + (i < extra ? 1 : 0);
            tasks[i] = (t2slatdit_gemm_task){dst, W, src, n_out, n_in, n_tok, r, r + count};
            pthread_create(&threads[i], NULL, t2slatdit_gemm_f32_worker, &tasks[i]);
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

static void t2slatdit_batch_gemm(float *dst, const qtensor *W, const qtensor *bias,
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
        else
            bias_f = t2slatdit_dequant_full(bias);
    }
    if (W->type == GGML_TYPE_F32) {
        t2slatdit_gemm_f32(dst, (const float *)W->data, bias_f, src,
                           n_tok, n_out, n_in, n_threads);
    } else if (W->type == GGML_TYPE_F16) {
        cpu_gemm_f16(dst, (const uint16_t *)W->data, bias_f, src,
                     n_tok, n_out, n_in, n_threads);
    } else {
        float *Wf = qt_dequant(W);
        t2slatdit_gemm_f32(dst, Wf, bias_f, src, n_tok, n_out, n_in, n_threads);
        free(Wf);
    }
    if (bias_f && bias && bias->type != GGML_TYPE_F32)
        free(bias_f);
}

/* ---- Norms ---- */

static void t2slatdit_layernorm_noaffine(float *dst, const float *src,
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

static void t2slatdit_layernorm_affine(float *dst, const float *src,
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

static void t2slatdit_adaln(float *dst, const float *src,
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

static void t2slatdit_rmsnorm(float *vec, int n_tok, int n_heads, int head_dim,
                              int stride, const float *w, float eps) {
    if (!w) return;
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = vec + (size_t)t * stride + h * head_dim;
            const float *wh = w + h * head_dim;
            float ss = 0.0f;
            for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
            float rms = 1.0f / sqrtf(ss / head_dim + eps);
            for (int i = 0; i < head_dim; i++) v[i] *= rms * wh[i];
        }
    }
}

/* ---- Per-call 3D RoPE from sparse coords [N,4] (b,z,y,x).
 * Layout matches upstream SparseRotaryPositionEmbedder:
 *   freq_dim = head_dim / 2 / 3              (= 21 for hd=128)
 *   per-token phases (cos,sin) shape [3*freq_dim]; if 3*freq_dim < hd/2,
 *   pad the trailing pairs with cos=1, sin=0 (identity rotation).
 * Output: rope_cos / rope_sin of shape [N, 3*freq_dim], laid out
 *   axis-major so the apply step matches t2dit_apply_rope_qkv. */
static void t2slatdit_compute_rope(const int32_t *coords, int N,
                                   int n_freqs, float theta,
                                   float **rope_cos_out, float **rope_sin_out) {
    float *freqs = (float *)malloc((size_t)n_freqs * sizeof(float));
    for (int j = 0; j < n_freqs; j++)
        freqs[j] = 1.0f / powf(theta, (float)j / (float)n_freqs);

    float *cs = (float *)malloc((size_t)N * 3 * n_freqs * sizeof(float));
    float *sn = (float *)malloc((size_t)N * 3 * n_freqs * sizeof(float));

    for (int i = 0; i < N; i++) {
        /* coords[i] = (b, z, y, x); use (z, y, x) — axis 0 = z, 1 = y, 2 = x */
        float c[3] = {(float)coords[i*4 + 1],
                      (float)coords[i*4 + 2],
                      (float)coords[i*4 + 3]};
        for (int axis = 0; axis < 3; axis++) {
            for (int j = 0; j < n_freqs; j++) {
                float arg = c[axis] * freqs[j];
                cs[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = cosf(arg);
                sn[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = sinf(arg);
            }
        }
    }
    free(freqs);
    *rope_cos_out = cs;
    *rope_sin_out = sn;
}

/* Apply 3D RoPE to Q and K within the QKV [N, 3*dim] buffer.
 * QKV per-token order: [Q(dim), K(dim), V(dim)]; each [n_heads, head_dim].
 * Pair convention: (v[2k], v[2k+1]) = re,im. Pairs 0..3*n_freqs-1 receive
 * the per-axis rotation; remaining pairs (head_dim/2 - 3*n_freqs) are
 * identity. */
static void t2slatdit_apply_rope_qkv(float *qkv, int N, int dim,
                                     int n_heads, int head_dim,
                                     const float *rope_cos, const float *rope_sin,
                                     int n_freqs) {
    int rot_pairs = 3 * n_freqs;
    for (int t = 0; t < N; t++) {
        const float *cs = rope_cos + (size_t)t * rot_pairs;
        const float *sn = rope_sin + (size_t)t * rot_pairs;
        for (int qk = 0; qk < 2; qk++) {
            for (int h = 0; h < n_heads; h++) {
                float *v = qkv + (size_t)t * 3 * dim + qk * dim + h * head_dim;
                for (int p = 0; p < rot_pairs; p++) {
                    int idx = p * 2;
                    float re = v[idx], im = v[idx + 1];
                    float c = cs[p], s = sn[p];
                    v[idx]     = re * c - im * s;
                    v[idx + 1] = re * s + im * c;
                }
                /* pairs >= rot_pairs are identity (cos=1, sin=0): leave alone */
            }
        }
    }
}

/* ---- Timestep embedding (identical to SS DiT) ---- */

static void t2slatdit_timestep_embed(float *out, float t_val,
                                     const t2slatdit_model *m) {
    int half = 128;
    float embed[256];
    for (int j = 0; j < half; j++) {
        float freq = expf(-(float)j / (float)half * logf(10000.0f));
        float arg = t_val * freq;
        embed[j] = cosf(arg);
        embed[half + j] = sinf(arg);
    }
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
    for (int i = 0; i < dim; i++)
        h1[i] = h1[i] / (1.0f + expf(-h1[i]));
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

static void t2slatdit_modulation(float *mod_out, const float *t_emb,
                                 const t2slatdit_model *m) {
    int dim = m->model_channels;
    int out_dim = 6 * dim;
    float *h = (float *)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < dim; i++)
        h[i] = t_emb[i] / (1.0f + expf(-t_emb[i]));
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

/* ---- Self-attention via cpu_cross_attention ---- */

static void t2slatdit_self_attention(float *out, const float *qkv,
                                     int n_tok, int dim, int n_heads,
                                     int head_dim, int n_threads) {
    float *Q_buf  = (float *)malloc((size_t)n_tok * dim * sizeof(float));
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

/* ---- Debug helpers ---- */
static int t2slat_dump_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("T2SLAT_DUMP"); v = (e && atoi(e)) ? 1 : 0; }
    return v;
}
static void t2slat_dump_npy(const char *name, const float *data, int n0, int n1) {
    if (!t2slat_dump_enabled()) return;
    char path[512];
    snprintf(path, sizeof(path), "/tmp/cpu_slat_%s.npy", name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    /* NPY 1.0 header (manual, f32 little-endian, C order) */
    char hdr[256];
    int n = snprintf(hdr, sizeof(hdr),
                     "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", n0, n1);
    int hlen = (n + 11 + 63) & ~63;
    while (n < hlen - 11) hdr[n++] = ' ';
    hdr[hlen - 11] = '\n';
    unsigned char magic[10] = {0x93,'N','U','M','P','Y',1,0, (unsigned char)(hlen-10), 0};
    fwrite(magic, 1, 10, f);
    fwrite(hdr, 1, hlen-10, f);
    fwrite(data, sizeof(float), (size_t)n0 * n1, f);
    fclose(f);
    fprintf(stderr, "[t2slat-dump] %s [%d,%d]\n", name, n0, n1);
}

/* ---- One DiT block forward ---- */

static void t2slatdit_block_forward(float *hidden, const t2slatdit_block *blk,
                                    const float *global_mod,
                                    const float *cond, int n_cond,
                                    int N,
                                    const float *rope_cos, const float *rope_sin,
                                    int block_idx,
                                    const t2slatdit_model *m, int n_threads) {
    int dim = m->model_channels;
    int hd = m->head_dim;
    int nh = m->n_heads;

    float *mod = (float *)malloc((size_t)6 * dim * sizeof(float));
    memcpy(mod, global_mod, (size_t)6 * dim * sizeof(float));
    if (blk->mod_bias.data) {
        float *mb = t2slatdit_dequant_full(&blk->mod_bias);
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

    float *tmp = (float *)malloc((size_t)N * dim * sizeof(float));
    float *qkv = (float *)malloc((size_t)N * 3 * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)N * dim * sizeof(float));

    /* Self-attention with adaLN */
    /* Note: upstream norm1 is LN-no-affine (the adaLN scale/shift come AFTER the LN).
     * t2slatdit_adaln fuses LN+adaLN. To compare to blk0_norm1 we need just LN(no-affine). */
    if (block_idx == 0 && t2slat_dump_enabled()) {
        float *ln_only = (float *)malloc((size_t)N * dim * sizeof(float));
        t2slatdit_layernorm_noaffine(ln_only, hidden, N, dim, m->ln_eps);
        t2slat_dump_npy("blk0_norm1", ln_only, N, dim);
        free(ln_only);
    }
    t2slatdit_adaln(tmp, hidden, shift_sa, scale_sa, N, dim, m->ln_eps);

    t2slatdit_batch_gemm(qkv, &blk->sa_qkv_w, &blk->sa_qkv_b,
                         tmp, N, 3 * dim, dim, n_threads);

    if (blk->sa_q_norm_w.data) {
        float *qn = t2slatdit_dequant_full(&blk->sa_q_norm_w);
        float *kn = t2slatdit_dequant_full(&blk->sa_k_norm_w);
        t2slatdit_rmsnorm(qkv,       N, nh, hd, 3 * dim, qn, m->ln_eps);
        t2slatdit_rmsnorm(qkv + dim, N, nh, hd, 3 * dim, kn, m->ln_eps);
        free(qn); free(kn);
    }

    t2slatdit_apply_rope_qkv(qkv, N, dim, nh, hd,
                             rope_cos, rope_sin, m->n_rope_freqs);

    t2slatdit_self_attention(attn_out, qkv, N, dim, nh, hd, n_threads);
    free(qkv);

    float *proj_out = (float *)malloc((size_t)N * dim * sizeof(float));
    t2slatdit_batch_gemm(proj_out, &blk->sa_proj_w, &blk->sa_proj_b,
                         attn_out, N, dim, dim, n_threads);
    if (block_idx == 0 && t2slat_dump_enabled())
        t2slat_dump_npy("blk0_self_attn", proj_out, N, dim);

    for (int t = 0; t < N; t++)
        for (int i = 0; i < dim; i++)
            hidden[(size_t)t * dim + i] += gate_sa[i] * proj_out[(size_t)t * dim + i];
    free(proj_out);

    /* Cross-attention */
    if (blk->ca_q_w.data && cond) {
        float *ln2_w = t2slatdit_dequant_full(&blk->norm2_w);
        float *ln2_b = t2slatdit_dequant_full(&blk->norm2_b);
        t2slatdit_layernorm_affine(tmp, hidden, ln2_w, ln2_b, N, dim, m->ln_eps);
        free(ln2_w); free(ln2_b);
        if (block_idx == 0 && t2slat_dump_enabled())
            t2slat_dump_npy("blk0_norm2", tmp, N, dim);

        float *cross_q = (float *)malloc((size_t)N * dim * sizeof(float));
        t2slatdit_batch_gemm(cross_q, &blk->ca_q_w, &blk->ca_q_b,
                             tmp, N, dim, dim, n_threads);

        if (blk->ca_q_norm_w.data) {
            float *qn = t2slatdit_dequant_full(&blk->ca_q_norm_w);
            t2slatdit_rmsnorm(cross_q, N, nh, hd, dim, qn, m->ln_eps);
            free(qn);
        }

        /* Compute KV from cond [n_cond, cond_dim] -> [n_cond, 2*dim] */
        float *kv = (float *)malloc((size_t)n_cond * 2 * dim * sizeof(float));
        t2slatdit_batch_gemm(kv, &blk->ca_kv_w, &blk->ca_kv_b,
                             cond, n_cond, 2 * dim, m->cond_dim, n_threads);
        if (blk->ca_k_norm_w.data) {
            float *kn = t2slatdit_dequant_full(&blk->ca_k_norm_w);
            t2slatdit_rmsnorm(kv, n_cond, nh, hd, 2 * dim, kn, m->ln_eps);
            free(kn);
        }

        cpu_cross_attention(attn_out, cross_q, kv,
                            N, n_cond, dim, nh, hd, n_threads);
        free(cross_q);
        free(kv);

        float *ca_proj = (float *)malloc((size_t)N * dim * sizeof(float));
        t2slatdit_batch_gemm(ca_proj, &blk->ca_proj_w, &blk->ca_proj_b,
                             attn_out, N, dim, dim, n_threads);
        if (block_idx == 0 && t2slat_dump_enabled())
            t2slat_dump_npy("blk0_cross_attn", ca_proj, N, dim);
        for (int i = 0; i < N * dim; i++)
            hidden[i] += ca_proj[i];
        free(ca_proj);
    }

    /* MLP with adaLN */
    t2slatdit_adaln(tmp, hidden, shift_mlp, scale_mlp, N, dim, m->ln_eps);

    int ffn = m->ffn_hidden;
    float *ffn_buf = (float *)malloc((size_t)N * ffn * sizeof(float));
    t2slatdit_batch_gemm(ffn_buf, &blk->mlp_fc1_w, &blk->mlp_fc1_b,
                         tmp, N, ffn, dim, n_threads);
    cpu_gelu(ffn_buf, N * ffn);

    float *mlp_out = (float *)malloc((size_t)N * dim * sizeof(float));
    t2slatdit_batch_gemm(mlp_out, &blk->mlp_fc2_w, &blk->mlp_fc2_b,
                         ffn_buf, N, dim, ffn, n_threads);
    free(ffn_buf);

    for (int t = 0; t < N; t++)
        for (int i = 0; i < dim; i++)
            hidden[(size_t)t * dim + i] += gate_mlp[i] * mlp_out[(size_t)t * dim + i];
    free(mlp_out);

    free(mod);
    free(tmp);
    free(attn_out);
}

/* ---- Forward ---- */

void t2slatdit_forward(float *out,
                       const float *x_t,
                       const int32_t *coords, int N,
                       float t,
                       const float *cond, int n_cond,
                       t2slatdit_model *m, int n_threads) {
    int dim = m->model_channels;

    /* Per-call RoPE tables */
    float *rope_cos = NULL, *rope_sin = NULL;
    t2slatdit_compute_rope(coords, N, m->n_rope_freqs, m->rope_theta,
                           &rope_cos, &rope_sin);

    /* Timestep embedding. The model's forward() takes raw t (the sampler is
     * responsible for the ×1000 scaling — see flow_euler.py _inference_model).
     * Caller passes t already in the convention the model expects. */
    float *t_emb = (float *)malloc((size_t)dim * sizeof(float));
    t2slatdit_timestep_embed(t_emb, t, m);

    if (t2slat_dump_enabled())
        t2slat_dump_npy("t_emb", t_emb, 1, dim);

    /* Shared modulation -> 6 chunks of [dim] */
    float *mod = (float *)malloc((size_t)6 * dim * sizeof(float));
    t2slatdit_modulation(mod, t_emb, m);
    free(t_emb);

    /* Input embedding: x_t [N, in_channels] -> [N, dim] */
    float *hidden = (float *)malloc((size_t)N * dim * sizeof(float));
    t2slatdit_batch_gemm(hidden, &m->x_embed_w, &m->x_embed_b,
                         x_t, N, dim, m->in_channels, n_threads);
    if (t2slat_dump_enabled()) {
        t2slat_dump_npy("input_layer_out", hidden, N, dim);
        t2slat_dump_npy("mod", mod, 1, 6 * dim);
    }

    for (int L = 0; L < m->n_blocks; L++) {
        t2slatdit_block_forward(hidden, &m->blocks[L], mod,
                                cond, n_cond, N,
                                rope_cos, rope_sin, L, m, n_threads);
        if (t2slat_dump_enabled() && L < 3) {
            char nm[64]; snprintf(nm, sizeof(nm), "block_%d_output", L);
            t2slat_dump_npy(nm, hidden, N, dim);
        }
    }

    t2slatdit_layernorm_noaffine(hidden, hidden, N, dim, m->ln_eps);

    t2slatdit_batch_gemm(out, &m->out_w, &m->out_b,
                         hidden, N, m->in_channels, dim, n_threads);

    free(hidden);
    free(mod);
    free(rope_cos);
    free(rope_sin);
}

/* ==================================================================== */
/* SafeTensors loader                                                    */
/* ==================================================================== */

#ifdef SAFETENSORS_H

t2slatdit_model *t2slatdit_load_safetensors(const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) return NULL;

    fprintf(stderr, "t2slatdit: opened safetensors, %d tensors\n", st->n_tensors);

    /* Detect prefix using same anchors as SS DiT */
    char prefix[256] = "";
    const char *anchors[] = {"t_embedder", "input_layer", "adaLN_modulation", NULL};
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
    fprintf(stderr, "t2slatdit: detected prefix: '%s'\n", prefix);

    #define T2SLAT_FIND(suffix) ({ \
        char _buf[512]; \
        snprintf(_buf, sizeof(_buf), "%s%s", prefix, suffix); \
        int _idx = safetensors_find(st, _buf); \
        (_idx >= 0) ? qt_make_tensor(st, _idx) : (qtensor){0}; \
    })

    /* Auto-detect architecture */
    int model_channels = 1536, n_heads = 12, head_dim = 128;
    int ffn_hidden = 8192, n_blocks = 0, in_channels = 32;
    int cond_dim = 1024;

    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const uint64_t *sh = safetensors_shape(st, i);
        int nd = safetensors_ndims(st, i);

        if (strstr(nm, "input_layer.weight") && nd == 2) {
            model_channels = (int)sh[0];
            in_channels = (int)sh[1];
        }
        if (strstr(nm, "cross_attn.to_kv.weight") && nd == 2) {
            cond_dim = (int)sh[1];
        }
        if (strstr(nm, "self_attn.q_rms_norm.gamma") && nd == 1) {
            head_dim = (int)sh[0];
        }
        if (strstr(nm, "blocks.0.mlp.mlp.0.weight") && nd == 2) {
            ffn_hidden = (int)sh[0];
        }
        const char *bp = strstr(nm, "blocks.");
        if (bp) {
            bp += 7;
            int blk = 0;
            while (*bp >= '0' && *bp <= '9') { blk = blk * 10 + (*bp - '0'); bp++; }
            if (blk + 1 > n_blocks) n_blocks = blk + 1;
        }
    }

    n_heads = model_channels / head_dim;
    int resolution = 32;

    fprintf(stderr, "t2slatdit: dim=%d heads=%d head_dim=%d blocks=%d ffn=%d "
                    "in_ch=%d cond=%d\n",
            model_channels, n_heads, head_dim, n_blocks, ffn_hidden,
            in_channels, cond_dim);

    t2slatdit_model *m = (t2slatdit_model *)calloc(1, sizeof(t2slatdit_model));
    m->model_channels = model_channels;
    m->n_heads = n_heads;
    m->head_dim = head_dim;
    m->ffn_hidden = ffn_hidden;
    m->n_blocks = n_blocks;
    m->in_channels = in_channels;
    m->cond_dim = cond_dim;
    m->resolution = resolution;
    m->ln_eps = 1e-6f;
    m->rope_theta = 10000.0f;
    m->n_rope_freqs = head_dim / 6;
    m->st_ctx = st;

    m->t_embed_fc1_w = T2SLAT_FIND("t_embedder.mlp.0.weight");
    m->t_embed_fc1_b = T2SLAT_FIND("t_embedder.mlp.0.bias");
    m->t_embed_fc2_w = T2SLAT_FIND("t_embedder.mlp.2.weight");
    m->t_embed_fc2_b = T2SLAT_FIND("t_embedder.mlp.2.bias");
    m->mod_w = T2SLAT_FIND("adaLN_modulation.1.weight");
    m->mod_b = T2SLAT_FIND("adaLN_modulation.1.bias");
    m->x_embed_w = T2SLAT_FIND("input_layer.weight");
    m->x_embed_b = T2SLAT_FIND("input_layer.bias");
    m->out_w = T2SLAT_FIND("out_layer.weight");
    m->out_b = T2SLAT_FIND("out_layer.bias");

    fprintf(stderr, "t2slatdit: t_embedder: %s, mod: %s, input_layer: %s, out_layer: %s\n",
            m->t_embed_fc1_w.data ? "ok" : "MISS",
            m->mod_w.data ? "ok" : "MISS",
            m->x_embed_w.data ? "ok" : "MISS",
            m->out_w.data ? "ok" : "MISS");

    m->blocks = (t2slatdit_block *)calloc((size_t)n_blocks, sizeof(t2slatdit_block));
    for (int L = 0; L < n_blocks; L++) {
        t2slatdit_block *blk = &m->blocks[L];
        char name[512];
        int idx;

        #define T2SLAT_LOAD(field, suffix) do { \
            snprintf(name, sizeof(name), "%sblocks.%d.%s", prefix, L, suffix); \
            idx = safetensors_find(st, name); \
            if (idx >= 0) blk->field = qt_make_tensor(st, idx); \
        } while(0)

        T2SLAT_LOAD(sa_qkv_w,     "self_attn.to_qkv.weight");
        T2SLAT_LOAD(sa_qkv_b,     "self_attn.to_qkv.bias");
        T2SLAT_LOAD(sa_q_norm_w,  "self_attn.q_rms_norm.gamma");
        T2SLAT_LOAD(sa_k_norm_w,  "self_attn.k_rms_norm.gamma");
        T2SLAT_LOAD(sa_proj_w,    "self_attn.to_out.weight");
        T2SLAT_LOAD(sa_proj_b,    "self_attn.to_out.bias");

        T2SLAT_LOAD(norm2_w,      "norm2.weight");
        T2SLAT_LOAD(norm2_b,      "norm2.bias");
        T2SLAT_LOAD(ca_q_w,       "cross_attn.to_q.weight");
        T2SLAT_LOAD(ca_q_b,       "cross_attn.to_q.bias");
        T2SLAT_LOAD(ca_kv_w,      "cross_attn.to_kv.weight");
        T2SLAT_LOAD(ca_kv_b,      "cross_attn.to_kv.bias");
        T2SLAT_LOAD(ca_q_norm_w,  "cross_attn.q_rms_norm.gamma");
        T2SLAT_LOAD(ca_k_norm_w,  "cross_attn.k_rms_norm.gamma");
        T2SLAT_LOAD(ca_proj_w,    "cross_attn.to_out.weight");
        T2SLAT_LOAD(ca_proj_b,    "cross_attn.to_out.bias");

        T2SLAT_LOAD(mlp_fc1_w,    "mlp.mlp.0.weight");
        T2SLAT_LOAD(mlp_fc1_b,    "mlp.mlp.0.bias");
        T2SLAT_LOAD(mlp_fc2_w,    "mlp.mlp.2.weight");
        T2SLAT_LOAD(mlp_fc2_b,    "mlp.mlp.2.bias");

        T2SLAT_LOAD(mod_bias,     "modulation");

        #undef T2SLAT_LOAD
    }
    #undef T2SLAT_FIND

    fprintf(stderr, "t2slatdit: loaded %d blocks (RoPE %d freqs/axis)\n",
            n_blocks, m->n_rope_freqs);
    return m;
}

#endif /* SAFETENSORS_H */

void t2slatdit_free(t2slatdit_model *m) {
    if (!m) return;
    free(m->blocks);
#ifdef SAFETENSORS_H
    if (m->st_ctx) safetensors_close((st_context *)m->st_ctx);
#endif
    free(m);
}

#endif /* T2SLATDIT_IMPLEMENTATION */
#endif /* T2SLATDIT_H */
