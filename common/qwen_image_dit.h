/*
 * qwen_image_dit.h - Qwen-Image MMDiT (Dual-Stream Diffusion Transformer)
 *
 * Usage:
 *   #define QIMG_DIT_IMPLEMENTATION
 *   #include "qwen_image_dit.h"
 *
 * Dependencies: gguf_loader.h, ggml_dequant.h, cpu_compute.h
 *
 * Architecture: Flux-style dual-stream DiT
 *   hidden=3072, heads=24, head_dim=128, FFN=12288, 60 blocks
 *   Dual-stream joint attention (image + text concatenated)
 *   adaLN-Zero modulation, per-head QK RMSNorm, GELU FFN
 *
 * API:
 *   qimg_dit_model *qimg_dit_load_gguf(const char *path);
 *   void            qimg_dit_free(qimg_dit_model *m);
 *   void            qimg_dit_forward(float *out, const float *img_tokens,
 *                                    int n_img, const float *txt_tokens,
 *                                    int n_txt, float timestep,
 *                                    qimg_dit_model *m, int n_threads);
 *   void            qimg_dit_patchify(float *out, const float *latent,
 *                                     int ch, int h, int w, int ps);
 *   void            qimg_dit_unpatchify(float *out, const float *tokens,
 *                                       int n_tok, int ch, int h, int w, int ps);
 */
#ifndef QIMG_DIT_H
#define QIMG_DIT_H

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
#ifndef T2DIT_H
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
#endif

typedef struct {
    /* Image attention (separate Q/K/V) */
    qtensor attn_q_w, attn_q_b;
    qtensor attn_k_w, attn_k_b;
    qtensor attn_v_w, attn_v_b;
    qtensor attn_out_w, attn_out_b;

    /* Text attention (separate Q/K/V) */
    qtensor attn_add_q_w, attn_add_q_b;
    qtensor attn_add_k_w, attn_add_k_b;
    qtensor attn_add_v_w, attn_add_v_b;
    qtensor attn_add_out_w, attn_add_out_b;

    /* Per-head QK RMSNorm */
    qtensor norm_q_w;           /* [head_dim] */
    qtensor norm_k_w;
    qtensor norm_added_q_w;
    qtensor norm_added_k_w;

    /* Image modulation: SiLU -> Linear(dim, 6*dim) */
    qtensor img_mod_w, img_mod_b;    /* [dim, 6*dim] */

    /* Image MLP: Linear->GELU->Linear */
    qtensor img_mlp_fc1_w, img_mlp_fc1_b;  /* [dim, mlp_hidden] */
    qtensor img_mlp_fc2_w, img_mlp_fc2_b;  /* [mlp_hidden, dim] */

    /* Text modulation */
    qtensor txt_mod_w, txt_mod_b;

    /* Text MLP: Linear->GELU->Linear */
    qtensor txt_mlp_fc1_w, txt_mlp_fc1_b;
    qtensor txt_mlp_fc2_w, txt_mlp_fc2_b;
} qimg_dit_block;

typedef struct {
    int hidden_dim;    /* 3072 */
    int n_heads;       /* 24 */
    int head_dim;      /* 128 */
    int n_blocks;      /* 60 */
    int in_channels;   /* 64 (16 latent * patch_size^2) */
    int out_channels;  /* 64 */
    int patch_size;    /* 2 */
    int txt_dim;       /* 3584 (joint_attention_dim) */
    int mlp_hidden;    /* 12288 */
    float ln_eps;      /* 1e-6 */

    /* Text norm (RMSNorm on encoder output before projection) */
    qtensor txt_norm_w;    /* [txt_dim] */

    /* Input projections */
    qtensor img_in_w, img_in_b;    /* [in_channels, hidden_dim] */
    qtensor txt_in_w, txt_in_b;    /* [txt_dim, hidden_dim] */

    /* Timestep embedder: sinusoidal(256) -> Linear -> SiLU -> Linear */
    qtensor t_fc1_w, t_fc1_b;     /* [256, hidden_dim] */
    qtensor t_fc2_w, t_fc2_b;     /* [hidden_dim, hidden_dim] */

    /* Final output */
    qtensor norm_out_w, norm_out_b;  /* [hidden_dim, 2*hidden_dim] shift+scale */
    qtensor proj_out_w, proj_out_b;  /* [hidden_dim, out_channels] */

    qimg_dit_block *blocks;

    /* RoPE config: axes_dims = [16, 56, 56], total = 128 = head_dim */
    int rope_axes[3];  /* temporal, height, width dims for RoPE */
    float rope_theta;  /* 10000.0 */

    void *gguf_ctx;        /* kept for mmap lifetime (gguf_context* or st_context*) */
    int   gguf_ctx_is_st;  /* 1 if gguf_ctx is st_context*, 0 if gguf_context* */

    /* Debug: dump intermediates for a specific block (-1 = disabled) */
    int dump_block;
    void (*dump_fn)(const char *name, const float *data, int n, void *ctx);
    void *dump_ctx;

    /* Pre-dequantized F32 weight storage (from qimg_dit_predequant) */
    float **predequant_bufs;  /* array of malloc'd F32 buffers */
    int     n_predequant;     /* count of buffers */
    int     is_predequanted;  /* 1 if predequant has been applied */
} qimg_dit_model;

/* Pre-dequantize all quantized weights to F32 for fast SIMD GEMM.
 * Call once after loading. Increases memory (~2GB for 60 blocks) but
 * eliminates repeated dequantization during inference. */
void qimg_dit_predequant(qimg_dit_model *m);

qimg_dit_model *qimg_dit_load_gguf(const char *path);
qimg_dit_model *qimg_dit_load_safetensors(const char *path);
void            qimg_dit_free(qimg_dit_model *m);

/* Forward: single denoising step.
 * img_tokens: [n_img, in_channels] patchified latent
 * txt_tokens: [n_txt, txt_dim] text encoder hidden states
 * out: [n_img, out_channels] velocity prediction */
void qimg_dit_forward(float *out, const float *img_tokens, int n_img,
                      const float *txt_tokens, int n_txt, float timestep,
                      qimg_dit_model *m, int n_threads);

/* Patchify: latent [ch, h, w] -> tokens [h/ps * w/ps, ch*ps*ps] */
void qimg_dit_patchify(float *out, const float *latent,
                       int ch, int h, int w, int ps);

/* Unpatchify: tokens [n_tok, ch*ps*ps] -> latent [ch, h, w] */
void qimg_dit_unpatchify(float *out, const float *tokens,
                         int n_tok, int ch, int h, int w, int ps);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef QIMG_DIT_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_loader.h"
#include "safetensors.h"

/* Custom type ID for FP8 E4M3 (not in GGML enum) */
#define QIMG_TYPE_F8_E4M3 200

/* FP8 E4M3 → F32 conversion */
static float qimg_fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0) {
        f = ldexpf((float)mant / 8.0f, -6);
    } else if (exp == 15 && mant == 7) {
        return 0.0f;
    } else {
        f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -f : f;
}

/* Pre-computed FP8 E4M3 → F32 lookup table (256 entries) */
static float qimg_fp8_lut[256];
static int qimg_fp8_lut_init = 0;

static void qimg_init_fp8_lut(void) {
    if (qimg_fp8_lut_init) return;
    for (int i = 0; i < 256; i++)
        qimg_fp8_lut[i] = qimg_fp8_e4m3_to_f32((uint8_t)i);
    qimg_fp8_lut_init = 1;
}

/* Dequantize one row of FP8 E4M3 data to F32 using LUT */
static void qimg_dequant_row_fp8(const uint8_t *src, float *dst, int n) {
    qimg_init_fp8_lut();
    int i = 0;
#if defined(__AVX2__)
    /* Process 32 bytes at a time: load 32 uint8, gather 32 floats from LUT */
    for (; i + 31 < n; i += 32) {
        for (int j = 0; j < 32; j++)
            dst[i + j] = qimg_fp8_lut[src[i + j]];
    }
#endif
    for (; i < n; i++)
        dst[i] = qimg_fp8_lut[src[i]];
}

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define QIMG_HAS_SSE2 1
#ifdef __AVX2__
#define QIMG_HAS_AVX2 1
#endif
#ifdef __FMA__
#define QIMG_HAS_FMA 1
#endif
#endif

/* ---- Helper: dequantize a row (kept for non-predequant path) ---- */

static void qimg_dequant_row(const qtensor *t, int row, float *dst) {
    int n_cols = t->n_cols;

    /* FP8 E4M3: 1 byte per element, simple row layout */
    if (t->type == QIMG_TYPE_F8_E4M3) {
        const uint8_t *src = (const uint8_t *)t->data + (size_t)row * n_cols;
        qimg_dequant_row_fp8(src, dst, n_cols);
        return;
    }
    /* F32: direct memcpy */
    if (t->type == GGML_TYPE_F32) {
        memcpy(dst, (const float *)t->data + (size_t)row * n_cols,
               (size_t)n_cols * sizeof(float));
        return;
    }

    int block_size, type_size;
    switch (t->type) {
        case GGML_TYPE_Q4_0:    block_size = 32;  type_size = 18;  break;
        case GGML_TYPE_Q4_1:    block_size = 32;  type_size = 20;  break;
        case GGML_TYPE_Q8_0:    block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K:    block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q6_K:    block_size = 256; type_size = 210; break;
        case GGML_TYPE_IQ4_XS:  block_size = 256; type_size = 136; break;
        case GGML_TYPE_F16:     block_size = 1;   type_size = 2;   break;
        case GGML_TYPE_BF16:    block_size = 1;   type_size = 2;   break;
        case GGML_TYPE_Q4_0_4_4: block_size = 32; type_size = 18;  break;
        case GGML_TYPE_Q4_0_4_8: block_size = 32; type_size = 18;  break;
        case GGML_TYPE_Q4_0_8_8: block_size = 32; type_size = 18;  break;
        default:
            fprintf(stderr, "qimg_dequant_row: unsupported type %u\n", t->type);
            memset(dst, 0, (size_t)n_cols * sizeof(float));
            return;
    }
    size_t row_bytes = (size_t)((n_cols + block_size - 1) / block_size) * type_size;
    const void *row_data = (const uint8_t *)t->data + (size_t)row * row_bytes;
    dequant_row(t->type, row_data, dst, n_cols);
}

/* Dequantize entire tensor to F32. Caller must free(). */
static float *qimg_dequant_full(const qtensor *t) {
    float *buf = (float *)malloc((size_t)t->n_rows * t->n_cols * sizeof(float));
    for (int r = 0; r < t->n_rows; r++)
        qimg_dequant_row(t, r, buf + (size_t)r * t->n_cols);
    return buf;
}

/* ---- Pre-dequantized weight: just F32 pointer + dims ---- */

typedef struct {
    float *data;   /* [n_rows, n_cols] F32, pre-dequantized */
    int    n_rows;
    int    n_cols;
} qimg_f32w;

/* Pre-dequantize a qtensor to F32. */
static qimg_f32w qimg_predequant(const qtensor *t) {
    qimg_f32w w = {NULL, 0, 0};
    if (!t || !t->data) return w;
    w.n_rows = t->n_rows;
    w.n_cols = t->n_cols;
    w.data = qimg_dequant_full(t);
    return w;
}

static void qimg_f32w_free(qimg_f32w *w) {
    if (w->data) { free(w->data); w->data = NULL; }
}

/* ---- SIMD dot product: a[n] . b[n] ---- */

static float qimg_dot(const float *a, const float *b, int n) {
#if defined(QIMG_HAS_AVX2) && defined(QIMG_HAS_FMA)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#elif defined(QIMG_HAS_SSE2)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + i),     _mm_loadu_ps(b + i)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
    }
    acc0 = _mm_add_ps(acc0, acc1);
    /* Horizontal sum */
    __m128 shuf = _mm_shuffle_ps(acc0, acc0, _MM_SHUFFLE(2, 3, 0, 1));
    acc0 = _mm_add_ps(acc0, shuf);
    shuf = _mm_movehl_ps(shuf, acc0);
    acc0 = _mm_add_ss(acc0, shuf);
    float sum = _mm_cvtss_f32(acc0);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

/* ---- Batched GEMM with pre-dequantized F32 weights ---- */
/* out[tok, n_out] = inp[tok, n_in] @ W^T + bias
 * W is [n_out, n_in] row-major F32 (pre-dequantized)
 * bias is [n_out] F32 or NULL */

static void qimg_gemm_f32(float *out, const float *W, const float *bias,
                          const float *inp, int n_tok, int n_out, int n_in) {
    for (int o = 0; o < n_out; o++) {
        const float *w_row = W + (size_t)o * n_in;
        float b = bias ? bias[o] : 0.0f;
        for (int t = 0; t < n_tok; t++) {
            const float *x = inp + (size_t)t * n_in;
            out[(size_t)t * n_out + o] = qimg_dot(w_row, x, n_in) + b;
        }
    }
}

/* ---- Legacy GEMM with on-the-fly dequant (for non-predequant path) ---- */

static void qimg_batch_gemm(float *out, const qtensor *w, const qtensor *bias,
                            const float *inp, int n_tok, int n_out, int n_in,
                            int n_threads) {
    (void)n_threads;

    /* Get bias as F32 (zero-copy if already F32) */
    const float *b = NULL;
    float *b_tmp = NULL;
    if (bias && bias->data) {
        if (bias->type == GGML_TYPE_F32)
            b = (const float *)bias->data;
        else { b_tmp = qimg_dequant_full(bias); b = b_tmp; }
    }

    /* Fast path: weights already F32 (pre-dequanted) — zero-copy SIMD GEMM */
    if (w->type == GGML_TYPE_F32) {
        qimg_gemm_f32(out, (const float *)w->data, b, inp, n_tok, n_out, n_in);
        if (b_tmp) free(b_tmp);
        return;
    }

    /* Quantized path: dequant one row at a time, use SIMD dot product */
    float *row_buf = (float *)malloc((size_t)n_in * sizeof(float));
    for (int o = 0; o < n_out; o++) {
        qimg_dequant_row(w, o, row_buf);
        float bv = b ? b[o] : 0.0f;
        for (int t = 0; t < n_tok; t++) {
            const float *x = inp + (size_t)t * n_in;
            out[(size_t)t * n_out + o] = qimg_dot(row_buf, x, n_in) + bv;
        }
    }
    free(row_buf);
    if (b_tmp) free(b_tmp);
}

/* ---- LayerNorm (no affine) ---- */

static void qimg_layernorm_noaff(float *out, const float *x, int n_tok, int dim,
                                 float eps) {
    for (int t = 0; t < n_tok; t++) {
        const float *row = x + (size_t)t * dim;
        float *dst = out + (size_t)t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += row[i];
        mean /= (float)dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = row[i] - mean; var += d * d; }
        var /= (float)dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++) dst[i] = (row[i] - mean) * inv;
    }
}

/* ---- RMSNorm (with weight) ---- */

static void qimg_rmsnorm(float *x, int n_tok, int dim, const float *w, float eps) {
    for (int t = 0; t < n_tok; t++) {
        float *row = x + (size_t)t * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) ss += row[i] * row[i];
        float inv = 1.0f / sqrtf(ss / (float)dim + eps);
        for (int i = 0; i < dim; i++) row[i] *= inv * w[i];
    }
}

/* Per-head RMSNorm on QKV: stride allows operating on interleaved data */
static void qimg_rmsnorm_per_head(float *x, int n_tok, int n_heads, int head_dim,
                                  const float *w, float eps) {
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *hd = x + (size_t)t * n_heads * head_dim + (size_t)h * head_dim;
            float ss = 0.0f;
            for (int i = 0; i < head_dim; i++) ss += hd[i] * hd[i];
            float inv = 1.0f / sqrtf(ss / (float)head_dim + eps);
            for (int i = 0; i < head_dim; i++) hd[i] *= inv * w[i];
        }
    }
}

/* ---- adaLN: out = LN(x) * (1 + scale) + shift ---- */

static void qimg_adaln(float *out, const float *x, const float *shift,
                       const float *scale, int n_tok, int dim, float eps) {
    /* First: layernorm without affine */
    qimg_layernorm_noaff(out, x, n_tok, dim, eps);
    /* Then: modulate */
    for (int t = 0; t < n_tok; t++)
        for (int i = 0; i < dim; i++) {
            size_t idx = (size_t)t * dim + i;
            out[idx] = out[idx] * (1.0f + scale[i]) + shift[i];
        }
}

/* ---- GELU (tanh approximation) ---- */

static void qimg_gelu(float *x, int n) {
    const float c = 0.7978845608f;  /* sqrt(2/pi) */
    const float k = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(c * (v + k * v * v * v)));
    }
}

/* ---- SiLU ---- */

static void qimg_silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* ---- Self-attention with online softmax ---- */

static void qimg_attention(float *out, const float *q, const float *k,
                           const float *v, int n_tok, int n_heads, int head_dim,
                           int n_threads) {
    (void)n_threads;
    int dim = n_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < n_tok; i++) {
            const float *qi = q + (size_t)i * dim + (size_t)h * head_dim;

            /* Online softmax: find max and compute exp-sum in one pass */
            float row_max = -1e30f;
            for (int j = 0; j < n_tok; j++) {
                const float *kj = k + (size_t)j * dim + (size_t)h * head_dim;
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) s += qi[d] * kj[d];
                s *= scale;
                if (s > row_max) row_max = s;
            }

            float exp_sum = 0.0f;
            float *acc = out + (size_t)i * dim + (size_t)h * head_dim;
            memset(acc, 0, (size_t)head_dim * sizeof(float));
            for (int j = 0; j < n_tok; j++) {
                const float *kj = k + (size_t)j * dim + (size_t)h * head_dim;
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) s += qi[d] * kj[d];
                s *= scale;
                float w = expf(s - row_max);
                exp_sum += w;
                const float *vj = v + (size_t)j * dim + (size_t)h * head_dim;
                for (int d = 0; d < head_dim; d++) acc[d] += w * vj[d];
            }
            float inv = 1.0f / exp_sum;
            for (int d = 0; d < head_dim; d++) acc[d] *= inv;
        }
    }
}

/* ---- Sinusoidal timestep embedding ---- */

static void qimg_timestep_embed(float *out, float t, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i / (float)half * logf(10000.0f));
        float angle = t * freq;
        out[i]        = cosf(angle);  /* cos first (flip_sin_to_cos=True) */
        out[half + i] = sinf(angle);
    }
}

/* ---- RoPE for image tokens (2D: height, width) ---- */

static void qimg_rope_2d(float *q, float *k, int n_tok, int n_heads,
                         int head_dim, int h_patches, int w_patches,
                         const int *axes_dims, float theta) {
    /* axes_dims = [temporal, height, width] = [16, 56, 56]
     * For images: only apply height and width components.
     * temporal dims stay unrotated.
     * RoPE applied within each head's dimensions:
     *   [0..axes_dims[0]-1]: temporal (skip for images)
     *   [axes_dims[0]..axes_dims[0]+axes_dims[1]-1]: height
     *   [axes_dims[0]+axes_dims[1]..]: width */
    int t_dim = axes_dims[0];  /* temporal dims to skip */
    int h_dim = axes_dims[1];  /* height rotation dims */
    int w_dim = axes_dims[2];  /* width rotation dims */

    for (int tok = 0; tok < n_tok; tok++) {
        int ph = tok / w_patches;  /* height position */
        int pw = tok % w_patches;  /* width position */

        for (int head = 0; head < n_heads; head++) {
            float *qh = q + (size_t)tok * n_heads * head_dim +
                         (size_t)head * head_dim;
            float *kh = k + (size_t)tok * n_heads * head_dim +
                         (size_t)head * head_dim;

            /* Height RoPE: rotate pairs at offset t_dim */
            int h_half = h_dim / 2;
            for (int i = 0; i < h_half; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)h_dim);
                float angle = (float)ph * freq;
                float cs = cosf(angle), sn = sinf(angle);
                int idx = t_dim + 2 * i;
                if (idx + 1 < head_dim) {
                    float q0 = qh[idx], q1 = qh[idx + 1];
                    qh[idx]     = q0 * cs - q1 * sn;
                    qh[idx + 1] = q0 * sn + q1 * cs;
                    float k0 = kh[idx], k1 = kh[idx + 1];
                    kh[idx]     = k0 * cs - k1 * sn;
                    kh[idx + 1] = k0 * sn + k1 * cs;
                }
            }

            /* Width RoPE: rotate pairs at offset t_dim + h_dim */
            int w_half = w_dim / 2;
            for (int i = 0; i < w_half; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)w_dim);
                float angle = (float)pw * freq;
                float cs = cosf(angle), sn = sinf(angle);
                int idx = t_dim + h_dim + 2 * i;
                if (idx + 1 < head_dim) {
                    float q0 = qh[idx], q1 = qh[idx + 1];
                    qh[idx]     = q0 * cs - q1 * sn;
                    qh[idx + 1] = q0 * sn + q1 * cs;
                    float k0 = kh[idx], k1 = kh[idx + 1];
                    kh[idx]     = k0 * cs - k1 * sn;
                    kh[idx + 1] = k0 * sn + k1 * cs;
                }
            }
        }
    }
}

/* ---- RoPE for text tokens (1D: position) ---- */

static void qimg_rope_1d(float *q, float *k, int n_tok, int n_heads,
                         int head_dim, float theta) {
    int half = head_dim / 2;
    for (int tok = 0; tok < n_tok; tok++) {
        for (int head = 0; head < n_heads; head++) {
            float *qh = q + (size_t)tok * n_heads * head_dim +
                         (size_t)head * head_dim;
            float *kh = k + (size_t)tok * n_heads * head_dim +
                         (size_t)head * head_dim;
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                float angle = (float)tok * freq;
                float cs = cosf(angle), sn = sinf(angle);
                float q0 = qh[2*i], q1 = qh[2*i + 1];
                qh[2*i]     = q0 * cs - q1 * sn;
                qh[2*i + 1] = q0 * sn + q1 * cs;
                float k0 = kh[2*i], k1 = kh[2*i + 1];
                kh[2*i]     = k0 * cs - k1 * sn;
                kh[2*i + 1] = k0 * sn + k1 * cs;
            }
        }
    }
}

/* ---- Patchify: latent [ch, h, w] -> tokens [h/ps * w/ps, ch*ps*ps] ---- */

void qimg_dit_patchify(float *out, const float *latent,
                       int ch, int h, int w, int ps) {
    int hp = h / ps, wp = w / ps;
    int tok_dim = ch * ps * ps;
    for (int py = 0; py < hp; py++) {
        for (int px = 0; px < wp; px++) {
            int tok = py * wp + px;
            float *dst = out + (size_t)tok * tok_dim;
            int idx = 0;
            for (int c = 0; c < ch; c++)
                for (int dy = 0; dy < ps; dy++)
                    for (int dx = 0; dx < ps; dx++) {
                        int y = py * ps + dy;
                        int x = px * ps + dx;
                        dst[idx++] = latent[(size_t)c * h * w + y * w + x];
                    }
        }
    }
}

/* ---- Unpatchify: tokens [n_tok, ch*ps*ps] -> latent [ch, h, w] ---- */

void qimg_dit_unpatchify(float *out, const float *tokens,
                         int n_tok, int ch, int h, int w, int ps) {
    int hp = h / ps, wp = w / ps;
    int tok_dim = ch * ps * ps;
    (void)n_tok;
    for (int py = 0; py < hp; py++) {
        for (int px = 0; px < wp; px++) {
            int tok = py * wp + px;
            const float *src = tokens + (size_t)tok * tok_dim;
            int idx = 0;
            for (int c = 0; c < ch; c++)
                for (int dy = 0; dy < ps; dy++)
                    for (int dx = 0; dx < ps; dx++) {
                        int y = py * ps + dy;
                        int x = px * ps + dx;
                        out[(size_t)c * h * w + y * w + x] = src[idx++];
                    }
        }
    }
}

/* ---- Single dual-stream block forward ---- */

static void qimg_dit_block_forward(
    float *img, float *txt,
    int n_img, int n_txt,
    const float *t_emb,
    qimg_dit_block *blk,
    int block_idx,
    qimg_dit_model *m,
    int n_threads)
{
    int dim = m->hidden_dim;
    /* Dump helper macro: only active for the selected block */
    #define DUMP(name, data, n) do { \
        if (m->dump_fn && block_idx == m->dump_block) \
            m->dump_fn(name, data, n, m->dump_ctx); \
    } while(0)
    int nh = m->n_heads;
    int hd = m->head_dim;
    int mlp_h = m->mlp_hidden;
    float eps = m->ln_eps;
    int n_total = n_img + n_txt;

    /* Compute modulations from timestep embedding */
    float *t_silu = (float *)malloc((size_t)dim * sizeof(float));
    memcpy(t_silu, t_emb, (size_t)dim * sizeof(float));
    qimg_silu(t_silu, dim);

    /* img_mod: SiLU(t_emb) -> Linear -> [6*dim] */
    float *img_mod = (float *)malloc((size_t)6 * dim * sizeof(float));
    qimg_batch_gemm(img_mod, &blk->img_mod_w, &blk->img_mod_b,
                    t_silu, 1, 6 * dim, dim, n_threads);

    /* txt_mod: SiLU(t_emb) -> Linear -> [6*dim] */
    float *txt_mod = (float *)malloc((size_t)6 * dim * sizeof(float));
    qimg_batch_gemm(txt_mod, &blk->txt_mod_w, &blk->txt_mod_b,
                    t_silu, 1, 6 * dim, dim, n_threads);
    free(t_silu);

    /* Extract 6 modulation vectors each */
    float *img_shift1 = img_mod;
    float *img_scale1 = img_mod + dim;
    float *img_gate1  = img_mod + 2 * dim;
    float *img_shift2 = img_mod + 3 * dim;
    float *img_scale2 = img_mod + 4 * dim;
    float *img_gate2  = img_mod + 5 * dim;

    float *txt_shift1 = txt_mod;
    float *txt_scale1 = txt_mod + dim;
    float *txt_gate1  = txt_mod + 2 * dim;
    float *txt_shift2 = txt_mod + 3 * dim;
    float *txt_scale2 = txt_mod + 4 * dim;
    float *txt_gate2  = txt_mod + 5 * dim;

    /* ---- Pre-attention: adaLN + QKV projections ---- */
    float *img_normed = (float *)malloc((size_t)n_img * dim * sizeof(float));
    float *txt_normed = (float *)malloc((size_t)n_txt * dim * sizeof(float));
    qimg_adaln(img_normed, img, img_shift1, img_scale1, n_img, dim, eps);
    qimg_adaln(txt_normed, txt, txt_shift1, txt_scale1, n_txt, dim, eps);
    DUMP("img_adaln", img_normed, n_img * dim);

    /* Image Q, K, V */
    float *img_q = (float *)malloc((size_t)n_img * dim * sizeof(float));
    float *img_k = (float *)malloc((size_t)n_img * dim * sizeof(float));
    float *img_v = (float *)malloc((size_t)n_img * dim * sizeof(float));
    qimg_batch_gemm(img_q, &blk->attn_q_w, &blk->attn_q_b,
                    img_normed, n_img, dim, dim, n_threads);
    qimg_batch_gemm(img_k, &blk->attn_k_w, &blk->attn_k_b,
                    img_normed, n_img, dim, dim, n_threads);
    qimg_batch_gemm(img_v, &blk->attn_v_w, &blk->attn_v_b,
                    img_normed, n_img, dim, dim, n_threads);

    /* Text Q, K, V */
    float *txt_q = (float *)malloc((size_t)n_txt * dim * sizeof(float));
    float *txt_k = (float *)malloc((size_t)n_txt * dim * sizeof(float));
    float *txt_v = (float *)malloc((size_t)n_txt * dim * sizeof(float));
    qimg_batch_gemm(txt_q, &blk->attn_add_q_w, &blk->attn_add_q_b,
                    txt_normed, n_txt, dim, dim, n_threads);
    qimg_batch_gemm(txt_k, &blk->attn_add_k_w, &blk->attn_add_k_b,
                    txt_normed, n_txt, dim, dim, n_threads);
    qimg_batch_gemm(txt_v, &blk->attn_add_v_w, &blk->attn_add_v_b,
                    txt_normed, n_txt, dim, dim, n_threads);
    free(img_normed);
    free(txt_normed);

    /* QK RMSNorm (per-head) */
    if (blk->norm_q_w.data) {
        float *nq = qimg_dequant_full(&blk->norm_q_w);
        float *nk = qimg_dequant_full(&blk->norm_k_w);
        qimg_rmsnorm_per_head(img_q, n_img, nh, hd, nq, eps);
        qimg_rmsnorm_per_head(img_k, n_img, nh, hd, nk, eps);
        free(nq); free(nk);
    }
    DUMP("img_q_normed", img_q, n_img * dim);
    if (blk->norm_added_q_w.data) {
        float *nq = qimg_dequant_full(&blk->norm_added_q_w);
        float *nk = qimg_dequant_full(&blk->norm_added_k_w);
        qimg_rmsnorm_per_head(txt_q, n_txt, nh, hd, nq, eps);
        qimg_rmsnorm_per_head(txt_k, n_txt, nh, hd, nk, eps);
        free(nq); free(nk);
    }

    /* RoPE: 2D for image, 1D for text */
    int hp = (int)sqrtf((float)n_img);  /* assume square for now */
    int wp = n_img / hp;
    qimg_rope_2d(img_q, img_k, n_img, nh, hd, hp, wp,
                 m->rope_axes, m->rope_theta);
    qimg_rope_1d(txt_q, txt_k, n_txt, nh, hd, m->rope_theta);

    /* ---- Joint attention: concatenate txt + img ---- */
    float *q_all = (float *)malloc((size_t)n_total * dim * sizeof(float));
    float *k_all = (float *)malloc((size_t)n_total * dim * sizeof(float));
    float *v_all = (float *)malloc((size_t)n_total * dim * sizeof(float));
    memcpy(q_all, txt_q, (size_t)n_txt * dim * sizeof(float));
    memcpy(q_all + (size_t)n_txt * dim, img_q,
           (size_t)n_img * dim * sizeof(float));
    memcpy(k_all, txt_k, (size_t)n_txt * dim * sizeof(float));
    memcpy(k_all + (size_t)n_txt * dim, img_k,
           (size_t)n_img * dim * sizeof(float));
    memcpy(v_all, txt_v, (size_t)n_txt * dim * sizeof(float));
    memcpy(v_all + (size_t)n_txt * dim, img_v,
           (size_t)n_img * dim * sizeof(float));
    free(img_q); free(img_k); free(img_v);
    free(txt_q); free(txt_k); free(txt_v);

    float *attn_out = (float *)malloc((size_t)n_total * dim * sizeof(float));
    qimg_attention(attn_out, q_all, k_all, v_all, n_total, nh, hd, n_threads);
    free(q_all); free(k_all); free(v_all);

    /* Split attention output back to txt and img */
    float *txt_attn = attn_out;
    float *img_attn = attn_out + (size_t)n_txt * dim;

    /* Output projections */
    float *img_proj = (float *)malloc((size_t)n_img * dim * sizeof(float));
    float *txt_proj = (float *)malloc((size_t)n_txt * dim * sizeof(float));
    qimg_batch_gemm(img_proj, &blk->attn_out_w, &blk->attn_out_b,
                    img_attn, n_img, dim, dim, n_threads);
    qimg_batch_gemm(txt_proj, &blk->attn_add_out_w, &blk->attn_add_out_b,
                    txt_attn, n_txt, dim, dim, n_threads);
    free(attn_out);

    /* Gated residual */
    for (int t = 0; t < n_img; t++)
        for (int i = 0; i < dim; i++)
            img[(size_t)t * dim + i] += img_gate1[i] * img_proj[(size_t)t * dim + i];
    for (int t = 0; t < n_txt; t++)
        for (int i = 0; i < dim; i++)
            txt[(size_t)t * dim + i] += txt_gate1[i] * txt_proj[(size_t)t * dim + i];
    free(img_proj); free(txt_proj);
    DUMP("img_after_attn", img, n_img * dim);
    DUMP("txt_after_attn", txt, n_txt * dim);

    /* ---- MLP with adaLN ---- */
    /* Image MLP */
    float *img_tmp = (float *)malloc((size_t)n_img * dim * sizeof(float));
    qimg_adaln(img_tmp, img, img_shift2, img_scale2, n_img, dim, eps);
    float *img_ffn = (float *)malloc((size_t)n_img * mlp_h * sizeof(float));
    qimg_batch_gemm(img_ffn, &blk->img_mlp_fc1_w, &blk->img_mlp_fc1_b,
                    img_tmp, n_img, mlp_h, dim, n_threads);
    qimg_gelu(img_ffn, n_img * mlp_h);
    float *img_mlp_out = (float *)malloc((size_t)n_img * dim * sizeof(float));
    qimg_batch_gemm(img_mlp_out, &blk->img_mlp_fc2_w, &blk->img_mlp_fc2_b,
                    img_ffn, n_img, dim, mlp_h, n_threads);
    free(img_ffn);
    for (int t = 0; t < n_img; t++)
        for (int i = 0; i < dim; i++)
            img[(size_t)t * dim + i] += img_gate2[i] * img_mlp_out[(size_t)t * dim + i];
    free(img_mlp_out);
    free(img_tmp);

    /* Text MLP */
    float *txt_tmp = (float *)malloc((size_t)n_txt * dim * sizeof(float));
    qimg_adaln(txt_tmp, txt, txt_shift2, txt_scale2, n_txt, dim, eps);
    float *txt_ffn = (float *)malloc((size_t)n_txt * mlp_h * sizeof(float));
    qimg_batch_gemm(txt_ffn, &blk->txt_mlp_fc1_w, &blk->txt_mlp_fc1_b,
                    txt_tmp, n_txt, mlp_h, dim, n_threads);
    qimg_gelu(txt_ffn, n_txt * mlp_h);
    float *txt_mlp_out = (float *)malloc((size_t)n_txt * dim * sizeof(float));
    qimg_batch_gemm(txt_mlp_out, &blk->txt_mlp_fc2_w, &blk->txt_mlp_fc2_b,
                    txt_ffn, n_txt, dim, mlp_h, n_threads);
    free(txt_ffn);
    for (int t = 0; t < n_txt; t++)
        for (int i = 0; i < dim; i++)
            txt[(size_t)t * dim + i] += txt_gate2[i] * txt_mlp_out[(size_t)t * dim + i];
    free(txt_mlp_out);
    free(txt_tmp);
    DUMP("img_after_mlp", img, n_img * dim);
    DUMP("txt_after_mlp", txt, n_txt * dim);

    free(img_mod);
    free(txt_mod);
    #undef DUMP
}

/* ---- Full forward pass ---- */

void qimg_dit_forward(float *out, const float *img_tokens, int n_img,
                      const float *txt_tokens, int n_txt, float timestep,
                      qimg_dit_model *m, int n_threads) {
    int dim = m->hidden_dim;

    /* 1. Timestep embedding: sinusoidal(256) -> SiLU(Linear) -> Linear */
    float t_sinusoidal[256];
    qimg_timestep_embed(t_sinusoidal, timestep, 256);
    float *t_emb = (float *)malloc((size_t)dim * sizeof(float));
    qimg_batch_gemm(t_emb, &m->t_fc1_w, &m->t_fc1_b,
                    t_sinusoidal, 1, dim, 256, n_threads);
    qimg_silu(t_emb, dim);
    float *t_emb2 = (float *)malloc((size_t)dim * sizeof(float));
    qimg_batch_gemm(t_emb2, &m->t_fc2_w, &m->t_fc2_b,
                    t_emb, 1, dim, dim, n_threads);
    free(t_emb);
    t_emb = t_emb2;

    /* 2. Text input: RMSNorm -> Linear projection */
    float *txt = (float *)malloc((size_t)n_txt * dim * sizeof(float));
    {
        float *txt_normed = (float *)malloc((size_t)n_txt * m->txt_dim * sizeof(float));
        memcpy(txt_normed, txt_tokens, (size_t)n_txt * m->txt_dim * sizeof(float));
        if (m->txt_norm_w.data) {
            float *nw = qimg_dequant_full(&m->txt_norm_w);
            qimg_rmsnorm(txt_normed, n_txt, m->txt_dim, nw, m->ln_eps);
            free(nw);
        }
        qimg_batch_gemm(txt, &m->txt_in_w, &m->txt_in_b,
                        txt_normed, n_txt, dim, m->txt_dim, n_threads);
        free(txt_normed);
    }

    /* 3. Image input: Linear projection */
    float *img = (float *)malloc((size_t)n_img * dim * sizeof(float));
    qimg_batch_gemm(img, &m->img_in_w, &m->img_in_b,
                    img_tokens, n_img, dim, m->in_channels, n_threads);

    /* 4. Process all blocks */
    for (int L = 0; L < m->n_blocks; L++) {
        fprintf(stderr, "\r  qimg_dit: block %d/%d", L + 1, m->n_blocks);
        qimg_dit_block_forward(img, txt, n_img, n_txt, t_emb,
                               &m->blocks[L], L, m, n_threads);
    }
    fprintf(stderr, "\n");
    free(txt);

    /* 5. Final output: adaLN -> proj_out */
    {
        float *t_silu = (float *)malloc((size_t)dim * sizeof(float));
        memcpy(t_silu, t_emb, (size_t)dim * sizeof(float));
        qimg_silu(t_silu, dim);
        float *final_mod = (float *)malloc((size_t)2 * dim * sizeof(float));
        qimg_batch_gemm(final_mod, &m->norm_out_w, &m->norm_out_b,
                        t_silu, 1, 2 * dim, dim, n_threads);
        free(t_silu);

        /* LastLayer: scale, shift = chunk(emb, 2) — scale is FIRST half */
        float *scale = final_mod;
        float *shift = final_mod + dim;
        qimg_adaln(img, img, shift, scale, n_img, dim, m->ln_eps);
        free(final_mod);
    }
    qimg_batch_gemm(out, &m->proj_out_w, &m->proj_out_b,
                    img, n_img, m->out_channels, dim, n_threads);

    free(img);
    free(t_emb);
}

/* ---- GGUF loading ---- */

static qtensor qimg_find_tensor(const gguf_context *gguf, const char *name,
                                int required) {
    qtensor t = {0};
    int idx = -1;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name.str, name) == 0) { idx = (int)i; break; }
    }
    if (idx < 0) {
        if (required) fprintf(stderr, "qimg_dit: missing tensor '%s'\n", name);
        return t;
    }
    t.data = gguf_tensor_data(gguf, idx);
    t.type = gguf->tensors[idx].type;
    t.n_dims = (int)gguf->tensors[idx].n_dims;
    if (t.n_dims > 4) t.n_dims = 4;
    for (int d = 0; d < t.n_dims; d++) t.dims[d] = gguf->tensors[idx].dims[d];
    t.n_cols = (int)gguf->tensors[idx].dims[0];
    uint64_t n_rows = 1;
    for (uint32_t d = 1; d < gguf->tensors[idx].n_dims; d++)
        n_rows *= gguf->tensors[idx].dims[d];
    t.n_rows = (int)n_rows;
    return t;
}

qimg_dit_model *qimg_dit_load_gguf(const char *path) {
    fprintf(stderr, "qimg_dit: loading %s\n", path);
    gguf_context *gguf = gguf_open(path, 1);
    if (!gguf) { fprintf(stderr, "qimg_dit: failed to open %s\n", path); return NULL; }

    qimg_dit_model *m = (qimg_dit_model *)calloc(1, sizeof(qimg_dit_model));
    m->gguf_ctx = gguf;
    m->dump_block = -1;
    m->dump_fn = NULL;

    /* Detect architecture from tensors */
    m->hidden_dim = 3072;
    m->n_heads = 24;
    m->head_dim = 128;
    m->in_channels = 64;
    m->out_channels = 64;  /* same as in_channels for Flux-style */
    m->patch_size = 2;
    m->txt_dim = 3584;
    m->mlp_hidden = 12288;
    m->ln_eps = 1e-6f;
    m->rope_theta = 10000.0f;
    m->rope_axes[0] = 16;  /* temporal */
    m->rope_axes[1] = 56;  /* height */
    m->rope_axes[2] = 56;  /* width */

    /* Auto-detect from tensor shapes */
    #define QIMG_FIND(name) qimg_find_tensor(gguf, name, 0)
    #define QIMG_FIND_REQ(name) qimg_find_tensor(gguf, name, 1)

    /* Detect dims from img_in */
    qtensor img_in_probe = QIMG_FIND("img_in.weight");
    if (img_in_probe.data) {
        m->in_channels = img_in_probe.n_cols;      /* ne[0] = in_features */
        m->hidden_dim = img_in_probe.n_rows;        /* ne[1] = out_features */
        m->n_heads = m->hidden_dim / m->head_dim;
    }

    qtensor txt_in_probe = QIMG_FIND("txt_in.weight");
    if (txt_in_probe.data) {
        m->txt_dim = txt_in_probe.n_cols;
    }

    qtensor mlp_probe = QIMG_FIND("transformer_blocks.0.img_mlp.net.0.proj.weight");
    if (mlp_probe.data) {
        m->mlp_hidden = mlp_probe.n_rows;
    }

    /* Count blocks */
    m->n_blocks = 0;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        const char *nm = gguf->tensors[i].name.str;
        const char *bp = strstr(nm, "transformer_blocks.");
        if (bp) {
            bp += 19;  /* strlen("transformer_blocks.") */
            int blk = 0;
            while (*bp >= '0' && *bp <= '9') { blk = blk * 10 + (*bp - '0'); bp++; }
            if (blk + 1 > m->n_blocks) m->n_blocks = blk + 1;
        }
    }
    m->out_channels = m->in_channels;  /* Flux-style: same in/out */

    fprintf(stderr, "qimg_dit: hidden=%d heads=%d head_dim=%d blocks=%d "
            "in_ch=%d txt_dim=%d mlp=%d\n",
            m->hidden_dim, m->n_heads, m->head_dim, m->n_blocks,
            m->in_channels, m->txt_dim, m->mlp_hidden);

    /* Load global tensors */
    m->txt_norm_w = QIMG_FIND("txt_norm.weight");
    m->img_in_w = QIMG_FIND_REQ("img_in.weight");
    m->img_in_b = QIMG_FIND("img_in.bias");
    m->txt_in_w = QIMG_FIND_REQ("txt_in.weight");
    m->txt_in_b = QIMG_FIND("txt_in.bias");

    m->t_fc1_w = QIMG_FIND_REQ("time_text_embed.timestep_embedder.linear_1.weight");
    m->t_fc1_b = QIMG_FIND("time_text_embed.timestep_embedder.linear_1.bias");
    m->t_fc2_w = QIMG_FIND_REQ("time_text_embed.timestep_embedder.linear_2.weight");
    m->t_fc2_b = QIMG_FIND("time_text_embed.timestep_embedder.linear_2.bias");

    m->norm_out_w = QIMG_FIND_REQ("norm_out.linear.weight");
    m->norm_out_b = QIMG_FIND("norm_out.linear.bias");
    m->proj_out_w = QIMG_FIND_REQ("proj_out.weight");
    m->proj_out_b = QIMG_FIND("proj_out.bias");

    /* Load per-block tensors */
    m->blocks = (qimg_dit_block *)calloc((size_t)m->n_blocks, sizeof(qimg_dit_block));
    for (int L = 0; L < m->n_blocks; L++) {
        char name[256];
        qimg_dit_block *blk = &m->blocks[L];

        #define BLK_LOAD(field, suffix) do { \
            snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, L); \
            blk->field = qimg_find_tensor(gguf, name, 1); \
        } while(0)
        #define BLK_OPT(field, suffix) do { \
            snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, L); \
            blk->field = qimg_find_tensor(gguf, name, 0); \
        } while(0)

        /* Image attention */
        BLK_LOAD(attn_q_w,    "attn.to_q.weight");
        BLK_OPT (attn_q_b,    "attn.to_q.bias");
        BLK_LOAD(attn_k_w,    "attn.to_k.weight");
        BLK_OPT (attn_k_b,    "attn.to_k.bias");
        BLK_LOAD(attn_v_w,    "attn.to_v.weight");
        BLK_OPT (attn_v_b,    "attn.to_v.bias");
        BLK_LOAD(attn_out_w,  "attn.to_out.0.weight");
        BLK_OPT (attn_out_b,  "attn.to_out.0.bias");

        /* Text attention */
        BLK_LOAD(attn_add_q_w,   "attn.add_q_proj.weight");
        BLK_OPT (attn_add_q_b,   "attn.add_q_proj.bias");
        BLK_LOAD(attn_add_k_w,   "attn.add_k_proj.weight");
        BLK_OPT (attn_add_k_b,   "attn.add_k_proj.bias");
        BLK_LOAD(attn_add_v_w,   "attn.add_v_proj.weight");
        BLK_OPT (attn_add_v_b,   "attn.add_v_proj.bias");
        BLK_LOAD(attn_add_out_w, "attn.to_add_out.weight");
        BLK_OPT (attn_add_out_b, "attn.to_add_out.bias");

        /* QK norms */
        BLK_OPT(norm_q_w,       "attn.norm_q.weight");
        BLK_OPT(norm_k_w,       "attn.norm_k.weight");
        BLK_OPT(norm_added_q_w, "attn.norm_added_q.weight");
        BLK_OPT(norm_added_k_w, "attn.norm_added_k.weight");

        /* Image modulation + MLP */
        BLK_LOAD(img_mod_w,     "img_mod.1.weight");
        BLK_OPT (img_mod_b,     "img_mod.1.bias");
        BLK_LOAD(img_mlp_fc1_w, "img_mlp.net.0.proj.weight");
        BLK_OPT (img_mlp_fc1_b, "img_mlp.net.0.proj.bias");
        BLK_LOAD(img_mlp_fc2_w, "img_mlp.net.2.weight");
        BLK_OPT (img_mlp_fc2_b, "img_mlp.net.2.bias");

        /* Text modulation + MLP */
        BLK_LOAD(txt_mod_w,     "txt_mod.1.weight");
        BLK_OPT (txt_mod_b,     "txt_mod.1.bias");
        BLK_LOAD(txt_mlp_fc1_w, "txt_mlp.net.0.proj.weight");
        BLK_OPT (txt_mlp_fc1_b, "txt_mlp.net.0.proj.bias");
        BLK_LOAD(txt_mlp_fc2_w, "txt_mlp.net.2.weight");
        BLK_OPT (txt_mlp_fc2_b, "txt_mlp.net.2.bias");

        #undef BLK_LOAD
        #undef BLK_OPT
    }

    #undef QIMG_FIND
    #undef QIMG_FIND_REQ

    fprintf(stderr, "qimg_dit: loaded %d blocks successfully\n", m->n_blocks);
    return m;
}

/* ---- Load DiT from FP8 safetensors ---- */

/* Helper: create qtensor pointing at mmap'd safetensors data (zero-copy for weights).
 * Small tensors (1D biases/norms) are converted to F32 immediately.
 * Large tensors (2D weights) stay as raw FP8 and are dequanted per-row during GEMM. */
static qtensor qimg_st_make_tensor(st_context *st, const char *name,
                                    float ***buf_list, int *buf_count) {
    qtensor t = {0};
    int idx = safetensors_find(st, name);
    if (idx < 0) return t;

    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t numel = 1;
    for (int d = 0; d < ndims; d++) numel *= shape[d];

    const char *dtype = safetensors_dtype(st, idx);
    int is_fp8 = (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0);

    if (ndims >= 2) {
        /* 2D weight: keep as mmap'd FP8, dequant per-row during GEMM */
        t.data = safetensors_data(st, idx);
        t.type = is_fp8 ? QIMG_TYPE_F8_E4M3 : GGML_TYPE_F32;
        t.n_dims = ndims;
        for (int d = 0; d < ndims && d < 4; d++) t.dims[d] = shape[d];
        t.n_rows = (int)shape[0];
        t.n_cols = (int)shape[1];
    } else {
        /* 1D bias/norm: convert to F32 immediately (small) */
        float *f32 = (float *)malloc(numel * sizeof(float));
        const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
        if (is_fp8) {
            for (size_t i = 0; i < numel; i++)
                f32[i] = qimg_fp8_e4m3_to_f32(src[i]);
        } else if (strcmp(dtype, "BF16") == 0) {
            const uint16_t *bf = (const uint16_t *)src;
            for (size_t i = 0; i < numel; i++) {
                uint32_t bits = (uint32_t)bf[i] << 16;
                memcpy(&f32[i], &bits, 4);
            }
        } else if (strcmp(dtype, "F32") == 0) {
            memcpy(f32, src, numel * sizeof(float));
        }
        *buf_list = (float **)realloc(*buf_list, (size_t)(*buf_count + 1) * sizeof(float *));
        (*buf_list)[*buf_count] = f32;
        (*buf_count)++;
        t.data = f32;
        t.type = GGML_TYPE_F32;
        t.n_dims = 1;
        t.dims[0] = shape[0];
        t.n_rows = 1;
        t.n_cols = (int)shape[0];
    }
    return t;
}

qimg_dit_model *qimg_dit_load_safetensors(const char *path) {
    fprintf(stderr, "qimg_dit: loading safetensors %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "qimg_dit: failed to open %s\n", path); return NULL; }

    qimg_dit_model *m = (qimg_dit_model *)calloc(1, sizeof(qimg_dit_model));
    m->dump_block = -1;

    float **bufs = NULL;
    int n_bufs = 0;

    m->hidden_dim = 3072; m->n_heads = 24; m->head_dim = 128;
    m->in_channels = 64; m->out_channels = 64; m->patch_size = 2;
    m->txt_dim = 3584; m->mlp_hidden = 12288;
    m->ln_eps = 1e-6f; m->rope_theta = 10000.0f;
    m->rope_axes[0] = 16; m->rope_axes[1] = 56; m->rope_axes[2] = 56;

    /* Count blocks */
    m->n_blocks = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *bp = strstr(nm, "transformer_blocks.");
        if (bp) {
            int blk = atoi(bp + 19);
            if (blk + 1 > m->n_blocks) m->n_blocks = blk + 1;
        }
    }
    fprintf(stderr, "qimg_dit: %d blocks detected\n", m->n_blocks);

    #define ST_LOAD(field, name) m->field = qimg_st_make_tensor(st, name, &bufs, &n_bufs)

    ST_LOAD(txt_norm_w, "txt_norm.weight");
    ST_LOAD(img_in_w, "img_in.weight"); ST_LOAD(img_in_b, "img_in.bias");
    ST_LOAD(txt_in_w, "txt_in.weight"); ST_LOAD(txt_in_b, "txt_in.bias");
    ST_LOAD(t_fc1_w, "time_text_embed.timestep_embedder.linear_1.weight");
    ST_LOAD(t_fc1_b, "time_text_embed.timestep_embedder.linear_1.bias");
    ST_LOAD(t_fc2_w, "time_text_embed.timestep_embedder.linear_2.weight");
    ST_LOAD(t_fc2_b, "time_text_embed.timestep_embedder.linear_2.bias");
    ST_LOAD(norm_out_w, "norm_out.linear.weight");
    ST_LOAD(norm_out_b, "norm_out.linear.bias");
    ST_LOAD(proj_out_w, "proj_out.weight");
    ST_LOAD(proj_out_b, "proj_out.bias");

    m->blocks = (qimg_dit_block *)calloc((size_t)m->n_blocks, sizeof(qimg_dit_block));
    for (int L = 0; L < m->n_blocks; L++) {
        qimg_dit_block *blk = &m->blocks[L];
        char name[256];
        #define BLK_ST(field, suffix) do { \
            snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, L); \
            blk->field = qimg_st_make_tensor(st, name, &bufs, &n_bufs); \
        } while(0)

        BLK_ST(attn_q_w, "attn.to_q.weight"); BLK_ST(attn_q_b, "attn.to_q.bias");
        BLK_ST(attn_k_w, "attn.to_k.weight"); BLK_ST(attn_k_b, "attn.to_k.bias");
        BLK_ST(attn_v_w, "attn.to_v.weight"); BLK_ST(attn_v_b, "attn.to_v.bias");
        BLK_ST(attn_out_w, "attn.to_out.0.weight"); BLK_ST(attn_out_b, "attn.to_out.0.bias");
        BLK_ST(attn_add_q_w, "attn.add_q_proj.weight"); BLK_ST(attn_add_q_b, "attn.add_q_proj.bias");
        BLK_ST(attn_add_k_w, "attn.add_k_proj.weight"); BLK_ST(attn_add_k_b, "attn.add_k_proj.bias");
        BLK_ST(attn_add_v_w, "attn.add_v_proj.weight"); BLK_ST(attn_add_v_b, "attn.add_v_proj.bias");
        BLK_ST(attn_add_out_w, "attn.to_add_out.weight"); BLK_ST(attn_add_out_b, "attn.to_add_out.bias");
        BLK_ST(norm_q_w, "attn.norm_q.weight"); BLK_ST(norm_k_w, "attn.norm_k.weight");
        BLK_ST(norm_added_q_w, "attn.norm_added_q.weight"); BLK_ST(norm_added_k_w, "attn.norm_added_k.weight");
        BLK_ST(img_mod_w, "img_mod.1.weight"); BLK_ST(img_mod_b, "img_mod.1.bias");
        BLK_ST(img_mlp_fc1_w, "img_mlp.net.0.proj.weight"); BLK_ST(img_mlp_fc1_b, "img_mlp.net.0.proj.bias");
        BLK_ST(img_mlp_fc2_w, "img_mlp.net.2.weight"); BLK_ST(img_mlp_fc2_b, "img_mlp.net.2.bias");
        BLK_ST(txt_mod_w, "txt_mod.1.weight"); BLK_ST(txt_mod_b, "txt_mod.1.bias");
        BLK_ST(txt_mlp_fc1_w, "txt_mlp.net.0.proj.weight"); BLK_ST(txt_mlp_fc1_b, "txt_mlp.net.0.proj.bias");
        BLK_ST(txt_mlp_fc2_w, "txt_mlp.net.2.weight"); BLK_ST(txt_mlp_fc2_b, "txt_mlp.net.2.bias");
        #undef BLK_ST
    }
    #undef ST_LOAD

    m->predequant_bufs = bufs;
    m->n_predequant = n_bufs;
    /* Keep st_context alive — 2D weight data pointers reference the mmap */
    m->gguf_ctx = st;
    m->gguf_ctx_is_st = 1;

    fprintf(stderr, "qimg_dit: loaded %d blocks (FP8 weights mmap'd, biases F32)\n",
            m->n_blocks);
    return m;
}

/* ---- Pre-dequantize a single qtensor in-place: convert to F32 ---- */

static void qimg_predequant_tensor(qtensor *t, float ***buf_list, int *buf_count) {
    if (!t->data || t->type == GGML_TYPE_F32) return;
    int n = t->n_rows * t->n_cols;
    if (n <= 0) return;
    float *f32 = (float *)malloc((size_t)n * sizeof(float));
    for (int r = 0; r < t->n_rows; r++)
        qimg_dequant_row(t, r, f32 + (size_t)r * t->n_cols);
    /* Patch qtensor to point at F32 data */
    t->data = f32;
    t->type = GGML_TYPE_F32;
    /* Track allocation for later free */
    *buf_list = (float **)realloc(*buf_list, (size_t)(*buf_count + 1) * sizeof(float *));
    (*buf_list)[*buf_count] = f32;
    (*buf_count)++;
}

void qimg_dit_predequant(qimg_dit_model *m) {
    if (m->is_predequanted) return;
    float **bufs = NULL;
    int n_bufs = 0;

    fprintf(stderr, "qimg_dit: pre-dequantizing global weights to F32...\n");
    clock_t t0 = clock();

    /* Only predequant GLOBAL weights (small, ~50MB total).
     * Block weights (~1.3GB each × 60 = 79GB) stay quantized and are
     * dequanted on-the-fly per GEMM call using the SIMD dot product. */
    #define PD(ptr) qimg_predequant_tensor(ptr, &bufs, &n_bufs)
    PD(&m->txt_norm_w);
    PD(&m->img_in_w); PD(&m->img_in_b);
    PD(&m->txt_in_w); PD(&m->txt_in_b);
    PD(&m->t_fc1_w); PD(&m->t_fc1_b);
    PD(&m->t_fc2_w); PD(&m->t_fc2_b);
    PD(&m->norm_out_w); PD(&m->norm_out_b);
    PD(&m->proj_out_w); PD(&m->proj_out_b);

    /* Predequant small per-block tensors (norms, biases — a few KB each) */
    for (int L = 0; L < m->n_blocks; L++) {
        qimg_dit_block *blk = &m->blocks[L];
        /* Biases: [3072] or [12288] or [18432] F32 — small */
        PD(&blk->attn_q_b); PD(&blk->attn_k_b); PD(&blk->attn_v_b);
        PD(&blk->attn_out_b);
        PD(&blk->attn_add_q_b); PD(&blk->attn_add_k_b); PD(&blk->attn_add_v_b);
        PD(&blk->attn_add_out_b);
        /* Per-head norms: [128] — tiny */
        PD(&blk->norm_q_w); PD(&blk->norm_k_w);
        PD(&blk->norm_added_q_w); PD(&blk->norm_added_k_w);
        /* Modulation + MLP biases */
        PD(&blk->img_mod_b); PD(&blk->txt_mod_b);
        PD(&blk->img_mlp_fc1_b); PD(&blk->img_mlp_fc2_b);
        PD(&blk->txt_mlp_fc1_b); PD(&blk->txt_mlp_fc2_b);
    }
    #undef PD

    m->predequant_bufs = bufs;
    m->n_predequant = n_bufs;
    m->is_predequanted = 1;

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "qimg_dit: pre-dequanted %d tensors (globals+biases) in %.1fs\n",
            n_bufs, elapsed);
}

void qimg_dit_free(qimg_dit_model *m) {
    if (!m) return;
    /* Free pre-dequanted buffers */
    if (m->predequant_bufs) {
        for (int i = 0; i < m->n_predequant; i++)
            free(m->predequant_bufs[i]);
        free(m->predequant_bufs);
    }
    if (m->blocks) free(m->blocks);
    if (m->gguf_ctx) {
        if (m->gguf_ctx_is_st)
            safetensors_close((st_context *)m->gguf_ctx);
        else
            gguf_close((gguf_context *)m->gguf_ctx);
    }
    free(m);
}

#endif /* QIMG_DIT_IMPLEMENTATION */
#endif /* QIMG_DIT_H */
