/*
 * pixel_perfect_depth.h - Pixel-Perfect-Depth monocular depth estimation (CPU)
 *
 * Pipeline: DA2 semantic encoder (DINOv2 ViT-L) → DiT diffusion (4 Euler steps) → depth
 *
 * Usage:
 *   #define PIXEL_PERFECT_DEPTH_IMPLEMENTATION
 *   #include "pixel_perfect_depth.h"
 *
 * Dependencies: pth_loader.h, ggml_dequant.h
 *
 * API:
 *   ppd_model  *ppd_load(const char *ppd_path, const char *sem_path, int verbose);
 *   void        ppd_free(ppd_model *m);
 *   ppd_result  ppd_predict(ppd_model *m, const uint8_t *rgb, int w, int h, int n_threads);
 *   void        ppd_result_free(ppd_result *r);
 */
#ifndef PIXEL_PERFECT_DEPTH_H
#define PIXEL_PERFECT_DEPTH_H

#include <stdint.h>
#include <stddef.h>
#include "pth_loader.h"
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Data Structures ---- */

typedef struct {
    float *ln1_w, *ln1_b, *ln2_w, *ln2_b;       /* LayerNorm F32 [dim] */
    uint16_t *qkv_w;  float *qkv_b;              /* [3*dim, dim] F16 + [3*dim] F32 */
    uint16_t *out_w;   float *out_b;              /* [dim, dim] */
    uint16_t *ffn_up_w;  float *ffn_up_b;         /* [4*dim, dim] */
    uint16_t *ffn_down_w; float *ffn_down_b;      /* [dim, 4*dim] */
    float *ls1, *ls2;                              /* LayerScale [dim] F32 */
} ppd_sem_block;

typedef struct {
    uint16_t *qkv_w;  float *qkv_b;              /* [3*dim, dim] */
    float *q_norm_w, *q_norm_b;                    /* [head_dim] per-head QK-norm */
    float *k_norm_w, *k_norm_b;
    uint16_t *out_w;   float *out_b;
    uint16_t *fc1_w;   float *fc1_b;              /* [4*dim, dim] */
    uint16_t *fc2_w;   float *fc2_b;              /* [dim, 4*dim] */
    uint16_t *adaln_w; float *adaln_b;            /* [6*dim, dim] adaLN modulation */
} ppd_dit_block;

typedef struct ppd_model {
    /* DA2 semantic encoder config */
    int sem_dim, sem_n_heads, sem_head_dim, sem_ffn, sem_n_blocks;
    int sem_patch_size;
    float sem_ln_eps;
    float *sem_pe_w;                               /* patch_embed conv [dim, 3, 14, 14] F32 */
    float *sem_pe_b;                               /* patch_embed bias [dim] F32 */
    float *sem_cls;                                /* CLS token [dim] F32 */
    float *sem_pos;                                /* pos_embed [1+n_patches, dim] F32 */
    int sem_pos_n;                                 /* total pos_embed tokens (1+n_patches) */
    int sem_pos_gH, sem_pos_gW;                    /* original pos_embed grid dims */
    ppd_sem_block *sem_blocks;
    float *sem_norm_w, *sem_norm_b;                /* final LayerNorm */

    /* DiT config */
    int dit_dim, dit_n_heads, dit_head_dim, dit_ffn, dit_n_blocks;
    int dit_patch_size;
    float dit_ln_eps;
    float dit_rope_freq;
    float *dit_xe_w, *dit_xe_b;                    /* x_embedder conv [dim, 4, 16, 16] F32 */
    uint16_t *dit_te_w1; float *dit_te_b1;         /* t_embedder FC1 [dim, 256] */
    uint16_t *dit_te_w2; float *dit_te_b2;         /* t_embedder FC2 [dim, dim] */
    uint16_t *dit_fus_w[3]; float *dit_fus_b[3];   /* proj_fusion 3 layers */
    uint16_t *dit_fin_adaln_w; float *dit_fin_adaln_b; /* final adaLN [2*dim, dim] */
    uint16_t *dit_fin_proj_w;  float *dit_fin_proj_b;  /* final linear [64, dim] */
    ppd_dit_block *dit_blocks;

    int verbose;
} ppd_model;

typedef struct {
    float *depth;
    int width, height;
} ppd_result;

ppd_model  *ppd_load(const char *ppd_path, const char *sem_path, int verbose);
void        ppd_free(ppd_model *m);
ppd_result  ppd_predict(ppd_model *m, const uint8_t *rgb, int w, int h, int n_threads);
void        ppd_result_free(ppd_result *r);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef PIXEL_PERFECT_DEPTH_IMPLEMENTATION

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

#define CPU_COMPUTE_IMPLEMENTATION
#include "cpu_compute.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double ppd_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}


/* ======================================================================== */
/* Section 1: Helpers                                                        */
/* ======================================================================== */

/* F32 → F16 conversion using truncation (matches GPU CUDA runner) */
static uint16_t ppd_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7BFF); /* clamp to ±65504, avoid Inf */
    return (uint16_t)(sign | (exp << 10) | mant);
}

/* Batch F32 → F16 conversion. Caller must free returned buffer. */
static uint16_t *ppd_fp32_to_fp16(const float *src, int n) {
    uint16_t *dst = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
    for (int i = 0; i < n; i++) dst[i] = ppd_f32_to_f16(src[i]);
    return dst;
}

/* LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b */
static void ppd_layernorm(float *dst, const float *src, const float *w,
                           const float *b, int n_tok, int dim, float eps) {
    cpu_layernorm(dst, src, w, b, n_tok, dim, eps);
}

/* GELU (tanh approximation) */
static void ppd_gelu(float *x, int n) { cpu_gelu(x, n); }

/* SiLU: x * sigmoid(x) */
static void ppd_silu(float *x, int n) { cpu_silu(x, n); }

/* Bilinear resize for CHW float image */
static void ppd_bilinear_resize(float *dst, const float *src,
                                  int C, int Hi, int Wi, int Ho, int Wo) {
    cpu_bilinear_resize(dst, src, C, Hi, Wi, Ho, Wo);
}

/* Interpolate position embeddings from original grid to new grid */
static float *ppd_interpolate_pos_embed(const float *patch_pe, int dim,
                                          int orig_gH, int orig_gW,
                                          int new_gH, int new_gW) {
    int new_n = new_gH * new_gW;
    float *out = (float *)malloc((size_t)new_n * dim * sizeof(float));
    for (int oh = 0; oh < new_gH; oh++) {
        for (int ow = 0; ow < new_gW; ow++) {
            float fy = (new_gH > 1) ? (float)oh * (orig_gH - 1) / (new_gH - 1) : 0;
            float fx = (new_gW > 1) ? (float)ow * (orig_gW - 1) / (new_gW - 1) : 0;
            int y0 = (int)fy, x0 = (int)fx;
            int y1 = (y0 + 1 < orig_gH) ? y0 + 1 : y0;
            int x1 = (x0 + 1 < orig_gW) ? x0 + 1 : x0;
            float dy = fy - y0, dx = fx - x0;
            int dst_tok = oh * new_gW + ow;
            for (int d = 0; d < dim; d++) {
                out[dst_tok * dim + d] =
                    patch_pe[(y0 * orig_gW + x0) * dim + d] * (1-dy) * (1-dx) +
                    patch_pe[(y0 * orig_gW + x1) * dim + d] * (1-dy) * dx +
                    patch_pe[(y1 * orig_gW + x0) * dim + d] * dy * (1-dx) +
                    patch_pe[(y1 * orig_gW + x1) * dim + d] * dy * dx;
            }
        }
    }
    return out;
}

/* Sinusoidal timestep embedding */
static void ppd_sinusoidal_embed(float *out, float t, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-logf(10000.0f) * (float)i / (float)half);
        float arg = t * freq;
        out[i] = cosf(arg);
        out[half + i] = sinf(arg);
    }
}

/* Box-Muller random normal generation */
static void ppd_generate_randn(float *buf, int n) {
    for (int i = 0; i < n - 1; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        float r = sqrtf(-2.0f * logf(u1));
        buf[i] = r * cosf(2.0f * (float)M_PI * u2);
        buf[i + 1] = r * sinf(2.0f * (float)M_PI * u2);
    }
    if (n % 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        buf[n - 1] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    }
}

/* Generate 2D grid position indices */
static void ppd_generate_grid_pos(int *pos_y, int *pos_x, int gH, int gW) {
    for (int h = 0; h < gH; h++)
        for (int w = 0; w < gW; w++) {
            int idx = h * gW + w;
            pos_y[idx] = h;
            pos_x[idx] = w;
        }
}

/* ======================================================================== */
/* Section 2: Core Operations                                                */
/* ======================================================================== */

/* ---- Threaded GEMM: Y[tok][row] = W[row,:] * X[tok,:] + bias[row] ---- */

static void ppd_gemm(float *dst, const uint16_t *W, const float *bias,
                      const float *src, int n_tok, int n_out, int n_in,
                      int n_threads) {
    cpu_gemm_f16(dst, W, bias, src, n_tok, n_out, n_in, n_threads);
}

/* ---- Threaded attention: head-parallel, flash-attention style ---- */

static void ppd_attention(float *out, const float *qkv, int n_tok, int dim,
                           int n_heads, int head_dim, int n_threads) {
    cpu_attention(out, qkv, n_tok, dim, n_heads, head_dim, n_threads);
}

/* Per-head QK LayerNorm on Q or K within interleaved QKV buffer */
static void ppd_qk_norm(float *qkv, const float *w, const float *b,
                          int n_tok, int n_heads, int head_dim, int stride,
                          float eps) {
    cpu_qk_norm(qkv, n_tok, n_heads, head_dim, stride, w, b, eps);
}

/* 2D RoPE: separate Y and X rotations within each head */
static void ppd_rope_2d(float *vec, int n_tok, int n_heads, int head_dim,
                          int stride, const int *pos_y, const int *pos_x,
                          float freq_base) {
    cpu_rope_2d(vec, n_tok, n_heads, head_dim, stride, pos_y, pos_x, freq_base);
}

/* adaLN: y = LN(x) * (1 + scale) + shift */
static void ppd_adaln_modulate(float *dst, const float *src,
                                const float *shift, const float *scale,
                                int n_tok, int dim, float eps) {
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
            y[i] = (x[i] - mean) * inv * (1.0f + scale[i]) + shift[i];
    }
}

/* Gate residual: dst[t*D+d] += gate[d] * src[t*D+d] */
static void ppd_gate_residual(float *dst, const float *src, const float *gate,
                               int n_tok, int dim) {
    for (int t = 0; t < n_tok; t++) {
        float *d = dst + t * dim;
        const float *s = src + t * dim;
        for (int i = 0; i < dim; i++) d[i] += gate[i] * s[i];
    }
}

/* Pixel shuffle 2x: [gH*gW, 4*dim] → [2*gH*2*gW, dim] */
static void ppd_pixel_shuffle_2x(float *dst, const float *src,
                                   int gH, int gW, int dim) {
    int oH = gH * 2, oW = gW * 2;
    for (int oh = 0; oh < oH; oh++) {
        for (int ow = 0; ow < oW; ow++) {
            int ih = oh / 2, iw = ow / 2;
            int sub = (oh % 2) * 2 + (ow % 2);
            int tok_in = ih * gW + iw;
            int tok_out = oh * oW + ow;
            for (int d = 0; d < dim; d++)
                dst[tok_out * dim + d] = src[tok_in * 4 * dim + sub * dim + d];
        }
    }
}

/* Unpatchify: [n_tok, ps*ps] → [gH*ps, gW*ps] */
static void ppd_unpatchify(float *dst, const float *src, int gH, int gW, int ps) {
    int H = gH * ps, W = gW * ps;
    for (int oh = 0; oh < H; oh++) {
        for (int ow = 0; ow < W; ow++) {
            int ph = oh / ps, pw = ow / ps;
            int kh = oh % ps, kw = ow % ps;
            int tok = ph * gW + pw;
            dst[oh * W + ow] = src[tok * ps * ps + kh * ps + kw];
        }
    }
}

/* Concat along dim: [N, dim_a] + [N, dim_b] → [N, dim_a+dim_b] */
static void ppd_concat_dim(float *dst, const float *a, const float *b,
                             int n_tok, int dim_a, int dim_b) {
    int total = dim_a + dim_b;
    for (int t = 0; t < n_tok; t++) {
        memcpy(dst + t * total, a + t * dim_a, (size_t)dim_a * sizeof(float));
        memcpy(dst + t * total + dim_a, b + t * dim_b, (size_t)dim_b * sizeof(float));
    }
}

/* Patch embedding via manual im2col conv2d.
 * src: [Ci, H, W], weight: [Co, Ci, kH, kW], bias: [Co]
 * Writes to dst[1+n_patches, Co] (token 0 is CLS, patches start at 1) */
static void ppd_patch_embed(float *dst, const float *src, const float *weight,
                              const float *bias, int H, int W,
                              int Ci, int Co, int ps, int gH, int gW) {
    /* Patch tokens start at row 1, row 0 reserved for CLS */
    for (int ph = 0; ph < gH; ph++) {
        for (int pw = 0; pw < gW; pw++) {
            int tok = 1 + ph * gW + pw;
            float *out = dst + tok * Co;
            for (int co = 0; co < Co; co++) {
                float sum = bias ? bias[co] : 0.0f;
                for (int ci = 0; ci < Ci; ci++) {
                    for (int kh = 0; kh < ps; kh++) {
                        int ih = ph * ps + kh;
                        for (int kw = 0; kw < ps; kw++) {
                            int iw = pw * ps + kw;
                            sum += weight[((co * Ci + ci) * ps + kh) * ps + kw]
                                 * src[ci * H * W + ih * W + iw];
                        }
                    }
                }
                out[co] = sum;
            }
        }
    }
}

/* DiT patch embedding (no CLS token) */
static void ppd_dit_patch_embed(float *dst, const float *src, const float *weight,
                                  const float *bias, int H, int W,
                                  int Ci, int Co, int ps, int gH, int gW) {
    for (int ph = 0; ph < gH; ph++) {
        for (int pw = 0; pw < gW; pw++) {
            int tok = ph * gW + pw;
            float *out = dst + tok * Co;
            for (int co = 0; co < Co; co++) {
                float sum = bias ? bias[co] : 0.0f;
                for (int ci = 0; ci < Ci; ci++) {
                    for (int kh = 0; kh < ps; kh++) {
                        int ih = ph * ps + kh;
                        for (int kw = 0; kw < ps; kw++) {
                            int iw = pw * ps + kw;
                            sum += weight[((co * Ci + ci) * ps + kh) * ps + kw]
                                 * src[ci * H * W + ih * W + iw];
                        }
                    }
                }
                out[co] = sum;
            }
        }
    }
}

/* ======================================================================== */
/* Section 3: Weight Loading                                                 */
/* ======================================================================== */

/* Convert BF16 raw data to F16 */
static uint16_t *ppd_bf16_to_fp16(const uint16_t *bf16, int n) {
    uint16_t *fp16 = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
    for (int i = 0; i < n; i++) {
        /* BF16 → F32 → F16 */
        uint32_t f32 = (uint32_t)bf16[i] << 16;
        float f;
        memcpy(&f, &f32, 4);
        fp16[i] = ppd_f32_to_f16(f);
    }
    return fp16;
}

/* Load tensor as F16 (malloc + convert from F32 or BF16) */
static uint16_t *ppd_load_f16(pth_context *pth, const char *name, int verbose) {
    int idx = pth_find(pth, name);
    if (idx < 0) {
        if (verbose >= 2) fprintf(stderr, "ppd: tensor '%s' not found\n", name);
        return NULL;
    }
    const uint64_t *shape = pth_shape(pth, idx);
    int ndims = pth_ndims(pth, idx);
    const char *dtype = pth_dtype(pth, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];

    uint16_t *dst;
    if (strcmp(dtype, "BF16") == 0 || strcmp(dtype, "BFloat16") == 0) {
        const uint16_t *src = (const uint16_t *)pth_data(pth, idx);
        dst = ppd_bf16_to_fp16(src, (int)n);
    } else if (strcmp(dtype, "F16") == 0 || strcmp(dtype, "Float16") == 0) {
        dst = (uint16_t *)malloc(n * sizeof(uint16_t));
        memcpy(dst, pth_data(pth, idx), n * sizeof(uint16_t));
    } else {
        /* Assume F32 */
        const float *src = (const float *)pth_data(pth, idx);
        dst = ppd_fp32_to_fp16(src, (int)n);
    }

    if (verbose >= 2) {
        fprintf(stderr, "ppd: loaded F16 '%s' (%s) [", name, dtype);
        for (int d = 0; d < ndims; d++) fprintf(stderr, "%s%lu", d ? "," : "", (unsigned long)shape[d]);
        fprintf(stderr, "]\n");
    }
    return dst;
}

/* Load tensor as F32 (malloc + convert from BF16 if needed, or copy) */
static float *ppd_load_f32(pth_context *pth, const char *name, int verbose) {
    int idx = pth_find(pth, name);
    if (idx < 0) {
        if (verbose >= 2) fprintf(stderr, "ppd: tensor '%s' not found\n", name);
        return NULL;
    }
    const uint64_t *shape = pth_shape(pth, idx);
    int ndims = pth_ndims(pth, idx);
    const char *dtype = pth_dtype(pth, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];

    float *dst = (float *)malloc(n * sizeof(float));
    if (strcmp(dtype, "BF16") == 0 || strcmp(dtype, "BFloat16") == 0) {
        const uint16_t *src = (const uint16_t *)pth_data(pth, idx);
        for (size_t i = 0; i < n; i++) {
            uint32_t f32 = (uint32_t)src[i] << 16;
            memcpy(&dst[i], &f32, 4);
        }
    } else if (strcmp(dtype, "F16") == 0 || strcmp(dtype, "Float16") == 0) {
        /* F16 → F32 */
        const uint16_t *src = (const uint16_t *)pth_data(pth, idx);
        for (size_t i = 0; i < n; i++) {
#if defined(__F16C__)
            __m128i v16 = _mm_set1_epi16((short)src[i]);
            __m128 v32 = _mm_cvtph_ps(v16);
            dst[i] = _mm_cvtss_f32(v32);
#else
            uint16_t h = src[i];
            uint32_t sign = ((uint32_t)(h >> 15)) << 31;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f32;
            if (exp == 0) f32 = sign;
            else if (exp == 31) f32 = sign | 0x7F800000 | (mant << 13);
            else f32 = sign | ((exp - 15 + 127) << 23) | (mant << 13);
            memcpy(&dst[i], &f32, 4);
#endif
        }
    } else {
        /* F32 */
        memcpy(dst, pth_data(pth, idx), n * sizeof(float));
    }

    if (verbose >= 2) {
        fprintf(stderr, "ppd: loaded F32 '%s' [", name);
        for (int d = 0; d < ndims; d++) fprintf(stderr, "%s%lu", d ? "," : "", (unsigned long)shape[d]);
        fprintf(stderr, "]\n");
    }
    return dst;
}

/* Load DA2 DINOv2 ViT-L semantic encoder weights */
static int ppd_load_sem(ppd_model *m, const char *path) {
    pth_context *pth = pth_open(path);
    if (!pth) return -1;

    if (m->verbose >= 1)
        fprintf(stderr, "ppd: loading DA2 from %s (%d tensors)\n", path, pth_count(pth));

    /* Detect prefix */
    const char *prefix = "";
    int qkv_idx = pth_find(pth, "blocks.0.attn.qkv.weight");
    if (qkv_idx < 0) {
        qkv_idx = pth_find(pth, "pretrained.blocks.0.attn.qkv.weight");
        if (qkv_idx >= 0) prefix = "pretrained.";
    }
    if (qkv_idx < 0) {
        fprintf(stderr, "ppd: cannot find QKV weight in %s\n", path);
        pth_close(pth);
        return -1;
    }

    const char *qkv_dtype = pth_dtype(pth, qkv_idx);
    if (m->verbose >= 1)
        fprintf(stderr, "ppd: DA2 weight dtype: %s\n", qkv_dtype);
    const uint64_t *qkv_shape = pth_shape(pth, qkv_idx);
    m->sem_dim = (int)qkv_shape[1]; /* [3*dim, dim] */
    m->sem_head_dim = 64;
    m->sem_n_heads = m->sem_dim / 64;
    m->sem_ffn = m->sem_dim * 4;
    m->sem_patch_size = 14;
    m->sem_ln_eps = 1e-6f;

    /* Count blocks */
    m->sem_n_blocks = 0;
    for (int i = 0; i < pth_count(pth); i++) {
        const char *n = pth_name(pth, i);
        const char *bp = strstr(n, "blocks.");
        if (bp) {
            int blk = atoi(bp + 7);
            if (blk + 1 > m->sem_n_blocks) m->sem_n_blocks = blk + 1;
        }
    }

    if (m->verbose >= 1)
        fprintf(stderr, "ppd: DA2 dim=%d heads=%d blocks=%d\n",
                m->sem_dim, m->sem_n_heads, m->sem_n_blocks);

    m->sem_blocks = (ppd_sem_block *)calloc((size_t)m->sem_n_blocks, sizeof(ppd_sem_block));

    char buf[256];
    for (int L = 0; L < m->sem_n_blocks; L++) {
        ppd_sem_block *b = &m->sem_blocks[L];

        snprintf(buf, sizeof(buf), "%sblocks.%d.norm1.weight", prefix, L);
        b->ln1_w = ppd_load_f32(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.norm1.bias", prefix, L);
        b->ln1_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.weight", prefix, L);
        b->qkv_w = ppd_load_f16(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.bias", prefix, L);
        b->qkv_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.weight", prefix, L);
        b->out_w = ppd_load_f16(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.bias", prefix, L);
        b->out_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.norm2.weight", prefix, L);
        b->ln2_w = ppd_load_f32(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.norm2.bias", prefix, L);
        b->ln2_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.weight", prefix, L);
        b->ffn_up_w = ppd_load_f16(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.bias", prefix, L);
        b->ffn_up_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.weight", prefix, L);
        b->ffn_down_w = ppd_load_f16(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.bias", prefix, L);
        b->ffn_down_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.ls1.gamma", prefix, L);
        b->ls1 = ppd_load_f32(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.ls2.gamma", prefix, L);
        b->ls2 = ppd_load_f32(pth, buf, m->verbose);
    }

    /* Patch embed, CLS, pos_embed, final norm */
    snprintf(buf, sizeof(buf), "%spatch_embed.proj.weight", prefix);
    m->sem_pe_w = ppd_load_f32(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%spatch_embed.proj.bias", prefix);
    m->sem_pe_b = ppd_load_f32(pth, buf, m->verbose);

    snprintf(buf, sizeof(buf), "%scls_token", prefix);
    {
        int idx = pth_find(pth, buf);
        if (idx >= 0) {
            /* cls_token shape: [1, 1, dim] → store [dim] */
            m->sem_cls = (float *)malloc((size_t)m->sem_dim * sizeof(float));
            const float *src = (const float *)pth_data(pth, idx);
            memcpy(m->sem_cls, src, (size_t)m->sem_dim * sizeof(float));
        }
    }

    snprintf(buf, sizeof(buf), "%spos_embed", prefix);
    {
        int idx = pth_find(pth, buf);
        if (idx >= 0) {
            const uint64_t *pe_shape = pth_shape(pth, idx);
            int pe_ndims = pth_ndims(pth, idx);
            int pe_n = (pe_ndims == 3) ? (int)pe_shape[1] : (int)pe_shape[0];
            m->sem_pos_n = pe_n;
            int n_patches = pe_n - 1;
            int g = (int)sqrtf((float)n_patches);
            m->sem_pos_gH = g;
            m->sem_pos_gW = g;
            m->sem_pos = (float *)malloc((size_t)pe_n * m->sem_dim * sizeof(float));
            memcpy(m->sem_pos, pth_data(pth, idx), (size_t)pe_n * m->sem_dim * sizeof(float));
            if (m->verbose >= 1)
                fprintf(stderr, "ppd: pos_embed: %d tokens, grid %dx%d\n", pe_n, g, g);
        }
    }

    snprintf(buf, sizeof(buf), "%snorm.weight", prefix);
    m->sem_norm_w = ppd_load_f32(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%snorm.bias", prefix);
    m->sem_norm_b = ppd_load_f32(pth, buf, m->verbose);

    pth_close(pth);
    if (m->verbose >= 1) fprintf(stderr, "ppd: DA2 weights loaded OK\n");
    return 0;
}

/* Load DiT weights */
static int ppd_load_dit(ppd_model *m, const char *path) {
    pth_context *pth = pth_open(path);
    if (!pth) return -1;

    if (m->verbose >= 1)
        fprintf(stderr, "ppd: loading DiT from %s (%d tensors)\n", path, pth_count(pth));

    m->dit_dim = 1024;
    m->dit_n_heads = 16;
    m->dit_head_dim = 64;
    m->dit_ffn = 4096;
    m->dit_n_blocks = 24;
    m->dit_patch_size = 16;
    m->dit_ln_eps = 1e-6f;
    m->dit_rope_freq = 100.0f;

    /* Detect prefix */
    const char *dit_pfx = "";
    if (pth_find(pth, "dit.x_embedder.proj.weight") >= 0)
        dit_pfx = "dit.";
    else if (pth_find(pth, "x_embedder.proj.weight") < 0) {
        fprintf(stderr, "ppd: cannot find x_embedder.proj.weight in %s\n", path);
        pth_close(pth);
        return -1;
    }

    m->dit_blocks = (ppd_dit_block *)calloc((size_t)m->dit_n_blocks, sizeof(ppd_dit_block));

    char buf[256];

    /* x_embedder (F32 conv) */
    snprintf(buf, sizeof(buf), "%sx_embedder.proj.weight", dit_pfx);
    {   int xe_idx = pth_find(pth, buf);
        if (xe_idx >= 0 && m->verbose >= 1)
            fprintf(stderr, "ppd: DiT weight dtype: %s\n", pth_dtype(pth, xe_idx)); }
    m->dit_xe_w = ppd_load_f32(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%sx_embedder.proj.bias", dit_pfx);
    m->dit_xe_b = ppd_load_f32(pth, buf, m->verbose);

    /* t_embedder */
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.0.weight", dit_pfx);
    m->dit_te_w1 = ppd_load_f16(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.0.bias", dit_pfx);
    m->dit_te_b1 = ppd_load_f32(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.2.weight", dit_pfx);
    m->dit_te_w2 = ppd_load_f16(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.2.bias", dit_pfx);
    m->dit_te_b2 = ppd_load_f32(pth, buf, m->verbose);

    /* proj_fusion */
    snprintf(buf, sizeof(buf), "%sproj_fusion.0.weight", dit_pfx);
    m->dit_fus_w[0] = ppd_load_f16(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%sproj_fusion.0.bias", dit_pfx);
    m->dit_fus_b[0] = ppd_load_f32(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%sproj_fusion.2.weight", dit_pfx);
    m->dit_fus_w[1] = ppd_load_f16(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%sproj_fusion.2.bias", dit_pfx);
    m->dit_fus_b[1] = ppd_load_f32(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%sproj_fusion.4.weight", dit_pfx);
    m->dit_fus_w[2] = ppd_load_f16(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%sproj_fusion.4.bias", dit_pfx);
    m->dit_fus_b[2] = ppd_load_f32(pth, buf, m->verbose);

    /* final layer */
    snprintf(buf, sizeof(buf), "%sfinal_layer.adaLN_modulation.1.weight", dit_pfx);
    m->dit_fin_adaln_w = ppd_load_f16(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%sfinal_layer.adaLN_modulation.1.bias", dit_pfx);
    m->dit_fin_adaln_b = ppd_load_f32(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%sfinal_layer.linear.weight", dit_pfx);
    m->dit_fin_proj_w = ppd_load_f16(pth, buf, m->verbose);
    snprintf(buf, sizeof(buf), "%sfinal_layer.linear.bias", dit_pfx);
    m->dit_fin_proj_b = ppd_load_f32(pth, buf, m->verbose);

    /* Per-block DiT weights */
    for (int L = 0; L < m->dit_n_blocks; L++) {
        ppd_dit_block *b = &m->dit_blocks[L];

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.weight", dit_pfx, L);
        b->qkv_w = ppd_load_f16(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.bias", dit_pfx, L);
        b->qkv_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.q_norm.weight", dit_pfx, L);
        b->q_norm_w = ppd_load_f32(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.q_norm.bias", dit_pfx, L);
        b->q_norm_b = ppd_load_f32(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.k_norm.weight", dit_pfx, L);
        b->k_norm_w = ppd_load_f32(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.k_norm.bias", dit_pfx, L);
        b->k_norm_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.weight", dit_pfx, L);
        b->out_w = ppd_load_f16(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.bias", dit_pfx, L);
        b->out_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.weight", dit_pfx, L);
        b->fc1_w = ppd_load_f16(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.bias", dit_pfx, L);
        b->fc1_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.weight", dit_pfx, L);
        b->fc2_w = ppd_load_f16(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.bias", dit_pfx, L);
        b->fc2_b = ppd_load_f32(pth, buf, m->verbose);

        snprintf(buf, sizeof(buf), "%sblocks.%d.adaLN_modulation.1.weight", dit_pfx, L);
        b->adaln_w = ppd_load_f16(pth, buf, m->verbose);
        snprintf(buf, sizeof(buf), "%sblocks.%d.adaLN_modulation.1.bias", dit_pfx, L);
        b->adaln_b = ppd_load_f32(pth, buf, m->verbose);
    }

    pth_close(pth);
    if (m->verbose >= 1) fprintf(stderr, "ppd: DiT weights loaded OK\n");
    return 0;
}

/* ======================================================================== */
/* Section 4: DA2 Semantic Encoder Forward Pass                              */
/* ======================================================================== */

static void ppd_da2_forward(ppd_model *m, float *semantics,
                             const uint8_t *rgb, int w, int h,
                             int proc_h, int proc_w, int n_threads) {
    int dim = m->sem_dim;
    int ps = m->sem_patch_size;
    int sem_gH = proc_h / 16;
    int sem_gW = proc_w / 16;
    int sem_h = sem_gH * ps;
    int sem_w = sem_gW * ps;
    int sem_np = sem_gH * sem_gW;
    int sem_nt = 1 + sem_np;

    /* 1. Resize + ImageNet normalize → [3, sem_h, sem_w] */
    float *img = (float *)malloc((size_t)3 * sem_h * sem_w * sizeof(float));
    {
        float *img_raw = (float *)malloc((size_t)3 * h * w * sizeof(float));
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3]  = {0.229f, 0.224f, 0.225f};
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    img_raw[c * h * w + y * w + x] =
                        (rgb[(y * w + x) * 3 + c] / 255.0f - mean[c]) / std[c];
        ppd_bilinear_resize(img, img_raw, 3, h, w, sem_h, sem_w);
        free(img_raw);
    }

    /* 2. Patch embed → hidden[1+np, dim] (token 0 = CLS placeholder) */
    float *hidden = (float *)calloc((size_t)sem_nt * dim, sizeof(float));
    ppd_patch_embed(hidden, img, m->sem_pe_w, m->sem_pe_b,
                    sem_h, sem_w, 3, dim, ps, sem_gH, sem_gW);
    free(img);

    /* 3. CLS token + position embedding (with interpolation) */
    {
        float *pos = m->sem_pos;
        float *pos_alloc = NULL;

        if (sem_gH != m->sem_pos_gH || sem_gW != m->sem_pos_gW) {
            float *cls_pe = m->sem_pos;
            float *patch_pe = m->sem_pos + dim;
            float *interp = ppd_interpolate_pos_embed(patch_pe, dim,
                                m->sem_pos_gH, m->sem_pos_gW, sem_gH, sem_gW);
            pos_alloc = (float *)malloc((size_t)sem_nt * dim * sizeof(float));
            memcpy(pos_alloc, cls_pe, (size_t)dim * sizeof(float));
            memcpy(pos_alloc + dim, interp, (size_t)sem_np * dim * sizeof(float));
            free(interp);
            pos = pos_alloc;
        }

        /* Token 0 = CLS + pos[0], patch tokens += pos[1..] */
        for (int i = 0; i < dim; i++)
            hidden[i] = m->sem_cls[i] + pos[i];
        for (int t = 1; t < sem_nt; t++)
            for (int i = 0; i < dim; i++)
                hidden[t * dim + i] += pos[t * dim + i];

        free(pos_alloc);
    }

    /* Diagnostic: dump hidden after CLS+pos_embed, before backbone */
    if (m->verbose >= 2) {
        fprintf(stderr, "  hidden_pre[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                hidden[0],hidden[1],hidden[2],hidden[3],hidden[4],hidden[5],hidden[6],hidden[7]);
        fprintf(stderr, "  patch1_pre[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                hidden[dim],hidden[dim+1],hidden[dim+2],hidden[dim+3],
                hidden[dim+4],hidden[dim+5],hidden[dim+6],hidden[dim+7]);
    }

    /* 4. Transformer blocks */
    float *ln_buf = (float *)malloc((size_t)sem_nt * dim * sizeof(float));
    float *qkv    = (float *)malloc((size_t)sem_nt * 3 * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)sem_nt * dim * sizeof(float));
    float *ffn_buf  = (float *)malloc((size_t)sem_nt * m->sem_ffn * sizeof(float));
    float *proj_out = (float *)malloc((size_t)sem_nt * dim * sizeof(float));

    for (int L = 0; L < m->sem_n_blocks; L++) {
        ppd_sem_block *b = &m->sem_blocks[L];

        /* LN1 → QKV → Attention → OutProj → LayerScale + residual */
        ppd_layernorm(ln_buf, hidden, b->ln1_w, b->ln1_b, sem_nt, dim, m->sem_ln_eps);
        ppd_gemm(qkv, b->qkv_w, b->qkv_b, ln_buf, sem_nt, 3 * dim, dim, n_threads);
        ppd_attention(attn_out, qkv, sem_nt, dim, m->sem_n_heads, m->sem_head_dim, n_threads);
        ppd_gemm(proj_out, b->out_w, b->out_b, attn_out, sem_nt, dim, dim, n_threads);

        if (b->ls1) {
            ppd_gate_residual(hidden, proj_out, b->ls1, sem_nt, dim);
        } else {
            for (int i = 0; i < sem_nt * dim; i++) hidden[i] += proj_out[i];
        }

        /* LN2 → FFN up → GELU → FFN down → LayerScale + residual */
        ppd_layernorm(ln_buf, hidden, b->ln2_w, b->ln2_b, sem_nt, dim, m->sem_ln_eps);
        ppd_gemm(ffn_buf, b->ffn_up_w, b->ffn_up_b, ln_buf, sem_nt, m->sem_ffn, dim, n_threads);
        ppd_gelu(ffn_buf, sem_nt * m->sem_ffn);
        ppd_gemm(proj_out, b->ffn_down_w, b->ffn_down_b, ffn_buf, sem_nt, dim, m->sem_ffn, n_threads);

        if (b->ls2) {
            ppd_gate_residual(hidden, proj_out, b->ls2, sem_nt, dim);
        } else {
            for (int i = 0; i < sem_nt * dim; i++) hidden[i] += proj_out[i];
        }

        /* Block-level diagnostic */
        if (m->verbose >= 2 && L == 0) {
            fprintf(stderr, "  blk0_out[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                    hidden[0],hidden[1],hidden[2],hidden[3],hidden[4],hidden[5],hidden[6],hidden[7]);
            fprintf(stderr, "  blk0_qkv[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    qkv[0],qkv[1],qkv[2],qkv[3],qkv[4],qkv[5],qkv[6],qkv[7]);
            fprintf(stderr, "  blk0_attn[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    attn_out[0],attn_out[1],attn_out[2],attn_out[3],
                    attn_out[4],attn_out[5],attn_out[6],attn_out[7]);
        }

        if (m->verbose >= 2 && L % 6 == 5)
            fprintf(stderr, "ppd: DA2 block %d/%d done\n", L + 1, m->sem_n_blocks);
    }

    /* 5. Final LayerNorm */
    ppd_layernorm(hidden, hidden, m->sem_norm_w, m->sem_norm_b, sem_nt, dim, m->sem_ln_eps);

    /* 6. Extract patch tokens (skip CLS) */
    memcpy(semantics, hidden + dim, (size_t)sem_np * dim * sizeof(float));

    free(hidden); free(ln_buf); free(qkv);
    free(attn_out); free(ffn_buf); free(proj_out);
}

/* ======================================================================== */
/* Section 5: DiT Diffusion Transformer Forward Pass                         */
/* ======================================================================== */

/* Single DiT block forward */
static void ppd_dit_block_forward(ppd_model *m, ppd_dit_block *b,
                                    float *hidden, float *ln_buf, float *qkv,
                                    float *attn_out, float *ffn_buf, float *proj_out,
                                    const float *t_embed_silu,
                                    const int *pos_y, const int *pos_x,
                                    int nt, int n_threads) {
    int dim = m->dit_dim;
    int stride_3dim = 3 * dim;

    /* adaLN modulation: Linear(SiLU(t_embed)) → [1, 6*dim] */
    float *modulation = (float *)malloc((size_t)6 * dim * sizeof(float));
    gemm_f16_f32_tokmajor(modulation, b->adaln_w, t_embed_silu,
                           6 * dim, dim, 1, 6 * dim, dim);
    for (int i = 0; i < 6 * dim; i++) modulation[i] += b->adaln_b[i];

    float *shift_msa = modulation;
    float *scale_msa = modulation + dim;
    float *gate_msa  = modulation + 2 * dim;
    float *shift_mlp = modulation + 3 * dim;
    float *scale_mlp = modulation + 4 * dim;
    float *gate_mlp  = modulation + 5 * dim;

    /* adaLN + norm1 */
    ppd_adaln_modulate(ln_buf, hidden, shift_msa, scale_msa, nt, dim, m->dit_ln_eps);

    /* QKV projection */
    ppd_gemm(qkv, b->qkv_w, b->qkv_b, ln_buf, nt, 3 * dim, dim, n_threads);

    /* QK-norm */
    ppd_qk_norm(qkv, b->q_norm_w, b->q_norm_b,
                nt, m->dit_n_heads, m->dit_head_dim, stride_3dim, m->dit_ln_eps);
    ppd_qk_norm(qkv + dim, b->k_norm_w, b->k_norm_b,
                nt, m->dit_n_heads, m->dit_head_dim, stride_3dim, m->dit_ln_eps);

    /* 2D RoPE on Q and K */
    ppd_rope_2d(qkv, nt, m->dit_n_heads, m->dit_head_dim,
                stride_3dim, pos_y, pos_x, m->dit_rope_freq);
    ppd_rope_2d(qkv + dim, nt, m->dit_n_heads, m->dit_head_dim,
                stride_3dim, pos_y, pos_x, m->dit_rope_freq);

    /* Attention */
    ppd_attention(attn_out, qkv, nt, dim, m->dit_n_heads, m->dit_head_dim, n_threads);

    /* Output projection + gate residual */
    ppd_gemm(proj_out, b->out_w, b->out_b, attn_out, nt, dim, dim, n_threads);
    ppd_gate_residual(hidden, proj_out, gate_msa, nt, dim);

    /* adaLN + norm2 */
    ppd_adaln_modulate(ln_buf, hidden, shift_mlp, scale_mlp, nt, dim, m->dit_ln_eps);

    /* MLP: fc1 → GELU → fc2 */
    ppd_gemm(ffn_buf, b->fc1_w, b->fc1_b, ln_buf, nt, m->dit_ffn, dim, n_threads);
    ppd_gelu(ffn_buf, nt * m->dit_ffn);
    ppd_gemm(proj_out, b->fc2_w, b->fc2_b, ffn_buf, nt, dim, m->dit_ffn, n_threads);
    ppd_gate_residual(hidden, proj_out, gate_mlp, nt, dim);

    free(modulation);
}

/* Single DiT denoising step (full forward pass) */
static void ppd_dit_step(ppd_model *m, float *pred, float *hidden,
                           const float *latent, const float *cond,
                           const float *semantics,
                           int proc_h, int proc_w, float t_cur, int n_threads) {
    int dim = m->dit_dim;
    int HW = proc_h * proc_w;
    int dit_gH_lo = proc_h / 16, dit_gW_lo = proc_w / 16;
    int dit_nt_lo = dit_gH_lo * dit_gW_lo;
    int dit_gH_hi = proc_h / 8, dit_gW_hi = proc_w / 8;
    int dit_nt_hi = dit_gH_hi * dit_gW_hi;
    int max_nt = dit_nt_hi;

    /* 1. Concat [latent, cond] → [4, proc_h, proc_w] */
    float *dit_input = (float *)malloc((size_t)4 * HW * sizeof(float));
    memcpy(dit_input, latent, (size_t)HW * sizeof(float));
    memcpy(dit_input + HW, cond, (size_t)3 * HW * sizeof(float));

    /* 2. Patch embed Conv2d(4, dim, k=16, s=16) → [dit_nt_lo, dim] */
    ppd_dit_patch_embed(hidden, dit_input, m->dit_xe_w, m->dit_xe_b,
                        proc_h, proc_w, 4, dim, 16, dit_gH_lo, dit_gW_lo);
    free(dit_input);

    /* 3. Timestep embedding */
    float sin_embed[256];
    ppd_sinusoidal_embed(sin_embed, t_cur, 256);
    float *t_embed = (float *)malloc((size_t)dim * sizeof(float));
    gemm_f16_f32_tokmajor(t_embed, m->dit_te_w1, sin_embed, dim, 256, 1, dim, 256);
    for (int i = 0; i < dim; i++) t_embed[i] += m->dit_te_b1[i];
    ppd_silu(t_embed, dim);
    float *t_embed2 = (float *)malloc((size_t)dim * sizeof(float));
    gemm_f16_f32_tokmajor(t_embed2, m->dit_te_w2, t_embed, dim, dim, 1, dim, dim);
    for (int i = 0; i < dim; i++) t_embed2[i] += m->dit_te_b2[i];
    /* t_embed2 is the raw timestep embedding, SiLU(t_embed2) used for adaLN */
    float *t_embed_silu = (float *)malloc((size_t)dim * sizeof(float));
    memcpy(t_embed_silu, t_embed2, (size_t)dim * sizeof(float));
    ppd_silu(t_embed_silu, dim);
    free(t_embed); free(t_embed2);

    /* Scratch buffers */
    float *ln_buf   = (float *)malloc((size_t)max_nt * dim * sizeof(float));
    float *qkv      = (float *)malloc((size_t)max_nt * 3 * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)max_nt * dim * sizeof(float));
    float *ffn_buf  = (float *)malloc((size_t)max_nt * m->dit_ffn * sizeof(float));
    float *proj_out = (float *)malloc((size_t)max_nt * dim * sizeof(float));

    /* 4. Low-res grid positions */
    int *pos_y_lo = (int *)malloc((size_t)dit_nt_lo * sizeof(int));
    int *pos_x_lo = (int *)malloc((size_t)dit_nt_lo * sizeof(int));
    ppd_generate_grid_pos(pos_y_lo, pos_x_lo, dit_gH_lo, dit_gW_lo);

    /* 5. Blocks 0-11 (low-res) */
    for (int L = 0; L < 12; L++) {
        ppd_dit_block_forward(m, &m->dit_blocks[L], hidden, ln_buf, qkv,
                              attn_out, ffn_buf, proj_out, t_embed_silu,
                              pos_y_lo, pos_x_lo, dit_nt_lo, n_threads);
    }
    free(pos_y_lo); free(pos_x_lo);

    /* 6. Semantic fusion */
    {
        /* Concat [hidden, semantics] → [nt_lo, 2*dim] */
        float *concat = (float *)malloc((size_t)dit_nt_lo * 2 * dim * sizeof(float));
        ppd_concat_dim(concat, hidden, semantics, dit_nt_lo, dim, dim);

        /* 3-layer MLP with SiLU — use two 4*dim buffers for ping-pong */
        float *fus_a = (float *)malloc((size_t)dit_nt_lo * 4 * dim * sizeof(float));
        float *fus_b = (float *)malloc((size_t)dit_nt_lo * 4 * dim * sizeof(float));
        /* Layer 0: Linear(2*dim, 4*dim) + SiLU */
        ppd_gemm(fus_a, m->dit_fus_w[0], m->dit_fus_b[0], concat, dit_nt_lo, 4 * dim, 2 * dim, n_threads);
        free(concat);
        ppd_silu(fus_a, dit_nt_lo * 4 * dim);
        /* Layer 1: Linear(4*dim, 4*dim) + SiLU */
        ppd_gemm(fus_b, m->dit_fus_w[1], m->dit_fus_b[1], fus_a, dit_nt_lo, 4 * dim, 4 * dim, n_threads);
        ppd_silu(fus_b, dit_nt_lo * 4 * dim);
        /* Layer 2: Linear(4*dim, 4*dim) */
        ppd_gemm(fus_a, m->dit_fus_w[2], m->dit_fus_b[2], fus_b, dit_nt_lo, 4 * dim, 4 * dim, n_threads);
        free(fus_b);

        /* Pixel shuffle 2×: [nt_lo, 4*dim] → [nt_hi, dim] */
        ppd_pixel_shuffle_2x(hidden, fus_a, dit_gH_lo, dit_gW_lo, dim);
        free(fus_a);
    }

    /* 7. High-res grid positions */
    int *pos_y_hi = (int *)malloc((size_t)dit_nt_hi * sizeof(int));
    int *pos_x_hi = (int *)malloc((size_t)dit_nt_hi * sizeof(int));
    ppd_generate_grid_pos(pos_y_hi, pos_x_hi, dit_gH_hi, dit_gW_hi);

    /* 8. Blocks 12-23 (high-res) */
    for (int L = 12; L < 24; L++) {
        ppd_dit_block_forward(m, &m->dit_blocks[L], hidden, ln_buf, qkv,
                              attn_out, ffn_buf, proj_out, t_embed_silu,
                              pos_y_hi, pos_x_hi, dit_nt_hi, n_threads);
    }
    free(pos_y_hi); free(pos_x_hi);

    /* 9. Final layer */
    {
        int out_ps = 8;
        int out_dim = out_ps * out_ps; /* 64 */

        /* Final adaLN: modulation → [1, 2*dim] → shift, scale */
        float *final_mod = (float *)malloc((size_t)2 * dim * sizeof(float));
        gemm_f16_f32_tokmajor(final_mod, m->dit_fin_adaln_w, t_embed_silu,
                               2 * dim, dim, 1, 2 * dim, dim);
        for (int i = 0; i < 2 * dim; i++) final_mod[i] += m->dit_fin_adaln_b[i];

        float *final_shift = final_mod;
        float *final_scale = final_mod + dim;

        ppd_adaln_modulate(ln_buf, hidden, final_shift, final_scale, dit_nt_hi, dim, m->dit_ln_eps);
        free(final_mod);

        /* Linear(dim, 64) → [nt_hi, 64] */
        ppd_gemm(proj_out, m->dit_fin_proj_w, m->dit_fin_proj_b,
                 ln_buf, dit_nt_hi, out_dim, dim, n_threads);

        /* Unpatchify: [nt_hi, 64] → [proc_h, proc_w] */
        ppd_unpatchify(pred, proj_out, dit_gH_hi, dit_gW_hi, out_ps);
    }
    free(t_embed_silu); free(ln_buf); free(qkv);
    free(attn_out); free(ffn_buf); free(proj_out);
}

/* ======================================================================== */
/* Section 6: Predict Pipeline                                               */
/* ======================================================================== */

ppd_result ppd_predict(ppd_model *m, const uint8_t *rgb, int w, int h, int n_threads) {
    ppd_result res = {0};
    if (!m) return res;

    double t0 = ppd_time_ms();

    /* 1. Processing resolution (round up to 16) */
    int proc_h = ((h + 15) / 16) * 16;
    int proc_w = ((w + 15) / 16) * 16;
    int HW = proc_h * proc_w;

    int sem_gH = proc_h / 16, sem_gW = proc_w / 16;
    int sem_np = sem_gH * sem_gW;

    if (m->verbose >= 1) {
        fprintf(stderr, "ppd: predict %dx%d → proc %dx%d, threads=%d\n",
                w, h, proc_w, proc_h, n_threads);
    }

    /* 2. DA2 semantic encoder → [sem_np, dim] */
    float *semantics = (float *)malloc((size_t)sem_np * m->sem_dim * sizeof(float));
    ppd_da2_forward(m, semantics, rgb, w, h, proc_h, proc_w, n_threads);

    double t1 = ppd_time_ms();
    if (m->verbose >= 1)
        fprintf(stderr, "ppd: DA2 encoder: %.1f ms (%d patches)\n", t1 - t0, sem_np);

    /* Quick diagnostic: dump semantics stats */
    if (m->verbose >= 2)
        fprintf(stderr, "  sem[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                semantics[0],semantics[1],semantics[2],semantics[3],
                semantics[4],semantics[5],semantics[6],semantics[7]);

    /* 3. Condition image: resize to [3, proc_h, proc_w], pixel/255 - 0.5 */
    float *cond = (float *)malloc((size_t)3 * HW * sizeof(float));
    {
        float *img_f = (float *)malloc((size_t)3 * h * w * sizeof(float));
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    img_f[c * h * w + y * w + x] = rgb[(y * w + x) * 3 + c] / 255.0f - 0.5f;
        ppd_bilinear_resize(cond, img_f, 3, h, w, proc_h, proc_w);
        free(img_f);
    }

    /* 4. Initialize latent ~ N(0, 1) */
    float *latent = (float *)malloc((size_t)HW * sizeof(float));
    srand(42);
    ppd_generate_randn(latent, HW);

    /* 5. DiT diffusion: 4 Euler steps */
    double t2 = ppd_time_ms();
    float *dit_pred = (float *)malloc((size_t)HW * sizeof(float));
    int dit_nt_hi = (proc_h / 8) * (proc_w / 8);
    float *hidden = (float *)malloc((size_t)dit_nt_hi * m->dit_dim * sizeof(float));

    float timesteps[] = {1000.0f, 750.0f, 500.0f, 250.0f};
    float T = 1000.0f;

    for (int step = 0; step < 4; step++) {
        float t_cur = timesteps[step];
        float t_next = (step < 3) ? timesteps[step + 1] : 0.0f;
        float t_ratio = t_cur / T;
        float s_ratio = t_next / T;

        if (m->verbose >= 2)
            fprintf(stderr, "ppd: Euler step %d: t=%.0f → s=%.0f\n", step, t_cur, t_next);

        ppd_dit_step(m, dit_pred, hidden, latent, cond, semantics,
                     proc_h, proc_w, t_cur, n_threads);

        /* Euler step: update latent */
        for (int i = 0; i < HW; i++) {
            float xt = latent[i];
            float v = dit_pred[i];
            float pred_x0 = xt - t_ratio * v;
            float pred_xT = xt + (1.0f - t_ratio) * v;
            latent[i] = (1.0f - s_ratio) * pred_x0 + s_ratio * pred_xT;
        }

        /* Diagnostic: dump velocity and latent after first step */
        if (step == 0 && m->verbose >= 2)
            fprintf(stderr, "  step0 vel[0..3]: %.4f %.4f %.4f %.4f\n  step0 lat[0..3]: %.4f %.4f %.4f %.4f\n",
                    dit_pred[0],dit_pred[1],dit_pred[2],dit_pred[3],
                    latent[0],latent[1],latent[2],latent[3]);
    }

    double t3 = ppd_time_ms();
    if (m->verbose >= 1)
        fprintf(stderr, "ppd: DiT diffusion (4 steps): %.1f ms\n", t3 - t2);

    free(hidden); free(dit_pred); free(cond); free(semantics);

    /* 6. Depth = latent + 0.5 */
    for (int i = 0; i < HW; i++) latent[i] += 0.5f;

    /* 7. Resize to original resolution */
    if (proc_h != h || proc_w != w) {
        float *depth = (float *)malloc((size_t)w * h * sizeof(float));
        ppd_bilinear_resize(depth, latent, 1, proc_h, proc_w, h, w);
        free(latent);
        res.depth = depth;
    } else {
        res.depth = latent;
    }
    res.width = w;
    res.height = h;

    if (m->verbose >= 1)
        fprintf(stderr, "ppd: predict total: %.1f ms\n", ppd_time_ms() - t0);

    return res;
}

/* ======================================================================== */
/* Section 7: Load + Free                                                    */
/* ======================================================================== */

ppd_model *ppd_load(const char *ppd_path, const char *sem_path, int verbose) {
    ppd_model *m = (ppd_model *)calloc(1, sizeof(ppd_model));
    m->verbose = verbose;

    if (ppd_load_sem(m, sem_path) != 0) {
        fprintf(stderr, "ppd: failed to load DA2 semantic encoder\n");
        ppd_free(m);
        return NULL;
    }

    if (ppd_load_dit(m, ppd_path) != 0) {
        fprintf(stderr, "ppd: failed to load DiT weights\n");
        ppd_free(m);
        return NULL;
    }

    if (verbose >= 1)
        fprintf(stderr, "ppd: model loaded (DA2: %d blocks, DiT: %d blocks)\n",
                m->sem_n_blocks, m->dit_n_blocks);
    return m;
}

void ppd_free(ppd_model *m) {
    if (!m) return;

    /* Free DA2 blocks */
    if (m->sem_blocks) {
        for (int i = 0; i < m->sem_n_blocks; i++) {
            ppd_sem_block *b = &m->sem_blocks[i];
            free(b->ln1_w); free(b->ln1_b);
            free(b->ln2_w); free(b->ln2_b);
            free(b->qkv_w); free(b->qkv_b);
            free(b->out_w); free(b->out_b);
            free(b->ffn_up_w); free(b->ffn_up_b);
            free(b->ffn_down_w); free(b->ffn_down_b);
            free(b->ls1); free(b->ls2);
        }
        free(m->sem_blocks);
    }
    free(m->sem_pe_w); free(m->sem_pe_b);
    free(m->sem_cls); free(m->sem_pos);
    free(m->sem_norm_w); free(m->sem_norm_b);

    /* Free DiT blocks */
    if (m->dit_blocks) {
        for (int i = 0; i < m->dit_n_blocks; i++) {
            ppd_dit_block *b = &m->dit_blocks[i];
            free(b->qkv_w); free(b->qkv_b);
            free(b->q_norm_w); free(b->q_norm_b);
            free(b->k_norm_w); free(b->k_norm_b);
            free(b->out_w); free(b->out_b);
            free(b->fc1_w); free(b->fc1_b);
            free(b->fc2_w); free(b->fc2_b);
            free(b->adaln_w); free(b->adaln_b);
        }
        free(m->dit_blocks);
    }
    free(m->dit_xe_w); free(m->dit_xe_b);
    free(m->dit_te_w1); free(m->dit_te_b1);
    free(m->dit_te_w2); free(m->dit_te_b2);
    for (int i = 0; i < 3; i++) {
        free(m->dit_fus_w[i]); free(m->dit_fus_b[i]);
    }
    free(m->dit_fin_adaln_w); free(m->dit_fin_adaln_b);
    free(m->dit_fin_proj_w); free(m->dit_fin_proj_b);

    free(m);
}

void ppd_result_free(ppd_result *r) {
    if (r) { free(r->depth); r->depth = NULL; }
}

#endif /* PIXEL_PERFECT_DEPTH_IMPLEMENTATION */
#endif /* PIXEL_PERFECT_DEPTH_H */
