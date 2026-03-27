/*
 * qwen_image_vae.h - Qwen-Image 3D Causal VAE (decoder only for text-to-image)
 *
 * Usage:
 *   #define QIMG_VAE_IMPLEMENTATION
 *   #include "qwen_image_vae.h"
 *
 * Dependencies: safetensors.h, ggml_dequant.h (for BF16)
 *
 * Architecture: 3D Causal VAE with temporal dimension
 *   z_dim=16, base_dim=96, dim_mult=[1,2,4,4], num_res_blocks=2
 *   For image-only mode: temporal dim = 1 throughout
 *
 * API:
 *   qimg_vae_model *qimg_vae_load(const char *path);
 *   void            qimg_vae_free(qimg_vae_model *m);
 *   void            qimg_vae_decode(float *rgb_out, const float *latent,
 *                                   int lat_h, int lat_w, qimg_vae_model *m);
 */
#ifndef QIMG_VAE_H
#define QIMG_VAE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 2D feature map buffer (channels-first: [C, H, W]) */
typedef struct {
    float *data;
    int c, h, w;
} qimg_fmap;

/* ResBlock weights */
typedef struct {
    float *norm1_gamma;     /* [c_in] */
    float *conv1_weight;    /* [c_out, c_in, 3, 3] (2D, summed from 3D) */
    float *conv1_bias;      /* [c_out] */
    float *norm2_gamma;     /* [c_out] */
    float *conv2_weight;    /* [c_out, c_out, 3, 3] */
    float *conv2_bias;      /* [c_out] */
    float *shortcut_weight; /* [c_out, c_in, 1, 1] or NULL */
    float *shortcut_bias;   /* [c_out] or NULL */
    int c_in, c_out;
} qimg_vae_resblock;

/* Middle-block attention weights */
typedef struct {
    float *norm_gamma;     /* [c] */
    float *qkv_weight;    /* [3*c, c] (from 1x1 conv) */
    float *qkv_bias;      /* [3*c] */
    float *proj_weight;   /* [c, c] */
    float *proj_bias;     /* [c] */
    int c;
} qimg_vae_attn;

/* Spatial upsample (nearest-neighbor 2× + Conv2d) */
typedef struct {
    float *conv_weight;    /* [c_out, c_in, 3, 3] */
    float *conv_bias;      /* [c_out] */
    int c_in, c_out;
} qimg_vae_upsample;

typedef struct {
    int z_dim;        /* 16 */
    int base_dim;     /* 96 */

    /* post_quant_conv */
    float *pqc_weight;   /* [16, 16] (from 1x1x1) */
    float *pqc_bias;     /* [16] */

    /* decoder.conv1 */
    float *dec_conv1_weight;  /* [384, 16, 3, 3] (summed from 3D) */
    float *dec_conv1_bias;    /* [384] */

    /* decoder.middle: resblock, attention, resblock */
    qimg_vae_resblock mid_res0, mid_res2;
    qimg_vae_attn mid_attn;

    /* decoder.upsamples: 15 blocks */
    int n_up_blocks;
    qimg_vae_resblock *up_res;      /* [n_up_blocks] */
    qimg_vae_upsample *up_sample;   /* [n_up_blocks], some are {NULL} */
    int *up_has_sample;              /* [n_up_blocks], which blocks have resample */

    /* decoder.head */
    float *head_norm_gamma;  /* [96] */
    float *head_conv_weight; /* [3, 96, 3, 3] */
    float *head_conv_bias;   /* [3] */

    void *st_ctx;  /* safetensors context */
} qimg_vae_model;

qimg_vae_model *qimg_vae_load(const char *path);
void            qimg_vae_free(qimg_vae_model *m);
void            qimg_vae_decode(float *rgb_out, const float *latent,
                                int lat_h, int lat_w, qimg_vae_model *m);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef QIMG_VAE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "safetensors.h"

/* ---- BF16 conversion ---- */

static float qimg_bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

/* Convert BF16 buffer to F32. Caller must free(). */
static float *qimg_vae_bf16_to_f32(const void *src, size_t n) {
    float *dst = (float *)malloc(n * sizeof(float));
    const uint16_t *s = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++)
        dst[i] = qimg_bf16_to_f32(s[i]);
    return dst;
}

/* Load a safetensors tensor and convert BF16→F32. Returns NULL on missing. */
static float *qimg_vae_load_tensor(const st_context *st, const char *name,
                                   size_t *out_numel) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (out_numel) *out_numel = 0;
        return NULL;
    }
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t numel = 1;
    for (int d = 0; d < ndims; d++) numel *= shape[d];
    if (out_numel) *out_numel = numel;

    const char *dtype = safetensors_dtype(st, idx);
    void *data = safetensors_data(st, idx);
    if (strcmp(dtype, "BF16") == 0) {
        return qimg_vae_bf16_to_f32(data, numel);
    } else if (strcmp(dtype, "F32") == 0) {
        float *buf = (float *)malloc(numel * sizeof(float));
        memcpy(buf, data, numel * sizeof(float));
        return buf;
    } else if (strcmp(dtype, "F16") == 0) {
        /* F16 to F32 */
        float *buf = (float *)malloc(numel * sizeof(float));
        const uint16_t *s = (const uint16_t *)data;
        for (size_t i = 0; i < numel; i++) {
            uint32_t bits = (uint32_t)s[i];
            uint32_t sign = (bits >> 15) & 1;
            uint32_t exp  = (bits >> 10) & 0x1F;
            uint32_t mant = bits & 0x3FF;
            uint32_t f32;
            if (exp == 0) {
                f32 = sign << 31;
            } else if (exp == 31) {
                f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
            } else {
                f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            }
            memcpy(&buf[i], &f32, sizeof(float));
        }
        return buf;
    }
    return NULL;
}

/* Sum 3D conv kernel [Co, Ci, kD, kH, kW] → 2D [Co, Ci, kH, kW]
 * For T=1 with replicate causal padding. */
static float *qimg_vae_conv3d_to_2d(const float *w3d, int co, int ci,
                                     int kd, int kh, int kw) {
    size_t n2d = (size_t)co * ci * kh * kw;
    float *w2d = (float *)calloc(n2d, sizeof(float));
    for (int o = 0; o < co; o++)
        for (int i = 0; i < ci; i++)
            for (int h = 0; h < kh; h++)
                for (int w = 0; w < kw; w++) {
                    float sum = 0;
                    for (int d = 0; d < kd; d++)
                        sum += w3d[((((size_t)o * ci + i) * kd + d) * kh + h) * kw + w];
                    w2d[(((size_t)o * ci + i) * kh + h) * kw + w] = sum;
                }
    return w2d;
}

/* Load a 3D conv weight and convert to 2D. */
static float *qimg_vae_load_conv3d_as_2d(const st_context *st, const char *name,
                                          int *co, int *ci, int *kh, int *kw) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    const uint64_t *shape = safetensors_shape(st, idx);
    *co = (int)shape[0]; *ci = (int)shape[1];
    int kd = (int)shape[2]; *kh = (int)shape[3]; *kw = (int)shape[4];
    size_t numel = (size_t)(*co) * (*ci) * kd * (*kh) * (*kw);
    float *w3d = qimg_vae_bf16_to_f32(safetensors_data(st, idx), numel);
    float *w2d = qimg_vae_conv3d_to_2d(w3d, *co, *ci, kd, *kh, *kw);
    free(w3d);
    return w2d;
}

/* ---- Compute primitives ---- */

/* GroupNorm: scale-only (no bias), 32 groups */
static void qimg_vae_groupnorm(float *out, const float *x, const float *gamma,
                                int c, int h, int w) {
    int groups = 32;
    if (c < groups) groups = c;
    int cpg = c / groups;  /* channels per group */
    int spatial = h * w;

    for (int g = 0; g < groups; g++) {
        /* Compute mean and variance over spatial + channels in group */
        int n = cpg * spatial;
        float mean = 0;
        for (int gc = 0; gc < cpg; gc++) {
            int ch = g * cpg + gc;
            const float *src = x + (size_t)ch * spatial;
            for (int s = 0; s < spatial; s++) mean += src[s];
        }
        mean /= (float)n;

        float var = 0;
        for (int gc = 0; gc < cpg; gc++) {
            int ch = g * cpg + gc;
            const float *src = x + (size_t)ch * spatial;
            for (int s = 0; s < spatial; s++) {
                float d = src[s] - mean;
                var += d * d;
            }
        }
        var /= (float)n;
        float inv = 1.0f / sqrtf(var + 1e-6f);

        for (int gc = 0; gc < cpg; gc++) {
            int ch = g * cpg + gc;
            const float *src = x + (size_t)ch * spatial;
            float *dst = out + (size_t)ch * spatial;
            float g_val = gamma ? gamma[ch] : 1.0f;
            for (int s = 0; s < spatial; s++)
                dst[s] = (src[s] - mean) * inv * g_val;
        }
    }
}

/* SiLU activation */
static void qimg_vae_silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* Conv2D: input [ci, h, w] → output [co, oh, ow], kernel [co, ci, kh, kw]
 * padding = (kh-1)/2, pad_replicate: 0=zero padding, 1=replicate padding */
static void qimg_vae_conv2d_pad(float *out, const float *inp, const float *weight,
                                const float *bias, int ci, int h, int w,
                                int co, int kh, int kw, int stride,
                                int pad_replicate) {
    int pad_h = (kh - 1) / 2;
    int pad_w = (kw - 1) / 2;
    int oh = (h + 2 * pad_h - kh) / stride + 1;
    int ow = (w + 2 * pad_w - kw) / stride + 1;

    for (int oc = 0; oc < co; oc++) {
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < ci; ic++) {
                    for (int fy = 0; fy < kh; fy++) {
                        for (int fx = 0; fx < kw; fx++) {
                            int iy = oy * stride + fy - pad_h;
                            int ix = ox * stride + fx - pad_w;
                            if (pad_replicate) {
                                if (iy < 0) iy = 0;
                                if (iy >= h) iy = h - 1;
                                if (ix < 0) ix = 0;
                                if (ix >= w) ix = w - 1;
                            } else {
                                if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                                    continue;
                            }
                            float v = inp[(size_t)ic * h * w + iy * w + ix];
                            float wt = weight[((((size_t)oc * ci + ic) * kh) + fy) * kw + fx];
                            sum += v * wt;
                        }
                    }
                }
                out[(size_t)oc * oh * ow + oy * ow + ox] = sum;
            }
        }
    }
}

/* Convenience: replicate-padded conv2d (for 3D conv converted to 2D) */
static void qimg_vae_conv2d(float *out, const float *inp, const float *weight,
                             const float *bias, int ci, int h, int w,
                             int co, int kh, int kw, int stride) {
    qimg_vae_conv2d_pad(out, inp, weight, bias, ci, h, w, co, kh, kw, stride, 1);
}

/* Zero-padded conv2d (for 2D resample convolutions) */
static void qimg_vae_conv2d_zero(float *out, const float *inp, const float *weight,
                                  const float *bias, int ci, int h, int w,
                                  int co, int kh, int kw, int stride) {
    qimg_vae_conv2d_pad(out, inp, weight, bias, ci, h, w, co, kh, kw, stride, 0);
}

/* Nearest-neighbor 2× upsample: [c, h, w] → [c, 2h, 2w] */
static float *qimg_vae_nn_upsample(const float *inp, int c, int h, int w) {
    int oh = h * 2, ow = w * 2;
    float *out = (float *)malloc((size_t)c * oh * ow * sizeof(float));
    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                out[(size_t)ch * oh * ow + y * ow + x] =
                    inp[(size_t)ch * h * w + (y / 2) * w + (x / 2)];
            }
        }
    }
    return out;
}

/* Self-attention at spatial positions (channels as features) */
static void qimg_vae_spatial_attn(float *out, const float *inp,
                                   const float *norm_gamma,
                                   const float *qkv_w, const float *qkv_b,
                                   const float *proj_w, const float *proj_b,
                                   int c, int h, int w) {
    int spatial = h * w;

    /* GroupNorm */
    float *normed = (float *)malloc((size_t)c * spatial * sizeof(float));
    qimg_vae_groupnorm(normed, inp, norm_gamma, c, h, w);

    /* QKV projection (1x1 conv = per-pixel linear) */
    float *qkv = (float *)malloc((size_t)3 * c * spatial * sizeof(float));
    for (int s = 0; s < spatial; s++) {
        for (int o = 0; o < 3 * c; o++) {
            float sum = qkv_b ? qkv_b[o] : 0.0f;
            for (int i = 0; i < c; i++)
                sum += qkv_w[(size_t)o * c + i] * normed[(size_t)i * spatial + s];
            qkv[(size_t)o * spatial + s] = sum;
        }
    }
    free(normed);

    /* Split Q, K, V: each [c, spatial] */
    float *Q = qkv;
    float *K = qkv + (size_t)c * spatial;
    float *V = qkv + (size_t)2 * c * spatial;

    /* Attention: softmax(Q^T K / sqrt(c)) V^T → [c, spatial] */
    float scale = 1.0f / sqrtf((float)c);
    float *attn_out = (float *)malloc((size_t)c * spatial * sizeof(float));
    memset(attn_out, 0, (size_t)c * spatial * sizeof(float));

    for (int i = 0; i < spatial; i++) {
        /* Compute attention weights for position i */
        float max_s = -1e30f;
        for (int j = 0; j < spatial; j++) {
            float s = 0;
            for (int d = 0; d < c; d++)
                s += Q[(size_t)d * spatial + i] * K[(size_t)d * spatial + j];
            s *= scale;
            if (s > max_s) max_s = s;
        }
        float exp_sum = 0;
        for (int j = 0; j < spatial; j++) {
            float s = 0;
            for (int d = 0; d < c; d++)
                s += Q[(size_t)d * spatial + i] * K[(size_t)d * spatial + j];
            s *= scale;
            float w = expf(s - max_s);
            exp_sum += w;
            for (int d = 0; d < c; d++)
                attn_out[(size_t)d * spatial + i] += w * V[(size_t)d * spatial + j];
        }
        float inv = 1.0f / exp_sum;
        for (int d = 0; d < c; d++)
            attn_out[(size_t)d * spatial + i] *= inv;
    }
    free(qkv);

    /* Output projection (1x1 conv) + residual */
    for (int s = 0; s < spatial; s++) {
        for (int o = 0; o < c; o++) {
            float sum = proj_b ? proj_b[o] : 0.0f;
            for (int i = 0; i < c; i++)
                sum += proj_w[(size_t)o * c + i] * attn_out[(size_t)i * spatial + s];
            out[(size_t)o * spatial + s] = inp[(size_t)o * spatial + s] + sum;
        }
    }
    free(attn_out);
}

/* ResBlock forward: GroupNorm→SiLU→Conv→GroupNorm→SiLU→Conv + shortcut */
static void qimg_vae_resblock_forward(float *out, const float *inp,
                                       const qimg_vae_resblock *blk,
                                       int h, int w) {
    int ci = blk->c_in, co = blk->c_out;
    int spatial = h * w;

    /* GroupNorm → SiLU → Conv1 */
    float *tmp = (float *)malloc((size_t)ci * spatial * sizeof(float));
    qimg_vae_groupnorm(tmp, inp, blk->norm1_gamma, ci, h, w);
    qimg_vae_silu(tmp, ci * spatial);

    float *conv1_out = (float *)malloc((size_t)co * spatial * sizeof(float));
    qimg_vae_conv2d(conv1_out, tmp, blk->conv1_weight, blk->conv1_bias,
                    ci, h, w, co, 3, 3, 1);
    free(tmp);

    /* GroupNorm → SiLU → Conv2 */
    tmp = (float *)malloc((size_t)co * spatial * sizeof(float));
    qimg_vae_groupnorm(tmp, conv1_out, blk->norm2_gamma, co, h, w);
    qimg_vae_silu(tmp, co * spatial);
    free(conv1_out);

    float *conv2_out = (float *)malloc((size_t)co * spatial * sizeof(float));
    qimg_vae_conv2d(conv2_out, tmp, blk->conv2_weight, blk->conv2_bias,
                    co, h, w, co, 3, 3, 1);
    free(tmp);

    /* Shortcut + residual */
    if (blk->shortcut_weight) {
        /* 1x1 conv for channel change */
        for (int s = 0; s < spatial; s++) {
            for (int o = 0; o < co; o++) {
                float sum = blk->shortcut_bias ? blk->shortcut_bias[o] : 0.0f;
                for (int i = 0; i < ci; i++)
                    sum += blk->shortcut_weight[(size_t)o * ci + i] *
                           inp[(size_t)i * spatial + s];
                out[(size_t)o * spatial + s] = conv2_out[(size_t)o * spatial + s] + sum;
            }
        }
    } else {
        for (int i = 0; i < co * spatial; i++)
            out[i] = conv2_out[i] + inp[i];
    }
    free(conv2_out);
}

/* ---- Load helpers ---- */

static void qimg_vae_load_resblock(qimg_vae_resblock *blk, const st_context *st,
                                    const char *prefix) {
    char name[512];

    snprintf(name, sizeof(name), "%s.residual.0.gamma", prefix);
    blk->norm1_gamma = qimg_vae_load_tensor(st, name, NULL);
    /* gamma is stored as [C, 1, 1, 1] — just use first C values */

    int co, ci, kh, kw;
    snprintf(name, sizeof(name), "%s.residual.2.weight", prefix);
    blk->conv1_weight = qimg_vae_load_conv3d_as_2d(st, name, &co, &ci, &kh, &kw);
    snprintf(name, sizeof(name), "%s.residual.2.bias", prefix);
    blk->conv1_bias = qimg_vae_load_tensor(st, name, NULL);
    blk->c_in = ci;
    blk->c_out = co;

    snprintf(name, sizeof(name), "%s.residual.3.gamma", prefix);
    blk->norm2_gamma = qimg_vae_load_tensor(st, name, NULL);

    snprintf(name, sizeof(name), "%s.residual.6.weight", prefix);
    blk->conv2_weight = qimg_vae_load_conv3d_as_2d(st, name, &co, &ci, &kh, &kw);
    snprintf(name, sizeof(name), "%s.residual.6.bias", prefix);
    blk->conv2_bias = qimg_vae_load_tensor(st, name, NULL);

    snprintf(name, sizeof(name), "%s.shortcut.weight", prefix);
    int sc_idx = safetensors_find(st, name);
    if (sc_idx >= 0) {
        const uint64_t *sh = safetensors_shape(st, sc_idx);
        /* shortcut is [Co, Ci, 1, 1, 1] → just [Co, Ci] */
        size_t numel = sh[0] * sh[1];
        blk->shortcut_weight = qimg_vae_bf16_to_f32(safetensors_data(st, sc_idx), numel);
        snprintf(name, sizeof(name), "%s.shortcut.bias", prefix);
        blk->shortcut_bias = qimg_vae_load_tensor(st, name, NULL);
    }
}

/* ---- Public API ---- */

qimg_vae_model *qimg_vae_load(const char *path) {
    fprintf(stderr, "qimg_vae: loading %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "qimg_vae: failed to open\n"); return NULL; }

    qimg_vae_model *m = (qimg_vae_model *)calloc(1, sizeof(qimg_vae_model));
    m->st_ctx = st;
    m->z_dim = 16;
    m->base_dim = 96;

    /* post_quant_conv (conv2) */
    {
        size_t n;
        m->pqc_weight = qimg_vae_load_tensor(st, "conv2.weight", &n);
        m->pqc_bias = qimg_vae_load_tensor(st, "conv2.bias", NULL);
    }

    /* decoder.conv1: [384, 16, 3, 3, 3] → 2D */
    {
        int co, ci, kh, kw;
        m->dec_conv1_weight = qimg_vae_load_conv3d_as_2d(st, "decoder.conv1.weight",
                                                          &co, &ci, &kh, &kw);
        m->dec_conv1_bias = qimg_vae_load_tensor(st, "decoder.conv1.bias", NULL);
    }

    /* Middle block */
    qimg_vae_load_resblock(&m->mid_res0, st, "decoder.middle.0");
    qimg_vae_load_resblock(&m->mid_res2, st, "decoder.middle.2");

    /* Middle attention */
    m->mid_attn.c = 384;
    m->mid_attn.norm_gamma = qimg_vae_load_tensor(st, "decoder.middle.1.norm.gamma", NULL);
    {
        /* to_qkv is [1152, 384, 1, 1] 2D conv acting as linear */
        int idx = safetensors_find(st, "decoder.middle.1.to_qkv.weight");
        if (idx >= 0) {
            const uint64_t *sh = safetensors_shape(st, idx);
            size_t n = sh[0] * sh[1]; /* 1152 * 384 */
            m->mid_attn.qkv_weight = qimg_vae_bf16_to_f32(
                safetensors_data(st, idx), n);
        }
    }
    m->mid_attn.qkv_bias = qimg_vae_load_tensor(st, "decoder.middle.1.to_qkv.bias", NULL);
    {
        int idx = safetensors_find(st, "decoder.middle.1.proj.weight");
        if (idx >= 0) {
            const uint64_t *sh = safetensors_shape(st, idx);
            size_t n = sh[0] * sh[1];
            m->mid_attn.proj_weight = qimg_vae_bf16_to_f32(
                safetensors_data(st, idx), n);
        }
    }
    m->mid_attn.proj_bias = qimg_vae_load_tensor(st, "decoder.middle.1.proj.bias", NULL);

    /* Upsample blocks (0-14) */
    m->n_up_blocks = 15;
    m->up_res = (qimg_vae_resblock *)calloc(15, sizeof(qimg_vae_resblock));
    m->up_sample = (qimg_vae_upsample *)calloc(15, sizeof(qimg_vae_upsample));
    m->up_has_sample = (int *)calloc(15, sizeof(int));

    for (int i = 0; i < 15; i++) {
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "decoder.upsamples.%d", i);

        /* Check if this block has residual weights */
        char check_name[256];
        snprintf(check_name, sizeof(check_name),
                 "decoder.upsamples.%d.residual.2.weight", i);
        if (safetensors_find(st, check_name) >= 0) {
            qimg_vae_load_resblock(&m->up_res[i], st, prefix);
        } else {
            fprintf(stderr, "  upsample %d: resample-only (no residual)\n", i);
        }

        /* Check for resample */
        char rs_name[256];
        snprintf(rs_name, sizeof(rs_name), "decoder.upsamples.%d.resample.1.weight", i);
        int rs_idx = safetensors_find(st, rs_name);
        if (rs_idx >= 0) {
            m->up_has_sample[i] = 1;
            const uint64_t *sh = safetensors_shape(st, rs_idx);
            m->up_sample[i].c_out = (int)sh[0];
            m->up_sample[i].c_in = (int)sh[1];
            size_t n = sh[0] * sh[1] * sh[2] * sh[3];
            m->up_sample[i].conv_weight = qimg_vae_bf16_to_f32(
                safetensors_data(st, rs_idx), n);
            snprintf(rs_name, sizeof(rs_name),
                     "decoder.upsamples.%d.resample.1.bias", i);
            m->up_sample[i].conv_bias = qimg_vae_load_tensor(st, rs_name, NULL);
            fprintf(stderr, "  upsample %d: resample %d→%d\n",
                    i, m->up_sample[i].c_in, m->up_sample[i].c_out);
        }
    }

    /* Head */
    m->head_norm_gamma = qimg_vae_load_tensor(st, "decoder.head.0.gamma", NULL);
    {
        int co, ci, kh, kw;
        m->head_conv_weight = qimg_vae_load_conv3d_as_2d(st,
            "decoder.head.2.weight", &co, &ci, &kh, &kw);
    }
    m->head_conv_bias = qimg_vae_load_tensor(st, "decoder.head.2.bias", NULL);

    fprintf(stderr, "qimg_vae: loaded decoder (%d upsample blocks)\n", m->n_up_blocks);
    return m;
}

void qimg_vae_free(qimg_vae_model *m) {
    if (!m) return;
    /* TODO: free all weight buffers */
    if (m->st_ctx) safetensors_close((st_context *)m->st_ctx);
    if (m->up_res) free(m->up_res);
    if (m->up_sample) free(m->up_sample);
    if (m->up_has_sample) free(m->up_has_sample);
    free(m);
}

void qimg_vae_decode(float *rgb_out, const float *latent,
                     int lat_h, int lat_w, qimg_vae_model *m) {
    int h = lat_h, w = lat_w;
    int c = m->z_dim;  /* 16 */

    fprintf(stderr, "qimg_vae: decoding [%d, %d, %d]\n", c, h, w);

    /* post_quant_conv: 1×1 conv (identity-like for 16→16) */
    float *x = (float *)malloc((size_t)c * h * w * sizeof(float));
    if (m->pqc_weight) {
        /* 1x1 conv: out[o,s] = sum_i w[o,i] * inp[i,s] + bias[o] */
        int spatial = h * w;
        for (int s = 0; s < spatial; s++) {
            for (int o = 0; o < c; o++) {
                float sum = m->pqc_bias ? m->pqc_bias[o] : 0.0f;
                for (int i = 0; i < c; i++)
                    sum += m->pqc_weight[(size_t)o * c + i] *
                           latent[(size_t)i * spatial + s];
                x[(size_t)o * spatial + s] = sum;
            }
        }
    } else {
        memcpy(x, latent, (size_t)c * h * w * sizeof(float));
    }

    /* decoder.conv1: 16→384, 3×3 */
    c = 384;
    {
        float *out = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_conv2d(out, x, m->dec_conv1_weight, m->dec_conv1_bias,
                        16, h, w, c, 3, 3, 1);
        free(x);
        x = out;
    }
    fprintf(stderr, "  after conv1: [%d, %d, %d]\n", c, h, w);

    /* Middle block: ResBlock → Attention → ResBlock */
    {
        float *out = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_resblock_forward(out, x, &m->mid_res0, h, w);
        free(x); x = out;
    }
    {
        float *out = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_spatial_attn(out, x, m->mid_attn.norm_gamma,
                              m->mid_attn.qkv_weight, m->mid_attn.qkv_bias,
                              m->mid_attn.proj_weight, m->mid_attn.proj_bias,
                              c, h, w);
        free(x); x = out;
    }
    {
        float *out = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_resblock_forward(out, x, &m->mid_res2, h, w);
        free(x); x = out;
    }
    fprintf(stderr, "  after middle: [%d, %d, %d]\n", c, h, w);

    /* Upsample blocks */
    for (int i = 0; i < m->n_up_blocks; i++) {
        qimg_vae_resblock *rb = &m->up_res[i];

        /* ResBlock (skip if this is a resample-only block) */
        if (rb->conv1_weight) {
            int new_c = rb->c_out;
            float *out = (float *)malloc((size_t)new_c * h * w * sizeof(float));
            qimg_vae_resblock_forward(out, x, rb, h, w);
            free(x); x = out;
            c = new_c;
        }

        /* Spatial upsample if present */
        if (m->up_has_sample[i]) {
            qimg_vae_upsample *us = &m->up_sample[i];
            /* NN upsample 2× */
            float *up = qimg_vae_nn_upsample(x, c, h, w);
            h *= 2; w *= 2;
            /* Conv2d */
            int new_c2 = us->c_out;
            float *conv_out = (float *)malloc((size_t)new_c2 * h * w * sizeof(float));
            qimg_vae_conv2d_zero(conv_out, up, us->conv_weight, us->conv_bias,
                                 c, h, w, new_c2, 3, 3, 1);
            free(up); free(x);
            x = conv_out;
            c = new_c2;
        }

        if (i % 3 == 2 || m->up_has_sample[i])
            fprintf(stderr, "  after upsample %d: [%d, %d, %d]\n", i, c, h, w);
    }

    /* Head: GroupNorm → SiLU → Conv3d→2d(96→3) */
    {
        float *normed = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_groupnorm(normed, x, m->head_norm_gamma, c, h, w);
        qimg_vae_silu(normed, c * h * w);
        qimg_vae_conv2d(rgb_out, normed, m->head_conv_weight, m->head_conv_bias,
                        c, h, w, 3, 3, 3, 1);
        free(normed);
    }
    free(x);
    fprintf(stderr, "qimg_vae: decode complete [3, %d, %d]\n", h, w);
}

#endif /* QIMG_VAE_IMPLEMENTATION */
#endif /* QIMG_VAE_H */
