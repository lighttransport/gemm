/*
 * flux2_klein_vae.h - Flux.2 Klein VAE decoder (2D GroupNorm version)
 *
 * Usage:
 *   #define FLUX2_VAE_IMPLEMENTATION
 *   #include "flux2_klein_vae.h"
 *
 * Dependencies: safetensors.h
 *
 * Architecture: Standard 2D VAE decoder (SDXL-family)
 *   latent_channels=32, block_out_channels=[128,256,512,512]
 *   GroupNorm(num_groups=32), SiLU, layers_per_block=2
 *   4 up-blocks: 3 with nearest-neighbor 2× upsample, 1 final without
 *   Mid-block: 2 ResBlocks + 1 single-head self-attention
 *
 * NOTE: Provisional architecture. Run inspect_weights.py after downloading
 *       model files to confirm exact channel dimensions.
 *
 * API:
 *   flux2_vae_model *flux2_vae_load(const char *path);
 *   void             flux2_vae_free(flux2_vae_model *m);
 *   void             flux2_vae_decode(float *rgb_out, const float *latent,
 *                                     int lat_h, int lat_w,
 *                                     flux2_vae_model *m);
 */
#ifndef FLUX2_KLEIN_VAE_H
#define FLUX2_KLEIN_VAE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TODO: Confirm after weight inspection. Standard SDXL-family VAE defaults. */
#define FLUX2_VAE_LATENT_CHANNELS 32
#define FLUX2_VAE_NUM_GROUPS      32

typedef struct {
    float *norm1_w, *norm1_b;  /* GroupNorm [c_in] */
    float *conv1_w, *conv1_b;  /* [c_out, c_in, 3, 3] */
    float *norm2_w, *norm2_b;  /* GroupNorm [c_out] */
    float *conv2_w, *conv2_b;  /* [c_out, c_out, 3, 3] */
    float *skip_w,  *skip_b;   /* [c_out, c_in, 1, 1] or NULL if c_in==c_out */
    int c_in, c_out;
} flux2_vae_resblock;

typedef struct {
    float *norm_w, *norm_b;    /* GroupNorm [c] */
    float *q_w, *q_b;          /* [c, c] */
    float *k_w, *k_b;          /* [c, c] */
    float *v_w, *v_b;          /* [c, c] */
    float *out_w, *out_b;      /* [c, c] */
    int c;
} flux2_vae_attn;

typedef struct {
    float *conv_w, *conv_b;    /* [c, c, 3, 3] for upsample conv */
    int c_in, c_out;
} flux2_vae_upsample;

typedef struct {
    int latent_channels;   /* 32 */
    int num_groups;        /* 32 (GroupNorm) */

    /* post_quant_conv: [latent_ch, latent_ch, 1, 1] */
    float *pqc_w, *pqc_b;

    /* decoder.conv_in: [latent_ch → ch[-1], 3, 3] */
    float *conv_in_w, *conv_in_b;
    int conv_in_out_ch;    /* = block_out_channels[-1] = 512 */

    /* mid_block: 2 resblocks + 1 attention */
    flux2_vae_resblock mid_res0;
    flux2_vae_attn     mid_attn;
    flux2_vae_resblock mid_res1;

    /* up_blocks[4]: index 0 = closest to mid (highest ch) */
    /* Each has layers_per_block=2 resblocks + optional upsample */
    flux2_vae_resblock up_res[4][2];   /* [block][layer] */
    flux2_vae_upsample up_sample[4];   /* non-NULL if block has upsample */
    int up_has_sample[4];
    int up_in_ch[4];                   /* input channels for each block */
    int up_out_ch[4];                  /* output channels for each block */

    /* decoder.conv_norm_out: GroupNorm [out_ch] */
    float *norm_out_w, *norm_out_b;

    /* decoder.conv_out: [3, out_ch, 3, 3] */
    float *conv_out_w, *conv_out_b;
    int conv_out_in_ch;    /* = block_out_channels[0] = 128 */

    /* Batch norm parameters: applied to patchified latent (128 ch) before decode.
     * Applied as: latent_p[c] = latent_p[c] * bn_std[c] + bn_mean[c] */
    float *bn_mean;  /* [128] */
    float *bn_var;   /* [128] */
    float  bn_eps;   /* 1e-5 */
    int    bn_n_ch;  /* 128 = latent_channels * patch_size^2 */

    void *st_ctx;  /* safetensors context (kept for mmap lifetime) */
    int n_threads; /* parallelism for decode (default 1, set after load) */
} flux2_vae_model;

flux2_vae_model *flux2_vae_load(const char *path);
void             flux2_vae_free(flux2_vae_model *m);

/* Decode: latent [lat_ch, lat_h, lat_w] → rgb_out [3, lat_h*8, lat_w*8]
 * rgb_out is in [-1, 1] range (not clamped). Caller pre-allocates. */
void flux2_vae_decode(float *rgb_out, const float *latent,
                      int lat_h, int lat_w, flux2_vae_model *m);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef FLUX2_VAE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "safetensors.h"

static int flux2_vae_nthreads_g = 1;

/* ---- dtype conversion ---- */

static float flux2_vae_bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f; memcpy(&f, &bits, sizeof(float));
    return f;
}

static float flux2_vae_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f32;
    if (exp == 0)       f32 = sign << 31;
    else if (exp == 31) f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
    else                f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v; memcpy(&v, &f32, sizeof(float));
    return v;
}

static float *flux2_vae_load_tensor(const st_context *st, const char *name,
                                    size_t *out_numel) {
    int idx = safetensors_find(st, name);
    if (idx < 0) { if (out_numel) *out_numel = 0; return NULL; }
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t numel = 1;
    for (int d = 0; d < ndims; d++) numel *= shape[d];
    if (out_numel) *out_numel = numel;

    const char *dtype = safetensors_dtype(st, idx);
    const void *data  = safetensors_data(st, idx);

    float *buf = (float *)malloc(numel * sizeof(float));
    if (!buf) return NULL;

    if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *s = (const uint16_t *)data;
        for (size_t i = 0; i < numel; i++) buf[i] = flux2_vae_bf16_to_f32(s[i]);
    } else if (strcmp(dtype, "F16") == 0) {
        const uint16_t *s = (const uint16_t *)data;
        for (size_t i = 0; i < numel; i++) buf[i] = flux2_vae_f16_to_f32(s[i]);
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(buf, data, numel * sizeof(float));
    } else {
        fprintf(stderr, "flux2_vae: unsupported dtype '%s' for '%s'\n", dtype, name);
        free(buf); return NULL;
    }
    return buf;
}

/* Helper: try both weight/bias key patterns */
static void flux2_vae_load_wb(const st_context *st,
                               const char *prefix, const char *wname, const char *bname,
                               float **w, float **b) {
    char key[256];
    snprintf(key, sizeof(key), "%s.%s", prefix, wname);
    *w = flux2_vae_load_tensor(st, key, NULL);
    if (bname) {
        snprintf(key, sizeof(key), "%s.%s", prefix, bname);
        *b = flux2_vae_load_tensor(st, key, NULL);
    } else {
        *b = NULL;
    }
}

/* ---- Compute primitives ---- */

/* GroupNorm: channels-first [C, H*W], num_groups groups.
 * out[c,s] = (x[c,s] - mean_g) / sqrt(var_g + eps) * gamma[c] + beta[c] */
static void flux2_vae_groupnorm(float *out, const float *x,
                                 const float *gamma, const float *beta,
                                 int c, int h, int w, int num_groups) {
    int spatial = h * w;
    int group_size = c / num_groups;
    float eps = 1e-6f;

#ifdef _OPENMP
    #pragma omp parallel for num_threads(flux2_vae_nthreads_g) schedule(static)
#endif
    for (int g = 0; g < num_groups; g++) {
        int c0 = g * group_size;
        int ne = group_size * spatial;

        float mean = 0.0f;
        for (int ci = c0; ci < c0 + group_size; ci++)
            for (int s = 0; s < spatial; s++)
                mean += x[(size_t)ci * spatial + s];
        mean /= (float)ne;

        float var = 0.0f;
        for (int ci = c0; ci < c0 + group_size; ci++)
            for (int s = 0; s < spatial; s++) {
                float d = x[(size_t)ci * spatial + s] - mean;
                var += d * d;
            }
        var /= (float)ne;

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int ci = c0; ci < c0 + group_size; ci++)
            for (int s = 0; s < spatial; s++) {
                float v = (x[(size_t)ci * spatial + s] - mean) * inv_std;
                float g_val = gamma ? gamma[ci] : 1.0f;
                float b_val = beta  ? beta[ci]  : 0.0f;
                out[(size_t)ci * spatial + s] = v * g_val + b_val;
            }
    }
}

static void flux2_vae_silu(float *x, size_t n) {
    for (size_t i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* Conv2D: [ci, h, w] → [co, oh, ow], kernel [co, ci, kh, kw], same padding */
static void flux2_vae_conv2d(float *out, const float *in,
                              const float *w, const float *b,
                              int ci, int h, int w_in,
                              int co, int kh, int kw,
                              int pad) {
    int oh = h, ow = w_in;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) num_threads(flux2_vae_nthreads_g) schedule(static)
#endif
    for (int oc = 0; oc < co; oc++) {
        for (int y = 0; y < oh; y++) {
            float bias = b ? b[oc] : 0.0f;
            for (int x = 0; x < ow; x++) {
                float sum = bias;
                for (int ic = 0; ic < ci; ic++) {
                    for (int ky = 0; ky < kh; ky++) {
                        int iy = y + ky - pad;
                        if (iy < 0 || iy >= h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            int ix = x + kx - pad;
                            if (ix < 0 || ix >= w_in) continue;
                            float iv = in[(size_t)ic * h * w_in + iy * w_in + ix];
                            float kv = w[((size_t)oc * ci + ic) * kh * kw + ky * kw + kx];
                            sum += iv * kv;
                        }
                    }
                }
                out[(size_t)oc * oh * ow + y * ow + x] = sum;
            }
        }
    }
}

/* Nearest-neighbor 2× upsample: [c, h, w] → [c, 2h, 2w] */
static float *flux2_vae_upsample2x(const float *x, int c, int h, int w) {
    float *out = (float *)malloc((size_t)c * 2 * h * 2 * w * sizeof(float));
    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < h; y++)
            for (int x2 = 0; x2 < w; x2++) {
                float v = x[(size_t)ch * h * w + y * w + x2];
                out[(size_t)ch * 2*h * 2*w + (2*y)   * 2*w + 2*x2]   = v;
                out[(size_t)ch * 2*h * 2*w + (2*y)   * 2*w + 2*x2+1] = v;
                out[(size_t)ch * 2*h * 2*w + (2*y+1) * 2*w + 2*x2]   = v;
                out[(size_t)ch * 2*h * 2*w + (2*y+1) * 2*w + 2*x2+1] = v;
            }
    return out;
}

/* ResBlock forward: GroupNorm → SiLU → Conv3x3 → GroupNorm → SiLU → Conv3x3 + skip */
static void flux2_vae_resblock_forward(float *out, const float *x,
                                        const flux2_vae_resblock *rb,
                                        int h, int w, int num_groups) {
    int ci = rb->c_in, co = rb->c_out;
    size_t ci_sz = (size_t)ci * h * w;
    size_t co_sz = (size_t)co * h * w;

    float *normed = (float *)malloc(ci_sz * sizeof(float));
    flux2_vae_groupnorm(normed, x, rb->norm1_w, rb->norm1_b, ci, h, w, num_groups);
    flux2_vae_silu(normed, ci_sz);

    float *conv1_out = (float *)malloc(co_sz * sizeof(float));
    flux2_vae_conv2d(conv1_out, normed, rb->conv1_w, rb->conv1_b, ci, h, w, co, 3, 3, 1);
    free(normed);

    float *normed2 = (float *)malloc(co_sz * sizeof(float));
    flux2_vae_groupnorm(normed2, conv1_out, rb->norm2_w, rb->norm2_b, co, h, w, num_groups);
    flux2_vae_silu(normed2, co_sz);
    free(conv1_out);

    float *conv2_out = (float *)malloc(co_sz * sizeof(float));
    flux2_vae_conv2d(conv2_out, normed2, rb->conv2_w, rb->conv2_b, co, h, w, co, 3, 3, 1);
    free(normed2);

    /* Skip connection */
    const float *skip = x;
    float *skip_buf = NULL;
    if (rb->skip_w) {
        skip_buf = (float *)malloc(co_sz * sizeof(float));
        flux2_vae_conv2d(skip_buf, x, rb->skip_w, rb->skip_b, ci, h, w, co, 1, 1, 0);
        skip = skip_buf;
    }

    for (size_t i = 0; i < co_sz; i++)
        out[i] = conv2_out[i] + skip[i];

    free(conv2_out);
    if (skip_buf) free(skip_buf);
}

/* Single-head self-attention for mid-block: GroupNorm → Q/K/V → attn → proj + residual */
static void flux2_vae_mid_attn_forward(float *out, const float *x,
                                        const flux2_vae_attn *attn,
                                        int h, int w, int num_groups) {
    int c = attn->c;
    int spatial = h * w;
    size_t feat_sz = (size_t)c * spatial;

    /* GroupNorm */
    float *normed = (float *)malloc(feat_sz * sizeof(float));
    flux2_vae_groupnorm(normed, x, attn->norm_w, attn->norm_b, c, h, w, num_groups);

    /* Reshape to [spatial, c] for matmul: transpose from [c, spatial] */
    float *normed_t = (float *)malloc(feat_sz * sizeof(float));
    for (int ch = 0; ch < c; ch++)
        for (int s = 0; s < spatial; s++)
            normed_t[(size_t)s * c + ch] = normed[(size_t)ch * spatial + s];
    free(normed);

    /* Q, K, V projections: [spatial, c] × [c, c]^T → [spatial, c] */
    float *q = (float *)malloc((size_t)spatial * c * sizeof(float));
    float *k = (float *)malloc((size_t)spatial * c * sizeof(float));
    float *v = (float *)malloc((size_t)spatial * c * sizeof(float));

    for (int s = 0; s < spatial; s++) {
        for (int oc = 0; oc < c; oc++) {
            float sq = attn->q_b ? attn->q_b[oc] : 0.0f;
            float sk = attn->k_b ? attn->k_b[oc] : 0.0f;
            float sv = attn->v_b ? attn->v_b[oc] : 0.0f;
            for (int ic = 0; ic < c; ic++) {
                float xv = normed_t[(size_t)s * c + ic];
                sq += xv * attn->q_w[(size_t)oc * c + ic];
                sk += xv * attn->k_w[(size_t)oc * c + ic];
                sv += xv * attn->v_w[(size_t)oc * c + ic];
            }
            q[(size_t)s * c + oc] = sq;
            k[(size_t)s * c + oc] = sk;
            v[(size_t)s * c + oc] = sv;
        }
    }
    free(normed_t);

    /* Scaled dot-product attention: attn = softmax(Q @ K^T / sqrt(c)) @ V */
    float scale = 1.0f / sqrtf((float)c);
    float *attn_out = (float *)malloc((size_t)spatial * c * sizeof(float));

    /* Allocate attn weights [spatial, spatial] — may be large for big images */
    float *scores = (float *)malloc((size_t)spatial * spatial * sizeof(float));

    /* Compute Q @ K^T */
    for (int qi = 0; qi < spatial; qi++)
        for (int ki = 0; ki < spatial; ki++) {
            float s = 0.0f;
            for (int d = 0; d < c; d++)
                s += q[(size_t)qi * c + d] * k[(size_t)ki * c + d];
            scores[(size_t)qi * spatial + ki] = s * scale;
        }

    /* Softmax over key dim */
    for (int qi = 0; qi < spatial; qi++) {
        float *row = scores + (size_t)qi * spatial;
        float mx = row[0];
        for (int ki = 1; ki < spatial; ki++) if (row[ki] > mx) mx = row[ki];
        float sum = 0.0f;
        for (int ki = 0; ki < spatial; ki++) { row[ki] = expf(row[ki] - mx); sum += row[ki]; }
        float inv = 1.0f / sum;
        for (int ki = 0; ki < spatial; ki++) row[ki] *= inv;
    }

    /* attn_weights @ V */
    for (int qi = 0; qi < spatial; qi++)
        for (int d = 0; d < c; d++) {
            float s = 0.0f;
            for (int ki = 0; ki < spatial; ki++)
                s += scores[(size_t)qi * spatial + ki] * v[(size_t)ki * c + d];
            attn_out[(size_t)qi * c + d] = s;
        }

    free(scores); free(q); free(k); free(v);

    /* Output projection: [spatial, c] × [c, c]^T → [spatial, c] */
    float *proj_out = (float *)malloc((size_t)spatial * c * sizeof(float));
    for (int s = 0; s < spatial; s++) {
        for (int oc = 0; oc < c; oc++) {
            float sv = attn->out_b ? attn->out_b[oc] : 0.0f;
            for (int ic = 0; ic < c; ic++)
                sv += attn_out[(size_t)s * c + ic] * attn->out_w[(size_t)oc * c + ic];
            proj_out[(size_t)s * c + oc] = sv;
        }
    }
    free(attn_out);

    /* Transpose back [spatial, c] → [c, spatial] and add residual */
    for (int ch = 0; ch < c; ch++)
        for (int s = 0; s < spatial; s++)
            out[(size_t)ch * spatial + s] = x[(size_t)ch * spatial + s]
                                           + proj_out[(size_t)s * c + ch];
    free(proj_out);
}

/* ---- Weight loading ---- */

static void flux2_vae_load_resblock(flux2_vae_resblock *rb, const st_context *st,
                                     const char *prefix, int c_in, int c_out) {
    rb->c_in = c_in; rb->c_out = c_out;
    char key[512];

#define LOAD(field, suffix) do { \
    snprintf(key, sizeof(key), "%s." suffix, prefix); \
    rb->field = flux2_vae_load_tensor(st, key, NULL); \
} while(0)

    LOAD(norm1_w, "norm1.weight");
    LOAD(norm1_b, "norm1.bias");
    LOAD(conv1_w, "conv1.weight");
    LOAD(conv1_b, "conv1.bias");
    LOAD(norm2_w, "norm2.weight");
    LOAD(norm2_b, "norm2.bias");
    LOAD(conv2_w, "conv2.weight");
    LOAD(conv2_b, "conv2.bias");

    /* Skip connection (shortcut / conv_shortcut) */
    snprintf(key, sizeof(key), "%s.conv_shortcut.weight", prefix);
    rb->skip_w = flux2_vae_load_tensor(st, key, NULL);
    if (!rb->skip_w) {
        snprintf(key, sizeof(key), "%s.nin_shortcut.weight", prefix);
        rb->skip_w = flux2_vae_load_tensor(st, key, NULL);
    }
    if (rb->skip_w) {
        snprintf(key, sizeof(key), "%s.conv_shortcut.bias", prefix);
        rb->skip_b = flux2_vae_load_tensor(st, key, NULL);
        if (!rb->skip_b) {
            snprintf(key, sizeof(key), "%s.nin_shortcut.bias", prefix);
            rb->skip_b = flux2_vae_load_tensor(st, key, NULL);
        }
    } else {
        rb->skip_b = NULL;
    }
#undef LOAD
}

static void flux2_vae_load_attn(flux2_vae_attn *at, const st_context *st,
                                 const char *prefix, int c) {
    at->c = c;
    char key[512];
#define LOAD(field, suffix) do { \
    snprintf(key, sizeof(key), "%s." suffix, prefix); \
    at->field = flux2_vae_load_tensor(st, key, NULL); \
} while(0)
    LOAD(norm_w, "group_norm.weight");
    LOAD(norm_b, "group_norm.bias");
    LOAD(q_w, "to_q.weight"); LOAD(q_b, "to_q.bias");
    LOAD(k_w, "to_k.weight"); LOAD(k_b, "to_k.bias");
    LOAD(v_w, "to_v.weight"); LOAD(v_b, "to_v.bias");
    LOAD(out_w, "to_out.0.weight"); LOAD(out_b, "to_out.0.bias");
#undef LOAD
}

flux2_vae_model *flux2_vae_load(const char *path) {
    fprintf(stderr, "flux2_vae: loading %s\n", path);

    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "flux2_vae: failed to open %s\n", path);
        return NULL;
    }

    flux2_vae_model *m = (flux2_vae_model *)calloc(1, sizeof(flux2_vae_model));
    m->latent_channels = FLUX2_VAE_LATENT_CHANNELS;
    m->num_groups      = FLUX2_VAE_NUM_GROUPS;
    m->st_ctx          = st;

    char key[512];

    /* Batch norm parameters for latent de-normalization (patchified space, 128 ch) */
    m->bn_mean = flux2_vae_load_tensor(st, "bn.running_mean", NULL);
    m->bn_var  = flux2_vae_load_tensor(st, "bn.running_var",  NULL);
    m->bn_eps  = 1e-5f;
    /* bn_n_ch = latent_channels * 4 (for 2x2 patchification) */
    m->bn_n_ch = m->bn_mean ? 128 : 0;

    /* post_quant_conv */
    m->pqc_w = flux2_vae_load_tensor(st, "post_quant_conv.weight", NULL);
    m->pqc_b = flux2_vae_load_tensor(st, "post_quant_conv.bias",   NULL);

    /* decoder.conv_in */
    m->conv_in_w = flux2_vae_load_tensor(st, "decoder.conv_in.weight", NULL);
    m->conv_in_b = flux2_vae_load_tensor(st, "decoder.conv_in.bias",   NULL);

    /* Determine conv_in output channels (= block_out_channels[-1]) */
    {
        int idx = safetensors_find(st, "decoder.conv_in.weight");
        if (idx >= 0) {
            const uint64_t *sh = safetensors_shape(st, idx);
            m->conv_in_out_ch = (int)sh[0];
        } else {
            m->conv_in_out_ch = 512; /* provisional */
        }
    }
    fprintf(stderr, "flux2_vae: conv_in_out_ch=%d (lat_ch=%d)\n",
            m->conv_in_out_ch, m->latent_channels);

    /* mid_block */
    int mid_ch = m->conv_in_out_ch;
    flux2_vae_load_resblock(&m->mid_res0, st, "decoder.mid_block.resnets.0", mid_ch, mid_ch);
    flux2_vae_load_attn(&m->mid_attn, st, "decoder.mid_block.attentions.0", mid_ch);
    flux2_vae_load_resblock(&m->mid_res1, st, "decoder.mid_block.resnets.1", mid_ch, mid_ch);

    /* up_blocks: dynamically determine channel sizes from weight shapes */
    int block_out_ch[5] = {0,0,0,0,0}; /* 4 blocks + prev for conv_in */
    block_out_ch[4] = mid_ch;           /* mid block channels (= block_out[-1]) */

    for (int bi = 0; bi < 4; bi++) {
        /* Determine output channels of this block from resnets.0.conv1.weight shape */
        snprintf(key, sizeof(key), "decoder.up_blocks.%d.resnets.0.conv1.weight", bi);
        int idx = safetensors_find(st, key);
        int ch_out = mid_ch; /* fallback */
        (void)block_out_ch;

        if (idx >= 0) {
            const uint64_t *sh = safetensors_shape(st, idx);
            ch_out = (int)sh[0];
        }
        block_out_ch[bi] = ch_out;  /* store for next block's input */
        /* Note: block 0 output → block 1 input etc. (stored in reverse order below) */
    }

    /* Load up_blocks — each has 2 resblocks + optional upsample */
    /* The channel progression: mid_ch → ub[0] → ub[1] → ub[2] → ub[3] → conv_out_ch
     * In standard SDXL-family: [512,512,256,128] for block_out_channels=[128,256,512,512]
     * up_blocks are in reverse order from encoder (largest channels first in decoder).
     * up_blocks[0]: 512→512, upsample
     * up_blocks[1]: 512→256, upsample
     * up_blocks[2]: 256→128, upsample
     * up_blocks[3]: 128→128, no upsample */
    int cur_ch = mid_ch;
    for (int bi = 0; bi < 4; bi++) {
        snprintf(key, sizeof(key), "decoder.up_blocks.%d.resnets.0.conv1.weight", bi);
        int idx = safetensors_find(st, key);
        int blk_in  = cur_ch;
        int blk_out = cur_ch;
        if (idx >= 0) {
            const uint64_t *sh = safetensors_shape(st, idx);
            blk_out = (int)sh[0];
            blk_in  = (int)sh[1];
        }
        m->up_in_ch[bi]  = blk_in;
        m->up_out_ch[bi] = blk_out;

        /* Load resnets.0 and resnets.1 */
        snprintf(key, sizeof(key), "decoder.up_blocks.%d.resnets.0", bi);
        flux2_vae_load_resblock(&m->up_res[bi][0], st, key, blk_in, blk_out);

        /* resnets.1: input = blk_out (output of resnets.0) */
        snprintf(key, sizeof(key), "decoder.up_blocks.%d.resnets.1", bi);
        flux2_vae_load_resblock(&m->up_res[bi][1], st, key, blk_out, blk_out);

        /* Upsample: check for upsamplers.0.conv.weight */
        snprintf(key, sizeof(key), "decoder.up_blocks.%d.upsamplers.0.conv.weight", bi);
        float *up_conv_w = flux2_vae_load_tensor(st, key, NULL);
        if (up_conv_w) {
            m->up_has_sample[bi]     = 1;
            m->up_sample[bi].conv_w  = up_conv_w;
            snprintf(key, sizeof(key), "decoder.up_blocks.%d.upsamplers.0.conv.bias", bi);
            m->up_sample[bi].conv_b  = flux2_vae_load_tensor(st, key, NULL);
            m->up_sample[bi].c_in    = blk_out;
            m->up_sample[bi].c_out   = blk_out;
        } else {
            m->up_has_sample[bi] = 0;
        }
        cur_ch = blk_out;
        fprintf(stderr, "flux2_vae: up_block[%d] %d→%d, upsample=%d\n",
                bi, blk_in, blk_out, m->up_has_sample[bi]);
    }

    /* conv_norm_out */
    m->norm_out_w = flux2_vae_load_tensor(st, "decoder.conv_norm_out.weight", NULL);
    m->norm_out_b = flux2_vae_load_tensor(st, "decoder.conv_norm_out.bias",   NULL);

    /* conv_out */
    m->conv_out_w    = flux2_vae_load_tensor(st, "decoder.conv_out.weight", NULL);
    m->conv_out_b    = flux2_vae_load_tensor(st, "decoder.conv_out.bias",   NULL);
    m->conv_out_in_ch = cur_ch;

    fprintf(stderr, "flux2_vae: loaded OK. conv_out_in_ch=%d\n", cur_ch);
    return m;
}

void flux2_vae_free(flux2_vae_model *m) {
    if (!m) return;
    /* Free all allocated weight buffers */
    free(m->pqc_w); free(m->pqc_b);
    free(m->conv_in_w); free(m->conv_in_b);
    free(m->norm_out_w); free(m->norm_out_b);
    free(m->conv_out_w); free(m->conv_out_b);
    /* mid_block */
#define FREE_RB(rb) do { \
    free(rb.norm1_w); free(rb.norm1_b); \
    free(rb.conv1_w); free(rb.conv1_b); \
    free(rb.norm2_w); free(rb.norm2_b); \
    free(rb.conv2_w); free(rb.conv2_b); \
    free(rb.skip_w);  free(rb.skip_b);  \
} while(0)
    FREE_RB(m->mid_res0); FREE_RB(m->mid_res1);
    free(m->mid_attn.norm_w); free(m->mid_attn.norm_b);
    free(m->mid_attn.q_w); free(m->mid_attn.q_b);
    free(m->mid_attn.k_w); free(m->mid_attn.k_b);
    free(m->mid_attn.v_w); free(m->mid_attn.v_b);
    free(m->mid_attn.out_w); free(m->mid_attn.out_b);
    /* up_blocks */
    for (int bi = 0; bi < 4; bi++) {
        FREE_RB(m->up_res[bi][0]); FREE_RB(m->up_res[bi][1]);
        free(m->up_sample[bi].conv_w); free(m->up_sample[bi].conv_b);
    }
#undef FREE_RB
    if (m->st_ctx) safetensors_close((st_context *)m->st_ctx);
    free(m);
}

void flux2_vae_decode(float *rgb_out, const float *latent,
                      int lat_h, int lat_w, flux2_vae_model *m) {
    int nt = m->n_threads;
    if (nt <= 0) {
#ifdef _OPENMP
        nt = omp_get_max_threads();
#else
        nt = 1;
#endif
    }
    flux2_vae_nthreads_g = nt;
    int lc = m->latent_channels;
    int ng = m->num_groups;
    int h = lat_h, w = lat_w;
    int ps = 2; /* patch_size used by patchify/unpatchify */

    /* Stage 0: Batch-norm de-normalization in patchified latent space.
     * The pipeline applies: latent_p[c] = latent_p[c] * bn_std[c] + bn_mean[c]
     * where latent_p is the patchified form [lc*ps*ps, H_p, W_p].
     * In the unpatchified form [lc, lat_h, lat_w], this is equivalent to a
     * per-(ch, pr, pc) scale+shift where bn_ch = ch*ps*ps + pr*ps + pc. */
    float *latent_bn = (float *)malloc((size_t)lc * lat_h * lat_w * sizeof(float));
    if (m->bn_mean && m->bn_var) {
        int lat_h_p = lat_h / ps, lat_w_p = lat_w / ps;
        for (int ch = 0; ch < lc; ch++) {
            for (int pr = 0; pr < ps; pr++) {
                for (int pc = 0; pc < ps; pc++) {
                    int bn_ch = ch * (ps * ps) + pr * ps + pc;
                    float mn = m->bn_mean[bn_ch];
                    float sd = sqrtf(m->bn_var[bn_ch] + m->bn_eps);
                    for (int hp = 0; hp < lat_h_p; hp++) {
                        for (int wp = 0; wp < lat_w_p; wp++) {
                            int sh = hp * ps + pr, sw = wp * ps + pc;
                            size_t idx = (size_t)ch * lat_h * lat_w + (size_t)sh * lat_w + sw;
                            latent_bn[idx] = latent[idx] * sd + mn;
                        }
                    }
                }
            }
        }
    } else {
        memcpy(latent_bn, latent, (size_t)lc * lat_h * lat_w * sizeof(float));
    }
    latent = latent_bn;

    /* Stage 1: post_quant_conv (1×1) */
    size_t sz = (size_t)lc * h * w;
    float *x = (float *)malloc(sz * sizeof(float));
    if (m->pqc_w) {
        flux2_vae_conv2d(x, latent, m->pqc_w, m->pqc_b, lc, h, w, lc, 1, 1, 0);
    } else {
        memcpy(x, latent, sz * sizeof(float));
    }

    /* Stage 2: decoder.conv_in (3×3) */
    int c = m->conv_in_out_ch;
    float *t = (float *)malloc((size_t)c * h * w * sizeof(float));
    flux2_vae_conv2d(t, x, m->conv_in_w, m->conv_in_b, lc, h, w, c, 3, 3, 1);
    free(x); x = t;

    /* Stage 3: mid_block */
    {
        float *tmp = (float *)malloc((size_t)c * h * w * sizeof(float));
        flux2_vae_resblock_forward(tmp, x, &m->mid_res0, h, w, ng);
        free(x); x = tmp;
    }
    {
        float *tmp = (float *)malloc((size_t)c * h * w * sizeof(float));
        flux2_vae_mid_attn_forward(tmp, x, &m->mid_attn, h, w, ng);
        free(x); x = tmp;
    }
    {
        float *tmp = (float *)malloc((size_t)c * h * w * sizeof(float));
        flux2_vae_resblock_forward(tmp, x, &m->mid_res1, h, w, ng);
        free(x); x = tmp;
    }

    /* Stage 4: up_blocks */
    for (int bi = 0; bi < 4; bi++) {
        /* resnets.0 */
        {
            int new_c = m->up_res[bi][0].c_out;
            float *tmp = (float *)malloc((size_t)new_c * h * w * sizeof(float));
            flux2_vae_resblock_forward(tmp, x, &m->up_res[bi][0], h, w, ng);
            free(x); x = tmp; c = new_c;
        }
        /* resnets.1 */
        {
            int new_c = m->up_res[bi][1].c_out;
            float *tmp = (float *)malloc((size_t)new_c * h * w * sizeof(float));
            flux2_vae_resblock_forward(tmp, x, &m->up_res[bi][1], h, w, ng);
            free(x); x = tmp; c = new_c;
        }
        /* Upsample */
        if (m->up_has_sample[bi]) {
            float *up = flux2_vae_upsample2x(x, c, h, w);
            h *= 2; w *= 2;
            float *conv_out = (float *)malloc((size_t)c * h * w * sizeof(float));
            flux2_vae_conv2d(conv_out, up, m->up_sample[bi].conv_w,
                             m->up_sample[bi].conv_b, c, h, w, c, 3, 3, 1);
            free(up); free(x); x = conv_out;
        }
    }

    /* Stage 5: norm_out → SiLU → conv_out */
    {
        float *normed = (float *)malloc((size_t)c * h * w * sizeof(float));
        flux2_vae_groupnorm(normed, x, m->norm_out_w, m->norm_out_b, c, h, w, ng);
        flux2_vae_silu(normed, (size_t)c * h * w);
        free(x);

        /* conv_out: [c → 3, 3×3] */
        flux2_vae_conv2d(rgb_out, normed, m->conv_out_w, m->conv_out_b,
                         c, h, w, 3, 3, 3, 1);
        free(normed);
    }
    free(latent_bn);
}

#endif /* FLUX2_VAE_IMPLEMENTATION */
#endif /* FLUX2_KLEIN_VAE_H */
