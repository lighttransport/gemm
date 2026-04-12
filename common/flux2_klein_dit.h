/*
 * flux2_klein_dit.h - Flux.2 Klein DiT (double+single stream transformer)
 *
 * Confirmed architecture from weight inspection (flux-2-klein-4b-fp8.safetensors):
 *   hidden_dim=3072, n_heads=24, head_dim=128
 *   5 double-stream blocks + 20 single-stream blocks
 *   patch_in_channels=128, txt_dim=7680, n_ff=9216 (=3*H)
 *   FP8 E4M3 weights with per-tensor weight_scale
 *   SHARED global modulation (one set for all double blocks, one for all single)
 *   Fused QKV per block, fused gate+up MLP
 *
 * Usage:
 *   #define FLUX2_DIT_IMPLEMENTATION
 *   #include "flux2_klein_dit.h"
 *
 * API:
 *   flux2_dit_model *flux2_dit_load_safetensors(const char *path);
 *   void             flux2_dit_free(flux2_dit_model *m);
 *   void             flux2_dit_forward(float *out, const float *img_tokens, int n_img,
 *                                      const float *txt_tokens, int n_txt,
 *                                      float timestep, flux2_dit_model *m, int n_threads);
 */
#ifndef FLUX2_KLEIN_DIT_H
#define FLUX2_KLEIN_DIT_H

#include <stdint.h>
#include <stddef.h>
#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Confirmed architecture constants ---- */
#define FLUX2_HIDDEN_DIM          3072
#define FLUX2_N_HEADS             24
#define FLUX2_HEAD_DIM            128
#define FLUX2_N_DOUBLE_BLOCKS     5
#define FLUX2_N_SINGLE_BLOCKS     20
#define FLUX2_PATCH_IN_CHANNELS   128   /* img_in input dim (32 lat_ch * 2*2 patch) */
#define FLUX2_TXT_DIM             7680  /* text encoder output dim */
#define FLUX2_N_FF                9216  /* = 3 * hidden_dim */
#define FLUX2_ROPE_THETA          2000.0f

/* ---- Weight storage (dequanted to F32 at load time) ---- */
typedef struct {
    float *w;  /* [rows, cols] row-major */
    int    rows, cols;
} flux2_mat;

typedef struct {
    /* Per-head Q/K RMS norm scales [head_dim] */
    float *q_norm;   /* img_attn.norm.query_norm.scale */
    float *k_norm;   /* img_attn.norm.key_norm.scale */
    /* Fused QKV: [3*H, H] */
    flux2_mat qkv;   /* img_attn.qkv.weight */
    /* Output proj: [H, H] */
    flux2_mat proj;  /* img_attn.proj.weight */
    /* FFN: fused gate+up [2*n_ff, H], down [H, n_ff] */
    flux2_mat mlp_up;   /* img_mlp.0.weight  (gate+up combined) */
    flux2_mat mlp_down; /* img_mlp.2.weight */
} flux2_stream_block;

typedef struct {
    /* Two streams: img and txt */
    flux2_stream_block img;
    flux2_stream_block txt;
} flux2_double_block;

typedef struct {
    /* Per-head Q/K RMS norm scales [head_dim] */
    float *q_norm;   /* norm.query_norm.scale */
    float *k_norm;   /* norm.key_norm.scale */
    /* Fused [QKV(3H) + gate+up(2*n_ff)]: [9H, H] = [27648, 3072] */
    flux2_mat linear1;
    /* Fused output [H, H+n_ff]: [3072, 12288] */
    flux2_mat linear2;
} flux2_single_block;

typedef struct {
    int hidden_dim;
    int n_heads;
    int head_dim;
    int n_double_blocks;
    int n_single_blocks;
    int patch_in_channels;
    int txt_dim;
    int n_ff;

    /* Global embedders (BF16 → F32) */
    flux2_mat img_in;       /* [H, patch_in]  img_in.weight */
    flux2_mat txt_in;       /* [H, txt_dim]   txt_in.weight */
    flux2_mat time_in_lin1; /* [H, 256]       time_in.in_layer.weight */
    flux2_mat time_in_lin2; /* [H, H]         time_in.out_layer.weight */
    /* Bias optional */
    float *img_in_b;
    float *txt_in_b;
    float *time_in_lin1_b;
    float *time_in_lin2_b;

    /* Shared global modulation (BF16 → F32) — output through SiLU(temb) */
    flux2_mat mod_img; /* [6H, H]  double_stream_modulation_img.lin.weight */
    flux2_mat mod_txt; /* [6H, H]  double_stream_modulation_txt.lin.weight */
    flux2_mat mod_sgl; /* [3H, H]  single_stream_modulation.lin.weight */

    /* Output (BF16 → F32) */
    flux2_mat out_mod;  /* [2H, H]  final_layer.adaLN_modulation.1.weight */
    flux2_mat out_proj; /* [patch_in, H]  final_layer.linear.weight */

    /* Blocks */
    flux2_double_block *dblk;  /* [n_double_blocks] */
    flux2_single_block *sblk;  /* [n_single_blocks] */

    /* Memory tracking */
    float **bufs;
    int     n_bufs, cap_bufs;

    /* Debug: dump per-block intermediates (-1 = disabled) */
    int dump_dblk;    /* double-block index to dump */
    int dump_sblk;    /* single-block index to dump */
    void (*dump_fn)(const char *name, const float *data, int n, void *ctx);
    void *dump_ctx;
} flux2_dit_model;

flux2_dit_model *flux2_dit_load_safetensors(const char *path);
void             flux2_dit_free(flux2_dit_model *m);
void             flux2_dit_forward(float *out,
                                   const float *img_tokens, int n_img,
                                   const float *txt_tokens, int n_txt,
                                   float timestep,
                                   flux2_dit_model *m, int n_threads);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef FLUX2_DIT_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Global thread count set by flux2_dit_forward */
static int flux2_nthreads_g = 1;

/* ---- FP8 E4M3 LUT ---- */
static float g_fp8e4m3_lut[256];
static int   g_fp8e4m3_lut_init = 0;

static void flux2_fp8_lut_init(void) {
    if (g_fp8e4m3_lut_init) return;
    for (int i = 0; i < 256; i++) {
        uint8_t b = (uint8_t)i;
        int sign = (b >> 7) & 1;
        int exp  = (b >> 3) & 0xF;
        int mant = b & 0x7;
        float v;
        if (exp == 0 && mant == 0)    v = 0.0f;
        else if (exp == 15 && mant == 7) v = 0.0f; /* NaN → 0 */
        else if (exp == 0)            v = (float)mant / 8.0f * ldexpf(1.0f, -6);
        else                          v = (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, exp - 7);
        g_fp8e4m3_lut[i] = sign ? -v : v;
    }
    g_fp8e4m3_lut_init = 1;
}

static float flux2_bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f; memcpy(&f, &b, 4); return f;
}

/* ---- Buffer tracking ---- */
static void flux2_buf_track(flux2_dit_model *m, float *p) {
    if (!p) return;
    if (m->n_bufs >= m->cap_bufs) {
        m->cap_bufs = m->cap_bufs ? m->cap_bufs * 2 : 64;
        m->bufs = (float **)realloc(m->bufs, (size_t)m->cap_bufs * sizeof(float *));
    }
    m->bufs[m->n_bufs++] = p;
}

/* ---- Tensor loading ---- */

/* Load BF16 or F32 tensor from safetensors → malloc'd F32 array. */
static float *flux2_load_bf16(flux2_dit_model *m, st_context *st,
                               const char *name, int *rows, int *cols) {
    int idx = safetensors_find(st, name);
    if (idx < 0) { if (rows) *rows = 0; if (cols) *cols = 0; return NULL; }
    const uint64_t *sh = safetensors_shape(st, idx);
    int nd = safetensors_ndims(st, idx);
    int r = (nd >= 2) ? (int)sh[0] : 1;
    int c = (nd >= 2) ? (int)sh[1] : (int)sh[0];
    if (rows) *rows = r;
    if (cols) *cols = c;
    int n = r * c;
    float *out = (float *)malloc((size_t)n * sizeof(float));
    const char *dtype = safetensors_dtype(st, idx);
    if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *src = (const uint16_t *)safetensors_data(st, idx);
        for (int i = 0; i < n; i++) out[i] = flux2_bf16_to_f32(src[i]);
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(out, safetensors_data(st, idx), (size_t)n * sizeof(float));
    } else {
        fprintf(stderr, "flux2_dit: unexpected dtype '%s' for BF16 tensor %s\n", dtype, name);
        free(out); if (rows) *rows = 0; if (cols) *cols = 0; return NULL;
    }
    flux2_buf_track(m, out);
    return out;
}

/* Load FP8 E4M3 tensor × weight_scale → F32. */
static float *flux2_load_fp8(flux2_dit_model *m, st_context *st,
                              const char *wname, const char *sname,
                              int *rows, int *cols) {
    flux2_fp8_lut_init();
    int widx = safetensors_find(st, wname);
    if (widx < 0) { if (rows) *rows = 0; if (cols) *cols = 0; return NULL; }
    const uint64_t *sh = safetensors_shape(st, widx);
    int nd = safetensors_ndims(st, widx);
    int r = (nd >= 2) ? (int)sh[0] : 1;
    int c = (nd >= 2) ? (int)sh[1] : (int)sh[0];
    if (rows) *rows = r;
    if (cols) *cols = c;
    int n = r * c;

    float scale = 1.0f;
    int sidx = safetensors_find(st, sname);
    if (sidx >= 0) {
        const float *sp = (const float *)safetensors_data(st, sidx);
        scale = sp[0];
    }

    float *out = (float *)malloc((size_t)n * sizeof(float));
    const uint8_t *src = (const uint8_t *)safetensors_data(st, widx);
    for (int i = 0; i < n; i++) out[i] = g_fp8e4m3_lut[src[i]] * scale;
    flux2_buf_track(m, out);
    return out;
}

static flux2_mat flux2_mat_bf16(flux2_dit_model *m, st_context *st, const char *name) {
    flux2_mat mat = {0};
    mat.w = flux2_load_bf16(m, st, name, &mat.rows, &mat.cols);
    return mat;
}

static flux2_mat flux2_mat_fp8(flux2_dit_model *m, st_context *st,
                                const char *wname, const char *sname) {
    flux2_mat mat = {0};
    mat.w = flux2_load_fp8(m, st, wname, sname, &mat.rows, &mat.cols);
    return mat;
}

/* Auto-detect dtype: FP8 E4M3 → dequant with scale, BF16/F32 → direct load. */
static flux2_mat flux2_mat_auto(flux2_dit_model *m, st_context *st,
                                 const char *wname, const char *sname) {
    int idx = safetensors_find(st, wname);
    if (idx < 0) { flux2_mat mat = {0}; return mat; }
    const char *dtype = safetensors_dtype(st, idx);
    if (strcmp(dtype, "F8_E4M3") == 0)
        return flux2_mat_fp8(m, st, wname, sname);
    else
        return flux2_mat_bf16(m, st, wname);
}

/* ---- Load ---- */

flux2_dit_model *flux2_dit_load_safetensors(const char *path) {
    fprintf(stderr, "flux2_dit: loading %s\n", path);

    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "flux2_dit: failed to open %s\n", path); return NULL; }

    flux2_dit_model *m = (flux2_dit_model *)calloc(1, sizeof(flux2_dit_model));
    m->dump_dblk = -1;
    m->dump_sblk = -1;

    /* Detect dims from key tensor shapes */
    {
        int idx = safetensors_find(st, "img_in.weight");
        if (idx >= 0) {
            const uint64_t *sh = safetensors_shape(st, idx);
            m->hidden_dim = (int)sh[0];
            m->patch_in_channels = (int)sh[1];
        } else {
            m->hidden_dim = FLUX2_HIDDEN_DIM;
            m->patch_in_channels = FLUX2_PATCH_IN_CHANNELS;
        }
        idx = safetensors_find(st, "txt_in.weight");
        if (idx >= 0) m->txt_dim = (int)safetensors_shape(st, idx)[1];
        else m->txt_dim = FLUX2_TXT_DIM;

        /* n_heads from Q norm shape: head_dim = norm.scale size */
        m->head_dim = FLUX2_HEAD_DIM;
        m->n_heads  = m->hidden_dim / m->head_dim;

        /* n_ff from img_mlp.0.weight rows / 2 */
        idx = safetensors_find(st, "double_blocks.0.img_mlp.0.weight");
        if (idx >= 0) m->n_ff = (int)safetensors_shape(st, idx)[0] / 2;
        else m->n_ff = FLUX2_N_FF;

        /* Count blocks */
        m->n_double_blocks = 0;
        m->n_single_blocks = 0;
        for (int i = 0; i < st->n_tensors; i++) {
            const char *nm = safetensors_name(st, i);
            if (strncmp(nm, "double_blocks.", 14) == 0) {
                int b = atoi(nm + 14);
                if (b + 1 > m->n_double_blocks) m->n_double_blocks = b + 1;
            } else if (strncmp(nm, "single_blocks.", 14) == 0) {
                int b = atoi(nm + 14);
                if (b + 1 > m->n_single_blocks) m->n_single_blocks = b + 1;
            }
        }
        if (!m->n_double_blocks) m->n_double_blocks = FLUX2_N_DOUBLE_BLOCKS;
        if (!m->n_single_blocks) m->n_single_blocks = FLUX2_N_SINGLE_BLOCKS;
    }

    fprintf(stderr, "flux2_dit: H=%d heads=%d dblk=%d sblk=%d patch_in=%d txt_dim=%d n_ff=%d\n",
            m->hidden_dim, m->n_heads, m->n_double_blocks, m->n_single_blocks,
            m->patch_in_channels, m->txt_dim, m->n_ff);

    /* Global embedders */
    m->img_in   = flux2_mat_bf16(m, st, "img_in.weight");
    m->img_in_b = flux2_load_bf16(m, st, "img_in.bias", NULL, NULL);
    m->txt_in   = flux2_mat_bf16(m, st, "txt_in.weight");
    m->txt_in_b = flux2_load_bf16(m, st, "txt_in.bias", NULL, NULL);
    m->time_in_lin1 = flux2_mat_bf16(m, st, "time_in.in_layer.weight");
    m->time_in_lin1_b = flux2_load_bf16(m, st, "time_in.in_layer.bias", NULL, NULL);
    m->time_in_lin2 = flux2_mat_bf16(m, st, "time_in.out_layer.weight");
    m->time_in_lin2_b = flux2_load_bf16(m, st, "time_in.out_layer.bias", NULL, NULL);

    /* Shared global modulation */
    m->mod_img = flux2_mat_bf16(m, st, "double_stream_modulation_img.lin.weight");
    m->mod_txt = flux2_mat_bf16(m, st, "double_stream_modulation_txt.lin.weight");
    m->mod_sgl = flux2_mat_bf16(m, st, "single_stream_modulation.lin.weight");

    /* Output */
    m->out_mod  = flux2_mat_bf16(m, st, "final_layer.adaLN_modulation.1.weight");
    m->out_proj = flux2_mat_bf16(m, st, "final_layer.linear.weight");

    /* Double blocks */
    m->dblk = (flux2_double_block *)calloc(m->n_double_blocks, sizeof(flux2_double_block));
    for (int bi = 0; bi < m->n_double_blocks; bi++) {
        flux2_double_block *b = &m->dblk[bi];
        char wn[256], sn[256];

        /* img stream */
        snprintf(wn, sizeof(wn), "double_blocks.%d.img_attn.norm.query_norm.scale", bi);
        b->img.q_norm = flux2_load_bf16(m, st, wn, NULL, NULL);
        snprintf(wn, sizeof(wn), "double_blocks.%d.img_attn.norm.key_norm.scale", bi);
        b->img.k_norm = flux2_load_bf16(m, st, wn, NULL, NULL);

        snprintf(wn, sizeof(wn), "double_blocks.%d.img_attn.qkv.weight", bi);
        snprintf(sn, sizeof(sn), "double_blocks.%d.img_attn.qkv.weight_scale", bi);
        b->img.qkv = flux2_mat_auto(m, st, wn, sn);

        snprintf(wn, sizeof(wn), "double_blocks.%d.img_attn.proj.weight", bi);
        snprintf(sn, sizeof(sn), "double_blocks.%d.img_attn.proj.weight_scale", bi);
        b->img.proj = flux2_mat_auto(m, st, wn, sn);

        snprintf(wn, sizeof(wn), "double_blocks.%d.img_mlp.0.weight", bi);
        snprintf(sn, sizeof(sn), "double_blocks.%d.img_mlp.0.weight_scale", bi);
        b->img.mlp_up = flux2_mat_auto(m, st, wn, sn);

        snprintf(wn, sizeof(wn), "double_blocks.%d.img_mlp.2.weight", bi);
        snprintf(sn, sizeof(sn), "double_blocks.%d.img_mlp.2.weight_scale", bi);
        b->img.mlp_down = flux2_mat_auto(m, st, wn, sn);

        /* txt stream */
        snprintf(wn, sizeof(wn), "double_blocks.%d.txt_attn.norm.query_norm.scale", bi);
        b->txt.q_norm = flux2_load_bf16(m, st, wn, NULL, NULL);
        snprintf(wn, sizeof(wn), "double_blocks.%d.txt_attn.norm.key_norm.scale", bi);
        b->txt.k_norm = flux2_load_bf16(m, st, wn, NULL, NULL);

        snprintf(wn, sizeof(wn), "double_blocks.%d.txt_attn.qkv.weight", bi);
        snprintf(sn, sizeof(sn), "double_blocks.%d.txt_attn.qkv.weight_scale", bi);
        b->txt.qkv = flux2_mat_auto(m, st, wn, sn);

        snprintf(wn, sizeof(wn), "double_blocks.%d.txt_attn.proj.weight", bi);
        snprintf(sn, sizeof(sn), "double_blocks.%d.txt_attn.proj.weight_scale", bi);
        b->txt.proj = flux2_mat_auto(m, st, wn, sn);

        snprintf(wn, sizeof(wn), "double_blocks.%d.txt_mlp.0.weight", bi);
        snprintf(sn, sizeof(sn), "double_blocks.%d.txt_mlp.0.weight_scale", bi);
        b->txt.mlp_up = flux2_mat_auto(m, st, wn, sn);

        snprintf(wn, sizeof(wn), "double_blocks.%d.txt_mlp.2.weight", bi);
        snprintf(sn, sizeof(sn), "double_blocks.%d.txt_mlp.2.weight_scale", bi);
        b->txt.mlp_down = flux2_mat_auto(m, st, wn, sn);

        if ((bi + 1) % 2 == 0 || bi == m->n_double_blocks - 1)
            fprintf(stderr, "\r  double block %d/%d", bi + 1, m->n_double_blocks);
    }
    fprintf(stderr, "\n");

    /* Single blocks */
    m->sblk = (flux2_single_block *)calloc(m->n_single_blocks, sizeof(flux2_single_block));
    for (int bi = 0; bi < m->n_single_blocks; bi++) {
        flux2_single_block *b = &m->sblk[bi];
        char wn[256], sn[256];

        snprintf(wn, sizeof(wn), "single_blocks.%d.norm.query_norm.scale", bi);
        b->q_norm = flux2_load_bf16(m, st, wn, NULL, NULL);
        snprintf(wn, sizeof(wn), "single_blocks.%d.norm.key_norm.scale", bi);
        b->k_norm = flux2_load_bf16(m, st, wn, NULL, NULL);

        snprintf(wn, sizeof(wn), "single_blocks.%d.linear1.weight", bi);
        snprintf(sn, sizeof(sn), "single_blocks.%d.linear1.weight_scale", bi);
        b->linear1 = flux2_mat_auto(m, st, wn, sn);

        snprintf(wn, sizeof(wn), "single_blocks.%d.linear2.weight", bi);
        snprintf(sn, sizeof(sn), "single_blocks.%d.linear2.weight_scale", bi);
        b->linear2 = flux2_mat_auto(m, st, wn, sn);

        if ((bi + 1) % 5 == 0 || bi == m->n_single_blocks - 1)
            fprintf(stderr, "\r  single block %d/%d", bi + 1, m->n_single_blocks);
    }
    fprintf(stderr, "\n");

    safetensors_close(st);
    fprintf(stderr, "flux2_dit: loaded OK\n");
    return m;
}

void flux2_dit_free(flux2_dit_model *m) {
    if (!m) return;
    for (int i = 0; i < m->n_bufs; i++) free(m->bufs[i]);
    free(m->bufs);
    free(m->dblk);
    free(m->sblk);
    free(m);
}

/* ---- Compute primitives ---- */

/* GEMM: out[n, out_dim] = in[n, in_dim] @ W[out_dim, in_dim]^T + bias[out_dim] */
static void flux2_gemm(float *out, const float *in, int n, int in_dim,
                       const flux2_mat *W, const float *bias) {
    int W_rows = W->rows, W_cols = W->cols;
    const float *W_w = W->w;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) num_threads(flux2_nthreads_g) schedule(static)
#endif
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < W_rows; j++) {
            const float *x = in + (size_t)i * in_dim;
            const float *w = W_w + (size_t)j * W_cols;
            float s = bias ? bias[j] : 0.0f;
#ifdef __AVX2__
            int k = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; k + 7 < W_cols; k += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x+k), _mm256_loadu_ps(w+k), acc);
            float tmp[8]; _mm256_storeu_ps(tmp, acc);
            for (int t = 0; t < 8; t++) s += tmp[t];
            for (; k < W_cols; k++) s += x[k] * w[k];
#else
            for (int k = 0; k < W_cols; k++) s += x[k] * w[k];
#endif
            out[(size_t)i * W_rows + j] = s;
        }
    }
}

/* Layer normalization (no affine parameters). */
static void flux2_layernorm(float *x, int n, int d, float eps) {
    for (int i = 0; i < n; i++) {
        float *r = x + (size_t)i * d;
        float mean = 0.0f, var = 0.0f;
        for (int j = 0; j < d; j++) mean += r[j];
        mean /= (float)d;
        for (int j = 0; j < d; j++) { float v = r[j] - mean; var += v*v; }
        float inv = 1.0f / sqrtf(var / (float)d + eps);
        for (int j = 0; j < d; j++) r[j] = (r[j] - mean) * inv;
    }
}

/* Per-head RMS normalization with scale. x: [n_tok, n_heads, head_dim] */
static void flux2_rms_norm_heads(float *x, const float *scale,
                                  int n_tok, int n_heads, int head_dim) {
    float eps = 1e-5f;
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = x + ((size_t)t * n_heads + h) * head_dim;
            float rms = 0.0f;
            for (int d = 0; d < head_dim; d++) rms += v[d] * v[d];
            rms = 1.0f / sqrtf(rms / (float)head_dim + eps);
            for (int d = 0; d < head_dim; d++) v[d] *= rms * (scale ? scale[d] : 1.0f);
        }
    }
}

/* Sinusoidal timestep embedding → [256]. */
static void flux2_timestep_emb(float *out, float t, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i / (float)half * logf(10000.0f));
        out[i]        = cosf(t * freq);
        out[i + half] = sinf(t * freq);
    }
}

/* SiLU activation in-place. */
static void flux2_silu(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] *= 1.0f / (1.0f + expf(-x[i]));
}

/* SwiGLU: gate+up is [n, 2*n_ff] interleaved as [gate(n_ff), up(n_ff)].
 * out[n, n_ff] = silu(gate) * up */
static void flux2_swiglu(float *out, const float *gate_up, int n, int n_ff) {
    for (int i = 0; i < n; i++) {
        const float *gu = gate_up + (size_t)i * (2 * n_ff);
        float *o = out + (size_t)i * n_ff;
        for (int j = 0; j < n_ff; j++) {
            float g = gu[j];
            float u = gu[j + n_ff];
            o[j] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

/* Apply 1D RoPE to one 32-dim axis chunk of a head vector.
 * x: pointer to axis_dim floats, n_pairs = axis_dim/2, pos: scalar position. */
static void flux2_rope_1d_chunk(float *x, int n_pairs, float pos, float theta) {
    for (int j = 0; j < n_pairs; j++) {
        float freq = 1.0f / powf(theta, (float)j / (float)n_pairs);
        float angle = pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float x0 = x[2*j], x1 = x[2*j+1];
        x[2*j]   = x0 * c - x1 * s;
        x[2*j+1] = x0 * s + x1 * c;
    }
}

/* 2D RoPE for image tokens using axes_dim=(32,32,32,32), theta=2000.
 * img_ids per token: (t=0, row, col, l=0)
 * Axis 0 (dims  0..31): pos=0 → identity (skip)
 * Axis 1 (dims 32..63): pos=row → row rotation
 * Axis 2 (dims 64..95): pos=col → col rotation
 * Axis 3 (dims 96..127): pos=0 → identity (skip)
 * x: [n_tok, n_heads, head_dim] — applies in-place. */
static void flux2_rope_img(float *x, int n_tok, int n_heads, int head_dim,
                            int lat_h_patches, int lat_w_patches, float theta) {
    int axis_dim = head_dim / 4;   /* 32 dims per axis */
    int n_pairs  = axis_dim / 2;   /* 16 rotation pairs per axis */
    for (int t = 0; t < n_tok; t++) {
        int row = t / lat_w_patches;
        int col = t % lat_w_patches;
        for (int h = 0; h < n_heads; h++) {
            float *v = x + ((size_t)t * n_heads + h) * head_dim;
            /* Axis 0: identity (pos=0, skip) */
            /* Axis 1 (dims 32..63): row */
            flux2_rope_1d_chunk(v + axis_dim,     n_pairs, (float)row, theta);
            /* Axis 2 (dims 64..95): col */
            flux2_rope_1d_chunk(v + 2 * axis_dim, n_pairs, (float)col, theta);
            /* Axis 3: identity (pos=0, skip) */
        }
    }
}

/* 1D RoPE for text tokens using axes_dim=(32,32,32,32), theta=2000.
 * txt_ids per token: (t=0, h=0, w=0, l=seq_pos)
 * Axes 0,1,2 (dims 0..95): pos=0 → identity (skip)
 * Axis 3 (dims 96..127): pos=seq_pos → sequence rotation
 * x: [n_tok, n_heads, head_dim] — applies in-place. */
static void flux2_rope_txt(float *x, int n_tok, int n_heads, int head_dim, float theta) {
    int axis_dim = head_dim / 4;   /* 32 dims per axis */
    int n_pairs  = axis_dim / 2;   /* 16 rotation pairs per axis */
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = x + ((size_t)t * n_heads + h) * head_dim;
            /* Axes 0,1,2: identity (pos=0, skip) */
            /* Axis 3 (dims 96..127): sequence position */
            flux2_rope_1d_chunk(v + 3 * axis_dim, n_pairs, (float)t, theta);
        }
    }
}

/* Softmax in-place over last dim. x: [n_heads, n_seq]. */
static void flux2_softmax(float *x, int n_heads, int n_seq) {
    for (int h = 0; h < n_heads; h++) {
        float *r = x + (size_t)h * n_seq;
        float mx = r[0];
        for (int i = 1; i < n_seq; i++) if (r[i] > mx) mx = r[i];
        float sum = 0.0f;
        for (int i = 0; i < n_seq; i++) { r[i] = expf(r[i] - mx); sum += r[i]; }
        for (int i = 0; i < n_seq; i++) r[i] /= sum;
    }
}

/* Multi-head attention.
 * q: [n_q, n_heads, head_dim], k/v: [n_kv, n_heads, head_dim]
 * out: [n_q, n_heads * head_dim] */
static void flux2_mha(float *out, const float *q, const float *k, const float *v,
                      int n_q, int n_kv, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    float *attn = (float *)malloc((size_t)n_heads * n_kv * sizeof(float));

#ifdef _OPENMP
    #pragma omp parallel for num_threads(flux2_nthreads_g) schedule(static)
#endif
    for (int h = 0; h < n_heads; h++) {
        /* Q @ K^T */
        for (int qi = 0; qi < n_q; qi++) {
            const float *qr = q + ((size_t)qi * n_heads + h) * head_dim;
            for (int ki = 0; ki < n_kv; ki++) {
                const float *kr = k + ((size_t)ki * n_heads + h) * head_dim;
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) s += qr[d] * kr[d];
                attn[(size_t)h * n_kv + ki] = s * scale;
            }
            /* Softmax over kv dim */
            float *a = attn + (size_t)h * n_kv;
            float mx = a[0];
            for (int i = 1; i < n_kv; i++) if (a[i] > mx) mx = a[i];
            float sum = 0.0f;
            for (int i = 0; i < n_kv; i++) { a[i] = expf(a[i] - mx); sum += a[i]; }
            for (int i = 0; i < n_kv; i++) a[i] /= sum;
            /* A @ V */
            float *o = out + ((size_t)qi * n_heads + h) * head_dim;
            for (int d = 0; d < head_dim; d++) {
                float s = 0.0f;
                for (int vi = 0; vi < n_kv; vi++)
                    s += a[vi] * v[((size_t)vi * n_heads + h) * head_dim + d];
                o[d] = s;
            }
        }
    }
    free(attn);
}

/* Apply adaLN modulation: scale and shift a layer-normed tensor.
 * mod: [shift(H), scale(H)] or offset into a larger modulation vector.
 * x: [n, H] in/out. */
static void flux2_apply_mod(float *x, int n, int H,
                             const float *shift, const float *scale) {
    for (int i = 0; i < n; i++) {
        float *r = x + (size_t)i * H;
        for (int j = 0; j < H; j++)
            r[j] = r[j] * (1.0f + scale[j]) + shift[j];
    }
}

/* ---- Forward pass ---- */

void flux2_dit_forward(float *out,
                       const float *img_tokens, int n_img,
                       const float *txt_tokens, int n_txt,
                       float timestep,
                       flux2_dit_model *m, int n_threads) {
    flux2_nthreads_g = (n_threads > 0) ? n_threads : 1;
    int H      = m->hidden_dim;
    int nH     = m->n_heads;
    int hd     = m->head_dim;
    int n_ff   = m->n_ff;

    /* 1. Timestep embedding — FLUX uses time_factor=1000: sigma in [0,1] → t in [0,1000] */
    float t_raw[256];
    flux2_timestep_emb(t_raw, timestep * 1000.0f, 256);

    float *temb = (float *)malloc((size_t)H * sizeof(float));
    float *tmp  = (float *)malloc((size_t)H * sizeof(float));
    /* in_layer */
    flux2_gemm(temb, t_raw, 1, 256, &m->time_in_lin1, m->time_in_lin1_b);
    flux2_silu(temb, H);
    /* out_layer */
    flux2_gemm(tmp, temb, 1, H, &m->time_in_lin2, m->time_in_lin2_b);
    memcpy(temb, tmp, (size_t)H * sizeof(float));

    /* 2. Project img and txt tokens */
    float *img = (float *)malloc((size_t)n_img * H * sizeof(float));
    float *txt = (float *)malloc((size_t)n_txt * H * sizeof(float));
    flux2_gemm(img, img_tokens, n_img, m->patch_in_channels, &m->img_in, m->img_in_b);
    flux2_gemm(txt, txt_tokens, n_txt, m->txt_dim, &m->txt_in, m->txt_in_b);

    /* 3. Compute global modulations: SiLU(temb) → mod */
    float *temb_silu = (float *)malloc((size_t)H * sizeof(float));
    memcpy(temb_silu, temb, (size_t)H * sizeof(float));
    flux2_silu(temb_silu, H);

    float *mod_img_vec = (float *)malloc((size_t)(6 * H) * sizeof(float));
    float *mod_txt_vec = (float *)malloc((size_t)(6 * H) * sizeof(float));
    float *mod_sgl_vec = (float *)malloc((size_t)(3 * H) * sizeof(float));
    flux2_gemm(mod_img_vec, temb_silu, 1, H, &m->mod_img, NULL);
    flux2_gemm(mod_txt_vec, temb_silu, 1, H, &m->mod_txt, NULL);
    flux2_gemm(mod_sgl_vec, temb_silu, 1, H, &m->mod_sgl, NULL);
    free(temb_silu);

    /* Modulation vectors:
     * mod_img: [shift_attn(H), scale_attn(H), gate_attn(H), shift_ffn(H), scale_ffn(H), gate_ffn(H)]
     * mod_txt: same layout
     * mod_sgl: [shift(H), scale(H), gate(H)] */
    const float *mi_shift_a = mod_img_vec;
    const float *mi_scale_a = mod_img_vec + H;
    const float *mi_gate_a  = mod_img_vec + 2*H;
    const float *mi_shift_f = mod_img_vec + 3*H;
    const float *mi_scale_f = mod_img_vec + 4*H;
    const float *mi_gate_f  = mod_img_vec + 5*H;

    const float *mt_shift_a = mod_txt_vec;
    const float *mt_scale_a = mod_txt_vec + H;
    const float *mt_gate_a  = mod_txt_vec + 2*H;
    const float *mt_shift_f = mod_txt_vec + 3*H;
    const float *mt_scale_f = mod_txt_vec + 4*H;
    const float *mt_gate_f  = mod_txt_vec + 5*H;

    /* Scratch buffers */
    float *img_norm = (float *)malloc((size_t)n_img * H * sizeof(float));
    float *txt_norm = (float *)malloc((size_t)n_txt * H * sizeof(float));
    float *img_qkv  = (float *)malloc((size_t)n_img * 3 * H * sizeof(float));
    float *txt_qkv  = (float *)malloc((size_t)n_txt * 3 * H * sizeof(float));
    float *img_attn_out = (float *)malloc((size_t)n_img * H * sizeof(float));
    float *txt_attn_out = (float *)malloc((size_t)n_txt * H * sizeof(float));
    float *img_gate_up  = (float *)malloc((size_t)n_img * 2 * n_ff * sizeof(float));
    float *txt_gate_up  = (float *)malloc((size_t)n_txt * 2 * n_ff * sizeof(float));
    float *img_mlp_out  = (float *)malloc((size_t)n_img * n_ff * sizeof(float));
    float *txt_mlp_out  = (float *)malloc((size_t)n_txt * n_ff * sizeof(float));
    float *img_proj_out = (float *)malloc((size_t)n_img * H * sizeof(float));
    float *txt_proj_out = (float *)malloc((size_t)n_txt * H * sizeof(float));

    /* Determine spatial grid size for RoPE */
    /* n_img = (lat_h/ps) * (lat_w/ps); assume square for now */
    int grid_size = (int)sqrtf((float)n_img);
    if (grid_size * grid_size != n_img) grid_size = n_img; /* fallback: 1D */
    int lat_h_p = grid_size, lat_w_p = n_img / grid_size;

    /* Pre-allocated de-interleaved Q/K/V buffers (reused across block iterations).
     * GEMM outputs [n, 3H] interleaved per token; we de-interleave to [n, H] each. */
    int n_tot_all = n_img + n_txt;
    float *img_q_s = (float *)malloc((size_t)n_img * H * sizeof(float));
    float *img_k_s = (float *)malloc((size_t)n_img * H * sizeof(float));
    float *img_v_s = (float *)malloc((size_t)n_img * H * sizeof(float));
    float *txt_q_s = (float *)malloc((size_t)n_txt * H * sizeof(float));
    float *txt_k_s = (float *)malloc((size_t)n_txt * H * sizeof(float));
    float *txt_v_s = (float *)malloc((size_t)n_txt * H * sizeof(float));
    /* Single-stream: lin1 outputs [n_tot, 3H+2n_ff] interleaved */
    float *jq_s = (float *)malloc((size_t)n_tot_all * H * sizeof(float));
    float *jk_s = (float *)malloc((size_t)n_tot_all * H * sizeof(float));
    float *jv_s = (float *)malloc((size_t)n_tot_all * H * sizeof(float));
    float *jg_s = (float *)malloc((size_t)n_tot_all * 2 * n_ff * sizeof(float));

    /* Dump macros */
#define DUMP_DBL(name, data, n) do { \
    if (m->dump_fn && bi == m->dump_dblk) \
        m->dump_fn(name, data, n, m->dump_ctx); \
} while(0)
#define DUMP_SGL(name, data, n) do { \
    if (m->dump_fn && bi == m->dump_sblk) \
        m->dump_fn(name, data, n, m->dump_ctx); \
} while(0)
#define DUMP_GLOBAL(name, data, n) do { \
    if (m->dump_fn) m->dump_fn(name, data, n, m->dump_ctx); \
} while(0)

    /* Dump global checkpoints */
    DUMP_GLOBAL("temb", temb, H);
    DUMP_GLOBAL("img_projected", img, n_img * H);
    DUMP_GLOBAL("txt_projected", txt, n_txt * H);
    DUMP_GLOBAL("mod_img_vec", mod_img_vec, 6 * H);
    DUMP_GLOBAL("mod_txt_vec", mod_txt_vec, 6 * H);
    DUMP_GLOBAL("mod_sgl_vec", mod_sgl_vec, 3 * H);

    /* ---- Double-stream blocks ---- */
    for (int bi = 0; bi < m->n_double_blocks; bi++) {
        flux2_double_block *b = &m->dblk[bi];

        /* --- IMG stream --- */
        /* Pre-attn norm */
        memcpy(img_norm, img, (size_t)n_img * H * sizeof(float));
        flux2_layernorm(img_norm, n_img, H, 1e-6f);
        flux2_apply_mod(img_norm, n_img, H, mi_shift_a, mi_scale_a);
        DUMP_DBL("img_adaln", img_norm, n_img * H);

        /* QKV projection: output [n_img, 3H] interleaved per token → de-interleave */
        flux2_gemm(img_qkv, img_norm, n_img, H, &b->img.qkv, NULL);
        for (int t = 0; t < n_img; t++) {
            const float *src = img_qkv + (size_t)t * 3 * H;
            memcpy(img_q_s + (size_t)t * H, src,         H * sizeof(float));
            memcpy(img_k_s + (size_t)t * H, src + H,     H * sizeof(float));
            memcpy(img_v_s + (size_t)t * H, src + 2 * H, H * sizeof(float));
        }
        float *img_q = img_q_s, *img_k = img_k_s, *img_v = img_v_s;

        /* Per-head Q/K norm + RoPE */
        flux2_rms_norm_heads(img_q, b->img.q_norm, n_img, nH, hd);
        flux2_rms_norm_heads(img_k, b->img.k_norm, n_img, nH, hd);
        /* Reshape for rope: treat img_q as [n_img, nH, hd] */
        flux2_rope_img(img_q, n_img, nH, hd, lat_h_p, lat_w_p, FLUX2_ROPE_THETA);
        flux2_rope_img(img_k, n_img, nH, hd, lat_h_p, lat_w_p, FLUX2_ROPE_THETA);
        DUMP_DBL("img_q_normed", img_q, n_img * H);

        /* --- TXT stream (parallel) --- */
        memcpy(txt_norm, txt, (size_t)n_txt * H * sizeof(float));
        flux2_layernorm(txt_norm, n_txt, H, 1e-6f);
        flux2_apply_mod(txt_norm, n_txt, H, mt_shift_a, mt_scale_a);
        DUMP_DBL("txt_adaln", txt_norm, n_txt * H);

        flux2_gemm(txt_qkv, txt_norm, n_txt, H, &b->txt.qkv, NULL);
        for (int t = 0; t < n_txt; t++) {
            const float *src = txt_qkv + (size_t)t * 3 * H;
            memcpy(txt_q_s + (size_t)t * H, src,         H * sizeof(float));
            memcpy(txt_k_s + (size_t)t * H, src + H,     H * sizeof(float));
            memcpy(txt_v_s + (size_t)t * H, src + 2 * H, H * sizeof(float));
        }
        float *txt_q = txt_q_s, *txt_k = txt_k_s, *txt_v = txt_v_s;

        flux2_rms_norm_heads(txt_q, b->txt.q_norm, n_txt, nH, hd);
        flux2_rms_norm_heads(txt_k, b->txt.k_norm, n_txt, nH, hd);
        flux2_rope_txt(txt_q, n_txt, nH, hd, FLUX2_ROPE_THETA);
        flux2_rope_txt(txt_k, n_txt, nH, hd, FLUX2_ROPE_THETA);

        /* Joint attention: concat K,V across img+txt, separate Q per stream */
        int n_tot = n_img + n_txt;
        float *joint_k = (float *)malloc((size_t)n_tot * H * sizeof(float));
        float *joint_v = (float *)malloc((size_t)n_tot * H * sizeof(float));
        /* K: [img_k || txt_k] interleaved per head-token */
        memcpy(joint_k,                   img_k, (size_t)n_img * H * sizeof(float));
        memcpy(joint_k + (size_t)n_img*H, txt_k, (size_t)n_txt * H * sizeof(float));
        memcpy(joint_v,                   img_v, (size_t)n_img * H * sizeof(float));
        memcpy(joint_v + (size_t)n_img*H, txt_v, (size_t)n_txt * H * sizeof(float));

        /* Attention: img Q attends over joint K/V */
        float *img_attn_h = (float *)malloc((size_t)n_img * nH * hd * sizeof(float));
        float *txt_attn_h = (float *)malloc((size_t)n_txt * nH * hd * sizeof(float));
        flux2_mha(img_attn_h, img_q, joint_k, joint_v, n_img, n_tot, nH, hd);
        flux2_mha(txt_attn_h, txt_q, joint_k, joint_v, n_txt, n_tot, nH, hd);

        flux2_gemm(img_attn_out, img_attn_h, n_img, H, &b->img.proj, NULL);
        flux2_gemm(txt_attn_out, txt_attn_h, n_txt, H, &b->txt.proj, NULL);
        free(img_attn_h); free(txt_attn_h); free(joint_k); free(joint_v);

        /* Gated residual add */
        for (int i = 0; i < n_img * H; i++) img[i] += mi_gate_a[i % H] * img_attn_out[i];
        for (int i = 0; i < n_txt * H; i++) txt[i] += mt_gate_a[i % H] * txt_attn_out[i];
        DUMP_DBL("img_after_attn", img, n_img * H);
        DUMP_DBL("txt_after_attn", txt, n_txt * H);

        /* FFN img */
        memcpy(img_norm, img, (size_t)n_img * H * sizeof(float));
        flux2_layernorm(img_norm, n_img, H, 1e-6f);
        flux2_apply_mod(img_norm, n_img, H, mi_shift_f, mi_scale_f);
        flux2_gemm(img_gate_up, img_norm, n_img, H, &b->img.mlp_up, NULL);
        flux2_swiglu(img_mlp_out, img_gate_up, n_img, n_ff);
        flux2_gemm(img_proj_out, img_mlp_out, n_img, n_ff, &b->img.mlp_down, NULL);
        for (int i = 0; i < n_img * H; i++) img[i] += mi_gate_f[i % H] * img_proj_out[i];
        DUMP_DBL("img_after_mlp", img, n_img * H);

        /* FFN txt */
        memcpy(txt_norm, txt, (size_t)n_txt * H * sizeof(float));
        flux2_layernorm(txt_norm, n_txt, H, 1e-6f);
        flux2_apply_mod(txt_norm, n_txt, H, mt_shift_f, mt_scale_f);
        flux2_gemm(txt_gate_up, txt_norm, n_txt, H, &b->txt.mlp_up, NULL);
        flux2_swiglu(txt_mlp_out, txt_gate_up, n_txt, n_ff);
        flux2_gemm(txt_proj_out, txt_mlp_out, n_txt, n_ff, &b->txt.mlp_down, NULL);
        for (int i = 0; i < n_txt * H; i++) txt[i] += mt_gate_f[i % H] * txt_proj_out[i];
        DUMP_DBL("txt_after_mlp", txt, n_txt * H);
    }

    /* ---- Single-stream blocks ---- */
    int n_tot = n_txt + n_img;
    /* Concat [txt, img] tokens */
    float *joint = (float *)malloc((size_t)n_tot * H * sizeof(float));
    memcpy(joint,                   txt, (size_t)n_txt * H * sizeof(float));
    memcpy(joint + (size_t)n_txt*H, img, (size_t)n_img * H * sizeof(float));

    float *joint_norm = (float *)malloc((size_t)n_tot * H * sizeof(float));
    /* linear1 output: [n_tot, 9H] = QKV(3H) + gate_up(6H) */
    int lin1_out = 3 * H + 2 * n_ff;  /* = 9H for mlp_ratio=3 */
    float *lin1_buf = (float *)malloc((size_t)n_tot * lin1_out * sizeof(float));
    /* linear2 input: [n_tot, H+n_ff=4H] — concatenation of attn_out+mlp_out */
    int lin2_in = H + n_ff;
    float *lin2_in_buf = (float *)malloc((size_t)n_tot * lin2_in * sizeof(float));
    float *lin2_out    = (float *)malloc((size_t)n_tot * H * sizeof(float));

    const float *ms_shift = mod_sgl_vec;
    const float *ms_scale = mod_sgl_vec + H;
    const float *ms_gate  = mod_sgl_vec + 2*H;

    for (int bi = 0; bi < m->n_single_blocks; bi++) {
        flux2_single_block *b = &m->sblk[bi];

        /* Pre-norm + modulate */
        memcpy(joint_norm, joint, (size_t)n_tot * H * sizeof(float));
        flux2_layernorm(joint_norm, n_tot, H, 1e-6f);
        flux2_apply_mod(joint_norm, n_tot, H, ms_shift, ms_scale);
        DUMP_SGL("adaln", joint_norm, n_tot * H);

        /* Linear1: [9H, H] → split Q[3H slice → nH,hd], K, V, gate, up */
        flux2_gemm(lin1_buf, joint_norm, n_tot, H, &b->linear1, NULL);

        /* De-interleave [n_tot, 3H+2n_ff] per-token → separate [n_tot, H/2n_ff] buffers */
        for (int t = 0; t < n_tot; t++) {
            const float *src = lin1_buf + (size_t)t * lin1_out;
            memcpy(jq_s + (size_t)t * H,             src,         H * sizeof(float));
            memcpy(jk_s + (size_t)t * H,             src + H,     H * sizeof(float));
            memcpy(jv_s + (size_t)t * H,             src + 2 * H, H * sizeof(float));
            memcpy(jg_s + (size_t)t * 2 * n_ff,      src + 3 * H, 2 * n_ff * sizeof(float));
        }
        float *jq = jq_s, *jk = jk_s, *jv = jv_s, *jg = jg_s;

        /* Per-head Q/K norm + RoPE on combined sequence */
        flux2_rms_norm_heads(jq, b->q_norm, n_tot, nH, hd);
        flux2_rms_norm_heads(jk, b->k_norm, n_tot, nH, hd);
        /* txt tokens at front [0..n_txt), img tokens after */
        flux2_rope_txt(jq,            n_txt, nH, hd, FLUX2_ROPE_THETA);
        flux2_rope_txt(jk,            n_txt, nH, hd, FLUX2_ROPE_THETA);
        flux2_rope_img(jq + (size_t)n_txt * H, n_img, nH, hd,
                       lat_h_p, lat_w_p, FLUX2_ROPE_THETA);
        flux2_rope_img(jk + (size_t)n_txt * H, n_img, nH, hd,
                       lat_h_p, lat_w_p, FLUX2_ROPE_THETA);
        DUMP_SGL("q_normed", jq, n_tot * H);

        /* Self-attention over all tokens */
        float *attn_h = (float *)malloc((size_t)n_tot * nH * hd * sizeof(float));
        flux2_mha(attn_h, jq, jk, jv, n_tot, n_tot, nH, hd);

        /* Parallel MLP: SwiGLU(gate, up) */
        float *mlp_h = (float *)malloc((size_t)n_tot * n_ff * sizeof(float));
        flux2_swiglu(mlp_h, jg, n_tot, n_ff);

        /* Concat attn_out(H) + mlp_out(n_ff) → linear2 input [H+n_ff] */
        for (int t = 0; t < n_tot; t++) {
            memcpy(lin2_in_buf + (size_t)t * lin2_in,
                   attn_h + (size_t)t * H, (size_t)H * sizeof(float));
            memcpy(lin2_in_buf + (size_t)t * lin2_in + H,
                   mlp_h + (size_t)t * n_ff, (size_t)n_ff * sizeof(float));
        }
        free(attn_h); free(mlp_h);

        flux2_gemm(lin2_out, lin2_in_buf, n_tot, lin2_in, &b->linear2, NULL);

        /* Gated residual */
        for (int i = 0; i < n_tot * H; i++) joint[i] += ms_gate[i % H] * lin2_out[i];
        DUMP_SGL("after_block", joint, n_tot * H);
    }

    /* ---- Output ---- */
    /* Keep only img portion (discard txt) */
    float *img_out = joint + (size_t)n_txt * H;

    /* final_layer: adaLN + linear */
    float *img_final = (float *)malloc((size_t)n_img * H * sizeof(float));
    memcpy(img_final, img_out, (size_t)n_img * H * sizeof(float));
    flux2_layernorm(img_final, n_img, H, 1e-6f);

    /* adaLN_modulation: SiLU(temb) → [2H] */
    float *temb_silu2 = (float *)malloc((size_t)H * sizeof(float));
    memcpy(temb_silu2, temb, (size_t)H * sizeof(float));
    flux2_silu(temb_silu2, H);
    float *out_mod_vec = (float *)malloc((size_t)(2*H) * sizeof(float));
    flux2_gemm(out_mod_vec, temb_silu2, 1, H, &m->out_mod, NULL);
    free(temb_silu2);

    flux2_apply_mod(img_final, n_img, H, out_mod_vec, out_mod_vec + H);
    free(out_mod_vec);

    /* Final linear: [patch_in, H] → out[n_img, patch_in] */
    flux2_gemm(out, img_final, n_img, H, &m->out_proj, NULL);
    DUMP_GLOBAL("dit_output", out, n_img * m->patch_in_channels);

#undef DUMP_DBL
#undef DUMP_SGL
#undef DUMP_GLOBAL

    /* Cleanup */
    free(img_final);
    free(joint); free(joint_norm); free(lin1_buf);
    free(lin2_in_buf); free(lin2_out);
    free(img_norm); free(txt_norm);
    free(img_qkv); free(txt_qkv);
    free(img_q_s); free(img_k_s); free(img_v_s);
    free(txt_q_s); free(txt_k_s); free(txt_v_s);
    free(jq_s); free(jk_s); free(jv_s); free(jg_s);
    free(img_attn_out); free(txt_attn_out);
    free(img_gate_up); free(txt_gate_up);
    free(img_mlp_out); free(txt_mlp_out);
    free(img_proj_out); free(txt_proj_out);
    free(mod_img_vec); free(mod_txt_vec); free(mod_sgl_vec);
    free(temb); free(tmp);
    free(img); free(txt);
}

#endif /* FLUX2_DIT_IMPLEMENTATION */
#endif /* FLUX2_KLEIN_DIT_H */
