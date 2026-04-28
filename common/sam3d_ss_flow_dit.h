/*
 * sam3d_ss_flow_dit.h — stage-1 Sparse-Structure Flow DiT for sam-3d-objects.
 *
 * pytorch: model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py
 *          + modules/transformer/modulated.py::MOTModulatedTransformerCrossBlock
 *          + modules/attention/modules.py::MOTMultiHeadSelfAttention
 *
 * Architecture (from ss_generator.yaml):
 *   model_channels = 1024, num_blocks = 24, num_heads = 16, head_dim = 64,
 *   mlp_ratio = 4, pe_mode = "ape", qk_rms_norm = true,
 *   is_shortcut_model = true, share_mod = false.
 *
 * Multi-modality-of-transformer (MOT):
 *   Five INPUT modalities (yaml latent_mapping):
 *     shape              — in=8,  token_len=4096 (16³ grid, patch_size=1)
 *     6drotation_normalized — in=6,  token_len=1
 *     translation        — in=3,  token_len=1
 *     scale              — in=3,  token_len=1
 *     translation_scale  — in=1,  token_len=1
 *   latent_share_transformer merges the last four into one stream
 *   (`6drotation_normalized` = concat of [6drotation_normalized,
 *    translation, scale, translation_scale]), so inside the transformer
 *   there are TWO streams:
 *     shape               (N = 4096, D = 1024)
 *     6drotation_normalized (N = 4, D = 1024) — the "pose" stream
 *
 * Per-block layout:
 *   adaLN_modulation: SiLU + Linear(D → 6*D)             — SHARED (not per-modality)
 *   norm1[m]:  LayerNorm(D, elementwise_affine=False)    — per-modality, no learned w/b
 *   self_attn:
 *     to_qkv[m]: Linear(D → 3*D, bias=true)              — per-modality
 *     q_rms_norm[m], k_rms_norm[m]: MultiHeadRMSNorm (gamma [H, head_dim])
 *     to_out[m]: Linear(D → D)                           — per-modality
 *   norm2[m]:  LayerNorm(D, elementwise_affine=True)     — per-modality, LEARNED w/b
 *   cross_attn[m]:  MultiHeadAttention(D, ctx=D, full)   — per-modality
 *     to_q[m]: Linear(D → D), to_kv[m]: Linear(D → 2D), to_out[m]: Linear(D → D)
 *   norm3[m]:  LayerNorm(D, elementwise_affine=False)    — per-modality, no learned w/b
 *   mlp[m]:    FeedForwardNet(D, mlp_ratio)              — per-modality
 *     mlp.0: Linear(D → 4D), act=GELU (exact, approximate=none), mlp.2: Linear(4D → D)
 *
 * Attention segmentation (MOT, inference):
 *   "protect_modality_list" = ["shape"]. `shape` self-attends to shape only
 *   (4096→4096). `6drotation_normalized` attends to itself AND shape
 *   (queries 4 tokens attend to 4+4096 keys). Asymmetric: shape's K/V
 *   never sees the pose tokens, but pose's K/V includes shape.
 *
 * Conditioning inputs to forward:
 *   - latents_dict: {shape[4096,8], 6drotation_normalized[1,6],
 *     translation[1,3], scale[1,3], translation_scale[1,1]}
 *   - t: timestep tensor (B,); shortcut model also takes d
 *   - cond: DINOv2+PPE fused tokens (N_cond, 1024)
 *
 * t/d embedder: sinusoidal-positional(t, freq_dim=256) → Linear(256→1024)
 *   + SiLU + Linear(1024→1024). For shortcut: t_emb + d_emb combined.
 *
 * This header is header-only; call `#define SAM3D_SS_FLOW_DIT_IMPLEMENTATION`
 * in exactly one TU. Dependencies: safetensors.h, ggml_dequant.h.
 */

#ifndef SAM3D_SS_FLOW_DIT_H
#define SAM3D_SS_FLOW_DIT_H

#include "safetensors.h"
#include "ggml_dequant.h"
#include "cpu_compute.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "qtensor_utils.h"

/* 5 input modalities per YAML. In the transformer only 2 streams exist
 * (shape, and merged "pose" stream keyed as 6drotation_normalized), but
 * the latent_mapping (input/output projection) runs over all 5. */
#define SAM3D_SS_DIT_N_LATENTS 5
#define SAM3D_SS_DIT_N_STREAMS 2
#define SAM3D_SS_DIT_MAX_BLOCKS 32

typedef enum {
    SAM3D_SS_LAT_SHAPE = 0,
    SAM3D_SS_LAT_6DROT = 1,
    SAM3D_SS_LAT_TRANSLATION = 2,
    SAM3D_SS_LAT_SCALE = 3,
    SAM3D_SS_LAT_TRANSLATION_SCALE = 4
} sam3d_ss_lat_id;

/* In-transformer stream index. The 4 "pose" modalities are concatenated
 * into stream 1 in the order: 6drot, translation, scale, translation_scale. */
typedef enum {
    SAM3D_SS_STREAM_SHAPE = 0,
    SAM3D_SS_STREAM_POSE  = 1
} sam3d_ss_stream_id;

typedef struct {
    qtensor input_w, input_b;       /* [D, in_ch], [D] */
    qtensor out_w,   out_b;         /* [in_ch, D], [in_ch] */
    qtensor pos_emb;                /* [token_len, D] */
    int in_channels;
    int token_len;                  /* 4096 for shape, 1 for the others */
} sam3d_ss_latent_map;

/* Per-modality projections inside one MOT block. We store tensors for
 * both transformer streams: shape and 6drotation_normalized (pose). */
typedef struct {
    qtensor norm2_w, norm2_b;               /* [D] — only elementwise_affine=True */
    /* self-attn (per-stream) */
    qtensor sa_qkv_w, sa_qkv_b;             /* [3D, D], [3D] */
    qtensor sa_out_w, sa_out_b;             /* [D, D], [D] */
    qtensor sa_q_rms_gamma;                 /* [H, head_dim] */
    qtensor sa_k_rms_gamma;                 /* [H, head_dim] */
    /* cross-attn (per-stream) */
    qtensor xa_q_w,  xa_q_b;                /* [D, D] */
    qtensor xa_kv_w, xa_kv_b;               /* [2D, D] */
    qtensor xa_out_w, xa_out_b;             /* [D, D] */
    /* FFN (per-stream) */
    qtensor mlp_fc1_w, mlp_fc1_b;           /* [4D, D] */
    qtensor mlp_fc2_w, mlp_fc2_b;           /* [D, 4D] */
} sam3d_ss_block_stream;

typedef struct {
    /* Shared across streams */
    qtensor adaln_w, adaln_b;               /* [6D, D], [6D] */
    /* Per-stream parts (index by sam3d_ss_stream_id) */
    sam3d_ss_block_stream stream[SAM3D_SS_DIT_N_STREAMS];
} sam3d_ss_block;

typedef struct {
    /* Architecture */
    int dim;                    /* 1024 */
    int n_heads;                /* 16 */
    int head_dim;               /* 64 */
    int n_blocks;               /* 24 */
    int mlp_hidden;             /* 4096 = mlp_ratio * dim */
    int cond_channels;          /* 1024 */
    int freq_dim;               /* 256 — timestep frequency embed dim */
    int is_shortcut;            /* 1 if model has a d_embedder */
    float ln_eps;               /* 1e-6 */
    float time_scale;           /* 1000.0 from ss_generator.yaml */
    float ss_resolution;        /* 16 (16³ grid for shape) */

    /* Top-level weights */
    qtensor t_emb_fc1_w, t_emb_fc1_b;       /* [D, freq], [D] */
    qtensor t_emb_fc2_w, t_emb_fc2_b;       /* [D, D],    [D] */
    qtensor d_emb_fc1_w, d_emb_fc1_b;       /* optional (shortcut) */
    qtensor d_emb_fc2_w, d_emb_fc2_b;

    /* Per-modality latent mapping (5 entries) */
    sam3d_ss_latent_map latent[SAM3D_SS_DIT_N_LATENTS];

    /* 24 MOT blocks */
    sam3d_ss_block *blocks;

    st_context *st_ctx;
} sam3d_ss_flow_dit_model;

/* Load all weights. Returns NULL on failure. */
sam3d_ss_flow_dit_model *
sam3d_ss_flow_dit_load_safetensors(const char *path);

void
sam3d_ss_flow_dit_free(sam3d_ss_flow_dit_model *m);

/* Forward pass. Inputs and outputs are arrays of 5 float32 buffers, one
 * per modality (indexed by sam3d_ss_lat_id). Shapes:
 *   SHAPE:              [4096 * 8]   (16³ grid of 8-ch latents)
 *   6DROT:              [1 * 6]
 *   TRANSLATION:        [1 * 3]
 *   SCALE:              [1 * 3]
 *   TRANSLATION_SCALE:  [1 * 1]
 * `cond` is [n_cond, 1024] (DINOv2+PPE fused tokens from cond_fuser).
 * `t` is the current flow-matching timestep, **already multiplied by
 * `time_scale`** (i.e. in [0, 1000]). The model's TimestepEmbedder takes
 * `t` directly — upstream's `_generate_dynamics` does `t * time_scale`
 * before reaching the model, so production callers must do the same.
 * `d` is the shortcut jump size (also pre-scaled; ignored if not shortcut).
 *
 * Returns 0 on success, <0 on failure. */
int
sam3d_ss_flow_dit_forward(const sam3d_ss_flow_dit_model *m,
                          const float *const *latents_in,
                          float *const *latents_out,
                          const float *cond, int n_cond,
                          float t, float d,
                          int n_threads);

#ifdef SAM3D_SS_FLOW_DIT_IMPLEMENTATION

/* ─── Numeric primitives ─────────────────────────────────────────── */

static int ssdit_numel(const qtensor *t) {
    if (!t->data) return 0;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    return n;
}

static const float *ssdit_row_f32(const qtensor *t, int *owned) {
    *owned = 0;
    if (!t || !t->data) return NULL;
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    if (t->type != GGML_TYPE_F16) {
        fprintf(stderr, "[ssdit] unsupported tensor dtype %d — only F32/F16 supported\n",
                (int)t->type);
        return NULL;
    }
    int n = ssdit_numel(t);
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    const uint16_t *src = (const uint16_t *)t->data;
    for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    *owned = 1;
    return buf;
}

/* out[M,N] = in[M,K] @ W[N,K]^T + b[N]  (W is row-major [N,K]).
 *
 * Defers to cpu_gemm_f32 (cpu_compute.h) which has AVX2+FMA SIMD AND
 * collapse(2) parallelization over both (M, N). The previous scalar
 * `omp parallel for over m` failed to parallelize the very common
 * M=1 calls (emb_out / adaln_w / TimestepEmbedder) — slat_dit's ODE
 * step time was dominated by those serial scalar reductions. */
static void ssdit_gemm(float *out, const qtensor *W, const qtensor *b,
                       const float *in, int M, int N, int K, int nthr) {
    int wo = 0; const float *wf = ssdit_row_f32(W, &wo);
    int bo = 0; const float *bf = b ? ssdit_row_f32(b, &bo) : NULL;
    cpu_gemm_f32(out, wf, bf, in, M, N, K, nthr);
    if (wo) free((void *)wf);
    if (bo) free((void *)bf);
}

static void ssdit_silu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* GELU(tanh-approx): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))) */
static void ssdit_gelu_tanh_inplace(float *x, int n) {
    const float k = 0.7978845608028654f;       /* sqrt(2/pi) */
    const float c = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float u = k * (v + c * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(u));
    }
}

/* LayerNorm with optional affine (w,b) and optional AdaLN modulation (shift,scale).
 * All four params may be NULL independently:
 *   y[i] = ((x[i]-mean)*inv) * (w?w[i]:1) * (scale?1+scale[i]:1) + (b?b[i]:0) + (shift?shift[i]:0)
 * Shift/scale are shared across tokens; w/b are per-dim qtensors. */
static void ssdit_layernorm(float *out, const float *in,
                            const qtensor *w, const qtensor *b,
                            const float *shift, const float *scale,
                            int n_tok, int dim, float eps) {
    int wo = 0, bo = 0;
    const float *ww = w ? ssdit_row_f32(w, &wo) : NULL;
    const float *bb = b ? ssdit_row_f32(b, &bo) : NULL;
    for (int t = 0; t < n_tok; t++) {
        const float *x = in  + (size_t)t * dim;
        float       *y = out + (size_t)t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++) {
            float v = (x[i] - mean) * inv;
            if (ww) v *= ww[i];
            if (scale) v *= 1.0f + scale[i];
            if (bb) v += bb[i];
            if (shift) v += shift[i];
            y[i] = v;
        }
    }
    if (wo) free((void *)ww);
    if (bo) free((void *)bb);
}

/* MultiHeadRMSNorm (upstream): F.normalize(x, dim=-1) * gamma * sqrt(head_dim).
 * Note this is NOT classic RMSNorm — it divides by L2-norm (not RMS) and then
 * multiplies by gamma AND sqrt(head_dim). gamma has shape [n_heads, head_dim]. */
static void ssdit_mhrmsnorm(float *v, int n_tok, int n_heads, int head_dim,
                            int stride, const float *gamma) {
    float scale = sqrtf((float)head_dim);
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *x = v + (size_t)t * stride + h * head_dim;
            const float *g = gamma + h * head_dim;
            double ss = 0.0;
            for (int i = 0; i < head_dim; i++) ss += (double)x[i] * x[i];
            float inv = 1.0f / (sqrtf((float)ss) + 1e-12f);
            for (int i = 0; i < head_dim; i++) x[i] = x[i] * inv * g[i] * scale;
        }
    }
}

/* Single-head online SDPA (streaming softmax; no N_q × N_k scratch).
 * Q is [N_q, head_dim], K and V are [N_k, head_dim] (contig per row).
 * out is [N_q, head_dim]. Scale defaults to 1/sqrt(head_dim). */
/* Sinusoidal timestep embedding, upstream-compatible:
 *   half = freq_dim / 2
 *   freqs[j] = exp(-log(10000) * j / half)
 *   args[j] = t * freqs[j]
 *   embed = [cos(args); sin(args)]
 * out must have room for `freq_dim` floats. */
static void ssdit_freq_embed(float *out, float t, int freq_dim) {
    int half = freq_dim / 2;
    float neg_log10k = -logf(10000.0f);
    for (int j = 0; j < half; j++) {
        float freq = expf(neg_log10k * (float)j / (float)half);
        float arg = t * freq;
        out[j] = cosf(arg);
        out[half + j] = sinf(arg);
    }
}

/* TimestepEmbedder MLP: freq_embed(t) → Linear(freq_dim→D) → SiLU → Linear(D→D). */
static void ssdit_time_mlp(float *out, float t,
                           const qtensor *fc1_w, const qtensor *fc1_b,
                           const qtensor *fc2_w, const qtensor *fc2_b,
                           int freq_dim, int dim) {
    float *emb = (float *)malloc((size_t)freq_dim * sizeof(float));
    float *h1  = (float *)malloc((size_t)dim      * sizeof(float));
    ssdit_freq_embed(emb, t, freq_dim);
    ssdit_gemm(h1, fc1_w, fc1_b, emb, 1, dim, freq_dim, 1);
    ssdit_silu_inplace(h1, dim);
    ssdit_gemm(out, fc2_w, fc2_b, h1, 1, dim, dim, 1);
    free(emb);
    free(h1);
}

static int sam3d_ssdit_load_latent(st_context *ctx, const char *name,
                                   sam3d_ss_latent_map *m) {
    char buf[256];
    int rc = 0;
    snprintf(buf, sizeof(buf), "latent_mapping.%s.input_layer.weight", name);
    rc |= qt_find(ctx, buf, &m->input_w);
    snprintf(buf, sizeof(buf), "latent_mapping.%s.input_layer.bias", name);
    rc |= qt_find(ctx, buf, &m->input_b);
    snprintf(buf, sizeof(buf), "latent_mapping.%s.out_layer.weight", name);
    rc |= qt_find(ctx, buf, &m->out_w);
    snprintf(buf, sizeof(buf), "latent_mapping.%s.out_layer.bias", name);
    rc |= qt_find(ctx, buf, &m->out_b);
    snprintf(buf, sizeof(buf), "latent_mapping.%s.pos_emb", name);
    rc |= qt_find(ctx, buf, &m->pos_emb);
    if (rc) return rc;
    m->in_channels = (int)m->input_w.dims[1];
    m->token_len   = (int)m->pos_emb.dims[0];
    return 0;
}

static int sam3d_ssdit_load_block_stream(st_context *ctx, int block_idx,
                                         const char *mod,
                                         sam3d_ss_block_stream *s) {
    char buf[256];
    int rc = 0;
#define FIND(out_field, fmt) do {                                       \
    snprintf(buf, sizeof(buf), "blocks.%d." fmt, block_idx, mod);       \
    rc |= qt_find(ctx, buf, &s->out_field);                    \
} while (0)
    FIND(norm2_w,        "norm2.%s.weight");
    FIND(norm2_b,        "norm2.%s.bias");
    FIND(sa_qkv_w,       "self_attn.to_qkv.%s.weight");
    FIND(sa_qkv_b,       "self_attn.to_qkv.%s.bias");
    FIND(sa_out_w,       "self_attn.to_out.%s.weight");
    FIND(sa_out_b,       "self_attn.to_out.%s.bias");
    FIND(sa_q_rms_gamma, "self_attn.q_rms_norm.%s.gamma");
    FIND(sa_k_rms_gamma, "self_attn.k_rms_norm.%s.gamma");
    FIND(xa_q_w,         "cross_attn.%s.to_q.weight");
    FIND(xa_q_b,         "cross_attn.%s.to_q.bias");
    FIND(xa_kv_w,        "cross_attn.%s.to_kv.weight");
    FIND(xa_kv_b,        "cross_attn.%s.to_kv.bias");
    FIND(xa_out_w,       "cross_attn.%s.to_out.weight");
    FIND(xa_out_b,       "cross_attn.%s.to_out.bias");
    FIND(mlp_fc1_w,      "mlp.%s.mlp.0.weight");
    FIND(mlp_fc1_b,      "mlp.%s.mlp.0.bias");
    FIND(mlp_fc2_w,      "mlp.%s.mlp.2.weight");
    FIND(mlp_fc2_b,      "mlp.%s.mlp.2.bias");
#undef FIND
    return rc;
}

sam3d_ss_flow_dit_model *
sam3d_ss_flow_dit_load_safetensors(const char *path) {
    st_context *ctx = safetensors_open(path);
    if (!ctx) {
        fprintf(stderr, "sam3d_ss_flow_dit: cannot open %s\n", path);
        return NULL;
    }
    sam3d_ss_flow_dit_model *m = (sam3d_ss_flow_dit_model *)calloc(1, sizeof(*m));
    if (!m) { safetensors_close(ctx); return NULL; }
    m->st_ctx = ctx;

    m->dim = 1024;
    m->n_heads = 16;
    m->head_dim = 64;
    m->mlp_hidden = 4096;
    m->cond_channels = 1024;
    m->freq_dim = 256;
    m->ln_eps = 1e-6f;
    m->time_scale = 1000.0f;
    m->ss_resolution = 16.0f;

    int rc = 0;
    rc |= qt_find(ctx, "t_embedder.mlp.0.weight", &m->t_emb_fc1_w);
    rc |= qt_find(ctx, "t_embedder.mlp.0.bias",   &m->t_emb_fc1_b);
    rc |= qt_find(ctx, "t_embedder.mlp.2.weight", &m->t_emb_fc2_w);
    rc |= qt_find(ctx, "t_embedder.mlp.2.bias",   &m->t_emb_fc2_b);
    if (rc) {
        fprintf(stderr, "sam3d_ss_flow_dit: missing t_embedder weights\n");
        sam3d_ss_flow_dit_free(m); return NULL;
    }
    /* d_embedder is present only in shortcut models; absence is OK. */
    int rc_d = 0;
    rc_d |= qt_find(ctx, "d_embedder.mlp.0.weight", &m->d_emb_fc1_w);
    rc_d |= qt_find(ctx, "d_embedder.mlp.0.bias",   &m->d_emb_fc1_b);
    rc_d |= qt_find(ctx, "d_embedder.mlp.2.weight", &m->d_emb_fc2_w);
    rc_d |= qt_find(ctx, "d_embedder.mlp.2.bias",   &m->d_emb_fc2_b);
    m->is_shortcut = (rc_d == 0) ? 1 : 0;

    static const char *lat_names[SAM3D_SS_DIT_N_LATENTS] = {
        "shape", "6drotation_normalized", "translation", "scale", "translation_scale"
    };
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        if (sam3d_ssdit_load_latent(ctx, lat_names[i], &m->latent[i]) != 0) {
            fprintf(stderr, "sam3d_ss_flow_dit: missing latent_mapping.%s.*\n",
                    lat_names[i]);
            sam3d_ss_flow_dit_free(m); return NULL;
        }
    }

    int n_blocks = 0;
    for (int i = 0; i < SAM3D_SS_DIT_MAX_BLOCKS; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "blocks.%d.adaLN_modulation.1.weight", i);
        if (safetensors_find(ctx, buf) < 0) break;
        n_blocks = i + 1;
    }
    if (n_blocks == 0) {
        fprintf(stderr, "sam3d_ss_flow_dit: no blocks found\n");
        sam3d_ss_flow_dit_free(m); return NULL;
    }
    m->n_blocks = n_blocks;
    m->blocks = (sam3d_ss_block *)calloc(n_blocks, sizeof(sam3d_ss_block));
    if (!m->blocks) { sam3d_ss_flow_dit_free(m); return NULL; }

    /* Indices must match sam3d_ss_stream_id. */
    static const char *stream_names[SAM3D_SS_DIT_N_STREAMS] = {
        "shape", "6drotation_normalized"
    };
    for (int i = 0; i < n_blocks; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "blocks.%d.adaLN_modulation.1.weight", i);
        if (qt_find(ctx, buf, &m->blocks[i].adaln_w) != 0) {
            fprintf(stderr, "sam3d_ss_flow_dit: block %d missing adaln_w\n", i);
            sam3d_ss_flow_dit_free(m); return NULL;
        }
        snprintf(buf, sizeof(buf), "blocks.%d.adaLN_modulation.1.bias", i);
        if (qt_find(ctx, buf, &m->blocks[i].adaln_b) != 0) {
            fprintf(stderr, "sam3d_ss_flow_dit: block %d missing adaln_b\n", i);
            sam3d_ss_flow_dit_free(m); return NULL;
        }
        for (int s = 0; s < SAM3D_SS_DIT_N_STREAMS; s++) {
            if (sam3d_ssdit_load_block_stream(ctx, i, stream_names[s],
                                              &m->blocks[i].stream[s]) != 0) {
                fprintf(stderr, "sam3d_ss_flow_dit: block %d stream %s load failed\n",
                        i, stream_names[s]);
                sam3d_ss_flow_dit_free(m); return NULL;
            }
        }
    }

    fprintf(stderr,
            "sam3d_ss_flow_dit: loaded %d blocks, dim=%d heads=%d head_dim=%d "
            "cond=%d shortcut=%s\n",
            m->n_blocks, m->dim, m->n_heads, m->head_dim, m->cond_channels,
            m->is_shortcut ? "yes" : "no");
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        fprintf(stderr,
                "  latent[%d] in_ch=%d token_len=%d\n",
                i, m->latent[i].in_channels, m->latent[i].token_len);
    }
    return m;
}

void sam3d_ss_flow_dit_free(sam3d_ss_flow_dit_model *m) {
    if (!m) return;
    free(m->blocks);
    if (m->st_ctx) safetensors_close(m->st_ctx);
    free(m);
}

/* ─── Forward pass ─────────────────────────────────────────────────── */

/* Project one modality into its 4096/1 × D token stream. out must point
 * at a [token_len, D] float32 buffer; pos_emb is pre-baked into the
 * latent mapping so we just add it. */
static int ssdit_project_input(const sam3d_ss_latent_map *L, int dim,
                               const float *in, float *out, int nthr) {
    int T = L->token_len;
    ssdit_gemm(out, &L->input_w, &L->input_b, in, T, dim, L->in_channels, nthr);
    int po = 0; const float *pe = ssdit_row_f32(&L->pos_emb, &po);
    for (int t = 0; t < T; t++) {
        float *y = out + (size_t)t * dim;
        const float *p = pe  + (size_t)t * dim;
        for (int i = 0; i < dim; i++) y[i] += p[i];
    }
    if (po) free((void *)pe);
    return 0;
}

/* LayerNorm(no affine, last dim) → Linear back to in_channels. */
static int ssdit_project_output(const sam3d_ss_latent_map *L, int dim,
                                const float *in, float *out, float eps, int nthr) {
    int T = L->token_len;
    float *ln = (float *)malloc((size_t)T * dim * sizeof(float));
    ssdit_layernorm(ln, in, NULL, NULL, NULL, NULL, T, dim, eps);
    ssdit_gemm(out, &L->out_w, &L->out_b, ln, T, L->in_channels, dim, nthr);
    free(ln);
    return 0;
}

/* One-stream self-attention half: to_qkv → qk rms_norm → unused here (MOT
 * attention combines streams). Produces q,k,v buffers of shape
 * [N, n_heads, head_dim] with interleaved layout [token, head, dim]. */
static void ssdit_sa_project_qkv(const sam3d_ss_block_stream *s,
                                 int dim, int n_heads, int head_dim,
                                 const float *x, int N,
                                 float *q, float *k, float *v, int nthr) {
    float *qkv = (float *)malloc((size_t)N * 3 * dim * sizeof(float));
    ssdit_gemm(qkv, &s->sa_qkv_w, &s->sa_qkv_b, x, N, 3 * dim, dim, nthr);
    /* qkv layout from Linear: [N, 3*dim] contiguous. Upstream reshapes
     * to [N, 3, n_heads, head_dim] so qkv[n, 0, h, :] = qkv[n*3*dim + h*head_dim].
     * Split into q,k,v with layout [N, n_heads, head_dim]. */
    int qstride = n_heads * head_dim;
    for (int n = 0; n < N; n++) {
        const float *src = qkv + (size_t)n * 3 * dim;
        memcpy(q + (size_t)n * qstride, src + 0 * dim, (size_t)dim * sizeof(float));
        memcpy(k + (size_t)n * qstride, src + 1 * dim, (size_t)dim * sizeof(float));
        memcpy(v + (size_t)n * qstride, src + 2 * dim, (size_t)dim * sizeof(float));
    }
    free(qkv);

    /* Per-head RMSNorm for q and k. */
    int wo_q = 0, wo_k = 0;
    const float *gq = ssdit_row_f32(&s->sa_q_rms_gamma, &wo_q);
    const float *gk = ssdit_row_f32(&s->sa_k_rms_gamma, &wo_k);
    ssdit_mhrmsnorm(q, N, n_heads, head_dim, qstride, gq);
    ssdit_mhrmsnorm(k, N, n_heads, head_dim, qstride, gk);
    if (wo_q) free((void *)gq);
    if (wo_k) free((void *)gk);
}

/* Interleave [N, dim] K + [N, dim] V into [N, 2*dim] layout expected by
 * cpu_cross_attention. */
static void ssdit_pack_kv(float *kv, const float *k, const float *v,
                          int N, int dim) {
    for (int n = 0; n < N; n++) {
        memcpy(kv + (size_t)n * 2 * dim + 0,   k + (size_t)n * dim, (size_t)dim * sizeof(float));
        memcpy(kv + (size_t)n * 2 * dim + dim, v + (size_t)n * dim, (size_t)dim * sizeof(float));
    }
}

/* Run MOT self-attention across both streams, write results into h_shape, h_pose.
 * shape is protected: attends only to itself. pose attends to [pose; shape]. */
static void ssdit_mot_self_attn(const sam3d_ss_block *blk, int dim,
                                int n_heads, int head_dim,
                                float *x_shape, int N_s,
                                float *x_pose,  int N_p,
                                float *h_shape, float *h_pose,
                                int nthr) {
    const sam3d_ss_block_stream *ss = &blk->stream[SAM3D_SS_STREAM_SHAPE];
    const sam3d_ss_block_stream *ps = &blk->stream[SAM3D_SS_STREAM_POSE];

    float *q_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *k_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *v_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *q_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    float *k_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    float *v_p = (float *)malloc((size_t)N_p * dim * sizeof(float));

    ssdit_sa_project_qkv(ss, dim, n_heads, head_dim, x_shape, N_s, q_s, k_s, v_s, nthr);
    ssdit_sa_project_qkv(ps, dim, n_heads, head_dim, x_pose,  N_p, q_p, k_p, v_p, nthr);

    /* shape self-attn: Q=q_s, KV=pack(k_s, v_s). */
    float *kv_s = (float *)malloc((size_t)N_s * 2 * dim * sizeof(float));
    ssdit_pack_kv(kv_s, k_s, v_s, N_s, dim);
    float *h_sa_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    cpu_cross_attention(h_sa_s, q_s, kv_s, N_s, N_s, dim, n_heads, head_dim, nthr);
    free(kv_s);

    /* pose self-attn: K/V = concat(pose, shape). */
    int N_kv = N_p + N_s;
    float *kv_p = (float *)malloc((size_t)N_kv * 2 * dim * sizeof(float));
    ssdit_pack_kv(kv_p,                        k_p, v_p, N_p, dim);
    ssdit_pack_kv(kv_p + (size_t)N_p * 2 * dim, k_s, v_s, N_s, dim);
    float *h_sa_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    cpu_cross_attention(h_sa_p, q_p, kv_p, N_p, N_kv, dim, n_heads, head_dim, nthr);
    free(kv_p);

    free(q_s); free(k_s); free(v_s);
    free(q_p); free(k_p); free(v_p);

    /* to_out per stream. */
    ssdit_gemm(h_shape, &ss->sa_out_w, &ss->sa_out_b, h_sa_s, N_s, dim, dim, nthr);
    ssdit_gemm(h_pose,  &ps->sa_out_w, &ps->sa_out_b, h_sa_p, N_p, dim, dim, nthr);
    free(h_sa_s);
    free(h_sa_p);
}

/* Cross-attention: Q = h_m (per-stream), K/V = shared `cond`. The xa_kv
 * projection already produces [N_c, 2*dim] — the layout cpu_cross_attention
 * expects — so no repacking is needed. */
static void ssdit_cross_attn(const sam3d_ss_block_stream *s,
                             int dim, int n_heads, int head_dim,
                             const float *h, int N,
                             const float *cond, int N_c,
                             float *out, int nthr) {
    float *q  = (float *)malloc((size_t)N   * dim     * sizeof(float));
    float *kv = (float *)malloc((size_t)N_c * 2 * dim * sizeof(float));
    float *O  = (float *)malloc((size_t)N   * dim     * sizeof(float));

    ssdit_gemm(q,  &s->xa_q_w,  &s->xa_q_b,  h,    N,   dim,     dim, nthr);
    ssdit_gemm(kv, &s->xa_kv_w, &s->xa_kv_b, cond, N_c, 2 * dim, dim, nthr);
    cpu_cross_attention(O, q, kv, N, N_c, dim, n_heads, head_dim, nthr);
    ssdit_gemm(out, &s->xa_out_w, &s->xa_out_b, O, N, dim, dim, nthr);
    free(q); free(kv); free(O);
}

/* MLP: Linear(D → 4D) → GELU(tanh-approx) → Linear(4D → D). */
static void ssdit_mlp_stream(const sam3d_ss_block_stream *s,
                             int dim, int mlp_hidden,
                             const float *in, int N, float *out, int nthr) {
    float *h1 = (float *)malloc((size_t)N * mlp_hidden * sizeof(float));
    ssdit_gemm(h1, &s->mlp_fc1_w, &s->mlp_fc1_b, in, N, mlp_hidden, dim, nthr);
    ssdit_gelu_tanh_inplace(h1, N * mlp_hidden);
    ssdit_gemm(out, &s->mlp_fc2_w, &s->mlp_fc2_b, h1, N, dim, mlp_hidden, nthr);
    free(h1);
}

/* One MOT block: norm1 → adaLN-modulate → MOT-SA → gate+residual →
 *                norm2 (affine) → cross_attn → residual →
 *                norm3 → adaLN-modulate_mlp → mlp → gate+residual. */
static void ssdit_block_forward(const sam3d_ss_flow_dit_model *m,
                                const sam3d_ss_block *blk,
                                const float *mod6,    /* [6*dim] */
                                float *x_shape, int N_s,
                                float *x_pose,  int N_p,
                                const float *cond, int N_c,
                                int nthr) {
    int dim = m->dim;
    const float *shift_msa = mod6 + 0 * dim;
    const float *scale_msa = mod6 + 1 * dim;
    const float *gate_msa  = mod6 + 2 * dim;
    const float *shift_mlp = mod6 + 3 * dim;
    const float *scale_mlp = mod6 + 4 * dim;
    const float *gate_mlp  = mod6 + 5 * dim;

    const sam3d_ss_block_stream *ss = &blk->stream[SAM3D_SS_STREAM_SHAPE];
    const sam3d_ss_block_stream *ps = &blk->stream[SAM3D_SS_STREAM_POSE];

    float *h_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *h_p = (float *)malloc((size_t)N_p * dim * sizeof(float));
    float *t_s = (float *)malloc((size_t)N_s * dim * sizeof(float));
    float *t_p = (float *)malloc((size_t)N_p * dim * sizeof(float));

    /* norm1 (no affine) + adaLN-modulate */
    ssdit_layernorm(h_s, x_shape, NULL, NULL, shift_msa, scale_msa, N_s, dim, m->ln_eps);
    ssdit_layernorm(h_p, x_pose,  NULL, NULL, shift_msa, scale_msa, N_p, dim, m->ln_eps);

    /* MOT self-attn. Writes h_s, h_p = to_out(sdpa(...)). */
    ssdit_mot_self_attn(blk, dim, m->n_heads, m->head_dim,
                        h_s, N_s, h_p, N_p, t_s, t_p, nthr);

    /* gate_msa + residual: x += t * gate_msa */
    for (int n = 0; n < N_s; n++)
        for (int i = 0; i < dim; i++)
            x_shape[n * dim + i] += t_s[n * dim + i] * gate_msa[i];
    for (int n = 0; n < N_p; n++)
        for (int i = 0; i < dim; i++)
            x_pose[n * dim + i] += t_p[n * dim + i] * gate_msa[i];

    /* norm2 (affine, per stream) → cross-attn → residual */
    ssdit_layernorm(h_s, x_shape, &ss->norm2_w, &ss->norm2_b, NULL, NULL, N_s, dim, m->ln_eps);
    ssdit_layernorm(h_p, x_pose,  &ps->norm2_w, &ps->norm2_b, NULL, NULL, N_p, dim, m->ln_eps);
    ssdit_cross_attn(ss, dim, m->n_heads, m->head_dim, h_s, N_s, cond, N_c, t_s, nthr);
    ssdit_cross_attn(ps, dim, m->n_heads, m->head_dim, h_p, N_p, cond, N_c, t_p, nthr);
    for (int n = 0; n < N_s * dim; n++) x_shape[n] += t_s[n];
    for (int n = 0; n < N_p * dim; n++) x_pose[n]  += t_p[n];

    /* norm3 (no affine) + adaLN-modulate_mlp → mlp → gate+residual */
    ssdit_layernorm(h_s, x_shape, NULL, NULL, shift_mlp, scale_mlp, N_s, dim, m->ln_eps);
    ssdit_layernorm(h_p, x_pose,  NULL, NULL, shift_mlp, scale_mlp, N_p, dim, m->ln_eps);
    ssdit_mlp_stream(ss, dim, m->mlp_hidden, h_s, N_s, t_s, nthr);
    ssdit_mlp_stream(ps, dim, m->mlp_hidden, h_p, N_p, t_p, nthr);
    for (int n = 0; n < N_s; n++)
        for (int i = 0; i < dim; i++)
            x_shape[n * dim + i] += t_s[n * dim + i] * gate_mlp[i];
    for (int n = 0; n < N_p; n++)
        for (int i = 0; i < dim; i++)
            x_pose[n * dim + i] += t_p[n * dim + i] * gate_mlp[i];

    free(h_s); free(h_p); free(t_s); free(t_p);
}

int sam3d_ss_flow_dit_forward(const sam3d_ss_flow_dit_model *m,
                              const float *const *latents_in,
                              float *const *latents_out,
                              const float *cond, int n_cond,
                              float t, float d,
                              int n_threads) {
    if (!m || !latents_in || !latents_out || !cond) return -1;
    if (n_threads <= 0) n_threads = 1;

    int dim = m->dim;

    /* Project each input modality into D-token streams, then merge the
     * four pose modalities into a single 4-token pose stream. */
    float *per_mod[SAM3D_SS_DIT_N_LATENTS] = {0};
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        const sam3d_ss_latent_map *L = &m->latent[i];
        per_mod[i] = (float *)malloc((size_t)L->token_len * dim * sizeof(float));
        ssdit_project_input(L, dim, latents_in[i], per_mod[i], n_threads);
    }
    int N_s = m->latent[SAM3D_SS_LAT_SHAPE].token_len;
    float *x_shape = per_mod[SAM3D_SS_LAT_SHAPE];

    /* Pose stream = concat of [6drot, translation, scale, translation_scale].
     * Each contributes `token_len` (=1) tokens with its own pos_emb already
     * added. Total = 4 tokens. */
    int N_p = 0;
    for (int i = SAM3D_SS_LAT_6DROT; i <= SAM3D_SS_LAT_TRANSLATION_SCALE; i++)
        N_p += m->latent[i].token_len;
    float *x_pose = (float *)malloc((size_t)N_p * dim * sizeof(float));
    {
        float *dst = x_pose;
        for (int i = SAM3D_SS_LAT_6DROT; i <= SAM3D_SS_LAT_TRANSLATION_SCALE; i++) {
            int tl = m->latent[i].token_len;
            memcpy(dst, per_mod[i], (size_t)tl * dim * sizeof(float));
            dst += (size_t)tl * dim;
            free(per_mod[i]);
            per_mod[i] = NULL;
        }
    }

    /* Timestep + shortcut embedding. Upstream TimestepEmbedder takes t
     * directly (no time_scale); any sampler-level scaling is the caller's
     * responsibility. */
    float *t_emb = (float *)calloc((size_t)dim, sizeof(float));
    ssdit_time_mlp(t_emb, t,
                   &m->t_emb_fc1_w, &m->t_emb_fc1_b,
                   &m->t_emb_fc2_w, &m->t_emb_fc2_b,
                   m->freq_dim, dim);
    if (m->is_shortcut) {
        float *d_emb = (float *)calloc((size_t)dim, sizeof(float));
        ssdit_time_mlp(d_emb, d,
                       &m->d_emb_fc1_w, &m->d_emb_fc1_b,
                       &m->d_emb_fc2_w, &m->d_emb_fc2_b,
                       m->freq_dim, dim);
        for (int i = 0; i < dim; i++) t_emb[i] += d_emb[i];
        free(d_emb);
    }

    /* Per-block: adaLN_modulation(SiLU(t_emb)) → 6*dim. */
    float *mod6 = (float *)malloc((size_t)6 * dim * sizeof(float));
    float *t_silu = (float *)malloc((size_t)dim * sizeof(float));

    for (int b = 0; b < m->n_blocks; b++) {
        memcpy(t_silu, t_emb, (size_t)dim * sizeof(float));
        ssdit_silu_inplace(t_silu, dim);
        ssdit_gemm(mod6, &m->blocks[b].adaln_w, &m->blocks[b].adaln_b,
                   t_silu, 1, 6 * dim, dim, n_threads);
        ssdit_block_forward(m, &m->blocks[b], mod6,
                            x_shape, N_s, x_pose, N_p,
                            cond, n_cond, n_threads);
    }
    free(mod6);
    free(t_silu);
    free(t_emb);

    /* Split pose stream back into 4 per-modality buffers and run each
     * modality's LayerNorm+Linear out. Shape uses its own stream directly. */
    ssdit_project_output(&m->latent[SAM3D_SS_LAT_SHAPE], dim,
                         x_shape, latents_out[SAM3D_SS_LAT_SHAPE],
                         m->ln_eps, n_threads);
    {
        const float *src = x_pose;
        for (int i = SAM3D_SS_LAT_6DROT; i <= SAM3D_SS_LAT_TRANSLATION_SCALE; i++) {
            ssdit_project_output(&m->latent[i], dim, src,
                                 latents_out[i], m->ln_eps, n_threads);
            src += (size_t)m->latent[i].token_len * dim;
        }
    }

    free(x_shape);
    free(x_pose);
    return 0;
}

#endif /* SAM3D_SS_FLOW_DIT_IMPLEMENTATION */

#ifdef __cplusplus
}
#endif

#endif /* SAM3D_SS_FLOW_DIT_H */
