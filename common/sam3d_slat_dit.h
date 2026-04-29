/*
 * sam3d_slat_dit.h — stage-2 Sparse-Latent Flow DiT for sam-3d-objects.
 *
 * pytorch: model/backbone/tdfy_dit/models/structured_latent_flow.py
 *          + modules/sparse/transformer/modulated.py::ModulatedSparseTransformerCrossBlock
 *
 * Architecture (from slat_generator.yaml):
 *   model_channels = 1024, num_blocks = 24, num_heads = 16, head_dim = 64,
 *   mlp_ratio = 4, patch_size = 2 (1 downsample stage),
 *   io_block_channels = [128], num_io_res_blocks = 2,
 *   in_channels = out_channels = 8, resolution = 64,
 *   pe_mode = "ape", qk_rms_norm = true,
 *   is_shortcut_model = false (standard flow; no d_embedder),
 *   attn_mode = "full"  — NOT shift-window. Shift-window is only in the
 *   SLAT GS decoder (step 8), not in this DiT.
 *
 *   Inference steps: 12.
 *
 * Forward (per denoising step):
 *   1. input_layer: SparseLinear(8 → 128) on feats; coords unchanged.
 *   2. For each of the num_io_res_blocks (here 2) input_blocks:
 *      - block 0: SparseResBlock3d (C_in=128, C_out=128), NO downsample.
 *      - block 1: SparseResBlock3d (C_in=128, C_out=1024), downsample=True.
 *      skips append h.feats BEFORE downsample (and AFTER feats in torch
 *      impl; see note in _updown).
 *   3. pos_embedder: APE sinusoidal over (z,y,x) coords; added to h.feats.
 *   4. 24 × ModulatedSparseTransformerCrossBlock (full sparse self-attn
 *      over all tokens per-batch, cross-attn with cond_tokens).
 *   5. For each of the out_blocks (here 2):
 *      - block 0: upsample then SparseResBlock3d, skip-concat in=2048, out=128.
 *      - block 1: no updown, SparseResBlock3d, skip-concat in=256, out=128.
 *   6. F.layer_norm(feats, last_dim_shape) — NO affine, normalize only.
 *   7. out_layer: SparseLinear(128 → 8).
 *
 * Conv layout in ckpt: weight shape [out_C, k=3, k=3, k=3, in_C]. That is
 * the memory layout sp3d_conv3d_forward expects ([out_C, K3=27, in_C]),
 * with K3 enumerated as z*9 + y*3 + x, so we can pass the raw buffer
 * directly.
 *
 * This header is header-only; call `#define SAM3D_SLAT_DIT_IMPLEMENTATION`
 * in exactly one TU. Depends on safetensors.h, ggml_dequant.h, sparse3d.h.
 */

#ifndef SAM3D_SLAT_DIT_H
#define SAM3D_SLAT_DIT_H

#include "safetensors.h"
#include "ggml_dequant.h"
#include "sparse3d.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "qtensor_utils.h"

/* One SparseResBlock3d with optional updown.
 *   norm1 (elementwise_affine=True):   [C_in]   w/b
 *   conv1 (submanifold 3x3x3):         [C_out, 27, C_in] w/b
 *   norm2 (elementwise_affine=False):  — no learned params —
 *   conv2 (submanifold 3x3x3):         [C_out, 27, C_out] w/b (zero-init)
 *   emb_layers.1 (SiLU applied before):[2*C_out, 1024] w/b — produces scale+shift
 *   skip_connection: SparseLinear(C_in, C_out) if C_in != C_out, else Identity.
 * updown is one of { NONE, DOWNSAMPLE_2, UPSAMPLE_2 } — applied to x BEFORE
 * norm1 (torch _updown). */
typedef enum {
    SAM3D_SLAT_UPDOWN_NONE = 0,
    SAM3D_SLAT_UPDOWN_DOWN = 1,
    SAM3D_SLAT_UPDOWN_UP   = 2
} sam3d_slat_updown_mode;

typedef struct {
    qtensor norm1_w, norm1_b;       /* [C_in] */
    qtensor conv1_w, conv1_b;       /* [C_out, 27, C_in], [C_out] */
    qtensor conv2_w, conv2_b;       /* [C_out, 27, C_out], [C_out] */
    qtensor emb_w,   emb_b;         /* [2*C_out, D_emb], [2*C_out] */
    qtensor skip_w,  skip_b;        /* [C_out, C_in], [C_out] — absent if C_in==C_out */
    int C_in, C_out;
    sam3d_slat_updown_mode updown;
    int has_skip;
} sam3d_slat_io_block;

/* One ModulatedSparseTransformerCrossBlock. Single-stream (unlike the SS DiT's
 * MOT variant), so no per-modality fanout. */
typedef struct {
    /* Shared AdaLN modulation: SiLU + Linear(D → 6D) */
    qtensor adaln_w, adaln_b;               /* [6D, D], [6D] */
    /* norm1: elementwise_affine=False — no params */
    qtensor norm2_w, norm2_b;               /* [D] — elementwise_affine=True */
    /* norm3: elementwise_affine=False — no params */
    /* Self-attention (full sparse) */
    qtensor sa_qkv_w, sa_qkv_b;             /* [3D, D], [3D] */
    qtensor sa_out_w, sa_out_b;             /* [D, D],  [D] */
    qtensor sa_q_rms_gamma, sa_k_rms_gamma; /* [H, head_dim] */
    /* Cross-attention (dense over cond tokens) */
    qtensor xa_q_w,  xa_q_b;                /* [D, D] */
    qtensor xa_kv_w, xa_kv_b;               /* [2D, D_ctx=D] */
    qtensor xa_out_w, xa_out_b;             /* [D, D] */
    /* FFN */
    qtensor mlp_fc1_w, mlp_fc1_b;           /* [4D, D] */
    qtensor mlp_fc2_w, mlp_fc2_b;           /* [D, 4D] */
} sam3d_slat_block;

typedef struct {
    int dim;                /* 1024 */
    int n_heads;            /* 16 */
    int head_dim;           /* 64 */
    int cond_channels;      /* 1024 */
    int in_channels;        /* 8 */
    int out_channels;       /* 8 */
    int freq_dim;            /* 256 */
    int n_blocks;           /* 24 */
    int n_io_res_blocks;    /* 2 */
    int resolution;         /* 64 (post-patch: 32) */
    int patch_size;         /* 2 */
    float mlp_ratio;        /* 4.0 */
    float ln_eps;           /* 1e-6 */

    /* input_layer: Linear(8 → io_block_channels[0]=128) */
    qtensor input_w, input_b;               /* [128, 8], [128] */
    /* out_layer: Linear(io_block_channels[0]=128 → 8) */
    qtensor out_w, out_b;                   /* [8, 128], [8] */

    /* TimestepEmbedder MLP */
    qtensor t_emb_fc1_w, t_emb_fc1_b;       /* [D, freq_dim], [D] */
    qtensor t_emb_fc2_w, t_emb_fc2_b;       /* [D, D], [D] */

    /* input_blocks / out_blocks (num_io_res_blocks each: last is downsample /
     * first is upsample). */
    sam3d_slat_io_block *input_blocks;      /* [n_io_res_blocks] */
    sam3d_slat_io_block *out_blocks;        /* [n_io_res_blocks] */

    /* Transformer blocks */
    sam3d_slat_block *blocks;               /* [n_blocks] */

    st_context *st_ctx;
} sam3d_slat_dit_model;

sam3d_slat_dit_model *sam3d_slat_dit_load_safetensors(const char *path);
void                  sam3d_slat_dit_free(sam3d_slat_dit_model *m);

/* Forward pass. Reserved for step 7b. Returns 0 on success, <0 on failure.
 * The tensor at *x may be replaced (resampled to different N / C); callers
 * must pass the address of their sp3d_tensor* so the resampled pointer is
 * propagated back. The old tensor is freed internally. */
int sam3d_slat_dit_forward(const sam3d_slat_dit_model *m,
                            sp3d_tensor **x,            /* in/out sparse (feats=8 ch) */
                            float t,                     /* flow matching timestep */
                            const float *cond, int n_cond,
                            int n_threads);

/* Optional integration hook used by CUDA runners that still keep the sparse
 * input/output resblocks on CPU but want to replace the dense 24-block
 * transformer stack. The hook receives post-APE feats [N,dim] in-place and
 * must write post-transformer feats back into the same buffer. */
typedef int (*sam3d_slat_dit_transformer_hook_fn)(void *user,
                                                  float *feats, int N,
                                                  const int32_t *coords,
                                                  const float *t_emb,
                                                  const float *cond,
                                                  int n_cond,
                                                  int dim, int n_blocks);
void sam3d_slat_dit_set_transformer_hook(sam3d_slat_dit_transformer_hook_fn fn,
                                         void *user);

/* Optional hook for replacing APE + all transformer blocks as one device
 * boundary. This avoids doing APE on host before a GPU transformer hook.
 * The hook receives pre-APE feats and must leave post-transformer feats. */
typedef int (*sam3d_slat_dit_ape_transformer_hook_fn)(void *user,
                                                      float *feats, int N,
                                                      const int32_t *coords,
                                                      const float *t_emb,
                                                      const float *cond,
                                                      int n_cond,
                                                      int dim, int n_blocks);
void sam3d_slat_dit_set_ape_transformer_hook(sam3d_slat_dit_ape_transformer_hook_fn fn,
                                             void *user);

/* Optional hook for input_layer SparseLinear(8 -> io_block_channels[0]).
 * The hook must replace *xp features with width out_channels. */
typedef int (*sam3d_slat_dit_input_layer_hook_fn)(void *user,
                                                  sp3d_tensor **xp,
                                                  const qtensor *input_w,
                                                  const qtensor *input_b,
                                                  int out_channels);
void sam3d_slat_dit_set_input_layer_hook(sam3d_slat_dit_input_layer_hook_fn fn,
                                         void *user);

/* Optional hook for replacing individual sparse IO SparseResBlock3d calls.
 * The hook must mutate *xp exactly like slat_resblock_forward: apply any
 * up/down sampling, replace coords/features, and preserve batch_size. */
typedef int (*sam3d_slat_dit_io_block_hook_fn)(void *user,
                                               int is_output,
                                               int block_idx,
                                               const sam3d_slat_io_block *bk,
                                               sp3d_tensor **xp,
                                               const float *t_emb,
                                               const int32_t *up_target_coords,
                                               int up_target_N,
                                               int dim,
                                               float ln_eps);
void sam3d_slat_dit_set_io_block_hook(sam3d_slat_dit_io_block_hook_fn fn,
                                      void *user);

/* Optional hook for the final no-affine LayerNorm + SparseLinear out_layer.
 * The hook must replace *xp features with width out_channels. */
typedef int (*sam3d_slat_dit_final_layer_hook_fn)(void *user,
                                                  sp3d_tensor **xp,
                                                  const qtensor *out_w,
                                                  const qtensor *out_b,
                                                  int out_channels,
                                                  float eps);
void sam3d_slat_dit_set_final_layer_hook(sam3d_slat_dit_final_layer_hook_fn fn,
                                         void *user);

#ifdef __cplusplus
}
#endif

#endif /* SAM3D_SLAT_DIT_H */

/* ================================================================== */
#ifdef SAM3D_SLAT_DIT_IMPLEMENTATION
#ifndef SAM3D_SLAT_DIT_IMPL_ONCE
#define SAM3D_SLAT_DIT_IMPL_ONCE

static sam3d_slat_dit_transformer_hook_fn g_slat_transformer_hook = NULL;
static void *g_slat_transformer_hook_user = NULL;
static sam3d_slat_dit_ape_transformer_hook_fn g_slat_ape_transformer_hook = NULL;
static void *g_slat_ape_transformer_hook_user = NULL;
static sam3d_slat_dit_input_layer_hook_fn g_slat_input_layer_hook = NULL;
static void *g_slat_input_layer_hook_user = NULL;
static sam3d_slat_dit_io_block_hook_fn g_slat_io_block_hook = NULL;
static void *g_slat_io_block_hook_user = NULL;
static sam3d_slat_dit_final_layer_hook_fn g_slat_final_layer_hook = NULL;
static void *g_slat_final_layer_hook_user = NULL;

void sam3d_slat_dit_set_transformer_hook(sam3d_slat_dit_transformer_hook_fn fn,
                                         void *user)
{
    g_slat_transformer_hook = fn;
    g_slat_transformer_hook_user = user;
}

void sam3d_slat_dit_set_ape_transformer_hook(sam3d_slat_dit_ape_transformer_hook_fn fn,
                                             void *user)
{
    g_slat_ape_transformer_hook = fn;
    g_slat_ape_transformer_hook_user = user;
}

void sam3d_slat_dit_set_input_layer_hook(sam3d_slat_dit_input_layer_hook_fn fn,
                                         void *user)
{
    g_slat_input_layer_hook = fn;
    g_slat_input_layer_hook_user = user;
}

void sam3d_slat_dit_set_io_block_hook(sam3d_slat_dit_io_block_hook_fn fn,
                                      void *user)
{
    g_slat_io_block_hook = fn;
    g_slat_io_block_hook_user = user;
}

void sam3d_slat_dit_set_final_layer_hook(sam3d_slat_dit_final_layer_hook_fn fn,
                                         void *user)
{
    g_slat_final_layer_hook = fn;
    g_slat_final_layer_hook_user = user;
}

/* ---- Loader: io_block (input or out) ---- */
static int sam3d_slat_load_io_block(st_context *ctx, const char *prefix, int idx,
                                     sam3d_slat_io_block *b,
                                     sam3d_slat_updown_mode updown) {
    char buf[256];
    int rc = 0;
#define FIND(field, fmt) do {                                    \
    snprintf(buf, sizeof(buf), "%s.%d." fmt, prefix, idx);       \
    rc |= qt_find(ctx, buf, &b->field);                   \
} while (0)
#define FIND_OPT(field, fmt) do {                                \
    snprintf(buf, sizeof(buf), "%s.%d." fmt, prefix, idx);       \
    qt_find_opt(ctx, buf, &b->field);                     \
} while (0)
    FIND(norm1_w, "norm1.weight");
    FIND(norm1_b, "norm1.bias");
    FIND(conv1_w, "conv1.conv.weight");
    FIND(conv1_b, "conv1.conv.bias");
    FIND(conv2_w, "conv2.conv.weight");
    FIND(conv2_b, "conv2.conv.bias");
    FIND(emb_w,   "emb_layers.1.weight");
    FIND(emb_b,   "emb_layers.1.bias");
    /* skip_connection is absent when C_in == C_out (nn.Identity). */
    FIND_OPT(skip_w, "skip_connection.weight");
    FIND_OPT(skip_b, "skip_connection.bias");
#undef FIND
#undef FIND_OPT
    if (rc) return rc;
    /* conv1 weight is [out_C, 3, 3, 3, in_C] → n_rows=out_C, n_cols=27*C_in. */
    b->C_out = b->conv1_w.n_rows;
    b->C_in  = b->conv1_w.n_cols / 27;
    b->has_skip = (b->skip_w.data != NULL);
    b->updown = updown;
    return 0;
}

/* ---- Loader: one transformer block ---- */
static int sam3d_slat_load_block(st_context *ctx, int idx, sam3d_slat_block *b) {
    char buf[256];
    int rc = 0;
#define FIND(field, fmt) do {                                    \
    snprintf(buf, sizeof(buf), "blocks.%d." fmt, idx);           \
    rc |= qt_find(ctx, buf, &b->field);                   \
} while (0)
    FIND(adaln_w,        "adaLN_modulation.1.weight");
    FIND(adaln_b,        "adaLN_modulation.1.bias");
    FIND(norm2_w,        "norm2.weight");
    FIND(norm2_b,        "norm2.bias");
    FIND(sa_qkv_w,       "self_attn.to_qkv.weight");
    FIND(sa_qkv_b,       "self_attn.to_qkv.bias");
    FIND(sa_out_w,       "self_attn.to_out.weight");
    FIND(sa_out_b,       "self_attn.to_out.bias");
    FIND(sa_q_rms_gamma, "self_attn.q_rms_norm.gamma");
    FIND(sa_k_rms_gamma, "self_attn.k_rms_norm.gamma");
    FIND(xa_q_w,         "cross_attn.to_q.weight");
    FIND(xa_q_b,         "cross_attn.to_q.bias");
    FIND(xa_kv_w,        "cross_attn.to_kv.weight");
    FIND(xa_kv_b,        "cross_attn.to_kv.bias");
    FIND(xa_out_w,       "cross_attn.to_out.weight");
    FIND(xa_out_b,       "cross_attn.to_out.bias");
    FIND(mlp_fc1_w,      "mlp.mlp.0.weight");
    FIND(mlp_fc1_b,      "mlp.mlp.0.bias");
    FIND(mlp_fc2_w,      "mlp.mlp.2.weight");
    FIND(mlp_fc2_b,      "mlp.mlp.2.bias");
#undef FIND
    return rc;
}

sam3d_slat_dit_model *sam3d_slat_dit_load_safetensors(const char *path) {
    st_context *ctx = safetensors_open(path);
    if (!ctx) {
        fprintf(stderr, "sam3d_slat_dit: cannot open %s\n", path);
        return NULL;
    }
    sam3d_slat_dit_model *m =
        (sam3d_slat_dit_model *)calloc(1, sizeof(*m));
    if (!m) { safetensors_close(ctx); return NULL; }
    m->st_ctx = ctx;

    m->dim = 1024;
    m->n_heads = 16;
    m->head_dim = 64;
    m->cond_channels = 1024;
    m->in_channels = 8;
    m->out_channels = 8;
    m->freq_dim = 256;
    m->n_io_res_blocks = 2;
    m->resolution = 64;
    m->patch_size = 2;
    m->mlp_ratio = 4.0f;
    m->ln_eps = 1e-6f;

    int rc = 0;
    rc |= qt_find(ctx, "input_layer.weight", &m->input_w);
    rc |= qt_find(ctx, "input_layer.bias",   &m->input_b);
    rc |= qt_find(ctx, "out_layer.weight",   &m->out_w);
    rc |= qt_find(ctx, "out_layer.bias",     &m->out_b);
    rc |= qt_find(ctx, "t_embedder.mlp.0.weight", &m->t_emb_fc1_w);
    rc |= qt_find(ctx, "t_embedder.mlp.0.bias",   &m->t_emb_fc1_b);
    rc |= qt_find(ctx, "t_embedder.mlp.2.weight", &m->t_emb_fc2_w);
    rc |= qt_find(ctx, "t_embedder.mlp.2.bias",   &m->t_emb_fc2_b);
    if (rc) {
        fprintf(stderr, "sam3d_slat_dit: missing top-level weights\n");
        sam3d_slat_dit_free(m); return NULL;
    }

    /* Count transformer blocks. */
    int n_blocks = 0;
    for (int i = 0; i < 64; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "blocks.%d.adaLN_modulation.1.weight", i);
        if (safetensors_find(ctx, buf) < 0) break;
        n_blocks = i + 1;
    }
    if (n_blocks == 0) {
        fprintf(stderr, "sam3d_slat_dit: no transformer blocks found\n");
        sam3d_slat_dit_free(m); return NULL;
    }
    m->n_blocks = n_blocks;
    m->blocks = (sam3d_slat_block *)calloc(n_blocks, sizeof(sam3d_slat_block));
    if (!m->blocks) { sam3d_slat_dit_free(m); return NULL; }
    for (int i = 0; i < n_blocks; i++) {
        if (sam3d_slat_load_block(ctx, i, &m->blocks[i]) != 0) {
            fprintf(stderr, "sam3d_slat_dit: block %d load failed\n", i);
            sam3d_slat_dit_free(m); return NULL;
        }
    }

    /* input_blocks: last one is the downsample (updown=DOWN). */
    m->input_blocks = (sam3d_slat_io_block *)calloc(m->n_io_res_blocks,
                                                      sizeof(sam3d_slat_io_block));
    if (!m->input_blocks) { sam3d_slat_dit_free(m); return NULL; }
    for (int i = 0; i < m->n_io_res_blocks; i++) {
        sam3d_slat_updown_mode u = (i == m->n_io_res_blocks - 1) ?
                                        SAM3D_SLAT_UPDOWN_DOWN :
                                        SAM3D_SLAT_UPDOWN_NONE;
        if (sam3d_slat_load_io_block(ctx, "input_blocks", i,
                                      &m->input_blocks[i], u) != 0) {
            fprintf(stderr, "sam3d_slat_dit: input_block %d load failed\n", i);
            sam3d_slat_dit_free(m); return NULL;
        }
    }

    /* out_blocks: first one is the upsample (updown=UP). */
    m->out_blocks = (sam3d_slat_io_block *)calloc(m->n_io_res_blocks,
                                                    sizeof(sam3d_slat_io_block));
    if (!m->out_blocks) { sam3d_slat_dit_free(m); return NULL; }
    for (int i = 0; i < m->n_io_res_blocks; i++) {
        sam3d_slat_updown_mode u = (i == 0) ? SAM3D_SLAT_UPDOWN_UP
                                             : SAM3D_SLAT_UPDOWN_NONE;
        if (sam3d_slat_load_io_block(ctx, "out_blocks", i,
                                      &m->out_blocks[i], u) != 0) {
            fprintf(stderr, "sam3d_slat_dit: out_block %d load failed\n", i);
            sam3d_slat_dit_free(m); return NULL;
        }
    }

    fprintf(stderr,
            "sam3d_slat_dit: loaded %d transformer blocks, "
            "%d input/out res blocks, dim=%d heads=%d\n",
            m->n_blocks, m->n_io_res_blocks, m->dim, m->n_heads);
    for (int i = 0; i < m->n_io_res_blocks; i++) {
        fprintf(stderr,
                "  input_blocks[%d] C_in=%d C_out=%d updown=%d skip=%d\n",
                i, m->input_blocks[i].C_in, m->input_blocks[i].C_out,
                m->input_blocks[i].updown, m->input_blocks[i].has_skip);
    }
    for (int i = 0; i < m->n_io_res_blocks; i++) {
        fprintf(stderr,
                "  out_blocks[%d]   C_in=%d C_out=%d updown=%d skip=%d\n",
                i, m->out_blocks[i].C_in, m->out_blocks[i].C_out,
                m->out_blocks[i].updown, m->out_blocks[i].has_skip);
    }
    return m;
}

void sam3d_slat_dit_free(sam3d_slat_dit_model *m) {
    if (!m) return;
    free(m->blocks);
    free(m->input_blocks);
    free(m->out_blocks);
    if (m->st_ctx) safetensors_close(m->st_ctx);
    free(m);
}

/* ---- Local helpers (small wrappers; reuse ssdit_* from same TU) ---- */

static void slat_dequant_to_f32(const qtensor *t, float *dst, int n) {
    if (!t || !t->data) { memset(dst, 0, (size_t)n * sizeof(float)); return; }
    if (t->type == GGML_TYPE_F32) {
        memcpy(dst, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) {
            uint16_t h = src[i];
            uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m_ = h & 0x3FF;
            uint32_t bits;
            if (e == 0) {
                if (m_ == 0) bits = s << 31;
                else { e = 1; while (!(m_ & 0x400)) { m_ <<= 1; e--; }
                       m_ &= 0x3FF; bits = (s << 31) | ((e + 112) << 23) | (m_ << 13); }
            } else if (e == 31) bits = (s << 31) | 0x7F800000 | (m_ << 13);
            else                bits = (s << 31) | ((e + 112) << 23) | (m_ << 13);
            memcpy(&dst[i], &bits, 4);
        }
    } else {
        memset(dst, 0, (size_t)n * sizeof(float));
    }
}

/* APE: per-axis sin/cos at freq_dim freqs, concat along axis-0 then channel,
 * pad zeros to dim. coords are (z, y, x) i32. */
static void slat_apply_ape(float *feats, const int32_t *coords, int N, int dim) {
    int in_axes = 3;
    int freq_dim = dim / in_axes / 2;        /* 1024/3/2 = 170 */
    int per_axis = freq_dim * 2;             /* 340 */
    int filled   = per_axis * in_axes;       /* 1020 */
    /* freqs[j] = 1.0 / 10000^(j/freq_dim) */
    float *freqs = (float *)malloc((size_t)freq_dim * sizeof(float));
    for (int j = 0; j < freq_dim; j++)
        freqs[j] = 1.0f / powf(10000.0f, (float)j / (float)freq_dim);
    for (int i = 0; i < N; i++) {
        float *row = feats + (size_t)i * dim;
        /* coord layout: (batch, z, y, x); we use (z, y, x) at offset 1..3. */
        for (int a = 0; a < in_axes; a++) {
            float v = (float)coords[i * 4 + 1 + a];
            for (int j = 0; j < freq_dim; j++) {
                float arg = v * freqs[j];
                /* Per upstream _sin_cos_embedding: out=cat([sin(out), cos(out)], -1)
                 * with out=[N, freq_dim] before cat, then reshape to [N, in*per_axis].
                 * For a single coord the per-axis layout is [sin(j=0..F-1), cos(j=0..F-1)]. */
                row[a * per_axis + j]            += sinf(arg);
                row[a * per_axis + freq_dim + j] += cosf(arg);
            }
        }
        /* zero-pad tail [filled..dim) — we add nothing (already accumulated). */
        (void)filled;
    }
    free(freqs);
}

/* Run one SparseResBlock3d. Mutates *xp (sp3d_tensor pointer):
 *   on entry, *xp owns coords+feats (C=block->C_in)
 *   on exit,  *xp owns possibly-resampled coords with feats of width C_out.
 * up_target_coords/up_target_N are required iff updown==UP.
 * t_emb is shared [dim] vector; nthr is #threads. */
static void slat_trace_npy(const char *dir, const char *name,
                            const float *data, int rank, const int *shape);

static int slat_resblock_forward(const sam3d_slat_dit_model *m,
                                  const sam3d_slat_io_block *bk,
                                  sp3d_tensor **xp,
                                  const float *t_emb,
                                  const int32_t *up_target_coords,
                                  int up_target_N,
                                  int nthr) {
    sp3d_tensor *x = *xp;
    int dim = m->dim;
    int C_in = bk->C_in, C_out = bk->C_out;

    /* emb_out = Linear(SiLU(t_emb)) → [1, 2*C_out]. emb_w shape [2*C_out, dim]. */
    float *t_silu = (float *)malloc((size_t)dim * sizeof(float));
    memcpy(t_silu, t_emb, (size_t)dim * sizeof(float));
    ssdit_silu_inplace(t_silu, dim);
    float *emb_out = (float *)malloc((size_t)2 * C_out * sizeof(float));
    ssdit_gemm(emb_out, &bk->emb_w, &bk->emb_b, t_silu, 1, 2 * C_out, dim, nthr);
    free(t_silu);
    const float *scale = emb_out;            /* [C_out] */
    const float *shift = emb_out + C_out;    /* [C_out] */

    /* _updown(x). For UP we need target coords. Downsample uses pool_mode=2
     * (mean with include_self=True) to match pytorch scatter_reduce semantics —
     * effectively dividing by (count+1) not count. */
    sp3d_tensor *xu = x;
    if (bk->updown == SAM3D_SLAT_UPDOWN_DOWN) {
        xu = sp3d_downsample(x, 2, 2);
        sp3d_free(x);
    } else if (bk->updown == SAM3D_SLAT_UPDOWN_UP) {
        xu = sp3d_upsample(x, 2, up_target_coords, up_target_N);
        sp3d_free(x);
    }
    int N = xu->N;

    /* skip = skip_connection(xu): [N, C_out] (Identity if C_in==C_out) */
    float *skip = (float *)malloc((size_t)N * C_out * sizeof(float));
    if (bk->has_skip) {
        ssdit_gemm(skip, &bk->skip_w, &bk->skip_b, xu->feats,
                   N, C_out, C_in, nthr);
    } else {
        memcpy(skip, xu->feats, (size_t)N * C_out * sizeof(float));
    }

    /* h = norm1(xu->feats) [affine, no modulation], then SiLU. */
    float *h = (float *)malloc((size_t)N * C_in * sizeof(float));
    ssdit_layernorm(h, xu->feats, &bk->norm1_w, &bk->norm1_b,
                    NULL, NULL, N, C_in, m->ln_eps);
    ssdit_silu_inplace(h, N * C_in);

    /* h via conv1. sp3d_conv3d_forward reads feats off the tensor; temporarily
     * swap in our post-norm h, then restore. */
    float *w1 = (float *)malloc((size_t)C_out * 27 * C_in * sizeof(float));
    slat_dequant_to_f32(&bk->conv1_w, w1, C_out * 27 * C_in);
    float *b1 = (float *)malloc((size_t)C_out * sizeof(float));
    slat_dequant_to_f32(&bk->conv1_b, b1, C_out);
    float *saved = xu->feats;
    xu->feats = h;
    int sC = xu->C;
    xu->C = C_in;
    float *h2 = (float *)malloc((size_t)N * C_out * sizeof(float));
    sp3d_conv3d_forward(h2, xu, w1, b1, C_in, C_out, 3, nthr);
    xu->feats = saved;
    xu->C = sC;
    free(w1); free(b1); free(h);

    /* h2 = norm2(h2) [no affine] then * (1 + scale) + shift (broadcast over N). */
    float *h3 = (float *)malloc((size_t)N * C_out * sizeof(float));
    ssdit_layernorm(h3, h2, NULL, NULL, shift, scale, N, C_out, m->ln_eps);
    /* h3 = SiLU(h3) */
    ssdit_silu_inplace(h3, N * C_out);

    /* h3 → conv2: same swap trick as conv1. */
    float *w2 = (float *)malloc((size_t)C_out * 27 * C_out * sizeof(float));
    slat_dequant_to_f32(&bk->conv2_w, w2, C_out * 27 * C_out);
    float *b2 = (float *)malloc((size_t)C_out * sizeof(float));
    slat_dequant_to_f32(&bk->conv2_b, b2, C_out);
    saved = xu->feats; sC = xu->C;
    xu->feats = h3; xu->C = C_out;
    float *h4 = (float *)malloc((size_t)N * C_out * sizeof(float));
    sp3d_conv3d_forward(h4, xu, w2, b2, C_out, C_out, 3, nthr);
    xu->feats = saved; xu->C = sC;
    free(w2); free(b2); free(h2); free(h3);

    /* h_out = h4 + skip */
    for (int i = 0; i < N * C_out; i++) h4[i] += skip[i];
    free(skip);
    free(emb_out);

    /* Replace xu's feats with h4 (width C_out). */
    free(xu->feats);
    xu->feats = h4;
    xu->C = C_out;
    *xp = xu;
    return 0;
}

/* One ModulatedSparseTransformerCrossBlock. Mutates x->feats in place. */
static int slat_transformer_block(const sam3d_slat_dit_model *m,
                                   const sam3d_slat_block *blk,
                                   sp3d_tensor *x,
                                   const float *t_emb,
                                   const float *cond, int n_cond,
                                   int nthr) {
    int dim = m->dim;
    int H = m->n_heads;
    int hd = m->head_dim;
    int N = x->N;

    /* mod = Linear(SiLU(t_emb)) → [1, 6D]. Then chunk(6, dim=1) → 6 [D] vectors. */
    float *t_silu = (float *)malloc((size_t)dim * sizeof(float));
    memcpy(t_silu, t_emb, (size_t)dim * sizeof(float));
    ssdit_silu_inplace(t_silu, dim);
    float *mod6 = (float *)malloc((size_t)6 * dim * sizeof(float));
    ssdit_gemm(mod6, &blk->adaln_w, &blk->adaln_b, t_silu, 1, 6 * dim, dim, nthr);
    free(t_silu);
    const float *shift_msa = mod6 + 0 * dim;
    const float *scale_msa = mod6 + 1 * dim;
    const float *gate_msa  = mod6 + 2 * dim;
    const float *shift_mlp = mod6 + 3 * dim;
    const float *scale_mlp = mod6 + 4 * dim;
    const float *gate_mlp  = mod6 + 5 * dim;

    /* h = norm1(x.feats) [no affine] then * (1+scale_msa) + shift_msa */
    float *h = (float *)malloc((size_t)N * dim * sizeof(float));
    ssdit_layernorm(h, x->feats, NULL, NULL, shift_msa, scale_msa,
                    N, dim, m->ln_eps);

    /* qkv = to_qkv(h) → [N, 3D] */
    float *qkv = (float *)malloc((size_t)N * 3 * dim * sizeof(float));
    ssdit_gemm(qkv, &blk->sa_qkv_w, &blk->sa_qkv_b, h, N, 3 * dim, dim, nthr);
    /* qk_rms_norm: per-head L2-norm * gamma * sqrt(head_dim).
     * qkv layout per token: [Q(dim), K(dim), V(dim)] each as [H, hd]. */
    int qo = 0, ko = 0;
    const float *gq = ssdit_row_f32(&blk->sa_q_rms_gamma, &qo);
    const float *gk = ssdit_row_f32(&blk->sa_k_rms_gamma, &ko);
    /* Q at offset 0, stride 3*dim; K at offset dim, stride 3*dim. */
    ssdit_mhrmsnorm(qkv,             N, H, hd, 3 * dim, gq);
    ssdit_mhrmsnorm(qkv + dim,       N, H, hd, 3 * dim, gk);
    if (qo) free((void *)gq);
    if (ko) free((void *)gk);
    /* Sparse self-attn (full): output [N, dim]. */
    float *sa_out = (float *)malloc((size_t)N * dim * sizeof(float));
    sp3d_attention(sa_out, qkv, x, H, hd, nthr);
    free(qkv);
    /* to_out projection */
    float *sa_proj = (float *)malloc((size_t)N * dim * sizeof(float));
    ssdit_gemm(sa_proj, &blk->sa_out_w, &blk->sa_out_b, sa_out, N, dim, dim, nthr);
    free(sa_out);
    /* Apply gate_msa, residual into x. */
    for (int i = 0; i < N; i++)
        for (int c = 0; c < dim; c++)
            x->feats[i * dim + c] += sa_proj[i * dim + c] * gate_msa[c];
    free(sa_proj);

    /* h = norm2(x) [affine] → cross_attn(h, cond) → residual */
    ssdit_layernorm(h, x->feats, &blk->norm2_w, &blk->norm2_b,
                    NULL, NULL, N, dim, m->ln_eps);
    float *q_x  = (float *)malloc((size_t)N      * dim     * sizeof(float));
    float *kv_c = (float *)malloc((size_t)n_cond * 2 * dim * sizeof(float));
    float *xa   = (float *)malloc((size_t)N      * dim     * sizeof(float));
    ssdit_gemm(q_x,  &blk->xa_q_w,  &blk->xa_q_b,  h,    N,      dim,     dim, nthr);
    ssdit_gemm(kv_c, &blk->xa_kv_w, &blk->xa_kv_b, cond, n_cond, 2 * dim, dim, nthr);
    cpu_cross_attention(xa, q_x, kv_c, N, n_cond, dim, H, hd, nthr);
    free(q_x); free(kv_c);
    float *xa_proj = (float *)malloc((size_t)N * dim * sizeof(float));
    ssdit_gemm(xa_proj, &blk->xa_out_w, &blk->xa_out_b, xa, N, dim, dim, nthr);
    free(xa);
    for (int i = 0; i < N * dim; i++) x->feats[i] += xa_proj[i];
    free(xa_proj);

    /* norm3(x) [no affine] then * (1+scale_mlp) + shift_mlp → MLP → gate_mlp + residual */
    ssdit_layernorm(h, x->feats, NULL, NULL, shift_mlp, scale_mlp,
                    N, dim, m->ln_eps);
    int mlp_hidden = (int)(m->mlp_ratio * (float)dim + 0.5f);
    float *mh = (float *)malloc((size_t)N * mlp_hidden * sizeof(float));
    ssdit_gemm(mh, &blk->mlp_fc1_w, &blk->mlp_fc1_b, h, N, mlp_hidden, dim, nthr);
    ssdit_gelu_tanh_inplace(mh, N * mlp_hidden);
    float *mh2 = (float *)malloc((size_t)N * dim * sizeof(float));
    ssdit_gemm(mh2, &blk->mlp_fc2_w, &blk->mlp_fc2_b, mh, N, dim, mlp_hidden, nthr);
    free(mh);
    for (int i = 0; i < N; i++)
        for (int c = 0; c < dim; c++)
            x->feats[i * dim + c] += mh2[i * dim + c] * gate_mlp[c];
    free(mh2);

    free(h);
    free(mod6);
    return 0;
}

/* Debug dumping: when SLAT_DIT_TRACE=1 in env, write intermediate stage
 * tensors as .npy under $SLAT_DIT_TRACE_DIR (default /tmp/sam3d_ref). */
static void slat_trace_npy(const char *dir, const char *name,
                            const float *data, int rank, const int *shape) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.npy", dir, name);
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    /* numpy magic */
    fwrite("\x93NUMPY", 1, 6, fp);
    unsigned char ver[2] = {1, 0};
    fwrite(ver, 1, 2, fp);
    char hdr[256];
    int p = snprintf(hdr, sizeof(hdr),
                     "{'descr': '<f4', 'fortran_order': False, 'shape': (");
    for (int i = 0; i < rank; i++) {
        p += snprintf(hdr + p, sizeof(hdr) - p, "%d%s", shape[i],
                      (rank == 1 || i + 1 < rank) ? ", " : "");
    }
    p += snprintf(hdr + p, sizeof(hdr) - p, "), }");
    int total = 10 + p;
    int pad = (16 - (total % 16)) % 16;
    int hdr_len = p + pad;
    while (pad-- > 0) hdr[p++] = ' ';
    hdr[p - 1] = '\n';
    unsigned short hl = (unsigned short)hdr_len;
    fwrite(&hl, 2, 1, fp);
    fwrite(hdr, 1, hdr_len, fp);
    long n = 1;
    for (int i = 0; i < rank; i++) n *= shape[i];
    fwrite(data, 4, n, fp);
    fclose(fp);
}

int sam3d_slat_dit_forward(const sam3d_slat_dit_model *m,
                            sp3d_tensor **xp, float t,
                            const float *cond, int n_cond,
                            int n_threads) {
    if (!m || !xp || !*xp || !cond) return -1;
    if (n_threads <= 0) n_threads = 1;
    sp3d_tensor *x = *xp;
    int dim = m->dim;
    int nthr = n_threads;
    const char *trace = getenv("SLAT_DIT_TRACE");
    const char *tdir  = getenv("SLAT_DIT_TRACE_DIR");
    if (!tdir) tdir = "/tmp/sam3d_ref";
    int do_trace = (trace && trace[0] && trace[0] != '0');

    /* --- Timestep embedding [dim]. SLAT TimestepEmbedder reuses the one from
     * sparse_structure_flow.py and feeds t directly (no time_factor scaling). */
    float *t_emb = (float *)malloc((size_t)dim * sizeof(float));
    ssdit_time_mlp(t_emb, t,
                   &m->t_emb_fc1_w, &m->t_emb_fc1_b,
                   &m->t_emb_fc2_w, &m->t_emb_fc2_b,
                   m->freq_dim, dim);
    if (do_trace) {
        int sh[1] = {dim};
        slat_trace_npy(tdir, "c_t_emb", t_emb, 1, sh);
    }

    /* --- input_layer: SparseLinear(C_in=8 → 128) on x->feats --- */
    int io_C0 = m->input_blocks[0].C_in;     /* 128 */
    if (g_slat_input_layer_hook) {
        if (g_slat_input_layer_hook(g_slat_input_layer_hook_user, &x,
                                    &m->input_w, &m->input_b, io_C0) != 0) {
            fprintf(stderr, "slat: input layer hook failed\n");
            *xp = x; return -1;
        }
    } else {
        float *h0 = (float *)malloc((size_t)x->N * io_C0 * sizeof(float));
        sp3d_linear(h0, x->feats, x->N, &m->input_w, &m->input_b,
                    io_C0, m->in_channels, nthr);
        free(x->feats);
        x->feats = h0;
        x->C = io_C0;
    }
    if (do_trace) {
        int sh[2] = {x->N, io_C0};
        slat_trace_npy(tdir, "c_h_after_input_layer", x->feats, 2, sh);
    }

    /* --- input_blocks (with skip stash) --- */
    int n_io = m->n_io_res_blocks;
    /* Per-skip: copy feats and coords (we own these). */
    float    **skip_feats  = (float    **)calloc((size_t)n_io, sizeof(float *));
    int32_t  **skip_coords = (int32_t  **)calloc((size_t)n_io, sizeof(int32_t *));
    int       *skip_N      = (int       *)calloc((size_t)n_io, sizeof(int));
    int       *skip_C      = (int       *)calloc((size_t)n_io, sizeof(int));
    for (int i = 0; i < n_io; i++) {
        int brc = g_slat_io_block_hook
            ? g_slat_io_block_hook(g_slat_io_block_hook_user, 0, i,
                                   &m->input_blocks[i], &x, t_emb,
                                   NULL, 0, dim, m->ln_eps)
            : slat_resblock_forward(m, &m->input_blocks[i], &x, t_emb, NULL, 0, nthr);
        if (brc != 0) {
            fprintf(stderr, "slat: input_block[%d] failed\n", i);
            *xp = x; return -2;
        }
        skip_N[i] = x->N;
        skip_C[i] = x->C;
        skip_feats[i]  = (float *)malloc((size_t)x->N * x->C * sizeof(float));
        skip_coords[i] = (int32_t *)malloc((size_t)x->N * 4   * sizeof(int32_t));
        memcpy(skip_feats[i],  x->feats,  (size_t)x->N * x->C * sizeof(float));
        memcpy(skip_coords[i], x->coords, (size_t)x->N * 4   * sizeof(int32_t));
        if (do_trace) {
            char nm[64]; snprintf(nm, sizeof(nm), "c_h_after_input_block_%d", i);
            int sh[2] = {x->N, x->C};
            slat_trace_npy(tdir, nm, x->feats, 2, sh);
            char nmc[64]; snprintf(nmc, sizeof(nmc), "c_coords_after_input_block_%d", i);
            int shc[2] = {x->N, 4};
            float *cf = (float *)malloc((size_t)x->N * 4 * sizeof(float));
            for (int k = 0; k < x->N * 4; k++) cf[k] = (float)x->coords[k];
            slat_trace_npy(tdir, nmc, cf, 2, shc);
            free(cf);
        }
    }

    /* --- APE pos emb + Transformer blocks --- */
    if (g_slat_ape_transformer_hook && !do_trace) {
        if (g_slat_ape_transformer_hook(g_slat_ape_transformer_hook_user,
                                        x->feats, x->N, x->coords,
                                        t_emb, cond, n_cond, dim,
                                        m->n_blocks) != 0) {
            fprintf(stderr, "slat: APE+transformer hook failed\n");
            *xp = x; return -3;
        }
    } else {
        slat_apply_ape(x->feats, x->coords, x->N, dim);
        if (do_trace) {
            int sh[2] = {x->N, dim};
            slat_trace_npy(tdir, "c_h_after_ape", x->feats, 2, sh);
        }
    }

    if (!g_slat_ape_transformer_hook || do_trace) {
      if (g_slat_transformer_hook) {
        if (g_slat_transformer_hook(g_slat_transformer_hook_user,
                                    x->feats, x->N, x->coords,
                                    t_emb, cond, n_cond, dim, m->n_blocks) != 0) {
            fprintf(stderr, "slat: transformer hook failed\n");
            *xp = x; return -3;
        }
        if (do_trace) {
            int sh[2] = {x->N, dim};
            slat_trace_npy(tdir, "c_h_after_block_23", x->feats, 2, sh);
        }
      } else {
        for (int b = 0; b < m->n_blocks; b++) {
            if (slat_transformer_block(m, &m->blocks[b], x, t_emb, cond, n_cond, nthr) != 0) {
                fprintf(stderr, "slat: transformer block[%d] failed\n", b);
                *xp = x; return -3;
            }
            if (do_trace && (b == 0 || b == m->n_blocks - 1)) {
                char nm[64]; snprintf(nm, sizeof(nm), "c_h_after_block_%d", b);
                int sh[2] = {x->N, dim};
                slat_trace_npy(tdir, nm, x->feats, 2, sh);
            }
        }
      }
    }

    /* --- out_blocks (skip-concat reversed) --- */
    for (int oi = 0; oi < n_io; oi++) {
        int si = n_io - 1 - oi;
        const sam3d_slat_io_block *bk = &m->out_blocks[oi];
        /* Concat x->feats and skip[si]->feats along feature dim. Both must
         * share row layout (same N and same coord ordering). */
        int N_pre = x->N;
        int Cx = x->C;
        int Cs = skip_C[si];
        int Cc = Cx + Cs;
        if (N_pre != skip_N[si]) {
            fprintf(stderr, "slat: out_block[%d] skip N mismatch %d vs %d\n",
                    oi, N_pre, skip_N[si]);
            *xp = x; return -4;
        }
        float *cat = (float *)malloc((size_t)N_pre * Cc * sizeof(float));
        for (int i = 0; i < N_pre; i++) {
            memcpy(cat + (size_t)i * Cc,        x->feats        + (size_t)i * Cx, (size_t)Cx * sizeof(float));
            memcpy(cat + (size_t)i * Cc + Cx,   skip_feats[si]  + (size_t)i * Cs, (size_t)Cs * sizeof(float));
        }
        free(x->feats);
        x->feats = cat;
        x->C = Cc;
        /* For UP block, target coords are skip[oi-1+1]'s coords in *out* order
         * which is skip_coords[si - 1]? Actually the upsample target should be
         * the coords ONE LEVEL UP — i.e. the next skip in reverse iteration
         * (the skip that out_blocks[oi+1] will consume). */
        const int32_t *up_target_coords = NULL;
        int up_target_N = 0;
        if (bk->updown == SAM3D_SLAT_UPDOWN_UP) {
            int next_si = si - 1;
            if (next_si < 0) {
                fprintf(stderr, "slat: out_block[%d] UP has no upper skip\n", oi);
                *xp = x; return -5;
            }
            up_target_coords = skip_coords[next_si];
            up_target_N      = skip_N[next_si];
        }
        int brc = g_slat_io_block_hook
            ? g_slat_io_block_hook(g_slat_io_block_hook_user, 1, oi,
                                   bk, &x, t_emb, up_target_coords,
                                   up_target_N, dim, m->ln_eps)
            : slat_resblock_forward(m, bk, &x, t_emb, up_target_coords, up_target_N, nthr);
        if (brc != 0) {
            fprintf(stderr, "slat: out_block[%d] failed\n", oi);
            *xp = x; return -6;
        }
        if (do_trace) {
            char nm[64]; snprintf(nm, sizeof(nm), "c_h_after_out_block_%d", oi);
            int sh[2] = {x->N, x->C};
            slat_trace_npy(tdir, nm, x->feats, 2, sh);
        }
    }

    if (g_slat_final_layer_hook) {
        if (g_slat_final_layer_hook(g_slat_final_layer_hook_user, &x,
                                    &m->out_w, &m->out_b,
                                    m->out_channels, 1e-5f) != 0) {
            fprintf(stderr, "slat: final layer hook failed\n");
            *xp = x; return -7;
        }
    } else {
        /* --- F.layer_norm (no affine) on last dim --- */
        float *ln = (float *)malloc((size_t)x->N * x->C * sizeof(float));
        ssdit_layernorm(ln, x->feats, NULL, NULL, NULL, NULL, x->N, x->C, 1e-5f);
        free(x->feats);
        x->feats = ln;

        /* --- out_layer: SparseLinear(128 → 8) --- */
        float *yo = (float *)malloc((size_t)x->N * m->out_channels * sizeof(float));
        sp3d_linear(yo, x->feats, x->N, &m->out_w, &m->out_b,
                    m->out_channels, x->C, nthr);
        free(x->feats);
        x->feats = yo;
        x->C = m->out_channels;
    }

    for (int i = 0; i < n_io; i++) { free(skip_feats[i]); free(skip_coords[i]); }
    free(skip_feats); free(skip_coords); free(skip_N); free(skip_C);
    free(t_emb);
    *xp = x;
    return 0;
}

#endif /* SAM3D_SLAT_DIT_IMPL_ONCE */
#endif /* SAM3D_SLAT_DIT_IMPLEMENTATION */
