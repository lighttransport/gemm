/*
 * sam3d_gs_decoder.h — SLAT → 3D-Gaussian decoder for sam-3d-objects.
 *
 * pytorch: model/backbone/tdfy_dit/models/structured_latent_vae/decoder_gs.py
 *          + model/backbone/tdfy_dit/modules/sparse/transformer/blocks.py
 *                 ::SparseTransformerBlock
 *
 * Architecture (from slat_decoder_gs.yaml):
 *   model_channels = 768, num_blocks = 12, num_heads = 12, head_dim = 64,
 *   mlp_ratio = 4, attn_mode = "swin", window_size = 8, pe_mode = "ape",
 *   latent_channels = 8, resolution = 64, qk_rms_norm = false.
 *   representation_config: num_gaussians=32, voxel_size=1.5,
 *     scaling_bias=0.004, opacity_bias=0.1, scaling_activation=softplus,
 *     perturb_offset=true, lr={_xyz:1, _features_dc:1, _opacity:1,
 *     _scaling:1, _rotation:0.1}.
 *
 * Forward (step 8b):
 *   1. input_layer: SparseLinear(8 → 768) on feats; coords unchanged.
 *   2. h += pos_embedder(coords)              (sinusoidal APE per-axis)
 *   3. 12 × SparseTransformerBlock:
 *        h = x.replace(norm1(x.feats))          no-affine LN
 *        h = swin_attn(h)                        shift-window sparse self-attn,
 *                                                window=8, shift alternates per block
 *        x = x + h
 *        h = x.replace(norm2(x.feats))          no-affine LN
 *        h = mlp(h)                              Linear(768→3072) + GELU(tanh) + Linear(3072→768)
 *        x = x + h
 *   4. F.layer_norm(feats, last_dim_shape) — no affine
 *   5. out_layer: SparseLinear(768 → 448)
 *
 * Output 448 channels = layout:
 *   _xyz          [0..96)   32 gaussians × 3
 *   _features_dc  [96..192) 32 × 3
 *   _scaling      [192..288) 32 × 3
 *   _rotation     [288..416) 32 × 4
 *   _opacity      [416..448) 32 × 1
 *
 * Swin attn config (block_attn_config in base.py):
 *   attn_mode="swin" → windowed with window_size=8,
 *   shift_window = (0,0,0) on even blocks, (4,4,4) on odd blocks.
 *   Partition sparse voxels into 8³ windows by floor((coord-shift)/8);
 *   within each window do full self-attn.
 *
 * Post-forward `to_representation`:
 *   xyz = (coords[:,1:] + 0.5) / resolution                     per-voxel center
 *   offset = feats[:, _xyz_range].reshape(-1, 32, 3) * lr[_xyz]
 *   if perturb_offset: offset += atanh(hammersley(32) * 2 - 1) / voxel_size
 *   offset = tanh(offset) / resolution * 0.5 * voxel_size
 *   gaussian.xyz = (xyz.unsqueeze(1) + offset).reshape(-1, 3)   flatten 32 gaussians
 *   gaussian.features_dc = feats[:, dc] * 1.0                   [N*32, 1, 3]
 *   gaussian.scaling = feats[:, scl] * 1.0                      (activation applied in Gaussian)
 *   gaussian.rotation = feats[:, rot] * 0.1                     [N*32, 4] quaternion
 *   gaussian.opacity = feats[:, op] * 1.0
 *
 * This header is header-only; call `#define SAM3D_GS_DECODER_IMPLEMENTATION`
 * in exactly one TU. Depends on safetensors.h, ggml_dequant.h, sparse3d.h.
 */

#ifndef SAM3D_GS_DECODER_H
#define SAM3D_GS_DECODER_H

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

typedef struct {
    qtensor attn_qkv_w, attn_qkv_b;   /* [3D, D], [3D] */
    qtensor attn_out_w, attn_out_b;   /* [D, D], [D] */
    qtensor mlp_fc1_w,  mlp_fc1_b;    /* [4D, D], [4D] */
    qtensor mlp_fc2_w,  mlp_fc2_b;    /* [D, 4D], [D] */
} sam3d_gs_block;

typedef struct {
    /* Architecture */
    int dim;               /* 768 */
    int n_heads;           /* 12 */
    int head_dim;          /* 64 */
    int n_blocks;          /* 12 */
    int in_channels;       /* 8 */
    int out_channels;      /* 448 */
    int resolution;        /* 64 */
    int window_size;       /* 8 */
    int num_gaussians;     /* 32 */
    float mlp_ratio;       /* 4.0 */
    float ln_eps;          /* 1e-5  (pytorch F.layer_norm default) */

    float voxel_size;      /* 1.5 */
    float scaling_bias;    /* 0.004 */
    float opacity_bias;    /* 0.1 */
    int perturb_offset;    /* 1 */
    /* lr multipliers applied to each layout slice: */
    float lr_xyz;          /* 1.0 */
    float lr_features_dc;  /* 1.0 */
    float lr_scaling;      /* 1.0 */
    float lr_rotation;     /* 0.1 */
    float lr_opacity;      /* 1.0 */

    /* Channel ranges into the 448-wide out feature vector. */
    int r_xyz[2];
    int r_features_dc[2];
    int r_scaling[2];
    int r_rotation[2];
    int r_opacity[2];

    /* Top-level weights */
    qtensor input_w, input_b;                  /* [D, 8], [D] */
    qtensor out_w,   out_b;                    /* [448, D], [448] */
    qtensor offset_perturbation;               /* [num_gaussians, 3] — precomputed Hammersley */

    /* Transformer blocks */
    sam3d_gs_block *blocks;                    /* [n_blocks] */

    st_context *st_ctx;
} sam3d_gs_decoder_model;

sam3d_gs_decoder_model *sam3d_gs_decoder_load_safetensors(const char *path);
void                    sam3d_gs_decoder_free(sam3d_gs_decoder_model *m);

/* Runs input_layer + APE + 12 SparseTransformerBlocks + final no-affine LN
 * + out_layer. Writes [N, out_channels=448] fp32 into *out_feats
 * (caller-owned, malloc'd). Returns 0 on success, <0 on failure. */
int sam3d_gs_decoder_transformer(const sam3d_gs_decoder_model *m,
                                  const sp3d_tensor *x,
                                  float **out_feats,
                                  int n_threads);
/* Same transformer body, stopping before final no-affine LN + out_layer.
 * Returns malloc'd [N, dim] hidden features. Used by the mesh decoder,
 * which shares the sparse transformer trunk with the GS decoder. */
int sam3d_gs_decoder_hidden_transformer(const sam3d_gs_decoder_model *m,
                                        const sp3d_tensor *x,
                                        float **out_h,
                                        int n_threads);

typedef int (*sam3d_gs_input_ape_hook_fn)(void *user,
                                          const int32_t *coords,
                                          const float *feats,
                                          int N, int in_channels,
                                          const qtensor *input_w,
                                          const qtensor *input_b,
                                          int dim,
                                          float **out_h);
typedef int (*sam3d_gs_final_layer_hook_fn)(void *user,
                                            const float *h,
                                            int N, int dim,
                                            const qtensor *out_w,
                                            const qtensor *out_b,
                                            int out_channels,
                                            float eps,
                                            float **out_feats);
typedef int (*sam3d_gs_window_attn_hook_fn)(void *user,
                                            float *out,
                                            const float *qkv,
                                            const sp3d_tensor *x,
                                            int window_size,
                                            const int shift[3],
                                            int n_heads,
                                            int head_dim);
typedef int (*sam3d_gs_attn_block_hook_fn)(void *user,
                                           float *h,
                                           const sp3d_tensor *x,
                                           int N, int dim,
                                           const sam3d_gs_block *blk,
                                           int window_size,
                                           const int shift[3],
                                           int n_heads,
                                           int head_dim,
                                           float eps);
typedef int (*sam3d_gs_mlp_hook_fn)(void *user,
                                    float *h,
                                    int N, int dim,
                                    const sam3d_gs_block *blk,
                                    int hidden,
                                    float eps);
typedef int (*sam3d_gs_block_hook_fn)(void *user,
                                      float *h,
                                      const sp3d_tensor *x,
                                      int N, int dim,
                                      const sam3d_gs_block *blk,
                                      int window_size,
                                      const int shift[3],
                                      int n_heads,
                                      int head_dim,
                                      int hidden,
                                      float eps);
typedef int (*sam3d_gs_stack_hook_fn)(void *user,
                                      float *h,
                                      const sp3d_tensor *x,
                                      int N, int dim,
                                      const sam3d_gs_block *blocks,
                                      int n_blocks,
                                      int window_size,
                                      int n_heads,
                                      int head_dim,
                                      int hidden,
                                      float eps);
typedef int (*sam3d_gs_transformer_hook_fn)(void *user,
                                            const sp3d_tensor *x,
                                            const sam3d_gs_decoder_model *m,
                                            float **out_feats);
void sam3d_gs_decoder_set_input_ape_hook(sam3d_gs_input_ape_hook_fn fn,
                                         void *user);
void sam3d_gs_decoder_set_final_layer_hook(sam3d_gs_final_layer_hook_fn fn,
                                           void *user);
void sam3d_gs_decoder_set_window_attn_hook(sam3d_gs_window_attn_hook_fn fn,
                                           void *user);
void sam3d_gs_decoder_set_attn_block_hook(sam3d_gs_attn_block_hook_fn fn,
                                          void *user);
void sam3d_gs_decoder_set_mlp_hook(sam3d_gs_mlp_hook_fn fn,
                                   void *user);
void sam3d_gs_decoder_set_block_hook(sam3d_gs_block_hook_fn fn,
                                     void *user);
void sam3d_gs_decoder_set_stack_hook(sam3d_gs_stack_hook_fn fn,
                                     void *user);
void sam3d_gs_decoder_set_transformer_hook(sam3d_gs_transformer_hook_fn fn,
                                           void *user);

/* Apply to_representation on [N, 448] feats. Each output pointer may be
 * NULL to skip. Output shapes: xyz [N*G, 3] in (z,y,x) order matching the
 * pytorch ref; dc [N*G, 1, 3]; scaling [N*G, 3]; rotation [N*G, 4];
 * opacity [N*G, 1]. Buffers must be caller-allocated. */
int sam3d_gs_decoder_to_representation(const sam3d_gs_decoder_model *m,
                                        const int32_t *coords,
                                        const float   *feats_448,
                                        int N,
                                        float *xyz_out,
                                        float *dc_out,
                                        float *scaling_out,
                                        float *rotation_out,
                                        float *opacity_out);

#ifdef __cplusplus
}
#endif

#endif /* SAM3D_GS_DECODER_H */

/* ================================================================== */
#ifdef SAM3D_GS_DECODER_IMPLEMENTATION
#ifndef SAM3D_GS_DECODER_IMPL_ONCE
#define SAM3D_GS_DECODER_IMPL_ONCE

/* Need cpu_attention() for the windowed self-attn inner loop. */
#ifndef CPU_COMPUTE_H
#define CPU_COMPUTE_IMPLEMENTATION
#include "cpu_compute.h"
#endif

static sam3d_gs_input_ape_hook_fn g_gs_input_ape_hook = NULL;
static void *g_gs_input_ape_hook_user = NULL;
static sam3d_gs_final_layer_hook_fn g_gs_final_layer_hook = NULL;
static void *g_gs_final_layer_hook_user = NULL;
static sam3d_gs_window_attn_hook_fn g_gs_window_attn_hook = NULL;
static void *g_gs_window_attn_hook_user = NULL;
static sam3d_gs_attn_block_hook_fn g_gs_attn_block_hook = NULL;
static void *g_gs_attn_block_hook_user = NULL;
static sam3d_gs_mlp_hook_fn g_gs_mlp_hook = NULL;
static void *g_gs_mlp_hook_user = NULL;
static sam3d_gs_block_hook_fn g_gs_block_hook = NULL;
static void *g_gs_block_hook_user = NULL;
static sam3d_gs_stack_hook_fn g_gs_stack_hook = NULL;
static void *g_gs_stack_hook_user = NULL;
static sam3d_gs_transformer_hook_fn g_gs_transformer_hook = NULL;
static void *g_gs_transformer_hook_user = NULL;

void sam3d_gs_decoder_set_input_ape_hook(sam3d_gs_input_ape_hook_fn fn,
                                         void *user)
{
    g_gs_input_ape_hook = fn;
    g_gs_input_ape_hook_user = user;
}

void sam3d_gs_decoder_set_final_layer_hook(sam3d_gs_final_layer_hook_fn fn,
                                           void *user)
{
    g_gs_final_layer_hook = fn;
    g_gs_final_layer_hook_user = user;
}

void sam3d_gs_decoder_set_window_attn_hook(sam3d_gs_window_attn_hook_fn fn,
                                           void *user)
{
    g_gs_window_attn_hook = fn;
    g_gs_window_attn_hook_user = user;
}

void sam3d_gs_decoder_set_attn_block_hook(sam3d_gs_attn_block_hook_fn fn,
                                          void *user)
{
    g_gs_attn_block_hook = fn;
    g_gs_attn_block_hook_user = user;
}

void sam3d_gs_decoder_set_mlp_hook(sam3d_gs_mlp_hook_fn fn,
                                   void *user)
{
    g_gs_mlp_hook = fn;
    g_gs_mlp_hook_user = user;
}

void sam3d_gs_decoder_set_block_hook(sam3d_gs_block_hook_fn fn,
                                     void *user)
{
    g_gs_block_hook = fn;
    g_gs_block_hook_user = user;
}

void sam3d_gs_decoder_set_stack_hook(sam3d_gs_stack_hook_fn fn,
                                     void *user)
{
    g_gs_stack_hook = fn;
    g_gs_stack_hook_user = user;
}

void sam3d_gs_decoder_set_transformer_hook(sam3d_gs_transformer_hook_fn fn,
                                           void *user)
{
    g_gs_transformer_hook = fn;
    g_gs_transformer_hook_user = user;
}

static int sam3d_gs_load_block(st_context *ctx, int idx, sam3d_gs_block *b) {
    char buf[256];
    int rc = 0;
#define FIND(field, fmt) do {                                    \
    snprintf(buf, sizeof(buf), "blocks.%d." fmt, idx);           \
    rc |= qt_find(ctx, buf, &b->field);                     \
} while (0)
    FIND(attn_qkv_w, "attn.to_qkv.weight");
    FIND(attn_qkv_b, "attn.to_qkv.bias");
    FIND(attn_out_w, "attn.to_out.weight");
    FIND(attn_out_b, "attn.to_out.bias");
    FIND(mlp_fc1_w,  "mlp.mlp.0.weight");
    FIND(mlp_fc1_b,  "mlp.mlp.0.bias");
    FIND(mlp_fc2_w,  "mlp.mlp.2.weight");
    FIND(mlp_fc2_b,  "mlp.mlp.2.bias");
#undef FIND
    return rc;
}

sam3d_gs_decoder_model *sam3d_gs_decoder_load_safetensors(const char *path) {
    st_context *ctx = safetensors_open(path);
    if (!ctx) {
        fprintf(stderr, "sam3d_gs_decoder: cannot open %s\n", path);
        return NULL;
    }
    sam3d_gs_decoder_model *m =
        (sam3d_gs_decoder_model *)calloc(1, sizeof(*m));
    if (!m) { safetensors_close(ctx); return NULL; }
    m->st_ctx = ctx;

    m->dim = 768;
    m->n_heads = 12;
    m->head_dim = 64;
    m->in_channels = 8;
    m->resolution = 64;
    m->window_size = 8;
    m->num_gaussians = 32;
    m->mlp_ratio = 4.0f;
    m->ln_eps = 1e-5f;

    m->voxel_size = 1.5f;
    m->scaling_bias = 0.004f;
    m->opacity_bias = 0.1f;
    m->perturb_offset = 1;
    m->lr_xyz = 1.0f;
    m->lr_features_dc = 1.0f;
    m->lr_scaling = 1.0f;
    m->lr_rotation = 0.1f;
    m->lr_opacity = 1.0f;

    /* Layout ranges: _xyz | _features_dc | _scaling | _rotation | _opacity. */
    int G = m->num_gaussians;
    int start = 0;
    m->r_xyz[0]          = start; m->r_xyz[1]          = start + G * 3; start = m->r_xyz[1];
    m->r_features_dc[0]  = start; m->r_features_dc[1]  = start + G * 3; start = m->r_features_dc[1];
    m->r_scaling[0]      = start; m->r_scaling[1]      = start + G * 3; start = m->r_scaling[1];
    m->r_rotation[0]     = start; m->r_rotation[1]     = start + G * 4; start = m->r_rotation[1];
    m->r_opacity[0]      = start; m->r_opacity[1]      = start + G;     start = m->r_opacity[1];
    m->out_channels = start;  /* 448 for G=32 */

    int rc = 0;
    rc |= qt_find(ctx, "input_layer.weight", &m->input_w);
    rc |= qt_find(ctx, "input_layer.bias",   &m->input_b);
    rc |= qt_find(ctx, "out_layer.weight",   &m->out_w);
    rc |= qt_find(ctx, "out_layer.bias",     &m->out_b);
    /* offset_perturbation is a registered buffer; optional — regenerate from
     * Hammersley at runtime if absent. */
    qt_find_opt(ctx, "offset_perturbation", &m->offset_perturbation);
    if (rc) {
        fprintf(stderr, "sam3d_gs_decoder: missing top-level weights\n");
        sam3d_gs_decoder_free(m); return NULL;
    }
    if (m->out_w.n_rows != m->out_channels) {
        fprintf(stderr, "sam3d_gs_decoder: out_w rows=%d != layout total %d\n",
                m->out_w.n_rows, m->out_channels);
        sam3d_gs_decoder_free(m); return NULL;
    }

    int n_blocks = 0;
    for (int i = 0; i < 64; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "blocks.%d.attn.to_qkv.weight", i);
        if (safetensors_find(ctx, buf) < 0) break;
        n_blocks = i + 1;
    }
    if (n_blocks == 0) {
        fprintf(stderr, "sam3d_gs_decoder: no blocks found\n");
        sam3d_gs_decoder_free(m); return NULL;
    }
    m->n_blocks = n_blocks;
    m->blocks = (sam3d_gs_block *)calloc(n_blocks, sizeof(sam3d_gs_block));
    if (!m->blocks) { sam3d_gs_decoder_free(m); return NULL; }
    for (int i = 0; i < n_blocks; i++) {
        if (sam3d_gs_load_block(ctx, i, &m->blocks[i]) != 0) {
            fprintf(stderr, "sam3d_gs_decoder: block %d load failed\n", i);
            sam3d_gs_decoder_free(m); return NULL;
        }
    }

    fprintf(stderr,
            "sam3d_gs_decoder: loaded %d blocks, dim=%d heads=%d head_dim=%d "
            "out_channels=%d num_gaussians=%d window=%d\n",
            m->n_blocks, m->dim, m->n_heads, m->head_dim,
            m->out_channels, m->num_gaussians, m->window_size);
    if (m->offset_perturbation.data) {
        fprintf(stderr,
                "  offset_perturbation: (%llu, %llu)\n",
                (unsigned long long)m->offset_perturbation.dims[0],
                (unsigned long long)m->offset_perturbation.dims[1]);
    } else {
        fprintf(stderr,
                "  offset_perturbation: ABSENT (will regenerate from Hammersley)\n");
    }
    return m;
}

void sam3d_gs_decoder_free(sam3d_gs_decoder_model *m) {
    if (!m) return;
    free(m->blocks);
    if (m->st_ctx) safetensors_close(m->st_ctx);
    free(m);
}

/* ====== Absolute Position Embedder (sin/cos per axis) ==================== */
/* Computes per-voxel 768-d positional embedding from (z, y, x) coords.
 * Layout per voxel: [sin(z*f[0..127]), cos(z*f[0..127]),
 *                    sin(y*f[0..127]), cos(y*f[0..127]),
 *                    sin(x*f[0..127]), cos(x*f[0..127])].
 * Pytorch ref: AbsolutePositionEmbedder(channels=768, in_channels=3). */
static void sam3d_gs_ape_add(float *feats, const int32_t *coords,
                              int N, int channels, int in_channels) {
    int freq_dim = channels / in_channels / 2;
    int per_axis = freq_dim * 2;
    float *freqs = (float *)malloc((size_t)freq_dim * sizeof(float));
    for (int j = 0; j < freq_dim; j++) {
        float e = (float)j / (float)freq_dim;
        freqs[j] = 1.0f / powf(10000.0f, e);
    }
    for (int i = 0; i < N; i++) {
        float *row = feats + (size_t)i * channels;
        for (int axis = 0; axis < in_channels; axis++) {
            float c = (float)coords[i * 4 + 1 + axis];  /* skip batch */
            float *base = row + axis * per_axis;
            for (int j = 0; j < freq_dim; j++) {
                float theta = c * freqs[j];
                base[j]            += sinf(theta);
                base[j + freq_dim] += cosf(theta);
            }
        }
        /* channels == in_channels * per_axis here (768 = 3*256), no pad. */
    }
    free(freqs);
}

/* Key+index pair sorted when building window partitions. */
typedef struct { int64_t k; int v; } sam3d_gs_kv;

static int sam3d_gs_kv_cmp(const void *a, const void *b) {
    int64_t ka = ((const sam3d_gs_kv *)a)->k;
    int64_t kb = ((const sam3d_gs_kv *)b)->k;
    return (ka < kb) ? -1 : (ka > kb);
}

/* ====== Windowed partition ============================================== */
/* Computes per-voxel window id = sum over 3 axes of
 *   ((coord[axis] + shift[axis]) / window_size) * offset[axis]
 * where offset[axis] is the cumulative product of num_windows along
 * later axes (matches pytorch calc_window_partition).
 * Returns malloc'd fwd_indices [N] that group voxels by (batch, wid).
 * Also outputs seq_lens[] contiguous counts per unique (batch,wid),
 * and batch_starts[] — but we only need fwd_indices and seq_lens here. */
typedef struct {
    int  seq_len;
    int  start;
} sam3d_gs_win_run;

static void sam3d_gs_partition(const sp3d_tensor *x,
                                int window_size, const int shift[3],
                                int **fwd_out,
                                sam3d_gs_win_run **runs_out,
                                int *n_runs_out) {
    int N = x->N;
    /* Compute max coords per axis after shift (per-batch collapsed). */
    int32_t mx[3] = {0, 0, 0};
    for (int i = 0; i < N; i++) {
        for (int a = 0; a < 3; a++) {
            int32_t v = x->coords[i * 4 + 1 + a] + shift[a];
            if (v > mx[a]) mx[a] = v;
        }
    }
    int nw[3];
    for (int a = 0; a < 3; a++) nw[a] = (int)(mx[a] / window_size) + 1;

    /* offset[axis] = product of nw[a] for a > axis — matches
     * cumprod([1] + NUM_WINDOWS[::-1])[::-1]. */
    int64_t off[3];
    off[2] = 1;
    off[1] = (int64_t)nw[2];
    off[0] = (int64_t)nw[1] * nw[2];

    /* Global offset per batch so windows of different batches don't mix. */
    int64_t per_batch = (int64_t)nw[0] * nw[1] * nw[2];

    /* Compute wid per voxel. */
    int64_t *wid = (int64_t *)malloc((size_t)N * sizeof(int64_t));
    int    *fwd  = (int    *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) {
        int32_t b = x->coords[i * 4];
        int64_t s = (int64_t)b * per_batch;
        for (int a = 0; a < 3; a++) {
            int32_t v = (x->coords[i * 4 + 1 + a] + shift[a]) / window_size;
            s += (int64_t)v * off[a];
        }
        wid[i] = s;
        fwd[i] = i;
    }

    sam3d_gs_kv *kv = (sam3d_gs_kv *)malloc((size_t)N * sizeof(sam3d_gs_kv));
    for (int i = 0; i < N; i++) { kv[i].k = wid[i]; kv[i].v = i; }
    qsort(kv, (size_t)N, sizeof(sam3d_gs_kv), sam3d_gs_kv_cmp);
    for (int i = 0; i < N; i++) fwd[i] = kv[i].v;

    /* Build runs: groups of contiguous entries with equal wid. */
    int max_runs = N > 0 ? N : 1;
    sam3d_gs_win_run *runs = (sam3d_gs_win_run *)malloc(
            (size_t)max_runs * sizeof(sam3d_gs_win_run));
    int nr = 0;
    int i = 0;
    while (i < N) {
        int j = i + 1;
        while (j < N && kv[j].k == kv[i].k) j++;
        runs[nr].start = i;
        runs[nr].seq_len = j - i;
        nr++;
        i = j;
    }

    free(wid);
    free(kv);

    *fwd_out = fwd;
    *runs_out = runs;
    *n_runs_out = nr;
}

/* ====== Windowed sparse self-attention ================================== */
/* qkv: [N, 3*dim] interleaved [Q, K, V] per token.
 * out: [N, dim]. Partition voxels into windows via sam3d_gs_partition,
 * gather QKV within each window into a contiguous buffer, run cpu_attention,
 * scatter back. */
static void sam3d_gs_windowed_attention(float *out, const float *qkv,
                                         const sp3d_tensor *x,
                                         int window_size, const int shift[3],
                                         int n_heads, int head_dim,
                                         int n_threads) {
    if (g_gs_window_attn_hook &&
        g_gs_window_attn_hook(g_gs_window_attn_hook_user, out, qkv, x,
                              window_size, shift, n_heads, head_dim) == 0)
        return;

    int dim = n_heads * head_dim;
    int qkv_stride = 3 * dim;

    int *fwd = NULL;
    sam3d_gs_win_run *runs = NULL;
    int nr = 0;
    sam3d_gs_partition(x, window_size, shift, &fwd, &runs, &nr);

    /* Find max window size to size the per-window buffer. */
    int max_len = 0;
    for (int r = 0; r < nr; r++)
        if (runs[r].seq_len > max_len) max_len = runs[r].seq_len;

    float *qkv_win = (float *)malloc((size_t)max_len * qkv_stride * sizeof(float));
    float *out_win = (float *)malloc((size_t)max_len * dim * sizeof(float));

    for (int r = 0; r < nr; r++) {
        int len = runs[r].seq_len;
        int start = runs[r].start;
        /* Gather QKV into contiguous window buffer. */
        for (int t = 0; t < len; t++) {
            int src = fwd[start + t];
            memcpy(qkv_win + (size_t)t * qkv_stride,
                   qkv + (size_t)src * qkv_stride,
                   (size_t)qkv_stride * sizeof(float));
        }
        /* Run attention on this window. */
        cpu_attention(out_win, qkv_win, len, dim, n_heads, head_dim, n_threads);
        /* Scatter back. */
        for (int t = 0; t < len; t++) {
            int dst = fwd[start + t];
            memcpy(out + (size_t)dst * dim,
                   out_win + (size_t)t * dim,
                   (size_t)dim * sizeof(float));
        }
    }

    free(qkv_win);
    free(out_win);
    free(fwd);
    free(runs);
}

/* ====== Transformer forward ============================================= */

int sam3d_gs_decoder_hidden_transformer(const sam3d_gs_decoder_model *m,
                                        const sp3d_tensor *x,
                                        float **out_feats,
                                        int n_threads) {
    if (!m || !x || !out_feats) return -1;
    int N = x->N;
    int D = m->dim;

    float *h = (float *)malloc((size_t)N * D * sizeof(float));
    if (!h) return -2;

    /* 1. input_layer: 8 -> 768 */
    if (g_gs_input_ape_hook) {
        free(h);
        h = NULL;
        if (g_gs_input_ape_hook(g_gs_input_ape_hook_user, x->coords,
                                x->feats, N, m->in_channels,
                                &m->input_w, &m->input_b, D, &h) != 0 ||
            !h)
            return -2;
    } else {
        sp3d_linear(h, x->feats, N, &m->input_w, &m->input_b,
                    D, m->in_channels, n_threads);

        /* 2. h += pos_embedder(coords[:,1:]) */
        sam3d_gs_ape_add(h, x->coords, N, D, 3);
    }

    /* Scratch buffers reused across blocks. */
    float *h_norm = (float *)malloc((size_t)N * D * sizeof(float));
    float *qkv    = (float *)malloc((size_t)N * 3 * D * sizeof(float));
    float *attn   = (float *)malloc((size_t)N * D * sizeof(float));
    int mlp_hidden = (int)(D * m->mlp_ratio + 0.5f);
    float *mlp_h  = (float *)malloc((size_t)N * mlp_hidden * sizeof(float));
    if (!h_norm || !qkv || !attn || !mlp_h) {
        free(h); free(h_norm); free(qkv); free(attn); free(mlp_h);
        return -3;
    }

    if (g_gs_stack_hook) {
        if (g_gs_stack_hook(g_gs_stack_hook_user, h, x, N, D,
                            m->blocks, m->n_blocks, m->window_size,
                            m->n_heads, m->head_dim, mlp_hidden,
                            1e-6f) != 0) {
            free(h); free(h_norm); free(qkv); free(attn); free(mlp_h);
            return -8;
        }
    } else {
        for (int b = 0; b < m->n_blocks; b++) {
            const sam3d_gs_block *blk = &m->blocks[b];
            int shift_v = (b & 1) ? (m->window_size / 2) : 0;
            int shift[3] = {shift_v, shift_v, shift_v};

        if (g_gs_block_hook) {
            if (g_gs_block_hook(g_gs_block_hook_user, h, x,
                                N, D, blk, m->window_size, shift,
                                m->n_heads, m->head_dim, mlp_hidden,
                                1e-6f) != 0) {
                free(h); free(h_norm); free(qkv); free(attn); free(mlp_h);
                return -7;
            }
            continue;
        }

        /* --- Self-attn subblock --- */
        if (g_gs_attn_block_hook) {
            if (g_gs_attn_block_hook(g_gs_attn_block_hook_user, h, x,
                                     N, D, blk, m->window_size, shift,
                                     m->n_heads, m->head_dim, 1e-6f) != 0) {
                free(h); free(h_norm); free(qkv); free(attn); free(mlp_h);
                return -6;
            }
        } else {
            /* norm1 (no affine, eps=1e-6) */
            sp3d_layernorm(h_norm, h, NULL, NULL, N, D, 1e-6f);
            /* attn.to_qkv: dim -> 3*dim */
            sp3d_linear(qkv, h_norm, N, &blk->attn_qkv_w, &blk->attn_qkv_b,
                        3 * D, D, n_threads);
            /* Windowed sparse self-attention */
            sam3d_gs_windowed_attention(attn, qkv, x,
                                         m->window_size, shift,
                                         m->n_heads, m->head_dim, n_threads);
            /* attn.to_out: dim -> dim, into h_norm for re-use, then residual */
            sp3d_linear(h_norm, attn, N, &blk->attn_out_w, &blk->attn_out_b,
                        D, D, n_threads);
            for (int i = 0; i < N * D; i++) h[i] += h_norm[i];
        }

        /* --- MLP subblock --- */
        if (g_gs_mlp_hook) {
            if (g_gs_mlp_hook(g_gs_mlp_hook_user, h, N, D, blk,
                              mlp_hidden, 1e-6f) != 0) {
                free(h); free(h_norm); free(qkv); free(attn); free(mlp_h);
                return -5;
            }
        } else {
            sp3d_layernorm(h_norm, h, NULL, NULL, N, D, 1e-6f);
            sp3d_linear(mlp_h, h_norm, N, &blk->mlp_fc1_w, &blk->mlp_fc1_b,
                        mlp_hidden, D, n_threads);
            sp3d_gelu(mlp_h, N * mlp_hidden);
            sp3d_linear(h_norm, mlp_h, N, &blk->mlp_fc2_w, &blk->mlp_fc2_b,
                        D, mlp_hidden, n_threads);
            for (int i = 0; i < N * D; i++) h[i] += h_norm[i];
        }
    }
    }

    free(h_norm); free(qkv); free(attn); free(mlp_h);
    *out_feats = h;
    return 0;
}

int sam3d_gs_decoder_transformer(const sam3d_gs_decoder_model *m,
                                  const sp3d_tensor *x,
                                  float **out_feats,
                                  int n_threads) {
    if (!m || !x || !out_feats) return -1;
    int N = x->N;
    int D = m->dim;
    int G = m->out_channels;

    if (g_gs_transformer_hook) {
        float *hook_out = NULL;
        if (g_gs_transformer_hook(g_gs_transformer_hook_user, x, m,
                                  &hook_out) != 0 ||
            !hook_out)
            return -9;
        *out_feats = hook_out;
        return 0;
    }

    float *h = NULL;
    int rc = sam3d_gs_decoder_hidden_transformer(m, x, &h, n_threads);
    if (rc != 0 || !h) return rc ? rc : -2;

    float *feats_out = NULL;
    if (g_gs_final_layer_hook) {
        if (g_gs_final_layer_hook(g_gs_final_layer_hook_user, h, N, D,
                                  &m->out_w, &m->out_b, G, 1e-5f,
                                  &feats_out) != 0 ||
            !feats_out) {
            free(h);
            return -4;
        }
    } else {
        float *h_final = (float *)malloc((size_t)N * D * sizeof(float));
        if (!h_final) { free(h); return -4; }
        sp3d_layernorm(h_final, h, NULL, NULL, N, D, 1e-5f);

        feats_out = (float *)malloc((size_t)N * G * sizeof(float));
        if (!feats_out) { free(h); free(h_final); return -4; }
        sp3d_linear(feats_out, h_final, N, &m->out_w, &m->out_b,
                    G, D, n_threads);
        free(h_final);
    }

    free(h);
    *out_feats = feats_out;
    return 0;
}

/* ====== to_representation ============================================== */

int sam3d_gs_decoder_to_representation(const sam3d_gs_decoder_model *m,
                                        const int32_t *coords,
                                        const float   *feats_448,
                                        int N,
                                        float *xyz_out,
                                        float *dc_out,
                                        float *scaling_out,
                                        float *rotation_out,
                                        float *opacity_out) {
    if (!m || !coords || !feats_448) return -1;
    int G = m->num_gaussians;
    int C = m->out_channels;
    float inv_res = 1.0f / (float)m->resolution;

    /* Cache offset_perturbation if present. */
    const float *perturb = NULL;
    if (m->offset_perturbation.data && m->perturb_offset) {
        perturb = (const float *)m->offset_perturbation.data;
    }

    for (int v = 0; v < N; v++) {
        float zyx[3];
        for (int a = 0; a < 3; a++)
            zyx[a] = ((float)coords[v * 4 + 1 + a] + 0.5f) * inv_res;
        const float *fp = feats_448 + (size_t)v * C;

        /* _xyz: per-voxel offset + perturbation → tanh → add center */
        if (xyz_out) {
            const float *off_base = fp + m->r_xyz[0];
            for (int g = 0; g < G; g++) {
                float off[3];
                for (int a = 0; a < 3; a++) {
                    float o = off_base[g * 3 + a] * m->lr_xyz;
                    if (perturb) o += perturb[g * 3 + a];
                    o = tanhf(o) * inv_res * 0.5f * m->voxel_size;
                    off[a] = o;
                }
                float *dst = xyz_out + ((size_t)v * G + g) * 3;
                dst[0] = zyx[0] + off[0];
                dst[1] = zyx[1] + off[1];
                dst[2] = zyx[2] + off[2];
            }
        }
        /* _features_dc: [G, 1, 3] flattened to [G, 3] storage. */
        if (dc_out) {
            const float *src = fp + m->r_features_dc[0];
            float *dst = dc_out + (size_t)v * G * 3;
            for (int k = 0; k < G * 3; k++) dst[k] = src[k] * m->lr_features_dc;
        }
        if (scaling_out) {
            const float *src = fp + m->r_scaling[0];
            float *dst = scaling_out + (size_t)v * G * 3;
            for (int k = 0; k < G * 3; k++) dst[k] = src[k] * m->lr_scaling;
        }
        if (rotation_out) {
            const float *src = fp + m->r_rotation[0];
            float *dst = rotation_out + (size_t)v * G * 4;
            for (int k = 0; k < G * 4; k++) dst[k] = src[k] * m->lr_rotation;
        }
        if (opacity_out) {
            const float *src = fp + m->r_opacity[0];
            float *dst = opacity_out + (size_t)v * G;
            for (int k = 0; k < G; k++) dst[k] = src[k] * m->lr_opacity;
        }
    }
    return 0;
}

#endif /* SAM3D_GS_DECODER_IMPL_ONCE */
#endif /* SAM3D_GS_DECODER_IMPLEMENTATION */
