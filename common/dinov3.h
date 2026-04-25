/*
 * dinov3.h - DINOv3 ViT-L/16 vision encoder (TRELLIS.2 conditioning)
 *
 * Usage:
 *   #define DINOV3_IMPLEMENTATION
 *   #include "dinov3.h"
 *
 * Dependencies: ggml_dequant.h, safetensors.h
 *
 * API:
 *   dinov3_model  *dinov3_load_safetensors(const char *st_path);
 *   void           dinov3_free(dinov3_model *m);
 *   dinov3_result  dinov3_encode(dinov3_model *m, const uint8_t *rgb,
 *                                int w, int h, int n_threads);
 *   void           dinov3_result_free(dinov3_result *r);
 *
 * DINOv3 ViT-L/16 vs DINOv2-L differences:
 *   - Patch size: 16 (not 14)
 *   - Image size: 512 (not 518)
 *   - No learned positional embedding: RoPE only (all layers)
 *   - 4 storage/register tokens after CLS: [CLS, stor1..4, patch1..1024]
 *   - RoPE NOT applied to CLS or storage tokens (only patches)
 *   - Standard GELU MLP (not SwiGLU)
 *   - Output: all 1029 tokens after final LayerNorm
 */
#ifndef DINOV3_H
#define DINOV3_H

#include <stdint.h>
#include <stddef.h>
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "qtensor_utils.h"

typedef struct {
    qtensor ln1_w, ln1_b;
    qtensor attn_qkv_w, attn_qkv_b;
    qtensor attn_q_norm_w, attn_q_norm_b;  /* optional: per-head QK norm */
    qtensor attn_k_norm_w, attn_k_norm_b;
    qtensor attn_out_w, attn_out_b;
    qtensor ls1;                            /* optional: LayerScale */
    qtensor ln2_w, ln2_b;
    qtensor ffn_up_w, ffn_up_b;            /* GELU MLP: fc1 [ffn_dim, dim] */
    qtensor ffn_down_w, ffn_down_b;        /* fc2 [dim, ffn_dim] */
    /* SwiGLU MLP (DINOv3-H+ / sam-3d-body):
     *   y = w3( silu(w1(x)+b1) * (w2(x)+b2) ) + b3
     * w1, w2: [ffn_hidden, dim]; w3: [dim, ffn_hidden]. Populated when
     * the checkpoint ships `mlp.w1/w2/w3` instead of `mlp.fc1/fc2`. */
    qtensor ffn_w1_w, ffn_w1_b;
    qtensor ffn_w2_w, ffn_w2_b;
    qtensor ffn_w3_w, ffn_w3_b;
    qtensor ls2;                            /* optional: LayerScale */
} dinov3_block;

typedef struct {
    int n_blocks, dim, n_heads, head_dim, ffn_hidden;
    int patch_size, image_size;
    int grid_h, grid_w, n_patches;
    int n_storage;   /* 4 storage/register tokens */
    int n_tokens;    /* 1 CLS + n_storage + n_patches */
    float ln_eps;
    float image_mean[3], image_std[3];
    float rope_freq_base;
    int has_qk_norm;    /* auto-detected from weights */
    int has_layerscale; /* auto-detected from weights */
    int has_swiglu;     /* auto-detected: 1 if mlp.w1/w2/w3 present, 0 for fc1/fc2 */
    int has_rope_periods_tensor; /* auto-detected: 1 if rope_embed.periods was loaded */

    /* When set (default 0), dinov3_encode applies the learned final
     * LayerNorm (norm_w/norm_b). Kept opt-in so existing TRELLIS.2
     * call sites that use the historical unparameterized-LN feature
     * extraction keep their behavior unchanged. sam-3d-body sets
     * this to 1. */
    int use_learned_final_norm;

    qtensor patch_embed_w, patch_embed_b;
    qtensor cls_token;
    qtensor storage_tokens;  /* [n_storage, dim] */
    qtensor norm_w, norm_b;  /* final LayerNorm */
    qtensor rope_periods;    /* optional: saved [rope_dim4] fp/bf16 periods */

    dinov3_block *blocks;

    /* SafeTensors context (kept alive for mmap'd weights) */
    void *st_ctx;
} dinov3_model;

typedef struct {
    float *features;  /* [n_tokens, dim] = [1029, 1024] for ViT-L/16 */
    int n_tokens;
    int dim;
} dinov3_result;

dinov3_model  *dinov3_load_safetensors(const char *st_path);
void           dinov3_free(dinov3_model *m);
dinov3_result  dinov3_encode(dinov3_model *m, const uint8_t *rgb,
                             int w, int h, int n_threads);
/* Bypass the built-in resize+normalize preprocessing. `chw` is a
 * [3, H, W] f32 tensor already normalized with the model's mean/std.
 * H, W must equal `m->image_size` for the default square layout.
 * Used by verify_*.c to anchor on pytorch-produced normalized inputs. */
dinov3_result  dinov3_encode_from_normalized(dinov3_model *m, const float *chw,
                                             int w, int h, int n_threads);
void           dinov3_result_free(dinov3_result *r);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef DINOV3_IMPLEMENTATION

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

static double dinov3_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Tensor helpers ---- */

/* ---- Batch GEMM ---- */

static void dinov3_batch_gemm(float *dst, const qtensor *W, const qtensor *bias,
                               const float *src, int n_tok, int n_out, int n_in,
                               int n_threads) {
    if (!W->data) {
        memset(dst, 0, (size_t)n_tok * n_out * sizeof(float));
        return;
    }
    float *bf_alloc = NULL;
    const float *bf = NULL;
    if (bias && bias->data) {
        if (bias->type == GGML_TYPE_F32) {
            bf = (const float *)bias->data;
        } else {
            bf_alloc = (float *)malloc((size_t)n_out * sizeof(float));
            qt_dequant_row(bias, 0, bf_alloc);
            bf = bf_alloc;
        }
    }
    if (W->type == GGML_TYPE_F16) {
        cpu_gemm_f16(dst, (const uint16_t *)W->data, (float *)bf, src,
                     n_tok, n_out, n_in, n_threads);
    } else if (W->type == GGML_TYPE_F32) {
        cpu_gemm_f32(dst, (const float *)W->data, bf, src,
                     n_tok, n_out, n_in, n_threads);
    } else {
        /* Quantized: dequant whole W once, then F32 GEMM. */
        float *Wf = qt_dequant(W);
        cpu_gemm_f32(dst, Wf, bf, src, n_tok, n_out, n_in, n_threads);
        free(Wf);
    }
    free(bf_alloc);
}

/* ---- LayerNorm ---- */

static void dinov3_layernorm_batch(float *dst, const float *src, const qtensor *w,
                                    const qtensor *b, int n_tok, int dim, float eps) {
    float *wf = (float *)malloc((size_t)dim * sizeof(float));
    float *bf = (float *)malloc((size_t)dim * sizeof(float));
    qt_dequant_row(w, 0, wf);
    qt_dequant_row(b, 0, bf);
    cpu_layernorm(dst, src, wf, bf, n_tok, dim, eps);
    free(wf); free(bf);
}

/* ---- QK Normalization ---- */

static void dinov3_qk_norm_batch(float *vec, int n_tok, int n_heads, int head_dim,
                                   const qtensor *nw, const qtensor *nb, float eps) {
    if (!nw->data) return;
    float *w = (float *)malloc((size_t)head_dim * sizeof(float));
    float *b = (float *)malloc((size_t)head_dim * sizeof(float));
    qt_dequant_row(nw, 0, w);
    qt_dequant_row(nb, 0, b);
    cpu_qk_norm(vec, n_tok, n_heads, head_dim, n_heads * head_dim, w, b, eps);
    free(w); free(b);
}

/* ---- LayerScale ---- */

static void dinov3_layerscale(float *x, const qtensor *gamma, int n_tok, int dim) {
    if (!gamma->data) return;
    float *g = (float *)malloc((size_t)dim * sizeof(float));
    qt_dequant_row(gamma, 0, g);
    for (int t = 0; t < n_tok; t++) {
        float *v = x + t * dim;
        for (int i = 0; i < dim; i++) v[i] *= g[i];
    }
    free(g);
}

/* ==================================================================== */
/* SafeTensors loading                                                   */
/* ==================================================================== */

#ifdef SAFETENSORS_H

dinov3_model *dinov3_load_safetensors(const char *st_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) return NULL;

    fprintf(stderr, "dinov3: safetensors opened, %d tensors\n", st->n_tensors);

    /* Print first few tensor names for debugging */
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

    /* Detect common prefix by scanning tensor names */
    char prefix[256] = "";
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "patch_embed")) {
            /* Extract prefix up to "patch_embed" */
            const char *p = strstr(nm, "patch_embed");
            size_t pl = (size_t)(p - nm);
            if (pl < sizeof(prefix)) {
                memcpy(prefix, nm, pl);
                prefix[pl] = '\0';
            }
            break;
        }
    }
    fprintf(stderr, "dinov3: detected prefix: '%s'\n", prefix);

    /* Helper to find tensor with detected prefix */
    /* We'll use a local lambda-like approach */
    #define DINOV3_FIND(name_suffix) ({ \
        char _buf[512]; \
        snprintf(_buf, sizeof(_buf), "%s%s", prefix, name_suffix); \
        int _idx = safetensors_find(st, _buf); \
        (_idx >= 0) ? qt_make_tensor(st, _idx) : (qtensor){0}; \
    })

    /* Auto-detect parameters */
    int embed_dim = 1024, n_heads = 16, head_dim = 64;
    int n_blocks = 0, ffn_hidden = 4096;
    int patch_size = 16, image_size = 512;
    int n_storage = 0;
    int has_qk_norm = 0, has_layerscale = 0;

    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const uint64_t *sh = safetensors_shape(st, i);

        if (strstr(nm, "patch_embed") && (strstr(nm, "proj.weight") || strstr(nm, "weight"))) {
            int nd = safetensors_ndims(st, i);
            if (nd == 4) {
                embed_dim = (int)sh[0];
                patch_size = (int)sh[2];
                fprintf(stderr, "dinov3: patch_embed: embed_dim=%d, patch_size=%d\n",
                        embed_dim, patch_size);
            }
        }
        /* Detect storage/register tokens (timm: "reg_token", official: "register_tokens") */
        if (strstr(nm, "reg_token") || strstr(nm, "register_tokens") || strstr(nm, "storage_tokens")) {
            int nd = safetensors_ndims(st, i);
            if (nd == 3) n_storage = (int)sh[1];
            else if (nd == 2) n_storage = (int)sh[0];
            fprintf(stderr, "dinov3: detected %d storage/register tokens\n", n_storage);
        }
        /* Detect head_dim from QK norm */
        if (strstr(nm, "q_norm.weight") || strstr(nm, "qk_norm")) {
            head_dim = (int)sh[0];
            has_qk_norm = 1;
        }
        /* Detect LayerScale (timm: "gamma_1/gamma_2", official: "ls1.gamma") */
        if (strstr(nm, "ls1") || strstr(nm, "gamma_1") || strstr(nm, "gamma1")) {
            has_layerscale = 1;
        }
        /* Detect FFN hidden from first block */
        if (strstr(nm, "blocks.0.mlp.fc1.weight") || strstr(nm, "blocks.0.ffn.fc1.weight")) {
            ffn_hidden = (int)sh[0];
        }
        /* Detect SwiGLU MLP (mlp.w1 alongside mlp.w3). The presence of
         * w1 is enough — sam-3d-body ckpt: [5120, 1280]. */
        if (strstr(nm, "blocks.0.mlp.w1.weight")) {
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

    n_heads = embed_dim / head_dim;
    int grid_h = image_size / patch_size;
    int grid_w = grid_h;
    int n_patches = grid_h * grid_w;
    int n_tokens = 1 + n_storage + n_patches;

    fprintf(stderr, "dinov3: dim=%d, heads=%d, head_dim=%d, blocks=%d, ffn=%d\n",
            embed_dim, n_heads, head_dim, n_blocks, ffn_hidden);
    fprintf(stderr, "dinov3: patch=%d, image=%d, grid=%dx%d, patches=%d, storage=%d, tokens=%d\n",
            patch_size, image_size, grid_h, grid_w, n_patches, n_storage, n_tokens);
    fprintf(stderr, "dinov3: qk_norm=%d, layerscale=%d\n", has_qk_norm, has_layerscale);

    /* Allocate model */
    dinov3_model *m = (dinov3_model *)calloc(1, sizeof(dinov3_model));
    m->dim = embed_dim;
    m->n_heads = n_heads;
    m->head_dim = head_dim;
    m->n_blocks = n_blocks;
    m->ffn_hidden = ffn_hidden;
    m->patch_size = patch_size;
    m->image_size = image_size;
    m->grid_h = grid_h;
    m->grid_w = grid_w;
    m->n_patches = n_patches;
    m->n_storage = n_storage;
    m->n_tokens = n_tokens;
    m->ln_eps = 1e-6f;
    m->rope_freq_base = 100.0f;
    m->has_qk_norm = has_qk_norm;
    m->has_layerscale = has_layerscale;
    m->image_mean[0] = 0.485f; m->image_mean[1] = 0.456f; m->image_mean[2] = 0.406f;
    m->image_std[0]  = 0.229f; m->image_std[1]  = 0.224f; m->image_std[2]  = 0.225f;
    m->st_ctx = st;

    /* Load embeddings */
    m->cls_token     = DINOV3_FIND("cls_token");
    m->patch_embed_w = DINOV3_FIND("patch_embed.proj.weight");
    m->patch_embed_b = DINOV3_FIND("patch_embed.proj.bias");

    /* Try multiple names for storage/register tokens */
    m->storage_tokens = DINOV3_FIND("reg_token");
    if (!m->storage_tokens.data)
        m->storage_tokens = DINOV3_FIND("register_tokens");
    if (!m->storage_tokens.data)
        m->storage_tokens = DINOV3_FIND("storage_tokens");

    /* Final norm */
    m->norm_w = DINOV3_FIND("norm.weight");
    m->norm_b = DINOV3_FIND("norm.bias");

    /* If no patch_embed found, try alternate naming */
    if (!m->patch_embed_w.data) {
        m->patch_embed_w = DINOV3_FIND("patch_embed.weight");
        m->patch_embed_b = DINOV3_FIND("patch_embed.bias");
    }

    /* Log which embeddings were found */
    fprintf(stderr, "dinov3: cls_token: %s\n", m->cls_token.data ? "loaded" : "MISSING");
    fprintf(stderr, "dinov3: storage_tokens: %s\n", m->storage_tokens.data ? "loaded" : "none");
    fprintf(stderr, "dinov3: patch_embed_w: %s\n", m->patch_embed_w.data ? "loaded" : "MISSING");
    fprintf(stderr, "dinov3: norm: %s\n", m->norm_w.data ? "loaded" : "MISSING");

    /* Load blocks */
    m->blocks = (dinov3_block *)calloc((size_t)m->n_blocks, sizeof(dinov3_block));
    for (int L = 0; L < m->n_blocks; L++) {
        dinov3_block *blk = &m->blocks[L];
        char name[512];

        /* LayerNorm 1 (try both norm1 and attn.norm1) */
        snprintf(name, sizeof(name), "%sblocks.%d.norm1.weight", prefix, L);
        int idx = safetensors_find(st, name);
        if (idx < 0) {
            snprintf(name, sizeof(name), "%sblocks.%d.attn.norm1.weight", prefix, L);
            idx = safetensors_find(st, name);
        }
        if (idx >= 0) blk->ln1_w = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.norm1.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx < 0) {
            snprintf(name, sizeof(name), "%sblocks.%d.attn.norm1.bias", prefix, L);
            idx = safetensors_find(st, name);
        }
        if (idx >= 0) blk->ln1_b = qt_make_tensor(st, idx);

        /* Attention QKV */
        snprintf(name, sizeof(name), "%sblocks.%d.attn.qkv.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->attn_qkv_w = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.attn.qkv.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->attn_qkv_b = qt_make_tensor(st, idx);

        /* QK norm (optional) */
        snprintf(name, sizeof(name), "%sblocks.%d.attn.q_norm.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->attn_q_norm_w = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.attn.q_norm.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->attn_q_norm_b = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.attn.k_norm.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->attn_k_norm_w = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.attn.k_norm.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->attn_k_norm_b = qt_make_tensor(st, idx);

        /* Attention output projection */
        snprintf(name, sizeof(name), "%sblocks.%d.attn.proj.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->attn_out_w = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.attn.proj.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->attn_out_b = qt_make_tensor(st, idx);

        /* LayerScale 1 (optional: "gamma_1" in timm, "ls1.gamma" in official) */
        snprintf(name, sizeof(name), "%sblocks.%d.gamma_1", prefix, L);
        idx = safetensors_find(st, name);
        if (idx < 0) {
            snprintf(name, sizeof(name), "%sblocks.%d.ls1.gamma", prefix, L);
            idx = safetensors_find(st, name);
        }
        if (idx < 0) {
            snprintf(name, sizeof(name), "%sblocks.%d.attn.ls1", prefix, L);
            idx = safetensors_find(st, name);
        }
        if (idx >= 0) blk->ls1 = qt_make_tensor(st, idx);

        /* LayerNorm 2 */
        snprintf(name, sizeof(name), "%sblocks.%d.norm2.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ln2_w = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.norm2.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ln2_b = qt_make_tensor(st, idx);

        /* FFN (GELU MLP: fc1 + fc2) */
        snprintf(name, sizeof(name), "%sblocks.%d.mlp.fc1.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_up_w = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.mlp.fc1.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_up_b = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.mlp.fc2.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_down_w = qt_make_tensor(st, idx);

        snprintf(name, sizeof(name), "%sblocks.%d.mlp.fc2.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_down_b = qt_make_tensor(st, idx);

        /* SwiGLU FFN (DINOv3-H+: mlp.w1 / mlp.w2 / mlp.w3) */
        snprintf(name, sizeof(name), "%sblocks.%d.mlp.w1.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_w1_w = qt_make_tensor(st, idx);
        snprintf(name, sizeof(name), "%sblocks.%d.mlp.w1.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_w1_b = qt_make_tensor(st, idx);
        snprintf(name, sizeof(name), "%sblocks.%d.mlp.w2.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_w2_w = qt_make_tensor(st, idx);
        snprintf(name, sizeof(name), "%sblocks.%d.mlp.w2.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_w2_b = qt_make_tensor(st, idx);
        snprintf(name, sizeof(name), "%sblocks.%d.mlp.w3.weight", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_w3_w = qt_make_tensor(st, idx);
        snprintf(name, sizeof(name), "%sblocks.%d.mlp.w3.bias", prefix, L);
        idx = safetensors_find(st, name);
        if (idx >= 0) blk->ffn_w3_b = qt_make_tensor(st, idx);

        /* LayerScale 2 (optional: "gamma_2" in timm, "ls2.gamma" in official) */
        snprintf(name, sizeof(name), "%sblocks.%d.gamma_2", prefix, L);
        idx = safetensors_find(st, name);
        if (idx < 0) {
            snprintf(name, sizeof(name), "%sblocks.%d.ls2.gamma", prefix, L);
            idx = safetensors_find(st, name);
        }
        if (idx < 0) {
            snprintf(name, sizeof(name), "%sblocks.%d.ls2", prefix, L);
            idx = safetensors_find(st, name);
        }
        if (idx >= 0) blk->ls2 = qt_make_tensor(st, idx);

        /* Verify essential block tensors */
        if (L == 0) {
            if (!blk->ln1_w.data) fprintf(stderr, "dinov3: WARNING: block 0 ln1_w missing\n");
            if (!blk->attn_qkv_w.data) fprintf(stderr, "dinov3: WARNING: block 0 qkv_w missing\n");
            if (!blk->attn_out_w.data) fprintf(stderr, "dinov3: WARNING: block 0 attn_out_w missing\n");
            int has_gelu = blk->ffn_up_w.data != NULL;
            int has_swi  = blk->ffn_w1_w.data != NULL && blk->ffn_w3_w.data != NULL;
            if (!has_gelu && !has_swi)
                fprintf(stderr, "dinov3: WARNING: block 0 MLP missing (neither fc1/fc2 nor w1/w2/w3)\n");
            m->has_swiglu = has_swi;
        }
    }

    #undef DINOV3_FIND

    /* Saved RoPE periods (DINOv3-H+ ships these in the checkpoint). */
    int rp_idx = safetensors_find(st, "rope_embed.periods");
    if (rp_idx < 0) {
        char name[256];
        snprintf(name, sizeof(name), "%srope_embed.periods", prefix);
        rp_idx = safetensors_find(st, name);
    }
    if (rp_idx >= 0) {
        m->rope_periods = qt_make_tensor(st, rp_idx);
        m->has_rope_periods_tensor = 1;
        fprintf(stderr, "dinov3: loaded saved rope_embed.periods (%d elems)\n",
                qt_numel(&m->rope_periods));
    }

    fprintf(stderr, "dinov3: loaded %d blocks%s%s\n", m->n_blocks,
            m->has_swiglu ? ", swiglu MLP" : ", gelu MLP",
            m->has_rope_periods_tensor ? ", saved rope periods" : "");
    return m;
}

#endif /* SAFETENSORS_H */

/* ==================================================================== */
/* API: dinov3_free                                                      */
/* ==================================================================== */

void dinov3_free(dinov3_model *m) {
    if (!m) return;
    free(m->blocks);
#ifdef SAFETENSORS_H
    if (m->st_ctx) safetensors_close((st_context *)m->st_ctx);
#endif
    free(m);
}

/* ==================================================================== */
/* API: dinov3_encode                                                    */
/* ==================================================================== */

/* Forward pass after preprocessing. Takes ownership of `img_norm`
 * (layout: [3, target_h, target_w], row-major). */
static dinov3_result dinov3_encode_norm_(dinov3_model *m, float *img_norm,
                                          int n_threads) {
    dinov3_result result = {0};
    int ps = m->patch_size;
    int gh = m->grid_h, gw = m->grid_w;
    int np = m->n_patches;
    int ns = m->n_storage;
    int nt = m->n_tokens;
    int dim = m->dim;
    int target_h = gh * ps;
    int target_w = gw * ps;

    if (n_threads < 1) n_threads = 1;
    double t_start = dinov3_time_ms();

    /* ─── Step 2: Patch embedding ─── */
    /* Token layout: [CLS, stor1..stor_ns, patch1..patch_np] */
    int patch_start = 1 + ns;  /* index of first patch token */
    float *hidden = (float *)calloc((size_t)nt * dim, sizeof(float));
    {
        float *pw = qt_dequant(&m->patch_embed_w);
        float *pb = qt_dequant(&m->patch_embed_b);
        /* patch_embed_w: [Co=dim, Ci=3, kH=ps, kW=ps] in PyTorch row-major order */
        int Co = dim, Ci = 3;
        for (int py = 0; py < gh; py++) {
            for (int px = 0; px < gw; px++) {
                int tok = patch_start + py * gw + px;
                float *out = hidden + tok * dim;
                if (pb) memcpy(out, pb, (size_t)dim * sizeof(float));
                for (int co = 0; co < Co; co++) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < Ci; ci++) {
                        for (int kh = 0; kh < ps; kh++) {
                            for (int kw = 0; kw < ps; kw++) {
                                int ih = py * ps + kh;
                                int iw = px * ps + kw;
                                sum += pw[((co * Ci + ci) * ps + kh) * ps + kw]
                                     * img_norm[ci * target_h * target_w + ih * target_w + iw];
                            }
                        }
                    }
                    out[co] += sum;
                }
            }
        }
        free(pw); free(pb);
    }
    free(img_norm);

    /* ─── Step 3: CLS token + storage tokens (NO positional embeddings) ─── */
    {
        float *cls = qt_dequant(&m->cls_token);
        if (cls) {
            /* cls_token may be [1, dim] or [dim] */
            memcpy(hidden, cls, (size_t)dim * sizeof(float));
            free(cls);
        }

        if (m->storage_tokens.data && ns > 0) {
            float *stor = qt_dequant(&m->storage_tokens);
            if (stor) {
                /* storage_tokens: [1, n_storage, dim] or [n_storage, dim] */
                int offset = 0;
                if (m->storage_tokens.n_dims == 3) {
                    /* Skip leading batch dim: data starts at same place */
                }
                for (int s = 0; s < ns; s++) {
                    memcpy(hidden + (1 + s) * dim, stor + (offset + s) * dim,
                           (size_t)dim * sizeof(float));
                }
                free(stor);
            }
        }

        /* DINOv3: NO learned positional embeddings — RoPE is the only position encoding */
    }

    /* ─── Build DINOv3 RoPE sin/cos embedding for patch tokens ─── */
    /* DINOv3 RoPE: 0.5-centered coords normalized to [-1,1], periods = temp^(2i/(dim//2))
     * Embedding layout: [sin_y_0..sin_y_{D/4-1}, sin_x_0..sin_x_{D/4-1},
     *                     sin_y_0..sin_y_{D/4-1}, sin_x_0..sin_x_{D/4-1}]  (tiled 2x for rotate_half)
     * Same for cos. Total: [np, head_dim] for sin and cos each.
     * Applied only to patch tokens (prefix tokens are skipped). */
    int hd = m->head_dim;           /* 64 */
    int rope_dim4 = hd / 4;         /* 16: frequencies per spatial axis */
    float *rope_sin = (float *)malloc((size_t)np * hd * sizeof(float));
    float *rope_cos = (float *)malloc((size_t)np * hd * sizeof(float));
    {
        /* Periods: prefer the saved `rope_embed.periods` tensor (DINOv3-H+
         * ships them in the checkpoint); fall back to computing from
         * rope_freq_base when absent. */
        float periods[64]; /* max rope_dim4 */
        if (m->has_rope_periods_tensor && m->rope_periods.data) {
            float *tmp = qt_dequant(&m->rope_periods);
            int n = qt_numel(&m->rope_periods);
            int k = n < rope_dim4 ? n : rope_dim4;
            for (int j = 0; j < k; j++) periods[j] = tmp[j];
            for (int j = k; j < rope_dim4; j++) periods[j] = tmp[k - 1];
            free(tmp);
        } else {
            for (int j = 0; j < rope_dim4; j++) {
                float exponent = 2.0f * (float)j / (float)(hd / 2);
                periods[j] = powf(m->rope_freq_base, exponent);
            }
        }
        /* Compute coords: (0.5 + i) / N * 2 - 1 for each patch position */
        for (int p = 0; p < np; p++) {
            int py = p / gw;
            int px = p % gw;
            float cy = ((0.5f + (float)py) / (float)gh) * 2.0f - 1.0f;
            float cx = ((0.5f + (float)px) / (float)gw) * 2.0f - 1.0f;
            /* angles: 2*pi * coord / period for each freq */
            float angles_y[64], angles_x[64];
            for (int j = 0; j < rope_dim4; j++) {
                angles_y[j] = 2.0f * 3.14159265358979323846f * cy / periods[j];
                angles_x[j] = 2.0f * 3.14159265358979323846f * cx / periods[j];
            }
            /* Tile (rotate_half layout): [angles_y, angles_x, angles_y, angles_x] = hd values */
            float *s = rope_sin + p * hd;
            float *c = rope_cos + p * hd;
            for (int j = 0; j < rope_dim4; j++) {
                s[j]                = sinf(angles_y[j]);
                s[rope_dim4 + j]    = sinf(angles_x[j]);
                s[2*rope_dim4 + j]  = sinf(angles_y[j]);
                s[3*rope_dim4 + j]  = sinf(angles_x[j]);
                c[j]                = cosf(angles_y[j]);
                c[rope_dim4 + j]    = cosf(angles_x[j]);
                c[2*rope_dim4 + j]  = cosf(angles_y[j]);
                c[3*rope_dim4 + j]  = cosf(angles_x[j]);
            }
        }
    }

    /* ─── Step 4: Transformer blocks ─── */
    double t_backbone = dinov3_time_ms();
    float *ln_buf   = (float *)malloc((size_t)nt * dim * sizeof(float));
    float *qkv      = (float *)malloc((size_t)nt * 3 * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)nt * dim * sizeof(float));
    int ffn_dim = m->ffn_hidden;
    float *ffn_buf  = (float *)malloc((size_t)nt * ffn_dim * sizeof(float));

    for (int L = 0; L < m->n_blocks; L++) {
        dinov3_block *blk = &m->blocks[L];

        /* LayerNorm 1 */
        dinov3_layernorm_batch(ln_buf, hidden, &blk->ln1_w, &blk->ln1_b,
                               nt, dim, m->ln_eps);

        /* QKV projection */
        dinov3_batch_gemm(qkv, &blk->attn_qkv_w, &blk->attn_qkv_b,
                          ln_buf, nt, 3 * dim, dim, n_threads);

        /* QK normalization (if present) */
        if (m->has_qk_norm && blk->attn_q_norm_w.data) {
            float *q_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            float *k_flat = (float *)malloc((size_t)nt * dim * sizeof(float));
            for (int t = 0; t < nt; t++) {
                memcpy(q_flat + t * dim, qkv + t * 3 * dim, (size_t)dim * sizeof(float));
                memcpy(k_flat + t * dim, qkv + t * 3 * dim + dim, (size_t)dim * sizeof(float));
            }
            dinov3_qk_norm_batch(q_flat, nt, m->n_heads, m->head_dim,
                                  &blk->attn_q_norm_w, &blk->attn_q_norm_b, m->ln_eps);
            dinov3_qk_norm_batch(k_flat, nt, m->n_heads, m->head_dim,
                                  &blk->attn_k_norm_w, &blk->attn_k_norm_b, m->ln_eps);
            for (int t = 0; t < nt; t++) {
                memcpy(qkv + t * 3 * dim, q_flat + t * dim, (size_t)dim * sizeof(float));
                memcpy(qkv + t * 3 * dim + dim, k_flat + t * dim, (size_t)dim * sizeof(float));
            }
            free(q_flat); free(k_flat);
        }

        /* DINOv3 RoPE: apply only to patch tokens (skip prefix tokens)
         * rotate_half convention: out = x * cos + rotate_half(x) * sin
         * where rotate_half([x0..x_{D/2-1}, x_{D/2}..x_{D-1}]) =
         *       [-x_{D/2}..-x_{D-1}, x_0..x_{D/2-1}] */
        {
            int half_hd = hd / 2;
            for (int p = 0; p < np; p++) {
                int t = patch_start + p;
                const float *s = rope_sin + p * hd;
                const float *c = rope_cos + p * hd;
                /* Apply to Q and K for each head */
                for (int qk = 0; qk < 2; qk++) {
                    for (int h = 0; h < m->n_heads; h++) {
                        float *v = qkv + t * 3 * dim + qk * dim + h * hd;
                        /* rotate_half: [-v[D/2..D-1], v[0..D/2-1]] */
                        float tmp[128]; /* max head_dim */
                        for (int j = 0; j < half_hd; j++) {
                            tmp[j]           = -v[half_hd + j];
                            tmp[half_hd + j] =  v[j];
                        }
                        for (int j = 0; j < hd; j++) {
                            v[j] = v[j] * c[j] + tmp[j] * s[j];
                        }
                    }
                }
            }
        }

        /* Multi-head attention */
        cpu_attention(attn_out, qkv, nt, dim, m->n_heads, m->head_dim, n_threads);

        /* Attention output projection */
        float *proj_out = (float *)malloc((size_t)nt * dim * sizeof(float));
        dinov3_batch_gemm(proj_out, &blk->attn_out_w, &blk->attn_out_b,
                          attn_out, nt, dim, dim, n_threads);

        /* LayerScale 1 + residual */
        dinov3_layerscale(proj_out, &blk->ls1, nt, dim);
        for (int i = 0; i < nt * dim; i++) hidden[i] += proj_out[i];
        free(proj_out);

        /* LayerNorm 2 */
        dinov3_layernorm_batch(ln_buf, hidden, &blk->ln2_w, &blk->ln2_b,
                               nt, dim, m->ln_eps);

        /* FFN: GELU MLP (fc1 → GELU → fc2), or SwiGLU (w3(silu(w1x+b1)*(w2x+b2))+b3) */
        float *ffn_out = (float *)malloc((size_t)nt * dim * sizeof(float));
        if (m->has_swiglu && blk->ffn_w1_w.data && blk->ffn_w3_w.data) {
            int fd = blk->ffn_w1_w.n_rows;
            float *gate = ffn_buf;  /* reuse */
            float *up   = (float *)malloc((size_t)nt * fd * sizeof(float));
            dinov3_batch_gemm(gate, &blk->ffn_w1_w, &blk->ffn_w1_b,
                              ln_buf, nt, fd, dim, n_threads);
            dinov3_batch_gemm(up,   &blk->ffn_w2_w, &blk->ffn_w2_b,
                              ln_buf, nt, fd, dim, n_threads);
            /* SiLU(gate) * up */
            for (int i = 0; i < nt * fd; i++) {
                float g = gate[i];
                gate[i] = (g / (1.0f + expf(-g))) * up[i];
            }
            free(up);
            dinov3_batch_gemm(ffn_out, &blk->ffn_w3_w, &blk->ffn_w3_b,
                              gate, nt, dim, fd, n_threads);
        } else if (blk->ffn_up_w.data) {
            int fd = blk->ffn_up_w.n_rows;
            dinov3_batch_gemm(ffn_buf, &blk->ffn_up_w, &blk->ffn_up_b,
                              ln_buf, nt, fd, dim, n_threads);
            /* GELU activation */
            for (int i = 0; i < nt * fd; i++) {
                float v = ffn_buf[i];
                ffn_buf[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
            }
            dinov3_batch_gemm(ffn_out, &blk->ffn_down_w, &blk->ffn_down_b,
                              ffn_buf, nt, dim, fd, n_threads);
        } else {
            memset(ffn_out, 0, (size_t)nt * dim * sizeof(float));
        }

        /* LayerScale 2 + residual */
        dinov3_layerscale(ffn_out, &blk->ls2, nt, dim);
        for (int i = 0; i < nt * dim; i++) hidden[i] += ffn_out[i];
        free(ffn_out);
    }

    free(ln_buf); free(qkv); free(attn_out); free(ffn_buf);
    free(rope_sin); free(rope_cos);

    double t_norm = dinov3_time_ms();
    fprintf(stderr, "dinov3: preprocess+embed %.1f ms, backbone %.1f ms (%d threads)\n",
            t_backbone - t_start, t_norm - t_backbone, n_threads);

    /* ─── Step 5: Final LayerNorm ─── */
    /* Two modes:
     *   - use_learned_final_norm=0 (default): unparameterized LN
     *     (x-mean)/std — matches TRELLIS.2's feature-extraction hack.
     *   - use_learned_final_norm=1: standard LayerNorm with the
     *     model's norm_w / norm_b (sam-3d-body, DINOv3 feature path). */
    float *output = (float *)malloc((size_t)nt * dim * sizeof(float));
    if (m->use_learned_final_norm && m->norm_w.data) {
        dinov3_layernorm_batch(output, hidden, &m->norm_w, &m->norm_b,
                               nt, dim, m->ln_eps);
    } else {
        for (int t = 0; t < nt; t++) {
            const float *xi = hidden + t * dim;
            float *yi = output + t * dim;
            float mean = 0.0f;
            for (int i = 0; i < dim; i++) mean += xi[i];
            mean /= dim;
            float var = 0.0f;
            for (int i = 0; i < dim; i++) { float d = xi[i] - mean; var += d * d; }
            var /= dim;
            float inv = 1.0f / sqrtf(var + m->ln_eps);
            for (int i = 0; i < dim; i++) yi[i] = (xi[i] - mean) * inv;
        }
    }
    free(hidden);

    double t_end = dinov3_time_ms();
    fprintf(stderr, "dinov3: total %.1f ms\n", t_end - t_start);

    result.features = output;
    result.n_tokens = nt;
    result.dim = dim;
    return result;
}

dinov3_result dinov3_encode(dinov3_model *m, const uint8_t *rgb,
                            int img_w, int img_h, int n_threads) {
    int ps = m->patch_size;
    int target_h = m->grid_h * ps;
    int target_w = m->grid_w * ps;
    float *img_norm = (float *)malloc((size_t)3 * target_h * target_w * sizeof(float));
    for (int c = 0; c < 3; c++) {
        for (int oh = 0; oh < target_h; oh++) {
            float fy = (target_h > 1) ? (float)oh * (img_h - 1) / (target_h - 1) : 0.0f;
            int y0 = (int)fy; int y1 = y0 + 1 < img_h ? y0 + 1 : y0;
            float dy = fy - y0;
            for (int ow = 0; ow < target_w; ow++) {
                float fx = (target_w > 1) ? (float)ow * (img_w - 1) / (target_w - 1) : 0.0f;
                int x0 = (int)fx; int x1 = x0 + 1 < img_w ? x0 + 1 : x0;
                float dx = fx - x0;
                float v = (float)rgb[(y0 * img_w + x0) * 3 + c] * (1-dy)*(1-dx)
                        + (float)rgb[(y0 * img_w + x1) * 3 + c] * (1-dy)*dx
                        + (float)rgb[(y1 * img_w + x0) * 3 + c] * dy*(1-dx)
                        + (float)rgb[(y1 * img_w + x1) * 3 + c] * dy*dx;
                img_norm[c * target_h * target_w + oh * target_w + ow] =
                    (v / 255.0f - m->image_mean[c]) / m->image_std[c];
            }
        }
    }
    return dinov3_encode_norm_(m, img_norm, n_threads);
}

dinov3_result dinov3_encode_from_normalized(dinov3_model *m, const float *chw,
                                            int img_w, int img_h, int n_threads) {
    dinov3_result zero = {0};
    int target_h = m->grid_h * m->patch_size;
    int target_w = m->grid_w * m->patch_size;
    if (img_w != target_w || img_h != target_h) {
        fprintf(stderr, "dinov3_encode_from_normalized: input %dx%d != expected %dx%d\n",
                img_w, img_h, target_w, target_h);
        return zero;
    }
    size_t nbytes = (size_t)3 * target_h * target_w * sizeof(float);
    float *img_norm = (float *)malloc(nbytes);
    memcpy(img_norm, chw, nbytes);
    return dinov3_encode_norm_(m, img_norm, n_threads);
}

void dinov3_result_free(dinov3_result *r) {
    if (!r) return;
    free(r->features);
    r->features = NULL;
}

#endif /* DINOV3_IMPLEMENTATION */
#endif /* DINOV3_H */
