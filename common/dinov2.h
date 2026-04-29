/*
 * dinov2.h - DINOv2-L/14 + 4 register tokens vision encoder
 * (SAM-3D-Objects conditioning backbone).
 *
 * Usage:
 *   #define DINOV2_IMPLEMENTATION
 *   #include "dinov2.h"
 *
 * Dependencies: ggml_dequant.h, safetensors.h, cpu_compute.h
 *
 * API:
 *   dinov2_model  *dinov2_load_safetensors(const char *st_path);
 *   void           dinov2_free(dinov2_model *m);
 *   dinov2_result  dinov2_encode(dinov2_model *m, const uint8_t *rgb,
 *                                int w, int h, int n_threads);
 *   void           dinov2_result_free(dinov2_result *r);
 *
 * DINOv2-L/14+reg vs DINOv3-L/16:
 *   - Patch=14, image_size=518, grid=37×37 (1369 patches)
 *   - Learned absolute pos_embed: [1, 1+N_patches_orig, dim] = [1,1370,1024]
 *     bicubic-interpolated when the test-time grid differs (Keys a=-0.75,
 *     interp_offset=0.1 — matches the reference impl).
 *   - Register tokens (4) concatenated AFTER CLS, BEFORE patches:
 *       [CLS, reg_1..reg_4, patch_1..patch_N]   (total = 1 + 4 + N_p)
 *   - pos_embed is added to CLS+patches ONLY (not register tokens).
 *   - Standard GELU MLP (no SwiGLU).
 *   - LayerScale ls1/ls2 present on every block.
 *   - Final LayerNorm uses the model's learned weight/bias.
 *   - Output: all tokens post-norm (1 + 4 + grid_h*grid_w).
 */
#ifndef DINOV2_H
#define DINOV2_H

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
    qtensor attn_out_w, attn_out_b;
    qtensor ls1;                            /* LayerScale gamma */
    qtensor ln2_w, ln2_b;
    qtensor ffn_up_w, ffn_up_b;             /* fc1 [ffn_dim, dim] */
    qtensor ffn_down_w, ffn_down_b;         /* fc2 [dim, ffn_dim] */
    qtensor ls2;
} dinov2_block;

typedef struct {
    int n_blocks, dim, n_heads, head_dim, ffn_hidden;
    int patch_size, image_size;
    int grid_h, grid_w, n_patches;
    int n_register;                         /* 4 */
    int orig_grid;                          /* 37 (from pos_embed) */
    int n_tokens;                           /* 1 + n_register + n_patches */
    float ln_eps;
    float image_mean[3], image_std[3];
    int prenorm_features;                   /* 1 → skip final LN (default 0) */

    qtensor patch_embed_w, patch_embed_b;
    qtensor cls_token;                      /* [1, 1, dim] */
    qtensor register_tokens;                /* [1, n_register, dim] */
    qtensor pos_embed;                      /* [1, 1 + orig_grid^2, dim] */
    qtensor norm_w, norm_b;                 /* final LayerNorm */

    dinov2_block *blocks;

    void *st_ctx;                           /* st_context * (alive for mmap data) */
} dinov2_model;

typedef struct {
    float *features;                        /* [n_tokens, dim] */
    int n_tokens;
    int dim;
} dinov2_result;

dinov2_model  *dinov2_load_safetensors(const char *st_path);
void           dinov2_free(dinov2_model *m);
dinov2_result  dinov2_encode(dinov2_model *m, const uint8_t *rgb,
                             int w, int h, int n_threads);
/* Same as dinov2_encode but takes already-normalized f32 CHW input at
 * exactly grid*patch size; skips the bilinear-resize + ImageNet-norm
 * step. Pass in_h = in_w = grid*patch_size. */
dinov2_result  dinov2_encode_f32(dinov2_model *m, const float *chw,
                                 int w, int h, int n_threads);
/* Strip the n_register register tokens from a result, leaving only
 * [CLS, patch_0, ..., patch_{N-1}] = (1 + n_patches) tokens. Matches
 * the standard dinov2 Dino.forward output layout. Rewrites r in place. */
void           dinov2_result_drop_registers(dinov2_result *r, int n_register);
void           dinov2_result_free(dinov2_result *r);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef DINOV2_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#define CPU_COMPUTE_IMPLEMENTATION
#include "cpu_compute.h"

static double dinov2_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Return a F32 row pointer for t, no-alloc if t is already F32. *owned is
 * set to the malloc'd buffer that the caller must free (NULL if no alloc). */
static const float *dinov2_row_f32(const qtensor *t, int dim, float **owned) {
    *owned = NULL;
    if (!t || !t->data) return NULL;
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    float *buf = (float *)malloc((size_t)dim * sizeof(float));
    qt_dequant_row(t, 0, buf);
    *owned = buf;
    return buf;
}

static void dinov2_batch_gemm(float *dst, const qtensor *W, const qtensor *bias,
                              const float *src, int n_tok, int n_out, int n_in,
                              int n_threads) {
    if (!W->data) {
        memset(dst, 0, (size_t)n_tok * n_out * sizeof(float));
        return;
    }
    float *bown = NULL;
    const float *b = bias ? dinov2_row_f32(bias, n_out, &bown) : NULL;
    if (W->type == GGML_TYPE_F16) {
        /* cpu_gemm_f16 expects a mutable float*; fine since it doesn't write. */
        cpu_gemm_f16(dst, (const uint16_t *)W->data, (float *)b, src,
                     n_tok, n_out, n_in, n_threads);
    } else if (W->type == GGML_TYPE_F32) {
        cpu_gemm_f32(dst, (const float *)W->data, b, src,
                        n_tok, n_out, n_in, n_threads);
    } else {
        /* Quantized: dequant the whole matrix once, then F32 GEMM. */
        float *Wf = qt_dequant(W);
        cpu_gemm_f32(dst, Wf, b, src, n_tok, n_out, n_in, n_threads);
        free(Wf);
    }
    free(bown);
}

static void dinov2_layernorm_batch(float *dst, const float *src, const qtensor *w,
                                   const qtensor *b, int n_tok, int dim, float eps) {
    float *wown = NULL, *bown = NULL;
    const float *wf = dinov2_row_f32(w, dim, &wown);
    const float *bf = dinov2_row_f32(b, dim, &bown);
    cpu_layernorm(dst, src, wf, bf, n_tok, dim, eps);
    free(wown); free(bown);
}

static void dinov2_layerscale(float *x, const qtensor *gamma, int n_tok, int dim) {
    if (!gamma->data) return;
    float *gown = NULL;
    const float *g = dinov2_row_f32(gamma, dim, &gown);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n_tok; t++) {
        float *v = x + (size_t)t * dim;
        for (int i = 0; i < dim; i++) v[i] *= g[i];
    }
    free(gown);
}

/* ==================================================================== */
/* SafeTensors loading                                                   */
/* ==================================================================== */

#ifdef SAFETENSORS_H

dinov2_model *dinov2_load_safetensors(const char *st_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) return NULL;

    fprintf(stderr, "dinov2: safetensors opened, %d tensors\n", st->n_tensors);

    /* Auto-detect geometry from tensor shapes. */
    int embed_dim = 1024, head_dim = 64, patch_size = 14, image_size = 518;
    int n_blocks = 0, ffn_hidden = 4096;
    int n_register = 0, orig_grid = 37;

    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const uint64_t *sh = safetensors_shape(st, i);
        int nd = safetensors_ndims(st, i);

        if (strcmp(nm, "patch_embed.proj.weight") == 0 && nd == 4) {
            embed_dim  = (int)sh[0];
            patch_size = (int)sh[2];
        }
        if (strcmp(nm, "register_tokens") == 0) {
            n_register = (nd == 3) ? (int)sh[1] : (int)sh[0];
        }
        if (strcmp(nm, "pos_embed") == 0) {
            /* shape (1, 1 + orig_grid^2, dim) */
            int ntok_minus_cls = (nd == 3) ? (int)sh[1] - 1 : (int)sh[0] - 1;
            int g = (int)(sqrtf((float)ntok_minus_cls) + 0.5f);
            if (g * g == ntok_minus_cls) orig_grid = g;
        }
        if (strcmp(nm, "blocks.0.mlp.fc1.weight") == 0) {
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

    int n_heads = embed_dim / head_dim;
    int grid_h = image_size / patch_size;      /* 37 for 518/14 */
    int grid_w = grid_h;
    int n_patches = grid_h * grid_w;
    int n_tokens = 1 + n_register + n_patches;

    fprintf(stderr, "dinov2: dim=%d, heads=%d, head_dim=%d, blocks=%d, ffn=%d\n",
            embed_dim, n_heads, head_dim, n_blocks, ffn_hidden);
    fprintf(stderr, "dinov2: patch=%d, image=%d, grid=%dx%d (orig=%d), "
                    "n_reg=%d, n_tokens=%d\n",
            patch_size, image_size, grid_h, grid_w, orig_grid,
            n_register, n_tokens);

    dinov2_model *m = (dinov2_model *)calloc(1, sizeof(dinov2_model));
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
    m->n_register = n_register;
    m->orig_grid = orig_grid;
    m->n_tokens = n_tokens;
    m->ln_eps = 1e-6f;
    m->prenorm_features = 0;
    /* ImageNet normalization (DINOv2 upstream + SAM-3D preprocess). */
    m->image_mean[0] = 0.485f; m->image_mean[1] = 0.456f; m->image_mean[2] = 0.406f;
    m->image_std[0]  = 0.229f; m->image_std[1]  = 0.224f; m->image_std[2]  = 0.225f;
    m->st_ctx = st;

    #define DINOV2_FIND(nm) ({ int _i = safetensors_find(st, nm); \
                              (_i >= 0) ? qt_make_tensor(st, _i) : (qtensor){0}; })

    m->cls_token       = DINOV2_FIND("cls_token");
    m->register_tokens = DINOV2_FIND("register_tokens");
    m->pos_embed       = DINOV2_FIND("pos_embed");
    m->patch_embed_w   = DINOV2_FIND("patch_embed.proj.weight");
    m->patch_embed_b   = DINOV2_FIND("patch_embed.proj.bias");
    m->norm_w          = DINOV2_FIND("norm.weight");
    m->norm_b          = DINOV2_FIND("norm.bias");

    fprintf(stderr, "dinov2: cls=%s reg=%s pos=%s patch=%s norm=%s\n",
            m->cls_token.data       ? "ok" : "MISSING",
            m->register_tokens.data ? "ok" : (n_register ? "MISSING" : "none"),
            m->pos_embed.data       ? "ok" : "MISSING",
            m->patch_embed_w.data   ? "ok" : "MISSING",
            m->norm_w.data          ? "ok" : "MISSING");

    m->blocks = (dinov2_block *)calloc((size_t)m->n_blocks, sizeof(dinov2_block));
    for (int L = 0; L < m->n_blocks; L++) {
        dinov2_block *blk = &m->blocks[L];
        char nm[256];
        #define LOAD(field, suffix) do {                            \
            snprintf(nm, sizeof(nm), "blocks.%d.%s", L, suffix);    \
            int _i = safetensors_find(st, nm);                      \
            if (_i >= 0) blk->field = qt_make_tensor(st, _i);   \
        } while (0)

        LOAD(ln1_w,        "norm1.weight");
        LOAD(ln1_b,        "norm1.bias");
        LOAD(attn_qkv_w,   "attn.qkv.weight");
        LOAD(attn_qkv_b,   "attn.qkv.bias");
        LOAD(attn_out_w,   "attn.proj.weight");
        LOAD(attn_out_b,   "attn.proj.bias");
        LOAD(ls1,          "ls1.gamma");
        LOAD(ln2_w,        "norm2.weight");
        LOAD(ln2_b,        "norm2.bias");
        LOAD(ffn_up_w,     "mlp.fc1.weight");
        LOAD(ffn_up_b,     "mlp.fc1.bias");
        LOAD(ffn_down_w,   "mlp.fc2.weight");
        LOAD(ffn_down_b,   "mlp.fc2.bias");
        LOAD(ls2,          "ls2.gamma");

        #undef LOAD

        if (L == 0) {
            if (!blk->ln1_w.data)      fprintf(stderr, "dinov2: WARN block 0 ln1_w missing\n");
            if (!blk->attn_qkv_w.data) fprintf(stderr, "dinov2: WARN block 0 qkv missing\n");
            if (!blk->attn_out_w.data) fprintf(stderr, "dinov2: WARN block 0 attn_out missing\n");
            if (!blk->ffn_up_w.data)   fprintf(stderr, "dinov2: WARN block 0 ffn_up missing\n");
            if (!blk->ls1.data)        fprintf(stderr, "dinov2: WARN block 0 ls1 missing\n");
        }
    }
    #undef DINOV2_FIND

    fprintf(stderr, "dinov2: loaded %d blocks\n", m->n_blocks);
    return m;
}

#endif /* SAFETENSORS_H */

void dinov2_free(dinov2_model *m) {
    if (!m) return;
    free(m->blocks);
#ifdef SAFETENSORS_H
    if (m->st_ctx) safetensors_close((st_context *)m->st_ctx);
#endif
    free(m);
}

/* ==================================================================== */
/* API: dinov2_encode                                                    */
/* ==================================================================== */

/* Core encode — operates on a pre-normalized float32 CHW buffer already
 * at (target_h, target_w) = (grid*patch, grid*patch). */
static dinov2_result dinov2_encode_core(dinov2_model *m, float *img_norm,
                                        int n_threads);

dinov2_result dinov2_encode(dinov2_model *m, const uint8_t *rgb,
                            int img_w, int img_h, int n_threads) {
    int ps       = m->patch_size;
    int gh       = m->grid_h, gw = m->grid_w;
    int target_h = gh * ps;                  /* 518 */
    int target_w = gw * ps;

    if (n_threads < 1) n_threads = 1;

    /* Preprocess — bilinear resize to 518×518 + ImageNet norm. */
    float *img_norm = (float *)malloc((size_t)3 * target_h * target_w * sizeof(float));
    for (int c = 0; c < 3; c++) {
        for (int oh = 0; oh < target_h; oh++) {
            float fy = (target_h > 1) ? (float)oh * (img_h - 1) / (target_h - 1) : 0.0f;
            int y0 = (int)fy; int y1 = y0 + 1 < img_h ? y0 + 1 : y0;
            float dy = fy - (float)y0;
            for (int ow = 0; ow < target_w; ow++) {
                float fx = (target_w > 1) ? (float)ow * (img_w - 1) / (target_w - 1) : 0.0f;
                int x0 = (int)fx; int x1 = x0 + 1 < img_w ? x0 + 1 : x0;
                float dx = fx - (float)x0;
                float v = (float)rgb[(y0 * img_w + x0) * 3 + c] * (1 - dy) * (1 - dx)
                        + (float)rgb[(y0 * img_w + x1) * 3 + c] * (1 - dy) * dx
                        + (float)rgb[(y1 * img_w + x0) * 3 + c] * dy       * (1 - dx)
                        + (float)rgb[(y1 * img_w + x1) * 3 + c] * dy       * dx;
                img_norm[c * target_h * target_w + oh * target_w + ow] =
                    (v / 255.0f - m->image_mean[c]) / m->image_std[c];
            }
        }
    }
    return dinov2_encode_core(m, img_norm, n_threads);
}

dinov2_result dinov2_encode_f32(dinov2_model *m, const float *chw,
                                int img_w, int img_h, int n_threads) {
    int ps       = m->patch_size;
    int gh       = m->grid_h, gw = m->grid_w;
    int target_h = gh * ps;
    int target_w = gw * ps;
    if (n_threads < 1) n_threads = 1;
    if (img_w != target_w || img_h != target_h) {
        fprintf(stderr,
                "dinov2_encode_f32: input %dx%d != expected %dx%d\n",
                img_w, img_h, target_w, target_h);
        dinov2_result r = {0};
        return r;
    }
    size_t n = (size_t)3 * target_h * target_w;
    float *img_norm = (float *)malloc(n * sizeof(float));
    memcpy(img_norm, chw, n * sizeof(float));
    return dinov2_encode_core(m, img_norm, n_threads);
}

void dinov2_result_drop_registers(dinov2_result *r, int n_register) {
    if (!r || !r->features || n_register <= 0) return;
    int dim = r->dim;
    int keep_rest = r->n_tokens - 1 - n_register;   /* patches */
    if (keep_rest < 0) keep_rest = 0;
    /* Shift [patch_0..patch_N] to sit immediately after [CLS]. */
    memmove(r->features + (size_t)dim,
            r->features + (size_t)(1 + n_register) * dim,
            (size_t)keep_rest * dim * sizeof(float));
    r->n_tokens = 1 + keep_rest;
}

/* img_norm is consumed (freed) by this function. */
static dinov2_result dinov2_encode_core(dinov2_model *m, float *img_norm,
                                        int n_threads) {
    dinov2_result result = {0};
    int ps  = m->patch_size;
    int gh  = m->grid_h, gw = m->grid_w;
    int np  = m->n_patches;
    int nr  = m->n_register;
    int nt  = m->n_tokens;
    int dim = m->dim;
    int target_h = gh * ps;
    int target_w = gw * ps;
    int patch_start = 1 + nr;
    double t_start = dinov2_time_ms();

    /* ─── Step 2: Patch embedding (Conv2d stride=ps, kernel=ps) ─── */
    float *hidden = (float *)calloc((size_t)nt * dim, sizeof(float));
    {
        float *pw = qt_dequant(&m->patch_embed_w);   /* [Co, Ci, ps, ps] */
        float *pb = qt_dequant(&m->patch_embed_b);
        int Co = dim, Ci = 3;
#if defined(_OPENMP)
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int py = 0; py < gh; py++) {
            for (int px = 0; px < gw; px++) {
                int tok = patch_start + py * gw + px;
                float *out = hidden + (size_t)tok * dim;
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

    /* ─── Step 3: CLS token + pos_embed (CLS+patches), then register tokens ─── */
    {
        float *cls = qt_dequant(&m->cls_token);
        if (cls) {
            memcpy(hidden, cls, (size_t)dim * sizeof(float));   /* [CLS] at tok 0 */
            free(cls);
        }

        /* pos_embed shape: [1, 1 + M*M, dim]. Apply to [CLS] + patches. */
        float *pe_orig = qt_dequant(&m->pos_embed);
        if (pe_orig) {
            /* CLS pos-embed: unchanged. */
            for (int i = 0; i < dim; i++) hidden[i] += pe_orig[i];
            const float *patch_pe_src = pe_orig + dim;          /* [M*M, dim] */
            int M = m->orig_grid;
            if (gh == M && gw == M) {
                for (int p = 0; p < np; p++) {
                    float *dst = hidden + (patch_start + p) * dim;
                    const float *src = patch_pe_src + p * dim;
                    for (int i = 0; i < dim; i++) dst[i] += src[i];
                }
            } else {
                float *patch_pe = (float *)malloc((size_t)np * dim * sizeof(float));
                cpu_interp_pos_embed_bicubic(patch_pe_src, M, patch_pe, gh, gw, dim);
                for (int p = 0; p < np; p++) {
                    float *dst = hidden + (patch_start + p) * dim;
                    const float *src = patch_pe + p * dim;
                    for (int i = 0; i < dim; i++) dst[i] += src[i];
                }
                free(patch_pe);
            }
            free(pe_orig);
        }

        /* Register tokens (no pos_embed). Placed between CLS and patches. */
        if (m->register_tokens.data && nr > 0) {
            float *reg = qt_dequant(&m->register_tokens);
            if (reg) {
                for (int r = 0; r < nr; r++)
                    memcpy(hidden + (1 + r) * dim, reg + r * dim,
                           (size_t)dim * sizeof(float));
                free(reg);
            }
        }
    }

    /* ─── Step 4: Transformer blocks ─── */
    double t_backbone = dinov2_time_ms();
    float *ln_buf   = (float *)malloc((size_t)nt * dim * sizeof(float));
    float *qkv      = (float *)malloc((size_t)nt * 3 * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)nt * dim * sizeof(float));
    float *proj_out = (float *)malloc((size_t)nt * dim * sizeof(float));
    int ffn_dim     = m->ffn_hidden;
    float *ffn_buf  = (float *)malloc((size_t)nt * ffn_dim * sizeof(float));
    float *ffn_out  = (float *)malloc((size_t)nt * dim * sizeof(float));

    for (int L = 0; L < m->n_blocks; L++) {
        dinov2_block *blk = &m->blocks[L];

        /* pre-attn LN */
        dinov2_layernorm_batch(ln_buf, hidden, &blk->ln1_w, &blk->ln1_b,
                               nt, dim, m->ln_eps);
        /* QKV projection */
        dinov2_batch_gemm(qkv, &blk->attn_qkv_w, &blk->attn_qkv_b,
                          ln_buf, nt, 3 * dim, dim, n_threads);
        /* Multi-head attention (no RoPE, no QK norm). */
        cpu_attention(attn_out, qkv, nt, dim, m->n_heads, m->head_dim, n_threads);
        /* Output projection */
        dinov2_batch_gemm(proj_out, &blk->attn_out_w, &blk->attn_out_b,
                          attn_out, nt, dim, dim, n_threads);
        /* LayerScale 1 + residual */
        dinov2_layerscale(proj_out, &blk->ls1, nt, dim);
        int n_nd = nt * dim;
        int n_ff = nt * ffn_dim;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_nd; i++) hidden[i] += proj_out[i];

        /* pre-MLP LN */
        dinov2_layernorm_batch(ln_buf, hidden, &blk->ln2_w, &blk->ln2_b,
                               nt, dim, m->ln_eps);
        /* MLP: fc1 → exact GELU → fc2. Upstream DINOv2 uses nn.GELU() with
         * default approximate='none' (erf-based). */
        dinov2_batch_gemm(ffn_buf, &blk->ffn_up_w, &blk->ffn_up_b,
                          ln_buf, nt, ffn_dim, dim, n_threads);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_ff; i++) {
            float v = ffn_buf[i];
            ffn_buf[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
        }
        dinov2_batch_gemm(ffn_out, &blk->ffn_down_w, &blk->ffn_down_b,
                          ffn_buf, nt, dim, ffn_dim, n_threads);
        /* LayerScale 2 + residual */
        dinov2_layerscale(ffn_out, &blk->ls2, nt, dim);
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_nd; i++) hidden[i] += ffn_out[i];
    }

    free(ln_buf); free(qkv); free(attn_out);
    free(proj_out); free(ffn_buf); free(ffn_out);

    double t_norm = dinov2_time_ms();

    /* ─── Step 5: Final LayerNorm (learned weight/bias) ─── */
    float *output = (float *)malloc((size_t)nt * dim * sizeof(float));
    if (m->prenorm_features || !m->norm_w.data) {
        memcpy(output, hidden, (size_t)nt * dim * sizeof(float));
    } else {
        dinov2_layernorm_batch(output, hidden, &m->norm_w, &m->norm_b,
                               nt, dim, m->ln_eps);
    }
    free(hidden);

    double t_end = dinov2_time_ms();
    fprintf(stderr, "dinov2: preprocess+embed %.1f ms, backbone %.1f ms, "
                    "total %.1f ms (%d threads)\n",
            t_backbone - t_start, t_norm - t_backbone, t_end - t_start, n_threads);

    result.features = output;
    result.n_tokens = nt;
    result.dim = dim;
    return result;
}

void dinov2_result_free(dinov2_result *r) {
    if (!r) return;
    free(r->features);
    r->features = NULL;
}

#endif /* DINOV2_IMPLEMENTATION */
#endif /* DINOV2_H */
