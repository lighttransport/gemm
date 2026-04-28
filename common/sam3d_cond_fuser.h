/*
 * sam3d_cond_fuser.h — PointPatchEmbed + EmbedderFuser for sam-3d-objects.
 *
 * PointPatchEmbed (pytorch:
 *   sam3d_objects/model/backbone/dit/embedder/pointmap.py):
 *   - input:  MoGe pointmap [3, H, W], resized nearest to 256×256
 *     (remap_output="linear" → pass-through; sam3d config)
 *   - Linear(3, 512) on each (x,y,z); invalid-xyz replacement for NaN
 *     pixels (pytorch: x[~valid]=0 then x[~valid]+=invalid_xyz_token)
 *   - reshape into 32×32 windows of 8×8 pixels = 64 points/window
 *   - prepend CLS (512), add pos_embed_window (65, 512)
 *   - 1× timm ViT block (LN + MHA heads=16 head_dim=32 qkv_bias +
 *     LN + MLP mlp_ratio=2 → 1024 → 512, exact GELU, no LayerScale)
 *   - extract CLS per window → (1024, 512)
 *   - bilinear-interp pos_embed (1, D, 32, 32) to (num_wh, num_ww, D);
 *     sam3d always runs at input_size 256 → 32×32, no resize needed.
 *
 * EmbedderFuser (pytorch:
 *   sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py):
 *   - per-modality projection_net = LayerNorm(D_in) + llama3-FF(D_in → 2816 → 1024)
 *     SwiGLU: out = w2(silu(w1(x)) * w3(x)).  No biases on w1/w2/w3.
 *   - add learned idx_emb[pos_group_idx] to each modality's token slice
 *   - concatenate all modalities along token dimension
 *
 * For the ss_generator pipeline, kwargs_info is:
 *     module 0 (dino, image): [("image","cropped"), ("rgb_image","full")]
 *     module 1 (dino, mask ): [("mask","cropped"),  ("rgb_image_mask","full")]
 *     module 2 (ppe        ): [("pointmap","cropped"),("rgb_pointmap","full")]
 *   pos_group order-of-encounter: cropped=0, full=1 → idx_emb shape (3,1024),
 *   entries 0,1 used.
 *
 * All buffers are row-major float32 on host. Header-only; call
 * `#define SAM3D_COND_FUSER_IMPLEMENTATION` in exactly one TU.
 * Dependencies: safetensors.h, ggml_dequant.h (same as dinov2.h).
 */

#ifndef SAM3D_COND_FUSER_H
#define SAM3D_COND_FUSER_H

#include "safetensors.h"
#include "ggml_dequant.h"

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
    int input_size;             /* 256 */
    int patch_size;             /* 8 */
    int embed_dim;              /* 512 */
    int num_patches;            /* 32 (per row) */
    int n_heads;                /* 16 */
    int head_dim;               /* 32 */
    int ffn_hidden;             /* 1024 */
    float ln_eps;               /* 1e-6 */

    qtensor point_proj_w, point_proj_b;     /* [D,3], [D] */
    qtensor invalid_xyz_token;              /* [D] */
    qtensor cls_token;                      /* [1,1,D] */
    qtensor pos_embed;                      /* [1, D, Np, Np] */
    qtensor pos_embed_window;               /* [1, 1 + P^2, D] */

    qtensor ln1_w, ln1_b;
    qtensor attn_qkv_w, attn_qkv_b;         /* [3D, D], [3D] */
    qtensor attn_proj_w, attn_proj_b;       /* [D, D], [D] */
    qtensor ln2_w, ln2_b;
    qtensor mlp_fc1_w, mlp_fc1_b;           /* [ffn_h, D], [ffn_h] */
    qtensor mlp_fc2_w, mlp_fc2_b;           /* [D, ffn_h], [D] */

    st_context *st_ctx;
} sam3d_ppe_model;

typedef struct {
    int embed_dim_in;           /* D_in */
    int embed_dim_out;          /* 1024 */
    int ffn_hidden;             /* 2816 */

    qtensor ln_w, ln_b;         /* LayerNorm(D_in), eps=1e-5 */
    qtensor w1;                 /* [ffn_h, D_in] */
    qtensor w2;                 /* [D_out, ffn_h] */
    qtensor w3;                 /* [ffn_h, D_in] */
} sam3d_fuser_projection;

typedef struct {
    int n_modalities;           /* 3: dino-img, dino-mask, point */
    int embed_dim_out;          /* 1024 */
    sam3d_fuser_projection *projs;
    qtensor idx_emb;            /* [n_pos_groups, D_out]; 0=cropped, 1=full */
    float *idx_emb_f32;         /* dequantized cache (owned) */
    st_context *st_ctx;
} sam3d_fuser_model;

/* Modality indices for sam3d_fuser_project() — order is pinned by the
 * safetensors key layout (0.* = dino-image, 1.* = dino-mask, 2.* = point)
 * and must match the ss_generator condition_embedder kwargs_info in the
 * shipped pipeline.yaml. */
enum {
    SAM3D_FUSER_MOD_DINO_IMG = 0,
    SAM3D_FUSER_MOD_DINO_MSK = 1,
    SAM3D_FUSER_MOD_POINT    = 2,
};
enum {
    SAM3D_FUSER_POS_CROPPED = 0,
    SAM3D_FUSER_POS_FULL    = 1,
};

sam3d_ppe_model *sam3d_ppe_load_safetensors(const char *st_path);
void             sam3d_ppe_free(sam3d_ppe_model *m);

/* Encode a pointmap into (num_patches^2, embed_dim).
 * xyz is row-major [H, W, 3] f32. valid_mask may be NULL — on NULL,
 * pixels with non-finite xyz are treated as invalid. The encoder
 * nearest-resizes xyz to (input_size × input_size) then runs the
 * per-window ViT block. Caller free()s the returned buffer. */
float *sam3d_ppe_encode(sam3d_ppe_model *m,
                        const float *xyz, int img_h, int img_w,
                        const uint8_t *valid_mask,
                        int n_threads);

sam3d_fuser_model *sam3d_fuser_load_safetensors(const char *st_path);
void               sam3d_fuser_free(sam3d_fuser_model *m);

/* in:  [n_tokens, D_in]
 * returns [n_tokens, D_out] (caller frees). */
float *sam3d_fuser_project(const sam3d_fuser_model *m, int modality_idx,
                           const float *in, int n_tokens, int n_threads);

/* tokens += idx_emb[pos_idx] (broadcast over n_tokens rows). */
void   sam3d_fuser_add_pos(const sam3d_fuser_model *m, int pos_idx,
                           float *tokens, int n_tokens);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef SAM3D_COND_FUSER_IMPLEMENTATION

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/* ---- qtensor helpers (mirror dinov2.h). ---- */

/* Zero-copy F32 pointer if t is F32, else dequantize into a fresh buf.
 * *owned is set to 1 iff a new buffer was allocated (caller must free). */
static const float *scf_row_f32(const qtensor *t, int *owned) {
    *owned = 0;
    if (!t || !t->data) return NULL;
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    *owned = 1;
    return qt_dequant(t);
}

static qtensor scf_find(st_context *st, const char *name) {
    int i = safetensors_find(st, name);
    if (i < 0) return (qtensor){0};
    return qt_make_tensor(st, i);
}

/* out[M,N] = in[M,K] @ w[N,K]^T + bias[N]. */
static void scf_gemm(float *out, const qtensor *w, const qtensor *b,
                     const float *in, int M, int N, int K, int n_threads) {
    int w_owned = 0;
    const float *wf = scf_row_f32(w, &w_owned);
    int b_owned = 0;
    const float *bf = b ? scf_row_f32(b, &b_owned) : NULL;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#else
    (void)n_threads;
#endif
    for (int m = 0; m < M; m++) {
        const float *xi = in + (size_t)m * K;
        float *yo = out + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            float s = bf ? bf[n] : 0.0f;
            const float *wr = wf + (size_t)n * K;
            int k = 0;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 acc = _mm256_setzero_ps();
            for (; k + 8 <= K; k += 8) {
                __m256 a = _mm256_loadu_ps(wr + k);
                __m256 x = _mm256_loadu_ps(xi + k);
                acc = _mm256_fmadd_ps(a, x, acc);
            }
            float buf[8];
            _mm256_storeu_ps(buf, acc);
            s += buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
#endif
            for (; k < K; k++) s += wr[k] * xi[k];
            yo[n] = s;
        }
    }

    if (w_owned) free((void *)wf);
    if (b_owned) free((void *)bf);
}

static void scf_layernorm_batch(float *out, const float *in,
                                const qtensor *w, const qtensor *b,
                                int n_tokens, int dim, float eps) {
    int w_owned = 0; const float *ww = scf_row_f32(w, &w_owned);
    int b_owned = 0; const float *bb = b ? scf_row_f32(b, &b_owned) : NULL;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n_tokens; t++) {
        const float *x = in + (size_t)t * dim;
        float *y = out + (size_t)t * dim;
        double sum = 0.0, sum2 = 0.0;
        for (int i = 0; i < dim; i++) { sum += x[i]; sum2 += (double)x[i] * x[i]; }
        float mean = (float)(sum / dim);
        float var  = (float)(sum2 / dim) - mean * mean;
        float rstd = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++) {
            float v = (x[i] - mean) * rstd;
            if (ww) v *= ww[i];
            if (bb) v += bb[i];
            y[i] = v;
        }
    }
    if (w_owned) free((void *)ww);
    if (b_owned) free((void *)bb);
}

static void scf_softmax_row(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ---- PointPatchEmbed ----------------------------------------------------- */

sam3d_ppe_model *sam3d_ppe_load_safetensors(const char *st_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) {
        fprintf(stderr, "sam3d_ppe_load: open %s failed\n", st_path);
        return NULL;
    }

    sam3d_ppe_model *m = (sam3d_ppe_model *)calloc(1, sizeof(*m));
    m->st_ctx = st;
    m->ln_eps = 1e-6f;

    m->point_proj_w      = scf_find(st, "point_proj.weight");
    m->point_proj_b      = scf_find(st, "point_proj.bias");
    m->invalid_xyz_token = scf_find(st, "invalid_xyz_token");
    m->cls_token         = scf_find(st, "cls_token");
    m->pos_embed         = scf_find(st, "pos_embed");
    m->pos_embed_window  = scf_find(st, "pos_embed_window");
    m->ln1_w             = scf_find(st, "blocks.0.norm1.weight");
    m->ln1_b             = scf_find(st, "blocks.0.norm1.bias");
    m->attn_qkv_w        = scf_find(st, "blocks.0.attn.qkv.weight");
    m->attn_qkv_b        = scf_find(st, "blocks.0.attn.qkv.bias");
    m->attn_proj_w       = scf_find(st, "blocks.0.attn.proj.weight");
    m->attn_proj_b       = scf_find(st, "blocks.0.attn.proj.bias");
    m->ln2_w             = scf_find(st, "blocks.0.norm2.weight");
    m->ln2_b             = scf_find(st, "blocks.0.norm2.bias");
    m->mlp_fc1_w         = scf_find(st, "blocks.0.mlp.fc1.weight");
    m->mlp_fc1_b         = scf_find(st, "blocks.0.mlp.fc1.bias");
    m->mlp_fc2_w         = scf_find(st, "blocks.0.mlp.fc2.weight");
    m->mlp_fc2_b         = scf_find(st, "blocks.0.mlp.fc2.bias");

    if (!m->point_proj_w.data || !m->cls_token.data || !m->pos_embed.data) {
        fprintf(stderr, "sam3d_ppe_load: required tensor missing\n");
        sam3d_ppe_free(m);
        return NULL;
    }

    m->embed_dim   = (int)m->point_proj_w.dims[0];        /* 512 */
    m->num_patches = (int)m->pos_embed.dims[2];           /* 32 */
    int pew_sz     = (int)m->pos_embed_window.dims[1];    /* 65 */
    m->patch_size  = (int)(sqrtf((float)(pew_sz - 1)) + 0.5f);
    m->input_size  = m->num_patches * m->patch_size;
    m->n_heads     = 16;
    m->head_dim    = m->embed_dim / m->n_heads;
    m->ffn_hidden  = (int)m->mlp_fc1_w.dims[0];

    fprintf(stderr,
            "sam3d_ppe: dim=%d heads=%d head_dim=%d ffn=%d patch=%d "
            "grid=%dx%d input=%d\n",
            m->embed_dim, m->n_heads, m->head_dim, m->ffn_hidden,
            m->patch_size, m->num_patches, m->num_patches, m->input_size);

    return m;
}

void sam3d_ppe_free(sam3d_ppe_model *m) {
    if (!m) return;
    if (m->st_ctx) safetensors_close(m->st_ctx);
    free(m);
}

float *sam3d_ppe_encode(sam3d_ppe_model *m,
                        const float *xyz, int img_h, int img_w,
                        const uint8_t *valid_mask, int n_threads) {
    int D    = m->embed_dim;
    int P    = m->patch_size;
    int Np   = m->num_patches;
    int S    = m->input_size;
    int Nwin = Np * Np;
    int WL   = 1 + P * P;
    if (n_threads < 1) n_threads = 1;

    /* ─── Step 1: nearest-neighbor resize (H,W,3) → (S,S,3). ─── */
    float *xyz_rs = (float *)malloc((size_t)S * S * 3 * sizeof(float));
    uint8_t *valid_rs = (uint8_t *)malloc((size_t)S * S);
    for (int oy = 0; oy < S; oy++) {
        int iy = (int)((float)oy * img_h / S);
        if (iy >= img_h) iy = img_h - 1;
        for (int ox = 0; ox < S; ox++) {
            int ix = (int)((float)ox * img_w / S);
            if (ix >= img_w) ix = img_w - 1;
            const float *src = xyz + ((size_t)iy * img_w + ix) * 3;
            float *dst = xyz_rs + ((size_t)oy * S + ox) * 3;
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            uint8_t v;
            if (valid_mask) {
                v = valid_mask[iy * img_w + ix] ? 1 : 0;
            } else {
                v = (isfinite(src[0]) && isfinite(src[1]) &&
                     isfinite(src[2])) ? 1 : 0;
            }
            valid_rs[oy * S + ox] = v;
            if (!v) { dst[0] = dst[1] = dst[2] = 0.0f; }
        }
    }

    /* ─── Step 2: point_proj(xyz) → (S,S,D); invalid pixels get
     *     invalid_xyz_token (pytorch: x[~valid]=0 then x[~valid]+=tok). ─── */
    float *pp_w    = qt_dequant(&m->point_proj_w);      /* [D, 3] */
    float *pp_b    = qt_dequant(&m->point_proj_b);      /* [D]    */
    float *inv_tok = qt_dequant(&m->invalid_xyz_token); /* [D]    */
    float *xflat   = (float *)malloc((size_t)S * S * D * sizeof(float));
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            int idx = y * S + x;
            const float *src = xyz_rs + (size_t)idx * 3;
            float *dst = xflat + (size_t)idx * D;
            if (!valid_rs[idx]) {
                memcpy(dst, inv_tok, (size_t)D * sizeof(float));
            } else {
                for (int d = 0; d < D; d++) {
                    dst[d] = pp_b[d]
                           + pp_w[d * 3 + 0] * src[0]
                           + pp_w[d * 3 + 1] * src[1]
                           + pp_w[d * 3 + 2] * src[2];
                }
            }
        }
    }
    free(pp_w); free(pp_b); free(inv_tok);
    free(xyz_rs); free(valid_rs);

    /* ─── Step 3: window reshape → (Nwin, WL, D) with CLS + pos_embed_window. ─── */
    float *cls  = qt_dequant(&m->cls_token);         /* [D] */
    float *pew  = qt_dequant(&m->pos_embed_window);  /* [65, D] */
    float *toks = (float *)malloc((size_t)Nwin * WL * D * sizeof(float));
#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static) num_threads(n_threads)
#endif
    for (int wy = 0; wy < Np; wy++) {
        for (int wx = 0; wx < Np; wx++) {
            int w = wy * Np + wx;
            float *wt = toks + (size_t)w * WL * D;
            memcpy(wt, cls, (size_t)D * sizeof(float));
            for (int py = 0; py < P; py++) {
                for (int px = 0; px < P; px++) {
                    int src_y = wy * P + py;
                    int src_x = wx * P + px;
                    const float *src = xflat + ((size_t)src_y * S + src_x) * D;
                    float *dst = wt + (size_t)(1 + py * P + px) * D;
                    memcpy(dst, src, (size_t)D * sizeof(float));
                }
            }
            for (int t = 0; t < WL; t++) {
                float *tt = wt + (size_t)t * D;
                for (int d = 0; d < D; d++) tt[d] += pew[t * D + d];
            }
        }
    }
    free(xflat); free(cls); free(pew);

    /* ─── Step 4: single ViT block (LN → MHA → LN → MLP, residuals). ─── */
    int BN = Nwin * WL;
    float *ln_buf = (float *)malloc((size_t)BN * D * sizeof(float));
    float *qkv    = (float *)malloc((size_t)BN * 3 * D * sizeof(float));
    float *attn   = (float *)malloc((size_t)BN * D * sizeof(float));
    float *proj   = (float *)malloc((size_t)BN * D * sizeof(float));
    float *ffn    = (float *)malloc((size_t)BN * m->ffn_hidden * sizeof(float));

    scf_layernorm_batch(ln_buf, toks, &m->ln1_w, &m->ln1_b, BN, D, m->ln_eps);
    scf_gemm(qkv, &m->attn_qkv_w, &m->attn_qkv_b, ln_buf,
             BN, 3 * D, D, n_threads);

    int hD = m->head_dim, nH = m->n_heads;
    float scale = 1.0f / sqrtf((float)hD);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (int w = 0; w < Nwin; w++) {
        float *scores = (float *)malloc((size_t)WL * WL * sizeof(float));
        for (int h = 0; h < nH; h++) {
            for (int i = 0; i < WL; i++) {
                const float *Qi = qkv + ((size_t)(w * WL + i) * 3 + 0) * D
                                      + h * hD;
                float *row = scores + i * WL;
                for (int j = 0; j < WL; j++) {
                    const float *Kj = qkv + ((size_t)(w * WL + j) * 3 + 1) * D
                                          + h * hD;
                    float s = 0.0f;
                    for (int k = 0; k < hD; k++) s += Qi[k] * Kj[k];
                    row[j] = s * scale;
                }
                scf_softmax_row(row, WL);
            }
            for (int i = 0; i < WL; i++) {
                float *Oi = attn + (size_t)(w * WL + i) * D + h * hD;
                for (int k = 0; k < hD; k++) Oi[k] = 0.0f;
                for (int j = 0; j < WL; j++) {
                    float s = scores[i * WL + j];
                    const float *Vj = qkv + ((size_t)(w * WL + j) * 3 + 2) * D
                                          + h * hD;
                    for (int k = 0; k < hD; k++) Oi[k] += s * Vj[k];
                }
            }
        }
        free(scores);
    }

    scf_gemm(proj, &m->attn_proj_w, &m->attn_proj_b, attn,
             BN, D, D, n_threads);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < BN * D; i++) toks[i] += proj[i];

    scf_layernorm_batch(ln_buf, toks, &m->ln2_w, &m->ln2_b, BN, D, m->ln_eps);
    scf_gemm(ffn, &m->mlp_fc1_w, &m->mlp_fc1_b, ln_buf,
             BN, m->ffn_hidden, D, n_threads);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < BN * m->ffn_hidden; i++) {
        /* exact GELU (timm default nn.GELU, not tanh-approx). */
        float v = ffn[i];
        ffn[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    }
    scf_gemm(proj, &m->mlp_fc2_w, &m->mlp_fc2_b, ffn,
             BN, D, m->ffn_hidden, n_threads);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < BN * D; i++) toks[i] += proj[i];

    free(ln_buf); free(qkv); free(attn); free(proj); free(ffn);

    /* ─── Step 5: keep CLS per window → (Nwin, D). ─── */
    float *win_cls = (float *)malloc((size_t)Nwin * D * sizeof(float));
    for (int w = 0; w < Nwin; w++) {
        memcpy(win_cls + (size_t)w * D,
               toks   + (size_t)w * WL * D,
               (size_t)D * sizeof(float));
    }
    free(toks);

    /* ─── Step 6: add pos_embed. sam3d always runs at input_size 256 →
     *     pos_embed is already (1, D, 32, 32) matching the window grid;
     *     bilinear resize only needed if input_size were different. ─── */
    float *pe = qt_dequant(&m->pos_embed);     /* CHW: [D, Np, Np] */
    for (int w = 0; w < Nwin; w++) {
        int wy = w / Np, wx = w % Np;
        float *dst = win_cls + (size_t)w * D;
        for (int d = 0; d < D; d++) {
            dst[d] += pe[(size_t)d * Np * Np + wy * Np + wx];
        }
    }
    free(pe);

    return win_cls;
}

/* ---- EmbedderFuser ------------------------------------------------------- */

sam3d_fuser_model *sam3d_fuser_load_safetensors(const char *st_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) {
        fprintf(stderr, "sam3d_fuser_load: open %s failed\n", st_path);
        return NULL;
    }
    sam3d_fuser_model *m = (sam3d_fuser_model *)calloc(1, sizeof(*m));
    m->st_ctx = st;

    int n_mod = 0;
    for (int i = 0; i < 16; i++) {
        char k[64]; snprintf(k, sizeof(k), "%d.0.weight", i);
        if (safetensors_find(st, k) >= 0) n_mod = i + 1; else break;
    }
    if (n_mod == 0) {
        fprintf(stderr, "sam3d_fuser_load: no projection entries found\n");
        sam3d_fuser_free(m);
        return NULL;
    }
    m->n_modalities = n_mod;
    m->projs = (sam3d_fuser_projection *)calloc((size_t)n_mod, sizeof(*m->projs));

    for (int i = 0; i < n_mod; i++) {
        sam3d_fuser_projection *p = &m->projs[i];
        char k[64];
        snprintf(k, sizeof(k), "%d.0.weight",    i); p->ln_w = scf_find(st, k);
        snprintf(k, sizeof(k), "%d.0.bias",      i); p->ln_b = scf_find(st, k);
        snprintf(k, sizeof(k), "%d.1.w1.weight", i); p->w1   = scf_find(st, k);
        snprintf(k, sizeof(k), "%d.1.w2.weight", i); p->w2   = scf_find(st, k);
        snprintf(k, sizeof(k), "%d.1.w3.weight", i); p->w3   = scf_find(st, k);
        if (!p->ln_w.data || !p->w1.data || !p->w2.data || !p->w3.data) {
            fprintf(stderr,
                    "sam3d_fuser_load: modality %d missing tensors\n", i);
            sam3d_fuser_free(m);
            return NULL;
        }
        p->embed_dim_in  = (int)p->ln_w.dims[0];
        p->ffn_hidden    = (int)p->w1.dims[0];
        p->embed_dim_out = (int)p->w2.dims[0];
    }
    m->embed_dim_out = m->projs[0].embed_dim_out;

    m->idx_emb = scf_find(st, "idx_emb");
    if (m->idx_emb.data) {
        m->idx_emb_f32 = qt_dequant(&m->idx_emb);
    } else {
        fprintf(stderr,
                "sam3d_fuser_load: idx_emb missing — positional embeddings "
                "will not be applied; rerun convert_ckpt.py to include it\n");
    }

    fprintf(stderr, "sam3d_fuser: n_modalities=%d D_out=%d ffn_h=%d\n",
            n_mod, m->embed_dim_out, m->projs[0].ffn_hidden);
    return m;
}

void sam3d_fuser_free(sam3d_fuser_model *m) {
    if (!m) return;
    if (m->st_ctx) safetensors_close(m->st_ctx);
    free(m->projs);
    free(m->idx_emb_f32);
    free(m);
}

float *sam3d_fuser_project(const sam3d_fuser_model *m, int modality_idx,
                           const float *in, int n_tokens, int n_threads) {
    if (modality_idx < 0 || modality_idx >= m->n_modalities) return NULL;
    const sam3d_fuser_projection *p = &m->projs[modality_idx];
    int Di = p->embed_dim_in;
    int Dh = p->ffn_hidden;
    int Do = p->embed_dim_out;
    if (n_threads < 1) n_threads = 1;

    float *ln = (float *)malloc((size_t)n_tokens * Di * sizeof(float));
    scf_layernorm_batch(ln, in, &p->ln_w, &p->ln_b, n_tokens, Di, 1e-5f);

    float *h1 = (float *)malloc((size_t)n_tokens * Dh * sizeof(float));
    float *h3 = (float *)malloc((size_t)n_tokens * Dh * sizeof(float));
    scf_gemm(h1, &p->w1, NULL, ln, n_tokens, Dh, Di, n_threads);
    scf_gemm(h3, &p->w3, NULL, ln, n_tokens, Dh, Di, n_threads);
    free(ln);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n_tokens * Dh; i++) {
        float v = h1[i];
        float silu = v / (1.0f + expf(-v));
        h1[i] = silu * h3[i];
    }
    free(h3);

    float *out = (float *)malloc((size_t)n_tokens * Do * sizeof(float));
    scf_gemm(out, &p->w2, NULL, h1, n_tokens, Do, Dh, n_threads);
    free(h1);
    return out;
}

void sam3d_fuser_add_pos(const sam3d_fuser_model *m, int pos_idx,
                         float *tokens, int n_tokens) {
    if (!m->idx_emb_f32) return;
    int Do = m->embed_dim_out;
    int n_pos = (int)m->idx_emb.dims[0];
    if (pos_idx < 0 || pos_idx >= n_pos) return;
    const float *row = m->idx_emb_f32 + (size_t)pos_idx * Do;
    for (int t = 0; t < n_tokens; t++) {
        float *tt = tokens + (size_t)t * Do;
        for (int d = 0; d < Do; d++) tt[d] += row[d];
    }
}

#endif /* SAM3D_COND_FUSER_IMPLEMENTATION */
#endif /* SAM3D_COND_FUSER_H */
