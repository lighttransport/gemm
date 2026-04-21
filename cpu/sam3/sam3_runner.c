/* SAM 3 CPU runner — preprocess + patch_embed + pos_embed + pre_norm + ViT.
 *
 * Weights loaded from facebook/sam3's `sam3.model.safetensors` (HF Sam3Model
 * schema). Keys consumed are prefixed `detector_model.`; we strip that prefix
 * internally so the remaining names match HF Sam3Model.state_dict() exactly.
 *
 * Verified per-stage against ref/sam3/gen_image_ref.py dumps.
 */
#include "sam3_runner.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"
#include <stddef.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define CPU_COMPUTE_IMPLEMENTATION
#include "cpu_compute.h"

#define SAM3_IMAGE_SIZE   1008
#define SAM3_PATCH        14
#define SAM3_EMBED_DIM    1024
#define SAM3_HEADS        16
#define SAM3_HEAD_DIM     64
#define SAM3_MLP_DIM      4736
#define SAM3_N_BLOCKS     32
#define SAM3_WIN          24
#define SAM3_PRETRAIN_SZ  24    /* learned pos_embed is 24x24 */
#define SAM3_ROPE_THETA   10000.0f
#define SAM3_LN_EPS       1e-6f

static const int sam3_global_layers[4] = {7, 15, 23, 31};
static int sam3_is_global(int bi) {
    for (int i = 0; i < 4; i++) if (sam3_global_layers[i] == bi) return 1;
    return 0;
}

/* ---- F32 → F16 conversion (truncation; same policy as ppd_f32_to_f16). ---- */
static uint16_t sam3_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7bff);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static uint16_t *sam3_alloc_f16(const float *src, size_t n) {
    uint16_t *dst = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!dst) return NULL;
    for (size_t i = 0; i < n; i++) dst[i] = sam3_f32_to_f16(src[i]);
    return dst;
}

#define SAM3_FPN_DIM     256
#define SAM3_FPN_N_LEV   4

#define SAM3_TEXT_DIM    1024
#define SAM3_TEXT_HEADS  16
#define SAM3_TEXT_HD     64
#define SAM3_TEXT_MLP    4096
#define SAM3_TEXT_LAYERS 24
#define SAM3_TEXT_CTX    32
#define SAM3_TEXT_VOCAB  49408
#define SAM3_TEXT_LN_EPS 1e-5f

#define SAM3_DETR_DIM    256
#define SAM3_DETR_HEADS  8
#define SAM3_DETR_HD     32
#define SAM3_DETR_MLP    2048
#define SAM3_DETR_LAYERS 6
#define SAM3_DETR_LN_EPS 1e-6f

#define SAM3_DETR_QUERIES  200
#define SAM3_DETR_QPLUS1   201             /* presence token + queries */
#define SAM3_DETR_PRESENCE_CLAMP 10.0f

typedef struct {
    float scale;                       /* 4, 2, 1, 0.5 */
    int   intermediate_c;              /* after scale_layers */
    /* scale_layers (optional): up to 2 ConvTranspose2d k=2,s=2. */
    const float *ct0_w, *ct0_b;        /* (in_c, out_c, 2, 2), (out_c) or NULL */
    int ct0_in, ct0_out;
    const float *ct1_w, *ct1_b;
    int ct1_in, ct1_out;
    int has_gelu_between;              /* scale=4 has GELU between CT0 and CT1 */
    int has_maxpool;                   /* scale=0.5 */
    /* proj1: Conv 1x1 (intermediate_c → fpn_dim) */
    const float *p1_w, *p1_b;
    /* proj2: Conv 3x3 pad1 (fpn_dim → fpn_dim) */
    const float *p2_w, *p2_b;
    /* Output size. */
    int out_h, out_w;
} sam3_fpn_layer;

typedef struct {
    /* mmap'd F32 (norms + biases). */
    const float *norm1_w, *norm1_b;
    const float *norm2_w, *norm2_b;
    /* Converted F16 weight matrices + concatenated F32 biases. */
    uint16_t *qkv_w;                /* (3*D, D), layout [Q; K; V]. */
    float    *qkv_b;                /* (3*D)   , layout [bq; bk; bv]. */
    uint16_t *o_w;                  /* (D, D)                         */
    const float *o_b;               /* (D)                             */
    uint16_t *fc1_w;                /* (MLP, D)                        */
    const float *fc1_b;             /* (MLP)                           */
    uint16_t *fc2_w;                /* (D, MLP)                        */
    const float *fc2_b;             /* (D)                             */
    int is_global;
} sam3_vit_block;

struct sam3_ctx {
    sam3_config cfg;
    st_context *st;

    /* Stage 1 weights. */
    const float *patch_proj;        /* (1024, 3, 14, 14) F32 */
    const float *pos_embed;         /* (1, 576, 1024) F32 */

    /* Pre-block norm. */
    const float *pre_norm_w, *pre_norm_b;

    /* 32 ViT blocks. */
    sam3_vit_block blocks[SAM3_N_BLOCKS];

    /* FPN neck (4 levels). */
    sam3_fpn_layer fpn[SAM3_FPN_N_LEV];
    float *fpn_out[SAM3_FPN_N_LEV];    /* (256, H, W) each, malloc'd */

    /* CLIP text encoder weights. */
    const float *text_tok_embed;       /* (49408, 1024) F32 */
    const float *text_pos_embed;       /* (32, 1024) F32 */
    const float *text_final_ln_w;
    const float *text_final_ln_b;
    sam3_vit_block text_blocks[SAM3_TEXT_LAYERS];
    /* Text buffers. */
    float *text_tok_a;                 /* (32, 1024) */
    float *text_tok_b;                 /* (32, 1024) */
    float *text_qkv;                   /* (32, 3*1024) */
    float *text_mlp;                   /* (32, 4096) */
    float *text_attn_scores;           /* (HEADS, 32, 32) */
    int32_t text_input_ids[SAM3_TEXT_CTX];
    int32_t text_attn_mask[SAM3_TEXT_CTX];
    int text_ids_ready;
    int text_done;

    /* Precomputed RoPE cos/sin tables, layout (seq, head_dim) F32.
     * rope_win: seq = 24*24 = 576 (one window, scale=1).
     * rope_glb: seq = 72*72 = 5184 (whole grid, scale=1/3). */
    float *rope_win_cos, *rope_win_sin;
    float *rope_glb_cos, *rope_glb_sin;

    /* Buffers. */
    float *pixel_values;            /* (3, 1008, 1008) CHW fp32 */
    float *tok_a;                   /* (n_tok, D) — ping */
    float *tok_b;                   /* (n_tok, D) — pong */
    float *qkv_buf;                 /* (n_tok, 3*D) */
    float *attn_buf;                /* (n_tok, D) */
    float *mlp_buf;                 /* (n_tok, MLP) */

    int n_tok;                      /* 72*72 = 5184 */
    int grid;                       /* 72 */
    int embed_ready;                /* patch+pos done */
    int vit_started;                /* pre_norm applied at least once */
    int vit_last_block;             /* last block index run so far (-1 if none) */

    /* Text → DETR projection (1024 → 256). */
    uint16_t *text_proj_w;               /* (256, 1024) F16 */
    const float *text_proj_b;            /* (256) */

    /* DETR encoder layers. sam3_vit_block reused for self-attn + MLP + 2 LNs.
     * Cross-attn (different K/V source) stored separately; norm3 is LN
     * before the MLP. */
    sam3_vit_block detr_enc[SAM3_DETR_LAYERS];
    uint16_t *detr_enc_cq_w[SAM3_DETR_LAYERS];
    uint16_t *detr_enc_ck_w[SAM3_DETR_LAYERS];
    uint16_t *detr_enc_cv_w[SAM3_DETR_LAYERS];
    uint16_t *detr_enc_co_w[SAM3_DETR_LAYERS];
    const float *detr_enc_cq_b[SAM3_DETR_LAYERS];
    const float *detr_enc_ck_b[SAM3_DETR_LAYERS];
    const float *detr_enc_cv_b[SAM3_DETR_LAYERS];
    const float *detr_enc_co_b[SAM3_DETR_LAYERS];
    const float *detr_enc_norm3_w[SAM3_DETR_LAYERS];
    const float *detr_enc_norm3_b[SAM3_DETR_LAYERS];

    /* DETR buffers. */
    float *detr_vis;        /* (5184, 256) running features */
    float *detr_scratch;    /* (5184, 256) */
    float *detr_q;          /* (5184, 256) */
    float *detr_k;          /* (5184, 256) (reused for 32) */
    float *detr_v;          /* (5184, 256) */
    float *detr_mlp;        /* (5184, 2048) */
    float *detr_text_pooled;/* (32, 256) projected text (post-projection) */
    float *detr_pos;        /* (5184, 256) sine pos */
    int detr_done;

    /* DETR decoder: all weights in F16 (GEMM) + F32 bias/norm pointers. */
    const float *dd_query_embed;           /* (200, 256) F32 */
    const float *dd_ref_points;            /* (200, 4)  F32 */
    const float *dd_presence_token;        /* (1, 256)  F32 */
    const float *dd_output_ln_w, *dd_output_ln_b;
    const float *dd_presence_ln_w, *dd_presence_ln_b;
    /* box_head: 256→256→256→4 with ReLU between. */
    uint16_t *dd_box_w[3];  const float *dd_box_b[3];
    int dd_box_out[3];
    /* presence_head: 256→256→256→1. */
    uint16_t *dd_pres_w[3]; const float *dd_pres_b[3];
    int dd_pres_out[3];
    /* ref_point_head: 512→256→256. */
    uint16_t *dd_rph_w[2];  const float *dd_rph_b[2];
    int dd_rph_out[2];
    /* box_rpb_embed_{x,y}: 2→256→8. */
    uint16_t *dd_rpbx_w[2]; const float *dd_rpbx_b[2]; int dd_rpbx_out[2];
    uint16_t *dd_rpby_w[2]; const float *dd_rpby_b[2]; int dd_rpby_out[2];
    /* 6 decoder layers: store all 4 attention blocks + 4 LNs + MLP. */
    struct {
        uint16_t *sa_q, *sa_k, *sa_v, *sa_o;
        const float *sa_qb, *sa_kb, *sa_vb, *sa_ob;
        const float *sa_ln_w, *sa_ln_b;
        uint16_t *ta_q, *ta_k, *ta_v, *ta_o;
        const float *ta_qb, *ta_kb, *ta_vb, *ta_ob;
        const float *ta_ln_w, *ta_ln_b;
        uint16_t *va_q, *va_k, *va_v, *va_o;
        const float *va_qb, *va_kb, *va_vb, *va_ob;
        const float *va_ln_w, *va_ln_b;
        uint16_t *fc1, *fc2;
        const float *fc1_b, *fc2_b;
        const float *mlp_ln_w, *mlp_ln_b;
    } dd[SAM3_DETR_LAYERS];

    /* Decoder buffers. */
    float *dd_hs;              /* (201, 256) hidden states (presence + queries) */
    float *dd_hs_in;           /* (201, 256) scratch for Q/K precompute */
    float *dd_scratch;         /* (201, 256) */
    float *dd_q;               /* (201, 256) */
    float *dd_k;               /* (max(201, 5184, 32), 256) */
    float *dd_v;               /* same */
    float *dd_mlp;             /* (201, 2048) */
    float *dd_qpos;            /* (201, 256) query pos (row 0 = 0 for presence) */
    float *dd_ref_boxes;       /* (200, 4) current reference boxes (sigmoid space) */
    float *dd_rpb;             /* (8, 201, 5184) additive vision cross-attn bias */
    float *dd_sine_box;        /* (200, 1024) = 200 * (4 * 256) */
    float *dd_pred_boxes;      /* (200, 4) final refined boxes (last layer) */
    float  dd_presence_logits[SAM3_DETR_LAYERS]; /* per-layer presence logit (batch=1) */
    float *dd_inter;           /* (6, 200, 256) per-layer output_ln(query_hidden) */
    int dd_done;

    /* Dot-product scoring weights + outputs. */
    uint16_t *ds_tmlp_w[2];    /* 2-layer MLP: (2048,256), (256,2048) F16 */
    const float *ds_tmlp_b[2]; /* biases */
    const float *ds_tnorm_w;   /* (256,) */
    const float *ds_tnorm_b;
    uint16_t *ds_tproj_w;      /* (256, 256) */
    const float *ds_tproj_b;
    uint16_t *ds_qproj_w;      /* (256, 256) */
    const float *ds_qproj_b;
    float *ds_scores;          /* (6, 200) */
    int    ds_done;

    /* Mask decoder weights. */
    uint16_t *md_pca_q_w, *md_pca_k_w, *md_pca_v_w, *md_pca_o_w;
    const float *md_pca_q_b, *md_pca_k_b, *md_pca_v_b, *md_pca_o_b;
    const float *md_pca_ln_w, *md_pca_ln_b;
    const float *md_conv_w[3];    /* Conv2d 256→256 k=3 weights (F32) */
    const float *md_conv_b[3];
    const float *md_gn_w[3];      /* GroupNorm(8,256) */
    const float *md_gn_b[3];
    uint16_t *md_me_w[3];         /* mask_embedder Linear 256→256 F16 */
    const float *md_me_b[3];
    const float *md_ip_w;         /* instance_projection Conv2d 256→256 k=1 F32 */
    const float *md_ip_b;
    const float *md_sp_w;         /* semantic_projection Conv2d 256→1 k=1 F32 */
    const float *md_sp_b;

    /* Mask decoder outputs. */
    float *md_enc_mod;       /* (5184, 256) encoder after prompt_cross_attn */
    float *md_pixel;         /* (256, 288, 288) pixel decoder output */
    float *md_instance;      /* (256, 288, 288) instance_projection */
    float *md_mask_emb;      /* (200, 256) */
    float *md_pred_masks;    /* (200, 288, 288) */
    float *md_semantic;      /* (1, 288, 288) */
    int    md_done;

    /* Post-processing (instance segmentation). */
    int     pp_n_kept;
    int     pp_th, pp_tw;
    float  *pp_scores;      /* (pp_n_kept,) */
    float  *pp_boxes;       /* (pp_n_kept, 4) xyxy scaled to target size */
    uint8_t *pp_masks;      /* (pp_n_kept, pp_th, pp_tw) binary */
    int     pp_done;
};

/* ---- Safetensors helper with `detector_model.` prefix strip. ---- */
static const void *sam3_get(const st_context *st, const char *short_name,
                            const char *expect_dtype, size_t expect_bytes)
{
    char key[256];
    snprintf(key, sizeof(key), "detector_model.%s", short_name);
    int i = safetensors_find(st, key);
    if (i < 0) {
        fprintf(stderr, "sam3: missing tensor '%s'\n", key); return NULL;
    }
    if (expect_dtype && strcmp(safetensors_dtype(st, i), expect_dtype) != 0) {
        fprintf(stderr, "sam3: '%s' dtype=%s (want %s)\n",
                key, safetensors_dtype(st, i), expect_dtype); return NULL;
    }
    if (expect_bytes && safetensors_nbytes(st, i) != expect_bytes) {
        fprintf(stderr, "sam3: '%s' nbytes=%zu (want %zu)\n",
                key, safetensors_nbytes(st, i), expect_bytes); return NULL;
    }
    return safetensors_data(st, i);
}

/* ---- RoPE table: precompute (seq, 64) cos and sin for 2D axial RoPE with
 * pairwise rotation (repeat_interleave(2)). Matches Sam3ViTRotaryEmbedding. */
static void sam3_build_rope(float *cos_out, float *sin_out,
                            int end_x, int end_y, float scale)
{
    const int dim = SAM3_HEAD_DIM;          /* 64 */
    const int quarter = dim / 4;             /* 16 */
    float freqs[16];
    for (int j = 0; j < quarter; j++) {
        /* freqs[j] = 1 / theta^((4*j)/dim) */
        freqs[j] = 1.0f / powf(SAM3_ROPE_THETA, (float)(4 * j) / (float)dim);
    }
    const int seq = end_x * end_y;
    for (int t = 0; t < seq; t++) {
        float px = (float)(t % end_x) * scale;
        float py = (float)(t / end_x) * scale;
        /* Pattern per token (len 64):
         *   [fx0,fx0, fx1,fx1, ..., fx15,fx15, fy0,fy0, ..., fy15,fy15]
         * and we take element-wise cos/sin. */
        float *c = cos_out + (size_t)t * dim;
        float *s = sin_out + (size_t)t * dim;
        for (int j = 0; j < quarter; j++) {
            float thx = px * freqs[j];
            float cx = cosf(thx), sx = sinf(thx);
            c[2 * j] = cx;     c[2 * j + 1] = cx;
            s[2 * j] = sx;     s[2 * j + 1] = sx;
        }
        for (int j = 0; j < quarter; j++) {
            float thy = py * freqs[j];
            float cy = cosf(thy), sy = sinf(thy);
            int off = 2 * quarter + 2 * j;  /* = 32 + 2j */
            c[off] = cy;       c[off + 1] = cy;
            s[off] = sy;       s[off + 1] = sy;
        }
    }
}

/* Apply pairwise-rotation RoPE to a (n_tok, n_heads * head_dim) buffer.
 * For each pair (i, i+1) of a head vector:
 *   x'[i]   = x[i]   * cos[i] - x[i+1] * sin[i]
 *   x'[i+1] = x[i+1] * cos[i] + x[i]   * sin[i]
 * Since cos/sin are repeat_interleaved, cos[i]==cos[i+1] and sin[i]==sin[i+1];
 * we just read cos[2p], sin[2p] for pair p. The per-token stride through
 * `buf` is `stride` (3*D for fused QKV, D for plain). The rope table is
 * indexed by tok_idx (position within the RoPE grid of size rope_seq). */
static void sam3_apply_rope_inplace(float *buf,
                                    const float *cos_tbl, const float *sin_tbl,
                                    int n_tok, int stride,
                                    const int *tok_to_rope_idx)
{
    const int hd = SAM3_HEAD_DIM, heads = SAM3_HEADS;
    const int pairs = hd / 2;
    for (int t = 0; t < n_tok; t++) {
        int ri = tok_to_rope_idx ? tok_to_rope_idx[t] : t;
        const float *c = cos_tbl + (size_t)ri * hd;
        const float *s = sin_tbl + (size_t)ri * hd;
        float *v = buf + (size_t)t * stride;
        for (int h = 0; h < heads; h++) {
            float *hv = v + h * hd;
            for (int p = 0; p < pairs; p++) {
                float cc = c[2 * p], ss = s[2 * p];
                float a = hv[2 * p], b = hv[2 * p + 1];
                hv[2 * p]     = a * cc - b * ss;
                hv[2 * p + 1] = b * cc + a * ss;
            }
        }
    }
}

/* ---- Preprocess: HWC uint8 RGB → CHW f32 at image_size×image_size. ---- */
static void sam3_preprocess(const uint8_t *rgb, int src_h, int src_w,
                            float *out_chw, int dst)
{
    uint8_t *resized = (uint8_t *)malloc((size_t)dst * dst * 3);
    stbir_resize_uint8_linear(rgb, src_w, src_h, 0,
                              resized, dst, dst, 0, STBIR_RGB);
    const int n = dst * dst;
    const float inv = 1.0f / 127.5f;
    for (int i = 0; i < n; i++) {
        out_chw[0 * n + i] = (float)resized[i * 3 + 0] * inv - 1.0f;
        out_chw[1 * n + i] = (float)resized[i * 3 + 1] * inv - 1.0f;
        out_chw[2 * n + i] = (float)resized[i * 3 + 2] * inv - 1.0f;
    }
    free(resized);
}

/* ---- Patch embed: Conv2d(3,1024,k=14,s=14,bias=False). ---- */
static void sam3_patch_embed(float *dst_n1024, const float *img_chw,
                             const float *W, int grid, int img_size)
{
    const int D = SAM3_EMBED_DIM;
    const int Ksq = SAM3_PATCH * SAM3_PATCH;
    const int Cksq = 3 * Ksq;
    const int HW = img_size * img_size;

    for (int py = 0; py < grid; py++) {
        for (int px = 0; px < grid; px++) {
            float *out = dst_n1024 + ((size_t)py * grid + px) * D;
            float patch[3 * SAM3_PATCH * SAM3_PATCH];
            int idx = 0;
            for (int c = 0; c < 3; c++) {
                const float *ch = img_chw + (size_t)c * HW;
                for (int ky = 0; ky < SAM3_PATCH; ky++) {
                    const float *row = ch + (py * SAM3_PATCH + ky) * img_size
                                        + px * SAM3_PATCH;
                    for (int kx = 0; kx < SAM3_PATCH; kx++)
                        patch[idx++] = row[kx];
                }
            }
            for (int o = 0; o < D; o++) {
                const float *w = W + (size_t)o * Cksq;
                float acc = 0.0f;
                for (int k = 0; k < Cksq; k++) acc += w[k] * patch[k];
                out[o] = acc;
            }
        }
    }
}

/* ---- Tile 24x24 pos_embed 3x3 into 72x72 grid, added in-place. ---- */
static void sam3_add_pos_embed(float *tokens_n1024,
                               const float *pos_24x24x1024, int grid)
{
    const int D = SAM3_EMBED_DIM;
    const int P = SAM3_PRETRAIN_SZ;
    for (int py = 0; py < grid; py++) {
        int sy = py % P;
        for (int px = 0; px < grid; px++) {
            int sx = px % P;
            float       *t = tokens_n1024 + ((size_t)py * grid + px) * D;
            const float *s = pos_24x24x1024 + ((size_t)sy * P + sx) * D;
            for (int d = 0; d < D; d++) t[d] += s[d];
        }
    }
}

/* ---- Block loader. ---- */
static int sam3_load_block(sam3_ctx *ctx, int bi)
{
    sam3_vit_block *b = &ctx->blocks[bi];
    b->is_global = sam3_is_global(bi);
    const int D = SAM3_EMBED_DIM;
    const int MLP = SAM3_MLP_DIM;

    char key[256];
    #define GETF32(short_name, nbytes) \
        (const float *)sam3_get(ctx->st, (short_name), "F32", (nbytes))

    #define BK(buf, suf) \
        (snprintf(key, sizeof(key), \
          "vision_encoder.backbone.layers.%d." suf, bi), key)

    b->norm1_w = GETF32(BK(b, "layer_norm1.weight"), D * 4);
    b->norm1_b = GETF32(BK(b, "layer_norm1.bias"),   D * 4);
    b->norm2_w = GETF32(BK(b, "layer_norm2.weight"), D * 4);
    b->norm2_b = GETF32(BK(b, "layer_norm2.bias"),   D * 4);
    if (!b->norm1_w || !b->norm1_b || !b->norm2_w || !b->norm2_b) return -1;

    const float *qw = GETF32(BK(b, "attention.q_proj.weight"), (size_t)D * D * 4);
    const float *kw = GETF32(BK(b, "attention.k_proj.weight"), (size_t)D * D * 4);
    const float *vw = GETF32(BK(b, "attention.v_proj.weight"), (size_t)D * D * 4);
    const float *qb = GETF32(BK(b, "attention.q_proj.bias"), D * 4);
    const float *kb = GETF32(BK(b, "attention.k_proj.bias"), D * 4);
    const float *vb = GETF32(BK(b, "attention.v_proj.bias"), D * 4);
    const float *ow = GETF32(BK(b, "attention.o_proj.weight"), (size_t)D * D * 4);
    const float *ob = GETF32(BK(b, "attention.o_proj.bias"), D * 4);
    if (!qw || !kw || !vw || !qb || !kb || !vb || !ow || !ob) return -1;

    /* Fuse Q/K/V into one (3*D, D) F16 matrix. */
    float *fused = (float *)malloc((size_t)3 * D * D * sizeof(float));
    if (!fused) return -1;
    memcpy(fused + (size_t)0 * D * D, qw, (size_t)D * D * sizeof(float));
    memcpy(fused + (size_t)1 * D * D, kw, (size_t)D * D * sizeof(float));
    memcpy(fused + (size_t)2 * D * D, vw, (size_t)D * D * sizeof(float));
    b->qkv_w = sam3_alloc_f16(fused, (size_t)3 * D * D);
    free(fused);
    if (!b->qkv_w) return -1;

    b->qkv_b = (float *)malloc((size_t)3 * D * sizeof(float));
    if (!b->qkv_b) return -1;
    memcpy(b->qkv_b + 0 * D, qb, D * sizeof(float));
    memcpy(b->qkv_b + 1 * D, kb, D * sizeof(float));
    memcpy(b->qkv_b + 2 * D, vb, D * sizeof(float));

    b->o_w = sam3_alloc_f16(ow, (size_t)D * D);
    b->o_b = ob;
    if (!b->o_w) return -1;

    const float *w1 = GETF32(BK(b, "mlp.fc1.weight"), (size_t)MLP * D * 4);
    const float *b1 = GETF32(BK(b, "mlp.fc1.bias"), MLP * 4);
    const float *w2 = GETF32(BK(b, "mlp.fc2.weight"), (size_t)D * MLP * 4);
    const float *b2 = GETF32(BK(b, "mlp.fc2.bias"), D * 4);
    if (!w1 || !b1 || !w2 || !b2) return -1;
    b->fc1_w = sam3_alloc_f16(w1, (size_t)MLP * D);
    b->fc1_b = b1;
    b->fc2_w = sam3_alloc_f16(w2, (size_t)D * MLP);
    b->fc2_b = b2;
    if (!b->fc1_w || !b->fc2_w) return -1;

    #undef GETF32
    #undef BK
    return 0;
}

static void sam3_free_block(sam3_vit_block *b)
{
    free(b->qkv_w); free(b->qkv_b);
    free(b->o_w);
    free(b->fc1_w); free(b->fc2_w);
    memset(b, 0, sizeof(*b));
}

/* ---- Conv / upsample / pool primitives (NCHW). ---- */

/* 2D conv: kernel k×k, stride 1, pad=p. Input (Ci,H,W), output (Co,H,W)
 * when pad=(k-1)/2. Weight (Co,Ci,k,k). OpenMP over output pixels. */
static void sam3_conv2d(float *out, const float *in, const float *W,
                        const float *bias, int Ci, int Co, int H, int W_,
                        int k, int pad)
{
    const int Ho = H, Wo = W_;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int oh = 0; oh < Ho; oh++) {
        for (int ow = 0; ow < Wo; ow++) {
            for (int oc = 0; oc < Co; oc++) {
                float acc = bias ? bias[oc] : 0.0f;
                for (int kh = 0; kh < k; kh++) {
                    int ih = oh + kh - pad;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < k; kw++) {
                        int iw = ow + kw - pad;
                        if (iw < 0 || iw >= W_) continue;
                        const float *w = W + ((size_t)oc * Ci * k * k)
                                            + ((size_t)kh * k + kw);
                        for (int ic = 0; ic < Ci; ic++) {
                            acc += w[ic * k * k] *
                                   in[((size_t)ic * H + ih) * W_ + iw];
                        }
                    }
                }
                out[((size_t)oc * Ho + oh) * Wo + ow] = acc;
            }
        }
    }
}

/* ConvTranspose2d k=2, s=2, no padding. Input (Ci, Hi, Wi) →
 * Output (Co, Hi*2, Wi*2). PyTorch weight layout (Ci, Co, 2, 2). */
static void sam3_convT_k2s2(float *out, const float *in, const float *W,
                            const float *bias, int Ci, int Co, int Hi, int Wi)
{
    const int Ho = Hi * 2, Wo = Wi * 2;
    /* Zero-init (or fill with bias). */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int oc = 0; oc < Co; oc++) {
        for (int oh = 0; oh < Ho; oh++) {
            float *row = out + ((size_t)oc * Ho + oh) * Wo;
            float bv = bias ? bias[oc] : 0.0f;
            for (int ow = 0; ow < Wo; ow++) row[ow] = bv;
        }
    }
    /* Scatter-add: for each input pixel, add contributions to 2x2 output. */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int oc = 0; oc < Co; oc++) {
        for (int ih = 0; ih < Hi; ih++) {
            for (int iw = 0; iw < Wi; iw++) {
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int oh = ih * 2 + kh, ow = iw * 2 + kw;
                        float acc = 0.0f;
                        for (int ic = 0; ic < Ci; ic++) {
                            /* W shape (Ci, Co, 2, 2) */
                            float w = W[((size_t)ic * Co + oc) * 4 + kh * 2 + kw];
                            acc += w * in[((size_t)ic * Hi + ih) * Wi + iw];
                        }
                        out[((size_t)oc * Ho + oh) * Wo + ow] += acc;
                    }
                }
            }
        }
    }
}

/* MaxPool2d k=2, s=2. Input (C, H, W) → Output (C, H/2, W/2). */
static void sam3_maxpool_k2s2(float *out, const float *in,
                              int C, int H, int W_)
{
    const int Ho = H / 2, Wo = W_ / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < Ho; oh++) {
            for (int ow = 0; ow < Wo; ow++) {
                int ih = oh * 2, iw = ow * 2;
                float m = in[((size_t)c * H + ih) * W_ + iw];
                float v;
                v = in[((size_t)c * H + ih) * W_ + iw + 1]; if (v > m) m = v;
                v = in[((size_t)c * H + ih + 1) * W_ + iw]; if (v > m) m = v;
                v = in[((size_t)c * H + ih + 1) * W_ + iw + 1]; if (v > m) m = v;
                out[((size_t)c * Ho + oh) * Wo + ow] = m;
            }
        }
    }
}

/* ---- FPN loader. ---- */
static int sam3_load_fpn(sam3_ctx *ctx)
{
    const float scales[4] = {4.0f, 2.0f, 1.0f, 0.5f};
    const int grid = ctx->grid;
    const int D = SAM3_EMBED_DIM;
    char key[256];
    #define LOAD(name) \
        (const float *)sam3_get(ctx->st, key, "F32", 0)

    for (int i = 0; i < SAM3_FPN_N_LEV; i++) {
        sam3_fpn_layer *L = &ctx->fpn[i];
        memset(L, 0, sizeof(*L));
        L->scale = scales[i];
        int in_c = D;
        int inter = in_c;

        if (scales[i] == 4.0f) {
            /* Two CTs with GELU between. */
            snprintf(key, sizeof(key),
              "vision_encoder.neck.fpn_layers.%d.scale_layers.0.weight", i);
            L->ct0_w = LOAD(key);
            snprintf(key, sizeof(key),
              "vision_encoder.neck.fpn_layers.%d.scale_layers.0.bias", i);
            L->ct0_b = LOAD(key);
            L->ct0_in = in_c; L->ct0_out = in_c / 2;
            snprintf(key, sizeof(key),
              "vision_encoder.neck.fpn_layers.%d.scale_layers.2.weight", i);
            L->ct1_w = LOAD(key);
            snprintf(key, sizeof(key),
              "vision_encoder.neck.fpn_layers.%d.scale_layers.2.bias", i);
            L->ct1_b = LOAD(key);
            L->ct1_in = in_c / 2; L->ct1_out = in_c / 4;
            L->has_gelu_between = 1;
            inter = in_c / 4;
            L->out_h = grid * 4; L->out_w = grid * 4;
        } else if (scales[i] == 2.0f) {
            snprintf(key, sizeof(key),
              "vision_encoder.neck.fpn_layers.%d.scale_layers.0.weight", i);
            L->ct0_w = LOAD(key);
            snprintf(key, sizeof(key),
              "vision_encoder.neck.fpn_layers.%d.scale_layers.0.bias", i);
            L->ct0_b = LOAD(key);
            L->ct0_in = in_c; L->ct0_out = in_c / 2;
            inter = in_c / 2;
            L->out_h = grid * 2; L->out_w = grid * 2;
        } else if (scales[i] == 1.0f) {
            inter = in_c;
            L->out_h = grid; L->out_w = grid;
        } else if (scales[i] == 0.5f) {
            L->has_maxpool = 1;
            inter = in_c;
            L->out_h = grid / 2; L->out_w = grid / 2;
        }
        L->intermediate_c = inter;

        snprintf(key, sizeof(key),
          "vision_encoder.neck.fpn_layers.%d.proj1.weight", i);
        L->p1_w = LOAD(key);
        snprintf(key, sizeof(key),
          "vision_encoder.neck.fpn_layers.%d.proj1.bias", i);
        L->p1_b = LOAD(key);
        snprintf(key, sizeof(key),
          "vision_encoder.neck.fpn_layers.%d.proj2.weight", i);
        L->p2_w = LOAD(key);
        snprintf(key, sizeof(key),
          "vision_encoder.neck.fpn_layers.%d.proj2.bias", i);
        L->p2_b = LOAD(key);
        if (!L->p1_w || !L->p1_b || !L->p2_w || !L->p2_b) return -1;

        ctx->fpn_out[i] = (float *)malloc(
            (size_t)SAM3_FPN_DIM * L->out_h * L->out_w * sizeof(float));
        if (!ctx->fpn_out[i]) return -1;
    }
    #undef LOAD
    return 0;
}

static int sam3_load_text_block(sam3_ctx *ctx, int li);
static int sam3_load_detr_block(sam3_ctx *ctx, int li);
static int sam3_load_detr_dec(sam3_ctx *ctx);
static int sam3_load_dot_score(sam3_ctx *ctx);
static int sam3_load_mask_decoder(sam3_ctx *ctx);

sam3_ctx *sam3_create(const sam3_config *cfg)
{
    sam3_ctx *ctx = (sam3_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->cfg = *cfg;
    if (ctx->cfg.image_size == 0) ctx->cfg.image_size = SAM3_IMAGE_SIZE;
    if (ctx->cfg.num_threads == 0)
        ctx->cfg.num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (!cfg->ckpt_path) {
        fprintf(stderr, "sam3_create: ckpt_path required\n");
        free(ctx); return NULL;
    }
    ctx->st = safetensors_open(cfg->ckpt_path);
    if (!ctx->st) {
        fprintf(stderr, "sam3_create: cannot open %s\n", cfg->ckpt_path);
        free(ctx); return NULL;
    }

    const int D = SAM3_EMBED_DIM;
    const size_t patch_bytes = (size_t)D * 3 * SAM3_PATCH * SAM3_PATCH * 4;
    const size_t pos_bytes   = (size_t)SAM3_PRETRAIN_SZ * SAM3_PRETRAIN_SZ * D * 4;

    ctx->patch_proj = (const float *)sam3_get(
        ctx->st, "vision_encoder.backbone.embeddings.patch_embeddings.projection.weight",
        "F32", patch_bytes);
    ctx->pos_embed = (const float *)sam3_get(
        ctx->st, "vision_encoder.backbone.embeddings.position_embeddings",
        "F32", pos_bytes);
    ctx->pre_norm_w = (const float *)sam3_get(
        ctx->st, "vision_encoder.backbone.layer_norm.weight", "F32", D * 4);
    ctx->pre_norm_b = (const float *)sam3_get(
        ctx->st, "vision_encoder.backbone.layer_norm.bias", "F32", D * 4);
    if (!ctx->patch_proj || !ctx->pos_embed || !ctx->pre_norm_w || !ctx->pre_norm_b) {
        sam3_destroy(ctx); return NULL;
    }

    ctx->grid  = ctx->cfg.image_size / SAM3_PATCH;
    ctx->n_tok = ctx->grid * ctx->grid;

    /* RoPE tables. */
    const int hd = SAM3_HEAD_DIM;
    ctx->rope_win_cos = (float *)malloc((size_t)SAM3_WIN * SAM3_WIN * hd * 4);
    ctx->rope_win_sin = (float *)malloc((size_t)SAM3_WIN * SAM3_WIN * hd * 4);
    ctx->rope_glb_cos = (float *)malloc((size_t)ctx->grid * ctx->grid * hd * 4);
    ctx->rope_glb_sin = (float *)malloc((size_t)ctx->grid * ctx->grid * hd * 4);
    if (!ctx->rope_win_cos || !ctx->rope_win_sin ||
        !ctx->rope_glb_cos || !ctx->rope_glb_sin) {
        sam3_destroy(ctx); return NULL;
    }
    /* Windowed: rotary_input_size=(window,window)=24; scale = window/24 = 1. */
    sam3_build_rope(ctx->rope_win_cos, ctx->rope_win_sin,
                    SAM3_WIN, SAM3_WIN, 1.0f);
    /* Global: rotary_input_size=(grid,grid)=72; scale = window/72 = 1/3. */
    sam3_build_rope(ctx->rope_glb_cos, ctx->rope_glb_sin,
                    ctx->grid, ctx->grid,
                    (float)SAM3_WIN / (float)ctx->grid);

    /* Buffers. */
    ctx->pixel_values = (float *)malloc((size_t)3 * ctx->cfg.image_size *
                                        ctx->cfg.image_size * sizeof(float));
    ctx->tok_a = (float *)malloc((size_t)ctx->n_tok * D * sizeof(float));
    ctx->tok_b = (float *)malloc((size_t)ctx->n_tok * D * sizeof(float));
    ctx->qkv_buf = (float *)malloc((size_t)ctx->n_tok * 3 * D * sizeof(float));
    ctx->attn_buf = (float *)malloc((size_t)ctx->n_tok * D * sizeof(float));
    ctx->mlp_buf = (float *)malloc((size_t)ctx->n_tok * SAM3_MLP_DIM * sizeof(float));
    if (!ctx->pixel_values || !ctx->tok_a || !ctx->tok_b ||
        !ctx->qkv_buf || !ctx->attn_buf || !ctx->mlp_buf) {
        sam3_destroy(ctx); return NULL;
    }

    fprintf(stderr, "sam3: loading 32 ViT blocks ...\n");
    for (int bi = 0; bi < SAM3_N_BLOCKS; bi++) {
        if (sam3_load_block(ctx, bi) != 0) {
            fprintf(stderr, "sam3: failed to load block %d\n", bi);
            sam3_destroy(ctx); return NULL;
        }
    }
    fprintf(stderr, "sam3: all blocks loaded\n");
    if (sam3_load_fpn(ctx) != 0) {
        fprintf(stderr, "sam3: failed to load FPN neck\n");
        sam3_destroy(ctx); return NULL;
    }

    /* CLIP text encoder weights + buffers. */
    ctx->text_tok_embed = (const float *)sam3_get(ctx->st,
        "text_encoder.text_model.embeddings.token_embedding.weight",
        "F32", (size_t)SAM3_TEXT_VOCAB * SAM3_TEXT_DIM * 4);
    ctx->text_pos_embed = (const float *)sam3_get(ctx->st,
        "text_encoder.text_model.embeddings.position_embedding.weight",
        "F32", (size_t)SAM3_TEXT_CTX * SAM3_TEXT_DIM * 4);
    ctx->text_final_ln_w = (const float *)sam3_get(ctx->st,
        "text_encoder.text_model.final_layer_norm.weight",
        "F32", SAM3_TEXT_DIM * 4);
    ctx->text_final_ln_b = (const float *)sam3_get(ctx->st,
        "text_encoder.text_model.final_layer_norm.bias",
        "F32", SAM3_TEXT_DIM * 4);
    if (!ctx->text_tok_embed || !ctx->text_pos_embed ||
        !ctx->text_final_ln_w || !ctx->text_final_ln_b) {
        sam3_destroy(ctx); return NULL;
    }
    fprintf(stderr, "sam3: loading 24 text blocks ...\n");
    for (int li = 0; li < SAM3_TEXT_LAYERS; li++) {
        if (sam3_load_text_block(ctx, li) != 0) {
            fprintf(stderr, "sam3: failed text block %d\n", li);
            sam3_destroy(ctx); return NULL;
        }
    }
    ctx->text_tok_a = (float *)malloc((size_t)SAM3_TEXT_CTX * SAM3_TEXT_DIM * 4);
    ctx->text_tok_b = (float *)malloc((size_t)SAM3_TEXT_CTX * SAM3_TEXT_DIM * 4);
    ctx->text_qkv   = (float *)malloc((size_t)SAM3_TEXT_CTX * 3 * SAM3_TEXT_DIM * 4);
    ctx->text_mlp   = (float *)malloc((size_t)SAM3_TEXT_CTX * SAM3_TEXT_MLP * 4);
    ctx->text_attn_scores = (float *)malloc(
        (size_t)SAM3_TEXT_CTX * SAM3_TEXT_CTX * sizeof(float));
    if (!ctx->text_tok_a || !ctx->text_tok_b || !ctx->text_qkv ||
        !ctx->text_mlp || !ctx->text_attn_scores) {
        sam3_destroy(ctx); return NULL;
    }

    /* Text → DETR projection. */
    {
        const float *tpw = (const float *)sam3_get(ctx->st,
            "text_projection.weight", "F32",
            (size_t)SAM3_DETR_DIM * SAM3_TEXT_DIM * 4);
        ctx->text_proj_b = (const float *)sam3_get(ctx->st,
            "text_projection.bias", "F32", SAM3_DETR_DIM * 4);
        if (!tpw || !ctx->text_proj_b) { sam3_destroy(ctx); return NULL; }
        ctx->text_proj_w = sam3_alloc_f16(tpw,
            (size_t)SAM3_DETR_DIM * SAM3_TEXT_DIM);
        if (!ctx->text_proj_w) { sam3_destroy(ctx); return NULL; }
    }

    fprintf(stderr, "sam3: loading 6 DETR encoder layers ...\n");
    for (int li = 0; li < SAM3_DETR_LAYERS; li++) {
        if (sam3_load_detr_block(ctx, li) != 0) {
            fprintf(stderr, "sam3: failed DETR enc %d\n", li);
            sam3_destroy(ctx); return NULL;
        }
    }
    /* DETR buffers. */
    const int N = ctx->n_tok;
    ctx->detr_vis     = (float *)malloc((size_t)N * SAM3_DETR_DIM * 4);
    ctx->detr_scratch = (float *)malloc((size_t)N * SAM3_DETR_DIM * 4);
    ctx->detr_q       = (float *)malloc((size_t)N * SAM3_DETR_DIM * 4);
    ctx->detr_k       = (float *)malloc((size_t)N * SAM3_DETR_DIM * 4);
    ctx->detr_v       = (float *)malloc((size_t)N * SAM3_DETR_DIM * 4);
    ctx->detr_mlp     = (float *)malloc((size_t)N * SAM3_DETR_MLP * 4);
    ctx->detr_text_pooled = (float *)malloc(
        (size_t)SAM3_TEXT_CTX * SAM3_DETR_DIM * 4);
    ctx->detr_pos     = (float *)malloc((size_t)N * SAM3_DETR_DIM * 4);
    if (!ctx->detr_vis || !ctx->detr_scratch || !ctx->detr_q ||
        !ctx->detr_k || !ctx->detr_v || !ctx->detr_mlp ||
        !ctx->detr_text_pooled || !ctx->detr_pos) {
        sam3_destroy(ctx); return NULL;
    }

    /* DETR decoder. */
    fprintf(stderr, "sam3: loading DETR decoder ...\n");
    if (sam3_load_detr_dec(ctx) != 0) {
        fprintf(stderr, "sam3: failed DETR dec load\n");
        sam3_destroy(ctx); return NULL;
    }
    {
        const int D = SAM3_DETR_DIM;
        const int Nt = SAM3_DETR_QPLUS1;
        const int NH = SAM3_DETR_HEADS;
        const int Nq = SAM3_DETR_QUERIES;
        const int Npix = N;
        ctx->dd_hs       = (float *)malloc((size_t)Nt * D * 4);
        ctx->dd_hs_in    = (float *)malloc((size_t)Nt * D * 4);
        ctx->dd_scratch  = (float *)malloc((size_t)Nt * D * 4);
        ctx->dd_q        = (float *)malloc((size_t)Nt * D * 4);
        size_t kv_rows = (size_t)(Npix > Nt ? Npix : Nt);
        ctx->dd_k        = (float *)malloc(kv_rows * D * 4);
        ctx->dd_v        = (float *)malloc(kv_rows * D * 4);
        ctx->dd_mlp      = (float *)malloc((size_t)Nt * SAM3_DETR_MLP * 4);
        ctx->dd_qpos     = (float *)malloc((size_t)Nt * D * 4);
        ctx->dd_ref_boxes = (float *)malloc((size_t)Nq * 4 * 4);
        ctx->dd_rpb      = (float *)malloc((size_t)NH * Nt * Npix * 4);
        ctx->dd_sine_box = (float *)malloc((size_t)Nq * 4 * 128 * 4);
        ctx->dd_pred_boxes = (float *)malloc((size_t)Nq * 4 * 4);
        ctx->dd_inter = (float *)malloc(
            (size_t)SAM3_DETR_LAYERS * Nq * D * 4);
        if (!ctx->dd_hs || !ctx->dd_hs_in || !ctx->dd_scratch ||
            !ctx->dd_q || !ctx->dd_k || !ctx->dd_v || !ctx->dd_mlp ||
            !ctx->dd_qpos || !ctx->dd_ref_boxes || !ctx->dd_rpb ||
            !ctx->dd_sine_box || !ctx->dd_pred_boxes || !ctx->dd_inter) {
            sam3_destroy(ctx); return NULL;
        }
    }

    /* Dot-product scoring. */
    fprintf(stderr, "sam3: loading dot_product_scoring ...\n");
    if (sam3_load_dot_score(ctx) != 0) {
        fprintf(stderr, "sam3: failed dot_product_scoring load\n");
        sam3_destroy(ctx); return NULL;
    }
    ctx->ds_scores = (float *)malloc(
        (size_t)SAM3_DETR_LAYERS * SAM3_DETR_QUERIES * 4);
    if (!ctx->ds_scores) { sam3_destroy(ctx); return NULL; }

    /* Mask decoder. */
    fprintf(stderr, "sam3: loading mask_decoder ...\n");
    if (sam3_load_mask_decoder(ctx) != 0) {
        fprintf(stderr, "sam3: failed mask_decoder load\n");
        sam3_destroy(ctx); return NULL;
    }
    {
        const int H0 = ctx->fpn[0].out_h, W0 = ctx->fpn[0].out_w;
        ctx->md_enc_mod = (float *)malloc(
            (size_t)ctx->n_tok * SAM3_DETR_DIM * 4);
        ctx->md_pixel   = (float *)malloc(
            (size_t)SAM3_DETR_DIM * H0 * W0 * 4);
        ctx->md_instance = (float *)malloc(
            (size_t)SAM3_DETR_DIM * H0 * W0 * 4);
        ctx->md_mask_emb = (float *)malloc(
            (size_t)SAM3_DETR_QUERIES * SAM3_DETR_DIM * 4);
        ctx->md_pred_masks = (float *)malloc(
            (size_t)SAM3_DETR_QUERIES * H0 * W0 * 4);
        ctx->md_semantic = (float *)malloc((size_t)H0 * W0 * 4);
        if (!ctx->md_enc_mod || !ctx->md_pixel || !ctx->md_instance ||
            !ctx->md_mask_emb || !ctx->md_pred_masks || !ctx->md_semantic) {
            sam3_destroy(ctx); return NULL;
        }
    }

    ctx->vit_last_block = -1;
    return ctx;
}

void sam3_destroy(sam3_ctx *ctx)
{
    if (!ctx) return;
    for (int bi = 0; bi < SAM3_N_BLOCKS; bi++) sam3_free_block(&ctx->blocks[bi]);
    for (int li = 0; li < SAM3_TEXT_LAYERS; li++) sam3_free_block(&ctx->text_blocks[li]);
    free(ctx->text_tok_a); free(ctx->text_tok_b);
    free(ctx->text_qkv); free(ctx->text_mlp);
    free(ctx->text_attn_scores);
    free(ctx->text_proj_w);
    for (int li = 0; li < SAM3_DETR_LAYERS; li++) {
        sam3_free_block(&ctx->detr_enc[li]);
        free(ctx->detr_enc_cq_w[li]); free(ctx->detr_enc_ck_w[li]);
        free(ctx->detr_enc_cv_w[li]); free(ctx->detr_enc_co_w[li]);
    }
    free(ctx->detr_vis); free(ctx->detr_scratch);
    free(ctx->detr_q); free(ctx->detr_k); free(ctx->detr_v);
    free(ctx->detr_mlp); free(ctx->detr_text_pooled); free(ctx->detr_pos);
    for (int i = 0; i < 3; i++) {
        free(ctx->dd_box_w[i]); free(ctx->dd_pres_w[i]);
    }
    for (int i = 0; i < 2; i++) {
        free(ctx->dd_rph_w[i]);
        free(ctx->dd_rpbx_w[i]); free(ctx->dd_rpby_w[i]);
    }
    for (int li = 0; li < SAM3_DETR_LAYERS; li++) {
        free(ctx->dd[li].sa_q); free(ctx->dd[li].sa_k);
        free(ctx->dd[li].sa_v); free(ctx->dd[li].sa_o);
        free(ctx->dd[li].ta_q); free(ctx->dd[li].ta_k);
        free(ctx->dd[li].ta_v); free(ctx->dd[li].ta_o);
        free(ctx->dd[li].va_q); free(ctx->dd[li].va_k);
        free(ctx->dd[li].va_v); free(ctx->dd[li].va_o);
        free(ctx->dd[li].fc1); free(ctx->dd[li].fc2);
    }
    free(ctx->dd_hs); free(ctx->dd_hs_in); free(ctx->dd_scratch);
    free(ctx->dd_q); free(ctx->dd_k); free(ctx->dd_v);
    free(ctx->dd_mlp); free(ctx->dd_qpos); free(ctx->dd_ref_boxes);
    free(ctx->dd_rpb); free(ctx->dd_sine_box); free(ctx->dd_pred_boxes);
    free(ctx->dd_inter);
    free(ctx->ds_tmlp_w[0]); free(ctx->ds_tmlp_w[1]);
    free(ctx->ds_tproj_w); free(ctx->ds_qproj_w); free(ctx->ds_scores);
    free(ctx->md_pca_q_w); free(ctx->md_pca_k_w);
    free(ctx->md_pca_v_w); free(ctx->md_pca_o_w);
    for (int i = 0; i < 3; i++) free(ctx->md_me_w[i]);
    free(ctx->md_enc_mod); free(ctx->md_pixel); free(ctx->md_instance);
    free(ctx->md_mask_emb); free(ctx->md_pred_masks); free(ctx->md_semantic);
    free(ctx->pp_scores); free(ctx->pp_boxes); free(ctx->pp_masks);
    for (int i = 0; i < SAM3_FPN_N_LEV; i++) free(ctx->fpn_out[i]);
    free(ctx->rope_win_cos); free(ctx->rope_win_sin);
    free(ctx->rope_glb_cos); free(ctx->rope_glb_sin);
    free(ctx->pixel_values);
    free(ctx->tok_a); free(ctx->tok_b);
    free(ctx->qkv_buf); free(ctx->attn_buf); free(ctx->mlp_buf);
    if (ctx->st) safetensors_close(ctx->st);
    free(ctx);
}

int sam3_set_image(sam3_ctx *ctx, const uint8_t *rgb, int h, int w)
{
    if (!ctx || !rgb || h <= 0 || w <= 0) return -1;
    sam3_preprocess(rgb, h, w, ctx->pixel_values, ctx->cfg.image_size);
    sam3_patch_embed(ctx->tok_a, ctx->pixel_values, ctx->patch_proj,
                     ctx->grid, ctx->cfg.image_size);
    sam3_add_pos_embed(ctx->tok_a, ctx->pos_embed, ctx->grid);
    ctx->embed_ready = 1;
    ctx->vit_started = 0;
    ctx->vit_last_block = -1;
    return 0;
}

int sam3_set_pixel_values(sam3_ctx *ctx, const float *pv_chw)
{
    if (!ctx || !pv_chw) return -1;
    memcpy(ctx->pixel_values, pv_chw,
           (size_t)3 * ctx->cfg.image_size * ctx->cfg.image_size * sizeof(float));
    sam3_patch_embed(ctx->tok_a, ctx->pixel_values, ctx->patch_proj,
                     ctx->grid, ctx->cfg.image_size);
    sam3_add_pos_embed(ctx->tok_a, ctx->pos_embed, ctx->grid);
    ctx->embed_ready = 1;
    ctx->vit_started = 0;
    ctx->vit_last_block = -1;
    return 0;
}

const float *sam3_get_vit_embed(const sam3_ctx *ctx, int *out_n_tok, int *out_dim)
{
    if (!ctx) return NULL;
    if (out_n_tok) *out_n_tok = ctx->n_tok;
    if (out_dim)   *out_dim   = SAM3_EMBED_DIM;
    return ctx->tok_a;
}

/* ---- Window partition/unpartition for 72x72 grid with window=24.
 *
 * Since 72 % 24 == 0, no padding. Partitioning (H, W, D) row-major into
 * (num_windows, ws*ws, D) such that window (wy, wx) contains rows [wy*ws,
 * (wy+1)*ws) × cols [wx*ws, (wx+1)*ws).
 *
 * num_windows = (H/ws) * (W/ws) = 3 * 3 = 9. Output layout per window is
 * row-major (ws rows of ws cols), i.e. the index within a window maps to
 * local (dy*ws + dx) → rope seq index. */
static void sam3_window_partition(float *dst, const float *src,
                                  int H, int W, int D, int ws)
{
    const int nwh = H / ws, nww = W / ws;
    const int ws2 = ws * ws;
    for (int wy = 0; wy < nwh; wy++) {
        for (int wx = 0; wx < nww; wx++) {
            int widx = wy * nww + wx;
            for (int dy = 0; dy < ws; dy++) {
                const float *sr = src + (((size_t)(wy * ws + dy) * W) +
                                         (size_t)wx * ws) * D;
                float *dr = dst + ((size_t)widx * ws2 + (size_t)dy * ws) * D;
                memcpy(dr, sr, (size_t)ws * D * sizeof(float));
            }
        }
    }
}

static void sam3_window_unpartition(float *dst, const float *src,
                                    int H, int W, int D, int ws)
{
    const int nwh = H / ws, nww = W / ws;
    const int ws2 = ws * ws;
    for (int wy = 0; wy < nwh; wy++) {
        for (int wx = 0; wx < nww; wx++) {
            int widx = wy * nww + wx;
            for (int dy = 0; dy < ws; dy++) {
                float *dr = dst + (((size_t)(wy * ws + dy) * W) +
                                   (size_t)wx * ws) * D;
                const float *sr = src + ((size_t)widx * ws2 + (size_t)dy * ws) * D;
                memcpy(dr, sr, (size_t)ws * D * sizeof(float));
            }
        }
    }
}

/* ---- Single ViT block forward. x is (n_tok, D), spatial layout row-major
 * over (H, W). Writes in-place. Intermediate bufs provided by ctx. */
static void sam3_vit_block_forward(sam3_ctx *ctx, int bi, float *x)
{
    const sam3_vit_block *b = &ctx->blocks[bi];
    const int D = SAM3_EMBED_DIM;
    const int MLP = SAM3_MLP_DIM;
    const int H = ctx->grid, W = ctx->grid;
    const int n_tok = ctx->n_tok;
    const int nt = ctx->cfg.num_threads;

    /* ---- Attention branch ---- */
    /* y = ln1(x)  (written to tok_b) */
    cpu_layernorm(ctx->tok_b, x, b->norm1_w, b->norm1_b, n_tok, D, SAM3_LN_EPS);

    int attn_n_tok, attn_windows, attn_stride;
    const float *rope_cos, *rope_sin;
    float *attn_in;              /* input to qkv gemm, shape (attn_n_tok*attn_windows, D) */

    if (b->is_global) {
        attn_n_tok = n_tok;      /* full 5184 */
        attn_windows = 1;
        attn_stride = n_tok;
        attn_in = ctx->tok_b;
        rope_cos = ctx->rope_glb_cos; rope_sin = ctx->rope_glb_sin;
    } else {
        /* Partition tok_b into 9 windows of 576 tokens into attn_buf (reused). */
        sam3_window_partition(ctx->attn_buf, ctx->tok_b, H, W, D, SAM3_WIN);
        attn_n_tok = SAM3_WIN * SAM3_WIN;                 /* 576 */
        attn_windows = (H / SAM3_WIN) * (W / SAM3_WIN);   /* 9  */
        attn_stride = attn_n_tok;
        attn_in = ctx->attn_buf;
        rope_cos = ctx->rope_win_cos; rope_sin = ctx->rope_win_sin;
    }

    /* qkv_gemm: y → qkv (attn_n_tok*attn_windows, 3*D). Fused cpu_gemm_f16
     * computes Y[tok * 3D + r] = dot(W[r,:], X[tok,:]); we use Y_stride=3D
     * so the layout per token is [Q(D), K(D), V(D)]. */
    const int total_tok = attn_n_tok * attn_windows;
    cpu_gemm_f16(ctx->qkv_buf, b->qkv_w, b->qkv_b, attn_in,
                 total_tok, 3 * D, D, nt);

    /* Apply 2D RoPE to Q and K only. Stride 3*D per token; offset 0 for Q,
     * offset D for K. The rope index within the window is t % attn_n_tok. */
    /* Build/reuse a simple identity mapping: rope_idx = t % attn_n_tok. */
    /* We inline the loop rather than allocating a mapping array. */
    {
        const int hd = SAM3_HEAD_DIM, heads = SAM3_HEADS;
        const int pairs = hd / 2;
        const int stride3 = 3 * D;
        for (int t = 0; t < total_tok; t++) {
            int ri = t % attn_n_tok;
            const float *c = rope_cos + (size_t)ri * hd;
            const float *s = rope_sin + (size_t)ri * hd;
            float *qv = ctx->qkv_buf + (size_t)t * stride3;       /* Q head 0 */
            float *kv = qv + D;                                    /* K head 0 */
            for (int h = 0; h < heads; h++) {
                float *qh = qv + h * hd;
                float *kh = kv + h * hd;
                for (int p = 0; p < pairs; p++) {
                    float cc = c[2 * p], ss = s[2 * p];
                    float qa = qh[2 * p], qb = qh[2 * p + 1];
                    qh[2 * p]     = qa * cc - qb * ss;
                    qh[2 * p + 1] = qb * cc + qa * ss;
                    float ka = kh[2 * p], kb = kh[2 * p + 1];
                    kh[2 * p]     = ka * cc - kb * ss;
                    kh[2 * p + 1] = kb * cc + ka * ss;
                }
            }
        }
    }

    /* Attention per window (cpu_attention expects (N, 3D) fused qkv and
     * writes (N, D) output). */
    for (int w = 0; w < attn_windows; w++) {
        float *qkv_w = ctx->qkv_buf + (size_t)w * attn_n_tok * 3 * D;
        float *out_w = ctx->attn_buf + (size_t)w * attn_n_tok * D;
        cpu_attention(out_w, qkv_w, attn_n_tok, D, SAM3_HEADS, SAM3_HEAD_DIM, nt);
    }

    /* O projection: reuse qkv_buf first D slab as scratch (need new buf —
     * use tok_b which is free now since we took its data through RoPE). */
    cpu_gemm_f16(ctx->tok_b, b->o_w, b->o_b, ctx->attn_buf,
                 total_tok, D, D, nt);

    /* Unpartition windows if needed, writing into attn_buf (reusing). Then
     * accumulate residual into x. */
    float *attn_result;
    if (b->is_global) {
        attn_result = ctx->tok_b;  /* shape (n_tok, D) already in grid order */
    } else {
        sam3_window_unpartition(ctx->attn_buf, ctx->tok_b, H, W, D, SAM3_WIN);
        attn_result = ctx->attn_buf;
    }
    for (int i = 0; i < n_tok * D; i++) x[i] += attn_result[i];

    /* ---- MLP branch ---- */
    /* y = ln2(x) into tok_b */
    cpu_layernorm(ctx->tok_b, x, b->norm2_w, b->norm2_b, n_tok, D, SAM3_LN_EPS);
    /* fc1: (n_tok, MLP) */
    cpu_gemm_f16(ctx->mlp_buf, b->fc1_w, b->fc1_b, ctx->tok_b,
                 n_tok, MLP, D, nt);
    cpu_gelu(ctx->mlp_buf, n_tok * MLP);
    /* fc2: (n_tok, D) into attn_buf (reuse) */
    cpu_gemm_f16(ctx->attn_buf, b->fc2_w, b->fc2_b, ctx->mlp_buf,
                 n_tok, D, MLP, nt);
    for (int i = 0; i < n_tok * D; i++) x[i] += ctx->attn_buf[i];
}

int sam3_run_vit(sam3_ctx *ctx, int stop_at_block)
{
    if (!ctx || !ctx->embed_ready) return -1;
    if (stop_at_block < 0 || stop_at_block >= SAM3_N_BLOCKS) return -1;

    /* Apply the pre-block LayerNorm once. Writes into tok_a in place. */
    if (!ctx->vit_started) {
        /* Copy, normalize, overwrite. */
        const int D = SAM3_EMBED_DIM;
        float *tmp = (float *)malloc((size_t)ctx->n_tok * D * sizeof(float));
        if (!tmp) return -1;
        memcpy(tmp, ctx->tok_a, (size_t)ctx->n_tok * D * sizeof(float));
        cpu_layernorm(ctx->tok_a, tmp, ctx->pre_norm_w, ctx->pre_norm_b,
                      ctx->n_tok, D, SAM3_LN_EPS);
        free(tmp);
        ctx->vit_started = 1;
    }

    int start = ctx->vit_last_block + 1;
    for (int bi = start; bi <= stop_at_block; bi++) {
        sam3_vit_block_forward(ctx, bi, ctx->tok_a);
        ctx->vit_last_block = bi;
    }
    return 0;
}

const float *sam3_get_vit_output(const sam3_ctx *ctx, int *out_n_tok, int *out_dim)
{
    if (!ctx) return NULL;
    if (out_n_tok) *out_n_tok = ctx->n_tok;
    if (out_dim)   *out_dim   = SAM3_EMBED_DIM;
    return ctx->tok_a;
}

/* ---- FPN forward. ---- */
int sam3_run_fpn(sam3_ctx *ctx)
{
    if (!ctx || ctx->vit_last_block != SAM3_N_BLOCKS - 1) {
        fprintf(stderr, "sam3_run_fpn: run ViT through block 31 first\n");
        return -1;
    }
    const int D = SAM3_EMBED_DIM;
    const int H = ctx->grid, W = ctx->grid;

    /* ViT output is (n_tok, D) row-major over (H, W). Transpose to
     * NCHW (D, H, W) once for the conv ops. */
    float *nchw = (float *)malloc((size_t)D * H * W * sizeof(float));
    if (!nchw) return -1;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const float *t = ctx->tok_a + ((size_t)y * W + x) * D;
            for (int c = 0; c < D; c++)
                nchw[((size_t)c * H + y) * W + x] = t[c];
        }
    }

    for (int li = 0; li < SAM3_FPN_N_LEV; li++) {
        sam3_fpn_layer *L = &ctx->fpn[li];
        /* Scale stage. */
        float *scaled = NULL;
        int sc_c = D, sc_h = H, sc_w = W;

        if (L->scale == 4.0f) {
            /* CT0 */
            int h1 = H * 2, w1 = W * 2;
            float *t1 = (float *)malloc((size_t)L->ct0_out * h1 * w1 * sizeof(float));
            sam3_convT_k2s2(t1, nchw, L->ct0_w, L->ct0_b, L->ct0_in, L->ct0_out, H, W);
            /* GELU */
            size_t nn = (size_t)L->ct0_out * h1 * w1;
            cpu_gelu(t1, (int)nn);
            /* CT1 */
            int h2 = h1 * 2, w2 = w1 * 2;
            float *t2 = (float *)malloc((size_t)L->ct1_out * h2 * w2 * sizeof(float));
            sam3_convT_k2s2(t2, t1, L->ct1_w, L->ct1_b, L->ct1_in, L->ct1_out, h1, w1);
            free(t1);
            scaled = t2; sc_c = L->ct1_out; sc_h = h2; sc_w = w2;
        } else if (L->scale == 2.0f) {
            int h1 = H * 2, w1 = W * 2;
            float *t1 = (float *)malloc((size_t)L->ct0_out * h1 * w1 * sizeof(float));
            sam3_convT_k2s2(t1, nchw, L->ct0_w, L->ct0_b, L->ct0_in, L->ct0_out, H, W);
            scaled = t1; sc_c = L->ct0_out; sc_h = h1; sc_w = w1;
        } else if (L->scale == 1.0f) {
            /* Use nchw directly (don't free; we need it for other levels). */
            scaled = nchw; sc_c = D; sc_h = H; sc_w = W;
        } else { /* 0.5 */
            int h1 = H / 2, w1 = W / 2;
            float *t1 = (float *)malloc((size_t)D * h1 * w1 * sizeof(float));
            sam3_maxpool_k2s2(t1, nchw, D, H, W);
            scaled = t1; sc_c = D; sc_h = h1; sc_w = w1;
        }

        /* proj1: Conv 1x1 (sc_c → 256). */
        float *p1 = (float *)malloc((size_t)SAM3_FPN_DIM * sc_h * sc_w * sizeof(float));
        sam3_conv2d(p1, scaled, L->p1_w, L->p1_b, sc_c, SAM3_FPN_DIM,
                    sc_h, sc_w, 1, 0);
        if (scaled != nchw) free(scaled);

        /* proj2: Conv 3x3 pad=1 (256 → 256). */
        sam3_conv2d(ctx->fpn_out[li], p1, L->p2_w, L->p2_b,
                    SAM3_FPN_DIM, SAM3_FPN_DIM, sc_h, sc_w, 3, 1);
        free(p1);
    }
    free(nchw);
    return 0;
}

const float *sam3_get_fpn(const sam3_ctx *ctx, int level,
                          int *out_c, int *out_h, int *out_w)
{
    if (!ctx || level < 0 || level >= SAM3_FPN_N_LEV) return NULL;
    if (out_c) *out_c = SAM3_FPN_DIM;
    if (out_h) *out_h = ctx->fpn[level].out_h;
    if (out_w) *out_w = ctx->fpn[level].out_w;
    return ctx->fpn_out[level];
}

/* ---- CLIP text encoder ----
 * Architecture: token_embed + pos_embed → 24x (LN1 → causal MHA → residual
 * → LN2 → fc1 → QuickGELU(1.702) → fc2 → residual) → final_layer_norm.
 * Weights F32 on disk, converted to F16 for cpu_gemm_f16. Attention is
 * causal-only; padding mask ignored (valid tokens still match ref for
 * positions up to first PAD). head_dim=64 → cpu_attention AVX2 path OK. */

static int sam3_load_text_block(sam3_ctx *ctx, int li)
{
    sam3_vit_block *b = &ctx->text_blocks[li];
    const int D = SAM3_TEXT_DIM;
    const int MLP = SAM3_TEXT_MLP;
    char key[256];
    #define TGETF32(short_name, nbytes) \
        (const float *)sam3_get(ctx->st, (short_name), "F32", (nbytes))
    #define TK(suf) \
        (snprintf(key, sizeof(key), \
         "text_encoder.text_model.encoder.layers.%d." suf, li), key)

    b->norm1_w = TGETF32(TK("layer_norm1.weight"), D * 4);
    b->norm1_b = TGETF32(TK("layer_norm1.bias"),   D * 4);
    b->norm2_w = TGETF32(TK("layer_norm2.weight"), D * 4);
    b->norm2_b = TGETF32(TK("layer_norm2.bias"),   D * 4);
    if (!b->norm1_w || !b->norm1_b || !b->norm2_w || !b->norm2_b) return -1;

    const float *qw = TGETF32(TK("self_attn.q_proj.weight"), (size_t)D*D*4);
    const float *kw = TGETF32(TK("self_attn.k_proj.weight"), (size_t)D*D*4);
    const float *vw = TGETF32(TK("self_attn.v_proj.weight"), (size_t)D*D*4);
    const float *qb = TGETF32(TK("self_attn.q_proj.bias"), D*4);
    const float *kb = TGETF32(TK("self_attn.k_proj.bias"), D*4);
    const float *vb = TGETF32(TK("self_attn.v_proj.bias"), D*4);
    const float *ow = TGETF32(TK("self_attn.out_proj.weight"), (size_t)D*D*4);
    const float *ob = TGETF32(TK("self_attn.out_proj.bias"), D*4);
    if (!qw || !kw || !vw || !qb || !kb || !vb || !ow || !ob) return -1;

    float *fused = (float *)malloc((size_t)3 * D * D * sizeof(float));
    if (!fused) return -1;
    memcpy(fused + (size_t)0*D*D, qw, (size_t)D*D*sizeof(float));
    memcpy(fused + (size_t)1*D*D, kw, (size_t)D*D*sizeof(float));
    memcpy(fused + (size_t)2*D*D, vw, (size_t)D*D*sizeof(float));
    b->qkv_w = sam3_alloc_f16(fused, (size_t)3*D*D);
    free(fused);
    if (!b->qkv_w) return -1;
    b->qkv_b = (float *)malloc((size_t)3 * D * sizeof(float));
    if (!b->qkv_b) return -1;
    memcpy(b->qkv_b + 0*D, qb, D*sizeof(float));
    memcpy(b->qkv_b + 1*D, kb, D*sizeof(float));
    memcpy(b->qkv_b + 2*D, vb, D*sizeof(float));

    b->o_w = sam3_alloc_f16(ow, (size_t)D*D);
    b->o_b = ob;
    if (!b->o_w) return -1;

    const float *w1 = TGETF32(TK("mlp.fc1.weight"), (size_t)MLP*D*4);
    const float *b1 = TGETF32(TK("mlp.fc1.bias"), MLP*4);
    const float *w2 = TGETF32(TK("mlp.fc2.weight"), (size_t)D*MLP*4);
    const float *b2 = TGETF32(TK("mlp.fc2.bias"), D*4);
    if (!w1 || !b1 || !w2 || !b2) return -1;
    b->fc1_w = sam3_alloc_f16(w1, (size_t)MLP*D);
    b->fc1_b = b1;
    b->fc2_w = sam3_alloc_f16(w2, (size_t)D*MLP);
    b->fc2_b = b2;
    if (!b->fc1_w || !b->fc2_w) return -1;
    #undef TGETF32
    #undef TK
    return 0;
}

static void sam3_gelu_exact(float *x, int n)
{
    /* Erf-based GELU (HF ACT2FN["gelu"]). SAM 3 text config uses hidden_act
     * "gelu" which is GELUActivation (erf), not quick_gelu. */
    const float c = 0.70710678118654752440f; /* 1/sqrt(2) */
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + erff(v * c));
    }
}

/* Causal MHA: qkv (T, 3*D), out (T, D). T small (32); scalar impl. */
static void sam3_text_causal_attn(float *out, const float *qkv,
                                  float *scores, int T)
{
    const int D = SAM3_TEXT_DIM;
    const int H = SAM3_TEXT_HEADS;
    const int HD = SAM3_TEXT_HD;
    const float scale = 1.0f / sqrtf((float)HD);
    /* Zero output. */
    for (int i = 0; i < T * D; i++) out[i] = 0.0f;

    for (int h = 0; h < H; h++) {
        /* Compute Q·Kᵀ * scale into scores[t][k] for t in [0,T), k<=t. */
        for (int t = 0; t < T; t++) {
            const float *q = qkv + (size_t)t * 3 * D + 0 * D + h * HD;
            for (int k = 0; k <= t; k++) {
                const float *kv = qkv + (size_t)k * 3 * D + 1 * D + h * HD;
                float s = 0.0f;
                for (int d = 0; d < HD; d++) s += q[d] * kv[d];
                scores[(size_t)t * T + k] = s * scale;
            }
            /* Softmax over k in [0, t]. */
            float *row = scores + (size_t)t * T;
            float mx = row[0];
            for (int k = 1; k <= t; k++) if (row[k] > mx) mx = row[k];
            float sum = 0.0f;
            for (int k = 0; k <= t; k++) { row[k] = expf(row[k] - mx); sum += row[k]; }
            float inv = 1.0f / sum;
            for (int k = 0; k <= t; k++) row[k] *= inv;
            /* out[t, h] += sum_k row[k] * V[k, h]. */
            float *o = out + (size_t)t * D + h * HD;
            for (int k = 0; k <= t; k++) {
                const float *vv = qkv + (size_t)k * 3 * D + 2 * D + h * HD;
                float a = row[k];
                for (int d = 0; d < HD; d++) o[d] += a * vv[d];
            }
        }
    }
}

static void sam3_text_block_forward(sam3_ctx *ctx, int li)
{
    const sam3_vit_block *b = &ctx->text_blocks[li];
    const int D = SAM3_TEXT_DIM;
    const int MLP = SAM3_TEXT_MLP;
    const int T = SAM3_TEXT_CTX;
    const int nt = ctx->cfg.num_threads;
    float *x = ctx->text_tok_a;

    /* LN1 → tok_b */
    cpu_layernorm(ctx->text_tok_b, x, b->norm1_w, b->norm1_b,
                  T, D, SAM3_TEXT_LN_EPS);
    /* QKV gemm → text_qkv (T, 3D) */
    cpu_gemm_f16(ctx->text_qkv, b->qkv_w, b->qkv_b, ctx->text_tok_b,
                 T, 3 * D, D, nt);
    /* Causal attention → text_tok_b (reused, T,D). */
    sam3_text_causal_attn(ctx->text_tok_b, ctx->text_qkv,
                          ctx->text_attn_scores, T);
    /* out_proj → text_mlp (reuse first T*D). */
    cpu_gemm_f16(ctx->text_mlp, b->o_w, b->o_b, ctx->text_tok_b,
                 T, D, D, nt);
    for (int i = 0; i < T * D; i++) x[i] += ctx->text_mlp[i];

    /* LN2 → tok_b */
    cpu_layernorm(ctx->text_tok_b, x, b->norm2_w, b->norm2_b,
                  T, D, SAM3_TEXT_LN_EPS);
    /* fc1 → text_mlp (T, MLP) */
    cpu_gemm_f16(ctx->text_mlp, b->fc1_w, b->fc1_b, ctx->text_tok_b,
                 T, MLP, D, nt);
    sam3_gelu_exact(ctx->text_mlp, T * MLP);
    /* fc2 → text_tok_b (T, D) */
    cpu_gemm_f16(ctx->text_tok_b, b->fc2_w, b->fc2_b, ctx->text_mlp,
                 T, D, MLP, nt);
    for (int i = 0; i < T * D; i++) x[i] += ctx->text_tok_b[i];
}

int sam3_set_input_ids(sam3_ctx *ctx, const int32_t *ids,
                       const int32_t *mask)
{
    if (!ctx || !ids) return -1;
    for (int t = 0; t < SAM3_TEXT_CTX; t++) {
        int id = ids[t];
        if (id < 0 || id >= SAM3_TEXT_VOCAB) {
            fprintf(stderr, "sam3_set_input_ids: id %d at pos %d out of range\n",
                    id, t);
            return -1;
        }
        ctx->text_input_ids[t] = id;
        ctx->text_attn_mask[t] = mask ? mask[t] : 1;
    }
    ctx->text_ids_ready = 1;
    ctx->text_done = 0;
    return 0;
}

int sam3_run_text(sam3_ctx *ctx)
{
    if (!ctx || !ctx->text_ids_ready) return -1;
    const int D = SAM3_TEXT_DIM;
    const int T = SAM3_TEXT_CTX;

    /* Embed: x[t,:] = tok_embed[ids[t], :] + pos_embed[t, :]. */
    for (int t = 0; t < T; t++) {
        const float *te = ctx->text_tok_embed +
                          (size_t)ctx->text_input_ids[t] * D;
        const float *pe = ctx->text_pos_embed + (size_t)t * D;
        float *x = ctx->text_tok_a + (size_t)t * D;
        for (int d = 0; d < D; d++) x[d] = te[d] + pe[d];
    }

    for (int li = 0; li < SAM3_TEXT_LAYERS; li++)
        sam3_text_block_forward(ctx, li);

    /* final_layer_norm (into tok_a in place via tmp). */
    float *tmp = (float *)malloc((size_t)T * D * sizeof(float));
    if (!tmp) return -1;
    memcpy(tmp, ctx->text_tok_a, (size_t)T * D * sizeof(float));
    cpu_layernorm(ctx->text_tok_a, tmp,
                  ctx->text_final_ln_w, ctx->text_final_ln_b,
                  T, D, SAM3_TEXT_LN_EPS);
    free(tmp);
    ctx->text_done = 1;
    return 0;
}

const float *sam3_get_text_output(const sam3_ctx *ctx,
                                  int *out_len, int *out_dim)
{
    if (!ctx || !ctx->text_done) return NULL;
    if (out_len) *out_len = SAM3_TEXT_CTX;
    if (out_dim) *out_dim = SAM3_TEXT_DIM;
    return ctx->text_tok_a;
}

/* ---- DETR encoder (6 layers, hidden=256, 8h×32hd, MLP=2048).
 * Input: FPN level 2 (256, 72, 72) vision features and text encoder output
 * (32, 1024) projected to (32, 256). Output: (5184, 256). */

/* Sine 2D position embedding matching Sam3SinePositionEmbedding with
 * num_pos_feats=128, normalize=True, temperature=10000. Grid (H, W),
 * mask all zero. Output layout (H*W, 2*num_pos_feats)=H*W×256 row-major,
 * per pixel: [pos_y(128), pos_x(128)] where pos_y/x = interleaved
 * sin,cos,sin,cos,... of (coord * scale) / temperature^(2*(d//2)/F). */
static void sam3_build_sine_pos(float *out, int H, int W, int num_pos_feats)
{
    const float scale = 2.0f * (float)M_PI;
    const float eps = 1e-6f;
    const float denom_y = (float)H + eps;
    const float denom_x = (float)W + eps;
    float *dim_t = (float *)malloc(num_pos_feats * sizeof(float));
    for (int d = 0; d < num_pos_feats; d++) {
        int e = 2 * (d / 2);
        dim_t[d] = powf(10000.0f, (float)e / (float)num_pos_feats);
    }
    for (int y = 0; y < H; y++) {
        float y_embed = (float)(y + 1) / denom_y * scale;
        for (int x = 0; x < W; x++) {
            float x_embed = (float)(x + 1) / denom_x * scale;
            float *row = out + ((size_t)y * W + x) * (2 * num_pos_feats);
            float *py = row;
            float *px = row + num_pos_feats;
            for (int d = 0; d < num_pos_feats; d++) {
                float ey = y_embed / dim_t[d];
                float ex = x_embed / dim_t[d];
                py[d] = (d & 1) ? cosf(ey) : sinf(ey);
                px[d] = (d & 1) ? cosf(ex) : sinf(ex);
            }
        }
    }
    free(dim_t);
}

static int sam3_load_detr_block(sam3_ctx *ctx, int li)
{
    sam3_vit_block *b = &ctx->detr_enc[li];
    const int D = SAM3_DETR_DIM;
    const int MLP = SAM3_DETR_MLP;
    char key[256];
    #define DGETF32(short_name, nbytes) \
        (const float *)sam3_get(ctx->st, (short_name), "F32", (nbytes))
    #define DK(suf) \
        (snprintf(key, sizeof(key), "detr_encoder.layers.%d." suf, li), key)

    b->norm1_w = DGETF32(DK("layer_norm1.weight"), D * 4);
    b->norm1_b = DGETF32(DK("layer_norm1.bias"),   D * 4);
    b->norm2_w = DGETF32(DK("layer_norm2.weight"), D * 4);
    b->norm2_b = DGETF32(DK("layer_norm2.bias"),   D * 4);
    ctx->detr_enc_norm3_w[li] = DGETF32(DK("layer_norm3.weight"), D * 4);
    ctx->detr_enc_norm3_b[li] = DGETF32(DK("layer_norm3.bias"),   D * 4);
    if (!b->norm1_w || !ctx->detr_enc_norm3_w[li]) return -1;

    /* Self-attn fused QKV. */
    const float *qw = DGETF32(DK("self_attn.q_proj.weight"), (size_t)D*D*4);
    const float *kw = DGETF32(DK("self_attn.k_proj.weight"), (size_t)D*D*4);
    const float *vw = DGETF32(DK("self_attn.v_proj.weight"), (size_t)D*D*4);
    const float *qb = DGETF32(DK("self_attn.q_proj.bias"), D*4);
    const float *kb = DGETF32(DK("self_attn.k_proj.bias"), D*4);
    const float *vb = DGETF32(DK("self_attn.v_proj.bias"), D*4);
    const float *ow = DGETF32(DK("self_attn.o_proj.weight"), (size_t)D*D*4);
    const float *ob = DGETF32(DK("self_attn.o_proj.bias"), D*4);
    if (!qw || !ow) return -1;
    float *fused = (float *)malloc((size_t)3*D*D*sizeof(float));
    if (!fused) return -1;
    memcpy(fused + 0*D*D, qw, (size_t)D*D*sizeof(float));
    memcpy(fused + 1*D*D, kw, (size_t)D*D*sizeof(float));
    memcpy(fused + 2*D*D, vw, (size_t)D*D*sizeof(float));
    b->qkv_w = sam3_alloc_f16(fused, (size_t)3*D*D); free(fused);
    b->qkv_b = (float *)malloc((size_t)3*D*sizeof(float));
    memcpy(b->qkv_b+0*D, qb, D*4); memcpy(b->qkv_b+1*D, kb, D*4);
    memcpy(b->qkv_b+2*D, vb, D*4);
    b->o_w = sam3_alloc_f16(ow, (size_t)D*D); b->o_b = ob;

    /* Cross-attn (separate). */
    const float *cqw = DGETF32(DK("cross_attn.q_proj.weight"), (size_t)D*D*4);
    const float *ckw = DGETF32(DK("cross_attn.k_proj.weight"), (size_t)D*D*4);
    const float *cvw = DGETF32(DK("cross_attn.v_proj.weight"), (size_t)D*D*4);
    const float *cow = DGETF32(DK("cross_attn.o_proj.weight"), (size_t)D*D*4);
    if (!cqw || !ckw || !cvw || !cow) return -1;
    ctx->detr_enc_cq_w[li] = sam3_alloc_f16(cqw, (size_t)D*D);
    ctx->detr_enc_ck_w[li] = sam3_alloc_f16(ckw, (size_t)D*D);
    ctx->detr_enc_cv_w[li] = sam3_alloc_f16(cvw, (size_t)D*D);
    ctx->detr_enc_co_w[li] = sam3_alloc_f16(cow, (size_t)D*D);
    ctx->detr_enc_cq_b[li] = DGETF32(DK("cross_attn.q_proj.bias"), D*4);
    ctx->detr_enc_ck_b[li] = DGETF32(DK("cross_attn.k_proj.bias"), D*4);
    ctx->detr_enc_cv_b[li] = DGETF32(DK("cross_attn.v_proj.bias"), D*4);
    ctx->detr_enc_co_b[li] = DGETF32(DK("cross_attn.o_proj.bias"), D*4);

    /* MLP. */
    const float *w1 = DGETF32(DK("mlp.fc1.weight"), (size_t)MLP*D*4);
    const float *b1 = DGETF32(DK("mlp.fc1.bias"), MLP*4);
    const float *w2 = DGETF32(DK("mlp.fc2.weight"), (size_t)D*MLP*4);
    const float *b2 = DGETF32(DK("mlp.fc2.bias"), D*4);
    if (!w1 || !b1 || !w2 || !b2) return -1;
    b->fc1_w = sam3_alloc_f16(w1, (size_t)MLP*D); b->fc1_b = b1;
    b->fc2_w = sam3_alloc_f16(w2, (size_t)D*MLP); b->fc2_b = b2;

    #undef DGETF32
    #undef DK
    return 0;
}

/* Multi-head attention with head_dim=32. Q (Nq, D), K (Nk, D), V (Nk, D).
 * D = heads*hd. Writes out (Nq, D). text_key_mask[Nk]: if non-NULL,
 * positions with mask==0 are masked out (scores set to -inf). Scalar +
 * OpenMP over (head, query). Uses a per-thread score buffer of size Nk. */
static void sam3_mha_detr_full(float *out, const float *Q, const float *K,
                               const float *V, int Nq, int Nk,
                               const int32_t *key_mask,
                               const float *bias_hqk)
{
    const int H = SAM3_DETR_HEADS;
    const int HD = SAM3_DETR_HD;
    const int D = SAM3_DETR_DIM;
    const float scale = 1.0f / sqrtf((float)HD);
    /* Zero out. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nq * D; i++) out[i] = 0.0f;

    #pragma omp parallel
    {
        float *scores = (float *)malloc((size_t)Nk * sizeof(float));
        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < H; h++) {
            for (int q = 0; q < Nq; q++) {
                const float *qv = Q + (size_t)q * D + h * HD;
                const float *bias_row = bias_hqk
                    ? bias_hqk + ((size_t)h * Nq + q) * Nk : NULL;
                float mx = -INFINITY;
                for (int k = 0; k < Nk; k++) {
                    const float *kv = K + (size_t)k * D + h * HD;
                    float s = 0.0f;
                    for (int d = 0; d < HD; d++) s += qv[d] * kv[d];
                    s *= scale;
                    if (bias_row) s += bias_row[k];
                    if (key_mask && !key_mask[k]) s = -INFINITY;
                    scores[k] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.0f;
                for (int k = 0; k < Nk; k++) {
                    scores[k] = expf(scores[k] - mx);
                    sum += scores[k];
                }
                float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
                float *o = out + (size_t)q * D + h * HD;
                for (int k = 0; k < Nk; k++) {
                    const float *vv = V + (size_t)k * D + h * HD;
                    float a = scores[k] * inv;
                    for (int d = 0; d < HD; d++) o[d] += a * vv[d];
                }
            }
        }
        free(scores);
    }
}

static void sam3_mha_detr(float *out, const float *Q, const float *K,
                          const float *V, int Nq, int Nk,
                          const int32_t *key_mask)
{
    sam3_mha_detr_full(out, Q, K, V, Nq, Nk, key_mask, NULL);
}

static void sam3_relu(float *x, int n)
{
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

static void sam3_detr_enc_layer(sam3_ctx *ctx, int li)
{
    const int N = ctx->n_tok;                 /* 5184 */
    const int D = SAM3_DETR_DIM;
    const int MLP = SAM3_DETR_MLP;
    const int T = SAM3_TEXT_CTX;              /* 32 */
    const int nt = ctx->cfg.num_threads;
    const sam3_vit_block *b = &ctx->detr_enc[li];
    float *x = ctx->detr_vis;

    /* Self-attn: residual = x; y = LN1(x); y_pos = y + pos_encoding.
     * Q = K = Wq(y_pos), V = Wv(y). */
    cpu_layernorm(ctx->detr_scratch, x, b->norm1_w, b->norm1_b,
                  N, D, SAM3_DETR_LN_EPS);
    /* V input is scratch (no pos). */
    /* Q/K input is scratch + pos. */
    float *qk_in = ctx->detr_q;  /* temp buffer for y_pos */
    for (int i = 0; i < N * D; i++) qk_in[i] = ctx->detr_scratch[i] + ctx->detr_pos[i];

    /* Compute fused QKV using qkv_w on qk_in — that gives Q,K with pos,
     * but V should use scratch (no pos). Easier: split into 3 gemms:
     *   Q = Wq(qk_in), K = Wk(qk_in), V = Wv(scratch). */
    /* b->qkv_w is fused (3*D, D) = [Q; K; V]. Extract each slab manually via
     * strided access: cpu_gemm_f16 uses full weight, so we run the fused
     * gemm twice — once on qk_in for Q+K (we discard the V slab computed
     * from qk_in) and once with a V-only recompute. Cheaper alternative:
     * run full fused on qk_in, then patch V with a separate gemm using
     * the V sub-weight. The V weight lives at b->qkv_w + 2*D*D. */
    cpu_gemm_f16(ctx->detr_mlp, b->qkv_w, b->qkv_b, qk_in,
                 N, 3 * D, D, nt);  /* into detr_mlp (N, 3D); uses part of buffer */
    /* Split into Q, K into detr_q/k; recompute V using V weight+bias from scratch. */
    for (int t = 0; t < N; t++) {
        memcpy(ctx->detr_q + (size_t)t * D,
               ctx->detr_mlp + (size_t)t * 3 * D + 0 * D, D * sizeof(float));
        memcpy(ctx->detr_k + (size_t)t * D,
               ctx->detr_mlp + (size_t)t * 3 * D + 1 * D, D * sizeof(float));
    }
    /* V = Wv @ scratch + bv; V weight is slab 2 of qkv_w (F16 row-major
     * (3D, D)). Its row offset into the flat F16 buffer is 2*D*D. */
    cpu_gemm_f16(ctx->detr_v, b->qkv_w + (size_t)2 * D * D,
                 b->qkv_b + 2 * D, ctx->detr_scratch,
                 N, D, D, nt);

    /* Self-attn. */
    sam3_mha_detr(ctx->detr_mlp /* reuse as out */,
                  ctx->detr_q, ctx->detr_k, ctx->detr_v, N, N, NULL);
    /* O proj → add to residual x. */
    cpu_gemm_f16(ctx->detr_scratch, b->o_w, b->o_b, ctx->detr_mlp,
                 N, D, D, nt);
    for (int i = 0; i < N * D; i++) x[i] += ctx->detr_scratch[i];

    /* Cross-attn: residual = x; y = LN2(x); Q = Wq(y); K = Wk(text);
     * V = Wv(text). attn(..., mask=text_mask). */
    cpu_layernorm(ctx->detr_scratch, x, b->norm2_w, b->norm2_b,
                  N, D, SAM3_DETR_LN_EPS);
    cpu_gemm_f16(ctx->detr_q, ctx->detr_enc_cq_w[li], ctx->detr_enc_cq_b[li],
                 ctx->detr_scratch, N, D, D, nt);
    cpu_gemm_f16(ctx->detr_k, ctx->detr_enc_ck_w[li], ctx->detr_enc_ck_b[li],
                 ctx->detr_text_pooled, T, D, D, nt);
    cpu_gemm_f16(ctx->detr_v, ctx->detr_enc_cv_w[li], ctx->detr_enc_cv_b[li],
                 ctx->detr_text_pooled, T, D, D, nt);
    sam3_mha_detr(ctx->detr_mlp, ctx->detr_q, ctx->detr_k, ctx->detr_v,
                  N, T, ctx->text_attn_mask);
    cpu_gemm_f16(ctx->detr_scratch, ctx->detr_enc_co_w[li], ctx->detr_enc_co_b[li],
                 ctx->detr_mlp, N, D, D, nt);
    for (int i = 0; i < N * D; i++) x[i] += ctx->detr_scratch[i];

    /* MLP: residual = x; y = LN3(x); fc1 → ReLU → fc2. */
    cpu_layernorm(ctx->detr_scratch, x,
                  ctx->detr_enc_norm3_w[li], ctx->detr_enc_norm3_b[li],
                  N, D, SAM3_DETR_LN_EPS);
    cpu_gemm_f16(ctx->detr_mlp, b->fc1_w, b->fc1_b, ctx->detr_scratch,
                 N, MLP, D, nt);
    sam3_relu(ctx->detr_mlp, N * MLP);
    cpu_gemm_f16(ctx->detr_scratch, b->fc2_w, b->fc2_b, ctx->detr_mlp,
                 N, D, MLP, nt);
    for (int i = 0; i < N * D; i++) x[i] += ctx->detr_scratch[i];
}

int sam3_run_detr_enc(sam3_ctx *ctx)
{
    if (!ctx) return -1;
    if (ctx->vit_last_block != SAM3_N_BLOCKS - 1 || !ctx->fpn_out[2] ||
        !ctx->text_done) {
        fprintf(stderr, "sam3_run_detr_enc: need ViT+FPN+text first\n");
        return -1;
    }
    const int N = ctx->n_tok;
    const int D = SAM3_DETR_DIM;
    const int T = SAM3_TEXT_CTX;
    const int nt = ctx->cfg.num_threads;

    /* Vision input: FPN level 2 is (256, 72, 72) NCHW — flatten to (N, 256)
     * tokens row-major over (y, x). */
    const float *src = ctx->fpn_out[2];
    for (int y = 0; y < ctx->grid; y++) {
        for (int x = 0; x < ctx->grid; x++) {
            float *dst = ctx->detr_vis + ((size_t)y * ctx->grid + x) * D;
            for (int c = 0; c < D; c++)
                dst[c] = src[((size_t)c * ctx->grid + y) * ctx->grid + x];
        }
    }

    /* Text projection 1024 → 256. */
    cpu_gemm_f16(ctx->detr_text_pooled, ctx->text_proj_w, ctx->text_proj_b,
                 ctx->text_tok_a, T, D, SAM3_TEXT_DIM, nt);

    /* Build sine pos embedding for (72, 72) at num_pos_feats=128. */
    sam3_build_sine_pos(ctx->detr_pos, ctx->grid, ctx->grid, D / 2);

    for (int li = 0; li < SAM3_DETR_LAYERS; li++) {
        sam3_detr_enc_layer(ctx, li);
    }
    ctx->detr_done = 1;
    return 0;
}

const float *sam3_get_detr_enc(const sam3_ctx *ctx, int *out_n, int *out_dim)
{
    if (!ctx || !ctx->detr_done) return NULL;
    if (out_n)   *out_n   = ctx->n_tok;
    if (out_dim) *out_dim = SAM3_DETR_DIM;
    return ctx->detr_vis;
}

/* ======================================================================
 *                         DETR decoder
 * ====================================================================== */

static int sam3_load_detr_dec(sam3_ctx *ctx)
{
    const int D = SAM3_DETR_DIM;
    const int MLP = SAM3_DETR_MLP;
    char key[256];
    #define DGETF32(nm, nb) \
        (const float *)sam3_get(ctx->st, (nm), "F32", (nb))

    ctx->dd_query_embed   = DGETF32("detr_decoder.query_embed.weight",
                                    (size_t)200 * D * 4);
    ctx->dd_ref_points    = DGETF32("detr_decoder.reference_points.weight",
                                    (size_t)200 * 4 * 4);
    ctx->dd_presence_token = DGETF32("detr_decoder.presence_token.weight",
                                     (size_t)1 * D * 4);
    ctx->dd_output_ln_w   = DGETF32("detr_decoder.output_layer_norm.weight",
                                    D * 4);
    ctx->dd_output_ln_b   = DGETF32("detr_decoder.output_layer_norm.bias",
                                    D * 4);
    ctx->dd_presence_ln_w = DGETF32("detr_decoder.presence_layer_norm.weight",
                                    D * 4);
    ctx->dd_presence_ln_b = DGETF32("detr_decoder.presence_layer_norm.bias",
                                    D * 4);
    if (!ctx->dd_query_embed || !ctx->dd_ref_points ||
        !ctx->dd_presence_token || !ctx->dd_output_ln_w ||
        !ctx->dd_presence_ln_w) return -1;

    /* box_head 3-layer: 256→256→256→4. */
    int boxd[3] = {D, D, 4};
    ctx->dd_box_out[0] = D; ctx->dd_box_out[1] = D; ctx->dd_box_out[2] = 4;
    for (int i = 0; i < 3; i++) {
        int out_d = boxd[i];
        int in_d  = (i == 0) ? D : ((i == 1) ? D : D);
        snprintf(key, sizeof(key),
                 "detr_decoder.box_head.layer%d.weight", i + 1);
        const float *w = DGETF32(key, (size_t)out_d * in_d * 4);
        snprintf(key, sizeof(key),
                 "detr_decoder.box_head.layer%d.bias", i + 1);
        const float *b = DGETF32(key, (size_t)out_d * 4);
        if (!w || !b) return -1;
        ctx->dd_box_w[i] = sam3_alloc_f16(w, (size_t)out_d * in_d);
        ctx->dd_box_b[i] = b;
    }
    /* presence_head 3-layer: 256→256→256→1. */
    int presd[3] = {D, D, 1};
    for (int i = 0; i < 3; i++) {
        int out_d = presd[i];
        int in_d  = D;
        ctx->dd_pres_out[i] = out_d;
        snprintf(key, sizeof(key),
                 "detr_decoder.presence_head.layer%d.weight", i + 1);
        const float *w = DGETF32(key, (size_t)out_d * in_d * 4);
        snprintf(key, sizeof(key),
                 "detr_decoder.presence_head.layer%d.bias", i + 1);
        const float *b = DGETF32(key, (size_t)out_d * 4);
        if (!w || !b) return -1;
        ctx->dd_pres_w[i] = sam3_alloc_f16(w, (size_t)out_d * in_d);
        ctx->dd_pres_b[i] = b;
    }
    /* ref_point_head 2-layer: 512→256→256. */
    int rphd_in[2]  = {512, D};
    int rphd_out[2] = {D, D};
    for (int i = 0; i < 2; i++) {
        ctx->dd_rph_out[i] = rphd_out[i];
        snprintf(key, sizeof(key),
                 "detr_decoder.ref_point_head.layer%d.weight", i + 1);
        const float *w = DGETF32(key, (size_t)rphd_out[i] * rphd_in[i] * 4);
        snprintf(key, sizeof(key),
                 "detr_decoder.ref_point_head.layer%d.bias", i + 1);
        const float *b = DGETF32(key, (size_t)rphd_out[i] * 4);
        if (!w || !b) return -1;
        ctx->dd_rph_w[i] = sam3_alloc_f16(w,
            (size_t)rphd_out[i] * rphd_in[i]);
        ctx->dd_rph_b[i] = b;
    }
    /* box_rpb_embed_x/y 2-layer: 2→256→8. */
    int rpbd_in[2]  = {2, D};
    int rpbd_out[2] = {D, SAM3_DETR_HEADS};
    for (int i = 0; i < 2; i++) {
        ctx->dd_rpbx_out[i] = rpbd_out[i];
        ctx->dd_rpby_out[i] = rpbd_out[i];
        snprintf(key, sizeof(key),
                 "detr_decoder.box_rpb_embed_x.layer%d.weight", i + 1);
        const float *wx = DGETF32(key, (size_t)rpbd_out[i]*rpbd_in[i]*4);
        snprintf(key, sizeof(key),
                 "detr_decoder.box_rpb_embed_x.layer%d.bias", i + 1);
        const float *bx = DGETF32(key, (size_t)rpbd_out[i] * 4);
        snprintf(key, sizeof(key),
                 "detr_decoder.box_rpb_embed_y.layer%d.weight", i + 1);
        const float *wy = DGETF32(key, (size_t)rpbd_out[i]*rpbd_in[i]*4);
        snprintf(key, sizeof(key),
                 "detr_decoder.box_rpb_embed_y.layer%d.bias", i + 1);
        const float *by = DGETF32(key, (size_t)rpbd_out[i] * 4);
        if (!wx || !wy) return -1;
        ctx->dd_rpbx_w[i] = sam3_alloc_f16(wx,
            (size_t)rpbd_out[i] * rpbd_in[i]);
        ctx->dd_rpby_w[i] = sam3_alloc_f16(wy,
            (size_t)rpbd_out[i] * rpbd_in[i]);
        ctx->dd_rpbx_b[i] = bx;
        ctx->dd_rpby_b[i] = by;
    }

    /* 6 decoder layers. */
    for (int li = 0; li < SAM3_DETR_LAYERS; li++) {
        #define DK2(fmt, ...) (snprintf(key, sizeof(key), fmt, ##__VA_ARGS__), key)
        /* Self-attn. */
        const float *w;
        #define LOAD_QKVO(prefix, dst_q, dst_k, dst_v, dst_o,            \
                          dst_qb, dst_kb, dst_vb, dst_ob,                \
                          dst_lnw, dst_lnb, ln_prefix)                    \
            do {                                                          \
                w = DGETF32(DK2("detr_decoder.layers.%d." prefix          \
                                ".q_proj.weight", li), (size_t)D*D*4);   \
                if (!w) return -1;                                        \
                ctx->dd[li].dst_q = sam3_alloc_f16(w, (size_t)D*D);      \
                ctx->dd[li].dst_qb = DGETF32(DK2("detr_decoder.layers.%d." prefix \
                                ".q_proj.bias", li), D*4);                \
                w = DGETF32(DK2("detr_decoder.layers.%d." prefix          \
                                ".k_proj.weight", li), (size_t)D*D*4);   \
                ctx->dd[li].dst_k = sam3_alloc_f16(w, (size_t)D*D);      \
                ctx->dd[li].dst_kb = DGETF32(DK2("detr_decoder.layers.%d." prefix \
                                ".k_proj.bias", li), D*4);                \
                w = DGETF32(DK2("detr_decoder.layers.%d." prefix          \
                                ".v_proj.weight", li), (size_t)D*D*4);   \
                ctx->dd[li].dst_v = sam3_alloc_f16(w, (size_t)D*D);      \
                ctx->dd[li].dst_vb = DGETF32(DK2("detr_decoder.layers.%d." prefix \
                                ".v_proj.bias", li), D*4);                \
                w = DGETF32(DK2("detr_decoder.layers.%d." prefix          \
                                ".o_proj.weight", li), (size_t)D*D*4);   \
                ctx->dd[li].dst_o = sam3_alloc_f16(w, (size_t)D*D);      \
                ctx->dd[li].dst_ob = DGETF32(DK2("detr_decoder.layers.%d." prefix \
                                ".o_proj.bias", li), D*4);                \
                ctx->dd[li].dst_lnw = DGETF32(DK2("detr_decoder.layers.%d." \
                                ln_prefix "_layer_norm.weight", li), D*4);\
                ctx->dd[li].dst_lnb = DGETF32(DK2("detr_decoder.layers.%d." \
                                ln_prefix "_layer_norm.bias", li), D*4); \
            } while (0)

        LOAD_QKVO("self_attn", sa_q, sa_k, sa_v, sa_o,
                  sa_qb, sa_kb, sa_vb, sa_ob, sa_ln_w, sa_ln_b,
                  "self_attn");
        LOAD_QKVO("text_cross_attn", ta_q, ta_k, ta_v, ta_o,
                  ta_qb, ta_kb, ta_vb, ta_ob, ta_ln_w, ta_ln_b,
                  "text_cross_attn");
        LOAD_QKVO("vision_cross_attn", va_q, va_k, va_v, va_o,
                  va_qb, va_kb, va_vb, va_ob, va_ln_w, va_ln_b,
                  "vision_cross_attn");

        w = DGETF32(DK2("detr_decoder.layers.%d.mlp.fc1.weight", li),
                    (size_t)MLP*D*4);
        if (!w) return -1;
        ctx->dd[li].fc1 = sam3_alloc_f16(w, (size_t)MLP*D);
        ctx->dd[li].fc1_b = DGETF32(DK2("detr_decoder.layers.%d.mlp.fc1.bias", li), MLP*4);
        w = DGETF32(DK2("detr_decoder.layers.%d.mlp.fc2.weight", li),
                    (size_t)D*MLP*4);
        ctx->dd[li].fc2 = sam3_alloc_f16(w, (size_t)D*MLP);
        ctx->dd[li].fc2_b = DGETF32(DK2("detr_decoder.layers.%d.mlp.fc2.bias", li), D*4);
        ctx->dd[li].mlp_ln_w = DGETF32(DK2("detr_decoder.layers.%d.mlp_layer_norm.weight", li), D*4);
        ctx->dd[li].mlp_ln_b = DGETF32(DK2("detr_decoder.layers.%d.mlp_layer_norm.bias", li), D*4);

        #undef LOAD_QKVO
        #undef DK2
    }
    #undef DGETF32
    return 0;
}

static float sam3_sigmoidf(float x)
{
    if (x >= 0.0f) { float e = expf(-x); return 1.0f / (1.0f + e); }
    float e = expf(x); return e / (1.0f + e);
}

static float sam3_inverse_sigmoidf(float x)
{
    const float eps = 1e-3f;
    if (x < 0.0f) x = 0.0f; else if (x > 1.0f) x = 1.0f;
    float x1 = x < eps ? eps : x;
    float x2 = (1.0f - x) < eps ? eps : (1.0f - x);
    return logf(x1 / x2);
}

/* Apply a Sam3DecoderMLP: ReLU(l1) → [ReLU(l2) → l3] if 3 layers, else l2. */
static void sam3_decoder_mlp(float *dst, const float *src, int N,
                             const uint16_t *const *W, const float *const *B,
                             const int *out_dims, int in_dim, int n_layers,
                             float *scratch1, float *scratch2, int nt)
{
    /* scratch1, scratch2 must be large enough for max hidden. */
    const float *in = src;
    int cur_in = in_dim;
    float *ping = scratch1, *pong = scratch2;
    for (int i = 0; i < n_layers; i++) {
        int out_d = out_dims[i];
        float *out = (i == n_layers - 1) ? dst : ping;
        cpu_gemm_f16(out, W[i], B[i], in, N, out_d, cur_in, nt);
        if (i < n_layers - 1) sam3_relu(out, N * out_d);
        in = out;
        cur_in = out_d;
        float *tmp = ping; ping = pong; pong = tmp;
    }
}

/* Encode boxes (Nq, 4) with num_pos_feats=128, scale=2pi, no normalize.
 * Output (Nq, 512) = cat(pos_y, pos_x, pos_w, pos_h). Boxes are cxcywh. */
static void sam3_encode_boxes(float *dst, const float *boxes, int Nq)
{
    const int NF = 128;
    const float scale = 2.0f * (float)M_PI;
    float dim_t[128];
    for (int j = 0; j < NF; j++) {
        dim_t[j] = powf(10000.0f, (float)(2 * (j / 2)) / (float)NF);
    }
    /* Per coord index: 0=x, 1=y, 2=w, 3=h. Final output order: y, x, w, h. */
    const int order[4] = {1, 0, 2, 3};
    for (int q = 0; q < Nq; q++) {
        float *out = dst + (size_t)q * 4 * NF;
        for (int ci = 0; ci < 4; ci++) {
            int c = order[ci];
            float v = boxes[(size_t)q * 4 + c] * scale;
            float *seg = out + (size_t)ci * NF;
            /* For pair p: seg[2p] = sin(v/dim_t[2p]), seg[2p+1] = cos(v/dim_t[2p+1]). */
            for (int p = 0; p < NF / 2; p++) {
                float a = v / dim_t[2 * p];
                float b = v / dim_t[2 * p + 1];
                seg[2 * p]     = sinf(a);
                seg[2 * p + 1] = cosf(b);
            }
        }
    }
}

/* Compute RPB. reference_boxes (200, 4) cxcywh sigmoid-space.
 * Output: (HEADS, 201, H*W) where row 0 (presence) is zeros. */
static void sam3_compute_rpb(float *rpb, const float *ref_boxes,
                             int Nq, int H, int W, const sam3_ctx *ctx,
                             int nt)
{
    const int NH = SAM3_DETR_HEADS;
    const int HD = SAM3_DETR_DIM;
    const int Npix = H * W;
    const float log2_8 = 3.0f;

    /* Build deltas_x (Nq, W, 2) and deltas_y (Nq, H, 2). */
    float *dx_log = (float *)malloc((size_t)Nq * W * 2 * sizeof(float));
    float *dy_log = (float *)malloc((size_t)Nq * H * 2 * sizeof(float));
    for (int q = 0; q < Nq; q++) {
        float cx = ref_boxes[(size_t)q * 4 + 0];
        float cy = ref_boxes[(size_t)q * 4 + 1];
        float w_  = ref_boxes[(size_t)q * 4 + 2];
        float h_  = ref_boxes[(size_t)q * 4 + 3];
        float x1 = cx - 0.5f * w_, x2 = cx + 0.5f * w_;
        float y1 = cy - 0.5f * h_, y2 = cy + 0.5f * h_;
        float *rx = dx_log + (size_t)q * W * 2;
        float *ry = dy_log + (size_t)q * H * 2;
        for (int x = 0; x < W; x++) {
            float cx_pos = (float)x / (float)W;
            float d0 = (cx_pos - x1) * 8.0f;
            float d1 = (cx_pos - x2) * 8.0f;
            rx[x * 2 + 0] = (d0 >= 0 ? 1.0f : -1.0f) *
                            log2f(fabsf(d0) + 1.0f) / log2_8;
            rx[x * 2 + 1] = (d1 >= 0 ? 1.0f : -1.0f) *
                            log2f(fabsf(d1) + 1.0f) / log2_8;
        }
        for (int y = 0; y < H; y++) {
            float cy_pos = (float)y / (float)H;
            float d0 = (cy_pos - y1) * 8.0f;
            float d1 = (cy_pos - y2) * 8.0f;
            ry[y * 2 + 0] = (d0 >= 0 ? 1.0f : -1.0f) *
                            log2f(fabsf(d0) + 1.0f) / log2_8;
            ry[y * 2 + 1] = (d1 >= 0 ? 1.0f : -1.0f) *
                            log2f(fabsf(d1) + 1.0f) / log2_8;
        }
    }

    /* Apply 2-layer MLPs. Input rows: x (Nq*W, 2), y (Nq*H, 2).
     * Outputs: (Nq*W, NH) and (Nq*H, NH). Intermediate (Nq*max, HD). */
    int max_pix = (H > W) ? H : W;
    float *hx = (float *)malloc((size_t)Nq * W * HD * sizeof(float));
    float *hy = (float *)malloc((size_t)Nq * H * HD * sizeof(float));
    float *ex = (float *)malloc((size_t)Nq * W * NH * sizeof(float));
    float *ey = (float *)malloc((size_t)Nq * H * NH * sizeof(float));
    cpu_gemm_f16(hx, ctx->dd_rpbx_w[0], ctx->dd_rpbx_b[0],
                 dx_log, Nq * W, HD, 2, nt);
    sam3_relu(hx, Nq * W * HD);
    cpu_gemm_f16(ex, ctx->dd_rpbx_w[1], ctx->dd_rpbx_b[1],
                 hx, Nq * W, NH, HD, nt);
    cpu_gemm_f16(hy, ctx->dd_rpby_w[0], ctx->dd_rpby_b[0],
                 dy_log, Nq * H, HD, 2, nt);
    sam3_relu(hy, Nq * H * HD);
    cpu_gemm_f16(ey, ctx->dd_rpby_w[1], ctx->dd_rpby_b[1],
                 hy, Nq * H, NH, HD, nt);
    free(hx); free(hy);

    /* Assemble: for each (h, q, (y,x)) set rpb = ex[q,x,h] + ey[q,y,h],
     * with row 0 (presence) = 0. */
    (void)max_pix;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < NH; h++) {
        for (int q = 0; q < Nq; q++) {
            float *out = rpb + ((size_t)h * (Nq + 1) + (q + 1)) * Npix;
            const float *eyq = ey + (size_t)q * H * NH;
            const float *exq = ex + (size_t)q * W * NH;
            for (int y = 0; y < H; y++) {
                float yv = eyq[y * NH + h];
                for (int x = 0; x < W; x++) {
                    out[y * W + x] = yv + exq[x * NH + h];
                }
            }
        }
    }
    /* Zero presence row (q=0). */
    for (int h = 0; h < NH; h++) {
        memset(rpb + (size_t)h * (Nq + 1) * Npix, 0, Npix * sizeof(float));
    }
    free(dx_log); free(dy_log); free(ex); free(ey);
}

static void sam3_detr_dec_layer(sam3_ctx *ctx, int li,
                                const float *text_feats, int T,
                                const int32_t *text_mask)
{
    const int D = SAM3_DETR_DIM;
    const int MLP = SAM3_DETR_MLP;
    const int Nt = SAM3_DETR_QPLUS1;          /* 201 */
    const int N  = ctx->n_tok;                /* 5184 */
    const int nt = ctx->cfg.num_threads;
    /* x is ctx->dd_hs (post-norm, updated in-place each sub-block). */
    float *x = ctx->dd_hs;

    /* -------- Self-attention. -------- */
    /* Q/K add query_pos to hidden; V uses hidden directly. */
    for (int i = 0; i < Nt * D; i++)
        ctx->dd_hs_in[i] = x[i] + ctx->dd_qpos[i];
    cpu_gemm_f16(ctx->dd_q, ctx->dd[li].sa_q, ctx->dd[li].sa_qb,
                 ctx->dd_hs_in, Nt, D, D, nt);
    cpu_gemm_f16(ctx->dd_k, ctx->dd[li].sa_k, ctx->dd[li].sa_kb,
                 ctx->dd_hs_in, Nt, D, D, nt);
    cpu_gemm_f16(ctx->dd_v, ctx->dd[li].sa_v, ctx->dd[li].sa_vb,
                 x, Nt, D, D, nt);
    sam3_mha_detr_full(ctx->dd_mlp, ctx->dd_q, ctx->dd_k, ctx->dd_v,
                       Nt, Nt, NULL, NULL);
    cpu_gemm_f16(ctx->dd_scratch, ctx->dd[li].sa_o, ctx->dd[li].sa_ob,
                 ctx->dd_mlp, Nt, D, D, nt);
    for (int i = 0; i < Nt * D; i++) x[i] += ctx->dd_scratch[i];
    /* Post-norm. */
    cpu_layernorm(ctx->dd_scratch, x,
                  ctx->dd[li].sa_ln_w, ctx->dd[li].sa_ln_b,
                  Nt, D, SAM3_DETR_LN_EPS);
    memcpy(x, ctx->dd_scratch, (size_t)Nt * D * sizeof(float));

    /* -------- Text cross-attention. -------- */
    for (int i = 0; i < Nt * D; i++)
        ctx->dd_hs_in[i] = x[i] + ctx->dd_qpos[i];
    cpu_gemm_f16(ctx->dd_q, ctx->dd[li].ta_q, ctx->dd[li].ta_qb,
                 ctx->dd_hs_in, Nt, D, D, nt);
    cpu_gemm_f16(ctx->dd_k, ctx->dd[li].ta_k, ctx->dd[li].ta_kb,
                 text_feats, T, D, D, nt);
    cpu_gemm_f16(ctx->dd_v, ctx->dd[li].ta_v, ctx->dd[li].ta_vb,
                 text_feats, T, D, D, nt);
    sam3_mha_detr_full(ctx->dd_mlp, ctx->dd_q, ctx->dd_k, ctx->dd_v,
                       Nt, T, text_mask, NULL);
    cpu_gemm_f16(ctx->dd_scratch, ctx->dd[li].ta_o, ctx->dd[li].ta_ob,
                 ctx->dd_mlp, Nt, D, D, nt);
    for (int i = 0; i < Nt * D; i++) x[i] += ctx->dd_scratch[i];
    cpu_layernorm(ctx->dd_scratch, x,
                  ctx->dd[li].ta_ln_w, ctx->dd[li].ta_ln_b,
                  Nt, D, SAM3_DETR_LN_EPS);
    memcpy(x, ctx->dd_scratch, (size_t)Nt * D * sizeof(float));

    /* -------- Vision cross-attention (with RPB). -------- */
    /* Q = x + qpos; K = vision + pos; V = vision. */
    for (int i = 0; i < Nt * D; i++)
        ctx->dd_hs_in[i] = x[i] + ctx->dd_qpos[i];
    cpu_gemm_f16(ctx->dd_q, ctx->dd[li].va_q, ctx->dd[li].va_qb,
                 ctx->dd_hs_in, Nt, D, D, nt);
    /* Key source: detr_enc output + detr_pos. Reuse mlp buffer (5184,256). */
    float *kin = ctx->detr_mlp;  /* temp: fits since MLP=2048 >> 256 */
    for (int i = 0; i < N * D; i++)
        kin[i] = ctx->detr_vis[i] + ctx->detr_pos[i];
    cpu_gemm_f16(ctx->dd_k, ctx->dd[li].va_k, ctx->dd[li].va_kb,
                 kin, N, D, D, nt);
    cpu_gemm_f16(ctx->dd_v, ctx->dd[li].va_v, ctx->dd[li].va_vb,
                 ctx->detr_vis, N, D, D, nt);
    sam3_mha_detr_full(ctx->dd_mlp, ctx->dd_q, ctx->dd_k, ctx->dd_v,
                       Nt, N, NULL, ctx->dd_rpb);
    cpu_gemm_f16(ctx->dd_scratch, ctx->dd[li].va_o, ctx->dd[li].va_ob,
                 ctx->dd_mlp, Nt, D, D, nt);
    for (int i = 0; i < Nt * D; i++) x[i] += ctx->dd_scratch[i];
    cpu_layernorm(ctx->dd_scratch, x,
                  ctx->dd[li].va_ln_w, ctx->dd[li].va_ln_b,
                  Nt, D, SAM3_DETR_LN_EPS);
    memcpy(x, ctx->dd_scratch, (size_t)Nt * D * sizeof(float));

    /* -------- MLP. -------- */
    cpu_gemm_f16(ctx->dd_mlp, ctx->dd[li].fc1, ctx->dd[li].fc1_b,
                 x, Nt, MLP, D, nt);
    sam3_relu(ctx->dd_mlp, Nt * MLP);
    cpu_gemm_f16(ctx->dd_scratch, ctx->dd[li].fc2, ctx->dd[li].fc2_b,
                 ctx->dd_mlp, Nt, D, MLP, nt);
    for (int i = 0; i < Nt * D; i++) x[i] += ctx->dd_scratch[i];
    cpu_layernorm(ctx->dd_scratch, x,
                  ctx->dd[li].mlp_ln_w, ctx->dd[li].mlp_ln_b,
                  Nt, D, SAM3_DETR_LN_EPS);
    memcpy(x, ctx->dd_scratch, (size_t)Nt * D * sizeof(float));
}

int sam3_run_detr_dec(sam3_ctx *ctx)
{
    if (!ctx || !ctx->detr_done) return -1;
    const int D = SAM3_DETR_DIM;
    const int Nq = SAM3_DETR_QUERIES;       /* 200 */
    const int Nt = SAM3_DETR_QPLUS1;        /* 201 */
    const int N  = ctx->n_tok;              /* 5184 */
    const int T  = SAM3_TEXT_CTX;           /* 32 */
    const int nt = ctx->cfg.num_threads;

    /* Initial reference_boxes = sigmoid(reference_points). */
    for (int i = 0; i < Nq * 4; i++)
        ctx->dd_ref_boxes[i] = sam3_sigmoidf(ctx->dd_ref_points[i]);

    /* Hidden states: [presence_token; query_embed]. */
    memcpy(ctx->dd_hs + 0,
           ctx->dd_presence_token, (size_t)D * sizeof(float));
    memcpy(ctx->dd_hs + D,
           ctx->dd_query_embed, (size_t)Nq * D * sizeof(float));

    /* Text features (projected to 256) and mask. */
    const float *text_feats = ctx->detr_text_pooled;  /* (32, 256) */
    /* Presence-logit scratch. Use dd_v transiently. */

    for (int li = 0; li < SAM3_DETR_LAYERS; li++) {
        /* Build sine encoding for current reference_boxes, then query_pos. */
        sam3_encode_boxes(ctx->dd_sine_box, ctx->dd_ref_boxes, Nq);
        /* ref_point_head MLP: 512→256→256; output (Nq, 256). */
        int rph_outs[2] = {ctx->dd_rph_out[0], ctx->dd_rph_out[1]};
        sam3_decoder_mlp(ctx->dd_qpos + D,    /* skip row 0 (presence) */
                         ctx->dd_sine_box,
                         Nq,
                         (const uint16_t * const*)ctx->dd_rph_w,
                         ctx->dd_rph_b, rph_outs, 512, 2,
                         ctx->dd_mlp, ctx->dd_scratch, nt);
        /* Presence row of qpos = 0. */
        for (int i = 0; i < D; i++) ctx->dd_qpos[i] = 0.0f;

        /* RPB. */
        sam3_compute_rpb(ctx->dd_rpb, ctx->dd_ref_boxes,
                         Nq, ctx->grid, ctx->grid, ctx, nt);

        /* Layer forward. */
        sam3_detr_dec_layer(ctx, li, text_feats, T, ctx->text_attn_mask);

        /* Box refinement: delta = box_head(output_layer_norm(query_hidden)). */
        float *qh = ctx->dd_inter + (size_t)li * Nq * D;
        cpu_layernorm(qh, ctx->dd_hs + D,
                      ctx->dd_output_ln_w, ctx->dd_output_ln_b,
                      Nq, D, SAM3_DETR_LN_EPS);
        float *delta = (float *)malloc((size_t)Nq * 4 * sizeof(float));
        int box_outs[3] = {ctx->dd_box_out[0], ctx->dd_box_out[1],
                           ctx->dd_box_out[2]};
        sam3_decoder_mlp(delta, qh, Nq,
                         (const uint16_t * const*)ctx->dd_box_w,
                         ctx->dd_box_b, box_outs, D, 3,
                         ctx->dd_mlp, ctx->dd_scratch, nt);
        for (int q = 0; q < Nq; q++) {
            for (int c = 0; c < 4; c++) {
                float rb = ctx->dd_ref_boxes[q * 4 + c];
                float inv = sam3_inverse_sigmoidf(rb);
                float nb = sam3_sigmoidf(delta[q * 4 + c] + inv);
                ctx->dd_ref_boxes[q * 4 + c] = nb;
            }
        }
        free(delta);

        /* Presence logit. */
        float ph[SAM3_DETR_DIM];
        float pl_in[SAM3_DETR_DIM];
        cpu_layernorm(pl_in, ctx->dd_hs,
                      ctx->dd_presence_ln_w, ctx->dd_presence_ln_b,
                      1, D, SAM3_DETR_LN_EPS);
        float *ping = ph, *pong = NULL;
        (void)pong;
        int pres_outs[3] = {ctx->dd_pres_out[0], ctx->dd_pres_out[1],
                            ctx->dd_pres_out[2]};
        float pl_out[SAM3_DETR_DIM];
        float tmp1[SAM3_DETR_DIM], tmp2[SAM3_DETR_DIM];
        sam3_decoder_mlp(pl_out, pl_in, 1,
                         (const uint16_t * const*)ctx->dd_pres_w,
                         ctx->dd_pres_b, pres_outs, D, 3,
                         tmp1, tmp2, nt);
        float v = pl_out[0];
        if (v > SAM3_DETR_PRESENCE_CLAMP) v = SAM3_DETR_PRESENCE_CLAMP;
        if (v < -SAM3_DETR_PRESENCE_CLAMP) v = -SAM3_DETR_PRESENCE_CLAMP;
        ctx->dd_presence_logits[li] = v;
        (void)ping;
    }

    /* Final pred boxes: cxcywh → xyxy. */
    for (int q = 0; q < Nq; q++) {
        float cx = ctx->dd_ref_boxes[q * 4 + 0];
        float cy = ctx->dd_ref_boxes[q * 4 + 1];
        float w = ctx->dd_ref_boxes[q * 4 + 2];
        float h = ctx->dd_ref_boxes[q * 4 + 3];
        ctx->dd_pred_boxes[q * 4 + 0] = cx - 0.5f * w;
        ctx->dd_pred_boxes[q * 4 + 1] = cy - 0.5f * h;
        ctx->dd_pred_boxes[q * 4 + 2] = cx + 0.5f * w;
        ctx->dd_pred_boxes[q * 4 + 3] = cy + 0.5f * h;
    }

    ctx->dd_done = 1;
    return 0;
}

const float *sam3_get_detr_dec_boxes(const sam3_ctx *ctx)
{
    return (ctx && ctx->dd_done) ? ctx->dd_pred_boxes : NULL;
}

const float *sam3_get_detr_dec_presence(const sam3_ctx *ctx)
{
    return (ctx && ctx->dd_done) ? ctx->dd_presence_logits : NULL;
}

const float *sam3_get_detr_dec_hidden(const sam3_ctx *ctx)
{
    if (!ctx || !ctx->dd_done) return NULL;
    /* Last-layer output_layer_norm(query_hidden). */
    return ctx->dd_inter +
           (size_t)(SAM3_DETR_LAYERS - 1) * SAM3_DETR_QUERIES * SAM3_DETR_DIM;
}

/* ======================================================================
 *                       Dot-product scoring
 * ====================================================================== */

static int sam3_load_dot_score(sam3_ctx *ctx)
{
    const int D = SAM3_DETR_DIM;
    const int MLP = SAM3_DETR_MLP;
    #define DGETF32(nm, nb) (const float *)sam3_get(ctx->st, (nm), "F32", (nb))

    const float *w1 = DGETF32("dot_product_scoring.text_mlp.layer1.weight",
                              (size_t)MLP * D * 4);
    const float *b1 = DGETF32("dot_product_scoring.text_mlp.layer1.bias",
                              MLP * 4);
    const float *w2 = DGETF32("dot_product_scoring.text_mlp.layer2.weight",
                              (size_t)D * MLP * 4);
    const float *b2 = DGETF32("dot_product_scoring.text_mlp.layer2.bias",
                              D * 4);
    if (!w1 || !b1 || !w2 || !b2) return -1;
    ctx->ds_tmlp_w[0] = sam3_alloc_f16(w1, (size_t)MLP * D);
    ctx->ds_tmlp_w[1] = sam3_alloc_f16(w2, (size_t)D * MLP);
    ctx->ds_tmlp_b[0] = b1; ctx->ds_tmlp_b[1] = b2;

    ctx->ds_tnorm_w = DGETF32("dot_product_scoring.text_mlp_out_norm.weight",
                              D * 4);
    ctx->ds_tnorm_b = DGETF32("dot_product_scoring.text_mlp_out_norm.bias",
                              D * 4);
    const float *tpw = DGETF32("dot_product_scoring.text_proj.weight",
                               (size_t)D * D * 4);
    ctx->ds_tproj_b = DGETF32("dot_product_scoring.text_proj.bias", D * 4);
    const float *qpw = DGETF32("dot_product_scoring.query_proj.weight",
                               (size_t)D * D * 4);
    ctx->ds_qproj_b = DGETF32("dot_product_scoring.query_proj.bias", D * 4);
    if (!tpw || !qpw || !ctx->ds_tnorm_w || !ctx->ds_tnorm_b ||
        !ctx->ds_tproj_b || !ctx->ds_qproj_b) return -1;
    ctx->ds_tproj_w = sam3_alloc_f16(tpw, (size_t)D * D);
    ctx->ds_qproj_w = sam3_alloc_f16(qpw, (size_t)D * D);
    #undef DGETF32
    return 0;
}

int sam3_run_dot_score(sam3_ctx *ctx)
{
    if (!ctx || !ctx->dd_done) return -1;
    const int D = SAM3_DETR_DIM;
    const int MLP = SAM3_DETR_MLP;
    const int T = SAM3_TEXT_CTX;
    const int Nq = SAM3_DETR_QUERIES;
    const int nt = ctx->cfg.num_threads;

    /* text_mlp on detr_text_pooled (T, D) -> h (T, D). */
    float *tmp_mlp = (float *)malloc((size_t)T * MLP * sizeof(float));
    float *h       = (float *)malloc((size_t)T * D * sizeof(float));
    if (!tmp_mlp || !h) { free(tmp_mlp); free(h); return -1; }
    cpu_gemm_f16(tmp_mlp, ctx->ds_tmlp_w[0], ctx->ds_tmlp_b[0],
                 ctx->detr_text_pooled, T, MLP, D, nt);
    sam3_relu(tmp_mlp, T * MLP);
    cpu_gemm_f16(h, ctx->ds_tmlp_w[1], ctx->ds_tmlp_b[1],
                 tmp_mlp, T, D, MLP, nt);
    free(tmp_mlp);
    /* residual + LN. */
    for (int i = 0; i < T * D; i++) h[i] += ctx->detr_text_pooled[i];
    float *h_ln = (float *)malloc((size_t)T * D * sizeof(float));
    if (!h_ln) { free(h); return -1; }
    cpu_layernorm(h_ln, h, ctx->ds_tnorm_w, ctx->ds_tnorm_b,
                  T, D, SAM3_DETR_LN_EPS);
    free(h);

    /* Pool over valid tokens. */
    float pooled[SAM3_DETR_DIM] = {0};
    int nv = 0;
    for (int t = 0; t < T; t++) {
        if (ctx->text_attn_mask[t]) {
            for (int d = 0; d < D; d++) pooled[d] += h_ln[t * D + d];
            nv++;
        }
    }
    if (nv < 1) nv = 1;
    for (int d = 0; d < D; d++) pooled[d] /= (float)nv;
    free(h_ln);

    /* proj_text (1, D). */
    float proj_text[SAM3_DETR_DIM];
    cpu_gemm_f16(proj_text, ctx->ds_tproj_w, ctx->ds_tproj_b,
                 pooled, 1, D, D, nt);

    /* For each layer, project queries and dot with proj_text. */
    const float scale = 1.0f / sqrtf((float)D);
    float *qproj = (float *)malloc((size_t)Nq * D * sizeof(float));
    if (!qproj) return -1;
    for (int li = 0; li < SAM3_DETR_LAYERS; li++) {
        const float *qh = ctx->dd_inter + (size_t)li * Nq * D;
        cpu_gemm_f16(qproj, ctx->ds_qproj_w, ctx->ds_qproj_b,
                     qh, Nq, D, D, nt);
        for (int q = 0; q < Nq; q++) {
            float acc = 0.0f;
            for (int d = 0; d < D; d++)
                acc += qproj[q * D + d] * proj_text[d];
            acc *= scale;
            if (acc > 12.0f) acc = 12.0f;
            if (acc < -12.0f) acc = -12.0f;
            ctx->ds_scores[li * Nq + q] = acc;
        }
    }
    free(qproj);
    ctx->ds_done = 1;
    return 0;
}

const float *sam3_get_dot_scores(const sam3_ctx *ctx)
{
    return (ctx && ctx->ds_done) ? ctx->ds_scores : NULL;
}

/* ======================================================================
 *                          Mask decoder
 * ====================================================================== */

static int sam3_load_mask_decoder(sam3_ctx *ctx)
{
    const int D = SAM3_DETR_DIM;
    #define DGETF32(nm, nb) (const float *)sam3_get(ctx->st, (nm), "F32", (nb))

    /* prompt_cross_attn. */
    const float *qw = DGETF32("mask_decoder.prompt_cross_attn.q_proj.weight",
                              (size_t)D * D * 4);
    const float *kw = DGETF32("mask_decoder.prompt_cross_attn.k_proj.weight",
                              (size_t)D * D * 4);
    const float *vw = DGETF32("mask_decoder.prompt_cross_attn.v_proj.weight",
                              (size_t)D * D * 4);
    const float *ow = DGETF32("mask_decoder.prompt_cross_attn.o_proj.weight",
                              (size_t)D * D * 4);
    ctx->md_pca_q_b = DGETF32("mask_decoder.prompt_cross_attn.q_proj.bias", D*4);
    ctx->md_pca_k_b = DGETF32("mask_decoder.prompt_cross_attn.k_proj.bias", D*4);
    ctx->md_pca_v_b = DGETF32("mask_decoder.prompt_cross_attn.v_proj.bias", D*4);
    ctx->md_pca_o_b = DGETF32("mask_decoder.prompt_cross_attn.o_proj.bias", D*4);
    ctx->md_pca_ln_w = DGETF32("mask_decoder.prompt_cross_attn_norm.weight", D*4);
    ctx->md_pca_ln_b = DGETF32("mask_decoder.prompt_cross_attn_norm.bias", D*4);
    if (!qw || !kw || !vw || !ow || !ctx->md_pca_ln_w) return -1;
    ctx->md_pca_q_w = sam3_alloc_f16(qw, (size_t)D * D);
    ctx->md_pca_k_w = sam3_alloc_f16(kw, (size_t)D * D);
    ctx->md_pca_v_w = sam3_alloc_f16(vw, (size_t)D * D);
    ctx->md_pca_o_w = sam3_alloc_f16(ow, (size_t)D * D);

    /* pixel_decoder conv_layers + norms (3 each; only 2 are used at inference). */
    for (int i = 0; i < 3; i++) {
        char k[128];
        snprintf(k, sizeof(k), "mask_decoder.pixel_decoder.conv_layers.%d.weight", i);
        ctx->md_conv_w[i] = DGETF32(k, (size_t)D * D * 3 * 3 * 4);
        snprintf(k, sizeof(k), "mask_decoder.pixel_decoder.conv_layers.%d.bias", i);
        ctx->md_conv_b[i] = DGETF32(k, D * 4);
        snprintf(k, sizeof(k), "mask_decoder.pixel_decoder.norms.%d.weight", i);
        ctx->md_gn_w[i] = DGETF32(k, D * 4);
        snprintf(k, sizeof(k), "mask_decoder.pixel_decoder.norms.%d.bias", i);
        ctx->md_gn_b[i] = DGETF32(k, D * 4);
        if (!ctx->md_conv_w[i] || !ctx->md_gn_w[i]) return -1;
    }
    /* mask_embedder 3-layer Linear 256→256. */
    for (int i = 0; i < 3; i++) {
        char k[128];
        snprintf(k, sizeof(k), "mask_decoder.mask_embedder.layers.%d.weight", i);
        const float *w = DGETF32(k, (size_t)D * D * 4);
        snprintf(k, sizeof(k), "mask_decoder.mask_embedder.layers.%d.bias", i);
        const float *b = DGETF32(k, D * 4);
        if (!w || !b) return -1;
        ctx->md_me_w[i] = sam3_alloc_f16(w, (size_t)D * D);
        ctx->md_me_b[i] = b;
    }
    ctx->md_ip_w = DGETF32("mask_decoder.instance_projection.weight",
                           (size_t)D * D * 1 * 1 * 4);
    ctx->md_ip_b = DGETF32("mask_decoder.instance_projection.bias", D * 4);
    ctx->md_sp_w = DGETF32("mask_decoder.semantic_projection.weight",
                           (size_t)1 * D * 1 * 1 * 4);
    ctx->md_sp_b = DGETF32("mask_decoder.semantic_projection.bias", 1 * 4);
    if (!ctx->md_ip_w || !ctx->md_sp_w) return -1;
    #undef DGETF32
    return 0;
}

/* GroupNorm with 8 groups over (C, H, W) — C=256 → 32 channels per group.
 * Normalize across group_c * H * W. Affine per channel. eps=1e-5. */
static void sam3_groupnorm_8(float *x, const float *w, const float *b,
                             int C, int H, int W)
{
    const int G = 8;
    const int gc = C / G;
    const int spatial = H * W;
    const float eps = 1e-5f;
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < G; g++) {
        int c0 = g * gc;
        size_t n = (size_t)gc * spatial;
        double sum = 0.0, sqsum = 0.0;
        for (int c = 0; c < gc; c++) {
            const float *row = x + (size_t)(c0 + c) * spatial;
            for (int i = 0; i < spatial; i++) {
                float v = row[i]; sum += v; sqsum += (double)v * v;
            }
        }
        float mean = (float)(sum / (double)n);
        float var  = (float)(sqsum / (double)n) - mean * mean;
        float inv  = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < gc; c++) {
            float *row = x + (size_t)(c0 + c) * spatial;
            float ww = w[c0 + c];
            float bb = b[c0 + c];
            for (int i = 0; i < spatial; i++) {
                row[i] = (row[i] - mean) * inv * ww + bb;
            }
        }
    }
}

/* Nearest-neighbor upsample (C, Hi, Wi) -> (C, Ho, Wo). */
static void sam3_nn_upsample(float *out, const float *in,
                             int C, int Hi, int Wi, int Ho, int Wo)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < Ho; oh++) {
            int ih = (int)((long long)oh * Hi / Ho);
            if (ih >= Hi) ih = Hi - 1;
            const float *src = in + ((size_t)c * Hi + ih) * Wi;
            float *dst = out + ((size_t)c * Ho + oh) * Wo;
            for (int ow = 0; ow < Wo; ow++) {
                int iw = (int)((long long)ow * Wi / Wo);
                if (iw >= Wi) iw = Wi - 1;
                dst[ow] = src[iw];
            }
        }
    }
}

int sam3_run_mask_dec(sam3_ctx *ctx)
{
    if (!ctx || !ctx->dd_done) return -1;
    const int D = SAM3_DETR_DIM;
    const int Nt = ctx->n_tok;
    const int T = SAM3_TEXT_CTX;
    const int Nq = SAM3_DETR_QUERIES;
    const int nt = ctx->cfg.num_threads;
    const int H2 = ctx->fpn[2].out_h, W2 = ctx->fpn[2].out_w;  /* 72 */
    const int H1 = ctx->fpn[1].out_h, W1 = ctx->fpn[1].out_w;  /* 144 */
    const int H0 = ctx->fpn[0].out_h, W0 = ctx->fpn[0].out_w;  /* 288 */

    /* 1) prompt_cross_attn: encoder attends to text_features. */
    /*    residual = encoder_hidden_states (= ctx->detr_vis). */
    /*    normed = LN(encoder). */
    float *normed = (float *)malloc((size_t)Nt * D * sizeof(float));
    float *q_proj = (float *)malloc((size_t)Nt * D * sizeof(float));
    float *k_proj = (float *)malloc((size_t)T  * D * sizeof(float));
    float *v_proj = (float *)malloc((size_t)T  * D * sizeof(float));
    float *attn_out = (float *)malloc((size_t)Nt * D * sizeof(float));
    float *o_out    = (float *)malloc((size_t)Nt * D * sizeof(float));
    if (!normed || !q_proj || !k_proj || !v_proj || !attn_out || !o_out) {
        free(normed); free(q_proj); free(k_proj); free(v_proj);
        free(attn_out); free(o_out); return -1;
    }
    cpu_layernorm(normed, ctx->detr_vis, ctx->md_pca_ln_w, ctx->md_pca_ln_b,
                  Nt, D, SAM3_DETR_LN_EPS);
    cpu_gemm_f16(q_proj, ctx->md_pca_q_w, ctx->md_pca_q_b, normed, Nt, D, D, nt);
    cpu_gemm_f16(k_proj, ctx->md_pca_k_w, ctx->md_pca_k_b,
                 ctx->detr_text_pooled, T, D, D, nt);
    cpu_gemm_f16(v_proj, ctx->md_pca_v_w, ctx->md_pca_v_b,
                 ctx->detr_text_pooled, T, D, D, nt);
    sam3_mha_detr_full(attn_out, q_proj, k_proj, v_proj, Nt, T,
                       ctx->text_attn_mask, NULL);
    cpu_gemm_f16(o_out, ctx->md_pca_o_w, ctx->md_pca_o_b, attn_out,
                 Nt, D, D, nt);
    for (size_t i = 0; i < (size_t)Nt * D; i++)
        ctx->md_enc_mod[i] = ctx->detr_vis[i] + o_out[i];
    free(normed); free(q_proj); free(k_proj); free(v_proj);
    free(attn_out); free(o_out);

    /* 2) pixel_decoder: start from encoder reshaped to (C=256, H2, W2). */
    /*    backbone_features = [fpn0, fpn1, fpn2]; fpn2 replaced by md_enc_mod. */
    float *prev = (float *)malloc((size_t)D * H1 * W1 * sizeof(float));
    float *up   = (float *)malloc((size_t)D * H0 * W0 * sizeof(float));
    if (!prev || !up) { free(prev); free(up); return -1; }

    /* Reshape md_enc_mod (Nt=5184, 256) -> (256, 72, 72) via transpose. */
    float *start = (float *)malloc((size_t)D * H2 * W2 * sizeof(float));
    if (!start) { free(prev); free(up); return -1; }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < D; c++) {
        for (int h = 0; h < H2; h++) {
            float *dst = start + ((size_t)c * H2 + h) * W2;
            for (int w_ = 0; w_ < W2; w_++) {
                dst[w_] = ctx->md_enc_mod[(size_t)(h * W2 + w_) * D + c];
            }
        }
    }

    /* Stage 0: upsample 72->144, add fpn1, conv+GN+ReLU. */
    sam3_nn_upsample(prev, start, D, H2, W2, H1, W1);
    for (size_t i = 0; i < (size_t)D * H1 * W1; i++)
        prev[i] += ctx->fpn_out[1][i];
    /* conv2d (C,H,W) -> (C,H,W) k=3 pad=1 */
    float *c_out = (float *)malloc((size_t)D * H1 * W1 * sizeof(float));
    sam3_conv2d(c_out, prev, ctx->md_conv_w[0], ctx->md_conv_b[0],
                D, D, H1, W1, 3, 1);
    sam3_groupnorm_8(c_out, ctx->md_gn_w[0], ctx->md_gn_b[0], D, H1, W1);
    sam3_relu(c_out, D * H1 * W1);
    free(start);

    /* Stage 1: upsample 144->288, add fpn0, conv+GN+ReLU -> ctx->md_pixel. */
    sam3_nn_upsample(up, c_out, D, H1, W1, H0, W0);
    for (size_t i = 0; i < (size_t)D * H0 * W0; i++)
        up[i] += ctx->fpn_out[0][i];
    sam3_conv2d(ctx->md_pixel, up, ctx->md_conv_w[1], ctx->md_conv_b[1],
                D, D, H0, W0, 3, 1);
    sam3_groupnorm_8(ctx->md_pixel, ctx->md_gn_w[1], ctx->md_gn_b[1],
                     D, H0, W0);
    sam3_relu(ctx->md_pixel, D * H0 * W0);
    free(c_out); free(prev); free(up);

    /* 3) instance_projection: Conv2d k=1 256→256. */
    sam3_conv2d(ctx->md_instance, ctx->md_pixel, ctx->md_ip_w, ctx->md_ip_b,
                D, D, H0, W0, 1, 0);
    /* 4) semantic_projection: Conv2d k=1 256→1. */
    sam3_conv2d(ctx->md_semantic, ctx->md_pixel, ctx->md_sp_w, ctx->md_sp_b,
                D, 1, H0, W0, 1, 0);

    /* 5) mask_embedder: 3-layer MLP on last-layer decoder hidden. */
    const float *queries = ctx->dd_inter +
        (size_t)(SAM3_DETR_LAYERS - 1) * Nq * D;
    {
        float *a = (float *)malloc((size_t)Nq * D * sizeof(float));
        float *b = (float *)malloc((size_t)Nq * D * sizeof(float));
        if (!a || !b) { free(a); free(b); return -1; }
        cpu_gemm_f16(a, ctx->md_me_w[0], ctx->md_me_b[0], queries, Nq, D, D, nt);
        sam3_relu(a, Nq * D);
        cpu_gemm_f16(b, ctx->md_me_w[1], ctx->md_me_b[1], a, Nq, D, D, nt);
        sam3_relu(b, Nq * D);
        cpu_gemm_f16(ctx->md_mask_emb, ctx->md_me_w[2], ctx->md_me_b[2],
                     b, Nq, D, D, nt);
        free(a); free(b);
    }

    /* 6) einsum("qc,chw->qhw") */
    const size_t HW = (size_t)H0 * W0;
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < Nq; q++) {
        float *dst = ctx->md_pred_masks + (size_t)q * HW;
        const float *me = ctx->md_mask_emb + (size_t)q * D;
        for (size_t i = 0; i < HW; i++) {
            float acc = 0.0f;
            for (int c = 0; c < D; c++)
                acc += me[c] * ctx->md_instance[c * HW + i];
            dst[i] = acc;
        }
    }
    ctx->md_done = 1;
    return 0;
}

const float *sam3_get_pred_masks(const sam3_ctx *ctx,
                                  int *out_q, int *out_h, int *out_w)
{
    if (!ctx || !ctx->md_done) return NULL;
    if (out_q) *out_q = SAM3_DETR_QUERIES;
    if (out_h) *out_h = ctx->fpn[0].out_h;
    if (out_w) *out_w = ctx->fpn[0].out_w;
    return ctx->md_pred_masks;
}

const float *sam3_get_semantic_seg(const sam3_ctx *ctx,
                                    int *out_h, int *out_w)
{
    if (!ctx || !ctx->md_done) return NULL;
    if (out_h) *out_h = ctx->fpn[0].out_h;
    if (out_w) *out_w = ctx->fpn[0].out_w;
    return ctx->md_semantic;
}

/* ======================================================================
 *                          Post-processing
 * ====================================================================== */

/* Bilinear resize (C, Hi, Wi) -> (C, Ho, Wo) with align_corners=False,
 * matching torch.nn.functional.interpolate(mode='bilinear'). */
static void sam3_bilinear_resize_acf(float *dst, const float *src,
                                     int C, int Hi, int Wi, int Ho, int Wo)
{
    const float sy = (float)Hi / (float)Ho;
    const float sx = (float)Wi / (float)Wo;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < Ho; oh++) {
            float fy = (oh + 0.5f) * sy - 0.5f;
            int y0 = (int)floorf(fy);
            int y1 = y0 + 1;
            float dy = fy - (float)y0;
            if (y0 < 0) { y0 = 0; dy = 0.0f; }
            if (y1 < 0) y1 = 0;
            if (y0 > Hi - 1) y0 = Hi - 1;
            if (y1 > Hi - 1) y1 = Hi - 1;
            for (int ow = 0; ow < Wo; ow++) {
                float fx = (ow + 0.5f) * sx - 0.5f;
                int x0 = (int)floorf(fx);
                int x1 = x0 + 1;
                float dx = fx - (float)x0;
                if (x0 < 0) { x0 = 0; dx = 0.0f; }
                if (x1 < 0) x1 = 0;
                if (x0 > Wi - 1) x0 = Wi - 1;
                if (x1 > Wi - 1) x1 = Wi - 1;
                const float *sp = src + (size_t)c * Hi * Wi;
                float v00 = sp[y0 * Wi + x0], v01 = sp[y0 * Wi + x1];
                float v10 = sp[y1 * Wi + x0], v11 = sp[y1 * Wi + x1];
                float v = v00 * (1-dy) * (1-dx) + v01 * (1-dy) * dx
                        + v10 *    dy  * (1-dx) + v11 *    dy  * dx;
                dst[(size_t)c * Ho * Wo + oh * Wo + ow] = v;
            }
        }
    }
}

int sam3_run_postprocess(sam3_ctx *ctx, int target_h, int target_w,
                         float score_threshold, float mask_threshold)
{
    if (!ctx || !ctx->md_done) return -1;
    if (target_h <= 0 || target_w <= 0) return -1;
    const int Nq = SAM3_DETR_QUERIES;
    const int H0 = ctx->fpn[0].out_h, W0 = ctx->fpn[0].out_w;

    /* Free previous run output. */
    free(ctx->pp_scores); ctx->pp_scores = NULL;
    free(ctx->pp_boxes);  ctx->pp_boxes  = NULL;
    free(ctx->pp_masks);  ctx->pp_masks  = NULL;
    ctx->pp_n_kept = 0;
    ctx->pp_done = 0;

    /* Last-layer dot-score logits + presence. */
    const float *logits = ctx->ds_scores +
        (size_t)(SAM3_DETR_LAYERS - 1) * Nq;
    float pres = ctx->dd_presence_logits[SAM3_DETR_LAYERS - 1];
    float pres_s = 1.0f / (1.0f + expf(-pres));

    /* Score = sigmoid(logit) * sigmoid(presence), threshold filter. */
    int *keep_idx = (int *)malloc((size_t)Nq * sizeof(int));
    int n = 0;
    for (int q = 0; q < Nq; q++) {
        float s = 1.0f / (1.0f + expf(-logits[q])) * pres_s;
        if (s > score_threshold) keep_idx[n++] = q;
    }
    if (n == 0) {
        free(keep_idx);
        ctx->pp_th = target_h; ctx->pp_tw = target_w;
        ctx->pp_done = 1;
        return 0;
    }

    ctx->pp_scores = (float *)malloc((size_t)n * sizeof(float));
    ctx->pp_boxes  = (float *)malloc((size_t)n * 4 * sizeof(float));
    ctx->pp_masks  = (uint8_t *)malloc((size_t)n * target_h * target_w);
    if (!ctx->pp_scores || !ctx->pp_boxes || !ctx->pp_masks) {
        free(keep_idx);
        free(ctx->pp_scores); free(ctx->pp_boxes); free(ctx->pp_masks);
        ctx->pp_scores = NULL; ctx->pp_boxes = NULL; ctx->pp_masks = NULL;
        return -1;
    }

    /* Fill scores + boxes. */
    for (int i = 0; i < n; i++) {
        int q = keep_idx[i];
        ctx->pp_scores[i] =
            (1.0f / (1.0f + expf(-logits[q]))) * pres_s;
        float *b = ctx->pp_boxes + (size_t)i * 4;
        const float *pb = ctx->dd_pred_boxes + (size_t)q * 4;
        b[0] = pb[0] * (float)target_w;
        b[1] = pb[1] * (float)target_h;
        b[2] = pb[2] * (float)target_w;
        b[3] = pb[3] * (float)target_h;
    }

    /* Gather kept masks into contiguous buffer (n, 288, 288). */
    const size_t HW = (size_t)H0 * W0;
    float *kept = (float *)malloc((size_t)n * HW * sizeof(float));
    if (!kept) { free(keep_idx); return -1; }
    for (int i = 0; i < n; i++) {
        const float *src = ctx->md_pred_masks +
            (size_t)keep_idx[i] * HW;
        memcpy(kept + (size_t)i * HW, src, HW * sizeof(float));
    }
    free(keep_idx);

    /* Sigmoid in-place. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float *row = kept + (size_t)i * HW;
        for (size_t j = 0; j < HW; j++) {
            float v = row[j];
            row[j] = 1.0f / (1.0f + expf(-v));
        }
    }

    /* Bilinear resize to target. */
    float *resized = (float *)malloc(
        (size_t)n * target_h * target_w * sizeof(float));
    if (!resized) { free(kept); return -1; }
    sam3_bilinear_resize_acf(resized, kept, n, H0, W0, target_h, target_w);
    free(kept);

    /* Binarize to uint8. */
    size_t tot = (size_t)n * target_h * target_w;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < tot; i++) {
        ctx->pp_masks[i] = (resized[i] > mask_threshold) ? 1 : 0;
    }
    free(resized);

    ctx->pp_n_kept = n;
    ctx->pp_th = target_h; ctx->pp_tw = target_w;
    ctx->pp_done = 1;
    return 0;
}

const float *sam3_get_final_scores(const sam3_ctx *ctx, int *out_n)
{
    if (!ctx || !ctx->pp_done) return NULL;
    if (out_n) *out_n = ctx->pp_n_kept;
    return ctx->pp_scores;
}

const float *sam3_get_final_boxes(const sam3_ctx *ctx, int *out_n)
{
    if (!ctx || !ctx->pp_done) return NULL;
    if (out_n) *out_n = ctx->pp_n_kept;
    return ctx->pp_boxes;
}

const uint8_t *sam3_get_final_masks(const sam3_ctx *ctx,
                                     int *out_n, int *out_h, int *out_w)
{
    if (!ctx || !ctx->pp_done) return NULL;
    if (out_n) *out_n = ctx->pp_n_kept;
    if (out_h) *out_h = ctx->pp_th;
    if (out_w) *out_w = ctx->pp_tw;
    return ctx->pp_masks;
}

int sam3_predict_text(sam3_ctx *ctx, const char *phrase,
                      int max_masks, float *out_masks, float *out_scores)
{
    (void)ctx; (void)phrase; (void)max_masks;
    (void)out_masks; (void)out_scores;
    fprintf(stderr, "sam3_predict_text: not implemented yet\n");
    return -1;
}
