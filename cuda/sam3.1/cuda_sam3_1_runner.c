/*
 * cuda_sam3_1_runner.c - HIP/ROCm SAM 3 runner (RDNA4).
 *
 * Phase 1: preprocess + patch_embed + pos_embed + pre-LN.
 * Phase 2: 32 ViT blocks (2D axial RoPE, windowed/global MHA, MLP).
 *
 * Weights load from sam3.model.safetensors via common/safetensors.h with
 * the `detector.` prefix stripped (sam3.1 — differs from sam3).
 *
 * SPDX-License-Identifier: MIT
 */

#include "cuda_sam3_1_runner.h"
#include "../cuew.h"
#include "../cuda_kernels_common.h"
#include "cuda_sam3_1_kernels.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
/* SAFETENSORS_IMPLEMENTATION emits external-linkage symbols. When linked
 * alongside another TU that already supplies them (e.g. diffusion-server
 * links server.c which carries SAFETENSORS_IMPLEMENTATION, or alongside
 * cuda/sam3/cuda_sam3_runner.c), define CUDA_SAM3_1_RUNNER_EXTERNAL_IMPLS
 * to skip. Standalone cuda/sam3.1 build is unaffected. */
#ifndef CUDA_SAM3_1_RUNNER_EXTERNAL_IMPLS
#define SAFETENSORS_IMPLEMENTATION
#endif
#include "../../common/safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef HIP_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_POINTER CU_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    CU_LAUNCH_PARAM_BUFFER_SIZE
#define HIP_LAUNCH_PARAM_END            CU_LAUNCH_PARAM_END
#endif

#define S3_IMG     1008
#define S3_PATCH   14
#define S3_GRID    (S3_IMG / S3_PATCH)   /* 72 */
#define S3_DIM     1024
#define S3_HEADS   16
#define S3_HEAD    64
#define S3_MLP     4736
#define S3_PRE     24
#define S3_WIN     24
#define S3_NTOK    (S3_GRID * S3_GRID)   /* 5184 */
#define S3_N_WIN   ((S3_GRID / S3_WIN) * (S3_GRID / S3_WIN))  /* 9 */
#define S3_WINTOK  (S3_WIN * S3_WIN)     /* 576 */
#define S3_NBLK    32
#define S3_EPS     1e-6f
#define S3_THETA   10000.0f
#define S3_FPN_DIM 256
/* sam3.1 has 3 conv stages (convs.0..2): ×4, ×2, ×1 upsample; no ×0.5
 * downsample stage like sam3. */
#define S3_FPN_LEV 3
#define S3_T_DIM    1024
#define S3_T_HEADS  16
#define S3_T_HD     64
#define S3_T_MLP    4096
#define S3_T_LAYERS 24
#define S3_T_CTX    32
#define S3_T_VOCAB  49408
#define S3_T_EPS    1e-5f

#define S3_DETR_DIM     256
#define S3_DETR_HEADS   8
#define S3_DETR_HD      32
#define S3_DETR_MLP     2048
#define S3_DETR_LAYERS  6
#define S3_DETR_LN_EPS  1e-5f
#define S3_DETR_Q       200
#define S3_DETR_QP1     201
#define S3_DETR_PCLAMP  10.0f

static int is_global(int bi) {
    return bi == 7 || bi == 15 || bi == 23 || bi == 31;
}

typedef struct {
    void *norm1_w, *norm1_b;  /* F32 (D,) */
    void *norm2_w, *norm2_b;
    void *qkv_w;              /* F16 (3*D, D) */
    void *qkv_b;              /* F32 (3*D,) */
    void *o_w;                /* F16 (D, D) */
    void *o_b;                /* F32 (D,) */
    void *fc1_w;              /* F16 (MLP, D) */
    void *fc1_b;              /* F32 (MLP,) */
    void *fc2_w;              /* F16 (D, MLP) */
    void *fc2_b;              /* F32 (D,) */
    int is_global;
} s3_block;

typedef struct {
    float scale;               /* 4, 2, 1, 0.5 */
    int   inter_c;             /* channels after scale stage */
    int   out_h, out_w;
    /* ct0/ct1: F16 weight (Ci, Co, 2, 2), F32 bias (Co,). NULL if unused. */
    void *ct0_w, *ct0_b; int ct0_in, ct0_out;
    void *ct1_w, *ct1_b; int ct1_in, ct1_out;
    int   has_gelu_between;
    int   has_maxpool;
    /* proj1: 1x1 conv (inter_c → 256). proj2: 3x3 pad=1 (256 → 256). */
    void *p1_w, *p1_b;
    void *p2_w, *p2_b;
    /* Output buffer (256, out_h, out_w). */
    void *d_out;
} s3_fpn_layer;

struct cuda_sam3_1_ctx {
    cuda_sam3_1_config cfg;
    int device_id;
    int verbose;
    int sm;           /* compute capability (major*10+minor) */
    int use_mma;      /* 1 if sm >= 80 and env does not force tiled */
    hipModule_t mod;

    /* kernels */
    hipFunction_t fn_resize, fn_ln, fn_gemm, fn_gemm_tiled, fn_gelu, fn_add;
    hipFunction_t fn_patch, fn_pos_add;
    hipFunction_t fn_win_part, fn_win_unpart, fn_rope;
    hipFunction_t fn_kv_tx, fn_fa;
    hipFunction_t fn_tok2nchw, fn_c1x1, fn_c3x3, fn_c1x1_f32, fn_c3x3_f32, fn_ct2, fn_mp2;
    hipFunction_t fn_gelu_erf, fn_clip_causal;

    st_context *st;

    /* weights */
    void *w_patch;
    void *w_pos;
    void *w_pre_ln_w, *w_pre_ln_b;
    s3_block blk[S3_NBLK];
    void *d_rope_win_cos, *d_rope_win_sin;  /* (576, 64) F32 */
    void *d_rope_glb_cos, *d_rope_glb_sin;  /* (5184, 64) F32 */

    /* activations */
    void *d_img_u8; size_t img_u8_cap;
    void *d_img_f32;
    void *d_tok;      /* x — residual stream (5184, 1024) */
    void *d_tmp;      /* LN output, also residual add target */
    void *d_win;      /* window-permuted or spatial scratch (5184, 1024) */
    void *d_qkv;      /* (5184, 3*1024) */
    void *d_attn;     /* (5184, 1024) */
    void *d_mlp;      /* (5184, 4736) */
    void *d_Kt;       /* (heads, n_max, head_dim) = (16, 5184, 64) */
    void *d_Vt;
    int ready;
    int vit_started;
    int last_block;

    s3_fpn_layer fpn[S3_FPN_LEV];
    int fpn_ready;

    /* CLIP text encoder. */
    const float *h_tok_embed;   /* mmap'd F32 (49408, 1024), host-side gather */
    const float *h_pos_embed;   /* mmap'd F32 (32, 1024) */
    void *t_final_ln_w, *t_final_ln_b;
    s3_block t_blk[S3_T_LAYERS];  /* reuses s3_block struct */
    void *d_text_tok;     /* (32, 1024) F32 */
    void *d_text_tmp;     /* (32, 1024) F32 */
    void *d_text_qkv;     /* (32, 3*1024) F32 */
    void *d_text_attn;    /* (32, 1024) F32 */
    void *d_text_mlp;     /* (32, 4096) F32 */
    int32_t text_ids[S3_T_CTX];
    int32_t text_mask[S3_T_CTX];
    int text_ids_ready;
    int text_done;

    /* DETR encoder. */
    hipFunction_t fn_nchw2tok, fn_add_pos, fn_relu, fn_mha32;
    void *w_text_proj_w;  /* F16 (256, 1024) */
    void *w_text_proj_b;  /* F32 (256,) */
    /* per-layer DETR weights */
    struct {
        void *ln1_w, *ln1_b, *ln2_w, *ln2_b, *ln3_w, *ln3_b;
        void *qkv_w, *qkv_b;       /* fused self-attn (3D, D) F16, bias (3D,) F32 */
        void *o_w, *o_b;           /* self-attn out proj */
        void *cq_w, *cq_b;         /* cross-attn Q: (D, D) */
        void *ck_w, *ck_b;
        void *cv_w, *cv_b;
        void *co_w, *co_b;
        void *fc1_w, *fc1_b;       /* (MLP, D) / (MLP,) */
        void *fc2_w, *fc2_b;
    } detr_enc[S3_DETR_LAYERS];
    /* Buffers. */
    void *d_detr_vis;       /* (N, 256) residual stream */
    void *d_detr_scratch;   /* (N, 256) LN output / temp */
    void *d_detr_q;         /* (N, 256) */
    void *d_detr_k;         /* (N, 256) self, (32,256) cross */
    void *d_detr_v;         /* (N, 256) self, (32,256) cross */
    void *d_detr_qkv;       /* (N, 3*256) fused self-attn gemm output */
    void *d_detr_attn;      /* (N, 256) attention output */
    void *d_detr_mlp;       /* (N, 2048) */
    void *d_detr_pos;       /* (N, 256) sine pos */
    void *d_detr_text_pooled; /* (32, 256) projected text */
    void *d_text_mask_i32;  /* (32,) int32 device */
    int detr_done;

    /* DETR decoder. */
    struct {
        void *sa_q, *sa_qb, *sa_k, *sa_kb, *sa_v, *sa_vb, *sa_o, *sa_ob;
        void *sa_ln_w, *sa_ln_b;
        void *ta_q, *ta_qb, *ta_k, *ta_kb, *ta_v, *ta_vb, *ta_o, *ta_ob;
        void *ta_ln_w, *ta_ln_b;
        void *va_q, *va_qb, *va_k, *va_kb, *va_v, *va_vb, *va_o, *va_ob;
        void *va_ln_w, *va_ln_b;
        void *fc1, *fc1_b, *fc2, *fc2_b;
        void *mlp_ln_w, *mlp_ln_b;
    } dd[S3_DETR_LAYERS];
    /* Aux MLPs / static weights. */
    void *dd_query_embed;        /* device F32 (200, 256) */
    float *dd_ref_points_host;   /* host F32 (200, 4) */
    void *dd_presence_token;     /* device F32 (256,) */
    void *dd_output_ln_w, *dd_output_ln_b;
    void *dd_presence_ln_w, *dd_presence_ln_b;
    void *dd_box_w[3], *dd_box_b[3];      /* 256→256→256→4 F16 + F32 biases */
    void *dd_pres_w[3], *dd_pres_b[3];    /* 256→256→256→1 */
    void *dd_rph_w[2],  *dd_rph_b[2];     /* 512→256→256 */
    void *dd_rpbx_w[2], *dd_rpbx_b[2];    /* 2→256→8 */
    void *dd_rpby_w[2], *dd_rpby_b[2];
    /* Runtime buffers. */
    void *d_dd_hs;       /* (201, 256) residual stream */
    void *d_dd_qpos;     /* (201, 256) query pos */
    void *d_dd_hs_in;    /* (201, 256) hs + qpos */
    void *d_dd_q;        /* (201, 256) */
    void *d_dd_k;        /* (5184, 256) or (32, 256) */
    void *d_dd_v;        /* (5184, 256) or (32, 256) */
    void *d_dd_attn;     /* (201, 256) */
    void *d_dd_mlp;      /* (201, 2048) */
    void *d_dd_scratch;  /* (201, 256) */
    void *d_dd_kin;      /* (5184, 256) vision+pos for va.K gemm input */
    void *d_dd_rpb;      /* (8, 201, 5184) RPB bias */
    void *d_dd_sine_box; /* (200, 512) */
    void *d_dd_rph_hidden; /* (200, 256) */
    void *d_dd_box_hidden; /* (200, 256) */
    void *d_dd_pres_hidden;/* (1, 256) */
    void *d_dd_qh;       /* (200, 256) output_ln'd query hidden last layer */
    void *d_dd_delta;    /* (200, 4) */
    void *d_dd_presence; /* (1, 1) device temp */
    void *d_dd_dx_log;   /* (200, 72, 2) */
    void *d_dd_dy_log;   /* (200, 72, 2) */
    void *d_dd_ex;       /* (200, 72, 8) */
    void *d_dd_ey;       /* (200, 72, 8) */

    float dd_ref_boxes[S3_DETR_Q * 4];
    float dd_pred_boxes[S3_DETR_Q * 4];
    float dd_presence_logits[S3_DETR_LAYERS];
    int dd_done;
    /* Per-layer post-output_ln query hidden (6, 200, 256) — fed to
     * dot_product_scoring + mask_embedder. */
    void *d_dd_inter;

    /* dot_product_scoring. */
    void *ds_tmlp_w[2], *ds_tmlp_b[2];  /* F16 + F32 bias: 256→2048, 2048→256 */
    void *ds_tnorm_w, *ds_tnorm_b;
    void *ds_tproj_w,  *ds_tproj_b;     /* 256→256 F16 */
    void *ds_qproj_w,  *ds_qproj_b;     /* 256→256 F16 */
    void *d_ds_tmlp_h;                  /* (32, 2048) */
    void *d_ds_text_h;                  /* (32, 256) */
    void *d_ds_text_ln;                 /* (32, 256) */
    void *d_ds_pooled;                  /* (1, 256) */
    void *d_ds_proj_text;               /* (1, 256) */
    void *d_ds_qproj;                   /* (200, 256) */
    void *d_ds_scores_li;               /* (200, 1) device temp */
    float ds_scores[S3_DETR_LAYERS * S3_DETR_Q];
    int ds_done;

    /* Mask decoder weights. */
    void *md_pca_q_w, *md_pca_q_b;
    void *md_pca_k_w, *md_pca_k_b;
    void *md_pca_v_w, *md_pca_v_b;
    void *md_pca_o_w, *md_pca_o_b;
    void *md_pca_ln_w, *md_pca_ln_b;
    void *md_conv_w[2], *md_conv_b[2];       /* F16 + F32, 3x3 256→256 */
    void *md_gn_w[2],   *md_gn_b[2];
    void *md_me_w[3],   *md_me_b[3];         /* F16 + F32, 256→256 */
    void *md_ip_w, *md_ip_b;                 /* 1x1 256→256 */
    void *md_sp_w, *md_sp_b;                 /* 1x1 256→1 */
    /* Mask decoder buffers. */
    void *d_md_enc_ln;    /* (5184, 256) */
    void *d_md_enc_mod;   /* (5184, 256) */
    void *d_md_q;         /* (5184, 256) */
    void *d_md_k;         /* (32, 256) */
    void *d_md_v;         /* (32, 256) */
    void *d_md_attn;      /* (5184, 256) */
    void *d_md_o;         /* (5184, 256) */
    void *d_md_start;     /* (256, 72, 72) */
    void *d_md_prev;      /* (256, 144, 144) */
    void *d_md_cout;      /* (256, 144, 144) */
    void *d_md_up;        /* (256, 288, 288) */
    void *d_md_pixel;     /* (256, 288, 288) */
    void *d_md_instance;  /* (256, 288, 288) */
    void *d_md_semantic;  /* (1, 288, 288) */
    void *d_md_me_hid;    /* (200, 256) */
    void *d_md_mask_emb;  /* (200, 256) */
    void *d_md_pred_masks;/* (200, 288, 288) */
    int md_done;

    /* Post-process state (host). */
    float   *pp_scores;
    float   *pp_boxes;
    uint8_t *pp_masks;
    int pp_n_kept, pp_th, pp_tw, pp_done;

    /* Decoder kernels. */
    hipFunction_t fn_zero_row, fn_rpb_assemble;
    hipFunction_t fn_tok2chw, fn_groupnorm, fn_upnn, fn_einsum_qchw;
};

/* ==== safetensors helpers ==== */

static int st_find_pfx(const st_context *st, const char *s) {
    char k[256]; snprintf(k, sizeof(k), "detector.%s", s);
    return safetensors_find(st, k);
}
static void *upload_f32(const st_context *st, const char *s, size_t *n) {
    int i = st_find_pfx(st, s);
    if (i < 0) { fprintf(stderr, "sam3.1: missing %s\n", s); return NULL; }
    if (strcmp(safetensors_dtype(st, i), "F32")) { fprintf(stderr, "sam3.1: %s not F32\n", s); return NULL; }
    size_t nb = safetensors_nbytes(st, i);
    if (n) *n = nb / 4;
    return hip_upload_raw(safetensors_data((st_context *)st, i), nb);
}
static void *upload_f16(const st_context *st, const char *s, size_t *n) {
    int i = st_find_pfx(st, s);
    if (i < 0) { fprintf(stderr, "sam3.1: missing %s\n", s); return NULL; }
    if (strcmp(safetensors_dtype(st, i), "F32")) { fprintf(stderr, "sam3.1: %s not F32\n", s); return NULL; }
    size_t c = safetensors_nbytes(st, i) / 4;
    const float *src = (const float *)safetensors_data((st_context *)st, i);
    uint16_t *t = (uint16_t *)malloc(c * 2);
    for (size_t k = 0; k < c; k++) t[k] = hip_f32_to_f16(src[k]);
    void *d = hip_upload_raw(t, c * 2);
    free(t);
    if (n) *n = c;
    return d;
}

/* Host-side pointer to a mmap'd F32 tensor. */
static const float *host_f32(const st_context *st, const char *s) {
    int i = st_find_pfx(st, s);
    if (i < 0) return NULL;
    if (strcmp(safetensors_dtype(st, i), "F32")) return NULL;
    return (const float *)safetensors_data((st_context *)st, i);
}

/* Split an OpenAI-MHA fused in_proj_{weight,bias} tensor into three
 * separate Q/K/V weight (F16) + bias (F32) device uploads. Weight shape
 * (3*D, D) → three (D, D) F16 blocks; bias shape (3*D,) → three (D,). */
static int split_fused_qkv(const st_context *st, const char *wkey,
                            const char *bkey, int D,
                            void **qw, void **kw, void **vw,
                            void **qb, void **kb, void **vb) {
    const float *w = host_f32(st, wkey);
    const float *b = host_f32(st, bkey);
    if (!w || !b) return -1;
    size_t DD = (size_t)D * D;
    uint16_t *tw = (uint16_t *)malloc(DD * 2);
    for (int part = 0; part < 3; part++) {
        for (size_t i = 0; i < DD; i++) tw[i] = hip_f32_to_f16(w[part*DD + i]);
        void *d = hip_upload_raw(tw, DD * 2);
        if (part == 0) *qw = d; else if (part == 1) *kw = d; else *vw = d;
    }
    free(tw);
    *qb = hip_upload_raw(b + 0*D, (size_t)D * 4);
    *kb = hip_upload_raw(b + 1*D, (size_t)D * 4);
    *vb = hip_upload_raw(b + 2*D, (size_t)D * 4);
    return 0;
}

/* Upload a fused OpenAI-MHA in_proj_{weight,bias} without splitting —
 * weight as F16 (3*D, D), bias as F32 (3*D,). */
static int load_fused_qkv(const st_context *st, const char *wkey,
                           const char *bkey, void **w_out, void **b_out) {
    *w_out = upload_f16(st, wkey, NULL);
    *b_out = upload_f32(st, bkey, NULL);
    return (*w_out && *b_out) ? 0 : -1;
}

/* sam3.1 text encoder is OpenAI-CLIP: pre-norm, fused in-checkpoint
 * attn.in_proj_{weight,bias} as (3*D, D) / (3*D), c_fc/c_proj MLP. */
static int load_text_block(cuda_sam3_1_ctx *c, int li) {
    s3_block *b = &c->t_blk[li];
    b->is_global = 0;
    char k[192];
    #define TK(suf) (snprintf(k, sizeof(k), \
        "backbone.language_backbone.encoder.transformer.resblocks.%d." suf, li), k)
    b->norm1_w = upload_f32(c->st, TK("ln_1.weight"), NULL);
    b->norm1_b = upload_f32(c->st, TK("ln_1.bias"),   NULL);
    b->norm2_w = upload_f32(c->st, TK("ln_2.weight"), NULL);
    b->norm2_b = upload_f32(c->st, TK("ln_2.bias"),   NULL);
    b->qkv_w = upload_f16(c->st, TK("attn.in_proj_weight"), NULL);
    b->qkv_b = upload_f32(c->st, TK("attn.in_proj_bias"),   NULL);
    b->o_w  = upload_f16(c->st, TK("attn.out_proj.weight"), NULL);
    b->o_b  = upload_f32(c->st, TK("attn.out_proj.bias"),   NULL);
    b->fc1_w = upload_f16(c->st, TK("mlp.c_fc.weight"), NULL);
    b->fc1_b = upload_f32(c->st, TK("mlp.c_fc.bias"),   NULL);
    b->fc2_w = upload_f16(c->st, TK("mlp.c_proj.weight"), NULL);
    b->fc2_b = upload_f32(c->st, TK("mlp.c_proj.bias"),   NULL);
    #undef TK
    if (!b->norm1_w || !b->norm1_b || !b->norm2_w || !b->norm2_b ||
        !b->qkv_w || !b->qkv_b || !b->o_w || !b->o_b ||
        !b->fc1_w || !b->fc1_b || !b->fc2_w || !b->fc2_b) return -1;
    return 0;
}

/* sam3.1: qkv is pre-fused in-checkpoint as (3*D, D). Direct F16 upload. */
static int load_block(cuda_sam3_1_ctx *c, int bi) {
    s3_block *b = &c->blk[bi];
    b->is_global = is_global(bi);
    char k[192];
    #define K(suf) (snprintf(k, sizeof(k), "backbone.vision_backbone.trunk.blocks.%d." suf, bi), k)
    b->norm1_w = upload_f32(c->st, K("norm1.weight"), NULL);
    b->norm1_b = upload_f32(c->st, K("norm1.bias"),   NULL);
    b->norm2_w = upload_f32(c->st, K("norm2.weight"), NULL);
    b->norm2_b = upload_f32(c->st, K("norm2.bias"),   NULL);
    b->qkv_w = upload_f16(c->st, K("attn.qkv.weight"), NULL);
    b->qkv_b = upload_f32(c->st, K("attn.qkv.bias"),   NULL);
    b->o_w  = upload_f16(c->st, K("attn.proj.weight"), NULL);
    b->o_b  = upload_f32(c->st, K("attn.proj.bias"),   NULL);
    b->fc1_w = upload_f16(c->st, K("mlp.fc1.weight"), NULL);
    b->fc1_b = upload_f32(c->st, K("mlp.fc1.bias"),   NULL);
    b->fc2_w = upload_f16(c->st, K("mlp.fc2.weight"), NULL);
    b->fc2_b = upload_f32(c->st, K("mlp.fc2.bias"),   NULL);
    /* TODO(Phase C): load attn.freqs_cis (complex64, (576, 32)) for 2D RoPE
     * once the ViT attention kernel is switched from RPB → RoPE. */
    #undef K
    if (!b->norm1_w || !b->norm1_b || !b->norm2_w || !b->norm2_b ||
        !b->qkv_w || !b->qkv_b || !b->o_w || !b->o_b ||
        !b->fc1_w || !b->fc1_b || !b->fc2_w || !b->fc2_b) return -1;
    return 0;
}

/* RoPE table: (seq, head_dim) cos/sin with repeat_interleave(2) pattern. */
static void build_rope(float *cos_tbl, float *sin_tbl, int ex, int ey, float scale) {
    const int hd = S3_HEAD;
    const int q = hd / 4;   /* 16 */
    float freqs[16];
    for (int j = 0; j < q; j++)
        freqs[j] = 1.0f / powf(S3_THETA, (float)(4 * j) / (float)hd);
    int seq = ex * ey;
    for (int t = 0; t < seq; t++) {
        float px = (float)(t % ex) * scale;
        float py = (float)(t / ex) * scale;
        float *c = cos_tbl + (size_t)t * hd;
        float *s = sin_tbl + (size_t)t * hd;
        for (int j = 0; j < q; j++) {
            float th = px * freqs[j];
            float cx = cosf(th), sx = sinf(th);
            c[2*j] = cx;     c[2*j+1] = cx;
            s[2*j] = sx;     s[2*j+1] = sx;
        }
        for (int j = 0; j < q; j++) {
            float th = py * freqs[j];
            float cy = cosf(th), sy = sinf(th);
            int o = 2*q + 2*j;
            c[o] = cy;       c[o+1] = cy;
            s[o] = sy;       s[o+1] = sy;
        }
    }
}

/* ==== compile / launch plumbing ==== */

static int compile_kernels(cuda_sam3_1_ctx *c) {
    size_t la = strlen(hip_kernels_common_src);
    size_t lb = strlen(cuda_sam3_1_kernels_src);
    char *src = (char *)malloc(la + lb + 1);
    memcpy(src, hip_kernels_common_src, la);
    memcpy(src + la, cuda_sam3_1_kernels_src, lb + 1);
    int rc = hip_compile_kernels(&c->mod, c->device_id, src, "sam3_kernels",
                                 c->verbose, "sam3");
    free(src);
    if (rc < 0) return -1;
    #define GET(f, s) do { \
        if (hipModuleGetFunction(&c->f, c->mod, s) != hipSuccess) { \
            fprintf(stderr, "sam3.1: missing %s\n", s); return -1; } \
    } while (0)
    GET(fn_resize,    "resize_normalize");
    GET(fn_ln,        "layernorm_f32");
    /* gemm_f16_f32: MMA m16n8k16 tensor-core path. Correct on sm_120 after the
     * Blackwell a1/a2 fragment-swap fix; verified vs CPU/tiled in verify_mma_gemm.
     * Launched as grid=(ceil(n_out/256), ceil(n_tok/16)), block=128, smem=16*16*4. */
    GET(fn_gemm,      "gemm_f16_f32");
    GET(fn_gemm_tiled,"gemm_tiled_f16_f32");
    GET(fn_gelu,      "gelu_f32");
    GET(fn_add,       "add_f32");
    GET(fn_patch,     "patch_embed_sam3");
    GET(fn_pos_add,   "pos_embed_tile_add");
    GET(fn_win_part,  "window_partition_f32");
    GET(fn_win_unpart,"window_unpartition_f32");
    GET(fn_rope,      "rope_apply_qk_f32");
    GET(fn_kv_tx,     "kv_transpose");
    GET(fn_fa,        "flash_attn_tiled_f32");
    GET(fn_tok2nchw,  "tokens_to_nchw_f32");
    GET(fn_c1x1,      "conv2d_1x1_f16");
    GET(fn_c3x3,      "conv2d_3x3_pad1_f16");
    GET(fn_c1x1_f32,  "conv2d_1x1_f32");
    GET(fn_c3x3_f32,  "conv2d_3x3_pad1_f32");
    GET(fn_ct2,       "convT_k2s2_f16");
    GET(fn_mp2,       "maxpool_k2s2_f32");
    GET(fn_gelu_erf,  "gelu_erf_f32");
    GET(fn_clip_causal, "clip_causal_attn_f32");
    GET(fn_nchw2tok,  "nchw_to_tokens_f32");
    GET(fn_add_pos,   "add_with_pos_f32");
    GET(fn_relu,      "relu_f32");  /* from hip_kernels_common */
    GET(fn_mha32,     "mha_hd32_f32");
    GET(fn_rpb_assemble, "rpb_assemble_f32");
    GET(fn_tok2chw,    "tokens_to_chw_f32");
    GET(fn_groupnorm,  "groupnorm_f32");
    GET(fn_upnn,       "nearest_upsample_f32");
    GET(fn_einsum_qchw,"einsum_qc_chw_f32");
    #undef GET
    return 0;
}

static int launch(hipFunction_t fn, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz,
                   unsigned shmem, void *p, size_t pb) {
    void *cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, p,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &pb,
                    HIP_LAUNCH_PARAM_END };
    hipError_t e = hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, bz, shmem, 0, NULL, cfg);
    if (e != hipSuccess) {
        fprintf(stderr, "sam3.1: launch err=%d grid=%ux%ux%u block=%ux%ux%u\n",
                (int)e, gx, gy, gz, bx, by, bz);
        return -1;
    }
    return 0;
}

static int r_gemm(cuda_sam3_1_ctx *c, void *Y, const void *W, const void *X,
                   const void *bias, int n_out, int n_in, int n_tok) {
    struct __attribute__((packed)) { void *Y; const void *W, *X, *bias; int n_out, n_in, n_tok; } p =
        { Y, W, X, bias, n_out, n_in, n_tok };
    /* MMA m16n8k16 requires sm_80+, n_in % 16 == 0, and benefits from n_out >= 8.
     * Fall back to the shared-memory tiled kernel on older GPUs, when the env
     * var forces tiled, or for small/odd shapes that appear in the DETR
     * decoder heads (box deltas n_out=4, presence n_out=1, RPH/RPB MLPs with
     * non-16-aligned inputs). */
    if (!c->use_mma || (n_in % 16) != 0 || n_out < 8) {
        return launch(c->fn_gemm_tiled, (unsigned)((n_out+63)/64), (unsigned)((n_tok+15)/16), 1,
                      16, 16, 1, 0, &p, sizeof(p));
    }
    return launch(c->fn_gemm, (unsigned)((n_out+255)/256), (unsigned)((n_tok+15)/16), 1,
                  128, 1, 1, 16*16*sizeof(float), &p, sizeof(p));
}
static int r_ln_eps(cuda_sam3_1_ctx *c, void *dst, const void *src, const void *w, const void *b, int n_tok, int dim, float eps) {
    struct __attribute__((packed)) { void *dst; const void *src, *w, *b; int dim; float eps; } p =
        { dst, src, w, b, dim, eps };
    return launch(c->fn_ln, n_tok, 1, 1, 256, 1, 1, 256*4, &p, sizeof(p));
}
static int r_ln(cuda_sam3_1_ctx *c, void *dst, const void *src, const void *w, const void *b, int n_tok, int dim) {
    return r_ln_eps(c, dst, src, w, b, n_tok, dim, S3_EPS);
}
static int r_gelu(cuda_sam3_1_ctx *c, void *X, int n) {
    struct __attribute__((packed)) { void *X; int n; } p = { X, n };
    return launch(c->fn_gelu, (n+255)/256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_add(cuda_sam3_1_ctx *c, void *dst, const void *src, int n) {
    struct __attribute__((packed)) { void *dst; const void *src; int n; } p = { dst, src, n };
    return launch(c->fn_add, (n+255)/256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_part(cuda_sam3_1_ctx *c, void *dst, const void *src, int H, int W, int D, int ws) {
    struct __attribute__((packed)) { void *d; const void *s; int H, W, D, ws; } p = { dst, src, H, W, D, ws };
    int total = H * W * D;
    return launch(c->fn_win_part, (total+255)/256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_unpart(cuda_sam3_1_ctx *c, void *dst, const void *src, int H, int W, int D, int ws) {
    struct __attribute__((packed)) { void *d; const void *s; int H, W, D, ws; } p = { dst, src, H, W, D, ws };
    int total = H * W * D;
    return launch(c->fn_win_unpart, (total+255)/256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_rope(cuda_sam3_1_ctx *c, void *qkv, const void *cos_t, const void *sin_t,
                   int rope_seq, int n_tok) {
    struct __attribute__((packed)) { void *qkv; const void *c, *s; int rope_seq, n_tok, heads, hd, D; } p =
        { qkv, cos_t, sin_t, rope_seq, n_tok, S3_HEADS, S3_HEAD, S3_DIM };
    return launch(c->fn_rope, n_tok, 1, 1, S3_HEADS, 1, 1, 0, &p, sizeof(p));
}
static int r_kvtx(cuda_sam3_1_ctx *c, void *Kt, void *Vt, const void *qkv, int n_tok) {
    struct __attribute__((packed)) { void *K, *V; const void *qkv; int n_tok, dim, heads, hd; } p =
        { Kt, Vt, qkv, n_tok, S3_DIM, S3_HEADS, S3_HEAD };
    int total = n_tok * S3_DIM;
    return launch(c->fn_kv_tx, (total+255)/256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_fa(cuda_sam3_1_ctx *c, void *out, const void *qkv, const void *Kt, const void *Vt, int n_tok) {
    float scale = 1.0f / sqrtf((float)S3_HEAD);
    struct __attribute__((packed)) { void *out; const void *qkv, *K, *V; int n_tok, dim, heads, hd; float sc; } p =
        { out, qkv, Kt, Vt, n_tok, S3_DIM, S3_HEADS, S3_HEAD, scale };
    /* FA kernel: grid (heads, ceil(n_tok/64)), block 64, shmem = 2*16*64 = 8192 B */
    unsigned gy = (n_tok + 63) / 64;
    unsigned shmem = 2u * 16u * 64u * sizeof(float);
    return launch(c->fn_fa, S3_HEADS, gy, 1, 64, 1, 1, shmem, &p, sizeof(p));
}

/* ==== create / destroy ==== */

cuda_sam3_1_ctx *cuda_sam3_1_create(const cuda_sam3_1_config *cfg) {
    if (!cfg || !cfg->ckpt_path) return NULL;
    cuda_sam3_1_ctx *c = (cuda_sam3_1_ctx *)calloc(1, sizeof(*c));
    c->cfg = *cfg;
    c->device_id = cfg->device_ordinal;
    c->verbose = cfg->verbose;
    c->last_block = -1;
    if (!c->cfg.image_size) c->cfg.image_size = S3_IMG;
    if (c->cfg.image_size != S3_IMG) { fprintf(stderr, "sam3.1: image_size must be %d\n", S3_IMG); free(c); return NULL; }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) { free(c); return NULL; }
    if (hipSetDevice(c->device_id) != hipSuccess) { free(c); return NULL; }

    /* Detect compute capability to decide whether the MMA m16n8k16 path is
     * usable. The instruction requires sm_80 (Ampere) or newer. Older GPUs
     * (sm_70 Volta, sm_75 Turing, sm_6x Pascal/Maxwell) must use the tiled
     * shared-memory GEMM. SAM3_GEMM=tiled forces tiled on any arch (useful
     * for A/B testing or avoiding the ~1e-4 F16-multiply drift of MMA). */
    {
        int major = 0, minor = 0;
        CUdevice dev;
        if (cuDeviceGet(&dev, c->device_id) == CUDA_SUCCESS) {
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        }
        c->sm = major * 10 + minor;
        const char *force = getenv("SAM3_GEMM");
        int force_tiled = (force && (!strcmp(force, "tiled") || !strcmp(force, "TILED")));
        /* cfg.precision: fp16 = MMA (fast, default), fp32 = tiled F32-accum
         * (slower, lower drift). bf16/fp8 not yet implemented -> fall back to fp16. */
        const char *prec = cfg->precision;
        int prec_forces_tiled = 0;
        const char *prec_note = "";
        if (prec && *prec) {
            if (!strcmp(prec, "fp16") || !strcmp(prec, "FP16") || !strcmp(prec, "half")) {
                prec_note = " (precision=fp16)";
            } else if (!strcmp(prec, "fp32") || !strcmp(prec, "FP32") || !strcmp(prec, "float")) {
                prec_forces_tiled = 1;
                prec_note = " (precision=fp32: forcing tiled F32-accum path)";
            } else if (!strcmp(prec, "bf16") || !strcmp(prec, "BF16") || !strcmp(prec, "bfloat16")) {
                prec_note = " (precision=bf16 not yet implemented, falling back to fp16)";
            } else if (!strcmp(prec, "fp8") || !strcmp(prec, "FP8")) {
                prec_note = " (precision=fp8 not yet implemented, falling back to fp16)";
            } else {
                fprintf(stderr, "sam3.1: unknown precision '%s' — using fp16\n", prec);
            }
        }
        c->use_mma = (c->sm >= 80) && !force_tiled && !prec_forces_tiled;
        if (c->verbose) fprintf(stderr, "sam3.1: sm_%d -> gemm=%s%s%s\n",
            c->sm, c->use_mma ? "mma" : "tiled",
            force_tiled ? " (SAM3_GEMM=tiled override)" : "",
            prec_note);
    }

    if (compile_kernels(c) != 0) { free(c); return NULL; }

    c->st = safetensors_open(cfg->ckpt_path);
    if (!c->st) { fprintf(stderr, "sam3.1: safetensors_open failed\n"); free(c); return NULL; }

    c->w_patch    = upload_f16(c->st, "backbone.vision_backbone.trunk.patch_embed.proj.weight", NULL);
    /* sam3.1 DOES have a learned pos_embed (1, 577, 1024): cls_token row +
     * 24×24 grid. Reference forward (ViT.forward in vitdet.py) drops the
     * cls row and tiles the remaining (24,24,1024) 3× to (72,72,1024)
     * before the blocks. pos_embed_tile_add already implements that tile
     * when P=24. (2D RoPE is still applied inside each attention block;
     * the two are stacked, not exclusive.) */
    {
        const float *pe = host_f32(c->st, "backbone.vision_backbone.trunk.pos_embed");
        if (!pe) { fprintf(stderr, "sam3.1: missing trunk.pos_embed\n"); cuda_sam3_1_destroy(c); return NULL; }
        /* Skip the first (cls_token) row, leaving (24*24, 1024) = (P,P,D). */
        c->w_pos = hip_upload_raw(pe + S3_DIM, (size_t)S3_PRE * S3_PRE * S3_DIM * 4);
        if (!c->w_pos) { cuda_sam3_1_destroy(c); return NULL; }
    }
    /* sam3.1 trunk has ln_pre (before blocks) only; no post-block norm in
     * the trunk — post-trunk ops live in the `convs` stack. */
    c->w_pre_ln_w = upload_f32(c->st, "backbone.vision_backbone.trunk.ln_pre.weight", NULL);
    c->w_pre_ln_b = upload_f32(c->st, "backbone.vision_backbone.trunk.ln_pre.bias",   NULL);
    if (!c->w_patch || !c->w_pre_ln_w || !c->w_pre_ln_b) { cuda_sam3_1_destroy(c); return NULL; }

    if (c->verbose) fprintf(stderr, "sam3.1: loading %d ViT blocks ...\n", S3_NBLK);
    for (int bi = 0; bi < S3_NBLK; bi++) {
        if (load_block(c, bi)) { cuda_sam3_1_destroy(c); return NULL; }
    }

    /* sam3.1 vision-backbone conv stack: `backbone.vision_backbone.convs.{0..2}`.
     *   convs.0 (×4): dconv_2x2_0 (1024→512) + dconv_2x2_1 (512→256) + conv_1x1 (256→256) + conv_3x3 (256→256)
     *   convs.1 (×2): dconv_2x2   (1024→512) + conv_1x1 (512→256)  + conv_3x3 (256→256)
     *   convs.2 (×1):                           conv_1x1 (1024→256) + conv_3x3 (256→256)
     * (interactive_convs.*, propagation_convs.* have the same shape but are
     * used only by the interactive/propagation paths — loading them is
     * deferred until the forward-pass wiring lands in Phase C.)
     */
    if (c->verbose) fprintf(stderr, "sam3.1: loading vision-backbone conv stack ...\n");
    const float scales[3] = { 4.0f, 2.0f, 1.0f };
    for (int li = 0; li < S3_FPN_LEV; li++) {
        s3_fpn_layer *L = &c->fpn[li];
        memset(L, 0, sizeof(*L));
        L->scale = scales[li];
        int in_c = S3_DIM, inter = in_c;
        char k[192];
        if (scales[li] == 4.0f) {
            snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.dconv_2x2_0.weight", li);
            L->ct0_w = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.dconv_2x2_0.bias", li);
            L->ct0_b = upload_f32(c->st, k, NULL);
            L->ct0_in = in_c; L->ct0_out = in_c / 2;                  /* 1024→512 */
            snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.dconv_2x2_1.weight", li);
            L->ct1_w = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.dconv_2x2_1.bias", li);
            L->ct1_b = upload_f32(c->st, k, NULL);
            L->ct1_in = in_c / 2; L->ct1_out = in_c / 4;              /* 512→256 */
            L->has_gelu_between = 1;
            inter = in_c / 4;
            L->out_h = L->out_w = S3_GRID * 4;
        } else if (scales[li] == 2.0f) {
            snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.dconv_2x2.weight", li);
            L->ct0_w = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.dconv_2x2.bias", li);
            L->ct0_b = upload_f32(c->st, k, NULL);
            L->ct0_in = in_c; L->ct0_out = in_c / 2;                  /* 1024→512 */
            inter = in_c / 2;
            L->out_h = L->out_w = S3_GRID * 2;
        } else {
            /* scales[li] == 1.0f — no ConvTranspose; p1 goes directly 1024→256. */
            inter = in_c;
            L->out_h = L->out_w = S3_GRID;
        }
        L->inter_c = inter;
        /* Keep FPN conv projections in F32 — with Ci up to 1024 the F16
         * weight-quant drift on the 1x1 would compound ~0.5 max_abs on
         * level 2, which then feeds the DETR encoder. F32 cost is ~4 MB. */
        snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.conv_1x1.weight", li);
        L->p1_w = upload_f32(c->st, k, NULL);
        snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.conv_1x1.bias", li);
        L->p1_b = upload_f32(c->st, k, NULL);
        snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.conv_3x3.weight", li);
        L->p2_w = upload_f32(c->st, k, NULL);
        snprintf(k, sizeof(k), "backbone.vision_backbone.convs.%d.conv_3x3.bias", li);
        L->p2_b = upload_f32(c->st, k, NULL);
        if (!L->p1_w || !L->p1_b || !L->p2_w || !L->p2_b) {
            fprintf(stderr, "sam3.1: convs stage %d missing proj weights\n", li);
            cuda_sam3_1_destroy(c); return NULL;
        }
        size_t out_sz = (size_t)S3_FPN_DIM * L->out_h * L->out_w * 4;
        if (hipMalloc(&L->d_out, out_sz) != hipSuccess) {
            cuda_sam3_1_destroy(c); return NULL;
        }
    }

    /* Vision activation buffers (must live before any forward pass). */
    {
        size_t tok_b  = (size_t)S3_NTOK * S3_DIM * 4;             /* 21 MB */
        size_t qkv_b  = (size_t)S3_NTOK * 3 * S3_DIM * 4;         /* 64 MB */
        size_t mlp_b  = (size_t)S3_NTOK * S3_MLP * 4;             /* 97 MB */
        size_t kv_b   = (size_t)S3_HEADS * S3_NTOK * S3_HEAD * 4; /* 21 MB */
        size_t img_b  = (size_t)3 * S3_IMG * S3_IMG * 4;
        if (hipMalloc(&c->d_img_f32, img_b) != hipSuccess ||
            hipMalloc(&c->d_tok,  tok_b)  != hipSuccess ||
            hipMalloc(&c->d_tmp,  tok_b)  != hipSuccess ||
            hipMalloc(&c->d_win,  tok_b)  != hipSuccess ||
            hipMalloc(&c->d_qkv,  qkv_b)  != hipSuccess ||
            hipMalloc(&c->d_attn, tok_b)  != hipSuccess ||
            hipMalloc(&c->d_mlp,  mlp_b)  != hipSuccess ||
            hipMalloc(&c->d_Kt,   kv_b)   != hipSuccess ||
            hipMalloc(&c->d_Vt,   kv_b)   != hipSuccess) {
            fprintf(stderr, "sam3.1: hipMalloc failed\n");
            cuda_sam3_1_destroy(c); return NULL;
        }
    }

    /* RoPE tables (built eagerly so the vision-only path still works). */
    {
        size_t nwin = (size_t)S3_WIN * S3_WIN * S3_HEAD;
        size_t nglb = (size_t)S3_GRID * S3_GRID * S3_HEAD;
        float *tcos = (float *)malloc(nwin * 4), *tsin = (float *)malloc(nwin * 4);
        build_rope(tcos, tsin, S3_WIN, S3_WIN, 1.0f);
        c->d_rope_win_cos = hip_upload_raw(tcos, nwin * 4);
        c->d_rope_win_sin = hip_upload_raw(tsin, nwin * 4);
        free(tcos); free(tsin);
        tcos = (float *)malloc(nglb * 4); tsin = (float *)malloc(nglb * 4);
        build_rope(tcos, tsin, S3_GRID, S3_GRID, (float)S3_WIN / (float)S3_GRID);
        c->d_rope_glb_cos = hip_upload_raw(tcos, nglb * 4);
        c->d_rope_glb_sin = hip_upload_raw(tsin, nglb * 4);
        free(tcos); free(tsin);
    }

    /* Phase-C escape hatch: skip everything past the vision backbone so
     * vision-only verify binaries (verify_patch_embed / verify_vit) can
     * run before the text encoder + heads have been retargeted to the
     * sam3.1 key layout. */
    if (getenv("SAM31_VISION_ONLY")) {
        if (c->verbose) fprintf(stderr,
            "sam3.1: SAM31_VISION_ONLY set — stopping after vision backbone\n");
        return c;
    }

    /* OpenAI-CLIP text encoder. */
    if (c->verbose) fprintf(stderr, "sam3.1: loading OpenAI-CLIP text encoder ...\n");
    c->h_tok_embed = host_f32(c->st, "backbone.language_backbone.encoder.token_embedding.weight");
    c->h_pos_embed = host_f32(c->st, "backbone.language_backbone.encoder.positional_embedding");
    c->t_final_ln_w = upload_f32(c->st, "backbone.language_backbone.encoder.ln_final.weight", NULL);
    c->t_final_ln_b = upload_f32(c->st, "backbone.language_backbone.encoder.ln_final.bias", NULL);
    if (!c->h_tok_embed || !c->h_pos_embed || !c->t_final_ln_w || !c->t_final_ln_b) {
        fprintf(stderr, "sam3.1: missing CLIP text encoder weights\n");
        cuda_sam3_1_destroy(c); return NULL;
    }
    for (int li = 0; li < S3_T_LAYERS; li++) {
        if (load_text_block(c, li)) {
            fprintf(stderr, "sam3.1: failed to load text block %d\n", li);
            cuda_sam3_1_destroy(c); return NULL;
        }
    }
    size_t text_tok_b = (size_t)S3_T_CTX * S3_T_DIM * 4;
    size_t text_qkv_b = (size_t)S3_T_CTX * 3 * S3_T_DIM * 4;
    size_t text_mlp_b = (size_t)S3_T_CTX * S3_T_MLP * 4;
    if (hipMalloc(&c->d_text_tok,  text_tok_b) != hipSuccess ||
        hipMalloc(&c->d_text_tmp,  text_tok_b) != hipSuccess ||
        hipMalloc(&c->d_text_qkv,  text_qkv_b) != hipSuccess ||
        hipMalloc(&c->d_text_attn, text_tok_b) != hipSuccess ||
        hipMalloc(&c->d_text_mlp,  text_mlp_b) != hipSuccess) {
        fprintf(stderr, "sam3.1: text buffer alloc failed\n");
        cuda_sam3_1_destroy(c); return NULL;
    }

    /* language_backbone.resizer: Linear(1024 -> 256) that projects ln_final
     * tokens into DETR's d_model (256). This is NOT the OpenAI-CLIP
     * `text_projection` (1024 -> 512, used for similarity heads); the DETR
     * path consumes the resizer output.  Weight layout is (out=256, in=1024). */
    c->w_text_proj_w = upload_f16(c->st, "backbone.language_backbone.resizer.weight", NULL);
    c->w_text_proj_b = upload_f32(c->st, "backbone.language_backbone.resizer.bias", NULL);
    if (!c->w_text_proj_w || !c->w_text_proj_b) {
        fprintf(stderr, "sam3.1: missing language_backbone.resizer weights\n");
        cuda_sam3_1_destroy(c); return NULL;
    }

    /* DETR encoder layers. */
    if (c->verbose) fprintf(stderr, "sam3.1: loading DETR encoder (%d layers) ...\n", S3_DETR_LAYERS);
    for (int li = 0; li < S3_DETR_LAYERS; li++) {
        char k[192], wk[192], bk[192];
        #define DK(suf) (snprintf(k, sizeof(k), "transformer.encoder.layers.%d." suf, li), k)
        #define DKWB(path, w, b) do { \
            snprintf(wk, sizeof(wk), "transformer.encoder.layers.%d." path ".%s", li, w); \
            snprintf(bk, sizeof(bk), "transformer.encoder.layers.%d." path ".%s", li, b); \
        } while (0)
        c->detr_enc[li].ln1_w = upload_f32(c->st, DK("norm1.weight"), NULL);
        c->detr_enc[li].ln1_b = upload_f32(c->st, DK("norm1.bias"),   NULL);
        c->detr_enc[li].ln2_w = upload_f32(c->st, DK("norm2.weight"), NULL);
        c->detr_enc[li].ln2_b = upload_f32(c->st, DK("norm2.bias"),   NULL);
        c->detr_enc[li].ln3_w = upload_f32(c->st, DK("norm3.weight"), NULL);
        c->detr_enc[li].ln3_b = upload_f32(c->st, DK("norm3.bias"),   NULL);

        /* Self-attn: fused in_proj_weight (3D, D) / in_proj_bias (3D,). */
        DKWB("self_attn", "in_proj_weight", "in_proj_bias");
        if (load_fused_qkv(c->st, wk, bk, &c->detr_enc[li].qkv_w, &c->detr_enc[li].qkv_b)) {
            cuda_sam3_1_destroy(c); return NULL;
        }
        c->detr_enc[li].o_w = upload_f16(c->st, DK("self_attn.out_proj.weight"), NULL);
        c->detr_enc[li].o_b = upload_f32(c->st, DK("self_attn.out_proj.bias"),   NULL);

        /* Cross-attn-image: fused in_proj → split into q/k/v. */
        DKWB("cross_attn_image", "in_proj_weight", "in_proj_bias");
        if (split_fused_qkv(c->st, wk, bk, S3_DETR_DIM,
                             &c->detr_enc[li].cq_w, &c->detr_enc[li].ck_w, &c->detr_enc[li].cv_w,
                             &c->detr_enc[li].cq_b, &c->detr_enc[li].ck_b, &c->detr_enc[li].cv_b)) {
            cuda_sam3_1_destroy(c); return NULL;
        }
        c->detr_enc[li].co_w = upload_f16(c->st, DK("cross_attn_image.out_proj.weight"), NULL);
        c->detr_enc[li].co_b = upload_f32(c->st, DK("cross_attn_image.out_proj.bias"),   NULL);

        /* MLP: linear1/linear2 (sam3's mlp.fc1/fc2 equivalent). */
        c->detr_enc[li].fc1_w = upload_f16(c->st, DK("linear1.weight"), NULL);
        c->detr_enc[li].fc1_b = upload_f32(c->st, DK("linear1.bias"),   NULL);
        c->detr_enc[li].fc2_w = upload_f16(c->st, DK("linear2.weight"), NULL);
        c->detr_enc[li].fc2_b = upload_f32(c->st, DK("linear2.bias"),   NULL);
        #undef DK
        #undef DKWB
        if (!c->detr_enc[li].o_w || !c->detr_enc[li].fc2_w) {
            fprintf(stderr, "sam3.1: detr layer %d missing weights\n", li);
            cuda_sam3_1_destroy(c); return NULL;
        }
    }

    /* DETR activation buffers. */
    {
        size_t N = S3_NTOK;
        size_t DD = S3_DETR_DIM;
        if (hipMalloc(&c->d_detr_vis,        N * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_detr_scratch,    N * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_detr_q,          N * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_detr_k,          N * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_detr_v,          N * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_detr_qkv,        N * 3 * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_detr_attn,       N * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_detr_mlp,        N * S3_DETR_MLP * 4) != hipSuccess ||
            hipMalloc(&c->d_detr_pos,        N * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_detr_text_pooled, (size_t)S3_T_CTX * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_text_mask_i32,   (size_t)S3_T_CTX * 4) != hipSuccess) {
            fprintf(stderr, "sam3.1: detr buffer alloc failed\n");
            cuda_sam3_1_destroy(c); return NULL;
        }
    }
    /* Build sine 2D pos embed (72x72, 128 feats/axis). */
    {
        const int H = S3_GRID, W = S3_GRID, np = S3_DETR_DIM / 2;
        const float scale = 2.0f * (float)M_PI;
        const float eps = 1e-6f;
        const float dy = (float)H + eps, dx = (float)W + eps;
        float *dim_t = (float *)malloc(np * 4);
        for (int d = 0; d < np; d++) {
            int e = 2 * (d / 2);
            dim_t[d] = powf(10000.0f, (float)e / (float)np);
        }
        float *pos = (float *)malloc((size_t)H * W * S3_DETR_DIM * 4);
        for (int y = 0; y < H; y++) {
            float ye = (float)(y + 1) / dy * scale;
            for (int x = 0; x < W; x++) {
                float xe = (float)(x + 1) / dx * scale;
                float *row = pos + ((size_t)y * W + x) * S3_DETR_DIM;
                float *py = row, *px = row + np;
                for (int d = 0; d < np; d++) {
                    float ey = ye / dim_t[d];
                    float ex = xe / dim_t[d];
                    py[d] = (d & 1) ? cosf(ey) : sinf(ey);
                    px[d] = (d & 1) ? cosf(ex) : sinf(ex);
                }
            }
        }
        hipMemcpy(c->d_detr_pos, pos, (size_t)H * W * S3_DETR_DIM * 4, hipMemcpyHostToDevice);
        free(pos); free(dim_t);
    }

    /* DETR decoder. */
    if (c->verbose) fprintf(stderr, "sam3.1: loading DETR decoder ...\n");
    {
        /* Static tensors. sam3 → sam3.1 rename map:
         *   detr_decoder                    → transformer.decoder
         *   output_layer_norm               → norm
         *   presence_layer_norm             → presence_token_out_norm
         *   box_head.layer{1,2,3}           → bbox_embed.layers.{0,1,2}
         *   presence_head.layer{1,2,3}      → presence_token_head.layers.{0,1,2}
         *   ref_point_head.layer{1,2}       → ref_point_head.layers.{0,1}
         *   box_rpb_embed_{x,y}.layer{1,2}  → boxRPB_embed_{x,y}.layers.{0,1}
         * Per-layer attention renames applied below. */
        c->dd_query_embed = upload_f32(c->st, "transformer.decoder.query_embed.weight", NULL);
        const float *rp = host_f32(c->st, "transformer.decoder.reference_points.weight");
        if (!c->dd_query_embed || !rp) { cuda_sam3_1_destroy(c); return NULL; }
        c->dd_ref_points_host = (float *)malloc(S3_DETR_Q * 4 * 4);
        memcpy(c->dd_ref_points_host, rp, S3_DETR_Q * 4 * 4);
        c->dd_presence_token = upload_f32(c->st, "transformer.decoder.presence_token.weight", NULL);
        c->dd_output_ln_w = upload_f32(c->st, "transformer.decoder.norm.weight", NULL);
        c->dd_output_ln_b = upload_f32(c->st, "transformer.decoder.norm.bias", NULL);
        c->dd_presence_ln_w = upload_f32(c->st, "transformer.decoder.presence_token_out_norm.weight", NULL);
        c->dd_presence_ln_b = upload_f32(c->st, "transformer.decoder.presence_token_out_norm.bias", NULL);
        if (!c->dd_presence_token || !c->dd_output_ln_w || !c->dd_presence_ln_w) {
            cuda_sam3_1_destroy(c); return NULL;
        }

        char k[192];
        /* bbox_embed: 256→256→256→4. */
        for (int i = 0; i < 3; i++) {
            snprintf(k, sizeof(k), "transformer.decoder.bbox_embed.layers.%d.weight", i);
            c->dd_box_w[i] = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "transformer.decoder.bbox_embed.layers.%d.bias", i);
            c->dd_box_b[i] = upload_f32(c->st, k, NULL);
            if (!c->dd_box_w[i] || !c->dd_box_b[i]) { cuda_sam3_1_destroy(c); return NULL; }
        }
        /* presence_token_head: 256→256→256→1. */
        for (int i = 0; i < 3; i++) {
            snprintf(k, sizeof(k), "transformer.decoder.presence_token_head.layers.%d.weight", i);
            c->dd_pres_w[i] = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "transformer.decoder.presence_token_head.layers.%d.bias", i);
            c->dd_pres_b[i] = upload_f32(c->st, k, NULL);
            if (!c->dd_pres_w[i] || !c->dd_pres_b[i]) { cuda_sam3_1_destroy(c); return NULL; }
        }
        /* ref_point_head: 512→256→256. */
        for (int i = 0; i < 2; i++) {
            snprintf(k, sizeof(k), "transformer.decoder.ref_point_head.layers.%d.weight", i);
            c->dd_rph_w[i] = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "transformer.decoder.ref_point_head.layers.%d.bias", i);
            c->dd_rph_b[i] = upload_f32(c->st, k, NULL);
            if (!c->dd_rph_w[i] || !c->dd_rph_b[i]) { cuda_sam3_1_destroy(c); return NULL; }
        }
        /* boxRPB_embed_{x,y}: 2→256→8. */
        for (int i = 0; i < 2; i++) {
            snprintf(k, sizeof(k), "transformer.decoder.boxRPB_embed_x.layers.%d.weight", i);
            c->dd_rpbx_w[i] = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "transformer.decoder.boxRPB_embed_x.layers.%d.bias", i);
            c->dd_rpbx_b[i] = upload_f32(c->st, k, NULL);
            snprintf(k, sizeof(k), "transformer.decoder.boxRPB_embed_y.layers.%d.weight", i);
            c->dd_rpby_w[i] = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "transformer.decoder.boxRPB_embed_y.layers.%d.bias", i);
            c->dd_rpby_b[i] = upload_f32(c->st, k, NULL);
            if (!c->dd_rpbx_w[i] || !c->dd_rpby_w[i]) { cuda_sam3_1_destroy(c); return NULL; }
        }

        /* 6 decoder layers.
         * sam3 → sam3.1 per-layer attention names:
         *   self_attn           → self_attn         (fused in_proj)
         *   text_cross_attn     → ca_text           (fused in_proj)
         *   vision_cross_attn   → cross_attn        (fused in_proj)
         *   self_attn_layer_norm         → norm1
         *   text_cross_attn_layer_norm   → catext_norm
         *   vision_cross_attn_layer_norm → norm2
         *   mlp_layer_norm               → norm3
         *   mlp.fc1/fc2                  → linear1/linear2 */
        #define DLOAD_F16(dst, fmt, ...) do { \
            snprintf(k, sizeof(k), fmt, __VA_ARGS__); \
            dst = upload_f16(c->st, k, NULL); \
            if (!dst) { fprintf(stderr, "sam3.1: missing %s\n", k); cuda_sam3_1_destroy(c); return NULL; } \
        } while (0)
        #define DLOAD_F32(dst, fmt, ...) do { \
            snprintf(k, sizeof(k), fmt, __VA_ARGS__); \
            dst = upload_f32(c->st, k, NULL); \
            if (!dst) { fprintf(stderr, "sam3.1: missing %s\n", k); cuda_sam3_1_destroy(c); return NULL; } \
        } while (0)
        for (int li = 0; li < S3_DETR_LAYERS; li++) {
            #define LOAD_FUSED_ATT(P, dq, dk, dv, dO, dqb, dkb, dvb, dob) do { \
                char wk[192], bk[192]; \
                snprintf(wk, sizeof(wk), "transformer.decoder.layers.%d." P ".in_proj_weight", li); \
                snprintf(bk, sizeof(bk), "transformer.decoder.layers.%d." P ".in_proj_bias",   li); \
                if (split_fused_qkv(c->st, wk, bk, S3_DETR_DIM, \
                                     &c->dd[li].dq, &c->dd[li].dk, &c->dd[li].dv, \
                                     &c->dd[li].dqb, &c->dd[li].dkb, &c->dd[li].dvb)) { \
                    fprintf(stderr, "sam3.1: decoder layer %d " P " in_proj split failed\n", li); \
                    cuda_sam3_1_destroy(c); return NULL; \
                } \
                DLOAD_F16(c->dd[li].dO,  "transformer.decoder.layers.%d." P ".out_proj.weight", li); \
                DLOAD_F32(c->dd[li].dob, "transformer.decoder.layers.%d." P ".out_proj.bias",   li); \
            } while (0)
            LOAD_FUSED_ATT("self_attn",  sa_q, sa_k, sa_v, sa_o, sa_qb, sa_kb, sa_vb, sa_ob);
            DLOAD_F32(c->dd[li].sa_ln_w, "transformer.decoder.layers.%d.norm1.weight", li);
            DLOAD_F32(c->dd[li].sa_ln_b, "transformer.decoder.layers.%d.norm1.bias",   li);

            LOAD_FUSED_ATT("ca_text",    ta_q, ta_k, ta_v, ta_o, ta_qb, ta_kb, ta_vb, ta_ob);
            DLOAD_F32(c->dd[li].ta_ln_w, "transformer.decoder.layers.%d.catext_norm.weight", li);
            DLOAD_F32(c->dd[li].ta_ln_b, "transformer.decoder.layers.%d.catext_norm.bias",   li);

            LOAD_FUSED_ATT("cross_attn", va_q, va_k, va_v, va_o, va_qb, va_kb, va_vb, va_ob);
            DLOAD_F32(c->dd[li].va_ln_w, "transformer.decoder.layers.%d.norm2.weight", li);
            DLOAD_F32(c->dd[li].va_ln_b, "transformer.decoder.layers.%d.norm2.bias",   li);
            #undef LOAD_FUSED_ATT

            DLOAD_F16(c->dd[li].fc1,   "transformer.decoder.layers.%d.linear1.weight", li);
            DLOAD_F32(c->dd[li].fc1_b, "transformer.decoder.layers.%d.linear1.bias",   li);
            DLOAD_F16(c->dd[li].fc2,   "transformer.decoder.layers.%d.linear2.weight", li);
            DLOAD_F32(c->dd[li].fc2_b, "transformer.decoder.layers.%d.linear2.bias",   li);
            DLOAD_F32(c->dd[li].mlp_ln_w, "transformer.decoder.layers.%d.norm3.weight", li);
            DLOAD_F32(c->dd[li].mlp_ln_b, "transformer.decoder.layers.%d.norm3.bias",   li);
        }
        #undef DLOAD_F16
        #undef DLOAD_F32

        /* Runtime buffers. */
        size_t DD = S3_DETR_DIM, Nt = S3_DETR_QP1, Nq = S3_DETR_Q, N = S3_NTOK;
        if (hipMalloc(&c->d_dd_hs,      Nt * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_qpos,    Nt * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_hs_in,   Nt * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_q,       Nt * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_k,       N  * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_v,       N  * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_attn,    Nt * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_mlp,     Nt * S3_DETR_MLP * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_scratch, N  * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_kin,     N  * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_rpb,     (size_t)S3_DETR_HEADS * Nt * N * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_sine_box, Nq * 512 * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_rph_hidden, Nq * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_box_hidden, Nq * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_pres_hidden, DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_qh,       Nq * DD * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_delta,    Nq * 4 * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_presence, 4) != hipSuccess ||
            hipMalloc(&c->d_dd_dx_log,   Nq * S3_GRID * 2 * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_dy_log,   Nq * S3_GRID * 2 * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_ex,       Nq * S3_GRID * S3_DETR_HEADS * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_ey,       Nq * S3_GRID * S3_DETR_HEADS * 4) != hipSuccess ||
            hipMalloc(&c->d_dd_inter,
                      (size_t)S3_DETR_LAYERS * Nq * DD * 4) != hipSuccess) {
            fprintf(stderr, "sam3.1: decoder buffer alloc failed\n");
            cuda_sam3_1_destroy(c); return NULL;
        }
    }

    /* dot_product_scoring weights + runtime buffers. */
    if (c->verbose) fprintf(stderr, "sam3.1: loading dot_product_scoring ...\n");
    {
        const int DD = S3_DETR_DIM, T = S3_T_CTX, MLP = S3_DETR_MLP, Nq = S3_DETR_Q;
        /* sam3.1: dot_prod_scoring (singular). text_mlp/text_proj/query_proj
         * renamed to prompt_mlp/prompt_proj/hs_proj (pointer semantics same). */
        c->ds_tmlp_w[0] = upload_f16(c->st, "dot_prod_scoring.prompt_mlp.layers.0.weight", NULL);
        c->ds_tmlp_b[0] = upload_f32(c->st, "dot_prod_scoring.prompt_mlp.layers.0.bias", NULL);
        c->ds_tmlp_w[1] = upload_f16(c->st, "dot_prod_scoring.prompt_mlp.layers.1.weight", NULL);
        c->ds_tmlp_b[1] = upload_f32(c->st, "dot_prod_scoring.prompt_mlp.layers.1.bias", NULL);
        c->ds_tnorm_w  = upload_f32(c->st, "dot_prod_scoring.prompt_mlp.out_norm.weight", NULL);
        c->ds_tnorm_b  = upload_f32(c->st, "dot_prod_scoring.prompt_mlp.out_norm.bias", NULL);
        c->ds_tproj_w  = upload_f16(c->st, "dot_prod_scoring.prompt_proj.weight", NULL);
        c->ds_tproj_b  = upload_f32(c->st, "dot_prod_scoring.prompt_proj.bias", NULL);
        c->ds_qproj_w  = upload_f16(c->st, "dot_prod_scoring.hs_proj.weight", NULL);
        c->ds_qproj_b  = upload_f32(c->st, "dot_prod_scoring.hs_proj.bias", NULL);
        if (!c->ds_tmlp_w[0] || !c->ds_tnorm_w || !c->ds_tproj_w || !c->ds_qproj_w) {
            fprintf(stderr, "sam3.1: failed dot_product_scoring load\n");
            cuda_sam3_1_destroy(c); return NULL;
        }
        if (hipMalloc(&c->d_ds_tmlp_h,   T * MLP * 4)  != hipSuccess ||
            hipMalloc(&c->d_ds_text_h,   T * DD  * 4)  != hipSuccess ||
            hipMalloc(&c->d_ds_text_ln,  T * DD  * 4)  != hipSuccess ||
            hipMalloc(&c->d_ds_pooled,   DD * 4)        != hipSuccess ||
            hipMalloc(&c->d_ds_proj_text,DD * 4)        != hipSuccess ||
            hipMalloc(&c->d_ds_qproj,    Nq * DD * 4)   != hipSuccess ||
            hipMalloc(&c->d_ds_scores_li,Nq * 4)        != hipSuccess) {
            fprintf(stderr, "sam3.1: dot_product_scoring buffer alloc failed\n");
            cuda_sam3_1_destroy(c); return NULL;
        }
    }

    /* Mask decoder weights + runtime buffers. */
    if (c->verbose) fprintf(stderr, "sam3.1: loading mask_decoder ...\n");
    {
        const int DD = S3_DETR_DIM, T = S3_T_CTX, N = S3_NTOK, Nq = S3_DETR_Q;
        const int H2 = 72, W2 = 72, H1 = 144, W1 = 144, H0 = 288, W0 = 288;
        /* sam3 → sam3.1 rename map:
         *   mask_decoder                → segmentation_head
         *   prompt_cross_attn.{q,k,v,o}_proj → cross_attend_prompt (fused in_proj)
         *   prompt_cross_attn_norm      → cross_attn_norm
         *   mask_embedder.layers        → mask_predictor.mask_embed.layers
         *   instance_projection         → instance_seg_head
         *   semantic_projection         → semantic_seg_head
         * sam3.1 has 3 pixel_decoder conv_layers (layer 2 currently unused in
         * the sam3 forward path — loading only 2 for now). */
        if (split_fused_qkv(c->st,
                             "segmentation_head.cross_attend_prompt.in_proj_weight",
                             "segmentation_head.cross_attend_prompt.in_proj_bias",
                             S3_DETR_DIM,
                             &c->md_pca_q_w, &c->md_pca_k_w, &c->md_pca_v_w,
                             &c->md_pca_q_b, &c->md_pca_k_b, &c->md_pca_v_b)) {
            fprintf(stderr, "sam3.1: failed cross_attend_prompt split\n");
            cuda_sam3_1_destroy(c); return NULL;
        }
        c->md_pca_o_w = upload_f16(c->st, "segmentation_head.cross_attend_prompt.out_proj.weight", NULL);
        c->md_pca_o_b = upload_f32(c->st, "segmentation_head.cross_attend_prompt.out_proj.bias", NULL);
        c->md_pca_ln_w= upload_f32(c->st, "segmentation_head.cross_attn_norm.weight", NULL);
        c->md_pca_ln_b= upload_f32(c->st, "segmentation_head.cross_attn_norm.bias", NULL);
        for (int i = 0; i < 2; i++) {
            char k[192];
            snprintf(k, sizeof(k), "segmentation_head.pixel_decoder.conv_layers.%d.weight", i);
            c->md_conv_w[i] = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "segmentation_head.pixel_decoder.conv_layers.%d.bias", i);
            c->md_conv_b[i] = upload_f32(c->st, k, NULL);
            snprintf(k, sizeof(k), "segmentation_head.pixel_decoder.norms.%d.weight", i);
            c->md_gn_w[i]   = upload_f32(c->st, k, NULL);
            snprintf(k, sizeof(k), "segmentation_head.pixel_decoder.norms.%d.bias", i);
            c->md_gn_b[i]   = upload_f32(c->st, k, NULL);
        }
        for (int i = 0; i < 3; i++) {
            char k[192];
            snprintf(k, sizeof(k), "segmentation_head.mask_predictor.mask_embed.layers.%d.weight", i);
            c->md_me_w[i] = upload_f16(c->st, k, NULL);
            snprintf(k, sizeof(k), "segmentation_head.mask_predictor.mask_embed.layers.%d.bias", i);
            c->md_me_b[i] = upload_f32(c->st, k, NULL);
        }
        c->md_ip_w = upload_f16(c->st, "segmentation_head.instance_seg_head.weight", NULL);
        c->md_ip_b = upload_f32(c->st, "segmentation_head.instance_seg_head.bias", NULL);
        c->md_sp_w = upload_f16(c->st, "segmentation_head.semantic_seg_head.weight", NULL);
        c->md_sp_b = upload_f32(c->st, "segmentation_head.semantic_seg_head.bias", NULL);
        if (!c->md_pca_q_w || !c->md_conv_w[0] || !c->md_me_w[0] ||
            !c->md_ip_w || !c->md_sp_w) {
            fprintf(stderr, "sam3.1: failed mask_decoder load\n");
            cuda_sam3_1_destroy(c); return NULL;
        }
        if (hipMalloc(&c->d_md_enc_ln,    N  * DD * 4)         != hipSuccess ||
            hipMalloc(&c->d_md_enc_mod,   N  * DD * 4)         != hipSuccess ||
            hipMalloc(&c->d_md_q,         N  * DD * 4)         != hipSuccess ||
            hipMalloc(&c->d_md_k,         T  * DD * 4)         != hipSuccess ||
            hipMalloc(&c->d_md_v,         T  * DD * 4)         != hipSuccess ||
            hipMalloc(&c->d_md_attn,      N  * DD * 4)         != hipSuccess ||
            hipMalloc(&c->d_md_o,         N  * DD * 4)         != hipSuccess ||
            hipMalloc(&c->d_md_start,    (size_t)DD*H2*W2 * 4) != hipSuccess ||
            hipMalloc(&c->d_md_prev,     (size_t)DD*H1*W1 * 4) != hipSuccess ||
            hipMalloc(&c->d_md_cout,     (size_t)DD*H1*W1 * 4) != hipSuccess ||
            hipMalloc(&c->d_md_up,       (size_t)DD*H0*W0 * 4) != hipSuccess ||
            hipMalloc(&c->d_md_pixel,    (size_t)DD*H0*W0 * 4) != hipSuccess ||
            hipMalloc(&c->d_md_instance, (size_t)DD*H0*W0 * 4) != hipSuccess ||
            hipMalloc(&c->d_md_semantic, (size_t)1 *H0*W0 * 4) != hipSuccess ||
            hipMalloc(&c->d_md_me_hid,    Nq * DD * 4)         != hipSuccess ||
            hipMalloc(&c->d_md_mask_emb,  Nq * DD * 4)         != hipSuccess ||
            hipMalloc(&c->d_md_pred_masks,(size_t)Nq*H0*W0*4)  != hipSuccess) {
            fprintf(stderr, "sam3.1: mask_decoder buffer alloc failed\n");
            cuda_sam3_1_destroy(c); return NULL;
        }
    }

    if (c->verbose) fprintf(stderr, "sam3.1: ready (patch+pos ready, 32 ViT blocks loaded)\n");
    return c;
}

void cuda_sam3_1_destroy(cuda_sam3_1_ctx *c) {
    if (!c) return;
    if (c->st) safetensors_close(c->st);
    if (c->mod) hipModuleUnload(c->mod);
    free(c);
}

/* ==== forward ==== */

static int run_patch_pos(cuda_sam3_1_ctx *c) {
    struct __attribute__((packed)) { void *out; const void *w, *img; int grid, img_size, D; } pe =
        { c->d_tok, c->w_patch, c->d_img_f32, S3_GRID, S3_IMG, S3_DIM };
    if (launch(c->fn_patch, S3_GRID, S3_GRID, 1, 128, 1, 1, 0, &pe, sizeof(pe))) return -1;
    /* sam3.1 uses 2D RoPE inside each ViT block — no learned pos_embed at the
     * trunk entrance. w_pos is NULL; skip pos_add entirely. */
    if (c->w_pos) {
        struct __attribute__((packed)) { void *t; const void *p; int g, P, D; } pp =
            { c->d_tok, c->w_pos, S3_GRID, S3_PRE, S3_DIM };
        if (launch(c->fn_pos_add, S3_GRID, S3_GRID, 1, 256, 1, 1, 0, &pp, sizeof(pp))) return -1;
    }
    hipDeviceSynchronize();
    c->ready = 1;
    c->vit_started = 0;
    c->last_block = -1;
    return 0;
}

int cuda_sam3_1_set_image(cuda_sam3_1_ctx *c, const uint8_t *rgb, int h, int w) {
    if (!c || !rgb) return -1;
    size_t bytes = (size_t)h * w * 3;
    if (bytes > c->img_u8_cap) {
        if (c->d_img_u8) hipFree(c->d_img_u8);
        if (hipMalloc(&c->d_img_u8, bytes) != hipSuccess) return -1;
        c->img_u8_cap = bytes;
    }
    hipMemcpy(c->d_img_u8, rgb, bytes, hipMemcpyHostToDevice);
    /* SAM 3 preprocess: mean=0.5 std=0.5 (px/127.5 - 1). */
    struct __attribute__((packed)) {
        void *dst; const void *src; int sw, sh, dw, dh;
        float m0, m1, m2, i0, i1, i2;
    } rp = { c->d_img_f32, c->d_img_u8, w, h, S3_IMG, S3_IMG,
             0.5f, 0.5f, 0.5f, 2.0f, 2.0f, 2.0f };
    int total = S3_IMG * S3_IMG;
    if (launch(c->fn_resize, (total + 255) / 256, 1, 1, 256, 1, 1, 0, &rp, sizeof(rp))) return -1;
    return run_patch_pos(c);
}

int cuda_sam3_1_set_pixel_values(cuda_sam3_1_ctx *c, const float *pv) {
    if (!c || !pv) return -1;
    size_t bytes = (size_t)3 * S3_IMG * S3_IMG * 4;
    hipMemcpy(c->d_img_f32, pv, bytes, hipMemcpyHostToDevice);
    return run_patch_pos(c);
}

/* One ViT block forward. x lives in d_tok (residual stream). */
static int block_forward(cuda_sam3_1_ctx *c, int bi) {
    s3_block *b = &c->blk[bi];
    /* LN1: tmp = ln1(tok) */
    if (r_ln(c, c->d_tmp, c->d_tok, b->norm1_w, b->norm1_b, S3_NTOK, S3_DIM)) return -1;

    const void *rope_cos = b->is_global ? c->d_rope_glb_cos : c->d_rope_win_cos;
    const void *rope_sin = b->is_global ? c->d_rope_glb_sin : c->d_rope_win_sin;
    int rope_seq = b->is_global ? S3_NTOK : S3_WINTOK;
    void *attn_in;

    if (b->is_global) {
        attn_in = c->d_tmp;   /* full 5184 tokens, spatial order */
    } else {
        /* window_partition: d_tmp → d_win (9 × 576, 1024) */
        if (r_part(c, c->d_win, c->d_tmp, S3_GRID, S3_GRID, S3_DIM, S3_WIN)) return -1;
        attn_in = c->d_win;
    }

    /* QKV GEMM: (NTOK, 3D) = attn_in @ qkv_w^T + qkv_b */
    if (r_gemm(c, c->d_qkv, b->qkv_w, attn_in, b->qkv_b, 3 * S3_DIM, S3_DIM, S3_NTOK)) return -1;

    /* RoPE on Q,K in-place, rope_seq-aware. */
    if (r_rope(c, c->d_qkv, rope_cos, rope_sin, rope_seq, S3_NTOK)) return -1;

    /* Attention per window: run 1 (global) or 9 (windowed) FA calls. */
    int windows = b->is_global ? 1 : S3_N_WIN;
    int nper = b->is_global ? S3_NTOK : S3_WINTOK;
    for (int w = 0; w < windows; w++) {
        void *qkv_w = (char *)c->d_qkv + (size_t)w * nper * 3 * S3_DIM * 4;
        void *out_w = (char *)c->d_attn + (size_t)w * nper * S3_DIM * 4;
        if (r_kvtx(c, c->d_Kt, c->d_Vt, qkv_w, nper)) return -1;
        if (r_fa(c, out_w, qkv_w, c->d_Kt, c->d_Vt, nper)) return -1;
    }

    /* O projection: attn_buf = o(attn) (N, D). Still in window-order if windowed. */
    if (r_gemm(c, c->d_tmp, b->o_w, c->d_attn, b->o_b, S3_DIM, S3_DIM, S3_NTOK)) return -1;

    /* Unpartition back to spatial order if windowed. */
    if (!b->is_global) {
        if (r_unpart(c, c->d_win, c->d_tmp, S3_GRID, S3_GRID, S3_DIM, S3_WIN)) return -1;
        if (r_add(c, c->d_tok, c->d_win, S3_NTOK * S3_DIM)) return -1;
    } else {
        if (r_add(c, c->d_tok, c->d_tmp, S3_NTOK * S3_DIM)) return -1;
    }

    /* MLP: LN2 → fc1 → GELU → fc2 → residual */
    if (r_ln(c, c->d_tmp, c->d_tok, b->norm2_w, b->norm2_b, S3_NTOK, S3_DIM)) return -1;
    if (r_gemm(c, c->d_mlp, b->fc1_w, c->d_tmp, b->fc1_b, S3_MLP, S3_DIM, S3_NTOK)) return -1;
    if (r_gelu(c, c->d_mlp, S3_NTOK * S3_MLP)) return -1;
    if (r_gemm(c, c->d_attn, b->fc2_w, c->d_mlp, b->fc2_b, S3_DIM, S3_MLP, S3_NTOK)) return -1;
    if (r_add(c, c->d_tok, c->d_attn, S3_NTOK * S3_DIM)) return -1;
    return 0;
}

int cuda_sam3_1_run_vit(cuda_sam3_1_ctx *c, int stop_at_block) {
    if (!c || !c->ready) return -1;
    if (stop_at_block < 0 || stop_at_block >= S3_NBLK) return -1;
    if (!c->vit_started) {
        /* Apply pre-block LN once: tmp = ln(tok); tok = tmp. */
        if (r_ln(c, c->d_tmp, c->d_tok, c->w_pre_ln_w, c->w_pre_ln_b, S3_NTOK, S3_DIM)) return -1;
        hipMemcpy(c->d_tok, c->d_tmp, (size_t)S3_NTOK * S3_DIM * 4, hipMemcpyDeviceToDevice);
        c->vit_started = 1;
    }
    int start = c->last_block + 1;
    for (int bi = start; bi <= stop_at_block; bi++) {
        if (block_forward(c, bi)) {
            fprintf(stderr, "sam3.1: block_forward(%d) failed\n", bi);
            return -1;
        }
        c->last_block = bi;
    }
    return 0;
}

/* ==== FPN neck ==== */

static int r_tok2nchw(cuda_sam3_1_ctx *c, void *dst, const void *src,
                        int H, int W, int D) {
    struct __attribute__((packed)) { void *d; const void *s; int H, W, D; } p = { dst, src, H, W, D };
    int total = H * W * D;
    return launch(c->fn_tok2nchw, (total+255)/256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_c1x1(cuda_sam3_1_ctx *c, void *y, const void *w, const void *b,
                    const void *x, int Ci, int Co, int H, int W) {
    struct __attribute__((packed)) { void *y; const void *w, *b, *x; int Ci, Co, H, W; } p = { y, w, b, x, Ci, Co, H, W };
    return launch(c->fn_c1x1, (W+15)/16, (H+15)/16, Co, 16, 16, 1, 0, &p, sizeof(p));
}
static int r_c3x3(cuda_sam3_1_ctx *c, void *y, const void *w, const void *b,
                    const void *x, int Ci, int Co, int H, int W) {
    struct __attribute__((packed)) { void *y; const void *w, *b, *x; int Ci, Co, H, W; } p = { y, w, b, x, Ci, Co, H, W };
    return launch(c->fn_c3x3, (W+15)/16, (H+15)/16, Co, 16, 16, 1, 0, &p, sizeof(p));
}
static int r_c1x1_f32(cuda_sam3_1_ctx *c, void *y, const void *w, const void *b,
                      const void *x, int Ci, int Co, int H, int W) {
    struct __attribute__((packed)) { void *y; const void *w, *b, *x; int Ci, Co, H, W; } p = { y, w, b, x, Ci, Co, H, W };
    return launch(c->fn_c1x1_f32, (W+15)/16, (H+15)/16, Co, 16, 16, 1, 0, &p, sizeof(p));
}
static int r_c3x3_f32(cuda_sam3_1_ctx *c, void *y, const void *w, const void *b,
                      const void *x, int Ci, int Co, int H, int W) {
    struct __attribute__((packed)) { void *y; const void *w, *b, *x; int Ci, Co, H, W; } p = { y, w, b, x, Ci, Co, H, W };
    return launch(c->fn_c3x3_f32, (W+15)/16, (H+15)/16, Co, 16, 16, 1, 0, &p, sizeof(p));
}
static int r_ct2(cuda_sam3_1_ctx *c, void *y, const void *w, const void *b,
                   const void *x, int Ci, int Co, int H, int W) {
    struct __attribute__((packed)) { void *y; const void *w, *b, *x; int Ci, Co, H, W; } p = { y, w, b, x, Ci, Co, H, W };
    int OH = H * 2, OW = W * 2;
    return launch(c->fn_ct2, (OW+15)/16, (OH+15)/16, Co, 16, 16, 1, 0, &p, sizeof(p));
}
static int r_mp2(cuda_sam3_1_ctx *c, void *y, const void *x, int C, int H, int W) {
    struct __attribute__((packed)) { void *y; const void *x; int C, H, W; } p = { y, x, C, H, W };
    int OH = H / 2, OW = W / 2;
    return launch(c->fn_mp2, (OW+15)/16, (OH+15)/16, C, 16, 16, 1, 0, &p, sizeof(p));
}

int cuda_sam3_1_run_fpn(cuda_sam3_1_ctx *c) {
    if (!c) return -1;
    if (c->last_block != S3_NBLK - 1) {
        fprintf(stderr, "sam3.1: run_fpn requires ViT through block 31\n");
        return -1;
    }
    /* Transpose d_tok (N, D) → NCHW (D, 72, 72) into d_mlp (enough room). */
    void *nchw = c->d_mlp;
    if (r_tok2nchw(c, nchw, c->d_tok, S3_GRID, S3_GRID, S3_DIM)) return -1;

    /* Scratch buffers; reuse activation buffers where possible. */
    size_t sc1_sz = (size_t)(S3_DIM / 2) * (S3_GRID * 2) * (S3_GRID * 2) * 4;
    size_t sc2_sz = (size_t)(S3_DIM / 4) * (S3_GRID * 4) * (S3_GRID * 4) * 4;
    void *d_sc1 = NULL, *d_sc2 = NULL;  /* allocated lazily per level */
    size_t p1_sz = (size_t)S3_FPN_DIM * (S3_GRID * 4) * (S3_GRID * 4) * 4;
    void *d_p1 = NULL;
    if (hipMalloc(&d_p1, p1_sz) != hipSuccess) return -1;

    for (int li = 0; li < S3_FPN_LEV; li++) {
        s3_fpn_layer *L = &c->fpn[li];
        void *scaled = NULL;
        int sc_c = S3_DIM, sc_h = S3_GRID, sc_w = S3_GRID;

        if (L->scale == 4.0f) {
            if (!d_sc1 && hipMalloc(&d_sc1, sc1_sz) != hipSuccess) goto fail;
            if (!d_sc2 && hipMalloc(&d_sc2, sc2_sz) != hipSuccess) goto fail;
            if (r_ct2(c, d_sc1, L->ct0_w, L->ct0_b, nchw,
                      L->ct0_in, L->ct0_out, S3_GRID, S3_GRID)) goto fail;
            if (r_gelu(c, d_sc1, L->ct0_out * S3_GRID*2 * S3_GRID*2)) goto fail;
            if (r_ct2(c, d_sc2, L->ct1_w, L->ct1_b, d_sc1,
                      L->ct1_in, L->ct1_out, S3_GRID*2, S3_GRID*2)) goto fail;
            scaled = d_sc2; sc_c = L->ct1_out; sc_h = S3_GRID*4; sc_w = S3_GRID*4;
        } else if (L->scale == 2.0f) {
            if (!d_sc1 && hipMalloc(&d_sc1, sc1_sz) != hipSuccess) goto fail;
            if (r_ct2(c, d_sc1, L->ct0_w, L->ct0_b, nchw,
                      L->ct0_in, L->ct0_out, S3_GRID, S3_GRID)) goto fail;
            scaled = d_sc1; sc_c = L->ct0_out; sc_h = S3_GRID*2; sc_w = S3_GRID*2;
        } else if (L->scale == 1.0f) {
            scaled = nchw; sc_c = S3_DIM; sc_h = S3_GRID; sc_w = S3_GRID;
        } else {
            /* 0.5: reuse d_sc1 for the pooled output. Allocate if not present. */
            if (!d_sc1 && hipMalloc(&d_sc1, sc1_sz) != hipSuccess) goto fail;
            if (r_mp2(c, d_sc1, nchw, S3_DIM, S3_GRID, S3_GRID)) goto fail;
            scaled = d_sc1; sc_c = S3_DIM; sc_h = S3_GRID/2; sc_w = S3_GRID/2;
        }
        /* proj1 1x1: inter_c → 256 (F32 weights to avoid F16 quant drift). */
        if (r_c1x1_f32(c, d_p1, L->p1_w, L->p1_b, scaled,
                       sc_c, S3_FPN_DIM, sc_h, sc_w)) goto fail;
        /* proj2 3x3 pad=1: 256 → 256 into final output (F32 weights). */
        if (r_c3x3_f32(c, L->d_out, L->p2_w, L->p2_b, d_p1,
                       S3_FPN_DIM, S3_FPN_DIM, sc_h, sc_w)) goto fail;
    }
    hipDeviceSynchronize();
    if (d_sc1) hipFree(d_sc1);
    if (d_sc2) hipFree(d_sc2);
    hipFree(d_p1);
    c->fpn_ready = 1;
    return 0;
fail:
    if (d_sc1) hipFree(d_sc1);
    if (d_sc2) hipFree(d_sc2);
    if (d_p1)  hipFree(d_p1);
    return -1;
}

int cuda_sam3_1_debug_override_detr_inputs(cuda_sam3_1_ctx *c,
                                           const float *fpn2_nchw,
                                           const float *text_tokens) {
    if (!c || !fpn2_nchw || !text_tokens) return -1;
    const s3_fpn_layer *L2 = &c->fpn[2];
    size_t vsz = (size_t)S3_FPN_DIM * L2->out_h * L2->out_w * 4;
    if (hipMemcpy(L2->d_out, fpn2_nchw, vsz, hipMemcpyHostToDevice) != hipSuccess)
        return -1;
    size_t tsz = (size_t)S3_T_CTX * S3_T_DIM * 4;
    if (hipMemcpy(c->d_text_tok, text_tokens, tsz, hipMemcpyHostToDevice) != hipSuccess)
        return -1;
    c->fpn_ready = 1;
    c->text_done = 1;
    c->last_block = S3_NBLK - 1; /* satisfy run_detr_enc precondition */
    return 0;
}

int cuda_sam3_1_get_fpn(const cuda_sam3_1_ctx *c, int level, float *out,
                      int *out_c, int *out_h, int *out_w) {
    if (!c || !out || level < 0 || level >= S3_FPN_LEV) return -1;
    const s3_fpn_layer *L = &c->fpn[level];
    size_t sz = (size_t)S3_FPN_DIM * L->out_h * L->out_w * 4;
    hipMemcpy(out, L->d_out, sz, hipMemcpyDeviceToHost);
    if (out_c) *out_c = S3_FPN_DIM;
    if (out_h) *out_h = L->out_h;
    if (out_w) *out_w = L->out_w;
    return 0;
}

/* ==== CLIP text encoder ==== */

int cuda_sam3_1_set_input_ids(cuda_sam3_1_ctx *c, const int32_t *ids,
                            const int32_t *attn_mask) {
    if (!c || !ids) return -1;
    int saw_eos = 0;
    for (int t = 0; t < S3_T_CTX; t++) {
        int id = ids[t];
        if (id < 0 || id >= S3_T_VOCAB) return -1;
        c->text_ids[t] = id;
        if (attn_mask) {
            c->text_mask[t] = attn_mask[t] ? 1 : 0;
        } else {
            c->text_mask[t] = saw_eos ? 0 : 1;
            if (id == 49407) saw_eos = 1;
        }
    }
    hipMemcpy(c->d_text_mask_i32, c->text_mask, S3_T_CTX * 4, hipMemcpyHostToDevice);
    c->text_ids_ready = 1;
    c->text_done = 0;
    return 0;
}

static int r_gelu_erf(cuda_sam3_1_ctx *c, void *X, int n) {
    struct __attribute__((packed)) { void *X; int n; } p = { X, n };
    return launch(c->fn_gelu_erf, (n+255)/256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}

static int r_clip_causal(cuda_sam3_1_ctx *c, void *out, const void *qkv, int T) {
    float scale = 1.0f / sqrtf((float)S3_T_HD);
    struct __attribute__((packed)) { void *out; const void *qkv; int T, D, heads, hd; float sc; } p =
        { out, qkv, T, S3_T_DIM, S3_T_HEADS, S3_T_HD, scale };
    return launch(c->fn_clip_causal, S3_T_HEADS, 1, 1, T, 1, 1, 0, &p, sizeof(p));
}

int cuda_sam3_1_run_text(cuda_sam3_1_ctx *c) {
    if (!c || !c->text_ids_ready) return -1;
    const int T = S3_T_CTX, D = S3_T_DIM;
    /* Gather token embeddings + add position embeddings on host, upload. */
    float *host = (float *)malloc((size_t)T * D * 4);
    if (!host) return -1;
    for (int t = 0; t < T; t++) {
        const float *te = c->h_tok_embed + (size_t)c->text_ids[t] * D;
        const float *pe = c->h_pos_embed + (size_t)t * D;
        float *x = host + (size_t)t * D;
        for (int d = 0; d < D; d++) x[d] = te[d] + pe[d];
    }
    hipMemcpy(c->d_text_tok, host, (size_t)T * D * 4, hipMemcpyHostToDevice);
    free(host);

    for (int li = 0; li < S3_T_LAYERS; li++) {
        s3_block *b = &c->t_blk[li];
        /* LN1 → tmp */
        if (r_ln_eps(c, c->d_text_tmp, c->d_text_tok, b->norm1_w, b->norm1_b,
                      T, D, S3_T_EPS)) return -1;
        /* QKV gemm → qkv (T, 3D) */
        if (r_gemm(c, c->d_text_qkv, b->qkv_w, c->d_text_tmp, b->qkv_b,
                   3 * D, D, T)) return -1;
        /* Causal attention → text_attn (T, D) */
        if (r_clip_causal(c, c->d_text_attn, c->d_text_qkv, T)) return -1;
        /* O projection → text_tmp (T, D) */
        if (r_gemm(c, c->d_text_tmp, b->o_w, c->d_text_attn, b->o_b,
                   D, D, T)) return -1;
        /* Residual: tok += tmp. */
        if (r_add(c, c->d_text_tok, c->d_text_tmp, T * D)) return -1;
        /* LN2 → tmp */
        if (r_ln_eps(c, c->d_text_tmp, c->d_text_tok, b->norm2_w, b->norm2_b,
                      T, D, S3_T_EPS)) return -1;
        /* fc1 → mlp (T, MLP) */
        if (r_gemm(c, c->d_text_mlp, b->fc1_w, c->d_text_tmp, b->fc1_b,
                   S3_T_MLP, D, T)) return -1;
        /* Exact GELU (erf). */
        if (r_gelu_erf(c, c->d_text_mlp, T * S3_T_MLP)) return -1;
        /* fc2 → text_attn (T, D) */
        if (r_gemm(c, c->d_text_attn, b->fc2_w, c->d_text_mlp, b->fc2_b,
                   D, S3_T_MLP, T)) return -1;
        /* Residual: tok += attn. */
        if (r_add(c, c->d_text_tok, c->d_text_attn, T * D)) return -1;
    }

    /* Final LN: tmp = ln(tok); tok = tmp. */
    if (r_ln_eps(c, c->d_text_tmp, c->d_text_tok, c->t_final_ln_w, c->t_final_ln_b,
                  T, D, S3_T_EPS)) return -1;
    hipMemcpy(c->d_text_tok, c->d_text_tmp, (size_t)T * D * 4,
              hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();
    c->text_done = 1;
    return 0;
}

int cuda_sam3_1_get_text_output(const cuda_sam3_1_ctx *c, float *out,
                               int *out_len, int *out_dim) {
    if (!c || !out) return -1;
    hipMemcpy(out, c->d_text_tok, (size_t)S3_T_CTX * S3_T_DIM * 4,
              hipMemcpyDeviceToHost);
    if (out_len) *out_len = S3_T_CTX;
    if (out_dim) *out_dim = S3_T_DIM;
    return 0;
}

/* ==== DETR encoder ==== */

static int r_nchw2tok(cuda_sam3_1_ctx *c, void *dst, const void *src,
                       int C, int H, int W) {
    struct __attribute__((packed)) { void *d; const void *s; int C, H, W; } p = { dst, src, C, H, W };
    int total = C * H * W;
    return launch(c->fn_nchw2tok, (total + 255) / 256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_add_pos(cuda_sam3_1_ctx *c, void *dst, const void *src, const void *pos, int n) {
    struct __attribute__((packed)) { void *d; const void *s, *p; int n; } p = { dst, src, pos, n };
    return launch(c->fn_add_pos, (n + 255) / 256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_relu(cuda_sam3_1_ctx *c, void *x, int n) {
    struct __attribute__((packed)) { void *x; int n; } p = { x, n };
    return launch(c->fn_relu, (n + 255) / 256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
/* Launch MHA with online softmax, head_dim=32. */
static int r_mha32_bias(cuda_sam3_1_ctx *c, void *out, const void *Q, const void *K,
                         const void *V, const void *key_mask, const void *bias_hqk,
                         int Nq, int Nk) {
    float scale = 1.0f / sqrtf((float)S3_DETR_HD);
    struct __attribute__((packed)) { void *out; const void *Q, *K, *V, *key_mask, *bias_hqk;
             int Nq, Nk, D, heads, hd; float sc; } p =
        { out, Q, K, V, key_mask, bias_hqk, Nq, Nk, S3_DETR_DIM, S3_DETR_HEADS, S3_DETR_HD, scale };
    unsigned shmem = S3_DETR_HD * sizeof(float);
    return launch(c->fn_mha32, S3_DETR_HEADS, Nq, 1, S3_DETR_HD, 1, 1, shmem,
                   &p, sizeof(p));
}
static int r_mha32(cuda_sam3_1_ctx *c, void *out, const void *Q, const void *K,
                    const void *V, const void *key_mask,
                    int Nq, int Nk) {
    return r_mha32_bias(c, out, Q, K, V, key_mask, NULL, Nq, Nk);
}

/* Extract Q or K slab from fused QKV (N, 3D) into (N, D). Uses a no-op
 * gemm-alternative: plain device memcpy with stride. We use a small launch
 * via the existing add kernel pattern. Simpler: per-token memcpy. */
static int extract_qkv_slab(cuda_sam3_1_ctx *c, void *dst, const void *src,
                             int N, int D, int slab_idx) {
    (void)c;
    const char *s = (const char *)src + (size_t)slab_idx * D * sizeof(float);
    /* Strided D2D copy: N tokens, stride 3D. */
    for (int t = 0; t < N; t++) {
        void *d = (char *)dst + (size_t)t * D * sizeof(float);
        const void *ss = s + (size_t)t * 3 * D * sizeof(float);
        if (hipMemcpyAsync(d, ss, (size_t)D * sizeof(float),
                            hipMemcpyDeviceToDevice, 0) != hipSuccess) return -1;
    }
    return 0;
}

static int detr_enc_layer(cuda_sam3_1_ctx *c, int li) {
    const int N = S3_NTOK;
    const int D = S3_DETR_DIM;
    const int T = S3_T_CTX;
    void *x = c->d_detr_vis;
    /* --- Self-attn --- */
    if (r_ln_eps(c, c->d_detr_scratch, x,
                  c->detr_enc[li].ln1_w, c->detr_enc[li].ln1_b,
                  N, D, S3_DETR_LN_EPS)) return -1;
    /* qk_in = scratch + pos (stored in d_detr_q as scratch area). */
    if (r_add_pos(c, c->d_detr_q, c->d_detr_scratch, c->d_detr_pos, N * D)) return -1;
    /* Fused QKV gemm on qk_in -> d_detr_qkv (N, 3D). */
    if (r_gemm(c, c->d_detr_qkv, c->detr_enc[li].qkv_w, c->d_detr_q,
               c->detr_enc[li].qkv_b, 3 * D, D, N)) return -1;
    /* Split Q, K into d_detr_q, d_detr_k. */
    if (extract_qkv_slab(c, c->d_detr_q, c->d_detr_qkv, N, D, 0)) return -1;
    if (extract_qkv_slab(c, c->d_detr_k, c->d_detr_qkv, N, D, 1)) return -1;
    /* Recompute V from LN(no-pos) using V-only weight (slab 2 of qkv_w).
     * F16 weight stride: 2*D*D elements. Bias offset: 2*D. */
    {
        const char *Vw = (const char *)c->detr_enc[li].qkv_w
                          + (size_t)2 * D * D * sizeof(uint16_t);
        const char *Vb = (const char *)c->detr_enc[li].qkv_b
                          + (size_t)2 * D * sizeof(float);
        if (r_gemm(c, c->d_detr_v, (const void *)Vw, c->d_detr_scratch,
                   (const void *)Vb, D, D, N)) return -1;
    }
    /* Self-attn into d_detr_attn. */
    if (r_mha32(c, c->d_detr_attn, c->d_detr_q, c->d_detr_k, c->d_detr_v,
                 NULL, N, N)) return -1;
    /* O proj → scratch, add to x. */
    if (r_gemm(c, c->d_detr_scratch, c->detr_enc[li].o_w, c->d_detr_attn,
               c->detr_enc[li].o_b, D, D, N)) return -1;
    if (r_add(c, x, c->d_detr_scratch, N * D)) return -1;

    /* --- Cross-attn to text --- */
    if (r_ln_eps(c, c->d_detr_scratch, x,
                  c->detr_enc[li].ln2_w, c->detr_enc[li].ln2_b,
                  N, D, S3_DETR_LN_EPS)) return -1;
    if (r_gemm(c, c->d_detr_q, c->detr_enc[li].cq_w, c->d_detr_scratch,
               c->detr_enc[li].cq_b, D, D, N)) return -1;
    if (r_gemm(c, c->d_detr_k, c->detr_enc[li].ck_w, c->d_detr_text_pooled,
               c->detr_enc[li].ck_b, D, D, T)) return -1;
    if (r_gemm(c, c->d_detr_v, c->detr_enc[li].cv_w, c->d_detr_text_pooled,
               c->detr_enc[li].cv_b, D, D, T)) return -1;
    if (r_mha32(c, c->d_detr_attn, c->d_detr_q, c->d_detr_k, c->d_detr_v,
                 c->d_text_mask_i32, N, T)) return -1;
    if (r_gemm(c, c->d_detr_scratch, c->detr_enc[li].co_w, c->d_detr_attn,
               c->detr_enc[li].co_b, D, D, N)) return -1;
    if (r_add(c, x, c->d_detr_scratch, N * D)) return -1;

    /* --- MLP --- */
    if (r_ln_eps(c, c->d_detr_scratch, x,
                  c->detr_enc[li].ln3_w, c->detr_enc[li].ln3_b,
                  N, D, S3_DETR_LN_EPS)) return -1;
    if (r_gemm(c, c->d_detr_mlp, c->detr_enc[li].fc1_w, c->d_detr_scratch,
               c->detr_enc[li].fc1_b, S3_DETR_MLP, D, N)) return -1;
    if (r_relu(c, c->d_detr_mlp, N * S3_DETR_MLP)) return -1;
    if (r_gemm(c, c->d_detr_scratch, c->detr_enc[li].fc2_w, c->d_detr_mlp,
               c->detr_enc[li].fc2_b, D, S3_DETR_MLP, N)) return -1;
    if (r_add(c, x, c->d_detr_scratch, N * D)) return -1;
    return 0;
}

int cuda_sam3_1_run_detr_enc(cuda_sam3_1_ctx *c) {
    if (!c) return -1;
    if (c->last_block != S3_NBLK - 1 || !c->fpn_ready || !c->text_done) {
        fprintf(stderr, "sam3.1: run_detr_enc needs ViT+FPN+text first\n");
        return -1;
    }
    const int N = S3_NTOK, D = S3_DETR_DIM, T = S3_T_CTX;
    /* Vision: FPN level 2 (256, 72, 72) NCHW -> (N, 256) tokens. */
    if (r_nchw2tok(c, c->d_detr_vis, c->fpn[2].d_out, D, S3_GRID, S3_GRID)) return -1;
    /* Text projection (1024 -> 256) on final LN'd text output. */
    if (r_gemm(c, c->d_detr_text_pooled, c->w_text_proj_w, c->d_text_tok,
               c->w_text_proj_b, D, S3_T_DIM, T)) return -1;
    int nl = S3_DETR_LAYERS;
    const char *env = getenv("SAM3_1_DETR_ENC_LAYERS");
    if (env) { int v = atoi(env); if (v >= 0 && v <= S3_DETR_LAYERS) nl = v; }
    for (int li = 0; li < nl; li++) {
        if (detr_enc_layer(c, li)) return -1;
    }
    hipDeviceSynchronize();
    c->detr_done = 1;
    return 0;
}

int cuda_sam3_1_get_detr_enc(const cuda_sam3_1_ctx *c, float *out, int *out_n, int *out_dim) {
    if (!c || !c->detr_done || !out) return -1;
    hipMemcpy(out, c->d_detr_vis, (size_t)S3_NTOK * S3_DETR_DIM * 4,
              hipMemcpyDeviceToHost);
    if (out_n) *out_n = S3_NTOK;
    if (out_dim) *out_dim = S3_DETR_DIM;
    return 0;
}

/* ==== DETR decoder ==== */

static float sig_f(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}
static float inv_sig_f(float x) {
    float c = x < 1e-5f ? 1e-5f : x;
    if (c > 1.0f - 1e-5f) c = 1.0f - 1e-5f;
    return logf(c / (1.0f - c));
}

/* Encode 4D boxes (cxcywh) to (Nq, 512) = cat(sin/cos[y,x,w,h]). Host side. */
static void encode_boxes(float *dst, const float *boxes, int Nq) {
    const int NF = 128;
    const float scale = 2.0f * (float)M_PI;
    float dim_t[128];
    for (int j = 0; j < NF; j++)
        dim_t[j] = powf(10000.0f, (float)(2 * (j / 2)) / (float)NF);
    const int order[4] = {1, 0, 2, 3};
    for (int q = 0; q < Nq; q++) {
        float *out = dst + (size_t)q * 4 * NF;
        for (int ci = 0; ci < 4; ci++) {
            int c = order[ci];
            float v = boxes[q * 4 + c] * scale;
            float *seg = out + ci * NF;
            for (int p = 0; p < NF / 2; p++) {
                float a = v / dim_t[2*p], b = v / dim_t[2*p + 1];
                seg[2*p]     = sinf(a);
                seg[2*p + 1] = cosf(b);
            }
        }
    }
}

/* Compute dx_log/dy_log (Nq, grid, 2) host, upload to device. */
static void compute_rpb_deltas(float *dx, float *dy, const float *boxes,
                                int Nq, int H, int W) {
    const float log2_8 = 3.0f;
    for (int q = 0; q < Nq; q++) {
        float cx = boxes[q*4+0], cy = boxes[q*4+1];
        float w  = boxes[q*4+2], h  = boxes[q*4+3];
        float x1 = cx - 0.5f*w, x2 = cx + 0.5f*w;
        float y1 = cy - 0.5f*h, y2 = cy + 0.5f*h;
        float *rx = dx + (size_t)q * W * 2;
        float *ry = dy + (size_t)q * H * 2;
        for (int x = 0; x < W; x++) {
            float cp = (float)x / (float)W;
            float d0 = (cp - x1) * 8.0f, d1 = (cp - x2) * 8.0f;
            rx[x*2+0] = (d0 >= 0 ? 1.f : -1.f) * log2f(fabsf(d0)+1.f) / log2_8;
            rx[x*2+1] = (d1 >= 0 ? 1.f : -1.f) * log2f(fabsf(d1)+1.f) / log2_8;
        }
        for (int y = 0; y < H; y++) {
            float cp = (float)y / (float)H;
            float d0 = (cp - y1) * 8.0f, d1 = (cp - y2) * 8.0f;
            ry[y*2+0] = (d0 >= 0 ? 1.f : -1.f) * log2f(fabsf(d0)+1.f) / log2_8;
            ry[y*2+1] = (d1 >= 0 ? 1.f : -1.f) * log2f(fabsf(d1)+1.f) / log2_8;
        }
    }
}

static int r_tok2chw(cuda_sam3_1_ctx *c, void *dst, const void *src,
                       int C, int H, int W) {
    struct __attribute__((packed)) { void *d; const void *s; int C, H, W; } p = { dst, src, C, H, W };
    int total = C * H * W;
    return launch(c->fn_tok2chw, (total + 255) / 256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_groupnorm(cuda_sam3_1_ctx *c, void *x, const void *w, const void *b,
                         int C, int H, int W, int G) {
    struct __attribute__((packed)) { void *x; const void *w, *b; int C, H, W, G; float eps; } p =
        { x, w, b, C, H, W, G, 1e-5f };
    return launch(c->fn_groupnorm, G, 1, 1, 256, 1, 1, 256 * 2 * 4, &p, sizeof(p));
}
static int r_upnn(cuda_sam3_1_ctx *c, void *dst, const void *src,
                    int C, int Hi, int Wi, int Ho, int Wo) {
    struct __attribute__((packed)) { void *d; const void *s; int C, Hi, Wi, Ho, Wo; } p =
        { dst, src, C, Hi, Wi, Ho, Wo };
    int total = C * Ho * Wo;
    return launch(c->fn_upnn, (total + 255) / 256, 1, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_einsum_qchw(cuda_sam3_1_ctx *c, void *out, const void *me,
                           const void *instance, int Nq, int D, int H, int W) {
    struct __attribute__((packed)) { void *o; const void *m, *i; int Nq, D, H, W; } p =
        { out, me, instance, Nq, D, H, W };
    int HW = H * W;
    return launch(c->fn_einsum_qchw, (HW + 255) / 256, Nq, 1, 256, 1, 1, 0, &p, sizeof(p));
}
static int r_rpb_assemble(cuda_sam3_1_ctx *c, void *rpb, const void *ex,
                           const void *ey, int Nq, int H, int W) {
    struct __attribute__((packed)) { void *rpb; const void *ex, *ey; int Nq, H, W, heads; } p =
        { rpb, ex, ey, Nq, H, W, S3_DETR_HEADS };
    int total = S3_DETR_HEADS * Nq * H * W;
    return launch(c->fn_rpb_assemble, (total + 255) / 256, 1, 1, 256, 1, 1, 0,
                   &p, sizeof(p));
}

static int dec_layer(cuda_sam3_1_ctx *c, int li) {
    const int Nt = S3_DETR_QP1, Nq = S3_DETR_Q, D = S3_DETR_DIM,
              MLP = S3_DETR_MLP, T = S3_T_CTX, N = S3_NTOK;
    void *hs = c->d_dd_hs;

    /* Self-attn (post-norm). */
    if (r_add_pos(c, c->d_dd_hs_in, hs, c->d_dd_qpos, Nt * D)) return -1;
    if (r_gemm(c, c->d_dd_q, c->dd[li].sa_q, c->d_dd_hs_in, c->dd[li].sa_qb, D, D, Nt)) return -1;
    if (r_gemm(c, c->d_dd_k, c->dd[li].sa_k, c->d_dd_hs_in, c->dd[li].sa_kb, D, D, Nt)) return -1;
    if (r_gemm(c, c->d_dd_v, c->dd[li].sa_v, hs,             c->dd[li].sa_vb, D, D, Nt)) return -1;
    if (r_mha32(c, c->d_dd_attn, c->d_dd_q, c->d_dd_k, c->d_dd_v, NULL, Nt, Nt)) return -1;
    if (r_gemm(c, c->d_dd_scratch, c->dd[li].sa_o, c->d_dd_attn, c->dd[li].sa_ob, D, D, Nt)) return -1;
    if (r_add(c, hs, c->d_dd_scratch, Nt * D)) return -1;
    if (r_ln_eps(c, c->d_dd_scratch, hs, c->dd[li].sa_ln_w, c->dd[li].sa_ln_b, Nt, D, S3_DETR_LN_EPS)) return -1;
    hipMemcpy(hs, c->d_dd_scratch, (size_t)Nt * D * 4, hipMemcpyDeviceToDevice);

    /* Text cross-attn. */
    if (r_add_pos(c, c->d_dd_hs_in, hs, c->d_dd_qpos, Nt * D)) return -1;
    if (r_gemm(c, c->d_dd_q, c->dd[li].ta_q, c->d_dd_hs_in,        c->dd[li].ta_qb, D, D, Nt)) return -1;
    if (r_gemm(c, c->d_dd_k, c->dd[li].ta_k, c->d_detr_text_pooled, c->dd[li].ta_kb, D, D, T)) return -1;
    if (r_gemm(c, c->d_dd_v, c->dd[li].ta_v, c->d_detr_text_pooled, c->dd[li].ta_vb, D, D, T)) return -1;
    if (r_mha32(c, c->d_dd_attn, c->d_dd_q, c->d_dd_k, c->d_dd_v, c->d_text_mask_i32, Nt, T)) return -1;
    if (r_gemm(c, c->d_dd_scratch, c->dd[li].ta_o, c->d_dd_attn, c->dd[li].ta_ob, D, D, Nt)) return -1;
    if (r_add(c, hs, c->d_dd_scratch, Nt * D)) return -1;
    if (r_ln_eps(c, c->d_dd_scratch, hs, c->dd[li].ta_ln_w, c->dd[li].ta_ln_b, Nt, D, S3_DETR_LN_EPS)) return -1;
    hipMemcpy(hs, c->d_dd_scratch, (size_t)Nt * D * 4, hipMemcpyDeviceToDevice);

    /* Vision cross-attn with RPB. kin = detr_vis + detr_pos (per-token add). */
    if (r_add_pos(c, c->d_dd_hs_in, hs, c->d_dd_qpos, Nt * D)) return -1;
    if (r_gemm(c, c->d_dd_q, c->dd[li].va_q, c->d_dd_hs_in, c->dd[li].va_qb, D, D, Nt)) return -1;
    if (r_add_pos(c, c->d_dd_kin, c->d_detr_vis, c->d_detr_pos, N * D)) return -1;
    if (r_gemm(c, c->d_dd_k, c->dd[li].va_k, c->d_dd_kin, c->dd[li].va_kb, D, D, N)) return -1;
    if (r_gemm(c, c->d_dd_v, c->dd[li].va_v, c->d_detr_vis, c->dd[li].va_vb, D, D, N)) return -1;
    if (r_mha32_bias(c, c->d_dd_attn, c->d_dd_q, c->d_dd_k, c->d_dd_v, NULL, c->d_dd_rpb, Nt, N)) return -1;
    if (r_gemm(c, c->d_dd_scratch, c->dd[li].va_o, c->d_dd_attn, c->dd[li].va_ob, D, D, Nt)) return -1;
    if (r_add(c, hs, c->d_dd_scratch, Nt * D)) return -1;
    if (r_ln_eps(c, c->d_dd_scratch, hs, c->dd[li].va_ln_w, c->dd[li].va_ln_b, Nt, D, S3_DETR_LN_EPS)) return -1;
    hipMemcpy(hs, c->d_dd_scratch, (size_t)Nt * D * 4, hipMemcpyDeviceToDevice);

    /* MLP (post-norm). */
    if (r_gemm(c, c->d_dd_mlp, c->dd[li].fc1, hs, c->dd[li].fc1_b, MLP, D, Nt)) return -1;
    if (r_relu(c, c->d_dd_mlp, Nt * MLP)) return -1;
    if (r_gemm(c, c->d_dd_scratch, c->dd[li].fc2, c->d_dd_mlp, c->dd[li].fc2_b, D, MLP, Nt)) return -1;
    if (r_add(c, hs, c->d_dd_scratch, Nt * D)) return -1;
    if (r_ln_eps(c, c->d_dd_scratch, hs, c->dd[li].mlp_ln_w, c->dd[li].mlp_ln_b, Nt, D, S3_DETR_LN_EPS)) return -1;
    hipMemcpy(hs, c->d_dd_scratch, (size_t)Nt * D * 4, hipMemcpyDeviceToDevice);
    return 0;
}

int cuda_sam3_1_run_detr_dec(cuda_sam3_1_ctx *c) {
    if (!c || !c->detr_done) return -1;
    const int Nt = S3_DETR_QP1, Nq = S3_DETR_Q, D = S3_DETR_DIM;
    const int H = S3_GRID, W = S3_GRID;

    /* Init ref_boxes = sigmoid(ref_points). */
    for (int i = 0; i < Nq * 4; i++)
        c->dd_ref_boxes[i] = sig_f(c->dd_ref_points_host[i]);

    /* Init hs = [presence_token; query_embed]. */
    hipMemcpy(c->d_dd_hs, c->dd_presence_token, D * 4, hipMemcpyDeviceToDevice);
    hipMemcpy((char *)c->d_dd_hs + D * 4, c->dd_query_embed, (size_t)Nq * D * 4,
              hipMemcpyDeviceToDevice);

    for (int li = 0; li < S3_DETR_LAYERS; li++) {
        if (c->verbose) fprintf(stderr, "  dec layer %d ...\n", li);
        /* Build qpos: [0; ref_point_head(encode_boxes(ref_boxes))] on device. */
        float sine_box_host[S3_DETR_Q * 512];
        encode_boxes(sine_box_host, c->dd_ref_boxes, Nq);
        hipMemcpy(c->d_dd_sine_box, sine_box_host, (size_t)Nq * 512 * 4,
                  hipMemcpyHostToDevice);
        /* layer 1: 512 → 256 with ReLU. */
        if (r_gemm(c, c->d_dd_rph_hidden, c->dd_rph_w[0], c->d_dd_sine_box,
                   c->dd_rph_b[0], D, 512, Nq)) return -1;
        if (r_relu(c, c->d_dd_rph_hidden, Nq * D)) return -1;
        /* layer 2: 256 → 256 into qpos[1..]. */
        void *qpos_rest = (char *)c->d_dd_qpos + D * 4;
        if (r_gemm(c, qpos_rest, c->dd_rph_w[1], c->d_dd_rph_hidden,
                   c->dd_rph_b[1], D, D, Nq)) return -1;
        /* Zero qpos row 0 (presence). */
        hipMemset(c->d_dd_qpos, 0, D * 4);

        /* RPB: host-compute dx/dy_log, upload, two tiny gemms, assemble. */
        float *dx_h = (float *)malloc((size_t)Nq * W * 2 * 4);
        float *dy_h = (float *)malloc((size_t)Nq * H * 2 * 4);
        compute_rpb_deltas(dx_h, dy_h, c->dd_ref_boxes, Nq, H, W);
        hipMemcpy(c->d_dd_dx_log, dx_h, (size_t)Nq * W * 2 * 4, hipMemcpyHostToDevice);
        hipMemcpy(c->d_dd_dy_log, dy_h, (size_t)Nq * H * 2 * 4, hipMemcpyHostToDevice);
        free(dx_h); free(dy_h);
        /* MLP x: (Nq*W, 2) -> hidden (Nq*W, 256) -> ex (Nq*W, 8). Reuse d_dd_rpb as
         * scratch — it's memset to zero below before rpb_assemble runs, so the
         * transient hx content is overwritten. d_dd_mlp is too small (1.6 MB vs
         * the 14.7 MB we need here: 200*72*256 floats). */
        void *hx = c->d_dd_rpb;
        if (r_gemm(c, hx, c->dd_rpbx_w[0], c->d_dd_dx_log, c->dd_rpbx_b[0],
                   D, 2, Nq * W)) return -1;
        if (r_relu(c, hx, Nq * W * D)) return -1;
        if (r_gemm(c, c->d_dd_ex, c->dd_rpbx_w[1], hx, c->dd_rpbx_b[1],
                   S3_DETR_HEADS, D, Nq * W)) return -1;
        if (r_gemm(c, hx, c->dd_rpby_w[0], c->d_dd_dy_log, c->dd_rpby_b[0],
                   D, 2, Nq * H)) return -1;
        if (r_relu(c, hx, Nq * H * D)) return -1;
        if (r_gemm(c, c->d_dd_ey, c->dd_rpby_w[1], hx, c->dd_rpby_b[1],
                   S3_DETR_HEADS, D, Nq * H)) return -1;
        /* Zero entire RPB; assemble kernel writes only Nq rows (presence stays 0). */
        hipMemset(c->d_dd_rpb, 0, (size_t)S3_DETR_HEADS * Nt * H * W * 4);
        if (r_rpb_assemble(c, c->d_dd_rpb, c->d_dd_ex, c->d_dd_ey, Nq, H, W)) return -1;

        /* Layer forward. */
        if (dec_layer(c, li)) return -1;

        /* Box refinement. qh = output_ln(hs[1..]); delta = box_head(qh). */
        void *hs_rest = (char *)c->d_dd_hs + D * 4;
        if (r_ln_eps(c, c->d_dd_qh, hs_rest, c->dd_output_ln_w, c->dd_output_ln_b,
                      Nq, D, S3_DETR_LN_EPS)) return -1;
        /* Save per-layer qh for dot_score + mask_embedder. */
        hipMemcpy((char *)c->d_dd_inter + (size_t)li * Nq * D * 4, c->d_dd_qh,
                  (size_t)Nq * D * 4, hipMemcpyDeviceToDevice);
        /* box_head 3-layer: 256->256->256->4 with ReLU between. */
        if (r_gemm(c, c->d_dd_box_hidden, c->dd_box_w[0], c->d_dd_qh,
                   c->dd_box_b[0], D, D, Nq)) return -1;
        if (r_relu(c, c->d_dd_box_hidden, Nq * D)) return -1;
        if (r_gemm(c, c->d_dd_rph_hidden, c->dd_box_w[1], c->d_dd_box_hidden,
                   c->dd_box_b[1], D, D, Nq)) return -1;
        if (r_relu(c, c->d_dd_rph_hidden, Nq * D)) return -1;
        if (r_gemm(c, c->d_dd_delta, c->dd_box_w[2], c->d_dd_rph_hidden,
                   c->dd_box_b[2], 4, D, Nq)) return -1;
        hipDeviceSynchronize();
        float delta_h[S3_DETR_Q * 4];
        hipMemcpy(delta_h, c->d_dd_delta, (size_t)Nq * 4 * 4, hipMemcpyDeviceToHost);
        for (int q = 0; q < Nq; q++) {
            for (int cc = 0; cc < 4; cc++) {
                float rb = c->dd_ref_boxes[q*4+cc];
                float inv = inv_sig_f(rb);
                c->dd_ref_boxes[q*4+cc] = sig_f(delta_h[q*4+cc] + inv);
            }
        }

        /* Presence logit: pl_in = presence_ln(hs[0]); MLP 256->256->256->1 -> clamp. */
        if (r_ln_eps(c, c->d_dd_pres_hidden, c->d_dd_hs,
                      c->dd_presence_ln_w, c->dd_presence_ln_b, 1, D, S3_DETR_LN_EPS)) return -1;
        if (r_gemm(c, c->d_dd_box_hidden, c->dd_pres_w[0], c->d_dd_pres_hidden,
                   c->dd_pres_b[0], D, D, 1)) return -1;
        if (r_relu(c, c->d_dd_box_hidden, D)) return -1;
        if (r_gemm(c, c->d_dd_rph_hidden, c->dd_pres_w[1], c->d_dd_box_hidden,
                   c->dd_pres_b[1], D, D, 1)) return -1;
        if (r_relu(c, c->d_dd_rph_hidden, D)) return -1;
        if (r_gemm(c, c->d_dd_presence, c->dd_pres_w[2], c->d_dd_rph_hidden,
                   c->dd_pres_b[2], 1, D, 1)) return -1;
        hipDeviceSynchronize();
        float pl; hipMemcpy(&pl, c->d_dd_presence, 4, hipMemcpyDeviceToHost);
        if (pl >  S3_DETR_PCLAMP) pl =  S3_DETR_PCLAMP;
        if (pl < -S3_DETR_PCLAMP) pl = -S3_DETR_PCLAMP;
        c->dd_presence_logits[li] = pl;
    }

    /* cxcywh -> xyxy. */
    for (int q = 0; q < Nq; q++) {
        float cx = c->dd_ref_boxes[q*4+0], cy = c->dd_ref_boxes[q*4+1];
        float w  = c->dd_ref_boxes[q*4+2], h  = c->dd_ref_boxes[q*4+3];
        c->dd_pred_boxes[q*4+0] = cx - 0.5f*w;
        c->dd_pred_boxes[q*4+1] = cy - 0.5f*h;
        c->dd_pred_boxes[q*4+2] = cx + 0.5f*w;
        c->dd_pred_boxes[q*4+3] = cy + 0.5f*h;
    }

    c->dd_done = 1;
    return 0;
}

int cuda_sam3_1_get_detr_dec_boxes(const cuda_sam3_1_ctx *c, float *out) {
    if (!c || !c->dd_done || !out) return -1;
    memcpy(out, c->dd_pred_boxes, S3_DETR_Q * 4 * 4);
    return 0;
}
int cuda_sam3_1_get_detr_dec_presence(const cuda_sam3_1_ctx *c, float *out) {
    if (!c || !c->dd_done || !out) return -1;
    memcpy(out, c->dd_presence_logits, S3_DETR_LAYERS * 4);
    return 0;
}
int cuda_sam3_1_get_detr_dec_hidden(const cuda_sam3_1_ctx *c, float *out) {
    if (!c || !c->dd_done || !out) return -1;
    hipMemcpy(out, c->d_dd_qh, (size_t)S3_DETR_Q * S3_DETR_DIM * 4,
              hipMemcpyDeviceToHost);
    return 0;
}

int cuda_sam3_1_run_dot_score(cuda_sam3_1_ctx *c) {
    if (!c || !c->dd_done) return -1;
    const int D = S3_DETR_DIM, T = S3_T_CTX, MLP = S3_DETR_MLP,
              Nq = S3_DETR_Q, L = S3_DETR_LAYERS;
    /* text_mlp (2-layer) + residual + LN on d_detr_text_pooled (T, D). */
    if (r_gemm(c, c->d_ds_tmlp_h, c->ds_tmlp_w[0], c->d_detr_text_pooled,
               c->ds_tmlp_b[0], MLP, D, T)) return -1;
    if (r_relu(c, c->d_ds_tmlp_h, T * MLP)) return -1;
    if (r_gemm(c, c->d_ds_text_h, c->ds_tmlp_w[1], c->d_ds_tmlp_h,
               c->ds_tmlp_b[1], D, MLP, T)) return -1;
    if (r_add(c, c->d_ds_text_h, c->d_detr_text_pooled, T * D)) return -1;
    if (r_ln_eps(c, c->d_ds_text_ln, c->d_ds_text_h,
                 c->ds_tnorm_w, c->ds_tnorm_b, T, D, S3_DETR_LN_EPS)) return -1;

    /* Masked mean pool over valid tokens -> (D,). Small: T=32, D=256. */
    float text_ln_h[S3_T_CTX * S3_DETR_DIM];
    hipMemcpy(text_ln_h, c->d_ds_text_ln, (size_t)T * D * 4, hipMemcpyDeviceToHost);
    float pooled[S3_DETR_DIM] = {0};
    int nv = 0;
    for (int t = 0; t < T; t++) {
        if (c->text_mask[t]) {
            for (int d = 0; d < D; d++) pooled[d] += text_ln_h[t * D + d];
            nv++;
        }
    }
    if (nv < 1) nv = 1;
    for (int d = 0; d < D; d++) pooled[d] /= (float)nv;
    hipMemcpy(c->d_ds_pooled, pooled, (size_t)D * 4, hipMemcpyHostToDevice);

    /* proj_text = pooled @ tproj_w.T + tproj_b  -> (1, D). */
    if (r_gemm(c, c->d_ds_proj_text, c->ds_tproj_w, c->d_ds_pooled,
               c->ds_tproj_b, D, D, 1)) return -1;

    /* For each layer: qproj = qh_li @ qproj_w.T + b -> (Nq, D);
     * scores = qproj @ proj_text.T -> (Nq, 1); scale + clamp host.
     * Dot product is done host-side (proj_text is F32; gemm weights must be
     * F16) — Nq*D=51200 muls is negligible. */
    const float scale = 1.0f / sqrtf((float)D);
    float proj_text[S3_DETR_DIM];
    hipMemcpy(proj_text, c->d_ds_proj_text, (size_t)D * 4, hipMemcpyDeviceToHost);
    float *qproj_h = (float *)malloc((size_t)Nq * D * 4);
    if (!qproj_h) return -1;
    for (int li = 0; li < L; li++) {
        const void *qh_li = (const char *)c->d_dd_inter +
                             (size_t)li * Nq * D * 4;
        if (r_gemm(c, c->d_ds_qproj, c->ds_qproj_w, qh_li,
                   c->ds_qproj_b, D, D, Nq)) { free(qproj_h); return -1; }
        hipMemcpy(qproj_h, c->d_ds_qproj, (size_t)Nq * D * 4, hipMemcpyDeviceToHost);
        for (int q = 0; q < Nq; q++) {
            float acc = 0.0f;
            for (int d = 0; d < D; d++)
                acc += qproj_h[q * D + d] * proj_text[d];
            acc *= scale;
            if (acc >  12.0f) acc =  12.0f;
            if (acc < -12.0f) acc = -12.0f;
            c->ds_scores[li * Nq + q] = acc;
        }
    }
    free(qproj_h);
    c->ds_done = 1;
    return 0;
}

int cuda_sam3_1_get_dot_scores(const cuda_sam3_1_ctx *c, float *out) {
    if (!c || !c->ds_done || !out) return -1;
    memcpy(out, c->ds_scores, (size_t)S3_DETR_LAYERS * S3_DETR_Q * 4);
    return 0;
}

int cuda_sam3_1_run_mask_dec(cuda_sam3_1_ctx *c) {
    if (!c || !c->dd_done) return -1;
    const int D = S3_DETR_DIM, T = S3_T_CTX, N = S3_NTOK, Nq = S3_DETR_Q;
    const int H2 = 72, W2 = 72, H1 = 144, W1 = 144, H0 = 288, W0 = 288;

    /* 1) prompt_cross_attn: LN + residual cross-attn on detr_vis tokens. */
    if (r_ln_eps(c, c->d_md_enc_ln, c->d_detr_vis,
                  c->md_pca_ln_w, c->md_pca_ln_b, N, D, S3_DETR_LN_EPS)) return -1;
    if (r_gemm(c, c->d_md_q, c->md_pca_q_w, c->d_md_enc_ln, c->md_pca_q_b, D, D, N)) return -1;
    if (r_gemm(c, c->d_md_k, c->md_pca_k_w, c->d_detr_text_pooled, c->md_pca_k_b, D, D, T)) return -1;
    if (r_gemm(c, c->d_md_v, c->md_pca_v_w, c->d_detr_text_pooled, c->md_pca_v_b, D, D, T)) return -1;
    if (r_mha32(c, c->d_md_attn, c->d_md_q, c->d_md_k, c->d_md_v,
                 c->d_text_mask_i32, N, T)) return -1;
    if (r_gemm(c, c->d_md_o, c->md_pca_o_w, c->d_md_attn, c->md_pca_o_b, D, D, N)) return -1;
    /* enc_mod = detr_vis + o. */
    hipMemcpy(c->d_md_enc_mod, c->d_detr_vis, (size_t)N * D * 4, hipMemcpyDeviceToDevice);
    if (r_add(c, c->d_md_enc_mod, c->d_md_o, N * D)) return -1;

    /* 2) tokens -> (C, H2, W2). */
    if (r_tok2chw(c, c->d_md_start, c->d_md_enc_mod, D, H2, W2)) return -1;

    /* 3) Stage 0: upsample 72->144, add fpn[1], conv3x3 + GN + ReLU. */
    if (r_upnn(c, c->d_md_prev, c->d_md_start, D, H2, W2, H1, W1)) return -1;
    if (r_add(c, c->d_md_prev, c->fpn[1].d_out, D * H1 * W1)) return -1;
    if (r_c3x3(c, c->d_md_cout, c->md_conv_w[0], c->md_conv_b[0],
                c->d_md_prev, D, D, H1, W1)) return -1;
    if (r_groupnorm(c, c->d_md_cout, c->md_gn_w[0], c->md_gn_b[0], D, H1, W1, 8)) return -1;
    if (r_relu(c, c->d_md_cout, D * H1 * W1)) return -1;

    /* Stage 1: upsample 144->288, add fpn[0], conv3x3 + GN + ReLU -> pixel. */
    if (r_upnn(c, c->d_md_up, c->d_md_cout, D, H1, W1, H0, W0)) return -1;
    if (r_add(c, c->d_md_up, c->fpn[0].d_out, D * H0 * W0)) return -1;
    if (r_c3x3(c, c->d_md_pixel, c->md_conv_w[1], c->md_conv_b[1],
                c->d_md_up, D, D, H0, W0)) return -1;
    if (r_groupnorm(c, c->d_md_pixel, c->md_gn_w[1], c->md_gn_b[1], D, H0, W0, 8)) return -1;
    if (r_relu(c, c->d_md_pixel, D * H0 * W0)) return -1;

    /* 4) instance_projection (1x1), semantic_projection (1x1). */
    if (r_c1x1(c, c->d_md_instance, c->md_ip_w, c->md_ip_b,
                c->d_md_pixel, D, D, H0, W0)) return -1;
    if (r_c1x1(c, c->d_md_semantic, c->md_sp_w, c->md_sp_b,
                c->d_md_pixel, D, 1, H0, W0)) return -1;

    /* 5) mask_embedder on last-layer qh (d_dd_qh, (Nq, D)). */
    if (r_gemm(c, c->d_md_me_hid, c->md_me_w[0], c->d_dd_qh,
                c->md_me_b[0], D, D, Nq)) return -1;
    if (r_relu(c, c->d_md_me_hid, Nq * D)) return -1;
    if (r_gemm(c, c->d_md_mask_emb, c->md_me_w[1], c->d_md_me_hid,
                c->md_me_b[1], D, D, Nq)) return -1;
    if (r_relu(c, c->d_md_mask_emb, Nq * D)) return -1;
    if (r_gemm(c, c->d_md_me_hid, c->md_me_w[2], c->d_md_mask_emb,
                c->md_me_b[2], D, D, Nq)) return -1;
    /* d_md_me_hid now holds mask_emb (Nq, D). */

    /* 6) einsum qc,chw -> qhw. */
    if (r_einsum_qchw(c, c->d_md_pred_masks, c->d_md_me_hid,
                        c->d_md_instance, Nq, D, H0, W0)) return -1;

    c->md_done = 1;
    return 0;
}

int cuda_sam3_1_get_pred_masks(const cuda_sam3_1_ctx *c, float *out,
                             int *out_q, int *out_h, int *out_w) {
    if (!c || !c->md_done || !out) return -1;
    hipMemcpy(out, c->d_md_pred_masks, (size_t)S3_DETR_Q * 288 * 288 * 4,
              hipMemcpyDeviceToHost);
    if (out_q) *out_q = S3_DETR_Q;
    if (out_h) *out_h = 288;
    if (out_w) *out_w = 288;
    return 0;
}

int cuda_sam3_1_get_semantic_seg(const cuda_sam3_1_ctx *c, float *out,
                                int *out_h, int *out_w) {
    if (!c || !c->md_done || !out) return -1;
    hipMemcpy(out, c->d_md_semantic, (size_t)288 * 288 * 4, hipMemcpyDeviceToHost);
    if (out_h) *out_h = 288;
    if (out_w) *out_w = 288;
    return 0;
}

static void pp_bilinear_acf(float *dst, const float *src,
                             int C, int Hi, int Wi, int Ho, int Wo) {
    const float sy = (float)Hi / (float)Ho;
    const float sx = (float)Wi / (float)Wo;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < Ho; oh++) {
            float fy = (oh + 0.5f) * sy - 0.5f;
            int y0 = (int)floorf(fy); int y1 = y0 + 1;
            float dy = fy - (float)y0;
            if (y0 < 0) { y0 = 0; dy = 0.0f; }
            if (y1 < 0) y1 = 0;
            if (y0 > Hi - 1) y0 = Hi - 1;
            if (y1 > Hi - 1) y1 = Hi - 1;
            for (int ow = 0; ow < Wo; ow++) {
                float fx = (ow + 0.5f) * sx - 0.5f;
                int x0 = (int)floorf(fx); int x1 = x0 + 1;
                float dx = fx - (float)x0;
                if (x0 < 0) { x0 = 0; dx = 0.0f; }
                if (x1 < 0) x1 = 0;
                if (x0 > Wi - 1) x0 = Wi - 1;
                if (x1 > Wi - 1) x1 = Wi - 1;
                const float *sp = src + (size_t)c * Hi * Wi;
                float v00 = sp[y0*Wi + x0], v01 = sp[y0*Wi + x1];
                float v10 = sp[y1*Wi + x0], v11 = sp[y1*Wi + x1];
                float v = v00*(1-dy)*(1-dx) + v01*(1-dy)*dx
                        + v10*   dy *(1-dx) + v11*   dy *dx;
                dst[(size_t)c*Ho*Wo + oh*Wo + ow] = v;
            }
        }
    }
}

int cuda_sam3_1_run_postprocess(cuda_sam3_1_ctx *c, int target_h, int target_w,
                              float score_threshold, float mask_threshold) {
    if (!c || !c->md_done || !c->ds_done) return -1;
    if (target_h <= 0 || target_w <= 0) return -1;
    const int Nq = S3_DETR_Q, H0 = 288, W0 = 288;

    free(c->pp_scores); c->pp_scores = NULL;
    free(c->pp_boxes);  c->pp_boxes  = NULL;
    free(c->pp_masks);  c->pp_masks  = NULL;
    c->pp_n_kept = 0; c->pp_done = 0;

    const float *logits = c->ds_scores + (size_t)(S3_DETR_LAYERS - 1) * Nq;
    float pres = c->dd_presence_logits[S3_DETR_LAYERS - 1];
    float pres_s = 1.0f / (1.0f + expf(-pres));

    int *keep = (int *)malloc((size_t)Nq * sizeof(int));
    int n = 0;
    for (int q = 0; q < Nq; q++) {
        float s = (1.0f / (1.0f + expf(-logits[q]))) * pres_s;
        if (s > score_threshold) keep[n++] = q;
    }
    if (n == 0) {
        free(keep);
        c->pp_th = target_h; c->pp_tw = target_w; c->pp_done = 1;
        return 0;
    }

    c->pp_scores = (float *)malloc((size_t)n * 4);
    c->pp_boxes  = (float *)malloc((size_t)n * 4 * 4);
    c->pp_masks  = (uint8_t *)malloc((size_t)n * target_h * target_w);
    if (!c->pp_scores || !c->pp_boxes || !c->pp_masks) { free(keep); return -1; }

    for (int i = 0; i < n; i++) {
        int q = keep[i];
        c->pp_scores[i] = (1.0f / (1.0f + expf(-logits[q]))) * pres_s;
        float *b = c->pp_boxes + (size_t)i * 4;
        const float *pb = c->dd_pred_boxes + (size_t)q * 4;
        b[0] = pb[0] * (float)target_w;
        b[1] = pb[1] * (float)target_h;
        b[2] = pb[2] * (float)target_w;
        b[3] = pb[3] * (float)target_h;
    }

    const size_t HW = (size_t)H0 * W0;
    float *all_masks = (float *)malloc((size_t)Nq * HW * sizeof(float));
    if (!all_masks) { free(keep); return -1; }
    hipMemcpy(all_masks, c->d_md_pred_masks, (size_t)Nq * HW * 4, hipMemcpyDeviceToHost);

    float *kept = (float *)malloc((size_t)n * HW * sizeof(float));
    if (!kept) { free(all_masks); free(keep); return -1; }
    for (int i = 0; i < n; i++)
        memcpy(kept + (size_t)i * HW, all_masks + (size_t)keep[i] * HW, HW * 4);
    free(all_masks); free(keep);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float *row = kept + (size_t)i * HW;
        for (size_t j = 0; j < HW; j++)
            row[j] = 1.0f / (1.0f + expf(-row[j]));
    }

    float *resized = (float *)malloc((size_t)n * target_h * target_w * sizeof(float));
    if (!resized) { free(kept); return -1; }
    pp_bilinear_acf(resized, kept, n, H0, W0, target_h, target_w);
    free(kept);

    size_t tot = (size_t)n * target_h * target_w;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < tot; i++)
        c->pp_masks[i] = (resized[i] > mask_threshold) ? 1 : 0;
    free(resized);

    c->pp_n_kept = n; c->pp_th = target_h; c->pp_tw = target_w; c->pp_done = 1;
    return 0;
}

const float *cuda_sam3_1_get_final_scores(const cuda_sam3_1_ctx *c, int *out_n) {
    if (!c || !c->pp_done) return NULL;
    if (out_n) *out_n = c->pp_n_kept;
    return c->pp_scores;
}
const float *cuda_sam3_1_get_final_boxes(const cuda_sam3_1_ctx *c, int *out_n) {
    if (!c || !c->pp_done) return NULL;
    if (out_n) *out_n = c->pp_n_kept;
    return c->pp_boxes;
}
const uint8_t *cuda_sam3_1_get_final_masks(const cuda_sam3_1_ctx *c,
                                          int *out_n, int *out_h, int *out_w) {
    if (!c || !c->pp_done) return NULL;
    if (out_n) *out_n = c->pp_n_kept;
    if (out_h) *out_h = c->pp_th;
    if (out_w) *out_w = c->pp_tw;
    return c->pp_masks;
}

int cuda_sam3_1_get_vit_embed(const cuda_sam3_1_ctx *c, float *out, int *n, int *d) {
    if (!c || !c->ready || !out) return -1;
    hipDeviceSynchronize();
    if (hipMemcpy(out, c->d_tok, (size_t)S3_NTOK * S3_DIM * 4, hipMemcpyDeviceToHost) != hipSuccess)
        return -1;
    if (n) *n = S3_NTOK;
    if (d) *d = S3_DIM;
    return 0;
}
