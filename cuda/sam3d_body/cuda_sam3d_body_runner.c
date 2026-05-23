/* CUDA runner for SAM 3D Body.
 *
 * Supports DINOv3-H+ and ViT-H backbones, CUDA encoder/ray/token/decoder
 * stages, and CPU MHR-in-the-loop execution for the production path.
 *
 * Upstream layout (from ckpt slice `sam3d_body_dinov3.safetensors`, or
 * `sam3d_body_dinov3_fp32.safetensors` when --precision fp32 is used,
 * prefix `backbone.encoder.`):
 *   patch_embed.proj.weight   F32 (1280, 3, 16, 16)   — kept F32; tiny
 *   patch_embed.proj.bias     F32 (1280,)
 *   cls_token                 F32 (1, 1, 1280)
 *   storage_tokens            F32 (1, 4, 1280)        — 4 register tokens
 *   rope_embed.periods        F32 (16,)               — saved RoPE periods
 *   norm.weight/bias          F32 (1280,)             — final LayerNorm
 *   blocks.{i=0..31}:
 *     norm1.weight/bias       F32 (1280,)
 *     norm2.weight/bias       F32 (1280,)
 *     attn.qkv.weight         F16/F32 (3840, 1280)    — big matmul
 *     attn.qkv.bias           F32 (3840,)             — all-zero in ckpt
 *     attn.proj.weight        F16/F32 (1280, 1280)
 *     attn.proj.bias          F32 (1280,)
 *     ls1.gamma, ls2.gamma    F32 (1280,)
 *     mlp.w1/w2.weight        F16/F32 (5120, 1280)    — SwiGLU gates
 *     mlp.w1/w2.bias          F32 (5120,)
 *     mlp.w3.weight           F16/F32 (1280, 5120)    — SwiGLU down-proj
 *     mlp.w3.bias             F32 (1280,)
 *
 * Dims: D=1280, H=20, Dh=64, N=32 blocks, ffn=5120, patch=16, default
 *       image=512x512 → grid 32×32 = 1024 patch tokens + 1 CLS + 4 register.
 *
 * SPDX-License-Identifier: MIT
 */

#include "cuda_sam3d_body_runner.h"
#include "../cuew.h"
#include "../cublasew.h"
#include "../cuda_kernels_common.h"
#include "cuda_sam3d_body_kernels.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"

#ifndef CUDA_SAM3D_BODY_RUNNER_EXTERNAL_IMPLS
#define SAFETENSORS_IMPLEMENTATION
#endif
#include "../../common/safetensors.h"

#if defined(_OPENMP)
#  include <omp.h>
#  define SAM3DB_MHR_THREADS() omp_get_max_threads()
#else
#  define SAM3DB_MHR_THREADS() 1
#endif

/* CPU decoder + MHR helpers — declarations only. The implementations
 * live in sam3d_body_cpu.c, which is linked alongside this TU. They
 * provide: sam3d_body_decoder_load/free, compute_bbox_affine,
 * default_cam_int, compute_ray_cond_xyz, compute_condition_info,
 * invalid_prompt_token, decode_pose_raw, keypoints_from_mesh,
 * camera_project, mhr_load/free/forward, plus the
 * s3db_extract_jcoords_meters static helper available via the header. */
#include "../../common/sam3d_body_decoder.h"
#include "../../common/sam3d_body_mhr.h"
#define CPU_COMPUTE_IMPLEMENTATION
#include "../../common/cpu_compute.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUDA_SAM3D_BODY_E_OK              (0)
#define CUDA_SAM3D_BODY_E_INVAL           (-1)
#define CUDA_SAM3D_BODY_E_NOT_IMPLEMENTED (-2)
#define CUDA_SAM3D_BODY_E_LOAD            (-3)

static double sb_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

/* DINOv3-H+ constants (match common/dinov3.h for sam-3d-body). */
#define SB_DIM       1280
#define SB_HEADS     20
#define SB_HEAD_DIM  64
#define SB_N_BLK     32
#define SB_FFN       5120
#define SB_PATCH     16
#define SB_IMG       512
#define SB_GRID      (SB_IMG / SB_PATCH)     /* 32 */
#define SB_N_PATCH   (SB_GRID * SB_GRID)      /* 1024 */
#define SB_N_STORAGE 4
#define SB_N_TOK     (1 + SB_N_STORAGE + SB_N_PATCH)  /* 1029 */
#define SB_LN_EPS    1e-5f

/* ViT-H/16 (vit_hmr_512_384) constants — see common/sam3d_body_vit.h.
 * dim=1280, heads=16, head_dim=80, depth=32, ffn=5120, patch=16 (pad=2),
 * input 512×384 → grid 32×24 = 768 patches. No CLS/storage tokens. */
#define SB_VITH_DIM        1280
#define SB_VITH_HEADS      16
#define SB_VITH_HEAD_DIM   80
#define SB_VITH_N_BLK      32
#define SB_VITH_FFN        5120
#define SB_VITH_PATCH      16
#define SB_VITH_PAD        2
#define SB_VITH_IMG_H      512
#define SB_VITH_IMG_W      384
#define SB_VITH_GRID_H     (SB_VITH_IMG_H / SB_VITH_PATCH)   /* 32 */
#define SB_VITH_GRID_W     (SB_VITH_IMG_W / SB_VITH_PATCH)   /* 24 */
#define SB_VITH_N_PATCH    (SB_VITH_GRID_H * SB_VITH_GRID_W) /* 768 */
#define SB_VITH_LN_EPS     1e-6f

typedef struct {
    void *norm1_w, *norm1_b;    /* F32 (D,) */
    void *norm2_w, *norm2_b;
    void *qkv_w;                /* F16 (3D, D) */
    void *qkv_b;                /* F32 (3D,)  — all-zero in ckpt but uploaded anyway */
    void *proj_w;               /* F16 (D, D) */
    void *proj_b;               /* F32 (D,) */
    void *ls1;                  /* F32 (D,) */
    void *ls2;                  /* F32 (D,) */
    void *w1_w, *w1_b;          /* F16 (FFN, D) / F32 (FFN,) */
    void *w2_w, *w2_b;          /* F16 (FFN, D) / F32 (FFN,) */
    void *w3_w, *w3_b;          /* F16 (D, FFN) / F32 (D,) */
} sb_block;

/* ViT-H/16 block weights — lighter than DINOv3 (no LayerScale, no
 * SwiGLU). 2-weight GELU MLP: fc1: D→FFN, fc2: FFN→D. */
typedef struct {
    void *norm1_w, *norm1_b;    /* F32 (D,) */
    void *norm2_w, *norm2_b;
    void *qkv_w;                /* F16 (3D, D) */
    void *qkv_b;                /* F32 (3D,) */
    void *proj_w;               /* F16 (D, D) */
    void *proj_b;               /* F32 (D,) */
    void *fc1_w, *fc1_b;        /* F16 (FFN, D) / F32 (FFN,) */
    void *fc2_w, *fc2_b;        /* F16 (D, FFN) / F32 (D,) */
} sb_block_vith;

/* Decoder constants. */
#define SB_DEC_DIM       1024   /* token channel */
#define SB_DEC_KV_DIM    SB_DIM /* image channel = encoder dim 1280 */
#define SB_DEC_HEADS     8
#define SB_DEC_HEAD_DIM  64
#define SB_DEC_PROJ_DIM  (SB_DEC_HEADS * SB_DEC_HEAD_DIM) /* 512 */
#define SB_DEC_FFN       1024
#define SB_DEC_LAYERS    6
#define SB_DEC_KP        70
#define SB_DEC_HAND_TOK  2
#define SB_DEC_NPOSE     519
#define SB_DEC_NCAM      3
#define SB_DEC_COND      3
#define SB_DEC_LN_EPS    1e-6f

/* One decoder-layer's device weights (mirrors common/sam3d_body_decoder.h). */
typedef struct {
    void *ln1_w,    *ln1_b;
    void *ln_pe_1_w,*ln_pe_1_b;
    void *ln2_1_w,  *ln2_1_b;
    void *ln2_2_w,  *ln2_2_b;
    void *ln_pe_2_w,*ln_pe_2_b;
    void *ln3_w,    *ln3_b;
    void *sa_q_w,   *sa_q_b;     /* (512, 1024) f32 */
    void *sa_k_w,   *sa_k_b;
    void *sa_v_w,   *sa_v_b;
    void *sa_proj_w,*sa_proj_b;  /* (1024, 512) */
    void *ca_q_w,   *ca_q_b;     /* (512, 1024) */
    void *ca_k_w,   *ca_k_b;     /* (512, 1280) */
    void *ca_v_w,   *ca_v_b;     /* (512, 1280) */
    void *ca_proj_w,*ca_proj_b;  /* (1024, 512) */
    void *ffn0_w,   *ffn0_b;     /* (1024, 1024) */
    void *ffn1_w,   *ffn1_b;     /* (1024, 1024) */
} sb_dec_layer_w;

/* All decoder + MHR-head device weights. F32 throughout (CPU port keeps
 * decoder weights in F32 since they're small enough for free precision). */
typedef struct {
    int loaded;

    /* Top-level weights (decoder safetensors). */
    void *ray_cond_conv_w;       /* (1280, 1379, 1, 1) — bias=False */
    void *ray_cond_norm_w, *ray_cond_norm_b; /* (1280,) */
    void *init_to_token_w, *init_to_token_b;     /* (1024, 525) */
    void *prev_to_token_w, *prev_to_token_b;     /* (1024, 522) */
    void *prompt_to_token_w, *prompt_to_token_b; /* (1024, 1280) */
    void *keypoint_embedding;    /* (70, 1024) */
    void *keypoint3d_embedding;  /* (70, 1024) */
    void *hand_box_embedding;    /* (2, 1024) */
    void *kp_feat_linear_w, *kp_feat_linear_b;       /* (1024, 1280) */
    void *kp_posemb_l0_w, *kp_posemb_l0_b;           /* (1024, 2) */
    void *kp_posemb_l1_w, *kp_posemb_l1_b;           /* (1024, 1024) */
    void *kp3d_posemb_l0_w, *kp3d_posemb_l0_b;       /* (1024, 3) */
    void *kp3d_posemb_l1_w, *kp3d_posemb_l1_b;       /* (1024, 1024) */
    void *prompt_pe_gauss;       /* (2, 640) — for get_dense_pe */
    void *invalid_point_embed;   /* (1, 1280) */
    void *no_mask_embed;         /* (1, 1280) */
    const float *no_mask_embed_h; /* (1, 1280) host view, st_context-owned */

    sb_dec_layer_w layers[SB_DEC_LAYERS];

    void *norm_final_w, *norm_final_b;  /* (1024,) */

    /* MHR-head regression (mhr_head safetensors). */
    void *head_pose_l0_w,   *head_pose_l0_b;    /* (1024, 1024) */
    void *head_pose_l1_w,   *head_pose_l1_b;    /* (519, 1024) */
    void *head_camera_l0_w, *head_camera_l0_b;  /* (1024, 1024) */
    void *head_camera_l1_w, *head_camera_l1_b;  /* (3, 1024) */
    void *init_pose;             /* (1, 519) */
    void *init_camera;           /* (1, 3) */

    /* MHR-head decode tables (head_pose.* in mhr_head safetensors). */
    void *scale_mean;            /* (68,) f32 */
    void *scale_comps;           /* (28, 68) f32 */
    void *hand_pose_mean;        /* (54,) f32 */
    void *hand_pose_comps;       /* (54, 54) f32 */
    void *hand_idx_left;         /* (27,) i64 */
    void *hand_idx_right;        /* (27,) i64 */
    void *keypoint_mapping;      /* (308, 18566) f32 */
    void *local_to_world_wrist;  /* (3, 3) f32 */
    void *right_wrist_coords;    /* (3,) f32 */
    void *root_coords;           /* (3,) f32 */
    void *nonhand_param_idxs;    /* (145,) i64 */

    /* faces (36874, 3) i64 — kept on host for OBJ writer (loaded via
     * the host i64 buffer, copied into ctx->faces as i32). */
    int64_t *host_faces_i64;     int n_faces;

    /* Host (mmap-backed) pointers to the norm_final + head_pose + head_camera
     * weights so we can run that tail block on CPU. The GEMVs are tiny
     * (1024×1024 + 1024×519 + 1024×3 per call); the host copies cost nothing
     * since safetensors_data() returns into the mmap. Reduces D↔H bounces in
     * the between-layer / post-loop pose+cam regression path. */
    const float *norm_final_w_h, *norm_final_b_h;        /* (1024,) */
    const float *head_pose_l0_w_h, *head_pose_l0_b_h;    /* (1024, 1024) / (1024,) */
    const float *head_pose_l1_w_h, *head_pose_l1_b_h;    /* (519, 1024)  / (519,)  */
    const float *head_camera_l0_w_h, *head_camera_l0_b_h;/* (1024, 1024) / (1024,) */
    const float *head_camera_l1_w_h, *head_camera_l1_b_h;/* (3, 1024)    / (3,)    */
} sb_dec_w;

typedef struct {
    float *data;
    int    n, c;
} f32_2d;

typedef struct {
    void *xpe_n, *cpe_n, *x_ln, *qk;
    void *Q, *K, *V, *attn, *proj;
    void *x1, *q_in, *k_in, *ctx_ln;
    void *x2, *x_ln3, *ffn_h, *ffn_o;
} sb_dec_layer_scratch_dev;

typedef struct {
    void *gxy, *invalid;
    void *p_h, *p_o;
    void *kp_feats, *kp_proj;
    void *kp3d_norm, *p3_h, *p3_o;
} sb_kpu_scratch_dev;

typedef struct {
    void *norm_row;
    void *pose_h, *pose_raw;
    void *cam_h, *cam_raw;
} sb_head_scratch_dev;

struct cuda_sam3d_body_ctx {
    cuda_sam3d_body_config cfg;
    int device_id;
    int verbose;
    int sm;
    hipModule_t mod;
    hipFunction_t fn_sentinel;
    hipFunction_t fn_resize, fn_patch, fn_patch_im2col, fn_prepend;
    hipFunction_t fn_bf16_round;
    hipFunction_t fn_ln, fn_ln_sqrtdiv, fn_ln_welford, fn_ln_welford_warp;
    hipFunction_t fn_ln_torchvec;
    hipFunction_t fn_gemm_tiled, fn_gemm_tiled_f32;
    hipFunction_t fn_rope_qk, fn_kv_tx, fn_qkv_tx, fn_heads_merge;
    hipFunction_t fn_attn_prob_v_serial;
    hipFunction_t fn_fa, fn_sdpa_qkv;
    hipFunction_t fn_scale_rows, fn_scale_softmax, fn_softmax_warp;
    hipFunction_t fn_silu_mul, fn_layerscale_add;
    hipFunction_t fn_ray_cond_fourier, fn_encoder_to_preconv, fn_chw_to_tok;
    hipFunction_t fn_conv1x1_chw, fn_ln_chw;
    hipFunction_t fn_linear_bias, fn_add_bias_rows;
    hipFunction_t fn_gemm_f32, fn_add_two, fn_add_inplace, fn_gelu_inplace, fn_sdpa;
    hipFunction_t fn_relu_inplace, fn_grid_sample, fn_kp_pelvis_norm,
                  fn_augment_overwrite_mask, fn_dense_pe_tok;
    /* ViT-H specific kernels. */
    hipFunction_t fn_patch_pad2, fn_pos_embed_vith, fn_qkv_split;
    /* MHR-on-GPU helper kernels. */
    hipFunction_t fn_mhr_blend;
    hipFunction_t fn_mhr_lbs;
    hipFunction_t fn_mhr_matvec;
    hipFunction_t fn_mhr_keypoints;
    hipFunction_t fn_mhr_project;
    st_context *st_enc;
    cublasew_context *cublas;
    int use_cublas;
    int weights_ready;

    /* encoder weights (device ptrs). */
    void    *w_patch_w, *w_patch_b;    /* F32 (D, 3, P, P) / F32 (D,) */
    void    *w_cls;                     /* F32 (1, 1, D) */
    void    *w_storage;                 /* F32 (1, S, D) */
    void    *w_rope_periods;            /* F32 (16,) */
    void    *w_rope_cos, *w_rope_sin;   /* F32 (N_PATCH, HEAD_DIM) — built host-side */
    void    *w_norm_w, *w_norm_b;       /* F32 (D,) final LN */
    sb_block blk[SB_N_BLK];

    /* ViT-H/16 backbone (separate set; only loaded when
     * cfg.backbone == VITH). */
    void    *w_vith_pos_embed;          /* F32 (1, n_patches+1, D) */
    sb_block_vith vith_blk[SB_VITH_N_BLK];

    /* Decoder + MHR-head safetensors and uploaded weights. */
    st_context *st_dec;
    st_context *st_mhr;
    sb_dec_w    dec;

    /* CPU model + MHR assets — used in the per-layer decoder loop for
     * decode_pose_raw, hybrid/CPU mhr_forward, keypoints_from_mesh, and
     * camera_project. cpu_mhr is NULL until cfg.mhr_assets_dir is set. */
    sam3d_body_decoder_model *cpu_dec;
    sam3d_body_mhr_assets    *cpu_mhr;

    /* Optional production hybrid GPU MHR path, requested with
     * SAM3D_BODY_GPU_MHR=1. Static shape/face/pose/LBS/keypoint tensors are
     * uploaded lazily on first decoder use; small parameter/skeleton stages
     * remain on host. */
    int      mhr_gpu_requested;
    int      mhr_gpu_enabled;
    void    *d_mhr_pc_linear_weight;    /* (55317, 3000) f32 */
    void    *d_mhr_pc_h;                /* (3000,) f32 scratch */
    void    *d_mhr_pc_out;              /* (55317,) f32 scratch */
    void    *d_mhr_shape_coeffs;        /* (45,) f32 scratch */
    void    *d_mhr_face_coeffs;         /* (72,) f32 scratch */
    void    *d_mhr_jstate;              /* (127, 8) f32 scratch */
    void    *d_mhr_blend_vectors;       /* (45, 55317) f32 */
    void    *d_mhr_blend_base;          /* (55317,) f32 */
    void    *d_mhr_face_vectors;        /* (72, 55317) f32 */
    void    *d_mhr_skin_indices;        /* (51337,) i32 */
    void    *d_mhr_skin_weights;        /* (51337,) f32 */
    void    *d_mhr_vert_indices;        /* (51337,) i64 */
    void    *d_mhr_rest;                /* (55317,) f32 scratch */
    void    *d_mhr_face;                /* (55317,) f32 scratch */
    void    *d_mhr_skinned;             /* (55317,) f32 scratch */
    void    *d_mhr_gskel;               /* (127, 8) f32 scratch */
    void    *d_mhr_keypoint_mapping;    /* (308, 18566) f32 */
    void    *d_mhr_kp3d;                /* (70, 3) f32 scratch */
    void    *d_mhr_kp2d_full;           /* (70, 2) f32 scratch */
    void    *d_mhr_kp2d_crop;           /* (70, 2) f32 scratch */
    void    *d_mhr_kp2d_depth;          /* (70,) f32 scratch */
    void    *d_mhr_pred_cam_t;          /* (3,) f32 scratch */
    void    *d_kpu_img;                 /* cached img_emb for kp_token_update */
    size_t   d_kpu_img_cap;
    int      kpu_use_cached_img;
    float   *dense_pe_tok_h;            /* cached dense PE in token form */
    int      dense_pe_gh, dense_pe_gw, dense_pe_dc;
    void    *d_dense_pe_tok;            /* cached dense PE in token form */
    size_t   d_dense_pe_tok_bytes;

    /* device runtime buffers. */
    void    *d_img_u8;  size_t img_u8_cap;   /* (Hin, Win, 3) u8, host-resized */
    void    *d_img_f32;                       /* (3, IMG_H, IMG_W) f32 normalized */
    void    *d_patch_cols;                    /* (N_PATCH, 3*PATCH*PATCH) f32 */
    void    *d_tok;                           /* (N_TOK, D) f32 residual stream */
    void    *d_ln;                            /* (N_TOK, D) f32 LN scratch */
    void    *d_qkv;                           /* (N_TOK, 3D) f32 fused QKV */
    void    *d_qt, *d_kt, *d_vt;              /* (HEADS, N_TOK, HEAD_DIM) */
    void    *d_attn_scores;                   /* (HEADS, N_TOK, N_TOK) */
    void    *d_attn;                          /* (N_TOK, D) attention output */
    void    *d_proj;                          /* (N_TOK, D) post-proj scratch */
    void    *d_gate, *d_up;                   /* (N_TOK, FFN) SwiGLU */

    /* host-side runtime state. */
    uint8_t *image_rgb;  int img_w, img_h;
    float    bbox[4];    int has_bbox;
    float    focal_hint;
    int      has_norm_input;  /* 1 if d_img_f32 already populated via debug API */

    /* TopdownAffine outputs cached by run_encoder so run_decoder can reuse
     * them (CPU runner mirrors this — encoder + decoder must agree on the
     * same warp matrix or the decoder cam_int / rays_xyz disagree with the
     * actual image fed to DINOv3). */
    int      self_pp_set;
    float    self_center[2];
    float    self_scale[2];
    float    self_warp[6];
    float    self_cam_int[9];

    f32_2d   encoder_tokens;
    int      encoder_tokens_dev_valid;
    float   *mhr_params;   int mhr_params_n;
    float    cam_t[3];
    float    focal_px;
    float   *vertices;     int n_vertices;
    int32_t *faces;        int n_faces;
    float   *keypoints_3d; int n_kp_3d;
    float   *keypoints_2d; int n_kp_2d;
};

static int cuda_sam3d_body_mhr_forward_hybrid(
        cuda_sam3d_body_ctx *ctx,
        const float *model_params, const float *shape, const float *face,
        int apply_correctives, int n_threads, float *scratch,
        float *out_skinned_verts, float *out_global_skel);
static int cuda_sam3d_body_keypoints_project_cached(
        cuda_sam3d_body_ctx *ctx,
        const float *global_skel_cm,
        const float *pred_cam,
        const sam3d_body_camera_batch *batch,
        float *kp3d_host,
        float *kp2d_full_host,
        float *kp2d_crop_host,
        float *kp2d_depth_host,
        float *pred_cam_t_host);
static int cuda_sam3d_body_ensure_mhr_gpu_cache(cuda_sam3d_body_ctx *ctx);
static int cuda_sam3d_body_run_decoder_layer_dev(
        cuda_sam3d_body_ctx *ctx, int layer_idx,
        void *d_x, const void *d_ctx, void *d_xpe, const void *d_cpe,
        int N_q, int N_c, void *d_out,
        sb_dec_layer_scratch_dev *S);
static int cuda_sam3d_body_kp_token_update_dev(
        cuda_sam3d_body_ctx *ctx, int layer_idx, int H, int W, int N_q,
        const float *kp2d_cropped_host, const float *kp2d_depth_host,
        const void *d_img_chw, void *d_tokens, void *d_aug,
        sb_kpu_scratch_dev *S);
static int cuda_sam3d_body_norm_heads_dev(
        cuda_sam3d_body_ctx *ctx, const void *d_tokens,
        float *pose_raw_host, float *cam_raw_host,
        sb_head_scratch_dev *S);
static int cuda_sam3d_body_ray_cond_from_encoder_dev(
        cuda_sam3d_body_ctx *ctx, const void *d_encoder_tokens,
        int n_prefix, const float *rays_hwc, int H, int W,
        void *d_out_chw);
static int cuda_sam3d_body_chw_to_tok_dev(
        cuda_sam3d_body_ctx *ctx, void *d_out_tok,
        const void *d_in_chw, int N, int D);

static void sb_debug_dump_device_f32(cuda_sam3d_body_ctx *ctx,
                                     const char *dir,
                                     const char *stem,
                                     int idx,
                                     const void *dev,
                                     size_t n_float)
{
    if (!ctx || !dir || !dir[0] || !stem || !dev || n_float == 0)
        return;
    float *host = (float *)malloc(n_float * sizeof(float));
    if (!host)
        return;
    if (hipMemcpy(host, dev, n_float * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        free(host);
        return;
    }
    char path[1024];
    if (idx >= 0)
        snprintf(path, sizeof(path), "%s/%s%02d.bin", dir, stem, idx);
    else
        snprintf(path, sizeof(path), "%s/%s.bin", dir, stem);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(host, sizeof(float), n_float, f);
        fclose(f);
        if (ctx->verbose >= 2)
            fprintf(stderr, "sam3d_body: dumped %s (%zu f32)\n",
                    path, n_float);
    }
    free(host);
}

/* ===================== safetensors helpers ===================== */

static int sb_find(const st_context *st, const char *s) {
    char k[256]; snprintf(k, sizeof(k), "backbone.encoder.%s", s);
    return safetensors_find(st, k);
}

/* ViT-H prefix is `backbone.` (no `encoder.` segment). */
static int sb_find_vith(const st_context *st, const char *s) {
    char k[256]; snprintf(k, sizeof(k), "backbone.%s", s);
    return safetensors_find(st, k);
}

static void *sb_upload_f32(const st_context *st, const char *s, size_t expect_n) {
    int i = sb_find(st, s);
    if (i < 0) { fprintf(stderr, "sam3d_body: missing %s\n", s); return NULL; }
    if (strcmp(safetensors_dtype(st, i), "F32") != 0) {
        fprintf(stderr, "sam3d_body: %s not F32 (%s)\n", s, safetensors_dtype(st, i));
        return NULL;
    }
    size_t nb = safetensors_nbytes(st, i);
    if (expect_n && nb / 4 != expect_n) {
        fprintf(stderr, "sam3d_body: %s size mismatch: got %zu f32, expected %zu\n",
                s, nb / 4, expect_n);
        return NULL;
    }
    return hip_upload_raw(safetensors_data((st_context *)st, i), nb);
}

static void *sb_upload_f16(const st_context *st, const char *s, size_t expect_n) {
    int i = sb_find(st, s);
    if (i < 0) { fprintf(stderr, "sam3d_body: missing %s\n", s); return NULL; }
    const char *dt = safetensors_dtype(st, i);
    size_t nb = safetensors_nbytes(st, i);
    size_t n;
    if (!strcmp(dt, "F16")) {
        n = nb / 2;
        if (expect_n && n != expect_n) {
            fprintf(stderr, "sam3d_body: %s f16 size mismatch: %zu vs %zu\n",
                    s, n, expect_n);
            return NULL;
        }
        return hip_upload_raw(safetensors_data((st_context *)st, i), nb);
    }
    if (!strcmp(dt, "F32")) {
        n = nb / 4;
        if (expect_n && n != expect_n) {
            fprintf(stderr, "sam3d_body: %s f32→f16 size mismatch: %zu vs %zu\n",
                    s, n, expect_n);
            return NULL;
        }
        const float *src = (const float *)safetensors_data((st_context *)st, i);
        uint16_t *t = (uint16_t *)malloc(n * 2);
        for (size_t k = 0; k < n; k++) t[k] = hip_f32_to_f16(src[k]);
        void *d = hip_upload_raw(t, n * 2);
        free(t);
        return d;
    }
    fprintf(stderr, "sam3d_body: %s unsupported dtype %s\n", s, dt);
    return NULL;
}

/* `backbone.<name>` variants for the ViT-H backbone. */
static void *sb_upload_vith_f32(const st_context *st, const char *s,
                                size_t expect_n) {
    int i = sb_find_vith(st, s);
    if (i < 0) { fprintf(stderr, "sam3d_body_vith: missing %s\n", s); return NULL; }
    if (strcmp(safetensors_dtype(st, i), "F32") != 0) {
        fprintf(stderr, "sam3d_body_vith: %s not F32 (%s)\n",
                s, safetensors_dtype(st, i));
        return NULL;
    }
    size_t nb = safetensors_nbytes(st, i);
    if (expect_n && nb / 4 != expect_n) {
        fprintf(stderr, "sam3d_body_vith: %s size mismatch: got %zu f32, expected %zu\n",
                s, nb / 4, expect_n);
        return NULL;
    }
    return hip_upload_raw(safetensors_data((st_context *)st, i), nb);
}

static void *sb_upload_vith_f16(const st_context *st, const char *s,
                                size_t expect_n) {
    int i = sb_find_vith(st, s);
    if (i < 0) { fprintf(stderr, "sam3d_body_vith: missing %s\n", s); return NULL; }
    const char *dt = safetensors_dtype(st, i);
    size_t nb = safetensors_nbytes(st, i);
    size_t n;
    if (!strcmp(dt, "F16")) {
        n = nb / 2;
        if (expect_n && n != expect_n) {
            fprintf(stderr, "sam3d_body_vith: %s f16 size mismatch: %zu vs %zu\n",
                    s, n, expect_n);
            return NULL;
        }
        return hip_upload_raw(safetensors_data((st_context *)st, i), nb);
    }
    if (!strcmp(dt, "F32")) {
        n = nb / 4;
        if (expect_n && n != expect_n) {
            fprintf(stderr, "sam3d_body_vith: %s f32→f16 size mismatch: %zu vs %zu\n",
                    s, n, expect_n);
            return NULL;
        }
        const float *src = (const float *)safetensors_data((st_context *)st, i);
        uint16_t *t = (uint16_t *)malloc(n * 2);
        for (size_t k = 0; k < n; k++) t[k] = hip_f32_to_f16(src[k]);
        void *d = hip_upload_raw(t, n * 2);
        free(t);
        return d;
    }
    fprintf(stderr, "sam3d_body_vith: %s unsupported dtype %s\n", s, dt);
    return NULL;
}

static int sb_load_block_vith(cuda_sam3d_body_ctx *c, int bi) {
    sb_block_vith *b = &c->vith_blk[bi];
    char k[160];
    #define K(suf) (snprintf(k, sizeof(k), "blocks.%d." suf, bi), k)
    b->norm1_w = sb_upload_vith_f32(c->st_enc, K("norm1.weight"), SB_VITH_DIM);
    b->norm1_b = sb_upload_vith_f32(c->st_enc, K("norm1.bias"),   SB_VITH_DIM);
    b->norm2_w = sb_upload_vith_f32(c->st_enc, K("norm2.weight"), SB_VITH_DIM);
    b->norm2_b = sb_upload_vith_f32(c->st_enc, K("norm2.bias"),   SB_VITH_DIM);
    b->qkv_w   = sb_upload_vith_f16(c->st_enc, K("attn.qkv.weight"),
                                    (size_t)3 * SB_VITH_DIM * SB_VITH_DIM);
    b->qkv_b   = sb_upload_vith_f32(c->st_enc, K("attn.qkv.bias"),
                                    (size_t)3 * SB_VITH_DIM);
    b->proj_w  = sb_upload_vith_f16(c->st_enc, K("attn.proj.weight"),
                                    (size_t)SB_VITH_DIM * SB_VITH_DIM);
    b->proj_b  = sb_upload_vith_f32(c->st_enc, K("attn.proj.bias"), SB_VITH_DIM);
    b->fc1_w   = sb_upload_vith_f16(c->st_enc, K("mlp.fc1.weight"),
                                    (size_t)SB_VITH_FFN * SB_VITH_DIM);
    b->fc1_b   = sb_upload_vith_f32(c->st_enc, K("mlp.fc1.bias"), SB_VITH_FFN);
    b->fc2_w   = sb_upload_vith_f16(c->st_enc, K("mlp.fc2.weight"),
                                    (size_t)SB_VITH_DIM * SB_VITH_FFN);
    b->fc2_b   = sb_upload_vith_f32(c->st_enc, K("mlp.fc2.bias"), SB_VITH_DIM);
    #undef K
    if (!b->norm1_w || !b->norm1_b || !b->norm2_w || !b->norm2_b ||
        !b->qkv_w || !b->qkv_b || !b->proj_w || !b->proj_b ||
        !b->fc1_w || !b->fc1_b || !b->fc2_w || !b->fc2_b)
        return -1;
    return 0;
}

static int sb_load_block(cuda_sam3d_body_ctx *c, int bi) {
    sb_block *b = &c->blk[bi];
    char k[160];
    #define K(suf) (snprintf(k, sizeof(k), "blocks.%d." suf, bi), k)
    b->norm1_w = sb_upload_f32(c->st_enc, K("norm1.weight"), SB_DIM);
    b->norm1_b = sb_upload_f32(c->st_enc, K("norm1.bias"),   SB_DIM);
    b->norm2_w = sb_upload_f32(c->st_enc, K("norm2.weight"), SB_DIM);
    b->norm2_b = sb_upload_f32(c->st_enc, K("norm2.bias"),   SB_DIM);
    const int fp32 = c->cfg.precision && !strcmp(c->cfg.precision, "fp32");
    b->qkv_w   = fp32 ? sb_upload_f32(c->st_enc, K("attn.qkv.weight"), (size_t)3 * SB_DIM * SB_DIM)
                       : sb_upload_f16(c->st_enc, K("attn.qkv.weight"), (size_t)3 * SB_DIM * SB_DIM);
    b->qkv_b   = sb_upload_f32(c->st_enc, K("attn.qkv.bias"),   (size_t)3 * SB_DIM);
    b->proj_w  = fp32 ? sb_upload_f32(c->st_enc, K("attn.proj.weight"), (size_t)SB_DIM * SB_DIM)
                       : sb_upload_f16(c->st_enc, K("attn.proj.weight"), (size_t)SB_DIM * SB_DIM);
    b->proj_b  = sb_upload_f32(c->st_enc, K("attn.proj.bias"),   SB_DIM);
    b->ls1     = sb_upload_f32(c->st_enc, K("ls1.gamma"), SB_DIM);
    b->ls2     = sb_upload_f32(c->st_enc, K("ls2.gamma"), SB_DIM);
    b->w1_w    = fp32 ? sb_upload_f32(c->st_enc, K("mlp.w1.weight"), (size_t)SB_FFN * SB_DIM)
                       : sb_upload_f16(c->st_enc, K("mlp.w1.weight"), (size_t)SB_FFN * SB_DIM);
    b->w1_b    = sb_upload_f32(c->st_enc, K("mlp.w1.bias"),   SB_FFN);
    b->w2_w    = fp32 ? sb_upload_f32(c->st_enc, K("mlp.w2.weight"), (size_t)SB_FFN * SB_DIM)
                       : sb_upload_f16(c->st_enc, K("mlp.w2.weight"), (size_t)SB_FFN * SB_DIM);
    b->w2_b    = sb_upload_f32(c->st_enc, K("mlp.w2.bias"),   SB_FFN);
    b->w3_w    = fp32 ? sb_upload_f32(c->st_enc, K("mlp.w3.weight"), (size_t)SB_DIM * SB_FFN)
                       : sb_upload_f16(c->st_enc, K("mlp.w3.weight"), (size_t)SB_DIM * SB_FFN);
    b->w3_b    = sb_upload_f32(c->st_enc, K("mlp.w3.bias"),   SB_DIM);
    #undef K
    if (!b->norm1_w || !b->norm1_b || !b->norm2_w || !b->norm2_b ||
        !b->qkv_w || !b->qkv_b || !b->proj_w || !b->proj_b ||
        !b->ls1 || !b->ls2 ||
        !b->w1_w || !b->w1_b || !b->w2_w || !b->w2_b || !b->w3_w || !b->w3_b)
        return -1;
    return 0;
}

/* ---- decoder/MHR-head loaders (Step 5a) ---- */

/* Look up `name` in `st`, verify dtype is F32, upload to device. */
/* Return host pointer into the mmapped safetensors blob (no copy, no
 * upload). Caller must keep the st_context alive. */
static const float *sb_host_st_f32(const st_context *st, const char *name,
                                   size_t expect_n) {
    int i = safetensors_find((st_context *)st, name);
    if (i < 0) {
        fprintf(stderr, "sam3d_body: missing %s\n", name);
        return NULL;
    }
    if (strcmp(safetensors_dtype((st_context *)st, i), "F32") != 0) {
        fprintf(stderr, "sam3d_body: %s not F32 (got %s)\n",
                name, safetensors_dtype((st_context *)st, i));
        return NULL;
    }
    size_t nb = safetensors_nbytes((st_context *)st, i);
    if (expect_n && nb / 4 != expect_n) {
        fprintf(stderr, "sam3d_body: %s size %zu f32 vs %zu expected\n",
                name, nb / 4, expect_n);
        return NULL;
    }
    return (const float *)safetensors_data((st_context *)st, i);
}

static void *sb_upload_st_f32(const st_context *st, const char *name,
                              size_t expect_n) {
    int i = safetensors_find((st_context *)st, name);
    if (i < 0) {
        fprintf(stderr, "sam3d_body: missing %s\n", name);
        return NULL;
    }
    if (strcmp(safetensors_dtype((st_context *)st, i), "F32") != 0) {
        fprintf(stderr, "sam3d_body: %s not F32 (got %s)\n",
                name, safetensors_dtype((st_context *)st, i));
        return NULL;
    }
    size_t nb = safetensors_nbytes((st_context *)st, i);
    if (expect_n && nb / 4 != expect_n) {
        fprintf(stderr, "sam3d_body: %s size %zu f32 vs %zu expected\n",
                name, nb / 4, expect_n);
        return NULL;
    }
    return hip_upload_raw(safetensors_data((st_context *)st, i), nb);
}

static void *sb_upload_st_i64(const st_context *st, const char *name,
                              size_t expect_n) {
    int i = safetensors_find((st_context *)st, name);
    if (i < 0) {
        fprintf(stderr, "sam3d_body: missing %s\n", name);
        return NULL;
    }
    if (strcmp(safetensors_dtype((st_context *)st, i), "I64") != 0) {
        fprintf(stderr, "sam3d_body: %s not I64 (got %s)\n",
                name, safetensors_dtype((st_context *)st, i));
        return NULL;
    }
    size_t nb = safetensors_nbytes((st_context *)st, i);
    if (expect_n && nb / 8 != expect_n) {
        fprintf(stderr, "sam3d_body: %s size %zu i64 vs %zu expected\n",
                name, nb / 8, expect_n);
        return NULL;
    }
    return hip_upload_raw(safetensors_data((st_context *)st, i), nb);
}

static void *sb_upload_qtensor_raw(const qtensor *q, size_t expect_n,
                                   size_t elem_size, const char *name)
{
    if (!q || !q->data) {
        fprintf(stderr, "sam3d_body: missing MHR tensor %s\n", name);
        return NULL;
    }
    if (expect_n == 0) {
        fprintf(stderr, "sam3d_body: MHR tensor %s needs explicit size\n", name);
        return NULL;
    }
    size_t nb = expect_n * elem_size;
    return hip_upload_raw(q->data, nb);
}

static void cuda_sam3d_body_free_mhr_gpu_cache(cuda_sam3d_body_ctx *c)
{
    if (!c) return;
    if (c->d_mhr_pc_linear_weight) hipFree(c->d_mhr_pc_linear_weight);
    if (c->d_mhr_pc_h) hipFree(c->d_mhr_pc_h);
    if (c->d_mhr_pc_out) hipFree(c->d_mhr_pc_out);
    if (c->d_mhr_shape_coeffs) hipFree(c->d_mhr_shape_coeffs);
    if (c->d_mhr_face_coeffs) hipFree(c->d_mhr_face_coeffs);
    if (c->d_mhr_jstate) hipFree(c->d_mhr_jstate);
    if (c->d_mhr_blend_vectors) hipFree(c->d_mhr_blend_vectors);
    if (c->d_mhr_blend_base) hipFree(c->d_mhr_blend_base);
    if (c->d_mhr_face_vectors) hipFree(c->d_mhr_face_vectors);
    if (c->d_mhr_skin_indices) hipFree(c->d_mhr_skin_indices);
    if (c->d_mhr_skin_weights) hipFree(c->d_mhr_skin_weights);
    if (c->d_mhr_vert_indices) hipFree(c->d_mhr_vert_indices);
    if (c->d_mhr_rest) hipFree(c->d_mhr_rest);
    if (c->d_mhr_face) hipFree(c->d_mhr_face);
    if (c->d_mhr_skinned) hipFree(c->d_mhr_skinned);
    if (c->d_mhr_gskel) hipFree(c->d_mhr_gskel);
    if (c->d_mhr_keypoint_mapping) hipFree(c->d_mhr_keypoint_mapping);
    if (c->d_mhr_kp3d) hipFree(c->d_mhr_kp3d);
    if (c->d_mhr_kp2d_full) hipFree(c->d_mhr_kp2d_full);
    if (c->d_mhr_kp2d_crop) hipFree(c->d_mhr_kp2d_crop);
    if (c->d_mhr_kp2d_depth) hipFree(c->d_mhr_kp2d_depth);
    if (c->d_mhr_pred_cam_t) hipFree(c->d_mhr_pred_cam_t);
    c->d_mhr_pc_linear_weight = NULL;
    c->d_mhr_pc_h = NULL;
    c->d_mhr_pc_out = NULL;
    c->d_mhr_shape_coeffs = NULL;
    c->d_mhr_face_coeffs = NULL;
    c->d_mhr_jstate = NULL;
    c->d_mhr_blend_vectors = NULL;
    c->d_mhr_blend_base = NULL;
    c->d_mhr_face_vectors = NULL;
    c->d_mhr_skin_indices = NULL;
    c->d_mhr_skin_weights = NULL;
    c->d_mhr_vert_indices = NULL;
    c->d_mhr_rest = NULL;
    c->d_mhr_face = NULL;
    c->d_mhr_skinned = NULL;
    c->d_mhr_gskel = NULL;
    c->d_mhr_keypoint_mapping = NULL;
    c->d_mhr_kp3d = NULL;
    c->d_mhr_kp2d_full = NULL;
    c->d_mhr_kp2d_crop = NULL;
    c->d_mhr_kp2d_depth = NULL;
    c->d_mhr_pred_cam_t = NULL;
    c->mhr_gpu_enabled = 0;
}

static int cuda_sam3d_body_ensure_mhr_gpu_cache(cuda_sam3d_body_ctx *c)
{
    if (!c || !c->mhr_gpu_requested) return CUDA_SAM3D_BODY_E_INVAL;
    if (c->mhr_gpu_enabled) return CUDA_SAM3D_BODY_E_OK;
    if (!c->cpu_mhr || !c->cpu_dec) return CUDA_SAM3D_BODY_E_INVAL;

    const sam3d_body_mhr_assets *a = c->cpu_mhr;
    const size_t Vd = (size_t)S3DM_N_VERTS * 3;
    const int K = c->cpu_dec->n_keypoints;
    c->d_mhr_pc_linear_weight = sb_upload_qtensor_raw(
        &a->pc_linear_weight, Vd * S3DM_N_PC_H,
        sizeof(float), "pose_correctives.linear_weight");
    c->d_mhr_blend_vectors = sb_upload_qtensor_raw(
        &a->blend_shape_vectors, (size_t)S3DM_N_SHAPE * Vd,
        sizeof(float), "blend_shape.shape_vectors");
    c->d_mhr_blend_base = sb_upload_qtensor_raw(
        &a->blend_base_shape, Vd, sizeof(float), "blend_shape.base_shape");
    c->d_mhr_face_vectors = sb_upload_qtensor_raw(
        &a->face_shape_vectors, (size_t)S3DM_N_FACE * Vd,
        sizeof(float), "face_shape_vectors");
    c->d_mhr_skin_indices = sb_upload_qtensor_raw(
        &a->skin_indices_flat, S3DM_N_SKIN, sizeof(int32_t), "skin_indices_flat");
    c->d_mhr_skin_weights = sb_upload_qtensor_raw(
        &a->skin_weights_flat, S3DM_N_SKIN, sizeof(float), "skin_weights_flat");
    c->d_mhr_vert_indices = sb_upload_qtensor_raw(
        &a->vert_indices_flat, S3DM_N_SKIN, sizeof(int64_t), "vert_indices_flat");
    c->d_mhr_keypoint_mapping = sb_upload_qtensor_raw(
        &c->cpu_dec->keypoint_mapping,
        (size_t)c->cpu_dec->keypoint_mapping.dims[0] *
        (size_t)c->cpu_dec->keypoint_mapping.dims[1],
        sizeof(float), "keypoint_mapping");

    if (c->d_mhr_pc_linear_weight &&
        c->d_mhr_blend_vectors && c->d_mhr_blend_base &&
        c->d_mhr_face_vectors && c->d_mhr_skin_indices &&
        c->d_mhr_skin_weights && c->d_mhr_vert_indices &&
        c->d_mhr_keypoint_mapping &&
        hipMalloc(&c->d_mhr_pc_h, (size_t)S3DM_N_PC_H * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_pc_out, Vd * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_shape_coeffs, (size_t)S3DM_N_SHAPE * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_face_coeffs, (size_t)S3DM_N_FACE * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_jstate, (size_t)S3DM_N_JOINTS * 8 * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_rest, Vd * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_face, Vd * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_skinned, Vd * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_gskel, (size_t)S3DM_N_JOINTS * 8 * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_kp3d, (size_t)K * 3 * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_kp2d_full, (size_t)K * 2 * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_kp2d_crop, (size_t)K * 2 * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_kp2d_depth, (size_t)K * sizeof(float)) == hipSuccess &&
        hipMalloc(&c->d_mhr_pred_cam_t, 3 * sizeof(float)) == hipSuccess) {
        c->mhr_gpu_enabled = 1;
        if (c->verbose >= 1)
            fprintf(stderr, "sam3d_body: hybrid GPU MHR enabled "
                            "(cached MHR weights + keypoints)\n");
        return CUDA_SAM3D_BODY_E_OK;
    }

    fprintf(stderr, "sam3d_body: hybrid GPU MHR cache failed; "
                    "falling back to CPU MHR\n");
    cuda_sam3d_body_free_mhr_gpu_cache(c);
    c->mhr_gpu_requested = 0;
    return CUDA_SAM3D_BODY_E_LOAD;
}

/* Per-layer load. Returns 0 on success, -1 on failure. */
static int sb_load_dec_layer(cuda_sam3d_body_ctx *c, int li) {
    sb_dec_layer_w *L = &c->dec.layers[li];
    char p[192];
    const st_context *dec = c->st_dec;

    #define LD_LN(prefix, fld_w, fld_b)                                 \
        do {                                                            \
            snprintf(p, sizeof(p),                                      \
                "decoder.layers.%d." prefix ".weight", li);             \
            L->fld_w = sb_upload_st_f32(dec, p, SB_DEC_DIM);            \
            snprintf(p, sizeof(p),                                      \
                "decoder.layers.%d." prefix ".bias", li);               \
            L->fld_b = sb_upload_st_f32(dec, p, SB_DEC_DIM);            \
            if (!L->fld_w || !L->fld_b) return -1;                      \
        } while (0)

    /* For LNs that normalize the (1280,) image stream. */
    #define LD_LN_KV(prefix, fld_w, fld_b)                              \
        do {                                                            \
            snprintf(p, sizeof(p),                                      \
                "decoder.layers.%d." prefix ".weight", li);             \
            L->fld_w = sb_upload_st_f32(dec, p, SB_DEC_KV_DIM);         \
            snprintf(p, sizeof(p),                                      \
                "decoder.layers.%d." prefix ".bias", li);               \
            L->fld_b = sb_upload_st_f32(dec, p, SB_DEC_KV_DIM);         \
            if (!L->fld_w || !L->fld_b) return -1;                      \
        } while (0)

    #define LD_PROJ(prefix, fld_w, fld_b, w_n, b_n)                     \
        do {                                                            \
            snprintf(p, sizeof(p),                                      \
                "decoder.layers.%d." prefix ".weight", li);             \
            L->fld_w = sb_upload_st_f32(dec, p, w_n);                   \
            snprintf(p, sizeof(p),                                      \
                "decoder.layers.%d." prefix ".bias", li);               \
            L->fld_b = sb_upload_st_f32(dec, p, b_n);                   \
            if (!L->fld_w || !L->fld_b) return -1;                      \
        } while (0)

    LD_LN("ln1",     ln1_w,     ln1_b);
    LD_LN("ln_pe_1", ln_pe_1_w, ln_pe_1_b);
    LD_LN("ln2_1",   ln2_1_w,   ln2_1_b);
    LD_LN_KV("ln2_2",   ln2_2_w,   ln2_2_b);
    LD_LN_KV("ln_pe_2", ln_pe_2_w, ln_pe_2_b);
    LD_LN("ln3",     ln3_w,     ln3_b);

    /* Self-attn: q/k/v 1024→512, proj 512→1024. */
    LD_PROJ("self_attn.q_proj", sa_q_w, sa_q_b,
            (size_t)SB_DEC_PROJ_DIM * SB_DEC_DIM, SB_DEC_PROJ_DIM);
    LD_PROJ("self_attn.k_proj", sa_k_w, sa_k_b,
            (size_t)SB_DEC_PROJ_DIM * SB_DEC_DIM, SB_DEC_PROJ_DIM);
    LD_PROJ("self_attn.v_proj", sa_v_w, sa_v_b,
            (size_t)SB_DEC_PROJ_DIM * SB_DEC_DIM, SB_DEC_PROJ_DIM);
    LD_PROJ("self_attn.proj", sa_proj_w, sa_proj_b,
            (size_t)SB_DEC_DIM * SB_DEC_PROJ_DIM, SB_DEC_DIM);

    /* Cross-attn: q 1024→512, k/v 1280→512, proj 512→1024. */
    LD_PROJ("cross_attn.q_proj", ca_q_w, ca_q_b,
            (size_t)SB_DEC_PROJ_DIM * SB_DEC_DIM, SB_DEC_PROJ_DIM);
    LD_PROJ("cross_attn.k_proj", ca_k_w, ca_k_b,
            (size_t)SB_DEC_PROJ_DIM * SB_DEC_KV_DIM, SB_DEC_PROJ_DIM);
    LD_PROJ("cross_attn.v_proj", ca_v_w, ca_v_b,
            (size_t)SB_DEC_PROJ_DIM * SB_DEC_KV_DIM, SB_DEC_PROJ_DIM);
    LD_PROJ("cross_attn.proj", ca_proj_w, ca_proj_b,
            (size_t)SB_DEC_DIM * SB_DEC_PROJ_DIM, SB_DEC_DIM);

    /* FFN: 1024→1024→1024 (ReLU between, per CPU port). */
    LD_PROJ("ffn.layers.0.0", ffn0_w, ffn0_b,
            (size_t)SB_DEC_FFN * SB_DEC_DIM, SB_DEC_FFN);
    LD_PROJ("ffn.layers.1", ffn1_w, ffn1_b,
            (size_t)SB_DEC_DIM * SB_DEC_FFN, SB_DEC_DIM);

    #undef LD_LN
    #undef LD_LN_KV
    #undef LD_PROJ
    return 0;
}

/* Resolve a per-variant safetensors path. Tries
 *   <dir>/sam3d_body_<variant>_<bucket>.safetensors
 * first, then falls back to the legacy unprefixed alias
 *   <dir>/sam3d_body_<bucket>.safetensors
 * (kept for the dinov3 default — matches the slicer in
 * cpu/sam3d_body/convert_ckpt.py). Returns NULL only if neither exists. */
static char *sb_resolve_variant_path(const char *dir, const char *bucket,
                                     cuda_sam3d_body_backbone_t backbone,
                                     char *buf, size_t buflen)
{
    const char *tag = (backbone == CUDA_SAM3D_BODY_BACKBONE_VITH) ? "vith"
                                                                  : "dinov3";
    snprintf(buf, buflen, "%s/sam3d_body_%s_%s.safetensors", dir, tag, bucket);
    FILE *f = fopen(buf, "rb");
    if (f) { fclose(f); return buf; }
    snprintf(buf, buflen, "%s/sam3d_body_%s.safetensors", dir, bucket);
    f = fopen(buf, "rb");
    if (f) { fclose(f); return buf; }
    return NULL;
}

static int sb_load_decoder(cuda_sam3d_body_ctx *c) {
    if (c->dec.loaded) return 0;
    char path[1024];
    if (!sb_resolve_variant_path(c->cfg.safetensors_dir, "decoder",
                                 c->cfg.backbone, path, sizeof(path))) {
        fprintf(stderr, "sam3d_body: cannot find decoder safetensors in %s "
                        "(tried sam3d_body_{vith,dinov3}_decoder + legacy)\n",
                c->cfg.safetensors_dir);
        return -1;
    }
    c->st_dec = safetensors_open(path);
    if (!c->st_dec) {
        fprintf(stderr, "sam3d_body: cannot open %s\n", path);
        return -1;
    }
    if (!sb_resolve_variant_path(c->cfg.safetensors_dir, "mhr_head",
                                 c->cfg.backbone, path, sizeof(path))) {
        fprintf(stderr, "sam3d_body: cannot find mhr_head safetensors in %s\n",
                c->cfg.safetensors_dir);
        return -1;
    }
    c->st_mhr = safetensors_open(path);
    if (!c->st_mhr) {
        fprintf(stderr, "sam3d_body: cannot open %s\n", path);
        return -1;
    }

    sb_dec_w *D = &c->dec;
    const st_context *dec = c->st_dec;
    const st_context *mhr = c->st_mhr;

    /* ray_cond_emb (conv has bias=False, omit). */
    D->ray_cond_conv_w = sb_upload_st_f32(dec, "ray_cond_emb.conv.weight",
                                          (size_t)SB_DEC_KV_DIM * 1379);
    D->ray_cond_norm_w = sb_upload_st_f32(dec, "ray_cond_emb.norm.weight",
                                          SB_DEC_KV_DIM);
    D->ray_cond_norm_b = sb_upload_st_f32(dec, "ray_cond_emb.norm.bias",
                                          SB_DEC_KV_DIM);

    /* Token construction. */
    D->init_to_token_w = sb_upload_st_f32(dec, "init_to_token_mhr.weight",
                                          (size_t)SB_DEC_DIM *
                                          (SB_DEC_NPOSE + SB_DEC_NCAM + SB_DEC_COND));
    D->init_to_token_b = sb_upload_st_f32(dec, "init_to_token_mhr.bias",
                                          SB_DEC_DIM);
    D->prev_to_token_w = sb_upload_st_f32(dec, "prev_to_token_mhr.weight",
                                          (size_t)SB_DEC_DIM *
                                          (SB_DEC_NPOSE + SB_DEC_NCAM));
    D->prev_to_token_b = sb_upload_st_f32(dec, "prev_to_token_mhr.bias",
                                          SB_DEC_DIM);
    D->prompt_to_token_w = sb_upload_st_f32(dec, "prompt_to_token.weight",
                                            (size_t)SB_DEC_DIM * SB_DEC_KV_DIM);
    D->prompt_to_token_b = sb_upload_st_f32(dec, "prompt_to_token.bias",
                                            SB_DEC_DIM);
    D->keypoint_embedding   = sb_upload_st_f32(dec, "keypoint_embedding.weight",
                                               (size_t)SB_DEC_KP * SB_DEC_DIM);
    D->keypoint3d_embedding = sb_upload_st_f32(dec, "keypoint3d_embedding.weight",
                                               (size_t)SB_DEC_KP * SB_DEC_DIM);
    D->hand_box_embedding   = sb_upload_st_f32(dec, "hand_box_embedding.weight",
                                               (size_t)SB_DEC_HAND_TOK * SB_DEC_DIM);
    D->prompt_pe_gauss = sb_upload_st_f32(dec,
        "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
        2 * 640);
    D->invalid_point_embed = sb_upload_st_f32(dec,
        "prompt_encoder.invalid_point_embed.weight",
        (size_t)SB_DEC_KV_DIM);
    D->no_mask_embed = sb_upload_st_f32(dec,
        "prompt_encoder.no_mask_embed.weight",
        (size_t)SB_DEC_KV_DIM);
    D->no_mask_embed_h = sb_host_st_f32(dec,
        "prompt_encoder.no_mask_embed.weight",
        (size_t)SB_DEC_KV_DIM);

    /* Keypoint-token update path. */
    D->kp_feat_linear_w = sb_upload_st_f32(dec, "keypoint_feat_linear.weight",
                                           (size_t)SB_DEC_DIM * SB_DEC_KV_DIM);
    D->kp_feat_linear_b = sb_upload_st_f32(dec, "keypoint_feat_linear.bias",
                                           SB_DEC_DIM);
    D->kp_posemb_l0_w = sb_upload_st_f32(dec,
        "keypoint_posemb_linear.layers.0.0.weight", (size_t)SB_DEC_DIM * 2);
    D->kp_posemb_l0_b = sb_upload_st_f32(dec,
        "keypoint_posemb_linear.layers.0.0.bias",   SB_DEC_DIM);
    D->kp_posemb_l1_w = sb_upload_st_f32(dec,
        "keypoint_posemb_linear.layers.1.weight",   (size_t)SB_DEC_DIM * SB_DEC_DIM);
    D->kp_posemb_l1_b = sb_upload_st_f32(dec,
        "keypoint_posemb_linear.layers.1.bias",     SB_DEC_DIM);
    D->kp3d_posemb_l0_w = sb_upload_st_f32(dec,
        "keypoint3d_posemb_linear.layers.0.0.weight", (size_t)SB_DEC_DIM * 3);
    D->kp3d_posemb_l0_b = sb_upload_st_f32(dec,
        "keypoint3d_posemb_linear.layers.0.0.bias",   SB_DEC_DIM);
    D->kp3d_posemb_l1_w = sb_upload_st_f32(dec,
        "keypoint3d_posemb_linear.layers.1.weight",   (size_t)SB_DEC_DIM * SB_DEC_DIM);
    D->kp3d_posemb_l1_b = sb_upload_st_f32(dec,
        "keypoint3d_posemb_linear.layers.1.bias",     SB_DEC_DIM);

    if (!D->ray_cond_conv_w || !D->ray_cond_norm_w || !D->ray_cond_norm_b ||
        !D->init_to_token_w || !D->init_to_token_b ||
        !D->prev_to_token_w || !D->prev_to_token_b ||
        !D->prompt_to_token_w || !D->prompt_to_token_b ||
        !D->keypoint_embedding || !D->keypoint3d_embedding ||
        !D->hand_box_embedding ||
        !D->prompt_pe_gauss || !D->invalid_point_embed ||
        !D->no_mask_embed ||
        !D->no_mask_embed_h ||
        !D->kp_feat_linear_w || !D->kp_feat_linear_b ||
        !D->kp_posemb_l0_w || !D->kp_posemb_l0_b ||
        !D->kp_posemb_l1_w || !D->kp_posemb_l1_b ||
        !D->kp3d_posemb_l0_w || !D->kp3d_posemb_l0_b ||
        !D->kp3d_posemb_l1_w || !D->kp3d_posemb_l1_b)
        return -1;

    for (int li = 0; li < SB_DEC_LAYERS; li++) {
        if (sb_load_dec_layer(c, li) < 0) {
            fprintf(stderr, "sam3d_body: decoder layer %d load failed\n", li);
            return -1;
        }
    }

    D->norm_final_w = sb_upload_st_f32(dec, "decoder.norm_final.weight",
                                       SB_DEC_DIM);
    D->norm_final_b = sb_upload_st_f32(dec, "decoder.norm_final.bias",
                                       SB_DEC_DIM);
    if (!D->norm_final_w || !D->norm_final_b) return -1;

    /* MHR head regression (mhr_head safetensors). */
    D->head_pose_l0_w   = sb_upload_st_f32(mhr,
        "head_pose.proj.layers.0.0.weight", (size_t)SB_DEC_DIM * SB_DEC_DIM);
    D->head_pose_l0_b   = sb_upload_st_f32(mhr,
        "head_pose.proj.layers.0.0.bias",   SB_DEC_DIM);
    D->head_pose_l1_w   = sb_upload_st_f32(mhr,
        "head_pose.proj.layers.1.weight",   (size_t)SB_DEC_NPOSE * SB_DEC_DIM);
    D->head_pose_l1_b   = sb_upload_st_f32(mhr,
        "head_pose.proj.layers.1.bias",     SB_DEC_NPOSE);
    D->head_camera_l0_w = sb_upload_st_f32(mhr,
        "head_camera.proj.layers.0.0.weight", (size_t)SB_DEC_DIM * SB_DEC_DIM);
    D->head_camera_l0_b = sb_upload_st_f32(mhr,
        "head_camera.proj.layers.0.0.bias",   SB_DEC_DIM);
    D->head_camera_l1_w = sb_upload_st_f32(mhr,
        "head_camera.proj.layers.1.weight",   (size_t)SB_DEC_NCAM * SB_DEC_DIM);
    D->head_camera_l1_b = sb_upload_st_f32(mhr,
        "head_camera.proj.layers.1.bias",     SB_DEC_NCAM);
    D->init_pose   = sb_upload_st_f32(mhr, "init_pose.weight",   SB_DEC_NPOSE);
    D->init_camera = sb_upload_st_f32(mhr, "init_camera.weight", SB_DEC_NCAM);

    if (!D->head_pose_l0_w || !D->head_pose_l0_b ||
        !D->head_pose_l1_w || !D->head_pose_l1_b ||
        !D->head_camera_l0_w || !D->head_camera_l0_b ||
        !D->head_camera_l1_w || !D->head_camera_l1_b ||
        !D->init_pose || !D->init_camera)
        return -1;

    /* Host (mmap-backed) views of the same tensors for the CPU
     * norm_and_heads tail. */
    D->norm_final_w_h = sb_host_st_f32(dec, "decoder.norm_final.weight",
                                       SB_DEC_DIM);
    D->norm_final_b_h = sb_host_st_f32(dec, "decoder.norm_final.bias",
                                       SB_DEC_DIM);
    D->head_pose_l0_w_h = sb_host_st_f32(mhr,
        "head_pose.proj.layers.0.0.weight", (size_t)SB_DEC_DIM * SB_DEC_DIM);
    D->head_pose_l0_b_h = sb_host_st_f32(mhr,
        "head_pose.proj.layers.0.0.bias",   SB_DEC_DIM);
    D->head_pose_l1_w_h = sb_host_st_f32(mhr,
        "head_pose.proj.layers.1.weight",   (size_t)SB_DEC_NPOSE * SB_DEC_DIM);
    D->head_pose_l1_b_h = sb_host_st_f32(mhr,
        "head_pose.proj.layers.1.bias",     SB_DEC_NPOSE);
    D->head_camera_l0_w_h = sb_host_st_f32(mhr,
        "head_camera.proj.layers.0.0.weight", (size_t)SB_DEC_DIM * SB_DEC_DIM);
    D->head_camera_l0_b_h = sb_host_st_f32(mhr,
        "head_camera.proj.layers.0.0.bias",   SB_DEC_DIM);
    D->head_camera_l1_w_h = sb_host_st_f32(mhr,
        "head_camera.proj.layers.1.weight",   (size_t)SB_DEC_NCAM * SB_DEC_DIM);
    D->head_camera_l1_b_h = sb_host_st_f32(mhr,
        "head_camera.proj.layers.1.bias",     SB_DEC_NCAM);
    if (!D->norm_final_w_h || !D->norm_final_b_h ||
        !D->head_pose_l0_w_h || !D->head_pose_l0_b_h ||
        !D->head_pose_l1_w_h || !D->head_pose_l1_b_h ||
        !D->head_camera_l0_w_h || !D->head_camera_l0_b_h ||
        !D->head_camera_l1_w_h || !D->head_camera_l1_b_h)
        return -1;

    /* MHR-head decode tables. */
    D->scale_mean       = sb_upload_st_f32(mhr, "head_pose.scale_mean",      68);
    D->scale_comps      = sb_upload_st_f32(mhr, "head_pose.scale_comps",     28 * 68);
    D->hand_pose_mean   = sb_upload_st_f32(mhr, "head_pose.hand_pose_mean",  54);
    D->hand_pose_comps  = sb_upload_st_f32(mhr, "head_pose.hand_pose_comps", 54 * 54);
    D->hand_idx_left    = sb_upload_st_i64(mhr, "head_pose.hand_joint_idxs_left",  27);
    D->hand_idx_right   = sb_upload_st_i64(mhr, "head_pose.hand_joint_idxs_right", 27);
    D->keypoint_mapping = sb_upload_st_f32(mhr, "head_pose.keypoint_mapping",
                                           (size_t)308 * 18566);
    D->local_to_world_wrist = sb_upload_st_f32(mhr,
        "head_pose.local_to_world_wrist", 3 * 3);
    D->right_wrist_coords   = sb_upload_st_f32(mhr,
        "head_pose.right_wrist_coords",   3);
    D->root_coords          = sb_upload_st_f32(mhr,
        "head_pose.root_coords",          3);
    D->nonhand_param_idxs   = sb_upload_st_i64(mhr,
        "head_pose.nonhand_param_idxs",   145);

    if (!D->scale_mean || !D->scale_comps ||
        !D->hand_pose_mean || !D->hand_pose_comps ||
        !D->hand_idx_left || !D->hand_idx_right ||
        !D->keypoint_mapping || !D->local_to_world_wrist ||
        !D->right_wrist_coords || !D->root_coords ||
        !D->nonhand_param_idxs)
        return -1;

    /* faces stays on host — used by OBJ writer + ctx->faces. */
    {
        int fi = safetensors_find((st_context *)mhr, "head_pose.faces");
        if (fi < 0 ||
            strcmp(safetensors_dtype((st_context *)mhr, fi), "I64") != 0) {
            fprintf(stderr, "sam3d_body: head_pose.faces missing or not I64\n");
            return -1;
        }
        size_t nb = safetensors_nbytes((st_context *)mhr, fi);
        size_t nf = nb / (sizeof(int64_t) * 3);
        D->host_faces_i64 = (int64_t *)malloc(nb);
        if (!D->host_faces_i64) return -1;
        memcpy(D->host_faces_i64,
               safetensors_data((st_context *)mhr, fi), nb);
        D->n_faces = (int)nf;
    }

    D->loaded = 1;
    if (c->verbose >= 1)
        fprintf(stderr, "sam3d_body: decoder + MHR-head loaded "
                        "(6 layers, 145-token body branch, %d faces)\n",
                D->n_faces);
    return 0;
}

static int sb_compile(cuda_sam3d_body_ctx *c) {
    size_t la = strlen(cuda_kernels_common_src);
    size_t lb = strlen(cuda_sam3d_body_kernels_src);
    char *src = (char *)malloc(la + lb + 1);
    if (!src) return -1;
    memcpy(src, cuda_kernels_common_src, la);
    memcpy(src + la, cuda_sam3d_body_kernels_src, lb + 1);
    int rc = hip_compile_kernels(&c->mod, c->device_id, src,
                                 "sam3d_body_kernels", c->verbose, "sam3d_body");
    free(src);
    if (rc < 0) return -1;
    c->sm = rc;
    #define BIND(field, name) do { \
        if (hipModuleGetFunction(&c->field, c->mod, name) != hipSuccess) { \
            fprintf(stderr, "sam3d_body: missing kernel %s\n", name); \
            return -1; \
        } \
    } while (0)
    BIND(fn_sentinel, "sam3d_body_sentinel");
    BIND(fn_resize,   "resize_normalize");
    BIND(fn_patch,    "patch_embed_sam3d");
    BIND(fn_patch_im2col, "patch_im2col_sam3d");
    BIND(fn_prepend,  "prepend_special_tokens");
    BIND(fn_bf16_round,     "bf16_round_inplace_f32");
    BIND(fn_ln,             "layernorm_f32");
    BIND(fn_ln_sqrtdiv,     "layernorm_sqrtdiv_f32");
    BIND(fn_ln_welford,     "layernorm_welford_f32");
    BIND(fn_ln_welford_warp, "layernorm_welford_warp_f32");
    BIND(fn_ln_torchvec,    "layernorm_torchvec_f32");
    BIND(fn_gemm_tiled,     "gemm_tiled_f16_f32");
    BIND(fn_gemm_tiled_f32, "gemm_tiled_f32_f32");
    BIND(fn_rope_qk,        "rope_apply_qk_rh_sam3d");
    BIND(fn_kv_tx,          "kv_transpose");
    BIND(fn_qkv_tx,         "qkv_transpose_heads_f32");
    BIND(fn_heads_merge,    "heads_to_interleaved_f32");
    BIND(fn_attn_prob_v_serial, "attn_prob_v_serial_f32");
    BIND(fn_scale_rows,     "scale_rows_f32");
    BIND(fn_scale_softmax,  "scale_softmax_rows_f32");
    BIND(fn_softmax_warp,   "softmax_warp_1024_rows_f32");
    BIND(fn_fa,             "flash_attn_tiled_f32");
    BIND(fn_sdpa_qkv,       "sdpa_qkv_t_f32");
    BIND(fn_silu_mul,       "silu_mul_f32");
    BIND(fn_layerscale_add, "layerscale_add_f32");
    BIND(fn_ray_cond_fourier, "ray_cond_fourier_chw_f32");
    BIND(fn_encoder_to_preconv, "encoder_tokens_to_preconv_nomask_f32");
    BIND(fn_chw_to_tok,        "chw_to_tok_f32");
    BIND(fn_conv1x1_chw,      "conv1x1_chw_f32");
    BIND(fn_ln_chw,           "layernorm_chw_f32");
    BIND(fn_linear_bias,      "linear_f32_bias");
    BIND(fn_add_bias_rows,    "add_bias_rows_f32");
    BIND(fn_gemm_f32,         "gemm_f32_bias");
    BIND(fn_add_two,          "add_two_f32");
    BIND(fn_add_inplace,      "add_inplace_f32");
    BIND(fn_gelu_inplace,     "gelu_inplace_f32");
    BIND(fn_sdpa,             "sdpa_f32");
    BIND(fn_relu_inplace,     "relu_inplace_f32");
    BIND(fn_grid_sample,      "grid_sample_chw_f32");
    BIND(fn_kp_pelvis_norm,   "kp_pelvis_norm_f32");
    BIND(fn_augment_overwrite_mask, "augment_overwrite_with_mask_f32");
    BIND(fn_dense_pe_tok,     "dense_pe_tok_f32");
    BIND(fn_patch_pad2,     "patch_embed_pad2_f32");
    BIND(fn_pos_embed_vith, "pos_embed_add_vith_f32");
    BIND(fn_qkv_split,      "qkv_split_f32");
    BIND(fn_mhr_blend,      "mhr_blend_combine_f32");
    BIND(fn_mhr_lbs,        "mhr_lbs_skin_f32");
    BIND(fn_mhr_matvec,     "mhr_matvec_f32");
    BIND(fn_mhr_keypoints,  "mhr_keypoints_from_mesh_f32");
    BIND(fn_mhr_project,    "mhr_camera_project_f32");
    #undef BIND
    return 0;
}

static int sb_launch(hipFunction_t fn, unsigned gx, unsigned gy, unsigned gz,
                     unsigned bx, unsigned by, unsigned bz,
                     unsigned shmem, void *p, size_t pb) {
    void *cfg[] = { CU_LAUNCH_PARAM_BUFFER_POINTER, p,
                    CU_LAUNCH_PARAM_BUFFER_SIZE,    &pb,
                    CU_LAUNCH_PARAM_END };
    hipError_t e = hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, bz, shmem, 0, NULL, cfg);
    if (e != hipSuccess) {
        fprintf(stderr, "sam3d_body: launch err=%d grid=%ux%ux%u block=%ux%ux%u\n",
                (int)e, gx, gy, gz, bx, by, bz);
        return -1;
    }
    return 0;
}

static int sb_precision_is_bf16(const cuda_sam3d_body_ctx *ctx)
{
    const char *p = ctx && ctx->cfg.precision ? ctx->cfg.precision : NULL;
    return p && p[0] && !strcmp(p, "bf16");
}

static int sb_precision_is_fp32(const cuda_sam3d_body_ctx *ctx)
{
    const char *p = ctx && ctx->cfg.precision ? ctx->cfg.precision : NULL;
    return p && p[0] && !strcmp(p, "fp32");
}

static int sb_precision_is_supported(const char *p)
{
    return !p || !p[0] || !strcmp(p, "bf16") ||
           !strcmp(p, "fp16") || !strcmp(p, "fp32");
}

static int sb_bf16_round(cuda_sam3d_body_ctx *ctx, void *x, int n)
{
    if (!sb_precision_is_bf16(ctx) || !x || n <= 0) return 0;
    struct __attribute__((packed)) {
        void *x; int n;
    } p = { x, n };
    return sb_launch(ctx->fn_bf16_round, (unsigned)((n + 255) / 256), 1, 1,
                     256, 1, 1, 0, &p, sizeof(p));
}

static int sb_gemm_encoder(cuda_sam3d_body_ctx *ctx, void *Y,
                           const void *W, const void *X, const void *bias,
                           int n_out, int n_in, int n_tok)
{
    if (ctx && ctx->use_cublas && sb_precision_is_fp32(ctx) &&
        getenv("SAM3D_BODY_CUBLAS_GEMM") &&
        cublasew_gemm_f32_pedantic_rowmajor_nt(ctx->cublas,
                                               (CUdeviceptr)(uintptr_t)Y,
                                               (CUdeviceptr)(uintptr_t)W,
                                               (CUdeviceptr)(uintptr_t)X,
                                               n_tok, n_out, n_in) == 0) {
        if (bias) {
            struct __attribute__((packed)) {
                void *x; const void *b; int N, D;
            } p = { Y, bias, n_tok, n_out };
            int total = n_tok * n_out;
            if (sb_launch(ctx->fn_add_bias_rows,
                          (unsigned)((total + 255) / 256), 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return -1;
        }
        return 0;
    }

    hipFunction_t fn = sb_precision_is_fp32(ctx) ?
        ctx->fn_gemm_tiled_f32 : ctx->fn_gemm_tiled;
    struct __attribute__((packed)) {
        void *Y; const void *W, *X, *bias; int n_out, n_in, n_tok;
    } p = { Y, W, X, bias, n_out, n_in, n_tok };
    return sb_launch(fn,
                     (unsigned)((n_out + 63) / 64),
                     (unsigned)((n_tok + 15) / 16), 1, 16, 16, 1,
                     0, &p, sizeof(p));
}

/* ===================== public API ===================== */

cuda_sam3d_body_ctx *cuda_sam3d_body_create(const cuda_sam3d_body_config *cfg)
{
    if (!cfg || !cfg->safetensors_dir) return NULL;
    if (!sb_precision_is_supported(cfg->precision)) {
        fprintf(stderr, "sam3d_body: unsupported precision '%s' (expected fp16, fp32 or bf16)\n",
                cfg->precision);
        return NULL;
    }

    cuda_sam3d_body_ctx *c = (cuda_sam3d_body_ctx *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cfg = *cfg;
    c->device_id = cfg->device_ordinal;
    c->verbose = cfg->verbose;
    if (c->cfg.image_size <= 0) c->cfg.image_size = SB_IMG;
    if (c->cfg.image_height <= 0) c->cfg.image_height = c->cfg.image_size;
    if (c->cfg.image_width  <= 0) c->cfg.image_width  = c->cfg.image_size;
    if (c->cfg.backbone != CUDA_SAM3D_BODY_BACKBONE_VITH) {
        if (c->cfg.image_height <= 0 || c->cfg.image_width <= 0 ||
            (c->cfg.image_height % SB_PATCH) != 0 ||
            (c->cfg.image_width  % SB_PATCH) != 0) {
            fprintf(stderr,
                    "sam3d_body: DINOv3 image shape must be positive and "
                    "divisible by patch size %d, got %dx%d\n",
                    SB_PATCH, c->cfg.image_height, c->cfg.image_width);
            free(c); return NULL;
        }
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "sam3d_body: cuew init failed\n");
        free(c); return NULL;
    }
    hipError_t set_device_err = hipSetDevice(c->device_id);
    if (set_device_err != hipSuccess) {
        const char *err_s = NULL;
        hipGetErrorString(set_device_err, &err_s);
        fprintf(stderr, "sam3d_body: hipSetDevice(%d) failed: %s (%d)\n",
                c->device_id, err_s ? err_s : "unknown",
                (int)set_device_err);
        free(c); return NULL;
    }
    if (sb_compile(c) < 0) {
        fprintf(stderr, "sam3d_body: kernel compile failed\n");
        free(c); return NULL;
    }
    if (sb_precision_is_fp32(c) &&
        (getenv("SAM3D_BODY_CUBLAS_GEMM") ||
         getenv("SAM3D_BODY_CUBLAS_PATCH") ||
         getenv("SAM3D_BODY_CUBLAS_ATTN")) &&
        cublasewCreate(&c->cublas, 0) == 0) {
        c->use_cublas = 1;
        if (c->verbose >= 1)
            fprintf(stderr, "sam3d_body: cuBLAS fp32 path enabled\n");
    }

    char path[512];

    if (c->cfg.backbone == CUDA_SAM3D_BODY_BACKBONE_VITH) {
        snprintf(path, sizeof(path), "%s/sam3d_body_vith.safetensors",
                 cfg->safetensors_dir);
        c->st_enc = safetensors_open(path);
        if (!c->st_enc) {
            fprintf(stderr, "sam3d_body_vith: cannot open %s\n", path);
            cuda_sam3d_body_destroy(c); return NULL;
        }
        if (c->verbose >= 1)
            fprintf(stderr, "sam3d_body_vith: opened %s\n", path);

        c->w_patch_w = sb_upload_vith_f32(c->st_enc, "patch_embed.proj.weight",
                                          (size_t)SB_VITH_DIM * 3 * SB_VITH_PATCH * SB_VITH_PATCH);
        c->w_patch_b = sb_upload_vith_f32(c->st_enc, "patch_embed.proj.bias",
                                          SB_VITH_DIM);
        c->w_vith_pos_embed = sb_upload_vith_f32(c->st_enc, "pos_embed",
                                                 (size_t)(SB_VITH_N_PATCH + 1) * SB_VITH_DIM);
        c->w_norm_w = sb_upload_vith_f32(c->st_enc, "last_norm.weight",
                                         SB_VITH_DIM);
        c->w_norm_b = sb_upload_vith_f32(c->st_enc, "last_norm.bias",
                                         SB_VITH_DIM);
        if (!c->w_patch_w || !c->w_patch_b ||
            !c->w_vith_pos_embed || !c->w_norm_w || !c->w_norm_b) {
            cuda_sam3d_body_destroy(c); return NULL;
        }

        for (int i = 0; i < SB_VITH_N_BLK; i++) {
            if (sb_load_block_vith(c, i) < 0) {
                fprintf(stderr, "sam3d_body_vith: failed to load block %d\n", i);
                cuda_sam3d_body_destroy(c); return NULL;
            }
        }

        c->weights_ready = 1;
        if (c->verbose >= 1)
            fprintf(stderr, "sam3d_body_vith: encoder weights uploaded "
                            "(%d blocks, D=%d, FFN=%d, head_dim=%d, sm_%d)\n",
                    SB_VITH_N_BLK, SB_VITH_DIM, SB_VITH_FFN, SB_VITH_HEAD_DIM, c->sm);

        /* Decoder + MHR loading is shared between dinov3 and vit-h (the heads
         * + MHR rig are variant-tagged but loaded by the same code below).
         * Jump past the dinov3-specific block tensor uploads and into the
         * shared decoder/MHR section. */
        goto load_decoder_and_mhr;
    }

    snprintf(path, sizeof(path), "%s/%s", cfg->safetensors_dir,
             sb_precision_is_fp32(c) ? "sam3d_body_dinov3_fp32.safetensors"
                                     : "sam3d_body_dinov3.safetensors");
    c->st_enc = safetensors_open(path);
    if (!c->st_enc) {
        fprintf(stderr, "sam3d_body: cannot open %s\n", path);
        cuda_sam3d_body_destroy(c); return NULL;
    }
    if (c->verbose >= 1)
        fprintf(stderr, "sam3d_body: opened %s\n", path);

    /* Non-block tensors. */
    c->w_patch_w       = sb_upload_f32(c->st_enc, "patch_embed.proj.weight",
                                       (size_t)SB_DIM * 3 * SB_PATCH * SB_PATCH);
    c->w_patch_b       = sb_upload_f32(c->st_enc, "patch_embed.proj.bias", SB_DIM);
    c->w_cls           = sb_upload_f32(c->st_enc, "cls_token", SB_DIM);
    c->w_storage       = sb_upload_f32(c->st_enc, "storage_tokens",
                                       (size_t)SB_N_STORAGE * SB_DIM);
    c->w_rope_periods  = sb_upload_f32(c->st_enc, "rope_embed.periods", 16);
    c->w_norm_w        = sb_upload_f32(c->st_enc, "norm.weight", SB_DIM);
    c->w_norm_b        = sb_upload_f32(c->st_enc, "norm.bias",   SB_DIM);
    if (!c->w_patch_w || !c->w_patch_b || !c->w_cls || !c->w_storage ||
        !c->w_rope_periods || !c->w_norm_w || !c->w_norm_b) {
        cuda_sam3d_body_destroy(c); return NULL;
    }

    /* Build 2D RoPE cos/sin tables host-side from rope_embed.periods.
     * Layout matches common/dinov3.h: [sin_y(rope_dim4), sin_x(rope_dim4),
     * sin_y, sin_x] tiled across head_dim, computed for each position on
     * the configured DINOv3 patch grid. */
    {
        const int gh = c->cfg.image_height / SB_PATCH;
        const int gw = c->cfg.image_width / SB_PATCH;
        const int hd = SB_HEAD_DIM;
        const int rope_dim4 = hd / 4;       /* 16 */
        const int np = gh * gw;
        int p_idx = sb_find(c->st_enc, "rope_embed.periods");
        if (p_idx < 0) { cuda_sam3d_body_destroy(c); return NULL; }
        const float *periods = (const float *)
            safetensors_data(c->st_enc, p_idx);
        float *rcos = (float *)malloc((size_t)np * hd * sizeof(float));
        float *rsin = (float *)malloc((size_t)np * hd * sizeof(float));
        if (!rcos || !rsin) {
            free(rcos); free(rsin);
            cuda_sam3d_body_destroy(c); return NULL;
        }
        for (int p = 0; p < np; p++) {
            int py = p / gw, px = p % gw;
            float cy = ((0.5f + (float)py) / (float)gh) * 2.0f - 1.0f;
            float cx = ((0.5f + (float)px) / (float)gw) * 2.0f - 1.0f;
            float *s = rsin + (size_t)p * hd;
            float *kc = rcos + (size_t)p * hd;
            for (int j = 0; j < rope_dim4; j++) {
                float ay = 2.0f * 3.14159265358979323846f * cy / periods[j];
                float ax = 2.0f * 3.14159265358979323846f * cx / periods[j];
                float sy = sinf(ay), sx = sinf(ax);
                float cyv = cosf(ay), cxv = cosf(ax);
                s[j]                = sy;
                s[rope_dim4 + j]    = sx;
                s[2 * rope_dim4 + j] = sy;
                s[3 * rope_dim4 + j] = sx;
                kc[j]                = cyv;
                kc[rope_dim4 + j]    = cxv;
                kc[2 * rope_dim4 + j] = cyv;
                kc[3 * rope_dim4 + j] = cxv;
            }
        }
        c->w_rope_cos = hip_upload_raw(rcos, (size_t)np * hd * sizeof(float));
        c->w_rope_sin = hip_upload_raw(rsin, (size_t)np * hd * sizeof(float));
        free(rcos); free(rsin);
        if (!c->w_rope_cos || !c->w_rope_sin) {
            cuda_sam3d_body_destroy(c); return NULL;
        }
        if (c->verbose >= 1)
            fprintf(stderr, "sam3d_body: built 2D RoPE tables (%dx%d, %d)\n",
                    gh, gw, hd);
    }

    /* 32 blocks. */
    for (int i = 0; i < SB_N_BLK; i++) {
        if (sb_load_block(c, i) < 0) {
            fprintf(stderr, "sam3d_body: failed to load block %d\n", i);
            cuda_sam3d_body_destroy(c); return NULL;
        }
    }

    c->weights_ready = 1;
    if (c->verbose >= 1)
        fprintf(stderr, "sam3d_body: encoder weights uploaded "
                        "(%d blocks, D=%d, FFN=%d, sm_%d)\n",
                SB_N_BLK, SB_DIM, SB_FFN, c->sm);

load_decoder_and_mhr:
    /* Decoder + MHR-head safetensors. */
    if (sb_load_decoder(c) < 0) {
        fprintf(stderr, "sam3d_body: decoder load failed\n");
        cuda_sam3d_body_destroy(c); return NULL;
    }

    /* CPU model + MHR assets for the run_decoder MHR-in-the-loop path.
     * These are optional — verify_*.c binaries that don't need a full
     * end-to-end forward (verify_dinov3, verify_ray_cond, etc.) leave
     * mhr_assets_dir NULL and the loaders are skipped. */
    {
        char p1[1024], p2[1024];
        if (!sb_resolve_variant_path(cfg->safetensors_dir, "decoder",
                                     cfg->backbone, p1, sizeof(p1)) ||
            !sb_resolve_variant_path(cfg->safetensors_dir, "mhr_head",
                                     cfg->backbone, p2, sizeof(p2))) {
            fprintf(stderr, "sam3d_body: variant safetensors missing in %s\n",
                    cfg->safetensors_dir);
            cuda_sam3d_body_destroy(c); return NULL;
        }
        c->cpu_dec = sam3d_body_decoder_load(p1, p2);
        if (!c->cpu_dec) {
            fprintf(stderr, "sam3d_body: CPU decoder_load failed (%s, %s)\n",
                    p1, p2);
            cuda_sam3d_body_destroy(c); return NULL;
        }

        if (cfg->mhr_assets_dir) {
            snprintf(p1, sizeof(p1), "%s/sam3d_body_mhr_jit.safetensors",
                     cfg->mhr_assets_dir);
            snprintf(p2, sizeof(p2), "%s/sam3d_body_mhr_jit.json",
                     cfg->mhr_assets_dir);
            c->cpu_mhr = sam3d_body_mhr_load(p1, p2);
            if (!c->cpu_mhr) {
                fprintf(stderr, "sam3d_body: MHR asset load failed (%s, %s)\n",
                        p1, p2);
                cuda_sam3d_body_destroy(c); return NULL;
            }
            const char *gpu_mhr = getenv("SAM3D_BODY_GPU_MHR");
            if (gpu_mhr && gpu_mhr[0] && strcmp(gpu_mhr, "0") != 0) {
                c->mhr_gpu_requested = 1;
                if (c->verbose >= 1)
                    fprintf(stderr, "sam3d_body: hybrid GPU MHR requested "
                                    "(lazy cache)\n");
            }
        }
    }

    return c;
}

static void sb_free_dev(void *p) { if (p) hipFree(p); }

void cuda_sam3d_body_destroy(cuda_sam3d_body_ctx *ctx)
{
    if (!ctx) return;
    sb_free_dev(ctx->w_patch_w); sb_free_dev(ctx->w_patch_b);
    sb_free_dev(ctx->w_cls); sb_free_dev(ctx->w_storage);
    sb_free_dev(ctx->w_rope_periods);
    sb_free_dev(ctx->w_rope_cos); sb_free_dev(ctx->w_rope_sin);
    sb_free_dev(ctx->w_norm_w); sb_free_dev(ctx->w_norm_b);
    sb_free_dev(ctx->w_vith_pos_embed);
    for (int i = 0; i < SB_N_BLK; i++) {
        sb_block *b = &ctx->blk[i];
        sb_free_dev(b->norm1_w); sb_free_dev(b->norm1_b);
        sb_free_dev(b->norm2_w); sb_free_dev(b->norm2_b);
        sb_free_dev(b->qkv_w);   sb_free_dev(b->qkv_b);
        sb_free_dev(b->proj_w);  sb_free_dev(b->proj_b);
        sb_free_dev(b->ls1);     sb_free_dev(b->ls2);
        sb_free_dev(b->w1_w);    sb_free_dev(b->w1_b);
        sb_free_dev(b->w2_w);    sb_free_dev(b->w2_b);
        sb_free_dev(b->w3_w);    sb_free_dev(b->w3_b);
    }
    for (int i = 0; i < SB_VITH_N_BLK; i++) {
        sb_block_vith *b = &ctx->vith_blk[i];
        sb_free_dev(b->norm1_w); sb_free_dev(b->norm1_b);
        sb_free_dev(b->norm2_w); sb_free_dev(b->norm2_b);
        sb_free_dev(b->qkv_w);   sb_free_dev(b->qkv_b);
        sb_free_dev(b->proj_w);  sb_free_dev(b->proj_b);
        sb_free_dev(b->fc1_w);   sb_free_dev(b->fc1_b);
        sb_free_dev(b->fc2_w);   sb_free_dev(b->fc2_b);
    }
    sb_free_dev(ctx->d_img_u8);
    sb_free_dev(ctx->d_img_f32);
    sb_free_dev(ctx->d_patch_cols);
    sb_free_dev(ctx->d_tok);
    sb_free_dev(ctx->d_ln);
    sb_free_dev(ctx->d_qkv);
    sb_free_dev(ctx->d_qt);
    sb_free_dev(ctx->d_kt);
    sb_free_dev(ctx->d_vt);
    sb_free_dev(ctx->d_attn_scores);
    sb_free_dev(ctx->d_attn);
    sb_free_dev(ctx->d_proj);
    sb_free_dev(ctx->d_gate);
    sb_free_dev(ctx->d_up);
    /* Decoder + MHR-head weights. */
    sb_dec_w *D = &ctx->dec;
    sb_free_dev(D->ray_cond_conv_w);
    sb_free_dev(D->ray_cond_norm_w); sb_free_dev(D->ray_cond_norm_b);
    sb_free_dev(D->init_to_token_w); sb_free_dev(D->init_to_token_b);
    sb_free_dev(D->prev_to_token_w); sb_free_dev(D->prev_to_token_b);
    sb_free_dev(D->prompt_to_token_w); sb_free_dev(D->prompt_to_token_b);
    sb_free_dev(D->keypoint_embedding);
    sb_free_dev(D->keypoint3d_embedding);
    sb_free_dev(D->hand_box_embedding);
    sb_free_dev(D->kp_feat_linear_w); sb_free_dev(D->kp_feat_linear_b);
    sb_free_dev(D->kp_posemb_l0_w); sb_free_dev(D->kp_posemb_l0_b);
    sb_free_dev(D->kp_posemb_l1_w); sb_free_dev(D->kp_posemb_l1_b);
    sb_free_dev(D->kp3d_posemb_l0_w); sb_free_dev(D->kp3d_posemb_l0_b);
    sb_free_dev(D->kp3d_posemb_l1_w); sb_free_dev(D->kp3d_posemb_l1_b);
    sb_free_dev(D->prompt_pe_gauss);
    sb_free_dev(D->invalid_point_embed);
    sb_free_dev(D->no_mask_embed);
    for (int li = 0; li < SB_DEC_LAYERS; li++) {
        sb_dec_layer_w *L = &D->layers[li];
        sb_free_dev(L->ln1_w); sb_free_dev(L->ln1_b);
        sb_free_dev(L->ln_pe_1_w); sb_free_dev(L->ln_pe_1_b);
        sb_free_dev(L->ln2_1_w); sb_free_dev(L->ln2_1_b);
        sb_free_dev(L->ln2_2_w); sb_free_dev(L->ln2_2_b);
        sb_free_dev(L->ln_pe_2_w); sb_free_dev(L->ln_pe_2_b);
        sb_free_dev(L->ln3_w); sb_free_dev(L->ln3_b);
        sb_free_dev(L->sa_q_w); sb_free_dev(L->sa_q_b);
        sb_free_dev(L->sa_k_w); sb_free_dev(L->sa_k_b);
        sb_free_dev(L->sa_v_w); sb_free_dev(L->sa_v_b);
        sb_free_dev(L->sa_proj_w); sb_free_dev(L->sa_proj_b);
        sb_free_dev(L->ca_q_w); sb_free_dev(L->ca_q_b);
        sb_free_dev(L->ca_k_w); sb_free_dev(L->ca_k_b);
        sb_free_dev(L->ca_v_w); sb_free_dev(L->ca_v_b);
        sb_free_dev(L->ca_proj_w); sb_free_dev(L->ca_proj_b);
        sb_free_dev(L->ffn0_w); sb_free_dev(L->ffn0_b);
        sb_free_dev(L->ffn1_w); sb_free_dev(L->ffn1_b);
    }
    sb_free_dev(D->norm_final_w); sb_free_dev(D->norm_final_b);
    sb_free_dev(D->head_pose_l0_w); sb_free_dev(D->head_pose_l0_b);
    sb_free_dev(D->head_pose_l1_w); sb_free_dev(D->head_pose_l1_b);
    sb_free_dev(D->head_camera_l0_w); sb_free_dev(D->head_camera_l0_b);
    sb_free_dev(D->head_camera_l1_w); sb_free_dev(D->head_camera_l1_b);
    sb_free_dev(D->init_pose); sb_free_dev(D->init_camera);
    sb_free_dev(D->scale_mean); sb_free_dev(D->scale_comps);
    sb_free_dev(D->hand_pose_mean); sb_free_dev(D->hand_pose_comps);
    sb_free_dev(D->hand_idx_left); sb_free_dev(D->hand_idx_right);
    sb_free_dev(D->keypoint_mapping);
    sb_free_dev(D->local_to_world_wrist);
    sb_free_dev(D->right_wrist_coords);
    sb_free_dev(D->root_coords);
    sb_free_dev(D->nonhand_param_idxs);
    sb_free_dev(ctx->d_mhr_pc_linear_weight);
    sb_free_dev(ctx->d_mhr_pc_h);
    sb_free_dev(ctx->d_mhr_pc_out);
    sb_free_dev(ctx->d_mhr_shape_coeffs);
    sb_free_dev(ctx->d_mhr_face_coeffs);
    sb_free_dev(ctx->d_mhr_jstate);
    sb_free_dev(ctx->d_mhr_blend_vectors);
    sb_free_dev(ctx->d_mhr_blend_base);
    sb_free_dev(ctx->d_mhr_face_vectors);
    sb_free_dev(ctx->d_mhr_skin_indices);
    sb_free_dev(ctx->d_mhr_skin_weights);
    sb_free_dev(ctx->d_mhr_vert_indices);
    sb_free_dev(ctx->d_mhr_rest);
    sb_free_dev(ctx->d_mhr_face);
    sb_free_dev(ctx->d_mhr_skinned);
    sb_free_dev(ctx->d_mhr_gskel);
    sb_free_dev(ctx->d_mhr_keypoint_mapping);
    sb_free_dev(ctx->d_mhr_kp3d);
    sb_free_dev(ctx->d_mhr_kp2d_full);
    sb_free_dev(ctx->d_mhr_kp2d_crop);
    sb_free_dev(ctx->d_mhr_kp2d_depth);
    sb_free_dev(ctx->d_mhr_pred_cam_t);
    sb_free_dev(ctx->d_kpu_img);
    sb_free_dev(ctx->d_dense_pe_tok);
    free(ctx->dense_pe_tok_h);
    free(D->host_faces_i64);
    if (ctx->cublas) cublasewDestroy(ctx->cublas);
    if (ctx->mod) hipModuleUnload(ctx->mod);
    if (ctx->st_enc) safetensors_close(ctx->st_enc);
    if (ctx->st_dec) safetensors_close(ctx->st_dec);
    if (ctx->st_mhr) safetensors_close(ctx->st_mhr);
    if (ctx->cpu_dec) sam3d_body_decoder_free(ctx->cpu_dec);
    if (ctx->cpu_mhr) sam3d_body_mhr_free(ctx->cpu_mhr);
    free(ctx->image_rgb);
    free(ctx->encoder_tokens.data);
    free(ctx->mhr_params);
    free(ctx->vertices);
    free(ctx->faces);
    free(ctx->keypoints_3d);
    free(ctx->keypoints_2d);
    free(ctx);
}

int cuda_sam3d_body_set_image(cuda_sam3d_body_ctx *ctx, const uint8_t *rgb,
                              int width, int height, const float bbox[4])
{
    if (!ctx || !rgb || width <= 0 || height <= 0) return CUDA_SAM3D_BODY_E_INVAL;
    size_t bytes = (size_t)width * height * 3;
    free(ctx->image_rgb);
    ctx->image_rgb = (uint8_t *)malloc(bytes);
    if (!ctx->image_rgb) return CUDA_SAM3D_BODY_E_INVAL;
    memcpy(ctx->image_rgb, rgb, bytes);
    ctx->img_w = width; ctx->img_h = height;
    ctx->has_bbox = 0;
    if (bbox) { memcpy(ctx->bbox, bbox, sizeof(ctx->bbox)); ctx->has_bbox = 1; }
    return CUDA_SAM3D_BODY_E_OK;
}

int cuda_sam3d_body_set_focal(cuda_sam3d_body_ctx *ctx, float f)
{
    if (!ctx) return CUDA_SAM3D_BODY_E_INVAL;
    ctx->focal_hint = f;
    return CUDA_SAM3D_BODY_E_OK;
}

static int cuda_sam3d_body_ensure_dense_pe_tok_host(
        cuda_sam3d_body_ctx *ctx, int gh, int gw, int Dc, float **out)
{
    if (!ctx || !ctx->cpu_dec || gh <= 0 || gw <= 0 || Dc <= 0 || !out)
        return CUDA_SAM3D_BODY_E_INVAL;
    const int N_C = gh * gw;
    if (ctx->dense_pe_tok_h &&
        ctx->dense_pe_gh == gh &&
        ctx->dense_pe_gw == gw &&
        ctx->dense_pe_dc == Dc) {
        *out = ctx->dense_pe_tok_h;
        return CUDA_SAM3D_BODY_E_OK;
    }

    float *image_pe_chw = (float *)malloc((size_t)Dc * N_C * sizeof(float));
    float *tok = (float *)malloc((size_t)N_C * Dc * sizeof(float));
    if (!image_pe_chw || !tok) {
        free(image_pe_chw);
        free(tok);
        return CUDA_SAM3D_BODY_E_LOAD;
    }
    if (sam3d_body_get_dense_pe(ctx->cpu_dec, gh, gw, /*n_threads=*/0,
                                image_pe_chw) != 0) {
        free(image_pe_chw);
        free(tok);
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    for (int n = 0; n < N_C; n++)
        for (int c = 0; c < Dc; c++)
            tok[(size_t)n * Dc + c] = image_pe_chw[(size_t)c * N_C + n];
    free(image_pe_chw);

    free(ctx->dense_pe_tok_h);
    ctx->dense_pe_tok_h = tok;
    ctx->dense_pe_gh = gh;
    ctx->dense_pe_gw = gw;
    ctx->dense_pe_dc = Dc;
    sb_free_dev(ctx->d_dense_pe_tok);
    ctx->d_dense_pe_tok = NULL;
    ctx->d_dense_pe_tok_bytes = 0;

    *out = ctx->dense_pe_tok_h;
    return CUDA_SAM3D_BODY_E_OK;
}

static int cuda_sam3d_body_ensure_dense_pe_tok_dev(
        cuda_sam3d_body_ctx *ctx, int gh, int gw, int Dc, void **out)
{
    if (!ctx || !out) return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->cpu_dec || !ctx->dec.prompt_pe_gauss)
        return CUDA_SAM3D_BODY_E_INVAL;
    const int npf = (int)ctx->cpu_dec->prompt_pe_gauss.dims[1];
    if (Dc != npf * 2) return CUDA_SAM3D_BODY_E_INVAL;
    const size_t bytes = (size_t)gh * (size_t)gw * (size_t)Dc *
                         sizeof(float);
    if (ctx->d_dense_pe_tok &&
        ctx->d_dense_pe_tok_bytes == bytes &&
        ctx->dense_pe_gh == gh &&
        ctx->dense_pe_gw == gw &&
        ctx->dense_pe_dc == Dc) {
        *out = ctx->d_dense_pe_tok;
        return CUDA_SAM3D_BODY_E_OK;
    }
    sb_free_dev(ctx->d_dense_pe_tok);
    ctx->d_dense_pe_tok = NULL;
    ctx->d_dense_pe_tok_bytes = 0;
    if (hipMalloc(&ctx->d_dense_pe_tok, bytes) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    struct __attribute__((packed)) {
        void *out; const void *G; int H; int W; int npf; int use_square_x;
    } p = {
        ctx->d_dense_pe_tok, ctx->dec.prompt_pe_gauss,
        gh, gw, npf, ctx->cpu_dec->kp_sample_scale_x ? 1 : 0
    };
    unsigned bx = 256;
    unsigned gx = (unsigned)(((size_t)gh * (size_t)gw * (size_t)npf +
                              bx - 1) / bx);
    if (sb_launch(ctx->fn_dense_pe_tok, gx, 1, 1, bx, 1, 1, 0,
                  &p, sizeof(p)) < 0) {
        sb_free_dev(ctx->d_dense_pe_tok);
        ctx->d_dense_pe_tok = NULL;
        return CUDA_SAM3D_BODY_E_LOAD;
    }
    ctx->d_dense_pe_tok_bytes = bytes;
    ctx->dense_pe_gh = gh;
    ctx->dense_pe_gw = gw;
    ctx->dense_pe_dc = Dc;
    *out = ctx->d_dense_pe_tok;
    return CUDA_SAM3D_BODY_E_OK;
}

/* ViT-H/16 (vit_hmr_512_384) encoder forward.
 *
 * Pipeline:
 *   patch_embed_pad2 → pos_embed_add_vith
 *   for L in 32 blocks:
 *     LN1 → QKV gemm → qkv_split → sdpa_f32 → proj gemm → residual_add
 *     LN2 → fc1 gemm → GELU → fc2 gemm → residual_add
 *   last_norm → host readback (768, 1280)
 *
 * No CLS/storage tokens, no RoPE, no LayerScale. Forward in fp32 with
 * f16 weights (matches DINOv3 path).
 *
 * Buffer reuse:
 *   d_tok   (N, D)   residual stream
 *   d_ln    (N, D)   LN scratch
 *   d_qkv   (N, 3D)  fused QKV
 *   d_kt    (N, D)   Q packed (reuse — sdpa needs (N, H*D_h)=(N, D))
 *   d_vt    (N, D)   K packed
 *   d_attn  (N, D)   V packed (reused as sdpa scratch, then attn output)
 *                    -- actually V needs to live through sdpa, so we use
 *                    -- d_proj for V and write sdpa output to d_attn.
 *   d_proj  (N, D)   V packed AND post-proj scratch (V is consumed by sdpa
 *                    before proj runs — safe to reuse).
 *   d_gate  (N, FFN) FFN intermediate (fc1 out → GELU → fc2 in).
 */
static int cuda_sam3d_body_run_encoder_vith(cuda_sam3d_body_ctx *ctx)
{
    const int IMG_H   = SB_VITH_IMG_H;
    const int IMG_W   = SB_VITH_IMG_W;
    const int GW      = SB_VITH_GRID_W;
    const int GH      = SB_VITH_GRID_H;
    const int N_TOK   = SB_VITH_N_PATCH;
    const int D       = SB_VITH_DIM;
    const int FFN     = SB_VITH_FFN;
    const int HEADS   = SB_VITH_HEADS;
    const int HEAD_DIM = SB_VITH_HEAD_DIM;

    if (!ctx->has_norm_input) {
        /* Caller (cuda_sam3d_body_run_encoder dispatch) is expected to have
         * either run host-side preprocess + upload, or populated d_img_f32
         * via debug_set_normalized_input from verify_vith. */
        fprintf(stderr, "sam3d_body_vith: d_img_f32 not populated\n");
        return CUDA_SAM3D_BODY_E_INVAL;
    }

    if (!ctx->d_tok) {
        if (hipMalloc(&ctx->d_tok,
                      (size_t)N_TOK * D * sizeof(float)) != hipSuccess)
            return CUDA_SAM3D_BODY_E_LOAD;
    }

    /* patch_embed_pad2 → d_tok rows [0..N_TOK). */
    {
        struct __attribute__((packed)) {
            void *out; const void *img; const void *w; const void *bias;
            int gw, gh, dim, ps, img_h, img_w, pad;
        } p = { ctx->d_tok, ctx->d_img_f32, ctx->w_patch_w, ctx->w_patch_b,
                GW, GH, D, SB_VITH_PATCH, IMG_H, IMG_W, SB_VITH_PAD };
        if (sb_launch(ctx->fn_patch_pad2, (unsigned)N_TOK, 1, 1, 256, 1, 1,
                      0, &p, sizeof(p)) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
    }

    /* pos_embed_add_vith — d_tok += pos[1+t] + pos[0]. */
    {
        struct __attribute__((packed)) {
            void *hidden; const void *pos; int n_patches, dim;
        } p = { ctx->d_tok, ctx->w_vith_pos_embed, N_TOK, D };
        unsigned total = (unsigned)((size_t)N_TOK * D);
        if (sb_launch(ctx->fn_pos_embed_vith, (total + 255) / 256, 1, 1,
                      256, 1, 1, 0, &p, sizeof(p)) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
    }

    /* Allocate per-block scratch on first use. Sizes match ViT-H widths. */
    #define ENSURE(field, bytes) do { \
        if (!ctx->field) { \
            if (hipMalloc(&ctx->field, (bytes)) != hipSuccess) \
                return CUDA_SAM3D_BODY_E_LOAD; \
        } \
    } while (0)
    ENSURE(d_ln,   (size_t)N_TOK * D * sizeof(float));
    ENSURE(d_qkv,  (size_t)N_TOK * 3 * D * sizeof(float));
    /* d_kt / d_vt are sized for HEADS*N_TOK*HEAD_DIM = N_TOK*D — reuse
     * them as Q-packed and K-packed (N, D). */
    ENSURE(d_kt,   (size_t)N_TOK * D * sizeof(float));
    ENSURE(d_vt,   (size_t)N_TOK * D * sizeof(float));
    ENSURE(d_attn, (size_t)N_TOK * D * sizeof(float));
    ENSURE(d_proj, (size_t)N_TOK * D * sizeof(float));
    ENSURE(d_gate, (size_t)N_TOK * FFN * sizeof(float));
    #undef ENSURE

    (void)HEADS;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int L = 0; L < SB_VITH_N_BLK; L++) {
        sb_block_vith *b = &ctx->vith_blk[L];

        /* LN1: d_ln <- LN(d_tok). */
        {
            struct __attribute__((packed)) {
                void *dst; const void *src, *w, *bb; int dim; float eps;
            } p = { ctx->d_ln, ctx->d_tok, b->norm1_w, b->norm1_b,
                    D, SB_VITH_LN_EPS };
            if (sb_launch(ctx->fn_ln, N_TOK, 1, 1, 256, 1, 1,
                          256 * 4, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* QKV: d_qkv <- gemm(qkv_w, d_ln) + qkv_b.  Out cols = 3*D. */
        {
            struct __attribute__((packed)) {
                void *Y; const void *W, *X, *bias; int n_out, n_in, n_tok;
            } p = { ctx->d_qkv, b->qkv_w, ctx->d_ln, b->qkv_b,
                    3 * D, D, N_TOK };
            if (sb_launch(ctx->fn_gemm_tiled,
                          (unsigned)((3 * D + 63) / 64),
                          (unsigned)((N_TOK + 15) / 16), 1, 16, 16, 1,
                          0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* qkv_split_f32: split d_qkv (N, 3D) into Q=d_kt, K=d_vt, V=d_proj
         * each (N, D) row-major. */
        {
            struct __attribute__((packed)) {
                void *Q, *K, *V; const void *qkv; int n, dim;
            } p = { ctx->d_kt, ctx->d_vt, ctx->d_proj, ctx->d_qkv, N_TOK, D };
            unsigned total = (unsigned)(N_TOK * D);
            if (sb_launch(ctx->fn_qkv_split, (total + 255) / 256, 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* sdpa_f32: d_attn (N, D) <- softmax(Q K^T * scale) V.
         *   q=d_kt, k=d_vt, v=d_proj — all (N, H*D_h) row-major.
         * Grid: (N_q, H)   Block: (256,)   Shmem: (256 + N_k) * 4. */
        {
            struct __attribute__((packed)) {
                void *out; const void *q, *k, *v;
                int n_q, n_k, h, d_h; float sc;
            } p = { ctx->d_attn, ctx->d_kt, ctx->d_vt, ctx->d_proj,
                    N_TOK, N_TOK, HEADS, HEAD_DIM, scale };
            unsigned shmem = (unsigned)((256 + N_TOK) * sizeof(float));
            if (sb_launch(ctx->fn_sdpa, (unsigned)N_TOK, (unsigned)HEADS, 1,
                          256, 1, 1, shmem, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* proj: d_proj <- gemm(proj_w, d_attn) + proj_b. */
        {
            struct __attribute__((packed)) {
                void *Y; const void *W, *X, *bias; int n_out, n_in, n_tok;
            } p = { ctx->d_proj, b->proj_w, ctx->d_attn, b->proj_b,
                    D, D, N_TOK };
            if (sb_launch(ctx->fn_gemm_tiled,
                          (unsigned)((D + 63) / 64),
                          (unsigned)((N_TOK + 15) / 16), 1, 16, 16, 1,
                          0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* residual: d_tok += d_proj (no LayerScale for ViT-H). */
        {
            struct __attribute__((packed)) {
                void *a; const void *b; int n;
            } p = { ctx->d_tok, ctx->d_proj, N_TOK * D };
            unsigned total = (unsigned)(N_TOK * D);
            if (sb_launch(ctx->fn_add_inplace, (total + 255) / 256, 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* LN2: d_ln <- LN(d_tok). */
        {
            struct __attribute__((packed)) {
                void *dst; const void *src, *w, *bb; int dim; float eps;
            } p = { ctx->d_ln, ctx->d_tok, b->norm2_w, b->norm2_b,
                    D, SB_VITH_LN_EPS };
            if (sb_launch(ctx->fn_ln, N_TOK, 1, 1, 256, 1, 1,
                          256 * 4, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* fc1: d_gate <- gemm(fc1_w, d_ln) + fc1_b.  (N, FFN) */
        {
            struct __attribute__((packed)) {
                void *Y; const void *W, *X, *bias; int n_out, n_in, n_tok;
            } p = { ctx->d_gate, b->fc1_w, ctx->d_ln, b->fc1_b,
                    FFN, D, N_TOK };
            if (sb_launch(ctx->fn_gemm_tiled,
                          (unsigned)((FFN + 63) / 64),
                          (unsigned)((N_TOK + 15) / 16), 1, 16, 16, 1,
                          0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* GELU in-place. */
        {
            struct __attribute__((packed)) {
                void *x; int n;
            } p = { ctx->d_gate, N_TOK * FFN };
            unsigned total = (unsigned)(N_TOK * FFN);
            if (sb_launch(ctx->fn_gelu_inplace, (total + 255) / 256, 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* fc2: d_proj <- gemm(fc2_w, d_gate) + fc2_b.  (N, D) */
        {
            struct __attribute__((packed)) {
                void *Y; const void *W, *X, *bias; int n_out, n_in, n_tok;
            } p = { ctx->d_proj, b->fc2_w, ctx->d_gate, b->fc2_b,
                    D, FFN, N_TOK };
            if (sb_launch(ctx->fn_gemm_tiled,
                          (unsigned)((D + 63) / 64),
                          (unsigned)((N_TOK + 15) / 16), 1, 16, 16, 1,
                          0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* residual: d_tok += d_proj. */
        {
            struct __attribute__((packed)) {
                void *a; const void *b; int n;
            } p = { ctx->d_tok, ctx->d_proj, N_TOK * D };
            unsigned total = (unsigned)(N_TOK * D);
            if (sb_launch(ctx->fn_add_inplace, (total + 255) / 256, 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
    }

    /* last_norm: d_proj <- LN(d_tok, last_norm_w, last_norm_b). */
    {
        struct __attribute__((packed)) {
            void *dst; const void *src, *w, *bb; int dim; float eps;
        } p = { ctx->d_proj, ctx->d_tok, ctx->w_norm_w, ctx->w_norm_b,
                D, SB_VITH_LN_EPS };
        if (sb_launch(ctx->fn_ln, N_TOK, 1, 1, 256, 1, 1,
                      256 * 4, &p, sizeof(p)) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
    }

    if (hipDeviceSynchronize() != hipSuccess) return CUDA_SAM3D_BODY_E_LOAD;

    /* Copy final encoder tokens to host as (N_TOK, D) row-major unless the
     * production GPU-MHR decoder can consume ctx->d_proj directly. */
    free(ctx->encoder_tokens.data);
    ctx->encoder_tokens.data = NULL;
    ctx->encoder_tokens_dev_valid = 1;
    if (!ctx->mhr_gpu_requested) {
        ctx->encoder_tokens.data =
            (float *)malloc((size_t)N_TOK * D * sizeof(float));
        if (!ctx->encoder_tokens.data) return CUDA_SAM3D_BODY_E_LOAD;
        if (hipMemcpy(ctx->encoder_tokens.data, ctx->d_proj,
                      (size_t)N_TOK * D * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess)
            return CUDA_SAM3D_BODY_E_LOAD;
    }
    ctx->encoder_tokens.n = N_TOK;
    ctx->encoder_tokens.c = D;

    if (ctx->verbose >= 1)
        fprintf(stderr, "sam3d_body_vith: encoder forward done "
                        "(%d blocks, N_TOK=%d, D=%d)\n",
                SB_VITH_N_BLK, N_TOK, D);

    return CUDA_SAM3D_BODY_E_OK;
}

int cuda_sam3d_body_run_encoder(cuda_sam3d_body_ctx *ctx)
{
    if (!ctx) return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->weights_ready) return CUDA_SAM3D_BODY_E_LOAD;
    if (!ctx->image_rgb && !ctx->has_norm_input)
        return CUDA_SAM3D_BODY_E_INVAL;

    const double t_encoder0 = sb_time_ms();

    if (ctx->cfg.backbone == CUDA_SAM3D_BODY_BACKBONE_VITH) {
        /* Host-side preprocess for vit_hmr_512_384: TopdownAffine to a
         * 512×512 canvas + ImageNet norm + W-axis crop [:, :, :, 64:-64]
         * → (3, 512, 384). Mirrors the CPU runner. Skipped if the verify
         * path already populated d_img_f32 via debug_set_normalized_input. */
        if (!ctx->has_norm_input) {
            const int S = SB_VITH_IMG_H;        /* 512 */
            const int W_crop = SB_VITH_IMG_W;   /* 384 */
            float bbox_xyxy[4];
            if (ctx->has_bbox) {
                memcpy(bbox_xyxy, ctx->bbox, sizeof(bbox_xyxy));
            } else {
                bbox_xyxy[0] = 0; bbox_xyxy[1] = 0;
                bbox_xyxy[2] = (float)ctx->img_w; bbox_xyxy[3] = (float)ctx->img_h;
            }
            sam3d_body_compute_bbox_affine(bbox_xyxy, /*padding=*/1.25f,
                                           /*aspect_ratio_pre=*/0.75f, S, S,
                                           ctx->self_center, ctx->self_scale,
                                           ctx->self_warp);
            sam3d_body_default_cam_int(ctx->img_w, ctx->img_h, ctx->self_cam_int);
            if (ctx->focal_hint > 0) {
                ctx->self_cam_int[0] = ctx->focal_hint;
                ctx->self_cam_int[4] = ctx->focal_hint;
            }
            ctx->self_pp_set = 1;

            float *chw_full = (float *)malloc((size_t)3 * S * S * sizeof(float));
            if (!chw_full) return CUDA_SAM3D_BODY_E_LOAD;
            if (sam3d_body_preprocess_image(ctx->image_rgb, ctx->img_w, ctx->img_h,
                                            ctx->self_warp, S, S, chw_full) != 0) {
                free(chw_full);
                return CUDA_SAM3D_BODY_E_LOAD;
            }
            float *chw_crop = (float *)malloc((size_t)3 * S * W_crop * sizeof(float));
            if (!chw_crop) { free(chw_full); return CUDA_SAM3D_BODY_E_LOAD; }
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < S; y++) {
                    const float *src = chw_full + ((size_t)c * S + y) * S + 64;
                    float       *dst = chw_crop + ((size_t)c * S + y) * W_crop;
                    memcpy(dst, src, (size_t)W_crop * sizeof(float));
                }
            }
            free(chw_full);

            size_t bytes = (size_t)3 * S * W_crop * sizeof(float);
            if (!ctx->d_img_f32) {
                if (hipMalloc(&ctx->d_img_f32, bytes) != hipSuccess) {
                    free(chw_crop);
                    return CUDA_SAM3D_BODY_E_LOAD;
                }
            }
            if (hipMemcpy(ctx->d_img_f32, chw_crop, bytes,
                          hipMemcpyHostToDevice) != hipSuccess) {
                free(chw_crop);
                return CUDA_SAM3D_BODY_E_LOAD;
            }
            free(chw_crop);
            ctx->has_norm_input = 1;
        }
        return cuda_sam3d_body_run_encoder_vith(ctx);
    }

    const int IMG_H = ctx->cfg.image_height;
    const int IMG_W = ctx->cfg.image_width;
    const int GH  = IMG_H / SB_PATCH;
    const int GW  = IMG_W / SB_PATCH;
    const int N_PATCH = GH * GW;
    const int N_TOK   = 1 + SB_N_STORAGE + N_PATCH;
    double t_preprocess_ms = 0.0;
    double t_embed_ms = 0.0;
    double t_blocks_ms = 0.0;
    double t_final_norm_ms = 0.0;
    double t_readback_ms = 0.0;

    if (!ctx->d_tok) {
        if (hipMalloc(&ctx->d_tok,
                      (size_t)N_TOK * SB_DIM * sizeof(float)) != hipSuccess)
            return CUDA_SAM3D_BODY_E_LOAD;
    }

    if (!ctx->has_norm_input) {
        /* Match the CPU runner's preprocess: TopdownAffine (bbox + 0.75
         * aspect ratio + 1.25 padding) → cv2.warpAffine → ImageNet norm
         * → HWC→CHW. We do this on the host (sam3d_body_preprocess_image)
         * and upload the (3, IMG_H, IMG_W) f32 result; on-device naive
         * bilinear resize would feed DINOv3 a stretched image when the
         * input aspect ratio differs from the crop aspect. */
        const double t0 = sb_time_ms();
        float bbox_xyxy[4];
        if (ctx->has_bbox) {
            memcpy(bbox_xyxy, ctx->bbox, sizeof(bbox_xyxy));
        } else {
            bbox_xyxy[0] = 0; bbox_xyxy[1] = 0;
            bbox_xyxy[2] = (float)ctx->img_w; bbox_xyxy[3] = (float)ctx->img_h;
        }
        float aspect_ratio_pre = 0.75f;
        if (IMG_H != IMG_W)
            aspect_ratio_pre = (float)IMG_W / (float)IMG_H;
        sam3d_body_compute_bbox_affine(bbox_xyxy, /*padding=*/1.25f,
                                       aspect_ratio_pre, IMG_W, IMG_H,
                                       ctx->self_center, ctx->self_scale,
                                       ctx->self_warp);
        sam3d_body_default_cam_int(ctx->img_w, ctx->img_h, ctx->self_cam_int);
        if (ctx->focal_hint > 0) {
            ctx->self_cam_int[0] = ctx->focal_hint;
            ctx->self_cam_int[4] = ctx->focal_hint;
        }
        ctx->self_pp_set = 1;

        float *chw = (float *)malloc((size_t)3 * IMG_H * IMG_W * sizeof(float));
        if (!chw) return CUDA_SAM3D_BODY_E_LOAD;
        if (sam3d_body_preprocess_image(ctx->image_rgb, ctx->img_w, ctx->img_h,
                                        ctx->self_warp, IMG_W, IMG_H, chw) != 0) {
            free(chw);
            return CUDA_SAM3D_BODY_E_LOAD;
        }

        if (!ctx->d_img_f32) {
            if (hipMalloc(&ctx->d_img_f32,
                          (size_t)3 * IMG_H * IMG_W * sizeof(float)) != hipSuccess) {
                free(chw);
                return CUDA_SAM3D_BODY_E_LOAD;
            }
        }
        if (hipMemcpy(ctx->d_img_f32, chw,
                      (size_t)3 * IMG_H * IMG_W * sizeof(float),
                      hipMemcpyHostToDevice) != hipSuccess) {
            free(chw);
            return CUDA_SAM3D_BODY_E_LOAD;
        }
        free(chw);
        t_preprocess_ms = sb_time_ms() - t0;
    }

    const double t_embed0 = sb_time_ms();

    /* prepend CLS + 4 storage tokens → rows 0..4 */
    {
        struct __attribute__((packed)) {
            void *out; const void *cls; const void *storage;
            int n_storage, dim;
        } p = { ctx->d_tok, ctx->w_cls, ctx->w_storage, SB_N_STORAGE, SB_DIM };
        unsigned total = (unsigned)((1 + SB_N_STORAGE) * SB_DIM);
        if (sb_launch(ctx->fn_prepend, (total + 255) / 256, 1, 1, 256, 1, 1,
                      0, &p, sizeof(p)) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
    }

    /* patch embed → rows 5..5+n_patch */
    if (sb_precision_is_fp32(ctx) && ctx->use_cublas &&
        getenv("SAM3D_BODY_CUBLAS_PATCH")) {
        const int K_PATCH = 3 * SB_PATCH * SB_PATCH;
        if (!ctx->d_patch_cols) {
            if (hipMalloc(&ctx->d_patch_cols,
                          (size_t)N_PATCH * K_PATCH * sizeof(float)) != hipSuccess)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
        {
            struct __attribute__((packed)) {
                void *cols; const void *img; int gw, ps, img_h, img_w;
            } p = { ctx->d_patch_cols, ctx->d_img_f32,
                    GW, SB_PATCH, IMG_H, IMG_W };
            if (sb_launch(ctx->fn_patch_im2col,
                          (unsigned)((K_PATCH + 255) / 256),
                          (unsigned)N_PATCH, 1, 256, 1, 1,
                          0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
        void *patch_out = (char *)ctx->d_tok +
            (size_t)(1 + SB_N_STORAGE) * SB_DIM * sizeof(float);
        if (cublasew_gemm_f32_pedantic_rowmajor_nt(
                ctx->cublas, (CUdeviceptr)(uintptr_t)patch_out,
                (CUdeviceptr)(uintptr_t)ctx->w_patch_w,
                (CUdeviceptr)(uintptr_t)ctx->d_patch_cols,
                N_PATCH, SB_DIM, K_PATCH) != 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        {
            struct __attribute__((packed)) {
                void *x; const void *b; int N, D;
            } p = { patch_out, ctx->w_patch_b, N_PATCH, SB_DIM };
            int total = N_PATCH * SB_DIM;
            if (sb_launch(ctx->fn_add_bias_rows,
                          (unsigned)((total + 255) / 256), 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
    } else {
        struct __attribute__((packed)) {
            void *out; const void *img; const void *w; const void *bias;
            int gw, dim, ps, img_h, img_w, base_tok;
        } p = { ctx->d_tok, ctx->d_img_f32, ctx->w_patch_w, ctx->w_patch_b,
                GW, SB_DIM, SB_PATCH, IMG_H, IMG_W, 1 + SB_N_STORAGE };
        if (sb_launch(ctx->fn_patch, (unsigned)N_PATCH, 1, 1, 256, 1, 1,
                      0, &p, sizeof(p)) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
    }
    if (sb_bf16_round(ctx, ctx->d_tok, N_TOK * SB_DIM) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (hipDeviceSynchronize() != hipSuccess) return CUDA_SAM3D_BODY_E_LOAD;
    t_embed_ms = sb_time_ms() - t_embed0;
    const char *debug_dump_dir = getenv("SAM3D_BODY_DINOV3_DUMP_BLOCKS_DIR");
    sb_debug_dump_device_f32(ctx, debug_dump_dir, "cuda_dinov3_preblocks",
                             -1, ctx->d_tok, (size_t)N_TOK * SB_DIM);

    /* Allocate per-block runtime scratch on first use. */
    #define ENSURE(field, bytes) do { \
        if (!ctx->field) { \
            if (hipMalloc(&ctx->field, (bytes)) != hipSuccess) \
                return CUDA_SAM3D_BODY_E_LOAD; \
        } \
    } while (0)
    ENSURE(d_ln,   (size_t)N_TOK * SB_DIM * sizeof(float));
    ENSURE(d_qkv,  (size_t)N_TOK * 3 * SB_DIM * sizeof(float));
    if (sb_precision_is_fp32(ctx) && ctx->use_cublas &&
        getenv("SAM3D_BODY_CUBLAS_ATTN")) {
        ENSURE(d_qt, (size_t)SB_HEADS * N_TOK * SB_HEAD_DIM * sizeof(float));
        ENSURE(d_attn_scores,
               (size_t)SB_HEADS * N_TOK * N_TOK * sizeof(float));
    }
    ENSURE(d_kt,   (size_t)SB_HEADS * N_TOK * SB_HEAD_DIM * sizeof(float));
    ENSURE(d_vt,   (size_t)SB_HEADS * N_TOK * SB_HEAD_DIM * sizeof(float));
    ENSURE(d_attn, (size_t)N_TOK * SB_DIM * sizeof(float));
    ENSURE(d_proj, (size_t)N_TOK * SB_DIM * sizeof(float));
    ENSURE(d_gate, (size_t)N_TOK * SB_FFN * sizeof(float));
    ENSURE(d_up,   (size_t)N_TOK * SB_FFN * sizeof(float));
    #undef ENSURE

    /* ─── 32 transformer blocks ─── */
    const double t_blocks0 = sb_time_ms();
    const float scale = 1.0f / sqrtf((float)SB_HEAD_DIM);
    const int patch_start = 1 + SB_N_STORAGE;
    const int use_welford_ln =
        sb_precision_is_fp32(ctx) && getenv("SAM3D_BODY_WELFORD_LN");
    const int use_welford_warp_ln =
        sb_precision_is_fp32(ctx) && getenv("SAM3D_BODY_WELFORD_WARP_LN");
    const int use_sqrtdiv_ln =
        sb_precision_is_fp32(ctx) && getenv("SAM3D_BODY_SQRTDIV_LN");
    const int use_torchvec_ln =
        sb_precision_is_fp32(ctx) && getenv("SAM3D_BODY_TORCHVEC_LN");
    int torchvec_ln_y = 4;
    if (use_torchvec_ln) {
        const char *ln_y_s = getenv("SAM3D_BODY_TORCHVEC_LN_Y");
        if (ln_y_s) {
            int v = atoi(ln_y_s);
            if (v == 1 || v == 2 || v == 4 || v == 8)
                torchvec_ln_y = v;
        }
    }
    hipFunction_t fn_dinov3_ln = use_torchvec_ln ? ctx->fn_ln_torchvec :
        (use_welford_warp_ln ? ctx->fn_ln_welford_warp :
        (use_welford_ln ? ctx->fn_ln_welford :
            (use_sqrtdiv_ln ? ctx->fn_ln_sqrtdiv : ctx->fn_ln)));
    const unsigned dinov3_ln_shmem = use_welford_warp_ln ?
        (unsigned)(32 * sizeof(int) + 64 * sizeof(float)) :
        (use_torchvec_ln ? (unsigned)(3 * torchvec_ln_y * sizeof(float)) :
        (use_welford_ln ?
            (unsigned)(256 * (2 * sizeof(float) + sizeof(int))) :
            (unsigned)(256 * sizeof(float))));

    for (int L = 0; L < SB_N_BLK; L++) {
        sb_block *b = &ctx->blk[L];

        /* LN1: d_ln <- LN(d_tok, norm1_w, norm1_b) */
        {
            struct __attribute__((packed)) {
                void *dst; const void *src, *w, *bb; int dim; float eps;
            } p = { ctx->d_ln, ctx->d_tok, b->norm1_w, b->norm1_b,
                    SB_DIM, SB_LN_EPS };
            if (sb_launch(fn_dinov3_ln, N_TOK, 1, 1,
                          use_torchvec_ln ? 32u : 256u,
                          use_torchvec_ln ? (unsigned)torchvec_ln_y : 1u, 1,
                          dinov3_ln_shmem, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
        if (sb_bf16_round(ctx, ctx->d_ln, N_TOK * SB_DIM) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0)
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_norm1", -1,
                                     ctx->d_ln, (size_t)N_TOK * SB_DIM);

        /* QKV: d_qkv <- gemm(qkv_w, d_ln) + qkv_b. Out cols = 3*D. */
        if (sb_gemm_encoder(ctx, ctx->d_qkv, b->qkv_w, ctx->d_ln, b->qkv_b,
                            3 * SB_DIM, SB_DIM, N_TOK) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (sb_bf16_round(ctx, ctx->d_qkv, N_TOK * 3 * SB_DIM) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0)
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_qkv", -1,
                                     ctx->d_qkv, (size_t)N_TOK * 3 * SB_DIM);

        /* RoPE on Q,K (patch tokens only). */
        {
            struct __attribute__((packed)) {
                void *qkv; const void *cos_t, *sin_t;
                int n_patch, patch_start, heads, head_dim, D;
            } p = { ctx->d_qkv, ctx->w_rope_cos, ctx->w_rope_sin,
                    N_PATCH, patch_start, SB_HEADS, SB_HEAD_DIM, SB_DIM };
            if (sb_launch(ctx->fn_rope_qk, (unsigned)N_PATCH, 1, 1,
                          SB_HEADS, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
        if (sb_bf16_round(ctx, ctx->d_qkv, N_TOK * 3 * SB_DIM) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0)
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_qkv_rope", -1,
                                     ctx->d_qkv, (size_t)N_TOK * 3 * SB_DIM);

        const int use_cublas_attn =
            sb_precision_is_fp32(ctx) && ctx->use_cublas &&
            getenv("SAM3D_BODY_CUBLAS_ATTN");
        const int torch_sdpa_style =
            use_cublas_attn && getenv("SAM3D_BODY_TORCH_SDPA_STYLE");

        /* Deinterleave K/V, and Q as well for the cuBLAS strict path. */
        if (use_cublas_attn) {
            struct __attribute__((packed)) {
                void *Q, *K, *V; const void *qkv;
                int n_tok, dim, heads, hd; float qk_scale;
            } p = { ctx->d_qt, ctx->d_kt, ctx->d_vt, ctx->d_qkv,
                    N_TOK, SB_DIM, SB_HEADS, SB_HEAD_DIM,
                    torch_sdpa_style ? sqrtf(scale) : 1.0f };
            unsigned total = (unsigned)(N_TOK * SB_DIM);
            if (sb_launch(ctx->fn_qkv_tx, (total + 255) / 256, 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        } else {
            struct __attribute__((packed)) {
                void *K, *V; const void *qkv;
                int n_tok, dim, heads, hd;
            } p = { ctx->d_kt, ctx->d_vt, ctx->d_qkv,
                    N_TOK, SB_DIM, SB_HEADS, SB_HEAD_DIM };
            unsigned total = (unsigned)(N_TOK * SB_DIM);
            if (sb_launch(ctx->fn_kv_tx, (total + 255) / 256, 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }

        /* FlashAttention: d_attn <- softmax(Q K^T * scale) V. */
        if (use_cublas_attn) {
            const int cublas_attn_fast =
                getenv("SAM3D_BODY_CUBLAS_ATTN_FAST") != NULL;
            const int qk_strided_qkv =
                getenv("SAM3D_BODY_ATTN_QK_STRIDED_QKV") != NULL;
            for (int h = 0; h < SB_HEADS; h++) {
                CUdeviceptr Sh = (CUdeviceptr)(uintptr_t)
                    ((char *)ctx->d_attn_scores + (size_t)h * N_TOK * N_TOK * sizeof(float));
                int rc_gemm;
                if (qk_strided_qkv) {
                    CUdeviceptr Qh = (CUdeviceptr)(uintptr_t)
                        ((char *)ctx->d_qkv + (size_t)h * SB_HEAD_DIM * sizeof(float));
                    CUdeviceptr Kh = (CUdeviceptr)(uintptr_t)
                        ((char *)ctx->d_qkv + ((size_t)SB_DIM + (size_t)h * SB_HEAD_DIM) * sizeof(float));
                    rc_gemm = cublasew_gemm_f32_pedantic_rowmajor_nt_strided(
                        ctx->cublas, Sh, Kh, 3 * SB_DIM, Qh, 3 * SB_DIM,
                        N_TOK, N_TOK, SB_HEAD_DIM);
                } else {
                    CUdeviceptr Qh = (CUdeviceptr)(uintptr_t)
                        ((char *)ctx->d_qt + (size_t)h * N_TOK * SB_HEAD_DIM * sizeof(float));
                    CUdeviceptr Kh = (CUdeviceptr)(uintptr_t)
                        ((char *)ctx->d_kt + (size_t)h * N_TOK * SB_HEAD_DIM * sizeof(float));
                    rc_gemm = cublas_attn_fast ?
                        cublasew_gemm_f32_rowmajor_nt(
                            ctx->cublas, Sh, Kh, Qh, N_TOK, N_TOK, SB_HEAD_DIM) :
                        cublasew_gemm_f32_pedantic_rowmajor_nt(
                            ctx->cublas, Sh, Kh, Qh, N_TOK, N_TOK, SB_HEAD_DIM);
                }
                if (rc_gemm != 0)
                    return CUDA_SAM3D_BODY_E_LOAD;
            }
            if (torch_sdpa_style) {
                struct __attribute__((packed)) {
                    void *x; int rows, cols;
                } p = { ctx->d_attn_scores, SB_HEADS * N_TOK, N_TOK };
                if (sb_launch(ctx->fn_softmax_warp,
                              (unsigned)((SB_HEADS * N_TOK + 3) / 4), 1, 1,
                              32, 4, 1, 0,
                              &p, sizeof(p)) < 0)
                    return CUDA_SAM3D_BODY_E_LOAD;
            } else if (getenv("SAM3D_BODY_WARP_SOFTMAX")) {
                struct __attribute__((packed)) {
                    void *x; int n; float sc;
                } ps = { ctx->d_attn_scores,
                         SB_HEADS * N_TOK * N_TOK, scale };
                if (sb_launch(ctx->fn_scale_rows,
                              (unsigned)((SB_HEADS * N_TOK * N_TOK + 255) / 256),
                              1, 1, 256, 1, 1, 0, &ps, sizeof(ps)) < 0)
                    return CUDA_SAM3D_BODY_E_LOAD;
                struct __attribute__((packed)) {
                    void *x; int rows, cols;
                } p = { ctx->d_attn_scores, SB_HEADS * N_TOK, N_TOK };
                if (sb_launch(ctx->fn_softmax_warp,
                              (unsigned)((SB_HEADS * N_TOK + 3) / 4), 1, 1,
                              32, 4, 1, 0,
                              &p, sizeof(p)) < 0)
                    return CUDA_SAM3D_BODY_E_LOAD;
            } else {
                struct __attribute__((packed)) {
                    void *x; int rows, cols; float sc;
                } p = { ctx->d_attn_scores, SB_HEADS * N_TOK, N_TOK, scale };
                if (sb_launch(ctx->fn_scale_softmax,
                              (unsigned)(SB_HEADS * N_TOK), 1, 1,
                              256, 1, 1, 256 * sizeof(float),
                              &p, sizeof(p)) < 0)
                    return CUDA_SAM3D_BODY_E_LOAD;
            }
            if (L == 0)
                sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                         "cuda_dinov3_b00_attn_prob", -1,
                                         ctx->d_attn_scores,
                                         (size_t)SB_HEADS * N_TOK * N_TOK);
            int value_serial_this_block =
                getenv("SAM3D_BODY_ATTN_VALUE_SERIAL") != NULL;
            const char *serial_until =
                getenv("SAM3D_BODY_ATTN_VALUE_SERIAL_UNTIL");
            const char *serial_from =
                getenv("SAM3D_BODY_ATTN_VALUE_SERIAL_FROM");
            const char *serial_skip =
                getenv("SAM3D_BODY_ATTN_VALUE_SERIAL_SKIP");
            if (serial_until)
                value_serial_this_block |= L < atoi(serial_until);
            if (serial_from)
                value_serial_this_block |= L >= atoi(serial_from);
            if (serial_skip) {
                const char *p_skip = serial_skip;
                while (*p_skip) {
                    char *end = NULL;
                    long v = strtol(p_skip, &end, 10);
                    if (end == p_skip) {
                        p_skip++;
                        continue;
                    }
                    if (v == L) {
                        value_serial_this_block = 0;
                        break;
                    }
                    p_skip = end;
                }
            }
            if (value_serial_this_block) {
                struct __attribute__((packed)) {
                    void *out; const void *prob, *V;
                    int n_tok, dim, heads, hd;
                } p = { ctx->d_attn, ctx->d_attn_scores, ctx->d_vt,
                        N_TOK, SB_DIM, SB_HEADS, SB_HEAD_DIM };
                unsigned total = (unsigned)(N_TOK * SB_DIM);
                if (sb_launch(ctx->fn_attn_prob_v_serial,
                              (total + 255) / 256, 1, 1,
                              256, 1, 1, 0, &p, sizeof(p)) < 0)
                    return CUDA_SAM3D_BODY_E_LOAD;
            } else if (getenv("SAM3D_BODY_ATTN_VALUE_STRIDED_QKV")) {
                for (int h = 0; h < SB_HEADS; h++) {
                    CUdeviceptr Sh = (CUdeviceptr)(uintptr_t)
                        ((char *)ctx->d_attn_scores + (size_t)h * N_TOK * N_TOK * sizeof(float));
                    CUdeviceptr Vh = (CUdeviceptr)(uintptr_t)
                        ((char *)ctx->d_qkv + ((size_t)2 * SB_DIM + (size_t)h * SB_HEAD_DIM) * sizeof(float));
                    CUdeviceptr Oh = (CUdeviceptr)(uintptr_t)
                        ((char *)ctx->d_attn + (size_t)h * SB_HEAD_DIM * sizeof(float));
                    if (cublasew_gemm_f32_pedantic_rowmajor_nn_stridedB(
                            ctx->cublas, Oh, SB_DIM, Sh, Vh, 3 * SB_DIM,
                            N_TOK, SB_HEAD_DIM, N_TOK) != 0)
                        return CUDA_SAM3D_BODY_E_LOAD;
                }
            } else {
            for (int h = 0; h < SB_HEADS; h++) {
                CUdeviceptr Sh = (CUdeviceptr)(uintptr_t)
                    ((char *)ctx->d_attn_scores + (size_t)h * N_TOK * N_TOK * sizeof(float));
                CUdeviceptr Vh = (CUdeviceptr)(uintptr_t)
                    ((char *)ctx->d_vt + (size_t)h * N_TOK * SB_HEAD_DIM * sizeof(float));
                CUdeviceptr Oh = (CUdeviceptr)(uintptr_t)
                    ((char *)ctx->d_qt + (size_t)h * N_TOK * SB_HEAD_DIM * sizeof(float));
                int rc_gemm = cublas_attn_fast ?
                    cublasew_gemm_f32_rowmajor_nn(
                        ctx->cublas, Oh, SB_HEAD_DIM, Sh, Vh, N_TOK,
                        SB_HEAD_DIM, N_TOK) :
                    cublasew_gemm_f32_pedantic_rowmajor_nn(
                        ctx->cublas, Oh, SB_HEAD_DIM, Sh, Vh, N_TOK,
                        SB_HEAD_DIM, N_TOK);
                if (rc_gemm != 0)
                    return CUDA_SAM3D_BODY_E_LOAD;
            }
            {
                struct __attribute__((packed)) {
                    void *out; const void *heads_buf;
                    int n_tok, dim, heads, hd;
                } p = { ctx->d_attn, ctx->d_qt,
                        N_TOK, SB_DIM, SB_HEADS, SB_HEAD_DIM };
                unsigned total = (unsigned)(N_TOK * SB_DIM);
                if (sb_launch(ctx->fn_heads_merge,
                              (total + 255) / 256, 1, 1,
                              256, 1, 1, 0, &p, sizeof(p)) < 0)
                    return CUDA_SAM3D_BODY_E_LOAD;
            }
            }
        } else if (sb_precision_is_fp32(ctx) && getenv("SAM3D_BODY_PRECISE_ATTN")) {
            struct __attribute__((packed)) {
                void *out; const void *qkv, *K, *V;
                int n_tok, dim, heads, hd; float sc;
            } p = { ctx->d_attn, ctx->d_qkv, ctx->d_kt, ctx->d_vt,
                    N_TOK, SB_DIM, SB_HEADS, SB_HEAD_DIM, scale };
            unsigned shmem = (256u + (unsigned)N_TOK) * sizeof(float);
            if (sb_launch(ctx->fn_sdpa_qkv, (unsigned)N_TOK, SB_HEADS, 1,
                          256, 1, 1, shmem, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        } else {
            struct __attribute__((packed)) {
                void *out; const void *qkv, *K, *V;
                int n_tok, dim, heads, hd; float sc;
            } p = { ctx->d_attn, ctx->d_qkv, ctx->d_kt, ctx->d_vt,
                    N_TOK, SB_DIM, SB_HEADS, SB_HEAD_DIM, scale };
            unsigned gy = (unsigned)((N_TOK + 63) / 64);
            unsigned shmem = 2u * 16u * 64u * sizeof(float);
            if (sb_launch(ctx->fn_fa, SB_HEADS, gy, 1, 64, 1, 1,
                          shmem, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
        if (sb_bf16_round(ctx, ctx->d_attn, N_TOK * SB_DIM) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0)
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_attn_preproj", -1,
                                     ctx->d_attn, (size_t)N_TOK * SB_DIM);

        /* Output projection: d_proj <- gemm(proj_w, d_attn) + proj_b. */
        if (sb_gemm_encoder(ctx, ctx->d_proj, b->proj_w, ctx->d_attn,
                            b->proj_b, SB_DIM, SB_DIM, N_TOK) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (sb_bf16_round(ctx, ctx->d_proj, N_TOK * SB_DIM) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0)
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_attn_proj", -1,
                                     ctx->d_proj, (size_t)N_TOK * SB_DIM);

        /* LayerScale1 + residual: d_tok += d_proj * ls1. */
        {
            struct __attribute__((packed)) {
                void *hidden; const void *proj, *gamma; int n_tok, dim;
            } p = { ctx->d_tok, ctx->d_proj, b->ls1, N_TOK, SB_DIM };
            unsigned total = (unsigned)(N_TOK * SB_DIM);
            if (sb_launch(ctx->fn_layerscale_add, (total + 255) / 256, 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
        if (sb_bf16_round(ctx, ctx->d_tok, N_TOK * SB_DIM) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0)
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_x_attn", -1,
                                     ctx->d_tok, (size_t)N_TOK * SB_DIM);

        /* LN2: d_ln <- LN(d_tok, norm2_w, norm2_b) */
        {
            struct __attribute__((packed)) {
                void *dst; const void *src, *w, *bb; int dim; float eps;
            } p = { ctx->d_ln, ctx->d_tok, b->norm2_w, b->norm2_b,
                    SB_DIM, SB_LN_EPS };
            if (sb_launch(fn_dinov3_ln, N_TOK, 1, 1,
                          use_torchvec_ln ? 32u : 256u,
                          use_torchvec_ln ? (unsigned)torchvec_ln_y : 1u, 1,
                          dinov3_ln_shmem, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
        if (sb_bf16_round(ctx, ctx->d_ln, N_TOK * SB_DIM) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0)
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_norm2", -1,
                                     ctx->d_ln, (size_t)N_TOK * SB_DIM);

        /* SwiGLU: d_gate <- gemm(w1, d_ln) + b1; d_up <- gemm(w2, d_ln) + b2. */
        if (sb_gemm_encoder(ctx, ctx->d_gate, b->w1_w, ctx->d_ln, b->w1_b,
                            SB_FFN, SB_DIM, N_TOK) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (sb_gemm_encoder(ctx, ctx->d_up, b->w2_w, ctx->d_ln, b->w2_b,
                            SB_FFN, SB_DIM, N_TOK) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (sb_bf16_round(ctx, ctx->d_gate, N_TOK * SB_FFN) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (sb_bf16_round(ctx, ctx->d_up, N_TOK * SB_FFN) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0) {
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_mlp_w1", -1,
                                     ctx->d_gate, (size_t)N_TOK * SB_FFN);
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_mlp_w2", -1,
                                     ctx->d_up, (size_t)N_TOK * SB_FFN);
        }

        /* d_gate := silu(d_gate) * d_up. */
        {
            struct __attribute__((packed)) {
                void *gate; const void *up; int n;
            } p = { ctx->d_gate, ctx->d_up, N_TOK * SB_FFN };
            unsigned total = (unsigned)(N_TOK * SB_FFN);
            if (sb_launch(ctx->fn_silu_mul, (total + 255) / 256, 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
        if (sb_bf16_round(ctx, ctx->d_gate, N_TOK * SB_FFN) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0)
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_mlp_act", -1,
                                     ctx->d_gate, (size_t)N_TOK * SB_FFN);

        /* w3: d_proj <- gemm(w3, d_gate) + b3.   (FFN→D)  reuse d_proj */
        if (sb_gemm_encoder(ctx, ctx->d_proj, b->w3_w, ctx->d_gate, b->w3_b,
                            SB_DIM, SB_FFN, N_TOK) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (sb_bf16_round(ctx, ctx->d_proj, N_TOK * SB_DIM) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
        if (L == 0)
            sb_debug_dump_device_f32(ctx, debug_dump_dir,
                                     "cuda_dinov3_b00_mlp_w3", -1,
                                     ctx->d_proj, (size_t)N_TOK * SB_DIM);

        /* LayerScale2 + residual: d_tok += d_proj * ls2. */
        {
            struct __attribute__((packed)) {
                void *hidden; const void *proj, *gamma; int n_tok, dim;
            } p = { ctx->d_tok, ctx->d_proj, b->ls2, N_TOK, SB_DIM };
            unsigned total = (unsigned)(N_TOK * SB_DIM);
            if (sb_launch(ctx->fn_layerscale_add, (total + 255) / 256, 1, 1,
                          256, 1, 1, 0, &p, sizeof(p)) < 0)
                return CUDA_SAM3D_BODY_E_LOAD;
        }
        if (sb_bf16_round(ctx, ctx->d_tok, N_TOK * SB_DIM) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;

        sb_debug_dump_device_f32(ctx, debug_dump_dir, "cuda_dinov3_block",
                                 L, ctx->d_tok, (size_t)N_TOK * SB_DIM);
    }
    if (hipDeviceSynchronize() != hipSuccess) return CUDA_SAM3D_BODY_E_LOAD;
    t_blocks_ms = sb_time_ms() - t_blocks0;

    /* Final LayerNorm in-place via d_proj scratch, then copy back to d_tok. */
    const double t_final0 = sb_time_ms();
    {
        struct __attribute__((packed)) {
            void *dst; const void *src, *w, *bb; int dim; float eps;
        } p = { ctx->d_proj, ctx->d_tok, ctx->w_norm_w, ctx->w_norm_b,
                SB_DIM, SB_LN_EPS };
        if (sb_launch(fn_dinov3_ln, N_TOK, 1, 1,
                      use_torchvec_ln ? 32u : 256u,
                      use_torchvec_ln ? (unsigned)torchvec_ln_y : 1u, 1,
                      dinov3_ln_shmem, &p, sizeof(p)) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
    }
    if (sb_bf16_round(ctx, ctx->d_proj, N_TOK * SB_DIM) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;

    if (hipDeviceSynchronize() != hipSuccess) return CUDA_SAM3D_BODY_E_LOAD;
    sb_debug_dump_device_f32(ctx, debug_dump_dir, "cuda_dinov3_final",
                             -1, ctx->d_proj, (size_t)N_TOK * SB_DIM);
    t_final_norm_ms = sb_time_ms() - t_final0;

    /* Copy final encoder tokens (post-LN) to host. */
    const double t_read0 = sb_time_ms();
    free(ctx->encoder_tokens.data);
    ctx->encoder_tokens.data = NULL;
    ctx->encoder_tokens_dev_valid = 1;
    if (!ctx->mhr_gpu_requested) {
        ctx->encoder_tokens.data =
            (float *)malloc((size_t)N_TOK * SB_DIM * sizeof(float));
        if (!ctx->encoder_tokens.data) return CUDA_SAM3D_BODY_E_LOAD;
        if (hipMemcpy(ctx->encoder_tokens.data, ctx->d_proj,
                      (size_t)N_TOK * SB_DIM * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess)
            return CUDA_SAM3D_BODY_E_LOAD;
    }
    ctx->encoder_tokens.n = N_TOK;
    ctx->encoder_tokens.c = SB_DIM;
    t_readback_ms = sb_time_ms() - t_read0;

    if (ctx->verbose >= 1) {
        fprintf(stderr, "sam3d_body: encoder forward done "
                        "(%d blocks, N_TOK=%d, D=%d)\n",
                SB_N_BLK, N_TOK, SB_DIM);
        fprintf(stderr,
                "[sam3d_body][timing] encoder total %.3f ms "
                "(preprocess+upload %.3f, prepend+patch %.3f, "
                "blocks %.3f, final_norm %.3f, readback %.3f)\n",
                sb_time_ms() - t_encoder0, t_preprocess_ms, t_embed_ms,
                t_blocks_ms, t_final_norm_ms, t_readback_ms);
    }

    return CUDA_SAM3D_BODY_E_OK;
}

/* Self-driven decoder forward. Mirrors cpu/sam3d_body/sam3d_body_runner.c
 * (self_drive_decoder_inputs + sam3d_body_decoder_forward_full) but the
 * heavy GEMMs (ray_cond_emb, build_tokens, the 6 decoder layers,
 * norm_final + heads, kp_token_update) run on the GPU via the
 * cuda_sam3d_body_debug_run_* entry points used by verify_decoder.c.
 *
 * By default MHR skinning + keypoints_from_mesh + camera_project still run
 * on CPU. With SAM3D_BODY_GPU_MHR=1, a lazy cached hybrid path keeps small
 * parameter/skeleton stages on CPU, while shape/face blends, the heavy
 * pose-correctives dense GEMV, rest-vertex combine, LBS, keypoint
 * regression, and projection run on cached GPU tensors. */
int cuda_sam3d_body_run_decoder(cuda_sam3d_body_ctx *ctx)
{
    if (!ctx) return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->encoder_tokens.data && !ctx->encoder_tokens_dev_valid) {
        fprintf(stderr, "[cuda_sam3d_body] run_decoder: encoder tokens "
                        "not populated — call run_encoder first\n");
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    if (!ctx->cpu_dec) return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->cpu_mhr) {
        fprintf(stderr, "[cuda_sam3d_body] run_decoder: MHR assets not "
                        "loaded — set cfg.mhr_assets_dir at create time\n");
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    const double t_decoder0 = sb_time_ms();
    double t_mhr_cache_ms = 0.0;
    if (ctx->mhr_gpu_requested && !ctx->mhr_gpu_enabled) {
        const double t_cache0 = sb_time_ms();
        int cache_rc = cuda_sam3d_body_ensure_mhr_gpu_cache(ctx);
        t_mhr_cache_ms = sb_time_ms() - t_cache0;
        if (cache_rc != CUDA_SAM3D_BODY_E_OK)
            return cache_rc;
    }
    double t_geom_ms = 0.0, t_encoder_permute_ms = 0.0, t_rays_ms = 0.0;
    double t_ray_cond_ms = 0.0, t_dense_pe_ms = 0.0, t_ctx_permute_ms = 0.0;
    double t_condition_ms = 0.0, t_build_tokens_ms = 0.0, t_workspace_ms = 0.0;
    double t_layer_ms[SB_DEC_LAYERS] = {0};
    double t_norm_heads_ms[SB_DEC_LAYERS + 1] = {0};
    double t_decode_pose_ms[SB_DEC_LAYERS + 1] = {0};
    double t_mhr_ms[SB_DEC_LAYERS + 1] = {0};
    double t_keypoints_ms[SB_DEC_LAYERS + 1] = {0};
    double t_project_ms[SB_DEC_LAYERS + 1] = {0};
    double t_kp_update_ms[SB_DEC_LAYERS] = {0};
    double t_output_ms = 0.0;

    const sam3d_body_decoder_model *m = ctx->cpu_dec;
    const int Dc  = m->kv_dim;          /* 1280 */
    const int D   = m->dim;             /* 1024 */
    const int K   = m->n_keypoints;     /* 70 */
    const int NL  = m->n_layers;        /* 6 */
    const int N_Q = 1 + 1 + 1 + m->n_hand_tokens + 2 * K;  /* 145 */
    const int V   = 18439;
    const int J   = 127;
    /* Encoder-grid + image-size geometry depends on backbone:
     *   dinov3-h+:  configurable patch-aligned HxW, n_prefix=5 (CLS + 4 reg)
     *   vit-h:      IMG_H=512, IMG_W=384, gh=32, gw=24, N_C=768, n_prefix=0 */
    const int is_vith = (ctx->cfg.backbone == CUDA_SAM3D_BODY_BACKBONE_VITH);
    const int IMG_H = is_vith ? SB_VITH_IMG_H : ctx->cfg.image_height;
    const int IMG_W = is_vith ? SB_VITH_IMG_W : ctx->cfg.image_width;
    const int gh    = is_vith ? SB_VITH_GRID_H : (ctx->cfg.image_height / SB_PATCH);
    const int gw    = is_vith ? SB_VITH_GRID_W : (ctx->cfg.image_width / SB_PATCH);
    const int N_C   = gh * gw;
    const int n_prefix = is_vith ? 0 : (1 + SB_N_STORAGE);

    if (ctx->encoder_tokens.n != n_prefix + N_C ||
        ctx->encoder_tokens.c != Dc) {
        fprintf(stderr, "[cuda_sam3d_body] encoder shape mismatch: "
                        "tokens=(%d,%d), expected (%d,%d)\n",
                ctx->encoder_tokens.n, ctx->encoder_tokens.c,
                n_prefix + N_C, Dc);
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    const int dev_decoder = ctx->mhr_gpu_enabled && ctx->encoder_tokens_dev_valid;

    /* TopdownAffine + cam_int are cached by run_encoder so the warp
     * matrix used to produce the DINOv3 input matches the rays_xyz /
     * cam_int seen by the decoder. Fall back to recomputing only if the
     * encoder was driven via the debug normalized-input path. */
    float bbox_xyxy[4], center[2], scale[2], warp[6], cam_int[9];
    const double t_geom0 = sb_time_ms();
    if (ctx->self_pp_set) {
        memcpy(center,  ctx->self_center,  sizeof(center));
        memcpy(scale,   ctx->self_scale,   sizeof(scale));
        memcpy(warp,    ctx->self_warp,    sizeof(warp));
        memcpy(cam_int, ctx->self_cam_int, sizeof(cam_int));
    } else {
        if (ctx->has_bbox) {
            memcpy(bbox_xyxy, ctx->bbox, sizeof(bbox_xyxy));
        } else {
            bbox_xyxy[0] = 0; bbox_xyxy[1] = 0;
            bbox_xyxy[2] = (float)ctx->img_w; bbox_xyxy[3] = (float)ctx->img_h;
        }
        /* Recompute the same affine shape that run_encoder would have cached.
         * ViT-H still uses the upstream 512x512 canvas before W-axis crop. */
        int affine_w = IMG_W, affine_h = IMG_H;
        float aspect_ratio_pre = is_vith ? 0.75f : 1.0f;
        if (is_vith) {
            affine_w = SB_VITH_IMG_H;
            affine_h = SB_VITH_IMG_H;
        } else if (IMG_H != IMG_W) {
            aspect_ratio_pre = (float)IMG_W / (float)IMG_H;
        }
        sam3d_body_compute_bbox_affine(bbox_xyxy, /*padding=*/1.25f,
                                       aspect_ratio_pre,
                                       affine_w, affine_h,
                                       center, scale, warp);
        sam3d_body_default_cam_int(ctx->img_w, ctx->img_h, cam_int);
        if (ctx->focal_hint > 0) {
            cam_int[0] = ctx->focal_hint;
            cam_int[4] = ctx->focal_hint;
        }
    }
    (void)bbox_xyxy;
    t_geom_ms = sb_time_ms() - t_geom0;

    float warp_for_rays[6];
    memcpy(warp_for_rays, warp, sizeof(warp_for_rays));
    int ray_img_h = IMG_H;
    int ray_img_w = IMG_W;
    int ray_rect_reinterpret = (!is_vith && IMG_H != IMG_W);
    if (is_vith) {
        ray_img_h = SB_VITH_IMG_H;
        ray_img_w = SB_VITH_IMG_H;
    }

    /* ---- Permute encoder tokens for the host-backed decoder path. ---- */
    const double t_perm0 = sb_time_ms();
    float *img_emb_pre = NULL;
    void *d_img_emb_chw = NULL;
    int d_img_emb_chw_owned = 0;
    if (!dev_decoder) {
        img_emb_pre = (float *)malloc((size_t)Dc * N_C * sizeof(float));
        if (!img_emb_pre) return CUDA_SAM3D_BODY_E_LOAD;
        const float *patch = ctx->encoder_tokens.data + (size_t)n_prefix * Dc;
        const float *no_mask = ctx->dec.no_mask_embed_h;
        if (!no_mask) { free(img_emb_pre); return CUDA_SAM3D_BODY_E_LOAD; }
        {
            for (int n = 0; n < N_C; n++)
                for (int c = 0; c < Dc; c++)
                    img_emb_pre[(size_t)c * N_C + n] =
                        patch[(size_t)n * Dc + c] + no_mask[c];
        }
    }
    t_encoder_permute_ms = sb_time_ms() - t_perm0;

    /* ---- ray_cond_xyz (host) ---- */
    const double t_rays0 = sb_time_ms();
    float *rays_xyz = (float *)malloc((size_t)N_C * 3 * sizeof(float));
    if (!rays_xyz) { free(img_emb_pre); return CUDA_SAM3D_BODY_E_LOAD; }
    if (sam3d_body_compute_ray_cond_xyz_ex(cam_int, warp_for_rays,
                                           ray_img_h, ray_img_w,
                                           gh, gw, ray_rect_reinterpret,
                                           rays_xyz) != 0) {
        free(img_emb_pre); free(rays_xyz);
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    t_rays_ms = sb_time_ms() - t_rays0;

    /* ---- ray_cond_emb on GPU → img_emb_post (Dc, H, W) ---- */
    const double t_raycond0 = sb_time_ms();
    float *img_emb = NULL;
    if (dev_decoder) {
        const size_t b_img = (size_t)Dc * N_C * sizeof(float);
        if (hipMalloc(&d_img_emb_chw, b_img) != hipSuccess) {
            free(rays_xyz);
            return CUDA_SAM3D_BODY_E_LOAD;
        }
        d_img_emb_chw_owned = 1;
        if (cuda_sam3d_body_ray_cond_from_encoder_dev(
                ctx, ctx->d_proj, n_prefix, rays_xyz, gh, gw,
                d_img_emb_chw) != CUDA_SAM3D_BODY_E_OK) {
            free(rays_xyz);
            sb_free_dev(d_img_emb_chw);
            return CUDA_SAM3D_BODY_E_LOAD;
        }
    } else {
        img_emb = (float *)malloc((size_t)Dc * N_C * sizeof(float));
        if (!img_emb) { free(img_emb_pre); free(rays_xyz); return CUDA_SAM3D_BODY_E_LOAD; }
        if (cuda_sam3d_body_debug_run_ray_cond(ctx, img_emb_pre, rays_xyz,
                                               gh, gw, img_emb) != 0) {
            free(img_emb_pre); free(rays_xyz); free(img_emb);
            return CUDA_SAM3D_BODY_E_LOAD;
        }
    }
    t_ray_cond_ms = sb_time_ms() - t_raycond0;
    free(img_emb_pre);
    free(rays_xyz);

    /* ---- dense_pe in token form. Cached because it only depends on grid. ---- */
    const double t_dense0 = sb_time_ms();
    float *ctx_pe_tok = NULL;
    void *d_ctx_pe_tok_cached = NULL;
    int ctx_pe_tok_owned = 0;
    int pe_rc = dev_decoder
        ? cuda_sam3d_body_ensure_dense_pe_tok_dev(
              ctx, gh, gw, Dc, &d_ctx_pe_tok_cached)
        : cuda_sam3d_body_ensure_dense_pe_tok_host(
              ctx, gh, gw, Dc, &ctx_pe_tok);
    if (pe_rc != CUDA_SAM3D_BODY_E_OK) {
        free(img_emb);
        if (d_img_emb_chw_owned) sb_free_dev(d_img_emb_chw);
        return pe_rc;
    }
    t_dense_pe_ms = sb_time_ms() - t_dense0;

    /* image_emb in token form (HW, Dc) for the decoder layer's context_in. */
    const double t_ctx0 = sb_time_ms();
    float *ctx_in = NULL;
    if (!dev_decoder) {
        ctx_in = (float *)malloc((size_t)N_C * Dc * sizeof(float));
        if (!ctx_in) {
            free(img_emb);
            if (ctx_pe_tok_owned) free(ctx_pe_tok);
            if (d_img_emb_chw_owned) sb_free_dev(d_img_emb_chw);
            return CUDA_SAM3D_BODY_E_LOAD;
        }
        for (int n = 0; n < N_C; n++)
            for (int c = 0; c < Dc; c++)
                ctx_in[(size_t)n * Dc + c] = img_emb[(size_t)c * N_C + n];
    }
    t_ctx_permute_ms = sb_time_ms() - t_ctx0;

    /* ---- condition_info, init_input, prev_input, prompt_in ---- */
    const double t_cond0 = sb_time_ms();
    float ori_img_size[2] = { (float)ctx->img_w, (float)ctx->img_h };
    float img_size[2]     = {
        (float)(is_vith ? SB_VITH_IMG_H : IMG_W),
        (float)(is_vith ? SB_VITH_IMG_H : IMG_H),
    };
    float bbox_scale1[1]  = { scale[0] };
    float condition_info[3];
    sam3d_body_compute_condition_info(center, bbox_scale1, ori_img_size,
                                      cam_int, /*use_intrin_center=*/0,
                                      condition_info);

    const float *ip = (const float *)m->init_pose.data;
    const float *ic = (const float *)m->init_camera.data;
    float init_input[525], prev_input[522], prompt_in[1280];
    memcpy(init_input,         condition_info, 3 * sizeof(float));
    memcpy(init_input + 3,     ip, 519 * sizeof(float));
    memcpy(init_input + 3+519, ic,   3 * sizeof(float));
    memcpy(prev_input,         ip, 519 * sizeof(float));
    memcpy(prev_input + 519,   ic,   3 * sizeof(float));
    if (sam3d_body_invalid_prompt_token(m, prompt_in) != 0) {
        free(img_emb); if (ctx_pe_tok_owned) free(ctx_pe_tok); free(ctx_in);
        if (d_img_emb_chw_owned) sb_free_dev(d_img_emb_chw);
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    t_condition_ms = sb_time_ms() - t_cond0;

    /* ---- build_tokens on GPU → init_x, init_xpe ---- */
    const double t_build0 = sb_time_ms();
    float *init_x   = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *init_xpe = (float *)malloc((size_t)N_Q * D * sizeof(float));
    if (!init_x || !init_xpe) {
        free(img_emb); if (ctx_pe_tok_owned) free(ctx_pe_tok); free(ctx_in);
        if (d_img_emb_chw_owned) sb_free_dev(d_img_emb_chw);
        free(init_x); free(init_xpe);
        return CUDA_SAM3D_BODY_E_LOAD;
    }
    if (cuda_sam3d_body_debug_run_build_tokens(ctx, init_input, prev_input,
                                               prompt_in, init_x, init_xpe) != 0) {
        free(img_emb); if (ctx_pe_tok_owned) free(ctx_pe_tok); free(ctx_in);
        if (d_img_emb_chw_owned) sb_free_dev(d_img_emb_chw);
        free(init_x); free(init_xpe);
        return CUDA_SAM3D_BODY_E_LOAD;
    }
    t_build_tokens_ms = sb_time_ms() - t_build0;

    /* ---- camera_batch ---- */
    sam3d_body_camera_batch B;
    memcpy(B.cam_int,      cam_int, 9 * sizeof(float));
    memcpy(B.bbox_center,  center,  2 * sizeof(float));
    B.bbox_scale = scale[0];
    memcpy(B.ori_img_size, ori_img_size, 2 * sizeof(float));
    memcpy(B.img_size,     img_size,     2 * sizeof(float));
    memcpy(B.affine_trans, warp_for_rays, 6 * sizeof(float));
    B.use_intrin_center    = 0;
    B.default_scale_factor = 1.0f;
    int rc_total = 0, rc = 0;

    /* ---- Per-layer working buffers + MHR scratch ---- */
    const double t_workspace0 = sb_time_ms();
    const void *d_kpu_img_run = NULL;
    if (ctx->mhr_gpu_enabled) {
        size_t b_img = (size_t)Dc * (size_t)gh * (size_t)gw * sizeof(float);
        if (d_img_emb_chw) {
            d_kpu_img_run = d_img_emb_chw;
        } else if (ctx->d_kpu_img_cap < b_img) {
            sb_free_dev(ctx->d_kpu_img);
            ctx->d_kpu_img = NULL;
            ctx->d_kpu_img_cap = 0;
            if (hipMalloc(&ctx->d_kpu_img, b_img) != hipSuccess) {
                free(img_emb); if (ctx_pe_tok_owned) free(ctx_pe_tok); free(ctx_in);
                if (d_img_emb_chw_owned) sb_free_dev(d_img_emb_chw);
                return CUDA_SAM3D_BODY_E_LOAD;
            }
            ctx->d_kpu_img_cap = b_img;
        }
        if (!d_kpu_img_run &&
            hipMemcpy(ctx->d_kpu_img, img_emb, b_img,
                      hipMemcpyHostToDevice) != hipSuccess) {
            free(img_emb); if (ctx_pe_tok_owned) free(ctx_pe_tok); free(ctx_in);
            if (d_img_emb_chw_owned) sb_free_dev(d_img_emb_chw);
            return CUDA_SAM3D_BODY_E_LOAD;
        }
        if (!d_kpu_img_run) d_kpu_img_run = ctx->d_kpu_img;
        ctx->kpu_use_cached_img = 1;
    }
    float *tokens   = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *tokens_b = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *augment  = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *tokens_n = (float *)malloc((size_t)N_Q * D * sizeof(float));
    const size_t mhr_scratch_floats = (size_t)1 *
        (889 + (size_t)J*8*2 + (size_t)V*3*3);
    float *mhr_scratch = (float *)malloc(mhr_scratch_floats * sizeof(float));
    float *verts_cm = (float *)malloc((size_t)V * 3 * sizeof(float));
    float *gskel_cm = (float *)malloc((size_t)J * 8 * sizeof(float));
    float *verts_m  = (float *)malloc((size_t)V * 3 * sizeof(float));
    float *jc_m     = (float *)malloc((size_t)J * 3 * sizeof(float));
    void *d_dec_tokens = NULL, *d_dec_tokens_b = NULL, *d_dec_aug = NULL;
    void *d_dec_ctx = NULL, *d_dec_cpe = NULL;
    int d_dec_cpe_owned = 0;
    sb_dec_layer_scratch_dev DS;
    memset(&DS, 0, sizeof(DS));
    sb_kpu_scratch_dev KS;
    memset(&KS, 0, sizeof(KS));
    sb_head_scratch_dev HS;
    memset(&HS, 0, sizeof(HS));
    if (!tokens || !tokens_b || !augment || !tokens_n ||
        !mhr_scratch || !verts_cm || !gskel_cm || !verts_m || !jc_m) {
        free(img_emb); if (ctx_pe_tok_owned) free(ctx_pe_tok); free(ctx_in);
        if (d_img_emb_chw_owned) sb_free_dev(d_img_emb_chw);
        free(init_x); free(init_xpe);
        free(tokens); free(tokens_b); free(augment); free(tokens_n);
        free(mhr_scratch); free(verts_cm); free(gskel_cm); free(verts_m); free(jc_m);
        return CUDA_SAM3D_BODY_E_LOAD;
    }
    memcpy(tokens,  init_x,   (size_t)N_Q * D * sizeof(float));
    memcpy(augment, init_xpe, (size_t)N_Q * D * sizeof(float));
    if (dev_decoder) {
        const size_t bx_q = (size_t)N_Q * D * sizeof(float);
        const size_t bx_c = (size_t)N_C * Dc * sizeof(float);
        const size_t b_qE = (size_t)N_Q * SB_DEC_HEADS * SB_DEC_HEAD_DIM * sizeof(float);
        const size_t b_kvE = (size_t)((N_Q > N_C ? N_Q : N_C)) *
                             SB_DEC_HEADS * SB_DEC_HEAD_DIM * sizeof(float);
        const size_t b_qF = (size_t)N_Q * SB_DEC_FFN * sizeof(float);
        #define DEV_ALLOC(ptr, bytes) do { \
            if (hipMalloc(&(ptr), (bytes)) != hipSuccess) { \
                rc_total = -1; goto cleanup; \
            } \
        } while (0)
        DEV_ALLOC(d_dec_tokens, bx_q);
        DEV_ALLOC(d_dec_tokens_b, bx_q);
        DEV_ALLOC(d_dec_aug, bx_q);
        DEV_ALLOC(d_dec_ctx, bx_c);
        d_dec_cpe = d_ctx_pe_tok_cached;
        d_dec_cpe_owned = 0;
        if (!d_dec_cpe) {
            DEV_ALLOC(d_dec_cpe, bx_c);
            d_dec_cpe_owned = 1;
        }
        DEV_ALLOC(DS.xpe_n, bx_q);
        DEV_ALLOC(DS.cpe_n, bx_c);
        DEV_ALLOC(DS.x_ln, bx_q);
        DEV_ALLOC(DS.qk, bx_q);
        DEV_ALLOC(DS.Q, b_qE);
        DEV_ALLOC(DS.K, b_kvE);
        DEV_ALLOC(DS.V, b_kvE);
        DEV_ALLOC(DS.attn, b_qE);
        DEV_ALLOC(DS.proj, bx_q);
        DEV_ALLOC(DS.x1, bx_q);
        DEV_ALLOC(DS.q_in, bx_q);
        DEV_ALLOC(DS.k_in, bx_c);
        DEV_ALLOC(DS.ctx_ln, bx_c);
        DEV_ALLOC(DS.x2, bx_q);
        DEV_ALLOC(DS.x_ln3, bx_q);
        DEV_ALLOC(DS.ffn_h, b_qF);
        DEV_ALLOC(DS.ffn_o, bx_q);
        DEV_ALLOC(KS.gxy, (size_t)K * 2 * sizeof(float));
        DEV_ALLOC(KS.invalid, (size_t)K * sizeof(int));
        DEV_ALLOC(KS.p_h, (size_t)K * D * sizeof(float));
        DEV_ALLOC(KS.p_o, (size_t)K * D * sizeof(float));
        DEV_ALLOC(KS.kp_feats, (size_t)K * Dc * sizeof(float));
        DEV_ALLOC(KS.kp_proj, (size_t)K * D * sizeof(float));
        DEV_ALLOC(KS.kp3d_norm, (size_t)K * 3 * sizeof(float));
        DEV_ALLOC(KS.p3_h, (size_t)K * D * sizeof(float));
        DEV_ALLOC(KS.p3_o, (size_t)K * D * sizeof(float));
        DEV_ALLOC(HS.norm_row, (size_t)D * sizeof(float));
        DEV_ALLOC(HS.pose_h, (size_t)D * sizeof(float));
        DEV_ALLOC(HS.pose_raw, (size_t)SB_DEC_NPOSE * sizeof(float));
        DEV_ALLOC(HS.cam_h, (size_t)D * sizeof(float));
        DEV_ALLOC(HS.cam_raw, (size_t)SB_DEC_NCAM * sizeof(float));
        #undef DEV_ALLOC
        if (hipMemcpy(d_dec_tokens, tokens, bx_q, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(d_dec_aug, augment, bx_q, hipMemcpyHostToDevice) != hipSuccess) {
            rc_total = -1; goto cleanup;
        }
        if (d_img_emb_chw) {
            const double t_ctx_dev0 = sb_time_ms();
            if (cuda_sam3d_body_chw_to_tok_dev(
                    ctx, d_dec_ctx, d_img_emb_chw, N_C, Dc) !=
                CUDA_SAM3D_BODY_E_OK ||
                hipDeviceSynchronize() != hipSuccess) {
                rc_total = -1; goto cleanup;
            }
            t_ctx_permute_ms += sb_time_ms() - t_ctx_dev0;
        } else if (hipMemcpy(d_dec_ctx, ctx_in, bx_c,
                             hipMemcpyHostToDevice) != hipSuccess) {
            rc_total = -1; goto cleanup;
        }
        if (d_dec_cpe_owned &&
            hipMemcpy(d_dec_cpe, ctx_pe_tok, bx_c,
                      hipMemcpyHostToDevice) != hipSuccess) {
            rc_total = -1; goto cleanup;
        }
    }
    free(init_x); free(init_xpe);
    t_workspace_ms = sb_time_ms() - t_workspace0;

    /* ---- Per-layer loop. ---- */
    float pose_raw[519], cam_raw[3], pose519[519], cam3[3];
    float mp_buf[204], shape_buf[45], face_buf[72];
    float kp3d_post[70 * 3];
    float kp2d_full[70 * 2], kp2d_crop[70 * 2], kp2d_dep[70];
    float pred_cam_t_world[3];

    for (int li = 0; li < NL; li++) {
        double t0 = sb_time_ms();
        if (dev_decoder) {
            rc = cuda_sam3d_body_run_decoder_layer_dev(
                    ctx, li, d_dec_tokens, d_dec_ctx, d_dec_aug, d_dec_cpe,
                    N_Q, N_C, d_dec_tokens_b, &DS);
        } else {
            rc = cuda_sam3d_body_debug_run_decoder_layer(
                    ctx, li, tokens, ctx_in, augment, ctx_pe_tok,
                    N_Q, N_C, tokens_b);
        }
        if (rc != 0) { rc_total = rc; goto cleanup; }
        if (dev_decoder) {
            void *td = d_dec_tokens; d_dec_tokens = d_dec_tokens_b; d_dec_tokens_b = td;
            if (hipDeviceSynchronize() != hipSuccess) {
                rc_total = -1; goto cleanup;
            }
        } else {
            float *t = tokens; tokens = tokens_b; tokens_b = t;
        }
        t_layer_ms[li] += sb_time_ms() - t0;

        if (li >= NL - 1) break;

        t0 = sb_time_ms();
        if (dev_decoder) {
            rc = cuda_sam3d_body_norm_heads_dev(
                    ctx, d_dec_tokens, pose_raw, cam_raw, &HS);
        } else {
            rc = cuda_sam3d_body_debug_run_norm_and_heads(
                    ctx, tokens, N_Q, NULL, pose_raw, cam_raw);
        }
        if (rc != 0) { rc_total = rc; goto cleanup; }
        t_norm_heads_ms[li] += sb_time_ms() - t0;
        for (int i = 0; i < 519; i++) pose519[i] = pose_raw[i] + ip[i];
        for (int i = 0; i < 3;   i++) cam3[i]    = cam_raw[i]  + ic[i];

        t0 = sb_time_ms();
        if (sam3d_body_decode_pose_raw(m, pose519, /*enable_hand_model=*/0,
                                       mp_buf, shape_buf, face_buf) != 0) {
            rc_total = -1; goto cleanup;
        }
        t_decode_pose_ms[li] += sb_time_ms() - t0;
        t0 = sb_time_ms();
        int mhr_rc = ctx->mhr_gpu_enabled
            ? cuda_sam3d_body_mhr_forward_hybrid(
                  ctx, mp_buf, shape_buf, face_buf,
                  1, SAM3DB_MHR_THREADS(), mhr_scratch, NULL, gskel_cm)
            : sam3d_body_mhr_forward((const sam3d_body_mhr_assets *)ctx->cpu_mhr,
                                      mp_buf, shape_buf, face_buf,
                                      1, 1, SAM3DB_MHR_THREADS(), mhr_scratch,
                                      verts_cm, gskel_cm);
        if (mhr_rc != 0) {
            rc_total = -1; goto cleanup;
        }
        t_mhr_ms[li] += sb_time_ms() - t0;
        t0 = sb_time_ms();
        if (ctx->mhr_gpu_enabled) {
            if (cuda_sam3d_body_keypoints_project_cached(
                    ctx, gskel_cm, cam3, &B,
                    kp3d_post, kp2d_full, kp2d_crop, kp2d_dep,
                    pred_cam_t_world) != CUDA_SAM3D_BODY_E_OK) {
                rc_total = -1; goto cleanup;
            }
        } else {
            for (int i = 0; i < V * 3; i++) verts_m[i] = verts_cm[i] * 0.01f;
            for (int j = 0; j < J; j++) {
                jc_m[j*3 + 0] = gskel_cm[(size_t)j*8 + 0] * 0.01f;
                jc_m[j*3 + 1] = gskel_cm[(size_t)j*8 + 1] * 0.01f;
                jc_m[j*3 + 2] = gskel_cm[(size_t)j*8 + 2] * 0.01f;
            }
            if (sam3d_body_keypoints_from_mesh(m, verts_m, jc_m,
                                               /*enable_hand_model=*/0,
                                               /*n_threads=*/0, kp3d_post) != 0) {
                rc_total = -1; goto cleanup;
            }
        }
        t_keypoints_ms[li] += sb_time_ms() - t0;
        if (!ctx->mhr_gpu_enabled) {
            t0 = sb_time_ms();
            if (sam3d_body_camera_project(kp3d_post, cam3, &B, K,
                                          kp2d_full, kp2d_crop, kp2d_dep,
                                          pred_cam_t_world) != 0) {
                rc_total = -1; goto cleanup;
            }
            t_project_ms[li] += sb_time_ms() - t0;
        }

        t0 = sb_time_ms();
        if (dev_decoder) {
            rc = cuda_sam3d_body_kp_token_update_dev(
                    ctx, li, gh, gw, N_Q, kp2d_crop, kp2d_dep,
                    d_kpu_img_run, d_dec_tokens, d_dec_aug, &KS);
        } else {
            rc = cuda_sam3d_body_debug_run_kp_token_update(
                    ctx, li, img_emb, gh, gw,
                    kp2d_crop, kp2d_dep, kp3d_post,
                    N_Q, tokens, augment);
        }
        if (rc != 0) { rc_total = rc; goto cleanup; }
        t_kp_update_ms[li] += sb_time_ms() - t0;
    }

    /* ---- Final norm + heads + MHR (post-loop). ---- */
    {
    const int ti = NL;
    double t0 = sb_time_ms();
    if (dev_decoder) {
        rc = cuda_sam3d_body_norm_heads_dev(
                ctx, d_dec_tokens, pose_raw, cam_raw, &HS);
    } else {
        rc = cuda_sam3d_body_debug_run_norm_and_heads(
                ctx, tokens, N_Q, NULL, pose_raw, cam_raw);
    }
    if (rc != 0) { rc_total = rc; goto cleanup; }
    t_norm_heads_ms[ti] += sb_time_ms() - t0;
    for (int i = 0; i < 519; i++) pose519[i] = pose_raw[i] + ip[i];
    for (int i = 0; i < 3;   i++) cam3[i]    = cam_raw[i]  + ic[i];

    t0 = sb_time_ms();
    if (sam3d_body_decode_pose_raw(m, pose519, /*enable_hand_model=*/0,
                                   mp_buf, shape_buf, face_buf) != 0) {
        rc_total = -1; goto cleanup;
    }
    t_decode_pose_ms[ti] += sb_time_ms() - t0;
    t0 = sb_time_ms();
    int mhr_rc = ctx->mhr_gpu_enabled
        ? cuda_sam3d_body_mhr_forward_hybrid(
              ctx, mp_buf, shape_buf, face_buf,
              1, SAM3DB_MHR_THREADS(), mhr_scratch, verts_cm, gskel_cm)
        : sam3d_body_mhr_forward((const sam3d_body_mhr_assets *)ctx->cpu_mhr,
                                  mp_buf, shape_buf, face_buf,
                                  1, 1, SAM3DB_MHR_THREADS(), mhr_scratch,
                                  verts_cm, gskel_cm);
    if (mhr_rc != 0) {
        rc_total = -1; goto cleanup;
    }
    t_mhr_ms[ti] += sb_time_ms() - t0;
    t0 = sb_time_ms();
    for (int i = 0; i < V * 3; i++) verts_m[i] = verts_cm[i] * 0.01f;
    if (ctx->mhr_gpu_enabled) {
        if (cuda_sam3d_body_keypoints_project_cached(
                ctx, gskel_cm, cam3, &B,
                kp3d_post, kp2d_full, kp2d_crop, kp2d_dep,
                pred_cam_t_world) != CUDA_SAM3D_BODY_E_OK) {
            rc_total = -1; goto cleanup;
        }
    } else {
        for (int j = 0; j < J; j++) {
            jc_m[j*3 + 0] = gskel_cm[(size_t)j*8 + 0] * 0.01f;
            jc_m[j*3 + 1] = gskel_cm[(size_t)j*8 + 1] * 0.01f;
            jc_m[j*3 + 2] = gskel_cm[(size_t)j*8 + 2] * 0.01f;
        }
        if (sam3d_body_keypoints_from_mesh(m, verts_m, jc_m,
                                           /*enable_hand_model=*/0,
                                           /*n_threads=*/0, kp3d_post) != 0) {
            rc_total = -1; goto cleanup;
        }
    }
    t_keypoints_ms[ti] += sb_time_ms() - t0;
    if (!ctx->mhr_gpu_enabled) {
        t0 = sb_time_ms();
        if (sam3d_body_camera_project(kp3d_post, cam3, &B, K,
                                      kp2d_full, kp2d_crop, kp2d_dep,
                                      pred_cam_t_world) != 0) {
            rc_total = -1; goto cleanup;
        }
        t_project_ms[ti] += sb_time_ms() - t0;
    }
    }

    /* ---- Populate ctx outputs (vertices in meters, kp arrays, mhr_params,
     *      cam_t world, focal). ---- */
    {
    const double t_output0 = sb_time_ms();
    free(ctx->vertices);
    ctx->vertices = (float *)malloc((size_t)V * 3 * sizeof(float));
    if (!ctx->vertices) { rc_total = -1; goto cleanup; }
    /* Post-flip cam frame: matches sam3d_body_decoder_forward_full. */
    for (int i = 0; i < V; i++) {
        ctx->vertices[i*3 + 0] =  verts_m[i*3 + 0];
        ctx->vertices[i*3 + 1] = -verts_m[i*3 + 1];
        ctx->vertices[i*3 + 2] = -verts_m[i*3 + 2];
    }
    ctx->n_vertices = V;

    free(ctx->keypoints_3d);
    ctx->keypoints_3d = (float *)malloc((size_t)K * 3 * sizeof(float));
    if (!ctx->keypoints_3d) { rc_total = -1; goto cleanup; }
    memcpy(ctx->keypoints_3d, kp3d_post, (size_t)K * 3 * sizeof(float));
    ctx->n_kp_3d = K;

    free(ctx->keypoints_2d);
    ctx->keypoints_2d = (float *)malloc((size_t)K * 2 * sizeof(float));
    if (!ctx->keypoints_2d) { rc_total = -1; goto cleanup; }
    memcpy(ctx->keypoints_2d, kp2d_full, (size_t)K * 2 * sizeof(float));
    ctx->n_kp_2d = K;

    free(ctx->mhr_params);
    ctx->mhr_params = (float *)malloc(519 * sizeof(float));
    if (!ctx->mhr_params) { rc_total = -1; goto cleanup; }
    memcpy(ctx->mhr_params, pose519, 519 * sizeof(float));
    ctx->mhr_params_n = 519;

    memcpy(ctx->cam_t, pred_cam_t_world, sizeof(ctx->cam_t));
    ctx->focal_px = cam_int[0];

    /* Faces from the MHR-head asset (kept on host as i64 → i32). */
    if (m->faces.data && !ctx->faces) {
        int F = (int)m->faces.dims[0];
        ctx->faces = (int32_t *)malloc((size_t)F * 3 * sizeof(int32_t));
        if (!ctx->faces) { rc_total = -1; goto cleanup; }
        const int64_t *src = (const int64_t *)m->faces.data;
        for (int i = 0; i < F * 3; i++)
            ctx->faces[i] = (int32_t)src[i];
        ctx->n_faces = F;
    }
    t_output_ms = sb_time_ms() - t_output0;
    }

cleanup:
    ctx->kpu_use_cached_img = 0;
    if (ctx->verbose >= 1) {
        double sum_layer = 0.0, sum_norm = 0.0, sum_decode = 0.0;
        double sum_mhr = 0.0, sum_kp = 0.0, sum_proj = 0.0, sum_kpu = 0.0;
        for (int i = 0; i < NL; i++) {
            sum_layer += t_layer_ms[i];
            sum_kpu += t_kp_update_ms[i];
        }
        for (int i = 0; i <= NL; i++) {
            sum_norm += t_norm_heads_ms[i];
            sum_decode += t_decode_pose_ms[i];
            sum_mhr += t_mhr_ms[i];
            sum_kp += t_keypoints_ms[i];
            sum_proj += t_project_ms[i];
        }
        fprintf(stderr,
                "[sam3d_body][timing] decoder total %.3f ms "
                "(mhr_cache %.3f, geom %.3f, encoder_permute %.3f, rays %.3f, "
                "ray_cond %.3f, dense_pe %.3f, ctx_permute %.3f, "
                "condition %.3f, build_tokens %.3f, workspace %.3f, "
                "layers %.3f, norm_heads %.3f, decode_pose %.3f, "
                "mhr %.3f, keypoints %.3f, camera_project %.3f, "
                "kp_update %.3f, output %.3f)\n",
                sb_time_ms() - t_decoder0,
                t_mhr_cache_ms, t_geom_ms, t_encoder_permute_ms, t_rays_ms,
                t_ray_cond_ms, t_dense_pe_ms, t_ctx_permute_ms,
                t_condition_ms, t_build_tokens_ms, t_workspace_ms,
                sum_layer, sum_norm, sum_decode, sum_mhr, sum_kp,
                sum_proj, sum_kpu, t_output_ms);
        for (int i = 0; i < NL; i++) {
            fprintf(stderr,
                    "[sam3d_body][timing] decoder layer %d: layer %.3f ms",
                    i, t_layer_ms[i]);
            if (i < NL - 1) {
                fprintf(stderr,
                        ", norm_heads %.3f, decode_pose %.3f, mhr %.3f, "
                        "keypoints %.3f, camera_project %.3f, kp_update %.3f",
                        t_norm_heads_ms[i], t_decode_pose_ms[i],
                        t_mhr_ms[i], t_keypoints_ms[i], t_project_ms[i],
                        t_kp_update_ms[i]);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr,
                "[sam3d_body][timing] decoder final: norm_heads %.3f ms, "
                "decode_pose %.3f, mhr %.3f, keypoints %.3f, "
                "camera_project %.3f\n",
                t_norm_heads_ms[NL], t_decode_pose_ms[NL], t_mhr_ms[NL],
                t_keypoints_ms[NL], t_project_ms[NL]);
    }
    free(img_emb); if (ctx_pe_tok_owned) free(ctx_pe_tok); free(ctx_in);
    if (d_img_emb_chw_owned) sb_free_dev(d_img_emb_chw);
    free(tokens); free(tokens_b); free(augment); free(tokens_n);
    free(mhr_scratch); free(verts_cm); free(gskel_cm); free(verts_m); free(jc_m);
    sb_free_dev(d_dec_tokens);
    sb_free_dev(d_dec_tokens_b);
    sb_free_dev(d_dec_aug);
    sb_free_dev(d_dec_ctx);
    if (d_dec_cpe_owned) sb_free_dev(d_dec_cpe);
    sb_free_dev(DS.xpe_n);
    sb_free_dev(DS.cpe_n);
    sb_free_dev(DS.x_ln);
    sb_free_dev(DS.qk);
    sb_free_dev(DS.Q);
    sb_free_dev(DS.K);
    sb_free_dev(DS.V);
    sb_free_dev(DS.attn);
    sb_free_dev(DS.proj);
    sb_free_dev(DS.x1);
    sb_free_dev(DS.q_in);
    sb_free_dev(DS.k_in);
    sb_free_dev(DS.ctx_ln);
    sb_free_dev(DS.x2);
    sb_free_dev(DS.x_ln3);
    sb_free_dev(DS.ffn_h);
    sb_free_dev(DS.ffn_o);
    sb_free_dev(KS.gxy);
    sb_free_dev(KS.invalid);
    sb_free_dev(KS.p_h);
    sb_free_dev(KS.p_o);
    sb_free_dev(KS.kp_feats);
    sb_free_dev(KS.kp_proj);
    sb_free_dev(KS.kp3d_norm);
    sb_free_dev(KS.p3_h);
    sb_free_dev(KS.p3_o);
    sb_free_dev(HS.norm_row);
    sb_free_dev(HS.pose_h);
    sb_free_dev(HS.pose_raw);
    sb_free_dev(HS.cam_h);
    sb_free_dev(HS.cam_raw);
    return (rc_total == 0) ? CUDA_SAM3D_BODY_E_OK
                           : CUDA_SAM3D_BODY_E_LOAD;
}

/* MHR skinning is folded into run_decoder; keep this entry point as a shim
 * for API symmetry. */
int cuda_sam3d_body_run_mhr(cuda_sam3d_body_ctx *ctx)
{
    if (!ctx) return CUDA_SAM3D_BODY_E_INVAL;
    return ctx->vertices ? CUDA_SAM3D_BODY_E_OK
                         : CUDA_SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int cuda_sam3d_body_run_all(cuda_sam3d_body_ctx *ctx)
{
    const double t0 = sb_time_ms();
    double t_encoder = 0.0, t_decoder = 0.0, t_mhr = 0.0;
    double s = sb_time_ms();
    int rc = cuda_sam3d_body_run_encoder(ctx);
    if (rc != CUDA_SAM3D_BODY_E_OK) return rc;
    t_encoder = sb_time_ms() - s;
    s = sb_time_ms();
    rc = cuda_sam3d_body_run_decoder(ctx);
    if (rc != CUDA_SAM3D_BODY_E_OK) return rc;
    t_decoder = sb_time_ms() - s;
    s = sb_time_ms();
    rc = cuda_sam3d_body_run_mhr(ctx);
    t_mhr = sb_time_ms() - s;
    if (ctx && ctx->verbose >= 1) {
        fprintf(stderr,
                "[sam3d_body][timing] run_all total %.3f ms "
                "(encoder %.3f, decoder %.3f, mhr_shim %.3f)\n",
                sb_time_ms() - t0, t_encoder, t_decoder, t_mhr);
    }
    return rc;
}

int cuda_sam3d_body_get_encoder_tokens(cuda_sam3d_body_ctx *ctx,
                                       float *out, int *out_n, int *out_dim)
{
    if (!ctx || !out_n || !out_dim) return CUDA_SAM3D_BODY_E_INVAL;
    *out_n = ctx->encoder_tokens.n;
    *out_dim = ctx->encoder_tokens.c;
    if (out && ctx->encoder_tokens.data) {
        memcpy(out, ctx->encoder_tokens.data,
               (size_t)ctx->encoder_tokens.n *
               (size_t)ctx->encoder_tokens.c * sizeof(float));
        return CUDA_SAM3D_BODY_E_OK;
    }
    if (ctx->encoder_tokens_dev_valid) {
        if (out &&
            hipMemcpy(out, ctx->d_proj,
                      (size_t)ctx->encoder_tokens.n *
                      (size_t)ctx->encoder_tokens.c * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess)
            return CUDA_SAM3D_BODY_E_LOAD;
        return CUDA_SAM3D_BODY_E_OK;
    }
    return CUDA_SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int cuda_sam3d_body_get_mhr_params(cuda_sam3d_body_ctx *ctx, float *out, int *out_n)
{
    if (!ctx || !out_n) return CUDA_SAM3D_BODY_E_INVAL;
    *out_n = ctx->mhr_params_n;
    if (out && ctx->mhr_params)
        memcpy(out, ctx->mhr_params, (size_t)ctx->mhr_params_n * sizeof(float));
    return ctx->mhr_params ? CUDA_SAM3D_BODY_E_OK
                           : CUDA_SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int cuda_sam3d_body_get_cam(cuda_sam3d_body_ctx *ctx,
                            float *out_cam_t_xyz, float *out_focal_px)
{
    if (!ctx) return CUDA_SAM3D_BODY_E_INVAL;
    if (out_cam_t_xyz) memcpy(out_cam_t_xyz, ctx->cam_t, sizeof(ctx->cam_t));
    if (out_focal_px)  *out_focal_px = ctx->focal_px;
    return ctx->focal_px > 0 ? CUDA_SAM3D_BODY_E_OK
                             : CUDA_SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int cuda_sam3d_body_get_vertices(cuda_sam3d_body_ctx *ctx, float *out, int *out_v)
{
    if (!ctx || !out_v) return CUDA_SAM3D_BODY_E_INVAL;
    *out_v = ctx->n_vertices;
    if (out && ctx->vertices)
        memcpy(out, ctx->vertices, (size_t)ctx->n_vertices * 3 * sizeof(float));
    return ctx->vertices ? CUDA_SAM3D_BODY_E_OK
                         : CUDA_SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int cuda_sam3d_body_get_faces(cuda_sam3d_body_ctx *ctx, int32_t *out, int *out_f)
{
    if (!ctx || !out_f) return CUDA_SAM3D_BODY_E_INVAL;
    *out_f = ctx->n_faces;
    if (out && ctx->faces)
        memcpy(out, ctx->faces, (size_t)ctx->n_faces * 3 * sizeof(int32_t));
    return ctx->faces ? CUDA_SAM3D_BODY_E_OK : CUDA_SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int cuda_sam3d_body_get_keypoints_3d(cuda_sam3d_body_ctx *ctx, float *out, int *out_k)
{
    if (!ctx || !out_k) return CUDA_SAM3D_BODY_E_INVAL;
    *out_k = ctx->n_kp_3d;
    if (out && ctx->keypoints_3d)
        memcpy(out, ctx->keypoints_3d, (size_t)ctx->n_kp_3d * 3 * sizeof(float));
    return ctx->keypoints_3d ? CUDA_SAM3D_BODY_E_OK
                             : CUDA_SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int cuda_sam3d_body_get_keypoints_2d(cuda_sam3d_body_ctx *ctx, float *out, int *out_k)
{
    if (!ctx || !out_k) return CUDA_SAM3D_BODY_E_INVAL;
    *out_k = ctx->n_kp_2d;
    if (out && ctx->keypoints_2d)
        memcpy(out, ctx->keypoints_2d, (size_t)ctx->n_kp_2d * 2 * sizeof(float));
    return ctx->keypoints_2d ? CUDA_SAM3D_BODY_E_OK
                             : CUDA_SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int cuda_sam3d_body_debug_override_encoder(cuda_sam3d_body_ctx *ctx,
                                           const float *tokens, int n, int dim)
{
    if (!ctx || !tokens || n <= 0 || dim <= 0) return CUDA_SAM3D_BODY_E_INVAL;
    free(ctx->encoder_tokens.data);
    ctx->encoder_tokens.data = (float *)malloc((size_t)n * dim * sizeof(float));
    if (!ctx->encoder_tokens.data) return CUDA_SAM3D_BODY_E_INVAL;
    memcpy(ctx->encoder_tokens.data, tokens, (size_t)n * dim * sizeof(float));
    ctx->encoder_tokens.n = n; ctx->encoder_tokens.c = dim;
    ctx->encoder_tokens_dev_valid = 0;
    return CUDA_SAM3D_BODY_E_OK;
}

int cuda_sam3d_body_debug_override_mhr_params(cuda_sam3d_body_ctx *ctx,
                                              const float *params, int n)
{
    if (!ctx || !params || n <= 0) return CUDA_SAM3D_BODY_E_INVAL;
    free(ctx->mhr_params);
    ctx->mhr_params = (float *)malloc((size_t)n * sizeof(float));
    if (!ctx->mhr_params) return CUDA_SAM3D_BODY_E_INVAL;
    memcpy(ctx->mhr_params, params, (size_t)n * sizeof(float));
    ctx->mhr_params_n = n;
    return CUDA_SAM3D_BODY_E_OK;
}

int cuda_sam3d_body_debug_set_normalized_input(cuda_sam3d_body_ctx *ctx,
                                               const float *chw, int H, int W)
{
    if (!ctx || !chw || H <= 0 || W <= 0) return CUDA_SAM3D_BODY_E_INVAL;
    /* DINOv3 accepts the configured HxW; ViT-H remains fixed 512x384. */
    int exp_h, exp_w;
    if (ctx->cfg.backbone == CUDA_SAM3D_BODY_BACKBONE_VITH) {
        exp_h = SB_VITH_IMG_H; exp_w = SB_VITH_IMG_W;
    } else {
        exp_h = ctx->cfg.image_height; exp_w = ctx->cfg.image_width;
    }
    if (H != exp_h || W != exp_w) {
        fprintf(stderr,
                "sam3d_body: normalized input must be %dx%d, got %dx%d\n",
                exp_h, exp_w, H, W);
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    size_t bytes = (size_t)3 * H * W * sizeof(float);
    if (!ctx->d_img_f32) {
        if (hipMalloc(&ctx->d_img_f32, bytes) != hipSuccess)
            return CUDA_SAM3D_BODY_E_LOAD;
    }
    if (hipMemcpy(ctx->d_img_f32, chw, bytes,
                  hipMemcpyHostToDevice) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    ctx->has_norm_input = 1;
    return CUDA_SAM3D_BODY_E_OK;
}

int cuda_sam3d_body_debug_run_ray_cond(cuda_sam3d_body_ctx *ctx,
                                       const float *image_emb_chw,
                                       const float *rays_hwc,
                                       int H, int W,
                                       float *out_chw)
{
    if (!ctx || !image_emb_chw || !rays_hwc || !out_chw || H <= 0 || W <= 0)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->dec.loaded) {
        fprintf(stderr, "sam3d_body: decoder not loaded\n");
        return CUDA_SAM3D_BODY_E_LOAD;
    }
    const int C_img    = SB_DEC_KV_DIM;          /* 1280 */
    const int num_bands = 16;
    const int C_fp     = 3 + 6 * num_bands;       /* 99 */
    const int C_pre    = C_img + C_fp;            /* 1379 */
    const int N        = H * W;

    void *d_pre = NULL, *d_post = NULL, *d_rays = NULL, *d_out = NULL;
    size_t pre_bytes  = (size_t)C_pre * N * sizeof(float);
    size_t post_bytes = (size_t)C_img * N * sizeof(float);
    size_t rays_bytes = (size_t)N * 3 * sizeof(float);
    if (hipMalloc(&d_pre,  pre_bytes)  != hipSuccess) goto fail;
    if (hipMalloc(&d_post, post_bytes) != hipSuccess) goto fail;
    if (hipMalloc(&d_rays, rays_bytes) != hipSuccess) goto fail;
    if (hipMalloc(&d_out,  post_bytes) != hipSuccess) goto fail;

    /* Upload image_emb into rows [0..C_img) of preconv. */
    if (hipMemcpy(d_pre, image_emb_chw,
                  (size_t)C_img * N * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_rays, rays_hwc, rays_bytes,
                  hipMemcpyHostToDevice) != hipSuccess) goto fail;

    /* Fill rows [C_img..C_img+99) with fourier features. */
    {
        float *fp_chw = (float *)d_pre + (size_t)C_img * N;
        struct __attribute__((packed)) {
            void *fp_chw; void *rays; int N; int nb;
        } a = { fp_chw, d_rays, N, num_bands };
        unsigned bx = 256;
        unsigned gx = (unsigned)((N + bx - 1) / bx);
        if (sb_launch(ctx->fn_ray_cond_fourier, gx, 1, 1, bx, 1, 1, 0,
                      &a, sizeof(a)) < 0) goto fail;
    }

    /* postconv = ray_cond_conv_w (C_img, C_pre) @ preconv (C_pre, N) */
    {
        struct __attribute__((packed)) {
            void *out; void *W; void *src; int Co; int Ci; int N;
        } a = { d_post, ctx->dec.ray_cond_conv_w, d_pre,
                C_img, C_pre, N };
        unsigned bx = 256;
        unsigned gx = (unsigned)((N + bx - 1) / bx);
        unsigned gy = (unsigned)C_img;
        if (sb_launch(ctx->fn_conv1x1_chw, gx, gy, 1, bx, 1, 1, 0,
                      &a, sizeof(a)) < 0) goto fail;
    }

    /* out = LayerNorm2d(postconv) using ray_cond_norm.{weight,bias}. */
    {
        struct __attribute__((packed)) {
            void *dst; void *src; void *g; void *b;
            int C; int N; float eps;
        } a = { d_out, d_post,
                ctx->dec.ray_cond_norm_w, ctx->dec.ray_cond_norm_b,
                C_img, N, SB_DEC_LN_EPS };
        unsigned bx = 256;
        unsigned gx = (unsigned)((N + bx - 1) / bx);
        if (sb_launch(ctx->fn_ln_chw, gx, 1, 1, bx, 1, 1, 0,
                      &a, sizeof(a)) < 0) goto fail;
    }

    if (hipMemcpy(out_chw, d_out, post_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess) goto fail;

    hipFree(d_pre); hipFree(d_post); hipFree(d_rays); hipFree(d_out);
    return CUDA_SAM3D_BODY_E_OK;

fail:
    if (d_pre)  hipFree(d_pre);
    if (d_post) hipFree(d_post);
    if (d_rays) hipFree(d_rays);
    if (d_out)  hipFree(d_out);
    return CUDA_SAM3D_BODY_E_LOAD;
}

static int cuda_sam3d_body_ray_cond_from_encoder_dev(
        cuda_sam3d_body_ctx *ctx, const void *d_encoder_tokens,
        int n_prefix, const float *rays_hwc, int H, int W,
        void *d_out_chw)
{
    if (!ctx || !d_encoder_tokens || !rays_hwc || !d_out_chw ||
        H <= 0 || W <= 0 || !ctx->dec.loaded || !ctx->dec.no_mask_embed)
        return CUDA_SAM3D_BODY_E_INVAL;

    const int C_img = SB_DEC_KV_DIM;
    const int num_bands = 16;
    const int C_fp = 3 + 6 * num_bands;
    const int C_pre = C_img + C_fp;
    const int N = H * W;
    void *d_pre = NULL, *d_post = NULL, *d_rays = NULL;
    size_t pre_bytes = (size_t)C_pre * N * sizeof(float);
    size_t post_bytes = (size_t)C_img * N * sizeof(float);
    size_t rays_bytes = (size_t)N * 3 * sizeof(float);

    if (hipMalloc(&d_pre, pre_bytes) != hipSuccess) goto fail;
    if (hipMalloc(&d_post, post_bytes) != hipSuccess) goto fail;
    if (hipMalloc(&d_rays, rays_bytes) != hipSuccess) goto fail;
    if (hipMemcpy(d_rays, rays_hwc, rays_bytes,
                  hipMemcpyHostToDevice) != hipSuccess) goto fail;

    {
        struct __attribute__((packed)) {
            void *out; const void *tokens; const void *no_mask;
            int n_prefix; int N; int D;
        } p = {
            d_pre, d_encoder_tokens, ctx->dec.no_mask_embed,
            n_prefix, N, C_img
        };
        unsigned bx = 256;
        unsigned gx = (unsigned)(((size_t)N * C_img + bx - 1) / bx);
        if (sb_launch(ctx->fn_encoder_to_preconv, gx, 1, 1, bx, 1, 1, 0,
                      &p, sizeof(p)) < 0) goto fail;
    }
    {
        float *fp_chw = (float *)d_pre + (size_t)C_img * N;
        struct __attribute__((packed)) {
            void *fp_chw; void *rays; int N; int nb;
        } p = { fp_chw, d_rays, N, num_bands };
        unsigned bx = 256;
        unsigned gx = (unsigned)((N + bx - 1) / bx);
        if (sb_launch(ctx->fn_ray_cond_fourier, gx, 1, 1, bx, 1, 1, 0,
                      &p, sizeof(p)) < 0) goto fail;
    }
    {
        struct __attribute__((packed)) {
            void *out; void *W; void *src; int Co; int Ci; int N;
        } p = { d_post, ctx->dec.ray_cond_conv_w, d_pre,
                C_img, C_pre, N };
        unsigned bx = 256;
        unsigned gx = (unsigned)((N + bx - 1) / bx);
        unsigned gy = (unsigned)C_img;
        if (sb_launch(ctx->fn_conv1x1_chw, gx, gy, 1, bx, 1, 1, 0,
                      &p, sizeof(p)) < 0) goto fail;
    }
    {
        struct __attribute__((packed)) {
            void *dst; void *src; void *g; void *b;
            int C; int N; float eps;
        } p = { d_out_chw, d_post,
                ctx->dec.ray_cond_norm_w, ctx->dec.ray_cond_norm_b,
                C_img, N, SB_DEC_LN_EPS };
        unsigned bx = 256;
        unsigned gx = (unsigned)((N + bx - 1) / bx);
        if (sb_launch(ctx->fn_ln_chw, gx, 1, 1, bx, 1, 1, 0,
                      &p, sizeof(p)) < 0) goto fail;
    }
    hipFree(d_pre);
    hipFree(d_post);
    hipFree(d_rays);
    return CUDA_SAM3D_BODY_E_OK;

fail:
    if (d_pre) hipFree(d_pre);
    if (d_post) hipFree(d_post);
    if (d_rays) hipFree(d_rays);
    return CUDA_SAM3D_BODY_E_LOAD;
}

static int cuda_sam3d_body_chw_to_tok_dev(
        cuda_sam3d_body_ctx *ctx, void *d_out_tok,
        const void *d_in_chw, int N, int D)
{
    if (!ctx || !d_out_tok || !d_in_chw || N <= 0 || D <= 0)
        return CUDA_SAM3D_BODY_E_INVAL;
    struct __attribute__((packed)) {
        void *out; const void *in; int N; int D;
    } p = { d_out_tok, d_in_chw, N, D };
    unsigned bx = 256;
    unsigned gx = (unsigned)(((size_t)N * D + bx - 1) / bx);
    return (sb_launch(ctx->fn_chw_to_tok, gx, 1, 1, bx, 1, 1, 0,
                      &p, sizeof(p)) < 0)
        ? CUDA_SAM3D_BODY_E_LOAD
        : CUDA_SAM3D_BODY_E_OK;
}

int cuda_sam3d_body_debug_run_build_tokens(cuda_sam3d_body_ctx *ctx,
                                           const float *init_in,
                                           const float *prev_in,
                                           const float *prompt_in,
                                           float *x_out,
                                           float *x_pe_out)
{
    if (!ctx || !init_in || !prev_in || !prompt_in || !x_out || !x_pe_out)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->dec.loaded) {
        fprintf(stderr, "sam3d_body: decoder not loaded\n");
        return CUDA_SAM3D_BODY_E_LOAD;
    }
    const int D       = SB_DEC_DIM;                          /* 1024 */
    const int NKP     = SB_DEC_KP;                           /* 70   */
    const int NHB     = SB_DEC_HAND_TOK;                     /* 2    */
    const int D_init  = SB_DEC_NPOSE + SB_DEC_NCAM + SB_DEC_COND;  /* 525 */
    const int D_prev  = SB_DEC_NPOSE + SB_DEC_NCAM;                /* 522 */
    const int D_prompt = SB_DEC_KV_DIM;                            /* 1280 */
    const int N_TOK   = 1 + 1 + 1 + NHB + NKP + NKP;         /* 145  */
    const size_t row_b = (size_t)D * sizeof(float);
    const size_t tok_b = (size_t)N_TOK * D * sizeof(float);

    void *d_init = NULL, *d_prev = NULL, *d_prompt = NULL;
    void *d_x = NULL, *d_xpe = NULL;
    if (hipMalloc(&d_init,   D_init  * sizeof(float)) != hipSuccess) goto fail;
    if (hipMalloc(&d_prev,   D_prev  * sizeof(float)) != hipSuccess) goto fail;
    if (hipMalloc(&d_prompt, D_prompt * sizeof(float)) != hipSuccess) goto fail;
    if (hipMalloc(&d_x,   tok_b) != hipSuccess) goto fail;
    if (hipMalloc(&d_xpe, tok_b) != hipSuccess) goto fail;

    if (hipMemcpy(d_init,   init_in,   D_init  * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_prev,   prev_in,   D_prev  * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_prompt, prompt_in, D_prompt * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) goto fail;

    /* Helper macro: launch linear_f32_bias for a single output row. */
    #define LINEAR(p_out, p_w, p_bias, p_x, p_do, p_di) do { \
        struct __attribute__((packed)) { \
            void *o; void *w; void *xx; void *bb; int do_; int di_; \
        } a = { (p_out), (p_w), (p_x), (p_bias), (p_do), (p_di) }; \
        unsigned bx = 256; \
        unsigned gx = (unsigned)(((p_do) + bx - 1) / bx); \
        if (sb_launch(ctx->fn_linear_bias, gx, 1, 1, bx, 1, 1, 0, \
                      &a, sizeof(a)) < 0) goto fail; \
    } while (0)

    /* Slot 0 = init_to_token(init_in). */
    LINEAR((char *)d_x + 0 * row_b,
           ctx->dec.init_to_token_w, ctx->dec.init_to_token_b,
           d_init, D, D_init);
    /* Slot 1 = prev_to_token(prev_in). */
    LINEAR((char *)d_x + 1 * row_b,
           ctx->dec.prev_to_token_w, ctx->dec.prev_to_token_b,
           d_prev, D, D_prev);
    /* Slot 2 = prompt_to_token(prompt_in). */
    LINEAR((char *)d_x + 2 * row_b,
           ctx->dec.prompt_to_token_w, ctx->dec.prompt_to_token_b,
           d_prompt, D, D_prompt);
    #undef LINEAR

    /* Slots 3..4 = hand_box_embedding (2, D). */
    if (hipMemcpy((char *)d_x + 3 * row_b,
                  ctx->dec.hand_box_embedding,
                  (size_t)NHB * row_b,
                  hipMemcpyDeviceToDevice) != hipSuccess) goto fail;
    /* Slots 5..74 = keypoint_embedding (70, D). */
    if (hipMemcpy((char *)d_x + (3 + NHB) * row_b,
                  ctx->dec.keypoint_embedding,
                  (size_t)NKP * row_b,
                  hipMemcpyDeviceToDevice) != hipSuccess) goto fail;
    /* Slots 75..144 = keypoint3d_embedding (70, D). */
    if (hipMemcpy((char *)d_x + (3 + NHB + NKP) * row_b,
                  ctx->dec.keypoint3d_embedding,
                  (size_t)NKP * row_b,
                  hipMemcpyDeviceToDevice) != hipSuccess) goto fail;

    /* x_pe: zero everywhere except slots 1 and 2 (copies of slots 1/2 in x). */
    if (hipMemset(d_xpe, 0, tok_b) != hipSuccess) goto fail;
    if (hipMemcpy((char *)d_xpe + 1 * row_b,
                  (char *)d_x + 1 * row_b, row_b,
                  hipMemcpyDeviceToDevice) != hipSuccess) goto fail;
    if (hipMemcpy((char *)d_xpe + 2 * row_b,
                  (char *)d_x + 2 * row_b, row_b,
                  hipMemcpyDeviceToDevice) != hipSuccess) goto fail;

    if (hipMemcpy(x_out, d_x, tok_b,
                  hipMemcpyDeviceToHost) != hipSuccess) goto fail;
    if (hipMemcpy(x_pe_out, d_xpe, tok_b,
                  hipMemcpyDeviceToHost) != hipSuccess) goto fail;

    hipFree(d_init); hipFree(d_prev); hipFree(d_prompt);
    hipFree(d_x); hipFree(d_xpe);
    return CUDA_SAM3D_BODY_E_OK;

fail:
    if (d_init)   hipFree(d_init);
    if (d_prev)   hipFree(d_prev);
    if (d_prompt) hipFree(d_prompt);
    if (d_x)      hipFree(d_x);
    if (d_xpe)    hipFree(d_xpe);
    return CUDA_SAM3D_BODY_E_LOAD;
}

/* ---- Decoder-layer helpers (host-side launchers) ---- */

static int sb_dec_ln(cuda_sam3d_body_ctx *c, void *dst, const void *src,
                     const void *w, const void *b, int N, int D, float eps)
{
    struct __attribute__((packed)) {
        void *dst; const void *src; const void *w; const void *b;
        int dim; float eps;
    } p = { dst, src, w, b, D, eps };
    return sb_launch(c->fn_ln, (unsigned)N, 1, 1, 256, 1, 1,
                     256 * 4, &p, sizeof(p));
}

static int sb_dec_gemm(cuda_sam3d_body_ctx *c, void *Y, const void *X,
                       const void *W, const void *b,
                       int N, int D_in, int D_out)
{
    struct __attribute__((packed)) {
        void *Y; const void *X; const void *W; const void *b;
        int N; int D_in; int D_out;
    } p = { Y, X, W, b, N, D_in, D_out };
    unsigned bx = 16, by = 16;
    unsigned gx = (unsigned)((N + bx - 1) / bx);
    unsigned gy = (unsigned)((D_out + by - 1) / by);
    return sb_launch(c->fn_gemm_f32, gx, gy, 1, bx, by, 1, 0, &p, sizeof(p));
}

static int sb_dec_add_two(cuda_sam3d_body_ctx *c, void *out, const void *a,
                          const void *b, int n)
{
    struct __attribute__((packed)) {
        void *out; const void *a; const void *b; int n;
    } p = { out, a, b, n };
    unsigned bx = 256;
    unsigned gx = (unsigned)((n + bx - 1) / bx);
    return sb_launch(c->fn_add_two, gx, 1, 1, bx, 1, 1, 0, &p, sizeof(p));
}

static int sb_dec_add_inplace(cuda_sam3d_body_ctx *c, void *a, const void *b, int n)
{
    struct __attribute__((packed)) {
        void *a; const void *b; int n;
    } p = { a, b, n };
    unsigned bx = 256;
    unsigned gx = (unsigned)((n + bx - 1) / bx);
    return sb_launch(c->fn_add_inplace, gx, 1, 1, bx, 1, 1, 0, &p, sizeof(p));
}

static int sb_dec_gelu(cuda_sam3d_body_ctx *c, void *x, int n)
{
    struct __attribute__((packed)) { void *x; int n; } p = { x, n };
    unsigned bx = 256;
    unsigned gx = (unsigned)((n + bx - 1) / bx);
    return sb_launch(c->fn_gelu_inplace, gx, 1, 1, bx, 1, 1, 0, &p, sizeof(p));
}

static int sb_dec_sdpa(cuda_sam3d_body_ctx *c, void *out, const void *q,
                       const void *k, const void *v,
                       int N_q, int N_k, int H, int D_h)
{
    float scale = 1.0f / sqrtf((float)D_h);
    struct __attribute__((packed)) {
        void *out; const void *q; const void *k; const void *v;
        int N_q; int N_k; int H; int D_h; float scale;
    } p = { out, q, k, v, N_q, N_k, H, D_h, scale };
    unsigned shmem = (256 + (unsigned)N_k) * 4u;
    return sb_launch(c->fn_sdpa, (unsigned)N_q, (unsigned)H, 1, 256, 1, 1,
                     shmem, &p, sizeof(p));
}

static int cuda_sam3d_body_run_decoder_layer_dev(
        cuda_sam3d_body_ctx *ctx, int layer_idx,
        void *d_x, const void *d_ctx, void *d_xpe, const void *d_cpe,
        int N_q, int N_c, void *d_out,
        sb_dec_layer_scratch_dev *S)
{
    if (!ctx || !d_x || !d_ctx || !d_xpe || !d_cpe || !d_out || !S)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->dec.loaded || layer_idx < 0 || layer_idx >= SB_DEC_LAYERS)
        return CUDA_SAM3D_BODY_E_INVAL;

    const int D    = SB_DEC_DIM;
    const int Dc   = SB_DEC_KV_DIM;
    const int H    = SB_DEC_HEADS;
    const int Dh   = SB_DEC_HEAD_DIM;
    const int E    = H * Dh;
    const int F    = SB_DEC_FFN;
    const float eps = SB_DEC_LN_EPS;
    const int skip_first_pe = (layer_idx == 0);
    sb_dec_layer_w *L = &ctx->dec.layers[layer_idx];

    if (sb_dec_ln(ctx, S->xpe_n, d_xpe, L->ln_pe_1_w, L->ln_pe_1_b,
                  N_q, D, eps) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_ln(ctx, S->cpe_n, d_cpe, L->ln_pe_2_w, L->ln_pe_2_b,
                  N_c, Dc, eps) < 0) return CUDA_SAM3D_BODY_E_LOAD;

    if (sb_dec_ln(ctx, S->x_ln, d_x, L->ln1_w, L->ln1_b,
                  N_q, D, eps) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (skip_first_pe) {
        if (hipMemcpy(S->qk, S->x_ln, (size_t)N_q * D * sizeof(float),
                      hipMemcpyDeviceToDevice) != hipSuccess)
            return CUDA_SAM3D_BODY_E_LOAD;
    } else {
        if (sb_dec_add_two(ctx, S->qk, S->x_ln, S->xpe_n, N_q * D) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
    }
    if (sb_dec_gemm(ctx, S->Q, S->qk, L->sa_q_w, L->sa_q_b,
                    N_q, D, E) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->K, S->qk, L->sa_k_w, L->sa_k_b,
                    N_q, D, E) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->V, S->x_ln, L->sa_v_w, L->sa_v_b,
                    N_q, D, E) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_sdpa(ctx, S->attn, S->Q, S->K, S->V,
                    N_q, N_q, H, Dh) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->proj, S->attn, L->sa_proj_w, L->sa_proj_b,
                    N_q, E, D) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_add_two(ctx, S->x1, d_x, S->proj, N_q * D) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;

    if (sb_dec_ln(ctx, S->q_in, S->x1, L->ln2_1_w, L->ln2_1_b,
                  N_q, D, eps) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_add_inplace(ctx, S->q_in, S->xpe_n, N_q * D) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_ln(ctx, S->ctx_ln, d_ctx, L->ln2_2_w, L->ln2_2_b,
                  N_c, Dc, eps) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_add_two(ctx, S->k_in, S->ctx_ln, S->cpe_n, N_c * Dc) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->Q, S->q_in, L->ca_q_w, L->ca_q_b,
                    N_q, D, E) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->K, S->k_in, L->ca_k_w, L->ca_k_b,
                    N_c, Dc, E) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->V, S->ctx_ln, L->ca_v_w, L->ca_v_b,
                    N_c, Dc, E) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_sdpa(ctx, S->attn, S->Q, S->K, S->V,
                    N_q, N_c, H, Dh) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->proj, S->attn, L->ca_proj_w, L->ca_proj_b,
                    N_q, E, D) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_add_two(ctx, S->x2, S->x1, S->proj, N_q * D) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;

    if (sb_dec_ln(ctx, S->x_ln3, S->x2, L->ln3_w, L->ln3_b,
                  N_q, D, eps) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->ffn_h, S->x_ln3, L->ffn0_w, L->ffn0_b,
                    N_q, D, F) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gelu(ctx, S->ffn_h, N_q * F) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->ffn_o, S->ffn_h, L->ffn1_w, L->ffn1_b,
                    N_q, F, D) < 0) return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_add_two(ctx, d_out, S->x2, S->ffn_o, N_q * D) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;

    return CUDA_SAM3D_BODY_E_OK;
}

int cuda_sam3d_body_debug_run_decoder_layer(cuda_sam3d_body_ctx *ctx,
                                            int layer_idx,
                                            const float *x_in,
                                            const float *context_in,
                                            const float *x_pe_in,
                                            const float *context_pe_in,
                                            int N_q, int N_c,
                                            float *x_out)
{
    if (!ctx || !x_in || !context_in || !x_pe_in || !context_pe_in || !x_out)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->dec.loaded || layer_idx < 0 || layer_idx >= SB_DEC_LAYERS)
        return CUDA_SAM3D_BODY_E_INVAL;

    const int D    = SB_DEC_DIM;     /* 1024 */
    const int Dc   = SB_DEC_KV_DIM;  /* 1280 */
    const int H    = SB_DEC_HEADS;   /* 8 */
    const int Dh   = SB_DEC_HEAD_DIM;/* 64 */
    const int E    = H * Dh;         /* 512 */
    const int F    = SB_DEC_FFN;     /* 1024 */
    const float eps = SB_DEC_LN_EPS;
    const int skip_first_pe = (layer_idx == 0);

    sb_dec_layer_w *L = &ctx->dec.layers[layer_idx];

    /* Device buffers. */
    void *d_x = NULL, *d_ctx = NULL, *d_xpe = NULL, *d_cpe = NULL;
    void *d_xpe_n = NULL, *d_cpe_n = NULL;
    void *d_x_ln = NULL, *d_qk = NULL;
    void *d_Q = NULL, *d_K = NULL, *d_V = NULL, *d_attn = NULL, *d_proj = NULL;
    void *d_x1 = NULL, *d_q_in = NULL, *d_k_in = NULL, *d_ctx_ln = NULL;
    void *d_x2 = NULL, *d_x_ln3 = NULL, *d_ffn_h = NULL, *d_ffn_o = NULL;

    size_t bx_q = (size_t)N_q * D * sizeof(float);
    size_t bx_c = (size_t)N_c * Dc * sizeof(float);
    size_t b_qE = (size_t)N_q * E * sizeof(float);
    size_t b_qF = (size_t)N_q * F * sizeof(float);

    if (hipMalloc(&d_x,    bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_ctx,  bx_c) != hipSuccess) goto fail;
    if (hipMalloc(&d_xpe,  bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_cpe,  bx_c) != hipSuccess) goto fail;
    if (hipMalloc(&d_xpe_n, bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_cpe_n, bx_c) != hipSuccess) goto fail;
    if (hipMalloc(&d_x_ln, bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_qk,   bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_Q,    b_qE) != hipSuccess) goto fail;
    if (hipMalloc(&d_K,    (size_t)((N_q > N_c ? N_q : N_c)) * E * sizeof(float)) != hipSuccess) goto fail;
    if (hipMalloc(&d_V,    (size_t)((N_q > N_c ? N_q : N_c)) * E * sizeof(float)) != hipSuccess) goto fail;
    if (hipMalloc(&d_attn, b_qE) != hipSuccess) goto fail;
    if (hipMalloc(&d_proj, bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_x1,   bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_q_in, bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_k_in, bx_c) != hipSuccess) goto fail;
    if (hipMalloc(&d_ctx_ln, bx_c) != hipSuccess) goto fail;
    if (hipMalloc(&d_x2,    bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_x_ln3, bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_ffn_h, b_qF) != hipSuccess) goto fail;
    if (hipMalloc(&d_ffn_o, bx_q) != hipSuccess) goto fail;

    if (hipMemcpy(d_x,   x_in,   bx_q, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_ctx, context_in, bx_c, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_xpe, x_pe_in, bx_q, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_cpe, context_pe_in, bx_c, hipMemcpyHostToDevice) != hipSuccess) goto fail;

    /* Pre-normalize PEs. */
    if (sb_dec_ln(ctx, d_xpe_n, d_xpe, L->ln_pe_1_w, L->ln_pe_1_b, N_q, D,  eps) < 0) goto fail;
    if (sb_dec_ln(ctx, d_cpe_n, d_cpe, L->ln_pe_2_w, L->ln_pe_2_b, N_c, Dc, eps) < 0) goto fail;

    /* ===== Self-attention ===== */
    if (sb_dec_ln(ctx, d_x_ln, d_x, L->ln1_w, L->ln1_b, N_q, D, eps) < 0) goto fail;

    if (skip_first_pe) {
        if (hipMemcpy(d_qk, d_x_ln, bx_q, hipMemcpyDeviceToDevice) != hipSuccess) goto fail;
    } else {
        if (sb_dec_add_two(ctx, d_qk, d_x_ln, d_xpe_n, (int)(N_q * D)) < 0) goto fail;
    }
    /* Q = qk @ Wq^T + bq, K = qk @ Wk^T + bk, V = x_ln @ Wv^T + bv */
    if (sb_dec_gemm(ctx, d_Q, d_qk, L->sa_q_w, L->sa_q_b, N_q, D, E) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_K, d_qk, L->sa_k_w, L->sa_k_b, N_q, D, E) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_V, d_x_ln, L->sa_v_w, L->sa_v_b, N_q, D, E) < 0) goto fail;
    if (sb_dec_sdpa(ctx, d_attn, d_Q, d_K, d_V, N_q, N_q, H, Dh) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_proj, d_attn, L->sa_proj_w, L->sa_proj_b, N_q, E, D) < 0) goto fail;

    /* x1 = x + sa_proj */
    if (sb_dec_add_two(ctx, d_x1, d_x, d_proj, (int)(N_q * D)) < 0) goto fail;

    /* ===== Cross-attention =====
     * q = ln2_1(x1) + x_pe_n
     * ctx_ln = ln2_2(context)
     * k = ctx_ln + cpe_n; v = ctx_ln
     */
    if (sb_dec_ln(ctx, d_q_in, d_x1, L->ln2_1_w, L->ln2_1_b, N_q, D, eps) < 0) goto fail;
    if (sb_dec_add_inplace(ctx, d_q_in, d_xpe_n, (int)(N_q * D)) < 0) goto fail;
    if (sb_dec_ln(ctx, d_ctx_ln, d_ctx, L->ln2_2_w, L->ln2_2_b, N_c, Dc, eps) < 0) goto fail;
    if (sb_dec_add_two(ctx, d_k_in, d_ctx_ln, d_cpe_n, (int)(N_c * Dc)) < 0) goto fail;

    if (sb_dec_gemm(ctx, d_Q, d_q_in,   L->ca_q_w, L->ca_q_b, N_q, D,  E) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_K, d_k_in,   L->ca_k_w, L->ca_k_b, N_c, Dc, E) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_V, d_ctx_ln, L->ca_v_w, L->ca_v_b, N_c, Dc, E) < 0) goto fail;
    if (sb_dec_sdpa(ctx, d_attn, d_Q, d_K, d_V, N_q, N_c, H, Dh) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_proj, d_attn, L->ca_proj_w, L->ca_proj_b, N_q, E, D) < 0) goto fail;

    /* x2 = x1 + ca_proj */
    if (sb_dec_add_two(ctx, d_x2, d_x1, d_proj, (int)(N_q * D)) < 0) goto fail;

    /* ===== FFN =====
     * h = GELU(ffn0(ln3(x2))); out = ffn1(h); x_out = x2 + out
     */
    if (sb_dec_ln(ctx, d_x_ln3, d_x2, L->ln3_w, L->ln3_b, N_q, D, eps) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_ffn_h, d_x_ln3, L->ffn0_w, L->ffn0_b, N_q, D, F) < 0) goto fail;
    if (sb_dec_gelu(ctx, d_ffn_h, (int)(N_q * F)) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_ffn_o, d_ffn_h, L->ffn1_w, L->ffn1_b, N_q, F, D) < 0) goto fail;
    if (sb_dec_add_two(ctx, d_x1, d_x2, d_ffn_o, (int)(N_q * D)) < 0) goto fail;
        /* reuse d_x1 as final out scratch */

    if (hipMemcpy(x_out, d_x1, bx_q, hipMemcpyDeviceToHost) != hipSuccess) goto fail;

    hipFree(d_x); hipFree(d_ctx); hipFree(d_xpe); hipFree(d_cpe);
    hipFree(d_xpe_n); hipFree(d_cpe_n);
    hipFree(d_x_ln); hipFree(d_qk);
    hipFree(d_Q); hipFree(d_K); hipFree(d_V);
    hipFree(d_attn); hipFree(d_proj);
    hipFree(d_x1); hipFree(d_q_in); hipFree(d_k_in); hipFree(d_ctx_ln);
    hipFree(d_x2); hipFree(d_x_ln3); hipFree(d_ffn_h); hipFree(d_ffn_o);
    return CUDA_SAM3D_BODY_E_OK;

fail:
    if (d_x)    hipFree(d_x);
    if (d_ctx)  hipFree(d_ctx);
    if (d_xpe)  hipFree(d_xpe);
    if (d_cpe)  hipFree(d_cpe);
    if (d_xpe_n) hipFree(d_xpe_n);
    if (d_cpe_n) hipFree(d_cpe_n);
    if (d_x_ln) hipFree(d_x_ln);
    if (d_qk)   hipFree(d_qk);
    if (d_Q)    hipFree(d_Q);
    if (d_K)    hipFree(d_K);
    if (d_V)    hipFree(d_V);
    if (d_attn) hipFree(d_attn);
    if (d_proj) hipFree(d_proj);
    if (d_x1)   hipFree(d_x1);
    if (d_q_in) hipFree(d_q_in);
    if (d_k_in) hipFree(d_k_in);
    if (d_ctx_ln) hipFree(d_ctx_ln);
    if (d_x2)   hipFree(d_x2);
    if (d_x_ln3) hipFree(d_x_ln3);
    if (d_ffn_h) hipFree(d_ffn_h);
    if (d_ffn_o) hipFree(d_ffn_o);
    return CUDA_SAM3D_BODY_E_LOAD;
}

/* ===== kp_token_update (Step 5e) ============================================
 *
 * Mirrors sam3d_body_kp_token_update in common/sam3d_body_decoder.h:
 *   - 2D path: invalid_mask (xy out of [0,1] OR depth<1e-5) → kp_posemb
 *     Linear(2→1024) → ReLU → Linear(1024→1024), zero on invalid →
 *     overwrite augment[5..75). grid_sample(image_emb, kp2d*2, zeros pad)
 *     → kp_feat Linear(1280→1024) → ADD into tokens[5..75).
 *   - 3D path: pelvis-norm subtract avg(joints[9],joints[10]) → kp3d_posemb
 *     Linear(3→1024) → ReLU → Linear(1024→1024) → overwrite augment[75..145).
 * Layer n_layers-1 is short-circuited (matches upstream guard).
 */
static int sb_dec_relu(cuda_sam3d_body_ctx *c, void *x, int n)
{
    struct __attribute__((packed)) { void *x; int n; } p = { x, n };
    unsigned bx = 256;
    unsigned gx = (unsigned)((n + bx - 1) / bx);
    return sb_launch(c->fn_relu_inplace, gx, 1, 1, bx, 1, 1, 0, &p, sizeof(p));
}

static int sb_dec_grid_sample(cuda_sam3d_body_ctx *c, void *out, const void *src,
                              const void *gxy, const void *invalid,
                              int K, int C, int H, int W)
{
    struct __attribute__((packed)) {
        void *out; const void *src; const void *gxy; const void *invalid;
        int K; int C; int H; int W;
    } p = { out, src, gxy, invalid, K, C, H, W };
    unsigned bx = 256;
    unsigned gx = (unsigned)K;
    unsigned gy = (unsigned)((C + bx - 1) / bx);
    return sb_launch(c->fn_grid_sample, gx, gy, 1, bx, 1, 1, 0, &p, sizeof(p));
}

static int sb_dec_pelvis_norm(cuda_sam3d_body_ctx *c, void *out,
                              const void *kp3d, int K)
{
    struct __attribute__((packed)) {
        void *out; const void *kp3d; int K;
    } p = { out, kp3d, K };
    unsigned bx = 64;
    unsigned gx = (unsigned)((K * 3 + bx - 1) / bx);
    return sb_launch(c->fn_kp_pelvis_norm, gx, 1, 1, bx, 1, 1, 0, &p, sizeof(p));
}

static int sb_dec_aug_overwrite(cuda_sam3d_body_ctx *c, void *augment,
                                int start_row, const void *posemb,
                                const void *invalid, int K, int D)
{
    /* Pointer args first, scalar args last — matches the kernel signature
     * and avoids the natural-alignment padding hole that __attribute__((packed))
     * would otherwise miscompute when ints are interleaved with pointers. */
    struct __attribute__((packed)) {
        void *augment; const void *posemb; const void *invalid;
        int start_row; int K; int D;
    } p = { augment, posemb, invalid, start_row, K, D };
    unsigned bx = 256;
    unsigned gx = (unsigned)K;
    unsigned gy = (unsigned)((D + bx - 1) / bx);
    return sb_launch(c->fn_augment_overwrite_mask, gx, gy, 1, bx, 1, 1,
                     0, &p, sizeof(p));
}

static int cuda_sam3d_body_kp_token_update_dev(
        cuda_sam3d_body_ctx *ctx, int layer_idx, int H, int W, int N_q,
        const float *kp2d_cropped_host, const float *kp2d_depth_host,
        const void *d_img_chw, void *d_tokens, void *d_aug,
        sb_kpu_scratch_dev *S)
{
    if (!ctx || !kp2d_cropped_host || !kp2d_depth_host ||
        !d_img_chw || !d_tokens || !d_aug || !S ||
        !ctx->d_mhr_kp2d_crop || !ctx->d_mhr_kp2d_depth || !ctx->d_mhr_kp3d)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->dec.loaded || layer_idx < 0 || layer_idx >= SB_DEC_LAYERS)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (layer_idx == SB_DEC_LAYERS - 1) return CUDA_SAM3D_BODY_E_OK;

    const int D = SB_DEC_DIM;
    const int C = SB_DEC_KV_DIM;
    const int K = SB_DEC_KP;
    const int kp2d_start = 1 + 1 + 1 + SB_DEC_HAND_TOK;
    const int kp3d_start = kp2d_start + K;
    if (kp3d_start + K > N_q) return CUDA_SAM3D_BODY_E_INVAL;

    int invalid_h[70];
    float gxy_h[70 * 2];
    const int scale_x = (ctx->cfg.backbone == CUDA_SAM3D_BODY_BACKBONE_VITH);
    for (int i = 0; i < K; i++) {
        float x01 = kp2d_cropped_host[i*2 + 0] + 0.5f;
        float y01 = kp2d_cropped_host[i*2 + 1] + 0.5f;
        invalid_h[i] = (x01 < 0.0f) || (x01 > 1.0f) ||
                       (y01 < 0.0f) || (y01 > 1.0f) ||
                       (kp2d_depth_host[i] < 1e-5f);
        float gx = kp2d_cropped_host[i*2 + 0] * 2.0f;
        if (scale_x && W != H) gx *= (float)H / (float)W;
        gxy_h[i*2 + 0] = gx;
        gxy_h[i*2 + 1] = kp2d_cropped_host[i*2 + 1] * 2.0f;
    }

    size_t b_kp2 = (size_t)K * 2 * sizeof(float);
    size_t b_inv = (size_t)K * sizeof(int);

    if (hipMemcpy(S->gxy, gxy_h, b_kp2, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(S->invalid, invalid_h, b_inv, hipMemcpyHostToDevice) != hipSuccess) goto fail;

    if (sb_dec_gemm(ctx, S->p_h, ctx->d_mhr_kp2d_crop,
                    ctx->dec.kp_posemb_l0_w, ctx->dec.kp_posemb_l0_b,
                    K, 2, D) < 0) goto fail;
    if (sb_dec_relu(ctx, S->p_h, K * D) < 0) goto fail;
    if (sb_dec_gemm(ctx, S->p_o, S->p_h,
                    ctx->dec.kp_posemb_l1_w, ctx->dec.kp_posemb_l1_b,
                    K, D, D) < 0) goto fail;
    if (sb_dec_aug_overwrite(ctx, d_aug, kp2d_start, S->p_o, S->invalid,
                             K, D) < 0) goto fail;
    if (sb_dec_grid_sample(ctx, S->kp_feats, d_img_chw, S->gxy, S->invalid,
                           K, C, H, W) < 0) goto fail;
    if (sb_dec_gemm(ctx, S->kp_proj, S->kp_feats,
                    ctx->dec.kp_feat_linear_w, ctx->dec.kp_feat_linear_b,
                    K, C, D) < 0) goto fail;
    {
        float *d_tokens_kp = (float *)d_tokens + (size_t)kp2d_start * D;
        if (sb_dec_add_inplace(ctx, d_tokens_kp, S->kp_proj, K * D) < 0) goto fail;
    }

    if (sb_dec_pelvis_norm(ctx, S->kp3d_norm, ctx->d_mhr_kp3d, K) < 0) goto fail;
    if (sb_dec_gemm(ctx, S->p3_h, S->kp3d_norm,
                    ctx->dec.kp3d_posemb_l0_w, ctx->dec.kp3d_posemb_l0_b,
                    K, 3, D) < 0) goto fail;
    if (sb_dec_relu(ctx, S->p3_h, K * D) < 0) goto fail;
    if (sb_dec_gemm(ctx, S->p3_o, S->p3_h,
                    ctx->dec.kp3d_posemb_l1_w, ctx->dec.kp3d_posemb_l1_b,
                    K, D, D) < 0) goto fail;
    if (sb_dec_aug_overwrite(ctx, d_aug, kp3d_start, S->p3_o, NULL,
                             K, D) < 0) goto fail;
    return CUDA_SAM3D_BODY_E_OK;

fail:
    return CUDA_SAM3D_BODY_E_LOAD;
}

static int cuda_sam3d_body_norm_heads_dev(
        cuda_sam3d_body_ctx *ctx, const void *d_tokens,
        float *pose_raw_host, float *cam_raw_host,
        sb_head_scratch_dev *S)
{
    if (!ctx || !d_tokens || !pose_raw_host || !cam_raw_host || !S ||
        !S->norm_row || !S->pose_h || !S->pose_raw || !S->cam_h || !S->cam_raw)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->dec.loaded) return CUDA_SAM3D_BODY_E_INVAL;

    const int D = SB_DEC_DIM;
    const int Np = SB_DEC_NPOSE;
    const int Nc = SB_DEC_NCAM;
    if (sb_dec_ln(ctx, S->norm_row, d_tokens,
                  ctx->dec.norm_final_w, ctx->dec.norm_final_b,
                  1, D, 1e-6f) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->pose_h, S->norm_row,
                    ctx->dec.head_pose_l0_w, ctx->dec.head_pose_l0_b,
                    1, D, D) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_relu(ctx, S->pose_h, D) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->pose_raw, S->pose_h,
                    ctx->dec.head_pose_l1_w, ctx->dec.head_pose_l1_b,
                    1, D, Np) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;

    if (sb_dec_gemm(ctx, S->cam_h, S->norm_row,
                    ctx->dec.head_camera_l0_w, ctx->dec.head_camera_l0_b,
                    1, D, D) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_relu(ctx, S->cam_h, D) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (sb_dec_gemm(ctx, S->cam_raw, S->cam_h,
                    ctx->dec.head_camera_l1_w, ctx->dec.head_camera_l1_b,
                    1, D, Nc) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;

    if (hipMemcpy(pose_raw_host, S->pose_raw,
                  (size_t)Np * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (hipMemcpy(cam_raw_host, S->cam_raw,
                  (size_t)Nc * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    return CUDA_SAM3D_BODY_E_OK;
}

int cuda_sam3d_body_debug_run_kp_token_update(cuda_sam3d_body_ctx *ctx,
                                              int layer_idx,
                                              const float *image_emb_chw,
                                              int H, int W,
                                              const float *kp2d_cropped,
                                              const float *kp2d_depth,
                                              const float *kp3d_camera,
                                              int N_q,
                                              float *tokens,
                                              float *token_augment)
{
    if (!ctx || !image_emb_chw || !kp2d_cropped || !kp2d_depth || !kp3d_camera ||
        !tokens || !token_augment)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->dec.loaded || layer_idx < 0 || layer_idx >= SB_DEC_LAYERS)
        return CUDA_SAM3D_BODY_E_INVAL;

    /* Last-layer short-circuit (matches sam3d_body_kp_token_update). */
    if (layer_idx == SB_DEC_LAYERS - 1) return 0;

    const int D = SB_DEC_DIM;          /* 1024 */
    const int C = SB_DEC_KV_DIM;       /* 1280 */
    const int K = SB_DEC_KP;           /* 70 */
    const int kp2d_start = 1 + 1 + 1 + SB_DEC_HAND_TOK;          /* 5 */
    const int kp3d_start = kp2d_start + K;                        /* 75 */
    if (kp3d_start + K > N_q) return CUDA_SAM3D_BODY_E_INVAL;

    /* Host-side invalid mask (cheap; K=70). */
    int invalid_h[70];
    for (int i = 0; i < K; i++) {
        float x01 = kp2d_cropped[i*2 + 0] + 0.5f;
        float y01 = kp2d_cropped[i*2 + 1] + 0.5f;
        invalid_h[i] = (x01 < 0.0f) || (x01 > 1.0f) ||
                       (y01 < 0.0f) || (y01 > 1.0f) ||
                       (kp2d_depth[i] < 1e-5f);
    }
    /* Pre-multiply gxy by 2 (sample_points = kp2d_cropped * 2 -> [-1, 1]).
     * ViT-H width-crops a square crop before the backbone, so upstream scales
     * x by H/W for that backbone only. Rectangular DINOv3 does not. */
    float gxy_h[70 * 2];
    const int scale_x = (ctx->cfg.backbone == CUDA_SAM3D_BODY_BACKBONE_VITH);
    for (int i = 0; i < K; i++) {
        float gx = kp2d_cropped[i*2 + 0] * 2.0f;
        if (scale_x && W != H) gx *= (float)H / (float)W;
        gxy_h[i*2 + 0] = gx;
        gxy_h[i*2 + 1] = kp2d_cropped[i*2 + 1] * 2.0f;
    }

    /* Device buffers. */
    void *d_tokens = NULL, *d_aug = NULL, *d_img = NULL;
    void *d_gxy = NULL, *d_invalid = NULL;
    void *d_kp2d_in = NULL, *d_p_h = NULL, *d_p_o = NULL;
    void *d_kp_feats = NULL, *d_kp_proj = NULL;
    void *d_kp3d_in = NULL, *d_kp3d_norm = NULL;
    void *d_p3_h = NULL, *d_p3_o = NULL;

    size_t bx_q  = (size_t)N_q * D * sizeof(float);
    size_t b_img = (size_t)C * (size_t)H * (size_t)W * sizeof(float);
    size_t b_kpD = (size_t)K * D * sizeof(float);
    size_t b_kpC = (size_t)K * C * sizeof(float);
    size_t b_kp2 = (size_t)K * 2 * sizeof(float);
    size_t b_kp3 = (size_t)K * 3 * sizeof(float);
    size_t b_inv = (size_t)K * sizeof(int);

    int use_cached_img = ctx->kpu_use_cached_img &&
                         ctx->d_kpu_img &&
                         ctx->d_kpu_img_cap >= b_img;

    if (hipMalloc(&d_tokens,    bx_q) != hipSuccess) goto fail;
    if (hipMalloc(&d_aug,       bx_q) != hipSuccess) goto fail;
    if (use_cached_img) {
        d_img = ctx->d_kpu_img;
    } else if (hipMalloc(&d_img, b_img) != hipSuccess) goto fail;
    if (hipMalloc(&d_gxy,       b_kp2) != hipSuccess) goto fail;
    if (hipMalloc(&d_invalid,   b_inv) != hipSuccess) goto fail;
    if (hipMalloc(&d_kp2d_in,   b_kp2) != hipSuccess) goto fail;
    if (hipMalloc(&d_p_h,       b_kpD) != hipSuccess) goto fail;
    if (hipMalloc(&d_p_o,       b_kpD) != hipSuccess) goto fail;
    if (hipMalloc(&d_kp_feats,  b_kpC) != hipSuccess) goto fail;
    if (hipMalloc(&d_kp_proj,   b_kpD) != hipSuccess) goto fail;
    if (hipMalloc(&d_kp3d_in,   b_kp3) != hipSuccess) goto fail;
    if (hipMalloc(&d_kp3d_norm, b_kp3) != hipSuccess) goto fail;
    if (hipMalloc(&d_p3_h,      b_kpD) != hipSuccess) goto fail;
    if (hipMalloc(&d_p3_o,      b_kpD) != hipSuccess) goto fail;

    if (hipMemcpy(d_tokens, tokens,        bx_q, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_aug,    token_augment, bx_q, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (!use_cached_img &&
        hipMemcpy(d_img, image_emb_chw, b_img, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_gxy,    gxy_h,         b_kp2, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_invalid, invalid_h,    b_inv, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_kp2d_in, kp2d_cropped, b_kp2, hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_kp3d_in, kp3d_camera,  b_kp3, hipMemcpyHostToDevice) != hipSuccess) goto fail;

    /* ---- 2D path -------------------------------------------------- */
    /* posemb_h = kp2d_in @ kp_posemb_l0_w^T + kp_posemb_l0_b  (K, 2 → K, D) */
    if (sb_dec_gemm(ctx, d_p_h, d_kp2d_in,
                    ctx->dec.kp_posemb_l0_w, ctx->dec.kp_posemb_l0_b,
                    K, 2, D) < 0) goto fail;
    if (sb_dec_relu(ctx, d_p_h, K * D) < 0) goto fail;
    /* posemb_o = posemb_h @ kp_posemb_l1_w^T + kp_posemb_l1_b  (K, D → K, D) */
    if (sb_dec_gemm(ctx, d_p_o, d_p_h,
                    ctx->dec.kp_posemb_l1_w, ctx->dec.kp_posemb_l1_b,
                    K, D, D) < 0) goto fail;
    /* augment[5..75) = invalid ? 0 : posemb_o */
    if (sb_dec_aug_overwrite(ctx, d_aug, kp2d_start, d_p_o, d_invalid,
                             K, D) < 0) goto fail;

    /* grid_sample image_emb at kp2d*2 (zeros padding); zero for invalid kp. */
    if (sb_dec_grid_sample(ctx, d_kp_feats, d_img, d_gxy, d_invalid,
                           K, C, H, W) < 0) goto fail;
    /* kp_proj = kp_feats @ kp_feat_linear_w^T + b   (K, C → K, D) */
    if (sb_dec_gemm(ctx, d_kp_proj, d_kp_feats,
                    ctx->dec.kp_feat_linear_w, ctx->dec.kp_feat_linear_b,
                    K, C, D) < 0) goto fail;
    /* tokens[5..75) += kp_proj  (size = K*D contiguous starting at row 5) */
    {
        float *d_tokens_kp = (float *)d_tokens + (size_t)kp2d_start * D;
        if (sb_dec_add_inplace(ctx, d_tokens_kp, d_kp_proj, K * D) < 0) goto fail;
    }

    /* ---- 3D path -------------------------------------------------- */
    if (sb_dec_pelvis_norm(ctx, d_kp3d_norm, d_kp3d_in, K) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_p3_h, d_kp3d_norm,
                    ctx->dec.kp3d_posemb_l0_w, ctx->dec.kp3d_posemb_l0_b,
                    K, 3, D) < 0) goto fail;
    if (sb_dec_relu(ctx, d_p3_h, K * D) < 0) goto fail;
    if (sb_dec_gemm(ctx, d_p3_o, d_p3_h,
                    ctx->dec.kp3d_posemb_l1_w, ctx->dec.kp3d_posemb_l1_b,
                    K, D, D) < 0) goto fail;
    /* augment[75..145) = posemb_3d  (no mask) */
    if (sb_dec_aug_overwrite(ctx, d_aug, kp3d_start, d_p3_o, NULL,
                             K, D) < 0) goto fail;

    /* Copy back. */
    if (hipMemcpy(tokens,        d_tokens, bx_q, hipMemcpyDeviceToHost) != hipSuccess) goto fail;
    if (hipMemcpy(token_augment, d_aug,    bx_q, hipMemcpyDeviceToHost) != hipSuccess) goto fail;

    hipFree(d_tokens); hipFree(d_aug);
    if (!use_cached_img) hipFree(d_img);
    hipFree(d_gxy); hipFree(d_invalid); hipFree(d_kp2d_in);
    hipFree(d_p_h); hipFree(d_p_o);
    hipFree(d_kp_feats); hipFree(d_kp_proj);
    hipFree(d_kp3d_in); hipFree(d_kp3d_norm);
    hipFree(d_p3_h); hipFree(d_p3_o);
    return 0;

fail:
    if (d_tokens) hipFree(d_tokens);
    if (d_aug) hipFree(d_aug);
    if (d_img && !use_cached_img) hipFree(d_img);
    if (d_gxy) hipFree(d_gxy);
    if (d_invalid) hipFree(d_invalid);
    if (d_kp2d_in) hipFree(d_kp2d_in);
    if (d_p_h) hipFree(d_p_h);
    if (d_p_o) hipFree(d_p_o);
    if (d_kp_feats) hipFree(d_kp_feats);
    if (d_kp_proj) hipFree(d_kp_proj);
    if (d_kp3d_in) hipFree(d_kp3d_in);
    if (d_kp3d_norm) hipFree(d_kp3d_norm);
    if (d_p3_h) hipFree(d_p3_h);
    if (d_p3_o) hipFree(d_p3_o);
    return CUDA_SAM3D_BODY_E_LOAD;
}

/* CPU implementation. Tail of the decoder (norm_final + 2 head MLPs)
 * has trivial compute (~2.6 MFLOP per call) and the outputs head straight
 * back to the host for MHR — running it on CPU avoids one full
 * tokens upload + GPU GEMM chain + multi-buffer download per call.
 * Math is identical to the prior NVRTC kernels: PyTorch LayerNorm
 * (eps=1e-6, biased variance) + Linear(W,b) + ReLU + Linear(W,b) on
 * the row-0 ("pose token") slot. */
int cuda_sam3d_body_debug_run_norm_and_heads(cuda_sam3d_body_ctx *ctx,
                                             const float *tokens_in,
                                             int N_q,
                                             float *tokens_norm,
                                             float *pose_raw,
                                             float *cam_raw)
{
    if (!ctx || !tokens_in || !pose_raw || !cam_raw)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->dec.loaded) return CUDA_SAM3D_BODY_E_INVAL;

    const int D  = SB_DEC_DIM;       /* 1024 */
    const int Np = SB_DEC_NPOSE;     /* 519  */
    const int Nc = SB_DEC_NCAM;      /* 3    */
    const float eps = 1e-6f;

    const float *gamma = ctx->dec.norm_final_w_h;
    const float *beta  = ctx->dec.norm_final_b_h;

    /* When the caller doesn't want all tokens LN'd, we still need row 0
     * (the pose token) for the head MLPs. */
    float row0_scratch[SB_DEC_DIM];
    const int n_rows = tokens_norm ? N_q : 1;
    int n_threads = tokens_norm ? SAM3DB_MHR_THREADS() : 1;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads)
#else
    (void)n_threads;
#endif
    for (int t = 0; t < n_rows; t++) {
        const float *src = tokens_in + (size_t)t * D;
        float *dst = tokens_norm ? tokens_norm + (size_t)t * D
                                 : row0_scratch;
        float mean = 0.0f;
        for (int j = 0; j < D; j++) mean += src[j];
        mean /= (float)D;
        float var = 0.0f;
        for (int j = 0; j < D; j++) {
            float d = src[j] - mean;
            var += d * d;
        }
        var /= (float)D;
        float inv = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < D; j++) {
            dst[j] = (src[j] - mean) * inv * gamma[j] + beta[j];
        }
    }

    const float *pose_tok = tokens_norm ? tokens_norm : row0_scratch;

    /* head_pose: Linear(D->D) + ReLU + Linear(D->Np). */
    float h_buf[SB_DEC_DIM];
    cpu_gemm_f32(h_buf, ctx->dec.head_pose_l0_w_h,
                        ctx->dec.head_pose_l0_b_h,
                        pose_tok, /*n_tok=*/1, /*n_out=*/D, /*n_in=*/D,
                        n_threads);
    for (int j = 0; j < D; j++) if (h_buf[j] < 0.0f) h_buf[j] = 0.0f;
    cpu_gemm_f32(pose_raw, ctx->dec.head_pose_l1_w_h,
                           ctx->dec.head_pose_l1_b_h,
                           h_buf, 1, Np, D, n_threads);

    /* head_camera: Linear(D->D) + ReLU + Linear(D->Nc). */
    cpu_gemm_f32(h_buf, ctx->dec.head_camera_l0_w_h,
                        ctx->dec.head_camera_l0_b_h,
                        pose_tok, 1, D, D, n_threads);
    for (int j = 0; j < D; j++) if (h_buf[j] < 0.0f) h_buf[j] = 0.0f;
    cpu_gemm_f32(cam_raw, ctx->dec.head_camera_l1_w_h,
                          ctx->dec.head_camera_l1_b_h,
                          h_buf, 1, Nc, D, n_threads);

    return 0;
}

static void sb_mhr_batch6d_from_xyz(const float *e, float *out6)
{
    float cx = cosf(e[0]), sx = sinf(e[0]);
    float cy = cosf(e[1]), sy = sinf(e[1]);
    float cz = cosf(e[2]), sz = sinf(e[2]);
    out6[0] = cy * cz;
    out6[1] = cy * sz;
    out6[2] = -sy;
    out6[3] = -cx * sz + sx * sy * cz;
    out6[4] = cx * cz + sx * sy * sz;
    out6[5] = sx * cy;
}

static int cuda_sam3d_body_mhr_pose_correctives_cached(
        cuda_sam3d_body_ctx *ctx, const float *joint_params, float *out_verts)
{
    if (!ctx || !ctx->mhr_gpu_enabled || !ctx->cpu_mhr ||
        !joint_params)
        return CUDA_SAM3D_BODY_E_INVAL;

    const sam3d_body_mhr_assets *a = ctx->cpu_mhr;
    const int V_d = S3DM_N_VERTS * 3;
    const int HID = S3DM_N_PC_H;
    const int NNZ = S3DM_N_PC_NNZ;
    const int64_t *spi_row = (const int64_t *)a->pc_sparse_indices.data;
    const int64_t *spi_col = spi_row + NNZ;
    const float   *spw     = (const float *)a->pc_sparse_weight.data;

    float feat[S3DM_N_PC_IN];
    float h[S3DM_N_PC_H];

    for (int jj = 0; jj < 125; jj++) {
        const float *e = joint_params + (size_t)(jj + 2) * 7 + 3;
        sb_mhr_batch6d_from_xyz(e, feat + (size_t)jj * 6);
    }
    for (int jj = 0; jj < 125; jj++) {
        feat[(size_t)jj * 6 + 0] -= 1.0f;
        feat[(size_t)jj * 6 + 4] -= 1.0f;
    }

    memset(h, 0, (size_t)HID * sizeof(float));
    for (int k = 0; k < NNZ; k++) {
        int row = (int)spi_row[k];
        int col = (int)spi_col[k];
        h[row] += spw[k] * feat[col];
    }
    for (int i = 0; i < HID; i++) {
        if (h[i] < 0.0f) h[i] = 0.0f;
    }

    if (hipMemcpy(ctx->d_mhr_pc_h, h, (size_t)HID * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;

    struct __attribute__((packed)) {
        void *Y; const void *W; const void *X; int D_in; int D_out;
    } p = { ctx->d_mhr_pc_out, ctx->d_mhr_pc_linear_weight,
            ctx->d_mhr_pc_h, HID, V_d };
    unsigned bx = 256;
    unsigned gx = (unsigned)V_d;
    if (sb_launch(ctx->fn_mhr_matvec, gx, 1, 1, bx, 1, 1, 0,
                  &p, sizeof(p)) < 0)
        return CUDA_SAM3D_BODY_E_LOAD;

    if (out_verts) {
        if (hipMemcpy(out_verts, ctx->d_mhr_pc_out,
                      (size_t)V_d * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess)
            return CUDA_SAM3D_BODY_E_LOAD;
    }
    return CUDA_SAM3D_BODY_E_OK;
}

static int cuda_sam3d_body_mhr_blend_cached(
        cuda_sam3d_body_ctx *ctx,
        const float *coeffs,
        void *d_coeffs,
        const void *d_vectors,
        const void *d_base,
        int N_basis,
        void *d_out)
{
    const int V_d = S3DM_N_VERTS * 3;
    if (hipMemcpy(d_coeffs, coeffs, (size_t)N_basis * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    struct __attribute__((packed)) {
        void *out; const void *coeffs; const void *vec; const void *base;
        int N_basis; int V_d;
    } p = { d_out, d_coeffs, d_vectors, d_base, N_basis, V_d };
    unsigned bx = 256;
    unsigned gx = (unsigned)((V_d + bx - 1) / bx);
    return sb_launch(ctx->fn_mhr_blend, gx, 1, 1, bx, 1, 1, 0,
                     &p, sizeof(p));
}

static int cuda_sam3d_body_mhr_lbs_cached(
        cuda_sam3d_body_ctx *ctx,
        const float *global_skel_host,
        const void *d_rest_verts,
        float *out_verts_host)
{
    const sam3d_body_mhr_assets *a = ctx->cpu_mhr;
    const int J = S3DM_N_JOINTS;
    const int V = S3DM_N_VERTS;
    const int K = S3DM_N_SKIN;
    const float *IBP = (const float *)a->inverse_bind_pose.data;
    float jstate_host[S3DM_N_JOINTS * 8];

    for (int j = 0; j < J; j++) {
        const float *s1 = global_skel_host + (size_t)j * 8;
        const float *s2 = IBP + (size_t)j * 8;
        float *o        = jstate_host + (size_t)j * 8;
        float sc1 = s1[7], sc2 = s2[7];
        float vx = sc1 * s2[0], vy = sc1 * s2[1], vz = sc1 * s2[2];
        float qx = s1[3], qy = s1[4], qz = s1[5], qw = s1[6];
        float avx = qy*vz - qz*vy;
        float avy = qz*vx - qx*vz;
        float avz = qx*vy - qy*vx;
        float aavx = qy*avz - qz*avy;
        float aavy = qz*avx - qx*avz;
        float aavz = qx*avy - qy*avx;
        o[0] = s1[0] + vx + 2.0f * (avx * qw + aavx);
        o[1] = s1[1] + vy + 2.0f * (avy * qw + aavy);
        o[2] = s1[2] + vz + 2.0f * (avz * qw + aavz);
        float ax = s1[3], ay = s1[4], az = s1[5], aw = s1[6];
        float bx = s2[3], by = s2[4], bz = s2[5], bw = s2[6];
        o[3] = aw*bx + ax*bw + ay*bz - az*by;
        o[4] = aw*by - ax*bz + ay*bw + az*bx;
        o[5] = aw*bz + ax*by - ay*bx + az*bw;
        o[6] = aw*bw - ax*bx - ay*by - az*bz;
        o[7] = sc1 * sc2;
    }

    int rc = CUDA_SAM3D_BODY_E_LOAD;
    if (hipMemcpy(ctx->d_mhr_jstate, jstate_host,
                  (size_t)J * 8 * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess)
        goto done;
    if (hipMemset(ctx->d_mhr_skinned, 0,
                  (size_t)V * 3 * sizeof(float)) != hipSuccess)
        goto done;

    struct __attribute__((packed)) {
        void *out;
        const void *jstate; const void *rv;
        const void *si; const void *vi; const void *sw;
        int K;
    } p = {
        ctx->d_mhr_skinned, ctx->d_mhr_jstate, d_rest_verts,
        ctx->d_mhr_skin_indices, ctx->d_mhr_vert_indices,
        ctx->d_mhr_skin_weights, K
    };
    unsigned bx = 256;
    unsigned gx = (unsigned)((K + bx - 1) / bx);
    if (sb_launch(ctx->fn_mhr_lbs, gx, 1, 1, bx, 1, 1, 0,
                  &p, sizeof(p)) < 0)
        goto done;
    if (out_verts_host) {
        if (hipMemcpy(out_verts_host, ctx->d_mhr_skinned,
                      (size_t)V * 3 * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess)
            goto done;
    }
    rc = CUDA_SAM3D_BODY_E_OK;

done:
    return rc;
}

static int cuda_sam3d_body_keypoints_project_cached(
        cuda_sam3d_body_ctx *ctx,
        const float *global_skel_cm,
        const float *pred_cam,
        const sam3d_body_camera_batch *batch,
        float *kp3d_host,
        float *kp2d_full_host,
        float *kp2d_crop_host,
        float *kp2d_depth_host,
        float *pred_cam_t_host)
{
    if (!ctx || !ctx->mhr_gpu_enabled || !global_skel_cm || !pred_cam ||
        !batch || !ctx->d_mhr_skinned || !ctx->d_mhr_keypoint_mapping)
        return CUDA_SAM3D_BODY_E_INVAL;

    const int V = S3DM_N_VERTS;
    const int J = S3DM_N_JOINTS;
    const int K = ctx->cpu_dec ? ctx->cpu_dec->n_keypoints : SB_DEC_KP;
    if (K <= 0) return CUDA_SAM3D_BODY_E_INVAL;

    if (hipMemcpy(ctx->d_mhr_gskel, global_skel_cm,
                  (size_t)J * 8 * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;

    {
        struct __attribute__((packed)) {
            void *out; const void *verts; const void *gskel; const void *km;
            int V; int J; int K;
        } p = {
            ctx->d_mhr_kp3d, ctx->d_mhr_skinned, ctx->d_mhr_gskel,
            ctx->d_mhr_keypoint_mapping, V, J, K
        };
        unsigned bx = 256;
        unsigned gx = (unsigned)(K * 3);
        if (sb_launch(ctx->fn_mhr_keypoints, gx, 1, 1, bx, 1, 1, 0,
                      &p, sizeof(p)) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
    }

    {
        const float *KM = batch->cam_int;
        const float *AT = batch->affine_trans;
        struct __attribute__((packed)) {
            const void *kp3d;
            void *kp2d_full; void *kp2d_crop; void *kp2d_depth; void *pred_cam_t;
            float pred0; float pred1; float pred2;
            float bbox_scale; float bbox_cx; float bbox_cy;
            float ori_w; float ori_h;
            float img_w; float img_h;
            float k00; float k01; float k02;
            float k10; float k11; float k12;
            float k02_center; float k12_center;
            float a00; float a01; float a02;
            float a10; float a11; float a12;
            int use_intrin_center; int K;
        } p = {
            ctx->d_mhr_kp3d,
            ctx->d_mhr_kp2d_full, ctx->d_mhr_kp2d_crop,
            ctx->d_mhr_kp2d_depth, ctx->d_mhr_pred_cam_t,
            pred_cam[0], pred_cam[1], pred_cam[2],
            batch->bbox_scale, batch->bbox_center[0], batch->bbox_center[1],
            batch->ori_img_size[0], batch->ori_img_size[1],
            batch->img_size[0], batch->img_size[1],
            KM[0], KM[1], KM[2],
            KM[3], KM[4], KM[5],
            KM[2], KM[5],
            AT[0], AT[1], AT[2],
            AT[3], AT[4], AT[5],
            batch->use_intrin_center, K
        };
        unsigned bx = 128;
        unsigned gx = (unsigned)((K + (int)bx - 1) / (int)bx);
        if (sb_launch(ctx->fn_mhr_project, gx, 1, 1, bx, 1, 1, 0,
                      &p, sizeof(p)) < 0)
            return CUDA_SAM3D_BODY_E_LOAD;
    }

    if (kp3d_host &&
        hipMemcpy(kp3d_host, ctx->d_mhr_kp3d,
                  (size_t)K * 3 * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (kp2d_full_host &&
        hipMemcpy(kp2d_full_host, ctx->d_mhr_kp2d_full,
                  (size_t)K * 2 * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (kp2d_crop_host &&
        hipMemcpy(kp2d_crop_host, ctx->d_mhr_kp2d_crop,
                  (size_t)K * 2 * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (kp2d_depth_host &&
        hipMemcpy(kp2d_depth_host, ctx->d_mhr_kp2d_depth,
                  (size_t)K * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    if (pred_cam_t_host &&
        hipMemcpy(pred_cam_t_host, ctx->d_mhr_pred_cam_t,
                  3 * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return CUDA_SAM3D_BODY_E_LOAD;
    return CUDA_SAM3D_BODY_E_OK;
}

static int cuda_sam3d_body_mhr_forward_hybrid(
        cuda_sam3d_body_ctx *ctx,
        const float *model_params, const float *shape, const float *face,
        int apply_correctives, int n_threads, float *scratch,
        float *out_skinned_verts, float *out_global_skel)
{
    if (!ctx || !ctx->mhr_gpu_enabled || !ctx->cpu_mhr ||
        !model_params || !shape || !out_global_skel)
        return CUDA_SAM3D_BODY_E_INVAL;

    const sam3d_body_mhr_assets *a = ctx->cpu_mhr;
    const int B = 1;
    const int V = S3DM_N_VERTS;
    const int J = S3DM_N_JOINTS;
    size_t need = (size_t)B * (S3DM_N_PTRANS_OUT + J * 8 + V * 3 * 3);
    float *buf = scratch;
    int owned = 0;
    if (!buf) {
        buf = (float *)malloc(need * sizeof(float));
        if (!buf) return CUDA_SAM3D_BODY_E_LOAD;
        owned = 1;
    }
    float *jp      = buf;
    float *lskel   = jp + (size_t)B * S3DM_N_PTRANS_OUT;

    int r = CUDA_SAM3D_BODY_E_OK;
    if (sam3d_body_mhr_parameter_transform(a, model_params, B, n_threads, jp) != 0 ||
        sam3d_body_mhr_joint_params_to_local_skel(a, jp, B, lskel) != 0 ||
        sam3d_body_mhr_local_to_global_skel(a, lskel, B, out_global_skel) != 0) {
        r = CUDA_SAM3D_BODY_E_LOAD;
        goto done;
    }
    if (face) {
        r = cuda_sam3d_body_mhr_blend_cached(
                ctx, face, ctx->d_mhr_face_coeffs,
                ctx->d_mhr_face_vectors, NULL, S3DM_N_FACE, ctx->d_mhr_face);
        if (r != CUDA_SAM3D_BODY_E_OK) {
            r = CUDA_SAM3D_BODY_E_LOAD;
            goto done;
        }
    } else {
        if (hipMemset(ctx->d_mhr_face, 0,
                      (size_t)V * 3 * sizeof(float)) != hipSuccess) {
            r = CUDA_SAM3D_BODY_E_LOAD;
            goto done;
        }
    }
    r = cuda_sam3d_body_mhr_blend_cached(
            ctx, shape, ctx->d_mhr_shape_coeffs,
            ctx->d_mhr_blend_vectors, ctx->d_mhr_blend_base,
            S3DM_N_SHAPE, ctx->d_mhr_rest);
    if (r != CUDA_SAM3D_BODY_E_OK) goto done;
    if (apply_correctives) {
        r = cuda_sam3d_body_mhr_pose_correctives_cached(ctx, jp, NULL);
        if (r != CUDA_SAM3D_BODY_E_OK) goto done;
    } else {
        if (hipMemset(ctx->d_mhr_pc_out, 0,
                      (size_t)V * 3 * sizeof(float)) != hipSuccess) {
            r = CUDA_SAM3D_BODY_E_LOAD;
            goto done;
        }
    }
    if (sb_dec_add_inplace(ctx, ctx->d_mhr_rest, ctx->d_mhr_face, V * 3) < 0 ||
        sb_dec_add_inplace(ctx, ctx->d_mhr_rest, ctx->d_mhr_pc_out, V * 3) < 0) {
        r = CUDA_SAM3D_BODY_E_LOAD;
        goto done;
    }
    r = cuda_sam3d_body_mhr_lbs_cached(ctx, out_global_skel,
                                       ctx->d_mhr_rest, out_skinned_verts);

done:
    if (owned) free(buf);
    return r;
}

/* =====================================================================
 * MHR-on-GPU debug helpers.
 *
 * The production opt-in path above caches the MHR shape/face/pose/LBS
 * tensors and uses them inside run_decoder when SAM3D_BODY_GPU_MHR=1.
 * The helpers below are still useful for isolated verifier coverage:
 *   - blend_shape       (45 → V*3)   — uses mhr_blend_combine_f32
 *   - face_expressions  (72 → V*3)   — uses mhr_blend_combine_f32
 *   - pose_correctives  (jp 127×7 → V*3) — host 6D+sparse+ReLU,
 *                                          GPU dense GEMV via gemm_f32_bias
 *
 * The helper entry points upload weights ephemerally on each call, so they
 * remain verifier/debug APIs rather than the production fast path.
 * verify_mhr.c diffs each helper against the CPU reference.
 * ===================================================================== */

static int cuda_sam3d_body_run_blend_combine(cuda_sam3d_body_ctx *ctx,
                                             const float *coeffs,
                                             const float *vectors_host,
                                             const float *base_host,
                                             int N_basis,
                                             int V_d,
                                             float *out_host)
{
    void *d_coeffs = NULL, *d_vec = NULL, *d_base = NULL, *d_out = NULL;
    size_t b_coeffs = (size_t)N_basis * sizeof(float);
    size_t b_vec    = (size_t)N_basis * (size_t)V_d * sizeof(float);
    size_t b_base   = (size_t)V_d * sizeof(float);
    size_t b_out    = (size_t)V_d * sizeof(float);

    if (hipMalloc(&d_coeffs, b_coeffs) != hipSuccess) goto fail;
    if (hipMalloc(&d_vec,    b_vec)    != hipSuccess) goto fail;
    if (base_host && hipMalloc(&d_base, b_base) != hipSuccess) goto fail;
    if (hipMalloc(&d_out,    b_out)    != hipSuccess) goto fail;

    if (hipMemcpy(d_coeffs, coeffs, b_coeffs,
                  hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (hipMemcpy(d_vec, vectors_host, b_vec,
                  hipMemcpyHostToDevice) != hipSuccess) goto fail;
    if (base_host &&
        hipMemcpy(d_base, base_host, b_base,
                  hipMemcpyHostToDevice) != hipSuccess) goto fail;

    struct __attribute__((packed)) {
        void *out; const void *coeffs; const void *vec; const void *base;
        int N_basis; int V_d;
    } p = { d_out, d_coeffs, d_vec, base_host ? d_base : NULL, N_basis, V_d };
    unsigned bx = 256;
    unsigned gx = (unsigned)((V_d + bx - 1) / bx);
    if (sb_launch(ctx->fn_mhr_blend, gx, 1, 1, bx, 1, 1, 0, &p, sizeof(p)) < 0)
        goto fail;

    if (hipMemcpy(out_host, d_out, b_out,
                  hipMemcpyDeviceToHost) != hipSuccess) goto fail;

    hipFree(d_coeffs); hipFree(d_vec);
    if (d_base) hipFree(d_base);
    hipFree(d_out);
    return 0;

fail:
    if (d_coeffs) hipFree(d_coeffs);
    if (d_vec)    hipFree(d_vec);
    if (d_base)   hipFree(d_base);
    if (d_out)    hipFree(d_out);
    return CUDA_SAM3D_BODY_E_LOAD;
}

int cuda_sam3d_body_debug_run_blend_shape(cuda_sam3d_body_ctx *ctx,
                                          const float *shape_coeffs,
                                          float *out_verts)
{
    if (!ctx || !shape_coeffs || !out_verts) return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->cpu_mhr) {
        fprintf(stderr, "sam3d_body: MHR assets not loaded — "
                        "set cfg.mhr_assets_dir at create time\n");
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    const sam3d_body_mhr_assets *a = ctx->cpu_mhr;
    const float *SV = (const float *)a->blend_shape_vectors.data;
    const float *BS = (const float *)a->blend_base_shape.data;
    return cuda_sam3d_body_run_blend_combine(ctx, shape_coeffs, SV, BS,
                                             S3DM_N_SHAPE,
                                             S3DM_N_VERTS * 3,
                                             out_verts);
}

int cuda_sam3d_body_debug_run_face_expressions(cuda_sam3d_body_ctx *ctx,
                                               const float *face_coeffs,
                                               float *out_verts)
{
    if (!ctx || !face_coeffs || !out_verts) return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->cpu_mhr) {
        fprintf(stderr, "sam3d_body: MHR assets not loaded — "
                        "set cfg.mhr_assets_dir at create time\n");
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    const sam3d_body_mhr_assets *a = ctx->cpu_mhr;
    const float *SV = (const float *)a->face_shape_vectors.data;
    return cuda_sam3d_body_run_blend_combine(ctx, face_coeffs, SV, NULL,
                                             S3DM_N_FACE,
                                             S3DM_N_VERTS * 3,
                                             out_verts);
}

/* pose_correctives — split execution.
 *
 * The 6D feat + subtract-identity + sparse-matvec + ReLU stages add up to
 * ~53k mul-adds; cheaper to keep on the host than to launch tiny kernels
 * for each. The dense Linear (55317, 3000) GEMV that follows is the only
 * GPU-worthy chunk (166M FMAs) and reuses the existing gemm_f32_bias.
 */
static void cpc_batch6d_from_xyz(const float *e, float *out6)
{
    float cx = cosf(e[0]), sx = sinf(e[0]);
    float cy = cosf(e[1]), sy = sinf(e[1]);
    float cz = cosf(e[2]), sz = sinf(e[2]);
    out6[0] = cy * cz;
    out6[1] = cy * sz;
    out6[2] = -sy;
    out6[3] = -cx * sz + sx * sy * cz;
    out6[4] = cx * cz + sx * sy * sz;
    out6[5] = sx * cy;
}

int cuda_sam3d_body_debug_run_pose_correctives(cuda_sam3d_body_ctx *ctx,
                                               const float *joint_params,
                                               float *out_verts)
{
    if (!ctx || !joint_params || !out_verts) return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->cpu_mhr) {
        fprintf(stderr, "sam3d_body: MHR assets not loaded — "
                        "set cfg.mhr_assets_dir at create time\n");
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    const sam3d_body_mhr_assets *a = ctx->cpu_mhr;
    const int V_d = S3DM_N_VERTS * 3;          /* 55317 */
    const int HID = S3DM_N_PC_H;               /* 3000  */
    const int NNZ = S3DM_N_PC_NNZ;             /* 53136 */
    const int FEAT_IN = S3DM_N_PC_IN;          /* 750   */

    const int64_t *spi_row = (const int64_t *)a->pc_sparse_indices.data;
    const int64_t *spi_col = spi_row + NNZ;
    const float   *spw     = (const float *)a->pc_sparse_weight.data;
    const float   *LW      = (const float *)a->pc_linear_weight.data;

    float *feat = (float *)malloc((size_t)FEAT_IN * sizeof(float));
    float *h    = (float *)malloc((size_t)HID     * sizeof(float));
    if (!feat || !h) { free(feat); free(h); return CUDA_SAM3D_BODY_E_LOAD; }

    /* 6D feat from joints 2..126 (125 joints), euler at jp[j, 3:6]. */
    for (int jj = 0; jj < 125; jj++) {
        const float *e = joint_params + (size_t)(jj + 2) * 7 + 3;
        cpc_batch6d_from_xyz(e, feat + (size_t)jj * 6);
    }
    for (int jj = 0; jj < 125; jj++) {
        feat[(size_t)jj * 6 + 0] -= 1.0f;
        feat[(size_t)jj * 6 + 4] -= 1.0f;
    }

    /* Sparse matvec + ReLU. */
    memset(h, 0, (size_t)HID * sizeof(float));
    for (int k = 0; k < NNZ; k++) {
        int row = (int)spi_row[k];
        int col = (int)spi_col[k];
        h[row] += spw[k] * feat[col];
    }
    for (int i = 0; i < HID; i++) if (h[i] < 0.0f) h[i] = 0.0f;
    free(feat);

    /* Dense GEMV on GPU: out (1, V_d) = h (1, HID) @ LW^T (V_d, HID). */
    void *d_h = NULL, *d_LW = NULL, *d_out = NULL;
    size_t b_h   = (size_t)HID * sizeof(float);
    size_t b_LW  = (size_t)V_d * (size_t)HID * sizeof(float);
    size_t b_out = (size_t)V_d * sizeof(float);

    if (hipMalloc(&d_h,   b_h)   != hipSuccess) goto pc_fail;
    if (hipMalloc(&d_LW,  b_LW)  != hipSuccess) goto pc_fail;
    if (hipMalloc(&d_out, b_out) != hipSuccess) goto pc_fail;

    if (hipMemcpy(d_h,  h,  b_h,  hipMemcpyHostToDevice) != hipSuccess) goto pc_fail;
    if (hipMemcpy(d_LW, LW, b_LW, hipMemcpyHostToDevice) != hipSuccess) goto pc_fail;

    /* gemm_f32_bias: Y(N, D_out) = X(N, D_in) @ W^T(D_out, D_in) + b. */
    struct __attribute__((packed)) {
        void *Y; const void *X; const void *W; const void *b;
        int N; int D_in; int D_out;
    } p = { d_out, d_h, d_LW, NULL, 1, HID, V_d };
    unsigned bx = 16, by = 16;
    unsigned gx = (unsigned)((1   + bx - 1) / bx);
    unsigned gy = (unsigned)((V_d + by - 1) / by);
    if (sb_launch(ctx->fn_gemm_f32, gx, gy, 1, bx, by, 1, 0, &p, sizeof(p)) < 0)
        goto pc_fail;

    if (hipMemcpy(out_verts, d_out, b_out,
                  hipMemcpyDeviceToHost) != hipSuccess) goto pc_fail;

    hipFree(d_h); hipFree(d_LW); hipFree(d_out);
    free(h);
    return 0;

pc_fail:
    if (d_h)   hipFree(d_h);
    if (d_LW)  hipFree(d_LW);
    if (d_out) hipFree(d_out);
    free(h);
    return CUDA_SAM3D_BODY_E_LOAD;
}

/* LBS skin_points — debug MHR-on-GPU helper.
 *
 * Mirrors sam3d_body_mhr_skin_points (B=1): caller supplies the
 * global_skel (J=127, 8) and we compute jstate = skel_multiply(global,
 * inverse_bind_pose) on the host (J=127 trivial ops), upload it, then
 * scatter-add w * skel_transform_point(jstate[j], rest_verts[v]) into
 * out_verts[v] for K=51337 skin entries via atomicAdd.
 *
 * Output buffer is zeroed on device before the launch. Numerics match the
 * CPU reference up to atomicAdd reduction order; the per-vert sum has at
 * most ~16 contributions (typical LBS) so f32 accumulation drift stays at
 * ~1e-5 absolute on cm-scale verts.
 */
int cuda_sam3d_body_debug_run_lbs_skin(cuda_sam3d_body_ctx *ctx,
                                       const float *global_skel_host,
                                       const float *rest_verts_host,
                                       float *out_verts_host)
{
    if (!ctx || !global_skel_host || !rest_verts_host || !out_verts_host)
        return CUDA_SAM3D_BODY_E_INVAL;
    if (!ctx->cpu_mhr) {
        fprintf(stderr, "sam3d_body: MHR assets not loaded — "
                        "set cfg.mhr_assets_dir at create time\n");
        return CUDA_SAM3D_BODY_E_INVAL;
    }
    const sam3d_body_mhr_assets *a = ctx->cpu_mhr;
    const int J = S3DM_N_JOINTS;
    const int V = S3DM_N_VERTS;
    const int K = S3DM_N_SKIN;

    const int32_t *si  = (const int32_t *)a->skin_indices_flat.data;
    const float   *sw  = (const float   *)a->skin_weights_flat.data;
    const int64_t *vi  = (const int64_t *)a->vert_indices_flat.data;
    const float   *IBP = (const float   *)a->inverse_bind_pose.data;

    /* Compute jstate = skel_multiply(global, IBP) on host (J=127 ops). The
     * static-inline impl in sam3d_body_mhr.h is gated by IMPLEMENTATION; do
     * the math inline here to avoid pulling in the full impl block. */
    float *jstate_host = (float *)malloc((size_t)J * 8 * sizeof(float));
    if (!jstate_host) return CUDA_SAM3D_BODY_E_LOAD;
    for (int j = 0; j < J; j++) {
        const float *s1 = global_skel_host + (size_t)j * 8;
        const float *s2 = IBP + (size_t)j * 8;
        float *o        = jstate_host + (size_t)j * 8;
        float sc1 = s1[7], sc2 = s2[7];
        /* rot = quat_rot_vec(q1, sc1 * t2) */
        float vx = sc1 * s2[0], vy = sc1 * s2[1], vz = sc1 * s2[2];
        float qx = s1[3], qy = s1[4], qz = s1[5], qw = s1[6];
        float avx = qy*vz - qz*vy;
        float avy = qz*vx - qx*vz;
        float avz = qx*vy - qy*vx;
        float aavx = qy*avz - qz*avy;
        float aavy = qz*avx - qx*avz;
        float aavz = qx*avy - qy*avx;
        o[0] = s1[0] + vx + 2.0f * (avx * qw + aavx);
        o[1] = s1[1] + vy + 2.0f * (avy * qw + aavy);
        o[2] = s1[2] + vz + 2.0f * (avz * qw + aavz);
        /* q1 * q2  (xyzw, w last) */
        float ax = s1[3], ay = s1[4], az = s1[5], aw = s1[6];
        float bx = s2[3], by = s2[4], bz = s2[5], bw = s2[6];
        o[3] = aw*bx + ax*bw + ay*bz - az*by;
        o[4] = aw*by - ax*bz + ay*bw + az*bx;
        o[5] = aw*bz + ax*by - ay*bx + az*bw;
        o[6] = aw*bw - ax*bx - ay*by - az*bz;
        o[7] = sc1 * sc2;
    }

    void *d_js = NULL, *d_rv = NULL, *d_si = NULL, *d_vi = NULL,
         *d_sw = NULL, *d_out = NULL;
    size_t b_js  = (size_t)J * 8 * sizeof(float);
    size_t b_rv  = (size_t)V * 3 * sizeof(float);
    size_t b_si  = (size_t)K * sizeof(int32_t);
    size_t b_vi  = (size_t)K * sizeof(int64_t);
    size_t b_sw  = (size_t)K * sizeof(float);
    size_t b_out = (size_t)V * 3 * sizeof(float);

    if (hipMalloc(&d_js,  b_js)  != hipSuccess) goto lbs_fail;
    if (hipMalloc(&d_rv,  b_rv)  != hipSuccess) goto lbs_fail;
    if (hipMalloc(&d_si,  b_si)  != hipSuccess) goto lbs_fail;
    if (hipMalloc(&d_vi,  b_vi)  != hipSuccess) goto lbs_fail;
    if (hipMalloc(&d_sw,  b_sw)  != hipSuccess) goto lbs_fail;
    if (hipMalloc(&d_out, b_out) != hipSuccess) goto lbs_fail;

    if (hipMemcpy(d_js, jstate_host,     b_js, hipMemcpyHostToDevice) != hipSuccess) goto lbs_fail;
    if (hipMemcpy(d_rv, rest_verts_host, b_rv, hipMemcpyHostToDevice) != hipSuccess) goto lbs_fail;
    if (hipMemcpy(d_si, si, b_si, hipMemcpyHostToDevice) != hipSuccess) goto lbs_fail;
    if (hipMemcpy(d_vi, vi, b_vi, hipMemcpyHostToDevice) != hipSuccess) goto lbs_fail;
    if (hipMemcpy(d_sw, sw, b_sw, hipMemcpyHostToDevice) != hipSuccess) goto lbs_fail;
    if (hipMemset(d_out, 0, b_out) != hipSuccess) goto lbs_fail;

    struct __attribute__((packed)) {
        void *out;
        const void *jstate; const void *rv;
        const void *si; const void *vi; const void *sw;
        int K;
    } p = { d_out, d_js, d_rv, d_si, d_vi, d_sw, K };
    unsigned bx = 256;
    unsigned gx = (unsigned)((K + bx - 1) / bx);
    if (sb_launch(ctx->fn_mhr_lbs, gx, 1, 1, bx, 1, 1, 0, &p, sizeof(p)) < 0)
        goto lbs_fail;

    if (hipMemcpy(out_verts_host, d_out, b_out,
                  hipMemcpyDeviceToHost) != hipSuccess) goto lbs_fail;

    hipFree(d_js); hipFree(d_rv); hipFree(d_si); hipFree(d_vi);
    hipFree(d_sw); hipFree(d_out);
    free(jstate_host);
    return 0;

lbs_fail:
    if (d_js)  hipFree(d_js);
    if (d_rv)  hipFree(d_rv);
    if (d_si)  hipFree(d_si);
    if (d_vi)  hipFree(d_vi);
    if (d_sw)  hipFree(d_sw);
    if (d_out) hipFree(d_out);
    free(jstate_host);
    return CUDA_SAM3D_BODY_E_LOAD;
}
