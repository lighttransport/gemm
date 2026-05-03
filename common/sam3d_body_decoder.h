/*
 * sam3d_body_decoder.h — sam-3d-body promptable decoder + MHR head.
 *
 * Usage:
 *   #define SAM3D_BODY_DECODER_IMPLEMENTATION
 *   #include "sam3d_body_decoder.h"
 *
 * Dependencies: ggml_dequant.h, safetensors.h, cpu_compute.h
 *
 * Architecture (confirmed by step-1 safetensors probe):
 *   - 6 decoder layers, token dim=1024, n_heads=8, head_dim=64
 *   - Cross-attention q from tokens (1024→512), k/v from image (1280→512)
 *   - Self-attention q/k/v all 1024→512
 *   - FFN hidden = 1024 (not 4× — small FFN)
 *   - Per-layer norms: ln1, ln2_1, ln2_2 (1280), ln3, ln_pe_1, ln_pe_2 (1280)
 *   - Input projections: init_to_token_mhr (Linear 525→1024),
 *     prev_to_token_mhr (Linear 522→1024), prompt_to_token (Linear 1280→1024)
 *   - Learnable embeddings: keypoint_embedding (70, 1024),
 *     keypoint3d_embedding (70, 1024), hand_box_embedding (2, 1024)
 *   - Keypoint-token update: keypoint_feat_linear (1280→1024),
 *     keypoint_posemb_linear (MLP 2→1024→1024→1024),
 *     keypoint3d_posemb_linear (MLP similar, 3D)
 *   - ray_cond_emb: Conv2d(1379, 1280, 1×1) + LN(1280) — added to image tokens
 *   - hand_pe_layer.positional_encoding_gaussian_matrix: (2, 640) SAM PE
 *
 * MHR head (in separate safetensors file sam3d_body_mhr_head.safetensors):
 *   - head_pose.proj: FFN(1024 → 1024 → 519)
 *   - head_camera.proj: FFN(1024 → 1024 → 3)
 *   - init_pose.weight: (1, 519); init_camera.weight: (1, 3)
 *   - bbox_embed: MLP(1024 → 1024 → 1024 → 4) (hand detection)
 *   - head_pose.{scale_mean, scale_comps, keypoint_mapping, faces,
 *     joint_rotation, hand_pose_{mean,comps}, ...} — MHR skinning buffers
 *
 * Scope v1:
 *   - Body branch only (skip _hand variants)
 *   - Empty prompts (dummy keypoints [0, 0, -2])
 *   - DINOv3 32x32 and ViT-H 32x24 image-token grids
 *   - Produces 519 MHR params + 3 cam_t values; focal from runner's fov hint.
 *
 * The implementation covers ray conditioning, token construction, all six
 * decoder layers, iterative keypoint-token updates, final MHR/camera heads,
 * and the public wrapper used by the CPU/CUDA runners.
 */

#ifndef SAM3D_BODY_DECODER_H
#define SAM3D_BODY_DECODER_H

#include <stdint.h>
#include <stddef.h>
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SAM3D_BODY_DECODER_E_OK              (0)
#define SAM3D_BODY_DECODER_E_INVAL           (-1)
#define SAM3D_BODY_DECODER_E_NOT_IMPLEMENTED (-2)
#define SAM3D_BODY_DECODER_E_LOAD            (-3)

#include "qtensor_utils.h"

/* ---- Per-layer weights (TransformerDecoderLayer in upstream) ------ */
typedef struct {
    /* Pre-norms for token / image streams. ln_pe_{1,2} normalize the
     * token_augment / image_augment PE streams separately. */
    qtensor ln1_w,    ln1_b;      /* LN on tokens before self-attn */
    qtensor ln_pe_1_w,ln_pe_1_b;  /* LN on token_augment */
    qtensor ln2_1_w,  ln2_1_b;    /* LN on tokens before cross-attn */
    qtensor ln2_2_w,  ln2_2_b;    /* LN on image (1280) before cross-attn */
    qtensor ln_pe_2_w,ln_pe_2_b;  /* LN on image_augment (1280) */
    qtensor ln3_w,    ln3_b;      /* LN on tokens before FFN */

    /* Self-attention: 1024 → 512 per projection, out 512 → 1024. */
    qtensor sa_q_w, sa_q_b;   /* (512, 1024) */
    qtensor sa_k_w, sa_k_b;   /* (512, 1024) */
    qtensor sa_v_w, sa_v_b;   /* (512, 1024) */
    qtensor sa_proj_w, sa_proj_b; /* (1024, 512) */

    /* Cross-attention: q 1024→512, k/v 1280→512, out 512→1024. */
    qtensor ca_q_w, ca_q_b;   /* (512, 1024) */
    qtensor ca_k_w, ca_k_b;   /* (512, 1280) */
    qtensor ca_v_w, ca_v_b;   /* (512, 1280) */
    qtensor ca_proj_w, ca_proj_b; /* (1024, 512) */

    /* FFN: 1024 → 1024 → 1024 (ffn.layers.0.0 + ffn.layers.1). */
    qtensor ffn0_w, ffn0_b;   /* (1024, 1024) */
    qtensor ffn1_w, ffn1_b;   /* (1024, 1024) */
} sam3d_body_decoder_layer;

/* ---- Full decoder model (body branch only, v1). ------------------- */
typedef struct sam3d_body_decoder_model_t {
    /* Config. */
    int n_layers;      /* 6 */
    int dim;           /* 1024 (token channel) */
    int kv_dim;        /* 1280 (image channel, DINOv3 output) */
    int n_heads;       /* 8 */
    int head_dim;      /* 64 (proj dim 512 / 8) */
    int ffn_hidden;    /* 1024 */
    int n_keypoints;   /* 70 */
    int n_hand_tokens; /* 2 */
    int npose;         /* 519 (from MHR head) */
    int ncam;          /* 3 */
    int cond_dim;      /* 3 */
    float ln_eps;      /* 1e-6 */

    /* Input projections. */
    qtensor init_to_token_w, init_to_token_b;  /* (1024, npose+ncam+cond) = (1024, 525) */
    qtensor prev_to_token_w, prev_to_token_b;  /* (1024, npose+ncam)      = (1024, 522) */
    qtensor prompt_to_token_w, prompt_to_token_b; /* (1024, 1280) */

    /* Learnable embeddings. */
    qtensor keypoint_embedding;       /* (70, 1024) */
    qtensor keypoint3d_embedding;     /* (70, 1024) */
    qtensor hand_box_embedding;       /* (2, 1024) */

    /* Keypoint-token update path. */
    qtensor kp_feat_linear_w, kp_feat_linear_b;     /* (1024, 1280) */
    /* keypoint_posemb_linear: FFN(2, 1024, 1024, 1024) — upstream MLP with 2 hidden FC. */
    qtensor kp_posemb_l0_w, kp_posemb_l0_b;         /* (1024, 2) */
    qtensor kp_posemb_l1_w, kp_posemb_l1_b;         /* (1024, 1024) */
    /* keypoint3d_posemb_linear: same shape but input 3. */
    qtensor kp3d_posemb_l0_w, kp3d_posemb_l0_b;     /* (1024, 3) */
    qtensor kp3d_posemb_l1_w, kp3d_posemb_l1_b;     /* (1024, 1024) */

    /* Ray conditioning (adds to image embeddings before decoder). */
    qtensor ray_cond_conv_w, ray_cond_conv_b;       /* (1280, 1379, 1, 1) */
    qtensor ray_cond_norm_w, ray_cond_norm_b;       /* (1280,) */
    /* SAM-style random-projection PE.
     * `prompt_pe_gauss` is the body decoder's prompt_encoder.pe_layer
     * gaussian matrix (used by get_dense_pe to build the (1280, H, W)
     * context positional encoding consumed as `image_pe`/`context_pe`
     * by the decoder). `hand_pe_gauss` is the analogous matrix for the
     * hand decoder branch (unused in v1 body-only path). */
    qtensor prompt_pe_gauss;                        /* (2, 640) */
    qtensor hand_pe_gauss;                          /* (2, 640) */
    /* Single-token "invalid" prompt embedding used when no real
     * keypoint prompts are supplied; broadcast to (B, 1, 1280) and fed
     * through prompt_to_token. */
    qtensor invalid_point_embed;                    /* (1, 1280) */

    /* Per-layer weights. */
    sam3d_body_decoder_layer *layers;               /* [n_layers] */

    /* Final norm after all decoder layers. */
    qtensor norm_final_w, norm_final_b;             /* (1024,) */

    /* MHR head (loaded from sam3d_body_mhr_head.safetensors). */
    qtensor head_pose_l0_w, head_pose_l0_b;         /* (1024, 1024) */
    qtensor head_pose_l1_w, head_pose_l1_b;         /* (519, 1024) */
    qtensor head_camera_l0_w, head_camera_l0_b;     /* (1024, 1024) */
    qtensor head_camera_l1_w, head_camera_l1_b;     /* (3, 1024) */
    qtensor init_pose;       /* (1, 519) — learnable init MHR param token */
    qtensor init_camera;     /* (1, 3)   — learnable init cam_t token */

    /* MHR head decode buffers (stages 1-5 + 12). All from head_pose.* in
     * sam3d_body_mhr_head.safetensors. */
    qtensor scale_mean;       /* (68,)        f32 */
    qtensor scale_comps;      /* (28, 68)     f32 */
    qtensor hand_pose_mean;   /* (54,)        f32 */
    qtensor hand_pose_comps;  /* (54, 54)     f32 */
    qtensor hand_idx_left;    /* (27,)        i64 — into 133-dim body_pose */
    qtensor hand_idx_right;   /* (27,)        i64 */
    qtensor keypoint_mapping; /* (308, 18566) f32 */
    qtensor local_to_world_wrist; /* (3, 3)   f32 — wrist-frame transform */
    qtensor right_wrist_coords;   /* (3,)     f32 */
    qtensor root_coords;          /* (3,)     f32 */
    qtensor nonhand_param_idxs;   /* (145,)   i64 — zeroed in model_params */
    qtensor faces;                /* (36874, 3) i64 — mesh triangle indices */

    /* Optional hand-bbox detection head (unused in v1 body-only path). */
    qtensor bbox_embed_l0_w, bbox_embed_l0_b;       /* (1024, 1024) */
    qtensor bbox_embed_l1_w, bbox_embed_l1_b;       /* (1024, 1024) */
    qtensor bbox_embed_l2_w, bbox_embed_l2_b;       /* (4, 1024) */
    qtensor hand_cls_w, hand_cls_b;                 /* (2, 1024) */

    /* Backing buffer for parsed safetensors (kept alive until free). */
    void *_st_decoder;   /* decoder safetensors context */
    void *_st_mhr_head;  /* mhr_head safetensors context */
} sam3d_body_decoder_model;

/* Result from forward(): 519 MHR params + 3 cam_t. */
typedef struct {
    float mhr_params[519];
    float cam_t[3];
} sam3d_body_decoder_result;

/* ---- Public API. All loaders: NULL on failure. -------------------- */

/* Load both safetensors files (decoder + mhr_head) and populate model. */
sam3d_body_decoder_model *sam3d_body_decoder_load(const char *decoder_sft,
                                                  const char *mhr_head_sft);
void                      sam3d_body_decoder_free(sam3d_body_decoder_model *m);

/* sam3d_body_decoder_forward — convenience entry point that bundles
 * ray_cond_emb + get_dense_pe + build_tokens + forward_full in one call.
 * Definition is placed below the struct definitions (see after
 * sam3d_body_decoder_forward_full). */

/*
 * ray_cond_emb sub-module (step 4c): apply Conv2d(1379,1280,1x1) + LN2d
 * to (image_embeddings ⊕ fourier(rays_xyz)).
 *
 *   image_emb: (1280, 32, 32) f32, channel-first, patch-token grid
 *   rays_xyz:  (32, 32, 3)    f32, per-spatial (dx, dy, 1) after the
 *              downsample + z-append. Produces a 99-dim fourier feature
 *              per position (3 raw + 16×3 sin + 16×3 cos).
 *   out:       (1280, 32, 32) f32
 *
 * Fourier bands: linspace(1, 32, 16) per axis, sin/cos(π·pos·band).
 * LN2d eps = 1e-6, applied per-spatial over the 1280 channels.
 */
int sam3d_body_ray_cond_emb_forward(const sam3d_body_decoder_model *m,
                                    const float *image_emb,
                                    const float *rays_xyz,
                                    int H, int W,
                                    int n_threads,
                                    float *out);

/*
 * SAM PositionEmbeddingRandom.get_dense_pe: build the (embed_dim, H, W)
 * dense positional encoding consumed as `image_pe`/`context_pe` by the
 * decoder cross-attention. Uses the prompt_encoder.pe_layer gaussian
 * matrix (m->prompt_pe_gauss, shape (2, embed_dim/2)).
 *
 * For each grid cell (yi, xi) the (x, y) center is normalized to the
 * square crop grid. DINOv3 uses H==W. ViT-H uses a 32x24 embedding grid
 * sliced from the center of the 32x32 crop, so x is offset by (H-W)/2
 * and normalized by H, not W.
 *
 *   out: (embed_dim=1280, H, W) f32, contiguous CHW layout.
 *
 * embed_dim is inferred from the gaussian matrix (cols × 2).
 */
int sam3d_body_get_dense_pe(const sam3d_body_decoder_model *m,
                            int H, int W, int n_threads, float *out);

/*
 * Build the prompt input for build_tokens when no real keypoints are
 * supplied: equivalent to running prompt_encoder on a single dummy
 * keypoint with label==-2 (invalid). Output is the (1280,) vector that
 * gets fed as `prompt_in` to sam3d_body_build_tokens.
 *
 *   out: (embed_dim=1280,) f32 — m->invalid_point_embed.weight.
 */
int sam3d_body_invalid_prompt_token(const sam3d_body_decoder_model *m,
                                    float *out);

/*
 * Build the (1, H_out, W_out, 3) ray-condition tensor consumed by
 * ray_cond_emb_forward. Mirrors meta_arch.sam3d_body.get_ray_condition
 * + the in-CameraEncoder antialias-bilinear downsample + z-append.
 *
 * Steps (per upstream):
 *   1. meshgrid_xy = arange(H_in) × arange(W_in)              (H_in, W_in, 2)
 *   2. inverse-affine apply (only diagonal scale + col-2 trans)
 *   3. inverse-camera apply (subtract (cx, cy), divide by (fx, fy))
 *   4. If the output grid is rectangular (ViT-H: 32x24), center-crop
 *      the x ramp to the crop width before downsampling. Upstream crops
 *      the 512x512 body crop to 512x384 before the ViT-H patch embed.
 *   5. F.interpolate(antialias=True, bilinear, scale=1/patch_size)
 *      separable triangle kernel along H then W
 *   5. concat z=1 along the last axis
 *
 * Because the inverse-affine + inverse-camera transforms are
 * coordinate-only (x depends only on column j; y depends only on row
 * i), the H- and W- antialias filters reduce to a 1D triangle filter
 * applied independently to a precomputed H_in-long y-ramp and a
 * W_in-long x-ramp. Boundary clipping (output pixels near the image
 * edge see an asymmetric window) IS applied — matches PyTorch.
 *
 * Inputs:
 *   cam_int      (3,3)  fx, fy, cx, cy at indices [0,0],[1,1],[0,2],[1,2]
 *   affine_trans (2,3)  a00, a11, t02, t12 used; off-diagonal a01/a10
 *                       are assumed 0 (TopdownAffine for body always is)
 *   H_in, W_in          full input resolution (e.g. 512, 512)
 *   H_out, W_out        downsampled resolution (e.g. 32, 32)
 *
 * Output:
 *   out: (H_out, W_out, 3) f32, last channel = 1.0
 */
int sam3d_body_compute_ray_cond_xyz(const float *cam_int,
                                    const float *affine_trans,
                                    int H_in, int W_in,
                                    int H_out, int W_out,
                                    float *out);

/*
 * Default camera intrinsics for sam-3d-body (prepare_batch.py).
 *   fx = fy = sqrt(W² + H²); cx = W/2; cy = H/2.
 * Output: cam_int (3, 3) row-major.
 */
void sam3d_body_default_cam_int(int W, int H, float *cam_int);

/*
 * fix_aspect_ratio (data/transforms/bbox_utils.py).
 * Input: scale (sw, sh); aspect_ratio = w/h target.
 * If sw > sh * aspect_ratio: scale = (sw, sw/aspect_ratio)
 *                      else: scale = (sh*aspect_ratio, sh)
 */
void sam3d_body_fix_aspect_ratio(const float *scale_in, float aspect_ratio,
                                 float *scale_out);

/*
 * get_warp_matrix (data/transforms/bbox_utils.py) for rot=0, shift=(0,0).
 * Input: center (2,) cx, cy; scale (2,) src_w (only [0] is used);
 *        output_size (out_w, out_h).
 * Output: warp_mat (2, 3) row-major.
 *
 * For rot=0: warp = (s, 0, out_w/2 - cx*s; 0, s, out_h/2 - cy*s)
 *   where s = out_w / src_w (note: y-scale also uses out_w, matching upstream).
 *
 * NOTE: upstream uses cv2.getAffineTransform on 3 src/dst points; for rot=0
 * the closed form above is exact (verified against the reference dump).
 */
void sam3d_body_get_warp_matrix(const float *center, const float *scale,
                                int out_w, int out_h, float *warp_mat);

/*
 * CLIFF condition_info (meta_arch.sam3d_body._get_decoder_condition).
 *   condition_info = ((cx - W/2)/f, (cy - H/2)/f, b/f)   if !use_intrin_center
 *   condition_info = ((cx - cam_cx)/f, (cy - cam_cy)/f, b/f)   else
 * where:
 *   cx, cy = bbox_center (pixels)
 *   f      = cam_int[0,0] (focal length)
 *   W, H   = ori_img_size
 *   b      = bbox_scale[0]
 *   cam_cx, cam_cy = cam_int[0,2], cam_int[1,2]
 *
 * Output: condition_info (3,).
 */
void sam3d_body_compute_condition_info(const float *bbox_center,
                                       const float *bbox_scale,
                                       const float *ori_img_size,
                                       const float *cam_int,
                                       int use_intrin_center,
                                       float *condition_info);

/*
 * GetBBoxCenterScale + TopdownAffine composition (data/transforms/common.py).
 *
 *   bbox_xyxy: (4,) [x1, y1, x2, y2]
 *   padding:   1.25 upstream
 *   aspect_ratio_pre: 0.75 upstream (first fix pass)
 *   out_w, out_h: 512, 512 upstream
 *
 * Output:
 *   center    (2,), scale (2,)   — final scale after TWO fix_aspect_ratio
 *                                  passes (0.75 then out_w/out_h)
 *   warp_mat  (2, 3)
 */
void sam3d_body_compute_bbox_affine(const float *bbox_xyxy,
                                    float padding,
                                    float aspect_ratio_pre,
                                    int out_w, int out_h,
                                    float *center,
                                    float *scale,
                                    float *warp_mat);

/*
 * cv2.warpAffine + ImageNet norm + HWC→CHW.
 *
 *   img_rgb_hwc: u8 (H_in, W_in, 3)
 *   warp_mat:    (2, 3) forward affine src→dst (same matrix cv2 accepts)
 *   out_chw:     f32 (3, H_out, W_out), normalized
 *
 * OOB pixels use 0 before normalization — i.e. output = (0/255 - mean)/std
 * for any dst pixel that maps outside the source image, matching
 * cv2.warpAffine(flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=0).
 */
int sam3d_body_preprocess_image(const uint8_t *img_rgb_hwc,
                                int W_in, int H_in,
                                const float *warp_mat_2x3,
                                int W_out, int H_out,
                                float *out_chw);

/*
 * Token construction (step 4d):
 *
 *   1. init_token   = init_to_token_mhr(init_input)   where
 *                     init_input = concat(condition_info(3), init_pose(519), init_camera(3))
 *                     → (1, 525), projected to (1, 1024).
 *   2. prev_token   = prev_to_token_mhr(prev_input)   where
 *                     prev_input = concat(init_pose(519), init_camera(3))
 *                     → (1, 522) (NO condition_info), projected to (1, 1024).
 *   3. prompt_token = prompt_to_token(prompt_in)      where
 *                     prompt_in = prompt_encoder(keypoints) → (1, 1280),
 *                     projected to (1, 1024).
 *
 * Assembly of 145-token stack:
 *
 *   x [B=1, 145, 1024]:
 *     [0]       = init_token (pose)
 *     [1]       = prev_token
 *     [2]       = prompt_token
 *     [3..4]    = hand_box_embedding.weight (2, 1024)
 *     [5..74]   = keypoint_embedding.weight (70, 1024)
 *     [75..144] = keypoint3d_embedding.weight (70, 1024)
 *
 *   x_pe [B=1, 145, 1024] (token_augment):
 *     [0]       = zeros
 *     [1]       = prev_token
 *     [2]       = prompt_token
 *     [3..144]  = zeros (kp posembs get populated later by the per-layer
 *                 keypoint_token_update_fn, step 4f)
 *
 * Inputs to this routine are the POST-concat flat vectors:
 *   init_input: (525,) f32  = [condition_info(3); init_pose(519); init_camera(3)]
 *   prev_input: (522,) f32  = [init_pose(519); init_camera(3)]
 *   prompt_in:  (1280,) f32 = prompt_encoder output (all-invalid keypoints case).
 *
 * Output buffers must be 145*1024 f32 each.
 */
int sam3d_body_build_tokens(const sam3d_body_decoder_model *m,
                            const float *init_input,   /* (525,) */
                            const float *prev_input,   /* (522,) */
                            const float *prompt_in,    /* (1280,) */
                            int n_threads,
                            float *x_out,     /* (145, 1024) */
                            float *x_pe_out); /* (145, 1024) */

/*
 * TransformerDecoderLayer forward (step 4e), body branch with repeat_pe=True.
 *
 *   layer_idx      ∈ [0, n_layers). Layer 0 has skip_first_pe=True, so its
 *                  self-attn does NOT add x_pe to q/k.
 *   x_in[N_q*D]    tokens, N_q=145, D=1024
 *   context_in[N_c*D_c]  image tokens, N_c=1024, D_c=1280
 *   x_pe_in[N_q*D]       token PE (token_augment). Normalized via ln_pe_1 internally.
 *   context_pe_in[N_c*D_c] image PE (image_augment). Normalized via ln_pe_2 internally.
 *   x_out[N_q*D]   updated tokens.
 *   context_out    unused with enable_twoway=False; caller may pass NULL.
 *                  If non-NULL, it is memcpy'd from context_in for API symmetry.
 */
int sam3d_body_decoder_layer_forward(const sam3d_body_decoder_model *m,
                                     int layer_idx,
                                     const float *x_in,
                                     const float *context_in,
                                     const float *x_pe_in,
                                     const float *context_pe_in,
                                     int N_q, int N_c,
                                     int n_threads,
                                     float *x_out,
                                     float *context_out);

/* Apply decoder.norm_final to the 145-token stack. Out same shape. */
int sam3d_body_norm_final(const sam3d_body_decoder_model *m,
                          const float *x_in,    /* (N_q, 1024) */
                          int N_q,
                          int n_threads,
                          float *x_out);

/*
 * Step 4g-ii: end-to-end decoder forward using PRESET per-layer pose
 * outputs. This bypasses the per-layer head_pose / camera_project /
 * _full_to_crop chain (which would require running MHR forward 5x — see
 * the iterative `forward_decoder_with_mhr` path planned for v1.1) and
 * instead consumes the (kp2d_cropped, kp2d_depth, kp3d) triplets the
 * caller has already computed (e.g. from the reference dumps for
 * verification, or from a separate MHR runner).
 *
 *   image_emb_chw    (kv_dim=1280, H, W) f32 — POST-ray_cond image
 *                    embeddings, channel-major. Used by both cross-attn
 *                    (after flatten + permute) and grid_sample inside
 *                    kp_token_update.
 *   image_pe_chw     (kv_dim=1280, H, W) f32 — image_augment (PE) channel
 *                    -major; flattened/permuted internally.
 *   initial_tokens   (N_q=145, D=1024) f32 — output of build_tokens.
 *   initial_augment  (N_q=145, D=1024) f32 — token_augment from build_tokens.
 *   kp2d_cropped_pl  (n_layers-1, K=70, 2)  f32 — per intermediate layer.
 *   kp2d_depth_pl    (n_layers-1, K=70)     f32
 *   kp3d_pl          (n_layers-1, K=70, 3)  f32 — POST-flip camera frame.
 *   out              filled with mhr_params (519) + cam_t (3); cam_t here
 *                    is just out_cam_raw (head_camera.proj output + init_camera);
 *                    the perspective_projection cam_t lives in the per-layer
 *                    pose_output instead.
 */
int sam3d_body_decoder_forward_preset(
    const sam3d_body_decoder_model *m,
    const float *image_emb_chw,
    const float *image_pe_chw,
    int H, int W,
    const float *initial_tokens,
    const float *initial_augment,
    const float *kp2d_cropped_pl,
    const float *kp2d_depth_pl,
    const float *kp3d_pl,
    int n_threads,
    sam3d_body_decoder_result *out);

/*
 * Step 4g-i: between-layer keypoint-token update (2D + 3D combined).
 *
 * Mirrors keypoint_token_update_fn_comb in upstream sam3d_body.py — runs
 * keypoint_token_update_fn (2D, kps_emb_start_idx=5) then
 * keypoint3d_token_update_fn (3D, kps3d_emb_start_idx=75).
 *
 *   layer_idx       per-layer index. The last layer (layer_idx == n_layers-1)
 *                   short-circuits to a pass-through (matches upstream
 *                   guard).
 *   image_emb       (kv_dim=1280, H, W) f32 — image embeddings AFTER ray_cond
 *                   (the same tensor passed as decoder context). Channel-major.
 *   H, W            spatial dims (H=W=32 for DINOv3 H+).
 *   kp2d_cropped    (70, 2) f32 — pose_output["pred_keypoints_2d_cropped"]
 *                   in [-0.5, 0.5] (output of _full_to_crop minus 0.5).
 *   kp2d_depth      (70,)   f32 — pose_output["pred_keypoints_2d_depth"]
 *                   (z-component before perspective divide; used to mask
 *                   behind-camera points with depth < 1e-5).
 *   kp3d_camera     (70, 3) f32 — pose_output["pred_keypoints_3d"], the
 *                   POST-flip (camera-frame) 3D points.
 *   N_q             token count (145 in body branch).
 *   tokens          (N_q, dim=1024) f32 in/out — kp_feat_linear feature
 *                   ADDED into rows [5..75); other rows untouched.
 *   token_augment   (N_q, dim=1024) f32 in/out — kp_posemb_linear written
 *                   to rows [5..75) (zeroed where invalid_mask is true);
 *                   kp3d_posemb_linear written to rows [75..145). Other
 *                   rows untouched.
 */
int sam3d_body_kp_token_update(const sam3d_body_decoder_model *m,
                               int layer_idx,
                               const float *image_emb,
                               int H, int W,
                               const float *kp2d_cropped,
                               const float *kp2d_depth,
                               const float *kp3d_camera,
                               int N_q,
                               int n_threads,
                               float *tokens,
                               float *token_augment);

/*
 * Apply the MHR pose + camera regression heads on the pose-token row of
 * the norm_final-output token stack (upstream: head_pose.proj(tokens[:,0])
 * and head_camera.proj(tokens[:,0]), each a 2-layer FFN(1024→1024→N) with
 * GELU between). Output is the "raw" projection (before + init_pose /
 * + init_camera), matching head_pose_proj_raw.npy / head_camera_proj_raw.npy.
 *
 *   pose_token:     (1024,) f32 — tokens_norm[0]
 *   out_pose_raw:   (519,) f32
 *   out_cam_raw:    (3,)   f32
 */
int sam3d_body_apply_heads_raw(const sam3d_body_decoder_model *m,
                               const float *pose_token,
                               int n_threads,
                               float *out_pose_raw,
                               float *out_cam_raw);

/*
 * Decode head_pose.proj raw output into MHR inputs (stages 1-5).
 *
 *   pose_raw: (519,) f32 = head_pose.proj raw + init_pose
 *
 *   pred_pose_raw breakdown:
 *     [0..6)     global_rot_6d
 *     [6..266)   pred_pose_cont (260)
 *     [266..311) shape (45)
 *     [311..339) scale (28)
 *     [339..447) hand (108) — left+right PCA-6D ×54
 *     [447..519) face (72) — zeroed in body-only path
 *
 * Outputs (caller-allocated):
 *   out_mhr_model_params (204): cat(full_pose_params(136), scales(68))
 *   out_shape            (45):  identity blend coefficients (= pred_shape)
 *   out_face             (72):  expression blend (zeros in body-only path)
 *
 * Per upstream MHRHead.forward:
 *   global_rot_euler = rotmat_to_euler("ZYX", rot6d_to_rotmat(global_rot_6d))
 *   pred_pose_euler  = compact_cont_to_model_params_body(pred_pose_cont)
 *                      with hand-mask + jaw zeroed
 *   full_pose_params = cat(zeros(3), global_rot_euler, pred_pose_euler[:130])
 *                      shape (136,); replace_hands_in_pose drops in hand
 *                      PCA decode at hand_idx_left / hand_idx_right.
 *   scales           = scale_mean + pred_scale @ scale_comps  shape (68,)
 *   model_params     = cat(full_pose_params, scales)          shape (204,)
 */
int sam3d_body_decode_pose_raw(const sam3d_body_decoder_model *m,
                               const float *pose_raw,
                               int enable_hand_model,
                               float *out_mhr_model_params,
                               float *out_shape,
                               float *out_face);

/*
 * Stage 12: 70-keypoint regression from mesh + joint coords.
 *
 *   verts:        (V=18439, 3) f32 — pre-camera-flip vertices (cm→m already)
 *   joint_coords: (J=127,   3) f32 — pre-camera-flip joint coords (m)
 *   out_kpts:     (70, 3)     f32 — flipped on (y,z) to camera system
 *
 * keypoint_mapping shape: (308, 18566) where 18566 = V + J. We compute:
 *   K = keypoint_mapping @ cat(verts, joint_coords)   → (308, 3)
 * then take the first 70 rows and flip y,z signs.
 */
int sam3d_body_keypoints_from_mesh(const sam3d_body_decoder_model *m,
                                   const float *verts_m,
                                   const float *joint_coords_m,
                                   int enable_hand_model,
                                   int n_threads,
                                   float *out_kpts);

/*
 * Per-image batch parameters used by camera_project (mirrors the
 * subset of `batch` consumed by PerspectiveHead.perspective_projection
 * and BaseModel._full_to_crop). All in original-image coords unless
 * noted; row-major.
 */
typedef struct {
    float cam_int[9];           /* (3,3) intrinsic matrix */
    float bbox_center[2];       /* upstream `batch["bbox_center"]` row */
    float bbox_scale;           /* upstream `batch["bbox_scale"][..., 0]` */
    float ori_img_size[2];      /* original image (W, H) — currently unused
                                 * by perspective_projection but kept for
                                 * symmetry with upstream batch dict */
    float img_size[2];          /* model input (W, H) — used by
                                 * perspective_projection to compute (cx, cy)
                                 * and by _full_to_crop for /img_size − 0.5 */
    float affine_trans[6];      /* (2,3) row-major; full→crop affine */
    int   use_intrin_center;    /* upstream cfg.MODEL.DECODER.USE_INTRIN_CENTER */
    float default_scale_factor; /* PerspectiveHead.default_scale_factor (1.0) */
} sam3d_body_camera_batch;

/*
 * Apply PerspectiveHead.perspective_projection (cam_t scale-translate
 * + j3d_cam = j3d + cam_t + project via cam_int) followed by
 * BaseModel._full_to_crop on the projected 2D points.
 *
 *   j3d_post_flip:    (K, 3) f32 — pred_keypoints_3d from MHRHead
 *                     (already y/z flipped to camera system); pre-cam_t.
 *   pred_cam:         (3,)   f32 — head_camera output + init_camera =
 *                     (s, tx, ty); function applies the (-1, +1, -1)
 *                     channel flip internally before use.
 *   batch:            per-image batch params.
 *   K:                number of keypoints (70 for body).
 *   out_kp2d:         (K, 2) f32 — 2D in original-image coords (output
 *                     of perspective_projection before _full_to_crop).
 *   out_kp2d_cropped: (K, 2) f32 — _full_to_crop output, in [-0.5, 0.5].
 *   out_kp2d_depth:   (K,)   f32 — z component of (j3d + cam_t).
 *   out_pred_cam_t:   (3,)   f32 — final camera translation [tx+cx, ty+cy, tz].
 *
 * Any of the four out pointers may be NULL to skip that output.
 */
int sam3d_body_camera_project(const float *j3d_post_flip,
                              const float *pred_cam,
                              const sam3d_body_camera_batch *batch,
                              int K,
                              float *out_kp2d,
                              float *out_kp2d_cropped,
                              float *out_kp2d_depth,
                              float *out_pred_cam_t);

/*
 * Forward decl of MHR assets (full definition in sam3d_body_mhr.h).
 * Callers of sam3d_body_decoder_forward_full must include sam3d_body_mhr.h
 * and ensure SAM3D_BODY_MHR_IMPLEMENTATION is defined in some TU so the
 * symbol `sam3d_body_mhr_forward` is linked.
 */
struct sam3d_body_mhr_assets_t;

typedef struct {
    /* Final regression outputs (post add_init). */
    float mhr_params[519];
    float cam_t[3];                /* head_camera output + init_camera (s,tx,ty) */
    /* Final 70-keypoint outputs (cam frame, post-flip; 2D in orig-image coords). */
    float pred_keypoints_3d[70 * 3];        /* m, post-flip cam frame */
    float pred_keypoints_2d[70 * 2];        /* original-image (W,H) px */
    float pred_keypoints_2d_cropped[70 * 2];/* crop, in [-0.5, 0.5] */
    float pred_keypoints_2d_depth[70];       /* z component (m) */
    float pred_cam_t_world[3];               /* (tx+cx, ty+cy, tz) */
    /* Optional caller-allocated buffers (NULL OK to skip). */
    float *pred_vertices;          /* (V*3,) post-flip cam frame; V=18439, m */
    float *mhr_model_params;       /* (204,) */
    float *shape;                  /* (45,) */
    float *face;                   /* (72,) */
    float *global_skel;            /* (J*8,) pre-flip cm; J=127 */
    /* Optional per-layer (li=0..n_layers-1) debug dumps; each should point
     * to (n_layers * <size>) floats or be NULL. Only intermediate layers
     * 0..n_layers-2 are written (final layer skips MHR-in-loop). */
    float *dbg_layer_kp3d;          /* (n_layers, 70, 3) post-flip m */
    float *dbg_layer_kp2d_crop;     /* (n_layers, 70, 2) */
    float *dbg_layer_kp2d_depth;    /* (n_layers, 70) */
    float *dbg_layer_cam_t;         /* (n_layers, 3) cam_t (tx+cx, ty+cy, tz) */
    float *dbg_layer_tokens_out;    /* (n_layers, N_Q, D) post-decoder_layer, pre-kp_update */
    float *dbg_layer_pose_raw;      /* (n_layers, 519) raw head_pose.proj at intermediate */
} sam3d_body_decoder_full_result;

/*
 * Iterative end-to-end decoder forward (production path; no preset MHR
 * intermediates). Mirrors meta_arch.sam3d_body.forward_decoder + the
 * 6-iteration body branch of MHRHead.forward.
 *
 * Per layer (layer < n_layers-1):
 *   tokens       = decoder_layer_forward(tokens, x_pe, image_emb, image_pe)
 *   tokens_norm  = norm_final(tokens)
 *   pose_raw,cam = apply_heads_raw(tokens_norm[0])
 *   model_params = decode_pose_raw(pose_raw + init_pose)
 *   verts_cm,gskel_cm = mhr_forward(model_params, shape, face)
 *   kp3d_m       = keypoints_from_mesh(verts_cm/100, gskel_cm[:,0:3]/100)
 *   kp2d_*       = camera_project(kp3d_m, cam + init_camera)
 *   tokens,x_pe  = kp_token_update(tokens, x_pe, kp2d_cropped, kp2d_depth, kp3d_m)
 *
 * Final layer skips kp_token_update; final pose_token feeds final
 * apply_heads_raw + MHR forward + keypoints_from_mesh + camera_project,
 * populating `out`.
 *
 *   image_emb_chw (kv_dim, H, W) post-ray_cond.
 *   image_pe_chw  (kv_dim, H, W) image PE.
 *   initial_tokens (145, 1024) from build_tokens.
 *   initial_augment (145, 1024) from build_tokens.
 */
int sam3d_body_decoder_forward_full(
    const sam3d_body_decoder_model *m,
    const struct sam3d_body_mhr_assets_t *mhr,
    const sam3d_body_camera_batch *cam_batch,
    const float *image_emb_chw,
    const float *image_pe_chw,
    int H, int W,
    const float *initial_tokens,
    const float *initial_augment,
    int n_threads,
    sam3d_body_decoder_full_result *out);

/*
 * sam3d_body_decoder_forward — self-driven body-branch entry point.
 *
 * Bundles the four assembly steps (ray_cond_emb + get_dense_pe +
 * compute_condition_info + build_tokens) and delegates to
 * sam3d_body_decoder_forward_full. Identical math to the iterative path
 * sam3d_body_runner.c uses, packaged so callers that already have raw
 * DINOv3 patch tokens + a camera_batch can skip the runner glue.
 *
 *   image_tokens_patch: (H*W, kv_dim=1280) f32 — DINOv3 patch tokens with
 *                       CLS + register tokens already stripped.
 *   H, W:               patch grid (32×32 for dinov3, 24×32 for vith).
 *   rays_hwc:           (H, W, 3) f32 — (dx, dy, 1) per spatial after the
 *                       encoder downsample + z-append.
 *   condition_info:     (3,) f32 — CLIFF-style bbox/cam encoding.
 *
 * Defined when SAM3D_BODY_DECODER_FULL_IMPLEMENTATION is set.
 * out->pred_vertices / mhr_model_params / shape / face / global_skel /
 * dbg_layer_* may be set by the caller; NULL skips them.
 */
int sam3d_body_decoder_forward(
    const sam3d_body_decoder_model *m,
    const struct sam3d_body_mhr_assets_t *mhr,
    const sam3d_body_camera_batch *cam_batch,
    const float *image_tokens_patch,
    int H, int W,
    const float *rays_hwc,
    const float *condition_info,
    int n_threads,
    sam3d_body_decoder_full_result *out);

#ifdef __cplusplus
}
#endif

/* ==================================================================== */
#ifdef SAM3D_BODY_DECODER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SAFETENSORS_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION 1
#endif
#include "safetensors.h"

/* MHR API (no impl) — needed by sam3d_body_decoder_forward_full, which
 * is gated behind SAM3D_BODY_DECODER_FULL_IMPLEMENTATION so consumers that
 * don't need the MHR-in-the-loop path don't pull `sam3d_body_mhr_forward`
 * into the link. The caller TU must define both
 * SAM3D_BODY_DECODER_FULL_IMPLEMENTATION and SAM3D_BODY_MHR_IMPLEMENTATION. */
#ifdef SAM3D_BODY_DECODER_FULL_IMPLEMENTATION
#include "sam3d_body_mhr.h"
#endif

static int s3db_find_f32(st_context *st, const char *name, qtensor *out)
{
    int i = safetensors_find(st, name);
    if (i < 0) return -1;
    const char *dt = safetensors_dtype(st, i);
    if (strcmp(dt, "F32") != 0) {
        fprintf(stderr, "sam3d_body_decoder: %s unexpected dtype %s (want F32)\n",
                name, dt);
        return -1;
    }
    out->data = (void *)safetensors_data(st, i);
    out->type = 1 /* FP32 */;
    int nd = safetensors_ndims(st, i);
    const uint64_t *shape = safetensors_shape(st, i);
    out->n_dims = nd;
    for (int d = 0; d < nd && d < 4; d++) out->dims[d] = shape[d];
    out->n_rows = (nd >= 1) ? (int)out->dims[0] : 1;
    out->n_cols = (nd >= 2) ? (int)out->dims[1] : 1;
    return 0;
}

/* Small wrapper that records the key but doesn't error on missing. */
static int s3db_find_f32_opt(st_context *st, const char *name, qtensor *out)
{
    int i = safetensors_find(st, name);
    if (i < 0) { memset(out, 0, sizeof(*out)); return -1; }
    return s3db_find_f32(st, name, out);
}

static int s3db_find_i64(st_context *st, const char *name, qtensor *out)
{
    int i = safetensors_find(st, name);
    if (i < 0) return -1;
    const char *dt = safetensors_dtype(st, i);
    if (strcmp(dt, "I64") != 0) {
        fprintf(stderr, "sam3d_body_decoder: %s unexpected dtype %s (want I64)\n",
                name, dt);
        return -1;
    }
    out->data = (void *)safetensors_data(st, i);
    out->type = 0 /* not a quant type; raw int */;
    int nd = safetensors_ndims(st, i);
    const uint64_t *shape = safetensors_shape(st, i);
    out->n_dims = nd;
    for (int d = 0; d < nd && d < 4; d++) out->dims[d] = shape[d];
    out->n_rows = (nd >= 1) ? (int)out->dims[0] : 1;
    out->n_cols = (nd >= 2) ? (int)out->dims[1] : 1;
    return 0;
}

sam3d_body_decoder_model *sam3d_body_decoder_load(const char *decoder_sft,
                                                  const char *mhr_head_sft)
{
    if (!decoder_sft || !mhr_head_sft) return NULL;

    sam3d_body_decoder_model *m =
        (sam3d_body_decoder_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;

    m->n_layers = 6;
    m->dim = 1024;
    m->kv_dim = 1280;
    m->n_heads = 8;
    m->head_dim = 64;
    m->ffn_hidden = 1024;
    m->n_keypoints = 70;
    m->n_hand_tokens = 2;
    m->npose = 519;
    m->ncam = 3;
    m->cond_dim = 3;
    m->ln_eps = 1e-6f;

    st_context *dec = safetensors_open(decoder_sft);
    st_context *mhr = safetensors_open(mhr_head_sft);
    if (!dec || !mhr) {
        fprintf(stderr, "sam3d_body_decoder: safetensors_open failed (dec=%p mhr=%p)\n",
                (void *)dec, (void *)mhr);
        if (dec) safetensors_close(dec);
        if (mhr) safetensors_close(mhr);
        free(m);
        return NULL;
    }
    m->_st_decoder = dec;
    m->_st_mhr_head = mhr;

    /* Per-layer weights are loaded in later substeps (4d..4g). The
     * ray_cond sub-module (step 4c) is loaded eagerly since it's the
     * first callable stage. */
    m->layers = (sam3d_body_decoder_layer *)calloc((size_t)m->n_layers,
                                                   sizeof(*m->layers));
    if (!m->layers) { sam3d_body_decoder_free(m); return NULL; }

    int bad = 0;
    bad |= s3db_find_f32(dec, "ray_cond_emb.conv.weight", &m->ray_cond_conv_w);
    bad |= s3db_find_f32(dec, "ray_cond_emb.norm.weight", &m->ray_cond_norm_w);
    bad |= s3db_find_f32(dec, "ray_cond_emb.norm.bias",   &m->ray_cond_norm_b);
    if (bad) {
        fprintf(stderr, "sam3d_body_decoder: ray_cond_emb weights missing\n");
        sam3d_body_decoder_free(m);
        return NULL;
    }
    /* conv has bias=False — leave ray_cond_conv_b zeroed. */

    /* ---- Token-construction weights (step 4d). ---- */
    bad  = s3db_find_f32(dec, "init_to_token_mhr.weight",  &m->init_to_token_w);
    bad |= s3db_find_f32(dec, "init_to_token_mhr.bias",    &m->init_to_token_b);
    bad |= s3db_find_f32(dec, "prev_to_token_mhr.weight",  &m->prev_to_token_w);
    bad |= s3db_find_f32(dec, "prev_to_token_mhr.bias",    &m->prev_to_token_b);
    bad |= s3db_find_f32(dec, "prompt_to_token.weight",    &m->prompt_to_token_w);
    bad |= s3db_find_f32(dec, "prompt_to_token.bias",      &m->prompt_to_token_b);
    bad |= s3db_find_f32(dec, "keypoint_embedding.weight",   &m->keypoint_embedding);
    bad |= s3db_find_f32(dec, "keypoint3d_embedding.weight", &m->keypoint3d_embedding);
    bad |= s3db_find_f32(dec, "hand_box_embedding.weight",   &m->hand_box_embedding);
    bad |= s3db_find_f32(dec, "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
                        &m->prompt_pe_gauss);
    bad |= s3db_find_f32(dec, "prompt_encoder.invalid_point_embed.weight",
                        &m->invalid_point_embed);
    /* hand_pe is optional in v1 body-only path. */
    s3db_find_f32_opt(dec, "hand_pe_layer.positional_encoding_gaussian_matrix",
                      &m->hand_pe_gauss);
    if (bad) {
        fprintf(stderr, "sam3d_body_decoder: token-construction weights missing\n");
        sam3d_body_decoder_free(m);
        return NULL;
    }

    /* ---- Keypoint-token update weights (step 4f, used by per-layer
     *      keypoint_token_update_fn). ---- */
    bad  = s3db_find_f32(dec, "keypoint_feat_linear.weight",                &m->kp_feat_linear_w);
    bad |= s3db_find_f32(dec, "keypoint_feat_linear.bias",                  &m->kp_feat_linear_b);
    bad |= s3db_find_f32(dec, "keypoint_posemb_linear.layers.0.0.weight",   &m->kp_posemb_l0_w);
    bad |= s3db_find_f32(dec, "keypoint_posemb_linear.layers.0.0.bias",     &m->kp_posemb_l0_b);
    bad |= s3db_find_f32(dec, "keypoint_posemb_linear.layers.1.weight",     &m->kp_posemb_l1_w);
    bad |= s3db_find_f32(dec, "keypoint_posemb_linear.layers.1.bias",       &m->kp_posemb_l1_b);
    bad |= s3db_find_f32(dec, "keypoint3d_posemb_linear.layers.0.0.weight", &m->kp3d_posemb_l0_w);
    bad |= s3db_find_f32(dec, "keypoint3d_posemb_linear.layers.0.0.bias",   &m->kp3d_posemb_l0_b);
    bad |= s3db_find_f32(dec, "keypoint3d_posemb_linear.layers.1.weight",   &m->kp3d_posemb_l1_w);
    bad |= s3db_find_f32(dec, "keypoint3d_posemb_linear.layers.1.bias",     &m->kp3d_posemb_l1_b);
    if (bad) {
        fprintf(stderr, "sam3d_body_decoder: kp-token-update weights missing\n");
        sam3d_body_decoder_free(m);
        return NULL;
    }

    /* ---- Per-layer TransformerDecoderLayer weights (step 4e). ---- */
    for (int li = 0; li < m->n_layers; li++) {
        sam3d_body_decoder_layer *L = &m->layers[li];
        char p[192];
#define FIND_LN(prefix, fld_w, fld_b)                                                  \
        do {                                                                           \
            snprintf(p, sizeof(p), "decoder.layers.%d." prefix ".weight", li);         \
            if (s3db_find_f32(dec, p, &L->fld_w)) { bad = 1; break; }                  \
            snprintf(p, sizeof(p), "decoder.layers.%d." prefix ".bias", li);           \
            if (s3db_find_f32(dec, p, &L->fld_b)) { bad = 1; break; }                  \
        } while (0)
#define FIND_PROJ(prefix, fld_w, fld_b)                                                \
        do {                                                                           \
            snprintf(p, sizeof(p), "decoder.layers.%d." prefix ".weight", li);         \
            if (s3db_find_f32(dec, p, &L->fld_w)) { bad = 1; break; }                  \
            snprintf(p, sizeof(p), "decoder.layers.%d." prefix ".bias", li);           \
            if (s3db_find_f32(dec, p, &L->fld_b)) { bad = 1; break; }                  \
        } while (0)

        FIND_LN("ln1",     ln1_w,     ln1_b);
        FIND_LN("ln_pe_1", ln_pe_1_w, ln_pe_1_b);
        FIND_LN("ln2_1",   ln2_1_w,   ln2_1_b);
        FIND_LN("ln2_2",   ln2_2_w,   ln2_2_b);
        FIND_LN("ln_pe_2", ln_pe_2_w, ln_pe_2_b);
        FIND_LN("ln3",     ln3_w,     ln3_b);

        FIND_PROJ("self_attn.q_proj",  sa_q_w,    sa_q_b);
        FIND_PROJ("self_attn.k_proj",  sa_k_w,    sa_k_b);
        FIND_PROJ("self_attn.v_proj",  sa_v_w,    sa_v_b);
        FIND_PROJ("self_attn.proj",    sa_proj_w, sa_proj_b);

        FIND_PROJ("cross_attn.q_proj", ca_q_w,    ca_q_b);
        FIND_PROJ("cross_attn.k_proj", ca_k_w,    ca_k_b);
        FIND_PROJ("cross_attn.v_proj", ca_v_w,    ca_v_b);
        FIND_PROJ("cross_attn.proj",   ca_proj_w, ca_proj_b);

        FIND_PROJ("ffn.layers.0.0", ffn0_w, ffn0_b);
        FIND_PROJ("ffn.layers.1",   ffn1_w, ffn1_b);
#undef FIND_LN
#undef FIND_PROJ
        if (bad) break;
    }
    if (bad) {
        fprintf(stderr, "sam3d_body_decoder: per-layer weights missing\n");
        sam3d_body_decoder_free(m);
        return NULL;
    }

    /* Final LN on tokens after last layer. */
    bad  = s3db_find_f32(dec, "decoder.norm_final.weight", &m->norm_final_w);
    bad |= s3db_find_f32(dec, "decoder.norm_final.bias",   &m->norm_final_b);
    if (bad) {
        fprintf(stderr, "sam3d_body_decoder: decoder.norm_final weights missing\n");
        sam3d_body_decoder_free(m);
        return NULL;
    }

    /* ---- MHR head regression weights (step 4f). ---- */
    bad  = s3db_find_f32(mhr, "head_pose.proj.layers.0.0.weight", &m->head_pose_l0_w);
    bad |= s3db_find_f32(mhr, "head_pose.proj.layers.0.0.bias",   &m->head_pose_l0_b);
    bad |= s3db_find_f32(mhr, "head_pose.proj.layers.1.weight",   &m->head_pose_l1_w);
    bad |= s3db_find_f32(mhr, "head_pose.proj.layers.1.bias",     &m->head_pose_l1_b);
    bad |= s3db_find_f32(mhr, "head_camera.proj.layers.0.0.weight", &m->head_camera_l0_w);
    bad |= s3db_find_f32(mhr, "head_camera.proj.layers.0.0.bias",   &m->head_camera_l0_b);
    bad |= s3db_find_f32(mhr, "head_camera.proj.layers.1.weight",   &m->head_camera_l1_w);
    bad |= s3db_find_f32(mhr, "head_camera.proj.layers.1.bias",     &m->head_camera_l1_b);
    bad |= s3db_find_f32(mhr, "init_pose.weight",   &m->init_pose);
    bad |= s3db_find_f32(mhr, "init_camera.weight", &m->init_camera);
    if (bad) {
        fprintf(stderr, "sam3d_body_decoder: MHR head regression weights missing\n");
        sam3d_body_decoder_free(m);
        return NULL;
    }

    /* MHR head PCA + keypoint assets (stages 1-5 + 12). */
    bad  = s3db_find_f32(mhr, "head_pose.scale_mean",       &m->scale_mean);
    bad |= s3db_find_f32(mhr, "head_pose.scale_comps",      &m->scale_comps);
    bad |= s3db_find_f32(mhr, "head_pose.hand_pose_mean",   &m->hand_pose_mean);
    bad |= s3db_find_f32(mhr, "head_pose.hand_pose_comps",  &m->hand_pose_comps);
    bad |= s3db_find_i64(mhr, "head_pose.hand_joint_idxs_left",  &m->hand_idx_left);
    bad |= s3db_find_i64(mhr, "head_pose.hand_joint_idxs_right", &m->hand_idx_right);
    bad |= s3db_find_f32(mhr, "head_pose.keypoint_mapping", &m->keypoint_mapping);
    bad |= s3db_find_f32(mhr, "head_pose.local_to_world_wrist", &m->local_to_world_wrist);
    bad |= s3db_find_f32(mhr, "head_pose.right_wrist_coords",   &m->right_wrist_coords);
    bad |= s3db_find_f32(mhr, "head_pose.root_coords",          &m->root_coords);
    bad |= s3db_find_i64(mhr, "head_pose.nonhand_param_idxs",   &m->nonhand_param_idxs);
    bad |= s3db_find_i64(mhr, "head_pose.faces",                &m->faces);
    if (bad) {
        fprintf(stderr, "sam3d_body_decoder: MHR head PCA/keypoint assets missing\n");
        sam3d_body_decoder_free(m);
        return NULL;
    }

    fprintf(stderr, "sam3d_body_decoder: loaded "
                    "(decoder=%s mhr_head=%s) — ray_cond_emb + tokens + "
                    "%d layers + norm_final + MHR regression heads ready\n",
            decoder_sft, mhr_head_sft, m->n_layers);
    return m;
}

void sam3d_body_decoder_free(sam3d_body_decoder_model *m)
{
    if (!m) return;
    if (m->_st_decoder) safetensors_close((st_context *)m->_st_decoder);
    if (m->_st_mhr_head) safetensors_close((st_context *)m->_st_mhr_head);
    free(m->layers);
    free(m);
}

/* ---- ray_cond_emb forward (step 4c) ------------------------------- */

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Fills `out` (N_fourier = 3 + 2*3*num_bands = 99) per spatial position.
 * Bands: linspace(1, max_res/2=32, 16). Layout per position:
 *   [pos_x, pos_y, pos_z,
 *    sin(π·x·b0..15), sin(π·y·b0..15), sin(π·z·b0..15),
 *    cos(π·x·b0..15), cos(π·y·b0..15), cos(π·z·b0..15)]  — matches upstream
 */
static void s3db_fourier_pos_encoding(const float *rays_xyz,
                                      int H, int W, int num_bands,
                                      float *out /* (H*W, 3 + 6*num_bands) */)
{
    const int dim_out = 3 + 2 * 3 * num_bands;
    const int N = H * W;
    /* Linear frequency band table: start=1, end=max_res/2=32. */
    float bands[32]; /* up to 32 bands — more than enough for num_bands=16 */
    const float end = 32.0f;
    for (int b = 0; b < num_bands; b++) {
        bands[b] = (num_bands == 1) ? 1.0f
                                    : 1.0f + (end - 1.0f) * (float)b / (float)(num_bands - 1);
    }
    for (int n = 0; n < N; n++) {
        const float *p = rays_xyz + (size_t)n * 3;
        float *o = out + (size_t)n * dim_out;
        o[0] = p[0]; o[1] = p[1]; o[2] = p[2];
        /* 48 product values: axis-major then band-major. */
        float products[48];
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < num_bands; b++)
                products[a * num_bands + b] = p[a] * bands[b];
        float *sinv = o + 3;
        float *cosv = o + 3 + 3 * num_bands;
        for (int i = 0; i < 3 * num_bands; i++) {
            float v = (float)M_PI * products[i];
            sinv[i] = sinf(v);
            cosv[i] = cosf(v);
        }
    }
}

/* 1×1 conv ≡ matmul: out[c, n] = Σ_i W[c, i] * src[i, n].
 * Source layout: (C_in, N). Dest layout: (C_out, N). Weight: (C_out, C_in).
 * Parallelized over C_out. */
static void s3db_conv1x1_f32(const float *W, const float *src,
                             int C_out, int C_in, int N,
                             float *dst, int n_threads)
{
    (void)n_threads;
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int c = 0; c < C_out; c++) {
        const float *wrow = W + (size_t)c * C_in;
        float *drow = dst + (size_t)c * N;
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int i = 0; i < C_in; i++)
                acc += (double)wrow[i] * (double)src[(size_t)i * N + n];
            drow[n] = (float)acc;
        }
    }
}

/* LayerNorm2d (upstream custom): per-spatial over channels, (C,) gamma/beta.
 * in/out layout: (C, H*W). */
static void s3db_layer_norm2d(float *dst, const float *src,
                              const float *gamma, const float *beta,
                              int C, int N, float eps, int n_threads)
{
    (void)n_threads;
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int n = 0; n < N; n++) {
        /* mean along C at this spatial pos. Non-contiguous: stride N. */
        double sum = 0.0;
        for (int c = 0; c < C; c++) sum += (double)src[(size_t)c * N + n];
        float mean = (float)(sum / (double)C);
        double var = 0.0;
        for (int c = 0; c < C; c++) {
            float d = src[(size_t)c * N + n] - mean;
            var += (double)d * (double)d;
        }
        float inv = 1.0f / sqrtf((float)(var / (double)C) + eps);
        for (int c = 0; c < C; c++) {
            float v = (src[(size_t)c * N + n] - mean) * inv;
            dst[(size_t)c * N + n] = v * gamma[c] + beta[c];
        }
    }
}

int sam3d_body_ray_cond_emb_forward(const sam3d_body_decoder_model *m,
                                    const float *image_emb,
                                    const float *rays_xyz,
                                    int H, int W,
                                    int n_threads,
                                    float *out)
{
    if (!m || !image_emb || !rays_xyz || !out) return SAM3D_BODY_DECODER_E_INVAL;
    if (!m->ray_cond_conv_w.data) return SAM3D_BODY_DECODER_E_LOAD;
    const int C_img = m->kv_dim;          /* 1280 */
    const int num_bands = 16;
    const int C_fp = 3 + 2 * 3 * num_bands; /* 99 */
    const int C_in = C_img + C_fp;        /* 1379 */
    const int N = H * W;

    /* 1. Fourier features: (N, 99) */
    float *fp_packed = (float *)malloc((size_t)N * (size_t)C_fp * sizeof(float));
    if (!fp_packed) return SAM3D_BODY_DECODER_E_LOAD;
    s3db_fourier_pos_encoding(rays_xyz, H, W, num_bands, fp_packed);

    /* 2. Transpose fourier to (C_fp, N). */
    float *fp = (float *)malloc((size_t)C_fp * (size_t)N * sizeof(float));
    if (!fp) { free(fp_packed); return SAM3D_BODY_DECODER_E_LOAD; }
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C_fp; c++)
            fp[(size_t)c * N + n] = fp_packed[(size_t)n * C_fp + c];
    free(fp_packed);

    /* 3. Concat along C: preconv = [image_emb (C_img, N); fp (C_fp, N)] */
    float *preconv = (float *)malloc((size_t)C_in * (size_t)N * sizeof(float));
    if (!preconv) { free(fp); return SAM3D_BODY_DECODER_E_LOAD; }
    memcpy(preconv,
           image_emb,
           (size_t)C_img * (size_t)N * sizeof(float));
    memcpy(preconv + (size_t)C_img * N,
           fp,
           (size_t)C_fp  * (size_t)N * sizeof(float));
    free(fp);

    /* 4. 1×1 conv (no bias). W is (C_img, C_in). */
    float *postconv = (float *)malloc((size_t)C_img * (size_t)N * sizeof(float));
    if (!postconv) { free(preconv); return SAM3D_BODY_DECODER_E_LOAD; }
    s3db_conv1x1_f32((const float *)m->ray_cond_conv_w.data,
                     preconv, C_img, C_in, N, postconv, n_threads);
    free(preconv);

    /* 5. LayerNorm2d with learnable gamma/beta. */
    s3db_layer_norm2d(out, postconv,
                      (const float *)m->ray_cond_norm_w.data,
                      (const float *)m->ray_cond_norm_b.data,
                      C_img, N, m->ln_eps, n_threads);
    free(postconv);
    return SAM3D_BODY_DECODER_E_OK;
}

/* ---- SAM PE get_dense_pe + invalid prompt (step 8c-ii) ----------- */

int sam3d_body_get_dense_pe(const sam3d_body_decoder_model *m,
                            int H, int W, int n_threads, float *out)
{
    if (!m || !out || H <= 0 || W <= 0) return SAM3D_BODY_DECODER_E_INVAL;
    if (!m->prompt_pe_gauss.data) return SAM3D_BODY_DECODER_E_LOAD;
    /* gaussian_matrix is (2, num_pos_feats); embed_dim = 2 * num_pos_feats. */
    const int npf = (int)m->prompt_pe_gauss.dims[1];
    const float *G = (const float *)m->prompt_pe_gauss.data; /* (2, npf) */

    (void)n_threads;
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int yi = 0; yi < H; yi++) {
        const float yn = ((float)yi + 0.5f) / (float)H;        /* [0,1] */
        const float ys = 2.0f * yn - 1.0f;                     /* [-1,1] */
        const float x_offset = (W != H) ? 0.5f * (float)(H - W) : 0.0f;
        for (int xi = 0; xi < W; xi++) {
            const float xn = ((float)xi + x_offset + 0.5f) / (float)H;
            const float xs = 2.0f * xn - 1.0f;
            const float two_pi = 6.2831853071795864769f;
            for (int k = 0; k < npf; k++) {
                /* coords @ G  →  xs*G[0,k] + ys*G[1,k] */
                float v = xs * G[0 * npf + k] + ys * G[1 * npf + k];
                v *= two_pi;
                float s = sinf(v), c = cosf(v);
                /* CHW: out[k, yi, xi]  and out[npf+k, yi, xi]. */
                out[((size_t)k         * H + yi) * W + xi] = s;
                out[((size_t)(npf + k) * H + yi) * W + xi] = c;
            }
        }
    }
    return SAM3D_BODY_DECODER_E_OK;
}

int sam3d_body_invalid_prompt_token(const sam3d_body_decoder_model *m,
                                    float *out)
{
    if (!m || !out) return SAM3D_BODY_DECODER_E_INVAL;
    if (!m->invalid_point_embed.data) return SAM3D_BODY_DECODER_E_LOAD;
    const int D = m->kv_dim;  /* 1280 */
    /* Single dummy keypoint with label==-2: _pe_encoding output is
     * zeroed and overwritten with invalid_point_embed.weight (1, 1280). */
    memcpy(out, m->invalid_point_embed.data, (size_t)D * sizeof(float));
    return SAM3D_BODY_DECODER_E_OK;
}

/* ---- ray_cond builder (step 8c-ii) -------------------------------- */

/* 1D antialias-bilinear downsample (separable triangle filter).
 *
 * For a downsampling factor s = N_in / N_out > 1:
 *   - support = max(s, 1)
 *   - per output i_out: center = (i_out + 0.5) * s - 0.5
 *     window (input coords) = [center - support, center + support]
 *     weight w(j) = max(0, 1 - |center - j| / support) (triangle)
 *     out[i_out] = Σ_j w(j) * in[j] / Σ_j w(j)   (boundary clipping
 *                                                 sums over j ∈ [0, N_in))
 *
 * Mirrors aten/src/ATen/native/UpSample{Bilinear2d}.cpp antialias kernel.
 */
static void s3db_antialias_downsample_1d(const float *in, int N_in,
                                         float *out, int N_out)
{
    const double scale = (double)N_in / (double)N_out;
    const double support = (scale > 1.0) ? scale : 1.0;
    const double inv_support = 1.0 / support;

    for (int i_out = 0; i_out < N_out; i_out++) {
        const double center = ((double)i_out + 0.5) * scale - 0.5;
        int j_min = (int)floor(center - support + 1e-9);
        int j_max = (int)floor(center + support - 1e-9);
        if (j_min < 0) j_min = 0;
        if (j_max >= N_in) j_max = N_in - 1;

        double sum_w = 0.0, sum_wf = 0.0;
        for (int j = j_min; j <= j_max; j++) {
            double dist = fabs(center - (double)j) * inv_support;
            double w = 1.0 - dist;
            if (w <= 0.0) continue;
            sum_w  += w;
            sum_wf += w * (double)in[j];
        }
        out[i_out] = (sum_w > 0.0) ? (float)(sum_wf / sum_w) : 0.0f;
    }
}

int sam3d_body_compute_ray_cond_xyz(const float *cam_int,
                                    const float *affine_trans,
                                    int H_in, int W_in,
                                    int H_out, int W_out,
                                    float *out)
{
    if (!cam_int || !affine_trans || !out) return SAM3D_BODY_DECODER_E_INVAL;
    if (H_in <= 0 || W_in <= 0 || H_out <= 0 || W_out <= 0)
        return SAM3D_BODY_DECODER_E_INVAL;

    const float fx = cam_int[0 * 3 + 0];
    const float fy = cam_int[1 * 3 + 1];
    const float cx = cam_int[0 * 3 + 2];
    const float cy = cam_int[1 * 3 + 2];

    const float a00 = affine_trans[0 * 3 + 0];
    const float a11 = affine_trans[1 * 3 + 1];
    const float t02 = affine_trans[0 * 3 + 2];
    const float t12 = affine_trans[1 * 3 + 2];

    /* x_full[j] = (j/a00 - t02/a00 - cx) / fx,  j ∈ [0, W_in)
     * y_full[i] = (i/a11 - t12/a11 - cy) / fy,  i ∈ [0, H_in)
     * Both are linear ramps; apply 1D antialias separately. */
    float *x_full = (float *)malloc((size_t)W_in * sizeof(float));
    float *y_full = (float *)malloc((size_t)H_in * sizeof(float));
    if (!x_full || !y_full) { free(x_full); free(y_full); return SAM3D_BODY_DECODER_E_LOAD; }

    const double inv_a00 = 1.0 / (double)a00;
    const double inv_a11 = 1.0 / (double)a11;
    const double inv_fx  = 1.0 / (double)fx;
    const double inv_fy  = 1.0 / (double)fy;
    for (int j = 0; j < W_in; j++) {
        double xp = (double)j * inv_a00 - (double)t02 * inv_a00;
        x_full[j] = (float)((xp - (double)cx) * inv_fx);
    }
    for (int i = 0; i < H_in; i++) {
        double yp = (double)i * inv_a11 - (double)t12 * inv_a11;
        y_full[i] = (float)((yp - (double)cy) * inv_fy);
    }

    float *x_ds = (float *)malloc((size_t)W_out * sizeof(float));
    float *y_ds = (float *)malloc((size_t)H_out * sizeof(float));
    if (!x_ds || !y_ds) {
        free(x_full); free(y_full); free(x_ds); free(y_ds);
        return SAM3D_BODY_DECODER_E_LOAD;
    }
    const float *x_src = x_full;
    int x_src_n = W_in;
    if (W_out != H_out) {
        int crop_w = (int)lrint((double)W_in * (double)W_out / (double)H_out);
        if (crop_w > 0 && crop_w <= W_in) {
            int crop_x0 = (W_in - crop_w) / 2;
            x_src = x_full + crop_x0;
            x_src_n = crop_w;
        }
    }
    s3db_antialias_downsample_1d(x_src, x_src_n, x_ds, W_out);
    s3db_antialias_downsample_1d(y_full, H_in, y_ds, H_out);
    free(x_full); free(y_full);

    /* Pack as (H_out, W_out, 3) with z=1. */
    for (int yi = 0; yi < H_out; yi++) {
        for (int xi = 0; xi < W_out; xi++) {
            size_t base = ((size_t)yi * W_out + xi) * 3;
            out[base + 0] = x_ds[xi];
            out[base + 1] = y_ds[yi];
            out[base + 2] = 1.0f;
        }
    }
    free(x_ds); free(y_ds);
    return SAM3D_BODY_DECODER_E_OK;
}

void sam3d_body_default_cam_int(int W, int H, float *cam_int)
{
    if (!cam_int) return;
    const double f = sqrt((double)W * (double)W + (double)H * (double)H);
    cam_int[0] = (float)f; cam_int[1] = 0.0f;     cam_int[2] = 0.5f * (float)W;
    cam_int[3] = 0.0f;     cam_int[4] = (float)f; cam_int[5] = 0.5f * (float)H;
    cam_int[6] = 0.0f;     cam_int[7] = 0.0f;     cam_int[8] = 1.0f;
}

void sam3d_body_fix_aspect_ratio(const float *scale_in, float aspect_ratio,
                                 float *scale_out)
{
    if (!scale_in || !scale_out || aspect_ratio <= 0.0f) return;
    const float sw = scale_in[0], sh = scale_in[1];
    if (sw > sh * aspect_ratio) {
        scale_out[0] = sw;
        scale_out[1] = sw / aspect_ratio;
    } else {
        scale_out[0] = sh * aspect_ratio;
        scale_out[1] = sh;
    }
}

void sam3d_body_get_warp_matrix(const float *center, const float *scale,
                                int out_w, int out_h, float *warp_mat)
{
    if (!center || !scale || !warp_mat) return;
    const float s = (float)out_w / scale[0];
    warp_mat[0] = s;    warp_mat[1] = 0.0f; warp_mat[2] = 0.5f * (float)out_w - center[0] * s;
    warp_mat[3] = 0.0f; warp_mat[4] = s;    warp_mat[5] = 0.5f * (float)out_h - center[1] * s;
}

void sam3d_body_compute_condition_info(const float *bbox_center,
                                       const float *bbox_scale,
                                       const float *ori_img_size,
                                       const float *cam_int,
                                       int use_intrin_center,
                                       float *condition_info)
{
    if (!bbox_center || !bbox_scale || !ori_img_size || !cam_int ||
        !condition_info) return;
    const float f  = cam_int[0];
    const float cx = use_intrin_center ? cam_int[2] : 0.5f * ori_img_size[0];
    const float cy = use_intrin_center ? cam_int[5] : 0.5f * ori_img_size[1];
    condition_info[0] = (bbox_center[0] - cx) / f;
    condition_info[1] = (bbox_center[1] - cy) / f;
    condition_info[2] = bbox_scale[0] / f;
}

void sam3d_body_compute_bbox_affine(const float *bbox_xyxy,
                                    float padding,
                                    float aspect_ratio_pre,
                                    int out_w, int out_h,
                                    float *center,
                                    float *scale,
                                    float *warp_mat)
{
    if (!bbox_xyxy || !center || !scale || !warp_mat) return;
    center[0] = 0.5f * (bbox_xyxy[0] + bbox_xyxy[2]);
    center[1] = 0.5f * (bbox_xyxy[1] + bbox_xyxy[3]);
    float s_raw[2];
    s_raw[0] = (bbox_xyxy[2] - bbox_xyxy[0]) * padding;
    s_raw[1] = (bbox_xyxy[3] - bbox_xyxy[1]) * padding;
    float s1[2] = {0.0f, 0.0f};
    sam3d_body_fix_aspect_ratio(s_raw, aspect_ratio_pre, s1);
    const float ar2 = (out_h > 0) ? ((float)out_w / (float)out_h) : 1.0f;
    sam3d_body_fix_aspect_ratio(s1, ar2, scale);
    sam3d_body_get_warp_matrix(center, scale, out_w, out_h, warp_mat);
}

int sam3d_body_preprocess_image(const uint8_t *img_rgb_hwc,
                                int W_in, int H_in,
                                const float *warp_mat_2x3,
                                int W_out, int H_out,
                                float *out_chw)
{
    if (!img_rgb_hwc || !warp_mat_2x3 || !out_chw) return SAM3D_BODY_DECODER_E_INVAL;
    if (W_in <= 0 || H_in <= 0 || W_out <= 0 || H_out <= 0)
        return SAM3D_BODY_DECODER_E_INVAL;

    /* Invert 2x3 affine (dst→src). */
    const double a00 = warp_mat_2x3[0], a01 = warp_mat_2x3[1], a02 = warp_mat_2x3[2];
    const double a10 = warp_mat_2x3[3], a11 = warp_mat_2x3[4], a12 = warp_mat_2x3[5];
    const double det = a00 * a11 - a01 * a10;
    if (fabs(det) < 1e-12) return SAM3D_BODY_DECODER_E_INVAL;
    const double inv_det = 1.0 / det;
    const double i00 =  a11 * inv_det;
    const double i01 = -a01 * inv_det;
    const double i02 = (a01 * a12 - a02 * a11) * inv_det;
    const double i10 = -a10 * inv_det;
    const double i11 =  a00 * inv_det;
    const double i12 = (a02 * a10 - a00 * a12) * inv_det;

    const float mean_c[3] = { 0.485f, 0.456f, 0.406f };
    const float std_c [3] = { 0.229f, 0.224f, 0.225f };
    const float inv_std[3] = { 1.0f / std_c[0], 1.0f / std_c[1], 1.0f / std_c[2] };
    const float neg_mean_over_std[3] = {
        -mean_c[0] * inv_std[0],
        -mean_c[1] * inv_std[1],
        -mean_c[2] * inv_std[2],
    };
    const size_t plane = (size_t)H_out * (size_t)W_out;

    /* NOTE: we compute bilinear in double precision then round to uint8 (matching
     * the ToTensor()-followed-by-normalize path). cv2.warpAffine internally
     * quantizes subpixel coords to INTER_TAB_SIZE=32 with fixed-point AB_BITS=10
     * accumulation, which can differ from us by up to a couple of uint8 units on
     * rare pixels. Downstream DINOv3 runs in bf16, so the residual delta is
     * within model precision. */
    for (int y = 0; y < H_out; y++) {
        for (int x = 0; x < W_out; x++) {
            const double sx = i00 * (double)x + i01 * (double)y + i02;
            const double sy = i10 * (double)x + i11 * (double)y + i12;
            const int ix0 = (int)floor(sx);
            const int iy0 = (int)floor(sy);
            const int ix1 = ix0 + 1;
            const int iy1 = iy0 + 1;
            const float fx = (float)(sx - (double)ix0);
            const float fy = (float)(sy - (double)iy0);
            const float w00 = (1.0f - fx) * (1.0f - fy);
            const float w10 = fx * (1.0f - fy);
            const float w01 = (1.0f - fx) * fy;
            const float w11 = fx * fy;

            const int in00 = (ix0 >= 0 && ix0 < W_in && iy0 >= 0 && iy0 < H_in);
            const int in10 = (ix1 >= 0 && ix1 < W_in && iy0 >= 0 && iy0 < H_in);
            const int in01 = (ix0 >= 0 && ix0 < W_in && iy1 >= 0 && iy1 < H_in);
            const int in11 = (ix1 >= 0 && ix1 < W_in && iy1 >= 0 && iy1 < H_in);

            for (int c = 0; c < 3; c++) {
                float v = 0.0f;
                if (in00) v += w00 * (float)img_rgb_hwc[((size_t)iy0 * W_in + ix0) * 3 + c];
                if (in10) v += w10 * (float)img_rgb_hwc[((size_t)iy0 * W_in + ix1) * 3 + c];
                if (in01) v += w01 * (float)img_rgb_hwc[((size_t)iy1 * W_in + ix0) * 3 + c];
                if (in11) v += w11 * (float)img_rgb_hwc[((size_t)iy1 * W_in + ix1) * 3 + c];
                /* cv2.warpAffine returns uint8 — round then /255 to match
                 * the upstream VisionTransformWrapper(ToTensor()) path. */
                int iv = (int)floorf(v + 0.5f);
                if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
                const float norm = ((float)iv / 255.0f) * inv_std[c]
                                 + neg_mean_over_std[c];
                out_chw[(size_t)c * plane + (size_t)y * W_out + x] = norm;
            }
        }
    }
    return SAM3D_BODY_DECODER_E_OK;
}

/* ---- token construction (step 4d) ---------------------------------- */

/* out[d] = Σ_k W[d,k]*x[k] + b[d]. All f32. */
static void s3db_linear_f32(const float *W, const float *b,
                            const float *x, int D_out, int D_in,
                            float *out)
{
    for (int d = 0; d < D_out; d++) {
        const float *wrow = W + (size_t)d * D_in;
        double acc = b ? (double)b[d] : 0.0;
        for (int k = 0; k < D_in; k++) acc += (double)wrow[k] * (double)x[k];
        out[d] = (float)acc;
    }
}

int sam3d_body_build_tokens(const sam3d_body_decoder_model *m,
                            const float *init_input,
                            const float *prev_input,
                            const float *prompt_in,
                            int n_threads,
                            float *x_out,
                            float *x_pe_out)
{
    (void)n_threads;
    if (!m || !init_input || !prev_input || !prompt_in || !x_out || !x_pe_out)
        return SAM3D_BODY_DECODER_E_INVAL;
    if (!m->init_to_token_w.data || !m->prev_to_token_w.data ||
        !m->prompt_to_token_w.data || !m->keypoint_embedding.data ||
        !m->keypoint3d_embedding.data || !m->hand_box_embedding.data)
        return SAM3D_BODY_DECODER_E_LOAD;

    const int D   = m->dim;             /* 1024 */
    const int NKP = m->n_keypoints;     /* 70   */
    const int NHB = m->n_hand_tokens;   /* 2    */
    const int D_init   = m->npose + m->ncam + m->cond_dim;  /* 525 */
    const int D_prev   = m->npose + m->ncam;                /* 522 */
    const int D_prompt = m->kv_dim;                         /* 1280 */
    const int N_tok    = 1 + 1 + 1 + NHB + NKP + NKP;       /* 145 */
    (void)N_tok;

    /* Projections → 3 tokens of dim 1024. */
    float *pose_tok   = x_out + 0 * D;  /* slot 0 */
    float *prev_tok   = x_out + 1 * D;  /* slot 1 */
    float *prompt_tok = x_out + 2 * D;  /* slot 2 */

    s3db_linear_f32((const float *)m->init_to_token_w.data,
                    (const float *)m->init_to_token_b.data,
                    init_input, D, D_init, pose_tok);
    s3db_linear_f32((const float *)m->prev_to_token_w.data,
                    (const float *)m->prev_to_token_b.data,
                    prev_input, D, D_prev, prev_tok);
    s3db_linear_f32((const float *)m->prompt_to_token_w.data,
                    (const float *)m->prompt_to_token_b.data,
                    prompt_in, D, D_prompt, prompt_tok);

    /* Learned embeddings copied in. */
    memcpy(x_out + (1 + 1 + 1) * (size_t)D,
           m->hand_box_embedding.data,
           (size_t)NHB * (size_t)D * sizeof(float));
    memcpy(x_out + (1 + 1 + 1 + NHB) * (size_t)D,
           m->keypoint_embedding.data,
           (size_t)NKP * (size_t)D * sizeof(float));
    memcpy(x_out + (1 + 1 + 1 + NHB + NKP) * (size_t)D,
           m->keypoint3d_embedding.data,
           (size_t)NKP * (size_t)D * sizeof(float));

    /* x_pe (token_augment): zero everywhere except slot 1 (prev) and slot 2 (prompt). */
    memset(x_pe_out, 0, (size_t)(1 + 1 + 1 + NHB + NKP + NKP) * (size_t)D * sizeof(float));
    memcpy(x_pe_out + 1 * (size_t)D, prev_tok,   (size_t)D * sizeof(float));
    memcpy(x_pe_out + 2 * (size_t)D, prompt_tok, (size_t)D * sizeof(float));

    return SAM3D_BODY_DECODER_E_OK;
}

/* ---- TransformerDecoderLayer forward (step 4e) ---------------------- */

/* out[n, d] = b[d] + Σ_k W[d, k] * x[n, k].
 * Inputs: x (N, D_in), W (D_out, D_in), b (D_out) optional. */
static void s3db_linear_batch(float *out, const float *x, const float *W,
                              const float *b, int N, int D_in, int D_out,
                              int n_threads)
{
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D_out; d++) {
            const float *xrow = x + (size_t)n * D_in;
            const float *wrow = W + (size_t)d * D_in;
            double acc = b ? (double)b[d] : 0.0;
            for (int k = 0; k < D_in; k++) acc += (double)wrow[k] * (double)xrow[k];
            out[(size_t)n * D_out + d] = (float)acc;
        }
    }
}

/* LayerNorm (D channels per row), parallel over rows. */
static void s3db_layer_norm_batch(float *out, const float *x, int N, int D,
                                  const float *gamma, const float *beta, float eps,
                                  int n_threads)
{
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int n = 0; n < N; n++) {
        const float *xr = x + (size_t)n * D;
        double sum = 0.0;
        for (int i = 0; i < D; i++) sum += xr[i];
        float mean = (float)(sum / (double)D);
        double ssq = 0.0;
        for (int i = 0; i < D; i++) { double d = xr[i] - mean; ssq += d * d; }
        float invs = 1.0f / sqrtf((float)(ssq / (double)D) + eps);
        float *o = out + (size_t)n * D;
        for (int i = 0; i < D; i++) o[i] = (xr[i] - mean) * invs * gamma[i] + beta[i];
    }
}

/* GELU (exact, via erf) — torch default for nn.GELU. */
static inline float s3db_gelu_f32(float x)
{
    /* torch: x * 0.5 * (1 + erf(x / sqrt(2))) */
    return x * 0.5f * (1.0f + erff(x * 0.70710678118654752440f));
}

/* Scaled dot-product attention with B=1, H heads, head_dim D_h, emb = H*D_h.
 *   q (N_q, H*D_h), k (N_k, H*D_h), v (N_k, H*D_h), out (N_q, H*D_h).
 * Parallel over (h, n_q). Softmax applied per head per query with -max stab.
 * Memory for scores allocated per-thread via VLA; caller must ensure N_k
 * fits comfortably (used with N_k ∈ {145, 1024}). */
static void s3db_sdpa_f32(float *out, const float *q, const float *k,
                          const float *v, int N_q, int N_k, int H, int D_h,
                          int n_threads)
{
    const int E = H * D_h;
    const float scale = 1.0f / sqrtf((float)D_h);
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int h = 0; h < H; h++) {
        for (int nq = 0; nq < N_q; nq++) {
            /* alloc per-(h,nq) on the heap to avoid huge stack with VLA. */
            float *scores = (float *)malloc((size_t)N_k * sizeof(float));
            const float *qv = q + (size_t)nq * E + (size_t)h * D_h;
            float smax = -INFINITY;
            for (int nk = 0; nk < N_k; nk++) {
                const float *kv = k + (size_t)nk * E + (size_t)h * D_h;
                float s = 0.f;
                for (int d = 0; d < D_h; d++) s += qv[d] * kv[d];
                s *= scale;
                scores[nk] = s;
                if (s > smax) smax = s;
            }
            float sumexp = 0.f;
            for (int nk = 0; nk < N_k; nk++) {
                scores[nk] = expf(scores[nk] - smax);
                sumexp += scores[nk];
            }
            float inv = 1.0f / sumexp;
            float *o = out + (size_t)nq * E + (size_t)h * D_h;
            for (int d = 0; d < D_h; d++) {
                float acc = 0.f;
                for (int nk = 0; nk < N_k; nk++)
                    acc += scores[nk] * v[(size_t)nk * E + (size_t)h * D_h + d];
                o[d] = acc * inv;
            }
            free(scores);
        }
    }
}

/* Compute attn(q_in, k_in, v_in) with per-proj projection + output proj.
 *   q_in (N_q, Dq_in), k_in (N_k, Dk_in), v_in (N_k, Dv_in)
 *   Projected into (N_q, E) / (N_k, E) / (N_k, E) via q/k/v_proj.
 *   Output: (N_q, D_out) after output proj.
 * Scratch buffers allocated internally. */
static int s3db_attn_block(float *out,
                           const float *q_in, const float *k_in, const float *v_in,
                           int N_q, int N_k,
                           int Dq_in, int Dk_in, int Dv_in,
                           int H, int D_h, int D_out,
                           const float *Wq, const float *bq,
                           const float *Wk, const float *bk,
                           const float *Wv, const float *bv,
                           const float *Wproj, const float *bproj,
                           int n_threads)
{
    const int E = H * D_h;
    float *Q = (float *)malloc((size_t)N_q * E * sizeof(float));
    float *K = (float *)malloc((size_t)N_k * E * sizeof(float));
    float *V = (float *)malloc((size_t)N_k * E * sizeof(float));
    float *ctx = (float *)malloc((size_t)N_q * E * sizeof(float));
    if (!Q || !K || !V || !ctx) { free(Q); free(K); free(V); free(ctx); return -1; }
    s3db_linear_batch(Q, q_in, Wq, bq, N_q, Dq_in, E, n_threads);
    s3db_linear_batch(K, k_in, Wk, bk, N_k, Dk_in, E, n_threads);
    s3db_linear_batch(V, v_in, Wv, bv, N_k, Dv_in, E, n_threads);
    s3db_sdpa_f32(ctx, Q, K, V, N_q, N_k, H, D_h, n_threads);
    s3db_linear_batch(out, ctx, Wproj, bproj, N_q, E, D_out, n_threads);
    free(Q); free(K); free(V); free(ctx);
    return 0;
}

int sam3d_body_decoder_layer_forward(const sam3d_body_decoder_model *m,
                                     int layer_idx,
                                     const float *x_in,
                                     const float *context_in,
                                     const float *x_pe_in,
                                     const float *context_pe_in,
                                     int N_q, int N_c,
                                     int n_threads,
                                     float *x_out,
                                     float *context_out)
{
    if (!m || !x_in || !context_in || !x_pe_in || !context_pe_in || !x_out)
        return SAM3D_BODY_DECODER_E_INVAL;
    if (layer_idx < 0 || layer_idx >= m->n_layers)
        return SAM3D_BODY_DECODER_E_INVAL;
    const sam3d_body_decoder_layer *L = &m->layers[layer_idx];
    if (!L->ln1_w.data) return SAM3D_BODY_DECODER_E_LOAD;

    const int D   = m->dim;         /* 1024 */
    const int Dc  = m->kv_dim;      /* 1280 */
    const int H   = m->n_heads;     /* 8    */
    const int Dh  = m->head_dim;    /* 64   */
    const float eps = m->ln_eps;
    const int skip_first_pe = (layer_idx == 0);

    /* Pre-normalize x_pe / context_pe via ln_pe_{1,2}. */
    float *x_pe_n  = (float *)malloc((size_t)N_q * D  * sizeof(float));
    float *ctx_pe_n = (float *)malloc((size_t)N_c * Dc * sizeof(float));
    if (!x_pe_n || !ctx_pe_n) { free(x_pe_n); free(ctx_pe_n); return SAM3D_BODY_DECODER_E_LOAD; }
    s3db_layer_norm_batch(x_pe_n,  x_pe_in,  N_q, D,
                          (const float *)L->ln_pe_1_w.data,
                          (const float *)L->ln_pe_1_b.data, eps, n_threads);
    s3db_layer_norm_batch(ctx_pe_n, context_pe_in, N_c, Dc,
                          (const float *)L->ln_pe_2_w.data,
                          (const float *)L->ln_pe_2_b.data, eps, n_threads);

    /* ===== Self-attention block ===== */
    float *x_ln1 = (float *)malloc((size_t)N_q * D * sizeof(float));
    if (!x_ln1) { free(x_pe_n); free(ctx_pe_n); return SAM3D_BODY_DECODER_E_LOAD; }
    s3db_layer_norm_batch(x_ln1, x_in, N_q, D,
                          (const float *)L->ln1_w.data,
                          (const float *)L->ln1_b.data, eps, n_threads);

    /* q = k = x_ln1 + x_pe_n (unless skip_first_pe); v = x_ln1 */
    float *qk_buf = (float *)malloc((size_t)N_q * D * sizeof(float));
    if (!qk_buf) { free(x_pe_n); free(ctx_pe_n); free(x_ln1); return SAM3D_BODY_DECODER_E_LOAD; }
    if (skip_first_pe) {
        memcpy(qk_buf, x_ln1, (size_t)N_q * D * sizeof(float));
    } else {
        for (size_t i = 0; i < (size_t)N_q * D; i++) qk_buf[i] = x_ln1[i] + x_pe_n[i];
    }

    float *sa_out = (float *)malloc((size_t)N_q * D * sizeof(float));
    if (!sa_out) { free(x_pe_n); free(ctx_pe_n); free(x_ln1); free(qk_buf); return SAM3D_BODY_DECODER_E_LOAD; }
    int rc = s3db_attn_block(sa_out,
                             qk_buf, qk_buf, x_ln1,
                             N_q, N_q, D, D, D,
                             H, Dh, D,
                             (const float *)L->sa_q_w.data, (const float *)L->sa_q_b.data,
                             (const float *)L->sa_k_w.data, (const float *)L->sa_k_b.data,
                             (const float *)L->sa_v_w.data, (const float *)L->sa_v_b.data,
                             (const float *)L->sa_proj_w.data, (const float *)L->sa_proj_b.data,
                             n_threads);
    free(qk_buf); free(x_ln1);
    if (rc) { free(x_pe_n); free(ctx_pe_n); free(sa_out); return SAM3D_BODY_DECODER_E_LOAD; }

    /* x = x + sa_out */
    float *x1 = (float *)malloc((size_t)N_q * D * sizeof(float));
    if (!x1) { free(x_pe_n); free(ctx_pe_n); free(sa_out); return SAM3D_BODY_DECODER_E_LOAD; }
    for (size_t i = 0; i < (size_t)N_q * D; i++) x1[i] = x_in[i] + sa_out[i];
    free(sa_out);

    /* ===== Cross-attention block =====
     * q = ln2_1(x1) + x_pe_n
     * k = ln2_2(context) + ctx_pe_n
     * v = ln2_2(context)
     */
    float *q_in = (float *)malloc((size_t)N_q * D  * sizeof(float));
    float *ctx_ln = (float *)malloc((size_t)N_c * Dc * sizeof(float));
    if (!q_in || !ctx_ln) { free(x_pe_n); free(ctx_pe_n); free(x1); free(q_in); free(ctx_ln); return SAM3D_BODY_DECODER_E_LOAD; }
    s3db_layer_norm_batch(q_in, x1, N_q, D,
                          (const float *)L->ln2_1_w.data,
                          (const float *)L->ln2_1_b.data, eps, n_threads);
    for (size_t i = 0; i < (size_t)N_q * D; i++) q_in[i] += x_pe_n[i];
    s3db_layer_norm_batch(ctx_ln, context_in, N_c, Dc,
                          (const float *)L->ln2_2_w.data,
                          (const float *)L->ln2_2_b.data, eps, n_threads);
    /* k = ctx_ln + ctx_pe_n (separate buffer since v reads ctx_ln) */
    float *k_in = (float *)malloc((size_t)N_c * Dc * sizeof(float));
    if (!k_in) { free(x_pe_n); free(ctx_pe_n); free(x1); free(q_in); free(ctx_ln); return SAM3D_BODY_DECODER_E_LOAD; }
    for (size_t i = 0; i < (size_t)N_c * Dc; i++) k_in[i] = ctx_ln[i] + ctx_pe_n[i];

    float *ca_out = (float *)malloc((size_t)N_q * D * sizeof(float));
    if (!ca_out) { free(x_pe_n); free(ctx_pe_n); free(x1); free(q_in); free(ctx_ln); free(k_in); return SAM3D_BODY_DECODER_E_LOAD; }
    rc = s3db_attn_block(ca_out,
                         q_in, k_in, ctx_ln,
                         N_q, N_c, D, Dc, Dc,
                         H, Dh, D,
                         (const float *)L->ca_q_w.data, (const float *)L->ca_q_b.data,
                         (const float *)L->ca_k_w.data, (const float *)L->ca_k_b.data,
                         (const float *)L->ca_v_w.data, (const float *)L->ca_v_b.data,
                         (const float *)L->ca_proj_w.data, (const float *)L->ca_proj_b.data,
                         n_threads);
    free(q_in); free(k_in); free(ctx_ln);
    free(x_pe_n); free(ctx_pe_n);
    if (rc) { free(x1); free(ca_out); return SAM3D_BODY_DECODER_E_LOAD; }

    float *x2 = (float *)malloc((size_t)N_q * D * sizeof(float));
    if (!x2) { free(x1); free(ca_out); return SAM3D_BODY_DECODER_E_LOAD; }
    for (size_t i = 0; i < (size_t)N_q * D; i++) x2[i] = x1[i] + ca_out[i];
    free(x1); free(ca_out);

    /* ===== FFN =====
     * h = GELU(Linear0(ln3(x2)))
     * out = Linear1(h)
     * x = x2 + out
     */
    float *x_ln3 = (float *)malloc((size_t)N_q * D * sizeof(float));
    if (!x_ln3) { free(x2); return SAM3D_BODY_DECODER_E_LOAD; }
    s3db_layer_norm_batch(x_ln3, x2, N_q, D,
                          (const float *)L->ln3_w.data,
                          (const float *)L->ln3_b.data, eps, n_threads);

    const int H_ffn = m->ffn_hidden;   /* 1024 */
    float *ffn_h = (float *)malloc((size_t)N_q * H_ffn * sizeof(float));
    if (!ffn_h) { free(x2); free(x_ln3); return SAM3D_BODY_DECODER_E_LOAD; }
    s3db_linear_batch(ffn_h, x_ln3, (const float *)L->ffn0_w.data,
                      (const float *)L->ffn0_b.data, N_q, D, H_ffn, n_threads);
    free(x_ln3);
    for (size_t i = 0; i < (size_t)N_q * H_ffn; i++) ffn_h[i] = s3db_gelu_f32(ffn_h[i]);

    float *ffn_out = (float *)malloc((size_t)N_q * D * sizeof(float));
    if (!ffn_out) { free(x2); free(ffn_h); return SAM3D_BODY_DECODER_E_LOAD; }
    s3db_linear_batch(ffn_out, ffn_h, (const float *)L->ffn1_w.data,
                      (const float *)L->ffn1_b.data, N_q, H_ffn, D, n_threads);
    free(ffn_h);

    for (size_t i = 0; i < (size_t)N_q * D; i++) x_out[i] = x2[i] + ffn_out[i];
    free(x2); free(ffn_out);

    /* enable_twoway=False: context passes through unchanged. */
    if (context_out && context_out != context_in)
        memcpy(context_out, context_in, (size_t)N_c * Dc * sizeof(float));

    return SAM3D_BODY_DECODER_E_OK;
}

/* ---- Step 4g-i: keypoint-token update (combined 2D + 3D) ------------ */

/* Bilinear grid_sample at one (x, y) location, align_corners=False, padding
 * "zeros". src is (C, H, W) channel-major. dst (length C) is the sampled
 * vector. Out-of-bounds neighbors contribute 0. */
static void s3db_grid_sample_one(const float *src, int C, int H, int W,
                                 float gx_norm, float gy_norm, float *dst)
{
    /* PyTorch align_corners=False: input pixel = (norm + 1) * size / 2 - 0.5 */
    float xf = (gx_norm + 1.0f) * (float)W * 0.5f - 0.5f;
    float yf = (gy_norm + 1.0f) * (float)H * 0.5f - 0.5f;

    int x0 = (int)floorf(xf), x1 = x0 + 1;
    int y0 = (int)floorf(yf), y1 = y0 + 1;
    float ax = xf - (float)x0, ay = yf - (float)y0;

    float w00 = (1.0f - ax) * (1.0f - ay);
    float w10 = ax * (1.0f - ay);
    float w01 = (1.0f - ax) * ay;
    float w11 = ax * ay;

    int v00 = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H);
    int v10 = (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H);
    int v01 = (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H);
    int v11 = (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H);

    size_t HW = (size_t)H * (size_t)W;
    for (int c = 0; c < C; c++) {
        const float *plane = src + (size_t)c * HW;
        float v = 0.0f;
        if (v00) v += w00 * plane[(size_t)y0 * W + x0];
        if (v10) v += w10 * plane[(size_t)y0 * W + x1];
        if (v01) v += w01 * plane[(size_t)y1 * W + x0];
        if (v11) v += w11 * plane[(size_t)y1 * W + x1];
        dst[c] = v;
    }
}

int sam3d_body_kp_token_update(const sam3d_body_decoder_model *m,
                               int layer_idx,
                               const float *image_emb,
                               int H, int W,
                               const float *kp2d_cropped,
                               const float *kp2d_depth,
                               const float *kp3d_camera,
                               int N_q,
                               int n_threads,
                               float *tokens,
                               float *token_augment)
{
    if (!m || !tokens || !token_augment || !image_emb ||
        !kp2d_cropped || !kp2d_depth || !kp3d_camera)
        return SAM3D_BODY_DECODER_E_INVAL;

    /* Last-layer short-circuit (matches upstream guard). */
    if (layer_idx == m->n_layers - 1) return SAM3D_BODY_DECODER_E_OK;

    const int D     = m->dim;          /* 1024 */
    const int C_img = m->kv_dim;       /* 1280 */
    const int K     = m->n_keypoints;  /* 70 */
    const int kp2d_start = 1 + 1 + 1 + m->n_hand_tokens;          /* 5  */
    const int kp3d_start = kp2d_start + K;                         /* 75 */
    if (kp3d_start + K > N_q) return SAM3D_BODY_DECODER_E_INVAL;

    /* ---- 2D update -------------------------------------------------- */

    /* Build the 70x2 input pred_keypoints_2d_cropped (already in -0.5..0.5).
     * invalid_mask: out-of-bounds in [0,1] after +0.5, OR depth < 1e-5.
     */
    int *invalid = (int *)malloc((size_t)K * sizeof(int));
    if (!invalid) return SAM3D_BODY_DECODER_E_LOAD;
    for (int i = 0; i < K; i++) {
        float x01 = kp2d_cropped[i*2 + 0] + 0.5f;
        float y01 = kp2d_cropped[i*2 + 1] + 0.5f;
        invalid[i] = (x01 < 0.0f) || (x01 > 1.0f)
                  || (y01 < 0.0f) || (y01 > 1.0f)
                  || (kp2d_depth[i] < 1e-5f);
    }

    /* keypoint_posemb_linear: Linear(2,D) → ReLU → Linear(D,D), no identity. */
    float *posemb_h   = (float *)malloc((size_t)K * D * sizeof(float));
    float *posemb_out = (float *)malloc((size_t)K * D * sizeof(float));
    if (!posemb_h || !posemb_out) {
        free(invalid); free(posemb_h); free(posemb_out);
        return SAM3D_BODY_DECODER_E_LOAD;
    }
    s3db_linear_batch(posemb_h, kp2d_cropped,
                      (const float *)m->kp_posemb_l0_w.data,
                      (const float *)m->kp_posemb_l0_b.data,
                      K, 2, D, n_threads);
    for (size_t i = 0; i < (size_t)K * D; i++)
        if (posemb_h[i] < 0.0f) posemb_h[i] = 0.0f;
    s3db_linear_batch(posemb_out, posemb_h,
                      (const float *)m->kp_posemb_l1_w.data,
                      (const float *)m->kp_posemb_l1_b.data,
                      K, D, D, n_threads);
    free(posemb_h);

    /* token_augment[5..75) = posemb * (~invalid) — overwrite. */
    for (int i = 0; i < K; i++) {
        float *dst = token_augment + (size_t)(kp2d_start + i) * D;
        if (invalid[i]) {
            for (int d = 0; d < D; d++) dst[d] = 0.0f;
        } else {
            const float *src = posemb_out + (size_t)i * D;
            for (int d = 0; d < D; d++) dst[d] = src[d];
        }
    }
    free(posemb_out);

    /* grid_sample: sample_points = pred_keypoints_2d_cropped * 2 (so they
     * land in [-1, 1]). Image is 32x32 (DINOv3 H+ patch grid). */
    float *kp_feats = (float *)calloc((size_t)K * C_img, sizeof(float));
    if (!kp_feats) { free(invalid); return SAM3D_BODY_DECODER_E_LOAD; }
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int i = 0; i < K; i++) {
        if (invalid[i]) continue;  /* zero from calloc */
        float gx = kp2d_cropped[i*2 + 0] * 2.0f;
        float gy = kp2d_cropped[i*2 + 1] * 2.0f;
        /* ViT-H uses a rectangular image embedding grid (32x24). Upstream
         * scales x by H/W before grid_sample so normalized crop coords still
         * refer to the original square crop before width cropping. */
        if (W != H) gx *= (float)H / (float)W;
        s3db_grid_sample_one(image_emb, C_img, H, W, gx, gy,
                             kp_feats + (size_t)i * C_img);
    }

    /* keypoint_feat_linear: Linear(C_img, D). Then ADD to tokens[5..75). */
    float *kp_feat_proj = (float *)malloc((size_t)K * D * sizeof(float));
    if (!kp_feat_proj) { free(invalid); free(kp_feats); return SAM3D_BODY_DECODER_E_LOAD; }
    s3db_linear_batch(kp_feat_proj, kp_feats,
                      (const float *)m->kp_feat_linear_w.data,
                      (const float *)m->kp_feat_linear_b.data,
                      K, C_img, D, n_threads);
    free(kp_feats);

    for (int i = 0; i < K; i++) {
        float *dst = tokens + (size_t)(kp2d_start + i) * D;
        const float *src = kp_feat_proj + (size_t)i * D;
        for (int d = 0; d < D; d++) dst[d] += src[d];
    }
    free(kp_feat_proj);
    free(invalid);

    /* ---- 3D update -------------------------------------------------- */

    /* Pelvis-normalize: subtract (kp[9] + kp[10]) / 2 from every kp. */
    float pelvis[3];
    pelvis[0] = (kp3d_camera[9*3 + 0] + kp3d_camera[10*3 + 0]) * 0.5f;
    pelvis[1] = (kp3d_camera[9*3 + 1] + kp3d_camera[10*3 + 1]) * 0.5f;
    pelvis[2] = (kp3d_camera[9*3 + 2] + kp3d_camera[10*3 + 2]) * 0.5f;
    float *kp3d_norm = (float *)malloc((size_t)K * 3 * sizeof(float));
    if (!kp3d_norm) return SAM3D_BODY_DECODER_E_LOAD;
    for (int i = 0; i < K; i++) {
        kp3d_norm[i*3 + 0] = kp3d_camera[i*3 + 0] - pelvis[0];
        kp3d_norm[i*3 + 1] = kp3d_camera[i*3 + 1] - pelvis[1];
        kp3d_norm[i*3 + 2] = kp3d_camera[i*3 + 2] - pelvis[2];
    }

    /* keypoint3d_posemb_linear: Linear(3,D) → ReLU → Linear(D,D). */
    float *p3_h   = (float *)malloc((size_t)K * D * sizeof(float));
    float *p3_out = (float *)malloc((size_t)K * D * sizeof(float));
    if (!p3_h || !p3_out) {
        free(kp3d_norm); free(p3_h); free(p3_out);
        return SAM3D_BODY_DECODER_E_LOAD;
    }
    s3db_linear_batch(p3_h, kp3d_norm,
                      (const float *)m->kp3d_posemb_l0_w.data,
                      (const float *)m->kp3d_posemb_l0_b.data,
                      K, 3, D, n_threads);
    free(kp3d_norm);
    for (size_t i = 0; i < (size_t)K * D; i++)
        if (p3_h[i] < 0.0f) p3_h[i] = 0.0f;
    s3db_linear_batch(p3_out, p3_h,
                      (const float *)m->kp3d_posemb_l1_w.data,
                      (const float *)m->kp3d_posemb_l1_b.data,
                      K, D, D, n_threads);
    free(p3_h);

    /* token_augment[75..145) = kp3d_posemb (overwrite). */
    for (int i = 0; i < K; i++) {
        float *dst = token_augment + (size_t)(kp3d_start + i) * D;
        const float *src = p3_out + (size_t)i * D;
        for (int d = 0; d < D; d++) dst[d] = src[d];
    }
    free(p3_out);

    return SAM3D_BODY_DECODER_E_OK;
}

/* ---- norm_final + MHR regression heads (step 4f-A) ------------------ */

int sam3d_body_norm_final(const sam3d_body_decoder_model *m,
                          const float *x_in, int N_q, int n_threads,
                          float *x_out)
{
    if (!m || !x_in || !x_out) return SAM3D_BODY_DECODER_E_INVAL;
    if (!m->norm_final_w.data || !m->norm_final_b.data)
        return SAM3D_BODY_DECODER_E_LOAD;
    s3db_layer_norm_batch(x_out, x_in, N_q, m->dim,
                          (const float *)m->norm_final_w.data,
                          (const float *)m->norm_final_b.data,
                          m->ln_eps, n_threads);
    return SAM3D_BODY_DECODER_E_OK;
}

int sam3d_body_apply_heads_raw(const sam3d_body_decoder_model *m,
                               const float *pose_token,
                               int n_threads,
                               float *out_pose_raw,
                               float *out_cam_raw)
{
    if (!m || !pose_token || !out_pose_raw || !out_cam_raw)
        return SAM3D_BODY_DECODER_E_INVAL;
    if (!m->head_pose_l0_w.data || !m->head_camera_l0_w.data)
        return SAM3D_BODY_DECODER_E_LOAD;

    const int D = m->dim;       /* 1024 */
    const int Np = m->npose;    /* 519 */
    const int Nc = m->ncam;     /* 3   */

    /* head_pose.proj: Linear(D→D) → GELU → Linear(D→519) */
    float *h = (float *)malloc((size_t)D * sizeof(float));
    if (!h) return SAM3D_BODY_DECODER_E_LOAD;

    s3db_linear_batch(h, pose_token,
                      (const float *)m->head_pose_l0_w.data,
                      (const float *)m->head_pose_l0_b.data,
                      1, D, D, n_threads);
    for (int i = 0; i < D; i++) h[i] = h[i] > 0.0f ? h[i] : 0.0f; /* ReLU — FFN default act_layer */
    s3db_linear_batch(out_pose_raw, h,
                      (const float *)m->head_pose_l1_w.data,
                      (const float *)m->head_pose_l1_b.data,
                      1, D, Np, n_threads);

    /* head_camera.proj: Linear(D→D) → ReLU → Linear(D→3) */
    s3db_linear_batch(h, pose_token,
                      (const float *)m->head_camera_l0_w.data,
                      (const float *)m->head_camera_l0_b.data,
                      1, D, D, n_threads);
    for (int i = 0; i < D; i++) h[i] = h[i] > 0.0f ? h[i] : 0.0f;
    s3db_linear_batch(out_cam_raw, h,
                      (const float *)m->head_camera_l1_w.data,
                      (const float *)m->head_camera_l1_b.data,
                      1, D, Nc, n_threads);

    free(h);
    return SAM3D_BODY_DECODER_E_OK;
}

/* ================================================================== */
/* Head decode (stages 1-5): pose_raw[519] -> mhr_model_params[204] +  */
/*                            shape[45] + face[72]                     */
/* ================================================================== */

/* Body params: 23 three-DoF Euler XYZ joints, scattered into 133-vec.
 * Layout per joint: 3 indices in `body3dof_idxs[j*3..j*3+3]`. Mirror of
 * `all_param_3dof_rot_idxs` in mhr_utils.compact_cont_to_model_params_body. */
static const int s3db_body3dof_idxs[23 * 3] = {
    0, 2, 4,    6, 8, 10,    12, 13, 14,   15, 16, 17,   18, 19, 20,
    21, 22, 23, 24, 25, 26,  27, 28, 29,   34, 35, 36,   37, 38, 39,
    44, 45, 46, 53, 54, 55,  64, 65, 66,
    85, 69, 73, 86, 70, 79,  87, 71, 82,   88, 72, 76,   91, 92, 93,
    112, 96, 100, 113, 97, 106, 114, 98, 109, 115, 99, 103,
    130, 131, 132,
};
/* 58 single-DoF rotation slots into 133-vec (sin/cos pairs in cont). */
static const int s3db_body1dof_rot_idxs[58] = {
    1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52,
    56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84,
    89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118,
    119, 120, 121, 122, 123,
};
/* 6 translation slots (passed through directly from cont). */
static const int s3db_body1dof_trans_idxs[6] = { 124, 125, 126, 127, 128, 129 };

/* Hand mask: zero these slots in body_pose_euler[133] before drop-in. */
static const int s3db_mhr_hand_idxs[54] = {
    62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,
    86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,
    107,108,109,110,111,112,113,114,115,
};

/* Hand interleaved DoF spec: [3,1,1,3,1,1,3,1,1,3,1,1,2,3,1,1] sums to 27.
 * Per joint, output dim = k. cont input dim per joint = 2*k.
 * Total cont = 54, total model_params = 27. */
static const int s3db_hand_dof_seq[16] = { 3,1,1,3,1,1,3,1,1,3,1,1,2,3,1,1 };

static inline float s3db_dot3(const float *a, const float *b)
{ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

static inline void s3db_normalize3(float *v)
{
    double n2 = (double)v[0]*v[0] + (double)v[1]*v[1] + (double)v[2]*v[2];
    if (n2 < 1e-20) return;
    float s = (float)(1.0 / sqrt(n2));
    v[0] *= s; v[1] *= s; v[2] *= s;
}

/* rot6d_to_rotmat: 6 → 3×3 (column-major: cols=b1,b2,b3). */
static void s3db_rot6d_to_rotmat(const float *x6, float R[9])
{
    /* Per upstream: x = x.reshape(-1,2,3).permute(0,2,1) → (3,2)
     *   a1 = first 3 elements, a2 = next 3 elements. */
    float a1[3] = { x6[0], x6[1], x6[2] };
    float a2[3] = { x6[3], x6[4], x6[5] };
    float b1[3] = { a1[0], a1[1], a1[2] };
    s3db_normalize3(b1);
    float dot = s3db_dot3(b1, a2);
    float b2[3] = { a2[0]-dot*b1[0], a2[1]-dot*b1[1], a2[2]-dot*b1[2] };
    s3db_normalize3(b2);
    float b3[3] = { b1[1]*b2[2]-b1[2]*b2[1],
                    b1[2]*b2[0]-b1[0]*b2[2],
                    b1[0]*b2[1]-b1[1]*b2[0] };
    /* R: col0=b1, col1=b2, col2=b3, row-major flat */
    R[0]=b1[0]; R[1]=b2[0]; R[2]=b3[0];
    R[3]=b1[1]; R[4]=b2[1]; R[5]=b3[1];
    R[6]=b1[2]; R[7]=b2[2]; R[8]=b3[2];
}

/* roma "ZYX" intrinsic Euler from rotmat: returns [αz, βy, γx] s.t.
 * R = Rz(αz) Ry(βy) Rx(γx). */
static void s3db_rotmat_to_euler_zyx(const float R[9], float out3[3])
{
    /* R[2,0] = -sy   →   sy = -R[6] */
    /* R[1,0]/R[0,0] = sz/cz when cy>0 */
    /* R[2,1]/R[2,2] = sx/cx when cy>0 */
    float r20 = R[6];
    float cy = sqrtf(R[0]*R[0] + R[3]*R[3]);
    out3[1] = atan2f(-r20, cy);             /* βy */
    out3[0] = atan2f(R[3], R[0]);           /* αz */
    out3[2] = atan2f(R[7], R[8]);           /* γx */
}

/* roma "xyz" Euler convention is extrinsic xyz, equivalent to
 * `R = Rz(c) @ Ry(b) @ Rx(a)` with input order [a, b, c].
 * Same matrix shape as ZYX [αz, βy, γx] = (a=γx, b=βy, c=αz).
 *
 *   R[0,0]=cz*cb       R[0,1]=cz*sb*sa-sz*ca   R[0,2]=cz*sb*ca+sz*sa
 *   R[1,0]=sz*cb       R[1,1]=sz*sb*sa+cz*ca   R[1,2]=sz*sb*ca-cz*sa
 *   R[2,0]=-sb         R[2,1]=cb*sa            R[2,2]=cb*ca
 */
static void s3db_euler_xyz_to_rotmat(const float e[3], float R[9])
{
    float a = e[0], b = e[1], c = e[2];
    float ca = cosf(a), sa = sinf(a);
    float cb = cosf(b), sb = sinf(b);
    float cz = cosf(c), sz = sinf(c);
    R[0] = cz*cb;      R[1] = cz*sb*sa - sz*ca;  R[2] = cz*sb*ca + sz*sa;
    R[3] = sz*cb;      R[4] = sz*sb*sa + cz*ca;  R[5] = sz*sb*ca - cz*sa;
    R[6] = -sb;        R[7] = cb*sa;             R[8] = cb*ca;
}

static void s3db_rotmat_to_euler_xyz(const float R[9], float out3[3])
{
    /* sb = -R[2,0]; cb = sqrt(R[0,0]² + R[1,0]²). */
    float cb = sqrtf(R[0]*R[0] + R[3]*R[3]);
    out3[1] = atan2f(-R[6], cb);                /* b */
    out3[0] = atan2f(R[7], R[8]);               /* a */
    out3[2] = atan2f(R[3], R[0]);               /* c */
}

/* 3x3 matmul: C = A @ B (row-major, all 3x3). */
static inline void s3db_mat3_mm(const float A[9], const float B[9], float C[9])
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float s = 0.0f;
            for (int k = 0; k < 3; k++) s += A[i*3+k] * B[k*3+j];
            C[i*3+j] = s;
        }
    }
}

/* y = M @ x (3x3 @ 3). */
static inline void s3db_mat3_mv(const float M[9], const float x[3], float y[3])
{
    y[0] = M[0]*x[0] + M[1]*x[1] + M[2]*x[2];
    y[1] = M[3]*x[0] + M[4]*x[1] + M[5]*x[2];
    y[2] = M[6]*x[0] + M[7]*x[1] + M[8]*x[2];
}

/* batchXYZfrom6D: cont[6] → Euler XYZ[3].
 *
 * Per upstream: x_raw = poses[..., :3]; y_raw = poses[..., 3:];
 *   x = norm(x_raw); z = norm(cross(x, y_raw)); y = cross(z, x);
 *   M = stack([x, y, z], dim=-1)  (cols)
 * Then extract XYZ Euler via:
 *   sy = sqrt(M[0,0]² + M[1,0]²)
 *   if not singular:
 *     ex = atan2(M[2,1], M[2,2])
 *     ey = atan2(-M[2,0], sy)
 *     ez = atan2(M[1,0], M[0,0])
 *   else:
 *     ex = atan2(-M[1,2], M[1,1])
 *     ey = atan2(-M[2,0], sy)
 *     ez = 0
 */
static void s3db_xyz_from_6d(const float *cont6, float out3[3])
{
    float x[3]   = { cont6[0], cont6[1], cont6[2] };
    float yraw[3]= { cont6[3], cont6[4], cont6[5] };
    s3db_normalize3(x);
    float z[3] = { x[1]*yraw[2]-x[2]*yraw[1],
                   x[2]*yraw[0]-x[0]*yraw[2],
                   x[0]*yraw[1]-x[1]*yraw[0] };
    s3db_normalize3(z);
    float y[3] = { z[1]*x[2]-z[2]*x[1],
                   z[2]*x[0]-z[0]*x[2],
                   z[0]*x[1]-z[1]*x[0] };
    /* M cols=[x,y,z] flat row-major: M[i,j] */
    float M[9] = {
        x[0], y[0], z[0],
        x[1], y[1], z[1],
        x[2], y[2], z[2],
    };
    float sy = sqrtf(M[0]*M[0] + M[3]*M[3]);
    int singular = sy < 1e-6f;
    float ex, ey, ez;
    if (!singular) {
        ex = atan2f(M[7], M[8]);
        ey = atan2f(-M[6], sy);
        ez = atan2f(M[3], M[0]);
    } else {
        ex = atan2f(-M[5], M[4]);
        ey = atan2f(-M[6], sy);
        ez = 0.0f;
    }
    out3[0] = ex; out3[1] = ey; out3[2] = ez;
}

/* compact_cont_to_model_params_body: 260 → 133 */
static void s3db_compact_cont_to_body(const float *cont260, float *body133)
{
    memset(body133, 0, 133 * sizeof(float));
    /* 23 × 6D → 23 × XYZ_Euler scattered into 69 slots. */
    const int N3 = 23;
    for (int j = 0; j < N3; j++) {
        float xyz[3];
        s3db_xyz_from_6d(cont260 + j * 6, xyz);
        body133[s3db_body3dof_idxs[j*3 + 0]] = xyz[0];
        body133[s3db_body3dof_idxs[j*3 + 1]] = xyz[1];
        body133[s3db_body3dof_idxs[j*3 + 2]] = xyz[2];
    }
    /* 58 sincos pairs → atan2 → 58 slots. cont layout after 138 dims. */
    const int N1 = 58;
    const float *p1 = cont260 + 23 * 6;  /* offset 138 */
    for (int j = 0; j < N1; j++) {
        float s = p1[j*2 + 0];
        float c = p1[j*2 + 1];
        body133[s3db_body1dof_rot_idxs[j]] = atan2f(s, c);
    }
    /* 6 translation passthrough. cont layout after 138 + 116 = 254 dims. */
    const float *pt = cont260 + 138 + 116;
    for (int j = 0; j < 6; j++) {
        body133[s3db_body1dof_trans_idxs[j]] = pt[j];
    }
}

/* compact_cont_to_model_params_hand: 54 cont → 27 model_params.
 * Walks `s3db_hand_dof_seq` interleaving the 3DoF blocks (6 cont → 3 params
 * via batchXYZfrom6D) and 1/2DoF blocks (2 cont → atan2 angle). */
static void s3db_compact_cont_to_hand(const float *cont54, float *hand27)
{
    int ci = 0;     /* read cursor into cont54 */
    int mi = 0;     /* write cursor into hand27 */
    for (int i = 0; i < 16; i++) {
        int k = s3db_hand_dof_seq[i];
        if (k == 3) {
            float xyz[3];
            s3db_xyz_from_6d(cont54 + ci, xyz);
            hand27[mi+0] = xyz[0];
            hand27[mi+1] = xyz[1];
            hand27[mi+2] = xyz[2];
            ci += 6;
            mi += 3;
        } else {
            /* k==1 or k==2: each "joint" emits k atan2 angles, taking
             * 2*k cont dims as k sincos pairs. The Python uses unflatten
             * (-1, (-1, 2)) and then takes [..., -2], [..., -1] which
             * means s = pair[0], c = pair[1]. */
            for (int kk = 0; kk < k; kk++) {
                float s = cont54[ci + 0];
                float c = cont54[ci + 1];
                hand27[mi] = atan2f(s, c);
                ci += 2;
                mi += 1;
            }
        }
    }
    /* Sanity: ci==54, mi==27. */
}

int sam3d_body_decode_pose_raw(const sam3d_body_decoder_model *m,
                               const float *pose_raw,
                               int enable_hand_model,
                               float *out_mhr_model_params,
                               float *out_shape,
                               float *out_face)
{
    if (!m || !pose_raw || !out_mhr_model_params || !out_shape || !out_face)
        return SAM3D_BODY_DECODER_E_INVAL;

    /* --- Stage 1+2: rot6d → rotmat → ZYX Euler. */
    float R[9];
    s3db_rot6d_to_rotmat(pose_raw + 0, R);
    float global_rot_euler[3];
    s3db_rotmat_to_euler_zyx(R, global_rot_euler);

    /* --- Stage 3: body cont → 133 Euler model params, then mask. */
    float body133[133];
    s3db_compact_cont_to_body(pose_raw + 6, body133);
    /* zero hand idxs */
    for (int i = 0; i < 54; i++) body133[s3db_mhr_hand_idxs[i]] = 0.0f;
    /* zero last 3 (jaw) */
    body133[130] = 0.0f; body133[131] = 0.0f; body133[132] = 0.0f;

    /* --- enable_hand_model branch (mhr_head.py mhr_forward lines 181-196):
     *   R_xyz = euler_to_rotmat("xyz", global_rot_ori)        # convention switch
     *   R_new = R_xyz @ local_to_world_wrist
     *   global_rot = rotmat_to_euler("xyz", R_new)
     *   global_trans = -(euler_to_rotmat("xyz", global_rot)
     *                    @ (right_wrist_coords - root_coords)
     *                    + root_coords)                       # global_trans_ori = 0
     * Body branch keeps global_rot=ZYX-Euler and global_trans=0. */
    float grot_used[3];
    float global_trans[3] = { 0.0f, 0.0f, 0.0f };
    if (enable_hand_model) {
        const float *L2W = (const float *)m->local_to_world_wrist.data;
        const float *RWC = (const float *)m->right_wrist_coords.data;
        const float *RC  = (const float *)m->root_coords.data;
        float Rxyz_ori[9], Rwrist[9], grot_xyz[3];
        s3db_euler_xyz_to_rotmat(global_rot_euler, Rxyz_ori);
        s3db_mat3_mm(Rxyz_ori, L2W, Rwrist);
        s3db_rotmat_to_euler_xyz(Rwrist, grot_xyz);
        float Rxyz_new[9];
        s3db_euler_xyz_to_rotmat(grot_xyz, Rxyz_new);
        float wrist_off[3] = { RWC[0]-RC[0], RWC[1]-RC[1], RWC[2]-RC[2] };
        float Rw[3];
        s3db_mat3_mv(Rxyz_new, wrist_off, Rw);
        global_trans[0] = -(Rw[0] + RC[0]);
        global_trans[1] = -(Rw[1] + RC[1]);
        global_trans[2] = -(Rw[2] + RC[2]);
        grot_used[0] = grot_xyz[0];
        grot_used[1] = grot_xyz[1];
        grot_used[2] = grot_xyz[2];
    } else {
        grot_used[0] = global_rot_euler[0];
        grot_used[1] = global_rot_euler[1];
        grot_used[2] = global_rot_euler[2];
    }

    /* --- Stage 4a: assemble full_pose_params(136) =
     *   [trans*10(3), global_rot(3), body133[:130]]. */
    float full_pose[136];
    full_pose[0] = global_trans[0] * 10.0f;
    full_pose[1] = global_trans[1] * 10.0f;
    full_pose[2] = global_trans[2] * 10.0f;
    full_pose[3] = grot_used[0];
    full_pose[4] = grot_used[1];
    full_pose[5] = grot_used[2];
    memcpy(full_pose + 6, body133, 130 * sizeof(float));

    /* --- Stage 4b: hand PCA decode + drop-in.
     *   pred_hand[108] = [left_pca(54); right_pca(54)]
     *   for each side: cont = hand_pose_mean + pca @ hand_pose_comps  → (54,)
     *                  model = compact_cont_to_hand(cont)             → (27,)
     *                  full_pose[hand_idx_*]] = model
     * hand_pose_comps shape: (54, 54). einsum("da,ab->db", pca, comps):
     *   cont[b] = sum_a pca[a] * comps[a, b]
     */
    const float *pred_hand_l = pose_raw + 339;
    const float *pred_hand_r = pose_raw + 339 + 54;
    const float *hpm = (const float *)m->hand_pose_mean.data;
    const float *hpc = (const float *)m->hand_pose_comps.data;
    const int64_t *hil = (const int64_t *)m->hand_idx_left.data;
    const int64_t *hir = (const int64_t *)m->hand_idx_right.data;
    for (int side = 0; side < 2; side++) {
        const float *pca = (side == 0) ? pred_hand_l : pred_hand_r;
        const int64_t *hi = (side == 0) ? hil : hir;
        float cont54[54];
        for (int b = 0; b < 54; b++) {
            double s = (double)hpm[b];
            for (int a = 0; a < 54; a++) {
                s += (double)pca[a] * (double)hpc[a*54 + b];
            }
            cont54[b] = (float)s;
        }
        float hand27[27];
        s3db_compact_cont_to_hand(cont54, hand27);
        for (int j = 0; j < 27; j++) {
            full_pose[hi[j]] = hand27[j];
        }
    }

    /* --- Stage 5: scales = scale_mean + pred_scale @ scale_comps. */
    float scales68[68];
    const float *sm = (const float *)m->scale_mean.data;
    const float *sc = (const float *)m->scale_comps.data;  /* (28, 68) */
    const float *pred_scale = pose_raw + 311;
    for (int b = 0; b < 68; b++) {
        double s = (double)sm[b];
        for (int a = 0; a < 28; a++) {
            s += (double)pred_scale[a] * (double)sc[a*68 + b];
        }
        scales68[b] = (float)s;
    }

    /* --- Assemble (204,) and the side outputs. */
    memcpy(out_mhr_model_params,        full_pose, 136 * sizeof(float));
    memcpy(out_mhr_model_params + 136,  scales68,   68 * sizeof(float));

    /* --- Stage 5b: nonhand zero-out (mhr_head.py:225) — hand branch only. */
    if (enable_hand_model) {
        const int64_t *nz = (const int64_t *)m->nonhand_param_idxs.data;
        size_t n_nz = (size_t)m->nonhand_param_idxs.dims[0];
        for (size_t i = 0; i < n_nz; i++) {
            out_mhr_model_params[nz[i]] = 0.0f;
        }
    }

    memcpy(out_shape, pose_raw + 266, 45 * sizeof(float));
    /* expr_params zeroed in body-only branch (mhr_head.py line 316). */
    (void)out_face;
    memset(out_face, 0, 72 * sizeof(float));

    return SAM3D_BODY_DECODER_E_OK;
}

/* ================================================================== */
/* Stage 12: 70-keypoint regression                                    */
/* ================================================================== */
int sam3d_body_keypoints_from_mesh(const sam3d_body_decoder_model *m,
                                   const float *verts_m,
                                   const float *joint_coords_m,
                                   int enable_hand_model,
                                   int n_threads,
                                   float *out_kpts)
{
    (void)n_threads;
    if (!m || !verts_m || !joint_coords_m || !out_kpts)
        return SAM3D_BODY_DECODER_E_INVAL;
    if (!m->keypoint_mapping.data) return SAM3D_BODY_DECODER_E_LOAD;

    const int V = 18439;
    const int J = 127;
    const int VJ = V + J;       /* 18566 */
    const int NK = 70;
    const float *KM = (const float *)m->keypoint_mapping.data;  /* (308, 18566) */

    /* For each of the first 70 keypoints: out[k, c] = sum_i KM[k,i] * X[i,c].
     * X[i,c] = verts_m[i,c] for i<V, joint_coords_m[i-V,c] for i in [V, VJ).
     * Body path: head_pose has enable_hand_model=False, all 70 computed.
     * Hand path: keep [21, 42); zero [0, 21) and [42, 70) (mhr_head.py:255). */
    memset(out_kpts, 0, NK * 3 * sizeof(float));
    int k_lo = enable_hand_model ? 21 : 0;
    int k_hi = enable_hand_model ? 42 : NK;
    for (int k = k_lo; k < k_hi; k++) {
        const float *Wk = KM + (size_t)k * VJ;
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int i = 0; i < V; i++) {
            double w = (double)Wk[i];
            sx += w * (double)verts_m[i*3 + 0];
            sy += w * (double)verts_m[i*3 + 1];
            sz += w * (double)verts_m[i*3 + 2];
        }
        for (int i = 0; i < J; i++) {
            double w = (double)Wk[V + i];
            sx += w * (double)joint_coords_m[i*3 + 0];
            sy += w * (double)joint_coords_m[i*3 + 1];
            sz += w * (double)joint_coords_m[i*3 + 2];
        }
        /* Camera-system flip on (y, z). */
        out_kpts[k*3 + 0] = (float)sx;
        out_kpts[k*3 + 1] = (float)(-sy);
        out_kpts[k*3 + 2] = (float)(-sz);
    }
    return SAM3D_BODY_DECODER_E_OK;
}

/* ================================================================== */
/* camera_project + _full_to_crop                                       */
/* ================================================================== */
int sam3d_body_camera_project(const float *j3d_post_flip,
                              const float *pred_cam,
                              const sam3d_body_camera_batch *batch,
                              int K,
                              float *out_kp2d,
                              float *out_kp2d_cropped,
                              float *out_kp2d_depth,
                              float *out_pred_cam_t)
{
    if (!j3d_post_flip || !pred_cam || !batch || K <= 0)
        return SAM3D_BODY_DECODER_E_INVAL;

    /* Apply camera-system flip on pred_cam: pc[0] *= -1; pc[2] *= -1. */
    const float s  = -pred_cam[0];
    const float tx =  pred_cam[1];
    const float ty = -pred_cam[2];

    const float scale_factor = (batch->default_scale_factor > 0.0f)
                               ? batch->default_scale_factor : 1.0f;
    const float bs = batch->bbox_scale * s * scale_factor + 1e-8f;
    const float fx = batch->cam_int[0];          /* K[0,0] */
    const float tz = 2.0f * fx / bs;
    float cx, cy;
    if (!batch->use_intrin_center) {
        cx = 2.0f * (batch->bbox_center[0] - batch->ori_img_size[0] * 0.5f) / bs;
        cy = 2.0f * (batch->bbox_center[1] - batch->ori_img_size[1] * 0.5f) / bs;
    } else {
        cx = 2.0f * (batch->bbox_center[0] - batch->cam_int[2]) / bs;
        cy = 2.0f * (batch->bbox_center[1] - batch->cam_int[5]) / bs;
    }
    const float ct[3] = { tx + cx, ty + cy, tz };
    if (out_pred_cam_t) {
        out_pred_cam_t[0] = ct[0];
        out_pred_cam_t[1] = ct[1];
        out_pred_cam_t[2] = ct[2];
    }

    const float *KM = batch->cam_int;            /* (3,3) row-major */
    const float *AT = batch->affine_trans;       /* (2,3) row-major */
    const float invIW = 1.0f / batch->img_size[0];
    const float invIH = 1.0f / batch->img_size[1];

    for (int k = 0; k < K; k++) {
        float p0 = j3d_post_flip[k*3 + 0] + ct[0];
        float p1 = j3d_post_flip[k*3 + 1] + ct[1];
        float p2 = j3d_post_flip[k*3 + 2] + ct[2];
        if (out_kp2d_depth) out_kp2d_depth[k] = p2;

        /* Perspective: y = K @ (p / p.z), take first two channels. */
        float invz = 1.0f / p2;
        float xn = p0 * invz, yn = p1 * invz;
        float u = KM[0] * xn + KM[1] * yn + KM[2];
        float v = KM[3] * xn + KM[4] * yn + KM[5];
        if (out_kp2d) {
            out_kp2d[k*2 + 0] = u;
            out_kp2d[k*2 + 1] = v;
        }
        if (out_kp2d_cropped) {
            float ax = AT[0]*u + AT[1]*v + AT[2];
            float ay = AT[3]*u + AT[4]*v + AT[5];
            out_kp2d_cropped[k*2 + 0] = ax * invIW - 0.5f;
            out_kp2d_cropped[k*2 + 1] = ay * invIH - 0.5f;
        }
    }
    return SAM3D_BODY_DECODER_E_OK;
}

/* sam3d_body_decoder_forward implementation lives in the
 * SAM3D_BODY_DECODER_FULL_IMPLEMENTATION block below — it depends on
 * forward_full + sam3d_body_mhr_forward. */

int sam3d_body_decoder_forward_preset(
    const sam3d_body_decoder_model *m,
    const float *image_emb_chw,
    const float *image_pe_chw,
    int H, int W,
    const float *initial_tokens,
    const float *initial_augment,
    const float *kp2d_cropped_pl,
    const float *kp2d_depth_pl,
    const float *kp3d_pl,
    int n_threads,
    sam3d_body_decoder_result *out)
{
    if (!m || !image_emb_chw || !image_pe_chw ||
        !initial_tokens || !initial_augment || !out)
        return SAM3D_BODY_DECODER_E_INVAL;
    if (!kp2d_cropped_pl || !kp2d_depth_pl || !kp3d_pl)
        return SAM3D_BODY_DECODER_E_INVAL;

    const int N_Q = 1 + 1 + 1 + m->n_hand_tokens
                  + m->n_keypoints + m->n_keypoints;       /* 145 */
    const int D   = m->dim;                                /* 1024 */
    const int N_C = H * W;                                 /* 1024 for 32×32 */
    const int Dc  = m->kv_dim;                             /* 1280 */
    const int K   = m->n_keypoints;                        /* 70 */

    /* Flatten (Dc, H, W) → (H*W, Dc) once for image_emb and image_pe. */
    float *ctx    = (float *)malloc((size_t)N_C * Dc * sizeof(float));
    float *ctx_pe = (float *)malloc((size_t)N_C * Dc * sizeof(float));
    if (!ctx || !ctx_pe) { free(ctx); free(ctx_pe); return SAM3D_BODY_DECODER_E_LOAD; }
    for (int n = 0; n < N_C; n++) {
        for (int c = 0; c < Dc; c++) {
            ctx[(size_t)n * Dc + c]    = image_emb_chw[(size_t)c * N_C + n];
            ctx_pe[(size_t)n * Dc + c] = image_pe_chw[(size_t)c * N_C + n];
        }
    }

    /* Working buffers for tokens / augment, ping-pong. */
    float *tokens   = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *augment  = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *tokens_b = (float *)malloc((size_t)N_Q * D * sizeof(float));
    if (!tokens || !augment || !tokens_b) {
        free(ctx); free(ctx_pe); free(tokens); free(augment); free(tokens_b);
        return SAM3D_BODY_DECODER_E_LOAD;
    }
    memcpy(tokens,  initial_tokens,  (size_t)N_Q * D * sizeof(float));
    memcpy(augment, initial_augment, (size_t)N_Q * D * sizeof(float));

    for (int li = 0; li < m->n_layers; li++) {
        int rc = sam3d_body_decoder_layer_forward(
            m, li, tokens, ctx, augment, ctx_pe, N_Q, N_C,
            n_threads, tokens_b, NULL);
        if (rc != SAM3D_BODY_DECODER_E_OK) {
            free(ctx); free(ctx_pe); free(tokens); free(augment); free(tokens_b);
            return rc;
        }
        /* Swap tokens <- tokens_b. */
        float *tmp = tokens; tokens = tokens_b; tokens_b = tmp;

        /* Apply preset kp_token_update between layers (skip after last). */
        if (li < m->n_layers - 1) {
            const float *kp2d  = kp2d_cropped_pl + (size_t)li * K * 2;
            const float *dep   = kp2d_depth_pl   + (size_t)li * K;
            const float *kp3d  = kp3d_pl         + (size_t)li * K * 3;
            rc = sam3d_body_kp_token_update(
                m, li, image_emb_chw, H, W, kp2d, dep, kp3d, N_Q,
                n_threads, tokens, augment);
            if (rc != SAM3D_BODY_DECODER_E_OK) {
                free(ctx); free(ctx_pe); free(tokens); free(augment); free(tokens_b);
                return rc;
            }
        }
    }
    free(ctx); free(ctx_pe); free(tokens_b);

    /* Final norm + heads. */
    float *tokens_norm = (float *)malloc((size_t)N_Q * D * sizeof(float));
    if (!tokens_norm) { free(tokens); free(augment); return SAM3D_BODY_DECODER_E_LOAD; }
    int rc = sam3d_body_norm_final(m, tokens, N_Q, n_threads, tokens_norm);
    free(tokens); free(augment);
    if (rc != SAM3D_BODY_DECODER_E_OK) { free(tokens_norm); return rc; }

    float pose_raw[519];
    float cam_raw[3];
    rc = sam3d_body_apply_heads_raw(m, tokens_norm, n_threads, pose_raw, cam_raw);
    free(tokens_norm);
    if (rc != SAM3D_BODY_DECODER_E_OK) return rc;

    /* Add init_pose / init_camera (matches FFN(add_identity=False) path). */
    const float *ip = (const float *)m->init_pose.data;
    const float *ic = (const float *)m->init_camera.data;
    for (int i = 0; i < 519; i++) out->mhr_params[i] = pose_raw[i] + ip[i];
    for (int i = 0; i < 3;   i++) out->cam_t[i]      = cam_raw[i]  + ic[i];
    return SAM3D_BODY_DECODER_E_OK;
}

/* ================================================================== */
/* Iterative forward: decoder + per-layer MHR-in-the-loop              */
/* ================================================================== */

#ifdef SAM3D_BODY_DECODER_FULL_IMPLEMENTATION

/* Helper: from gskel (J, 8) extract jcoords (J, 3) and divide by 100. */
static void s3db_extract_jcoords_meters(const float *gskel_cm, int J,
                                        float *jc_m_out)
{
    for (int j = 0; j < J; j++) {
        jc_m_out[j*3 + 0] = gskel_cm[(size_t)j*8 + 0] * 0.01f;
        jc_m_out[j*3 + 1] = gskel_cm[(size_t)j*8 + 1] * 0.01f;
        jc_m_out[j*3 + 2] = gskel_cm[(size_t)j*8 + 2] * 0.01f;
    }
}

int sam3d_body_decoder_forward_full(
    const sam3d_body_decoder_model *m,
    const struct sam3d_body_mhr_assets_t *mhr,
    const sam3d_body_camera_batch *cam_batch,
    const float *image_emb_chw,
    const float *image_pe_chw,
    int H, int W,
    const float *initial_tokens,
    const float *initial_augment,
    int n_threads,
    sam3d_body_decoder_full_result *out)
{
    if (!m || !mhr || !cam_batch || !image_emb_chw || !image_pe_chw ||
        !initial_tokens || !initial_augment || !out)
        return SAM3D_BODY_DECODER_E_INVAL;

    const int N_Q = 1 + 1 + 1 + m->n_hand_tokens
                  + m->n_keypoints + m->n_keypoints;       /* 145 */
    const int D   = m->dim;                                /* 1024 */
    const int N_C = H * W;                                 /* 1024 for 32×32 */
    const int Dc  = m->kv_dim;                             /* 1280 */
    const int K   = m->n_keypoints;                        /* 70 */
    const int V   = 18439;
    const int J   = 127;

    const float *ip = (const float *)m->init_pose.data;
    const float *ic = (const float *)m->init_camera.data;

    /* Flatten image_emb / pe (CHW → HW×C) once. */
    float *ctx        = (float *)malloc((size_t)N_C * Dc * sizeof(float));
    float *ctx_pe     = (float *)malloc((size_t)N_C * Dc * sizeof(float));
    /* Token ping-pong + intermediate norm. */
    float *tokens     = (float *)malloc((size_t)N_Q * D  * sizeof(float));
    float *augment    = (float *)malloc((size_t)N_Q * D  * sizeof(float));
    float *tokens_b   = (float *)malloc((size_t)N_Q * D  * sizeof(float));
    float *tokens_n   = (float *)malloc((size_t)N_Q * D  * sizeof(float));
    /* MHR scratch + outputs (per layer reused, plus final). */
    const size_t mhr_scratch_floats = (size_t)1 *
        (889 + (size_t)J*8*2 + (size_t)V*3*3);
    float *mhr_scratch = (float *)malloc(mhr_scratch_floats * sizeof(float));
    float *verts_cm    = (float *)malloc((size_t)V * 3 * sizeof(float));
    float *gskel_cm    = (float *)malloc((size_t)J * 8 * sizeof(float));
    float *verts_m     = (float *)malloc((size_t)V * 3 * sizeof(float));
    float *jc_m        = (float *)malloc((size_t)J * 3 * sizeof(float));

    if (!ctx || !ctx_pe || !tokens || !augment || !tokens_b || !tokens_n ||
        !mhr_scratch || !verts_cm || !gskel_cm || !verts_m || !jc_m) {
        free(ctx); free(ctx_pe); free(tokens); free(augment); free(tokens_b);
        free(tokens_n); free(mhr_scratch); free(verts_cm); free(gskel_cm);
        free(verts_m); free(jc_m);
        return SAM3D_BODY_DECODER_E_LOAD;
    }

    for (int n = 0; n < N_C; n++) {
        for (int c = 0; c < Dc; c++) {
            ctx   [(size_t)n * Dc + c] = image_emb_chw[(size_t)c * N_C + n];
            ctx_pe[(size_t)n * Dc + c] = image_pe_chw [(size_t)c * N_C + n];
        }
    }
    memcpy(tokens,  initial_tokens,  (size_t)N_Q * D * sizeof(float));
    memcpy(augment, initial_augment, (size_t)N_Q * D * sizeof(float));

    int rc = SAM3D_BODY_DECODER_E_OK;
    float pose_raw[519], cam_raw[3];
    float pose519[519], cam3[3];
    float mp_buf[204], shape_buf[45], face_buf[72];
    float kp3d_post_flip[70 * 3];
    float kp2d_full[70 * 2], kp2d_crop[70 * 2], kp2d_dep[70];
    float pred_cam_t_world[3];

    for (int li = 0; li < m->n_layers; li++) {
        rc = sam3d_body_decoder_layer_forward(
                m, li, tokens, ctx, augment, ctx_pe, N_Q, N_C,
                n_threads, tokens_b, NULL);
        if (rc != SAM3D_BODY_DECODER_E_OK) goto done;
        { float *tmp = tokens; tokens = tokens_b; tokens_b = tmp; }

        if (out->dbg_layer_tokens_out)
            memcpy(out->dbg_layer_tokens_out + (size_t)li * N_Q * D,
                   tokens, (size_t)N_Q * D * sizeof(float));

        if (li >= m->n_layers - 1) break;

        /* Per-layer pose_output. norm_final → tokens_n. */
        rc = sam3d_body_norm_final(m, tokens, N_Q, n_threads, tokens_n);
        if (rc != SAM3D_BODY_DECODER_E_OK) goto done;

        rc = sam3d_body_apply_heads_raw(m, tokens_n /* pose_token at row 0 */,
                                        n_threads, pose_raw, cam_raw);
        if (rc != SAM3D_BODY_DECODER_E_OK) goto done;
        if (out->dbg_layer_pose_raw)
            memcpy(out->dbg_layer_pose_raw + (size_t)li * 519,
                   pose_raw, 519 * sizeof(float));
        for (int i = 0; i < 519; i++) pose519[i] = pose_raw[i] + ip[i];
        for (int i = 0; i < 3;   i++) cam3[i]    = cam_raw[i]  + ic[i];

        rc = sam3d_body_decode_pose_raw(m, pose519, /*enable_hand_model*/0,
                                        mp_buf, shape_buf, face_buf);
        if (rc != SAM3D_BODY_DECODER_E_OK) goto done;

        /* mhr_forward: outputs in cm. apply_correctives=1 (do_pcblend=True
         * upstream default at intermediate token_to_pose_output_fn). */
        rc = sam3d_body_mhr_forward((const sam3d_body_mhr_assets *)mhr,
                                    mp_buf, shape_buf, face_buf,
                                    1, 1, n_threads, mhr_scratch,
                                    verts_cm, gskel_cm);
        if (rc) { rc = SAM3D_BODY_DECODER_E_LOAD; goto done; }

        /* cm → m. */
        for (int i = 0; i < V * 3; i++) verts_m[i] = verts_cm[i] * 0.01f;
        s3db_extract_jcoords_meters(gskel_cm, J, jc_m);

        rc = sam3d_body_keypoints_from_mesh(m, verts_m, jc_m,
                                            /*enable_hand_model*/0,
                                            n_threads, kp3d_post_flip);
        if (rc != SAM3D_BODY_DECODER_E_OK) goto done;

        rc = sam3d_body_camera_project(kp3d_post_flip, cam3, cam_batch, K,
                                       kp2d_full, kp2d_crop, kp2d_dep,
                                       pred_cam_t_world);
        if (rc != SAM3D_BODY_DECODER_E_OK) goto done;

        if (out->dbg_layer_kp3d)
            memcpy(out->dbg_layer_kp3d + (size_t)li * K * 3,
                   kp3d_post_flip, (size_t)K * 3 * sizeof(float));
        if (out->dbg_layer_kp2d_crop)
            memcpy(out->dbg_layer_kp2d_crop + (size_t)li * K * 2,
                   kp2d_crop, (size_t)K * 2 * sizeof(float));
        if (out->dbg_layer_kp2d_depth)
            memcpy(out->dbg_layer_kp2d_depth + (size_t)li * K,
                   kp2d_dep, (size_t)K * sizeof(float));
        if (out->dbg_layer_cam_t)
            memcpy(out->dbg_layer_cam_t + (size_t)li * 3,
                   pred_cam_t_world, 3 * sizeof(float));

        rc = sam3d_body_kp_token_update(
                m, li, image_emb_chw, H, W,
                kp2d_crop, kp2d_dep, kp3d_post_flip,
                N_Q, n_threads, tokens, augment);
        if (rc != SAM3D_BODY_DECODER_E_OK) goto done;
    }

    /* Final norm + heads. */
    rc = sam3d_body_norm_final(m, tokens, N_Q, n_threads, tokens_n);
    if (rc != SAM3D_BODY_DECODER_E_OK) goto done;

    rc = sam3d_body_apply_heads_raw(m, tokens_n, n_threads, pose_raw, cam_raw);
    if (rc != SAM3D_BODY_DECODER_E_OK) goto done;
    for (int i = 0; i < 519; i++) out->mhr_params[i] = pose_raw[i] + ip[i];
    for (int i = 0; i < 3;   i++) out->cam_t[i]      = cam_raw[i]  + ic[i];

    rc = sam3d_body_decode_pose_raw(m, out->mhr_params, /*enable_hand_model*/0,
                                    mp_buf, shape_buf, face_buf);
    if (rc != SAM3D_BODY_DECODER_E_OK) goto done;

    rc = sam3d_body_mhr_forward((const sam3d_body_mhr_assets *)mhr,
                                mp_buf, shape_buf, face_buf,
                                1, 1, n_threads, mhr_scratch,
                                verts_cm, gskel_cm);
    if (rc) { rc = SAM3D_BODY_DECODER_E_LOAD; goto done; }

    for (int i = 0; i < V * 3; i++) verts_m[i] = verts_cm[i] * 0.01f;
    s3db_extract_jcoords_meters(gskel_cm, J, jc_m);

    rc = sam3d_body_keypoints_from_mesh(m, verts_m, jc_m,
                                        /*enable_hand_model*/0,
                                        n_threads, out->pred_keypoints_3d);
    if (rc != SAM3D_BODY_DECODER_E_OK) goto done;

    rc = sam3d_body_camera_project(out->pred_keypoints_3d, out->cam_t,
                                   cam_batch, K,
                                   out->pred_keypoints_2d,
                                   out->pred_keypoints_2d_cropped,
                                   out->pred_keypoints_2d_depth,
                                   out->pred_cam_t_world);
    if (rc != SAM3D_BODY_DECODER_E_OK) goto done;

    /* Optional outputs. Vertices: post-flip cam frame in m. */
    if (out->pred_vertices) {
        for (int i = 0; i < V; i++) {
            out->pred_vertices[i*3 + 0] =  verts_m[i*3 + 0];
            out->pred_vertices[i*3 + 1] = -verts_m[i*3 + 1];
            out->pred_vertices[i*3 + 2] = -verts_m[i*3 + 2];
        }
    }
    if (out->mhr_model_params) memcpy(out->mhr_model_params, mp_buf,
                                      204 * sizeof(float));
    if (out->shape) memcpy(out->shape, shape_buf, 45 * sizeof(float));
    if (out->face)  memcpy(out->face,  face_buf,  72 * sizeof(float));
    if (out->global_skel) memcpy(out->global_skel, gskel_cm,
                                 (size_t)J * 8 * sizeof(float));

done:
    free(ctx); free(ctx_pe); free(tokens); free(augment); free(tokens_b);
    free(tokens_n); free(mhr_scratch); free(verts_cm); free(gskel_cm);
    free(verts_m); free(jc_m);
    return rc;
}

int sam3d_body_decoder_forward(
    const sam3d_body_decoder_model *m,
    const struct sam3d_body_mhr_assets_t *mhr,
    const sam3d_body_camera_batch *cam_batch,
    const float *image_tokens_patch,
    int H, int W,
    const float *rays_hwc,
    const float *condition_info,
    int n_threads,
    sam3d_body_decoder_full_result *out)
{
    if (!m || !mhr || !cam_batch || !image_tokens_patch || !rays_hwc ||
        !condition_info || !out)
        return SAM3D_BODY_DECODER_E_INVAL;
    if (H <= 0 || W <= 0)
        return SAM3D_BODY_DECODER_E_INVAL;

    const int Dc = m->kv_dim;                              /* 1280 */
    const int D  = m->dim;                                 /* 1024 */
    const int N_Q = 1 + 1 + 1 + m->n_hand_tokens
                  + m->n_keypoints + m->n_keypoints;       /* 145 */
    const int N_C = H * W;
    const size_t b_emb = (size_t)Dc * (size_t)N_C * sizeof(float);

    int rc = SAM3D_BODY_DECODER_E_OK;
    float *img_emb_pre  = NULL;
    float *img_emb_post = NULL;
    float *img_pe_chw   = NULL;
    float *tokens_init  = NULL;
    float *augment_init = NULL;

    img_emb_pre  = (float *)malloc(b_emb);
    img_emb_post = (float *)malloc(b_emb);
    img_pe_chw   = (float *)malloc(b_emb);
    tokens_init  = (float *)calloc(N_Q * D, sizeof(float));
    augment_init = (float *)calloc(N_Q * D, sizeof(float));
    if (!img_emb_pre || !img_emb_post || !img_pe_chw ||
        !tokens_init || !augment_init) {
        rc = SAM3D_BODY_DECODER_E_INVAL;
        goto done_fwd;
    }

    /* Transpose patch tokens (H*W, Dc) → (Dc, H*W). */
    for (int p = 0; p < N_C; p++) {
        const float *src = image_tokens_patch + (size_t)p * Dc;
        for (int c = 0; c < Dc; c++) {
            img_emb_pre[(size_t)c * N_C + p] = src[c];
        }
    }

    rc = sam3d_body_ray_cond_emb_forward(m, img_emb_pre, rays_hwc, H, W,
                                         n_threads, img_emb_post);
    if (rc != SAM3D_BODY_DECODER_E_OK) goto done_fwd;

    rc = sam3d_body_get_dense_pe(m, H, W, n_threads, img_pe_chw);
    if (rc != SAM3D_BODY_DECODER_E_OK) goto done_fwd;

    const float *ip = (const float *)m->init_pose.data;     /* (519,) */
    const float *ic = (const float *)m->init_camera.data;   /* (3,)   */
    float init_input[525];
    float prev_input[522];
    float prompt_in[1280];
    memcpy(init_input,            condition_info, 3   * sizeof(float));
    memcpy(init_input + 3,        ip,             519 * sizeof(float));
    memcpy(init_input + 3 + 519,  ic,             3   * sizeof(float));
    memcpy(prev_input,            ip,             519 * sizeof(float));
    memcpy(prev_input + 519,      ic,             3   * sizeof(float));
    rc = sam3d_body_invalid_prompt_token(m, prompt_in);
    if (rc != SAM3D_BODY_DECODER_E_OK) goto done_fwd;

    rc = sam3d_body_build_tokens(m, init_input, prev_input, prompt_in,
                                 n_threads, tokens_init, augment_init);
    if (rc != SAM3D_BODY_DECODER_E_OK) goto done_fwd;

    rc = sam3d_body_decoder_forward_full(
            m, mhr, cam_batch,
            img_emb_post, img_pe_chw, H, W,
            tokens_init, augment_init,
            n_threads, out);

done_fwd:
    free(img_emb_pre); free(img_emb_post); free(img_pe_chw);
    free(tokens_init); free(augment_init);
    return rc;
}

#endif /* SAM3D_BODY_DECODER_FULL_IMPLEMENTATION */

#endif /* SAM3D_BODY_DECODER_IMPLEMENTATION */
#endif /* SAM3D_BODY_DECODER_H */
