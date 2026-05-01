/*
 * sam3d_body_runner.h — CPU runner for FAIR's SAM 3D Body.
 *
 * Pipeline (matches sam_3d_body/models/meta_arch/sam3d_body.py):
 *   preprocess (crop + normalize) → DINOv3 / ViT-H backbone →
 *   promptable decoder (optional 2D keypoints + mask prompts) →
 *   MHR head (regress MHR params: global_rot, body_pose, shape,
 *   scale, hand_pose, face) + camera head →
 *   MHR skinning (params → 3D vertices) →
 *   camera unproject → 2D/3D keypoints + vertices in camera frame.
 *
 * Current scope: DINOv3 and ViT-H backbone variants, single-person
 * RGB input with an optional image-space bbox. CLI tools can obtain
 * that bbox from native RT-DETR-S auto-crop; this core runner treats
 * bbox selection as caller-owned and falls back to the full image.
 *
 * All buffers are row-major, float32 on host. Per-module safetensors
 * slices are produced by cpu/sam3d_body/convert_ckpt.py from the
 * HF-downloaded model.ckpt.
 */

#ifndef SAM3D_BODY_RUNNER_H
#define SAM3D_BODY_RUNNER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SAM3D_BODY_BACKBONE_DINOV3 = 0,   /* sam-3d-body-dinov3 (v1) */
    SAM3D_BODY_BACKBONE_VITH   = 1,   /* sam-3d-body-vith   (v2) */
} sam3d_body_backbone_t;

typedef struct {
    /* Directory containing the per-module safetensors slices (output of
     * convert_ckpt.py). Required. */
    const char *safetensors_dir;
    /* Path to mhr_model assets (dumped via convert_mhr_assets.py from
     * the ckpt's assets/mhr_model.pt). Required for MHR skinning. */
    const char *mhr_assets_dir;
    /* Backbone variant to load. */
    sam3d_body_backbone_t backbone;
    uint64_t seed;
    int      n_threads;       /* default: OMP_NUM_THREADS or 1 */
    int      verbose;
} sam3d_body_config;

typedef struct sam3d_body_ctx sam3d_body_ctx;

/* ---------------- lifecycle ---------------- */
sam3d_body_ctx *sam3d_body_create(const sam3d_body_config *cfg);
void            sam3d_body_destroy(sam3d_body_ctx *ctx);

/* ---------------- inputs ---------------- */
/* RGB image, arbitrary HxW. The runner handles cropping + resize to
 * the backbone's input_size. bbox in image coords (x0,y0,x1,y1); if
 * NULL the entire image is treated as the crop. */
int sam3d_body_set_image(sam3d_body_ctx *ctx, const uint8_t *rgb,
                         int width, int height, const float bbox[4]);

/* Optional prompts (v1: not wired). */
int sam3d_body_set_keypoints_2d(sam3d_body_ctx *ctx,
                                const float *kp_xy, int k);
int sam3d_body_set_mask(sam3d_body_ctx *ctx, const uint8_t *mask,
                        int width, int height);

/* Optional focal length hint (pixels). If <=0, model-estimated focal
 * is used (camera_head output). */
int sam3d_body_set_focal(sam3d_body_ctx *ctx, float focal_px);

/* ---------------- stages ---------------- */
int sam3d_body_run_encoder(sam3d_body_ctx *ctx);
int sam3d_body_run_decoder(sam3d_body_ctx *ctx);
int sam3d_body_run_mhr(sam3d_body_ctx *ctx);

/* End-to-end convenience (runs all stages in order). */
int sam3d_body_run_all(sam3d_body_ctx *ctx);

/* ---------------- readbacks ---------------- */
/* Encoder token grid. Pass out==NULL to query shape. */
int sam3d_body_get_encoder_tokens(sam3d_body_ctx *ctx, float *out,
                                  int *out_n_tokens, int *out_dim);

/* MHR regression head output — raw parameter vector (npose floats). */
int sam3d_body_get_mhr_params(sam3d_body_ctx *ctx, float *out, int *out_n);

/* Camera translation (3,) and focal length (1,). */
int sam3d_body_get_cam(sam3d_body_ctx *ctx, float *out_cam_t_xyz,
                       float *out_focal_px);

/* Skinned mesh: vertices [V,3], faces [F,3] (int32, shared across
 * persons), keypoints [K,3] 3D and [K,2] 2D-projected. */
int sam3d_body_get_vertices(sam3d_body_ctx *ctx, float *out, int *out_v);
int sam3d_body_get_faces(sam3d_body_ctx *ctx, int32_t *out, int *out_f);
int sam3d_body_get_keypoints_3d(sam3d_body_ctx *ctx, float *out, int *out_k);
int sam3d_body_get_keypoints_2d(sam3d_body_ctx *ctx, float *out, int *out_k);

/* ---------------- debug overrides ---------------- */
/* Replace a stage's input with the pytorch-reference dump so
 * verify_*.c can isolate per-stage drift. */
int sam3d_body_debug_override_encoder(sam3d_body_ctx *ctx,
                                      const float *tokens, int n, int dim);
int sam3d_body_debug_override_mhr_params(sam3d_body_ctx *ctx,
                                         const float *params, int n);

/*
 * Feed pre-computed decoder-stage inputs so run_decoder can exercise
 * the production decoder_forward_full + MHR-in-the-loop path without
 * requiring the (not-yet-ported) ray_cond / prompt_encoder / TopdownAffine
 * computation from raw image. Intended for verify_end_to_end and
 * reference-driven test_sam3d_body runs.
 *
 *   image_emb_chw : (kv_dim=1280, H, W) f32 — POST-ray_cond image embeddings
 *   image_pe_chw  : (kv_dim=1280, H, W) f32 — image_augment (context_pe)
 *   init_x        : (145, 1024)         f32 — output of build_tokens
 *   init_xpe      : (145, 1024)         f32 — token_augment from build_tokens
 *   H, W          : image-emb spatial dims (32, 32 for DINOv3-H+)
 *   cam_int       : (9,) f32 — row-major 3x3 intrinsic matrix
 *   bbox_center   : (2,) f32 bbox_scale: (1,) f32 — GetBBoxCenterScale output
 *   ori_img_size  : (2,) f32 — (W, H) of original image
 *   img_size      : (2,) f32 — (W, H) of model input (crop)
 *   affine_trans  : (6,) f32 — row-major 2x3 TopdownAffine matrix
 *   use_intrin_center : cfg.MODEL.DECODER.USE_INTRIN_CENTER (0 for DINOv3 v1)
 *   default_scale_factor : PerspectiveHead.default_scale_factor (1.0)
 */
int sam3d_body_debug_override_decoder_inputs(
    sam3d_body_ctx *ctx,
    const float *image_emb_chw,
    const float *image_pe_chw,
    int H, int W,
    const float *init_x,
    const float *init_xpe,
    const float *cam_int,
    const float *bbox_center,
    const float *bbox_scale,
    const float *ori_img_size,
    const float *img_size,
    const float *affine_trans,
    int use_intrin_center,
    float default_scale_factor);

#ifdef __cplusplus
}
#endif

#endif /* SAM3D_BODY_RUNNER_H */
