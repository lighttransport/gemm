/* HIP/ROCm runner for SAM 3D Body (DINOv3 variant, RDNA4).
 *
 * Mirrors the CPU runner's stage boundaries so verify_*.c binaries can
 * diff against the same /tmp/sam3d_body_ref/ npy dumps.
 *
 * Kernel organization: HIPRTC source strings in hip_sam3d_body_kernels.h,
 * runtime compile via rocew.h in hip_sam3d_body_runner.c.
 */

#ifndef HIP_SAM3D_BODY_RUNNER_H_
#define HIP_SAM3D_BODY_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_sam3d_body_ctx hip_sam3d_body_ctx;

/* Backbone variant. Mirrors the CPU runner's sam3d_body_backbone_t — kept
 * as a separate enum here so the CUDA header doesn't pull in the CPU
 * runner. Default is DINOV3 (v1). */
typedef enum {
    HIP_SAM3D_BODY_BACKBONE_DINOV3 = 0,
    HIP_SAM3D_BODY_BACKBONE_VITH   = 1,
} hip_sam3d_body_backbone_t;

typedef struct {
    const char *safetensors_dir;
    const char *mhr_assets_dir;
    int image_size;          /* 512 for dinov3_vith16plus; for VITH this
                              * is the input height (W is fixed at 384) */
    int device_ordinal;
    int verbose;
    /* precision: "bf16" (upstream/PyTorch reference), "fp16" (faster).
     * NULL or "" -> "bf16". */
    const char *precision;
    /* Backbone variant. Defaults to DINOV3 (=0) when zero-initialized. */
    hip_sam3d_body_backbone_t backbone;
} hip_sam3d_body_config;

hip_sam3d_body_ctx *hip_sam3d_body_create(const hip_sam3d_body_config *cfg);
void                 hip_sam3d_body_destroy(hip_sam3d_body_ctx *ctx);

/* Inputs. */
int hip_sam3d_body_set_image(hip_sam3d_body_ctx *ctx, const uint8_t *rgb,
                              int width, int height, const float bbox[4]);
int hip_sam3d_body_set_focal(hip_sam3d_body_ctx *ctx, float focal_px);

/* Stages. */
int hip_sam3d_body_run_encoder(hip_sam3d_body_ctx *ctx);
int hip_sam3d_body_run_decoder(hip_sam3d_body_ctx *ctx);
int hip_sam3d_body_run_mhr(hip_sam3d_body_ctx *ctx);
int hip_sam3d_body_run_all(hip_sam3d_body_ctx *ctx);

/* Readbacks (host buffers; pass NULL to query shape). */
int hip_sam3d_body_get_encoder_tokens(hip_sam3d_body_ctx *ctx,
                                       float *out, int *out_n, int *out_dim);
int hip_sam3d_body_get_mhr_params(hip_sam3d_body_ctx *ctx,
                                   float *out, int *out_n);
int hip_sam3d_body_get_cam(hip_sam3d_body_ctx *ctx,
                            float *out_cam_t_xyz, float *out_focal_px);
int hip_sam3d_body_get_vertices(hip_sam3d_body_ctx *ctx, float *out, int *out_v);
int hip_sam3d_body_get_faces(hip_sam3d_body_ctx *ctx, int32_t *out, int *out_f);
int hip_sam3d_body_get_keypoints_3d(hip_sam3d_body_ctx *ctx, float *out, int *out_k);
int hip_sam3d_body_get_keypoints_2d(hip_sam3d_body_ctx *ctx, float *out, int *out_k);

/* Debug overrides for per-stage verify. */
int hip_sam3d_body_debug_override_encoder(hip_sam3d_body_ctx *ctx,
                                           const float *tokens, int n, int dim);
int hip_sam3d_body_debug_override_mhr_params(hip_sam3d_body_ctx *ctx,
                                              const float *params, int n);

/* Set the encoder input directly from a pre-normalized (3, IMG, IMG) f32
 * tensor — bypasses set_image's u8 upload + on-device resize/normalize.
 * Used by verify_dinov3 to feed the same tensor as the Python reference. */
int hip_sam3d_body_debug_set_normalized_input(hip_sam3d_body_ctx *ctx,
                                               const float *chw, int H, int W);

/* Run only the ray_cond_emb step:
 *   image_emb_chw   : (1280, H*W) f32 host buffer  (CHW)
 *   rays_hwc        : (H, W, 3) f32 host buffer
 *   out_chw         : (1280, H*W) f32 host buffer (caller-allocated)
 * H and W must match the encoder grid (typically 32x32 for image_size=512). */
int hip_sam3d_body_debug_run_ray_cond(hip_sam3d_body_ctx *ctx,
                                       const float *image_emb_chw,
                                       const float *rays_hwc,
                                       int H, int W,
                                       float *out_chw);

/* Run only the build_tokens step:
 *   init_in   (525,) f32  = [condition_info(3); init_pose(519); init_camera(3)]
 *   prev_in   (522,) f32  = [init_pose(519); init_camera(3)]
 *   prompt_in (1280,) f32 = prompt_encoder output (all-invalid keypoints)
 *   x_out     (145, 1024) f32 (caller-allocated)
 *   x_pe_out  (145, 1024) f32 (caller-allocated)
 */
int hip_sam3d_body_debug_run_build_tokens(hip_sam3d_body_ctx *ctx,
                                           const float *init_in,
                                           const float *prev_in,
                                           const float *prompt_in,
                                           float *x_out,
                                           float *x_pe_out);

/* Run one TransformerDecoderLayer forward.
 *   layer_idx ∈ [0, 6). Layer 0 has skip_first_pe=True.
 *   x_in        (N_q, 1024) f32
 *   context_in  (N_c, 1280) f32
 *   x_pe_in     (N_q, 1024) f32
 *   context_pe_in (N_c, 1280) f32
 *   x_out       (N_q, 1024) f32 (caller-allocated)
 * N_q is typically 145 (body branch); N_c is typically 1024 (32×32 grid). */
int hip_sam3d_body_debug_run_decoder_layer(hip_sam3d_body_ctx *ctx,
                                            int layer_idx,
                                            const float *x_in,
                                            const float *context_in,
                                            const float *x_pe_in,
                                            const float *context_pe_in,
                                            int N_q, int N_c,
                                            float *x_out);

/* Run one keypoint-token-update step (between-layer 2D + 3D refresh).
 *   layer_idx ∈ [0, n_layers - 1). Last layer is short-circuited per
 *     upstream guard; the host returns OK without touching tokens/augment.
 *   image_emb_chw : (1280, H*W) f32 (CHW from ray_cond_emb output).
 *   H, W          : encoder grid (typically 32, 32).
 *   kp2d_cropped  : (70, 2) f32, in [-0.5, 0.5].
 *   kp2d_depth    : (70,) f32; <1e-5 marks an invalid keypoint.
 *   kp3d_camera   : (70, 3) f32 in camera space.
 *   tokens        : (N_q, 1024) f32 in/out — kp_feat add to rows [5..75).
 *   token_augment : (N_q, 1024) f32 in/out — posemb overwrite to
 *                   rows [5..75) (2D) and [75..145) (3D).
 * N_q is typically 145. */
int hip_sam3d_body_debug_run_kp_token_update(hip_sam3d_body_ctx *ctx,
                                              int layer_idx,
                                              const float *image_emb_chw,
                                              int H, int W,
                                              const float *kp2d_cropped,
                                              const float *kp2d_depth,
                                              const float *kp3d_camera,
                                              int N_q,
                                              float *tokens,
                                              float *token_augment);

/* Apply norm_final + head_pose.proj + head_camera.proj. The "_raw" outputs
 * are the pre-init-add projections (mhr_params before init_pose / cam_t
 * before init_camera).
 *   tokens_in   (N_q, 1024) f32 — final-layer token output.
 *   tokens_norm (N_q, 1024) f32 (caller-allocated; pass NULL to skip).
 *   pose_raw    (519,) f32 (caller-allocated).
 *   cam_raw     (3,)   f32 (caller-allocated).
 */
int hip_sam3d_body_debug_run_norm_and_heads(hip_sam3d_body_ctx *ctx,
                                             const float *tokens_in,
                                             int N_q,
                                             float *tokens_norm,
                                             float *pose_raw,
                                             float *cam_raw);

/* Speculative MHR-on-GPU helpers (exploratory — off the production path).
 *
 * Step 7 was officially closed via CPU OpenMP parallelization (see PORT.md).
 * These helpers exist to validate the GPU implementation of MHR's largest
 * GEMVs and are not weight-cached (each call uploads weights). All require
 * mhr_assets_dir to be passed to hip_sam3d_body_create. Outputs match
 * the CPU counterparts:
 *   sam3d_body_mhr_blend_shape       — (B=1, V*3=55317) f32
 *   sam3d_body_mhr_face_expressions  — (B=1, V*3=55317) f32
 *   sam3d_body_mhr_pose_correctives  — (B=1, V*3=55317) f32
 */
int hip_sam3d_body_debug_run_blend_shape(hip_sam3d_body_ctx *ctx,
                                          const float *shape_coeffs,  /* (45,) */
                                          float *out_verts);          /* (V*3,) */
int hip_sam3d_body_debug_run_face_expressions(hip_sam3d_body_ctx *ctx,
                                               const float *face_coeffs, /* (72,) */
                                               float *out_verts);        /* (V*3,) */
/* pose_correctives — 6D feat + sparse matvec + ReLU on host, dense Linear
 * (55317, 3000) GEMV on GPU. joint_params shape (127, 7) f32. */
int hip_sam3d_body_debug_run_pose_correctives(hip_sam3d_body_ctx *ctx,
                                               const float *joint_params,
                                               float *out_verts);        /* (V*3,) */

/* LBS skin_points — scatter-add over K=51337 skin entries on GPU.
 *   global_skel : (J=127, 8) f32  output of local_to_global_skel walker
 *   rest_verts  : (V=18439, 3) f32
 *   out_verts   : (V*3,) f32      skinned vertex output
 * jstate = skel_multiply(global, inverse_bind_pose) is computed on the
 * host (J=127 trivial ops), uploaded once, then the kernel scatter-adds
 * via atomicAdd. Mirrors sam3d_body_mhr_skin_points (B=1). */
int hip_sam3d_body_debug_run_lbs_skin(hip_sam3d_body_ctx *ctx,
                                       const float *global_skel,
                                       const float *rest_verts,
                                       float *out_verts);

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_BODY_RUNNER_H_ */
