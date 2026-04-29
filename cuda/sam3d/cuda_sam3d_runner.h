/* CUDA runner for FAIR's SAM 3D Objects (image → 3D Gaussian Splat).
 *
 * Mirrors the CPU runner in cpu/sam3d/sam3d_runner.h so verify_*.c
 * binaries (yet to be ported) can diff against the same
 * /tmp/sam3d_ref/ dumps. Stage entry points return
 * CUDA_SAM3D_E_NOT_IMPLEMENTED until their NVRTC kernels land.
 *
 * Pipeline order (same as CPU):
 *   set_image / set_mask / set_pointmap →
 *   run_dinov2 → run_cond_fuser → run_ss_dit → run_ss_decode →
 *   run_slat_dit → run_slat_gs_decode → get_gaussians.
 *
 * Per-module safetensors (matches cpu/sam3d/convert_ckpt.py output):
 *   sam3d_dinov2.safetensors            (DINOv2-L/14+reg encoder)
 *   sam3d_point_patch_embed.safetensors (pointmap embed)
 *   sam3d_cond_fuser.safetensors        (Llama SwiGLU fuser)
 *   sam3d_ss_dit.safetensors            (shortcut DiT, 24 blocks)
 *   sam3d_ss_decoder.safetensors        (3D-conv SS-VAE)
 *   sam3d_slat_dit.safetensors          (SLAT flow, 24 blocks)
 *   sam3d_slat_gs_decoder.safetensors   (Gaussian-splat decoder)
 */

#ifndef CUDA_SAM3D_RUNNER_H_
#define CUDA_SAM3D_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_sam3d_ctx cuda_sam3d_ctx;

typedef struct {
    /* Per-module safetensors directory (same files the CPU runner reads).
     * Falls back to siblingof(pipeline_yaml)/safetensors if NULL. */
    const char *safetensors_dir;
    /* Optional path to pipeline.yaml — used to resolve safetensors_dir
     * when the explicit path is NULL. */
    const char *pipeline_yaml;
    int      device_ordinal;
    int      verbose;
    /* Sampler knobs (forwarded to flow-matching schedulers). */
    uint64_t seed;
    int      ss_steps;     /* default 2  (shortcut ODE) */
    int      slat_steps;   /* default 12 (standard flow) */
    float    cfg_scale;    /* default 2.0 (SS shape CFG strength) */
    /* Compute precision for tensor-core paths. "fp16" | "bf16" | "fp32".
     * NULL or "" → "fp16". bf16 matches upstream's reference. */
    const char *precision;
} cuda_sam3d_config;

/* Same 17-scalar PLY-export channel order as the CPU runner. */
#define CUDA_SAM3D_GS_STRIDE 17

cuda_sam3d_ctx *cuda_sam3d_create(const cuda_sam3d_config *cfg);
void            cuda_sam3d_destroy(cuda_sam3d_ctx *ctx);

/* Inputs — idempotent, callable before any run_*. */
int cuda_sam3d_set_image_rgba(cuda_sam3d_ctx *ctx, const uint8_t *rgba,
                              int width, int height);
int cuda_sam3d_set_mask(cuda_sam3d_ctx *ctx, const uint8_t *mask,
                        int width, int height);
/* Pointmap float32 [H, W, 3]. v1: required (MoGe port deferred). */
int cuda_sam3d_set_pointmap(cuda_sam3d_ctx *ctx, const float *pmap,
                            int width, int height);

/* Stage dispatchers. Each presumes prior stages already ran (or refs
 * were injected via cuda_sam3d_debug_override_*). */
int cuda_sam3d_run_dinov2(cuda_sam3d_ctx *ctx);
int cuda_sam3d_run_cond_fuser(cuda_sam3d_ctx *ctx);
int cuda_sam3d_run_ss_dit(cuda_sam3d_ctx *ctx);
int cuda_sam3d_run_ss_decode(cuda_sam3d_ctx *ctx);
int cuda_sam3d_run_slat_dit(cuda_sam3d_ctx *ctx);
int cuda_sam3d_run_slat_gs_decode(cuda_sam3d_ctx *ctx);

/* Read-back accessors. Pass NULL `out` to query shape only. */
int cuda_sam3d_get_dinov2_tokens(cuda_sam3d_ctx *ctx, float *out,
                                 int *out_n, int *out_c);
int cuda_sam3d_get_cond_tokens(cuda_sam3d_ctx *ctx, float *out,
                               int *out_n, int *out_c);
int cuda_sam3d_get_ss_latent(cuda_sam3d_ctx *ctx, float *out, int *out_dims /*4*/);
int cuda_sam3d_get_occupancy(cuda_sam3d_ctx *ctx, float *out, int *out_dims /*3*/);
int cuda_sam3d_get_slat_tokens(cuda_sam3d_ctx *ctx, float *out_feats,
                               int32_t *out_coords, int *out_n, int *out_c);
int cuda_sam3d_get_gaussians(cuda_sam3d_ctx *ctx, float *out, int *out_n);

/* Debug overrides — feed pytorch ref tensors at any stage boundary so a
 * single verify_*.c can isolate drift. */
int cuda_sam3d_debug_override_dinov2(cuda_sam3d_ctx *ctx, const float *tokens,
                                     int n, int c);
int cuda_sam3d_debug_override_cond(cuda_sam3d_ctx *ctx, const float *tokens,
                                   int n, int c);

/* Single-call SS DiT forward — bypasses the integrator so verify_ss_dit can
 * diff per-call against pytorch. Lazy-loads the DiT model. latents_in/out
 * are 5 buffers per modality (sized via cuda_sam3d_ss_dit_lat_elts).
 * cond: [n_cond × cond_channels] f32; query channels via
 * cuda_sam3d_ss_dit_info(). */
int cuda_sam3d_debug_ss_dit_forward(cuda_sam3d_ctx *ctx,
                                    const float *const *latents_in,
                                    float *const *latents_out,
                                    const float *cond, int n_cond,
                                    float t, float d);

/* Architecture query (lazy-loads DiT model). Any out_* may be NULL. */
int cuda_sam3d_ss_dit_info(cuda_sam3d_ctx *ctx,
                           int *out_n_blocks, int *out_dim,
                           int *out_cond_channels, int *out_is_shortcut);

int cuda_sam3d_ss_dit_n_latents(void);
int cuda_sam3d_ss_dit_lat_elts (int modality_id);

/* Single-call SLAT DiT forward — bypasses the integrator so verify_slat_dit
 * can diff per-call against pytorch. Lazy-loads the SLAT DiT model.
 * coords[N,4] (b,z,y,x) i32, feats[N,in_channels] f32, returns out_feats
 * [N,out_channels] f32 in the caller-provided buffer. cond:
 * [n_cond × cond_channels] f32. */
int cuda_sam3d_debug_slat_dit_forward(cuda_sam3d_ctx *ctx,
                                      const int32_t *coords,
                                      const float *feats, int N,
                                      float t,
                                      const float *cond, int n_cond,
                                      float *out_feats);

/* Architecture query (lazy-loads SLAT DiT model). Any out_* may be NULL. */
int cuda_sam3d_slat_dit_info(cuda_sam3d_ctx *ctx,
                             int *out_in_channels, int *out_out_channels,
                             int *out_cond_channels);

/* Single transformer forward of the SLAT GS decoder — bypasses the
 * to_representation step. Returns malloc'd [N × out_channels] f32 in
 * *out_feats (caller free()s). */
int cuda_sam3d_debug_slat_gs_transformer(cuda_sam3d_ctx *ctx,
                                         const int32_t *coords,
                                         const float *feats, int N,
                                         float **out_feats, int *out_c);

/* Decode raw out_feats[N, out_channels] (e.g. from
 * cuda_sam3d_debug_slat_gs_transformer) into per-gaussian buffers. Each
 * out pointer may be NULL to skip. Sizes:
 *   xyz [N*G, 3], dc [N*G, 3], scaling [N*G, 3], rotation [N*G, 4],
 *   opacity [N*G]. G = cuda_sam3d_slat_gs_info().num_gaussians. */
int cuda_sam3d_debug_slat_gs_to_representation(cuda_sam3d_ctx *ctx,
                                               const int32_t *coords,
                                               const float *feats_out, int N,
                                               float *xyz_out, float *dc_out,
                                               float *scaling_out, float *rotation_out,
                                               float *opacity_out);

/* CUDA fused to_representation + PLY-layout pack for raw GS decoder
 * out_feats. Returns malloc'd [N*G, CUDA_SAM3D_GS_STRIDE] f32. */
int cuda_sam3d_debug_slat_gs_pack_ply(cuda_sam3d_ctx *ctx,
                                      const int32_t *coords,
                                      const float *feats_out, int N,
                                      float **out_ply, int *out_total);

/* Architecture query (lazy-loads SLAT GS decoder model). Any out_* may be NULL. */
int cuda_sam3d_slat_gs_info(cuda_sam3d_ctx *ctx,
                            int *out_in_channels, int *out_out_channels,
                            int *out_num_gaussians);
int cuda_sam3d_debug_override_ss_latent(cuda_sam3d_ctx *ctx, const float *latent,
                                        const int *dims);
int cuda_sam3d_debug_override_occupancy(cuda_sam3d_ctx *ctx, const float *occ,
                                        const int *dims);
int cuda_sam3d_debug_override_slat(cuda_sam3d_ctx *ctx, const float *feats,
                                   const int32_t *coords, int n, int c);

/* Status codes. */
#define CUDA_SAM3D_E_OK              ( 0)
#define CUDA_SAM3D_E_INVAL           (-1)
#define CUDA_SAM3D_E_NOT_IMPLEMENTED (-2)
#define CUDA_SAM3D_E_LOAD            (-3)
#define CUDA_SAM3D_E_NO_INPUT        (-4)

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SAM3D_RUNNER_H_ */
