/*
 * sam3d_runner.h — CPU runner for FAIR's SAM 3D Objects.
 *
 * Pipeline (matches sam3d_objects/pipeline/inference_pipeline_pointmap.py):
 *   preprocess → DINOv2 tokens + MoGe pointmap patches + mask →
 *   CondEmbedderFuser → SS Flow DiT (flow-matching, 25 steps) →
 *   SS-VAE decoder (64³ occupancy) → voxel prune →
 *   SLAT Flow DiT (sparse, shift-window attn) → SLAT GS decoder →
 *   Gaussian splat (.ply).
 *
 * Stage boundaries are resumable so each verify_*.c can stop anywhere.
 * MoGe is stubbed in v1 — caller supplies a precomputed pointmap.
 *
 * All buffers are row-major, float32 on host. Multi-safetensors
 * checkpoint is discovered from the pipeline.yaml path.
 */

#ifndef SAM3D_RUNNER_H
#define SAM3D_RUNNER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Path to pipeline.yaml (tells us which submodule safetensors
     * files to load and from which relative directory). */
    const char *pipeline_yaml;
    /* Optional explicit path to the per-module safetensors directory
     * (the output of cpu/sam3d/convert_ckpt.py). Falls back to
     * "<pipeline_yaml dir>/../safetensors/" if NULL. */
    const char *safetensors_dir;
    /* Random seed forwarded to the flow-matching sampler. */
    uint64_t    seed;
    int         ss_steps;      /* default 2  (shortcut ODE) */
    int         slat_steps;    /* default 12 (standard flow) */
    float       cfg_scale;     /* default 2.0 (SS shape cfg strength) */
    int         n_threads;     /* default: OMP_NUM_THREADS or 1 */
    int         verbose;
} sam3d_config;

typedef struct sam3d_ctx sam3d_ctx;

/* 14 floats per gaussian in PLY-export order:
 *   x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity
 *   scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
 * (17 scalars; the "14" shorthand refers to the logical groups.) */
#define SAM3D_GS_STRIDE 17

/* Lifecycle. */
sam3d_ctx *sam3d_create(const sam3d_config *cfg);
void       sam3d_destroy(sam3d_ctx *ctx);

/* Resolved per-module safetensors directory (either cfg.safetensors_dir
 * or siblingof(pipeline_yaml)/safetensors). Valid for ctx lifetime. */
const char *sam3d_safetensors_dir(const sam3d_ctx *ctx);

/* Input setters — idempotent, callable before any run_*. */
int sam3d_set_image_rgba(sam3d_ctx *ctx, const uint8_t *rgba,
                         int width, int height);
int sam3d_set_mask(sam3d_ctx *ctx, const uint8_t *mask,
                   int width, int height);
/* Pointmap float32 [H, W, 3]. v1: required (MoGe port deferred). */
int sam3d_set_pointmap(sam3d_ctx *ctx, const float *pmap,
                       int width, int height);

/* Stage entrypoints — each presumes prior stages already ran (or
 * refs were injected). */
int sam3d_run_dinov2(sam3d_ctx *ctx);
int sam3d_run_cond_fuser(sam3d_ctx *ctx);
int sam3d_run_ss_dit(sam3d_ctx *ctx);
int sam3d_run_ss_decode(sam3d_ctx *ctx);
int sam3d_run_slat_dit(sam3d_ctx *ctx);
int sam3d_run_slat_gs_decode(sam3d_ctx *ctx);

/* Read-back accessors. Return 0 on success; fill *out_n and *out_c. */
int sam3d_get_dinov2_tokens(sam3d_ctx *ctx, float *out, int *out_n, int *out_c);
int sam3d_get_cond_tokens(sam3d_ctx *ctx, float *out, int *out_n, int *out_c);
int sam3d_get_ss_latent(sam3d_ctx *ctx, float *out, int *out_dims /*4*/);
int sam3d_get_occupancy(sam3d_ctx *ctx, float *out, int *out_dims /*3*/);
int sam3d_get_slat_tokens(sam3d_ctx *ctx, float *out_feats,
                          int32_t *out_coords, int *out_n, int *out_c);
/* Gaussians: [n_gauss, SAM3D_GS_STRIDE]. */
int sam3d_get_gaussians(sam3d_ctx *ctx, float *out, int *out_n);

/* Debug: override stage inputs with pytorch-reference tensors, so
 * verify_*.c can isolate drift to a single stage. */
int sam3d_debug_override_dinov2(sam3d_ctx *ctx, const float *tokens,
                                int n, int c);
int sam3d_debug_override_cond(sam3d_ctx *ctx, const float *tokens,
                              int n, int c);
int sam3d_debug_override_ss_latent(sam3d_ctx *ctx, const float *latent,
                                   const int *dims);
int sam3d_debug_override_occupancy(sam3d_ctx *ctx, const float *occ,
                                   const int *dims);
int sam3d_debug_override_slat(sam3d_ctx *ctx, const float *feats,
                              const int32_t *coords, int n, int c);

#ifdef __cplusplus
}
#endif

#endif /* SAM3D_RUNNER_H */
