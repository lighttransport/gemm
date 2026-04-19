/*
 * vulkan_trellis2_runner.h - Vulkan TRELLIS.2 Stage 1 runner (image -> 3D occupancy)
 *
 * Uses pre-compiled SPIR-V shaders. F32 weights on GPU (HOST_VISIBLE SSBOs).
 * Mirrors the CUDA runner API with Vulkan compute backend.
 *
 * Pipeline:
 *   1. DINOv3 ViT-L/16 encoder -> [1029, 1024] features
 *   2. DiT diffusion transformer (30 blocks, flow matching) -> [8, 16, 16, 16] latent
 *   3. Structure decoder (Conv3D + ResBlocks) -> [1, 64, 64, 64] occupancy
 *
 * Usage:
 *   vulkan_trellis2_runner *r = vulkan_trellis2_init(0, 1);
 *   vulkan_trellis2_load_weights(r, "dinov3.safetensors",
 *                                    "stage1_dit.safetensors",
 *                                    "decoder.safetensors");
 *   // Run full pipeline with pre-computed features
 *   float occupancy[64*64*64];
 *   vulkan_trellis2_run_stage1(r, features, noise, occupancy, 12, 7.5f, 0.7f, 42);
 *   vulkan_trellis2_free(r);
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef VULKAN_TRELLIS2_RUNNER_H
#define VULKAN_TRELLIS2_RUNNER_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vulkan_trellis2_runner vulkan_trellis2_runner;

/* Initialize Vulkan context and load shaders */
vulkan_trellis2_runner *vulkan_trellis2_init(int device_id, int verbose);

/* Load weights from safetensors files:
 *   dinov3_path:  DINOv3 ViT-L/16 encoder weights (F32, timm format)
 *   dit_path:     Stage 1 DiT weights (BF16)
 *   decoder_path: Structure decoder weights (F16/F32) */
int vulkan_trellis2_load_weights(vulkan_trellis2_runner *r,
                                  const char *dinov3_path,
                                  const char *dit_path,
                                  const char *decoder_path);

/* ---- Per-stage verification API ---- */

/* Run DINOv3 encoder.
 *   image_f32: [3, 512, 512] F32 pre-processed image (ImageNet-normalized, CHW)
 *   output:    [1029, 1024] F32 buffer (must be pre-allocated)
 *   Returns 0 on success. */
int vulkan_trellis2_run_dinov3(vulkan_trellis2_runner *r,
                                const float *image_f32,
                                float *output);

/* Run DiT single forward pass.
 *   x_t:       [4096, 8] or [8, 16, 16, 16] F32 noisy latent (token-major)
 *   timestep:  scalar timestep (already rescaled)
 *   features:  [1029, 1024] F32 conditioning from DINOv3
 *   output:    [4096, 8] F32 predicted velocity (must be pre-allocated)
 *   Returns 0 on success. */
int vulkan_trellis2_run_dit(vulkan_trellis2_runner *r,
                             const float *x_t,
                             float timestep,
                             const float *features,
                             float *output);

/* Run structure decoder.
 *   latent:    [8, 16, 16, 16] F32 latent (channel-first NCDHW)
 *   output:    [64*64*64] F32 occupancy logits (must be pre-allocated)
 *   Returns 0 on success. */
int vulkan_trellis2_run_decoder(vulkan_trellis2_runner *r,
                                 const float *latent,
                                 float *output);

/* Run full Stage 1 pipeline: features + noise -> occupancy.
 *   features:       [1029, 1024] F32 DINOv3 conditioning
 *   initial_noise:  [8, 16, 16, 16] F32 initial noise (NULL = generate from seed)
 *   output:         [64*64*64] F32 occupancy logits (must be pre-allocated)
 *   n_steps:        Euler steps (default 12)
 *   cfg_scale:      Guidance scale (default 7.5)
 *   cfg_rescale:    CFG rescale factor (default 0.7, 0 = no rescale)
 *   seed:           Random seed (used if initial_noise is NULL)
 *   Returns 0 on success. */
int vulkan_trellis2_run_stage1(vulkan_trellis2_runner *r,
                                const float *features,
                                const float *initial_noise,
                                float *output,
                                int n_steps,
                                float cfg_scale,
                                float cfg_rescale,
                                uint32_t seed);

/* ---- Stage 2/3 sparse DiT API ---- */

/* Load Stage 2 (shape flow) DiT weights (separate file from Stage 1) */
int vulkan_trellis2_load_stage2(vulkan_trellis2_runner *r, const char *path);

/* Load Stage 3 (texture flow) DiT weights */
int vulkan_trellis2_load_stage3(vulkan_trellis2_runner *r, const char *path);

/* Run Stage 2 shape DiT single forward step.
 *   x_t:    [N, 32] F32 sparse voxel features (token-major)
 *   coords: [N, 4] int32 (batch, z, y, x) — for RoPE computation
 *   cond:   [1029, 1024] F32 DINOv3 features
 *   output: [N, 32] F32 predicted velocity */
int vulkan_trellis2_run_stage2_dit(vulkan_trellis2_runner *r,
                                     const float *x_t, float timestep,
                                     const float *cond_features,
                                     const int32_t *coords, int N,
                                     float *output);

/* Run Stage 3 texture DiT single forward step.
 *   x_t:    [N, 64] = [noise_32, shape_slat_norm_32] concatenated
 *   coords: [N, 4] int32 — same sparse coords as Stage 2
 *   cond:   [1029, 1024] F32 DINOv3 features
 *   output: [N, 32] F32 predicted velocity (texture channels only) */
int vulkan_trellis2_run_stage3_dit(vulkan_trellis2_runner *r,
                                     const float *x_t, float timestep,
                                     const float *cond_features,
                                     const int32_t *coords, int N,
                                     float *output);

/* ---- Shape decoder (SC-VAE) ---- */

/* Run shape decoder (currently CPU-based, uses sparse 3D conv).
 *   dec_path:   path to shape decoder safetensors
 *   slat:       [N, 32] F32 denormalized shape latent from Stage 2
 *   coords:     [N, 4] int32 sparse coordinates
 *   out_feats:  [*out_N, 7] F32 per-voxel predictions (pre-allocated for max)
 *   out_coords: [*out_N, 4] int32 upsampled coords (pre-allocated for max)
 *   out_N:      actual number of output voxels (written on return) */
int vulkan_trellis2_run_shape_decoder(vulkan_trellis2_runner *r,
                                        const char *dec_path,
                                        const float *slat,
                                        const int32_t *coords, int N,
                                        float *out_feats,
                                        int32_t *out_coords,
                                        int *out_N);

/* ---- Full Stage 2/3 sampling pipelines ---- */

/* Run full Stage 2 pipeline: occupancy + features -> shape latent.
 *   features:     [1029, 1024] F32 DINOv3 conditioning
 *   occupancy:    [64*64*64] F32 occupancy logits from Stage 1
 *   shape_latent: [*out_N, 32] F32 output (must be pre-allocated for max N)
 *   out_coords:   [*out_N, 4] int32 output (must be pre-allocated)
 *   out_N:        number of sparse voxels (written on return)
 *   n_steps:      Euler steps (default 12)
 *   cfg_scale:    guidance scale (default 7.5)
 *   cfg_rescale:  rescale factor (default 0.7, 0 = none) */
int vulkan_trellis2_run_stage2(vulkan_trellis2_runner *r,
                                const float *features,
                                const float *occupancy,
                                float *shape_latent,
                                int32_t *out_coords,
                                int *out_N,
                                int n_steps,
                                float cfg_scale,
                                float cfg_rescale,
                                uint32_t seed);

/* Run full Stage 3 pipeline: shape_latent + features -> texture latent.
 *   features:     [1029, 1024] F32 DINOv3 conditioning
 *   shape_latent: [N, 32] F32 denormalized shape latent from Stage 2
 *   coords:       [N, 4] int32 sparse coordinates from Stage 2
 *   tex_latent:   [N, 32] F32 output (must be pre-allocated)
 *   n_steps:      Euler steps (default 12)
 *   seed:         Random seed (Stage 3 uses seed+2 internally) */
int vulkan_trellis2_run_stage3(vulkan_trellis2_runner *r,
                                const float *features,
                                const float *shape_latent,
                                const int32_t *coords, int N,
                                float *tex_latent,
                                int n_steps,
                                uint32_t seed);

/* Free runner and all GPU resources */
void vulkan_trellis2_free(vulkan_trellis2_runner *r);

#ifdef __cplusplus
}
#endif

#endif /* VULKAN_TRELLIS2_RUNNER_H */
