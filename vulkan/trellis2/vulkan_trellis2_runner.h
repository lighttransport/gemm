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

/* Free runner and all GPU resources */
void vulkan_trellis2_free(vulkan_trellis2_runner *r);

#ifdef __cplusplus
}
#endif

#endif /* VULKAN_TRELLIS2_RUNNER_H */
