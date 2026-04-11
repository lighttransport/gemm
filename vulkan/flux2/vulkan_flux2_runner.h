/*
 * vulkan_flux2_runner.h - Vulkan Flux.2 Klein text-to-image runner
 *
 * Uses pre-compiled SPIR-V shaders and DEVICE_LOCAL SSBOs.
 * F32 weights on GPU, F32 compute.
 *
 * Port of cuda_flux2_runner.h for cross-platform Vulkan compute.
 *
 * Runs the 5 double-stream + 20 single-stream DiT blocks on GPU.
 * VAE decode falls back to CPU.
 *
 * Usage:
 *   vulkan_flux2_runner *r = vulkan_flux2_init(0, 1, "build/shaders");
 *   vulkan_flux2_load_dit(r, "flux-2-klein-4b.safetensors");
 *   vulkan_flux2_load_vae(r, "flux2-vae.safetensors");
 *   vulkan_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt, t, guidance, out);
 *   vulkan_flux2_free(r);
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef VULKAN_FLUX2_RUNNER_H
#define VULKAN_FLUX2_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vulkan_flux2_runner vulkan_flux2_runner;

/* Initialize Vulkan context and load shaders.
 * shader_dir: path to directory containing compiled .spv files */
vulkan_flux2_runner *vulkan_flux2_init(int device_id, int verbose, const char *shader_dir);

int  vulkan_flux2_load_dit(vulkan_flux2_runner *r, const char *safetensors_path);
int  vulkan_flux2_load_vae(vulkan_flux2_runner *r, const char *safetensors_path);
void vulkan_flux2_free(vulkan_flux2_runner *r);

/* Run single DiT denoising step on GPU.
 *   img_tokens: [n_img, patch_in_ch] patchified latent (CPU, F32)
 *   txt_tokens: [n_txt, txt_dim]    text hidden states (CPU, F32)
 *   timestep:   sigma * 1000
 *   guidance:   guidance scale (unused for distilled, pass 0)
 *   out:        [n_img, patch_in_ch] velocity (CPU, F32, pre-allocated)
 * Returns 0 on success. */
int vulkan_flux2_dit_step(vulkan_flux2_runner *r,
                          const float *img_tokens, int n_img,
                          const float *txt_tokens, int n_txt,
                          float timestep, float guidance, float *out);

/* VAE decode (CPU fallback). */
int vulkan_flux2_vae_decode(vulkan_flux2_runner *r,
                            const float *latent, int lat_h, int lat_w,
                            float *out_rgb);

#ifdef __cplusplus
}
#endif

#endif /* VULKAN_FLUX2_RUNNER_H */
