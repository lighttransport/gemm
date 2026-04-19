/*
 * vulkan_qimg_runner.h - Vulkan Qwen-Image text-to-image runner
 *
 * Uses pre-compiled SPIR-V shaders and DEVICE_LOCAL SSBOs.
 * F32 weights on GPU, F32 compute.
 *
 * Port of cuda_qimg_runner.h for cross-platform Vulkan compute.
 *
 * Processes the 60-block MMDiT by loading one block at a time
 * (dequant FP8→F32 on CPU, upload, compute, free) to fit within 16GB VRAM.
 *
 * Pipeline:
 *   1. Text encoder (CPU) → hidden states [N_txt, 3584]
 *   2. MMDiT denoising (GPU) × N_steps → latent [16, H/8, W/8]
 *   3. VAE decoder (GPU) → RGB image [3, H, W]
 *
 * Usage:
 *   vulkan_qimg_runner *r = vulkan_qimg_init(0, 1, "build/shaders");
 *   vulkan_qimg_load_dit(r, "qwen_image_fp8_e4m3fn.safetensors");
 *   vulkan_qimg_load_vae(r, "qwen_image_vae.safetensors");
 *   vulkan_qimg_dit_step(r, img_tok, n_img, txt_tok, n_txt, t, out);
 *   vulkan_qimg_free(r);
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef VULKAN_QIMG_RUNNER_H
#define VULKAN_QIMG_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vulkan_qimg_runner vulkan_qimg_runner;

/* Initialize Vulkan context and load shaders.
 * shader_dir: path to directory containing compiled .spv files */
vulkan_qimg_runner *vulkan_qimg_init(int device_id, int verbose, const char *shader_dir);

int  vulkan_qimg_load_dit(vulkan_qimg_runner *r, const char *safetensors_path);
int  vulkan_qimg_load_vae(vulkan_qimg_runner *r, const char *safetensors_path);
void vulkan_qimg_free(vulkan_qimg_runner *r);

/* Run single DiT denoising step on GPU.
 *   img_tokens: [n_img, 64] patchified latent (CPU, F32)
 *   txt_tokens: [n_txt, 3584] text hidden states (CPU, F32)
 *   out:        [n_img, 64] velocity (CPU, F32, pre-allocated)
 * Returns 0 on success. */
int vulkan_qimg_dit_step(vulkan_qimg_runner *r,
                         const float *img_tokens, int n_img,
                         const float *txt_tokens, int n_txt,
                         float timestep, float *out);

/* VAE decode on GPU.
 *   latent:  [16, lat_h, lat_w] F32 (CPU)
 *   out_rgb: [3, lat_h*8, lat_w*8] F32 (CPU, pre-allocated)
 * Returns 0 on success. */
int vulkan_qimg_vae_decode(vulkan_qimg_runner *r,
                           const float *latent, int lat_h, int lat_w,
                           float *out_rgb);

#ifdef __cplusplus
}
#endif

#endif /* VULKAN_QIMG_RUNNER_H */
