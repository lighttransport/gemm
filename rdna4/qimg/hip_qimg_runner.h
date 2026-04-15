/*
 * hip_qimg_runner.h - HIP/ROCm Qwen-Image text-to-image runner (RDNA4)
 *
 * Uses HIPRTC to compile HIP C kernels at runtime (no hipcc needed).
 * F32 weights on GPU, F32 compute. Targets RDNA4 (gfx1200/gfx1201).
 *
 * Port of cuda_qimg_runner.h for AMD ROCm/HIP.
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
 *   hip_qimg_runner *r = hip_qimg_init(0, 1);
 *   hip_qimg_load_dit(r, "qwen_image_fp8_e4m3fn.safetensors");
 *   hip_qimg_load_vae(r, "qwen_image_vae.safetensors");
 *   hip_qimg_dit_step(r, img_tok, n_img, txt_tok, n_txt, t, out);
 *   hip_qimg_free(r);
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef HIP_QIMG_RUNNER_H
#define HIP_QIMG_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_qimg_runner hip_qimg_runner;

hip_qimg_runner *hip_qimg_init(int device_id, int verbose);
int  hip_qimg_load_dit(hip_qimg_runner *r, const char *safetensors_path);
int  hip_qimg_load_vae(hip_qimg_runner *r, const char *safetensors_path);
void hip_qimg_free(hip_qimg_runner *r);

/* Run single DiT denoising step on GPU.
 *   img_tokens: [n_img, 64] patchified latent (CPU)
 *   txt_tokens: [n_txt, 3584] text hidden states (CPU)
 *   out: [n_img, 64] velocity prediction (CPU, pre-allocated)
 *   Returns 0 on success. */
int hip_qimg_dit_step(hip_qimg_runner *r,
                      const float *img_tokens, int n_img,
                      const float *txt_tokens, int n_txt,
                      float timestep, float *out);

/* VAE decode on GPU.
 *   latent: [16, lat_h, lat_w] F32 (CPU)
 *   out_rgb: [3, lat_h*8, lat_w*8] F32 (CPU, pre-allocated)
 *   Returns 0 on success. */
int hip_qimg_vae_decode(hip_qimg_runner *r,
                        const float *latent, int lat_h, int lat_w,
                        float *out_rgb);

#ifdef __cplusplus
}
#endif

#endif /* HIP_QIMG_RUNNER_H */
