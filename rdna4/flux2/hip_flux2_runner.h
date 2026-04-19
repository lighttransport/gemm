/*
 * hip_flux2_runner.h - HIP/ROCm Flux.2 Klein text-to-image runner (RDNA4)
 *
 * Uses HIPRTC to compile HIP C kernels at runtime (no hipcc needed).
 * F32 weights on GPU, F32 compute. Targets RDNA4 (gfx1200/gfx1201).
 *
 * Port of cuda_flux2_runner.h for AMD ROCm/HIP.
 *
 * Runs the 5 double-stream + 20 single-stream DiT blocks on GPU.
 * VAE decode falls back to CPU.
 *
 * Usage:
 *   hip_flux2_runner *r = hip_flux2_init(0, 1);
 *   hip_flux2_load_dit(r, "flux-2-klein-4b.safetensors");
 *   hip_flux2_load_vae(r, "flux2-vae.safetensors");
 *   hip_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt, t, guidance, out);
 *   hip_flux2_free(r);
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef HIP_FLUX2_RUNNER_H
#define HIP_FLUX2_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_flux2_runner hip_flux2_runner;

hip_flux2_runner *hip_flux2_init(int device_id, int verbose);
int  hip_flux2_load_dit(hip_flux2_runner *r, const char *safetensors_path);
int  hip_flux2_load_vae(hip_flux2_runner *r, const char *safetensors_path);
void hip_flux2_free(hip_flux2_runner *r);

/* Run single DiT denoising step on GPU.
 *   img_tokens: [n_img, patch_in_ch] patchified latent (CPU, F32)
 *   txt_tokens: [n_txt, txt_dim]    text hidden states (CPU, F32)
 *   timestep:   sigma * 1000
 *   guidance:   guidance scale (unused for distilled, pass 0)
 *   out:        [n_img, patch_in_ch] velocity (CPU, F32, pre-allocated)
 * Returns 0 on success. */
int hip_flux2_dit_step(hip_flux2_runner *r,
                       const float *img_tokens, int n_img,
                       const float *txt_tokens, int n_txt,
                       float timestep, float guidance, float *out);

/* VAE decode (CPU fallback).
 *   latent:   [32, lat_h, lat_w] F32 (CPU)
 *   out_rgb:  [3, lat_h*8, lat_w*8] F32 (CPU, pre-allocated)
 * Returns 0 on success. */
int hip_flux2_vae_decode(hip_flux2_runner *r,
                         const float *latent, int lat_h, int lat_w,
                         float *out_rgb);

#ifdef __cplusplus
}
#endif

#endif /* HIP_FLUX2_RUNNER_H */
