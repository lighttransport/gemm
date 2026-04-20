/*
 * hip_trellis2_runner.h - HIP/ROCm TRELLIS.2 Stage 1 runner (RDNA4)
 *
 * Uses HIPRTC to compile HIP C kernels at runtime (no hipcc needed).
 * F32 weights on GPU (BF16 converted on load), F32 compute.
 * Targets RDNA4 (gfx1200/gfx1201).
 *
 * Pipeline: DINOv3 (CPU) -> Stage 1 DiT 30-block (GPU) -> Decoder (GPU)
 *
 * Port of cuda_trellis2_runner for AMD ROCm/HIP.
 *
 * Usage:
 *   hip_trellis2_runner *r = hip_trellis2_init(0, 1);
 *   hip_trellis2_load_dit(r, "ss_flow_img_dit_1_3B_64_bf16.safetensors");
 *   hip_trellis2_load_decoder(r, "ss_dec_conv3d_16l8_fp16.safetensors");
 *   hip_trellis2_dit_step(r, noise_flat, features, t, out_vel);
 *   hip_trellis2_decode(r, latent, occupancy);
 *   hip_trellis2_free(r);
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef HIP_TRELLIS2_RUNNER_H
#define HIP_TRELLIS2_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_trellis2_runner hip_trellis2_runner;

/* Initialize HIP context. Returns NULL on failure. */
hip_trellis2_runner *hip_trellis2_init(int device_id, int verbose);

/* Load Stage 1 DiT weights (BF16 safetensors → GPU F32).
 * Returns 0 on success. Precomputes RoPE phases. */
int hip_trellis2_load_dit(hip_trellis2_runner *r, const char *safetensors_path);

/* Load Stage 1 decoder weights.
 * Returns 0 on success. */
int hip_trellis2_load_decoder(hip_trellis2_runner *r, const char *safetensors_path);

/* Free all resources. */
void hip_trellis2_free(hip_trellis2_runner *r);

/* Run a single DiT denoising step on GPU.
 *   noise_flat:  [4096, 8] F32 (CPU) — noise token sequence [NCDHW reshaped]
 *   features:    [1029, 1024] F32 (CPU) — DINOv3 conditioning features
 *   t:           timestep in range [0, 1000]
 *   out_vel:     [4096, 8] F32 (CPU, pre-allocated) — predicted velocity
 * Invalidates cross-attn KV cache if features changed.
 * Returns 0 on success.
 */
int hip_trellis2_dit_step(hip_trellis2_runner *r,
                          const float *noise_flat,
                          const float *features,
                          float t,
                          float *out_vel);

/* Invalidate cross-attention KV cache (call before switching features). */
void hip_trellis2_invalidate_kv(hip_trellis2_runner *r);

/* Decode latent to occupancy.
 *   latent:     [8, 16, 16, 16] F32 (CPU)
 *   occupancy:  [64, 64, 64] F32 (CPU, pre-allocated) — raw logits
 * Returns 0 on success.
 */
int hip_trellis2_decode(hip_trellis2_runner *r,
                        const float *latent,
                        float *occupancy);

/* Dump a single block's hidden state for verification.
 *   noise_flat:  [4096, 8] F32 (CPU)
 *   features:    [1029, 1024] F32 (CPU)
 *   t:           timestep
 *   block_idx:   which block to dump (0..29)
 *   out_hidden:  [4096, 1536] F32 (CPU, pre-allocated)
 * Returns 0 on success.
 */
int hip_trellis2_dump_block(hip_trellis2_runner *r,
                             const float *noise_flat,
                             const float *features,
                             float t,
                             int block_idx,
                             float *out_hidden);

/* Invalidate cross-attention KV cache (call when features change). */
void hip_trellis2_invalidate_kv_cache(hip_trellis2_runner *r);

/* Block 0 detailed intermediate dump. Any pointer may be NULL to skip.
 *   mod          : [6*1536]         shared+block-0 adaLN output
 *   ln_h_sa      : [4096, 1536]     after first adaLN, before QKV
 *   q_post/k_post: [4096, 1536]     Q/K after RMSNorm+RoPE
 *   v            : [4096, 1536]     V after split
 *   sa_proj      : [4096, 1536]     self-attn output projection (pre-gate)
 *   h_post_sa    : [4096, 1536]     hidden after gated residual
 *   ca_proj      : [4096, 1536]     cross-attn output projection
 *   h_post_ca    : [4096, 1536]     hidden after ca residual
 *   ln_h_mlp     : [4096, 1536]     after 2nd adaLN
 *   mlp_proj     : [4096, 1536]     MLP FC2 output (pre-gate)
 *   h_post_mlp   : [4096, 1536]     hidden after MLP (=block 0 output)
 */
typedef struct {
    float *input_embed; /* [4096, 1536] h after input_layer */
    float *mod;
    float *ln_h_sa;
    float *q_post;
    float *k_post;
    float *v;
    float *sa_proj;
    float *h_post_sa;
    float *ca_proj;
    float *h_post_ca;
    float *ln_h_mlp;
    float *mlp_proj;
    float *h_post_mlp;
} hip_trellis2_b0_dbg;

int hip_trellis2_dump_b0_detail(hip_trellis2_runner *r,
                                 const float *noise_flat,
                                 const float *features,
                                 float t,
                                 hip_trellis2_b0_dbg *dbg);

#ifdef __cplusplus
}
#endif

#endif /* HIP_TRELLIS2_RUNNER_H */
