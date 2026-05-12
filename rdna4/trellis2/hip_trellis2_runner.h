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

/* Free SS DiT weights/scratch/KV. Reclaims ~5 GB on a 1.3B-param F32 DiT.
 * Safe to call after SS sampling completes. Subsequent dit_step calls will
 * fail; reload with hip_trellis2_load_dit. */
void hip_trellis2_unload_dit(hip_trellis2_runner *r);

/* Free SS decoder weights + 4× scratch volumes. Safe to call after decode. */
void hip_trellis2_unload_decoder(hip_trellis2_runner *r);

/* Free shape SLAT DiT weights/scratch/KV. Reclaims ~3 GB. Safe to call after
 * shape SLAT sampling produces the dense feats fed to shape_dec. */
void hip_trellis2_unload_slat_dit(hip_trellis2_runner *r);

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

/* ---- Stage-2 shape decoder (sparse FlexiDualGrid VAE decoder) -------------
 *
 * Takes a structured latent (post-DiT, denormalized) and produces a triangle
 * mesh. Internally runs the 4-stage hierarchical sparse decoder
 * (SparseConvNeXt + Channel-to-Spatial) followed by FDG mesh extraction.
 *
 * The shape decoder owns a separate HIP module, hipBLASLt + Triton bridges,
 * and persistent scratch — independent of the SS DiT/decoder above. It is
 * legal to load it without loading the SS DiT (and vice versa).
 */

/* Load shape_dec safetensors weights (host-side). Returns 0 on success. */
int hip_trellis2_load_shape_dec(hip_trellis2_runner *r,
                                const char *safetensors_path);

typedef struct {
    float *vertices;   /* heap-allocated [n_verts*3]; caller frees with
                        * hip_trellis2_shape_dec_mesh_free */
    int   *triangles;  /* heap-allocated [n_tris*3] */
    int    n_verts;
    int    n_tris;
} hip_trellis2_mesh;

/* Run shape_dec end-to-end on a sparse structured latent.
 *   slat_feats: [N, slat_C] F32 (CPU)
 *   coords:     [N, 4]      I32 (CPU) — (b, z, y, x)
 *   N:          number of voxels
 *   slat_C:     latent channel count (matches checkpoint's from_latent input)
 *   out:        pre-zeroed mesh struct; populated on success.
 * Returns 0 on success. */
int hip_trellis2_run_shape_dec(hip_trellis2_runner *r,
                               const float *slat_feats,
                               const int32_t *coords,
                               int N,
                               int slat_C,
                               hip_trellis2_mesh *out);

void hip_trellis2_shape_dec_mesh_free(hip_trellis2_mesh *m);

/* ---- Stage-3 Tex decoder (sparse 4-stage hierarchical PBR decoder) -------
 *
 * Same architecture as the shape decoder but trained to output per-voxel
 * PBR feats (out_channels read from checkpoint, typically 6). Independent
 * runner slot — can coexist with the shape decoder. The caller is
 * responsible for downstream UV / texture baking; this entry point just
 * returns the dense final-stage feats + their (b,z,y,x) coords on the host.
 */

/* Load tex_dec safetensors weights. Returns 0 on success. */
int hip_trellis2_load_tex_dec(hip_trellis2_runner *r,
                              const char *safetensors_path);

/* Free tex_dec weights + scratch. Safe to call after texture baking. */
void hip_trellis2_unload_tex_dec(hip_trellis2_runner *r);

/* Run tex_dec end-to-end. Returns 0 on success.
 *   slat_feats   : [N, slat_C] F32 (CPU; usually the denormalized tex SLat)
 *   coords       : [N, 4]      I32 (CPU; b,z,y,x at the SLat resolution)
 *   out_feats    : populated to a heap-allocated [out_N, out_channels] F32
 *                  (caller frees with free()). Channels match checkpoint
 *                  output_layer.weight.
 *   out_coords   : populated to a heap-allocated [out_N, 4] I32 (b,z,y,x at
 *                  the highest-resolution stage; caller frees with free()).
 *   out_N        : number of voxels in the highest-resolution stage.
 */
int hip_trellis2_run_tex_dec(hip_trellis2_runner *r,
                             const float *slat_feats,
                             const int32_t *coords,
                             int N, int slat_C,
                             float **out_feats,
                             int32_t **out_coords,
                             int *out_N);

/* ---- Stage-2 SLAT (Sparse-Latent) DiT ------------------------------------
 *
 * Sister of the SS DiT above. Same 30-block ModulatedTransformerCrossBlock
 * architecture (dim=1536, heads=12, head_dim=128, ffn=8192, cond_dim=1024),
 * but tokens are sparse (variable N per call indexed by [N,4] coords) and
 * in_channels=32. RoPE is computed per-call from coords (axis-major,
 * 21 freqs/axis, trailing identity to fill head_dim/2 = 64 pairs).
 *
 * Independent weight set + scratch from the SS DiT — both can coexist on
 * the same runner. F32 weights on GPU for the v1 wiring (BF16/FP8 follow-up).
 */

/* Load SLAT DiT safetensors weights to GPU (F32). Returns 0 on success. */
int  hip_trellis2_load_slat_dit(hip_trellis2_runner *r,
                                const char *safetensors_path);

/* Single SLAT denoising step.
 *   x_t      [N, 32]        F32 host
 *   coords   [N, 4]         I32 host (b, z, y, x); z/y/x in [0, 32)
 *   N        number of sparse tokens (1..8192 supported)
 *   t_value  raw timestep (caller multiplies by 1000 if mimicking sampler;
 *            this fn does NOT scale t internally — matches _inference_model)
 *   cond     [n_cond, 1024] F32 host (drop the leading batch=1 dim)
 *   n_cond   length of conditioning sequence (typically 1029)
 *   out_vel  [N, 32]        F32 host (pre-allocated)
 *
 * Internally rebuilds RoPE tables from coords every call; rebuilds the
 * cross-attn KV cache when (cond pointer, n_cond) differs from the previous
 * call or after hip_trellis2_invalidate_slat_kv.
 */
int  hip_trellis2_slat_dit_step(hip_trellis2_runner *r,
                                const float *x_t,
                                const int32_t *coords, int N,
                                float t_value,
                                const float *cond, int n_cond,
                                float *out_vel);

/* Force a rebuild of the SLAT cross-attn KV cache on the next step. */
void hip_trellis2_invalidate_slat_kv(hip_trellis2_runner *r);

/* ---- Stage-3 Tex DiT (texture flow) -------------------------------------
 *
 * Sister of the shape SLAT DiT. Same 30-block ModulatedTransformerCrossBlock
 * but in_channels=64 (noise_32 + shape_norm_32 concatenated per voxel) and
 * out_channels=32 (texture-latent velocity). Independent weights, KV cache,
 * and scratch — runs after shape SLAT during e2e texgen.
 *
 * x_t [N,64], coords [N,4], same RoPE-3D / sampler conventions as SLAT
 * (callers pass raw t in [0,1]; runner multiplies by 1000 internally).
 */

/* Load tex DiT safetensors weights to GPU. Returns 0 on success. */
int  hip_trellis2_load_tex_dit(hip_trellis2_runner *r,
                               const char *safetensors_path);

/* Single tex denoising step. Inputs/outputs match slat_dit_step except
 * x_t is [N, 64] and out_vel is [N, 32]. */
int  hip_trellis2_tex_dit_step(hip_trellis2_runner *r,
                                const float *x_t,
                                const int32_t *coords, int N,
                                float t_value,
                                const float *cond, int n_cond,
                                float *out_vel);

/* Force a rebuild of the tex cross-attn KV cache on the next step. */
void hip_trellis2_invalidate_tex_kv(hip_trellis2_runner *r);

#ifdef __cplusplus
}
#endif

#endif /* HIP_TRELLIS2_RUNNER_H */
