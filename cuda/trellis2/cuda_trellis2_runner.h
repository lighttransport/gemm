/*
 * cuda_trellis2_runner.h - CUDA TRELLIS.2 Stage 1 runner (image -> 3D structure)
 *
 * Uses NVRTC to compile CUDA C kernels at runtime (no nvcc needed).
 * F32 weights on GPU (BF16->F32 at load time), F32 compute.
 *
 * Pipeline:
 *   1. DINOv3 ViT-L/16 image encoder -> [1029, 1024] features
 *   2. DiT flow matching (30 blocks, 12 Euler steps, CFG) -> [8, 16, 16, 16] latent
 *   3. Structure decoder (Conv3D + ResBlocks) -> [64, 64, 64] occupancy
 *   4. Marching cubes -> OBJ mesh
 *
 * Usage:
 *   cuda_trellis2_runner *r = cuda_trellis2_init(0, 1);
 *   cuda_trellis2_load_weights(r, "dinov3.st", "stage1.st", "decoder.st");
 *   float *occupancy = cuda_trellis2_predict(r, rgb, w, h, 12, 7.5f, 42);
 *   // occupancy is [64*64*64] on CPU — run marching cubes, write OBJ
 *   free(occupancy);
 *   cuda_trellis2_free(r);
 */
#ifndef CUDA_TRELLIS2_RUNNER_H
#define CUDA_TRELLIS2_RUNNER_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_trellis2_runner cuda_trellis2_runner;

/* Initialize CUDA context and compile kernels */
cuda_trellis2_runner *cuda_trellis2_init(int device_id, int verbose);

/* Load weights from safetensors files */
int cuda_trellis2_load_weights(cuda_trellis2_runner *r,
                                const char *dinov3_path,
                                const char *stage1_path,
                                const char *decoder_path);

/* Run full pipeline: RGB image -> occupancy grid [64^3].
 * Returns CPU-side float[64*64*64] (caller must free). */
float *cuda_trellis2_predict(cuda_trellis2_runner *r,
                              const uint8_t *rgb, int w, int h,
                              int n_steps, float cfg_scale,
                              uint32_t seed);

/* Set GEMM mode: 0=F16 weights (default), 1=F32 weights */
void cuda_trellis2_set_f32_gemm(cuda_trellis2_runner *r, int enable);

void cuda_trellis2_free(cuda_trellis2_runner *r);

/* ---- Per-stage API (for testing/debugging) ---- */

/* Run DINOv3 encoder. image_f32: [3, 512, 512] CHW normalized. output: [1029, 1024] */
int cuda_trellis2_run_dinov3(cuda_trellis2_runner *r,
                              const float *image_f32, float *output);

/* Run DiT single forward step. */
int cuda_trellis2_run_dit(cuda_trellis2_runner *r,
                           const float *x_t, float timestep,
                           const float *cond_features, float *output);

/* Run structure decoder. latent: [8,16,16,16], output: [64^3] */
int cuda_trellis2_run_decoder(cuda_trellis2_runner *r,
                               const float *latent, float *output);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_TRELLIS2_RUNNER_H */
