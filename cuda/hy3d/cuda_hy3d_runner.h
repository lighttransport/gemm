/*
 * cuda_hy3d_runner.h - CUDA Hunyuan3D-2.1 runner (text/image → 3D mesh)
 *
 * Uses NVRTC to compile CUDA C kernels at runtime (no nvcc needed).
 * F16 weights on GPU, F32 compute. Works on sm_70+ (Volta through Blackwell).
 *
 * Pipeline:
 *   1. DINOv2-L image encoder → [1370, 1024] patch tokens
 *   2. DiT diffusion transformer (21 blocks, flow matching) → [4096, 64] latents
 *   3. ShapeVAE decoder (16 transformer blocks) → SDF grid
 *   4. Marching cubes → 3D mesh (OBJ/PLY)
 *
 * Usage:
 *   cuda_hy3d_runner *r = cuda_hy3d_init(0, 1);
 *   cuda_hy3d_load_weights(r, "conditioner.safetensors",
 *                              "model.safetensors",
 *                              "vae.safetensors");
 *   hy3d_mesh m = cuda_hy3d_predict(r, rgb, w, h, 30, 7.5f, 256);
 *   mc_write_obj("output.obj", ...);
 *   cuda_hy3d_mesh_free(&m);
 *   cuda_hy3d_free(r);
 */
#ifndef CUDA_HY3D_RUNNER_H
#define CUDA_HY3D_RUNNER_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *vertices;    /* [n_verts, 3] */
    int   *triangles;   /* [n_tris, 3] */
    int    n_verts;
    int    n_tris;
} hy3d_mesh;

typedef struct cuda_hy3d_runner cuda_hy3d_runner;

/* Initialize CUDA context and compile kernels */
cuda_hy3d_runner *cuda_hy3d_init(int device_id, int verbose);

/* Load weights from safetensors files:
 *   conditioner_path: DINOv2 image encoder weights
 *   model_path:       DiT diffusion transformer weights
 *   vae_path:         ShapeVAE decoder weights */
int cuda_hy3d_load_weights(cuda_hy3d_runner *r,
                           const char *conditioner_path,
                           const char *model_path,
                           const char *vae_path);

/* Run full pipeline: RGB image → 3D mesh
 *   rgb:            [h, w, 3] uint8 RGB image
 *   w, h:           image dimensions
 *   n_steps:        diffusion sampling steps (20-50, default 30)
 *   guidance_scale: CFG guidance scale (default 7.5)
 *   grid_res:       marching cubes grid resolution (default 256)
 *   seed:           random seed for noise (0 = random) */
hy3d_mesh cuda_hy3d_predict(cuda_hy3d_runner *r,
                            const uint8_t *rgb, int w, int h,
                            int n_steps, float guidance_scale,
                            int grid_res, uint32_t seed);

/* Set GEMM precision mode. Call BEFORE cuda_hy3d_load_weights().
 *   0 = F16 weights, F32 compute (default, memory efficient)
 *   1 = F32 weights, F32 compute (matches PyTorch reference exactly) */
void cuda_hy3d_set_f32_gemm(cuda_hy3d_runner *r, int enable);

/* Free runner and all GPU resources */
void cuda_hy3d_free(cuda_hy3d_runner *r);

/* ---- Per-stage verification API ---- */

/* Run DINOv2 encoder only.
 *   image_f32: [3, 518, 518] F32 pre-processed image (ImageNet-normalized, CHW)
 *   output:    [1370, 1024] F32 buffer (must be pre-allocated)
 *   Returns 0 on success. */
int cuda_hy3d_run_dinov2(cuda_hy3d_runner *r,
                          const float *image_f32,
                          float *output);

/* Run ShapeVAE decoder + SDF query.
 *   latents:   [4096, 64] F32 input latents
 *   grid_res:  marching cubes grid resolution (e.g. 8, 32, 256)
 *   sdf_out:   [grid_res^3] F32 buffer (must be pre-allocated)
 *   Returns 0 on success. */
int cuda_hy3d_run_vae(cuda_hy3d_runner *r,
                       const float *latents,
                       int grid_res,
                       float *sdf_out);

/* Run DiT single forward pass.
 *   latents:   [4096, 64] F32 noisy latents
 *   timestep:  scalar timestep (e.g. 0.5)
 *   context:   [1370, 1024] F32 conditioning from DINOv2
 *   output:    [4096, 64] F32 buffer (must be pre-allocated)
 *   Returns 0 on success. */
int cuda_hy3d_run_dit(cuda_hy3d_runner *r,
                       const float *latents,
                       float timestep,
                       const float *context,
                       float *output);

/* Free mesh data */
static inline void cuda_hy3d_mesh_free(hy3d_mesh *m) {
    if (m) {
        free(m->vertices);  m->vertices = NULL;
        free(m->triangles); m->triangles = NULL;
        m->n_verts = m->n_tris = 0;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_HY3D_RUNNER_H */
