/*
 * cuda_ppd_runner.h - CUDA Pixel-Perfect-Depth (PPD) runner
 *
 * Uses NVRTC to compile CUDA C kernels at runtime (no nvcc needed).
 * F16 weights on GPU, F32 compute. Works on sm_70+ (Volta through Blackwell).
 *
 * PPD pipeline:
 *   1. Semantic encoder (DA2 DINOv2 ViT-L) → [T, 1024] patch tokens
 *   2. DiT diffusion transformer (24 blocks, 4 Euler steps) → depth [H, W]
 *
 * Usage:
 *   cuda_ppd_runner *r = cuda_ppd_init(0, 1);
 *   cuda_ppd_load_weights(r, "ppd.pth", "depth_anything_v2_vitl.pth");
 *   ppd_result res = cuda_ppd_predict(r, rgb, w, h);
 *   cuda_ppd_free(r);
 */
#ifndef CUDA_PPD_RUNNER_H
#define CUDA_PPD_RUNNER_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *depth;        /* [H, W] output depth map */
    int width, height;
} ppd_result;

typedef struct cuda_ppd_runner cuda_ppd_runner;

cuda_ppd_runner *cuda_ppd_init(int device_id, int verbose);
int cuda_ppd_load_weights(cuda_ppd_runner *r,
                           const char *ppd_pth_path,
                           const char *sem_pth_path);
ppd_result cuda_ppd_predict(cuda_ppd_runner *r, const uint8_t *rgb, int w, int h);
void cuda_ppd_free(cuda_ppd_runner *r);

static inline void ppd_result_free(ppd_result *r) {
    if (r) { free(r->depth); r->depth = NULL; }
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PPD_RUNNER_H */
