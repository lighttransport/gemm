/*
 * hip_ppd_runner.h - HIP/ROCm Pixel-Perfect-Depth (PPD) runner
 *
 * Uses HIPRTC to compile HIP C kernels at runtime (no hipcc needed).
 * F16 weights on GPU, F32 compute. Targets RDNA4 (gfx1200/gfx1201).
 *
 * PPD pipeline:
 *   1. Semantic encoder (DA2 DINOv2 ViT-L) -> [T, 1024] patch tokens
 *   2. DiT diffusion transformer (24 blocks, 4 Euler steps) -> depth [H, W]
 *
 * Usage:
 *   hip_ppd_runner *r = hip_ppd_init(0, 1);
 *   hip_ppd_load_weights(r, "ppd.pth", "depth_anything_v2_vitl.pth");
 *   ppd_result res = hip_ppd_predict(r, rgb, w, h);
 *   hip_ppd_free(r);
 */
#ifndef HIP_PPD_RUNNER_H
#define HIP_PPD_RUNNER_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *depth;        /* [H, W] output depth map */
    int width, height;
} ppd_result;

typedef struct hip_ppd_runner hip_ppd_runner;

hip_ppd_runner *hip_ppd_init(int device_id, int verbose);
int hip_ppd_load_weights(hip_ppd_runner *r,
                          const char *ppd_pth_path,
                          const char *sem_pth_path);
ppd_result hip_ppd_predict(hip_ppd_runner *r, const uint8_t *rgb, int w, int h);
void hip_ppd_free(hip_ppd_runner *r);

static inline void ppd_result_free(ppd_result *r) {
    if (r) { free(r->depth); r->depth = NULL; }
}

#ifdef __cplusplus
}
#endif

#endif /* HIP_PPD_RUNNER_H */
