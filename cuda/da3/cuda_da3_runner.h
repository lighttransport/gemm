/*
 * cuda_da3_runner.h - CUDA DA3 depth estimation runner
 *
 * Uses NVRTC to compile CUDA C kernels at runtime (no nvcc needed).
 * F16 weights on GPU, F32 compute. Works on sm_70+ (V100 through Blackwell).
 *
 * Supports full DA3NESTED-GIANT-LARGE-1.1 output modalities:
 *   - Depth + confidence (main DPT)
 *   - Pose estimation (CameraDec)
 *   - Rays + sky segmentation (aux DPT branch)
 *   - 3D Gaussians (GSDPT)
 *   - Metric depth (nested ViT-L)
 *
 * Usage:
 *   cuda_da3_runner *r = cuda_da3_init(0, 1);
 *   cuda_da3_load_safetensors(r, "model.safetensors", "config.json");
 *   da3_full_result res = cuda_da3_predict_full(r, rgb, w, h, DA3_OUTPUT_ALL, NULL);
 *   cuda_da3_free(r);
 */
#ifndef CUDA_DA3_RUNNER_H
#define CUDA_DA3_RUNNER_H

#include <stdint.h>
#include <string.h>
#include "../../common/gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Output modality flags */
#ifndef DA3_OUTPUT_DEPTH
#define DA3_OUTPUT_DEPTH      0x01
#define DA3_OUTPUT_POSE       0x02
#define DA3_OUTPUT_RAYS       0x04
#define DA3_OUTPUT_GAUSSIANS  0x08
#define DA3_OUTPUT_ALL        0x0F
#endif

/* Legacy result (depth + confidence only) */
typedef struct {
    float *depth;
    float *confidence;
    int width, height;
} da3_cuda_result;

/* Full result with all output modalities */
#ifndef DEPTH_ANYTHING3_H
typedef struct {
    float *depth, *confidence;
    float *rays;
    float *ray_confidence;
    float *sky_seg;
    float pose[9];
    float *gaussians;
    float *metric_depth;
    int width, height;
    int has_pose, has_rays, has_gaussians, has_metric;
} da3_full_result;
#endif

typedef struct cuda_da3_runner cuda_da3_runner;

cuda_da3_runner *cuda_da3_init(int device_id, int verbose);
int cuda_da3_load_weights(cuda_da3_runner *r, gguf_context *gguf);
int cuda_da3_load_safetensors(cuda_da3_runner *r, const char *st_path, const char *config_path);
da3_cuda_result cuda_da3_predict(cuda_da3_runner *r, const uint8_t *rgb, int w, int h);
da3_full_result cuda_da3_predict_full(cuda_da3_runner *r, const uint8_t *rgb,
                                       int w, int h, int output_flags,
                                       const float *pose_in);
void cuda_da3_free(cuda_da3_runner *r);

static inline void da3_cuda_result_free(da3_cuda_result *r) {
    if (r) { free(r->depth); free(r->confidence); r->depth = r->confidence = NULL; }
}

#ifndef DEPTH_ANYTHING3_H
static inline void da3_full_result_free(da3_full_result *r) {
    if (r) {
        free(r->depth); free(r->confidence);
        free(r->rays); free(r->ray_confidence); free(r->sky_seg);
        free(r->gaussians); free(r->metric_depth);
        memset(r, 0, sizeof(*r));
    }
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* CUDA_DA3_RUNNER_H */
