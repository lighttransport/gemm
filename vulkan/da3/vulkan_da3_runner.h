/*
 * vulkan_da3_runner.h - Vulkan DA3 depth estimation runner
 *
 * Uses pre-compiled SPIR-V shaders. F32 weights on GPU (HOST_VISIBLE SSBOs).
 * Port of hip_da3_runner.h for cross-platform Vulkan compute.
 *
 * Supports DA3-Small depth + confidence output.
 *
 * Usage:
 *   vulkan_da3_runner *r = vulkan_da3_init(0, 1);
 *   vulkan_da3_load_safetensors(r, "model.safetensors", "config.json");
 *   da3_vk_result res = vulkan_da3_predict(r, rgb, w, h);
 *   da3_vk_result_free(&res);
 *   vulkan_da3_free(r);
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef VULKAN_DA3_RUNNER_H
#define VULKAN_DA3_RUNNER_H

#include <stdint.h>
#include <stdlib.h>

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

/* Result (depth + confidence) */
typedef struct {
    float *depth;
    float *confidence;
    int width, height;
} da3_vk_result;

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

typedef struct vulkan_da3_runner vulkan_da3_runner;

/* Initialize Vulkan context and load shaders.
 * shader_dir: path to directory containing compiled .spv files (e.g. "build/shaders") */
vulkan_da3_runner *vulkan_da3_init(int device_id, int verbose, const char *shader_dir);

/* Load weights from safetensors file */
int vulkan_da3_load_safetensors(vulkan_da3_runner *r, const char *st_path, const char *config_path);

/* Run depth estimation (depth + confidence only) */
da3_vk_result vulkan_da3_predict(vulkan_da3_runner *r, const uint8_t *rgb, int w, int h);

/* Run full depth estimation with output flags.
 * pose_in: optional camera pose[9] for CameraEnc conditioning (NULL to skip) */
da3_full_result vulkan_da3_predict_full(vulkan_da3_runner *r, const uint8_t *rgb,
                                          int w, int h, int output_flags,
                                          const float *pose_in);

/* Free runner */
void vulkan_da3_free(vulkan_da3_runner *r);

static inline void da3_vk_result_free(da3_vk_result *r) {
    if (r) { free(r->depth); free(r->confidence); r->depth = r->confidence = NULL; }
}

#ifndef DEPTH_ANYTHING3_H
static inline void da3_full_result_free(da3_full_result *r) {
    if (r) {
        free(r->depth); free(r->confidence); free(r->rays);
        free(r->ray_confidence); free(r->sky_seg); free(r->gaussians);
        free(r->metric_depth);
        memset(r, 0, sizeof(*r));
    }
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* VULKAN_DA3_RUNNER_H */
