/* Native Pixal3D single-view inference. All tuning is explicit configuration.
 * SPDX-License-Identifier: MIT */
#ifndef PIXAL3D_H
#define PIXAL3D_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum { PIXAL3D_CPU, PIXAL3D_CUDA, PIXAL3D_ROCM } pixal3d_backend;
typedef struct pixal3d_context pixal3d_context;
/* Separate configuration keeps the original options/result ABI unchanged. */
typedef enum { PIXAL3D_GPU_LEGACY, PIXAL3D_GPU_RESIDENT } pixal3d_gpu_execution;
typedef enum { PIXAL3D_KERNEL_AUTO, PIXAL3D_KERNEL_BLAS, PIXAL3D_KERNEL_MMA } pixal3d_gpu_kernels;
typedef struct {
    size_t struct_size;
    uint32_t version;
    pixal3d_gpu_execution execution;
    pixal3d_gpu_kernels kernels;
    const char *profile_json;
} pixal3d_gpu_options;
void pixal3d_default_gpu_options(pixal3d_gpu_options *options);
int pixal3d_configure_gpu(pixal3d_context *context, const pixal3d_gpu_options *options);
typedef struct {
    pixal3d_backend backend;
    int device, threads;
    size_t vram_budget_mib;
    const char *model_dir, *dinov3_path, *naf_path, *dump_dir;
    uint32_t seed;
    int texture_size, decimation_target;
} pixal3d_options;
typedef struct {
    float fov;        /* Horizontal FOV in radians, for the cropped image. */
    float distance;   /* <=0: derive from FOV with upstream framing. */
    float mesh_scale; /* Must be positive; normally 1. */
} pixal3d_camera;
typedef struct {
    const uint8_t *pixels; /* HWC RGB or RGBA, contiguous. */
    const uint8_t *mask;   /* HW, required for RGB; overrides RGBA alpha. */
    int width, height, channels;
} pixal3d_image;
typedef struct {
    double elapsed_seconds;
    size_t peak_device_bytes, peak_host_bytes;
    int shape_tokens, vertices, triangles;
} pixal3d_stats;
typedef struct {
    float *vertices, *normals, *uvs;
    uint32_t *triangles;
    uint8_t *base_color_rgba, *metallic_roughness_rgb;
    int vertex_count, triangle_count, texture_size;
    pixal3d_stats stats;
} pixal3d_result;

void pixal3d_default_options(pixal3d_options *options);
pixal3d_context *pixal3d_create(const pixal3d_options *options);
const char *pixal3d_last_error(const pixal3d_context *context);
int pixal3d_generate(pixal3d_context *context, const pixal3d_image *image, const pixal3d_camera *camera,
                     pixal3d_result *result);
/* result must be zero-initialized; free a previous result before reusing it.
 * write_glb returns -1 on error and prints the export diagnostic to stderr. */
int pixal3d_write_glb(const char *path, const pixal3d_result *result);
void pixal3d_result_free(pixal3d_result *result);
void pixal3d_destroy(pixal3d_context *context);

/* Testable math interface. Coordinates always use upstream (b,x,y,z),
 * regardless of historical z,y,x labels in TRELLIS.2 headers. */
int pixal3d_camera_distance(float fov, float mesh_scale, float *distance);
int pixal3d_project(const int32_t *coords, size_t count, int grid_resolution, int image_resolution,
                    const pixal3d_camera *camera, float *xy_normalized);
int pixal3d_sample_features(const float *hwc, int height, int width, int channels, const float *xy_normalized,
                            size_t count, float *out);
/* Returns unique coordinate count, or -1. output needs count*4 integers. */
int64_t pixal3d_cascade_coords(const int32_t *coords, size_t count, int low_resolution, int high_resolution,
                               int32_t *output);
void pixal3d_euler_cfg(float *x, const float *positive, const float *negative, size_t count, float t,
                       float t_next, float guidance, float rescale, float sigma_min);
#ifdef __cplusplus
}
#endif
#endif
