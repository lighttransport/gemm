#ifndef GENIA_CUDA_H
#define GENIA_CUDA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct genia_cuda genia_cuda;
/* Primitive library, not yet an end-to-end reconstruction runner.
 * Host inputs/outputs; callers serialize calls on each context.
 * Budget includes all library device buffers, excludes driver/module memory. */
genia_cuda *genia_cuda_open(int device, size_t budget_bytes);
void genia_cuda_close(genia_cuda *ctx);
const char *genia_cuda_error(void);
size_t genia_cuda_peak_bytes(const genia_cuda *ctx);
int genia_cuda_visibility(genia_cuda *ctx, const int32_t *xyz, int count,
    const float camera[3], const uint8_t *occupancy, int grid,
    float neighbor_tolerance, uint8_t *visible);
/* scores[views,count]; velocities[views,count,channels]; output[count,channels]. */
int genia_cuda_fuse(genia_cuda *ctx, const float *scores, const float *velocities,
    int views, int count, int channels, float temperature, float floor_weight, float *output);
int genia_cuda_tweedie(genia_cuda *ctx, const float *latent, const float *velocity,
    int count, float time, float *clean);
int genia_cuda_guidance(genia_cuda *ctx, const float *velocity, const float *gradient,
    int count, float weight, int normalize, float *guided);
/* Free Gaussian projection/compositing and vector-Jacobian products.
 * Gradients are externally supplied image-loss derivatives, not a loss preset.
 * rgba and depth may be NULL; row_gradient [count,17] and camera_gradient [12]
 * may be NULL. unused PLY normal gradients are zero. All arrays are host-side. */
int genia_cuda_render(genia_cuda *ctx, const float *ply_rows, int count,
    const float world_to_camera[12], const float intrinsics[9], int width, int height,
    const float background[3], const float *rgba_gradient, const float *depth_gradient,
    float *rgba, float *depth, float *row_gradient, float *camera_gradient);
#ifdef __cplusplus
}
#endif
#endif
