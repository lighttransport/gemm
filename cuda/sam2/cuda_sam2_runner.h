#ifndef CUDA_SAM2_RUNNER_H_
#define CUDA_SAM2_RUNNER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_sam2_ctx cuda_sam2_ctx;

typedef struct {
    const char *ckpt_path;
    int image_size;
    int device_ordinal;
    int verbose;
} cuda_sam2_config;

cuda_sam2_ctx *cuda_sam2_create(const cuda_sam2_config *cfg);
void cuda_sam2_destroy(cuda_sam2_ctx *ctx);

int cuda_sam2_set_image(cuda_sam2_ctx *ctx, const uint8_t *rgb, int h, int w);
int cuda_sam2_set_points(cuda_sam2_ctx *ctx, const float *xy, const int32_t *labels, int n_points);
int cuda_sam2_set_box(cuda_sam2_ctx *ctx, float x0, float y0, float x1, float y1);
int cuda_sam2_run(cuda_sam2_ctx *ctx);
const float *cuda_sam2_get_scores(const cuda_sam2_ctx *ctx, int *out_n);
const uint8_t *cuda_sam2_get_masks(const cuda_sam2_ctx *ctx, int *out_n, int *out_h, int *out_w);

#ifdef __cplusplus
}
#endif

#endif
