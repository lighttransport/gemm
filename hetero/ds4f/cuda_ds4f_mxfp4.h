#ifndef CUDA_DS4F_MXFP4_H
#define CUDA_DS4F_MXFP4_H
#include <stddef.h>
#include <stdint.h>
typedef struct cuda_ds4f_mxfp4 cuda_ds4f_mxfp4;
cuda_ds4f_mxfp4 *cuda_ds4f_mxfp4_create(int device_id, int verbose);
void cuda_ds4f_mxfp4_destroy(cuda_ds4f_mxfp4 *ctx);
int cuda_ds4f_mxfp4_load(cuda_ds4f_mxfp4 *ctx, const uint8_t *w,
                         const uint8_t *scale, int rows, int cols);
int cuda_ds4f_mxfp4_gemm(cuda_ds4f_mxfp4 *ctx, float *dst, const float *x,
                         int tokens, int rows, int cols);
#endif
