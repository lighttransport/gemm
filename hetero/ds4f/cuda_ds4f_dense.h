#ifndef CUDA_DS4F_DENSE_H
#define CUDA_DS4F_DENSE_H

#include "../../common/ds4f.h"

typedef struct cuda_ds4f_dense cuda_ds4f_dense;

cuda_ds4f_dense *cuda_ds4f_dense_create(int device_id, int verbose);
void cuda_ds4f_dense_destroy(cuda_ds4f_dense *ctx);
int cuda_ds4f_dense_bind_tensor(cuda_ds4f_dense *ctx, ds4f_tensor *t);
int cuda_ds4f_dense_gemm_tensor(void *ctx, float *dst,
                                const ds4f_tensor *t, const float *x,
                                int M, int Ystride, int Xstride);
int cuda_ds4f_dense_shared_ffn(void *ctx, float *dst,
                               const ds4f_tensor *w1,
                               const ds4f_tensor *w3,
                               const ds4f_tensor *w2,
                               const float *x, int M, int inter, int C,
                               float lim);

#endif
