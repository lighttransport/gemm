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

#endif
