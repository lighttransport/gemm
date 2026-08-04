#ifndef DUAL_DS4F_PREFILL_H
#define DUAL_DS4F_PREFILL_H

#include "hip_ds4f_dense.h"
#include "cuda_ds4f_mxfp4.h"
#include "../../common/ds4f.h"

typedef struct dual_ds4f_prefill dual_ds4f_prefill;

dual_ds4f_prefill *dual_ds4f_prefill_create(int hip_device, int cuda_device,
                                             int verbose);
void dual_ds4f_prefill_destroy(dual_ds4f_prefill *ctx);
int dual_ds4f_prefill_bind_tensor(dual_ds4f_prefill *ctx, ds4f_tensor *t);
void dual_ds4f_prefill_attach_model(ds4f_model *model,
                                    dual_ds4f_prefill *ctx);

int dual_ds4f_prefill_gemm(void *ctx, float *dst, const ds4f_tensor *t,
                           const float *x, int M, int Ystride, int Xstride);
int dual_ds4f_prefill_gemm_multi(
    void *ctx, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, const int *M, const int *Ystride,
    const int *Xstride, int n);

#endif
