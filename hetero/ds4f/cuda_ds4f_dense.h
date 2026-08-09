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
int cuda_ds4f_dense_gemm_tensors(void *ctx, float *const *dst,
                                const ds4f_tensor *const *t,
                                const float *const *x, const int *M,
                                const int *Ystride, const int *Xstride, int n);
int cuda_ds4f_dense_prefill_attention(void *ctx,float *dst,const float *q,
    const uint16_t *kv,const float *sink,int M,int pos0,int n_heads,
    int head_dim,int kv_dim,int kv_slots,int window,float scale);
int cuda_ds4f_dense_oproj(void *ctx,float *dst,const ds4f_tensor *wa,
    const ds4f_tensor *wb,const float *x,int M,int groups,int gin,int lora,
    int H,int C,int ointer);
int cuda_ds4f_dense_head_argmax(void *ctx,int *token,const ds4f_tensor *head,
    const float *x,int cols);
int cuda_ds4f_dense_shared_ffn(void *ctx, float *dst,
                               const ds4f_tensor *w1,
                               const ds4f_tensor *w3,
                               const ds4f_tensor *w2,
                               const float *x, int M, int inter, int C,
                               float lim);
int cuda_ds4f_dense_shared_ffn_begin(void *ctx,float *dst,
                               const ds4f_tensor *w1,const ds4f_tensor *w3,
                               const ds4f_tensor *w2,const float *x,int M,
                               int inter,int C,float lim);
int cuda_ds4f_dense_shared_ffn_wait(void *ctx,float *dst,int M,int C);

#endif
