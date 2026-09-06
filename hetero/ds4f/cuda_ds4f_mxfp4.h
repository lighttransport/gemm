#ifndef CUDA_DS4F_MXFP4_H
#define CUDA_DS4F_MXFP4_H
#include <stddef.h>
#include <stdint.h>
typedef struct cuda_ds4f_mxfp4 cuda_ds4f_mxfp4;
cuda_ds4f_mxfp4 *cuda_ds4f_mxfp4_create(int device_id, int verbose);
/* cache_mb is a hard device allocation ceiling for the contiguous weight
 * pool.  The server passes 12288 MiB so a 16 GiB display GPU retains 4 GiB. */
cuda_ds4f_mxfp4 *cuda_ds4f_mxfp4_create_ex(int device_id, int verbose,
                                            int cache_mb);
void cuda_ds4f_mxfp4_destroy(cuda_ds4f_mxfp4 *ctx);
void cuda_ds4f_mxfp4_set_terms(cuda_ds4f_mxfp4 *ctx, int terms);
void cuda_ds4f_mxfp4_set_no_evict(cuda_ds4f_mxfp4 *ctx, int enabled);
int cuda_ds4f_mxfp4_load(cuda_ds4f_mxfp4 *ctx, const uint8_t *w,
                         const uint8_t *scale, int rows, int cols);
int cuda_ds4f_mxfp4_load_packed(cuda_ds4f_mxfp4 *ctx, const uint8_t *packed,
                                int rows, int cols);
int cuda_ds4f_mxfp4_gemm(cuda_ds4f_mxfp4 *ctx, float *dst, const float *x,
                         int tokens, int rows, int cols);
int cuda_ds4f_mxfp4_gemm_batch(cuda_ds4f_mxfp4 *ctx, int n,
                               float *const *dst, const uint8_t *const *w,
                               const uint8_t *const *s, const float *const *x,
                               const int *M, const int *rows, const int *cols);
int cuda_ds4f_mxfp4_gemm_batch_packed(cuda_ds4f_mxfp4 *ctx, int n,
                               float *const *dst, const uint8_t *const *packed,
                               const float *const *x, const int *M,
                               const int *rows, const int *cols);
int cuda_ds4f_mxfp4_warm(cuda_ds4f_mxfp4 *ctx, const uint8_t *w,
                         const uint8_t *scale, int rows, int cols);
#endif
