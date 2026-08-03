/*
 * Host-side S3 dense FP8/E8M0 HIPRTC API.
 *
 * This owns the persistent FP8/E8M0 device-weight bank and the single-matvec
 * launch used by the first DS4F integration slice.  It establishes the byte
 * layout and device conversion contract before batched dense dispatches.
 */
#ifndef HIP_DS4F_DENSE_H
#define HIP_DS4F_DENSE_H

#include <stddef.h>
#include <stdint.h>

typedef struct ds4f_tensor ds4f_tensor;

typedef struct hip_ds4f_dense hip_ds4f_dense;

/* DS4F_HIP_BLOCK_THREADS selects the one-row block size (64, 128, or 256).
 * The default is 128 on gfx1201; it can be overridden for another GPU. */

/* Compile the HIPRTC module for device_id. Returns NULL if HIP/ROCm is not
 * available or compilation/loading fails. */
hip_ds4f_dense *hip_ds4f_dense_create(int device_id, int verbose);

void hip_ds4f_dense_destroy(hip_ds4f_dense *ctx);

/* Upload one FP8/E8M0 matrix and retain it on the device. A later
 * hip_ds4f_dense_matvec_loaded call reuses this allocation. */
int hip_ds4f_dense_load(hip_ds4f_dense *ctx,
                        const uint8_t *w, const uint8_t *s,
                        int rows, int cols);

/* Add a matrix to the shared-module device-weight bank and return its stable
 * id. Unlike hip_ds4f_dense_load(), this preserves all previously added
 * matrices, so a whole model layer can be resident at once. */
int hip_ds4f_dense_add(hip_ds4f_dense *ctx,
                       const uint8_t *w, const uint8_t *s,
                       int rows, int cols);

/* Upload and tag a common ds4f_tensor with the returned bank id. */
int hip_ds4f_dense_bind_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t);
/* Upload an FP8 tensor with the CPU-compatible block/lane reduction used by
 * the quality-sensitive KV projection path. */
int hip_ds4f_dense_bind_fp8_ordered_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t);
int hip_ds4f_dense_bind_bf16_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t);
/* Upload an FP8 tensor as an exact BF16 promotion while leaving the common
 * tensor in its original FP8 form. This is useful for selectively accelerating
 * hot shared-expert GEMMs without doubling the whole dense bank. */
int hip_ds4f_dense_bind_fp8_bf16_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t);
/* Upload an FP8 tensor as exact FP16 values. Every finite FP8 value times its
 * power-of-two E8M0 scale is representable in FP16 within the model range;
 * the existing RDNA4 FP16-weight GEMM can therefore replace per-element FP8
 * decode without changing the quantized weights. */
int hip_ds4f_dense_bind_fp8_fp16_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t);

/* Run against the matrix most recently passed to hip_ds4f_dense_load. */
int hip_ds4f_dense_matvec_loaded(hip_ds4f_dense *ctx,
                                 const float *x, float *y);

int hip_ds4f_dense_matvec_id(hip_ds4f_dense *ctx, int id,
                             const float *x, float *y);

/* Adapter for ds4f_model.gpu_dense_matvec.  The tensor must carry a valid
 * gpu_id returned by hip_ds4f_dense_add(). */
int hip_ds4f_dense_matvec_tensor(void *ctx, float *dst,
                                 const ds4f_tensor *t, const float *x);

/* Compute the grouped/block-diagonal o-projection in one GPU launch. The
 * tensor is the full FP8 wo_a matrix; each output row selects its activation
 * group from xbase using (goff + row) / glora. */
int hip_ds4f_dense_matvec_blockdiag(
    void *ctx, float *dst, const ds4f_tensor *t, const float *xbase,
    int gin, int glora, int goff);

/* Batched prefill GEMM: dst[M,Ystride] = W[rows,cols] * x[M,Xstride]^T.
 * BF16 uses a BF16-correct sibling of the RDNA4 16x64 tiled GEMM; FP8 tensors
 * use the same tile geometry with fused FP8/E8M0 dequant. */
int hip_ds4f_dense_gemm_tensor(
    void *ctx, float *dst, const ds4f_tensor *t, const float *x,
    int M, int Ystride, int Xstride);
/* Fuse independent GEMMs that share the same token-major activation tile.
 * The current implementation intentionally requires equal M/K/X; callers
 * transparently fall back to hip_ds4f_dense_gemm_tensor otherwise. */
int hip_ds4f_dense_gemm_tensors(
    void *ctx, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, const int *M, const int *Ystride,
    const int *Xstride, int n);

/* Start up to two independent tensor matvecs on separate streams. The
 * corresponding wait call downloads the outputs passed in dst. */
int hip_ds4f_dense_matvec_tensors_async(
    void *ctx, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, int n);
int hip_ds4f_dense_wait_tensors(void *ctx);

/* One in-flight operation variant for CPU/GPU overlap. The input upload is
 * queued, the caller may do CPU work, and wait() synchronizes and downloads y. */
int hip_ds4f_dense_matvec_loaded_async(hip_ds4f_dense *ctx, const float *x);
int hip_ds4f_dense_matvec_id_async(hip_ds4f_dense *ctx, int id, const float *x);
int hip_ds4f_dense_wait(hip_ds4f_dense *ctx, float *y);

/* Compute y = (w * s) dot x.
 *
 * w is rows*cols bytes, row-major FP8 E4M3FN.
 * s is ceil(rows/128)*ceil(cols/128) bytes, row-major E8M0.
 * x and y are cols/rows F32.  Dimensions may be non-multiples of 128. */
int hip_ds4f_dense_matvec(hip_ds4f_dense *ctx,
                          const uint8_t *w, const uint8_t *s,
                          const float *x, float *y,
                          int rows, int cols);

#endif /* HIP_DS4F_DENSE_H */
