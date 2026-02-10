#ifndef NORM_SVE_H
#define NORM_SVE_H

#include <stdint.h>

/* ════════════════════════════════════════════════════════════════
 * SVE-Optimized LayerNorm & RMSNorm for A64FX
 *
 * All kernels use 8x-unrolled SVE with SW prefetch.
 * FP16 variants: uint16_t* input/output, FP32 compute.
 * gamma/beta may be NULL (skip affine transform).
 * ════════════════════════════════════════════════════════════════ */

/* ── RMSNorm Forward ── */

/* Single-row FP32: y[i] = gamma[i] * x[i] / rms(x) */
void rmsnorm_fwd_f32(const float *x, float *y, const float *gamma,
                     float eps, int N);

/* Single-row FP16 in/out, FP32 compute */
void rmsnorm_fwd_f16(const uint16_t *x_f16, uint16_t *y_f16,
                     const float *gamma, float eps, int N);

/* Batch: OpenMP parallel for over M rows */
void rmsnorm_batch_fwd_f32(const float *x, float *y, const float *gamma,
                           float eps, int M, int N);

void rmsnorm_batch_fwd_f16(const uint16_t *x_f16, uint16_t *y_f16,
                           const float *gamma, float eps, int M, int N);

/* ── LayerNorm Forward ── */

/* Single-row FP32: y[i] = gamma[i] * (x[i] - mean) / std + beta[i] */
void layernorm_fwd_f32(const float *x, float *y, const float *gamma,
                       const float *beta, float eps, int N);

/* Single-row FP16 in/out, FP32 compute */
void layernorm_fwd_f16(const uint16_t *x_f16, uint16_t *y_f16,
                       const float *gamma, const float *beta,
                       float eps, int N);

/* Batch: OpenMP parallel for over M rows */
void layernorm_batch_fwd_f32(const float *x, float *y, const float *gamma,
                             const float *beta, float eps, int M, int N);

void layernorm_batch_fwd_f16(const uint16_t *x_f16, uint16_t *y_f16,
                             const float *gamma, const float *beta,
                             float eps, int M, int N);

/* ── RMSNorm Backward ──
 * dx[i] = (dy[i]*gamma[i] - x[i]*ds/N) * inv_rms
 *   where ds = sum(dy[j]*gamma[j]*x[j]) / N
 * dgamma[i] += dy[i] * x[i] * inv_rms
 */
void rmsnorm_bwd_f32(const float *dy, const float *x, float *dx,
                     const float *gamma, float *dgamma,
                     float inv_rms, int N);

/* Batch backward: per-thread dgamma reduction */
void rmsnorm_batch_bwd_f32(const float *dy, const float *x, float *dx,
                           const float *gamma, float *dgamma,
                           const float *inv_rms, int M, int N);

/* ── LayerNorm Backward ──
 * dx[i] = inv_std * (dy[i]*gamma[i] - (ds + db*xhat[i])/N)
 *   where xhat = (x - mean)*inv_std
 *         ds = sum(dy*gamma*xhat)
 *         db = sum(dy*gamma)
 * dgamma[i] += dy[i] * xhat[i]
 * dbeta[i]  += dy[i]
 */
void layernorm_bwd_f32(const float *dy, const float *x, float *dx,
                       const float *gamma, float *dgamma, float *dbeta,
                       float mean, float inv_std, int N);

/* Batch backward: per-thread dgamma/dbeta reduction */
void layernorm_batch_bwd_f32(const float *dy, const float *x, float *dx,
                             const float *gamma, float *dgamma, float *dbeta,
                             const float *mean, const float *inv_std,
                             int M, int N);

/* ── Reference / Scalar ── */

/* Double-precision reference (for accuracy comparison) */
void rmsnorm_ref_f64(const float *x, float *y, const float *gamma,
                     float eps, int N);
void layernorm_ref_f64(const float *x, float *y, const float *gamma,
                       const float *beta, float eps, int N);

/* Scalar FP32 baseline */
void rmsnorm_scalar_f32(const float *x, float *y, const float *gamma,
                        float eps, int N);
void layernorm_scalar_f32(const float *x, float *y, const float *gamma,
                          const float *beta, float eps, int N);

#endif /* NORM_SVE_H */
