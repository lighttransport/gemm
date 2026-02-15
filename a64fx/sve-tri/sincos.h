#ifndef SINCOS_H
#define SINCOS_H

#include <stdint.h>

/*
 * SVE-optimized sincos for fp32 on A64FX
 *
 * Two implementations:
 *   1) sincos_poly  - Minimax polynomial (Horner scheme with FMLA)
 *   2) sincos_ftmad - SVE trig accelerator (FTSSEL/FTSMUL/FTMAD)
 *
 * Both compute sin and cos simultaneously, sharing range reduction.
 * Accuracy: ~24-bit (full fp32) for |x| < 2^20.
 */

/* Compute sin(theta[i]) and cos(theta[i]) for i=0..n-1 */
void sve_sincos_poly_f32(const float *theta, float *sin_out,
                         float *cos_out, int64_t n);
void sve_sincos_ftmad_f32(const float *theta, float *sin_out,
                          float *cos_out, int64_t n);

/*
 * RoPE: Rotary Position Embedding
 *
 * For each pair (x[2i], x[2i+1]) with angle theta[i]:
 *   x'[2i]   = x[2i]*cos(theta[i]) - x[2i+1]*sin(theta[i])
 *   x'[2i+1] = x[2i]*sin(theta[i]) + x[2i+1]*cos(theta[i])
 *
 * dim must be even. theta has dim/2 elements.
 */
void sve_rope_poly_f32(float *x, const float *theta, int64_t dim);
void sve_rope_ftmad_f32(float *x, const float *theta, int64_t dim);

#endif /* SINCOS_H */
