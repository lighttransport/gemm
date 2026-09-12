#ifndef DS41F_SVE_H
#define DS41F_SVE_H

#include <stddef.h>
#include <stdint.h>
/* Independent contiguous groups; weight-scale row blocks must not cross groups. */
int ds41f_fp8_grouped_matvec(float *out,const uint8_t *weight,const uint8_t *scale,
                            const float *x,size_t groups,size_t group_rows,size_t cols);
void ds41f_bf16_f32_matvec(float *out,const uint16_t *w,const float *x,size_t rows,size_t cols);
int ds41f_mxfp4_matvec(float *out, const uint8_t *w, const uint8_t *scale,
                        const float *x, size_t rows, size_t cols);
int ds41f_fp8_matvec(float *out, const uint8_t *w, const uint8_t *scale,
                      const float *x, size_t rows, size_t cols);

/* Dispatch entry points. On non-A64FX builds these use the reference path,
 * which keeps the unit tests portable while preserving one runtime ABI. */
void ds41f_bf16_matvec(float *out, const uint16_t *w, const uint16_t *x,
                       size_t rows, size_t cols);
void ds41f_rmsnorm_fast(float *out, const float *x, const uint16_t *weight,
                        size_t n, float eps);

#endif
