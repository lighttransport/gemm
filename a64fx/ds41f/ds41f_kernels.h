#ifndef DS41F_KERNELS_H
#define DS41F_KERNELS_H

#include <stddef.h>
#include <stdint.h>

float ds41f_fp8_e4m3_to_f32(uint8_t x);
float ds41f_e8m0_to_f32(uint8_t x);
uint8_t ds41f_f32_to_fp8(float x);
/* BF16 input rounding, group-32 dynamic FP8 with power-of-two scales;
 * returns dequantized floats for the initial CPU linear implementation. */
/* Set once before worker threads start; zero retains serial small inputs. */
void ds41f_set_quant_parallel(int enabled);
int ds41f_get_quant_parallel(void);
int ds41f_act_quant(float *out,const float *x,size_t n);
int ds41f_act_quant_ref(float *out,const float *x,size_t n);
uint16_t ds41f_f32_to_bf16(float x);
float ds41f_bf16_to_f32(uint16_t x);
void ds41f_round_bf16(float *x,size_t n);
/* Checkpoint MXFP4: adjacent low/high nibbles, E2M1 max=6,
 * per-row E8M0 scale for each 32 logical columns. */
int ds41f_mxfp4_matvec_ref(float *out, const uint8_t *packed,
                          const uint8_t *scales, const float *x,
                          size_t rows, size_t cols);

/* FP8 dense scales: [ceil(rows/32), ceil(cols/block_cols)].
 * The INT8 helper is generic and does NOT decode packed MXFP4 experts. */
void ds41f_fp8_matvec_ref(float *out, const uint8_t *w, const uint8_t *scale,
                          const float *x, size_t rows, size_t cols,
                          size_t block_cols);
void ds41f_i8_matvec_ref(float *out, const int8_t *w, const uint8_t *scale,
                         const float *x, size_t rows, size_t cols,
                         size_t block_cols);
void ds41f_rmsnorm_ref(float *out, const float *x, const uint16_t *weight,
                       size_t n, float eps);

#endif
