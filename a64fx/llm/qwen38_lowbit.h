#ifndef QWEN38_LOWBIT_H
#define QWEN38_LOWBIT_H

#include <stddef.h>
#include <stdint.h>

/* Internal execution formats, deliberately not GGML type numbers. */
enum q38_lowbit_format { Q38_LB_NVFP4 = 1, Q38_LB_FP6_E2M3 = 2 };
enum q38_lowbit_arithmetic { Q38_LB_F32 = 0, Q38_LB_A8 = 8, Q38_LB_A16 = 16 };

typedef struct { uint8_t codes[4][64], scale[4][8]; } q38_fp4_tile;
/* 512 true FP6 codes (384 bytes) and 16 E8M0 scales: 6.25 bits/weight. */
typedef struct {
    uint8_t low[4][64], high[4][32], scale[2][8];
} q38_fp6_tile;

/* One 16-column activation replicated across eight output rows. A16 uses
 * q = lo + 256*hi, with a centered signed low byte; reduce every subblock. */
typedef struct {
    int8_t lo[2][64], hi[2][64];
    float scale;
} q38_lowbit_act;

size_t q38_lowbit_bytes(int format, int rows, int cols);
float q38_fp6_decode(uint8_t code);
int q38_fp6_encode(float value, uint8_t *code);
float q38_fp4_scale(uint8_t code);
int q38_lowbit_pack_fp6(void *out, size_t bytes, const float *src,
                       size_t stride, int rows, int cols);
int q38_lowbit_pack_nvfp4(void *out, size_t bytes, const void *src,
                         size_t row_bytes, int rows, int cols);
int q38_lowbit_validate(const void *weights, size_t bytes,
                       int format, int rows, int cols);
int q38_lowbit_dequant_row(float *out, const void *weights,
                          int format, int rows, int cols, int row);
int q38_lowbit_prepare(q38_lowbit_act *out, size_t count,
                      const float *x, int cols, int arithmetic);
int q38_lowbit_prepare_sve(q38_lowbit_act *out, size_t count,
                          const float *x, int cols, int arithmetic);
int q38_lowbit_reference(float *out, const void *weights, int format,
                        const float *x, int rows, int cols);
int q38_lowbit_dot(float *out, const void *weights, int format,
                   const q38_lowbit_act *act, int arithmetic,
                   int rows, int cols);
/* Same contract as dot; compiled only for AArch64 SVE. */
int q38_lowbit_sve(float *out, const void *weights, int format,
                   const q38_lowbit_act *act, int arithmetic,
                   int rows, int cols);
/* Fused dequant + FP32 FMA with the original FP32 activation (no A quantizer).
 * The portable double-accumulation reference remains the numerical oracle. */
int q38_lowbit_sve_f32(float *out, const void *weights, int format,
                     const float *x, int rows, int cols);

#endif
