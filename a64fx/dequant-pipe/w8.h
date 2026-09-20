#ifndef DEQUANT_W8_H
#define DEQUANT_W8_H
#include <stdint.h>
/* Fixed SVE512, K=128, eight N vectors in K-major order.
 * FP16: N=256, 32768 weight bytes. FP32: N=128, 16384 bytes.
 * One shared activation vector; sequential K-order FMA, matching accumulation
 * width. No scales, clipping, activation quantization, or padded/tail reads.
 * FP8 supports E4M3FN and IEEE E5M2, including subnormals, signed zero,
 * NaNs, and E5M2 infinities. NaN payload preservation is not promised. */
void w8_i8_f16(const uint8_t *, const _Float16 *, _Float16 *);
void w8_i8_f32(const uint8_t *, const float *, float *);
void w8_e4m3_f16(const uint8_t *, const _Float16 *, _Float16 *);
void w8_e4m3_f32(const uint8_t *, const float *, float *);
void w8_e5m2_f16(const uint8_t *, const _Float16 *, _Float16 *);
void w8_e5m2_f16_native(const uint8_t *, const _Float16 *, _Float16 *);
void w8_e5m2_f32(const uint8_t *, const float *, float *);
int verify_w8(void);
#endif
