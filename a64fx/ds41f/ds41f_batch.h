#ifndef DS41F_BATCH_H
#define DS41F_BATCH_H
#include <stddef.h>
#include <stdint.h>
/* One group, useful for exact packing regressions. */
int ds41f_batch_quantize32(uint16_t out[32],const float x[32]);
/* Token-major x[batch][cols], out[batch][rows]. FP8 32x32 weight scales or
 * MXFP4 adjacent nibbles with per-row group-32 scales. Quantizes activations
 * exactly like the decode path and uses BF16 packed SVE FMA panels. */
int ds41f_quant_batch(float *out,const uint8_t *weight,const uint8_t *scale,
                      const float *x,size_t rows,size_t cols,size_t batch,int fp4);
/* Optional diagnostic wall times: B-pack, A-pack, GEMM, BF16 rounding. */
int ds41f_quant_batch_profile(float *out,const uint8_t *weight,const uint8_t *scale,
                      const float *x,size_t rows,size_t cols,size_t batch,int fp4,double times[4]);
#endif
