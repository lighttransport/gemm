#ifndef DS41F_FP4_SDOT_H
#define DS41F_FP4_SDOT_H
#include "ds41f_int8.h"
/* Four rows, K blocks, then four original 16-byte nibble groups. Weight
 * values/scales remain lossless. Runtime input quantization is approximate. */
int ds41f_mxfp4_pack_sdot(uint8_t *packed,uint8_t *scales,const uint8_t *w,
                   const uint8_t *s,size_t rows,size_t cols);
int ds41f_mxfp4_sdot_prepared(float *out,const uint8_t *w,const uint8_t *s,
                           const ds41f_int8_input *input,size_t rows,size_t cols,int reference);
int ds41f_mxfp4_sdot(float *out,const uint8_t *w,const uint8_t *s,const float *x,
                   size_t rows,size_t cols,int reference);
int ds41f_mxfp4_sdot_pair_prepared(float *gate,float *up,const uint8_t *wg,const uint8_t *sg,
                                  const uint8_t *wu,const uint8_t *su,const ds41f_int8_input *input,
                                  size_t rows,size_t cols);
int ds41f_mxfp4_sdot_matmul(float *out,size_t output_stride,const uint8_t *weight,const uint8_t *scale,
                           const ds41f_int8_input *input,size_t rows,size_t cols,size_t batch);
#endif
