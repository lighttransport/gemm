#ifndef DS41F_INT8_H
#define DS41F_INT8_H
#include <stddef.h>
#include <stdint.h>

/* Four rows, K blocks, then 32-column chunks containing four contiguous
 * 32-byte rows. Scales are per row and K block; padding rows are zero. */
typedef struct {
    size_t rows, cols, block, bytes;
    int8_t *weight;
    float *scale;
} ds41f_int8;

typedef struct {size_t elements,block;int8_t *data;float *scale;} ds41f_int8_input;
int ds41f_int8_prepare_input(ds41f_int8_input *out,const float *x,size_t elements,size_t block);
void ds41f_int8_input_free(ds41f_int8_input *input);
int ds41f_int8_matvec_prepared(float *out,const ds41f_int8 *q,const ds41f_int8_input *input,
                              size_t group_rows,int reference);
int ds41f_int8_from_fp8(ds41f_int8 *out, const uint8_t *weight,
                      const uint8_t *scale, size_t rows, size_t cols, size_t block);
void ds41f_int8_free(ds41f_int8 *q);
int ds41f_int8_matvec(float *out, const ds41f_int8 *q, const float *x,
                     size_t group_rows, int reference);
/* Up to six tokens; output/input rows are token-major. FP8 activation
 * rounding, when required by the model, precedes this INT8-only operator. */
int ds41f_int8_matmul_prepared(float *out,size_t output_stride,const ds41f_int8 *q,
                              const ds41f_int8_input *input,size_t batch,size_t group_rows,int reference);
int ds41f_int8_matmul(float *out,size_t output_stride,const ds41f_int8 *q,const float *x,
                      size_t input_stride,size_t batch,size_t group_rows,int reference);
#endif
