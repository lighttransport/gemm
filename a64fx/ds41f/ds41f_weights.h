#ifndef DS41F_WEIGHTS_H
#define DS41F_WEIGHTS_H
#include <stddef.h>
#include "ds41f_int8.h"
typedef struct {
    char name[192],dtype[16];
    size_t rows,cols,bytes;
    size_t global_rows,row_start;
    void *data;
    ds41f_int8 int8;
} ds41f_weight;
typedef struct { ds41f_weight *items; size_t count,bytes; int fresh_pages,packed_experts; void *input_cache; } ds41f_weights;
int ds41f_weights_load(ds41f_weights *store,const char *stage,const char *prefix,size_t limit);
int ds41f_weights_load_local(ds41f_weights *store,const char *stage,const char *prefix,
                            size_t limit,int fresh_pages);
int ds41f_weights_pack_experts(ds41f_weights *store,size_t limit);
int ds41f_weights_enable_input_cache(ds41f_weights *store);
int ds41f_linear_int8_cached(const ds41f_weights *store,const ds41f_weight *weight,
                            float *out,const float *x,size_t group_rows,int fp8_quantize);
int ds41f_weights_check_tp(ds41f_weights *store,const char *stage,int tp,int rank);
int ds41f_weights_requantize_fp8(ds41f_weights *store,size_t block,size_t limit,int projections_only);
void ds41f_weights_free(ds41f_weights *store);
const ds41f_weight *ds41f_weight_find(const ds41f_weights *store,const char *name);
/* raw=1 skips checkpoint FP8 activation quantization and BF16 output rounding.
 * The experimental INT8 matvec still applies its own input quantization. */
int ds41f_linear(const ds41f_weights *store,const char *base,float *out,
                 const float *x,int raw);
int ds41f_norm(const ds41f_weights *store,const char *name,float *out,const float *x);
void ds41f_round_bf16(float *x,size_t n);
#endif
