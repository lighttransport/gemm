#ifndef DS41F_WEIGHTS_H
#define DS41F_WEIGHTS_H
#include <stddef.h>
typedef struct {
    char name[192],dtype[16];
    size_t rows,cols,bytes;
    void *data;
} ds41f_weight;
typedef struct { ds41f_weight *items; size_t count,bytes; } ds41f_weights;
int ds41f_weights_load(ds41f_weights *store,const char *stage,const char *prefix,size_t limit);
void ds41f_weights_free(ds41f_weights *store);
const ds41f_weight *ds41f_weight_find(const ds41f_weights *store,const char *name);
/* raw=1: no activation quantization/BF16 output rounding, for promoted
 * checkpoint tensors and grouped BF16 output projection. */
int ds41f_linear(const ds41f_weights *store,const char *base,float *out,
                 const float *x,int raw);
int ds41f_norm(const ds41f_weights *store,const char *name,float *out,const float *x);
void ds41f_round_bf16(float *x,size_t n);
#endif
