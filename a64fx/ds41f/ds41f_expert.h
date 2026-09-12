#ifndef DS41F_EXPERT_H
#define DS41F_EXPERT_H
#include <stdint.h>
typedef struct { uint8_t *weight[3], *scale[3]; int packed_sdot; } ds41f_expert;
int ds41f_expert_load(ds41f_expert *expert,const char *stage,int layer,int id);
void ds41f_expert_free(ds41f_expert *expert);
/* BF16 boundaries and group-32 dynamic FP8 activations. Scratch contains
 * 3*2304+5120 floats and must not alias x/out. */
int ds41f_expert_forward(const ds41f_expert *expert,float *out,const float *x,
                          float route_weight,float *scratch,int reference);
int ds41f_expert_forward_fused(const ds41f_expert *e,float *out,const float *x,
                               float route_weight,float *scratch,int reference,int fused);
#endif
