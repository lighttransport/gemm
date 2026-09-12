#ifndef DS41F_OPS_H
#define DS41F_OPS_H
#include <stddef.h>
/* Single-token reference operations transcribed from inference/model.py. */
void ds41f_swiglu(float *out,const float *gate,const float *up,size_t n,float limit);
int ds41f_gate(const float *logits,const float *bias,int experts,int k,
               float temperature,float route_scale,int *ids,float *weights);
void ds41f_hc_split(const float mix[24],const float scale[3],const float base[24],
                    int iters,float eps,float pre[4],float post[4],float comb[16]);
void ds41f_hc_pre(float *out,const float *x,const float pre[4],size_t dim);
void ds41f_hc_post(float *out,const float *x,const float *residual,
                   const float post[4],const float comb[16],size_t dim);
void ds41f_engram_fuse(float *x,const float *key,const float *value,
                       const float *qw,const float *kw,size_t dim,float eps);
void ds41f_rope(float *x,size_t heads,size_t dim,size_t rope_dim,size_t pos,
                 double theta,double factor,int original,int inverse);
int ds41f_sparse_attention(float *out,const float *q,const float *kv,
                            const float *sink,const int *ids,size_t selected,
                            size_t tokens,size_t heads,size_t dim);
int ds41f_sparse_attention_ref(float *out,const float *q,const float *kv,
                            const float *sink,const int *ids,size_t selected,
                            size_t tokens,size_t heads,size_t dim);
void ds41f_pool_pair(float *out,const float *a,const float *b,
                      const float *sa,const float *sb,size_t dim);
#endif
