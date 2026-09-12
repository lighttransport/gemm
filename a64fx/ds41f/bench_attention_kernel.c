#define _POSIX_C_SOURCE 200809L
#include "ds41f_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void)
{
    size_t heads=64,dim=512,tokens=640;
    float *q=malloc(heads*dim*4),*kv=malloc(tokens*dim*4),*out=malloc(heads*dim*4),*ref=malloc(heads*dim*4);
    float sink[64];int ids[640];if(!q||!kv||!out||!ref)return 1;
    for(size_t i=0;i<heads*dim;++i)q[i]=sinf((float)i*.031f);
    for(size_t i=0;i<tokens*dim;++i)kv[i]=sinf((float)i*.017f);
    for(size_t h=0;h<heads;++h)sink[h]=(float)((int)(h%9)-4)*2;
    for(size_t i=0;i<tokens;++i)ids[i]=i%17?(int)i:-1;
    double start=now();if(ds41f_sparse_attention_ref(ref,q,kv,sink,ids,tokens,tokens,heads,dim))return 1;
    double reference_time=now()-start;
    if(ds41f_sparse_attention(out,q,kv,sink,ids,tokens,tokens,heads,dim))return 1;
    float worst=0;for(size_t i=0;i<heads*dim;++i){float err=fabsf(out[i]-ref[i]);worst=fmaxf(worst,err);
        if(!isfinite(out[i])||err>5e-5f*(1+fabsf(ref[i]))){fprintf(stderr,"attention mismatch %zu %g %g\n",i,out[i],ref[i]);return 1;}}
    double best=1e9;for(int it=0;it<10;++it){start=now();if(ds41f_sparse_attention(out,q,kv,sink,ids,tokens,tokens,heads,dim))return 1;
        double elapsed=now()-start;if(elapsed<best)best=elapsed;}
    reference_time=1e9;
    for(int it=0;it<5;++it){start=now();if(ds41f_sparse_attention_ref(ref,q,kv,sink,ids,tokens,tokens,heads,dim))return 1;
        double elapsed=now()-start;if(elapsed<reference_time)reference_time=elapsed;}
    printf("SPARSE_ATTN PASS heads=%zu dim=%zu selected=%zu max_abs=%g reference_ms=%.3f SVE_ms=%.3f speedup=%.2f\n",heads,dim,tokens,worst,reference_time*1e3,best*1e3,reference_time/best);
    free(q);free(kv);free(out);free(ref);return 0;
}
