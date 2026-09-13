#include "ds41f_cache.h"
#include "ds41f_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct {float score;int id;} candidate;
static int ranked(const void *a,const void *b)
{
    const candidate *x=a,*y=b;
    if(x->score!=y->score)return x->score>y->score?-1:1;
    return (x->id>y->id)-(x->id<y->id);
}
static int selection_large(void)
{
    const size_t n=32771,k=2048;
    float *scores=malloc(n*sizeof *scores);
    candidate *reference=malloc(n*sizeof *reference);
    int *ids=malloc(k*sizeof *ids);
    unsigned char *selected=calloc(n,1);
    if(!scores||!reference||!ids||!selected)return 1;
    size_t valid=0;uint32_t rng=1;
    for(size_t i=0;i<n;++i){rng=rng*1664525u+1013904223u;
        scores[i]=(float)(rng%1009);
        if(i%53==0)scores[i]=-INFINITY;
        if(i%127==0)scores[i]=NAN;
        if(i%8192==0)scores[i]=INFINITY;
        if(!isnan(scores[i])&&scores[i]!=-INFINITY)
            reference[valid++]=(candidate){scores[i],(int)i};
    }
    qsort(reference,valid,sizeof *reference,ranked);
    for(size_t i=0;i<k;++i)selected[reference[i].id]=1;
    if(ds41f_select_topk(scores,n,k,ids)!=k)return 1;
    for(size_t i=0;i<k;++i)
        if(ids[i]<0||(size_t)ids[i]>=n||!selected[ids[i]]||(i&&ids[i]<=ids[i-1]))return 1;
    free(scores);free(reference);free(ids);free(selected);return 0;
}
int main(void)
{
    float x[32],y[32];uint8_t packed[18];
    for(int i=0;i<32;++i)x[i]=(i&1)?-6:3;
    for(int mode=0;mode<2;++mode){int group=mode?16:32;
        if(ds41f_fp4_pack(packed,x,32,group,mode)||ds41f_fp4_unpack(y,packed,32,group,mode))return 1;
        for(int i=0;i<32;++i)if(x[i]!=y[i])return 1;}
    float scores[7]={1,5,3,5,-INFINITY,2,INFINITY};int ids[4];
    if(ds41f_select_topk(scores,7,4,ids)!=4||ids[0]!=1||ids[1]!=2||ids[2]!=3||ids[3]!=6)return 1;
    if(ds41f_select_topk(scores,7,2,ids)!=2||ids[0]!=1||ids[1]!=6)return 1;
    const float values[16]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
    for(int mode=0;mode<2;++mode)for(int scale=0;scale<255;++scale){
        if(mode&&scale==127)continue;
        for(int i=0;i<16;++i)packed[i]=(uint8_t)((2*i%16)|((2*i+1)%16)<<4);
        packed[16]=packed[17]=(uint8_t)scale;
        if(ds41f_fp4_unpack(y,packed,32,mode?16:32,mode))return 1;
        float s=mode?ds41f_fp8_e4m3_to_f32((uint8_t)scale):ds41f_e8m0_to_f32((uint8_t)scale);
        for(int i=0;i<32;++i){union {float f;uint32_t u;}actual={y[i]},expected={ds41f_bf16_to_f32(ds41f_f32_to_bf16(values[i%16]*s))};
            if(actual.u!=expected.u)return 1;}
    }
    if(selection_large())return 1;
    puts("DS41F_CACHE PASS packed_scales signed_nibbles selection_ties infinity top2048_of32771");return 0;
}
