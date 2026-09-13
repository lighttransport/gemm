#include "ds41f_ops.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void gate_reference(const float *logits,const float *bias,int n,int k,
                           float temperature,float scale,int *ids,float *weights)
{
    float scores[385];
    for(int i=0;i<n;++i){float x=logits[i]/temperature;
        scores[i]=sqrtf(fmaxf(x,0)+log1pf(expf(-fabsf(x))));}
    float sum=0;
    for(int j=0;j<k;++j){int best=-1;
        for(int i=0;i<n;++i){int used=0;for(int p=0;p<j;++p)if(ids[p]==i)used=1;
            if(!used&&(best<0||scores[i]+bias[i]>scores[best]+bias[best]))best=i;}
        ids[j]=best;weights[j]=scores[best];sum+=weights[j];}
    for(int j=0;j<k;++j)weights[j]*=scale/(k>1?sum+1e-20f:1);
}
int main(void)
{
    const size_t lengths[]={1,15,16,17,31,32,33,129279,129280,129281};
    float *x=malloc(129281*sizeof(float));if(!x)return 2;
    size_t cases=0;
    for(size_t shape=0;shape<sizeof lengths/sizeof lengths[0];++shape){size_t n=lengths[shape];
        for(int mode=0;mode<4;++mode){
            for(size_t i=0;i<n;++i)x[i]=mode==0?-0.f:mode==1?-(float)(i%19)-1.f:sinf((float)i);
            if(mode==2)x[n-1]=100.f;
            if(mode==3){x[n/2]=100.f;x[n-1]=100.f;}
            size_t expected=0,actual=n;for(size_t i=0;i<n;++i)if(x[i]>x[expected])expected=i;
            if(ds41f_argmax_finite(x,n,&actual)||actual!=expected)return 1;
            ++cases;
        }
        const float invalid[]={NAN,INFINITY,-INFINITY};
        for(size_t kind=0;kind<3;++kind)for(size_t at=0;at<3;++at){size_t where=at==0?0:at==1?n/2:n-1;
            float saved=x[where];x[where]=invalid[kind];size_t actual=n;
            if(ds41f_argmax_finite(x,n,&actual)!=EDOM||actual!=n)return 1;
            x[where]=saved;++cases;}
    }
    size_t index=123;
    if(ds41f_argmax_finite(NULL,1,&index)!=EINVAL||ds41f_argmax_finite(x,0,&index)!=EINVAL||
       ds41f_argmax_finite(x,1,NULL)!=EINVAL)return 1;
    free(x);
    const int experts[]={1,7,383,384,385};
    float logits[385],bias[385],weights[6],expected[6];int ids[6],expected_ids[6];
    size_t gates=0;
    for(size_t shape=0;shape<5;++shape)for(int mode=0;mode<4;++mode){int n=experts[shape],k=n<6?n:6;
        for(int i=0;i<n;++i){logits[i]=mode==0?0:mode==1?-1000.f:sin((double)i*.21)*100.f;
            bias[i]=mode==3?cos((double)i)*10.f:0;}
        gate_reference(logits,bias,n,k,.75f,1.5f,expected_ids,expected);
        if(ds41f_gate(logits,bias,n,k,.75f,1.5f,ids,weights)||
           memcmp(ids,expected_ids,(size_t)k*sizeof(int))||memcmp(weights,expected,(size_t)k*sizeof(float)))return 1;
        ++gates;
    }
    logits[384]=NAN;if(ds41f_gate(logits,bias,385,6,1,1,ids,weights)!=EDOM)return 1;
    logits[384]=0;bias[0]=INFINITY;if(ds41f_gate(logits,bias,385,6,1,1,ids,weights)!=EDOM)return 1;
    printf("SELECTION PASS argmax=%zu gate=%zu first_ties nonfinite_rejection bit_exact_weights\n",cases,gates);
    return 0;
}
