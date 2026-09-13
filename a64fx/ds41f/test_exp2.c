#include "ds41f_ops.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint32_t reference(int32_t x,int poly)
{
    if(x< -2031616)return 0;
    if(x>=0)return 0x7fffffff;
    int e=x/65536-(x%65536!=0);uint32_t f=(uint32_t)(x-e*65536),m;
    if(poly){m=(f*0x2bdcu)>>16;m+=0x5320;m=(m*f)>>16;m+=0x807b;}
    else m=((f*31791u)>>16)+31791u;
    int shift=e+16;return shift<0?m>>(-shift):m<<shift;
}
int main(void)
{
    size_t n=2031619;int32_t *x=malloc(n*4);uint32_t *y=malloc((n+2)*4);
    if(!x||!y)return 1;
    for(size_t i=0;i<n;++i)x[i]=(int32_t)i-2031617;
    for(int poly=0;poly<2;++poly){y[0]=y[n+1]=0xdeadbeef;
        if(ds41f_exp2_q31(y+1,x,n,poly))return 1;
        for(size_t i=0;i<n;++i){double ref=exp2((double)x[i]/65536),got=y[i+1]/2147483648.0;
            if(y[i+1]!=reference(x[i],poly)||fabs(ref-got)>(poly?.0038:.0300)*ref+0x1p-31){
                fprintf(stderr,"EXP2 fail poly=%d q16=%d got=%u\n",poly,x[i],y[i+1]);return 1;}
            if(i&&y[i+1]<y[i])return 1;
        }
        if(y[0]!=0xdeadbeef||y[n+1]!=0xdeadbeef)return 1;
        int32_t edge[]={INT_MIN,-2031617,-2031616,-1048576,-1,0,1,INT_MAX};uint32_t out[8];
        if(ds41f_exp2_q31(out,edge,8,poly))return 1;
        for(size_t i=0;i<8;++i)if(out[i]!=reference(edge[i],poly))return 1;
    }
    free(x);free(y);
    size_t counts[]={0,1,17,128,639,640};
    for(size_t c=0;c<6;++c)for(int pattern=0;pattern<4;++pattern){size_t n=counts[c];
        float score[641],ref[640],sink=pattern==3?90.f:0;
        for(size_t i=0;i<n;++i)ref[i]=pattern==0?-INFINITY:pattern==1?0.f:-(float)(i%300)*.25f;
        if(ds41f_attention_softmax(ref,n,sink,0))return 1;
        for(int mode=0;mode<4;++mode){
            for(size_t i=0;i<n;++i)score[i]=pattern==0?-INFINITY:pattern==1?0.f:-(float)(i%300)*.25f;
            score[n]=123456;
            if(ds41f_attention_softmax(score,n,sink,mode))return 1;
            double sum=0;for(size_t i=0;i<n;++i){sum+=score[i];
                double tolerance=(mode==3?.065:mode==2?.008:1e-5)*(double)ref[i]+2e-8;
                if(!isfinite(score[i])||fabs(score[i]-ref[i])>tolerance||score[i]<0)return 1;}
            if(score[n]!=123456||sum>1.00001)return 1;
            if(pattern==0&&sum!=0)return 1;
            if(pattern==1&&n==1&&score[0]!=.5f)return 1;
        }
    }
    float bad=NAN;if(ds41f_attention_softmax(&bad,1,0,2)!=EDOM)return 1;
    puts("EXP2 PASS exhaustive_Q16_domain affine/poly2 scalar_oracle monotonic tails underflow canaries softmax_sink masks denominator64");
    return 0;
}
