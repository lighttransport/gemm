#define _POSIX_C_SOURCE 200809L
#include "ds41f_ops.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int reference(const float *logits,const float *bias,int n,int k,float temp,float scale,int *ids,float *weights)
{
    float scores[384];int invalid=0;
    #pragma omp parallel for schedule(static) if(n>=384) reduction(|:invalid)
    for(int i=0;i<n;++i){float x=logits[i]/temp;
        if(!isfinite(x)||!isfinite(bias[i])){invalid=1;scores[i]=0;continue;}
        scores[i]=sqrtf(fmaxf(x,0)+log1pf(expf(-fabsf(x))));}
    if(invalid)return EDOM;
    float sum=0;for(int j=0;j<k;++j){int best=-1;
        for(int i=0;i<n;++i){int used=0;for(int p=0;p<j;++p)if(ids[p]==i)used=1;
            if(!used&&(best<0||scores[i]+bias[i]>scores[best]+bias[best]))best=i;}
        ids[j]=best;weights[j]=scores[best];sum+=weights[j];}
    for(int j=0;j<k;++j)weights[j]*=scale/(k>1?sum+1e-20f:1);
    return 0;
}
int main(void)
{
    float x[384],bias[384],a[385],b[385];int ia[385],ib[385],cases=0;
    for(int n=1;n<=384;n=n==1?7:n==7?128:n==128?384:385)for(int k=1;k<=n;k=k==1?3:k==3?6:n+1){
        for(int mode=0;mode<5;++mode){for(int i=0;i<n;++i){x[i]=mode==0?0:mode==1?(float)(i%5):sinf(i*.13f)*37;bias[i]=mode<2?0:cosf(i*.21f)*3;}
            if(mode==3)x[n-1]=NAN;
            if(mode==4)bias[n-1]=INFINITY;
            ia[k]=ib[k]=12345;a[k]=b[k]=12345;
            int ra=reference(x,bias,n,k,1.3f,2.5f,ia,a),rb=ds41f_gate(x,bias,n,k,1.3f,2.5f,ib,b);
            if(ra!=rb||(!ra&&(memcmp(a,b,(k+1)*4)||memcmp(ia,ib,(k+1)*sizeof(int))))){fprintf(stderr,"GATE FAIL n=%d k=%d mode=%d\n",n,k,mode);return 1;}++cases;
        }
    }
    for(int i=0;i<384;++i){x[i]=sinf(i*.13f)*37;bias[i]=cosf(i*.21f)*3;}
    for(int mode=0;mode<2;++mode){double start=now();for(int it=0;it<1000;++it){int rc=mode?ds41f_gate(x,bias,384,6,1.3f,2.5f,ib,b):reference(x,bias,384,6,1.3f,2.5f,ib,b);if(rc)return 1;}
        printf("GATE mode=%d mean_us=%.3f\n",mode,(now()-start)*1e3);}
    printf("GATE PASS cases=%d bit_exact ties negative invalid canaries\n",cases);return 0;
}
