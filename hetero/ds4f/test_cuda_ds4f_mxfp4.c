#include "cuda_ds4f_mxfp4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    const int M=argc>1?atoi(argv[1]):128,N=128,K=4096,nb=K/32; uint8_t *w=malloc((size_t)N*K/2),*s=malloc((size_t)N*nb);
    float *x=malloc((size_t)M*K*4),*y=malloc((size_t)M*N*4); if(!w||!s||!x||!y)return 2;
    for(int r=0;r<N;r++)for(int b=0;b<nb;b++){s[(size_t)r*nb+b]=(uint8_t)(120+(r*13+b*7)%15);for(int j=0;j<16;j++)w[(size_t)r*K/2+b*16+j]=(uint8_t)((r*29+b*17+j*7+3)&255);}
    for(int r=0;r<M;r++)for(int i=0;i<K;i++)x[(size_t)r*K+i]=(((r*31+i*17)%1009)-504)/173.f;
    fprintf(stderr,"host x0=%g x56=%g\n",x[0],x[56*K]); cuda_ds4f_mxfp4 *c=cuda_ds4f_mxfp4_create(0,1);
    int rc=c&&cuda_ds4f_mxfp4_load(c,w,s,N,K)==0&&cuda_ds4f_mxfp4_gemm(c,y,x,M,N,K)==0;
    const float lut[8]={0,.5f,1,1.5f,2,3,4,6}; float ma=0, mr=0; int bad=0;
    float ref0=0, ref1=0, ref128=0; int mridx=0;
    for(int r=0;r<M;r++) for(int n=0;n<N;n++) { float ref=0;
        for(int k=0;k<K;k++) { int kb=k/32, j=k%32; float amax=0; for(int z=0;z<32;z++) if(fabsf(x[(size_t)r*K+kb*32+z])>amax) amax=fabsf(x[(size_t)r*K+kb*32+z]); int ee=amax>0?((int)lrintf(log2f(amax))-2+127):0; if(ee<0)ee=0;if(ee>254)ee=254; float as=ldexpf(1.0f,ee-127); float av=x[(size_t)r*K+k]/as; int aq=0;float ae=fabsf(av)-lut[0];for(int qi=1;qi<8;qi++){float de=fabsf(fabsf(av)-lut[qi]);if(de<ae){ae=de;aq=qi;}}if(av<0)aq|=8;
            uint8_t z=w[(size_t)n*K/2+kb*16+(j/2)]; uint8_t q=(j&1)?(z>>4):(z&15); float ws=ldexpf(1.0f,(int)s[(size_t)n*nb+kb]-127); float v=ws*lut[q&7]*(q&8?-1.f:1.f); ref += as*lut[aq&7]*(aq&8?-1.f:1.f)*v; }
        if(r==0&&n==0) ref0=ref;
        if(r==0&&n==1) ref1=ref;
        if(r==1&&n==0) ref128=ref;
        float d=fabsf(y[(size_t)r*N+n]-ref);
        if(!isfinite(y[(size_t)r*N+n])) bad++;
        if(d>ma){ma=d;mridx=r*N+n;}
        if(fabsf(ref)>1e-6f&&d/fabsf(ref)>mr)mr=d/fabsf(ref);
    }
    const int pass = rc && bad == 0 && ma < 0.1f;
    printf("cuda bridge %s y0=%g ref0=%g y1=%g ref1=%g y128=%g ref128=%g y129=%g bad=%d max_abs=%g idx=%d gpu=%g max_rel=%g\n",pass?"PASS":"FAIL",y[0],ref0,M>1?y[1]:0,ref1,M>1?y[128]:0,ref128,M>1?y[129]:0,bad,ma,mridx,y[mridx],mr);cuda_ds4f_mxfp4_destroy(c);free(w);free(s);free(x);free(y);return pass?0:1;
}
