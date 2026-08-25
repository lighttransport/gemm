#define _POSIX_C_SOURCE 200112L
#include "fp4_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static unsigned rng=3;static float rnd(void){rng=rng*1664525u+1013904223u;return ((rng>>8)/8388608.f)-1.f;}
static void*aa(size_t n){void*p=NULL;return posix_memalign(&p,256,(n+255)&~255ULL)?NULL:p;}

int main(int argc,char**argv){int n=argc>1?atoi(argv[1]):2048,k=argc>2?atoi(argv[2]):4096;
    if(n%32||k%32){fprintf(stderr,"N and K must be multiples of 32\n");return 2;}
    float*src=aa((size_t)n*k*4);for(size_t i=0;i<(size_t)n*k;++i)src[i]=rnd()*.125f;
    int ms[]={1,6,12,24},kcs[]={256,0};
    printf("single-core N32 FP4 GEMM N=%d K=%d\n",n,k);
    for(int f=0;f<3;++f){fp4_matrix w;if(fp4_matrix_alloc(&w,(fp4_format)f,n,k)||
            fp4_quantize_f32(&w,src)||fp4_matrix_prepare_n32(&w))return 1;
        for(int mi=0;mi<4;++mi){int m=ms[mi];_Float16*a=aa((size_t)m*k*2);float*c=aa((size_t)m*n*4);
            for(size_t i=0;i<(size_t)m*k;++i)a[i]=(_Float16)(rnd()*.25f);
            for(int ci=0;ci<2;++ci){int kc=kcs[ci];fp4_gemm_f16_n32(c,a,&w,m,kc);
                double best=1e9;for(int rep=0;rep<5;++rep){double t=now();fp4_gemm_f16_n32(c,a,&w,m,kc);double d=now()-t;if(d<best)best=d;}
                printf("format=%s M=%d kc=%d ms=%.3f gflops=%.2f\n",fp4_format_name((fp4_format)f),m,kc,best*1e3,2.0*m*n*k/best/1e9);}
            if(m%6==0)for(int ci=0;ci<2;++ci){int kc=kcs[ci];fp4_gemm_f16_l1(c,a,&w,m,kc);
                double best=1e9;for(int rep=0;rep<5;++rep){double t=now();fp4_gemm_f16_l1(c,a,&w,m,kc);double d=now()-t;if(d<best)best=d;}
                printf("format=%s kernel=l1asm M=%d kc=%d ms=%.3f gflops=%.2f\n",fp4_format_name((fp4_format)f),m,kc,best*1e3,2.0*m*n*k/best/1e9);}
            free(a);free(c);}fp4_matrix_free(&w);}free(src);return 0;}
