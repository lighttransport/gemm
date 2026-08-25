#include "fp4_gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned rng=1;
static float rnd(void){rng=rng*1664525u+1013904223u;return ((rng>>8)/8388608.0f)-1.0f;}

int main(void){
    const float canonical[8]={0,.5f,1,1.5f,2,3,4,6};
    for(int i=0;i<8;++i){uint8_t q=fp4_e2m1_encode(canonical[i]);
        if(fabsf(fp4_e2m1_decode(q)-canonical[i])>0){fprintf(stderr,"e2m1 %d\n",i);return 1;}}
    for(int e=1;e<254;e+=17){float x=fp4_e8m0_decode((uint8_t)e);
        if(fp4_e8m0_encode_ceil(x)!=(uint8_t)e){fprintf(stderr,"e8m0 %d\n",e);return 1;}}
    for(int b=0;b<=0x7e;++b){float x=fp4_e4m3_decode_positive((uint8_t)b);
        uint8_t q=fp4_e4m3_encode_positive(x);
        if(q!=(uint8_t)b && !(b==0x7f)){fprintf(stderr,"e4m3 %d -> %d\n",b,q);return 1;}}
    enum{M=7,N=32,K=64}; float*w=malloc((size_t)N*K*4),*ref=malloc((size_t)M*N*4),*got=malloc((size_t)M*N*4),*got2=malloc((size_t)M*N*4);
    _Float16*a=malloc((size_t)M*K*2);if(!w||!ref||!got||!got2||!a)return 1;
    for(int i=0;i<N*K;++i)w[i]=rnd()*0.25f;for(int i=0;i<M*K;++i)a[i]=(_Float16)(rnd()*0.5f);
    for(int f=0;f<3;++f){fp4_matrix p;if(fp4_matrix_alloc(&p,(fp4_format)f,N,K)||fp4_quantize_f32(&p,w)||fp4_matrix_prepare_n32(&p))return 1;
        fp4_gemm_reference(ref,a,&p,M,1);
        for(int kc=0;kc<=64;kc+=32){if(fp4_gemm_f16(got,a,&p,M,kc,2)||fp4_gemm_f16_n32(got2,a,&p,M,kc))return 1;double num=0,den=0,num2=0;
            for(int i=0;i<M*N;++i){double d=got[i]-ref[i],d2=got2[i]-ref[i];num+=d*d;num2+=d2*d2;den+=(double)ref[i]*ref[i];}
            double rel=sqrt(num/(den+1e-30)),rel2=sqrt(num2/(den+1e-30));printf("%s kc=%d row_rel=%.6g n32_rel=%.6g\n",fp4_format_name((fp4_format)f),kc,rel,rel2);
            if(!isfinite(rel2)||rel2>0.08)return 1;
            if(!isfinite(rel)||rel>0.08)return 1;
            if(fp4_gemm_f16_l1(got,a,&p,6,kc))return 1;
            double nl=0,dl=0;for(int i=0;i<6*N;++i){double d=got[i]-ref[i];nl+=d*d;dl+=(double)ref[i]*ref[i];}
            double rl=sqrt(nl/(dl+1e-30));printf("%s kc=%d l1asm_rel=%.6g\n",fp4_format_name((fp4_format)f),kc,rl);
            if(!isfinite(rl)||rl>0.08)return 1;}
        fp4_matrix_free(&p);}
    free(w);free(ref);free(got);free(got2);free(a);puts("FP4 GEMM tests: PASS");return 0;
}
