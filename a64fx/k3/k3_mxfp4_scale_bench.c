#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <arm_sve.h>
#include "ggml_dequant.h"
#include "k3_moe.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

/* Variant: scales pre-converted to f32, still broadcast per block (isolates the
 * e8m0 conversion from the GPR->SVE move). */
static void mv_f32scale(float *dst,const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
    const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
    const float*S0,const float*S1,const float*S2,const float*S3,
    const float*S4,const float*S5,const float*S6,const float*S7,const float*x,int k){
    svbool_t pg=svptrue_b32();svfloat32_t kv=svld1(pg,ds4f_kvalues_mxfp4_f32);
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    int nb=k/32;
#pragma clang loop unroll_count(2)
    for(int b=0;b<nb;++b){
        svfloat32_t xl=svld1(pg,x+(size_t)b*32),xh=svld1(pg,x+(size_t)b*32+16);
#define R(W,S,A) do{svuint32_t z=svld1ub_u32(pg,(W)+(size_t)b*16); \
    svuint32_t lo=svand_n_u32_x(pg,z,15),hi=svand_n_u32_x(pg,svlsr_n_u32_x(pg,z,4),15); \
    svfloat32_t p=svmul_x(pg,svtbl_f32(kv,lo),xl);p=svmla_x(pg,p,svtbl_f32(kv,hi),xh); \
    (A)=svmla_n_f32_x(pg,(A),p,(S)[b]);}while(0)
        R(w0,S0,a0);R(w1,S1,a1);R(w2,S2,a2);R(w3,S3,a3);
        R(w4,S4,a4);R(w5,S5,a5);R(w6,S6,a6);R(w7,S7,a7);
#undef R
    }
    dst[0]=svaddv(pg,a0);dst[1]=svaddv(pg,a1);dst[2]=svaddv(pg,a2);dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4);dst[5]=svaddv(pg,a5);dst[6]=svaddv(pg,a6);dst[7]=svaddv(pg,a7);
}
/* Variant: no scale at all -- upper bound if the broadcast were free. */
static void mv_noscale(float *dst,const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
    const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,const float*x,int k){
    svbool_t pg=svptrue_b32();svfloat32_t kv=svld1(pg,ds4f_kvalues_mxfp4_f32);
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    int nb=k/32;
#pragma clang loop unroll_count(2)
    for(int b=0;b<nb;++b){
        svfloat32_t xl=svld1(pg,x+(size_t)b*32),xh=svld1(pg,x+(size_t)b*32+16);
#define R(W,A) do{svuint32_t z=svld1ub_u32(pg,(W)+(size_t)b*16); \
    svuint32_t lo=svand_n_u32_x(pg,z,15),hi=svand_n_u32_x(pg,svlsr_n_u32_x(pg,z,4),15); \
    (A)=svmla_x(pg,(A),svtbl_f32(kv,lo),xl);(A)=svmla_x(pg,(A),svtbl_f32(kv,hi),xh);}while(0)
        R(w0,a0);R(w1,a1);R(w2,a2);R(w3,a3);R(w4,a4);R(w5,a5);R(w6,a6);R(w7,a7);
#undef R
    }
    dst[0]=svaddv(pg,a0);dst[1]=svaddv(pg,a1);dst[2]=svaddv(pg,a2);dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4);dst[5]=svaddv(pg,a5);dst[6]=svaddv(pg,a6);dst[7]=svaddv(pg,a7);
}
/* Variant: e8m0 byte -> f32 via a 256-entry table, so the scale arrives in an
 * FP register and svmla_n needs no GPR->FPR crossing.  Same bytes on the wire. */
static float e8m0_tab[256];
static void e8m0_tab_init(void){ for(int i=0;i<256;i++){uint32_t b=(uint32_t)i<<23; memcpy(&e8m0_tab[i],&b,4);} }
static void mv_lut(float *dst,const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
    const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
    const uint8_t*S0,const uint8_t*S1,const uint8_t*S2,const uint8_t*S3,
    const uint8_t*S4,const uint8_t*S5,const uint8_t*S6,const uint8_t*S7,const float*x,int k){
    svbool_t pg=svptrue_b32();svfloat32_t kv=svld1(pg,ds4f_kvalues_mxfp4_f32);
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    int nb=k/32;
#pragma clang loop unroll_count(2)
    for(int b=0;b<nb;++b){
        svfloat32_t xl=svld1(pg,x+(size_t)b*32),xh=svld1(pg,x+(size_t)b*32+16);
#define R(W,S,A) do{svuint32_t z=svld1ub_u32(pg,(W)+(size_t)b*16); \
    svuint32_t lo=svand_n_u32_x(pg,z,15),hi=svand_n_u32_x(pg,svlsr_n_u32_x(pg,z,4),15); \
    svfloat32_t p=svmul_x(pg,svtbl_f32(kv,lo),xl);p=svmla_x(pg,p,svtbl_f32(kv,hi),xh); \
    (A)=svmla_n_f32_x(pg,(A),p,e8m0_tab[(S)[b]]);}while(0)
        R(w0,S0,a0);R(w1,S1,a1);R(w2,S2,a2);R(w3,S3,a3);
        R(w4,S4,a4);R(w5,S5,a5);R(w6,S6,a6);R(w7,S7,a7);
#undef R
    }
    dst[0]=svaddv(pg,a0);dst[1]=svaddv(pg,a1);dst[2]=svaddv(pg,a2);dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4);dst[5]=svaddv(pg,a5);dst[6]=svaddv(pg,a6);dst[7]=svaddv(pg,a7);
}
int main(void){
    e8m0_tab_init();
    int k=3584, rows=256, groups=rows/8, reps=200;
    size_t wr=(size_t)k/2, sr=(size_t)k/32;
    uint8_t *w=aligned_alloc(256,(size_t)rows*wr), *s=aligned_alloc(256,(size_t)rows*sr);
    float *sf=aligned_alloc(256,(size_t)rows*sr*4), *x=aligned_alloc(256,k*4), y[8]; volatile float sink=0;
    for(size_t i=0;i<(size_t)rows*wr;i++) w[i]=(uint8_t)(i*37);
    for(size_t i=0;i<(size_t)rows*sr;i++){ s[i]=127; sf[i]=1.0f; }
    for(int i=0;i<k;i++) x[i]=1.0f/(i+1);
    double macs=(double)rows*k*reps;
    double b1=1e9,b2=1e9,b3=1e9,t;
    for(int r=0;r<3;r++){
      t=now(); for(int q=0;q<reps;q++) for(int g=0;g<groups;g++){const uint8_t*W=w+(size_t)g*8*wr;const uint8_t*S=s+(size_t)g*8*sr;
        k3_matvec_mxfp4_8row(y,W,W+wr,W+2*wr,W+3*wr,W+4*wr,W+5*wr,W+6*wr,W+7*wr,S,S+sr,S+2*sr,S+3*sr,S+4*sr,S+5*sr,S+6*sr,S+7*sr,x,k); sink+=y[0]+y[7];} 
      if(now()-t<b1)b1=now()-t;
      t=now(); for(int q=0;q<reps;q++) for(int g=0;g<groups;g++){const uint8_t*W=w+(size_t)g*8*wr;const float*S=sf+(size_t)g*8*sr;
        mv_f32scale(y,W,W+wr,W+2*wr,W+3*wr,W+4*wr,W+5*wr,W+6*wr,W+7*wr,S,S+sr,S+2*sr,S+3*sr,S+4*sr,S+5*sr,S+6*sr,S+7*sr,x,k); sink+=y[0]+y[7];} 
      if(now()-t<b2)b2=now()-t;
      t=now(); for(int q=0;q<reps;q++) for(int g=0;g<groups;g++){const uint8_t*W=w+(size_t)g*8*wr;
        mv_noscale(y,W,W+wr,W+2*wr,W+3*wr,W+4*wr,W+5*wr,W+6*wr,W+7*wr,x,k); sink+=y[0]+y[7];} 
      if(now()-t<b3)b3=now()-t;
    }
    printf("1 thread, k=%d rows=%d\n",k,rows);
    printf("  current (e8m0 byte -> f32 -> broadcast) : %6.2f Gmac/s\n",macs/b1/1e9);
    printf("  pre-converted f32 scale, still broadcast: %6.2f Gmac/s  (%.2fx)\n",macs/b2/1e9,b1/b2);
    printf("  no scale at all (upper bound)           : %6.2f Gmac/s  (%.2fx)\n",macs/b3/1e9,b1/b3);
    { double b4=1e9;
      for(int r=0;r<3;r++){ double t=now();
        for(int q=0;q<reps;q++) for(int g=0;g<groups;g++){const uint8_t*W=w+(size_t)g*8*wr;const uint8_t*S=s+(size_t)g*8*sr;
          mv_lut(y,W,W+wr,W+2*wr,W+3*wr,W+4*wr,W+5*wr,W+6*wr,W+7*wr,S,S+sr,S+2*sr,S+3*sr,S+4*sr,S+5*sr,S+6*sr,S+7*sr,x,k); sink+=y[0]+y[7];}
        if(now()-t<b4)b4=now()-t; }
      printf("  e8m0 via 256-entry f32 LUT (same bytes) : %6.2f Gmac/s  (%.2fx)\n",macs/b4/1e9,b1/b4); }
    printf("(sink %g)\n",(float)sink);
    return 0;
}
