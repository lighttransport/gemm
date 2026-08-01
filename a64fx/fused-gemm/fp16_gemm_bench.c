#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <arm_sve.h>
#include <omp.h>
static double sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint32_t lcg(uint32_t*s){*s=*s*1664525u+1013904223u;return *s;}
static inline uint16_t f2bf(float f){uint32_t b;__builtin_memcpy(&b,&f,4);return (uint16_t)((b+0x8000u+((b>>16)&1))>>16);}
#define A8(pfx,init) svfloat32_t pfx##0=init,pfx##1=init,pfx##2=init,pfx##3=init,pfx##4=init,pfx##5=init,pfx##6=init,pfx##7=init
#define H8(pfx,init) svfloat16_t pfx##0=init,pfx##1=init,pfx##2=init,pfx##3=init,pfx##4=init,pfx##5=init,pfx##6=init,pfx##7=init
int main(int argc,char**argv){
  int rows=argc>1?atoi(argv[1]):8192, cols=argc>2?atoi(argv[2]):6144, N=argc>3?atoi(argv[3]):32, reps=8, KBLK=argc>4?atoi(argv[4]):512;
  uint16_t*Wb=aligned_alloc(256,(size_t)rows*cols*2);
  __fp16 *Wh=aligned_alloc(256,(size_t)rows*cols*2), *Xh=aligned_alloc(256,(size_t)N*cols*2);
  float  *X =aligned_alloc(256,(size_t)N*cols*4), *Yb=aligned_alloc(256,(size_t)N*rows*4), *Yh=aligned_alloc(256,(size_t)N*rows*4);
  #pragma omp parallel for schedule(static)
  for(int r=0;r<rows;r++){uint32_t s=1u+(uint32_t)r*2654435761u; for(int c=0;c<cols;c++){float w=((int)(lcg(&s)%2001)-1000)*3e-4f; Wb[(size_t)r*cols+c]=f2bf(w); Wh[(size_t)r*cols+c]=(__fp16)w;}}
  for(size_t i=0;i<(size_t)N*cols;i++){uint32_t s=1; float x=((int)((lcg(&s)+(uint32_t)i)%4001)-2000)*1e-3f; X[i]=x; Xh[i]=(__fp16)x;}
  double tb=1e30;
  for(int it=0;it<reps;it++){ double t0=sec(); int nb=rows/8;
    #pragma omp parallel for schedule(static)
    for(int bi=0;bi<nb;bi++){ int r=bi*8; svbool_t pt=svptrue_b32(); int vl=(int)svcntw();
      const uint16_t*w0=Wb+(size_t)r*cols,*w1=w0+cols,*w2=w0+2*(size_t)cols,*w3=w0+3*(size_t)cols,*w4=w0+4*(size_t)cols,*w5=w0+5*(size_t)cols,*w6=w0+6*(size_t)cols,*w7=w0+7*(size_t)cols;
      for(int t=0;t<N;t++){ const float*x=X+(size_t)t*cols; A8(a,svdup_f32(0));
        for(int i=0;i<cols;i+=vl){ svbool_t pg=svwhilelt_b32(i,cols); svfloat32_t xv=svld1(pg,x+i);
          a0=svmla_x(pg,a0,svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w0+i),16)),xv); a1=svmla_x(pg,a1,svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w1+i),16)),xv);
          a2=svmla_x(pg,a2,svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w2+i),16)),xv); a3=svmla_x(pg,a3,svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w3+i),16)),xv);
          a4=svmla_x(pg,a4,svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w4+i),16)),xv); a5=svmla_x(pg,a5,svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w5+i),16)),xv);
          a6=svmla_x(pg,a6,svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w6+i),16)),xv); a7=svmla_x(pg,a7,svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w7+i),16)),xv); }
        float*y=Yb+(size_t)t*rows+r; y[0]=svaddv_f32(pt,a0);y[1]=svaddv_f32(pt,a1);y[2]=svaddv_f32(pt,a2);y[3]=svaddv_f32(pt,a3);y[4]=svaddv_f32(pt,a4);y[5]=svaddv_f32(pt,a5);y[6]=svaddv_f32(pt,a6);y[7]=svaddv_f32(pt,a7);} }
    double dt=sec()-t0; if(dt<tb)tb=dt; }
  double th=1e30;
  for(int it=0;it<reps;it++){ double t0=sec(); int nb=rows/8;
    #pragma omp parallel for schedule(static)
    for(int bi=0;bi<nb;bi++){ int r=bi*8; svbool_t pt=svptrue_b16(); int vh=(int)svcnth();
      const __fp16*w0=Wh+(size_t)r*cols,*w1=w0+cols,*w2=w0+2*(size_t)cols,*w3=w0+3*(size_t)cols,*w4=w0+4*(size_t)cols,*w5=w0+5*(size_t)cols,*w6=w0+6*(size_t)cols,*w7=w0+7*(size_t)cols;
      for(int t=0;t<N;t++){ const __fp16*x=Xh+(size_t)t*cols; float f0=0,f1=0,f2=0,f3=0,f4=0,f5=0,f6=0,f7=0;
        for(int k0=0;k0<cols;k0+=KBLK){ int k1=k0+KBLK<cols?k0+KBLK:cols; H8(a,svdup_f16(0));
          for(int i=k0;i<k1;i+=vh){ svbool_t pg=svwhilelt_b16(i,k1); svfloat16_t xv=svld1_f16(pg,x+i);
            a0=svmla_f16_x(pg,a0,svld1_f16(pg,w0+i),xv);a1=svmla_f16_x(pg,a1,svld1_f16(pg,w1+i),xv);a2=svmla_f16_x(pg,a2,svld1_f16(pg,w2+i),xv);a3=svmla_f16_x(pg,a3,svld1_f16(pg,w3+i),xv);
            a4=svmla_f16_x(pg,a4,svld1_f16(pg,w4+i),xv);a5=svmla_f16_x(pg,a5,svld1_f16(pg,w5+i),xv);a6=svmla_f16_x(pg,a6,svld1_f16(pg,w6+i),xv);a7=svmla_f16_x(pg,a7,svld1_f16(pg,w7+i),xv); }
          f0+=(float)svaddv_f16(pt,a0);f1+=(float)svaddv_f16(pt,a1);f2+=(float)svaddv_f16(pt,a2);f3+=(float)svaddv_f16(pt,a3);f4+=(float)svaddv_f16(pt,a4);f5+=(float)svaddv_f16(pt,a5);f6+=(float)svaddv_f16(pt,a6);f7+=(float)svaddv_f16(pt,a7); }
        float*y=Yh+(size_t)t*rows+r; y[0]=f0;y[1]=f1;y[2]=f2;y[3]=f3;y[4]=f4;y[5]=f5;y[6]=f6;y[7]=f7; } }
    double dt=sec()-t0; if(dt<th)th=dt; }
  double sse=0,sref=0,mx=0;
  for(int t=0;t<N;t++)for(int r=0;r<rows;r++){double a=Yb[(size_t)t*rows+r],b=Yh[(size_t)t*rows+r],d=a-b;sse+=d*d;sref+=a*a;double rel=fabs(d)/(fabs(a)+1e-6);if(rel>mx)mx=rel;}
  double ops=2.0*(double)N*rows*cols;
  printf("rows=%d cols=%d N=%d KBLK=%d  bf16widen=%.0f Gop/s  fp16=%.0f Gop/s  speedup=%.2fx  rms(fp16-vs-bf16ref)=%.2e maxrel=%.2e\n",rows,cols,N,KBLK,ops/tb/1e9,ops/th/1e9,tb/th,sqrt(sse/(sref+1e-30)),mx);
  return 0;}
