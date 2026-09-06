/* A64FX: register-blocked (4row x 5tok) GEMM — bf16-widen->f32-svmla  vs  native-fp16 svmla_f16
 * (with per-KBLK fp32 re-accumulation to bound the fp16 accumulation error). Weight-streaming,
 * compute-bound regime (M>=8, weights reused across the 5-token block).
 *
 * FINDING (2026-07-03, A64FX, rows=8192 cols=6144, 48t, correct NUMA):
 *   KBLK=6144 (full-K fp16 accum, single reduce):  bf16 2341 -> fp16 2877 Gop/s = 1.23x, rms 2.8e-3
 *   KBLK=512  (accurate per-block fp32 re-accum):   bf16 2341 -> fp16 1219 Gop/s = 0.51x (2x SLOWER!)
 * So native fp16 is NOT a compelling GEMM lever in the natural dot-product structure: the accurate
 * fp32-accum variant is 2x slower (the per-block svaddv reductions dominate), and even the lossy
 * full-K variant is only 1.23x over the already-good bf16-widen path.
 * ROOT CAUSE: a dot-product GEMM needs an svaddv reduction per output; integer `svdot` builds that
 * INTO the instruction (int16 svdot_s64 / int8 svdot_s32 hit 2-4x over w8a16), but float `svmla`
 * does not. The 89%-of-fp16-peak result (FP16_GEMM_CEILING.md) is a hand-tuned ASSEMBLY
 * output-stationary 12x2 tile (C accumulators stay in registers, stored as fp16 with no per-element
 * reduction) — a different GEMM structure that mainly helps L1-resident / small-K (attention).
 * KEEP for reference / other architectures where an output-stationary fp16 GEMM (or fp16-tolerant
 * attention) applies. Build: fcc -Nclang -O3 -march=armv8.2-a+sve+fp16 -ffp-contract=fast -fopenmp.
 * Run: ./fp16_rb <rows> <cols> <N> <KBLK>   (N rounded to a multiple of 5).  Companion: fp16_gemm_bench.c
 * (the simpler 8row x 1tok / BW-bound version). */
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
int main(int argc,char**argv){
  int rows=argc>1?atoi(argv[1]):8192, cols=argc>2?atoi(argv[2]):6144, N=argc>3?atoi(argv[3]):30, reps=8, KBLK=argc>4?atoi(argv[4]):512;
  N=(N/5)*5; if(N<5)N=5;
  uint16_t*Wb=aligned_alloc(256,(size_t)rows*cols*2); __fp16*Wh=aligned_alloc(256,(size_t)rows*cols*2),*Xh=aligned_alloc(256,(size_t)N*cols*2);
  float*X=aligned_alloc(256,(size_t)N*cols*4),*Yb=aligned_alloc(256,(size_t)N*rows*4),*Yh=aligned_alloc(256,(size_t)N*rows*4);
  #pragma omp parallel for schedule(static)
  for(int r=0;r<rows;r++){uint32_t s=1u+(uint32_t)r*2654435761u;for(int c=0;c<cols;c++){float w=((int)(lcg(&s)%2001)-1000)*3e-4f;Wb[(size_t)r*cols+c]=f2bf(w);Wh[(size_t)r*cols+c]=(__fp16)w;}}
  for(size_t i=0;i<(size_t)N*cols;i++){uint32_t s=1;float x=((int)((lcg(&s)+(uint32_t)i)%4001)-2000)*1e-3f;X[i]=x;Xh[i]=(__fp16)x;}
  double tb=1e30;
  for(int it=0;it<reps;it++){double t0=sec();int nb=rows/4;
    #pragma omp parallel for schedule(static)
    for(int bi=0;bi<nb;bi++){int r=bi*4;svbool_t pt=svptrue_b32();int vl=(int)svcntw();
      const uint16_t*w0=Wb+(size_t)r*cols,*w1=w0+cols,*w2=w0+2*(size_t)cols,*w3=w0+3*(size_t)cols;
      for(int t=0;t+4<N;t+=5){const float*x0=X+(size_t)t*cols,*x1=x0+cols,*x2=x0+2*(size_t)cols,*x3=x0+3*(size_t)cols,*x4=x0+4*(size_t)cols;
        svfloat32_t b00=svdup_f32(0),b01=svdup_f32(0),b02=svdup_f32(0),b03=svdup_f32(0),b10=svdup_f32(0),b11=svdup_f32(0),b12=svdup_f32(0),b13=svdup_f32(0),b20=svdup_f32(0),b21=svdup_f32(0),b22=svdup_f32(0),b23=svdup_f32(0),b30=svdup_f32(0),b31=svdup_f32(0),b32=svdup_f32(0),b33=svdup_f32(0),b40=svdup_f32(0),b41=svdup_f32(0),b42=svdup_f32(0),b43=svdup_f32(0);
        for(int i=0;i<cols;i+=vl){svbool_t pg=svwhilelt_b32(i,cols);
          svfloat32_t v0=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w0+i),16)),v1=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w1+i),16)),v2=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w2+i),16)),v3=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w3+i),16));
          svfloat32_t xv=svld1(pg,x0+i);b00=svmla_x(pg,b00,v0,xv);b01=svmla_x(pg,b01,v1,xv);b02=svmla_x(pg,b02,v2,xv);b03=svmla_x(pg,b03,v3,xv);
          xv=svld1(pg,x1+i);b10=svmla_x(pg,b10,v0,xv);b11=svmla_x(pg,b11,v1,xv);b12=svmla_x(pg,b12,v2,xv);b13=svmla_x(pg,b13,v3,xv);
          xv=svld1(pg,x2+i);b20=svmla_x(pg,b20,v0,xv);b21=svmla_x(pg,b21,v1,xv);b22=svmla_x(pg,b22,v2,xv);b23=svmla_x(pg,b23,v3,xv);
          xv=svld1(pg,x3+i);b30=svmla_x(pg,b30,v0,xv);b31=svmla_x(pg,b31,v1,xv);b32=svmla_x(pg,b32,v2,xv);b33=svmla_x(pg,b33,v3,xv);
          xv=svld1(pg,x4+i);b40=svmla_x(pg,b40,v0,xv);b41=svmla_x(pg,b41,v1,xv);b42=svmla_x(pg,b42,v2,xv);b43=svmla_x(pg,b43,v3,xv);}
        float*y0=Yb+(size_t)t*rows+r,*y1=y0+rows,*y2=y0+2*rows,*y3=y0+3*rows,*y4=y0+4*rows;
        y0[0]=svaddv_f32(pt,b00);y0[1]=svaddv_f32(pt,b01);y0[2]=svaddv_f32(pt,b02);y0[3]=svaddv_f32(pt,b03);y1[0]=svaddv_f32(pt,b10);y1[1]=svaddv_f32(pt,b11);y1[2]=svaddv_f32(pt,b12);y1[3]=svaddv_f32(pt,b13);
        y2[0]=svaddv_f32(pt,b20);y2[1]=svaddv_f32(pt,b21);y2[2]=svaddv_f32(pt,b22);y2[3]=svaddv_f32(pt,b23);y3[0]=svaddv_f32(pt,b30);y3[1]=svaddv_f32(pt,b31);y3[2]=svaddv_f32(pt,b32);y3[3]=svaddv_f32(pt,b33);
        y4[0]=svaddv_f32(pt,b40);y4[1]=svaddv_f32(pt,b41);y4[2]=svaddv_f32(pt,b42);y4[3]=svaddv_f32(pt,b43);}}
    double dt=sec()-t0;if(dt<tb)tb=dt;}
  double th=1e30;
  for(int it=0;it<reps;it++){double t0=sec();int nb=rows/4;
    #pragma omp parallel for schedule(static)
    for(int bi=0;bi<nb;bi++){int r=bi*4;svbool_t pt=svptrue_b16();int vh=(int)svcnth();
      const __fp16*w0=Wh+(size_t)r*cols,*w1=w0+cols,*w2=w0+2*(size_t)cols,*w3=w0+3*(size_t)cols;
      for(int t=0;t+4<N;t+=5){const __fp16*x0=Xh+(size_t)t*cols,*x1=x0+cols,*x2=x0+2*(size_t)cols,*x3=x0+3*(size_t)cols,*x4=x0+4*(size_t)cols;
        float g[20];for(int u=0;u<20;u++)g[u]=0;
        for(int k0=0;k0<cols;k0+=KBLK){int k1=k0+KBLK<cols?k0+KBLK:cols;
          svfloat16_t b00=svdup_f16(0),b01=svdup_f16(0),b02=svdup_f16(0),b03=svdup_f16(0),b10=svdup_f16(0),b11=svdup_f16(0),b12=svdup_f16(0),b13=svdup_f16(0),b20=svdup_f16(0),b21=svdup_f16(0),b22=svdup_f16(0),b23=svdup_f16(0),b30=svdup_f16(0),b31=svdup_f16(0),b32=svdup_f16(0),b33=svdup_f16(0),b40=svdup_f16(0),b41=svdup_f16(0),b42=svdup_f16(0),b43=svdup_f16(0);
          for(int i=k0;i<k1;i+=vh){svbool_t pg=svwhilelt_b16(i,k1);
            svfloat16_t v0=svld1_f16(pg,w0+i),v1=svld1_f16(pg,w1+i),v2=svld1_f16(pg,w2+i),v3=svld1_f16(pg,w3+i);
            svfloat16_t xv=svld1_f16(pg,x0+i);b00=svmla_f16_x(pg,b00,v0,xv);b01=svmla_f16_x(pg,b01,v1,xv);b02=svmla_f16_x(pg,b02,v2,xv);b03=svmla_f16_x(pg,b03,v3,xv);
            xv=svld1_f16(pg,x1+i);b10=svmla_f16_x(pg,b10,v0,xv);b11=svmla_f16_x(pg,b11,v1,xv);b12=svmla_f16_x(pg,b12,v2,xv);b13=svmla_f16_x(pg,b13,v3,xv);
            xv=svld1_f16(pg,x2+i);b20=svmla_f16_x(pg,b20,v0,xv);b21=svmla_f16_x(pg,b21,v1,xv);b22=svmla_f16_x(pg,b22,v2,xv);b23=svmla_f16_x(pg,b23,v3,xv);
            xv=svld1_f16(pg,x3+i);b30=svmla_f16_x(pg,b30,v0,xv);b31=svmla_f16_x(pg,b31,v1,xv);b32=svmla_f16_x(pg,b32,v2,xv);b33=svmla_f16_x(pg,b33,v3,xv);
            xv=svld1_f16(pg,x4+i);b40=svmla_f16_x(pg,b40,v0,xv);b41=svmla_f16_x(pg,b41,v1,xv);b42=svmla_f16_x(pg,b42,v2,xv);b43=svmla_f16_x(pg,b43,v3,xv);}
          g[0]+=(float)svaddv_f16(pt,b00);g[1]+=(float)svaddv_f16(pt,b01);g[2]+=(float)svaddv_f16(pt,b02);g[3]+=(float)svaddv_f16(pt,b03);g[4]+=(float)svaddv_f16(pt,b10);g[5]+=(float)svaddv_f16(pt,b11);g[6]+=(float)svaddv_f16(pt,b12);g[7]+=(float)svaddv_f16(pt,b13);
          g[8]+=(float)svaddv_f16(pt,b20);g[9]+=(float)svaddv_f16(pt,b21);g[10]+=(float)svaddv_f16(pt,b22);g[11]+=(float)svaddv_f16(pt,b23);g[12]+=(float)svaddv_f16(pt,b30);g[13]+=(float)svaddv_f16(pt,b31);g[14]+=(float)svaddv_f16(pt,b32);g[15]+=(float)svaddv_f16(pt,b33);
          g[16]+=(float)svaddv_f16(pt,b40);g[17]+=(float)svaddv_f16(pt,b41);g[18]+=(float)svaddv_f16(pt,b42);g[19]+=(float)svaddv_f16(pt,b43);}
        float*y0=Yh+(size_t)t*rows+r,*y1=y0+rows,*y2=y0+2*rows,*y3=y0+3*rows,*y4=y0+4*rows;
        for(int j=0;j<4;j++){y0[j]=g[j];y1[j]=g[4+j];y2[j]=g[8+j];y3[j]=g[12+j];y4[j]=g[16+j];}}}
    double dt=sec()-t0;if(dt<th)th=dt;}
  double sse=0,sref=0;for(int t=0;t<N;t++)for(int r=0;r<rows;r++){double a=Yb[(size_t)t*rows+r],b=Yh[(size_t)t*rows+r],d=a-b;sse+=d*d;sref+=a*a;}
  double ops=2.0*(double)N*rows*cols;
  printf("rows=%d cols=%d N=%d KBLK=%d  bf16widen=%.0f  fp16=%.0f Gop/s  speedup=%.2fx  rms=%.2e\n",rows,cols,N,KBLK,ops/tb/1e9,ops/th/1e9,tb/th,sqrt(sse/(sref+1e-30)));
  return 0;}
