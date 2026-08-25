#define _POSIX_C_SOURCE 200809L
#include "fp8_gemm.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static unsigned r=7;static unsigned rnd(void){r=r*1664525u+1013904223u;return r;}
static void run(const char*n,float*out,const float*a,const fp8_matrix*w,fp8_decode_mode exact){
 double dt[7];fp8_gemv_f32_omp(out,a,w,12,exact);for(int j=0;j<7;++j){double t=now();fp8_gemv_f32_omp(out,a,w,12,exact);dt[j]=now()-t;}
 for(int i=1;i<7;++i){double x=dt[i];int j=i;while(j&&dt[j-1]>x){dt[j]=dt[j-1];--j;}dt[j]=x;}double t=dt[3];
 printf("kernel=%s median_ms=%.3f gflops=%.2f source_GB/s=%.2f\n",n,t*1e3,2.0*w->n*w->k/t/1e9,((size_t)w->n*w->k+(size_t)(w->n/128)*(w->k/128)*4)/t/1e9);}
static void run_i8(const char*n,float*out,const float*a,const fp8_i8_matrix*w){
 double dt[7];fp8_i8_gemv_f32_omp(out,a,w,12);for(int j=0;j<7;++j){double t=now();fp8_i8_gemv_f32_omp(out,a,w,12);dt[j]=now()-t;}
 for(int i=1;i<7;++i){double x=dt[i];int j=i;while(j&&dt[j-1]>x){dt[j]=dt[j-1];--j;}dt[j]=x;}double t=dt[3];
 int tiles=128/w->lane_group;printf("kernel=%s G=%dx%d median_ms=%.3f gflops=%.2f source_GB/s=%.2f checksum=%.7g\n",n,w->lane_group,w->scale_group,t*1e3,2.0*w->n*w->k/t/1e9,((size_t)w->n*w->k+(size_t)(w->n/128)*(w->k/w->scale_group)*tiles*4)/t/1e9,out[w->n/3]);}
int main(int ac,char**av){int n=ac>1?atoi(av[1]):32768,k=ac>2?atoi(av[2]):4096;fp8_matrix w;if(fp8_matrix_alloc(&w,n,k))return 1;
 float*a=aligned_alloc(256,(size_t)k*4),*out=aligned_alloc(256,(size_t)n*4);if(!a||!out)return 1;
 for(size_t i=0;i<(size_t)n*k;++i){uint8_t x=(uint8_t)rnd();if(!(x&0x78))x|=8;if((x&0x7f)==0x7f)x=0x7e|(x&0x80);if(i%6000==0)x=(uint8_t)(rnd()&7);w.codes[i]=x;}
 for(size_t i=0;i<(size_t)(n/128)*(k/128);++i)w.scales[i]=1;for(int i=0;i<k;++i)a[i]=((int)(rnd()&255)-128)/512.0f;
 printf("FP8 E4M3FN N=%d K=%d source=%.1f MiB\n",n,k,((size_t)n*k+(size_t)(n/128)*(k/128)*4)/1048576.0);
 int group=ac>3?atoi(av[3]):128,lane_group=ac>4?atoi(av[4]):128;fp8_i8_matrix wa={0};double t=now();if(fp8_matrix_prepare_i8_tile(&w,&wa,FP8_I8_ABSMAX,group,lane_group))return 1;printf("absmax_prepare_ms=%.3f\n",(now()-t)*1e3);
 if(fp8_matrix_prepare_fast(&w))return 1;printf("exceptions=%u (%.5f%%)\n",w.exception_offsets[n/128],100.0*w.exception_offsets[n/128]/((size_t)n*k));
 run("sparse_exact",out,a,&w,FP8_DECODE_SPARSE_EXACT);
 run_i8("i8_absmax",out,a,&wa);
 run("f16decode_f32fma",out,a,&w,FP8_DECODE_F16_BITS);
 run("exact_lut",out,a,&w,FP8_DECODE_LUT_EXACT);
 run("exact",out,a,&w,FP8_DECODE_INLINE_EXACT);fp8_i8_matrix_free(&wa);fp8_matrix_free(&w);free(a);free(out);return 0;}
