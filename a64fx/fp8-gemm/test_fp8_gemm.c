#include "fp8_gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
static unsigned r=3;static unsigned rnd(void){r=r*1664525u+1013904223u;return r;}
int main(void){fp8_matrix w;fp8_i8_matrix wi={0};if(fp8_matrix_alloc(&w,256,256))return 1;
 float*a=malloc(256*4),*ref=malloc(256*4),*got=malloc(256*4),*iref=malloc(256*4);if(!a||!ref||!got||!iref)return 1;
 for(int i=0;i<256*256;++i)w.codes[i]=(uint8_t)rnd();
 for(int i=0;i<4;++i)w.scales[i]=ldexpf(1.0f,(int)(rnd()%9)-4);
 for(int i=0;i<256;++i)a[i]=((int)(rnd()&255)-128)/512.0f;
 if(fp8_gemv_reference(ref,a,&w)||fp8_matrix_prepare_i8(&w,&wi,FP8_I8_ABSMAX)||
    fp8_i8_gemv_reference(iref,a,&wi)||fp8_i8_gemv_f32_omp(got,a,&wi,2))return 1;
 double ne=0,de=0;for(int i=0;i<256;++i){double d=got[i]-iref[i];ne+=d*d;de+=(double)iref[i]*iref[i];}
 double rel=sqrt(ne/(de+1e-30));printf("i8 assembly relative_l2=%.8g\n",rel);if(!isfinite(rel)||rel>2e-6)return 1;
 ne=0;de=0;for(int i=0;i<256;++i){double d=iref[i]-ref[i];ne+=d*d;de+=(double)ref[i]*ref[i];}
 printf("i8 requant relative_l2=%.8g\n",sqrt(ne/(de+1e-30)));
 fp8_i8_matrix_free(&wi);
 if(fp8_matrix_prepare_i8(&w,&wi,FP8_I8_MSE)||fp8_i8_gemv_reference(iref,a,&wi)||
    fp8_i8_gemv_f32_omp(got,a,&wi,2))return 1;
 ne=0;de=0;for(int i=0;i<256;++i){double d=got[i]-iref[i];ne+=d*d;de+=(double)iref[i]*iref[i];}
 rel=sqrt(ne/(de+1e-30));printf("i8 mse assembly relative_l2=%.8g\n",rel);if(!isfinite(rel)||rel>2e-6)return 1;
 fp8_i8_matrix_free(&wi);
 if(fp8_matrix_prepare_i8_tile(&w,&wi,FP8_I8_ABSMAX,2,32)||
    fp8_i8_gemv_reference(iref,a,&wi)||fp8_i8_gemv_f32_omp(got,a,&wi,2))return 1;
 ne=0;de=0;for(int i=0;i<256;++i){double d=got[i]-iref[i];ne+=d*d;de+=(double)iref[i]*iref[i];}
 rel=sqrt(ne/(de+1e-30));printf("i8 32x2 assembly relative_l2=%.8g\n",rel);if(!isfinite(rel)||rel>2e-6)return 1;
 if(fp8_matrix_prepare_fast(&w)||fp8_gemv_f32_omp(got,a,&w,2,FP8_DECODE_INLINE_EXACT))return 1;
 ne=0;de=0;for(int i=0;i<256;++i){double d=got[i]-ref[i];ne+=d*d;de+=(double)ref[i]*ref[i];}
 rel=sqrt(ne/(de+1e-30));printf("exact relative_l2=%.8g\n",rel);if(!isfinite(rel)||rel>2e-6)return 1;
 if(fp8_gemv_f32_omp(got,a,&w,2,FP8_DECODE_LUT_EXACT))return 1;ne=0;de=0;
 for(int i=0;i<256;++i){double d=got[i]-ref[i];ne+=d*d;de+=(double)ref[i]*ref[i];}
 rel=sqrt(ne/(de+1e-30));printf("lut relative_l2=%.8g\n",rel);if(!isfinite(rel)||rel>2e-6)return 1;
 if(fp8_gemv_f32_omp(got,a,&w,2,FP8_DECODE_SPARSE_EXACT))return 1;ne=0;de=0;
 for(int i=0;i<256;++i){double d=got[i]-ref[i];ne+=d*d;de+=(double)ref[i]*ref[i];}
 rel=sqrt(ne/(de+1e-30));printf("sparse relative_l2=%.8g\n",rel);if(!isfinite(rel)||rel>2e-6)return 1;
 puts("FP8 GEMV tests: PASS");fp8_i8_matrix_free(&wi);fp8_matrix_free(&w);free(a);free(ref);free(got);free(iref);return 0;}
