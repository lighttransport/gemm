#include "lightrig_mlp2.h"

void lt_mlp2_f32_scalar(const float *x,const float *w1,const float *b1,const float *w2,const float *b2,float *h,float *y,size_t in,size_t hidden,size_t out){
  size_t i,j;for(i=0;i<hidden;++i){float v=b1[i];for(j=0;j<in;++j)v+=w1[i*in+j]*x[j];h[i]=v>0?v:0;}for(i=0;i<out;++i){float v=b2[i];for(j=0;j<hidden;++j)v+=w2[i*hidden+j]*h[j];y[i]=v;}
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
__attribute__((target("avx2,fma"))) static float dot(const float *a,const float *b,size_t n){__m256 s=_mm256_setzero_ps();size_t i=0;for(;i+8<=n;i+=8)s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);float t[8];_mm256_storeu_ps(t,s);float r=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];for(;i<n;++i)r+=a[i]*b[i];return r;}
__attribute__((target("avx2,fma"))) static void avx(const float*x,const float*w1,const float*b1,const float*w2,const float*b2,float*h,float*y,size_t in,size_t hidden,size_t out){size_t i;for(i=0;i<hidden;++i){float v=dot(w1+i*in,x,in)+b1[i];h[i]=v>0?v:0;}for(i=0;i<out;++i)y[i]=dot(w2+i*hidden,h,hidden)+b2[i];}
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
static float neon_dot(const float *a,const float *b,size_t n){float32x4_t s=vdupq_n_f32(0);size_t i=0;for(;i+4<=n;i+=4)s=vmlaq_f32(s,vld1q_f32(a+i),vld1q_f32(b+i));float r=vaddvq_f32(s);for(;i<n;++i)r+=a[i]*b[i];return r;}
static void neon(const float*x,const float*w1,const float*b1,const float*w2,const float*b2,float*h,float*y,size_t in,size_t hidden,size_t out){size_t i;for(i=0;i<hidden;++i){float v=neon_dot(w1+i*in,x,in)+b1[i];h[i]=v>0?v:0;}for(i=0;i<out;++i)y[i]=neon_dot(w2+i*hidden,h,hidden)+b2[i];}
#endif
void lt_mlp2_f32(const float*x,const float*w1,const float*b1,const float*w2,const float*b2,float*h,float*y,size_t in,size_t hidden,size_t out){
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma")){avx(x,w1,b1,w2,b2,h,y,in,hidden,out);return;}
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
  neon(x,w1,b1,w2,b2,h,y,in,hidden,out);return;
#endif
  lt_mlp2_f32_scalar(x,w1,b1,w2,b2,h,y,in,hidden,out);
}
const char *lt_mlp2_f32_backend(void){
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma"))return "avx2-fma";
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
  return "neon";
#endif
  return "scalar";
}
