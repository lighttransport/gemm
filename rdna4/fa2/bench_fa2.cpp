/*
 * bench_fa2.cpp - RDNA4/gfx1201 flash attention, gau-nernst blog techniques.
 *   BF16/FP16 (190 TF/s peak) + FP8 e4m3 (350 TF/s peak); HD=128 default, HD=64 via -DHD.
 *   BQ=256, BKV=32, 16 waves. baseline ('p'), +LDS XOR swizzle ('s'), +double-buffer ('d').
 *   K/V/P transpose, FP32 online softmax (fast exp2). Standalone HIPRTC, no SDK.
 *
 * make -C rdna4/fa2
 * rdna4/fa2/bench_fa2 --n-tok 256 --heads 2 --check --mode all
 * rdna4/fa2/bench_fa2 --n-tok 4096 --heads 16 --iters 100 --mode all
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

static float f16_to_f32(uint16_t v) {
    uint32_t s = (v >> 15) & 1; int e = (v >> 10) & 0x1F; uint32_t m = v & 0x3FF;
    if (e == 0) return m ? ldexpf((float)m / 1024.0f, -14) * (s?-1:1) : (s?-0.0f:0.0f);
    if (e == 31) return m ? NAN : (s?-INFINITY:INFINITY);
    return ldexpf(1.0f + (float)m/1024.0f, e-15) * (s?-1:1);
}
static uint16_t f32_to_bf16(float f){union{float f;uint32_t i;}u;u.f=f;return (uint16_t)((u.i+0x8000)>>16);}
static float bf16d(uint16_t v){union{uint32_t i;float f;}u;u.i=(uint32_t)v<<16;return u.f;}
static float fp8d(uint8_t v){int s=(v>>7)&1,e=(v>>3)&15,m=v&7;float sg=s?-1:1;if(e==0)return sg*m/8.0f*ldexpf(1,-6);return sg*ldexpf(1+m/8.0f,e-7);}
static double timer_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
static double cosine_sim(const float*a,const float*b,size_t n){double d=0,x=0,y=0;for(size_t i=0;i<n;i++){d+=(double)a[i]*b[i];x+=(double)a[i]*a[i];y+=(double)b[i]*b[i];}return(x==0||y==0)?0:d/(sqrt(x)*sqrt(y));}
static float max_abs_diff(const float*a,const float*b,size_t n){float m=0;for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>m)m=d;}return m;}
static void fill_f16(uint16_t*d,size_t n,float a,unsigned*r){for(size_t i=0;i<n;i++){unsigned x=*r=*r*1103515245u+12345u;float u=((int)(x&0xFFFFFFu)-0x800000)/(float)0x800000;d[i]=hip_f32_to_f16(u*a);}}
static void fill_bf16(uint16_t*d,size_t n,float a,unsigned*r){for(size_t i=0;i<n;i++){unsigned x=*r=*r*1103515245u+12345u;float u=((int)(x&0xFFFFFFu)-0x800000)/(float)0x800000;d[i]=f32_to_bf16(u*a);}}
static void fill_fp8(uint8_t*d,size_t n,float a,unsigned*r){for(size_t i=0;i<n;i++){unsigned x=*r=*r*1103515245u+12345u;float u=((int)(x&0xFFFFFFu)-0x800000)/(float)0x800000*a;if(u>448)u=448;if(u<-448)u=-448;d[i]=hip_f32_to_fp8_e4m3(u);}}

static void fa_ref(float*out,const float*Q,const float*K,const float*V,int N,int H,int hd){
    int dim=H*hd; float sc=1.0f/sqrtf((float)hd); float*s=(float*)malloc(N*sizeof(float));
    for(int h=0;h<H;h++)for(int q=0;q<N;q++){float mx=-1e30f;
        for(int k=0;k<N;k++){float a=0;for(int d=0;d<hd;d++)a+=Q[(size_t)q*dim+h*hd+d]*K[((size_t)h*N+k)*hd+d];a*=sc;s[k]=a;if(a>mx)mx=a;}
        float l=0;for(int k=0;k<N;k++){s[k]=expf(s[k]-mx);l+=s[k];}float il=l>0?1/l:0;
        for(int d=0;d<hd;d++){float ac=0;for(int k=0;k<N;k++)ac+=s[k]*il*V[((size_t)h*N+k)*hd+d];out[(size_t)q*dim+h*hd+d]=ac;}}
    free(s);
}

static const char *kernel_src = R"FASRC(
typedef unsigned int u32;
typedef unsigned char u8;
typedef float float8 __attribute__((ext_vector_type(8)));
#ifndef HD
#define HD 128
#endif
#define K_NB (HD/16)
#define LOG2E 1.4426950408889634f
__device__ __forceinline__ float exp2_fast(float x){if(x<=-24.0f)return 0.0f;if(x>8.0f)x=8.0f;union{int i;float f;}u;u.i=(int)((x+127.0f)*8388608.0f);return u.f;}
/* blog XOR swizzle: spread BKV rows of HD-stride K across LDS banks, 8-elem aligned */
#ifdef SWZ
#define SW(r) (((r)&7)<<4)
#else
#define SW(r) 0
#endif
#ifdef USE_BF16
  typedef unsigned short wt; typedef wt v8 __attribute__((ext_vector_type(8)));
  #define MMA(A,B,C) C=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(A,B,C)
  __device__ __forceinline__ wt CVT(float f){union{float f;u32 i;}u;u.f=f;return (wt)(u.i>>16);}
#else
  typedef _Float16 wt; typedef wt v8 __attribute__((ext_vector_type(8)));
  #define MMA(A,B,C) C=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(A,B,C)
  __device__ __forceinline__ wt CVT(float f){return (wt)f;}
#endif
extern "C" __global__ __launch_bounds__(512,1)
void fa(float*out,const wt*Q,const wt*Kt,const wt*Vt,int N,int H,float isd){
  enum{BQ=256,BKV=32,WC=16,CH=2}; int h=blockIdx.x,qb=blockIdx.y,q0=qb*BQ,tid=threadIdx.x,wid=tid>>5,lid=tid&31,hf=lid>>4,id=lid&15,dim=H*HD;
  __shared__ wt smK[BKV*HD]; __shared__ wt smV[HD*BKV]; __shared__ wt smP[WC*CH*256]; wt*sP=smP+wid*CH*256;
  v8 qr[K_NB/4][4]; for(int g=0;g<K_NB/4;g++)for(int i=0;i<4;i++){int kb=g*4+i,r=q0+wid*16+id; v8 t; for(int j=0;j<8;j++){int d=kb*16+hf*8+j;t[j]=(r<N)?Q[(size_t)r*dim+h*HD+d]:(wt)0;}qr[g][i]=t;}
  float8 O[K_NB]; for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]=0; float mi[8],li[8]; for(int i=0;i<8;i++){mi[i]=-1e30f;li[i]=0;}
  int nt=(N+BKV-1)/BKV,tot=BKV*HD,st=WC*32*16;
  for(int t=0;t<nt;t++){int kv0=t*BKV;
    for(int o=tid*16;o<tot;o+=st){int r=o/HD,d=o%HD,kv=kv0+r;size_t b=((size_t)h*N+kv)*HD+d;if(kv<N&&d+16<=HD){v8 k0=*(const v8*)(Kt+b),k1=*(const v8*)(Kt+b+8),v0=*(const v8*)(Vt+b),v1=*(const v8*)(Vt+b+8);*(v8*)(smK+r*HD+(d^SW(r)))=k0;*(v8*)(smK+r*HD+((d+8)^SW(r)))=k1;for(int i=0;i<8;i++)smV[(d+i)*BKV+r]=v0[i];for(int i=0;i<8;i++)smV[(d+8+i)*BKV+r]=v1[i];}else for(int i=0;i<16;i++){wt kk=0,vv=0;if(kv<N&&d+i<HD){kk=Kt[b+i];vv=Vt[b+i];}smK[r*HD+((d+i)^SW(r))]=kk;smV[(d+i)*BKV+r]=vv;}}
    __syncthreads(); float8 S[CH]; for(int c=0;c<CH;c++)for(int i=0;i<8;i++)S[c][i]=0;
    for(int g=0;g<K_NB/4;g++)for(int c=0;c<CH;c++){v8 kr[4];for(int i=0;i<4;i++){int kb=g*4+i,r=c*16+id,d=kb*16+hf*8;kr[i]=*(const v8*)(smK+r*HD+(d^SW(r)));}for(int i=0;i<4;i++)MMA(qr[g][i],kr[i],S[c]);}
    float rm[8];for(int i=0;i<8;i++){float mx=-1e30f;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float s=S[c][i]*isd*LOG2E;if(kv>=N)s=-1e30f;S[c][i]=s;mx=fmaxf(mx,s);}mx=fmaxf(mx,__shfl_xor(mx,1,32));mx=fmaxf(mx,__shfl_xor(mx,2,32));mx=fmaxf(mx,__shfl_xor(mx,4,32));mx=fmaxf(mx,__shfl_xor(mx,8,32));rm[i]=mx;}
    float al[8];for(int i=0;i<8;i++){float nm=fmaxf(mi[i],rm[i]);al[i]=exp2_fast(mi[i]-nm);float lo=0;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float p=exp2_fast(S[c][i]-nm+8.0f);if(kv>=N)p=0;S[c][i]=p;lo+=p;}li[i]=li[i]*al[i]+lo;mi[i]=nm;}
    for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]*=al[i];
    for(int c=0;c<CH;c++){wt*pc=sP+c*256;for(int i=0;i<8;i++)pc[(hf*8+i)*16+id]=CVT(S[c][i]);}__builtin_amdgcn_s_waitcnt(0);__syncthreads();
    for(int c=0;c<CH;c++){wt*pc=sP+c*256;v8 ap=*(const v8*)(pc+id*16+hf*8);for(int kb=0;kb<K_NB;kb++){int d=kb*16+id;MMA(ap,*(const v8*)(smV+d*BKV+c*16+hf*8),O[kb]);}}
  }
  for(int i=0;i<8;i++){float l=li[i];l+=__shfl_xor(l,1,32);l+=__shfl_xor(l,2,32);l+=__shfl_xor(l,4,32);l+=__shfl_xor(l,8,32);li[i]=l>0?1/l:0;}
  for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++){int r=q0+wid*16+hf*8+i,d=kb*16+id;if(r<N)out[(size_t)r*dim+h*HD+d]=O[kb][i]*li[i];}
}
/* double-buffered: prefetch tile t+1 into alt buffer while computing t */
#define LD(KK,VV,KV0) for(int o=tid*16;o<tot;o+=st){int r=o/HD,d=o%HD,kv=(KV0)+r;size_t b=((size_t)h*N+kv)*HD+d;if(kv<N&&d+16<=HD){v8 k0=*(const v8*)(Kt+b),k1=*(const v8*)(Kt+b+8),v0=*(const v8*)(Vt+b),v1=*(const v8*)(Vt+b+8);*(v8*)((KK)+r*HD+(d^SW(r)))=k0;*(v8*)((KK)+r*HD+((d+8)^SW(r)))=k1;for(int i=0;i<8;i++)(VV)[(d+i)*BKV+r]=v0[i];for(int i=0;i<8;i++)(VV)[(d+8+i)*BKV+r]=v1[i];}else for(int i=0;i<16;i++){wt kk=0,vv=0;if(kv<N&&d+i<HD){kk=Kt[b+i];vv=Vt[b+i];}(KK)[r*HD+((d+i)^SW(r))]=kk;(VV)[(d+i)*BKV+r]=vv;}}
extern "C" __global__ __launch_bounds__(512,1)
void fa_db(float*out,const wt*Q,const wt*Kt,const wt*Vt,int N,int H,float isd){
  enum{BQ=256,BKV=32,WC=16,CH=2}; int h=blockIdx.x,qb=blockIdx.y,q0=qb*BQ,tid=threadIdx.x,wid=tid>>5,lid=tid&31,hf=lid>>4,id=lid&15,dim=H*HD;
  __shared__ wt smK[2][BKV*HD]; __shared__ wt smV[2][HD*BKV]; __shared__ wt smP[WC*CH*256]; wt*sP=smP+wid*CH*256;
  v8 qr[K_NB/4][4]; for(int g=0;g<K_NB/4;g++)for(int i=0;i<4;i++){int kb=g*4+i,r=q0+wid*16+id; v8 t; for(int j=0;j<8;j++){int d=kb*16+hf*8+j;t[j]=(r<N)?Q[(size_t)r*dim+h*HD+d]:(wt)0;}qr[g][i]=t;}
  float8 O[K_NB]; for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]=0; float mi[8],li[8]; for(int i=0;i<8;i++){mi[i]=-1e30f;li[i]=0;}
  int nt=(N+BKV-1)/BKV,tot=BKV*HD,st=WC*32*16; LD(smK[0],smV[0],0); __syncthreads();
  for(int t=0;t<nt;t++){int kv0=t*BKV,cur=t&1,nx=cur^1; wt*cK=smK[cur],*cV=smV[cur];
    if(t+1<nt){LD(smK[nx],smV[nx],(t+1)*BKV);}
    float8 S[CH]; for(int c=0;c<CH;c++)for(int i=0;i<8;i++)S[c][i]=0;
    for(int g=0;g<K_NB/4;g++)for(int c=0;c<CH;c++){v8 kr[4];for(int i=0;i<4;i++){int kb=g*4+i,r=c*16+id,d=kb*16+hf*8;kr[i]=*(const v8*)(cK+r*HD+(d^SW(r)));}for(int i=0;i<4;i++)MMA(qr[g][i],kr[i],S[c]);}
    float rm[8];for(int i=0;i<8;i++){float mx=-1e30f;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float s=S[c][i]*isd*LOG2E;if(kv>=N)s=-1e30f;S[c][i]=s;mx=fmaxf(mx,s);}mx=fmaxf(mx,__shfl_xor(mx,1,32));mx=fmaxf(mx,__shfl_xor(mx,2,32));mx=fmaxf(mx,__shfl_xor(mx,4,32));mx=fmaxf(mx,__shfl_xor(mx,8,32));rm[i]=mx;}
    float al[8];for(int i=0;i<8;i++){float nm=fmaxf(mi[i],rm[i]);al[i]=exp2_fast(mi[i]-nm);float lo=0;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float p=exp2_fast(S[c][i]-nm+8.0f);if(kv>=N)p=0;S[c][i]=p;lo+=p;}li[i]=li[i]*al[i]+lo;mi[i]=nm;}
    for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]*=al[i];
    for(int c=0;c<CH;c++){wt*pc=sP+c*256;for(int i=0;i<8;i++)pc[(hf*8+i)*16+id]=CVT(S[c][i]);}__builtin_amdgcn_s_waitcnt(0);__syncthreads();
    for(int c=0;c<CH;c++){wt*pc=sP+c*256;v8 ap=*(const v8*)(pc+id*16+hf*8);for(int kb=0;kb<K_NB;kb++){int d=kb*16+id;MMA(ap,*(const v8*)(cV+d*BKV+c*16+hf*8),O[kb]);}}
    __syncthreads();
  }
  for(int i=0;i<8;i++){float l=li[i];l+=__shfl_xor(l,1,32);l+=__shfl_xor(l,2,32);l+=__shfl_xor(l,4,32);l+=__shfl_xor(l,8,32);li[i]=l>0?1/l:0;}
  for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++){int r=q0+wid*16+hf*8+i,d=kb*16+id;if(r<N)out[(size_t)r*dim+h*HD+d]=O[kb][i]*li[i];}
}
)FASRC";

static const char *fp8_src = R"FP8(
typedef unsigned int u32; typedef unsigned char u8;
typedef float float8 __attribute__((ext_vector_type(8)));
typedef u32 i2 __attribute__((ext_vector_type(2)));
#ifndef HD
#define HD 128
#endif
#define K_NB (HD/16)
#define LOG2E 1.4426950408889634f
#define WMF(A,B,C) C=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(A,B,C)
__device__ __forceinline__ float exp2_fast(float x){if(x<=-24.0f)return 0;if(x>8)x=8;union{int i;float f;}u;u.i=(int)((x+127.0f)*8388608.0f);return u.f;}
__device__ __forceinline__ u8 q8(float f){if(f<=0)return 0;union{float f;u32 i;}u;u.f=f;int e=(int)((u.i>>23)&0xff)-120;u32 m=(u.i>>20)&7;if(e>15||(e==15&&m>6))return 0x7e;if(e<=0)return 0;return(u8)(((e&15)<<3)|m);}
extern "C" __global__ __launch_bounds__(512,1)
void fa(float*out,const u8*Q,const u8*Kt,const u8*Vt,int N,int H,float isd){
  enum{BQ=256,BKV=32,WC=16,CH=2}; int h=blockIdx.x,qb=blockIdx.y,q0=qb*BQ,tid=threadIdx.x,wid=tid>>5,lid=tid&31,hf=lid>>4,id=lid&15,dim=H*HD;
  __shared__ u8 smK[BKV*HD]; __shared__ u8 smV[HD*BKV]; __shared__ u8 smP[WC*CH*256]; u8*sP=smP+wid*CH*256;
  float8 O[K_NB]; for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]=0; float mi[8],li[8]; for(int i=0;i<8;i++){mi[i]=-1e30f;li[i]=0;}
  int nt=(N+BKV-1)/BKV,tot=BKV*HD,st=WC*32*16;
  for(int t=0;t<nt;t++){int kv0=t*BKV;for(int o=tid*16;o<tot;o+=st){int r=o/HD,d=o%HD,kv=kv0+r;size_t b=((size_t)h*N+kv)*HD+d;for(int i=0;i<16;i++){u8 kk=0,vv=0;if(kv<N&&d+i<HD){kk=Kt[b+i];vv=Vt[b+i];}smK[r*HD+d+i]=kk;smV[(d+i)*BKV+r]=vv;}}__syncthreads();
    float8 S[CH];for(int c=0;c<CH;c++)for(int i=0;i<8;i++)S[c][i]=0;
    for(int kb=0;kb<K_NB;kb++){i2 a,bb;u8 *pa=(u8*)&a,*pb=(u8*)&bb;int r=q0+wid*16+id,off=kb*16+hf*8;for(int j=0;j<8;j++){pa[j]=(r<N)?Q[(size_t)r*dim+h*HD+off+j]:0;}for(int c=0;c<CH;c++){int kr=c*16+id;for(int j=0;j<8;j++)pb[j]=smK[kr*HD+off+j];WMF(a,bb,S[c]);}}
    float rm[8];for(int i=0;i<8;i++){float mx=-1e30f;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float s=S[c][i]*isd*LOG2E;if(kv>=N)s=-1e30f;S[c][i]=s;mx=fmaxf(mx,s);}mx=fmaxf(mx,__shfl_xor(mx,1,32));mx=fmaxf(mx,__shfl_xor(mx,2,32));mx=fmaxf(mx,__shfl_xor(mx,4,32));mx=fmaxf(mx,__shfl_xor(mx,8,32));rm[i]=mx;}
    float al[8];for(int i=0;i<8;i++){float nm=fmaxf(mi[i],rm[i]);al[i]=exp2_fast(mi[i]-nm);float lo=0;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float p=exp2_fast(S[c][i]-nm);if(kv>=N)p=0;S[c][i]=p;lo+=p;}li[i]=li[i]*al[i]+lo;mi[i]=nm;}
    for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]*=al[i];
    for(int c=0;c<CH;c++){u8*pc=sP+c*256;for(int i=0;i<8;i++)pc[(hf*8+i)*16+id]=q8(S[c][i]);}__builtin_amdgcn_s_waitcnt(0);__syncthreads();
    for(int c=0;c<CH;c++){u8*pc=sP+c*256;i2 ap=*(const i2*)(pc+id*16+hf*8);for(int kb=0;kb<K_NB;kb++){int d=kb*16+id;WMF(ap,*(const i2*)(smV+d*BKV+c*16+hf*8),O[kb]);}}
  }
  for(int i=0;i<8;i++){float l=li[i];l+=__shfl_xor(l,1,32);l+=__shfl_xor(l,2,32);l+=__shfl_xor(l,4,32);l+=__shfl_xor(l,8,32);li[i]=l>0?1/l:0;}
  for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++){int r=q0+wid*16+hf*8+i,d=kb*16+id;if(r<N)out[(size_t)r*dim+h*HD+d]=O[kb][i]*li[i];}
}
)FP8";

typedef struct{const char*name;hipFunction_t f;int dt;}kern;/*dt:0 16-bit,1 fp8*/
static int compile(hipModule_t*m,int dev,const char*src,const char*nm,int bf16,int swz,int hd){
  const char*a=rocewGetRDNA4ArchString(dev);hipDeviceProp_t p;if(!a){hipGetDeviceProperties(&p,dev);a=p.gcnArchName;}
  hiprtcProgram pr;if(hiprtcCreateProgram(&pr,src,nm,0,0,0)!=HIPRTC_SUCCESS)return -1;char af[64];snprintf(af,64,"--gpu-architecture=%s",a);char hf[24];snprintf(hf,24,"-DHD=%d",hd);
  /* bf16 builtin miscompiles under -ffast-math (breaks exp2 bit-cast) -> -O2, no ffast-math */
  const char*o[5]={af,bf16?"-O2":"-O3",bf16?"-DUSE_BF16":"-ffast-math",swz?"-DSWZ":hf,hf};
  if(hiprtcCompileProgram(pr,5,o)!=HIPRTC_SUCCESS){size_t s;hiprtcGetProgramLogSize(pr,&s);char*l=(char*)malloc(s+1);hiprtcGetProgramLog(pr,l);l[s]=0;fprintf(stderr,"%s:%s\n",nm,l);free(l);return -1;}
  size_t cs;hiprtcGetCodeSize(pr,&cs);char*c=(char*)malloc(cs);hiprtcGetCode(pr,c);hiprtcDestroyProgram(&pr);hipError_t e=hipModuleLoadData(m,c);free(c);return e==hipSuccess?1:-1;}
static int run(kern*k,int N,int H,int it,int chk,float am,double pk,int hd){
  if(N%64){fprintf(stderr,"skip N%%64\n");return 0;}int dim=H*hd;size_t ne=(size_t)N*dim,qb=ne*(k->dt==1?1:2),ob=ne*4;
  uint16_t*hQ=(uint16_t*)malloc(ne*2),*hK=(uint16_t*)malloc(ne*2),*hV=(uint16_t*)malloc(ne*2);uint8_t*qQ=(uint8_t*)malloc(ne),*qK=(uint8_t*)malloc(ne),*qV=(uint8_t*)malloc(ne);
  unsigned r=0x1234abcd;if(k->dt==2){fill_bf16(hQ,ne,am,&r);fill_bf16(hK,ne,am,&r);fill_bf16(hV,ne,am,&r);}else{fill_f16(hQ,ne,am,&r);fill_f16(hK,ne,am,&r);fill_f16(hV,ne,am,&r);}fill_fp8(qQ,ne,am,&r);fill_fp8(qK,ne,am,&r);fill_fp8(qV,ne,am,&r);
  void*dQ=hip_upload_raw(k->dt?(void*)qQ:(void*)hQ,qb),*dK=hip_upload_raw(k->dt?(void*)qK:(void*)hK,qb),*dV=hip_upload_raw(k->dt?(void*)qV:(void*)hV,qb),*dO=0;hipMalloc(&dO,ob);hipMemset(dO,0,ob);
  float isd=1.0f/sqrtf((float)hd);dim3 bl(512,1,1),gr((unsigned)H,(unsigned)((N+255)/256),1);void*ar[]={&dO,&dQ,&dK,&dV,&N,&H,&isd};
  hipError_t le=hipModuleLaunchKernel(k->f,gr.x,gr.y,gr.z,bl.x,bl.y,bl.z,0,0,ar,0);hipError_t se=hipDeviceSynchronize();if(le||se)printf("  [%-5s] launch=%d sync=%d\n",k->name,le,se);
  double cs=-2,md=-1;if(chk){float*hO=(float*)malloc(ob),*rf=(float*)malloc(ob);hipMemcpy(hO,dO,ob,hipMemcpyDeviceToHost);float*fQ=(float*)malloc(ne*4),*fK=(float*)malloc(ne*4),*fV=(float*)malloc(ne*4);for(size_t i=0;i<ne;i++){fQ[i]=k->dt==1?fp8d(qQ[i]):k->dt==2?bf16d(hQ[i]):f16_to_f32(hQ[i]);fK[i]=k->dt==1?fp8d(qK[i]):k->dt==2?bf16d(hK[i]):f16_to_f32(hK[i]);fV[i]=k->dt==1?fp8d(qV[i]):k->dt==2?bf16d(hV[i]):f16_to_f32(hV[i]);}fa_ref(rf,fQ,fK,fV,N,H,hd);cs=cosine_sim(hO,rf,ne);md=max_abs_diff(hO,rf,ne);free(hO);free(rf);free(fQ);free(fK);free(fV);}
  double t0=timer_ms();for(int i=0;i<it;i++)hipModuleLaunchKernel(k->f,gr.x,gr.y,gr.z,bl.x,bl.y,bl.z,0,0,ar,0);hipDeviceSynchronize();double ms=(timer_ms()-t0)/it;double tf=4.0*H*N*N*hd/(ms*1e-3)*1e-12;
  printf("  [%-5s] S=%5d H=%2d %8.4f ms %7.2f TF/s %5.1f%%",k->name,N,H,ms,tf,pk>0?100*tf/pk:0);if(chk)printf(" cos=%.6f maxd=%.4f",cs,md);printf("\n");
  hipFree(dO);hipFree(dQ);hipFree(dK);hipFree(dV);free(hQ);free(hK);free(hV);free(qQ);free(qK);free(qV);return 0;}
int main(int c,char**v){int N=1024,H=16,it=100,ck=0,hd=128;float am=1;double pk=190;const char*md="all";for(int i=1;i<c;i++){if(!strcmp(v[i],"--n-tok"))N=atoi(v[++i]);else if(!strcmp(v[i],"--heads"))H=atoi(v[++i]);else if(!strcmp(v[i],"--dim"))hd=atoi(v[++i]);else if(!strcmp(v[i],"--iters"))it=atoi(v[++i]);else if(!strcmp(v[i],"--mode"))md=v[++i];else if(!strcmp(v[i],"--check"))ck=1;else if(!strcmp(v[i],"--abs-max"))am=atof(v[++i]);else if(!strcmp(v[i],"--peak"))pk=atof(v[++i]);}
  rocewInit(ROCEW_INIT_HIP|ROCEW_INIT_HIPRTC);hipSetDevice(0);hipModule_t mf,mb,ms,mfp;compile(&mf,0,kernel_src,"f16",0,0,hd);compile(&mb,0,kernel_src,"b16",1,0,hd);compile(&ms,0,kernel_src,"b16s",1,1,hd);compile(&mfp,0,fp8_src,"fp8",0,0,hd);
  kern ks[5];memset(ks,0,sizeof ks);int n=0;hipModuleGetFunction(&ks[n].f,mb,"fa");ks[n].name="b16";ks[n++].dt=2;hipModuleGetFunction(&ks[n].f,ms,"fa");ks[n].name="b16s";ks[n++].dt=2;hipModuleGetFunction(&ks[n].f,ms,"fa_db");ks[n].name="b16d";ks[n++].dt=2;hipModuleGetFunction(&ks[n].f,mf,"fa");ks[n].name="f16";ks[n++].dt=0;hipModuleGetFunction(&ks[n].f,mfp,"fa");ks[n].name="fp8";ks[n++].dt=1;
  printf("fa2 (iters=%d hd=%d peak=%.0f%s)\n",it,hd,pk,ck?" check":"");for(int i=0;i<n;i++)if(ks[i].f&&(!strcmp(md,"all")||!strcmp(md,ks[i].name)))run(&ks[i],N,H,it,ck,am,ks[i].dt==1?350:pk,hd);return 0;}
