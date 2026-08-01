// Stage 2: 1 BQ=16 x BKV=16 tile, hd=128 (8 K-MMA accumulate) via K-LDS,
// + uniform "softmax" -> CVT P-store -> V-MMA. Reproduces fa2's bf16 ops minus loop.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
static const char*src=
"typedef unsigned int u32;\n"
"typedef float f8 __attribute__((ext_vector_type(8)));\n"
"#ifdef T_BF16\n typedef unsigned short wt; typedef wt v8 __attribute__((ext_vector_type(8)));\n"
" #define MMA(a,b,c) c=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c)\n"
" __device__ wt CVT(float f){union{float f;u32 i;}u;u.f=f;return (wt)((u.i+0x8000u)>>16);}\n"
"#else\n typedef _Float16 wt; typedef wt v8 __attribute__((ext_vector_type(8)));\n"
" #define MMA(a,b,c) c=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c)\n __device__ wt CVT(float f){return (wt)f;}\n#endif\n"
"extern \"C\" __global__ void mm(float*O,const wt*Q,const wt*K,const wt*V){int l=threadIdx.x,hf=l>>4,id=l&15;\n"
" __shared__ wt sK[16*128]; __shared__ wt sV[128*16]; __shared__ wt sP[256];\n"
" for(int o=l;o<2048;o+=32){sK[o]=K[o];sV[o]=V[o];} __syncthreads();\n"
" f8 S; for(int i=0;i<8;i++)S[i]=0; v8 qr,kr;\n"
" for(int kb=0;kb<8;kb++){for(int i=0;i<8;i++){qr[i]=Q[id*128+kb*16+hf*8+i];kr[i]=sK[id*128+kb*16+hf*8+i];}MMA(qr,kr,S);}\n"
" for(int i=0;i<8;i++)sP[(hf*8+i)*16+id]=CVT(S[i]); __syncthreads();\n"
" v8 ap; for(int i=0;i<8;i++)ap[i]=sP[id*16+hf*8+i]; f8 Oc; for(int i=0;i<8;i++)Oc[i]=0;\n"
" v8 vb; for(int i=0;i<8;i++)vb[i]=sV[(id)*16+hf*8+i]; MMA(ap,vb,Oc);\n"
" for(int i=0;i<8;i++)O[(hf*8+i)*16+id]=Oc[i];}\n";
static uint16_t bf(float f){union{float f;uint32_t i;}u;u.f=f;return (uint16_t)((u.i+0x8000)>>16);}
static float bfd(uint16_t v){union{uint32_t i;float f;}u;u.i=(uint32_t)v<<16;return u.f;}
int main(int c,char**v){int b16=c>1&&!strcmp(v[1],"bf16");rocewInit(ROCEW_INIT_HIP|ROCEW_INIT_HIPRTC);hipSetDevice(0);
  hipModule_t m;hiprtcProgram p;hiprtcCreateProgram(&p,src,"mm",0,0,0);const char*o[3]={"--gpu-architecture=gfx1201","-O3",b16?"-DT_BF16":"-ffast-math"};
  if(hiprtcCompileProgram(p,3,o)!=HIPRTC_SUCCESS){size_t s;hiprtcGetProgramLogSize(p,&s);char*l=(char*)malloc(s+1);hiprtcGetProgramLog(p,l);l[s]=0;printf("%s\n",l);return 1;}
  size_t cs;hiprtcGetCodeSize(p,&cs);char*cc=(char*)malloc(cs);hiprtcGetCode(p,cc);hipModule_t mm2;hipModuleLoadData(&mm2,cc);hipFunction_t f;hipModuleGetFunction(&f,mm2,"mm");
  float Q[16*128],K[16*128],V[128*16],O[256];uint16_t hQ[16*128],hK[16*128],hV[128*16];unsigned r=1;
  for(int i=0;i<2048;i++){Q[i]=((int)((r=r*1103515245u+12345u)>>8&255)-128)/256.f;K[i]=((int)((r=r*1103515245u+12345u)>>8&255)-128)/256.f;V[i]=((int)((r=r*1103515245u+12345u)>>8&255)-128)/256.f;hQ[i]=b16?bf(Q[i]):hip_f32_to_f16(Q[i]);hK[i]=b16?bf(K[i]):hip_f32_to_f16(K[i]);hV[i]=b16?bf(V[i]):hip_f32_to_f16(V[i]);if(b16){Q[i]=bfd(hQ[i]);K[i]=bfd(hK[i]);V[i]=bfd(hV[i]);}}
  float S[256],R[256];for(int q=0;q<16;q++)for(int k=0;k<16;k++){float a=0;for(int d=0;d<128;d++)a+=Q[q*128+d]*K[k*128+d];S[q*16+k]=a;}
  for(int q=0;q<16;q++)for(int d=0;d<16;d++){float a=0;for(int k=0;k<16;k++)a+=S[q*16+k]*V[k*16+d];R[q*16+d]=a;}
  void*dQ=hip_upload_raw(hQ,4096),*dK=hip_upload_raw(hK,4096),*dV=hip_upload_raw(hV,4096),*dO;hipMalloc(&dO,1024);void*ar[]={&dO,&dQ,&dK,&dV};
  hipModuleLaunchKernel(f,1,1,1,32,1,1,0,0,ar,0);hipDeviceSynchronize();hipMemcpy(O,dO,1024,hipMemcpyDeviceToHost);
  double e=0,n=0;for(int i=0;i<256;i++){e+=fabs(O[i]-R[i]);n+=fabs(R[i]);}printf("MID %s: O[0]=%.3f ref=%.3f L1err=%.4f\n",b16?"bf16":"f16",O[0],R[0],e/n);return 0;}
