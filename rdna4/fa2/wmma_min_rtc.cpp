// HIPRTC path: compile WMMA_KSRC at runtime, run, compare vs CPU. mode: bf16|f16.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "wmma_min.h"
static const char*src=WMMA_KSRC;
static uint16_t bf(float f){union{float f;uint32_t i;}u;u.f=f;return (uint16_t)((u.i+0x8000)>>16);}
static float bfd(uint16_t v){union{uint32_t i;float f;}u;u.i=(uint32_t)v<<16;return u.f;}
int main(int c,char**v){int bf16=c>1&&!strcmp(v[1],"bf16");
  rocewInit(ROCEW_INIT_HIP|ROCEW_INIT_HIPRTC);hipSetDevice(0);
  hipModule_t m;hiprtcProgram p;hiprtcCreateProgram(&p,src,"mm",0,0,0);
  const char*o[3]={"--gpu-architecture=gfx1201","-O3",bf16?"-DT_BF16":"-ffast-math"};
  if(hiprtcCompileProgram(p,3,o)!=HIPRTC_SUCCESS){size_t s;hiprtcGetProgramLogSize(p,&s);char*l=(char*)malloc(s+1);hiprtcGetProgramLog(p,l);l[s]=0;printf("%s\n",l);return 1;}
  size_t cs;hiprtcGetCodeSize(p,&cs);char*cc=(char*)malloc(cs);hiprtcGetCode(p,cc);hipModuleLoadData(&m,cc);hipFunction_t f;hipModuleGetFunction(&f,m,"mm");
  float A[256],B[256],C[256],R[256];unsigned r=1;uint16_t hA[256],hB[256];
  for(int i=0;i<256;i++){A[i]=((int)((r=r*1103515245u+12345u)>>8&255)-128)/128.f;B[i]=((int)((r=r*1103515245u+12345u)>>8&255)-128)/128.f;hA[i]=bf16?bf(A[i]):hip_f32_to_f16(A[i]);hB[i]=bf16?bf(B[i]):hip_f32_to_f16(B[i]);if(bf16){A[i]=bfd(hA[i]);B[i]=bfd(hB[i]);}else{A[i]=A[i];}}
  for(int rr=0;rr<16;rr++)for(int cc2=0;cc2<16;cc2++){float a=0;for(int k=0;k<16;k++)a+=A[rr*16+k]*B[k*16+cc2];R[rr*16+cc2]=a;}
  void*dA=hip_upload_raw(hA,512),*dB=hip_upload_raw(hB,512),*dC;hipMalloc(&dC,1024);void*ar[]={&dC,&dA,&dB};
  hipModuleLaunchKernel(f,1,1,1,32,1,1,0,0,ar,0);hipDeviceSynchronize();hipMemcpy(C,dC,1024,hipMemcpyDeviceToHost);
  double e=0,n=0;for(int i=0;i<256;i++){e+=fabs(C[i]-R[i]);n+=fabs(R[i]);}printf("RTC %s: C[0]=%.3f ref=%.3f  L1err=%.4f\n",bf16?"bf16":"f16",C[0],R[0],e/n);return 0;}
