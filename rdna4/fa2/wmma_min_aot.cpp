// AOT path: same WMMA kernel compiled by amdclang++ (no HIPRTC). mode: -DT_BF16 or f16.
#include <cstring>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
typedef float f8 __attribute__((ext_vector_type(8)));
#ifdef T_BF16
typedef unsigned short wt; typedef wt v8 __attribute__((ext_vector_type(8)));
#define MMA(a,b,c) c=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c)
#define NAME "bf16"
#else
typedef _Float16 wt; typedef wt v8 __attribute__((ext_vector_type(8)));
#define MMA(a,b,c) c=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c)
#define NAME "f16"
#endif
__global__ void mm(float*C,const wt*A,const wt*B){int l=threadIdx.x,half=l>>4,id=l&15;v8 a,b;f8 c;for(int i=0;i<8;i++){a[i]=A[id*16+half*8+i];b[i]=B[(half*8+i)*16+id];c[i]=0;}MMA(a,b,c);for(int i=0;i<8;i++)C[(half*8+i)*16+id]=c[i];}
static uint16_t bf(float f){union{float f;uint32_t i;}u;u.f=f;return(uint16_t)((u.i+0x8000)>>16);}static float bfd(uint16_t v){union{uint32_t i;float f;}u;u.i=(uint32_t)v<<16;return u.f;}
int main(){float A[256],B[256],C[256],R[256];unsigned r=1;uint16_t hA[256],hB[256];int b16=!strcmp(NAME,"bf16");
  for(int i=0;i<256;i++){A[i]=((int)((r=r*1103515245u+12345u)>>8&255)-128)/128.f;B[i]=((int)((r=r*1103515245u+12345u)>>8&255)-128)/128.f;{_Float16 ha=(_Float16)A[i],hb=(_Float16)B[i];uint16_t fa,fb;memcpy(&fa,&ha,2);memcpy(&fb,&hb,2);hA[i]=b16?bf(A[i]):fa;hB[i]=b16?bf(B[i]):fb;}if(b16){A[i]=bfd(hA[i]);B[i]=bfd(hB[i]);}}
  for(int rr=0;rr<16;rr++)for(int cc=0;cc<16;cc++){float a=0;for(int k=0;k<16;k++)a+=A[rr*16+k]*B[k*16+cc];R[rr*16+cc]=a;}
  void*dA,*dB,*dC;hipMalloc(&dA,512);hipMalloc(&dB,512);hipMalloc(&dC,1024);hipMemcpy(dA,hA,512,hipMemcpyHostToDevice);hipMemcpy(dB,hB,512,hipMemcpyHostToDevice);
  mm<<<1,32>>>((float*)dC,(wt*)dA,(wt*)dB);hipDeviceSynchronize();hipMemcpy(C,dC,1024,hipMemcpyDeviceToHost);
  double e=0,n=0;for(int i=0;i<256;i++){e+=fabs(C[i]-R[i]);n+=fabs(R[i]);}printf("AOT %s: C[0]=%.3f ref=%.3f  L1err=%.4f\n",NAME,C[0],R[0],e/n);}
