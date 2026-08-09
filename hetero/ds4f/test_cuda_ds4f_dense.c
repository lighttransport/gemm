#include "../../cuda/cuew.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
static float fp8(uint8_t v) {
    int s=v>>7,e=(v>>3)&15,m=v&7; float x;
    if(e==0)x=ldexpf((float)m,-9); else if(e==15&&m==7)x=0; else x=ldexpf(1.f+m*.125f,e-7);
    return s?-x:x;
}
int main(void) {
    const int M=64,N=128,K=256,SC=(K+127)/128;
    size_t wb=(size_t)N*K, sb=(size_t)((N+127)/128)*SC;
    uint8_t *w=malloc(wb),*s=malloc(sb); float *x=malloc((size_t)M*K*4),*y=malloc((size_t)M*N*4);
    for(size_t i=0;i<wb;i++)w[i]=(uint8_t)((i*13+17)%126);
    memset(s,127,sb); for(size_t i=0;i<(size_t)M*K;i++)x[i]=((int)(i*7%31)-15)/17.f;
    if(cuewInit(CUEW_INIT_CUDA)!=CUEW_SUCCESS||cuInit(0)!=CUDA_SUCCESS)return 2;
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn; CUstream st;
    cuDeviceGet(&dev,0); cuDevicePrimaryCtxRetain(&ctx,dev); cuCtxSetCurrent(ctx); cuStreamCreate(&st,0);
    if(cuModuleLoad(&mod,"cuda/llm/ds4f_dense_kernels.cubin")!=CUDA_SUCCESS||
       cuModuleGetFunction(&fn,mod,"ds4f_cuda_dense_gemm")!=CUDA_SUCCESS)return 3;
    CUdeviceptr dw,ds,dx,dy; cuMemAlloc(&dw,wb);cuMemAlloc(&ds,sb);cuMemAlloc(&dx,(size_t)M*K*4);cuMemAlloc(&dy,(size_t)M*N*4);
    cuMemcpyHtoD(dw,w,wb);cuMemcpyHtoD(ds,s,sb);cuMemcpyHtoD(dx,x,(size_t)M*K*4);
    int kind=0; void *args[]={&dy,&dw,&ds,&dx,(void*)&N,(void*)&K,(void*)&M,(void*)&SC,&kind};
    if(cuLaunchKernel(fn,(N+31)/32,(M+63)/64,1,256,1,1,0,st,args,NULL)!=CUDA_SUCCESS)return 4;
    cuStreamSynchronize(st);cuMemcpyDtoH(y,dy,(size_t)M*N*4);
    float ma=0,mr=0; double se=0,sr=0; int mismatch=0;
    for(int m=0;m<M;m++){int bi=0,br=0;float by=y[(size_t)m*N],bz=-INFINITY;
        for(int n=0;n<N;n++){float z=0;for(int k=0;k<K;k++)z+=x[(size_t)m*K+k]*fp8(w[(size_t)n*K+k]);float d=fabsf(z-y[(size_t)m*N+n]);if(d>ma)ma=d;if(d/(fabsf(z)+1)>mr)mr=d/(fabsf(z)+1);se+=(double)d*d;sr+=(double)z*z;if(z>bz){bz=z;br=n;}if(y[(size_t)m*N+n]>by){by=y[(size_t)m*N+n];bi=n;}}
        mismatch += bi != br;}
    double t0=now_s();for(int i=0;i<100;i++)cuLaunchKernel(fn,(N+31)/32,(M+63)/64,1,256,1,1,0,st,args,NULL);cuStreamSynchronize(st);double ms=(now_s()-t0)*10;
    double rel_l2=sqrt(se/(sr+1e-30));
    printf("CUDA DS4F compact FP8 WMMA: max_abs=%g max_rel=%g rel_l2=%g argmax=%d/%d %.3f ms/call %s\n",ma,mr,rel_l2,mismatch,M,ms,(rel_l2<3e-3&&mismatch==0?"PASS":"FAIL"));
    cuMemFree(dy);cuMemFree(dx);cuMemFree(ds);cuMemFree(dw);cuStreamDestroy(st);cuModuleUnload(mod);cuDevicePrimaryCtxRelease(dev);
    return rel_l2<3e-3&&mismatch==0?0:1;
}
