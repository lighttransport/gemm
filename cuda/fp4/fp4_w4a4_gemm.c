/* Tiled NVFP4 W4A4 GEMM on sm_120a, built on the validated block-scale mma
 * (layout cracked in fp4_gemm_test.c). Computes
 *   D[m,n] = sum_k e2m1(A[m,k])*sA[m,k/16] * e2m1(B[n,k])*sB[n,k/16]
 * (A row-major MxK, B "col" NxK == A@B^T) and validates vs a CPU reference.
 *
 * One warp per 16x8 output tile; K-loop in steps of 64; A/B/scales read
 * straight from global (correctness-first; shared-mem staging is a later
 * optimization). M%16==0, N%8==0, K%64==0.
 *
 * Build: gcc -O2 -I.. -o fp4_w4a4_gemm fp4_w4a4_gemm.c ../cuew.c -ldl -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuew.h"

#define CHECK_CUDA(call) do{CUresult e=(call);if(e!=CUDA_SUCCESS){const char*s; \
    cuGetErrorString(e,&s);fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,s);exit(1);}}while(0)

static const float E2M1[16]={0,0.5f,1,1.5f,2,3,4,6,-0.f,-0.5f,-1,-1.5f,-2,-3,-4,-6};
static float ue4m3_decode(unsigned char b){int e=(b>>3)&0xF,m=b&0x7;
    if(e==0)return ldexpf((float)m/8.0f,1-7); return ldexpf(1.0f+(float)m/8.0f,e-7);}

static const char *KSRC =
"extern \"C\" __global__ void w4a4_gemm(const unsigned int* A, const unsigned int* B,\n"
"     const unsigned char* sA, const unsigned char* sB, float* D, int M, int N, int K){\n"
"  int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int lane=threadIdx.x&31;\n"
"  int ntn=N>>3; int tm=warp/ntn, tn=warp%ntn; int M0=tm*16, N0=tn*8;\n"
"  if(M0>=M) return;\n"
"  int g=lane>>2, t=lane&3; int Ku=K>>3, Kg=K>>4;\n"
"  int rA0=(M0+g)*Ku, rA1=(M0+g+8)*Ku, cB=(N0+g)*Ku;\n"
"  int saRow=(t==0)?(M0+g):(t==1)?(M0+g+8):-1;\n"
"  int sbCol=(t==0)?(N0+g):-1;\n"
"  float d0=0.f,d1=0.f,d2=0.f,d3=0.f;\n"
"  int nkc=K>>6;\n"
"  for(int kc=0;kc<nkc;kc++){\n"
"    unsigned int a0=A[rA0+kc*8+t], a1=A[rA1+kc*8+t], a2=A[rA0+kc*8+t+4], a3=A[rA1+kc*8+t+4];\n"
"    unsigned int b0=B[cB+kc*8+t], b1=B[cB+kc*8+t+4];\n"
"    unsigned int sfa=0x38383838u, sfb=0x38383838u;\n"
"    if(saRow>=0){const unsigned char*p=&sA[saRow*Kg+kc*4]; sfa=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"    if(sbCol>=0){const unsigned char*p=&sB[sbCol*Kg+kc*4]; sfb=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"    asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
"      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
"      : \"+f\"(d0),\"+f\"(d1),\"+f\"(d2),\"+f\"(d3)\n"
"      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1),\"r\"(sfa),\"r\"(sfb));\n"
"  }\n"
"  D[(M0+g)*N+N0+2*t]=d0; D[(M0+g)*N+N0+2*t+1]=d1;\n"
"  D[(M0+g+8)*N+N0+2*t]=d2; D[(M0+g+8)*N+N0+2*t+1]=d3;\n"
"}\n";

static CUmodule compile_sm120a(const char*src){
    nvrtcProgram p; if(nvrtcCreateProgram(&p,src,"k",0,NULL,NULL)!=NVRTC_SUCCESS)return NULL;
    const char*o[]={"--gpu-architecture=sm_120a"};
    if(nvrtcCompileProgram(p,1,o)!=NVRTC_SUCCESS){size_t l=0;nvrtcGetProgramLogSize(p,&l);
        char*b=malloc(l+1);nvrtcGetProgramLog(p,b);b[l]=0;fprintf(stderr,"REJECTED:\n%s\n",b);free(b);return NULL;}
    CUmodule m=NULL; size_t bs=0;
    if(nvrtcGetCUBINSize&&nvrtcGetCUBINSize(p,&bs)==NVRTC_SUCCESS&&bs>0){char*bl=malloc(bs);
        nvrtcGetCUBIN(p,bl);nvrtcDestroyProgram(&p);if(cuModuleLoadData(&m,bl)!=CUDA_SUCCESS)m=NULL;free(bl);}
    else{size_t ps=0;nvrtcGetPTXSize(p,&ps);char*x=malloc(ps);nvrtcGetPTX(p,x);nvrtcDestroyProgram(&p);
        if(cuModuleLoadData(&m,x)!=CUDA_SUCCESS)m=NULL;free(x);}
    return m;
}

int main(int argc,char**argv){
    int M=256,N=512,K=512;
    for(int i=1;i+1<argc;i++){if(!strcmp(argv[i],"-m"))M=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-n"))N=atoi(argv[++i]);else if(!strcmp(argv[i],"-k"))K=atoi(argv[++i]);}
    if(M%16||N%8||K%64){fprintf(stderr,"need M%%16,N%%8,K%%64\n");return 1;}
    if(cuewInit(CUEW_INIT_CUDA|CUEW_INIT_NVRTC)!=CUEW_SUCCESS){fprintf(stderr,"cuewInit\n");return 1;}
    CHECK_CUDA(cuInit(0)); CUdevice dev;CUcontext ctx;
    CHECK_CUDA(cuDeviceGet(&dev,0)); CHECK_CUDA(cuCtxCreate(&ctx,0,dev));
    CUmodule mod=compile_sm120a(KSRC); if(!mod){fprintf(stderr,"compile fail\n");return 1;}
    CUfunction fn; CHECK_CUDA(cuModuleGetFunction(&fn,mod,"w4a4_gemm"));

    int Ku=K/8, Kg=K/16;
    unsigned char *Ac=malloc((size_t)M*K), *Bc=malloc((size_t)N*K);
    unsigned char *sA=malloc((size_t)M*Kg), *sB=malloc((size_t)N*Kg);
    srand(7);
    for(size_t i=0;i<(size_t)M*K;i++)Ac[i]=rand()&0xF;
    for(size_t i=0;i<(size_t)N*K;i++)Bc[i]=rand()&0xF;
    for(size_t i=0;i<(size_t)M*Kg;i++)sA[i]=0x30+(rand()%16);  /* ue4m3 ~[.5,2) */
    for(size_t i=0;i<(size_t)N*Kg;i++)sB[i]=0x30+(rand()%16);
    unsigned int *Au=calloc((size_t)M*Ku,4), *Bu=calloc((size_t)N*Ku,4);
    for(int m=0;m<M;m++)for(int u=0;u<Ku;u++){unsigned v=0;for(int i=0;i<8;i++)v|=((unsigned)Ac[m*K+u*8+i]&0xF)<<(i*4);Au[m*Ku+u]=v;}
    for(int n=0;n<N;n++)for(int u=0;u<Ku;u++){unsigned v=0;for(int i=0;i<8;i++)v|=((unsigned)Bc[n*K+u*8+i]&0xF)<<(i*4);Bu[n*Ku+u]=v;}

    CUdeviceptr dA,dB,dsA,dsB,dD;
    CHECK_CUDA(cuMemAlloc(&dA,(size_t)M*Ku*4)); CHECK_CUDA(cuMemAlloc(&dB,(size_t)N*Ku*4));
    CHECK_CUDA(cuMemAlloc(&dsA,(size_t)M*Kg)); CHECK_CUDA(cuMemAlloc(&dsB,(size_t)N*Kg));
    CHECK_CUDA(cuMemAlloc(&dD,(size_t)M*N*4));
    CHECK_CUDA(cuMemcpyHtoD(dA,Au,(size_t)M*Ku*4)); CHECK_CUDA(cuMemcpyHtoD(dB,Bu,(size_t)N*Ku*4));
    CHECK_CUDA(cuMemcpyHtoD(dsA,sA,(size_t)M*Kg)); CHECK_CUDA(cuMemcpyHtoD(dsB,sB,(size_t)N*Kg));

    int nwarps=(M/16)*(N/8), threads=128, blocks=(nwarps*32+threads-1)/threads;
    void*args[]={&dA,&dB,&dsA,&dsB,&dD,&M,&N,&K};
    /* timing */
    CUevent e0,e1; cuEventCreate(&e0,0); cuEventCreate(&e1,0);
    CHECK_CUDA(cuLaunchKernel(fn,blocks,1,1,threads,1,1,0,0,args,NULL)); /* warm */
    CHECK_CUDA(cuCtxSynchronize());
    cuEventRecord(e0,0);
    int iters=50; for(int it=0;it<iters;it++) CHECK_CUDA(cuLaunchKernel(fn,blocks,1,1,threads,1,1,0,0,args,NULL));
    cuEventRecord(e1,0); CHECK_CUDA(cuCtxSynchronize());
    float ms=0; cuEventElapsedTime(&ms,e0,e1); ms/=iters;
    double tops=2.0*M*N*K/(ms*1e-3)/1e12;

    float *D=malloc((size_t)M*N*4); CHECK_CUDA(cuMemcpyDtoH(D,dD,(size_t)M*N*4));
    /* CPU ref */
    double num=0,den=0; int nbad=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double r=0;
        for(int k=0;k<K;k++) r+=(double)E2M1[Ac[m*K+k]]*ue4m3_decode(sA[m*Kg+k/16])
                                 *E2M1[Bc[n*K+k]]*ue4m3_decode(sB[n*Kg+k/16]);
        double d=D[m*N+n]-r; num+=d*d; den+=r*r;
        if(fabs(d)>1e-3*(1+fabs(r))&&nbad<6){printf("  mism[%d,%d] got %.4f ref %.4f\n",m,n,D[m*N+n],r);nbad++;}}
    printf("W4A4 GEMM %dx%dx%d: rel_L2=%.6f %s  | %.3f ms, %.1f TOPS\n",
           M,N,K,sqrt(num/(den+1e-12)), num/den<1e-5?"PASS":"FAIL", ms, tops);
    return 0;
}
