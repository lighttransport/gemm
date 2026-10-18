/* Optimized tiled NVFP4 W4A4 GEMM on sm_120a.
 *
 * Computes D[m,n] = sum_k e2m1(A[m,k])*sA[m,k/16] * e2m1(B[n,k])*sB[n,k/16]
 * (A row-major MxK, B "col" NxK == A@B^T).
 *
 * Two kernels:
 *   w4a4_gemm       - naive (one warp per 16x8 tile, no reuse) = oracle/baseline.
 *   w4a4_gemm_opt   - shared-mem tiled, register-blocked (BMxBN per block,
 *                     each warp 32x32 = 2 m-subtiles x 4 n-subtiles).
 *
 * Verify: opt vs naive on-GPU (fast); CPU ref only at small sizes.
 *
 * Build: gcc -O2 -I.. -o fp4_w4a4_opt fp4_w4a4_opt.c ../cuew.c -ldl -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuew.h"

#define CHECK_CUDA(call) do{CUresult e=(call);if(e!=CUDA_SUCCESS){const char*s; \
    cuGetErrorString(e,&s);fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,s);exit(1);}}while(0)

#include "fp4_gemm_kernels.h"

static CUmodule compile_sm120a(const char*src){
    nvrtcProgram p; if(nvrtcCreateProgram(&p,src,"k",0,NULL,NULL)!=NVRTC_SUCCESS)return NULL;
    const char*o[]={"--gpu-architecture=sm_120a"};
    if(nvrtcCompileProgram(p,1,o)!=NVRTC_SUCCESS){size_t l=0;nvrtcGetProgramLogSize(p,&l);
        char*b=malloc(l+1);nvrtcGetProgramLog(p,b);b[l]=0;fprintf(stderr,"REJECTED:\n%s\n",b);free(b);return NULL;}
    CUmodule m=NULL; size_t bs=0;
    if(nvrtcGetCUBINSize&&nvrtcGetCUBINSize(p,&bs)==NVRTC_SUCCESS&&bs>0){char*bl=malloc(bs);
        nvrtcGetCUBIN(p,bl);nvrtcDestroyProgram(&p);if(cuModuleLoadData(&m,bl)!=CUDA_SUCCESS)m=NULL;free(bl);}
    else{size_t ps=0;nvrtcGetPTXSize(p,&ps);char*x=malloc(ps);nvrtcGetPTX(p,x);nvrtcDestroyProgram(&p);
        if(cuModuleLoadData(&m,x)!=CUDA_SUCCESS){m=NULL;}free(x);}
    return m;
}

static float time_kernel(CUfunction fn, int blocks_x,int blocks_y,int threads, void**args,int iters){
    CUevent e0,e1; cuEventCreate(&e0,0); cuEventCreate(&e1,0);
    CHECK_CUDA(cuLaunchKernel(fn,blocks_x,blocks_y,1,threads,1,1,0,0,args,NULL));
    CHECK_CUDA(cuCtxSynchronize());
    cuEventRecord(e0,0);
    for(int it=0;it<iters;it++) CHECK_CUDA(cuLaunchKernel(fn,blocks_x,blocks_y,1,threads,1,1,0,0,args,NULL));
    cuEventRecord(e1,0); CHECK_CUDA(cuCtxSynchronize());
    float ms=0; cuEventElapsedTime(&ms,e0,e1); return ms/iters;
}

int main(int argc,char**argv){
    int M=256,N=3072,K=3072, verify=1;
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-m"))M=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-n"))N=atoi(argv[++i]);else if(!strcmp(argv[i],"-k"))K=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--no-verify"))verify=0;}
    if(K%64){fprintf(stderr,"need K%%64\n");return 1;}
    if(cuewInit(CUEW_INIT_CUDA|CUEW_INIT_NVRTC)!=CUEW_SUCCESS){fprintf(stderr,"cuewInit\n");return 1;}
    CHECK_CUDA(cuInit(0)); CUdevice dev;CUcontext ctx;
    CHECK_CUDA(cuDeviceGet(&dev,0)); CHECK_CUDA(cuCtxCreate(&ctx,0,dev));
    CUmodule mN=compile_sm120a(KSRC_NAIVE), mO=compile_sm120a(KSRC_OPT), mC=compile_sm120a(KSRC_CPA);
    if(!mN||!mO||!mC){fprintf(stderr,"compile fail (naive=%p opt=%p cpa=%p)\n",(void*)mN,(void*)mO,(void*)mC);return 1;}
    CUfunction fnN,fnO,fnC;
    CHECK_CUDA(cuModuleGetFunction(&fnN,mN,"w4a4_gemm"));
    CHECK_CUDA(cuModuleGetFunction(&fnO,mO,"w4a4_gemm_opt"));
    CHECK_CUDA(cuModuleGetFunction(&fnC,mC,"w4a4_gemm_cpa"));

    int Ku=K/8, Kg=K/16;
    unsigned char *Ac=malloc((size_t)M*K), *Bc=malloc((size_t)N*K);
    unsigned char *sA=malloc((size_t)M*Kg), *sB=malloc((size_t)N*Kg);
    srand(7);
    for(size_t i=0;i<(size_t)M*K;i++)Ac[i]=rand()&0xF;
    for(size_t i=0;i<(size_t)N*K;i++)Bc[i]=rand()&0xF;
    for(size_t i=0;i<(size_t)M*Kg;i++)sA[i]=0x30+(rand()%16);
    for(size_t i=0;i<(size_t)N*Kg;i++)sB[i]=0x30+(rand()%16);
    unsigned int *Au=calloc((size_t)M*Ku,4), *Bu=calloc((size_t)N*Ku,4);
    for(int m=0;m<M;m++)for(int u=0;u<Ku;u++){unsigned v=0;for(int i=0;i<8;i++)v|=((unsigned)Ac[m*K+u*8+i]&0xF)<<(i*4);Au[m*Ku+u]=v;}
    for(int n=0;n<N;n++)for(int u=0;u<Ku;u++){unsigned v=0;for(int i=0;i<8;i++)v|=((unsigned)Bc[n*K+u*8+i]&0xF)<<(i*4);Bu[n*Ku+u]=v;}

    CUdeviceptr dA,dB,dsA,dsB,dDn,dDo,dDc;
    CHECK_CUDA(cuMemAlloc(&dA,(size_t)M*Ku*4)); CHECK_CUDA(cuMemAlloc(&dB,(size_t)N*Ku*4));
    CHECK_CUDA(cuMemAlloc(&dsA,(size_t)M*Kg)); CHECK_CUDA(cuMemAlloc(&dsB,(size_t)N*Kg));
    CHECK_CUDA(cuMemAlloc(&dDn,(size_t)M*N*4)); CHECK_CUDA(cuMemAlloc(&dDo,(size_t)M*N*4));
    CHECK_CUDA(cuMemAlloc(&dDc,(size_t)M*N*4));
    CHECK_CUDA(cuMemcpyHtoD(dA,Au,(size_t)M*Ku*4)); CHECK_CUDA(cuMemcpyHtoD(dB,Bu,(size_t)N*Ku*4));
    CHECK_CUDA(cuMemcpyHtoD(dsA,sA,(size_t)M*Kg)); CHECK_CUDA(cuMemcpyHtoD(dsB,sB,(size_t)N*Kg));

    /* --- naive (oracle), requires M%16,N%8 --- */
    float msN=0; int haveN=0;
    if(M%16==0 && N%8==0){
        int nwarps=(M/16)*(N/8), threads=128, blocks=(nwarps*32+threads-1)/threads;
        void*args[]={&dA,&dB,&dsA,&dsB,&dDn,&M,&N,&K};
        msN=time_kernel(fnN,blocks,1,threads,args,50); haveN=1;
    }
    /* --- opt (sync) and cpa (cp.async double-buffer) --- */
    int BM=64,BN=128, bx=(N+BN-1)/BN, by=(M+BM-1)/BM;
    void*argsO[]={&dA,&dB,&dsA,&dsB,&dDo,&M,&N,&K};
    void*argsC[]={&dA,&dB,&dsA,&dsB,&dDc,&M,&N,&K};
    float msO=time_kernel(fnO,bx,by,256,argsO,50);
    float msC=time_kernel(fnC,bx,by,256,argsC,50);

    double topsN = haveN? 2.0*M*N*K/(msN*1e-3)/1e12 : 0;
    double topsO = 2.0*M*N*K/(msO*1e-3)/1e12;
    double topsC = 2.0*M*N*K/(msC*1e-3)/1e12;

    /* correctness vs naive on GPU */
    if(verify && haveN){
        float *Dn=malloc((size_t)M*N*4),*Dx=malloc((size_t)M*N*4);
        CHECK_CUDA(cuMemcpyDtoH(Dn,dDn,(size_t)M*N*4));
        for(int which=0;which<2;which++){
            CHECK_CUDA(cuMemcpyDtoH(Dx, which?dDc:dDo, (size_t)M*N*4));
            double num=0,den=0,maxd=0; int nbad=0;
            for(size_t i=0;i<(size_t)M*N;i++){double d=Dx[i]-Dn[i]; num+=d*d; den+=(double)Dn[i]*Dn[i];
                if(fabs(d)>maxd)maxd=fabs(d);
                if(fabs(d)>1e-3*(1+fabs(Dn[i]))&&nbad<4){printf("  mism[%zu] %s %.4f naive %.4f\n",i,which?"cpa":"opt",Dx[i],Dn[i]);nbad++;}}
            printf("  %s-vs-naive: rel_L2=%.3e max|d|=%.3e %s\n",which?"cpa":"opt",
                   sqrt(num/(den+1e-12)),maxd,num/den<1e-9?"BIT-MATCH":(num/den<1e-5?"OK":"FAIL"));
        }
        free(Dn);free(Dx);
    }
    printf("W4A4 %dx%dx%d:  naive %.3f ms %.1f TOPS | opt %.3f ms %.1f TOPS %.2fx | cpa %.3f ms %.1f TOPS %.2fx\n",
           M,N,K, haveN?msN:0, topsN, msO, topsO, haveN?msN/msO:0, msC, topsC, haveN?msN/msC:0);
    return 0;
}
