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

static const float E2M1[16]={0,0.5f,1,1.5f,2,3,4,6,-0.f,-0.5f,-1,-1.5f,-2,-3,-4,-6};
static float ue4m3_decode(unsigned char b){int e=(b>>3)&0xF,m=b&0x7;
    if(e==0){ return ldexpf((float)m/8.0f,1-7); } return ldexpf(1.0f+(float)m/8.0f,e-7);}

static const char *KSRC_NAIVE =
"extern \"C\" __global__ void w4a4_gemm(const unsigned int* A, const unsigned int* B,\n"
"     const unsigned char* sA, const unsigned char* sB, float* D, int M, int N, int K){\n"
"  int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int lane=threadIdx.x&31;\n"
"  int ntn=N>>3; int tm=warp/ntn, tn=warp%ntn; int M0=tm*16, N0=tn*8;\n"
"  if(M0>=M) return;\n"
"  int g=lane>>2, t=lane&3; int Ku=K>>3, Kg=K>>4;\n"
"  long rA0=(long)(M0+g)*Ku, rA1=(long)(M0+g+8)*Ku, cB=(long)(N0+g)*Ku;\n"
"  int saRow=(t==0)?(M0+g):(t==1)?(M0+g+8):-1;\n"
"  int sbCol=(t==0)?(N0+g):-1;\n"
"  float d0=0.f,d1=0.f,d2=0.f,d3=0.f;\n"
"  int nkc=K>>6;\n"
"  for(int kc=0;kc<nkc;kc++){\n"
"    unsigned int a0=A[rA0+kc*8+t], a1=A[rA1+kc*8+t], a2=A[rA0+kc*8+t+4], a3=A[rA1+kc*8+t+4];\n"
"    unsigned int b0=B[cB+kc*8+t], b1=B[cB+kc*8+t+4];\n"
"    unsigned int sfa=0x38383838u, sfb=0x38383838u;\n"
"    if(saRow>=0){const unsigned char*p=&sA[(long)saRow*Kg+kc*4]; sfa=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"    if(sbCol>=0){const unsigned char*p=&sB[(long)sbCol*Kg+kc*4]; sfb=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"    asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
"      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
"      : \"+f\"(d0),\"+f\"(d1),\"+f\"(d2),\"+f\"(d3)\n"
"      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1),\"r\"(sfa),\"r\"(sfb));\n"
"  }\n"
"  D[(long)(M0+g)*N+N0+2*t]=d0; D[(long)(M0+g)*N+N0+2*t+1]=d1;\n"
"  D[(long)(M0+g+8)*N+N0+2*t]=d2; D[(long)(M0+g+8)*N+N0+2*t+1]=d3;\n"
"}\n";

/* Tiled, register-blocked. BM=64 BN=128 BK=64, 256 threads (8 warps, 2x4),
 * each warp 32x32 = MSUB(2) x NSUB(4) mma m16n8k64 tiles. */
static const char *KSRC_OPT =
"#define BM 64\n"
"#define BN 128\n"
"#define BK 64\n"
"#define WN_WARPS 4\n"
"#define MSUB 2\n"
"#define NSUB 4\n"
"extern \"C\" __global__ __launch_bounds__(256) void w4a4_gemm_opt(\n"
"    const unsigned int* __restrict__ A, const unsigned int* __restrict__ B,\n"
"    const unsigned char* __restrict__ sA, const unsigned char* __restrict__ sB,\n"
"    float* __restrict__ D, int M, int N, int K){\n"
"  __shared__ unsigned int smA[BM][BK/8];       /* 64*8 uint32 = 2KB */\n"
"  __shared__ unsigned int smB[BN][BK/8];       /* 128*8 uint32 = 4KB */\n"
"  __shared__ unsigned char smSA[BM][BK/16];    /* 64*4 = 256B */\n"
"  __shared__ unsigned char smSB[BN][BK/16];    /* 128*4 = 512B */\n"
"  int bm0 = blockIdx.y*BM, bn0 = blockIdx.x*BN;\n"
"  int tid = threadIdx.x, warp = tid>>5, lane = tid&31;\n"
"  int wm = warp/WN_WARPS, wn = warp%WN_WARPS;   /* warp row 0..1, col 0..3 */\n"
"  int g = lane>>2, t = lane&3;\n"
"  long Ku = K>>3, Kg = K>>4;\n"
"  float acc[MSUB][NSUB][4];\n"
"  #pragma unroll\n"
"  for(int i=0;i<MSUB;i++)for(int j=0;j<NSUB;j++)for(int e=0;e<4;e++)acc[i][j][e]=0.f;\n"
"  int nkt = K/BK;\n"
"  for(int kt=0; kt<nkt; kt++){\n"
"    long k0u = (long)kt*(BK/8), k0g = (long)kt*(BK/16);\n"
"    /* cooperative loads (256 threads) */\n"
"    #pragma unroll\n"
"    for(int idx=tid; idx<BM*(BK/8); idx+=256){ int r=idx/(BK/8), c=idx%(BK/8);\n"
"      int gr=bm0+r; smA[r][c]=(gr<M)?A[(long)gr*Ku+k0u+c]:0u; }\n"
"    #pragma unroll\n"
"    for(int idx=tid; idx<BN*(BK/8); idx+=256){ int r=idx/(BK/8), c=idx%(BK/8);\n"
"      int gr=bn0+r; smB[r][c]=(gr<N)?B[(long)gr*Ku+k0u+c]:0u; }\n"
"    #pragma unroll\n"
"    for(int idx=tid; idx<BM*(BK/16); idx+=256){ int r=idx/(BK/16), c=idx%(BK/16);\n"
"      int gr=bm0+r; smSA[r][c]=(gr<M)?sA[(long)gr*Kg+k0g+c]:0x38; }\n"
"    #pragma unroll\n"
"    for(int idx=tid; idx<BN*(BK/16); idx+=256){ int r=idx/(BK/16), c=idx%(BK/16);\n"
"      int gr=bn0+r; smSB[r][c]=(gr<N)?sB[(long)gr*Kg+k0g+c]:0x38; }\n"
"    __syncthreads();\n"
"    /* load fragments from shared */\n"
"    unsigned int af[MSUB][4], sfa[MSUB], bf[NSUB][2], sfb[NSUB];\n"
"    #pragma unroll\n"
"    for(int i=0;i<MSUB;i++){ int mr=wm*(BM/2)+i*16;\n"
"      af[i][0]=smA[mr+g][t]; af[i][1]=smA[mr+g+8][t];\n"
"      af[i][2]=smA[mr+g][t+4]; af[i][3]=smA[mr+g+8][t+4];\n"
"      unsigned int s=0x38383838u;\n"
"      if(t==0){const unsigned char*p=smSA[mr+g]; s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      else if(t==1){const unsigned char*p=smSA[mr+g+8]; s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      sfa[i]=s; }\n"
"    #pragma unroll\n"
"    for(int j=0;j<NSUB;j++){ int nr=wn*(BN/WN_WARPS)+j*8;\n"
"      bf[j][0]=smB[nr+g][t]; bf[j][1]=smB[nr+g][t+4];\n"
"      unsigned int s=0x38383838u;\n"
"      if(t==0){const unsigned char*p=smSB[nr+g]; s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      sfb[j]=s; }\n"
"    #pragma unroll\n"
"    for(int i=0;i<MSUB;i++)\n"
"      #pragma unroll\n"
"      for(int j=0;j<NSUB;j++){\n"
"        asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
"          \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
"          : \"+f\"(acc[i][j][0]),\"+f\"(acc[i][j][1]),\"+f\"(acc[i][j][2]),\"+f\"(acc[i][j][3])\n"
"          : \"r\"(af[i][0]),\"r\"(af[i][1]),\"r\"(af[i][2]),\"r\"(af[i][3]),\n"
"            \"r\"(bf[j][0]),\"r\"(bf[j][1]),\"r\"(sfa[i]),\"r\"(sfb[j]));\n"
"      }\n"
"    __syncthreads();\n"
"  }\n"
"  #pragma unroll\n"
"  for(int i=0;i<MSUB;i++)\n"
"    #pragma unroll\n"
"    for(int j=0;j<NSUB;j++){\n"
"      int mr=bm0+wm*(BM/2)+i*16, nc=bn0+wn*(BN/WN_WARPS)+j*8;\n"
"      int row0=mr+g, row1=mr+g+8, col=nc+2*t;\n"
"      if(col<N){\n"
"        if(row0<M){D[(long)row0*N+col]=acc[i][j][0]; D[(long)row0*N+col+1]=acc[i][j][1];}\n"
"        if(row1<M){D[(long)row1*N+col]=acc[i][j][2]; D[(long)row1*N+col+1]=acc[i][j][3];}\n"
"      }\n"
"    }\n"
"}\n";

/* cp.async double-buffered variant of the tiled kernel. Same BMxBN tiling and
 * fragment/scale layout, but prefetches K-tile kt+1 into the other shared buffer
 * while computing kt, hiding global-load latency. OOB rows use cp.async src-size=0
 * (zero-fill). Vectorized uint4 loads for the code tiles. */
static const char *KSRC_CPA =
"#define BM 64\n"
"#define BN 128\n"
"#define BK 64\n"
"#define WN_WARPS 4\n"
"#define MSUB 2\n"
"#define NSUB 4\n"
"__device__ __forceinline__ unsigned smad(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n"
"__device__ __forceinline__ void cpa16(unsigned s,const void*g,int sz){\n"
"  asm volatile(\"cp.async.cg.shared.global [%0],[%1],16,%2;\\n\"::\"r\"(s),\"l\"(g),\"r\"(sz));}\n"
"__device__ __forceinline__ void cpa4(unsigned s,const void*g,int sz){\n"
"  asm volatile(\"cp.async.ca.shared.global [%0],[%1],4,%2;\\n\"::\"r\"(s),\"l\"(g),\"r\"(sz));}\n"
"__device__ __forceinline__ void load_tile(\n"
"    unsigned int sa0[BM][BK/8], unsigned int sb0[BN][BK/8],\n"
"    unsigned char ssa[BM][BK/16], unsigned char ssb[BN][BK/16],\n"
"    const unsigned int* A,const unsigned int* B,\n"
"    const unsigned char* sA,const unsigned char* sB,\n"
"    int bm0,int bn0,int M,int N,long Ku,long Kg,int tid,int kt){\n"
"  long k0u=(long)kt*(BK/8), k0g=(long)kt*(BK/16);\n"
"  for(int idx=tid; idx<BM*2; idx+=256){ int r=idx>>1, q=(idx&1)*4; int gr=bm0+r;\n"
"    cpa16(smad(&sa0[r][q]), &A[(long)gr*Ku+k0u+q], gr<M?16:0); }\n"
"  for(int idx=tid; idx<BN*2; idx+=256){ int r=idx>>1, q=(idx&1)*4; int gr=bn0+r;\n"
"    cpa16(smad(&sb0[r][q]), &B[(long)gr*Ku+k0u+q], gr<N?16:0); }\n"
"  for(int idx=tid; idx<BM; idx+=256){ int gr=bm0+idx;\n"
"    cpa4(smad(&ssa[idx][0]), &sA[(long)gr*Kg+k0g], gr<M?4:0); }\n"
"  for(int idx=tid; idx<BN; idx+=256){ int gr=bn0+idx;\n"
"    cpa4(smad(&ssb[idx][0]), &sB[(long)gr*Kg+k0g], gr<N?4:0); }\n"
"  asm volatile(\"cp.async.commit_group;\\n\"); }\n"
"extern \"C\" __global__ __launch_bounds__(256) void w4a4_gemm_cpa(\n"
"    const unsigned int* __restrict__ A, const unsigned int* __restrict__ B,\n"
"    const unsigned char* __restrict__ sA, const unsigned char* __restrict__ sB,\n"
"    float* __restrict__ D, int M, int N, int K){\n"
"  __shared__ unsigned int  smA[2][BM][BK/8];\n"
"  __shared__ unsigned int  smB[2][BN][BK/8];\n"
"  __shared__ unsigned char smSA[2][BM][BK/16];\n"
"  __shared__ unsigned char smSB[2][BN][BK/16];\n"
"  int bm0=blockIdx.y*BM, bn0=blockIdx.x*BN;\n"
"  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;\n"
"  int wm=warp/WN_WARPS, wn=warp%WN_WARPS;\n"
"  int g=lane>>2, t=lane&3;\n"
"  long Ku=K>>3, Kg=K>>4;\n"
"  float acc[MSUB][NSUB][4];\n"
"  #pragma unroll\n"
"  for(int i=0;i<MSUB;i++)for(int j=0;j<NSUB;j++)for(int e=0;e<4;e++)acc[i][j][e]=0.f;\n"
"  int nkt=K/BK;\n"
"  load_tile(smA[0],smB[0],smSA[0],smSB[0],A,B,sA,sB,bm0,bn0,M,N,Ku,Kg,tid,0);\n"
"  for(int kt=0; kt<nkt; kt++){\n"
"    int cur=kt&1, nxt=cur^1;\n"
"    if(kt+1<nkt){ load_tile(smA[nxt],smB[nxt],smSA[nxt],smSB[nxt],A,B,sA,sB,bm0,bn0,M,N,Ku,Kg,tid,kt+1);\n"
"      asm volatile(\"cp.async.wait_group %0;\\n\"::\"n\"(1)); }  /* cur ready, next in flight */\n"
"    else asm volatile(\"cp.async.wait_group %0;\\n\"::\"n\"(0));  /* last tile: wait for it */\n"
"    __syncthreads();\n"
"    unsigned int af[MSUB][4], sfa[MSUB], bf[NSUB][2], sfb[NSUB];\n"
"    #pragma unroll\n"
"    for(int i=0;i<MSUB;i++){ int mr=wm*(BM/2)+i*16;\n"
"      af[i][0]=smA[cur][mr+g][t]; af[i][1]=smA[cur][mr+g+8][t];\n"
"      af[i][2]=smA[cur][mr+g][t+4]; af[i][3]=smA[cur][mr+g+8][t+4];\n"
"      unsigned int s=0x38383838u;\n"
"      if(t==0){const unsigned char*p=smSA[cur][mr+g]; s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      else if(t==1){const unsigned char*p=smSA[cur][mr+g+8]; s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      sfa[i]=s; }\n"
"    #pragma unroll\n"
"    for(int j=0;j<NSUB;j++){ int nr=wn*(BN/WN_WARPS)+j*8;\n"
"      bf[j][0]=smB[cur][nr+g][t]; bf[j][1]=smB[cur][nr+g][t+4];\n"
"      unsigned int s=0x38383838u;\n"
"      if(t==0){const unsigned char*p=smSB[cur][nr+g]; s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      sfb[j]=s; }\n"
"    #pragma unroll\n"
"    for(int i=0;i<MSUB;i++)\n"
"      #pragma unroll\n"
"      for(int j=0;j<NSUB;j++){\n"
"        asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
"          \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
"          : \"+f\"(acc[i][j][0]),\"+f\"(acc[i][j][1]),\"+f\"(acc[i][j][2]),\"+f\"(acc[i][j][3])\n"
"          : \"r\"(af[i][0]),\"r\"(af[i][1]),\"r\"(af[i][2]),\"r\"(af[i][3]),\n"
"            \"r\"(bf[j][0]),\"r\"(bf[j][1]),\"r\"(sfa[i]),\"r\"(sfb[j]));\n"
"      }\n"
"    __syncthreads();\n"  /* buffer-reuse barrier: next iter's load_tile overwrites the buffer this iter just read */
"  }\n"
"  #pragma unroll\n"
"  for(int i=0;i<MSUB;i++)\n"
"    #pragma unroll\n"
"    for(int j=0;j<NSUB;j++){\n"
"      int mr=bm0+wm*(BM/2)+i*16, nc=bn0+wn*(BN/WN_WARPS)+j*8;\n"
"      int row0=mr+g, row1=mr+g+8, col=nc+2*t;\n"
"      if(col<N){\n"
"        if(row0<M){D[(long)row0*N+col]=acc[i][j][0]; D[(long)row0*N+col+1]=acc[i][j][1];}\n"
"        if(row1<M){D[(long)row1*N+col]=acc[i][j][2]; D[(long)row1*N+col+1]=acc[i][j][3];}\n"
"      }\n"
"    }\n"
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
