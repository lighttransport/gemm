/* End-to-end W4A4 op validation on REAL block-0 to_qkv data (dumped by Python).
 * GPU: fp4 activation-quant + tiled W4A4 block-scale GEMM. CPU: low-rank +
 * per-channel scale + bias. Compares to nunchaku ground-truth y (expect the
 * W4A4 activation-quant noise floor, rel ~0.10).
 *
 * Build: gcc -O2 -I.. -o fp4_w4a4_qkv_test fp4_w4a4_qkv_test.c ../cuew.c -ldl -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuew.h"
#define CHECK_CUDA(c) do{CUresult e=(c);if(e!=CUDA_SUCCESS){const char*s;cuGetErrorString(e,&s);\
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,s);exit(1);}}while(0)
static const float E2M1[16]={0,0.5f,1,1.5f,2,3,4,6,-0.f,-0.5f,-1,-1.5f,-2,-3,-4,-6};

static void* rdbin(const char*p,size_t bytes){FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open %s\n",p);exit(1);}
    void*b=malloc(bytes); if(fread(b,1,bytes,f)!=bytes){fprintf(stderr,"short %s\n",p);exit(1);} fclose(f);return b;}

static const char *KSRC =
"__device__ __forceinline__ unsigned char f32_to_e4m3(float v){\n"
"  unsigned short p; asm(\"cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;\":\"=h\"(p):\"f\"(v),\"f\"(v)); return p&0xFF; }\n"
"__device__ __forceinline__ float e4m3_decode(unsigned char b){int e=(b>>3)&0xF,m=b&7;\n"
"  if(e==0) return (m/8.0f)*0.015625f; return (1.0f+m/8.0f)*exp2f((float)(e-7)); }\n"
"__device__ __forceinline__ unsigned char nearest_e2m1(float q){\n"
"  unsigned char s=q<0?8:0; float a=fabsf(q); unsigned char c;\n"
"  if(a<0.25f)c=0;else if(a<0.75f)c=1;else if(a<1.25f)c=2;else if(a<1.75f)c=3;\n"
"  else if(a<2.5f)c=4;else if(a<3.5f)c=5;else if(a<5.0f)c=6;else c=7; return s|c; }\n"
"extern \"C\" __global__ void fp4_quant_act(const float* X, unsigned int* Ac, unsigned char* As, int M, int K){\n"
"  int ng=K>>4; long gi=(long)blockIdx.x*blockDim.x+threadIdx.x; if(gi>=(long)M*ng) return;\n"
"  int m=gi/ng, grp=gi%ng; long k0=(long)grp*16;\n"
"  float amax=0; for(int i=0;i<16;i++){float v=fabsf(X[(long)m*K+k0+i]); amax=fmaxf(amax,v);}\n"
"  float sc=amax*0.16666667f; unsigned char sb=f32_to_e4m3(sc); float sd=e4m3_decode(sb);\n"
"  float inv = sd>0.f? 1.0f/sd : 0.f; As[(long)m*ng+grp]=sb;\n"
"  for(int u=0;u<2;u++){ unsigned int v=0; for(int i=0;i<8;i++){ long k=k0+u*8+i;\n"
"      float q=X[(long)m*K+k]*inv; v|=((unsigned)nearest_e2m1(q))<<(i*4);} Ac[(long)m*(K>>3)+grp*2+u]=v; } }\n"
"extern \"C\" __global__ void w4a4_gemm(const unsigned int* A, const unsigned int* B,\n"
"     const unsigned char* sA, const unsigned char* sB, float* D, int M, int N, int K){\n"
"  int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int lane=threadIdx.x&31;\n"
"  int ntn=N>>3; int tm=warp/ntn, tn=warp%ntn; int M0=tm*16, N0=tn*8; if(M0>=M) return;\n"
"  int g=lane>>2, t=lane&3; int Ku=K>>3, Kg=K>>4;\n"
"  long rA0=(long)(M0+g)*Ku, rA1=(long)(M0+g+8)*Ku, cB=(long)(N0+g)*Ku;\n"
"  int saRow=(t==0)?(M0+g):(t==1)?(M0+g+8):-1; int sbCol=(t==0)?(N0+g):-1;\n"
"  float d0=0.f,d1=0.f,d2=0.f,d3=0.f; int nkc=K>>6;\n"
"  for(int kc=0;kc<nkc;kc++){\n"
"    unsigned int a0=A[rA0+kc*8+t],a1=A[rA1+kc*8+t],a2=A[rA0+kc*8+t+4],a3=A[rA1+kc*8+t+4];\n"
"    unsigned int b0=B[cB+kc*8+t],b1=B[cB+kc*8+t+4];\n"
"    unsigned int sfa=0x38383838u,sfb=0x38383838u;\n"
"    if(saRow>=0){const unsigned char*p=&sA[(long)saRow*Kg+kc*4]; sfa=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"    if(sbCol>=0){const unsigned char*p=&sB[(long)sbCol*Kg+kc*4]; sfb=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"    asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
"      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
"      : \"+f\"(d0),\"+f\"(d1),\"+f\"(d2),\"+f\"(d3)\n"
"      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1),\"r\"(sfa),\"r\"(sfb));\n"
"  }\n"
"  D[(long)(M0+g)*N+N0+2*t]=d0; D[(long)(M0+g)*N+N0+2*t+1]=d1;\n"
"  D[(long)(M0+g+8)*N+N0+2*t]=d2; D[(long)(M0+g+8)*N+N0+2*t+1]=d3; }\n";

static CUmodule compile_sm120a(const char*src){
    nvrtcProgram p; if(nvrtcCreateProgram(&p,src,"k",0,NULL,NULL)!=NVRTC_SUCCESS)return NULL;
    const char*o[]={"--gpu-architecture=sm_120a"};
    if(nvrtcCompileProgram(p,1,o)!=NVRTC_SUCCESS){size_t l=0;nvrtcGetProgramLogSize(p,&l);char*b=malloc(l+1);
        nvrtcGetProgramLog(p,b);b[l]=0;fprintf(stderr,"REJECTED:\n%s\n",b);free(b);return NULL;}
    CUmodule m=NULL; size_t bs=0;
    if(nvrtcGetCUBINSize&&nvrtcGetCUBINSize(p,&bs)==NVRTC_SUCCESS&&bs>0){char*bl=malloc(bs);nvrtcGetCUBIN(p,bl);
        nvrtcDestroyProgram(&p);if(cuModuleLoadData(&m,bl)!=CUDA_SUCCESS)m=NULL;free(bl);}
    else{size_t ps=0;nvrtcGetPTXSize(p,&ps);char*x=malloc(ps);nvrtcGetPTX(p,x);nvrtcDestroyProgram(&p);
        if(cuModuleLoadData(&m,x)!=CUDA_SUCCESS)m=NULL;free(x);}
    return m;
}

int main(void){
    const int M=256,IN=3072,OUT=9216,R=128,NG=IN/16,Ku=IN/8;
    float *x=rdbin("/tmp/qkv_x.bin",(size_t)M*IN*4);
    float *y=rdbin("/tmp/qkv_y.bin",(size_t)M*OUT*4);
    unsigned char *codes=rdbin("/tmp/qkv_codes.bin",(size_t)OUT*IN);     /* [OUT][IN] 0..15 */
    unsigned char *wsb=rdbin("/tmp/qkv_wsb.bin",(size_t)OUT*NG);          /* [OUT][NG] e4m3 */
    float *wcwt=rdbin("/tmp/qkv_wcwt.bin",(size_t)OUT*4);
    float *pd=rdbin("/tmp/qkv_pd.bin",(size_t)R*IN*4);                    /* [R][IN] */
    float *pu=rdbin("/tmp/qkv_pu.bin",(size_t)OUT*R*4);                   /* [OUT][R] */
    float *bias=rdbin("/tmp/qkv_bias.bin",(size_t)OUT*4);

    /* pack weight codes -> Bu[OUT][Ku] uint (8 codes/uint) */
    unsigned int *Bu=calloc((size_t)OUT*Ku,4);
    for(int o=0;o<OUT;o++)for(int u=0;u<Ku;u++){unsigned v=0;for(int i=0;i<8;i++)v|=((unsigned)codes[(size_t)o*IN+u*8+i]&0xF)<<(i*4);Bu[(size_t)o*Ku+u]=v;}

    if(cuewInit(CUEW_INIT_CUDA|CUEW_INIT_NVRTC)!=CUEW_SUCCESS){fprintf(stderr,"cuewInit\n");return 1;}
    CHECK_CUDA(cuInit(0)); CUdevice dev;CUcontext ctx;
    CHECK_CUDA(cuDeviceGet(&dev,0)); CHECK_CUDA(cuCtxCreate(&ctx,0,dev));
    CUmodule mod=compile_sm120a(KSRC); if(!mod)return 1;
    CUfunction kq,kg; CHECK_CUDA(cuModuleGetFunction(&kq,mod,"fp4_quant_act"));
    CHECK_CUDA(cuModuleGetFunction(&kg,mod,"w4a4_gemm"));

    CUdeviceptr dX,dAc,dAs,dB,dwsb,dD;
    CHECK_CUDA(cuMemAlloc(&dX,(size_t)M*IN*4)); CHECK_CUDA(cuMemcpyHtoD(dX,x,(size_t)M*IN*4));
    CHECK_CUDA(cuMemAlloc(&dAc,(size_t)M*Ku*4)); CHECK_CUDA(cuMemAlloc(&dAs,(size_t)M*NG));
    CHECK_CUDA(cuMemAlloc(&dB,(size_t)OUT*Ku*4)); CHECK_CUDA(cuMemcpyHtoD(dB,Bu,(size_t)OUT*Ku*4));
    CHECK_CUDA(cuMemAlloc(&dwsb,(size_t)OUT*NG)); CHECK_CUDA(cuMemcpyHtoD(dwsb,wsb,(size_t)OUT*NG));
    CHECK_CUDA(cuMemAlloc(&dD,(size_t)M*OUT*4));

    /* 1. activation quant */
    int Mi=M,Ki=IN,Ni=OUT; long ngroups=(long)M*NG;
    void*qa[]={&dX,&dAc,&dAs,&Mi,&Ki}; unsigned qb=(unsigned)((ngroups+255)/256);
    CHECK_CUDA(cuLaunchKernel(kq,qb,1,1,256,1,1,0,0,qa,NULL));
    /* 2. W4A4 GEMM */
    int nwarps=(M/16)*(OUT/8),threads=128,blocks=(nwarps*32+threads-1)/threads;
    void*ga[]={&dAc,&dB,&dAs,&dwsb,&dD,&Mi,&Ni,&Ki};
    CHECK_CUDA(cuLaunchKernel(kg,blocks,1,1,threads,1,1,0,0,ga,NULL));
    CHECK_CUDA(cuCtxSynchronize());
    float *D=malloc((size_t)M*OUT*4); CHECK_CUDA(cuMemcpyDtoH(D,dD,(size_t)M*OUT*4));

    /* 3. CPU low-rank: la=x@pd^T [M][R]; lo=la@pu^T [M][OUT]; out=D*wcwt+lo+bias */
    float *la=calloc((size_t)M*R,4);
    for(int m=0;m<M;m++)for(int rr=0;rr<R;rr++){double s=0;for(int i=0;i<IN;i++)s+=(double)x[(size_t)m*IN+i]*pd[(size_t)rr*IN+i]; la[(size_t)m*R+rr]=(float)s;}
    double num=0,den=0;
    for(int m=0;m<M;m++)for(int o=0;o<OUT;o++){double lo=0;for(int rr=0;rr<R;rr++)lo+=(double)la[(size_t)m*R+rr]*pu[(size_t)o*R+rr];
        double v=(double)D[(size_t)m*OUT+o]*wcwt[o]+lo+bias[o]; double r=y[(size_t)m*OUT+o];
        double d=v-r; num+=d*d; den+=r*r;}
    printf("W4A4 to_qkv end-to-end (GPU quant+GEMM, CPU lora): rel_L2 vs GT y = %.5f\n", sqrt(num/(den+1e-12)));
    printf("(expected ~0.10 = W4A4 activation-quant noise floor; Stage-1 W4A16 baseline was 0.1006)\n");
    return 0;
}
