#include "mmvq.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
extern "C" void ggml_abort(const char*f,int l,const char*fmt,...){abort();}
int ggml_cuda_get_device(){int d=0;cudaGetDevice(&d);return d;}
void ggml_cuda_error(const char*s,const char*fn,const char*f,int l,const char*m){abort();}
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %d\n",__LINE__);exit(1);}}while(0)
static uint16_t f2h(float f){uint32_t b;memcpy(&b,&f,4);uint32_t s=(b>>16)&0x8000,e=(b>>23)&0xff,m=b&0x7fffff;
 if(e<=112)return s;if(e>=143)return s|0x7c00;return s|((e-112)<<10)|(m>>13);}
static float h2f(uint16_t h){uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,o;
 if(e==0){if(m==0)o=s<<31;else{e=113;while(!(m&0x400)){m<<=1;e--;}m&=0x3ff;o=(s<<31)|(e<<23)|(m<<13);}}
 else if(e==0x1f)o=(s<<31)|(0xff<<23)|(m<<13);else o=(s<<31)|((e-15+127)<<23)|(m<<13);float r;memcpy(&r,&o,4);return r;}
#define BSZ 34  // q8_0: half d + int8 qs[32]
static void oracle(const uint8_t*W,const float*X,float*dst,int N,int K){
 int nb=K/32; for(int n=0;n<N;n++){double s=0;
  for(int b=0;b<nb;b++){const uint8_t*bp=W+((size_t)n*nb+b)*BSZ; float d=h2f(*(const uint16_t*)bp);
   const int8_t*qs=(const int8_t*)(bp+2); for(int i=0;i<32;i++) s+=(double)(d*qs[i])*X[b*32+i];}
  dst[n]=(float)s;}}
static double rl2(const float*a,const float*b,int n){double nu=0,de=0;for(int i=0;i<n;i++){double d=a[i]-b[i];nu+=d*d;de+=(double)b[i]*b[i];}return sqrt(nu/(de+1e-30));}
__global__ void qk(const float*x,void*vy,int n){int wpb=blockDim.x/32,wid=blockIdx.x*wpb+threadIdx.x/32,lane=threadIdx.x%32;
 if(wid>=n/32)return;float xi=x[wid*32+lane],amax=fabsf(xi),sum=xi;amax=warp_reduce_max<32>(amax);sum=warp_reduce_sum<32>(sum);
 float d=amax/127.0f;int8_t q=amax==0?0:roundf(xi/d);block_q8_1*y=(block_q8_1*)vy;y[wid].qs[lane]=q;if(lane==0)y[wid].ds=make_half2(d,sum);}
__global__ void __launch_bounds__(calc_nwarps(GGML_TYPE_Q8_0,1,get_device_table_id())*32,1)
wrap(const void*vx,const void*vy,float*dst,uint32_t nc,uint3 ny,uint32_t srx,uint32_t scy,uint32_t scd,uint3 cr,uint3 sr){
 ggml_cuda_mm_fusion_args_device fz{};mul_mat_vec_q<GGML_TYPE_Q8_0,1,false,false>(vx,vy,nullptr,fz,dst,nc,ny,srx,scy,scd,cr,0u,0u,0u,sr,0u,0u,0u,0u);}
int main(int argc,char**argv){int N=argc>1?atoi(argv[1]):3840,K=argc>2?atoi(argv[2]):3840;
 int nb=K/32;size_t wb=(size_t)N*nb*BSZ;uint8_t*W=(uint8_t*)malloc(wb);float*X=(float*)malloc(K*4);srand(5);
 for(size_t i=0;i<wb;i++)W[i]=rand()&0xff;
 for(int n=0;n<N;n++)for(int b=0;b<nb;b++)*(uint16_t*)(W+((size_t)n*nb+b)*BSZ)=f2h(0.005f+(rand()/(float)RAND_MAX)*0.02f);
 for(int i=0;i<K;i++)X[i]=((rand()/(float)RAND_MAX)-0.5f)*2;
 float*ref=(float*)malloc(N*4);oracle(W,X,ref,N,K);
 char*dW;float*dX,*dD;void*dY;CK(cudaMalloc(&dW,wb));CK(cudaMalloc(&dX,K*4));CK(cudaMalloc(&dD,N*4));CK(cudaMalloc(&dY,(K/32)*sizeof(block_q8_1)));
 CK(cudaMemcpy(dW,W,wb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dX,X,K*4,cudaMemcpyHostToDevice));
 qk<<<(K/32+7)/8,256>>>(dX,dY,K);CK(cudaMemset(dD,0,N*4));
 uint3 one=init_fastdiv_values(1);
 wrap<<<dim3(N,1,1),dim3(32,calc_nwarps(GGML_TYPE_Q8_0,1,MMVQ_PARAMETERS_GENERIC),1)>>>(dW,dY,dD,(uint32_t)K,one,K/32,0u,N,one,one);
 CK(cudaDeviceSynchronize());float*out=(float*)malloc(N*4);CK(cudaMemcpy(out,dD,N*4,cudaMemcpyDeviceToHost));
 double e=rl2(out,ref,N);printf("Q8_0 mmvq vs CPU-F32: rel_L2=%.6f  %s\n",e,e<0.05?"PASS":"FAIL");return e<0.05?0:1;}
