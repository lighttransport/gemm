// Isolation test for the Q6_K vendored stream-K MMQ bug (rel_L2=0.25 in-model).
// Feeds random standard-gguf Q6_K weights through llama's mul_mat_q<Q6_K,128> and
// compares vs a CPU-F32 oracle decoding the SAME bytes with ggml's q6_K formula.
// PASS here => kernel is fine, bug is in the runner integration. FAIL => kernel/quant.
//
// build:
//  nvcc -arch=sm_120a -O3 -std=c++17 \
//    -I<llama>/ggml/src/ggml-cuda -I<llama>/ggml/include -I<llama>/ggml/src \
//    mmq_q6k_vendor_test.cu <llama>/ggml/src/ggml-cuda/quantize.cu -o q6ktest
#include "mmq.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>

void quantize_mmq_q8_1_cuda(const float*,const int32_t*,void*,ggml_type,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,cudaStream_t);

extern "C" void ggml_abort(const char*f,int l,const char*fmt,...){fprintf(stderr,"abort %s:%d %s\n",f,l,fmt);abort();}
int ggml_cuda_get_device(){int d=0;cudaGetDevice(&d);return d;}
void ggml_cuda_error(const char*s,const char*fn,const char*f,int l,const char*m){fprintf(stderr,"cuerr %s %s %s:%d %s\n",s,fn,f,l,m);abort();}
#define CK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static float h2f(uint16_t h){uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,o;
  if(!e){if(!m)o=s<<31;else{e=127-15+1;while(!(m&0x400)){m<<=1;e--;}m&=0x3ff;o=(s<<31)|(e<<23)|(m<<13);}}
  else if(e==0x1f)o=(s<<31)|(0xff<<23)|(m<<13);else o=(s<<31)|((e-15+127)<<23)|(m<<13);
  float f;memcpy(&f,&o,4);return f;}
static uint16_t f2h(float f){uint32_t x;memcpy(&x,&f,4);uint32_t s=(x>>16)&0x8000,e=(x>>23)&0xff,m=x&0x7fffff;
  if(e<=112)return (uint16_t)s; if(e>=143)return (uint16_t)(s|0x7c00); return (uint16_t)(s|((e-112)<<10)|(m>>13));}

// standard ggml q6_K block = 210 bytes
struct bq6 { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };

static void decode_q6k(const uint8_t*bp,float*y){
  const bq6*b=(const bq6*)bp; float d=h2f(b->d);
  const uint8_t*ql=b->ql,*qh=b->qh; const int8_t*sc=b->scales;
  for(int half=0;half<2;half++){
    for(int l=0;l<32;l++){int is=l/16;
      int8_t q1=(int8_t)((ql[l+0]&0xF)|(((qh[l]>>0)&3)<<4))-32;
      int8_t q2=(int8_t)((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
      int8_t q3=(int8_t)((ql[l+0]>>4)|(((qh[l]>>4)&3)<<4))-32;
      int8_t q4=(int8_t)((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
      y[l+0]=d*sc[is+0]*q1; y[l+32]=d*sc[is+2]*q2; y[l+64]=d*sc[is+4]*q3; y[l+96]=d*sc[is+6]*q4;}
    y+=128; ql+=64; qh+=32; sc+=8;}
}
static void oracle(const uint8_t*W,const float*X,float*dst,int M,int N,int K){
  int nb=K/256; size_t rb=(size_t)nb*210; float*wf=(float*)malloc((size_t)K*4);
  for(int n=0;n<N;n++){for(int b=0;b<nb;b++)decode_q6k(W+(size_t)n*rb+b*210,wf+b*256);
    for(int m=0;m<M;m++){double s=0;const float*x=X+(size_t)m*K;for(int k=0;k<K;k++)s+=(double)wf[k]*x[k];dst[(size_t)m*N+n]=(float)s;}}
  free(wf);
}
static double rl2(const float*a,const float*b,int n){double nu=0,de=0;for(int i=0;i<n;i++){double d=a[i]-b[i];nu+=d*d;de+=(double)b[i]*b[i];}return sqrt(nu/(de+1e-30));}

static float*g_fix=nullptr;static size_t g_cap=0;
static void launch_q6k(const char*W,const int*yq8,float*dst,int N,int K,int M,int nsm,cudaStream_t st){
  const int mmq_x=128,ws=32,nw=mmq_get_nwarps_host(1200,ws),my=get_mmq_y_host(1200);
  size_t nbs=mmq_get_nbytes_shared<GGML_TYPE_Q6_K>(mmq_x,my,1200,ws,nw);
  CK(cudaFuncSetAttribute(mul_mat_q<GGML_TYPE_Q6_K,mmq_x,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)nbs));
  int nty=(N+my-1)/my,ntx=(M+mmq_x-1)/mmq_x,ntiles=ntx*nty,nw2=(ntiles+nsm-1)/nsm;
  int eff=100*ntiles/(nsm*nw2),sk=(eff>=90)?ntiles:nsm; bool fx=(ntiles%sk)!=0;
  uint3 bp=init_fastdiv_values((unsigned long long)K/256),one=init_fastdiv_values(1),ntxfd=init_fastdiv_values((unsigned long long)ntx);
  int s01=K/256;
  if(fx){size_t need=(size_t)sk*mmq_x*my*4;if(need>g_cap){if(g_fix)cudaFree(g_fix);CK(cudaMalloc(&g_fix,need));g_cap=need;}}
  float*tf=fx?g_fix:nullptr;
  mul_mat_q<GGML_TYPE_Q6_K,mmq_x,false><<<dim3(sk,1,1),dim3(ws,nw,1),nbs,st>>>(
    W,yq8,nullptr,nullptr,dst,tf, bp,N,M,s01,M,N, one,one,0,0,0, one,one,0,0,0, ntxfd);
  if(fx) mul_mat_q_stream_k_fixup<GGML_TYPE_Q6_K,mmq_x,false><<<dim3(sk,my/ws,1),dim3(ws,nw/2,1),0,st>>>(
    nullptr,nullptr,dst,g_fix, bp,N,M,N, one,0,one,0, ntxfd);
}

int main(int argc,char**argv){
  if(argc>1 && !strcmp(argv[1],"--verify-dump")){
    FILE*f=fopen("/tmp/q6dump.bin","rb"); if(!f){printf("no dump\n");return 1;}
    int M,N,K; fread(&M,4,1,f);fread(&N,4,1,f);fread(&K,4,1,f);
    int nb=K/256; size_t wb=(size_t)N*nb*210,xb=(size_t)M*K*4,db=(size_t)M*N*4;
    uint8_t*W=(uint8_t*)malloc(wb); float*X=(float*)malloc(xb),*D=(float*)malloc(db);
    fread(W,1,wb,f);fread(X,1,xb,f);fread(D,1,db,f);fclose(f);
    printf("verify-dump M=%d N=%d K=%d\n",M,N,K);
    float*ref=(float*)malloc(db); oracle(W,X,ref,M,N,K);
    double e=rl2(D,ref,M*N);
    printf("in-model dst vs CPU-F32 oracle (same bytes): rel_L2=%.6f %s\n",e,e<0.05?"PASS (kernel correct in-model)":"FAIL (cubin kernel wrong in-model)");
    printf("  ref[0..3]=%.3f %.3f %.3f %.3f\n  dst[0..3]=%.3f %.3f %.3f %.3f\n",ref[0],ref[1],ref[2],ref[3],D[0],D[1],D[2],D[3]);
    return 0;
  }
  int M=argc>1?atoi(argv[1]):512,N=argc>2?atoi(argv[2]):3840,K=argc>3?atoi(argv[3]):3840;
  if(N%128||K%256){printf("need N%%128 K%%256\n");return 1;}
  printf("Q6_K vendor test M=%d N=%d K=%d\n",M,N,K);
  int nb=K/256; size_t wb=(size_t)N*nb*210;
  uint8_t*W=(uint8_t*)malloc(wb); float*X=(float*)malloc((size_t)M*K*4);
  srand(7);
  for(int n=0;n<N;n++)for(int b=0;b<nb;b++){bq6*blk=(bq6*)(W+((size_t)n*nb+b)*210);
    for(int i=0;i<128;i++)blk->ql[i]=rand()&0xff; for(int i=0;i<64;i++)blk->qh[i]=rand()&0xff;
    for(int i=0;i<16;i++)blk->scales[i]=(int8_t)((rand()%64)-0); // 0..63 typical q6_K scales
    blk->d=f2h(0.001f+(rand()/(float)RAND_MAX)*0.01f);}
  for(int i=0;i<M*K;i++)X[i]=((rand()/(float)RAND_MAX)-0.5f)*2.f;
  float*ref=(float*)malloc((size_t)M*N*4); oracle(W,X,ref,M,N,K);
  char*dW;float*dX,*dD;void*dY; int64_t ne10p=GGML_PAD(K,512);
  size_t yb=(size_t)M*ne10p*sizeof(block_q8_1)/QK8_1+256*sizeof(block_q8_1_mmq);
  CK(cudaMalloc(&dW,wb));CK(cudaMalloc(&dX,(size_t)M*K*4));CK(cudaMalloc(&dD,(size_t)M*N*4));CK(cudaMalloc(&dY,yb));
  CK(cudaMemcpy(dW,W,wb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dX,X,(size_t)M*K*4,cudaMemcpyHostToDevice));
  quantize_mmq_q8_1_cuda(dX,nullptr,dY,GGML_TYPE_Q6_K,K,K,0,0,ne10p,M,1,1,0);
  CK(cudaGetLastError());
  int nsm=0;cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0);
  CK(cudaMemset(dD,0,(size_t)M*N*4));
  launch_q6k(dW,(const int*)dY,dD,N,K,M,nsm,0); CK(cudaGetLastError());CK(cudaDeviceSynchronize());
  float*out=(float*)malloc((size_t)M*N*4); CK(cudaMemcpy(out,dD,(size_t)M*N*4,cudaMemcpyDeviceToHost));
  double e=rl2(out,ref,M*N);
  printf("rel_L2 = %.6f  %s\n  ref[0..3]=%.3f %.3f %.3f %.3f\n  out[0..3]=%.3f %.3f %.3f %.3f\n",
    e,e<0.05?"PASS":"FAIL",ref[0],ref[1],ref[2],ref[3],out[0],out[1],out[2],out[3]);
  return e<0.05?0:1;
}
