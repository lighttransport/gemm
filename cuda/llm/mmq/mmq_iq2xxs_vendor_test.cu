// Phase 0 proof: vendored llama.cpp mul_mat_q (IQ2_XXS) vs CPU F32 oracle.
// Validates correctness (rel_L2 vs float ground truth) and benchmarks GFLOP/s
// against our hand kernel's known number (mma_v3 WN8/TG24 ~17374 GFLOP/s @
// M512 N8192 K5376). Compile WITH llama.cpp's quantize.cu for the activation
// quantizer. See plan: close the 2x prefill gap by porting mul_mat_q.
//
// build:
//  nvcc -arch=sm_120a -O3 -std=c++17 \
//    -I <llama>/ggml/src/ggml-cuda -I <llama>/ggml/include -I <llama>/ggml/src \
//    mmq_iq2xxs_vendor_test.cu <llama>/ggml/src/ggml-cuda/quantize.cu -o mmq_vendor_test
//  ./mmq_vendor_test [M N K]

#include "mmq.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

// host wrapper from llama's quantize.cu (compiled alongside)
void quantize_mmq_q8_1_cuda(const float * x, const int32_t * ids, void * vy, ggml_type type_src0,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// Minimal host stubs for the ggml symbols referenced by GGML_ASSERT/CUDA_CHECK
// (normally provided by ggml.c / ggml-cuda common.cu — not linked here).
extern "C" void ggml_abort(const char *file,int line,const char *fmt,...){
    fprintf(stderr,"ggml_abort %s:%d: %s\n",file,line,fmt); abort(); }
int ggml_cuda_get_device(){ int d=0; cudaGetDevice(&d); return d; }
void ggml_cuda_error(const char *stmt,const char *func,const char *file,int line,const char *msg){
    fprintf(stderr,"cuda error %s @ %s %s:%d: %s\n",stmt,func,file,line,msg); abort(); }

#define CK_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static float half_to_float_h(uint16_t h){
    uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,out;
    if(e==0){ if(m==0) out=s<<31; else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; out=(s<<31)|(e<<23)|(m<<13);} }
    else if(e==0x1f) out=(s<<31)|(0xff<<23)|(m<<13);
    else out=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&out,4); return f;
}

// IQ2_XXS host decode of one 66-byte block (256 elems) using llama's grid+ksigns.
static uint64_t h_grid[256]; static uint8_t h_ksigns[128];
static void decode_block_iq2xxs(const uint8_t *bp, float *out){
    float d = half_to_float_h(*(const uint16_t*)bp);
    const uint16_t *qs = (const uint16_t*)(bp+2);
    for(int ib=0; ib<8; ib++){
        uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
        uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
        float db = d*(0.5f+(float)(a1>>28))*0.25f;
        for(int l=0;l<4;l++){
            uint8_t idx=(a0>>(8*l))&0xff; uint64_t gv=h_grid[idx];
            uint8_t s=h_ksigns[(a1>>(7*l))&127];
            for(int j=0;j<8;j++){
                int8_t g=(int8_t)((gv>>(8*j))&0xff);
                out[ib*32+l*8+j]=db*(float)g*((s&(1<<j))?-1.0f:1.0f);
            }
        }
    }
}

static void cpu_oracle(const uint8_t *W,const float *X,float *dst,int M,int N,int K){
    int nb=K/256; size_t rb=(size_t)nb*66;
    float *wf=(float*)malloc((size_t)K*sizeof(float));
    for(int n=0;n<N;n++){
        for(int b=0;b<nb;b++) decode_block_iq2xxs(W+(size_t)n*rb+b*66, wf+b*256);
        for(int m=0;m<M;m++){ double s=0; const float*x=X+(size_t)m*K; for(int k=0;k<K;k++) s+=(double)wf[k]*(double)x[k]; dst[(size_t)m*N+n]=(float)s; }
    }
    free(wf);
}
static double rel_l2(const float*a,const float*b,int n){ double nu=0,de=0; for(int i=0;i<n;i++){double d=(double)a[i]-b[i]; nu+=d*d; de+=(double)b[i]*b[i];} return sqrt(nu/(de+1e-30)); }

// replicate launch_mul_mat_q no-stream-k dense path for IQ2_XXS, mmq_x=128
static void launch_vendor_iq2xxs(const char*W,const int*yq8,float*dst,int N,int K,int M,cudaStream_t st){
    const int cc=1200, warp_size=32;
    const int nwarps=mmq_get_nwarps_host(cc,warp_size);
    const int mmq_y=get_mmq_y_host(cc);
    const int mmq_x=128;
    size_t nbs=mmq_get_nbytes_shared<GGML_TYPE_IQ2_XXS>(mmq_x,mmq_y,cc,warp_size,nwarps);
    CK_CHECK(cudaFuncSetAttribute(mul_mat_q<GGML_TYPE_IQ2_XXS,mmq_x,false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)nbs));
    int nty=(N+mmq_y-1)/mmq_y, ntx=(M+mmq_x-1)/mmq_x;
    dim3 grid(nty,ntx,1), block(warp_size,nwarps,1);
    uint3 bp=init_fastdiv_values((uint64_t)K/256);
    uint3 ntxfd=init_fastdiv_values((uint64_t)ntx);
    uint3 one=init_fastdiv_values(1);
    int s01=K/256;            // stride_row_x (blocks per weight row)
    int stride_col_dst=N;     // == nrows_dst (=> dst[token*N+row])
    mul_mat_q<GGML_TYPE_IQ2_XXS,mmq_x,false><<<grid,block,nbs,st>>>(
        W,yq8,nullptr,nullptr,dst,nullptr,
        bp, N, M, s01, M, stride_col_dst,
        one, one, 0,0,0,
        one, one, 0,0,0,
        ntxfd);
}

// stream-k path (llama default config)
static float *g_tmp_fixup=nullptr; static size_t g_tmp_cap=0;
static void launch_vendor_iq2xxs_sk(const char*W,const int*yq8,float*dst,int N,int K,int M,int nsm,cudaStream_t st){
    const int cc=1200, warp_size=32;
    const int nwarps=mmq_get_nwarps_host(cc,warp_size);
    const int mmq_y=get_mmq_y_host(cc);
    const int mmq_x=128;
    size_t nbs=mmq_get_nbytes_shared<GGML_TYPE_IQ2_XXS>(mmq_x,mmq_y,cc,warp_size,nwarps);
    CK_CHECK(cudaFuncSetAttribute(mul_mat_q<GGML_TYPE_IQ2_XXS,mmq_x,false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)nbs));
    int nty=(N+mmq_y-1)/mmq_y, ntx=(M+mmq_x-1)/mmq_x;
    int ntiles_dst=ntx*nty;
    int tiles_nwaves=(ntiles_dst+nsm-1)/nsm;
    int tiles_eff=100*ntiles_dst/(nsm*tiles_nwaves);
    int sk_x=(tiles_eff>=90)?ntiles_dst:nsm;
    bool fixup_needed=(ntiles_dst % sk_x)!=0;
    uint3 bp=init_fastdiv_values((uint64_t)K/256);
    uint3 ntxfd=init_fastdiv_values((uint64_t)ntx);
    uint3 one=init_fastdiv_values(1);
    int s01=K/256, stride_col_dst=N;
    if(fixup_needed){ size_t need=(size_t)sk_x*mmq_x*mmq_y*sizeof(float);
        if(need>g_tmp_cap){ if(g_tmp_fixup) cudaFree(g_tmp_fixup); CK_CHECK(cudaMalloc(&g_tmp_fixup,need)); g_tmp_cap=need; } }
    dim3 grid(sk_x,1,1), block(warp_size,nwarps,1);
    mul_mat_q<GGML_TYPE_IQ2_XXS,mmq_x,false><<<grid,block,nbs,st>>>(
        W,yq8,nullptr,nullptr,dst, fixup_needed?g_tmp_fixup:nullptr,
        bp, N, M, s01, M, stride_col_dst, one, one, 0,0,0, one, one, 0,0,0, ntxfd);
    if(fixup_needed){
        dim3 fg(sk_x, mmq_y/warp_size, 1), fb(warp_size, nwarps/2, 1);
        mul_mat_q_stream_k_fixup<GGML_TYPE_IQ2_XXS,mmq_x,false><<<fg,fb,0,st>>>(
            nullptr,nullptr,dst,g_tmp_fixup, bp, N, M, stride_col_dst, one, 0, one, 0, ntxfd);
    }
}

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):512, N=argc>2?atoi(argv[2]):8192, K=argc>3?atoi(argv[3]):5376;
    if(N%128||K%256){ printf("need N%%128==0 K%%256==0\n"); return 1; }
    printf("M=%d N=%d K=%d (vendored llama mul_mat_q IQ2_XXS)\n",M,N,K);
    // pull llama's grid + ksigns to host for the oracle
    CK_CHECK(cudaMemcpyFromSymbol(h_grid, iq2xxs_grid, sizeof(h_grid)));
    CK_CHECK(cudaMemcpyFromSymbol(h_ksigns, ksigns_iq2xs, sizeof(h_ksigns)));

    int nb=K/256; size_t wbytes=(size_t)N*nb*66;
    uint8_t *W=(uint8_t*)malloc(wbytes); float *X=(float*)malloc((size_t)M*K*sizeof(float));
    srand(1234);
    for(size_t i=0;i<wbytes;i++) W[i]=rand()&0xff;
    for(int n=0;n<N;n++) for(int b=0;b<nb;b++){ float d=0.01f+(rand()/(float)RAND_MAX)*0.05f; *(uint16_t*)(W+((size_t)n*nb+b)*66)= (uint16_t)0; // placeholder
        // store d as half
        uint32_t fb; memcpy(&fb,&d,4); uint32_t sgn=(fb>>16)&0x8000,exp=((fb>>23)&0xff),man=fb&0x7fffff; uint16_t hh;
        if(exp<=112) hh=(uint16_t)sgn; else if(exp>=143) hh=(uint16_t)(sgn|0x7c00); else hh=(uint16_t)(sgn|((exp-112)<<10)|(man>>13));
        *(uint16_t*)(W+((size_t)n*nb+b)*66)=hh; }
    for(int i=0;i<M*K;i++) X[i]=((rand()/(float)RAND_MAX)-0.5f)*2.0f;

    float *ref=(float*)malloc((size_t)M*N*sizeof(float));
    cpu_oracle(W,X,ref,M,N,K);

    char *dW; float *dX,*dDst; void *dY;
    int64_t ne10p=GGML_PAD(K,512);
    size_t ybytes=(size_t)M*ne10p*sizeof(block_q8_1)/QK8_1 + 256*sizeof(block_q8_1_mmq);
    CK_CHECK(cudaMalloc(&dW,wbytes)); CK_CHECK(cudaMalloc(&dX,(size_t)M*K*sizeof(float)));
    CK_CHECK(cudaMalloc(&dDst,(size_t)M*N*sizeof(float))); CK_CHECK(cudaMalloc(&dY,ybytes));
    CK_CHECK(cudaMemcpy(dW,W,wbytes,cudaMemcpyHostToDevice));
    CK_CHECK(cudaMemcpy(dX,X,(size_t)M*K*sizeof(float),cudaMemcpyHostToDevice));

    // quantize activations -> block_q8_1_mmq (ne00=K, s01=K, ne0=ne10_padded, ne1=M)
    quantize_mmq_q8_1_cuda(dX,nullptr,dY,GGML_TYPE_IQ2_XXS, K, K,0,0, ne10p, M,1,1, 0);
    CK_CHECK(cudaGetLastError());
    CK_CHECK(cudaMemset(dDst,0,(size_t)M*N*sizeof(float)));
    launch_vendor_iq2xxs(dW,(const int*)dY,dDst,N,K,M,0);
    CK_CHECK(cudaGetLastError()); CK_CHECK(cudaDeviceSynchronize());

    float *out=(float*)malloc((size_t)M*N*sizeof(float));
    CK_CHECK(cudaMemcpy(out,dDst,(size_t)M*N*sizeof(float),cudaMemcpyDeviceToHost));
    double e=rel_l2(out,ref,M*N);
    printf("vendor mul_mat_q vs CPU-F32 oracle: rel_L2 = %.6f  %s\n", e, e<0.05?"PASS":"FAIL");
    printf("  ref[0..3]=%.3f %.3f %.3f %.3f\n  out[0..3]=%.3f %.3f %.3f %.3f\n",
        ref[0],ref[1],ref[2],ref[3],out[0],out[1],out[2],out[3]);

    // timing (no-stream-k)
    int iters=300; cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for(int i=0;i<iters;i++) launch_vendor_iq2xxs(dW,(const int*)dY,dDst,N,K,M,0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1); ms/=iters;
    double flop=2.0*M*N*K;
    printf("  vendor(no-streamk): %.3f ms  %.1f GFLOP/s  (our mma_v3 WN8/TG24 ref ~17374)\n", ms, flop/(ms*1e6));

    // stream-k path (llama default)
    int nsm=0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    CK_CHECK(cudaMemset(dDst,0,(size_t)M*N*sizeof(float)));
    launch_vendor_iq2xxs_sk(dW,(const int*)dY,dDst,N,K,M,nsm,0);
    CK_CHECK(cudaGetLastError()); CK_CHECK(cudaDeviceSynchronize());
    CK_CHECK(cudaMemcpy(out,dDst,(size_t)M*N*sizeof(float),cudaMemcpyDeviceToHost));
    double esk=rel_l2(out,ref,M*N);
    printf("vendor stream-k vs CPU-F32 oracle: rel_L2 = %.6f  %s\n", esk, esk<0.05?"PASS":"FAIL");
    cudaEventRecord(t0);
    for(int i=0;i<iters;i++) launch_vendor_iq2xxs_sk(dW,(const int*)dY,dDst,N,K,M,nsm,0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float mssk; cudaEventElapsedTime(&mssk,t0,t1); mssk/=iters;
    printf("  vendor(stream-k):   %.3f ms  %.1f GFLOP/s  (nsm=%d)\n", mssk, flop/(mssk*1e6), nsm);
    return e<0.05?0:1;
}
