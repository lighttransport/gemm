// Root-cause probe: vendored llama mul_mat_vec_q (IQ3_S, ncols_dst=1 DECODE) vs a
// CPU-F32 oracle built from llama's dequantize_row_iq3_s.  If this PASSES, the
// vendored kernel is correct and the in-model failure is an integration/weight issue;
// if it FAILS, the kernel itself is wrong for IQ3_S.
//
// build:
//  nvcc -arch=sm_120a -O3 -std=c++17 -I ../mmq_vendor \
//    -I <llama>/ggml/src/ggml-cuda -I <llama>/ggml/include -I <llama>/ggml/src \
//    mmvq_iq3s_vendor_test.cu -o mmvq_iq3s_test
//  ./mmvq_iq3s_test [N K]

#include "mmvq.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>

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
static uint16_t float_to_half_h(float f){
    uint32_t fb; memcpy(&fb,&f,4); uint32_t sgn=(fb>>16)&0x8000,exp=((fb>>23)&0xff),man=fb&0x7fffff;
    if(exp<=112) return (uint16_t)sgn; if(exp>=143) return (uint16_t)(sgn|0x7c00);
    return (uint16_t)(sgn|((exp-112)<<10)|(man>>13));
}

// block_iq3_s = d(2) qs[64] qh[8] signs[32] scales[4] = 110 bytes
#define IQ3S_BYTES 110
static uint32_t h_grid[512];
static const uint8_t kmask[8] = {1,2,4,8,16,32,64,128};
static void decode_block_iq3_s(const uint8_t* bp, float* y){
    float d = half_to_float_h(*(const uint16_t*)bp);
    const uint8_t* qs = bp + 2;
    const uint8_t* qh = bp + 2 + 64;
    const uint8_t* signs = bp + 2 + 64 + 8;
    const uint8_t* sc = bp + 2 + 64 + 8 + 32;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        float db1 = d*(1 + 2*(sc[ib32/2] & 0xf));
        float db2 = d*(1 + 2*(sc[ib32/2] >> 4));
        for (int l=0;l<4;l++){
            const uint8_t* g1=(const uint8_t*)(h_grid + (qs[2*l+0] | ((qh[0]<<(8-2*l))&256)));
            const uint8_t* g2=(const uint8_t*)(h_grid + (qs[2*l+1] | ((qh[0]<<(7-2*l))&256)));
            for(int j=0;j<4;j++){
                y[j+0]=db1*g1[j]*((signs[l]&kmask[j+0])?-1.f:1.f);
                y[j+4]=db1*g2[j]*((signs[l]&kmask[j+4])?-1.f:1.f);
            }
            y+=8;
        }
        qs+=8; signs+=4;
        for (int l=0;l<4;l++){
            const uint8_t* g1=(const uint8_t*)(h_grid + (qs[2*l+0] | ((qh[1]<<(8-2*l))&256)));
            const uint8_t* g2=(const uint8_t*)(h_grid + (qs[2*l+1] | ((qh[1]<<(7-2*l))&256)));
            for(int j=0;j<4;j++){
                y[j+0]=db2*g1[j]*((signs[l]&kmask[j+0])?-1.f:1.f);
                y[j+4]=db2*g2[j]*((signs[l]&kmask[j+4])?-1.f:1.f);
            }
            y+=8;
        }
        qh+=2; qs+=8; signs+=4;
    }
}
static void cpu_oracle(const uint8_t *W,const float *X,float *dst,int N,int K){
    int nb=K/256; size_t rb=(size_t)nb*IQ3S_BYTES;
    float *wf=(float*)malloc((size_t)K*sizeof(float));
    for(int n=0;n<N;n++){
        for(int b=0;b<nb;b++) decode_block_iq3_s(W+(size_t)n*rb+b*IQ3S_BYTES, wf+b*256);
        double s=0; for(int k=0;k<K;k++) s+=(double)wf[k]*(double)X[k]; dst[n]=(float)s;
    }
    free(wf);
}
static double rel_l2(const float*a,const float*b,int n){ double nu=0,de=0; for(int i=0;i<n;i++){double d=(double)a[i]-b[i]; nu+=d*d; de+=(double)b[i]*b[i];} return sqrt(nu/(de+1e-30)); }

__global__ void quant_q8_1(const float* __restrict__ x, void* __restrict__ vy, const int n){
    const int wpb=blockDim.x/WARP_SIZE;
    const int wid=blockIdx.x*wpb + threadIdx.x/WARP_SIZE;
    const int lane=threadIdx.x%WARP_SIZE;
    if(wid>=n/QK8_1) return;
    const float xi=x[wid*QK8_1+lane];
    float amax=fabsf(xi), sum=xi;
    amax=warp_reduce_max<QK8_1>(amax); sum=warp_reduce_sum<QK8_1>(sum);
    const float d=amax/127.0f;
    const int8_t q = amax==0.0f?0:roundf(xi/d);
    block_q8_1* y=(block_q8_1*)vy;
    y[wid].qs[lane]=q;
    if(lane==0) y[wid].ds=make_half2(d,sum);
}
__global__ void __launch_bounds__(calc_nwarps(GGML_TYPE_IQ3_S,1,get_device_table_id())*32,1)
mmvq_wrap(const void* vx,const void* vy,float* dst,
          const uint32_t ncols_x,const uint3 nch_y,const uint32_t srx,
          const uint32_t scy,const uint32_t scd,const uint3 chr,const uint3 smr){
    ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q<GGML_TYPE_IQ3_S,1,false,false>(vx,vy,nullptr,fusion,dst,
        ncols_x,nch_y,srx,scy,scd,chr,0u,0u,0u,smr,0u,0u,0u,0u);
}

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):5376, K=argc>2?atoi(argv[2]):21504;
    if(K%256){ printf("need K%%256==0\n"); return 1; }
    printf("M=1 (decode) N=%d K=%d (vendored llama mul_mat_vec_q IQ3_S)\n",N,K);
    CK_CHECK(cudaMemcpyFromSymbol(h_grid, iq3s_grid, sizeof(h_grid)));

    int nb=K/256; size_t wbytes=(size_t)N*nb*IQ3S_BYTES;
    uint8_t *W=(uint8_t*)malloc(wbytes); float *X=(float*)malloc((size_t)K*sizeof(float));
    srand(1234);
    for(size_t i=0;i<wbytes;i++) W[i]=rand()&0xff;
    for(int n=0;n<N;n++) for(int b=0;b<nb;b++){ float d=0.01f+(rand()/(float)RAND_MAX)*0.05f;
        *(uint16_t*)(W+((size_t)n*nb+b)*IQ3S_BYTES)=float_to_half_h(d); }
    for(int i=0;i<K;i++) X[i]=((rand()/(float)RAND_MAX)-0.5f)*2.0f;

    float *ref=(float*)malloc((size_t)N*sizeof(float));
    cpu_oracle(W,X,ref,N,K);

    char *dW; float *dX,*dDst; void *dY;
    size_t ybytes=(size_t)(K/QK8_1)*sizeof(block_q8_1);
    CK_CHECK(cudaMalloc(&dW,wbytes)); CK_CHECK(cudaMalloc(&dX,(size_t)K*sizeof(float)));
    CK_CHECK(cudaMalloc(&dDst,(size_t)N*sizeof(float))); CK_CHECK(cudaMalloc(&dY,ybytes));
    CK_CHECK(cudaMemcpy(dW,W,wbytes,cudaMemcpyHostToDevice));
    CK_CHECK(cudaMemcpy(dX,X,(size_t)K*sizeof(float),cudaMemcpyHostToDevice));

    quant_q8_1<<<(K/QK8_1+7)/8, 256>>>(dX,dY,K);
    CK_CHECK(cudaGetLastError());
    CK_CHECK(cudaMemset(dDst,0,(size_t)N*sizeof(float)));

    uint3 one=init_fastdiv_values(1);
    uint32_t srx=K/256, scd=N;
    dim3 grid(N,1,1), block(32, calc_nwarps(GGML_TYPE_IQ3_S,1,MMVQ_PARAMETERS_GENERIC), 1);
    mmvq_wrap<<<grid,block>>>(dW,dY,dDst,(uint32_t)K,one,srx,0u,scd,one,one);
    CK_CHECK(cudaGetLastError()); CK_CHECK(cudaDeviceSynchronize());

    float *out=(float*)malloc((size_t)N*sizeof(float));
    CK_CHECK(cudaMemcpy(out,dDst,(size_t)N*sizeof(float),cudaMemcpyDeviceToHost));
    double e=rel_l2(out,ref,N);
    printf("vendor mul_mat_vec_q IQ3_S vs CPU-F32 oracle: rel_L2 = %.6f  %s\n", e, e<0.05?"PASS":"FAIL");
    printf("  ref[0..3]=%.3f %.3f %.3f %.3f\n  out[0..3]=%.3f %.3f %.3f %.3f\n",
        ref[0],ref[1],ref[2],ref[3],out[0],out[1],out[2],out[3]);
    return e<0.05?0:1;
}
