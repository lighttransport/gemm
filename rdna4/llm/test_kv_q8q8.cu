#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "cpy-utils.cuh"
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
typedef unsigned short half_raw;
// RUNNER_DECLARATIONS
#define CHECK(call) do { hipError_t e=(call); if(e!=hipSuccess) { \
    fprintf(stderr,"%s: %s\n",#call,hipGetErrorString(e)); exit(2); } } while(0)

__global__ void reference_quant(const float *x, block_q8_0 *y, int blocks) {
    int b=blockIdx.x*blockDim.x+threadIdx.x;
    if(b<blocks) quantize_f32_q8_0_block(x+32*b,y+b);
}

int main() {
    size_t checked=0, errors=0, code_errors=0, scale_errors=0, pack_errors=0;
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> dist(-5,5);
    for(int D: {32,64,128,256}) for(int M: {1,31,32,33,511,512,513}) {
        const int H=3,P=7,T=P+M,dim=H*D,n=M*dim,N=T*dim,S=N/32,G=256;
        std::vector<float> input(2*n);
        for(int i=0;i<2*n;++i) {
            float x=dist(rng);
            switch((i/32)%5) {
                case 0: x=0; break;
                case 1: x*=1.e-10f; break;
                case 2: x*=5000; break;
                case 3: x=(i%32==0)?127.0f:((i&1)?1:-1)*((i%31)+0.5f); break;
            }
            input[i]=x;
        }
        float *x; half_raw *ks,*vs; signed char *kc,*vc; half_raw *packed;
        block_q8_0 *ref;
        CHECK(hipMalloc(&x,2*n*sizeof(float)));
        CHECK(hipMalloc(&ks,(S+G)*sizeof(half_raw))); CHECK(hipMalloc(&vs,(S+G)*sizeof(half_raw)));
        CHECK(hipMalloc(&kc,N+G)); CHECK(hipMalloc(&vc,N+G));
        CHECK(hipMalloc(&packed,(N+G)*sizeof(half_raw)));
        CHECK(hipMalloc(&ref,2*n/32*sizeof(block_q8_0)));
        std::vector<half_raw> init(S+G,0x3c00);
        CHECK(hipMemcpy(x,input.data(),2*n*sizeof(float),hipMemcpyHostToDevice));
        CHECK(hipMemcpy(ks,init.data(),init.size()*sizeof(half_raw),hipMemcpyHostToDevice));
        CHECK(hipMemcpy(vs,init.data(),init.size()*sizeof(half_raw),hipMemcpyHostToDevice));
        CHECK(hipMemset(kc,85,N+G)); CHECK(hipMemset(vc,85,N+G));
        hipLaunchKernelGGL(kv_cache_store_q8q8_batch,dim3(M*H),dim3(256),0,0,kc,vc,ks,vs,x,x+n,H,D,P,M);
        hipLaunchKernelGGL(reference_quant,dim3((2*n/32+255)/256),dim3(256),0,0,x,ref,2*n/32);
        CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
        std::vector<block_q8_0> oracle(2*n/32);
        CHECK(hipMemcpy(oracle.data(),ref,oracle.size()*sizeof(block_q8_0),hipMemcpyDeviceToHost));
        for(int v=0;v<2;++v) {
            signed char *cache=v?vc:kc; half_raw *scale=v?vs:ks;
            std::vector<signed char> codes(N+G); std::vector<half_raw> scales(S+G);
            CHECK(hipMemcpy(codes.data(),cache,N+G,hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(scales.data(),scale,(S+G)*sizeof(half_raw),hipMemcpyDeviceToHost));
            for(int i=0;i<N+G;++i) {
                int j=i-P*dim;
                int expected=(j>=0&&j<n)?oracle[v*n/32+j/32].qs[j%32]:85;
                if(codes[i]!=expected) {++errors; ++code_errors; if(code_errors<4 && j>=0 && j<n) {
                    float am=0; for(int k=0;k<32;++k) am=fmaxf(am,fabsf(input[v*n+(j/32)*32+k]));
                    float xx=input[v*n+j],dd=am/127.0f,ii=1.0f/dd;
                    printf("code D=%d M=%d i=%d got=%d ref=%d x=%.9g max=%.9g d=%.9g inv=%.9g product=%.9g host=%.9g\n",D,M,i,(int)codes[i],expected,xx,am,dd,ii,xx*ii,roundf(xx*ii));
                }} ++checked;
            }
            for(int i=0;i<S+G;++i) {
                int b=i-P*dim/32;
                half_raw expected=0x3c00;
                if(b>=0&&b<n/32) memcpy(&expected,&oracle[v*n/32+b].d,2);
                if(scales[i]!=expected) {++errors; ++scale_errors; if(scale_errors<4) printf("scale D=%d M=%d i=%d got=%x ref=%x\n",D,M,i,scales[i],expected);} ++checked;
            }
            for(int transpose=0;transpose<2;++transpose) {
                CHECK(hipMemset(packed,85,(N+G)*2));
                if(transpose) {
                    hipLaunchKernelGGL(pack_kv_q8q4_f16,dim3(T,H),dim3(D),0,0,packed,cache,(const float *)scale,T,H,D,-1);
                } else {
                    hipLaunchKernelGGL(unpack_kv_q8q4_decode_f16,dim3((N+255)/256),dim3(256),0,0,packed,cache,(const float *)scale,T,H,D,-1);
                }
                CHECK(hipGetLastError());
                std::vector<half_raw> values(N+G);
                CHECK(hipMemcpy(values.data(),packed,(N+G)*2,hipMemcpyDeviceToHost));
                for(int i=0;i<N;++i) {
                    int t=i/dim,h=i%dim/D,d=i%D;
                    int out=transpose?(h*T+t)*D+d:i;
                    __half sh; memcpy(&sh,&scales[i/32],2);
                    __half expected=__float2half((float)codes[i]*__half2float(sh));
                    half_raw bits; memcpy(&bits,&expected,2);
                    if(values[out]!=bits) {++errors; ++pack_errors; if(pack_errors<4) printf("pack D=%d M=%d i=%d got=%x ref=%x\n",D,M,i,values[out],bits);} ++checked;
                }
                for(int i=N;i<N+G;++i) {errors+=values[i]!=0x5555; ++checked;}
            }
        }
        CHECK(hipFree(x)); CHECK(hipFree(ks)); CHECK(hipFree(vs));
        CHECK(hipFree(kc)); CHECK(hipFree(vc)); CHECK(hipFree(packed)); CHECK(hipFree(ref));
    }
    printf("Q8/Q8 cache replay: checked=%zu mismatches=%zu\n",checked,errors);
    printf("codes=%zu scales=%zu pack=%zu\n",code_errors,scale_errors,pack_errors);
    return errors?1:0;
}
