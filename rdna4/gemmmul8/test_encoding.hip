// Independent checks of GEMMul8's device residue encoders.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#include "vendor/GEMMul8/src/oz2/mod/mod_core.hpp"
#include "vendor/GEMMul8/src/oz2/common/make_f8.hpp"
#include "vendor/GEMMul8/src/oz2/common/fp8_awe_lut.hpp"
#pragma clang diagnostic pop
#define HIP(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);}}while(0)
using gemmul8::Backend;
namespace plan=gemmul8::common::fp8_plan;

template<unsigned index> __global__ void encode_i8(const int32_t *x,int8_t *y,int n){
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n)y[i]=int8_t(gemmul8::mod::calc_mod<Backend::INT8,index>(x[i]));
}
template<int p> __global__ void encode_fp8(uint32_t *out){
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<p)out[i]=uint32_t(plan::encoding_device<p>.value[i]);
}
static int decode(uint32_t bits){
    bits&=255;if(!(bits&127))return 0;int exponent=int((bits>>3)&15)-10;
    double x=std::ldexp(double(8+(bits&7)),exponent);if(bits&128)x=-x;
    if(x!=int(x)||fabs(x)>32||(fabs(x)>16&&(int(fabs(x))&1))) {fprintf(stderr,"non-exact limb 0x%x\n",bits);exit(1);}return int(x);
}
template<int p> static void test_fp8(){
    constexpr auto scheme=plan::scheme<p>;static_assert(scheme.w!=0);
    uint32_t *device;std::vector<uint32_t> enc(p);HIP(hipMalloc(&device,enc.size()*4));
    hipLaunchKernelGGL((encode_fp8<p>),dim3((p+255)/256),dim3(256),0,0,device);HIP(hipGetLastError());HIP(hipMemcpy(enc.data(),device,enc.size()*4,hipMemcpyDeviceToHost));
    for(int i=0;i<p;i++)for(int trial=0;trial<4;trial++){
        int j=(i*37+trial*103+17)%p,a=i-p/2,b=j-p/2;int64_t sum=0;
        for(unsigned limb=0;limb<scheme.products;limb++)sum+=int64_t(scheme.coefficient[limb])*decode(enc[i]>>(8*limb))*decode(enc[j]>>(8*limb));
        int64_t rem=(sum-int64_t(a)*b)%p;if(rem<0)rem+=p;
        if(rem){fprintf(stderr,"FP8 residue encoding mismatch p=%d a=%d b=%d remainder=%lld\n",p,a,b,(long long)rem);exit(1);}
    }
    HIP(hipFree(device));printf("PASS FP8 residue encoder p=%d cases=%d\n",p,p*4);
}
template<unsigned index> static void test_i8(){
    constexpr int p=gemmul8::common::table::moduli<Backend::INT8,index>;
    std::vector<int32_t>in;for(int i=-4096;i<=4096;i++)in.push_back(i);in.push_back(INT32_MIN);in.push_back(INT32_MAX);
    int32_t *di;int8_t *doo;std::vector<int8_t>out(in.size());HIP(hipMalloc(&di,in.size()*4));HIP(hipMalloc(&doo,out.size()));
    HIP(hipMemcpy(di,in.data(),in.size()*4,hipMemcpyHostToDevice));hipLaunchKernelGGL((encode_i8<index>),dim3((in.size()+255)/256),dim3(256),0,0,di,doo,int(in.size()));HIP(hipGetLastError());HIP(hipMemcpy(out.data(),doo,out.size(),hipMemcpyDeviceToHost));
    for(size_t i=0;i<in.size();i++){int64_t delta=int64_t(out[i])-in[i];if(delta%p){fprintf(stderr,"INT8 encoding mismatch modulus=%d input=%d output=%d\n",p,in[i],int(out[i]));exit(1);}}
    HIP(hipFree(di));HIP(hipFree(doo));printf("PASS INT8 residue encoder p=%d cases=%zu\n",p,in.size());
}
int main(){
    test_i8<0>();test_i8<1>();test_i8<2>();test_i8<3>();test_i8<4>();test_i8<5>();test_i8<6>();test_i8<7>();test_i8<8>();test_i8<9>();
    test_i8<10>();test_i8<11>();test_i8<12>();test_i8<13>();test_i8<14>();test_i8<15>();test_i8<16>();test_i8<17>();test_i8<18>();test_i8<19>();
    test_fp8<1283>();test_fp8<343>();test_fp8<11583>();
}
