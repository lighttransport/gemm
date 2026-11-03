// Isolated test of GEMMul8's actual device CRT tree against GMP/MPFR.
#include <hip/hip_runtime.h>
#include <gmp.h>
#include <mpfr.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "vendor/GEMMul8/src/oz2/common/fp8_plan.hpp"
#include "vendor/GEMMul8/src/oz2/undo_scaling/crt_tree.hpp"
#define HIP(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);}}while(0)
using namespace gemmul8::undo_scaling;
template<bool fp8,unsigned count> struct moduli {
    static constexpr uint32_t get(unsigned i) {
        if constexpr(fp8)return uint32_t(gemmul8::common::fp8_plan::modulus(count,i,false));
        constexpr uint32_t p[20]={256,255,253,251,247,241,239,233,229,227,223,217,211,199,197,193,191,181,179,173};
        return p[i];
    }
};
template<unsigned count> struct input {
    static constexpr bool complex=false;
    const int32_t *values;
    template<unsigned i> __device__ int2 load() const{return {values[i],0};}
};
template<bool fp8,unsigned count>
__global__ void reconstruct(const int32_t *residues,double *out,int samples) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=samples)return;
    using root=crt::node<moduli<fp8,count>,0,count>;
    auto x=root::reconstruct(input<count>{residues+size_t(i)*count});
    out[i]=crt::centered_double(x.re,root::modulus);
}
template<bool fp8,unsigned count> static void test(){
    constexpr int samples=1024;std::vector<int32_t>r(size_t(samples)*count);std::vector<double>expected(samples),out(samples);
    mpz_t product,half,x;mpz_inits(product,half,x,nullptr);mpz_set_ui(product,1);
    for(unsigned j=0;j<count;j++)mpz_mul_ui(product,product,moduli<fp8,count>::get(j));
    mpz_fdiv_q_2exp(half,product,1);
    gmp_randstate_t rng;gmp_randinit_default(rng);gmp_randseed_ui(rng,12345+count+fp8);
    mpfr_t reference;mpfr_init2(reference,512);
    for(int i=0;i<samples;i++){
        if(i==0)mpz_set_ui(x,0);else if(i==1)mpz_set_ui(x,1);else if(i==2)mpz_set_si(x,-1);
        else if(i==3)mpz_set(x,half);else if(i==4){mpz_neg(x,half);if(mpz_even_p(product))mpz_add_ui(x,x,1);}
        else {mpz_urandomm(x,rng,product);if(mpz_cmp(x,half)>0)mpz_sub(x,x,product);}
        for(unsigned j=0;j<count;j++){
            uint32_t p=moduli<fp8,count>::get(j);uint32_t v=uint32_t(mpz_fdiv_ui(x,p));
            r[size_t(i)*count+j]=v>p/2?int32_t(v)-int32_t(p):int32_t(v);
        }
        mpfr_set_z(reference,x,MPFR_RNDN);expected[i]=mpfr_get_d(reference,MPFR_RNDN);
    }
    int32_t *dr;double *dout;HIP(hipMalloc(&dr,r.size()*sizeof(int32_t)));HIP(hipMalloc(&dout,out.size()*sizeof(double)));
    HIP(hipMemcpy(dr,r.data(),r.size()*sizeof(int32_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL((reconstruct<fp8,count>),dim3((samples+127)/128),dim3(128),0,0,dr,dout,samples);HIP(hipGetLastError());HIP(hipDeviceSynchronize());
    HIP(hipMemcpy(out.data(),dout,out.size()*sizeof(double),hipMemcpyDeviceToHost));
    for(int i=0;i<samples;i++)if(out[i]!=expected[i]){fprintf(stderr,"CRT mismatch backend=%s count=%u sample=%d got=%a expected=%a\n",fp8?"fp8":"int8",count,i,out[i],expected[i]);exit(1);}
    HIP(hipFree(dr));HIP(hipFree(dout));mpfr_clear(reference);gmp_randclear(rng);mpz_clears(product,half,x,nullptr);
    printf("PASS CRT %s moduli=%u samples=%d\n",fp8?"fp8":"int8",count,samples);
}
int main(){test<false,15>();test<false,20>();test<true,11>();test<true,20>();}
