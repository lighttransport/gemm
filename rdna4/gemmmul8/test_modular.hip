#include "wmma_modular.hpp"
#include <hip/hip_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <algorithm>
#define CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess) { fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e)); exit(1); } } while(0)

static void test(ozaki2_wmma::backend mode, int m, int n, int k, int batches, bool extreme=false) {
    size_t lda=k+7, ldb=k+9, ldc=m+5;
    size_t sa=lda*m+11, sb=ldb*n+13, sc=ldc*n+17;
    std::vector<uint8_t> a(sa*batches),b(sb*batches);
    std::vector<int> ai(sa*batches),bi(sb*batches),ref(sc*batches,0);
    for (int z=0;z<batches;z++) {
        for(int row=0;row<m;row++)for(int x=0;x<k;x++) {
            size_t i=z*sa+row*lda+x;int v=extreme?(mode==ozaki2_wmma::backend::fp8?32:-128):((row*13+x*17+z*7+(x*row)%19)%31)-15;ai[i]=v;
            a[i]=mode==ozaki2_wmma::backend::fp8?__hip_fp8_e4m3(float(v)).__x:uint8_t(int8_t(v));
        }
        for(int col=0;col<n;col++)for(int x=0;x<k;x++) {
            size_t i=z*sb+col*ldb+x;int v=extreme?(mode==ozaki2_wmma::backend::fp8?32:-128):((col*7+x*11+z*3+(x*col)%23)%31)-15;bi[i]=v;
            b[i]=mode==ozaki2_wmma::backend::fp8?__hip_fp8_e4m3(float(v)).__x:uint8_t(int8_t(v));
        }
        for(int col=0;col<n;col++)for(int row=0;row<m;row++)for(int x=0;x<k;x++)
            ref[z*sc+col*ldc+row]+=ai[z*sa+row*lda+x]*bi[z*sb+col*ldb+x];
    }
    uint8_t *da,*db;void *dc;
    CHECK(hipMalloc(&da,a.size()));CHECK(hipMalloc(&db,b.size()));CHECK(hipMalloc(&dc,sc*batches*4));
    CHECK(hipMemcpy(da,a.data(),a.size(),hipMemcpyHostToDevice));CHECK(hipMemcpy(db,b.data(),b.size(),hipMemcpyHostToDevice));
    CHECK(hipMemset(dc,0,sc*batches*4));
    for(int repeat=1;repeat<=(extreme?1:2);repeat++) {
        CHECK(ozaki2_wmma::product(mode,nullptr,m,n,k,da,lda,sa,db,ldb,sb,dc,ldc,sc,batches,repeat==2));
        CHECK(hipDeviceSynchronize());std::vector<int32_t> raw(sc*batches);CHECK(hipMemcpy(raw.data(),dc,raw.size()*4,hipMemcpyDeviceToHost));
        for(int z=0;z<batches;z++)for(int col=0;col<n;col++)for(int row=0;row<m;row++) {
            size_t i=z*sc+col*ldc+row;int got;
            if(mode==ozaki2_wmma::backend::fp8) {float f;__builtin_memcpy(&f,&raw[i],4);got=int(f);if(f!=float(got))exit(1);} else got=raw[i];
            if(got!=repeat*ref[i]){fprintf(stderr,"mismatch mode=%d row=%d col=%d batch=%d k=%d got=%d expected=%d\n",int(mode),row,col,z,k,got,repeat*ref[i]);exit(1);}
        }
    }
    CHECK(hipFree(da));CHECK(hipFree(db));CHECK(hipFree(dc));
    printf("PASS mode=%s M=%d N=%d K=%d batch=%d\n",mode==ozaki2_wmma::backend::fp8?"fp8":"int8",m,n,k,batches);
}
int main(){
    for(auto mode:{ozaki2_wmma::backend::int8,ozaki2_wmma::backend::fp8}) {
        test(mode,17,33,19,3);test(mode,65,35,257,2);test(mode,3,5,8192,1);
        int max=mode==ozaki2_wmma::backend::fp8?ozaki2_wmma::max_k_fp8:ozaki2_wmma::max_k_int8;
        test(mode,2,3,max,1,true);
        for(int v=0;v<32;v++){ozaki2_wmma::set_variant(v);test(mode,65,35,257,2);}
        ozaki2_wmma::set_variant(-1);
        if(ozaki2_wmma::product(mode,nullptr,1,1,max+1,nullptr,max+1,0,nullptr,max+1,0,nullptr,1,0,1,false)!=hipErrorInvalidValue)return 1;
    }
}
