// RDNA4 modular-product building block. See README.md for scope.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
using v2i = int __attribute__((ext_vector_type(2)));
using v8i = int __attribute__((ext_vector_type(8)));
using v8f = float __attribute__((ext_vector_type(8)));
#define HIP_CHECK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);} }while(0)
// A and B are rows of K; output is row major M x N.
// Exactness requires K*max(abs(A))*max(abs(B)) <= 2^24 (FP8),
// or <= INT32_MAX (INT8). The caller must chunk larger reductions.
template<bool FP8>
__global__ void product(const uint8_t *a,const uint8_t *b,double *c,int m,int n,int k) {
    int lane=threadIdx.x, idx=lane&15, half=lane>>4;
    int row=blockIdx.y*16, col=blockIdx.x*16;
    v8i ci={}; v8f cf={};
    for(int kk=0;kk<k;kk+=16){
        v2i av={},bv={};
        for(int t=0;t<8;t++){
            int kp=kk+half*8+t;
            uint32_t x=(row+idx<m && kp<k)?a[size_t(row+idx)*k+kp]:0;
            uint32_t y=(col+idx<n && kp<k)?b[size_t(col+idx)*k+kp]:0;
            av[t/4]=int(uint32_t(av[t/4])|(x<<(8*(t%4))));
            bv[t/4]=int(uint32_t(bv[t/4])|(y<<(8*(t%4))));
        }
        if constexpr(FP8) cf=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av,bv,cf);
        else ci=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true,av,true,bv,ci,false);
    }
    for(int t=0;t<8;t++){
        int r=row+half*8+t, s=col+idx;
        if(r<m && s<n)c[size_t(r)*n+s]=FP8?double(cf[t]):double(ci[t]);
    }
}
static void blas_check(rocblas_status s){if(s!=rocblas_status_success){fprintf(stderr,"rocBLAS status %d\n",int(s));exit(1);}}
int main(int argc,char **argv){
    int m=argc>1?atoi(argv[1]):257,n=argc>2?atoi(argv[2]):193,k=argc>3?atoi(argv[3]):255;
    if(m<=0||n<=0||k<=0||k>65536){fprintf(stderr,"usage: bench_wmma [M N K], positive dimensions, K <= 65536\n");return 2;}
    std::vector<double>a(size_t(m)*k),b(size_t(n)*k),out(size_t(m)*n),ref(out.size());
    std::vector<uint8_t>ai(a.size()),bi(b.size()),af(a.size()),bf(b.size());
    // Small integers are exactly representable in E4M3; asymmetric patterns
    // exercise every fragment coordinate, including negative values.
    for(size_t i=0;i<a.size();i++){int x=int((i*17+i/7)%31)-15;a[i]=x;ai[i]=uint8_t(int8_t(x));af[i]=__hip_fp8_e4m3(float(x)).__x;}
    for(size_t i=0;i<b.size();i++){int x=int((i*11+i/13)%31)-15;b[i]=x;bi[i]=uint8_t(int8_t(x));bf[i]=__hip_fp8_e4m3(float(x)).__x;}
    uint8_t *da,*db;double *dc,*dra,*drb,*dref;
    HIP_CHECK(hipMalloc(&da,a.size()));HIP_CHECK(hipMalloc(&db,b.size()));HIP_CHECK(hipMalloc(&dc,out.size()*8));
    HIP_CHECK(hipMalloc(&dra,a.size()*8));HIP_CHECK(hipMalloc(&drb,b.size()*8));HIP_CHECK(hipMalloc(&dref,out.size()*8));
    HIP_CHECK(hipMemcpy(dra,a.data(),a.size()*8,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(drb,b.data(),b.size()*8,hipMemcpyHostToDevice));
    rocblas_handle h;blas_check(rocblas_create_handle(&h));double one=1,zero=0;
    auto native=[&]{blas_check(rocblas_dgemm(h,rocblas_operation_transpose,rocblas_operation_none,n,m,k,&one,drb,k,dra,k,&zero,dref,n));};
    native();HIP_CHECK(hipDeviceSynchronize());HIP_CHECK(hipMemcpy(ref.data(),dref,out.size()*8,hipMemcpyDeviceToHost));
    hipEvent_t start,stop;HIP_CHECK(hipEventCreate(&start));HIP_CHECK(hipEventCreate(&stop));
    auto timing=[&](auto launch){for(int i=0;i<5;i++)launch();HIP_CHECK(hipDeviceSynchronize());std::vector<float>times;for(int i=0;i<30;i++){HIP_CHECK(hipEventRecord(start));launch();HIP_CHECK(hipEventRecord(stop));HIP_CHECK(hipEventSynchronize(stop));float ms;HIP_CHECK(hipEventElapsedTime(&ms,start,stop));times.push_back(ms);}std::sort(times.begin(),times.end());return times[15];};
    printf("backend,m,n,k,median_ms,raw_gops,max_abs_vs_native\n");
    float native_ms=timing(native);printf("native_fp64,%d,%d,%d,%.6f,%.3f,0\n",m,n,k,native_ms,2.*m*n*k/native_ms/1e6);
    for(int mode=0;mode<2;mode++){
        HIP_CHECK(hipMemcpy(da,mode?af.data():ai.data(),a.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(db,mode?bf.data():bi.data(),b.size(),hipMemcpyHostToDevice));
        auto launch=[&]{dim3 grid((n+15)/16,(m+15)/16);if(mode)product<true><<<grid,32>>>(da,db,dc,m,n,k);else product<false><<<grid,32>>>(da,db,dc,m,n,k);HIP_CHECK(hipGetLastError());};
        float ms=timing(launch);HIP_CHECK(hipMemcpy(out.data(),dc,out.size()*8,hipMemcpyDeviceToHost));double err=0;
        for(size_t i=0;i<out.size();i++){if(!std::isfinite(out[i]))return 1;err=std::max(err,std::abs(out[i]-ref[i]));}
        printf("%s,%d,%d,%d,%.6f,%.3f,%.0f\n",mode?"fp8_wmma":"int8_wmma",m,n,k,ms,2.*m*n*k/ms/1e6,err);
        if(err!=0){fprintf(stderr,"exact modular-product validation failed\n");return 1;}
    }
    HIP_CHECK(hipEventDestroy(start));HIP_CHECK(hipEventDestroy(stop));blas_check(rocblas_destroy_handle(h));
    for(void *p:{(void*)da,(void*)db,(void*)dc,(void*)dra,(void*)drb,(void*)dref})HIP_CHECK(hipFree(p));
}
