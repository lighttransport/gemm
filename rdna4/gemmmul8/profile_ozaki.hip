// Internal GEMMul8 phase profiler; build against the PROFILE=on library.
#include "vendor/GEMMul8/include/gemmul8.hpp"
#include "wmma_modular.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define HIP(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);}}while(0)
#define BLAS(x) do{auto e=(x);if(e!=0){fprintf(stderr,"%s: %d\n",#x,int(e));exit(1);}}while(0)
int main(int argc,char **argv){
    if(argc!=6){fprintf(stderr,"usage: profile_ozaki M N K int8|fp8 MODULI\n");return 2;}
    int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]),moduli=atoi(argv[5]);bool fp8=!strcmp(argv[4],"fp8");
    if(m<=0||n<=0||k<=0||moduli<2||moduli>20||(!fp8&&strcmp(argv[4],"int8")))return 2;
    std::vector<double>a(size_t(m)*k),b(size_t(n)*k);for(size_t i=0;i<a.size();i++)a[i]=double(int(i%37)-18)/19.;for(size_t i=0;i<b.size();i++)b[i]=double(int(i%41)-20)/21.;
    double *da,*db,*dc;HIP(hipMalloc(&da,a.size()*8));HIP(hipMalloc(&db,b.size()*8));HIP(hipMalloc(&dc,size_t(m)*n*8));HIP(hipMemcpy(da,a.data(),a.size()*8,hipMemcpyHostToDevice));HIP(hipMemcpy(db,b.data(),b.size()*8,hipMemcpyHostToDevice));
    hipblasLtHandle_t h;BLAS(hipblasLtCreate(&h));double one=1,zero=0;ozaki2_wmma::set_variant(27);
    auto run=[&](void *work){
        if(fp8)return gemmul8::gemmLt<double,gemmul8::Backend::FP8>(h,HIPBLAS_OP_N,HIPBLAS_OP_N,n,m,k,&one,db,n,da,k,&zero,dc,n,moduli,true,work);
        return gemmul8::gemmLt<double,gemmul8::Backend::INT8>(h,HIPBLAS_OP_N,HIPBLAS_OP_N,n,m,k,&one,db,n,da,k,&zero,dc,n,moduli,true,work);
    };
    auto sizes=run(nullptr);void *work;HIP(hipMalloc(&work,size_t(sizes[0])));for(int i=0;i<3;i++)run(work);
    std::vector<double> v[4];for(int i=0;i<15;i++){auto t=run(work);for(int j=0;j<4;j++)v[j].push_back(t[j]*1000);}
    for(auto &x:v)std::sort(x.begin(),x.end());
    printf("backend,m,n,k,moduli,scale_pack_ms,wmma_ms,mod_reduce_ms,crt_rescale_ms\n%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",argv[4],m,n,k,moduli,v[0][7],v[1][7],v[2][7],v[3][7]);
    HIP(hipFree(work));HIP(hipFree(da));HIP(hipFree(db));HIP(hipFree(dc));BLAS(hipblasLtDestroy(h));
}
