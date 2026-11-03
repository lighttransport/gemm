#include "ozaki2.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define HIP(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);}}while(0)
#define OZ(x) do{auto e=(x);if(e!=ozaki2::status::success){fprintf(stderr,"%s: %s\n",#x,ozaki2::status_string(e));exit(1);}}while(0)
static uint64_t random_state=42;
static double rnd(){random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return double(random_state>>11)*0x1p-52-1;}
int main(int argc,char **argv){
    if(argc!=5){fprintf(stderr,"usage: tune_ozaki M N K int8|fp8\n");return 2;}
    int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]);bool fp8=!strcmp(argv[4],"fp8");
    if(m<=0||n<=0||k<=0||(!fp8&&strcmp(argv[4],"int8")))return 2;
    std::vector<double>a(size_t(m)*k),b(size_t(k)*n),ref(size_t(m)*n),got(ref.size());
    for(auto &x:a)x=rnd();for(auto &x:b)x=rnd();
    double *da,*db,*dc;HIP(hipMalloc(&da,a.size()*8));HIP(hipMalloc(&db,b.size()*8));HIP(hipMalloc(&dc,ref.size()*8));
    HIP(hipMemcpy(da,a.data(),a.size()*8,hipMemcpyHostToDevice));HIP(hipMemcpy(db,b.data(),b.size()*8,hipMemcpyHostToDevice));
    ozaki2::context *ctx;OZ(ozaki2::create(&ctx));ozaki2::options opt;
    opt.arithmetic=fp8?ozaki2::backend::fp8:ozaki2::backend::int8;opt.moduli=fp8?11:15;
    size_t available,total;HIP(hipMemGetInfo(&available,&total));opt.max_workspace_bytes=std::min(size_t(8ull<<30),available/4*3);
    ozaki2::gemm p{m,n,k,ozaki2::operation::normal,ozaki2::operation::normal,1,da,size_t(k),db,size_t(n),0,dc,size_t(n)};
    size_t bytes;OZ(ozaki2::workspace_size(ctx,p,opt,&bytes));void *work;HIP(hipMalloc(&work,bytes));
    auto run=[&]{OZ(ozaki2::dgemm(ctx,p,opt,work,bytes));};
    hipEvent_t start,stop;HIP(hipEventCreate(&start));HIP(hipEventCreate(&stop));
    printf("backend,m,n,k,variant,tile_m,tile_n,tile_k,waves,buffers,median_ms,max_abs_diff\n");
    for(int variant=-1;variant<32;variant++){
        opt.wmma_variant=variant;run();run();HIP(hipDeviceSynchronize());
        HIP(hipEventRecord(start));run();HIP(hipEventRecord(stop));HIP(hipEventSynchronize(stop));float trial;HIP(hipEventElapsedTime(&trial,start,stop));
        int reps=std::min(1000,std::max(1,int(ceil(5./std::max(.001,double(trial))))));std::vector<double>ms;
        for(int j=0;j<5;j++){HIP(hipEventRecord(start));for(int i=0;i<reps;i++)run();HIP(hipEventRecord(stop));HIP(hipEventSynchronize(stop));float t;HIP(hipEventElapsedTime(&t,start,stop));ms.push_back(t/reps);}
        std::sort(ms.begin(),ms.end());HIP(hipMemcpy(got.data(),dc,got.size()*8,hipMemcpyDeviceToHost));double diff=0;
        if(variant<0)ref=got;else for(size_t i=0;i<got.size();i++){if(!std::isfinite(got[i])){diff=INFINITY;break;}diff=std::max(diff,std::abs(got[i]-ref[i]));}
        printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.9g\n",argv[4],m,n,k,variant,variant<0?64:((variant&1)?128:64),variant<0?64:((variant&2)?128:64),variant<0?32:((variant&4)?64:32),variant<0?4:((variant&8)?8:4),variant<0?1:((variant&16)?2:1),ms[2],diff);fflush(stdout);
        if(diff!=0){fprintf(stderr,"variant %d differs from validated default\n",variant);return 1;}
    }
    HIP(hipEventDestroy(start));HIP(hipEventDestroy(stop));HIP(hipFree(work));HIP(hipFree(da));HIP(hipFree(db));HIP(hipFree(dc));ozaki2::destroy(ctx);
}
