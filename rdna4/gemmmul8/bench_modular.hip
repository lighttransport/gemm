// Raw selected WMMA product throughput; excludes all Ozaki-II conversion.
#include "wmma_modular.hpp"
#include <hip/hip_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define HIP(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);}}while(0)
int main(int argc,char **argv){
    int m=argc>1?atoi(argv[1]):1024,n=argc>2?atoi(argv[2]):1024,k=argc>3?atoi(argv[3]):1024;
    if(m<=0||n<=0||k<=0||k>ozaki2_wmma::max_k_fp8){fprintf(stderr,"usage: bench_modular [M N K<=%d]\n",ozaki2_wmma::max_k_fp8);return 2;}
    std::vector<int8_t>a(size_t(m)*k),b(size_t(n)*k);std::vector<uint8_t>af(a.size()),bf(b.size());
    for(size_t i=0;i<a.size();i++){a[i]=int8_t(int((i*17+i/7)%31)-15);af[i]=__hip_fp8_e4m3(float(a[i])).__x;}
    for(size_t i=0;i<b.size();i++){b[i]=int8_t(int((i*11+i/13)%31)-15);bf[i]=__hip_fp8_e4m3(float(b[i])).__x;}
    uint8_t *da,*db;void *dc;HIP(hipMalloc(&da,a.size()));HIP(hipMalloc(&db,b.size()));HIP(hipMalloc(&dc,size_t(m)*n*4));
    hipEvent_t begin,end;HIP(hipEventCreate(&begin));HIP(hipEventCreate(&end));
    puts("backend,m,n,k,variant,median_ms,p10_ms,p90_ms,raw_tops,min_batch_ms,max_abs_error");
    for(int mode=0;mode<2;mode++){
        auto type=mode?ozaki2_wmma::backend::fp8:ozaki2_wmma::backend::int8;int variant=27;
        HIP(hipMemcpy(da,mode?af.data():reinterpret_cast<uint8_t *>(a.data()),a.size(),hipMemcpyHostToDevice));
        HIP(hipMemcpy(db,mode?bf.data():reinterpret_cast<uint8_t *>(b.data()),b.size(),hipMemcpyHostToDevice));ozaki2_wmma::set_variant(variant);
        auto run=[&]{HIP(ozaki2_wmma::product(type,nullptr,m,n,k,da,k,0,db,k,0,dc,m,0,1,false));};
        for(int i=0;i<5;i++)run();HIP(hipDeviceSynchronize());
        HIP(hipEventRecord(begin));run();HIP(hipEventRecord(end));HIP(hipEventSynchronize(end));float trial;HIP(hipEventElapsedTime(&trial,begin,end));
        int reps=int(std::min(100000.,std::max(1.,ceil(120./std::max(.001,double(trial))))));double min_batch=INFINITY;std::vector<double>times;
        for(int i=0;i<30;){HIP(hipEventRecord(begin));for(int j=0;j<reps;j++)run();HIP(hipEventRecord(end));HIP(hipEventSynchronize(end));float ms;HIP(hipEventElapsedTime(&ms,begin,end));
            if(ms<100){if(reps>=100000)return 1;reps=int(std::min(100000.,ceil(reps*120./std::max(.001,double(ms)))));continue;}
            min_batch=std::min(min_batch,double(ms));times.push_back(ms/reps);i++;}
        std::sort(times.begin(),times.end());std::vector<int32_t>raw(size_t(m)*n);HIP(hipMemcpy(raw.data(),dc,raw.size()*4,hipMemcpyDeviceToHost));int maxerr=0;
        for(int s=0;s<256;s++){int row=int((uint64_t(s)*7919)%unsigned(m)),col=int((uint64_t(s)*104729)%unsigned(n)),ref=0;
            for(int t=0;t<k;t++)ref+=int(a[size_t(row)*k+t])*int(b[size_t(col)*k+t]);size_t p=size_t(col)*m+row;int got;
            if(mode){float f;__builtin_memcpy(&f,&raw[p],4);got=int(f);if(f!=float(got))return 1;}else got=raw[p];maxerr=std::max(maxerr,abs(got-ref));}
        double ms=times[15];printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.3f,%d\n",mode?"fp8":"int8",m,n,k,variant,ms,times[2],times[26],2.*m*n*k/(ms*1e9),min_batch,maxerr);if(maxerr)return 1;
    }
    HIP(hipEventDestroy(begin));HIP(hipEventDestroy(end));HIP(hipFree(da));HIP(hipFree(db));HIP(hipFree(dc));
}
