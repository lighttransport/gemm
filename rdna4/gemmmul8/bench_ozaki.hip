#include "ozaki2.hpp"
#include <rocblas/rocblas.h>
#include <mpfr.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);} }while(0)
#define BLAS_CHECK(x) do { auto e=(x); if(e!=0){fprintf(stderr,"%s: status %d\n",#x,int(e));exit(1);} }while(0)
static uint64_t state=42;
static double rnd() {state^=state<<13;state^=state>>7;state^=state<<17;return double(state>>11)*0x1p-52-1;}
static uint64_t ulp_distance(double a,double b) {
    if(a==0&&b==0)return 0;
    uint64_t ua,ub;std::memcpy(&ua,&a,8);std::memcpy(&ub,&b,8);
    ua=(ua>>63)?(~ua+1ULL):(ua|0x8000000000000000ULL);
    ub=(ub>>63)?(~ub+1ULL):(ub|0x8000000000000000ULL);
    return ua>ub?ua-ub:ub-ua;
}
int main(int argc,char **argv){
    int m=128,n=128,k=128,iters=30,moduli=0,warmups=5,variant=-2,device=0;bool fast=true;const char *which="int8", *distribution="uniform";size_t workspace_mib=8192;double min_sample_ms=100;uint64_t seed=42;
    for(int i=1;i<argc;i++) {
        if(i+1>=argc){fprintf(stderr,"missing value for %s\n",argv[i]);return 2;}
        const char *v=argv[++i];
        if(!strcmp(argv[i-1],"--device"))device=atoi(v);else if(!strcmp(argv[i-1],"--m"))m=atoi(v);else if(!strcmp(argv[i-1],"--n"))n=atoi(v);
        else if(!strcmp(argv[i-1],"--k"))k=atoi(v);else if(!strcmp(argv[i-1],"--iters"))iters=atoi(v);
        else if(!strcmp(argv[i-1],"--variant"))variant=atoi(v);else if(!strcmp(argv[i-1],"--warmups"))warmups=atoi(v);else if(!strcmp(argv[i-1],"--seed"))seed=state=strtoull(v,nullptr,0);else if(!strcmp(argv[i-1],"--min-sample-ms"))min_sample_ms=atof(v);
        else if(!strcmp(argv[i-1],"--workspace-mib"))workspace_mib=strtoull(v,nullptr,0);else if(!strcmp(argv[i-1],"--distribution"))distribution=v;
        else if(!strcmp(argv[i-1],"--moduli"))moduli=atoi(v);else if(!strcmp(argv[i-1],"--backend"))which=v;
        else if(!strcmp(argv[i-1],"--scaling")){if(strcmp(v,"fast")&&strcmp(v,"accurate"))return 2;fast=!strcmp(v,"fast");}
        else {fprintf(stderr,"unknown option %s\n",argv[i-1]);return 2;}
    }
    bool fp8=!strcmp(which,"fp8");if(!fp8&&strcmp(which,"int8"))return 2;
    if(!moduli)moduli=fp8?11:15;
    if(device<0||m<=0||n<=0||k<=0||iters<=0||warmups<0||moduli<2||moduli>20||variant< -2||variant>31||!workspace_mib||workspace_mib>SIZE_MAX/(1<<20)||!state||!std::isfinite(min_sample_ms)||min_sample_ms<=0)return 2;
    if(strcmp(distribution,"uniform")&&strcmp(distribution,"wide")&&strcmp(distribution,"cancellation")&&strcmp(distribution,"zero")&&strcmp(distribution,"subnormal-input")&&strcmp(distribution,"subnormal-output")&&strcmp(distribution,"huge"))return 2;
    HIP_CHECK(hipSetDevice(device));
    std::vector<double> a(size_t(m)*k),b(size_t(k)*n),c(size_t(m)*n),native(c.size());
    for(auto &x:a)x=rnd();for(auto &x:b)x=rnd();
    if(!strcmp(distribution,"wide")) {
        for(auto &x:a)x=std::ldexp(x,int(state%61)-30),rnd();
        for(auto &x:b)x=std::ldexp(x,int(state%61)-30),rnd();
    } else if(!strcmp(distribution,"cancellation")) {
        for(int row=0;row<m;row++)for(int t=0;t+1<k;t+=2)a[size_t(row)*k+t+1]=a[size_t(row)*k+t];
        for(int t=0;t+1<k;t+=2)for(int col=0;col<n;col++)b[size_t(t+1)*n+col]=-b[size_t(t)*n+col]+std::ldexp(rnd(),-48);
    } else if(!strcmp(distribution,"zero")){std::fill(a.begin(),a.end(),0);
    } else if(!strcmp(distribution,"subnormal-input")) {
        for(auto &x:a)x=std::ldexp(x,-1070);for(auto &x:b)x=std::ldexp(x,500);
    } else if(!strcmp(distribution,"subnormal-output")) {
        for(auto &x:a)x=std::ldexp(x,-530);for(auto &x:b)x=std::ldexp(x,-530);
    } else if(!strcmp(distribution,"huge")) {
        for(auto &x:a)x=std::ldexp(x,500);for(auto &x:b)x=std::ldexp(x,500);
    }

    double *da,*db,*dc,*dn;HIP_CHECK(hipMalloc(&da,a.size()*8));HIP_CHECK(hipMalloc(&db,b.size()*8));
    HIP_CHECK(hipMalloc(&dc,c.size()*8));HIP_CHECK(hipMalloc(&dn,c.size()*8));
    HIP_CHECK(hipMemcpy(da,a.data(),a.size()*8,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(db,b.data(),b.size()*8,hipMemcpyHostToDevice));
    ozaki2::context *h;auto cs=ozaki2::create(&h);if(cs!=ozaki2::status::success){fprintf(stderr,"ozaki2::create: %s\n",ozaki2::status_string(cs));return 1;}rocblas_handle rb;BLAS_CHECK(rocblas_create_handle(&rb));
    double alpha=1,beta=0;
    size_t available,total;HIP_CHECK(hipMemGetInfo(&available,&total));
    size_t cap=std::min(workspace_mib*size_t(1<<20),available/4*3);
    ozaki2::options opt;opt.arithmetic=fp8?ozaki2::backend::fp8:ozaki2::backend::int8;
    opt.moduli=moduli;opt.fast_scaling=fast;opt.wmma_variant=variant;opt.max_workspace_bytes=cap;
    ozaki2::gemm problem{m,n,k,ozaki2::operation::normal,ozaki2::operation::normal,alpha,da,size_t(k),db,size_t(n),beta,dc,size_t(n)};
    size_t bytes=0;cs=ozaki2::workspace_size(h,problem,opt,&bytes);
    if(cs!=ozaki2::status::success){fprintf(stderr,"workspace query: %s, cap=%zu, free=%zu\n",ozaki2::status_string(cs),cap,available);return 1;}
    auto run=[&](void *workspace){auto result=ozaki2::dgemm(h,problem,opt,workspace,bytes);if(result!=ozaki2::status::success){fprintf(stderr,"ozaki2::dgemm: %s\n",ozaki2::status_string(result));exit(1);}};
    void *work;HIP_CHECK(hipMalloc(&work,bytes));
    auto native_run=[&]{BLAS_CHECK(rocblas_dgemm(rb,rocblas_operation_none,rocblas_operation_none,n,m,k,&alpha,db,n,da,k,&beta,dn,n));};
    hipEvent_t begin,end;HIP_CHECK(hipEventCreate(&begin));HIP_CHECK(hipEventCreate(&end));
    struct timing {double median,p10,p90,min_batch_ms,max_batch_ms;int repeats;};
    auto time=[&](auto f){
        for(int i=0;i<warmups;i++)f();HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(begin));f();HIP_CHECK(hipEventRecord(end));HIP_CHECK(hipEventSynchronize(end));
        float trial;HIP_CHECK(hipEventElapsedTime(&trial,begin,end));
        double estimate=std::ceil(1.2*min_sample_ms/std::max(double(trial),0.001));
        int repeats=int(std::max(1.,std::min(100000.,estimate)));double min_batch=INFINITY,max_batch=0;
        std::vector<double> v;
        for(int i=0;i<iters;){
            HIP_CHECK(hipEventRecord(begin));for(int j=0;j<repeats;j++)f();
            HIP_CHECK(hipEventRecord(end));HIP_CHECK(hipEventSynchronize(end));float ms;HIP_CHECK(hipEventElapsedTime(&ms,begin,end));
            if(double(ms)<min_sample_ms){
                if(repeats>=100000){fprintf(stderr,"cannot reach requested sample duration\n");exit(1);}
                double required=std::ceil(repeats*1.2*min_sample_ms/std::max(double(ms),0.001));
                repeats=int(std::min(100000.,std::max(double(repeats+1),required)));continue;
            }
            min_batch=std::min(min_batch,double(ms));max_batch=std::max(max_batch,double(ms));
            v.push_back(double(ms)/repeats);++i;
        }
        std::sort(v.begin(),v.end());return timing{v[v.size()/2],v[(v.size()-1)/10],v[(v.size()-1)*9/10],min_batch,max_batch,repeats};
    };
    timing nt=time(native_run);timing ot=time([&]{run(work);});double native_ms=nt.median,ms=ot.median;
    HIP_CHECK(hipMemcpy(c.data(),dc,c.size()*8,hipMemcpyDeviceToHost));HIP_CHECK(hipMemcpy(native.data(),dn,c.size()*8,hipMemcpyDeviceToHost));
    size_t nonfinite=0,zero_mismatch=0;long double diff=0,norm=0;for(size_t i=0;i<c.size();i++){if(!std::isfinite(c[i])||!std::isfinite(native[i])){++nonfinite;diff=INFINITY;continue;}long double d=(long double)c[i]-native[i];diff+=d*d;norm+=(long double)native[i]*native[i];}
    long double err=0,native_abs_error=0;double nerr=0,native_err=0;size_t samples=c.size() <= 4096 ? c.size() : size_t(256);
    std::vector<uint64_t> ulps,native_ulps;
    mpfr_t sum,x,y;mpfr_inits2(256,sum,x,y,(mpfr_ptr)0);
    for(size_t s=0;s<samples;s++){
        size_t ix=samples==c.size()?s:(s*(c.size()-1))/(samples-1);int row=int(ix/n),col=int(ix%n);mpfr_set_zero(sum,0);long double scale=0;
        for(int t=0;t<k;t++){double av=a[size_t(row)*k+t],bv=b[size_t(t)*n+col];mpfr_set_d(x,av,MPFR_RNDN);mpfr_set_d(y,bv,MPFR_RNDN);mpfr_fma(sum,x,y,sum,MPFR_RNDN);scale+=fabsl((long double)av*bv);}
        double rounded=mpfr_get_d(sum,MPFR_RNDN);if(std::isfinite(rounded)&&std::isfinite(c[ix])&&std::isfinite(native[ix])){ulps.push_back(ulp_distance(c[ix],rounded));native_ulps.push_back(ulp_distance(native[ix],rounded));}
        mpfr_set_d(x,c[ix],MPFR_RNDN);mpfr_sub(x,x,sum,MPFR_RNDN);mpfr_abs(x,x,MPFR_RNDN);long double e=mpfr_get_ld(x,MPFR_RNDU);err=std::max(err,e);if(scale)nerr=std::max(nerr,double(e/scale));else if(e!=0){nerr=INFINITY;++zero_mismatch;}
        mpfr_set_d(x,native[ix],MPFR_RNDN);mpfr_sub(x,x,sum,MPFR_RNDN);mpfr_abs(x,x,MPFR_RNDN);long double ne=mpfr_get_ld(x,MPFR_RNDU);native_abs_error=std::max(native_abs_error,ne);if(scale)native_err=std::max(native_err,double(ne/scale));else if(ne!=0)native_err=INFINITY;
    }
    mpfr_clears(sum,x,y,(mpfr_ptr)0);
    std::sort(ulps.begin(),ulps.end());std::sort(native_ulps.begin(),native_ulps.end());
    auto pct=[](const std::vector<uint64_t>&v,size_t p)->unsigned long long{return v.empty()?0:v[(v.size()-1)*p/100];};
    printf("backend,device,m,n,k,moduli,scaling,distribution,variant,seed,iters,warmups,min_sample_ms,workspace_limit_bytes,workspace_bytes,ms,p10_ms,p90_ms,native_ms,native_p10_ms,native_p90_ms,tflops,native_tflops,speedup,max_abs_mpfr,native_max_abs_mpfr,sampled_norm_error,sampled_native_norm_error,frob_vs_native,samples,ulp_p50,ulp_p95,ulp_max,native_ulp_p50,native_ulp_p95,native_ulp_max,ulp_samples,zero_denominator_mismatches,nonfinite_outputs,min_batch_ms,max_batch_ms,native_min_batch_ms,native_max_batch_ms,repeats,native_repeats\n");
    printf("%s,%d,%d,%d,%d,%d,%s,%s,%d,%llu,%d,%d,%.6f,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.9Lg,%.9Lg,%.9g,%.9g,%.9Lg,%zu,%llu,%llu,%llu,%llu,%llu,%llu,%zu,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%d,%d\n",which,device,m,n,k,moduli,fast?"fast":"accurate",distribution,variant==-2?27:variant,(unsigned long long)seed,iters,warmups,min_sample_ms,cap,bytes,ms,ot.p10,ot.p90,native_ms,nt.p10,nt.p90,2.*m*n*k/(ms*1e9),2.*m*n*k/(native_ms*1e9),native_ms/ms,err,native_abs_error,nerr,native_err,norm?sqrtl(diff/norm):sqrtl(diff),samples,pct(ulps,50),pct(ulps,95),pct(ulps,100),pct(native_ulps,50),pct(native_ulps,95),pct(native_ulps,100),ulps.size(),zero_mismatch,nonfinite,ot.min_batch_ms,ot.max_batch_ms,nt.min_batch_ms,nt.max_batch_ms,ot.repeats,nt.repeats);
    HIP_CHECK(hipEventDestroy(begin));HIP_CHECK(hipEventDestroy(end));HIP_CHECK(hipFree(work));HIP_CHECK(hipFree(da));HIP_CHECK(hipFree(db));HIP_CHECK(hipFree(dc));HIP_CHECK(hipFree(dn));ozaki2::destroy(h);BLAS_CHECK(rocblas_destroy_handle(rb));
    return nonfinite==0 && zero_mismatch==0 && nerr<=std::max(4*native_err,32*0x1p-53)?0:1;
}
