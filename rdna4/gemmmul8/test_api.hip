#include "ozaki2.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mpfr.h>
#include <vector>
#define HIP(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);}}while(0)
#define OZ(x) do{auto e=(x);if(e!=ozaki2::status::success){fprintf(stderr,"%s: %s\n",#x,ozaki2::status_string(e));exit(1);}}while(0)
using namespace ozaki2;
static void verify(const std::vector<double> &a,const std::vector<double> &b,const std::vector<double> &c,
                   const std::vector<double> &initial,int m,int n,int k,size_t lda,size_t ldb,size_t ldc,
                   operation ta,operation tb,double alpha,double beta){
    mpfr_t sum,x,y,z;mpfr_inits2(256,sum,x,y,z,(mpfr_ptr)0);
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){
        mpfr_set_zero(sum,0);long double scale=0;
        for(int t=0;t<k;t++){
            double av=ta==operation::normal?a[i*lda+t]:a[t*lda+i];
            double bv=tb==operation::normal?b[t*ldb+j]:b[j*ldb+t];
            mpfr_set_d(x,av,MPFR_RNDN);mpfr_set_d(y,bv,MPFR_RNDN);mpfr_fma(sum,x,y,sum,MPFR_RNDN);
            scale+=fabsl((long double)av*bv);
        }
        mpfr_mul_d(sum,sum,alpha,MPFR_RNDN);mpfr_set_d(x,initial[i*ldc+j],MPFR_RNDN);mpfr_set_d(y,beta,MPFR_RNDN);mpfr_fma(sum,x,y,sum,MPFR_RNDN);
        scale=fabsl((long double)alpha)*scale+fabsl((long double)beta*initial[i*ldc+j]);
        mpfr_set_d(z,c[i*ldc+j],MPFR_RNDN);mpfr_sub(z,z,sum,MPFR_RNDN);mpfr_abs(z,z,MPFR_RNDN);long double err=mpfr_get_ld(z,MPFR_RNDU);
        if(!std::isfinite(c[i*ldc+j])||(scale?err>32*0x1p-53L*scale:err!=0)){
            fprintf(stderr,"MPFR mismatch row=%d col=%d got=%a normalized_error=%Lg\n",i,j,c[i*ldc+j],scale?err/scale:err);exit(1);
        }
    }
    mpfr_clears(sum,x,y,z,(mpfr_ptr)0);
}
static void run(context *ctx,backend b,bool fast,operation ta,operation tb) {
    int m=19,n=23,k=29;size_t lda=(ta==operation::normal?k:m)+3,ldb=(tb==operation::normal?n:k)+5,ldc=n+7;
    size_t ar=ta==operation::normal?m:k,br=tb==operation::normal?k:n;
    std::vector<double>a(ar*lda),bb(br*ldb),c(size_t(m)*ldc),orig(c.size());
    for(size_t i=0;i<a.size();i++)a[i]=(int((i*17+i/13)%101)-50)/37.;
    for(size_t i=0;i<bb.size();i++)bb[i]=(int((i*11+i/7)%107)-53)/43.;
    for(size_t i=0;i<c.size();i++)orig[i]=c[i]=(int(i%17)-8)/11.;
    double *da,*db,*dc;HIP(hipMalloc(&da,a.size()*8));HIP(hipMalloc(&db,bb.size()*8));HIP(hipMalloc(&dc,c.size()*8));
    HIP(hipMemcpy(da,a.data(),a.size()*8,hipMemcpyHostToDevice));HIP(hipMemcpy(db,bb.data(),bb.size()*8,hipMemcpyHostToDevice));HIP(hipMemcpy(dc,c.data(),c.size()*8,hipMemcpyHostToDevice));
    gemm p{m,n,k,ta,tb,1.25,da,lda,db,ldb,-.75,dc,ldc};options o;o.arithmetic=b;o.moduli=b==backend::fp8?11:15;o.fast_scaling=fast;
    size_t bytes;OZ(workspace_size(ctx,p,o,&bytes));void *w;HIP(hipMalloc(&w,bytes));hipStream_t stream;HIP(hipStreamCreate(&stream));
    if(dgemm(ctx,p,o,w,bytes-1,stream)!=status::insufficient_workspace)exit(1);
    OZ(dgemm(ctx,p,o,w,bytes,stream));HIP(hipStreamSynchronize(stream));HIP(hipMemcpy(c.data(),dc,c.size()*8,hipMemcpyDeviceToHost));
    verify(a,bb,c,orig,m,n,k,lda,ldb,ldc,ta,tb,1.25,-.75);
    for(int i=0;i<m;i++)for(size_t j=n;j<ldc;j++)if(c[i*ldc+j]!=orig[i*ldc+j])exit(1);
    // Force a smaller-than-full workspace so edge panels and all transpose
    // combinations are exercised while alpha and beta are nontrivial.
    options limited=o;limited.max_workspace_bytes=bytes-256;size_t panel_bytes;
    OZ(workspace_size(ctx,p,limited,&panel_bytes));if(panel_bytes>limited.max_workspace_bytes)exit(1);
    HIP(hipMemcpyAsync(dc,orig.data(),orig.size()*8,hipMemcpyHostToDevice,stream));
    OZ(dgemm(ctx,p,limited,w,bytes,stream));HIP(hipStreamSynchronize(stream));
    HIP(hipMemcpy(c.data(),dc,c.size()*8,hipMemcpyDeviceToHost));
    verify(a,bb,c,orig,m,n,k,lda,ldb,ldc,ta,tb,1.25,-.75);
    // beta=0 must not read a NaN-filled prior C, even when A*B is computed.
    p.beta=0;HIP(hipMemsetAsync(dc,0xff,c.size()*8,stream));OZ(dgemm(ctx,p,o,w,bytes,stream));
    HIP(hipStreamSynchronize(stream));HIP(hipMemcpy(c.data(),dc,c.size()*8,hipMemcpyDeviceToHost));
    verify(a,bb,c,orig,m,n,k,lda,ldb,ldc,ta,tb,1.25,0);
    p.alpha=0;p.a=nullptr;p.b=nullptr;p.beta=0;HIP(hipMemsetAsync(dc,0xff,c.size()*8,stream));OZ(dgemm(ctx,p,o,nullptr,0,stream));HIP(hipStreamSynchronize(stream));HIP(hipMemcpy(c.data(),dc,c.size()*8,hipMemcpyDeviceToHost));
    for(int i=0;i<m;i++)for(int j=0;j<n;j++)if(c[i*ldc+j]!=0)exit(1);
    HIP(hipStreamDestroy(stream));HIP(hipFree(w));HIP(hipFree(da));HIP(hipFree(db));HIP(hipFree(dc));
    printf("PASS backend=%s scaling=%s trans=%d%d\n",b==backend::fp8?"fp8":"int8",fast?"fast":"accurate",int(ta),int(tb));
}
static void extreme(context *ctx,backend type,int aexp,int bexp,int alphaexp,double expected){
    double *da,*db,*dc;double ha[4],hb[4],hc=std::numeric_limits<double>::quiet_NaN();
    for(int i=0;i<4;i++){ha[i]=std::ldexp(1.,aexp);hb[i]=std::ldexp(1.,bexp);}
    HIP(hipMalloc(&da,sizeof(ha)));HIP(hipMalloc(&db,sizeof(hb)));HIP(hipMalloc(&dc,sizeof(hc)));
    HIP(hipMemcpy(da,ha,sizeof(ha),hipMemcpyHostToDevice));HIP(hipMemcpy(db,hb,sizeof(hb),hipMemcpyHostToDevice));HIP(hipMemcpy(dc,&hc,sizeof(hc),hipMemcpyHostToDevice));
    gemm p{1,1,4,operation::normal,operation::normal,std::ldexp(1.,alphaexp),da,4,db,1,0,dc,1};options o;o.arithmetic=type; // Exercise automatic backend-specific modulus defaults.
    size_t bytes;OZ(workspace_size(ctx,p,o,&bytes));void *work;HIP(hipMalloc(&work,bytes));OZ(dgemm(ctx,p,o,work,bytes));HIP(hipDeviceSynchronize());HIP(hipMemcpy(&hc,dc,8,hipMemcpyDeviceToHost));
    if(hc!=expected){fprintf(stderr,"extreme backend=%d aexp=%d bexp=%d alphaexp=%d got=%a expected=%a\n",int(type),aexp,bexp,alphaexp,hc,expected);exit(1);}
    options capped=o;capped.max_workspace_bytes=1024;if(workspace_size(ctx,p,capped,&bytes)!=status::insufficient_workspace)exit(1);
    p.k=0;p.alpha=1;p.beta=2;p.a=nullptr;p.b=nullptr;OZ(dgemm(ctx,p,o,nullptr,0));HIP(hipDeviceSynchronize());HIP(hipMemcpy(&hc,dc,8,hipMemcpyDeviceToHost));if(hc!=2*expected)exit(1);
    HIP(hipFree(work));HIP(hipFree(da));HIP(hipFree(db));HIP(hipFree(dc));
    printf("PASS extreme backend=%s aexp=%d bexp=%d alphaexp=%d\n",type==backend::fp8?"fp8":"int8",aexp,bexp,alphaexp);
}

int main(){context *ctx;OZ(create(&ctx));
    gemm empty{0,2,3,operation::normal,operation::normal,1,nullptr,0,nullptr,0,0,nullptr,0};options empty_opt;size_t empty_bytes=1;
    OZ(workspace_size(ctx,empty,empty_opt,&empty_bytes));if(empty_bytes!=0)exit(1);OZ(dgemm(ctx,empty,empty_opt,nullptr,0));
for(auto b:{backend::int8,backend::fp8}){for(bool f:{true,false})for(auto a:{operation::normal,operation::transpose})for(auto c:{operation::normal,operation::transpose})run(ctx,b,f,a,c);
    extreme(ctx,b,-1074,500,0,std::ldexp(1.,-572));extreme(ctx,b,-530,-530,0,std::ldexp(1.,-1058));extreme(ctx,b,600,300,-800,std::ldexp(1.,102));
    extreme(ctx,b,1023,-1074,0,std::ldexp(1.,-49));extreme(ctx,b,-530,-530,1023,std::ldexp(1.,-35));extreme(ctx,b,600,300,-1074,std::ldexp(1.,-172));}destroy(ctx);}
