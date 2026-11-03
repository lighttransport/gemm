#include "ozaki2.hpp"
#include "wmma_modular.hpp"
#include "vendor/GEMMul8/include/gemmul8.hpp"
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace ozaki2 {
struct context { hipblasLtHandle_t handle; };
static bool valid(const gemm &p, const options &o) {
    if(p.m<0||p.n<0||p.k<0||(o.moduli!=0&&(o.moduli<2||o.moduli>20))||o.wmma_variant< -2||o.wmma_variant>31||
       (o.arithmetic!=backend::int8&&o.arithmetic!=backend::fp8)||
       (p.trans_a!=operation::normal&&p.trans_a!=operation::transpose)||
       (p.trans_b!=operation::normal&&p.trans_b!=operation::transpose)||
       !std::isfinite(p.alpha)||!std::isfinite(p.beta))return false;
    if(!p.m||!p.n)return true;
    if(!p.c||p.ldc<size_t(p.n))return false;
    if(!p.k||p.alpha==0)return true;
    return p.a&&p.b&&p.lda>=size_t(p.trans_a==operation::normal?p.k:p.m)&&
           p.ldb>=size_t(p.trans_b==operation::normal?p.n:p.k);
}
status create(context **out) {
    if(!out)return status::invalid_argument;
    *out=nullptr;context *c=new(std::nothrow) context{};if(!c)return status::backend_error;
    if(hipblasLtCreate(&c->handle)!=HIPBLAS_STATUS_SUCCESS){delete c;return status::backend_error;}
    *out=c;return status::success;
}
void destroy(context *ctx) {if(ctx){gemmul8::clear_configLt(ctx->handle);hipblasLtDestroy(ctx->handle);delete ctx;}}
static std::vector<double> call(context *ctx,const gemm &p,const options &o,void *work,hipStream_t stream) {
    gemmul8::set_memory_savingLt(ctx->handle,o.max_workspace_bytes!=0);
    if(o.max_workspace_bytes)gemmul8::set_max_worksizeLt(ctx->handle,o.max_workspace_bytes);
    int count=o.moduli?o.moduli:(o.arithmetic==backend::int8?15:11);
    auto opb=p.trans_b==operation::normal?HIPBLAS_OP_N:HIPBLAS_OP_T;
    auto opa=p.trans_a==operation::normal?HIPBLAS_OP_N:HIPBLAS_OP_T;
    if(o.arithmetic==backend::int8)
        return gemmul8::gemmLt<double,gemmul8::Backend::INT8>(ctx->handle,opb,opa,p.n,p.m,p.k,&p.alpha,p.b,p.ldb,p.a,p.lda,&p.beta,p.c,p.ldc,count,o.fast_scaling,work,nullptr,nullptr,false,false,false,false,stream);
    return gemmul8::gemmLt<double,gemmul8::Backend::FP8>(ctx->handle,opb,opa,p.n,p.m,p.k,&p.alpha,p.b,p.ldb,p.a,p.lda,&p.beta,p.c,p.ldc,count,o.fast_scaling,work,nullptr,nullptr,false,false,false,false,stream);
}
struct layout {size_t a,b,c,overhead,inner,total;int tile_m,tile_n;gemm normalized;options inner_opts;};
static bool add(size_t a,size_t b,size_t *out){return !__builtin_add_overflow(a,b,out);}
static bool mul(size_t a,size_t b,size_t *out){return !__builtin_mul_overflow(a,b,out);}
static bool round256(size_t n,size_t *out){size_t x;if(!add(n,255,&x))return false;*out=x&~size_t(255);return true;}
static status try_layout(context *ctx,const gemm &p,const options &o,int tm,int tn,layout *l) {
    size_t aelems,belems,celems,ab,bb,cb;
    if(!mul(size_t(tm),size_t(p.k),&aelems)||!mul(size_t(p.k),size_t(tn),&belems)||
       !mul(size_t(tm),size_t(tn),&celems)||!mul(aelems,8,&ab)||!mul(belems,8,&bb)||!mul(celems,8,&cb)||
       !round256(ab,&l->a)||!round256(bb,&l->b)||!round256(cb,&l->c))return status::invalid_argument;
    // 255 bytes cover arbitrary workspace alignment; 256 bytes hold exponents.
    size_t v=511;if(!add(v,l->a,&v)||!add(v,l->b,&v)||!add(v,l->c,&v))return status::invalid_argument;
    if(o.max_workspace_bytes&&v>=o.max_workspace_bytes)return status::insufficient_workspace;
    l->overhead=v;l->tile_m=tm;l->tile_n=tn;l->normalized=p;
    l->normalized.m=tm;l->normalized.n=tn;l->normalized.trans_a=operation::normal;l->normalized.trans_b=operation::normal;
    l->normalized.alpha=1;l->normalized.beta=0;l->normalized.lda=size_t(p.k);
    l->normalized.ldb=size_t(tn);l->normalized.ldc=size_t(tn);
    // Panels are selected here. Avoid hidden K blocking in upstream, which
    // would sum rounded partial FP64 products and change numerical behavior.
    l->inner_opts=o;l->inner_opts.max_workspace_bytes=0;
    try {auto sizes=call(ctx,l->normalized,l->inner_opts,nullptr,nullptr);
        if(sizes.empty()||!std::isfinite(sizes[0])||sizes[0]<=0||sizes[0]>double(std::numeric_limits<size_t>::max()))return status::backend_error;
        l->inner=size_t(sizes[0]);if(!add(v,l->inner,&l->total))return status::invalid_argument;
        if(o.max_workspace_bytes&&l->total>o.max_workspace_bytes)return status::insufficient_workspace;
        return status::success;
    }catch(...){return status::backend_error;}
}
static status make_layout(context *ctx,const gemm &p,const options &o,layout *l) {
    int tm=p.m,tn=p.n;
    for(;;){status s=try_layout(ctx,p,o,tm,tn,l);if(s!=status::insufficient_workspace)return s;
        if(tm==1&&tn==1)return s;
        if((tm>=tn&&tm>1)||tn==1)tm=tm/2+tm%2;else tn=tn/2+tn%2;
    }
}
status workspace_size(context *ctx,const gemm &p,const options &o,size_t *bytes) {
    if(!ctx||!bytes||!valid(p,o))return status::invalid_argument;
    *bytes=0;if(!p.m||!p.n||!p.k||p.alpha==0)return status::success;
    layout l{};auto s=make_layout(ctx,p,o,&l);if(s==status::success)*bytes=l.total;return s;
}
__device__ int exponent_of(double x) {
    uint64_t bits=uint64_t(__double_as_longlong(x))&0x7fffffffffffffffULL;
    unsigned e=unsigned(bits>>52);uint64_t f=bits&0xfffffffffffffULL;
    if(e)return int(e)-1023;
    return f ? 63-__clzll(f)-1074 : -2147483647;
}
__global__ void find_exponent(const double *x,size_t ld,int rows,int cols,int *out) {
    __shared__ int local[256];size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t(rows)*cols;
    int v=i<total?exponent_of(x[(i/size_t(cols))*ld+i%size_t(cols)]):-2147483647;
    local[threadIdx.x]=v;__syncthreads();
    for(int step=128;step;step>>=1){if(int(threadIdx.x)<step)local[threadIdx.x]=max(local[threadIdx.x],local[threadIdx.x+step]);__syncthreads();}
    if(!threadIdx.x)atomicMax(out,local[0]);
}
__global__ void pack_a(const double *x,size_t ld,int m,int k,bool trans,int row0,int rows,const int *exponent,double *out) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t(rows)*k;
    if(i<total){int row=row0+int(i/size_t(k)),col=int(i%size_t(k));int e=*exponent;if(e< -2000)e=0;
        double v=row<m?(trans?x[size_t(col)*ld+row]:x[size_t(row)*ld+col]):0;out[i]=scalbn(v,-e);}
}
__global__ void pack_b(const double *x,size_t ld,int n,int k,bool trans,int col0,int cols,const int *exponent,double *out) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t(k)*cols;
    if(i<total){int row=int(i/size_t(cols)),col=col0+int(i%size_t(cols));int e=*exponent;if(e< -2000)e=0;
        double v=col<n?(trans?x[size_t(col)*ld+row]:x[size_t(row)*ld+col]):0;out[i]=scalbn(v,-e);}
}
__global__ void scale_c(double *c,size_t ldc,int m,int n,double beta) {
    size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t(m)*n;
    if(t<total){size_t i=(t/size_t(n))*ldc+t%size_t(n);c[i]=beta==0?0:beta*c[i];}
}
__global__ void finish(const double *product,const int *exponents,double *c,size_t ldc,int rows,int cols,
                       int tile_n,int row0,int col0,double alpha_mant,int alpha_exp,double beta) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,total=size_t(rows)*cols;
    if(i<total){int ea=exponents[0],eb=exponents[1];if(ea< -2000)ea=0;if(eb< -2000)eb=0;
        size_t row=i/size_t(cols),col=i%size_t(cols);double value=scalbn(product[row*size_t(tile_n)+col]*alpha_mant,ea+eb+alpha_exp);
        size_t ci=(size_t(row0)+row)*ldc+size_t(col0)+col;c[ci]=beta==0?value:fma(beta,c[ci],value);
    }
}
status dgemm(context *ctx,const gemm &p,const options &o,void *workspace,size_t bytes,hipStream_t stream) {
    if(!ctx||!valid(p,o))return status::invalid_argument;
    if(!p.m||!p.n)return status::success;
    if(!p.k||p.alpha==0){size_t total=size_t(p.m)*p.n;hipLaunchKernelGGL(scale_c,dim3((total+255)/256),dim3(256),0,stream,p.c,p.ldc,p.m,p.n,p.beta);return hipGetLastError()==hipSuccess?status::success:status::hip_error;}
    layout l{};status s=make_layout(ctx,p,o,&l);if(s!=status::success)return s;
    if(!workspace||bytes<l.total)return status::insufficient_workspace;
    uintptr_t base=(reinterpret_cast<uintptr_t>(workspace)+255)&~uintptr_t(255);
    int *exps=reinterpret_cast<int *>(base);double *a=reinterpret_cast<double *>(base+256);
    double *b=reinterpret_cast<double *>(base+256+l.a);double *c=reinterpret_cast<double *>(base+256+l.a+l.b);
    void *inner=reinterpret_cast<void *>(base+256+l.a+l.b+l.c);
    if(hipMemsetAsync(exps,0x80,2*sizeof(int),stream)!=hipSuccess)return status::hip_error;
    int ar=p.trans_a==operation::normal?p.m:p.k,ac=p.trans_a==operation::normal?p.k:p.m;
    int br=p.trans_b==operation::normal?p.k:p.n,bc=p.trans_b==operation::normal?p.n:p.k;
    auto grid=[](size_t elems){return dim3((elems+255)/256);};
    hipLaunchKernelGGL(find_exponent,grid(size_t(ar)*ac),dim3(256),0,stream,p.a,p.lda,ar,ac,exps);
    hipLaunchKernelGGL(find_exponent,grid(size_t(br)*bc),dim3(256),0,stream,p.b,p.ldb,br,bc,exps+1);
    if(hipGetLastError()!=hipSuccess)return status::hip_error;
    l.normalized.a=a;l.normalized.b=b;l.normalized.c=c;
    int alpha_exp;double alpha_mant=std::frexp(p.alpha,&alpha_exp);
    // Every inner GEMM uses the same full-sized panel; edge panels are zero
    // padded. Reusing its workspace cannot accidentally under-allocate a tail.
    ozaki2_wmma::set_variant(o.wmma_variant==-2?27:o.wmma_variant);
    for(int64_t row_offset=0;row_offset<p.m;row_offset+=l.tile_m){
        int row0=int(row_offset);int rows=std::min(l.tile_m,p.m-row0);
        hipLaunchKernelGGL(pack_a,grid(size_t(l.tile_m)*p.k),dim3(256),0,stream,p.a,p.lda,p.m,p.k,p.trans_a==operation::transpose,row0,l.tile_m,exps,a);
        for(int64_t col_offset=0;col_offset<p.n;col_offset+=l.tile_n){
            int col0=int(col_offset);int cols=std::min(l.tile_n,p.n-col0);
            hipLaunchKernelGGL(pack_b,grid(size_t(p.k)*l.tile_n),dim3(256),0,stream,p.b,p.ldb,p.n,p.k,p.trans_b==operation::transpose,col0,l.tile_n,exps+1,b);
            if(hipGetLastError()!=hipSuccess)return status::hip_error;
            try{call(ctx,l.normalized,l.inner_opts,inner,stream);}catch(...){return status::backend_error;}
            hipLaunchKernelGGL(finish,grid(size_t(rows)*cols),dim3(256),0,stream,c,exps,p.c,p.ldc,rows,cols,l.tile_n,row0,col0,alpha_mant,alpha_exp,p.beta);
            if(hipGetLastError()!=hipSuccess)return status::hip_error;
        }
    }
    return hipGetLastError()==hipSuccess?status::success:status::hip_error;
}
const char *status_string(status s){switch(s){case status::success:return "success";case status::invalid_argument:return "invalid argument";case status::insufficient_workspace:return "insufficient workspace";case status::hip_error:return "HIP error";case status::backend_error:return "GEMMul8 backend error";}return "unknown";}
}
