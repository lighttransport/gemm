/* Standalone C ABI around pinned SageAttention AMD gfx12 INT8 QK / FP8 PV.
 * Device implementation: Apache-2.0, see extract_sage.py and SAGE_LICENSE. */
#include <hip/hip_runtime.h>
#define SAGEATTN_GFX12_BUILD_AUX 1
#define SAGEATTN_GFX12_BUILD_PREPARE 0
#define SAGEATTN_GFX12_BUILD_ATTN_F16 0
#define SAGEATTN_GFX12_BUILD_ATTN_FP8 1
#define SAGEATTN_GFX12_BUILD_RAWQ_FP8 0
#include "../../tmp/qimg21/sage_device.inc"
#include <mutex>
namespace {
struct Workspace {
    void *ptr=nullptr; size_t bytes=0; int device=-1;
    ~Workspace() { if(ptr)(void)hipFree(ptr); }
};
thread_local Workspace work;
__global__ void pack_hnd(__hip_bfloat16 *out,const __hip_bfloat16 *in,int rows,int heads,int stride) {
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(long)rows*heads*128) { int d=i%128,r=(i/128)%rows,h=i/(128*rows);out[i]=in[(long)r*stride+h*128+d]; }
}
__global__ void unpack_hnd(__hip_bfloat16*out,const __hip_bfloat16*in,int rows,int heads,int stride) {
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<(long)rows*heads*128){int d=i%128,r=(i/128)%rows,h=i/(128*rows);out[(long)r*stride+h*128+d]=in[i];}
}
__global__ void pack_hnd_padded(__hip_bfloat16*out,const __hip_bfloat16*in,int rows,int padded,int heads,int stride) {
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<(long)padded*heads*128){int d=i%128,r=(i/128)%padded,h=i/(128*padded);out[i]=r<rows?in[(long)r*stride+h*128+d]:__float2bfloat16(0.f);}
}
__global__ void unpack_hnd_padded(__hip_bfloat16*out,const __hip_bfloat16*in,int rows,int padded,int heads,int stride) {
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<(long)rows*heads*128){int d=i%128,r=(i/128)%rows,h=i/(128*rows);out[(long)r*stride+h*128+d]=in[((long)h*padded+r)*128+d];}
}
/* Parallel column statistics over [heads][rows][128] (the serial per-head versions below cost ~2.8 ms and
 * ~1.8 ms at 8k rows): each block reduces a row chunk of one head; partials meet through atomics. */
constexpr int kStatChunk=128;
/* The attention kernel consumes K/V in 64-row tiles: with nk % 64 != 0 the last tile of the last head reads past
 * the INT8 K / FP8 V buffers. Zeroed slack keeps those reads finite (masked logits * NaN would still be NaN). */
constexpr size_t kSlack=(size_t)128*128*4;
__global__ void column_stats(float *sum,unsigned *absmax,const __hip_bfloat16 *x,int rows) {
    int d=threadIdx.x,h=blockIdx.y,r0=blockIdx.x*kStatChunk,r1=min(rows,r0+kStatChunk);
    float s=0,m=0;
    for(int r=r0;r<r1;r++){float v=__bfloat162float(x[((long)h*rows+r)*128+d]);s+=v;m=fmaxf(m,fabsf(v));}
    if(sum)atomicAdd(sum+h*128+d,s);
    if(absmax)atomicMax(absmax+h*128+d,__float_as_uint(m));   /* non-negative floats order like their bits */
}
__global__ void subtract_mean(__hip_bfloat16 *k,const float *sum,int rows,int heads) {
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(long)rows*heads*128){int d=i%128,h=i/(128L*rows);k[i]=__float2bfloat16(__bfloat162float(k[i])-sum[h*128+d]/rows);}
}
__global__ void scales_from_absmax(float *scales,const unsigned *absmax,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float mx=__uint_as_float(absmax[i]);scales[i]=mx?mx/448.f:1.f;}
}
/* out[r*os + h*128 + d] (f32, token-major) = in[h][r][d] (bf16, padded rows) */
__global__ void unpack_hnd_f32(float*out,const __hip_bfloat16*in,int rows,int padded,int heads,int os) {
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<(long)rows*heads*128){int d=i%128,r=(i/128)%rows,h=i/(128*rows);out[(long)r*os+h*128+d]=__bfloat162float(in[((long)h*padded+r)*128+d]);}
}
__global__ void smooth_k(__hip_bfloat16 *k,int rows,int heads) {
    int d=threadIdx.x,h=blockIdx.x;float mean=0;
    for(int r=0;r<rows;r++)mean+=__bfloat162float(k[((long)h*rows+r)*128+d]);mean/=rows;
    for(int r=0;r<rows;r++){long i=((long)h*rows+r)*128+d;k[i]=__float2bfloat16(__bfloat162float(k[i])-mean);}
}
__global__ void value_scales(float*scales,const __hip_bfloat16*v,int rows,int heads) {
    int d=threadIdx.x,h=blockIdx.x;float mx=0;
    for(int r=0;r<rows;r++)mx=fmaxf(mx,fabsf(__bfloat162float(v[((long)h*rows+r)*128+d])));
    scales[h*128+d]=mx ? mx/448.f : 1.f;
}
}
extern "C" int q21f_sage_attention(void*out,const void*q,const void*k,const void*v,int nq,int nk,int heads,int qs,int ks,int os,int mask,int accum,void*handle) {
    if(!out||!q||!k||!v||nq<1||nk<1||nq>262144||nk>262144||heads<1||heads>64||qs<heads*128||ks<heads*128||os<heads*128||mask||accum<0||accum>1)return hipErrorInvalidValue;
    hipStream_t stream=(hipStream_t)handle;int dev;hipError_t rc=hipGetDevice(&dev);if(rc)return rc;
    int padded_q=(nq+127)/128*128;
    size_t qn=(size_t)padded_q*heads*128,kn=(size_t)nk*heads*128;
    int qgroups=(padded_q+31)/32,kgroups=(nk+63)/64;
    size_t bytes=qn*5+kn*6+2*kSlack+(size_t)heads*(qgroups+kgroups+128)*4+(size_t)heads*128*8+4096;
    if(work.device!=dev||work.bytes<bytes) {
        if(work.ptr){rc=hipDeviceSynchronize();if(rc)return rc;rc=hipFree(work.ptr);if(rc)return rc;work.ptr=nullptr;}
        rc=hipMalloc(&work.ptr,bytes);if(rc)return rc;work.bytes=bytes;work.device=dev;
    }
    char*p=(char*)work.ptr;
    auto alloc=[&](size_t n){void*r=p;p+=((n+255)/256)*256;return r;};
    auto oh=(__hip_bfloat16*)alloc(qn*2);auto qh=(__hip_bfloat16*)alloc(qn*2);auto kh=(__hip_bfloat16*)alloc(kn*2);auto vh=(__hip_bfloat16*)alloc(kn*2);
    auto q8=(int8_t*)alloc(qn);auto k8=(int8_t*)alloc(kn+kSlack);auto v8=(uint8_t*)alloc(kn+kSlack);
    auto qscale=(float*)alloc((size_t)heads*qgroups*4);auto kscale=(float*)alloc((size_t)heads*kgroups*4);auto vs=(float*)alloc((size_t)heads*128*4);
    rc=hipMemsetAsync(qh,0,qn*2,stream);if(rc)return rc;
    pack_hnd_padded<<<(qn+255)/256,256,0,stream>>>(qh,(const __hip_bfloat16*)q,nq,padded_q,heads,qs);
    pack_hnd<<<(kn+255)/256,256,0,stream>>>(kh,(const __hip_bfloat16*)k,nk,heads,ks);
    pack_hnd<<<(kn+255)/256,256,0,stream>>>(vh,(const __hip_bfloat16*)v,nk,heads,ks);
    auto ksum=(float*)alloc((size_t)heads*128*4);auto vmax=(unsigned*)alloc((size_t)heads*128*4);
    rc=hipMemsetAsync(k8+kn,0,kSlack,stream);if(rc)return rc;rc=hipMemsetAsync(v8+kn,0,kSlack,stream);if(rc)return rc;
    rc=hipMemsetAsync(ksum,0,(size_t)heads*128*4,stream);if(rc)return rc;rc=hipMemsetAsync(vmax,0,(size_t)heads*128*4,stream);if(rc)return rc;
    column_stats<<<dim3((nk+kStatChunk-1)/kStatChunk,heads),128,0,stream>>>(ksum,nullptr,kh,nk);
    subtract_mean<<<(kn+255)/256,256,0,stream>>>(kh,ksum,nk,heads);
    quant_qk_int8_hnd_kernel<__hip_bfloat16,128><<<dim3(qgroups+kgroups,heads),256,0,stream>>>(qh,kh,q8,k8,qscale,kscale,1,heads,heads,padded_q,nk,qgroups,kgroups);
    column_stats<<<dim3((nk+kStatChunk-1)/kStatChunk,heads),128,0,stream>>>(nullptr,vmax,vh,nk);
    scales_from_absmax<<<(heads*128+255)/256,256,0,stream>>>(vs,vmax,heads*128);
    transpose_value_fp8_scaled_hnd_kernel<__hip_bfloat16><<<dim3((nk+127)/128,8,heads),256,0,stream>>>(vh,vs,v8,heads,nk,128);
    qk_int8_sv_f8_native_2q_kernel<64,128,0,8,true,128,true,false,__hip_bfloat16><<<dim3((nq+127)/128,heads),128,0,stream>>>(q8,k8,v8,oh,qscale,kscale,vs,1,padded_q,nk,heads,heads,
        qn,128,(int64_t)padded_q*128,kn,128,(int64_t)nk*128,kn,nk,(int64_t)nk*128,qn,128,(int64_t)padded_q*128,
        (int64_t)heads*qgroups,qgroups,(int64_t)heads*kgroups,kgroups,kHND,0.08838834764831845f,false);
    unpack_hnd_padded<<<((size_t)nq*heads*128+255)/256,256,0,stream>>>((__hip_bfloat16*)out,oh,nq,padded_q,heads,os);
    return hipGetLastError();
}

/* Head-major entry for runners that already hold RMSNorm+RoPE'd BF16 Q/K/V as [heads][rows][128]:
 * qh has padded_q rows per head (rows >= nq zero), kh is mean-smoothed in place. out is f32 token-major
 * (row stride os elements), as the qimg Edit-2511 runner's residual stream expects. */
/* NOTE: the caller and this plugin must share one HIP runtime instance (two different libamdhip64 builds in one
 * process do not order or synchronize each other's queues); the qimg runner checks this before enabling Sage. */
extern "C" int q21f_sage_attention_hnd(float*out,const void*qh_,void*kh_,const void*vh_,int nq,int padded_q,int nk,int heads,int os,void*handle) {
    if(!out||!qh_||!kh_||!vh_||nq<1||nk<1||padded_q<nq||padded_q%128||heads<1||heads>64||os<heads*128)return hipErrorInvalidValue;
    hipStream_t stream=(hipStream_t)handle;int dev;hipError_t rc=hipGetDevice(&dev);if(rc)return rc;
    auto qh=(const __hip_bfloat16*)qh_;auto kh=(__hip_bfloat16*)kh_;auto vh=(const __hip_bfloat16*)vh_;
    size_t qn=(size_t)padded_q*heads*128,kn=(size_t)nk*heads*128;
    int qgroups=(padded_q+31)/32,kgroups=(nk+63)/64;
    size_t bytes=qn*3+kn*2+2*kSlack+(size_t)heads*(qgroups+kgroups+128)*4+(size_t)heads*128*8+4096;
    if(work.device!=dev||work.bytes<bytes) {
        if(work.ptr){rc=hipDeviceSynchronize();if(rc)return rc;rc=hipFree(work.ptr);if(rc)return rc;work.ptr=nullptr;}
        rc=hipMalloc(&work.ptr,bytes);if(rc)return rc;work.bytes=bytes;work.device=dev;
    }
    char*p=(char*)work.ptr;
    auto alloc=[&](size_t n){void*r=p;p+=((n+255)/256)*256;return r;};
    auto oh=(__hip_bfloat16*)alloc(qn*2);auto q8=(int8_t*)alloc(qn);auto k8=(int8_t*)alloc(kn+kSlack);auto v8=(uint8_t*)alloc(kn+kSlack);
    auto qscale=(float*)alloc((size_t)heads*qgroups*4);auto kscale=(float*)alloc((size_t)heads*kgroups*4);auto vs=(float*)alloc((size_t)heads*128*4);
    auto ksum=(float*)alloc((size_t)heads*128*4);auto vmax=(unsigned*)alloc((size_t)heads*128*4);
    rc=hipMemsetAsync(k8+kn,0,kSlack,stream);if(rc)return rc;rc=hipMemsetAsync(v8+kn,0,kSlack,stream);if(rc)return rc;
    rc=hipMemsetAsync(ksum,0,(size_t)heads*128*4,stream);if(rc)return rc;rc=hipMemsetAsync(vmax,0,(size_t)heads*128*4,stream);if(rc)return rc;
    column_stats<<<dim3((nk+kStatChunk-1)/kStatChunk,heads),128,0,stream>>>(ksum,nullptr,kh,nk);
    subtract_mean<<<(kn+255)/256,256,0,stream>>>(kh,ksum,nk,heads);
    quant_qk_int8_hnd_kernel<__hip_bfloat16,128><<<dim3(qgroups+kgroups,heads),256,0,stream>>>(qh,kh,q8,k8,qscale,kscale,1,heads,heads,padded_q,nk,qgroups,kgroups);
    column_stats<<<dim3((nk+kStatChunk-1)/kStatChunk,heads),128,0,stream>>>(nullptr,vmax,vh,nk);
    scales_from_absmax<<<(heads*128+255)/256,256,0,stream>>>(vs,vmax,heads*128);
    transpose_value_fp8_scaled_hnd_kernel<__hip_bfloat16><<<dim3((nk+127)/128,8,heads),256,0,stream>>>(vh,vs,v8,heads,nk,128);
    qk_int8_sv_f8_native_2q_kernel<64,128,0,8,true,128,true,false,__hip_bfloat16><<<dim3((nq+127)/128,heads),128,0,stream>>>(q8,k8,v8,oh,qscale,kscale,vs,1,padded_q,nk,heads,heads,
        qn,128,(int64_t)padded_q*128,kn,128,(int64_t)nk*128,kn,nk,(int64_t)nk*128,qn,128,(int64_t)padded_q*128,
        (int64_t)heads*qgroups,qgroups,(int64_t)heads*kgroups,kgroups,kHND,0.08838834764831845f,false);
    unpack_hnd_f32<<<((size_t)nq*heads*128+255)/256,256,0,stream>>>(out,oh,nq,padded_q,heads,os);
    return hipGetLastError();
}
