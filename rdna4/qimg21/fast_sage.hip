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
    size_t bytes=qn*5+kn*6+(size_t)heads*(qgroups+kgroups+128)*4+1024;
    if(work.device!=dev||work.bytes<bytes) {
        if(work.ptr){rc=hipDeviceSynchronize();if(rc)return rc;rc=hipFree(work.ptr);if(rc)return rc;work.ptr=nullptr;}
        rc=hipMalloc(&work.ptr,bytes);if(rc)return rc;work.bytes=bytes;work.device=dev;
    }
    char*p=(char*)work.ptr;
    auto alloc=[&](size_t n){void*r=p;p+=((n+255)/256)*256;return r;};
    auto oh=(__hip_bfloat16*)alloc(qn*2);auto qh=(__hip_bfloat16*)alloc(qn*2);auto kh=(__hip_bfloat16*)alloc(kn*2);auto vh=(__hip_bfloat16*)alloc(kn*2);
    auto q8=(int8_t*)alloc(qn);auto k8=(int8_t*)alloc(kn);auto v8=(uint8_t*)alloc(kn);
    auto qscale=(float*)alloc((size_t)heads*qgroups*4);auto kscale=(float*)alloc((size_t)heads*kgroups*4);auto vs=(float*)alloc((size_t)heads*128*4);
    rc=hipMemsetAsync(qh,0,qn*2,stream);if(rc)return rc;
    pack_hnd_padded<<<(qn+255)/256,256,0,stream>>>(qh,(const __hip_bfloat16*)q,nq,padded_q,heads,qs);
    pack_hnd<<<(kn+255)/256,256,0,stream>>>(kh,(const __hip_bfloat16*)k,nk,heads,ks);
    pack_hnd<<<(kn+255)/256,256,0,stream>>>(vh,(const __hip_bfloat16*)v,nk,heads,ks);
    smooth_k<<<heads,128,0,stream>>>(kh,nk,heads);
    quant_qk_int8_hnd_kernel<__hip_bfloat16,128><<<dim3(qgroups+kgroups,heads),256,0,stream>>>(qh,kh,q8,k8,qscale,kscale,1,heads,heads,padded_q,nk,qgroups,kgroups);
    value_scales<<<heads,128,0,stream>>>(vs,vh,nk,heads);
    transpose_value_fp8_scaled_hnd_kernel<__hip_bfloat16><<<dim3((nk+127)/128,8,heads),256,0,stream>>>(vh,vs,v8,heads,nk,128);
    qk_int8_sv_f8_native_2q_kernel<64,128,0,8,true,128,true,false,__hip_bfloat16><<<dim3((nq+127)/128,heads),128,0,stream>>>(q8,k8,v8,oh,qscale,kscale,vs,1,padded_q,nk,heads,heads,
        qn,128,(int64_t)padded_q*128,kn,128,(int64_t)nk*128,kn,nk,(int64_t)nk*128,qn,128,(int64_t)padded_q*128,
        (int64_t)heads*qgroups,qgroups,(int64_t)heads*kgroups,kgroups,kHND,0.08838834764831845f,false);
    unpack_hnd_padded<<<((size_t)nq*heads*128+255)/256,256,0,stream>>>((__hip_bfloat16*)out,oh,nq,padded_q,heads,os);
    return hipGetLastError();
}
