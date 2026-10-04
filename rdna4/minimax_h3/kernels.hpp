// SPDX-License-Identifier: MIT
#pragma once
namespace h3 {
#ifdef HV15N_ROCM
inline const char *prelude=R"HIP(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
__device__ float h3_f16r(float v){return __half2float(__float2half(v));}
)HIP";
#else
// NVRTC prelude: wave32 shuffles map to full-warp sync shuffles.
inline const char *prelude=R"CUDA(
#define __shfl_down(v,d) __shfl_down_sync(0xffffffffu,v,d)
#define __shfl_xor(v,d,w) __shfl_xor_sync(0xffffffffu,v,d,w)
__device__ float h3_f16r(float v){unsigned short h;float r;asm("cvt.rn.f16.f32 %0,%1;":"=h"(h):"f"(v));asm("cvt.f32.f16 %0,%1;":"=f"(r):"h"(h));return r;}
extern "C" {
// [rows, heads*dim] F32 -> [heads, rows, dim] 16-bit (kind 1 BF16, 2 FP16), nearest-even.
__global__ void h3_pack_heads(unsigned short *out,const float *x,int rows,int heads,int dim,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*heads*dim)return;
    int d=i%dim,h=(i/dim)%heads;long r=i/((long)heads*dim);unsigned short v;
    if(kind==1)asm("cvt.rn.bf16.f32 %0,%1;":"=h"(v):"f"(x[i]));else asm("cvt.rn.f16.f32 %0,%1;":"=h"(v):"f"(x[i]));
    out[((long)h*rows+r)*dim+d]=v;
}
__global__ void h3_unpack_heads(float *out,const unsigned short *x,int rows,int heads,int dim,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*heads*dim)return;
    int d=i%dim,h=(i/dim)%heads;long r=i/((long)heads*dim);unsigned short v=x[((long)h*rows+r)*dim+d];float f;
    if(kind==1)f=__uint_as_float((unsigned)v<<16);else asm("cvt.f32.f16 %0,%1;":"=f"(f):"h"(v));
    out[i]=f;
}
}
)CUDA";
#endif
inline const char *source=R"HIP(
__device__ float rnd(float v,int kind){
    if(kind==2)return h3_f16r(v);
    if(kind==1){unsigned u=__float_as_uint(v);if((u&0x7f800000u)!=0x7f800000u)u+=0x7fff+((u>>16)&1);return __uint_as_float(u&0xffff0000u);}
    return v;
}
extern "C" {
// Converts exact INT32 GEMM sums in place, matching video_gemm_i8's epilogue.
__global__ void h3_dequant(float *y,const float *xs,const float *ws,int m,int n,int rounded){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)m*n)return;
    float v=(float)__float_as_int(y[i])*(xs[i/n]*ws[i%n]);y[i]=rounded?rnd(v,1):v;
}
// Fused linear epilogue: optional bias add, then activation-dtype rounding (in place).
__global__ void h3_bias_round(float *y,const float *b,long n,int c,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=n)return;float v=y[i];if(b)v+=b[i%c];y[i]=rnd(v,kind);
}
__global__ void h3_round(float *x,long n,int kind){long i=(long)blockIdx.x*256+threadIdx.x;if(i<n)x[i]=rnd(x[i],kind);}
__global__ void h3_qwen_angles(float *out,int rows){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*64)return;
    int r=i/64,p=i%64;float a=float(r)*(1.f/powf(5000000.f,float(p)/64.f));
    out[i*2]=cosf(a);out[i*2+1]=sinf(a);
}
__global__ void h3_angles(float *out,const float *phases,long n,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i<n){out[i*2]=rnd(cosf(phases[i]),kind);out[i*2+1]=rnd(sinf(phases[i]),kind);}
}
__global__ void h3_pack_bf16(unsigned short *out,const float *x,long n){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i<n)out[i]=__float_as_uint(rnd(x[i],1))>>16;
}
__global__ void h3_rotate(float *out,const float *x,int groups,int kind){
    int g=blockIdx.x,i=threadIdx.x;if(g>=groups)return;
    __shared__ float a[256],b[256];a[i]=x[g*256+i];__syncthreads();
    for(int stride=1;stride<=64;stride*=4){
        int d=(i/stride)%4,base=i-d*stride;float v=0;
        for(int j=0;j<4;j++)v+=(j==3-d?-1.f:1.f)*a[base+j*stride];
        b[i]=v*.5f;__syncthreads();a[i]=b[i];__syncthreads();
    }
    out[g*256+i]=rnd(a[i],kind);
}
__global__ void h3_quant(signed char *out,float *scales,const float *x,int rows,int k,int kind){
    int row=blockIdx.x,tid=threadIdx.x;if(row>=rows)return;
    __shared__ float a[256];float mx=0;
    for(int c=tid;c<k;c+=256)mx=fmaxf(mx,fabsf(x[(long)row*k+c]));a[tid]=mx;__syncthreads();
    for(int d=128;d;d/=2){if(tid<d)a[tid]=fmaxf(a[tid],a[tid+d]);__syncthreads();}
    // PyTorch scalar division multiplies by its FP32 reciprocal. Preserve
    // that operation order: one-ULP scale changes can cross BF16 output ties.
    float scale=fmaxf(a[0]*(1.f/127.f),1e-30f);if(tid==0)scales[row]=scale;
    for(int c=tid;c<k;c+=256){float math_scale=rnd(scale,kind);if(math_scale==0)math_scale=kind==2?6.103515625e-5f:1.17549435e-38f;float v=rnd(x[(long)row*k+c]/math_scale,kind);int q=(int)rintf(v);out[(long)row*k+c]=(signed char)max(-128,min(127,q));}
}
__global__ void h3_norm(float *out,const float *x,const float *w,const float *bias,int rows,int k,int mode,float eps,int kind){
    int row=blockIdx.x,tid=threadIdx.x;if(row>=rows)return;
    __shared__ float a[256],b[256];float sum=0,sq=0;
    if(mode && k%4==0){
        // Match the pinned PyTorch RMSNorm: four adjacent values per thread,
        // shuffle-down within wave32, then an eight-wave tree reduction.
        for(int c=tid*4;c<k;c+=256*4)
            for(int j=0;j<4;j++){float v=x[(long)row*k+c+j];sq+=v*v;}
        for(int d=16;d;d/=2)sq+=__shfl_down(sq,d);
        if((tid&31)==0)b[tid/32]=sq;__syncthreads();
        for(int d=4;d;d/=2){if(tid<d)b[tid]+=b[tid+d];__syncthreads();}
    }else{
        for(int c=tid;c<k;c+=256){float v=x[(long)row*k+c];sum+=v;sq+=v*v;}a[tid]=sum;b[tid]=sq;__syncthreads();
        for(int d=128;d;d/=2){if(tid<d){a[tid]+=a[tid+d];b[tid]+=b[tid+d];}__syncthreads();}
    }
    float mean=mode?0:a[0]/k,inv=rsqrtf(fmaxf(b[0]/k-mean*mean,0.f)+eps);
    for(int c=tid;c<k;c+=256){float v=(x[(long)row*k+c]-mean)*inv;if(mode){if(mode==1)v=rnd(v,kind);v=rnd(v*(w?w[c]:1),kind);if(bias)v=rnd(v+bias[c],kind);}else v=rnd(v*(w?w[c]:1)+(bias?bias[c]:0),kind);out[(long)row*k+c]=v;}
}
__global__ void h3_swiglu(float *out,const float *x,int rows,int k,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*k)return;int row=i/k,c=i%k;
    float a=x[(long)row*k*2+c],b=x[(long)row*k*2+k+c];out[i]=rnd(rnd(a/(1.f+expf(-a)),kind)*b,kind);
}
__global__ void h3_qkv(float *q,float *k,float *v,const float *x,int rows,int heads,int dim,int interleaved){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*heads*dim)return;
    int row=i/(heads*dim),head=(i/dim)%heads,d=i%dim;
    long base=interleaved?((long)row*heads+head)*3*dim:(long)row*heads*3*dim+head*dim;
    q[i]=x[base+d];k[i]=x[base+dim*(interleaved?1:heads)+d];v[i]=x[base+dim*(interleaved?2:2*heads)+d];
}
__global__ void h3_qwen_attention(float *out,const float *q,const float *k,const float *v,int rows,int heads,int kvheads,int dim,float root_scale){
    int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,kh=head/(heads/kvheads);
    __shared__ float scores[512],prob[512];
    for(int j=lane;j<rows;j+=32){
        float dot=-__int_as_float(0x7f800000);
        if(j<=row){dot=0;
            for(int d=0;d<dim;d++)dot=fmaf(q[((long)row*heads+head)*dim+d]*root_scale,k[((long)j*kvheads+kh)*dim+d]*root_scale,dot);
        }
        scores[j]=dot;
    }
    __syncthreads();
    int width=1;while(width<rows && width<32)width*=2;
    int pos=lane%width;float mx=-__int_as_float(0x7f800000);
    for(int j=pos;j<rows;j+=width)mx=fmaxf(mx,scores[j]);
    for(int d=width/2;d;d/=2)mx=fmaxf(mx,__shfl_xor(mx,d,width));
    float sum=0;
    for(int j=pos;j<rows;j+=width){float e=expf(scores[j]-mx);if(lane<width)prob[j]=e;sum+=e;}
    for(int d=width/2;d;d/=2)sum+=__shfl_xor(sum,d,width);
    if(lane<width)for(int j=pos;j<rows;j+=width)prob[j]/=sum;
    __syncthreads();
    for(int d=lane;d<dim;d+=32){float acc=0;
        for(int j=0;j<rows;j++)acc=fmaf(prob[j],v[((long)j*kvheads+kh)*dim+d],acc);
        out[((long)row*heads+head)*dim+d]=rnd(acc,1);
    }
}
__global__ void h3_rope(float *x,const float *angles,int rows,int heads,int dim,int pairs,int kind,int eager){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*heads*pairs)return;
    int p=i%pairs,h=(i/pairs)%heads,r=i/(heads*pairs);long at=((long)r*heads+h)*dim;
    float c=angles[((long)r*pairs+p)*2],s=angles[((long)r*pairs+p)*2+1],a=x[at+p],b=x[at+p+pairs];
    x[at+p]=rnd((eager?rnd(a*c,kind):a*c)-b*s,kind);x[at+p+pairs]=rnd((eager?rnd(b*c,kind):b*c)+a*s,kind);
}
__global__ void h3_mod(float *out,const float *x,const float *mod,int rows,int dim,int text,int audio,int chunk,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*dim)return;
    int row=i/dim,c=i%dim,m=row<text?1:row<text+audio?5:0; // two timestep rows, three modalities each
    long base=(long)m*6*dim;
    float scale=rnd(mod[base+(chunk+1)*dim+c],kind),shift=rnd(mod[base+chunk*dim+c],kind);
    out[i]=rnd(rnd(x[i]*rnd(1.f+scale,kind),kind)+shift,kind);
}
__global__ void h3_gate(float *x,const float *delta,const float *mod,int rows,int dim,int text,int audio,int chunk,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*dim)return;
    int r=i/dim,c=i%dim,m=r<text?1:r<text+audio?5:0;
    x[i]=rnd(x[i]+delta[i]*rnd(mod[(long)m*6*dim+chunk*dim+c],kind),kind);
}
__global__ void h3_scale_add(float *x,const float *delta,const float *scale,int rows,int k,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i<(long)rows*k)x[i]=rnd(x[i]+delta[i]*scale[i%k],kind);
}
__global__ void h3_patch(float *out,const float *x,int t,int h,int w){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)t*h*w*24)return;
    int f=i/96,feature=i%96,c=feature/4,dy=(feature%4)/2,dx=feature%2;
    int tt=f/((h/2)*(w/2)),yy=(f/(w/2))%(h/2),xx=f%(w/2);
    out[i]=x[(((long)tt*h+yy*2+dy)*w+xx*2+dx)*24+c];
}
__global__ void h3_unpatch(float *out,const float *x,int t,int h,int w){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)t*h*w*24)return;
    int c=i%24,xx=(i/24)%w,yy=(i/(24*w))%h,tt=i/(24*w*h);
    out[i]=x[(((long)tt*(h/2)+yy/2)*(w/2)+xx/2)*96+c*4+(yy%2)*2+xx%2];
}
__global__ void h3_decode_patch(float *out,const float *x,int t,int h,int w){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)t*4*h*16*w*16*3)return;
    int c=i%3,xx=(i/3)%(w*16),yy=(i/(3*w*16))%(h*16),tt=i/(3*w*16*h*16);
    int feature=((c*4+tt%4)*16+yy%16)*16+xx%16;
    out[i]=x[(((long)(tt/4)*h+yy/16)*w+xx/16)*3072+feature];
}
}
)HIP";

#ifndef HV15N_ROCM
// CUDA-only fused kernels. Each reproduces the arithmetic of the unfused chain it
// replaces (same expressions, same reduction order), so results are bit-identical.
inline const char *fused_source=R"CUDA(
__device__ __forceinline__ unsigned short h3_bits16(float v,int kind){
    unsigned short h;if(kind==1)asm("cvt.rn.bf16.f32 %0,%1;":"=h"(h):"f"(v));else asm("cvt.rn.f16.f32 %0,%1;":"=h"(h):"f"(v));return h;
}
__device__ __forceinline__ float h3_from16(unsigned short h,int kind){
    if(kind==1)return __uint_as_float((unsigned)h<<16);float f;asm("cvt.f32.f16 %0,%1;":"=f"(f):"h"(h));return f;
}
// Deferred linear epilogue: INT32 sums (h3_dequant) or raw F32 (h3_bias_round).
__device__ __forceinline__ float h3_lin(const float *y,const float *xs,const float *ws,const float *b,
                                        long row,long n,long col,int i32,int kind){
    float v;
    if(i32){v=(float)__float_as_int(y[row*n+col])*(xs[row]*ws[col]);if(kind==1)v=rnd(v,1);if(b){v+=b[col];v=rnd(v,kind);}}
    else{v=y[row*n+col];if(b)v+=b[col];v=rnd(v,kind);}
    return v;
}
extern "C" {
// h3_norm (mode 2 RMSNorm with optional weight) writing 16-bit activations.
__global__ void h3x_norm16(unsigned short *out,const float *x,const float *w,int rows,int k,float eps,int kind){
    int row=blockIdx.x,tid=threadIdx.x;if(row>=rows)return;
    __shared__ float b[256];float sq=0;
    for(int c=tid*4;c<k;c+=256*4)
        for(int j=0;j<4;j++){float v=x[(long)row*k+c+j];sq+=v*v;}
    for(int d=16;d;d/=2)sq+=__shfl_down(sq,d);
    if((tid&31)==0)b[tid/32]=sq;__syncthreads();
    for(int d=4;d;d/=2){if(tid<d)b[tid]+=b[tid+d];__syncthreads();}
    float mean=0,inv=rsqrtf(fmaxf(b[0]/k-mean*mean,0.f)+eps);
    for(int c=tid;c<k;c+=256){float v=(x[(long)row*k+c]-mean)*inv;v=rnd(v*(w?w[c]:1),kind);out[(long)row*k+c]=h3_bits16(v,kind);}
}
// h3_qkv + per-head h3_norm (mode 2) + h3_rope (eager) + h3_pack_heads, one warp per (row, head).
__global__ void h3x_qkv_pack(unsigned short *pq,unsigned short *pk,unsigned short *pv,const float *y,
        const float *xs,const float *ws,const float *bias,int i32,int rows,int heads,int dim,int interleaved,
        const float *qw,const float *kw,const float *angles,int pairs,int kind,float eps){
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    long item=(long)blockIdx.x*4+warp;if(item>=(long)rows*heads)return;
    int row=item/heads,head=item%heads;
    __shared__ float buf[4][128];float *v=buf[warp];
    long n=(long)heads*3*dim;
    long base=interleaved?(long)head*3*dim:(long)head*dim;
    long off[3]={base,base+(long)dim*(interleaved?1:heads),base+(long)dim*(interleaved?2:2*heads)};
    unsigned short *dst[3]={pq,pk,pv};
    for(int part=0;part<3;part++){
        for(int d=lane;d<dim;d+=32)v[d]=h3_lin(y,xs,ws,bias,row,n,off[part]+d,i32,kind);
        __syncwarp();
        if(part<2){
            const float *w=part?kw:qw;
            float sq=0;
            if(lane*4<dim)for(int j=0;j<4;j++){float t=v[lane*4+j];sq+=t*t;}
            for(int d=16;d;d/=2)sq+=__shfl_down(sq,d);
            sq=__shfl_sync(0xffffffffu,sq,0);
            float mean=0,inv=rsqrtf(fmaxf(sq/dim-mean*mean,0.f)+eps);
            __syncwarp();
            for(int d=lane;d<dim;d+=32){float t=(v[d]-mean)*inv;v[d]=rnd(t*(w?w[d]:1),kind);}
            __syncwarp();
            if(angles){
                for(int p=lane;p<pairs;p+=32){
                    float c=angles[((long)row*pairs+p)*2],s=angles[((long)row*pairs+p)*2+1],a=v[p],b=v[p+pairs];
                    v[p]=rnd(rnd(a*c,kind)-b*s,kind);v[p+pairs]=rnd(rnd(b*c,kind)+a*s,kind);
                }
                __syncwarp();
            }
        }
        for(int d=lane;d<dim;d+=32)dst[part][((long)head*rows+row)*dim+d]=h3_bits16(v[d],kind);
        __syncwarp();
    }
}
// [heads, rows, dim] -> [rows, heads*dim], 16-bit to 16-bit.
__global__ void h3x_unpack16(unsigned short *out,const unsigned short *x,int rows,int heads,int dim){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*heads*dim)return;
    int d=i%dim,h=(i/dim)%heads;long r=i/((long)heads*dim);out[i]=x[((long)h*rows+r)*dim+d];
}
// h3_bias_round + h3_scale_add: x = rnd(x + rnd(y + b) * s).
__global__ void h3x_scale_add(float *x,const float *y,const float *bias,const float *scale,int rows,int k,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*k)return;
    float d=h3_lin(y,nullptr,nullptr,bias,i/k,k,i%k,0,kind);x[i]=rnd(x[i]+d*scale[i%k],kind);
}
// h3_bias_round + h3_swiglu writing 16-bit activations.
__global__ void h3x_swiglu16(unsigned short *out,const float *y,const float *bias,int rows,int k,int kind){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*k)return;int row=i/k,c=i%k;
    float a=h3_lin(y,nullptr,nullptr,bias,row,2*k,c,0,kind),b=h3_lin(y,nullptr,nullptr,bias,row,2*k,k+c,0,kind);
    out[i]=h3_bits16(rnd(rnd(a/(1.f+expf(-a)),kind)*b,kind),kind);
}
// h3_dequant (+bias) + h3_gate on rows [row0, row0+rows): x = rnd(x + delta * rnd(mod)).
__global__ void h3x_dequant_gate(float *x,const float *y,const float *xs,const float *ws,const float *bias,
        const float *mod,int rows,int dim,int row0,int text,int audio,int chunk){
    long i=(long)blockIdx.x*256+threadIdx.x;if(i>=(long)rows*dim)return;
    int lr=i/dim,c=i%dim,r=row0+lr,m=r<text?1:r<text+audio?5:0;
    float delta=h3_lin(y,xs,ws,bias,lr,dim,c,1,1);long at=(long)r*dim+c;
    x[at]=rnd(x[at]+delta*rnd(mod[(long)m*6*dim+chunk*dim+c],1),1);
}
// Factorized ConvRot-256 (h3_rotate) + h3_quant for one row per 256-thread block, with
// an input prologue: 0 plain F32 rows, 1 RMSNorm+modulate (h3_norm mode 2 + h3_mod),
// 2 SwiGLU of deferred INT32 fc1 sums, 3 packed BF16 attention heads [H,S,D].
__global__ void h3x_row_quant(signed char *out,float *scales,int rows,int k,int mode,const float *x,
        const float *nw,const float *mod,int row0,int text,int audio,int chunk,float eps,
        const float *acc,const float *axs,const float *aws,const unsigned short *heads,int nheads,int hdim,int hrows){
    int lr=blockIdx.x,tid=threadIdx.x;if(lr>=rows)return;
    int r=row0+lr;
    __shared__ unsigned short rot[14336];
    __shared__ float a[256],b[256],red[256];
    float inv=0,mean=0;
    if(mode==1){
        float sq=0;
        for(int c=tid*4;c<k;c+=256*4)
            for(int j=0;j<4;j++){float v=x[(long)r*k+c+j];sq+=v*v;}
        for(int d=16;d;d/=2)sq+=__shfl_down(sq,d);
        if((tid&31)==0)b[tid/32]=sq;__syncthreads();
        for(int d=4;d;d/=2){if(tid<d)b[tid]+=b[tid+d];__syncthreads();}
        inv=rsqrtf(fmaxf(b[0]/k-mean*mean,0.f)+eps);
        __syncthreads();
    }
    int m=r<text?1:r<text+audio?5:0;long mbase=(long)m*6*k;
    float mx=0;
    for(int g=0;g<k/256;g++){
        int c=g*256+tid;float val;
        if(mode==0)val=x[(long)r*k+c];
        else if(mode==1){
            float v=(x[(long)r*k+c]-mean)*inv;v=rnd(v*(nw?nw[c]:1),1);
            float scale=rnd(mod[mbase+(chunk+1)*k+c],1),shift=rnd(mod[mbase+chunk*k+c],1);
            val=rnd(rnd(v*rnd(1.f+scale,1),1)+shift,1);
        }else if(mode==2){
            float ga=h3_lin(acc,axs,aws,nullptr,lr,2*k,c,1,1),gb=h3_lin(acc,axs,aws,nullptr,lr,2*k,k+c,1,1);
            val=rnd(rnd(ga/(1.f+expf(-ga)),1)*gb,1);
        }else val=h3_from16(heads[((long)(c/hdim)*hrows+r)*hdim+c%hdim],1);
        int i=tid;a[i]=val;__syncthreads();
        for(int stride=1;stride<=64;stride*=4){
            int d=(i/stride)%4,base=i-d*stride;float v=0;
            for(int j=0;j<4;j++)v+=(j==3-d?-1.f:1.f)*a[base+j*stride];
            b[i]=v*.5f;__syncthreads();a[i]=b[i];__syncthreads();
        }
        float o=rnd(a[i],1);rot[c]=h3_bits16(o,1);mx=fmaxf(mx,fabsf(o));
        __syncthreads();
    }
    red[tid]=mx;__syncthreads();
    for(int d=128;d;d/=2){if(tid<d)red[tid]=fmaxf(red[tid],red[tid+d]);__syncthreads();}
    float scale=fmaxf(red[0]*(1.f/127.f),1e-30f);if(tid==0)scales[lr]=scale;
    for(int c=tid;c<k;c+=256){float math_scale=rnd(scale,1);if(math_scale==0)math_scale=1.17549435e-38f;float v=rnd(h3_from16(rot[c],1)/math_scale,1);int q=(int)rintf(v);out[(long)lr*k+c]=(signed char)max(-128,min(127,q));}
}
}
)CUDA";
#else
inline const char *fused_source="";
#endif
}
