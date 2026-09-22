/* Qwen Image 2.1 HIP compatibility layer.
 *
 * The Qwen 2.1 host graph is intentionally kept separate from the older
 * rdna4/qimg runner: its checkpoint is 32 blocks, hidden size 4096, 32 heads,
 * and BF16 rather than the legacy 60-block/3072 model ABI.  This small layer
 * supplies the CUDA-driver-shaped surface used by the shared Qwen 2.1 graph,
 * backed by dynamically loaded HIP and HIPRTC calls.
 */
#ifndef HIP_QIMG21_RUNNER_H
#define HIP_QIMG21_RUNNER_H

#include "../rocew.h"
#include "../hip_kernels_common.h"
#include "../qimg/hip_qimg_kernels.h"
#include "../../common/safetensors.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t qimg_f32_to_bf16_rne(float x) {
    uint32_t bits; memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

typedef int CUresult;
typedef int CUdevice;
typedef void *CUdeviceptr;
typedef hipCtx_t CUcontext;
typedef hipStream_t CUstream;
typedef hipModule_t CUmodule;
typedef hipFunction_t CUfunction;

#define CUDA_SUCCESS hipSuccess
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES hipFuncAttributeMaxDynamicSharedMemorySize
#define cuMemAlloc(p,n) hipMalloc((void **)(p), (n))
#define cuMemFree(p) hipFree((void *)(p))
#define cuMemcpyHtoD(d,s,n) hipMemcpy((void *)(d), (const void *)(s), (n), hipMemcpyHostToDevice)
#define cuMemcpyDtoH(d,s,n) hipMemcpy((void *)(d), (const void *)(s), (n), hipMemcpyDeviceToHost)
#define cuMemcpyDtoD(d,s,n) hipMemcpy((void *)(d), (const void *)(s), (n), hipMemcpyDeviceToDevice)
#define cuMemcpyHtoDAsync(d,s,n,st) hipMemcpyAsync((void *)(d), (const void *)(s), (n), hipMemcpyHostToDevice, (st))
#define cuMemcpyDtoHAsync(d,s,n,st) hipMemcpyAsync((void *)(d), (const void *)(s), (n), hipMemcpyDeviceToHost, (st))
/* ROCm 7.x/10.x keeps the CUDA-shaped context entry point for source
 * compatibility, but on some installations hipCtxSynchronize() returns
 * hipErrorNotSupported (801) for runtime-created contexts.  Device-wide
 * synchronization is the supported equivalent and also covers module API
 * launches on the active device. */
#define cuCtxSynchronize() hipDeviceSynchronize()
#define cuStreamSynchronize(s) hipStreamSynchronize(s)
#define cuLaunchKernel hipModuleLaunchKernel
#define cuModuleGetFunction hipModuleGetFunction
#define cuModuleUnload hipModuleUnload
#define cuFuncSetAttribute hipFuncSetAttribute

typedef struct cuda_qimg_runner {
    CUdevice device;
    CUstream stream;
    CUfunction cast_f32_to_bf16;
    CUfunction bf16_to_f32_add_bias;
    CUfunction gemm_bf16;
    CUfunction gemm_bf16_f64;
    CUfunction gemm_bf16_wmma;
    CUfunction gemm_int8_s32;
    CUfunction quant_act_perrow_int8;
    CUfunction dequant_int32_to_bf16;
    CUfunction euler_step;
    CUfunction vae_conv2d;
    CUfunction vae_conv2d_3x3_wmma;
    CUfunction vae_conv2d_1x1_wmma;
    CUfunction vae_rmsnorm;
    CUfunction vae_silu;
    CUfunction vae_up2x;
    CUfunction vae_transpose_chw_to_sc;
    CUfunction vae_transpose_sc_to_chw;
    CUfunction vae_attn_sc;
    CUfunction truncate_bf16;
    int vae_rms_contiguous_channels;
    CUmodule common_module;
    CUmodule wmma_module;
    CUmodule vae_module;
    void *cublaslt_ctx;
    int use_int8;
    int verbose;
} cuda_qimg_runner;

static const char *hip_qimg21_common_src =
"extern \"C\" __global__ void q21_cast_f32_bf16(const float* x, unsigned short* y, int n){"
"int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int b;memcpy(&b,x+i,4);"
"b+=0x7fffu+((b>>16)&1u);y[i]=(unsigned short)(b>>16);}\n"
"extern \"C\" __global__ void q21_gemm_bf16(float* y,const unsigned short* w,"
"const unsigned short* x,int no,int ni,int nt){int j=blockIdx.x*blockDim.x+threadIdx.x;"
"int i=blockIdx.y*blockDim.y+threadIdx.y;if(j>=no||i>=nt)return;float a=0;"
"for(int k=0;k<ni;k++){unsigned int xb=(unsigned int)x[i*ni+k]<<16;"
"unsigned int wb=(unsigned int)w[j*ni+k]<<16;float xf,wf;memcpy(&xf,&xb,4);memcpy(&wf,&wb,4);a+=xf*wf;}"
"y[i*no+j]=a;}\n"
"extern \"C\" __global__ void q21_gemm_bf16_f64(float* y,const unsigned short* w,"
"const unsigned short* x,int no,int ni,int nt){int j=blockIdx.x*blockDim.x+threadIdx.x;"
"int i=blockIdx.y;if(j>=no||i>=nt)return;double a=0;"
"for(int k=0;k<ni;k++){unsigned int xb=(unsigned int)x[i*ni+k]<<16;"
"unsigned int wb=(unsigned int)w[j*ni+k]<<16;float xf,wf;memcpy(&xf,&xb,4);memcpy(&wf,&wb,4);"
"a+=(double)xf*(double)wf;}y[i*no+j]=(float)a;}\n";

static const char *hip_qimg21_wmma_src =
"#if defined(__gfx1200__) || defined(__gfx1201__)\n"
"extern \"C\" __global__ void q21_gemm_bf16_wmma(float*Y,const unsigned short*W,const unsigned short*X,int no,int ni,int nt){"
"int tid=threadIdx.x,wave=tid>>5,lane=tid&31,wM=wave&1,wN=wave>>1,half=lane>>4,idx=lane&15;"
"int m0=blockIdx.y*128,n0=blockIdx.x*128;__shared__ short A[128*16],B[128*16];"
"typedef float float8 __attribute__((ext_vector_type(8)));typedef short bf16x8 __attribute__((ext_vector_type(8)));"
"float8 c00={0,0,0,0,0,0,0,0},c01=c00,c10=c00,c11=c00,c20=c00,c21=c00,c30=c00,c31=c00;"
"for(int k=0;k<ni;k+=16){for(int it=0;it<8;it++){int e=tid*8+it,r=e>>4,q=e&15,rr=m0+r,kk=k+q;A[r*16+q]=(rr<nt&&kk<ni)?X[rr*ni+kk]:0;}"
"for(int it=0;it<8;it++){int e=tid*8+it,r=e>>4,q=e&15,cc=n0+r,kk=k+q;B[r*16+q]=(cc<no&&kk<ni)?W[cc*ni+kk]:0;}__syncthreads();"
"int am=wM*64,bm=wN*32;bf16x8 a0,a1,a2,a3,b0,b1;for(int i=0;i<8;i++){a0[i]=A[(am+idx)*16+half*8+i];a1[i]=A[(am+16+idx)*16+half*8+i];a2[i]=A[(am+32+idx)*16+half*8+i];a3[i]=A[(am+48+idx)*16+half*8+i];b0[i]=B[(bm+idx)*16+half*8+i];b1[i]=B[(bm+16+idx)*16+half*8+i];}"
"c00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,c00);c01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,c01);"
"c10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,c10);c11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,c11);"
"c20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,c20);c21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,c21);"
"c30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,c30);c31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,c31);__syncthreads();}"
"float8*cs[8]={&c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31};int ms[8]={0,0,16,16,32,32,48,48},ns[8]={0,16,0,16,0,16,0,16};"
"for(int t=0;t<8;t++){int col=n0+wN*32+ns[t]+idx;if(col>=no)continue;float8 v=*cs[t];for(int i=0;i<8;i++){int row=m0+wM*64+ms[t]+half*8+i;if(row<nt)Y[row*no+col]=v[i];}}}\n"
"#else\n"
"extern \"C\" __global__ void q21_gemm_bf16_wmma(float*Y,const unsigned short*W,const unsigned short*X,int no,int ni,int nt){int j=blockIdx.x*blockDim.x+threadIdx.x,i=blockIdx.y;if(j>=no||i>=nt)return;float a=0;for(int k=0;k<ni;k++){unsigned xb=(unsigned)X[i*ni+k]<<16,wb=(unsigned)W[j*ni+k]<<16;float xf,wf;memcpy(&xf,&xb,4);memcpy(&wf,&wb,4);a+=xf*wf;}Y[i*no+j]=a;}\n"
"#endif\n";

/* The Qwen VAE attention graph uses [spatial, channel] scratch buffers. */
static const char *hip_qimg21_vae_extra_src =
"extern \"C\" __global__ void q21_transpose_chw_to_sc(float *o,const float *x,int C,int S){"
"int c=blockIdx.y*blockDim.y+threadIdx.y,s=blockIdx.x*blockDim.x+threadIdx.x;"
"if(c<C&&s<S)o[s*C+c]=x[c*S+s];}\n"
"extern \"C\" __global__ void q21_transpose_sc_to_chw(float *o,const float *x,int C,int S){"
"int c=blockIdx.y*blockDim.y+threadIdx.y,s=blockIdx.x*blockDim.x+threadIdx.x;"
"if(c<C&&s<S)o[c*S+s]=x[s*C+c];}\n"
"extern \"C\" __global__ void q21_vae_attn_sc(float *o,const float *q,const float *k,const float *v,int S,int C,float scale){"
"int qi=blockIdx.x,lane=threadIdx.x;if(qi>=S)return;float qv[64];int ept=C/32;"
"for(int e=0;e<ept;e++)qv[e]=q[qi*C+lane*ept+e];float m=-1e30f,l=0;float ov[64];"
"for(int e=0;e<ept;e++)ov[e]=0;for(int kj=0;kj<S;kj++){float d=0;"
"for(int e=0;e<ept;e++)d+=qv[e]*k[kj*C+lane*ept+e];for(int z=16;z;z>>=1)d+=__shfl_xor(d,z);"
"float score=d*scale,n=fmaxf(m,score),a=expf(m-n),p=expf(score-n);l=l*a+p;"
"for(int e=0;e<ept;e++)ov[e]=ov[e]*a+p*v[kj*C+lane*ept+e];m=n;}"
"float inv=l>0?1.0f/l:0;for(int e=0;e<ept;e++)o[qi*C+lane*ept+e]=ov[e]*inv;}\n";

static void *checked_cuMemAlloc(size_t nbytes) {
    void *p = NULL;
    hipError_t rc = hipMalloc(&p, nbytes);
    if (rc != hipSuccess) {
        const char *es = "?";
        if (hipGetErrorString) hipGetErrorString(rc, &es);
        fprintf(stderr, "qimg21-hip: hipMalloc(%zu) failed: %d (%s)\n",
                nbytes, (int)rc, es);
        return NULL;
    }
    return p;
}

static int hip_qimg21_init_kernels(cuda_qimg_runner *r) {
    CUmodule module = NULL;
    if (hip_compile_kernels(&module, r->device, hip_qimg21_common_src,
                            "qimg21_hip_common.hip", r->verbose,
                            "qimg21-hip") < 0) {
        fprintf(stderr, "qimg21-hip: common module compilation/loading failed\n");
        return -1;
    }
    hipError_t cast_rc = hipModuleGetFunction(&r->cast_f32_to_bf16, module, "q21_cast_f32_bf16");
    hipError_t gemm_rc = hipModuleGetFunction(&r->gemm_bf16, module, "q21_gemm_bf16");
    hipError_t f64_rc = hipModuleGetFunction(&r->gemm_bf16_f64, module, "q21_gemm_bf16_f64");
    if (cast_rc != hipSuccess || gemm_rc != hipSuccess || f64_rc != hipSuccess) {
        const char *cast_es = "?", *gemm_es = "?", *f64_es = "?";
        if (hipGetErrorString) {
            hipGetErrorString(cast_rc, &cast_es);
            hipGetErrorString(gemm_rc, &gemm_es);
            hipGetErrorString(f64_rc, &f64_es);
        }
        fprintf(stderr, "qimg21-hip: common functions unavailable: cast=%d (%s) gemm=%d (%s) f64=%d (%s)\n",
                (int)cast_rc, cast_es, (int)gemm_rc, gemm_es, (int)f64_rc, f64_es);
        hipModuleUnload(module); return -1;
    }
    r->common_module = module;
    if (hip_compile_kernels(&r->wmma_module, r->device, hip_qimg21_wmma_src,
                            "qimg21_hip_wmma.hip", r->verbose,
                            "qimg21-hip-wmma") >= 0) {
        hipError_t wmma_rc = hipModuleGetFunction(&r->gemm_bf16_wmma,
                                                   r->wmma_module,
                                                   "q21_gemm_bf16_wmma");
        if (wmma_rc != hipSuccess) {
            const char *es = "?";
            if (hipGetErrorString) hipGetErrorString(wmma_rc, &es);
            fprintf(stderr, "qimg21-hip-wmma: kernel unavailable: %d (%s); using BF16 fallback\n",
                    (int)wmma_rc, es);
            r->gemm_bf16_wmma = NULL;
        }
    }

    size_t a = strlen(hip_kernels_common_src), b = strlen(hip_qimg_specific_kernels),
           c = strlen(hip_qimg21_vae_extra_src);
    char *vae_src = (char *)malloc(a + b + c + 1);
    if (!vae_src) return -1;
    memcpy(vae_src, hip_kernels_common_src, a);
    memcpy(vae_src + a, hip_qimg_specific_kernels, b);
    memcpy(vae_src + a + b, hip_qimg21_vae_extra_src, c);
    vae_src[a + b + c] = 0;
    int vrc = hip_compile_kernels(&r->vae_module, r->device, vae_src,
                                  "qimg21_hip_vae.hip", r->verbose,
                                  "qimg21-hip-vae");
    free(vae_src);
    if (vrc < 0) {
        fprintf(stderr, "qimg21-hip: VAE module compilation/loading failed\n");
        return -1;
    }
#define VAE_GET(dst, name) do { \
        hipError_t _rc = hipModuleGetFunction(&(dst), r->vae_module, (name)); \
        if (_rc != hipSuccess) { \
            const char *_es = "?"; if (hipGetErrorString) hipGetErrorString(_rc, &_es); \
            fprintf(stderr, "qimg21-hip: missing VAE kernel %s: %d (%s)\n", (name), (int)_rc, _es); \
            return -1; \
        } \
    } while (0)
    VAE_GET(r->euler_step, "euler_step_f32");
    VAE_GET(r->vae_conv2d, "vae_conv2d_f32");
    VAE_GET(r->vae_conv2d_3x3_wmma, "vae_conv2d_3x3_wmma_f32");
    VAE_GET(r->vae_conv2d_1x1_wmma, "vae_conv2d_1x1_wmma_f32");
    VAE_GET(r->vae_rmsnorm, "vae_rmsnorm_f32");
    VAE_GET(r->vae_silu, "vae_silu_f32");
    VAE_GET(r->vae_up2x, "nn_upsample2x_f32");
    VAE_GET(r->truncate_bf16, "truncate_bf16_f32");
    VAE_GET(r->vae_transpose_chw_to_sc, "q21_transpose_chw_to_sc");
    VAE_GET(r->vae_transpose_sc_to_chw, "q21_transpose_sc_to_chw");
    VAE_GET(r->vae_attn_sc, "q21_vae_attn_sc");
#undef VAE_GET
    return 0;
}

static cuda_qimg_runner *cuda_qimg_init(int device_id, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) return NULL;
    if (hipSetDevice(device_id) != hipSuccess) return NULL;
    cuda_qimg_runner *r = (cuda_qimg_runner *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->device = device_id; r->verbose = verbose;
    if (hipStreamCreate(&r->stream) != hipSuccess || hip_qimg21_init_kernels(r) != 0) {
        if (r->vae_module) hipModuleUnload(r->vae_module);
        if (r->wmma_module) hipModuleUnload(r->wmma_module);
        if (r->common_module) hipModuleUnload(r->common_module);
        if (r->stream) hipStreamDestroy(r->stream);
        free(r);
        return NULL;
    }
    return r;
}

static void cuda_qimg_free(cuda_qimg_runner *r) {
    if (!r) return;
    if (r->wmma_module) hipModuleUnload(r->wmma_module);
    if (r->vae_module) hipModuleUnload(r->vae_module);
    if (r->common_module) hipModuleUnload(r->common_module);
    if (r->stream) hipStreamDestroy(r->stream);
    free(r);
}

static void vae_bf16(cuda_qimg_runner *r, void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->truncate_bf16, (unsigned)((n + 255) / 256), 1, 1,
                          256, 1, 1, 0, r->stream, args, NULL);
}

static void vae_op_conv2d(cuda_qimg_runner *r, void *out, void *inp, void *w, void *b,
                          int ci, int h, int ws, int co, int kh, int kw, int rep) {
    int n = co * h * ws;
    if ((ci % 16) == 0 && (co % 16) == 0 && (h * ws % 16) == 0) {
        unsigned gx = (unsigned)(co / 16), gy = (unsigned)(h * ws / 16);
        if (kh == 3 && kw == 3) {
            void *args[] = {&out,&inp,&w,&b,&ci,&h,&ws,&co,&rep};
            hipModuleLaunchKernel(r->vae_conv2d_3x3_wmma,gx,gy,1,32,1,1,0,r->stream,args,NULL);
            return;
        }
        if (kh == 1 && kw == 1) {
            void *args[] = {&out,&inp,&w,&b,&ci,&h,&ws,&co};
            hipModuleLaunchKernel(r->vae_conv2d_1x1_wmma,gx,gy,1,32,1,1,0,r->stream,args,NULL);
            return;
        }
    }
    void *args[] = {&out,&inp,&w,&b,&ci,&h,&ws,&co,&kh,&kw,&rep};
    hipModuleLaunchKernel(r->vae_conv2d, (unsigned)((n + 255) / 256), 1, 1,
                          256, 1, 1, 0, r->stream, args, NULL);
}

static void vae_op_gn(cuda_qimg_runner *r, void *out, void *inp, void *gamma, int C, int S) {
    void *args[] = {&out,&inp,&gamma,&C,&S};
    hipModuleLaunchKernel(r->vae_rmsnorm, (unsigned)((S + 255) / 256), 1, 1,
                          256, 1, 1, 0, r->stream, args, NULL);
}

static void vae_op_silu(cuda_qimg_runner *r, void *x, int n) {
    void *args[] = {&x,&n};
    hipModuleLaunchKernel(r->vae_silu, (unsigned)((n + 255) / 256), 1, 1,
                          256, 1, 1, 0, r->stream, args, NULL);
}

static void *vae_op_upsample(cuda_qimg_runner *r, void *inp, int C, int H, int W) {
    void *out = checked_cuMemAlloc((size_t)C * H * 2 * W * 2 * sizeof(float));
    if (!out) return NULL;
    int n = C * H * 2 * W * 2;
    void *args[] = {&out,&inp,&C,&H,&W};
    hipModuleLaunchKernel(r->vae_up2x, (unsigned)((n + 255) / 256), 1, 1,
                          256, 1, 1, 0, r->stream, args, NULL);
    return out;
}

static void *vae_resblock_gpu(cuda_qimg_runner *r, void *x,
                              void *n1, void *c1, void *b1,
                              void *n2, void *c2, void *b2,
                              void *scw, void *scb,
                              int ci, int co, int h, int w) {
    int sp = h * w;
    /* Norm1 writes ci channels; norm2 reuses this buffer for co channels. */
    void *tmp = checked_cuMemAlloc((size_t)(ci > co ? ci : co) * sp * sizeof(float));
    void *c1o = checked_cuMemAlloc((size_t)co * sp * sizeof(float));
    void *c2o = checked_cuMemAlloc((size_t)co * sp * sizeof(float));
    void *out = checked_cuMemAlloc((size_t)co * sp * sizeof(float));
    if (!tmp || !c1o || !c2o || !out) goto fail;
    vae_op_gn(r,tmp,x,n1,ci,sp); vae_op_silu(r,tmp,ci*sp);
    vae_op_conv2d(r,c1o,tmp,c1,b1,ci,h,w,co,3,3,0);
    vae_op_gn(r,tmp,c1o,n2,co,sp); vae_op_silu(r,tmp,co*sp);
    vae_op_conv2d(r,c2o,tmp,c2,b2,co,h,w,co,3,3,0);
    if (scw) vae_op_conv2d(r,out,x,scw,scb,ci,h,w,co,1,1,0);
    else hipMemcpyDtoD(out,x,(size_t)co*sp*sizeof(float));
    { float one=1.0f; int n=co*sp; void *a[]={&out,&c2o,&one,&n};
      hipModuleLaunchKernel(r->euler_step,(unsigned)((n+255)/256),1,1,256,1,1,0,r->stream,a,NULL); }
    vae_bf16(r,out,co*sp);
    hipFree(tmp); hipFree(c1o); hipFree(c2o); return out;
fail:
    if (tmp) hipFree(tmp);
    if (c1o) hipFree(c1o);
    if (c2o) hipFree(c2o);
    if (out) hipFree(out);
    return NULL;
}

static void *qimg_st_upload_f32(st_context *st, const char *name) {
    int idx = safetensors_find(st, name); if (idx < 0) return NULL;
    size_t n = 1; const uint64_t *shape = safetensors_shape(st, idx);
    for (int d = 0; d < safetensors_ndims(st, idx); d++) n *= shape[d];
    float *tmp = (float *)malloc(n * sizeof(float)); if (!tmp) return NULL;
    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    const char *dtype = safetensors_dtype(st, idx);
    if (!strcmp(dtype, "F32")) memcpy(tmp, src, n * sizeof(float));
    else if (!strcmp(dtype, "BF16")) for (size_t i=0;i<n;i++) { uint32_t b=(uint32_t)((const uint16_t *)src)[i]<<16; memcpy(tmp+i,&b,4); }
    else { free(tmp); return NULL; }
    void *d = checked_cuMemAlloc(n * sizeof(float));
    if (!d || hipMemcpy(d, tmp, n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) { if(d) hipFree(d); free(tmp); return NULL; }
    free(tmp); return d;
}

static void *qimg_st_upload_bf16(st_context *st, const char *name) {
    int idx = safetensors_find(st, name); if (idx < 0) return NULL;
    size_t n = 1; const uint64_t *shape = safetensors_shape(st, idx);
    for (int d = 0; d < safetensors_ndims(st, idx); d++) n *= shape[d];
    const char *dtype = safetensors_dtype(st, idx); const void *src = safetensors_data(st, idx);
    void *d = checked_cuMemAlloc(n * sizeof(uint16_t)); if (!d) return NULL;
    if (!strcmp(dtype, "BF16")) { if (hipMemcpy(d, src, n*2, hipMemcpyHostToDevice) != hipSuccess) { hipFree(d); return NULL; } return d; }
    hipFree(d); return NULL;
}

#endif
