#ifndef CUDA_LINEAR_F32_H
#define CUDA_LINEAR_F32_H
/* Shared host-facing FP32 linear layer using the repository CUDA GEMM.
 * Hybrid vision runners keep attention/image ops on the host. No vendor BLAS. */
#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_gemm_f32_kernels.h"

static struct {
    CUcontext context;
    CUmodule module;
    CUfunction gemm;
    cu_buf_slot x,w,b,y;
} cuda_linear_f32;

static void cuda_linear_f32_free(void)
{
    if (!cuda_linear_f32.context) return;
    cuCtxSetCurrent(cuda_linear_f32.context);
    cu_buf_slot_free(&cuda_linear_f32.x); cu_buf_slot_free(&cuda_linear_f32.w);
    cu_buf_slot_free(&cuda_linear_f32.b); cu_buf_slot_free(&cuda_linear_f32.y);
    if (cuda_linear_f32.module) cuModuleUnload(cuda_linear_f32.module);
    cuCtxDestroy(cuda_linear_f32.context);
    memset(&cuda_linear_f32,0,sizeof(cuda_linear_f32));
}

static int cuda_linear_f32_init(int index)
{
    CUdevice device;
    if (cuewInit(CUEW_INIT_CUDA|CUEW_INIT_NVRTC)!=CUEW_SUCCESS) return -1;
    CU_CHECK(cuInit(0)); CU_CHECK(cuDeviceGet(&device,index));
    CU_CHECK(cuCtxCreate(&cuda_linear_f32.context,0,device));
    const char *src="extern \"C\" {\n" CUDA_GEMM_F32_BIAS_SRC "}\n";
    /* The compiler returns the target SM on success, not zero. */
    if (cu_compile_kernels_ex(&cuda_linear_f32.module,device,src,"linear_f32.cu",0,"linear_f32",0)<0) return -1;
    CU_CHECK(cuModuleGetFunction(&cuda_linear_f32.gemm,cuda_linear_f32.module,"gemm_f32_bias"));
    return 0;
}

static int cuda_linear_f32_run(float *out,const float *w,const float *b,const float *x,int m,int n,int k)
{
    size_t xb=(size_t)m*k*sizeof(float),wb=(size_t)n*k*sizeof(float),yb=(size_t)m*n*sizeof(float);
    if (cu_buf_slot_ensure(&cuda_linear_f32.x,xb,"input") || cu_buf_slot_ensure(&cuda_linear_f32.w,wb,"weight") ||
        cu_buf_slot_ensure(&cuda_linear_f32.y,yb,"output") || cu_buf_slot_ensure(&cuda_linear_f32.b,n*sizeof(float),"bias")) return -1;
    CU_CHECK(cuMemcpyHtoD(cuda_linear_f32.x.ptr,x,xb)); CU_CHECK(cuMemcpyHtoD(cuda_linear_f32.w.ptr,w,wb));
    CUdeviceptr bias=b ? cuda_linear_f32.b.ptr : 0;
    if (b) CU_CHECK(cuMemcpyHtoD(bias,b,n*sizeof(float)));
    void *args[]={&cuda_linear_f32.y.ptr,&cuda_linear_f32.x.ptr,&cuda_linear_f32.w.ptr,&bias,&m,&k,&n};
    CU_CHECK(cuLaunchKernel(cuda_linear_f32.gemm,(m+15)/16,(n+15)/16,1,16,16,1,0,NULL,args,NULL));
    CU_CHECK(cuMemcpyDtoH(out,cuda_linear_f32.y.ptr,yb));
    return 0;
}

#endif
