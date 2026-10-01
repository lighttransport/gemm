/* SPDX-License-Identifier: MIT
 * Shared driver-shaped HIP boundary for portable C graphs. No CUDA libraries.
 * Device pointers retain integer byte arithmetic used by the original graphs.
 */
#ifndef RDNA4_CUDA_DRIVER_COMPAT_H
#define RDNA4_CUDA_DRIVER_COMPAT_H
#include "rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "hip_runner_common.h"
/* Override the legacy Qwen wrapper when included by its shared fast graph. */
#undef cuCtxSynchronize
#undef cuStreamSynchronize
#undef cuMemAlloc
#undef cuMemFree
#undef cuMemcpyHtoD
#undef cuMemcpyDtoH
#undef cuMemcpyDtoD
#undef cuMemcpyHtoDAsync
#undef cuMemcpyDtoHAsync

#ifndef HIP_QIMG21_RUNNER_H
    typedef int CUresult;
    typedef int CUdevice;
    typedef uintptr_t CUdeviceptr;
    typedef hipCtx_t CUcontext;
    typedef hipModule_t CUmodule;
    typedef hipFunction_t CUfunction;
    typedef hipStream_t CUstream;
#endif
    typedef hipEvent_t CUevent;
#define CU_FREE(p) do { if (p) cuMemFree(p); (p) = 0; } while (0)
#define CUDA_SUCCESS hipSuccess
#define CUEW_SUCCESS ROCEW_SUCCESS
#define CUEW_INIT_CUDA ROCEW_INIT_HIP
#define CUEW_INIT_NVRTC ROCEW_INIT_HIPRTC
#define CU_STREAM_NON_BLOCKING hipStreamNonBlocking
#define CU_EVENT_DISABLE_TIMING hipEventDisableTiming
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES hipFuncAttributeMaxDynamicSharedMemorySize
#define CU_COMPILE_ARCH_A 0
#define cuewInit rocewInit
#define cuInit hipInit
#define cuDeviceGetCount hipGetDeviceCount
static inline int cuDeviceGet(int *device, int ordinal) {
    int count;
    int rc = hipGetDeviceCount(&count);
    if (rc) return rc;
    if (ordinal < 0 || ordinal >= count) return hipErrorInvalidValue;
    *device = ordinal;
    return hipSuccess;
}
#define cuDeviceGetName hipDeviceGetName
static inline int cuDevicePrimaryCtxRetain(CUcontext *ctx, int device) { return hipCtxCreate(ctx, 0, device); }
#define cuCtxCreate hipCtxCreate
#define cuCtxDestroy hipCtxDestroy
#define cuCtxSetCurrent hipCtxSetCurrent
#define cuCtxSynchronize hipDeviceSynchronize
#define cuStreamCreate hipStreamCreateWithFlags
#define cuStreamDestroy hipStreamDestroy
#define cuStreamSynchronize hipStreamSynchronize
#define cuStreamWaitEvent hipStreamWaitEvent
#define cuEventCreate hipEventCreateWithFlags
#define cuEventDestroy hipEventDestroy
#define cuEventRecord hipEventRecord
#define cuEventElapsedTime hipEventElapsedTime
#define cuEventSynchronize hipEventSynchronize
#define cuModuleGetFunction hipModuleGetFunction
#define cuModuleUnload hipModuleUnload
#define cuModuleLoadData hipModuleLoadData
#define cuFuncSetAttribute hipFuncSetAttribute
#define cuLaunchKernel hipModuleLaunchKernel
#define cuMemAlloc(p,n) hipMalloc((void **)(p),(n))
#define cuMemFree(p) hipFree((void *)(uintptr_t)(p))
#define cuMemAllocHost(p,n) hipHostMalloc((p),(n),0)
#define cuMemHostAlloc hipHostMalloc
#define cuMemFreeHost hipHostFree
#define cuMemGetInfo hipMemGetInfo
#define cuMemcpyHtoD(d,s,n) hipMemcpy((void *)(uintptr_t)(d),(s),(n),hipMemcpyHostToDevice)
#define cuMemcpyDtoH(d,s,n) hipMemcpy((d),(void *)(uintptr_t)(s),(n),hipMemcpyDeviceToHost)
#define cuMemcpyDtoD(d,s,n) hipMemcpy((void *)(uintptr_t)(d),(void *)(uintptr_t)(s),(n),hipMemcpyDeviceToDevice)
#define cuMemcpyHtoDAsync(d,s,n,st) hipMemcpyAsync((void *)(uintptr_t)(d),(s),(n),hipMemcpyHostToDevice,(st))
#define cuMemcpyDtoHAsync(d,s,n,st) hipMemcpyAsync((d),(void *)(uintptr_t)(s),(n),hipMemcpyDeviceToHost,(st))
#define cuMemcpyDtoDAsync(d,s,n,st) hipMemcpyAsync((void *)(uintptr_t)(d),(void *)(uintptr_t)(s),(n),hipMemcpyDeviceToDevice,(st))
#define cuMemsetD8Async(d,v,n,st) hipMemsetAsync((void *)(uintptr_t)(d),(v),(n),(st))
/* D32 zero-fill is the only use in the shared fast graph. */
static inline int cuMemsetD32Async(CUdeviceptr d, unsigned value, size_t count, CUstream stream) {
    if (value != 0) return hipErrorInvalidValue;
    return hipMemsetAsync((void *)(uintptr_t)d, 0, count * 4, stream);
}
static inline CUdeviceptr cu_upload_raw(const void *data, size_t bytes) {
    return (CUdeviceptr)(uintptr_t)hip_upload_raw(data, bytes);
}
static int cu_compile_kernels_ex(CUmodule *module, int device, const char *source,
                                const char *name, int verbose, const char *prefix, int flags) {
    (void)flags;
    (void)hip_compile_kernels;
    (void)hip_f32_to_f16; (void)hip_f32_to_fp8_e4m3; (void)hip_fp8_e4m3_to_f32;
    static const char preamble[] =
        "#define __shfl_xor_sync(mask,x,lane) __shfl_xor(x,lane)\n"
        "#define __shfl_down_sync(mask,x,lane) __shfl_down(x,lane)\n"
        "#define __shfl_sync(mask,x,lane) __shfl(x,lane)\n"
        "#define __syncwarp(...) __builtin_amdgcn_wave_barrier()\n"
        "#define __frcp_rn(x) (1.0f/(x))\n";
    size_t n = sizeof(preamble) + strlen(source);
    char *combined = malloc(n);
    if (!combined) return -1;
    snprintf(combined, n, "%s%s", preamble, source);
    int rc = hip_compile_kernels_ex(module, device, combined, name, verbose, prefix, 1);
    free(combined);
    return rc;
}
static inline int cu_compile_kernels(CUmodule *module, int device, const char *source,
                             const char *name, int verbose, const char *prefix) {
    return cu_compile_kernels_ex(module, device, source, name, verbose, prefix, 0);
}
#endif
