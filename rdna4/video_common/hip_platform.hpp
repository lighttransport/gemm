// SPDX-License-Identifier: MIT
// Driver-shaped HIP boundary used by the shared native video graphs.
#pragma once
#include "../rocew.h"
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <initializer_list>
using CUresult = hipError_t;
using CUdeviceptr = uintptr_t;
using CUcontext = hipCtx_t;
using CUstream = hipStream_t;
using CUevent = hipEvent_t;
using CUmodule = hipModule_t;
using CUfunction = hipFunction_t;
constexpr auto CUDA_SUCCESS = hipSuccess;
constexpr auto CU_STREAM_NON_BLOCKING = hipStreamNonBlocking;
constexpr auto CU_EVENT_DISABLE_TIMING = hipEventDisableTiming;
#define cuCtxSetCurrent hipCtxSetCurrent
#define cuCtxDestroy hipCtxDestroy
#define cuStreamCreate hipStreamCreateWithFlags
#define cuStreamDestroy hipStreamDestroy
#define cuStreamSynchronize hipStreamSynchronize
#define cuStreamWaitEvent hipStreamWaitEvent
#define cuEventCreate hipEventCreateWithFlags
#define cuEventQuery hipEventQuery
#define cuEventDestroy hipEventDestroy
#define cuEventRecord hipEventRecord
#define cuEventSynchronize hipEventSynchronize
#define cuModuleGetFunction hipModuleGetFunction
#define cuModuleUnload hipModuleUnload
#define cuModuleLoadData hipModuleLoadData
#define cuLaunchKernel hipModuleLaunchKernel
#define cuMemAlloc(p, n) hipMalloc(reinterpret_cast<void **>(p), n)
#define cuMemFree(p) hipFree(reinterpret_cast<void *>(p))
#define cuMemHostAlloc hipHostMalloc
#define cuMemFreeHost hipHostFree
#define cuMemGetInfo hipMemGetInfo
#define cuMemcpyHtoDAsync(d, s, n, t)                                                              \
    hipMemcpyAsync(reinterpret_cast<void *>(d), s, n, hipMemcpyHostToDevice, t)
#define cuMemcpyDtoH(d, s, n) hipMemcpy(d, reinterpret_cast<void *>(s), n, hipMemcpyDeviceToHost)
#define cuMemcpyDtoHAsync(d, s, n, t)                                                              \
    hipMemcpyAsync(d, reinterpret_cast<void *>(s), n, hipMemcpyDeviceToHost, t)
#define cuMemcpyDtoDAsync(d, s, n, t)                                                              \
    hipMemcpyAsync(reinterpret_cast<void *>(d), reinterpret_cast<void *>(s), n,                    \
                   hipMemcpyDeviceToDevice, t)
inline CUresult cuMemsetD32Async(CUdeviceptr p, unsigned value, size_t count, CUstream stream) {
    if (value)
        return hipErrorInvalidValue;
    return hipMemsetAsync(reinterpret_cast<void *>(p), 0, count * 4, stream);
}
inline CUresult cuGetErrorString(CUresult e, const char **s) { return hipGetErrorString(e, s); }
// Optional hipBLAS comparison backend, loaded only when explicitly requested.
struct cublasew_context {
    void *library = nullptr, *handle = nullptr;
    int (*destroy)(void *) = nullptr;
    int (*gemm)(void *, int, int, int, int, int, const void *, const void *, int, int, const void *,
                int, int, const void *, void *, int, int, int, int) = nullptr;
};
inline int cublasewCreate(cublasew_context **out, CUstream stream) {
    auto c = new cublasew_context;
    c->library = dlopen("libhipblas.so", RTLD_NOW | RTLD_LOCAL);
    if (!c->library)
        for (auto path : {"/opt/rocm/core-10.0/lib/libhipblas.so", "/opt/rocm/lib/libhipblas.so",
                          "/opt/rocm/core-7.14/lib/libhipblas.so"}) {
            c->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (c->library)
                break;
        }
    if (!c->library) {
        delete c;
        return -1;
    }
    auto create = reinterpret_cast<int (*)(void **)>(dlsym(c->library, "hipblasCreate"));
    auto set_stream =
        reinterpret_cast<int (*)(void *, CUstream)>(dlsym(c->library, "hipblasSetStream"));
    c->destroy = reinterpret_cast<int (*)(void *)>(dlsym(c->library, "hipblasDestroy"));
    c->gemm = reinterpret_cast<decltype(c->gemm)>(dlsym(c->library, "hipblasGemmEx"));
    if (!create || !set_stream || !c->destroy || !c->gemm || create(&c->handle) ||
        set_stream(c->handle, stream)) {
        if (c->handle && c->destroy)
            c->destroy(c->handle);
        dlclose(c->library);
        delete c;
        return -1;
    }
    *out = c;
    return 0;
}
inline void cublasewDestroy(cublasew_context *c) {
    if (c) {
        c->destroy(c->handle);
        dlclose(c->library);
        delete c;
    }
}
inline int video_hipblas_gemm(cublasew_context *c, CUdeviceptr y, CUdeviceptr w, CUdeviceptr x,
                              int m, int n, int k, int type) {
    float alpha = 1, beta = 0;
    return c->gemm(c->handle, 112, 111, n, m, k, &alpha, reinterpret_cast<void *>(w), type, k,
                   reinterpret_cast<void *>(x), type, k, &beta, reinterpret_cast<void *>(y), 0, n,
                   2, 160); // HIPBLAS_COMPUTE_32F; ROCm does not support 32F_PEDANTIC.
}
inline int cublasew_gemm_f32_pedantic_rowmajor_nt(cublasew_context *c, CUdeviceptr y, CUdeviceptr w,
                                                  CUdeviceptr x, int m, int n, int k) {
    return video_hipblas_gemm(c, y, w, x, m, n, k, 0);
}
inline int cublasew_gemm_f16_f16_f32_rowmajor_nt(cublasew_context *c, CUdeviceptr y, CUdeviceptr w,
                                                 CUdeviceptr x, int m, int n, int k) {
    return video_hipblas_gemm(c, y, w, x, m, n, k, 2);
}
