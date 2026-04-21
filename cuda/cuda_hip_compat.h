#ifndef CUDA_HIP_COMPAT_H_
#define CUDA_HIP_COMPAT_H_

#include "cuew.h"
#include "cuda_runner_common.h"

typedef CUmodule   hipModule_t;
typedef CUfunction hipFunction_t;
typedef CUresult   hipError_t;
typedef void      *hipStream_t;

#ifndef hipSuccess
#define hipSuccess CUDA_SUCCESS
#endif

#ifndef ROCEW_SUCCESS
#define ROCEW_SUCCESS 0
#endif
#ifndef ROCEW_INIT_HIP
#define ROCEW_INIT_HIP 1
#endif
#ifndef ROCEW_INIT_HIPRTC
#define ROCEW_INIT_HIPRTC 2
#endif

#ifndef hipMemcpyHostToHost
#define hipMemcpyHostToHost   0
#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2
#define hipMemcpyDeviceToDevice 3
#endif

static inline int rocewInit(int flags) {
    (void)flags;
    return (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) == CUEW_SUCCESS) ? ROCEW_SUCCESS : -1;
}

static inline CUresult hipSetDevice(int device_id) {
    static CUcontext g_ctx[16] = {0};
    static int g_inited = 0;
    if (!g_inited) {
        CUresult e = cuInit(0);
        if (e != CUDA_SUCCESS) return e;
        g_inited = 1;
    }
    if (device_id < 0 || device_id >= 16) return CUDA_ERROR_INVALID_VALUE;
    if (!g_ctx[device_id]) {
        CUdevice dev;
        CUresult e = cuDeviceGet(&dev, device_id);
        if (e != CUDA_SUCCESS) return e;
        e = cuCtxCreate(&g_ctx[device_id], 0, dev);
        if (e != CUDA_SUCCESS) return e;
    }
    return cuCtxSetCurrent(g_ctx[device_id]);
}

static inline CUresult hipGetErrorString(CUresult err, const char **s) {
    return cuGetErrorString(err, s);
}

static inline CUresult hipMalloc(void **ptr, size_t bytes) {
    return cuMemAlloc((CUdeviceptr *)ptr, bytes);
}

static inline CUresult hipFree(void *ptr) {
    return cuMemFree((CUdeviceptr)ptr);
}

static inline CUresult hipMemcpy(void *dst, const void *src, size_t bytes, int kind) {
    switch (kind) {
        case hipMemcpyHostToDevice:
            return cuMemcpyHtoD((CUdeviceptr)dst, src, bytes);
        case hipMemcpyDeviceToHost:
            return cuMemcpyDtoH(dst, (CUdeviceptr)src, bytes);
        case hipMemcpyDeviceToDevice:
            return cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, bytes);
        case hipMemcpyHostToHost:
            memcpy(dst, src, bytes);
            return CUDA_SUCCESS;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
}

static inline CUresult hipMemcpyAsync(void *dst, const void *src, size_t bytes, int kind, hipStream_t stream) {
    (void)stream;
    return hipMemcpy(dst, src, bytes, kind);
}

static inline CUresult hipMemset(void *dst, int value, size_t bytes) {
    return cuMemsetD8((CUdeviceptr)dst, (unsigned char)value, bytes);
}

static inline CUresult hipDeviceSynchronize(void) {
    return cuCtxSynchronize();
}

static inline CUresult hipModuleLoadData(hipModule_t *mod, const void *image) {
    return cuModuleLoadData(mod, image);
}

static inline CUresult hipModuleUnload(hipModule_t mod) {
    return cuModuleUnload(mod);
}

static inline CUresult hipModuleGetFunction(hipFunction_t *fn, hipModule_t mod, const char *name) {
    return cuModuleGetFunction(fn, mod, name);
}

static inline CUresult hipModuleLaunchKernel(hipFunction_t f,
                                             unsigned gx, unsigned gy, unsigned gz,
                                             unsigned bx, unsigned by, unsigned bz,
                                             unsigned sharedMemBytes, hipStream_t stream,
                                             void **kernelParams, void **extra) {
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedMemBytes, (CUstream)stream,
                          kernelParams, extra);
}

static inline uint16_t hip_f32_to_f16(float f) {
    return cu_f32_to_f16(f);
}

static inline void *hip_upload_raw(const void *data, size_t bytes) {
    return (void *)cu_upload_raw(data, bytes);
}

static inline int hip_compile_kernels(hipModule_t *module, int device_id,
                                      const char *source, const char *prog_name,
                                      int verbose, const char *prefix) {
    CUdevice dev;
    if (cuDeviceGet(&dev, device_id) != CUDA_SUCCESS) return -1;
    return cu_compile_kernels(module, dev, source, prog_name, verbose, prefix);
}

#define hip_kernels_common_src cuda_kernels_common_src

#endif
