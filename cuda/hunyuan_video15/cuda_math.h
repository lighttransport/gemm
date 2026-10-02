#ifndef HV15_CUDA_MATH_H
#define HV15_CUDA_MATH_H
#include "stable-diffusion.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#ifdef __cplusplus
extern "C" {
#endif
SD_API bool hv15_cuda_f32_begin(ggml_backend_t backend);
SD_API bool hv15_cuda_f32_end(ggml_backend_t backend);
#ifdef __cplusplus
}
/* A backend is used by one serialized inference operation at a time. */
struct HV15CudaF32Scope {
    ggml_backend_t backend;
    HV15CudaF32Scope(const HV15CudaF32Scope&) = delete;
    HV15CudaF32Scope& operator=(const HV15CudaF32Scope&) = delete;
    explicit HV15CudaF32Scope(ggml_backend_t value, bool enabled) : backend(enabled ? value : nullptr) {
        if (backend) GGML_ASSERT(hv15_cuda_f32_begin(backend));
    }
    ~HV15CudaF32Scope() {
        if (backend) GGML_ASSERT(hv15_cuda_f32_end(backend));
    }
};
#endif
#endif
