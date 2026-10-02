/* Scoped IEEE FP32 cuBLAS math for the pinned native CUDA backend.
 * F32 accumulation alone does not disable its default TF32 input rounding.
 * Compile with the backend's definitions/includes to match its private ABI. */
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "cuda_math.h"
#include <array>
#include <mutex>
#include <unordered_map>

namespace {
std::mutex scopes_mutex;
using Modes = std::array<cublasMath_t, GGML_CUDA_MAX_STREAMS>;
std::unordered_map<ggml_backend_t, Modes> scopes;
}

extern "C" SD_API bool hv15_cuda_f32_begin(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    ggml_backend_synchronize(backend);
    std::lock_guard<std::mutex> lock(scopes_mutex);
    if (scopes.count(backend)) return false;
    auto ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    const int previous_stream = ctx->curr_stream_no;
    Modes modes;
    int completed = 0;
    for (; completed < GGML_CUDA_MAX_STREAMS; ++completed) {
        ctx->curr_stream_no = completed;
        auto handle = ctx->cublas_handle();
        if (cublasGetMathMode(handle, &modes[completed]) != CUBLAS_STATUS_SUCCESS ||
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) break;
    }
    ctx->curr_stream_no = previous_stream;
    if (completed != GGML_CUDA_MAX_STREAMS) {
        for (int i=0; i<completed; ++i)
            cublasSetMathMode(ctx->cublas_handles[ctx->device][i], modes[i]);
        return false;
    }
    scopes.emplace(backend, modes);
    return true;
}

extern "C" SD_API bool hv15_cuda_f32_end(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    ggml_backend_synchronize(backend);
    std::lock_guard<std::mutex> lock(scopes_mutex);
    auto found = scopes.find(backend);
    if (found == scopes.end()) return false;
    auto ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    bool success = true;
    for (int i=0; i<GGML_CUDA_MAX_STREAMS; ++i)
        success = cublasSetMathMode(ctx->cublas_handles[ctx->device][i], found->second[i]) == CUBLAS_STATUS_SUCCESS && success;
    scopes.erase(found);
    return success;
}
