#pragma once
#include "herk_core.hpp"

namespace gemmul8::oz2::herk {

template <typename TA, typename TC, Backend BACKEND>
inline std::vector<double> herk_launch(
    common::Handle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    size_t n, size_t k,
    const common::underlying_t<TC> *alpha,
    const TA *const A, size_t lda,
    const common::underlying_t<TC> *beta,
    TC *const C, size_t ldc,
    int num_moduli, bool fastmode,
    void *const work, void *const workA,
    bool enable_skip_scalA,
    bool skip_scalA,
    cudaStream_t stream //
) {
    switch (num_moduli) {
    case 2: return herk_core<TA, TC, BACKEND, 2U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 3: return herk_core<TA, TC, BACKEND, 3U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 4: return herk_core<TA, TC, BACKEND, 4U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 5: return herk_core<TA, TC, BACKEND, 5U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 6: return herk_core<TA, TC, BACKEND, 6U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 7: return herk_core<TA, TC, BACKEND, 7U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 8: return herk_core<TA, TC, BACKEND, 8U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 9: return herk_core<TA, TC, BACKEND, 9U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 10: return herk_core<TA, TC, BACKEND, 10U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 11: return herk_core<TA, TC, BACKEND, 11U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 12: return herk_core<TA, TC, BACKEND, 12U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 13: return herk_core<TA, TC, BACKEND, 13U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 14: return herk_core<TA, TC, BACKEND, 14U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 15: return herk_core<TA, TC, BACKEND, 15U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 16: return herk_core<TA, TC, BACKEND, 16U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 17: return herk_core<TA, TC, BACKEND, 17U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 18: return herk_core<TA, TC, BACKEND, 18U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 19: return herk_core<TA, TC, BACKEND, 19U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    case 20: return herk_core<TA, TC, BACKEND, 20U>(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, fastmode, work, workA, enable_skip_scalA, skip_scalA, stream);
    default: assert(false && "unsupported"); return std::vector<double>(4, 0.0);
    }
}

} // namespace gemmul8::oz2::herk
