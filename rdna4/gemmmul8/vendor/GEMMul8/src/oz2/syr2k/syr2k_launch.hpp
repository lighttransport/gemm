#pragma once
#include "syr2k_core.hpp"

namespace gemmul8::oz2::syr2k {

template <typename TA, typename TB, typename TC, Backend BACKEND>
inline std::vector<double> syr2k_launch(
    common::Handle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    size_t n, size_t k,
    const TC *alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC *beta,
    TC *const C, size_t ldc,
    int num_moduli, bool fastmode,
    void *const work, void *const workA, void *const workB,
    bool enable_skip_scalA, bool enable_skip_scalB,
    bool skip_scalA, bool skip_scalB,
    cudaStream_t stream //
) {
    switch (num_moduli) {
    case 2: return syr2k_core<TA, TB, TC, BACKEND, 2U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 3: return syr2k_core<TA, TB, TC, BACKEND, 3U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 4: return syr2k_core<TA, TB, TC, BACKEND, 4U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 5: return syr2k_core<TA, TB, TC, BACKEND, 5U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 6: return syr2k_core<TA, TB, TC, BACKEND, 6U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 7: return syr2k_core<TA, TB, TC, BACKEND, 7U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 8: return syr2k_core<TA, TB, TC, BACKEND, 8U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 9: return syr2k_core<TA, TB, TC, BACKEND, 9U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 10: return syr2k_core<TA, TB, TC, BACKEND, 10U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 11: return syr2k_core<TA, TB, TC, BACKEND, 11U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 12: return syr2k_core<TA, TB, TC, BACKEND, 12U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 13: return syr2k_core<TA, TB, TC, BACKEND, 13U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 14: return syr2k_core<TA, TB, TC, BACKEND, 14U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 15: return syr2k_core<TA, TB, TC, BACKEND, 15U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 16: return syr2k_core<TA, TB, TC, BACKEND, 16U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 17: return syr2k_core<TA, TB, TC, BACKEND, 17U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 18: return syr2k_core<TA, TB, TC, BACKEND, 18U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 19: return syr2k_core<TA, TB, TC, BACKEND, 19U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    case 20: return syr2k_core<TA, TB, TC, BACKEND, 20U>(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, fastmode, work, workA, workB, enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB, stream);
    default: assert(false && "unsupported"); return std::vector<double>(4, 0.0);
    }
}

} // namespace gemmul8::oz2::syr2k
