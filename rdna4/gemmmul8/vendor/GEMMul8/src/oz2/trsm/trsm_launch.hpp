#pragma once
#include "trsm_core.hpp"

namespace gemmul8::oz2::trsm {

template <typename TA, typename TB, Backend BACKEND>
inline std::vector<double> trsm_launch(
    common::Handle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    size_t m, size_t n,
    const TB *alpha,
    const TA *const A, size_t lda,
    TB *const B, size_t ldb,
    int num_moduli, bool fastmode,
    void *const work,
    cudaStream_t stream //
) {
    switch (num_moduli) {
    case 2: return trsm_core<TA, TB, BACKEND, 2U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 3: return trsm_core<TA, TB, BACKEND, 3U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 4: return trsm_core<TA, TB, BACKEND, 4U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 5: return trsm_core<TA, TB, BACKEND, 5U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 6: return trsm_core<TA, TB, BACKEND, 6U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 7: return trsm_core<TA, TB, BACKEND, 7U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 8: return trsm_core<TA, TB, BACKEND, 8U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 9: return trsm_core<TA, TB, BACKEND, 9U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 10: return trsm_core<TA, TB, BACKEND, 10U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 11: return trsm_core<TA, TB, BACKEND, 11U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 12: return trsm_core<TA, TB, BACKEND, 12U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 13: return trsm_core<TA, TB, BACKEND, 13U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 14: return trsm_core<TA, TB, BACKEND, 14U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 15: return trsm_core<TA, TB, BACKEND, 15U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 16: return trsm_core<TA, TB, BACKEND, 16U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 17: return trsm_core<TA, TB, BACKEND, 17U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 18: return trsm_core<TA, TB, BACKEND, 18U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 19: return trsm_core<TA, TB, BACKEND, 19U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    case 20: return trsm_core<TA, TB, BACKEND, 20U>(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, fastmode, work, stream);
    default: assert(false && "unsupported"); return std::vector<double>(4, 0.0);
    }
}

} // namespace gemmul8::oz2::trsm
