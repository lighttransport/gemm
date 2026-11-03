#pragma once
#include "../../include/trsm.hpp"
#include "../oz2/common/common.hpp"
#include "../oz2/trsm/trsm_launch.hpp"

namespace gemmul8 {

template <typename TA, Backend BACKEND, typename TB>
std::vector<double> trsm(
    cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    size_t m, size_t n,
    const TB *alpha,
    const TA *const A, size_t lda,
    TB *const B, size_t ldb,
    int num_moduli, bool fastmode,
    void *const work //
) {
    static_assert(std::is_same_v<TA, TB>, "TA and TB must be same.");

    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    if (num_moduli > 1) {
        return oz2::trsm::trsm_launch<TA, TB, BACKEND>(
            common::Handle_t(common::CublasTag{}, handle),
            side, uplo, trans, diag, m, n,
            alpha, A, lda, B, ldb,
            num_moduli, fastmode, work, stream);
    } else if (num_moduli < 0) {
        assert(false && "not implemented yet");
    } else {
        assert(false && "unsupported");
    }

    return std::vector<double>(4, 0.0);
}

template <typename TA, Backend BACKEND, typename TB>
std::vector<double> trsmLt(
    cublasLtHandle_t handle,
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
    static_assert(std::is_same_v<TA, TB>, "TA and TB must be same.");

    if (num_moduli > 1) {
        return oz2::trsm::trsm_launch<TA, TB, BACKEND>(
            common::Handle_t(common::CublasLtTag{}, handle),
            side, uplo, trans, diag, m, n,
            alpha, A, lda, B, ldb,
            num_moduli, fastmode, work, stream);
    } else if (num_moduli < 0) {
        assert(false && "not implemented yet");
    } else {
        assert(false && "unsupported");
    }

    return std::vector<double>(4, 0.0);
}

} // namespace gemmul8
