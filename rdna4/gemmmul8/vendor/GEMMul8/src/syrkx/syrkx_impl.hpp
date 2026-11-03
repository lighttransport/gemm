#pragma once
#include "../../include/syrkx.hpp"
#include "../oz2/common/common.hpp"
#include "../oz2/syrkx/syrkx_launch.hpp"

namespace gemmul8 {

template <typename TA, Backend BACKEND, typename TB, typename TC>
std::vector<double> syrkx(
    cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    size_t n, size_t k,
    const TC *alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC *beta,
    TC *const C, size_t ldc,
    int num_moduli, bool fastmode,
    void *const work,
    void *const workA,
    void *const workB,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    bool skip_scalA,
    bool skip_scalB //
) {
    static_assert(common::isComplex<TA> == common::isComplex<TB> &&
                      common::isComplex<TB> == common::isComplex<TC>,
                  "TA, TB, and TC must be all real or all complex");

    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    if (num_moduli > 1) {
        return oz2::syrkx::syrkx_launch<TA, TB, TC, BACKEND>(
            common::Handle_t(common::CublasTag{}, handle),
            uplo, trans, n, k,
            alpha, A, lda, B, ldb, beta, C, ldc,
            num_moduli, fastmode, work, workA, workB,
            enable_skip_scalA, enable_skip_scalB,
            skip_scalA, skip_scalB,
            stream);
    } else if (num_moduli < 0) {
        assert(false && "not implemented yet");
    } else {
        assert(false && "unsupported");
    }

    return std::vector<double>(4, 0.0);
}

template <typename TA, Backend BACKEND, typename TB, typename TC>
std::vector<double> syrkxLt(
    cublasLtHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    size_t n, size_t k,
    const TC *alpha,
    const TA *const A, size_t lda,
    const TB *const B, size_t ldb,
    const TC *beta,
    TC *const C, size_t ldc,
    int num_moduli, bool fastmode,
    void *const work,
    void *const workA,
    void *const workB,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    bool skip_scalA,
    bool skip_scalB,
    cudaStream_t stream //
) {
    static_assert(common::isComplex<TA> == common::isComplex<TB> &&
                      common::isComplex<TB> == common::isComplex<TC>,
                  "TA, TB, and TC must be all real or all complex");

    if (num_moduli > 1) {
        return oz2::syrkx::syrkx_launch<TA, TB, TC, BACKEND>(
            common::Handle_t(common::CublasLtTag{}, handle),
            uplo, trans, n, k,
            alpha, A, lda, B, ldb, beta, C, ldc,
            num_moduli, fastmode, work, workA, workB,
            enable_skip_scalA, enable_skip_scalB,
            skip_scalA, skip_scalB,
            stream);
    } else if (num_moduli < 0) {
        assert(false && "not implemented yet");
    } else {
        assert(false && "unsupported");
    }

    return std::vector<double>(4, 0.0);
}

} // namespace gemmul8
