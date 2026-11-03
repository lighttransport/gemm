#pragma once
#include "../../include/syrk.hpp"
#include "../oz2/common/common.hpp"
#include "../oz2/syrk/syrk_launch.hpp"

namespace gemmul8 {

template <typename TA, Backend BACKEND, typename TC>
std::vector<double> syrk(
    cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    size_t n, size_t k,
    const TC *alpha, const TA *const A, size_t lda,
    const TC *beta, TC *const C, size_t ldc,
    int num_moduli, bool fastmode,
    void *const work,
    void *const workA,
    bool enable_skip_scalA,
    bool skip_scalA //
) {
    static_assert(common::isComplex<TA> == common::isComplex<TC>,
                  "TA and TC must be all real or all complex");

    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    if (num_moduli > 1) {
        return oz2::syrk::syrk_launch<TA, TC, BACKEND>(
            common::Handle_t(common::CublasTag{}, handle),
            uplo, trans, n, k,
            alpha, A, lda, beta, C, ldc,
            num_moduli, fastmode, work, workA,
            enable_skip_scalA, skip_scalA,
            stream);
    } else if (num_moduli < 0) {
        assert(false && "not implemented yet");
    } else {
        assert(false && "unsupported");
    }

    return std::vector<double>(4, 0.0);
}

template <typename TA, Backend BACKEND, typename TC>
std::vector<double> syrkLt(
    cublasLtHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    size_t n, size_t k,
    const TC *alpha, const TA *const A, size_t lda,
    const TC *beta, TC *const C, size_t ldc,
    int num_moduli, bool fastmode,
    void *const work,
    void *const workA,
    bool enable_skip_scalA,
    bool skip_scalA,
    cudaStream_t stream //
) {
    static_assert(common::isComplex<TA> == common::isComplex<TC>,
                  "TA and TC must be all real or all complex");

    if (num_moduli > 1) {
        return oz2::syrk::syrk_launch<TA, TC, BACKEND>(
            common::Handle_t(common::CublasLtTag{}, handle),
            uplo, trans, n, k,
            alpha, A, lda, beta, C, ldc,
            num_moduli, fastmode, work, workA,
            enable_skip_scalA, skip_scalA,
            stream);
    } else if (num_moduli < 0) {
        assert(false && "not implemented yet");
    } else {
        assert(false && "unsupported");
    }

    return std::vector<double>(4, 0.0);
}

} // namespace gemmul8
