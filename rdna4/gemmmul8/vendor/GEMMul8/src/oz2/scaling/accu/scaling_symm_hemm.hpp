#pragma once
#include "config.hpp"
#include "calc_sft.hpp"

#include "../general/scaling_general_declaration.hpp"
#include "../general/helper_temp.hpp"

namespace gemmul8::scaling::accu {

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO_A, bool HERM,
          bool STORE_TRANSPOSE = false>
inline void scaling_symm_hemm_launch(
    const cudaStream_t stream,
    const cublasSideMode_t side,
    const unsigned m, const unsigned n, const unsigned k,
    const T *const A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> &A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const sftA,
    common::matptr_t<common::hi_t<BACKEND>, common::isComplex<T>> &C_hi,
    const size_t ldc_hi,
    const int16_t *const fixed_delta //
) {
    if (side == CUBLAS_SIDE_LEFT) {

        if (fixed_delta) {
            float *const amaxC = general::temporary_memory<common::low_t<BACKEND>, float>(A_lo.ptr0);
            calc_sft_after_rowwise_with_delta_launch<T, BACKEND, NUM_MODULI, CUBLAS_FILL_MODE_FULL>(
                stream, m, n, k, C_hi, ldc_hi, sftA, fixed_delta, amaxC);
        } else {
            using HiT        = common::hi_t<BACKEND>;
            HiT *const amaxC = general::temporary_memory<common::low_t<BACKEND>, HiT>(A_lo.ptr0);

            calc_sft_after_rowwise_launch<T, BACKEND, NUM_MODULI, CUBLAS_FILL_MODE_FULL>(
                stream, m, n, k, C_hi, ldc_hi, sftA, amaxC);
        }

    } else {

        if (fixed_delta) {
            calc_sft_after_colwise_with_delta<T, BACKEND, NUM_MODULI, CUBLAS_FILL_MODE_FULL>
                <<<n, threads_accu, 0, stream>>>(
                    m, k, C_hi, ldc_hi, fixed_delta, sftA);
        } else {
            calc_sft_after_colwise<T, BACKEND, NUM_MODULI, CUBLAS_FILL_MODE_FULL>
                <<<n, threads_accu, 0, stream>>>(
                    m, k, C_hi, ldc_hi, sftA);
        }
    }

    const unsigned nA = (side == CUBLAS_SIDE_LEFT) ? m : n;
    if constexpr (HERM) {

        general::scaling_hemm<T, BACKEND, NUM_MODULI, UPLO_A, STORE_TRANSPOSE>(
            stream, nA, A, lda, A_lo, lda_lo, incA_lo, sftA);

    } else {

        general::scaling_symm<T, BACKEND, NUM_MODULI, UPLO_A>(
            stream, nA, A, lda, A_lo, lda_lo, incA_lo, sftA);
    }
}

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO_A>
void scaling_symm(
    const cudaStream_t stream,
    const cublasSideMode_t side,
    const unsigned m, const unsigned n, const unsigned k,
    const T *const A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> &A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const sftA,
    common::matptr_t<common::hi_t<BACKEND>, common::isComplex<T>> &C_hi,
    const size_t ldc_hi,
    const int16_t *const fixed_delta //
) {
    static_assert(UPLO_A == CUBLAS_FILL_MODE_UPPER || UPLO_A == CUBLAS_FILL_MODE_LOWER,
                  "scaling_symm requires UPLO = UPPER or LOWER.");

    scaling_symm_hemm_launch<T, BACKEND, NUM_MODULI, UPLO_A, false>(
        stream, side, m, n, k, A, lda, A_lo, lda_lo, incA_lo,
        sftA, C_hi, ldc_hi, fixed_delta);
}

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO_A,
          bool STORE_TRANSPOSE>
void scaling_hemm(
    const cudaStream_t stream,
    const cublasSideMode_t side,
    const unsigned m, const unsigned n, const unsigned k,
    const T *const A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> &A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const sftA,
    common::matptr_t<common::hi_t<BACKEND>, common::isComplex<T>> &C_hi,
    const size_t ldc_hi,
    const int16_t *const fixed_delta //
) {
    static_assert(UPLO_A == CUBLAS_FILL_MODE_UPPER || UPLO_A == CUBLAS_FILL_MODE_LOWER,
                  "scaling_hemm requires UPLO = UPPER or LOWER.");

    scaling_symm_hemm_launch<T, BACKEND, NUM_MODULI, UPLO_A, true, STORE_TRANSPOSE>(
        stream, side, m, n, k, A, lda, A_lo, lda_lo, incA_lo,
        sftA, C_hi, ldc_hi, fixed_delta);
}

} // namespace gemmul8::scaling::accu
