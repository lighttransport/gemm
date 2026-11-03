#pragma once
#include "config.hpp"
#include "calc_sft.hpp"
#include "scaling_colwise.hpp"

#include "../general/scaling_general_declaration.hpp"
#include "../general/helper_temp.hpp"

namespace gemmul8::scaling::accu {

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO_A, cublasDiagType_t DIAG_A, cublasFillMode_t UPLO_C>
void scaling(
    const cudaStream_t stream,
    const cublasOperation_t op_A, const cublasSideMode_t side,
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
            float *const partial_amax = general::temporary_memory<common::low_t<BACKEND>, float>(A_lo.ptr0);

            calc_sft_after_rowwise_with_delta_launch<T, BACKEND, NUM_MODULI, UPLO_C>(
                stream, m, n, k, C_hi, ldc_hi, sftA, fixed_delta, partial_amax);
        } else {
            using HiT               = common::hi_t<BACKEND>;
            HiT *const partial_amax = general::temporary_memory<common::low_t<BACKEND>, HiT>(A_lo.ptr0);

            calc_sft_after_rowwise_launch<T, BACKEND, NUM_MODULI, UPLO_C>(
                stream, m, n, k, C_hi, ldc_hi, sftA, partial_amax);
        }

        if constexpr (UPLO_A != CUBLAS_FILL_MODE_FULL) {
            general::memset_low_mats_async<T, BACKEND, NUM_MODULI>(stream, A_lo, incA_lo);
        }

        if (op_A == CUBLAS_OP_N) {

            // A: m x k -> A_lo: k x m
            general::scaling_rowwise<T, BACKEND, NUM_MODULI, UPLO_A, DIAG_A, false>(
                stream, m, k, A, lda, A_lo, lda_lo, incA_lo, sftA);

        } else {

            // A: k x m -> A_lo: k x m
            if (op_A == CUBLAS_OP_T) {

                scaling_colwise_launch<T, BACKEND, NUM_MODULI, UPLO_A, DIAG_A, false>(
                    stream, k, m, A, lda, A_lo, lda_lo >> 2, incA_lo >> 2, sftA);

            } else {

                constexpr bool CONJ = (common::isComplex<T>) ? true : false;
                scaling_colwise_launch<T, BACKEND, NUM_MODULI, UPLO_A, DIAG_A, CONJ>(
                    stream, k, m, A, lda, A_lo, lda_lo >> 2, incA_lo >> 2, sftA);
            }
        }

    } else {

        if (fixed_delta) {
            calc_sft_after_colwise_with_delta<T, BACKEND, NUM_MODULI, UPLO_C>
                <<<n, threads_accu, 0, stream>>>(
                    m, k, C_hi, ldc_hi, fixed_delta, sftA);
        } else {
            calc_sft_after_colwise<T, BACKEND, NUM_MODULI, UPLO_C>
                <<<n, threads_accu, 0, stream>>>(
                    m, k, C_hi, ldc_hi, sftA);
        }

        if constexpr (UPLO_A != CUBLAS_FILL_MODE_FULL) {
            general::memset_low_mats_async<T, BACKEND, NUM_MODULI>(stream, A_lo, incA_lo);
        }

        if (op_A == CUBLAS_OP_N) {

            // A: k x n -> A_lo: k x n
            scaling_colwise_launch<T, BACKEND, NUM_MODULI, UPLO_A, DIAG_A, false>(
                stream, k, n, A, lda, A_lo, lda_lo >> 2, incA_lo >> 2, sftA);

        } else {

            // A: n x k -> A_lo: k x n
            if (op_A == CUBLAS_OP_T) {

                general::scaling_rowwise<T, BACKEND, NUM_MODULI, UPLO_A, DIAG_A, false>(
                    stream, n, k, A, lda, A_lo, lda_lo, incA_lo, sftA);

            } else {

                constexpr bool CONJ = (common::isComplex<T>) ? true : false;
                general::scaling_rowwise<T, BACKEND, NUM_MODULI, UPLO_A, DIAG_A, CONJ>(
                    stream, n, k, A, lda, A_lo, lda_lo, incA_lo, sftA);
            }
        }
    }
}

} // namespace gemmul8::scaling::accu
