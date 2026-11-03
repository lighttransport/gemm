#pragma once
#include "../../common/common.hpp"

namespace gemmul8::scaling::accu {

template <typename T, Backend BACKEND,
          cublasFillMode_t UPLO = CUBLAS_FILL_MODE_FULL,
          cublasDiagType_t DIAG = CUBLAS_DIAG_NON_UNIT>
void extract(
    const cudaStream_t stream,
    const cublasOperation_t op_A, const cublasSideMode_t side,
    const unsigned rows_A, const unsigned cols_A,
    const T *const A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> &A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const sftA);

template <typename T, Backend BACKEND, cublasFillMode_t UPLO>
void extract_symm(
    const cudaStream_t stream,
    const unsigned n,
    const T *const A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> &A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const sftA);

template <typename T, Backend BACKEND, cublasFillMode_t UPLO,
          bool STORE_TRANSPOSE = false>
void extract_hemm(
    const cudaStream_t stream,
    const unsigned n,
    const T *const A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> &A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const sftA);

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO_A = CUBLAS_FILL_MODE_FULL,
          cublasDiagType_t DIAG_A = CUBLAS_DIAG_NON_UNIT,
          cublasFillMode_t UPLO_C = CUBLAS_FILL_MODE_FULL>
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
    const int16_t *const fixed_delta = nullptr);

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO_C>
void scaling_syrk(
    const cudaStream_t stream,
    const cublasOperation_t op_A,
    const unsigned n, const unsigned k,
    const T *const A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> &A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const sftA,
    common::matptr_t<common::hi_t<BACKEND>, common::isComplex<T>> &C_hi,
    const size_t ldc_hi);

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO_C>
void scaling_herk(
    const cudaStream_t stream,
    const cublasOperation_t op_A,
    const unsigned n, const unsigned k,
    const T *const A, const size_t lda,
    common::matptr_t<common::low_t<BACKEND>, common::isComplex<T>> &A_lo,
    const size_t lda_lo, const size_t incA_lo,
    int16_t *const sftA,
    common::matptr_t<common::hi_t<BACKEND>, common::isComplex<T>> &C_hi,
    const size_t ldc_hi);

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
    const int16_t *const fixed_delta = nullptr);

template <typename T, Backend BACKEND, unsigned NUM_MODULI,
          cublasFillMode_t UPLO_A,
          bool STORE_TRANSPOSE = false>
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
    const int16_t *const fixed_delta = nullptr);

} // namespace gemmul8::scaling::accu
