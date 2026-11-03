#pragma once
#include "../common/common.hpp"

namespace gemmul8::oz2::trsm {

template <typename T> inline cublasStatus_t small_trsm(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const T *alpha, const T *A, int lda, T *B, int ldb);

template <> inline cublasStatus_t small_trsm<float>(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb //
) { return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

template <> inline cublasStatus_t small_trsm<double>(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb //
) { return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

template <> inline cublasStatus_t small_trsm<cuFloatComplex>(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const cuFloatComplex *alpha, const cuFloatComplex *A, int lda, cuFloatComplex *B, int ldb //
) { return cublasCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

template <> inline cublasStatus_t small_trsm<cuDoubleComplex>(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb //
) { return cublasZtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

} // namespace gemmul8::oz2::trsm
