#pragma once
#include "../../common/common.hpp"

namespace gemmul8::scaling::general {

template <bool UPPER>
__device__ __forceinline__ bool tri_elem_active(
    const unsigned row,
    const unsigned col //
) {
    if constexpr (UPPER) {
        return row <= col;
    } else {
        return row >= col;
    }
}

template <bool UPPER>
__device__ __forceinline__ bool tri_tile_zero(
    const unsigned rowBase,
    const unsigned colBase //
) {
    if constexpr (UPPER) {
        return rowBase >= colBase + common::TILE_DIM;
    } else {
        return rowBase + common::TILE_DIM <= colBase;
    }
}

template <bool UPPER>
__device__ __forceinline__ bool tri_tile_full_active(
    const unsigned rowBase,
    const unsigned colBase //
) {
    if constexpr (UPPER) {
        return rowBase + common::TILE_DIM <= colBase;
    } else {
        return rowBase >= colBase + common::TILE_DIM;
    }
}

template <bool UPPER, typename T, cublasDiagType_t DIAG>
__device__ __forceinline__ T tri_col_value(
    const T *const __restrict__ in,
    const unsigned row, const unsigned col,
    const unsigned rows //
) {
    if (row >= rows) return common::Tconst<T>::zero();

    if constexpr (UPPER) {
        if (row > col) return common::Tconst<T>::zero();
    } else {
        if (row < col) return common::Tconst<T>::zero();
    }

    if constexpr (DIAG == CUBLAS_DIAG_UNIT) {
        if (row == col) return common::Tconst<T>::one();
    }

    return in[row];
}

template <bool UPPER, typename T, cublasDiagType_t DIAG>
__device__ __forceinline__ T tri_mat_value(
    const T *const __restrict__ A,
    const size_t lda,
    const unsigned row, const unsigned col,
    const unsigned rows, const unsigned cols //
) {
    if (row >= rows || col >= cols) return common::Tconst<T>::zero();

    if constexpr (UPPER) {
        if (row > col) return common::Tconst<T>::zero();
    } else {
        if (row < col) return common::Tconst<T>::zero();
    }

    if constexpr (DIAG == CUBLAS_DIAG_UNIT) {
        if (row == col) return common::Tconst<T>::one();
    }

    return A[col * lda + row];
}

template <cublasFillMode_t UPLO>
__device__ __forceinline__ uint2 column_active_range(const unsigned length) {
    if constexpr (UPLO == CUBLAS_FILL_MODE_UPPER) {
        return uint2{threadIdx.x, min(length, blockIdx.x + 1U)};
    } else if constexpr (UPLO == CUBLAS_FILL_MODE_LOWER) {
        return uint2{min(length, blockIdx.x) + threadIdx.x, length};
    } else {
        return uint2{threadIdx.x, length};
    }
}

template <cublasFillMode_t UPLO, cublasDiagType_t DIAG>
__device__ __forceinline__ uint2 column_load_range(const unsigned length) {
    if constexpr (DIAG == CUBLAS_DIAG_UNIT && UPLO == CUBLAS_FILL_MODE_UPPER) {
        return uint2{threadIdx.x, min(length, blockIdx.x)};
    } else if constexpr (DIAG == CUBLAS_DIAG_UNIT && UPLO == CUBLAS_FILL_MODE_LOWER) {
        return uint2{min(length, blockIdx.x + 1U) + threadIdx.x, length};
    } else {
        return column_active_range<UPLO>(length);
    }
}

} // namespace gemmul8::scaling::general
