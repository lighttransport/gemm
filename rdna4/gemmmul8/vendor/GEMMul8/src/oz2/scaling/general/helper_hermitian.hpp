#pragma once
#include "../../common/common.hpp"

namespace gemmul8::scaling::general {

template <bool HERM, typename T>
__device__ __forceinline__ T hemm_diag_value(
    T a,
    const unsigned row_idx, const unsigned col //
) {
    if constexpr (HERM && common::isComplex<T>) {
        a.y = (row_idx == col)
                  ? common::Tconst<common::underlying_t<T>>::zero()
                  : a.y;
    }
    return a;
}

template <bool HERM, typename T>
__device__ __forceinline__ T hemm_mirror_value(T a) {
    constexpr bool CONJ = HERM && common::isComplex<T>;
    return common::conj<T, CONJ>(a);
}

template <bool HERM, bool STORE_TRANSPOSE, typename T>
__device__ __forceinline__ T hemm_active_store_value(
    T a,
    const unsigned row,
    const unsigned col //
) {
    a = hemm_diag_value<HERM, T>(a, row, col);

    if constexpr (HERM && STORE_TRANSPOSE && common::isComplex<T>) {
        return common::conj<T, true>(a);
    } else {
        return a;
    }
}

template <bool HERM, bool STORE_TRANSPOSE, typename T>
__device__ __forceinline__ T hemm_mirror_store_value(T a) {
    if constexpr (HERM && common::isComplex<T>) {
        if constexpr (STORE_TRANSPOSE) {
            return a;
        } else {
            return common::conj<T, true>(a);
        }
    } else {
        return a;
    }
}

} // namespace gemmul8::scaling::general
