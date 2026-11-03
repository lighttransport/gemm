#pragma once
#include "../common/common.hpp"

namespace gemmul8::oz2::core {

template <cublasFillMode_t UPLO>
inline constexpr cublasFillMode_t flip_uplo =
    (UPLO == CUBLAS_FILL_MODE_UPPER) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

}
