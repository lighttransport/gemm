#pragma once

namespace gemmul8::mod {

inline constexpr unsigned threads_x  = 32;
inline constexpr unsigned threads_1d = 256;

template <Backend BACKEND>
inline unsigned select_threads_y(const size_t sizeC) {
    if constexpr (BACKEND == Backend::INT8) {
        return (sizeC <= (1U << 24)) ? 4U : 32U;
    } else {
        return (sizeC <= (1U << 24)) ? 4U : 8U;
    }
}

} // namespace gemmul8::mod
