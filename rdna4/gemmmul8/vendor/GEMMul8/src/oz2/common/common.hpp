#pragma once
#include "include.hpp"
#include "template_type.hpp"
#include "template_math.hpp"
#include "handle.hpp"

namespace gemmul8::common {

#if defined(__HIPCC__)
inline constexpr bool isCUDA = false;
inline constexpr bool isHIP  = true;
#else
inline constexpr bool isCUDA = true;
inline constexpr bool isHIP  = false;
#endif

inline constexpr int TILE_DIM    = 32; // better than 16 for A100, GH200
inline constexpr size_t PAD_SIZE = 256;

//------------------------------
// Pad size to multiple of PAD_SIZE (for alignment)
//------------------------------
static inline size_t padding(const size_t n) { return PAD_SIZE * ((n + PAD_SIZE - 1) / PAD_SIZE); }
static inline void *align(void *p) {
    constexpr std::uintptr_t A = PAD_SIZE;
    std::uintptr_t x           = reinterpret_cast<std::uintptr_t>(p);
    x                          = (x + (A - 1)) & ~(A - 1);
    return reinterpret_cast<void *>(x);
}

} // namespace gemmul8::common
