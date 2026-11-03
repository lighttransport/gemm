#pragma once
#include "../../common/common.hpp"

namespace gemmul8::scaling::general {

template <typename Tin, typename Tout>
inline Tout *temporary_memory(Tin *in) {
    static_assert(alignof(Tout) <= common::PAD_SIZE, "Unexpected alignment requirement.");
    static_assert(sizeof(Tin) == 1, "This temporary reuse assumes 1-byte low_t.");

    return reinterpret_cast<Tout *>(in);
}

} // namespace gemmul8::scaling::general
