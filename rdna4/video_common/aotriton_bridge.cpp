// SPDX-License-Identifier: MIT
#include "aotriton_bridge.h"
#include <aotriton/flash.h>
#include <cmath>

extern "C" int video_aotriton_bridge_abi(void) { return 1; }

static int forward(uint64_t q, uint64_t k, uint64_t v, uint64_t out, uint64_t lse, int rows,
                   int heads, int kv_heads, int dim, int kind, float scale, void *stream,
                   bool head_major) {
    using namespace aotriton;
    using namespace aotriton::v2::flash;
    if (!q || !k || !v || !out || !lse || rows < 1 || heads < 1 || kv_heads < 1 ||
        heads % kv_heads || (dim != 64 && dim != 128) || (kind != 1 && kind != 2) ||
        !std::isfinite(scale))
        return int(hipErrorInvalidValue);
    const auto dtype = kind == 1 ? kBFloat16 : kFloat16;
    auto tensor = [&](uint64_t pointer, int count, bool packed) {
        return T4(intptr_t(pointer), {1, uint64_t(count), uint64_t(rows), uint64_t(dim)},
                  {uint64_t(rows) * count * dim, packed ? uint64_t(rows) * dim : uint64_t(dim),
                   packed ? uint64_t(dim) : uint64_t(count) * dim, 1},
                  dtype);
    };
    auto null4 = T4::get_null_tensor(dtype);
    auto null0 = T0::get_null_tensor(kUInt64);
    T2 logsum(intptr_t(lse), {uint64_t(heads), uint64_t(rows)}, {uint64_t(rows), 1}, kFloat32);
    return int(attn_fwd(tensor(q, heads, head_major), tensor(k, kv_heads, head_major),
                        tensor(v, kv_heads, head_major), null4, scale, logsum,
                        tensor(out, heads, false), 0.f, null0, null0, 0, null0, null0, null4, false,
                        null0, Stream(reinterpret_cast<hipStream_t>(stream)), nullptr));
}

extern "C" int video_aotriton_forward(uint64_t q, uint64_t k, uint64_t v, uint64_t out,
                                      uint64_t lse, int rows, int heads, int kv_heads, int dim,
                                      int kind, float scale, void *stream) {
    return forward(q, k, v, out, lse, rows, heads, kv_heads, dim, kind, scale, stream, false);
}
extern "C" int video_aotriton_forward_heads(uint64_t q, uint64_t k, uint64_t v, uint64_t out,
                                            uint64_t lse, int rows, int heads, int kv_heads,
                                            int dim, int kind, float scale, void *stream) {
    return forward(q, k, v, out, lse, rows, heads, kv_heads, dim, kind, scale, stream, true);
}
