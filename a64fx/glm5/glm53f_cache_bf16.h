#ifndef GLM53F_CACHE_BF16_H
#define GLM53F_CACHE_BF16_H
#include <stdint.h>
#include <string.h>

/* Round-to-nearest, ties-to-even. Preserve infinities and keep NaNs as NaNs. */
static inline uint16_t glm53f_cache_bf16_round(float x) {
    uint32_t bits; memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return (uint16_t)(bits >> 16) | ((bits & 0x7fffffu) ? 0x40u : 0u);
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static inline float glm53f_cache_bf16_expand(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16; float out;
    memcpy(&out, &bits, sizeof(out)); return out;
}
#endif
