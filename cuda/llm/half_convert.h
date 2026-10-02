/* Host IEEE-754 half conversion, round to nearest with ties to even. */
#ifndef CLLM_HALF_CONVERT_H
#define CLLM_HALF_CONVERT_H
#include <stdint.h>
#include <string.h>

static inline uint16_t cllm_f32_to_f16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x7fffffu;
    int exponent = (int)((bits >> 23) & 255u) - 127;
    if (exponent == 128)
        return (uint16_t)(sign | 0x7c00u | (mantissa ? ((mantissa >> 13) | 0x200u) : 0u));
    if (exponent > 15) return (uint16_t)(sign | 0x7c00u);
    if (exponent < -25) return (uint16_t)sign;
    uint32_t result, remainder, halfway;
    if (exponent < -14) {
        /* Half subnormals have unit 2^-24; do not shift by 13 again. */
        int shift = -exponent - 1;
        mantissa |= 0x800000u;
        result = mantissa >> shift;
        remainder = mantissa & ((1u << shift) - 1u);
        halfway = 1u << (shift - 1);
    } else {
        result = ((uint32_t)(exponent + 15) << 10) | (mantissa >> 13);
        remainder = mantissa & 0x1fffu;
        halfway = 0x1000u;
    }
    result += remainder > halfway || (remainder == halfway && (result & 1u));
    return (uint16_t)(sign | result);
}

static inline uint16_t cllm_bf16_to_f16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float number;
    memcpy(&number, &bits, sizeof(number));
    return cllm_f32_to_f16(number);
}
#endif
