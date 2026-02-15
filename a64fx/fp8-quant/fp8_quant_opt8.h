#ifndef FP8_QUANT_OPT8_H
#define FP8_QUANT_OPT8_H

#include <stdint.h>
#include <stddef.h>

// Optimized kernels with x8 unroll for maximum latency hiding
void fp16_to_fp8_e5m2_sve_unroll8(const uint16_t* src, uint8_t* dst, size_t n);
void fp32_to_fp8_e5m2_sve_unroll8(const float* src, uint8_t* dst, size_t n);

#endif
