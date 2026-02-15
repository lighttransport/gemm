#ifndef FP8_QUANT_OPT_H
#define FP8_QUANT_OPT_H

#include <stdint.h>
#include <stddef.h>

// Optimized unrolled kernels (x4 unroll to hide latencies)
void fp16_to_fp8_e5m2_sve_unroll4(const uint16_t* src, uint8_t* dst, size_t n);
void fp32_to_fp8_e5m2_sve_unroll4(const float* src, uint8_t* dst, size_t n);

#endif
