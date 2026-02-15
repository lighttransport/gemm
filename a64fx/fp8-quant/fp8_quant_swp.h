#ifndef FP8_QUANT_SWP_H
#define FP8_QUANT_SWP_H

#include <stdint.h>
#include <stddef.h>

// Software pipelined kernels - overlap load/compute/store across iterations
void fp16_to_fp8_e5m2_sve_swp(const uint16_t* src, uint8_t* dst, size_t n);
void fp32_to_fp8_e5m2_sve_swp(const float* src, uint8_t* dst, size_t n);

// FCVT-based kernel: FP32 -> FP16 (hardware) -> E5M2 (bit ops)
void fp32_to_fp8_e5m2_sve_fcvt(const float* src, uint8_t* dst, size_t n);

#endif
