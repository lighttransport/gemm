#ifndef FP8_QUANT_H
#define FP8_QUANT_H

#include <stdint.h>
#include <stddef.h>

// FP8 E4M3 format: 1 sign, 4 exponent, 3 mantissa bits
// Bias = 7, max value = 448 (with special NaN encoding)
// FP8 E5M2 format: 1 sign, 5 exponent, 2 mantissa bits
// Bias = 15, max value = 57344

typedef uint8_t fp8_e4m3_t;
typedef uint8_t fp8_e5m2_t;

// FP16 -> FP8 conversions
void fp16_to_fp8_e4m3_sve(const uint16_t* src, fp8_e4m3_t* dst, size_t n);
void fp16_to_fp8_e5m2_sve(const uint16_t* src, fp8_e5m2_t* dst, size_t n);

// FP32 -> FP8 conversions
void fp32_to_fp8_e4m3_sve(const float* src, fp8_e4m3_t* dst, size_t n);
void fp32_to_fp8_e5m2_sve(const float* src, fp8_e5m2_t* dst, size_t n);

// Scalar reference implementations for validation
fp8_e4m3_t fp16_to_fp8_e4m3_scalar(uint16_t x);
fp8_e5m2_t fp16_to_fp8_e5m2_scalar(uint16_t x);
fp8_e4m3_t fp32_to_fp8_e4m3_scalar(float x);
fp8_e5m2_t fp32_to_fp8_e5m2_scalar(float x);

// FP8 -> FP16/FP32 dequantization
uint16_t fp8_e4m3_to_fp16_scalar(fp8_e4m3_t x);
uint16_t fp8_e5m2_to_fp16_scalar(fp8_e5m2_t x);
float fp8_e4m3_to_fp32_scalar(fp8_e4m3_t x);
float fp8_e5m2_to_fp32_scalar(fp8_e5m2_t x);

#endif // FP8_QUANT_H
