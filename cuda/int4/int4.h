#ifndef CUDA_INT4_H
#define CUDA_INT4_H

#include <stddef.h>
#include <stdint.h>
#include "../cuew.h"
#include "int4_numeric.h"

enum { INT4_E2M1 = 0, INT4_E0M3 = 1 };
enum { INT4_E2_E2 = 0, INT4_E0_E2 = 1, INT4_E2_E0 = 2, INT4_E0_E0 = 3 };
enum { INT4_VARIANTS = 9 };

typedef struct {
    CUdeviceptr codes, scales;
    int rows, k, kp;
} int4_matrix;

typedef struct {
    uint64_t clipped_values, clamped_scales;
} int4_quant_stats;

typedef struct {
    CUmodule modules[4];
    CUfunction gemm[4][INT4_VARIANTS], quant, dequant_bf16, unpack_int8;
    CUfunction probe[4], probe_bf16, probe_int8, probe_s4;
} int4_context;

extern const char *const int4_variant_names[INT4_VARIANTS];
int int4_padded_k(int k);
size_t int4_codes_size(int rows, int k);
size_t int4_scales_size(int rows, int k);
int int4_quantize_cpu(const float *src, int rows, int k, uint8_t *codes,
                      uint8_t *scales, int format, int4_quant_stats *stats);
float int4_decode_cpu(const uint8_t *codes, const uint8_t *scales,
                      int row, int col, int kp, int format);

/* Caller owns the current CUDA context and all matrix/workspace allocations.
 * Code pointers require 16-byte alignment; scales and FP32 pointers require
 * 4-byte alignment. Statistics require 8-byte alignment.
 * No allocation, tuning or synchronization occurs in quantize/gemm calls.
 * Input to quantization must be finite FP32. Optional device stats are two
 * uint64 counters, accumulated into caller-zeroed storage. */
int int4_init(int4_context *ctx, const char *cubin_dir);
void int4_destroy(int4_context *ctx);
CUresult int4_quantize_f32(int4_context *ctx, CUdeviceptr src, int4_matrix dst,
                          int format, CUdeviceptr stats, CUstream stream);
CUresult int4_gemm(int4_context *ctx, int4_matrix a, int4_matrix b,
                   CUdeviceptr output, int formats, int variant, CUstream stream);

#endif
