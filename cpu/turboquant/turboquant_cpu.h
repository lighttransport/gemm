/*
 * turboquant_cpu.h - Standalone TurboQuant3 MSE-only CPU kernels.
 *
 * Format: 128 f32 values -> WHT/PolarQuant 3-bit centroid indices + fp16
 * norm-correction scale. This is intentionally separate from GGML TQ1/TQ2.
 */
#ifndef TURBOQUANT_CPU_H
#define TURBOQUANT_CPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TQ3_BLOCK_SIZE 128
#define TQ3_PACKED_BYTES 48

typedef struct {
    uint16_t scale_f16;
    uint16_t reserved;
    uint8_t qs[TQ3_PACKED_BYTES];
} tq3_block;

size_t tq3_num_blocks(int n);
size_t tq3_row_bytes(int n);

int tq3_quantize_row_f32(tq3_block *dst, const float *src, int n, uint64_t seed);
int tq3_dequantize_row_f32(float *dst, const tq3_block *src, int n, uint64_t seed);
float tq3_dot_row_f32(const tq3_block *qrow, const float *x, int n, uint64_t seed);

const char *tq3_cpu_backend(void);

#ifdef __cplusplus
}
#endif

#endif
