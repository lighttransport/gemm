#ifndef TURBOQUANT_CPU_INTERNAL_H
#define TURBOQUANT_CPU_INTERNAL_H

#include "turboquant_cpu.h"

#include <stdint.h>

extern const float tq3_centroids[8];
extern const float tq3_thresholds[7];
extern const float tq4_centroids[16];
extern const float tq4_thresholds[15];

uint64_t tq3_splitmix64(uint64_t *x);
float tq3_f16_to_f32(uint16_t h);
uint16_t tq3_f32_to_f16(float f);
void tq3_unpack_indices(uint8_t idx[TQ3_BLOCK_SIZE], const tq3_block *blk);
void tq3_pack_indices(tq3_block *blk, const uint8_t idx[TQ3_BLOCK_SIZE]);
void tq4_unpack_indices(uint8_t idx[TQ3_BLOCK_SIZE], const tq4_block *blk);
void tq4_pack_indices(tq4_block *blk, const uint8_t idx[TQ3_BLOCK_SIZE]);
void tq3_forward_rotate(float y[TQ3_BLOCK_SIZE], const float x[TQ3_BLOCK_SIZE],
                        uint64_t seed, uint64_t block_index);
void tq3_inverse_rotate(float x[TQ3_BLOCK_SIZE], const float y[TQ3_BLOCK_SIZE],
                        uint64_t seed, uint64_t block_index);
float tq3_dot_block_scalar(const tq3_block *blk, const float *x,
                           uint64_t seed, uint64_t block_index);
float tq4_dot_block_scalar(const tq4_block *blk, const float *x,
                           uint64_t seed, uint64_t block_index);

int tq3_x86_has_sse2(void);
int tq3_x86_has_avx2(void);
float tq3_dot_block_sse2(const tq3_block *blk, const float *x,
                         uint64_t seed, uint64_t block_index);
float tq3_dot_block_avx2(const tq3_block *blk, const float *x,
                         uint64_t seed, uint64_t block_index);
float tq4_dot_block_sse2(const tq4_block *blk, const float *x,
                         uint64_t seed, uint64_t block_index);
float tq4_dot_block_avx2(const tq4_block *blk, const float *x,
                         uint64_t seed, uint64_t block_index);

#endif
