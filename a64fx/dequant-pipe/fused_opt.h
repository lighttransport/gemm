#ifndef FUSED_OPT_H
#define FUSED_OPT_H
#include <stddef.h>
#include <stdint.h>

/* Fixed SVE512, K=128, N=64. Full A16 records contain two K-major
 * split-nibble blocks (8192 bytes), followed by 128 signed INT16 column sums.
 * Activation digits are [b0 low][b0 high][b1 low][b1 high], 128 bytes each.
 * Outputs are 128 INT32 values in block/column order. FP4 uses its x2 lattice.
 * FP16 kernels retain the existing four-block N-lane layout and K order.
 * All kernels omit model block scales. */
enum { FUSED_FULL_BYTES = 8448 };
void pack_i16x8_full(int8_t *, const int16_t *, size_t);
void pack_weight_sums(uint8_t *, int);
int verify_fused_opt(void);
void report_fused_mapping(const void *);
void fused_i4_i16x8_full_sve(const uint8_t *, const int8_t *, int32_t *);
void fused_fp4_i16x8_full_sve(const uint8_t *, const int8_t *, int32_t *);
void fused_i4_i16_opt_sve(const uint8_t *, const int16_t *, int64_t *);
void fused_fp4_i16_opt_sve(const uint8_t *, const int16_t *, int64_t *);
void fused_i4_f16_opt1_sve(const uint8_t *, const _Float16 *, _Float16 *);
void fused_fp4_f16_opt1_sve(const uint8_t *, const _Float16 *, _Float16 *);
void fused_i4_f16_opt2_sve(const uint8_t *, const _Float16 *, _Float16 *);
void fused_fp4_f16_opt2_sve(const uint8_t *, const _Float16 *, _Float16 *);
void fused_i4_f16_pipe_sve(const uint8_t *, const _Float16 *, _Float16 *);
void fused_fp4_f16_pipe_sve(const uint8_t *, const _Float16 *, _Float16 *);
#endif
