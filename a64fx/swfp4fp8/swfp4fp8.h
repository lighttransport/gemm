#ifndef SWFP4FP8_H
#define SWFP4FP8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SWFP4FP8_NVFP4_G16 = 0,
    SWFP4FP8_QPN8_TILE32 = 1,
    SWFP4FP8_MXFP4_G32 = 2,
    SWFP4FP8_FP8_BLOCK128 = 3
} swfp4fp8_format;

typedef enum {
    SWFP4FP8_LAYOUT_CANONICAL = 0,
    SWFP4FP8_LAYOUT_V100_QPN = 1
} swfp4fp8_source_layout;

typedef enum {
    SWFP4FP8_KERNEL_AUTO = 0,
    SWFP4FP8_KERNEL_PANEL = 1,
    SWFP4FP8_KERNEL_ROW = 2,
    SWFP4FP8_KERNEL_FP8_FTZ = 3,
    SWFP4FP8_KERNEL_FP4_SDOT = 4
} swfp4fp8_kernel;

typedef struct swfp4fp8_context swfp4fp8_context;
typedef struct swfp4fp8_matrix swfp4fp8_matrix;

/* cpu_ids may be NULL.  The A64FX default is cores 12..(12+nthreads-1). */
int swfp4fp8_context_create(swfp4fp8_context **out, int nthreads,
                            const int *cpu_ids);
void swfp4fp8_context_destroy(swfp4fp8_context *ctx);

/*
 * Logical weights are W[N,K].  Canonical FP4 codes contain two values per
 * byte.  NVFP4 pairs adjacent K values; MXFP4 stores low nibbles for k=0..15
 * and high nibbles for k=16..31 in each K=32 block.
 *
 * NVFP4 scales: uint8 E4M3 [N,K/16], plus global_scale.
 * QPN8 scales:  float [N/32], one scale per output tile (includes the
 *                reference decoder's x256 compensation).
 * MXFP4 scales: uint8 E8M0 [N,K/32].
 * FP8 block scales: uint8 E8M0 [ceil(N/128),K/128].
 *
 * V100_QPN accepts the exact [N/32,K/16,32,8-or-16] packed code order and,
 * for NVFP4, [N/32,K/16,32] packed scales.  N must be a multiple of 32.
 */
int swfp4fp8_pack_nvfp4(swfp4fp8_context *ctx, swfp4fp8_matrix **out,
                        size_t n, size_t k, const uint8_t *codes,
                        const uint8_t *scales, float global_scale,
                        swfp4fp8_source_layout layout);
int swfp4fp8_pack_qpn8(swfp4fp8_context *ctx, swfp4fp8_matrix **out,
                       size_t n, size_t k, const uint8_t *codes,
                       const float *tile_scales,
                       swfp4fp8_source_layout layout);
int swfp4fp8_pack_mxfp4(swfp4fp8_context *ctx, swfp4fp8_matrix **out,
                        size_t n, size_t k, const uint8_t *codes,
                        const uint8_t *scales);
int swfp4fp8_pack_fp8_block128(swfp4fp8_context *ctx,
                               swfp4fp8_matrix **out, size_t n, size_t k,
                               const uint8_t *codes,
                               const uint8_t *block_scales);
void swfp4fp8_matrix_destroy(swfp4fp8_matrix *matrix);

size_t swfp4fp8_matrix_n(const swfp4fp8_matrix *matrix);
size_t swfp4fp8_matrix_k(const swfp4fp8_matrix *matrix);
size_t swfp4fp8_matrix_bytes(const swfp4fp8_matrix *matrix);
swfp4fp8_format swfp4fp8_matrix_format(const swfp4fp8_matrix *matrix);

/* A[M,K] and C[M,N] are row-major.  FP16 is IEEE binary16 in uint16_t. */
int swfp4fp8_gemm_f32(swfp4fp8_context *ctx,
                      const swfp4fp8_matrix *matrix,
                      const float *a, size_t lda, float *c, size_t ldc,
                      size_t m, swfp4fp8_kernel kernel);
int swfp4fp8_gemm_f16(swfp4fp8_context *ctx,
                      const swfp4fp8_matrix *matrix,
                      const uint16_t *a, size_t lda, uint16_t *c, size_t ldc,
                      size_t m, swfp4fp8_kernel kernel);

/* Scalar helpers are public for validation and import tooling. */
float swfp4fp8_decode_e2m1(uint8_t code);
float swfp4fp8_decode_e4m3(uint8_t code);
float swfp4fp8_decode_e8m0(uint8_t code);
int swfp4fp8_unpack_codes(const swfp4fp8_matrix *matrix, uint8_t *canonical);

const char *swfp4fp8_format_name(swfp4fp8_format format);
const char *swfp4fp8_kernel_name(swfp4fp8_kernel kernel);

#ifdef __cplusplus
}
#endif
#endif
