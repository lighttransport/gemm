#ifndef A64FX_FP4_GEMM_H
#define A64FX_FP4_GEMM_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    FP4_MX = 0,
    FP4_NV_1D = 1,
    FP4_NV_2D = 2,
} fp4_format;

typedef struct {
    fp4_format format;
    int n;
    int k;
    float global_scale;
    uint8_t *codes;
    uint8_t *scales;
    size_t code_bytes;
    size_t scale_bytes;
    /* Single-core compute layout: [N/32][K][16 packed bytes] plus one
     * FP16 scale vector for every format scaling block. */
    uint8_t *codes_n32;
    _Float16 *scales_n32;
    size_t scales_n32_count;
} fp4_matrix;

const char *fp4_format_name(fp4_format format);
int fp4_matrix_alloc(fp4_matrix *matrix, fp4_format format, int n, int k);
void fp4_matrix_free(fp4_matrix *matrix);
int fp4_quantize_f32(fp4_matrix *matrix, const float *weights);
int fp4_matrix_prepare_n32(fp4_matrix *matrix);
float fp4_dequant_value(const fp4_matrix *matrix, int row, int col);

/* C[M,N] = A[M,K] * W[N,K]^T. A is FP16 and C is FP32. promotion_k is
 * 0 for one pure-FP16 accumulation span, otherwise a multiple of 32. */
int fp4_gemm_f16(float *c, const _Float16 *a, const fp4_matrix *w,
                  int m, int promotion_k, int threads);
int fp4_gemm_f16_n32(float *c, const _Float16 *a, const fp4_matrix *w,
                      int m, int promotion_k);
int fp4_gemm_f16_l1(float *c, const _Float16 *a, const fp4_matrix *w,
                     int m, int promotion_k);
int fp4_gemm_f16_l2(float *c, const _Float16 *a, const fp4_matrix *w,
                     int m, int promotion_k);
int fp4_gemm_reference(float *c, const _Float16 *a, const fp4_matrix *w,
                        int m, int fp16_products);

uint8_t fp4_e2m1_encode(float value);
float fp4_e2m1_decode(uint8_t value);
uint8_t fp4_e8m0_encode_ceil(float value);
float fp4_e8m0_decode(uint8_t value);
uint8_t fp4_e4m3_encode_positive(float value);
float fp4_e4m3_decode_positive(uint8_t value);

#endif
