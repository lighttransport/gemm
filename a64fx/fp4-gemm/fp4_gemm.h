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
    uint8_t *codes_u8;
    int8_t *codes_sdot;
    uint8_t *codes_sdot4;
    uint8_t *codes_pair;
    uint8_t *codes_t8;
    uint8_t *codes_t8_half;
    uint8_t *codes_t12;
    uint8_t *codes_t8_affine;
    uint32_t *codes_bitplane;
    /* Persistent K-major FP16 sidecar: one [K][64] tile per N64 group. */
    _Float16 *weights_bf16;
    size_t weights_bf16_bytes;
    _Float16 *scales_n32;
    _Float16 *scales_sdot;
    _Float16 *scales_pair;
    _Float16 *scales_t8;
    _Float16 *scales_t12;
    float *scales_sdot4;
    size_t scales_n32_count;
} fp4_matrix;

typedef struct {
    int k;
    int scale_group;
    int8_t *codes;
    float *scales;        /* activation quantizer scale divided by two */
} fp4_i8_activation;

typedef struct {
    int k;
    int scale_group;
    int8_t *codes;
    float *scales;        /* activation quantizer scale divided by two */
    int16_t *tables;      /* [K/2][256], activation-dependent pair dots */
    int16_t *tables16;    /* [K][32], duplicated 16-entry nibble dots */
} fp4_pair_activation;

const char *fp4_format_name(fp4_format format);
int fp4_matrix_alloc(fp4_matrix *matrix, fp4_format format, int n, int k);
void fp4_matrix_free(fp4_matrix *matrix);
int fp4_quantize_f32(fp4_matrix *matrix, const float *weights);
int fp4_matrix_prepare_n32(fp4_matrix *matrix);
int fp4_matrix_prepare_half(fp4_matrix *matrix);
int fp4_matrix_prepare_affine(fp4_matrix *matrix);
int fp4_matrix_prepare_t12(fp4_matrix *matrix);
int fp4_matrix_prepare_u8(fp4_matrix *matrix);
int fp4_matrix_prepare_bitplane(fp4_matrix *matrix);
int fp4_matrix_prepare_sdot(fp4_matrix *matrix);
int fp4_matrix_prepare_sdot4(fp4_matrix *matrix);
int fp4_matrix_prepare_pair(fp4_matrix *matrix);
int fp4_matrix_prepare_bf16(fp4_matrix *matrix, int threads);
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
int fp4_gemm_f16_l1panel(float *c, const _Float16 *a, const fp4_matrix *w,
                          int m, int promotion_k);
int fp4_gemm_f16_l2_omp(float *c, const _Float16 *a, const fp4_matrix *w,
                         int m, int promotion_k, int threads);
int fp4_gemm_f16_n32_omp(float *c, const _Float16 *a, const fp4_matrix *w,
                          int m, int promotion_k, int threads);
int fp4_gemm_f16_bf16cache_omp(float *c, const _Float16 *a,
                                const fp4_matrix *w, int m,
                                int promotion_k, int threads);
/* Pack A as [ceil(M/12)][K][12]. The caller owns the buffer and may reuse it
 * for any number of same-K weight projections while the source A is fixed. */
size_t fp4_packed_a_m12_bytes(int m, int k);
int fp4_pack_a_m12(_Float16 *packed_a, const _Float16 *a, int m, int k,
                    int threads);
int fp4_gemm_f16_bf16cache_prepacked_omp(float *c,
                                          const _Float16 *packed_a,
                                          const fp4_matrix *w, int m,
                                          int promotion_k, int threads);
int fp4_gemm_f16_half_omp(float *c, const _Float16 *a, const fp4_matrix *w,
                           int m, int promotion_k, int threads);
int fp4_gemm_f16_t12_omp(float *c, const _Float16 *a, const fp4_matrix *w,
                          int m, int promotion_k, int threads);
int fp4_gemm_f16_affine_omp(float *c, const _Float16 *a, const fp4_matrix *w,
                             int m, int promotion_k, int threads);
int fp4_gemm_f16_u8tbl_omp(float *c, const _Float16 *a, const fp4_matrix *w,
                         int m, int promotion_k, int threads);
int fp4_gemm_f16_bitplane_omp(float *c, const _Float16 *a,
                         const fp4_matrix *w, int m, int promotion_k,
                         int threads);
int fp4_i8_activation_prepare(fp4_i8_activation *q, const float *a, int k,
                               int scale_group);
void fp4_i8_activation_free(fp4_i8_activation *q);
int fp4_gemv_i8_sdot_omp(float *c, const fp4_i8_activation *a,
                          const fp4_matrix *w, int threads);
int fp4_gemv_i8_sdot4_omp(float *c, const fp4_i8_activation *a,
                           const fp4_matrix *w, int threads);
int fp4_pair_activation_prepare(fp4_pair_activation *q, const float *a,
                                 int k, int scale_group);
void fp4_pair_activation_free(fp4_pair_activation *q);
int fp4_gemv_pair_lut_omp(float *c, const fp4_pair_activation *a,
                           const fp4_matrix *w, int threads);
int fp4_gemv_pair_tbl_omp(float *c, const fp4_pair_activation *a,
                           const fp4_matrix *w, int threads);
int fp4_gemm_reference(float *c, const _Float16 *a, const fp4_matrix *w,
                        int m, int fp16_products);

uint8_t fp4_e2m1_encode(float value);
float fp4_e2m1_decode(uint8_t value);
uint8_t fp4_e8m0_encode_ceil(float value);
float fp4_e8m0_decode(uint8_t value);
uint8_t fp4_e4m3_encode_positive(float value);
float fp4_e4m3_decode_positive(uint8_t value);

#endif
