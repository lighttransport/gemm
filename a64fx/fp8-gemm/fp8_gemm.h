#ifndef A64FX_FP8_GEMM_H
#define A64FX_FP8_GEMM_H

#include <stdint.h>
typedef struct {uint32_t k;uint8_t lane,raw,canonical,pad;} fp8_exception;
typedef enum {
    FP8_DECODE_SPARSE_EXACT = 0,
    FP8_DECODE_INLINE_EXACT = 1,
    FP8_DECODE_LUT_EXACT = 2,
    FP8_DECODE_F16_BITS = 3,
} fp8_decode_mode;

typedef struct {
    int n, k;
    uint8_t *codes;       /* [N/128][K][4][32], K-interleaved */
    float *scales;        /* [N/128][K/128] */
    fp8_exception *exceptions;
    uint32_t *exception_offsets; /* N/128 + 1 */
} fp8_matrix;

typedef enum {
    FP8_I8_ABSMAX = 0,
    FP8_I8_MSE = 1,
} fp8_i8_policy;

typedef struct {
    int n, k;
    int scale_group;      /* consecutive K rows sharing one scale */
    int lane_group;       /* output lanes sharing one scale: 32 or 128 */
    int8_t *codes;        /* [N/128][K][4][32], K-interleaved */
    float *scales;        /* combined scale, [N/128][K/scale_group] */
} fp8_i8_matrix;

int fp8_matrix_alloc(fp8_matrix *w, int n, int k);
void fp8_matrix_free(fp8_matrix *w);
int fp8_matrix_prepare_fast(fp8_matrix *w);
float fp8_e4m3fn_decode(uint8_t x);
int fp8_gemv_f32_omp(float *out, const float *a, const fp8_matrix *w,
                     int threads, fp8_decode_mode mode);
int fp8_gemv_reference(float *out, const float *a, const fp8_matrix *w);
int fp8_i8_matrix_alloc(fp8_i8_matrix *w, int n, int k);
void fp8_i8_matrix_free(fp8_i8_matrix *w);
int fp8_matrix_prepare_i8(const fp8_matrix *src, fp8_i8_matrix *dst,
                          fp8_i8_policy policy);
int fp8_matrix_prepare_i8_group(const fp8_matrix *src, fp8_i8_matrix *dst,
                                fp8_i8_policy policy, int scale_group);
int fp8_matrix_prepare_i8_tile(const fp8_matrix *src, fp8_i8_matrix *dst,
                               fp8_i8_policy policy, int scale_group,
                               int lane_group);
int fp8_i8_gemv_f32_omp(float *out, const float *a,
                        const fp8_i8_matrix *w, int threads);
int fp8_i8_gemv_reference(float *out, const float *a,
                          const fp8_i8_matrix *w);

#endif
