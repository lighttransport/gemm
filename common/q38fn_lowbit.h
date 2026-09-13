#ifndef Q38FN_LOWBIT_H
#define Q38FN_LOWBIT_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct __attribute__((packed)) {
    uint16_t scale;
    uint8_t low[16];
    uint32_t high;
} q38fn_q5_block;

_Static_assert(sizeof(q38fn_q5_block) == 22, "Q5 block layout");

size_t q38fn_q5_bytes(size_t rows, size_t columns);
size_t q38fn_q8_bytes(size_t rows, size_t columns);
int q38fn_q8_quantize_bf16(int8_t *destination, float *scales,
                            const uint16_t *source, size_t rows,
                            size_t columns);
int q38fn_q5_quantize_bf16(q38fn_q5_block *destination,
                           const uint16_t *source,
                           size_t rows, size_t columns);
int q38fn_q5_matvec(float *output, const q38fn_q5_block *weight,
                    const float *input, size_t rows, size_t columns);
int q38fn_q5_matvec_pair(float output[2], const q38fn_q5_block *weight0,
                         const q38fn_q5_block *weight1,
                         const float *input, size_t columns);
/* Batched matrices share the same row/column shape.  The matrices and
 * vectors are row-major and contiguous; this keeps one OpenMP team alive for
 * the whole expert group instead of starting one team per matrix. */
int q38fn_q5_matvec_many(float *outputs, const q38fn_q5_block *weights,
                         const float *inputs, size_t matrices, size_t rows,
                         size_t columns);
int q38fn_q5_matvec_indexed(float *outputs, const q38fn_q5_block *weights,
                            const float *inputs, const size_t *indices,
                            size_t matrices, size_t rows, size_t columns);
int q38fn_q5_dequantize_row(float *output, const q38fn_q5_block *weight,
                            size_t columns);

#ifdef Q38FN_LOWBIT_IMPLEMENTATION

#include <math.h>
#include <string.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

static float q38fn_q5_bf16(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t q38fn_q5_f32_to_f16(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = (int)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t fraction = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        fraction = (fraction | 0x800000u) >> (1 - exponent);
        return (uint16_t)(sign | ((fraction + 0x1000u) >> 13));
    }
    if (exponent >= 31)
        return (uint16_t)(sign | 0x7c00u | (fraction ? 0x0200u : 0));
    fraction += 0x1000u;
    if (fraction & 0x800000u) {
        fraction = 0;
        if (++exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (fraction >> 13));
}

static float q38fn_q5_f16_to_f32(uint16_t value)
{
    uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)value >> 10) & 31u;
    uint32_t fraction = (uint32_t)value & 1023u;
    uint32_t bits;
    if (!exponent) {
        if (!fraction) bits = sign;
        else {
            exponent = 127 - 15 + 1;
            while (!(fraction & 1024u)) { fraction <<= 1; --exponent; }
            bits = sign | (exponent << 23) | ((fraction & 1023u) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (fraction << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

size_t q38fn_q5_bytes(size_t rows, size_t columns)
{
    if (!rows || !columns || columns % 32 || rows > SIZE_MAX / columns)
        return 0;
    return rows * (columns / 32) * sizeof(q38fn_q5_block);
}

size_t q38fn_q8_bytes(size_t rows, size_t columns)
{
    return rows * columns + rows * sizeof(float);
}

int q38fn_q8_quantize_bf16(int8_t *destination, float *scales,
                           const uint16_t *source, size_t rows,
                           size_t columns)
{
    if (!destination || !scales || !source || !columns) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0; row < rows; ++row) {
        float maximum = 0.0f;
        for (size_t column = 0; column < columns; ++column)
            maximum = fmaxf(maximum, fabsf(q38fn_q5_bf16(
                source[row * columns + column])));
        float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scales[row] = scale;
        float inverse = 1.0f / scale;
        for (size_t column = 0; column < columns; ++column) {
            long value = lrintf(q38fn_q5_bf16(
                source[row * columns + column]) * inverse);
            destination[row * columns + column] = (int8_t)(value < -127 ? -127 :
                value > 127 ? 127 : value);
        }
    }
    return 0;
}

int q38fn_q5_quantize_bf16(q38fn_q5_block *destination,
                           const uint16_t *source,
                           size_t rows, size_t columns)
{
    if (!destination || !source || !q38fn_q5_bytes(rows, columns)) return -1;
    size_t blocks = columns / 32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks; ++block) {
            const uint16_t *input = source + row * columns + block * 32;
            q38fn_q5_block *output = destination + row * blocks + block;
            float maximum = 0.0f;
            for (int i = 0; i < 32; ++i)
                maximum = fmaxf(maximum, fabsf(q38fn_q5_bf16(input[i])));
            float scale = maximum > 0.0f ? maximum / 15.0f : 1.0f;
            output->scale = q38fn_q5_f32_to_f16(scale);
            output->high = 0;
            memset(output->low, 0, sizeof(output->low));
            for (int i = 0; i < 32; ++i) {
                long quantized = lrintf(q38fn_q5_bf16(input[i]) / scale);
                if (quantized < -16) quantized = -16;
                if (quantized > 15) quantized = 15;
                unsigned code = (unsigned)(quantized + 16);
                if (i < 16) output->low[i] = (uint8_t)(code & 15u);
                else output->low[i - 16] |= (uint8_t)((code & 15u) << 4);
                output->high |= ((code >> 4) & 1u) << i;
            }
        }
    }
    return 0;
}

int q38fn_q5_matvec(float *output, const q38fn_q5_block *weight,
                    const float *input, size_t rows, size_t columns)
{
    if (!output || !weight || !input || !q38fn_q5_bytes(rows, columns)) return -1;
    size_t blocks = columns / 32;
#if defined(__ARM_FEATURE_SVE) && !defined(Q38FN_Q5_SCALAR)
    static int prefetch_blocks = -1;
    if (prefetch_blocks < 0) {
        const char *value = getenv("Q38FN_TP_Q5_PREFETCH_BLOCKS");
        prefetch_blocks = value ? atoi(value) : 32;
        if (prefetch_blocks < 0) prefetch_blocks = 0;
    }
    svbool_t predicate = svptrue_b32();
    svuint32_t indices = svindex_u32(0, 1);
    /* Four independent rows expose enough accumulator and decode ILP to hide
     * A64FX load/convert latency.  Keep this opt-in until end-to-end TP runs
     * establish which matrix sizes benefit. */
    int use_4row = rows >= 16 && getenv("Q38FN_TP_Q5_4ROW");
    if (use_4row && getenv("Q38FN_TP_Q5_4ROW_GATE_ONLY"))
        use_4row = rows == 256 && columns == 6144;
    if (use_4row) {
        size_t groups = rows / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t group = 0; group < groups; ++group) {
            svfloat32_t s00=svdup_f32(0),s01=svdup_f32(0);
            svfloat32_t s10=svdup_f32(0),s11=svdup_f32(0);
            svfloat32_t s20=svdup_f32(0),s21=svdup_f32(0);
            svfloat32_t s30=svdup_f32(0),s31=svdup_f32(0);
            for (size_t block = 0; block < blocks; ++block) {
                svfloat32_t x0 = svld1_f32(predicate, input + block * 32);
                svfloat32_t x1 = svld1_f32(predicate, input + block * 32 + 16);
#define Q38FN_Q5_ACCUM_ROW(R,A0,A1) do { const q38fn_q5_block *source=weight+(group*4+(size_t)(R))*blocks+block; uint32_t high; memcpy(&high,&source->high,sizeof(high)); svuint32_t low=svld1ub_u32(predicate,source->low); svuint32_t high0=svand_n_u32_x(predicate,svlsr_u32_x(predicate,svdup_u32(high),indices),1); svuint32_t high1=svand_n_u32_x(predicate,svlsr_u32_x(predicate,svdup_u32(high>>16),indices),1); svint32_t value0=svsub_n_s32_x(predicate,svreinterpret_s32_u32(svorr_u32_x(predicate,svand_n_u32_x(predicate,low,15),svlsl_n_u32_x(predicate,high0,4))),16); svint32_t value1=svsub_n_s32_x(predicate,svreinterpret_s32_u32(svorr_u32_x(predicate,svlsr_n_u32_x(predicate,low,4),svlsl_n_u32_x(predicate,high1,4))),16); svfloat32_t scale=svdup_f32(q38fn_q5_f16_to_f32(source->scale)); (A0)=svmla_f32_x(predicate,(A0),x0,svmul_f32_x(predicate,svcvt_f32_s32_x(predicate,value0),scale)); (A1)=svmla_f32_x(predicate,(A1),x1,svmul_f32_x(predicate,svcvt_f32_s32_x(predicate,value1),scale)); } while(0)
                Q38FN_Q5_ACCUM_ROW(0,s00,s01);
                Q38FN_Q5_ACCUM_ROW(1,s10,s11);
                Q38FN_Q5_ACCUM_ROW(2,s20,s21);
                Q38FN_Q5_ACCUM_ROW(3,s30,s31);
#undef Q38FN_Q5_ACCUM_ROW
            }
            output[group*4]=svaddv_f32(predicate,s00)+svaddv_f32(predicate,s01);
            output[group*4+1]=svaddv_f32(predicate,s10)+svaddv_f32(predicate,s11);
            output[group*4+2]=svaddv_f32(predicate,s20)+svaddv_f32(predicate,s21);
            output[group*4+3]=svaddv_f32(predicate,s30)+svaddv_f32(predicate,s31);
        }
        for (size_t row = groups * 4; row < rows; ++row)
            if (q38fn_q5_matvec(output + row, weight + row * blocks,
                                input, 1, columns)) return -1;
        return 0;
    }
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows >= 8)
#endif
    for (size_t row = 0; row < rows; ++row) {
#if defined(__ARM_FEATURE_SVE) && !defined(Q38FN_Q5_SCALAR)
        svfloat32_t sum0 = svdup_f32(0), sum1 = svdup_f32(0);
        svfloat32_t sum2 = svdup_f32(0), sum3 = svdup_f32(0);
        for (size_t block = 0; block < blocks; ++block) {
            const q38fn_q5_block *source = weight + row * blocks + block;
            if (prefetch_blocks && block + (size_t)prefetch_blocks < blocks)
                __builtin_prefetch(source + prefetch_blocks, 0, 0);
            uint32_t high;
            memcpy(&high, &source->high, sizeof(high));
            svuint32_t low = svld1ub_u32(predicate, source->low);
            svuint32_t high0 = svand_n_u32_x(predicate,
                svlsr_u32_x(predicate, svdup_u32(high), indices), 1);
            svuint32_t high1 = svand_n_u32_x(predicate,
                svlsr_u32_x(predicate, svdup_u32(high >> 16), indices), 1);
            svint32_t value0 = svsub_n_s32_x(predicate,
                svreinterpret_s32_u32(svorr_u32_x(predicate,
                    svand_n_u32_x(predicate, low, 15),
                    svlsl_n_u32_x(predicate, high0, 4))), 16);
            svint32_t value1 = svsub_n_s32_x(predicate,
                svreinterpret_s32_u32(svorr_u32_x(predicate,
                    svlsr_n_u32_x(predicate, low, 4),
                    svlsl_n_u32_x(predicate, high1, 4))), 16);
            svfloat32_t scale = svdup_f32(q38fn_q5_f16_to_f32(source->scale));
            svfloat32_t x0 = svld1_f32(predicate, input + block * 32);
            svfloat32_t x1 = svld1_f32(predicate, input + block * 32 + 16);
            svfloat32_t product0=svmul_f32_x(predicate,svcvt_f32_s32_x(predicate,value0),scale);
            svfloat32_t product1=svmul_f32_x(predicate,svcvt_f32_s32_x(predicate,value1),scale);
            if(block&1){sum2=svmla_f32_x(predicate,sum2,x0,product0);sum3=svmla_f32_x(predicate,sum3,x1,product1);}
            else{sum0=svmla_f32_x(predicate,sum0,x0,product0);sum1=svmla_f32_x(predicate,sum1,x1,product1);}
        }
        output[row] = svaddv_f32(predicate, svadd_f32_x(predicate,sum0,sum2)) +
                      svaddv_f32(predicate, svadd_f32_x(predicate,sum1,sum3));
#else
        float sum = 0.0f;
        for (size_t block = 0; block < blocks; ++block) {
            const q38fn_q5_block *source = weight + row * blocks + block;
            float scale = q38fn_q5_f16_to_f32(source->scale);
            for (int i = 0; i < 32; ++i) {
                unsigned low = i < 16 ? source->low[i] & 15u :
                                        source->low[i - 16] >> 4;
                int value = (int)(low | (((source->high >> i) & 1u) << 4)) - 16;
                sum += scale * value * input[block * 32 + (size_t)i];
            }
        }
        output[row] = sum;
#endif
    }
    return 0;
}

int q38fn_q5_matvec_pair(float output[2], const q38fn_q5_block *weight0,
                         const q38fn_q5_block *weight1,
                         const float *input, size_t columns)
{
    if (!output || !weight0 || !weight1 || !input || !columns ||
        columns % 32) return -1;
#if defined(__ARM_FEATURE_SVE) && !defined(Q38FN_Q5_SCALAR)
    svbool_t pg = svptrue_b32();
    svuint32_t indices = svindex_u32(0, 1);
    svfloat32_t s00 = svdup_f32(0), s01 = svdup_f32(0);
    svfloat32_t s10 = svdup_f32(0), s11 = svdup_f32(0);
    size_t blocks = columns / 32;
    for (size_t block = 0; block < blocks; ++block) {
        svfloat32_t x0 = svld1_f32(pg, input + block * 32);
        svfloat32_t x1 = svld1_f32(pg, input + block * 32 + 16);
#define Q38FN_Q5_PAIR_ROW(W,A0,A1) do { \
        const q38fn_q5_block *source = (W) + block; \
        uint32_t high; memcpy(&high, &source->high, sizeof(high)); \
        svuint32_t low = svld1ub_u32(pg, source->low); \
        svuint32_t h0 = svand_n_u32_x(pg, svlsr_u32_x(pg, svdup_u32(high), indices), 1); \
        svuint32_t h1 = svand_n_u32_x(pg, svlsr_u32_x(pg, svdup_u32(high >> 16), indices), 1); \
        svint32_t v0 = svsub_n_s32_x(pg, svreinterpret_s32_u32(svorr_u32_x(pg, svand_n_u32_x(pg, low, 15), svlsl_n_u32_x(pg, h0, 4))), 16); \
        svint32_t v1 = svsub_n_s32_x(pg, svreinterpret_s32_u32(svorr_u32_x(pg, svlsr_n_u32_x(pg, low, 4), svlsl_n_u32_x(pg, h1, 4))), 16); \
        svfloat32_t scale = svdup_f32(q38fn_q5_f16_to_f32(source->scale)); \
        (A0) = svmla_f32_x(pg, (A0), x0, svmul_f32_x(pg, svcvt_f32_s32_x(pg, v0), scale)); \
        (A1) = svmla_f32_x(pg, (A1), x1, svmul_f32_x(pg, svcvt_f32_s32_x(pg, v1), scale)); \
    } while (0)
        Q38FN_Q5_PAIR_ROW(weight0, s00, s01);
        Q38FN_Q5_PAIR_ROW(weight1, s10, s11);
#undef Q38FN_Q5_PAIR_ROW
    }
    output[0] = svaddv_f32(pg, s00) + svaddv_f32(pg, s01);
    output[1] = svaddv_f32(pg, s10) + svaddv_f32(pg, s11);
    return 0;
#else
    if (q38fn_q5_matvec(output, weight0, input, 1, columns)) return -1;
    return q38fn_q5_matvec(output + 1, weight1, input, 1, columns);
#endif
}

int q38fn_q5_matvec_many(float *outputs, const q38fn_q5_block *weights,
                         const float *inputs, size_t matrices, size_t rows,
                         size_t columns)
{
    if (!outputs || !weights || !inputs || !matrices || !rows ||
        !q38fn_q5_bytes(rows, columns)) return -1;
    size_t blocks = columns / 32;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (size_t matrix = 0; matrix < matrices; ++matrix) {
        for (size_t row = 0; row < rows; ++row) {
            q38fn_q5_matvec(outputs + matrix * rows + row,
                weights + matrix * rows * blocks + row * blocks,
                inputs + matrix * columns, 1, columns);
        }
    }
    return 0;
}

int q38fn_q5_matvec_indexed(float *outputs, const q38fn_q5_block *weights,
                            const float *inputs, const size_t *indices,
                            size_t matrices, size_t rows, size_t columns)
{
    if (!outputs || !weights || !inputs || !indices || !matrices || !rows ||
        !q38fn_q5_bytes(rows, columns)) return -1;
    size_t blocks = columns / 32;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (size_t matrix = 0; matrix < matrices; ++matrix) {
        for (size_t row = 0; row < rows; ++row) {
            q38fn_q5_matvec(outputs + matrix * rows + row,
                weights + indices[matrix] * rows * blocks + row * blocks,
                inputs + matrix * columns, 1, columns);
        }
    }
    return 0;
}

int q38fn_q5_dequantize_row(float *output, const q38fn_q5_block *weight,
                            size_t columns)
{
    if (!output || !weight || !columns || columns % 32) return -1;
    for (size_t block = 0; block < columns / 32; ++block) {
        const q38fn_q5_block *source = weight + block;
        float scale = q38fn_q5_f16_to_f32(source->scale);
        for (int i = 0; i < 32; ++i) {
            unsigned low = i < 16 ? source->low[i] & 15u :
                                    source->low[i - 16] >> 4;
            int value = (int)(low | (((source->high >> i) & 1u) << 4)) - 16;
            output[block * 32 + (size_t)i] = scale * value;
        }
    }
    return 0;
}
#endif
#endif
