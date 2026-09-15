#ifndef GLM53F_PREFILL_GEMM_H
#define GLM53F_PREFILL_GEMM_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#include "../../common/glm53f_ref.h"
#include "glm53f_expert_kern.h"

/* A model-shared 32 MiB arena holds ONE projection, never an expanded model.
 * Pack once for all four 8-token panels, avoiding repeated BF16/FP8 decode.
 * The largest production projection is 4096x1536 (24 MiB plus row padding).
 * Storage and activations remain at their original precision outside here. */
enum { GLM53F_GEMM_ARENA_FLOATS = 8 * 1024 * 1024 };

static inline float glm53f_prefill_fp8(uint8_t q) {
    if (!(q & 0x78)) {
        float v = (q & 7) * (1.0f / 512.0f);
        return q & 128 ? -v : v;
    }
    if ((q & 127) == 127) return NAN;
    uint32_t bits = ((uint32_t)(q & 128) << 24) |
                    ((uint32_t)(((q >> 3) & 15) + 120) << 23) |
                    ((uint32_t)(q & 7) << 20);
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static inline void glm53f_gemm_8x48(float *out, int stride, const float *x,
        int xstride, const float *packed, int cols, int tokens, int rows) {
#if defined(__ARM_FEATURE_SVE)
    if (svcntw() == 16) {
        svbool_t pg = svptrue_b32();
#define GLM53F_G8_DECL(T) svfloat32_t a##T=svdup_f32(0), b##T=a##T, c##T=a##T
        GLM53F_G8_DECL(0); GLM53F_G8_DECL(1); GLM53F_G8_DECL(2); GLM53F_G8_DECL(3);
        GLM53F_G8_DECL(4); GLM53F_G8_DECL(5); GLM53F_G8_DECL(6); GLM53F_G8_DECL(7);
#undef GLM53F_G8_DECL
        for (int k = 0; k < cols; ++k) {
            svfloat32_t w0=svld1(pg,packed+k*48), w1=svld1(pg,packed+k*48+16), w2=svld1(pg,packed+k*48+32);
#define GLM53F_G8_STEP(T) do { \
            float v = (T) < tokens ? x[(size_t)(T)*xstride+k] : 0; \
            a##T=svmla_n_f32_x(pg,a##T,w0,v); b##T=svmla_n_f32_x(pg,b##T,w1,v); \
            c##T=svmla_n_f32_x(pg,c##T,w2,v); \
        } while(0)
            GLM53F_G8_STEP(0); GLM53F_G8_STEP(1); GLM53F_G8_STEP(2); GLM53F_G8_STEP(3);
            GLM53F_G8_STEP(4); GLM53F_G8_STEP(5); GLM53F_G8_STEP(6); GLM53F_G8_STEP(7);
#undef GLM53F_G8_STEP
        }
#define GLM53F_G8_STORE(T) do { if ((T) < tokens) { \
        svst1(svwhilelt_b32(0,rows),out+(size_t)(T)*stride,a##T); \
        if (rows>16) svst1(svwhilelt_b32(16,rows),out+(size_t)(T)*stride+16,b##T); \
        if (rows>32) svst1(svwhilelt_b32(32,rows),out+(size_t)(T)*stride+32,c##T); \
        } } while(0)
        GLM53F_G8_STORE(0); GLM53F_G8_STORE(1); GLM53F_G8_STORE(2); GLM53F_G8_STORE(3);
        GLM53F_G8_STORE(4); GLM53F_G8_STORE(5); GLM53F_G8_STORE(6); GLM53F_G8_STORE(7);
#undef GLM53F_G8_STORE
        return;
    }
#endif
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
            float sum = 0;
            for (int k = 0; k < cols; ++k) sum += packed[k*48+r] * x[(size_t)t*xstride+k];
            out[(size_t)t*stride+r] = sum;
        }
}

/* Orphaned worksharing: all workers enter together inside one parallel team.
 * N panels own their packed cache lines. FP8 scale lookup uses the ORIGINAL
 * row /128, including 48-column tiles which cross scale-block boundaries. */
static inline void glm53f_prefill_gemm_team(float *out, const void *weight,
        const float *scale, const float *x, int tokens, int rows, int cols,
        int fp8, float *arena) {
    int blocks = (rows + 47) / 48;
#pragma omp for collapse(2) schedule(static)
    for (int b = 0; b < blocks; ++b)
    for (int kb = 0; kb < cols; kb += 128) {
        float *p = arena + (size_t)b * cols * 48;
        for (int r = 0; r < 48; ++r) {
            int row = b * 48 + r;
            int end = cols - kb < 128 ? cols : kb + 128;
#if defined(__ARM_FEATURE_SVE)
            for (int k = kb; k < end; k += (int)svcntw()) {
                svbool_t pg = svwhilelt_b32(k, end);
                svfloat32_t value = svdup_f32(0);
                if (row < rows) {
                    if (fp8) {
                        value = glm53f_fp8_e4m3_bits(pg, (const uint8_t *)weight + (size_t)row*cols, k);
                        value = svmul_n_f32_x(pg, value, scale[(size_t)(row/128)*((cols+127)/128)+k/128]);
                    } else value = svreinterpret_f32_u32(svlsl_n_u32_x(pg,
                        svld1uh_u32(pg, (const uint16_t *)weight + (size_t)row*cols+k),16));
                }
                svst1_scatter_u32offset_f32(pg, p+(size_t)k*48+r, svindex_u32(0,48*sizeof(float)), value);
            }
#else
            for (int k = kb; k < end; ++k) {
                float value = 0;
                if (row < rows) {
                    if (fp8) value = glm53f_prefill_fp8(((const uint8_t *)weight)[(size_t)row*cols+k]) *
                        scale[(size_t)(row/128)*((cols+127)/128)+k/128];
                    else value = glm53f_bf16_to_f32(((const uint16_t *)weight)[(size_t)row*cols+k]);
                }
                p[(size_t)k*48+r] = value;
            }
#endif
        }
    }
#pragma omp for collapse(2) schedule(static)
    for (int b = 0; b < blocks; ++b)
        for (int t = 0; t < tokens; t += 8) {
            int n = tokens-t < 8 ? tokens-t : 8;
            int r = rows-b*48 < 48 ? rows-b*48 : 48;
            /* Constant full-panel specialization removes seven token-tail
             * branches from every K iteration in LLVM's generic expansion. */
            if (n == 8)
                glm53f_gemm_8x48(out+(size_t)t*rows+b*48, rows, x+(size_t)t*cols,
                    cols, arena+(size_t)b*cols*48, cols, 8, r);
            else
                glm53f_gemm_8x48(out+(size_t)t*rows+b*48, rows, x+(size_t)t*cols,
                    cols, arena+(size_t)b*cols*48, cols, n, r);
        }
}
#endif
