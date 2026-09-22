#ifndef A64FX_Q8_K128_H
#define A64FX_Q8_K128_H

#include <stddef.h>
#include <stdint.h>

/* One record covers 128 output rows by 128 input columns.  The four Q8_0
 * scale vectors are retained as IEEE FP16, followed by signed bytes in
 * K-major order. */
enum {
    Q8_K128_N = 128,
    Q8_K128_K = 128,
    Q8_K128_SCALE_BYTES = 4 * Q8_K128_N * 2,
    Q8_K128_WEIGHT_BYTES = Q8_K128_K * Q8_K128_N,
    Q8_K128_RECORD_BYTES = Q8_K128_SCALE_BYTES + Q8_K128_WEIGHT_BYTES
};

void q8_k128_f32(const uint8_t *record, const float *x, float *y);
/* A15 records interleave four input bytes per output row. Original Q8 scales
 * are unchanged. Activation quantization alone is lossy; keep opt-in. */
void q8_k128_a15_dot(const uint8_t *record, const int8_t *hi, const int8_t *lo,
                     const float *scales, float *y);
#if defined(__ARM_FEATURE_SVE)
#include <math.h>
#include <arm_sve.h>
typedef struct {
    int8_t hi[128], lo[128];
    float scales[4];
} q8_k128_a15_activation;

static inline void q8_k128_quantize_a15(const float *x, q8_k128_a15_activation *a)
{
    int8_t *hi = a->hi, *lo = a->lo;
    float *scales = a->scales;
    for (int b = 0; b < 4; b++) {
        svbool_t pg = svptrue_b32();
        svfloat32_t maxima = svdup_f32(0.0f);
        for (int k = 0; k < 32; k += (int)svcntw()) {
            svbool_t tail = svwhilelt_b32(k, 32);
            svfloat32_t v = svld1(tail, x + b * 32 + k);
            maxima = svmax_f32_m(tail, maxima, svabs_f32_x(tail, v));
        }
        float amax = svmaxv_f32(pg, maxima);
        float inv = amax > 0.0f ? 16256.0f / amax : 0.0f;
        scales[b] = amax / 16256.0f;
        for (int k = 0; k < 32; k += (int)svcntw()) {
            int j = b * 32 + k;
            svbool_t tail = svwhilelt_b32(k, 32);
            svfloat32_t v = svmul_n_f32_x(tail, svld1(tail, x + j), inv);
            svfloat32_t half = svsel_f32(svcmplt_n_f32(tail, v, 0.0f),
                                         svdup_f32(-0.5f), svdup_f32(0.5f));
            svint32_t q = svcvt_s32_f32_x(tail, svadd_f32_x(tail, v, half));
            q = svmax_n_s32_x(tail, svmin_n_s32_x(tail, q, 16256), -16256);
            svint32_t h = svasr_n_s32_x(tail, svadd_n_s32_x(tail, q, 64), 7);
            svint32_t l = svsub_s32_x(tail, q, svlsl_n_s32_x(tail, h, 7));
            svst1b_s32(tail, hi + j, h);
            svst1b_s32(tail, lo + j, l);
        }
    }
}

/* One collective call per matrix across all nt workers. Double buffering
 * prevents a fast worker's next call from overwriting a slow consumer's data.
 * The arrival barrier publishes disjoint quantization tiles with release/acquire.
 * A workspace belongs to one tensor and one worker pool. */
typedef struct {
    int count, sense;
    uint8_t padding[248];
    q8_k128_a15_activation data[];
} q8_k128_a15_shared;

static inline size_t q8_k128_a15_shared_bytes(int tiles)
{
    return (sizeof(q8_k128_a15_shared) +
            (size_t)2 * tiles * sizeof(q8_k128_a15_activation) + 255) & ~(size_t)255;
}

static inline const q8_k128_a15_activation *q8_k128_quantize_a15_shared(
    q8_k128_a15_shared *cache, const float *x, int tiles, int tid, int nt)
{
    int sense = 1 - __atomic_load_n(&cache->sense, __ATOMIC_ACQUIRE);
    q8_k128_a15_activation *a = cache->data + (size_t)sense * tiles;
    for (int k = tiles * tid / nt; k < tiles * (tid + 1) / nt; k++)
        q8_k128_quantize_a15(x + (size_t)k * Q8_K128_K, &a[k]);
    if (__atomic_add_fetch(&cache->count, 1, __ATOMIC_ACQ_REL) == nt) {
        __atomic_store_n(&cache->count, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&cache->sense, sense, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&cache->sense, __ATOMIC_ACQUIRE) != sense)
            __asm__ __volatile__("yield" ::: "memory");
    }
    return a;
}

static inline void q8_k128_apply(int mode, const uint8_t *record,
                                 const float *x, float *y)
{
    if (mode != 2) { q8_k128_f32(record, x, y); return; }
    q8_k128_a15_activation a;
    q8_k128_quantize_a15(x, &a);
    q8_k128_a15_dot(record, a.hi, a.lo, a.scales, y);
}
#endif
void q8_stream_256_sve(const uint8_t *data, size_t bytes);

#endif
