/* ds4f_kernels_x86.h — portable (non-SVE) implementations of the DS4F small
 * kernel primitives.
 *
 * ds4f_impl.h was written for A64FX and reaches for SVE intrinsics directly in
 * a set of small helpers. This header supplies the same helpers for every other
 * target, so ds4f_impl.h can be built on x86 without forking the model code.
 * It is included by ds4f_impl.h only when __ARM_FEATURE_SVE is absent.
 *
 * These are the DENSE-path helpers (attention scoring, KV axpy, mHC mixing).
 * Under the hetero/ds4f plan that work moves to the GPU, so the bar here is
 * "correct and auto-vectorizable", not "hand-tuned". The kernels that decide
 * decode throughput are the MXFP4 routed-expert matvecs, which get explicit
 * AVX2 treatment in ggml_dequant.h.
 *
 * Numerics: written to match the SVE versions' observable behaviour, including
 * the two places where that is not simply "the obvious C loop":
 *   - ds4f_quant_x_sdot_into uses svcvt_s32_f32_x, which truncates toward zero
 *     rather than rounding to nearest. The C here truncates too.
 *   - the dot/score reductions are horizontal sums over vector lanes, so their
 *     f32 summation order differs from a plain sequential loop either way. The
 *     callers already treat these as reorder-tolerant (see the argmax-safety
 *     notes in ds4f_impl.h).
 */
#ifndef DS4F_KERNELS_X86_H
#define DS4F_KERNELS_X86_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/* bf16 (upper 16 bits of f32) -> f32 */
static inline float ds4f_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

/* ---- dots ---- */
static inline float ds4f_sve_dot_f32(const float *q, const float *k, int n) {
    float acc = 0.f;
    for (int d = 0; d < n; d++) acc += q[d] * k[d];
    return acc;
}

static inline float ds4f_sve_dot_bf16(const float *q, const uint16_t *k, int n) {
    float acc = 0.f;
    for (int d = 0; d < n; d++) acc += q[d] * ds4f_bf16_to_f32(k[d]);
    return acc;
}

static inline float ds4f_sve_dot_i8s(const float *q, const int8_t *k,
                                     const float *sc, int n) {
    float acc = 0.f;
    for (int d = 0; d < n; d++) acc += q[d] * ((float)k[d] * sc[d]);
    return acc;
}

/* int4 KV: nibble d lives in k4[d>>1], low nibble for even d. The 4-bit code is
 * two's complement (0..15 -> -8..7), matching DS4F_I4_SIGN in ds4f_impl.h. It is
 * open-coded here so this header does not depend on that define's position. */
static inline float ds4f_sve_dot_i4s(const float *q, const uint8_t *k4,
                                     const float *sc, int n) {
    float acc = 0.f;
    for (int d = 0; d < n; d++) {
        int nb = (d & 1) ? (k4[d >> 1] >> 4) : (k4[d >> 1] & 0xF);
        acc += q[d] * ((float)(((nb) ^ 8) - 8) * sc[d]);
    }
    return acc;
}

/* ---- axpy ---- */
static inline void ds4f_sve_axpy_f32(float *out, const float *k, float w, int n) {
    for (int d = 0; d < n; d++) out[d] += k[d] * w;
}

static inline void ds4f_sve_axpy_bf16(float *out, const uint16_t *k, float w, int n) {
    for (int d = 0; d < n; d++) out[d] += ds4f_bf16_to_f32(k[d]) * w;
}

static inline void ds4f_sve_axpy_i8s(float *out, const int8_t *k,
                                     const float *sc, float w, int n) {
    for (int d = 0; d < n; d++) out[d] += ((float)k[d] * sc[d]) * w;
}

static inline void ds4f_sve_axpy_i4s(float *out, const uint8_t *k4,
                                     const float *sc, float w, int n) {
    for (int d = 0; d < n; d++) {
        int nb = (d & 1) ? (k4[d >> 1] >> 4) : (k4[d >> 1] & 0xF);
        out[d] += w * ((float)(((nb) ^ 8) - 8) * sc[d]);
    }
}

/* ---- 8-head fused variants: one KV row scattered into 8 output rows ---- */
static inline void ds4f_axpy8_f32(float *out, int os, const float *kv,
                                  const float w[8], int n) {
    for (int h = 0; h < 8; h++) {
        float *o = out + (size_t)h * os;
        float wh = w[h];
        for (int d = 0; d < n; d++) o[d] += kv[d] * wh;
    }
}

static inline void ds4f_axpy8_bf16(float *out, int os, const uint16_t *kv,
                                   const float w[8], int n) {
    for (int h = 0; h < 8; h++) {
        float *o = out + (size_t)h * os;
        float wh = w[h];
        for (int d = 0; d < n; d++) o[d] += ds4f_bf16_to_f32(kv[d]) * wh;
    }
}

/* ---- 8-head fused scoring: one KV row against 8 query rows ---- */
static inline void ds4f_score8_f32(float s[8], const float *q, int qs,
                                   const float *kv, int K) {
    for (int h = 0; h < 8; h++) {
        const float *qh = q + (size_t)h * qs;
        float a = 0.f;
        for (int d = 0; d < K; d++) a += qh[d] * kv[d];
        s[h] = a;
    }
}

static inline void ds4f_score8_bf16(float s[8], const float *q, int qs,
                                    const uint16_t *kv, int K) {
    for (int h = 0; h < 8; h++) {
        const float *qh = q + (size_t)h * qs;
        float a = 0.f;
        for (int d = 0; d < K; d++) a += qh[d] * ds4f_bf16_to_f32(kv[d]);
        s[h] = a;
    }
}

/* ---- int8 activation quantization for the W8A8 dense path ----
 * Per 64-element block: scale = absmax/127, value = trunc(x * 127/absmax),
 * clamped to [-127,127]. Truncation (not round-to-nearest) matches
 * svcvt_s32_f32_x in the SVE original. */
static inline void ds4f_quant_x_sdot_into(const float *x, int K,
                                          int8_t *xq, float *xs) {
    int nb = K / 64;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + (size_t)b * 64;
        float amax = 0.f;
        for (int j = 0; j < 64; j++) { float a = fabsf(xb[j]); if (a > amax) amax = a; }
        float inv = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        xs[b] = amax / 127.0f;
        int8_t *q = xq + (size_t)b * 64;
        for (int j = 0; j < 64; j++) {
            int v = (int)(xb[j] * inv);          /* truncate toward zero */
            if (v > 127) v = 127; else if (v < -127) v = -127;
            q[j] = (int8_t)v;
        }
    }
}

#endif /* DS4F_KERNELS_X86_H */
