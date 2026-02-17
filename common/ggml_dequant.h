/*
 * ggml_dequant.h - Dequantization routines for GGML Q8_0, Q4_K, and Q6_K formats
 *
 * Usage:
 *   #define GGML_DEQUANT_IMPLEMENTATION
 *   #include "ggml_dequant.h"
 *
 * Dependencies: gguf_loader.h (for ggml_dtype enum)
 *
 * API:
 *   void dequantize_row_q8_0(const void *src, float *dst, int n);
 *   void dequantize_row_q4_K(const void *src, float *dst, int n);
 *   void dequantize_row_q6_K(const void *src, float *dst, int n);
 *   int  dequant_row(uint32_t ggml_type, const void *src, float *dst, int n);
 */
#ifndef GGML_DEQUANT_H
#define GGML_DEQUANT_H

#include <stdint.h>
#include <stddef.h>
#include "gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Q8_0 block: 32 elements, 34 bytes */
typedef struct {
    uint16_t d;       /* block scale (fp16) */
    int8_t   qs[32];  /* quantized values */
} block_q8_0;

/* Q4_K block: 256 elements, 144 bytes */
typedef struct {
    uint16_t d;         /* super-block scale (fp16) */
    uint16_t dmin;      /* super-block min (fp16) */
    uint8_t scales[12]; /* 6-bit scales and mins for 8 sub-blocks, packed */
    uint8_t qs[128];    /* 4-bit quants, 2 per byte */
} block_q4_K;

/* Q6_K block: 256 elements, 210 bytes */
typedef struct {
    uint8_t ql[128];    /* lower 4 bits of quants */
    uint8_t qh[64];     /* upper 2 bits of quants */
    int8_t  scales[16]; /* scales for 16 sub-blocks of 16 */
    uint16_t d;         /* super-block scale (fp16) */
} block_q6_K;

static inline float ggml_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            /* subnormal */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    __builtin_memcpy(&result, &f, 4);
    return result;
}

void dequantize_row_q8_0(const void *src, float *dst, int n);
void dequantize_row_q4_K(const void *src, float *dst, int n);
void dequantize_row_q6_K(const void *src, float *dst, int n);

/* Dequantize a row. Returns 0 on success, -1 if type unsupported. */
int dequant_row(uint32_t ggml_type, const void *src, float *dst, int n);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef GGML_DEQUANT_IMPLEMENTATION

#include <string.h>
#include <math.h>

void dequantize_row_q8_0(const void *src, float *dst, int n) {
    const int nb = n / 32;
    const block_q8_0 *blocks = (const block_q8_0 *)src;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            dst[i * 32 + j] = d * blocks[i].qs[j];
        }
    }
}

void dequantize_row_q4_K(const void *src, float *dst, int n) {
    const int nb = n / 256;
    const block_q4_K *blocks = (const block_q4_K *)src;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        const float d   = ggml_fp16_to_fp32(b->d);
        const float dmin = ggml_fp16_to_fp32(b->dmin);

        /* Decode 6-bit scales and mins from 12 packed bytes.
         * There are 8 sub-blocks of 32 elements each.
         * scales[0..5] contain lower 4 bits: scales[j] & 0x3F for scale, (scales[j] >> 4) for min? No.
         * Actually the packing in llama.cpp is:
         *   For j in 0..7:
         *     sc = scale for sub-block j (6 bits)
         *     m  = min for sub-block j (6 bits)
         *   Packed into 12 bytes as:
         *     bytes 0-3: low 4 bits of sc[0..7] packed as (sc[2j+1]<<4 | sc[2j]) for j=0..3  -- no wait
         *
         * Let me use the llama.cpp approach directly:
         *   For j < 4: sc[j] = scales[j] & 63, then high bits from scales[j+8]
         *   For j >= 4: sc[j] = (scales[j-4] >> 6) | ((scales[j] & 0xF) << 2) ... no
         *
         * The canonical approach from ggml:
         */
        uint8_t sc[8], m_[8];

        /* Lower 4 bits of scales and mins from first 8 bytes */
        for (int j = 0; j < 4; j++) {
            sc[j]     = b->scales[j] & 63;
            sc[j + 4] = b->scales[j + 4] & 63;
            m_[j]     = b->scales[j] >> 6;       /* 2 bits */
            m_[j + 4] = b->scales[j + 4] >> 6;   /* 2 bits */
        }

        /* Upper bits from bytes 8-11 */
        for (int j = 0; j < 4; j++) {
            /* Actually let me re-derive from llama.cpp source.
             * In ggml-quants.c, get_scale_min_k4:
             *   if (j < 4) {
             *     d = scales[j] & 63;  m = scales[j+4] & 63;
             *   } else {
             *     d = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
             *     m = (scales[j+4] >>  4) | ((scales[j-0] >> 6) << 4);
             *   }
             * Wait, that's different. Let me re-check.
             */
            (void)j;
        }

        /* Re-do properly following llama.cpp get_scale_min_k4 */
        uint8_t utmp[8]; /* final scale, min pairs */
        /* For sub-block j (0..7): scale=utmp_sc[j], min=utmp_m[j] */

        /* Inline the get_scale_min_k4 logic */
        float sc_f[8], m_f[8];
        for (int j = 0; j < 8; j++) {
            uint8_t d_val, m_val;
            if (j < 4) {
                d_val = b->scales[j] & 63;
                m_val = b->scales[j + 4] & 63;
            } else {
                d_val = (b->scales[j + 4] & 0xF) | ((b->scales[j - 4] >> 6) << 4);
                m_val = (b->scales[j + 4] >>  4) | ((b->scales[j]     >> 6) << 4);
            }
            sc_f[j] = d * d_val;
            m_f[j]  = dmin * m_val;
        }
        (void)utmp;
        (void)sc;
        (void)m_;

        /* Dequantize: each sub-block j has 32 elements from qs[j*16 .. j*16+15] */
        for (int j = 0; j < 8; j++) {
            const uint8_t *q = b->qs + j * 16;
            float *y = dst + i * 256 + j * 32;
            for (int l = 0; l < 16; l++) {
                y[l]      = sc_f[j] * (q[l] & 0xF) - m_f[j];
                y[l + 16] = sc_f[j] * (q[l] >> 4)  - m_f[j];
            }
        }
    }
}

void dequantize_row_q6_K(const void *src, float *dst, int n) {
    const int nb = n / 256;
    const block_q6_K *blocks = (const block_q6_K *)src;

    for (int i = 0; i < nb; i++) {
        const block_q6_K *b = &blocks[i];
        const float d = ggml_fp16_to_fp32(b->d);
        float *y = dst + i * 256;

        /* 256 elements processed in two halves of 128.
         * Each half uses 64 bytes ql, 32 bytes qh, 8 scale values.
         * Within each half, 32 iterations produce 4 outputs each. */
        const uint8_t *ql = b->ql;
        const uint8_t *qh = b->qh;
        const int8_t  *sc = b->scales;

        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

int dequant_row(uint32_t type, const void *src, float *dst, int n) {
    switch (type) {
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0(src, dst, n);
            return 0;
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K(src, dst, n);
            return 0;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K(src, dst, n);
            return 0;
        case GGML_TYPE_F32:
            memcpy(dst, src, n * sizeof(float));
            return 0;
        case GGML_TYPE_F16: {
            const uint16_t *s = (const uint16_t *)src;
            for (int i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(s[i]);
            return 0;
        }
        default:
            return -1;
    }
}

#endif /* GGML_DEQUANT_IMPLEMENTATION */
#endif /* GGML_DEQUANT_H */
