#ifndef E4_PACK_H
#define E4_PACK_H
#include <stddef.h>
#include <stdint.h>
/* P9 version 1: 64-byte header followed by K=128 records, each containing
 * N magnitude bytes (FP16 magnitude bits >>7), then N/8 sign bytes.
 * N=256/128 and L=32/16 for FP16/FP32. Column j's sign is bit
 * (j/64)*64 + (j%L)*(64/L) + (j%64)/L, numbered LSB-first in the sign plane.
 * Header little-endian u32[0]=0x31503945, [1]=bits,
 * [2]=has_nan, remaining bytes zero. This is 9 bits/weight plus the header.
 * NaNs use magnitude 252 (quiet half NaN). No numeric values are discarded.
 * Caller supplies complete, non-overlapping input/output buffers. Kernels
 * require trusted packer output (or a successfully unpack-validated record),
 * SVE512, K=128, and the fixed N above; these are not bounded file parsers. */
size_t e4_p9_bytes(int bits);
int e4_pack_p9(uint8_t *dst, const uint8_t *src, int bits);
int e4_unpack_p9(uint8_t *dst, const uint8_t *src, int bits);
/* Both arrays remain available for the exact fallback. Prepared activations
 * are reusable across output tiles; the benchmark reports preparation time.
 * Keep FPCR unchanged between preparation and kernel calls. The fast FP32
 * path requires FPCR=0, finite |a| <= 0x1.fffffep15, and no weight NaNs;
 * otherwise it unpacks and uses native sequential FP32 FMA. Numerical output
 * is exact (NaN classification only); FPSR exception flags are not promised. */
typedef struct {
    float scaled[128];
    float original[128];
    uint32_t safe;
} e4_p9_activation;
void e4_p9_prepare(e4_p9_activation *, const float *);
void w8_e4m3_p9_f16(const uint8_t *, const _Float16 *, _Float16 *);
void w8_e4m3_p9_f32(const uint8_t *, const e4_p9_activation *, float *);
void e4_p9_f32_fallback(const uint8_t *, const e4_p9_activation *, float *);
/* Exactly eight bits/weight: replace each native byte q by (q+1) mod 256,
 * retaining the K-major K=128, N=256 layout. */
void w8_e4m3_bias_f16(const uint8_t *, const _Float16 *, _Float16 *);
#endif
