#include "e4_pack.h"
#include "w8.h"
#include <math.h>
#include <string.h>

_Static_assert(offsetof(e4_p9_activation, original) == 512, "assembly activation ABI");
_Static_assert(offsetof(e4_p9_activation, safe) == 1024, "assembly guard ABI");

static uint16_t half_magnitude(unsigned code)
{
    static const uint16_t sub[8] = {0, 0x1800, 0x1c00, 0x1e00, 0x2000, 0x2100, 0x2200, 0x2300};
    code &= 127;
    return code < 8 ? sub[code] : code == 127 ? 0x7e00 : (uint16_t)((code << 7) + 0x2000);
}
size_t e4_p9_bytes(int bits)
{
    return bits == 16 ? 36928 : bits == 32 ? 18496 : 0;
}
int e4_pack_p9(uint8_t *dst, const uint8_t *src, int bits)
{
    size_t bytes = e4_p9_bytes(bits);
    if (!dst || !src || !bytes) return -1;
    const int n = bits == 16 ? 256 : 128;
    uint32_t header[16] = {0x31503945u, (uint32_t)bits, 0};
    memset(dst, 0, bytes);
    for (int k = 0; k < 128; ++k) {
        uint8_t *row = dst + 64 + (size_t)k * (n + n / 8);
        for (int j = 0; j < n; ++j) {
            unsigned q = src[k * n + j];
            row[j] = (uint8_t)(half_magnitude(q) >> 7);
            unsigned lanes = bits == 16 ? 32 : 16;
            unsigned bit = (unsigned)j / 64 * 64 + (unsigned)j % lanes * (64 / lanes) + (unsigned)j % 64 / lanes;
            row[n + bit / 8] |= (uint8_t)((q >> 7) << (bit % 8));
            header[2] |= (q & 127) == 127;
        }
    }
    memcpy(dst, header, sizeof(header));
    return 0;
}
int e4_unpack_p9(uint8_t *dst, const uint8_t *src, int bits)
{
    uint32_t header[16];
    if (!dst || !src || !e4_p9_bytes(bits)) return -1;
    memcpy(header, src, sizeof(header));
    if (header[0] != 0x31503945u || header[1] != (unsigned)bits || header[2] > 1) return -1;
    for (int i = 3; i < 16; ++i) if (header[i]) return -1;
    uint8_t inverse[256] = {0}, valid[256] = {0};
    for (int q = 0; q < 128; ++q) {
        unsigned i = half_magnitude(q) >> 7;
        inverse[i] = (uint8_t)q;
        valid[i] = 1;
    }
    const int n = bits == 16 ? 256 : 128;
    unsigned has_nan = 0;
    for (int k = 0; k < 128; ++k) {
        const uint8_t *row = src + 64 + (size_t)k * (n + n / 8);
        for (int j = 0; j < n; ++j) {
            if (!valid[row[j]]) return -1;
            unsigned q = inverse[row[j]];
            has_nan |= q == 127;
            unsigned lanes = bits == 16 ? 32 : 16;
            unsigned bit = (unsigned)j / 64 * 64 + (unsigned)j % lanes * (64 / lanes) + (unsigned)j % 64 / lanes;
            dst[k * n + j] = (uint8_t)(q | (((row[n + bit / 8] >> (bit % 8)) & 1) << 7));
        }
    }
    return has_nan == header[2] ? 0 : -1;
}
void e4_p9_prepare(e4_p9_activation *dst, const float *src)
{
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    dst->safe = fpcr == 0;
    for (int k = 0; k < 128; ++k) {
        float a = src[k];
        dst->original[k] = a;
        /* E4 magnitudes interpreted as FP32 bits are w * 2^-112.
         * Scaling a by 2^112 is exact whenever the result is finite;
         * even the smallest FP32 subnormal becomes a normal number. */
        if (!isfinite(a) || fabsf(a) > 0x1.fffffep15f) {
            dst->safe = 0;
            dst->scaled[k] = 0;
        } else dst->scaled[k] = a * 0x1p112f;
    }
}
void e4_p9_f32_fallback(const uint8_t *src, const e4_p9_activation *a, float *out)
{
    uint8_t raw[128 * 128] __attribute__((aligned(256)));
    if (e4_unpack_p9(raw, src, 32)) {
        for (int j = 0; j < 128; ++j) out[j] = NAN;
        return;
    }
    w8_e4m3_f32(raw, a->original, out);
}
