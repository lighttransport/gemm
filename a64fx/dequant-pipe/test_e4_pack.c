#include "e4_pack.h"
#include "w8.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "E4 pack failed at line %d: %s\n", __LINE__, #c); return 1; } } while (0)

int main(void)
{
    uint8_t raw[32768], packed[36928], back[32768];
    for (size_t j = 0; j < sizeof(raw); ++j) raw[j] = (uint8_t)j;
    CHECK(e4_p9_bytes(8) == 0);
    CHECK(e4_pack_p9(packed, raw, 8) == -1);
    CHECK(e4_pack_p9(NULL, raw, 16) == -1);
    CHECK(e4_unpack_p9(back, NULL, 16) == -1);
    for (int bits = 16; bits <= 32; bits += 16) {
        size_t count = bits == 16 ? 32768 : 16384;
        CHECK(e4_pack_p9(packed, raw, bits) == 0);
        CHECK(e4_unpack_p9(back, packed, bits) == 0);
        CHECK(memcmp(raw, back, count) == 0);
        /* Magic, width, invalid/mismatched NaN flag, reserved header and
         * unassigned magnitude code must all be rejected. */
        const int offsets[] = {0, 4, 8, 12, 63, 64};
        for (size_t i = 0; i < sizeof(offsets)/sizeof(offsets[0]); ++i) {
            int off = offsets[i];
            uint8_t saved = packed[off];
            packed[off] = off == 8 ? 2 : (uint8_t)(saved ^ 1);
            CHECK(e4_unpack_p9(back, packed, bits) == -1);
            packed[off] = saved;
        }
        packed[8] = 0;
        CHECK(e4_unpack_p9(back, packed, bits) == -1);
        packed[8] = 1;
        CHECK(e4_unpack_p9(back, packed, bits == 16 ? 32 : 16) == -1);
    }
    for (size_t j = 0; j < sizeof(raw); ++j)
        raw[j] = (uint8_t)(((j * 17) % 127) | ((j & 1) << 7));
    CHECK(e4_pack_p9(packed, raw, 32) == 0);
    const float cases[] = {0, -0.0f, 0x1p-149f, -0x1p-149f, 0x1p-126f,
        0x1.fffffep15f, -0x1.fffffep15f, 0x1p16f, -0x1p16f,
        FLT_MAX, -FLT_MAX, INFINITY, -INFINITY, NAN, 0x1.234568p-120f};
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    CHECK(fpcr == 0);
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        float a[128], got[128], ref[128];
        e4_p9_activation p;
        for (int k = 0; k < 128; ++k) a[k] = (k & 1) ? -cases[i] : cases[i];
        e4_p9_prepare(&p, a);
        CHECK(p.safe == (unsigned)(isfinite(cases[i]) && fabsf(cases[i]) <= 0x1.fffffep15f));
        CHECK(memcmp(p.original, a, sizeof(a)) == 0);
        w8_e4m3_f32(raw, a, ref);
        w8_e4m3_p9_f32(packed, &p, got);
        for (int n = 0; n < 128; ++n)
            CHECK((isnan(got[n]) && isnan(ref[n])) || memcmp(got+n, ref+n, 4) == 0);
    }
    /* Nondefault FPCR rejects rescaling. Restore FPCR before assertions. */
    float a[128], got[128], ref[128];
    e4_p9_activation p;
    for (int k = 0; k < 128; ++k) a[k] = (k % 7 - 3) * 0x1p-149f;
    uint64_t fz = fpcr | (1ul << 24);
    __asm__ volatile("msr fpcr, %0; isb" :: "r"(fz) : "memory");
    e4_p9_prepare(&p, a);
    w8_e4m3_f32(raw, a, ref);
    w8_e4m3_p9_f32(packed, &p, got);
    __asm__ volatile("msr fpcr, %0; isb" :: "r"(fpcr) : "memory");
    CHECK(p.safe == 0);
    CHECK(memcmp(got, ref, sizeof(got)) == 0);
    puts("E4 packed format: PASS (roundtrip, invalid records, activation limits, subnormals, FPCR fallback)");
    return 0;
}
