#include "glm53f_mla_cache_f16.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
int main(void) {
    enum { COUNT = 515 };
    float input[COUNT], expected[COUNT], actual[COUNT];
    uint16_t packed[COUNT + 16];
    uint32_t seed = 43;
    for (int i = 0; i < COUNT; ++i) {
        seed = seed * 1664525u + 1013904223u;
        input[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f;
    }
    const uint32_t edge[] = {0,0x80000000,0x33000000,0xb3000000,0x38800000,
        0x477fe000,0x47800000,0x7f800000,0xff800000,0x7fc00000};
    for (unsigned i = 0; i < sizeof(edge)/sizeof(edge[0]); ++i) memcpy(input + i, edge + i, 4);
    int bad = 0;
    for (int count = 1; count <= COUNT; count += count < 20 ? 1 : 37) {
        for (int i = 0; i < COUNT + 16; ++i) packed[i] = 0x1234;
        glm53f_mla_cache_f16_store(packed, input, count);
        for (int i = 0; i < count; i += (int)svcntw()) {
            svbool_t p = svwhilelt_b32(i, count);
            svfloat32_t f = svld1_f32(p, input + i);
            svst1_f32(p, expected + i, svcvt_f32_f16_x(p, svcvt_f16_f32_x(p, f)));
            svst1_f32(p, actual + i, glm53f_mla_cache_f16_load(p, packed + i));
        }
        bad |= memcmp(expected, actual, count * sizeof(float)) != 0;
        for (int i = count; i < COUNT + 16; ++i) bad |= packed[i] != 0x1234;
    }
    printf("MLA_CACHE_F16 rounded_view_and_bounds %s\n", bad ? "FAIL" : "PASS");
    return bad;
}
