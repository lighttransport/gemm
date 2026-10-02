/* Compare host conversion to independent x86 F16C hardware, including ties. */
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include "half_convert.h"

static int check(float value) {
    uint16_t got = cllm_f32_to_f16(value);
    uint16_t want = (uint16_t)_cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    if (isnan(value)) {
        if ((got & 0x7c00u) == 0x7c00u && (got & 0x3ffu)) return 0;
    } else if (got == want) return 0;
    fprintf(stderr, "half conversion %.9g: %04x != %04x\n", value, got, want);
    return 1;
}

int main(void) {
    for (uint32_t i = 0; i < 65536; i++) {
        uint32_t bits = i << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        if (check(value) || cllm_bf16_to_f16((uint16_t)i) != cllm_f32_to_f16(value)) return 1;
        if (check(_cvtsh_ss((uint16_t)i))) return 1;
    }
    for (uint32_t i = 0; i < 0x7bff; i++) {
        float midpoint = (_cvtsh_ss((uint16_t)i) + _cvtsh_ss((uint16_t)(i+1))) * .5f;
        float values[] = {midpoint, nextafterf(midpoint, INFINITY), nextafterf(midpoint, -INFINITY)};
        for (int j = 0; j < 3; j++)
            if (check(values[j]) || check(-values[j])) return 1;
    }
    uint32_t state = 19;
    for (int i = 0; i < 1000000; i++) {
        state = state * 1664525u + 1013904223u;
        float value;
        memcpy(&value, &state, sizeof(value));
        if (check(value)) return 1;
    }
    if (check(65520.f) || check(nextafterf(65520.f, -INFINITY)) || check(-65520.f)) return 1;
    puts("PASS: all BF16/F16 values, half rounding boundaries, 1000000 F32 patterns");
    return 0;
}
