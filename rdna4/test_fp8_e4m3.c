#include <math.h>
#include <stdio.h>
#include "rocew.h"

#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "hip_runner_common.h"

int main(void) {
    static const float values[] = {
        -448.0f, -32.0f, -1.0f, -0.125f, 0.0f,
        0.125f, 1.0f, 32.0f, 448.0f, 1.0625f, 1.1875f
    };
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        uint8_t enc = hip_f32_to_fp8_e4m3(values[i]);
        float got = hip_fp8_e4m3_to_f32(enc);
        float scale = fmaxf(1.0f, fabsf(values[i]));
        if (!isfinite(got) || fabsf(got - values[i]) / scale > 0.125f) {
            fprintf(stderr, "FP8 roundtrip failed: x=%g enc=0x%02x got=%g\n",
                    values[i], enc, got);
            return 1;
        }
    }
    if (!isnan(hip_fp8_e4m3_to_f32(0x7f))) return 1;
    puts("FP8 E4M3 encode/decode: PASS");
    return 0;
}
