#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#define Q38FN_RUNTIME_IMPLEMENTATION
#include "../common/q38fn_runtime.h"

#include <math.h>
#include <stdio.h>

int main(void)
{
    static const uint16_t bf16[] = {0x0000, 0x3f80, 0xc020, 0x7f80};
    static const float expected[] = {0.0f, 1.0f, -2.5f, INFINITY};

    for (size_t i = 0; i < sizeof(bf16) / sizeof(bf16[0]); ++i) {
        float actual = q38fn_bf16_to_f32(bf16[i]);
        if (actual != expected[i]) {
            fprintf(stderr, "BF16 conversion failed at %zu\n", i);
            return 1;
        }
    }
    if (q38fn_f32_checksum(expected, 4) != UINT64_C(0x976510fbcf756e43)) {
        fprintf(stderr, "checksum changed: %016llx\n",
                (unsigned long long)q38fn_f32_checksum(expected, 4));
        return 1;
    }
    {
        q38fn_delta_state state = {0};
        if (q38fn_delta_state_init(&state) != 0 || !state.conv || !state.recurrent) {
            fprintf(stderr, "delta state allocation failed\n");
            q38fn_delta_state_destroy(&state);
            return 1;
        }
        if (state.conv[Q38FN_LINEAR_CONV_DIM * Q38FN_LINEAR_CONV_KERNEL - 1] != 0.0f ||
            state.recurrent[Q38FN_LINEAR_VALUE_HEADS * Q38FN_LINEAR_HEAD_DIM *
                            Q38FN_LINEAR_HEAD_DIM - 1] != 0.0f) {
            fprintf(stderr, "delta state was not zero initialized\n");
            q38fn_delta_state_destroy(&state);
            return 1;
        }
        q38fn_delta_state_destroy(&state);
    }
    puts("Q38FN_RUNTIME_TEST ok");
    return 0;
}
