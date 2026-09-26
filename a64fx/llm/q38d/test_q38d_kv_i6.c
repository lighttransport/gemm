#include "q38d_kv_i6.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    enum { DIM = 256, ROWS = 97, BYTES = 192 };
    assert(q38d_kv_i6_row_bytes(DIM) == BYTES);
    float src[DIM], unpacked[DIM], scale;
    uint8_t packed[BYTES];
    float worst = 0.0f;
    for (int r = 0; r < ROWS; r++) {
        for (int d = 0; d < DIM; d++)
            src[d] = 0.8f * sinf((float)(r * DIM + d) * 0.013f) +
                     0.2f * cosf((float)(r + d) * 0.071f);
        assert(q38d_kv_i6_pack_row(src, packed, DIM, &scale) == 0);
        q38d_kv_i6_unpack_f32_256(packed, unpacked);
        for (int d = 0; d < DIM; d++) {
            assert(unpacked[d] == (float)q38d_kv_i6_get(packed, DIM, (size_t)d));
            float got = (float)q38d_kv_i6_get(packed, DIM, (size_t)d) * scale;
            float error = fabsf(got - src[d]);
            if (error > worst) worst = error;
            assert(error <= scale * 0.501f);
        }
    }
    src[0] = INFINITY;
    assert(q38d_kv_i6_pack_row(src, packed, DIM, &scale) == -1);
    printf("PASS: q38d KV INT6 rows=%d dim=%d bytes=%d max_abs=%.6g\n", ROWS, DIM, BYTES, worst);
    return 0;
}
