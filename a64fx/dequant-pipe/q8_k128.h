#ifndef A64FX_Q8_K128_H
#define A64FX_Q8_K128_H

#include <stddef.h>
#include <stdint.h>

/* One record covers 64 output rows by 128 input columns.  The four Q8_0
 * scale vectors are retained as IEEE FP16, followed by signed bytes in
 * K-major order. */
enum {
    Q8_K128_N = 64,
    Q8_K128_K = 128,
    Q8_K128_SCALE_BYTES = 4 * Q8_K128_N * 2,
    Q8_K128_WEIGHT_BYTES = Q8_K128_K * Q8_K128_N,
    Q8_K128_RECORD_BYTES = Q8_K128_SCALE_BYTES + Q8_K128_WEIGHT_BYTES
};

void q8_k128_f32(const uint8_t *record, const float *x, float *y);
void q8_stream_256_sve(const uint8_t *data, size_t bytes);

#endif
