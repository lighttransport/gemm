#define Q38FN_LOWBIT_IMPLEMENTATION
#include "../common/q38fn_lowbit.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static uint16_t bf16(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

int main(void)
{
    enum { ROWS = 9, COLS = 64 };
    uint16_t source[ROWS * COLS];
    q38fn_q5_block packed[ROWS * COLS / 32];
    float input[COLS], output[ROWS], reference[ROWS];
    for (int i = 0; i < ROWS * COLS; ++i)
        source[i] = bf16(sinf((float)(i * 17 + 3)) * 0.125f);
    for (int i = 0; i < COLS; ++i)
        input[i] = (float)(((i * 1103515245u + 12345u) >> 16) & 1023) /
                   4096.0f - 0.125f;
    if (q38fn_q5_bytes(ROWS, COLS) != sizeof(packed) ||
        q38fn_q5_bytes(ROWS, 31) != 0 ||
        q38fn_q5_quantize_bf16(packed, source, ROWS, COLS) ||
        q38fn_q5_matvec(output, packed, input, ROWS, COLS)) return 1;
    double error2 = 0.0, reference2 = 0.0, dot = 0.0, output2 = 0.0;
    for (int row = 0; row < ROWS; ++row) {
        reference[row] = 0.0f;
        for (int column = 0; column < COLS; ++column)
            reference[row] += q38fn_q5_bf16(source[row * COLS + column]) * input[column];
        double error = output[row] - reference[row];
        error2 += error * error;
        reference2 += (double)reference[row] * reference[row];
        output2 += (double)output[row] * output[row];
        dot += (double)output[row] * reference[row];
    }
    double relative_l2 = sqrt(error2 / reference2);
    double cosine = dot / sqrt(reference2 * output2);
    printf("Q38FN_LOWBIT_TEST rel_l2=%.6f cosine=%.6f\n", relative_l2, cosine);
    return !(relative_l2 < 0.08 && cosine > 0.995);
}
