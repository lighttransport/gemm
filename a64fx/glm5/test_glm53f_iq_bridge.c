/* Focused coverage for the llama.cpp-compatible Q8_0 matvec contract. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm53f_iq_bridge.h"
#include "../../common/ggml_dequant.h"

enum { ROWS = 5, COLUMNS = 256, BLOCKS = COLUMNS / 32 };

static void quantize_reference(block_q8_0 *q, const float *x) {
    for (int b = 0; b < BLOCKS; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            float a = fabsf(x[32 * b + j]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        q[b].d = ggml_fp32_to_fp16(d);
        for (int j = 0; j < 32; ++j) {
            long v = lrintf(x[32 * b + j] * id);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            q[b].qs[j] = (int8_t)v;
        }
    }
}

static float dot_reference(const block_q8_0 *weight,
                           const block_q8_0 *input) {
    float sum = 0.0f;
    for (int b = 0; b < BLOCKS; ++b) {
        int dot = 0;
        for (int j = 0; j < 32; ++j)
            dot += (int)weight[b].qs[j] * (int)input[b].qs[j];
        sum += dot * ggml_fp16_to_fp32(weight[b].d) *
                     ggml_fp16_to_fp32(input[b].d);
    }
    return sum;
}

static void fill_input(float *input, int mode) {
    uint32_t state = 0x12345678u + (uint32_t)mode;
    for (int i = 0; i < COLUMNS; ++i) {
        state = state * 1664525u + 1013904223u;
        float random = ((int)(state >> 8) % 20001 - 10000) * 0.0001f;
        if (mode == 0) input[i] = 0.0f;
        else if (mode == 1) input[i] = (i & 1) ? -0.75f : 0.75f;
        else if (mode == 2) input[i] = random;
        else input[i] = i % 32 == 7 ? (i & 32 ? -31.0f : 29.0f) : random * 0.01f;
    }
}

int main(void) {
    const size_t row_bytes = BLOCKS * sizeof(block_q8_0);
    block_q8_0 *weights = calloc(ROWS, row_bytes);
    block_q8_0 input_q[BLOCKS];
    float input[COLUMNS], actual[ROWS], expected[ROWS];
    int failed = weights == NULL;
    if (!weights) return 2;

    for (int r = 0; r < ROWS; ++r)
        for (int b = 0; b < BLOCKS; ++b) {
            block_q8_0 *w = (block_q8_0 *)((uint8_t *)weights +
                                           (size_t)r * row_bytes) + b;
            float scale = 0.0025f * (float)(1 + r + 2 * b);
            w->d = ggml_fp32_to_fp16(scale);
            for (int j = 0; j < 32; ++j)
                w->qs[j] = (int8_t)(((r * 53 + b * 29 + j * 17) % 255) - 127);
        }

    for (int mode = 0; mode < 4; ++mode) {
        fill_input(input, mode);
        quantize_reference(input_q, input);
        for (int r = 0; r < ROWS; ++r)
            expected[r] = dot_reference(
                (const block_q8_0 *)((const uint8_t *)weights +
                                     (size_t)r * row_bytes), input_q);
        if (glm53f_iq_matvec(actual, (const uint8_t *)weights,
                             GLM53F_GGML_Q8_0, ROWS, COLUMNS, input)) {
            fprintf(stderr, "Q8_0 bridge rejected valid matrix\n");
            failed = 1;
            continue;
        }
        double err2 = 0.0, ref2 = 0.0;
        float max_abs = 0.0f;
        for (int r = 0; r < ROWS; ++r) {
            float d = actual[r] - expected[r];
            err2 += (double)d * d;
            ref2 += (double)expected[r] * expected[r];
            if (fabsf(d) > max_abs) max_abs = fabsf(d);
        }
        double rel = ref2 != 0.0 ? sqrt(err2 / ref2) : sqrt(err2);
        int ok = rel <= 2e-6 && max_abs <= 2e-4f;
        printf("GLM53F_IQ_Q8_0 mode=%d rel_l2=%.9g max_abs=%.9g %s\n",
               mode, rel, max_abs, ok ? "PASS" : "FAIL");
        failed |= !ok;
    }

    if (glm53f_iq_matvec(actual, (const uint8_t *)weights,
                         GLM53F_GGML_Q8_0, ROWS, COLUMNS - 1, input) == 0) {
        fprintf(stderr, "Q8_0 bridge accepted an incomplete block\n");
        failed = 1;
    }
    free(weights);
    printf("SENTINEL glm53f_iq_bridge_q8_0=%s\n", failed ? "FAIL" : "PASS");
    return failed;
}
