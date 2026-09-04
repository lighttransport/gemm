/* Reuse the validated GLM-5.2 A64FX mixed-IQ kernels in the GLM-5.3 graph.
 * Keeping this bridge in one translation unit avoids duplicating the large IQ
 * lookup tables in every consumer of glm53f_expert_kern.h. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GLM5_IMPL
#include "../../common/glm5.h"
#include "glm53f_iq_bridge.h"

int glm53f_iq_type_supported(int type) {
    return type == GLM53F_GGML_IQ2_XS ||
           type == GLM53F_GGML_IQ3_XXS ||
           type == GLM53F_GGML_IQ4_XS;
}

static float iq_row(int type, const uint8_t *row,
                    const glm5_iq_q8_block *xq, int blocks) {
    glm5_tensor tensor = {0};
    tensor.type = (glm5_qtype)type;
    return glm5_iq_q8_row(&tensor, row, xq, blocks);
}

int glm53f_iq_expert_weighted(
        float *output, const glm53f_iq_part *parts, const float *weights,
        int count, const float *input, float *gate_up, float *activation) {
    enum { HIDDEN = 4096, GU_STRIDE = 1024, ACT_STRIDE = 512 };
    glm5_iq_q8_block input_q[HIDDEN / 256];
    glm5_iq_q8_block act_q[9][ACT_STRIDE / 256];
    size_t gate_rb[9], down_rb[9];
    int gate_blocks = HIDDEN / 256;
    if (!output || !parts || !weights || !input || !gate_up || !activation ||
        count < 1 || count > 9) return -1;
    pthread_once(&glm5_iq_lut_once, glm5_iq_init_luts);
    glm5_iq_quant_q8(input_q, input, HIDDEN);
    for (int k = 0; k < count; ++k) {
        if (!glm53f_iq_type_supported(parts[k].gate_type) ||
            !glm53f_iq_type_supported(parts[k].down_type) ||
            parts[k].inter < 256 || parts[k].inter > ACT_STRIDE ||
            parts[k].inter % 256) return -1;
        gate_rb[k] = dequant_row_size((uint32_t)parts[k].gate_type, HIDDEN);
        down_rb[k] = dequant_row_size((uint32_t)parts[k].down_type, parts[k].inter);
    }
#pragma omp parallel
    {
        int total_gate_rows = 0;
        for (int k = 0; k < count; ++k) total_gate_rows += 2 * parts[k].inter;
#pragma omp for schedule(static)
        for (int q = 0; q < total_gate_rows; ++q) {
            int k = 0, r = q;
            while (r >= 2 * parts[k].inter) r -= 2 * parts[k++].inter;
            gate_up[(size_t)k * GU_STRIDE + r] = iq_row(
                parts[k].gate_type, parts[k].gate_up + (size_t)r * gate_rb[k],
                input_q, gate_blocks);
        }
#pragma omp for schedule(static)
        for (int q = 0; q < count * ACT_STRIDE; ++q) {
            int k = q / ACT_STRIDE, i = q - k * ACT_STRIDE;
            if (i < parts[k].inter) {
                float g = gate_up[(size_t)k * GU_STRIDE + i];
                float u = gate_up[(size_t)k * GU_STRIDE + parts[k].inter + i];
                if (g > 10) g = 10; if (g < -100) g = -100;
                if (u > 10) u = 10; if (u < -10) u = -10;
                activation[(size_t)k * ACT_STRIDE + i] =
                    (g / (1.0f + expf(-g))) * u;
            }
        }
#pragma omp single
        for (int k = 0; k < count; ++k)
            glm5_iq_quant_q8(act_q[k], activation + (size_t)k * ACT_STRIDE,
                             parts[k].inter);
#pragma omp for schedule(static)
        for (int r = 0; r < HIDDEN; ++r) {
            float sum = 0.0f;
            for (int k = 0; k < count; ++k)
                sum += weights[k] * iq_row(
                    parts[k].down_type,
                    parts[k].down + (size_t)r * down_rb[k], act_q[k],
                    parts[k].inter / 256);
            output[r] = sum;
        }
    }
    return 0;
}
