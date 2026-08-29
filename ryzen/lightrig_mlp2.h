#ifndef LIGHTTRANSPORT_GEMM_LIGHTRIG_MLP2_H
#define LIGHTTRANSPORT_GEMM_LIGHTRIG_MLP2_H

#include <stddef.h>

/* Row-major two-layer ReLU MLP. Scratch must hold hidden_count floats. */
void lt_mlp2_f32_scalar(const float *input, const float *fc1_weight,
                        const float *fc1_bias, const float *fc2_weight,
                        const float *fc2_bias, float *scratch, float *output,
                        size_t input_count, size_t hidden_count, size_t output_count);
void lt_mlp2_f32(const float *input, const float *fc1_weight,
                 const float *fc1_bias, const float *fc2_weight,
                 const float *fc2_bias, float *scratch, float *output,
                 size_t input_count, size_t hidden_count, size_t output_count);
const char *lt_mlp2_f32_backend(void);
#endif
