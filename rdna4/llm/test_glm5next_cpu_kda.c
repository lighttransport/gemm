#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "../../common/glm5next_cpu_kda.h"

int main(int argc, char **argv) {
    gguf_shards *model;
    glm5next_config config;
    char error[160];
    float *hidden, *out, *dsa_out, *state, *conv, *streams, *full_state, *full_conv;
    int i;
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    model = gguf_open_shards(argv[1], 2);
    if (!model) return 1;
    if (glm5next_config_load(model->metadata, &config, error, sizeof(error)) != 0 ||
        glm5next_validate_tensors(model, &config, error, sizeof(error)) != 0) {
        fprintf(stderr, "contract failure: %s\n", error);
        glm5next_config_free(&config); gguf_close_shards(model); return 1;
    }
    hidden = (float *)malloc((size_t)config.hidden_size * sizeof(float));
    out = (float *)malloc((size_t)config.hidden_size * sizeof(float));
    dsa_out = (float *)malloc((size_t)config.hidden_size * sizeof(float));
    state = (float *)calloc((size_t)config.attention_heads * config.linear_head_dim *
                            config.linear_head_dim, sizeof(float));
    conv = (float *)calloc((size_t)3 * config.attention_heads * config.linear_head_dim *
                           (config.short_conv_kernel - 1), sizeof(float));
    streams = (float *)malloc((size_t)config.hc_count * config.hidden_size * sizeof(float));
    full_state = (float *)calloc((size_t)config.attention_heads * config.linear_head_dim *
                                 config.linear_head_dim, sizeof(float));
    full_conv = (float *)calloc((size_t)3 * config.attention_heads * config.linear_head_dim *
                                (config.short_conv_kernel - 1), sizeof(float));
    if (!hidden || !out || !dsa_out || !state || !conv || !streams || !full_state || !full_conv) return 1;
    for (i = 0; i < config.hidden_size; ++i)
        hidden[i] = 0.01f * sinf((float)(i + 1));
    if (glm5next_cpu_kda_forward(model, 0, &config, hidden, out, state, conv) != 0) {
        fprintf(stderr, "KDA layer execution failed\n"); return 1;
    }
    for (i = 0; i < config.hidden_size; ++i)
        if (!isfinite(out[i])) { fprintf(stderr, "KDA produced non-finite output\n"); return 1; }
    for (i = 0; i < config.hc_count; ++i)
        memcpy(streams + (size_t)i * config.hidden_size, hidden,
               (size_t)config.hidden_size * sizeof(float));
    if (glm5next_cpu_kda_dense_block(model, 0, &config, streams, full_state, full_conv) != 0) {
        fprintf(stderr, "complete KDA+dense block execution failed\n"); return 1;
    }
    for (i = 0; i < config.hc_count * config.hidden_size; ++i)
        if (!isfinite(streams[i])) { fprintf(stderr, "complete block produced non-finite output\n"); return 1; }
    if (glm5next_cpu_dsa_forward(model, 3, &config, hidden, dsa_out) != 0) {
        fprintf(stderr, "DSA layer execution failed\n"); return 1;
    }
    for (i = 0; i < config.hidden_size; ++i)
        if (!isfinite(dsa_out[i])) { fprintf(stderr, "DSA produced non-finite output\n"); return 1; }
    for (i = 0; i < config.hc_count; ++i)
        memcpy(streams + (size_t)i * config.hidden_size, hidden,
               (size_t)config.hidden_size * sizeof(float));
    if (glm5next_cpu_dsa_moe_block(model, 3, &config, streams) != 0) {
        fprintf(stderr, "complete DSA+MoE block execution failed\n"); return 1;
    }
    for (i = 0; i < config.hc_count * config.hidden_size; ++i)
        if (!isfinite(streams[i])) { fprintf(stderr, "complete DSA block produced non-finite output\n"); return 1; }
    {
        double ss = 0.0;
        for (i = 0; i < config.hidden_size; ++i) ss += (double)out[i] * out[i];
        double fs = 0.0;
        for (i = 0; i < config.hidden_size; ++i) fs += (double)streams[i] * streams[i];
        printf("GLM5NEXT_CPU_KDA PASS out_rms=%.7g state0=%.7g block_stream0_rms=%.7g\n",
               sqrt(ss / config.hidden_size), state[0], sqrt(fs / config.hidden_size));
    }
    free(hidden); free(out); free(dsa_out); free(state); free(conv); free(streams); free(full_state); free(full_conv);
    glm5next_config_free(&config); gguf_close_shards(model);
    return 0;
}
