// test_qwen3_ffn.c - Test Qwen3-style SwiGLU FFN
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "ffn_qwen3.h"

int main() {
    printf("================================================================\n");
    printf("Qwen3-style SwiGLU FFN Test\n");
    printf("================================================================\n\n");

    // Test D=256 configuration
    printf("Testing D=256 configuration...\n");
    const qwen3_ffn_config* config_256 = qwen3_get_config(QWEN3_CONFIG_256);
    printf("Config: %s\n", config_256->name);
    printf("D_model=%d, D_ff=%d, expansion=%.2fx\n\n",
           config_256->D_model, config_256->D_ff, config_256->expansion);

    const int M = 6;  // Batch size (must be divisible by 6)
    const int D_256 = 256;
    const int D_ff_256 = 1024;

    // Allocate and initialize
    int8_t* input_256 = aligned_alloc(64, M * D_256);
    int8_t* W_gate_256 = aligned_alloc(64, D_256 * D_ff_256);
    int8_t* W_up_256 = aligned_alloc(64, D_256 * D_ff_256);
    int8_t* W_down_256 = aligned_alloc(64, D_ff_256 * D_256);
    int32_t* output_256 = aligned_alloc(64, M * D_256 * sizeof(int32_t));

    // Simple initialization: all 1s
    for (int i = 0; i < M * D_256; i++) input_256[i] = 1;
    for (int i = 0; i < D_256 * D_ff_256; i++) W_gate_256[i] = 1;
    for (int i = 0; i < D_256 * D_ff_256; i++) W_up_256[i] = 1;
    for (int i = 0; i < D_ff_256 * D_256; i++) W_down_256[i] = 1;

    printf("Running D=256 SwiGLU FFN...\n");
    qwen3_ffn_forward_d256(input_256, W_gate_256, W_up_256, W_down_256, output_256, M);

    printf("Checking D=256 output...\n");
    // With all 1s: gate = D*1 = 256, up = 256, SiLU(256)*256, then sum D_ff times
    printf("Sample outputs: output[0]=%d, output[100]=%d, output[255]=%d\n",
           output_256[0], output_256[100], output_256[255]);
    printf("D=256 test completed.\n\n");

    free(input_256);
    free(W_gate_256);
    free(W_up_256);
    free(W_down_256);
    free(output_256);

    // Test D=512 configuration
    printf("Testing D=512 configuration...\n");
    const qwen3_ffn_config* config_512 = qwen3_get_config(QWEN3_CONFIG_512);
    printf("Config: %s\n", config_512->name);
    printf("D_model=%d, D_ff=%d, expansion=%.2fx\n\n",
           config_512->D_model, config_512->D_ff, config_512->expansion);

    const int D_512 = 512;
    const int D_ff_512 = 2048;

    int8_t* input_512 = aligned_alloc(64, M * D_512);
    int8_t* W_gate_512 = aligned_alloc(64, D_512 * D_ff_512);
    int8_t* W_up_512 = aligned_alloc(64, D_512 * D_ff_512);
    int8_t* W_down_512 = aligned_alloc(64, D_ff_512 * D_512);
    int32_t* output_512 = aligned_alloc(64, M * D_512 * sizeof(int32_t));

    for (int i = 0; i < M * D_512; i++) input_512[i] = 1;
    for (int i = 0; i < D_512 * D_ff_512; i++) W_gate_512[i] = 1;
    for (int i = 0; i < D_512 * D_ff_512; i++) W_up_512[i] = 1;
    for (int i = 0; i < D_ff_512 * D_512; i++) W_down_512[i] = 1;

    printf("Running D=512 SwiGLU FFN...\n");
    qwen3_ffn_forward_d512(input_512, W_gate_512, W_up_512, W_down_512, output_512, M);

    printf("Checking D=512 output...\n");
    printf("Sample outputs: output[0]=%d, output[256]=%d, output[511]=%d\n",
           output_512[0], output_512[256], output_512[511]);
    printf("D=512 test completed.\n\n");

    free(input_512);
    free(W_gate_512);
    free(W_up_512);
    free(W_down_512);
    free(output_512);

    printf("================================================================\n");
    printf("All Qwen3 configurations:\n");
    for (int i = 0; i < 5; i++) {
        const qwen3_ffn_config* cfg = qwen3_get_config((qwen3_config_t)i);
        printf("  %s: D=%d, D_ff=%d, %.2fx expansion\n",
               cfg->name, cfg->D_model, cfg->D_ff, cfg->expansion);
    }
    printf("================================================================\n");

    return 0;
}
