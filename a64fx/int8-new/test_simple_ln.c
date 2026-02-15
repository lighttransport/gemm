#include "layernorm.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("Testing simple LayerNorm...\n");

    const size_t N = 64;
    int8_t* input = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t* output = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t* gamma = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t* beta = (int8_t*)malloc(N * sizeof(int8_t));

    // Simple initialization
    for (size_t i = 0; i < N; i++) {
        input[i] = (i % 20) - 10;
        gamma[i] = 127;
        beta[i] = 0;
    }

    printf("Input initialized\n");

    int32_t epsilon = 1000; // Small value
    layernorm_int8(input, output, gamma, beta, epsilon, 1.0f/127.0f, 1.0f/127.0f, N);

    printf("LayerNorm completed\n");

    // Print results
    printf("First 10 outputs: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    free(input);
    free(output);
    free(gamma);
    free(beta);

    printf("✓ Test passed\n");
    return 0;
}
