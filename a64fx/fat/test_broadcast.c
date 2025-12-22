// test_broadcast.c - Debug broadcast issue in ASM

#include <stdio.h>
#include <stdlib.h>
#include <arm_sve.h>

extern void rmsnorm_f32_asm(const float* input, const float* gamma,
                            float* output, size_t dim, float eps);

int main() {
    size_t dim = 64;  // Small for debugging
    float eps = 1e-5f;

    float* input = aligned_alloc(64, dim * sizeof(float));
    float* gamma = aligned_alloc(64, dim * sizeof(float));
    float* output = aligned_alloc(64, dim * sizeof(float));

    // Simple test: all 1s
    for (size_t i = 0; i < dim; i++) {
        input[i] = 1.0f;
        gamma[i] = 1.0f;
    }

    // Reference calculation:
    // sum_sq = dim * 1.0 = 64
    // mean_sq = 64 / 64 = 1.0
    // inv_std = 1 / sqrt(1.0 + 1e-5) ≈ 0.99999
    // output = 1.0 * 1.0 * 0.99999 ≈ 1.0
    printf("Expected output ~= 1.0\n\n");

    rmsnorm_f32_asm(input, gamma, output, dim, eps);

    printf("Actual outputs:\n");
    for (size_t i = 0; i < 16; i++) {
        printf("  output[%zu] = %f\n", i, output[i]);
    }
    printf("  ...\n");
    printf("  output[%zu] = %f\n", dim-1, output[dim-1]);

    // Now test with SVE intrinsics reference
    printf("\nWith intrinsics:\n");
    extern void rmsnorm_f32_intrin(const float*, const float*, float*, size_t, float);
    float* output2 = aligned_alloc(64, dim * sizeof(float));
    rmsnorm_f32_intrin(input, gamma, output2, dim, eps);
    for (size_t i = 0; i < 16; i++) {
        printf("  output[%zu] = %f\n", i, output2[i]);
    }

    free(input);
    free(gamma);
    free(output);
    free(output2);

    return 0;
}
