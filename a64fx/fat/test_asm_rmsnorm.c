#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void rmsnorm_f32_asm(const float* input, const float* gamma,
                            float* output, size_t dim, float eps);

static void rmsnorm_f32_ref(const float* input, const float* gamma,
                            float* output, size_t dim, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum_sq += input[i] * input[i];
    }
    float mean_sq = sum_sq / (float)dim;
    float inv_std = 1.0f / sqrtf(mean_sq + eps);
    for (size_t i = 0; i < dim; i++) {
        output[i] = input[i] * gamma[i] * inv_std;
    }
}

int main() {
    size_t dim = 64;
    float eps = 1e-5f;
    
    float* input = aligned_alloc(64, dim * sizeof(float));
    float* gamma = aligned_alloc(64, dim * sizeof(float));
    float* output = aligned_alloc(64, dim * sizeof(float));
    float* ref = aligned_alloc(64, dim * sizeof(float));
    
    // All 1s for easy verification
    for (size_t i = 0; i < dim; i++) {
        input[i] = 1.0f;
        gamma[i] = 1.0f;
    }
    
    rmsnorm_f32_ref(input, gamma, ref, dim, eps);
    rmsnorm_f32_asm(input, gamma, output, dim, eps);
    
    printf("With all inputs=1.0, gamma=1.0:\n");
    printf("Reference output[0] = %f (expected ~1.0)\n", ref[0]);
    printf("ASM output[0]       = %f\n", output[0]);
    printf("Error: %f\n", fabsf(output[0] - ref[0]));
    
    printf("\nFirst 8 ASM outputs:\n");
    for (int i = 0; i < 8; i++) {
        printf("  output[%d] = %f (ref=%f)\n", i, output[i], ref[i]);
    }
    
    free(input);
    free(gamma);
    free(output);
    free(ref);
    return 0;
}
