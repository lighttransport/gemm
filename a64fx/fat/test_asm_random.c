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
    size_t dim = 4096;
    float eps = 1e-5f;
    
    float* input = aligned_alloc(64, dim * sizeof(float));
    float* gamma = aligned_alloc(64, dim * sizeof(float));
    float* output = aligned_alloc(64, dim * sizeof(float));
    float* ref = aligned_alloc(64, dim * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < dim; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        gamma[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.75f;
    }
    
    rmsnorm_f32_ref(input, gamma, ref, dim, eps);
    rmsnorm_f32_asm(input, gamma, output, dim, eps);
    
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    int max_err_idx = 0;
    
    for (size_t i = 0; i < dim; i++) {
        float abs_err = fabsf(output[i] - ref[i]);
        float rel_err = (ref[i] != 0.0f) ? abs_err / fabsf(ref[i]) : abs_err;
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
            max_abs_err = abs_err;
            max_err_idx = i;
        }
    }
    
    printf("Max error at index %d:\n", max_err_idx);
    printf("  ref[%d] = %f\n", max_err_idx, ref[max_err_idx]);
    printf("  out[%d] = %f\n", max_err_idx, output[max_err_idx]);
    printf("  input[%d] = %f\n", max_err_idx, input[max_err_idx]);
    printf("  gamma[%d] = %f\n", max_err_idx, gamma[max_err_idx]);
    printf("  rel_err = %.2e\n", max_rel_err);
    printf("  abs_err = %.2e\n", max_abs_err);
    
    printf("\nFirst 8 comparisons:\n");
    for (int i = 0; i < 8; i++) {
        float abs_err = fabsf(output[i] - ref[i]);
        float rel_err = (ref[i] != 0.0f) ? abs_err / fabsf(ref[i]) : abs_err;
        printf("  [%d] ref=%.6f out=%.6f rel_err=%.2e\n", i, ref[i], output[i], rel_err);
    }
    
    free(input);
    free(gamma);
    free(output);
    free(ref);
    return 0;
}
