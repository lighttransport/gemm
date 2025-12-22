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
    
    float max_rel = 0.0f;
    float max_abs = 0.0f;
    int worst_idx = -1;
    
    for (size_t i = 0; i < dim; i++) {
        float abs_err = fabsf(output[i] - ref[i]);
        float denom = fabsf(ref[i]);
        float rel_err = (denom > 1e-10f) ? (abs_err / denom) : 0.0f;
        
        if (abs_err > max_abs) {
            max_abs = abs_err;
        }
        if (rel_err > max_rel) {
            max_rel = rel_err;
            worst_idx = i;
        }
    }
    
    printf("Summary:\n");
    printf("  max_rel_err = %.2e at index %d\n", max_rel, worst_idx);
    printf("  max_abs_err = %.2e\n", max_abs);
    
    if (worst_idx >= 0) {
        printf("\nWorst case at index %d:\n", worst_idx);
        printf("  ref = %.10f\n", ref[worst_idx]);
        printf("  out = %.10f\n", output[worst_idx]);
        printf("  input = %.10f\n", input[worst_idx]);
        printf("  gamma = %.10f\n", gamma[worst_idx]);
        float diff = output[worst_idx] - ref[worst_idx];
        printf("  diff = %.10e\n", diff);
    }
    
    printf("\nSample outputs:\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%d] ref=%.8f out=%.8f diff=%.2e\n", 
               i, ref[i], output[i], output[i] - ref[i]);
    }
    
    free(input);
    free(gamma);
    free(output);
    free(ref);
    return 0;
}
