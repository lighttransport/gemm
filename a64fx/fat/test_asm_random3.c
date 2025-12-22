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
    
    // Zero initialize output arrays
    memset(output, 0, dim * sizeof(float));
    memset(ref, 0, dim * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < dim; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        gamma[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.75f;
    }
    
    rmsnorm_f32_ref(input, gamma, ref, dim, eps);
    rmsnorm_f32_asm(input, gamma, output, dim, eps);
    
    // Check for NaN/Inf
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < dim; i++) {
        if (isnan(output[i])) nan_count++;
        if (isinf(output[i])) inf_count++;
    }
    printf("NaN count: %d, Inf count: %d\n", nan_count, inf_count);
    
    // Manual error calculation
    float max_rel = 0.0f;
    float max_abs = 0.0f;
    int worst_rel_idx = -1;
    int worst_abs_idx = -1;
    
    for (size_t i = 0; i < dim; i++) {
        float abs_err = fabsf(output[i] - ref[i]);
        float rel_err = (fabsf(ref[i]) > 1e-10f) ? (abs_err / fabsf(ref[i])) : 0.0f;
        
        if (abs_err > max_abs) {
            max_abs = abs_err;
            worst_abs_idx = i;
        }
        if (rel_err > max_rel) {
            max_rel = rel_err;
            worst_rel_idx = i;
        }
    }
    
    printf("max_rel_err = %.2e at index %d\n", max_rel, worst_rel_idx);
    printf("max_abs_err = %.2e at index %d\n", max_abs, worst_abs_idx);
    
    if (worst_rel_idx >= 0) {
        printf("\nWorst rel at [%d]: ref=%.8f, out=%.8f\n", 
               worst_rel_idx, ref[worst_rel_idx], output[worst_rel_idx]);
    }
    if (worst_abs_idx >= 0) {
        printf("Worst abs at [%d]: ref=%.8f, out=%.8f\n", 
               worst_abs_idx, ref[worst_abs_idx], output[worst_abs_idx]);
    }
    
    free(input);
    free(gamma);
    free(output);
    free(ref);
    return 0;
}
