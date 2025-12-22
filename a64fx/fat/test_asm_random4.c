#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    
    memset(output, 0, dim * sizeof(float));
    memset(ref, 0, dim * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < dim; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        gamma[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.75f;
    }
    
    rmsnorm_f32_ref(input, gamma, ref, dim, eps);
    rmsnorm_f32_asm(input, gamma, output, dim, eps);
    
    // Manual error calculation with explicit init
    volatile float max_rel = 0.0f;
    volatile float max_abs = 0.0f;
    volatile int worst_rel_idx = -1;
    volatile int worst_abs_idx = -1;
    
    printf("Computing errors...\n");
    for (size_t i = 0; i < dim; i++) {
        float diff = output[i] - ref[i];
        float abs_err = (diff >= 0) ? diff : -diff;
        float ref_abs = (ref[i] >= 0) ? ref[i] : -ref[i];
        float rel_err = (ref_abs > 1e-10f) ? (abs_err / ref_abs) : 0.0f;
        
        if (i < 3) {
            printf("  [%zu] diff=%e abs_err=%e rel_err=%e\n", i, diff, abs_err, rel_err);
        }
        
        if (abs_err > max_abs) {
            max_abs = abs_err;
            worst_abs_idx = i;
        }
        if (rel_err > max_rel) {
            max_rel = rel_err;
            worst_rel_idx = i;
        }
    }
    
    printf("\nResults:\n");
    printf("max_rel_err = %.6e at index %d\n", (double)max_rel, worst_rel_idx);
    printf("max_abs_err = %.6e at index %d\n", (double)max_abs, worst_abs_idx);
    
    free(input);
    free(gamma);
    free(output);
    free(ref);
    return 0;
}
