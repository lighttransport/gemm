#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly5(const float* input, float* output, size_t count);

int main() {
    size_t count = 16384;
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output = aligned_alloc(64, count * sizeof(float));
    float* ref = aligned_alloc(64, count * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < count; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }
    for (size_t i = 0; i < count; i++) ref[i] = expf(input[i]);
    
    exp_f32_poly5(input, output, count);
    
    float max_rel = 0, max_abs = 0;
    size_t max_rel_idx = 0, max_abs_idx = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max_rel) { max_rel = rel; max_rel_idx = i; }
        if (err > max_abs) { max_abs = err; max_abs_idx = i; }
    }
    
    printf("max_rel=%.2e at [%zu]: in=%.6f out=%.6e ref=%.6e\n", 
           max_rel, max_rel_idx, input[max_rel_idx], output[max_rel_idx], ref[max_rel_idx]);
    printf("max_abs=%.2e at [%zu]: in=%.6f out=%.6e ref=%.6e\n", 
           max_abs, max_abs_idx, input[max_abs_idx], output[max_abs_idx], ref[max_abs_idx]);
    
    // Verify the calculation
    float verif_err = fabsf(output[max_rel_idx] - ref[max_rel_idx]);
    float verif_rel = verif_err / fabsf(ref[max_rel_idx]);
    printf("\nVerification at max_rel_idx: err=%.6e rel=%.6e\n", verif_err, verif_rel);
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
