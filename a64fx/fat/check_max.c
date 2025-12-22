#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly2(const float* input, float* output, size_t count);

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
    
    exp_f32_poly2(input, output, count);
    
    // Check first 20 values
    printf("First 20 error values:\n");
    for (size_t i = 0; i < 20; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        printf("  [%2zu] out=%.4e ref=%.4e err=%.2e rel=%.4e\n",
               i, output[i], ref[i], err, rel);
    }
    
    // Now find max manually with explicit check
    float max_rel = 0;
    size_t max_idx = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max_rel) {
            max_rel = rel;
            max_idx = i;
            printf("New max at [%zu]: rel=%.6e\n", i, rel);
        }
    }
    printf("\nFinal max: idx=%zu rel=%.6e\n", max_idx, max_rel);
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
