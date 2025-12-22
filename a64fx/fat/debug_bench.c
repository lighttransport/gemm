#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);
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
    
    // Compute reference
    for (size_t i = 0; i < count; i++) ref[i] = expf(input[i]);
    
    // Test poly1
    exp_f32_poly1(input, output, count);
    float max_rel1 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max_rel1) max_rel1 = rel;
    }
    printf("poly1 max_rel = %.2e\n", max_rel1);
    
    // Test poly5
    exp_f32_poly5(input, output, count);
    float max_rel5 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max_rel5) max_rel5 = rel;
    }
    printf("poly5 max_rel = %.2e\n", max_rel5);
    
    // Check first few
    printf("\nFirst 8 values (poly5):\n");
    for (int i = 0; i < 8; i++) {
        printf("  output[%d] = %.6e, ref[%d] = %.6e\n", i, output[i], i, ref[i]);
    }
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
