#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);
extern void exp_f32_poly2(const float* input, float* output, size_t count);
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
    
    printf("Testing in sequence like benchmark:\n\n");
    
    // poly1
    exp_f32_poly1(input, output, count);
    float max1 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max1) max1 = rel;
    }
    printf("poly1: max_rel = %.2e\n", max1);
    printf("  sample: input[0]=%.6f output[0]=%.6e ref[0]=%.6e\n", 
           input[0], output[0], ref[0]);
    
    // poly2
    exp_f32_poly2(input, output, count);
    float max2 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max2) max2 = rel;
    }
    printf("poly2: max_rel = %.2e\n", max2);
    printf("  sample: input[0]=%.6f output[0]=%.6e ref[0]=%.6e\n", 
           input[0], output[0], ref[0]);
    
    // poly5
    exp_f32_poly5(input, output, count);
    float max5 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max5) max5 = rel;
    }
    printf("poly5: max_rel = %.2e\n", max5);
    printf("  sample: input[0]=%.6f output[0]=%.6e ref[0]=%.6e\n", 
           input[0], output[0], ref[0]);
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
