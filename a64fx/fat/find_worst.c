#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);
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
    
    // poly1
    exp_f32_poly1(input, output, count);
    float max1 = 0;
    size_t idx1 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max1) { max1 = rel; idx1 = i; }
    }
    printf("poly1 worst at [%zu]: input=%.6f out=%.6e ref=%.6e rel=%.4f\n",
           idx1, input[idx1], output[idx1], ref[idx1], max1);
    
    // poly2
    exp_f32_poly2(input, output, count);
    float max2 = 0;
    size_t idx2 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max2) { max2 = rel; idx2 = i; }
    }
    printf("poly2 worst at [%zu]: input=%.6f out=%.6e ref=%.6e rel=%.4f\n",
           idx2, input[idx2], output[idx2], ref[idx2], max2);
    
    // Check poly2 index
    printf("\nDetailed check for idx2=%zu:\n", idx2);
    printf("  input[%zu] = %.10f\n", idx2, input[idx2]);
    printf("  ref[%zu]   = %.10e\n", idx2, ref[idx2]);
    printf("  output[%zu] = %.10e\n", idx2, output[idx2]);
    printf("  expf(input[%zu]) = %.10e\n", idx2, expf(input[idx2]));
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
