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
    
    float max2 = 0.0f;  // Explicitly initialize
    size_t idx2 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0.0f) ? err / fabsf(ref[i]) : err;
        if (rel > max2) { 
            max2 = rel; 
            idx2 = i; 
        }
    }
    printf("poly2 worst at [%zu]: max2=%.6e (%.4f)\n", idx2, max2, max2);
    printf("  input=%.6f output=%.6e ref=%.6e\n",
           input[idx2], output[idx2], ref[idx2]);
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
